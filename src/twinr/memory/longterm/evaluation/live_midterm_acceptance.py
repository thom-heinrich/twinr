"""Run a live midterm-memory write/read/usage acceptance check.

The acceptance flow exercises the real Twinr/OpenAI/ChonkyDB path end to end:
it asks the live model to answer a seed turn, persists the resulting turn into
an isolated remote namespace, proves the midterm packets exist locally and
remotely, then starts a fresh reader runtime and verifies that a second live
model answer uses the recalled memory. The final result is emitted as JSON and
persisted as an ops artifact that the web debug page can display.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import tempfile
import time
from typing import Iterator

from twinr.agent.base_agent import TwinrConfig
from twinr.agent.base_agent.conversation.language import memory_and_response_contract
from twinr.memory.longterm.evaluation.live_midterm_attest import (
    LiveMidtermAttestResult,
    LiveMidtermSeedTurn,
    write_live_midterm_attest_artifacts,
)
from twinr.memory.longterm.runtime.service import LongTermMemoryService
from twinr.providers.openai import OpenAIBackend
from twinr.text_utils import folded_lookup_text


_LOGGER = logging.getLogger(__name__)
_MIDTERM_LOGGER_NAME = "twinr.memory.longterm.storage.midterm_store"
_REMOTE_STATE_LOGGER_NAME = "twinr.memory.longterm.storage.remote_state"
_FLUSH_TIMEOUT_S = 60.0
_PERSISTENCE_WAIT_TIMEOUT_S = 90.0
_PERSISTENCE_POLL_INTERVAL_S = 2.0
_MODEL_TIMEOUT_S = 45.0
_MODEL_MAX_RETRIES = 1


@dataclass(frozen=True, slots=True)
class _AcceptanceScenario:
    """Define the fixed live prompts used by the midterm attestation."""

    seed_prompts: tuple[str, ...]
    follow_up_query: str
    expected_answer_terms: tuple[str, ...]


_DEFAULT_SCENARIO = _AcceptanceScenario(
    seed_prompts=(
        (
            "Bitte merke dir einfach Folgendes für später: "
            "Meine Tochter Lea bringt mir heute Abend um 19 Uhr eine Thermoskanne "
            "mit selbstgemachter Linsensuppe vorbei."
        ),
    ),
    follow_up_query="Was bringt mir Lea heute Abend um 19 Uhr vorbei?",
    expected_answer_terms=("thermoskanne", "linsensuppe"),
)


class _ListLogHandler(logging.Handler):
    """Collect log records into an in-memory list."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.messages.append(self.format(record))
        except Exception:  # pragma: no cover - defensive logging isolation
            self.messages.append(str(record.getMessage()))


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in stable ISO-8601 form."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_base_project_root(env_path: Path, config: TwinrConfig) -> Path:
    """Resolve the authoritative project root for an env-backed Twinr config."""

    configured = Path(str(getattr(config, "project_root", ".") or ".")).expanduser()
    if configured.is_absolute():
        return configured.resolve(strict=False)
    return (env_path.parent.resolve(strict=False) / configured).resolve(strict=False)


def _resolve_personality_dir(base_project_root: Path, config: TwinrConfig) -> Path:
    """Resolve the real personality directory that live acceptance must reuse."""

    raw_personality_dir = Path(str(getattr(config, "personality_dir", "personality") or "personality")).expanduser()
    candidate = raw_personality_dir if raw_personality_dir.is_absolute() else (base_project_root / raw_personality_dir)
    resolved = candidate.resolve(strict=False)
    if not resolved.is_dir():
        raise FileNotFoundError(f"Personality directory not found for live midterm acceptance: {resolved}")
    return resolved


def _build_isolated_config(
    *,
    base_config: TwinrConfig,
    base_project_root: Path,
    runtime_root: Path,
    remote_namespace: str,
    background_store_turns: bool,
) -> TwinrConfig:
    """Build one isolated runtime config that still points at live providers."""

    source_personality_dir = _resolve_personality_dir(base_project_root, base_config)
    personality_dir = runtime_root / "personality"
    state_dir = runtime_root / "state"
    shutil.copytree(source_personality_dir, personality_dir, dirs_exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    return replace(
        base_config,
        project_root=str(runtime_root),
        personality_dir="personality",
        memory_markdown_path=str(state_dir / "MEMORY.md"),
        runtime_state_path=str(runtime_root / "runtime-state.json"),
        long_term_memory_enabled=True,
        long_term_memory_mode="remote_primary",
        long_term_memory_remote_required=True,
        long_term_memory_path=str(state_dir / "chonkydb"),
        long_term_memory_remote_namespace=remote_namespace,
        long_term_memory_background_store_turns=background_store_turns,
        openai_enable_web_search=False,
        restore_runtime_state_on_startup=False,
        **_isolated_runtime_state_overrides(state_dir=state_dir),
    )


def _isolated_runtime_state_overrides(*, state_dir: Path) -> dict[str, str]:
    """Return runtime-owned state files that must stay inside one isolated root."""

    return {
        "reminder_store_path": str(state_dir / "reminders.json"),
        "automation_store_path": str(state_dir / "automations.json"),
        "voice_profile_store_path": str(state_dir / "voice_profile.json"),
        "adaptive_timing_store_path": str(state_dir / "adaptive_timing.json"),
    }


def _normalize_lookup_text(text: object | None) -> str:
    """Fold and whitespace-normalize text for deterministic membership checks."""

    return " ".join(folded_lookup_text(str(text or "")).split())


def _safe_namespace_suffix(value: str) -> str:
    """Normalize a probe id into a ChonkyDB-safe namespace suffix."""

    safe_chars: list[str] = []
    for char in str(value or "").lower():
        if char.isalnum():
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    normalized = "".join(safe_chars).strip("_")
    return normalized or "midterm_live"


def _matched_terms(text: str, expected_terms: tuple[str, ...]) -> tuple[str, ...]:
    """Return the expected lookup terms present in the supplied text."""

    normalized = _normalize_lookup_text(text)
    matches: list[str] = []
    for item in expected_terms:
        normalized_item = _normalize_lookup_text(item)
        if normalized_item and normalized_item in normalized:
            matches.append(item)
    return tuple(matches)


def _packet_ids_from_payload(payload: object | None) -> tuple[str, ...]:
    """Extract stored midterm packet ids from a raw payload mapping."""

    if not isinstance(payload, dict):
        return ()
    packets = payload.get("packets")
    if not isinstance(packets, list):
        return ()
    packet_ids: list[str] = []
    for item in packets:
        if not isinstance(item, dict):
            continue
        packet_id = " ".join(str(item.get("packet_id", "") or "").split()).strip()
        if packet_id:
            packet_ids.append(packet_id)
    return tuple(packet_ids)


def _await_midterm_persistence(
    service: LongTermMemoryService,
    *,
    timeout_s: float,
    poll_interval_s: float,
) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
    """Wait until midterm packets exist both locally and in remote storage."""

    deadline = time.monotonic() + max(0.1, float(timeout_s))
    writer_packet_ids: tuple[str, ...] = ()
    remote_packet_ids: tuple[str, ...] = ()
    while True:
        remaining_s = max(0.1, deadline - time.monotonic())
        flush_finished = bool(service.flush(timeout_s=min(_FLUSH_TIMEOUT_S, remaining_s)))
        writer_packet_ids = tuple(packet.packet_id for packet in service.midterm_store.load_packets())
        remote_packet_ids = service.midterm_store.remote_current_packet_ids()
        if writer_packet_ids and remote_packet_ids == writer_packet_ids:
            return True, writer_packet_ids, remote_packet_ids
        if flush_finished and writer_packet_ids and remote_packet_ids:
            return True, writer_packet_ids, remote_packet_ids
        if time.monotonic() >= deadline:
            return False, writer_packet_ids, remote_packet_ids
        time.sleep(min(max(0.1, poll_interval_s), max(0.1, deadline - time.monotonic())))


def _capture_system_messages(conversation: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    """Return only system messages from a provider conversation tuple."""

    return tuple(str(content) for role, content in conversation if str(role) == "system")


def _midterm_context_present(system_messages: tuple[str, ...]) -> bool:
    """Return whether the rendered provider context includes midterm memory."""

    return any("twinr_long_term_midterm_context_v1" in message for message in system_messages)


def _classify_path_warning(log_text: str | None) -> tuple[str | None, str | None]:
    """Normalize the last local-path warning into a stable operator class."""

    text = " ".join(str(log_text or "").split()).strip()
    if not text:
        return None, None
    if "outside the configured Twinr memory root" in text:
        return "outside_root_local_fallback_skipped", text
    return "other_path_warning", text


def _result_packet_sets_match(result: LiveMidtermAttestResult) -> bool:
    """Return whether writer, remote, and fresh-reader packet ids all agree."""

    writer = tuple(result.writer_packet_ids)
    remote = tuple(result.remote_packet_ids)
    reader = tuple(result.fresh_reader_packet_ids)
    if not writer or not remote or not reader:
        return False
    return writer == remote == reader


@contextmanager
def _capture_warning_logs(logger_name: str) -> Iterator[_ListLogHandler]:
    """Temporarily collect warning-level log messages from one logger."""

    logger = logging.getLogger(logger_name)
    handler = _ListLogHandler()
    previous_level = logger.level
    previous_propagate = logger.propagate
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


def _configure_openai_backend_client(
    backend: OpenAIBackend,
    *,
    timeout_s: float,
    max_retries: int,
) -> None:
    """Best-effort replace the backend client with bounded request options."""

    client = getattr(backend, "_client", None)
    with_options = getattr(client, "with_options", None)
    if client is None or not callable(with_options):
        return
    try:
        configured = with_options(timeout=timeout_s, max_retries=max_retries)
    except TypeError:
        try:
            configured = with_options(timeout=timeout_s)
        except TypeError:
            return
    try:
        setattr(backend, "_client", configured)
    except Exception:
        _LOGGER.debug("Could not replace OpenAI backend client for live midterm acceptance.", exc_info=True)


def _close_openai_backend(backend: OpenAIBackend | None) -> None:
    """Close one backend client best-effort after an acceptance run."""

    if backend is None:
        return
    client = getattr(backend, "_client", None)
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            _LOGGER.warning("OpenAI backend close failed during live midterm acceptance cleanup.", exc_info=True)


def _shutdown_service(service: LongTermMemoryService | None) -> None:
    """Shut down one temporary long-term service without masking prior failures."""

    if service is None:
        return
    try:
        service.shutdown(timeout_s=5.0)
    except Exception:
        _LOGGER.warning("Long-term service shutdown failed during live midterm acceptance cleanup.", exc_info=True)
    retriever = getattr(service, "retriever", None)
    subtext_builder = getattr(retriever, "subtext_builder", None)
    if subtext_builder is not None:
        shutdown = getattr(subtext_builder, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown(wait=True)
            except Exception:
                _LOGGER.warning(
                    "Long-term subtext builder shutdown wait failed during live midterm acceptance cleanup.",
                    exc_info=True,
                )
    for attr_name, label in (
        ("prepared_context_front", "prepared context front"),
        ("provider_answer_front", "provider answer front"),
    ):
        front = getattr(service, attr_name, None)
        shutdown = getattr(front, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown(wait=True)
            except Exception:
                _LOGGER.warning(
                    "Long-term %s shutdown wait failed during live midterm acceptance cleanup.",
                    label,
                    exc_info=True,
                )


def _run_seed_turn(
    *,
    service: LongTermMemoryService,
    config: TwinrConfig,
    backend: OpenAIBackend,
    prompt: str,
) -> LiveMidtermSeedTurn:
    """Ask the live model for one seed reply and enqueue that turn for memory."""

    response = backend.respond_with_metadata(
        prompt,
        conversation=_provider_conversation(service=service, config=config, query_text=prompt),
        allow_web_search=False,
    )
    response_text = str(getattr(response, "text", "") or "").strip()
    if not response_text:
        raise RuntimeError("Live seed turn returned an empty assistant response.")
    service.enqueue_conversation_turn(
        transcript=prompt,
        response=response_text,
        source="live_midterm_acceptance",
    )
    return LiveMidtermSeedTurn(
        prompt=prompt,
        response_text=response_text,
        model=getattr(response, "model", None),
        request_id=getattr(response, "request_id", None),
        response_id=getattr(response, "response_id", None),
    )


def _provider_conversation(
    *,
    service: LongTermMemoryService,
    config: TwinrConfig,
    query_text: str,
) -> tuple[tuple[str, str], ...]:
    """Build the provider conversation tuple used for live attestation turns."""

    messages: list[tuple[str, str]] = []
    try:
        contract = memory_and_response_contract(config.openai_realtime_language)
    except Exception:
        contract = None
    if contract:
        messages.append(("system", contract))
    context = service.build_provider_context(query_text)
    for item in context.system_messages():
        text = str(item or "").strip()
        if text:
            messages.append(("system", text))
    return tuple(messages)


def _assert_ready_result(result: LiveMidtermAttestResult) -> None:
    """Raise when the attestation result does not prove the required chain."""

    if not result.flush_ok:
        raise RuntimeError("Long-term writer flush did not complete successfully.")
    if not _result_packet_sets_match(result):
        raise RuntimeError(
            "Writer, remote, and fresh-reader midterm packet ids do not match after the live run."
        )
    if not result.midterm_context_present:
        raise RuntimeError("Fresh reader did not render midterm memory into provider context.")
    missing_terms = tuple(
        term for term in result.expected_answer_terms if term not in set(result.matched_answer_terms)
    )
    if missing_terms:
        raise RuntimeError(
            "Fresh reader LLM answer did not use the recalled memory strongly enough: "
            + ", ".join(missing_terms)
        )
    if result.midterm_warning_logs:
        raise RuntimeError(
            "Midterm store emitted warning logs during the attestation run: "
            + result.midterm_warning_logs
        )
    if result.outside_remote_read_logs:
        raise RuntimeError(
            "Remote read unexpectedly emitted a path warning for the probe path: "
            + result.outside_remote_read_logs
        )
    if result.last_path_warning_class != "outside_root_local_fallback_skipped":
        raise RuntimeError(
            "Local fallback path warning class was not observed or changed unexpectedly."
        )


def run_live_midterm_acceptance(
    *,
    env_path: str | Path = ".env",
    probe_id: str | None = None,
    write_artifacts: bool = True,
) -> LiveMidtermAttestResult:
    """Run the live midterm memory attestation and persist its evidence."""

    resolved_env_path = Path(env_path).expanduser().resolve(strict=False)
    started_at = _utc_now_iso()
    effective_probe_id = " ".join(str(probe_id or f"midterm_live_{started_at.replace(':', '').replace('-', '')}").split()).strip()
    base_config = TwinrConfig.from_env(resolved_env_path)
    base_project_root = _normalize_base_project_root(resolved_env_path, base_config)
    runtime_namespace = f"twinr_midterm_attest_{_safe_namespace_suffix(effective_probe_id)}"

    writer_service: LongTermMemoryService | None = None
    fresh_reader_service: LongTermMemoryService | None = None
    writer_backend: OpenAIBackend | None = None
    fresh_reader_backend: OpenAIBackend | None = None

    result = LiveMidtermAttestResult(
        probe_id=effective_probe_id,
        status="running",
        started_at=started_at,
        finished_at=started_at,
        env_path=str(resolved_env_path),
        base_project_root=str(base_project_root),
        runtime_namespace=runtime_namespace,
        follow_up_query=_DEFAULT_SCENARIO.follow_up_query,
        expected_answer_terms=_DEFAULT_SCENARIO.expected_answer_terms,
    )

    try:
        if not base_config.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for live midterm acceptance.")
        if not base_config.chonkydb_base_url or not base_config.chonkydb_api_key:
            raise RuntimeError("ChonkyDB credentials are required for live midterm acceptance.")

        with ExitStack() as stack:
            writer_temp_dir = stack.enter_context(tempfile.TemporaryDirectory(prefix=f"{effective_probe_id}_writer_"))
            fresh_reader_temp_dir = stack.enter_context(tempfile.TemporaryDirectory(prefix=f"{effective_probe_id}_reader_"))
            writer_root = Path(writer_temp_dir).resolve(strict=False)
            fresh_reader_root = Path(fresh_reader_temp_dir).resolve(strict=False)
            result = replace(
                result,
                writer_root=str(writer_root),
                fresh_reader_root=str(fresh_reader_root),
            )

            writer_config = _build_isolated_config(
                base_config=base_config,
                base_project_root=base_project_root,
                runtime_root=writer_root,
                remote_namespace=runtime_namespace,
                background_store_turns=True,
            )
            fresh_reader_config = _build_isolated_config(
                base_config=base_config,
                base_project_root=base_project_root,
                runtime_root=fresh_reader_root,
                remote_namespace=runtime_namespace,
                background_store_turns=False,
            )

            writer_service = LongTermMemoryService.from_config(writer_config)
            fresh_reader_service = LongTermMemoryService.from_config(fresh_reader_config)
            writer_service.ensure_remote_ready()
            fresh_reader_service.ensure_remote_ready()
            writer_backend = OpenAIBackend(writer_config)
            fresh_reader_backend = OpenAIBackend(fresh_reader_config)
            _configure_openai_backend_client(
                writer_backend,
                timeout_s=_MODEL_TIMEOUT_S,
                max_retries=_MODEL_MAX_RETRIES,
            )
            _configure_openai_backend_client(
                fresh_reader_backend,
                timeout_s=_MODEL_TIMEOUT_S,
                max_retries=_MODEL_MAX_RETRIES,
            )

            with _capture_warning_logs(_MIDTERM_LOGGER_NAME) as midterm_logs:
                seed_turns = tuple(
                    _run_seed_turn(
                        service=writer_service,
                        config=writer_config,
                        backend=writer_backend,
                        prompt=prompt,
                    )
                    for prompt in _DEFAULT_SCENARIO.seed_prompts
                )
                flush_ok, writer_packet_ids, remote_packet_ids = _await_midterm_persistence(
                    writer_service,
                    timeout_s=_PERSISTENCE_WAIT_TIMEOUT_S,
                    poll_interval_s=_PERSISTENCE_POLL_INTERVAL_S,
                )
                remote_state = writer_service.midterm_store.remote_state

                outside_probe_path = writer_root / "outside_midterm_probe.json"
                with _capture_warning_logs(_REMOTE_STATE_LOGGER_NAME) as remote_read_logs:
                    if remote_state is not None:
                        remote_state.load_snapshot(
                            snapshot_kind="midterm",
                            local_path=outside_probe_path,
                        )
                outside_remote_read_logs = "\n".join(remote_read_logs.messages).strip() or None

                with _capture_warning_logs(_REMOTE_STATE_LOGGER_NAME) as fallback_logs:
                    remote_state._load_local_snapshot(
                        outside_probe_path,
                        snapshot_kind="midterm",
                    )
                outside_fallback_logs = "\n".join(fallback_logs.messages).strip() or None
                path_warning_class, path_warning_message = _classify_path_warning(outside_fallback_logs)

                provider_conversation = _provider_conversation(
                    service=fresh_reader_service,
                    config=fresh_reader_config,
                    query_text=_DEFAULT_SCENARIO.follow_up_query,
                )
                provider_system_messages = _capture_system_messages(provider_conversation)
                midterm_context_present = _midterm_context_present(provider_system_messages)
                follow_up_response = fresh_reader_backend.respond_with_metadata(
                    _DEFAULT_SCENARIO.follow_up_query,
                    conversation=provider_conversation,
                    allow_web_search=False,
                )
                follow_up_answer_text = str(getattr(follow_up_response, "text", "") or "").strip()
                matched_terms = _matched_terms(
                    follow_up_answer_text,
                    _DEFAULT_SCENARIO.expected_answer_terms,
                )
                fresh_reader_packets = fresh_reader_service.midterm_store.load_packets()
                fresh_reader_packet_ids = tuple(packet.packet_id for packet in fresh_reader_packets)

                result = replace(
                    result,
                    status="ok",
                    finished_at=_utc_now_iso(),
                    flush_ok=flush_ok,
                    midterm_context_present=midterm_context_present,
                    follow_up_answer_text=follow_up_answer_text,
                    follow_up_model=getattr(follow_up_response, "model", None),
                    follow_up_request_id=getattr(follow_up_response, "request_id", None),
                    follow_up_response_id=getattr(follow_up_response, "response_id", None),
                    matched_answer_terms=matched_terms,
                    writer_packet_ids=writer_packet_ids,
                    remote_packet_ids=remote_packet_ids,
                    fresh_reader_packet_ids=fresh_reader_packet_ids,
                    seed_turns=seed_turns,
                    midterm_warning_logs="\n".join(midterm_logs.messages).strip() or None,
                    outside_remote_read_logs=outside_remote_read_logs,
                    outside_fallback_logs=outside_fallback_logs,
                    last_path_warning_class=path_warning_class,
                    last_path_warning_message=path_warning_message,
                )
                _assert_ready_result(result)
    except Exception as exc:
        result = replace(
            result,
            status="fail",
            finished_at=_utc_now_iso(),
            error_message=f"{type(exc).__name__}: {exc}",
        )
    finally:
        _close_openai_backend(writer_backend)
        _close_openai_backend(fresh_reader_backend)
        _shutdown_service(writer_service)
        _shutdown_service(fresh_reader_service)

    if write_artifacts:
        result = write_live_midterm_attest_artifacts(
            result,
            project_root=base_project_root,
        )
    return result


def main() -> int:
    """Run the CLI entrypoint and print the structured attestation payload."""

    import argparse

    parser = argparse.ArgumentParser(description="Run the live midterm memory acceptance check.")
    parser.add_argument("--env-file", default=".env", help="Twinr env file used for live provider and ChonkyDB credentials.")
    parser.add_argument("--probe-id", default=None, help="Optional explicit probe id used for the isolated namespace and artifact names.")
    parser.add_argument(
        "--skip-artifact-write",
        action="store_true",
        help="Print the result but do not update the rolling ops artifact.",
    )
    args = parser.parse_args()

    result = run_live_midterm_acceptance(
        env_path=args.env_file,
        probe_id=args.probe_id,
        write_artifacts=not args.skip_artifact_write,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.ready else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
