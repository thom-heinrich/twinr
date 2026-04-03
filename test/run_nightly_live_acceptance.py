"""Run a real-call nightly acceptance against OpenAI, RSS, and ChonkyDB.

Purpose
-------
Seed one isolated remote namespace with enough reminders, durable memory,
world-intelligence subscriptions, and explicit personality-learning signals so
the nightly orchestrator must materialize visible changes. This is the bounded
operator proof that nightly work does more than build a digest: it should also
refresh world awareness, produce continuity shifts, and commit accepted
personality deltas.

Usage
-----
Command-line invocation examples::

    PYTHONPATH=src python3 test/run_nightly_live_acceptance.py --env-file .env
    PYTHONPATH=src python3 test/run_nightly_live_acceptance.py --env-file /twinr/.env --probe-id pi_nightly

Outputs
-------
- JSON summary written to stdout.
- Per-run artifacts under ``artifacts/acceptance/nightly_live_acceptance/<probe_id>/``.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import tempfile

from twinr.agent.base_agent import TwinrConfig
from twinr.agent.personality.intelligence.models import WorldIntelligenceConfigRequest
from twinr.agent.personality.models import InteractionSignal
from twinr.agent.personality.signals import (
    STYLE_INITIATIVE_DELTA_TARGET,
    STYLE_VERBOSITY_DELTA_TARGET,
)
from twinr.agent.workflows.realtime_runtime.nightly import TwinrNightlyOrchestrator
from twinr.memory.longterm.core.models import LongTermMemoryObjectV1, LongTermSourceRefV1
from twinr.memory.longterm.evaluation.live_midterm_acceptance import (
    _build_isolated_config,
    _close_openai_backend,
    _configure_openai_backend_client,
    _normalize_base_project_root,
    _shutdown_service,
)
from twinr.memory.longterm.runtime.service import LongTermMemoryService
from twinr.memory.reminders import ReminderStore
from twinr.providers.openai import OpenAIBackend

_MODEL_TIMEOUT_S = 45.0
_MODEL_MAX_RETRIES = 1
_DEFAULT_FEED_URLS = (
    "https://www.tagesschau.de/index~rss2.xml",
    "https://rss.dw.com/rdf/rss-en-world",
)


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in stable ISO-8601 form."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_namespace_suffix(value: str) -> str:
    """Normalize one probe id into a ChonkyDB-safe namespace suffix."""

    safe_chars: list[str] = []
    for char in str(value or "").lower():
        safe_chars.append(char if char.isalnum() else "_")
    normalized = "".join(safe_chars).strip("_")
    return normalized or "nightly_live"


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one JSON payload atomically to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_CLOEXEC", 0), 0o600)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp_path, path)


def _acceptance_source(event_id: str) -> LongTermSourceRefV1:
    """Build one canonical synthetic source reference."""

    return LongTermSourceRefV1(
        source_type="synthetic_acceptance",
        event_ids=(event_id,),
        speaker="user",
        modality="text",
    )


@dataclass(slots=True)
class _AcceptanceRuntime:
    """Expose the tiny runtime surface that nightly orchestration needs."""

    long_term_memory: LongTermMemoryService
    reminder_store: ReminderStore

    def peek_due_reminders(self, *, limit: int = 1):
        return self.reminder_store.peek_due(limit=limit)


@dataclass(frozen=True, slots=True)
class LiveNightlyAcceptanceResult:
    """Capture one complete live nightly acceptance run."""

    probe_id: str
    status: str
    started_at: str
    finished_at: str
    env_path: str
    base_project_root: str
    runtime_namespace: str
    target_local_day: str | None = None
    artifact_dir: str | None = None
    nightly_state_path: str | None = None
    nightly_digest_path: str | None = None
    nightly_summary_path: str | None = None
    result_action: str | None = None
    result_reason: str | None = None
    last_status: str | None = None
    world_refresh_status: str | None = None
    world_awareness_thread_count: int = 0
    accepted_personality_delta_count: int = 0
    new_insights: tuple[str, ...] = ()
    continuity_shifts: tuple[str, ...] = ()
    headline_lines: tuple[str, ...] = ()
    weather_summary: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe mapping."""

        payload = asdict(self)
        payload["new_insights"] = list(self.new_insights)
        payload["continuity_shifts"] = list(self.continuity_shifts)
        payload["headline_lines"] = list(self.headline_lines)
        return payload


def _seed_reflection_memory(service: LongTermMemoryService) -> None:
    """Seed one relationship thread that the reflector can summarize."""

    relationship = LongTermMemoryObjectV1(
        memory_id="fact:janina_wife_live",
        kind="fact",
        summary="Janina is the user's wife.",
        source=_acceptance_source("janina:wife"),
        status="active",
        confidence=0.98,
        slot_key="relationship:user:main:wife",
        value_key="person:janina",
        attributes={
            "person_ref": "person:janina",
            "person_name": "Janina",
            "relation": "wife",
            "fact_type": "relationship",
            "support_count": 2,
        },
    )
    appointment = LongTermMemoryObjectV1(
        memory_id="event:janina_eye_laser_live",
        kind="event",
        summary="Janina has eye laser treatment at the eye doctor on 2026-04-04.",
        source=_acceptance_source("janina:eye_laser"),
        status="active",
        confidence=0.93,
        slot_key="event:person:janina:eye_laser_treatment:2026-04-04",
        value_key="event:janina_eye_laser_2026_04_04",
        valid_from="2026-04-04",
        valid_to="2026-04-04",
        sensitivity="sensitive",
        attributes={
            "person_ref": "person:janina",
            "person_name": "Janina",
            "memory_domain": "appointment",
            "event_domain": "appointment",
            "action": "eye laser treatment",
            "place": "the eye doctor",
            "support_count": 1,
        },
    )
    service.object_store.commit_active_delta(object_upserts=(relationship, appointment))


def _seed_personality_signals(service: LongTermMemoryService) -> None:
    """Seed explicit signals that should survive policy gating as deltas."""

    learning = service.personality_learning
    if learning is None:
        raise RuntimeError("personality learning is not configured")
    learning.background_loop.enqueue_interaction_signal(
        InteractionSignal(
            signal_id="sig:initiative:nightly_live",
            signal_kind="initiative_preference",
            target="initiative_preference",
            summary="The user explicitly wants Twinr to take a bit more initiative with helpful prompts.",
            confidence=0.92,
            impact=0.7,
            evidence_count=1,
            source_event_ids=("turn:init:nightly_live",),
            delta_target=STYLE_INITIATIVE_DELTA_TARGET,
            delta_value=0.11,
            delta_summary="The user explicitly asked Twinr to take a bit more initiative.",
            explicit_user_requested=True,
            provenance="nightly_live_acceptance",
        )
    )
    learning.background_loop.enqueue_interaction_signal(
        InteractionSignal(
            signal_id="sig:verbosity:nightly_live",
            signal_kind="verbosity_preference",
            target="verbosity_preference",
            summary="The user explicitly wants slightly shorter but still warm answers.",
            confidence=0.91,
            impact=0.6,
            evidence_count=1,
            source_event_ids=("turn:verbosity:nightly_live",),
            delta_target=STYLE_VERBOSITY_DELTA_TARGET,
            delta_value=-0.08,
            delta_summary="The user explicitly asked for slightly shorter answers.",
            explicit_user_requested=True,
            provenance="nightly_live_acceptance",
        )
    )


def _seed_world_intelligence(
    service: LongTermMemoryService,
    *,
    backend: OpenAIBackend,
) -> None:
    """Subscribe one fresh world-intelligence set without pre-refreshing it."""

    learning = service.personality_learning
    if learning is None:
        raise RuntimeError("personality learning is not configured")
    learning.configure_world_intelligence(
        request=WorldIntelligenceConfigRequest(
            action="subscribe",
            label="Berlin World/Local Mix",
            location_hint="Berlin, DE",
            region="Berlin",
            topics=("Berlin", "community", "world"),
            feed_urls=_DEFAULT_FEED_URLS,
            scope="local",
            priority=0.9,
            refresh_interval_hours=1,
            refresh_after_change=False,
            created_by="nightly_live_acceptance",
        ),
        search_backend=backend,
    )


def _seed_reminders(
    store: ReminderStore,
    *,
    target_local_day: datetime,
) -> None:
    """Seed several reminders for the prepared target day."""

    target = target_local_day.replace(hour=7, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    store.schedule(due_at=(target + timedelta(minutes=10)).isoformat(), summary="Hausarzt anrufen")
    store.schedule(due_at=(target + timedelta(minutes=30)).isoformat(), summary="Medikamente sortieren")
    store.schedule(due_at=(target + timedelta(minutes=90)).isoformat(), summary="Janina zur Augenklinik begleiten")


def _copy_if_present(source: Path, destination: Path) -> str | None:
    """Copy one artifact into the acceptance dir when it exists."""

    if not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


def run_live_nightly_acceptance(
    *,
    env_path: str,
    probe_id: str | None = None,
) -> LiveNightlyAcceptanceResult:
    """Run the bounded live nightly acceptance and return its result."""

    resolved_env_path = Path(env_path).expanduser().resolve(strict=False)
    started_at = _utc_now_iso()
    effective_probe_id = " ".join(
        str(probe_id or f"nightly_live_{started_at.replace(':', '').replace('-', '')}").split()
    ).strip()
    base_config = TwinrConfig.from_env(resolved_env_path)
    base_project_root = _normalize_base_project_root(resolved_env_path, base_config)
    runtime_namespace = f"twinr_nightly_live_{_safe_namespace_suffix(effective_probe_id)}"
    artifact_dir = (
        base_project_root
        / "artifacts"
        / "acceptance"
        / "nightly_live_acceptance"
        / effective_probe_id
    )

    service: LongTermMemoryService | None = None
    backend: OpenAIBackend | None = None
    result = LiveNightlyAcceptanceResult(
        probe_id=effective_probe_id,
        status="running",
        started_at=started_at,
        finished_at=started_at,
        env_path=str(resolved_env_path),
        base_project_root=str(base_project_root),
        runtime_namespace=runtime_namespace,
        artifact_dir=str(artifact_dir),
    )

    try:
        if not base_config.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for live nightly acceptance.")
        if not base_config.chonkydb_base_url or not base_config.chonkydb_api_key:
            raise RuntimeError("ChonkyDB credentials are required for live nightly acceptance.")

        with tempfile.TemporaryDirectory(prefix=f"{effective_probe_id}_nightly_") as temp_dir:
            runtime_root = Path(temp_dir).resolve(strict=False)
            config = _build_isolated_config(
                base_config=base_config,
                base_project_root=base_project_root,
                runtime_root=runtime_root,
                remote_namespace=runtime_namespace,
                background_store_turns=False,
            )
            config = replace(
                config,
                openai_web_search_city="Berlin",
                openai_web_search_region="Berlin",
                openai_web_search_country="DE",
                nightly_orchestration_after_local="00:30",
                display_reserve_bus_nightly_after_local="00:30",
                display_reserve_bus_refresh_after_local="05:30",
            )

            service = LongTermMemoryService.from_config(config)
            service.ensure_remote_ready()
            backend = OpenAIBackend(config)
            _configure_openai_backend_client(
                backend,
                timeout_s=_MODEL_TIMEOUT_S,
                max_retries=_MODEL_MAX_RETRIES,
            )

            _seed_reflection_memory(service)
            _seed_personality_signals(service)
            _seed_world_intelligence(service, backend=backend)

            local_now = datetime.now().astimezone().replace(hour=1, minute=15, second=0, microsecond=0) + timedelta(days=1)
            reminder_store = ReminderStore(config.reminder_store_path, timezone_name=config.local_timezone_name)
            _seed_reminders(reminder_store, target_local_day=local_now)

            runtime = _AcceptanceRuntime(long_term_memory=service, reminder_store=reminder_store)
            orchestrator = TwinrNightlyOrchestrator(
                config=config,
                runtime=runtime,
                text_backend=backend,
                search_backend=backend,
                print_backend=backend,
                remote_ready=lambda: True,
                background_allowed=lambda: True,
            )
            orchestration_result = orchestrator.maybe_run(local_now=local_now)
            state = orchestrator.state_store.load()
            digest = orchestrator.digest_store.load()
            summary = orchestrator.summary_store.load()

            if state is None or digest is None or summary is None:
                raise RuntimeError("Nightly acceptance did not write all expected artifacts.")
            if orchestration_result.action != "prepared":
                raise RuntimeError(
                    f"Nightly acceptance expected action=prepared, got {orchestration_result.action}:{orchestration_result.reason}."
                )
            if summary.world_refresh_status != "refreshed":
                raise RuntimeError(f"Nightly acceptance expected world_refresh_status=refreshed, got {summary.world_refresh_status}.")
            if summary.world_awareness_thread_count <= 0:
                raise RuntimeError("Nightly acceptance expected world awareness threads > 0.")
            if summary.accepted_personality_delta_count <= 0:
                raise RuntimeError("Nightly acceptance expected accepted personality deltas > 0.")
            if not summary.new_insights:
                raise RuntimeError("Nightly acceptance expected non-empty new insights.")
            if not summary.continuity_shifts:
                raise RuntimeError("Nightly acceptance expected non-empty continuity shifts.")

            state_copy = _copy_if_present(
                orchestrator.state_store.path,
                artifact_dir / "nightly_run_state.json",
            )
            digest_copy = _copy_if_present(
                orchestrator.digest_store.path,
                artifact_dir / "nightly_prepared_digest.json",
            )
            summary_copy = _copy_if_present(
                orchestrator.summary_store.path,
                artifact_dir / "nightly_consolidation_summary.json",
            )

            result = replace(
                result,
                status="ok",
                finished_at=_utc_now_iso(),
                target_local_day=digest.target_local_day,
                nightly_state_path=state_copy,
                nightly_digest_path=digest_copy,
                nightly_summary_path=summary_copy,
                result_action=orchestration_result.action,
                result_reason=orchestration_result.reason,
                last_status=state.last_status,
                world_refresh_status=summary.world_refresh_status,
                world_awareness_thread_count=summary.world_awareness_thread_count,
                accepted_personality_delta_count=summary.accepted_personality_delta_count,
                new_insights=summary.new_insights,
                continuity_shifts=summary.continuity_shifts,
                headline_lines=digest.headline_lines,
                weather_summary=digest.weather_summary,
            )
            _atomic_write_json(artifact_dir / "nightly_live_acceptance.json", result.to_dict())
    except Exception as exc:
        result = replace(
            result,
            status="failed",
            finished_at=_utc_now_iso(),
            error_message=f"{type(exc).__name__}: {exc}",
        )
        _atomic_write_json(artifact_dir / "nightly_live_acceptance.json", result.to_dict())
    finally:
        _close_openai_backend(backend)
        _shutdown_service(service)

    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Run the live nightly acceptance suite.")
    parser.add_argument("--env-file", default=".env", help="Path to the Twinr env file.")
    parser.add_argument("--probe-id", default=None, help="Optional stable probe id / namespace suffix.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entrypoint and print the structured result JSON."""

    args = _build_arg_parser().parse_args(argv)
    result = run_live_nightly_acceptance(
        env_path=args.env_file,
        probe_id=args.probe_id,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False))
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
