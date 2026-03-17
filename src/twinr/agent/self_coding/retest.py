"""Run bounded operator-triggered retests for active self-coding skill packages."""

from __future__ import annotations

import asyncio  # AUDIT-FIX(#7): Support best-effort cleanup for async-capable backends.
import contextlib  # AUDIT-FIX(#2): Preserve primary failures while attaching secondary persistence/cleanup notes.
import inspect  # AUDIT-FIX(#7): Detect awaitable cleanup results from backend clients.
import os  # AUDIT-FIX(#9): Honor backward-compatible timeout overrides from .env / environment variables.
import queue  # AUDIT-FIX(#9): Implement bounded backend calls without heavyweight dependencies.
import re  # AUDIT-FIX(#1): Sanitize operator-controlled identifiers and redact sensitive exception fragments.
import threading  # AUDIT-FIX(#12): Serialize same-skill retests inside the single-process runtime.
import time  # AUDIT-FIX(#9): Enforce monotonic deadlines for bounded retests.
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from zoneinfo import ZoneInfo  # AUDIT-FIX(#11): Validate configured timezone names eagerly.

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.self_coding.execution_runs import SelfCodingExecutionRunService
from twinr.agent.self_coding.health import SelfCodingHealthService
from twinr.agent.self_coding.runtime import SelfCodingSkillExecutionService, skill_package_from_document
from twinr.agent.self_coding.store import SelfCodingStore
from twinr.agent.self_coding.status import ArtifactKind, LearnedSkillStatus
from twinr.automations.sensors import build_sensor_trigger

# AUDIT-FIX(#1): Reject path-like / control-character identifiers before they reach the file-backed store.
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
# AUDIT-FIX(#1): Keep operator-visible error text compact and redact common secret/token patterns.
_SECRET_REDACTION_RE = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9_\-]{8,}|bearer\s+[A-Za-z0-9._\-]+|api[_\- ]?key\s*[:=]\s*\S+|"
    r"token\s*[:=]\s*\S+|password\s*[:=]\s*\S+|secret\s*[:=]\s*\S+)"
)
# AUDIT-FIX(#12): Prevent overlapping retests for the same project/skill pair in the single-process deployment model.
_RETEST_LOCKS: dict[tuple[str, str], threading.RLock] = {}
_RETEST_LOCKS_GUARD = threading.Lock()

# AUDIT-FIX(#9): New timeout knobs remain optional and default safely when older .env files omit them.
_BACKEND_TIMEOUT_ENV_KEYS = (
    "SELF_CODING_RETEST_BACKEND_TIMEOUT_SECONDS",
    "TWINR_SELF_CODING_RETEST_BACKEND_TIMEOUT_SECONDS",
)
_MAX_DURATION_ENV_KEYS = (
    "SELF_CODING_RETEST_MAX_DURATION_SECONDS",
    "TWINR_SELF_CODING_RETEST_MAX_DURATION_SECONDS",
)


@dataclass(frozen=True, slots=True)
class SelfCodingSkillRetestResult:
    """Summarize one operator-triggered capture retest."""

    skill_id: str
    version: int
    environment: str
    status: str
    duration_seconds: float
    search_call_count: int
    summary_call_count: int
    spoken_count: int
    delivered: bool
    details: str


@dataclass(slots=True)
class _MemorySpeechOutput:
    """Capture retest speech without speaking it aloud in the operator path."""

    spoken_texts: list[str]

    def __init__(self) -> None:
        self.spoken_texts = []

    def speak(self, text: str) -> None:
        self.spoken_texts.append(str(text))


@dataclass(slots=True)
class _CountingBackendProxy:
    """Count backend calls made during an operator retest."""

    backend: Any
    timeout_seconds: float | None = None  # AUDIT-FIX(#9): Bound backend calls on flaky home Wi-Fi.
    deadline_monotonic: float | None = None  # AUDIT-FIX(#9): Stop new calls after the retest deadline expires.
    search_call_count: int = 0
    summary_call_count: int = 0

    # AUDIT-FIX(#4): Preserve compatibility with skill packages that use backend attributes beyond the two counted calls.
    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)

    # AUDIT-FIX(#9): Fail fast once the overall retest deadline has been exceeded.
    def _check_deadline(self) -> None:
        if self.deadline_monotonic is not None and time.monotonic() > self.deadline_monotonic:
            raise TimeoutError("operator retest exceeded its maximum duration")

    def search_live_info_with_metadata(
        self,
        question: str,
        *,
        conversation=None,
        location_hint=None,
        date_context=None,
        **kwargs: Any,
    ) -> Any:
        self._check_deadline()  # AUDIT-FIX(#9): Avoid starting fresh backend work after the retest timed out.
        self.search_call_count += 1
        return _run_callable_with_timeout(
            self.backend.search_live_info_with_metadata,
            question,
            timeout_seconds=self.timeout_seconds,
            action="backend live search",
            conversation=conversation,
            location_hint=location_hint,
            date_context=date_context,
            **kwargs,
        )

    def respond_with_metadata(
        self,
        prompt: str,
        *,
        instructions=None,
        allow_web_search=None,
        **kwargs: Any,
    ) -> Any:
        self._check_deadline()  # AUDIT-FIX(#9): Avoid starting fresh backend work after the retest timed out.
        self.summary_call_count += 1
        return _run_callable_with_timeout(
            self.backend.respond_with_metadata,
            prompt,
            timeout_seconds=self.timeout_seconds,
            action="backend summary response",
            instructions=instructions,
            allow_web_search=allow_web_search,
            **kwargs,
        )


class _RetestOwner:
    """Expose the bounded owner surface expected by a compiled skill package."""

    def __init__(self, *, backend: _CountingBackendProxy, timezone_name: str) -> None:
        self.config = SimpleNamespace(local_timezone_name=timezone_name)
        self.print_backend = backend
        self.agent_provider = backend
        self.runtime = SimpleNamespace(search_provider_conversation_context=lambda: ())
        self._speech_output = _MemorySpeechOutput()
        self._latest_sensor_observation_facts = {"camera": {"person_visible": True, "count_persons": 1}}
        self._night_mode = False
        self._presence_session_id = 1
        self._ignored_empty_speech_count = 0  # AUDIT-FIX(#6): Track empty utterances without failing the whole retest.

    def _current_presence_session_id(self) -> int | None:
        return self._presence_session_id

    def _speak_automation_text(self, entry: object, text: str) -> None:
        del entry
        spoken = str(text).strip()
        if not spoken:
            self._ignored_empty_speech_count += 1  # AUDIT-FIX(#6): Ignore blank speech instead of crashing the operator path.
            return
        self._speech_output.speak(spoken)

    @property
    def spoken_texts(self) -> tuple[str, ...]:
        return tuple(self._speech_output.spoken_texts)

    @property
    def ignored_empty_speech_count(self) -> int:
        return self._ignored_empty_speech_count


def run_self_coding_skill_retest(
    *,
    project_root: str | Path,
    env_file: str | Path,
    skill_id: str,
    version: int | None = None,
    environment: str = "web",
    backend_factory: Any | None = None,
) -> SelfCodingSkillRetestResult:
    """Execute one capture-only retest for the active or selected skill-package version."""

    # AUDIT-FIX(#1): Validate operator-controlled identifiers before they reach file-backed services and health metadata.
    skill_id = _validate_identifier("skill_id", skill_id)
    environment = _validate_environment(environment)
    version = _validate_version(version)
    root = _resolve_existing_directory("project_root", project_root)  # AUDIT-FIX(#1): Canonicalize the project root early.
    env_path = _resolve_existing_file("env_file", env_file)  # AUDIT-FIX(#1): Canonicalize the .env path early.

    # AUDIT-FIX(#12): Serialize retests for the same project/skill pair to reduce store/run-record races.
    with _get_retest_lock(root, skill_id):
        config = TwinrConfig.from_env(env_path)
        timezone_name = _validate_timezone_name(getattr(config, "local_timezone_name", "UTC"))  # AUDIT-FIX(#11)
        backend_timeout_seconds = _read_positive_float_override(
            config=config,
            env_file=env_path,
            attr_names=("self_coding_retest_backend_timeout_seconds",),
            env_keys=_BACKEND_TIMEOUT_ENV_KEYS,
            default=20.0,
            minimum=0.5,
        )
        retest_max_duration_seconds = _read_positive_float_override(
            config=config,
            env_file=env_path,
            attr_names=("self_coding_retest_max_duration_seconds",),
            env_keys=_MAX_DURATION_ENV_KEYS,
            default=90.0,
            minimum=1.0,
        )

        store = SelfCodingStore.from_project_root(root)
        health = SelfCodingHealthService(store=store)
        execution_runs = SelfCodingExecutionRunService(store=store)

        monotonic_started = time.monotonic()
        deadline_monotonic = monotonic_started + retest_max_duration_seconds
        activation: Any | None = None
        backend: _CountingBackendProxy | None = None
        owner: _RetestOwner | None = None
        run_record: Any | None = None
        delivered = False
        status = "failed"  # AUDIT-FIX(#8): Default to failed until the full retest completes and persists cleanly.
        details = "Retest did not complete."
        current_action = "loading activation"
        primary_error: BaseException | None = None
        secondary_errors: list[str] = []

        try:
            activation = _load_activation(store=store, skill_id=skill_id, version=version)
            if activation.status != LearnedSkillStatus.ACTIVE:
                raise ValueError("operator retest requires an active learned skill version")
            artifact_kind = _artifact_kind(store=store, activation=activation)
            if artifact_kind != str(ArtifactKind.SKILL_PACKAGE.value).strip().casefold():
                raise ValueError("operator retest currently supports skill-package versions only")

            current_action = "creating execution-run record"
            run_record = execution_runs.start_run(
                run_kind="retest",
                skill_id=activation.skill_id,
                version=activation.version,
                metadata={"environment": environment},
            )

            current_action = "loading skill package artifact"
            artifact_text = store.read_text_artifact(str(activation.artifact_id))
            if not str(artifact_text).strip():
                raise ValueError("skill package artifact is empty")
            package = skill_package_from_document(artifact_text)
            runtime = SelfCodingSkillExecutionService(store=store, health_service=None)
            if backend_factory is None:
                from twinr.providers.openai import OpenAIBackend

                backend_factory = OpenAIBackend
            backend_instance = _build_backend(backend_factory=backend_factory, config=config)  # AUDIT-FIX(#4)
            backend = _CountingBackendProxy(
                backend=backend_instance,
                timeout_seconds=backend_timeout_seconds,
                deadline_monotonic=deadline_monotonic,
            )
            owner = _RetestOwner(backend=backend, timezone_name=timezone_name)

            now = datetime.now(UTC)
            scheduled_triggers = tuple(getattr(package, "scheduled_triggers", ()) or ())
            sensor_triggers = tuple(getattr(package, "sensor_triggers", ()) or ())
            if not scheduled_triggers and not sensor_triggers:
                details = "Retest completed; skill package defines no scheduled or sensor triggers."

            for trigger in scheduled_triggers:
                _ensure_not_timed_out(deadline_monotonic)  # AUDIT-FIX(#9): Keep the overall operator path bounded.
                trigger_id = _validated_trigger_id(trigger)
                current_action = f"executing scheduled trigger {trigger_id!r}"  # AUDIT-FIX(#8): Preserve trigger context in failures.
                spoken_before = len(owner.spoken_texts)
                result = runtime.execute_scheduled(
                    owner,
                    skill_id=activation.skill_id,
                    version=activation.version,
                    trigger_id=trigger_id,
                    now=now,
                )
                delivered = delivered or _delivery_observed(
                    result,
                    spoken_before=spoken_before,
                    spoken_after=len(owner.spoken_texts),
                )

            for trigger in sensor_triggers:
                _ensure_not_timed_out(deadline_monotonic)  # AUDIT-FIX(#9): Keep the overall operator path bounded.
                trigger_id = _validated_trigger_id(trigger)
                current_action = f"executing sensor trigger {trigger_id!r}"  # AUDIT-FIX(#8): Preserve trigger context in failures.
                event_name = _sensor_event_name(trigger)
                spoken_before = len(owner.spoken_texts)
                result = runtime.execute_sensor_event(
                    owner,
                    skill_id=activation.skill_id,
                    version=activation.version,
                    trigger_id=trigger_id,
                    event_name=event_name,
                    now=now,
                )
                delivered = delivered or _delivery_observed(
                    result,
                    spoken_before=spoken_before,
                    spoken_after=len(owner.spoken_texts),
                )

            delivered = delivered or bool(owner.spoken_texts)  # AUDIT-FIX(#5): Count captured speech as delivery even if runtime omits flags.
            status = "passed"
            details = _success_details(
                details=details,
                search_call_count=backend.search_call_count,
                summary_call_count=backend.summary_call_count,
                spoken_count=len(owner.spoken_texts),
                delivered=delivered,
                ignored_empty_speech_count=owner.ignored_empty_speech_count,
            )
        except Exception as exc:
            primary_error = exc
            status = "failed"
            details = _failure_details(
                current_action=current_action,
                exc=exc,
                ignored_empty_speech_count=owner.ignored_empty_speech_count if owner is not None else 0,
            )
        finally:
            duration_seconds = max(0.0, time.monotonic() - monotonic_started)  # AUDIT-FIX(#9): Use monotonic time for stable operator diagnostics.

            finish_metadata = {
                "environment": environment,
                "search_call_count": backend.search_call_count if backend is not None else 0,
                "summary_call_count": backend.summary_call_count if backend is not None else 0,
                "spoken_count": len(owner.spoken_texts) if owner is not None else 0,
                "delivered": bool(delivered),
            }

            try:
                _safe_close_backend(backend)
            except Exception as exc:
                secondary_errors.append(f"backend cleanup failed: {_safe_error_text(exc)}")  # AUDIT-FIX(#7)

            if primary_error is None and secondary_errors:
                status = "failed"
                details = _append_secondary_error_details(details, secondary_errors)

            finish_reason = details if status == "failed" else None
            if run_record is not None:
                try:
                    if status == "passed":
                        execution_runs.finish_run(
                            run_record,
                            status="completed",
                            metadata=finish_metadata,
                        )
                    else:
                        execution_runs.finish_run(
                            run_record,
                            status="failed",
                            reason=_safe_reason_text(finish_reason or "operator retest failed"),
                            metadata=finish_metadata,
                        )
                except Exception as exc:
                    secondary_errors.append(f"execution run persistence failed: {_safe_error_text(exc)}")  # AUDIT-FIX(#2)

            if primary_error is None and secondary_errors:
                status = "failed"
                details = _append_secondary_error_details(details, secondary_errors)

            suite_id = activation.skill_id if activation is not None else skill_id
            version_metadata = activation.version if activation is not None else version
            backend_model_label = _backend_model_label(config=config, backend=backend)  # AUDIT-FIX(#10)
            reasoning_effort_label = _reasoning_effort_label(config=config)  # AUDIT-FIX(#10)

            try:
                health.record_live_e2e_status(
                    suite_id=suite_id,
                    environment=environment,
                    status=status,
                    duration_seconds=duration_seconds,
                    model=backend_model_label,
                    reasoning_effort=reasoning_effort_label,
                    details=details,
                    metadata={"version": version_metadata},
                )
            except Exception as exc:
                secondary_errors.append(f"health persistence failed: {_safe_error_text(exc)}")  # AUDIT-FIX(#2)

        if primary_error is not None:
            _raise_with_notes(primary_error, secondary_errors)  # AUDIT-FIX(#2): Re-raise the primary failure without masking it.
        if secondary_errors:
            raise RuntimeError("; ".join(secondary_errors))  # AUDIT-FIX(#2): Surface post-run persistence/cleanup failures.

        return SelfCodingSkillRetestResult(
            skill_id=activation.skill_id if activation is not None else skill_id,
            version=activation.version if activation is not None else (version if version is not None else 0),
            environment=environment,
            status=status,
            duration_seconds=duration_seconds,
            search_call_count=backend.search_call_count if backend is not None else 0,
            summary_call_count=backend.summary_call_count if backend is not None else 0,
            spoken_count=len(owner.spoken_texts) if owner is not None else 0,
            delivered=delivered,
            details=details,
        )


def _load_activation(*, store: SelfCodingStore, skill_id: str, version: int | None) -> Any:
    if version is not None:
        return store.load_activation(skill_id, version=version)
    active_activations = [
        activation
        for activation in store.list_activations(skill_id=skill_id)
        if activation.status == LearnedSkillStatus.ACTIVE
    ]
    if not active_activations:
        raise FileNotFoundError(f"no active activation found for {skill_id!r}")
    if len(active_activations) > 1:
        versions = sorted(
            {
                _normalize_operator_text(getattr(activation, "version", "unknown")) or "unknown"
                for activation in active_activations
            }
        )
        raise RuntimeError(
            f"multiple active activations found for {skill_id!r}: {', '.join(versions)}"
        )  # AUDIT-FIX(#3): Fail closed instead of silently choosing a non-deterministic active version.
    return active_activations[0]


def _artifact_kind(*, store: SelfCodingStore, activation: Any) -> str:
    metadata = dict(getattr(activation, "metadata", {}) or {})
    text = str(metadata.get("artifact_kind") or "").strip()
    if text:
        return text.casefold()  # AUDIT-FIX(#12): Normalize persisted metadata before comparing enum values.
    artifact = store.load_artifact(str(activation.artifact_id))
    kind_value = getattr(getattr(artifact, "kind", None), "value", getattr(artifact, "kind", None))
    kind_text = str(kind_value or "").strip()
    if not kind_text:
        raise ValueError("activation artifact kind is missing")  # AUDIT-FIX(#12): Fail clearly on corrupted artifact metadata.
    return kind_text.casefold()


# AUDIT-FIX(#1): Centralize identifier validation for file-backed skill ids and health metadata fields.
def _validate_identifier(field_name: str, value: object) -> str:
    text = _normalize_operator_text(value)
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    if not _SAFE_IDENTIFIER_RE.fullmatch(text):
        raise ValueError(
            f"{field_name} must match {_SAFE_IDENTIFIER_RE.pattern!r}; got {text!r}"
        )
    return text


# AUDIT-FIX(#1): Keep environment labels safe for metadata/logging while preserving backward-compatible simple values.
def _validate_environment(environment: object) -> str:
    text = _normalize_operator_text(environment)
    if not text:
        raise ValueError("environment must not be empty")
    if len(text) > 64:
        raise ValueError("environment must be at most 64 characters long")
    if not _SAFE_IDENTIFIER_RE.fullmatch(text):
        raise ValueError("environment may only contain letters, numbers, dot, underscore, and hyphen")
    return text


# AUDIT-FIX(#1): Reject bool / negative versions before they reach store lookup code.
def _validate_version(version: int | None) -> int | None:
    if version is None:
        return None
    if isinstance(version, bool) or not isinstance(version, int):
        raise TypeError("version must be an integer or None")
    if version < 1:
        raise ValueError("version must be >= 1")
    return version


# AUDIT-FIX(#1): Resolve the project root once and fail with a clear error instead of a late store exception.
def _resolve_existing_directory(field_name: str, value: str | Path) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(f"{field_name} must point to an existing directory: {resolved}")
    return resolved


# AUDIT-FIX(#1): Resolve the env file once and fail with a clear error instead of a late config exception.
def _resolve_existing_file(field_name: str, value: str | Path) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"{field_name} must point to an existing file: {resolved}")
    return resolved


# AUDIT-FIX(#10): Catch invalid timezone configuration before skill execution uses it for scheduling logic.
def _validate_timezone_name(timezone_name: object) -> str:
    text = _normalize_operator_text(timezone_name)
    if not text:
        raise ValueError("local_timezone_name must not be empty")
    try:
        ZoneInfo(text)
    except Exception as exc:  # pragma: no cover - depends on host tzdata availability.
        raise ValueError(f"invalid local_timezone_name {text!r}") from exc
    return text


# AUDIT-FIX(#11): Reuse one lock per project/skill pair to reduce same-process retest races on the file-backed store.
def _get_retest_lock(project_root: Path, skill_id: str) -> threading.RLock:
    key = (str(project_root), skill_id)
    with _RETEST_LOCKS_GUARD:
        lock = _RETEST_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _RETEST_LOCKS[key] = lock
        return lock


# AUDIT-FIX(#9): Keep timeout overrides backward compatible by checking config attributes, process env, then the .env file directly.
def _read_positive_float_override(
    *,
    config: TwinrConfig,
    env_file: Path,
    attr_names: tuple[str, ...],
    env_keys: tuple[str, ...],
    default: float,
    minimum: float,
) -> float:
    candidates: list[object] = []
    candidates.extend(getattr(config, name, None) for name in attr_names)
    candidates.extend(os.environ.get(name) for name in env_keys)
    candidates.extend(_read_env_file_value(env_file, name) for name in env_keys)
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if not text:
            continue
        try:
            value = float(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid timeout value {candidate!r}") from exc
        if value < minimum:
            raise ValueError(f"timeout value must be >= {minimum}")
        return value
    return default


# AUDIT-FIX(#9): Lightweight .env parsing keeps the fix self-contained and avoids new dependencies on RPi 4.
def _read_env_file_value(env_file: Path, key: str) -> str | None:
    with env_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].lstrip()
            if "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() != key:
                continue
            cleaned = value.strip()
            if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
                cleaned = cleaned[1:-1]
            return cleaned
    return None


# AUDIT-FIX(#9): Bound blocking backend calls with a daemon thread so operator retests cannot hang indefinitely on network stalls.
def _run_callable_with_timeout(
    func: Any,
    *args: Any,
    timeout_seconds: float | None,
    action: str,
    **kwargs: Any,
) -> Any:
    if timeout_seconds is None:
        return func(*args, **kwargs)

    result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def runner() -> None:
        try:
            result_queue.put((True, func(*args, **kwargs)))
        except BaseException as exc:  # pragma: no cover - exercised via caller behavior.
            result_queue.put((False, exc))

    thread = threading.Thread(
        target=runner,
        name=f"self-coding-retest-{action.replace(' ', '-')}",
        daemon=True,
    )
    thread.start()
    try:
        succeeded, payload = result_queue.get(timeout=timeout_seconds)
    except queue.Empty as exc:
        raise TimeoutError(f"{action} timed out after {timeout_seconds:.1f}s") from exc
    if succeeded:
        return payload
    raise payload


# AUDIT-FIX(#9): Stop between triggers once the configured maximum retest duration has elapsed.
def _ensure_not_timed_out(deadline_monotonic: float | None) -> None:
    if deadline_monotonic is not None and time.monotonic() > deadline_monotonic:
        raise TimeoutError("operator retest exceeded its maximum duration")


# AUDIT-FIX(#4): Support both factories and already-instantiated backends to preserve testing/backward compatibility.
def _build_backend(*, backend_factory: Any, config: TwinrConfig) -> Any:
    if hasattr(backend_factory, "search_live_info_with_metadata") and hasattr(backend_factory, "respond_with_metadata"):
        return backend_factory
    if not callable(backend_factory):
        raise TypeError("backend_factory must be a callable or a backend instance")
    try:
        return backend_factory(config=config)
    except TypeError as exc:
        try:
            return backend_factory(config)
        except TypeError:
            raise TypeError("backend_factory must accept the Twinr config as a positional or keyword argument") from exc


# AUDIT-FIX(#5): Interpret the broad set of runtime return shapes without assuming a dict result.
def _delivery_observed(result: Any, *, spoken_before: int, spoken_after: int) -> bool:
    if spoken_after > spoken_before:
        return True
    if result is None:
        return False
    if isinstance(result, Mapping):
        return bool(result.get("delivered"))
    delivered = getattr(result, "delivered", None)
    if delivered is not None:
        return bool(delivered)
    return False


# AUDIT-FIX(#8): Keep failure messages actionable by naming the current stage/trigger and redacting sensitive fragments.
def _failure_details(*, current_action: str, exc: BaseException, ignored_empty_speech_count: int) -> str:
    detail = f"Retest failed during {current_action}: {_safe_error_text(exc)}"
    if ignored_empty_speech_count:
        detail = f"{detail} Ignored {ignored_empty_speech_count} empty speech item(s)."
    return detail


# AUDIT-FIX(#5): Report meaningful outcome metrics instead of only raw counters.
def _success_details(
    *,
    details: str,
    search_call_count: int,
    summary_call_count: int,
    spoken_count: int,
    delivered: bool,
    ignored_empty_speech_count: int,
) -> str:
    parts = []
    if details:
        parts.append(details)
    if "Retest completed" not in details:
        parts.append(
            f"Retest completed with {search_call_count} searches, {summary_call_count} summaries, "
            f"and {spoken_count} captured speeches."
        )
    if not delivered:
        parts.append("No delivery signal was observed.")
    if ignored_empty_speech_count:
        parts.append(f"Ignored {ignored_empty_speech_count} empty speech item(s).")
    return " ".join(part.strip() for part in parts if part and part.strip())


# AUDIT-FIX(#2): Keep secondary persistence/cleanup failures visible without duplicating them in the operator detail string.
def _append_secondary_error_details(details: str, secondary_errors: list[str]) -> str:
    suffix = f"Secondary error: {'; '.join(secondary_errors)}"
    if not details:
        return suffix
    if suffix in details:
        return details
    return f"{details} {suffix}".strip()


# AUDIT-FIX(#8): Trigger ids are needed for deterministic diagnostics when a compiled trigger fails.
def _validated_trigger_id(trigger: Any) -> str:
    trigger_id = _normalize_operator_text(getattr(trigger, "trigger_id", ""))
    if not trigger_id:
        raise ValueError("skill package trigger is missing trigger_id")
    return trigger_id


# AUDIT-FIX(#8): Raise a clear validation error when sensor triggers cannot provide a dispatchable event name.
def _sensor_event_name(trigger: Any) -> str:
    sensor_trigger_kind = _normalize_operator_text(getattr(trigger, "sensor_trigger_kind", ""))
    if not sensor_trigger_kind:
        raise ValueError("sensor trigger is missing sensor_trigger_kind")
    built_trigger = build_sensor_trigger(
        sensor_trigger_kind,
        hold_seconds=getattr(trigger, "hold_seconds", None),
        cooldown_seconds=getattr(trigger, "cooldown_seconds", None),
    )
    event_name = _normalize_operator_text(getattr(built_trigger, "event_name", ""))
    if not event_name:
        raise ValueError(f"sensor trigger {sensor_trigger_kind!r} did not expose an event_name")
    return event_name


# AUDIT-FIX(#13): Avoid hard-coded health metadata that drifts away from the real backend configuration.
def _backend_model_label(*, config: TwinrConfig, backend: _CountingBackendProxy | None) -> str:
    for attr_name in ("model", "openai_model", "llm_model", "reasoning_model"):
        value = getattr(config, attr_name, None)
        if value:
            return _normalize_operator_text(value)
    if backend is not None:
        return type(backend.backend).__name__
    return "unknown"


# AUDIT-FIX(#13): Avoid hard-coded reasoning metadata that can become false after backend/config changes.
def _reasoning_effort_label(*, config: TwinrConfig) -> str:
    for attr_name in ("reasoning_effort", "openai_reasoning_effort"):
        value = getattr(config, attr_name, None)
        if value:
            return _normalize_operator_text(value)
    return "unknown"


# AUDIT-FIX(#7): Close backend resources best-effort to avoid leaking sessions/sockets across retests.
def _safe_close_backend(backend: _CountingBackendProxy | None) -> None:
    if backend is None:
        return
    for method_name in ("close", "aclose"):
        method = getattr(backend.backend, method_name, None)
        if not callable(method):
            continue
        result = method()
        if inspect.isawaitable(result):
            _run_awaitable_with_timeout(result, timeout_seconds=5.0, action=f"backend {method_name}")
        return


# AUDIT-FIX(#7): Run async cleanup safely from sync code, even when the caller already sits inside an event loop thread.
def _run_awaitable_with_timeout(awaitable: Any, *, timeout_seconds: float, action: str) -> None:
    def runner() -> None:
        asyncio.run(awaitable)

    _run_callable_with_timeout(runner, timeout_seconds=timeout_seconds, action=action)


# AUDIT-FIX(#2): Attach persistence/cleanup notes to the primary exception instead of masking it.
def _raise_with_notes(exc: BaseException, notes: list[str]) -> None:
    for note in notes:
        with contextlib.suppress(Exception):
            exc.add_note(note)
    raise exc


# AUDIT-FIX(#1): Keep operator-facing error details readable, single-line, and redacted.
def _safe_error_text(exc: BaseException, *, limit: int = 240) -> str:
    raw = _SECRET_REDACTION_RE.sub("[redacted]", " ".join(str(exc).split()))
    if not raw:
        return type(exc).__name__
    if len(raw) > limit:
        raw = f"{raw[: limit - 1].rstrip()}…"
    return f"{type(exc).__name__}: {raw}"


# AUDIT-FIX(#1): Keep persisted failure reasons compact because stores/health dashboards often expect short strings.
def _safe_reason_text(reason: str, *, limit: int = 160) -> str:
    text = _normalize_operator_text(reason)
    if len(text) > limit:
        return f"{text[: limit - 1].rstrip()}…"
    return text or "operator retest failed"


# AUDIT-FIX(#1): Normalize arbitrary text inputs by removing control whitespace and trimming edges.
def _normalize_operator_text(value: object) -> str:
    return " ".join(str(value).split())


__all__ = [
    "SelfCodingSkillRetestResult",
    "run_self_coding_skill_retest",
]