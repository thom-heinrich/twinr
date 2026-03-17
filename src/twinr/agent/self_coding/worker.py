"""Create and run local self-coding compile jobs."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import threading
from typing import Any, Iterator
from uuid import uuid4

from twinr.agent.self_coding.codex_driver import (
    CodexCompileArtifact,
    CodexCompileEvent,
    CodexCompileProgress,
    CodexCompileRequest,
    CodexCompileResult,
    CodexCompileWorkspaceBuilder,
    CodexDriverError,
    CodexExecFallbackDriver,
    CodexSdkDriver,
)
from twinr.agent.self_coding.compiler import (
    build_compile_prompt,
    validate_compile_artifact,
)
from twinr.agent.self_coding.contracts import CompileJobRecord, CompileRunStatusRecord, RequirementsDialogueSession
from twinr.agent.self_coding.status import ArtifactKind, CompileJobStatus, CompileTarget, RequirementsDialogueStatus
from twinr.agent.self_coding.store import SelfCodingStore

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_SECRET_KEY_RE = re.compile(r"(secret|token|password|passwd|api[_-]?key|authorization|cookie)", re.IGNORECASE)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(authorization|api[_ -]?key|token|secret|password|passwd|cookie)\b\s*[:=]\s*([^\s,;]+)"
)
_BEARER_TOKEN_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_VALID_SUFFIX_RE = re.compile(r"^\.[A-Za-z0-9][A-Za-z0-9._+-]{0,15}$")
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")

_MAX_ERROR_MESSAGE_CHARS = 240
_MAX_IDENTIFIER_CHARS = 128
_MAX_METADATA_DEPTH = 4
_MAX_METADATA_ITEMS = 32
_MAX_METADATA_STRING_CHARS = 512
_MAX_MEDIA_TYPE_CHARS = 128
_MAX_PROMPT_FIELD_CHARS = 4096


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


MAX_ARTIFACTS_PER_JOB = _env_int(
    "TWINR_SELF_CODING_MAX_ARTIFACTS_PER_JOB",
    default=12,
    minimum=1,
    maximum=64,
)
MAX_TEXT_ARTIFACT_BYTES = _env_int(
    "TWINR_SELF_CODING_MAX_TEXT_ARTIFACT_BYTES",
    default=2_000_000,
    minimum=8_192,
    maximum=16_000_000,
)
MAX_REVIEW_BYTES = _env_int(
    "TWINR_SELF_CODING_MAX_REVIEW_BYTES",
    default=256_000,
    minimum=4_096,
    maximum=4_000_000,
)
MAX_EVENT_LOG_BYTES = _env_int(
    "TWINR_SELF_CODING_MAX_EVENT_LOG_BYTES",
    default=1_000_000,
    minimum=8_192,
    maximum=8_000_000,
)
MAX_COMPILE_RUN_ATTEMPTS = _env_int(
    "TWINR_SELF_CODING_MAX_COMPILE_RUN_ATTEMPTS",
    default=2,
    minimum=1,
    maximum=4,
)
COMPILE_JOB_HEARTBEAT_SECONDS = _env_int(
    "TWINR_SELF_CODING_COMPILE_HEARTBEAT_SECONDS",
    default=30,
    minimum=5,
    maximum=300,
)
STALE_JOB_SECONDS = _env_int(
    "TWINR_SELF_CODING_STALE_JOB_SECONDS",
    default=900,
    minimum=60,
    maximum=86_400,
)


class _TrackedRLock:
    __slots__ = ("lock", "refcount")

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.refcount = 0


# AUDIT-FIX(#6): Track lock lifetimes so per-job/per-session lock registries do not leak forever in a long-lived Pi process.
_LOCK_REGISTRY_GUARD = threading.Lock()
_SESSION_LOCKS: dict[str, _TrackedRLock] = {}
_JOB_LOCKS: dict[str, _TrackedRLock] = {}


# AUDIT-FIX(#6): Acquire named locks with ref-counted cleanup instead of keeping one RLock per historical job forever.
@contextmanager
def _acquire_named_lock(registry: dict[str, _TrackedRLock], key: str) -> Iterator[threading.RLock]:
    normalized_key = str(key or "").strip() or "__empty__"
    with _LOCK_REGISTRY_GUARD:
        entry = registry.get(normalized_key)
        if entry is None:
            entry = _TrackedRLock()
            registry[normalized_key] = entry
        entry.refcount += 1
    entry.lock.acquire()
    try:
        yield entry.lock
    finally:
        entry.lock.release()
        with _LOCK_REGISTRY_GUARD:
            entry.refcount -= 1
            if entry.refcount <= 0:
                registry.pop(normalized_key, None)


def _normalize_text(value: Any, *, max_chars: int) -> str:
    text = "" if value is None else str(value)
    text = _CONTROL_CHAR_RE.sub(" ", text)
    text = " ".join(text.split())
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def _redact_secretish_text(text: str) -> str:
    redacted = _SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}=<redacted>", text)
    redacted = _BEARER_TOKEN_RE.sub("Bearer <redacted>", redacted)
    return redacted


def _safe_error_message(error: Any) -> str:
    if isinstance(error, BaseException):
        prefix = type(error).__name__
        message = _redact_secretish_text(_normalize_text(str(error), max_chars=_MAX_ERROR_MESSAGE_CHARS))
        if message:
            return _normalize_text(f"{prefix}: {message}", max_chars=_MAX_ERROR_MESSAGE_CHARS)
        return prefix
    message = _redact_secretish_text(
        _normalize_text(error or "unknown compile failure", max_chars=_MAX_ERROR_MESSAGE_CHARS)
    )
    return message or "unknown compile failure"


# AUDIT-FIX(#2): Reject path-like and malformed IDs before they touch the file-backed store.
def _require_safe_identifier(value: Any, *, label: str) -> str:
    candidate = "" if value is None else str(value).strip()
    if not candidate or len(candidate) > _MAX_IDENTIFIER_CHARS:
        raise ValueError(f"{label} is invalid")
    if _CONTROL_CHAR_RE.search(candidate) or not _SAFE_IDENTIFIER_RE.fullmatch(candidate):
        raise ValueError(f"{label} is invalid")
    return candidate


def _safe_media_type(value: Any) -> str:
    candidate = _normalize_text(value or "text/plain", max_chars=_MAX_MEDIA_TYPE_CHARS).lower().replace(" ", "")
    if "/" not in candidate:
        return "text/plain"
    return candidate


def _utf8_size(text: str) -> int:
    return len(text.encode("utf-8"))


def _truncate_utf8_text(text: str, *, max_bytes: int, trailer: str) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    trailer_encoded = trailer.encode("utf-8")
    if len(trailer_encoded) >= max_bytes:
        trimmed_trailer = trailer_encoded[:max_bytes]
        while True:
            try:
                return trimmed_trailer.decode("utf-8")
            except UnicodeDecodeError as exc:
                trimmed_trailer = trimmed_trailer[: exc.start]
    prefix = encoded[: max_bytes - len(trailer_encoded)]
    while True:
        try:
            return prefix.decode("utf-8") + trailer
        except UnicodeDecodeError as exc:
            prefix = prefix[: exc.start]


def _safe_json_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= _MAX_METADATA_DEPTH:
        return _normalize_text(value, max_chars=_MAX_METADATA_STRING_CHARS)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        # AUDIT-FIX(#7): Convert NaN/Infinity to strings so persisted JSON stays standards-compliant.
        return value if math.isfinite(value) else _normalize_text(value, max_chars=32)
    if isinstance(value, str):
        return _redact_secretish_text(_normalize_text(value, max_chars=_MAX_METADATA_STRING_CHARS))
    if isinstance(value, datetime):
        aware_value = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return aware_value.isoformat()
    if isinstance(value, Path):
        return _normalize_text(value.as_posix(), max_chars=_MAX_METADATA_STRING_CHARS)
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, dict):
        items = list(value.items())
        sanitized: dict[str, Any] = {}
        for index, (raw_key, raw_value) in enumerate(items[:_MAX_METADATA_ITEMS]):
            key = _normalize_text(raw_key, max_chars=64) or f"key_{index}"
            sanitized[key] = "<redacted>" if _SECRET_KEY_RE.search(key) else _safe_json_value(
                raw_value, depth=depth + 1
            )
        if len(items) > _MAX_METADATA_ITEMS:
            sanitized["__truncated__"] = len(items) - _MAX_METADATA_ITEMS
        return sanitized
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        sanitized_items = [_safe_json_value(item, depth=depth + 1) for item in items[:_MAX_METADATA_ITEMS]]
        if len(items) > _MAX_METADATA_ITEMS:
            sanitized_items.append(f"<truncated:{len(items) - _MAX_METADATA_ITEMS}>")
        return sanitized_items
    return _normalize_text(value, max_chars=_MAX_METADATA_STRING_CHARS)


def _metadata_dict(value: Any) -> dict[str, Any]:
    sanitized = _safe_json_value(value)
    return sanitized if isinstance(sanitized, dict) else {"value": sanitized}


def _driver_attempts_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        candidate = _normalize_text(value, max_chars=64)
        return (candidate,) if candidate else ()
    if isinstance(value, (list, tuple, set, frozenset)):
        normalized = tuple(
            item
            for item in (_normalize_text(entry, max_chars=64) for entry in value)
            if item
        )
        return normalized
    candidate = _normalize_text(value, max_chars=64)
    return (candidate,) if candidate else ()


def _event_signature(event: CodexCompileEvent) -> str:
    return json.dumps(event.to_payload(), sort_keys=True, ensure_ascii=False, default=str)


def _coerce_text(value: Any) -> str:
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _truncate_review_text(text: str) -> str:
    return _truncate_utf8_text(
        text,
        max_bytes=MAX_REVIEW_BYTES,
        trailer="\n\n[review truncated to fit local storage limits]\n",
    )


def _truncate_event_log_text(text: str) -> str:
    trailer = json.dumps(
        {
            "kind": "log_truncated",
            "message": "Compile log truncated to fit local storage limits.",
            "metadata": {},
        },
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"
    return _truncate_utf8_text(text, max_bytes=MAX_EVENT_LOG_BYTES, trailer=trailer)


def _enforce_artifact_size(text: str, *, label: str) -> str:
    if _utf8_size(text) > MAX_TEXT_ARTIFACT_BYTES:
        raise CodexDriverError(f"{label} exceeds {MAX_TEXT_ARTIFACT_BYTES} bytes")
    return text


def _internal_event(kind: str, message: Any, *, metadata: dict[str, Any] | None = None) -> CodexCompileEvent:
    return CodexCompileEvent(
        kind=kind,
        message=_normalize_text(message, max_chars=_MAX_METADATA_STRING_CHARS),
        metadata=_metadata_dict(metadata),
    )


# AUDIT-FIX(#2): Coerce persisted/LLM-suggested targets back onto the supported enum values.
def _coerce_compile_target(value: Any) -> CompileTarget:
    normalized = _normalize_text(value, max_chars=64).lower()
    if value == CompileTarget.SKILL_PACKAGE or normalized in {"skill_package", "compiletarget.skill_package"}:
        return CompileTarget.SKILL_PACKAGE
    if value == CompileTarget.AUTOMATION_MANIFEST or normalized in {
        "",
        "automation_manifest",
        "compiletarget.automation_manifest",
    }:
        return CompileTarget.AUTOMATION_MANIFEST
    return CompileTarget.AUTOMATION_MANIFEST


# AUDIT-FIX(#4): Parse persisted timestamps defensively so stale-job recovery works across aware/naive store records.
def _coerce_utc_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    parsed = value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(parsed, datetime):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


# AUDIT-FIX(#4): Treat old in-progress jobs as recoverable after the worker disappears mid-compile.
def _job_is_stale(job: CompileJobRecord, *, now: datetime | None = None) -> bool:
    if job.status not in (CompileJobStatus.COMPILING, CompileJobStatus.VALIDATING):
        return False
    updated_at = _coerce_utc_datetime(getattr(job, "updated_at", None))
    if updated_at is None:
        return True
    reference = now if now is not None else datetime.now(UTC)
    return reference - updated_at > timedelta(seconds=STALE_JOB_SECONDS)


class _BoundedCompileEventBuffer:
    """Keep compile events within a fixed byte budget."""

    def __init__(self, *, max_bytes: int) -> None:
        self.max_bytes = max(1_024, int(max_bytes))
        self._events: deque[CodexCompileEvent] = deque()
        self._sizes: deque[int] = deque()
        self._total_bytes = 0
        self._truncated_event: CodexCompileEvent | None = None
        self._truncated_event_size = 0

    @staticmethod
    def _event_size(event: CodexCompileEvent) -> int:
        return _utf8_size(_event_signature(event) + "\n")

    def append(self, event: CodexCompileEvent) -> None:
        self._events.append(event)
        size = self._event_size(event)
        self._sizes.append(size)
        self._total_bytes += size
        self._trim_to_budget()

    def to_tuple(self) -> tuple[CodexCompileEvent, ...]:
        if self._truncated_event is None:
            return tuple(self._events)
        return (self._truncated_event, *tuple(self._events))

    # AUDIT-FIX(#1): Bound in-memory event accumulation so a noisy compiler cannot OOM the single-process worker.
    def _trim_to_budget(self) -> None:
        if self._total_bytes <= self.max_bytes:
            return
        if self._truncated_event is None:
            self._truncated_event = _internal_event(
                "log_truncated",
                "Compile log truncated to fit in-memory limits.",
                metadata={"max_bytes": self.max_bytes},
            )
            self._truncated_event_size = self._event_size(self._truncated_event)
            self._total_bytes += self._truncated_event_size
        while self._total_bytes > self.max_bytes and self._events:
            self._total_bytes -= self._sizes.popleft()
            self._events.popleft()


class LocalCodexCompileDriver:
    """Prefer the pinned Codex SDK bridge and fall back to `codex exec --json`."""

    def __init__(
        self,
        *,
        primary: object | None = None,
        fallback: object | None = None,
    ) -> None:
        self.primary = primary if primary is not None else CodexSdkDriver()
        self.fallback = fallback if fallback is not None else CodexExecFallbackDriver()

    def run_compile(self, request: CodexCompileRequest, *, event_sink=None) -> CodexCompileResult:
        errors: list[str] = []
        # AUDIT-FIX(#1): Cap retained compile events to the persisted log budget instead of keeping an unbounded list in RAM.
        combined_events = _BoundedCompileEventBuffer(max_bytes=MAX_EVENT_LOG_BYTES)
        total_event_count = 0
        for attempt_index, driver in enumerate((self.primary, self.fallback), start=1):
            driver_name = type(driver).__name__
            sink_failure_message: str | None = None

            def _forward(event: CodexCompileEvent, progress: CodexCompileProgress) -> None:
                nonlocal sink_failure_message, total_event_count
                enriched_event = _event_with_driver_metadata(event, driver_name=driver_name, attempt_index=attempt_index)
                total_event_count += 1
                combined_events.append(enriched_event)
                if event_sink is None:
                    return
                try:
                    event_sink(
                        enriched_event,
                        CodexCompileProgress(
                            driver_name=driver_name,
                            thread_id=getattr(progress, "thread_id", None),
                            turn_id=getattr(progress, "turn_id", None),
                            event_count=total_event_count,
                            last_event_kind=enriched_event.kind,
                            final_message_seen=getattr(progress, "final_message_seen", None),
                            turn_completed=getattr(progress, "turn_completed", None),
                            error_message=_safe_error_message(getattr(progress, "error_message", None))
                            if getattr(progress, "error_message", None)
                            else None,
                            metadata={"driver_attempt": attempt_index, **_metadata_dict(getattr(progress, "metadata", {}))},
                        ),
                    )
                except Exception as sink_exc:
                    if sink_failure_message is None:
                        sink_failure_message = _safe_error_message(sink_exc)
                        total_event_count += 1
                        combined_events.append(
                            _event_with_driver_metadata(
                                _internal_event(
                                    "event_sink_failure",
                                    sink_failure_message,
                                    metadata={"driver_attempt": attempt_index},
                                ),
                                driver_name=driver_name,
                                attempt_index=attempt_index,
                            )
                        )

            try:
                result = driver.run_compile(request, event_sink=_forward)
                returned_events = tuple(getattr(result, "events", ()) or ())
                _merge_compile_events(
                    combined_events,
                    returned_events,
                    driver_name=driver_name,
                    attempt_index=attempt_index,
                )
                return replace(result, events=combined_events.to_tuple())
            except Exception as exc:
                safe_error = _safe_error_message(exc)
                errors.append(f"{driver_name}: {safe_error}")
                total_event_count += 1
                failure_event = _event_with_driver_metadata(
                    CodexCompileEvent(
                        kind="driver_failure",
                        message=safe_error,
                        metadata={"driver_attempt": attempt_index},
                    ),
                    driver_name=driver_name,
                    attempt_index=attempt_index,
                )
                combined_events.append(failure_event)
                if event_sink is not None:
                    try:
                        event_sink(
                            failure_event,
                            CodexCompileProgress(
                                driver_name=driver_name,
                                event_count=total_event_count,
                                last_event_kind=failure_event.kind,
                                error_message=safe_error,
                                metadata={"driver_attempt": attempt_index},
                            ),
                        )
                    except Exception as sink_exc:
                        if sink_failure_message is None:
                            sink_failure_message = _safe_error_message(sink_exc)
                            total_event_count += 1
                            combined_events.append(
                                _event_with_driver_metadata(
                                    _internal_event(
                                        "event_sink_failure",
                                        sink_failure_message,
                                        metadata={"driver_attempt": attempt_index},
                                    ),
                                    driver_name=driver_name,
                                    attempt_index=attempt_index,
                                )
                            )
                continue
        raise CodexDriverError("; ".join(errors) or "No Codex compile driver succeeded")


class SelfCodingCompileWorker:
    """Create queued compile jobs and execute them through a local Codex driver."""

    def __init__(
        self,
        *,
        store: SelfCodingStore,
        driver: object | None = None,
        workspace_builder: CodexCompileWorkspaceBuilder | None = None,
    ) -> None:
        self.store = store
        self.driver = driver if driver is not None else LocalCodexCompileDriver()
        self.workspace_builder = workspace_builder if workspace_builder is not None else CodexCompileWorkspaceBuilder()

    # AUDIT-FIX(#3): Normalize unexpected store failures into bounded, redacted driver errors.
    @staticmethod
    def _raise_safe_store_error(exc: Exception) -> None:
        raise CodexDriverError(_safe_error_message(exc)) from exc

    # AUDIT-FIX(#4): Requeue stale in-progress jobs so a reboot/crash does not strand the session forever.
    def _recover_stale_job_if_needed(self, job: CompileJobRecord) -> CompileJobRecord:
        if not _job_is_stale(job):
            return job
        return self.store.save_job(
            replace(
                job,
                status=CompileJobStatus.QUEUED,
                updated_at=datetime.now(UTC),
                last_error="Recovered stale in-progress compile job after worker interruption.",
            )
        )

    def ensure_job_for_session(self, session: RequirementsDialogueSession) -> CompileJobRecord:
        """Return an existing queued job for a ready session or create a new one."""

        try:
            if session.status != RequirementsDialogueStatus.READY_FOR_COMPILE:
                raise ValueError("compile jobs require a ready_for_compile session")
            # AUDIT-FIX(#2): Validate file-backed identifiers and normalize requested target before persisting any job record.
            session_id = _require_safe_identifier(session.session_id, label="session_id")
            skill_spec = session.to_skill_spec()
            suggested_target = getattr(getattr(session, "feasibility", None), "suggested_target", None)
            requested_target = _coerce_compile_target(suggested_target)
            spec_hash = self._spec_hash(skill_spec.to_payload())
            with _acquire_named_lock(_SESSION_LOCKS, session_id):
                self.store.save_dialogue_session(session)
                existing = self.store.find_job_for_session(session_id)
                if existing is not None and existing.spec_hash == spec_hash and existing.requested_target == requested_target:
                    existing = self._recover_stale_job_if_needed(existing)
                    if existing.status in (
                        CompileJobStatus.QUEUED,
                        CompileJobStatus.COMPILING,
                        CompileJobStatus.VALIDATING,
                        CompileJobStatus.SOFT_LAUNCH_READY,
                    ):
                        return existing
                job = CompileJobRecord(
                    job_id=f"job_{uuid4().hex}",
                    skill_id=skill_spec.skill_id,
                    skill_name=skill_spec.name,
                    status=CompileJobStatus.QUEUED,
                    requested_target=requested_target,
                    spec_hash=spec_hash,
                    required_capabilities=skill_spec.capabilities,
                    metadata={
                        "session_id": session_id,
                        # AUDIT-FIX(#7): Bound and sanitize persisted request summaries before they hit JSON-backed state.
                        "request_summary": _normalize_text(session.request_summary, max_chars=_MAX_PROMPT_FIELD_CHARS),
                    },
                )
                return self.store.save_job(job)
        except ValueError:
            raise
        except CodexDriverError:
            raise
        except Exception as exc:
            self._raise_safe_store_error(exc)

    def run_job(self, job_id: str) -> CompileJobRecord:
        """Run one queued compile job and persist its log and output artifacts."""

        # AUDIT-FIX(#2): Reject malformed job IDs before touching the file-backed store.
        normalized_job_id = _require_safe_identifier(job_id, label="job_id")
        with _acquire_named_lock(_JOB_LOCKS, normalized_job_id):
            job: CompileJobRecord | None = None
            status_record: CompileRunStatusRecord | None = None
            try:
                job = self.store.load_job(normalized_job_id)
                job = self._recover_stale_job_if_needed(job)
                if job.status != CompileJobStatus.QUEUED:
                    return job
                session = self._load_job_session(job)
                started_at = datetime.now(UTC)
                active_job = self.store.save_job(
                    replace(
                        job,
                        status=CompileJobStatus.COMPILING,
                        updated_at=started_at,
                        attempt_count=int(getattr(job, "attempt_count", 0) or 0) + 1,
                        last_error=None,
                    )
                )
                last_job_heartbeat = started_at
                status_record = self.store.save_compile_status(
                    CompileRunStatusRecord(
                        job_id=active_job.job_id,
                        phase="starting",
                        started_at=started_at,
                        updated_at=started_at,
                    )
                )

                prompt = self._build_prompt(active_job, session)
                # AUDIT-FIX(#1): Keep streamed event history bounded even on all-driver failure paths.
                streamed_events = _BoundedCompileEventBuffer(max_bytes=MAX_EVENT_LOG_BYTES)
                progress_persist_error: str | None = None

                def _event_sink(event: CodexCompileEvent, progress: CodexCompileProgress) -> None:
                    nonlocal active_job, last_job_heartbeat, status_record, progress_persist_error
                    streamed_events.append(event)
                    try:
                        status_record = self._record_compile_progress(
                            current=status_record,
                            event=event,
                            progress=progress,
                        )
                        # AUDIT-FIX(#4): Heartbeat the in-progress job record so stale-job recovery does not resurrect live compiles.
                        now = datetime.now(UTC)
                        if now - last_job_heartbeat >= timedelta(seconds=COMPILE_JOB_HEARTBEAT_SECONDS):
                            active_job = self.store.save_job(replace(active_job, updated_at=now))
                            last_job_heartbeat = now
                    except Exception as exc:
                        progress_persist_error = _safe_error_message(exc)

                try:
                    with self.workspace_builder.build(job=active_job, session=session, prompt=prompt) as request:
                        result = self.driver.run_compile(
                            request,
                            event_sink=_event_sink,
                        )
                except Exception as exc:
                    safe_error = _safe_error_message(exc)
                    job_with_failure_log = self._try_persist_failure_log(
                        active_job,
                        events=streamed_events.to_tuple(),
                        error_message=safe_error,
                        progress_persist_error=progress_persist_error,
                    )
                    if self._should_retry_compile(active_job, result_status="failed"):
                        return self._queue_retry(job_with_failure_log, safe_error, compile_status=status_record)
                    return self._mark_failed(job_with_failure_log, safe_error, compile_status=status_record)

                validating_diagnostics: dict[str, Any] = {"result_status": result.status}
                if progress_persist_error:
                    validating_diagnostics["progress_persist_error"] = progress_persist_error
                try:
                    status_record = self._save_compile_status_transition(
                        status_record,
                        phase="validating",
                        diagnostics=validating_diagnostics,
                    )
                except Exception:
                    pass

                try:
                    job_with_logs, has_current_target_artifact = self._persist_driver_result(active_job, session, result)
                except Exception as exc:
                    safe_error = _safe_error_message(exc)
                    latest_job = self._safe_load_job(active_job.job_id, default=active_job)
                    if self._should_retry_compile(active_job, result_status="failed"):
                        return self._queue_retry(latest_job, safe_error, compile_status=status_record)
                    return self._mark_failed(latest_job, safe_error, compile_status=status_record)
                target_kind = _target_artifact_kind(job_with_logs.requested_target)
                effective_result_status = result.status
                failure_message = _safe_error_message(result.summary)
                if result.status == "ok" and not has_current_target_artifact:
                    # AUDIT-FIX(#5): Treat missing or stale target artifacts as a failed compile so retry logic still engages.
                    effective_result_status = "failed"
                    failure_message = (
                        f"Compile completed without required {_normalize_text(getattr(target_kind, 'value', target_kind), max_chars=64)} artifact"
                    )
                if effective_result_status == "ok":
                    completed = self.store.save_job(
                        replace(
                            job_with_logs,
                            status=CompileJobStatus.SOFT_LAUNCH_READY,
                            updated_at=datetime.now(UTC),
                            last_error=None,
                        )
                    )
                    try:
                        self._save_compile_status_transition(
                            status_record,
                            phase="completed",
                            completed_at=datetime.now(UTC),
                            diagnostics={
                                "result_status": result.status,
                                "artifact_count": len(getattr(completed, "artifact_ids", ()) or ()),
                            },
                        )
                    except Exception:
                        pass
                    return completed
                if self._should_retry_compile(active_job, result_status=effective_result_status):
                    return self._queue_retry(job_with_logs, failure_message, compile_status=status_record)
                return self._mark_failed(job_with_logs, failure_message, compile_status=status_record)
            except Exception as exc:
                # AUDIT-FIX(#3): Best-effort mark unexpected store/state failures without leaking raw filesystem details.
                safe_error = _safe_error_message(exc)
                if job is not None:
                    latest_job = self._safe_load_job(job.job_id, default=job)
                    try:
                        return self._mark_failed(latest_job, safe_error, compile_status=status_record)
                    except Exception:
                        pass
                raise CodexDriverError(safe_error) from exc

    def _load_job_session(self, job: CompileJobRecord) -> RequirementsDialogueSession:
        session_id = _require_safe_identifier(
            _metadata_dict(getattr(job, "metadata", {})).get("session_id", ""),
            label="session_id",
        )
        return self.store.load_dialogue_session(session_id)

    def _persist_driver_result(
        self,
        job: CompileJobRecord,
        session: RequirementsDialogueSession,
        result: CodexCompileResult,
    ) -> tuple[CompileJobRecord, bool]:
        artifacts = tuple(getattr(result, "artifacts", ()) or ())
        if len(artifacts) > MAX_ARTIFACTS_PER_JOB:
            raise CodexDriverError(
                f"compile result returned {len(artifacts)} artifacts; limit is {MAX_ARTIFACTS_PER_JOB}"
            )
        current = self.store.save_job(
            replace(
                job,
                status=CompileJobStatus.VALIDATING,
                updated_at=datetime.now(UTC),
                last_error=None,
            )
        )
        current = self._persist_event_log_artifact(
            current,
            events=tuple(getattr(result, "events", ()) or ()),
            driver_result_status=result.status,
        )

        persisted_target_artifact = False

        review_text = _coerce_text(getattr(result, "review", ""))
        if review_text:
            review_artifact = self.store.write_text_artifact(
                job_id=current.job_id,
                kind=ArtifactKind.REVIEW,
                text=_truncate_review_text(review_text),
                media_type="text/plain",
                summary="Local Codex compile review summary.",
                metadata={"driver_result_status": _normalize_text(result.status, max_chars=64)},
                suffix=".txt",
            )
            current = self.store.append_artifact_to_job(current.job_id, review_artifact.artifact_id)

        for index, artifact in enumerate(artifacts, start=1):
            content = _coerce_text(artifact.content)
            summary = _normalize_text(
                artifact.summary or f"Local Codex compile artifact {index}.",
                max_chars=_MAX_METADATA_STRING_CHARS,
            )
            metadata = {
                **_metadata_dict(getattr(artifact, "metadata", {})),
                "artifact_name": _normalize_text(getattr(artifact, "artifact_name", ""), max_chars=128),
            }
            suffix = _artifact_suffix(artifact)
            media_type = _safe_media_type(getattr(artifact, "media_type", "text/plain"))
            if artifact.kind == ArtifactKind.AUTOMATION_MANIFEST:
                validated_artifact = validate_compile_artifact(
                    job=current,
                    session=session,
                    artifact=artifact,
                )
                content = _coerce_text(validated_artifact.content)
                summary = _normalize_text(validated_artifact.summary or summary, max_chars=_MAX_METADATA_STRING_CHARS)
                metadata = {
                    **metadata,
                    **_metadata_dict(getattr(validated_artifact, "metadata", {})),
                }
                suffix = ".json"
                media_type = "application/json"
            elif artifact.kind == ArtifactKind.SKILL_PACKAGE:
                validated_artifact = validate_compile_artifact(
                    job=current,
                    session=session,
                    artifact=artifact,
                )
                content = _coerce_text(validated_artifact.content)
                summary = _normalize_text(validated_artifact.summary or summary, max_chars=_MAX_METADATA_STRING_CHARS)
                metadata = {
                    **metadata,
                    **_metadata_dict(getattr(validated_artifact, "metadata", {})),
                }
                suffix = ".json"
                media_type = "application/json"
            content = _enforce_artifact_size(content, label=f"artifact {index}")
            persisted = self.store.write_text_artifact(
                job_id=current.job_id,
                kind=artifact.kind,
                text=content,
                media_type=media_type,
                summary=summary,
                metadata=metadata,
                suffix=suffix,
            )
            current = self.store.append_artifact_to_job(current.job_id, persisted.artifact_id)
            if artifact.kind == _target_artifact_kind(job.requested_target):
                persisted_target_artifact = True

        return current, persisted_target_artifact

    def _persist_event_log_artifact(
        self,
        job: CompileJobRecord,
        *,
        events: tuple[CodexCompileEvent, ...],
        driver_result_status: str,
    ) -> CompileJobRecord:
        event_log_text = self._event_log_text(events)
        if _utf8_size(event_log_text) > MAX_EVENT_LOG_BYTES:
            event_log_text = _truncate_event_log_text(event_log_text)
        log_artifact = self.store.write_text_artifact(
            job_id=job.job_id,
            kind=ArtifactKind.LOG,
            text=event_log_text,
            media_type="application/x-ndjson",
            summary="Local Codex compile event log.",
            metadata={"driver_result_status": _normalize_text(driver_result_status, max_chars=64)},
            suffix=".jsonl",
        )
        return self.store.append_artifact_to_job(job.job_id, log_artifact.artifact_id)

    def _try_persist_failure_log(
        self,
        job: CompileJobRecord,
        *,
        events: tuple[CodexCompileEvent, ...],
        error_message: str,
        progress_persist_error: str | None,
    ) -> CompileJobRecord:
        extra_events = list(events)
        if progress_persist_error:
            extra_events.append(
                _internal_event(
                    "status_persist_failure",
                    progress_persist_error,
                    metadata={"job_id": job.job_id},
                )
            )
        extra_events.append(
            _internal_event(
                "compile_failed",
                error_message,
                metadata={"job_id": job.job_id},
            )
        )
        try:
            return self._persist_event_log_artifact(
                job,
                events=tuple(extra_events),
                driver_result_status="failed",
            )
        except Exception:
            return job

    def _safe_load_job(self, job_id: str, *, default: CompileJobRecord) -> CompileJobRecord:
        try:
            return self.store.load_job(job_id)
        except Exception:
            return default

    def _mark_failed(
        self,
        job: CompileJobRecord,
        error: str,
        *,
        compile_status: CompileRunStatusRecord | None = None,
    ) -> CompileJobRecord:
        safe_error = _safe_error_message(error)
        failed = self.store.save_job(
            replace(
                job,
                status=CompileJobStatus.FAILED,
                updated_at=datetime.now(UTC),
                last_error=safe_error,
            )
        )
        if compile_status is not None:
            try:
                self._save_compile_status_transition(
                    compile_status,
                    phase="failed",
                    completed_at=datetime.now(UTC),
                    error_message=safe_error,
                )
            except Exception:
                pass
        return failed

    def _queue_retry(
        self,
        job: CompileJobRecord,
        error: str,
        *,
        compile_status: CompileRunStatusRecord | None = None,
    ) -> CompileJobRecord:
        retry_job = self.store.save_job(
            replace(
                job,
                status=CompileJobStatus.QUEUED,
                updated_at=datetime.now(UTC),
                last_error=_safe_error_message(error),
            )
        )
        if compile_status is not None:
            try:
                self._save_compile_status_transition(
                    compile_status,
                    phase="retrying",
                    error_message=_safe_error_message(error),
                    diagnostics={"retry_attempt": int(getattr(job, "attempt_count", 0) or 0)},
                )
            except Exception:
                pass
        return self.run_job(retry_job.job_id)

    @staticmethod
    def _should_retry_compile(job: CompileJobRecord, *, result_status: str) -> bool:
        normalized_status = _normalize_text(result_status, max_chars=32).lower()
        if normalized_status != "failed":
            return False
        return int(getattr(job, "attempt_count", 0) or 0) < MAX_COMPILE_RUN_ATTEMPTS

    @staticmethod
    def _spec_hash(payload: dict[str, Any]) -> str:
        stable_payload = dict(payload)
        stable_payload.pop("created_at", None)
        encoded = json.dumps(stable_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return sha256(encoded).hexdigest()

    @staticmethod
    def _build_prompt(job: CompileJobRecord, session: RequirementsDialogueSession) -> str:
        return build_compile_prompt(job, session)

    @staticmethod
    def _event_log_text(events: tuple[CodexCompileEvent, ...]) -> str:
        return "\n".join(
            json.dumps(event.to_payload(), sort_keys=True, ensure_ascii=False, default=str) for event in events
        ) + ("\n" if events else "")

    def _record_compile_progress(
        self,
        *,
        current: CompileRunStatusRecord,
        event: CodexCompileEvent,
        progress: CodexCompileProgress,
    ) -> CompileRunStatusRecord:
        driver_name = _normalize_text(
            getattr(progress, "driver_name", None) or getattr(current, "driver_name", None),
            max_chars=64,
        ) or None
        driver_attempts = _driver_attempts_tuple(getattr(current, "driver_attempts", ()))
        if driver_name and driver_name not in driver_attempts:
            driver_attempts = tuple((*driver_attempts, driver_name))
        event_count = getattr(progress, "event_count", None)
        if not isinstance(event_count, int) or event_count < 0:
            event_count = int(getattr(current, "event_count", 0) or 0)
        return self.store.save_compile_status(
            CompileRunStatusRecord(
                job_id=current.job_id,
                phase="streaming",
                driver_name=driver_name,
                driver_attempts=driver_attempts,
                event_count=event_count,
                last_event_kind=getattr(progress, "last_event_kind", None) or event.kind,
                last_event_message=_normalize_text(
                    event.message or getattr(current, "last_event_message", None),
                    max_chars=_MAX_METADATA_STRING_CHARS,
                ),
                thread_id=getattr(progress, "thread_id", None) or getattr(current, "thread_id", None),
                turn_id=getattr(progress, "turn_id", None) or getattr(current, "turn_id", None),
                final_message_seen=(
                    getattr(current, "final_message_seen", False)
                    if getattr(progress, "final_message_seen", None) is None
                    else bool(getattr(progress, "final_message_seen"))
                ),
                turn_completed=(
                    getattr(current, "turn_completed", False)
                    if getattr(progress, "turn_completed", None) is None
                    else bool(getattr(progress, "turn_completed"))
                ),
                started_at=getattr(current, "started_at", None) or datetime.now(UTC),
                updated_at=datetime.now(UTC),
                error_message=_safe_error_message(getattr(progress, "error_message", None))
                if getattr(progress, "error_message", None)
                else getattr(current, "error_message", None),
                diagnostics=_metadata_dict(getattr(current, "diagnostics", {})),
            )
        )

    def _save_compile_status_transition(
        self,
        current: CompileRunStatusRecord,
        *,
        phase: str,
        completed_at: datetime | None = None,
        error_message: str | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> CompileRunStatusRecord:
        merged_diagnostics = _metadata_dict(getattr(current, "diagnostics", {}))
        if diagnostics:
            merged_diagnostics.update(_metadata_dict(diagnostics))
        return self.store.save_compile_status(
            CompileRunStatusRecord(
                job_id=current.job_id,
                phase=phase,
                driver_name=getattr(current, "driver_name", None),
                driver_attempts=_driver_attempts_tuple(getattr(current, "driver_attempts", ())),
                event_count=int(getattr(current, "event_count", 0) or 0),
                last_event_kind=getattr(current, "last_event_kind", None),
                last_event_message=getattr(current, "last_event_message", None),
                thread_id=getattr(current, "thread_id", None),
                turn_id=getattr(current, "turn_id", None),
                final_message_seen=bool(getattr(current, "final_message_seen", False)),
                turn_completed=bool(getattr(current, "turn_completed", False)),
                started_at=getattr(current, "started_at", None),
                updated_at=datetime.now(UTC),
                completed_at=completed_at,
                error_message=_safe_error_message(error_message)
                if error_message
                else getattr(current, "error_message", None),
                diagnostics=merged_diagnostics,
            )
        )


def _artifact_suffix(artifact: CodexCompileArtifact) -> str:
    if artifact.kind in {ArtifactKind.AUTOMATION_MANIFEST, ArtifactKind.SKILL_PACKAGE}:
        return ".json"
    suffix = Path(str(getattr(artifact, "artifact_name", "") or "")).suffix.strip()
    if suffix and _VALID_SUFFIX_RE.fullmatch(suffix):
        return suffix.lower()
    return ".txt"


def _event_with_driver_metadata(
    event: CodexCompileEvent,
    *,
    driver_name: str,
    attempt_index: int,
) -> CodexCompileEvent:
    return CodexCompileEvent(
        kind=event.kind,
        message=_normalize_text(event.message, max_chars=_MAX_METADATA_STRING_CHARS),
        metadata={
            **_metadata_dict(getattr(event, "metadata", {})),
            "driver_name": _normalize_text(driver_name, max_chars=64),
            "driver_attempt": attempt_index,
        },
    )


def _merge_compile_events(
    combined_events: _BoundedCompileEventBuffer,
    returned_events: tuple[CodexCompileEvent, ...],
    *,
    driver_name: str,
    attempt_index: int,
) -> None:
    seen_signatures = {_event_signature(event) for event in combined_events.to_tuple()}
    for event in returned_events:
        enriched = _event_with_driver_metadata(
            event,
            driver_name=driver_name,
            attempt_index=attempt_index,
        )
        signature = _event_signature(enriched)
        if signature not in seen_signatures:
            combined_events.append(enriched)
            seen_signatures.add(signature)


def _target_artifact_kind(target: CompileTarget) -> ArtifactKind:
    if target == CompileTarget.SKILL_PACKAGE:
        return ArtifactKind.SKILL_PACKAGE
    return ArtifactKind.AUTOMATION_MANIFEST