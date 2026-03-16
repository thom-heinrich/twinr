"""Create and run local self-coding compile jobs."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import threading
from typing import Any
from uuid import uuid4

from twinr.agent.self_coding.codex_driver import (
    CodexAppServerDriver,
    CodexCompileArtifact,
    CodexCompileEvent,
    CodexCompileProgress,
    CodexCompileRequest,
    CodexCompileResult,
    CodexCompileWorkspaceBuilder,
    CodexDriverError,
    CodexExecFallbackDriver,
)
from twinr.agent.self_coding.compiler import compile_automation_manifest_content
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

# AUDIT-FIX(#3): Add env-backed guardrails so compiler output cannot exhaust the Pi's SD card.
_MAX_ERROR_MESSAGE_CHARS = 240
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

# AUDIT-FIX(#1): Serialize session/job mutations to stop duplicate jobs and double execution against a file-backed store.
_LOCK_REGISTRY_GUARD = threading.Lock()
_SESSION_LOCKS: dict[str, threading.RLock] = {}
_JOB_LOCKS: dict[str, threading.RLock] = {}


def _named_lock(registry: dict[str, threading.RLock], key: str) -> threading.RLock:
    normalized_key = str(key or "").strip() or "__empty__"
    with _LOCK_REGISTRY_GUARD:
        lock = registry.get(normalized_key)
        if lock is None:
            lock = threading.RLock()
            registry[normalized_key] = lock
        return lock


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


# AUDIT-FIX(#6): Never persist raw exception strings; normalize and redact them before they can leak upstream.
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
    if value is None or isinstance(value, (bool, int, float)):
        return value
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


# AUDIT-FIX(#5): Coerce every metadata/diagnostics payload to a JSON-safe mapping before merging or serializing it.
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


class LocalCodexCompileDriver:
    """Prefer app-server and fall back to `codex exec --json` when needed."""

    def __init__(
        self,
        *,
        primary: object | None = None,
        fallback: object | None = None,
    ) -> None:
        # AUDIT-FIX(#11): Use explicit None checks so valid falsy test doubles are not silently replaced.
        self.primary = primary if primary is not None else CodexAppServerDriver()
        # AUDIT-FIX(#11): Use explicit None checks so valid falsy test doubles are not silently replaced.
        self.fallback = fallback if fallback is not None else CodexExecFallbackDriver()

    def run_compile(self, request: CodexCompileRequest, *, event_sink=None) -> CodexCompileResult:
        errors: list[str] = []
        combined_events: list[CodexCompileEvent] = []
        for attempt_index, driver in enumerate((self.primary, self.fallback), start=1):
            driver_name = type(driver).__name__
            sink_failure_message: str | None = None

            def _forward(event: CodexCompileEvent, progress: CodexCompileProgress) -> None:
                nonlocal sink_failure_message
                enriched_event = _event_with_driver_metadata(event, driver_name=driver_name, attempt_index=attempt_index)
                combined_events.append(enriched_event)
                if event_sink is None:
                    return
                try:
                    # AUDIT-FIX(#4): Shield progress sinks so status persistence bugs do not abort the compile or suppress fallback.
                    event_sink(
                        enriched_event,
                        CodexCompileProgress(
                            driver_name=driver_name,
                            thread_id=getattr(progress, "thread_id", None),
                            turn_id=getattr(progress, "turn_id", None),
                            event_count=len(combined_events),
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
                # AUDIT-FIX(#8): Merge returned events even after streaming so late/fallback events are not dropped.
                _merge_compile_events(
                    combined_events,
                    returned_events,
                    driver_name=driver_name,
                    attempt_index=attempt_index,
                )
                return replace(result, events=tuple(combined_events))
            except Exception as exc:
                # AUDIT-FIX(#4): Catch unexpected driver exceptions too, so a buggy primary still falls back to the secondary driver.
                safe_error = _safe_error_message(exc)
                errors.append(f"{driver_name}: {safe_error}")
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
                                event_count=len(combined_events),
                                last_event_kind=failure_event.kind,
                                error_message=safe_error,
                                metadata={"driver_attempt": attempt_index},
                            ),
                        )
                    except Exception as sink_exc:
                        if sink_failure_message is None:
                            sink_failure_message = _safe_error_message(sink_exc)
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
        # AUDIT-FIX(#11): Use explicit None checks so valid falsy dependency injections survive.
        self.driver = driver if driver is not None else LocalCodexCompileDriver()
        # AUDIT-FIX(#11): Use explicit None checks so valid falsy dependency injections survive.
        self.workspace_builder = workspace_builder if workspace_builder is not None else CodexCompileWorkspaceBuilder()

    def ensure_job_for_session(self, session: RequirementsDialogueSession) -> CompileJobRecord:
        """Return an existing queued job for a ready session or create a new one."""

        if session.status != RequirementsDialogueStatus.READY_FOR_COMPILE:
            raise ValueError("compile jobs require a ready_for_compile session")
        skill_spec = session.to_skill_spec()
        suggested_target = getattr(getattr(session, "feasibility", None), "suggested_target", None)
        requested_target = suggested_target or CompileTarget.AUTOMATION_MANIFEST
        spec_hash = self._spec_hash(skill_spec.to_payload())
        session_lock = _named_lock(_SESSION_LOCKS, session.session_id)
        with session_lock:
            # AUDIT-FIX(#1): Serialize save/find/create to remove the TOCTOU window that can create duplicate jobs.
            self.store.save_dialogue_session(session)
            existing = self.store.find_job_for_session(session.session_id)
            if (
                existing is not None
                and existing.spec_hash == spec_hash
                and existing.requested_target == requested_target
                and existing.status in (
                    CompileJobStatus.QUEUED,
                    CompileJobStatus.COMPILING,
                    CompileJobStatus.VALIDATING,
                    CompileJobStatus.SOFT_LAUNCH_READY,
                )
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
                    "session_id": session.session_id,
                    "request_summary": session.request_summary,
                },
            )
            return self.store.save_job(job)

    def run_job(self, job_id: str) -> CompileJobRecord:
        """Run one queued compile job and persist its log and output artifacts."""

        normalized_job_id = str(job_id or "").strip()
        if not normalized_job_id:
            raise ValueError("job_id is required")
        job_lock = _named_lock(_JOB_LOCKS, normalized_job_id)
        with job_lock:
            # AUDIT-FIX(#2): Serialize per-job execution and treat non-queued jobs as idempotent no-ops.
            job = self.store.load_job(normalized_job_id)
            if job.status != CompileJobStatus.QUEUED:
                return job
            try:
                session = self._load_job_session(job)
            except Exception as exc:
                # AUDIT-FIX(#7): Corrupt job/session metadata must fail the job instead of crashing the worker.
                return self._mark_failed(job, _safe_error_message(exc))
            started_at = datetime.now(UTC)
            compiling_job = self.store.save_job(
                replace(
                    job,
                    status=CompileJobStatus.COMPILING,
                    updated_at=started_at,
                    attempt_count=int(getattr(job, "attempt_count", 0) or 0) + 1,
                    last_error=None,
                )
            )
            status_record = self.store.save_compile_status(
                CompileRunStatusRecord(
                    job_id=compiling_job.job_id,
                    phase="starting",
                    started_at=started_at,
                    updated_at=started_at,
                )
            )

            prompt = self._build_prompt(compiling_job, session)
            streamed_events: list[CodexCompileEvent] = []
            progress_persist_error: str | None = None

            def _event_sink(event: CodexCompileEvent, progress: CodexCompileProgress) -> None:
                nonlocal status_record, progress_persist_error
                streamed_events.append(event)
                try:
                    status_record = self._record_compile_progress(
                        current=status_record,
                        event=event,
                        progress=progress,
                    )
                except Exception as exc:
                    # AUDIT-FIX(#4): Status-stream persistence is best-effort; do not kill the compile because telemetry failed.
                    progress_persist_error = _safe_error_message(exc)

            try:
                # AUDIT-FIX(#7): Wrap workspace creation too, otherwise setup errors bypass failure marking and leave jobs stuck.
                with self.workspace_builder.build(job=compiling_job, session=session, prompt=prompt) as request:
                    result = self.driver.run_compile(
                        request,
                        event_sink=_event_sink,
                    )
            except Exception as exc:
                safe_error = _safe_error_message(exc)
                # AUDIT-FIX(#9): Persist whatever event stream we captured even when every driver attempt failed.
                job_with_failure_log = self._try_persist_failure_log(
                    compiling_job,
                    events=tuple(streamed_events),
                    error_message=safe_error,
                    progress_persist_error=progress_persist_error,
                )
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
                job_with_logs = self._persist_driver_result(compiling_job, session, result)
            except Exception as exc:
                safe_error = _safe_error_message(exc)
                latest_job = self._safe_load_job(compiling_job.job_id, default=compiling_job)
                return self._mark_failed(latest_job, safe_error, compile_status=status_record)
            target_kind = _target_artifact_kind(job_with_logs.requested_target)
            has_target_artifact = any(
                artifact.kind == target_kind for artifact in self.store.list_artifacts(job_id=job_with_logs.job_id)
            )
            if result.status == "ok" and has_target_artifact:
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
                            "artifact_count": len(self.store.list_artifacts(job_id=completed.job_id)),
                        },
                    )
                except Exception:
                    pass
                return completed
            return self._mark_failed(job_with_logs, _safe_error_message(result.summary), compile_status=status_record)

    def _load_job_session(self, job: CompileJobRecord) -> RequirementsDialogueSession:
        # AUDIT-FIX(#5): Guard against missing/None metadata payloads when loading the owning dialogue session.
        session_id = str(_metadata_dict(getattr(job, "metadata", {})).get("session_id", "") or "").strip()
        if not session_id:
            raise ValueError(f"Compile job {job.job_id!r} is missing session_id metadata")
        return self.store.load_dialogue_session(session_id)

    def _persist_driver_result(
        self,
        job: CompileJobRecord,
        session: RequirementsDialogueSession,
        result: CodexCompileResult,
    ) -> CompileJobRecord:
        artifacts = tuple(getattr(result, "artifacts", ()) or ())
        if len(artifacts) > MAX_ARTIFACTS_PER_JOB:
            # AUDIT-FIX(#3): Reject pathological result fan-out before it can fill the local store.
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
                # AUDIT-FIX(#5): Sanitize artifact metadata from drivers/manifests before persisting it to JSON-backed state.
                "artifact_name": _normalize_text(getattr(artifact, "artifact_name", ""), max_chars=128),
            }
            suffix = _artifact_suffix(artifact)
            media_type = _safe_media_type(getattr(artifact, "media_type", "text/plain"))
            if artifact.kind == ArtifactKind.AUTOMATION_MANIFEST:
                compiled_manifest = compile_automation_manifest_content(
                    job=current,
                    session=session,
                    raw_content=artifact.content,
                )
                content = _coerce_text(compiled_manifest.content)
                summary = _normalize_text(compiled_manifest.summary or summary, max_chars=_MAX_METADATA_STRING_CHARS)
                metadata = {
                    **metadata,
                    **_metadata_dict(getattr(compiled_manifest, "metadata", {})),
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

        return current

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
            # AUDIT-FIX(#12): JSONL/NDJSON logs must be labeled correctly for downstream readers.
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

    @staticmethod
    def _spec_hash(payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return sha256(encoded).hexdigest()

    @staticmethod
    def _build_prompt(job: CompileJobRecord, session: RequirementsDialogueSession) -> str:
        target = _normalize_text(job.requested_target.value, max_chars=128)
        skill_name = _normalize_text(session.skill_name, max_chars=256)
        request_summary = _normalize_text(session.request_summary, max_chars=_MAX_PROMPT_FIELD_CHARS)
        return (
            "You are compiling a Twinr self-coding request into reviewable artifacts.\n\n"
            "Read the workspace files `REQUEST.md`, `skill_spec.json`, `dialogue_session.json`, and `compile_job.json`.\n"
            "Return only JSON that matches the provided output schema.\n"
            "Do not ask follow-up questions.\n"
            f"Requested target: {target}\n"
            f"Skill name: {skill_name}\n"
            f"Request summary: {request_summary}\n"
            "Rules:\n"
            "- If the request fits the target, set `status` to `ok` and include at least one artifact of the requested kind.\n"
            "- If the request does not fit the target, set `status` to `unsupported`, explain why in `summary`, and keep artifacts empty.\n"
            "- Put all artifact contents directly into the JSON response; do not describe patches or external files.\n"
        )

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
        # AUDIT-FIX(#5): `current.diagnostics` may be None or non-serializable; sanitize before copying/updating it.
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


# AUDIT-FIX(#10): Restrict suffixes to a sane allow-list instead of trusting model-supplied filenames.
def _artifact_suffix(artifact: CodexCompileArtifact) -> str:
    suffix = Path(str(getattr(artifact, "artifact_name", "") or "")).suffix.strip()
    if suffix and _VALID_SUFFIX_RE.fullmatch(suffix):
        return suffix.lower()
    return ".txt"


# AUDIT-FIX(#5): Sanitize event metadata before merging it so `None`, paths, datetimes, or secret-bearing fields cannot break logs.
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
    combined_events: list[CodexCompileEvent],
    returned_events: tuple[CodexCompileEvent, ...],
    *,
    driver_name: str,
    attempt_index: int,
) -> None:
    seen_signatures = {_event_signature(event) for event in combined_events}
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