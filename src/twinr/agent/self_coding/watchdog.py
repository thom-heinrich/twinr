"""Surface stale self-coding runs and provide bounded operator cleanup helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import logging
import math
from typing import Any

from twinr.agent.self_coding.contracts import CompileRunStatusRecord, ExecutionRunStatusRecord
from twinr.agent.self_coding.status import CompileJobStatus
from twinr.agent.self_coding.store import SelfCodingStore

_LOGGER = logging.getLogger(__name__)  # AUDIT-FIX(#3): Log snapshot degradation paths instead of letting one bad store read take down the whole operator page.
_DEFAULT_CLEANUP_REASON = "operator_cleanup"  # AUDIT-FIX(#8): Reuse one normalized fallback reason so persisted audit strings stay bounded and predictable.
_MAX_REASON_LENGTH = 512
_MAX_RECORD_ID_LENGTH = 255

_TERMINAL_COMPILE_PHASES = frozenset({"completed", "failed", "aborted", "cleaned"})
_TERMINAL_EXECUTION_STATUSES = frozenset({"completed", "failed", "timed_out", "aborted", "cleaned"})


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _validate_threshold_seconds(value: Any, *, field_name: str) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite non-negative number of seconds.") from exc
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative number of seconds.")
    return seconds


def _coerce_utc_datetime(value: datetime, *, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime instance.")
    # AUDIT-FIX(#2): Normalize both naive and aware timestamps to UTC before any age math so legacy files cannot crash stale-run detection.
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _isoformat_z(value: datetime) -> str:
    return _coerce_utc_datetime(value, field_name="timestamp").isoformat().replace("+00:00", "Z")


def _validate_record_id(value: str, *, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} must be a non-empty string.")
    text = str(value).strip()
    # AUDIT-FIX(#1): Reject traversal primitives and control characters before IDs reach the file-backed store.
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string.")
    if len(text) > _MAX_RECORD_ID_LENGTH:
        raise ValueError(f"{field_name} exceeds {_MAX_RECORD_ID_LENGTH} characters.")
    if text in {".", ".."} or "/" in text or "\\" in text or "\x00" in text:
        raise ValueError(f"{field_name} contains unsafe path characters.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ValueError(f"{field_name} contains control characters.")
    return text


def _safe_mapping(value: Any) -> dict[str, Any]:
    # AUDIT-FIX(#4): Older or partially-written store files may deserialize metadata/diagnostics as None or non-mappings; coerce them safely.
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _sanitize_reason(value: Any, *, fallback: str | None = None) -> str | None:
    text = "" if value is None else str(value)
    # AUDIT-FIX(#8): Strip control characters, collapse whitespace, and cap size before persisting or rendering operator-facing reasons.
    cleaned = "".join(" " if ord(ch) < 32 or ord(ch) == 127 else ch for ch in text)
    cleaned = " ".join(cleaned.split())
    if not cleaned and fallback is not None:
        cleaned = fallback
    if not cleaned:
        return None
    if len(cleaned) > _MAX_REASON_LENGTH:
        cleaned = cleaned[:_MAX_REASON_LENGTH].rstrip()
    return cleaned or (fallback if fallback is not None else None)


def _display_text(value: Any, *, fallback: str) -> str:
    return _sanitize_reason(value, fallback=fallback) or fallback


def _diagnostic_reason(diagnostics: dict[str, Any] | Mapping[str, Any] | Any) -> str | None:
    safe_diagnostics = _safe_mapping(diagnostics)
    for key in ("timeout_reason", "fallback_reason", "error", "message"):
        text = _sanitize_reason(safe_diagnostics.get(key), fallback=None)
        if text:
            return text
    return None


def _age_seconds_or_none(updated_at: Any, *, now: datetime, field_name: str) -> float | None:
    try:
        updated_at_utc = _coerce_utc_datetime(updated_at, field_name=field_name)
    except (TypeError, ValueError):
        return None
    return max(0.0, (now - updated_at_utc).total_seconds())


def _is_compile_terminal(status: CompileRunStatusRecord) -> bool:
    return status.phase in _TERMINAL_COMPILE_PHASES or status.completed_at is not None


def _is_execution_terminal(record: ExecutionRunStatusRecord) -> bool:
    return record.status in _TERMINAL_EXECUTION_STATUSES or record.completed_at is not None


def _is_stale(updated_at: Any, *, now: datetime, threshold_seconds: float, field_name: str) -> bool:
    age_seconds = _age_seconds_or_none(updated_at, now=now, field_name=field_name)
    return age_seconds is None or age_seconds >= threshold_seconds


def _build_unreadable_compile_row(status: Any, exc: Exception, *, threshold_seconds: float) -> "StaleCompileRunRow":
    return StaleCompileRunRow(
        job_id=_display_text(getattr(status, "job_id", None), fallback="<unknown>"),
        phase=_display_text(getattr(status, "phase", None), fallback="unknown"),
        age_seconds=threshold_seconds,
        reason=_sanitize_reason(f"unreadable compile status: {exc}", fallback="unreadable compile status"),
    )


def _build_unreadable_execution_row(record: Any, exc: Exception, *, threshold_seconds: float) -> "StaleExecutionRunRow":
    version_value = getattr(record, "version", 0)
    try:
        version = int(version_value)
    except (TypeError, ValueError):
        version = 0
    return StaleExecutionRunRow(
        run_id=_display_text(getattr(record, "run_id", None), fallback="<unknown>"),
        run_kind=_display_text(getattr(record, "run_kind", None), fallback="unknown"),
        skill_id=_display_text(getattr(record, "skill_id", None), fallback="unknown"),
        version=version,
        age_seconds=threshold_seconds,
        status=_display_text(getattr(record, "status", None), fallback="unknown"),
        reason=_sanitize_reason(f"unreadable execution status: {exc}", fallback="unreadable execution status"),
    )


@dataclass(frozen=True, slots=True)
class SelfCodingWatchdogThresholds:
    """Describe when compile or execution work should be treated as stale."""

    stale_compile_seconds: float = 900.0
    stale_execution_seconds: float = 900.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stale_compile_seconds",
            _validate_threshold_seconds(self.stale_compile_seconds, field_name="stale_compile_seconds"),
        )  # AUDIT-FIX(#7): Fail fast on negative/NaN/infinite compile thresholds so operator cleanup boundaries stay meaningful.
        object.__setattr__(
            self,
            "stale_execution_seconds",
            _validate_threshold_seconds(self.stale_execution_seconds, field_name="stale_execution_seconds"),
        )  # AUDIT-FIX(#7): Fail fast on negative/NaN/infinite execution thresholds so operator cleanup boundaries stay meaningful.


@dataclass(frozen=True, slots=True)
class StaleCompileRunRow:
    """Summarize one stale compile record for the operator UI."""

    job_id: str
    phase: str
    age_seconds: float
    skill_id: str | None = None
    skill_name: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class StaleExecutionRunRow:
    """Summarize one stale sandbox/retest run for the operator UI."""

    run_id: str
    run_kind: str
    skill_id: str
    version: int
    age_seconds: float
    status: str
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class SelfCodingWatchdogSnapshot:
    """Hold the operator-facing stale-run picture at one moment in time."""

    stale_compile_runs: tuple[StaleCompileRunRow, ...] = ()
    stale_execution_runs: tuple[StaleExecutionRunRow, ...] = ()


class SelfCodingRunWatchdog:
    """Compute stale compile and execution rows from the file-backed store."""

    def __init__(
        self,
        *,
        store: SelfCodingStore,
        thresholds: SelfCodingWatchdogThresholds | None = None,
    ) -> None:
        self.store = store
        self.thresholds = thresholds if thresholds is not None else SelfCodingWatchdogThresholds()

    def build_snapshot(self, *, now: datetime | None = None) -> SelfCodingWatchdogSnapshot:
        """Return stale compile and execution rows for the current operator page."""

        effective_now = _utc_now() if now is None else _coerce_utc_datetime(now, field_name="now")  # AUDIT-FIX(#2): Normalize caller-provided `now` values to UTC.
        return SelfCodingWatchdogSnapshot(
            stale_compile_runs=self._stale_compile_rows(effective_now),
            stale_execution_runs=self._stale_execution_rows(effective_now),
        )

    def _stale_compile_rows(self, now: datetime) -> tuple[StaleCompileRunRow, ...]:
        rows: list[StaleCompileRunRow] = []
        try:
            statuses = tuple(self.store.list_compile_statuses())
        except Exception:
            _LOGGER.exception("Failed to list compile statuses for the self-coding watchdog snapshot.")
            return ()  # AUDIT-FIX(#3): Degrade to an empty section instead of crashing the entire operator page on store read failures.
        for status in statuses:
            try:
                if _is_compile_terminal(status):
                    continue
                age_seconds = _age_seconds_or_none(
                    getattr(status, "updated_at", None),
                    now=now,
                    field_name=f"compile status {_display_text(getattr(status, 'job_id', None), fallback='<unknown>')}.updated_at",
                )
                if age_seconds is None:
                    raise ValueError("missing or invalid updated_at")
                if age_seconds < self.thresholds.stale_compile_seconds:
                    continue
                job = None
                try:
                    safe_job_id = _validate_record_id(getattr(status, "job_id", ""), field_name="job_id")
                except ValueError:
                    safe_job_id = None
                if safe_job_id is not None:
                    try:
                        job = self.store.load_job(safe_job_id)
                    except FileNotFoundError:
                        job = None
                    except Exception:
                        _LOGGER.exception("Failed to load linked self-coding job %s while building the watchdog snapshot.", safe_job_id)
                        job = None
                rows.append(
                    StaleCompileRunRow(
                        job_id=_display_text(getattr(status, "job_id", None), fallback="<unknown>"),
                        phase=_display_text(getattr(status, "phase", None), fallback="unknown"),
                        age_seconds=age_seconds,
                        skill_id=None if job is None else getattr(job, "skill_id", None),
                        skill_name=None if job is None else getattr(job, "skill_name", None),
                        reason=_sanitize_reason(getattr(status, "error_message", None), fallback=None)
                        or _diagnostic_reason(getattr(status, "diagnostics", None)),
                    )
                )
            except Exception as exc:
                _LOGGER.exception("Failed to inspect one compile status while building the self-coding watchdog snapshot.")
                rows.append(_build_unreadable_compile_row(status, exc, threshold_seconds=self.thresholds.stale_compile_seconds))  # AUDIT-FIX(#3): Surface corrupt records as operator-visible stale rows instead of losing them silently.
        rows.sort(key=lambda row: (-row.age_seconds, row.job_id))  # AUDIT-FIX(#9): Keep operator lists deterministic so repeated refreshes do not reshuffle stale runs.
        return tuple(rows)

    def _stale_execution_rows(self, now: datetime) -> tuple[StaleExecutionRunRow, ...]:
        rows: list[StaleExecutionRunRow] = []
        try:
            records = tuple(self.store.list_execution_runs())
        except Exception:
            _LOGGER.exception("Failed to list execution runs for the self-coding watchdog snapshot.")
            return ()  # AUDIT-FIX(#3): Degrade to an empty section instead of crashing the entire operator page on store read failures.
        for record in records:
            try:
                if _is_execution_terminal(record):
                    continue
                age_seconds = _age_seconds_or_none(
                    getattr(record, "updated_at", None),
                    now=now,
                    field_name=f"execution run {_display_text(getattr(record, 'run_id', None), fallback='<unknown>')}.updated_at",
                )
                if age_seconds is None:
                    raise ValueError("missing or invalid updated_at")
                if age_seconds < self.thresholds.stale_execution_seconds:
                    continue
                metadata = _safe_mapping(getattr(record, "metadata", None))
                rows.append(
                    StaleExecutionRunRow(
                        run_id=_display_text(getattr(record, "run_id", None), fallback="<unknown>"),
                        run_kind=_display_text(getattr(record, "run_kind", None), fallback="unknown"),
                        skill_id=_display_text(getattr(record, "skill_id", None), fallback="unknown"),
                        version=int(getattr(record, "version", 0)),
                        age_seconds=age_seconds,
                        status=_display_text(getattr(record, "status", None), fallback="unknown"),
                        reason=_sanitize_reason(getattr(record, "reason", None), fallback=None)
                        or _diagnostic_reason(metadata),
                    )
                )
            except Exception as exc:
                _LOGGER.exception("Failed to inspect one execution run while building the self-coding watchdog snapshot.")
                rows.append(_build_unreadable_execution_row(record, exc, threshold_seconds=self.thresholds.stale_execution_seconds))  # AUDIT-FIX(#3): Surface corrupt records as operator-visible stale rows instead of losing them silently.
        rows.sort(key=lambda row: (-row.age_seconds, row.run_id))  # AUDIT-FIX(#9): Keep operator lists deterministic so repeated refreshes do not reshuffle stale runs.
        return tuple(rows)


def cleanup_stale_compile_status(
    *,
    store: SelfCodingStore,
    job_id: str,
    reason: str,
    thresholds: SelfCodingWatchdogThresholds | None = None,
    now: datetime | None = None,
) -> CompileRunStatusRecord:
    """Mark one stale compile record as aborted and fail the linked job when still eligible."""

    effective_now = _utc_now() if now is None else _coerce_utc_datetime(now, field_name="now")
    safe_job_id = _validate_record_id(job_id, field_name="job_id")  # AUDIT-FIX(#1): Validate IDs before they reach any file-backed store path construction.
    effective_thresholds = thresholds if thresholds is not None else SelfCodingWatchdogThresholds()
    cleanup_reason = _sanitize_reason(reason, fallback=_DEFAULT_CLEANUP_REASON) or _DEFAULT_CLEANUP_REASON
    status = store.load_compile_status(safe_job_id)
    if _is_compile_terminal(status) or not _is_stale(
        getattr(status, "updated_at", None),
        now=effective_now,
        threshold_seconds=effective_thresholds.stale_compile_seconds,
        field_name=f"compile status {safe_job_id}.updated_at",
    ):
        return status  # AUDIT-FIX(#5): Make operator cleanup idempotent and bounded so terminal or freshly-updated runs cannot be overwritten during races.
    diagnostics = _safe_mapping(getattr(status, "diagnostics", None))  # AUDIT-FIX(#4): Tolerate legacy or partially-written diagnostics payloads during cleanup.
    diagnostics["cleanup_reason"] = cleanup_reason
    diagnostics["cleanup_at"] = _isoformat_z(effective_now)
    updated_status = replace(
        status,
        phase="aborted",
        updated_at=effective_now,
        completed_at=effective_now,
        error_message=cleanup_reason,
        diagnostics=diagnostics,
    )
    original_job = None
    updated_job = None
    try:
        original_job = store.load_job(safe_job_id)
    except FileNotFoundError:
        original_job = None
    except Exception:
        _LOGGER.exception("Failed to load linked self-coding job %s during stale compile cleanup.", safe_job_id)
        original_job = None
    if original_job is not None and original_job.status == CompileJobStatus.COMPILING:
        updated_job = replace(
            original_job,
            status=CompileJobStatus.FAILED,
            updated_at=effective_now,
            last_error=cleanup_reason,
        )
        store.save_job(updated_job)  # AUDIT-FIX(#6): Persist the linked job transition before flipping the compile status so a later failure cannot hide an apparently active job.
    try:
        return store.save_compile_status(updated_status)
    except Exception:
        if updated_job is not None and original_job is not None:
            try:
                store.save_job(original_job)
            except Exception:
                _LOGGER.exception("Failed to roll back linked self-coding job %s after compile-status cleanup failure.", safe_job_id)
        raise  # AUDIT-FIX(#6): Best-effort rollback narrows partial-write windows on file-backed state without changing the store contract.


def cleanup_stale_execution_run(
    *,
    store: SelfCodingStore,
    run_id: str,
    reason: str,
    thresholds: SelfCodingWatchdogThresholds | None = None,
    now: datetime | None = None,
) -> ExecutionRunStatusRecord:
    """Mark one stale execution record as cleaned while keeping the audit trail."""

    effective_now = _utc_now() if now is None else _coerce_utc_datetime(now, field_name="now")
    safe_run_id = _validate_record_id(run_id, field_name="run_id")  # AUDIT-FIX(#1): Validate IDs before they reach any file-backed store path construction.
    effective_thresholds = thresholds if thresholds is not None else SelfCodingWatchdogThresholds()
    cleanup_reason = _sanitize_reason(reason, fallback=_DEFAULT_CLEANUP_REASON) or _DEFAULT_CLEANUP_REASON
    record = store.load_execution_run(safe_run_id)
    if _is_execution_terminal(record) or not _is_stale(
        getattr(record, "updated_at", None),
        now=effective_now,
        threshold_seconds=effective_thresholds.stale_execution_seconds,
        field_name=f"execution run {safe_run_id}.updated_at",
    ):
        return record  # AUDIT-FIX(#5): Make operator cleanup idempotent and bounded so terminal or freshly-updated runs cannot be overwritten during races.
    metadata = _safe_mapping(getattr(record, "metadata", None))  # AUDIT-FIX(#4): Tolerate legacy or partially-written metadata payloads during cleanup.
    metadata["cleanup_at"] = _isoformat_z(effective_now)
    metadata["cleanup_reason"] = cleanup_reason
    return store.save_execution_run(
        replace(
            record,
            status="cleaned",
            reason=cleanup_reason,
            updated_at=effective_now,
            completed_at=effective_now,
            metadata=metadata,
        )
    )


__all__ = [
    "SelfCodingRunWatchdog",
    "SelfCodingWatchdogSnapshot",
    "SelfCodingWatchdogThresholds",
    "cleanup_stale_compile_status",
    "cleanup_stale_execution_run",
]