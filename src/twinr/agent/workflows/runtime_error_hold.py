"""Hold a Twinr runtime in a stable error state after fatal loop failures.

This module keeps the Pi runtime process alive long enough for the display
companion and supervisor to observe a stable error state instead of bouncing
through repeated child exits. It does not recover the failed loop; it only
refreshes the runtime snapshot while an operator-visible error is active.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
import time

from twinr.agent.base_agent.runtime import TwinrRuntime


_LOGGER = logging.getLogger(__name__)
_DEFAULT_REFRESH_INTERVAL_S = 1.0
_MIN_REFRESH_INTERVAL_S = 0.25


def _default_emit(line: str) -> None:
    """Print one bounded error-hold telemetry line."""

    print(line, flush=True)


def _normalize_error_message(error: BaseException | str) -> str:
    """Return one compact operator-safe error string."""

    text = " ".join(str(error or "").split()).strip()
    if not text and isinstance(error, BaseException):
        text = error.__class__.__name__
    return text or "Twinr runtime failed during loop bootstrap."


def _best_effort_refresh_snapshot(runtime: TwinrRuntime, *, error_message: str) -> None:
    """Persist the current runtime snapshot without raising on secondary faults."""

    persist = getattr(runtime, "_persist_snapshot", None)
    if callable(persist):
        try:
            persist(error_message=error_message)
            return
        except Exception:
            _LOGGER.exception("Failed to refresh runtime snapshot through _persist_snapshot().")
    snapshot_store = getattr(runtime, "snapshot_store", None)
    memory = getattr(runtime, "memory", None)
    if snapshot_store is None or memory is None:
        return
    try:
        snapshot_store.save(
            status=str(getattr(getattr(runtime, "status", None), "value", "error") or "error"),
            memory_turns=tuple(getattr(memory, "turns", ())),
            memory_raw_tail=tuple(getattr(memory, "raw_tail", ())),
            memory_ledger=tuple(getattr(memory, "ledger", ())),
            memory_search_results=tuple(getattr(memory, "search_results", ())),
            memory_state=getattr(memory, "state", None),
            last_transcript=getattr(runtime, "last_transcript", None),
            last_response=getattr(runtime, "last_response", None),
            error_message=error_message,
            user_voice_status=getattr(runtime, "user_voice_status", None),
            user_voice_confidence=getattr(runtime, "user_voice_confidence", None),
            user_voice_checked_at=getattr(runtime, "user_voice_checked_at", None),
            user_voice_user_id=getattr(runtime, "user_voice_user_id", None),
            user_voice_user_display_name=getattr(runtime, "user_voice_user_display_name", None),
            user_voice_match_source=getattr(runtime, "user_voice_match_source", None),
        )
    except Exception:
        _LOGGER.exception("Failed to refresh runtime snapshot through snapshot_store.save().")


def hold_runtime_error_state(
    *,
    runtime: TwinrRuntime,
    error: BaseException | str,
    emit: Callable[[str], None] = _default_emit,
    sleep: Callable[[float], object] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    duration_s: float | None = None,
    refresh_interval_s: float = _DEFAULT_REFRESH_INTERVAL_S,
) -> int:
    """Keep the runtime alive in `error` so display and supervisor stay stable.

    Args:
        runtime: Live runtime instance whose snapshot must remain fresh.
        error: Fatal loop/bootstrap failure that should be surfaced.
        emit: Best-effort telemetry sink.
        sleep: Injected sleep primitive for tests.
        monotonic: Injected clock for tests.
        duration_s: Optional bounded hold duration. `None` means hold forever.
        refresh_interval_s: Snapshot refresh cadence while holding the error.

    Returns:
        `1` when a bounded hold duration completes.
    """

    message = _normalize_error_message(error)
    interval_s = max(_MIN_REFRESH_INTERVAL_S, float(refresh_interval_s or _DEFAULT_REFRESH_INTERVAL_S))
    bounded_duration = None if duration_s is None else max(0.0, float(duration_s))
    try:
        runtime.fail(message)
    except Exception:
        _LOGGER.exception("Failed to enter runtime error state before starting the error hold.")
    try:
        emit(f"status={getattr(getattr(runtime, 'status', None), 'value', 'error')}")
        emit(f"error={message}")
    except Exception:
        _LOGGER.exception("Failed to emit runtime error-hold telemetry.")

    started_at = monotonic()
    next_refresh_at = started_at
    while True:
        now = monotonic()
        if now >= next_refresh_at:
            _best_effort_refresh_snapshot(runtime, error_message=message)
            next_refresh_at = now + interval_s
        if bounded_duration is not None and (now - started_at) >= bounded_duration:
            return 1
        remaining_s = interval_s
        if bounded_duration is not None:
            remaining_s = min(remaining_s, max(0.0, bounded_duration - (now - started_at)))
            if remaining_s <= 0.0:
                return 1
        sleep(max(0.05, remaining_s))


__all__ = ["hold_runtime_error_state"]
