"""Hold a Twinr runtime in a stable error state after fatal loop failures.

This module keeps the Pi runtime process alive long enough for the display
companion and supervisor to observe a stable error state instead of bouncing
through repeated child exits. It does not recover the failed loop; it refreshes
the runtime snapshot and optional supervisor heartbeats while an operator-visible
error is active.
"""

# CHANGELOG: 2026-03-28
# BUG-1: Reassert the runtime error state during the hold loop so concurrent
#        writers cannot silently clear the intended stable `error` status.
# BUG-2: Make snapshot refresh resilient to partially initialized memory fields
#        such as `None`, which previously could break refreshes via `tuple(None)`.
# BUG-3: Sanitize non-finite or invalid timing inputs to avoid `time.sleep(inf)`
#        crashes or stalled refresh cadences during the very path meant to stay up.
# BUG-4: Rate-limit repeated secondary-fault logging so a broken snapshot path
#        cannot spam logs every second and wear SD storage on long error holds.
# SEC-1: Sanitize, truncate, and redact operator-visible error text before
#        emitting it, persisting it, or forwarding it to supervisor status hooks.
# SEC-2: Neutralize control sequences in error text to prevent terminal/log
#        injection in journald, consoles, and display-adjacent operator tooling.
# IMP-1: Add zero-dependency systemd `sd_notify` STATUS/WATCHDOG support using
#        `NOTIFY_SOCKET`/`WATCHDOG_USEC`, which is a practical 2026 Pi default.
# IMP-2: Add optional `stop_event`/`should_stop` for interruptible, supervisor-
#        controlled exits instead of forcing signal-only termination.
# IMP-3: Freeze mutable snapshot fields with best-effort cloning before save to
#        reduce drift from concurrent mutation while the process is parked.

from __future__ import annotations

from collections.abc import Callable
import copy
import logging
import math
import os
import re
import socket
import threading
import time

from twinr.agent.base_agent.runtime.runtime import TwinrRuntime


_LOGGER = logging.getLogger(__name__)
_DEFAULT_REFRESH_INTERVAL_S = 1.0
_DEFAULT_REPEATED_LOG_INTERVAL_S = 60.0
_DEFAULT_STATUS_INTERVAL_S = 5.0
_MIN_REFRESH_INTERVAL_S = 0.25
_MIN_SLEEP_S = 0.01
_MAX_ERROR_MESSAGE_CHARS = 240
_ERROR_STATUS = "error"

_ANSI_ESCAPE_RE = re.compile(
    r"""
    (?:\x1B[@-Z\\-_])                # 7-bit C1 Fe
    |(?:\x1B\[ [0-?]* [ -/]* [@-~])  # CSI ... Cmd
    |(?:\x1B\] .*? (?:\x07|\x1B\\))  # OSC ... BEL/ST
    """,
    re.VERBOSE,
)
_SENSITIVE_FIELD_RE = re.compile(
    r"(?i)\b("
    r"authorization|access[_ -]?token|refresh[_ -]?token|id[_ -]?token|"
    r"api[_ -]?key|secret|password|passwd|cookie|session(?:id)?|dsn"
    r")\b\s*[:=]\s*([^\s,;]+)"
)
_BEARER_TOKEN_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+\b")


def _default_emit(line: str) -> None:
    """Print one bounded error-hold telemetry line."""

    print(line, flush=True)


def _redact_sensitive_text(text: str) -> str:
    text = _SENSITIVE_FIELD_RE.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
    text = _BEARER_TOKEN_RE.sub("Bearer [REDACTED]", text)
    return text


def _sanitize_operator_text(text: str) -> str:
    """Return a compact operator-safe string for logs, displays, and status sinks."""

    cleaned = _ANSI_ESCAPE_RE.sub("", text or "")
    cleaned = "".join(char if char.isprintable() else " " for char in cleaned)
    cleaned = _redact_sensitive_text(" ".join(cleaned.split()).strip())
    if len(cleaned) > _MAX_ERROR_MESSAGE_CHARS:
        cleaned = f"{cleaned[: _MAX_ERROR_MESSAGE_CHARS - 1].rstrip()}…"
    return cleaned


def _normalize_error_message(error: BaseException | str) -> str:
    """Return one compact operator-safe error string."""

    raw_text = _sanitize_operator_text(str(error or ""))
    if isinstance(error, BaseException):
        name = error.__class__.__name__
        if raw_text and raw_text != name:
            return f"{name}: {raw_text}"
        return name
    return raw_text or "Twinr runtime failed during loop bootstrap."


def _coerce_non_negative_seconds(
    value: float | None,
    *,
    default: float | None,
    minimum: float,
) -> float | None:
    """Parse a finite timeout/interval value without throwing during error hold."""

    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(parsed):
        return default
    return max(minimum, parsed)


def _coerce_optional_duration_s(value: float | None) -> float | None:
    return _coerce_non_negative_seconds(value, default=None, minimum=0.0)


def _coerce_refresh_interval_s(value: float | None) -> float:
    coerced = _coerce_non_negative_seconds(
        value,
        default=_DEFAULT_REFRESH_INTERVAL_S,
        minimum=_MIN_REFRESH_INTERVAL_S,
    )
    return _DEFAULT_REFRESH_INTERVAL_S if coerced is None else coerced


def _current_status_value(runtime: TwinrRuntime) -> str:
    value = getattr(getattr(runtime, "status", None), "value", None)
    return str(value or _ERROR_STATUS)


def _ensure_runtime_error_state(runtime: TwinrRuntime, message: str) -> None:
    if _current_status_value(runtime) == _ERROR_STATUS:
        return
    runtime.fail(message)


def _tuple_or_empty(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, set):
        return tuple(value)
    if isinstance(value, dict):
        return tuple(value.items())
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return (value,)


def _clone_snapshot_value(value: object) -> object:
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _snapshot_payload(runtime: TwinrRuntime, *, error_message: str) -> dict[str, object]:
    memory = getattr(runtime, "memory", None)
    return {
        "status": _ERROR_STATUS,
        "memory_turns": _tuple_or_empty(getattr(memory, "turns", ())),
        "memory_raw_tail": _tuple_or_empty(getattr(memory, "raw_tail", ())),
        "memory_ledger": _tuple_or_empty(getattr(memory, "ledger", ())),
        "memory_search_results": _tuple_or_empty(getattr(memory, "search_results", ())),
        "memory_state": _clone_snapshot_value(getattr(memory, "state", None)),
        "last_transcript": _clone_snapshot_value(getattr(runtime, "last_transcript", None)),
        "last_response": _clone_snapshot_value(getattr(runtime, "last_response", None)),
        "error_message": error_message,
        "user_voice_status": _clone_snapshot_value(getattr(runtime, "user_voice_status", None)),
        "user_voice_confidence": _clone_snapshot_value(getattr(runtime, "user_voice_confidence", None)),
        "user_voice_checked_at": _clone_snapshot_value(getattr(runtime, "user_voice_checked_at", None)),
        "user_voice_user_id": _clone_snapshot_value(getattr(runtime, "user_voice_user_id", None)),
        "user_voice_user_display_name": _clone_snapshot_value(
            getattr(runtime, "user_voice_user_display_name", None)
        ),
        "user_voice_match_source": _clone_snapshot_value(getattr(runtime, "user_voice_match_source", None)),
    }


def _best_effort_refresh_snapshot(runtime: TwinrRuntime, *, error_message: str) -> None:
    """Persist the current runtime snapshot without hiding the primary failure."""

    persist = getattr(runtime, "_persist_snapshot", None)
    if callable(persist) and _current_status_value(runtime) == _ERROR_STATUS:
        persist(error_message=error_message)
        return
    snapshot_store = getattr(runtime, "snapshot_store", None)
    if snapshot_store is None:
        return
    snapshot_store.save(**_snapshot_payload(runtime, error_message=error_message))


class _RepeatAwareLogger:
    __slots__ = ("_interval_ns", "_states")

    def __init__(self, interval_s: float = _DEFAULT_REPEATED_LOG_INTERVAL_S) -> None:
        self._interval_ns = max(1, int(interval_s * 1_000_000_000))
        self._states: dict[str, tuple[int, int]] = {}

    def exception(self, key: str, message: str, exc: BaseException, *, now_ns: int) -> None:
        last_logged_at_ns, suppressed = self._states.get(key, (0, 0))
        if last_logged_at_ns and (now_ns - last_logged_at_ns) < self._interval_ns:
            self._states[key] = (last_logged_at_ns, suppressed + 1)
            return
        suffix = ""
        if suppressed:
            suffix = f" [suppressed {suppressed} repeats]"
        self._states[key] = (now_ns, 0)
        _LOGGER.exception(
            "%s%s",
            message,
            suffix,
            exc_info=(type(exc), exc, exc.__traceback__),
        )

    def clear(self, key: str) -> None:
        self._states.pop(key, None)


class _SystemdNotifier:
    __slots__ = ("_notify_socket", "_watchdog_interval_ns")

    def __init__(self, notify_socket: str, watchdog_interval_ns: int | None) -> None:
        self._notify_socket = notify_socket
        self._watchdog_interval_ns = watchdog_interval_ns

    @classmethod
    def from_environment(cls) -> _SystemdNotifier | None:
        notify_socket = os.environ.get("NOTIFY_SOCKET")
        if not notify_socket:
            return None

        watchdog_interval_ns: int | None = None
        watchdog_usec = os.environ.get("WATCHDOG_USEC")
        watchdog_pid = os.environ.get("WATCHDOG_PID")
        if watchdog_usec and (not watchdog_pid or watchdog_pid == str(os.getpid())):
            try:
                watchdog_usec_int = int(watchdog_usec)
            except (TypeError, ValueError, OverflowError):
                watchdog_usec_int = 0
            if watchdog_usec_int > 0:
                watchdog_interval_ns = watchdog_usec_int * 1_000

        return cls(notify_socket=notify_socket, watchdog_interval_ns=watchdog_interval_ns)

    @property
    def heartbeat_interval_ns(self) -> int:
        if self._watchdog_interval_ns is not None:
            return max(1, self._watchdog_interval_ns // 2)
        return int(_DEFAULT_STATUS_INTERVAL_S * 1_000_000_000)

    def notify_status(self, status_text: str, *, watchdog: bool) -> None:
        fields = [f"STATUS={_sanitize_operator_text(status_text)}"]
        if watchdog and self._watchdog_interval_ns is not None:
            fields.append("WATCHDOG=1")
        self._notify(*fields)

    def notify_stopping(self, status_text: str) -> None:
        self._notify("STOPPING=1", f"STATUS={_sanitize_operator_text(status_text)}")

    def _notify(self, *fields: str) -> None:
        address = self._notify_socket
        if address.startswith("@"):
            address = "\0" + address[1:]
        payload = "\n".join(field for field in fields if field).encode("utf-8", errors="replace")
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
            client.connect(address)
            client.sendall(payload)


def _resolve_monotonic_ns(
    monotonic: Callable[[], float],
    monotonic_ns: Callable[[], int] | None,
) -> Callable[[], int]:
    if monotonic_ns is not None:
        return monotonic_ns
    if monotonic is time.monotonic:
        return time.monotonic_ns
    return lambda: int(monotonic() * 1_000_000_000)


def _stop_requested(
    *,
    stop_event: threading.Event | None,
    should_stop: Callable[[], bool] | None,
) -> bool:
    if stop_event is not None and stop_event.is_set():
        return True
    if should_stop is not None:
        try:
            return bool(should_stop())
        except Exception:
            _LOGGER.exception("Error while evaluating error-hold stop condition.")
    return False


def hold_runtime_error_state(
    *,
    runtime: TwinrRuntime,
    error: BaseException | str,
    emit: Callable[[str], None] = _default_emit,
    sleep: Callable[[float], object] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    duration_s: float | None = None,
    refresh_interval_s: float = _DEFAULT_REFRESH_INTERVAL_S,
    stop_event: threading.Event | None = None,
    should_stop: Callable[[], bool] | None = None,
    monotonic_ns: Callable[[], int] | None = None,
    systemd_notify: bool = True,
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
        stop_event: Optional interruptible shutdown signal for cooperative exits.
        should_stop: Optional predicate evaluated between waits for shutdown.
        monotonic_ns: Optional nanosecond clock for high-uptime precision.
        systemd_notify: When `True`, emit best-effort `sd_notify` status and
            watchdog heartbeats if `NOTIFY_SOCKET` is present in the environment.

    Returns:
        `1` when a bounded hold duration completes or a cooperative stop is requested.
    """

    clock_ns = _resolve_monotonic_ns(monotonic, monotonic_ns)
    message = _normalize_error_message(error)
    interval_s = _coerce_refresh_interval_s(refresh_interval_s)
    interval_ns = int(interval_s * 1_000_000_000)
    bounded_duration_s = _coerce_optional_duration_s(duration_s)
    bounded_duration_ns = (
        None if bounded_duration_s is None else int(bounded_duration_s * 1_000_000_000)
    )
    log_state = _RepeatAwareLogger()
    notifier = _SystemdNotifier.from_environment() if systemd_notify else None

    try:
        runtime.fail(message)
    except Exception as exc:
        log_state.exception(
            "runtime.fail",
            "Failed to enter runtime error state before starting the error hold.",
            exc,
            now_ns=clock_ns(),
        )

    try:
        emit(f"status={_ERROR_STATUS}")
        emit(f"error={message}")
    except Exception as exc:
        log_state.exception(
            "emit",
            "Failed to emit runtime error-hold telemetry.",
            exc,
            now_ns=clock_ns(),
        )

    started_at_ns = clock_ns()
    next_refresh_at_ns = started_at_ns
    next_status_at_ns = started_at_ns
    status_text = f"Twinr error-hold active: {message}"

    while True:
        now_ns = clock_ns()

        if now_ns >= next_refresh_at_ns:
            try:
                _ensure_runtime_error_state(runtime, message)
                log_state.clear("runtime.fail")
            except Exception as exc:
                log_state.exception(
                    "runtime.fail",
                    "Failed to reassert runtime error state during the error hold.",
                    exc,
                    now_ns=now_ns,
                )
            try:
                _best_effort_refresh_snapshot(runtime, error_message=message)
                log_state.clear("snapshot")
            except Exception as exc:
                log_state.exception(
                    "snapshot",
                    "Failed to refresh runtime snapshot during the error hold.",
                    exc,
                    now_ns=now_ns,
                )
            next_refresh_at_ns = now_ns + interval_ns

        if notifier is not None and now_ns >= next_status_at_ns:
            try:
                notifier.notify_status(status_text, watchdog=True)
                log_state.clear("systemd_notify")
            except Exception as exc:
                log_state.exception(
                    "systemd_notify",
                    "Failed to update systemd error-hold status/watchdog state.",
                    exc,
                    now_ns=now_ns,
                )
            next_status_at_ns = now_ns + notifier.heartbeat_interval_ns

        if bounded_duration_ns is not None and (now_ns - started_at_ns) >= bounded_duration_ns:
            if notifier is not None:
                try:
                    notifier.notify_stopping(f"Twinr error-hold finished after {bounded_duration_s:.3f}s")
                except Exception as exc:
                    log_state.exception(
                        "systemd_notify",
                        "Failed to send systemd stopping notification for the error hold.",
                        exc,
                        now_ns=clock_ns(),
                    )
            return 1

        if _stop_requested(stop_event=stop_event, should_stop=should_stop):
            if notifier is not None:
                try:
                    notifier.notify_stopping("Twinr error-hold stopped by supervisor request")
                except Exception as exc:
                    log_state.exception(
                        "systemd_notify",
                        "Failed to send systemd stopping notification for supervisor stop.",
                        exc,
                        now_ns=clock_ns(),
                    )
            return 1

        wake_at_ns = next_refresh_at_ns
        if notifier is not None:
            wake_at_ns = min(wake_at_ns, next_status_at_ns)
        if bounded_duration_ns is not None:
            wake_at_ns = min(wake_at_ns, started_at_ns + bounded_duration_ns)

        remaining_s = max(0.0, (wake_at_ns - clock_ns()) / 1_000_000_000)
        timeout_s = remaining_s if remaining_s > 0.0 else _MIN_SLEEP_S

        if stop_event is not None:
            if stop_event.wait(timeout_s):
                continue
        else:
            sleep(timeout_s)


__all__ = ["hold_runtime_error_state"]