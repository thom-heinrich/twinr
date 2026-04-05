# CHANGELOG: 2026-03-28
# BUG-1: Fixed unsafe teardown where loop.sleep/loop.stop_requested were restored even if the companion thread was still alive.
# BUG-2: Fixed lock/initialization race by moving loop construction under the display-loop lock, so hardware init no longer happens before exclusive ownership.
# BUG-3: Fixed stop-signal regression by composing stop_event with the loop's original stop_requested() instead of replacing it.
# BUG-4: Fixed the best-effort telemetry path so emit failures no longer crash or mask the caller's real exception.
# BUG-5: Fixed overly aggressive shutdown timing by replacing the hard 5s cap with a hardware-aware timeout that can cover real e-paper refresh latencies.
# SEC-1: Prevented accidental contextvars inheritance into the long-lived companion thread on Python 3.14+ by explicitly starting it with an empty Context when supported.
# SEC-2: Sanitized telemetry lines to a single line to reduce log-forging / parser-confusion risk from exception text.
# IMP-1: Added best-effort non-blocking / cancellable lock acquisition for modern lock libraries (for example filelock/portalocker-style acquire APIs) to avoid indefinite startup stalls under lock contention.
# IMP-2: Added a startup barrier so "started" is emitted only after the companion actually owns the lock and the loop is ready to run.
# IMP-3: Moved lock ownership, loop patching, and loop restoration fully into the worker thread to keep lifecycle and cleanup in one place.

"""Run the status display loop as an optional companion thread.

Use this context manager around other runtime loops when Twinr should keep the
e-paper panel updated without requiring a separate process invocation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import Context
import inspect
import math
from threading import Event, Thread
from time import monotonic
from typing import Protocol

from twinr.agent.base_agent.config import TwinrConfig


class DisplayLoopLike(Protocol):
    """Describe the minimal interface required by the companion runner."""

    sleep: Callable[[float], object]
    stop_requested: Callable[[], bool]

    def run(self, *, duration_s: float | None = None) -> int: ...


EmitFn = Callable[[str], None]
LockFactoryFn = Callable[[TwinrConfig, str], object]
LockOwnerFn = Callable[[TwinrConfig, str], int | None]
LoopFactoryFn = Callable[[TwinrConfig], DisplayLoopLike]

_DISPLAY_LOOP_NAME = "display-loop"
_LOCK_BUSY = object()


def _default_emit(line: str) -> None:
    """Print a bounded telemetry line."""
    print(line, flush=True)


def _default_lock_factory(config: TwinrConfig, loop_name: str) -> object:
    """Build the loop-instance lock object for a display loop."""
    from twinr.ops.locks import loop_instance_lock

    return loop_instance_lock(config, loop_name)


def _default_lock_owner(config: TwinrConfig, loop_name: str) -> int | None:
    """Return the PID that currently owns the display-loop lock."""
    from twinr.ops.locks import loop_lock_owner

    return loop_lock_owner(config, loop_name)


def _default_loop_factory(config: TwinrConfig) -> DisplayLoopLike:
    """Build the default status-display loop from configuration."""
    from twinr.display.service import TwinrStatusDisplayLoop

    return TwinrStatusDisplayLoop.from_config(config)


def _safe_emit(emit: EmitFn, line: str) -> None:
    """Emit a single-line best-effort telemetry record."""
    safe_line = " ".join(str(line).splitlines()).strip()
    if not safe_line:
        return
    try:
        emit(safe_line)
    except Exception:
        return


def _record_memory_phase(
    config: TwinrConfig,
    *,
    label: str,
    owner_label: str | None = None,
    owner_detail: str | None = None,
    replace: bool = False,
) -> None:
    """Best-effort bridge into the shared streaming-memory attribution store."""

    try:
        from twinr.ops.process_memory import record_streaming_memory_phase

        record_streaming_memory_phase(
            config,
            label=label,
            owner_label=owner_label,
            owner_detail=owner_detail,
            replace=replace,
        )
    except Exception:
        return


def _exception_summary(exc: BaseException) -> str:
    """Format an exception compactly for best-effort telemetry."""
    return f"{type(exc).__name__}: {exc}"


def _config_float(
    config: TwinrConfig,
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Read a float config field defensively."""
    raw = getattr(config, name, default)
    try:
        value = float(raw if raw is not None else default)
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _sleep_quantum_seconds(config: TwinrConfig) -> float:
    """Return the cooperative sleep slice for responsive shutdown."""
    interval_s = _config_float(
        config,
        "display_poll_interval_s",
        0.5,
        minimum=0.05,
        maximum=60.0,
    )
    return max(0.05, min(0.25, interval_s))


def _startup_timeout_seconds(config: TwinrConfig) -> float:
    """Wait briefly for the worker to confirm that it really started."""
    interval_s = _config_float(
        config,
        "display_poll_interval_s",
        0.5,
        minimum=0.05,
        maximum=60.0,
    )
    return max(0.05, min(0.25, interval_s * 2.0))


def _join_timeout_seconds(config: TwinrConfig) -> float:
    """Choose a shutdown timeout that covers common e-paper refresh latencies."""
    poll_s = _config_float(
        config,
        "display_poll_interval_s",
        0.5,
        minimum=0.05,
        maximum=60.0,
    )
    configured_s = _config_float(
        config,
        "display_shutdown_timeout_s",
        15.0,
        minimum=1.0,
        maximum=60.0,
    )
    return max(configured_s, poll_s * 4.0)


def _callable_parameter_names(fn: Callable[..., object]) -> set[str]:
    """Best-effort callable signature inspection for lock adapters."""
    try:
        return set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return set()


def _looks_like_lock_contention(exc: BaseException) -> bool:
    """Classify common immediate-lock-failure exceptions conservatively."""
    name = exc.__class__.__name__.lower()
    module = exc.__class__.__module__.lower()
    text = str(exc).lower()

    if module.startswith("filelock") and name == "timeout":
        return True
    if module.startswith("portalocker") and (
        "alreadylocked" in name or "lockexception" in name or "wouldblock" in name
    ):
        return True
    if module.startswith("fasteners") and "timeout" in name:
        return True

    signals = (
        "already locked",
        "alreadylocked",
        "would block",
        "resource temporarily unavailable",
    )
    return any(signal in text for signal in signals)


def _try_acquire_lock_in_worker(
    lock: object,
    *,
    stop_event: Event,
) -> Callable[[], None] | object | None:
    """Try to acquire a lock without blocking forever.

    Returns:
        callable: release callback when acquisition succeeded through an acquire()/release() API
        _LOCK_BUSY: lock is currently held elsewhere
        None: unsupported, caller should fall back to ``with lock:``
    """
    acquire = getattr(lock, "acquire", None)
    release = getattr(lock, "release", None)
    if not callable(acquire) or not callable(release):
        return None

    params = _callable_parameter_names(acquire)
    if not params:
        return None

    kwargs: dict[str, object] = {}

    if "blocking" in params:
        kwargs["blocking"] = False
    elif "timeout" in params:
        kwargs["timeout"] = 0
        if "fail_when_locked" in params:
            kwargs["fail_when_locked"] = True
        if "check_interval" in params:
            kwargs["check_interval"] = 0.05
        if "poll_interval" in params:
            kwargs["poll_interval"] = 0.05
    elif "fail_when_locked" in params:
        kwargs["fail_when_locked"] = True
    else:
        return None

    if "cancel_check" in params:
        kwargs["cancel_check"] = stop_event.is_set

    try:
        acquired = acquire(**kwargs)
    except Exception as exc:
        if _looks_like_lock_contention(exc):
            return _LOCK_BUSY
        raise

    if isinstance(acquired, bool) and not acquired:
        return _LOCK_BUSY

    def _release() -> None:
        release()

    return _release


def _interruptible_sleep(
    *,
    original_sleep: Callable[[float], object],
    stop_event: Event,
    duration_s: float,
    quantum_s: float,
) -> None:
    """Sleep cooperatively without discarding the loop's original sleep callable."""
    try:
        remaining = float(duration_s)
    except (TypeError, ValueError):
        remaining = 0.0
    if not math.isfinite(remaining):
        remaining = 0.0

    if remaining <= 0:
        original_sleep(0.0)
        return

    deadline = monotonic() + remaining
    while True:
        if stop_event.is_set():
            return
        remaining = deadline - monotonic()
        if remaining <= 0:
            return
        original_sleep(min(quantum_s, remaining))


def _thread_kwargs_for_context_isolation() -> dict[str, object]:
    """Use an empty Context on Python versions that support Thread(context=...)."""
    try:
        params = inspect.signature(Thread).parameters
    except (TypeError, ValueError):
        params = {}
    if "context" in params:
        return {"context": Context()}
    return {}


@contextmanager
def optional_display_companion(
    config: TwinrConfig,
    *,
    enabled: bool,
    emit: EmitFn = _default_emit,
    loop_factory: LoopFactoryFn = _default_loop_factory,
    lock_owner: LockOwnerFn = _default_lock_owner,
    lock_factory: LockFactoryFn = _default_lock_factory,
) -> Iterator[None]:
    """Run the display loop in a background thread when enabled.

    If the display companion is disabled or another process already owns the
    display-loop lock, this context manager becomes a no-op.

    Args:
        config: Runtime configuration for lock lookup and loop construction.
        enabled: Whether the companion should start at all.
        emit: Best-effort telemetry sink for lifecycle events.
        loop_factory: Factory that builds the loop object to run.
        lock_owner: Callable that reports the current loop-lock owner PID.
        lock_factory: Factory that builds the display-lock object or context manager.

    Yields:
        ``None`` while the companion thread, if any, is active.
    """
    if not enabled:
        yield
        return

    if lock_owner(config, _DISPLAY_LOOP_NAME) is not None:
        yield
        return

    stop_event = Event()
    started_event = Event()
    finished_event = Event()
    state: dict[str, str] = {"status": "starting"}

    def _run_locked_loop() -> None:
        loop = loop_factory(config)
        original_sleep = loop.sleep
        original_stop_requested = loop.stop_requested
        try:
            loop.sleep = lambda duration_s: _interruptible_sleep(
                original_sleep=original_sleep,
                stop_event=stop_event,
                duration_s=duration_s,
                quantum_s=_sleep_quantum_seconds(config),
            )
            loop.stop_requested = lambda: stop_event.is_set() or bool(original_stop_requested())
            started_event.set()
            state["status"] = "started"
            _record_memory_phase(
                config,
                label="display_companion.thread_started",
                owner_label="display_companion.thread",
                owner_detail="Display companion thread acquired the display-loop lock and entered loop.run().",
                replace=True,
            )
            loop.run()
            state["status"] = "stopped"
        finally:
            loop.sleep = original_sleep
            loop.stop_requested = original_stop_requested

    def _run() -> None:
        try:
            lock = lock_factory(config, _DISPLAY_LOOP_NAME)
            release_lock = _try_acquire_lock_in_worker(lock, stop_event=stop_event)

            if release_lock is _LOCK_BUSY:
                state["status"] = "busy"
                _safe_emit(emit, "display_companion=busy")
                return

            if callable(release_lock):
                try:
                    if stop_event.is_set():
                        state["status"] = "stopped-before-start"
                        return
                    _run_locked_loop()
                finally:
                    release_lock()
                return

            with lock:
                if stop_event.is_set():
                    state["status"] = "stopped-before-start"
                    return
                _run_locked_loop()
        except Exception as exc:
            summary = _exception_summary(exc)
            if state.get("status") == "starting":
                state["status"] = "failed-before-start"
            else:
                state["status"] = "failed"
            _safe_emit(emit, f"display_companion=failed:{summary}")
        finally:
            finished_event.set()

    thread = Thread(
        target=_run,
        name="twinr-display-companion",
        daemon=True,
        **_thread_kwargs_for_context_isolation(),
    )
    thread.start()

    started_event.wait(timeout=_startup_timeout_seconds(config))
    if started_event.is_set():
        _safe_emit(emit, "display_companion=started")
    elif finished_event.is_set() and state.get("status") in {"busy", "failed-before-start"}:
        pass
    else:
        _safe_emit(emit, "display_companion=starting")

    try:
        yield
    finally:
        stop_event.set()
        try:
            thread.join(timeout=_join_timeout_seconds(config))
        except Exception as exc:
            _safe_emit(emit, f"display_companion=join-failed:{_exception_summary(exc)}")
        if thread.is_alive():
            _safe_emit(emit, "display_companion=stop-timeout")
