"""Wake signals for in-process companions and standalone display-loop processes."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import errno
import os
import signal
from threading import Event, Lock, current_thread, main_thread
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from twinr.agent.base_agent.config import TwinrConfig


_LISTENERS: set[Event] = set()
_LOCK = Lock()
_DISPLAY_COMPANION_WAKE_SIGNAL = signal.SIGUSR1
_PROCESS_WAKE_EVENT = Event()
_PROCESS_WAKE_LISTENER_COUNT = 0
_PROCESS_WAKE_SIGNAL_INSTALLED = False
_PROCESS_WAKE_PREVIOUS_HANDLER: object = signal.SIG_DFL
_PIDFD_UNSUPPORTED_ERRNOS = {
    getattr(errno, "ENOSYS", 38),
    getattr(errno, "EINVAL", 22),
    getattr(errno, "EPERM", 1),
    getattr(errno, "ENODEV", 19),
    getattr(errno, "EOPNOTSUPP", 95),
}
_PIDFD_MISSING_ERRNOS = {
    getattr(errno, "ESRCH", 3),
    getattr(errno, "EBADF", 9),
}


def _handle_display_process_wakeup_signal(_signum: int, _frame) -> None:
    """Mark the current process display loop as needing immediate wakeup."""

    _PROCESS_WAKE_EVENT.set()


def _signal_display_process(pid: int) -> bool:
    """Send one exact wake signal to a live display-loop process."""

    resolved_pid = int(pid)
    if resolved_pid <= 0:
        return False

    pidfd_open = getattr(os, "pidfd_open", None)
    pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
    pidfd: int | None = None
    if callable(pidfd_open):
        try:
            pidfd = int(pidfd_open(resolved_pid, 0))
        except OSError as exc:
            if exc.errno not in (_PIDFD_UNSUPPORTED_ERRNOS | _PIDFD_MISSING_ERRNOS):
                raise
            pidfd = None
        if pidfd is not None and callable(pidfd_send_signal):
            try:
                pidfd_send_signal(pidfd, _DISPLAY_COMPANION_WAKE_SIGNAL)
                return True
            except ProcessLookupError:
                return False
            except OSError as exc:
                if exc.errno in _PIDFD_MISSING_ERRNOS:
                    return False
                if exc.errno not in _PIDFD_UNSUPPORTED_ERRNOS:
                    raise
            finally:
                os.close(pidfd)
    try:
        os.kill(resolved_pid, _DISPLAY_COMPANION_WAKE_SIGNAL)
    except ProcessLookupError:
        return False
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        raise
    return True


def _display_loop_owner_pid(config: object) -> int | None:
    """Return the current display-loop owner PID from the authoritative lock."""

    from twinr.ops.locks import loop_lock_owner

    return loop_lock_owner(cast("TwinrConfig", config), "display-loop")


@contextmanager
def register_display_companion_wakeup_listener(event: Event) -> Iterator[None]:
    """Register one companion wake listener for the current process."""

    with _LOCK:
        _LISTENERS.add(event)
    try:
        yield
    finally:
        with _LOCK:
            _LISTENERS.discard(event)


@contextmanager
def register_display_process_wakeup_listener() -> Iterator[Event]:
    """Register the current process to receive direct display-loop wake signals."""

    global _PROCESS_WAKE_LISTENER_COUNT
    global _PROCESS_WAKE_PREVIOUS_HANDLER
    global _PROCESS_WAKE_SIGNAL_INSTALLED

    with _LOCK:
        if _PROCESS_WAKE_LISTENER_COUNT == 0:
            _PROCESS_WAKE_EVENT.clear()
        _PROCESS_WAKE_LISTENER_COUNT += 1
        if not _PROCESS_WAKE_SIGNAL_INSTALLED and current_thread() is main_thread():
            _PROCESS_WAKE_PREVIOUS_HANDLER = signal.getsignal(_DISPLAY_COMPANION_WAKE_SIGNAL)
            signal.signal(
                _DISPLAY_COMPANION_WAKE_SIGNAL,
                _handle_display_process_wakeup_signal,
            )
            _PROCESS_WAKE_SIGNAL_INSTALLED = True
    try:
        yield _PROCESS_WAKE_EVENT
    finally:
        with _LOCK:
            _PROCESS_WAKE_LISTENER_COUNT = max(0, _PROCESS_WAKE_LISTENER_COUNT - 1)
            if _PROCESS_WAKE_LISTENER_COUNT == 0:
                if _PROCESS_WAKE_SIGNAL_INSTALLED:
                    signal.signal(
                        _DISPLAY_COMPANION_WAKE_SIGNAL,
                        cast(signal.Handlers, _PROCESS_WAKE_PREVIOUS_HANDLER),
                    )
                    _PROCESS_WAKE_SIGNAL_INSTALLED = False
                    _PROCESS_WAKE_PREVIOUS_HANDLER = signal.SIG_DFL
                _PROCESS_WAKE_EVENT.clear()


def request_display_companion_wakeup(config: object | None = None) -> bool:
    """Wake live display companions and, when configured, the standalone display loop."""

    with _LOCK:
        listeners = tuple(_LISTENERS)
        has_process_listener = _PROCESS_WAKE_LISTENER_COUNT > 0
    if has_process_listener:
        _PROCESS_WAKE_EVENT.set()
    for listener in listeners:
        listener.set()
    woke_local = has_process_listener or bool(listeners)

    if config is None:
        return woke_local

    owner_pid = _display_loop_owner_pid(config)
    if owner_pid is None:
        return woke_local
    if owner_pid == os.getpid():
        _PROCESS_WAKE_EVENT.set()
        return True
    return _signal_display_process(owner_pid) or woke_local
