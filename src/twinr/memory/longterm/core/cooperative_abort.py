"""Cooperative abort scopes for long-term background operations.

Background long-term persistence runs inside worker threads that need a
graceful, bounded way to stop retry/backoff loops once shutdown has been
requested. This module exposes a tiny thread-local abort scope so lower-level
remote storage code can poll for that signal without importing runtime worker
internals.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
import threading
import time


_POLL_SLICE_S = 0.05
_THREAD_STATE = threading.local()


class LongTermOperationCancelledError(RuntimeError):
    """Signal that one cooperative long-term operation was aborted."""


@dataclass(frozen=True, slots=True)
class LongTermOperationAbortScope:
    """Carry one cooperative abort signal for the current worker thread."""

    should_abort: Callable[[], bool]
    wait_for_abort: Callable[[float], bool] | None = None
    label: str | None = None

    def abort_requested(self) -> bool:
        """Return whether the surrounding operation should stop now."""

        try:
            return bool(self.should_abort())
        except Exception:
            return False

    def wait(self, timeout_s: float) -> bool:
        """Wait up to one bounded timeout for an abort signal."""

        resolved_timeout_s = max(0.0, float(timeout_s))
        waiter = self.wait_for_abort
        if callable(waiter):
            try:
                return bool(waiter(resolved_timeout_s))
            except Exception:
                pass
        deadline = time.monotonic() + resolved_timeout_s
        while True:
            if self.abort_requested():
                return True
            remaining_timeout_s = deadline - time.monotonic()
            if remaining_timeout_s <= 0.0:
                return self.abort_requested()
            time.sleep(min(_POLL_SLICE_S, remaining_timeout_s))


def _scope_stack() -> list[LongTermOperationAbortScope]:
    stack = getattr(_THREAD_STATE, "abort_scope_stack", None)
    if isinstance(stack, list):
        return stack
    stack = []
    _THREAD_STATE.abort_scope_stack = stack
    return stack


@contextmanager
def longterm_operation_abort_scope(
    *,
    should_abort: Callable[[], bool],
    wait_for_abort: Callable[[float], bool] | None = None,
    label: str | None = None,
):
    """Install one thread-local cooperative abort scope for nested operations."""

    scope = LongTermOperationAbortScope(
        should_abort=should_abort,
        wait_for_abort=wait_for_abort,
        label=label,
    )
    stack = _scope_stack()
    stack.append(scope)
    try:
        yield scope
    finally:
        for index in range(len(stack) - 1, -1, -1):
            if stack[index] is scope:
                del stack[index]
                break


def current_longterm_operation_abort_scope() -> LongTermOperationAbortScope | None:
    """Return the active cooperative abort scope for the current thread."""

    stack = getattr(_THREAD_STATE, "abort_scope_stack", None)
    if isinstance(stack, list) and stack:
        return stack[-1]
    return None


def raise_if_longterm_operation_cancelled(reason: str) -> None:
    """Raise when the current worker thread was asked to abort."""

    scope = current_longterm_operation_abort_scope()
    if scope is not None and scope.abort_requested():
        raise LongTermOperationCancelledError(str(reason))


def sleep_with_longterm_operation_abort(delay_s: float, *, reason: str) -> None:
    """Sleep cooperatively, aborting early when shutdown was requested."""

    resolved_delay_s = max(0.0, float(delay_s))
    if resolved_delay_s <= 0.0:
        return
    scope = current_longterm_operation_abort_scope()
    if scope is None:
        time.sleep(resolved_delay_s)
        return
    if scope.abort_requested() or scope.wait(resolved_delay_s):
        raise LongTermOperationCancelledError(str(reason))


__all__ = [
    "LongTermOperationAbortScope",
    "LongTermOperationCancelledError",
    "current_longterm_operation_abort_scope",
    "longterm_operation_abort_scope",
    "raise_if_longterm_operation_cancelled",
    "sleep_with_longterm_operation_abort",
]
