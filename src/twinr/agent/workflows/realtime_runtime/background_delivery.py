# CHANGELOG: 2026-03-27
# BUG-1: Fixed self-deadlock on non-reentrant transition locks when callbacks re-entered background gating helpers.
# BUG-2: Fixed missing explicit in-flight ownership by adding generation-checked idle-window leases.
# SEC-1: Removed lock-held execution from the primary API so slow or untrusted callbacks cannot trivially freeze delivery-state transitions.
# IMP-1: Added lease-based prepare/commit flow with revalidation (`BackgroundDeliveryLease.assert_valid()` / `.run_locked()`).
# IMP-2: Added typed block reasons, delivery snapshots, and a legacy compatibility helper for controlled migration.

"""Idle-window gating helpers for realtime background delivery.

2026 model:
1. Claim the idle window quickly under the transition lock.
2. Do expensive preparation outside the lock.
3. Revalidate and commit through ``BackgroundDeliveryLease`` immediately before
   the externally visible transition.

This keeps lock hold times short on Raspberry Pi class edge devices while still
preserving a precise "lost the idle window" signal for callers that opt into the
lease-aware API.
"""

from __future__ import annotations

import inspect
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Final, Protocol, TypeVar, cast

_T = TypeVar("_T")

_LOCK_NAME: Final = "_background_delivery_transition_lock"
_STATE_ATTR: Final = "_background_delivery_gate_state"
_REENTRANCY_LOCAL = threading.local()


class BackgroundDeliveryLoop(Protocol):
    """Structural type for the subset of loop API used by this module."""

    runtime: Any
    _conversation_session_active: bool

    def _get_lock(self, name: str) -> Any:
        ...


class BackgroundBlockReason(StrEnum):
    """Stable reason codes for why new background delivery cannot begin."""

    BUSY = "busy"
    CONVERSATION_ACTIVE = "conversation_active"
    DELIVERY_INFLIGHT = "delivery_inflight"
    STALE_LEASE = "stale_lease"
    LEASE_RELEASED = "lease_released"


class BackgroundDeliveryBlocked(RuntimeError):
    """Signal that background work could not claim or keep its idle window."""

    __slots__ = ("reason", "code")

    def __init__(self, reason: str | BackgroundBlockReason) -> None:
        code = reason if isinstance(reason, BackgroundBlockReason) else None
        reason_text = reason.value if isinstance(reason, BackgroundBlockReason) else reason
        super().__init__(reason_text)
        self.reason = reason_text
        self.code = code


@dataclass(slots=True)
class _BackgroundDeliveryGateState:
    generation: int = 0
    inflight: bool = False
    owner_thread_id: int | None = None
    reserved_monotonic_ns: int | None = None


@dataclass(slots=True, frozen=True)
class BackgroundDeliverySnapshot:
    """Cheap observability snapshot for debugging and metrics."""

    runtime_status: str | None
    conversation_session_active: bool
    delivery_inflight: bool
    delivery_generation: int
    delivery_owner_thread_id: int | None
    delivery_age_ms: int | None


@dataclass(slots=True)
class BackgroundDeliveryLease:
    """A claimed idle window for background delivery.

    Callers should do heavy preparation work outside the transition lock and then
    either:
      * call ``assert_valid()`` immediately before an irreversible side effect, or
      * wrap the final state/output mutation in ``run_locked()``.
    """

    loop: BackgroundDeliveryLoop
    generation: int
    owner_thread_id: int
    released: bool = False

    def blocked_reason(self) -> str | None:
        """Return why this lease is no longer valid."""

        with _transition_lock(self.loop):
            return _lease_block_reason_locked(self.loop, self)

    def is_valid(self) -> bool:
        """Report whether this lease still owns a valid idle window."""

        return self.blocked_reason() is None

    def assert_valid(self) -> None:
        """Raise if this lease has become invalid."""

        reason = self.blocked_reason()
        if reason is not None:
            raise BackgroundDeliveryBlocked(reason)

    def age_ms(self) -> int | None:
        """Return the lease age in milliseconds, or ``None`` when stale."""

        with _transition_lock(self.loop):
            state = _gate_state_locked(self.loop)
            if state.generation != self.generation or state.reserved_monotonic_ns is None:
                return None
            return max(0, (time.monotonic_ns() - state.reserved_monotonic_ns) // 1_000_000)

    def run_locked(self, action: Callable[[], _T]) -> _T:
        """Run the final transition atomically while this lease is still valid."""

        with _transition_lock(self.loop):
            reason = _lease_block_reason_locked(self.loop, self)
            if reason is not None:
                raise BackgroundDeliveryBlocked(reason)
            return action()

    def release(self) -> None:
        """Release this lease idempotently."""

        with _transition_lock(self.loop):
            if self.released:
                return

            state = _gate_state_locked(self.loop)
            if (
                state.inflight
                and state.generation == self.generation
                and state.owner_thread_id == self.owner_thread_id
            ):
                state.inflight = False
                state.owner_thread_id = None
                state.reserved_monotonic_ns = None

            self.released = True

    def __enter__(self) -> BackgroundDeliveryLease:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False


def _lock_key(loop: BackgroundDeliveryLoop) -> tuple[int, str]:
    return (id(loop), _LOCK_NAME)


def _lock_depths() -> dict[tuple[int, str], int]:
    depths = getattr(_REENTRANCY_LOCAL, "depths", None)
    if depths is None:
        depths = {}
        _REENTRANCY_LOCAL.depths = depths
    return cast(dict[tuple[int, str], int], depths)


@contextmanager
def _transition_lock(loop: BackgroundDeliveryLoop):
    """Reentrancy-safe wrapper around the shared transition lock.

    The original helper assumed callers would never re-enter the same helper
    stack while the lock was held. On ordinary ``threading.Lock`` instances that
    can deadlock. This wrapper keeps lock semantics unchanged for other threads
    while allowing same-thread helper re-entry.
    """

    depths = _lock_depths()
    key = _lock_key(loop)
    depth = depths.get(key, 0)

    if depth:
        depths[key] = depth + 1
        try:
            yield
        finally:
            next_depth = depths[key] - 1
            if next_depth:
                depths[key] = next_depth
            else:
                depths.pop(key, None)
        return

    with loop._get_lock(_LOCK_NAME):
        depths[key] = 1
        try:
            yield
        finally:
            depths.pop(key, None)


def _runtime_status_value(loop: BackgroundDeliveryLoop) -> str | None:
    return getattr(getattr(getattr(loop, "runtime", None), "status", None), "value", None)


def _gate_state_locked(loop: BackgroundDeliveryLoop) -> _BackgroundDeliveryGateState:
    state = getattr(loop, _STATE_ATTR, None)
    if state is None:
        state = _BackgroundDeliveryGateState()
        setattr(loop, _STATE_ATTR, state)
    return cast(_BackgroundDeliveryGateState, state)


def _lease_block_reason_locked(loop: BackgroundDeliveryLoop, lease: BackgroundDeliveryLease) -> str | None:
    if lease.released:
        return BackgroundBlockReason.LEASE_RELEASED.value

    state = _gate_state_locked(loop)
    if (
        not state.inflight
        or state.generation != lease.generation
        or state.owner_thread_id != lease.owner_thread_id
    ):
        return BackgroundBlockReason.STALE_LEASE.value

    if _runtime_status_value(loop) != "waiting":
        return BackgroundBlockReason.BUSY.value
    if bool(getattr(loop, "_conversation_session_active", False)):
        return BackgroundBlockReason.CONVERSATION_ACTIVE.value
    return None


def background_delivery_snapshot(loop: BackgroundDeliveryLoop) -> BackgroundDeliverySnapshot:
    """Return a metrics/debug snapshot of current delivery-gate state."""

    with _transition_lock(loop):
        state = _gate_state_locked(loop)
        delivery_age_ms = None
        if state.inflight and state.reserved_monotonic_ns is not None:
            delivery_age_ms = max(0, (time.monotonic_ns() - state.reserved_monotonic_ns) // 1_000_000)

        return BackgroundDeliverySnapshot(
            runtime_status=_runtime_status_value(loop),
            conversation_session_active=bool(getattr(loop, "_conversation_session_active", False)),
            delivery_inflight=state.inflight,
            delivery_generation=state.generation,
            delivery_owner_thread_id=state.owner_thread_id,
            delivery_age_ms=delivery_age_ms,
        )


def background_block_reason_locked(loop: BackgroundDeliveryLoop) -> str | None:
    """Return why *new* background output is blocked while the transition lock is held."""

    if _runtime_status_value(loop) != "waiting":
        return BackgroundBlockReason.BUSY.value
    if bool(getattr(loop, "_conversation_session_active", False)):
        return BackgroundBlockReason.CONVERSATION_ACTIVE.value

    # BREAKING: callers may now also receive "delivery_inflight" when another
    # background delivery has already reserved the idle window.
    if _gate_state_locked(loop).inflight:
        return BackgroundBlockReason.DELIVERY_INFLIGHT.value
    return None


def background_block_reason(loop: BackgroundDeliveryLoop) -> str | None:
    """Return why new background output is currently blocked."""

    with _transition_lock(loop):
        return background_block_reason_locked(loop)


def background_work_allowed(loop: BackgroundDeliveryLoop) -> bool:
    """Report whether a *new* background delivery may currently start."""

    with _transition_lock(loop):
        return background_block_reason_locked(loop) is None


def _reserve_background_delivery_locked(loop: BackgroundDeliveryLoop) -> BackgroundDeliveryLease:
    reason = background_block_reason_locked(loop)
    if reason is not None:
        raise BackgroundDeliveryBlocked(reason)

    state = _gate_state_locked(loop)
    state.generation += 1
    state.inflight = True
    state.owner_thread_id = threading.get_ident()
    state.reserved_monotonic_ns = time.monotonic_ns()

    return BackgroundDeliveryLease(
        loop=loop,
        generation=state.generation,
        owner_thread_id=state.owner_thread_id,
    )


def acquire_background_delivery_lease(loop: BackgroundDeliveryLoop) -> BackgroundDeliveryLease:
    """Claim the idle window and return a lease that can be revalidated later."""

    with _transition_lock(loop):
        return _reserve_background_delivery_locked(loop)


def _callable_accepts_one_positional_argument(action: Callable[..., Any]) -> bool | None:
    try:
        signature = inspect.signature(action)
    except (TypeError, ValueError):
        return None

    required_positional = 0
    total_positional = 0
    has_varargs = False

    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            has_varargs = True
            continue
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            continue

        total_positional += 1
        if parameter.default is inspect.Signature.empty:
            required_positional += 1

    if has_varargs:
        return required_positional <= 1
    return required_positional <= 1 <= total_positional


def begin_background_delivery(
    loop: BackgroundDeliveryLoop,
    action: Callable[..., _T],
) -> _T:
    """Run background delivery work with a lease-aware prepare/commit protocol.

    The frontier API accepts ``action(lease)`` and executes the callback
    *outside* the transition lock. For controlled migration, existing
    zero-argument callbacks are still routed through the legacy lock-held path
    instead of failing at runtime.

    Migration pattern:
        def action(lease: BackgroundDeliveryLease) -> T:
            prepared = prepare()
            lease.assert_valid()
            return lease.run_locked(lambda: commit(prepared))
    """

    accepts_lease = _callable_accepts_one_positional_argument(action)
    if accepts_lease is False:
        return begin_background_delivery_legacy_locked(
            loop,
            cast(Callable[[], _T], action),
        )

    with acquire_background_delivery_lease(loop) as lease:
        return cast(Callable[[BackgroundDeliveryLease], _T], action)(lease)


def begin_background_delivery_legacy_locked(
    loop: BackgroundDeliveryLoop,
    action: Callable[[], _T],
) -> _T:
    """Compatibility path for old zero-argument callbacks.

    This preserves the old "run the callback while holding the transition lock"
    behavior, but should be treated as a migration helper, not the frontier API.
    """

    with _transition_lock(loop):
        lease = _reserve_background_delivery_locked(loop)
        try:
            return action()
        finally:
            lease.release()


__all__ = [
    "BackgroundBlockReason",
    "BackgroundDeliveryBlocked",
    "BackgroundDeliveryLease",
    "BackgroundDeliveryLoop",
    "BackgroundDeliverySnapshot",
    "acquire_background_delivery_lease",
    "background_block_reason",
    "background_block_reason_locked",
    "background_delivery_snapshot",
    "background_work_allowed",
    "begin_background_delivery",
    "begin_background_delivery_legacy_locked",
]
