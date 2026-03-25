"""Idle-window gating helpers for realtime background delivery."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

_T = TypeVar("_T")


class BackgroundDeliveryBlocked(RuntimeError):
    """Signal that background work lost its idle delivery window."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def background_block_reason_locked(loop: Any) -> str | None:
    """Return why background output is blocked while the transition lock is held."""

    if getattr(getattr(loop.runtime, "status", None), "value", None) != "waiting":
        return "busy"
    if bool(getattr(loop, "_conversation_session_active", False)):
        return "conversation_active"
    return None


def background_block_reason(loop: Any) -> str | None:
    """Return why background output is currently blocked."""

    with loop._get_lock("_background_delivery_transition_lock"):
        return background_block_reason_locked(loop)


def background_work_allowed(loop: Any) -> bool:
    """Report whether background work may currently start."""

    with loop._get_lock("_background_delivery_transition_lock"):
        return background_block_reason_locked(loop) is None


def begin_background_delivery(loop: Any, action: Callable[[], _T]) -> _T:
    """Run one output-state transition only while Twinr is still idle."""

    with loop._get_lock("_background_delivery_transition_lock"):
        reason = background_block_reason_locked(loop)
        if reason is not None:
            raise BackgroundDeliveryBlocked(reason)
        return action()
