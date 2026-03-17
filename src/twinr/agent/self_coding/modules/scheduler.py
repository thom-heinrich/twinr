"""Expose bounded scheduling primitives for future self_coding skills."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NoReturn  # AUDIT-FIX(#2): Make the hard-fail helper's non-returning contract explicit.

from twinr.agent.self_coding.status import CapabilityRiskClass

from .base import SelfCodingModuleFunction, SelfCodingModuleSpec, runtime_unavailable


def _require_non_empty_str(name: str, value: str) -> None:
    # AUDIT-FIX(#3): Reject blank/non-string inputs early so caller bugs are not hidden behind generic unavailable errors.
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{name} must not be empty or whitespace")


def _require_callable(name: str, value: Callable[..., Any]) -> None:
    # AUDIT-FIX(#3): Validate callback inputs explicitly because this stub otherwise discards the callback object completely.
    if not callable(value):
        raise TypeError(f"{name} must be callable")


def _raise_runtime_unavailable(operation: str) -> NoReturn:
    # AUDIT-FIX(#1): Centralize deterministic fail-fast behavior for the intentionally unavailable scheduler capability.
    runtime_unavailable(operation)
    # AUDIT-FIX(#2): Preserve the public API contract even if runtime_unavailable() is mocked or refactored to return.
    raise RuntimeError(f"{operation} is unavailable in this runtime")


def every(interval: str, callback: Callable[..., Any]) -> str:
    """Schedule one recurring callback with a bounded human-readable interval."""

    _require_non_empty_str("interval", interval)  # AUDIT-FIX(#3): Catch malformed interval values before delegating.
    _require_callable("callback", callback)  # AUDIT-FIX(#3): Catch non-callable callback values before delegating.
    _raise_runtime_unavailable("scheduler.every")  # AUDIT-FIX(#1): Fail fast in one consistent, deterministic code path.


def at(when: str, callback: Callable[..., Any]) -> str:
    """Schedule one callback for a bounded absolute time."""

    _require_non_empty_str("when", when)  # AUDIT-FIX(#3): Catch malformed absolute-time values before delegating.
    _require_callable("callback", callback)  # AUDIT-FIX(#3): Catch non-callable callback values before delegating.
    _raise_runtime_unavailable("scheduler.at")  # AUDIT-FIX(#1): Fail fast in one consistent, deterministic code path.


def after(delay: str, callback: Callable[..., Any]) -> str:
    """Schedule one callback after a bounded delay."""

    _require_non_empty_str("delay", delay)  # AUDIT-FIX(#3): Catch malformed delay values before delegating.
    _require_callable("callback", callback)  # AUDIT-FIX(#3): Catch non-callable callback values before delegating.
    _raise_runtime_unavailable("scheduler.after")  # AUDIT-FIX(#1): Fail fast in one consistent, deterministic code path.


def cancel(job_id: str) -> None:
    """Cancel one previously scheduled callback."""

    _require_non_empty_str("job_id", job_id)  # AUDIT-FIX(#3): Catch malformed job identifiers before delegating.
    _raise_runtime_unavailable("scheduler.cancel")  # AUDIT-FIX(#1): Fail fast in one consistent, deterministic code path.


MODULE_SPEC = SelfCodingModuleSpec(
    capability_id="scheduler",
    module_name="scheduler",
    summary=(
        "Schedule bounded follow-ups and recurring checks for learned skills "
        "when the scheduler runtime is available."
    ),  # AUDIT-FIX(#4): Make runtime availability explicit so capability discovery is less misleading.
    risk_class=CapabilityRiskClass.MODERATE,
    public_api=(
        SelfCodingModuleFunction(
            name="every",
            signature="every(interval: str, callback: Callable[..., Any]) -> str",
            summary=(
                "Schedule one recurring callback using a short human-readable "
                "interval when the scheduler runtime is available."
            ),  # AUDIT-FIX(#4): Make runtime availability explicit at function metadata level.
            returns="a scheduler job identifier",
            effectful=True,
            tags=("effectful", "time"),
        ),
        SelfCodingModuleFunction(
            name="at",
            signature="at(when: str, callback: Callable[..., Any]) -> str",
            summary=(
                "Schedule one callback at a bounded wall-clock time when the "
                "scheduler runtime is available."
            ),  # AUDIT-FIX(#4): Make runtime availability explicit at function metadata level.
            returns="a scheduler job identifier",
            effectful=True,
            tags=("effectful", "time"),
        ),
        SelfCodingModuleFunction(
            name="after",
            signature="after(delay: str, callback: Callable[..., Any]) -> str",
            summary=(
                "Schedule one callback after a bounded delay when the "
                "scheduler runtime is available."
            ),  # AUDIT-FIX(#4): Make runtime availability explicit at function metadata level.
            returns="a scheduler job identifier",
            effectful=True,
            tags=("effectful", "time"),
        ),
        SelfCodingModuleFunction(
            name="cancel",
            signature="cancel(job_id: str) -> None",
            summary=(
                "Cancel one previously created scheduler job when the "
                "scheduler runtime is available."
            ),  # AUDIT-FIX(#4): Make runtime availability explicit at function metadata level.
            effectful=True,
            tags=("effectful", "time"),
        ),
    ),
    tags=("time", "builtin", "scheduler"),
)

__all__ = ["MODULE_SPEC", "after", "at", "cancel", "every"]