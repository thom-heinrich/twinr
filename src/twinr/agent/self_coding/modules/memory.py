"""Expose bounded self_coding memory helpers for future sandboxed skills."""

from __future__ import annotations

from typing import Any, NoReturn

from twinr.agent.self_coding.status import CapabilityRiskClass

from .base import SelfCodingModuleFunction, SelfCodingModuleSpec, runtime_unavailable


# AUDIT-FIX(#1): Fail closed locally and keep this placeholder API explicit until a real
# bounded backing store exists in the runtime.
def _raise_runtime_unavailable(operation: str) -> NoReturn:
    runtime_unavailable(operation)
    raise RuntimeError(f"{operation} is unavailable in this runtime")


def remember(key: str, value: Any, ttl: str | None = None) -> None:
    """Persist one small JSON-safe value under a stable skill-local key."""

    _raise_runtime_unavailable("memory.remember")


# AUDIT-FIX(#2): The contract explicitly allows None when the key is absent, so the type
# surface must advertise that to callers and code generators.
def recall(key: str) -> Any | None:
    """Load one previously stored JSON-safe value by key."""

    _raise_runtime_unavailable("memory.recall")


def forget(key: str) -> None:
    """Delete one stored key from the skill-local memory namespace."""

    _raise_runtime_unavailable("memory.forget")


# AUDIT-FIX(#3): Enforce an integer, positive bound locally so the public API does not
# accept invalid or silently coerced limits.
def recent(prefix: str, limit: int = 5) -> list[Any]:
    """Return a bounded list of recent values for one key prefix."""

    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be an integer >= 1")
    if limit < 1:
        raise ValueError("limit must be >= 1")

    _raise_runtime_unavailable("memory.recent")


# AUDIT-FIX(#1): Mark the spec as reserved/unavailable instead of implying a live storage
# backend that skill planners can safely depend on today.
MODULE_SPEC = SelfCodingModuleSpec(
    capability_id="memory",
    module_name="memory",
    summary=(
        "Reserved bounded self-coding memory API. In the current runtime, calls are "
        "unavailable until a backing store is wired in."
    ),
    risk_class=CapabilityRiskClass.MODERATE,
    public_api=(
        SelfCodingModuleFunction(
            name="remember",
            signature="remember(key: str, value: Any, ttl: str | None = None) -> None",
            summary="Store one small JSON-safe value for later skill use when supported.",
            effectful=True,
            tags=("effectful", "storage"),
        ),
        SelfCodingModuleFunction(
            name="recall",
            # AUDIT-FIX(#2): Keep the machine-readable signature aligned with the actual
            # None-on-miss contract.
            signature="recall(key: str) -> Any | None",
            summary="Read one previously stored value by key when supported.",
            returns="the stored value or None when the key is absent",
            tags=("read_only", "storage"),
        ),
        SelfCodingModuleFunction(
            name="forget",
            signature="forget(key: str) -> None",
            summary="Delete one stored key from the skill-local memory namespace when supported.",
            effectful=True,
            tags=("effectful", "storage"),
        ),
        SelfCodingModuleFunction(
            name="recent",
            signature="recent(prefix: str, limit: int = 5) -> list[Any]",
            summary="Return a bounded recent history for one key prefix when supported.",
            returns="a list of recent JSON-safe values",
            tags=("read_only", "storage"),
        ),
    ),
    tags=("memory", "builtin", "storage"),
)

__all__ = ["MODULE_SPEC", "forget", "recall", "recent", "remember"]