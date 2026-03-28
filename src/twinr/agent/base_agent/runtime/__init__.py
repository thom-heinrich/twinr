"""Runtime assembly and the focused runtime mixins for Twinr live in this package."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from twinr.agent.base_agent.runtime.runtime import TwinrRuntime

__all__ = ["TwinrRuntime"]


def __getattr__(name: str) -> object:
    if name != "TwinrRuntime":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module("twinr.agent.base_agent.runtime.runtime")
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
