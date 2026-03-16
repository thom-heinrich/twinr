"""Expose the thin ``twinr.agent`` package surface and child package boundary.

Import ``TwinrConfig``, ``TwinrRuntime``, and ``TwinrStatus`` from here when a
caller only needs the small compatibility surface. Reach into
``twinr.agent.base_agent``, ``twinr.agent.tools``, or
``twinr.agent.workflows`` for focused runtime, tool, and workflow behavior
instead of growing this package root.
"""

from __future__ import annotations

from importlib import import_module

__all__ = ["TwinrConfig", "TwinrRuntime", "TwinrStatus"]

_EXPORTS = {
    "TwinrConfig": "twinr.agent.base_agent",
    "TwinrRuntime": "twinr.agent.base_agent",
    "TwinrStatus": "twinr.agent.base_agent",
}


def __getattr__(name: str) -> object:
    """Resolve lazy root exports from their owning child package."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)


def __dir__() -> list[str]:
    """Return the stable root export list for interactive callers."""
    return sorted(set(globals()) | set(__all__))
