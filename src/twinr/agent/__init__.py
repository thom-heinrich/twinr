from __future__ import annotations

from importlib import import_module

__all__ = ["TwinrConfig", "TwinrRuntime", "TwinrStatus"]

_EXPORTS = {
    "TwinrConfig": "twinr.agent.base_agent",
    "TwinrRuntime": "twinr.agent.base_agent",
    "TwinrStatus": "twinr.agent.base_agent",
}


def __getattr__(name: str) -> object:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
