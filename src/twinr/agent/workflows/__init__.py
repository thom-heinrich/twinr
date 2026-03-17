"""Expose the canonical workflow loop entry points lazily for Twinr."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "SocialTriggerDecision",
    "SocialTriggerEngine",
    "TwinrHardwareLoop",
    "TwinrRealtimeHardwareLoop",
    "TwinrStreamingHardwareLoop",
]

_EXPORTS = {
    "SocialTriggerDecision": "twinr.proactive",
    "SocialTriggerEngine": "twinr.proactive",
    "TwinrHardwareLoop": "twinr.agent.legacy.classic_hardware_loop",
    "TwinrRealtimeHardwareLoop": "twinr.agent.workflows.realtime_runner",
    "TwinrStreamingHardwareLoop": "twinr.agent.workflows.streaming_runner",
}


def __getattr__(name: str) -> object:
    """Resolve exported workflow symbols without eager runner imports."""

    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)


def __dir__() -> list[str]:
    """Return the combined static and lazy export names for introspection."""

    return sorted(set(globals()) | set(__all__))
