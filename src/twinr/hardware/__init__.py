"""Expose Twinr's stable hardware package imports lazily.

Import camera, PIR, and voice-profile entry points from here when callers need
the package-level surface without eagerly loading config-heavy hardware
dependencies during unrelated imports such as audio helpers.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "CapturedPhoto",
    "GpioPirMonitor",
    "PirBinding",
    "PirMotionEvent",
    "V4L2StillCamera",
    "VoiceAssessment",
    "VoiceProfileMonitor",
    "VoiceProfileStore",
    "VoiceProfileSummary",
    "VoiceProfileTemplate",
    "build_pir_binding",
    "configured_pir_monitor",
    "voice_profile_store_path",
]

_EXPORTS = {
    "CapturedPhoto": "twinr.hardware.camera",
    "GpioPirMonitor": "twinr.hardware.pir",
    "PirBinding": "twinr.hardware.pir",
    "PirMotionEvent": "twinr.hardware.pir",
    "V4L2StillCamera": "twinr.hardware.camera",
    "VoiceAssessment": "twinr.hardware.voice_profile",
    "VoiceProfileMonitor": "twinr.hardware.voice_profile",
    "VoiceProfileStore": "twinr.hardware.voice_profile",
    "VoiceProfileSummary": "twinr.hardware.voice_profile",
    "VoiceProfileTemplate": "twinr.hardware.voice_profile",
    "build_pir_binding": "twinr.hardware.pir",
    "configured_pir_monitor": "twinr.hardware.pir",
    "voice_profile_store_path": "twinr.hardware.voice_profile",
}


def __getattr__(name: str) -> object:
    """Resolve exported symbols lazily from their owning modules."""

    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)


def __dir__() -> list[str]:
    """Return the combined static and lazy export names for introspection."""

    return sorted(set(globals()) | set(__all__))
