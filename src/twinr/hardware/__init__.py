"""Expose Twinr's stable hardware package imports lazily.

Import camera, PIR, and voice-profile entry points from here when callers need
the package-level surface without eagerly loading config-heavy hardware
dependencies during unrelated imports such as audio helpers.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from twinr.hardware.camera import CapturedPhoto, V4L2StillCamera
    from twinr.hardware.drone_service import (
        DroneMissionRequest,
        DroneMissionStatus,
        DronePoseSnapshot,
        DroneSafetyStatus,
        DroneServiceConfig,
        DroneStateSnapshot,
        DroneTelemetrySnapshot,
        RemoteDroneServiceClient,
    )
    from twinr.hardware.pir import GpioPirMonitor, PirBinding, PirMotionEvent, build_pir_binding, configured_pir_monitor
    from twinr.hardware.voice_profile import (
        VoiceAssessment,
        VoiceProfileMonitor,
        VoiceProfileStore,
        VoiceProfileSummary,
        VoiceProfileTemplate,
        voice_profile_store_path,
    )

    _TYPECHECK_EXPORTS = (
        CapturedPhoto,
        DroneMissionRequest,
        DroneMissionStatus,
        DronePoseSnapshot,
        DroneSafetyStatus,
        DroneServiceConfig,
        DroneStateSnapshot,
        DroneTelemetrySnapshot,
        GpioPirMonitor,
        PirBinding,
        PirMotionEvent,
        RemoteDroneServiceClient,
        V4L2StillCamera,
        VoiceAssessment,
        VoiceProfileMonitor,
        VoiceProfileStore,
        VoiceProfileSummary,
        VoiceProfileTemplate,
        build_pir_binding,
        configured_pir_monitor,
        voice_profile_store_path,
    )

__all__ = [
    "CapturedPhoto",
    "DroneMissionRequest",
    "DroneMissionStatus",
    "DroneServiceConfig",
    "DroneStateSnapshot",
    "DroneSafetyStatus",
    "DronePoseSnapshot",
    "DroneTelemetrySnapshot",
    "GpioPirMonitor",
    "PirBinding",
    "PirMotionEvent",
    "RemoteDroneServiceClient",
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
    "DroneMissionRequest": "twinr.hardware.drone_service",
    "DroneMissionStatus": "twinr.hardware.drone_service",
    "DronePoseSnapshot": "twinr.hardware.drone_service",
    "DroneSafetyStatus": "twinr.hardware.drone_service",
    "DroneServiceConfig": "twinr.hardware.drone_service",
    "DroneStateSnapshot": "twinr.hardware.drone_service",
    "DroneTelemetrySnapshot": "twinr.hardware.drone_service",
    "GpioPirMonitor": "twinr.hardware.pir",
    "PirBinding": "twinr.hardware.pir",
    "PirMotionEvent": "twinr.hardware.pir",
    "RemoteDroneServiceClient": "twinr.hardware.drone_service",
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
