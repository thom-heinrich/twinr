"""Hardware integrations for Twinr."""

from twinr.hardware.camera import CapturedPhoto, V4L2StillCamera
from twinr.hardware.pir import GpioPirMonitor, PirBinding, PirMotionEvent, build_pir_binding, configured_pir_monitor
from twinr.hardware.voice_profile import (
    VoiceAssessment,
    VoiceProfileMonitor,
    VoiceProfileStore,
    VoiceProfileSummary,
    VoiceProfileTemplate,
    voice_profile_store_path,
)

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
