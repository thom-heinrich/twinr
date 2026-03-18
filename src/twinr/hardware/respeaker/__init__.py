"""Expose ReSpeaker XVF3800 probe and primitive helpers.

This package keeps XVF3800 host-control transport, raw parameter reads,
typed primitive models, and USB/ALSA readiness probing isolated from the
rest of Twinr. Callers import from this package-level surface so future
internal transport changes do not leak into ops or runtime code.
"""

from twinr.hardware.respeaker.models import (
    ReSpeakerCaptureDevice,
    ReSpeakerDirectionSnapshot,
    ReSpeakerMuteSnapshot,
    ReSpeakerParameterRead,
    ReSpeakerParameterSpec,
    ReSpeakerParameterType,
    ReSpeakerPrimitiveSnapshot,
    ReSpeakerProbeResult,
    ReSpeakerSignalSnapshot,
    ReSpeakerTransportAvailability,
    ReSpeakerUsbDevice,
)
from twinr.hardware.respeaker.probe import config_targets_respeaker, probe_respeaker_xvf3800
from twinr.hardware.respeaker.raw_reads import (
    AEC_AZIMUTH_VALUES_PARAMETER,
    AEC_SPENERGY_VALUES_PARAMETER,
    AUDIO_MGR_SELECTED_AZIMUTHS_PARAMETER,
    DEFAULT_RESPEAKER_PARAMETER_SPECS,
    DOA_VALUE_PARAMETER,
    GPO_READ_VALUES_PARAMETER,
    VERSION_PARAMETER,
    read_default_respeaker_parameters,
)
from twinr.hardware.respeaker.snapshot_service import capture_respeaker_primitive_snapshot
from twinr.hardware.respeaker.signal_provider import ReSpeakerSignalProvider
from twinr.hardware.respeaker.transport import ReSpeakerLibusbTransport

__all__ = [
    "AEC_AZIMUTH_VALUES_PARAMETER",
    "AEC_SPENERGY_VALUES_PARAMETER",
    "AUDIO_MGR_SELECTED_AZIMUTHS_PARAMETER",
    "DEFAULT_RESPEAKER_PARAMETER_SPECS",
    "DOA_VALUE_PARAMETER",
    "GPO_READ_VALUES_PARAMETER",
    "VERSION_PARAMETER",
    "ReSpeakerCaptureDevice",
    "ReSpeakerDirectionSnapshot",
    "ReSpeakerLibusbTransport",
    "ReSpeakerMuteSnapshot",
    "ReSpeakerParameterRead",
    "ReSpeakerParameterSpec",
    "ReSpeakerParameterType",
    "ReSpeakerPrimitiveSnapshot",
    "ReSpeakerProbeResult",
    "ReSpeakerSignalProvider",
    "ReSpeakerSignalSnapshot",
    "ReSpeakerTransportAvailability",
    "ReSpeakerUsbDevice",
    "capture_respeaker_primitive_snapshot",
    "config_targets_respeaker",
    "probe_respeaker_xvf3800",
    "read_default_respeaker_parameters",
]
