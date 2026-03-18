"""Official XVF3800 parameter specs used by Twinr primitive snapshots."""

from __future__ import annotations

from twinr.hardware.respeaker.models import (
    ReSpeakerParameterRead,
    ReSpeakerParameterSpec,
    ReSpeakerParameterType,
    ReSpeakerProbeResult,
    ReSpeakerTransportAvailability,
)
from twinr.hardware.respeaker.transport import ReSpeakerLibusbTransport


VERSION_PARAMETER = ReSpeakerParameterSpec(
    name="VERSION",
    resid=48,
    cmdid=0,
    value_count=3,
    access_mode="ro",
    value_type=ReSpeakerParameterType.UINT8,
    description="Firmware version as major, minor, patch bytes.",
)
AEC_AZIMUTH_VALUES_PARAMETER = ReSpeakerParameterSpec(
    name="AEC_AZIMUTH_VALUES",
    resid=33,
    cmdid=75,
    value_count=4,
    access_mode="ro",
    value_type=ReSpeakerParameterType.RADIANS,
    description="Beam azimuth values in radians for beam1, beam2, free-running, and auto-select.",
)
AEC_SPENERGY_VALUES_PARAMETER = ReSpeakerParameterSpec(
    name="AEC_SPENERGY_VALUES",
    resid=33,
    cmdid=80,
    value_count=4,
    access_mode="ro",
    value_type=ReSpeakerParameterType.FLOAT,
    description="Speech energy values for beam1, beam2, free-running, and auto-select.",
)
AUDIO_MGR_SELECTED_AZIMUTHS_PARAMETER = ReSpeakerParameterSpec(
    name="AUDIO_MGR_SELECTED_AZIMUTHS",
    resid=35,
    cmdid=11,
    value_count=2,
    access_mode="ro",
    value_type=ReSpeakerParameterType.RADIANS,
    description="Processed speaker DoA and auto-select beam DoA in radians.",
)
DOA_VALUE_PARAMETER = ReSpeakerParameterSpec(
    name="DOA_VALUE",
    resid=20,
    cmdid=18,
    value_count=2,
    access_mode="ro",
    value_type=ReSpeakerParameterType.UINT16,
    description="DoA degrees and speech-detected flag.",
)
GPO_READ_VALUES_PARAMETER = ReSpeakerParameterSpec(
    name="GPO_READ_VALUES",
    resid=20,
    cmdid=0,
    value_count=5,
    access_mode="ro",
    value_type=ReSpeakerParameterType.UINT8,
    description="Current logic levels for exposed XVF3800 GPO pins.",
)

DEFAULT_RESPEAKER_PARAMETER_SPECS = (
    VERSION_PARAMETER,
    DOA_VALUE_PARAMETER,
    AEC_AZIMUTH_VALUES_PARAMETER,
    AEC_SPENERGY_VALUES_PARAMETER,
    AUDIO_MGR_SELECTED_AZIMUTHS_PARAMETER,
    GPO_READ_VALUES_PARAMETER,
)


def read_default_respeaker_parameters(
    transport: ReSpeakerLibusbTransport,
    *,
    probe: ReSpeakerProbeResult | None = None,
) -> tuple[ReSpeakerTransportAvailability, dict[str, ReSpeakerParameterRead]]:
    """Read the default bounded primitive parameter set from one XVF3800."""

    return transport.capture_reads(DEFAULT_RESPEAKER_PARAMETER_SPECS, probe=probe)
