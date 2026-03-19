"""Official XVF3800 parameter specs used by Twinr primitive snapshots."""

from __future__ import annotations

from typing import Final

from twinr.hardware.respeaker.models import (
    ReSpeakerParameterRead,
    ReSpeakerParameterSpec,
    ReSpeakerParameterType,
    ReSpeakerProbeResult,
    ReSpeakerTransportAvailability,
)
from twinr.hardware.respeaker.transport import ReSpeakerLibusbTransport

ParameterDefinition = tuple[
    str,
    int,
    int,
    int,
    str,
    ReSpeakerParameterType,
    str,
]

# AUDIT-FIX(#1): Keep a canonical immutable definition table so runtime reads can
# be rebuilt from pristine data instead of reusing shared singleton spec objects.
_VERSION_PARAMETER_DEFINITION: Final[ParameterDefinition] = (
    "VERSION",
    48,
    0,
    3,
    "ro",
    ReSpeakerParameterType.UINT8,
    "Firmware version as major, minor, patch bytes.",
)
_AEC_AZIMUTH_VALUES_PARAMETER_DEFINITION: Final[ParameterDefinition] = (
    "AEC_AZIMUTH_VALUES",
    33,
    75,
    4,
    "ro",
    ReSpeakerParameterType.RADIANS,
    "Beam azimuth values in radians for beam1, beam2, free-running, and auto-select.",
)
_AEC_SPENERGY_VALUES_PARAMETER_DEFINITION: Final[ParameterDefinition] = (
    "AEC_SPENERGY_VALUES",
    33,
    80,
    4,
    "ro",
    ReSpeakerParameterType.FLOAT,
    "Speech energy values for beam1, beam2, free-running, and auto-select.",
)
_AUDIO_MGR_SELECTED_AZIMUTHS_PARAMETER_DEFINITION: Final[ParameterDefinition] = (
    "AUDIO_MGR_SELECTED_AZIMUTHS",
    35,
    11,
    2,
    "ro",
    ReSpeakerParameterType.RADIANS,
    "Processed speaker DoA and auto-select beam DoA in radians.",
)
_DOA_VALUE_PARAMETER_DEFINITION: Final[ParameterDefinition] = (
    "DOA_VALUE",
    20,
    18,
    2,
    "ro",
    ReSpeakerParameterType.UINT16,
    "DoA degrees and speech-detected flag.",
)
_GPO_READ_VALUES_PARAMETER_DEFINITION: Final[ParameterDefinition] = (
    "GPO_READ_VALUES",
    20,
    0,
    5,
    "ro",
    ReSpeakerParameterType.UINT8,
    "Current logic levels for exposed XVF3800 GPO pins.",
)

_PARAMETER_DEFINITIONS: Final[tuple[ParameterDefinition, ...]] = (
    _VERSION_PARAMETER_DEFINITION,
    _AEC_AZIMUTH_VALUES_PARAMETER_DEFINITION,
    _AEC_SPENERGY_VALUES_PARAMETER_DEFINITION,
    _AUDIO_MGR_SELECTED_AZIMUTHS_PARAMETER_DEFINITION,
    _DOA_VALUE_PARAMETER_DEFINITION,
    _GPO_READ_VALUES_PARAMETER_DEFINITION,
)


def _validate_parameter_definitions(definitions: tuple[ParameterDefinition, ...]) -> None:
    """Fail fast on local schema corruption instead of silently mis-decoding device output."""

    seen_names: set[str] = set()
    seen_commands: set[tuple[int, int]] = set()

    for name, resid, cmdid, value_count, access_mode, _value_type, description in definitions:
        if not name:
            raise ValueError("ReSpeaker parameter name must not be empty.")
        if name in seen_names:
            raise ValueError(f"Duplicate ReSpeaker parameter name: {name}.")
        seen_names.add(name)

        if value_count <= 0:
            raise ValueError(f"{name} must declare at least one value.")
        if access_mode not in {"ro", "rw", "wo"}:
            raise ValueError(f"{name} has unsupported access_mode: {access_mode}.")
        if not description:
            raise ValueError(f"{name} must include a non-empty description.")

        command_key = (resid, cmdid)
        if command_key in seen_commands:
            raise ValueError(
                f"Duplicate ReSpeaker command tuple detected for {name}: "
                f"resid={resid}, cmdid={cmdid}."
            )
        seen_commands.add(command_key)


# AUDIT-FIX(#2): Validate the baked-in command table at import time so edit
# mistakes fail fast before any USB traffic reaches the device.
_validate_parameter_definitions(_PARAMETER_DEFINITIONS)


def _build_parameter_spec(definition: ParameterDefinition) -> ReSpeakerParameterSpec:
    (
        name,
        resid,
        cmdid,
        value_count,
        access_mode,
        value_type,
        description,
    ) = definition
    return ReSpeakerParameterSpec(
        name=name,
        resid=resid,
        cmdid=cmdid,
        value_count=value_count,
        access_mode=access_mode,
        value_type=value_type,
        description=description,
    )


VERSION_PARAMETER = _build_parameter_spec(_VERSION_PARAMETER_DEFINITION)
AEC_AZIMUTH_VALUES_PARAMETER = _build_parameter_spec(_AEC_AZIMUTH_VALUES_PARAMETER_DEFINITION)
AEC_SPENERGY_VALUES_PARAMETER = _build_parameter_spec(_AEC_SPENERGY_VALUES_PARAMETER_DEFINITION)
AUDIO_MGR_SELECTED_AZIMUTHS_PARAMETER = _build_parameter_spec(
    _AUDIO_MGR_SELECTED_AZIMUTHS_PARAMETER_DEFINITION
)
DOA_VALUE_PARAMETER = _build_parameter_spec(_DOA_VALUE_PARAMETER_DEFINITION)
GPO_READ_VALUES_PARAMETER = _build_parameter_spec(_GPO_READ_VALUES_PARAMETER_DEFINITION)

DEFAULT_RESPEAKER_PARAMETER_SPECS: Final[tuple[ReSpeakerParameterSpec, ...]] = (
    VERSION_PARAMETER,
    DOA_VALUE_PARAMETER,
    AEC_AZIMUTH_VALUES_PARAMETER,
    AEC_SPENERGY_VALUES_PARAMETER,
    AUDIO_MGR_SELECTED_AZIMUTHS_PARAMETER,
    GPO_READ_VALUES_PARAMETER,
)


def _build_runtime_default_respeaker_parameter_specs() -> tuple[ReSpeakerParameterSpec, ...]:
    return (
        _build_parameter_spec(_VERSION_PARAMETER_DEFINITION),
        _build_parameter_spec(_DOA_VALUE_PARAMETER_DEFINITION),
        _build_parameter_spec(_AEC_AZIMUTH_VALUES_PARAMETER_DEFINITION),
        _build_parameter_spec(_AEC_SPENERGY_VALUES_PARAMETER_DEFINITION),
        _build_parameter_spec(_AUDIO_MGR_SELECTED_AZIMUTHS_PARAMETER_DEFINITION),
        _build_parameter_spec(_GPO_READ_VALUES_PARAMETER_DEFINITION),
    )


def read_default_respeaker_parameters(
    transport: ReSpeakerLibusbTransport,
    *,
    probe: ReSpeakerProbeResult | None = None,
) -> tuple[ReSpeakerTransportAvailability, dict[str, ReSpeakerParameterRead]]:
    """Read the default bounded primitive parameter set from one XVF3800."""

    # AUDIT-FIX(#1): Pass fresh spec instances into the transport so any downstream
    # mutation cannot poison future reads across the process.
    parameter_specs = _build_runtime_default_respeaker_parameter_specs()

    try:
        return transport.capture_reads(parameter_specs, probe=probe)
    except Exception as exc:
        # AUDIT-FIX(#3): Preserve the original exception type while adding enough
        # snapshot context for operations and recovery logic upstream.
        exc.add_note(
            "Twinr XVF3800 default snapshot read failed for VERSION, DOA_VALUE, "
            "AEC_AZIMUTH_VALUES, AEC_SPENERGY_VALUES, "
            "AUDIO_MGR_SELECTED_AZIMUTHS, and GPO_READ_VALUES."
        )
        raise