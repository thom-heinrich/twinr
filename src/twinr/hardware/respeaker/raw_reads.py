# CHANGELOG: 2026-03-28
# BUG-1: Fail closed on firmware-semantic drift for AUDIO_MGR_SELECTED_AZIMUTHS on XVF3800 <= 2.0.0.
# BUG-2: Exception note enrichment no longer masks the original transport error on Python runtimes without BaseException.add_note().
# SEC-1: No practical file-local exploit path was found; added read-only catalog/index views as low-cost in-process hardening.
# IMP-1: Replaced per-call spec rebuilding with one immutable canonical snapshot catalog and O(1) lookup indexes.
# IMP-2: Generalized the read boundary from one concrete libusb class to a protocol-compatible transport API, ready for hotplug/async backends.

"""Official XVF3800 parameter specs used by Twinr primitive snapshots."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol, TypeAlias, runtime_checkable

from twinr.hardware.respeaker.models import (
    ReSpeakerParameterRead,
    ReSpeakerParameterSpec,
    ReSpeakerParameterType,
    ReSpeakerProbeResult,
    ReSpeakerTransportAvailability,
)

ParameterDefinition: TypeAlias = tuple[
    str,
    int,
    int,
    int,
    str,
    ReSpeakerParameterType,
    str,
]
FirmwareVersion: TypeAlias = tuple[int, int, int]
CommandKey: TypeAlias = tuple[int, int]


@runtime_checkable
class ReSpeakerReadTransport(Protocol):
    """Minimal transport surface required by the default snapshot reader."""

    def capture_reads(
        self,
        specs: Iterable[ReSpeakerParameterSpec],
        *,
        probe: ReSpeakerProbeResult | None = None,
    ) -> tuple[ReSpeakerTransportAvailability, dict[str, ReSpeakerParameterRead]]:
        ...


@dataclass(frozen=True, slots=True)
class DefaultSnapshotParameterEntry:
    """One default XVF3800 snapshot parameter with optional semantic guards."""

    spec: ReSpeakerParameterSpec
    minimum_semantic_firmware: FirmwareVersion | None = None
    semantic_guard_reason: str | None = None

    def __post_init__(self) -> None:
        if self.minimum_semantic_firmware is None:
            if self.semantic_guard_reason is not None:
                raise ValueError(
                    "semantic_guard_reason requires minimum_semantic_firmware to be set."
                )
            return

        object.__setattr__(
            self,
            "minimum_semantic_firmware",
            _normalize_firmware_version(
                self.minimum_semantic_firmware,
                field_name="minimum_semantic_firmware",
            ),
        )
        if not self.semantic_guard_reason:
            raise ValueError(
                f"{self.spec.name} defines minimum_semantic_firmware without a guard reason."
            )


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
    (
        "Processed speaker DoA and auto-select beam DoA in radians. "
        "XVF3800 firmware 2.0.0 and earlier used a different default beam-selection "
        "semantic, so Twinr guards this parameter unless firmware is >= 2.0.1."
    ),
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


def _normalize_firmware_version(
    version: FirmwareVersion,
    *,
    field_name: str,
) -> FirmwareVersion:
    if len(version) != 3:
        raise ValueError(f"{field_name} must contain exactly three integer components.")

    normalized: list[int] = []
    for index, component in enumerate(version):
        if isinstance(component, bool) or not isinstance(component, int):
            raise TypeError(f"{field_name}[{index}] must be an int.")
        if component < 0:
            raise ValueError(f"{field_name}[{index}] must be >= 0.")
        normalized.append(component)

    return (normalized[0], normalized[1], normalized[2])


def _validate_parameter_definitions(definitions: tuple[ParameterDefinition, ...]) -> None:
    """Fail fast on local schema corruption instead of silently mis-decoding device output."""

    seen_names: set[str] = set()
    seen_commands: set[CommandKey] = set()

    for name, resid, cmdid, value_count, access_mode, _value_type, description in definitions:
        if not name:
            raise ValueError("ReSpeaker parameter name must not be empty.")
        if name in seen_names:
            raise ValueError(f"Duplicate ReSpeaker parameter name: {name}.")
        seen_names.add(name)

        if isinstance(resid, bool) or not isinstance(resid, int) or not (0 <= resid <= 0xFFFF):
            raise ValueError(f"{name} has invalid resid: {resid}.")
        if isinstance(cmdid, bool) or not isinstance(cmdid, int) or not (0 <= cmdid <= 0x7F):
            raise ValueError(f"{name} has invalid cmdid: {cmdid}.")
        if isinstance(value_count, bool) or not isinstance(value_count, int) or value_count <= 0:
            raise ValueError(f"{name} must declare at least one value.")
        if access_mode not in {"ro", "rw", "wo"}:
            raise ValueError(f"{name} has unsupported access_mode: {access_mode}.")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"{name} must include a non-empty description.")

        command_key = (resid, cmdid)
        if command_key in seen_commands:
            raise ValueError(
                f"Duplicate ReSpeaker command tuple detected for {name}: "
                f"resid={resid}, cmdid={cmdid}."
            )
        seen_commands.add(command_key)


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


def _make_default_snapshot_entry(
    definition: ParameterDefinition,
    *,
    minimum_semantic_firmware: FirmwareVersion | None = None,
    semantic_guard_reason: str | None = None,
) -> DefaultSnapshotParameterEntry:
    return DefaultSnapshotParameterEntry(
        spec=_build_parameter_spec(definition),
        minimum_semantic_firmware=minimum_semantic_firmware,
        semantic_guard_reason=semantic_guard_reason,
    )


DEFAULT_RESPEAKER_SNAPSHOT_PARAMETER_CATALOG: Final[tuple[DefaultSnapshotParameterEntry, ...]] = (
    _make_default_snapshot_entry(_VERSION_PARAMETER_DEFINITION),
    _make_default_snapshot_entry(_DOA_VALUE_PARAMETER_DEFINITION),
    _make_default_snapshot_entry(_AEC_AZIMUTH_VALUES_PARAMETER_DEFINITION),
    _make_default_snapshot_entry(_AEC_SPENERGY_VALUES_PARAMETER_DEFINITION),
    _make_default_snapshot_entry(
        _AUDIO_MGR_SELECTED_AZIMUTHS_PARAMETER_DEFINITION,
        minimum_semantic_firmware=(2, 0, 1),
        semantic_guard_reason=(
            "AUDIO_MGR_SELECTED_AZIMUTHS changes meaning on XVF3800 firmware <= 2.0.0."
        ),
    ),
    _make_default_snapshot_entry(_GPO_READ_VALUES_PARAMETER_DEFINITION),
)

DEFAULT_RESPEAKER_PARAMETER_SPECS: Final[tuple[ReSpeakerParameterSpec, ...]] = tuple(
    entry.spec for entry in DEFAULT_RESPEAKER_SNAPSHOT_PARAMETER_CATALOG
)
DEFAULT_RESPEAKER_PARAMETER_NAMES: Final[tuple[str, ...]] = tuple(
    spec.name for spec in DEFAULT_RESPEAKER_PARAMETER_SPECS
)

_DEFAULT_RESPEAKER_PARAMETER_SPECS_BY_NAME: Final[dict[str, ReSpeakerParameterSpec]] = {
    spec.name: spec for spec in DEFAULT_RESPEAKER_PARAMETER_SPECS
}
DEFAULT_RESPEAKER_PARAMETER_SPECS_BY_NAME: Final[Mapping[str, ReSpeakerParameterSpec]] = (
    MappingProxyType(_DEFAULT_RESPEAKER_PARAMETER_SPECS_BY_NAME)
)

_DEFAULT_RESPEAKER_PARAMETER_SPECS_BY_COMMAND: Final[dict[CommandKey, ReSpeakerParameterSpec]] = {
    (spec.resid, spec.cmdid): spec for spec in DEFAULT_RESPEAKER_PARAMETER_SPECS
}
DEFAULT_RESPEAKER_PARAMETER_SPECS_BY_COMMAND: Final[
    Mapping[CommandKey, ReSpeakerParameterSpec]
] = MappingProxyType(_DEFAULT_RESPEAKER_PARAMETER_SPECS_BY_COMMAND)

_DEFAULT_RESPEAKER_SNAPSHOT_ENTRIES_BY_NAME: Final[dict[str, DefaultSnapshotParameterEntry]] = {
    entry.spec.name: entry for entry in DEFAULT_RESPEAKER_SNAPSHOT_PARAMETER_CATALOG
}
DEFAULT_RESPEAKER_SNAPSHOT_ENTRIES_BY_NAME: Final[
    Mapping[str, DefaultSnapshotParameterEntry]
] = MappingProxyType(_DEFAULT_RESPEAKER_SNAPSHOT_ENTRIES_BY_NAME)

_DEFAULT_RESPEAKER_SEMANTIC_MIN_FIRMWARE_BY_NAME: Final[dict[str, FirmwareVersion]] = {
    entry.spec.name: entry.minimum_semantic_firmware
    for entry in DEFAULT_RESPEAKER_SNAPSHOT_PARAMETER_CATALOG
    if entry.minimum_semantic_firmware is not None
}
DEFAULT_RESPEAKER_SEMANTIC_MIN_FIRMWARE_BY_NAME: Final[
    Mapping[str, FirmwareVersion]
] = MappingProxyType(_DEFAULT_RESPEAKER_SEMANTIC_MIN_FIRMWARE_BY_NAME)

VERSION_PARAMETER = DEFAULT_RESPEAKER_PARAMETER_SPECS_BY_NAME["VERSION"]
AEC_AZIMUTH_VALUES_PARAMETER = DEFAULT_RESPEAKER_PARAMETER_SPECS_BY_NAME["AEC_AZIMUTH_VALUES"]
AEC_SPENERGY_VALUES_PARAMETER = DEFAULT_RESPEAKER_PARAMETER_SPECS_BY_NAME["AEC_SPENERGY_VALUES"]
AUDIO_MGR_SELECTED_AZIMUTHS_PARAMETER = DEFAULT_RESPEAKER_PARAMETER_SPECS_BY_NAME[
    "AUDIO_MGR_SELECTED_AZIMUTHS"
]
DOA_VALUE_PARAMETER = DEFAULT_RESPEAKER_PARAMETER_SPECS_BY_NAME["DOA_VALUE"]
GPO_READ_VALUES_PARAMETER = DEFAULT_RESPEAKER_PARAMETER_SPECS_BY_NAME["GPO_READ_VALUES"]


def get_default_respeaker_parameter_spec(name: str) -> ReSpeakerParameterSpec:
    """Return one canonical default snapshot spec by name."""

    return DEFAULT_RESPEAKER_PARAMETER_SPECS_BY_NAME[name]


def get_default_respeaker_parameter_spec_by_command(
    resid: int,
    cmdid: int,
) -> ReSpeakerParameterSpec | None:
    """Return one canonical default snapshot spec by XVF3800 command tuple."""

    return DEFAULT_RESPEAKER_PARAMETER_SPECS_BY_COMMAND.get((resid, cmdid))


def get_default_respeaker_parameter_specs() -> tuple[ReSpeakerParameterSpec, ...]:
    """Return the canonical bounded default snapshot spec tuple."""

    return DEFAULT_RESPEAKER_PARAMETER_SPECS


def _extract_firmware_version(
    reads: Mapping[str, ReSpeakerParameterRead],
) -> FirmwareVersion | None:
    version_read = reads.get("VERSION")
    if version_read is None or not version_read.ok:
        return None

    decoded = version_read.decoded_value
    if not isinstance(decoded, tuple) or len(decoded) != 3:
        return None

    version: list[int] = []
    for index, component in enumerate(decoded):
        if isinstance(component, bool) or not isinstance(component, int):
            return None
        if component < 0:
            return None
        version.append(component)

    return (version[0], version[1], version[2])


def _format_firmware_version(version: FirmwareVersion) -> str:
    return ".".join(str(component) for component in version)


def assess_default_respeaker_snapshot_semantics(
    reads: Mapping[str, ReSpeakerParameterRead],
) -> tuple[str, ...]:
    """Return human-readable semantic compatibility issues for one default snapshot."""

    firmware_version = _extract_firmware_version(reads)
    if firmware_version is None:
        return ()

    issues: list[str] = []
    for entry in DEFAULT_RESPEAKER_SNAPSHOT_PARAMETER_CATALOG:
        minimum_semantic_firmware = entry.minimum_semantic_firmware
        if minimum_semantic_firmware is None or firmware_version >= minimum_semantic_firmware:
            continue

        current_read = reads.get(entry.spec.name)
        if current_read is None or not current_read.ok:
            continue

        issues.append(
            f"{entry.spec.name} requires firmware >= "
            f"{_format_firmware_version(minimum_semantic_firmware)} "
            f"but device reports {_format_firmware_version(firmware_version)}"
        )

    return tuple(issues)


def _apply_default_respeaker_semantic_guards(
    reads: dict[str, ReSpeakerParameterRead],
) -> None:
    firmware_version = _extract_firmware_version(reads)
    if firmware_version is None:
        return

    for entry in DEFAULT_RESPEAKER_SNAPSHOT_PARAMETER_CATALOG:
        minimum_semantic_firmware = entry.minimum_semantic_firmware
        if minimum_semantic_firmware is None or firmware_version >= minimum_semantic_firmware:
            continue

        current_read = reads.get(entry.spec.name)
        if current_read is None or not current_read.ok:
            continue

        reads[entry.spec.name] = ReSpeakerParameterRead(
            spec=current_read.spec,
            captured_at=current_read.captured_at,
            ok=False,
            attempt_count=current_read.attempt_count,
            error=(
                "unsupported_firmware_semantics:"
                f"{entry.spec.name}:requires>={_format_firmware_version(minimum_semantic_firmware)}:"
                f"got={_format_firmware_version(firmware_version)}"
            ),
        )


def _maybe_add_exception_note(exc: BaseException, note: str) -> None:
    add_note = getattr(exc, "add_note", None)
    if callable(add_note):
        add_note(note)


def read_default_respeaker_parameters(
    transport: ReSpeakerReadTransport,
    *,
    probe: ReSpeakerProbeResult | None = None,
    strict_firmware_semantics: bool = True,
) -> tuple[ReSpeakerTransportAvailability, dict[str, ReSpeakerParameterRead]]:
    """Read the default bounded primitive parameter set from one XVF3800."""

    try:
        availability, reads = transport.capture_reads(
            DEFAULT_RESPEAKER_PARAMETER_SPECS,
            probe=probe,
        )
        reads = dict(reads)

        if strict_firmware_semantics:
            # BREAKING: On XVF3800 firmware <= 2.0.0, AUDIO_MGR_SELECTED_AZIMUTHS
            # had a different default semantic. Twinr now fails that read closed
            # instead of returning a misleading "ok" value.
            _apply_default_respeaker_semantic_guards(reads)

        return availability, reads
    except Exception as exc:
        _maybe_add_exception_note(
            exc,
            "Twinr XVF3800 default snapshot read failed for VERSION, DOA_VALUE, "
            "AEC_AZIMUTH_VALUES, AEC_SPENERGY_VALUES, "
            "AUDIO_MGR_SELECTED_AZIMUTHS, and GPO_READ_VALUES.",
        )
        raise


__all__ = (
    "AUDIO_MGR_SELECTED_AZIMUTHS_PARAMETER",
    "AEC_AZIMUTH_VALUES_PARAMETER",
    "AEC_SPENERGY_VALUES_PARAMETER",
    "CommandKey",
    "DEFAULT_RESPEAKER_PARAMETER_NAMES",
    "DEFAULT_RESPEAKER_PARAMETER_SPECS",
    "DEFAULT_RESPEAKER_PARAMETER_SPECS_BY_COMMAND",
    "DEFAULT_RESPEAKER_PARAMETER_SPECS_BY_NAME",
    "DEFAULT_RESPEAKER_SEMANTIC_MIN_FIRMWARE_BY_NAME",
    "DEFAULT_RESPEAKER_SNAPSHOT_ENTRIES_BY_NAME",
    "DEFAULT_RESPEAKER_SNAPSHOT_PARAMETER_CATALOG",
    "DOA_VALUE_PARAMETER",
    "DefaultSnapshotParameterEntry",
    "FirmwareVersion",
    "GPO_READ_VALUES_PARAMETER",
    "ParameterDefinition",
    "ReSpeakerReadTransport",
    "VERSION_PARAMETER",
    "assess_default_respeaker_snapshot_semantics",
    "get_default_respeaker_parameter_spec",
    "get_default_respeaker_parameter_spec_by_command",
    "get_default_respeaker_parameter_specs",
    "read_default_respeaker_parameters",
)