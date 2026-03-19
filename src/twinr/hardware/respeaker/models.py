"""Typed models for ReSpeaker XVF3800 probing and host-control reads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
import re
from typing import TypeVar

_ValueT = TypeVar("_ValueT")

# AUDIT-FIX(#2): Restrict generated ALSA identifiers to an ALSA-safe card-id subset and
# fall back to numeric card/device indices when the parsed identifier is malformed.
_ALSA_CARD_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

# AUDIT-FIX(#3): Bound control-transfer payload sizes to the USB control-transfer limit.
_USB_CONTROL_TRANSFER_MAX_LENGTH = 0xFFFF


# AUDIT-FIX(#1): Freeze mapping payloads without breaking dict/json/dataclasses.asdict
# compatibility for file-backed state and telemetry serialization.
class FrozenDict(dict[str, _ValueT]):
    """Minimal immutable dict for snapshot payloads."""

    def _immutable(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(f"{self.__class__.__name__} is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def copy(self) -> dict[str, _ValueT]:
        """Return a mutable shallow copy."""

        return dict(self)

    def __hash__(self) -> int:
        """Hash the mapping by content instead of insertion order."""

        return hash(frozenset(self.items()))


# AUDIT-FIX(#3): Centralize strict scalar validation for control-transfer metadata so
# malformed device specs fail fast instead of generating invalid USB requests.
def _require_int(
    name: str,
    value: object,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _require_non_empty_str(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value


def _require_finite_float(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return normalized


def _coerce_sequence(name: str, value: object) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    raise TypeError(f"{name} must be a tuple or sequence")


# AUDIT-FIX(#4): Validate decoded payloads against the declared parameter spec so
# success/failure state cannot drift away from the underlying USB response.
def _normalize_decoded_value(
    spec: "ReSpeakerParameterSpec",
    decoded_value: tuple[int | float | None, ...] | str | None,
) -> tuple[int | float | None, ...] | str | None:
    if decoded_value is None:
        return None

    if isinstance(decoded_value, str):
        if spec.value_type is not ReSpeakerParameterType.CHAR:
            raise TypeError("decoded_value strings are only valid for CHAR parameters")
        if len(decoded_value.encode("utf-8")) > spec.value_count:
            raise ValueError("decoded_value exceeds declared CHAR payload length")
        return decoded_value

    values = _coerce_sequence("decoded_value", decoded_value)
    if len(values) != spec.value_count:
        raise ValueError(
            "decoded_value length does not match spec.value_count "
            f"({len(values)} != {spec.value_count})"
        )

    if spec.value_type is ReSpeakerParameterType.UINT8:
        return tuple(
            _require_int(f"decoded_value[{idx}]", value, minimum=0, maximum=0xFF)
            for idx, value in enumerate(values)
        )
    if spec.value_type is ReSpeakerParameterType.UINT16:
        return tuple(
            _require_int(f"decoded_value[{idx}]", value, minimum=0, maximum=0xFFFF)
            for idx, value in enumerate(values)
        )
    if spec.value_type is ReSpeakerParameterType.UINT32:
        return tuple(
            _require_int(f"decoded_value[{idx}]", value, minimum=0, maximum=0xFFFFFFFF)
            for idx, value in enumerate(values)
        )
    if spec.value_type is ReSpeakerParameterType.INT32:
        return tuple(
            _require_int(
                f"decoded_value[{idx}]",
                value,
                minimum=-(2**31),
                maximum=(2**31) - 1,
            )
            for idx, value in enumerate(values)
        )
    if spec.value_type is ReSpeakerParameterType.CHAR:
        return tuple(
            _require_int(f"decoded_value[{idx}]", value, minimum=0, maximum=0xFF)
            for idx, value in enumerate(values)
        )
    # XMOS float-style parameters can expose NaN as a device sentinel for
    # "no selected azimuth". Preserve that as ``None`` at the raw-read edge so
    # later snapshot projections stay serializable and deterministic.
    return tuple(_normalize_optional_device_float(f"decoded_value[{idx}]", value) for idx, value in enumerate(values))


def _normalize_optional_device_float(name: str, value: object) -> float | None:
    """Normalize one device float, treating NaN as an explicit unknown."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number or None")
    normalized = float(value)
    if normalized != normalized:
        return None
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite or NaN")
    return normalized


# AUDIT-FIX(#6): Coerce tuple-typed snapshot fields at runtime and reject NaN/Inf or
# invalid GPIO levels before the state reaches persistence or orchestration layers.
def _normalize_optional_float_tuple(name: str, value: object) -> tuple[float | None, ...] | None:
    if value is None:
        return None
    values = _coerce_sequence(name, value)
    normalized: list[float | None] = []
    for idx, item in enumerate(values):
        if item is None:
            normalized.append(None)
            continue
        normalized.append(_require_finite_float(f"{name}[{idx}]", item))
    return tuple(normalized)


def _normalize_optional_logic_level_tuple(name: str, value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    values = _coerce_sequence(name, value)
    normalized: list[int] = []
    for idx, item in enumerate(values):
        level = _require_int(f"{name}[{idx}]", item, minimum=0, maximum=1)
        normalized.append(level)
    return tuple(normalized)


def _normalize_optional_version_tuple(name: str, value: object) -> tuple[int, int, int] | None:
    if value is None:
        return None
    values = _coerce_sequence(name, value)
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three integers")
    normalized = tuple(
        _require_int(f"{name}[{idx}]", item, minimum=0)
        for idx, item in enumerate(values)
    )
    return normalized  # type: ignore[return-value]


def _freeze_raw_reads(value: object) -> FrozenDict["ReSpeakerParameterRead"]:
    if not isinstance(value, Mapping):
        raise TypeError("raw_reads must be a mapping")
    normalized = FrozenDict(value)
    for key, item in normalized.items():
        _require_non_empty_str("raw_reads key", key)
        if not isinstance(item, ReSpeakerParameterRead):
            raise TypeError("raw_reads values must be ReSpeakerParameterRead instances")
    return normalized


class ReSpeakerParameterType(StrEnum):
    """Describe the host-control payload encoding for one parameter."""

    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    INT32 = "int32"
    FLOAT = "float"
    RADIANS = "radians"
    CHAR = "char"


class ReSpeakerMemoryClass(StrEnum):
    """Describe how one ReSpeaker-derived fact may be used downstream."""

    EPHEMERAL_STATE = "ephemeral_state"
    SESSION_MEMORY = "session_memory"
    OBSERVED_PREFERENCE = "observed_preference"
    CONFIRMED_PREFERENCE = "confirmed_preference"


@dataclass(frozen=True, slots=True)
class ReSpeakerUsbDevice:
    """Describe one XVF3800 USB enumeration row."""

    bus: str | None
    device: str | None
    vendor_id: str
    product_id: str
    description: str
    raw_line: str


@dataclass(frozen=True, slots=True)
class ReSpeakerCaptureDevice:
    """Describe one XVF3800 ALSA capture device row."""

    card_index: int | None
    card_name: str
    card_label: str
    device_index: int | None
    raw_line: str

    def __post_init__(self) -> None:
        # AUDIT-FIX(#2): Reject negative ALSA indices so invalid fallback hw:<card>,<dev>
        # identifiers are never generated from malformed parser output.
        if self.card_index is not None:
            _require_int("card_index", self.card_index, minimum=0)
        if self.device_index is not None:
            _require_int("device_index", self.device_index, minimum=0)

    @property
    def hw_identifier(self) -> str | None:
        """Return one stable ALSA identifier when possible."""

        safe_card_name = self.card_name.strip()

        # AUDIT-FIX(#2): Use the symbolic CARD form only for ALSA-safe identifiers and
        # otherwise fall back to numeric hw:<card>[,<dev>] addressing.
        if safe_card_name and _ALSA_CARD_NAME_RE.fullmatch(safe_card_name):
            if self.device_index is None:
                return f"hw:CARD={safe_card_name}"
            return f"hw:CARD={safe_card_name},DEV={self.device_index}"
        if self.card_index is None:
            return None
        if self.device_index is None:
            return f"hw:{self.card_index}"
        return f"hw:{self.card_index},{self.device_index}"


@dataclass(frozen=True, slots=True)
class ReSpeakerProbeResult:
    """Describe the current host-visible XVF3800 USB and ALSA state."""

    usb_device: ReSpeakerUsbDevice | None
    capture_device: ReSpeakerCaptureDevice | None
    lsusb_available: bool
    arecord_available: bool

    @property
    def usb_visible(self) -> bool:
        """Return whether the XVF3800 is visible on the USB bus."""

        return self.usb_device is not None

    @property
    def capture_ready(self) -> bool:
        """Return whether an ALSA capture card was found for the XVF3800."""

        return self.capture_device is not None

    @property
    def state(self) -> str:
        """Return one conservative high-level runtime state label."""

        # AUDIT-FIX(#5): Require arecord availability before reporting audio_ready so
        # inconsistent hydrated objects cannot mask a degraded host probe.
        if self.capture_ready and self.arecord_available:
            return "audio_ready"
        if self.usb_visible and self.arecord_available:
            return "usb_visible_no_capture"
        if self.capture_ready:
            return "usb_visible_capture_unknown" if self.usb_visible else "not_detected"
        if self.usb_visible:
            return "usb_visible_capture_unknown"
        if not self.lsusb_available and not self.arecord_available:
            return "probe_unavailable"
        return "not_detected"


@dataclass(frozen=True, slots=True)
class ReSpeakerParameterSpec:
    """Describe one official XVF3800 host-control read parameter."""

    name: str
    resid: int
    cmdid: int
    value_count: int
    access_mode: str
    value_type: ReSpeakerParameterType
    description: str

    def __post_init__(self) -> None:
        # AUDIT-FIX(#3): Enforce spec invariants up front so malformed metadata cannot
        # produce impossible USB request values or oversized read lengths later.
        _require_non_empty_str("name", self.name)
        _require_non_empty_str("access_mode", self.access_mode)
        _require_int("resid", self.resid, minimum=0, maximum=0xFFFF)
        _require_int("cmdid", self.cmdid, minimum=0, maximum=0x7F)
        _require_int("value_count", self.value_count, minimum=0)
        if not isinstance(self.value_type, ReSpeakerParameterType):
            object.__setattr__(self, "value_type", ReSpeakerParameterType(self.value_type))
        if self.read_length > _USB_CONTROL_TRANSFER_MAX_LENGTH:
            raise ValueError("read_length exceeds the USB control-transfer size limit")

    @property
    def request_value(self) -> int:
        """Return the read request value used by XMOS vendor control reads."""

        return 0x80 | int(self.cmdid)

    @property
    def read_length(self) -> int:
        """Return the expected read payload length including status byte."""

        if self.value_type in (ReSpeakerParameterType.UINT8, ReSpeakerParameterType.CHAR):
            return self.value_count + 1
        if self.value_type is ReSpeakerParameterType.UINT16:
            return (self.value_count * 2) + 1
        return (self.value_count * 4) + 1


@dataclass(frozen=True, slots=True)
class ReSpeakerParameterRead:
    """Store one bounded host-control read result."""

    spec: ReSpeakerParameterSpec
    captured_at: float
    ok: bool
    attempt_count: int
    status_code: int | None = None
    decoded_value: tuple[int | float | None, ...] | str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        # AUDIT-FIX(#4): Fail fast on contradictory result states so downstream logic can
        # trust ok/error/status/decoded_value to describe one coherent probe outcome.
        if not isinstance(self.spec, ReSpeakerParameterSpec):
            raise TypeError("spec must be a ReSpeakerParameterSpec")
        object.__setattr__(self, "captured_at", _require_finite_float("captured_at", self.captured_at))
        _require_int("attempt_count", self.attempt_count, minimum=1)
        if self.status_code is not None:
            _require_int("status_code", self.status_code, minimum=0, maximum=0xFF)
        if self.ok and self.error is not None:
            raise ValueError("successful reads must not include an error")
        if self.ok and self.status_code not in (None, 0):
            raise ValueError("successful reads must use status_code 0 or None")
        if not self.ok and self.error is None and self.status_code is None:
            raise ValueError("failed reads must include error text or a status_code")
        object.__setattr__(self, "decoded_value", _normalize_decoded_value(self.spec, self.decoded_value))


@dataclass(frozen=True, slots=True)
class ReSpeakerTransportAvailability:
    """Describe whether host-control transport is usable right now."""

    backend: str
    available: bool
    reason: str | None = None
    requires_elevated_permissions: bool = False

    def __post_init__(self) -> None:
        # AUDIT-FIX(#6): Reject blank backend identifiers so persisted transport state
        # always remains attributable to one concrete probing backend.
        _require_non_empty_str("backend", self.backend)


@dataclass(frozen=True, slots=True)
class ReSpeakerDirectionSnapshot:
    """Store typed directional primitives derived from host-control reads."""

    captured_at: float
    speech_detected: bool | None = None
    room_quiet: bool | None = None
    doa_degrees: int | None = None
    beam_azimuth_degrees: tuple[float | None, ...] | None = None
    beam_speech_energies: tuple[float | None, ...] | None = None
    selected_azimuth_degrees: tuple[float | None, ...] | None = None

    def __post_init__(self) -> None:
        # AUDIT-FIX(#6): Normalize tuple-typed directional data to immutable validated
        # tuples and reject non-finite values before they reach beam-selection logic.
        object.__setattr__(self, "captured_at", _require_finite_float("captured_at", self.captured_at))
        if self.doa_degrees is not None:
            _require_int("doa_degrees", self.doa_degrees)
        object.__setattr__(
            self,
            "beam_azimuth_degrees",
            _normalize_optional_float_tuple("beam_azimuth_degrees", self.beam_azimuth_degrees),
        )
        object.__setattr__(
            self,
            "beam_speech_energies",
            _normalize_optional_float_tuple("beam_speech_energies", self.beam_speech_energies),
        )
        object.__setattr__(
            self,
            "selected_azimuth_degrees",
            _normalize_optional_float_tuple("selected_azimuth_degrees", self.selected_azimuth_degrees),
        )
        if (
            self.beam_azimuth_degrees is not None
            and self.beam_speech_energies is not None
            and len(self.beam_azimuth_degrees) != len(self.beam_speech_energies)
        ):
            raise ValueError("beam_azimuth_degrees and beam_speech_energies must have equal length")


@dataclass(frozen=True, slots=True)
class ReSpeakerMuteSnapshot:
    """Store typed mute-adjacent primitives derived from host-control reads."""

    captured_at: float
    mute_active: bool | None = None
    gpo_logic_levels: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        # AUDIT-FIX(#6): Normalize GPIO state to immutable binary tuples so mute state
        # snapshots cannot be mutated or polluted with invalid logic levels.
        object.__setattr__(self, "captured_at", _require_finite_float("captured_at", self.captured_at))
        object.__setattr__(
            self,
            "gpo_logic_levels",
            _normalize_optional_logic_level_tuple("gpo_logic_levels", self.gpo_logic_levels),
        )


@dataclass(frozen=True, slots=True)
class ReSpeakerPrimitiveSnapshot:
    """Aggregate one typed XVF3800 primitive snapshot."""

    captured_at: float
    probe: ReSpeakerProbeResult
    transport: ReSpeakerTransportAvailability
    firmware_version: tuple[int, int, int] | None
    direction: ReSpeakerDirectionSnapshot
    mute: ReSpeakerMuteSnapshot
    raw_reads: Mapping[str, ReSpeakerParameterRead] = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        # AUDIT-FIX(#1): Freeze raw_reads so frozen snapshots stay immutable even when
        # shared across async tasks, caches, and file-backed state refreshes.
        object.__setattr__(self, "raw_reads", _freeze_raw_reads(self.raw_reads))

        # AUDIT-FIX(#6): Normalize version/timestamp payloads and reject wrong nested
        # object types before they can poison persisted primitive snapshots.
        object.__setattr__(self, "captured_at", _require_finite_float("captured_at", self.captured_at))
        object.__setattr__(
            self,
            "firmware_version",
            _normalize_optional_version_tuple("firmware_version", self.firmware_version),
        )
        if not isinstance(self.probe, ReSpeakerProbeResult):
            raise TypeError("probe must be a ReSpeakerProbeResult")
        if not isinstance(self.transport, ReSpeakerTransportAvailability):
            raise TypeError("transport must be a ReSpeakerTransportAvailability")
        if not isinstance(self.direction, ReSpeakerDirectionSnapshot):
            raise TypeError("direction must be a ReSpeakerDirectionSnapshot")
        if not isinstance(self.mute, ReSpeakerMuteSnapshot):
            raise TypeError("mute must be a ReSpeakerMuteSnapshot")

    @property
    def device_runtime_mode(self) -> str:
        """Return the conservative runtime mode derived from the host probe."""

        return self.probe.state

    @property
    def host_control_ready(self) -> bool:
        """Return whether XVF3800 host-control reads succeeded."""

        return self.transport.available


@dataclass(frozen=True, slots=True)
class ReSpeakerClaimMetadata:
    """Describe the provenance and confidence for one ReSpeaker-derived claim."""

    captured_at: float
    source: str
    source_type: str
    confidence: float
    sensor_window_ms: int
    memory_class: ReSpeakerMemoryClass = ReSpeakerMemoryClass.EPHEMERAL_STATE
    session_id: int | None = None
    requires_confirmation: bool = False

    def __post_init__(self) -> None:
        """Normalize one claim metadata record into a bounded immutable form."""

        object.__setattr__(self, "captured_at", _require_finite_float("captured_at", self.captured_at))
        _require_non_empty_str("source", self.source)
        _require_non_empty_str("source_type", self.source_type)
        object.__setattr__(
            self,
            "confidence",
            _require_finite_float("confidence", self.confidence, minimum=0.0),
        )
        if self.confidence > 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        _require_int("sensor_window_ms", self.sensor_window_ms, minimum=0)
        if self.session_id is not None:
            _require_int("session_id", self.session_id, minimum=0)
        if not isinstance(self.memory_class, ReSpeakerMemoryClass):
            object.__setattr__(
                self,
                "memory_class",
                ReSpeakerMemoryClass(str(self.memory_class)),
            )

    def to_payload(self) -> dict[str, object]:
        """Serialize one claim metadata record into plain automation facts."""

        return {
            "captured_at": self.captured_at,
            "source": self.source,
            "source_type": self.source_type,
            "confidence": self.confidence,
            "sensor_window_ms": self.sensor_window_ms,
            "memory_class": self.memory_class.value,
            "session_id": self.session_id,
            "requires_confirmation": self.requires_confirmation,
        }

    def with_session_id(self, session_id: int | None) -> "ReSpeakerClaimMetadata":
        """Return one claim metadata copy with a session identifier applied."""

        return ReSpeakerClaimMetadata(
            captured_at=self.captured_at,
            source=self.source,
            source_type=self.source_type,
            confidence=self.confidence,
            sensor_window_ms=self.sensor_window_ms,
            memory_class=self.memory_class,
            session_id=session_id,
            requires_confirmation=self.requires_confirmation,
        )


def _freeze_claim_contract(value: object) -> FrozenDict[ReSpeakerClaimMetadata]:
    """Freeze a ReSpeaker claim-contract mapping into an immutable dictionary."""

    if not isinstance(value, Mapping):
        raise TypeError("claim_contract must be a mapping")
    normalized = FrozenDict(value)
    for key, item in normalized.items():
        _require_non_empty_str("claim_contract key", key)
        if not isinstance(item, ReSpeakerClaimMetadata):
            raise TypeError("claim_contract values must be ReSpeakerClaimMetadata instances")
    return normalized


@dataclass(frozen=True, slots=True)
class ReSpeakerSignalSnapshot:
    """Store one runtime-facing XVF3800 signal observation."""

    captured_at: float
    source: str
    source_type: str
    sensor_window_ms: int
    device_runtime_mode: str
    host_control_ready: bool
    transport_reason: str | None = None
    requires_elevated_permissions: bool = False
    firmware_version: tuple[int, int, int] | None = None
    speech_detected: bool | None = None
    room_quiet: bool | None = None
    recent_speech_age_s: float | None = None
    assistant_output_active: bool | None = None
    azimuth_deg: int | None = None
    direction_confidence: float | None = None
    beam_activity: tuple[float | None, ...] | None = None
    speech_overlap_likely: bool | None = None
    barge_in_detected: bool | None = None
    mute_active: bool | None = None
    gpo_logic_levels: tuple[int, ...] | None = None
    claim_contract: Mapping[str, ReSpeakerClaimMetadata] = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        # AUDIT-FIX(#6): Normalize runtime-facing snapshot fields so downstream voice
        # policy sees immutable, bounded, and attributable signal data.
        object.__setattr__(self, "captured_at", _require_finite_float("captured_at", self.captured_at))
        _require_non_empty_str("source", self.source)
        _require_non_empty_str("source_type", self.source_type)
        _require_non_empty_str("device_runtime_mode", self.device_runtime_mode)
        _require_int("sensor_window_ms", self.sensor_window_ms, minimum=0)
        object.__setattr__(
            self,
            "firmware_version",
            _normalize_optional_version_tuple("firmware_version", self.firmware_version),
        )
        if self.recent_speech_age_s is not None:
            object.__setattr__(
                self,
                "recent_speech_age_s",
                _require_finite_float("recent_speech_age_s", self.recent_speech_age_s, minimum=0.0),
            )
        if self.azimuth_deg is not None:
            _require_int("azimuth_deg", self.azimuth_deg)
        if self.direction_confidence is not None:
            direction_confidence = _require_finite_float("direction_confidence", self.direction_confidence)
            if not 0.0 <= direction_confidence <= 1.0:
                raise ValueError("direction_confidence must be between 0.0 and 1.0")
            object.__setattr__(self, "direction_confidence", direction_confidence)
        object.__setattr__(
            self,
            "beam_activity",
            _normalize_optional_float_tuple("beam_activity", self.beam_activity),
        )
        object.__setattr__(
            self,
            "gpo_logic_levels",
            _normalize_optional_logic_level_tuple("gpo_logic_levels", self.gpo_logic_levels),
        )
        object.__setattr__(self, "claim_contract", _freeze_claim_contract(self.claim_contract))
