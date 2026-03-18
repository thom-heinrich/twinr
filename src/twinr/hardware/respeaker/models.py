"""Typed models for ReSpeaker XVF3800 probing and host-control reads."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ReSpeakerParameterType(StrEnum):
    """Describe the host-control payload encoding for one parameter."""

    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    INT32 = "int32"
    FLOAT = "float"
    RADIANS = "radians"
    CHAR = "char"


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

    @property
    def hw_identifier(self) -> str | None:
        """Return one stable ALSA ``hw:CARD=...`` identifier when possible."""

        if not self.card_name:
            return None
        if self.device_index is None:
            return f"hw:CARD={self.card_name}"
        return f"hw:CARD={self.card_name},DEV={self.device_index}"


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

        if self.capture_ready:
            return "audio_ready"
        if self.usb_visible and self.arecord_available:
            return "usb_visible_no_capture"
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
    decoded_value: tuple[int | float, ...] | str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ReSpeakerTransportAvailability:
    """Describe whether host-control transport is usable right now."""

    backend: str
    available: bool
    reason: str | None = None
    requires_elevated_permissions: bool = False


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


@dataclass(frozen=True, slots=True)
class ReSpeakerMuteSnapshot:
    """Store typed mute-adjacent primitives derived from host-control reads."""

    captured_at: float
    mute_active: bool | None = None
    gpo_logic_levels: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class ReSpeakerPrimitiveSnapshot:
    """Aggregate one typed XVF3800 primitive snapshot."""

    captured_at: float
    probe: ReSpeakerProbeResult
    transport: ReSpeakerTransportAvailability
    firmware_version: tuple[int, int, int] | None
    direction: ReSpeakerDirectionSnapshot
    mute: ReSpeakerMuteSnapshot
    raw_reads: dict[str, ReSpeakerParameterRead] = field(default_factory=dict)

    @property
    def device_runtime_mode(self) -> str:
        """Return the conservative runtime mode derived from the host probe."""

        return self.probe.state

    @property
    def host_control_ready(self) -> bool:
        """Return whether XVF3800 host-control reads succeeded."""

        return self.transport.available


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
    azimuth_deg: int | None = None
    beam_activity: tuple[float | None, ...] | None = None
    mute_active: bool | None = None
    gpo_logic_levels: tuple[int, ...] | None = None
