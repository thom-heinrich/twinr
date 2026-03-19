"""Capture typed XVF3800 primitive snapshots for Twinr."""

from __future__ import annotations

import math
import operator  # AUDIT-FIX(#7): Use exact integer coercion instead of lossy int(...) truncation.
import threading  # AUDIT-FIX(#2): Serialize hardware access to prevent interleaved USB reads.
import time
from collections.abc import Mapping  # AUDIT-FIX(#6): Defensively normalize raw read mappings.
from typing import cast  # AUDIT-FIX(#6): Keep normalized raw read values precisely typed.

from twinr.hardware.respeaker.models import (
    ReSpeakerDirectionSnapshot,
    ReSpeakerMuteSnapshot,
    ReSpeakerParameterRead,
    ReSpeakerPrimitiveSnapshot,
    ReSpeakerProbeResult,
)
from twinr.hardware.respeaker.probe import probe_respeaker_xvf3800
from twinr.hardware.respeaker.raw_reads import read_default_respeaker_parameters
from twinr.hardware.respeaker.transport import ReSpeakerLibusbTransport

_CAPTURE_LOCK = threading.Lock()  # AUDIT-FIX(#2): Hardware snapshots must not overlap on the same USB device.
_MAX_CAPTURED_AT_FUTURE_SKEW_SECONDS = 5.0  # AUDIT-FIX(#6): Ignore obviously bogus future timestamps.


def capture_respeaker_primitive_snapshot(
    *,
    transport: ReSpeakerLibusbTransport | None = None,
    probe: ReSpeakerProbeResult | None = None,
) -> ReSpeakerPrimitiveSnapshot:
    """Capture one XVF3800 primitive snapshot.

    The snapshot remains conservative. It surfaces only primitives that are
    directly readable from the XVF3800 today and leaves malformed or
    unsupported values as explicit ``None`` instead of guessed values.
    """
    # AUDIT-FIX(#8): Align the public contract with the actual mute-pin interpretation below.

    owned_transport: ReSpeakerLibusbTransport | None = None

    with _CAPTURE_LOCK:  # AUDIT-FIX(#2): Prevent concurrent probe/read cycles from interleaving.
        resolved_probe = probe if probe is not None else probe_respeaker_xvf3800()  # AUDIT-FIX(#5): Only auto-probe on an explicit None.
        if transport is None:
            owned_transport = ReSpeakerLibusbTransport()  # AUDIT-FIX(#3): Track ownership so internally created transports are always closed.
            resolved_transport = owned_transport
        else:
            resolved_transport = transport

        try:
            availability, raw_reads = read_default_respeaker_parameters(
                resolved_transport,
                probe=resolved_probe,
            )
            raw_reads_snapshot = _normalized_raw_reads(raw_reads)  # AUDIT-FIX(#6): Filter malformed payloads before field access.
            capture_completed_at = time.time()  # AUDIT-FIX(#4): Timestamp after hardware I/O completes, not before it starts.

            direction_time = _latest_read_time(raw_reads_snapshot, fallback=capture_completed_at)
            direction = ReSpeakerDirectionSnapshot(
                captured_at=direction_time,
                speech_detected=_speech_detected(raw_reads_snapshot.get("DOA_VALUE")),
                room_quiet=_room_quiet(raw_reads_snapshot.get("DOA_VALUE")),
                doa_degrees=_doa_degrees(raw_reads_snapshot.get("DOA_VALUE")),
                beam_azimuth_degrees=_degrees_from_radian_read(raw_reads_snapshot.get("AEC_AZIMUTH_VALUES")),
                beam_speech_energies=_float_tuple(raw_reads_snapshot.get("AEC_SPENERGY_VALUES"), expected_count=4),
                selected_azimuth_degrees=_degrees_from_radian_read(raw_reads_snapshot.get("AUDIO_MGR_SELECTED_AZIMUTHS")),
            )
            gpo_logic_levels = _logic_level_tuple(
                raw_reads_snapshot.get("GPO_READ_VALUES"),
                expected_count=5,
            )  # AUDIT-FIX(#1): Accept only real 0/1 logic levels for mute-related pins.
            mute = ReSpeakerMuteSnapshot(
                captured_at=direction_time,
                mute_active=_mute_active(raw_reads_snapshot.get("GPO_READ_VALUES")),
                gpo_logic_levels=gpo_logic_levels,
            )

            return ReSpeakerPrimitiveSnapshot(
                captured_at=capture_completed_at,
                probe=resolved_probe,
                transport=availability,
                firmware_version=_firmware_version(raw_reads_snapshot.get("VERSION")),
                direction=direction,
                mute=mute,
                raw_reads=dict(raw_reads_snapshot),  # AUDIT-FIX(#6): Return a stable snapshot copy, not the mutable source mapping.
            )
        finally:
            _close_transport_if_owned(owned_transport)  # AUDIT-FIX(#3): Prevent libusb handle leaks on success and failure.


def _normalized_raw_reads(raw_reads: object) -> dict[str, ReSpeakerParameterRead]:
    # AUDIT-FIX(#6): Accept only mapping entries that look like typed parameter reads.
    if not isinstance(raw_reads, Mapping):
        return {}

    normalized: dict[str, ReSpeakerParameterRead] = {}
    for key, value in raw_reads.items():
        if isinstance(key, str) and _is_parameter_read_like(value):
            normalized[key] = cast(ReSpeakerParameterRead, value)
    return normalized


def _latest_read_time(raw_reads: Mapping[str, ReSpeakerParameterRead], *, fallback: float) -> float:
    captured_times: list[float] = []
    latest_allowed_time = fallback + _MAX_CAPTURED_AT_FUTURE_SKEW_SECONDS  # AUDIT-FIX(#6): Drop impossible future timestamps instead of propagating them.
    for read in raw_reads.values():
        captured_at = _captured_at_or_none(read)
        if captured_at is None:
            continue
        if captured_at < 0.0 or captured_at > latest_allowed_time:
            continue
        captured_times.append(captured_at)
    return max(captured_times) if captured_times else fallback


def _firmware_version(read: ReSpeakerParameterRead | None) -> tuple[int, int, int] | None:
    values = _int_tuple(read, expected_count=3)
    if values is None:
        return None
    return values[0], values[1], values[2]


def _speech_detected(read: ReSpeakerParameterRead | None) -> bool | None:
    values = _int_tuple(read, expected_count=2)
    if values is None:
        return None
    return _binary_flag(values[1])  # AUDIT-FIX(#1): Reject corrupted flag values instead of treating any non-zero as True.


def _room_quiet(read: ReSpeakerParameterRead | None) -> bool | None:
    speech_detected = _speech_detected(read)
    if speech_detected is None:
        return None
    return not speech_detected


def _doa_degrees(read: ReSpeakerParameterRead | None) -> int | None:
    values = _int_tuple(read, expected_count=2)
    if values is None:
        return None
    return values[0] % 360


def _degrees_from_radian_read(read: ReSpeakerParameterRead | None) -> tuple[float | None, ...] | None:
    values = _float_tuple(read, expected_count=None)
    if values is None:
        return None
    return tuple(_degrees_or_none(value) for value in values)


def _degrees_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    normalized = float(value)
    if not math.isfinite(normalized):
        return None
    return math.degrees(normalized) % 360.0


def _mute_active(read: ReSpeakerParameterRead | None) -> bool | None:
    """Return whether the XVF3800 mute control pin is currently asserted.

    Seeed documents ``GPO_READ_VALUES`` in the order ``X0D11, X0D30, X0D31,
    X0D33, X0D39`` and states that pin ``X0D30`` drives the microphone mute
    circuit plus the red mute LED. A high level therefore means the microphones
    are muted.
    """

    values = _logic_level_tuple(read, expected_count=5)  # AUDIT-FIX(#1): Only trusted digital logic levels may drive the mute status.
    if values is None:
        return None
    return bool(values[1])


def _float_tuple(
    read: ReSpeakerParameterRead | None,
    *,
    expected_count: int | None,
) -> tuple[float | None, ...] | None:
    decoded_value = _decoded_tuple(read)  # AUDIT-FIX(#6): Guard against malformed read objects before dereferencing attributes.
    if decoded_value is None:
        return None
    if expected_count is not None and len(decoded_value) != expected_count:
        return None

    values: list[float | None] = []
    for value in decoded_value:
        normalized = _coerce_float(value)
        values.append(normalized if normalized is not None and math.isfinite(normalized) else None)
    return tuple(values)


def _int_tuple(
    read: ReSpeakerParameterRead | None,
    *,
    expected_count: int,
) -> tuple[int, ...] | None:
    decoded_value = _decoded_tuple(read)  # AUDIT-FIX(#6): Guard against malformed read objects before dereferencing attributes.
    if decoded_value is None:
        return None
    if len(decoded_value) != expected_count:
        return None

    normalized: list[int] = []
    for value in decoded_value:
        normalized_value = _coerce_int_exact(value)  # AUDIT-FIX(#7): Reject lossy or ambiguous integer coercions.
        if normalized_value is None:
            return None
        normalized.append(normalized_value)
    return tuple(normalized)


def _logic_level_tuple(
    read: ReSpeakerParameterRead | None,
    *,
    expected_count: int,
) -> tuple[int, ...] | None:
    # AUDIT-FIX(#1): Share strict 0/1 validation across mute and GPO fields.
    values = _int_tuple(read, expected_count=expected_count)
    if values is None:
        return None
    if any(value not in (0, 1) for value in values):
        return None
    return values


def _binary_flag(value: int) -> bool | None:
    # AUDIT-FIX(#1): Binary device flags must be exactly 0 or 1.
    if value not in (0, 1):
        return None
    return bool(value)


def _decoded_tuple(read: ReSpeakerParameterRead | None) -> tuple[object, ...] | None:
    # AUDIT-FIX(#6): Centralize defensive access to external read objects.
    if read is None or not _is_parameter_read_like(read):
        return None
    if not bool(getattr(read, "ok", False)):
        return None

    decoded_value = getattr(read, "decoded_value", None)
    if not isinstance(decoded_value, tuple):
        return None
    return decoded_value


def _captured_at_or_none(read: ReSpeakerParameterRead | None) -> float | None:
    # AUDIT-FIX(#6): Centralize timestamp sanitation for possibly malformed reads.
    if read is None or not _is_parameter_read_like(read):
        return None
    captured_at = _coerce_float(getattr(read, "captured_at", None))
    if captured_at is None or not math.isfinite(captured_at):
        return None
    return captured_at


def _is_parameter_read_like(value: object) -> bool:
    # AUDIT-FIX(#6): Use duck typing so malformed/mock objects are rejected safely.
    return (
        hasattr(value, "captured_at")
        and hasattr(value, "decoded_value")
        and hasattr(value, "ok")
    )


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):  # AUDIT-FIX(#6): Treat oversized numeric payloads as invalid instead of crashing.
        return None


def _coerce_int_exact(value: object) -> int | None:
    # AUDIT-FIX(#7): Allow only exact integers; reject bools and lossy float coercions.
    if isinstance(value, bool):
        return None

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped, 10)
        except ValueError:
            return None

    try:
        return operator.index(value)
    except TypeError:
        pass

    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _close_transport_if_owned(transport: ReSpeakerLibusbTransport | None) -> None:
    # AUDIT-FIX(#3): Best-effort cleanup for transports allocated inside this module.
    if transport is None:
        return

    close_method = getattr(transport, "close", None)
    if not callable(close_method):
        return

    try:
        close_method()
    except Exception:
        return