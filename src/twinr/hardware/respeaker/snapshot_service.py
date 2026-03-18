"""Capture typed XVF3800 primitive snapshots for Twinr."""

from __future__ import annotations

import math
import time

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


def capture_respeaker_primitive_snapshot(
    *,
    transport: ReSpeakerLibusbTransport | None = None,
    probe: ReSpeakerProbeResult | None = None,
) -> ReSpeakerPrimitiveSnapshot:
    """Capture one bounded XVF3800 primitive snapshot.

    The snapshot remains conservative. It surfaces only primitives that are
    directly readable from the XVF3800 today and leaves unsupported or
    unproven interpretations, such as mute-state mapping, as explicit
    ``None`` instead of guessed values.
    """

    snapshot_time = time.time()
    resolved_probe = probe or probe_respeaker_xvf3800()
    resolved_transport = transport or ReSpeakerLibusbTransport()
    availability, raw_reads = read_default_respeaker_parameters(resolved_transport, probe=resolved_probe)

    direction_time = _latest_read_time(raw_reads, fallback=snapshot_time)
    direction = ReSpeakerDirectionSnapshot(
        captured_at=direction_time,
        speech_detected=_speech_detected(raw_reads.get("DOA_VALUE")),
        room_quiet=_room_quiet(raw_reads.get("DOA_VALUE")),
        doa_degrees=_doa_degrees(raw_reads.get("DOA_VALUE")),
        beam_azimuth_degrees=_degrees_from_radian_read(raw_reads.get("AEC_AZIMUTH_VALUES")),
        beam_speech_energies=_float_tuple(raw_reads.get("AEC_SPENERGY_VALUES"), expected_count=4),
        selected_azimuth_degrees=_degrees_from_radian_read(raw_reads.get("AUDIO_MGR_SELECTED_AZIMUTHS")),
    )
    mute = ReSpeakerMuteSnapshot(
        captured_at=direction_time,
        mute_active=None,
        gpo_logic_levels=_int_tuple(raw_reads.get("GPO_READ_VALUES"), expected_count=5),
    )

    return ReSpeakerPrimitiveSnapshot(
        captured_at=snapshot_time,
        probe=resolved_probe,
        transport=availability,
        firmware_version=_firmware_version(raw_reads.get("VERSION")),
        direction=direction,
        mute=mute,
        raw_reads=raw_reads,
    )


def _latest_read_time(raw_reads: dict[str, ReSpeakerParameterRead], *, fallback: float) -> float:
    captured_times = [read.captured_at for read in raw_reads.values()]
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
    return bool(values[1])


def _room_quiet(read: ReSpeakerParameterRead | None) -> bool | None:
    speech_detected = _speech_detected(read)
    if speech_detected is None:
        return None
    return not speech_detected


def _doa_degrees(read: ReSpeakerParameterRead | None) -> int | None:
    values = _int_tuple(read, expected_count=2)
    if values is None:
        return None
    return int(values[0]) % 360


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


def _float_tuple(
    read: ReSpeakerParameterRead | None,
    *,
    expected_count: int | None,
) -> tuple[float | None, ...] | None:
    if read is None or not read.ok or not isinstance(read.decoded_value, tuple):
        return None
    if expected_count is not None and len(read.decoded_value) != expected_count:
        return None
    values: list[float | None] = []
    for value in read.decoded_value:
        normalized = _coerce_float(value)
        values.append(normalized if normalized is not None and math.isfinite(normalized) else None)
    return tuple(values)


def _int_tuple(
    read: ReSpeakerParameterRead | None,
    *,
    expected_count: int,
) -> tuple[int, ...] | None:
    if read is None or not read.ok or not isinstance(read.decoded_value, tuple):
        return None
    if len(read.decoded_value) != expected_count:
        return None
    normalized: list[int] = []
    for value in read.decoded_value:
        try:
            normalized.append(int(value))
        except (TypeError, ValueError):
            return None
    return tuple(normalized)


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
