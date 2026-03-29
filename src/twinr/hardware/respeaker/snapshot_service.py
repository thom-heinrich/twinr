# CHANGELOG: 2026-03-28
# BUG-1: Stop modulo-wrapping invalid DOA values; XVF3800 DOA_VALUE is a documented 0..359 angle plus VAD flag, so out-of-range angles now fail closed.
# BUG-2: Stop advertising "room_quiet" as the inverse of VAD; the hardware exposes speech detection, not ambient quietness, so this field now returns False only when speech is present and None otherwise.
# BUG-3: Guard AUDIO_MGR_SELECTED_AZIMUTHS against legacy <=2.0.0 firmware semantics; return None instead of mislabeling old-firmware data.
# SEC-1: Fail closed on malformed firmware-controlled payload lengths/ranges so rogue or faulty USB responses cannot be normalized into plausible-looking sensor outputs.
# IMP-1: Add cooperative inter-process capture locking for Linux/Pi deployments in addition to the in-process lock.
# IMP-2: Return named tuple subclasses for ordered beam/GPO fields to keep tuple drop-in compatibility while making each slot self-describing.
# IMP-3: Add high-resolution slow-capture observability using perf_counter_ns.

"""Capture typed XVF3800 primitive snapshots for Twinr."""

from __future__ import annotations

import contextlib
import logging
import math
import operator
import os
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping
from typing import NamedTuple, cast

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

_LOGGER = logging.getLogger(__name__)

_CAPTURE_LOCK = threading.RLock()
_MAX_CAPTURED_AT_FUTURE_SKEW_SECONDS = 5.0
_SLOW_CAPTURE_WARN_SECONDS = 0.250

_INTERPROCESS_LOCK_PATH_ENV = "TWINR_RESPEAKER_CAPTURE_LOCKFILE"
_INTERPROCESS_LOCK_FALLBACK_PATH = os.path.join(
    tempfile.gettempdir(),
    "twinr-respeaker-xvf3800.capture.lock",
)

_VERSION_COMPONENT_COUNT = 3
_VERSION_COMPONENT_MIN = 0
_VERSION_COMPONENT_MAX = 255

_DOA_WORD_COUNT = 2
_DOA_MIN_DEGREES = 0
_DOA_MAX_DEGREES_EXCLUSIVE = 360

_BEAM_COUNT = 4
_SELECTED_AZIMUTH_COUNT = 2
_GPO_COUNT = 5

_BINARY_FLAG_LOW = 0
_BINARY_FLAG_HIGH = 1

_MAX_VALID_AZIMUTH_RADIANS = math.tau + 1e-6
_LEGACY_SELECTED_AZIMUTHS_MAX_FIRMWARE = (2, 0, 0)


class ReSpeakerBeamAzimuthDegrees(NamedTuple):
    focused_beam_1_degrees: float | None
    focused_beam_2_degrees: float | None
    free_running_beam_degrees: float | None
    auto_selected_beam_degrees: float | None


class ReSpeakerSelectedAzimuthDegrees(NamedTuple):
    processed_speaker_azimuth_degrees: float | None
    auto_selected_beam_degrees: float | None


class ReSpeakerBeamSpeechEnergies(NamedTuple):
    focused_beam_1: float | None
    focused_beam_2: float | None
    free_running_beam: float | None
    auto_selected_beam: float | None


class ReSpeakerGpoLogicLevels(NamedTuple):
    x0d11: int
    x0d30_mute: int
    x0d31_amp_enable_active_low: int
    x0d33_ws2812_power: int
    x0d39: int


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
    owned_transport: ReSpeakerLibusbTransport | None = None
    capture_started_perf_ns = time.perf_counter_ns()

    try:
        with _CAPTURE_LOCK:
            with _interprocess_capture_lock():
                resolved_probe = probe if probe is not None else probe_respeaker_xvf3800()
                if transport is None:
                    owned_transport = ReSpeakerLibusbTransport()
                    resolved_transport = owned_transport
                else:
                    resolved_transport = transport

                try:
                    availability, raw_reads = read_default_respeaker_parameters(
                        resolved_transport,
                        probe=resolved_probe,
                    )
                    raw_reads_snapshot = _normalized_raw_reads(raw_reads)
                    capture_completed_at = time.time()

                    firmware_version = _firmware_version(raw_reads_snapshot.get("VERSION"))
                    direction_time = _latest_read_time(
                        raw_reads_snapshot,
                        fallback=capture_completed_at,
                    )

                    direction = ReSpeakerDirectionSnapshot(
                        captured_at=direction_time,
                        speech_detected=_speech_detected(raw_reads_snapshot.get("DOA_VALUE")),
                        room_quiet=_room_quiet(raw_reads_snapshot.get("DOA_VALUE")),
                        doa_degrees=_doa_degrees(raw_reads_snapshot.get("DOA_VALUE")),
                        beam_azimuth_degrees=_beam_azimuth_degrees(
                            raw_reads_snapshot.get("AEC_AZIMUTH_VALUES")
                        ),
                        beam_speech_energies=_beam_speech_energies(
                            raw_reads_snapshot.get("AEC_SPENERGY_VALUES")
                        ),
                        selected_azimuth_degrees=_selected_azimuth_degrees(
                            raw_reads_snapshot.get("AUDIO_MGR_SELECTED_AZIMUTHS"),
                            firmware_version=firmware_version,
                        ),
                    )
                    gpo_logic_levels = _gpo_logic_levels(raw_reads_snapshot.get("GPO_READ_VALUES"))
                    mute = ReSpeakerMuteSnapshot(
                        captured_at=direction_time,
                        mute_active=_mute_active_from_logic_levels(gpo_logic_levels),
                        gpo_logic_levels=gpo_logic_levels,
                    )

                    return ReSpeakerPrimitiveSnapshot(
                        captured_at=capture_completed_at,
                        probe=resolved_probe,
                        transport=availability,
                        firmware_version=firmware_version,
                        direction=direction,
                        mute=mute,
                        raw_reads=dict(raw_reads_snapshot),
                    )
                finally:
                    _close_transport_if_owned(owned_transport)
    finally:
        duration_seconds = (time.perf_counter_ns() - capture_started_perf_ns) / 1_000_000_000
        if duration_seconds >= _SLOW_CAPTURE_WARN_SECONDS:
            _LOGGER.warning(
                "Slow XVF3800 primitive snapshot capture.",
                extra={"capture_duration_seconds": duration_seconds},
            )


@contextlib.contextmanager
def _interprocess_capture_lock() -> Iterator[None]:
    """Serialize cooperative access across Twinr processes on Linux/Pi hosts."""
    lock_path = _capture_lock_path()

    try:
        import fcntl
    except ImportError:
        yield
        return

    lock_parent = os.path.dirname(lock_path) or "."
    os.makedirs(lock_parent, exist_ok=True)
    try:
        with _open_capture_lock_handle(lock_path) as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                try:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
    except OSError:
        _LOGGER.warning("Failed to acquire XVF3800 inter-process capture lock.", exc_info=True)
        yield


def _open_capture_lock_handle(lock_path: str):
    """Open one cooperative lockfile, reusing existing read-only files when needed.

    Root-run services can legitimately create the shared lock as ``0644`` under
    ``/run/lock``. A later non-root diagnostic process still only needs a file
    descriptor for ``flock(LOCK_EX)``, so fall back to a read-only handle when
    append mode fails with ``PermissionError`` on an existing lockfile.
    """

    try:
        return open(lock_path, "a+b")
    except PermissionError:
        if os.path.exists(lock_path) and os.access(lock_path, os.R_OK):
            return open(lock_path, "rb")
        raise


def _capture_lock_path() -> str:
    configured_path = os.environ.get(_INTERPROCESS_LOCK_PATH_ENV, "").strip()
    if configured_path:
        return configured_path
    default_path = "/run/lock/twinr-respeaker-xvf3800.capture.lock"
    default_parent = os.path.dirname(default_path) or "."
    if os.path.isdir(default_parent) and os.access(default_parent, os.W_OK):
        return default_path
    return _INTERPROCESS_LOCK_FALLBACK_PATH


def _normalized_raw_reads(raw_reads: object) -> dict[str, ReSpeakerParameterRead]:
    if not isinstance(raw_reads, Mapping):
        return {}

    normalized: dict[str, ReSpeakerParameterRead] = {}
    for key, value in raw_reads.items():
        if isinstance(key, str) and _is_parameter_read_like(value):
            normalized[key] = cast(ReSpeakerParameterRead, value)
    return normalized


def _latest_read_time(raw_reads: Mapping[str, ReSpeakerParameterRead], *, fallback: float) -> float:
    captured_times: list[float] = []
    latest_allowed_time = fallback + _MAX_CAPTURED_AT_FUTURE_SKEW_SECONDS
    for read in raw_reads.values():
        captured_at = _captured_at_or_none(read)
        if captured_at is None:
            continue
        if captured_at < 0.0 or captured_at > latest_allowed_time:
            continue
        captured_times.append(captured_at)
    return max(captured_times) if captured_times else fallback


def _firmware_version(read: ReSpeakerParameterRead | None) -> tuple[int, int, int] | None:
    values = _bounded_int_tuple(
        read,
        expected_count=_VERSION_COMPONENT_COUNT,
        min_value=_VERSION_COMPONENT_MIN,
        max_value=_VERSION_COMPONENT_MAX,
    )
    if values is None:
        return None
    return values[0], values[1], values[2]


def _speech_detected(read: ReSpeakerParameterRead | None) -> bool | None:
    values = _uint16_tuple(read, expected_count=_DOA_WORD_COUNT)
    if values is None:
        return None
    return _binary_flag(values[1])


def _room_quiet(read: ReSpeakerParameterRead | None) -> bool | None:
    # BREAKING: XVF3800 exposes speech presence here, not a calibrated room-noise
    # or "quiet room" signal. Returning `not speech_detected` produced false
    # positives for TV/music/appliance noise. We now fail open on silence/no-speech.
    speech_detected = _speech_detected(read)
    if speech_detected is None:
        return None
    if speech_detected:
        return False
    return None


def _doa_degrees(read: ReSpeakerParameterRead | None) -> int | None:
    values = _uint16_tuple(read, expected_count=_DOA_WORD_COUNT)
    if values is None:
        return None

    doa = values[0]
    if not (_DOA_MIN_DEGREES <= doa < _DOA_MAX_DEGREES_EXCLUSIVE):
        return None
    return doa


def _beam_azimuth_degrees(
    read: ReSpeakerParameterRead | None,
) -> ReSpeakerBeamAzimuthDegrees | None:
    values = _degrees_tuple(read, expected_count=_BEAM_COUNT)
    if values is None:
        return None
    return ReSpeakerBeamAzimuthDegrees(*values)


def _selected_azimuth_degrees(
    read: ReSpeakerParameterRead | None,
    *,
    firmware_version: tuple[int, int, int] | None,
) -> ReSpeakerSelectedAzimuthDegrees | None:
    # BREAKING: XMOS documents different semantics for AUDIO_MGR_SELECTED_AZIMUTHS
    # on XVF3800 firmware <= 2.0.0. Returning those values under the newer meaning
    # is worse than omitting them.
    if firmware_version is not None and firmware_version <= _LEGACY_SELECTED_AZIMUTHS_MAX_FIRMWARE:
        return None

    values = _degrees_tuple(read, expected_count=_SELECTED_AZIMUTH_COUNT)
    if values is None:
        return None
    return ReSpeakerSelectedAzimuthDegrees(*values)


def _beam_speech_energies(
    read: ReSpeakerParameterRead | None,
) -> ReSpeakerBeamSpeechEnergies | None:
    values = _non_negative_float_tuple(read, expected_count=_BEAM_COUNT)
    if values is None:
        return None
    return ReSpeakerBeamSpeechEnergies(*values)


def _gpo_logic_levels(read: ReSpeakerParameterRead | None) -> ReSpeakerGpoLogicLevels | None:
    values = _logic_level_tuple(read, expected_count=_GPO_COUNT)
    if values is None:
        return None
    return ReSpeakerGpoLogicLevels(*values)


def _mute_active_from_logic_levels(gpo_logic_levels: ReSpeakerGpoLogicLevels | None) -> bool | None:
    """Return whether the XVF3800 mute control pin is currently asserted.

    Seeed documents ``GPO_READ_VALUES`` in the order ``X0D11, X0D30, X0D31,
    X0D33, X0D39`` and states that pin ``X0D30`` drives the microphone mute
    circuit plus the red mute LED. A high level therefore means the microphones
    are muted.
    """
    if gpo_logic_levels is None:
        return None
    return bool(gpo_logic_levels.x0d30_mute)


def _degrees_tuple(
    read: ReSpeakerParameterRead | None,
    *,
    expected_count: int,
) -> tuple[float | None, ...] | None:
    values = _float_tuple(read, expected_count=expected_count)
    if values is None:
        return None
    return tuple(_degrees_or_none(value) for value in values)


def _degrees_or_none(value: float | None) -> float | None:
    if value is None:
        return None

    normalized = float(value)
    if not math.isfinite(normalized):
        return None
    if normalized < 0.0 or normalized > _MAX_VALID_AZIMUTH_RADIANS:
        return None
    if math.isclose(normalized, math.tau, abs_tol=1e-6):
        return 0.0
    return math.degrees(normalized)


def _float_tuple(
    read: ReSpeakerParameterRead | None,
    *,
    expected_count: int | None,
) -> tuple[float | None, ...] | None:
    decoded_value = _decoded_tuple(read)
    if decoded_value is None:
        return None
    if expected_count is not None and len(decoded_value) != expected_count:
        return None

    values: list[float | None] = []
    for value in decoded_value:
        normalized = _coerce_float(value)
        values.append(normalized if normalized is not None and math.isfinite(normalized) else None)
    return tuple(values)


def _non_negative_float_tuple(
    read: ReSpeakerParameterRead | None,
    *,
    expected_count: int,
) -> tuple[float | None, ...] | None:
    values = _float_tuple(read, expected_count=expected_count)
    if values is None:
        return None
    return tuple(value if value is None or value >= 0.0 else None for value in values)


def _uint16_tuple(
    read: ReSpeakerParameterRead | None,
    *,
    expected_count: int,
) -> tuple[int, ...] | None:
    return _bounded_int_tuple(read, expected_count=expected_count, min_value=0, max_value=0xFFFF)


def _bounded_int_tuple(
    read: ReSpeakerParameterRead | None,
    *,
    expected_count: int,
    min_value: int | None,
    max_value: int | None,
) -> tuple[int, ...] | None:
    values = _int_tuple(read, expected_count=expected_count)
    if values is None:
        return None

    for value in values:
        if min_value is not None and value < min_value:
            return None
        if max_value is not None and value > max_value:
            return None
    return values


def _int_tuple(
    read: ReSpeakerParameterRead | None,
    *,
    expected_count: int,
) -> tuple[int, ...] | None:
    decoded_value = _decoded_tuple(read)
    if decoded_value is None:
        return None
    if len(decoded_value) != expected_count:
        return None

    normalized: list[int] = []
    for value in decoded_value:
        normalized_value = _coerce_int_exact(value)
        if normalized_value is None:
            return None
        normalized.append(normalized_value)
    return tuple(normalized)


def _logic_level_tuple(
    read: ReSpeakerParameterRead | None,
    *,
    expected_count: int,
) -> tuple[int, ...] | None:
    values = _bounded_int_tuple(read, expected_count=expected_count, min_value=0, max_value=1)
    if values is None:
        return None
    return values


def _binary_flag(value: int) -> bool | None:
    if value not in (_BINARY_FLAG_LOW, _BINARY_FLAG_HIGH):
        return None
    return bool(value)


def _decoded_tuple(read: ReSpeakerParameterRead | None) -> tuple[object, ...] | None:
    if read is None or not _is_parameter_read_like(read):
        return None
    if not bool(getattr(read, "ok", False)):
        return None

    decoded_value = getattr(read, "decoded_value", None)
    if not isinstance(decoded_value, tuple):
        return None
    return decoded_value


def _captured_at_or_none(read: ReSpeakerParameterRead | None) -> float | None:
    if read is None or not _is_parameter_read_like(read):
        return None
    captured_at = _coerce_float(getattr(read, "captured_at", None))
    if captured_at is None or not math.isfinite(captured_at):
        return None
    return captured_at


def _is_parameter_read_like(value: object) -> bool:
    return (
        hasattr(value, "captured_at")
        and hasattr(value, "decoded_value")
        and hasattr(value, "ok")
    )


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _coerce_int_exact(value: object) -> int | None:
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
    if transport is None:
        return

    close_method = getattr(transport, "close", None)
    if not callable(close_method):
        return

    try:
        close_method()
    except Exception:
        return
