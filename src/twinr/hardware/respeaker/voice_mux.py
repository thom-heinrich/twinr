"""Resolve and enforce the productive XVF3800 USB voice-output mux contract.

Twinr's live transcript-first wake/listen path needs one explicit mono lane
from the XVF3800 USB stream. XMOS defines which internal output pair carries
the ASR beam, while the Seeed 6-channel USB firmware exposes those pairs on
ALSA in its own interleaved lane order. This helper reads the live mux
contract, enables ASR output when necessary, and exposes the authoritative
USB capture lane for the active ASR beam.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from twinr.hardware.respeaker.models import (
    ReSpeakerParameterRead,
    ReSpeakerParameterSpec,
    ReSpeakerParameterType,
    ReSpeakerProbeResult,
    ReSpeakerTransportAvailability,
)
from twinr.hardware.respeaker.transport import ReSpeakerLibusbTransport


_MUX_PAIR_WIDTH = 2
_MUX_OUTPUT_PAIR_COUNT = 6
_MUX_LEFT_OUTPUT_PAIR_COUNT = 3
_MUX_RIGHT_PRIMARY_PAIR_INDEX = 3
_MUX_CATEGORY_ASR_OUTPUT = 7
_MUX_SOURCE_AUTO_SELECT_BEAM = 3
_REQUIRED_ASR_MUX_PAIR = (
    _MUX_CATEGORY_ASR_OUTPUT,
    _MUX_SOURCE_AUTO_SELECT_BEAM,
)
_AEC_ASROUTONOFF_ENABLED = 1

AEC_ASROUTONOFF_PARAMETER = ReSpeakerParameterSpec(
    name="AEC_ASROUTONOFF",
    resid=33,
    cmdid=35,
    value_count=1,
    access_mode="rw",
    value_type=ReSpeakerParameterType.INT32,
    description="Enable the XVF3800 ASR beam output instead of raw AEC residual lanes.",
)

AUDIO_MGR_OP_R_PARAMETER = ReSpeakerParameterSpec(
    name="AUDIO_MGR_OP_R",
    resid=35,
    cmdid=19,
    value_count=2,
    access_mode="rw",
    value_type=ReSpeakerParameterType.UINT8,
    description="Primary right-side output mux pair for the XVF3800 audio manager.",
)

AUDIO_MGR_OP_ALL_PARAMETER = ReSpeakerParameterSpec(
    name="AUDIO_MGR_OP_ALL",
    resid=35,
    cmdid=23,
    value_count=12,
    access_mode="rw",
    value_type=ReSpeakerParameterType.UINT8,
    description="All six XVF3800 USB output mux pairs in unpacked channel order.",
)


@runtime_checkable
class ReSpeakerVoiceMuxTransport(Protocol):
    """Describe the narrow host-control transport surface this module needs."""

    def capture_reads(
        self,
        specs: Iterable[ReSpeakerParameterSpec],
        *,
        probe: ReSpeakerProbeResult | None = None,
    ) -> tuple[ReSpeakerTransportAvailability, dict[str, ReSpeakerParameterRead]]:
        """Read one bounded batch of host-control parameters."""

    def write_parameter(
        self,
        spec: ReSpeakerParameterSpec,
        values: Sequence[object] | str | bytes | bytearray,
        *,
        probe: ReSpeakerProbeResult | None = None,
    ) -> ReSpeakerTransportAvailability:
        """Write one bounded host-control parameter."""

    def close(self) -> None:
        """Release any cached transport resources."""


@dataclass(frozen=True, slots=True)
class ReSpeakerVoiceMuxState:
    """Store the productive voice-related XVF3800 mux state."""

    asr_output_enabled: bool
    output_pairs: tuple[tuple[int, int], ...]

    @property
    def asr_output_pair_index(self) -> int | None:
        """Return the XMOS output-pair slot that currently carries the ASR beam."""

        if not self.asr_output_enabled:
            return None
        for index, pair in enumerate(self.output_pairs):
            if pair == _REQUIRED_ASR_MUX_PAIR:
                return index
        return None

    @property
    def asr_capture_channel_index(self) -> int | None:
        """Return the zero-based USB/ALSA capture lane for the live ASR beam.

        XMOS exposes `AUDIO_MGR_OP_ALL` as three left output pairs followed by
        three right output pairs. The Seeed 6-channel USB firmware then exposes
        those stages to ALSA as interleaved left/right lanes:
        `[L0, R0, L1, R1, L2, R2]`.
        """

        output_pair_index = self.asr_output_pair_index
        if output_pair_index is None:
            return None
        return resolve_respeaker_usb_capture_channel_index(output_pair_index)

    @property
    def voice_capture_channels(self) -> int:
        """Return the number of unpacked USB output channels currently exposed."""

        return len(self.output_pairs)


def capture_respeaker_voice_mux_state(
    *,
    transport: ReSpeakerVoiceMuxTransport | None = None,
) -> ReSpeakerVoiceMuxState:
    """Read the current XVF3800 voice mux state from host control."""

    owned_transport = transport is None
    resolved_transport = transport or ReSpeakerLibusbTransport()
    try:
        return _capture_mux_state(resolved_transport)
    finally:
        if owned_transport:
            resolved_transport.close()


def ensure_respeaker_voice_mux_contract(
    *,
    transport: ReSpeakerVoiceMuxTransport | None = None,
) -> ReSpeakerVoiceMuxState:
    """Enforce and verify the productive XVF3800 ASR output contract."""

    owned_transport = transport is None
    resolved_transport = transport or ReSpeakerLibusbTransport()
    try:
        state = _capture_mux_state(resolved_transport)
        if not state.asr_output_enabled:
            _require_successful_write(
                resolved_transport.write_parameter(
                    AEC_ASROUTONOFF_PARAMETER,
                    (_AEC_ASROUTONOFF_ENABLED,),
                ),
                parameter_name=AEC_ASROUTONOFF_PARAMETER.name,
            )
        if _resolve_output_pair(state.output_pairs, _MUX_RIGHT_PRIMARY_PAIR_INDEX) != _REQUIRED_ASR_MUX_PAIR:
            _require_successful_write(
                resolved_transport.write_parameter(
                    AUDIO_MGR_OP_R_PARAMETER,
                    _REQUIRED_ASR_MUX_PAIR,
                ),
                parameter_name=AUDIO_MGR_OP_R_PARAMETER.name,
            )
        if (not state.asr_output_enabled) or (
            _resolve_output_pair(state.output_pairs, _MUX_RIGHT_PRIMARY_PAIR_INDEX)
            != _REQUIRED_ASR_MUX_PAIR
        ):
            state = _capture_mux_state(resolved_transport)
        if not state.asr_output_enabled:
            raise RuntimeError("XVF3800 AEC_ASROUTONOFF remains disabled after host-control write.")
        if state.asr_capture_channel_index is None:
            raise RuntimeError(
                "XVF3800 voice mux does not expose the required ASR lane. "
                f"Current mux pairs: {format_respeaker_voice_mux_pairs(state.output_pairs)}"
            )
        return state
    finally:
        if owned_transport:
            resolved_transport.close()


def format_respeaker_voice_mux_pairs(output_pairs: Sequence[tuple[int, int]]) -> str:
    """Return one compact string for logging/debugging current mux pairs."""

    return "".join(f"({category},{source})" for category, source in output_pairs)


def resolve_respeaker_usb_capture_channel_index(output_pair_index: int) -> int:
    """Translate one XMOS output-pair index into the productive USB capture lane."""

    normalized_output_pair_index = int(output_pair_index)
    if normalized_output_pair_index < 0 or normalized_output_pair_index >= _MUX_OUTPUT_PAIR_COUNT:
        raise ValueError(
            "output_pair_index must be within the productive XVF3800 6-pair contract: "
            f"{normalized_output_pair_index}"
        )
    if normalized_output_pair_index < _MUX_LEFT_OUTPUT_PAIR_COUNT:
        return normalized_output_pair_index * 2
    return ((normalized_output_pair_index - _MUX_LEFT_OUTPUT_PAIR_COUNT) * 2) + 1


def _capture_mux_state(transport: ReSpeakerVoiceMuxTransport) -> ReSpeakerVoiceMuxState:
    availability, reads = transport.capture_reads(
        (
            AEC_ASROUTONOFF_PARAMETER,
            AUDIO_MGR_OP_ALL_PARAMETER,
        )
    )
    if not availability.available:
        raise RuntimeError(
            "XVF3800 host-control transport is unavailable for voice mux resolution"
            + _availability_reason_suffix(availability)
        )

    asr_output_enabled = bool(
        _require_decoded_int_tuple(
            _require_successful_read(reads, AEC_ASROUTONOFF_PARAMETER),
            expected_count=1,
        )[0]
    )
    raw_pairs = _require_decoded_int_tuple(
        _require_successful_read(reads, AUDIO_MGR_OP_ALL_PARAMETER),
        expected_count=AUDIO_MGR_OP_ALL_PARAMETER.value_count,
    )
    output_pairs = tuple(
        (raw_pairs[index], raw_pairs[index + 1])
        for index in range(0, len(raw_pairs), _MUX_PAIR_WIDTH)
    )
    if len(output_pairs) != _MUX_OUTPUT_PAIR_COUNT:
        raise RuntimeError(
            "XVF3800 voice mux returned an unexpected output pair count: "
            f"{len(output_pairs)}"
        )
    return ReSpeakerVoiceMuxState(
        asr_output_enabled=asr_output_enabled,
        output_pairs=output_pairs,
    )


def _require_successful_read(
    reads: dict[str, ReSpeakerParameterRead],
    spec: ReSpeakerParameterSpec,
) -> ReSpeakerParameterRead:
    read = reads.get(spec.name)
    if read is None:
        raise RuntimeError(f"XVF3800 host-control did not return {spec.name}.")
    if not read.ok:
        raise RuntimeError(
            f"XVF3800 host-control read failed for {spec.name}: {read.error or 'unknown_error'}"
        )
    return read


def _require_decoded_int_tuple(
    read: ReSpeakerParameterRead,
    *,
    expected_count: int,
) -> tuple[int, ...]:
    decoded_value = read.decoded_value
    if not isinstance(decoded_value, tuple) or len(decoded_value) != expected_count:
        raise RuntimeError(
            f"{read.spec.name} returned an invalid decoded value: {decoded_value!r}"
        )
    resolved: list[int] = []
    for value in decoded_value:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(
                f"{read.spec.name} returned a non-numeric value: {decoded_value!r}"
            )
        resolved.append(int(value))
    return tuple(resolved)


def _require_successful_write(
    availability: ReSpeakerTransportAvailability,
    *,
    parameter_name: str,
) -> None:
    if availability.available:
        return
    raise RuntimeError(
        f"XVF3800 host-control write failed for {parameter_name}"
        + _availability_reason_suffix(availability)
    )


def _availability_reason_suffix(availability: ReSpeakerTransportAvailability) -> str:
    reason = str(availability.reason or "").strip()
    if not reason:
        return "."
    return f": {reason}"


def _resolve_output_pair(
    output_pairs: Sequence[tuple[int, int]],
    index: int,
) -> tuple[int, int] | None:
    if index < 0 or index >= len(output_pairs):
        return None
    return output_pairs[index]


__all__ = [
    "AEC_ASROUTONOFF_PARAMETER",
    "AUDIO_MGR_OP_ALL_PARAMETER",
    "AUDIO_MGR_OP_R_PARAMETER",
    "ReSpeakerVoiceMuxState",
    "capture_respeaker_voice_mux_state",
    "ensure_respeaker_voice_mux_contract",
    "format_respeaker_voice_mux_pairs",
    "resolve_respeaker_usb_capture_channel_index",
]
