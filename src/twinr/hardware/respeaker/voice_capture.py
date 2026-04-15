"""Resolve the productive XVF3800 USB voice-capture contract.

Twinr's only supported wake/listen path streams room audio continuously from
the Pi to the host transcript-first gateway. The productive mono lane must
come from the same live XVF3800 host-control mux contract that Twinr enforces
for voice output, including the single authoritative translation from XMOS
output-pair slots to the Seeed USB firmware's ALSA lane order.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from typing import Callable

from twinr.hardware.respeaker.probe import config_targets_respeaker
from twinr.hardware.respeaker.voice_mux import (
    ReSpeakerVoiceMuxState,
    ensure_respeaker_voice_mux_contract,
)


_PCM16_SAMPLE_WIDTH_BYTES = 2
_DIRECT_CAPTURE_ROUTE = "direct_pcm"
_XVF3800_USB_ASR_CAPTURE_ROUTE = "respeaker_usb_asr_lane"
_XVF3800_USB_CAPTURE_CHANNELS = 6


@dataclass(frozen=True, slots=True)
class ReSpeakerVoiceCaptureContract:
    """Describe how one productive capture path should be read and projected."""

    capture_channels: int
    transport_channels: int
    extract_channel_index: int | None
    route_label: str


def resolve_respeaker_voice_capture_contract(
    *,
    capture_device: str,
    transport_channels: int,
    mux_state: ReSpeakerVoiceMuxState | None = None,
    voice_mux_resolver: Callable[[], ReSpeakerVoiceMuxState] | None = None,
) -> ReSpeakerVoiceCaptureContract:
    """Return the productive capture contract for one configured audio device.

    The productive XVF3800 USB firmware exposes six unpacked output lanes, but
    the live host-control mux contract defines which XMOS output pair carries
    the ASR beam. Twinr therefore derives the extraction index from that mux
    state's USB-lane mapping instead of maintaining a second hardcoded lane map
    here.
    """

    normalized_transport_channels = max(1, int(transport_channels))
    if normalized_transport_channels != 1:
        return ReSpeakerVoiceCaptureContract(
            capture_channels=normalized_transport_channels,
            transport_channels=normalized_transport_channels,
            extract_channel_index=None,
            route_label=_DIRECT_CAPTURE_ROUTE,
        )

    if not config_targets_respeaker(capture_device):
        return ReSpeakerVoiceCaptureContract(
            capture_channels=normalized_transport_channels,
            transport_channels=normalized_transport_channels,
            extract_channel_index=None,
            route_label=_DIRECT_CAPTURE_ROUTE,
    )

    resolved_mux_resolver = voice_mux_resolver or ensure_respeaker_voice_mux_contract
    resolved_mux_state = mux_state or resolved_mux_resolver()
    if resolved_mux_state.voice_capture_channels != _XVF3800_USB_CAPTURE_CHANNELS:
        raise RuntimeError(
            "ReSpeaker USB capture width no longer matches the productive 6-channel firmware contract: "
            f"{resolved_mux_state.voice_capture_channels}"
        )
    if not resolved_mux_state.asr_output_enabled:
        raise RuntimeError("ReSpeaker voice mux did not keep the required ASR output enabled.")
    if resolved_mux_state.asr_capture_channel_index is None:
        raise RuntimeError("ReSpeaker voice mux does not expose a resolved ASR capture lane.")

    return ReSpeakerVoiceCaptureContract(
        capture_channels=_XVF3800_USB_CAPTURE_CHANNELS,
        transport_channels=1,
        extract_channel_index=resolved_mux_state.asr_capture_channel_index,
        route_label=_XVF3800_USB_ASR_CAPTURE_ROUTE,
    )


def project_respeaker_capture_frame(
    pcm_bytes: bytes,
    *,
    contract: ReSpeakerVoiceCaptureContract,
) -> bytes:
    """Project one captured PCM16 frame into the mono transport contract."""

    if not pcm_bytes:
        return b""

    if contract.extract_channel_index is None:
        return _trim_incomplete_bytes(
            pcm_bytes,
            alignment=_PCM16_SAMPLE_WIDTH_BYTES * max(1, int(contract.transport_channels)),
        )

    return extract_pcm16_channel(
        pcm_bytes,
        channels=contract.capture_channels,
        channel_index=contract.extract_channel_index,
    )


def extract_pcm16_channel(
    pcm_bytes: bytes,
    *,
    channels: int,
    channel_index: int,
) -> bytes:
    """Return one channel from interleaved PCM16 frames."""

    normalized_channels = max(1, int(channels))
    normalized_channel_index = int(channel_index)
    if normalized_channel_index < 0 or normalized_channel_index >= normalized_channels:
        raise ValueError(
            f"channel_index must be in [0, {normalized_channels - 1}] for {normalized_channels} channel PCM"
        )

    frame_alignment = _PCM16_SAMPLE_WIDTH_BYTES * normalized_channels
    aligned_pcm = _trim_incomplete_bytes(pcm_bytes, alignment=frame_alignment)
    if not aligned_pcm:
        return b""
    if normalized_channels == 1:
        return aligned_pcm

    interleaved_samples = array("h")
    interleaved_samples.frombytes(aligned_pcm)
    selected_samples = array("h", interleaved_samples[normalized_channel_index::normalized_channels])
    return selected_samples.tobytes()


def _trim_incomplete_bytes(payload: bytes, *, alignment: int) -> bytes:
    if alignment <= 1 or not payload:
        return payload
    usable_length = len(payload) - (len(payload) % alignment)
    if usable_length <= 0:
        return b""
    return payload[:usable_length]


__all__ = [
    "ReSpeakerVoiceCaptureContract",
    "extract_pcm16_channel",
    "project_respeaker_capture_frame",
    "resolve_respeaker_voice_capture_contract",
]
