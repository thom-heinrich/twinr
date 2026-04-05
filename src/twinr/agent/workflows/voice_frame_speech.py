"""Annotate streamed edge voice frames with bounded speech-likelihood evidence.

The host-side transcript-first scanner already understands optional per-frame
speech probabilities, but the Pi transport path previously never populated
them. This helper keeps the classifier wiring and normalization separate from
the edge websocket transport loop.

Important lifetime contract:
- live room audio on the Pi is an effectively unbounded stream
- the host scanner already performs temporal aggregation over successive frames
- keeping recurrent VAD state alive across the full room stream causes drift on
  stationary non-speech noise

Because of that, this helper always classifies each transported frame as one
bounded window and does not preserve classifier stream state across frames.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Protocol, SupportsFloat, SupportsIndex, cast

from twinr.hardware.respeaker.pcm_content_classifier import (
    classify_pcm_speech_likeness,
)


_LOGGER = logging.getLogger(__name__)


class _SpeechClassifier(Protocol):
    """Describe the narrow PCM speech-classifier surface this helper needs."""

    def __call__(
        self,
        pcm_bytes: bytes | None,
        *,
        sample_rate: int | None,
        channels: int | None,
        stream_id: object | None = None,
        end_of_stream: bool = False,
    ) -> object:
        """Return one classifier result object."""


@dataclass(frozen=True, slots=True)
class EdgeVoiceFrameSpeechEvidence:
    """Carry the bounded speech evidence attached to one streamed frame."""

    speech_probability: float | None = None


class EdgeVoiceFrameSpeechAnnotator:
    """Classify streamed PCM frames without growing the transport module."""

    def __init__(
        self,
        *,
        sample_rate: int,
        channels: int,
        classifier: _SpeechClassifier = classify_pcm_speech_likeness,
    ) -> None:
        self._sample_rate = max(1, int(sample_rate))
        self._channels = max(1, int(channels))
        self._classifier = classifier
        self._classifier_failed = False

    def classify_frame(
        self,
        pcm_bytes: bytes,
        *,
        stream_ended: bool = False,
    ) -> EdgeVoiceFrameSpeechEvidence:
        """Return bounded speech evidence for one streamed PCM frame."""

        if not pcm_bytes and not stream_ended:
            return EdgeVoiceFrameSpeechEvidence()

        try:
            evidence = self._classifier(
                pcm_bytes,
                sample_rate=self._sample_rate,
                channels=self._channels,
                stream_id=None,
                end_of_stream=False,
            )
        except Exception:
            if not self._classifier_failed:
                _LOGGER.warning(
                    "Voice frame speech classifier failed; omitting speech_probability side-channel",
                    exc_info=True,
                )
                self._classifier_failed = True
            return EdgeVoiceFrameSpeechEvidence()

        self._classifier_failed = False
        return EdgeVoiceFrameSpeechEvidence(
            speech_probability=_normalize_optional_probability(
                getattr(evidence, "speech_probability", None)
            )
        )

    def reset(self) -> None:
        """Keep the public surface stable for callers that already call reset()."""

        return


def _normalize_optional_probability(value: object | None) -> float | None:
    """Clamp one optional probability into the transport-safe ``[0.0, 1.0]`` range."""

    if value is None:
        return None
    try:
        probability = float(
            cast(SupportsFloat | SupportsIndex | str | bytes | bytearray, value)
        )
    except (TypeError, ValueError):
        return None
    if probability < 0.0:
        return 0.0
    if probability > 1.0:
        return 1.0
    return probability
