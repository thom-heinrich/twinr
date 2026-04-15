"""Shared internal types and pure helpers for the orchestrator voice session."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Protocol


class _TranscriptBackend(Protocol):
    """Describe the minimal transcription surface the session needs."""

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        """Transcribe one bounded audio payload into plain text."""


@dataclass(slots=True)
class _RecentFrame:
    """Store one recent PCM fragment plus its bounded level metadata."""

    pcm_bytes: bytes
    rms: int
    duration_ms: int
    speech_probability: float | None = None


@dataclass(slots=True)
class _PendingTranscriptUtterance:
    """Track one same-stream utterance for transcript-first routing."""

    origin_state: str
    wait_id: str | None = None
    item_id: str | None = None
    stage1_attempted: bool = False
    stage1_seeded: bool = False
    frames: deque[_RecentFrame] = field(default_factory=deque)
    captured_ms: int = 0
    max_capture_ms: int = 0
    active_ms: int = 0
    trailing_silence_ms: int = 0
    speech_active: bool = False


def _pcm16_rms(samples: bytes) -> int:
    """Return the root-mean-square amplitude for one PCM16 fragment."""

    if not samples:
        return 0
    usable = len(samples) - (len(samples) % 2)
    if usable <= 0:
        return 0
    import math as _math
    import sys as _sys
    from array import array

    pcm_samples = array("h")
    pcm_samples.frombytes(samples[:usable])
    if _sys.byteorder != "little":
        pcm_samples.byteswap()
    mean_square = sum(sample * sample for sample in pcm_samples) / len(pcm_samples)
    return int(_math.sqrt(mean_square))


def _normalize_text_length(text: str) -> int:
    """Return a simple transcript-length signal without regex product logic."""

    return len("".join(str(text or "").split()))


__all__ = [
    "_PendingTranscriptUtterance",
    "_RecentFrame",
    "_TranscriptBackend",
    "_normalize_text_length",
    "_pcm16_rms",
]
