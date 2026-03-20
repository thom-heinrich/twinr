"""Decide whether wakeword detections should be accepted or verified.

This module normalizes backend selection, can gate detector hits through a
local clip-level verifier before any optional STT recheck, and returns
structured decisions that the proactive runtime can log and act on
consistently.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from twinr.hardware.audio import AmbientAudioCaptureWindow, pcm16_to_wav_bytes

from .matching import WakewordMatch, match_wakeword_transcript, wakeword_primary_prompt


class WakewordTranscriber(Protocol):
    """Describe the transcription contract required by wakeword verification."""

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        """Transcribe the supplied audio bytes into text."""
        ...


class WakewordCaptureVerifier(Protocol):
    """Describe one verifier that inspects a localized wakeword capture."""

    def verify(self, capture: AmbientAudioCaptureWindow, *, detector_match: WakewordMatch) -> "WakewordVerification":
        """Verify the supplied detector hit against the captured audio window."""
        ...


def normalize_wakeword_backend(value: object | None, *, default: str) -> str:
    """Normalize one wakeword backend name with a safe default."""

    normalized = str(value or default).strip().lower()
    if normalized not in {"openwakeword", "stt", "disabled"}:
        return default
    return normalized


def normalize_wakeword_verifier_mode(value: object | None, *, default: str = "ambiguity_only") -> str:
    """Normalize one verifier mode with a safe default."""

    normalized = str(value or default).strip().lower()
    if normalized not in {"disabled", "ambiguity_only", "always"}:
        return default
    return normalized


@dataclass(frozen=True, slots=True)
class WakewordVerification:
    """Describe the result of STT wakeword verification."""

    status: str
    transcript: str = ""
    normalized_transcript: str = ""
    matched_phrase: str | None = None
    remaining_text: str = ""
    backend: str = "stt"
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class WakewordDecision:
    """Describe the final wakeword decision returned to runtime callers."""

    detected: bool
    outcome: str
    match: WakewordMatch
    source: str
    backend_used: str
    primary_backend: str
    fallback_backend: str | None
    verifier_mode: str
    verifier_used: bool
    verifier_status: str
    verifier_reason: str | None = None
    local_verifier_used: bool = False
    local_verifier_status: str = "not_needed"
    local_verifier_reason: str | None = None
    capture_path: str | None = None


class SttWakewordVerifier:
    """Recheck one wakeword capture with STT transcript matching."""

    def __init__(
        self,
        *,
        backend: WakewordTranscriber,
        phrases: tuple[str, ...],
        language: str | None = None,
        min_prefix_ratio: float = 0.9,
    ) -> None:
        self.backend = backend
        self.phrases = tuple(str(item).strip() for item in phrases if str(item).strip())
        self.language = (language or "").strip() or None
        self.min_prefix_ratio = float(min_prefix_ratio)

    def verify(self, capture: AmbientAudioCaptureWindow, *, detector_match: WakewordMatch) -> WakewordVerification:
        """Verify one detector hit against the configured wakeword phrases."""

        try:
            audio_bytes = pcm16_to_wav_bytes(
                capture.pcm_bytes,
                sample_rate=capture.sample_rate,
                channels=capture.channels,
            )
        except Exception as exc:
            return WakewordVerification(status="error", reason=f"wav:{exc.__class__.__name__}")
        prompt = wakeword_primary_prompt(self.phrases)
        try:
            transcript = str(
                self.backend.transcribe(
                    audio_bytes,
                    filename="wakeword-verifier.wav",
                    content_type="audio/wav",
                    language=self.language,
                    prompt=prompt,
                )
                or ""
            )
        except Exception as exc:
            return WakewordVerification(status="error", reason=f"stt:{exc.__class__.__name__}")
        matched = match_wakeword_transcript(
            transcript,
            phrases=self.phrases,
            min_prefix_ratio=self.min_prefix_ratio,
            backend="stt",
            detector_label=detector_match.detector_label,
            score=detector_match.score,
        )
        if matched.detected:
            return WakewordVerification(
                status="accepted",
                transcript=matched.transcript,
                normalized_transcript=matched.normalized_transcript,
                matched_phrase=matched.matched_phrase,
                remaining_text=matched.remaining_text,
            )
        return WakewordVerification(
            status="rejected",
            transcript=matched.transcript,
            normalized_transcript=matched.normalized_transcript,
        )


class WakewordDecisionPolicy:
    """Apply fallback and verifier rules to one wakeword match."""

    def __init__(
        self,
        *,
        primary_backend: str,
        fallback_backend: str | None = None,
        verifier_mode: str = "ambiguity_only",
        verifier_margin: float = 0.08,
        primary_threshold: float | None = None,
        verifier: WakewordCaptureVerifier | None = None,
        local_verifier: WakewordCaptureVerifier | None = None,
    ) -> None:
        self.primary_backend = normalize_wakeword_backend(primary_backend, default="openwakeword")
        normalized_fallback = normalize_wakeword_backend(fallback_backend, default="stt") if fallback_backend else "disabled"
        self.fallback_backend = None if normalized_fallback == "disabled" else normalized_fallback
        self.verifier_mode = normalize_wakeword_verifier_mode(verifier_mode)
        self.verifier_margin = max(0.0, min(float(verifier_margin), 1.0))
        self.primary_threshold = (
            None
            if primary_threshold is None
            else max(0.0, min(float(primary_threshold), 1.0))
        )
        self.verifier = verifier
        self.local_verifier = local_verifier

    def decide(
        self,
        *,
        match: WakewordMatch,
        capture: AmbientAudioCaptureWindow | None,
        source: str,
    ) -> WakewordDecision:
        """Convert one wakeword match into the final runtime decision."""

        backend_used = normalize_wakeword_backend(match.backend, default=self.primary_backend)
        if not match.detected:
            return WakewordDecision(
                detected=False,
                outcome="rejected",
                match=match,
                source=source,
                backend_used=backend_used,
                primary_backend=self.primary_backend,
                fallback_backend=self.fallback_backend,
                verifier_mode=self.verifier_mode,
                verifier_used=False,
                verifier_status="not_needed",
                local_verifier_used=False,
                local_verifier_status="not_needed",
            )

        if backend_used != self.primary_backend:
            fallback_outcome = "fallback_detected" if backend_used == self.fallback_backend else "detected"
            return WakewordDecision(
                detected=True,
                outcome=fallback_outcome,
                match=match,
                source=source,
                backend_used=backend_used,
                primary_backend=self.primary_backend,
                fallback_backend=self.fallback_backend,
                verifier_mode=self.verifier_mode,
                verifier_used=False,
                verifier_status="not_needed",
                local_verifier_used=False,
                local_verifier_status="not_needed",
            )

        local_verifier_used = False
        local_verifier_status = "not_needed"
        local_verifier_reason: str | None = None
        if self.local_verifier is not None:
            if capture is None:
                local_verification = WakewordVerification(
                    status="error",
                    backend="local_sequence",
                    reason="missing_capture",
                )
            else:
                local_verification = self.local_verifier.verify(capture, detector_match=match)
            if local_verification.status == "accepted":
                local_verifier_used = True
                local_verifier_status = local_verification.status
                local_verifier_reason = local_verification.reason
            elif local_verification.status == "skipped":
                local_verifier_status = "not_needed"
            else:
                return WakewordDecision(
                    detected=False,
                    outcome="rejected_by_local_verifier",
                    match=match,
                    source=source,
                    backend_used=backend_used,
                    primary_backend=self.primary_backend,
                    fallback_backend=self.fallback_backend,
                    verifier_mode=self.verifier_mode,
                    verifier_used=False,
                    verifier_status="not_needed",
                    local_verifier_used=True,
                    local_verifier_status=local_verification.status,
                    local_verifier_reason=local_verification.reason,
                )

        if not self._should_verify(match):
            return WakewordDecision(
                detected=True,
                outcome="verified" if self.verifier_mode == "always" else "detected",
                match=match,
                source=source,
                backend_used=backend_used,
                primary_backend=self.primary_backend,
                fallback_backend=self.fallback_backend,
                verifier_mode=self.verifier_mode,
                verifier_used=False,
                verifier_status="not_needed",
                local_verifier_used=local_verifier_used,
                local_verifier_status=local_verifier_status,
                local_verifier_reason=local_verifier_reason,
            )

        if capture is None or self.verifier is None:
            return WakewordDecision(
                detected=True,
                outcome="detected_unverified",
                match=match,
                source=source,
                backend_used=backend_used,
                primary_backend=self.primary_backend,
                fallback_backend=self.fallback_backend,
                verifier_mode=self.verifier_mode,
                verifier_used=False,
                verifier_status="unavailable",
                verifier_reason="missing_capture_or_verifier",
                local_verifier_used=local_verifier_used,
                local_verifier_status=local_verifier_status,
                local_verifier_reason=local_verifier_reason,
            )

        verification = self.verifier.verify(capture, detector_match=match)
        if verification.status == "accepted":
            verified_match = replace(
                match,
                transcript=verification.transcript,
                normalized_transcript=verification.normalized_transcript,
                matched_phrase=verification.matched_phrase or match.matched_phrase,
                remaining_text=verification.remaining_text,
            )
            return WakewordDecision(
                detected=True,
                outcome="verified",
                match=verified_match,
                source=source,
                backend_used=backend_used,
                primary_backend=self.primary_backend,
                fallback_backend=self.fallback_backend,
                verifier_mode=self.verifier_mode,
                verifier_used=True,
                verifier_status=verification.status,
                local_verifier_used=local_verifier_used,
                local_verifier_status=local_verifier_status,
                local_verifier_reason=local_verifier_reason,
            )

        if verification.status == "rejected":
            rejected_match = replace(
                match,
                transcript=verification.transcript,
                normalized_transcript=verification.normalized_transcript,
            )
            return WakewordDecision(
                detected=False,
                outcome="rejected_by_verifier",
                match=rejected_match,
                source=source,
                backend_used=backend_used,
                primary_backend=self.primary_backend,
                fallback_backend=self.fallback_backend,
                verifier_mode=self.verifier_mode,
                verifier_used=True,
                verifier_status=verification.status,
                local_verifier_used=local_verifier_used,
                local_verifier_status=local_verifier_status,
                local_verifier_reason=local_verifier_reason,
            )

        return WakewordDecision(
            detected=True,
            outcome="detected_unverified",
            match=match,
            source=source,
            backend_used=backend_used,
            primary_backend=self.primary_backend,
            fallback_backend=self.fallback_backend,
            verifier_mode=self.verifier_mode,
            verifier_used=True,
            verifier_status=verification.status,
            verifier_reason=verification.reason,
            local_verifier_used=local_verifier_used,
            local_verifier_status=local_verifier_status,
            local_verifier_reason=local_verifier_reason,
        )

    def _should_verify(self, match: WakewordMatch) -> bool:
        if self.verifier_mode == "disabled":
            return False
        if self.verifier_mode == "always":
            return True
        if match.matched_phrase is None:
            return True
        if self.primary_threshold is None or match.score is None:
            return False
        return float(match.score) < (self.primary_threshold + self.verifier_margin)


__all__ = [
    "SttWakewordVerifier",
    "WakewordDecision",
    "WakewordDecisionPolicy",
    "WakewordVerification",
    "normalize_wakeword_backend",
    "normalize_wakeword_verifier_mode",
]
