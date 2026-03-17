"""Own low-evidence transcript recovery and verifier gating for streaming STT."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from hashlib import sha1
from typing import Any

from twinr.agent.base_agent.conversation.turn_controller import StreamingTurnController, _normalize_turn_text
from twinr.agent.base_agent.contracts import StreamingSpeechToTextProvider
from twinr.hardware.audio import SpeechCaptureResult, pcm16_duration_ms, pcm16_to_wav_bytes


@dataclass(frozen=True, slots=True)
class TranscriptVerifierGateDecision:
    """Describe whether the expensive verifier path should run for one turn."""

    should_verify: bool
    reason: str
    effective_audio_ms: int
    transcript_chars: int
    word_count: int
    max_capture_ms: int
    max_words: int
    max_chars: int
    saw_speech_final: bool
    saw_utterance_end: bool
    confidence: float | None


class StreamingTranscriptVerifierRuntime:
    """Recover weak transcripts and gate the verifier path with explicit KPIs."""

    def __init__(self, loop) -> None:
        self._loop = loop

    def best_effort_streaming_transcript_hint(
        self,
        *,
        partial_text: str,
        controller: StreamingTurnController | None,
    ) -> str:
        """Return the most informative transcript hint seen so far."""

        transcript_hint = partial_text.strip()
        if controller is not None:
            controller_hint = controller.latest_transcript()
            if controller_hint:
                transcript_hint = controller_hint.strip()
        return transcript_hint

    def maybe_recover_low_evidence_streaming_transcript(
        self,
        *,
        stt_provider: StreamingSpeechToTextProvider,
        capture_result: SpeechCaptureResult | None,
        transcript: str,
        saw_interim: bool,
        capture_ms: int,
    ) -> str:
        """Retry short or empty streaming transcripts against the full capture."""

        loop = self._loop
        cleaned = str(transcript or "").strip()
        if capture_result is None or not capture_result.pcm_bytes:
            loop._trace_event(
                "streaming_transcript_recovery_selected",
                kind="decision",
                details={
                    "selected_source": "original",
                    "reason": "empty_capture",
                    "original": _text_summary(cleaned),
                },
            )
            return cleaned

        if cleaned:
            word_count = len(cleaned.split())
            min_chars = max(8, int(loop.config.streaming_early_transcript_min_chars))
            looks_complete = len(cleaned) >= min_chars and word_count >= 3
            if looks_complete:
                loop._trace_event(
                    "streaming_transcript_recovery_selected",
                    kind="decision",
                    details={
                        "selected_source": "original",
                        "reason": "streaming_transcript_looks_complete",
                        "original": _text_summary(cleaned),
                    },
                )
                return cleaned
            if saw_interim and capture_ms < 1200 and word_count >= 2:
                loop._trace_event(
                    "streaming_transcript_recovery_selected",
                    kind="decision",
                    details={
                        "selected_source": "original",
                        "reason": "short_interim_turn",
                        "original": _text_summary(cleaned),
                    },
                )
                return cleaned
        try:
            audio_bytes = pcm16_to_wav_bytes(
                capture_result.pcm_bytes,
                sample_rate=loop._recorder_sample_rate(),
                channels=loop.config.audio_channels,
            )
            recovered = stt_provider.transcribe(
                audio_bytes,
                filename="twinr-streaming-recover.wav",
                content_type="audio/wav",
                language=loop.config.deepgram_stt_language,
            ).strip()
        except Exception as exc:
            loop.emit(f"stt_streaming_recover_failed={type(exc).__name__}")
            loop._trace_event(
                "streaming_transcript_recovery_failed",
                kind="exception",
                level="WARN",
                details={
                    "error_type": type(exc).__name__,
                    "original": _text_summary(cleaned),
                    "capture_audio_ms": self.captured_audio_duration_ms(capture_result=capture_result),
                },
            )
            return cleaned

        if not recovered:
            loop._trace_event(
                "streaming_transcript_recovery_selected",
                kind="decision",
                details={
                    "selected_source": "original",
                    "reason": "batch_recovery_empty",
                    "original": _text_summary(cleaned),
                },
            )
            return cleaned
        if not cleaned:
            loop.emit("stt_streaming_recovered_via_batch=true")
            loop._trace_event(
                "streaming_transcript_recovery_selected",
                kind="decision",
                details={
                    "selected_source": "batch_recovered",
                    "reason": "streaming_transcript_empty",
                    "original": _text_summary(cleaned),
                    "recovered": _text_summary(recovered),
                },
            )
            return recovered
        recovered_words = len(recovered.split())
        if len(recovered) <= len(cleaned) and recovered_words <= len(cleaned.split()):
            loop._trace_event(
                "streaming_transcript_recovery_selected",
                kind="decision",
                details={
                    "selected_source": "original",
                    "reason": "batch_not_more_informative",
                    "original": _text_summary(cleaned),
                    "recovered": _text_summary(recovered),
                },
            )
            return cleaned

        loop.emit("stt_streaming_recovered_via_batch=true")
        loop._trace_event(
            "streaming_transcript_recovery_selected",
            kind="decision",
            details={
                "selected_source": "batch_recovered",
                "reason": "batch_more_informative",
                "original": _text_summary(cleaned),
                "recovered": _text_summary(recovered),
            },
        )
        return recovered

    def captured_audio_duration_ms(
        self,
        *,
        capture_result: SpeechCaptureResult | None,
    ) -> int:
        """Measure the captured audio duration from PCM instead of wall clock."""

        loop = self._loop
        if capture_result is None or not capture_result.pcm_bytes:
            return 0
        return pcm16_duration_ms(
            capture_result.pcm_bytes,
            sample_rate=loop._recorder_sample_rate(),
            channels=loop.config.audio_channels,
        )

    def verification_gate(
        self,
        *,
        transcript: str,
        capture_result: SpeechCaptureResult | None,
        capture_ms: int,
        saw_speech_final: bool,
        saw_utterance_end: bool,
        confidence: float | None,
    ) -> TranscriptVerifierGateDecision:
        """Return the explicit KPI gate for the expensive verifier path."""

        loop = self._loop
        verifier = getattr(loop, "transcript_verifier_provider", None)
        cleaned = str(transcript or "").strip()
        word_count = len(cleaned.split()) if cleaned else 0
        max_capture_ms = max(1000, int(loop.config.streaming_transcript_verifier_max_capture_ms))
        max_words = max(1, int(loop.config.streaming_transcript_verifier_max_words))
        max_chars = max(8, int(loop.config.streaming_transcript_verifier_max_chars))
        effective_audio_ms = self.captured_audio_duration_ms(capture_result=capture_result) or max(0, int(capture_ms))

        if verifier is None or not callable(getattr(verifier, "transcribe", None)):
            decision = TranscriptVerifierGateDecision(
                should_verify=False,
                reason="verifier_unavailable",
                effective_audio_ms=effective_audio_ms,
                transcript_chars=len(cleaned),
                word_count=word_count,
                max_capture_ms=max_capture_ms,
                max_words=max_words,
                max_chars=max_chars,
                saw_speech_final=saw_speech_final,
                saw_utterance_end=saw_utterance_end,
                confidence=confidence,
            )
            self._trace_gate(decision)
            return decision
        if capture_result is None or not capture_result.pcm_bytes:
            decision = TranscriptVerifierGateDecision(
                should_verify=False,
                reason="empty_capture",
                effective_audio_ms=effective_audio_ms,
                transcript_chars=len(cleaned),
                word_count=word_count,
                max_capture_ms=max_capture_ms,
                max_words=max_words,
                max_chars=max_chars,
                saw_speech_final=saw_speech_final,
                saw_utterance_end=saw_utterance_end,
                confidence=confidence,
            )
            self._trace_gate(decision)
            return decision
        if effective_audio_ms > max_capture_ms:
            decision = TranscriptVerifierGateDecision(
                should_verify=False,
                reason="capture_too_long",
                effective_audio_ms=effective_audio_ms,
                transcript_chars=len(cleaned),
                word_count=word_count,
                max_capture_ms=max_capture_ms,
                max_words=max_words,
                max_chars=max_chars,
                saw_speech_final=saw_speech_final,
                saw_utterance_end=saw_utterance_end,
                confidence=confidence,
            )
            self._trace_gate(decision)
            return decision
        if not cleaned:
            decision = TranscriptVerifierGateDecision(
                should_verify=True,
                reason="empty_transcript",
                effective_audio_ms=effective_audio_ms,
                transcript_chars=0,
                word_count=0,
                max_capture_ms=max_capture_ms,
                max_words=max_words,
                max_chars=max_chars,
                saw_speech_final=saw_speech_final,
                saw_utterance_end=saw_utterance_end,
                confidence=confidence,
            )
            self._trace_gate(decision)
            return decision
        if word_count > max_words and len(cleaned) > max_chars:
            decision = TranscriptVerifierGateDecision(
                should_verify=False,
                reason="transcript_above_gate",
                effective_audio_ms=effective_audio_ms,
                transcript_chars=len(cleaned),
                word_count=word_count,
                max_capture_ms=max_capture_ms,
                max_words=max_words,
                max_chars=max_chars,
                saw_speech_final=saw_speech_final,
                saw_utterance_end=saw_utterance_end,
                confidence=confidence,
            )
            self._trace_gate(decision)
            return decision
        if saw_utterance_end and not saw_speech_final:
            decision = TranscriptVerifierGateDecision(
                should_verify=True,
                reason="utterance_end_without_speech_final",
                effective_audio_ms=effective_audio_ms,
                transcript_chars=len(cleaned),
                word_count=word_count,
                max_capture_ms=max_capture_ms,
                max_words=max_words,
                max_chars=max_chars,
                saw_speech_final=saw_speech_final,
                saw_utterance_end=saw_utterance_end,
                confidence=confidence,
            )
            self._trace_gate(decision)
            return decision
        if confidence is None:
            decision = TranscriptVerifierGateDecision(
                should_verify=word_count <= max_words,
                reason="missing_confidence_short_turn" if word_count <= max_words else "missing_confidence_gate_closed",
                effective_audio_ms=effective_audio_ms,
                transcript_chars=len(cleaned),
                word_count=word_count,
                max_capture_ms=max_capture_ms,
                max_words=max_words,
                max_chars=max_chars,
                saw_speech_final=saw_speech_final,
                saw_utterance_end=saw_utterance_end,
                confidence=None,
            )
            self._trace_gate(decision)
            return decision
        decision = TranscriptVerifierGateDecision(
            should_verify=confidence < float(loop.config.streaming_transcript_verifier_min_confidence),
            reason="confidence_below_threshold"
            if confidence < float(loop.config.streaming_transcript_verifier_min_confidence)
            else "confidence_gate_closed",
            effective_audio_ms=effective_audio_ms,
            transcript_chars=len(cleaned),
            word_count=word_count,
            max_capture_ms=max_capture_ms,
            max_words=max_words,
            max_chars=max_chars,
            saw_speech_final=saw_speech_final,
            saw_utterance_end=saw_utterance_end,
            confidence=confidence,
        )
        self._trace_gate(decision)
        return decision

    def build_streaming_transcript_verifier_prompt(
        self,
        *,
        transcript_hint: str,
    ) -> str:
        """Build the weak-context prompt for the verifier STT provider."""

        loop = self._loop
        conversation = loop.runtime.supervisor_provider_conversation_context()
        tail_lines: list[str] = []
        for role, content in conversation[-2:]:
            role_text = str(role or "").strip()
            content_text = str(content or "").strip()
            if not role_text or not content_text:
                continue
            tail_lines.append(f"{role_text}: {content_text}")
        context_block = "\n".join(tail_lines).strip()
        prompt_lines = [
            "Die Audiodatei enthält eine kurze deutsche Äußerung an einen Sprachassistenten.",
            "Transkribiere wörtlich auf Deutsch.",
            "Behalte umgangssprachliche Kurzformen wie 'geht's', 'hab's' oder 'wie wär's' korrekt bei.",
            "Rate nicht. Wenn ein Streaming-Hinweis unten steht, nutze ihn nur als schwachen Kontext, nicht als Wahrheit.",
        ]
        if context_block:
            prompt_lines.append("Letzter Gesprächskontext:")
            prompt_lines.append(context_block)
        normalized_hint = str(transcript_hint or "").strip()
        if normalized_hint:
            prompt_lines.append(f"Streaming-Hinweis: {normalized_hint}")
        return "\n".join(prompt_lines)

    def maybe_verify_streaming_transcript(
        self,
        *,
        capture_result: SpeechCaptureResult | None,
        transcript: str,
        capture_ms: int,
        saw_speech_final: bool,
        saw_utterance_end: bool,
        confidence: float | None,
    ) -> str:
        """Verify short or suspicious transcripts with the secondary STT path."""

        loop = self._loop
        gate = self.verification_gate(
            transcript=transcript,
            capture_result=capture_result,
            capture_ms=capture_ms,
            saw_speech_final=saw_speech_final,
            saw_utterance_end=saw_utterance_end,
            confidence=confidence,
        )
        cleaned = str(transcript or "").strip()
        if not gate.should_verify:
            loop._trace_event(
                "streaming_transcript_verifier_selected",
                kind="decision",
                details={
                    "selected_source": "original",
                    "reason": gate.reason,
                    "original": _text_summary(cleaned),
                },
            )
            return cleaned

        verifier = getattr(loop, "transcript_verifier_provider", None)
        if verifier is None or capture_result is None:
            return cleaned

        try:
            audio_bytes = pcm16_to_wav_bytes(
                capture_result.pcm_bytes,
                sample_rate=loop._recorder_sample_rate(),
                channels=loop.config.audio_channels,
            )
            verified = verifier.transcribe(
                audio_bytes,
                filename="twinr-streaming-verify.wav",
                content_type="audio/wav",
                language=loop.config.deepgram_stt_language,
                prompt=self.build_streaming_transcript_verifier_prompt(
                    transcript_hint=transcript,
                ),
            ).strip()
        except Exception as exc:
            loop.emit(f"stt_streaming_verify_failed={type(exc).__name__}")
            loop._trace_event(
                "streaming_transcript_verifier_failed",
                kind="exception",
                level="WARN",
                details={
                    "error_type": type(exc).__name__,
                    "original": _text_summary(cleaned),
                    "gate_reason": gate.reason,
                },
            )
            return cleaned

        if not verified:
            loop._trace_event(
                "streaming_transcript_verifier_selected",
                kind="decision",
                details={
                    "selected_source": "original",
                    "reason": "verifier_empty",
                    "original": _text_summary(cleaned),
                },
            )
            return cleaned
        if not cleaned:
            loop.emit("stt_streaming_verified_via_openai=true")
            loop._trace_event(
                "streaming_transcript_verifier_selected",
                kind="decision",
                details={
                    "selected_source": "verifier",
                    "reason": "original_empty",
                    "original": _text_summary(cleaned),
                    "verified": _text_summary(verified),
                    "similarity": None,
                },
            )
            return verified
        if _normalize_turn_text(verified) == _normalize_turn_text(cleaned):
            loop.emit("stt_streaming_verified_via_openai=true")
            loop._trace_event(
                "streaming_transcript_verifier_selected",
                kind="decision",
                details={
                    "selected_source": "verifier",
                    "reason": "normalized_match",
                    "original": _text_summary(cleaned),
                    "verified": _text_summary(verified),
                    "similarity": 1.0,
                },
            )
            return verified

        similarity = SequenceMatcher(
            None,
            _normalize_turn_text(cleaned),
            _normalize_turn_text(verified),
        ).ratio()
        if similarity < 0.9 or len(verified) > len(cleaned):
            loop.emit("stt_streaming_verifier_disagreement=true")
            loop.emit("stt_streaming_verified_via_openai=true")
            loop._trace_event(
                "streaming_transcript_verifier_selected",
                kind="decision",
                details={
                    "selected_source": "verifier",
                    "reason": "verifier_more_informative",
                    "original": _text_summary(cleaned),
                    "verified": _text_summary(verified),
                    "similarity": round(similarity, 4),
                },
            )
            return verified
        loop._trace_event(
            "streaming_transcript_verifier_selected",
            kind="decision",
            details={
                "selected_source": "original",
                "reason": "original_retained",
                "original": _text_summary(cleaned),
                "verified": _text_summary(verified),
                "similarity": round(similarity, 4),
            },
        )
        return cleaned

    def _trace_gate(self, decision: TranscriptVerifierGateDecision) -> None:
        loop = self._loop
        loop._trace_event(
            "streaming_transcript_verifier_gate",
            kind="decision",
            details={
                "should_verify": decision.should_verify,
                "reason": decision.reason,
                "transcript_chars": decision.transcript_chars,
                "word_count": decision.word_count,
                "saw_speech_final": decision.saw_speech_final,
                "saw_utterance_end": decision.saw_utterance_end,
                "confidence": decision.confidence,
                "max_capture_ms": decision.max_capture_ms,
                "max_words": decision.max_words,
                "max_chars": decision.max_chars,
            },
            kpi={"effective_audio_ms": decision.effective_audio_ms},
        )


def _text_summary(value: Any) -> dict[str, Any]:
    """Describe transcript text safely for forensics without raw content."""

    normalized = str(value or "").strip()
    if not normalized:
        return {"present": False, "chars": 0, "words": 0, "sha12": None}
    return {
        "present": True,
        "chars": len(normalized),
        "words": len(normalized.split()),
        "sha12": sha1(normalized.encode("utf-8")).hexdigest()[:12],
    }
