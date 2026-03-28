"""Own low-evidence transcript recovery and verifier gating for streaming STT."""
# CHANGELOG: 2026-03-28
# BUG-1: Verifier prompt is now language-aware instead of hard-coding German, preventing cross-language transcription bias when the runtime language changes.
# BUG-2: Recovery/verifier arbitration no longer accepts any longer alternative transcript by default; selection is now evidence-aware and conservative against hallucinated expansions.
# BUG-3: Recovery uploads are now bounded by duration, byte size, and timeout to prevent long/noisy captures from stalling turns or triggering oversized fallback requests.
# SEC-1: Transcript forensics fingerprints no longer use unsalted SHA-1; logs now use keyed BLAKE2s fingerprints so short commands are not dictionary-recoverable from log access alone.
# SEC-2: Verifier prompt context is now sanitized and tightly truncated to reduce prompt-injection bias and unnecessary transcript/context exposure.
# IMP-1: The runtime now consumes richer provider responses (text/logprobs/confidence/usage) when available and uses calibration-aware evidence scoring for transcript selection.
# IMP-2: Optional local verifier/recovery providers can now be chained in as low-latency privacy-preserving fallbacks (e.g. sherpa-onnx on Raspberry Pi) without changing callers.

from __future__ import annotations

import concurrent.futures
import math
import os
import re
import secrets
from dataclasses import dataclass
from difflib import SequenceMatcher
from hashlib import blake2s
from typing import Any, Iterable, Mapping, Sequence

from twinr.agent.base_agent.conversation.decision_core import normalize_turn_text
from twinr.agent.base_agent.conversation.turn_controller import StreamingTurnController
from twinr.agent.base_agent.contracts import StreamingSpeechToTextProvider
from twinr.hardware.audio import SpeechCaptureResult, pcm16_duration_ms, pcm16_to_wav_bytes


_TRANSCRIBE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=max(1, int(os.getenv("TWINR_TRANSCRIPT_EXECUTOR_WORKERS", "2"))),
    thread_name_prefix="twinr-stt",
)
_FORENSICS_KEY = (
    os.getenv("TWINR_TRANSCRIPT_FORENSICS_SALT", "").encode("utf-8") or secrets.token_bytes(16)
)
_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_GERMAN_FILLERS = {
    "äh",
    "ähm",
    "hm",
    "hmm",
    "also",
    "bitte",
    "mal",
    "doch",
    "okay",
    "ok",
    "ja",
    "nee",
    "ne",
}
_ENGLISH_FILLERS = {"uh", "um", "hmm", "hm", "okay", "ok", "please", "yeah", "yep", "nope"}


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


@dataclass(frozen=True, slots=True)
class _TranscriptCandidate:
    text: str
    normalized: str
    source: str
    provider_label: str
    confidence: float | None
    avg_logprob: float | None
    usage_seconds: float | None
    words: int
    chars: int


@dataclass(frozen=True, slots=True)
class _TranscriptSelection:
    text: str
    reason: str
    selected_source: str
    similarity: float | None
    token_similarity: float | None
    overlap_ratio: float | None
    containment_ratio: float | None
    growth_ratio: float | None


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

        partial_hint = _clean_text(partial_text)
        controller_hint = ""
        if controller is not None:
            controller_hint = _clean_text(controller.latest_transcript())
        return self._choose_better_hint(partial_hint, controller_hint)

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
        cleaned = _clean_text(transcript)
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

        captured_audio_ms = self.captured_audio_duration_ms(capture_result=capture_result)
        min_chars = max(8, int(_cfg(loop.config, "streaming_early_transcript_min_chars", 8)))
        if cleaned:
            word_count = len(cleaned.split())
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

        recovery_max_capture_ms = max(
            1000,
            int(
                _cfg(
                    loop.config,
                    "streaming_transcript_recovery_max_capture_ms",
                    max(
                        4000,
                        int(_cfg(loop.config, "streaming_transcript_verifier_max_capture_ms", 4000)),
                    ),
                )
            ),
        )
        if captured_audio_ms > recovery_max_capture_ms:
            loop._trace_event(
                "streaming_transcript_recovery_selected",
                kind="decision",
                details={
                    "selected_source": "original",
                    "reason": "capture_too_long_for_recovery",
                    "original": _text_summary(cleaned),
                    "capture_audio_ms": captured_audio_ms,
                    "max_capture_ms": recovery_max_capture_ms,
                },
            )
            return cleaned

        audio_bytes = self._capture_to_wav_bytes(
            capture_result=capture_result,
            max_audio_ms=recovery_max_capture_ms,
            mode="recovery",
        )
        if audio_bytes is None:
            loop._trace_event(
                "streaming_transcript_recovery_selected",
                kind="decision",
                details={
                    "selected_source": "original",
                    "reason": "capture_unusable_for_recovery",
                    "original": _text_summary(cleaned),
                    "capture_audio_ms": captured_audio_ms,
                },
            )
            return cleaned

        providers = self._recovery_providers(primary=stt_provider)
        last_exc: Exception | None = None
        recovered_candidate: _TranscriptCandidate | None = None
        for provider in providers:
            try:
                recovered_candidate = self._transcribe_candidate(
                    provider=provider,
                    audio_bytes=audio_bytes,
                    filename="twinr-streaming-recover.wav",
                    content_type="audio/wav",
                    language=_cfg(loop.config, "deepgram_stt_language", None),
                    prompt=None,
                    mode="recovery",
                )
                if recovered_candidate.text:
                    break
            except Exception as exc:  # pragma: no cover - defensive around provider transport errors
                last_exc = exc
                loop.emit(f"stt_streaming_recover_failed={type(exc).__name__}")
                loop._trace_event(
                    "streaming_transcript_recovery_failed",
                    kind="exception",
                    level="WARN",
                    details={
                        "error_type": type(exc).__name__,
                        "provider": _provider_label(provider),
                        "original": _text_summary(cleaned),
                        "capture_audio_ms": captured_audio_ms,
                    },
                )
        if recovered_candidate is None:
            return cleaned
        if not recovered_candidate.text:
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

        original_candidate = self._make_candidate(
            text=cleaned,
            source="original",
            provider_label="streaming",
            confidence=None,
            avg_logprob=None,
            usage_seconds=None,
        )
        selection = self._select_between_candidates(
            original=original_candidate,
            alternative=recovered_candidate,
            mode="recovery",
            original_confidence=None,
            original_is_weak=(not cleaned) or original_candidate.words <= 1 or original_candidate.chars < min_chars,
            original_language=_cfg(loop.config, "deepgram_stt_language", None),
        )
        if selection.selected_source != "original":
            loop.emit("stt_streaming_recovered_via_batch=true")
        loop._trace_event(
            "streaming_transcript_recovery_selected",
            kind="decision",
            details={
                "selected_source": selection.selected_source,
                "reason": selection.reason,
                "original": _text_summary(cleaned),
                "recovered": _text_summary(recovered_candidate.text),
                "provider": recovered_candidate.provider_label,
                "similarity": selection.similarity,
                "token_similarity": selection.token_similarity,
                "overlap_ratio": selection.overlap_ratio,
                "containment_ratio": selection.containment_ratio,
                "growth_ratio": selection.growth_ratio,
                "last_error_type": type(last_exc).__name__ if last_exc is not None else None,
            },
        )
        return selection.text

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
        cleaned = _clean_text(transcript)
        word_count = len(cleaned.split()) if cleaned else 0
        max_capture_ms = max(1000, int(_cfg(loop.config, "streaming_transcript_verifier_max_capture_ms", 4000)))
        max_words = max(1, int(_cfg(loop.config, "streaming_transcript_verifier_max_words", 4)))
        max_chars = max(8, int(_cfg(loop.config, "streaming_transcript_verifier_max_chars", 24)))
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
            should_verify = word_count <= max_words
            decision = TranscriptVerifierGateDecision(
                should_verify=should_verify,
                reason="missing_confidence_short_turn" if should_verify else "missing_confidence_gate_closed",
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
        min_confidence = float(_cfg(loop.config, "streaming_transcript_verifier_min_confidence", 0.84))
        decision = TranscriptVerifierGateDecision(
            should_verify=confidence < min_confidence,
            reason="confidence_below_threshold" if confidence < min_confidence else "confidence_gate_closed",
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
        language = str(_cfg(loop.config, "deepgram_stt_language", "") or "").strip().lower()
        conversation = loop.runtime.supervisor_provider_conversation_context()
        max_context_chars = max(0, int(_cfg(loop.config, "streaming_transcript_verifier_context_max_chars", 240)))
        max_hint_chars = max(0, int(_cfg(loop.config, "streaming_transcript_verifier_hint_max_chars", 96)))

        tail_lines: list[str] = []
        remaining = max_context_chars
        for role, content in reversed(list(conversation[-2:])):
            if remaining <= 0:
                break
            role_text = _sanitize_context_value(role, limit=24)
            content_text = _sanitize_context_value(content, limit=min(remaining, 160))
            if not role_text or not content_text:
                continue
            entry = f'- {role_text}: "{content_text}"'
            tail_lines.append(entry)
            remaining -= len(entry)
        tail_lines.reverse()
        context_block = "\n".join(tail_lines).strip()
        normalized_hint = _sanitize_context_value(transcript_hint, limit=max_hint_chars)

        prompt_lines = self._verifier_instruction_lines(language=language)
        if context_block:
            prompt_lines.append("")
            prompt_lines.append(self._context_heading(language=language))
            prompt_lines.append(context_block)
        if normalized_hint:
            prompt_lines.append("")
            prompt_lines.append(self._hint_heading(language=language))
            prompt_lines.append(f'"{normalized_hint}"')
        return "\n".join(line for line in prompt_lines if line).strip()

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
        cleaned = _clean_text(transcript)
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
        if verifier is None or capture_result is None or not capture_result.pcm_bytes:
            return cleaned

        audio_bytes = self._capture_to_wav_bytes(
            capture_result=capture_result,
            max_audio_ms=gate.max_capture_ms,
            mode="verify",
        )
        if audio_bytes is None:
            loop._trace_event(
                "streaming_transcript_verifier_selected",
                kind="decision",
                details={
                    "selected_source": "original",
                    "reason": "capture_unusable_for_verifier",
                    "original": _text_summary(cleaned),
                },
            )
            return cleaned

        original_candidate = self._make_candidate(
            text=cleaned,
            source="original",
            provider_label="streaming",
            confidence=confidence,
            avg_logprob=None,
            usage_seconds=None,
        )

        providers = self._verifier_providers(primary=verifier)
        verified_candidate: _TranscriptCandidate | None = None
        last_exc: Exception | None = None
        for provider in providers:
            try:
                verified_candidate = self._transcribe_candidate(
                    provider=provider,
                    audio_bytes=audio_bytes,
                    filename="twinr-streaming-verify.wav",
                    content_type="audio/wav",
                    language=_cfg(loop.config, "deepgram_stt_language", None),
                    prompt=self.build_streaming_transcript_verifier_prompt(transcript_hint=cleaned),
                    mode="verify",
                )
                if verified_candidate.text:
                    break
            except Exception as exc:  # pragma: no cover - defensive around provider transport errors
                last_exc = exc
                loop.emit(f"stt_streaming_verify_failed={type(exc).__name__}")
                loop._trace_event(
                    "streaming_transcript_verifier_failed",
                    kind="exception",
                    level="WARN",
                    details={
                        "error_type": type(exc).__name__,
                        "provider": _provider_label(provider),
                        "original": _text_summary(cleaned),
                        "gate_reason": gate.reason,
                    },
                )
        if verified_candidate is None or not verified_candidate.text:
            loop._trace_event(
                "streaming_transcript_verifier_selected",
                kind="decision",
                details={
                    "selected_source": "original",
                    "reason": "verifier_empty" if verified_candidate is not None else "verifier_failed",
                    "original": _text_summary(cleaned),
                    "gate_reason": gate.reason,
                    "last_error_type": type(last_exc).__name__ if last_exc is not None else None,
                },
            )
            return cleaned

        selection = self._select_between_candidates(
            original=original_candidate,
            alternative=verified_candidate,
            mode="verify",
            original_confidence=None if self._gate_implies_weak_original(gate) else confidence,
            original_is_weak=self._gate_implies_weak_original(gate),
            original_language=_cfg(loop.config, "deepgram_stt_language", None),
        )
        if selection.selected_source != "original":
            self._emit_verifier_success(
                provider_label=verified_candidate.provider_label,
                primary_provider_label=_provider_label(verifier),
            )
            if (selection.similarity is not None and selection.similarity < 0.90) or (
                selection.token_similarity is not None and selection.token_similarity < 0.90
            ):
                loop.emit("stt_streaming_verifier_disagreement=true")

        loop._trace_event(
            "streaming_transcript_verifier_selected",
            kind="decision",
            details={
                "selected_source": selection.selected_source,
                "reason": selection.reason,
                "original": _text_summary(cleaned),
                "verified": _text_summary(verified_candidate.text),
                "provider": verified_candidate.provider_label,
                "gate_reason": gate.reason,
                "similarity": selection.similarity,
                "token_similarity": selection.token_similarity,
                "overlap_ratio": selection.overlap_ratio,
                "containment_ratio": selection.containment_ratio,
                "growth_ratio": selection.growth_ratio,
                "avg_logprob": verified_candidate.avg_logprob,
                "verifier_confidence": verified_candidate.confidence,
                "usage_seconds": verified_candidate.usage_seconds,
            },
        )
        return selection.text

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

    def _transcribe_candidate(
        self,
        *,
        provider: Any,
        audio_bytes: bytes,
        filename: str,
        content_type: str,
        language: str | None,
        prompt: str | None,
        mode: str,
    ) -> _TranscriptCandidate:
        timeout_s = float(_cfg(self._loop.config, "streaming_transcript_provider_timeout_s", 4.5))
        attempts: list[dict[str, Any]] = []
        base_kwargs: dict[str, Any] = {
            "filename": filename,
            "content_type": content_type,
            "language": language,
        }
        if prompt:
            base_kwargs["prompt"] = prompt

        if _provider_may_support_rich_transcription(provider):
            rich_kwargs = dict(base_kwargs)
            rich_kwargs["response_format"] = "json"
            rich_kwargs["include"] = ["logprobs"]
            attempts.append(rich_kwargs)
        attempts.append(base_kwargs)

        last_exc: Exception | None = None
        for kwargs in attempts:
            try:
                raw_result = self._run_blocking_transcribe(
                    provider=provider,
                    audio_bytes=audio_bytes,
                    timeout_s=timeout_s,
                    kwargs=kwargs,
                )
                candidate = self._coerce_provider_result(
                    raw_result=raw_result,
                    source="verifier" if mode == "verify" else "batch_recovered",
                    provider_label=_provider_label(provider),
                )
                if candidate.text or kwargs is attempts[-1]:
                    return candidate
            except TypeError as exc:
                last_exc = exc
                continue
            except Exception:
                raise
        if last_exc is not None:
            raise last_exc
        return self._make_candidate(
            text="",
            source="verifier" if mode == "verify" else "batch_recovered",
            provider_label=_provider_label(provider),
            confidence=None,
            avg_logprob=None,
            usage_seconds=None,
        )

    def _run_blocking_transcribe(
        self,
        *,
        provider: Any,
        audio_bytes: bytes,
        timeout_s: float,
        kwargs: Mapping[str, Any],
    ) -> Any:
        future = _TRANSCRIBE_EXECUTOR.submit(
            provider.transcribe,
            audio_bytes,
            **dict(kwargs),
        )
        try:
            return future.result(timeout=max(0.1, timeout_s))
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise TimeoutError("streaming transcript provider timed out") from exc

    def _capture_to_wav_bytes(
        self,
        *,
        capture_result: SpeechCaptureResult,
        max_audio_ms: int,
        mode: str,
    ) -> bytes | None:
        loop = self._loop
        if not capture_result.pcm_bytes:
            return None

        effective_audio_ms = self.captured_audio_duration_ms(capture_result=capture_result)
        if effective_audio_ms <= 0:
            # Some test doubles and edge captures provide frame-aligned PCM that truncates
            # below 1 ms when converted to an integer duration. Treat any aligned payload as a
            # minimally valid short clip instead of rejecting recovery/verifier retries outright.
            frame_bytes = max(1, 2 * max(1, int(loop.config.audio_channels)))
            aligned_pcm_bytes = len(capture_result.pcm_bytes) - (
                len(capture_result.pcm_bytes) % frame_bytes
            )
            if aligned_pcm_bytes <= 0:
                return None
            effective_audio_ms = 1
        if effective_audio_ms > max_audio_ms:
            return None

        audio_bytes = pcm16_to_wav_bytes(
            capture_result.pcm_bytes,
            sample_rate=loop._recorder_sample_rate(),
            channels=loop.config.audio_channels,
        )
        max_upload_bytes = max(
            64 * 1024,
            int(_cfg(loop.config, "streaming_transcript_max_upload_bytes", 1_500_000)),
        )
        if len(audio_bytes) > max_upload_bytes:
            loop._trace_event(
                f"streaming_transcript_{mode}_audio_rejected",
                kind="decision",
                details={
                    "reason": "wav_too_large",
                    "audio_bytes": len(audio_bytes),
                    "max_upload_bytes": max_upload_bytes,
                    "audio_ms": effective_audio_ms,
                },
            )
            return None
        return audio_bytes

    def _select_between_candidates(
        self,
        *,
        original: _TranscriptCandidate,
        alternative: _TranscriptCandidate,
        mode: str,
        original_confidence: float | None,
        original_is_weak: bool,
        original_language: str | None,
    ) -> _TranscriptSelection:
        reason_prefix = "verifier" if mode == "verify" else mode

        if not alternative.text:
            return _TranscriptSelection(
                text=original.text,
                reason=f"{reason_prefix}_alternative_empty",
                selected_source="original",
                similarity=None,
                token_similarity=None,
                overlap_ratio=None,
                containment_ratio=None,
                growth_ratio=None,
            )
        if not original.text:
            return _TranscriptSelection(
                text=alternative.text,
                reason="original_empty",
                selected_source=alternative.source,
                similarity=None,
                token_similarity=None,
                overlap_ratio=None,
                containment_ratio=None,
                growth_ratio=None,
            )
        if original.normalized == alternative.normalized:
            preferred = alternative if self._candidate_score(alternative) >= self._candidate_score(original, fallback_confidence=original_confidence) else original
            return _TranscriptSelection(
                text=preferred.text,
                reason="normalized_match",
                selected_source=preferred.source if preferred is alternative else "original",
                similarity=1.0,
                token_similarity=1.0,
                overlap_ratio=1.0,
                containment_ratio=1.0,
                growth_ratio=round(_ratio(alternative.chars, original.chars), 4),
            )

        metrics = _compare_texts(original.normalized, alternative.normalized)
        similarity = metrics["char_similarity"]
        token_similarity = metrics["token_similarity"]
        overlap_ratio = metrics["overlap_ratio"]
        containment_ratio = metrics["containment_ratio"]
        reverse_containment = metrics["reverse_containment_ratio"]
        growth_ratio = metrics["growth_ratio"]

        min_similarity = float(_cfg(self._loop.config, "streaming_transcript_selection_min_similarity", 0.74))
        min_overlap = float(_cfg(self._loop.config, "streaming_transcript_selection_min_overlap", 0.60))
        max_growth_ratio = float(_cfg(self._loop.config, "streaming_transcript_selection_max_growth_ratio", 1.80))
        filler_bonus = _filler_expansion_bonus(
            base=original.normalized,
            expanded=alternative.normalized,
            language=str(original_language or ""),
        )
        original_score = self._candidate_score(original, fallback_confidence=original_confidence)
        alternative_score = self._candidate_score(alternative)

        safe_expansion = (
            containment_ratio >= 0.85
            and token_similarity >= 0.72
            and growth_ratio <= max_growth_ratio
        )
        safe_contraction = (
            reverse_containment >= 0.85
            and token_similarity >= 0.72
        )

        if not original_is_weak and similarity < min_similarity and overlap_ratio < min_overlap and containment_ratio < 0.70:
            return _TranscriptSelection(
                text=original.text,
                reason="high_disagreement_original_retained",
                selected_source="original",
                similarity=round(similarity, 4),
                token_similarity=round(token_similarity, 4),
                overlap_ratio=round(overlap_ratio, 4),
                containment_ratio=round(containment_ratio, 4),
                growth_ratio=round(growth_ratio, 4),
            )

        alternative_has_better_evidence = alternative_score + filler_bonus >= original_score + 0.05
        very_short_weak_original = original_is_weak and (original.words <= 2 or original.chars <= 12)
        alternative_is_more_informative = alternative.words >= (original.words + 2) or (
            alternative.chars >= (original.chars + 8)
        )

        if (
            very_short_weak_original
            and containment_ratio >= 0.99
            and alternative_is_more_informative
            and (token_similarity >= 0.40 or overlap_ratio >= 0.40)
        ):
            return _TranscriptSelection(
                text=alternative.text,
                reason=f"{reason_prefix}_more_informative",
                selected_source=alternative.source,
                similarity=round(similarity, 4),
                token_similarity=round(token_similarity, 4),
                overlap_ratio=round(overlap_ratio, 4),
                containment_ratio=round(containment_ratio, 4),
                growth_ratio=round(growth_ratio, 4),
            )

        if safe_expansion and (original_is_weak or filler_bonus > 0.0) and (alternative_has_better_evidence or original_is_weak):
            return _TranscriptSelection(
                text=alternative.text,
                reason="supported_expansion_selected",
                selected_source=alternative.source,
                similarity=round(similarity, 4),
                token_similarity=round(token_similarity, 4),
                overlap_ratio=round(overlap_ratio, 4),
                containment_ratio=round(containment_ratio, 4),
                growth_ratio=round(growth_ratio, 4),
            )

        if safe_contraction and alternative_has_better_evidence:
            return _TranscriptSelection(
                text=alternative.text,
                reason="supported_contraction_selected",
                selected_source=alternative.source,
                similarity=round(similarity, 4),
                token_similarity=round(token_similarity, 4),
                overlap_ratio=round(overlap_ratio, 4),
                containment_ratio=round(containment_ratio, 4),
                growth_ratio=round(growth_ratio, 4),
            )

        supported_weak_match = (
            similarity >= 0.60
            or overlap_ratio >= 0.50
            or (token_similarity >= 0.40 and containment_ratio >= 0.50)
            or (
                original.words == 1
                and containment_ratio >= 0.99
                and alternative_is_more_informative
            )
        )
        alternative_preserves_detail = (
            alternative.words >= original.words and alternative.chars + 4 >= original.chars
        )

        if original_is_weak and supported_weak_match:
            if alternative_score >= original_score or alternative_preserves_detail:
                return _TranscriptSelection(
                    text=alternative.text,
                    reason="weak_original_supported_alternative_selected",
                    selected_source=alternative.source,
                    similarity=round(similarity, 4),
                    token_similarity=round(token_similarity, 4),
                    overlap_ratio=round(overlap_ratio, 4),
                    containment_ratio=round(containment_ratio, 4),
                    growth_ratio=round(growth_ratio, 4),
                )

        return _TranscriptSelection(
            text=original.text,
            reason="original_retained",
            selected_source="original",
            similarity=round(similarity, 4),
            token_similarity=round(token_similarity, 4),
            overlap_ratio=round(overlap_ratio, 4),
            containment_ratio=round(containment_ratio, 4),
            growth_ratio=round(growth_ratio, 4),
        )

    def _candidate_score(
        self,
        candidate: _TranscriptCandidate,
        *,
        fallback_confidence: float | None = None,
    ) -> float:
        score = 0.0
        confidence = candidate.confidence if candidate.confidence is not None else fallback_confidence
        if confidence is not None:
            score += 0.55 * max(0.0, min(1.0, float(confidence)))
        if candidate.avg_logprob is not None:
            score += 0.35 * math.exp(max(-5.0, min(0.0, candidate.avg_logprob)))
        score += min(candidate.words, 6) / 6.0 * 0.08
        score += min(candidate.chars, 48) / 48.0 * 0.02
        return round(score, 4)

    def _coerce_provider_result(
        self,
        *,
        raw_result: Any,
        source: str,
        provider_label: str,
    ) -> _TranscriptCandidate:
        if isinstance(raw_result, str):
            return self._make_candidate(
                text=raw_result,
                source=source,
                provider_label=provider_label,
                confidence=None,
                avg_logprob=None,
                usage_seconds=None,
            )

        text = _pick_first_non_empty(
            _lookup(raw_result, "text"),
            _lookup(raw_result, "transcript"),
            _lookup(raw_result, "output_text"),
        )
        confidence = _coerce_float(
            _pick_first_non_empty(
                _lookup(raw_result, "confidence"),
                _lookup(_lookup(raw_result, "metadata"), "confidence"),
            )
        )
        usage = _lookup(raw_result, "usage")
        usage_seconds = _extract_usage_seconds(usage)
        avg_logprob = _extract_avg_logprob(_lookup(raw_result, "logprobs"))
        return self._make_candidate(
            text=text,
            source=source,
            provider_label=provider_label,
            confidence=confidence,
            avg_logprob=avg_logprob,
            usage_seconds=usage_seconds,
        )

    def _make_candidate(
        self,
        *,
        text: Any,
        source: str,
        provider_label: str,
        confidence: float | None,
        avg_logprob: float | None,
        usage_seconds: float | None,
    ) -> _TranscriptCandidate:
        cleaned = _clean_text(text)
        normalized = normalize_turn_text(cleaned) if cleaned else ""
        return _TranscriptCandidate(
            text=cleaned,
            normalized=normalized,
            source=source,
            provider_label=provider_label,
            confidence=confidence,
            avg_logprob=avg_logprob,
            usage_seconds=usage_seconds,
            words=len(cleaned.split()) if cleaned else 0,
            chars=len(cleaned),
        )

    def _choose_better_hint(self, partial_hint: str, controller_hint: str) -> str:
        if not controller_hint:
            return partial_hint
        if not partial_hint:
            return controller_hint
        partial_candidate = self._make_candidate(
            text=partial_hint,
            source="partial",
            provider_label="streaming",
            confidence=None,
            avg_logprob=None,
            usage_seconds=None,
        )
        controller_candidate = self._make_candidate(
            text=controller_hint,
            source="controller",
            provider_label="streaming",
            confidence=None,
            avg_logprob=None,
            usage_seconds=None,
        )
        selection = self._select_between_candidates(
            original=partial_candidate,
            alternative=controller_candidate,
            mode="hint",
            original_confidence=None,
            original_is_weak=partial_candidate.words < 2,
            original_language=_cfg(self._loop.config, "deepgram_stt_language", None),
        )
        return selection.text

    def _gate_implies_weak_original(self, gate: TranscriptVerifierGateDecision) -> bool:
        return gate.reason in {
            "empty_transcript",
            "utterance_end_without_speech_final",
            "confidence_below_threshold",
            "missing_confidence_short_turn",
        }

    def _verifier_providers(self, *, primary: Any) -> list[Any]:
        extras = [
            getattr(self._loop, "local_transcript_verifier_provider", None),
            getattr(self._loop, "fallback_transcript_verifier_provider", None),
        ]
        return _dedupe_providers([primary, *extras])

    def _recovery_providers(self, *, primary: Any) -> list[Any]:
        extras = [
            getattr(self._loop, "local_transcript_recovery_provider", None),
            getattr(self._loop, "fallback_transcript_recovery_provider", None),
        ]
        return _dedupe_providers([primary, *extras])

    def _emit_verifier_success(self, *, provider_label: str, primary_provider_label: str) -> None:
        normalized = provider_label.lower()
        self._loop.emit("stt_streaming_verified_via_verifier=true")
        primary_normalized = primary_provider_label.lower()
        if (
            normalized == primary_normalized
            or "openai" in normalized
            or any(token in normalized for token in ("4o", "transcribe", "whisper"))
        ):
            self._loop.emit("stt_streaming_verified_via_openai=true")

    def _verifier_instruction_lines(self, *, language: str) -> list[str]:
        lang = language.strip().lower()
        if lang.startswith("de"):
            return [
                "Die Audiodatei enthält eine kurze Äußerung an einen Sprachassistenten.",
                "Transkribiere wörtlich in der gesprochenen Sprache.",
                "Übersetze nicht und rate nicht.",
                "Behalte kurze umgangssprachliche Formen korrekt bei.",
                "Nutze jeden Hinweis unten nur als schwachen Kontext, niemals als Wahrheit.",
            ]
        if lang.startswith("en"):
            return [
                "The audio contains a short utterance to a voice assistant.",
                "Transcribe verbatim in the spoken language.",
                "Do not translate and do not guess.",
                "Keep contractions and colloquial short forms exactly when the audio supports them.",
                "Treat any hint below as weak context, never as ground truth.",
            ]
        return [
            "Verbatim transcript only.",
            "Use the spoken language from the audio.",
            "Do not translate. Do not guess.",
            "Treat any hint below as weak context, never as ground truth.",
        ]

    def _context_heading(self, *, language: str) -> str:
        return "Letzter Gesprächskontext (schwacher Hinweis):" if language.strip().lower().startswith("de") else "Recent conversation context (weak hint):"

    def _hint_heading(self, *, language: str) -> str:
        return "Streaming-Hinweis (schwacher Hinweis):" if language.strip().lower().startswith("de") else "Streaming hint (weak hint):"


def _text_summary(value: Any) -> dict[str, Any]:
    """Describe transcript text safely for forensics without raw content."""

    normalized = _clean_text(value)
    if not normalized:
        return {"present": False, "chars": 0, "words": 0, "sha12": None}
    digest = blake2s(normalized.encode("utf-8"), digest_size=6, key=_FORENSICS_KEY).hexdigest()
    return {
        "present": True,
        "chars": len(normalized),
        "words": len(normalized.split()),
        "sha12": digest,
    }


def _cfg(config: Any, name: str, default: Any) -> Any:
    try:
        value = getattr(config, name)
    except Exception:
        return default
    return default if value is None else value


def _lookup(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _pick_first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                return value
            continue
        return value
    return ""


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_usage_seconds(usage: Any) -> float | None:
    seconds = _coerce_float(_lookup(usage, "seconds"))
    return seconds if seconds is not None and seconds >= 0 else None


def _extract_avg_logprob(logprobs: Any) -> float | None:
    if not logprobs or not isinstance(logprobs, Sequence) or isinstance(logprobs, (str, bytes, bytearray)):
        return None
    values: list[float] = []
    for item in logprobs:
        value = _coerce_float(_lookup(item, "logprob"))
        if value is not None:
            values.append(value)
    if not values:
        return None
    return sum(values) / len(values)


def _dedupe_providers(providers: Iterable[Any]) -> list[Any]:
    seen: set[int] = set()
    unique: list[Any] = []
    for provider in providers:
        if provider is None or not callable(getattr(provider, "transcribe", None)):
            continue
        ident = id(provider)
        if ident in seen:
            continue
        seen.add(ident)
        unique.append(provider)
    return unique


def _clean_text(value: Any) -> str:
    text = str(value or "")
    text = _CONTROL_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _sanitize_context_value(value: Any, *, limit: int) -> str:
    text = _clean_text(value)
    if not text or limit <= 0:
        return ""
    text = text.replace('"', "'")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _provider_label(provider: Any) -> str:
    for attr_name in ("provider_label", "label", "name", "model"):
        value = getattr(provider, attr_name, None)
        if value:
            return str(value)
    return provider.__class__.__name__


def _provider_may_support_rich_transcription(provider: Any) -> bool:
    label = _provider_label(provider).lower()
    if any(token in label for token in ("openai", "4o", "transcribe", "whisper")):
        return True
    return bool(getattr(provider, "supports_transcribe_logprobs", False))


def _compare_texts(original: str, alternative: str) -> dict[str, float]:
    char_similarity = SequenceMatcher(None, original, alternative).ratio()
    original_tokens = original.split()
    alternative_tokens = alternative.split()
    token_similarity = SequenceMatcher(None, original_tokens, alternative_tokens).ratio()
    original_set = set(original_tokens)
    alternative_set = set(alternative_tokens)
    overlap = len(original_set & alternative_set)
    union = len(original_set | alternative_set)
    overlap_ratio = overlap / union if union else 0.0
    containment_ratio = overlap / len(original_set) if original_set else 0.0
    reverse_containment_ratio = overlap / len(alternative_set) if alternative_set else 0.0
    growth_ratio = _ratio(len(alternative), len(original))
    return {
        "char_similarity": char_similarity,
        "token_similarity": token_similarity,
        "overlap_ratio": overlap_ratio,
        "containment_ratio": containment_ratio,
        "reverse_containment_ratio": reverse_containment_ratio,
        "growth_ratio": growth_ratio,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(max(1, denominator))


def _filler_expansion_bonus(*, base: str, expanded: str, language: str) -> float:
    base_tokens = base.split()
    expanded_tokens = expanded.split()
    if not base_tokens or len(expanded_tokens) <= len(base_tokens):
        return 0.0
    base_set = set(base_tokens)
    extras = [token for token in expanded_tokens if token not in base_set]
    filler_set = _GERMAN_FILLERS if language.strip().lower().startswith("de") else _ENGLISH_FILLERS
    if extras and all(token in filler_set for token in extras):
        return 0.06
    return 0.0
