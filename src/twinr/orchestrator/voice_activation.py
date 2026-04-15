# CHANGELOG: 2026-03-29
# BUG-1: Fixed utterance-head false negatives caused by punctuation/empty tokens and hyphenated ASR output.
# BUG-2: Fixed silent backend/preprocessing failures by surfacing structured error metadata instead of collapsing them into plain "no match".
# BUG-3: Fixed accidental enablement of risky "*winner" recovery in contextual-bias mode, which could raise false activations.
# BUG-4: Removed wake-specific STT prompting from the productive matcher because guided activation prompts could distort generic transcript-first windows into unrelated text or empty matches.
# SEC-1: Added PCM frame-alignment, duration/size guards, and 16 kHz mono conditioning before remote upload to reduce practical Pi 4 DoS/cost risk.
# IMP-1: Added compatibility with structured 2026 ASR responses (text, words, timings, confidence, backend labels) while remaining drop-in for legacy str backends.
# IMP-2: Added timestamp/confidence-aware matching, bounded fuzzy prefix recovery, and more precise remainder extraction.

"""Match transcript-first voice activations on the remote thh1986 stream.

This module keeps the remote-only activation matching logic outside the
realtime and websocket orchestration layers. It accepts short ASR windows,
matches one configured activation prefix, and preserves any spoken remainder.

The 2026 upgrade keeps the original string-only API intact, while also
accepting richer ASR outputs that expose text, word timings, confidence, and
backend metadata.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
import math
import re
from typing import Any, Protocol

from twinr.hardware.audio import AmbientAudioCaptureWindow, pcm16_to_wav_bytes
from twinr.text_utils import folded_lookup_text


DEFAULT_VOICE_ACTIVATION_PHRASES: tuple[str, ...] = (
    "hey twinr",
    "he twinr",
    "hey twinna",
    "hey twina",
    "hey twinner",
    "hallo twinr",
    "hallo twinna",
    "hallo twina",
    "hallo twinner",
    "twinr hallo",
    "twinr hey",
    "twinna hallo",
    "twinna hey",
    "twina hallo",
    "twina hey",
    "twinner hallo",
    "twinner hey",
    "twinr",
    "twinna",
    "twina",
    "twinner",
)
_CONTEXTUAL_TYNNA_VOICE_ACTIVATION_PHRASES: tuple[str, ...] = (
    "hey tynna",
    "hallo tynna",
    "tynna hallo",
    "tynna hey",
    "tynna",
)

_GENERIC_ACTIVATION_WORDS = frozenset({"hey", "hallo", "he", "hi", "ok", "okay"})
_DEFAULT_MIN_PREFIX_RATIO = 0.9
_DEFAULT_MIN_FUZZY_WORD_RATIO = 0.84
_DEFAULT_TARGET_SAMPLE_RATE = 16_000
_DEFAULT_TARGET_CHANNELS = 1
_DEFAULT_MAX_AUDIO_SECONDS = 8.0
_DEFAULT_MAX_WAV_BYTES = 2_000_000
_TEXT_EDGE_STRIP_CHARS = " \t\r\n,.;:!?…-–—'\"`()[]{}<>"

_TWI_HEURISTIC_PREFIX = "twi"
_TWI_HEAD_VARIANT_MAX_EXTRA_PREFIX_CHARS = 4
_TWI_HEAD_VARIANT_MIN_RATIO = 0.7
_TWI_HEURISTIC_BLOCKLIST = frozenset(
    {
        "twice",
        "twig",
        "twigs",
        "twilight",
        "twin",
        "twine",
        "twins",
        "twirl",
        "twirls",
        "twist",
        "twisted",
        "twisting",
        "twists",
        "twitch",
        "twitchy",
    }
)

_WORD_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


class _TranscriptBackend(Protocol):
    """Describe the bounded ASR surface needed for activation matching."""

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> Any:
        """Transcribe one bounded audio payload into plain or structured text."""


@dataclass(frozen=True, slots=True)
class TranscriptWord:
    """One ASR token with optional timing and confidence."""

    text: str
    start_ms: float | None = None
    end_ms: float | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Normalized transcription result from a legacy or structured backend."""

    text: str
    words: tuple[TranscriptWord, ...] = ()
    backend: str = "remote_asr"
    detector_label: str | None = None
    confidence: float | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class VoiceActivationMatch:
    """Describe one activation detection or rejection result."""

    detected: bool
    transcript: str
    matched_phrase: str | None = None
    remaining_text: str = ""
    normalized_transcript: str = ""
    backend: str = "remote_asr"
    detector_label: str | None = None
    score: float | None = None
    error: str | None = None
    transcript_confidence: float | None = None
    activation_start_ms: float | None = None
    activation_end_ms: float | None = None
    audio_duration_s: float | None = None
    match_kind: str | None = None


@dataclass(frozen=True, slots=True)
class _PreparedAudio:
    wav_bytes: bytes
    duration_s: float | None = None
    sample_rate: int = _DEFAULT_TARGET_SAMPLE_RATE
    channels: int = _DEFAULT_TARGET_CHANNELS
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _TranscriptToken:
    original: str
    normalized: str
    start_ms: float | None = None
    end_ms: float | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class _ActivationCandidate:
    phrase: str
    remaining_text: str
    start_ms: float | None = None
    end_ms: float | None = None
    confidence: float | None = None
    match_kind: str = "exact"


class VoiceActivationPhraseMatcher:
    """Transcribe short audio windows and match configured activation phrases."""

    def __init__(
        self,
        *,
        backend: _TranscriptBackend,
        phrases: tuple[str, ...] | list[str],
        language: str | None = None,
        min_prefix_ratio: float = _DEFAULT_MIN_PREFIX_RATIO,
        suppress_transcription_errors: bool = True,
        target_sample_rate: int = _DEFAULT_TARGET_SAMPLE_RATE,
        target_channels: int = _DEFAULT_TARGET_CHANNELS,
        max_audio_seconds: float = _DEFAULT_MAX_AUDIO_SECONDS,
        max_wav_bytes: int = _DEFAULT_MAX_WAV_BYTES,
        enable_fuzzy_prefix_match: bool = True,
        min_fuzzy_word_ratio: float = _DEFAULT_MIN_FUZZY_WORD_RATIO,
        allow_twi_head_variant_recovery: bool | None = None,
        min_activation_word_confidence: float | None = None,
    ) -> None:
        self.backend = backend
        self.phrases = _normalize_phrases(phrases)
        self.language = _clean_text(language) or None
        self.min_prefix_ratio = _coerce_min_prefix_ratio(min_prefix_ratio)
        self.suppress_transcription_errors = bool(suppress_transcription_errors)
        self.target_sample_rate = _coerce_positive_int(target_sample_rate, _DEFAULT_TARGET_SAMPLE_RATE)
        self.target_channels = _coerce_target_channels(target_channels, _DEFAULT_TARGET_CHANNELS)
        self.max_audio_seconds = _coerce_positive_float(max_audio_seconds)
        self.max_wav_bytes = _coerce_positive_int(max_wav_bytes, _DEFAULT_MAX_WAV_BYTES)
        self.enable_fuzzy_prefix_match = bool(enable_fuzzy_prefix_match)
        self.min_fuzzy_word_ratio = _coerce_fuzzy_ratio(min_fuzzy_word_ratio)
        self.allow_twi_head_variant_recovery = allow_twi_head_variant_recovery
        self.min_activation_word_confidence = _coerce_optional_unit_float(
            min_activation_word_confidence
        )

    def detect(self, capture: AmbientAudioCaptureWindow) -> VoiceActivationMatch:
        """Transcribe one capture and return the best activation match."""

        if not self.phrases:
            return VoiceActivationMatch(
                detected=False,
                transcript="",
                normalized_transcript="",
                backend="remote_asr",
            )

        prepared_audio = _prepare_capture_for_transcription(
            capture,
            target_sample_rate=self.target_sample_rate,
            target_channels=self.target_channels,
            max_audio_seconds=self.max_audio_seconds,
            max_wav_bytes=self.max_wav_bytes,
        )
        if not prepared_audio.wav_bytes:
            return VoiceActivationMatch(
                detected=False,
                transcript="",
                normalized_transcript="",
                backend="remote_asr",
                error=prepared_audio.error,
                audio_duration_s=prepared_audio.duration_s,
            )

        transcript_result = self._safe_transcribe(prepared_audio, prompt=None)
        return self.match_transcript_result(
            transcript_result,
            audio_duration_s=prepared_audio.duration_s,
        )

    def match_transcript(self, transcript: str) -> VoiceActivationMatch:
        """Match one transcript against the configured activation phrases."""

        return match_voice_activation_transcript(
            transcript,
            phrases=self.phrases,
            min_prefix_ratio=self.min_prefix_ratio,
            backend="remote_asr",
            enable_fuzzy_prefix_match=self.enable_fuzzy_prefix_match,
            min_fuzzy_word_ratio=self.min_fuzzy_word_ratio,
            allow_twi_head_variant_recovery=self.allow_twi_head_variant_recovery,
            min_activation_word_confidence=self.min_activation_word_confidence,
        )

    def match_transcript_result(
        self,
        result: TranscriptionResult,
        *,
        audio_duration_s: float | None = None,
    ) -> VoiceActivationMatch:
        """Match one structured transcription result."""

        return match_voice_activation_transcript(
            result.text,
            phrases=self.phrases,
            min_prefix_ratio=self.min_prefix_ratio,
            backend=result.backend,
            detector_label=result.detector_label,
            score=None,
            transcript_words=result.words,
            transcript_confidence=result.confidence,
            error=result.error,
            audio_duration_s=audio_duration_s,
            enable_fuzzy_prefix_match=self.enable_fuzzy_prefix_match,
            min_fuzzy_word_ratio=self.min_fuzzy_word_ratio,
            allow_twi_head_variant_recovery=self.allow_twi_head_variant_recovery,
            min_activation_word_confidence=self.min_activation_word_confidence,
        )

    def _safe_transcribe(
        self,
        prepared_audio: _PreparedAudio,
        *,
        prompt: str | None,
    ) -> TranscriptionResult:
        try:
            return self._transcribe(prepared_audio, prompt=prompt)
        except Exception as exc:
            if not self.suppress_transcription_errors:
                raise
            return TranscriptionResult(
                text="",
                backend="remote_asr",
                error=exc.__class__.__name__,
            )

    def _transcribe(
        self,
        prepared_audio: _PreparedAudio,
        *,
        prompt: str | None,
    ) -> TranscriptionResult:
        raw_result = self.backend.transcribe(
            prepared_audio.wav_bytes,
            filename="voice-activation.wav",
            content_type="audio/wav",
            language=self.language,
            prompt=prompt,
        )
        return _coerce_transcription_result(raw_result, default_backend="remote_asr")


class VoiceActivationTailExtractor:
    """Extract continuation text after stage one already confirmed activation."""

    def __init__(
        self,
        *,
        backend: _TranscriptBackend,
        phrases: tuple[str, ...] | list[str],
        language: str | None = None,
        min_prefix_ratio: float = _DEFAULT_MIN_PREFIX_RATIO,
        target_sample_rate: int = _DEFAULT_TARGET_SAMPLE_RATE,
        target_channels: int = _DEFAULT_TARGET_CHANNELS,
        max_audio_seconds: float = _DEFAULT_MAX_AUDIO_SECONDS,
        max_wav_bytes: int = _DEFAULT_MAX_WAV_BYTES,
        enable_fuzzy_prefix_match: bool = True,
        min_fuzzy_word_ratio: float = _DEFAULT_MIN_FUZZY_WORD_RATIO,
        allow_twi_head_variant_recovery: bool | None = None,
        min_activation_word_confidence: float | None = None,
    ) -> None:
        self._phrase_matcher = VoiceActivationPhraseMatcher(
            backend=backend,
            phrases=phrases,
            language=language,
            min_prefix_ratio=min_prefix_ratio,
            target_sample_rate=target_sample_rate,
            target_channels=target_channels,
            max_audio_seconds=max_audio_seconds,
            max_wav_bytes=max_wav_bytes,
            enable_fuzzy_prefix_match=enable_fuzzy_prefix_match,
            min_fuzzy_word_ratio=min_fuzzy_word_ratio,
            allow_twi_head_variant_recovery=allow_twi_head_variant_recovery,
            min_activation_word_confidence=min_activation_word_confidence,
        )
        self.phrases = self._phrase_matcher.phrases

    def extract(self, capture: AmbientAudioCaptureWindow) -> str:
        """Return the spoken continuation text from one confirmed-activation capture."""

        match = self._phrase_matcher.detect(capture)
        if match.detected:
            return _clean_text(match.remaining_text)
        transcript = _clean_text(match.transcript)
        if not transcript:
            return ""
        normalized_transcript = _normalize_phrase_text(transcript)
        if normalized_transcript in self.phrases:
            return ""
        return transcript


def match_voice_activation_transcript(
    transcript: str,
    *,
    phrases: tuple[str, ...] | list[str],
    min_prefix_ratio: float = _DEFAULT_MIN_PREFIX_RATIO,
    backend: str = "remote_asr",
    detector_label: str | None = None,
    score: float | None = None,
    transcript_words: Sequence[TranscriptWord | Mapping[str, Any] | object] | None = None,
    transcript_confidence: float | None = None,
    error: str | None = None,
    audio_duration_s: float | None = None,
    enable_fuzzy_prefix_match: bool = True,
    min_fuzzy_word_ratio: float = _DEFAULT_MIN_FUZZY_WORD_RATIO,
    allow_twi_head_variant_recovery: bool | None = None,
    min_activation_word_confidence: float | None = None,
) -> VoiceActivationMatch:
    """Match one transcript prefix against configured activation phrases."""

    normalized_phrases = _normalize_phrases(phrases)
    cleaned_transcript = _clean_text(transcript)
    structured_words = _coerce_transcript_words(transcript_words)
    if not cleaned_transcript and structured_words:
        cleaned_transcript = " ".join(word.text for word in structured_words if _clean_text(word.text))
    tokens = _transcript_tokens(cleaned_transcript, structured_words)
    normalized_transcript = " ".join(token.normalized for token in tokens if token.normalized)

    if not normalized_transcript or not normalized_phrases:
        return VoiceActivationMatch(
            detected=False,
            transcript=cleaned_transcript,
            normalized_transcript=normalized_transcript,
            backend=backend,
            detector_label=detector_label,
            score=score,
            error=error,
            transcript_confidence=transcript_confidence,
            audio_duration_s=audio_duration_s,
        )

    clamped_ratio = _coerce_min_prefix_ratio(min_prefix_ratio)
    min_activation_word_confidence = _coerce_optional_unit_float(min_activation_word_confidence)
    exact_match = _match_activation_phrase_prefix(
        tokens=tokens,
        normalized_phrases=normalized_phrases,
        min_prefix_ratio=clamped_ratio,
        min_activation_word_confidence=min_activation_word_confidence,
    )
    if exact_match is not None:
        return VoiceActivationMatch(
            detected=True,
            transcript=cleaned_transcript,
            matched_phrase=exact_match.phrase,
            remaining_text=exact_match.remaining_text,
            normalized_transcript=normalized_transcript,
            backend=backend,
            detector_label=detector_label,
            score=score if score is not None else exact_match.confidence,
            error=error,
            transcript_confidence=transcript_confidence,
            activation_start_ms=exact_match.start_ms,
            activation_end_ms=exact_match.end_ms,
            audio_duration_s=audio_duration_s,
            match_kind=exact_match.match_kind,
        )

    if enable_fuzzy_prefix_match:
        fuzzy_match = _match_fuzzy_activation_phrase_prefix(
            tokens=tokens,
            normalized_phrases=normalized_phrases,
            min_prefix_ratio=clamped_ratio,
            min_fuzzy_word_ratio=_coerce_fuzzy_ratio(min_fuzzy_word_ratio),
            min_activation_word_confidence=min_activation_word_confidence,
        )
        if fuzzy_match is not None:
            return VoiceActivationMatch(
                detected=True,
                transcript=cleaned_transcript,
                matched_phrase=fuzzy_match.phrase,
                remaining_text=fuzzy_match.remaining_text,
                normalized_transcript=normalized_transcript,
                backend=backend,
                detector_label=detector_label,
                score=score if score is not None else fuzzy_match.confidence,
                error=error,
                transcript_confidence=transcript_confidence,
                activation_start_ms=fuzzy_match.start_ms,
                activation_end_ms=fuzzy_match.end_ms,
                audio_duration_s=audio_duration_s,
                match_kind=fuzzy_match.match_kind,
            )

    if _allows_twi_head_variant_recovery(
        normalized_phrases,
        explicit_opt_in=allow_twi_head_variant_recovery,
    ):
        head_variant_match = _match_twi_head_variant(
            tokens=tokens,
            activation_aliases=tuple(
                phrase
                for phrase in normalized_phrases
                if phrase.startswith(_TWI_HEURISTIC_PREFIX) and " " not in phrase
            ),
            min_activation_word_confidence=min_activation_word_confidence,
        )
        if head_variant_match is not None:
            return VoiceActivationMatch(
                detected=True,
                transcript=cleaned_transcript,
                matched_phrase=head_variant_match.phrase,
                remaining_text=head_variant_match.remaining_text,
                normalized_transcript=normalized_transcript,
                backend=backend,
                detector_label=detector_label,
                score=score if score is not None else head_variant_match.confidence,
                error=error,
                transcript_confidence=transcript_confidence,
                activation_start_ms=head_variant_match.start_ms,
                activation_end_ms=head_variant_match.end_ms,
                audio_duration_s=audio_duration_s,
                match_kind=head_variant_match.match_kind,
            )

    return VoiceActivationMatch(
        detected=False,
        transcript=cleaned_transcript,
        normalized_transcript=normalized_transcript,
        backend=backend,
        detector_label=detector_label,
        score=score,
        error=error,
        transcript_confidence=transcript_confidence,
        audio_duration_s=audio_duration_s,
    )


def normalize_activation_detector_label(label: str | None) -> str:
    """Normalize one detector label into phrase form."""

    return _normalize_phrase_text(str(label or "").replace("_", " ").replace("-", " "))


def phrase_from_activation_detector_label(
    label: str | None,
    *,
    phrases: tuple[str, ...] | list[str],
) -> str | None:
    """Resolve one detector label back to a configured activation phrase."""

    normalized_label = normalize_activation_detector_label(label)
    if not normalized_label:
        return None
    normalized_phrases = _normalize_phrases(phrases)
    if normalized_label in normalized_phrases:
        return normalized_label
    return None


def _match_activation_phrase_prefix(
    *,
    tokens: list[_TranscriptToken],
    normalized_phrases: tuple[str, ...],
    min_prefix_ratio: float,
    min_activation_word_confidence: float | None,
) -> _ActivationCandidate | None:
    """Match only utterance-head activation phrases, not mid-sentence mentions."""

    head_start_indices = _utterance_head_start_indices(tokens)
    for phrase in normalized_phrases:
        phrase_words = phrase.split()
        for candidate_word_count in range(len(phrase_words), 0, -1):
            phrase_prefix_words = phrase_words[:candidate_word_count]
            if _phrase_prefix_ratio(phrase_prefix_words, phrase_words) < min_prefix_ratio:
                continue
            phrase_prefix = tuple(phrase_prefix_words)
            for start_index, segment_tokens in _candidate_token_segments(
                tokens,
                candidate_word_count,
                start_indices=head_start_indices,
            ):
                segment_words = tuple(token.normalized for token in segment_tokens)
                if segment_words != phrase_prefix:
                    continue
                if not _activation_confidence_ok(
                    segment_tokens,
                    min_activation_word_confidence,
                ):
                    continue
                return _activation_candidate_from_tokens(
                    phrase=phrase,
                    tokens=tokens,
                    start_index=start_index,
                    consumed_word_count=candidate_word_count,
                    confidence=1.0,
                    match_kind="exact",
                )
    return None


def _match_fuzzy_activation_phrase_prefix(
    *,
    tokens: list[_TranscriptToken],
    normalized_phrases: tuple[str, ...],
    min_prefix_ratio: float,
    min_fuzzy_word_ratio: float,
    min_activation_word_confidence: float | None,
) -> _ActivationCandidate | None:
    """Recover small ASR spelling mistakes at the utterance head."""

    head_start_indices = _utterance_head_start_indices(tokens)
    best_candidate: _ActivationCandidate | None = None
    best_key: tuple[float, float, str] | None = None

    for phrase in normalized_phrases:
        phrase_words = phrase.split()
        for candidate_word_count in range(len(phrase_words), 0, -1):
            phrase_prefix_words = phrase_words[:candidate_word_count]
            if _phrase_prefix_ratio(phrase_prefix_words, phrase_words) < min_prefix_ratio:
                continue
            for start_index, segment_tokens in _candidate_token_segments(
                tokens,
                candidate_word_count,
                start_indices=head_start_indices,
            ):
                segment_words = tuple(token.normalized for token in segment_tokens)
                if segment_words == tuple(phrase_prefix_words):
                    continue
                if not _activation_confidence_ok(
                    segment_tokens,
                    min_activation_word_confidence,
                ):
                    continue
                if not _fuzzy_segment_allowed(segment_words, phrase_prefix_words):
                    continue
                ratios = tuple(
                    SequenceMatcher(a=segment_word, b=phrase_word).ratio()
                    for segment_word, phrase_word in zip(segment_words, phrase_prefix_words)
                )
                if not ratios or min(ratios) < min_fuzzy_word_ratio:
                    continue
                average_ratio = sum(ratios) / len(ratios)
                key = (average_ratio, float(candidate_word_count), phrase)
                if best_key is not None and key <= best_key:
                    continue
                best_key = key
                best_candidate = _activation_candidate_from_tokens(
                    phrase=phrase,
                    tokens=tokens,
                    start_index=start_index,
                    consumed_word_count=candidate_word_count,
                    confidence=average_ratio,
                    match_kind="fuzzy",
                )
    return best_candidate


def _candidate_token_segments(
    tokens: list[_TranscriptToken],
    phrase_word_count: int,
    *,
    start_indices: tuple[int, ...] | None = None,
) -> list[tuple[int, tuple[_TranscriptToken, ...]]]:
    if len(tokens) < phrase_word_count or phrase_word_count <= 0:
        return []
    segments: list[tuple[int, tuple[_TranscriptToken, ...]]] = []
    max_start = len(tokens) - phrase_word_count
    candidate_start_indices: Iterable[int]
    if start_indices is None:
        candidate_start_indices = range(max_start + 1)
    else:
        candidate_start_indices = start_indices
    for start_index in candidate_start_indices:
        if start_index < 0 or start_index > max_start:
            continue
        segments.append(
            (start_index, tuple(tokens[start_index : start_index + phrase_word_count]))
        )
    return segments


def _twi_heuristic_candidates(
    tokens: list[_TranscriptToken],
    index: int,
) -> tuple[tuple[str, int], ...]:
    if index < 0 or index >= len(tokens):
        return ()
    word = tokens[index].normalized
    if not word:
        return ()
    candidates: list[tuple[str, int]] = [(word, 1)]
    if index + 1 >= len(tokens):
        return tuple(candidates)
    next_word = tokens[index + 1].normalized
    if not next_word or next_word in _GENERIC_ACTIVATION_WORDS or len(next_word) > 3:
        return tuple(candidates)
    if word == "twin" or (word.startswith(_TWI_HEURISTIC_PREFIX) and len(word) <= 5):
        candidates.append((f"{word}{next_word}", 2))
    return tuple(candidates)


def _utterance_head_start_indices(tokens: list[_TranscriptToken]) -> tuple[int, ...]:
    """Return the bounded utterance-head offsets that may host one wake prefix."""

    if not tokens:
        return ()
    head_end_index = 0
    for index, token in enumerate(tokens):
        head_end_index = index
        if token.normalized not in _GENERIC_ACTIVATION_WORDS:
            break
    return tuple(range(head_end_index + 1))


def _match_twi_head_variant(
    *,
    tokens: list[_TranscriptToken],
    activation_aliases: tuple[str, ...],
    min_activation_word_confidence: float | None,
) -> _ActivationCandidate | None:
    """Match bounded utterance-head ASR variants like ``Gewinner`` to ``Twinner``."""

    if not tokens or not activation_aliases:
        return None
    head_index = 0
    while head_index < len(tokens) and tokens[head_index].normalized in _GENERIC_ACTIVATION_WORDS:
        head_index += 1
    if head_index >= len(tokens):
        return None
    for candidate, consumed_word_count in _twi_heuristic_candidates(tokens, head_index):
        candidate_tokens = tokens[head_index : head_index + consumed_word_count]
        if not _activation_confidence_ok(candidate_tokens, min_activation_word_confidence):
            continue
        matched_phrase = _best_twi_head_variant_phrase(candidate, activation_aliases)
        if not matched_phrase:
            continue
        return _activation_candidate_from_tokens(
            phrase=matched_phrase,
            tokens=tokens,
            start_index=head_index,
            consumed_word_count=consumed_word_count,
            confidence=_TWI_HEAD_VARIANT_MIN_RATIO,
            match_kind="twi_head_variant",
        )
    return None


def _best_twi_head_variant_phrase(candidate: str, aliases: tuple[str, ...]) -> str | None:
    """Resolve one bounded utterance-head ``*winner`` ASR variant back to a configured alias."""

    normalized_candidate = str(candidate or "")
    if not normalized_candidate or normalized_candidate in _TWI_HEURISTIC_BLOCKLIST:
        return None
    best_phrase: str | None = None
    best_key: tuple[float, int, str] | None = None
    for phrase in aliases:
        alias_tail = phrase[1:]
        extra_prefix_chars = len(normalized_candidate) - len(alias_tail)
        if (
            not alias_tail
            or not alias_tail.startswith("win")
            or extra_prefix_chars < 0
            or extra_prefix_chars > _TWI_HEAD_VARIANT_MAX_EXTRA_PREFIX_CHARS
            or not normalized_candidate.endswith(alias_tail)
        ):
            continue
        ratio = SequenceMatcher(a=normalized_candidate, b=phrase).ratio()
        if ratio < _TWI_HEAD_VARIANT_MIN_RATIO:
            continue
        key = (ratio, -extra_prefix_chars, phrase)
        if best_key is None or key > best_key:
            best_key = key
            best_phrase = phrase
    return best_phrase


def _activation_candidate_from_tokens(
    *,
    phrase: str,
    tokens: list[_TranscriptToken],
    start_index: int,
    consumed_word_count: int,
    confidence: float | None,
    match_kind: str,
) -> _ActivationCandidate:
    end_index = start_index + consumed_word_count
    remaining_text = " ".join(
        token.original for token in tokens[end_index:]
    ).strip(_TEXT_EDGE_STRIP_CHARS)
    matched_tokens = tokens[start_index:end_index]
    start_ms = matched_tokens[0].start_ms if matched_tokens else None
    end_ms = matched_tokens[-1].end_ms if matched_tokens else None
    return _ActivationCandidate(
        phrase=phrase,
        remaining_text=remaining_text,
        start_ms=start_ms,
        end_ms=end_ms,
        confidence=confidence,
        match_kind=match_kind,
    )


def _activation_confidence_ok(
    tokens: Sequence[_TranscriptToken],
    min_activation_word_confidence: float | None,
) -> bool:
    threshold = _coerce_optional_unit_float(min_activation_word_confidence)
    if threshold is None:
        return True
    confidences = [
        confidence
        for confidence in (token.confidence for token in tokens)
        if isinstance(confidence, (int, float)) and math.isfinite(float(confidence))
    ]
    if not confidences:
        return True
    average_confidence = sum(float(value) for value in confidences) / len(confidences)
    return average_confidence >= threshold


def _phrase_prefix_ratio(
    prefix_words: Sequence[str],
    full_phrase_words: Sequence[str],
) -> float:
    prefix = " ".join(prefix_words).strip()
    full_phrase = " ".join(full_phrase_words).strip()
    if not prefix or not full_phrase:
        return 0.0
    return len(prefix) / len(full_phrase)


def _fuzzy_segment_allowed(
    segment_words: Sequence[str],
    phrase_prefix_words: Sequence[str],
) -> bool:
    """Reject fuzzy recovery that drifts outside the explicit wake alias family."""

    for segment_word, phrase_word in zip(segment_words, phrase_prefix_words):
        if (
            phrase_word.startswith(_TWI_HEURISTIC_PREFIX)
            and not segment_word.startswith("tw")
        ):
            return False
    return True


def _iter_text_tokens(text: str) -> Iterable[str]:
    for match in _WORD_TOKEN_RE.finditer(_clean_text(text)):
        token = _clean_text(match.group(0))
        if token:
            yield token


def _normalize_text(text: str | bytes | bytearray | None) -> str:
    return folded_lookup_text(_coerce_text(text))


def _normalize_phrase_text(text: str | bytes | bytearray | None) -> str:
    normalized_words: list[str] = []
    for token in _iter_text_tokens(_coerce_text(text)):
        normalized_token = _normalize_text(token)
        if normalized_token:
            normalized_words.append(normalized_token)
    return " ".join(normalized_words)


def _normalize_phrases(phrases: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized_phrases: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        normalized_phrase = _normalize_phrase_text(phrase)
        if not normalized_phrase or normalized_phrase in seen:
            continue
        seen.add(normalized_phrase)
        normalized_phrases.append(normalized_phrase)
    normalized_phrases.sort(key=lambda value: (-len(value.split()), -len(value), value))
    return tuple(normalized_phrases)


def _coerce_text(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="ignore")
    return str(value)


def _clean_text(value: object | None) -> str:
    return " ".join(_coerce_text(value).split()).strip()


_DEFAULT_SAFE_ACTIVATION_PHRASE_SET = frozenset(_normalize_phrases(DEFAULT_VOICE_ACTIVATION_PHRASES))


def contextual_bias_voice_activation_phrases(
    phrases: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    """Return the contextual alias family used only in strong speaker-bias mode."""

    normalized_phrases = _normalize_phrases(phrases)
    if not normalized_phrases:
        return ()
    if any(phrase not in _DEFAULT_SAFE_ACTIVATION_PHRASE_SET for phrase in normalized_phrases):
        return normalized_phrases
    return _normalize_phrases(
        (*normalized_phrases, *_CONTEXTUAL_TYNNA_VOICE_ACTIVATION_PHRASES)
    )


# BREAKING: contextual "tynna" bias no longer auto-enables the riskier
# "*winner" recovery path. If you intentionally relied on the old behavior,
# pass allow_twi_head_variant_recovery=True.
def _allows_twi_head_variant_recovery(
    normalized_phrases: tuple[str, ...],
    *,
    explicit_opt_in: bool | None = None,
) -> bool:
    """Enable riskier ``*winner``-style recovery only for explicit broader twi* alias sets."""

    if explicit_opt_in is not None:
        return bool(explicit_opt_in)
    return any(
        " " not in phrase
        and (
            (
                phrase.startswith(_TWI_HEURISTIC_PREFIX)
                and phrase not in _DEFAULT_SAFE_ACTIVATION_PHRASE_SET
            )
            or phrase in _CONTEXTUAL_TYNNA_VOICE_ACTIVATION_PHRASES
        )
        for phrase in normalized_phrases
    )


def _coerce_min_prefix_ratio(value: object) -> float:
    if not isinstance(value, (int, float, str, bytes, bytearray)) or isinstance(value, bool):
        return _DEFAULT_MIN_PREFIX_RATIO
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return _DEFAULT_MIN_PREFIX_RATIO
    if not math.isfinite(ratio):
        return _DEFAULT_MIN_PREFIX_RATIO
    return max(0.5, min(1.0, ratio))


def _coerce_fuzzy_ratio(value: object) -> float:
    if not isinstance(value, (int, float, str, bytes, bytearray)) or isinstance(value, bool):
        return _DEFAULT_MIN_FUZZY_WORD_RATIO
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return _DEFAULT_MIN_FUZZY_WORD_RATIO
    if not math.isfinite(ratio):
        return _DEFAULT_MIN_FUZZY_WORD_RATIO
    return max(0.65, min(0.99, ratio))


def _coerce_positive_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return default
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    return coerced if coerced > 0 else default


def _coerce_target_channels(value: object, default: int) -> int:
    return _coerce_positive_int(value, default)


def _coerce_positive_float(value: object | None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(coerced) or coerced <= 0.0:
        return None
    return coerced


def _coerce_optional_float(value: object | None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(coerced):
        return None
    return coerced


def _coerce_optional_unit_float(value: object | None) -> float | None:
    coerced = _coerce_optional_float(value)
    if coerced is None:
        return None
    return max(0.0, min(1.0, coerced))


def _coerce_transcription_result(value: object, *, default_backend: str) -> TranscriptionResult:
    if isinstance(value, TranscriptionResult):
        return TranscriptionResult(
            text=_clean_text(value.text),
            words=_coerce_transcript_words(value.words),
            backend=_clean_text(value.backend) or default_backend,
            detector_label=_clean_text(value.detector_label) or None,
            confidence=_coerce_optional_float(value.confidence),
            error=_clean_text(value.error) or None,
        )

    if isinstance(value, (str, bytes, bytearray)):
        return TranscriptionResult(
            text=_clean_text(value),
            backend=default_backend,
        )

    if isinstance(value, Mapping):
        text = _clean_text(
            value.get("text")
            or value.get("transcript")
            or value.get("utterance")
            or value.get("result")
        )
        words = _coerce_transcript_words(
            value.get("words")
            or value.get("tokens")
            or value.get("word_timestamps")
        )
        segments = value.get("segments")
        text, words = _merge_segment_text_and_words(text=text, words=words, segments=segments)
        return TranscriptionResult(
            text=text,
            words=words,
            backend=_clean_text(value.get("backend") or value.get("provider") or default_backend)
            or default_backend,
            detector_label=_clean_text(value.get("detector_label") or value.get("label")) or None,
            confidence=_coerce_optional_float(
                value.get("confidence")
                or value.get("score")
                or value.get("avg_logprob")
            ),
            error=_clean_text(value.get("error")) or None,
        )

    text = _clean_text(
        getattr(value, "text", None)
        or getattr(value, "transcript", None)
        or getattr(value, "utterance", None)
        or getattr(value, "result", None)
    )
    words = _coerce_transcript_words(
        getattr(value, "words", None)
        or getattr(value, "tokens", None)
        or getattr(value, "word_timestamps", None)
    )
    segments = getattr(value, "segments", None)
    text, words = _merge_segment_text_and_words(text=text, words=words, segments=segments)
    return TranscriptionResult(
        text=text,
        words=words,
        backend=_clean_text(
            getattr(value, "backend", None) or getattr(value, "provider", None) or default_backend
        )
        or default_backend,
        detector_label=_clean_text(
            getattr(value, "detector_label", None) or getattr(value, "label", None)
        )
        or None,
        confidence=_coerce_optional_float(
            getattr(value, "confidence", None)
            or getattr(value, "score", None)
            or getattr(value, "avg_logprob", None)
        ),
        error=_clean_text(getattr(value, "error", None)) or None,
    )


def _merge_segment_text_and_words(
    *,
    text: str,
    words: tuple[TranscriptWord, ...],
    segments: object | None,
) -> tuple[str, tuple[TranscriptWord, ...]]:
    if not segments:
        return text, words
    segment_items = segments if isinstance(segments, Sequence) and not isinstance(segments, (str, bytes, bytearray)) else ()
    if not segment_items:
        return text, words

    merged_words = list(words)
    merged_text_parts: list[str] = [text] if text else []
    for segment in segment_items:
        if not text:
            merged_text_parts.append(
                _clean_text(
                    segment.get("text") if isinstance(segment, Mapping) else getattr(segment, "text", None)
                )
            )
        if not words:
            segment_words = _coerce_transcript_words(
                segment.get("words") if isinstance(segment, Mapping) else getattr(segment, "words", None)
            )
            merged_words.extend(segment_words)

    merged_text = _clean_text(" ".join(part for part in merged_text_parts if part))
    return merged_text, tuple(merged_words)


def _coerce_transcript_words(
    words: object | None,
) -> tuple[TranscriptWord, ...]:
    if words is None:
        return ()
    if isinstance(words, TranscriptWord):
        return (
            TranscriptWord(
                text=_clean_text(words.text),
                start_ms=words.start_ms,
                end_ms=words.end_ms,
                confidence=_coerce_optional_float(words.confidence),
            ),
        )
    if isinstance(words, Mapping) or isinstance(words, (str, bytes, bytearray)):
        return ()
    if not isinstance(words, Iterable):
        return ()
    coerced_words: list[TranscriptWord] = []
    for word in words:
        coerced_word = _coerce_one_transcript_word(word)
        if coerced_word is not None:
            coerced_words.append(coerced_word)
    return tuple(coerced_words)


def _coerce_one_transcript_word(word: object | None) -> TranscriptWord | None:
    if word is None:
        return None
    if isinstance(word, TranscriptWord):
        text = _clean_text(word.text)
        if not text:
            return None
        return TranscriptWord(
            text=text,
            start_ms=_coerce_optional_float(word.start_ms),
            end_ms=_coerce_optional_float(word.end_ms),
            confidence=_coerce_optional_float(word.confidence),
        )

    if isinstance(word, Mapping):
        text = _clean_text(
            word.get("word")
            or word.get("text")
            or word.get("token")
            or word.get("value")
        )
        if not text:
            return None
        return TranscriptWord(
            text=text,
            start_ms=_coerce_optional_float(word.get("start_ms"))
            or _seconds_to_ms(word.get("start")),
            end_ms=_coerce_optional_float(word.get("end_ms"))
            or _seconds_to_ms(word.get("end")),
            confidence=_coerce_optional_float(
                word.get("confidence")
                or word.get("score")
                or word.get("probability")
                or word.get("prob")
            ),
        )

    text = _clean_text(
        getattr(word, "word", None)
        or getattr(word, "text", None)
        or getattr(word, "token", None)
        or getattr(word, "value", None)
    )
    if not text:
        return None
    return TranscriptWord(
        text=text,
        start_ms=_coerce_optional_float(getattr(word, "start_ms", None))
        or _seconds_to_ms(getattr(word, "start", None)),
        end_ms=_coerce_optional_float(getattr(word, "end_ms", None))
        or _seconds_to_ms(getattr(word, "end", None)),
        confidence=_coerce_optional_float(
            getattr(word, "confidence", None)
            or getattr(word, "score", None)
            or getattr(word, "probability", None)
            or getattr(word, "prob", None)
        ),
    )


def _seconds_to_ms(value: object | None) -> float | None:
    seconds = _coerce_optional_float(value)
    if seconds is None:
        return None
    return seconds * 1000.0


def _transcript_tokens(
    transcript: str,
    words: tuple[TranscriptWord, ...],
) -> list[_TranscriptToken]:
    base_tokens: list[_TranscriptToken] = []
    for token in _iter_text_tokens(transcript):
        normalized_token = _normalize_text(token)
        if not normalized_token:
            continue
        base_tokens.append(
            _TranscriptToken(
                original=token,
                normalized=normalized_token,
            )
        )

    if not words:
        return base_tokens

    word_tokens: list[_TranscriptToken] = []
    for word in words:
        for token in _iter_text_tokens(word.text):
            normalized_token = _normalize_text(token)
            if not normalized_token:
                continue
            word_tokens.append(
                _TranscriptToken(
                    original=token,
                    normalized=normalized_token,
                    start_ms=word.start_ms,
                    end_ms=word.end_ms,
                    confidence=_coerce_optional_float(word.confidence),
                )
            )

    if not word_tokens:
        return base_tokens
    if not base_tokens:
        return word_tokens

    enriched_tokens: list[_TranscriptToken] = []
    word_index = 0
    for base_token in base_tokens:
        matched_word_token: _TranscriptToken | None = None
        scan_index = word_index
        while scan_index < len(word_tokens):
            candidate = word_tokens[scan_index]
            if candidate.normalized == base_token.normalized:
                matched_word_token = candidate
                word_index = scan_index + 1
                break
            scan_index += 1
        if matched_word_token is None:
            enriched_tokens.append(base_token)
            continue
        enriched_tokens.append(
            _TranscriptToken(
                original=base_token.original,
                normalized=base_token.normalized,
                start_ms=matched_word_token.start_ms,
                end_ms=matched_word_token.end_ms,
                confidence=matched_word_token.confidence,
            )
        )
    return enriched_tokens


def _prepare_capture_for_transcription(
    capture: AmbientAudioCaptureWindow,
    *,
    target_sample_rate: int,
    target_channels: int,
    max_audio_seconds: float | None,
    max_wav_bytes: int,
) -> _PreparedAudio:
    pcm_bytes = bytes(getattr(capture, "pcm_bytes", b"") or b"")
    sample_rate = _coerce_positive_int(getattr(capture, "sample_rate", 0), 0)
    channels = _coerce_positive_int(getattr(capture, "channels", 0), 0)
    if not pcm_bytes or sample_rate <= 0 or channels <= 0:
        return _PreparedAudio(wav_bytes=b"", duration_s=None, error=None)

    pcm_bytes = _align_pcm16_frames(pcm_bytes, channels)
    if not pcm_bytes:
        return _PreparedAudio(wav_bytes=b"", duration_s=None, error="invalid_pcm_alignment")

    if max_audio_seconds is not None:
        max_frame_count = max(1, int(sample_rate * max_audio_seconds))
        max_raw_bytes = max_frame_count * channels * 2
        if len(pcm_bytes) > max_raw_bytes:
            pcm_bytes = pcm_bytes[:max_raw_bytes]

    duration_s = len(pcm_bytes) / float(sample_rate * channels * 2)

    conditioned_pcm = pcm_bytes
    conditioned_sample_rate = sample_rate
    conditioned_channels = channels

    if channels != target_channels and target_channels == 1:
        conditioned_pcm = _downmix_pcm16_to_mono(conditioned_pcm, channels)
        conditioned_channels = 1

    if conditioned_sample_rate != target_sample_rate and conditioned_channels == 1:
        conditioned_pcm = _resample_mono_pcm16(
            conditioned_pcm,
            source_sample_rate=conditioned_sample_rate,
            target_sample_rate=target_sample_rate,
        )
        conditioned_sample_rate = target_sample_rate

    wav_bytes = pcm16_to_wav_bytes(
        conditioned_pcm,
        sample_rate=conditioned_sample_rate,
        channels=conditioned_channels,
    )
    if len(wav_bytes) > max_wav_bytes:
        return _PreparedAudio(
            wav_bytes=b"",
            duration_s=duration_s,
            sample_rate=conditioned_sample_rate,
            channels=conditioned_channels,
            error="audio_too_large",
        )

    return _PreparedAudio(
        wav_bytes=wav_bytes,
        duration_s=duration_s,
        sample_rate=conditioned_sample_rate,
        channels=conditioned_channels,
    )


def _align_pcm16_frames(pcm_bytes: bytes, channels: int) -> bytes:
    frame_size = max(1, channels) * 2
    if frame_size <= 0:
        return b""
    usable_length = len(pcm_bytes) - (len(pcm_bytes) % frame_size)
    if usable_length <= 0:
        return b""
    return pcm_bytes[:usable_length]


def _downmix_pcm16_to_mono(pcm_bytes: bytes, channels: int) -> bytes:
    if channels <= 1 or not pcm_bytes:
        return pcm_bytes
    samples = array("h")
    samples.frombytes(pcm_bytes)
    if not samples:
        return b""
    frame_count = len(samples) // channels
    mono_samples = array("h")
    for frame_index in range(frame_count):
        start = frame_index * channels
        frame = samples[start : start + channels]
        if not frame:
            continue
        average = int(round(sum(int(sample) for sample in frame) / len(frame)))
        mono_samples.append(_clamp_pcm16(average))
    return mono_samples.tobytes()


def _resample_mono_pcm16(
    pcm_bytes: bytes,
    *,
    source_sample_rate: int,
    target_sample_rate: int,
) -> bytes:
    if (
        not pcm_bytes
        or source_sample_rate <= 0
        or target_sample_rate <= 0
        or source_sample_rate == target_sample_rate
    ):
        return pcm_bytes

    source_samples = array("h")
    source_samples.frombytes(pcm_bytes)
    if len(source_samples) <= 1:
        return pcm_bytes

    target_length = max(1, int(round(len(source_samples) * target_sample_rate / source_sample_rate)))
    if target_length == len(source_samples):
        return pcm_bytes

    resampled = array("h")
    last_source_index = len(source_samples) - 1
    if target_length == 1:
        resampled.append(int(source_samples[0]))
        return resampled.tobytes()

    position_scale = last_source_index / float(target_length - 1)
    for target_index in range(target_length):
        source_position = target_index * position_scale
        left_index = int(source_position)
        right_index = min(left_index + 1, last_source_index)
        fraction = source_position - left_index
        left_value = int(source_samples[left_index])
        right_value = int(source_samples[right_index])
        interpolated = int(round(left_value + ((right_value - left_value) * fraction)))
        resampled.append(_clamp_pcm16(interpolated))
    return resampled.tobytes()


def _clamp_pcm16(value: int) -> int:
    return max(-32768, min(32767, int(value)))


__all__ = [
    "DEFAULT_VOICE_ACTIVATION_PHRASES",
    "TranscriptWord",
    "TranscriptionResult",
    "VoiceActivationMatch",
    "VoiceActivationPhraseMatcher",
    "VoiceActivationTailExtractor",
    "contextual_bias_voice_activation_phrases",
    "match_voice_activation_transcript",
    "normalize_activation_detector_label",
    "phrase_from_activation_detector_label",
]
