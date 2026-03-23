"""Match transcript-first voice activations on the remote thh1986 stream.

This module keeps the remote-only activation matching logic outside the
realtime and websocket orchestration layers. It accepts short ASR windows,
matches one configured activation prefix, and preserves any spoken remainder.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import math
from typing import Protocol

from twinr.hardware.audio import AmbientAudioCaptureWindow, pcm16_to_wav_bytes
from twinr.text_utils import folded_lookup_text


DEFAULT_VOICE_ACTIVATION_PHRASES: tuple[str, ...] = (
    "hey twinr",
    "he twinr",
    "hey twinna",
    "hey twina",
    "hey twinner",
    "hey twitter",
    "hallo twinr",
    "hallo twinna",
    "hallo twina",
    "hallo twinner",
    "hallo twitter",
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
    "twitter",
)

_PROMPT_CONTAMINATION_MARKERS = (
    "if a name sounds close",
    "use that spelling exactly",
    "the clip may include",
    "wake word variants",
    "return only what was actually spoken",
)
_GENERIC_ACTIVATION_WORDS = frozenset({"hey", "hallo", "he", "hi", "ok", "okay"})
_DEFAULT_MIN_PREFIX_RATIO = 0.9
_FALLBACK_PROMPT_NAMES = ("Twinr", "Twinna", "Twina", "Twinner")
_TWI_HEURISTIC_PREFIX = "twi"
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
    ) -> str:
        """Transcribe one bounded audio payload into plain text."""


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
    ) -> None:
        self.backend = backend
        self.phrases = _normalize_phrases(phrases)
        self.language = _clean_text(language) or None
        self.min_prefix_ratio = _coerce_min_prefix_ratio(min_prefix_ratio)
        self.suppress_transcription_errors = bool(suppress_transcription_errors)

    def detect(self, capture: AmbientAudioCaptureWindow) -> VoiceActivationMatch:
        """Transcribe one capture and return the best activation match."""

        if not self.phrases:
            return VoiceActivationMatch(
                detected=False,
                transcript="",
                normalized_transcript="",
                backend="remote_asr",
            )
        primary_prompt = voice_activation_primary_prompt(self.phrases)
        transcript = self._safe_transcribe(capture, prompt=primary_prompt)
        if transcript and _looks_like_prompt_contamination(transcript, prompt=primary_prompt):
            transcript = self._safe_transcribe(capture, prompt=None)
            if _looks_like_prompt_contamination(transcript, prompt=None):
                return VoiceActivationMatch(
                    detected=False,
                    transcript="",
                    normalized_transcript="",
                    backend="remote_asr",
                )
        return self.match_transcript(transcript)

    def match_transcript(self, transcript: str) -> VoiceActivationMatch:
        """Match one transcript against the configured activation phrases."""

        return match_voice_activation_transcript(
            transcript,
            phrases=self.phrases,
            min_prefix_ratio=self.min_prefix_ratio,
            backend="remote_asr",
        )

    def _safe_transcribe(
        self,
        capture: AmbientAudioCaptureWindow,
        *,
        prompt: str | None,
    ) -> str:
        try:
            return self._transcribe(capture, prompt=prompt)
        except Exception:
            if not self.suppress_transcription_errors:
                raise
            return ""

    def _transcribe(
        self,
        capture: AmbientAudioCaptureWindow,
        *,
        prompt: str | None,
    ) -> str:
        pcm_bytes = bytes(capture.pcm_bytes or b"")
        sample_rate = int(capture.sample_rate)
        channels = int(capture.channels)
        if not pcm_bytes or sample_rate <= 0 or channels <= 0:
            return ""
        audio_bytes = pcm16_to_wav_bytes(
            pcm_bytes,
            sample_rate=sample_rate,
            channels=channels,
        )
        transcript = self.backend.transcribe(
            audio_bytes,
            filename="voice-activation.wav",
            content_type="audio/wav",
            language=self.language,
            prompt=prompt,
        )
        return _clean_text(transcript)


class VoiceActivationTailExtractor:
    """Extract continuation text after stage one already confirmed activation."""

    def __init__(
        self,
        *,
        backend: _TranscriptBackend,
        phrases: tuple[str, ...] | list[str],
        language: str | None = None,
        min_prefix_ratio: float = _DEFAULT_MIN_PREFIX_RATIO,
    ) -> None:
        self._phrase_matcher = VoiceActivationPhraseMatcher(
            backend=backend,
            phrases=phrases,
            language=language,
            min_prefix_ratio=min_prefix_ratio,
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
        normalized_transcript = _normalize_text(transcript)
        if normalized_transcript in self.phrases:
            return ""
        return transcript


def voice_activation_primary_prompt(phrases: tuple[str, ...] | list[str]) -> str | None:
    """Build one transcription prompt from the configured activation names."""

    prompt_words: list[str] = []
    seen_phrases: set[str] = set()
    seen_words: set[str] = set()
    for phrase in phrases:
        normalized_phrase = _normalize_text(phrase)
        if not normalized_phrase or normalized_phrase in seen_phrases:
            continue
        seen_phrases.add(normalized_phrase)
        for word in normalized_phrase.split():
            if word in _GENERIC_ACTIVATION_WORDS:
                continue
            if word not in seen_words:
                seen_words.add(word)
                prompt_words.append(word.title())
    if not prompt_words:
        if not _normalize_phrases(phrases):
            return None
        prompt_words = list(_FALLBACK_PROMPT_NAMES)
    return ", ".join(prompt_words) + "."


def match_voice_activation_transcript(
    transcript: str,
    *,
    phrases: tuple[str, ...] | list[str],
    min_prefix_ratio: float = _DEFAULT_MIN_PREFIX_RATIO,
    backend: str = "remote_asr",
    detector_label: str | None = None,
    score: float | None = None,
) -> VoiceActivationMatch:
    """Match one transcript prefix against configured activation phrases."""

    normalized_phrases = _normalize_phrases(phrases)
    cleaned_transcript = _clean_text(transcript)
    original_words, normalized_words = _normalized_word_lists(cleaned_transcript)
    normalized_transcript = " ".join(word for word in normalized_words if word)
    if not normalized_transcript or not normalized_phrases:
        return VoiceActivationMatch(
            detected=False,
            transcript=cleaned_transcript,
            normalized_transcript=normalized_transcript,
            backend=backend,
            detector_label=detector_label,
            score=score,
        )

    clamped_ratio = _coerce_min_prefix_ratio(min_prefix_ratio)
    for phrase in normalized_phrases:
        phrase_words = phrase.split()
        for candidate_word_count in range(len(phrase_words), 0, -1):
            phrase_prefix = " ".join(phrase_words[:candidate_word_count])
            if len(phrase_prefix) / len(phrase) < clamped_ratio:
                continue
            for start_index, normalized_prefix in _candidate_segments(normalized_words, candidate_word_count):
                if normalized_prefix == phrase_prefix:
                    remaining_text = " ".join(original_words[start_index + candidate_word_count :]).strip(" ,.!?:;")
                    return VoiceActivationMatch(
                        detected=True,
                        transcript=cleaned_transcript,
                        matched_phrase=phrase,
                        remaining_text=remaining_text,
                        normalized_transcript=normalized_transcript,
                        backend=backend,
                        detector_label=detector_label,
                        score=score,
                    )
    heuristic_match = _match_twi_activation_heuristic(
        original_words=original_words,
        normalized_words=normalized_words,
        normalized_phrases=normalized_phrases,
    )
    if heuristic_match is not None:
        matched_phrase, remaining_text = heuristic_match
        return VoiceActivationMatch(
            detected=True,
            transcript=cleaned_transcript,
            matched_phrase=matched_phrase,
            remaining_text=remaining_text,
            normalized_transcript=normalized_transcript,
            backend=backend,
            detector_label=detector_label,
            score=score,
        )
    return VoiceActivationMatch(
        detected=False,
        transcript=cleaned_transcript,
        normalized_transcript=normalized_transcript,
        backend=backend,
        detector_label=detector_label,
        score=score,
    )


def normalize_activation_detector_label(label: str | None) -> str:
    """Normalize one detector label into phrase form."""

    return _normalize_text(str(label or "").replace("_", " ").replace("-", " "))


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


def _candidate_segments(
    normalized_words: list[str],
    phrase_word_count: int,
) -> list[tuple[int, str]]:
    if len(normalized_words) < phrase_word_count:
        return []
    segments: list[tuple[int, str]] = []
    max_start = len(normalized_words) - phrase_word_count
    for start_index in range(max_start + 1):
        segment = " ".join(normalized_words[start_index : start_index + phrase_word_count])
        segments.append((start_index, segment))
    return segments


def _match_twi_activation_heuristic(
    *,
    original_words: list[str],
    normalized_words: list[str],
    normalized_phrases: tuple[str, ...],
) -> tuple[str, str] | None:
    """Apply the approved `twi*` activation fallback for transcript-first ASR."""

    activation_aliases = tuple(
        phrase
        for phrase in normalized_phrases
        if phrase.startswith(_TWI_HEURISTIC_PREFIX) and " " not in phrase
    )
    if not activation_aliases:
        return None
    for index, word in enumerate(normalized_words):
        if not word or word in _GENERIC_ACTIVATION_WORDS:
            continue
        for candidate, consumed_word_count in _twi_heuristic_candidates(normalized_words, index):
            if candidate in _TWI_HEURISTIC_BLOCKLIST or not candidate.startswith(_TWI_HEURISTIC_PREFIX):
                continue
            matched_phrase = _best_twi_heuristic_phrase(candidate, activation_aliases)
            if not matched_phrase:
                continue
            remaining_text = " ".join(original_words[index + consumed_word_count :]).strip(" ,.!?:;")
            return matched_phrase, remaining_text
    return None


def _twi_heuristic_candidates(
    normalized_words: list[str],
    index: int,
) -> tuple[tuple[str, int], ...]:
    word = normalized_words[index]
    if not word:
        return ()
    candidates: list[tuple[str, int]] = [(word, 1)]
    if index + 1 >= len(normalized_words):
        return tuple(candidates)
    next_word = normalized_words[index + 1]
    if not next_word or next_word in _GENERIC_ACTIVATION_WORDS or len(next_word) > 3:
        return tuple(candidates)
    if word == "twin" or (word.startswith(_TWI_HEURISTIC_PREFIX) and len(word) <= 5):
        candidates.append((f"{word}{next_word}", 2))
    return tuple(candidates)


def _best_twi_heuristic_phrase(candidate: str, aliases: tuple[str, ...]) -> str | None:
    if not aliases:
        return None
    best_phrase: str | None = None
    best_key: tuple[int, float, int, str] | None = None
    for phrase in aliases:
        key = (
            _shared_prefix_length(candidate, phrase),
            SequenceMatcher(a=candidate, b=phrase).ratio(),
            -abs(len(candidate) - len(phrase)),
            phrase,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_phrase = phrase
    return best_phrase


def _shared_prefix_length(left: str, right: str) -> int:
    shared = 0
    for left_char, right_char in zip(left, right, strict=False):
        if left_char != right_char:
            break
        shared += 1
    return shared


def _normalize_text(text: str | bytes | bytearray | None) -> str:
    return folded_lookup_text(_coerce_text(text))


def _normalize_phrases(phrases: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized_phrases: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        normalized_phrase = _normalize_text(phrase)
        if not normalized_phrase or normalized_phrase in seen:
            continue
        seen.add(normalized_phrase)
        normalized_phrases.append(normalized_phrase)
    normalized_phrases.sort(key=lambda value: (-len(value.split()), -len(value), value))
    return tuple(normalized_phrases)


def _normalized_word_lists(text: str) -> tuple[list[str], list[str]]:
    original_words = text.split()
    normalized_words = [_normalize_text(word) for word in original_words]
    return original_words, normalized_words


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


def _coerce_min_prefix_ratio(value: object) -> float:
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return _DEFAULT_MIN_PREFIX_RATIO
    if not math.isfinite(ratio):
        return _DEFAULT_MIN_PREFIX_RATIO
    return max(0.5, min(1.0, ratio))


def _looks_like_prompt_contamination(text: str, *, prompt: str | None = None) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    if any(marker in normalized for marker in _PROMPT_CONTAMINATION_MARKERS):
        return True
    normalized_prompt = _normalize_text(prompt)
    if not normalized_prompt:
        return False
    prompt_words = tuple(word for word in normalized_prompt.split() if word)
    transcript_words = tuple(word for word in normalized.split() if word)
    prompt_word_set = set(prompt_words)
    transcript_word_set = set(transcript_words)
    if len(prompt_word_set) >= 2 and normalized == normalized_prompt:
        return True
    if len(prompt_word_set) >= 2 and len(transcript_word_set) >= 2 and transcript_word_set.issubset(prompt_word_set):
        return True
    return False


__all__ = [
    "DEFAULT_VOICE_ACTIVATION_PHRASES",
    "VoiceActivationMatch",
    "VoiceActivationPhraseMatcher",
    "VoiceActivationTailExtractor",
    "match_voice_activation_transcript",
    "normalize_activation_detector_label",
    "phrase_from_activation_detector_label",
    "voice_activation_primary_prompt",
]
