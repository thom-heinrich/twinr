"""Match wakeword transcripts and detector labels against configured phrases.

This module normalizes phrase lists, generates transcription prompts, filters
prompt contamination, and returns structured match results that preserve any
spoken text remaining after the wakeword.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from twinr.hardware.audio import AmbientAudioCaptureWindow, pcm16_to_wav_bytes
from twinr.providers.openai import OpenAIBackend
from twinr.text_utils import folded_lookup_text


DEFAULT_WAKEWORD_PHRASES: tuple[str, ...] = (
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

_PROMPT_CONTAMINATION_MARKERS = (
    "if a name sounds close",
    "use that spelling exactly",
    "the clip may include",
    "wake word variants",
    "return only what was actually spoken",
)
_DECORATIVE_MODEL_WORDS = frozenset(
    {
        "custom",
        "family",
        "model",
        "models",
        "multispeaker",
        "multivoice",
        "openwakeword",
        "wakeword",
    }
)
_GENERIC_WAKEWORD_WORDS = frozenset(
    {"hey", "hallo", "he", "hi", "ok", "okay"}
)  # AUDIT-FIX(#4): Exclude greeting/filler tokens so prompt guidance follows the configured wakeword names.
_DEFAULT_MIN_PREFIX_RATIO = 0.9  # AUDIT-FIX(#3): Centralize the safe default for malformed ratio config.
_FALLBACK_PROMPT_NAMES = (
    "Twinr",
    "Twinna",
    "Twina",
    "Twinner",
)  # AUDIT-FIX(#4): Preserve prior canonical spellings when no usable prompt terms can be derived.


@dataclass(frozen=True, slots=True)
class WakewordMatch:
    """Describe one wakeword detection or rejection result."""

    detected: bool
    transcript: str
    matched_phrase: str | None = None
    remaining_text: str = ""
    normalized_transcript: str = ""
    backend: str = "stt"
    detector_label: str | None = None
    score: float | None = None


class WakewordPhraseSpotter:
    """Transcribe short audio windows and match configured wakeword phrases."""

    def __init__(
        self,
        *,
        backend: OpenAIBackend,
        phrases: tuple[str, ...] | list[str],
        language: str | None = None,
        min_prefix_ratio: float = _DEFAULT_MIN_PREFIX_RATIO,
    ) -> None:
        self.backend = backend
        self.phrases = _normalize_phrases(phrases)  # AUDIT-FIX(#6): Normalize once, deduplicate, and prefer more specific phrases first.
        self.language = _clean_text(language) or None
        self.min_prefix_ratio = _coerce_min_prefix_ratio(min_prefix_ratio)  # AUDIT-FIX(#3): Parse malformed, NaN, and infinite config safely.

    def detect(self, capture: AmbientAudioCaptureWindow) -> WakewordMatch:
        """Transcribe one capture and return the best wakeword match."""

        if not self.phrases:  # AUDIT-FIX(#6): Short-circuit invalid configuration instead of making STT calls that can never match.
            return WakewordMatch(detected=False, transcript="", normalized_transcript="", backend="stt")
        primary_prompt = wakeword_primary_prompt(self.phrases)
        transcript = self._safe_transcribe(capture, prompt=primary_prompt)  # AUDIT-FIX(#2): Contain provider/audio failures and fail closed.
        if transcript and _looks_like_prompt_contamination(
            transcript,
            prompt=primary_prompt,
        ):  # AUDIT-FIX(#1): Detect echoes of the current prompt, not only stale marker strings.
            transcript = self._safe_transcribe(capture, prompt=None)  # AUDIT-FIX(#1): Retry once without prompt guidance before matching.
            if _looks_like_prompt_contamination(transcript, prompt=None):
                return WakewordMatch(
                    detected=False,
                    transcript="",
                    normalized_transcript="",
                    backend="stt",
                )  # AUDIT-FIX(#1): Prompt-contaminated transcripts must never reach the matcher.
        return self.match_transcript(transcript)

    def match_transcript(self, transcript: str) -> WakewordMatch:
        """Match a transcript against the configured wakeword phrases."""

        return match_wakeword_transcript(
            transcript,
            phrases=self.phrases,
            min_prefix_ratio=self.min_prefix_ratio,
            backend="stt",
        )

    def _safe_transcribe(
        self,
        capture: AmbientAudioCaptureWindow,
        *,
        prompt: str | None,
    ) -> str:
        try:
            return self._transcribe(capture, prompt=prompt)  # AUDIT-FIX(#2): Keep provider and audio conversion exceptions inside the wakeword boundary.
        except Exception:
            return ""  # AUDIT-FIX(#2): Wakeword detection must degrade to no-match instead of crashing the agent loop.

    def _transcribe(
        self,
        capture: AmbientAudioCaptureWindow,
        *,
        prompt: str | None,
    ) -> str:
        pcm_bytes = bytes(capture.pcm_bytes or b"")  # AUDIT-FIX(#2): Normalize bytes-like audio inputs and empty buffers onto a safe path.
        sample_rate = int(capture.sample_rate)
        channels = int(capture.channels)
        if not pcm_bytes or sample_rate <= 0 or channels <= 0:
            return ""  # AUDIT-FIX(#2): Invalid capture metadata should not propagate into WAV conversion or remote calls.
        audio_bytes = pcm16_to_wav_bytes(
            pcm_bytes,
            sample_rate=sample_rate,
            channels=channels,
        )
        transcript = self.backend.transcribe(
            audio_bytes,
            filename="wakeword.wav",
            content_type="audio/wav",
            language=self.language,
            prompt=prompt,
        )
        return _clean_text(transcript)  # AUDIT-FIX(#2): Coerce non-string backend results safely before downstream normalization.


class WakewordTailTranscriptExtractor:
    """Extract continuation text after stage one already confirmed the wakeword.

    Some low-latency STT paths omit the leading wake phrase and return only the
    spoken tail, even when the capture still begins with the wakeword. For the
    backend-led orchestrator path, stage one has already confirmed the wake hit,
    so the remaining-text extractor must accept either:

    - a transcript that still contains the wake phrase, or
    - an already-trimmed continuation transcript that starts after the wake.
    """

    def __init__(
        self,
        *,
        backend: OpenAIBackend,
        phrases: tuple[str, ...] | list[str],
        language: str | None = None,
        min_prefix_ratio: float = _DEFAULT_MIN_PREFIX_RATIO,
    ) -> None:
        self._phrase_spotter = WakewordPhraseSpotter(
            backend=backend,
            phrases=phrases,
            language=language,
            min_prefix_ratio=min_prefix_ratio,
        )
        self.phrases = self._phrase_spotter.phrases

    def extract(self, capture: AmbientAudioCaptureWindow) -> str:
        """Return the spoken continuation text from one confirmed-wake capture."""

        match = self._phrase_spotter.detect(capture)
        if match.detected:
            return _clean_text(match.remaining_text)
        transcript = _clean_text(match.transcript)
        if not transcript:
            return ""
        normalized_transcript = _normalize_text(transcript)
        if normalized_transcript in self.phrases:
            return ""
        return transcript


def wakeword_primary_prompt(phrases: tuple[str, ...] | list[str]) -> str | None:
    """Build a transcription prompt from the configured wakeword names."""

    prompt_words: list[str] = []
    seen_phrases: set[str] = set()
    seen_words: set[str] = set()
    for phrase in phrases:
        normalized_phrase = _normalize_text(phrase)
        if not normalized_phrase or normalized_phrase in seen_phrases:
            continue
        seen_phrases.add(normalized_phrase)
        for word in normalized_phrase.split():
            if word in _GENERIC_WAKEWORD_WORDS:
                continue
            if word not in seen_words:
                seen_words.add(word)
                prompt_words.append(word.title())
    if not prompt_words:
        if not _normalize_phrases(phrases):
            return None
        prompt_words = list(_FALLBACK_PROMPT_NAMES)
    return ", ".join(prompt_words) + "."  # AUDIT-FIX(#4): Bias transcription toward the active wakeword spellings instead of a hard-coded set.


def match_wakeword_transcript(
    transcript: str,
    *,
    phrases: tuple[str, ...] | list[str],
    min_prefix_ratio: float = _DEFAULT_MIN_PREFIX_RATIO,
    backend: str = "stt",
    detector_label: str | None = None,
    score: float | None = None,
) -> WakewordMatch:
    """Match a transcript prefix against configured wakeword phrases."""

    normalized_phrases = _normalize_phrases(phrases)  # AUDIT-FIX(#6): Use the same canonical phrase preparation everywhere.
    cleaned_transcript = _clean_text(transcript)
    original_words, normalized_words = _normalized_word_lists(
        cleaned_transcript
    )  # AUDIT-FIX(#6): Preserve token alignment so remaining_text slicing stays stable after normalization.
    normalized_transcript = " ".join(word for word in normalized_words if word)
    if not normalized_transcript or not normalized_phrases:
        return WakewordMatch(
            detected=False,
            transcript=cleaned_transcript,
            normalized_transcript=normalized_transcript,
            backend=backend,
            detector_label=detector_label,
            score=score,
        )

    clamped_ratio = _coerce_min_prefix_ratio(min_prefix_ratio)  # AUDIT-FIX(#5): Actually apply the configured prefix ratio to matching behavior.
    for phrase in normalized_phrases:
        phrase_words = phrase.split()
        for candidate_word_count in range(len(phrase_words), 0, -1):
            phrase_prefix = " ".join(phrase_words[:candidate_word_count])
            if len(phrase_prefix) / len(phrase) < clamped_ratio:
                continue
            for start_index, normalized_prefix in _candidate_segments(normalized_words, candidate_word_count):
                if normalized_prefix == phrase_prefix:
                    remaining_text = " ".join(original_words[start_index + candidate_word_count :]).strip(" ,.!?:;")
                    return WakewordMatch(
                        detected=True,
                        transcript=cleaned_transcript,
                        matched_phrase=phrase,
                        remaining_text=remaining_text,
                        normalized_transcript=normalized_transcript,
                        backend=backend,
                        detector_label=detector_label,
                        score=score,
                    )
    return WakewordMatch(
        detected=False,
        transcript=cleaned_transcript,
        normalized_transcript=normalized_transcript,
        backend=backend,
        detector_label=detector_label,
        score=score,
    )


def phrase_from_detector_label(
    label: str | None,
    *,
    phrases: tuple[str, ...] | list[str],
) -> str | None:
    """Resolve one detector label back to a configured wakeword phrase."""

    normalized_label = normalize_detector_label(label)
    if not normalized_label:
        return None
    normalized_phrases = _normalize_phrases(phrases)  # AUDIT-FIX(#6): Reuse canonical phrase preparation so detector-label lookup matches runtime matching.
    if normalized_label in normalized_phrases:
        return normalized_label
    return None


def normalize_detector_label(label: str | None) -> str:
    """Normalize an openWakeWord detector label into phrase form."""

    raw_label = _clean_text(label)
    if not raw_label:
        return ""
    normalized = _normalize_text(raw_label.replace("_", " ").replace("-", " "))
    if not normalized:
        return ""
    words = normalized.split()
    while words and _looks_like_version_token(words[-1]):
        words.pop()
    filtered_words = [word for word in words if word not in _DECORATIVE_MODEL_WORDS]
    return " ".join(filtered_words)  # AUDIT-FIX(#7): Decorative-only labels must normalize to empty, not back to the original garbage label.


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


def _normalize_text(text: str | bytes | bytearray | None) -> str:
    return folded_lookup_text(_coerce_text(text))  # AUDIT-FIX(#2): Harden normalization against None and bytes-like provider outputs.


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


def _looks_like_version_token(value: str) -> bool:
    token = value.strip().lower()
    if not token:
        return False
    if token.startswith("v"):
        token = token[1:]
    separators = {".", "_", "-"}
    parts: list[str] = []
    current: list[str] = []
    for char in token:
        if char.isdigit():
            current.append(char)
            continue
        if char in separators:
            if not current:
                return False
            parts.append("".join(current))
            current = []
            continue
        return False
    if current:
        parts.append("".join(current))
    return bool(parts)


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
        return True  # AUDIT-FIX(#1): Exact prompt echoes must never be eligible wakeword transcripts.
    if len(prompt_word_set) >= 2 and len(transcript_word_set) >= 2 and transcript_word_set.issubset(prompt_word_set):
        return True  # AUDIT-FIX(#1): Multi-name transcripts made only of prompt words are treated as contamination.
    return False


__all__ = [
    "DEFAULT_WAKEWORD_PHRASES",
    "WakewordMatch",
    "WakewordPhraseSpotter",
    "WakewordTailTranscriptExtractor",
    "match_wakeword_transcript",
    "normalize_detector_label",
    "phrase_from_detector_label",
    "wakeword_primary_prompt",
]
