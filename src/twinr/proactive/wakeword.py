from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re

from twinr.hardware.audio import AmbientAudioCaptureWindow, pcm16_to_wav_bytes
from twinr.providers.openai.backend import OpenAIBackend


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

_NON_ALNUM_RE = re.compile(r"[^0-9a-zA-ZäöüÄÖÜß]+")
_LEADING_WAKEWORD_FILLERS = frozenset(
    {
        "ja",
        "na",
        "hm",
        "hmm",
        "äh",
        "eh",
        "oh",
        "ah",
        "also",
        "hallo",
        "hey",
        "he",
        "hi",
        "bitte",
        "mal",
        "du",
    }
)
_MAX_LEADING_WAKEWORD_FILLERS = 2
_PROMPT_CONTAMINATION_MARKERS = (
    "if a name sounds close",
    "use that spelling exactly",
    "the clip may include",
    "wake word variants",
    "return only what was actually spoken",
)
_MODEL_VERSION_SUFFIX_RE = re.compile(r"(?:[\s_-]v?\d+(?:[._-]\d+)*)+$")


@dataclass(frozen=True, slots=True)
class WakewordMatch:
    detected: bool
    transcript: str
    matched_phrase: str | None = None
    remaining_text: str = ""
    normalized_transcript: str = ""
    backend: str = "stt"
    detector_label: str | None = None
    score: float | None = None


class WakewordPhraseSpotter:
    def __init__(
        self,
        *,
        backend: OpenAIBackend,
        phrases: tuple[str, ...],
        language: str | None = None,
        min_prefix_ratio: float = 0.9,
    ) -> None:
        self.backend = backend
        self.phrases = tuple(_normalize_text(phrase) for phrase in phrases if _normalize_text(phrase))
        self.language = (language or "").strip() or None
        self.min_prefix_ratio = max(0.5, min(1.0, float(min_prefix_ratio)))

    def detect(self, capture: AmbientAudioCaptureWindow) -> WakewordMatch:
        transcript = self._transcribe(capture, prompt=wakeword_primary_prompt(self.phrases))
        if _looks_like_prompt_contamination(transcript):
            transcript = self._transcribe(capture, prompt=None)
        return self.match_transcript(transcript)

    def match_transcript(self, transcript: str) -> WakewordMatch:
        return match_wakeword_transcript(
            transcript,
            phrases=self.phrases,
            min_prefix_ratio=self.min_prefix_ratio,
            backend="stt",
        )

    def _transcribe(
        self,
        capture: AmbientAudioCaptureWindow,
        *,
        prompt: str | None,
    ) -> str:
        audio_bytes = pcm16_to_wav_bytes(
            capture.pcm_bytes,
            sample_rate=capture.sample_rate,
            channels=capture.channels,
        )
        return self.backend.transcribe(
            audio_bytes,
            filename="wakeword.wav",
            content_type="audio/wav",
            language=self.language,
            prompt=prompt,
        )


def wakeword_primary_prompt(phrases: tuple[str, ...] | list[str]) -> str | None:
    if not phrases:
        return None
    canonical_names = ("Twinr", "Twinna", "Twina", "Twinner")
    return ", ".join(canonical_names) + "."


def match_wakeword_transcript(
    transcript: str,
    *,
    phrases: tuple[str, ...] | list[str],
    min_prefix_ratio: float = 0.9,
    backend: str = "stt",
    detector_label: str | None = None,
    score: float | None = None,
) -> WakewordMatch:
    normalized_phrases = tuple(_normalize_text(phrase) for phrase in phrases if _normalize_text(phrase))
    cleaned_transcript = " ".join(str(transcript).split()).strip()
    normalized_transcript = _normalize_text(cleaned_transcript)
    if not normalized_transcript:
        return WakewordMatch(
            detected=False,
            transcript=cleaned_transcript,
            normalized_transcript=normalized_transcript,
            backend=backend,
            detector_label=detector_label,
            score=score,
        )

    original_words = cleaned_transcript.split()
    normalized_words = normalized_transcript.split()
    for phrase in normalized_phrases:
        phrase_words = phrase.split()
        for start_index, normalized_prefix in _candidate_segments(normalized_words, len(phrase_words)):
            if normalized_prefix == phrase:
                remaining_text = " ".join(original_words[start_index + len(phrase_words) :]).strip(" ,.!?:;")
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
    for phrase in normalized_phrases:
        phrase_words = phrase.split()
        if len(phrase) < 5:
            continue
        for start_index, normalized_prefix in _candidate_segments(normalized_words, len(phrase_words)):
            if SequenceMatcher(None, normalized_prefix, phrase).ratio() >= min_prefix_ratio:
                remaining_text = " ".join(original_words[start_index + len(phrase_words) :]).strip(" ,.!?:;")
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
    min_ratio: float = 0.82,
) -> str | None:
    normalized_label = normalize_detector_label(label)
    if not normalized_label:
        return None
    normalized_phrases = tuple(_normalize_text(phrase) for phrase in phrases if _normalize_text(phrase))
    if normalized_label in normalized_phrases:
        return normalized_label
    best_phrase = ""
    best_score = 0.0
    for phrase in normalized_phrases:
        score = SequenceMatcher(None, normalized_label, phrase).ratio()
        if score > best_score:
            best_score = score
            best_phrase = phrase
    if best_phrase and best_score >= min_ratio:
        return best_phrase
    return normalized_label


def normalize_detector_label(label: str | None) -> str:
    raw_label = (label or "").strip()
    if not raw_label:
        return ""
    without_version = _MODEL_VERSION_SUFFIX_RE.sub("", raw_label)
    return _normalize_text(without_version.replace("_", " ").replace("-", " "))


def _candidate_segments(
    normalized_words: list[str],
    phrase_word_count: int,
) -> list[tuple[int, str]]:
    if len(normalized_words) < phrase_word_count:
        return []
    last_start = len(normalized_words) - phrase_word_count
    segments: list[tuple[int, str]] = []
    for start_index in range(0, min(_MAX_LEADING_WAKEWORD_FILLERS, last_start) + 1):
        if start_index > 0:
            leading_words = normalized_words[:start_index]
            if any(word not in _LEADING_WAKEWORD_FILLERS for word in leading_words):
                continue
        segment = " ".join(normalized_words[start_index : start_index + phrase_word_count])
        segments.append((start_index, segment))
    return segments


def _normalize_text(text: str) -> str:
    lowered = (text or "").strip().lower()
    if not lowered:
        return ""
    compact = _NON_ALNUM_RE.sub(" ", lowered)
    return " ".join(compact.split())


def _looks_like_prompt_contamination(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return any(marker in normalized for marker in _PROMPT_CONTAMINATION_MARKERS)


__all__ = [
    "DEFAULT_WAKEWORD_PHRASES",
    "WakewordMatch",
    "WakewordPhraseSpotter",
    "match_wakeword_transcript",
    "normalize_detector_label",
    "phrase_from_detector_label",
    "wakeword_primary_prompt",
]
