from __future__ import annotations

from dataclasses import dataclass

from twinr.hardware.audio import AmbientAudioCaptureWindow, pcm16_to_wav_bytes
from twinr.providers.openai.backend import OpenAIBackend
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
    normalized_label = normalize_detector_label(label)
    if not normalized_label:
        return None
    normalized_phrases = tuple(_normalize_text(phrase) for phrase in phrases if _normalize_text(phrase))
    if normalized_label in normalized_phrases:
        return normalized_label
    return None


def normalize_detector_label(label: str | None) -> str:
    raw_label = (label or "").strip()
    if not raw_label:
        return ""
    normalized = _normalize_text(raw_label.replace("_", " ").replace("-", " "))
    if not normalized:
        return ""
    words = normalized.split()
    while words and _looks_like_version_token(words[-1]):
        words.pop()
    filtered_words = [word for word in words if word not in _DECORATIVE_MODEL_WORDS]
    return " ".join(filtered_words) or normalized


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


def _normalize_text(text: str) -> str:
    return folded_lookup_text(text)


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
