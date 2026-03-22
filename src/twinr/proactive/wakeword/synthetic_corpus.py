"""Plan and augment Twinr's large synthetic Qwen3TTS wakeword corpus.

This module owns the deterministic corpus recipe for Twinr's custom WeKws
training path. It builds a broad wakeword-family request plan with many
speaker/style/generation combinations, applies channel and room degradations
without relying on heavyweight DSP stacks, and writes manifest rows that can be
fed into Twinr's existing WeKws export and benchmark pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Iterable
import wave

import numpy as np


QWEN3TTS_SUPPORTED_SPEAKERS: tuple[str, ...] = (
    "serena",
    "vivian",
    "uncle_fu",
    "ryan",
    "aiden",
    "ono_anna",
    "sohee",
    "eric",
    "dylan",
)

WAKEWORD_FAMILY: tuple[str, ...] = ("twinr", "twinna", "twina", "twinner")
CONFUSION_FAMILY: tuple[str, ...] = ("twin", "winner", "winter", "tina", "timer", "twitter")
DEFAULT_PREFIXES: tuple[str, ...] = ("", "hey ", "hallo ")
DEFAULT_FOLLOW_UPS: tuple[str, ...] = (
    "",
    " bitte hör zu",
    " ich habe eine frage",
    " kannst du mir helfen",
    " bist du da",
    " ich brauche hilfe",
)
DEFAULT_GENERIC_NEGATIVE_UTTERANCES: tuple[str, ...] = (
    "guten morgen",
    "gute nacht",
    "wie geht es dir",
    "danke schön",
    "ich warte hier",
    "kannst du mich hören",
    "ist jemand da",
    "ich komme später wieder",
    "bitte warte kurz",
    "ich brauche wasser",
    "ich möchte reden",
    "ich höre musik",
    "mach bitte leiser",
    "ich bin im wohnzimmer",
    "alles ist ruhig",
    "wann ist es so weit",
)
VALID_SYNTHETIC_LABELS: frozenset[str] = frozenset({"positive", "negative"})


@dataclass(frozen=True, slots=True)
class SyntheticStyleProfile:
    """Describe one style instruction family for Qwen3TTS synthesis."""

    key: str
    instruction: str | None


@dataclass(frozen=True, slots=True)
class SyntheticGenerationProfile:
    """Describe one sampling profile for synthetic speech generation."""

    key: str
    top_p: float
    temperature: float
    top_k: int
    repetition_penalty: float


@dataclass(frozen=True, slots=True)
class SyntheticAugmentationProfile:
    """Describe one lightweight channel degradation profile."""

    key: str
    gain_db: float = 0.0
    additive_noise_level: float = 0.0
    impulse_decay: float = 0.0
    impulse_length: int = 0
    lowpass_alpha: float | None = None
    highpass_alpha: float | None = None
    clip_ratio: float = 1.0
    downsample_factor: int = 1
    pre_silence_ms: int = 0


DEFAULT_STYLE_PROFILES: tuple[SyntheticStyleProfile, ...] = (
    SyntheticStyleProfile(key="plain", instruction=None),
    SyntheticStyleProfile(key="warm", instruction="Speak clearly, warmly, and naturally."),
    SyntheticStyleProfile(key="fast", instruction="Speak a little faster in a casual, natural way."),
    SyntheticStyleProfile(key="slow", instruction="Speak slightly slower with short natural hesitations."),
    SyntheticStyleProfile(key="soft", instruction="Speak softly and calmly as if the listener is a few steps away."),
    SyntheticStyleProfile(key="urgent", instruction="Speak with mild urgency but stay natural and intelligible."),
)

DEFAULT_GENERATION_PROFILES: tuple[SyntheticGenerationProfile, ...] = (
    SyntheticGenerationProfile(
        key="stable",
        top_p=0.88,
        temperature=0.55,
        top_k=40,
        repetition_penalty=1.03,
    ),
    SyntheticGenerationProfile(
        key="diverse",
        top_p=0.96,
        temperature=0.82,
        top_k=60,
        repetition_penalty=1.01,
    ),
)

DEFAULT_AUGMENTATION_PROFILES: tuple[SyntheticAugmentationProfile, ...] = (
    SyntheticAugmentationProfile(key="clean"),
    SyntheticAugmentationProfile(
        key="far_field",
        gain_db=-8.0,
        additive_noise_level=0.003,
        impulse_decay=0.85,
        impulse_length=9,
        lowpass_alpha=0.18,
        pre_silence_ms=120,
    ),
    SyntheticAugmentationProfile(
        key="room_noise",
        gain_db=-3.0,
        additive_noise_level=0.006,
        impulse_decay=0.7,
        impulse_length=7,
        pre_silence_ms=60,
    ),
    SyntheticAugmentationProfile(
        key="phone_band",
        gain_db=-5.0,
        additive_noise_level=0.002,
        lowpass_alpha=0.33,
        highpass_alpha=0.95,
        downsample_factor=2,
    ),
    SyntheticAugmentationProfile(
        key="clipped",
        gain_db=1.5,
        additive_noise_level=0.004,
        clip_ratio=0.72,
    ),
)

DEFAULT_SEEDS: tuple[int, ...] = (11, 23, 37)


@dataclass(frozen=True, slots=True)
class SyntheticWakewordRequest:
    """Describe one deterministic synthetic wakeword audio request."""

    utterance_id: str
    text: str
    label: str
    family_key: str
    split: str
    speaker: str
    style_key: str
    style_instruction: str | None
    generation_key: str
    top_p: float
    temperature: float
    top_k: int
    repetition_penalty: float
    seed: int
    augmentation_key: str
    output_rel_path: str


def render_wakeword_phrase(alias: str, *, prefix: str, suffix: str) -> str:
    """Render one canonical wakeword or confusion phrase."""

    normalized_prefix = str(prefix or "")
    normalized_alias = str(alias or "").strip()
    normalized_suffix = str(suffix or "")
    text = f"{normalized_prefix}{normalized_alias}{normalized_suffix}".strip()
    return " ".join(text.split())


def synthesize_phrase_inventory(
    *,
    aliases: tuple[str, ...],
    prefixes: tuple[str, ...] = DEFAULT_PREFIXES,
    follow_ups: tuple[str, ...] = DEFAULT_FOLLOW_UPS,
) -> tuple[tuple[str, str], ...]:
    """Expand aliases into many generic spoken utterance variants."""

    rows: list[tuple[str, str]] = []
    for alias in aliases:
        family_key = str(alias).strip().lower()
        for prefix in prefixes:
            for follow_up in follow_ups:
                rows.append(
                    (
                        family_key,
                        render_wakeword_phrase(alias, prefix=prefix, suffix=follow_up),
                    )
                )
    return tuple(rows)


def hash_to_split(utterance_id: str) -> str:
    """Map one deterministic utterance id to train/dev/test."""

    digest = hashlib.sha1(utterance_id.encode("utf-8")).digest()[0]
    ratio = float(digest) / 255.0
    if ratio < 0.82:
        return "train"
    if ratio < 0.91:
        return "dev"
    return "test"


def shard_speakers(
    speakers: Iterable[str],
    *,
    shard_index: int,
    shard_count: int,
) -> tuple[str, ...]:
    """Select one deterministic speaker shard."""

    normalized = tuple(str(item).strip().lower() for item in speakers if str(item).strip())
    if shard_count <= 0:
        raise ValueError("shard_count must be positive.")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must stay within the shard count.")
    return tuple(
        speaker
        for index, speaker in enumerate(normalized)
        if index % shard_count == shard_index
    )


def normalize_synthetic_labels(labels: Iterable[str] | None) -> tuple[str, ...]:
    """Normalize one synthetic-corpus label selection."""

    if labels is None:
        return ("positive", "negative")
    normalized: list[str] = []
    for item in labels:
        label = str(item).strip().lower()
        if not label or label == "all":
            return ("positive", "negative")
        if label not in VALID_SYNTHETIC_LABELS:
            raise ValueError(
                f"Unsupported synthetic label filter '{label}'. "
                f"Expected one of {sorted(VALID_SYNTHETIC_LABELS)} or 'all'."
            )
        if label not in normalized:
            normalized.append(label)
    return tuple(normalized) or ("positive", "negative")


def build_qwen3tts_synthetic_corpus_plan(
    *,
    speakers: tuple[str, ...] = QWEN3TTS_SUPPORTED_SPEAKERS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    style_profiles: tuple[SyntheticStyleProfile, ...] = DEFAULT_STYLE_PROFILES,
    generation_profiles: tuple[SyntheticGenerationProfile, ...] = DEFAULT_GENERATION_PROFILES,
    augmentation_profiles: tuple[SyntheticAugmentationProfile, ...] = DEFAULT_AUGMENTATION_PROFILES,
    labels: tuple[str, ...] = ("positive", "negative"),
    include_generic_negatives: bool = True,
    prefixes: tuple[str, ...] = DEFAULT_PREFIXES,
    follow_ups: tuple[str, ...] = DEFAULT_FOLLOW_UPS,
    generic_negative_utterances: tuple[str, ...] = DEFAULT_GENERIC_NEGATIVE_UTTERANCES,
    shard_index: int = 0,
    shard_count: int = 1,
) -> tuple[SyntheticWakewordRequest, ...]:
    """Build one deterministic synthetic corpus plan for Twinr wakewords."""

    selected_speakers = shard_speakers(
        speakers,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    selected_labels = normalize_synthetic_labels(labels)
    if not selected_speakers:
        return ()

    rows: list[tuple[str, str, str]] = []
    if "positive" in selected_labels:
        positive_rows = synthesize_phrase_inventory(
            aliases=WAKEWORD_FAMILY,
            prefixes=prefixes,
            follow_ups=follow_ups,
        )
        rows.extend(("positive", family_key, text) for family_key, text in positive_rows)
    if "negative" in selected_labels:
        confusion_rows = synthesize_phrase_inventory(
            aliases=CONFUSION_FAMILY,
            prefixes=prefixes,
            follow_ups=follow_ups,
        )
        rows.extend(("negative", family_key, text) for family_key, text in confusion_rows)
    if "negative" in selected_labels and include_generic_negatives:
        rows.extend(("negative", "generic", utterance) for utterance in generic_negative_utterances)

    requests: list[SyntheticWakewordRequest] = []
    for label, family_key, text in rows:
        for speaker in selected_speakers:
            for style in style_profiles:
                for generation in generation_profiles:
                    for seed in seeds:
                        for augmentation in augmentation_profiles:
                            utterance_id = (
                                f"{label}__{family_key}__{speaker}__{style.key}__"
                                f"{generation.key}__seed{seed}__{augmentation.key}"
                            )
                            digest = hashlib.sha1(f"{utterance_id}::{text}".encode("utf-8")).hexdigest()[:16]
                            utterance_key = f"{utterance_id}__{digest}"
                            split = hash_to_split(utterance_key)
                            requests.append(
                                SyntheticWakewordRequest(
                                    utterance_id=utterance_key,
                                    text=text,
                                    label=label,
                                    family_key=family_key,
                                    split=split,
                                    speaker=speaker,
                                    style_key=style.key,
                                    style_instruction=style.instruction,
                                    generation_key=generation.key,
                                    top_p=generation.top_p,
                                    temperature=generation.temperature,
                                    top_k=generation.top_k,
                                    repetition_penalty=generation.repetition_penalty,
                                    seed=seed,
                                    augmentation_key=augmentation.key,
                                    output_rel_path=f"audio/{split}/{utterance_key}.wav",
                                )
                            )
    return tuple(requests)


def _moving_average(samples: np.ndarray, alpha: float) -> np.ndarray:
    if samples.size == 0:
        return samples
    out = np.empty_like(samples)
    out[0] = samples[0]
    for index in range(1, samples.size):
        out[index] = alpha * samples[index] + (1.0 - alpha) * out[index - 1]
    return out


def _apply_lowpass(samples: np.ndarray, alpha: float | None) -> np.ndarray:
    if alpha is None:
        return samples
    return _moving_average(samples, float(alpha))


def _apply_highpass(samples: np.ndarray, alpha: float | None) -> np.ndarray:
    if alpha is None or samples.size == 0:
        return samples
    out = np.empty_like(samples)
    out[0] = samples[0]
    for index in range(1, samples.size):
        out[index] = alpha * (out[index - 1] + samples[index] - samples[index - 1])
    return out


def _apply_impulse_response(samples: np.ndarray, *, decay: float, length: int) -> np.ndarray:
    if samples.size == 0 or decay <= 0.0 or length <= 1:
        return samples
    kernel = np.power(float(decay), np.arange(length, dtype=np.float32))
    kernel /= max(1e-6, float(np.sum(kernel)))
    convolved = np.convolve(samples, kernel, mode="full")[: samples.size]
    return convolved.astype(np.float32, copy=False)


def _down_up_sample(samples: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1 or samples.size <= factor:
        return samples
    coarse = samples[::factor]
    source_axis = np.linspace(0.0, 1.0, coarse.size, dtype=np.float32)
    target_axis = np.linspace(0.0, 1.0, samples.size, dtype=np.float32)
    resampled = np.interp(target_axis, source_axis, coarse).astype(np.float32, copy=False)
    return resampled


def apply_augmentation(
    samples: np.ndarray,
    *,
    sample_rate: int,
    profile: SyntheticAugmentationProfile,
    seed: int,
) -> np.ndarray:
    """Apply one deterministic lightweight degradation profile."""

    if samples.ndim != 1:
        raise ValueError("samples must be one-dimensional mono audio.")
    out = samples.astype(np.float32, copy=True)
    if profile.pre_silence_ms > 0:
        pre_samples = int(round(float(sample_rate) * float(profile.pre_silence_ms) / 1000.0))
        out = np.concatenate([np.zeros(pre_samples, dtype=np.float32), out], axis=0)
    if profile.gain_db:
        out *= float(math.pow(10.0, profile.gain_db / 20.0))
    out = _apply_impulse_response(
        out,
        decay=float(profile.impulse_decay),
        length=int(profile.impulse_length),
    )
    out = _apply_lowpass(out, profile.lowpass_alpha)
    out = _apply_highpass(out, profile.highpass_alpha)
    out = _down_up_sample(out, int(profile.downsample_factor))
    if profile.additive_noise_level > 0.0:
        generator = np.random.default_rng(int(seed))
        out += generator.normal(
            loc=0.0,
            scale=float(profile.additive_noise_level),
            size=out.shape[0],
        ).astype(np.float32, copy=False)
    if 0.0 < profile.clip_ratio < 1.0:
        out = np.clip(out, -float(profile.clip_ratio), float(profile.clip_ratio))
    out = np.clip(out, -1.0, 1.0)
    return out.astype(np.float32, copy=False)


def float_audio_to_pcm16(samples: np.ndarray) -> np.ndarray:
    """Convert normalized float audio to little-endian PCM16."""

    clipped = np.clip(samples.astype(np.float32, copy=False), -1.0, 1.0)
    return np.round(clipped * 32767.0).astype("<i2")


def write_pcm16_wav(path: Path, *, samples: np.ndarray, sample_rate: int) -> None:
    """Write mono PCM16 audio to one WAV file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = float_audio_to_pcm16(samples)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm16.tobytes())


def manifest_row_for_request(
    request: SyntheticWakewordRequest,
    *,
    audio_path: Path,
) -> dict[str, object]:
    """Render one Twinr-compatible manifest row for a synthetic clip."""

    return {
        "utterance_id": request.utterance_id,
        "audio_path": str(audio_path),
        "label": request.label,
        "text": request.text,
        "family_key": request.family_key,
        "source": "qwen3tts_synthetic",
        "split": request.split,
        "speaker": request.speaker,
        "style_key": request.style_key,
        "style_instruction": request.style_instruction,
        "generation_key": request.generation_key,
        "top_p": request.top_p,
        "temperature": request.temperature,
        "top_k": request.top_k,
        "repetition_penalty": request.repetition_penalty,
        "seed": request.seed,
        "augmentation_key": request.augmentation_key,
    }


__all__ = [
    "CONFUSION_FAMILY",
    "DEFAULT_AUGMENTATION_PROFILES",
    "DEFAULT_FOLLOW_UPS",
    "DEFAULT_GENERATION_PROFILES",
    "DEFAULT_GENERIC_NEGATIVE_UTTERANCES",
    "DEFAULT_PREFIXES",
    "DEFAULT_SEEDS",
    "DEFAULT_STYLE_PROFILES",
    "QWEN3TTS_SUPPORTED_SPEAKERS",
    "SyntheticAugmentationProfile",
    "SyntheticGenerationProfile",
    "SyntheticStyleProfile",
    "SyntheticWakewordRequest",
    "VALID_SYNTHETIC_LABELS",
    "WAKEWORD_FAMILY",
    "apply_augmentation",
    "build_qwen3tts_synthetic_corpus_plan",
    "float_audio_to_pcm16",
    "hash_to_split",
    "manifest_row_for_request",
    "normalize_synthetic_labels",
    "render_wakeword_phrase",
    "shard_speakers",
    "synthesize_phrase_inventory",
    "write_pcm16_wav",
]
