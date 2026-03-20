"""Train and run Twinr's local second-stage wakeword cascade verifier.

This module owns the optional clip-level verifier that runs after a broad
openWakeWord detector candidate. It extracts aligned frame sequences from a
localized wakeword window and uses a sequence-aware verifier that compares the
temporal pattern against positive and confusing negative templates before a
small calibrated classifier makes the final accept/reject decision.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
import wave

import numpy as np
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from twinr.hardware.audio import AmbientAudioCaptureWindow

from .policy import WakewordVerification

_POSITIVE_LABELS = {"correct", "false_negative", "positive"}
_NEGATIVE_LABELS = {
    "false_positive",
    "background_tv",
    "cross_talk",
    "far_field",
    "noise",
    "negative",
}
_TARGET_SAMPLE_RATE = 16_000
_PREDICTION_PADDING_S = 1
_PREDICTION_CHUNK_SAMPLES = 1_280
_LEGACY_ASSET_VERSION = 1
_ASSET_VERSION = 2
_DEFAULT_DTW_BAND_RATIO = 0.2
_DTW_FALLBACK_DISTANCE = 8.0
_MAX_POSITIVE_TEMPLATES_PER_FAMILY = 2
_MAX_NEGATIVE_TEMPLATES_PER_FAMILY = 2
_MAX_BACKGROUND_NEGATIVE_TEMPLATES = 6
_MIN_FOCUS_NEGATIVE_FAMILIES = 4
_MAX_FOCUS_NEGATIVE_FAMILIES = 8
_POSITIVE_SAMPLE_WEIGHT = 2.1
_FOCUS_NEGATIVE_SAMPLE_WEIGHT = 2.0
_FOCUS_POSITIVE_SAMPLE_WEIGHT_BOOST = 1.85
_EXPLICIT_FOCUS_NEGATIVE_SAMPLE_WEIGHT_BOOST = 1.35
_FOCUS_POSITIVE_TEMPLATE_BONUS = 1
_FOCUS_NEGATIVE_TEMPLATE_BONUS = 1
_CANONICAL_FAMILY_ALIASES = {
    "twina": "twina",
    "twinna": "twinna",
    "twinner": "twinner",
    "twinr": "twinr",
    "twin": "twin",
    "winner": "winner",
    "winter": "winter",
    "tina": "tina",
    "timer": "timer",
    "twitter": "twitter",
}
_FOCUS_POSITIVE_FAMILIES = ("twinna", "twina", "twinner")
_FOCUS_POSITIVE_FAMILY_SET = frozenset(_FOCUS_POSITIVE_FAMILIES)
_EXPLICIT_FOCUS_NEGATIVE_FAMILIES = ("twin", "winner", "winter", "tina", "timer", "twitter")
_EXPLICIT_FOCUS_NEGATIVE_FAMILY_SET = frozenset(_EXPLICIT_FOCUS_NEGATIVE_FAMILIES)


@dataclass(frozen=True, slots=True)
class WakewordSequenceVerifierTrainingReport:
    """Describe one trained Twinr sequence verifier asset."""

    manifest_path: Path
    output_path: Path
    model_name: str
    auxiliary_models: tuple[str, ...]
    positive_clips: int
    negative_clips: int
    negative_seconds: float
    total_length_samples: int
    embedding_frames: int
    feature_dimensions: int


@dataclass(frozen=True, slots=True)
class _ManifestEntry:
    audio_path: Path
    label: str
    family_key: str


@dataclass(frozen=True, slots=True)
class _SequenceExample:
    index: int
    expected_detected: bool
    family_key: str
    sequence_matrix: np.ndarray
    score_summary: np.ndarray


@dataclass(frozen=True, slots=True)
class _SequenceTemplate:
    example_index: int
    family_key: str
    sequence_matrix: np.ndarray


def _normalize_label(value: object | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace(" ", "_")
    return normalized or None


def _family_tokens(value: object | None) -> tuple[str, ...]:
    """Split one phrase-like value into normalized alphanumeric tokens."""

    if value is None:
        return ()
    normalized = str(value).strip().lower()
    if not normalized:
        return ()
    tokens = []
    for raw_token in normalized.split():
        cleaned = "".join(character for character in raw_token if character.isalnum())
        if cleaned:
            tokens.append(cleaned)
    return tuple(tokens)


def _normalize_family_key(value: object | None) -> str:
    """Normalize one family key while preserving known Twinr confusion families.

    When a phrase contains a known Twinr wakeword or confusion token, use that
    canonical family rather than the trailing word. This keeps examples such as
    "Twinna wie ist das Wetter heute" grouped under the wakeword family instead
    of the sentence tail.
    """

    tokens = _family_tokens(value)
    if not tokens:
        return "unknown"
    for token in tokens:
        canonical = _CANONICAL_FAMILY_ALIASES.get(token)
        if canonical is not None:
            return canonical
    return "_".join(tokens) or "unknown"


def _manifest_family_key(payload: dict[str, object]) -> str:
    explicit_family = payload.get("family_key")
    if explicit_family is not None:
        return _normalize_family_key(explicit_family)
    for key in ("spoken_text", "text", "phrase"):
        candidate = payload.get(key)
        if candidate is not None:
            return _normalize_family_key(candidate)
    return "unknown"


def _load_manifest_entries(manifest_path: str | Path) -> list[_ManifestEntry]:
    """Load supported labeled capture entries from one JSONL or JSON-array manifest."""

    path = Path(manifest_path).expanduser().resolve(strict=False)
    if not path.exists():
        raise FileNotFoundError(path)
    raw_text = path.read_text(encoding="utf-8")
    stripped = raw_text.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        raw_entries = json.loads(stripped)
    else:
        raw_entries = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    entries: list[_ManifestEntry] = []
    for payload in raw_entries:
        if not isinstance(payload, dict):
            continue
        label = _normalize_label(payload.get("label"))
        if _expected_detected(label) is None:
            continue
        raw_path = str(payload.get("captured_audio_path") or payload.get("audio_path") or "").strip()
        if not raw_path:
            continue
        audio_path = Path(raw_path).expanduser()
        if not audio_path.is_absolute():
            audio_path = (path.parent / audio_path).resolve(strict=False)
        entries.append(
            _ManifestEntry(
                audio_path=audio_path,
                label=label or "",
                family_key=_manifest_family_key(payload),
            )
        )
    return entries


def _expected_detected(label: str | None) -> bool | None:
    normalized = _normalize_label(label)
    if normalized in _POSITIVE_LABELS:
        return True
    if normalized in _NEGATIVE_LABELS:
        return False
    return None


def _looks_like_local_model_path(value: str) -> bool:
    return bool(value) and (
        value.startswith(("~", ".", "/"))
        or "/" in value
        or "\\" in value
        or value.lower().endswith((".onnx", ".tflite", ".pb"))
    )


def _resolve_model_name(model_name: str) -> str:
    normalized = str(model_name or "").strip()
    if not normalized:
        raise ValueError("model_name must be a non-empty string.")
    if not _looks_like_local_model_path(normalized):
        return normalized
    candidate = Path(normalized).expanduser()
    if candidate.exists():
        return str(candidate.resolve(strict=False))
    return str(candidate.resolve(strict=False))


def _model_label(model_name: str) -> str:
    normalized = str(model_name or "").strip()
    if not normalized:
        raise ValueError("model_name must be a non-empty string.")
    if _looks_like_local_model_path(normalized):
        return Path(normalized).stem
    return normalized


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        frame_count = int(wav_file.getnframes())
        sample_rate = int(wav_file.getframerate())
    return frame_count / max(1, sample_rate)


def _load_pcm16_wav(path: Path, *, sample_rate: int = _TARGET_SAMPLE_RATE) -> np.ndarray:
    """Load one WAV file as mono PCM16 at the requested sample rate."""

    with wave.open(str(path), "rb") as wav_file:
        channels = int(wav_file.getnchannels())
        source_rate = int(wav_file.getframerate())
        sample_width = int(wav_file.getsampwidth())
        frame_count = int(wav_file.getnframes())
        pcm_bytes = wav_file.readframes(frame_count)
    if sample_width != 2:
        raise ValueError(f"{path} must be 16-bit PCM WAV.")
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
    if source_rate != sample_rate:
        samples = resample_poly(samples.astype(np.float32), sample_rate, source_rate)
        samples = np.clip(np.rint(samples), -32768, 32767).astype(np.int16)
    return samples


def _compute_total_length_samples(positive_paths: list[Path]) -> int:
    """Choose one deterministic verifier clip length from positive examples."""

    durations = [len(_load_pcm16_wav(path)) for path in positive_paths]
    median_samples = int(np.median(np.asarray(durations, dtype=np.int64)))
    total_length = int(round(median_samples / 1000.0) * 1000) + 12_000
    if total_length < 32_000:
        return 32_000
    if abs(total_length - 32_000) <= 4_000:
        return 32_000
    return total_length


def _prepare_fixed_clip(samples: np.ndarray, *, total_length_samples: int) -> np.ndarray:
    """Pad or trim one clip deterministically for the sequence verifier."""

    if samples.shape[0] >= total_length_samples:
        return samples[-total_length_samples:].astype(np.int16, copy=False)
    padded = np.zeros(total_length_samples, dtype=np.int16)
    start_index = total_length_samples - samples.shape[0]
    padded[start_index:start_index + samples.shape[0]] = samples
    return padded


def _resample_track(track: np.ndarray, *, target_length: int) -> np.ndarray:
    """Resize one 1-D score track onto a fixed number of frames."""

    if target_length <= 0:
        raise ValueError("target_length must be greater than 0.")
    if track.size == 0:
        return np.zeros(target_length, dtype=np.float32)
    if track.size == target_length:
        return track.astype(np.float32, copy=False)
    if track.size == 1:
        return np.full(target_length, float(track[0]), dtype=np.float32)
    source_positions = np.linspace(0.0, 1.0, num=track.size, dtype=np.float32)
    target_positions = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    return np.interp(target_positions, source_positions, track.astype(np.float32)).astype(np.float32)


def _resample_feature_frames(features: np.ndarray, *, target_frames: int) -> np.ndarray:
    """Resize one frame-by-dimension feature grid onto a fixed frame count."""

    if features.ndim != 2:
        raise ValueError("Sequence features must have shape (frames, dim).")
    if target_frames <= 0:
        raise ValueError("target_frames must be greater than 0.")
    if features.shape[0] == target_frames:
        return features.astype(np.float32, copy=False)
    if features.shape[0] == 0:
        return np.zeros((target_frames, features.shape[1]), dtype=np.float32)
    source_positions = np.linspace(0.0, 1.0, num=features.shape[0], dtype=np.float32)
    target_positions = np.linspace(0.0, 1.0, num=target_frames, dtype=np.float32)
    resized = np.empty((target_frames, features.shape[1]), dtype=np.float32)
    for column_index in range(features.shape[1]):
        resized[:, column_index] = np.interp(
            target_positions,
            source_positions,
            features[:, column_index].astype(np.float32),
        )
    return resized


def _prediction_track(
    prediction_frames: list[dict[str, float]] | tuple[dict[str, float], ...] | None,
    *,
    label: str,
) -> np.ndarray:
    if not prediction_frames:
        return np.zeros(0, dtype=np.float32)
    track = [float((frame or {}).get(label, 0.0)) for frame in prediction_frames]
    return np.asarray(track, dtype=np.float32)


def _build_sequence_matrix(
    *,
    embedding_frames: np.ndarray,
    score_tracks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Build one frame sequence plus score summary from embeddings and score tracks."""

    if embedding_frames.ndim != 2:
        raise ValueError("embedding_frames must have shape (frames, dim).")
    target_frames = embedding_frames.shape[0]
    aligned_tracks = [
        _resample_track(np.asarray(track, dtype=np.float32), target_length=target_frames)
        for track in score_tracks
    ]
    if aligned_tracks:
        track_matrix = np.stack(aligned_tracks, axis=1).astype(np.float32)
        track_deltas = np.diff(track_matrix, axis=0, prepend=track_matrix[:1]).astype(np.float32)
        sequence_matrix = np.concatenate(
            [embedding_frames.astype(np.float32), track_matrix, track_deltas],
            axis=1,
        ).astype(np.float32, copy=False)
        score_summary = np.concatenate(
            [
                track_matrix.max(axis=0),
                track_matrix.mean(axis=0),
                track_matrix.std(axis=0),
                track_deltas.mean(axis=0),
                track_deltas.std(axis=0),
            ],
            axis=0,
        ).astype(np.float32, copy=False)
    else:
        sequence_matrix = embedding_frames.astype(np.float32, copy=False)
        score_summary = np.zeros(0, dtype=np.float32)
    return sequence_matrix, score_summary


def _build_sequence_feature_vector(
    *,
    embedding_frames: np.ndarray,
    score_tracks: list[np.ndarray],
) -> np.ndarray:
    """Flatten aligned sequence features into one legacy classifier vector."""

    sequence_matrix, score_summary = _build_sequence_matrix(
        embedding_frames=embedding_frames,
        score_tracks=score_tracks,
    )
    flattened = sequence_matrix.reshape(-1).astype(np.float32, copy=False)
    return np.concatenate([flattened, score_summary], axis=0).astype(np.float32, copy=False)


def _capture_to_samples(capture: AmbientAudioCaptureWindow) -> np.ndarray:
    """Convert one capture window into mono 16 kHz PCM16 samples."""

    pcm_bytes = capture.pcm_bytes or b""
    if not pcm_bytes:
        return np.zeros(0, dtype=np.int16)
    usable = len(pcm_bytes) - (len(pcm_bytes) % 2)
    if usable <= 0:
        return np.zeros(0, dtype=np.int16)
    samples = np.frombuffer(pcm_bytes[:usable], dtype=np.int16)
    channels = max(1, int(capture.channels or 1))
    if channels > 1:
        usable_samples = len(samples) - (len(samples) % channels)
        if usable_samples <= 0:
            return np.zeros(0, dtype=np.int16)
        samples = samples[:usable_samples].reshape(-1, channels).astype(np.int32)
        samples = np.rint(np.mean(samples, axis=1)).astype(np.int16)
    sample_rate = max(1, int(capture.sample_rate or _TARGET_SAMPLE_RATE))
    if sample_rate != _TARGET_SAMPLE_RATE:
        samples = resample_poly(samples.astype(np.float32), _TARGET_SAMPLE_RATE, sample_rate)
        samples = np.clip(np.rint(samples), -32768, 32767).astype(np.int16)
    return samples


def _openwakeword_factory(*, model_name: str, inference_framework: str):
    from openwakeword.model import Model

    return Model(
        wakeword_models=[model_name],
        inference_framework=inference_framework,
    )


def _audio_features_factory(*, inference_framework: str):
    from openwakeword.utils import AudioFeatures

    return AudioFeatures(
        inference_framework=inference_framework,
        device="cpu",
        ncpu=1,
    )


def _sequence_feature_parts_for_clip(
    *,
    clip_samples: np.ndarray,
    target_frames: int,
    audio_features,
    primary_model,
    primary_label: str,
    auxiliary_model_objects: tuple[tuple[str, object], ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract aligned per-frame sequence features from one fixed-length clip."""

    embedding_frames = np.asarray(
        audio_features.embed_clips(np.expand_dims(clip_samples, axis=0), batch_size=1, ncpu=1)[0],
        dtype=np.float32,
    )
    resized_embeddings = _resample_feature_frames(embedding_frames, target_frames=target_frames)
    if hasattr(primary_model, "reset"):
        primary_model.reset()
    primary_predictions = primary_model.predict_clip(
        clip_samples,
        padding=_PREDICTION_PADDING_S,
        chunk_size=_PREDICTION_CHUNK_SAMPLES,
    )
    score_tracks = [_prediction_track(primary_predictions, label=primary_label)]
    for auxiliary_label, auxiliary_model in auxiliary_model_objects:
        if hasattr(auxiliary_model, "reset"):
            auxiliary_model.reset()
        auxiliary_predictions = auxiliary_model.predict_clip(
            clip_samples,
            padding=_PREDICTION_PADDING_S,
            chunk_size=_PREDICTION_CHUNK_SAMPLES,
        )
        score_tracks.append(_prediction_track(auxiliary_predictions, label=auxiliary_label))
    return _build_sequence_matrix(
        embedding_frames=resized_embeddings,
        score_tracks=score_tracks,
    )


def _sequence_normalizer(
    sequence_matrices: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate(sequence_matrices, axis=0).astype(np.float32, copy=False)
    frame_mean = stacked.mean(axis=0).astype(np.float32, copy=False)
    frame_std = stacked.std(axis=0).astype(np.float32, copy=False)
    frame_std = np.where(frame_std < 1e-6, 1.0, frame_std).astype(np.float32, copy=False)
    return frame_mean, frame_std


def _apply_sequence_normalizer(
    sequence_matrix: np.ndarray,
    *,
    frame_mean: np.ndarray,
    frame_std: np.ndarray,
) -> np.ndarray:
    return ((sequence_matrix.astype(np.float32) - frame_mean) / frame_std).astype(np.float32, copy=False)


def _dtw_band_radius(frame_count: int) -> int:
    return max(2, int(round(frame_count * _DEFAULT_DTW_BAND_RATIO)))


def _dtw_distance(sequence: np.ndarray, template: np.ndarray, *, band_radius: int) -> float:
    """Return one bounded DTW distance between two aligned frame sequences."""

    if sequence.ndim != 2 or template.ndim != 2:
        raise ValueError("sequence and template must have shape (frames, dim).")
    if sequence.shape[1] != template.shape[1]:
        raise ValueError("sequence and template must have the same feature dimension.")
    sequence_frames = int(sequence.shape[0])
    template_frames = int(template.shape[0])
    radius = max(abs(sequence_frames - template_frames), int(band_radius))
    previous = np.full(template_frames + 1, np.inf, dtype=np.float32)
    current = np.full(template_frames + 1, np.inf, dtype=np.float32)
    previous[0] = 0.0
    for sequence_index in range(1, sequence_frames + 1):
        current.fill(np.inf)
        target_frame = sequence[sequence_index - 1]
        start = max(1, sequence_index - radius)
        stop = min(template_frames, sequence_index + radius)
        for template_index in range(start, stop + 1):
            template_frame = template[template_index - 1]
            frame_cost = float(np.mean(np.square(target_frame - template_frame, dtype=np.float32)))
            current[template_index] = frame_cost + min(
                current[template_index - 1],
                previous[template_index],
                previous[template_index - 1],
            )
        previous, current = current, previous
    return float(previous[template_frames] / max(1.0, sequence_frames + template_frames))


def _sequence_groups(
    examples: list[_SequenceExample],
) -> dict[str, list[_SequenceExample]]:
    groups: dict[str, list[_SequenceExample]] = defaultdict(list)
    for example in examples:
        groups[example.family_key].append(example)
    return dict(groups)


def _select_template_examples(
    examples: list[_SequenceExample],
    *,
    max_templates: int,
    band_radius: int,
) -> list[_SequenceTemplate]:
    """Choose a small set of diverse medoid-like templates from one family."""

    if not examples:
        return []
    if len(examples) <= max_templates:
        return [
            _SequenceTemplate(
                example_index=example.index,
                family_key=example.family_key,
                sequence_matrix=example.sequence_matrix,
            )
            for example in examples
        ]
    pairwise = np.zeros((len(examples), len(examples)), dtype=np.float32)
    for left_index in range(len(examples)):
        for right_index in range(left_index + 1, len(examples)):
            distance = _dtw_distance(
                examples[left_index].sequence_matrix,
                examples[right_index].sequence_matrix,
                band_radius=band_radius,
            )
            pairwise[left_index, right_index] = distance
            pairwise[right_index, left_index] = distance
    selected = [int(np.argmin(pairwise.sum(axis=1)))]
    while len(selected) < max_templates:
        best_index = None
        best_distance = -1.0
        for candidate_index in range(len(examples)):
            if candidate_index in selected:
                continue
            distance_to_selected = min(pairwise[candidate_index, selected_index] for selected_index in selected)
            if distance_to_selected > best_distance:
                best_distance = float(distance_to_selected)
                best_index = candidate_index
        if best_index is None:
            break
        selected.append(best_index)
    return [
        _SequenceTemplate(
            example_index=examples[index].index,
            family_key=examples[index].family_key,
            sequence_matrix=examples[index].sequence_matrix,
        )
        for index in selected
    ]


def _focus_negative_families(
    negative_examples: list[_SequenceExample],
) -> tuple[str, ...]:
    """Pick the hardest negative families from stage-1 score evidence."""

    grouped_examples = _sequence_groups(negative_examples)
    family_stats: list[tuple[float, int, str]] = []
    for family_key, examples in grouped_examples.items():
        if not examples:
            continue
        hardness = float(
            np.mean(
                [
                    float(example.score_summary[0]) if example.score_summary.size else 0.0
                    for example in examples
                ],
                dtype=np.float32,
            )
        )
        family_stats.append((hardness, len(examples), family_key))
    family_stats.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    if not family_stats:
        return ()
    selected: list[str] = [
        family_key
        for family_key in _EXPLICIT_FOCUS_NEGATIVE_FAMILIES
        if family_key in grouped_examples
    ]
    hard_count = sum(1 for hardness, _, _ in family_stats if hardness >= 0.35)
    limit = min(
        _MAX_FOCUS_NEGATIVE_FAMILIES,
        max(_MIN_FOCUS_NEGATIVE_FAMILIES, hard_count or 0, len(selected)),
    )
    for _, _, family_key in family_stats:
        if family_key in selected:
            continue
        if len(selected) >= limit:
            break
        selected.append(family_key)
    return tuple(selected[:limit])


def _build_template_sets(
    positive_examples: list[_SequenceExample],
    negative_examples: list[_SequenceExample],
    *,
    band_radius: int,
) -> tuple[list[_SequenceTemplate], dict[str, list[_SequenceTemplate]], tuple[str, ...]]:
    positive_templates: list[_SequenceTemplate] = []
    for family_key, examples in _sequence_groups(positive_examples).items():
        positive_templates.extend(
            _select_template_examples(
                examples,
                max_templates=(
                    _MAX_POSITIVE_TEMPLATES_PER_FAMILY
                    + (_FOCUS_POSITIVE_TEMPLATE_BONUS if family_key in _FOCUS_POSITIVE_FAMILY_SET else 0)
                ),
                band_radius=band_radius,
            )
        )

    focus_negative_families = _focus_negative_families(negative_examples)
    grouped_negative_examples = _sequence_groups(negative_examples)
    negative_templates: dict[str, list[_SequenceTemplate]] = {}
    for family_key in focus_negative_families:
        templates = _select_template_examples(
            grouped_negative_examples.get(family_key, []),
            max_templates=(
                _MAX_NEGATIVE_TEMPLATES_PER_FAMILY
                + (_FOCUS_NEGATIVE_TEMPLATE_BONUS if family_key in _EXPLICIT_FOCUS_NEGATIVE_FAMILY_SET else 0)
            ),
            band_radius=band_radius,
        )
        if templates:
            negative_templates[family_key] = templates

    background_examples = [
        example
        for example in negative_examples
        if example.family_key not in set(focus_negative_families)
    ]
    if background_examples:
        ranked_background = sorted(
            background_examples,
            key=lambda example: float(example.score_summary[0]) if example.score_summary.size else 0.0,
            reverse=True,
        )
        negative_templates["__background__"] = _select_template_examples(
            ranked_background[: max(_MAX_BACKGROUND_NEGATIVE_TEMPLATES * 2, _MAX_BACKGROUND_NEGATIVE_TEMPLATES)],
            max_templates=_MAX_BACKGROUND_NEGATIVE_TEMPLATES,
            band_radius=band_radius,
        )
    return positive_templates, negative_templates, focus_negative_families


def _distance_summaries(distances: list[float]) -> tuple[float, float, float]:
    if not distances:
        return (_DTW_FALLBACK_DISTANCE, _DTW_FALLBACK_DISTANCE, 0.0)
    ordered = sorted(float(item) for item in distances)
    head = ordered[: min(2, len(ordered))]
    return (
        ordered[0],
        float(np.mean(np.asarray(head, dtype=np.float32))),
        float(np.std(np.asarray(ordered, dtype=np.float32))),
    )


def _template_distances(
    sequence_matrix: np.ndarray,
    templates: list[_SequenceTemplate],
    *,
    band_radius: int,
    exclude_example_index: int | None = None,
) -> list[float]:
    distances: list[float] = []
    for template in templates:
        if exclude_example_index is not None and template.example_index == exclude_example_index:
            continue
        distances.append(
            _dtw_distance(
                sequence_matrix,
                template.sequence_matrix,
                band_radius=band_radius,
            )
        )
    return distances


def _alignment_feature_vector(
    example: _SequenceExample,
    *,
    positive_templates: list[_SequenceTemplate],
    negative_templates: dict[str, list[_SequenceTemplate]],
    focus_negative_families: tuple[str, ...],
    band_radius: int,
    leave_one_out: bool,
) -> np.ndarray:
    exclude_index = example.index if leave_one_out else None
    positive_distances = _template_distances(
        example.sequence_matrix,
        positive_templates,
        band_radius=band_radius,
        exclude_example_index=exclude_index,
    )
    positive_min, positive_mean_top2, positive_std = _distance_summaries(positive_distances)
    background_distances = _template_distances(
        example.sequence_matrix,
        negative_templates.get("__background__", []),
        band_radius=band_radius,
        exclude_example_index=exclude_index,
    )
    background_min, background_mean_top2, background_std = _distance_summaries(background_distances)

    family_feature_values: list[float] = []
    focused_negative_distances: list[float] = []
    for family_key in focus_negative_families:
        distances = _template_distances(
            example.sequence_matrix,
            negative_templates.get(family_key, []),
            band_radius=band_radius,
            exclude_example_index=exclude_index,
        )
        family_min, family_mean_top2, family_std = _distance_summaries(distances)
        family_feature_values.extend(
            [
                family_min,
                family_mean_top2,
                family_std,
                family_min - positive_min,
            ]
        )
        focused_negative_distances.extend(distances)
    negative_min, negative_mean_top2, negative_std = _distance_summaries(
        focused_negative_distances if focused_negative_distances else background_distances
    )
    summary_vector = (
        example.score_summary.astype(np.float32, copy=False)
        if example.score_summary.size
        else np.zeros(0, dtype=np.float32)
    )
    base_vector = np.asarray(
        [
            positive_min,
            positive_mean_top2,
            positive_std,
            negative_min,
            negative_mean_top2,
            negative_std,
            negative_min - positive_min,
            negative_mean_top2 - positive_mean_top2,
            background_min,
            background_mean_top2,
            background_std,
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            base_vector,
            np.asarray(family_feature_values, dtype=np.float32),
            summary_vector,
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def _training_sample_weight(
    example: _SequenceExample,
    *,
    focus_negative_families: tuple[str, ...],
) -> float:
    primary_peak = float(example.score_summary[0]) if example.score_summary.size else 0.0
    if example.expected_detected:
        weight = _POSITIVE_SAMPLE_WEIGHT * (1.0 + (0.25 * primary_peak))
        if example.family_key in _FOCUS_POSITIVE_FAMILY_SET:
            return weight * _FOCUS_POSITIVE_SAMPLE_WEIGHT_BOOST
        return weight
    if example.family_key in set(focus_negative_families):
        weight = _FOCUS_NEGATIVE_SAMPLE_WEIGHT * (1.0 + primary_peak)
        if example.family_key in _EXPLICIT_FOCUS_NEGATIVE_FAMILY_SET:
            return weight * _EXPLICIT_FOCUS_NEGATIVE_SAMPLE_WEIGHT_BOOST
        return weight
    return 1.0 + (0.5 * primary_peak)


def train_wakeword_sequence_verifier_from_manifest(
    *,
    manifest_path: str | Path,
    output_path: str | Path,
    model_name: str,
    auxiliary_models: tuple[str, ...] = (),
    inference_framework: str = "onnx",
) -> WakewordSequenceVerifierTrainingReport:
    """Train one Twinr sequence-aware local verifier from room captures."""

    manifest = Path(manifest_path).expanduser().resolve(strict=False)
    entries = _load_manifest_entries(manifest)
    positive_entries = [entry for entry in entries if _expected_detected(entry.label) is True]
    negative_entries = [entry for entry in entries if _expected_detected(entry.label) is False]
    positive_paths = [entry.audio_path for entry in positive_entries]
    negative_paths = [entry.audio_path for entry in negative_entries]
    if len(positive_paths) < 3:
        raise ValueError("Sequence verifier training requires at least 3 positive clips.")
    negative_seconds = sum(_wav_duration_seconds(path) for path in negative_paths)
    if not negative_paths or negative_seconds < 10.0:
        raise ValueError("Sequence verifier training requires at least about 10 seconds of negative clips.")

    resolved_model_name = _resolve_model_name(model_name)
    resolved_auxiliary_models = tuple(_resolve_model_name(item) for item in auxiliary_models if str(item).strip())
    target_model_label = _model_label(resolved_model_name)
    auxiliary_labels = tuple(_model_label(item) for item in resolved_auxiliary_models)

    total_length_samples = _compute_total_length_samples(list(positive_paths))
    audio_features = _audio_features_factory(inference_framework=inference_framework)
    primary_model = _openwakeword_factory(model_name=resolved_model_name, inference_framework=inference_framework)
    auxiliary_model_objects = tuple(
        (
            label,
            _openwakeword_factory(model_name=aux_model_name, inference_framework=inference_framework),
        )
        for label, aux_model_name in zip(auxiliary_labels, resolved_auxiliary_models)
    )

    embedding_frames_count: int | None = None
    embedding_dim: int | None = None
    raw_examples: list[_SequenceExample] = []
    for example_index, entry in enumerate(entries):
        fixed_clip = _prepare_fixed_clip(
            _load_pcm16_wav(entry.audio_path),
            total_length_samples=total_length_samples,
        )
        base_embedding_frames = np.asarray(
            audio_features.embed_clips(np.expand_dims(fixed_clip, axis=0), batch_size=1, ncpu=1)[0],
            dtype=np.float32,
        )
        if embedding_frames_count is None:
            embedding_frames_count = int(base_embedding_frames.shape[0])
            embedding_dim = int(base_embedding_frames.shape[1])
        sequence_matrix, score_summary = _sequence_feature_parts_for_clip(
            clip_samples=fixed_clip,
            target_frames=int(embedding_frames_count or 0),
            audio_features=audio_features,
            primary_model=primary_model,
            primary_label=target_model_label,
            auxiliary_model_objects=auxiliary_model_objects,
        )
        raw_examples.append(
            _SequenceExample(
                index=example_index,
                expected_detected=bool(_expected_detected(entry.label)),
                family_key=entry.family_key,
                sequence_matrix=sequence_matrix,
                score_summary=score_summary,
            )
        )

    frame_mean, frame_std = _sequence_normalizer([example.sequence_matrix for example in raw_examples])
    normalized_examples = [
        _SequenceExample(
            index=example.index,
            expected_detected=example.expected_detected,
            family_key=example.family_key,
            sequence_matrix=_apply_sequence_normalizer(
                example.sequence_matrix,
                frame_mean=frame_mean,
                frame_std=frame_std,
            ),
            score_summary=example.score_summary,
        )
        for example in raw_examples
    ]

    positive_examples = [example for example in normalized_examples if example.expected_detected]
    negative_examples = [example for example in normalized_examples if not example.expected_detected]
    band_radius = _dtw_band_radius(int(embedding_frames_count or 0))
    positive_templates, negative_templates, focus_negative_families = _build_template_sets(
        positive_examples,
        negative_examples,
        band_radius=band_radius,
    )

    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    sample_weights: list[float] = []
    for example in normalized_examples:
        feature_rows.append(
            _alignment_feature_vector(
                example,
                positive_templates=positive_templates,
                negative_templates=negative_templates,
                focus_negative_families=focus_negative_families,
                band_radius=band_radius,
                leave_one_out=True,
            )
        )
        labels.append(1 if example.expected_detected else 0)
        sample_weights.append(
            _training_sample_weight(
                example,
                focus_negative_families=focus_negative_families,
            )
        )

    classifier = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=20260320,
                ),
            ),
        ]
    )
    feature_matrix = np.vstack(feature_rows).astype(np.float32, copy=False)
    label_array = np.asarray(labels, dtype=np.int64)
    classifier.fit(
        feature_matrix,
        label_array,
        classifier__sample_weight=np.asarray(sample_weights, dtype=np.float32),
    )

    resolved_output = Path(output_path).expanduser().resolve(strict=False)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": _ASSET_VERSION,
        "architecture": "dtw_margin_v1",
        "target_model_label": target_model_label,
        "primary_model_name": resolved_model_name,
        "auxiliary_models": resolved_auxiliary_models,
        "embedding_frames": int(embedding_frames_count or 0),
        "embedding_dim": int(embedding_dim or 0),
        "total_length_samples": int(total_length_samples),
        "inference_framework": str(inference_framework or "onnx").strip().lower() or "onnx",
        "frame_mean": frame_mean.astype(np.float32, copy=False),
        "frame_std": frame_std.astype(np.float32, copy=False),
        "positive_templates": [
            {
                "example_index": int(template.example_index),
                "family_key": template.family_key,
                "sequence_matrix": template.sequence_matrix.astype(np.float32, copy=False),
            }
            for template in positive_templates
        ],
        "negative_templates": {
            family_key: [
                {
                    "example_index": int(template.example_index),
                    "family_key": template.family_key,
                    "sequence_matrix": template.sequence_matrix.astype(np.float32, copy=False),
                }
                for template in templates
            ]
            for family_key, templates in negative_templates.items()
        },
        "focus_negative_families": focus_negative_families,
        "dtw_band_radius": int(band_radius),
        "classifier": classifier,
    }
    with resolved_output.open("wb") as output_file:
        pickle.dump(payload, output_file)

    return WakewordSequenceVerifierTrainingReport(
        manifest_path=manifest,
        output_path=resolved_output,
        model_name=resolved_model_name,
        auxiliary_models=resolved_auxiliary_models,
        positive_clips=len(positive_paths),
        negative_clips=len(negative_paths),
        negative_seconds=negative_seconds,
        total_length_samples=int(total_length_samples),
        embedding_frames=int(embedding_frames_count or 0),
        feature_dimensions=int(feature_matrix.shape[1]),
    )


class WakewordSequenceVerifier:
    """Load and score one trained Twinr sequence-aware verifier asset."""

    def __init__(self, *, payload: dict[str, object], asset_path: str | Path) -> None:
        self.asset_path = Path(asset_path).expanduser().resolve(strict=False)
        self.format_version = int(payload.get("format_version") or _LEGACY_ASSET_VERSION)
        self.architecture = str(payload.get("architecture") or "legacy_logistic").strip().lower()
        self.target_model_label = str(payload.get("target_model_label") or "").strip()
        self.primary_model_name = str(payload.get("primary_model_name") or "").strip()
        self.auxiliary_models = tuple(str(item).strip() for item in payload.get("auxiliary_models", ()) if str(item).strip())
        self.embedding_frames = int(payload.get("embedding_frames") or 0)
        self.embedding_dim = int(payload.get("embedding_dim") or 0)
        self.total_length_samples = int(payload.get("total_length_samples") or 0)
        self.inference_framework = str(payload.get("inference_framework") or "onnx").strip().lower() or "onnx"
        self.classifier = payload.get("classifier")
        self.frame_mean = None
        self.frame_std = None
        self.focus_negative_families: tuple[str, ...] = ()
        self.dtw_band_radius = 0
        self.positive_templates: list[_SequenceTemplate] = []
        self.negative_templates: dict[str, list[_SequenceTemplate]] = {}
        if not self.target_model_label:
            raise ValueError(f"{self.asset_path} is missing target_model_label.")
        if not self.primary_model_name:
            raise ValueError(f"{self.asset_path} is missing primary_model_name.")
        if self.embedding_frames <= 0 or self.embedding_dim <= 0:
            raise ValueError(f"{self.asset_path} is missing valid embedding dimensions.")
        if self.total_length_samples <= 0:
            raise ValueError(f"{self.asset_path} is missing total_length_samples.")
        if self.classifier is None:
            raise ValueError(f"{self.asset_path} is missing classifier.")
        if self.architecture == "dtw_margin_v1":
            self.frame_mean = np.asarray(payload.get("frame_mean"), dtype=np.float32)
            self.frame_std = np.asarray(payload.get("frame_std"), dtype=np.float32)
            if self.frame_mean.ndim != 1 or self.frame_std.ndim != 1 or self.frame_mean.shape != self.frame_std.shape:
                raise ValueError(f"{self.asset_path} is missing valid sequence normalizer arrays.")
            self.focus_negative_families = tuple(
                str(item).strip() for item in payload.get("focus_negative_families", ()) if str(item).strip()
            )
            self.dtw_band_radius = max(1, int(payload.get("dtw_band_radius") or 0))
            self.positive_templates = _load_template_payloads(payload.get("positive_templates"))
            negative_templates_payload = payload.get("negative_templates") or {}
            if not self.positive_templates or not isinstance(negative_templates_payload, dict):
                raise ValueError(f"{self.asset_path} is missing valid sequence templates.")
            self.negative_templates = {
                str(family_key).strip(): _load_template_payloads(raw_templates)
                for family_key, raw_templates in negative_templates_payload.items()
                if str(family_key).strip()
            }
            if self.dtw_band_radius <= 0:
                raise ValueError(f"{self.asset_path} is missing a valid dtw_band_radius.")
        elif self.architecture != "legacy_logistic":
            raise ValueError(f"{self.asset_path} has unsupported wakeword sequence verifier architecture.")
        self._audio_features = None
        self._primary_model = None
        self._auxiliary_model_objects: tuple[tuple[str, object], ...] | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> "WakewordSequenceVerifier":
        resolved = Path(path).expanduser().resolve(strict=False)
        with resolved.open("rb") as asset_file:
            payload = pickle.load(asset_file)
        if not isinstance(payload, dict):
            raise ValueError(f"{resolved} did not contain a supported sequence verifier asset.")
        format_version = int(payload.get("format_version") or _LEGACY_ASSET_VERSION)
        if format_version not in {_LEGACY_ASSET_VERSION, _ASSET_VERSION}:
            raise ValueError(f"{resolved} has unsupported wakeword sequence verifier format.")
        return cls(payload=payload, asset_path=resolved)

    def _get_audio_features(self):
        if self._audio_features is None:
            self._audio_features = _audio_features_factory(inference_framework=self.inference_framework)
        return self._audio_features

    def _get_primary_model(self):
        if self._primary_model is None:
            self._primary_model = _openwakeword_factory(
                model_name=self.primary_model_name,
                inference_framework=self.inference_framework,
            )
        return self._primary_model

    def _get_auxiliary_models(self) -> tuple[tuple[str, object], ...]:
        if self._auxiliary_model_objects is None:
            self._auxiliary_model_objects = tuple(
                (
                    _model_label(model_name),
                    _openwakeword_factory(
                        model_name=model_name,
                        inference_framework=self.inference_framework,
                    ),
                )
                for model_name in self.auxiliary_models
            )
        return self._auxiliary_model_objects

    def _sequence_parts_for_samples(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fixed_clip = _prepare_fixed_clip(samples, total_length_samples=self.total_length_samples)
        return _sequence_feature_parts_for_clip(
            clip_samples=fixed_clip,
            target_frames=self.embedding_frames,
            audio_features=self._get_audio_features(),
            primary_model=self._get_primary_model(),
            primary_label=self.target_model_label,
            auxiliary_model_objects=self._get_auxiliary_models(),
        )

    def score_capture(self, capture: AmbientAudioCaptureWindow) -> float:
        """Return one probability-like score for the supplied capture."""

        samples = _capture_to_samples(capture)
        if samples.size == 0:
            return 0.0
        sequence_matrix, score_summary = self._sequence_parts_for_samples(samples)
        if self.architecture == "legacy_logistic":
            feature_vector = np.concatenate(
                [sequence_matrix.reshape(-1).astype(np.float32, copy=False), score_summary],
                axis=0,
            ).astype(np.float32, copy=False)
        else:
            normalized_sequence = _apply_sequence_normalizer(
                sequence_matrix,
                frame_mean=self.frame_mean,
                frame_std=self.frame_std,
            )
            feature_vector = _alignment_feature_vector(
                _SequenceExample(
                    index=-1,
                    expected_detected=True,
                    family_key="runtime",
                    sequence_matrix=normalized_sequence,
                    score_summary=score_summary,
                ),
                positive_templates=self.positive_templates,
                negative_templates=self.negative_templates,
                focus_negative_families=self.focus_negative_families,
                band_radius=self.dtw_band_radius,
                leave_one_out=False,
            )
        probabilities = self.classifier.predict_proba(feature_vector.reshape(1, -1))
        return float(max(0.0, min(1.0, probabilities[0][1])))


def _load_template_payloads(raw_templates: object) -> list[_SequenceTemplate]:
    templates: list[_SequenceTemplate] = []
    if not isinstance(raw_templates, (list, tuple)):
        return templates
    for payload in raw_templates:
        if not isinstance(payload, dict):
            continue
        sequence_matrix = np.asarray(payload.get("sequence_matrix"), dtype=np.float32)
        if sequence_matrix.ndim != 2:
            continue
        templates.append(
            _SequenceTemplate(
                example_index=int(payload.get("example_index") or -1),
                family_key=_normalize_family_key(payload.get("family_key")),
                sequence_matrix=sequence_matrix.astype(np.float32, copy=False),
            )
        )
    return templates


class WakewordSequenceCaptureVerifier:
    """Verify localized wakeword captures with one or more sequence assets."""

    def __init__(
        self,
        *,
        verifier_models: dict[str, str],
        threshold: float = 0.5,
    ) -> None:
        self.threshold = max(0.0, min(float(threshold), 1.0))
        self._verifiers: dict[str, WakewordSequenceVerifier] = {}
        for model_label, asset_path in verifier_models.items():
            normalized_label = str(model_label or "").strip()
            normalized_path = str(asset_path or "").strip()
            if not normalized_label or not normalized_path:
                continue
            self._verifiers[normalized_label] = WakewordSequenceVerifier.from_path(normalized_path)

    def verify(
        self,
        capture: AmbientAudioCaptureWindow,
        *,
        detector_match,
    ) -> WakewordVerification:
        """Verify one wakeword capture against the configured sequence assets."""

        detector_label = str(getattr(detector_match, "detector_label", "") or "").strip()
        verifier = self._verifiers.get(detector_label)
        if verifier is None:
            return WakewordVerification(
                status="skipped",
                backend="local_sequence",
                reason="detector_label_unconfigured",
            )
        try:
            score = verifier.score_capture(capture)
        except Exception as exc:
            return WakewordVerification(
                status="error",
                backend="local_sequence",
                reason=f"{exc.__class__.__name__}",
            )
        if score >= self.threshold:
            return WakewordVerification(
                status="accepted",
                backend="local_sequence",
                reason=f"score:{score:.3f}",
            )
        return WakewordVerification(
            status="rejected",
            backend="local_sequence",
            reason=f"score:{score:.3f}",
        )


__all__ = [
    "WakewordSequenceCaptureVerifier",
    "WakewordSequenceVerifier",
    "WakewordSequenceVerifierTrainingReport",
    "train_wakeword_sequence_verifier_from_manifest",
]
