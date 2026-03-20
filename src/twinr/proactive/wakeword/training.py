"""Train reproducible wakeword base models against runtime-faithful acceptance.

This module owns the offline base-model training path for Twinr wakeword
detectors. It turns a generated wakeword dataset root with
``positive_train``/``negative_train``/``positive_test``/``negative_test``
directories into fixed-length embedding features, trains either Twinr's
scikit-learn MLP export path or the upstream ``openwakeword.train.Model``
loop, exports a local ONNX detector, and optionally tunes the operating
threshold against a labeled acceptance manifest using the same stream replay
path that Twinr runs on the Pi.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import random
from typing import Callable
import wave

import numpy as np
from scipy.signal import resample_poly

from twinr.agent.base_agent.config import TwinrConfig

from .evaluation import WakewordEvalMetrics, load_eval_manifest
from .matching import DEFAULT_WAKEWORD_PHRASES
from .promotion import evaluate_wakeword_stream_entries

_THRESHOLD_CANDIDATES = (
    0.00005,
    0.0001,
    0.0002,
    0.0003,
    0.0005,
    0.001,
    0.0015,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.08,
    0.12,
    0.2,
    0.3,
    0.5,
    0.7,
    0.85,
    0.93,
)

_POSITIVE_DIFFICULTY_PREFIXES = ("extra_pos_",)
_NEGATIVE_DIFFICULTY_PREFIXES = ("extra_neg_", "mined_neg_")


@dataclass(frozen=True, slots=True)
class WakewordBaseModelTrainingReport:
    """Describe one exported Twinr base-model training run."""

    dataset_root: Path
    output_model_path: Path
    metadata_path: Path
    working_dir: Path
    model_name: str
    total_length_samples: int
    train_positive_clips: int
    train_negative_clips: int
    validation_positive_clips: int
    validation_negative_clips: int
    training_rounds: int
    selected_threshold: float
    acceptance_metrics: WakewordEvalMetrics | None = None


def _load_pcm16_wav(path: Path, *, sample_rate: int = 16_000) -> np.ndarray:
    """Load one WAV file as mono 16-bit PCM at the requested sample rate."""

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
        samples = np.clip(samples, -32768.0, 32767.0).astype(np.int16)
    return samples


def _list_split_wavs(dataset_root: Path, split_name: str) -> list[Path]:
    """Resolve one generated wakeword split into a sorted list of WAV files."""

    split_root = dataset_root / split_name
    if not split_root.exists():
        raise FileNotFoundError(split_root)
    wavs = sorted(path for path in split_root.glob("*.wav") if path.is_file())
    if not wavs:
        raise ValueError(f"{split_root} did not contain any .wav files.")
    return wavs


def _compute_total_length_samples(positive_paths: list[Path]) -> int:
    """Choose one deterministic fixed clip length from positive examples."""

    durations = [len(_load_pcm16_wav(path)) for path in positive_paths]
    median_samples = int(np.median(np.asarray(durations, dtype=np.int64)))
    total_length = int(round(median_samples / 1000.0) * 1000) + 12_000
    if total_length < 32_000:
        return 32_000
    if abs(total_length - 32_000) <= 4_000:
        return 32_000
    return total_length


def _create_fixed_size_clip(
    samples: np.ndarray,
    *,
    total_length_samples: int,
    rng: random.Random,
    sample_rate: int = 16_000,
    max_end_jitter_seconds: float = 0.2,
) -> np.ndarray:
    """Pad or trim one clip so the wakeword lands near the end of the window."""

    if samples.shape[0] > total_length_samples:
        if rng.random() >= 0.5:
            return samples[:total_length_samples].astype(np.int16, copy=False)
        return samples[-total_length_samples:].astype(np.int16, copy=False)
    padded = np.zeros(total_length_samples, dtype=np.int16)
    max_end_jitter = int(rng.uniform(0.0, max_end_jitter_seconds) * sample_rate)
    start_index = max(0, total_length_samples - samples.shape[0] - max_end_jitter)
    padded[start_index:start_index + samples.shape[0]] = samples
    return padded


def _compute_feature_array(
    *,
    audio_paths: list[Path],
    total_length_samples: int,
    rounds: int,
    feature_extractor,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    """Convert a list of audio clips into one stacked openWakeWord feature array."""

    if not audio_paths:
        raise ValueError("At least one audio clip is required to compute wakeword features.")
    if rounds < 1:
        raise ValueError("rounds must be at least 1.")
    feature_batches: list[np.ndarray] = []
    batch: list[np.ndarray] = []
    for round_index in range(rounds):
        for audio_path in audio_paths:
            rng = random.Random(f"{seed}:{round_index}:{audio_path}")
            fixed_clip = _create_fixed_size_clip(
                _load_pcm16_wav(audio_path),
                total_length_samples=total_length_samples,
                rng=rng,
            )
            batch.append(fixed_clip)
            if len(batch) >= batch_size:
                feature_batches.append(
                    feature_extractor.embed_clips(
                        np.stack(batch, axis=0),
                        batch_size=min(batch_size, len(batch)),
                        ncpu=1,
                    )
                )
                batch.clear()
    if batch:
        feature_batches.append(
            feature_extractor.embed_clips(
                np.stack(batch, axis=0),
                batch_size=min(batch_size, len(batch)),
                ncpu=1,
            )
        )
    return np.vstack(feature_batches)


def _dataset_sample_weight(path: Path, *, positive: bool) -> float:
    """Assign one provenance-aware training weight for a dataset clip."""

    name = path.name.lower()
    if positive:
        if name.startswith("extra_pos_"):
            return 2.0
        return 1.0
    if name.startswith("mined_neg_"):
        return 6.0
    if name.startswith("extra_neg_"):
        return 3.0
    return 1.5


def _uses_difficulty_scoring(path: Path, *, positive: bool) -> bool:
    """Return whether one dataset clip should receive dynamic difficulty scoring."""

    name = path.name.lower()
    prefixes = _POSITIVE_DIFFICULTY_PREFIXES if positive else _NEGATIVE_DIFFICULTY_PREFIXES
    return any(name.startswith(prefix) for prefix in prefixes)


def _load_reference_peak_scores(
    *,
    audio_paths: list[Path],
    model_path: Path | None,
) -> dict[Path, float]:
    """Score the deployment-proximate dataset clips with one reference detector.

    Dynamic weighting should focus on the small subset of real room captures and
    mined confusions that most closely match the Pi deployment, not on thousands
    of synthetic clips that already act as broad regularization.
    """

    if model_path is None:
        return {}
    resolved_model_path = model_path.expanduser().resolve(strict=True)
    if not audio_paths:
        return {}

    from openwakeword.model import Model

    detector_key = resolved_model_path.stem
    model = Model(
        wakeword_models=[str(resolved_model_path)],
        inference_framework="onnx",
    )
    scores: dict[Path, float] = {}
    for audio_path in audio_paths:
        resolved_audio_path = audio_path.expanduser().resolve(strict=True)
        model.reset()
        predictions = model.predict_clip(str(resolved_audio_path), padding=1, chunk_size=1280)
        peak_score = max((float(frame.get(detector_key, 0.0)) for frame in predictions), default=0.0)
        scores[resolved_audio_path] = peak_score
    return scores


def _difficulty_weight_multiplier(
    *,
    score: float,
    positive: bool,
    difficulty_scale: float,
    difficulty_power: float,
) -> float:
    """Translate one reference detector score into a bounded sample multiplier."""

    normalized_score = min(1.0, max(0.0, float(score)))
    normalized_scale = max(0.0, float(difficulty_scale))
    normalized_power = max(1.0, float(difficulty_power))
    if normalized_scale <= 0.0:
        return 1.0
    if positive:
        difficulty = max(0.0, 1.0 - normalized_score)
    else:
        difficulty = normalized_score
    return 1.0 + normalized_scale * (difficulty ** normalized_power)


def _expanded_sample_weights(
    *,
    audio_paths: list[Path],
    rounds: int,
    positive: bool,
    difficulty_scores: dict[Path, float] | None = None,
    difficulty_scale: float = 0.0,
    difficulty_power: float = 2.0,
) -> np.ndarray:
    """Repeat one per-clip provenance weight to match feature-array order."""

    weights: list[float] = []
    for _round_index in range(max(1, int(rounds))):
        for audio_path in audio_paths:
            base_weight = _dataset_sample_weight(audio_path, positive=positive)
            resolved_audio_path = audio_path.expanduser().resolve(strict=False)
            difficulty_multiplier = 1.0
            if difficulty_scores is not None and resolved_audio_path in difficulty_scores:
                difficulty_multiplier = _difficulty_weight_multiplier(
                    score=float(difficulty_scores[resolved_audio_path]),
                    positive=positive,
                    difficulty_scale=float(difficulty_scale),
                    difficulty_power=float(difficulty_power),
                )
            weights.append(base_weight * difficulty_multiplier)
    return np.asarray(weights, dtype=np.float32)


def _flatten_false_positive_validation_features(feature_array: np.ndarray) -> np.ndarray:
    """Collapse per-clip feature tensors into one continuous false-positive stream."""

    if feature_array.ndim != 3 or feature_array.shape[0] == 0:
        raise ValueError("Wakeword validation features must have shape (clips, frames, dim).")
    return np.vstack(feature_array)


def _default_evaluation_config(model_path: Path) -> TwinrConfig:
    """Build one minimal Twinr config for offline threshold search."""

    return TwinrConfig(
        wakeword_enabled=True,
        wakeword_primary_backend="openwakeword",
        wakeword_fallback_backend="stt",
        wakeword_verifier_mode="disabled",
        wakeword_phrases=DEFAULT_WAKEWORD_PHRASES,
        wakeword_openwakeword_models=(str(model_path),),
        wakeword_openwakeword_inference_framework="onnx",
        wakeword_openwakeword_threshold=0.5,
        wakeword_openwakeword_patience_frames=1,
        wakeword_openwakeword_activation_samples=1,
        wakeword_openwakeword_deactivation_threshold=0.0,
    )


def _threshold_selection_key(metrics: WakewordEvalMetrics) -> tuple[int, int, float, float]:
    """Rank candidate thresholds with false negatives as the primary blocker.

    Twinr promotion guards fail closed on new false negatives. Threshold search
    must therefore minimize false negatives first, then false positives, and
    only then prefer stronger precision/recall ratios among equally safe
    candidates.
    """

    return (
        int(metrics.false_negative),
        int(metrics.false_positive),
        -float(metrics.precision),
        -float(metrics.recall),
    )


def _select_acceptance_threshold(
    *,
    model_path: Path,
    manifest_path: Path,
    evaluation_config: TwinrConfig | None,
) -> tuple[float, WakewordEvalMetrics]:
    """Search one operating threshold against a runtime-faithful manifest replay."""

    entries = load_eval_manifest(manifest_path)
    if not entries:
        raise ValueError(f"{manifest_path} did not contain any labeled wakeword clips.")
    base_config = evaluation_config or _default_evaluation_config(model_path)
    best_threshold = float(base_config.wakeword_openwakeword_threshold)
    best_metrics: WakewordEvalMetrics | None = None
    best_key: tuple[int, int, float, float] | None = None
    for threshold in _THRESHOLD_CANDIDATES:
        candidate_config = replace(
            base_config,
            wakeword_primary_backend="openwakeword",
            wakeword_verifier_mode="disabled",
            wakeword_openwakeword_models=(str(model_path),),
            wakeword_openwakeword_inference_framework="onnx",
            wakeword_openwakeword_threshold=threshold,
        )
        report = evaluate_wakeword_stream_entries(config=candidate_config, entries=entries)
        if (
            best_metrics is None
            or best_key is None
            or _threshold_selection_key(report.metrics) < best_key
        ):
            best_threshold = threshold
            best_metrics = report.metrics
            best_key = _threshold_selection_key(report.metrics)
    assert best_metrics is not None
    return best_threshold, best_metrics


def _export_sklearn_mlp_to_onnx(
    *,
    scaler,
    classifier,
    output_model_path: Path,
    model_name: str,
) -> None:
    """Export one fitted StandardScaler + MLP stack to Twinr's ONNX layout."""

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    if len(classifier.coefs_) != 3 or len(classifier.intercepts_) != 3:
        raise ValueError("Twinr MLP export currently requires exactly two hidden layers and one output layer.")

    flatten_shape = np.asarray([0, -1], dtype=np.int64)
    mean = np.asarray(scaler.mean_, dtype=np.float32)
    scale = np.asarray(scaler.scale_, dtype=np.float32)
    if mean.shape != scale.shape:
        raise ValueError("StandardScaler mean_ and scale_ must share one flattened feature shape.")

    initializers = [
        numpy_helper.from_array(flatten_shape, name="flatten_shape"),
        numpy_helper.from_array(mean, name="feature_mean"),
        numpy_helper.from_array(scale, name="feature_scale"),
        numpy_helper.from_array(np.asarray(classifier.coefs_[0], dtype=np.float32), name="layer_1_weights"),
        numpy_helper.from_array(np.asarray(classifier.intercepts_[0], dtype=np.float32), name="layer_1_bias"),
        numpy_helper.from_array(np.asarray(classifier.coefs_[1], dtype=np.float32), name="layer_2_weights"),
        numpy_helper.from_array(np.asarray(classifier.intercepts_[1], dtype=np.float32), name="layer_2_bias"),
        numpy_helper.from_array(np.asarray(classifier.coefs_[2], dtype=np.float32), name="layer_3_weights"),
        numpy_helper.from_array(np.asarray(classifier.intercepts_[2], dtype=np.float32), name="layer_3_bias"),
    ]

    nodes = [
        helper.make_node("Reshape", ["input", "flatten_shape"], ["flattened_features"], name="reshape_input"),
        helper.make_node("Sub", ["flattened_features", "feature_mean"], ["centered_features"], name="center_features"),
        helper.make_node("Div", ["centered_features", "feature_scale"], ["scaled_features"], name="scale_features"),
        helper.make_node("Gemm", ["scaled_features", "layer_1_weights", "layer_1_bias"], ["layer_1_linear"], name="layer_1_gemm"),
        helper.make_node("Relu", ["layer_1_linear"], ["layer_1_relu"], name="layer_1_relu_node"),
        helper.make_node("Gemm", ["layer_1_relu", "layer_2_weights", "layer_2_bias"], ["layer_2_linear"], name="layer_2_gemm"),
        helper.make_node("Relu", ["layer_2_linear"], ["layer_2_relu"], name="layer_2_relu_node"),
        helper.make_node("Gemm", ["layer_2_relu", "layer_3_weights", "layer_3_bias"], ["layer_3_linear"], name="layer_3_gemm"),
        helper.make_node("Sigmoid", ["layer_3_linear"], [model_name], name="output_probability"),
    ]
    graph = helper.make_graph(
        nodes,
        name=model_name,
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 16, 96])],
        outputs=[helper.make_tensor_value_info(model_name, TensorProto.FLOAT, [None, 1])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
        producer_name="twinr",
    )
    onnx.checker.check_model(model)
    onnx.save_model(model, str(output_model_path), save_as_external_data=False)


def _train_sklearn_mlp_model(
    *,
    positive_train_features_path: Path,
    negative_train_features_path: Path,
    positive_validation_features_path: Path,
    negative_validation_features_path: Path,
    output_model_path: Path,
    model_name: str,
    layer_dim: int,
    steps: int,
    seed: int,
    positive_train_weights: np.ndarray | None = None,
    negative_train_weights: np.ndarray | None = None,
) -> None:
    """Train Twinr's production-aligned StandardScaler + MLP wakeword model."""

    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    positive_train = np.load(positive_train_features_path).astype(np.float32)
    negative_train = np.load(negative_train_features_path).astype(np.float32)
    positive_validation = np.load(positive_validation_features_path).astype(np.float32)
    negative_validation = np.load(negative_validation_features_path).astype(np.float32)

    train_features = np.vstack((positive_train, negative_train))
    train_labels = np.hstack(
        (
            np.ones(positive_train.shape[0], dtype=np.int64),
            np.zeros(negative_train.shape[0], dtype=np.int64),
        )
    )
    sample_weight = None
    if positive_train_weights is not None or negative_train_weights is not None:
        resolved_positive_weights = (
            positive_train_weights
            if positive_train_weights is not None
            else np.ones(positive_train.shape[0], dtype=np.float32)
        )
        resolved_negative_weights = (
            negative_train_weights
            if negative_train_weights is not None
            else np.ones(negative_train.shape[0], dtype=np.float32)
        )
        if resolved_positive_weights.shape[0] != positive_train.shape[0]:
            raise ValueError("positive_train_weights must match positive_train feature count.")
        if resolved_negative_weights.shape[0] != negative_train.shape[0]:
            raise ValueError("negative_train_weights must match negative_train feature count.")
        sample_weight = np.hstack((resolved_positive_weights, resolved_negative_weights)).astype(np.float32)
    validation_features = np.vstack((positive_validation, negative_validation))
    validation_labels = np.hstack(
        (
            np.ones(positive_validation.shape[0], dtype=np.int64),
            np.zeros(negative_validation.shape[0], dtype=np.int64),
        )
    )

    flattened_train = train_features.reshape(train_features.shape[0], -1)
    flattened_validation = validation_features.reshape(validation_features.shape[0], -1)
    scaler = StandardScaler()
    scaler.fit(flattened_train)
    classifier = MLPClassifier(
        hidden_layer_sizes=(max(32, int(layer_dim)), max(16, int(layer_dim) // 4)),
        activation="relu",
        solver="adam",
        batch_size=min(256, max(32, flattened_train.shape[0] // 4 or 1)),
        learning_rate_init=0.001,
        max_iter=max(50, int(round(max(1, int(steps)) / 100.0))),
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=12,
        random_state=int(seed),
    )
    classifier.fit(scaler.transform(flattened_train), train_labels, sample_weight=sample_weight)

    validation_predictions = classifier.predict(scaler.transform(flattened_validation))
    if validation_predictions.shape[0] != validation_labels.shape[0]:
        raise RuntimeError("MLP validation output shape did not match validation labels.")

    _export_sklearn_mlp_to_onnx(
        scaler=scaler,
        classifier=classifier,
        output_model_path=output_model_path,
        model_name=model_name,
    )


def _train_openwakeword_model(
    *,
    positive_train_features_path: Path,
    negative_train_features_path: Path,
    positive_validation_features_path: Path,
    negative_validation_features_path: Path,
    false_positive_validation_features_path: Path,
    output_model_path: Path,
    model_name: str,
    model_type: str,
    layer_dim: int,
    steps: int,
    positive_per_batch: int,
    negative_per_batch: int,
    max_negative_weight: int,
    target_false_positives_per_hour: float,
) -> None:
    """Run the upstream openWakeWord training loop and export one ONNX model."""

    import torch
    from openwakeword.data import mmap_batch_generator
    from openwakeword.train import Model as OpenWakeWordTrainingModel

    input_shape = tuple(np.load(positive_validation_features_path, mmap_mode="r").shape[1:])
    if len(input_shape) != 2:
        raise ValueError("Wakeword positive validation features must have shape (clips, frames, dim).")

    label_transforms = {
        "positive": lambda labels: [1 for _ in labels],
        "adversarial_negative": lambda labels: [0 for _ in labels],
    }
    batch_generator = mmap_batch_generator(
        data_files={
            "positive": str(positive_train_features_path),
            "adversarial_negative": str(negative_train_features_path),
        },
        n_per_class={
            "positive": max(1, int(positive_per_batch)),
            "adversarial_negative": max(1, int(negative_per_batch)),
        },
        label_transform_funcs=label_transforms,
    )

    class IterDataset(torch.utils.data.IterableDataset):
        """Wrap the mmap batch generator for the upstream training loop."""

        def __init__(self, generator) -> None:
            super().__init__()
            self._generator = generator

        def __iter__(self):
            return self._generator

    train_loader = torch.utils.data.DataLoader(
        IterDataset(batch_generator),
        batch_size=None,
        num_workers=0,
    )

    false_positive_features = np.load(false_positive_validation_features_path)
    if false_positive_features.ndim != 2:
        raise ValueError("False-positive validation features must have shape (frames, dim).")
    fp_window_count = max(1, false_positive_features.shape[0] - input_shape[0] + 1)
    fp_windows = np.array(
        [false_positive_features[index:index + input_shape[0]] for index in range(fp_window_count)],
        dtype=np.float32,
    )
    fp_labels = np.zeros(fp_windows.shape[0], dtype=np.float32)
    false_positive_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(fp_windows), torch.from_numpy(fp_labels)),
        batch_size=len(fp_labels),
    )

    positive_validation = np.load(positive_validation_features_path)
    negative_validation = np.load(negative_validation_features_path)
    validation_features = np.vstack((positive_validation, negative_validation)).astype(np.float32)
    validation_labels = np.hstack(
        (
            np.ones(positive_validation.shape[0], dtype=np.float32),
            np.zeros(negative_validation.shape[0], dtype=np.float32),
        )
    )
    validation_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(validation_features),
            torch.from_numpy(validation_labels),
        ),
        batch_size=len(validation_labels),
    )

    training_model = OpenWakeWordTrainingModel(
        n_classes=1,
        input_shape=input_shape,
        model_type=model_type,
        layer_dim=int(layer_dim),
        seconds_per_example=(1280 * input_shape[0]) / 16_000.0,
    )
    best_model = training_model.auto_train(
        X_train=train_loader,
        X_val=validation_loader,
        false_positive_val_data=false_positive_loader,
        steps=int(steps),
        max_negative_weight=int(max_negative_weight),
        target_fp_per_hour=float(target_false_positives_per_hour),
    )
    _export_openwakeword_model_to_onnx(
        model=best_model,
        input_shape=input_shape,
        output_model_path=output_model_path,
        model_name=model_name,
    )


def _export_openwakeword_model_to_onnx(
    *,
    model: object,
    input_shape: tuple[int, int],
    output_model_path: Path,
    model_name: str,
) -> None:
    """Export one trained model to a single-file ONNX artifact.

    Newer Torch releases default to the dynamo-based exporter and may emit
    external-data ONNX graphs, even for compact Twinr wakeword models. The Pi
    runtime expects a single-file local model asset, so the export path forces
    the legacy exporter and verifies that no unresolved sidecar dependency
    survives the write.
    """

    import onnx
    import torch

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = output_model_path.with_suffix(output_model_path.suffix + ".data")
    if sidecar_path.exists():
        sidecar_path.unlink()

    torch.onnx.export(
        model.to("cpu"),
        torch.rand(input_shape)[None, ],
        str(output_model_path),
        output_names=[model_name],
        opset_version=13,
        dynamo=False,
        external_data=False,
    )

    exported_model = onnx.load(str(output_model_path), load_external_data=False)
    has_external_tensors = any(
        initializer.data_location == onnx.TensorProto.EXTERNAL
        for initializer in exported_model.graph.initializer
    )
    if not has_external_tensors:
        return
    if not sidecar_path.exists():
        raise RuntimeError(
            "Wakeword ONNX export referenced external tensor data without writing the expected sidecar file."
        )

    consolidated_model = onnx.load(str(output_model_path), load_external_data=True)
    onnx.save_model(consolidated_model, str(output_model_path), save_as_external_data=False)
    sidecar_path.unlink(missing_ok=True)


def train_wakeword_base_model_from_dataset_root(
    *,
    dataset_root: str | Path,
    output_model_path: str | Path,
    metadata_path: str | Path | None = None,
    model_name: str | None = None,
    acceptance_manifest: str | Path | None = None,
    workdir: str | Path | None = None,
    training_rounds: int = 2,
    model_type: str = "mlp",
    layer_dim: int = 128,
    steps: int = 20_000,
    positive_per_batch: int = 96,
    negative_per_batch: int = 288,
    max_negative_weight: int = 1_200,
    target_false_positives_per_hour: float = 0.2,
    feature_device: str = "cpu",
    feature_batch_size: int = 64,
    seed: int = 20260319,
    difficulty_reference_model_path: str | Path | None = None,
    difficulty_positive_scale: float = 0.0,
    difficulty_negative_scale: float = 0.0,
    difficulty_power: float = 2.0,
    evaluation_config: TwinrConfig | None = None,
    train_backend: Callable[..., None] | None = None,
    acceptance_evaluator: Callable[..., tuple[float, WakewordEvalMetrics]] | None = None,
) -> WakewordBaseModelTrainingReport:
    """Train one local Twinr openWakeWord base model from a dataset root.

    The dataset root must follow the structure produced by
    ``scripts/generate_multivoice_dataset.py``:

    - ``positive_train/*.wav``
    - ``negative_train/*.wav``
    - ``positive_test/*.wav``
    - ``negative_test/*.wav``
    """

    dataset_root_path = Path(dataset_root).expanduser().resolve(strict=True)
    output_model = Path(output_model_path).expanduser().resolve(strict=False)
    metadata = (
        Path(metadata_path).expanduser().resolve(strict=False)
        if metadata_path is not None
        else output_model.with_suffix(".metadata.json")
    )
    model_stem = (model_name or output_model.stem or "wakeword_model").strip()
    working_dir = (
        Path(workdir).expanduser().resolve(strict=False)
        if workdir is not None
        else output_model.parent / f".{output_model.stem}_training"
    )
    working_dir.mkdir(parents=True, exist_ok=True)

    positive_train_paths = _list_split_wavs(dataset_root_path, "positive_train")
    negative_train_paths = _list_split_wavs(dataset_root_path, "negative_train")
    positive_validation_paths = _list_split_wavs(dataset_root_path, "positive_test")
    negative_validation_paths = _list_split_wavs(dataset_root_path, "negative_test")
    total_length_samples = _compute_total_length_samples(positive_train_paths + positive_validation_paths)

    from openwakeword.utils import AudioFeatures

    feature_extractor = AudioFeatures(
        inference_framework="onnx",
        device=feature_device,
        ncpu=1,
    )
    positive_train_features = _compute_feature_array(
        audio_paths=positive_train_paths,
        total_length_samples=total_length_samples,
        rounds=max(1, int(training_rounds)),
        feature_extractor=feature_extractor,
        batch_size=max(1, int(feature_batch_size)),
        seed=seed,
    )
    negative_train_features = _compute_feature_array(
        audio_paths=negative_train_paths,
        total_length_samples=total_length_samples,
        rounds=max(1, int(training_rounds)),
        feature_extractor=feature_extractor,
        batch_size=max(1, int(feature_batch_size)),
        seed=seed + 1,
    )
    positive_validation_features = _compute_feature_array(
        audio_paths=positive_validation_paths,
        total_length_samples=total_length_samples,
        rounds=1,
        feature_extractor=feature_extractor,
        batch_size=max(1, int(feature_batch_size)),
        seed=seed + 2,
    )
    negative_validation_features = _compute_feature_array(
        audio_paths=negative_validation_paths,
        total_length_samples=total_length_samples,
        rounds=1,
        feature_extractor=feature_extractor,
        batch_size=max(1, int(feature_batch_size)),
        seed=seed + 3,
    )

    positive_train_features_path = working_dir / "positive_features_train.npy"
    negative_train_features_path = working_dir / "negative_features_train.npy"
    positive_validation_features_path = working_dir / "positive_features_validation.npy"
    negative_validation_features_path = working_dir / "negative_features_validation.npy"
    false_positive_validation_features_path = working_dir / "false_positive_validation_features.npy"
    np.save(positive_train_features_path, positive_train_features)
    np.save(negative_train_features_path, negative_train_features)
    np.save(positive_validation_features_path, positive_validation_features)
    np.save(negative_validation_features_path, negative_validation_features)
    np.save(
        false_positive_validation_features_path,
        _flatten_false_positive_validation_features(negative_validation_features),
    )

    resolved_model_type = str(model_type or "mlp").strip().lower() or "mlp"
    difficulty_reference_model = (
        Path(difficulty_reference_model_path).expanduser().resolve(strict=True)
        if difficulty_reference_model_path is not None
        else None
    )
    positive_difficulty_scores = _load_reference_peak_scores(
        audio_paths=[path for path in positive_train_paths if _uses_difficulty_scoring(path, positive=True)],
        model_path=difficulty_reference_model,
    )
    negative_difficulty_scores = _load_reference_peak_scores(
        audio_paths=[path for path in negative_train_paths if _uses_difficulty_scoring(path, positive=False)],
        model_path=difficulty_reference_model,
    )
    if train_backend is not None:
        train_backend(
            positive_train_features_path=positive_train_features_path,
            negative_train_features_path=negative_train_features_path,
            positive_validation_features_path=positive_validation_features_path,
            negative_validation_features_path=negative_validation_features_path,
            false_positive_validation_features_path=false_positive_validation_features_path,
            output_model_path=output_model,
            model_name=model_stem,
            model_type=resolved_model_type,
            layer_dim=int(layer_dim),
            steps=int(steps),
            positive_per_batch=int(positive_per_batch),
            negative_per_batch=int(negative_per_batch),
            max_negative_weight=int(max_negative_weight),
            target_false_positives_per_hour=float(target_false_positives_per_hour),
        )
    elif resolved_model_type == "mlp":
        _train_sklearn_mlp_model(
            positive_train_features_path=positive_train_features_path,
            negative_train_features_path=negative_train_features_path,
            positive_validation_features_path=positive_validation_features_path,
            negative_validation_features_path=negative_validation_features_path,
            output_model_path=output_model,
            model_name=model_stem,
            layer_dim=int(layer_dim),
            steps=int(steps),
            seed=int(seed),
            positive_train_weights=_expanded_sample_weights(
                audio_paths=positive_train_paths,
                rounds=max(1, int(training_rounds)),
                positive=True,
                difficulty_scores=positive_difficulty_scores,
                difficulty_scale=float(difficulty_positive_scale),
                difficulty_power=float(difficulty_power),
            ),
            negative_train_weights=_expanded_sample_weights(
                audio_paths=negative_train_paths,
                rounds=max(1, int(training_rounds)),
                positive=False,
                difficulty_scores=negative_difficulty_scores,
                difficulty_scale=float(difficulty_negative_scale),
                difficulty_power=float(difficulty_power),
            ),
        )
    else:
        _train_openwakeword_model(
            positive_train_features_path=positive_train_features_path,
            negative_train_features_path=negative_train_features_path,
            positive_validation_features_path=positive_validation_features_path,
            negative_validation_features_path=negative_validation_features_path,
            false_positive_validation_features_path=false_positive_validation_features_path,
            output_model_path=output_model,
            model_name=model_stem,
            model_type=resolved_model_type,
            layer_dim=int(layer_dim),
            steps=int(steps),
            positive_per_batch=int(positive_per_batch),
            negative_per_batch=int(negative_per_batch),
            max_negative_weight=int(max_negative_weight),
            target_false_positives_per_hour=float(target_false_positives_per_hour),
        )

    selected_threshold = (
        float(evaluation_config.wakeword_openwakeword_threshold)
        if evaluation_config is not None
        else 0.5
    )
    acceptance_metrics: WakewordEvalMetrics | None = None
    if acceptance_manifest is not None:
        threshold_search = acceptance_evaluator or _select_acceptance_threshold
        selected_threshold, acceptance_metrics = threshold_search(
            model_path=output_model,
            manifest_path=Path(acceptance_manifest).expanduser().resolve(strict=True),
            evaluation_config=evaluation_config,
        )

    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata_payload: dict[str, object] = {
        "model_path": str(output_model),
        "metadata_path": str(metadata),
        "dataset_root": str(dataset_root_path),
        "working_dir": str(working_dir),
        "model_name": model_stem,
        "model_type": resolved_model_type,
        "layer_dim": int(layer_dim),
        "steps": int(steps),
        "training_rounds": int(training_rounds),
        "feature_device": feature_device,
        "feature_batch_size": int(feature_batch_size),
        "seed": int(seed),
        "difficulty_reference_model_path": (
            str(difficulty_reference_model)
            if difficulty_reference_model is not None
            else None
        ),
        "difficulty_positive_scale": float(difficulty_positive_scale),
        "difficulty_negative_scale": float(difficulty_negative_scale),
        "difficulty_power": float(difficulty_power),
        "total_length_samples": int(total_length_samples),
        "train_positive_clips": len(positive_train_paths),
        "train_negative_clips": len(negative_train_paths),
        "validation_positive_clips": len(positive_validation_paths),
        "validation_negative_clips": len(negative_validation_paths),
        "selected_threshold": float(selected_threshold),
        "acceptance_manifest": (
            str(Path(acceptance_manifest).expanduser().resolve(strict=False))
            if acceptance_manifest is not None
            else None
        ),
        "acceptance_eval_mode": "runtime_stream_replay" if acceptance_manifest is not None else None,
    }
    if acceptance_metrics is not None:
        metadata_payload["acceptance_metrics"] = {
            "total": acceptance_metrics.total,
            "true_positive": acceptance_metrics.true_positive,
            "false_positive": acceptance_metrics.false_positive,
            "true_negative": acceptance_metrics.true_negative,
            "false_negative": acceptance_metrics.false_negative,
            "precision": round(acceptance_metrics.precision, 6),
            "recall": round(acceptance_metrics.recall, 6),
            "false_positive_rate": round(acceptance_metrics.false_positive_rate, 6),
            "false_negative_rate": round(acceptance_metrics.false_negative_rate, 6),
        }
    metadata.write_text(
        json.dumps(metadata_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return WakewordBaseModelTrainingReport(
        dataset_root=dataset_root_path,
        output_model_path=output_model,
        metadata_path=metadata,
        working_dir=working_dir,
        model_name=model_stem,
        total_length_samples=total_length_samples,
        train_positive_clips=len(positive_train_paths),
        train_negative_clips=len(negative_train_paths),
        validation_positive_clips=len(positive_validation_paths),
        validation_negative_clips=len(negative_validation_paths),
        training_rounds=int(training_rounds),
        selected_threshold=float(selected_threshold),
        acceptance_metrics=acceptance_metrics,
    )


__all__ = [
    "WakewordBaseModelTrainingReport",
    "train_wakeword_base_model_from_dataset_root",
]
