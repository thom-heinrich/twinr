"""Train reproducible openWakeWord-compatible Twinr base models.

This module owns the offline base-model training path for Twinr wakeword
detectors. It turns a generated wakeword dataset root with
``positive_train``/``negative_train``/``positive_test``/``negative_test``
directories into fixed-length embedding features, runs the upstream
``openwakeword.train.Model`` training loop, exports a local ONNX detector, and
optionally tunes the operating threshold against a labeled acceptance manifest.
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

from .evaluation import WakewordEvalMetrics, evaluate_wakeword_entries, load_eval_manifest
from .matching import DEFAULT_WAKEWORD_PHRASES

_THRESHOLD_CANDIDATES = (
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


def _score_acceptance_metrics(metrics: WakewordEvalMetrics) -> float:
    """Convert wakeword acceptance metrics into one threshold-selection score."""

    score = (metrics.precision * 0.6) + (metrics.recall * 0.4)
    score -= metrics.false_positive_rate
    return score


def _select_acceptance_threshold(
    *,
    model_path: Path,
    manifest_path: Path,
    evaluation_config: TwinrConfig | None,
) -> tuple[float, WakewordEvalMetrics]:
    """Search one operating threshold against a labeled acceptance manifest."""

    entries = load_eval_manifest(manifest_path)
    if not entries:
        raise ValueError(f"{manifest_path} did not contain any labeled wakeword clips.")
    base_config = evaluation_config or _default_evaluation_config(model_path)
    best_threshold = float(base_config.wakeword_openwakeword_threshold)
    best_metrics: WakewordEvalMetrics | None = None
    best_score = float("-inf")
    for threshold in _THRESHOLD_CANDIDATES:
        candidate_config = replace(
            base_config,
            wakeword_primary_backend="openwakeword",
            wakeword_verifier_mode="disabled",
            wakeword_openwakeword_models=(str(model_path),),
            wakeword_openwakeword_inference_framework="onnx",
            wakeword_openwakeword_threshold=threshold,
        )
        report = evaluate_wakeword_entries(config=candidate_config, entries=entries)
        candidate_score = _score_acceptance_metrics(report.metrics)
        if (
            best_metrics is None
            or candidate_score > best_score
            or (
                candidate_score == best_score
                and report.metrics.false_positive_rate < best_metrics.false_positive_rate
            )
        ):
            best_threshold = threshold
            best_metrics = report.metrics
            best_score = candidate_score
    assert best_metrics is not None
    return best_threshold, best_metrics


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
    model_type: str = "dnn",
    layer_dim: int = 128,
    steps: int = 20_000,
    positive_per_batch: int = 96,
    negative_per_batch: int = 288,
    max_negative_weight: int = 1_200,
    target_false_positives_per_hour: float = 0.2,
    feature_device: str = "cpu",
    feature_batch_size: int = 64,
    seed: int = 20260319,
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

    trainer = train_backend or _train_openwakeword_model
    trainer(
        positive_train_features_path=positive_train_features_path,
        negative_train_features_path=negative_train_features_path,
        positive_validation_features_path=positive_validation_features_path,
        negative_validation_features_path=negative_validation_features_path,
        false_positive_validation_features_path=false_positive_validation_features_path,
        output_model_path=output_model,
        model_name=model_stem,
        model_type=model_type,
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
        "model_type": model_type,
        "layer_dim": int(layer_dim),
        "steps": int(steps),
        "training_rounds": int(training_rounds),
        "feature_device": feature_device,
        "feature_batch_size": int(feature_batch_size),
        "seed": int(seed),
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
