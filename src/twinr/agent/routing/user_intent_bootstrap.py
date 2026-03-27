"""Build local user-intent bundles from labeled transcript datasets.

The first router stage can use either centroid similarity or a trained linear
softmax head over the frozen ONNX sentence embeddings.
"""

# CHANGELOG: 2026-03-27
# BUG-1: Rebuilds no longer leave stale classifier artifacts behind; bundles are now staged in a clean directory before promotion.
# BUG-2: JSONL split parsing now normalizes train/dev/test aliases and refuses silent dev/test leakage into training.
# BUG-3: The default reference date is now generated at build time instead of emitting a stale hard-coded provenance date.
# BUG-4: Empty datasets, blank texts, and incomplete label coverage now fail fast instead of producing broken centroids/heads.
# BUG-5: ONNX models that store weights in external data files are now repacked into a self-contained bundle model instead of being copied incompletely.
# SEC-1: The builder now validates ONNX graphs and repacks external-data models, reducing practical symlink/path-traversal exposure from hostile external-data references.
# SEC-2: Managed artifacts are written from a clean staging directory and NumPy arrays are saved with allow_pickle=False to avoid unnecessary pickle-enabled artifacts.
# IMP-1: The script can now optimize and dynamically quantize ONNX models for CPU/ARM deployment and emits build/metrics manifests.
# IMP-2: The linear head can use a balanced scikit-learn multinomial solver with small dev-set model selection when available, with safe fallback to the legacy fitter.
# IMP-3: The bundle now records calibration and abstention metadata (temperature, confidence thresholds, centroid radii) for open-set-aware routing.

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import uuid
from typing import Any, Mapping, Sequence
import warnings

import numpy as np

from .inference import OnnxSentenceEncoder
from .linear_head import fit_multiclass_linear_head
from .user_intent import USER_INTENT_LABEL_VALUES
from .user_intent_bundle import UserIntentBundleMetadata
from .user_intent_evaluation import (
    LabeledUserIntentSample,
    load_labeled_user_intent_samples,
)

try:  # Optional, but strongly recommended for safe ONNX repacking/validation.
    import onnx  # type: ignore
except Exception:  # pragma: no cover - optional dependency.
    onnx = None  # type: ignore[assignment]

try:  # Optional frontier optimization path.
    from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions  # type: ignore
    from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore
except Exception:  # pragma: no cover - optional dependency.
    GraphOptimizationLevel = None  # type: ignore[assignment]
    InferenceSession = None  # type: ignore[assignment]
    SessionOptions = None  # type: ignore[assignment]
    QuantType = None  # type: ignore[assignment]
    quantize_dynamic = None  # type: ignore[assignment]


_EXPECTED_LABELS: tuple[str, ...] = tuple(str(label).strip() for label in USER_INTENT_LABEL_VALUES)
_MANAGED_FILENAMES: frozenset[str] = frozenset(
    {
        "model.onnx",
        "model.ort",
        "model.optimized.onnx",
        "model.quantized.onnx",
        "model.required_operators.config",
        "tokenizer.json",
        "centroids.npy",
        "weights.npy",
        "bias.npy",
        "user_intent_metadata.json",
        "bundle_manifest.json",
        "bundle_metrics.json",
    }
)
_SPLIT_ALIASES: Mapping[str, str] = {
    "train": "train",
    "training": "train",
    "trn": "train",
    "dev": "dev",
    "valid": "dev",
    "validation": "dev",
    "val": "dev",
    "eval": "dev",
    "test": "test",
    "testing": "test",
    "tst": "test",
}


@dataclass(frozen=True)
class _BundleSample:
    text: str
    label: str
    split: str | None = None


@dataclass(frozen=True)
class _LinearHeadArtifacts:
    weights: np.ndarray
    bias: np.ndarray
    backend: str
    solver: str
    class_weight: str | None
    regularization_c: float | None


@dataclass(frozen=True)
class _PreparedModelArtifacts:
    model_file_name: str
    tokenizer_file_name: str
    files: tuple[Path, ...]
    repacked_external_data: bool
    optimized_model: bool
    quantized_model: bool
    onnx_validation_performed: bool
    warnings: tuple[str, ...]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _default_reference_date() -> str:
    return _utc_now().date().isoformat()


def _coerce_reference_date(reference_date: str | None) -> str:
    return reference_date or _default_reference_date()


def _normalize_split(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    return _SPLIT_ALIASES.get(normalized, normalized)


def _normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.split())


def _normalize_label(value: Any) -> str:
    return str(value).strip()


def _sanitize_samples(
    samples: Sequence[LabeledUserIntentSample],
    *,
    dataset_path: str | Path | None = None,
) -> tuple[_BundleSample, ...]:
    if not samples:
        source = f" from {dataset_path}" if dataset_path is not None else ""
        raise ValueError(f"No labeled user-intent samples were found{source}.")

    sanitized: list[_BundleSample] = []
    blank_text_count = 0
    unknown_labels: list[str] = []

    for sample in samples:
        text = _normalize_text(getattr(sample, "text", ""))
        label = _normalize_label(getattr(sample, "label", ""))
        split = _normalize_split(getattr(sample, "split", None))
        if not text:
            blank_text_count += 1
            continue
        if label not in _EXPECTED_LABELS:
            unknown_labels.append(label)
            continue
        sanitized.append(_BundleSample(text=text, label=label, split=split))

    if unknown_labels:
        unknown_display = ", ".join(sorted(set(unknown_labels)))
        raise ValueError(
            f"Dataset contains labels outside USER_INTENT_LABEL_VALUES: {unknown_display}"
        )
    if not sanitized:
        raise ValueError("All dataset rows were blank or invalid after sanitization.")
    if blank_text_count:
        warnings.warn(
            f"Dropped {blank_text_count} blank/whitespace-only samples before training.",
            stacklevel=2,
        )
    return tuple(sanitized)


def _partition_samples(
    samples: Sequence[_BundleSample],
) -> tuple[tuple[_BundleSample, ...], tuple[_BundleSample, ...], tuple[_BundleSample, ...]]:
    train_samples = tuple(sample for sample in samples if sample.split in (None, "train"))
    dev_samples = tuple(sample for sample in samples if sample.split == "dev")
    test_samples = tuple(sample for sample in samples if sample.split == "test")
    explicit_non_train = any(sample.split in {"dev", "test"} for sample in samples)
    explicit_unknown = sorted({sample.split for sample in samples if sample.split not in {None, "train", "dev", "test"}})

    if explicit_unknown:
        unknown_display = ", ".join(explicit_unknown)
        raise ValueError(
            f"Unsupported split names after normalization: {unknown_display}. "
            "Use train/dev/test (aliases like validation and training are accepted)."
        )

    if not train_samples:
        if explicit_non_train:
            raise ValueError(
                "No training samples were found after split normalization. "
                "Refusing to train on dev/test rows implicitly."
            )
        train_samples = tuple(samples)

    return train_samples, dev_samples, test_samples


def _validate_build_inputs(
    *,
    source_dir: str | Path,
    train_samples: Sequence[_BundleSample],
    max_length: int,
    pooling: str,
) -> None:
    if max_length <= 0:
        raise ValueError(f"max_length must be > 0, got {max_length}")
    if pooling not in {"mean", "cls", "prepooled"}:
        raise ValueError(f"Unsupported pooling={pooling!r}; expected mean, cls, or prepooled")

    source_path = Path(source_dir).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"User intent source dir does not exist: {source_path}")

    if not train_samples:
        raise ValueError("Training sample list is empty after preprocessing.")

    label_counts = Counter(sample.label for sample in train_samples)
    missing_labels = [label for label in _EXPECTED_LABELS if label not in label_counts]

    # BREAKING: fail fast on incomplete label coverage instead of silently emitting
    # unusable centroids/linear heads for labels the model never saw.
    if missing_labels:
        missing_display = ", ".join(missing_labels)
        raise ValueError(
            "Training data does not cover the full user-intent schema. "
            f"Missing labels: {missing_display}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_npy(path: Path, array: np.ndarray) -> None:
    np.save(path, np.asarray(array), allow_pickle=False)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _promote_stage_dir(stage_dir: Path, output_dir: Path) -> Path:
    output_dir = output_dir.expanduser()
    output_parent = output_dir.parent.resolve(strict=False)
    output_parent.mkdir(parents=True, exist_ok=True)

    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"output_dir must be a directory path, got existing file: {output_dir}")
    if output_dir.is_symlink():
        raise ValueError(f"Refusing to promote bundle into symlinked output dir: {output_dir}")

    backup_dir: Path | None = None
    if output_dir.exists():
        backup_dir = output_parent / f".{output_dir.name}.backup-{os.getpid()}-{uuid.uuid4().hex}"
        output_dir.rename(backup_dir)
    try:
        stage_dir.rename(output_dir)
    except Exception:
        if backup_dir is not None and not output_dir.exists():
            backup_dir.rename(output_dir)
        raise
    else:
        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)
    return output_dir


def _prepare_source_paths(source_dir: str | Path) -> tuple[Path, Path, Path]:
    source_path = Path(source_dir).expanduser().resolve(strict=True)
    model_path = source_path / "model.onnx"
    tokenizer_path = source_path / "tokenizer.json"
    if not model_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError(
            f"User intent source dir {source_path} must contain model.onnx and tokenizer.json"
        )
    return source_path, model_path, tokenizer_path


def _validate_onnx_model(model_path: Path) -> bool:
    if onnx is None:
        return False
    onnx.checker.check_model(str(model_path), full_check=False)
    return True


def _repack_onnx_to_self_contained(source_model_path: Path, destination_model_path: Path) -> tuple[bool, bool]:
    # BREAKING: safe bundle builds now require the onnx package so model graphs can
    # be validated and external-data references can be repacked into the bundle.
    if onnx is None:
        raise RuntimeError(
            "Building user-intent bundles now requires the 'onnx' package. "
            "Install it to validate and repack model.onnx safely."
        )

    validated = _validate_onnx_model(source_model_path)
    model_proto = onnx.load(str(source_model_path))
    repacked_external_data = bool(
        any(getattr(tensor, "data_location", 0) != 0 for tensor in model_proto.graph.initializer)
    )
    onnx.save_model(model_proto, str(destination_model_path), save_as_external_data=False)
    return repacked_external_data, validated


def _optimize_onnx_model(model_path: Path, optimized_model_path: Path) -> bool:
    if InferenceSession is None or SessionOptions is None or GraphOptimizationLevel is None:
        return False
    sess_options = SessionOptions()
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.optimized_model_filepath = str(optimized_model_path)
    # Loading the session performs the offline graph rewrite and serialization.
    InferenceSession(str(model_path), sess_options=sess_options, providers=["CPUExecutionProvider"])
    return optimized_model_path.exists()


def _quantize_onnx_model(source_model_path: Path, quantized_model_path: Path) -> bool:
    if quantize_dynamic is None or QuantType is None:
        return False
    quantize_dynamic(
        str(source_model_path),
        str(quantized_model_path),
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=False,
    )
    return quantized_model_path.exists()


def _export_ort_sidecar(model_path: Path) -> Path:
    try:
        from onnxruntime.tools.convert_onnx_models_to_ort import (  # type: ignore
            OptimizationStyle,
            convert_onnx_models_to_ort,
        )
    except Exception as exc:  # pragma: no cover - optional ORT tooling path.
        raise RuntimeError(f"ORT conversion tooling unavailable: {exc}") from exc

    ort_path = model_path.with_suffix(".ort")
    try:
        convert_onnx_models_to_ort(
            model_path,
            output_dir=model_path.parent,
            optimization_styles=[OptimizationStyle.Fixed],
            target_platform="arm",
            save_optimized_onnx_model=False,
            allow_conversion_failures=False,
            enable_type_reduction=False,
        )
    except Exception as exc:
        if ort_path.is_file():
            return ort_path
        raise RuntimeError(f"ORT conversion failed: {exc}") from exc
    if not ort_path.is_file():
        raise RuntimeError(f"ORT conversion finished without creating {ort_path.name}.")
    return ort_path


def _prepare_model_artifacts(
    *,
    source_dir: str | Path,
    stage_dir: Path,
    quantize_mode: str,
    optimize_model: bool,
) -> _PreparedModelArtifacts:
    _, model_path, tokenizer_path = _prepare_source_paths(source_dir)
    bundle_model_path = stage_dir / "model.onnx"
    bundle_tokenizer_path = stage_dir / "tokenizer.json"

    repacked_external_data, validated = _repack_onnx_to_self_contained(model_path, bundle_model_path)
    shutil.copy2(tokenizer_path, bundle_tokenizer_path)

    prepared_files: list[Path] = [bundle_model_path, bundle_tokenizer_path]
    prepared_warnings: list[str] = []
    optimized_model = False
    quantized_model = False
    active_model_file = bundle_model_path

    if optimize_model:
        optimized_path = stage_dir / "model.optimized.onnx"
        try:
            optimized_model = _optimize_onnx_model(bundle_model_path, optimized_path)
        except Exception as exc:  # pragma: no cover - depends on local ORT install/model support.
            prepared_warnings.append(f"ONNX optimization skipped: {exc}")
            optimized_model = False
        if optimized_model:
            prepared_files.append(optimized_path)
            active_model_file = optimized_path

    if quantize_mode not in {"off", "auto", "dynamic_int8"}:
        raise ValueError(f"Unsupported quantize_mode={quantize_mode!r}")

    should_try_quantization = quantize_mode in {"auto", "dynamic_int8"}
    if should_try_quantization:
        quantized_path = stage_dir / "model.quantized.onnx"
        try:
            quantized_model = _quantize_onnx_model(active_model_file, quantized_path)
        except Exception as exc:  # pragma: no cover - depends on local ORT install/model support.
            if quantize_mode == "dynamic_int8":
                raise
            prepared_warnings.append(f"ONNX quantization skipped: {exc}")
            quantized_model = False
        if quantized_model:
            prepared_files.append(quantized_path)
            shutil.copy2(quantized_path, bundle_model_path)
            active_model_file = bundle_model_path

    try:
        ort_path = _export_ort_sidecar(bundle_model_path)
    except Exception as exc:  # pragma: no cover - depends on optional ORT tooling.
        prepared_warnings.append(f"ORT export skipped: {exc}")
    else:
        prepared_files.append(ort_path)
        required_ops_config = bundle_model_path.with_name("model.required_operators.config")
        if required_ops_config.is_file():
            prepared_files.append(required_ops_config)

    return _PreparedModelArtifacts(
        model_file_name="model.onnx",
        tokenizer_file_name="tokenizer.json",
        files=tuple(prepared_files),
        repacked_external_data=repacked_external_data,
        optimized_model=optimized_model,
        quantized_model=quantized_model,
        onnx_validation_performed=validated,
        warnings=tuple(prepared_warnings),
    )


def _compute_embeddings(
    *,
    model_path: Path,
    tokenizer_path: Path,
    texts: Sequence[str],
    max_length: int,
    pooling: str,
    output_name: str | None,
) -> np.ndarray:
    encoder = OnnxSentenceEncoder(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        pooling=pooling,
        output_name=output_name,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(encoder.encode(list(texts)), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise ValueError(
            f"Expected embeddings with shape (n_samples, dim); got {embeddings.shape!r}"
        )
    return np.ascontiguousarray(embeddings, dtype=np.float32)


def _compute_centroids(
    embeddings: np.ndarray,
    labels: Sequence[str],
    *,
    label_order: Sequence[str],
) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings for centroid computation, got {embeddings.shape!r}")
    if len(labels) != embeddings.shape[0]:
        raise ValueError("Number of labels does not match number of embedding rows.")

    centroids = np.empty((len(label_order), embeddings.shape[1]), dtype=np.float32)
    label_array = np.asarray(labels, dtype=object)
    for index, label in enumerate(label_order):
        mask = label_array == label
        if not np.any(mask):
            raise ValueError(f"Cannot compute centroid for label {label!r}: no training samples.")
        centroid = embeddings[mask].mean(axis=0, dtype=np.float32)
        norm = float(np.linalg.norm(centroid))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError(f"Centroid for label {label!r} is degenerate.")
        centroids[index] = centroid / norm
    return np.ascontiguousarray(centroids, dtype=np.float32)


def _macro_f1_score(true_labels: Sequence[str], predicted_labels: Sequence[str], labels: Sequence[str]) -> float:
    if not true_labels:
        return float("nan")
    label_set = tuple(labels)
    scores: list[float] = []
    for label in label_set:
        tp = sum(1 for t, p in zip(true_labels, predicted_labels) if t == label and p == label)
        fp = sum(1 for t, p in zip(true_labels, predicted_labels) if t != label and p == label)
        fn = sum(1 for t, p in zip(true_labels, predicted_labels) if t == label and p != label)
        if tp == 0 and fp == 0 and fn == 0:
            scores.append(0.0)
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(0.0 if (precision + recall) == 0.0 else (2.0 * precision * recall) / (precision + recall))
    return float(np.mean(scores))


def _top2_from_probs(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if probabilities.shape[1] == 1:
        top1 = probabilities[:, 0]
        top2 = np.zeros_like(top1)
        return top1, top2
    partitioned = np.partition(probabilities, kth=probabilities.shape[1] - 2, axis=1)
    top2 = partitioned[:, -2]
    top1 = partitioned[:, -1]
    return top1, top2


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    scaled = logits / float(temperature)
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    np.clip(shifted, -60.0, 60.0, out=shifted)
    exp_scores = np.exp(shifted, dtype=np.float64)
    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.asarray(probabilities, dtype=np.float64)


def _negative_log_likelihood(probabilities: np.ndarray, true_indices: np.ndarray) -> float:
    eps = 1e-12
    clipped = np.clip(probabilities[np.arange(probabilities.shape[0]), true_indices], eps, 1.0)
    return float(-np.mean(np.log(clipped)))


def _fit_temperature(logits: np.ndarray, true_indices: np.ndarray) -> float:
    if logits.size == 0:
        return 1.0
    candidate_temperatures = np.logspace(-2, 1.5, num=72)
    best_temperature = 1.0
    best_nll = float("inf")
    for temperature in candidate_temperatures:
        probabilities = _softmax(logits, temperature=float(temperature))
        nll = _negative_log_likelihood(probabilities, true_indices)
        if nll < best_nll:
            best_nll = nll
            best_temperature = float(temperature)
    return best_temperature


def _expected_calibration_error(probabilities: np.ndarray, true_indices: np.ndarray, *, num_bins: int = 10) -> float:
    if probabilities.size == 0:
        return float("nan")
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == true_indices).astype(np.float64)

    edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right >= 1.0:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)
        if not np.any(mask):
            continue
        bin_conf = float(np.mean(confidences[mask]))
        bin_acc = float(np.mean(correct[mask]))
        ece += (np.sum(mask) / len(confidences)) * abs(bin_acc - bin_conf)
    return float(ece)


def _compute_linear_scores(embeddings: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32).reshape(-1)
    if weights.ndim != 2:
        raise ValueError(f"weights must be 2D, got {weights.shape!r}")
    if bias.ndim != 1:
        raise ValueError(f"bias must be 1D, got {bias.shape!r}")
    if weights.shape[0] == bias.shape[0] and weights.shape[1] == embeddings.shape[1]:
        return np.ascontiguousarray(embeddings @ weights.T + bias, dtype=np.float32)
    if weights.shape[1] == bias.shape[0] and weights.shape[0] == embeddings.shape[1]:
        return np.ascontiguousarray(embeddings @ weights + bias, dtype=np.float32)
    raise ValueError(
        "Linear head shape mismatch: embeddings shape "
        f"{embeddings.shape!r}, weights shape {weights.shape!r}, bias shape {bias.shape!r}"
    )


def _scikit_learn_linear_head(
    train_embeddings: np.ndarray,
    train_labels: Sequence[str],
    *,
    label_order: Sequence[str],
    dev_embeddings: np.ndarray | None,
    dev_labels: Sequence[str] | None,
) -> _LinearHeadArtifacts | None:
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore
    except Exception:  # pragma: no cover - optional dependency.
        return None

    label_to_index = {label: idx for idx, label in enumerate(label_order)}
    y_train = np.asarray([label_to_index[label] for label in train_labels], dtype=np.int64)

    candidate_cs: tuple[float, ...]
    if dev_embeddings is not None and dev_labels:
        candidate_cs = (0.25, 1.0, 4.0, 16.0)
    else:
        candidate_cs = (1.0,)

    best_model: Any = None
    best_score = (-float("inf"), -float("inf"))  # (macro_f1, -nll)
    for c_value in candidate_cs:
        logistic_kwargs: dict[str, Any] = {
            "solver": "lbfgs",
            "multi_class": "multinomial",
            "class_weight": "balanced",
            "C": c_value,
            "max_iter": 2000,
            "tol": 1e-4,
            "fit_intercept": True,
            "random_state": 0,
        }
        model = LogisticRegression(**logistic_kwargs)  # pylint: disable=unexpected-keyword-arg
        model.fit(np.asarray(train_embeddings, dtype=np.float64, order="C"), y_train)

        if dev_embeddings is not None and dev_labels:
            y_dev = np.asarray([label_to_index[label] for label in dev_labels], dtype=np.int64)
            probabilities = model.predict_proba(np.asarray(dev_embeddings, dtype=np.float64, order="C"))
            predictions = probabilities.argmax(axis=1)
            predicted_labels = [label_order[index] for index in predictions]
            macro_f1 = _macro_f1_score(dev_labels, predicted_labels, label_order)
            nll = _negative_log_likelihood(probabilities, y_dev)
            score = (macro_f1, -nll)
        else:
            score = (0.0, 0.0)

        if best_model is None or score > best_score:
            best_model = model
            best_score = score

    assert best_model is not None

    coef = np.asarray(best_model.coef_, dtype=np.float32)
    intercept = np.asarray(best_model.intercept_, dtype=np.float32)
    if coef.shape[0] == 1 and len(label_order) == 2:
        # Convert the binary logistic parameterization into an explicit 2-class softmax head.
        zero_weights = np.zeros_like(coef[0], dtype=np.float32)
        zero_bias = np.float32(0.0)
        coef = np.stack([zero_weights, coef[0]], axis=0)
        intercept = np.asarray([zero_bias, intercept[0]], dtype=np.float32)

    if coef.shape[0] != len(label_order):
        raise ValueError(
            f"Expected {len(label_order)} weight rows after fitting, got {coef.shape[0]}"
        )
    return _LinearHeadArtifacts(
        weights=np.ascontiguousarray(coef, dtype=np.float32),
        bias=np.ascontiguousarray(intercept.reshape(-1), dtype=np.float32),
        backend="sklearn",
        solver="lbfgs",
        class_weight="balanced",
        regularization_c=float(best_model.C),
    )


def _legacy_linear_head(
    train_embeddings: np.ndarray,
    train_labels: Sequence[str],
    *,
    label_order: Sequence[str],
    dev_embeddings: np.ndarray | None,
    dev_labels: Sequence[str] | None,
) -> _LinearHeadArtifacts:
    trained_head = fit_multiclass_linear_head(
        train_embeddings,
        list(train_labels),
        label_order=label_order,
        dev_embeddings=dev_embeddings,
        dev_labels=list(dev_labels) if dev_labels is not None else None,
    )
    return _LinearHeadArtifacts(
        weights=np.ascontiguousarray(np.asarray(trained_head.weights), dtype=np.float32),
        bias=np.ascontiguousarray(np.asarray(trained_head.bias).reshape(-1), dtype=np.float32),
        backend="legacy_fit_multiclass_linear_head",
        solver="unknown",
        class_weight=None,
        regularization_c=None,
    )


def _fit_linear_head_artifacts(
    train_embeddings: np.ndarray,
    train_labels: Sequence[str],
    *,
    label_order: Sequence[str],
    dev_embeddings: np.ndarray | None,
    dev_labels: Sequence[str] | None,
) -> _LinearHeadArtifacts:
    sklearn_head = _scikit_learn_linear_head(
        train_embeddings,
        train_labels,
        label_order=label_order,
        dev_embeddings=dev_embeddings,
        dev_labels=dev_labels,
    )
    if sklearn_head is not None:
        return sklearn_head
    return _legacy_linear_head(
        train_embeddings,
        train_labels,
        label_order=label_order,
        dev_embeddings=dev_embeddings,
        dev_labels=dev_labels,
    )


def _evaluate_classifier(
    *,
    classifier_type: str,
    label_order: Sequence[str],
    train_samples: Sequence[_BundleSample],
    train_embeddings: np.ndarray,
    dev_samples: Sequence[_BundleSample],
    dev_embeddings: np.ndarray | None,
    centroids: np.ndarray | None = None,
    linear_head: _LinearHeadArtifacts | None = None,
) -> dict[str, Any]:
    label_to_index = {label: index for index, label in enumerate(label_order)}
    report: dict[str, Any] = {
        "classifier_type": classifier_type,
        "train_size": len(train_samples),
        "dev_size": len(dev_samples),
        "train_label_counts": dict(sorted(Counter(sample.label for sample in train_samples).items())),
        "dev_label_counts": dict(sorted(Counter(sample.label for sample in dev_samples).items())),
        "calibration": {
            "temperature": 1.0,
            "max_probability_threshold": None,
            "margin_threshold": None,
            "strategy": None,
        },
    }

    if classifier_type == "embedding_centroid_v1":
        assert centroids is not None
        train_scores = np.ascontiguousarray(train_embeddings @ centroids.T, dtype=np.float32)
        centroid_radii: dict[str, float] = {}
        for index, label in enumerate(label_order):
            label_mask = np.asarray([sample.label == label for sample in train_samples], dtype=bool)
            label_scores = train_scores[label_mask, index]
            centroid_radii[label] = float(np.quantile(label_scores, 0.05))
        report["centroid_similarity_floor_p05"] = centroid_radii
    elif classifier_type == "embedding_linear_softmax_v1":
        assert linear_head is not None
        report["linear_head"] = {
            "backend": linear_head.backend,
            "solver": linear_head.solver,
            "class_weight": linear_head.class_weight,
            "regularization_c": linear_head.regularization_c,
        }
    else:
        raise ValueError(f"Unsupported classifier_type={classifier_type!r}")

    if not dev_samples or dev_embeddings is None:
        return report

    true_labels = [sample.label for sample in dev_samples]
    true_indices = np.asarray([label_to_index[label] for label in true_labels], dtype=np.int64)

    if classifier_type == "embedding_centroid_v1":
        assert centroids is not None
        logits = np.ascontiguousarray(dev_embeddings @ centroids.T, dtype=np.float32)
    else:
        assert linear_head is not None
        logits = _compute_linear_scores(dev_embeddings, linear_head.weights, linear_head.bias)

    temperature = _fit_temperature(logits, true_indices)
    probabilities = _softmax(logits, temperature=temperature)
    prediction_indices = probabilities.argmax(axis=1)
    predicted_labels = [label_order[index] for index in prediction_indices]

    top1, top2 = _top2_from_probs(probabilities)
    correct_mask = prediction_indices == true_indices
    correct_top1 = top1[correct_mask]
    correct_margins = (top1 - top2)[correct_mask]

    confidence_threshold = float(np.quantile(correct_top1, 0.05)) if correct_top1.size else None
    margin_threshold = float(np.quantile(correct_margins, 0.05)) if correct_margins.size else None

    report["dev_metrics"] = {
        "accuracy": float(np.mean(correct_mask)),
        "macro_f1": _macro_f1_score(true_labels, predicted_labels, label_order),
        "negative_log_likelihood": _negative_log_likelihood(probabilities, true_indices),
        "expected_calibration_error_10bin": _expected_calibration_error(probabilities, true_indices),
    }
    report["calibration"] = {
        "temperature": float(temperature),
        "max_probability_threshold": confidence_threshold,
        "margin_threshold": margin_threshold,
        "strategy": "dev_set_temperature_scaling_with_correct_prediction_p05_thresholds",
    }
    return report


def _metadata_payload(
    *,
    metadata: UserIntentBundleMetadata,
    model_id: str,
    build_report: Mapping[str, Any],
    model_artifacts: _PreparedModelArtifacts,
    source_dir: str | Path,
    reference_date: str,
) -> dict[str, Any]:
    payload = dict(metadata.to_dict())
    payload.update(
        {
            "bundle_built_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
            "reference_date": reference_date,
            "model_id": model_id,
            "source_dir": str(Path(source_dir).expanduser()),
            "onnx_validation_performed": model_artifacts.onnx_validation_performed,
            "repacked_external_data": model_artifacts.repacked_external_data,
            "optimized_model": model_artifacts.optimized_model,
            "quantized_model": model_artifacts.quantized_model,
            "open_set": build_report.get("calibration"),
        }
    )
    if "centroid_similarity_floor_p05" in build_report:
        payload["centroid_similarity_floor_p05"] = build_report["centroid_similarity_floor_p05"]
    if "linear_head" in build_report:
        payload["linear_head"] = build_report["linear_head"]
    return payload


def _write_bundle_manifests(
    *,
    stage_dir: Path,
    metadata_payload: Mapping[str, Any],
    build_report: Mapping[str, Any],
    source_dir: str | Path,
    dataset_path: str | Path | None,
    classifier_type: str,
    model_artifacts: _PreparedModelArtifacts,
) -> None:
    output_files = {
        path.name: {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(stage_dir.iterdir(), key=lambda item: item.name)
        if path.is_file()
    }
    manifest = {
        "built_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
        "source_dir": str(Path(source_dir).expanduser()),
        "dataset_path": str(Path(dataset_path).expanduser()) if dataset_path is not None else None,
        "classifier_type": classifier_type,
        "managed_filenames": sorted(_MANAGED_FILENAMES),
        "warnings": list(model_artifacts.warnings),
        "output_files": output_files,
    }
    _write_json(stage_dir / "bundle_manifest.json", manifest)
    _write_json(stage_dir / "bundle_metrics.json", dict(build_report))


def _build_bundle_common(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    train_samples: Sequence[_BundleSample],
    dev_samples: Sequence[_BundleSample],
    classifier: str,
    model_id: str | None,
    max_length: int,
    pooling: str,
    output_name: str | None,
    reference_date: str | None,
    dataset_path: str | Path | None,
    quantize_mode: str,
    optimize_model: bool,
) -> Path:
    _validate_build_inputs(
        source_dir=source_dir,
        train_samples=train_samples,
        max_length=max_length,
        pooling=pooling,
    )

    source_path, _, _ = _prepare_source_paths(source_dir)
    final_model_id = model_id or source_path.name
    final_reference_date = _coerce_reference_date(reference_date)

    output_path = Path(output_dir).expanduser().resolve(strict=False)
    output_parent = output_path.parent.resolve(strict=False)
    output_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.name}.stage-",
        dir=str(output_parent),
    ) as temp_dir_name:
        stage_dir = Path(temp_dir_name)
        model_artifacts = _prepare_model_artifacts(
            source_dir=source_dir,
            stage_dir=stage_dir,
            quantize_mode=quantize_mode,
            optimize_model=optimize_model,
        )

        model_path = stage_dir / model_artifacts.model_file_name
        tokenizer_path = stage_dir / model_artifacts.tokenizer_file_name

        train_embeddings = _compute_embeddings(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            texts=[sample.text for sample in train_samples],
            max_length=max_length,
            pooling=pooling,
            output_name=output_name,
        )
        dev_embeddings = (
            _compute_embeddings(
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                texts=[sample.text for sample in dev_samples],
                max_length=max_length,
                pooling=pooling,
                output_name=output_name,
            )
            if dev_samples
            else None
        )

        if classifier == "centroid":
            centroids = _compute_centroids(
                train_embeddings,
                [sample.label for sample in train_samples],
                label_order=_EXPECTED_LABELS,
            )
            _save_npy(stage_dir / "centroids.npy", centroids)
            build_report = _evaluate_classifier(
                classifier_type="embedding_centroid_v1",
                label_order=_EXPECTED_LABELS,
                train_samples=train_samples,
                train_embeddings=train_embeddings,
                dev_samples=dev_samples,
                dev_embeddings=dev_embeddings,
                centroids=centroids,
            )
            metadata = UserIntentBundleMetadata(
                schema_version=1,
                classifier_type="embedding_centroid_v1",
                labels=_EXPECTED_LABELS,
                model_id=final_model_id,
                max_length=max_length,
                pooling=pooling,
                output_name=output_name,
                normalize_embeddings=True,
                normalize_centroids=True,
                reference_date=final_reference_date,
            )
        elif classifier == "linear":
            linear_head = _fit_linear_head_artifacts(
                train_embeddings,
                [sample.label for sample in train_samples],
                label_order=_EXPECTED_LABELS,
                dev_embeddings=dev_embeddings,
                dev_labels=[sample.label for sample in dev_samples] if dev_samples else None,
            )
            _save_npy(stage_dir / "weights.npy", linear_head.weights)
            _save_npy(stage_dir / "bias.npy", linear_head.bias)
            build_report = _evaluate_classifier(
                classifier_type="embedding_linear_softmax_v1",
                label_order=_EXPECTED_LABELS,
                train_samples=train_samples,
                train_embeddings=train_embeddings,
                dev_samples=dev_samples,
                dev_embeddings=dev_embeddings,
                linear_head=linear_head,
            )
            metadata = UserIntentBundleMetadata(
                schema_version=1,
                classifier_type="embedding_linear_softmax_v1",
                labels=_EXPECTED_LABELS,
                model_id=final_model_id,
                max_length=max_length,
                pooling=pooling,
                output_name=output_name,
                normalize_embeddings=True,
                normalize_centroids=False,
                reference_date=final_reference_date,
            )
        else:
            raise ValueError(f"Unsupported classifier={classifier!r}")

        metadata_payload = _metadata_payload(
            metadata=metadata,
            model_id=final_model_id,
            build_report=build_report,
            model_artifacts=model_artifacts,
            source_dir=source_dir,
            reference_date=final_reference_date,
        )
        _write_json(stage_dir / "user_intent_metadata.json", metadata_payload)
        _write_bundle_manifests(
            stage_dir=stage_dir,
            metadata_payload=metadata_payload,
            build_report=build_report,
            source_dir=source_dir,
            dataset_path=dataset_path,
            classifier_type=metadata_payload["classifier_type"],
            model_artifacts=model_artifacts,
        )
        promoted_path = _promote_stage_dir(stage_dir, output_path)
        return promoted_path


def build_centroid_user_intent_bundle(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    train_samples: Sequence[LabeledUserIntentSample],
    model_id: str | None = None,
    max_length: int = 128,
    pooling: str = "mean",
    output_name: str | None = None,
    reference_date: str | None = None,
    quantize_mode: str = "auto",
    optimize_model: bool = True,
) -> Path:
    """Build one centroid-based user-intent bundle from local source model files."""

    sanitized_train = _sanitize_samples(train_samples)
    return _build_bundle_common(
        source_dir=source_dir,
        output_dir=output_dir,
        train_samples=sanitized_train,
        dev_samples=(),
        classifier="centroid",
        model_id=model_id,
        max_length=max_length,
        pooling=pooling,
        output_name=output_name,
        reference_date=reference_date,
        dataset_path=None,
        quantize_mode=quantize_mode,
        optimize_model=optimize_model,
    )


def build_linear_user_intent_bundle(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    train_samples: Sequence[LabeledUserIntentSample],
    dev_samples: Sequence[LabeledUserIntentSample] = (),
    model_id: str | None = None,
    max_length: int = 128,
    pooling: str = "mean",
    output_name: str | None = None,
    reference_date: str | None = None,
    quantize_mode: str = "auto",
    optimize_model: bool = True,
) -> Path:
    """Build one linear-head user-intent bundle from local source model files."""

    sanitized_train = _sanitize_samples(train_samples)
    sanitized_dev = _sanitize_samples(dev_samples) if dev_samples else ()
    return _build_bundle_common(
        source_dir=source_dir,
        output_dir=output_dir,
        train_samples=sanitized_train,
        dev_samples=sanitized_dev,
        classifier="linear",
        model_id=model_id,
        max_length=max_length,
        pooling=pooling,
        output_name=output_name,
        reference_date=reference_date,
        dataset_path=None,
        quantize_mode=quantize_mode,
        optimize_model=optimize_model,
    )


def build_centroid_user_intent_bundle_from_jsonl(
    *,
    source_dir: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    model_id: str | None = None,
    max_length: int = 128,
    pooling: str = "mean",
    output_name: str | None = None,
    reference_date: str | None = None,
    quantize_mode: str = "auto",
    optimize_model: bool = True,
) -> Path:
    """Build one user-intent bundle from a JSONL dataset with train/dev/test splits."""

    raw_samples = load_labeled_user_intent_samples(dataset_path)
    samples = _sanitize_samples(raw_samples, dataset_path=dataset_path)
    train_samples, dev_samples, _ = _partition_samples(samples)
    return _build_bundle_common(
        source_dir=source_dir,
        output_dir=output_dir,
        train_samples=train_samples,
        dev_samples=dev_samples,
        classifier="centroid",
        model_id=model_id,
        max_length=max_length,
        pooling=pooling,
        output_name=output_name,
        reference_date=reference_date,
        dataset_path=dataset_path,
        quantize_mode=quantize_mode,
        optimize_model=optimize_model,
    )


def build_linear_user_intent_bundle_from_jsonl(
    *,
    source_dir: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    model_id: str | None = None,
    max_length: int = 128,
    pooling: str = "mean",
    output_name: str | None = None,
    reference_date: str | None = None,
    quantize_mode: str = "auto",
    optimize_model: bool = True,
) -> Path:
    """Build one linear-head user-intent bundle from a JSONL dataset."""

    raw_samples = load_labeled_user_intent_samples(dataset_path)
    samples = _sanitize_samples(raw_samples, dataset_path=dataset_path)
    train_samples, dev_samples, _ = _partition_samples(samples)
    return _build_bundle_common(
        source_dir=source_dir,
        output_dir=output_dir,
        train_samples=train_samples,
        dev_samples=dev_samples,
        classifier="linear",
        model_id=model_id,
        max_length=max_length,
        pooling=pooling,
        output_name=output_name,
        reference_date=reference_date,
        dataset_path=dataset_path,
        quantize_mode=quantize_mode,
        optimize_model=optimize_model,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Twinr user-intent bundle.")
    parser.add_argument("--source-dir", required=True, help="Directory containing model.onnx and tokenizer.json")
    parser.add_argument("--dataset", required=True, help="JSONL file with text/label and optional split fields")
    parser.add_argument("--output-dir", required=True, help="Destination bundle directory")
    parser.add_argument("--model-id", default=None, help="Optional bundle/model identifier override")
    parser.add_argument("--max-length", type=int, default=128, help="Tokenizer truncation length")
    parser.add_argument("--pooling", default="mean", choices=("mean", "cls", "prepooled"))
    parser.add_argument("--output-name", default=None, help="Optional explicit ONNX output name")
    parser.add_argument("--classifier", default="centroid", choices=("centroid", "linear"))
    parser.add_argument(
        "--reference-date",
        default=None,
        help="Optional provenance date override; defaults to the UTC build date.",
    )
    parser.add_argument(
        "--quantize",
        dest="quantize_mode",
        default="auto",
        choices=("auto", "off", "dynamic_int8"),
        help=(
            "Model quantization policy. 'auto' tries CPU int8 dynamic quantization and "
            "falls back silently if ONNX Runtime quantization is unavailable."
        ),
    )
    parser.add_argument(
        "--no-optimize-model",
        action="store_true",
        help="Disable offline ONNX graph optimization before optional quantization.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for building user-intent bundles."""

    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    builder = (
        build_linear_user_intent_bundle_from_jsonl
        if args.classifier == "linear"
        else build_centroid_user_intent_bundle_from_jsonl
    )
    builder(
        source_dir=args.source_dir,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        model_id=args.model_id,
        max_length=args.max_length,
        pooling=args.pooling,
        output_name=args.output_name,
        reference_date=args.reference_date,
        quantize_mode=args.quantize_mode,
        optimize_model=not args.no_optimize_model,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via module CLI.
    raise SystemExit(main())
