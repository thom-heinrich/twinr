"""Build local semantic-router bundles from labeled transcript datasets.

Twinr router bundles reuse one ONNX sentence encoder and attach either
per-label embedding centroids or one trained linear softmax head. This keeps
the on-device runtime tiny while still giving the project a reproducible path
for calibration on real transcript datasets.

This 2026 revision adds three production-facing upgrades:

1. deterministic split handling for train/dev calibration,
2. atomic bundle staging to avoid half-written artifacts,
3. optional offline ONNX optimization plus ARM-friendly dynamic int8 quantization.

Example:

```bash
PYTHONPATH=src python3 -m twinr.agent.routing.bootstrap \
  --source-dir /models/paraphrase-multilingual-MiniLM-L12-v2-onnx \
  --dataset test/fixtures/router/router_samples.jsonl \
  --output-dir artifacts/router/multilingual-minilm-router
```
"""

# CHANGELOG: 2026-03-27
# BUG-1: Fixed silent train/dev leakage when JSONL datasets contain only `dev` samples or no explicit `train` split.
# BUG-2: Fixed partial/stale bundle emission by validating inputs early, requiring full train-label coverage, and staging outputs atomically.
# SEC-1: Fixed practical local file-clobber risk from pre-existing output trees/symlinks by building in a private temp directory and swapping only validated bundles.
# IMP-1: Added deterministic auto-dev splitting, build provenance reports, bundle health checks, and exact train/dev overlap detection.
# IMP-2: Added optional ONNX offline graph optimization, ARM-friendly dynamic int8 quantization, and post-hoc temperature scaling for linear heads.

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version as package_version
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence
import uuid
import warnings

import numpy as np

from .centroids import compute_label_centroids
from .bundle import SemanticRouterBundleMetadata, load_semantic_router_bundle
from .contracts import ROUTE_LABEL_VALUES
from .evaluation import (
    LabeledRouteSample,
    load_labeled_route_samples,
    score_semantic_router,
    tune_policy_thresholds,
)
from .inference import OnnxSentenceEncoder
from .linear_head import fit_multiclass_linear_head
from .service import LocalSemanticRouter


_DEFAULT_REFERENCE_DATE = "2026-03-27"
_DEFAULT_AUTO_DEV_RATIO = 0.20
_DEFAULT_SEED = 13
_DEFAULT_ENCODE_BATCH_SIZE = 256
_DEFAULT_MIN_PROBE_COSINE = 0.985

_DEV_SPLIT_ALIASES = {
    "dev": "dev",
    "val": "dev",
    "valid": "dev",
    "validation": "dev",
    "calib": "dev",
    "calibration": "dev",
    "development": "dev",
}
_TRAIN_SPLIT_ALIASES = {"train": "train", "training": "train"}
_TEST_SPLIT_ALIASES = {"test": "test", "eval": "test", "evaluation": "test"}
_KNOWN_SPLITS = set(_DEV_SPLIT_ALIASES) | set(_TRAIN_SPLIT_ALIASES) | set(_TEST_SPLIT_ALIASES)

_OPTIONAL_MODEL_SIDECARS = (
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "sentence_bert_config.json",
    "modules.json",
)


def _safe_package_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_router_metadata(path: Path, metadata: SemanticRouterBundleMetadata) -> None:
    _json_dump(path, metadata.to_dict())


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


def _normalize_split_name(split: str | None) -> str | None:
    if split is None:
        return None
    normalized = split.strip().lower()
    if not normalized:
        return None
    if normalized in _TRAIN_SPLIT_ALIASES:
        return "train"
    if normalized in _DEV_SPLIT_ALIASES:
        return "dev"
    if normalized in _TEST_SPLIT_ALIASES:
        return "test"
    raise ValueError(
        f"Unsupported dataset split {split!r}. Supported values are train/dev/test plus aliases: "
        f"{sorted(_KNOWN_SPLITS)}"
    )


def _normalize_sample_splits(samples: Sequence[LabeledRouteSample]) -> tuple[LabeledRouteSample, ...]:
    normalized: list[LabeledRouteSample] = []
    for sample in samples:
        normalized_split = _normalize_split_name(sample.split)
        if normalized_split == sample.split:
            normalized.append(sample)
        else:
            normalized.append(replace(sample, split=normalized_split))
    return tuple(normalized)


def _sample_key(sample: LabeledRouteSample) -> tuple[str, str]:
    return (sample.text.strip(), sample.label)


def _label_counts(samples: Sequence[LabeledRouteSample]) -> dict[str, int]:
    counts = Counter(sample.label for sample in samples)
    return {label: int(counts.get(label, 0)) for label in ROUTE_LABEL_VALUES}


def _validate_sample_payload(samples: Sequence[LabeledRouteSample], *, context: str) -> None:
    if not samples:
        raise ValueError(f"{context} samples must not be empty.")
    invalid_labels = sorted({sample.label for sample in samples if sample.label not in ROUTE_LABEL_VALUES})
    if invalid_labels:
        raise ValueError(
            f"{context} samples contain unsupported labels {invalid_labels}. "
            f"Expected labels: {list(ROUTE_LABEL_VALUES)}"
        )
    blank_count = sum(1 for sample in samples if not sample.text or not sample.text.strip())
    if blank_count:
        raise ValueError(f"{context} samples contain {blank_count} blank text entries.")


def _ensure_full_train_label_coverage(train_samples: Sequence[LabeledRouteSample]) -> None:
    missing = [label for label, count in _label_counts(train_samples).items() if count == 0]
    if missing:
        # BREAKING: bundle creation now fails fast instead of emitting classifiers with untrained route labels.
        raise ValueError(
            "Training split must contain at least one sample for every router label. "
            f"Missing labels: {missing}"
        )


def _assert_no_train_dev_overlap(
    train_samples: Sequence[LabeledRouteSample],
    dev_samples: Sequence[LabeledRouteSample],
) -> None:
    if not train_samples or not dev_samples:
        return
    train_keys = {_sample_key(sample) for sample in train_samples}
    dev_keys = {_sample_key(sample) for sample in dev_samples}
    overlap = sorted(train_keys & dev_keys)
    if overlap:
        preview = overlap[:5]
        raise ValueError(
            "Train/dev overlap detected. This silently corrupts calibration. "
            f"First overlapping samples: {preview}"
        )


def _can_make_stratified_dev_split(samples: Sequence[LabeledRouteSample]) -> bool:
    counts = _label_counts(samples)
    return bool(samples) and all(count >= 2 for count in counts.values())


def _stratified_holdout_split(
    samples: Sequence[LabeledRouteSample],
    *,
    dev_ratio: float,
    seed: int,
) -> tuple[tuple[LabeledRouteSample, ...], tuple[LabeledRouteSample, ...]]:
    if not 0.0 < dev_ratio < 1.0:
        raise ValueError(f"dev_ratio must be strictly between 0 and 1. Received {dev_ratio!r}")
    if not _can_make_stratified_dev_split(samples):
        raise ValueError(
            "Cannot create a stratified dev split because at least one route label has fewer than two samples."
        )

    label_to_indices = {label: [] for label in ROUTE_LABEL_VALUES}
    for index, sample in enumerate(samples):
        label_to_indices[sample.label].append(index)

    rng = np.random.default_rng(seed)
    dev_indices: set[int] = set()
    for label in ROUTE_LABEL_VALUES:
        indices = list(label_to_indices[label])
        rng.shuffle(indices)
        requested = int(round(len(indices) * dev_ratio))
        n_dev = min(len(indices) - 1, max(1, requested))
        dev_indices.update(indices[:n_dev])

    train_samples = tuple(sample for index, sample in enumerate(samples) if index not in dev_indices)
    dev_samples = tuple(sample for index, sample in enumerate(samples) if index in dev_indices)
    return train_samples, dev_samples


def _resolve_train_dev_samples_from_jsonl(
    samples: Sequence[LabeledRouteSample],
    *,
    auto_dev_split: bool,
    dev_ratio: float,
    seed: int,
) -> tuple[tuple[LabeledRouteSample, ...], tuple[LabeledRouteSample, ...], str]:
    normalized_samples = _normalize_sample_splits(tuple(samples))
    _validate_sample_payload(normalized_samples, context="dataset")

    train_like = [sample for sample in normalized_samples if sample.split == "train" or sample.split is None]
    dev_like = [sample for sample in normalized_samples if sample.split == "dev"]
    test_like = [sample for sample in normalized_samples if sample.split == "test"]

    if train_like and dev_like:
        train_samples = tuple(train_like)
        dev_samples = tuple(dev_like)
        strategy = "explicit_train_and_dev"
    elif train_like and not dev_like:
        if auto_dev_split and _can_make_stratified_dev_split(train_like):
            # BREAKING: JSONL/CLI builds now reserve a deterministic calibration split when possible.
            train_samples, dev_samples = _stratified_holdout_split(train_like, dev_ratio=dev_ratio, seed=seed)
            strategy = "auto_dev_from_train"
        else:
            train_samples = tuple(train_like)
            dev_samples = ()
            strategy = "train_only"
    elif not train_like and dev_like:
        if auto_dev_split and _can_make_stratified_dev_split(dev_like):
            train_samples, dev_samples = _stratified_holdout_split(dev_like, dev_ratio=dev_ratio, seed=seed)
            strategy = "auto_train_dev_from_dev_only"
        else:
            train_samples = tuple(dev_like)
            dev_samples = ()
            strategy = "dev_only_promoted_to_train_without_calibration"
    else:
        candidate = tuple(sample for sample in normalized_samples if sample.split != "test")
        if not candidate and test_like:
            raise ValueError("Dataset contains only test/eval samples; there is nothing to train on.")
        if auto_dev_split and _can_make_stratified_dev_split(candidate):
            train_samples, dev_samples = _stratified_holdout_split(candidate, dev_ratio=dev_ratio, seed=seed)
            strategy = "auto_train_dev_from_unsplit_dataset"
        else:
            train_samples = candidate
            dev_samples = ()
            strategy = "unsplit_dataset_promoted_to_train"

    _validate_sample_payload(train_samples, context="train")
    if dev_samples:
        _validate_sample_payload(dev_samples, context="dev")
    _assert_no_train_dev_overlap(train_samples, dev_samples)
    return train_samples, dev_samples, strategy


def _validate_source_dir(source_dir: str | Path) -> tuple[Path, Path, Path]:
    source_path = Path(source_dir).expanduser().resolve(strict=True)
    if not source_path.is_dir():
        raise NotADirectoryError(f"Semantic router source dir {source_path} is not a directory.")
    model_path = source_path / "model.onnx"
    tokenizer_path = source_path / "tokenizer.json"
    if not model_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError(
            f"Semantic router source dir {source_path} must contain model.onnx and tokenizer.json"
        )
    return source_path, model_path, tokenizer_path


def _validate_output_dir(source_path: Path, output_dir: str | Path) -> Path:
    raw_output_path = Path(output_dir).expanduser()
    if raw_output_path.exists():
        if raw_output_path.is_symlink():
            raise ValueError(f"Refusing to build into symlinked output path {raw_output_path}")
        if not raw_output_path.is_dir():
            raise ValueError(f"Output path {raw_output_path} exists and is not a directory.")
    output_path = raw_output_path.resolve(strict=False)
    if output_path == source_path:
        raise ValueError("Output directory must differ from source directory.")
    return output_path


def _copy_tokenizer_and_sidecars(source_dir: Path, destination_dir: Path) -> None:
    shutil.copy2(source_dir / "tokenizer.json", destination_dir / "tokenizer.json")
    for filename in _OPTIONAL_MODEL_SIDECARS:
        candidate = source_dir / filename
        if candidate.is_file():
            shutil.copy2(candidate, destination_dir / filename)


def _make_encoder(
    *,
    model_path: Path,
    tokenizer_path: Path,
    max_length: int,
    pooling: str,
    output_name: str | None,
) -> OnnxSentenceEncoder:
    return OnnxSentenceEncoder(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        pooling=pooling,
        output_name=output_name,
        normalize_embeddings=True,
    )


def _select_probe_texts(
    train_samples: Sequence[LabeledRouteSample],
    dev_samples: Sequence[LabeledRouteSample],
    *,
    limit: int = 8,
) -> list[str]:
    probe_texts: list[str] = []
    seen: set[str] = set()
    for sample in tuple(train_samples) + tuple(dev_samples):
        text = sample.text.strip()
        if text and text not in seen:
            seen.add(text)
            probe_texts.append(text)
        if len(probe_texts) >= limit:
            break
    return probe_texts or ["hello"]


def _encode_texts(
    encoder: OnnxSentenceEncoder,
    texts: Sequence[str],
    *,
    batch_size: int,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    batches: list[np.ndarray] = []
    step = max(1, int(batch_size))
    for start in range(0, len(texts), step):
        batch = list(texts[start : start + step])
        batches.append(np.asarray(encoder.encode(batch), dtype=np.float32))
    return np.concatenate(batches, axis=0)


def _compute_mean_cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError(f"Embedding shape mismatch: {left.shape} vs {right.shape}")
    cosine = np.sum(left * right, axis=1)
    return float(np.mean(cosine))


def _materialize_model_artifact(
    *,
    source_model_path: Path,
    source_tokenizer_path: Path,
    destination_dir: Path,
    max_length: int,
    pooling: str,
    output_name: str | None,
    probe_texts: Sequence[str],
    optimize_onnx: bool,
    quantize_onnx: str,
    min_probe_cosine: float,
    encode_batch_size: int,
) -> dict[str, Any]:
    destination_model_path = destination_dir / "model.onnx"
    report: dict[str, Any] = {
        "source_model_sha256": _sha256_file(source_model_path),
        "source_model_bytes": int(source_model_path.stat().st_size),
        "optimize_onnx": bool(optimize_onnx),
        "quantize_onnx": quantize_onnx,
        "optimized": False,
        "quantized": False,
        "fallback_to_source": False,
    }

    candidate_model_path = source_model_path
    optimized_path = destination_dir / ".model.optimized.onnx"
    quantized_path = destination_dir / ".model.quantized.onnx"

    if optimize_onnx:
        try:
            import onnxruntime as ort

            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            session_options.optimized_model_filepath = str(optimized_path)
            ort.InferenceSession(
                str(source_model_path),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            if optimized_path.is_file():
                candidate_model_path = optimized_path
                report["optimized"] = True
        except Exception as exc:  # pragma: no cover - depends on optional runtime.
            warnings.warn(f"ONNX optimization skipped: {type(exc).__name__}: {exc}")
            report["optimization_error"] = f"{type(exc).__name__}: {exc}"

    if quantize_onnx != "none":
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quantize_dynamic(
                model_input=str(candidate_model_path),
                model_output=str(quantized_path),
                weight_type=QuantType.QInt8,
                per_channel=(quantize_onnx == "arm64"),
                reduce_range=False,
            )
            if quantized_path.is_file():
                candidate_model_path = quantized_path
                report["quantized"] = True
        except Exception as exc:  # pragma: no cover - depends on optional runtime.
            warnings.warn(f"ONNX dynamic quantization skipped: {type(exc).__name__}: {exc}")
            report["quantization_error"] = f"{type(exc).__name__}: {exc}"

    shutil.copy2(candidate_model_path, destination_model_path)

    if candidate_model_path != source_model_path:
        try:
            raw_encoder = _make_encoder(
                model_path=source_model_path,
                tokenizer_path=source_tokenizer_path,
                max_length=max_length,
                pooling=pooling,
                output_name=output_name,
            )
            candidate_encoder = _make_encoder(
                model_path=destination_model_path,
                tokenizer_path=destination_dir / "tokenizer.json",
                max_length=max_length,
                pooling=pooling,
                output_name=output_name,
            )
            probe_batch_size = max(1, min(int(encode_batch_size), len(probe_texts)))
            raw_probe_embeddings = _encode_texts(raw_encoder, list(probe_texts), batch_size=probe_batch_size)
            candidate_probe_embeddings = _encode_texts(
                candidate_encoder, list(probe_texts), batch_size=probe_batch_size
            )
            mean_cosine = _compute_mean_cosine_similarity(raw_probe_embeddings, candidate_probe_embeddings)
            report["probe_mean_cosine"] = mean_cosine
            if not np.isfinite(mean_cosine) or mean_cosine < float(min_probe_cosine):
                warnings.warn(
                    "Optimized/quantized model drifted too far from the source encoder; "
                    "falling back to the original ONNX model."
                )
                shutil.copy2(source_model_path, destination_model_path)
                report["fallback_to_source"] = True
                report["fallback_reason"] = f"probe_mean_cosine={mean_cosine:.6f} < {min_probe_cosine:.6f}"
        except Exception as exc:
            warnings.warn(
                "Optimized/quantized model validation failed; falling back to the original ONNX model. "
                f"{type(exc).__name__}: {exc}"
            )
            shutil.copy2(source_model_path, destination_model_path)
            report["fallback_to_source"] = True
            report["fallback_reason"] = f"{type(exc).__name__}: {exc}"

    report["bundle_model_sha256"] = _sha256_file(destination_model_path)
    report["bundle_model_bytes"] = int(destination_model_path.stat().st_size)
    try:
        ort_path = _export_ort_sidecar(destination_model_path)
    except Exception as exc:  # pragma: no cover - depends on optional ORT tooling.
        warnings.warn(f"ORT export skipped: {type(exc).__name__}: {exc}")
        report["ort_sidecar_created"] = False
        report["ort_export_error"] = f"{type(exc).__name__}: {exc}"
    else:
        report["ort_sidecar_created"] = True
        report["ort_model_sha256"] = _sha256_file(ort_path)
        report["ort_model_bytes"] = int(ort_path.stat().st_size)
    for temporary_path in (optimized_path, quantized_path):
        if temporary_path.exists():
            temporary_path.unlink()
    return report


def _compute_linear_logits(embeddings: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64).reshape(-1)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, received shape {embeddings.shape}")
    if weights.ndim != 2:
        raise ValueError(f"Expected 2D weights, received shape {weights.shape}")

    if weights.shape[0] == bias.shape[0] and weights.shape[1] == embeddings.shape[1]:
        return embeddings @ weights.T + bias
    if weights.shape[1] == bias.shape[0] and weights.shape[0] == embeddings.shape[1]:
        return embeddings @ weights + bias
    raise ValueError(
        "Could not align linear-head weights, bias, and embeddings. "
        f"weights={weights.shape}, bias={bias.shape}, embeddings={embeddings.shape}"
    )


def _multiclass_log_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    log_probs = shifted - logsumexp
    return float(-np.mean(log_probs[np.arange(targets.shape[0]), targets]))


def _fit_temperature_from_logits(logits: np.ndarray, labels: Sequence[str]) -> tuple[float, dict[str, Any]]:
    if logits.ndim != 2 or logits.shape[0] == 0:
        return 1.0, {"applied": False, "reason": "empty_dev_logits"}

    label_to_index = {label: index for index, label in enumerate(ROUTE_LABEL_VALUES)}
    try:
        targets = np.asarray([label_to_index[label] for label in labels], dtype=np.int64)
    except KeyError as exc:
        return 1.0, {"applied": False, "reason": f"unknown_dev_label:{exc.args[0]}"}

    if np.unique(targets).shape[0] < 2:
        return 1.0, {"applied": False, "reason": "single_class_dev_split"}

    base_loss = _multiclass_log_loss(logits, targets)
    coarse_grid = np.exp(np.linspace(np.log(0.25), np.log(4.0), 41))
    best_temperature = 1.0
    best_loss = base_loss
    for temperature in coarse_grid:
        loss = _multiclass_log_loss(logits / temperature, targets)
        if loss < best_loss:
            best_temperature = float(temperature)
            best_loss = float(loss)

    fine_low = max(0.10, best_temperature / 1.5)
    fine_high = min(10.0, best_temperature * 1.5)
    fine_grid = np.exp(np.linspace(np.log(fine_low), np.log(fine_high), 61))
    for temperature in fine_grid:
        loss = _multiclass_log_loss(logits / temperature, targets)
        if loss < best_loss:
            best_temperature = float(temperature)
            best_loss = float(loss)

    return best_temperature, {
        "applied": abs(best_temperature - 1.0) > 1e-3 and best_loss + 1e-9 < base_loss,
        "temperature": float(best_temperature),
        "dev_log_loss_before": float(base_loss),
        "dev_log_loss_after": float(best_loss),
    }


def _swap_staged_directory(staged_path: Path, destination_path: Path) -> None:
    backup_path: Path | None = None
    if destination_path.exists():
        if destination_path.is_symlink():
            raise ValueError(f"Refusing to replace symlinked output path {destination_path}")
        backup_path = destination_path.with_name(f".{destination_path.name}.backup-{uuid.uuid4().hex}")
        os.replace(destination_path, backup_path)
    try:
        os.replace(staged_path, destination_path)
    except Exception:
        if backup_path is not None and backup_path.exists() and not destination_path.exists():
            os.replace(backup_path, destination_path)
        raise
    else:
        if backup_path is not None and backup_path.exists():
            shutil.rmtree(backup_path, ignore_errors=True)


def _build_report(
    *,
    source_path: Path,
    model_id: str,
    classifier_type: str,
    dataset_path: str | Path | None,
    split_strategy: str | None,
    train_samples: Sequence[LabeledRouteSample],
    dev_samples: Sequence[LabeledRouteSample],
    max_length: int,
    pooling: str,
    output_name: str | None,
    reference_date: str,
    model_report: Mapping[str, Any],
    temperature_report: Mapping[str, Any] | None,
    metadata: SemanticRouterBundleMetadata,
) -> dict[str, Any]:
    dataset_path_obj = Path(dataset_path).expanduser().resolve(strict=True) if dataset_path else None
    return {
        "schema_version": 2,
        "classifier_type": classifier_type,
        "model_id": model_id,
        "reference_date": reference_date,
        "source_dir": str(source_path),
        "dataset_path": str(dataset_path_obj) if dataset_path_obj else None,
        "dataset_sha256": _sha256_file(dataset_path_obj) if dataset_path_obj else None,
        "split_strategy": split_strategy,
        "train_count": len(train_samples),
        "dev_count": len(dev_samples),
        "train_label_counts": _label_counts(train_samples),
        "dev_label_counts": _label_counts(dev_samples),
        "max_length": int(max_length),
        "pooling": pooling,
        "output_name": output_name,
        "model_artifact": dict(model_report),
        "temperature_scaling": dict(temperature_report) if temperature_report else None,
        "policy": {
            "thresholds": dict(metadata.thresholds),
            "authoritative_labels": list(metadata.authoritative_labels),
            "min_margin": float(metadata.min_margin),
        },
        "versions": {
            "numpy": np.__version__,
            "scikit_learn": _safe_package_version("scikit-learn"),
            "onnxruntime": _safe_package_version("onnxruntime"),
        },
    }


def _common_metadata(
    *,
    classifier_type: str,
    model_id: str,
    max_length: int,
    pooling: str,
    output_name: str | None,
    normalize_centroids: bool,
    reference_date: str,
) -> SemanticRouterBundleMetadata:
    return SemanticRouterBundleMetadata(
        schema_version=1,
        classifier_type=classifier_type,
        labels=ROUTE_LABEL_VALUES,
        model_id=model_id,
        max_length=max_length,
        pooling=pooling,
        output_name=output_name,
        thresholds={"web": 0.6, "memory": 0.6, "tool": 0.6, "parametric": 1.0},
        authoritative_labels=("web", "memory", "tool"),
        min_margin=0.08,
        normalize_embeddings=True,
        normalize_centroids=normalize_centroids,
        reference_date=reference_date,
    )


def build_centroid_router_bundle(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    train_samples: Sequence[LabeledRouteSample],
    dev_samples: Sequence[LabeledRouteSample] = (),
    model_id: str | None = None,
    max_length: int = 128,
    pooling: str = "mean",
    output_name: str | None = None,
    reference_date: str = _DEFAULT_REFERENCE_DATE,
    dataset_path: str | Path | None = None,
    split_strategy: str | None = None,
    optimize_onnx: bool = True,
    quantize_onnx: str = "arm64",
    min_probe_cosine: float = _DEFAULT_MIN_PROBE_COSINE,
    encode_batch_size: int = _DEFAULT_ENCODE_BATCH_SIZE,
) -> Path:
    """Build one centroid-based router bundle from local source model files."""

    source_path, source_model_path, source_tokenizer_path = _validate_source_dir(source_dir)
    output_path = _validate_output_dir(source_path, output_dir)

    train_samples = tuple(train_samples)
    dev_samples = tuple(dev_samples)
    _validate_sample_payload(train_samples, context="train")
    if dev_samples:
        _validate_sample_payload(dev_samples, context="dev")
    _ensure_full_train_label_coverage(train_samples)
    _assert_no_train_dev_overlap(train_samples, dev_samples)

    probe_texts = _select_probe_texts(train_samples, dev_samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staged_path = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent))
    try:
        _copy_tokenizer_and_sidecars(source_path, staged_path)
        model_report = _materialize_model_artifact(
            source_model_path=source_model_path,
            source_tokenizer_path=source_tokenizer_path,
            destination_dir=staged_path,
            max_length=max_length,
            pooling=pooling,
            output_name=output_name,
            probe_texts=probe_texts,
            optimize_onnx=optimize_onnx,
            quantize_onnx=quantize_onnx,
            min_probe_cosine=min_probe_cosine,
            encode_batch_size=encode_batch_size,
        )
        encoder = _make_encoder(
            model_path=staged_path / "model.onnx",
            tokenizer_path=staged_path / "tokenizer.json",
            max_length=max_length,
            pooling=pooling,
            output_name=output_name,
        )
        train_embeddings = _encode_texts(
            encoder,
            [sample.text for sample in train_samples],
            batch_size=encode_batch_size,
        )
        centroids = np.asarray(compute_label_centroids(train_embeddings, train_samples), dtype=np.float32)

        metadata = _common_metadata(
            classifier_type="embedding_centroid_v1",
            model_id=model_id or source_path.name,
            max_length=max_length,
            pooling=pooling,
            output_name=output_name,
            normalize_centroids=True,
            reference_date=reference_date,
        )

        np.save(staged_path / "centroids.npy", centroids, allow_pickle=False)
        _write_router_metadata(staged_path / "router_metadata.json", metadata)

        if dev_samples:
            bundle = load_semantic_router_bundle(staged_path)
            router = LocalSemanticRouter(bundle, encoder=encoder, centroids=centroids)
            tuned_policy = tune_policy_thresholds(score_semantic_router(router, dev_samples))
            metadata = replace(
                metadata,
                thresholds=dict(tuned_policy.thresholds),
                authoritative_labels=tuned_policy.authoritative_labels,
                min_margin=tuned_policy.min_margin,
            )
            _write_router_metadata(staged_path / "router_metadata.json", metadata)

        report = _build_report(
            source_path=source_path,
            model_id=metadata.model_id,
            classifier_type=metadata.classifier_type,
            dataset_path=dataset_path,
            split_strategy=split_strategy,
            train_samples=train_samples,
            dev_samples=dev_samples,
            max_length=max_length,
            pooling=pooling,
            output_name=output_name,
            reference_date=reference_date,
            model_report=model_report,
            temperature_report=None,
            metadata=metadata,
        )
        _json_dump(staged_path / "build_report.json", report)
        load_semantic_router_bundle(staged_path)
        _swap_staged_directory(staged_path, output_path)
    except Exception:
        shutil.rmtree(staged_path, ignore_errors=True)
        raise
    return output_path


def build_linear_router_bundle(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    train_samples: Sequence[LabeledRouteSample],
    dev_samples: Sequence[LabeledRouteSample] = (),
    model_id: str | None = None,
    max_length: int = 128,
    pooling: str = "mean",
    output_name: str | None = None,
    reference_date: str = _DEFAULT_REFERENCE_DATE,
    dataset_path: str | Path | None = None,
    split_strategy: str | None = None,
    optimize_onnx: bool = True,
    quantize_onnx: str = "arm64",
    min_probe_cosine: float = _DEFAULT_MIN_PROBE_COSINE,
    encode_batch_size: int = _DEFAULT_ENCODE_BATCH_SIZE,
) -> Path:
    """Build one linear-head router bundle from local source model files."""

    source_path, source_model_path, source_tokenizer_path = _validate_source_dir(source_dir)
    output_path = _validate_output_dir(source_path, output_dir)

    train_samples = tuple(train_samples)
    dev_samples = tuple(dev_samples)
    _validate_sample_payload(train_samples, context="train")
    if dev_samples:
        _validate_sample_payload(dev_samples, context="dev")
    _ensure_full_train_label_coverage(train_samples)
    _assert_no_train_dev_overlap(train_samples, dev_samples)

    probe_texts = _select_probe_texts(train_samples, dev_samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staged_path = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent))
    try:
        _copy_tokenizer_and_sidecars(source_path, staged_path)
        model_report = _materialize_model_artifact(
            source_model_path=source_model_path,
            source_tokenizer_path=source_tokenizer_path,
            destination_dir=staged_path,
            max_length=max_length,
            pooling=pooling,
            output_name=output_name,
            probe_texts=probe_texts,
            optimize_onnx=optimize_onnx,
            quantize_onnx=quantize_onnx,
            min_probe_cosine=min_probe_cosine,
            encode_batch_size=encode_batch_size,
        )
        encoder = _make_encoder(
            model_path=staged_path / "model.onnx",
            tokenizer_path=staged_path / "tokenizer.json",
            max_length=max_length,
            pooling=pooling,
            output_name=output_name,
        )
        train_embeddings = _encode_texts(
            encoder,
            [sample.text for sample in train_samples],
            batch_size=encode_batch_size,
        )
        dev_embeddings = (
            _encode_texts(
                encoder,
                [sample.text for sample in dev_samples],
                batch_size=encode_batch_size,
            )
            if dev_samples
            else None
        )

        trained_head = fit_multiclass_linear_head(
            train_embeddings,
            [sample.label for sample in train_samples],
            label_order=ROUTE_LABEL_VALUES,
            dev_embeddings=dev_embeddings,
            dev_labels=[sample.label for sample in dev_samples] if dev_samples else None,
        )

        weights = np.asarray(trained_head.weights, dtype=np.float32)
        bias = np.asarray(trained_head.bias, dtype=np.float32).reshape(-1)

        temperature_report: dict[str, Any] | None = None
        if dev_embeddings is not None and len(dev_samples) > 0:
            logits = _compute_linear_logits(dev_embeddings.astype(np.float64), weights, bias)
            temperature, temperature_report = _fit_temperature_from_logits(
                logits,
                [sample.label for sample in dev_samples],
            )
            if temperature_report.get("applied"):
                weights = (weights / float(temperature)).astype(np.float32, copy=False)
                bias = (bias / float(temperature)).astype(np.float32, copy=False)

        metadata = _common_metadata(
            classifier_type="embedding_linear_softmax_v1",
            model_id=model_id or source_path.name,
            max_length=max_length,
            pooling=pooling,
            output_name=output_name,
            normalize_centroids=False,
            reference_date=reference_date,
        )

        np.save(staged_path / "weights.npy", weights, allow_pickle=False)
        np.save(staged_path / "bias.npy", bias, allow_pickle=False)
        _write_router_metadata(staged_path / "router_metadata.json", metadata)

        if dev_samples:
            bundle = load_semantic_router_bundle(staged_path)
            router = LocalSemanticRouter(bundle, encoder=encoder)
            tuned_policy = tune_policy_thresholds(score_semantic_router(router, dev_samples))
            metadata = replace(
                metadata,
                thresholds=dict(tuned_policy.thresholds),
                authoritative_labels=tuned_policy.authoritative_labels,
                min_margin=tuned_policy.min_margin,
            )
            _write_router_metadata(staged_path / "router_metadata.json", metadata)

        report = _build_report(
            source_path=source_path,
            model_id=metadata.model_id,
            classifier_type=metadata.classifier_type,
            dataset_path=dataset_path,
            split_strategy=split_strategy,
            train_samples=train_samples,
            dev_samples=dev_samples,
            max_length=max_length,
            pooling=pooling,
            output_name=output_name,
            reference_date=reference_date,
            model_report=model_report,
            temperature_report=temperature_report,
            metadata=metadata,
        )
        _json_dump(staged_path / "build_report.json", report)
        load_semantic_router_bundle(staged_path)
        _swap_staged_directory(staged_path, output_path)
    except Exception:
        shutil.rmtree(staged_path, ignore_errors=True)
        raise
    return output_path


def build_centroid_router_bundle_from_jsonl(
    *,
    source_dir: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    model_id: str | None = None,
    max_length: int = 128,
    pooling: str = "mean",
    output_name: str | None = None,
    reference_date: str = _DEFAULT_REFERENCE_DATE,
    auto_dev_split: bool = True,
    dev_ratio: float = _DEFAULT_AUTO_DEV_RATIO,
    seed: int = _DEFAULT_SEED,
    optimize_onnx: bool = True,
    quantize_onnx: str = "arm64",
    min_probe_cosine: float = _DEFAULT_MIN_PROBE_COSINE,
    encode_batch_size: int = _DEFAULT_ENCODE_BATCH_SIZE,
) -> Path:
    """Build one bundle from a JSONL dataset with explicit or inferred train/dev splits."""

    dataset_path = Path(dataset_path).expanduser().resolve(strict=True)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset path {dataset_path} is not a file.")
    samples = load_labeled_route_samples(dataset_path)
    train_samples, dev_samples, split_strategy = _resolve_train_dev_samples_from_jsonl(
        samples,
        auto_dev_split=auto_dev_split,
        dev_ratio=dev_ratio,
        seed=seed,
    )
    return build_centroid_router_bundle(
        source_dir=source_dir,
        output_dir=output_dir,
        train_samples=train_samples,
        dev_samples=dev_samples,
        model_id=model_id,
        max_length=max_length,
        pooling=pooling,
        output_name=output_name,
        reference_date=reference_date,
        dataset_path=dataset_path,
        split_strategy=split_strategy,
        optimize_onnx=optimize_onnx,
        quantize_onnx=quantize_onnx,
        min_probe_cosine=min_probe_cosine,
        encode_batch_size=encode_batch_size,
    )


def build_linear_router_bundle_from_jsonl(
    *,
    source_dir: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    model_id: str | None = None,
    max_length: int = 128,
    pooling: str = "mean",
    output_name: str | None = None,
    reference_date: str = _DEFAULT_REFERENCE_DATE,
    auto_dev_split: bool = True,
    dev_ratio: float = _DEFAULT_AUTO_DEV_RATIO,
    seed: int = _DEFAULT_SEED,
    optimize_onnx: bool = True,
    quantize_onnx: str = "arm64",
    min_probe_cosine: float = _DEFAULT_MIN_PROBE_COSINE,
    encode_batch_size: int = _DEFAULT_ENCODE_BATCH_SIZE,
) -> Path:
    """Build one linear-head router bundle from a JSONL dataset."""

    dataset_path = Path(dataset_path).expanduser().resolve(strict=True)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset path {dataset_path} is not a file.")
    samples = load_labeled_route_samples(dataset_path)
    train_samples, dev_samples, split_strategy = _resolve_train_dev_samples_from_jsonl(
        samples,
        auto_dev_split=auto_dev_split,
        dev_ratio=dev_ratio,
        seed=seed,
    )
    return build_linear_router_bundle(
        source_dir=source_dir,
        output_dir=output_dir,
        train_samples=train_samples,
        dev_samples=dev_samples,
        model_id=model_id,
        max_length=max_length,
        pooling=pooling,
        output_name=output_name,
        reference_date=reference_date,
        dataset_path=dataset_path,
        split_strategy=split_strategy,
        optimize_onnx=optimize_onnx,
        quantize_onnx=quantize_onnx,
        min_probe_cosine=min_probe_cosine,
        encode_batch_size=encode_batch_size,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Twinr semantic-router bundle.")
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
        default=_DEFAULT_REFERENCE_DATE,
        help="Frozen reference date for parametric-vs-web labeling semantics",
    )
    parser.add_argument(
        "--auto-dev-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When no explicit dev split exists, reserve a deterministic held-out calibration split when every label has >=2 samples.",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=_DEFAULT_AUTO_DEV_RATIO,
        help="Held-out dev fraction used only for automatic split inference.",
    )
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED, help="Deterministic seed for inferred train/dev splits.")
    parser.add_argument(
        "--optimize-onnx",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply offline ONNX Runtime graph optimization before writing model.onnx.",
    )
    parser.add_argument(
        "--quantize-onnx",
        default="arm64",
        choices=("none", "arm64", "generic"),
        help="Apply dynamic INT8 quantization to the staged ONNX model. `arm64` uses per-channel quantization.",
    )
    parser.add_argument(
        "--min-probe-cosine",
        type=float,
        default=_DEFAULT_MIN_PROBE_COSINE,
        help="Fallback to the original ONNX model if optimization/quantization reduces probe-text embedding cosine below this threshold.",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=_DEFAULT_ENCODE_BATCH_SIZE,
        help="Number of texts per encoder call during offline bundle construction.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for building local router bundles."""

    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    builder = (
        build_linear_router_bundle_from_jsonl
        if args.classifier == "linear"
        else build_centroid_router_bundle_from_jsonl
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
        auto_dev_split=args.auto_dev_split,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
        optimize_onnx=args.optimize_onnx,
        quantize_onnx=args.quantize_onnx,
        min_probe_cosine=args.min_probe_cosine,
        encode_batch_size=args.encode_batch_size,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via module CLI.
    raise SystemExit(main())
