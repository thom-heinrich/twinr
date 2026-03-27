# CHANGELOG: 2026-03-27
# BUG-1: reference_date is no longer a stale hard-coded constant; defaults are now resolved at runtime to today's ISO date.
# BUG-2: report serialization now deep-normalizes non-JSON-native values and guards empty splits, preventing silent/report-crashing failures.
# BUG-3: artifact writes are now atomic and protected by advisory file locks to avoid partial reports/bundles during concurrent runs.
# SEC-1: synthetic job parameters are now bounded by sane Raspberry-Pi-friendly limits, closing a practical disk/RAM exhaustion vector.
# IMP-1: default classifier selection is now AUTO (validation-driven centroid vs linear), with candidate comparison persisted in the report.
# IMP-2: reports now include artifact hashes, bundle sizes, duration, environment metadata, and optional confidence-aware calibration metrics.
# BREAKING: default classifier kind changed from "centroid" to "auto" so new runs may select a different backend than older releases.
# BREAKING: omitted --reference-date now resolves to the current date instead of the historical fixed date 2026-03-22.

"""Train and report synthetic-bootstrap semantic-router bundles.

This module orchestrates Twinr's offline semantic-router bootstrap workflow:
generate a synthetic JSONL corpus, build a local centroid or linear-head bundle
from an ONNX sentence model, score the resulting router on each dataset split,
and persist a compact machine-readable training report.

Example:

```bash
PYTHONPATH=src python3 -m twinr.agent.routing.training bootstrap-synthetic \
  --source-dir /models/paraphrase-multilingual-MiniLM-L12-v2-onnx \
  --work-dir artifacts/router/bootstrap
```
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, is_dataclass, replace
from datetime import date, datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
import uuid

try:  # pragma: no cover - fcntl is not available on all platforms.
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback.
    fcntl = None  # type: ignore[assignment]

from .bootstrap import (
    build_centroid_router_bundle_from_jsonl,
    build_linear_router_bundle_from_jsonl,
)
from .bundle import load_semantic_router_bundle
from .evaluation import (
    ScoredRouteRecord,
    evaluate_route_records,
    load_labeled_route_samples,
    score_semantic_router,
    split_labeled_route_samples,
)
from .service import LocalSemanticRouter
from .synthetic_corpus import (
    generate_synthetic_route_samples,
    generate_synthetic_user_intent_samples,
    write_synthetic_route_samples_jsonl,
    write_synthetic_user_intent_samples_jsonl,
)
from .two_stage import LocalUserIntentRouter, TwoStageLocalSemanticRouter
from .user_intent_bootstrap import (
    build_centroid_user_intent_bundle_from_jsonl,
    build_linear_user_intent_bundle_from_jsonl,
)
from .user_intent_bundle import load_user_intent_bundle
from .user_intent_evaluation import (
    ScoredUserIntentRecord,
    evaluate_user_intent_records,
    load_labeled_user_intent_samples,
    score_user_intent_router,
    split_labeled_user_intent_samples,
)


_BundleBuilder = Callable[..., Path]
_BundleLoader = Callable[[str | Path], Any]
_RouterFactory = Callable[[Any], Any]
_RouterScorer = Callable[[Any, Sequence[Any]], Sequence[Any]]
_RecordEvaluator = Callable[[Sequence[Any]], Any]

_REPORT_SCHEMA_VERSION = 2
_DEFAULT_LOCK_TIMEOUT_SECONDS = float(os.environ.get("TWINR_LOCK_TIMEOUT_SECONDS", "900"))
_DEFAULT_MAX_SAMPLES_PER_LABEL = int(os.environ.get("TWINR_MAX_SAMPLES_PER_LABEL", "32768"))
_DEFAULT_MAX_TEMPLATE_BUCKET = int(os.environ.get("TWINR_MAX_TEMPLATE_BUCKET", "8192"))
_SUPPORTED_CLASSIFIER_KINDS = frozenset({"centroid", "linear", "auto"})
_SUPPORTED_POOLING_KINDS = frozenset({"mean", "cls", "prepooled"})


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _today_reference_date() -> str:
    return date.today().isoformat()


def _normalize_reference_date(value: object | None) -> str:
    normalized = str(value or "").strip() or _today_reference_date()
    try:
        date.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"reference_date must be YYYY-MM-DD, got {value!r}") from exc
    return normalized


def _normalize_classifier_kind(value: object) -> str:
    """Return one validated offline classifier kind identifier."""

    normalized = str(value or "auto").strip().lower()
    if normalized not in _SUPPORTED_CLASSIFIER_KINDS:
        raise ValueError(f"Unsupported semantic-router classifier kind: {value!r}")
    return normalized


def _normalize_selection_metric(value: object) -> str:
    normalized = str(value or "frontier").strip().lower()
    if normalized not in {"frontier", "macro_f1", "accuracy"}:
        raise ValueError(f"Unsupported selection metric: {value!r}")
    return normalized


def _normalize_pooling(value: object) -> str:
    normalized = str(value or "mean").strip().lower()
    if normalized not in _SUPPORTED_POOLING_KINDS:
        raise ValueError(f"Unsupported pooling mode: {value!r}")
    return normalized


def _router_bundle_builder_for_classifier(kind: str) -> _BundleBuilder:
    """Return the backend-route bundle builder for one classifier kind."""

    normalized_kind = _normalize_classifier_kind(kind)
    if normalized_kind == "linear":
        return build_linear_router_bundle_from_jsonl
    if normalized_kind == "centroid":
        return build_centroid_router_bundle_from_jsonl
    raise ValueError("classifier_kind='auto' must be resolved before choosing a concrete builder")


def _user_intent_bundle_builder_for_classifier(kind: str) -> _BundleBuilder:
    """Return the user-intent bundle builder for one classifier kind."""

    normalized_kind = _normalize_classifier_kind(kind)
    if normalized_kind == "linear":
        return build_linear_user_intent_bundle_from_jsonl
    if normalized_kind == "centroid":
        return build_centroid_user_intent_bundle_from_jsonl
    raise ValueError("classifier_kind='auto' must be resolved before choosing a concrete builder")


@dataclass(frozen=True, slots=True)
class SemanticRouterTrainingReport:
    """Describe one offline semantic-router dataset and bundle training run."""

    dataset_path: Path
    bundle_dir: Path
    model_source_dir: Path
    model_id: str
    total_samples: int
    split_counts: dict[str, int]
    evaluations: dict[str, dict[str, object]]
    reference_date: str
    dataset_summary: dict[str, object] | None = None
    classifier_kind: str = "centroid"
    selection_metric: str | None = None
    selection_split: str | None = None
    candidate_reports: dict[str, dict[str, object]] | None = None
    report_schema_version: int = _REPORT_SCHEMA_VERSION
    created_at: str | None = None
    duration_seconds: float | None = None
    dataset_sha256: str | None = None
    bundle_sha256: str | None = None
    bundle_size_bytes: int | None = None
    dataset_size_bytes: int | None = None
    environment: dict[str, object] | None = None
    deployment_recommendations: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable training report."""

        payload: dict[str, object] = {
            "dataset_path": str(self.dataset_path),
            "bundle_dir": str(self.bundle_dir),
            "model_source_dir": str(self.model_source_dir),
            "model_id": self.model_id,
            "total_samples": int(self.total_samples),
            "split_counts": dict(self.split_counts),
            "evaluations": _json_safe(self.evaluations),
            "reference_date": self.reference_date,
            "classifier_kind": self.classifier_kind,
            "report_schema_version": int(self.report_schema_version),
        }
        if self.dataset_summary is not None:
            payload["dataset_summary"] = _json_safe(self.dataset_summary)
        if self.selection_metric is not None:
            payload["selection_metric"] = self.selection_metric
        if self.selection_split is not None:
            payload["selection_split"] = self.selection_split
        if self.candidate_reports is not None:
            payload["candidate_reports"] = _json_safe(self.candidate_reports)
        if self.created_at is not None:
            payload["created_at"] = self.created_at
        if self.duration_seconds is not None:
            payload["duration_seconds"] = float(self.duration_seconds)
        if self.dataset_sha256 is not None:
            payload["dataset_sha256"] = self.dataset_sha256
        if self.bundle_sha256 is not None:
            payload["bundle_sha256"] = self.bundle_sha256
        if self.bundle_size_bytes is not None:
            payload["bundle_size_bytes"] = int(self.bundle_size_bytes)
        if self.dataset_size_bytes is not None:
            payload["dataset_size_bytes"] = int(self.dataset_size_bytes)
        if self.environment is not None:
            payload["environment"] = _json_safe(self.environment)
        if self.deployment_recommendations is not None:
            payload["deployment_recommendations"] = _json_safe(self.deployment_recommendations)
        return payload


@dataclass(frozen=True, slots=True)
class UserIntentTrainingReport:
    """Describe one offline user-intent dataset and bundle training run."""

    dataset_path: Path
    bundle_dir: Path
    model_source_dir: Path
    model_id: str
    total_samples: int
    split_counts: dict[str, int]
    evaluations: dict[str, dict[str, object]]
    reference_date: str
    dataset_summary: dict[str, object] | None = None
    classifier_kind: str = "centroid"
    selection_metric: str | None = None
    selection_split: str | None = None
    candidate_reports: dict[str, dict[str, object]] | None = None
    report_schema_version: int = _REPORT_SCHEMA_VERSION
    created_at: str | None = None
    duration_seconds: float | None = None
    dataset_sha256: str | None = None
    bundle_sha256: str | None = None
    bundle_size_bytes: int | None = None
    dataset_size_bytes: int | None = None
    environment: dict[str, object] | None = None
    deployment_recommendations: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable training report."""

        payload: dict[str, object] = {
            "dataset_path": str(self.dataset_path),
            "bundle_dir": str(self.bundle_dir),
            "model_source_dir": str(self.model_source_dir),
            "model_id": self.model_id,
            "total_samples": int(self.total_samples),
            "split_counts": dict(self.split_counts),
            "evaluations": _json_safe(self.evaluations),
            "reference_date": self.reference_date,
            "classifier_kind": self.classifier_kind,
            "report_schema_version": int(self.report_schema_version),
        }
        if self.dataset_summary is not None:
            payload["dataset_summary"] = _json_safe(self.dataset_summary)
        if self.selection_metric is not None:
            payload["selection_metric"] = self.selection_metric
        if self.selection_split is not None:
            payload["selection_split"] = self.selection_split
        if self.candidate_reports is not None:
            payload["candidate_reports"] = _json_safe(self.candidate_reports)
        if self.created_at is not None:
            payload["created_at"] = self.created_at
        if self.duration_seconds is not None:
            payload["duration_seconds"] = float(self.duration_seconds)
        if self.dataset_sha256 is not None:
            payload["dataset_sha256"] = self.dataset_sha256
        if self.bundle_sha256 is not None:
            payload["bundle_sha256"] = self.bundle_sha256
        if self.bundle_size_bytes is not None:
            payload["bundle_size_bytes"] = int(self.bundle_size_bytes)
        if self.dataset_size_bytes is not None:
            payload["dataset_size_bytes"] = int(self.dataset_size_bytes)
        if self.environment is not None:
            payload["environment"] = _json_safe(self.environment)
        if self.deployment_recommendations is not None:
            payload["deployment_recommendations"] = _json_safe(self.deployment_recommendations)
        return payload


@dataclass(frozen=True, slots=True)
class TwoStageSemanticRouterTrainingReport:
    """Describe one synthetic bootstrap run that produces both router stages."""

    work_dir: Path
    model_source_dir: Path
    reference_date: str
    user_intent_dataset_path: Path
    backend_route_dataset_path: Path
    user_intent_bundle_dir: Path
    backend_route_bundle_dir: Path
    user_intent_report: UserIntentTrainingReport
    backend_route_report: SemanticRouterTrainingReport
    two_stage_backend_route_evaluations: dict[str, dict[str, object]]
    report_schema_version: int = _REPORT_SCHEMA_VERSION
    created_at: str | None = None
    duration_seconds: float | None = None
    work_dir_sha256: str | None = None
    environment: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable two-stage training report."""

        payload = {
            "work_dir": str(self.work_dir),
            "model_source_dir": str(self.model_source_dir),
            "reference_date": self.reference_date,
            "user_intent_dataset_path": str(self.user_intent_dataset_path),
            "backend_route_dataset_path": str(self.backend_route_dataset_path),
            "user_intent_bundle_dir": str(self.user_intent_bundle_dir),
            "backend_route_bundle_dir": str(self.backend_route_bundle_dir),
            "user_intent_report": self.user_intent_report.to_dict(),
            "backend_route_report": self.backend_route_report.to_dict(),
            "two_stage_backend_route_evaluations": _json_safe(self.two_stage_backend_route_evaluations),
            "report_schema_version": int(self.report_schema_version),
        }
        if self.created_at is not None:
            payload["created_at"] = self.created_at
        if self.duration_seconds is not None:
            payload["duration_seconds"] = float(self.duration_seconds)
        if self.work_dir_sha256 is not None:
            payload["work_dir_sha256"] = self.work_dir_sha256
        if self.environment is not None:
            payload["environment"] = _json_safe(self.environment)
        return payload


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return _json_safe(value.__dict__ if hasattr(value, "__dict__") else {field: getattr(value, field) for field in value.__dataclass_fields__})
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except Exception:
            return str(value)
    return str(value)


def _resolved_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _require_existing_dir(value: str | Path, *, label: str) -> Path:
    path = _resolved_path(value)
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")
    return path


def _require_existing_file(value: str | Path, *, label: str) -> Path:
    path = _resolved_path(value)
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def _normalize_model_id(model_id: str | None, *, fallback: str) -> str:
    normalized = str(model_id or "").strip()
    return normalized or str(fallback).strip() or "model"


def _validate_generation_args(
    *,
    samples_per_label: int,
    oversample_factor: float,
    max_near_duplicate_similarity: float,
    max_samples_per_template_bucket: int,
) -> None:
    if not 1 <= int(samples_per_label) <= _DEFAULT_MAX_SAMPLES_PER_LABEL:
        raise ValueError(
            "samples_per_label must be between 1 and "
            f"{_DEFAULT_MAX_SAMPLES_PER_LABEL} on this installation; "
            "override TWINR_MAX_SAMPLES_PER_LABEL if you intentionally need larger jobs"
        )
    if not 1.0 <= float(oversample_factor) <= 10.0:
        raise ValueError("oversample_factor must be between 1.0 and 10.0")
    if not 0.0 <= float(max_near_duplicate_similarity) <= 1.0:
        raise ValueError("max_near_duplicate_similarity must be between 0.0 and 1.0")
    if not 1 <= int(max_samples_per_template_bucket) <= _DEFAULT_MAX_TEMPLATE_BUCKET:
        raise ValueError(
            "max_samples_per_template_bucket must be between 1 and "
            f"{_DEFAULT_MAX_TEMPLATE_BUCKET} on this installation; "
            "override TWINR_MAX_TEMPLATE_BUCKET if you intentionally need larger jobs"
        )


def _validate_max_length(value: int) -> int:
    normalized = int(value)
    if not 1 <= normalized <= 4096:
        raise ValueError("max_length must be between 1 and 4096")
    return normalized


@contextmanager
def _advisory_file_lock(lock_path: str | Path, *, timeout_seconds: float = _DEFAULT_LOCK_TIMEOUT_SECONDS) -> Iterator[None]:
    path = _resolved_path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as lock_file:
        if fcntl is None:  # pragma: no cover - non-posix fallback.
            yield
            return
        started = time.monotonic()
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - started >= timeout_seconds:
                    raise TimeoutError(f"Timed out acquiring file lock: {path}")
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _atomic_write_text(path: str | Path, content: str, *, encoding: str = "utf-8") -> Path:
    target = _resolved_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding=encoding,
        dir=target.parent,
        prefix=f".{target.name}.tmp.",
        suffix=".partial",
        delete=False,
    ) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    os.replace(temp_path, target)
    return target


def _path_size_bytes(path: str | Path) -> int:
    resolved = _resolved_path(path)
    if not resolved.exists():
        return 0
    if resolved.is_file():
        return resolved.stat().st_size
    total = 0
    for child in sorted(resolved.rglob("*")):
        if child.is_file():
            total += child.stat().st_size
    return total


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: str | Path) -> str:
    root = _resolved_path(path)
    digest = hashlib.sha256()
    if root.is_file():
        digest.update(root.name.encode("utf-8"))
        digest.update(_sha256_file(root).encode("utf-8"))
        return digest.hexdigest()
    for child in sorted(root.rglob("*")):
        relative = child.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        if child.is_file():
            with open(child, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _promote_path_atomically(source: str | Path, destination: str | Path) -> Path:
    src = _resolved_path(source)
    dst = _resolved_path(destination)
    if not src.exists():
        raise FileNotFoundError(f"Cannot promote missing artifact: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    backup_path = dst.with_name(f".{dst.name}.bak.{uuid.uuid4().hex}")
    if dst.exists():
        os.replace(dst, backup_path)
    os.replace(src, dst)
    if backup_path.exists():
        if backup_path.is_dir():
            shutil.rmtree(backup_path, ignore_errors=True)
        else:
            backup_path.unlink(missing_ok=True)
    return dst


def _write_json_report(path: str | Path, payload: Mapping[str, object]) -> Path:
    return _atomic_write_text(path, json.dumps(_json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _capture_environment() -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count() or 1,
    }


def _edge_deployment_recommendations(*, pooling: str, reference_date: str) -> dict[str, object]:
    cpu_count = max(int(os.cpu_count() or 1), 1)
    return {
        "reference_date": reference_date,
        "onnxruntime": {
            "graph_optimization_level": "ORT_ENABLE_ALL",
            "inter_op_num_threads": 1,
            "intra_op_num_threads": min(cpu_count, 4),
            "notes": [
                "Consider dynamic INT8 quantization for CPU-bound edge inference.",
                "Consider ORT format export for reduced-size ONNX Runtime deployments.",
            ],
        },
        "embedding_model": {
            "pooling": pooling,
            "notes": [
                "Preserve the exact pooling/normalization contract used by the embedding model.",
                "If the model family supports Matryoshka truncation, benchmark reduced embedding dimensions.",
                "For future bundle refreshes, consider hard-negative mining before retraining the linear head.",
            ],
        },
    }


def _extract_sample_label(sample: Any) -> str | None:
    for attr in ("label", "route", "intent", "user_intent", "backend_route", "target"):
        value = getattr(sample, attr, None)
        if value is not None:
            return str(value)
    if isinstance(sample, Mapping):
        for key in ("label", "route", "intent", "user_intent", "backend_route", "target"):
            if key in sample and sample[key] is not None:
                return str(sample[key])
    return None


def _extract_prediction_label(decision: Any) -> str | None:
    for attr in (
        "label",
        "route",
        "intent",
        "user_intent",
        "backend_route",
        "selected_label",
        "predicted_label",
        "route_name",
        "name",
    ):
        value = getattr(decision, attr, None)
        if value is not None and not callable(value):
            return str(value)
    if isinstance(decision, Mapping):
        for key in (
            "label",
            "route",
            "intent",
            "user_intent",
            "backend_route",
            "selected_label",
            "predicted_label",
            "route_name",
            "name",
        ):
            if key in decision and decision[key] is not None:
                return str(decision[key])
    return None


def _extract_confidence(decision: Any) -> float | None:
    candidate_fields = (
        "confidence",
        "probability",
        "score",
        "calibrated_confidence",
        "max_probability",
        "max_score",
        "similarity",
    )
    for attr in candidate_fields:
        raw = getattr(decision, attr, None)
        if raw is None and isinstance(decision, Mapping):
            raw = decision.get(attr)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if 0.0 <= value <= 1.0:
            return value
        if attr == "similarity" and -1.0 <= value <= 1.0:
            return (value + 1.0) / 2.0
    return None


def _compute_confidence_metrics(records: Sequence[Any]) -> dict[str, object] | None:
    confidences: list[float] = []
    correctness: list[float] = []
    for record in records:
        sample = getattr(record, "sample", None)
        decision = getattr(record, "decision", None)
        if sample is None or decision is None:
            continue
        true_label = _extract_sample_label(sample)
        predicted_label = _extract_prediction_label(decision)
        confidence = _extract_confidence(decision)
        if true_label is None or predicted_label is None or confidence is None:
            continue
        confidences.append(confidence)
        correctness.append(1.0 if predicted_label == true_label else 0.0)
    if len(confidences) < 10:
        return None
    pairs = sorted(zip(confidences, correctness), key=lambda item: item[0])
    brier = sum((confidence - correct) ** 2 for confidence, correct in pairs) / len(pairs)
    num_bins = 10
    ece = 0.0
    for bin_index in range(num_bins):
        left = bin_index / num_bins
        right = (bin_index + 1) / num_bins
        bucket = [(confidence, correct) for confidence, correct in pairs if left <= confidence < right or (bin_index == num_bins - 1 and confidence == 1.0)]
        if not bucket:
            continue
        bucket_confidence = sum(confidence for confidence, _ in bucket) / len(bucket)
        bucket_accuracy = sum(correct for _, correct in bucket) / len(bucket)
        ece += abs(bucket_confidence - bucket_accuracy) * (len(bucket) / len(pairs))
    ranked = sorted(zip(confidences, correctness), key=lambda item: item[0], reverse=True)
    cumulative_errors = 0.0
    aurc = 0.0
    coverage_points: dict[str, float] = {}
    thresholds = (0.50, 0.70, 0.80, 0.90)
    for index, (confidence, correct) in enumerate(ranked, start=1):
        cumulative_errors += 1.0 - correct
        aurc += cumulative_errors / index
        for threshold in thresholds:
            key = f"coverage_at_{threshold:.2f}"
            if key not in coverage_points and confidence < threshold:
                coverage_points[key] = (index - 1) / len(ranked)
    for threshold in thresholds:
        key = f"coverage_at_{threshold:.2f}"
        coverage_points.setdefault(key, 1.0)
    aurc /= len(ranked)
    return {
        "supported_records": len(confidences),
        "brier_score": brier,
        "ece_10_bin": ece,
        "aurc": aurc,
        **coverage_points,
    }


def _empty_route_evaluation() -> dict[str, object]:
    return {
        "total": 0,
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "fallback_rate": 1.0,
        "authoritative_rate": 0.0,
        "unsafe_authoritative_error_rate": 0.0,
        "confusion_matrix": {},
        "per_label": {},
    }


def _empty_user_intent_evaluation() -> dict[str, object]:
    return {
        "total": 0,
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "confusion_matrix": {},
        "per_label": {},
    }


def _evaluation_to_dict(summary: Any, *, records: Sequence[Any] | None = None) -> dict[str, object]:
    payload = {
        "total": int(summary.total),
        "accuracy": float(summary.accuracy),
        "macro_f1": float(summary.macro_f1),
        "fallback_rate": float(summary.fallback_rate),
        "authoritative_rate": float(summary.authoritative_rate),
        "unsafe_authoritative_error_rate": float(summary.unsafe_authoritative_error_rate),
        "confusion_matrix": _json_safe(dict(summary.confusion_matrix)),
        "per_label": _json_safe(dict(summary.per_label)),
    }
    if records is not None:
        confidence_metrics = _compute_confidence_metrics(records)
        if confidence_metrics is not None:
            payload["confidence_metrics"] = _json_safe(confidence_metrics)
    return payload


def _user_evaluation_to_dict(summary: Any, *, records: Sequence[Any] | None = None) -> dict[str, object]:
    payload = {
        "total": int(summary.total),
        "accuracy": float(summary.accuracy),
        "macro_f1": float(summary.macro_f1),
        "confusion_matrix": _json_safe(dict(summary.confusion_matrix)),
        "per_label": _json_safe(dict(summary.per_label)),
    }
    if records is not None:
        confidence_metrics = _compute_confidence_metrics(records)
        if confidence_metrics is not None:
            payload["confidence_metrics"] = _json_safe(confidence_metrics)
    return payload


def _stage_model_id(model_id_prefix: str | None, *, suffix: str, fallback: str) -> str:
    base = str(model_id_prefix or fallback).strip()
    return f"{base}_{suffix}"


def _candidate_selection_split(evaluations: Mapping[str, Mapping[str, object]]) -> str | None:
    preferred = ("validation", "val", "dev", "development", "test", "train")
    for split_name in preferred:
        if split_name in evaluations and int(evaluations[split_name].get("total", 0)) > 0:
            return split_name
    for split_name, summary in evaluations.items():
        if int(summary.get("total", 0)) > 0:
            return split_name
    return None


def _candidate_rank(
    report: SemanticRouterTrainingReport | UserIntentTrainingReport,
    *,
    selection_metric: str,
) -> tuple[float, ...]:
    split_name = _candidate_selection_split(report.evaluations)
    if split_name is None:
        return (-1.0, -1.0, -1.0, -1.0, float("-inf"))
    evaluation = report.evaluations[split_name]
    macro_f1 = float(evaluation.get("macro_f1", 0.0))
    accuracy = float(evaluation.get("accuracy", 0.0))
    bundle_penalty = -float(report.bundle_size_bytes or 0)
    if selection_metric == "accuracy":
        return (accuracy, macro_f1, bundle_penalty)
    if selection_metric == "macro_f1":
        return (macro_f1, accuracy, bundle_penalty)
    unsafe_error = float(evaluation.get("unsafe_authoritative_error_rate", 0.0))
    fallback_rate = float(evaluation.get("fallback_rate", 0.0))
    aurc = float(evaluation.get("confidence_metrics", {}).get("aurc", 1.0)) if isinstance(evaluation.get("confidence_metrics"), Mapping) else 1.0
    return (
        macro_f1,
        accuracy,
        1.0 - unsafe_error,
        -aurc,
        -fallback_rate,
        bundle_penalty,
    )


def _write_route_samples_jsonl_atomic(samples: Sequence[Any], destination: str | Path) -> Path:
    target = _resolved_path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.parent / f".{target.name}.tmp.{uuid.uuid4().hex}.partial"
    write_synthetic_route_samples_jsonl(samples, temp_path)
    return _promote_path_atomically(temp_path, target)


def _write_user_intent_samples_jsonl_atomic(samples: Sequence[Any], destination: str | Path) -> Path:
    target = _resolved_path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.parent / f".{target.name}.tmp.{uuid.uuid4().hex}.partial"
    write_synthetic_user_intent_samples_jsonl(samples, temp_path)
    return _promote_path_atomically(temp_path, target)


def _bundle_lock_path(output_dir: str | Path) -> Path:
    output_path = _resolved_path(output_dir)
    return output_path.parent / f".{output_path.name}.lock"


def _run_router_training_once(
    *,
    source_dir: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    model_id: str | None,
    max_length: int,
    pooling: str,
    output_name: str | None,
    reference_date: str,
    classifier_kind: str,
    dataset_summary: dict[str, object] | None,
    bundle_builder: _BundleBuilder,
    bundle_loader: _BundleLoader,
    router_factory: _RouterFactory,
    router_scorer: _RouterScorer,
    record_evaluator: _RecordEvaluator,
) -> SemanticRouterTrainingReport:
    started = time.perf_counter()
    source_path = _require_existing_dir(source_dir, label="source_dir")
    dataset_file = _require_existing_file(dataset_path, label="dataset_path")
    output_root = _resolved_path(output_dir)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_output_dir = output_root.parent / f".{output_root.name}.candidate.{uuid.uuid4().hex}"
    resolved_bundle_builder = _router_bundle_builder_for_classifier(classifier_kind) if bundle_builder is build_centroid_router_bundle_from_jsonl else bundle_builder
    bundle_dir = Path(
        resolved_bundle_builder(
            source_dir=source_path,
            dataset_path=dataset_file,
            output_dir=temp_output_dir,
            model_id=model_id,
            max_length=max_length,
            pooling=pooling,
            output_name=output_name,
            reference_date=reference_date,
        )
    ).expanduser().resolve(strict=False)
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle builder did not create bundle_dir: {bundle_dir}")
    if not _path_is_relative_to(bundle_dir, temp_output_dir):
        raise RuntimeError(
            "Bundle builder returned a path outside its temporary output directory, which is unsafe for atomic promotion: "
            f"{bundle_dir} not under {temp_output_dir}"
        )
    samples = load_labeled_route_samples(dataset_file)
    grouped_samples = split_labeled_route_samples(samples)
    bundle = bundle_loader(bundle_dir)
    router = router_factory(bundle)
    evaluations: dict[str, dict[str, object]] = {}
    split_counts: dict[str, int] = {}
    for split_name, split_samples in grouped_samples.items():
        split_samples_tuple = tuple(split_samples)
        split_counts[split_name] = len(split_samples_tuple)
        if not split_samples_tuple:
            evaluations[split_name] = _empty_route_evaluation()
            continue
        records = tuple(router_scorer(router, split_samples_tuple))
        summary = record_evaluator(records)
        evaluations[split_name] = _evaluation_to_dict(summary, records=records)
    final_bundle_dir = _promote_path_atomically(bundle_dir, output_root)
    if temp_output_dir.exists():
        shutil.rmtree(temp_output_dir, ignore_errors=True)
    return SemanticRouterTrainingReport(
        dataset_path=dataset_file,
        bundle_dir=final_bundle_dir,
        model_source_dir=source_path,
        model_id=_normalize_model_id(model_id, fallback=source_path.name),
        total_samples=len(samples),
        split_counts=split_counts,
        evaluations=evaluations,
        reference_date=reference_date,
        dataset_summary=dataset_summary,
        classifier_kind=classifier_kind,
        created_at=_now_utc_iso(),
        duration_seconds=time.perf_counter() - started,
        dataset_sha256=_sha256_file(dataset_file),
        bundle_sha256=_sha256_tree(final_bundle_dir),
        bundle_size_bytes=_path_size_bytes(final_bundle_dir),
        dataset_size_bytes=_path_size_bytes(dataset_file),
        environment=_capture_environment(),
        deployment_recommendations=_edge_deployment_recommendations(pooling=pooling, reference_date=reference_date),
    )


def _run_user_intent_training_once(
    *,
    source_dir: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    model_id: str | None,
    max_length: int,
    pooling: str,
    output_name: str | None,
    reference_date: str,
    classifier_kind: str,
    dataset_summary: dict[str, object] | None,
    bundle_builder: _BundleBuilder,
    bundle_loader: _BundleLoader,
    router_factory: _RouterFactory,
    router_scorer: _RouterScorer,
    record_evaluator: _RecordEvaluator,
) -> UserIntentTrainingReport:
    started = time.perf_counter()
    source_path = _require_existing_dir(source_dir, label="source_dir")
    dataset_file = _require_existing_file(dataset_path, label="dataset_path")
    output_root = _resolved_path(output_dir)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_output_dir = output_root.parent / f".{output_root.name}.candidate.{uuid.uuid4().hex}"
    resolved_bundle_builder = _user_intent_bundle_builder_for_classifier(classifier_kind) if bundle_builder is build_centroid_user_intent_bundle_from_jsonl else bundle_builder
    bundle_dir = Path(
        resolved_bundle_builder(
            source_dir=source_path,
            dataset_path=dataset_file,
            output_dir=temp_output_dir,
            model_id=model_id,
            max_length=max_length,
            pooling=pooling,
            output_name=output_name,
            reference_date=reference_date,
        )
    ).expanduser().resolve(strict=False)
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle builder did not create bundle_dir: {bundle_dir}")
    if not _path_is_relative_to(bundle_dir, temp_output_dir):
        raise RuntimeError(
            "Bundle builder returned a path outside its temporary output directory, which is unsafe for atomic promotion: "
            f"{bundle_dir} not under {temp_output_dir}"
        )
    samples = load_labeled_user_intent_samples(dataset_file)
    grouped_samples = split_labeled_user_intent_samples(samples)
    bundle = bundle_loader(bundle_dir)
    router = router_factory(bundle)
    evaluations: dict[str, dict[str, object]] = {}
    split_counts: dict[str, int] = {}
    for split_name, split_samples in grouped_samples.items():
        split_samples_tuple = tuple(split_samples)
        split_counts[split_name] = len(split_samples_tuple)
        if not split_samples_tuple:
            evaluations[split_name] = _empty_user_intent_evaluation()
            continue
        records = tuple(router_scorer(router, split_samples_tuple))
        summary = record_evaluator(records)
        evaluations[split_name] = _user_evaluation_to_dict(summary, records=records)
    final_bundle_dir = _promote_path_atomically(bundle_dir, output_root)
    if temp_output_dir.exists():
        shutil.rmtree(temp_output_dir, ignore_errors=True)
    return UserIntentTrainingReport(
        dataset_path=dataset_file,
        bundle_dir=final_bundle_dir,
        model_source_dir=source_path,
        model_id=_normalize_model_id(model_id, fallback=source_path.name),
        total_samples=len(samples),
        split_counts=split_counts,
        evaluations=evaluations,
        reference_date=reference_date,
        dataset_summary=dataset_summary,
        classifier_kind=classifier_kind,
        created_at=_now_utc_iso(),
        duration_seconds=time.perf_counter() - started,
        dataset_sha256=_sha256_file(dataset_file),
        bundle_sha256=_sha256_tree(final_bundle_dir),
        bundle_size_bytes=_path_size_bytes(final_bundle_dir),
        dataset_size_bytes=_path_size_bytes(dataset_file),
        environment=_capture_environment(),
        deployment_recommendations=_edge_deployment_recommendations(pooling=pooling, reference_date=reference_date),
    )


def _write_training_report_if_requested(report_path: str | Path | None, report: SemanticRouterTrainingReport | UserIntentTrainingReport) -> None:
    if report_path is None:
        return
    _write_json_report(report_path, report.to_dict())


def train_router_bundle_from_jsonl(
    *,
    source_dir: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    report_path: str | Path | None = None,
    model_id: str | None = None,
    max_length: int = 128,
    pooling: str = "mean",
    output_name: str | None = None,
    reference_date: str | None = None,
    classifier_kind: str = "auto",
    selection_metric: str = "frontier",
    dataset_summary: dict[str, object] | None = None,
    bundle_builder: _BundleBuilder = build_centroid_router_bundle_from_jsonl,
    bundle_loader: _BundleLoader = load_semantic_router_bundle,
    router_factory: _RouterFactory = LocalSemanticRouter,
    router_scorer: _RouterScorer = score_semantic_router,
    record_evaluator: _RecordEvaluator = evaluate_route_records,
) -> SemanticRouterTrainingReport:
    """Build one router bundle from JSONL and score it on every split."""

    normalized_classifier = _normalize_classifier_kind(classifier_kind)
    normalized_selection_metric = _normalize_selection_metric(selection_metric)
    normalized_pooling = _normalize_pooling(pooling)
    normalized_reference_date = _normalize_reference_date(reference_date)
    normalized_max_length = _validate_max_length(max_length)
    output_root = _resolved_path(output_dir)
    lock_path = _bundle_lock_path(output_root)
    with _advisory_file_lock(lock_path):
        if normalized_classifier != "auto":
            report = _run_router_training_once(
                source_dir=source_dir,
                dataset_path=dataset_path,
                output_dir=output_root,
                model_id=model_id,
                max_length=normalized_max_length,
                pooling=normalized_pooling,
                output_name=output_name,
                reference_date=normalized_reference_date,
                classifier_kind=normalized_classifier,
                dataset_summary=dataset_summary,
                bundle_builder=bundle_builder,
                bundle_loader=bundle_loader,
                router_factory=router_factory,
                router_scorer=router_scorer,
                record_evaluator=record_evaluator,
            )
            _write_training_report_if_requested(report_path, report)
            return report

        candidate_reports: dict[str, SemanticRouterTrainingReport] = {}
        candidate_failures: dict[str, dict[str, object]] = {}
        for candidate_kind in ("centroid", "linear"):
            candidate_output_dir = output_root.parent / f".{output_root.name}.{candidate_kind}.selected"
            if candidate_output_dir.exists():
                if candidate_output_dir.is_dir():
                    shutil.rmtree(candidate_output_dir, ignore_errors=True)
                else:
                    candidate_output_dir.unlink(missing_ok=True)
            try:
                candidate_reports[candidate_kind] = _run_router_training_once(
                    source_dir=source_dir,
                    dataset_path=dataset_path,
                    output_dir=candidate_output_dir,
                    model_id=model_id,
                    max_length=normalized_max_length,
                    pooling=normalized_pooling,
                    output_name=output_name,
                    reference_date=normalized_reference_date,
                    classifier_kind=candidate_kind,
                    dataset_summary=dataset_summary,
                    bundle_builder=bundle_builder,
                    bundle_loader=bundle_loader,
                    router_factory=router_factory,
                    router_scorer=router_scorer,
                    record_evaluator=record_evaluator,
                )
            except Exception as exc:
                candidate_failures[candidate_kind] = {
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                }
        if not candidate_reports:
            raise RuntimeError(f"All router candidates failed: {candidate_failures!r}")
        selected_kind, selected_report = max(
            candidate_reports.items(),
            key=lambda item: _candidate_rank(item[1], selection_metric=normalized_selection_metric),
        )
        for candidate_kind, candidate_report in candidate_reports.items():
            if candidate_kind == selected_kind:
                continue
            if candidate_report.bundle_dir.exists():
                shutil.rmtree(candidate_report.bundle_dir, ignore_errors=True)
        final_bundle_dir = _promote_path_atomically(selected_report.bundle_dir, output_root)
        final_bundle_sha256 = _sha256_tree(final_bundle_dir)
        final_bundle_size_bytes = _path_size_bytes(final_bundle_dir)
        candidate_payloads = {
            kind: (
                replace(
                    report,
                    bundle_dir=final_bundle_dir,
                    bundle_sha256=final_bundle_sha256,
                    bundle_size_bytes=final_bundle_size_bytes,
                ).to_dict()
                if kind == selected_kind
                else report.to_dict()
            )
            for kind, report in candidate_reports.items()
        }
        candidate_payloads.update(candidate_failures)
        final_report = replace(
            selected_report,
            bundle_dir=final_bundle_dir,
            classifier_kind=selected_kind,
            selection_metric=normalized_selection_metric,
            selection_split=_candidate_selection_split(selected_report.evaluations),
            candidate_reports=candidate_payloads,
            bundle_sha256=final_bundle_sha256,
            bundle_size_bytes=final_bundle_size_bytes,
        )
        _write_training_report_if_requested(report_path, final_report)
        return final_report


def train_user_intent_bundle_from_jsonl(
    *,
    source_dir: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    report_path: str | Path | None = None,
    model_id: str | None = None,
    max_length: int = 128,
    pooling: str = "mean",
    output_name: str | None = None,
    reference_date: str | None = None,
    classifier_kind: str = "auto",
    selection_metric: str = "frontier",
    dataset_summary: dict[str, object] | None = None,
    bundle_builder: _BundleBuilder = build_centroid_user_intent_bundle_from_jsonl,
    bundle_loader: _BundleLoader = load_user_intent_bundle,
    router_factory: _RouterFactory = LocalUserIntentRouter,
    router_scorer: _RouterScorer = score_user_intent_router,
    record_evaluator: _RecordEvaluator = evaluate_user_intent_records,
) -> UserIntentTrainingReport:
    """Build one user-intent bundle from JSONL and score it on every split."""

    normalized_classifier = _normalize_classifier_kind(classifier_kind)
    normalized_selection_metric = _normalize_selection_metric(selection_metric)
    normalized_pooling = _normalize_pooling(pooling)
    normalized_reference_date = _normalize_reference_date(reference_date)
    normalized_max_length = _validate_max_length(max_length)
    output_root = _resolved_path(output_dir)
    lock_path = _bundle_lock_path(output_root)
    with _advisory_file_lock(lock_path):
        if normalized_classifier != "auto":
            report = _run_user_intent_training_once(
                source_dir=source_dir,
                dataset_path=dataset_path,
                output_dir=output_root,
                model_id=model_id,
                max_length=normalized_max_length,
                pooling=normalized_pooling,
                output_name=output_name,
                reference_date=normalized_reference_date,
                classifier_kind=normalized_classifier,
                dataset_summary=dataset_summary,
                bundle_builder=bundle_builder,
                bundle_loader=bundle_loader,
                router_factory=router_factory,
                router_scorer=router_scorer,
                record_evaluator=record_evaluator,
            )
            _write_training_report_if_requested(report_path, report)
            return report

        candidate_reports: dict[str, UserIntentTrainingReport] = {}
        candidate_failures: dict[str, dict[str, object]] = {}
        for candidate_kind in ("centroid", "linear"):
            candidate_output_dir = output_root.parent / f".{output_root.name}.{candidate_kind}.selected"
            if candidate_output_dir.exists():
                if candidate_output_dir.is_dir():
                    shutil.rmtree(candidate_output_dir, ignore_errors=True)
                else:
                    candidate_output_dir.unlink(missing_ok=True)
            try:
                candidate_reports[candidate_kind] = _run_user_intent_training_once(
                    source_dir=source_dir,
                    dataset_path=dataset_path,
                    output_dir=candidate_output_dir,
                    model_id=model_id,
                    max_length=normalized_max_length,
                    pooling=normalized_pooling,
                    output_name=output_name,
                    reference_date=normalized_reference_date,
                    classifier_kind=candidate_kind,
                    dataset_summary=dataset_summary,
                    bundle_builder=bundle_builder,
                    bundle_loader=bundle_loader,
                    router_factory=router_factory,
                    router_scorer=router_scorer,
                    record_evaluator=record_evaluator,
                )
            except Exception as exc:
                candidate_failures[candidate_kind] = {
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                }
        if not candidate_reports:
            raise RuntimeError(f"All user-intent candidates failed: {candidate_failures!r}")
        selected_kind, selected_report = max(
            candidate_reports.items(),
            key=lambda item: _candidate_rank(item[1], selection_metric=normalized_selection_metric),
        )
        for candidate_kind, candidate_report in candidate_reports.items():
            if candidate_kind == selected_kind:
                continue
            if candidate_report.bundle_dir.exists():
                shutil.rmtree(candidate_report.bundle_dir, ignore_errors=True)
        final_bundle_dir = _promote_path_atomically(selected_report.bundle_dir, output_root)
        final_bundle_sha256 = _sha256_tree(final_bundle_dir)
        final_bundle_size_bytes = _path_size_bytes(final_bundle_dir)
        candidate_payloads = {
            kind: (
                replace(
                    report,
                    bundle_dir=final_bundle_dir,
                    bundle_sha256=final_bundle_sha256,
                    bundle_size_bytes=final_bundle_size_bytes,
                ).to_dict()
                if kind == selected_kind
                else report.to_dict()
            )
            for kind, report in candidate_reports.items()
        }
        candidate_payloads.update(candidate_failures)
        final_report = replace(
            selected_report,
            bundle_dir=final_bundle_dir,
            classifier_kind=selected_kind,
            selection_metric=normalized_selection_metric,
            selection_split=_candidate_selection_split(selected_report.evaluations),
            candidate_reports=candidate_payloads,
            bundle_sha256=final_bundle_sha256,
            bundle_size_bytes=final_bundle_size_bytes,
        )
        _write_training_report_if_requested(report_path, final_report)
        return final_report


def bootstrap_synthetic_router_bundle(
    *,
    source_dir: str | Path,
    work_dir: str | Path,
    samples_per_label: int = 1024,
    seed: int = 20260322,
    oversample_factor: float = 1.7,
    max_near_duplicate_similarity: float = 0.92,
    max_samples_per_template_bucket: int = 40,
    model_id: str | None = None,
    max_length: int = 128,
    pooling: str = "mean",
    output_name: str | None = None,
    reference_date: str | None = None,
    classifier_kind: str = "auto",
    selection_metric: str = "frontier",
    trainer: Callable[..., SemanticRouterTrainingReport] = train_router_bundle_from_jsonl,
) -> SemanticRouterTrainingReport:
    """Generate one synthetic corpus, build one router bundle, and write a report."""

    _validate_generation_args(
        samples_per_label=samples_per_label,
        oversample_factor=oversample_factor,
        max_near_duplicate_similarity=max_near_duplicate_similarity,
        max_samples_per_template_bucket=max_samples_per_template_bucket,
    )
    root = _resolved_path(work_dir)
    dataset_path = root / "synthetic_router_samples.jsonl"
    bundle_dir = root / "bundle"
    report_path = root / "training_report.json"
    with _advisory_file_lock(root / ".bootstrap.lock"):
        samples, dataset_report = generate_synthetic_route_samples(
            samples_per_label=samples_per_label,
            seed=seed,
            oversample_factor=oversample_factor,
            max_near_duplicate_similarity=max_near_duplicate_similarity,
            max_samples_per_template_bucket=max_samples_per_template_bucket,
        )
        root.mkdir(parents=True, exist_ok=True)
        _write_route_samples_jsonl_atomic(samples, dataset_path)
        return trainer(
            source_dir=source_dir,
            dataset_path=dataset_path,
            output_dir=bundle_dir,
            report_path=report_path,
            model_id=model_id,
            max_length=max_length,
            pooling=pooling,
            output_name=output_name,
            reference_date=reference_date,
            classifier_kind=classifier_kind,
            selection_metric=selection_metric,
            dataset_summary=_json_safe(dataset_report.to_dict()),
        )


def bootstrap_two_stage_synthetic_router_bundle(
    *,
    source_dir: str | Path,
    work_dir: str | Path,
    samples_per_label: int = 1024,
    seed: int = 20260322,
    oversample_factor: float = 1.7,
    max_near_duplicate_similarity: float = 0.92,
    max_samples_per_template_bucket: int = 40,
    model_id_prefix: str | None = None,
    max_length: int = 128,
    pooling: str = "mean",
    output_name: str | None = None,
    reference_date: str | None = None,
    classifier_kind: str = "auto",
    selection_metric: str = "frontier",
) -> TwoStageSemanticRouterTrainingReport:
    """Generate dual synthetic corpora, train both stages, and score the combined router."""

    started = time.perf_counter()
    _validate_generation_args(
        samples_per_label=samples_per_label,
        oversample_factor=oversample_factor,
        max_near_duplicate_similarity=max_near_duplicate_similarity,
        max_samples_per_template_bucket=max_samples_per_template_bucket,
    )
    normalized_reference_date = _normalize_reference_date(reference_date)
    source_path = _require_existing_dir(source_dir, label="source_dir")
    root = _resolved_path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    backend_dataset_path = root / "backend_route_samples.jsonl"
    user_dataset_path = root / "user_intent_samples.jsonl"
    backend_bundle_dir = root / "backend_route_bundle"
    user_bundle_dir = root / "user_intent_bundle"
    backend_report_path = root / "backend_route_training_report.json"
    user_report_path = root / "user_intent_training_report.json"
    report_path = root / "training_report.json"
    with _advisory_file_lock(root / ".bootstrap-two-stage.lock"):
        backend_samples, backend_dataset_report = generate_synthetic_route_samples(
            samples_per_label=samples_per_label,
            seed=seed,
            oversample_factor=oversample_factor,
            max_near_duplicate_similarity=max_near_duplicate_similarity,
            max_samples_per_template_bucket=max_samples_per_template_bucket,
        )
        user_samples, user_dataset_report = generate_synthetic_user_intent_samples(
            samples_per_label=samples_per_label,
            seed=seed,
            oversample_factor=oversample_factor,
            max_near_duplicate_similarity=max_near_duplicate_similarity,
            max_samples_per_template_bucket=max_samples_per_template_bucket,
        )
        _write_route_samples_jsonl_atomic(backend_samples, backend_dataset_path)
        _write_user_intent_samples_jsonl_atomic(user_samples, user_dataset_path)
        backend_model_id = _stage_model_id(model_id_prefix, suffix="backend_route", fallback=source_path.name)
        user_model_id = _stage_model_id(model_id_prefix, suffix="user_intent", fallback=source_path.name)
        backend_report = train_router_bundle_from_jsonl(
            source_dir=source_path,
            dataset_path=backend_dataset_path,
            output_dir=backend_bundle_dir,
            report_path=backend_report_path,
            model_id=backend_model_id,
            max_length=max_length,
            pooling=pooling,
            output_name=output_name,
            reference_date=normalized_reference_date,
            classifier_kind=classifier_kind,
            selection_metric=selection_metric,
            dataset_summary=_json_safe(backend_dataset_report.to_dict()),
        )
        user_report = train_user_intent_bundle_from_jsonl(
            source_dir=source_path,
            dataset_path=user_dataset_path,
            output_dir=user_bundle_dir,
            report_path=user_report_path,
            model_id=user_model_id,
            max_length=max_length,
            pooling=pooling,
            output_name=output_name,
            reference_date=normalized_reference_date,
            classifier_kind=classifier_kind,
            selection_metric=selection_metric,
            dataset_summary=_json_safe(user_dataset_report.to_dict()),
        )
        two_stage_evaluations = _score_two_stage_backend_route_dataset(
            user_bundle_dir=user_bundle_dir,
            backend_bundle_dir=backend_bundle_dir,
            dataset_path=backend_dataset_path,
        )
        report = TwoStageSemanticRouterTrainingReport(
            work_dir=root,
            model_source_dir=source_path,
            reference_date=normalized_reference_date,
            user_intent_dataset_path=user_dataset_path,
            backend_route_dataset_path=backend_dataset_path,
            user_intent_bundle_dir=user_bundle_dir,
            backend_route_bundle_dir=backend_bundle_dir,
            user_intent_report=user_report,
            backend_route_report=backend_report,
            two_stage_backend_route_evaluations=two_stage_evaluations,
            created_at=_now_utc_iso(),
            duration_seconds=time.perf_counter() - started,
            work_dir_sha256=_sha256_tree(root),
            environment=_capture_environment(),
        )
        _write_json_report(report_path, report.to_dict())
        return report


def _score_two_stage_backend_route_dataset(
    *,
    user_bundle_dir: str | Path,
    backend_bundle_dir: str | Path,
    dataset_path: str | Path,
) -> dict[str, dict[str, object]]:
    """Run the full two-stage router on a backend-labeled dataset split by split."""

    backend_samples = load_labeled_route_samples(dataset_path)
    grouped_samples = split_labeled_route_samples(backend_samples)
    router = TwoStageLocalSemanticRouter(
        load_user_intent_bundle(user_bundle_dir),
        load_semantic_router_bundle(backend_bundle_dir),
    )
    evaluations: dict[str, dict[str, object]] = {}
    for split_name, split_samples in grouped_samples.items():
        split_samples_tuple = tuple(split_samples)
        if not split_samples_tuple:
            evaluations[split_name] = _empty_route_evaluation()
            continue
        records = tuple(
            ScoredRouteRecord(
                sample=sample,
                decision=router.classify(sample.text).route_decision,
            )
            for sample in split_samples_tuple
        )
        evaluations[split_name] = _evaluation_to_dict(evaluate_route_records(records), records=records)
    return evaluations


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Twinr local semantic-router bundles.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-synthetic", help="Generate a synthetic router corpus JSONL only")
    generate.add_argument("--output-path", required=True, help="Destination JSONL path")
    generate.add_argument("--samples-per-label", type=int, default=1024)
    generate.add_argument("--seed", type=int, default=20260322)
    generate.add_argument("--oversample-factor", type=float, default=1.7)
    generate.add_argument("--max-near-duplicate-similarity", type=float, default=0.92)
    generate.add_argument("--max-samples-per-template-bucket", type=int, default=40)
    generate.add_argument("--report-path", default=None)

    train = subparsers.add_parser("train-bundle", help="Build and evaluate one router bundle from an existing JSONL dataset")
    train.add_argument("--source-dir", required=True)
    train.add_argument("--dataset", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--report-path", default=None)
    train.add_argument("--model-id", default=None)
    train.add_argument("--max-length", type=int, default=128)
    train.add_argument("--pooling", default="mean", choices=tuple(sorted(_SUPPORTED_POOLING_KINDS)))
    train.add_argument("--output-name", default=None)
    train.add_argument("--classifier", default="auto", choices=tuple(sorted(_SUPPORTED_CLASSIFIER_KINDS)))
    train.add_argument("--selection-metric", default="frontier", choices=("frontier", "macro_f1", "accuracy"))
    train.add_argument("--reference-date", default=None)

    train_user = subparsers.add_parser(
        "train-user-intent-bundle",
        help="Build and evaluate one user-intent bundle from an existing JSONL dataset",
    )
    train_user.add_argument("--source-dir", required=True)
    train_user.add_argument("--dataset", required=True)
    train_user.add_argument("--output-dir", required=True)
    train_user.add_argument("--report-path", default=None)
    train_user.add_argument("--model-id", default=None)
    train_user.add_argument("--max-length", type=int, default=128)
    train_user.add_argument("--pooling", default="mean", choices=tuple(sorted(_SUPPORTED_POOLING_KINDS)))
    train_user.add_argument("--output-name", default=None)
    train_user.add_argument("--classifier", default="auto", choices=tuple(sorted(_SUPPORTED_CLASSIFIER_KINDS)))
    train_user.add_argument("--selection-metric", default="frontier", choices=("frontier", "macro_f1", "accuracy"))
    train_user.add_argument("--reference-date", default=None)

    bootstrap = subparsers.add_parser("bootstrap-synthetic", help="Generate a synthetic corpus and train a router bundle")
    bootstrap.add_argument("--source-dir", required=True)
    bootstrap.add_argument("--work-dir", required=True)
    bootstrap.add_argument("--samples-per-label", type=int, default=1024)
    bootstrap.add_argument("--seed", type=int, default=20260322)
    bootstrap.add_argument("--oversample-factor", type=float, default=1.7)
    bootstrap.add_argument("--max-near-duplicate-similarity", type=float, default=0.92)
    bootstrap.add_argument("--max-samples-per-template-bucket", type=int, default=40)
    bootstrap.add_argument("--model-id", default=None)
    bootstrap.add_argument("--max-length", type=int, default=128)
    bootstrap.add_argument("--pooling", default="mean", choices=tuple(sorted(_SUPPORTED_POOLING_KINDS)))
    bootstrap.add_argument("--output-name", default=None)
    bootstrap.add_argument("--classifier", default="auto", choices=tuple(sorted(_SUPPORTED_CLASSIFIER_KINDS)))
    bootstrap.add_argument("--selection-metric", default="frontier", choices=("frontier", "macro_f1", "accuracy"))
    bootstrap.add_argument("--reference-date", default=None)

    bootstrap_two_stage = subparsers.add_parser(
        "bootstrap-two-stage-synthetic",
        help="Generate synthetic user-intent and backend datasets, then train both router stages",
    )
    bootstrap_two_stage.add_argument("--source-dir", required=True)
    bootstrap_two_stage.add_argument("--work-dir", required=True)
    bootstrap_two_stage.add_argument("--samples-per-label", type=int, default=1024)
    bootstrap_two_stage.add_argument("--seed", type=int, default=20260322)
    bootstrap_two_stage.add_argument("--oversample-factor", type=float, default=1.7)
    bootstrap_two_stage.add_argument("--max-near-duplicate-similarity", type=float, default=0.92)
    bootstrap_two_stage.add_argument("--max-samples-per-template-bucket", type=int, default=40)
    bootstrap_two_stage.add_argument("--model-id-prefix", default=None)
    bootstrap_two_stage.add_argument("--max-length", type=int, default=128)
    bootstrap_two_stage.add_argument("--pooling", default="mean", choices=tuple(sorted(_SUPPORTED_POOLING_KINDS)))
    bootstrap_two_stage.add_argument("--output-name", default=None)
    bootstrap_two_stage.add_argument("--classifier", default="auto", choices=tuple(sorted(_SUPPORTED_CLASSIFIER_KINDS)))
    bootstrap_two_stage.add_argument("--selection-metric", default="frontier", choices=("frontier", "macro_f1", "accuracy"))
    bootstrap_two_stage.add_argument("--reference-date", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for synthetic-router generation and training."""

    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    if args.command == "generate-synthetic":
        _validate_generation_args(
            samples_per_label=args.samples_per_label,
            oversample_factor=args.oversample_factor,
            max_near_duplicate_similarity=args.max_near_duplicate_similarity,
            max_samples_per_template_bucket=args.max_samples_per_template_bucket,
        )
        samples, report = generate_synthetic_route_samples(
            samples_per_label=args.samples_per_label,
            seed=args.seed,
            oversample_factor=args.oversample_factor,
            max_near_duplicate_similarity=args.max_near_duplicate_similarity,
            max_samples_per_template_bucket=args.max_samples_per_template_bucket,
        )
        output_path = _write_route_samples_jsonl_atomic(samples, args.output_path)
        if args.report_path:
            _write_json_report(args.report_path, report.to_dict())
        print(json.dumps({"output_path": str(output_path), **_json_safe(report.to_dict())}, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "train-bundle":
        report = train_router_bundle_from_jsonl(
            source_dir=args.source_dir,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            report_path=args.report_path,
            model_id=args.model_id,
            max_length=args.max_length,
            pooling=args.pooling,
            output_name=args.output_name,
            classifier_kind=args.classifier,
            selection_metric=args.selection_metric,
            reference_date=args.reference_date,
        )
        print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "train-user-intent-bundle":
        report = train_user_intent_bundle_from_jsonl(
            source_dir=args.source_dir,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            report_path=args.report_path,
            model_id=args.model_id,
            max_length=args.max_length,
            pooling=args.pooling,
            output_name=args.output_name,
            classifier_kind=args.classifier,
            selection_metric=args.selection_metric,
            reference_date=args.reference_date,
        )
        print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "bootstrap-synthetic":
        report = bootstrap_synthetic_router_bundle(
            source_dir=args.source_dir,
            work_dir=args.work_dir,
            samples_per_label=args.samples_per_label,
            seed=args.seed,
            oversample_factor=args.oversample_factor,
            max_near_duplicate_similarity=args.max_near_duplicate_similarity,
            max_samples_per_template_bucket=args.max_samples_per_template_bucket,
            model_id=args.model_id,
            max_length=args.max_length,
            pooling=args.pooling,
            output_name=args.output_name,
            classifier_kind=args.classifier,
            selection_metric=args.selection_metric,
            reference_date=args.reference_date,
        )
        print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "bootstrap-two-stage-synthetic":
        report = bootstrap_two_stage_synthetic_router_bundle(
            source_dir=args.source_dir,
            work_dir=args.work_dir,
            samples_per_label=args.samples_per_label,
            seed=args.seed,
            oversample_factor=args.oversample_factor,
            max_near_duplicate_similarity=args.max_near_duplicate_similarity,
            max_samples_per_template_bucket=args.max_samples_per_template_bucket,
            model_id_prefix=args.model_id_prefix,
            max_length=args.max_length,
            pooling=args.pooling,
            output_name=args.output_name,
            classifier_kind=args.classifier,
            selection_metric=args.selection_metric,
            reference_date=args.reference_date,
        )
        print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
        return 0
    raise ValueError(f"Unsupported command: {args.command!r}")


if __name__ == "__main__":  # pragma: no cover - exercised via module CLI.
    raise SystemExit(main())