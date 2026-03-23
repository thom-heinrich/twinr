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

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

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


def _normalize_classifier_kind(value: object) -> str:
    """Return one validated offline classifier kind identifier."""

    normalized = str(value or "centroid").strip().lower()
    if normalized not in {"centroid", "linear"}:
        raise ValueError(f"Unsupported semantic-router classifier kind: {value!r}")
    return normalized


def _router_bundle_builder_for_classifier(kind: str) -> _BundleBuilder:
    """Return the backend-route bundle builder for one classifier kind."""

    normalized_kind = _normalize_classifier_kind(kind)
    if normalized_kind == "linear":
        return build_linear_router_bundle_from_jsonl
    return build_centroid_router_bundle_from_jsonl


def _user_intent_bundle_builder_for_classifier(kind: str) -> _BundleBuilder:
    """Return the user-intent bundle builder for one classifier kind."""

    normalized_kind = _normalize_classifier_kind(kind)
    if normalized_kind == "linear":
        return build_linear_user_intent_bundle_from_jsonl
    return build_centroid_user_intent_bundle_from_jsonl


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

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable training report."""

        payload: dict[str, object] = {
            "dataset_path": str(self.dataset_path),
            "bundle_dir": str(self.bundle_dir),
            "model_source_dir": str(self.model_source_dir),
            "model_id": self.model_id,
            "total_samples": int(self.total_samples),
            "split_counts": dict(self.split_counts),
            "evaluations": dict(self.evaluations),
            "reference_date": self.reference_date,
        }
        if self.dataset_summary is not None:
            payload["dataset_summary"] = dict(self.dataset_summary)
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

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable training report."""

        payload: dict[str, object] = {
            "dataset_path": str(self.dataset_path),
            "bundle_dir": str(self.bundle_dir),
            "model_source_dir": str(self.model_source_dir),
            "model_id": self.model_id,
            "total_samples": int(self.total_samples),
            "split_counts": dict(self.split_counts),
            "evaluations": dict(self.evaluations),
            "reference_date": self.reference_date,
        }
        if self.dataset_summary is not None:
            payload["dataset_summary"] = dict(self.dataset_summary)
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

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable two-stage training report."""

        return {
            "work_dir": str(self.work_dir),
            "model_source_dir": str(self.model_source_dir),
            "reference_date": self.reference_date,
            "user_intent_dataset_path": str(self.user_intent_dataset_path),
            "backend_route_dataset_path": str(self.backend_route_dataset_path),
            "user_intent_bundle_dir": str(self.user_intent_bundle_dir),
            "backend_route_bundle_dir": str(self.backend_route_bundle_dir),
            "user_intent_report": self.user_intent_report.to_dict(),
            "backend_route_report": self.backend_route_report.to_dict(),
            "two_stage_backend_route_evaluations": dict(self.two_stage_backend_route_evaluations),
        }


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
    reference_date: str = "2026-03-22",
    classifier_kind: str = "centroid",
    dataset_summary: dict[str, object] | None = None,
    bundle_builder: _BundleBuilder = build_centroid_router_bundle_from_jsonl,
    bundle_loader: _BundleLoader = load_semantic_router_bundle,
    router_factory: _RouterFactory = LocalSemanticRouter,
    router_scorer: _RouterScorer = score_semantic_router,
    record_evaluator: _RecordEvaluator = evaluate_route_records,
) -> SemanticRouterTrainingReport:
    """Build one router bundle from JSONL and score it on every split."""

    source_path = Path(source_dir).expanduser().resolve(strict=False)
    dataset_file = Path(dataset_path).expanduser().resolve(strict=False)
    resolved_bundle_builder = (
        _router_bundle_builder_for_classifier(classifier_kind)
        if bundle_builder is build_centroid_router_bundle_from_jsonl
        else bundle_builder
    )
    bundle_dir = Path(
        resolved_bundle_builder(
            source_dir=source_path,
            dataset_path=dataset_file,
            output_dir=output_dir,
            model_id=model_id,
            max_length=max_length,
            pooling=pooling,
            output_name=output_name,
            reference_date=reference_date,
        )
    ).expanduser().resolve(strict=False)
    samples = load_labeled_route_samples(dataset_file)
    grouped_samples = split_labeled_route_samples(samples)
    bundle = bundle_loader(bundle_dir)
    router = router_factory(bundle)
    evaluations = {}
    split_counts = {}
    for split_name, split_samples in grouped_samples.items():
        records = router_scorer(router, split_samples)
        summary = record_evaluator(records)
        evaluations[split_name] = _evaluation_to_dict(summary)
        split_counts[split_name] = len(split_samples)
    report = SemanticRouterTrainingReport(
        dataset_path=dataset_file,
        bundle_dir=bundle_dir,
        model_source_dir=source_path,
        model_id=str(model_id or source_path.name).strip(),
        total_samples=len(samples),
        split_counts=split_counts,
        evaluations=evaluations,
        reference_date=reference_date,
        dataset_summary=dataset_summary,
    )
    if report_path is not None:
        resolved_report_path = Path(report_path).expanduser().resolve(strict=False)
        resolved_report_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


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
    reference_date: str = "2026-03-22",
    classifier_kind: str = "centroid",
    dataset_summary: dict[str, object] | None = None,
    bundle_builder: _BundleBuilder = build_centroid_user_intent_bundle_from_jsonl,
    bundle_loader: _BundleLoader = load_user_intent_bundle,
    router_factory: _RouterFactory = LocalUserIntentRouter,
    router_scorer: _RouterScorer = score_user_intent_router,
    record_evaluator: _RecordEvaluator = evaluate_user_intent_records,
) -> UserIntentTrainingReport:
    """Build one user-intent bundle from JSONL and score it on every split."""

    source_path = Path(source_dir).expanduser().resolve(strict=False)
    dataset_file = Path(dataset_path).expanduser().resolve(strict=False)
    resolved_bundle_builder = (
        _user_intent_bundle_builder_for_classifier(classifier_kind)
        if bundle_builder is build_centroid_user_intent_bundle_from_jsonl
        else bundle_builder
    )
    bundle_dir = Path(
        resolved_bundle_builder(
            source_dir=source_path,
            dataset_path=dataset_file,
            output_dir=output_dir,
            model_id=model_id,
            max_length=max_length,
            pooling=pooling,
            output_name=output_name,
            reference_date=reference_date,
        )
    ).expanduser().resolve(strict=False)
    samples = load_labeled_user_intent_samples(dataset_file)
    grouped_samples = split_labeled_user_intent_samples(samples)
    bundle = bundle_loader(bundle_dir)
    router = router_factory(bundle)
    evaluations = {}
    split_counts = {}
    for split_name, split_samples in grouped_samples.items():
        records = router_scorer(router, split_samples)
        summary = record_evaluator(records)
        evaluations[split_name] = _user_evaluation_to_dict(summary)
        split_counts[split_name] = len(split_samples)
    report = UserIntentTrainingReport(
        dataset_path=dataset_file,
        bundle_dir=bundle_dir,
        model_source_dir=source_path,
        model_id=str(model_id or source_path.name).strip(),
        total_samples=len(samples),
        split_counts=split_counts,
        evaluations=evaluations,
        reference_date=reference_date,
        dataset_summary=dataset_summary,
    )
    if report_path is not None:
        resolved_report_path = Path(report_path).expanduser().resolve(strict=False)
        resolved_report_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_report_path.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


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
    reference_date: str = "2026-03-22",
    classifier_kind: str = "centroid",
    trainer: Callable[..., SemanticRouterTrainingReport] = train_router_bundle_from_jsonl,
) -> SemanticRouterTrainingReport:
    """Generate one synthetic corpus, build one router bundle, and write a report."""

    root = Path(work_dir).expanduser().resolve(strict=False)
    dataset_path = root / "synthetic_router_samples.jsonl"
    bundle_dir = root / "bundle"
    report_path = root / "training_report.json"
    samples, dataset_report = generate_synthetic_route_samples(
        samples_per_label=samples_per_label,
        seed=seed,
        oversample_factor=oversample_factor,
        max_near_duplicate_similarity=max_near_duplicate_similarity,
        max_samples_per_template_bucket=max_samples_per_template_bucket,
    )
    root.mkdir(parents=True, exist_ok=True)
    write_synthetic_route_samples_jsonl(samples, dataset_path)
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
        dataset_summary=dataset_report.to_dict(),
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
    reference_date: str = "2026-03-22",
    classifier_kind: str = "centroid",
) -> TwoStageSemanticRouterTrainingReport:
    """Generate dual synthetic corpora, train both stages, and score the combined router."""

    source_path = Path(source_dir).expanduser().resolve(strict=False)
    root = Path(work_dir).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    backend_dataset_path = root / "backend_route_samples.jsonl"
    user_dataset_path = root / "user_intent_samples.jsonl"
    backend_bundle_dir = root / "backend_route_bundle"
    user_bundle_dir = root / "user_intent_bundle"
    backend_report_path = root / "backend_route_training_report.json"
    user_report_path = root / "user_intent_training_report.json"
    report_path = root / "training_report.json"
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
    write_synthetic_route_samples_jsonl(backend_samples, backend_dataset_path)
    write_synthetic_user_intent_samples_jsonl(user_samples, user_dataset_path)
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
        reference_date=reference_date,
        classifier_kind=classifier_kind,
        dataset_summary=backend_dataset_report.to_dict(),
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
        reference_date=reference_date,
        classifier_kind=classifier_kind,
        dataset_summary=user_dataset_report.to_dict(),
    )
    two_stage_evaluations = _score_two_stage_backend_route_dataset(
        user_bundle_dir=user_bundle_dir,
        backend_bundle_dir=backend_bundle_dir,
        dataset_path=backend_dataset_path,
    )
    report = TwoStageSemanticRouterTrainingReport(
        work_dir=root,
        model_source_dir=source_path,
        reference_date=reference_date,
        user_intent_dataset_path=user_dataset_path,
        backend_route_dataset_path=backend_dataset_path,
        user_intent_bundle_dir=user_bundle_dir,
        backend_route_bundle_dir=backend_bundle_dir,
        user_intent_report=user_report,
        backend_route_report=backend_report,
        two_stage_backend_route_evaluations=two_stage_evaluations,
    )
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _evaluation_to_dict(summary: Any) -> dict[str, object]:
    return {
        "total": int(summary.total),
        "accuracy": float(summary.accuracy),
        "macro_f1": float(summary.macro_f1),
        "fallback_rate": float(summary.fallback_rate),
        "authoritative_rate": float(summary.authoritative_rate),
        "unsafe_authoritative_error_rate": float(summary.unsafe_authoritative_error_rate),
        "confusion_matrix": dict(summary.confusion_matrix),
        "per_label": dict(summary.per_label),
    }


def _user_evaluation_to_dict(summary: Any) -> dict[str, object]:
    return {
        "total": int(summary.total),
        "accuracy": float(summary.accuracy),
        "macro_f1": float(summary.macro_f1),
        "confusion_matrix": dict(summary.confusion_matrix),
        "per_label": dict(summary.per_label),
    }


def _stage_model_id(model_id_prefix: str | None, *, suffix: str, fallback: str) -> str:
    base = str(model_id_prefix or fallback).strip()
    return f"{base}_{suffix}"


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
        records = tuple(
            ScoredRouteRecord(
                sample=sample,
                decision=router.classify(sample.text).route_decision,
            )
            for sample in split_samples
        )
        evaluations[split_name] = _evaluation_to_dict(evaluate_route_records(records))
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
    train.add_argument("--pooling", default="mean", choices=("mean", "cls", "prepooled"))
    train.add_argument("--output-name", default=None)
    train.add_argument("--classifier", default="centroid", choices=("centroid", "linear"))
    train.add_argument("--reference-date", default="2026-03-22")

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
    train_user.add_argument("--pooling", default="mean", choices=("mean", "cls", "prepooled"))
    train_user.add_argument("--output-name", default=None)
    train_user.add_argument("--classifier", default="centroid", choices=("centroid", "linear"))
    train_user.add_argument("--reference-date", default="2026-03-22")

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
    bootstrap.add_argument("--pooling", default="mean", choices=("mean", "cls", "prepooled"))
    bootstrap.add_argument("--output-name", default=None)
    bootstrap.add_argument("--classifier", default="centroid", choices=("centroid", "linear"))
    bootstrap.add_argument("--reference-date", default="2026-03-22")

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
    bootstrap_two_stage.add_argument("--pooling", default="mean", choices=("mean", "cls", "prepooled"))
    bootstrap_two_stage.add_argument("--output-name", default=None)
    bootstrap_two_stage.add_argument("--classifier", default="centroid", choices=("centroid", "linear"))
    bootstrap_two_stage.add_argument("--reference-date", default="2026-03-22")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for synthetic-router generation and training."""

    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    if args.command == "generate-synthetic":
        samples, report = generate_synthetic_route_samples(
            samples_per_label=args.samples_per_label,
            seed=args.seed,
            oversample_factor=args.oversample_factor,
            max_near_duplicate_similarity=args.max_near_duplicate_similarity,
            max_samples_per_template_bucket=args.max_samples_per_template_bucket,
        )
        output_path = write_synthetic_route_samples_jsonl(samples, args.output_path)
        if args.report_path:
            report_path = Path(args.report_path).expanduser().resolve(strict=False)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"output_path": str(output_path), **report.to_dict()}, ensure_ascii=True))
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
            reference_date=args.reference_date,
        )
        print(json.dumps(report.to_dict(), ensure_ascii=True))
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
            reference_date=args.reference_date,
        )
        print(json.dumps(report.to_dict(), ensure_ascii=True))
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
            reference_date=args.reference_date,
        )
        print(json.dumps(report.to_dict(), ensure_ascii=True))
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
            reference_date=args.reference_date,
        )
        print(json.dumps(report.to_dict(), ensure_ascii=True))
        return 0
    raise ValueError(f"Unsupported command: {args.command!r}")


if __name__ == "__main__":  # pragma: no cover - exercised via module CLI.
    raise SystemExit(main())
