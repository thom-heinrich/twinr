"""Load labeled route datasets and evaluate or calibrate local router policy."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
from typing import Iterable, Sequence

from .contracts import ROUTE_LABEL_VALUES, SemanticRouteDecision, normalize_route_label
from .policy import SemanticRouterPolicy


@dataclass(frozen=True, slots=True)
class LabeledRouteSample:
    """Store one labeled transcript for router training or evaluation."""

    text: str
    label: str
    sample_id: str | None = None
    split: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", str(self.text or "").strip())
        object.__setattr__(self, "label", normalize_route_label(self.label))
        object.__setattr__(self, "sample_id", str(self.sample_id or "").strip() or None)
        object.__setattr__(self, "split", str(self.split or "").strip().lower() or None)
        object.__setattr__(self, "source", str(self.source or "").strip() or None)
        if not self.text:
            raise ValueError("LabeledRouteSample.text must not be empty.")


@dataclass(frozen=True, slots=True)
class ScoredRouteRecord:
    """Store one gold label together with one scored router decision."""

    sample: LabeledRouteSample
    decision: SemanticRouteDecision


@dataclass(frozen=True, slots=True)
class SemanticRouterEvaluation:
    """Summarize router quality over one labeled evaluation slice."""

    total: int
    accuracy: float
    macro_f1: float
    fallback_rate: float
    authoritative_rate: float
    unsafe_authoritative_error_rate: float
    confusion_matrix: dict[str, dict[str, int]]
    per_label: dict[str, dict[str, float]]


def load_labeled_route_samples(path: str | Path) -> tuple[LabeledRouteSample, ...]:
    """Load labeled route samples from one JSONL file."""

    samples: list[LabeledRouteSample] = []
    file_path = Path(path)
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        samples.append(
            LabeledRouteSample(
                text=payload["text"],
                label=payload["label"],
                sample_id=payload.get("id"),
                split=payload.get("split"),
                source=payload.get("source"),
            )
        )
    return tuple(samples)


def split_labeled_route_samples(
    samples: Sequence[LabeledRouteSample],
) -> dict[str, tuple[LabeledRouteSample, ...]]:
    """Group samples by their explicit `split` field."""

    grouped: dict[str, list[LabeledRouteSample]] = {}
    for sample in samples:
        key = sample.split or "unspecified"
        grouped.setdefault(key, []).append(sample)
    return {key: tuple(values) for key, values in grouped.items()}


def score_semantic_router(
    router,
    samples: Sequence[LabeledRouteSample],
    *,
    policy: SemanticRouterPolicy | None = None,
) -> tuple[ScoredRouteRecord, ...]:
    """Run the router over labeled samples and return scored records."""

    return tuple(
        ScoredRouteRecord(sample=sample, decision=router.classify(sample.text, policy=policy))
        for sample in samples
    )


def evaluate_route_records(
    records: Sequence[ScoredRouteRecord],
    *,
    policy: SemanticRouterPolicy | None = None,
) -> SemanticRouterEvaluation:
    """Compute classification and authority metrics for scored route records."""

    confusion_matrix = {
        gold: {predicted: 0 for predicted in ROUTE_LABEL_VALUES}
        for gold in ROUTE_LABEL_VALUES
    }
    applied_decisions = [
        record.decision.with_policy(policy) if policy is not None else record.decision
        for record in records
    ]
    total = len(applied_decisions)
    correct = 0
    authoritative_count = 0
    unsafe_authoritative_errors = 0
    for record, decision in zip(records, applied_decisions, strict=True):
        gold = record.sample.label
        predicted = decision.label
        confusion_matrix[gold][predicted] += 1
        if predicted == gold:
            correct += 1
        if decision.authoritative:
            authoritative_count += 1
            if predicted != gold:
                unsafe_authoritative_errors += 1
    per_label: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for label in ROUTE_LABEL_VALUES:
        true_positive = confusion_matrix[label][label]
        false_positive = sum(confusion_matrix[gold][label] for gold in ROUTE_LABEL_VALUES if gold != label)
        false_negative = sum(confusion_matrix[label][pred] for pred in ROUTE_LABEL_VALUES if pred != label)
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(sum(confusion_matrix[label].values())),
        }
        f1_values.append(f1)
    fallback_rate = _safe_ratio(total - authoritative_count, total)
    return SemanticRouterEvaluation(
        total=total,
        accuracy=_safe_ratio(correct, total),
        macro_f1=_safe_ratio(sum(f1_values), len(f1_values)),
        fallback_rate=fallback_rate,
        authoritative_rate=_safe_ratio(authoritative_count, total),
        unsafe_authoritative_error_rate=_safe_ratio(unsafe_authoritative_errors, total),
        confusion_matrix=confusion_matrix,
        per_label=per_label,
    )


def tune_policy_thresholds(
    records: Sequence[ScoredRouteRecord],
    *,
    authoritative_labels: Sequence[str] = ("web", "memory", "tool"),
    threshold_grid: Sequence[float] | None = None,
    min_margin_grid: Sequence[float] | None = None,
    required_recalls: dict[str, float] | None = None,
) -> SemanticRouterPolicy:
    """Grid-search per-class thresholds for safe local authority gating.

    The tuner operates on already-scored route records, so repeated policy
    sweeps do not need to rerun the encoder.
    """

    normalized_labels = tuple(normalize_route_label(label) for label in authoritative_labels)
    thresholds = tuple(
        float(value) for value in (threshold_grid or (0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85))
    )
    margins = tuple(float(value) for value in (min_margin_grid or (0.0, 0.03, 0.05, 0.08, 0.1, 0.12)))
    recall_targets = {
        normalize_route_label(label): float(target)
        for label, target in dict(required_recalls or {}).items()
    }
    best_policy: SemanticRouterPolicy | None = None
    best_key: tuple[float, ...] | None = None
    for min_margin in margins:
        for threshold_values in product(thresholds, repeat=len(normalized_labels)):
            candidate_thresholds = {
                label: threshold
                for label, threshold in zip(normalized_labels, threshold_values, strict=True)
            }
            policy = SemanticRouterPolicy(
                thresholds=candidate_thresholds,
                authoritative_labels=normalized_labels,
                min_margin=min_margin,
            )
            summary = _authority_summary(records, policy, recall_targets)
            key = (
                1.0 if summary["meets_targets"] else 0.0,
                -summary["unsafe_authoritative_errors"],
                summary["average_target_recall"],
                -summary["fallback_rate"],
                sum(candidate_thresholds.values()) / float(len(candidate_thresholds)),
                -min_margin,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_policy = policy
    if best_policy is None:
        raise ValueError("Policy tuning requires at least one scored record.")
    return best_policy


def _authority_summary(
    records: Sequence[ScoredRouteRecord],
    policy: SemanticRouterPolicy,
    recall_targets: dict[str, float],
) -> dict[str, float | bool]:
    gold_support = {label: 0 for label in ROUTE_LABEL_VALUES}
    successful_authority = {label: 0 for label in ROUTE_LABEL_VALUES}
    authoritative_count = 0
    unsafe_authoritative_errors = 0
    for record in records:
        gold_support[record.sample.label] += 1
        applied = record.decision.with_policy(policy)
        if applied.authoritative:
            authoritative_count += 1
            if applied.label == record.sample.label:
                successful_authority[record.sample.label] += 1
            else:
                unsafe_authoritative_errors += 1
    recalls = {
        label: _safe_ratio(successful_authority[label], gold_support[label])
        for label in recall_targets
    }
    meets_targets = all(recalls.get(label, 0.0) >= target for label, target in recall_targets.items())
    average_target_recall = _safe_ratio(sum(recalls.values()), len(recalls)) if recalls else 0.0
    return {
        "meets_targets": meets_targets,
        "average_target_recall": average_target_recall,
        "unsafe_authoritative_errors": float(unsafe_authoritative_errors),
        "fallback_rate": _safe_ratio(len(records) - authoritative_count, len(records)),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)
