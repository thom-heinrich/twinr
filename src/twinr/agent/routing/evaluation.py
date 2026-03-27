"""Load labeled route datasets and evaluate or calibrate local router policy."""

# CHANGELOG: 2026-03-27
# BUG-1: Fixed policy tuning on empty input. The previous implementation silently returned an arbitrary
# BUG-1: policy because grid-search still iterated over threshold candidates even when `records` was empty.
# BUG-2: Fixed tuner objective so it no longer prefers degenerate near-all-fallback policies just because
# BUG-2: they minimize raw authoritative error counts. Tuning now optimizes selective risk/coverage utility.
# BUG-3: Fixed `unsafe_authoritative_error_rate` to measure risk on authoritative decisions instead of
# BUG-3: incorrectly diluting the error rate over all samples.
# BUG-4: Fixed evaluation/scoring memory pressure by avoiding an unnecessary full copy of applied decisions.
# SEC-1: Refuse non-regular dataset paths (e.g. device files/FIFOs like `/dev/zero`) and stream JSONL line
# SEC-1: by line to prevent practical memory-/hang-based DoS on Raspberry Pi deployments.
# SEC-2: Reject oversized JSONL lines and surface precise line-numbered parse/schema errors.
# IMP-1: Added streaming iterators for dataset loading and router scoring so large evaluation slices remain
# IMP-1: usable on memory-constrained Raspberry Pi 4 systems.
# IMP-2: Upgraded evaluation to selective-prediction metrics used by 2026 routing/calibration work:
# IMP-2: coverage, selective risk, conservative Wilson upper bound, and per-label authority diagnostics.
# IMP-3: Upgraded threshold tuning to a risk-aware constrained search with optional risk/coverage targets,
# IMP-3: validation of grids/targets, duplicate-label normalization, and deterministic tie-breaking.
# BREAKING: `unsafe_authoritative_error_rate` now reports conditional error on authoritative decisions,
# BREAKING: which matches the field name and router-safety use-case. The old diluted quantity was unsafe.
# BREAKING: dataset loading now rejects non-regular files and oversized JSONL records to prevent local DoS.

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
import math
from pathlib import Path
from statistics import NormalDist
import stat
from typing import Iterable, Iterator, Protocol, Sequence

from .contracts import ROUTE_LABEL_VALUES, SemanticRouteDecision, normalize_route_label
from .policy import SemanticRouterPolicy


# Conservative per-record cap to keep malformed or hostile JSONL from exhausting Pi memory.
_MAX_JSONL_LINE_BYTES = 1_048_576  # 1 MiB
_DEFAULT_RISK_CONFIDENCE = 0.95

_CANONICAL_ROUTE_LABELS = tuple(
    dict.fromkeys(normalize_route_label(label) for label in ROUTE_LABEL_VALUES)
)
if not _CANONICAL_ROUTE_LABELS:
    raise ValueError("ROUTE_LABEL_VALUES must not be empty.")


class SupportsSemanticRouting(Protocol):
    """Minimal router protocol required by this module."""

    def classify(
        self,
        text: str,
        *,
        policy: SemanticRouterPolicy | None = None,
    ) -> SemanticRouteDecision: ...


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
        object.__setattr__(self, "label", _normalize_known_route_label(self.label))
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
    balanced_accuracy: float = 0.0
    weighted_f1: float = 0.0
    authoritative_count: int = 0
    authoritative_error_count: int = 0
    authoritative_correct_count: int = 0
    fallback_count: int = 0
    selective_risk_upper_bound_95: float = 1.0


def iter_labeled_route_samples(path: str | Path) -> Iterator[LabeledRouteSample]:
    """Yield labeled route samples from one JSONL file.

    The loader is intentionally streaming and only accepts regular files so a
    misconfigured or attacker-controlled path cannot hang evaluation by pointing
    to a device file or FIFO.
    """

    file_path = Path(path)
    try:
        file_stat = file_path.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Labeled route dataset not found: {file_path}") from exc
    except OSError as exc:
        raise OSError(f"Unable to stat labeled route dataset {file_path!s}: {exc}") from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(
            f"Labeled route dataset must be a regular file, got {file_path!s}."
        )

    with file_path.open("rb") as handle:
        line_number = 0
        while True:
            raw_line = handle.readline(_MAX_JSONL_LINE_BYTES + 1)
            if not raw_line:
                break
            line_number += 1
            if len(raw_line.rstrip(b"\r\n")) > _MAX_JSONL_LINE_BYTES:
                raise ValueError(
                    f"JSONL record at line {line_number} exceeds {_MAX_JSONL_LINE_BYTES} bytes."
                )
            try:
                line = raw_line.decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"Invalid UTF-8 in {file_path!s} at line {line_number}: {exc}"
                ) from exc
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {file_path!s} at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected a JSON object in {file_path!s} at line {line_number}, got {type(payload).__name__}."
                )
            try:
                yield LabeledRouteSample(
                    text=payload["text"],
                    label=payload["label"],
                    sample_id=payload.get("id"),
                    split=payload.get("split"),
                    source=payload.get("source"),
                )
            except KeyError as exc:
                raise ValueError(
                    f"Missing required field {exc.args[0]!r} in {file_path!s} at line {line_number}."
                ) from exc
            except Exception as exc:
                raise ValueError(
                    f"Invalid labeled route sample in {file_path!s} at line {line_number}: {exc}"
                ) from exc


def load_labeled_route_samples(path: str | Path) -> tuple[LabeledRouteSample, ...]:
    """Load labeled route samples from one JSONL file."""

    return tuple(iter_labeled_route_samples(path))


def split_labeled_route_samples(
    samples: Sequence[LabeledRouteSample],
) -> dict[str, tuple[LabeledRouteSample, ...]]:
    """Group samples by their explicit `split` field."""

    grouped: dict[str, list[LabeledRouteSample]] = {}
    for sample in samples:
        key = sample.split or "unspecified"
        grouped.setdefault(key, []).append(sample)
    return {key: tuple(values) for key, values in grouped.items()}


def iter_scored_route_records(
    router: SupportsSemanticRouting,
    samples: Iterable[LabeledRouteSample],
    *,
    policy: SemanticRouterPolicy | None = None,
) -> Iterator[ScoredRouteRecord]:
    """Yield scored route records without materializing the full evaluation slice."""

    for sample in samples:
        try:
            decision = router.classify(sample.text, policy=policy)
        except Exception as exc:
            sample_ref = sample.sample_id or sample.text[:80]
            raise RuntimeError(f"Router failed to classify sample {sample_ref!r}: {exc}") from exc
        yield ScoredRouteRecord(sample=sample, decision=decision)


def score_semantic_router(
    router: SupportsSemanticRouting,
    samples: Sequence[LabeledRouteSample],
    *,
    policy: SemanticRouterPolicy | None = None,
) -> tuple[ScoredRouteRecord, ...]:
    """Run the router over labeled samples and return scored records."""

    return tuple(iter_scored_route_records(router, samples, policy=policy))


def evaluate_route_records(
    records: Sequence[ScoredRouteRecord],
    *,
    policy: SemanticRouterPolicy | None = None,
) -> SemanticRouterEvaluation:
    """Compute classification and authority metrics for scored route records."""

    confusion_matrix = {
        gold: {predicted: 0 for predicted in _CANONICAL_ROUTE_LABELS}
        for gold in _CANONICAL_ROUTE_LABELS
    }
    total = len(records)
    if total == 0:
        return SemanticRouterEvaluation(
            total=0,
            accuracy=0.0,
            macro_f1=0.0,
            fallback_rate=0.0,
            authoritative_rate=0.0,
            unsafe_authoritative_error_rate=0.0,
            confusion_matrix=confusion_matrix,
            per_label={label: _empty_per_label_metrics() for label in _CANONICAL_ROUTE_LABELS},
            balanced_accuracy=0.0,
            weighted_f1=0.0,
            authoritative_count=0,
            authoritative_error_count=0,
            authoritative_correct_count=0,
            fallback_count=0,
            selective_risk_upper_bound_95=1.0,
        )

    correct = 0
    authoritative_count = 0
    authoritative_correct = 0
    authoritative_errors = 0
    fallback_count = 0
    predicted_support = {label: 0 for label in _CANONICAL_ROUTE_LABELS}
    authoritative_gold_support = {label: 0 for label in _CANONICAL_ROUTE_LABELS}
    authoritative_predicted_support = {label: 0 for label in _CANONICAL_ROUTE_LABELS}
    authoritative_true_positive = {label: 0 for label in _CANONICAL_ROUTE_LABELS}

    for record in records:
        decision = record.decision.with_policy(policy) if policy is not None else record.decision
        gold = _normalize_known_route_label(record.sample.label)
        predicted = _normalize_known_route_label(decision.label)
        confusion_matrix[gold][predicted] += 1
        predicted_support[predicted] += 1

        is_correct = predicted == gold
        if is_correct:
            correct += 1

        if decision.authoritative:
            authoritative_count += 1
            authoritative_gold_support[gold] += 1
            authoritative_predicted_support[predicted] += 1
            if is_correct:
                authoritative_correct += 1
                authoritative_true_positive[gold] += 1
            else:
                authoritative_errors += 1
        else:
            fallback_count += 1

    per_label: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    weighted_f1_sum = 0.0
    recall_sum = 0.0

    for label in _CANONICAL_ROUTE_LABELS:
        true_positive = confusion_matrix[label][label]
        false_positive = sum(
            confusion_matrix[gold][label]
            for gold in _CANONICAL_ROUTE_LABELS
            if gold != label
        )
        false_negative = sum(
            confusion_matrix[label][pred]
            for pred in _CANONICAL_ROUTE_LABELS
            if pred != label
        )
        support = sum(confusion_matrix[label].values())
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1 = _safe_ratio(2.0 * precision * recall, precision + recall)

        authoritative_precision = _safe_ratio(
            authoritative_true_positive[label],
            authoritative_predicted_support[label],
        )
        authoritative_recall = _safe_ratio(
            authoritative_true_positive[label],
            support,
        )
        authority_coverage = _safe_ratio(authoritative_gold_support[label], support)

        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
            "predicted_support": float(predicted_support[label]),
            "authoritative_precision": authoritative_precision,
            "authoritative_recall": authoritative_recall,
            "authority_coverage": authority_coverage,
            "authoritative_support": float(authoritative_gold_support[label]),
        }
        f1_values.append(f1)
        weighted_f1_sum += f1 * float(support)
        recall_sum += recall

    fallback_rate = _safe_ratio(fallback_count, total)
    selective_risk = _safe_ratio(authoritative_errors, authoritative_count)

    return SemanticRouterEvaluation(
        total=total,
        accuracy=_safe_ratio(correct, total),
        macro_f1=_safe_ratio(sum(f1_values), len(f1_values)),
        fallback_rate=fallback_rate,
        authoritative_rate=_safe_ratio(authoritative_count, total),
        unsafe_authoritative_error_rate=selective_risk,
        confusion_matrix=confusion_matrix,
        per_label=per_label,
        balanced_accuracy=_safe_ratio(recall_sum, len(_CANONICAL_ROUTE_LABELS)),
        weighted_f1=_safe_ratio(weighted_f1_sum, total),
        authoritative_count=authoritative_count,
        authoritative_error_count=authoritative_errors,
        authoritative_correct_count=authoritative_correct,
        fallback_count=fallback_count,
        selective_risk_upper_bound_95=_wilson_upper_bound(
            authoritative_errors,
            authoritative_count,
            confidence=_DEFAULT_RISK_CONFIDENCE,
        ),
    )


def tune_policy_thresholds(
    records: Sequence[ScoredRouteRecord],
    *,
    authoritative_labels: Sequence[str] = ("web", "memory", "tool"),
    threshold_grid: Sequence[float] | None = None,
    min_margin_grid: Sequence[float] | None = None,
    required_recalls: dict[str, float] | None = None,
    max_selective_risk: float | None = None,
    min_coverage: float | None = None,
    risk_confidence: float = _DEFAULT_RISK_CONFIDENCE,
) -> SemanticRouterPolicy:
    """Grid-search per-class thresholds for safe local authority gating.

    The tuner operates on already-scored route records, so repeated policy
    sweeps do not need to rerun the encoder.

    Frontier upgrade:
    - searches against selective risk / coverage utility instead of raw error count
    - can enforce optional risk and coverage constraints
    - uses a conservative Wilson upper bound for finite-sample risk guardrailing
    """

    if not records:
        raise ValueError("Policy tuning requires at least one scored record.")

    normalized_labels = _dedupe_preserve_order(
        _normalize_known_route_label(label) for label in authoritative_labels
    )
    if not normalized_labels:
        raise ValueError("authoritative_labels must not be empty.")

    thresholds = tuple(
        sorted(
            {
                _validate_unit_interval(float(value), name="threshold_grid")
                for value in (threshold_grid or (0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85))
            }
        )
    )
    if not thresholds:
        raise ValueError("threshold_grid must contain at least one value.")

    margins = tuple(
        sorted(
            {
                _validate_non_negative(float(value), name="min_margin_grid")
                for value in (min_margin_grid or (0.0, 0.03, 0.05, 0.08, 0.1, 0.12))
            }
        )
    )
    if not margins:
        raise ValueError("min_margin_grid must contain at least one value.")

    recall_targets = {
        _normalize_known_route_label(label): _validate_unit_interval(float(target), name=f"required_recalls[{label!r}]")
        for label, target in dict(required_recalls or {}).items()
    }
    invalid_recall_labels = tuple(label for label in recall_targets if label not in normalized_labels)
    if invalid_recall_labels:
        raise ValueError(
            "required_recalls contains labels that are not configured as authoritative: "
            + ", ".join(invalid_recall_labels)
        )

    if max_selective_risk is not None:
        max_selective_risk = _validate_unit_interval(max_selective_risk, name="max_selective_risk")
    if min_coverage is not None:
        min_coverage = _validate_unit_interval(min_coverage, name="min_coverage")
    _validate_open_unit_interval(risk_confidence, name="risk_confidence")

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
            summary = _authority_summary(
                records,
                policy,
                recall_targets,
                max_selective_risk=max_selective_risk,
                min_coverage=min_coverage,
                risk_confidence=risk_confidence,
            )
            average_threshold = sum(candidate_thresholds.values()) / float(len(candidate_thresholds))
            key = (
                1.0 if summary["meets_constraints"] else 0.0,
                float(summary["selective_utility"]),
                -float(summary["selective_risk_upper_bound"]),
                float(summary["coverage"]),
                float(summary["average_target_recall"]),
                -min_margin,
                -average_threshold,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_policy = policy

    if best_policy is None:
        raise ValueError("Policy tuning failed to evaluate any candidate policy.")
    return best_policy


def _authority_summary(
    records: Sequence[ScoredRouteRecord],
    policy: SemanticRouterPolicy,
    recall_targets: dict[str, float],
    *,
    max_selective_risk: float | None,
    min_coverage: float | None,
    risk_confidence: float,
) -> dict[str, float | bool]:
    gold_support = {label: 0 for label in _CANONICAL_ROUTE_LABELS}
    successful_authority = {label: 0 for label in _CANONICAL_ROUTE_LABELS}
    authoritative_count = 0
    authoritative_correct = 0
    authoritative_errors = 0

    for record in records:
        gold = _normalize_known_route_label(record.sample.label)
        gold_support[gold] += 1
        applied = record.decision.with_policy(policy)
        predicted = _normalize_known_route_label(applied.label)
        if applied.authoritative:
            authoritative_count += 1
            if predicted == gold:
                authoritative_correct += 1
                successful_authority[gold] += 1
            else:
                authoritative_errors += 1

    coverage = _safe_ratio(authoritative_count, len(records))
    selective_risk = _safe_ratio(authoritative_errors, authoritative_count)
    selective_risk_upper_bound = _wilson_upper_bound(
        authoritative_errors,
        authoritative_count,
        confidence=risk_confidence,
    )
    authoritative_precision = _safe_ratio(authoritative_correct, authoritative_count)
    selective_utility = _harmonic_mean(authoritative_precision, coverage)

    recalls = {
        label: _safe_ratio(successful_authority[label], gold_support[label])
        for label in recall_targets
    }
    meets_recall_targets = all(recalls.get(label, 0.0) >= target for label, target in recall_targets.items())
    meets_risk_target = (
        True if max_selective_risk is None else selective_risk_upper_bound <= max_selective_risk
    )
    meets_coverage_target = True if min_coverage is None else coverage >= min_coverage
    meets_constraints = (
        meets_recall_targets
        and meets_risk_target
        and meets_coverage_target
        and authoritative_count > 0
    )
    average_target_recall = _safe_ratio(sum(recalls.values()), len(recalls)) if recalls else 0.0
    return {
        "meets_constraints": meets_constraints,
        "average_target_recall": average_target_recall,
        "coverage": coverage,
        "selective_risk": selective_risk,
        "selective_risk_upper_bound": selective_risk_upper_bound,
        "authoritative_precision": authoritative_precision,
        "selective_utility": selective_utility,
        "authoritative_count": float(authoritative_count),
        "authoritative_errors": float(authoritative_errors),
    }


def _normalize_known_route_label(label: str) -> str:
    normalized = normalize_route_label(label)
    if normalized not in _CANONICAL_ROUTE_LABELS:
        raise ValueError(f"Unknown route label: {label!r}")
    return normalized


def _dedupe_preserve_order(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _empty_per_label_metrics() -> dict[str, float]:
    return {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 0.0,
        "predicted_support": 0.0,
        "authoritative_precision": 0.0,
        "authoritative_recall": 0.0,
        "authority_coverage": 0.0,
        "authoritative_support": 0.0,
    }


def _validate_unit_interval(value: float, *, name: str) -> float:
    if math.isnan(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} values must be within [0.0, 1.0], got {value!r}.")
    return value


def _validate_open_unit_interval(value: float, *, name: str) -> float:
    if math.isnan(value) or value <= 0.0 or value >= 1.0:
        raise ValueError(f"{name} must be within (0.0, 1.0), got {value!r}.")
    return value


def _validate_non_negative(value: float, *, name: str) -> float:
    if math.isnan(value) or value < 0.0:
        raise ValueError(f"{name} values must be >= 0.0, got {value!r}.")
    return value


def _harmonic_mean(left: float, right: float) -> float:
    return _safe_ratio(2.0 * left * right, left + right)


def _wilson_upper_bound(errors: int, total: int, *, confidence: float = _DEFAULT_RISK_CONFIDENCE) -> float:
    """Return the Wilson score upper bound for a Bernoulli error rate."""

    if total <= 0:
        return 1.0
    _validate_open_unit_interval(confidence, name="confidence")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    phat = float(errors) / float(total)
    z2_over_n = (z * z) / float(total)
    denominator = 1.0 + z2_over_n
    centre = phat + 0.5 * z2_over_n
    margin = z * math.sqrt((phat * (1.0 - phat) + 0.25 * z2_over_n) / float(total))
    return min(1.0, max(0.0, (centre + margin) / denominator))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)