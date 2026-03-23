"""Load labeled user-intent datasets and evaluate the first router stage."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from .user_intent import USER_INTENT_LABEL_VALUES, UserIntentDecision, normalize_user_intent_label


@dataclass(frozen=True, slots=True)
class LabeledUserIntentSample:
    """Store one labeled transcript for user-intent training or evaluation."""

    text: str
    label: str
    sample_id: str | None = None
    split: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", str(self.text or "").strip())
        object.__setattr__(self, "label", normalize_user_intent_label(self.label))
        object.__setattr__(self, "sample_id", str(self.sample_id or "").strip() or None)
        object.__setattr__(self, "split", str(self.split or "").strip().lower() or None)
        object.__setattr__(self, "source", str(self.source or "").strip() or None)
        if not self.text:
            raise ValueError("LabeledUserIntentSample.text must not be empty.")


@dataclass(frozen=True, slots=True)
class ScoredUserIntentRecord:
    """Store one gold user intent together with one scored router decision."""

    sample: LabeledUserIntentSample
    decision: UserIntentDecision


@dataclass(frozen=True, slots=True)
class UserIntentRouterEvaluation:
    """Summarize user-intent quality over one labeled evaluation slice."""

    total: int
    accuracy: float
    macro_f1: float
    confusion_matrix: dict[str, dict[str, int]]
    per_label: dict[str, dict[str, float]]


def load_labeled_user_intent_samples(path: str | Path) -> tuple[LabeledUserIntentSample, ...]:
    """Load labeled user-intent samples from one JSONL file."""

    samples: list[LabeledUserIntentSample] = []
    file_path = Path(path)
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        samples.append(
            LabeledUserIntentSample(
                text=payload["text"],
                label=payload["label"],
                sample_id=payload.get("id"),
                split=payload.get("split"),
                source=payload.get("source"),
            )
        )
    return tuple(samples)


def split_labeled_user_intent_samples(
    samples: Sequence[LabeledUserIntentSample],
) -> dict[str, tuple[LabeledUserIntentSample, ...]]:
    """Group user-intent samples by their explicit `split` field."""

    grouped: dict[str, list[LabeledUserIntentSample]] = {}
    for sample in samples:
        key = sample.split or "unspecified"
        grouped.setdefault(key, []).append(sample)
    return {key: tuple(values) for key, values in grouped.items()}


def score_user_intent_router(
    router,
    samples: Sequence[LabeledUserIntentSample],
) -> tuple[ScoredUserIntentRecord, ...]:
    """Run the user-intent router over labeled samples and return scored records."""

    return tuple(
        ScoredUserIntentRecord(sample=sample, decision=router.classify(sample.text))
        for sample in samples
    )


def evaluate_user_intent_records(
    records: Sequence[ScoredUserIntentRecord],
) -> UserIntentRouterEvaluation:
    """Compute classification metrics for scored user-intent records."""

    confusion_matrix = {
        gold: {predicted: 0 for predicted in USER_INTENT_LABEL_VALUES}
        for gold in USER_INTENT_LABEL_VALUES
    }
    total = len(records)
    correct = 0
    for record in records:
        gold = record.sample.label
        predicted = record.decision.label
        confusion_matrix[gold][predicted] += 1
        if predicted == gold:
            correct += 1
    per_label: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for label in USER_INTENT_LABEL_VALUES:
        true_positive = confusion_matrix[label][label]
        false_positive = sum(confusion_matrix[gold][label] for gold in USER_INTENT_LABEL_VALUES if gold != label)
        false_negative = sum(confusion_matrix[label][pred] for pred in USER_INTENT_LABEL_VALUES if pred != label)
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
    return UserIntentRouterEvaluation(
        total=total,
        accuracy=_safe_ratio(correct, total),
        macro_f1=_safe_ratio(sum(f1_values), len(f1_values)),
        confusion_matrix=confusion_matrix,
        per_label=per_label,
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)
