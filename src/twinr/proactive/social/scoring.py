"""Score proactive social-trigger evidence with bounded numeric helpers.

This module normalizes evidence weights and timing-derived hold signals before
the trigger engine turns them into one comparable score card.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable


def _finite_float(value: object) -> float | None:
    """Return one finite float or ``None`` for invalid input."""

    # AUDIT-FIX(#1): Centralize numeric sanitization so NaN/inf/invalid values fail closed instead of leaking into scores/JSON payloads.
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not isfinite(numeric):
        return None
    return numeric


def _non_negative_float(value: object) -> float:
    """Return one finite non-negative float or ``0.0``."""

    # AUDIT-FIX(#1): Keep weights finite and non-negative to prevent NaN/inf propagation through weighted averages.
    numeric = _finite_float(value)
    if numeric is None or numeric <= 0.0:
        return 0.0
    return numeric


def _normalize_evidence(
    evidence: Iterable["TriggerScoreEvidence"] | None,
) -> tuple["TriggerScoreEvidence", ...]:
    """Materialize and type-filter one evidence iterable."""

    # AUDIT-FIX(#2): Materialize evidence exactly once so generators cannot be exhausted and mutable inputs cannot alias internal state.
    if evidence is None:
        return ()
    try:
        materialized = tuple(evidence)
    except TypeError:
        return ()
    return tuple(item for item in materialized if isinstance(item, TriggerScoreEvidence))


def clamp_unit(value: float) -> float:
    """Clamp one numeric value into the closed unit interval."""

    numeric = _finite_float(value)  # AUDIT-FIX(#1): Treat invalid/non-finite inputs as safe-zero instead of returning NaN.
    if numeric is None or numeric <= 0.0:
        return 0.0
    if numeric >= 1.0:
        return 1.0
    return numeric


def bool_score(active: bool) -> float:
    """Convert one boolean signal into a unit score."""

    return 1.0 if active else 0.0


def hold_progress(now: float, since: float | None, target_s: float) -> float:
    """Return how far one active hold progressed toward its target."""

    if since is None:
        return 0.0

    now_value = _finite_float(now)  # AUDIT-FIX(#1): Guard timing math against invalid/non-finite caller inputs.
    since_value = _finite_float(since)
    target_value = _finite_float(target_s)
    if now_value is None or since_value is None or target_value is None:
        return 0.0
    if target_value < 0.0:
        return 0.0  # AUDIT-FIX(#3): Negative durations are invalid configuration and must fail closed.
    if target_value == 0.0:
        return 1.0
    return clamp_unit((now_value - since_value) / target_value)


def recent_progress(now: float, anchor_at: float | None, window_s: float) -> float:
    """Return how recent one anchor timestamp is within one window."""

    if anchor_at is None:
        return 0.0

    now_value = _finite_float(now)  # AUDIT-FIX(#1): Guard timing math against invalid/non-finite caller inputs.
    anchor_value = _finite_float(anchor_at)
    window_value = _finite_float(window_s)
    if now_value is None or anchor_value is None or window_value is None:
        return 0.0
    if window_value < 0.0:
        return 0.0  # AUDIT-FIX(#3): Negative recency windows are invalid configuration and must fail closed.
    if window_value == 0.0:
        return 1.0
    return clamp_unit(1.0 - ((now_value - anchor_value) / window_value))


@dataclass(frozen=True, slots=True)
class TriggerScoreEvidence:
    """Describe one weighted evidence item used in trigger scoring."""

    key: str
    value: float
    weight: float
    detail: str

    def __post_init__(self) -> None:
        """Normalize stored value and weight after construction."""

        object.__setattr__(self, "value", clamp_unit(self.value))  # AUDIT-FIX(#1): Persist only sanitized unit values to keep the dataclass internally consistent.
        object.__setattr__(self, "weight", _non_negative_float(self.weight))  # AUDIT-FIX(#1): Persist only finite, non-negative weights.

    @property
    def contribution(self) -> float:
        """Return the weighted contribution of this evidence item."""

        return self.weight * self.value

    def to_dict(self) -> dict[str, object]:
        """Serialize one evidence item for logs or JSON payloads."""

        return {
            "key": self.key,
            "value": round(self.value, 3),
            "weight": round(self.weight, 3),
            "contribution": round(self.contribution, 3),
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class WeightedTriggerScore:
    """Represent one scored trigger candidate and its normalized evidence."""

    score: float
    threshold: float
    evidence: tuple[TriggerScoreEvidence, ...]

    def __post_init__(self) -> None:
        """Normalize score, threshold, and evidence after construction."""

        object.__setattr__(self, "score", clamp_unit(self.score))  # AUDIT-FIX(#1): Prevent direct construction of invalid/non-finite scores.
        object.__setattr__(self, "threshold", clamp_unit(self.threshold))  # AUDIT-FIX(#1): Prevent direct construction of invalid/non-finite thresholds.
        object.__setattr__(self, "evidence", _normalize_evidence(self.evidence))  # AUDIT-FIX(#2): Enforce immutable, replay-safe evidence storage.

    @property
    def passed(self) -> bool:
        """Return whether the score meets or exceeds the threshold."""

        return self.score >= self.threshold


def weighted_trigger_score(
    *,
    threshold: float,
    evidence: Iterable[TriggerScoreEvidence] | None,
) -> WeightedTriggerScore:
    """Compute one weighted trigger score from normalized evidence."""

    evidence_items = _normalize_evidence(evidence)  # AUDIT-FIX(#2): Evaluate over a stable immutable snapshot of the evidence.
    total_weight = sum(item.weight for item in evidence_items)
    if total_weight <= 0.0:
        score = 0.0
    else:
        score = sum(item.contribution for item in evidence_items) / total_weight
    return WeightedTriggerScore(
        score=score,  # AUDIT-FIX(#4): Keep full precision internally; round only at presentation boundaries.
        threshold=threshold,  # AUDIT-FIX(#4): Keep full precision internally; round only at presentation boundaries.
        evidence=evidence_items,
    )


__all__ = [
    "TriggerScoreEvidence",
    "WeightedTriggerScore",
    "bool_score",
    "clamp_unit",
    "hold_progress",
    "recent_progress",
    "weighted_trigger_score",
]
