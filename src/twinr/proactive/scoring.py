from __future__ import annotations

from dataclasses import dataclass


def clamp_unit(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def bool_score(active: bool) -> float:
    return 1.0 if active else 0.0


def hold_progress(now: float, since: float | None, target_s: float) -> float:
    if since is None:
        return 0.0
    if target_s <= 0:
        return 1.0
    return clamp_unit((now - since) / target_s)


def recent_progress(now: float, anchor_at: float | None, window_s: float) -> float:
    if anchor_at is None:
        return 0.0
    if window_s <= 0:
        return 1.0
    return clamp_unit(1.0 - ((now - anchor_at) / window_s))


@dataclass(frozen=True, slots=True)
class TriggerScoreEvidence:
    key: str
    value: float
    weight: float
    detail: str

    @property
    def contribution(self) -> float:
        return max(0.0, self.weight) * clamp_unit(self.value)

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "value": round(clamp_unit(self.value), 3),
            "weight": round(max(0.0, self.weight), 3),
            "contribution": round(self.contribution, 3),
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class WeightedTriggerScore:
    score: float
    threshold: float
    evidence: tuple[TriggerScoreEvidence, ...]

    @property
    def passed(self) -> bool:
        return self.score >= self.threshold


def weighted_trigger_score(
    *,
    threshold: float,
    evidence: tuple[TriggerScoreEvidence, ...],
) -> WeightedTriggerScore:
    total_weight = sum(max(0.0, item.weight) for item in evidence)
    if total_weight <= 0:
        score = 0.0
    else:
        score = sum(item.contribution for item in evidence) / total_weight
    return WeightedTriggerScore(
        score=round(clamp_unit(score), 4),
        threshold=round(clamp_unit(threshold), 4),
        evidence=evidence,
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
