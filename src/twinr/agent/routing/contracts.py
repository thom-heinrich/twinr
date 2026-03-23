"""Define normalized route labels and scored local routing decisions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from .policy import SemanticRouterPolicy


class RouteLabel(str, Enum):
    """Enumerate the supported local semantic routing classes."""

    PARAMETRIC = "parametric"
    WEB = "web"
    MEMORY = "memory"
    TOOL = "tool"


ROUTE_LABEL_VALUES: tuple[str, ...] = tuple(label.value for label in RouteLabel)
_ROUTE_LABEL_SET = frozenset(ROUTE_LABEL_VALUES)


def normalize_route_label(value: object) -> str:
    """Return one validated route label string."""

    normalized = str(value or "").strip().lower()
    if normalized not in _ROUTE_LABEL_SET:
        raise ValueError(f"Unsupported semantic route label: {value!r}")
    return normalized


@dataclass(frozen=True, slots=True)
class SemanticRouteDecision:
    """Store one scored local semantic-routing result.

    Attributes:
        label: Highest-scoring route label.
        confidence: Softmax confidence for the selected label.
        margin: Confidence delta between the top-1 and top-2 labels.
        scores: Probability-like score distribution keyed by label.
        model_id: Human-readable model/bundle identifier.
        latency_ms: End-to-end local classification latency in milliseconds.
        authoritative: Whether the current routing policy allows this decision
            to bypass the supervisor lane.
        fallback_reason: Short machine-readable reason when the route is not
            authoritative.
    """

    label: str
    confidence: float
    margin: float
    scores: Mapping[str, float]
    model_id: str
    latency_ms: float
    authoritative: bool = False
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        normalized_label = normalize_route_label(self.label)
        normalized_scores = {
            normalize_route_label(key): float(value)
            for key, value in dict(self.scores).items()
        }
        missing = _ROUTE_LABEL_SET.difference(normalized_scores)
        if missing:
            raise ValueError(f"SemanticRouteDecision.scores missing labels: {sorted(missing)}")
        object.__setattr__(self, "label", normalized_label)
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "margin", float(self.margin))
        object.__setattr__(self, "scores", normalized_scores)
        object.__setattr__(self, "latency_ms", float(self.latency_ms))
        object.__setattr__(self, "model_id", str(self.model_id or "").strip())
        normalized_reason = str(self.fallback_reason or "").strip() or None
        object.__setattr__(self, "fallback_reason", normalized_reason)

    def score_for(self, label: str) -> float:
        """Return the score for one normalized route label."""

        return float(self.scores[normalize_route_label(label)])

    def ranked_labels(self) -> tuple[str, ...]:
        """Return labels ordered by descending score."""

        return tuple(
            label
            for label, _score in sorted(
                self.scores.items(),
                key=lambda item: (-float(item[1]), item[0]),
            )
        )

    def with_policy(self, policy: SemanticRouterPolicy) -> "SemanticRouteDecision":
        """Return a copy with `authoritative` derived from one policy."""

        authoritative, fallback_reason = policy.evaluate(
            self.label,
            confidence=self.confidence,
            margin=self.margin,
        )
        return replace(
            self,
            authoritative=authoritative,
            fallback_reason=fallback_reason,
        )
