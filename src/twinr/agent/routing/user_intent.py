"""Define user-centered routing labels and two-stage routing contracts.

The local backend router uses technical execution labels such as ``memory`` or
``tool``. This module adds the user-facing semantic layer that better matches
how a person naturally frames a request before Twinr decides which backend path
should answer it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .contracts import SemanticRouteDecision, normalize_route_label


class UserIntentLabel(str, Enum):
    """Enumerate the user-centered routing classes."""

    WISSEN = "wissen"
    NACHSCHAUEN = "nachschauen"
    PERSOENLICH = "persoenlich"
    MACHEN_ODER_PRUEFEN = "machen_oder_pruefen"


USER_INTENT_LABEL_VALUES: tuple[str, ...] = tuple(label.value for label in UserIntentLabel)
_USER_INTENT_LABEL_SET = frozenset(USER_INTENT_LABEL_VALUES)

DEFAULT_USER_INTENT_BY_ROUTE_LABEL: Mapping[str, str] = {
    "parametric": UserIntentLabel.WISSEN.value,
    "web": UserIntentLabel.NACHSCHAUEN.value,
    "memory": UserIntentLabel.PERSOENLICH.value,
    "tool": UserIntentLabel.MACHEN_ODER_PRUEFEN.value,
}

ALLOWED_ROUTE_LABELS_BY_USER_INTENT: Mapping[str, tuple[str, ...]] = {
    UserIntentLabel.WISSEN.value: ("parametric",),
    UserIntentLabel.NACHSCHAUEN.value: ("web",),
    UserIntentLabel.PERSOENLICH.value: ("memory", "tool"),
    UserIntentLabel.MACHEN_ODER_PRUEFEN.value: ("tool",),
}


def normalize_user_intent_label(value: object) -> str:
    """Return one validated user-centered label string."""

    normalized = str(value or "").strip().lower()
    if normalized not in _USER_INTENT_LABEL_SET:
        raise ValueError(f"Unsupported user intent label: {value!r}")
    return normalized


def default_user_intent_for_route_label(route_label: str) -> str:
    """Map one backend route label to its default user-centered class."""

    normalized_route_label = normalize_route_label(route_label)
    return DEFAULT_USER_INTENT_BY_ROUTE_LABEL[normalized_route_label]


def allowed_route_labels_for_user_intent(user_intent_label: str) -> tuple[str, ...]:
    """Return the backend labels that remain valid for one user intent."""

    normalized_user_label = normalize_user_intent_label(user_intent_label)
    return ALLOWED_ROUTE_LABELS_BY_USER_INTENT[normalized_user_label]


@dataclass(frozen=True, slots=True)
class UserIntentDecision:
    """Store one scored user-centered routing result."""

    label: str
    confidence: float
    margin: float
    scores: Mapping[str, float]
    model_id: str
    latency_ms: float

    def __post_init__(self) -> None:
        normalized_label = normalize_user_intent_label(self.label)
        normalized_scores = {
            normalize_user_intent_label(key): float(value)
            for key, value in dict(self.scores).items()
        }
        missing = _USER_INTENT_LABEL_SET.difference(normalized_scores)
        if missing:
            raise ValueError(f"UserIntentDecision.scores missing labels: {sorted(missing)}")
        object.__setattr__(self, "label", normalized_label)
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "margin", float(self.margin))
        object.__setattr__(self, "scores", normalized_scores)
        object.__setattr__(self, "model_id", str(self.model_id or "").strip())
        object.__setattr__(self, "latency_ms", float(self.latency_ms))

    def score_for(self, label: str) -> float:
        """Return the score for one normalized user-centered label."""

        return float(self.scores[normalize_user_intent_label(label)])

    def ranked_labels(self) -> tuple[str, ...]:
        """Return labels ordered by descending score."""

        return tuple(
            label
            for label, _score in sorted(
                self.scores.items(),
                key=lambda item: (-float(item[1]), item[0]),
            )
        )


@dataclass(frozen=True, slots=True)
class TwoStageSemanticRouteDecision:
    """Store the stage-1 user intent and the stage-2 backend route together."""

    user_intent: UserIntentDecision
    route_decision: SemanticRouteDecision
    allowed_route_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        normalized_allowed = tuple(
            normalize_route_label(label)
            for label in self.allowed_route_labels
        )
        if not normalized_allowed:
            raise ValueError("TwoStageSemanticRouteDecision.allowed_route_labels must not be empty.")
        object.__setattr__(self, "allowed_route_labels", normalized_allowed)
