"""Define the authority-gating policy for local semantic route decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .contracts import ROUTE_LABEL_VALUES, normalize_route_label


@dataclass(frozen=True, slots=True)
class SemanticRouterPolicy:
    """Store the calibration policy applied on top of local route scores."""

    thresholds: Mapping[str, float]
    authoritative_labels: tuple[str, ...] = ("web", "memory", "tool")
    min_margin: float = 0.0

    def __post_init__(self) -> None:
        normalized_thresholds = {
            normalize_route_label(label): max(0.0, min(1.0, float(value)))
            for label, value in dict(self.thresholds).items()
        }
        for label in ROUTE_LABEL_VALUES:
            normalized_thresholds.setdefault(label, 0.0)
        normalized_authoritative = tuple(
            normalize_route_label(label) for label in self.authoritative_labels
        )
        object.__setattr__(self, "thresholds", normalized_thresholds)
        object.__setattr__(self, "authoritative_labels", normalized_authoritative)
        object.__setattr__(self, "min_margin", max(0.0, float(self.min_margin)))

    def threshold_for(self, label: str) -> float:
        """Return the configured confidence threshold for one label."""

        return float(self.thresholds[normalize_route_label(label)])

    def evaluate(
        self,
        label: str,
        *,
        confidence: float,
        margin: float,
    ) -> tuple[bool, str | None]:
        """Return whether one scored route may bypass the supervisor lane."""

        normalized_label = normalize_route_label(label)
        if normalized_label not in self.authoritative_labels:
            return False, "label_not_authoritative"
        if float(confidence) < self.threshold_for(normalized_label):
            return False, "below_confidence_threshold"
        if float(margin) < self.min_margin:
            return False, "below_margin_threshold"
        return True, None
