"""Define the authority-gating policy for local semantic route decisions."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Final, TypeAlias

from .contracts import ROUTE_LABEL_VALUES, normalize_route_label


# CHANGELOG: 2026-03-27
# BUG-1: Reject non-finite / out-of-range confidence, margin, and config values; the old code could silently authorize NaN scores and silently clamp broken config.
# BUG-2: evaluate() now fails closed on invalid route labels instead of propagating exceptions into the caller.
# BUG-3: thresholds and margin_thresholds are now truly immutable; the old "frozen" dataclass was still mutable through the inner dict.
# SEC-1: Missing thresholds for authoritative labels now default deny instead of fail-open.
# SEC-2: Added optional conformal p-value / prediction-set-size checks so privileged lanes can require calibrated trust evidence.
# IMP-1: Added per-label margin thresholds and a structured SemanticRouteDecision audit record.
# IMP-2: Added frontier-ready hooks for calibrated/conformal routing while preserving evaluate() as a drop-in API.


DenialReason: TypeAlias = str

_REASON_INVALID_LABEL: Final[DenialReason] = "invalid_label"
_REASON_LABEL_NOT_AUTHORITATIVE: Final[DenialReason] = "label_not_authoritative"
_REASON_THRESHOLD_NOT_CONFIGURED: Final[DenialReason] = "threshold_not_configured"
_REASON_INVALID_CONFIDENCE: Final[DenialReason] = "invalid_confidence"
_REASON_BELOW_CONFIDENCE_THRESHOLD: Final[DenialReason] = "below_confidence_threshold"
_REASON_INVALID_MARGIN: Final[DenialReason] = "invalid_margin"
_REASON_BELOW_MARGIN_THRESHOLD: Final[DenialReason] = "below_margin_threshold"
_REASON_INVALID_PREDICTION_SET_SIZE: Final[DenialReason] = "invalid_prediction_set_size"
_REASON_MISSING_PREDICTION_SET_SIZE: Final[DenialReason] = "missing_prediction_set_size"
_REASON_PREDICTION_SET_TOO_LARGE: Final[DenialReason] = "prediction_set_too_large"
_REASON_INVALID_CONFORMAL_P_VALUE: Final[DenialReason] = "invalid_conformal_p_value"
_REASON_MISSING_CONFORMAL_P_VALUE: Final[DenialReason] = "missing_conformal_p_value"
_REASON_BELOW_CONFORMAL_P_VALUE_THRESHOLD: Final[DenialReason] = (
    "below_conformal_p_value_threshold"
)


class FrozenMapping(Mapping[str, float]):
    """Minimal immutable mapping with normal Mapping semantics."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, float]) -> None:
        self._data = dict(data)

    def __getitem__(self, key: str) -> float:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenMapping({self._data!r})"


def _coerce_probability(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite float in [0.0, 1.0]")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float in [0.0, 1.0]") from exc
    if not isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must be a finite float in [0.0, 1.0]")
    return numeric


def _coerce_non_negative_float(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite float >= 0.0")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float >= 0.0") from exc
    if not isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be a finite float >= 0.0")
    return numeric


def _coerce_optional_probability(name: str, value: object | None) -> float | None:
    if value is None:
        return None
    return _coerce_probability(name, value)


def _coerce_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= 1")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer >= 1") from exc
    if numeric < 1:
        raise ValueError(f"{name} must be an integer >= 1")
    return numeric


def _coerce_optional_positive_int(name: str, value: object | None) -> int | None:
    if value is None:
        return None
    return _coerce_positive_int(name, value)


def _normalize_threshold_mapping(
    raw_mapping: Mapping[str, float],
    *,
    value_name: str,
    value_parser: Callable[[str, object], float],
) -> tuple[dict[str, float], frozenset[str]]:
    normalized: dict[str, float] = {}
    for raw_label, raw_value in dict(raw_mapping).items():
        label = normalize_route_label(raw_label)
        if label in normalized:
            raise ValueError(
                f"{value_name} contains duplicate labels after normalization: {label!r}"
            )
        normalized[label] = value_parser(f"{value_name}[{label!r}]", raw_value)
    return normalized, frozenset(normalized)


@dataclass(frozen=True, slots=True)
class SemanticRouteDecision:
    """Structured audit record for one authority-gating decision."""

    allow: bool
    reason: DenialReason | None
    label: str
    confidence: float | None
    confidence_threshold: float | None
    margin: float | None
    margin_threshold: float | None
    prediction_set_size: int | None = None
    max_prediction_set_size: int | None = None
    conformal_p_value: float | None = None
    min_conformal_p_value: float | None = None

    def as_tuple(self) -> tuple[bool, str | None]:
        """Return the legacy `(allow, reason)` API."""
        return self.allow, self.reason


@dataclass(frozen=True, slots=True)
class SemanticRouterPolicy:
    """Store the calibrated policy applied on top of local route scores."""

    thresholds: Mapping[str, float]
    authoritative_labels: tuple[str, ...] = ("web", "memory", "tool")
    min_margin: float = 0.0
    margin_thresholds: Mapping[str, float] = field(default_factory=dict)
    max_prediction_set_size: int | None = None
    min_conformal_p_value: float | None = None

    # BREAKING: authoritative labels without an explicit threshold now deny by
    # default. Set this to False to restore the legacy fail-open behavior.
    require_explicit_thresholds: bool = True

    _configured_threshold_labels: frozenset[str] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _authoritative_label_set: frozenset[str] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        # BREAKING: invalid threshold values now raise ValueError instead of
        # being silently clamped into a potentially unsafe policy.
        normalized_thresholds, configured_threshold_labels = _normalize_threshold_mapping(
            self.thresholds,
            value_name="thresholds",
            value_parser=_coerce_probability,
        )
        normalized_margin_thresholds, _ = _normalize_threshold_mapping(
            self.margin_thresholds,
            value_name="margin_thresholds",
            value_parser=_coerce_non_negative_float,
        )

        normalized_authoritative = tuple(
            dict.fromkeys(
                normalize_route_label(label) for label in self.authoritative_labels
            )
        )
        authoritative_label_set = frozenset(normalized_authoritative)
        min_margin = _coerce_non_negative_float("min_margin", self.min_margin)
        min_conformal_p_value = _coerce_optional_probability(
            "min_conformal_p_value",
            self.min_conformal_p_value,
        )
        max_prediction_set_size = _coerce_optional_positive_int(
            "max_prediction_set_size",
            self.max_prediction_set_size,
        )

        default_threshold = 1.0 if self.require_explicit_thresholds else 0.0
        for label in ROUTE_LABEL_VALUES:
            normalized_thresholds.setdefault(label, default_threshold)

        for label in ROUTE_LABEL_VALUES:
            normalized_margin_thresholds.setdefault(label, min_margin)

        # BREAKING: thresholds and margin_thresholds are now exposed as immutable
        # mappings. Attempts to mutate them after construction now raise
        # TypeError.
        object.__setattr__(self, "thresholds", FrozenMapping(normalized_thresholds))
        object.__setattr__(
            self,
            "margin_thresholds",
            FrozenMapping(normalized_margin_thresholds),
        )
        object.__setattr__(self, "authoritative_labels", normalized_authoritative)
        object.__setattr__(self, "min_margin", min_margin)
        object.__setattr__(self, "min_conformal_p_value", min_conformal_p_value)
        object.__setattr__(self, "max_prediction_set_size", max_prediction_set_size)
        object.__setattr__(
            self,
            "_configured_threshold_labels",
            configured_threshold_labels,
        )
        object.__setattr__(self, "_authoritative_label_set", authoritative_label_set)

    def threshold_for(self, label: str) -> float:
        """Return the configured confidence threshold for one label."""
        return float(self.thresholds[normalize_route_label(label)])

    def margin_threshold_for(self, label: str) -> float:
        """Return the configured margin threshold for one label."""
        return float(self.margin_thresholds[normalize_route_label(label)])

    def decide(
        self,
        label: str,
        *,
        confidence: float,
        margin: float,
        prediction_set_size: int | None = None,
        conformal_p_value: float | None = None,
    ) -> SemanticRouteDecision:
        """Return a structured decision for one scored route."""

        raw_label = str(label)
        try:
            normalized_label = normalize_route_label(label)
        except (TypeError, ValueError):
            return SemanticRouteDecision(
                allow=False,
                reason=_REASON_INVALID_LABEL,
                label=raw_label,
                confidence=None,
                confidence_threshold=None,
                margin=None,
                margin_threshold=None,
                prediction_set_size=None,
                max_prediction_set_size=self.max_prediction_set_size,
                conformal_p_value=None,
                min_conformal_p_value=self.min_conformal_p_value,
            )

        confidence_threshold = self.threshold_for(normalized_label)
        margin_threshold = self.margin_threshold_for(normalized_label)

        if normalized_label not in self._authoritative_label_set:
            return SemanticRouteDecision(
                allow=False,
                reason=_REASON_LABEL_NOT_AUTHORITATIVE,
                label=normalized_label,
                confidence=None,
                confidence_threshold=confidence_threshold,
                margin=None,
                margin_threshold=margin_threshold,
                prediction_set_size=None,
                max_prediction_set_size=self.max_prediction_set_size,
                conformal_p_value=None,
                min_conformal_p_value=self.min_conformal_p_value,
            )

        if (
            self.require_explicit_thresholds
            and normalized_label not in self._configured_threshold_labels
        ):
            return SemanticRouteDecision(
                allow=False,
                reason=_REASON_THRESHOLD_NOT_CONFIGURED,
                label=normalized_label,
                confidence=None,
                confidence_threshold=confidence_threshold,
                margin=None,
                margin_threshold=margin_threshold,
                prediction_set_size=None,
                max_prediction_set_size=self.max_prediction_set_size,
                conformal_p_value=None,
                min_conformal_p_value=self.min_conformal_p_value,
            )

        try:
            normalized_confidence = _coerce_probability("confidence", confidence)
        except ValueError:
            return SemanticRouteDecision(
                allow=False,
                reason=_REASON_INVALID_CONFIDENCE,
                label=normalized_label,
                confidence=None,
                confidence_threshold=confidence_threshold,
                margin=None,
                margin_threshold=margin_threshold,
                prediction_set_size=None,
                max_prediction_set_size=self.max_prediction_set_size,
                conformal_p_value=None,
                min_conformal_p_value=self.min_conformal_p_value,
            )

        try:
            normalized_margin = _coerce_non_negative_float("margin", margin)
        except ValueError:
            return SemanticRouteDecision(
                allow=False,
                reason=_REASON_INVALID_MARGIN,
                label=normalized_label,
                confidence=normalized_confidence,
                confidence_threshold=confidence_threshold,
                margin=None,
                margin_threshold=margin_threshold,
                prediction_set_size=None,
                max_prediction_set_size=self.max_prediction_set_size,
                conformal_p_value=None,
                min_conformal_p_value=self.min_conformal_p_value,
            )

        normalized_prediction_set_size = None
        if self.max_prediction_set_size is not None:
            try:
                normalized_prediction_set_size = _coerce_optional_positive_int(
                    "prediction_set_size",
                    prediction_set_size,
                )
            except ValueError:
                return SemanticRouteDecision(
                    allow=False,
                    reason=_REASON_INVALID_PREDICTION_SET_SIZE,
                    label=normalized_label,
                    confidence=normalized_confidence,
                    confidence_threshold=confidence_threshold,
                    margin=normalized_margin,
                    margin_threshold=margin_threshold,
                    prediction_set_size=None,
                    max_prediction_set_size=self.max_prediction_set_size,
                    conformal_p_value=None,
                    min_conformal_p_value=self.min_conformal_p_value,
                )

            if normalized_prediction_set_size is None:
                return SemanticRouteDecision(
                    allow=False,
                    reason=_REASON_MISSING_PREDICTION_SET_SIZE,
                    label=normalized_label,
                    confidence=normalized_confidence,
                    confidence_threshold=confidence_threshold,
                    margin=normalized_margin,
                    margin_threshold=margin_threshold,
                    prediction_set_size=None,
                    max_prediction_set_size=self.max_prediction_set_size,
                    conformal_p_value=None,
                    min_conformal_p_value=self.min_conformal_p_value,
                )

            if normalized_prediction_set_size > self.max_prediction_set_size:
                return SemanticRouteDecision(
                    allow=False,
                    reason=_REASON_PREDICTION_SET_TOO_LARGE,
                    label=normalized_label,
                    confidence=normalized_confidence,
                    confidence_threshold=confidence_threshold,
                    margin=normalized_margin,
                    margin_threshold=margin_threshold,
                    prediction_set_size=normalized_prediction_set_size,
                    max_prediction_set_size=self.max_prediction_set_size,
                    conformal_p_value=None,
                    min_conformal_p_value=self.min_conformal_p_value,
                )

        normalized_conformal_p_value = None
        if self.min_conformal_p_value is not None:
            try:
                normalized_conformal_p_value = _coerce_optional_probability(
                    "conformal_p_value",
                    conformal_p_value,
                )
            except ValueError:
                return SemanticRouteDecision(
                    allow=False,
                    reason=_REASON_INVALID_CONFORMAL_P_VALUE,
                    label=normalized_label,
                    confidence=normalized_confidence,
                    confidence_threshold=confidence_threshold,
                    margin=normalized_margin,
                    margin_threshold=margin_threshold,
                    prediction_set_size=normalized_prediction_set_size,
                    max_prediction_set_size=self.max_prediction_set_size,
                    conformal_p_value=None,
                    min_conformal_p_value=self.min_conformal_p_value,
                )

            if normalized_conformal_p_value is None:
                return SemanticRouteDecision(
                    allow=False,
                    reason=_REASON_MISSING_CONFORMAL_P_VALUE,
                    label=normalized_label,
                    confidence=normalized_confidence,
                    confidence_threshold=confidence_threshold,
                    margin=normalized_margin,
                    margin_threshold=margin_threshold,
                    prediction_set_size=normalized_prediction_set_size,
                    max_prediction_set_size=self.max_prediction_set_size,
                    conformal_p_value=None,
                    min_conformal_p_value=self.min_conformal_p_value,
                )

            if normalized_conformal_p_value < self.min_conformal_p_value:
                return SemanticRouteDecision(
                    allow=False,
                    reason=_REASON_BELOW_CONFORMAL_P_VALUE_THRESHOLD,
                    label=normalized_label,
                    confidence=normalized_confidence,
                    confidence_threshold=confidence_threshold,
                    margin=normalized_margin,
                    margin_threshold=margin_threshold,
                    prediction_set_size=normalized_prediction_set_size,
                    max_prediction_set_size=self.max_prediction_set_size,
                    conformal_p_value=normalized_conformal_p_value,
                    min_conformal_p_value=self.min_conformal_p_value,
                )

        if normalized_confidence < confidence_threshold:
            return SemanticRouteDecision(
                allow=False,
                reason=_REASON_BELOW_CONFIDENCE_THRESHOLD,
                label=normalized_label,
                confidence=normalized_confidence,
                confidence_threshold=confidence_threshold,
                margin=normalized_margin,
                margin_threshold=margin_threshold,
                prediction_set_size=normalized_prediction_set_size,
                max_prediction_set_size=self.max_prediction_set_size,
                conformal_p_value=normalized_conformal_p_value,
                min_conformal_p_value=self.min_conformal_p_value,
            )

        if normalized_margin < margin_threshold:
            return SemanticRouteDecision(
                allow=False,
                reason=_REASON_BELOW_MARGIN_THRESHOLD,
                label=normalized_label,
                confidence=normalized_confidence,
                confidence_threshold=confidence_threshold,
                margin=normalized_margin,
                margin_threshold=margin_threshold,
                prediction_set_size=normalized_prediction_set_size,
                max_prediction_set_size=self.max_prediction_set_size,
                conformal_p_value=normalized_conformal_p_value,
                min_conformal_p_value=self.min_conformal_p_value,
            )

        return SemanticRouteDecision(
            allow=True,
            reason=None,
            label=normalized_label,
            confidence=normalized_confidence,
            confidence_threshold=confidence_threshold,
            margin=normalized_margin,
            margin_threshold=margin_threshold,
            prediction_set_size=normalized_prediction_set_size,
            max_prediction_set_size=self.max_prediction_set_size,
            conformal_p_value=normalized_conformal_p_value,
            min_conformal_p_value=self.min_conformal_p_value,
        )

    def evaluate(
        self,
        label: str,
        *,
        confidence: float,
        margin: float,
        prediction_set_size: int | None = None,
        conformal_p_value: float | None = None,
    ) -> tuple[bool, str | None]:
        """Return whether one scored route may bypass the supervisor lane."""

        return self.decide(
            label,
            confidence=confidence,
            margin=margin,
            prediction_set_size=prediction_set_size,
            conformal_p_value=conformal_p_value,
        ).as_tuple()