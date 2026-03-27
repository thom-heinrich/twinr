# CHANGELOG: 2026-03-27
# BUG-1: normalize_user_intent_label failed for UserIntentLabel enum members because
#        str(EnumMember) does not equal the member value for classic str/Enum mixes.
#        The normalizer now accepts enum members, bytes, umlauts, and common aliases.
# BUG-2: TwoStageSemanticRouteDecision accepted internally inconsistent states where
#        stage 2 selected a backend route that violated the stage-1 user-intent policy.
#        The contract now validates user-intent policy subsets and selected-route membership.
# BUG-3: UserIntentDecision trusted mutable/invalid score payloads, allowing silent score
#        mutation after construction and accepting duplicate/non-finite values.
#        Scores are now canonicalized, validated, and deep-frozen.
# SEC-1: Process-wide routing policy tables were mutable dictionaries. Any in-process code
#        path could tamper with them and change privacy-sensitive routing behavior at runtime.
#        They are now exposed as immutable MappingProxyType views.
# SEC-2: Stage-2 routing could bypass the stage-1 privacy boundary (for example routing a
#        "persoenlich" request to "web") because the contract did not enforce compatibility.
#        Construction now hard-fails on policy violations before dispatch.
# IMP-1: Added confidence/ambiguity helpers so higher-level routers can implement the
#        confidence-threshold and abstention patterns that are standard in 2026 routers.
# IMP-2: Kept the module Pi-friendly and zero-heavy-dependency while upgrading validation,
#        canonicalization, and policy enforcement for edge deployments.

"""Define user-centered routing labels and two-stage routing contracts.

The local backend router uses technical execution labels such as ``memory`` or
``tool``. This module adds the user-facing semantic layer that better matches
how a person naturally frames a request before Twinr decides which backend path
should answer it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from functools import lru_cache
from math import exp, fsum, isfinite
from re import compile as re_compile
from types import MappingProxyType
from typing import Final, Mapping
from unicodedata import normalize as unicode_normalize

from .contracts import SemanticRouteDecision, normalize_route_label

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover
    class StrEnum(str, Enum):
        """Compatibility shim for Python < 3.11."""
        pass


_ASCII_FOLD_TRANSLATION: Final = str.maketrans({
    "ä": "ae",
    "ö": "oe",
    "ü": "ue",
    "ß": "ss",
})
_SEPARATOR_RE: Final = re_compile(r"[\s\-/]+")
_NON_TOKEN_RE: Final = re_compile(r"[^a-z0-9_]+")
_SCORE_EPSILON: Final[float] = 1e-9


@unique
class UserIntentLabel(StrEnum):
    """Enumerate the user-centered routing classes."""

    WISSEN = "wissen"
    NACHSCHAUEN = "nachschauen"
    PERSOENLICH = "persoenlich"
    MACHEN_ODER_PRUEFEN = "machen_oder_pruefen"


USER_INTENT_LABEL_VALUES: tuple[str, ...] = tuple(label.value for label in UserIntentLabel)
_USER_INTENT_LABEL_SET: Final[frozenset[str]] = frozenset(USER_INTENT_LABEL_VALUES)

_USER_INTENT_LABEL_ALIASES: Final[Mapping[str, str]] = MappingProxyType({
    UserIntentLabel.WISSEN.value: UserIntentLabel.WISSEN.value,
    "knowledge": UserIntentLabel.WISSEN.value,
    "factual": UserIntentLabel.WISSEN.value,
    "facts": UserIntentLabel.WISSEN.value,
    UserIntentLabel.NACHSCHAUEN.value: UserIntentLabel.NACHSCHAUEN.value,
    "lookup": UserIntentLabel.NACHSCHAUEN.value,
    "look_up": UserIntentLabel.NACHSCHAUEN.value,
    "recherche": UserIntentLabel.NACHSCHAUEN.value,
    UserIntentLabel.PERSOENLICH.value: UserIntentLabel.PERSOENLICH.value,
    "personal": UserIntentLabel.PERSOENLICH.value,
    "private": UserIntentLabel.PERSOENLICH.value,
    UserIntentLabel.MACHEN_ODER_PRUEFEN.value: UserIntentLabel.MACHEN_ODER_PRUEFEN.value,
    "machenoderpruefen": UserIntentLabel.MACHEN_ODER_PRUEFEN.value,
    "do_or_check": UserIntentLabel.MACHEN_ODER_PRUEFEN.value,
    "act_or_verify": UserIntentLabel.MACHEN_ODER_PRUEFEN.value,
})

_DEFAULT_USER_INTENT_BY_ROUTE_LABEL_DATA: Final[dict[str, str]] = {
    "parametric": UserIntentLabel.WISSEN.value,
    "web": UserIntentLabel.NACHSCHAUEN.value,
    "memory": UserIntentLabel.PERSOENLICH.value,
    "tool": UserIntentLabel.MACHEN_ODER_PRUEFEN.value,
}

_ALLOWED_ROUTE_LABELS_BY_USER_INTENT_DATA: Final[dict[str, tuple[str, ...]]] = {
    UserIntentLabel.WISSEN.value: ("parametric",),
    UserIntentLabel.NACHSCHAUEN.value: ("web",),
    UserIntentLabel.PERSOENLICH.value: ("memory", "tool"),
    UserIntentLabel.MACHEN_ODER_PRUEFEN.value: ("tool",),
}

# BREAKING: exported policy tables are now immutable mapping views to prevent runtime tampering.
DEFAULT_USER_INTENT_BY_ROUTE_LABEL: Mapping[str, str] = MappingProxyType(
    dict(_DEFAULT_USER_INTENT_BY_ROUTE_LABEL_DATA)
)
# BREAKING: exported policy tables are now immutable mapping views to prevent runtime tampering.
ALLOWED_ROUTE_LABELS_BY_USER_INTENT: Mapping[str, tuple[str, ...]] = MappingProxyType(
    dict(_ALLOWED_ROUTE_LABELS_BY_USER_INTENT_DATA)
)


def _coerce_text(value: object) -> str:
    """Convert arbitrary input to one text candidate before canonicalization."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, UserIntentLabel):
        return value.value
    return str(value or "")


@lru_cache(maxsize=128)
def _normalize_user_intent_token(value: str) -> str:
    """Normalize user-intent text into one ASCII-safe token."""

    normalized = unicode_normalize("NFKC", value).strip().lower()
    normalized = normalized.translate(_ASCII_FOLD_TRANSLATION)
    normalized = _SEPARATOR_RE.sub("_", normalized)
    normalized = _NON_TOKEN_RE.sub("", normalized)
    normalized = normalized.strip("_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized


def normalize_user_intent_label(value: object) -> str:
    """Return one validated user-centered label string."""

    normalized = _normalize_user_intent_token(_coerce_text(value))
    canonical = _USER_INTENT_LABEL_ALIASES.get(normalized)
    if canonical is None:
        raise ValueError(f"Unsupported user intent label: {value!r}")
    return canonical


def default_user_intent_for_route_label(route_label: str) -> str:
    """Map one backend route label to its default user-centered class."""

    normalized_route_label = normalize_route_label(route_label)
    user_intent = DEFAULT_USER_INTENT_BY_ROUTE_LABEL.get(normalized_route_label)
    if user_intent is None:
        raise ValueError(
            f"No default user intent configured for route label: {normalized_route_label!r}"
        )
    return user_intent


def allowed_route_labels_for_user_intent(user_intent_label: str) -> tuple[str, ...]:
    """Return the backend labels that remain valid for one user intent."""

    normalized_user_label = normalize_user_intent_label(user_intent_label)
    allowed = ALLOWED_ROUTE_LABELS_BY_USER_INTENT.get(normalized_user_label)
    if allowed is None:
        raise ValueError(
            f"No allowed route labels configured for user intent: {normalized_user_label!r}"
        )
    return allowed


def is_route_label_allowed_for_user_intent(user_intent_label: str, route_label: str) -> bool:
    """Return whether one backend route is policy-compatible with one user intent."""

    normalized_route_label = normalize_route_label(route_label)
    return normalized_route_label in allowed_route_labels_for_user_intent(user_intent_label)


def _validated_non_negative_float(name: str, value: object) -> float:
    """Coerce one finite non-negative float field."""

    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    if numeric < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return numeric


def _canonicalize_score_mapping(scores: Mapping[object, object]) -> Mapping[str, float]:
    """Normalize, validate, and deep-freeze one score mapping."""

    try:
        raw_scores = dict(scores)
    except Exception as exc:  # pragma: no cover - defensive against exotic mappings
        raise TypeError("scores must be a mapping of labels to numeric values.") from exc

    normalized_scores: dict[str, float] = {}
    for raw_key, raw_value in raw_scores.items():
        normalized_key = normalize_user_intent_label(raw_key)
        if normalized_key in normalized_scores:
            raise ValueError(
                f"UserIntentDecision.scores contains duplicate entries for {normalized_key!r}."
            )
        numeric_value = float(raw_value)
        if not isfinite(numeric_value):
            raise ValueError(
                f"UserIntentDecision.scores[{normalized_key!r}] must be finite."
            )
        normalized_scores[normalized_key] = numeric_value

    missing = _USER_INTENT_LABEL_SET.difference(normalized_scores)
    if missing:
        raise ValueError(f"UserIntentDecision.scores missing labels: {sorted(missing)}")

    # BREAKING: scores are deep-frozen to prevent post-construction mutation.
    return MappingProxyType(normalized_scores)


def _softmax_probabilities(scores: Mapping[str, float]) -> dict[str, float]:
    """Convert arbitrary finite scores into one probability-like distribution."""

    max_score = max(float(score) for score in scores.values())
    exponents = {label: exp(float(score) - max_score) for label, score in scores.items()}
    total = fsum(exponents.values())
    if total <= 0.0:
        raise ValueError("Score normalization failed because exponent total is not positive.")
    return {label: value / total for label, value in exponents.items()}


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
        normalized_scores = _canonicalize_score_mapping(self.scores)
        normalized_label = normalize_user_intent_label(self.label)
        normalized_confidence = _validated_non_negative_float("confidence", self.confidence)
        normalized_margin = _validated_non_negative_float("margin", self.margin)
        normalized_latency_ms = _validated_non_negative_float("latency_ms", self.latency_ms)

        label_score = float(normalized_scores[normalized_label])
        top_score = max(float(score) for score in normalized_scores.values())
        if label_score + _SCORE_EPSILON < top_score:
            raise ValueError(
                "UserIntentDecision.label must correspond to a highest-scoring label."
            )

        object.__setattr__(self, "label", normalized_label)
        object.__setattr__(self, "confidence", normalized_confidence)
        object.__setattr__(self, "margin", normalized_margin)
        object.__setattr__(self, "scores", normalized_scores)
        object.__setattr__(self, "model_id", str(self.model_id or "").strip())
        object.__setattr__(self, "latency_ms", normalized_latency_ms)

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

    def probabilities(self) -> Mapping[str, float]:
        """Return one normalized probability-like distribution over labels."""

        values = [float(value) for value in self.scores.values()]
        if all(0.0 <= value <= 1.0 for value in values):
            total = fsum(values)
            if total > 0.0 and abs(total - 1.0) <= 1e-6 * len(values):
                return MappingProxyType(
                    {label: float(score) / total for label, score in self.scores.items()}
                )
        return MappingProxyType(_softmax_probabilities(self.scores))

    def calibrated_confidence(self) -> float:
        """Return one threshold-friendly confidence estimate for the selected label."""

        return float(self.probabilities()[self.label])

    def observed_margin(self) -> float:
        """Return the actual top-1 minus top-2 score gap from the stored scores."""

        ranked_scores = sorted((float(score) for score in self.scores.values()), reverse=True)
        if len(ranked_scores) < 2:
            return 0.0
        return ranked_scores[0] - ranked_scores[1]

    def is_confident(
        self,
        min_confidence: float = 0.0,
        min_margin: float = 0.0,
        *,
        use_calibrated_confidence: bool = False,
    ) -> bool:
        """Return whether the decision clears one confidence and margin gate."""

        selected_confidence = (
            self.calibrated_confidence() if use_calibrated_confidence else float(self.confidence)
        )
        return (
            selected_confidence >= float(min_confidence)
            and self.observed_margin() >= float(min_margin)
        )

    def should_abstain(
        self,
        min_confidence: float = 0.0,
        min_margin: float = 0.0,
        *,
        use_calibrated_confidence: bool = False,
    ) -> bool:
        """Return whether the decision should be treated as ambiguous."""

        return not self.is_confident(
            min_confidence=min_confidence,
            min_margin=min_margin,
            use_calibrated_confidence=use_calibrated_confidence,
        )


@dataclass(frozen=True, slots=True)
class TwoStageSemanticRouteDecision:
    """Store the stage-1 user intent and the stage-2 backend route together."""

    user_intent: UserIntentDecision
    route_decision: SemanticRouteDecision
    allowed_route_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        normalized_allowed = tuple(
            dict.fromkeys(normalize_route_label(label) for label in self.allowed_route_labels)
        )
        if not normalized_allowed:
            raise ValueError("TwoStageSemanticRouteDecision.allowed_route_labels must not be empty.")

        policy_allowed = allowed_route_labels_for_user_intent(self.user_intent.label)
        forbidden = tuple(label for label in normalized_allowed if label not in policy_allowed)
        if forbidden:
            # BREAKING: stage-2 route candidates must now be a subset of the stage-1 policy.
            raise ValueError(
                "TwoStageSemanticRouteDecision.allowed_route_labels contains labels "
                f"that violate the user-intent policy: {forbidden!r}"
            )

        selected_route_label = normalize_route_label(self.route_decision.label)
        if selected_route_label not in normalized_allowed:
            # BREAKING: inconsistent stage-2 winners now fail at construction time.
            raise ValueError(
                "TwoStageSemanticRouteDecision.route_decision.label is not contained in "
                "allowed_route_labels."
            )

        object.__setattr__(self, "allowed_route_labels", normalized_allowed)

    @property
    def selected_route_label(self) -> str:
        """Return the normalized stage-2 route label."""

        return normalize_route_label(self.route_decision.label)

    def is_route_allowed(self, route_label: str | None = None) -> bool:
        """Return whether one route label is allowed under the stored stage-1 policy."""

        selected_label = self.selected_route_label if route_label is None else route_label
        return normalize_route_label(selected_label) in self.allowed_route_labels

    def should_abstain(
        self,
        *,
        min_user_confidence: float = 0.0,
        min_user_margin: float = 0.0,
        min_route_confidence: float = 0.0,
        min_route_margin: float = 0.0,
        use_calibrated_user_confidence: bool = False,
    ) -> bool:
        """Return whether either routing stage is too uncertain to trust blindly."""

        route_confidence = float(getattr(self.route_decision, "confidence"))
        route_margin = float(getattr(self.route_decision, "margin"))
        if not isfinite(route_confidence) or not isfinite(route_margin):
            return True
        if route_confidence < float(min_route_confidence):
            return True
        if route_margin < float(min_route_margin):
            return True
        return self.user_intent.should_abstain(
            min_confidence=min_user_confidence,
            min_margin=min_user_margin,
            use_calibrated_confidence=use_calibrated_user_confidence,
        )


__all__ = (
    "ALLOWED_ROUTE_LABELS_BY_USER_INTENT",
    "DEFAULT_USER_INTENT_BY_ROUTE_LABEL",
    "TwoStageSemanticRouteDecision",
    "USER_INTENT_LABEL_VALUES",
    "UserIntentDecision",
    "UserIntentLabel",
    "allowed_route_labels_for_user_intent",
    "default_user_intent_for_route_label",
    "is_route_label_allowed_for_user_intent",
    "normalize_user_intent_label",
)