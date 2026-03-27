# CHANGELOG: 2026-03-27
# BUG-1: Fixed self-incompatibility where normalize_route_label(RouteLabel.X)
#         failed because Enum stringification produced "RouteLabel.X".
# BUG-2: Fixed silent drift between label/confidence/margin and scores by
#         canonicalizing all top-1 metadata from the score distribution.
# BUG-3: Fixed post-validation mutability of scores while preserving
#         dataclasses.asdict()/telemetry compatibility.
# BUG-4: Fixed acceptance of NaN/negative latency and invalid score payloads.
# SEC-1: No standalone practical exploit found in this isolated module; added
#         decision-integrity hardening so inconsistent routing metadata cannot
#         be injected or mutated after validation.
# IMP-1: Added from_scores(), from_logits(), entropy/normalized_entropy,
#         runner-up/top-k helpers, and richer policy handoff for
#         uncertainty-aware routing.
# IMP-2: Kept constructor compatibility while upgrading the object to a
#         calibration-friendly, immutable, JSON-safe 2026 routing primitive.

"""Define normalized route labels and scored local routing decisions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from inspect import Parameter, signature
from math import exp, fsum, isfinite, log
from typing import TYPE_CHECKING, Any

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return str(self.value)


if TYPE_CHECKING:
    from .policy import SemanticRouterPolicy


class RouteLabel(StrEnum):
    """Enumerate the supported local semantic routing classes."""

    PARAMETRIC = "parametric"
    WEB = "web"
    MEMORY = "memory"
    TOOL = "tool"


ROUTE_LABEL_VALUES: tuple[str, ...] = tuple(label.value for label in RouteLabel)
_ROUTE_LABEL_SET = frozenset(ROUTE_LABEL_VALUES)
_ROUTE_LABEL_INDEX = {label: index for index, label in enumerate(ROUTE_LABEL_VALUES)}
_NORMALIZATION_TOLERANCE = 1e-9


class FrozenScoreMap(Mapping[str, float]):
    """Immutable mapping optimized for the fixed semantic route label set.

    The live object is immutable, but `dataclasses.asdict()` still works because
    `__deepcopy__` returns a plain `dict`.
    """

    __slots__ = ("_values",)

    def __init__(self, scores: Mapping[str, float]):
        self._values = tuple(float(scores[label]) for label in ROUTE_LABEL_VALUES)

    def __getitem__(self, label: object) -> float:
        return self._values[_ROUTE_LABEL_INDEX[normalize_route_label(label)]]

    def __iter__(self):
        return iter(ROUTE_LABEL_VALUES)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return repr(self.as_dict())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        try:
            other_dict = {normalize_route_label(key): float(value) for key, value in other.items()}
        except (TypeError, ValueError):
            return False
        return self.as_dict() == other_dict

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, float]:
        return self.as_dict()

    def as_dict(self) -> dict[str, float]:
        return {label: self._values[index] for index, label in enumerate(ROUTE_LABEL_VALUES)}

    def copy(self) -> dict[str, float]:
        return self.as_dict()


def normalize_route_label(value: object) -> str:
    """Return one validated route label string."""

    if isinstance(value, RouteLabel):
        normalized = value.value
    else:
        normalized = str("" if value is None else value).strip().lower()
    if normalized not in _ROUTE_LABEL_SET:
        raise ValueError(f"Unsupported semantic route label: {value!r}")
    return normalized


def _optional_normalized_label(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, RouteLabel):
        return value.value
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized not in _ROUTE_LABEL_SET:
        raise ValueError(f"Unsupported semantic route label: {value!r}")
    return normalized


def _coerce_score_items(
    scores: Mapping[object, object] | Iterable[tuple[object, object]],
) -> list[tuple[object, object]]:
    if isinstance(scores, Mapping):
        return list(scores.items())
    try:
        raw_items = list(scores)
    except TypeError as exc:
        raise TypeError(
            "SemanticRouteDecision.scores must be a mapping or iterable of "
            "(label, score) pairs"
        ) from exc

    coerced: list[tuple[object, object]] = []
    for item in raw_items:
        try:
            raw_label, raw_score = item
        except Exception as exc:
            raise TypeError(
                "SemanticRouteDecision.scores iterable entries must unpack into "
                "(label, score) pairs"
            ) from exc
        coerced.append((raw_label, raw_score))
    return coerced


def _normalize_probability_scores(
    scores: Mapping[object, object] | Iterable[tuple[object, object]],
) -> FrozenScoreMap:
    raw_items = _coerce_score_items(scores)
    normalized_scores: dict[str, float] = {}
    seen_raw_keys: dict[str, object] = {}

    for raw_label, raw_score in raw_items:
        label = normalize_route_label(raw_label)
        if label in normalized_scores:
            raise ValueError(
                "SemanticRouteDecision.scores contains duplicate labels after normalization: "
                f"{seen_raw_keys[label]!r} and {raw_label!r}"
            )
        score = float(raw_score)
        if not isfinite(score):
            raise ValueError(f"SemanticRouteDecision.scores[{label!r}] must be finite, got {raw_score!r}")
        # BREAKING: negative / non-probability inputs now fail fast instead of
        # silently producing inconsistent routing metadata.
        if score < 0.0:
            raise ValueError(
                f"SemanticRouteDecision.scores[{label!r}] must be >= 0.0 for "
                f"probability-like inputs, got {raw_score!r}"
            )
        normalized_scores[label] = score
        seen_raw_keys[label] = raw_label

    missing = _ROUTE_LABEL_SET.difference(normalized_scores)
    if missing:
        raise ValueError(f"SemanticRouteDecision.scores missing labels: {sorted(missing)}")

    total = fsum(normalized_scores.values())
    if not isfinite(total) or total <= 0.0:
        raise ValueError("SemanticRouteDecision.scores must sum to a finite value > 0.0")

    if abs(total - 1.0) > _NORMALIZATION_TOLERANCE:
        normalized_scores = {label: score / total for label, score in normalized_scores.items()}

    ordered_scores = {label: normalized_scores[label] for label in ROUTE_LABEL_VALUES}
    return FrozenScoreMap(ordered_scores)


def _stable_softmax(
    logits: Mapping[object, object] | Iterable[tuple[object, object]],
) -> FrozenScoreMap:
    raw_items = _coerce_score_items(logits)
    normalized_logits: dict[str, float] = {}
    seen_raw_keys: dict[str, object] = {}

    for raw_label, raw_score in raw_items:
        label = normalize_route_label(raw_label)
        if label in normalized_logits:
            raise ValueError(
                "SemanticRouteDecision logits contain duplicate labels after normalization: "
                f"{seen_raw_keys[label]!r} and {raw_label!r}"
            )
        score = float(raw_score)
        if not isfinite(score):
            raise ValueError(f"SemanticRouteDecision logits[{label!r}] must be finite, got {raw_score!r}")
        normalized_logits[label] = score
        seen_raw_keys[label] = raw_label

    missing = _ROUTE_LABEL_SET.difference(normalized_logits)
    if missing:
        raise ValueError(f"SemanticRouteDecision logits missing labels: {sorted(missing)}")

    max_logit = max(normalized_logits.values())
    exp_scores = {label: exp(value - max_logit) for label, value in normalized_logits.items()}
    total = fsum(exp_scores.values())
    ordered_scores = {label: exp_scores[label] / total for label in ROUTE_LABEL_VALUES}
    return FrozenScoreMap(ordered_scores)


def _ranked_labels(scores: Mapping[str, float]) -> tuple[str, ...]:
    return tuple(
        label
        for label, _score in sorted(
            scores.items(),
            key=lambda item: (-float(item[1]), item[0]),
        )
    )


def _policy_evaluate(
    policy: "SemanticRouterPolicy",
    decision: "SemanticRouteDecision",
) -> tuple[bool, str | None]:
    evaluate = policy.evaluate
    available_kwargs: dict[str, object] = {
        "label": decision.label,
        "confidence": decision.confidence,
        "margin": decision.margin,
        "scores": decision.scores,
        "latency_ms": decision.latency_ms,
        "model_id": decision.model_id,
        "entropy": decision.entropy,
        "normalized_entropy": decision.normalized_entropy,
        "runner_up_label": decision.runner_up_label,
        "runner_up_confidence": decision.runner_up_confidence,
    }

    try:
        parameters = signature(evaluate).parameters
    except (TypeError, ValueError):
        result = evaluate(
            decision.label,
            confidence=decision.confidence,
            margin=decision.margin,
        )
    else:
        accepts_var_kwargs = any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values())
        kwargs = {
            name: value
            for name, value in available_kwargs.items()
            if name != "label" and (accepts_var_kwargs or name in parameters)
        }

        label_parameter = parameters.get("label")
        if label_parameter is not None and label_parameter.kind is Parameter.POSITIONAL_ONLY:
            result = evaluate(decision.label, **kwargs)
        elif label_parameter is not None:
            result = evaluate(label=decision.label, **kwargs)
        elif accepts_var_kwargs:
            result = evaluate(label=decision.label, **kwargs)
        else:
            result = evaluate(decision.label, **kwargs)

    if isinstance(result, tuple):
        if len(result) != 2:
            raise ValueError(
                "SemanticRouterPolicy.evaluate must return "
                "(authoritative, fallback_reason)"
            )
        authoritative, fallback_reason = result
    else:
        authoritative, fallback_reason = bool(result), None

    normalized_reason = str(fallback_reason).strip() or None if fallback_reason is not None else None
    return bool(authoritative), normalized_reason


@dataclass(frozen=True, slots=True)
class SemanticRouteDecision:
    """Store one scored local semantic-routing result.

    Attributes:
        label: Highest-scoring route label. Canonicalized from `scores`.
        confidence: Probability/confidence for the selected label. Derived from `scores`.
        margin: Confidence delta between the top-1 and top-2 labels. Derived from `scores`.
        scores: Immutable normalized score distribution keyed by label.
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
        normalized_scores = _normalize_probability_scores(self.scores)
        ranked = _ranked_labels(normalized_scores)
        canonical_label = ranked[0]
        canonical_confidence = float(normalized_scores[canonical_label])
        runner_up_confidence = float(normalized_scores[ranked[1]])
        canonical_margin = canonical_confidence - runner_up_confidence
        _optional_normalized_label(self.label)

        latency_ms = float(self.latency_ms)
        # BREAKING: invalid latency now fails fast instead of reaching telemetry
        # and policy layers as NaN / negative garbage.
        if not isfinite(latency_ms) or latency_ms < 0.0:
            raise ValueError(
                "SemanticRouteDecision.latency_ms must be finite and >= 0.0, "
                f"got {self.latency_ms!r}"
            )

        model_id = str(self.model_id or "").strip()
        authoritative = bool(self.authoritative)
        fallback_reason = str(self.fallback_reason or "").strip() or None
        if authoritative:
            fallback_reason = None

        # BREAKING: `label`, `confidence`, and `margin` are now canonicalized from
        # `scores`. Callers can no longer construct internally inconsistent
        # decisions that bypass policy with mismatched top-1 metadata.
        object.__setattr__(self, "label", canonical_label)
        object.__setattr__(self, "confidence", canonical_confidence)
        object.__setattr__(self, "margin", canonical_margin)
        object.__setattr__(self, "scores", normalized_scores)
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "latency_ms", latency_ms)
        object.__setattr__(self, "authoritative", authoritative)
        object.__setattr__(self, "fallback_reason", fallback_reason)

    @classmethod
    def from_scores(
        cls,
        scores: Mapping[object, object] | Iterable[tuple[object, object]],
        *,
        model_id: str,
        latency_ms: float,
        authoritative: bool = False,
        fallback_reason: str | None = None,
    ) -> "SemanticRouteDecision":
        """Create one canonical decision from probability-like scores."""

        return cls(
            label="",
            confidence=0.0,
            margin=0.0,
            scores=scores,
            model_id=model_id,
            latency_ms=latency_ms,
            authoritative=authoritative,
            fallback_reason=fallback_reason,
        )

    @classmethod
    def from_logits(
        cls,
        logits: Mapping[object, object] | Iterable[tuple[object, object]],
        *,
        model_id: str,
        latency_ms: float,
        authoritative: bool = False,
        fallback_reason: str | None = None,
    ) -> "SemanticRouteDecision":
        """Create one canonical decision from raw logits using stable softmax."""

        return cls(
            label="",
            confidence=0.0,
            margin=0.0,
            scores=_stable_softmax(logits),
            model_id=model_id,
            latency_ms=latency_ms,
            authoritative=authoritative,
            fallback_reason=fallback_reason,
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> "SemanticRouteDecision":
        return type(self)(
            label=self.label,
            confidence=self.confidence,
            margin=self.margin,
            scores=dict(self.scores),
            model_id=self.model_id,
            latency_ms=self.latency_ms,
            authoritative=self.authoritative,
            fallback_reason=self.fallback_reason,
        )

    def score_for(self, label: object) -> float:
        """Return the score for one normalized route label."""

        return float(self.scores[normalize_route_label(label)])

    def ranked_labels(self) -> tuple[str, ...]:
        """Return labels ordered by descending score."""

        return _ranked_labels(self.scores)

    def top_k(self, k: int = 2) -> tuple[tuple[str, float], ...]:
        """Return the top-k `(label, score)` pairs."""

        if k <= 0:
            return ()
        return tuple((label, self.score_for(label)) for label in self.ranked_labels()[:k])

    @property
    def runner_up_label(self) -> str:
        """Return the second-best route label."""

        return self.ranked_labels()[1]

    @property
    def runner_up_confidence(self) -> float:
        """Return the score of the runner-up route label."""

        return self.score_for(self.runner_up_label)

    @property
    def entropy(self) -> float:
        """Return Shannon entropy of the normalized score distribution."""

        return -fsum(score * log(score) for score in self.scores.values() if score > 0.0)

    @property
    def normalized_entropy(self) -> float:
        """Return entropy normalized into `[0.0, 1.0]`."""

        max_entropy = log(len(ROUTE_LABEL_VALUES))
        if max_entropy <= 0.0:
            return 0.0
        value = self.entropy / max_entropy
        return max(0.0, min(1.0, value))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe dictionary for telemetry and persistence."""

        return {
            "label": self.label,
            "confidence": self.confidence,
            "margin": self.margin,
            "scores": dict(self.scores),
            "model_id": self.model_id,
            "latency_ms": self.latency_ms,
            "authoritative": self.authoritative,
            "fallback_reason": self.fallback_reason,
            "entropy": self.entropy,
            "normalized_entropy": self.normalized_entropy,
            "runner_up_label": self.runner_up_label,
            "runner_up_confidence": self.runner_up_confidence,
        }

    def with_policy(self, policy: "SemanticRouterPolicy") -> "SemanticRouteDecision":
        """Return a copy with `authoritative` derived from one policy.

        Backward compatible with legacy policies that only accept
        `(label, confidence, margin)`, while also passing richer uncertainty
        metadata to newer policies when their signature supports it.
        """

        authoritative, fallback_reason = _policy_evaluate(policy, self)
        return replace(
            self,
            authoritative=authoritative,
            fallback_reason=fallback_reason,
        )