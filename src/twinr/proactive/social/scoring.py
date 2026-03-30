"""Score proactive social-trigger evidence with bounded numeric helpers.

This module normalizes evidence weights and timing-derived hold signals before
the trigger engine turns them into one comparable score card.
"""

# CHANGELOG: 2026-03-29
# BUG-1: Invalid thresholds (NaN / inf / out-of-range) previously clamped to 0.0,
#        which could fail open and incorrectly pass triggers.
# BUG-2: recent_progress() treated future timestamps / mixed clock domains as
#        maximally recent, and zero-window / zero-target cases could also fail open.
# BUG-3: key/detail were only type hinted as str; runtime misuse could break JSON output.
# SEC-1: _normalize_evidence() previously materialized unbounded iterables, enabling
#        memory / CPU denial-of-service on Pi-class hardware.
# IMP-1: Added monotonic-ns helpers and stricter elapsed-time semantics for long-running
#        edge deployments.
# IMP-2: Added precise fsum aggregation, evidence-mass gating, structured flags, and an
#        optional piecewise-linear calibrator for calibrated trigger scoring.

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from math import fsum, isfinite
from typing import Callable, Final, Iterable


MAX_EVIDENCE_ITEMS: Final[int] = 256


def _finite_float(value: object) -> float | None:
    """Return one finite float or ``None`` for invalid input."""

    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not isfinite(numeric):
        return None
    return numeric


def _finite_int(value: object) -> int | None:
    """Return one exact integer or ``None`` for invalid input."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    numeric = _finite_float(value)
    if numeric is None or not numeric.is_integer():
        return None
    return int(numeric)


def _non_negative_float(value: object) -> float:
    """Return one finite non-negative float or ``0.0``."""

    numeric = _finite_float(value)
    if numeric is None or numeric <= 0.0:
        return 0.0
    return numeric


def _coerce_text(value: object, *, fallback: str = "") -> str:
    """Return one best-effort string for logs / JSON payloads."""

    if isinstance(value, str):
        return value
    if value is None:
        return fallback
    try:
        return str(value)
    except Exception:
        return fallback


def _normalize_threshold(value: object) -> tuple[float, tuple[str, ...]]:
    """Return one safe threshold and optional normalization flags."""

    numeric = _finite_float(value)
    if numeric is None or numeric < 0.0 or numeric > 1.0:
        # BREAKING: Invalid thresholds now fail closed instead of clamping into the
        # unit interval where NaN/-1 could effectively become 0.0 and auto-pass.
        return 1.0, ("invalid_threshold",)
    return numeric, ()


def _normalize_flags(flags: Iterable[object] | None) -> tuple[str, ...]:
    """Return one stable, deduplicated tuple of flag strings."""

    if flags is None:
        return ()

    normalized: list[str] = []
    seen: set[str] = set()
    for flag in flags:
        text = _coerce_text(flag, fallback="")
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return tuple(normalized)


def _normalize_evidence(
    evidence: Iterable["TriggerScoreEvidence"] | None,
    *,
    max_items: int = MAX_EVIDENCE_ITEMS,
) -> tuple[tuple["TriggerScoreEvidence", ...], tuple[str, ...]]:
    """Materialize, cap, and type-filter one evidence iterable."""

    if evidence is None:
        return (), ()

    limit = _finite_int(max_items)
    if limit is None or limit <= 0:
        limit = MAX_EVIDENCE_ITEMS

    try:
        iterator = iter(evidence)
    except TypeError:
        return (), ("invalid_evidence_iterable",)

    items: list[TriggerScoreEvidence] = []
    dropped_invalid = False
    truncated = False

    for index, item in enumerate(iterator):
        if index >= limit:
            truncated = True
            break
        if isinstance(item, TriggerScoreEvidence):
            items.append(item)
        else:
            dropped_invalid = True

    flags: list[str] = []
    if dropped_invalid:
        flags.append("invalid_evidence_dropped")
    if truncated:
        # BREAKING: Evidence is now capped to avoid unbounded iterator materialization
        # on memory-constrained devices. Overflow is signaled via flags.
        flags.append("evidence_truncated")
    return tuple(items), tuple(flags)


def clamp_unit(value: float) -> float:
    """Clamp one numeric value into the closed unit interval."""

    numeric = _finite_float(value)
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

    now_value = _finite_float(now)
    since_value = _finite_float(since)
    target_value = _finite_float(target_s)
    if now_value is None or since_value is None or target_value is None:
        return 0.0

    elapsed = now_value - since_value
    if elapsed < 0.0 or target_value < 0.0:
        return 0.0
    if target_value == 0.0:
        # BREAKING: Zero target now only completes when the hold start is not in the
        # future. Mixed clock domains / future timestamps no longer auto-complete.
        return 1.0
    return clamp_unit(elapsed / target_value)


def recent_progress(now: float, anchor_at: float | None, window_s: float) -> float:
    """Return how recent one anchor timestamp is within one window."""

    if anchor_at is None:
        return 0.0

    now_value = _finite_float(now)
    anchor_value = _finite_float(anchor_at)
    window_value = _finite_float(window_s)
    if now_value is None or anchor_value is None or window_value is None:
        return 0.0

    elapsed = now_value - anchor_value
    if elapsed < 0.0 or window_value < 0.0:
        # BREAKING: Future timestamps now fail closed instead of being treated as
        # maximally recent, which previously produced false positives under clock skew
        # or mixed time.monotonic()/time.time() inputs.
        return 0.0
    if window_value == 0.0:
        # BREAKING: A zero recency window is now only "fresh" at the exact instant.
        # Historical anchors no longer score as fully recent.
        return 1.0 if elapsed == 0.0 else 0.0
    return clamp_unit(1.0 - (elapsed / window_value))


def hold_progress_ns(now_ns: int, since_ns: int | None, target_ns: int) -> float:
    """Return hold progress using monotonic nanosecond timestamps."""

    if since_ns is None:
        return 0.0

    now_value = _finite_int(now_ns)
    since_value = _finite_int(since_ns)
    target_value = _finite_int(target_ns)
    if now_value is None or since_value is None or target_value is None:
        return 0.0

    elapsed = now_value - since_value
    if elapsed < 0 or target_value < 0:
        return 0.0
    if target_value == 0:
        return 1.0
    return clamp_unit(elapsed / target_value)


def recent_progress_ns(
    now_ns: int,
    anchor_at_ns: int | None,
    window_ns: int,
) -> float:
    """Return recency progress using monotonic nanosecond timestamps."""

    if anchor_at_ns is None:
        return 0.0

    now_value = _finite_int(now_ns)
    anchor_value = _finite_int(anchor_at_ns)
    window_value = _finite_int(window_ns)
    if now_value is None or anchor_value is None or window_value is None:
        return 0.0

    elapsed = now_value - anchor_value
    if elapsed < 0 or window_value < 0:
        return 0.0
    if window_value == 0:
        return 1.0 if elapsed == 0 else 0.0
    return clamp_unit(1.0 - (elapsed / window_value))


@dataclass(frozen=True, slots=True)
class TriggerScoreEvidence:
    """Describe one weighted evidence item used in trigger scoring."""

    key: str
    value: float
    weight: float
    detail: str

    def __post_init__(self) -> None:
        """Normalize stored fields after construction."""

        object.__setattr__(self, "key", _coerce_text(self.key))
        object.__setattr__(self, "detail", _coerce_text(self.detail))
        object.__setattr__(self, "value", clamp_unit(self.value))
        object.__setattr__(self, "weight", _non_negative_float(self.weight))

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
class PiecewiseLinearCalibrator:
    """Map one raw unit score onto one calibrated unit score."""

    points: tuple[tuple[float, float], ...]
    _xs: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate one monotonic piecewise-linear calibration curve."""

        try:
            materialized = tuple((float(x), float(y)) for x, y in self.points)
        except Exception as exc:
            raise ValueError("points must be an iterable of (x, y) pairs") from exc

        if len(materialized) < 2:
            raise ValueError("at least two calibration points are required")

        normalized: list[tuple[float, float]] = []
        previous_x = -1.0
        previous_y = -1.0
        for x_raw, y_raw in materialized:
            x = clamp_unit(x_raw)
            y = clamp_unit(y_raw)
            if x < previous_x:
                raise ValueError("calibration points must be sorted by x")
            if x == previous_x:
                raise ValueError("duplicate x values are not allowed")
            if y < previous_y:
                raise ValueError("calibration points must be monotonic in y")
            normalized.append((x, y))
            previous_x = x
            previous_y = y

        points = tuple(normalized)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "_xs", tuple(x for x, _ in points))

    def __call__(self, score: float) -> float:
        """Return one calibrated score via piecewise-linear interpolation."""

        x = clamp_unit(score)
        index = bisect_right(self._xs, x)

        if index <= 0:
            return self.points[0][1]
        if index >= len(self.points):
            return self.points[-1][1]

        x0, y0 = self.points[index - 1]
        x1, y1 = self.points[index]
        if x1 == x0:
            return y1
        ratio = (x - x0) / (x1 - x0)
        return clamp_unit(y0 + ((y1 - y0) * ratio))

    def to_dict(self) -> dict[str, object]:
        """Serialize the calibration curve for logs or config introspection."""

        return {
            "points": [[round(x, 6), round(y, 6)] for x, y in self.points],
        }


@dataclass(frozen=True, slots=True)
class WeightedTriggerScore:
    """Represent one scored trigger candidate and its normalized evidence."""

    score: float
    threshold: float
    evidence: tuple[TriggerScoreEvidence, ...]
    raw_score: float | None = None
    total_weight: float = 0.0
    minimum_total_weight: float = 0.0
    flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize score, threshold, evidence, and diagnostics after construction."""

        threshold, threshold_flags = _normalize_threshold(self.threshold)
        evidence, evidence_flags = _normalize_evidence(self.evidence)

        score = clamp_unit(self.score)
        raw_score = score if self.raw_score is None else clamp_unit(self.raw_score)

        object.__setattr__(self, "score", score)
        object.__setattr__(self, "raw_score", raw_score)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "total_weight", _non_negative_float(self.total_weight))
        object.__setattr__(
            self,
            "minimum_total_weight",
            _non_negative_float(self.minimum_total_weight),
        )
        object.__setattr__(
            self,
            "flags",
            _normalize_flags((*self.flags, *threshold_flags, *evidence_flags)),
        )

    @property
    def passed(self) -> bool:
        """Return whether the score passes threshold and evidence-mass gates."""

        if "invalid_threshold" in self.flags:
            # BREAKING: An invalid threshold no longer passes merely because the score
            # equals 1.0 after normalization; bad config now disables the pass result.
            return False
        return (
            self.total_weight >= self.minimum_total_weight
            and self.score >= self.threshold
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize one score card for logs or JSON payloads."""

        return {
            "score": round(self.score, 3),
            "raw_score": round(
                self.raw_score if self.raw_score is not None else self.score,
                3,
            ),
            "threshold": round(self.threshold, 3),
            "passed": self.passed,
            "total_weight": round(self.total_weight, 3),
            "minimum_total_weight": round(self.minimum_total_weight, 3),
            "flags": list(self.flags),
            "evidence": [item.to_dict() for item in self.evidence],
        }


def weighted_trigger_score(
    *,
    threshold: float,
    evidence: Iterable[TriggerScoreEvidence] | None,
    minimum_total_weight: float = 0.0,
    calibrator: Callable[[float], float] | None = None,
    max_evidence_items: int = MAX_EVIDENCE_ITEMS,
) -> WeightedTriggerScore:
    """Compute one weighted trigger score from normalized evidence."""

    evidence_items, evidence_flags = _normalize_evidence(
        evidence,
        max_items=max_evidence_items,
    )

    total_weight = fsum(item.weight for item in evidence_items)
    if total_weight <= 0.0:
        raw_score = 0.0
    else:
        raw_score = clamp_unit(
            fsum(item.contribution for item in evidence_items) / total_weight,
        )

    score = raw_score
    flags: list[str] = list(evidence_flags)

    if calibrator is not None:
        try:
            calibrated = _finite_float(calibrator(raw_score))
        except Exception:
            calibrated = None
            flags.append("calibrator_error")

        if calibrated is None:
            score = 0.0
            flags.append("invalid_calibrator_output")
        else:
            score = clamp_unit(calibrated)
            flags.append("score_calibrated")

    if total_weight < _non_negative_float(minimum_total_weight):
        flags.append("insufficient_total_weight")

    return WeightedTriggerScore(
        score=score,
        raw_score=raw_score,
        threshold=threshold,
        evidence=evidence_items,
        total_weight=total_weight,
        minimum_total_weight=minimum_total_weight,
        flags=tuple(flags),
    )


__all__ = [
    "MAX_EVIDENCE_ITEMS",
    "PiecewiseLinearCalibrator",
    "TriggerScoreEvidence",
    "WeightedTriggerScore",
    "bool_score",
    "clamp_unit",
    "hold_progress",
    "hold_progress_ns",
    "recent_progress",
    "recent_progress_ns",
    "weighted_trigger_score",
]