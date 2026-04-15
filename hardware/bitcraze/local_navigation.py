"""Plan bounded local inspect motion from one short-range clearance envelope.

This module owns the host-side local-navigation policy for Twinr's first
Crazyflie autonomy slice. It does not talk to cflib directly and it does not
send setpoints. Instead it converts one local free-space observation into one
explicit, bounded motion decision that higher orchestration layers can execute
through the existing hover primitive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


_DIRECTION_ORDER = ("forward", "left", "right", "back")
_CLEARANCE_KEYS = ("front", "back", "left", "right", "up", "down")
_DIRECTION_TO_VECTOR = {
    "forward": (1.0, 0.0),
    "back": (-1.0, 0.0),
    "left": (0.0, 1.0),
    "right": (0.0, -1.0),
}
_DIRECTION_TO_CLEARANCE_KEY = {
    "forward": "front",
    "back": "back",
    "left": "left",
    "right": "right",
}


@dataclass(frozen=True, slots=True)
class LocalInspectNavigationPolicy:
    """Describe Twinr's bounded host-side local inspect policy."""

    target_height_m: float = 0.25
    nominal_translation_m: float = 0.20
    min_translation_m: float = 0.10
    max_translation_m: float = 0.30
    required_post_move_clearance_m: float = 0.35
    translation_velocity_mps: float = 0.15
    hover_settle_s: float = 0.35
    capture_dwell_s: float = 0.30

    def normalized(self) -> "LocalInspectNavigationPolicy":
        """Return one validated policy copy or raise on invalid bounds."""

        if self.target_height_m <= 0.0:
            raise ValueError("target_height_m must be > 0")
        if self.nominal_translation_m <= 0.0:
            raise ValueError("nominal_translation_m must be > 0")
        if self.min_translation_m <= 0.0:
            raise ValueError("min_translation_m must be > 0")
        if self.max_translation_m < self.min_translation_m:
            raise ValueError("max_translation_m must be >= min_translation_m")
        if self.nominal_translation_m < self.min_translation_m:
            raise ValueError("nominal_translation_m must be >= min_translation_m")
        if self.required_post_move_clearance_m <= 0.0:
            raise ValueError("required_post_move_clearance_m must be > 0")
        if self.translation_velocity_mps <= 0.0:
            raise ValueError("translation_velocity_mps must be > 0")
        if self.hover_settle_s < 0.0:
            raise ValueError("hover_settle_s must be >= 0")
        if self.capture_dwell_s < 0.0:
            raise ValueError("capture_dwell_s must be >= 0")
        return self


@dataclass(frozen=True, slots=True)
class LocalTranslationCandidate:
    """Describe one directionally valid translation candidate."""

    direction: str
    forward_m: float
    left_m: float
    clearance_m: float
    allowed_distance_m: float


@dataclass(frozen=True, slots=True)
class LocalInspectNavigationPlan:
    """Describe the bounded local inspect motion chosen for one mission run."""

    decision: str
    reason: str
    target_height_m: float
    translation_velocity_mps: float
    hover_settle_s: float
    capture_dwell_s: float
    selected_translation: LocalTranslationCandidate | None
    candidates: tuple[LocalTranslationCandidate, ...]
    clearance_snapshot: dict[str, float | None]


def _normalize_clearance_value(value: object) -> float | None:
    """Normalize one directional clearance reading in meters."""

    if value is None:
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0.0:
        return None
    return numeric


def normalize_clearance_snapshot(
    clearance_snapshot: Mapping[str, object],
) -> dict[str, float | None]:
    """Normalize one clearance mapping to Twinr's canonical directional keys."""

    normalized: dict[str, float | None] = {}
    for key in _CLEARANCE_KEYS:
        normalized[key] = _normalize_clearance_value(clearance_snapshot.get(key))
    return normalized


def _candidate_for_direction(
    *,
    direction: str,
    clearance_m: float | None,
    policy: LocalInspectNavigationPolicy,
) -> LocalTranslationCandidate | None:
    """Return one valid translation candidate for one body-frame direction."""

    if clearance_m is None:
        return None
    allowed_distance_m = min(
        float(policy.nominal_translation_m),
        float(policy.max_translation_m),
        float(clearance_m) - float(policy.required_post_move_clearance_m),
    )
    if allowed_distance_m < float(policy.min_translation_m):
        return None
    forward_sign, left_sign = _DIRECTION_TO_VECTOR[direction]
    return LocalTranslationCandidate(
        direction=direction,
        forward_m=forward_sign * allowed_distance_m,
        left_m=left_sign * allowed_distance_m,
        clearance_m=float(clearance_m),
        allowed_distance_m=allowed_distance_m,
    )


def plan_local_inspect_navigation(
    *,
    clearance_snapshot: Mapping[str, object],
    policy: LocalInspectNavigationPolicy,
) -> LocalInspectNavigationPlan:
    """Return one bounded local inspect plan from a directional clearance map."""

    normalized_policy = policy.normalized()
    normalized_clearance = normalize_clearance_snapshot(clearance_snapshot)

    candidates: list[LocalTranslationCandidate] = []
    for direction in _DIRECTION_ORDER:
        clearance_key = _DIRECTION_TO_CLEARANCE_KEY[direction]
        candidate = _candidate_for_direction(
            direction=direction,
            clearance_m=normalized_clearance[clearance_key],
            policy=normalized_policy,
        )
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        return LocalInspectNavigationPlan(
            decision="hover_anchor_only",
            reason=(
                "No lateral translation satisfied the bounded post-move clearance "
                "contract; inspect from the current hover anchor."
            ),
            target_height_m=float(normalized_policy.target_height_m),
            translation_velocity_mps=float(normalized_policy.translation_velocity_mps),
            hover_settle_s=float(normalized_policy.hover_settle_s),
            capture_dwell_s=float(normalized_policy.capture_dwell_s),
            selected_translation=None,
            candidates=(),
            clearance_snapshot=normalized_clearance,
        )

    selected = max(
        candidates,
        key=lambda candidate: (
            candidate.allowed_distance_m,
            -_DIRECTION_ORDER.index(candidate.direction),
        ),
    )
    return LocalInspectNavigationPlan(
        decision="translate_then_capture",
        reason=(
            f"Selected the body-frame `{selected.direction}` lane because it offered "
            f"the largest bounded travel budget ({selected.allowed_distance_m:.2f} m)."
        ),
        target_height_m=float(normalized_policy.target_height_m),
        translation_velocity_mps=float(normalized_policy.translation_velocity_mps),
        hover_settle_s=float(normalized_policy.hover_settle_s),
        capture_dwell_s=float(normalized_policy.capture_dwell_s),
        selected_translation=selected,
        candidates=tuple(candidates),
        clearance_snapshot=normalized_clearance,
    )
