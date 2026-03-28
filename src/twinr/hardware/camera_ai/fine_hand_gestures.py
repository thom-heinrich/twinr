# CHANGELOG: 2026-03-28
# BUG-1: Resolve gesture scores per detected hand instead of globally so a high-confidence "None" on one hand no longer suppresses a valid gesture on another hand.
# BUG-2: Reject mapped fallback gestures when a stronger competing or unmapped class (for example MediaPipe "Closed_Fist" / "ILoveYou") dominates the same hand.
# BUG-3: Prevent task-specific custom models from overriding strong unrelated built-in gestures with weak preferred-label predictions.
# SEC-1: Bound hot-path iteration and label normalization sizes to prevent practical local CPU/RAM exhaustion from malformed recognizer payloads or poisoned model metadata on Raspberry Pi deployments.
# IMP-1: Add per-hand ambiguity gating via competitor-score margins to reduce false positives from close classifier ties.
# IMP-2: Add cached label normalization and non-materializing bounded iteration for lower latency and steadier memory use on Raspberry Pi 4.
# IMP-3: Add StableFineHandGestureResolver for temporal persistence / hysteresis in live-stream HCI paths without breaking the existing functional API.

"""Resolve MediaPipe gesture-recognizer outputs into Twinr's fine-hand contract."""

from __future__ import annotations

# ##REFACTOR: 2026-03-28##

import math
import re
from collections.abc import Iterator, Mapping, Set as AbstractSet
from dataclasses import dataclass
from functools import lru_cache
from itertools import islice
from typing import Final

from .config import _clamp_ratio
from .models import AICameraFineHandGesture


_CAMEL_BOUNDARY_RE_1: Final[re.Pattern[str]] = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_BOUNDARY_RE_2: Final[re.Pattern[str]] = re.compile(r"([a-z0-9])([A-Z])")
_CATEGORY_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[^0-9a-zA-Z]+")

_MAX_HANDS_TO_SCAN: Final[int] = 4
_MAX_CATEGORIES_PER_HAND: Final[int] = 16
_MAX_CATEGORY_NAME_CHARS: Final[int] = 128
_MIN_COMPETING_SCORE_MARGIN: Final[float] = 0.05

_OPEN_PALM_CUSTOM_OVERRIDE_FLOOR: Final[float] = 0.72
_OPEN_PALM_CUSTOM_OVERRIDE_MARGIN: Final[float] = 0.08
_TASK_SPECIFIC_CONCRETE_OVERRIDE_FLOOR: Final[float] = 0.78
_TASK_SPECIFIC_CONCRETE_OVERRIDE_MARGIN: Final[float] = 0.12

BUILTIN_FINE_GESTURE_MAP: Final[dict[str, AICameraFineHandGesture]] = {
    "thumb_up": AICameraFineHandGesture.THUMBS_UP,
    "thumbs_up": AICameraFineHandGesture.THUMBS_UP,
    "thumb_down": AICameraFineHandGesture.THUMBS_DOWN,
    "thumbs_down": AICameraFineHandGesture.THUMBS_DOWN,
    "pointing_up": AICameraFineHandGesture.POINTING,
    "pointing": AICameraFineHandGesture.POINTING,
    "victory": AICameraFineHandGesture.PEACE_SIGN,
    "peace": AICameraFineHandGesture.PEACE_SIGN,
    "peace_sign": AICameraFineHandGesture.PEACE_SIGN,
    "open_palm": AICameraFineHandGesture.OPEN_PALM,
}

CUSTOM_FINE_GESTURE_MAP: Final[dict[str, AICameraFineHandGesture]] = {
    "thumb_up": AICameraFineHandGesture.THUMBS_UP,
    "thumbs_up": AICameraFineHandGesture.THUMBS_UP,
    "thumb_down": AICameraFineHandGesture.THUMBS_DOWN,
    "thumbs_down": AICameraFineHandGesture.THUMBS_DOWN,
    "victory": AICameraFineHandGesture.PEACE_SIGN,
    "peace": AICameraFineHandGesture.PEACE_SIGN,
    "peace_sign": AICameraFineHandGesture.PEACE_SIGN,
    "ok": AICameraFineHandGesture.OK_SIGN,
    "ok_sign": AICameraFineHandGesture.OK_SIGN,
    "okay": AICameraFineHandGesture.OK_SIGN,
    "middle_finger": AICameraFineHandGesture.MIDDLE_FINGER,
    "flip_off": AICameraFineHandGesture.MIDDLE_FINGER,
}

_CUSTOM_ONLY_FINE_GESTURES: Final[frozenset[AICameraFineHandGesture]] = frozenset(
    {
        AICameraFineHandGesture.OK_SIGN,
        AICameraFineHandGesture.MIDDLE_FINGER,
    }
)
_BUILTIN_PRIORITY_FINE_GESTURES: Final[frozenset[AICameraFineHandGesture]] = frozenset(
    {
        AICameraFineHandGesture.THUMBS_UP,
        AICameraFineHandGesture.THUMBS_DOWN,
        AICameraFineHandGesture.POINTING,
        AICameraFineHandGesture.PEACE_SIGN,
    }
)

_NEGATIVE_FINE_GESTURE_LABELS: Final[frozenset[str]] = frozenset(
    {
        "none",
        "no_gesture",
        "no_hand_gesture",
        "background",
        "other",
        "unknown",
    }
)
_BUILTIN_GESTURE_CATEGORY_DENYLIST: Final[tuple[str, ...]] = ("None",)


def _safe_ratio(value: object, *, default: float = 0.0) -> float:
    """Clamp one ratio-like value to one finite float within [0.0, 1.0]."""

    score = _clamp_ratio(value, default=default)
    try:
        normalized = float(score)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(normalized):
        return default
    return max(0.0, min(1.0, normalized))


def _iter_limited(value: object, *, limit: int) -> Iterator[object]:
    """Yield at most ``limit`` items from one optional iterable without materializing it."""

    if (
        limit <= 0
        or value is None
        or isinstance(value, (str, bytes, bytearray, memoryview))
        or isinstance(value, Mapping)
    ):
        return iter(())
    try:
        iterator = iter(value)
    except TypeError:
        return iter(())
    return islice(iterator, limit)


def _round_optional_score(score: float) -> float | None:
    """Round one positive score for external API use."""

    if score <= 0.0:
        return None
    return round(score, 3)


def _read_category_label(category: object) -> object:
    """Extract one classifier label from heterogeneous category containers."""

    return getattr(
        category,
        "category_name",
        getattr(
            category,
            "categoryName",
            getattr(category, "display_name", getattr(category, "displayName", None)),
        ),
    )


def _read_category_score(category: object) -> object:
    """Extract one classifier score from heterogeneous category containers."""

    return getattr(category, "score", getattr(category, "confidence", 0.0))


@lru_cache(maxsize=256)
def _normalize_category_text(raw: str) -> str:
    """Normalize one already-string label to stable snake_case."""

    normalized = _CAMEL_BOUNDARY_RE_1.sub(r"\1_\2", raw)
    normalized = _CAMEL_BOUNDARY_RE_2.sub(r"\1_\2", normalized)
    normalized = _CATEGORY_TOKEN_RE.sub("_", normalized.casefold())
    return normalized.strip("_")


def _normalize_category_map(
    category_map: Mapping[str, AICameraFineHandGesture],
) -> dict[str, AICameraFineHandGesture]:
    """Normalize mapping keys once per resolve call."""

    normalized: dict[str, AICameraFineHandGesture] = {}
    for raw_label, gesture in category_map.items():
        label = normalize_category_name(raw_label)
        if label:
            normalized[label] = gesture
    return normalized


def _resolve_hand_candidate(
    gesture_set: object,
    *,
    category_map: Mapping[str, AICameraFineHandGesture],
    min_score: float,
) -> tuple[AICameraFineHandGesture, float | None]:
    """Resolve one detected hand worth of categories into one bounded gesture choice."""

    best_gesture = AICameraFineHandGesture.NONE
    best_score = 0.0
    best_negative_score = 0.0
    strongest_competitor_score = 0.0

    for category in _iter_limited(gesture_set, limit=_MAX_CATEGORIES_PER_HAND):
        label = normalize_category_name(_read_category_label(category))
        if not label:
            continue

        score = _safe_ratio(_read_category_score(category), default=0.0)

        if label in _NEGATIVE_FINE_GESTURE_LABELS:
            if score > best_negative_score:
                best_negative_score = score
            if score > strongest_competitor_score:
                strongest_competitor_score = score
            continue

        mapped = category_map.get(label)
        if mapped is None:
            if score > strongest_competitor_score:
                strongest_competitor_score = score
            continue

        if mapped == best_gesture:
            if score > best_score:
                best_score = score
            continue

        if score > best_score:
            if best_gesture != AICameraFineHandGesture.NONE and best_score > strongest_competitor_score:
                strongest_competitor_score = best_score
            best_gesture = mapped
            best_score = score
        elif score > strongest_competitor_score:
            strongest_competitor_score = score

    if best_gesture == AICameraFineHandGesture.NONE:
        return AICameraFineHandGesture.NONE, None
    if best_score < min_score:
        return AICameraFineHandGesture.NONE, None
    if best_negative_score >= best_score:
        return AICameraFineHandGesture.NONE, None
    if (
        strongest_competitor_score > 0.0
        and (best_score - strongest_competitor_score) < _MIN_COMPETING_SCORE_MARGIN
    ):
        return AICameraFineHandGesture.NONE, None
    return best_gesture, _round_optional_score(best_score)


def resolve_fine_hand_gesture(
    *,
    result: object,
    category_map: Mapping[str, AICameraFineHandGesture],
    min_score: float,
) -> tuple[AICameraFineHandGesture, float | None]:
    """Map one gesture-recognizer result into Twinr's bounded fine-gesture enum."""

    min_score = _safe_ratio(
        min_score,
        default=0.0,
    )  # AUDIT-FIX(#2): Normalize config thresholds before comparing classifier outputs.
    gestures = getattr(
        result,
        "gestures",
        None,
    )  # AUDIT-FIX(#1): Treat malformed recognizer payloads as "no gesture" instead of crashing.
    normalized_map = _normalize_category_map(category_map)

    best_gesture = AICameraFineHandGesture.NONE
    best_score = 0.0

    for gesture_set in _iter_limited(
        gestures,
        limit=_MAX_HANDS_TO_SCAN,
    ):  # AUDIT-FIX(#5): Resolve each detected hand independently; MediaPipe results are per-hand arrays.
        candidate_gesture, candidate_score = _resolve_hand_candidate(
            gesture_set,
            category_map=normalized_map,
            min_score=min_score,
        )
        candidate_score_value = _safe_ratio(
            candidate_score,
            default=0.0,
        )  # AUDIT-FIX(#2): Reject non-finite and out-of-range classifier scores deterministically.
        if candidate_gesture == AICameraFineHandGesture.NONE or candidate_score_value <= best_score:
            continue
        best_gesture = candidate_gesture
        best_score = candidate_score_value

    if best_gesture == AICameraFineHandGesture.NONE:
        return AICameraFineHandGesture.NONE, None
    return best_gesture, _round_optional_score(best_score)


def prefer_gesture_choice(
    first: tuple[AICameraFineHandGesture, float | None],
    second: tuple[AICameraFineHandGesture, float | None],
) -> tuple[AICameraFineHandGesture, float | None]:
    """Return the stronger bounded gesture choice between two candidates."""

    first_score = _safe_ratio(
        first[1],
        default=0.0,
    )  # AUDIT-FIX(#3): Compare only bounded finite scores when arbitrating between detectors.
    second_score = _safe_ratio(
        second[1],
        default=0.0,
    )  # AUDIT-FIX(#3): Prevent malformed optional scores from crashing or biasing arbitration.
    if second[0] == AICameraFineHandGesture.NONE:
        return first
    if first[0] == AICameraFineHandGesture.NONE or second_score > first_score:
        return second
    return first


def combine_builtin_and_custom_gesture_choice(
    builtin: tuple[AICameraFineHandGesture, float | None],
    custom: tuple[AICameraFineHandGesture, float | None],
) -> tuple[AICameraFineHandGesture, float | None]:
    """Merge built-in and custom gesture choices without letting custom-only labels overrun built-ins.

    Twinr's staged custom model currently only covers labels like ``ok_sign`` and
    ``middle_finger``. Those should supplement the MediaPipe built-in recognizer,
    not globally outrank strong built-in labels such as ``victory`` or
    ``pointing_up``. Prefer built-in labels when both recognizers disagree on a
    concrete non-generic symbol, but still allow custom-only labels to win over
    generic ``none`` / ``open_palm`` outputs so OK-sign remains usable.
    """

    builtin_gesture, builtin_confidence = builtin
    custom_gesture, custom_confidence = custom
    builtin_score = _safe_ratio(builtin_confidence, default=0.0)
    custom_score = _safe_ratio(custom_confidence, default=0.0)

    if custom_gesture == AICameraFineHandGesture.NONE:
        return builtin
    if builtin_gesture in {AICameraFineHandGesture.NONE, AICameraFineHandGesture.UNKNOWN}:
        return custom
    if custom_gesture == builtin_gesture:
        if custom_score >= builtin_score:
            return custom
        return builtin

    if custom_gesture in _CUSTOM_ONLY_FINE_GESTURES:
        if builtin_gesture in _BUILTIN_PRIORITY_FINE_GESTURES:
            return builtin
        if builtin_gesture == AICameraFineHandGesture.OPEN_PALM:
            if custom_score >= max(
                _OPEN_PALM_CUSTOM_OVERRIDE_FLOOR,
                builtin_score + _OPEN_PALM_CUSTOM_OVERRIDE_MARGIN,
            ):
                return custom
            return builtin
        return custom
    return builtin


def combine_task_specific_custom_gesture_choice(
    builtin: tuple[AICameraFineHandGesture, float | None],
    custom: tuple[AICameraFineHandGesture, float | None],
    *,
    preferred_custom_gestures: AbstractSet[AICameraFineHandGesture],
) -> tuple[AICameraFineHandGesture, float | None]:
    """Prefer task-specific custom labels before falling back to generic arbitration.

    Twinr's Pi live-HCI path now targets exactly ``thumbs_up``, ``thumbs_down``,
    and ``peace_sign``. When a custom model is trained specifically for that
    product slice, its predictions for those labels should outrank the generic
    built-in recognizer when the custom model is stronger, or when the built-in
    recognizer only surfaces one generic fallback such as ``open_palm``.
    """

    if not preferred_custom_gestures:
        return combine_builtin_and_custom_gesture_choice(builtin, custom)

    builtin_gesture, builtin_confidence = builtin
    custom_gesture, custom_confidence = custom
    if custom_gesture == AICameraFineHandGesture.NONE:
        return builtin
    if custom_gesture not in preferred_custom_gestures:
        return combine_builtin_and_custom_gesture_choice(builtin, custom)

    custom_score = _safe_ratio(custom_confidence, default=0.0)
    builtin_score = _safe_ratio(builtin_confidence, default=0.0)

    if builtin_gesture in {AICameraFineHandGesture.NONE, AICameraFineHandGesture.UNKNOWN}:
        return custom
    if builtin_gesture == custom_gesture:
        if custom_score >= builtin_score:
            return custom
        return builtin
    if builtin_gesture in preferred_custom_gestures:
        if custom_score >= builtin_score:
            return custom
        return builtin
    if builtin_gesture == AICameraFineHandGesture.OPEN_PALM:
        if custom_score >= max(
            _OPEN_PALM_CUSTOM_OVERRIDE_FLOOR,
            builtin_score + _OPEN_PALM_CUSTOM_OVERRIDE_MARGIN,
        ):
            return custom
        return builtin

    if custom_score >= max(
        _TASK_SPECIFIC_CONCRETE_OVERRIDE_FLOOR,
        builtin_score + _TASK_SPECIFIC_CONCRETE_OVERRIDE_MARGIN,
    ):
        return custom
    return builtin


def normalize_category_name(value: object) -> str:
    """Normalize one classifier category label to a stable token."""

    raw = str(value or "").strip()
    if not raw:
        return ""
    if len(raw) > _MAX_CATEGORY_NAME_CHARS:
        raw = raw[:_MAX_CATEGORY_NAME_CHARS]
    return _normalize_category_text(
        raw,
    )  # AUDIT-FIX(#4): Split CamelCase/PascalCase and collapse punctuation into stable snake_case tokens.


def builtin_gesture_category_denylist() -> tuple[str, ...]:
    """Return raw canned-gesture labels to suppress via official classifier options.

    MediaPipe's canned gesture classifier explicitly includes the ``None`` label
    in its category set. Twinr already handles "no gesture" as the absence of a
    positive symbol, so ask the official classifier to filter that label before
    downstream arbitration has to compete against it.
    """

    return _BUILTIN_GESTURE_CATEGORY_DENYLIST


@dataclass(slots=True)
class StableFineHandGestureResolver:
    """Stabilize frame-level gesture choices for live-stream HCI.

    This resolver adds small-state temporal persistence on top of the stateless
    frame resolver above. It is intentionally lightweight for Raspberry Pi use:
    a new gesture must persist for ``promote_frames`` updates before activation,
    a running gesture survives up to ``release_frames - 1`` blank frames, and
    gesture switches require the contender to persist and approach the current
    score within ``switch_margin``.
    """

    promote_frames: int = 2
    release_frames: int = 2
    switch_margin: float = 0.08
    ema_alpha: float = 0.6

    _current_gesture: AICameraFineHandGesture = AICameraFineHandGesture.NONE
    _current_score: float = 0.0
    _candidate_gesture: AICameraFineHandGesture = AICameraFineHandGesture.NONE
    _candidate_score: float = 0.0
    _candidate_streak: int = 0
    _empty_streak: int = 0

    def __post_init__(self) -> None:
        self.promote_frames = max(1, int(self.promote_frames))
        self.release_frames = max(1, int(self.release_frames))
        self.switch_margin = _safe_ratio(self.switch_margin, default=0.08)
        self.ema_alpha = _safe_ratio(self.ema_alpha, default=0.6)

    def reset(self) -> None:
        """Clear all temporal state."""

        self._current_gesture = AICameraFineHandGesture.NONE
        self._current_score = 0.0
        self._candidate_gesture = AICameraFineHandGesture.NONE
        self._candidate_score = 0.0
        self._candidate_streak = 0
        self._empty_streak = 0

    def peek(self) -> tuple[AICameraFineHandGesture, float | None]:
        """Return the currently stabilized gesture without mutating state."""

        if self._current_gesture == AICameraFineHandGesture.NONE:
            return AICameraFineHandGesture.NONE, None
        return self._current_gesture, _round_optional_score(self._current_score)

    def _blend_score(self, current: float, observed: float) -> float:
        """Apply one tiny EMA to smooth stabilized confidences."""

        if current <= 0.0:
            return observed
        return (self.ema_alpha * observed) + ((1.0 - self.ema_alpha) * current)

    def _track_candidate(
        self,
        gesture: AICameraFineHandGesture,
        score: float,
    ) -> None:
        """Update the current promotion candidate."""

        if self._candidate_gesture == gesture:
            self._candidate_streak += 1
            self._candidate_score = self._blend_score(self._candidate_score, score)
            return
        self._candidate_gesture = gesture
        self._candidate_score = score
        self._candidate_streak = 1

    def _promote_candidate(self) -> tuple[AICameraFineHandGesture, float | None]:
        """Promote the persisted candidate to the active stabilized gesture."""

        self._current_gesture = self._candidate_gesture
        self._current_score = self._candidate_score
        self._candidate_gesture = AICameraFineHandGesture.NONE
        self._candidate_score = 0.0
        self._candidate_streak = 0
        self._empty_streak = 0
        return self.peek()

    def update(
        self,
        choice: tuple[AICameraFineHandGesture, float | None],
    ) -> tuple[AICameraFineHandGesture, float | None]:
        """Update temporal state from one new frame-level gesture choice."""

        gesture, confidence = choice
        score = _safe_ratio(confidence, default=0.0)

        if gesture in {AICameraFineHandGesture.NONE, AICameraFineHandGesture.UNKNOWN} or score <= 0.0:
            self._candidate_gesture = AICameraFineHandGesture.NONE
            self._candidate_score = 0.0
            self._candidate_streak = 0
            if self._current_gesture == AICameraFineHandGesture.NONE:
                return AICameraFineHandGesture.NONE, None
            self._empty_streak += 1
            if self._empty_streak < self.release_frames:
                return self.peek()
            self.reset()
            return AICameraFineHandGesture.NONE, None

        self._empty_streak = 0

        if self._current_gesture == AICameraFineHandGesture.NONE:
            self._track_candidate(gesture, score)
            if self._candidate_streak >= self.promote_frames:
                return self._promote_candidate()
            return AICameraFineHandGesture.NONE, None

        if gesture == self._current_gesture:
            self._current_score = self._blend_score(self._current_score, score)
            self._candidate_gesture = AICameraFineHandGesture.NONE
            self._candidate_score = 0.0
            self._candidate_streak = 0
            return self.peek()

        self._track_candidate(gesture, score)
        if (
            self._candidate_streak >= self.promote_frames
            and self._candidate_score + self.switch_margin >= self._current_score
        ):
            return self._promote_candidate()
        return self.peek()


__all__ = [
    "BUILTIN_FINE_GESTURE_MAP",
    "CUSTOM_FINE_GESTURE_MAP",
    "StableFineHandGestureResolver",
    "builtin_gesture_category_denylist",
    "combine_builtin_and_custom_gesture_choice",
    "combine_task_specific_custom_gesture_choice",
    "normalize_category_name",
    "prefer_gesture_choice",
    "resolve_fine_hand_gesture",
]