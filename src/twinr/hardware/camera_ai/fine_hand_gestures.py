"""Resolve MediaPipe gesture-recognizer outputs into Twinr's fine-hand contract."""

from __future__ import annotations

# ##REFACTOR: 2026-03-19##

import math
import re
from collections.abc import Mapping
from typing import Final

from .config import _clamp_ratio
from .models import AICameraFineHandGesture


_CAMEL_BOUNDARY_RE_1: Final[re.Pattern[str]] = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_BOUNDARY_RE_2: Final[re.Pattern[str]] = re.compile(r"([a-z0-9])([A-Z])")
_CATEGORY_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[^0-9a-zA-Z]+")

BUILTIN_FINE_GESTURE_MAP: Final[dict[str, AICameraFineHandGesture]] = {
    "thumb_up": AICameraFineHandGesture.THUMBS_UP,
    "thumbs_up": AICameraFineHandGesture.THUMBS_UP,
    "thumb_down": AICameraFineHandGesture.THUMBS_DOWN,
    "thumbs_down": AICameraFineHandGesture.THUMBS_DOWN,
    "pointing_up": AICameraFineHandGesture.POINTING,
    "pointing": AICameraFineHandGesture.POINTING,
    "open_palm": AICameraFineHandGesture.OPEN_PALM,
}

CUSTOM_FINE_GESTURE_MAP: Final[dict[str, AICameraFineHandGesture]] = {
    "ok": AICameraFineHandGesture.OK_SIGN,
    "ok_sign": AICameraFineHandGesture.OK_SIGN,
    "okay": AICameraFineHandGesture.OK_SIGN,
    "middle_finger": AICameraFineHandGesture.MIDDLE_FINGER,
    "flip_off": AICameraFineHandGesture.MIDDLE_FINGER,
}

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


def _safe_iterable(value: object) -> tuple[object, ...]:
    """Materialize one optional iterable into a safe tuple."""

    if value is None or isinstance(value, (str, bytes)):
        return ()
    try:
        return tuple(value)
    except TypeError:
        return ()


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
    gestures = _safe_iterable(
        getattr(result, "gestures", None)
    )  # AUDIT-FIX(#1): Treat malformed or non-iterable recognizer payloads as "no gesture" instead of crashing.
    best_gesture = AICameraFineHandGesture.NONE
    best_score = 0.0
    best_negative_score = 0.0
    for gesture_set in gestures:
        for category in _safe_iterable(
            gesture_set
        ):  # AUDIT-FIX(#1): Guard malformed per-hand category containers in callback/live-stream paths.
            label = normalize_category_name(getattr(category, "category_name", None))
            if not label:
                continue
            score = _safe_ratio(
                getattr(category, "score", 0.0),
                default=0.0,
            )  # AUDIT-FIX(#2): Reject non-finite and out-of-range classifier scores deterministically.
            if label in _NEGATIVE_FINE_GESTURE_LABELS and score > best_negative_score:
                best_negative_score = score
            mapped = category_map.get(label)
            if mapped is None:
                continue
            if score < min_score or score <= best_score:
                continue
            best_gesture = mapped
            best_score = score
    if best_gesture == AICameraFineHandGesture.NONE or best_negative_score >= best_score:
        return AICameraFineHandGesture.NONE, None
    return best_gesture, round(best_score, 3)


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


def normalize_category_name(value: object) -> str:
    """Normalize one classifier category label to a stable token."""

    raw = str(value or "").strip()
    if not raw:
        return ""
    normalized = _CAMEL_BOUNDARY_RE_1.sub(
        r"\1_\2",
        raw,
    )  # AUDIT-FIX(#4): Split common CamelCase/PascalCase labels before token normalization.
    normalized = _CAMEL_BOUNDARY_RE_2.sub(
        r"\1_\2",
        normalized,
    )  # AUDIT-FIX(#4): Preserve existing snake_case while normalizing custom-model label variants.
    normalized = _CATEGORY_TOKEN_RE.sub(
        "_",
        normalized.casefold(),
    )  # AUDIT-FIX(#4): Collapse punctuation, hyphens and repeated separators into stable snake_case tokens.
    return normalized.strip("_")


__all__ = [
    "BUILTIN_FINE_GESTURE_MAP",
    "CUSTOM_FINE_GESTURE_MAP",
    "normalize_category_name",
    "prefer_gesture_choice",
    "resolve_fine_hand_gesture",
]
