"""Load bounded per-gesture calibration for the social camera surface.

Twinr's live Pi gesture path should not rely on one global confirmation and
confidence threshold for every fine hand symbol. `OK_SIGN` and
`MIDDLE_FINGER` are materially easier to confuse than `THUMBS_UP`, while
`PEACE_SIGN` often benefits from a slightly longer confirmation window than a
generic `POINTING` pose. This helper keeps that policy in one small place and
optionally loads an operator-edited calibration file from
``state/mediapipe/gesture_calibration.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Final, SupportsFloat, SupportsIndex, cast

from twinr.agent.base_agent.config import TwinrConfig

from .engine import SocialFineHandGesture


_DEFAULT_CALIBRATION_PATH: Final[str] = "state/mediapipe/gesture_calibration.json"


@dataclass(frozen=True, slots=True)
class FineHandGesturePolicy:
    """Describe bounded user-facing acceptance rules for one hand symbol."""

    min_confidence: float
    confirm_samples: int
    hold_s: float
    min_visible_s: float = 0.0

    def __post_init__(self) -> None:
        min_confidence = _clamp_ratio(self.min_confidence, default=0.72)
        confirm_samples = max(1, int(self.confirm_samples))
        hold_s = max(0.0, min(1.5, _coerce_float(self.hold_s, default=0.45)))
        min_visible_s = max(0.0, min(3.0, _coerce_float(self.min_visible_s, default=0.0)))
        object.__setattr__(self, "min_confidence", min_confidence)
        object.__setattr__(self, "confirm_samples", confirm_samples)
        object.__setattr__(self, "hold_s", hold_s)
        object.__setattr__(self, "min_visible_s", min_visible_s)


@dataclass(frozen=True, slots=True)
class GestureCalibrationProfile:
    """Store calibrated per-gesture acceptance rules for the social layer."""

    fine_hand: dict[SocialFineHandGesture, FineHandGesturePolicy] = field(default_factory=dict)
    source_path: str | None = None

    @classmethod
    def defaults(cls) -> "GestureCalibrationProfile":
        """Return one conservative built-in fine-hand calibration profile."""

        return cls(
            fine_hand={
                SocialFineHandGesture.THUMBS_UP: FineHandGesturePolicy(0.68, 1, 0.35, 1.0),
                SocialFineHandGesture.THUMBS_DOWN: FineHandGesturePolicy(0.78, 1, 0.35, 1.0),
                SocialFineHandGesture.POINTING: FineHandGesturePolicy(0.70, 1, 0.32),
                SocialFineHandGesture.PEACE_SIGN: FineHandGesturePolicy(0.78, 1, 0.40, 1.0),
                SocialFineHandGesture.OK_SIGN: FineHandGesturePolicy(0.86, 1, 0.46),
                SocialFineHandGesture.MIDDLE_FINGER: FineHandGesturePolicy(0.90, 1, 0.28),
            }
        )

    @classmethod
    def from_runtime_config(cls, config: TwinrConfig | object) -> "GestureCalibrationProfile":
        """Load one optional runtime calibration file with conservative fallback."""

        defaults = cls.defaults()
        project_root = Path(getattr(config, "project_root", ".") or ".")
        calibration_path = (project_root / _DEFAULT_CALIBRATION_PATH).resolve(strict=False)
        if not calibration_path.is_file():
            return defaults
        try:
            payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        except Exception:
            return defaults
        return cls(
            fine_hand=_merge_fine_hand_policies(defaults.fine_hand, payload),
            source_path=str(calibration_path),
        )

    def fine_hand_policy(
        self,
        gesture: SocialFineHandGesture,
        *,
        fallback_min_confidence: float,
        fallback_confirm_samples: int,
        fallback_hold_s: float,
        fallback_min_visible_s: float = 0.0,
    ) -> FineHandGesturePolicy:
        """Return the calibrated policy for one gesture with bounded fallback."""

        calibrated = self.fine_hand.get(gesture)
        if calibrated is not None:
            return calibrated
        return FineHandGesturePolicy(
            min_confidence=fallback_min_confidence,
            confirm_samples=fallback_confirm_samples,
            hold_s=fallback_hold_s,
            min_visible_s=fallback_min_visible_s,
        )


def _merge_fine_hand_policies(
    defaults: dict[SocialFineHandGesture, FineHandGesturePolicy],
    payload: object,
) -> dict[SocialFineHandGesture, FineHandGesturePolicy]:
    """Merge one JSON payload onto the built-in fine-hand defaults."""

    merged = dict(defaults)
    fine_hand_payload = payload
    if isinstance(payload, dict):
        fine_hand_payload = payload.get("fine_hand", payload)
    if not isinstance(fine_hand_payload, dict):
        return merged
    for raw_name, raw_policy in fine_hand_payload.items():
        gesture = _coerce_fine_hand_gesture(raw_name)
        if gesture is None or gesture in {SocialFineHandGesture.NONE, SocialFineHandGesture.UNKNOWN}:
            continue
        if not isinstance(raw_policy, dict):
            continue
        fallback = merged.get(
            gesture,
            FineHandGesturePolicy(0.72, 1, 0.45),
        )
        merged[gesture] = FineHandGesturePolicy(
            min_confidence=_coerce_float(raw_policy.get("min_confidence"), default=fallback.min_confidence),
            confirm_samples=_coerce_int(raw_policy.get("confirm_samples"), default=fallback.confirm_samples),
            hold_s=_coerce_float(raw_policy.get("hold_s"), default=fallback.hold_s),
            min_visible_s=_coerce_float(raw_policy.get("min_visible_s"), default=fallback.min_visible_s),
        )
    return merged


def _coerce_fine_hand_gesture(value: object) -> SocialFineHandGesture | None:
    """Normalize one token into a known fine-hand gesture enum."""

    text = " ".join(str(value or "").strip().lower().replace("-", " ").split())
    if not text:
        return None
    normalized = text.replace(" ", "_")
    aliases = {
        "peace": SocialFineHandGesture.PEACE_SIGN,
        "victory": SocialFineHandGesture.PEACE_SIGN,
    }
    gesture = aliases.get(normalized)
    if gesture is not None:
        return gesture
    try:
        return SocialFineHandGesture(normalized)
    except ValueError:
        return None


def _coerce_float(value: object, *, default: float) -> float:
    """Return one finite float with a safe fallback."""

    try:
        numeric = float(cast(str | bytes | bytearray | SupportsFloat | SupportsIndex, value))
    except (TypeError, ValueError):
        return default
    if numeric != numeric:
        return default
    return numeric


def _coerce_int(value: object, *, default: int) -> int:
    """Return one positive integer with a safe fallback."""

    try:
        numeric = int(cast(str | bytes | bytearray | SupportsIndex, value))
    except (TypeError, ValueError):
        return default
    return max(1, numeric)


def _clamp_ratio(value: object, *, default: float) -> float:
    """Clamp one ratio-like value into ``[0.0, 1.0]``."""

    numeric = _coerce_float(value, default=default)
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


__all__ = [
    "FineHandGesturePolicy",
    "GestureCalibrationProfile",
]
