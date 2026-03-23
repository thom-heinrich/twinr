"""Persist bounded gesture-candidate frames for manual optical QA.

The live gesture lane can produce structured evidence without immediately
publishing an emoji. When that happens, operators need one visual artifact to
answer the concrete question: "What exactly was in frame when Twinr thought a
gesture might be present?" This helper keeps that capture path bounded with a
cooldown, finite retention, and compact JSON sidecars.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Mapping
import json
import math
import time

import numpy as np
from PIL import Image

from .config import AICameraAdapterConfig


_NEGATIVE_LABELS = frozenset({"", "none", "unknown", "unavailable"})


@dataclass(frozen=True, slots=True)
class GestureCandidateCaptureResult:
    """Describe one bounded gesture-candidate capture attempt."""

    saved: bool
    reasons: tuple[str, ...] = ()
    image_path: str | None = None
    metadata_path: str | None = None
    skipped_reason: str | None = None
    error: str | None = None

    def debug_fields(self) -> dict[str, object]:
        """Return one JSON-safe summary for the gesture debug payload."""

        return {
            "candidate_capture_saved": self.saved,
            "candidate_capture_reasons": list(self.reasons),
            "candidate_capture_image_path": self.image_path,
            "candidate_capture_metadata_path": self.metadata_path,
            "candidate_capture_skipped_reason": self.skipped_reason,
            "candidate_capture_error": self.error,
        }


class GestureCandidateCaptureStore:
    """Save one bounded JPEG plus compact metadata when gesture evidence appears."""

    def __init__(
        self,
        *,
        capture_dir: str,
        cooldown_s: float,
        max_images: int,
        clock: Any = time.time,
    ) -> None:
        self.capture_dir = Path(capture_dir)
        self.cooldown_s = max(0.0, float(cooldown_s))
        self.max_images = max(1, int(max_images))
        self._clock = clock
        self._lock = Lock()
        self._last_capture_at: float | None = None
        self._capture_sequence = 0

    @classmethod
    def from_config(
        cls,
        config: AICameraAdapterConfig,
        *,
        clock: Any = time.time,
    ) -> "GestureCandidateCaptureStore":
        """Build one bounded candidate-capture store from camera config."""

        return cls(
            capture_dir=config.gesture_candidate_capture_dir,
            cooldown_s=config.gesture_candidate_capture_cooldown_s,
            max_images=config.gesture_candidate_capture_max_images,
            clock=clock,
        )

    def maybe_capture(
        self,
        *,
        observed_at: float,
        frame_rgb: Any,
        debug_details: Mapping[str, object] | None,
    ) -> GestureCandidateCaptureResult:
        """Save one candidate frame when the debug evidence suggests gesture intent."""

        reasons = _candidate_reasons(debug_details)
        if not reasons:
            return GestureCandidateCaptureResult(saved=False, skipped_reason="no_candidate_signal")
        with self._lock:
            now = _finite_float_or_default(observed_at, default=self._clock())
            last_capture_at = self._last_capture_at
            if (
                last_capture_at is not None
                and self.cooldown_s > 0.0
                and (now - last_capture_at) < self.cooldown_s
            ):
                return GestureCandidateCaptureResult(
                    saved=False,
                    reasons=reasons,
                    skipped_reason="cooldown_active",
                )
            array = _normalize_rgb_array(frame_rgb)
            if array is None:
                return GestureCandidateCaptureResult(
                    saved=False,
                    reasons=reasons,
                    skipped_reason="frame_unavailable",
                )
            self.capture_dir.mkdir(parents=True, exist_ok=True)
            self._capture_sequence += 1
            stem = _capture_stem(observed_at=now, sequence=self._capture_sequence, reasons=reasons)
            image_path = self.capture_dir / f"{stem}.jpg"
            metadata_path = self.capture_dir / f"{stem}.json"
            try:
                Image.fromarray(array).save(image_path, format="JPEG", quality=85)
                metadata = {
                    "captured_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "observed_at": round(now, 6),
                    "reasons": list(reasons),
                    "frame_shape": [int(dimension) for dimension in array.shape],
                    "gesture_debug": _json_safe_value(dict(debug_details or {})),
                }
                metadata_path.write_text(
                    json.dumps(metadata, ensure_ascii=True, sort_keys=True, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                _best_effort_unlink(image_path)
                _best_effort_unlink(metadata_path)
                return GestureCandidateCaptureResult(
                    saved=False,
                    reasons=reasons,
                    skipped_reason="write_failed",
                    error=_error_code(exc),
                )
            self._last_capture_at = now
            self._prune_locked()
            return GestureCandidateCaptureResult(
                saved=True,
                reasons=reasons,
                image_path=str(image_path.resolve()),
                metadata_path=str(metadata_path.resolve()),
            )

    def _prune_locked(self) -> None:
        """Keep only the newest bounded set of capture artifacts."""

        stems = sorted(
            {
                path.stem
                for path in self.capture_dir.glob("gesture_candidate_*.jpg")
            }
            | {
                path.stem
                for path in self.capture_dir.glob("gesture_candidate_*.json")
            }
        )
        excess = max(0, len(stems) - self.max_images)
        for stem in stems[:excess]:
            _best_effort_unlink(self.capture_dir / f"{stem}.jpg")
            _best_effort_unlink(self.capture_dir / f"{stem}.json")


def _candidate_reasons(debug_details: Mapping[str, object] | None) -> tuple[str, ...]:
    """Return one stable ordered list of reasons why a frame is worth saving."""

    details = debug_details or {}
    reasons: list[str] = []
    if _coerce_non_negative_int(details.get("live_hand_count")) > 0:
        reasons.append("live_hand")
    if _normalized_label(details.get("live_fine_hand_gesture")) not in _NEGATIVE_LABELS:
        reasons.append("live_fine_gesture")
    if _normalized_label(details.get("live_gesture_event")) not in _NEGATIVE_LABELS:
        reasons.append("live_gesture_event")
    if _normalized_label(details.get("pose_fallback_fine_hand_gesture")) not in _NEGATIVE_LABELS:
        reasons.append("pose_fallback_fine_gesture")
    if _normalized_label(details.get("pose_fallback_gesture_event")) not in _NEGATIVE_LABELS:
        reasons.append("pose_fallback_gesture_event")
    if _normalized_label(details.get("pose_hint_source")) not in _NEGATIVE_LABELS:
        if _finite_float_or_default(details.get("pose_hint_confidence"), default=None) is not None:
            reasons.append("pose_hint")
    if _normalized_label(details.get("final_resolved_source")) not in _NEGATIVE_LABELS:
        reasons.append("resolved_candidate")
    return tuple(dict.fromkeys(reasons))


def _capture_stem(*, observed_at: float, sequence: int, reasons: tuple[str, ...]) -> str:
    """Build one stable bounded capture stem for image and metadata files."""

    timestamp = datetime.fromtimestamp(observed_at, tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    millis = int(round((observed_at % 1.0) * 1000.0))
    reason_suffix = "_".join(reason[:24] for reason in reasons[:3]) or "candidate"
    return f"gesture_candidate_{timestamp}{millis:03d}Z_{sequence:05d}_{reason_suffix}"


def _normalize_rgb_array(frame_rgb: Any) -> np.ndarray | None:
    """Coerce one frame-like object into a contiguous RGB uint8 array."""

    try:
        array = np.asarray(frame_rgb)
    except Exception:
        return None
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        return None
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return array


def _json_safe_value(value: object) -> object:
    """Convert one object tree into bounded JSON-safe primitives."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe_value(item) for item in value]
    return str(value)


def _normalized_label(value: object) -> str:
    text = str(value or "").strip().lower()
    return text or ""


def _finite_float_or_default(value: object, *, default: float | None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _coerce_non_negative_int(value: object) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, number)


def _best_effort_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except Exception:
        return


def _error_code(exc: Exception) -> str:
    return exc.__class__.__name__.strip().lower() or "capture_write_failed"


__all__ = [
    "GestureCandidateCaptureResult",
    "GestureCandidateCaptureStore",
]
