# CHANGELOG: 2026-03-28
# BUG-1: Accept tuple/list/numpy/mapping/object boxes. Picamera2 IMX500 boxes are tuple-like (x, y, w, h); the old code disabled pose/hint cache reuse.
# BUG-2: Replaced absolute normalized-threshold matching with identity-aware IoU/relative-scale matching. The old code broke on pixel-space boxes and could cross-bind nearby people.
# BUG-3: Cache reads/writes are now atomic under an internal RLock, preventing torn snapshots between observation and gesture lanes.
# BUG-4: Empty-result detection now checks array size/nbytes instead of only len(), avoiding false "non-empty" tensors such as shape (1, 0, 4).
# BUG-5: Wall/monotonic clocks are normalized across s/ms/us/ns injection sources so cache TTL math stays correct.
# SEC-1: Pose/gesture cache reuse now prefers stable track IDs and stronger association to prevent cross-person gesture spoofing in multi-person scenes.
# SEC-2: lock_timeout_s and pose_refresh_s are clamped to sane maxima to reduce config-poisoning / operator-misconfig denial-of-service.
# IMP-1: Added Picamera2/IMX500-compatible box parsing for attrs, mappings, tuples, numpy arrays, and corner formats.
# IMP-2: Added low-cost frontier association gates (track ID, IoU, relative center distance, area ratio) suited to Pi 4 live-stream pipelines.
# IMP-3: Extended error classification for common libcamera/Picamera2 timeout/dequeue/request-starvation failures.

# mypy: disable-error-code="attr-defined,assignment,var-annotated"
"""Cache, timing, and error helpers for the local AI-camera adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import math
import threading
import time

from ..models import AICameraMotionState
from .common import LOGGER
from .types import PoseResult

logger = LOGGER

_MISSING = object()
_MAX_LOCK_TIMEOUT_S = 10.0
_MAX_POSE_REFRESH_S = 5.0
_MIN_BOX_EPSILON = 1e-6
_RLOCK_TYPE = type(threading.RLock())
_ERROR_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("camera_busy", ("camera __init__ sequence did not complete", "device or resource busy")),
    ("camera_timeout", ("device timeout detected", "dequeue timer expired", "timed out waiting for next frame")),
    ("camera_request_starvation", ("requests stop being returned", "no frames are produced")),
    ("camera_resource_exhausted", ("cannot allocate memory", "out of memory")),
    ("camera_session_start_failed", ("session_start_failed",)),
    ("imx500_not_enumerated", ("requested camera dev-node not found",)),
    ("picamera2_unavailable", ("picamera2_unavailable",)),
    ("detection_capture_failed", ("detection_capture_failed",)),
    ("detection_outputs_missing", ("detection_outputs_missing",)),
    ("detection_outputs_invalid_container", ("detection_outputs_invalid_container",)),
    ("detection_outputs_incomplete", ("detection_outputs_incomplete",)),
    ("detection_parse_failed", ("detection_parse_failed",)),
    ("metadata_timeout", ("metadata_timeout",)),
    ("mediapipe_custom_gesture_model_missing", ("mediapipe_custom_gesture_model_missing",)),
    ("mediapipe_pose_model_missing", ("mediapipe_pose_model_missing",)),
    ("mediapipe_hand_landmarker_model_missing", ("mediapipe_hand_landmarker_model_missing",)),
    ("mediapipe_gesture_model_missing", ("mediapipe_gesture_model_missing",)),
    ("mediapipe_unavailable", ("mediapipe_unavailable",)),
    ("model_missing", ("model_missing",)),
    ("pose_dependency_missing", ("pose_dependency_missing",)),
    ("rgb_frame_missing", ("rgb_frame_missing",)),
    ("pose_decode_failed", ("pose_outputs_missing", "pose_people_missing", "operands could not be broadcast together")),
    ("pose_confidence_low", ("pose_confidence_low",)),
)

try:
    BaseExceptionGroup
except NameError:  # pragma: no cover - Python < 3.11
    _BASE_EXCEPTION_GROUP_TYPE: type[BaseException] | None = None
else:
    _BASE_EXCEPTION_GROUP_TYPE = BaseExceptionGroup


class AICameraAdapterStateMixin:
    """Keep cache, timing, and error helpers isolated from observation orchestration."""

    def _state_lock(self) -> threading.RLock:
        lock = getattr(self, "_adapter_state_lock", None)
        if isinstance(lock, _RLOCK_TYPE):
            return lock
        lock = threading.RLock()
        self._adapter_state_lock = lock
        return lock

    def _clear_pose_cache(self) -> None:
        with self._state_lock():
            self._last_pose_result = None
            self._last_pose_at = None
            self._last_pose_monotonic = None
            self._last_pose_box_metrics = None
            self._clear_pose_hint_cache()

    def _store_pose_result(
        self,
        pose: PoseResult,
        *,
        observed_at: float,
        observed_monotonic: float,
        primary_person_box: Any,
    ) -> None:
        box_metrics = self._extract_box_association_metrics(primary_person_box)
        sparse_keypoints = dict(pose.sparse_keypoints) if pose.sparse_keypoints else {}
        with self._state_lock():
            self._last_pose_result = pose
            self._last_pose_at = observed_at
            self._last_pose_monotonic = observed_monotonic
            self._last_pose_box_metrics = box_metrics
            if sparse_keypoints:
                self._last_pose_hint_keypoints = sparse_keypoints
                self._last_pose_hint_confidence = pose.pose_confidence
                self._last_pose_hint_monotonic = observed_monotonic
                self._last_pose_hint_box_metrics = box_metrics
            else:
                self._clear_pose_hint_cache()

    def _clear_motion_state(self) -> None:
        with self._state_lock():
            self._last_motion_box = None
            self._last_motion_person_count = 0
            self._last_motion_at = None
            self._last_motion_monotonic = None
            self._last_motion_state = AICameraMotionState.UNKNOWN
            self._last_motion_confidence = None

    def _clear_pose_hint_cache(self) -> None:
        with self._state_lock():
            self._last_pose_hint_keypoints = {}
            self._last_pose_hint_confidence = None
            self._last_pose_hint_monotonic = None
            self._last_pose_hint_box_metrics = None

    def _store_pose_hint_cache(
        self,
        *,
        sparse_keypoints: dict[int, tuple[float, float, float]],
        pose_confidence: float | None,
        observed_monotonic: float,
        primary_person_box: Any,
    ) -> None:
        with self._state_lock():
            self._last_pose_hint_keypoints = dict(sparse_keypoints)
            self._last_pose_hint_confidence = pose_confidence
            self._last_pose_hint_monotonic = observed_monotonic
            self._last_pose_hint_box_metrics = self._extract_box_association_metrics(primary_person_box)

    def _should_reuse_pose_cache(self, *, observed_monotonic: float, primary_person_box: Any) -> bool:
        refresh_s = self._pose_refresh_s()
        if refresh_s <= 0.0:
            return False
        current_metrics = self._extract_box_association_metrics(primary_person_box)
        if current_metrics is None:
            return False
        with self._state_lock():
            last_result = getattr(self, "_last_pose_result", None)
            last_monotonic = getattr(self, "_last_pose_monotonic", None)
            last_metrics = getattr(self, "_last_pose_box_metrics", None)
        if last_result is None or last_monotonic is None or last_metrics is None:
            return False
        age_s = observed_monotonic - last_monotonic
        return 0.0 <= age_s <= refresh_s and self._box_metrics_similar(last_metrics, current_metrics)

    def _should_reuse_pose_hint_cache(self, *, observed_monotonic: float, primary_person_box: Any) -> bool:
        refresh_s = self._pose_refresh_s()
        if refresh_s <= 0.0:
            return False
        current_metrics = self._extract_box_association_metrics(primary_person_box)
        if current_metrics is None:
            return False
        with self._state_lock():
            keypoints = getattr(self, "_last_pose_hint_keypoints", {}) or {}
            last_monotonic = getattr(self, "_last_pose_hint_monotonic", None)
            last_metrics = getattr(self, "_last_pose_hint_box_metrics", None)
        if not keypoints or last_monotonic is None or last_metrics is None:
            return False
        age_s = observed_monotonic - last_monotonic
        return 0.0 <= age_s <= refresh_s and self._box_metrics_similar(last_metrics, current_metrics)

    def _box_center_x(self, box: Any) -> float | None:
        metrics = self._extract_box_metrics(box)
        return None if metrics is None else self._coerce_finite_float(metrics.get("center_x"))

    def _attribute_float(self, obj: Any, attr: str) -> float | None:
        return self._lookup_float(obj, attr)

    def _extract_box_metrics(self, box: Any, *, _depth: int = 0) -> dict[str, float] | None:
        """Return the legacy public metric shape used by the pre-refactor adapter."""

        metrics = self._extract_box_association_metrics(box, _depth=_depth)
        if metrics is None:
            return None
        return {
            "center_x": float(metrics["center_x"]),
            "center_y": float(metrics["center_y"]),
            "width": float(metrics["width"]),
            "height": float(metrics["height"]),
        }

    def _extract_box_association_metrics(self, box: Any, *, _depth: int = 0) -> dict[str, Any] | None:
        if box is None:
            return None
        track_id = self._extract_track_id(box)
        cx = self._lookup_float(box, "center_x", "cx")
        cy = self._lookup_float(box, "center_y", "cy")
        w = self._lookup_float(box, "width")
        h = self._lookup_float(box, "height")
        if cx is not None and cy is not None and w is not None and h is not None:
            return self._metrics_from_cxcywh(cx, cy, w, h, track_id=track_id)
        left = self._lookup_float(box, "left", "x_min", "xmin", "x1")
        top = self._lookup_float(box, "top", "y_min", "ymin", "y1")
        right = self._lookup_float(box, "right", "x_max", "xmax", "x2")
        bottom = self._lookup_float(box, "bottom", "y_max", "ymax", "y2")
        if left is not None and top is not None and right is not None and bottom is not None:
            return self._metrics_from_ltrb(left, top, right, bottom, track_id=track_id)
        x = self._lookup_float(box, "x")
        y = self._lookup_float(box, "y")
        w = self._lookup_float(box, "w")
        h = self._lookup_float(box, "h")
        if x is not None and y is not None and w is not None and h is not None:
            return self._metrics_from_xywh(x, y, w, h, track_id=track_id)
        coords = self._coerce_box_sequence(box)
        if coords is not None:
            return self._metrics_from_xywh(*coords, track_id=track_id)
        if _depth == 0:
            nested = self._lookup_value(box, "box")
            if nested is _MISSING:
                nested = self._lookup_value(box, "bbox")
            if nested is not _MISSING and nested is not box:
                nested_metrics = self._extract_box_association_metrics(nested, _depth=1)
                if nested_metrics is not None and track_id is not None and nested_metrics.get("track_id") is None:
                    nested_metrics["track_id"] = track_id
                return nested_metrics
        return None

    def _box_metrics_similar(self, previous: dict[str, Any], current: dict[str, Any]) -> bool:
        prev_track = previous.get("track_id")
        curr_track = current.get("track_id")
        track_match = prev_track is not None and curr_track is not None and prev_track == curr_track
        if prev_track is not None and curr_track is not None and not track_match:
            return False
        prev_area = self._coerce_finite_float(previous.get("area"))
        curr_area = self._coerce_finite_float(current.get("area"))
        prev_w = self._coerce_finite_float(previous.get("width"))
        prev_h = self._coerce_finite_float(previous.get("height"))
        curr_w = self._coerce_finite_float(current.get("width"))
        curr_h = self._coerce_finite_float(current.get("height"))
        prev_cx = self._coerce_finite_float(previous.get("center_x"))
        prev_cy = self._coerce_finite_float(previous.get("center_y"))
        curr_cx = self._coerce_finite_float(current.get("center_x"))
        curr_cy = self._coerce_finite_float(current.get("center_y"))
        required = (prev_area, curr_area, prev_w, prev_h, curr_w, curr_h, prev_cx, prev_cy, curr_cx, curr_cy)
        if any(value is None for value in required):
            return False
        max_scale = max(prev_w, prev_h, curr_w, curr_h)
        if max_scale <= _MIN_BOX_EPSILON:
            return False
        rel_center = math.hypot(prev_cx - curr_cx, prev_cy - curr_cy) / max_scale
        larger_area = max(prev_area, curr_area)
        if larger_area <= _MIN_BOX_EPSILON:
            return False
        area_ratio = min(prev_area, curr_area) / larger_area
        prev_aspect = prev_w / max(prev_h, _MIN_BOX_EPSILON)
        curr_aspect = curr_w / max(curr_h, _MIN_BOX_EPSILON)
        aspect_ratio = min(prev_aspect, curr_aspect) / max(prev_aspect, curr_aspect)
        iou = self._box_metrics_iou(previous, current)
        if track_match:
            return area_ratio >= 0.25 and aspect_ratio >= 0.35 and (iou >= 0.10 or rel_center <= 0.75)
        return area_ratio >= 0.45 and aspect_ratio >= 0.50 and (
            iou >= 0.20 or (iou >= 0.10 and rel_center <= 0.35) or (rel_center <= 0.20 and area_ratio >= 0.70)
        )

    def _box_metrics_iou(self, first: dict[str, Any], second: dict[str, Any]) -> float:
        left = max(float(first["left"]), float(second["left"]))
        top = max(float(first["top"]), float(second["top"]))
        right = min(float(first["right"]), float(second["right"]))
        bottom = min(float(first["bottom"]), float(second["bottom"]))
        inter_w = max(0.0, right - left)
        inter_h = max(0.0, bottom - top)
        intersection = inter_w * inter_h
        union = max(float(first["area"]), 0.0) + max(float(second["area"]), 0.0) - intersection
        return 0.0 if union <= _MIN_BOX_EPSILON else intersection / union

    def _metrics_from_cxcywh(
        self, center_x: float, center_y: float, width: float, height: float, *, track_id: str | int | None
    ) -> dict[str, Any] | None:
        width, height = abs(width), abs(height)
        if width <= _MIN_BOX_EPSILON or height <= _MIN_BOX_EPSILON:
            return None
        half_w, half_h = width / 2.0, height / 2.0
        return {
            "center_x": center_x, "center_y": center_y, "width": width, "height": height,
            "left": center_x - half_w, "top": center_y - half_h, "right": center_x + half_w, "bottom": center_y + half_h,
            "area": width * height, "track_id": track_id,
        }

    def _metrics_from_xywh(
        self, x: float, y: float, width: float, height: float, *, track_id: str | int | None
    ) -> dict[str, Any] | None:
        return self._metrics_from_ltrb(min(x, x + width), min(y, y + height), max(x, x + width), max(y, y + height), track_id=track_id)

    def _metrics_from_ltrb(
        self, left: float, top: float, right: float, bottom: float, *, track_id: str | int | None
    ) -> dict[str, Any] | None:
        left, right = sorted((left, right))
        top, bottom = sorted((top, bottom))
        width, height = right - left, bottom - top
        if width <= _MIN_BOX_EPSILON or height <= _MIN_BOX_EPSILON:
            return None
        return {
            "center_x": left + (width / 2.0), "center_y": top + (height / 2.0),
            "width": width, "height": height, "left": left, "top": top, "right": right, "bottom": bottom,
            "area": width * height, "track_id": track_id,
        }

    def _lookup_value(self, obj: Any, key: str) -> Any:
        if obj is None:
            return _MISSING
        if isinstance(obj, Mapping):
            return obj.get(key, _MISSING)
        try:
            return getattr(obj, key)
        except Exception:
            return _MISSING

    def _lookup_float(self, obj: Any, *keys: str) -> float | None:
        for key in keys:
            value = self._lookup_value(obj, key)
            if value is _MISSING:
                continue
            value = self._coerce_finite_float(value)
            if value is not None:
                return value
        return None

    def _extract_track_id(self, box: Any) -> str | int | None:
        for key in ("track_id", "tracking_id", "detection_id", "person_id", "id"):
            value = self._lookup_value(box, key)
            if value is _MISSING:
                continue
            track_id = self._coerce_track_id(value)
            if track_id is not None:
                return track_id
        return None

    def _coerce_track_id(self, value: Any) -> str | int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value) if math.isfinite(value) and value.is_integer() else None
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return None

    def _coerce_box_sequence(self, box: Any) -> tuple[float, float, float, float] | None:
        if box is None or isinstance(box, (str, bytes, bytearray)) or isinstance(box, Mapping):
            return None
        try:
            if len(box) != 4:
                return None
        except TypeError:
            return None
        values: list[float] = []
        for idx in range(4):
            try:
                item = box[idx]
            except Exception:
                return None
            item = self._coerce_finite_float(item)
            if item is None:
                return None
            values.append(item)
        return values[0], values[1], values[2], values[3]

    def _is_empty_result(self, value: Any) -> bool:
        if value is None:
            return True
        size = self._lookup_value(value, "size")
        if size is not _MISSING:
            size = self._coerce_finite_float(size)
            if size is not None and size == 0.0:
                return True
        shape = self._lookup_value(value, "shape")
        if shape is not _MISSING:
            try:
                if any(int(dim) == 0 for dim in shape):
                    return True
            except Exception:
                pass
        if isinstance(value, memoryview):
            return value.nbytes == 0
        if isinstance(value, (bytes, bytearray, str)):
            return len(value) == 0
        try:
            return len(value) == 0
        except TypeError:
            return False

    def _coerce_finite_float(self, value: Any) -> float | None:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def _bounded_nonnegative_float(self, value: Any, *, default: float, maximum: float | None = None) -> float:
        value = self._coerce_finite_float(value)
        if value is None or value < 0.0:
            return default
        return min(value, maximum) if maximum is not None else value

    def _lock_timeout_s(self) -> float:
        # BREAKING: lock_timeout_s is hard-clamped to 10 seconds to avoid indefinite pipeline stalls on bad config.
        config = getattr(self, "config", None)
        return self._bounded_nonnegative_float(getattr(config, "lock_timeout_s", None), default=1.0, maximum=_MAX_LOCK_TIMEOUT_S)

    def _pose_refresh_s(self) -> float:
        # BREAKING: pose_refresh_s is hard-clamped to 5 seconds to stop stale human-state reuse.
        config = getattr(self, "config", None)
        return self._bounded_nonnegative_float(getattr(config, "pose_refresh_s", None), default=0.0, maximum=_MAX_POSE_REFRESH_S)

    def _classify_error(self, exc: Exception) -> str:
        message = " | ".join(self._iter_error_messages(exc))
        for code, patterns in _ERROR_PATTERNS:
            if any(pattern in message for pattern in patterns):
                return code
        if "timestamp" in message and "monotonic" in message:
            return "pose_timestamp_invalid"
        if isinstance(exc, FileNotFoundError):
            return "model_missing"
        if isinstance(exc, TimeoutError):
            return "timeout"
        return exc.__class__.__name__.lower()

    def _iter_error_messages(self, exc: BaseException) -> tuple[str, ...]:
        messages: list[str] = []
        seen: set[int] = set()
        pending: list[BaseException] = [exc]
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            text = " ".join(str(current).strip().lower().split())
            if text:
                messages.append(text)
            if _BASE_EXCEPTION_GROUP_TYPE is not None and isinstance(current, _BASE_EXCEPTION_GROUP_TYPE):
                pending.extend(child for child in reversed(current.exceptions) if isinstance(child, BaseException))
            for next_error in (current.__cause__, current.__context__):
                if isinstance(next_error, BaseException):
                    pending.append(next_error)
        return tuple(messages)

    def _normalize_clock_value_to_seconds(self, value: Any, *, monotonic: bool) -> float | None:
        value = self._coerce_finite_float(value)
        if value is None or value < 0.0:
            return None
        if monotonic:
            if value >= 1e14:
                return value / 1_000_000_000.0
            if value >= 1e11:
                return value / 1_000_000.0
            if value >= 1e8:
                return value / 1_000.0
            return value
        if value >= 1e17:
            return value / 1_000_000_000.0
        if value >= 1e14:
            return value / 1_000_000.0
        if value >= 1e11:
            return value / 1_000.0
        return value

    def _now(self) -> float:
        try:
            value = self._normalize_clock_value_to_seconds(self._clock(), monotonic=False)
        except Exception:
            value = None
        return time.time() if value is None else value

    def _monotonic_now(self) -> float:
        try:
            value = self._normalize_clock_value_to_seconds(self._monotonic_clock(), monotonic=True)
        except Exception:
            value = None
        return time.monotonic() if value is None else value
