# CHANGELOG: 2026-03-28
# BUG-1: Guard motion-state history against concurrent and out-of-order updates; preserve last good motion output on stale frames.
# BUG-2: Normalize foreign detection payloads robustly, including dict-like stubs and malformed scalar types, instead of silently dropping fields.
# BUG-3: Fix showing_intent_likely so positive intent evidence is not lost when attention score metadata is absent.
# SEC-1: Throttle repeated per-frame warning logs to prevent disk-filling failure storms on always-on Raspberry Pi deployments.
# IMP-1: Add lightweight temporal smoothing and hysteresis for single-person live-stream observations, aligned with 2026 edge-CV practice.
# IMP-2: Sanitize bounded probabilities/tri-state booleans and preserve richer model provenance for downstream telemetry and debugging.

# mypy: disable-error-code="attr-defined,has-type"
"""Observation composition, health, and face-anchor helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import is_dataclass, replace
from typing import Any
import math
import threading
import time

from ..detection import DetectionResult
from ..face_anchors import SupplementalFaceAnchorResult, merge_detection_with_face_anchors
from ..looking_signal import infer_fast_looking_signal
from ..models import (
    AICameraBodyPose,
    AICameraFineHandGesture,
    AICameraGestureEvent,
    AICameraMotionState,
    AICameraObservation,
    AICameraZone,
)
from ..motion import infer_motion_state
from .common import LOGGER
from .types import PoseResult

logger = LOGGER

_OBSERVATION_FALLBACK_FIELDS: tuple[str, ...] = (
    "observed_at",
    "camera_online",
    "camera_ready",
    "camera_ai_ready",
    "camera_error",
    "last_camera_frame_at",
    "last_camera_health_change_at",
    "person_count",
    "primary_person_box",
    "primary_person_zone",
    "visible_persons",
    "looking_toward_device",
    "looking_signal_state",
    "looking_signal_source",
    "person_near_device",
    "engaged_with_device",
    "visual_attention_score",
    "body_pose",
    "pose_confidence",
    "motion_state",
    "motion_confidence",
    "hand_or_object_near_camera",
    "showing_intent_likely",
    "gesture_event",
    "gesture_confidence",
    "fine_hand_gesture",
    "fine_hand_gesture_confidence",
    "objects",
    "model",
)


class AICameraAdapterCompositionMixin:
    """Compose stable observation payloads from detection, pose, and health state."""

    def _ensure_health_state(self) -> None:
        """Lazily initialize health state used by this mixin."""

        if getattr(self, "_health_lock", None) is None:
            self._health_lock = threading.RLock()
        if not hasattr(self, "_last_frame_at"):
            self._last_frame_at = None
        if not hasattr(self, "_last_health_signature"):
            self._last_health_signature = None
        if not hasattr(self, "_last_health_change_at"):
            self._last_health_change_at = None

    def _ensure_motion_state(self) -> None:
        """Lazily initialize motion state used by this mixin."""

        if getattr(self, "_motion_lock", None) is None:
            self._motion_lock = threading.RLock()
        if not hasattr(self, "_last_motion_box"):
            self._last_motion_box = None
        if not hasattr(self, "_last_motion_person_count"):
            self._last_motion_person_count = 0
        if not hasattr(self, "_last_motion_at"):
            self._last_motion_at = None
        if not hasattr(self, "_last_motion_monotonic"):
            self._last_motion_monotonic = None
        if not hasattr(self, "_last_motion_state"):
            self._last_motion_state = AICameraMotionState.UNKNOWN
        if not hasattr(self, "_last_motion_confidence"):
            self._last_motion_confidence = None

    def _ensure_observation_filter_state(self) -> None:
        """Lazily initialize observation-level temporal filter state."""

        if getattr(self, "_observation_filter_lock", None) is None:
            self._observation_filter_lock = threading.RLock()
        if not hasattr(self, "_observation_filter_state"):
            self._observation_filter_state = {}

    def _ensure_log_throttle_state(self) -> None:
        """Lazily initialize throttled logging state."""

        if getattr(self, "_log_throttle_lock", None) is None:
            self._log_throttle_lock = threading.RLock()
        if not hasattr(self, "_log_throttle_last_emit_at"):
            self._log_throttle_last_emit_at = {}

    def _finite_float(
        self,
        value: Any,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float | None:
        """Return one finite float optionally clamped to a range."""

        if value is None:
            return None
        try:
            coerced = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(coerced):
            return None
        if minimum is not None and coerced < minimum:
            coerced = minimum
        if maximum is not None and coerced > maximum:
            coerced = maximum
        return coerced

    def _coerce_probability(self, value: Any) -> float | None:
        """Return one finite probability in the closed interval [0, 1]."""

        return self._finite_float(value, minimum=0.0, maximum=1.0)

    def _coerce_non_negative_int(self, value: Any, *, default: int = 0) -> int:
        """Return one bounded non-negative integer."""

        if value is None:
            return default
        try:
            coerced = int(value)
        except (TypeError, ValueError):
            return default
        return max(0, coerced)

    def _coerce_optional_bool(self, value: Any) -> bool | None:
        """Return one tri-state boolean while tolerating common scalar encodings."""

        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "t", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "f", "no", "n", "off"}:
                return False
            return None
        if isinstance(value, (int, float)):
            finite_value = self._finite_float(value)
            if finite_value is None:
                return None
            if finite_value == 0.0:
                return False
            if finite_value == 1.0:
                return True
            return bool(finite_value)
        try:
            return bool(value)
        except Exception:
            return None

    def _coerce_bool(self, value: Any, *, default: bool = False) -> bool:
        """Return one strict boolean."""

        coerced = self._coerce_optional_bool(value)
        if coerced is None:
            return default
        return coerced

    def _coerce_zone(self, value: Any) -> AICameraZone:
        """Normalize foreign zone values into the stable enum."""

        if isinstance(value, AICameraZone):
            return value
        if isinstance(value, str):
            try:
                return AICameraZone[value.upper()]
            except Exception:
                try:
                    return AICameraZone(value)
                except Exception:
                    return AICameraZone.UNKNOWN
        return AICameraZone.UNKNOWN

    def _coerce_tupleish(self, value: Any) -> tuple[Any, ...]:
        """Return one tuple for tuple-like foreign payloads."""

        if value is None:
            return ()
        if isinstance(value, tuple):
            return value
        if isinstance(value, str):
            return (value,)
        try:
            return tuple(value)
        except TypeError:
            return (value,)

    def _config_float(
        self,
        name: str,
        default: float,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        """Read one optional float configuration value with bounds."""

        coerced = self._finite_float(getattr(self.config, name, default), minimum=minimum, maximum=maximum)
        if coerced is None:
            return default
        return coerced

    def _coerce_monotonic(self, value: Any) -> float:
        """Return one finite monotonic-ish timestamp or fall back to time.monotonic()."""

        observed_monotonic = self._finite_float(value)
        if observed_monotonic is None:
            return time.monotonic()
        return observed_monotonic

    def _read_attr_or_key(self, value: Any, name: str, default: Any = None) -> Any:
        """Read one attribute or mapping key from a foreign payload."""

        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    def _log_throttled(
        self,
        *,
        level: str,
        key: str,
        message: str,
        interval_s: float | None = None,
        exc_info: bool = False,
    ) -> None:
        """Emit one log message at most once per interval for a given key."""

        self._ensure_log_throttle_state()
        throttle_interval = self._config_float(
            "error_log_throttle_seconds",
            30.0 if interval_s is None else interval_s,
            minimum=0.0,
            maximum=3600.0,
        )
        now_monotonic = time.monotonic()
        throttle_key = (level, key)
        with self._log_throttle_lock:
            last_emit_at = self._log_throttle_last_emit_at.get(throttle_key)
            if last_emit_at is not None and (now_monotonic - last_emit_at) < throttle_interval:
                return
            self._log_throttle_last_emit_at[throttle_key] = now_monotonic
        log_fn = getattr(logger, level, logger.warning)
        log_fn(message, exc_info=exc_info)

    def _observation_copy(self, observation: AICameraObservation, **updates: Any) -> AICameraObservation:
        """Copy one observation while preserving unknown future fields whenever possible."""

        if hasattr(observation, "model_copy"):
            try:
                return observation.model_copy(update=updates)
            except Exception:
                pass
        if hasattr(observation, "copy"):
            try:
                return observation.copy(update=updates)
            except Exception:
                pass
        if is_dataclass(observation):
            try:
                return replace(observation, **updates)
            except Exception:
                pass
        if hasattr(observation, "model_dump"):
            try:
                data = observation.model_dump()
                data.update(updates)
                return AICameraObservation(**data)
            except Exception:
                pass
        if hasattr(observation, "dict"):
            try:
                data = observation.dict()
                data.update(updates)
                return AICameraObservation(**data)
            except Exception:
                pass
        data = {name: getattr(observation, name, None) for name in _OBSERVATION_FALLBACK_FIELDS}
        data.update(updates)
        return AICameraObservation(**data)

    def _compose_model_name(self, *, face_anchors: SupplementalFaceAnchorResult | None) -> str:
        """Return one stable model provenance string for this observation."""

        parts = ["local-imx500"]
        if str(getattr(self.config, "pose_backend", "") or "").strip().lower() == "mediapipe":
            parts.append("mediapipe")
        face_anchor_state = getattr(face_anchors, "state", None)
        if face_anchor_state not in {None, "disabled", "skipped", "backend_unavailable"}:
            parts.append("face-anchors")
        return "+".join(parts)

    def _smooth_probability_state(
        self,
        state: dict[str, Any],
        *,
        name: str,
        current: float | None,
        observed_monotonic: float,
        alpha: float,
        stale_after_s: float,
    ) -> float | None:
        """Apply one light EMA to a bounded score and hold through brief dropouts."""

        last_value_at = self._finite_float(state.get(f"{name}_value_at"))
        previous = self._coerce_probability(state.get(name))
        if current is None:
            if previous is not None and last_value_at is not None and 0.0 <= (observed_monotonic - last_value_at) <= stale_after_s:
                return previous
            state[name] = None
            state.pop(f"{name}_value_at", None)
            return None
        if previous is None or last_value_at is None or observed_monotonic < last_value_at or (observed_monotonic - last_value_at) > stale_after_s:
            smoothed = current
        else:
            smoothed = (alpha * current) + ((1.0 - alpha) * previous)
        smoothed = self._coerce_probability(smoothed)
        state[name] = smoothed
        state[f"{name}_value_at"] = observed_monotonic
        return smoothed

    def _smooth_boolean_state(
        self,
        state: dict[str, Any],
        *,
        name: str,
        current: bool | None,
        observed_monotonic: float,
        hold_seconds: float,
        stale_after_s: float,
    ) -> tuple[bool | None, bool]:
        """Apply one small hysteresis window to a tri-state boolean.

        Returns the filtered value and whether the previous value was held.
        """

        previous = self._coerce_optional_bool(state.get(name))
        last_value_at = self._finite_float(state.get(f"{name}_value_at"))
        if previous is None or last_value_at is None or observed_monotonic < last_value_at or (observed_monotonic - last_value_at) > stale_after_s:
            state[name] = current
            if current is None:
                state.pop(f"{name}_value_at", None)
            else:
                state[f"{name}_value_at"] = observed_monotonic
            return current, False

        held_previous = False
        value = current

        if current is None:
            value = previous
            held_previous = previous is not None
        elif previous is True and current is not True and (observed_monotonic - last_value_at) <= hold_seconds:
            value = True
            held_previous = True

        state[name] = value
        if current is None and value is None:
            state.pop(f"{name}_value_at", None)
        elif not held_previous and value is not None:
            state[f"{name}_value_at"] = observed_monotonic
        return value, held_previous

    def _apply_temporal_observation_filter(
        self,
        *,
        observed_monotonic: float,
        person_count: int,
        visual_attention: float | None,
        looking_toward_device: bool | None,
        looking_signal_state: str | None,
        looking_signal_source: str | None,
        engaged_with_device: bool | None,
        showing_intent_likely: bool | None,
    ) -> tuple[float | None, bool | None, str | None, str | None, bool | None, bool | None]:
        """Apply low-cost temporal filtering to noisy single-subject live-stream outputs."""

        self._ensure_observation_filter_state()
        if person_count >= 2:
            with self._observation_filter_lock:
                self._observation_filter_state.clear()
            return (
                visual_attention,
                looking_toward_device,
                looking_signal_state,
                looking_signal_source,
                engaged_with_device,
                showing_intent_likely,
            )

        attention_alpha = self._config_float("observation_attention_alpha", 0.65, minimum=0.0, maximum=1.0)
        hold_seconds = self._config_float("observation_signal_hold_seconds", 0.20, minimum=0.0, maximum=5.0)
        stale_after_s = self._config_float("observation_state_stale_after_seconds", 1.50, minimum=0.0, maximum=60.0)

        with self._observation_filter_lock:
            state: dict[str, Any] = self._observation_filter_state
            last_observed_monotonic = self._finite_float(state.get("last_observed_monotonic"))
            if last_observed_monotonic is not None and observed_monotonic < last_observed_monotonic:
                return (
                    visual_attention,
                    looking_toward_device,
                    looking_signal_state,
                    looking_signal_source,
                    engaged_with_device,
                    showing_intent_likely,
                )

            visual_attention = self._smooth_probability_state(
                state,
                name="visual_attention",
                current=visual_attention,
                observed_monotonic=observed_monotonic,
                alpha=attention_alpha,
                stale_after_s=stale_after_s,
            )

            looking_toward_device, held_looking = self._smooth_boolean_state(
                state,
                name="looking_toward_device",
                current=looking_toward_device,
                observed_monotonic=observed_monotonic,
                hold_seconds=hold_seconds,
                stale_after_s=stale_after_s,
            )
            if looking_signal_state is not None or looking_signal_source is not None:
                state["looking_signal_state"] = looking_signal_state
                state["looking_signal_source"] = looking_signal_source
            elif held_looking:
                looking_signal_state = state.get("looking_signal_state")
                looking_signal_source = state.get("looking_signal_source")

            engaged_with_device, _ = self._smooth_boolean_state(
                state,
                name="engaged_with_device",
                current=engaged_with_device,
                observed_monotonic=observed_monotonic,
                hold_seconds=hold_seconds,
                stale_after_s=stale_after_s,
            )
            showing_intent_likely, _ = self._smooth_boolean_state(
                state,
                name="showing_intent_likely",
                current=showing_intent_likely,
                observed_monotonic=observed_monotonic,
                hold_seconds=hold_seconds,
                stale_after_s=stale_after_s,
            )
            state["last_observed_monotonic"] = observed_monotonic

        return (
            visual_attention,
            looking_toward_device,
            looking_signal_state,
            looking_signal_source,
            engaged_with_device,
            showing_intent_likely,
        )

    def _coerce_observed_at(self, value: float | None) -> float:
        """Return one finite observation timestamp or fall back to the local clock."""

        observed_at = self._finite_float(value)
        if observed_at is None:
            return self._now()
        return observed_at

    def _compose_observation(
        self,
        *,
        observed_at: float,
        observed_monotonic: float,
        detection: DetectionResult,
        pose: PoseResult | None,
        pose_error: str | None,
        face_anchors: SupplementalFaceAnchorResult | None = None,
    ) -> AICameraObservation:
        """Merge detection and pose signals into one bounded observation."""

        detection = self._coerce_detection_result(detection)
        observed_at = self._coerce_observed_at(observed_at)
        observed_monotonic = self._coerce_monotonic(observed_monotonic)

        primary_person_box = detection.primary_person_box
        primary_person_zone = self._coerce_zone(detection.primary_person_zone)
        person_near_device = self._coerce_optional_bool(detection.person_near_device)
        hand_or_object_near_camera = self._coerce_bool(detection.hand_or_object_near_camera, default=False)

        attention_score_threshold = self._config_float("attention_score_threshold", 0.5, minimum=0.0, maximum=1.0)
        engaged_score_threshold = self._config_float("engaged_score_threshold", 0.5, minimum=0.0, maximum=1.0)

        visual_attention: float | None = None
        looking_toward_device: bool | None = None
        looking_signal_state: str | None = None
        looking_signal_source: str | None = None
        try:
            fast_looking_signal = infer_fast_looking_signal(
                detection=detection,
                face_anchors=face_anchors,
                attention_score_threshold=attention_score_threshold,
            )
            visual_attention = self._coerce_probability(getattr(fast_looking_signal, "visual_attention_score", None))
            looking_toward_device = self._coerce_optional_bool(getattr(fast_looking_signal, "looking_toward_device", None))
            looking_signal_source = getattr(fast_looking_signal, "source", None)
            looking_signal_state = getattr(fast_looking_signal, "state", None) if looking_signal_source is not None else None
        except Exception:
            self._log_throttled(
                level="warning",
                key="looking_signal_inference_failed",
                message="Local AI camera attention fusion failed; continuing with degraded observation composition.",
            )
            self._log_throttled(
                level="debug",
                key="looking_signal_inference_failed_debug",
                message="Local AI camera attention fusion exception details.",
                exc_info=True,
            )

        engaged_with_device: bool | None = None
        body_pose = AICameraBodyPose.UNKNOWN
        pose_confidence = None
        motion_state = AICameraMotionState.UNKNOWN
        motion_confidence = None
        gesture_event = AICameraGestureEvent.UNKNOWN if pose_error is not None else AICameraGestureEvent.NONE
        gesture_confidence = None
        fine_hand_gesture = AICameraFineHandGesture.UNKNOWN if pose_error is not None else AICameraFineHandGesture.NONE
        fine_hand_gesture_confidence = None
        showing_intent_likely: bool | None = None

        if visual_attention is not None or person_near_device is not None:
            engaged_with_device = person_near_device is True and (visual_attention or 0.0) >= engaged_score_threshold
        if hand_or_object_near_camera and (looking_toward_device is True or person_near_device is True):
            showing_intent_likely = True

        if pose is not None:
            pose_visual_attention = self._coerce_probability(getattr(pose, "visual_attention_score", None))
            if pose_visual_attention is not None:
                if visual_attention is None or pose_visual_attention >= visual_attention:
                    visual_attention = pose_visual_attention  # AUDIT-FIX(#7): Preserve higher-confidence pose attention when available.

            pose_looking_toward_device = self._coerce_optional_bool(getattr(pose, "looking_toward_device", None))
            if pose_looking_toward_device is True:
                looking_toward_device = True
                looking_signal_state = "confirmed"
                looking_signal_source = "pose_attention"
            elif pose_looking_toward_device is False and looking_toward_device is not True:
                looking_toward_device = False
                looking_signal_state = "inactive"
                if looking_signal_source is None:
                    looking_signal_source = "pose_attention"

            engaged_with_device = (
                detection.person_count > 0
                and (person_near_device is True or looking_toward_device is True)
                and (visual_attention or 0.0) >= engaged_score_threshold
            )

            body_pose = getattr(pose, "body_pose", AICameraBodyPose.UNKNOWN)
            pose_confidence = self._coerce_probability(getattr(pose, "pose_confidence", None))
            gesture_event = getattr(pose, "gesture_event", gesture_event)
            gesture_confidence = self._coerce_probability(getattr(pose, "gesture_confidence", None))
            fine_hand_gesture = getattr(pose, "fine_hand_gesture", fine_hand_gesture)
            fine_hand_gesture_confidence = self._coerce_probability(getattr(pose, "fine_hand_gesture_confidence", None))
            hand_or_object_near_camera = hand_or_object_near_camera or self._coerce_bool(
                getattr(pose, "hand_near_camera", False),
                default=False,
            )
            pose_showing_intent_likely = self._coerce_optional_bool(getattr(pose, "showing_intent_likely", None))
            if pose_showing_intent_likely is not None:
                showing_intent_likely = pose_showing_intent_likely  # AUDIT-FIX(#7): Keep fallback intent heuristics when pose-specific intent is unavailable.
            elif hand_or_object_near_camera and (looking_toward_device is True or person_near_device is True):
                showing_intent_likely = True

        motion_state, motion_confidence = self._resolve_motion(
            observed_at=observed_at,
            observed_monotonic=observed_monotonic,
            person_count=detection.person_count,
            primary_person_box=primary_person_box,
        )

        (
            visual_attention,
            looking_toward_device,
            looking_signal_state,
            looking_signal_source,
            engaged_with_device,
            showing_intent_likely,
        ) = self._apply_temporal_observation_filter(
            observed_monotonic=observed_monotonic,
            person_count=detection.person_count,
            visual_attention=visual_attention,
            looking_toward_device=looking_toward_device,
            looking_signal_state=looking_signal_state,
            looking_signal_source=looking_signal_source,
            engaged_with_device=engaged_with_device,
            showing_intent_likely=showing_intent_likely,
        )

        return AICameraObservation(
            observed_at=observed_at,
            camera_online=True,
            camera_ready=True,
            camera_ai_ready=True,
            camera_error=pose_error,
            person_count=detection.person_count,
            primary_person_box=primary_person_box,
            primary_person_zone=primary_person_zone,
            visible_persons=detection.visible_persons,
            looking_toward_device=looking_toward_device,
            looking_signal_state=looking_signal_state,
            looking_signal_source=looking_signal_source,
            person_near_device=person_near_device,
            engaged_with_device=engaged_with_device,
            visual_attention_score=visual_attention,
            body_pose=body_pose,
            pose_confidence=pose_confidence,
            motion_state=motion_state,
            motion_confidence=motion_confidence,
            hand_or_object_near_camera=hand_or_object_near_camera,
            showing_intent_likely=showing_intent_likely,
            gesture_event=gesture_event,
            gesture_confidence=gesture_confidence,
            fine_hand_gesture=fine_hand_gesture,
            fine_hand_gesture_confidence=fine_hand_gesture_confidence,
            objects=detection.objects,
            model=self._compose_model_name(face_anchors=face_anchors),
        )

    def _coerce_detection_result(self, detection: Any) -> DetectionResult:
        """Normalize foreign detection stubs into the stable detection contract."""

        if isinstance(detection, DetectionResult):
            return detection

        return DetectionResult(
            person_count=self._coerce_non_negative_int(self._read_attr_or_key(detection, "person_count", 0), default=0),
            primary_person_box=self._read_attr_or_key(detection, "primary_person_box", None),
            primary_person_zone=self._coerce_zone(
                self._read_attr_or_key(detection, "primary_person_zone", AICameraZone.UNKNOWN)
            ),
            visible_persons=self._coerce_tupleish(self._read_attr_or_key(detection, "visible_persons", ())),
            person_near_device=self._coerce_optional_bool(self._read_attr_or_key(detection, "person_near_device", None)),
            hand_or_object_near_camera=self._coerce_bool(
                self._read_attr_or_key(detection, "hand_or_object_near_camera", False),
                default=False,
            ),
            objects=self._coerce_tupleish(self._read_attr_or_key(detection, "objects", ())),
        )

    def _needs_rgb_frame_for_observation(self, *, detection: DetectionResult) -> bool:
        """Return whether this observation needs a preview RGB frame."""

        detection = self._coerce_detection_result(detection)
        if str(getattr(self.config, "pose_backend", "") or "").strip().lower() == "mediapipe" and detection.person_count > 0:
            return True
        return self._face_anchor_detector is not None and detection.person_count < 2

    def _needs_rgb_frame_for_attention(self, *, detection: DetectionResult) -> bool:
        """Return whether the cheap attention path needs one preview frame.

        Attention-only refresh uses RGB solely for optional face-anchor
        supplementation when SSD does not already see two people. It never
        promotes RGB capture to mandatory pose/gesture inference.
        """

        detection = self._coerce_detection_result(detection)
        return self._face_anchor_detector is not None and detection.person_count < 2

    def _capture_optional_rgb_frame(
        self,
        runtime: dict[str, Any],
        *,
        observed_at: float,
    ) -> tuple[Any | None, str | None]:
        """Capture one RGB frame without promoting preview failures to full camera failure."""

        try:
            return self._capture_rgb_frame(runtime, observed_at=observed_at), None
        except Exception as exc:
            code = self._classify_error(exc)
            self._log_throttled(
                level="warning",
                key=f"rgb_preview_capture_failed:{code}",
                message=f"Local AI camera RGB preview capture failed with {code}.",
            )
            self._log_throttled(
                level="debug",
                key=f"rgb_preview_capture_failed_debug:{code}",
                message="Local AI camera RGB preview exception details.",
                exc_info=True,
            )
            return None, code

    def _supplement_visible_persons_with_face_anchors(
        self,
        *,
        detection: DetectionResult,
        frame_rgb: Any | None,
    ) -> tuple[DetectionResult, SupplementalFaceAnchorResult]:
        """Add bounded supplemental face anchors and return the raw face result."""

        detection = self._coerce_detection_result(detection)

        if self._face_anchor_detector is None:
            return detection, SupplementalFaceAnchorResult(
                state="disabled",
                detail="face_anchor_detector_disabled",
            )
        if detection.person_count >= 2:
            return detection, SupplementalFaceAnchorResult(
                state="skipped",
                detail="multiple_people_already_detected",
            )
        if frame_rgb is None:
            return detection, SupplementalFaceAnchorResult(
                state="skipped",
                detail="face_anchor_frame_unavailable",
            )
        try:
            face_result = self._face_anchor_detector.detect(frame_rgb)
            return (
                merge_detection_with_face_anchors(
                    detection=detection,
                    face_anchors=face_result,
                ),
                face_result,
            )
        except Exception:
            self._log_throttled(
                level="warning",
                key="supplemental_face_anchor_detection_failed",
                message="Local AI camera supplemental face-anchor detection failed.",
            )
            self._log_throttled(
                level="debug",
                key="supplemental_face_anchor_detection_failed_debug",
                message="Local AI camera face-anchor exception details.",
                exc_info=True,
            )
            return detection, SupplementalFaceAnchorResult(
                state="backend_unavailable",
                detail="supplemental_face_anchor_detection_failed",
            )

    def _health_only_observation(
        self,
        *,
        observed_at: float,
        online: bool,
        ready: bool,
        ai_ready: bool,
        error: str | None,
    ) -> AICameraObservation:
        """Return one explicit health-only observation."""

        observation = AICameraObservation(
            observed_at=self._coerce_observed_at(observed_at),
            camera_online=online,
            camera_ready=ready,
            camera_ai_ready=ai_ready,
            camera_error=error,
        )
        return self._with_health(observation, online=online, ready=ready, ai_ready=ai_ready, error=error, frame_at=None)

    def _with_health(
        self,
        observation: AICameraObservation,
        *,
        online: bool,
        ready: bool,
        ai_ready: bool,
        error: str | None,
        frame_at: float | None,
    ) -> AICameraObservation:
        """Attach the latest health timestamps to one observation."""

        self._ensure_health_state()
        frame_at = self._coerce_observed_at(frame_at) if frame_at is not None else None

        with self._health_lock:  # AUDIT-FIX(#1): Serialize health/frame metadata updates across success and timeout paths.
            if frame_at is not None:
                self._last_frame_at = frame_at
            health_signature = (online, ready, ai_ready, error)
            if health_signature != self._last_health_signature:
                self._last_health_signature = health_signature
                self._last_health_change_at = observation.observed_at
            last_camera_frame_at = self._last_frame_at
            last_camera_health_change_at = self._last_health_change_at

        return self._observation_copy(
            observation,
            camera_online=online,
            camera_ready=ready,
            camera_ai_ready=ai_ready,
            camera_error=error,
            last_camera_frame_at=last_camera_frame_at,
            last_camera_health_change_at=last_camera_health_change_at,
        )

    def _resolve_motion(
        self,
        *,
        observed_at: float,
        observed_monotonic: float,
        person_count: int,
        primary_person_box: Any,
    ) -> tuple[AICameraMotionState, float | None]:
        """Derive one coarse motion state from recent primary-person box deltas."""

        self._ensure_motion_state()
        observed_at = self._coerce_observed_at(observed_at)
        observed_monotonic = self._coerce_monotonic(observed_monotonic)
        person_count = self._coerce_non_negative_int(person_count, default=0)

        with self._motion_lock:
            previous_observed_monotonic = self._finite_float(self._last_motion_monotonic)
            if previous_observed_monotonic is not None and observed_monotonic <= previous_observed_monotonic:
                return self._last_motion_state, self._coerce_probability(self._last_motion_confidence)

            try:
                motion_state, motion_confidence = infer_motion_state(
                    previous_box=self._last_motion_box,
                    current_box=primary_person_box,
                    previous_observed_at=previous_observed_monotonic,
                    current_observed_at=observed_monotonic,
                    previous_person_count=self._last_motion_person_count,
                    current_person_count=person_count,
                )
            except Exception:
                self._log_throttled(
                    level="warning",
                    key="motion_inference_failed",
                    message="Local AI camera motion inference failed; keeping the last known motion state.",
                )
                self._log_throttled(
                    level="debug",
                    key="motion_inference_failed_debug",
                    message="Local AI camera motion inference exception details.",
                    exc_info=True,
                )
                return self._last_motion_state, self._coerce_probability(self._last_motion_confidence)

            motion_confidence = self._coerce_probability(motion_confidence)
            self._last_motion_box = primary_person_box
            self._last_motion_person_count = person_count
            self._last_motion_at = observed_at
            self._last_motion_monotonic = observed_monotonic
            self._last_motion_state = motion_state
            self._last_motion_confidence = motion_confidence
            return motion_state, motion_confidence