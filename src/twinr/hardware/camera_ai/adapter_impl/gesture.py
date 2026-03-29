# CHANGELOG: 2026-03-28
# BUG-1: Stabilize live-pipeline timestamps with a monotonic wall-clock bridge so MediaPipe live/video APIs do not fail on backward wall-clock jumps.
# BUG-2: Preserve the legacy public gesture contract by keeping pose fallback limited to gesture-event recovery instead of silently changing fine-hand outputs.
# BUG-3: Use bounded, deep-copied debug snapshots to prevent aliasing and oversized/non-serializable debug payloads from corrupting operator diagnostics.
# SEC-1: Redact and bound exception/debug payloads before exposing them through get_last_gesture_debug_details(), while preserving local artifact paths needed for operator QA and tests.
# SEC-2: Bound debug payload cardinality and string size so hostile or buggy downstream debug producers cannot force large per-frame operator snapshots.
# IMP-1: Add short-lived gesture-target persistence to ride through IMX500/person-detector dropouts and MediaPipe live-stream frame skipping without losing the active user ROI.
# IMP-2: Expand workflow telemetry to expose the real target source values used by the adapter, including face-anchor promotion and recent-target cache reuse.
# IMP-3: Harden live-hand count/confidence extraction so malformed/partial runtime objects do not break observation emission.

# mypy: disable-error-code="attr-defined,arg-type,union-attr"
"""Dedicated live-gesture helpers for the local AI-camera adapter."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from twinr.agent.workflows.forensics import (
    current_workflow_forensics,
    current_workflow_trace_id,
    workflow_decision,
    workflow_event,
    workflow_span,
)

from ..detection import DetectionResult
from ..face_anchors import SupplementalFaceAnchorResult
from ..live_gesture_pipeline import LiveGestureObservePolicy
from ..models import (
    AICameraBox,
    AICameraFineHandGesture,
    AICameraGestureEvent,
    AICameraObservation,
    AICameraZone,
)
from .common import LOGGER
from .types import GesturePersonTargets, PoseResult

logger = LOGGER

_DEBUG_MAX_DEPTH = 4
_DEBUG_MAX_ITEMS = 80
_DEBUG_MAX_STRING = 240
_DEBUG_MAX_ERROR_STRING = 160
_GESTURE_TARGET_CACHE_TTL_S = 0.45
_GESTURE_PIPELINE_MIN_STEP_S = 0.001

_URL_RE = re.compile(r"https?://\S+")
_UNIX_PATH_RE = re.compile(r"(?<![\w.-])/(?:[^\s/]+/)*[^\s/]+")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


class AICameraAdapterGestureMixin:
    """Encapsulate the user-facing live gesture acknowledgement lane."""

    _LEGACY_PUBLIC_GESTURE_DEBUG_KEYS = (
        "stream_mode",
        "gesture_stream_source",
        "gesture_stream_temporal_enabled",
        "gesture_stream_temporal_reason",
        "gesture_stream_output_gesture",
        "gesture_stream_output_confidence",
        "gesture_stream_output_changed",
        "gesture_stream_resolved_source",
        "resolved_source",
        "forensics_active",
        "forensics_run_id",
        "forensics_trace_id",
        "forensics_zero_signal_capture_requested",
        "live_fine_hand_gesture",
        "live_fine_hand_gesture_confidence",
        "live_gesture_event",
        "live_gesture_confidence",
        "authoritative_gesture_active",
        "authoritative_gesture_key",
        "authoritative_gesture_token",
        "authoritative_gesture_rising",
        "authoritative_gesture_started_at",
        "authoritative_gesture_changed_at",
        "authoritative_gesture_source",
        "live_hand_count",
        "live_hand_count_exact",
        "effective_live_hand_box_count",
        "live_hand_box_source",
        "gesture_fast_path",
        "gesture_observe_policy",
        "pose_hint_source",
        "pose_hint_confidence",
        "pose_fallback_used",
        "pose_fallback_disabled_by_caller",
        "pose_fallback_error",
        "pose_fallback_fine_hand_gesture",
        "pose_fallback_fine_hand_gesture_confidence",
        "pose_fallback_gesture_event",
        "pose_fallback_gesture_confidence",
        "final_resolved_source",
        "detection_person_count",
        "detection_primary_person_zone",
        "detection_visible_person_count",
        "detection_primary_person_box_available",
        "gesture_target_source",
        "gesture_target_face_anchor_state",
        "gesture_target_face_anchor_count",
        "gesture_target_person_count",
        "gesture_target_primary_person_box_available",
        "person_roi_detection_count",
        "person_roi_block_reason",
        "person_roi_combined_gesture",
        "person_roi_combined_confidence",
        "full_frame_hand_attempt_reason",
        "candidate_capture_saved",
        "candidate_capture_reasons",
        "candidate_capture_image_path",
        "candidate_capture_metadata_path",
        "candidate_capture_skipped_reason",
        "candidate_capture_error",
    )

    def get_last_gesture_debug_details(self) -> dict[str, Any] | None:
        """Return the newest bounded gesture debug snapshot for operators."""

        if self._last_gesture_debug_details is None:
            return None
        snapshot = deepcopy(self._sanitize_debug_mapping(self._last_gesture_debug_details))
        if snapshot.get("pipeline_error") or snapshot.get("pipeline_error_message"):
            return snapshot
        filtered = {
            key: snapshot[key]
            for key in self._LEGACY_PUBLIC_GESTURE_DEBUG_KEYS
            if key in snapshot
        }
        if not filtered.get("forensics_active"):
            filtered["forensics_trace_id"] = None
        return filtered

    def _annotate_gesture_stream_debug_locked(self, *, source: str) -> None:
        """Overlay explicit stream-lane facts onto the current gesture debug snapshot."""

        current = dict(self._last_gesture_debug_details or {})
        current.update(
            {
                "stream_mode": "gesture_stream",
                "gesture_stream_source": source,
                "gesture_stream_temporal_enabled": bool(current.get("temporal_enabled")),
                "gesture_stream_temporal_reason": current.get("temporal_reason"),
                "gesture_stream_output_gesture": current.get("temporal_output_gesture"),
                "gesture_stream_output_confidence": current.get("temporal_output_confidence"),
                "gesture_stream_output_changed": bool(current.get("temporal_output_changed")),
                "gesture_stream_resolved_source": (
                    current.get("temporal_resolved_source")
                    or current.get("final_resolved_source")
                    or current.get("resolved_source")
                ),
            }
        )
        self._last_gesture_debug_details = current

    def _build_gesture_observation_locked(
        self,
        *,
        runtime: dict[str, Any],
        observed_at: float,
        observed_monotonic: float,
        detection: DetectionResult,
        frame_rgb: Any,
        frame_at: float | None,
        allow_pose_fallback: bool = True,
        gesture_fast_path: bool = False,
        stream_mode: bool = False,
        stream_source: str = "local_camera",
    ) -> AICameraObservation:
        """Run the dedicated gesture lane from one supplied detection/frame pair."""

        pipeline_observed_at = self._stable_gesture_pipeline_observed_at(
            observed_at=observed_at,
            observed_monotonic=observed_monotonic,
        )
        observe_policy = (
            LiveGestureObservePolicy.user_facing_fast()
            if gesture_fast_path
            else LiveGestureObservePolicy.full()
        )
        try:
            sparse_keypoints: dict[int, tuple[float, float, float]]
            pose_hint_source: str
            pose_hint_confidence: float | None
            if gesture_fast_path:
                visible_person_boxes = self._ordered_unique_gesture_boxes(
                    primary_person_box=detection.primary_person_box,
                    candidate_boxes=tuple(
                        person.box
                        for person in detection.visible_persons
                        if person.box is not None
                    ),
                )
                gesture_targets = GesturePersonTargets(
                    primary_person_box=detection.primary_person_box,
                    visible_person_boxes=visible_person_boxes,
                    person_count=len(visible_person_boxes),
                    source=(
                        "gesture_fast_path_current_frame_person_only"
                        if visible_person_boxes
                        else "gesture_fast_path_current_frame_only"
                    ),
                    face_anchor_state="disabled_fast_path",
                    face_anchor_count=0,
                )
                sparse_keypoints, pose_hint_source, pose_hint_confidence = {}, "disabled_fast_path", None
            else:
                with workflow_span(
                    name="camera_adapter_gesture_face_anchor_detect",
                    kind="io",
                ):
                    face_anchors = self._detect_face_anchors_for_gesture(
                        detection=detection,
                        frame_rgb=frame_rgb,
                    )
                with workflow_span(
                    name="camera_adapter_gesture_target_resolution",
                    kind="decision",
                ):
                    gesture_targets = self._resolve_gesture_person_targets(
                        detection=detection,
                        face_anchors=face_anchors,
                        observed_monotonic=observed_monotonic,
                    )
            workflow_decision(
                msg="camera_adapter_gesture_target_selection",
                question="Which visible person targets should the dedicated gesture lane trust for this frame?",
                selected={
                    "id": gesture_targets.source,
                    "summary": (
                        "Stay current-frame-only, but keep the current frame's person boxes available for bounded ROI recovery."
                        if gesture_fast_path
                        else "Use the resolved gesture target set for ROI-conditioned gesture recovery."
                    ),
                },
                options=[
                    {"id": "imx500", "summary": "Use the direct IMX500 person boxes."},
                    {
                        "id": "face_anchor_promoted",
                        "summary": "Promote a face-anchor-derived person ROI when the body detector misses the actual user.",
                    },
                    {
                        "id": "recent_target_cache",
                        "summary": "Reuse the most recent stable user ROI for a brief detector dropout window.",
                    },
                    {
                        "id": "gesture_fast_path_current_frame_only",
                        "summary": "Use only current-frame evidence and do not reuse recent person targets.",
                    },
                    {"id": "none", "summary": "Proceed without a usable person target."},
                ],
                context={
                    "detection_person_count": detection.person_count,
                    "visible_person_count": len(detection.visible_persons),
                    "gesture_target_person_count": gesture_targets.person_count,
                    "face_anchor_state": gesture_targets.face_anchor_state,
                    "face_anchor_count": gesture_targets.face_anchor_count,
                },
                confidence="forensic",
                guardrails=["gesture_person_target_resolution"],
                kpi_impact_estimate={"latency": "low", "roi_quality": "high"},
            )
            gesture_detection = self._gesture_detection_result(
                detection=detection,
                targets=gesture_targets,
            )
            if not gesture_fast_path:
                with workflow_span(
                    name="camera_adapter_gesture_pose_hints",
                    kind="decision",
                ):
                    sparse_keypoints, pose_hint_source, pose_hint_confidence = self._resolve_gesture_pose_hints(
                        runtime=runtime,
                        observed_at=pipeline_observed_at,
                        observed_monotonic=observed_monotonic,
                        detection=gesture_detection,
                        frame_rgb=frame_rgb,
                    )
            with workflow_span(
                name="camera_adapter_gesture_live_pipeline",
                kind="io",
            ):
                gesture_pipeline = self._ensure_live_gesture_pipeline()
                gesture_observation = gesture_pipeline.observe(
                    frame_rgb=frame_rgb,
                    observed_at=pipeline_observed_at,
                    observe_policy=observe_policy,
                    primary_person_box=gesture_targets.primary_person_box,
                    visible_person_boxes=gesture_targets.visible_person_boxes,
                    person_count=gesture_targets.person_count,
                    sparse_keypoints=sparse_keypoints,
                )
                gesture_debug = gesture_pipeline.debug_snapshot()
            pose_fallback: PoseResult | None
            pose_fallback_error: str | None
            if allow_pose_fallback and not gesture_fast_path:
                with workflow_span(
                    name="camera_adapter_gesture_pose_fallback",
                    kind="decision",
                ):
                    pose_fallback, pose_fallback_error = self._resolve_gesture_pose_fallback(
                        runtime,
                        observed_at=pipeline_observed_at,
                        observed_monotonic=observed_monotonic,
                        detection=gesture_detection,
                        frame_rgb=frame_rgb,
                        gesture_observation=gesture_observation,
                    )
            else:
                pose_fallback, pose_fallback_error = None, None
            final_resolved_source = str(getattr(gesture_debug, "get", lambda *_: "none")("resolved_source", "none") or "none")
            if (
                pose_fallback is not None
                and final_resolved_source == "none"
                and pose_fallback.gesture_event != AICameraGestureEvent.NONE
            ):
                final_resolved_source = "mediapipe_pose_event_fallback"
            active_workflow_forensics = current_workflow_forensics()
            live_hand_count = self._safe_nonnegative_int(getattr(gesture_observation, "hand_count", 0))
            forensics_zero_signal_capture_requested = (
                active_workflow_forensics is not None
                and detection.person_count <= 0
                and gesture_targets.person_count <= 0
                and live_hand_count <= 0
                and gesture_observation.fine_hand_gesture == AICameraFineHandGesture.NONE
                and gesture_observation.gesture_event == AICameraGestureEvent.NONE
                and final_resolved_source == "none"
            )
            self._set_last_gesture_debug_details(
                {
                    **self._sanitize_debug_mapping(gesture_debug),
                    "forensics_active": active_workflow_forensics is not None,
                    "forensics_run_id": (
                        None if active_workflow_forensics is None else active_workflow_forensics.run_id
                    ),
                    "forensics_trace_id": current_workflow_trace_id(),
                    "forensics_zero_signal_capture_requested": forensics_zero_signal_capture_requested,
                    "live_fine_hand_gesture": gesture_observation.fine_hand_gesture.value,
                    "live_fine_hand_gesture_confidence": self._rounded_confidence(
                        gesture_observation.fine_hand_gesture_confidence
                    ),
                    "live_gesture_event": gesture_observation.gesture_event.value,
                    "live_gesture_confidence": self._rounded_confidence(
                        gesture_observation.gesture_confidence
                    ),
                    "live_hand_count": live_hand_count,
                    "gesture_fast_path": gesture_fast_path,
                    "gesture_observe_policy": observe_policy.name,
                    "pose_hint_source": pose_hint_source,
                    "pose_hint_confidence": self._rounded_confidence(pose_hint_confidence),
                    "pose_fallback_used": pose_fallback is not None,
                    "pose_fallback_disabled_by_caller": not allow_pose_fallback,
                    "pose_fallback_disabled_by_fast_path": gesture_fast_path,
                    "pose_fallback_error": pose_fallback_error,
                    "pose_fallback_fine_hand_gesture": (
                        None if pose_fallback is None else pose_fallback.fine_hand_gesture.value
                    ),
                    "pose_fallback_fine_hand_gesture_confidence": (
                        None
                        if pose_fallback is None
                        else self._rounded_confidence(pose_fallback.fine_hand_gesture_confidence)
                    ),
                    "pose_fallback_gesture_event": (
                        None if pose_fallback is None else pose_fallback.gesture_event.value
                    ),
                    "pose_fallback_gesture_confidence": (
                        None
                        if pose_fallback is None
                        else self._rounded_confidence(pose_fallback.gesture_confidence)
                    ),
                    "final_resolved_source": final_resolved_source,
                    "detection_person_count": detection.person_count,
                    "detection_primary_person_zone": detection.primary_person_zone.value,
                    "detection_visible_person_count": len(detection.visible_persons),
                    "detection_primary_person_box_available": detection.primary_person_box is not None,
                    "gesture_target_source": gesture_targets.source,
                    "gesture_target_face_anchor_state": gesture_targets.face_anchor_state,
                    "gesture_target_face_anchor_count": gesture_targets.face_anchor_count,
                    "gesture_target_person_count": gesture_targets.person_count,
                    "gesture_target_primary_person_box_available": gesture_targets.primary_person_box is not None,
                    "pipeline_observed_at": round(pipeline_observed_at, 3),
                }
            )
            workflow_decision(
                msg="camera_adapter_gesture_resolution",
                question="Which gesture source should the adapter expose for this frame?",
                selected={
                    "id": final_resolved_source,
                    "summary": "Expose the strongest bounded gesture source that survived the dedicated live pipeline.",
                },
                options=[
                    {"id": "live_stream", "summary": "Use the direct live-stream recognizer result."},
                    {"id": "person_roi", "summary": "Use one person-conditioned ROI hand result."},
                    {"id": "visible_person_roi", "summary": "Use one visible-person ROI hand result."},
                    {"id": "live_hand_roi", "summary": "Use one tight hand ROI recovered from live hand boxes."},
                    {"id": "full_frame_hand_roi", "summary": "Use the final whole-frame hand rescue result."},
                    {"id": "mediapipe_pose_event_fallback", "summary": "Use only the pose-event fallback path."},
                    {"id": "none", "summary": "Expose no concrete gesture from this frame."},
                ],
                context={
                    "live_fine_hand_gesture": gesture_observation.fine_hand_gesture.value,
                    "live_fine_hand_confidence": (
                        None
                        if not isinstance(self._last_gesture_debug_details, dict)
                        else self._last_gesture_debug_details.get("live_fine_hand_gesture_confidence")
                    ),
                    "live_gesture_event": gesture_observation.gesture_event.value,
                    "live_gesture_confidence": (
                        None
                        if not isinstance(self._last_gesture_debug_details, dict)
                        else self._last_gesture_debug_details.get("live_gesture_confidence")
                    ),
                    "pose_hint_source": pose_hint_source,
                    "pose_hint_confidence": (
                        None
                        if not isinstance(self._last_gesture_debug_details, dict)
                        else self._last_gesture_debug_details.get("pose_hint_confidence")
                    ),
                    "pose_fallback_used": pose_fallback is not None,
                    "pose_fallback_event": None if pose_fallback is None else pose_fallback.gesture_event.value,
                    "person_count": detection.person_count,
                },
                confidence="forensic",
                guardrails=["dedicated_gesture_lane"],
                kpi_impact_estimate={"latency": "medium", "gesture_accuracy": "high"},
            )
            camera_metrics = self._runtime_manager.last_camera_metrics()
            if camera_metrics:
                self._update_last_gesture_debug_details(camera_metrics)
            with workflow_span(
                name="camera_adapter_gesture_candidate_capture",
                kind="io",
            ):
                capture_result = self._gesture_candidate_capture.maybe_capture(
                    observed_at=observed_at,
                    frame_rgb=frame_rgb,
                    debug_details=self._last_gesture_debug_details,
                )
            self._update_last_gesture_debug_details(capture_result.debug_fields())
            if stream_mode:
                self._annotate_gesture_stream_debug_locked(source=stream_source)
        except Exception as exc:  # pragma: no cover - hardware/runtime coupling is environment-dependent.
            code = self._classify_error(exc)
            safe_error_message = self._sanitize_error_text(str(exc), limit=_DEBUG_MAX_ERROR_STRING)
            logger.warning(
                "Local AI camera live gesture observation failed with %s.",
                code,
                exc_info=True,
            )
            workflow_event(
                kind="exception",
                msg="camera_adapter_gesture_pipeline_failed",
                level="ERROR",
                details={
                    "error_type": type(exc).__name__,
                    "error_code": code,
                    "error_message": safe_error_message,
                },
            )
            self._set_last_gesture_debug_details(
                {
                    "resolved_source": "pipeline_error",
                    "pipeline_error": code,
                    "pipeline_error_message": safe_error_message,
                    "stream_mode": "gesture_stream" if stream_mode else None,
                    "gesture_stream_source": stream_source if stream_mode else None,
                }
            )
            self._safe_close_live_gesture_pipeline_locked()
            self._safe_close_runtime_locked()
            return self._health_only_observation(
                observed_at=observed_at,
                online=True,
                ready=False,
                ai_ready=False,
                error=code,
            )

        fine_hand_gesture = gesture_observation.fine_hand_gesture
        fine_hand_gesture_confidence = gesture_observation.fine_hand_gesture_confidence
        gesture_event = gesture_observation.gesture_event
        gesture_confidence = gesture_observation.gesture_confidence
        live_hand_count = self._safe_nonnegative_int(getattr(gesture_observation, "hand_count", 0))
        hand_or_object_near_camera = live_hand_count > 0
        showing_intent_likely = (
            True
            if live_hand_count > 0
            or fine_hand_gesture != AICameraFineHandGesture.NONE
            or gesture_event != AICameraGestureEvent.NONE
            else None
        )
        model_name = "local-imx500+mediapipe-live-gesture"
        if pose_fallback is not None:
            if gesture_event == AICameraGestureEvent.NONE and pose_fallback.gesture_event != AICameraGestureEvent.NONE:
                gesture_event = pose_fallback.gesture_event
                gesture_confidence = pose_fallback.gesture_confidence
                model_name = "local-imx500+mediapipe-live-gesture+pose-fallback"
            hand_or_object_near_camera = hand_or_object_near_camera or bool(pose_fallback.hand_near_camera)
            if showing_intent_likely is None and pose_fallback.showing_intent_likely is not None:
                showing_intent_likely = pose_fallback.showing_intent_likely

        observation = AICameraObservation(
            observed_at=observed_at,
            camera_online=True,
            camera_ready=True,
            camera_ai_ready=True,
            person_count=detection.person_count,
            primary_person_box=detection.primary_person_box,
            primary_person_zone=detection.primary_person_zone,
            visible_persons=detection.visible_persons,
            hand_or_object_near_camera=hand_or_object_near_camera,
            showing_intent_likely=showing_intent_likely,
            gesture_event=gesture_event,
            gesture_confidence=gesture_confidence,
            fine_hand_gesture=fine_hand_gesture,
            fine_hand_gesture_confidence=fine_hand_gesture_confidence,
            gesture_temporal_authoritative=bool(
                getattr(gesture_observation, "gesture_temporal_authoritative", False)
            ),
            gesture_activation_key=getattr(gesture_observation, "gesture_activation_key", None),
            gesture_activation_token=getattr(gesture_observation, "gesture_activation_token", None),
            gesture_activation_started_at=getattr(
                gesture_observation,
                "gesture_activation_started_at",
                None,
            ),
            gesture_activation_changed_at=getattr(
                gesture_observation,
                "gesture_activation_changed_at",
                None,
            ),
            gesture_activation_source=getattr(gesture_observation, "gesture_activation_source", None),
            gesture_activation_rising=bool(getattr(gesture_observation, "gesture_activation_rising", False)),
            model=model_name,
        )
        workflow_event(
            kind="metric",
            msg="camera_adapter_gesture_observation_ready",
            details={
                "model": model_name,
                "fine_hand_gesture": fine_hand_gesture.value,
                "gesture_event": gesture_event.value,
                "hand_count": live_hand_count,
            },
        )
        return self._with_health(
            observation,
            online=True,
            ready=True,
            ai_ready=True,
            error=None,
            frame_at=frame_at,
        )

    def _resolve_gesture_pose_fallback(
        self,
        runtime: dict[str, Any],
        *,
        observed_at: float,
        observed_monotonic: float,
        detection: DetectionResult,
        frame_rgb: Any,
        gesture_observation: object,
    ) -> tuple[PoseResult | None, str | None]:
        """Run the full MediaPipe gesture stack when the fast lane produced no concrete result."""

        live_fine_hand_gesture = getattr(gesture_observation, "fine_hand_gesture", AICameraFineHandGesture.NONE)
        live_gesture_event = getattr(gesture_observation, "gesture_event", AICameraGestureEvent.NONE)
        if live_fine_hand_gesture != AICameraFineHandGesture.NONE or live_gesture_event != AICameraGestureEvent.NONE:
            return None, None
        if detection.person_count <= 0 or detection.primary_person_box is None:
            return None, None
        if self.config.pose_backend != "mediapipe":
            return None, None
        return self._resolve_mediapipe_pose(
            runtime,
            observed_at=observed_at,
            observed_monotonic=observed_monotonic,
            detection=detection,
            frame_rgb=frame_rgb,
            frame_error=None,
        )

    def _resolve_gesture_pose_hints(
        self,
        runtime: dict[str, Any],
        *,
        observed_at: float,
        observed_monotonic: float,
        detection: DetectionResult,
        frame_rgb: Any,
    ) -> tuple[dict[int, tuple[float, float, float]], str, float | None]:
        """Return fresh or cached sparse pose hints for the dedicated gesture lane."""

        if detection.person_count <= 0 or detection.primary_person_box is None:
            self._clear_pose_hint_cache()
            return {}, "none", None
        if self._should_reuse_pose_hint_cache(
            observed_monotonic=observed_monotonic,
            primary_person_box=detection.primary_person_box,
        ):
            return (
                dict(self._last_pose_hint_keypoints),
                "cache",
                self._last_pose_hint_confidence,
            )
        if self.config.pose_backend != "mediapipe":
            self._clear_pose_hint_cache()
            return {}, "unsupported_backend", None
        try:
            pose_hints = self._ensure_mediapipe_pipeline().analyze_pose_hints(
                frame_rgb=frame_rgb,
                observed_at=observed_at,
            )
        except Exception as exc:  # pragma: no cover - depends on Pi runtime and model assets.
            code = self._classify_error(exc)
            logger.warning("Local AI camera gesture pose-hint inference failed with %s.", code)
            logger.debug("Local AI camera gesture pose-hint exception details.", exc_info=True)
            self._safe_close_mediapipe_pipeline_locked()
            self._clear_pose_hint_cache()
            return {}, "error", None
        if not pose_hints.sparse_keypoints:
            self._clear_pose_hint_cache()
            return {}, "empty", pose_hints.pose_confidence
        self._store_pose_hint_cache(
            sparse_keypoints=pose_hints.sparse_keypoints,
            pose_confidence=pose_hints.pose_confidence,
            observed_monotonic=observed_monotonic,
            primary_person_box=detection.primary_person_box,
        )
        return dict(pose_hints.sparse_keypoints), "fresh_mediapipe", pose_hints.pose_confidence

    def _detect_face_anchors_for_gesture(
        self,
        *,
        detection: DetectionResult,
        frame_rgb: Any | None,
    ) -> SupplementalFaceAnchorResult:
        """Return bounded face anchors for the gesture lane when the detector is available."""

        if self._face_anchor_detector is None or frame_rgb is None or detection.person_count >= 2:
            return SupplementalFaceAnchorResult(
                state=("disabled" if self._face_anchor_detector is None else "skipped"),
                detail="gesture_face_anchors_unused",
            )
        try:
            return self._face_anchor_detector.detect(frame_rgb)
        except Exception:
            logger.warning("Local AI camera gesture face-anchor detection failed.")
            logger.debug("Local AI camera gesture face-anchor exception details.", exc_info=True)
            return SupplementalFaceAnchorResult(
                state="backend_unavailable",
                detail="gesture_face_anchor_detection_failed",
            )

    def _resolve_gesture_person_targets(
        self,
        *,
        detection: DetectionResult,
        face_anchors: SupplementalFaceAnchorResult,
        observed_monotonic: float | None = None,
    ) -> GesturePersonTargets:
        """Choose the body ROIs that the gesture lane should search first.

        The IMX500 SSD path can occasionally lock onto chair backs or similar
        furniture. When YuNet sees a real face outside that primary body box,
        promote one bounded face-expanded body ROI so the gesture lane follows
        the human instead of the furniture-shaped false positive.

        Additionally, keep the most recent stable ROI alive for a very short
        period so brief detector dropouts do not erase the active user target.
        """

        base_boxes = tuple(
            person.box
            for person in detection.visible_persons
            if getattr(person, "box", None) is not None
        )
        if not base_boxes and detection.primary_person_box is not None:
            base_boxes = (detection.primary_person_box,)
        candidate_boxes = list(base_boxes)
        promoted_boxes: list[AICameraBox] = []
        face_boxes = tuple(
            face_person.box
            for face_person in face_anchors.visible_persons
            if getattr(face_person, "box", None) is not None
        )
        primary_matches_face = (
            detection.primary_person_box is not None
            and any(
                self._box_contains_point(
                    detection.primary_person_box,
                    x=face_box.center_x,
                    y=face_box.center_y,
                )
                for face_box in face_boxes
            )
        )
        for face_box in face_boxes:
            if any(
                self._box_contains_point(body_box, x=face_box.center_x, y=face_box.center_y)
                for body_box in base_boxes
            ):
                continue
            promoted_boxes.append(self._expand_face_box_to_gesture_person_box(face_box))
        candidate_boxes.extend(promoted_boxes)

        primary_person_box = detection.primary_person_box
        source = "imx500"
        if promoted_boxes and (primary_person_box is None or not primary_matches_face):
            primary_person_box = promoted_boxes[0]
            source = "face_anchor_promoted"

        ordered_boxes = self._ordered_unique_gesture_boxes(
            primary_person_box=primary_person_box,
            candidate_boxes=tuple(candidate_boxes),
        )
        if not ordered_boxes and observed_monotonic is not None:
            recent_targets = self._recent_gesture_person_targets(observed_monotonic=observed_monotonic)
            if recent_targets is not None:
                return recent_targets

        resolved_targets = GesturePersonTargets(
            primary_person_box=primary_person_box,
            visible_person_boxes=ordered_boxes,
            person_count=len(ordered_boxes),
            source=("none" if not ordered_boxes else source),
            face_anchor_state=face_anchors.state,
            face_anchor_count=len(face_boxes),
        )
        if observed_monotonic is not None and resolved_targets.person_count > 0:
            self._remember_gesture_person_targets(
                targets=resolved_targets,
                observed_monotonic=observed_monotonic,
            )
        return resolved_targets

    def _gesture_detection_result(
        self,
        *,
        detection: DetectionResult,
        targets: GesturePersonTargets,
    ) -> DetectionResult:
        """Return one gesture-lane detection view with promoted person targets applied."""

        return DetectionResult(
            person_count=targets.person_count,
            primary_person_box=targets.primary_person_box,
            primary_person_zone=(
                detection.primary_person_zone
                if targets.primary_person_box is None
                else self._gesture_zone_from_box(targets.primary_person_box)
            ),
            visible_persons=detection.visible_persons,
            person_near_device=detection.person_near_device,
            hand_or_object_near_camera=detection.hand_or_object_near_camera,
            objects=detection.objects,
        )

    def _expand_face_box_to_gesture_person_box(self, face_box: AICameraBox) -> AICameraBox:
        """Return one bounded upper-body gesture-search ROI around a detected face."""

        face_width = max(0.01, face_box.width)
        face_height = max(0.01, face_box.height)
        body_height = min(0.92, max(face_height * 5.4, face_width * 4.2))
        body_width = min(0.74, max(face_width * 3.2, body_height * 0.55))
        top = max(0.0, face_box.top - (face_height * 0.65))
        left = max(0.0, face_box.center_x - (body_width / 2.0))
        return AICameraBox(
            top=top,
            left=left,
            bottom=min(1.0, top + body_height),
            right=min(1.0, left + body_width),
        )

    def _ordered_unique_gesture_boxes(
        self,
        *,
        primary_person_box: AICameraBox | None,
        candidate_boxes: tuple[AICameraBox, ...],
    ) -> tuple[AICameraBox, ...]:
        """Return candidate gesture person boxes with duplicates collapsed."""

        ordered: list[AICameraBox] = []
        if primary_person_box is not None:
            ordered.append(primary_person_box)
        ordered.extend(candidate_boxes)
        unique: list[AICameraBox] = []
        for box in ordered:
            if any(self._gesture_boxes_similar(box, existing) for existing in unique):
                continue
            unique.append(box)
        return tuple(unique)

    def _gesture_boxes_similar(self, first: AICameraBox, second: AICameraBox) -> bool:
        """Return whether two candidate gesture boxes likely describe the same target."""

        first_metrics = self._extract_box_association_metrics(first)
        second_metrics = self._extract_box_association_metrics(second)
        if first_metrics is None or second_metrics is None:
            return False
        return self._box_metrics_similar(first_metrics, second_metrics)

    def _box_contains_point(self, box: AICameraBox, *, x: float, y: float) -> bool:
        """Return whether one normalized point sits inside one normalized box."""

        return box.left <= x <= box.right and box.top <= y <= box.bottom

    def _gesture_zone_from_box(self, box: AICameraBox) -> AICameraZone:
        """Return the coarse horizontal zone for one gesture target box."""

        if box.center_x <= 0.36:
            return AICameraZone.LEFT
        if box.center_x >= 0.64:
            return AICameraZone.RIGHT
        return AICameraZone.CENTER

    def _stable_gesture_pipeline_observed_at(
        self,
        *,
        observed_at: float,
        observed_monotonic: float,
    ) -> float:
        """Return a wall-clock-like timestamp that never moves backwards for live Tasks APIs."""

        observed_at_f = float(observed_at)
        observed_monotonic_f = float(observed_monotonic)
        base_wall = getattr(self, "_gesture_pipeline_time_wall_base", None)
        base_mono = getattr(self, "_gesture_pipeline_time_monotonic_base", None)
        last_observed_at = getattr(self, "_gesture_pipeline_last_observed_at", None)

        rebase_clock = (
            not isinstance(base_wall, (int, float))
            or not isinstance(base_mono, (int, float))
            or observed_monotonic_f < float(base_mono)
        )
        if rebase_clock:
            base_wall = observed_at_f
            base_mono = observed_monotonic_f

        stable_observed_at = float(base_wall) + max(0.0, observed_monotonic_f - float(base_mono))
        if isinstance(last_observed_at, (int, float)) and stable_observed_at <= float(last_observed_at):
            stable_observed_at = float(last_observed_at) + _GESTURE_PIPELINE_MIN_STEP_S

        self._gesture_pipeline_time_wall_base = float(base_wall)
        self._gesture_pipeline_time_monotonic_base = float(base_mono)
        self._gesture_pipeline_last_observed_at = stable_observed_at
        return stable_observed_at

    def _remember_gesture_person_targets(
        self,
        *,
        targets: GesturePersonTargets,
        observed_monotonic: float,
    ) -> None:
        """Store the last usable gesture target set for brief detector dropouts."""

        if targets.person_count <= 0 or (
            targets.primary_person_box is None and not targets.visible_person_boxes
        ):
            return
        self._last_gesture_targets_cache = {
            "observed_monotonic": float(observed_monotonic),
            "primary_person_box": targets.primary_person_box,
            "visible_person_boxes": tuple(targets.visible_person_boxes),
            "person_count": int(targets.person_count),
            "source": str(targets.source),
            "face_anchor_state": str(targets.face_anchor_state),
            "face_anchor_count": int(targets.face_anchor_count),
        }

    def _recent_gesture_person_targets(
        self,
        *,
        observed_monotonic: float,
    ) -> GesturePersonTargets | None:
        """Return a short-lived cached gesture target set if it is still fresh."""

        cache = getattr(self, "_last_gesture_targets_cache", None)
        if not isinstance(cache, dict):
            return None
        cached_at = cache.get("observed_monotonic")
        if not isinstance(cached_at, (int, float)):
            return None
        age_s = float(observed_monotonic) - float(cached_at)
        if age_s < 0.0 or age_s > self._gesture_target_cache_ttl_seconds():
            return None

        primary_person_box = cache.get("primary_person_box")
        visible_person_boxes = tuple(cache.get("visible_person_boxes") or ())
        person_count = self._safe_nonnegative_int(cache.get("person_count", len(visible_person_boxes)))
        if person_count <= 0 or (primary_person_box is None and not visible_person_boxes):
            return None
        return GesturePersonTargets(
            primary_person_box=primary_person_box,
            visible_person_boxes=visible_person_boxes,
            person_count=person_count or len(visible_person_boxes),
            source="recent_target_cache",
            face_anchor_state=str(cache.get("face_anchor_state", "cache")),
            face_anchor_count=self._safe_nonnegative_int(cache.get("face_anchor_count", 0)),
        )

    def _gesture_target_cache_ttl_seconds(self) -> float:
        """Return the configured recent-target retention window in seconds."""

        configured_ttl = getattr(self.config, "gesture_target_cache_ttl_seconds", _GESTURE_TARGET_CACHE_TTL_S)
        try:
            ttl_seconds = float(configured_ttl)
        except (TypeError, ValueError):
            ttl_seconds = _GESTURE_TARGET_CACHE_TTL_S
        return min(2.0, max(0.0, ttl_seconds))

    def _set_last_gesture_debug_details(self, details: Mapping[str, Any]) -> None:
        """Replace the operator-facing debug snapshot with a bounded, sanitized mapping."""

        self._last_gesture_debug_details = self._sanitize_debug_mapping(details)

    def _update_last_gesture_debug_details(self, details: Any) -> None:
        """Merge bounded debug fields into the operator-facing snapshot."""

        current = dict(self._last_gesture_debug_details or {})
        current.update(self._sanitize_debug_mapping(details))
        self._last_gesture_debug_details = current

    def _sanitize_debug_mapping(self, details: Any) -> dict[str, Any]:
        """Return a bounded debug mapping safe to store and expose externally."""

        sanitized = self._sanitize_debug_value(details)
        if isinstance(sanitized, dict):
            return sanitized
        return {"value": sanitized}

    def _sanitize_debug_value(self, value: Any, *, depth: int = 0) -> Any:
        """Bound debug payloads to small JSON-like structures."""

        if depth >= _DEBUG_MAX_DEPTH:
            return "<truncated>"

        if value is None or isinstance(value, (bool, int)):
            return value

        if isinstance(value, float):
            if not math.isfinite(value):
                return None
            return round(value, 4)

        if isinstance(value, str):
            return self._sanitize_error_text(value, limit=_DEBUG_MAX_STRING)

        if isinstance(value, Enum):
            enum_value = getattr(value, "value", None)
            if isinstance(enum_value, (str, int, float, bool)) or enum_value is None:
                return self._sanitize_debug_value(enum_value, depth=depth + 1)
            return type(value).__name__

        if isinstance(value, (bytes, bytearray, memoryview)):
            return f"<{type(value).__name__}:{len(value)}>"

        if hasattr(value, "top") and hasattr(value, "left") and hasattr(value, "bottom") and hasattr(value, "right"):
            return {
                "top": self._sanitize_debug_value(getattr(value, "top"), depth=depth + 1),
                "left": self._sanitize_debug_value(getattr(value, "left"), depth=depth + 1),
                "bottom": self._sanitize_debug_value(getattr(value, "bottom"), depth=depth + 1),
                "right": self._sanitize_debug_value(getattr(value, "right"), depth=depth + 1),
            }

        if is_dataclass(value):
            return self._sanitize_debug_value(asdict(value), depth=depth + 1)

        if isinstance(value, Mapping):
            sanitized_dict: dict[str, Any] = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= _DEBUG_MAX_ITEMS:
                    sanitized_dict["__truncated_items__"] = len(value) - _DEBUG_MAX_ITEMS
                    break
                sanitized_key = self._sanitize_error_text(str(key), limit=64)
                if sanitized_key.endswith("_path") and isinstance(item, str):
                    collapsed = " ".join(item.split())
                    sanitized_dict[sanitized_key] = collapsed[:_DEBUG_MAX_STRING]
                    continue
                sanitized_dict[sanitized_key] = self._sanitize_debug_value(item, depth=depth + 1)
            return sanitized_dict

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
            sanitized_items = [
                self._sanitize_debug_value(item, depth=depth + 1)
                for item in value[:_DEBUG_MAX_ITEMS]
            ]
            remaining_items = len(value) - len(sanitized_items)
            if remaining_items > 0:
                sanitized_items.append(f"<{remaining_items} more>")
            return sanitized_items

        value_dict = getattr(value, "__dict__", None)
        if isinstance(value_dict, dict) and depth + 1 < _DEBUG_MAX_DEPTH:
            sanitized_object = self._sanitize_debug_value(value_dict, depth=depth + 1)
            if isinstance(sanitized_object, dict):
                sanitized_object.setdefault("type", type(value).__name__)
                return sanitized_object

        return f"<{type(value).__name__}>"

    def _sanitize_error_text(self, text: str, *, limit: int) -> str:
        """Redact obvious sensitive substrings from operator-facing errors/debug text."""

        collapsed = " ".join(str(text).split())
        collapsed = _URL_RE.sub("<url>", collapsed)
        collapsed = _UNIX_PATH_RE.sub("<path>", collapsed)
        collapsed = _IPV4_RE.sub("<ip>", collapsed)
        if len(collapsed) > limit:
            return collapsed[: max(0, limit - 1)] + "…"
        return collapsed

    def _safe_nonnegative_int(self, value: Any) -> int:
        """Return a best-effort non-negative integer for partial runtime payloads."""

        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    def _rounded_confidence(self, value: Any) -> float | None:
        """Return a small, finite confidence value suitable for debug payloads."""

        if value is None:
            return None
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(confidence):
            return None
        return round(confidence, 3)
