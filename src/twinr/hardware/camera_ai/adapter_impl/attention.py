# CHANGELOG: 2026-03-28
# BUG-1: Auxiliary face-anchor/debug failures could abort a valid fast attention observation; the fast path now degrades gracefully and preserves observation delivery.
# BUG-2: attention_showing_intent_reason emitted incorrect explanations for states where hand_near was already true; reason generation is now exact.
# BUG-3: Stale debug snapshots survived failed refreshes; the snapshot is now cleared before each build and replaced by an explicit error payload on failure.
# SEC-1: Camera metrics are now sanitized, size-bounded, and prevented from shadowing authoritative attention fields in operator/debug payloads.
# IMP-1: Added schema-versioned, timestamped, latency-aware telemetry for Pi-4 fast-path tuning and profiling.
# IMP-2: Fast path now skips unnecessary face-anchor work when no frame or no visible person is present, and surfaces optional richer gaze/head-pose metadata when upstream inference provides it.

"""Fast attention-only helpers for the local AI-camera adapter."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Final

from ..detection import DetectionResult
from ..face_anchors import SupplementalFaceAnchorResult
from ..looking_signal import (
    AttentionStreamState,
    StreamLookingSignal,
    infer_fast_looking_signal,
    infer_stream_looking_signal,
)
from ..models import AICameraObservation
from .common import LOGGER

logger = LOGGER

_ATTENTION_DEBUG_SCHEMA_VERSION: Final = 2
_DEFAULT_DETECTION_CENTER_FALLBACK_CEILING: Final = 0.35
_DEBUG_FLOAT_DIGITS: Final = 3
_DEBUG_TIMESTAMP_DIGITS: Final = 6
_MAX_DEBUG_STRING_CHARS: Final = 256
_MAX_CAMERA_METRICS: Final = 32
_ATTENTION_ERROR_LOG_INTERVAL_SECONDS: Final = 5.0


class AICameraAdapterAttentionMixin:
    """Keep HDMI attention refresh logic separate from the heavier gesture lane."""

    _LEGACY_PUBLIC_ATTENTION_DEBUG_KEYS: Final[tuple[str, ...]] = (
        "mode",
        "pose_inference_skipped",
        "pose_skip_reason",
        "attention_pipeline_note",
        "detection_person_count",
        "detection_visible_person_count",
        "detection_primary_person_zone",
        "detection_person_near_device",
        "detection_hand_or_object_near_camera",
        "attention_visual_attention_source",
        "attention_visual_attention_score",
        "attention_visual_attention_threshold",
        "attention_visual_attention_fallback_ceiling",
        "attention_looking_toward_device",
        "attention_looking_signal_state",
        "attention_looking_signal_source",
        "attention_looking_reason",
        "attention_face_anchor_state",
        "attention_face_anchor_count",
        "attention_face_anchor_match_confidence",
        "attention_face_anchor_match_center_x",
        "attention_face_anchor_match_center_y",
        "attention_hand_near_source",
        "attention_hand_or_object_near_camera",
        "attention_hand_near_reason",
        "attention_showing_intent_source",
        "attention_showing_intent_likely",
        "attention_showing_intent_reason",
        "detection_primary_person_center_x",
        "detection_primary_person_center_y",
        "detection_primary_person_area",
        "detection_primary_person_height",
        "stream_mode",
        "attention_stream_lane",
        "attention_stream_transition_state",
        "attention_stream_changed",
        "attention_stream_activation_dwell_s",
        "attention_stream_release_dwell_s",
        "attention_stream_candidate_age_s",
        "attention_instant_visual_attention_score",
        "attention_instant_looking_toward_device",
        "attention_instant_looking_signal_state",
        "attention_instant_looking_signal_source",
        "attention_instant_looking_reason",
        "attention_stream_candidate_state",
        "attention_stream_candidate_source",
        "attention_stream_candidate_reason",
        "attention_stream_candidate_score",
    )

    def get_last_attention_debug_details(self) -> dict[str, Any] | None:
        """Return the newest bounded attention debug snapshot for operators."""

        if self._last_attention_debug_details is None:
            return None
        snapshot = deepcopy(self._last_attention_debug_details)
        if snapshot.get("pipeline_error") or snapshot.get("frame_error"):
            return snapshot
        return {
            key: snapshot[key]
            for key in self._LEGACY_PUBLIC_ATTENTION_DEBUG_KEYS
            if key in snapshot
        }

    def _rate_limited_attention_log(self, key: str, message: str, *, exc_info: bool = False) -> None:
        """Avoid per-frame log storms when an auxiliary fast-path stage starts failing."""

        now = time.monotonic()
        last_by_key = getattr(self, "_attention_log_last_monotonic_by_key", None)
        if not isinstance(last_by_key, dict):
            last_by_key = {}
        last_seen = float(last_by_key.get(key, float("-inf")))
        if now - last_seen < _ATTENTION_ERROR_LOG_INTERVAL_SECONDS:
            return

        logger.warning(message, exc_info=exc_info)
        updated = dict(last_by_key)
        updated[key] = now
        self._attention_log_last_monotonic_by_key = updated

    def _coerce_debug_scalar(self, value: Any, *, digits: int = _DEBUG_FLOAT_DIGITS) -> Any:
        """Convert values to a JSON/log-friendly bounded scalar."""

        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            return round(value, digits) if math.isfinite(value) else None
        if isinstance(value, str):
            return value[:_MAX_DEBUG_STRING_CHARS]
        if isinstance(value, bytes):
            return f"<bytes:{len(value)}>"
        if isinstance(value, Mapping):
            return f"<mapping:{type(value).__name__}>"
        if isinstance(value, (list, tuple, set, frozenset)):
            return f"<sequence:{type(value).__name__}:{len(value)}>"

        enum_value = getattr(value, "value", None)
        if enum_value is not None and enum_value is not value:
            return self._coerce_debug_scalar(enum_value, digits=digits)

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return f"<{type(value).__name__}>"
        return round(numeric, digits) if math.isfinite(numeric) else None

    def _merge_reason(self, base: str | None, extra: str | None) -> str | None:
        """Combine error/debug reason tokens without duplicating them."""

        tokens = [token for token in (base, extra) if token]
        if not tokens:
            return None

        merged: list[str] = []
        for token in tokens:
            if token not in merged:
                merged.append(token)
        return "|".join(merged)

    def _extract_fallback_visual_attention_ceiling(self, looking_signal: Any) -> float | None:
        """Read the fallback score ceiling from the richest available source."""

        if getattr(looking_signal, "source", None) != "detection_center_fallback":
            return None

        candidates = (
            getattr(looking_signal, "fallback_visual_attention_ceiling", None),
            getattr(self.config, "attention_center_fallback_ceiling", None),
            getattr(self.config, "detection_center_fallback_ceiling", None),
            _DEFAULT_DETECTION_CENTER_FALLBACK_CEILING,
        )
        for candidate in candidates:
            try:
                numeric = float(candidate)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                return round(numeric, _DEBUG_FLOAT_DIGITS)
        return None

    def _attention_stream_state_for_lane(self, lane: str) -> AttentionStreamState:
        """Return the mutable attention-stream state for one logical lane."""

        state_by_lane = getattr(self, "_attention_stream_state_by_lane", None)
        if not isinstance(state_by_lane, dict):
            state_by_lane = {}
            self._attention_stream_state_by_lane = state_by_lane
        state = state_by_lane.get(lane)
        if isinstance(state, AttentionStreamState):
            return state
        state = AttentionStreamState()
        state_by_lane[lane] = state
        return state

    def _apply_attention_stream_locked(
        self,
        *,
        lane: str,
        detection: DetectionResult,
        face_anchors: SupplementalFaceAnchorResult | None,
        observation: AICameraObservation,
        observed_at: float,
    ) -> AICameraObservation:
        """Apply explicit live-stream stabilization on top of one instant observation."""

        stream_signal = infer_stream_looking_signal(
            detection=detection,
            face_anchors=face_anchors,
            attention_score_threshold=self.config.attention_score_threshold,
            state=self._attention_stream_state_for_lane(lane),
            observed_at=observed_at,
        )
        engaged_score_threshold = self._coerce_debug_scalar(getattr(self.config, "engaged_score_threshold", 0.5))
        try:
            engaged_threshold = float(engaged_score_threshold)
        except (TypeError, ValueError):
            engaged_threshold = 0.5
        stable_score = stream_signal.visual_attention_score
        engaged_with_device = (
            observation.person_near_device is True
            and (stable_score or 0.0) >= max(0.0, min(1.0, engaged_threshold))
        )
        showing_intent_likely = bool(
            observation.hand_or_object_near_camera
            and (stream_signal.looking_toward_device is True or observation.person_near_device is True)
        )
        updated_observation = self._observation_copy(
            observation,
            looking_toward_device=stream_signal.looking_toward_device,
            looking_signal_state=stream_signal.state if stream_signal.source is not None else None,
            looking_signal_source=stream_signal.source,
            visual_attention_score=stream_signal.visual_attention_score,
            engaged_with_device=engaged_with_device,
            showing_intent_likely=showing_intent_likely,
        )
        self._annotate_attention_stream_debug_locked(
            lane=lane,
            stream_signal=stream_signal,
            observation=updated_observation,
        )
        return updated_observation

    def _annotate_attention_stream_debug_locked(
        self,
        *,
        lane: str,
        stream_signal: StreamLookingSignal,
        observation: AICameraObservation,
    ) -> None:
        """Overlay explicit stream facts onto the latest attention debug snapshot."""

        payload = dict(self._last_attention_debug_details or {})
        instant_signal = stream_signal.instant_signal
        candidate_signal = stream_signal.candidate_signal
        payload.update(
            {
                "mode": "attention_stream",
                "stream_mode": "attention_stream",
                "attention_stream_lane": lane,
                "attention_stream_transition_state": stream_signal.transition_state,
                "attention_stream_changed": stream_signal.changed,
                "attention_stream_activation_dwell_s": self._coerce_debug_scalar(stream_signal.activation_dwell_s),
                "attention_stream_release_dwell_s": self._coerce_debug_scalar(stream_signal.release_dwell_s),
                "attention_stream_candidate_age_s": self._coerce_debug_scalar(stream_signal.candidate_age_s),
                "attention_instant_visual_attention_score": self._coerce_debug_scalar(
                    instant_signal.visual_attention_score
                ),
                "attention_instant_looking_toward_device": self._coerce_debug_scalar(
                    instant_signal.looking_toward_device
                ),
                "attention_instant_looking_signal_state": self._coerce_debug_scalar(instant_signal.state),
                "attention_instant_looking_signal_source": self._coerce_debug_scalar(instant_signal.source),
                "attention_instant_looking_reason": self._coerce_debug_scalar(instant_signal.reason),
                "attention_stream_candidate_state": self._coerce_debug_scalar(
                    None if candidate_signal is None else candidate_signal.state
                ),
                "attention_stream_candidate_source": self._coerce_debug_scalar(
                    None if candidate_signal is None else candidate_signal.source
                ),
                "attention_stream_candidate_reason": self._coerce_debug_scalar(
                    None if candidate_signal is None else candidate_signal.reason
                ),
                "attention_stream_candidate_score": self._coerce_debug_scalar(
                    None if candidate_signal is None else candidate_signal.visual_attention_score
                ),
                "attention_visual_attention_score": self._coerce_debug_scalar(observation.visual_attention_score),
                "attention_looking_toward_device": self._coerce_debug_scalar(observation.looking_toward_device),
                "attention_looking_signal_state": self._coerce_debug_scalar(observation.looking_signal_state),
                "attention_looking_signal_source": self._coerce_debug_scalar(observation.looking_signal_source),
                "attention_looking_reason": self._coerce_debug_scalar(stream_signal.reason),
                "attention_showing_intent_likely": self._coerce_debug_scalar(observation.showing_intent_likely),
                "attention_showing_intent_reason": self._derive_showing_intent_reason(observation),
            }
        )
        self._last_attention_debug_details = payload

    def _derive_showing_intent_reason(self, observation: AICameraObservation) -> str:
        """Explain the current showing-intent state without guessing hidden upstream logic."""

        if observation.showing_intent_likely is True:
            return "fallback_conditions_met"

        hand_near = observation.hand_or_object_near_camera is True
        looking = observation.looking_toward_device is True
        person_near = observation.person_near_device is True

        if not hand_near:
            return "hand_near_false"
        if looking and person_near:
            return "showing_intent_false_despite_hand_looking_person_near"
        if looking:
            return "waiting_for_person_near_only"
        if person_near:
            return "waiting_for_looking_only"
        return "looking_and_person_near_false"

    def _sanitize_camera_metrics(self, camera_metrics: Any) -> tuple[dict[str, Any], bool]:
        """Convert camera metrics into a bounded JSON-safe mapping."""

        if not isinstance(camera_metrics, Mapping):
            return {}, False

        sanitized: dict[str, Any] = {}
        truncated = False
        for index, (raw_key, raw_value) in enumerate(camera_metrics.items()):
            if index >= _MAX_CAMERA_METRICS:
                truncated = True
                break
            key = str(raw_key)[:_MAX_DEBUG_STRING_CHARS]
            sanitized[key] = self._coerce_debug_scalar(raw_value)
        return sanitized, truncated

    def _merge_camera_metrics(self, payload: dict[str, Any]) -> None:
        """Attach sanitized runtime camera metrics without allowing key shadowing."""

        try:
            camera_metrics = self._runtime_manager.last_camera_metrics()
        except Exception:
            self._rate_limited_attention_log(
                "attention_camera_metrics_failed",
                "Fast attention camera metrics retrieval failed; omitting camera metrics from debug payload.",
                exc_info=True,
            )
            payload["camera_metrics_error"] = "last_camera_metrics_failed"
            return

        if not camera_metrics:
            return

        sanitized_metrics, truncated = self._sanitize_camera_metrics(camera_metrics)
        if not sanitized_metrics:
            if truncated:
                payload["camera_metrics_truncated"] = True
            elif camera_metrics is not None and not isinstance(camera_metrics, Mapping):
                payload["camera_metrics_error"] = "last_camera_metrics_non_mapping"
            return

        payload["camera_metrics"] = sanitized_metrics
        reserved_keys = set(payload)

        # BREAKING: conflicting camera metric keys no longer overwrite authoritative
        # attention fields; they are preserved under `camera_metrics` and, when
        # flattened, emitted with a `camera_metric_` prefix instead.
        for key, value in sanitized_metrics.items():
            flattened_key = key if key not in reserved_keys else f"camera_metric_{key}"
            if flattened_key in payload:
                continue
            payload[flattened_key] = value

        if truncated:
            payload["camera_metrics_truncated"] = True

    def _extend_payload_from_optional_looking_signal(self, payload: dict[str, Any], looking_signal: Any) -> None:
        """Surface richer optional model outputs when upstream fast-looking inference provides them."""

        optional_fields: tuple[tuple[str, str, int], ...] = (
            ("model_name", "attention_looking_model_name", _DEBUG_FLOAT_DIGITS),
            ("model_version", "attention_looking_model_version", _DEBUG_FLOAT_DIGITS),
            ("tracking_state", "attention_looking_tracking_state", _DEBUG_FLOAT_DIGITS),
            ("raw_score", "attention_visual_attention_raw_score", _DEBUG_FLOAT_DIGITS),
            ("calibrated_score", "attention_visual_attention_calibrated_score", _DEBUG_FLOAT_DIGITS),
            ("score_uncertainty", "attention_visual_attention_uncertainty", _DEBUG_FLOAT_DIGITS),
            ("head_pose_yaw", "attention_head_pose_yaw_deg", _DEBUG_FLOAT_DIGITS),
            ("head_pose_pitch", "attention_head_pose_pitch_deg", _DEBUG_FLOAT_DIGITS),
            ("head_pose_roll", "attention_head_pose_roll_deg", _DEBUG_FLOAT_DIGITS),
            ("gaze_yaw", "attention_gaze_yaw_deg", _DEBUG_FLOAT_DIGITS),
            ("gaze_pitch", "attention_gaze_pitch_deg", _DEBUG_FLOAT_DIGITS),
            ("temporal_filter_state", "attention_temporal_filter_state", _DEBUG_FLOAT_DIGITS),
            ("temporal_window", "attention_temporal_window", _DEBUG_FLOAT_DIGITS),
            ("model_latency_ms", "attention_looking_model_latency_ms", _DEBUG_FLOAT_DIGITS),
        )
        for attr_name, payload_key, digits in optional_fields:
            value = getattr(looking_signal, attr_name, None)
            if value is None:
                continue
            payload[payload_key] = self._coerce_debug_scalar(value, digits=digits)

    def _build_attention_debug_details(
        self,
        *,
        detection: DetectionResult,
        observation: AICameraObservation,
        face_anchors: SupplementalFaceAnchorResult | None,
        observed_at: float | None = None,
        observed_monotonic: float | None = None,
        face_anchor_mode: str | None = None,
        face_anchor_error: str | None = None,
        face_anchor_stage_ms: float | None = None,
        compose_stage_ms: float | None = None,
        input_frame_present: bool | None = None,
    ) -> dict[str, Any]:
        """Summarize why fast attention facts did or did not become active.

        The HDMI attention loop deliberately skips pose/hand inference for
        latency. Expose that fact explicitly, together with the fallback score
        ceilings, so operators can see when `LOOKING`, `HAND_NEAR`, or
        `INTENT_LIKELY` were impossible on the fast path instead of treating
        them as random dropouts.
        """

        looking_signal = infer_fast_looking_signal(
            detection=detection,
            face_anchors=face_anchors,
            attention_score_threshold=self.config.attention_score_threshold,
        )
        visual_attention_score = observation.visual_attention_score
        attention_threshold = self._coerce_debug_scalar(self.config.attention_score_threshold)
        fallback_visual_attention_ceiling = self._extract_fallback_visual_attention_ceiling(looking_signal)
        looking_reason = self._coerce_debug_scalar(getattr(looking_signal, "reason", None))

        hand_near_reason = (
            "large_object_box_detected"
            if observation.hand_or_object_near_camera is True
            else "no_large_object_box_detected"
        )
        showing_intent_reason = self._derive_showing_intent_reason(observation)

        primary_box = detection.primary_person_box
        payload: dict[str, Any] = {
            "debug_schema_version": _ATTENTION_DEBUG_SCHEMA_VERSION,
            "mode": "attention_fast",
            "pose_inference_skipped": True,
            "pose_skip_reason": "latency_preserving_attention_fast_path",
            "attention_pipeline_note": "attention_fast_path_skips_pose_and_hand_inference",
            "attention_input_frame_present": input_frame_present,
            "detection_person_count": self._coerce_debug_scalar(detection.person_count),
            "detection_visible_person_count": self._coerce_debug_scalar(len(detection.visible_persons)),
            "detection_primary_person_zone": self._coerce_debug_scalar(detection.primary_person_zone),
            "detection_person_near_device": self._coerce_debug_scalar(detection.person_near_device),
            "detection_hand_or_object_near_camera": self._coerce_debug_scalar(detection.hand_or_object_near_camera),
            "attention_visual_attention_source": self._coerce_debug_scalar(getattr(looking_signal, "source", None)),
            "attention_visual_attention_score": self._coerce_debug_scalar(visual_attention_score),
            "attention_visual_attention_threshold": attention_threshold,
            "attention_visual_attention_fallback_ceiling": fallback_visual_attention_ceiling,
            "attention_looking_toward_device": self._coerce_debug_scalar(observation.looking_toward_device),
            "attention_looking_signal_state": self._coerce_debug_scalar(observation.looking_signal_state),
            "attention_looking_signal_source": self._coerce_debug_scalar(observation.looking_signal_source),
            "attention_looking_reason": looking_reason,
            "attention_face_anchor_state": self._coerce_debug_scalar(getattr(looking_signal, "face_anchor_state", None)),
            "attention_face_anchor_count": self._coerce_debug_scalar(getattr(looking_signal, "face_anchor_count", None)),
            "attention_face_anchor_match_confidence": self._coerce_debug_scalar(
                getattr(looking_signal, "matched_face_confidence", None)
            ),
            "attention_face_anchor_match_center_x": self._coerce_debug_scalar(
                getattr(looking_signal, "matched_face_center_x", None)
            ),
            "attention_face_anchor_match_center_y": self._coerce_debug_scalar(
                getattr(looking_signal, "matched_face_center_y", None)
            ),
            "attention_face_anchor_mode": self._coerce_debug_scalar(face_anchor_mode),
            "attention_face_anchor_error": self._coerce_debug_scalar(face_anchor_error),
            "attention_face_anchor_latency_ms": self._coerce_debug_scalar(face_anchor_stage_ms),
            "attention_hand_near_source": "detection_large_object_boxes",
            "attention_hand_or_object_near_camera": self._coerce_debug_scalar(observation.hand_or_object_near_camera),
            "attention_hand_near_reason": hand_near_reason,
            "attention_showing_intent_source": "detection_hand_plus_attention_fallback",
            "attention_showing_intent_likely": self._coerce_debug_scalar(observation.showing_intent_likely),
            # BREAKING: attention_showing_intent_reason now uses exact state names
            # instead of the previously incorrect legacy label "waiting_for_hand_near_only".
            "attention_showing_intent_reason": showing_intent_reason,
            "attention_observation_compose_latency_ms": self._coerce_debug_scalar(compose_stage_ms),
        }
        if observed_at is not None:
            payload["observed_at"] = self._coerce_debug_scalar(observed_at, digits=_DEBUG_TIMESTAMP_DIGITS)
        if observed_monotonic is not None:
            payload["observed_monotonic"] = self._coerce_debug_scalar(
                observed_monotonic,
                digits=_DEBUG_TIMESTAMP_DIGITS,
            )

        if primary_box is not None:
            payload.update(
                {
                    "detection_primary_person_center_x": self._coerce_debug_scalar(primary_box.center_x),
                    "detection_primary_person_center_y": self._coerce_debug_scalar(primary_box.center_y),
                    "detection_primary_person_area": self._coerce_debug_scalar(primary_box.area),
                    "detection_primary_person_height": self._coerce_debug_scalar(primary_box.height),
                }
            )

        self._extend_payload_from_optional_looking_signal(payload, looking_signal)
        self._merge_camera_metrics(payload)
        return payload

    def _build_attention_observation_locked(
        self,
        *,
        detection: DetectionResult,
        frame_rgb: Any | None,
        observed_at: float,
        observed_monotonic: float,
        stream_lane: str | None = None,
    ) -> AICameraObservation:
        """Compose one fast attention observation from supplied detection/frame facts."""

        pipeline_started = time.perf_counter()
        self._last_attention_debug_details = None

        detection_for_observation = detection
        face_anchors: SupplementalFaceAnchorResult | None = None
        face_anchor_mode = "not_attempted"
        face_anchor_error: str | None = None
        input_frame_present = frame_rgb is not None

        face_anchor_started = time.perf_counter()
        if frame_rgb is None:
            face_anchor_mode = "skipped_frame_unavailable"
        elif not detection.visible_persons:
            face_anchor_mode = "skipped_no_visible_persons"
        else:
            try:
                detection_for_observation, face_anchors = self._supplement_visible_persons_with_face_anchors(
                    detection=detection,
                    frame_rgb=frame_rgb,
                )
                face_anchor_mode = "supplemented" if face_anchors is not None else "no_face_anchor_match"
            except Exception:
                face_anchor_mode = "fallback_after_supplement_error"
                face_anchor_error = "face_anchor_supplement_failed"
                detection_for_observation = detection
                face_anchors = None
                self._rate_limited_attention_log(
                    "attention_face_anchor_supplement_failed",
                    "Fast attention face-anchor supplementation failed; continuing with detection-only attention.",
                    exc_info=True,
                )
        face_anchor_stage_ms = (time.perf_counter() - face_anchor_started) * 1000.0

        compose_started = time.perf_counter()
        try:
            observation = self._compose_observation(
                observed_at=observed_at,
                observed_monotonic=observed_monotonic,
                detection=detection_for_observation,
                pose=None,
                pose_error=None,
                face_anchors=face_anchors,
            )
        except Exception:
            if face_anchors is None and detection_for_observation is detection:
                raise

            face_anchor_mode = "fallback_after_compose_error"
            face_anchor_error = self._merge_reason(face_anchor_error, "compose_failed_with_face_anchors")
            detection_for_observation = detection
            face_anchors = None
            self._rate_limited_attention_log(
                "attention_compose_retry_without_face_anchors",
                "Fast attention compose failed with face-anchor enriched inputs; retrying without face anchors.",
                exc_info=True,
            )
            observation = self._compose_observation(
                observed_at=observed_at,
                observed_monotonic=observed_monotonic,
                detection=detection,
                pose=None,
                pose_error=None,
                face_anchors=None,
            )
        compose_stage_ms = (time.perf_counter() - compose_started) * 1000.0
        observation = self._observation_copy(
            observation,
            showing_intent_likely=bool(observation.showing_intent_likely),
        )

        debug_started = time.perf_counter()
        try:
            debug_payload = self._build_attention_debug_details(
                detection=detection_for_observation,
                observation=observation,
                face_anchors=face_anchors,
                observed_at=observed_at,
                observed_monotonic=observed_monotonic,
                face_anchor_mode=face_anchor_mode,
                face_anchor_error=face_anchor_error,
                face_anchor_stage_ms=face_anchor_stage_ms,
                compose_stage_ms=compose_stage_ms,
                input_frame_present=input_frame_present,
            )
            debug_payload["attention_debug_build_latency_ms"] = self._coerce_debug_scalar(
                (time.perf_counter() - debug_started) * 1000.0
            )
            debug_payload["attention_total_pipeline_latency_ms"] = self._coerce_debug_scalar(
                (time.perf_counter() - pipeline_started) * 1000.0
            )
            self._last_attention_debug_details = debug_payload
        except Exception:
            self._rate_limited_attention_log(
                "attention_debug_payload_failed",
                "Fast attention debug snapshot construction failed; returning observation with minimal error payload.",
                exc_info=True,
            )
            self._last_attention_debug_details = {
                "debug_schema_version": _ATTENTION_DEBUG_SCHEMA_VERSION,
                "mode": "attention_fast",
                "observed_at": self._coerce_debug_scalar(observed_at, digits=_DEBUG_TIMESTAMP_DIGITS),
                "observed_monotonic": self._coerce_debug_scalar(observed_monotonic, digits=_DEBUG_TIMESTAMP_DIGITS),
                "attention_input_frame_present": input_frame_present,
                "attention_face_anchor_mode": face_anchor_mode,
                "attention_face_anchor_error": self._coerce_debug_scalar(face_anchor_error),
                "attention_face_anchor_latency_ms": self._coerce_debug_scalar(face_anchor_stage_ms),
                "attention_observation_compose_latency_ms": self._coerce_debug_scalar(compose_stage_ms),
                "attention_debug_error": "build_failed",
                "attention_total_pipeline_latency_ms": self._coerce_debug_scalar(
                    (time.perf_counter() - pipeline_started) * 1000.0
                ),
            }
        if stream_lane is not None:
            observation = self._apply_attention_stream_locked(
                lane=stream_lane,
                detection=detection_for_observation,
                face_anchors=face_anchors,
                observation=observation,
                observed_at=observed_at,
            )
        return observation
