"""Shared productive perception-stream entrypoints for the local AI-camera adapter."""

from __future__ import annotations

from twinr.agent.workflows.forensics import workflow_event, workflow_span

from ..models import AICameraObservation
from .common import LOGGER

logger = LOGGER


class AICameraAdapterPerceptionMixin:
    """Expose one combined productive perception stream from a single capture."""

    def _lane_observation_ready(self, observation: AICameraObservation) -> bool:
        """Return whether one lane observation represents a usable camera frame."""

        return bool(observation.camera_online and observation.camera_ready)

    def _merge_perception_stream_observation_locked(
        self,
        *,
        attention_observation: AICameraObservation | None,
        gesture_observation: AICameraObservation | None,
        observed_at: float,
    ) -> AICameraObservation:
        """Merge attention and gesture lane results into one shared stream observation."""

        attention_ready = bool(
            attention_observation is not None and self._lane_observation_ready(attention_observation)
        )
        gesture_ready = bool(
            gesture_observation is not None and self._lane_observation_ready(gesture_observation)
        )
        if not attention_ready and not gesture_ready:
            return self._health_only_observation(
                observed_at=observed_at,
                online=True,
                ready=False,
                ai_ready=False,
                error="perception_stream_unavailable",
            )

        base = attention_observation if attention_ready else gesture_observation
        assert base is not None
        updates: dict[str, object] = {
            "model": "local-imx500-perception-stream",
        }
        if attention_ready and attention_observation is not None:
            updates.update(
                {
                    "looking_toward_device": attention_observation.looking_toward_device,
                    "looking_signal_state": attention_observation.looking_signal_state,
                    "looking_signal_source": attention_observation.looking_signal_source,
                    "person_near_device": attention_observation.person_near_device,
                    "engaged_with_device": attention_observation.engaged_with_device,
                    "visual_attention_score": attention_observation.visual_attention_score,
                }
            )
        if gesture_ready and gesture_observation is not None:
            updates.update(
                {
                    "hand_or_object_near_camera": gesture_observation.hand_or_object_near_camera,
                    "gesture_temporal_authoritative": gesture_observation.gesture_temporal_authoritative,
                    "gesture_event": gesture_observation.gesture_event,
                    "gesture_confidence": gesture_observation.gesture_confidence,
                    "fine_hand_gesture": gesture_observation.fine_hand_gesture,
                    "fine_hand_gesture_confidence": gesture_observation.fine_hand_gesture_confidence,
                    "gesture_activation_key": gesture_observation.gesture_activation_key,
                    "gesture_activation_token": gesture_observation.gesture_activation_token,
                    "gesture_activation_started_at": gesture_observation.gesture_activation_started_at,
                    "gesture_activation_changed_at": gesture_observation.gesture_activation_changed_at,
                    "gesture_activation_source": gesture_observation.gesture_activation_source,
                    "gesture_activation_rising": gesture_observation.gesture_activation_rising,
                }
            )
            if gesture_observation.showing_intent_likely is not None:
                updates["showing_intent_likely"] = gesture_observation.showing_intent_likely
        merged = self._observation_copy(base, **updates)
        return self._with_health(
            merged,
            online=True,
            ready=True,
            ai_ready=True,
            error=None,
            frame_at=observed_at,
        )

    def observe_perception_stream(self) -> AICameraObservation:
        """Capture one shared productive attention+gesture stream observation."""

        lock_timeout_s = self._lock_timeout_s()
        if not self._lock.acquire(timeout=lock_timeout_s):
            return self._health_only_observation(
                observed_at=self._now(),
                online=True,
                ready=False,
                ai_ready=False,
                error="camera_lock_timeout",
            )
        observed_at = self._now()
        observed_monotonic = self._monotonic_now()
        gesture_frame_at = self._normalize_live_frame_at(lane="perception_stream", candidate_s=observed_at)
        try:
            with workflow_span(
                name="camera_adapter_observe_perception_stream",
                kind="io",
                details={"observed_at": round(float(observed_at), 6), "source": "local_camera_stream"},
            ):
                try:
                    runtime = self._load_detection_runtime()
                except Exception as exc:  # pragma: no cover - depends on local environment.
                    code = self._classify_error(exc)
                    logger.warning("Local AI camera perception stream runtime load failed with %s.", code)
                    logger.debug("Local AI camera perception stream runtime load exception details.", exc_info=True)
                    workflow_event(
                        kind="exception",
                        msg="camera_adapter_perception_stream_runtime_load_failed",
                        level="ERROR",
                        details={"error_type": type(exc).__name__, "error_code": code},
                    )
                    self._record_attention_debug_details(
                        source="local_camera_stream",
                        frame_error=None,
                        pipeline_error=code,
                    )
                    self._last_gesture_debug_details = {
                        "resolved_source": "local_camera_stream",
                        "pipeline_error": code,
                        "gesture_fast_path": True,
                        "stream_mode": "gesture_stream",
                        "gesture_stream_source": "local_camera",
                    }
                    self._best_effort_close_live_gesture_pipeline_locked()
                    self._best_effort_close_runtime_locked()
                    return self._health_only_observation(
                        observed_at=observed_at,
                        online=False,
                        ready=False,
                        ai_ready=False,
                        error=code,
                    )
                online_error = self._probe_online(runtime)
                if online_error is not None:
                    logger.warning("Local AI camera perception stream online probe failed with %s.", online_error)
                    workflow_event(
                        kind="branch",
                        msg="camera_adapter_perception_stream_camera_offline",
                        details={"online_error": online_error},
                    )
                    self._record_attention_debug_details(
                        source="local_camera_stream",
                        frame_error=None,
                        pipeline_error=online_error,
                    )
                    self._last_gesture_debug_details = {
                        "resolved_source": "camera_offline",
                        "pipeline_error": online_error,
                        "stream_mode": "gesture_stream",
                        "gesture_stream_source": "local_camera",
                    }
                    self._best_effort_close_live_gesture_pipeline_locked()
                    self._best_effort_close_runtime_locked()
                    return self._health_only_observation(
                        observed_at=observed_at,
                        online=False,
                        ready=False,
                        ai_ready=False,
                        error=online_error,
                    )

                detection = self._coerce_detection_result(
                    self._capture_detection(runtime, observed_at=observed_at)
                )
                frame_rgb = self._capture_rgb_frame(runtime, observed_at=observed_at)

                attention_observation: AICameraObservation | None
                try:
                    attention_observation = self._build_attention_observation_locked(
                        detection=detection,
                        frame_rgb=frame_rgb,
                        observed_at=observed_at,
                        observed_monotonic=observed_monotonic,
                        stream_lane="local_attention_stream",
                    )
                except Exception as exc:  # pragma: no cover - environment-dependent.
                    code = self._classify_error(exc)
                    logger.warning("Local AI camera combined attention lane failed with %s.", code)
                    logger.debug("Local AI camera combined attention lane exception details.", exc_info=True)
                    workflow_event(
                        kind="exception",
                        msg="camera_adapter_perception_stream_attention_failed",
                        level="ERROR",
                        details={"error_type": type(exc).__name__, "error_code": code},
                    )
                    self._record_attention_debug_details(
                        source="local_camera_stream",
                        frame_error=None,
                        pipeline_error=code,
                    )
                    attention_observation = self._health_only_observation(
                        observed_at=observed_at,
                        online=True,
                        ready=False,
                        ai_ready=False,
                        error=code,
                    )

                gesture_observation = self._build_gesture_observation_locked(
                    runtime=runtime,
                    observed_at=observed_at,
                    observed_monotonic=observed_monotonic,
                    detection=detection,
                    frame_rgb=frame_rgb,
                    frame_at=gesture_frame_at,
                    gesture_fast_path=True,
                    stream_mode=True,
                    stream_source="local_camera",
                )
                return self._merge_perception_stream_observation_locked(
                    attention_observation=attention_observation,
                    gesture_observation=gesture_observation,
                    observed_at=observed_at,
                )
        except Exception as exc:  # pragma: no cover - hardware and library behavior are environment-dependent.
            code = self._classify_error(exc)
            logger.warning("Local AI camera perception stream observation failed with %s.", code)
            logger.debug("Local AI camera perception stream observation exception details.", exc_info=True)
            workflow_event(
                kind="exception",
                msg="camera_adapter_perception_stream_failed",
                level="ERROR",
                details={"error_type": type(exc).__name__, "error_code": code},
            )
            self._record_attention_debug_details(
                source="local_camera_stream",
                frame_error=None,
                pipeline_error=code,
            )
            self._last_gesture_debug_details = {
                "resolved_source": "local_camera_stream",
                "pipeline_error": code,
                "gesture_fast_path": True,
                "stream_mode": "gesture_stream",
                "gesture_stream_source": "local_camera",
            }
            self._best_effort_close_live_gesture_pipeline_locked()
            self._best_effort_close_runtime_locked()
            return self._health_only_observation(
                observed_at=observed_at,
                online=True,
                ready=False,
                ai_ready=False,
                error=code,
            )
        finally:
            self._lock.release()
