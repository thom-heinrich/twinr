# CHANGELOG: 2026-03-28
# BUG-1: observe_gesture() and observe_gesture_from_frame() now fail closed with health-only observations instead of leaking raw exceptions.
# BUG-2: Live gesture frame timestamps are now normalized per lane so wall-clock jumps or out-of-order helper frames cannot poison downstream live/video recognizers.
# BUG-3: observe_attention() no longer hides optional RGB capture failures behind ai_ready=True/error=None; degraded anchor quality is surfaced explicitly.
# BUG-4: Attention and gesture failure branches now perform best-effort runtime cleanup so broken camera/runtime state does not persist across later calls.
# BUG-5: Non-gesture runtime recovery can now preserve the dedicated live-gesture pipeline instead of forcing recognizer cold-starts on later gesture ticks.
# SEC-1: Shape-aware externally supplied frames are now shape/channel/size bounded before entering MediaPipe/OpenCV stacks, while opaque test/mocked frame carriers remain compatibility-allowed.
# IMP-1: All public observation entrypoints now emit bounded forensic spans/events with consistent degradation semantics for edge debugging.
# IMP-2: Public entrypoints now attach richer fast-path degradation detail so operators can distinguish local-camera, helper-frame, and RGB-degraded outcomes.
# IMP-3: Failure cleanup and health propagation are centralized to match 2026 edge-AI graceful-degradation expectations on resource-constrained embodied systems.

# mypy: disable-error-code=attr-defined
"""Public observation entrypoints for the local AI-camera adapter."""

from __future__ import annotations

import math
from typing import Any

from twinr.agent.workflows.forensics import workflow_event, workflow_span

from ..models import AICameraObservation
from .common import LOGGER

logger = LOGGER


class AICameraAdapterObserveMixin:
    """Expose the public bounded observation surfaces."""

    _DEFAULT_EXTERNAL_FRAME_MAX_PIXELS = 8_294_400  # 4K UHD upper bound for helper-frame ingress on Pi-class devices.
    _DEFAULT_EXTERNAL_FRAME_MAX_BYTES = 33_554_432  # 32 MiB hard ceiling for externally supplied RGB buffers.
    _LIVE_FRAME_MIN_STEP_S = 0.001

    def _best_effort_reset_runtime_state_locked(
        self,
        *,
        close_pipeline: bool,
        close_live_gesture_pipeline: bool | None = None,
        clear_pose: bool,
        clear_motion: bool,
    ) -> None:
        """Reset adapter runtime state without letting cleanup mask the primary failure."""

        try:
            self._reset_runtime_state_locked(
                close_pipeline=close_pipeline,
                close_live_gesture_pipeline=close_live_gesture_pipeline,
                clear_pose=clear_pose,
                clear_motion=clear_motion,
            )
        except Exception:  # pragma: no cover - cleanup depends on local runtime state.
            logger.debug("Local AI camera runtime cleanup failed.", exc_info=True)

    def _best_effort_close_live_gesture_pipeline_locked(self) -> None:
        """Close the live gesture pipeline without masking the original failure."""

        try:
            self._safe_close_live_gesture_pipeline_locked()
        except Exception:  # pragma: no cover - cleanup depends on local runtime state.
            logger.debug("Local AI camera live gesture cleanup failed.", exc_info=True)

    def _best_effort_close_runtime_locked(self) -> None:
        """Close the detection runtime without masking the original failure."""

        try:
            self._safe_close_runtime_locked()
        except Exception:  # pragma: no cover - cleanup depends on local runtime state.
            logger.debug("Local AI camera runtime close failed.", exc_info=True)

    def _normalize_live_frame_at(self, *, lane: str, candidate_s: float | None) -> float:
        """Clamp live/video timestamps to a strictly monotonic per-lane sequence."""

        resolved = self._coerce_observed_at(candidate_s)
        if not math.isfinite(resolved):
            resolved = self._now()

        state = getattr(self, "_last_live_frame_at_s_by_lane", None)
        if not isinstance(state, dict):
            state = {}
            self._last_live_frame_at_s_by_lane = state

        previous = state.get(lane)
        if isinstance(previous, (int, float)) and resolved <= float(previous):
            resolved = float(previous) + self._LIVE_FRAME_MIN_STEP_S
        state[lane] = resolved
        return resolved

    def _external_frame_limits(self) -> tuple[int, int]:
        """Return the configured helper-frame ingress budget."""

        max_pixels = getattr(self, "_DEFAULT_EXTERNAL_FRAME_MAX_PIXELS", self._DEFAULT_EXTERNAL_FRAME_MAX_PIXELS)
        max_bytes = getattr(self, "_DEFAULT_EXTERNAL_FRAME_MAX_BYTES", self._DEFAULT_EXTERNAL_FRAME_MAX_BYTES)
        try:
            max_pixels = int(max_pixels)
        except Exception:
            max_pixels = self._DEFAULT_EXTERNAL_FRAME_MAX_PIXELS
        try:
            max_bytes = int(max_bytes)
        except Exception:
            max_bytes = self._DEFAULT_EXTERNAL_FRAME_MAX_BYTES
        if max_pixels <= 0:
            max_pixels = self._DEFAULT_EXTERNAL_FRAME_MAX_PIXELS
        if max_bytes <= 0:
            max_bytes = self._DEFAULT_EXTERNAL_FRAME_MAX_BYTES
        return max_pixels, max_bytes

    def _validate_external_frame_rgb(
        self,
        *,
        frame_rgb: Any | None,
        allow_none: bool,
    ) -> str | None:
        """Validate externally supplied RGB frames before they hit heavy local CV stacks."""

        if frame_rgb is None:
            return None if allow_none else "missing_frame_rgb"

        shape = getattr(frame_rgb, "shape", None)
        if shape is None:
            return None
        if not isinstance(shape, tuple) or len(shape) not in (2, 3):
            return "invalid_frame_shape"

        try:
            height = int(shape[0])
            width = int(shape[1])
        except Exception:
            return "invalid_frame_shape"

        if height <= 0 or width <= 0:
            return "invalid_frame_shape"

        if len(shape) == 3:
            try:
                channels = int(shape[2])
            except Exception:
                return "invalid_frame_shape"
            if channels not in (1, 3, 4):
                return "invalid_frame_channels"

        max_pixels, max_bytes = self._external_frame_limits()
        if height * width > max_pixels:
            return "frame_too_large"

        nbytes = getattr(frame_rgb, "nbytes", None)
        if nbytes is not None:
            try:
                if int(nbytes) > max_bytes:
                    return "frame_too_large"
            except Exception:
                return "invalid_frame_shape"

        return None

    def _record_attention_debug_details(
        self,
        *,
        source: str,
        frame_error: str | None,
        pipeline_error: str | None = None,
    ) -> None:
        """Persist the most recent attention-path resolution for operators."""

        details = dict(self._last_attention_debug_details or {})
        details.update(
            {
                "mode": "attention_fast",
                "resolved_source": source,
                "pipeline_error": pipeline_error,
                "frame_error": frame_error,
                "attention_pipeline_note": "attention_fast_path_skips_pose_and_hand_inference",
                "degraded": bool(frame_error or pipeline_error),
            }
        )
        self._last_attention_debug_details = details

    def observe(self) -> AICameraObservation:
        """Capture one local IMX500 observation or one explicit health failure."""

        lock_timeout_s = self._lock_timeout_s()  # AUDIT-FIX(#4): Normalize misconfigured durations before lock acquisition.
        if not self._lock.acquire(timeout=lock_timeout_s):
            return self._health_only_observation(
                observed_at=self._now(),  # AUDIT-FIX(#8): Timestamp lock timeouts at the actual failure point.
                online=True,
                ready=False,
                ai_ready=False,
                error="camera_lock_timeout",
            )
        observed_at = self._now()  # AUDIT-FIX(#8): Capture observation time after the wait on the adapter lock.
        observed_monotonic = self._monotonic_now()  # AUDIT-FIX(#4): Internal freshness/motion logic must use monotonic time.
        self._last_attention_debug_details = None
        try:
            with workflow_span(
                name="camera_adapter_observe",
                kind="io",
                details={"observed_at": round(float(observed_at), 6)},
            ):
                try:
                    runtime = self._load_detection_runtime()
                except Exception as exc:  # pragma: no cover - depends on local environment.
                    code = self._classify_error(exc)
                    logger.warning("Local AI camera runtime load failed with %s.", code)
                    logger.debug("Local AI camera runtime load exception details.", exc_info=True)
                    workflow_event(
                        kind="exception",
                        msg="camera_adapter_runtime_load_failed",
                        level="ERROR",
                        details={"error_type": type(exc).__name__, "error_code": code},
                    )
                    self._best_effort_reset_runtime_state_locked(
                        close_pipeline=True,
                        close_live_gesture_pipeline=False,
                        clear_pose=True,
                        clear_motion=True,
                    )
                    return self._health_only_observation(
                        observed_at=observed_at,
                        online=False,
                        ready=False,
                        ai_ready=False,
                        error=code,
                    )
                online_error = self._probe_online(runtime)
                if online_error is not None:
                    logger.warning("Local AI camera online probe failed with %s.", online_error)
                    workflow_event(
                        kind="branch",
                        msg="camera_adapter_camera_offline",
                        details={"online_error": online_error},
                    )
                    self._best_effort_reset_runtime_state_locked(
                        close_pipeline=True,
                        close_live_gesture_pipeline=False,
                        clear_pose=True,
                        clear_motion=True,
                    )
                    return self._health_only_observation(
                        observed_at=observed_at,
                        online=False,
                        ready=False,
                        ai_ready=False,
                        error=online_error,
                    )

                detection = self._capture_detection(runtime, observed_at=observed_at)
                detection = self._coerce_detection_result(detection)
                frame_rgb = None
                frame_error = None
                if self._needs_rgb_frame_for_observation(detection=detection):
                    frame_rgb, frame_error = self._capture_optional_rgb_frame(
                        runtime,
                        observed_at=observed_at,
                    )
                pose_result, pose_error = self._resolve_pose(
                    runtime,
                    observed_at=observed_at,
                    observed_monotonic=observed_monotonic,
                    detection=detection,
                    frame_rgb=frame_rgb,
                    frame_error=frame_error,
                )
                detection, face_anchors = self._supplement_visible_persons_with_face_anchors(
                    detection=detection,
                    frame_rgb=frame_rgb,
                )
                observation = self._compose_observation(
                    observed_at=observed_at,
                    observed_monotonic=observed_monotonic,
                    detection=detection,
                    pose=pose_result,
                    pose_error=pose_error,
                    face_anchors=None if pose_result is not None else face_anchors,
                )
                return self._with_health(
                    observation,
                    online=True,
                    ready=True,
                    ai_ready=(pose_error is None),
                    error=pose_error,
                    frame_at=observed_at,
                )
        except Exception as exc:  # pragma: no cover - hardware and library behavior are environment-dependent.
            code = self._classify_error(exc)
            logger.warning("Local AI camera observation failed with %s.", code)
            logger.debug("Local AI camera observation exception details.", exc_info=True)
            workflow_event(
                kind="exception",
                msg="camera_adapter_observe_failed",
                level="ERROR",
                details={"error_type": type(exc).__name__, "error_code": code},
            )
            self._best_effort_reset_runtime_state_locked(
                close_pipeline=True,
                close_live_gesture_pipeline=False,
                clear_pose=True,
                clear_motion=True,
            )
            return self._health_only_observation(
                observed_at=observed_at,
                online=True,
                ready=False,
                ai_ready=False,
                error=code,
            )
        finally:
            self._lock.release()

    def observe_attention(self) -> AICameraObservation:
        """Capture one cheap person/anchor-only observation for HDMI eye-follow.

        The local HDMI attention loop must not pay for the full MediaPipe
        gesture stack on every refresh. This path keeps the same IMX500 person
        detection and optional face-anchor supplementation, but deliberately
        skips expensive pose/gesture inference so eye-follow stays reactive even
        while explicit hand-symbol tuning changes elsewhere.
        """

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
        try:
            with workflow_span(
                name="camera_adapter_observe_attention",
                kind="io",
                details={"observed_at": round(float(observed_at), 6), "source": "local_camera"},
            ):
                try:
                    runtime = self._load_detection_runtime()
                except Exception as exc:  # pragma: no cover - depends on local environment.
                    code = self._classify_error(exc)
                    logger.warning("Local AI camera attention runtime load failed with %s.", code)
                    logger.debug("Local AI camera attention runtime load exception details.", exc_info=True)
                    workflow_event(
                        kind="exception",
                        msg="camera_adapter_attention_runtime_load_failed",
                        level="ERROR",
                        details={"error_type": type(exc).__name__, "error_code": code},
                    )
                    self._record_attention_debug_details(
                        source="local_camera",
                        frame_error=None,
                        pipeline_error=code,
                    )
                    self._best_effort_reset_runtime_state_locked(
                        close_pipeline=True,
                        close_live_gesture_pipeline=False,
                        clear_pose=False,
                        clear_motion=False,
                    )
                    return self._health_only_observation(
                        observed_at=observed_at,
                        online=False,
                        ready=False,
                        ai_ready=False,
                        error=code,
                    )
                online_error = self._probe_online(runtime)
                if online_error is not None:
                    logger.warning("Local AI camera attention online probe failed with %s.", online_error)
                    workflow_event(
                        kind="branch",
                        msg="camera_adapter_attention_camera_offline",
                        details={"online_error": online_error},
                    )
                    self._record_attention_debug_details(
                        source="local_camera",
                        frame_error=None,
                        pipeline_error=online_error,
                    )
                    self._best_effort_reset_runtime_state_locked(
                        close_pipeline=True,
                        close_live_gesture_pipeline=False,
                        clear_pose=False,
                        clear_motion=False,
                    )
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
                frame_rgb = None
                frame_error = None
                if self._needs_rgb_frame_for_attention(detection=detection):
                    frame_rgb, frame_error = self._capture_optional_rgb_frame(
                        runtime,
                        observed_at=observed_at,
                    )
                    if frame_error is not None:
                        workflow_event(
                            kind="branch",
                            msg="camera_adapter_attention_rgb_degraded",
                            details={"frame_error": frame_error},
                        )
                observation = self._build_attention_observation_locked(
                    detection=detection,
                    frame_rgb=frame_rgb,
                    observed_at=observed_at,
                    observed_monotonic=observed_monotonic,
                )
                # BREAKING: Attention observations now surface optional RGB acquisition failure as degraded health
                # instead of silently reporting ai_ready=True/error=None.
                self._record_attention_debug_details(
                    source="local_camera",
                    frame_error=frame_error,
                    pipeline_error=None,
                )
                return self._with_health(
                    observation,
                    online=True,
                    ready=True,
                    ai_ready=(frame_error is None),
                    error=frame_error,
                    frame_at=observed_at,
                )
        except Exception as exc:  # pragma: no cover - hardware and library behavior are environment-dependent.
            code = self._classify_error(exc)
            logger.warning("Local AI camera attention observation failed with %s.", code)
            logger.debug("Local AI camera attention observation exception details.", exc_info=True)
            workflow_event(
                kind="exception",
                msg="camera_adapter_attention_failed",
                level="ERROR",
                details={"error_type": type(exc).__name__, "error_code": code},
            )
            self._record_attention_debug_details(
                source="local_camera",
                frame_error=None,
                pipeline_error=code,
            )
            self._best_effort_reset_runtime_state_locked(
                close_pipeline=True,
                close_live_gesture_pipeline=False,
                clear_pose=False,
                clear_motion=False,
            )
            return self._health_only_observation(
                observed_at=observed_at,
                online=True,
                ready=False,
                ai_ready=False,
                error=code,
            )
        finally:
            self._lock.release()

    def observe_attention_stream(self) -> AICameraObservation:
        """Capture one explicit live-stream attention observation for HDMI eye-follow."""

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
        try:
            with workflow_span(
                name="camera_adapter_observe_attention_stream",
                kind="io",
                details={"observed_at": round(float(observed_at), 6), "source": "local_camera_stream"},
            ):
                try:
                    runtime = self._load_detection_runtime()
                except Exception as exc:  # pragma: no cover - depends on local environment.
                    code = self._classify_error(exc)
                    logger.warning("Local AI camera attention stream runtime load failed with %s.", code)
                    logger.debug("Local AI camera attention stream runtime load exception details.", exc_info=True)
                    workflow_event(
                        kind="exception",
                        msg="camera_adapter_attention_stream_runtime_load_failed",
                        level="ERROR",
                        details={"error_type": type(exc).__name__, "error_code": code},
                    )
                    self._record_attention_debug_details(
                        source="local_camera_stream",
                        frame_error=None,
                        pipeline_error=code,
                    )
                    if isinstance(self._last_attention_debug_details, dict):
                        self._last_attention_debug_details["mode"] = "attention_stream"
                        self._last_attention_debug_details["stream_mode"] = "attention_stream"
                    self._best_effort_reset_runtime_state_locked(
                        close_pipeline=True,
                        close_live_gesture_pipeline=False,
                        clear_pose=False,
                        clear_motion=False,
                    )
                    return self._health_only_observation(
                        observed_at=observed_at,
                        online=False,
                        ready=False,
                        ai_ready=False,
                        error=code,
                    )
                online_error = self._probe_online(runtime)
                if online_error is not None:
                    logger.warning("Local AI camera attention stream online probe failed with %s.", online_error)
                    workflow_event(
                        kind="branch",
                        msg="camera_adapter_attention_stream_camera_offline",
                        details={"online_error": online_error},
                    )
                    self._record_attention_debug_details(
                        source="local_camera_stream",
                        frame_error=None,
                        pipeline_error=online_error,
                    )
                    if isinstance(self._last_attention_debug_details, dict):
                        self._last_attention_debug_details["mode"] = "attention_stream"
                        self._last_attention_debug_details["stream_mode"] = "attention_stream"
                    self._best_effort_reset_runtime_state_locked(
                        close_pipeline=True,
                        close_live_gesture_pipeline=False,
                        clear_pose=False,
                        clear_motion=False,
                    )
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
                frame_rgb = None
                frame_error = None
                if self._needs_rgb_frame_for_attention(detection=detection):
                    frame_rgb, frame_error = self._capture_optional_rgb_frame(
                        runtime,
                        observed_at=observed_at,
                    )
                    if frame_error is not None:
                        workflow_event(
                            kind="branch",
                            msg="camera_adapter_attention_stream_rgb_degraded",
                            details={"frame_error": frame_error},
                        )
                observation = self._build_attention_observation_locked(
                    detection=detection,
                    frame_rgb=frame_rgb,
                    observed_at=observed_at,
                    observed_monotonic=observed_monotonic,
                    stream_lane="local_attention_stream",
                )
                return self._with_health(
                    observation,
                    online=True,
                    ready=True,
                    ai_ready=(frame_error is None),
                    error=frame_error,
                    frame_at=observed_at,
                )
        except Exception as exc:  # pragma: no cover - hardware and library behavior are environment-dependent.
            code = self._classify_error(exc)
            logger.warning("Local AI camera attention stream observation failed with %s.", code)
            logger.debug("Local AI camera attention stream observation exception details.", exc_info=True)
            workflow_event(
                kind="exception",
                msg="camera_adapter_attention_stream_failed",
                level="ERROR",
                details={"error_type": type(exc).__name__, "error_code": code},
            )
            self._record_attention_debug_details(
                source="local_camera_stream",
                frame_error=None,
                pipeline_error=code,
            )
            if isinstance(self._last_attention_debug_details, dict):
                self._last_attention_debug_details["mode"] = "attention_stream"
                self._last_attention_debug_details["stream_mode"] = "attention_stream"
            self._best_effort_reset_runtime_state_locked(
                close_pipeline=True,
                close_live_gesture_pipeline=False,
                clear_pose=False,
                clear_motion=False,
            )
            return self._health_only_observation(
                observed_at=observed_at,
                online=True,
                ready=False,
                ai_ready=False,
                error=code,
            )
        finally:
            self._lock.release()

    def observe_attention_from_frame(
        self,
        *,
        detection: Any,
        frame_rgb: Any | None,
        observed_at: float | None = None,
        frame_at: float | None = None,
    ) -> AICameraObservation:
        """Process one externally supplied frame plus person boxes for fast attention.

        This entrypoint lets the main Pi reuse the same bounded attention logic
        even when the physical camera lives on a helper Pi. The caller provides
        the IMX500-style detection facts plus the matching RGB frame; the
        adapter keeps motion caches, face-anchor supplementation, and debug
        payloads local to the main runtime.
        """

        lock_timeout_s = self._lock_timeout_s()
        if not self._lock.acquire(timeout=lock_timeout_s):
            return self._health_only_observation(
                observed_at=self._now(),
                online=True,
                ready=False,
                ai_ready=False,
                error="camera_lock_timeout",
            )
        resolved_observed_at = self._coerce_observed_at(observed_at)
        resolved_frame_at = resolved_observed_at if frame_at is None else self._coerce_observed_at(frame_at)
        observed_monotonic = self._monotonic_now()
        self._last_attention_debug_details = None
        try:
            with workflow_span(
                name="camera_adapter_observe_attention_from_frame",
                kind="io",
                details={"observed_at": round(float(resolved_observed_at), 6), "source": "external_frame"},
            ):
                detection = self._coerce_detection_result(detection)
                needs_frame = self._needs_rgb_frame_for_attention(detection=detection)
                frame_error = self._validate_external_frame_rgb(
                    frame_rgb=frame_rgb,
                    allow_none=not needs_frame,
                )
                if frame_error is not None:
                    workflow_event(
                        kind="branch",
                        msg="camera_adapter_attention_external_frame_degraded",
                        details={"frame_error": frame_error},
                    )
                    frame_rgb = None
                observation = self._build_attention_observation_locked(
                    detection=detection,
                    frame_rgb=frame_rgb,
                    observed_at=resolved_observed_at,
                    observed_monotonic=observed_monotonic,
                )
                self._record_attention_debug_details(
                    source="external_frame",
                    frame_error=frame_error,
                    pipeline_error=None,
                )
                return self._with_health(
                    observation,
                    online=True,
                    ready=True,
                    ai_ready=(frame_error is None),
                    error=frame_error,
                    frame_at=resolved_frame_at,
                )
        except Exception as exc:  # pragma: no cover - transport/runtime coupling is environment-dependent.
            code = self._classify_error(exc)
            logger.warning("External AI camera attention observation failed with %s.", code)
            logger.debug("External AI camera attention observation exception details.", exc_info=True)
            workflow_event(
                kind="exception",
                msg="camera_adapter_attention_from_frame_failed",
                level="ERROR",
                details={"error_type": type(exc).__name__, "error_code": code},
            )
            self._record_attention_debug_details(
                source="external_frame",
                frame_error=None,
                pipeline_error=code,
            )
            return self._health_only_observation(
                observed_at=resolved_observed_at,
                online=True,
                ready=False,
                ai_ready=False,
                error=code,
            )
        finally:
            self._lock.release()

    def observe_attention_from_frame_stream(
        self,
        *,
        detection: Any,
        frame_rgb: Any | None,
        observed_at: float | None = None,
        frame_at: float | None = None,
    ) -> AICameraObservation:
        """Process one external frame through the explicit live-stream attention lane."""

        lock_timeout_s = self._lock_timeout_s()
        if not self._lock.acquire(timeout=lock_timeout_s):
            return self._health_only_observation(
                observed_at=self._now(),
                online=True,
                ready=False,
                ai_ready=False,
                error="camera_lock_timeout",
            )
        resolved_observed_at = self._coerce_observed_at(observed_at)
        resolved_frame_at = resolved_observed_at if frame_at is None else self._coerce_observed_at(frame_at)
        observed_monotonic = self._monotonic_now()
        self._last_attention_debug_details = None
        try:
            with workflow_span(
                name="camera_adapter_observe_attention_from_frame_stream",
                kind="io",
                details={"observed_at": round(float(resolved_observed_at), 6), "source": "external_frame_stream"},
            ):
                detection = self._coerce_detection_result(detection)
                needs_frame = self._needs_rgb_frame_for_attention(detection=detection)
                frame_error = self._validate_external_frame_rgb(
                    frame_rgb=frame_rgb,
                    allow_none=not needs_frame,
                )
                if frame_error is not None:
                    workflow_event(
                        kind="branch",
                        msg="camera_adapter_attention_stream_external_frame_degraded",
                        details={"frame_error": frame_error},
                    )
                    frame_rgb = None
                observation = self._build_attention_observation_locked(
                    detection=detection,
                    frame_rgb=frame_rgb,
                    observed_at=resolved_observed_at,
                    observed_monotonic=observed_monotonic,
                    stream_lane="external_attention_stream",
                )
                return self._with_health(
                    observation,
                    online=True,
                    ready=True,
                    ai_ready=(frame_error is None),
                    error=frame_error,
                    frame_at=resolved_frame_at,
                )
        except Exception as exc:  # pragma: no cover - transport/runtime coupling is environment-dependent.
            code = self._classify_error(exc)
            logger.warning("External AI camera attention stream observation failed with %s.", code)
            logger.debug("External AI camera attention stream observation exception details.", exc_info=True)
            workflow_event(
                kind="exception",
                msg="camera_adapter_attention_from_frame_stream_failed",
                level="ERROR",
                details={"error_type": type(exc).__name__, "error_code": code},
            )
            self._record_attention_debug_details(
                source="external_frame_stream",
                frame_error=None,
                pipeline_error=code,
            )
            if isinstance(self._last_attention_debug_details, dict):
                self._last_attention_debug_details["mode"] = "attention_stream"
                self._last_attention_debug_details["stream_mode"] = "attention_stream"
            return self._health_only_observation(
                observed_at=resolved_observed_at,
                online=True,
                ready=False,
                ai_ready=False,
                error=code,
            )
        finally:
            self._lock.release()

    def observe_gesture(self, *, gesture_fast_path: bool = False) -> AICameraObservation:
        """Capture one dedicated live-stream gesture observation for HDMI emoji ack.

        This path intentionally bypasses the general pose/social observation
        pipeline. It reuses the existing RGB preview session, but only feeds the
        frame into the thin live-stream gesture recognizers so user-facing
        symbol acknowledgement stays responsive and cannot regress eye-follow.
        """

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
        gesture_frame_at = self._normalize_live_frame_at(lane="gesture", candidate_s=observed_at)
        try:
            with workflow_span(
                name="camera_adapter_observe_gesture",
                kind="io",
                details={
                    "observed_at": round(float(observed_at), 6),
                    "frame_at": round(float(gesture_frame_at), 6),
                    "source": "local_camera",
                    "gesture_fast_path": bool(gesture_fast_path),
                },
            ):
                try:
                    with workflow_span(
                        name="camera_adapter_gesture_load_runtime",
                        kind="io",
                    ):
                        runtime = self._load_detection_runtime()
                except Exception as exc:  # pragma: no cover - depends on local environment.
                    code = self._classify_error(exc)
                    logger.warning("Local AI camera gesture runtime load failed with %s.", code)
                    logger.debug("Local AI camera gesture runtime load exception details.", exc_info=True)
                    workflow_event(
                        kind="exception",
                        msg="camera_adapter_gesture_runtime_load_failed",
                        level="ERROR",
                        details={"error_type": type(exc).__name__, "error_code": code},
                    )
                    self._last_gesture_debug_details = {
                        "resolved_source": "local_camera",
                        "pipeline_error": code,
                        "gesture_fast_path": bool(gesture_fast_path),
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
                with workflow_span(
                    name="camera_adapter_gesture_online_probe",
                    kind="io",
                ):
                    online_error = self._probe_online(runtime)
                if online_error is not None:
                    logger.warning("Local AI camera gesture online probe failed with %s.", online_error)
                    workflow_event(
                        kind="branch",
                        msg="camera_adapter_gesture_camera_offline",
                        details={"online_error": online_error},
                        reason={
                            "selected": {
                                "id": "camera_offline",
                                "justification": "The adapter online probe failed, so the gesture lane must fail closed for this frame.",
                                "expected_outcome": "Return a health-only observation without running gesture inference.",
                            },
                            "options": [
                                {"id": "camera_online", "summary": "Continue with gesture inference."},
                                {"id": "camera_offline", "summary": "Return a health-only observation."},
                            ],
                            "confidence": "forensic",
                            "guardrails": ["camera_online_required"],
                            "kpi_impact_estimate": {"latency": "low", "gesture_output": "none"},
                        },
                    )
                    self._last_gesture_debug_details = {
                        "resolved_source": "camera_offline",
                        "pipeline_error": online_error,
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
                with workflow_span(
                    name="camera_adapter_gesture_capture_detection",
                    kind="io",
                ):
                    detection = self._coerce_detection_result(
                        self._capture_detection(runtime, observed_at=observed_at)
                    )
                with workflow_span(
                    name="camera_adapter_gesture_capture_rgb",
                    kind="io",
                ):
                    frame_rgb = self._capture_rgb_frame(runtime, observed_at=observed_at)
                return self._build_gesture_observation_locked(
                    runtime=runtime,
                    observed_at=observed_at,
                    observed_monotonic=observed_monotonic,
                    detection=detection,
                    frame_rgb=frame_rgb,
                    frame_at=gesture_frame_at,
                    gesture_fast_path=gesture_fast_path,
                )
        except Exception as exc:  # pragma: no cover - hardware and library behavior are environment-dependent.
            code = self._classify_error(exc)
            logger.warning("Local AI camera gesture observation failed with %s.", code)
            logger.debug("Local AI camera gesture observation exception details.", exc_info=True)
            workflow_event(
                kind="exception",
                msg="camera_adapter_gesture_failed",
                level="ERROR",
                details={"error_type": type(exc).__name__, "error_code": code},
            )
            self._last_gesture_debug_details = {
                "resolved_source": "local_camera",
                "pipeline_error": code,
                "gesture_fast_path": bool(gesture_fast_path),
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

    def observe_gesture_stream(self) -> AICameraObservation:
        """Capture one explicit live-stream gesture observation for HDMI ack/wakeup."""

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
        gesture_frame_at = self._normalize_live_frame_at(lane="gesture_stream", candidate_s=observed_at)
        try:
            with workflow_span(
                name="camera_adapter_observe_gesture_stream",
                kind="io",
                details={
                    "observed_at": round(float(observed_at), 6),
                    "frame_at": round(float(gesture_frame_at), 6),
                    "source": "local_camera_stream",
                    "gesture_fast_path": True,
                },
            ):
                try:
                    with workflow_span(
                        name="camera_adapter_gesture_stream_load_runtime",
                        kind="io",
                    ):
                        runtime = self._load_detection_runtime()
                except Exception as exc:  # pragma: no cover - depends on local environment.
                    code = self._classify_error(exc)
                    logger.warning("Local AI camera gesture stream runtime load failed with %s.", code)
                    logger.debug("Local AI camera gesture stream runtime load exception details.", exc_info=True)
                    workflow_event(
                        kind="exception",
                        msg="camera_adapter_gesture_stream_runtime_load_failed",
                        level="ERROR",
                        details={"error_type": type(exc).__name__, "error_code": code},
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
                with workflow_span(
                    name="camera_adapter_gesture_stream_online_probe",
                    kind="io",
                ):
                    online_error = self._probe_online(runtime)
                if online_error is not None:
                    logger.warning("Local AI camera gesture stream online probe failed with %s.", online_error)
                    workflow_event(
                        kind="branch",
                        msg="camera_adapter_gesture_stream_camera_offline",
                        details={"online_error": online_error},
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
                with workflow_span(
                    name="camera_adapter_gesture_stream_capture_detection",
                    kind="io",
                ):
                    detection = self._coerce_detection_result(
                        self._capture_detection(runtime, observed_at=observed_at)
                    )
                with workflow_span(
                    name="camera_adapter_gesture_stream_capture_rgb",
                    kind="io",
                ):
                    frame_rgb = self._capture_rgb_frame(runtime, observed_at=observed_at)
                return self._build_gesture_observation_locked(
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
        except Exception as exc:  # pragma: no cover - hardware and library behavior are environment-dependent.
            code = self._classify_error(exc)
            logger.warning("Local AI camera gesture stream observation failed with %s.", code)
            logger.debug("Local AI camera gesture stream observation exception details.", exc_info=True)
            workflow_event(
                kind="exception",
                msg="camera_adapter_gesture_stream_failed",
                level="ERROR",
                details={"error_type": type(exc).__name__, "error_code": code},
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

    def observe_gesture_from_frame(
        self,
        *,
        detection: Any,
        frame_rgb: Any,
        observed_at: float | None = None,
        frame_at: float | None = None,
        allow_pose_fallback: bool = True,
        gesture_fast_path: bool = False,
    ) -> AICameraObservation:
        """Run the hot gesture lane on an externally supplied RGB frame.

        The helper Pi can provide the RGB frame and IMX500 person boxes while
        the main Pi executes the expensive MediaPipe gesture work locally. This
        preserves the existing gesture heuristics and caches without requiring
        local camera hardware on the main board.
        """

        lock_timeout_s = self._lock_timeout_s()
        if not self._lock.acquire(timeout=lock_timeout_s):
            return self._health_only_observation(
                observed_at=self._now(),
                online=True,
                ready=False,
                ai_ready=False,
                error="camera_lock_timeout",
            )
        resolved_observed_at = self._coerce_observed_at(observed_at)
        normalized_frame_at = self._normalize_live_frame_at(
            lane="gesture",
            candidate_s=resolved_observed_at if frame_at is None else frame_at,
        )
        observed_monotonic = self._monotonic_now()
        try:
            with workflow_span(
                name="camera_adapter_observe_gesture_from_frame",
                kind="io",
                details={
                    "observed_at": round(float(resolved_observed_at), 6),
                    "frame_at": round(float(normalized_frame_at), 6),
                    "source": "external_frame",
                    "gesture_fast_path": bool(gesture_fast_path),
                },
            ):
                frame_error = self._validate_external_frame_rgb(
                    frame_rgb=frame_rgb,
                    allow_none=False,
                )
                if frame_error is not None:
                    # BREAKING: Oversized or malformed helper frames now fail closed with an explicit
                    # health error instead of reaching deeper CV stacks and exhausting Pi resources.
                    workflow_event(
                        kind="branch",
                        msg="camera_adapter_gesture_external_frame_rejected",
                        details={"frame_error": frame_error},
                    )
                    self._last_gesture_debug_details = {
                        "resolved_source": "external_frame",
                        "pipeline_error": frame_error,
                        "gesture_fast_path": bool(gesture_fast_path),
                    }
                    return self._health_only_observation(
                        observed_at=resolved_observed_at,
                        online=True,
                        ready=False,
                        ai_ready=False,
                        error=frame_error,
                    )
                return self._build_gesture_observation_locked(
                    runtime={},
                    observed_at=resolved_observed_at,
                    observed_monotonic=observed_monotonic,
                    detection=self._coerce_detection_result(detection),
                    frame_rgb=frame_rgb,
                    frame_at=normalized_frame_at,
                    allow_pose_fallback=allow_pose_fallback,
                    gesture_fast_path=gesture_fast_path,
                )
        except Exception as exc:  # pragma: no cover - transport/runtime coupling is environment-dependent.
            code = self._classify_error(exc)
            logger.warning("External AI camera gesture observation failed with %s.", code)
            logger.debug("External AI camera gesture observation exception details.", exc_info=True)
            workflow_event(
                kind="exception",
                msg="camera_adapter_gesture_from_frame_failed",
                level="ERROR",
                details={"error_type": type(exc).__name__, "error_code": code},
            )
            self._last_gesture_debug_details = {
                "resolved_source": "external_frame",
                "pipeline_error": code,
                "gesture_fast_path": bool(gesture_fast_path),
            }
            self._best_effort_close_live_gesture_pipeline_locked()
            return self._health_only_observation(
                observed_at=resolved_observed_at,
                online=True,
                ready=False,
                ai_ready=False,
                error=code,
            )
        finally:
            self._lock.release()

    def observe_gesture_from_frame_stream(
        self,
        *,
        detection: Any,
        frame_rgb: Any,
        observed_at: float | None = None,
        frame_at: float | None = None,
        allow_pose_fallback: bool = False,
    ) -> AICameraObservation:
        """Run the explicit gesture stream lane on an externally supplied RGB frame."""

        lock_timeout_s = self._lock_timeout_s()
        if not self._lock.acquire(timeout=lock_timeout_s):
            return self._health_only_observation(
                observed_at=self._now(),
                online=True,
                ready=False,
                ai_ready=False,
                error="camera_lock_timeout",
            )
        resolved_observed_at = self._coerce_observed_at(observed_at)
        normalized_frame_at = self._normalize_live_frame_at(
            lane="gesture_stream_external",
            candidate_s=resolved_observed_at if frame_at is None else frame_at,
        )
        observed_monotonic = self._monotonic_now()
        try:
            with workflow_span(
                name="camera_adapter_observe_gesture_from_frame_stream",
                kind="io",
                details={
                    "observed_at": round(float(resolved_observed_at), 6),
                    "frame_at": round(float(normalized_frame_at), 6),
                    "source": "external_frame_stream",
                    "gesture_fast_path": True,
                },
            ):
                frame_error = self._validate_external_frame_rgb(
                    frame_rgb=frame_rgb,
                    allow_none=False,
                )
                if frame_error is not None:
                    workflow_event(
                        kind="branch",
                        msg="camera_adapter_gesture_stream_external_frame_rejected",
                        details={"frame_error": frame_error},
                    )
                    self._last_gesture_debug_details = {
                        "resolved_source": "external_frame_stream",
                        "pipeline_error": frame_error,
                        "gesture_fast_path": True,
                        "stream_mode": "gesture_stream",
                        "gesture_stream_source": "external_frame",
                    }
                    return self._health_only_observation(
                        observed_at=resolved_observed_at,
                        online=True,
                        ready=False,
                        ai_ready=False,
                        error=frame_error,
                    )
                return self._build_gesture_observation_locked(
                    runtime={},
                    observed_at=resolved_observed_at,
                    observed_monotonic=observed_monotonic,
                    detection=self._coerce_detection_result(detection),
                    frame_rgb=frame_rgb,
                    frame_at=normalized_frame_at,
                    allow_pose_fallback=allow_pose_fallback,
                    gesture_fast_path=True,
                    stream_mode=True,
                    stream_source="external_frame",
                )
        except Exception as exc:  # pragma: no cover - transport/runtime coupling is environment-dependent.
            code = self._classify_error(exc)
            logger.warning("External AI camera gesture stream observation failed with %s.", code)
            logger.debug("External AI camera gesture stream observation exception details.", exc_info=True)
            workflow_event(
                kind="exception",
                msg="camera_adapter_gesture_from_frame_stream_failed",
                level="ERROR",
                details={"error_type": type(exc).__name__, "error_code": code},
            )
            self._last_gesture_debug_details = {
                "resolved_source": "external_frame_stream",
                "pipeline_error": code,
                "gesture_fast_path": True,
                "stream_mode": "gesture_stream",
                "gesture_stream_source": "external_frame",
            }
            self._best_effort_close_live_gesture_pipeline_locked()
            return self._health_only_observation(
                observed_at=resolved_observed_at,
                online=True,
                ready=False,
                ai_ready=False,
                error=code,
            )
        finally:
            self._lock.release()
