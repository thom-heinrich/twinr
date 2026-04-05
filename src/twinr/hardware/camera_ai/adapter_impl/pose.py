# CHANGELOG: 2026-03-28
# BUG-1: Use monotonic frame timestamps for MediaPipe-style pipelines and compatible runtime hooks so NTP / RTC wall-clock jumps do not break ordered video timestamps.
# BUG-2: Replace brittle PoseNet-only tensor / postprocess assumptions with adaptive decoding that works with legacy PoseNet call signatures and newer IMX500 post-processors such as HigherHRNet / YOLO-pose style adapters.
# BUG-3: Bound cache reuse with freshness and box-overlap checks and return defensive copies so stale or externally mutated cache state cannot silently leak across frames.
# SEC-1: Guard against malformed / hostile model outputs by rejecting non-finite values and capping the number of pose candidates processed on the Pi host.
# SEC-2: Serialize shared pose-pipeline and cache mutations with a re-entrant lock to prevent practical close/reset/use races in concurrent adapter calls.
# IMP-1: Add adaptive hook invocation for MediaPipe Tasks style pipelines and runtime managers, including timestamp_ms support for LIVE_STREAM / VIDEO task patterns.
# IMP-2: Add short degraded-mode pose fallback on transient inference failures to reduce user-visible flicker for real-time assistant interactions.
"""Pose-resolution helpers for the local AI-camera adapter."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping, Sequence
from threading import RLock
from collections.abc import Callable
from typing import Any

from ..detection import DetectionResult
from ..pose_classification import classify_body_pose, classify_gesture
from ..pose_features import attention_score, hand_near_camera, parse_keypoints, support_pose_confidence
from ..pose_selection import rank_pose_candidates
from .common import LOGGER
from .types import PoseResult

logger = LOGGER


class AICameraAdapterPoseMixin:
    """Resolve full pose frames and deterministic pose-backed fallbacks."""

    def _resolve_pose(
        self,
        runtime: dict[str, Any],
        *,
        observed_at: float,
        observed_monotonic: float,
        detection: DetectionResult,
        frame_rgb: Any | None,
        frame_error: str | None,
    ) -> tuple[PoseResult | None, str | None]:
        """Return one fresh or cached pose result for the current detection frame."""

        with self._pose_resolution_lock():
            primary_person_box = self._select_pose_tracking_box(detection)

            if detection.person_count <= 0:
                self._safe_reset_mediapipe_temporal_state_locked()
                self._clear_pose_cache_state()
                return None, None

            if self.config.pose_backend == "mediapipe":
                return self._resolve_mediapipe_pose(
                    runtime,
                    observed_at=observed_at,
                    observed_monotonic=observed_monotonic,
                    detection=detection,
                    frame_rgb=frame_rgb,
                    frame_error=frame_error,
                )

            if not self.config.pose_network_path:
                self._clear_pose_cache_state()
                return None, None

            if self._can_reuse_pose_cache(
                observed_monotonic=observed_monotonic,
                primary_person_box=primary_person_box,
            ):
                return self._clone_pose_result(self._last_pose_result), None

            try:
                pose_postprocess = self._runtime_manager.load_pose_postprocess()
            except Exception as exc:  # AUDIT-FIX(#5): Missing/broken pose postprocess assets should degrade to detection-only instead of failing the whole observation.
                code = self._classify_error(exc)
                logger.warning("Local AI camera pose postprocess load failed with %s.", code)
                logger.debug("Local AI camera pose postprocess load exception details.", exc_info=True)
                cached_pose = self._reuse_recent_pose_after_failure(
                    observed_monotonic=observed_monotonic,
                    primary_person_box=primary_person_box,
                )
                if cached_pose is not None:
                    return cached_pose, code
                self._clear_pose_cache_state()
                return None, code

            try:
                pose = self._capture_pose(
                    runtime,
                    pose_postprocess=pose_postprocess,
                    observed_at=observed_at,
                    observed_monotonic=observed_monotonic,
                    primary_person_box=primary_person_box,
                )
            except Exception as exc:  # pragma: no cover - hardware-dependent path.
                code = self._classify_error(exc)
                logger.warning("Local AI camera pose decode failed with %s.", code)
                logger.debug("Local AI camera pose decode exception details.", exc_info=True)
                cached_pose = self._reuse_recent_pose_after_failure(
                    observed_monotonic=observed_monotonic,
                    primary_person_box=primary_person_box,
                )
                if cached_pose is not None:
                    return cached_pose, code
                self._clear_pose_cache_state()
                return None, code

            self._store_pose_result(
                pose,
                observed_at=observed_at,
                observed_monotonic=observed_monotonic,
                primary_person_box=primary_person_box,
            )
            self._record_pose_cache_metadata(
                observed_at=observed_at,
                observed_monotonic=observed_monotonic,
                primary_person_box=primary_person_box,
            )
            return self._clone_pose_result(pose), None

    def _resolve_mediapipe_pose(
        self,
        runtime: dict[str, Any],
        *,
        observed_at: float,
        observed_monotonic: float,
        detection: DetectionResult,
        frame_rgb: Any | None,
        frame_error: str | None,
    ) -> tuple[PoseResult | None, str | None]:
        """Run the Pi-side MediaPipe pose and gesture path on the RGB preview frame."""

        with self._pose_resolution_lock():
            primary_person_box = self._select_pose_tracking_box(detection)
            if primary_person_box is None:
                self._safe_reset_mediapipe_temporal_state_locked()
                self._clear_pose_cache_state()
                return None, None
            if frame_error is not None:
                self._safe_close_mediapipe_pipeline_locked()
                self._clear_pose_cache_state()
                return None, frame_error

            if self._can_reuse_pose_cache(
                observed_monotonic=observed_monotonic,
                primary_person_box=primary_person_box,
            ):
                return self._clone_pose_result(self._last_pose_result), None

            frame_timestamp_ms = self._to_frame_timestamp_ms(observed_monotonic)

            try:
                if frame_rgb is None:
                    frame_rgb = self._call_with_supported_kwargs(
                        self._capture_rgb_frame,
                        runtime,
                        observed_at=observed_at,
                        observed_monotonic=observed_monotonic,
                        timestamp_ms=frame_timestamp_ms,
                        frame_timestamp_ms=frame_timestamp_ms,
                    )
                pipeline = self._ensure_mediapipe_pipeline()
                result = self._call_mediapipe_analyze(
                    pipeline,
                    frame_rgb=frame_rgb,
                    observed_at=observed_at,
                    observed_monotonic=observed_monotonic,
                    primary_person_box=primary_person_box,
                )
            except Exception as exc:  # pragma: no cover - depends on Pi runtime and model assets.
                code = self._classify_error(exc)
                logger.warning("Local AI camera MediaPipe inference failed with %s.", code)
                logger.debug("Local AI camera MediaPipe inference exception details.", exc_info=True)
                self._safe_close_mediapipe_pipeline_locked()
                cached_pose = self._reuse_recent_pose_after_failure(
                    observed_monotonic=observed_monotonic,
                    primary_person_box=primary_person_box,
                )
                if cached_pose is not None:
                    return cached_pose, code
                self._clear_pose_cache_state()
                return None, code

            pose = self._build_pose_result(
                body_pose=getattr(result, "body_pose", None),
                pose_confidence=getattr(result, "pose_confidence", None),
                looking_toward_device=getattr(result, "looking_toward_device", False),
                visual_attention_score=getattr(result, "visual_attention_score", None),
                hand_near_camera=getattr(result, "hand_near_camera", False),
                showing_intent_likely=getattr(result, "showing_intent_likely", False),
                gesture_event=getattr(result, "gesture_event", None),
                gesture_confidence=getattr(result, "gesture_confidence", None),
                fine_hand_gesture=getattr(result, "fine_hand_gesture", None),
                fine_hand_gesture_confidence=getattr(result, "fine_hand_gesture_confidence", None),
                sparse_keypoints=getattr(result, "sparse_keypoints", None),
            )
            self._store_pose_result(
                pose,
                observed_at=observed_at,
                observed_monotonic=observed_monotonic,
                primary_person_box=primary_person_box,
            )
            self._record_pose_cache_metadata(
                observed_at=observed_at,
                observed_monotonic=observed_monotonic,
                primary_person_box=primary_person_box,
            )
            return self._clone_pose_result(pose), None

    def _capture_pose(
        self,
        runtime: dict[str, Any],
        *,
        pose_postprocess: Any,
        observed_at: float,
        observed_monotonic: float,
        primary_person_box: Any,
    ) -> PoseResult:
        """Capture one pose frame and decode one coarse pose sample."""

        session = self._runtime_manager.ensure_session(
            runtime,
            network_path=self.config.pose_network_path,
            task_name="pose",
        )
        frame_timestamp_ms = self._to_frame_timestamp_ms(observed_monotonic)
        metadata = self._call_with_supported_kwargs(
            self._runtime_manager.capture_metadata,
            session,
            observed_at=observed_at,
            observed_monotonic=observed_monotonic,
            timestamp_ms=frame_timestamp_ms,
            frame_timestamp_ms=frame_timestamp_ms,
        )
        outputs = session.imx500.get_outputs(metadata, add_batch=True)
        normalized_outputs = self._normalize_runtime_outputs(outputs)

        keypoints, scores, bboxes = self._run_pose_postprocess(
            pose_postprocess=pose_postprocess,
            normalized_outputs=normalized_outputs,
            session=session,
            primary_person_box=primary_person_box,
        )
        keypoints, scores, bboxes = self._limit_pose_candidates(keypoints, scores, bboxes)
        if self._is_empty_result(keypoints) or self._is_empty_result(scores) or self._is_empty_result(bboxes):
            raise RuntimeError("pose_people_missing")

        selected_keypoints, selected_score, selected_box = self._select_primary_pose(
            keypoints=keypoints,
            scores=scores,
            bboxes=bboxes,
            primary_person_box=primary_person_box,
        )
        self._validate_selected_pose_sample(
            selected_keypoints=selected_keypoints,
            selected_score=selected_score,
            selected_box=selected_box,
        )

        parsed_keypoints = parse_keypoints(
            selected_keypoints,
            frame_width=self.config.main_size[0],
            frame_height=self.config.main_size[1],
        )
        pose_confidence = support_pose_confidence(
            selected_score,
            parsed_keypoints,
            fallback_box=selected_box,
        )
        if pose_confidence is None or pose_confidence < self.config.pose_confidence_threshold:
            raise RuntimeError("pose_confidence_low")

        visual_attention = attention_score(parsed_keypoints, fallback_box=selected_box)
        looking_toward_device = visual_attention >= self.config.attention_score_threshold
        hand_near = hand_near_camera(parsed_keypoints, fallback_box=selected_box)
        gesture_event, gesture_confidence = classify_gesture(
            parsed_keypoints,
            attention_score=visual_attention,
            fallback_box=selected_box,
        )

        return self._build_pose_result(
            body_pose=classify_body_pose(parsed_keypoints, fallback_box=selected_box),
            pose_confidence=pose_confidence,
            looking_toward_device=looking_toward_device,
            visual_attention_score=visual_attention,
            hand_near_camera=hand_near,
            showing_intent_likely=hand_near and looking_toward_device,
            gesture_event=gesture_event,
            gesture_confidence=gesture_confidence,
            fine_hand_gesture=None,
            fine_hand_gesture_confidence=None,
            sparse_keypoints=parsed_keypoints,
        )

    def _select_primary_pose(
        self,
        *,
        keypoints: list[list[float]],
        scores: list[float],
        bboxes: list[list[float]],
        primary_person_box: Any,
    ) -> tuple[list[float], float, Any]:
        """Return the pose sample that best matches the primary person."""

        candidates = rank_pose_candidates(
            keypoints=keypoints,
            scores=scores,
            bboxes=bboxes,
            primary_person_box=primary_person_box,
            frame_width=self.config.main_size[0],
            frame_height=self.config.main_size[1],
        )
        if not candidates:
            raise RuntimeError("pose_people_missing")
        selected = candidates[0]
        return selected.raw_keypoints, selected.raw_score, selected.box

    def _pose_resolution_lock(self) -> RLock:
        """Return the shared re-entrant lock used for pose/cache operations."""

        lock = getattr(self, "_pose_resolution_guard_lock", None)
        if lock is None:
            lock = RLock()
            setattr(self, "_pose_resolution_guard_lock", lock)
        return lock

    def _to_frame_timestamp_ms(self, observed_monotonic: float) -> int:
        """Convert monotonic seconds to the millisecond timestamps expected by video/live-stream APIs."""

        return max(0, int(round(float(observed_monotonic) * 1000.0)))

    def _select_pose_tracking_box(self, detection: DetectionResult) -> Any:
        """Return the best tracking box for the current frame, falling back to the full frame when needed."""

        primary_person_box = getattr(detection, "primary_person_box", None)
        if primary_person_box is not None:
            return primary_person_box
        if getattr(detection, "person_count", 0) <= 0:
            return None
        return self._full_frame_box()

    def _full_frame_box(self) -> tuple[float, float, float, float]:
        """Return a box that spans the configured frame dimensions."""

        return (0.0, 0.0, float(self.config.main_size[0]), float(self.config.main_size[1]))

    def _clear_pose_cache_state(self) -> None:
        """Clear the external cache and the local cache metadata used for freshness guards."""

        self._clear_pose_cache()
        setattr(self, "_pose_cache_meta", None)

    def _record_pose_cache_metadata(
        self,
        *,
        observed_at: float,
        observed_monotonic: float,
        primary_person_box: Any,
    ) -> None:
        """Record local metadata for cache freshness and same-person checks."""

        setattr(
            self,
            "_pose_cache_meta",
            {
                "observed_at": float(observed_at),
                "observed_monotonic": float(observed_monotonic),
                "primary_person_box": self._canonical_box(primary_person_box),
            },
        )

    def _can_reuse_pose_cache(
        self,
        *,
        observed_monotonic: float,
        primary_person_box: Any,
    ) -> bool:
        """Return True when the cached pose is both fresh and refers to the same tracked person."""

        cached_pose = getattr(self, "_last_pose_result", None)
        if cached_pose is None:
            return False

        cache_meta = getattr(self, "_pose_cache_meta", None)
        if cache_meta is None:
            return False

        max_age_s = float(getattr(self.config, "pose_cache_max_age_s", 0.35))
        cache_age_s = float(observed_monotonic) - float(cache_meta.get("observed_monotonic", observed_monotonic))
        if cache_age_s < 0.0 or cache_age_s > max_age_s:
            return False

        cached_box = cache_meta.get("primary_person_box")
        if not self._boxes_match(cached_box, primary_person_box):
            return False

        should_reuse = getattr(self, "_should_reuse_pose_cache", None)
        reuse_check: Callable[..., object] | None = should_reuse if callable(should_reuse) else None
        if reuse_check is not None:
            try:
                return bool(
                    reuse_check(
                        observed_monotonic=observed_monotonic,
                        primary_person_box=primary_person_box,
                    )
                )
            except Exception:
                logger.debug("Pose cache reuse check failed; falling back to local freshness guard.", exc_info=True)
        return True

    def _reuse_recent_pose_after_failure(
        self,
        *,
        observed_monotonic: float,
        primary_person_box: Any,
    ) -> PoseResult | None:
        """Return a very recent cached pose after a transient failure, if it still matches the tracked person."""

        if not self._can_reuse_pose_cache(
            observed_monotonic=observed_monotonic,
            primary_person_box=primary_person_box,
        ):
            return None

        cache_meta = getattr(self, "_pose_cache_meta", None)
        if cache_meta is None:
            return None

        max_age_s = float(getattr(self.config, "pose_failure_reuse_max_age_s", 0.20))
        cache_age_s = float(observed_monotonic) - float(cache_meta.get("observed_monotonic", observed_monotonic))
        if cache_age_s < 0.0 or cache_age_s > max_age_s:
            return None

        return self._clone_pose_result(getattr(self, "_last_pose_result", None))

    def _clone_pose_result(self, pose: PoseResult | None) -> PoseResult | None:
        """Create a defensive copy of a cached pose result."""

        if pose is None:
            return None
        return self._build_pose_result(
            body_pose=getattr(pose, "body_pose", None),
            pose_confidence=getattr(pose, "pose_confidence", None),
            looking_toward_device=getattr(pose, "looking_toward_device", False),
            visual_attention_score=getattr(pose, "visual_attention_score", None),
            hand_near_camera=getattr(pose, "hand_near_camera", False),
            showing_intent_likely=getattr(pose, "showing_intent_likely", False),
            gesture_event=getattr(pose, "gesture_event", None),
            gesture_confidence=getattr(pose, "gesture_confidence", None),
            fine_hand_gesture=getattr(pose, "fine_hand_gesture", None),
            fine_hand_gesture_confidence=getattr(pose, "fine_hand_gesture_confidence", None),
            sparse_keypoints=getattr(pose, "sparse_keypoints", None),
        )

    def _build_pose_result(
        self,
        *,
        body_pose: Any,
        pose_confidence: Any,
        looking_toward_device: Any,
        visual_attention_score: Any,
        hand_near_camera: Any,
        showing_intent_likely: Any,
        gesture_event: Any,
        gesture_confidence: Any,
        fine_hand_gesture: Any,
        fine_hand_gesture_confidence: Any,
        sparse_keypoints: Any,
    ) -> PoseResult:
        """Build a normalized PoseResult with safe defaults."""

        return PoseResult(
            body_pose=body_pose,
            pose_confidence=self._coerce_optional_float(pose_confidence),
            looking_toward_device=bool(looking_toward_device),
            visual_attention_score=self._coerce_optional_float(visual_attention_score),
            hand_near_camera=bool(hand_near_camera),
            showing_intent_likely=bool(showing_intent_likely),
            gesture_event=gesture_event,
            gesture_confidence=self._coerce_optional_float(gesture_confidence),
            fine_hand_gesture=fine_hand_gesture,
            fine_hand_gesture_confidence=self._coerce_optional_float(fine_hand_gesture_confidence),
            sparse_keypoints=dict(sparse_keypoints or {}),
        )

    def _coerce_optional_float(self, value: Any) -> float | None:
        """Return a finite float or None."""

        if value is None:
            return None
        try:
            coerced = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(coerced):
            return None
        return coerced

    def _call_mediapipe_analyze(
        self,
        pipeline: Any,
        *,
        frame_rgb: Any,
        observed_at: float,
        observed_monotonic: float,
        primary_person_box: Any,
    ) -> Any:
        """Invoke the MediaPipe pipeline with whichever timestamp / box signature it supports."""

        frame_timestamp_ms = self._to_frame_timestamp_ms(observed_monotonic)
        return self._call_with_supported_kwargs(
            pipeline.analyze,
            frame_rgb=frame_rgb,
            observed_at=observed_at,
            observed_monotonic=observed_monotonic,
            timestamp_ms=frame_timestamp_ms,
            frame_timestamp_ms=frame_timestamp_ms,
            primary_person_box=primary_person_box,
            tracking_box=primary_person_box,
        )

    def _call_with_supported_kwargs(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Call a hook while filtering kwargs to what its signature accepts."""

        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return func(*args, **kwargs)

        parameters = signature.parameters
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_var_kwargs:
            filtered_kwargs = kwargs
        else:
            filtered_kwargs = {key: value for key, value in kwargs.items() if key in parameters}
        return func(*args, **filtered_kwargs)

    def _normalize_runtime_outputs(self, outputs: Any) -> list[Any]:
        """Normalize IMX500 outputs to a concrete list and reject missing tensors."""

        if outputs is None:
            raise RuntimeError("pose_outputs_missing")

        if isinstance(outputs, list):
            normalized_outputs = outputs
        elif isinstance(outputs, tuple):
            normalized_outputs = list(outputs)
        else:
            try:
                normalized_outputs = list(outputs)
            except TypeError:
                normalized_outputs = [outputs]

        if not normalized_outputs:
            raise RuntimeError("pose_outputs_missing")
        return normalized_outputs

    def _run_pose_postprocess(
        self,
        *,
        pose_postprocess: Any,
        normalized_outputs: list[Any],
        session: Any,
        primary_person_box: Any,
    ) -> tuple[Any, Any, Any]:
        """Run whichever pose postprocessor is installed and normalize its return value."""

        frame_size_hw = (self.config.main_size[1], self.config.main_size[0])
        input_size = getattr(session, "input_size", None) or (self.config.main_size[0], self.config.main_size[1])
        input_width, input_height = self._normalize_input_size(input_size)
        output_shape = self._infer_output_shape(normalized_outputs)
        detection_threshold = float(getattr(self.config, "pose_postprocess_detection_threshold", 0.3))
        network_postprocess = self._infer_network_postprocess_mode(pose_postprocess)

        common_kwargs = {
            "outputs": normalized_outputs,
            "output_tensors": normalized_outputs,
            "np_outputs": normalized_outputs,
            "img_size": frame_size_hw,
            "image_size": frame_size_hw,
            "img_w_pad": (0, 0),
            "img_h_pad": (0, 0),
            "detection_threshold": detection_threshold,
            "network_postprocess": network_postprocess,
            "input_image_size": (input_height, input_width),
            "input_size": (input_width, input_height),
            "output_shape": output_shape,
            "primary_person_box": primary_person_box,
            "frame_size": frame_size_hw,
            "frame_width": self.config.main_size[0],
            "frame_height": self.config.main_size[1],
        }

        if hasattr(pose_postprocess, "decode_pose"):
            raw_result = self._call_with_supported_kwargs(pose_postprocess.decode_pose, **common_kwargs)
        elif hasattr(pose_postprocess, "post_process"):
            raw_result = self._call_with_supported_kwargs(pose_postprocess.post_process, **common_kwargs)
        elif hasattr(pose_postprocess, "decode"):
            raw_result = self._call_with_supported_kwargs(pose_postprocess.decode, **common_kwargs)
        else:
            try:
                raw_result = self._call_with_supported_kwargs(pose_postprocess, **common_kwargs)
            except TypeError:
                raw_result = pose_postprocess(
                    normalized_outputs,
                    frame_size_hw,
                    (0, 0),
                    (0, 0),
                    network_postprocess,
                    input_image_size=(input_height, input_width),
                    output_shape=output_shape,
                )

        keypoints, scores, bboxes = self._normalize_pose_postprocess_result(raw_result)
        if self._is_empty_result(keypoints) or self._is_empty_result(scores) or self._is_empty_result(bboxes):
            raise RuntimeError("pose_people_missing")
        return keypoints, scores, bboxes

    def _normalize_input_size(self, input_size: Any) -> tuple[int, int]:
        """Normalize session input size to (width, height)."""

        if isinstance(input_size, Sequence) and len(input_size) >= 2:
            return int(input_size[0]), int(input_size[1])
        return int(self.config.main_size[0]), int(self.config.main_size[1])

    def _infer_output_shape(self, normalized_outputs: Sequence[Any]) -> tuple[int, int] | None:
        """Infer the spatial output shape from the first tensor when possible."""

        if not normalized_outputs:
            return None
        first = normalized_outputs[0]
        shape = getattr(first, "shape", None)
        if shape is None:
            return None
        try:
            ndim = len(shape)
        except TypeError:
            return None

        if ndim >= 4:
            return int(shape[1]), int(shape[2])
        if ndim == 3:
            return int(shape[0]), int(shape[1])
        return None

    def _infer_network_postprocess_mode(self, pose_postprocess: Any) -> bool:
        """Infer whether the model's postprocess expects network-level postprocessing to be enabled."""

        configured = getattr(self.config, "pose_network_postprocess", None)
        if configured is not None:
            return bool(configured)

        name_parts = [
            getattr(pose_postprocess, "__name__", ""),
            type(pose_postprocess).__name__,
            str(getattr(self.config, "pose_network_path", "") or ""),
        ]
        joined = " ".join(part.lower() for part in name_parts if part)
        return any(token in joined for token in ("higherhrnet", "yolo", "rtmo", "rtmw"))

    def _normalize_pose_postprocess_result(self, raw_result: Any) -> tuple[Any, Any, Any]:
        """Normalize different postprocess return conventions to (keypoints, scores, boxes)."""

        if raw_result is None:
            raise RuntimeError("pose_people_missing")

        if isinstance(raw_result, Mapping):
            keypoints = raw_result.get("keypoints")
            if keypoints is None:
                keypoints = raw_result.get("poses")
            scores = raw_result.get("scores")
            if scores is None:
                scores = raw_result.get("pose_scores")
            bboxes = raw_result.get("bboxes")
            if bboxes is None:
                bboxes = raw_result.get("boxes")
            return keypoints, scores, bboxes

        if hasattr(raw_result, "keypoints") or hasattr(raw_result, "poses"):
            keypoints = getattr(raw_result, "keypoints", None)
            if keypoints is None:
                keypoints = getattr(raw_result, "poses", None)
            scores = getattr(raw_result, "scores", None)
            if scores is None:
                scores = getattr(raw_result, "pose_scores", None)
            bboxes = getattr(raw_result, "bboxes", None)
            if bboxes is None:
                bboxes = getattr(raw_result, "boxes", None)
            return keypoints, scores, bboxes

        if isinstance(raw_result, Sequence) and len(raw_result) >= 3:
            first, second, third = raw_result[0], raw_result[1], raw_result[2]
            if self._looks_like_pose_boxes(first) and self._looks_like_pose_keypoints(third):
                return third, second, first
            if self._looks_like_pose_keypoints(first) and self._looks_like_pose_boxes(second):
                return first, third, second
            return first, second, third

        raise RuntimeError("pose_people_missing")

    def _looks_like_pose_keypoints(self, value: Any) -> bool:
        """Heuristically detect a keypoint tensor/list."""

        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                ndim = len(shape)
            except TypeError:
                ndim = 0
            if ndim >= 3:
                last_dim = int(shape[-1])
                return last_dim in (2, 3, 4)
        try:
            if len(value) == 0:
                return False
            first = value[0]
            return hasattr(first, "__len__") and len(first) > 0
        except Exception:
            return False

    def _looks_like_pose_boxes(self, value: Any) -> bool:
        """Heuristically detect a list/tensor of bounding boxes."""

        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                ndim = len(shape)
            except TypeError:
                ndim = 0
            if ndim >= 2:
                last_dim = int(shape[-1])
                return 4 <= last_dim <= 6
        try:
            if len(value) == 0:
                return False
            first = value[0]
            return hasattr(first, "__len__") and 4 <= len(first) <= 6
        except Exception:
            return False

    def _limit_pose_candidates(self, keypoints: Any, scores: Any, bboxes: Any) -> tuple[list[Any], list[Any], list[Any]]:
        """Cap the number of pose candidates processed on the Pi to bound CPU/RAM usage."""

        try:
            candidate_count = min(len(keypoints), len(scores), len(bboxes))
        except TypeError:
            raise RuntimeError("pose_people_missing") from None

        if candidate_count <= 0:
            raise RuntimeError("pose_people_missing")

        max_candidates = max(1, int(getattr(self.config, "pose_max_candidates", 32)))
        indices = list(range(candidate_count))
        if candidate_count > max_candidates:
            indices.sort(key=lambda index: self._score_for_sort(scores, index), reverse=True)
            indices = indices[:max_candidates]

        limited_keypoints = [keypoints[index] for index in indices]
        limited_scores = [scores[index] for index in indices]
        limited_bboxes = [bboxes[index] for index in indices]
        return limited_keypoints, limited_scores, limited_bboxes

    def _score_for_sort(self, scores: Any, index: int) -> float:
        """Return a sortable finite score."""

        try:
            score = float(scores[index])
        except Exception:
            return float("-inf")
        if not math.isfinite(score):
            return float("-inf")
        return score

    def _validate_selected_pose_sample(
        self,
        *,
        selected_keypoints: Any,
        selected_score: Any,
        selected_box: Any,
    ) -> None:
        """Reject malformed selected pose samples before feature extraction."""

        score = self._coerce_optional_float(selected_score)
        if score is None:
            raise RuntimeError("pose_confidence_low")

        box = self._canonical_box(selected_box)
        if box is not None and not all(math.isfinite(component) for component in box):
            raise RuntimeError("pose_box_invalid")

        shape = getattr(selected_keypoints, "shape", None)
        if shape is not None:
            try:
                total_size = 1
                for dimension in shape:
                    total_size *= int(dimension)
            except Exception:
                total_size = 0
            if total_size <= 0 or total_size > 4096:
                raise RuntimeError("pose_keypoints_invalid")

        if not self._pose_numbers_are_finite(selected_keypoints):
            raise RuntimeError("pose_keypoints_invalid")

    def _pose_numbers_are_finite(self, value: Any, *, budget: int = 512) -> bool:
        """Return True when the inspected pose payload contains only finite numeric values."""

        inspected = 0
        stack = [value]
        while stack and inspected < budget:
            current = stack.pop()
            if current is None:
                continue
            if isinstance(current, (str, bytes)):
                return False
            if isinstance(current, Mapping):
                stack.extend(current.values())
                continue
            if hasattr(current, "tolist"):
                try:
                    current = current.tolist()
                except Exception:
                    pass
            if isinstance(current, Sequence):
                stack.extend(reversed(list(current)))
                continue
            try:
                number = float(current)
            except (TypeError, ValueError):
                return False
            if not math.isfinite(number):
                return False
            inspected += 1
        return True

    def _canonical_box(self, box: Any) -> tuple[float, float, float, float] | None:
        """Convert different box representations to finite (x1, y1, x2, y2) coordinates."""

        if box is None:
            return None

        values: tuple[Any, Any, Any, Any] | None = None
        if isinstance(box, Mapping):
            if {"x", "y", "w", "h"} <= set(box):
                values = (box["x"], box["y"], box["w"], box["h"])
            elif {"left", "top", "right", "bottom"} <= set(box):
                values = (box["left"], box["top"], box["right"], box["bottom"])
        elif hasattr(box, "__len__"):
            try:
                if len(box) >= 4:
                    values = (box[0], box[1], box[2], box[3])
            except Exception:
                values = None
        else:
            maybe_values = (
                getattr(box, "x", None),
                getattr(box, "y", None),
                getattr(box, "w", None),
                getattr(box, "h", None),
            )
            if all(component is not None for component in maybe_values):
                values = maybe_values

        if values is None:
            return None

        try:
            x1, y1, x2_or_w, y2_or_h = (float(value) for value in values)
        except (TypeError, ValueError):
            return None

        if not all(math.isfinite(component) for component in (x1, y1, x2_or_w, y2_or_h)):
            return None

        x2 = x2_or_w
        y2 = y2_or_h
        if x2 <= x1 or y2 <= y1:
            x2 = x1 + max(0.0, x2_or_w)
            y2 = y1 + max(0.0, y2_or_h)

        return x1, y1, x2, y2

    def _boxes_match(self, cached_box: Any, current_box: Any) -> bool:
        """Return True when two boxes likely refer to the same tracked person."""

        cached = self._canonical_box(cached_box)
        current = self._canonical_box(current_box)
        if cached is None or current is None:
            return True

        iou_threshold = float(getattr(self.config, "pose_cache_box_iou_threshold", 0.30))
        return self._box_iou(cached, current) >= iou_threshold

    def _box_iou(
        self,
        left: tuple[float, float, float, float],
        right: tuple[float, float, float, float],
    ) -> float:
        """Compute IoU for canonical (x1, y1, x2, y2) boxes."""

        x1 = max(left[0], right[0])
        y1 = max(left[1], right[1])
        x2 = min(left[2], right[2])
        y2 = min(left[3], right[3])

        intersection_w = max(0.0, x2 - x1)
        intersection_h = max(0.0, y2 - y1)
        intersection = intersection_w * intersection_h
        if intersection <= 0.0:
            return 0.0

        left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
        right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
        union = left_area + right_area - intersection
        if union <= 0.0:
            return 0.0
        return intersection / union
