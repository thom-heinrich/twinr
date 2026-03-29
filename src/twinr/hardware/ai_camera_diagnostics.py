# CHANGELOG: 2026-03-28
# BUG-1: Fixed hard-coded pose tensor assumptions (min=3 tensors, fixed 17/34-axis swap, first-tensor-derived output_shape)
#        that could mis-decode valid IMX500 pose outputs and silently return wrong keypoints/candidate selection.
# BUG-2: Fixed acceptance of mismatched keypoint/score/box result lengths, which could corrupt ranking or raise later.
# SEC-1: Added bounded lock acquisition for diagnostics so one stalled hardware probe cannot indefinitely block all later probes.
# IMP-1: Added model-aware pose decoding fallback paths (raw spatial tensors vs network-postprocessed tensors) driven by metadata,
#        input size, and output shapes instead of a single HigherHRNet-only assumption.
# IMP-2: Added frontier diagnostics for model postprocess metadata, raw output shapes, decoder backend, and IMX500 KPI timings.
# BREAKING: AICameraPoseProbeDiagnostic gained additive fields for model/tensor/timing diagnostics.

"""Capture bounded diagnostics for local IMX500 pose selection on the Pi.

This module is intentionally separate from the runtime adapter so operator
debugging and acceptance probes can inspect detection boxes, candidate ranking,
and keypoint support without mixing that evidence path into the normal
observation contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Any

import numpy as np

from .camera_ai.adapter import LocalAICameraAdapter
from .camera_ai.models import AICameraBodyPose, AICameraBox, AICameraGestureEvent
from .camera_ai.pose_classification import (
    classify_body_pose as _classify_body_pose,
    classify_gesture as _classify_gesture,
)
from .camera_ai.pose_features import (
    attention_score as _attention_score,
    parse_keypoints as _parse_keypoints,
    strong_keypoint_count as _strong_keypoint_count,
    support_pose_confidence as _support_pose_confidence,
    visible_joint as _visible_joint,
)
from .camera_ai.pose_selection import rank_pose_candidates as _rank_pose_candidates

# Serialize hardware-facing diagnostic captures inside this module to reduce
# races on the shared adapter/session when diagnostics overlap.
_PROBE_CAPTURE_LOCK = threading.RLock()

# Keep "bounded diagnostics" actually bounded on the single-process Pi so
# malformed operator input cannot monopolize the event loop for minutes.
_MAX_SEQUENCE_ATTEMPTS = 8
_MAX_SEQUENCE_SLEEP_S = 2.0
_PROBE_CAPTURE_LOCK_TIMEOUT_S = 5.0

# HigherHRNet / COCO pose hints and safe fallbacks. Output layouts vary by
# model/export path, so these are treated as hints instead of hard contracts.
_POSE_CHANNEL_HINTS = frozenset({4, 17, 34})
_DEFAULT_POSE_CONFIDENCE_THRESHOLD = 0.3
_DEFAULT_OUTPUT_SCALE_DIVISOR = 2


@dataclass(frozen=True, slots=True)
class AICameraPoseSupportDiagnostic:
    """Summarize how much usable pose structure one candidate actually has."""

    strong_keypoint_count: int
    shoulders_visible: int
    hips_visible: int
    wrists_visible: int
    legs_visible: int
    face_visible: int


@dataclass(frozen=True, slots=True)
class AICameraPoseCandidateDiagnostic:
    """Describe one ranked pose candidate."""

    candidate_index: int
    raw_score: float
    normalized_score: float
    overlap: float
    center_similarity: float
    size_similarity: float
    selection_score: float
    box: AICameraBox


@dataclass(frozen=True, slots=True)
class AICameraPoseProbeDiagnostic:
    """Describe one bounded detection-plus-pose diagnostic sample."""

    observed_at: float
    person_count: int
    primary_person_box: AICameraBox | None
    pose_people_count: int
    candidates: tuple[AICameraPoseCandidateDiagnostic, ...]
    selected_candidate_index: int | None
    selected_box: AICameraBox | None
    selected_raw_score: float | None
    pose_confidence: float | None
    body_pose: AICameraBodyPose
    attention_score: float | None
    gesture_event: AICameraGestureEvent
    gesture_confidence: float | None
    support: AICameraPoseSupportDiagnostic | None
    camera_error: str | None = None
    pose_model_postprocess: str | None = None
    pose_decoder_backend: str | None = None
    pose_decoder_output_shape: tuple[int, int] | None = None
    pose_input_size: tuple[int, int] | None = None
    pose_output_shapes: tuple[tuple[int, ...], ...] = ()
    dnn_runtime_ms: float | None = None
    dsp_runtime_ms: float | None = None


def _finite_float_or_default(value: object, *, default: float | None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _finite_float(value: object, *, default: float = 0.0) -> float:
    number = _finite_float_or_default(value, default=default)
    return default if number is None else number


def _coerce_non_negative_int(
    value: object,
    *,
    default: int = 0,
    maximum: int | None = None,
) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        number = default
    if number < 0:
        number = 0
    if maximum is not None and number > maximum:
        number = maximum
    return number


def _coerce_optional_int(value: object) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number >= 0 else None


def _coerce_camera_error(value: object, *, default: str) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    return default


def _coerce_probability(value: object, *, default: float) -> float:
    number = _finite_float_or_default(value, default=default)
    if number is None:
        return default
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def _safe_observed_at(adapter: LocalAICameraAdapter) -> float:
    fallback = time.time()
    try:
        observed_at = adapter._now()
    except Exception:
        return fallback
    normalized = _finite_float_or_default(observed_at, default=fallback)
    return fallback if normalized is None else normalized


def _safe_classify_error(adapter: LocalAICameraAdapter, exc: Exception) -> str:
    default = f"unexpected_{exc.__class__.__name__.lower()}"
    try:
        return _coerce_camera_error(adapter._classify_error(exc), default=default)
    except Exception:
        return default


def _result_length(value: object) -> int:
    if value is None:
        return 0
    try:
        return max(0, int(len(value)))
    except (TypeError, ValueError):
        return 0


def _has_items(value: object) -> bool:
    return _result_length(value) > 0


def _shape_tuple(value: object) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        normalized = tuple(int(dim) for dim in shape)
    except (TypeError, ValueError):
        return None
    return normalized or None


def _normalize_shape_list(value: object) -> tuple[tuple[int, ...], ...]:
    if value is None:
        return ()
    shapes: list[tuple[int, ...]] = []
    try:
        iterator = list(value)
    except TypeError:
        return ()
    for item in iterator:
        try:
            shape = tuple(max(0, int(dim)) for dim in item)
        except TypeError:
            continue
        except ValueError:
            continue
        if shape:
            shapes.append(shape)
    return tuple(shapes)


def _normalize_size_pair(value: object, *, error_code: str) -> tuple[int, int]:
    try:
        width, height = value
    except Exception as exc:
        raise RuntimeError(error_code) from exc
    width_px = _coerce_non_negative_int(width)
    height_px = _coerce_non_negative_int(height)
    if width_px <= 0 or height_px <= 0:
        raise RuntimeError(error_code)
    return width_px, height_px


def _optional_size_pair(value: object) -> tuple[int, int] | None:
    try:
        return _normalize_size_pair(value, error_code="size_invalid")
    except RuntimeError:
        return None


def _normalize_body_pose(value: AICameraBodyPose | None) -> AICameraBodyPose:
    return AICameraBodyPose.UNKNOWN if value is None else value


def _normalize_gesture_event(value: AICameraGestureEvent | None) -> AICameraGestureEvent:
    return AICameraGestureEvent.UNKNOWN if value is None else value


def _build_candidate_diagnostic(item: Any) -> AICameraPoseCandidateDiagnostic:
    return AICameraPoseCandidateDiagnostic(
        candidate_index=_coerce_non_negative_int(getattr(item, "candidate_index", 0)),
        raw_score=_finite_float(getattr(item, "raw_score", 0.0)),
        normalized_score=_finite_float(getattr(item, "normalized_score", 0.0)),
        overlap=_finite_float(getattr(item, "overlap", 0.0)),
        center_similarity=_finite_float(getattr(item, "center_similarity", 0.0)),
        size_similarity=_finite_float(getattr(item, "size_similarity", 0.0)),
        selection_score=_finite_float(getattr(item, "selection_score", 0.0)),
        box=item.box,
    )


def _build_probe_diagnostic(
    *,
    observed_at: float,
    person_count: int,
    primary_person_box: AICameraBox | None,
    pose_people_count: int,
    candidates: tuple[AICameraPoseCandidateDiagnostic, ...],
    selected_candidate_index: int | None,
    selected_box: AICameraBox | None,
    selected_raw_score: float | None,
    pose_confidence: float | None,
    body_pose: AICameraBodyPose,
    attention_score: float | None,
    gesture_event: AICameraGestureEvent,
    gesture_confidence: float | None,
    support: AICameraPoseSupportDiagnostic | None,
    camera_error: str | None,
    pose_model_postprocess: str | None = None,
    pose_decoder_backend: str | None = None,
    pose_decoder_output_shape: tuple[int, int] | None = None,
    pose_input_size: tuple[int, int] | None = None,
    pose_output_shapes: tuple[tuple[int, ...], ...] = (),
    dnn_runtime_ms: float | None = None,
    dsp_runtime_ms: float | None = None,
) -> AICameraPoseProbeDiagnostic:
    return AICameraPoseProbeDiagnostic(
        observed_at=_finite_float(observed_at, default=time.time()),
        person_count=_coerce_non_negative_int(person_count),
        primary_person_box=primary_person_box,
        pose_people_count=_coerce_non_negative_int(pose_people_count),
        candidates=tuple(candidates),
        selected_candidate_index=_coerce_optional_int(selected_candidate_index),
        selected_box=selected_box,
        selected_raw_score=_finite_float_or_default(selected_raw_score, default=None),
        pose_confidence=_finite_float_or_default(pose_confidence, default=None),
        body_pose=_normalize_body_pose(body_pose),
        attention_score=_finite_float_or_default(attention_score, default=None),
        gesture_event=_normalize_gesture_event(gesture_event),
        gesture_confidence=_finite_float_or_default(gesture_confidence, default=None),
        support=support,
        camera_error=(
            _coerce_camera_error(camera_error, default="pose_probe_failed")
            if camera_error is not None
            else None
        ),
        pose_model_postprocess=(
            pose_model_postprocess.strip() if isinstance(pose_model_postprocess, str) and pose_model_postprocess.strip() else None
        ),
        pose_decoder_backend=(
            pose_decoder_backend.strip() if isinstance(pose_decoder_backend, str) and pose_decoder_backend.strip() else None
        ),
        pose_decoder_output_shape=_optional_size_pair(pose_decoder_output_shape),
        pose_input_size=_optional_size_pair(pose_input_size),
        pose_output_shapes=_normalize_shape_list(pose_output_shapes),
        dnn_runtime_ms=_finite_float_or_default(dnn_runtime_ms, default=None),
        dsp_runtime_ms=_finite_float_or_default(dsp_runtime_ms, default=None),
    )


def _tensor_supports_moveaxis(value: object) -> bool:
    return hasattr(value, "shape") and hasattr(value, "transpose")


def _move_axis(value: Any, source: int, destination: int) -> Any:
    if not _tensor_supports_moveaxis(value):
        return value
    try:
        return np.moveaxis(value, source, destination)
    except Exception:
        return value


def _normalize_tensor_layout(value: Any) -> Any:
    shape = _shape_tuple(value)
    if shape is None:
        return value

    if len(shape) == 4:
        if shape[-1] in _POSE_CHANNEL_HINTS:
            return value
        if shape[1] in _POSE_CHANNEL_HINTS:
            return _move_axis(value, 1, -1)
        return value

    if len(shape) == 3:
        if shape[-1] in _POSE_CHANNEL_HINTS:
            return value
        if shape[0] == 1 and shape[1] in _POSE_CHANNEL_HINTS:
            return _move_axis(value, 1, -1)
        if shape[0] in _POSE_CHANNEL_HINTS and shape[1] > 1 and shape[2] > 1:
            return _move_axis(value, 0, -1)
        return value

    return value


def _normalize_output_tensor_list(outputs: object) -> list[Any]:
    if outputs is None:
        raise RuntimeError("pose_outputs_missing")
    try:
        normalized_outputs = list(outputs)
    except TypeError as exc:
        raise RuntimeError("pose_outputs_missing") from exc
    if not normalized_outputs:
        raise RuntimeError("pose_outputs_missing")
    return [_normalize_tensor_layout(item) for item in normalized_outputs]


def _infer_spatial_pose_output_shape(outputs: list[Any]) -> tuple[int, int] | None:
    for output in outputs:
        shape = _shape_tuple(output)
        if shape is None:
            continue

        if len(shape) >= 4 and shape[-1] in _POSE_CHANNEL_HINTS:
            height = _coerce_non_negative_int(shape[-3])
            width = _coerce_non_negative_int(shape[-2])
            if height > 0 and width > 0:
                return height, width

        if len(shape) == 3 and shape[-1] in _POSE_CHANNEL_HINTS and shape[0] > 1 and shape[1] > 1:
            height = _coerce_non_negative_int(shape[0])
            width = _coerce_non_negative_int(shape[1])
            if height > 0 and width > 0:
                return height, width
    return None


def _default_pose_output_shape(input_size: tuple[int, int]) -> tuple[int, int]:
    width, height = input_size
    return max(1, height // _DEFAULT_OUTPUT_SCALE_DIVISOR), max(1, width // _DEFAULT_OUTPUT_SCALE_DIVISOR)


def _looks_like_network_postprocessed_pose(outputs: list[Any]) -> bool:
    if len(outputs) < 3:
        return False

    if _infer_spatial_pose_output_shape(outputs) is not None:
        return False

    shapes = [_shape_tuple(item) for item in outputs[:3]]
    if not all(shape is not None for shape in shapes):
        return False

    score_like = any(shape[-1] == 17 for shape in shapes if shape)
    coord_like = any(shape[-1] == 34 for shape in shapes if shape)
    flat_like = all(len(shape) <= 3 for shape in shapes if shape)
    return flat_like and score_like and coord_like


def _validate_pose_results(
    keypoints: object,
    scores: object,
    bboxes: object,
) -> tuple[object, object, object, int]:
    if not _has_items(keypoints) or not _has_items(scores) or not _has_items(bboxes):
        raise RuntimeError("pose_people_missing")

    keypoint_count = _result_length(keypoints)
    score_count = _result_length(scores)
    bbox_count = _result_length(bboxes)
    if keypoint_count <= 0 or score_count <= 0 or bbox_count <= 0:
        raise RuntimeError("pose_people_missing")
    if keypoint_count != score_count or keypoint_count != bbox_count:
        raise RuntimeError("pose_outputs_misaligned")

    return keypoints, scores, bboxes, keypoint_count


def _call_pose_postprocess(
    pose_postprocess: Any,
    *,
    outputs: list[Any],
    frame_size: tuple[int, int],
    input_size: tuple[int, int],
    output_shape: tuple[int, int],
    network_postprocess: bool,
    detection_threshold: float,
) -> tuple[object, object, object]:
    try:
        return pose_postprocess(
            outputs,
            frame_size,
            (0, 0),
            (0, 0),
            network_postprocess,
            input_image_size=(input_size[1], input_size[0]),
            output_shape=output_shape,
            detection_threshold=detection_threshold,
        )
    except TypeError:
        return pose_postprocess(
            outputs,
            frame_size,
            (0, 0),
            (0, 0),
            network_postprocess,
            input_image_size=(input_size[1], input_size[0]),
            output_shape=output_shape,
        )


def _run_pose_postprocess(
    pose_postprocess: Any,
    *,
    outputs: list[Any],
    frame_size: tuple[int, int],
    input_size: tuple[int, int],
    model_postprocess: str | None,
    detection_threshold: float,
) -> tuple[object, object, object, int, str, tuple[int, int]]:
    spatial_output_shape = _infer_spatial_pose_output_shape(outputs)
    default_output_shape = _default_pose_output_shape(input_size)
    prefer_network_postprocess = _looks_like_network_postprocessed_pose(outputs)

    attempts: list[tuple[str, bool, tuple[int, int]]] = []
    if prefer_network_postprocess:
        attempts.append(("higherhrnet_network_postprocess", True, default_output_shape))
        if spatial_output_shape is not None:
            attempts.append(("higherhrnet_raw", False, spatial_output_shape))
        if default_output_shape != spatial_output_shape:
            attempts.append(("higherhrnet_legacy", False, default_output_shape))
    else:
        if spatial_output_shape is not None:
            attempts.append(("higherhrnet_raw", False, spatial_output_shape))
        attempts.append(("higherhrnet_network_postprocess", True, default_output_shape))
        if spatial_output_shape != default_output_shape:
            attempts.append(("higherhrnet_legacy", False, default_output_shape))

    if isinstance(model_postprocess, str):
        lowered = model_postprocess.strip().lower()
        if "higher" in lowered and attempts:
            attempts = sorted(
                attempts,
                key=lambda item: 0 if item[0].startswith("higherhrnet") else 1,
            )

    deduped_attempts: list[tuple[str, bool, tuple[int, int]]] = []
    seen_attempts: set[tuple[bool, tuple[int, int]]] = set()
    for backend, network_postprocess, output_shape in attempts:
        key = (network_postprocess, output_shape)
        if key in seen_attempts:
            continue
        seen_attempts.add(key)
        deduped_attempts.append((backend, network_postprocess, output_shape))

    last_error: Exception | None = None
    for backend, network_postprocess, output_shape in deduped_attempts:
        try:
            keypoints, scores, bboxes = _call_pose_postprocess(
                pose_postprocess,
                outputs=outputs,
                frame_size=frame_size,
                input_size=input_size,
                output_shape=output_shape,
                network_postprocess=network_postprocess,
                detection_threshold=detection_threshold,
            )
            keypoints, scores, bboxes, people_count = _validate_pose_results(keypoints, scores, bboxes)
            return keypoints, scores, bboxes, people_count, backend, output_shape
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("pose_people_missing")


def _safe_get_model_postprocess(session: object) -> str | None:
    imx500 = getattr(session, "imx500", None)
    intrinsics = getattr(imx500, "network_intrinsics", None)
    if intrinsics is None:
        return None
    value = getattr(intrinsics, "postprocess", None)
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    return None


def _safe_get_output_shapes(session: object, metadata: object) -> tuple[tuple[int, ...], ...]:
    imx500 = getattr(session, "imx500", None)
    getter = getattr(imx500, "get_output_shapes", None)
    if getter is None:
        return ()
    try:
        return _normalize_shape_list(getter(metadata))
    except Exception:
        return ()


def _safe_get_kpi_info(session: object, metadata: object) -> tuple[float | None, float | None]:
    imx500 = getattr(session, "imx500", None)
    getter = getattr(imx500, "get_kpi_info", None)
    if getter is None:
        return None, None
    try:
        result = getter(metadata)
    except Exception:
        return None, None
    if not result or len(result) < 2:
        return None, None
    return (
        _finite_float_or_default(result[0], default=None),
        _finite_float_or_default(result[1], default=None),
    )


def _safe_get_input_size(session: object) -> tuple[int, int]:
    direct = getattr(session, "input_size", None)
    normalized = _optional_size_pair(direct)
    if normalized is not None:
        return normalized

    imx500 = getattr(session, "imx500", None)
    getter = getattr(imx500, "get_input_size", None)
    if getter is not None:
        try:
            normalized = _optional_size_pair(getter())
        except Exception:
            normalized = None
        if normalized is not None:
            return normalized

    raise RuntimeError("pose_input_size_invalid")


def _safe_get_frame_size(adapter: LocalAICameraAdapter, session: object) -> tuple[int, int]:
    normalized = _optional_size_pair(getattr(adapter.config, "main_size", None))
    if normalized is not None:
        return normalized

    picam2 = getattr(session, "picam2", None)
    if picam2 is not None:
        try:
            configuration = picam2.camera_configuration()
            normalized = _optional_size_pair(configuration.get("main", {}).get("size"))
        except Exception:
            normalized = None
        if normalized is not None:
            return normalized

    raise RuntimeError("frame_size_invalid")


def _resolve_pose_detection_threshold(adapter: LocalAICameraAdapter) -> float:
    config = getattr(adapter, "config", None)
    for attribute_name in (
        "pose_detection_threshold",
        "pose_threshold",
        "detection_threshold",
    ):
        value = getattr(config, attribute_name, None)
        if value is not None:
            return _coerce_probability(value, default=_DEFAULT_POSE_CONFIDENCE_THRESHOLD)
    return _DEFAULT_POSE_CONFIDENCE_THRESHOLD


def _acquire_probe_lock() -> bool:
    try:
        return _PROBE_CAPTURE_LOCK.acquire(timeout=_PROBE_CAPTURE_LOCK_TIMEOUT_S)
    except TypeError:
        return _PROBE_CAPTURE_LOCK.acquire()


def capture_pose_probe(adapter: LocalAICameraAdapter) -> AICameraPoseProbeDiagnostic:
    """Capture one diagnostic pose probe from the local IMX500 adapter."""

    observed_at = time.time()
    person_count = 0
    primary_person_box: AICameraBox | None = None
    pose_people_count = 0
    candidates: tuple[AICameraPoseCandidateDiagnostic, ...] = ()
    selected_candidate_index: int | None = None
    selected_box: AICameraBox | None = None
    selected_raw_score: float | None = None
    pose_confidence: float | None = None
    body_pose = AICameraBodyPose.UNKNOWN
    attention_score: float | None = None
    gesture_event = AICameraGestureEvent.UNKNOWN
    gesture_confidence: float | None = None
    support: AICameraPoseSupportDiagnostic | None = None
    pose_model_postprocess: str | None = None
    pose_decoder_backend: str | None = None
    pose_decoder_output_shape: tuple[int, int] | None = None
    pose_input_size: tuple[int, int] | None = None
    pose_output_shapes: tuple[tuple[int, ...], ...] = ()
    dnn_runtime_ms: float | None = None
    dsp_runtime_ms: float | None = None

    if not _acquire_probe_lock():
        observed_at = _safe_observed_at(adapter)
        return _build_probe_diagnostic(
            observed_at=observed_at,
            person_count=person_count,
            primary_person_box=primary_person_box,
            pose_people_count=pose_people_count,
            candidates=candidates,
            selected_candidate_index=selected_candidate_index,
            selected_box=selected_box,
            selected_raw_score=selected_raw_score,
            pose_confidence=pose_confidence,
            body_pose=body_pose,
            attention_score=attention_score,
            gesture_event=gesture_event,
            gesture_confidence=gesture_confidence,
            support=support,
            camera_error="pose_probe_busy",
        )

    try:
        observed_at = _safe_observed_at(adapter)
        try:
            runtime = adapter._load_detection_runtime()
            online_error = adapter._probe_online(runtime)
            if online_error is not None:
                return _build_probe_diagnostic(
                    observed_at=observed_at,
                    person_count=person_count,
                    primary_person_box=primary_person_box,
                    pose_people_count=pose_people_count,
                    candidates=candidates,
                    selected_candidate_index=selected_candidate_index,
                    selected_box=selected_box,
                    selected_raw_score=selected_raw_score,
                    pose_confidence=pose_confidence,
                    body_pose=body_pose,
                    attention_score=attention_score,
                    gesture_event=gesture_event,
                    gesture_confidence=gesture_confidence,
                    support=support,
                    camera_error=_coerce_camera_error(online_error, default="camera_offline"),
                )

            detection = adapter._capture_detection(runtime, observed_at=observed_at)
            person_count = _coerce_non_negative_int(detection.person_count)
            primary_person_box = detection.primary_person_box

            pose_network_path = getattr(adapter.config, "pose_network_path", None)
            if not pose_network_path:
                gesture_event = AICameraGestureEvent.NONE if person_count <= 0 else AICameraGestureEvent.UNKNOWN
                return _build_probe_diagnostic(
                    observed_at=observed_at,
                    person_count=person_count,
                    primary_person_box=primary_person_box,
                    pose_people_count=pose_people_count,
                    candidates=candidates,
                    selected_candidate_index=selected_candidate_index,
                    selected_box=selected_box,
                    selected_raw_score=selected_raw_score,
                    pose_confidence=pose_confidence,
                    body_pose=body_pose,
                    attention_score=attention_score,
                    gesture_event=gesture_event,
                    gesture_confidence=gesture_confidence,
                    support=support,
                    camera_error="pose_network_unconfigured",
                )

            if person_count <= 0:
                gesture_event = AICameraGestureEvent.NONE
                return _build_probe_diagnostic(
                    observed_at=observed_at,
                    person_count=person_count,
                    primary_person_box=primary_person_box,
                    pose_people_count=pose_people_count,
                    candidates=candidates,
                    selected_candidate_index=selected_candidate_index,
                    selected_box=selected_box,
                    selected_raw_score=selected_raw_score,
                    pose_confidence=pose_confidence,
                    body_pose=body_pose,
                    attention_score=attention_score,
                    gesture_event=gesture_event,
                    gesture_confidence=gesture_confidence,
                    support=support,
                    camera_error=None,
                )

            pose_postprocess = adapter._load_pose_postprocess()
            session = adapter._ensure_session(
                runtime,
                network_path=pose_network_path,
                task_name="pose",
            )
            metadata = adapter._capture_metadata(session, observed_at=observed_at)

            pose_model_postprocess = _safe_get_model_postprocess(session)
            pose_output_shapes = _safe_get_output_shapes(session, metadata)
            dnn_runtime_ms, dsp_runtime_ms = _safe_get_kpi_info(session, metadata)

            outputs = session.imx500.get_outputs(metadata, add_batch=True)
            normalized_outputs = _normalize_output_tensor_list(outputs)

            pose_input_size = _safe_get_input_size(session)
            frame_width, frame_height = _safe_get_frame_size(adapter, session)
            detection_threshold = _resolve_pose_detection_threshold(adapter)

            (
                keypoints,
                scores,
                bboxes,
                pose_people_count,
                pose_decoder_backend,
                pose_decoder_output_shape,
            ) = _run_pose_postprocess(
                pose_postprocess,
                outputs=normalized_outputs,
                frame_size=(frame_height, frame_width),
                input_size=pose_input_size,
                model_postprocess=pose_model_postprocess,
                detection_threshold=detection_threshold,
            )

            ranked = tuple(
                _rank_pose_candidates(
                    keypoints=keypoints,
                    scores=scores,
                    bboxes=bboxes,
                    primary_person_box=primary_person_box,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            )
            if not ranked:
                raise RuntimeError("pose_people_missing")

            candidates = tuple(_build_candidate_diagnostic(item) for item in ranked)
            selected = ranked[0]
            selected_candidate_index = _coerce_optional_int(getattr(selected, "candidate_index", None))
            selected_box = selected.box
            selected_raw_score = _finite_float_or_default(getattr(selected, "raw_score", None), default=None)

            parsed_keypoints = _parse_keypoints(
                selected.raw_keypoints,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            support = _build_support_diagnostic(parsed_keypoints)
            attention_score = _finite_float_or_default(
                _attention_score(parsed_keypoints, fallback_box=selected_box),
                default=None,
            )
            gesture_event, gesture_confidence = _classify_gesture(
                parsed_keypoints,
                attention_score=attention_score,
                fallback_box=selected_box,
            )
            gesture_event = _normalize_gesture_event(gesture_event)
            gesture_confidence = _finite_float_or_default(gesture_confidence, default=None)
            pose_confidence = _finite_float_or_default(
                _support_pose_confidence(
                    getattr(selected, "raw_score", None),
                    parsed_keypoints,
                    fallback_box=selected_box,
                ),
                default=None,
            )
            body_pose = _normalize_body_pose(
                _classify_body_pose(parsed_keypoints, fallback_box=selected_box)
            )

            return _build_probe_diagnostic(
                observed_at=observed_at,
                person_count=person_count,
                primary_person_box=primary_person_box,
                pose_people_count=pose_people_count,
                candidates=candidates,
                selected_candidate_index=selected_candidate_index,
                selected_box=selected_box,
                selected_raw_score=selected_raw_score,
                pose_confidence=pose_confidence,
                body_pose=body_pose,
                attention_score=attention_score,
                gesture_event=gesture_event,
                gesture_confidence=gesture_confidence,
                support=support,
                camera_error=None,
                pose_model_postprocess=pose_model_postprocess,
                pose_decoder_backend=pose_decoder_backend,
                pose_decoder_output_shape=pose_decoder_output_shape,
                pose_input_size=pose_input_size,
                pose_output_shapes=pose_output_shapes,
                dnn_runtime_ms=dnn_runtime_ms,
                dsp_runtime_ms=dsp_runtime_ms,
            )
        except Exception as exc:  # pragma: no cover - hardware path depends on Pi runtime.
            return _build_probe_diagnostic(
                observed_at=observed_at,
                person_count=person_count,
                primary_person_box=primary_person_box,
                pose_people_count=pose_people_count,
                candidates=candidates,
                selected_candidate_index=selected_candidate_index,
                selected_box=selected_box,
                selected_raw_score=selected_raw_score,
                pose_confidence=pose_confidence,
                body_pose=body_pose,
                attention_score=attention_score,
                gesture_event=gesture_event,
                gesture_confidence=gesture_confidence,
                support=support,
                camera_error=_safe_classify_error(adapter, exc),
                pose_model_postprocess=pose_model_postprocess,
                pose_decoder_backend=pose_decoder_backend,
                pose_decoder_output_shape=pose_decoder_output_shape,
                pose_input_size=pose_input_size,
                pose_output_shapes=pose_output_shapes,
                dnn_runtime_ms=dnn_runtime_ms,
                dsp_runtime_ms=dsp_runtime_ms,
            )
    finally:
        _PROBE_CAPTURE_LOCK.release()


def _coerce_sequence_attempts(attempts: object) -> int:
    total = _coerce_non_negative_int(
        attempts,
        default=1,
        maximum=_MAX_SEQUENCE_ATTEMPTS,
    )
    return max(1, total)


def _coerce_sequence_sleep_seconds(sleep_s: object) -> float:
    seconds = _finite_float_or_default(sleep_s, default=0.0)
    if seconds is None or seconds <= 0.0:
        return 0.0
    return min(seconds, _MAX_SEQUENCE_SLEEP_S)


def capture_pose_probe_sequence(
    adapter: LocalAICameraAdapter,
    *,
    attempts: int,
    sleep_s: float = 0.0,
) -> tuple[AICameraPoseProbeDiagnostic, ...]:
    """Capture several bounded diagnostic probes with one shared adapter."""

    total = _coerce_sequence_attempts(attempts)
    sleep_seconds = _coerce_sequence_sleep_seconds(sleep_s)
    probes: list[AICameraPoseProbeDiagnostic] = []
    for index in range(total):
        try:
            probes.append(capture_pose_probe(adapter))
        except Exception as exc:
            probes.append(
                _build_probe_diagnostic(
                    observed_at=_safe_observed_at(adapter),
                    person_count=0,
                    primary_person_box=None,
                    pose_people_count=0,
                    candidates=(),
                    selected_candidate_index=None,
                    selected_box=None,
                    selected_raw_score=None,
                    pose_confidence=None,
                    body_pose=AICameraBodyPose.UNKNOWN,
                    attention_score=None,
                    gesture_event=AICameraGestureEvent.UNKNOWN,
                    gesture_confidence=None,
                    support=None,
                    camera_error=_safe_classify_error(adapter, exc),
                )
            )
        if index < total - 1 and sleep_seconds > 0.0:
            try:
                adapter._sleep(sleep_seconds)
            except Exception:
                try:
                    time.sleep(sleep_seconds)
                except Exception:
                    pass
    return tuple(probes)


def _build_support_diagnostic(
    keypoints: dict[int, tuple[float, float, float]],
) -> AICameraPoseSupportDiagnostic:
    """Summarize visible keypoint support for one selected pose candidate."""

    return AICameraPoseSupportDiagnostic(
        strong_keypoint_count=_strong_keypoint_count(keypoints),
        shoulders_visible=sum(1 for index in (5, 6) if _visible_joint(keypoints, index) is not None),
        hips_visible=sum(1 for index in (11, 12) if _visible_joint(keypoints, index) is not None),
        wrists_visible=sum(1 for index in (9, 10) if _visible_joint(keypoints, index) is not None),
        legs_visible=sum(1 for index in (13, 14, 15, 16) if _visible_joint(keypoints, index) is not None),
        face_visible=sum(1 for index in (0, 1, 2) if _visible_joint(keypoints, index) is not None),
    )


__all__ = [
    "AICameraPoseCandidateDiagnostic",
    "AICameraPoseProbeDiagnostic",
    "AICameraPoseSupportDiagnostic",
    "capture_pose_probe",
    "capture_pose_probe_sequence",
]