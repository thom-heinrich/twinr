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

# AUDIT-FIX(#2): Serialize hardware-facing diagnostic captures inside this module
# to reduce races on the shared adapter/session when diagnostics overlap.
_PROBE_CAPTURE_LOCK = threading.RLock()

# AUDIT-FIX(#1): Keep "bounded diagnostics" actually bounded on the single-process
# Pi so malformed operator input cannot monopolize the event loop for minutes.
_MAX_SEQUENCE_ATTEMPTS = 8
_MAX_SEQUENCE_SLEEP_S = 2.0
_MIN_POSE_OUTPUT_TENSORS = 3
_EXPECTED_POSE_SCORE_AXIS = 17
_EXPECTED_POSE_COORD_AXIS = 34


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
    """Describe one ranked HigherHRNet candidate."""

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


# AUDIT-FIX(#7): Export only finite floats so diagnostics stay JSON-safe even if
# the model math produced NaN/Inf under degenerate inputs.
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


# AUDIT-FIX(#3): Timestamping and error classification are part of the failure
# path, so they need their own defensive fallbacks to avoid secondary crashes.
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


# AUDIT-FIX(#4): Avoid ambiguous truth-value checks on array-like model outputs
# and validate tensor shapes before indexing into them.
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


def _normalize_pose_outputs(outputs: object) -> tuple[list[Any], tuple[int, int]]:
    if outputs is None:
        raise RuntimeError("pose_outputs_missing")
    try:
        normalized_outputs = list(outputs)
    except TypeError as exc:
        raise RuntimeError("pose_outputs_missing") from exc
    if len(normalized_outputs) < _MIN_POSE_OUTPUT_TENSORS:
        raise RuntimeError("pose_outputs_missing")

    first_shape = _shape_tuple(normalized_outputs[0])
    second_shape = _shape_tuple(normalized_outputs[1])
    if (
        first_shape is not None
        and second_shape is not None
        and first_shape[-1] == _EXPECTED_POSE_SCORE_AXIS
        and second_shape[-1] == _EXPECTED_POSE_COORD_AXIS
    ):
        normalized_outputs[0], normalized_outputs[1] = normalized_outputs[1], normalized_outputs[0]
        first_shape = _shape_tuple(normalized_outputs[0])

    if first_shape is None or len(first_shape) < 3:
        raise RuntimeError("pose_outputs_invalid_shape")

    output_height = _coerce_non_negative_int(first_shape[-3])
    output_width = _coerce_non_negative_int(first_shape[-2])
    if output_height <= 0 or output_width <= 0:
        raise RuntimeError("pose_outputs_invalid_shape")
    return normalized_outputs, (output_height, output_width)


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


def _has_pose_network_path(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


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
    )


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

    # AUDIT-FIX(#2): Keep hardware-facing probe work single-filed inside this
    # module so overlapping diagnostics do not interleave on one camera session.
    with _PROBE_CAPTURE_LOCK:
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
            # AUDIT-FIX(#5): Preserve already-captured detection evidence so later
            # pose-stage failures remain diagnosable instead of collapsing to zeros.
            person_count = _coerce_non_negative_int(detection.person_count)
            primary_person_box = detection.primary_person_box

            pose_network_path = getattr(adapter.config, "pose_network_path", None)
            if not _has_pose_network_path(pose_network_path):
                # AUDIT-FIX(#6): Report missing pose-model configuration explicitly
                # so acceptance probes do not look falsely healthy.
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
            outputs = session.imx500.get_outputs(metadata, add_batch=True)
            normalized_outputs, output_shape = _normalize_pose_outputs(outputs)

            input_width, input_height = _normalize_size_pair(
                session.input_size,
                error_code="pose_input_size_invalid",
            )
            frame_width, frame_height = _normalize_size_pair(
                adapter.config.main_size,
                error_code="frame_size_invalid",
            )

            keypoints, scores, bboxes = pose_postprocess(
                normalized_outputs,
                (frame_height, frame_width),
                (0, 0),
                (0, 0),
                False,
                input_image_size=(input_height, input_width),
                output_shape=output_shape,
            )
            if not _has_items(keypoints) or not _has_items(scores) or not _has_items(bboxes):
                raise RuntimeError("pose_people_missing")

            pose_people_count = _result_length(keypoints)
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
            selected_candidate_index = _coerce_optional_int(selected.candidate_index)
            selected_box = selected.box
            selected_raw_score = _finite_float_or_default(selected.raw_score, default=None)

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
                    selected.raw_score,
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
            )


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

    # AUDIT-FIX(#1): Coerce and clamp operator-provided sequence parameters so
    # diagnostics stay bounded and malformed values do not crash the service.
    total = _coerce_sequence_attempts(attempts)
    sleep_seconds = _coerce_sequence_sleep_seconds(sleep_s)
    probes: list[AICameraPoseProbeDiagnostic] = []
    for index in range(total):
        # AUDIT-FIX(#3): Even if an unexpected regression escapes
        # capture_pose_probe(), the sequence API still returns an error sample.
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
