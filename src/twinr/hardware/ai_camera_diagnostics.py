"""Capture bounded diagnostics for local IMX500 pose selection on the Pi.

This module is intentionally separate from the runtime adapter so operator
debugging and acceptance probes can inspect detection boxes, candidate ranking,
and keypoint support without mixing that evidence path into the normal
observation contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

from .ai_camera import (
    AICameraBodyPose,
    AICameraBox,
    AICameraGestureEvent,
    LocalAICameraAdapter,
    _attention_score,
    _classify_body_pose,
    _classify_gesture,
    _parse_keypoints,
    _rank_pose_candidates,
    _strong_keypoint_count,
    _support_pose_confidence,
    _visible_joint,
)


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


def capture_pose_probe(adapter: LocalAICameraAdapter) -> AICameraPoseProbeDiagnostic:
    """Capture one diagnostic pose probe from the local IMX500 adapter."""

    observed_at = adapter._now()
    try:
        runtime = adapter._load_detection_runtime()
        online_error = adapter._probe_online(runtime)
        if online_error is not None:
            return AICameraPoseProbeDiagnostic(
                observed_at=observed_at,
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
                camera_error=online_error,
            )

        detection = adapter._capture_detection(runtime, observed_at=observed_at)
        if detection.person_count <= 0 or not adapter.config.pose_network_path:
            return AICameraPoseProbeDiagnostic(
                observed_at=observed_at,
                person_count=detection.person_count,
                primary_person_box=detection.primary_person_box,
                pose_people_count=0,
                candidates=(),
                selected_candidate_index=None,
                selected_box=None,
                selected_raw_score=None,
                pose_confidence=None,
                body_pose=AICameraBodyPose.UNKNOWN,
                attention_score=None,
                gesture_event=AICameraGestureEvent.NONE,
                gesture_confidence=None,
                support=None,
                camera_error=None,
            )

        pose_postprocess = adapter._load_pose_postprocess()
        session = adapter._ensure_session(
            runtime,
            network_path=adapter.config.pose_network_path,
            task_name="pose",
        )
        metadata = adapter._capture_metadata(session, observed_at=observed_at)
        outputs = session.imx500.get_outputs(metadata, add_batch=True)
        if not outputs or len(outputs) < 3:
            raise RuntimeError("pose_outputs_missing")
        normalized_outputs = list(outputs)
        if (
            len(normalized_outputs) >= 2
            and normalized_outputs[0].shape[-1] == 17
            and normalized_outputs[1].shape[-1] == 34
        ):
            normalized_outputs[0], normalized_outputs[1] = normalized_outputs[1], normalized_outputs[0]

        input_width, input_height = session.input_size
        keypoints, scores, bboxes = pose_postprocess(
            normalized_outputs,
            (adapter.config.main_size[1], adapter.config.main_size[0]),
            (0, 0),
            (0, 0),
            False,
            input_image_size=(input_height, input_width),
            output_shape=(normalized_outputs[0].shape[1], normalized_outputs[0].shape[2]),
        )
        if not keypoints or not scores or not bboxes:
            raise RuntimeError("pose_people_missing")

        ranked = _rank_pose_candidates(
            keypoints=keypoints,
            scores=scores,
            bboxes=bboxes,
            primary_person_box=detection.primary_person_box,
            frame_width=adapter.config.main_size[0],
            frame_height=adapter.config.main_size[1],
        )
        if not ranked:
            raise RuntimeError("pose_people_missing")
        selected = ranked[0]
        parsed_keypoints = _parse_keypoints(
            selected.raw_keypoints,
            frame_width=adapter.config.main_size[0],
            frame_height=adapter.config.main_size[1],
        )
        support = _build_support_diagnostic(parsed_keypoints)
        attention_score = _attention_score(parsed_keypoints, fallback_box=selected.box)
        gesture_event, gesture_confidence = _classify_gesture(
            parsed_keypoints,
            attention_score=attention_score,
            fallback_box=selected.box,
        )
        return AICameraPoseProbeDiagnostic(
            observed_at=observed_at,
            person_count=detection.person_count,
            primary_person_box=detection.primary_person_box,
            pose_people_count=len(keypoints),
            candidates=tuple(
                AICameraPoseCandidateDiagnostic(
                    candidate_index=item.candidate_index,
                    raw_score=item.raw_score,
                    normalized_score=item.normalized_score,
                    overlap=item.overlap,
                    center_similarity=item.center_similarity,
                    size_similarity=item.size_similarity,
                    selection_score=item.selection_score,
                    box=item.box,
                )
                for item in ranked
            ),
            selected_candidate_index=selected.candidate_index,
            selected_box=selected.box,
            selected_raw_score=selected.raw_score,
            pose_confidence=_support_pose_confidence(
                selected.raw_score,
                parsed_keypoints,
                fallback_box=selected.box,
            ),
            body_pose=_classify_body_pose(parsed_keypoints, fallback_box=selected.box),
            attention_score=attention_score,
            gesture_event=gesture_event,
            gesture_confidence=gesture_confidence,
            support=support,
            camera_error=None,
        )
    except Exception as exc:  # pragma: no cover - hardware path depends on Pi runtime.
        return AICameraPoseProbeDiagnostic(
            observed_at=observed_at,
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
            camera_error=adapter._classify_error(exc),
        )


def capture_pose_probe_sequence(
    adapter: LocalAICameraAdapter,
    *,
    attempts: int,
    sleep_s: float = 0.0,
) -> tuple[AICameraPoseProbeDiagnostic, ...]:
    """Capture several bounded diagnostic probes with one shared adapter."""

    total = max(1, int(attempts))
    probes: list[AICameraPoseProbeDiagnostic] = []
    for index in range(total):
        probes.append(capture_pose_probe(adapter))
        if index < total - 1 and sleep_s > 0.0:
            try:
                adapter._sleep(sleep_s)
            except Exception:
                time.sleep(sleep_s)
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
