
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.camera_ai import adapter as public_module
from twinr.hardware.camera_ai.config import AICameraAdapterConfig
from twinr.hardware.camera_ai.detection import DetectionResult
from twinr.hardware.camera_ai.mediapipe_pipeline import MediaPipeVisionResult
from twinr.hardware.camera_ai.models import (
    AICameraBodyPose,
    AICameraBox,
    AICameraFineHandGesture,
    AICameraGestureEvent,
    AICameraMotionState,
    AICameraVisiblePerson,
    AICameraZone,
)

try:
    from twinr.hardware.camera_ai import adapter_impl as impl_module
except (ImportError, ModuleNotFoundError):  # pragma: no cover - pre-refactor collection path
    impl_module = None


_EXPECTED_GOLDEN_DIGESTS = {
    "helpers": "1b3716f74c5fa02017b9d571e5d35246c5d657049a0f37d838d7846b9b6198fe",
    "observe": "4fa1ba3559320128202eb18977175e574faf0506889d05637436a7c1038e9e67",
    "attention": "ecc110c6317ef87eff55f052d960c3b60334f947f553db4a13d6701e071da3f0",
    "gesture": "da6e394eda37dc9a61feca51bc6555037520e20d84064170afa09aa6eae01a29",
}


def _normalize_payload(value: object) -> object:
    if is_dataclass(value):
        return {key: _normalize_payload(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _normalize_payload(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_payload(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def _payload_digest(payload: object) -> str:
    rendered = json.dumps(
        _normalize_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return sha256(rendered.encode("utf-8")).hexdigest()


def _build_helper_payload(module) -> dict[str, object]:
    adapter = module.LocalAICameraAdapter(
        clock=lambda: 123.456,
        monotonic_clock=lambda: 200.0,
    )
    return {
        "coerce_observed_at_none": adapter._coerce_observed_at(None),
        "coerce_observed_at_invalid": adapter._coerce_observed_at(float("nan")),
        "classify_busy": adapter._classify_error(RuntimeError("Device or resource busy")),
        "classify_pose_missing": adapter._classify_error(
            RuntimeError("pose_dependency_missing: opencv")
        ),
        "iter_error_messages": adapter._iter_error_messages(
            RuntimeError("outer failure", ValueError("inner failure"))
        ),
        "box_metrics_box": adapter._extract_box_metrics(
            AICameraBox(top=0.1, left=0.2, bottom=0.9, right=0.6)
        ),
        "box_metrics_xywh": adapter._extract_box_metrics(
            SimpleNamespace(x=0.2, y=0.1, w=0.4, h=0.8)
        ),
        "gesture_zone_left": str(
            adapter._gesture_zone_from_box(
                AICameraBox(top=0.1, left=0.0, bottom=0.8, right=0.2)
            )
        ),
        "gesture_zone_center": str(
            adapter._gesture_zone_from_box(
                AICameraBox(top=0.1, left=0.3, bottom=0.8, right=0.7)
            )
        ),
    }


def _sample_detection() -> DetectionResult:
    primary_box = AICameraBox(top=0.18, left=0.30, bottom=0.86, right=0.70)
    return DetectionResult(
        person_count=1,
        primary_person_box=primary_box,
        primary_person_zone=AICameraZone.CENTER,
        visible_persons=(
            AICameraVisiblePerson(
                box=primary_box,
                zone=AICameraZone.CENTER,
                confidence=0.81,
            ),
        ),
        person_near_device=True,
        hand_or_object_near_camera=False,
        objects=(),
    )


def _build_observe_payload(module) -> dict[str, object]:
    adapter = module.LocalAICameraAdapter(
        config=AICameraAdapterConfig(
            pose_backend="mediapipe",
            mediapipe_pose_model_path="state/mediapipe/models/pose_landmarker_full.task",
            mediapipe_gesture_model_path="state/mediapipe/models/gesture_recognizer.task",
        ),
        clock=lambda: 10.0,
        monotonic_clock=lambda: 20.0,
    )
    adapter._load_detection_runtime = lambda: {}
    adapter._probe_online = lambda runtime: None
    adapter._capture_detection = lambda runtime, observed_at: SimpleNamespace(
        person_count=1,
        primary_person_box=AICameraBox(top=0.12, left=0.24, bottom=0.92, right=0.68),
        primary_person_zone="center",
        person_near_device=True,
        hand_or_object_near_camera=False,
        objects=(),
    )
    adapter._capture_rgb_frame = lambda runtime, observed_at: object()
    adapter._ensure_mediapipe_pipeline = lambda: SimpleNamespace(
        analyze=lambda **_: MediaPipeVisionResult(
            body_pose=AICameraBodyPose.UPRIGHT,
            pose_confidence=0.74,
            looking_toward_device=True,
            visual_attention_score=0.82,
            hand_near_camera=True,
            showing_intent_likely=True,
            gesture_event=AICameraGestureEvent.WAVE,
            gesture_confidence=0.69,
            fine_hand_gesture=AICameraFineHandGesture.THUMBS_UP,
            fine_hand_gesture_confidence=0.88,
        )
    )
    adapter._resolve_motion = lambda **_: (AICameraMotionState.STILL, 0.57)
    observation = adapter.observe()
    return {
        "observation": observation,
        "last_health_signature": adapter._last_health_signature,
        "last_frame_at": adapter._last_frame_at,
    }


def _build_attention_payload(module) -> dict[str, object]:
    adapter = module.LocalAICameraAdapter(
        config=AICameraAdapterConfig(attention_score_threshold=0.62),
        clock=lambda: 88.0,
        monotonic_clock=lambda: 188.0,
    )
    observation = adapter.observe_attention_from_frame(
        detection=_sample_detection(),
        frame_rgb=None,
        observed_at=88.0,
        frame_at=87.5,
    )
    return {
        "observation": observation,
        "debug": adapter.get_last_attention_debug_details(),
    }


def _build_gesture_payload(module) -> dict[str, object]:
    adapter = module.LocalAICameraAdapter(
        config=AICameraAdapterConfig(
            pose_backend="mediapipe",
            mediapipe_pose_model_path="state/mediapipe/models/pose_landmarker_full.task",
            mediapipe_hand_landmarker_model_path="state/mediapipe/models/hand_landmarker.task",
            mediapipe_gesture_model_path="state/mediapipe/models/gesture_recognizer.task",
        ),
        clock=lambda: 88.0,
        monotonic_clock=lambda: 188.0,
    )
    adapter._resolve_gesture_pose_hints = lambda **kwargs: ({}, "none", None)
    adapter._ensure_live_gesture_pipeline = lambda: SimpleNamespace(
        observe=lambda **_: SimpleNamespace(
            hand_count=0,
            fine_hand_gesture=AICameraFineHandGesture.NONE,
            fine_hand_gesture_confidence=None,
            gesture_event=AICameraGestureEvent.NONE,
            gesture_confidence=None,
        ),
        debug_snapshot=lambda: {"resolved_source": "none"},
    )
    fallback_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _resolve_pose_fallback(*args, **kwargs):
        fallback_calls.append((args, kwargs))
        return (
            module.PoseResult(
                body_pose=AICameraBodyPose.UNKNOWN,
                pose_confidence=0.71,
                looking_toward_device=False,
                visual_attention_score=0.0,
                hand_near_camera=False,
                showing_intent_likely=False,
                gesture_event=AICameraGestureEvent.WAVE,
                gesture_confidence=0.71,
            ),
            None,
        )

    adapter._resolve_gesture_pose_fallback = _resolve_pose_fallback
    observation = adapter.observe_gesture_from_frame(
        detection=_sample_detection(),
        frame_rgb="frame",
        observed_at=88.0,
        frame_at=87.5,
        allow_pose_fallback=False,
    )
    return {
        "observation": observation,
        "debug": adapter.get_last_gesture_debug_details(),
        "fallback_calls": len(fallback_calls),
    }


def _golden_payloads(module) -> dict[str, object]:
    return {
        "helpers": _build_helper_payload(module),
        "observe": _build_observe_payload(module),
        "attention": _build_attention_payload(module),
        "gesture": _build_gesture_payload(module),
    }


class AICameraAdapterRefactorParityTests(unittest.TestCase):
    """Freeze the public AI-camera adapter behavior during module extraction."""

    def test_public_module_matches_golden_master(self) -> None:
        digests = {
            name: _payload_digest(payload)
            for name, payload in _golden_payloads(public_module).items()
        }
        self.assertEqual(digests, _EXPECTED_GOLDEN_DIGESTS)

    def test_public_module_matches_impl_when_available(self) -> None:
        if impl_module is None:
            self.skipTest("implementation package not present before refactor")
        public_digests = {
            name: _payload_digest(payload)
            for name, payload in _golden_payloads(public_module).items()
        }
        impl_digests = {
            name: _payload_digest(payload)
            for name, payload in _golden_payloads(impl_module).items()
        }
        self.assertEqual(public_digests, impl_digests)

    def test_public_exports_keep_stable_module_identity(self) -> None:
        self.assertEqual(public_module.LocalAICameraAdapter.__module__, public_module.__name__)
        self.assertEqual(public_module.PoseResult.__module__, public_module.__name__)


if __name__ == "__main__":
    unittest.main()
