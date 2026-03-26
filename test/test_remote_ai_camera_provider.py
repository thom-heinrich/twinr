"""Regression coverage for the peer-LAN remote AI-camera provider."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, cast
import unittest
from unittest.mock import patch
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.proactive.social.engine import SocialFineHandGesture, SocialPersonZone
from twinr.hardware.ai_camera import (
    AICameraFineHandGesture,
    AICameraObservation,
    AICameraZone,
)
from twinr.proactive.social.remote_camera_provider import (
    RemoteAICameraObservationProvider,
    RemoteAICameraProviderConfig,
    RemoteFrameAICameraObservationProvider,
)


class RemoteAICameraProviderTests(unittest.TestCase):
    def test_provider_maps_remote_payload_to_social_snapshot(self) -> None:
        provider = RemoteAICameraObservationProvider(
            config=RemoteAICameraProviderConfig(base_url="http://10.42.0.2:8767")
        )
        payload = {
            "observation": {
                "observed_at": 101.5,
                "camera_online": True,
                "camera_ready": True,
                "camera_ai_ready": True,
                "person_count": 1,
                "primary_person_zone": "center",
                "primary_person_box": {
                    "top": 0.1,
                    "left": 0.2,
                    "bottom": 0.8,
                    "right": 0.7,
                },
                "fine_hand_gesture": "thumbs_up",
                "fine_hand_gesture_confidence": 0.88,
                "model": "local-imx500+mediapipe",
            },
            "captured_at": 100.9,
            "model": "local-imx500+mediapipe",
        }

        with patch.object(provider, "_request_json", return_value=payload):
            snapshot = provider.observe()

        self.assertTrue(snapshot.observation.person_visible)
        self.assertEqual(snapshot.observation.primary_person_zone, SocialPersonZone.CENTER)
        self.assertEqual(snapshot.observation.fine_hand_gesture, SocialFineHandGesture.THUMBS_UP)
        self.assertAlmostEqual(snapshot.captured_at or 0.0, 100.9, places=3)
        self.assertEqual(snapshot.model, "local-imx500+mediapipe")
        self.assertEqual(snapshot.input_format, "remote-ai-proxy")

    def test_provider_preserves_remote_debug_details_for_gesture_refresh(self) -> None:
        provider = RemoteAICameraObservationProvider(
            config=RemoteAICameraProviderConfig(base_url="http://10.42.0.2:8767")
        )
        payload = {
            "observation": {
                "observed_at": 55.0,
                "camera_online": True,
                "camera_ready": True,
                "camera_ai_ready": True,
                "gesture_event": "none",
                "fine_hand_gesture": "peace_sign",
                "fine_hand_gesture_confidence": 0.79,
            },
            "debug_details": {
                "resolved_source": "builtin",
                "live_gesture": "peace_sign",
            },
        }

        with patch.object(provider, "_request_json", return_value=payload):
            snapshot = provider.observe_gesture()

        self.assertEqual(snapshot.observation.fine_hand_gesture, SocialFineHandGesture.PEACE_SIGN)
        self.assertEqual(
            provider.gesture_debug_details(),
            {
                "resolved_source": "builtin",
                "live_gesture": "peace_sign",
            },
        )

    def test_provider_degrades_to_health_snapshot_when_proxy_is_unreachable(self) -> None:
        provider = RemoteAICameraObservationProvider(
            config=RemoteAICameraProviderConfig(base_url="http://10.42.0.2:8767")
        )

        with patch.object(provider, "_request_json", side_effect=URLError("offline")):
            snapshot = provider.observe_attention()

        self.assertFalse(snapshot.observation.person_visible)
        self.assertFalse(snapshot.observation.camera_online)
        self.assertEqual(snapshot.observation.camera_error, "remote_ai_camera_unreachable")

    def test_remote_frame_provider_runs_gesture_hot_path_on_main_pi(self) -> None:
        class _FakeProcessor:
            def __init__(self) -> None:
                self.gesture_calls: list[dict[str, object]] = []
                self.attention_calls: list[dict[str, object]] = []

            def _coerce_detection_result(self, detection):
                return detection

            def _needs_rgb_frame_for_attention(self, *, detection):
                return True

            def observe_gesture_from_frame(self, **kwargs):
                self.gesture_calls.append(dict(kwargs))
                return AICameraObservation(
                    observed_at=71.0,
                    camera_online=True,
                    camera_ready=True,
                    camera_ai_ready=True,
                    person_count=1,
                    primary_person_zone="center",
                    fine_hand_gesture=AICameraFineHandGesture.PEACE_SIGN,
                    fine_hand_gesture_confidence=0.82,
                    model="local-imx500+mediapipe-live-gesture",
                )

            def observe_attention_from_frame(self, **kwargs):
                self.attention_calls.append(dict(kwargs))
                return AICameraObservation(
                    observed_at=71.0,
                    camera_online=True,
                    camera_ready=True,
                    camera_ai_ready=True,
                    person_count=1,
                    primary_person_zone="center",
                    model="local-imx500+mediapipe",
                )

            def get_last_gesture_debug_details(self):
                return {"resolved_source": "live_stream"}

            def get_last_attention_debug_details(self):
                return None

            def close(self):
                return None

        processor = _FakeProcessor()
        provider = RemoteFrameAICameraObservationProvider(
            config=RemoteAICameraProviderConfig(
                base_url="http://10.42.0.2:8767",
                input_format="remote-ai-frame",
            ),
            processor=cast(Any, processor),
        )
        remote_observation = AICameraObservation(
            observed_at=70.0,
            camera_online=True,
            camera_ready=True,
            camera_ai_ready=True,
            person_count=1,
            primary_person_zone=AICameraZone.CENTER,
            model="local-imx500+mediapipe",
        )

        with (
            patch.object(
                provider,
                "_fetch_remote_frame_bundle",
                return_value=(
                    remote_observation,
                    69.5,
                    "local-imx500+mediapipe",
                    {"cache_state": "busy_reused"},
                    "rgb-frame",
                ),
            ),
        ):
            snapshot = provider.observe_gesture()

        self.assertEqual(processor.gesture_calls[0]["frame_rgb"], "rgb-frame")
        self.assertFalse(processor.gesture_calls[0]["allow_pose_fallback"])
        self.assertEqual(snapshot.input_format, "remote-ai-frame")
        self.assertEqual(snapshot.observation.fine_hand_gesture, SocialFineHandGesture.PEACE_SIGN)
        self.assertEqual(snapshot.model, "remote-imx500-detection+local-mediapipe-live-gesture")
        gesture_debug = provider.gesture_debug_details()
        self.assertIsNotNone(gesture_debug)
        assert gesture_debug is not None
        self.assertEqual(gesture_debug["transport_mode"], "remote_frame_local_gesture")
        self.assertEqual(cast(dict[str, object], gesture_debug["remote_debug"])["cache_state"], "busy_reused")
        self.assertIn("provider_stage_ms", gesture_debug)

    def test_remote_frame_provider_runs_attention_hot_path_from_bundle(self) -> None:
        class _FakeProcessor:
            def __init__(self) -> None:
                self.attention_calls: list[dict[str, object]] = []

            def _coerce_detection_result(self, detection):
                return detection

            def _needs_rgb_frame_for_attention(self, *, detection):
                return True

            def observe_attention_from_frame(self, **kwargs):
                self.attention_calls.append(dict(kwargs))
                return AICameraObservation(
                    observed_at=73.0,
                    camera_online=True,
                    camera_ready=True,
                    camera_ai_ready=True,
                    person_count=0,
                    primary_person_zone=AICameraZone.UNKNOWN,
                    model="local-imx500+mediapipe",
                )

            def observe_gesture_from_frame(self, **kwargs):
                raise AssertionError("gesture path should not run in this test")

            def get_last_attention_debug_details(self):
                return {"attention_face_anchor_state": "no_face_detected"}

            def get_last_gesture_debug_details(self):
                return None

            def close(self):
                return None

        processor = _FakeProcessor()
        provider = RemoteFrameAICameraObservationProvider(
            config=RemoteAICameraProviderConfig(
                base_url="http://10.42.0.2:8767",
                input_format="remote-ai-frame",
            ),
            processor=cast(Any, processor),
        )
        remote_observation = AICameraObservation(
            observed_at=72.0,
            camera_online=True,
            camera_ready=True,
            camera_ai_ready=True,
            person_count=0,
            primary_person_zone=AICameraZone.UNKNOWN,
            model="local-imx500+mediapipe",
        )

        with patch.object(
            provider,
            "_fetch_remote_frame_bundle",
            return_value=(
                remote_observation,
                71.5,
                "local-imx500+mediapipe",
                {"bundle_mode": "detection_frame"},
                "rgb-frame",
            ),
        ):
            snapshot = provider.observe_attention()

        self.assertEqual(processor.attention_calls[0]["frame_rgb"], "rgb-frame")
        self.assertEqual(snapshot.input_format, "remote-ai-frame")
        self.assertTrue(snapshot.observation.camera_ready)
        attention_debug = provider.attention_debug_details()
        self.assertIsNotNone(attention_debug)
        assert attention_debug is not None
        self.assertEqual(attention_debug["transport_mode"], "remote_frame_local_attention")
        self.assertEqual(attention_debug["remote_route"], "observe_frame_bundle")
        self.assertIn("provider_stage_ms", attention_debug)

    def test_remote_frame_provider_preserves_explicit_helper_faults(self) -> None:
        class _FakeProcessor:
            def _coerce_detection_result(self, detection):
                return detection

            def _needs_rgb_frame_for_attention(self, *, detection):
                return True

            def observe_attention_from_frame(self, **kwargs):
                raise AssertionError("attention path should not run on helper fault")

            def observe_gesture_from_frame(self, **kwargs):
                raise AssertionError("gesture path should not run on helper fault")

            def get_last_attention_debug_details(self):
                return None

            def get_last_gesture_debug_details(self):
                return None

            def close(self):
                return None

        provider = RemoteFrameAICameraObservationProvider(
            config=RemoteAICameraProviderConfig(
                base_url="http://10.42.0.2:8767",
                input_format="remote-ai-frame",
            ),
            processor=cast(Any, _FakeProcessor()),
        )
        remote_observation = AICameraObservation(
            observed_at=80.0,
            camera_online=False,
            camera_ready=False,
            camera_ai_ready=False,
            camera_error="remote_ai_camera_unreachable",
        )

        with patch.object(
            provider,
            "_fetch_remote_frame_bundle",
            return_value=(
                remote_observation,
                80.0,
                "local-imx500+mediapipe",
                {"proxy_state": "offline"},
                None,
            ),
        ):
            snapshot = provider.observe_attention()

        self.assertFalse(snapshot.observation.camera_online)
        self.assertEqual(snapshot.observation.camera_error, "remote_ai_camera_unreachable")
        attention_debug = provider.attention_debug_details()
        self.assertIsNotNone(attention_debug)
        assert attention_debug is not None
        self.assertEqual(attention_debug["transport_mode"], "remote_frame_passthrough_fault")
        self.assertEqual(cast(dict[str, object], attention_debug["remote_debug"])["proxy_state"], "offline")
        self.assertEqual(attention_debug["remote_route"], "observe_frame_bundle")

    def test_remote_frame_bundle_requires_frame_png_when_helper_is_healthy(self) -> None:
        class _FakeProcessor:
            def _coerce_detection_result(self, detection):
                return detection

            def _needs_rgb_frame_for_attention(self, *, detection):
                return True

            def observe_attention_from_frame(self, **kwargs):
                raise AssertionError("attention path should not run on bundle decode failure")

            def observe_gesture_from_frame(self, **kwargs):
                raise AssertionError("gesture path should not run on bundle decode failure")

            def get_last_attention_debug_details(self):
                return None

            def get_last_gesture_debug_details(self):
                return None

            def close(self):
                return None

        provider = RemoteFrameAICameraObservationProvider(
            config=RemoteAICameraProviderConfig(
                base_url="http://10.42.0.2:8767",
                input_format="remote-ai-frame",
            ),
            processor=cast(Any, _FakeProcessor()),
        )

        with patch.object(
            provider,
            "_request_json",
            return_value={
                "observation": {
                    "observed_at": 91.0,
                    "camera_online": True,
                    "camera_ready": True,
                    "camera_ai_ready": True,
                    "person_count": 1,
                },
                "captured_at": 90.5,
            },
        ):
            snapshot = provider.observe_gesture()

        self.assertFalse(snapshot.observation.camera_online)
        self.assertEqual(snapshot.observation.camera_error, "remote_ai_camera_missing_frame_bundle")
        gesture_debug = provider.gesture_debug_details()
        self.assertIsNotNone(gesture_debug)
        assert gesture_debug is not None
        self.assertEqual(gesture_debug["transport_mode"], "remote_frame_passthrough_fault")


if __name__ == "__main__":
    unittest.main()
