from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.display.face_cues import DisplayFaceCue
from twinr.display.face_expressions import DisplayFaceGazeDirection
from twinr.proactive.runtime.display_attention import (
    DisplayAttentionCuePublisher,
    derive_display_attention_cue,
    resolve_display_attention_refresh_interval,
)


class DisplayAttentionCueTests(unittest.TestCase):
    def test_refresh_interval_uses_faster_default(self) -> None:
        self.assertEqual(resolve_display_attention_refresh_interval(TwinrConfig()), 0.2)

    def test_derives_user_facing_rightward_gaze_from_left_camera_anchor(self) -> None:
        decision = derive_display_attention_cue(
            config=TwinrConfig(proactive_capture_interval_s=6.0, display_attention_refresh_interval_s=0.6),
            live_facts={
                "camera": {
                    "person_visible": True,
                    "primary_person_center_x": 0.12,
                    "primary_person_center_y": 0.5,
                    "looking_toward_device": True,
                },
                "vad": {
                    "speech_detected": False,
                },
            },
        )

        self.assertTrue(decision.active)
        self.assertEqual(decision.reason, "visible_person")
        self.assertEqual(decision.gaze, DisplayFaceGazeDirection.RIGHT)
        self.assertLess(decision.hold_seconds, 6.0)
        self.assertEqual(decision.head_dx, 2)

    def test_derives_diagonal_upward_gaze_when_person_is_above_camera_level(self) -> None:
        decision = derive_display_attention_cue(
            config=TwinrConfig(proactive_capture_interval_s=6.0, display_attention_refresh_interval_s=0.6),
            live_facts={
                "camera": {
                    "person_visible": True,
                    "primary_person_center_x": 0.12,
                    "primary_person_center_y": 0.18,
                    "looking_toward_device": True,
                },
                "vad": {
                    "speech_detected": False,
                },
            },
        )

        self.assertTrue(decision.active)
        self.assertEqual(decision.gaze, DisplayFaceGazeDirection.UP_RIGHT)
        self.assertLess(decision.cue_gaze_y, 0)
        self.assertLess(decision.head_dy, 0)

    def test_derives_downward_gaze_when_person_is_below_camera_level(self) -> None:
        decision = derive_display_attention_cue(
            config=TwinrConfig(proactive_capture_interval_s=6.0, display_attention_refresh_interval_s=0.6),
            live_facts={
                "camera": {
                    "person_visible": True,
                    "primary_person_center_x": 0.50,
                    "primary_person_center_y": 0.84,
                },
                "vad": {
                    "speech_detected": False,
                },
            },
        )

        self.assertTrue(decision.active)
        self.assertEqual(decision.gaze, DisplayFaceGazeDirection.DOWN)
        self.assertGreater(decision.cue_gaze_y, 0)
        self.assertGreater(decision.head_dy, 0)

    def test_derives_subtle_head_turn_before_full_side_gaze(self) -> None:
        decision = derive_display_attention_cue(
            config=TwinrConfig(proactive_capture_interval_s=6.0, display_attention_refresh_interval_s=0.6),
            live_facts={
                "camera": {
                    "person_visible": True,
                    "primary_person_center_x": 0.45,
                    "primary_person_center_y": 0.5,
                },
                "vad": {
                    "speech_detected": False,
                },
            },
        )

        self.assertTrue(decision.active)
        self.assertEqual(decision.gaze, DisplayFaceGazeDirection.CENTER)
        self.assertEqual(decision.cue_gaze_x, 1)
        self.assertEqual(decision.head_dx, 1)

    def test_committed_side_gaze_uses_small_same_side_step_before_full_edge_lock(self) -> None:
        decision = derive_display_attention_cue(
            config=TwinrConfig(proactive_capture_interval_s=6.0, display_attention_refresh_interval_s=0.6),
            live_facts={
                "camera": {
                    "person_visible": True,
                    "primary_person_center_x": 0.40,
                    "primary_person_center_y": 0.5,
                },
                "vad": {
                    "speech_detected": False,
                },
            },
        )

        self.assertTrue(decision.active)
        self.assertEqual(decision.gaze, DisplayFaceGazeDirection.RIGHT)
        self.assertEqual(decision.cue_gaze_x, 1)
        self.assertEqual(decision.head_dx, 1)

    def test_derives_speaking_mouth_when_visible_person_is_current_speaker(self) -> None:
        decision = derive_display_attention_cue(
            config=TwinrConfig(proactive_capture_interval_s=1.0),
            live_facts={
                "camera": {
                    "person_visible": True,
                    "primary_person_zone": "center",
                    "engaged_with_device": True,
                },
                "vad": {
                    "speech_detected": True,
                },
                "speaker_association": {
                    "associated": True,
                },
            },
        )

        self.assertTrue(decision.active)
        self.assertEqual(decision.reason, "speaker_visible_person")
        self.assertEqual(decision.expression().mouth, "speak")

    def test_derives_session_focus_cue_from_serialized_attention_target(self) -> None:
        decision = derive_display_attention_cue(
            config=TwinrConfig(proactive_capture_interval_s=1.0),
            live_facts={
                "camera": {
                    "person_visible": False,
                },
                "attention_target": {
                    "state": "holding_session_focus",
                    "active": True,
                    "target_horizontal": "left",
                    "target_vertical": "center",
                    "focus_source": "speaker_association",
                    "session_focus_active": True,
                    "speaker_locked": False,
                    "showing_intent_active": False,
                    "confidence": 0.81,
                },
            },
        )

        self.assertTrue(decision.active)
        self.assertEqual(decision.reason, "holding_session_focus")
        self.assertEqual(decision.gaze, DisplayFaceGazeDirection.RIGHT)
        self.assertEqual(decision.expression().brows, "soft")

    def test_ignores_serialized_attention_target_when_camera_is_explicitly_unavailable(self) -> None:
        decision = derive_display_attention_cue(
            config=TwinrConfig(proactive_capture_interval_s=1.0),
            live_facts={
                "camera": {
                    "person_visible": False,
                    "camera_online": False,
                    "camera_ready": False,
                },
                "attention_target": {
                    "state": "holding_session_focus",
                    "active": True,
                    "target_horizontal": "left",
                    "target_vertical": "up",
                    "focus_source": "speaker_association",
                    "session_focus_active": True,
                    "speaker_locked": False,
                    "showing_intent_active": False,
                    "confidence": 0.81,
                },
            },
        )

        self.assertFalse(decision.active)
        self.assertEqual(decision.reason, "no_visible_person")

    def test_publisher_does_not_overwrite_active_foreign_cue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir, proactive_capture_interval_s=1.0)
            publisher = DisplayAttentionCuePublisher.from_config(config)
            publisher.store.save(
                DisplayFaceCue(source="operator", gaze_x=2, mouth="smile"),
                hold_seconds=10.0,
                now=datetime(2026, 3, 20, 8, 0, tzinfo=timezone.utc),
            )

            result = publisher.publish_from_facts(
                config=config,
                live_facts={
                    "camera": {
                        "person_visible": True,
                        "primary_person_center_x": 0.1,
                    },
                    "vad": {
                        "speech_detected": False,
                    },
                },
                now=datetime(2026, 3, 20, 8, 0, 1, tzinfo=timezone.utc),
            )
            loaded = publisher.store.load_active(now=datetime(2026, 3, 20, 8, 0, 1, tzinfo=timezone.utc))

        self.assertEqual(result.action, "blocked_foreign_cue")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.source, "operator")
        self.assertEqual(loaded.gaze_x, 2)

    def test_publisher_clears_its_own_cue_when_no_person_is_visible_after_hold_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir, proactive_capture_interval_s=1.0)
            publisher = DisplayAttentionCuePublisher.from_config(config)
            publisher.publish_from_facts(
                config=config,
                live_facts={
                    "camera": {
                        "person_visible": True,
                        "primary_person_center_x": 0.82,
                    },
                    "vad": {
                        "speech_detected": False,
                    },
                },
                now=datetime(2026, 3, 20, 8, 2, tzinfo=timezone.utc),
            )

            result = publisher.publish_from_facts(
                config=config,
                live_facts={
                    "camera": {
                        "person_visible": False,
                    },
                    "vad": {
                        "speech_detected": False,
                    },
                },
                now=datetime(2026, 3, 20, 8, 2, 3, tzinfo=timezone.utc),
            )

            loaded = publisher.store.load_active(now=datetime(2026, 3, 20, 8, 2, 3, tzinfo=timezone.utc))

        self.assertEqual(result.action, "cleared")
        self.assertIsNone(loaded)

    def test_publisher_refreshes_matching_cue_before_it_expires(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                proactive_capture_interval_s=6.0,
                display_attention_refresh_interval_s=0.6,
                display_face_cue_ttl_s=4.0,
            )
            publisher = DisplayAttentionCuePublisher.from_config(config)
            first_result = publisher.publish_from_facts(
                config=config,
                live_facts={
                    "camera": {
                        "person_visible": True,
                        "primary_person_center_x": 0.12,
                    },
                    "vad": {
                        "speech_detected": False,
                    },
                },
                now=datetime(2026, 3, 20, 8, 3, tzinfo=timezone.utc),
            )
            initial = publisher.store.load_active(now=datetime(2026, 3, 20, 8, 3, 0, 1000, tzinfo=timezone.utc))
            assert initial is not None

            refresh_result = publisher.publish_from_facts(
                config=config,
                live_facts={
                    "camera": {
                        "person_visible": True,
                        "primary_person_center_x": 0.12,
                    },
                    "vad": {
                        "speech_detected": False,
                    },
                },
                now=datetime(2026, 3, 20, 8, 3, 2, 700000, tzinfo=timezone.utc),
            )
            refreshed = publisher.store.load_active(now=datetime(2026, 3, 20, 8, 3, 4, 200000, tzinfo=timezone.utc))

        self.assertEqual(first_result.action, "updated")
        self.assertEqual(refresh_result.action, "refreshed")
        self.assertIsNotNone(refreshed)
        assert refreshed is not None
        self.assertEqual(refreshed.gaze_x, 2)
        self.assertGreater(str(refreshed.expires_at), str(initial.expires_at))

    def test_publisher_holds_recent_directional_cue_across_brief_center_jitter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                proactive_capture_interval_s=6.0,
                display_attention_refresh_interval_s=0.6,
                display_face_cue_ttl_s=4.0,
            )
            publisher = DisplayAttentionCuePublisher.from_config(config)
            publisher.publish_from_facts(
                config=config,
                live_facts={
                    "camera": {
                        "person_visible": True,
                        "primary_person_center_x": 0.12,
                    },
                    "vad": {
                        "speech_detected": False,
                    },
                },
                now=datetime(2026, 3, 20, 8, 4, tzinfo=timezone.utc),
            )

            jitter_result = publisher.publish_from_facts(
                config=config,
                live_facts={
                    "camera": {
                        "person_visible": True,
                        "primary_person_center_x": 0.50,
                    },
                    "vad": {
                        "speech_detected": False,
                    },
                },
                now=datetime(2026, 3, 20, 8, 4, 1, tzinfo=timezone.utc),
            )
            held = publisher.store.load_active(now=datetime(2026, 3, 20, 8, 4, 1, 1000, tzinfo=timezone.utc))

            settled_result = publisher.publish_from_facts(
                config=config,
                live_facts={
                    "camera": {
                        "person_visible": True,
                        "primary_person_center_x": 0.50,
                    },
                    "vad": {
                        "speech_detected": False,
                    },
                },
                now=datetime(2026, 3, 20, 8, 4, 3, tzinfo=timezone.utc),
            )
            settled = publisher.store.load_active(now=datetime(2026, 3, 20, 8, 4, 3, 1000, tzinfo=timezone.utc))

        self.assertTrue(jitter_result.decision.reason.endswith("held_direction"))
        self.assertEqual(jitter_result.decision.gaze, DisplayFaceGazeDirection.RIGHT)
        self.assertIsNotNone(held)
        assert held is not None
        self.assertEqual(held.gaze_x, 2)
        self.assertEqual(settled_result.decision.gaze, DisplayFaceGazeDirection.CENTER)
        self.assertIsNotNone(settled)
        assert settled is not None
        self.assertEqual(settled.gaze_x, 0)

    def test_publisher_holds_recent_direction_when_person_is_still_on_same_side(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                proactive_capture_interval_s=6.0,
                display_attention_refresh_interval_s=0.6,
                display_face_cue_ttl_s=4.0,
            )
            publisher = DisplayAttentionCuePublisher.from_config(config)
            publisher.publish_from_facts(
                config=config,
                live_facts={
                    "camera": {
                        "person_visible": True,
                        "primary_person_center_x": 0.12,
                    },
                    "vad": {
                        "speech_detected": False,
                    },
                },
                now=datetime(2026, 3, 20, 8, 4, tzinfo=timezone.utc),
            )

            held_result = publisher.publish_from_facts(
                config=config,
                live_facts={
                    "camera": {
                        "person_visible": True,
                        "primary_person_center_x": 0.46,
                    },
                    "vad": {
                        "speech_detected": False,
                    },
                },
                now=datetime(2026, 3, 20, 8, 4, 3, tzinfo=timezone.utc),
            )
            held = publisher.store.load_active(now=datetime(2026, 3, 20, 8, 4, 3, 1000, tzinfo=timezone.utc))

        self.assertFalse(held_result.decision.reason.endswith("held_direction"))
        self.assertEqual(held_result.decision.gaze, DisplayFaceGazeDirection.CENTER)
        self.assertEqual(held_result.decision.cue_gaze_x, 1)
        self.assertIsNotNone(held)
        assert held is not None
        self.assertEqual(held.gaze_x, 1)

    def test_publisher_updates_same_side_gaze_when_anchor_moves_closer_to_center(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                proactive_capture_interval_s=6.0,
                display_attention_refresh_interval_s=0.6,
                display_face_cue_ttl_s=4.0,
            )
            publisher = DisplayAttentionCuePublisher.from_config(config)
            first_result = publisher.publish_from_facts(
                config=config,
                live_facts={
                    "camera": {
                        "person_visible": True,
                        "primary_person_center_x": 0.12,
                    },
                    "vad": {
                        "speech_detected": False,
                    },
                },
                now=datetime(2026, 3, 20, 8, 4, tzinfo=timezone.utc),
            )

            same_side_result = publisher.publish_from_facts(
                config=config,
                live_facts={
                    "camera": {
                        "person_visible": True,
                        "primary_person_center_x": 0.40,
                    },
                    "vad": {
                        "speech_detected": False,
                    },
                },
                now=datetime(2026, 3, 20, 8, 4, 0, 500000, tzinfo=timezone.utc),
            )

        self.assertEqual(first_result.action, "updated")
        self.assertEqual(same_side_result.action, "updated")
        self.assertEqual(same_side_result.decision.cue_gaze_x, 1)

    def test_publisher_holds_recent_cue_across_brief_visual_dropout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                proactive_capture_interval_s=6.0,
                display_attention_refresh_interval_s=0.6,
                display_face_cue_ttl_s=4.0,
            )
            publisher = DisplayAttentionCuePublisher.from_config(config)
            publisher.publish_from_facts(
                config=config,
                live_facts={
                    "camera": {
                        "person_visible": True,
                        "primary_person_center_x": 0.12,
                    },
                    "vad": {
                        "speech_detected": False,
                    },
                },
                now=datetime(2026, 3, 20, 8, 5, tzinfo=timezone.utc),
            )

            dropout_result = publisher.publish_from_facts(
                config=config,
                live_facts={
                    "camera": {
                        "person_visible": False,
                    },
                    "vad": {
                        "speech_detected": False,
                    },
                },
                now=datetime(2026, 3, 20, 8, 5, 1, tzinfo=timezone.utc),
            )
            held = publisher.store.load_active(now=datetime(2026, 3, 20, 8, 5, 1, 1000, tzinfo=timezone.utc))

            cleared_result = publisher.publish_from_facts(
                config=config,
                live_facts={
                    "camera": {
                        "person_visible": False,
                    },
                    "vad": {
                        "speech_detected": False,
                    },
                },
                now=datetime(2026, 3, 20, 8, 5, 3, tzinfo=timezone.utc),
            )
            cleared = publisher.store.load_active(now=datetime(2026, 3, 20, 8, 5, 3, 1000, tzinfo=timezone.utc))

        self.assertTrue(dropout_result.decision.reason.endswith("held_cue"))
        self.assertTrue(dropout_result.decision.active)
        self.assertIsNotNone(held)
        assert held is not None
        self.assertEqual(held.gaze_x, 2)
        self.assertEqual(cleared_result.action, "cleared")
        self.assertIsNone(cleared)


if __name__ == "__main__":
    unittest.main()
