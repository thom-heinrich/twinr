from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.proactive.runtime.smart_home_context import SmartHomeContextTracker


def _local_facts(*, observed_at: float = 12.0) -> dict[str, object]:
    return {
        "sensor": {
            "observed_at": observed_at,
            "captured_at": observed_at,
            "voice_activation_armed": True,
            "voice_activation_presence_reason": "person_visible",
        },
        "pir": {
            "motion_detected": True,
            "low_motion": False,
            "no_motion_for_s": 0.0,
        },
        "camera": {
            "person_visible": True,
            "person_visible_for_s": 6.0,
        },
        "vad": {
            "speech_detected": False,
        },
        "audio_policy": {
            "presence_audio_active": False,
            "recent_follow_up_speech": False,
        },
    }


def _smart_home_facts() -> dict[str, object]:
    return {
        "smart_home": {
            "sensor_stream_live": True,
            "motion_detected": True,
            "motion_active_by_entity": {
                "route:1:same-room-motion": True,
                "route:1:hall-motion": False,
            },
            "device_offline": True,
            "offline_entity_ids": ["route:1:hall-motion"],
            "alarm_triggered": False,
            "recent_events": [
                {
                    "event_id": "evt-same-room-motion",
                    "provider": "hue",
                    "entity_id": "route:1:same-room-motion",
                    "event_kind": "motion_detected",
                    "observed_at": "2026-03-22T10:00:00Z",
                },
                {
                    "event_id": "evt-hall-motion",
                    "provider": "hue",
                    "entity_id": "route:1:hall-motion",
                    "event_kind": "motion_detected",
                    "observed_at": "2026-03-22T10:00:04Z",
                },
                {
                    "event_id": "evt-same-room-button",
                    "provider": "hue",
                    "entity_id": "route:1:same-room-button",
                    "event_kind": "button_pressed",
                    "observed_at": "2026-03-22T10:00:06Z",
                },
                {
                    "event_id": "evt-hall-offline",
                    "provider": "hue",
                    "entity_id": "route:1:hall-motion",
                    "event_kind": "device_offline",
                    "observed_at": "2026-03-22T10:00:09Z",
                },
            ],
        }
    }


class SmartHomeContextTrackerTests(unittest.TestCase):
    def test_from_env_reads_smart_home_context_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "TWINR_SMART_HOME_SAME_ROOM_ENTITY_IDS=route:1:same-room-motion, route:1:same-room-button",
                        "TWINR_SMART_HOME_SAME_ROOM_MOTION_WINDOW_S=45",
                        "TWINR_SMART_HOME_SAME_ROOM_BUTTON_WINDOW_S=12",
                        "TWINR_SMART_HOME_HOME_OCCUPANCY_WINDOW_S=240",
                        "TWINR_SMART_HOME_STREAM_STALE_AFTER_S=33",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            config = TwinrConfig.from_env(env_path)

        self.assertEqual(
            config.smart_home_same_room_entity_ids,
            ("route:1:same-room-motion", "route:1:same-room-button"),
        )
        self.assertEqual(config.smart_home_same_room_motion_window_s, 45.0)
        self.assertEqual(config.smart_home_same_room_button_window_s, 12.0)
        self.assertEqual(config.smart_home_home_occupancy_window_s, 240.0)
        self.assertEqual(config.smart_home_stream_stale_after_s, 33.0)

    def test_tracker_derives_layered_context_from_local_and_smart_home_facts(self) -> None:
        config = TwinrConfig(
            smart_home_same_room_entity_ids=("route:1:same-room-motion", "route:1:same-room-button"),
            smart_home_same_room_motion_window_s=60.0,
            smart_home_same_room_button_window_s=30.0,
            smart_home_home_occupancy_window_s=180.0,
            smart_home_stream_stale_after_s=120.0,
        )
        tracker = SmartHomeContextTracker.from_config(config)
        merged_facts = _local_facts(observed_at=12.0)
        merged_facts.update(_smart_home_facts())

        update = tracker.observe(
            observed_at=12.0,
            live_facts=merged_facts,
            incoming_facts=_smart_home_facts(),
        )
        facts = update.snapshot.apply_to_facts(merged_facts)

        self.assertTrue(facts["near_device_presence"]["occupied_likely"])
        self.assertTrue(facts["near_device_presence"]["person_visible"])
        self.assertTrue(facts["near_device_presence"]["voice_activation_armed"])
        self.assertTrue(facts["room_context"]["configured"])
        self.assertTrue(facts["room_context"]["available"])
        self.assertTrue(facts["room_context"]["same_room_motion_recent"])
        self.assertTrue(facts["room_context"]["same_room_button_recent"])
        self.assertEqual(
            facts["room_context"]["matched_entity_ids"],
            ["route:1:same-room-motion", "route:1:same-room-button"],
        )
        self.assertTrue(facts["home_context"]["home_occupied_likely"])
        self.assertTrue(facts["home_context"]["other_room_motion_recent"])
        self.assertTrue(facts["home_context"]["device_offline"])
        self.assertTrue(facts["home_context"]["stream_healthy"])
        self.assertIn("room_context.same_room_motion_recent", update.event_names)
        self.assertIn("home_context.other_room_motion_recent", update.event_names)

    def test_tracker_fails_room_and_home_context_closed_when_stream_goes_stale(self) -> None:
        config = TwinrConfig(
            smart_home_same_room_entity_ids=("route:1:same-room-motion",),
            smart_home_same_room_motion_window_s=60.0,
            smart_home_home_occupancy_window_s=180.0,
            smart_home_stream_stale_after_s=30.0,
        )
        tracker = SmartHomeContextTracker.from_config(config)
        merged_facts = _local_facts(observed_at=10.0)
        merged_facts.update(_smart_home_facts())

        tracker.observe(
            observed_at=10.0,
            live_facts=merged_facts,
            incoming_facts=_smart_home_facts(),
        )
        stale_update = tracker.observe(
            observed_at=50.0,
            live_facts=merged_facts,
            incoming_facts=_local_facts(observed_at=50.0),
        )

        self.assertTrue(stale_update.snapshot.room_context.sensor_stale)
        self.assertFalse(stale_update.snapshot.room_context.same_room_motion_recent)
        self.assertFalse(stale_update.snapshot.home_context.stream_healthy)
        self.assertFalse(stale_update.snapshot.home_context.other_room_motion_recent)
        self.assertIn("room_context.sensor_stale", stale_update.event_names)
        self.assertIn("home_context.stream_unhealthy", stale_update.event_names)


if __name__ == "__main__":
    unittest.main()
