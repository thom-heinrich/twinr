from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.display.respeaker_hci import DisplayReSpeakerHciStore, parse_respeaker_hci_state
from twinr.ops.events import TwinrOpsEventStore


class DisplayReSpeakerHciTests(unittest.TestCase):
    def test_parse_respeaker_hci_state_surfaces_muted_and_noise_blocked_fields(self) -> None:
        entries = (
            {
                "created_at": "2026-03-19T10:00:00Z",
                "event": "proactive_observation",
                "message": "changed",
                "data": {
                    "speech_detected": True,
                    "audio_device_runtime_mode": "audio_ready",
                    "audio_host_control_ready": True,
                    "audio_mute_active": True,
                    "audio_azimuth_deg": 277,
                    "audio_direction_confidence": 0.84,
                    "room_busy_or_overlapping": True,
                    "resume_window_open": True,
                    "audio_initiative_block_reason": "mute_blocks_voice_capture",
                    "respeaker_runtime_alert_code": "mic_muted",
                },
            },
        )

        state = parse_respeaker_hci_state(entries)

        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(
            state.state_fields(),
            (
                ("Mikrofon", "stumm"),
                ("Audio", "laut"),
                ("Richtung", "277°"),
                ("Ring", "stumm"),
            ),
        )
        self.assertTrue(state.direction_hint_available)
        self.assertIn("mic muted", state.hardware_log_lines())
        self.assertIn("heard speech", state.hardware_log_lines())
        self.assertIn("noise blocked overlap", state.hardware_log_lines())
        self.assertIn("resume window open", state.hardware_log_lines())

    def test_parse_respeaker_hci_state_surfaces_listening_speech_direction_and_ring_semantics(self) -> None:
        entries = (
            {
                "created_at": "2026-03-19T10:05:00Z",
                "event": "proactive_observation",
                "message": "changed",
                "data": {
                    "runtime_status": "listening",
                    "speech_detected": True,
                    "audio_recent_speech_age_s": 0.4,
                    "audio_device_runtime_mode": "audio_ready",
                    "audio_host_control_ready": True,
                    "audio_azimuth_deg": 91,
                    "audio_direction_confidence": 0.88,
                    "respeaker_runtime_alert_code": "ready",
                },
            },
        )

        state = parse_respeaker_hci_state(entries)

        self.assertIsNotNone(state)
        assert state is not None
        self.assertTrue(state.listening)
        self.assertTrue(state.heard_speech)
        self.assertTrue(state.direction_hint_available)
        self.assertEqual(state.led_ring_semantics, "listening_mute_only")
        self.assertEqual(state.led_ring_mode, "listening")
        self.assertEqual(
            state.state_fields(),
            (
                ("Mikrofon", "hört"),
                ("Audio", "Sprache"),
                ("Richtung", "91°"),
                ("Ring", "hört"),
            ),
        )
        self.assertIn("listening", state.hardware_log_lines())
        self.assertIn("heard speech", state.hardware_log_lines())

    def test_hci_store_reads_dfu_and_unavailable_states_from_event_store(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            event_store = TwinrOpsEventStore.from_config(config)
            event_store.append(
                event="respeaker_runtime_alert",
                message="ReSpeaker is visible on USB but has no ALSA capture device.",
                level="warning",
                data={
                    "alert_code": "dfu_mode",
                    "device_runtime_mode": "usb_visible_no_capture",
                    "host_control_ready": False,
                },
            )
            store = DisplayReSpeakerHciStore.from_config(config)

            state = store.load()

        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state.runtime_alert_code, "dfu_mode")
        self.assertEqual(state.state_fields(), (("ReSpeaker", "DFU"),))
