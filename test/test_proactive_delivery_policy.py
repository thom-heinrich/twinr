from datetime import datetime
from pathlib import Path
import sys
import tempfile
import unittest
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.agent.workflows.realtime_runtime.proactive_delivery import ProactiveDeliveryPolicy
from twinr.proactive.runtime.audio_policy import ReSpeakerAudioPolicySnapshot


class ProactiveDeliveryPolicyTests(unittest.TestCase):
    def test_quiet_hours_force_visual_only_for_non_safety_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            policy = ProactiveDeliveryPolicy.from_config(
                TwinrConfig(
                    project_root=temp_dir,
                    proactive_quiet_hours_visual_only_enabled=True,
                    proactive_quiet_hours_start_local="21:00",
                    proactive_quiet_hours_end_local="07:00",
                )
            )

        decision = policy.decide(
            monotonic_now=100.0,
            local_now=datetime(2026, 3, 19, 22, 30, tzinfo=ZoneInfo("Europe/Berlin")),
            source_id="attention_window",
            safety_exempt=False,
            audio_policy_snapshot=None,
        )

        self.assertEqual(decision.channel, "display")
        self.assertEqual(decision.reason, "quiet_hours_visual_only")

    def test_recent_ignored_prompt_switches_future_attempts_to_visual_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            policy = ProactiveDeliveryPolicy.from_config(
                TwinrConfig(
                    project_root=temp_dir,
                    proactive_quiet_hours_visual_only_enabled=False,
                    proactive_visual_first_audio_global_cooldown_s=300.0,
                    proactive_visual_first_audio_source_repeat_cooldown_s=900.0,
                )
            )

        policy.note_ignored(source_id="attention_window", monotonic_now=50.0)
        decision = policy.decide(
            monotonic_now=100.0,
            local_now=datetime(2026, 3, 19, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")),
            source_id="attention_window",
            safety_exempt=False,
            audio_policy_snapshot=None,
        )

        self.assertEqual(decision.channel, "display")
        self.assertEqual(decision.reason, "recent_audio_display_first_cooldown")

    def test_background_media_defers_speech_to_display(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            policy = ProactiveDeliveryPolicy.from_config(
                TwinrConfig(
                    project_root=temp_dir,
                    proactive_quiet_hours_visual_only_enabled=False,
                )
            )

        decision = policy.decide(
            monotonic_now=100.0,
            local_now=datetime(2026, 3, 19, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")),
            source_id="attention_window",
            safety_exempt=False,
            audio_policy_snapshot=ReSpeakerAudioPolicySnapshot(
                observed_at=10.0,
                background_media_likely=True,
                speech_delivery_defer_reason="background_media_active",
            ),
        )

        self.assertEqual(decision.channel, "display")
        self.assertEqual(decision.reason, "background_media_active")


if __name__ == "__main__":
    unittest.main()
