from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.display.ambient_impulse_cues import DisplayAmbientImpulseCueStore
from twinr.display.ambient_impulse_history import DisplayAmbientImpulseHistoryStore
from twinr.proactive.runtime.display_social_reserve import DisplaySocialReservePublisher


class DisplaySocialReservePublisherTests(unittest.TestCase):
    def test_publish_routes_visual_first_social_prompt_into_reserve_lane(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            publisher = DisplaySocialReservePublisher.from_config(config)
            now = datetime(2026, 3, 22, 14, 30, tzinfo=timezone.utc)

            result = publisher.publish(
                trigger_id="person_returned",
                prompt_text="Schön dich zu sehen. Wie geht's dir?",
                display_reason="background_media_active",
                hold_seconds=180.0,
                now=now,
            )

            cue = DisplayAmbientImpulseCueStore.from_config(config).load_active(now=now)
            history = DisplayAmbientImpulseHistoryStore.from_config(config).load()

        self.assertIsNotNone(cue)
        assert cue is not None
        self.assertEqual(cue.source, "proactive_social")
        self.assertEqual(cue.topic_key, "person_returned")
        self.assertEqual(cue.headline, "Schön dich zu sehen. Wie geht's dir?")
        self.assertEqual(cue.body, "")
        self.assertEqual(cue.action, "ask_one")
        self.assertEqual(cue.attention_state, "foreground")
        self.assertEqual(result.cue.headline, cue.headline)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].source, "social_trigger")
        self.assertEqual(history[0].topic_key, "person_returned")
        self.assertEqual(history[0].headline, cue.headline)
        self.assertEqual(history[0].metadata["display_reason"], "background_media_active")


if __name__ == "__main__":
    unittest.main()
