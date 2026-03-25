from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.proactive.runtime.display_reserve_runtime import (
    DisplayReserveRuntimePublisher,
    DisplayReserveRuntimeRequest,
)


class DisplayReserveRuntimePublisherTests(unittest.TestCase):
    def test_runtime_publisher_writes_one_shared_cue_and_history_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            publisher = DisplayReserveRuntimePublisher.from_config(
                config,
                default_source="proactive_ambient_impulse",
            )
            now = datetime(2026, 3, 22, 16, 30, tzinfo=timezone.utc)

            result = publisher.publish(
                DisplayReserveRuntimeRequest(
                    topic_key="ai companions",
                    title="AI companions",
                    cue_source="proactive_ambient_impulse",
                    history_source="world",
                    action="invite_follow_up",
                    attention_state="shared_thread",
                    eyebrow="",
                    headline="Wie entwickelst du deinen AI Begleiter?",
                    body="Mich interessiert, was dir daran gerade wichtig ist.",
                    symbol="heart",
                    accent="warm",
                    hold_seconds=240.0,
                    reason="plan[0] world_shared_thread",
                    candidate_family="world",
                    match_anchors=("AI companions",),
                    metadata={"lane": "reserve_bus"},
                ),
                now=now,
            )
            cue = publisher.active_store.load_active(now=now + timedelta(seconds=1))
            history = publisher.history_store.load()

        self.assertIsNotNone(cue)
        self.assertEqual(result.cue, cue)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].source, "world")
        self.assertEqual(history[0].metadata["candidate_family"], "world")
        self.assertEqual(history[0].metadata["lane"], "reserve_bus")
        self.assertEqual(history[0].headline, "Wie entwickelst du deinen AI Begleiter?")

    def test_show_visible_only_updates_cue_without_second_history_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            publisher = DisplayReserveRuntimePublisher.from_config(
                config,
                default_source="proactive_ambient_impulse",
            )
            now = datetime(2026, 3, 22, 16, 30, tzinfo=timezone.utc)
            request = DisplayReserveRuntimeRequest(
                topic_key="ai companions",
                title="AI companions",
                cue_source="proactive_ambient_impulse",
                history_source="world",
                action="invite_follow_up",
                attention_state="shared_thread",
                eyebrow="",
                headline="Wie entwickelst du deinen AI Begleiter?",
                body="Mich interessiert, was dir daran gerade wichtig ist.",
                symbol="heart",
                accent="warm",
                hold_seconds=240.0,
                reason="plan[0] world_shared_thread",
                candidate_family="world",
                match_anchors=("AI companions",),
                metadata={"lane": "reserve_bus"},
            )

            publisher.publish(request, now=now)
            result = publisher.show_visible_only(
                request,
                now=now + timedelta(minutes=10),
            )
            cue = publisher.active_store.load_active(now=now + timedelta(minutes=10, seconds=1))
            history = publisher.history_store.load()

        self.assertIsNotNone(cue)
        self.assertEqual(result.cue, cue)
        self.assertEqual(len(history), 1)


if __name__ == "__main__":
    unittest.main()
