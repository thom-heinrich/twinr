from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.proactive.runtime.display_reserve_candidates import load_display_reserve_candidates


class DisplayReserveCandidatesCompatibilityTests(unittest.TestCase):
    def test_wrapper_delegates_to_companion_flow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            now = datetime(2026, 3, 22, 9, 0, tzinfo=timezone.utc)
            returned = (
                AmbientDisplayImpulseCandidate(
                    topic_key="ai companions",
                    title="AI companions",
                    source="world",
                    action="brief_update",
                    attention_state="growing",
                    salience=0.8,
                    eyebrow="",
                    headline="AI companions",
                    body="Da ist gerade etwas spannend.",
                    symbol="sparkles",
                    accent="info",
                    reason="test",
                ),
            )

            with patch(
                "twinr.proactive.runtime.display_reserve_candidates.DisplayReserveCompanionFlow.load_candidates",
                return_value=returned,
            ) as mocked:
                candidates = load_display_reserve_candidates(
                    config,
                    local_now=now,
                    max_items=4,
                )

        self.assertEqual(candidates, returned)
        mocked.assert_called_once()


if __name__ == "__main__":
    unittest.main()
