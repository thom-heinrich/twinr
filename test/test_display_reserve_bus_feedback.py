from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.display.reserve_bus_feedback import (
    DisplayReserveBusFeedbackSignal,
    DisplayReserveBusFeedbackStore,
)


class DisplayReserveBusFeedbackStoreTests(unittest.TestCase):
    def test_store_roundtrip_and_expiry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            store = DisplayReserveBusFeedbackStore.from_config(config)
            now = datetime(2026, 3, 22, 13, 0, tzinfo=timezone.utc)

            saved = store.save(
                DisplayReserveBusFeedbackSignal(
                    topic_key="ai companions",
                    reaction="immediate_engagement",
                    intensity=0.92,
                    reason="The user immediately spoke about the shown card.",
                    requested_at=now.isoformat(),
                    expires_at=(now + timedelta(hours=2)).isoformat(),
                )
            )
            loaded = store.load_active(now=now + timedelta(minutes=30))
            expired = store.load_active(now=now + timedelta(hours=3))

        self.assertEqual(loaded, saved)
        self.assertIsNone(expired)


if __name__ == "__main__":
    unittest.main()
