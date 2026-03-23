from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.proactive.runtime.gesture_wakeup_priority import decide_gesture_wakeup_priority
from twinr.proactive.runtime.presence import PresenceSessionSnapshot


class GestureWakeupPriorityTests(unittest.TestCase):
    def test_blocks_visual_wakeup_while_runtime_is_not_waiting(self) -> None:
        decision = decide_gesture_wakeup_priority(
            runtime_status_value="answering",
            voice_path_enabled=True,
            presence_snapshot=None,
            recent_speech_guard_s=2.0,
        )

        self.assertFalse(decision.allow)
        self.assertEqual(decision.reason, "gesture_wakeup_suppressed_runtime_answering")

    def test_blocks_visual_wakeup_for_recent_speech_when_voice_path_is_enabled(self) -> None:
        snapshot = PresenceSessionSnapshot(
            armed=True,
            reason="speech_recent",
            last_speech_age_s=0.4,
        )

        decision = decide_gesture_wakeup_priority(
            runtime_status_value="waiting",
            voice_path_enabled=True,
            presence_snapshot=snapshot,
            recent_speech_guard_s=2.0,
        )

        self.assertFalse(decision.allow)
        self.assertEqual(decision.reason, "gesture_wakeup_suppressed_recent_speech")

    def test_allows_visual_wakeup_when_voice_path_is_idle(self) -> None:
        snapshot = PresenceSessionSnapshot(
            armed=True,
            reason="speech_old",
            last_speech_age_s=4.5,
        )

        decision = decide_gesture_wakeup_priority(
            runtime_status_value="waiting",
            voice_path_enabled=True,
            presence_snapshot=snapshot,
            recent_speech_guard_s=2.0,
        )

        self.assertTrue(decision.allow)
        self.assertEqual(decision.reason, "gesture_wakeup_allowed")


if __name__ == "__main__":
    unittest.main()
