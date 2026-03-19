from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.proactive.runtime.sensitive_behavior_gate import (
    evaluate_respeaker_sensitive_behavior_gate,
)


class ReSpeakerSensitiveBehaviorGateTests(unittest.TestCase):
    def test_non_sensitive_behavior_is_allowed(self) -> None:
        decision = evaluate_respeaker_sensitive_behavior_gate(
            candidate_sensitivity="normal",
            live_facts=None,
        )

        self.assertTrue(decision.allowed)
        self.assertIsNone(decision.reason)

    def test_sensitive_behavior_is_blocked_for_multi_person_context(self) -> None:
        decision = evaluate_respeaker_sensitive_behavior_gate(
            candidate_sensitivity="private",
            live_facts={
                "camera": {
                    "person_count": 2,
                },
                "audio_policy": {
                    "presence_audio_active": True,
                    "speaker_direction_stable": True,
                },
                "respeaker": {
                    "direction_confidence": 0.91,
                },
            },
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "sensitive_multi_person_context")
        self.assertTrue(decision.multi_person_likely)

    def test_sensitive_behavior_is_blocked_for_low_confidence_audio_context(self) -> None:
        decision = evaluate_respeaker_sensitive_behavior_gate(
            candidate_sensitivity="sensitive",
            live_facts={
                "sensor": {"presence_session_id": 17},
                "audio_policy": {
                    "presence_audio_active": True,
                    "speaker_direction_stable": False,
                },
                "respeaker": {
                    "direction_confidence": 0.42,
                },
            },
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "sensitive_low_confidence_audio_context")
        self.assertTrue(decision.low_confidence_audio)


if __name__ == "__main__":
    unittest.main()
