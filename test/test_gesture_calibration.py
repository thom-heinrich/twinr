from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.proactive.social.engine import SocialFineHandGesture
from twinr.proactive.social.gesture_calibration import GestureCalibrationProfile


class GestureCalibrationProfileTests(unittest.TestCase):
    def test_defaults_include_peace_ok_and_middle_finger(self) -> None:
        profile = GestureCalibrationProfile.defaults()

        self.assertIn(SocialFineHandGesture.PEACE_SIGN, profile.fine_hand)
        self.assertIn(SocialFineHandGesture.OK_SIGN, profile.fine_hand)
        self.assertIn(SocialFineHandGesture.MIDDLE_FINGER, profile.fine_hand)
        self.assertEqual(profile.fine_hand[SocialFineHandGesture.PEACE_SIGN].confirm_samples, 1)
        self.assertEqual(profile.fine_hand[SocialFineHandGesture.THUMBS_DOWN].confirm_samples, 1)
        self.assertEqual(profile.fine_hand[SocialFineHandGesture.OK_SIGN].confirm_samples, 1)
        self.assertEqual(profile.fine_hand[SocialFineHandGesture.MIDDLE_FINGER].confirm_samples, 1)
        self.assertAlmostEqual(profile.fine_hand[SocialFineHandGesture.THUMBS_UP].min_visible_s, 1.0, places=3)
        self.assertAlmostEqual(profile.fine_hand[SocialFineHandGesture.THUMBS_DOWN].min_visible_s, 1.0, places=3)
        self.assertAlmostEqual(profile.fine_hand[SocialFineHandGesture.PEACE_SIGN].min_visible_s, 1.0, places=3)

    def test_runtime_profile_loads_json_override(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            calibration_path = Path(temp_dir) / "state" / "mediapipe" / "gesture_calibration.json"
            calibration_path.parent.mkdir(parents=True, exist_ok=True)
            calibration_path.write_text(
                json.dumps(
                    {
                        "fine_hand": {
                            "peace_sign": {
                                "min_confidence": 0.79,
                                "confirm_samples": 3,
                                "hold_s": 0.51,
                                "min_visible_s": 1.2,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            profile = GestureCalibrationProfile.from_runtime_config(TwinrConfig(project_root=temp_dir))

        policy = profile.fine_hand[SocialFineHandGesture.PEACE_SIGN]
        self.assertEqual(policy.confirm_samples, 3)
        self.assertAlmostEqual(policy.min_confidence, 0.79, places=3)
        self.assertAlmostEqual(policy.hold_s, 0.51, places=3)
        self.assertAlmostEqual(policy.min_visible_s, 1.2, places=3)
        self.assertTrue(str(profile.source_path or "").endswith("state/mediapipe/gesture_calibration.json"))
