from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.camera_ai.gesture_candidate_capture import GestureCandidateCaptureStore


class GestureCandidateCaptureTests(unittest.TestCase):
    def test_store_saves_candidate_frame_and_metadata_for_pose_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GestureCandidateCaptureStore(
                capture_dir=tmpdir,
                cooldown_s=0.0,
                max_images=4,
            )

            result = store.maybe_capture(
                observed_at=1710000000.125,
                frame_rgb=np.full((10, 12, 3), 127, dtype=np.uint8),
                debug_details={
                    "pose_hint_source": "fresh_mediapipe",
                    "pose_hint_confidence": 0.81,
                    "final_resolved_source": "none",
                },
            )

            self.assertTrue(result.saved)
            self.assertIn("pose_hint", result.reasons)
            self.assertIsNotNone(result.image_path)
            self.assertIsNotNone(result.metadata_path)
            self.assertTrue(Path(result.image_path or "").is_file())
            metadata = json.loads(Path(result.metadata_path or "").read_text(encoding="utf-8"))
            self.assertEqual(metadata["reasons"], ["pose_hint"])
            self.assertEqual(metadata["gesture_debug"]["pose_hint_source"], "fresh_mediapipe")

    def test_store_honors_capture_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GestureCandidateCaptureStore(
                capture_dir=tmpdir,
                cooldown_s=2.0,
                max_images=4,
            )
            debug = {
                "live_hand_count": 1,
                "final_resolved_source": "none",
            }

            first = store.maybe_capture(
                observed_at=1710000000.0,
                frame_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
                debug_details=debug,
            )
            second = store.maybe_capture(
                observed_at=1710000001.0,
                frame_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
                debug_details=debug,
            )

            self.assertTrue(first.saved)
            self.assertFalse(second.saved)
            self.assertEqual(second.skipped_reason, "cooldown_active")

    def test_store_saves_frame_for_forensics_zero_signal_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GestureCandidateCaptureStore(
                capture_dir=tmpdir,
                cooldown_s=0.0,
                max_images=4,
            )

            result = store.maybe_capture(
                observed_at=1710000002.0,
                frame_rgb=np.full((8, 8, 3), 64, dtype=np.uint8),
                debug_details={
                    "forensics_zero_signal_capture_requested": True,
                    "final_resolved_source": "none",
                },
            )

            self.assertTrue(result.saved)
            self.assertEqual(result.reasons, ("forensics_zero_signal",))
            self.assertTrue(Path(result.image_path or "").is_file())
            metadata = json.loads(Path(result.metadata_path or "").read_text(encoding="utf-8"))
            self.assertEqual(metadata["reasons"], ["forensics_zero_signal"])
            self.assertTrue(metadata["gesture_debug"]["forensics_zero_signal_capture_requested"])

    def test_store_saves_frame_when_person_roi_localizes_hand_without_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GestureCandidateCaptureStore(
                capture_dir=tmpdir,
                cooldown_s=0.0,
                max_images=4,
            )

            result = store.maybe_capture(
                observed_at=1710000003.0,
                frame_rgb=np.full((8, 8, 3), 96, dtype=np.uint8),
                debug_details={
                    "person_roi_detection_count": 1,
                    "person_roi_combined_gesture": "none",
                    "final_resolved_source": "none",
                },
            )

            self.assertTrue(result.saved)
            self.assertEqual(result.reasons, ("person_roi_hand_without_symbol",))
            metadata = json.loads(Path(result.metadata_path or "").read_text(encoding="utf-8"))
            self.assertEqual(metadata["reasons"], ["person_roi_hand_without_symbol"])
            self.assertEqual(metadata["gesture_debug"]["person_roi_detection_count"], 1)

    def test_store_prunes_older_capture_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GestureCandidateCaptureStore(
                capture_dir=tmpdir,
                cooldown_s=0.0,
                max_images=2,
            )
            debug = {
                "live_hand_count": 1,
                "final_resolved_source": "none",
            }

            for offset in range(3):
                result = store.maybe_capture(
                    observed_at=1710000000.0 + offset,
                    frame_rgb=np.full((6, 6, 3), 20 * offset, dtype=np.uint8),
                    debug_details=debug,
                )
                self.assertTrue(result.saved)

            image_files = sorted(Path(tmpdir).glob("*.jpg"))
            metadata_files = sorted(Path(tmpdir).glob("*.json"))
            self.assertEqual(len(image_files), 2)
            self.assertEqual(len(metadata_files), 2)
            remaining_stems = {path.stem for path in image_files}
            self.assertEqual(remaining_stems, {path.stem for path in metadata_files})


if __name__ == "__main__":
    unittest.main()
