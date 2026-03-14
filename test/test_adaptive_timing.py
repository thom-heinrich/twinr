from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.adaptive_timing import AdaptiveTimingStore
from twinr.config import TwinrConfig


class AdaptiveTimingStoreTests(unittest.TestCase):
    def test_no_speech_timeout_widens_button_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "adaptive-timing.json"
            store = AdaptiveTimingStore(
                path,
                config=TwinrConfig(adaptive_timing_store_path=str(path)),
            )

            updated = store.record_no_speech_timeout(initial_source="button", follow_up=False)

        self.assertEqual(updated.button_start_timeout_s, 8.75)
        self.assertEqual(updated.button_timeout_count, 1)

    def test_repeated_fast_starts_shrink_button_window_slowly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "adaptive-timing.json"
            store = AdaptiveTimingStore(
                path,
                config=TwinrConfig(adaptive_timing_store_path=str(path)),
            )
            store.record_no_speech_timeout(initial_source="button", follow_up=False)
            store.record_capture(
                initial_source="button",
                follow_up=False,
                speech_started_after_ms=1800,
                resumed_after_pause_count=0,
            )
            store.record_capture(
                initial_source="button",
                follow_up=False,
                speech_started_after_ms=1900,
                resumed_after_pause_count=0,
            )
            updated = store.record_capture(
                initial_source="button",
                follow_up=False,
                speech_started_after_ms=1700,
                resumed_after_pause_count=0,
            )

        self.assertEqual(updated.button_start_timeout_s, 8.6)
        self.assertEqual(updated.button_fast_start_streak, 0)

    def test_resumed_pause_expands_pause_and_grace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "adaptive-timing.json"
            store = AdaptiveTimingStore(
                path,
                config=TwinrConfig(adaptive_timing_store_path=str(path)),
            )

            updated = store.record_capture(
                initial_source="button",
                follow_up=False,
                speech_started_after_ms=2400,
                resumed_after_pause_count=2,
            )

        self.assertEqual(updated.speech_pause_ms, 1440)
        self.assertEqual(updated.pause_grace_ms, 1080)
        self.assertEqual(updated.pause_resume_count, 2)

    def test_clean_ends_slowly_trim_pause_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "adaptive-timing.json"
            store = AdaptiveTimingStore(
                path,
                config=TwinrConfig(
                    adaptive_timing_store_path=str(path),
                    speech_pause_ms=1200,
                    adaptive_timing_pause_grace_ms=900,
                ),
            )
            store.record_capture(
                initial_source="button",
                follow_up=False,
                speech_started_after_ms=2400,
                resumed_after_pause_count=2,
            )
            store.record_capture(
                initial_source="button",
                follow_up=False,
                speech_started_after_ms=2200,
                resumed_after_pause_count=0,
            )
            store.record_capture(
                initial_source="button",
                follow_up=False,
                speech_started_after_ms=2100,
                resumed_after_pause_count=0,
            )
            updated = store.record_capture(
                initial_source="button",
                follow_up=False,
                speech_started_after_ms=2000,
                resumed_after_pause_count=0,
            )

        self.assertEqual(updated.speech_pause_ms, 1400)
        self.assertEqual(updated.pause_grace_ms, 1060)
        self.assertEqual(updated.clean_pause_streak, 0)


if __name__ == "__main__":
    unittest.main()
