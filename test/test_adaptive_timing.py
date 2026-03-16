from pathlib import Path
import sys
import tempfile
import unittest
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.conversation.adaptive_timing import AdaptiveTimingStore
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

        self.assertEqual(updated.speech_pause_ms, 1260)
        self.assertEqual(updated.pause_grace_ms, 490)
        self.assertEqual(updated.pause_resume_count, 2)

    def test_clean_ends_slowly_trim_pause_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "adaptive-timing.json"
            store = AdaptiveTimingStore(
                path,
                config=TwinrConfig(
                    adaptive_timing_store_path=str(path),
                    speech_pause_ms=1200,
                    adaptive_timing_pause_grace_ms=450,
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
            updated = store.record_capture(
                initial_source="button",
                follow_up=False,
                speech_started_after_ms=2000,
                resumed_after_pause_count=0,
            )

        self.assertEqual(updated.speech_pause_ms, 1200)
        self.assertEqual(updated.pause_grace_ms, 450)
        self.assertEqual(updated.clean_pause_streak, 0)

    def test_defaults_stay_bounded_for_snappy_turn_end(self) -> None:
        config = TwinrConfig(speech_pause_ms=800)
        store = AdaptiveTimingStore(Path("/tmp/unused-adaptive-timing.json"), config=config)

        default_profile = store.default_profile()

        self.assertEqual(default_profile.speech_pause_ms, 800)
        self.assertEqual(default_profile.pause_grace_ms, 450)
        self.assertLessEqual(
            default_profile.speech_pause_ms + default_profile.pause_grace_ms,
            1300,
        )

    def test_loaded_outlier_profile_is_clamped_close_to_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "adaptive-timing.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "profile": {
                            "button_start_timeout_s": 20.0,
                            "follow_up_start_timeout_s": 11.0,
                            "speech_pause_ms": 2200,
                            "pause_grace_ms": 1400,
                        },
                    }
                ),
                encoding="utf-8",
            )
            store = AdaptiveTimingStore(
                path,
                config=TwinrConfig(
                    adaptive_timing_store_path=str(path),
                    speech_pause_ms=800,
                    adaptive_timing_pause_grace_ms=450,
                ),
            )

            profile = store.current()

        self.assertEqual(profile.speech_pause_ms, 1200)
        self.assertEqual(profile.pause_grace_ms, 650)
        self.assertEqual(profile.button_start_timeout_s, 14.0)
        self.assertEqual(profile.follow_up_start_timeout_s, 8.0)


if __name__ == "__main__":
    unittest.main()
