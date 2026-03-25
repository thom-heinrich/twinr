from datetime import datetime
from pathlib import Path
import sys
import tempfile
import unittest
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.proactive import ProactiveGovernor, ProactiveGovernorCandidate


class ProactiveGovernorTests(unittest.TestCase):
    def _config(self, temp_dir: str, **overrides) -> TwinrConfig:
        defaults = {
            "project_root": temp_dir,
            "proactive_governor_enabled": True,
            "proactive_governor_active_reservation_ttl_s": 30.0,
            "proactive_governor_global_prompt_cooldown_s": 120.0,
            "proactive_governor_window_s": 600.0,
            "proactive_governor_window_prompt_limit": 4,
            "proactive_governor_presence_session_prompt_limit": 2,
            "proactive_governor_source_repeat_cooldown_s": 300.0,
        }
        defaults.update(overrides)
        return TwinrConfig(**defaults)

    def test_global_cooldown_blocks_second_non_safety_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            governor = ProactiveGovernor.from_config(self._config(temp_dir))
            first = governor.try_reserve(
                ProactiveGovernorCandidate(
                    source_kind="social",
                    source_id="attention_window",
                    summary="Gentle greeting.",
                ),
                now=datetime(2026, 3, 14, 10, 0, tzinfo=ZoneInfo("Europe/Berlin")),
            )
            self.assertTrue(first.allowed)
            governor.mark_delivered(first.reservation, now=datetime(2026, 3, 14, 10, 0, tzinfo=ZoneInfo("Europe/Berlin")))

            second = governor.try_reserve(
                ProactiveGovernorCandidate(
                    source_kind="reminder",
                    source_id="REM-1",
                    summary="Take medication.",
                    counts_toward_presence_budget=False,
                ),
                now=datetime(2026, 3, 14, 10, 1, tzinfo=ZoneInfo("Europe/Berlin")),
            )

        self.assertFalse(second.allowed)
        self.assertEqual(second.reason, "governor_global_prompt_cooldown_active")

    def test_safety_prompt_bypasses_global_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            governor = ProactiveGovernor.from_config(self._config(temp_dir))
            first = governor.try_reserve(
                ProactiveGovernorCandidate(
                    source_kind="reminder",
                    source_id="REM-1",
                    summary="Take medication.",
                    counts_toward_presence_budget=False,
                ),
                now=datetime(2026, 3, 14, 10, 0, tzinfo=ZoneInfo("Europe/Berlin")),
            )
            self.assertTrue(first.allowed)
            governor.mark_delivered(first.reservation, now=datetime(2026, 3, 14, 10, 0, tzinfo=ZoneInfo("Europe/Berlin")))

            second = governor.try_reserve(
                ProactiveGovernorCandidate(
                    source_kind="social",
                    source_id="possible_fall",
                    summary="Do you need help?",
                    safety_exempt=True,
                    counts_toward_presence_budget=False,
                ),
                now=datetime(2026, 3, 14, 10, 1, tzinfo=ZoneInfo("Europe/Berlin")),
            )

        self.assertTrue(second.allowed)

    def test_presence_session_budget_ignores_reminders_but_blocks_second_session_nudge(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            governor = ProactiveGovernor.from_config(
                self._config(
                    temp_dir,
                    proactive_governor_global_prompt_cooldown_s=0.0,
                    proactive_governor_source_repeat_cooldown_s=0.0,
                    proactive_governor_presence_session_prompt_limit=1,
                )
            )
            social = governor.try_reserve(
                ProactiveGovernorCandidate(
                    source_kind="social",
                    source_id="attention_window",
                    summary="Gentle greeting.",
                    presence_session_id=7,
                ),
                now=datetime(2026, 3, 14, 10, 0, tzinfo=ZoneInfo("Europe/Berlin")),
            )
            self.assertTrue(social.allowed)
            governor.mark_delivered(social.reservation, now=datetime(2026, 3, 14, 10, 0, tzinfo=ZoneInfo("Europe/Berlin")))

            reminder = governor.try_reserve(
                ProactiveGovernorCandidate(
                    source_kind="reminder",
                    source_id="REM-1",
                    summary="Take medication.",
                    presence_session_id=7,
                    counts_toward_presence_budget=False,
                ),
                now=datetime(2026, 3, 14, 10, 1, tzinfo=ZoneInfo("Europe/Berlin")),
            )
            self.assertTrue(reminder.allowed)
            governor.mark_delivered(reminder.reservation, now=datetime(2026, 3, 14, 10, 1, tzinfo=ZoneInfo("Europe/Berlin")))

            longterm = governor.try_reserve(
                ProactiveGovernorCandidate(
                    source_kind="longterm",
                    source_id="candidate:walk_weather",
                    summary="It looks like a good time for a walk.",
                    presence_session_id=7,
                ),
                now=datetime(2026, 3, 14, 10, 2, tzinfo=ZoneInfo("Europe/Berlin")),
            )

        self.assertFalse(longterm.allowed)
        self.assertEqual(longterm.reason, "governor_presence_session_budget_exhausted")

    def test_active_reservation_blocks_concurrent_prompt_start(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            governor = ProactiveGovernor.from_config(
                self._config(
                    temp_dir,
                    proactive_governor_global_prompt_cooldown_s=0.0,
                    proactive_governor_source_repeat_cooldown_s=0.0,
                )
            )
            first = governor.try_reserve(
                ProactiveGovernorCandidate(
                    source_kind="social",
                    source_id="attention_window",
                    summary="Gentle greeting.",
                ),
                now=datetime(2026, 3, 14, 10, 0, tzinfo=ZoneInfo("Europe/Berlin")),
            )
            self.assertTrue(first.allowed)

            second = governor.try_reserve(
                ProactiveGovernorCandidate(
                    source_kind="longterm",
                    source_id="candidate:walk_weather",
                    summary="It looks like a good time for a walk.",
                ),
                now=datetime(2026, 3, 14, 10, 0, 5, tzinfo=ZoneInfo("Europe/Berlin")),
            )

        self.assertFalse(second.allowed)
        self.assertEqual(second.reason, "prompt_inflight")

    def test_display_channel_bypasses_speech_cooldowns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            governor = ProactiveGovernor.from_config(self._config(temp_dir))
            first = governor.try_reserve(
                ProactiveGovernorCandidate(
                    source_kind="social",
                    source_id="attention_window",
                    summary="Gentle greeting.",
                ),
                now=datetime(2026, 3, 14, 10, 0, tzinfo=ZoneInfo("Europe/Berlin")),
            )
            self.assertTrue(first.allowed)
            governor.mark_delivered(first.reservation, now=datetime(2026, 3, 14, 10, 0, tzinfo=ZoneInfo("Europe/Berlin")))

            second = governor.try_reserve(
                ProactiveGovernorCandidate(
                    source_kind="social",
                    source_id="attention_window",
                    summary="Gentle greeting.",
                    channel="display",
                ),
                now=datetime(2026, 3, 14, 10, 1, tzinfo=ZoneInfo("Europe/Berlin")),
            )

        self.assertTrue(second.allowed)


if __name__ == "__main__":
    unittest.main()
