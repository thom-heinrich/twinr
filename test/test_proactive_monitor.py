"""Shared proactive-monitor test doubles and bounded smoke tests.

This module restores the lightweight fakes used by proactive coordinator tests.
The scripted observers replay their last value after the input sequence is
exhausted so follow-up ticks stay deterministic.
"""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.pir import PirMotionEvent
from twinr.proactive.social.engine import SocialAudioObservation, SocialBodyPose, SocialVisionObservation
from twinr.proactive.social.observers import ProactiveAudioSnapshot, ProactiveVisionSnapshot


class MutableClock:
    """Mutable callable clock used by proactive integration tests."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = float(now)

    def __call__(self) -> float:
        return float(self.now)


class FakeVisionObserver:
    """Replay scripted vision observations as proactive snapshots."""

    def __init__(self, observations: list[SocialVisionObservation]) -> None:
        self._observations = list(observations)
        self._last_observation = (
            self._observations[-1]
            if self._observations
            else SocialVisionObservation(person_visible=False)
        )

    def observe(self) -> ProactiveVisionSnapshot:
        if self._observations:
            self._last_observation = self._observations.pop(0)
        return ProactiveVisionSnapshot(
            observation=self._last_observation,
            response_text="fake_vision_observation",
        )


class FakeAudioObserver:
    """Replay scripted audio observations as proactive snapshots."""

    def __init__(self, observation: SocialAudioObservation | list[SocialAudioObservation]) -> None:
        if isinstance(observation, list):
            scripted = list(observation)
        else:
            scripted = [observation]
        self._observations = scripted
        self._last_observation = scripted[-1] if scripted else SocialAudioObservation(speech_detected=False)

    def observe(self) -> ProactiveAudioSnapshot:
        if self._observations:
            self._last_observation = self._observations.pop(0)
        return ProactiveAudioSnapshot(observation=self._last_observation)


class FakePirMonitor:
    """Replay scripted PIR events while exposing the current motion level."""

    def __init__(self, *, events: list[bool] | None = None, level: bool = False) -> None:
        self.events = list(events or [])
        self.level = bool(level)
        self._timestamp_ns = 0

    def open(self) -> "FakePirMonitor":
        return self

    def close(self) -> None:
        return None

    def poll(self, timeout: float = 0.0) -> PirMotionEvent | None:
        _ = timeout
        if not self.events:
            return None
        self.level = bool(self.events.pop(0))
        self._timestamp_ns += 1
        return PirMotionEvent(
            name="pir",
            line_offset=0,
            motion_detected=self.level,
            raw_edge="rising" if self.level else "falling",
            timestamp_ns=self._timestamp_ns,
        )

    def motion_detected(self) -> bool:
        return self.level


class ProactiveMonitorTestDoubleTests(unittest.TestCase):
    def test_fake_vision_observer_replays_last_observation_after_script_exhaustion(self) -> None:
        observer = FakeVisionObserver(
            [
                SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.UPRIGHT),
            ]
        )

        first = observer.observe()
        second = observer.observe()

        self.assertTrue(first.observation.person_visible)
        self.assertTrue(second.observation.person_visible)
        self.assertEqual(second.observation.body_pose, SocialBodyPose.UPRIGHT)

    def test_fake_pir_monitor_reports_polled_and_level_motion(self) -> None:
        monitor = FakePirMonitor(events=[True, False], level=False)

        first = monitor.poll()
        self.assertIsNotNone(first)
        assert first is not None
        self.assertTrue(first.motion_detected)
        self.assertTrue(monitor.motion_detected())

        second = monitor.poll()
        self.assertIsNotNone(second)
        assert second is not None
        self.assertFalse(second.motion_detected)
        self.assertFalse(monitor.motion_detected())


if __name__ == "__main__":
    unittest.main()
