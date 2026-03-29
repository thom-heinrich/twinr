"""Shared proactive-monitor test doubles and bounded smoke tests.

This module restores the lightweight fakes used by proactive coordinator tests.
The scripted observers replay their last value after the input sequence is
exhausted so follow-up ticks stay deterministic.
"""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.pir import PirMotionEvent
from twinr.proactive.social.engine import SocialAudioObservation, SocialBodyPose, SocialVisionObservation
from twinr.proactive.social.observers import ProactiveAudioSnapshot, ProactiveVisionSnapshot
from twinr.proactive.runtime.service_impl.monitor import ProactiveMonitorService


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

    def test_shared_perception_cycle_keeps_due_lanes_aligned_under_overrun(self) -> None:
        clock = MutableClock()

        class _StopEvent:
            def __init__(self) -> None:
                self._set = False

            def is_set(self) -> bool:
                return self._set

            def wait(self, timeout: float) -> bool:
                clock.now += float(timeout)
                return self._set

            def set(self) -> None:
                self._set = True

        stop_event = _StopEvent()

        class _Coordinator:
            def __init__(self) -> None:
                self.config = SimpleNamespace()
                self.runtime = SimpleNamespace(
                    ops_events=SimpleNamespace(append=lambda **_kwargs: None),
                )
                self.audio_observer = None
                self.attention_servo_controller = None
                self.pir_monitor = None
                self._display_perception_cycle = None
                self.open_calls: list[tuple[bool, bool]] = []
                self.attention_calls = 0
                self.gesture_calls = 0
                self.tick_calls = 0

            def _open_display_perception_cycle(self, *, attention_due: bool, gesture_due: bool) -> None:
                self.open_calls.append((attention_due, gesture_due))
                self._display_perception_cycle = object() if attention_due and gesture_due else None

            def _close_display_perception_cycle(self) -> None:
                self._display_perception_cycle = None

            def refresh_display_attention(self) -> bool:
                self.attention_calls += 1
                clock.now += 1.0
                return True

            def refresh_display_gesture_emoji(self) -> bool:
                self.gesture_calls += 1
                clock.now += 0.6
                if self.gesture_calls >= 2:
                    stop_event.set()
                return True

            def tick(self) -> None:
                self.tick_calls += 1

        service = ProactiveMonitorService(
            coordinator=_Coordinator(),
            poll_interval_s=0.2,
        )
        service._stop_event = stop_event  # type: ignore[assignment]

        with (
            patch("twinr.proactive.runtime.service_impl.monitor.time.monotonic", side_effect=clock),
            patch(
                "twinr.proactive.runtime.service_impl.monitor.resolve_display_attention_refresh_interval",
                return_value=0.6,
            ),
            patch(
                "twinr.proactive.runtime.service_impl.monitor.resolve_display_gesture_refresh_interval",
                return_value=0.2,
            ),
        ):
            service._run()  # pylint: disable=protected-access

        coordinator = service.coordinator
        self.assertGreaterEqual(coordinator.attention_calls, 2)
        self.assertGreaterEqual(coordinator.gesture_calls, 2)
        self.assertGreaterEqual(len(coordinator.open_calls), 2)
        self.assertEqual(coordinator.open_calls[:2], [(True, True), (True, True)])


if __name__ == "__main__":
    unittest.main()
