from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.respeaker.models import ReSpeakerSignalSnapshot
from twinr.hardware.respeaker.scheduled_provider import ScheduledReSpeakerSignalProvider


class _Clock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class _FakeInnerProvider:
    def __init__(self, snapshots: list[ReSpeakerSignalSnapshot]) -> None:
        self.snapshots = list(snapshots)
        self.calls = 0
        self.closed = False

    def observe(self) -> ReSpeakerSignalSnapshot:
        self.calls += 1
        if len(self.snapshots) > 1:
            return self.snapshots.pop(0)
        return self.snapshots[0]

    def close(self) -> None:
        self.closed = True


def _snapshot(
    *,
    captured_at: float,
    device_runtime_mode: str = "audio_ready",
    host_control_ready: bool = True,
    speech_detected: bool | None = None,
    recent_speech_age_s: float | None = None,
) -> ReSpeakerSignalSnapshot:
    return ReSpeakerSignalSnapshot(
        captured_at=captured_at,
        source="respeaker_xvf3800",
        source_type="observed",
        sensor_window_ms=1000,
        device_runtime_mode=device_runtime_mode,
        host_control_ready=host_control_ready,
        speech_detected=speech_detected,
        recent_speech_age_s=recent_speech_age_s,
    )


class ScheduledReSpeakerSignalProviderTests(unittest.TestCase):
    def test_idle_context_reuses_cached_snapshot_until_idle_interval_expires(self) -> None:
        clock = _Clock()
        inner = _FakeInnerProvider([
            _snapshot(captured_at=1.0),
            _snapshot(captured_at=7.0),
        ])
        provider = ScheduledReSpeakerSignalProvider(
            provider=inner,
            active_refresh_interval_s=1.0,
            degraded_refresh_interval_s=0.5,
            idle_refresh_interval_s=6.0,
            clock=clock,
        )
        provider.note_runtime_context(
            observed_at=0.0,
            motion_active=False,
            inspect_requested=False,
            presence_session_armed=False,
            assistant_output_active=False,
        )

        first = provider.observe()
        clock.now = 2.0
        second = provider.observe()
        clock.now = 6.1
        third = provider.observe()

        self.assertEqual(inner.calls, 2)
        self.assertEqual(first.captured_at, 1.0)
        self.assertEqual(second.captured_at, 1.0)
        self.assertEqual(third.captured_at, 7.0)

    def test_active_context_refreshes_on_active_interval(self) -> None:
        clock = _Clock()
        inner = _FakeInnerProvider([
            _snapshot(captured_at=1.0),
            _snapshot(captured_at=2.0),
        ])
        provider = ScheduledReSpeakerSignalProvider(
            provider=inner,
            active_refresh_interval_s=1.0,
            degraded_refresh_interval_s=0.5,
            idle_refresh_interval_s=6.0,
            clock=clock,
        )
        provider.note_runtime_context(
            observed_at=0.0,
            motion_active=True,
            inspect_requested=False,
            presence_session_armed=False,
            assistant_output_active=False,
        )

        first = provider.observe()
        clock.now = 1.2
        second = provider.observe()

        self.assertEqual(inner.calls, 2)
        self.assertEqual(first.captured_at, 1.0)
        self.assertEqual(second.captured_at, 2.0)

    def test_degraded_snapshot_refreshes_on_degraded_interval(self) -> None:
        clock = _Clock()
        inner = _FakeInnerProvider([
            _snapshot(captured_at=1.0, device_runtime_mode="not_detected", host_control_ready=False),
            _snapshot(captured_at=2.0, device_runtime_mode="not_detected", host_control_ready=False),
        ])
        provider = ScheduledReSpeakerSignalProvider(
            provider=inner,
            active_refresh_interval_s=1.0,
            degraded_refresh_interval_s=0.5,
            idle_refresh_interval_s=6.0,
            clock=clock,
        )

        first = provider.observe()
        clock.now = 0.6
        second = provider.observe()

        self.assertEqual(inner.calls, 2)
        self.assertEqual(first.captured_at, 1.0)
        self.assertEqual(second.captured_at, 2.0)


if __name__ == "__main__":
    unittest.main()
