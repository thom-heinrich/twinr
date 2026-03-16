from __future__ import annotations

from contextlib import contextmanager
from threading import Event
import unittest

from twinr.agent.base_agent import TwinrConfig
from twinr.display.companion import optional_display_companion


class _FakeDisplayLoop:
    def __init__(self) -> None:
        self.run_calls = 0
        self.started = Event()
        self.stopped = Event()
        self.sleep = lambda _seconds: None

    def run(self, *, duration_s: float | None = None) -> int:
        self.run_calls += 1
        self.started.set()
        self.sleep(60.0)
        self.stopped.set()
        return 0


@contextmanager
def _fake_lock(_config, _loop_name: str):
    yield


class DisplayCompanionTests(unittest.TestCase):
    def test_optional_display_companion_starts_and_stops_loop(self) -> None:
        loop = _FakeDisplayLoop()
        emitted: list[str] = []

        with optional_display_companion(
            TwinrConfig(display_poll_interval_s=0.0),
            enabled=True,
            emit=emitted.append,
            loop_factory=lambda _config: loop,
            lock_owner=lambda _config, _loop_name: None,
            lock_factory=_fake_lock,
        ):
            self.assertTrue(loop.started.wait(timeout=1.0))
            self.assertEqual(loop.run_calls, 1)

        self.assertTrue(loop.stopped.wait(timeout=1.0))
        self.assertIn("display_companion=started", emitted)

    def test_optional_display_companion_skips_when_display_loop_is_already_running(self) -> None:
        created = False

        def _loop_factory(_config):
            nonlocal created
            created = True
            return _FakeDisplayLoop()

        with optional_display_companion(
            TwinrConfig(),
            enabled=True,
            emit=lambda _line: None,
            loop_factory=_loop_factory,
            lock_owner=lambda _config, _loop_name: 123,
            lock_factory=_fake_lock,
        ):
            pass

        self.assertFalse(created)
