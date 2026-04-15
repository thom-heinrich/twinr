from __future__ import annotations

from contextlib import contextmanager
import os
from threading import Event, Thread
import time
import unittest
from unittest import mock

from twinr.agent.base_agent import TwinrConfig
from twinr.display.companion import optional_display_companion
from twinr.display import companion_signals as companion_signals_mod
from twinr.display.companion_signals import (
    register_display_process_wakeup_listener,
    request_display_companion_wakeup,
)


class _FakeDisplayLoop:
    def __init__(self) -> None:
        self.run_calls = 0
        self.started = Event()
        self.stopped = Event()
        self.sleep = lambda _seconds: None
        self.stop_requested = lambda: False
        self.sleep_calls = 0
        self.sleep_started = Event()

    def run(self, *, duration_s: float | None = None) -> int:
        self.run_calls += 1
        self.started.set()
        while not self.stop_requested():
            self.sleep_calls += 1
            self.sleep_started.set()
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

    def test_optional_display_companion_wakeup_interrupts_idle_sleep(self) -> None:
        loop = _FakeDisplayLoop()

        with optional_display_companion(
            TwinrConfig(display_poll_interval_s=0.0),
            enabled=True,
            emit=lambda _line: None,
            loop_factory=lambda _config: loop,
            lock_owner=lambda _config, _loop_name: None,
            lock_factory=_fake_lock,
        ):
            self.assertTrue(loop.started.wait(timeout=1.0))
            self.assertTrue(loop.sleep_started.wait(timeout=1.0))
            self.assertEqual(loop.sleep_calls, 1)
            request_display_companion_wakeup()
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and loop.sleep_calls < 2:
                time.sleep(0.01)
            self.assertGreaterEqual(loop.sleep_calls, 2)

    def test_display_process_wakeup_listener_sets_event_on_signal(self) -> None:
        with register_display_process_wakeup_listener() as wake_event:
            self.assertFalse(wake_event.is_set())

            def _send_signal() -> None:
                time.sleep(0.05)
                os.kill(os.getpid(), companion_signals_mod._DISPLAY_COMPANION_WAKE_SIGNAL)

            sender = Thread(target=_send_signal, daemon=True)
            sender.start()
            self.assertTrue(wake_event.wait(timeout=1.0))
            sender.join(timeout=1.0)

    def test_request_display_companion_wakeup_signals_external_display_loop_owner(self) -> None:
        config = TwinrConfig(project_root="/tmp/twinr-display-signal")

        with (
            mock.patch.object(companion_signals_mod, "_display_loop_owner_pid", return_value=4321),
            mock.patch.object(companion_signals_mod, "_signal_display_process", return_value=True) as signal_process,
            mock.patch("os.getpid", return_value=1111),
        ):
            self.assertTrue(request_display_companion_wakeup(config))

        signal_process.assert_called_once_with(4321)
