from pathlib import Path
import errno
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.pir import PirMonitorError
from twinr.proactive.runtime.pir_open_gate import open_pir_monitor_with_busy_retry


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, float(seconds))


class _BusyThenReadyMonitor:
    def __init__(self, *, busy_failures: int) -> None:
        self.busy_failures = busy_failures
        self.open_attempts = 0

    def open(self) -> "_BusyThenReadyMonitor":
        self.open_attempts += 1
        if self.busy_failures > 0:
            self.busy_failures -= 1
            raise PirMonitorError("Failed to open PIR monitor on 'gpiochip0'") from OSError(
                errno.EBUSY,
                "Device or resource busy",
            )
        return self


class _BrokenMonitor:
    def open(self) -> "_BrokenMonitor":
        raise PirMonitorError("Failed to open PIR monitor on 'gpiochip0'") from OSError(
            errno.ENODEV,
            "No such device",
        )


class PirOpenGateTests(unittest.TestCase):
    def test_open_retries_exact_busy_until_success(self) -> None:
        clock = _FakeClock()
        monitor = _BusyThenReadyMonitor(busy_failures=2)

        result = open_pir_monitor_with_busy_retry(
            monitor,
            timeout_s=1.0,
            retry_interval_s=0.1,
            monotonic=clock,
            sleep=clock.sleep,
        )

        self.assertEqual(monitor.open_attempts, 3)
        self.assertEqual(result.attempt_count, 3)
        self.assertEqual(result.busy_retry_count, 2)

    def test_open_propagates_non_busy_errors_without_retry(self) -> None:
        clock = _FakeClock()
        monitor = _BrokenMonitor()

        with self.assertRaises(PirMonitorError):
            open_pir_monitor_with_busy_retry(
                monitor,
                timeout_s=1.0,
                retry_interval_s=0.1,
                monotonic=clock,
                sleep=clock.sleep,
            )

    def test_open_fails_closed_when_busy_window_expires(self) -> None:
        clock = _FakeClock()
        monitor = _BusyThenReadyMonitor(busy_failures=10)

        with self.assertRaises(PirMonitorError):
            open_pir_monitor_with_busy_retry(
                monitor,
                timeout_s=0.15,
                retry_interval_s=0.1,
                monotonic=clock,
                sleep=clock.sleep,
            )

        self.assertGreaterEqual(monitor.open_attempts, 2)


if __name__ == "__main__":
    unittest.main()
