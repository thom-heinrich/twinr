from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.workflows.required_remote_watch import RequiredRemoteDependencyWatch


class RequiredRemoteDependencyWatchTests(unittest.TestCase):
    def test_exception_backoff_caps_before_overflow(self) -> None:
        watch = RequiredRemoteDependencyWatch(
            interval_s=0.25,
            refresh=lambda force: True,
            max_exception_backoff_s=30.0,
            exception_backoff_multiplier=2.0,
            exception_backoff_jitter_ratio=0.0,
        )

        with watch._condition:
            watch._consecutive_failures = 10_000

        delay_ns = watch._compute_next_wait_ns(ready=False, failed_by_exception=True)

        self.assertEqual(delay_ns, watch._max_exception_backoff_ns)

    def test_false_ready_backoff_caps_before_overflow(self) -> None:
        watch = RequiredRemoteDependencyWatch(
            interval_s=0.25,
            refresh=lambda force: False,
            max_exception_backoff_s=30.0,
            exception_backoff_multiplier=2.0,
            exception_backoff_jitter_ratio=0.0,
            backoff_on_false_ready=True,
        )

        with watch._condition:
            watch._consecutive_failures = 10_000

        delay_ns = watch._compute_next_wait_ns(ready=False, failed_by_exception=False)

        self.assertEqual(delay_ns, watch._max_exception_backoff_ns)
