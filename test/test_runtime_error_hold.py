from types import SimpleNamespace
import unittest

from twinr.agent.workflows.runtime_error_hold import hold_runtime_error_state


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, float(seconds))


class _FakeRuntime:
    def __init__(self) -> None:
        self.status = SimpleNamespace(value="waiting")
        self.fail_calls: list[str] = []
        self.persist_calls: list[dict[str, object]] = []

    def fail(self, message: str):
        self.fail_calls.append(message)
        self.status.value = "error"
        return self.status

    def _persist_snapshot(self, *, error_message: str | None = None) -> None:
        self.persist_calls.append(
            {
                "status": self.status.value,
                "error_message": error_message,
            }
        )


class RuntimeErrorHoldTests(unittest.TestCase):
    def test_hold_runtime_error_state_enters_error_and_refreshes_snapshot(self) -> None:
        runtime = _FakeRuntime()
        clock = _FakeClock()
        emitted: list[str] = []

        exit_code = hold_runtime_error_state(
            runtime=runtime,
            error=RuntimeError("capture unreadable"),
            emit=emitted.append,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
            duration_s=1.1,
            refresh_interval_s=0.5,
        )

        self.assertEqual(exit_code, 1)
        self.assertEqual(runtime.fail_calls, ["capture unreadable"])
        self.assertGreaterEqual(len(runtime.persist_calls), 3)
        self.assertTrue(all(call["status"] == "error" for call in runtime.persist_calls))
        self.assertTrue(all(call["error_message"] == "capture unreadable" for call in runtime.persist_calls))
        self.assertIn("status=error", emitted)
        self.assertIn("error=capture unreadable", emitted)
