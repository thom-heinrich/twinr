from pathlib import Path
import signal
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.ops.runtime_supervisor_process import PidProcessHandle, default_process_factory


class RuntimeSupervisorProcessTests(unittest.TestCase):
    def test_default_process_factory_starts_child_in_new_session(self) -> None:
        with mock.patch("twinr.ops.runtime_supervisor_process.subprocess.Popen") as popen_mock:
            process_mock = mock.Mock()
            process_mock.pid = 4321
            popen_mock.return_value = process_mock

            default_process_factory(
                ("python", "-m", "twinr", "--run-streaming-loop"),
                cwd=Path("/tmp/twinr"),
                env={"PYTHONPATH": "src"},
            )

        popen_mock.assert_called_once()
        self.assertTrue(popen_mock.call_args.kwargs["start_new_session"])
        self.assertTrue(popen_mock.call_args.kwargs["close_fds"])

    def test_pid_process_handle_terminate_signals_group_for_dedicated_child_session(self) -> None:
        pid_signals: list[tuple[int, int]] = []
        group_signals: list[tuple[int, int]] = []
        handle = PidProcessHandle(
            4321,
            pid_alive=lambda _pid: True,
            pid_signal=lambda pid, sig: pid_signals.append((pid, sig)),
            pid_getpgid=lambda pid: pid,
            group_signal=lambda pgid, sig: group_signals.append((pgid, sig)),
            monotonic=lambda: 0.0,
            sleep=lambda _seconds: None,
        )

        handle.terminate()
        handle.kill()

        self.assertEqual(pid_signals, [])
        self.assertEqual(
            group_signals,
            [
                (4321, signal.SIGTERM),
                (4321, signal.SIGKILL),
            ],
        )

    def test_pid_process_handle_falls_back_to_parent_pid_for_non_leader_process(self) -> None:
        pid_signals: list[tuple[int, int]] = []
        group_signals: list[tuple[int, int]] = []
        handle = PidProcessHandle(
            4321,
            pid_alive=lambda _pid: True,
            pid_signal=lambda pid, sig: pid_signals.append((pid, sig)),
            pid_getpgid=lambda _pid: 77,
            group_signal=lambda pgid, sig: group_signals.append((pgid, sig)),
            monotonic=lambda: 0.0,
            sleep=lambda _seconds: None,
        )

        handle.terminate()

        self.assertEqual(pid_signals, [(4321, signal.SIGTERM)])
        self.assertEqual(group_signals, [])


if __name__ == "__main__":
    unittest.main()
