from pathlib import Path
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.ops.locks import (
    TwinrInstanceAlreadyRunningError,
    TwinrInstanceLock,
    _open_lock_fd,
    loop_instance_lock,
    loop_lock_owner,
    loop_lock_path,
)


class LoopLockTests(unittest.TestCase):
    def test_lock_path_uses_runtime_state_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )

            path = loop_lock_path(config, "realtime-loop")

        self.assertEqual(path, Path(temp_dir) / "twinr-realtime-loop.lock")

    def test_second_acquire_fails_with_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "twinr-realtime-loop.lock"
            with TwinrInstanceLock(path, "realtime loop"):
                with self.assertRaisesRegex(
                    TwinrInstanceAlreadyRunningError,
                    r"Another Twinr realtime loop is already running",
                ):
                    with TwinrInstanceLock(path, "realtime loop"):
                        pass

    def test_lock_can_be_reacquired_after_release(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            lock = loop_instance_lock(config, "display-loop")

            with lock:
                self.assertTrue(loop_lock_path(config, "display-loop").exists())

            with loop_instance_lock(config, "display-loop"):
                self.assertTrue(loop_lock_path(config, "display-loop").exists())

    def test_loop_lock_owner_reports_active_pid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )

            self.assertIsNone(loop_lock_owner(config, "realtime-loop"))

            with loop_instance_lock(config, "realtime-loop"):
                owner = loop_lock_owner(config, "realtime-loop")

            self.assertIsInstance(owner, int)
            self.assertGreater(owner or 0, 0)
            self.assertIsNone(loop_lock_owner(config, "realtime-loop"))

    def test_open_lock_fd_uses_zero_mode_for_existing_lock_openat2(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "twinr-realtime-loop.lock"
            path.write_text("123\n", encoding="utf-8")
            recorded: list[tuple[int, int]] = []

            def fake_openat2(target: Path, *, flags: int, mode: int = 0, resolve: int = 0) -> int:
                recorded.append((flags, mode))
                return os.open(str(target), flags, 0o600)

            with mock.patch("twinr.ops.locks._openat2_fd", side_effect=fake_openat2):
                fd = _open_lock_fd(path, create=False)

            try:
                self.assertEqual(len(recorded), 1)
                self.assertEqual(recorded[0][1], 0)
            finally:
                os.close(fd)

    def test_open_lock_fd_keeps_create_mode_for_new_lock_openat2(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "twinr-realtime-loop.lock"
            recorded: list[tuple[int, int]] = []

            def fake_openat2(target: Path, *, flags: int, mode: int = 0, resolve: int = 0) -> int:
                recorded.append((flags, mode))
                return os.open(str(target), flags, 0o600)

            with mock.patch("twinr.ops.locks._openat2_fd", side_effect=fake_openat2):
                fd = _open_lock_fd(path, create=True)

            try:
                self.assertEqual(len(recorded), 1)
                self.assertEqual(recorded[0][1], 0o600)
            finally:
                os.close(fd)

    def test_forked_child_does_not_keep_lock_alive_after_parent_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_state = root / "runtime-state.json"
            child_script = textwrap.dedent(
                f"""
                import os
                import sys
                import time
                from pathlib import Path

                sys.path.insert(0, {str((Path(__file__).resolve().parents[1] / "src"))!r})

                from twinr.agent.base_agent.config import TwinrConfig
                from twinr.ops.locks import loop_instance_lock

                config = TwinrConfig(
                    project_root={str(root)!r},
                    runtime_state_path={str(runtime_state)!r},
                )

                with loop_instance_lock(config, "realtime-loop"):
                    pid = os.fork()
                    if pid == 0:
                        time.sleep(2.0)
                        os._exit(0)
                    os._exit(0)
                """
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

            proc = subprocess.Popen(
                [sys.executable, "-c", child_script],
                cwd=str(root),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            proc.wait(timeout=1.0)
            self.assertEqual(proc.returncode, 0)

            time.sleep(0.2)
            config = TwinrConfig(
                project_root=str(root),
                runtime_state_path=str(runtime_state),
            )

            with loop_instance_lock(config, "realtime-loop"):
                self.assertTrue(loop_lock_path(config, "realtime-loop").exists())


if __name__ == "__main__":
    unittest.main()
