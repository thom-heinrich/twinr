from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.ops.locks import TwinrInstanceLock, loop_instance_lock, loop_lock_path


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
                    RuntimeError,
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


if __name__ == "__main__":
    unittest.main()
