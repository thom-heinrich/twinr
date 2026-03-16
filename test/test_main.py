from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import importlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class _FakeDisplayLoop:
    def __init__(self) -> None:
        self.duration_s = None

    def run(self, *, duration_s: float | None = None) -> int:
        self.duration_s = duration_s
        return 0


class _FakeHardwareLoop:
    def __init__(self) -> None:
        self.duration_s = None

    def run(self, *, duration_s: float | None = None) -> int:
        self.duration_s = duration_s
        return 0


@contextmanager
def _fake_lock(_config, _name: str):
    yield


class MainCliTests(unittest.TestCase):
    def test_uses_pi_runtime_root_detects_twinr_env(self) -> None:
        main_mod = importlib.import_module("twinr.__main__")
        try:
            self.assertTrue(main_mod._uses_pi_runtime_root("/twinr/.env"))
            self.assertFalse(main_mod._uses_pi_runtime_root("/tmp/not-twinr.env"))
        finally:
            sys.modules.pop("twinr.__main__", None)

    def test_should_enable_display_companion_requires_real_pi_host(self) -> None:
        main_mod = importlib.import_module("twinr.__main__")
        try:
            with patch.object(main_mod, "_uses_pi_runtime_root", return_value=True):
                with patch.object(main_mod, "_is_raspberry_pi_host", return_value=True):
                    self.assertTrue(main_mod._should_enable_display_companion("/twinr/.env"))
                with patch.object(main_mod, "_is_raspberry_pi_host", return_value=False):
                    self.assertFalse(main_mod._should_enable_display_companion("/twinr/.env"))
        finally:
            sys.modules.pop("twinr.__main__", None)

    def test_assert_pi_runtime_root_rejects_non_pi_cwd_for_pi_env(self) -> None:
        main_mod = importlib.import_module("twinr.__main__")
        twinr_package = importlib.import_module("twinr")

        with tempfile.TemporaryDirectory() as temp_dir:
            fake_cwd = Path(temp_dir)
            with patch("pathlib.Path.cwd", return_value=fake_cwd):
                with patch.object(twinr_package, "__file__", "/twinr/src/twinr/__init__.py"):
                    with self.assertRaisesRegex(RuntimeError, "must be launched from /twinr"):
                        main_mod._assert_pi_runtime_root("/twinr/.env", command_name="run-streaming-loop")

        sys.modules.pop("twinr.__main__", None)

    def test_assert_pi_runtime_root_accepts_twinr_cwd_and_source_root(self) -> None:
        main_mod = importlib.import_module("twinr.__main__")
        twinr_package = importlib.import_module("twinr")
        try:
            with patch("pathlib.Path.cwd", return_value=Path("/twinr")):
                with patch.object(twinr_package, "__file__", "/twinr/src/twinr/__init__.py"):
                    main_mod._assert_pi_runtime_root("/twinr/.env", command_name="run-streaming-loop")
        finally:
            sys.modules.pop("twinr.__main__", None)

    def test_run_display_loop_does_not_require_workflow_imports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                f"TWINR_RUNTIME_STATE_PATH={root / 'runtime-state.json'}\n",
                encoding="utf-8",
            )
            fake_loop = _FakeDisplayLoop()
            fake_workflows = ModuleType("twinr.agent.workflows")
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(sys.modules, {"twinr.agent.workflows": fake_workflows}):
                    main_mod = importlib.import_module("twinr.__main__")
                    with patch("twinr.display.TwinrStatusDisplayLoop.from_config", return_value=fake_loop):
                        with patch("twinr.ops.loop_instance_lock", _fake_lock):
                            sys.argv = [
                                "twinr",
                                "--env-file",
                                str(env_path),
                                "--run-display-loop",
                                "--loop-duration",
                                "0",
                            ]
                            exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        self.assertEqual(fake_loop.duration_s, 0.0)

    def test_run_hardware_loop_enables_display_companion_for_pi_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    (
                        f"TWINR_RUNTIME_STATE_PATH={root / 'runtime-state.json'}",
                        "TWINR_OPENAI_API_KEY=sk-test",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            fake_loop = _FakeHardwareLoop()
            companion_calls: list[bool] = []
            fake_runner_module = ModuleType("twinr.agent.workflows.runner")
            fake_runner_module.TwinrHardwareLoop = lambda **_kwargs: fake_loop
            fake_openai_module = ModuleType("twinr.providers.openai")

            class _FakeBackend:
                def __init__(self, config) -> None:
                    self.config = config

            fake_openai_module.OpenAIBackend = _FakeBackend
            fake_companion_module = ModuleType("twinr.display.companion")

            @contextmanager
            def _fake_companion(_config, *, enabled: bool):
                companion_calls.append(enabled)
                yield

            fake_companion_module.optional_display_companion = _fake_companion
            fake_ops_module = ModuleType("twinr.ops")
            fake_ops_module.loop_instance_lock = _fake_lock
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(
                    sys.modules,
                    {
                        "twinr.agent.workflows.runner": fake_runner_module,
                        "twinr.providers.openai": fake_openai_module,
                        "twinr.display.companion": fake_companion_module,
                        "twinr.ops": fake_ops_module,
                    },
                ):
                    main_mod = importlib.import_module("twinr.__main__")
                    with patch.object(main_mod, "_should_enable_display_companion", return_value=True):
                        with patch.object(main_mod, "_assert_pi_runtime_root", return_value=None):
                            sys.argv = [
                                "twinr",
                                "--env-file",
                                str(env_path),
                                "--run-hardware-loop",
                                "--loop-duration",
                                "0",
                            ]
                            exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        self.assertEqual(companion_calls, [True])
        self.assertEqual(fake_loop.duration_s, 0.0)


if __name__ == "__main__":
    unittest.main()
