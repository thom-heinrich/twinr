from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
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

    def test_wakeword_label_capture_dispatches_to_proactive_helper(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("", encoding="utf-8")
            capture_path = root / "capture.wav"
            capture_path.write_bytes(b"RIFFtest")
            calls: list[tuple[Path, str, str | None]] = []
            fake_proactive_module = ModuleType("twinr.proactive")

            def _append_label(config, *, capture_path, label, notes=None):
                del config
                calls.append((Path(capture_path), label, notes))
                return {"data": {"label": label}}

            fake_proactive_module.append_wakeword_capture_label = _append_label
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(sys.modules, {"twinr.proactive": fake_proactive_module}):
                    main_mod = importlib.import_module("twinr.__main__")
                    sys.argv = [
                        "twinr",
                        "--env-file",
                        str(env_path),
                        "--wakeword-label-capture",
                        str(capture_path),
                        "--wakeword-label",
                        "correct",
                        "--wakeword-label-notes",
                        "operator confirmed",
                    ]
                    exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls, [(capture_path, "correct", "operator confirmed")])

    def test_wakeword_eval_dispatches_to_proactive_helper(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("TWINR_WAKEWORD_VERIFIER_MODE=disabled\n", encoding="utf-8")
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text("", encoding="utf-8")
            calls: list[tuple[Path, object | None]] = []
            fake_proactive_module = ModuleType("twinr.proactive")

            def _run_eval(*, config, manifest_path=None, backend=None):
                del config
                calls.append((Path(manifest_path), backend))
                return SimpleNamespace(
                    evaluated_entries=2,
                    metrics=SimpleNamespace(
                        precision=1.0,
                        recall=1.0,
                        false_positive_rate=0.0,
                        false_negative_rate=0.0,
                    ),
                    report_path=root / "report.json",
                )

            fake_proactive_module.run_wakeword_eval = _run_eval
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(sys.modules, {"twinr.proactive": fake_proactive_module}):
                    main_mod = importlib.import_module("twinr.__main__")
                    sys.argv = [
                        "twinr",
                        "--env-file",
                        str(env_path),
                        "--wakeword-eval",
                        "--wakeword-manifest",
                        str(manifest_path),
                    ]
                    exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls, [(manifest_path, None)])

    def test_wakeword_autotune_dispatches_to_proactive_helper(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("TWINR_WAKEWORD_VERIFIER_MODE=disabled\n", encoding="utf-8")
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text("", encoding="utf-8")
            calls: list[tuple[Path, object | None]] = []
            fake_proactive_module = ModuleType("twinr.proactive")

            def _autotune(*, config, manifest_path=None, backend=None):
                del config
                calls.append((Path(manifest_path), backend))
                return SimpleNamespace(
                    metrics=SimpleNamespace(
                        precision=0.9,
                        recall=0.8,
                        false_positive_rate=0.1,
                    ),
                    score=0.84,
                    profile_path=root / "wakeword_profile.json",
                    profile=SimpleNamespace(
                        threshold=0.08,
                        patience_frames=2,
                        activation_samples=3,
                    ),
                )

            fake_proactive_module.autotune_wakeword_profile = _autotune
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(sys.modules, {"twinr.proactive": fake_proactive_module}):
                    main_mod = importlib.import_module("twinr.__main__")
                    sys.argv = [
                        "twinr",
                        "--env-file",
                        str(env_path),
                        "--wakeword-autotune",
                        "--wakeword-manifest",
                        str(manifest_path),
                    ]
                    exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls, [(manifest_path, None)])


if __name__ == "__main__":
    unittest.main()
