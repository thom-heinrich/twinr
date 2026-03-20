from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import importlib
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.ops.locks import TwinrInstanceAlreadyRunningError


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


class _FakeRemoteMemoryWatchdog:
    def __init__(self) -> None:
        self.duration_s = None
        self.artifact_path = Path("/tmp/remote-memory-watchdog.json")
        self.interval_s = 1.0
        self.history_limit = 3600

    def run(self, *, duration_s: float | None = None) -> int:
        self.duration_s = duration_s
        return 0


class _FakeWhatsAppLoop:
    def __init__(self, *, config, runtime, backend) -> None:
        self.config = config
        self.runtime = runtime
        self.backend = backend
        self.duration_s = None

    def run(self, *, duration_s: float | None = None) -> int:
        self.duration_s = duration_s
        return 0


class _FakeRuntimeSupervisor:
    def __init__(self) -> None:
        self.duration_s = None
        self.env_file = None

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

    def test_should_enable_display_companion_defaults_to_real_pi_runtime_hosts(self) -> None:
        main_mod = importlib.import_module("twinr.__main__")
        try:
            config = main_mod.TwinrConfig()
            with patch.object(main_mod, "_is_raspberry_pi_host", return_value=True):
                self.assertTrue(main_mod._should_enable_display_companion(config, "/twinr/.env"))
            with patch.object(main_mod, "_is_raspberry_pi_host", return_value=False):
                self.assertFalse(main_mod._should_enable_display_companion(config, "/twinr/.env"))
            self.assertFalse(main_mod._should_enable_display_companion(config, "/home/thh/twinr/.env"))
        finally:
            sys.modules.pop("twinr.__main__", None)

    def test_should_enable_display_companion_honors_explicit_override(self) -> None:
        main_mod = importlib.import_module("twinr.__main__")
        try:
            enabled = main_mod.TwinrConfig(display_companion_enabled=True)
            disabled = main_mod.TwinrConfig(display_companion_enabled=False)
            with patch.object(main_mod, "_is_raspberry_pi_host", return_value=False):
                self.assertTrue(main_mod._should_enable_display_companion(enabled, "/twinr/.env"))
            with patch.object(main_mod, "_is_raspberry_pi_host", return_value=True):
                self.assertFalse(main_mod._should_enable_display_companion(disabled, "/twinr/.env"))
        finally:
            sys.modules.pop("twinr.__main__", None)

    def test_assert_pi_runtime_root_rejects_non_pi_cwd_for_pi_env(self) -> None:
        main_mod = importlib.import_module("twinr.__main__")
        twinr_package = importlib.import_module("twinr")

        with tempfile.TemporaryDirectory() as temp_dir:
            fake_cwd = Path(temp_dir)
            with patch.object(main_mod, "_is_raspberry_pi_host", return_value=True):
                with patch("pathlib.Path.cwd", return_value=fake_cwd):
                    with patch.object(twinr_package, "__file__", "/twinr/src/twinr/__init__.py"):
                        with self.assertRaisesRegex(RuntimeError, "must be launched from /twinr"):
                            main_mod._assert_pi_runtime_root("/twinr/.env", command_name="run-streaming-loop")

        sys.modules.pop("twinr.__main__", None)

    def test_assert_pi_runtime_root_rejects_non_pi_host_for_pi_env(self) -> None:
        main_mod = importlib.import_module("twinr.__main__")
        twinr_package = importlib.import_module("twinr")
        try:
            with patch.object(main_mod, "_is_raspberry_pi_host", return_value=False):
                with patch("pathlib.Path.cwd", return_value=Path("/twinr")):
                    with patch.object(twinr_package, "__file__", "/twinr/src/twinr/__init__.py"):
                        with self.assertRaisesRegex(RuntimeError, "only allowed on a Raspberry Pi host"):
                            main_mod._assert_pi_runtime_root("/twinr/.env", command_name="run-streaming-loop")
        finally:
            sys.modules.pop("twinr.__main__", None)

    def test_assert_pi_runtime_root_accepts_twinr_cwd_and_source_root(self) -> None:
        main_mod = importlib.import_module("twinr.__main__")
        twinr_package = importlib.import_module("twinr")
        try:
            with patch.object(main_mod, "_is_raspberry_pi_host", return_value=True):
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

    def test_main_primes_user_audio_env_before_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                f"TWINR_RUNTIME_STATE_PATH={root / 'runtime-state.json'}\n",
                encoding="utf-8",
            )
            fake_loop = _FakeDisplayLoop()
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                main_mod = importlib.import_module("twinr.__main__")
                with patch("twinr.display.TwinrStatusDisplayLoop.from_config", return_value=fake_loop):
                    with patch("twinr.ops.loop_instance_lock", _fake_lock):
                        with patch.object(main_mod, "prime_user_session_audio_env") as priming:
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
        priming.assert_called_once_with()

    def test_lock_contention_does_not_poison_runtime_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_state_path = root / "runtime-state.json"
            env_path = root / ".env"
            env_path.write_text(
                f"TWINR_RUNTIME_STATE_PATH={runtime_state_path}\n",
                encoding="utf-8",
            )
            fake_loop = _FakeDisplayLoop()
            original_argv = list(sys.argv)

            @contextmanager
            def _contention_lock(_config, _name: str):
                raise TwinrInstanceAlreadyRunningError(label="display loop", owner_pid=4242)
                yield

            try:
                sys.modules.pop("twinr.__main__", None)
                main_mod = importlib.import_module("twinr.__main__")
                with patch("twinr.display.TwinrStatusDisplayLoop.from_config", return_value=fake_loop):
                    with patch("twinr.ops.loop_instance_lock", _contention_lock):
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

            payload = json.loads(runtime_state_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["status"], "waiting")
        self.assertIsNone(payload["error_message"])

    def test_run_web_skips_runtime_bootstrap_before_uvicorn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    (
                        f"TWINR_RUNTIME_STATE_PATH={root / 'runtime-state.json'}",
                        "TWINR_WEB_HOST=127.0.0.1",
                        "TWINR_WEB_PORT=1447",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            fake_app = object()
            uvicorn_calls: list[tuple[object, str, int]] = []
            fake_uvicorn = ModuleType("uvicorn")
            fake_uvicorn.run = lambda app, host, port: uvicorn_calls.append((app, host, port))
            fake_web_module = ModuleType("twinr.web")
            fake_web_module.create_app = lambda provided_env: fake_app if Path(provided_env) == env_path else None
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(
                    sys.modules,
                    {
                        "twinr.web": fake_web_module,
                        "uvicorn": fake_uvicorn,
                    },
                ):
                    main_mod = importlib.import_module("twinr.__main__")
                    with patch.object(
                        main_mod,
                        "_build_runtime",
                        side_effect=AssertionError("run-web must not bootstrap the runtime"),
                    ):
                        sys.argv = [
                            "twinr",
                            "--env-file",
                            str(env_path),
                            "--run-web",
                        ]
                        exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        self.assertEqual(uvicorn_calls, [(fake_app, "127.0.0.1", 1447)])

    def test_run_whatsapp_channel_dispatches_to_channel_loop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    (
                        f"TWINR_RUNTIME_STATE_PATH={root / 'runtime-state.json'}",
                        "TWINR_WHATSAPP_ALLOW_FROM=+49 171 1234567",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            fake_runtime = SimpleNamespace(status=SimpleNamespace(value="waiting"))
            fake_backend = object()
            built_runtime_config = None
            lock_runtime_state_path = None
            fake_whatsapp_module = ModuleType("twinr.channels.whatsapp")
            fake_whatsapp_module.TwinrWhatsAppChannelLoop = _FakeWhatsAppLoop
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(sys.modules, {"twinr.channels.whatsapp": fake_whatsapp_module}):
                    main_mod = importlib.import_module("twinr.__main__")

                    def _build_runtime(config):
                        nonlocal built_runtime_config
                        built_runtime_config = config
                        return fake_runtime

                    @contextmanager
                    def _recording_lock(config, _name: str):
                        nonlocal lock_runtime_state_path
                        lock_runtime_state_path = str(config.runtime_state_path)
                        yield

                    with patch.object(main_mod, "_build_runtime", side_effect=_build_runtime):
                        with patch("twinr.providers.openai.OpenAIBackend", return_value=fake_backend):
                            with patch("twinr.ops.loop_instance_lock", _recording_lock):
                                sys.argv = [
                                    "twinr",
                                    "--env-file",
                                    str(env_path),
                                    "--run-whatsapp-channel",
                                    "--loop-duration",
                                    "0",
                                ]
                                exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        self.assertIsNotNone(built_runtime_config)
        self.assertIn("runtime-scopes/whatsapp-channel", str(built_runtime_config.runtime_state_path))
        self.assertFalse(built_runtime_config.restore_runtime_state_on_startup)
        self.assertEqual(lock_runtime_state_path, str(root / "runtime-state.json"))

    def test_run_whatsapp_channel_ensures_remote_watchdog_before_runtime_boot_on_pi(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    (
                        f"TWINR_RUNTIME_STATE_PATH={root / 'runtime-state.json'}",
                        "TWINR_OPENAI_API_KEY=sk-test",
                        "TWINR_WHATSAPP_ALLOW_FROM=+49 171 1234567",
                        "TWINR_LONG_TERM_MEMORY_ENABLED=true",
                        "TWINR_LONG_TERM_MEMORY_MODE=remote_primary",
                        "TWINR_LONG_TERM_MEMORY_REMOTE_REQUIRED=true",
                        "TWINR_LONG_TERM_MEMORY_REMOTE_RUNTIME_CHECK_MODE=watchdog_artifact",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            events: list[str] = []
            fake_runtime = SimpleNamespace(status=SimpleNamespace(value="waiting"))
            fake_backend = object()
            fake_whatsapp_module = ModuleType("twinr.channels.whatsapp")
            fake_whatsapp_module.TwinrWhatsAppChannelLoop = _FakeWhatsAppLoop
            fake_watchdog_module = ModuleType("twinr.ops.remote_memory_watchdog_companion")

            def _ensure_remote_memory_watchdog_process(_config, *, env_file):
                events.append(f"watchdog:{env_file}:{_config.runtime_state_path}")
                return 4321

            fake_watchdog_module.ensure_remote_memory_watchdog_process = _ensure_remote_memory_watchdog_process
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(
                    sys.modules,
                    {
                        "twinr.channels.whatsapp": fake_whatsapp_module,
                        "twinr.ops.remote_memory_watchdog_companion": fake_watchdog_module,
                    },
                ):
                    main_mod = importlib.import_module("twinr.__main__")

                    def _build_runtime(_config):
                        events.append("build_runtime")
                        return fake_runtime

                    with patch.object(main_mod, "_build_runtime", side_effect=_build_runtime):
                        with patch.object(main_mod, "_assert_pi_runtime_root", return_value=None):
                            with patch.object(main_mod, "_uses_pi_runtime_root", return_value=True):
                                with patch("twinr.providers.openai.OpenAIBackend", return_value=fake_backend):
                                    with patch("twinr.ops.loop_instance_lock", _fake_lock):
                                        sys.argv = [
                                            "twinr",
                                            "--env-file",
                                            str(env_path),
                                            "--run-whatsapp-channel",
                                            "--loop-duration",
                                            "0",
                                        ]
                                        exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        self.assertEqual(events, [f"watchdog:{env_path}:{root / 'runtime-state.json'}", "build_runtime"])

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
            fake_runner_module = ModuleType("twinr.agent.legacy.classic_hardware_loop")
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
                        "twinr.agent.legacy.classic_hardware_loop": fake_runner_module,
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

    def test_streaming_loop_ensures_remote_watchdog_companion_for_pi_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    (
                        f"TWINR_RUNTIME_STATE_PATH={root / 'runtime-state.json'}",
                        "TWINR_OPENAI_API_KEY=sk-test",
                        "TWINR_LONG_TERM_MEMORY_ENABLED=true",
                        "TWINR_LONG_TERM_MEMORY_MODE=remote_primary",
                        "TWINR_LONG_TERM_MEMORY_REMOTE_REQUIRED=true",
                        "TWINR_LONG_TERM_MEMORY_REMOTE_RUNTIME_CHECK_MODE=watchdog_artifact",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            fake_loop = _FakeHardwareLoop()
            watchdog_calls: list[str] = []
            fake_streaming_module = ModuleType("twinr.agent.workflows.streaming_runner")
            fake_streaming_module.TwinrStreamingHardwareLoop = lambda **_kwargs: fake_loop
            fake_openai_module = ModuleType("twinr.providers.openai")

            class _FakeBackend:
                def __init__(self, config) -> None:
                    self.config = config

            fake_openai_module.OpenAIBackend = _FakeBackend
            fake_providers_module = ModuleType("twinr.providers")
            fake_providers_module.build_streaming_provider_bundle = lambda *args, **kwargs: SimpleNamespace(
                print_backend=SimpleNamespace(),
                stt=SimpleNamespace(),
                agent=SimpleNamespace(),
                tts=SimpleNamespace(),
                tool_agent=SimpleNamespace(),
            )
            fake_companion_module = ModuleType("twinr.display.companion")

            @contextmanager
            def _fake_companion(_config, *, enabled: bool):
                yield

            fake_companion_module.optional_display_companion = _fake_companion
            fake_watchdog_module = ModuleType("twinr.ops.remote_memory_watchdog_companion")

            def _ensure_remote_memory_watchdog_process(_config, *, env_file):
                watchdog_calls.append(str(env_file))
                return 4321

            fake_watchdog_module.ensure_remote_memory_watchdog_process = _ensure_remote_memory_watchdog_process
            fake_ops_module = ModuleType("twinr.ops")
            fake_ops_module.loop_instance_lock = _fake_lock
            fake_runtime = SimpleNamespace(status=SimpleNamespace(value="waiting"))
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(
                    sys.modules,
                    {
                        "twinr.agent.workflows.streaming_runner": fake_streaming_module,
                        "twinr.providers": fake_providers_module,
                        "twinr.providers.openai": fake_openai_module,
                        "twinr.display.companion": fake_companion_module,
                        "twinr.ops": fake_ops_module,
                        "twinr.ops.remote_memory_watchdog_companion": fake_watchdog_module,
                    },
                ):
                    main_mod = importlib.import_module("twinr.__main__")
                    with patch.object(main_mod, "_should_enable_display_companion", return_value=False):
                        with patch.object(main_mod, "_assert_pi_runtime_root", return_value=None):
                            with patch.object(main_mod, "_uses_pi_runtime_root", return_value=True):
                                with patch.object(main_mod, "_build_runtime", return_value=fake_runtime):
                                    sys.argv = [
                                        "twinr",
                                        "--env-file",
                                        str(env_path),
                                        "--run-streaming-loop",
                                        "--loop-duration",
                                        "0",
                                    ]
                                    exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        self.assertEqual(watchdog_calls, [str(env_path)])

    def test_streaming_loop_acquires_lock_and_display_companion_before_runtime_bootstrap(self) -> None:
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
            events: list[str] = []
            fake_loop = _FakeHardwareLoop()
            fake_streaming_module = ModuleType("twinr.agent.workflows.streaming_runner")
            fake_streaming_module.TwinrStreamingHardwareLoop = lambda **_kwargs: (
                events.append("loop_init") or fake_loop
            )
            fake_openai_module = ModuleType("twinr.providers.openai")

            class _FakeBackend:
                def __init__(self, config) -> None:
                    del config
                    events.append("backend_init")

            fake_openai_module.OpenAIBackend = _FakeBackend
            fake_providers_module = ModuleType("twinr.providers")
            fake_providers_module.build_streaming_provider_bundle = lambda *args, **kwargs: (
                events.append("bundle_init")
                or SimpleNamespace(
                    print_backend=SimpleNamespace(),
                    stt=SimpleNamespace(),
                    agent=SimpleNamespace(),
                    tts=SimpleNamespace(),
                    tool_agent=SimpleNamespace(),
                )
            )
            fake_companion_module = ModuleType("twinr.display.companion")

            @contextmanager
            def _fake_companion(_config, *, enabled: bool):
                events.append(f"companion_enter:{str(enabled).lower()}")
                try:
                    yield
                finally:
                    events.append("companion_exit")

            fake_companion_module.optional_display_companion = _fake_companion
            fake_ops_module = ModuleType("twinr.ops")

            @contextmanager
            def _recording_lock(_config, _name: str):
                events.append("lock_enter")
                try:
                    yield
                finally:
                    events.append("lock_exit")

            fake_ops_module.loop_instance_lock = _recording_lock
            fake_runtime = SimpleNamespace(status=SimpleNamespace(value="waiting"))
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(
                    sys.modules,
                    {
                        "twinr.agent.workflows.streaming_runner": fake_streaming_module,
                        "twinr.providers": fake_providers_module,
                        "twinr.providers.openai": fake_openai_module,
                        "twinr.display.companion": fake_companion_module,
                        "twinr.ops": fake_ops_module,
                    },
                ):
                    main_mod = importlib.import_module("twinr.__main__")
                    with patch.object(main_mod, "_should_enable_display_companion", return_value=True):
                        with patch.object(main_mod, "_should_ensure_remote_watchdog_companion", return_value=False):
                            with patch.object(main_mod, "_assert_pi_runtime_root", return_value=None):
                                with patch.object(
                                    main_mod,
                                    "_build_runtime",
                                    side_effect=lambda _config: events.append("runtime_init") or fake_runtime,
                                ):
                                    sys.argv = [
                                        "twinr",
                                        "--env-file",
                                        str(env_path),
                                        "--run-streaming-loop",
                                        "--loop-duration",
                                        "0",
                                    ]
                                    exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            events[:6],
            [
                "lock_enter",
                "companion_enter:true",
                "runtime_init",
                "backend_init",
                "bundle_init",
                "loop_init",
            ],
        )
        self.assertEqual(events[-2:], ["companion_exit", "lock_exit"])

    def test_streaming_loop_holds_error_state_instead_of_exiting_companion_on_boot_failure(self) -> None:
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
            events: list[str] = []
            hold_calls: list[dict[str, object]] = []
            fake_streaming_module = ModuleType("twinr.agent.workflows.streaming_runner")

            def _raise_loop(**_kwargs):
                events.append("loop_init")
                raise RuntimeError("capture unreadable")

            fake_streaming_module.TwinrStreamingHardwareLoop = _raise_loop
            fake_openai_module = ModuleType("twinr.providers.openai")

            class _FakeBackend:
                def __init__(self, config) -> None:
                    del config
                    events.append("backend_init")

            fake_openai_module.OpenAIBackend = _FakeBackend
            fake_providers_module = ModuleType("twinr.providers")
            fake_providers_module.build_streaming_provider_bundle = lambda *args, **kwargs: (
                events.append("bundle_init")
                or SimpleNamespace(
                    print_backend=SimpleNamespace(),
                    stt=SimpleNamespace(),
                    agent=SimpleNamespace(),
                    tts=SimpleNamespace(),
                    tool_agent=SimpleNamespace(),
                )
            )
            fake_companion_module = ModuleType("twinr.display.companion")

            @contextmanager
            def _fake_companion(_config, *, enabled: bool):
                events.append(f"companion_enter:{str(enabled).lower()}")
                try:
                    yield
                finally:
                    events.append("companion_exit")

            fake_companion_module.optional_display_companion = _fake_companion
            fake_ops_module = ModuleType("twinr.ops")
            fake_ops_module.loop_instance_lock = _fake_lock
            fake_hold_module = ModuleType("twinr.agent.workflows.runtime_error_hold")

            def _fake_hold_runtime_error_state(*, runtime, error, duration_s, **_kwargs):
                hold_calls.append(
                    {
                        "runtime": runtime,
                        "error": str(error),
                        "duration_s": duration_s,
                    }
                )
                return 71

            fake_hold_module.hold_runtime_error_state = _fake_hold_runtime_error_state
            fake_runtime = SimpleNamespace(status=SimpleNamespace(value="waiting"))
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(
                    sys.modules,
                    {
                        "twinr.agent.workflows.streaming_runner": fake_streaming_module,
                        "twinr.agent.workflows.runtime_error_hold": fake_hold_module,
                        "twinr.providers": fake_providers_module,
                        "twinr.providers.openai": fake_openai_module,
                        "twinr.display.companion": fake_companion_module,
                        "twinr.ops": fake_ops_module,
                    },
                ):
                    main_mod = importlib.import_module("twinr.__main__")
                    with patch.object(main_mod, "_should_enable_display_companion", return_value=True):
                        with patch.object(main_mod, "_should_ensure_remote_watchdog_companion", return_value=False):
                            with patch.object(main_mod, "_assert_pi_runtime_root", return_value=None):
                                with patch.object(main_mod, "_build_runtime", return_value=fake_runtime):
                                    sys.argv = [
                                        "twinr",
                                        "--env-file",
                                        str(env_path),
                                        "--run-streaming-loop",
                                        "--loop-duration",
                                        "0",
                                    ]
                                    exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 71)
        self.assertEqual(len(hold_calls), 1)
        self.assertIs(hold_calls[0]["runtime"], fake_runtime)
        self.assertEqual(hold_calls[0]["error"], "capture unreadable")
        self.assertEqual(hold_calls[0]["duration_s"], 0.0)
        self.assertEqual(events[-1], "companion_exit")

    def test_watch_remote_memory_dispatches_without_runtime_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                f"TWINR_RUNTIME_STATE_PATH={root / 'runtime-state.json'}\n",
                encoding="utf-8",
            )
            fake_watchdog = _FakeRemoteMemoryWatchdog()
            fake_ops_module = ModuleType("twinr.ops")
            fake_ops_module.loop_instance_lock = _fake_lock
            fake_ops_module.RemoteMemoryWatchdog = SimpleNamespace(from_config=lambda _config: fake_watchdog)
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(sys.modules, {"twinr.ops": fake_ops_module}):
                    main_mod = importlib.import_module("twinr.__main__")
                    with patch.object(main_mod, "_build_runtime", side_effect=AssertionError("runtime must not be created")):
                        sys.argv = [
                            "twinr",
                            "--env-file",
                            str(env_path),
                            "--watch-remote-memory",
                            "--loop-duration",
                            "0",
                        ]
                        exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        self.assertEqual(fake_watchdog.duration_s, 0.0)

    def test_run_runtime_supervisor_dispatches_without_runtime_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                f"TWINR_RUNTIME_STATE_PATH={root / 'runtime-state.json'}\n",
                encoding="utf-8",
            )
            fake_supervisor = _FakeRuntimeSupervisor()
            fake_supervisor_module = ModuleType("twinr.ops.runtime_supervisor")
            fake_supervisor_module.TwinrRuntimeSupervisor = lambda **kwargs: (
                setattr(fake_supervisor, "env_file", kwargs["env_file"]) or fake_supervisor
            )
            fake_ops_module = ModuleType("twinr.ops")
            fake_ops_module.loop_instance_lock = _fake_lock
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(
                    sys.modules,
                    {
                        "twinr.ops": fake_ops_module,
                        "twinr.ops.runtime_supervisor": fake_supervisor_module,
                    },
                ):
                    main_mod = importlib.import_module("twinr.__main__")
                    with patch.object(main_mod, "_build_runtime", side_effect=AssertionError("runtime must not be created")):
                        sys.argv = [
                            "twinr",
                            "--env-file",
                            str(env_path),
                            "--run-runtime-supervisor",
                            "--loop-duration",
                            "0",
                        ]
                        exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        self.assertEqual(fake_supervisor.duration_s, 0.0)
        self.assertEqual(fake_supervisor.env_file, str(env_path))

    def test_self_coding_codex_self_test_dispatches_without_runtime_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("", encoding="utf-8")
            fake_environment_module = ModuleType("twinr.agent.self_coding.codex_driver.environment")
            fake_environment_module.collect_codex_sdk_environment_report = lambda **_kwargs: SimpleNamespace(
                status="ok",
                ready=True,
                detail="bridge self-test ok",
                node_version="v18.20.4",
                npm_version="9.2.0",
                codex_version="codex-cli 0.114.0",
                auth_present=True,
                local_self_test_ok=True,
                live_auth_check_ok=True,
            )
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(
                    sys.modules,
                    {"twinr.agent.self_coding.codex_driver.environment": fake_environment_module},
                ):
                    main_mod = importlib.import_module("twinr.__main__")
                    with patch.object(main_mod, "_build_runtime", side_effect=AssertionError("runtime must not be created")):
                        sys.argv = [
                            "twinr",
                            "--env-file",
                            str(env_path),
                            "--self-coding-codex-self-test",
                            "--self-coding-live-auth-check",
                        ]
                        exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)

    def test_self_coding_morning_briefing_acceptance_dispatches_without_runtime_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("", encoding="utf-8")
            calls: list[dict[str, object]] = []
            fake_acceptance_module = ModuleType("twinr.agent.self_coding.live_acceptance")

            def _run_live_acceptance(**kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    job_id="job_acceptance",
                    job_status="soft_launch_ready",
                    skill_id="morning_briefing",
                    version=1,
                    activation_status="active",
                    refresh_status="ok",
                    delivery_status="ok",
                    delivery_delivered=True,
                    search_call_count=3,
                    summary_call_count=1,
                    spoken_count=1,
                    last_summary_text="Guten Morgen.",
                )

            fake_acceptance_module.run_live_morning_briefing_acceptance = _run_live_acceptance
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(
                    sys.modules,
                    {"twinr.agent.self_coding.live_acceptance": fake_acceptance_module},
                ):
                    main_mod = importlib.import_module("twinr.__main__")
                    with patch.object(main_mod, "_build_runtime", side_effect=AssertionError("runtime must not be created")):
                        sys.argv = [
                            "twinr",
                            "--env-file",
                            str(env_path),
                            "--self-coding-morning-briefing-acceptance",
                            "--self-coding-acceptance-capture-only",
                        ]
                        exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0]["speak_out_loud"])
        self.assertEqual(calls[0]["live_e2e_environment"], "local")

    def test_long_term_memory_live_acceptance_dispatches_without_runtime_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("", encoding="utf-8")
            calls: list[dict[str, object]] = []
            fake_acceptance_module = ModuleType("twinr.memory.longterm.evaluation.live_memory_acceptance")

            def _run_live_memory_acceptance(**kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    probe_id="probe_live_memory",
                    status="ok",
                    ready=True,
                    passed_cases=8,
                    total_cases=8,
                    queue_before_count=1,
                    queue_after_count=0,
                    restart_queue_count=0,
                    artifact_path=str(root / "artifacts" / "stores" / "ops" / "memory_live_acceptance.json"),
                    report_path=str(root / "artifacts" / "reports" / "memory_live_acceptance" / "probe_live_memory.json"),
                )

            fake_acceptance_module.run_live_memory_acceptance = _run_live_memory_acceptance
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(
                    sys.modules,
                    {"twinr.memory.longterm.evaluation.live_memory_acceptance": fake_acceptance_module},
                ):
                    main_mod = importlib.import_module("twinr.__main__")
                    with patch.object(main_mod, "_build_runtime", side_effect=AssertionError("runtime must not be created")):
                        sys.argv = [
                            "twinr",
                            "--env-file",
                            str(env_path),
                            "--long-term-memory-live-acceptance",
                        ]
                        exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["env_path"], str(env_path))

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

    def test_wakeword_stream_eval_dispatches_to_proactive_helper(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("TWINR_WAKEWORD_VERIFIER_MODE=disabled\n", encoding="utf-8")
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text("", encoding="utf-8")
            calls: list[tuple[Path, object | None]] = []
            fake_proactive_module = ModuleType("twinr.proactive")

            def _run_stream_eval(*, config, manifest_path=None, backend=None):
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
                    accepted_detection_count=1,
                    total_audio_seconds=2.0,
                    report_path=root / "stream_report.json",
                )

            fake_proactive_module.run_wakeword_stream_eval = _run_stream_eval
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(sys.modules, {"twinr.proactive": fake_proactive_module}):
                    main_mod = importlib.import_module("twinr.__main__")
                    sys.argv = [
                        "twinr",
                        "--env-file",
                        str(env_path),
                        "--wakeword-stream-eval",
                        "--wakeword-manifest",
                        str(manifest_path),
                    ]
                    exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls, [(manifest_path, None)])

    def test_wakeword_promotion_eval_dispatches_to_proactive_helper(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("TWINR_WAKEWORD_VERIFIER_MODE=disabled\n", encoding="utf-8")
            spec_path = root / "promotion_spec.json"
            spec_path.write_text("{}\n", encoding="utf-8")
            calls: list[tuple[Path, object | None]] = []
            fake_proactive_module = ModuleType("twinr.proactive")

            def _run_promotion_eval(*, config, spec_path=None, backend=None):
                del config
                calls.append((Path(spec_path), backend))
                return SimpleNamespace(
                    passed=True,
                    blockers=(),
                    suite_results=(),
                    ambient_results=(),
                    report_path=root / "promotion_report.json",
                )

            fake_proactive_module.run_wakeword_promotion_eval = _run_promotion_eval
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(sys.modules, {"twinr.proactive": fake_proactive_module}):
                    main_mod = importlib.import_module("twinr.__main__")
                    sys.argv = [
                        "twinr",
                        "--env-file",
                        str(env_path),
                        "--wakeword-promotion-eval",
                        "--wakeword-promotion-spec",
                        str(spec_path),
                    ]
                    exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls, [(spec_path, None)])

    def test_runtime_init_failure_returns_error_without_unbound_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("TWINR_WAKEWORD_VERIFIER_MODE=disabled\n", encoding="utf-8")
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                main_mod = importlib.import_module("twinr.__main__")
                with patch.object(main_mod, "_ensure_remote_watchdog_for_runtime_boot", return_value=None):
                    with patch.object(main_mod, "_build_runtime", side_effect=RuntimeError("boom")):
                        sys.argv = [
                            "twinr",
                            "--env-file",
                            str(env_path),
                        ]
                        exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 1)

    def test_wakeword_train_verifier_dispatches_to_wakeword_helper(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("TWINR_WAKEWORD_VERIFIER_MODE=disabled\n", encoding="utf-8")
            manifest_path = root / "captured_manifest.json"
            manifest_path.write_text("[]\n", encoding="utf-8")
            model_path = root / "twinr_v1.onnx"
            model_path.write_bytes(b"model")
            output_path = root / "twinr_v1.verifier.pkl"
            calls: list[tuple[Path, Path, str, str]] = []
            fake_proactive_module = ModuleType("twinr.proactive")
            fake_wakeword_module = ModuleType("twinr.proactive.wakeword")

            def _train_verifier(*, manifest_path, output_path, model_name, inference_framework):
                calls.append((Path(manifest_path), Path(output_path), model_name, inference_framework))
                return SimpleNamespace(
                    manifest_path=Path(manifest_path),
                    output_path=Path(output_path),
                    model_name=model_name,
                    positive_clips=3,
                    negative_clips=2,
                    negative_seconds=12.0,
                )

            fake_proactive_module.wakeword = fake_wakeword_module
            fake_wakeword_module.train_wakeword_custom_verifier_from_manifest = _train_verifier
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(
                    sys.modules,
                    {
                        "twinr.proactive": fake_proactive_module,
                        "twinr.proactive.wakeword": fake_wakeword_module,
                    },
                ):
                    main_mod = importlib.import_module("twinr.__main__")
                    sys.argv = [
                        "twinr",
                        "--env-file",
                        str(env_path),
                        "--wakeword-train-verifier",
                        "--wakeword-manifest",
                        str(manifest_path),
                        "--wakeword-verifier-model",
                        str(model_path),
                        "--wakeword-verifier-output",
                        str(output_path),
                    ]
                    exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls, [(manifest_path, output_path, str(model_path), "tflite")])

    def test_wakeword_train_sequence_verifier_dispatches_to_wakeword_helper(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("TWINR_WAKEWORD_VERIFIER_MODE=disabled\n", encoding="utf-8")
            manifest_path = root / "captured_manifest.json"
            manifest_path.write_text("[]\n", encoding="utf-8")
            model_path = root / "twinr_v2.onnx"
            model_path.write_bytes(b"model")
            aux_model_path = root / "twinr_v1.onnx"
            aux_model_path.write_bytes(b"aux")
            output_path = root / "twinr_v2.sequence_verifier.pkl"
            calls: list[tuple[Path, Path, str, tuple[str, ...], str]] = []
            fake_proactive_module = ModuleType("twinr.proactive")
            fake_wakeword_module = ModuleType("twinr.proactive.wakeword")

            def _train_sequence_verifier(
                *,
                manifest_path,
                output_path,
                model_name,
                auxiliary_models,
                inference_framework,
            ):
                calls.append(
                    (
                        Path(manifest_path),
                        Path(output_path),
                        model_name,
                        tuple(auxiliary_models),
                        inference_framework,
                    )
                )
                return SimpleNamespace(
                    manifest_path=Path(manifest_path),
                    output_path=Path(output_path),
                    model_name=model_name,
                    auxiliary_models=tuple(auxiliary_models),
                    positive_clips=3,
                    negative_clips=2,
                    negative_seconds=12.0,
                    total_length_samples=32000,
                    embedding_frames=16,
                    feature_dimensions=1584,
                )

            fake_proactive_module.wakeword = fake_wakeword_module
            fake_wakeword_module.train_wakeword_sequence_verifier_from_manifest = _train_sequence_verifier
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(
                    sys.modules,
                    {
                        "twinr.proactive": fake_proactive_module,
                        "twinr.proactive.wakeword": fake_wakeword_module,
                    },
                ):
                    main_mod = importlib.import_module("twinr.__main__")
                    sys.argv = [
                        "twinr",
                        "--env-file",
                        str(env_path),
                        "--wakeword-train-sequence-verifier",
                        "--wakeword-manifest",
                        str(manifest_path),
                        "--wakeword-sequence-verifier-model",
                        str(model_path),
                        "--wakeword-sequence-verifier-aux-model",
                        str(aux_model_path),
                        "--wakeword-sequence-verifier-output",
                        str(output_path),
                    ]
                    exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            calls,
            [(manifest_path, output_path, str(model_path), (str(aux_model_path),), "tflite")],
        )

    def test_wakeword_train_model_dispatches_to_wakeword_helper(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("TWINR_WAKEWORD_VERIFIER_MODE=disabled\n", encoding="utf-8")
            dataset_root = root / "dataset"
            dataset_root.mkdir()
            manifest_path = root / "captured_manifest.json"
            manifest_path.write_text("[]\n", encoding="utf-8")
            model_path = root / "twinr_v2.onnx"
            metadata_path = root / "twinr_v2.metadata.json"
            calls: list[tuple[Path, Path, Path, Path, object, int, str, int, int, str, object, float, float, float]] = []
            fake_proactive_module = ModuleType("twinr.proactive")
            fake_wakeword_module = ModuleType("twinr.proactive.wakeword")

            def _train_model(
                *,
                dataset_root,
                output_model_path,
                metadata_path,
                acceptance_manifest,
                workdir,
                training_rounds,
                model_type,
                layer_dim,
                steps,
                feature_device,
                difficulty_reference_model_path,
                difficulty_positive_scale,
                difficulty_negative_scale,
                difficulty_power,
                evaluation_config,
            ):
                del evaluation_config
                calls.append(
                    (
                        Path(dataset_root),
                        Path(output_model_path),
                        Path(metadata_path),
                        Path(acceptance_manifest),
                        workdir,
                        training_rounds,
                        model_type,
                        layer_dim,
                        steps,
                        feature_device,
                        difficulty_reference_model_path,
                        difficulty_positive_scale,
                        difficulty_negative_scale,
                        difficulty_power,
                    )
                )
                return SimpleNamespace(
                    dataset_root=Path(dataset_root),
                    output_model_path=Path(output_model_path),
                    metadata_path=Path(metadata_path),
                    total_length_samples=32000,
                    train_positive_clips=2,
                    train_negative_clips=2,
                    validation_positive_clips=1,
                    validation_negative_clips=1,
                    selected_threshold=0.12,
                    acceptance_metrics=None,
                )

            fake_proactive_module.wakeword = fake_wakeword_module
            fake_wakeword_module.train_wakeword_base_model_from_dataset_root = _train_model
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(
                    sys.modules,
                    {
                        "twinr.proactive": fake_proactive_module,
                        "twinr.proactive.wakeword": fake_wakeword_module,
                    },
                ):
                    main_mod = importlib.import_module("twinr.__main__")
                    sys.argv = [
                        "twinr",
                        "--env-file",
                        str(env_path),
                        "--wakeword-train-model",
                        "--wakeword-dataset-root",
                        str(dataset_root),
                        "--wakeword-model-output",
                        str(model_path),
                        "--wakeword-model-metadata-output",
                        str(metadata_path),
                        "--wakeword-manifest",
                        str(manifest_path),
                        "--wakeword-training-rounds",
                        "3",
                        "--wakeword-training-model-type",
                        "mlp",
                        "--wakeword-training-layer-dim",
                        "256",
                        "--wakeword-training-steps",
                        "1234",
                        "--wakeword-training-feature-device",
                        "gpu",
                    ]
                    exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            calls,
            [(
                dataset_root,
                model_path,
                metadata_path,
                manifest_path,
                None,
                3,
                "mlp",
                256,
                1234,
                "gpu",
                None,
                0.0,
                0.0,
                2.0,
            )],
        )

    def test_wakeword_training_plan_writes_markdown(self) -> None:
        rendered = None
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("TWINR_WAKEWORD_VERIFIER_MODE=disabled\n", encoding="utf-8")
            output_path = root / "wakeword_training_plan.md"
            fake_proactive_module = ModuleType("twinr.proactive")
            fake_wakeword_module = ModuleType("twinr.proactive.wakeword")
            calls: list[Path] = []

            def _build_plan(*, project_root):
                calls.append(Path(project_root))
                return SimpleNamespace(
                    stage1_model_name="twinr_family_stage1_vnext",
                    stage1_phrase_profile="family",
                )

            def _render_plan(_plan):
                return "# fake wakeword training plan\n"

            fake_proactive_module.wakeword = fake_wakeword_module
            fake_wakeword_module.build_default_wakeword_training_plan = _build_plan
            fake_wakeword_module.render_wakeword_training_plan_markdown = _render_plan
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(
                    sys.modules,
                    {
                        "twinr.proactive": fake_proactive_module,
                        "twinr.proactive.wakeword": fake_wakeword_module,
                    },
                ):
                    main_mod = importlib.import_module("twinr.__main__")
                    sys.argv = [
                        "twinr",
                        "--env-file",
                        str(env_path),
                        "--wakeword-training-plan",
                        "--wakeword-training-plan-output",
                        str(output_path),
                    ]
                    exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

            rendered = output_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls, [root])
        self.assertEqual(rendered, "# fake wakeword training plan\n")

    def test_wakeword_kws_provision_dispatches_to_wakeword_helper(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("TWINR_WAKEWORD_VERIFIER_MODE=disabled\n", encoding="utf-8")
            output_dir = root / "kws"
            calls: list[tuple[Path, str, tuple[str, ...], bool]] = []
            fake_proactive_module = ModuleType("twinr.proactive")
            fake_wakeword_module = ModuleType("twinr.proactive.wakeword")

            def _provision_bundle(*, output_dir, bundle_id, phrases, explicit_keywords, force):
                del phrases
                calls.append((Path(output_dir), bundle_id, tuple(explicit_keywords), force))
                return SimpleNamespace(
                    bundle_id=bundle_id,
                    output_dir=Path(output_dir),
                    keyword_names=tuple(explicit_keywords),
                    tokens_path=Path(output_dir) / "tokens.txt",
                    encoder_path=Path(output_dir) / "encoder.onnx",
                    decoder_path=Path(output_dir) / "decoder.onnx",
                    joiner_path=Path(output_dir) / "joiner.onnx",
                    keywords_path=Path(output_dir) / "keywords.txt",
                )

            fake_proactive_module.wakeword = fake_wakeword_module
            fake_wakeword_module.provision_builtin_kws_bundle = _provision_bundle
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(
                    sys.modules,
                    {
                        "twinr.proactive": fake_proactive_module,
                        "twinr.proactive.wakeword": fake_wakeword_module,
                    },
                ):
                    main_mod = importlib.import_module("twinr.__main__")
                    sys.argv = [
                        "twinr",
                        "--env-file",
                        str(env_path),
                        "--wakeword-kws-provision",
                        "--wakeword-kws-output-dir",
                        str(output_dir),
                        "--wakeword-kws-bundle",
                        "gigaspeech_3_3m_bpe_int8",
                        "--wakeword-kws-keyword",
                        "Twinna",
                        "--wakeword-kws-keyword",
                        "Twinr",
                        "--wakeword-kws-force",
                    ]
                    exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            calls,
            [(output_dir, "gigaspeech_3_3m_bpe_int8", ("Twinna", "Twinr"), True)],
        )

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
