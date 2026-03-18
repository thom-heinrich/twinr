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
