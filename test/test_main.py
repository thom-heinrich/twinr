from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import importlib
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.ops.locks import TwinrInstanceAlreadyRunningError

_TEST_WHATSAPP_ALLOW_FROM_DISPLAY = "+1 555 555 4567"


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


class _FakeTurnRuntime:
    def __init__(self) -> None:
        self.status = SimpleNamespace(value="waiting")
        self.failed_message = None
        self.submitted_transcript = None

    def press_green_button(self) -> None:
        self.status.value = "listening"

    def submit_transcript(self, transcript: str) -> None:
        self.submitted_transcript = transcript
        self.status.value = "processing"

    def conversation_context(self) -> list[object]:
        return []

    def complete_agent_turn(self, answer: str) -> str:
        self.status.value = "speaking"
        return answer

    def finish_speaking(self) -> None:
        self.status.value = "waiting"

    def fail(self, message: str) -> None:
        self.failed_message = message
        self.status.value = "error"


class _FakeRemoteMemoryWatchdog:
    def __init__(self) -> None:
        self.duration_s = None
        self.artifact_path = Path("/tmp/remote-memory-watchdog.json")
        self.interval_s = 1.0
        self.history_limit = 3600

    def run(self, *, duration_s: float | None = None) -> int:
        self.duration_s = duration_s
        return 0


def _fake_runtime_env_module() -> ModuleType:
    module = ModuleType("twinr.ops.runtime_env")
    module.prime_user_session_audio_env = lambda: None
    return module


def _attach_fake_openai_image_input(module: ModuleType) -> None:
    class _FakeImageInput:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        @classmethod
        def from_path(cls, *_args, **_kwargs):
            return cls()

    module.OpenAIImageInput = _FakeImageInput


class _FakeWhatsAppLoop:
    last_instance = None

    def __init__(self, *, config, runtime, backend, tool_agent_provider=None, print_backend=None) -> None:
        self.config = config
        self.runtime = runtime
        self.backend = backend
        self.tool_agent_provider = tool_agent_provider
        self.print_backend = print_backend
        self.duration_s = None
        type(self).last_instance = self

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


class _FakeCliCamera:
    def __init__(self) -> None:
        self.output_root = None
        self.calls: list[dict[str, object]] = []

    def capture_photo(self, *, output_path=None, filename: str = "camera-capture.png"):
        self.calls.append(
            {
                "output_root": self.output_root,
                "output_path": output_path,
                "filename": filename,
            }
        )
        return SimpleNamespace(
            data=b"png-bytes",
            source_device="/dev/video0",
            input_format="rpicam-still",
            content_type="image/png",
            filename=filename,
        )


@contextmanager
def _fake_lock(_config, _name: str):
    yield


class MainCliTests(unittest.TestCase):
    def test_resolve_runtime_config_scopes_direct_cli_turns_without_restore(self) -> None:
        main_mod = importlib.import_module("twinr.__main__")
        try:
            config = main_mod.TwinrConfig(
                project_root="/tmp/twinr-project",
                runtime_state_path="state/runtime-state.json",
                restore_runtime_state_on_startup=True,
            )

            openai_args = SimpleNamespace(
                run_whatsapp_channel=False,
                demo_transcript=None,
                openai_prompt="Hallo",
                vision_prompt=None,
                orchestrator_probe_turn=None,
            )
            openai_runtime_config = main_mod._resolve_runtime_config(config, openai_args)

            demo_args = SimpleNamespace(
                run_whatsapp_channel=False,
                demo_transcript="Hallo",
                openai_prompt=None,
                vision_prompt=None,
                orchestrator_probe_turn=None,
            )
            demo_runtime_config = main_mod._resolve_runtime_config(config, demo_args)
        finally:
            sys.modules.pop("twinr.__main__", None)

        self.assertIn("runtime-scopes/openai-prompt/runtime-state.json", openai_runtime_config.runtime_state_path)
        self.assertFalse(openai_runtime_config.restore_runtime_state_on_startup)
        self.assertIn("runtime-scopes/demo-transcript/runtime-state.json", demo_runtime_config.runtime_state_path)
        self.assertFalse(demo_runtime_config.restore_runtime_state_on_startup)

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

    def test_should_disable_display_companion_when_runtime_supervisor_is_active(self) -> None:
        main_mod = importlib.import_module("twinr.__main__")
        try:
            enabled = main_mod.TwinrConfig(display_companion_enabled=True)
            with patch.dict(os.environ, {main_mod._RUNTIME_SUPERVISOR_ENV_KEY: "1"}):
                self.assertFalse(main_mod._should_enable_display_companion(enabled, "/twinr/.env"))
        finally:
            sys.modules.pop("twinr.__main__", None)

    def test_should_disable_display_companion_for_hdmi_wayland_voice_runtime_by_default_but_honor_override(
        self,
    ) -> None:
        main_mod = importlib.import_module("twinr.__main__")
        try:
            guarded = main_mod.TwinrConfig(
                display_driver="hdmi_wayland",
                voice_orchestrator_enabled=True,
                voice_orchestrator_ws_url="ws://192.0.2.10:8797/ws/orchestrator/voice",
            )
            explicitly_enabled = main_mod.TwinrConfig(
                display_driver="hdmi_wayland",
                voice_orchestrator_enabled=True,
                display_companion_enabled=True,
                voice_orchestrator_ws_url="ws://192.0.2.10:8797/ws/orchestrator/voice",
            )
            with patch.object(main_mod, "_is_raspberry_pi_host", return_value=True):
                self.assertFalse(main_mod._should_enable_display_companion(guarded, "/twinr/.env"))
                self.assertTrue(main_mod._should_enable_display_companion(explicitly_enabled, "/twinr/.env"))
        finally:
            sys.modules.pop("twinr.__main__", None)

    def test_capture_photo_for_cli_output_temporarily_allows_absolute_operator_path(self) -> None:
        main_mod = importlib.import_module("twinr.__main__")
        try:
            camera = _FakeCliCamera()
            output_path = Path("/tmp/twinr-camera.png")

            capture = main_mod._capture_photo_for_cli_output(
                camera,
                output_path=output_path,
                filename=output_path.name,
            )
        finally:
            sys.modules.pop("twinr.__main__", None)

        self.assertEqual(capture.filename, "twinr-camera.png")
        self.assertIsNone(camera.output_root)
        self.assertEqual(len(camera.calls), 1)
        self.assertEqual(camera.calls[0]["output_root"], output_path.parent.resolve(strict=False))
        self.assertEqual(camera.calls[0]["output_path"], output_path)

    def test_capture_photo_for_cli_output_restores_output_root_after_failure(self) -> None:
        main_mod = importlib.import_module("twinr.__main__")

        class _ExplodingCliCamera(_FakeCliCamera):
            def capture_photo(self, *, output_path=None, filename: str = "camera-capture.png"):
                del output_path, filename
                raise RuntimeError("boom")

        try:
            camera = _ExplodingCliCamera()
            with self.assertRaisesRegex(RuntimeError, "boom"):
                main_mod._capture_photo_for_cli_output(
                    camera,
                    output_path=Path("/tmp/twinr-camera.png"),
                    filename="twinr-camera.png",
                )
        finally:
            sys.modules.pop("twinr.__main__", None)

        self.assertIsNone(camera.output_root)

    def test_camera_capture_output_allows_absolute_operator_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                f"TWINR_RUNTIME_STATE_PATH={root / 'runtime-state.json'}\n",
                encoding="utf-8",
            )
            fake_camera = _FakeCliCamera()
            fake_camera_module = ModuleType("twinr.hardware.camera")
            fake_camera_module.V4L2StillCamera = SimpleNamespace(from_config=lambda _config: fake_camera)
            original_argv = list(sys.argv)
            stdout = StringIO()

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(sys.modules, {"twinr.hardware.camera": fake_camera_module}):
                    main_mod = importlib.import_module("twinr.__main__")
                    with patch.object(main_mod, "_ensure_remote_watchdog_for_runtime_boot", return_value=None):
                        with patch.object(main_mod, "_build_runtime", return_value=_FakeTurnRuntime()):
                            with patch("sys.stdout", stdout):
                                sys.argv = [
                                    "twinr",
                                    "--env-file",
                                    str(env_path),
                                    "--camera-capture-output",
                                    "/tmp/twinr-camera.png",
                                ]
                                exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(fake_camera.calls), 1)
        self.assertEqual(fake_camera.calls[0]["output_root"], Path("/tmp"))
        self.assertEqual(fake_camera.calls[0]["output_path"], Path("/tmp/twinr-camera.png"))
        self.assertIn("camera_capture=/tmp/twinr-camera.png", stdout.getvalue())
        self.assertIn("camera_input_format=rpicam-still", stdout.getvalue())

    def test_should_enable_respeaker_led_companion_defaults_to_targeted_real_pi_hosts(self) -> None:
        main_mod = importlib.import_module("twinr.__main__")
        fake_respeaker_module = ModuleType("twinr.hardware.respeaker")
        fake_respeaker_module.config_targets_respeaker = lambda *_devices: True
        fake_respeaker_module.probe_respeaker_xvf3800 = lambda: SimpleNamespace(usb_device=None)
        try:
            config = main_mod.TwinrConfig(audio_input_device="hw:CARD=Array,DEV=0")
            with patch.dict(sys.modules, {"twinr.hardware.respeaker": fake_respeaker_module}):
                with patch.object(main_mod, "_is_raspberry_pi_host", return_value=True):
                    self.assertTrue(main_mod._should_enable_respeaker_led_companion(config, "/twinr/.env"))
                with patch.object(main_mod, "_is_raspberry_pi_host", return_value=False):
                    self.assertFalse(main_mod._should_enable_respeaker_led_companion(config, "/twinr/.env"))
                self.assertFalse(main_mod._should_enable_respeaker_led_companion(config, "/home/thh/twinr/.env"))
        finally:
            sys.modules.pop("twinr.__main__", None)

    def test_should_enable_respeaker_led_companion_falls_back_to_usb_probe_for_default_devices(self) -> None:
        main_mod = importlib.import_module("twinr.__main__")
        fake_respeaker_module = ModuleType("twinr.hardware.respeaker")
        fake_respeaker_module.config_targets_respeaker = lambda *_devices: False
        fake_respeaker_module.probe_respeaker_xvf3800 = lambda: SimpleNamespace(usb_device=object())
        try:
            config = main_mod.TwinrConfig(
                audio_input_device="default",
                voice_orchestrator_audio_device="default",
                proactive_audio_input_device="default",
            )
            with patch.dict(sys.modules, {"twinr.hardware.respeaker": fake_respeaker_module}):
                with patch.object(main_mod, "_is_raspberry_pi_host", return_value=True):
                    self.assertTrue(main_mod._should_enable_respeaker_led_companion(config, "/twinr/.env"))
        finally:
            sys.modules.pop("twinr.__main__", None)

    def test_should_enable_respeaker_led_companion_honors_explicit_override(self) -> None:
        main_mod = importlib.import_module("twinr.__main__")
        try:
            enabled = main_mod.TwinrConfig(respeaker_led_enabled=True)
            disabled = main_mod.TwinrConfig(respeaker_led_enabled=False)
            with patch.object(main_mod, "_is_raspberry_pi_host", return_value=False):
                self.assertTrue(main_mod._should_enable_respeaker_led_companion(enabled, "/twinr/.env"))
            with patch.object(main_mod, "_is_raspberry_pi_host", return_value=True):
                self.assertFalse(main_mod._should_enable_respeaker_led_companion(disabled, "/twinr/.env"))
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

    def test_assert_authoritative_repo_runtime_root_rejects_non_repo_cwd(self) -> None:
        main_mod = importlib.import_module("twinr.__main__")
        twinr_package = importlib.import_module("twinr")
        module_file = getattr(main_mod, "__file__", None)
        self.assertIsNotNone(module_file)
        repo_root = Path(str(module_file)).resolve().parents[2]

        with tempfile.TemporaryDirectory() as temp_dir:
            fake_cwd = Path(temp_dir)
            with patch("pathlib.Path.cwd", return_value=fake_cwd):
                with patch.object(twinr_package, "__file__", str(repo_root / "src" / "twinr" / "__init__.py")):
                    with self.assertRaisesRegex(RuntimeError, "must be launched from"):
                        main_mod._assert_authoritative_repo_runtime_root(
                            "/home/thh/twinr/.voice_gateway_alias.env",
                            command_name="run-orchestrator-server",
                        )

        sys.modules.pop("twinr.__main__", None)

    def test_assert_authoritative_repo_runtime_root_rejects_non_repo_source_root(self) -> None:
        main_mod = importlib.import_module("twinr.__main__")
        twinr_package = importlib.import_module("twinr")
        module_file = getattr(main_mod, "__file__", None)
        self.assertIsNotNone(module_file)
        repo_root = Path(str(module_file)).resolve().parents[2]
        try:
            with patch("pathlib.Path.cwd", return_value=repo_root):
                with patch.object(twinr_package, "__file__", "/var/tmp/stale-stage/src/twinr/__init__.py"):
                    with self.assertRaisesRegex(RuntimeError, "must import from"):
                        main_mod._assert_authoritative_repo_runtime_root(
                            "/home/thh/twinr/.voice_gateway_alias.env",
                            command_name="run-orchestrator-server",
                        )
        finally:
            sys.modules.pop("twinr.__main__", None)

    def test_assert_authoritative_repo_runtime_root_accepts_leading_repo(self) -> None:
        main_mod = importlib.import_module("twinr.__main__")
        twinr_package = importlib.import_module("twinr")
        module_file = getattr(main_mod, "__file__", None)
        self.assertIsNotNone(module_file)
        repo_root = Path(str(module_file)).resolve().parents[2]
        try:
            with patch("pathlib.Path.cwd", return_value=repo_root):
                with patch.object(twinr_package, "__file__", str(repo_root / "src" / "twinr" / "__init__.py")):
                    main_mod._assert_authoritative_repo_runtime_root(
                        "/home/thh/twinr/.voice_gateway_alias.env",
                        command_name="run-orchestrator-server",
                    )
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

    def test_run_orchestrator_server_forces_websockets_sansio_backend(self) -> None:
        main_mod = importlib.import_module("twinr.__main__")
        try:
            config = main_mod.TwinrConfig(
                orchestrator_host="127.0.0.1",
                orchestrator_port=8798,
            )
            fake_app = object()
            uvicorn_calls: list[dict[str, object]] = []

            fake_orchestrator = ModuleType("twinr.orchestrator")
            fake_orchestrator.create_app = lambda env_path: fake_app
            fake_uvicorn = ModuleType("uvicorn")
            fake_uvicorn.run = lambda app, **kwargs: uvicorn_calls.append(
                {"app": app, **kwargs}
            )

            with patch.object(main_mod, "_assert_pi_runtime_root", return_value=None):
                with patch.dict(
                    sys.modules,
                    {
                        "twinr.orchestrator": fake_orchestrator,
                        "uvicorn": fake_uvicorn,
                    },
                ):
                    exit_code = main_mod._run_orchestrator_server(
                        config,
                        "/twinr/.env",
                    )
        finally:
            sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(uvicorn_calls), 1)
        self.assertIs(uvicorn_calls[0]["app"], fake_app)
        self.assertEqual(uvicorn_calls[0]["host"], "127.0.0.1")
        self.assertEqual(uvicorn_calls[0]["port"], 8798)
        self.assertEqual(uvicorn_calls[0]["ws"], "websockets-sansio")

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
                        with patch("twinr.ops.runtime_env.prime_user_session_audio_env") as priming:
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
            snapshot_payload = payload["payload"]

        self.assertEqual(exit_code, 1)
        self.assertEqual(snapshot_payload["status"], "waiting")
        self.assertIsNone(snapshot_payload["error_message"])

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

    def test_drone_status_dispatches_before_runtime_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    (
                        f"TWINR_RUNTIME_STATE_PATH={root / 'runtime-state.json'}",
                        "TWINR_DRONE_ENABLED=true",
                        "TWINR_DRONE_BASE_URL=http://127.0.0.1:8791",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            original_argv = list(sys.argv)
            stdout = StringIO()

            try:
                sys.modules.pop("twinr.__main__", None)
                main_mod = importlib.import_module("twinr.__main__")
                with patch.object(
                    main_mod,
                    "_build_runtime",
                    side_effect=AssertionError("drone-status must not bootstrap the runtime"),
                ):
                    with patch.object(main_mod, "_run_drone_cli_commands", return_value=0) as fake_drone:
                        with patch("sys.stdout", stdout):
                            sys.argv = [
                                "twinr",
                                "--env-file",
                                str(env_path),
                                "--drone-status",
                            ]
                            exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        fake_drone.assert_called_once()

    def test_drone_hover_test_dispatches_before_runtime_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    (
                        f"TWINR_RUNTIME_STATE_PATH={root / 'runtime-state.json'}",
                        "TWINR_DRONE_ENABLED=true",
                        "TWINR_DRONE_BASE_URL=http://127.0.0.1:8791",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                main_mod = importlib.import_module("twinr.__main__")
                with patch.object(
                    main_mod,
                    "_build_runtime",
                    side_effect=AssertionError("drone-hover-test must not bootstrap the runtime"),
                ):
                    with patch.object(main_mod, "_run_drone_cli_commands", return_value=0) as fake_drone:
                        sys.argv = [
                            "twinr",
                            "--env-file",
                            str(env_path),
                            "--drone-hover-test",
                        ]
                        exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        fake_drone.assert_called_once()

    def test_self_test_dispatches_before_runtime_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                f"TWINR_RUNTIME_STATE_PATH={root / 'runtime-state.json'}\n",
                encoding="utf-8",
            )
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                main_mod = importlib.import_module("twinr.__main__")
                with patch.object(
                    main_mod,
                    "_build_runtime",
                    side_effect=AssertionError("self-test must not bootstrap the runtime"),
                ):
                    with patch.object(main_mod, "_run_self_test_command", return_value=0) as fake_self_test:
                        sys.argv = [
                            "twinr",
                            "--env-file",
                            str(env_path),
                            "--self-test",
                            "drone_stack",
                        ]
                        exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        fake_self_test.assert_called_once()

    def test_run_whatsapp_channel_dispatches_to_channel_loop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    (
                        f"TWINR_RUNTIME_STATE_PATH={root / 'runtime-state.json'}",
                        f"TWINR_WHATSAPP_ALLOW_FROM={_TEST_WHATSAPP_ALLOW_FROM_DISPLAY}",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            fake_runtime = SimpleNamespace(status=SimpleNamespace(value="waiting"))
            fake_backend = object()
            fake_tool_agent = object()
            fake_print_backend = object()
            built_runtime_config = None
            lock_runtime_state_path = None
            fake_whatsapp_module = ModuleType("twinr.channels.whatsapp")
            fake_whatsapp_module.TwinrWhatsAppChannelLoop = _FakeWhatsAppLoop
            fake_providers_module = ModuleType("twinr.providers")
            fake_providers_module.build_streaming_provider_bundle = lambda *args, **kwargs: SimpleNamespace(
                tool_agent=fake_tool_agent,
                print_backend=fake_print_backend,
            )
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(
                    sys.modules,
                    {
                        "twinr.channels.whatsapp": fake_whatsapp_module,
                        "twinr.providers": fake_providers_module,
                    },
                ):
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
        self.assertTrue(built_runtime_config.restore_runtime_state_on_startup)
        self.assertEqual(lock_runtime_state_path, str(root / "runtime-state.json"))
        self.assertIsNotNone(_FakeWhatsAppLoop.last_instance)
        self.assertIs(_FakeWhatsAppLoop.last_instance.tool_agent_provider, fake_tool_agent)
        self.assertIs(_FakeWhatsAppLoop.last_instance.print_backend, fake_print_backend)

    def test_openai_prompt_uses_scoped_runtime_snapshot_without_restore(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    (
                        f"TWINR_RUNTIME_STATE_PATH={root / 'runtime-state.json'}",
                        "TWINR_OPENAI_API_KEY=sk-test",
                        "TWINR_RESTORE_RUNTIME_STATE_ON_STARTUP=true",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            backend_calls: list[dict[str, object]] = []
            fake_backend = SimpleNamespace(
                respond_with_metadata=lambda prompt, **kwargs: (
                    backend_calls.append({"prompt": prompt, **kwargs}),
                    SimpleNamespace(
                        text="ok",
                        response_id=None,
                        request_id=None,
                        used_web_search=False,
                    ),
                )[1]
            )
            original_argv = list(sys.argv)
            stdout = StringIO()

            try:
                sys.modules.pop("twinr.__main__", None)
                main_mod = importlib.import_module("twinr.__main__")
                with patch.object(
                    main_mod,
                    "_build_runtime",
                    side_effect=AssertionError("openai-prompt must not bootstrap the runtime"),
                ):
                    with patch("twinr.providers.openai.OpenAIBackend", return_value=fake_backend):
                        with patch("sys.stdout", stdout):
                            sys.argv = [
                                "twinr",
                                "--env-file",
                                str(env_path),
                                "--openai-prompt",
                                "Sag nur: ok.",
                            ]
                            exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(backend_calls), 1)
        self.assertEqual(backend_calls[0]["prompt"], "Sag nur: ok.")
        self.assertEqual(backend_calls[0]["conversation"], ())
        self.assertIn("status=waiting", stdout.getvalue())
        self.assertIn("response=ok", stdout.getvalue())

    def test_vision_prompt_uses_scoped_runtime_snapshot_without_restore(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    (
                        f"TWINR_RUNTIME_STATE_PATH={root / 'runtime-state.json'}",
                        "TWINR_OPENAI_API_KEY=sk-test",
                        "TWINR_RESTORE_RUNTIME_STATE_ON_STARTUP=true",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            backend_calls: list[dict[str, object]] = []
            fake_backend = SimpleNamespace(
                respond_to_images_with_metadata=lambda prompt, **kwargs: (
                    backend_calls.append({"prompt": prompt, **kwargs}),
                    SimpleNamespace(
                        text="rot",
                        response_id=None,
                        request_id=None,
                        used_web_search=False,
                    ),
                )[1]
            )
            fake_openai_module = ModuleType("twinr.providers.openai")
            fake_openai_module.OpenAIBackend = lambda config: fake_backend
            _attach_fake_openai_image_input(fake_openai_module)
            original_argv = list(sys.argv)
            stdout = StringIO()

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(sys.modules, {"twinr.providers.openai": fake_openai_module}):
                    main_mod = importlib.import_module("twinr.__main__")
                    with patch.object(
                        main_mod,
                        "_build_runtime",
                        side_effect=AssertionError("vision-prompt must not bootstrap the runtime"),
                    ):
                        with patch("sys.stdout", stdout):
                            sys.argv = [
                                "twinr",
                                "--env-file",
                                str(env_path),
                                "--vision-image",
                                "/tmp/reference.png",
                                "--vision-prompt",
                                "Nenne die dominante Farbe in genau einem Wort.",
                            ]
                            exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(backend_calls), 1)
        self.assertEqual(backend_calls[0]["prompt"], "Nenne die dominante Farbe in genau einem Wort.")
        self.assertEqual(len(backend_calls[0]["images"]), 1)
        self.assertEqual(backend_calls[0]["conversation"], ())
        self.assertIn("status=waiting", stdout.getvalue())
        self.assertIn("vision_image_count=1", stdout.getvalue())
        self.assertIn("response=rot", stdout.getvalue())

    def test_orchestrator_probe_turn_dispatches_to_lightweight_probe_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    (
                        f"TWINR_RUNTIME_STATE_PATH={root / 'runtime-state.json'}",
                        "TWINR_OPENAI_API_KEY=sk-test",
                        "TWINR_ORCHESTRATOR_WS_URL=ws://127.0.0.1:8797/ws/orchestrator",
                        "TWINR_ORCHESTRATOR_SHARED_SECRET=test-secret",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            fake_runtime = SimpleNamespace(status=SimpleNamespace(value="waiting"))
            fake_backend = object()
            probe_calls: list[dict[str, object]] = []
            fake_probe_module = ModuleType("twinr.orchestrator.probe_turn")

            def _run_probe_turn(*, config, runtime, backend, prompt, emit_line):
                probe_calls.append(
                    {
                        "config": config,
                        "runtime": runtime,
                        "backend": backend,
                        "prompt": prompt,
                    }
                )
                emit_line("probe_tool_handler_count=7")
                return SimpleNamespace(
                    deltas=("Teil ", "Antwort"),
                    result=SimpleNamespace(
                        text="Teil Antwort",
                        rounds=2,
                        used_web_search=True,
                        response_id="resp-1",
                        request_id="req-1",
                        model="gpt-5.4-mini",
                    ),
                    tool_handler_count=7,
                )

            fake_probe_module.run_orchestrator_probe_turn = _run_probe_turn
            fake_streaming_module = ModuleType("twinr.agent.workflows.streaming_runner")
            fake_streaming_module.TwinrStreamingHardwareLoop = lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("orchestrator-probe-turn must not bootstrap the streaming hardware loop")
            )
            original_argv = list(sys.argv)
            stdout = StringIO()

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(
                    sys.modules,
                    {
                        "twinr.orchestrator.probe_turn": fake_probe_module,
                        "twinr.agent.workflows.streaming_runner": fake_streaming_module,
                    },
                ):
                    main_mod = importlib.import_module("twinr.__main__")
                    with patch.object(main_mod, "_build_runtime", return_value=fake_runtime):
                        with patch("twinr.providers.openai.OpenAIBackend", return_value=fake_backend):
                            with patch("sys.stdout", stdout):
                                sys.argv = [
                                    "twinr",
                                    "--env-file",
                                    str(env_path),
                                    "--orchestrator-probe-turn",
                                    "Wie gehts dir denn",
                                ]
                                exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(probe_calls), 1)
        self.assertIs(probe_calls[0]["runtime"], fake_runtime)
        self.assertIs(probe_calls[0]["backend"], fake_backend)
        self.assertEqual(probe_calls[0]["prompt"], "Wie gehts dir denn")
        self.assertIn("probe_tool_handler_count=7", stdout.getvalue())
        self.assertIn("streamed=Teil Antwort", stdout.getvalue())
        self.assertIn("response=Teil Antwort", stdout.getvalue())
        self.assertIn("used_web_search=true", stdout.getvalue())

    def test_run_whatsapp_channel_ensures_remote_watchdog_before_runtime_boot_on_pi(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    (
                        f"TWINR_RUNTIME_STATE_PATH={root / 'runtime-state.json'}",
                        "TWINR_OPENAI_API_KEY=sk-test",
                        f"TWINR_WHATSAPP_ALLOW_FROM={_TEST_WHATSAPP_ALLOW_FROM_DISPLAY}",
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
            fake_tool_agent = object()
            fake_print_backend = object()
            fake_whatsapp_module = ModuleType("twinr.channels.whatsapp")
            fake_whatsapp_module.TwinrWhatsAppChannelLoop = _FakeWhatsAppLoop
            fake_providers_module = ModuleType("twinr.providers")
            fake_providers_module.build_streaming_provider_bundle = lambda *args, **kwargs: SimpleNamespace(
                tool_agent=fake_tool_agent,
                print_backend=fake_print_backend,
            )
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
                        "twinr.providers": fake_providers_module,
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
            _attach_fake_openai_image_input(fake_openai_module)
            fake_providers_module = ModuleType("twinr.providers")
            fake_providers_module.build_streaming_provider_bundle = lambda *args, **kwargs: SimpleNamespace(
                print_backend=SimpleNamespace(),
                stt=SimpleNamespace(),
                agent=SimpleNamespace(),
                tts=SimpleNamespace(),
                tool_agent=SimpleNamespace(),
            )
            fake_companion_module = ModuleType("twinr.display.companion")
            fake_respeaker_led_module = ModuleType("twinr.hardware.respeaker.companion")

            @contextmanager
            def _fake_companion(_config, *, enabled: bool):
                yield

            fake_companion_module.optional_display_companion = _fake_companion

            @contextmanager
            def _fake_led_companion(_config, *, enabled: bool):
                yield

            fake_respeaker_led_module.optional_respeaker_led_companion = _fake_led_companion
            fake_watchdog_module = ModuleType("twinr.ops.remote_memory_watchdog_companion")

            def _ensure_remote_memory_watchdog_process(_config, *, env_file):
                watchdog_calls.append(str(env_file))
                return 4321

            fake_watchdog_module.ensure_remote_memory_watchdog_process = _ensure_remote_memory_watchdog_process
            fake_ops_module = ModuleType("twinr.ops")
            fake_ops_module.loop_instance_lock = _fake_lock
            fake_runtime_env_module = _fake_runtime_env_module()
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
                        "twinr.hardware.respeaker.companion": fake_respeaker_led_module,
                        "twinr.ops": fake_ops_module,
                        "twinr.ops.runtime_env": fake_runtime_env_module,
                        "twinr.ops.remote_memory_watchdog_companion": fake_watchdog_module,
                    },
                ):
                    main_mod = importlib.import_module("twinr.__main__")
                    with patch.object(main_mod, "_should_enable_display_companion", return_value=False):
                        with patch.object(main_mod, "_should_enable_respeaker_led_companion", return_value=False):
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
            _attach_fake_openai_image_input(fake_openai_module)
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
            fake_respeaker_led_module = ModuleType("twinr.hardware.respeaker.companion")

            @contextmanager
            def _fake_companion(_config, *, enabled: bool):
                events.append(f"display_companion_enter:{str(enabled).lower()}")
                try:
                    yield
                finally:
                    events.append("display_companion_exit")

            fake_companion_module.optional_display_companion = _fake_companion

            @contextmanager
            def _fake_led_companion(_config, *, enabled: bool):
                events.append(f"led_companion_enter:{str(enabled).lower()}")
                try:
                    yield
                finally:
                    events.append("led_companion_exit")

            fake_respeaker_led_module.optional_respeaker_led_companion = _fake_led_companion
            fake_ops_module = ModuleType("twinr.ops")
            fake_runtime_env_module = _fake_runtime_env_module()

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
                        "twinr.hardware.respeaker.companion": fake_respeaker_led_module,
                        "twinr.ops": fake_ops_module,
                        "twinr.ops.runtime_env": fake_runtime_env_module,
                    },
                ):
                    main_mod = importlib.import_module("twinr.__main__")
                    with patch.object(main_mod, "_should_enable_display_companion", return_value=True):
                        with patch.object(main_mod, "_should_enable_respeaker_led_companion", return_value=True):
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
            events[:7],
            [
                "lock_enter",
                "display_companion_enter:true",
                "led_companion_enter:true",
                "runtime_init",
                "backend_init",
                "bundle_init",
                "loop_init",
            ],
        )
        self.assertEqual(events[-3:], ["led_companion_exit", "display_companion_exit", "lock_exit"])

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
            _attach_fake_openai_image_input(fake_openai_module)
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
            fake_respeaker_led_module = ModuleType("twinr.hardware.respeaker.companion")

            @contextmanager
            def _fake_companion(_config, *, enabled: bool):
                events.append(f"display_companion_enter:{str(enabled).lower()}")
                try:
                    yield
                finally:
                    events.append("display_companion_exit")

            fake_companion_module.optional_display_companion = _fake_companion

            @contextmanager
            def _fake_led_companion(_config, *, enabled: bool):
                events.append(f"led_companion_enter:{str(enabled).lower()}")
                try:
                    yield
                finally:
                    events.append("led_companion_exit")

            fake_respeaker_led_module.optional_respeaker_led_companion = _fake_led_companion
            fake_ops_module = ModuleType("twinr.ops")
            fake_ops_module.loop_instance_lock = _fake_lock
            fake_runtime_env_module = _fake_runtime_env_module()
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
                        "twinr.hardware.respeaker.companion": fake_respeaker_led_module,
                        "twinr.ops": fake_ops_module,
                        "twinr.ops.runtime_env": fake_runtime_env_module,
                    },
                ):
                    main_mod = importlib.import_module("twinr.__main__")
                    with patch.object(main_mod, "_should_enable_display_companion", return_value=True):
                        with patch.object(main_mod, "_should_enable_respeaker_led_companion", return_value=True):
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
        self.assertEqual(events[-2:], ["led_companion_exit", "display_companion_exit"])

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
            fake_runtime_env_module = _fake_runtime_env_module()
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(
                    sys.modules,
                    {
                        "twinr.ops": fake_ops_module,
                        "twinr.ops.runtime_env": fake_runtime_env_module,
                    },
                ):
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

    def test_watch_remote_memory_returns_special_exit_code_when_owner_already_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                f"TWINR_RUNTIME_STATE_PATH={root / 'runtime-state.json'}\n",
                encoding="utf-8",
            )
            fake_watchdog = _FakeRemoteMemoryWatchdog()
            fake_ops_module = ModuleType("twinr.ops")
            fake_ops_module.RemoteMemoryWatchdog = SimpleNamespace(from_config=lambda _config: fake_watchdog)
            fake_runtime_env_module = _fake_runtime_env_module()
            original_argv = list(sys.argv)

            @contextmanager
            def _contention_lock(_config, _name: str):
                raise TwinrInstanceAlreadyRunningError(label="remote memory watchdog", owner_pid=4242)
                yield

            fake_ops_module.loop_instance_lock = _contention_lock

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(
                    sys.modules,
                    {
                        "twinr.ops": fake_ops_module,
                        "twinr.ops.runtime_env": fake_runtime_env_module,
                    },
                ):
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

        self.assertEqual(exit_code, 75)
        self.assertIsNone(fake_watchdog.duration_s)

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
            fake_runtime_env_module = _fake_runtime_env_module()
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(
                    sys.modules,
                    {
                        "twinr.ops": fake_ops_module,
                        "twinr.ops.runtime_env": fake_runtime_env_module,
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

    def test_non_voice_e2e_acceptance_dispatches_without_runtime_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("", encoding="utf-8")
            calls: list[dict[str, object]] = []
            fake_acceptance_module = ModuleType("twinr.orchestrator.non_voice_acceptance")

            def _run_non_voice_acceptance(**kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    run_id="non_voice_123",
                    status="ok",
                    ready=True,
                    direct_case=SimpleNamespace(status="ok", used_web_search=False, error_message=None),
                    tool_case=SimpleNamespace(status="ok", used_web_search=True, error_message=None),
                    memory_result=SimpleNamespace(status="ok", ready=True, error_message=None),
                    artifact_path=str(root / "artifacts" / "stores" / "ops" / "non_voice_e2e_acceptance.json"),
                    report_path=str(root / "artifacts" / "reports" / "non_voice_e2e_acceptance" / "non_voice_123.json"),
                )

            fake_acceptance_module.run_non_voice_acceptance = _run_non_voice_acceptance
            original_argv = list(sys.argv)

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(
                    sys.modules,
                    {"twinr.orchestrator.non_voice_acceptance": fake_acceptance_module},
                ):
                    main_mod = importlib.import_module("twinr.__main__")
                    with patch.object(main_mod, "_build_runtime", side_effect=AssertionError("runtime must not be created")):
                        sys.argv = [
                            "twinr",
                            "--env-file",
                            str(env_path),
                            "--non-voice-e2e-acceptance",
                        ]
                        exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["env_path"], str(env_path))

    def test_runtime_init_failure_returns_error_without_unbound_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("", encoding="utf-8")
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

    def test_proactive_audio_observe_once_prints_runtime_faithful_perception_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                f"TWINR_RUNTIME_STATE_PATH={root / 'runtime-state.json'}\n",
                encoding="utf-8",
            )
            fake_audio_perception_module = ModuleType("twinr.proactive.runtime.audio_perception")
            calls: list[object] = []
            fake_snapshot = SimpleNamespace(name="snapshot")

            def _observe(config):
                calls.append(config)
                return fake_snapshot

            def _render(snapshot):
                self.assertIs(snapshot, fake_snapshot)
                return (
                    "proactive_audio_room_context=speech",
                    "proactive_device_directed_speech_candidate=true",
                    "proactive_background_media_likely=false",
                )

            fake_audio_perception_module.observe_audio_perception_once = _observe
            fake_audio_perception_module.render_audio_perception_snapshot_lines = _render
            original_argv = list(sys.argv)
            stdout = StringIO()

            try:
                sys.modules.pop("twinr.__main__", None)
                with patch.dict(
                    sys.modules,
                    {"twinr.proactive.runtime.audio_perception": fake_audio_perception_module},
                ):
                    main_mod = importlib.import_module("twinr.__main__")
                    with patch("sys.stdout", stdout):
                        sys.argv = [
                            "twinr",
                            "--env-file",
                            str(env_path),
                            "--proactive-audio-observe-once",
                        ]
                        exit_code = main_mod.main()
            finally:
                sys.argv = original_argv
                sys.modules.pop("twinr.__main__", None)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(calls), 1)
        output_lines = stdout.getvalue().splitlines()
        self.assertIn("proactive_audio_room_context=speech", output_lines)
        self.assertIn("proactive_device_directed_speech_candidate=true", output_lines)
        self.assertIn("proactive_background_media_likely=false", output_lines)


if __name__ == "__main__":
    unittest.main()
