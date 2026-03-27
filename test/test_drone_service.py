"""Regression coverage for Twinr's bounded drone-daemon slice."""

from __future__ import annotations

from http.server import ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.drone_service import DroneServiceConfig, RemoteDroneServiceClient

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "hardware" / "ops" / "drone_daemon.py"
_SPEC = importlib.util.spec_from_file_location("twinr_drone_daemon_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class _HealthyRadioProvider:
    workspace = Path("/tmp")
    bitcraze_python = Path("/tmp/python")

    def snapshot(self) -> dict[str, object]:
        return {"radio_ready": True, "radio_version": "3.2", "error": None}


class DroneServiceTests(unittest.TestCase):
    def test_normalize_executable_path_preserves_virtualenv_launcher_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bin_dir = root / "bitcraze-venv" / "bin"
            bin_dir.mkdir(parents=True)
            launcher_target = bin_dir / "python3"
            launcher_target.symlink_to("/usr/bin/python3")
            launcher = bin_dir / "python"
            launcher.symlink_to("python3")

            normalized = _MODULE._normalize_executable_path(launcher)
            self.assertEqual(normalized, launcher)
            self.assertEqual(normalized.name, "python")
            self.assertNotEqual(normalized, launcher.resolve(strict=False))

    def test_drone_service_config_reads_new_env_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    (
                        "TWINR_DRONE_ENABLED=true",
                        "TWINR_DRONE_BASE_URL=http://127.0.0.1:8791/",
                        "TWINR_DRONE_REQUIRE_MANUAL_ARM=false",
                        "TWINR_DRONE_MISSION_TIMEOUT_S=61",
                        "TWINR_DRONE_REQUEST_TIMEOUT_S=3.5",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            config = TwinrConfig.from_env(env_path)
            drone = DroneServiceConfig.from_config(config)

        self.assertTrue(drone.enabled)
        self.assertEqual(drone.base_url, "http://127.0.0.1:8791")
        self.assertFalse(drone.require_manual_arm)
        self.assertEqual(drone.mission_timeout_s, 61.0)
        self.assertEqual(drone.request_timeout_s, 3.5)

    def test_remote_client_reads_state_and_pending_manual_arm_mission(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = _MODULE.DroneDaemonService(
                repo_root=Path(temp_dir),
                env_file=None,
                pose_provider=_MODULE.StubPoseProvider(healthy=True),
                radio_provider=_HealthyRadioProvider(),
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), _MODULE.build_handler(service))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                client = RemoteDroneServiceClient(base_url=base_url, timeout_s=1.0)
                state = client.state()
                mission = client.create_inspect_mission(target_hint="self test")
                cancelled = client.cancel_mission(mission.mission_id)
            finally:
                server.shutdown()
                thread.join(timeout=3.0)
                server.server_close()

        self.assertEqual(state.service_status, "ready")
        self.assertEqual(state.skill_layer_mode, "stationary_observe_only")
        self.assertTrue(state.safety.can_arm)
        self.assertEqual(mission.state, "pending_manual_arm")
        self.assertEqual(cancelled.state, "cancelled")

    def test_manual_arm_runs_stubbed_stationary_inspection_to_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                f"TWINR_RUNTIME_STATE_PATH={root / 'runtime-state.json'}\n",
                encoding="utf-8",
            )
            service = _MODULE.DroneDaemonService(
                repo_root=Path(__file__).resolve().parents[1],
                env_file=env_path,
                pose_provider=_MODULE.StubPoseProvider(healthy=True),
                radio_provider=_HealthyRadioProvider(),
            )
            service._capture_stationary_inspection = lambda mission: "artifact.json"  # type: ignore[attr-defined]
            server = ThreadingHTTPServer(("127.0.0.1", 0), _MODULE.build_handler(service))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                client = RemoteDroneServiceClient(base_url=base_url, timeout_s=1.0)
                mission = client.create_inspect_mission(target_hint="shelf 3")
                armed = client.manual_arm(mission.mission_id)
                for _ in range(40):
                    current = client.mission_status(mission.mission_id)
                    if current.state == "completed":
                        break
                    time.sleep(0.05)
                else:
                    self.fail("mission did not complete within the bounded wait window")
            finally:
                server.shutdown()
                thread.join(timeout=3.0)
                server.server_close()

        self.assertEqual(mission.state, "pending_manual_arm")
        self.assertIn(armed.state, {"armed", "running"})
        self.assertEqual(current.state, "completed")
        self.assertEqual(current.artifact_name, "artifact.json")

    def test_hover_test_requires_explicit_hover_skill_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = _MODULE.DroneDaemonService(
                repo_root=Path(temp_dir),
                env_file=None,
                pose_provider=_MODULE.StubPoseProvider(healthy=True),
                radio_provider=_HealthyRadioProvider(),
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), _MODULE.build_handler(service))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                client = RemoteDroneServiceClient(base_url=base_url, timeout_s=1.0)
                with self.assertRaisesRegex(RuntimeError, "hover_test missions require"):
                    client.create_hover_test_mission()
            finally:
                server.shutdown()
                thread.join(timeout=3.0)
                server.server_close()

    def test_manual_arm_runs_stubbed_hover_test_to_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                f"TWINR_RUNTIME_STATE_PATH={root / 'runtime-state.json'}\n",
                encoding="utf-8",
            )
            service = _MODULE.DroneDaemonService(
                repo_root=Path(__file__).resolve().parents[1],
                env_file=env_path,
                pose_provider=_MODULE.StubPoseProvider(healthy=True),
                radio_provider=_HealthyRadioProvider(),
                skill_layer_mode="bounded_hover_test_only",
            )
            service._run_hover_test_mission = lambda mission: "hover.json"  # type: ignore[attr-defined]
            server = ThreadingHTTPServer(("127.0.0.1", 0), _MODULE.build_handler(service))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                client = RemoteDroneServiceClient(base_url=base_url, timeout_s=1.0)
                mission = client.create_hover_test_mission(max_duration_s=20.0)
                armed = client.manual_arm(mission.mission_id)
                for _ in range(40):
                    current = client.mission_status(mission.mission_id)
                    if current.state == "completed":
                        break
                    time.sleep(0.05)
                else:
                    self.fail("hover mission did not complete within the bounded wait window")
            finally:
                server.shutdown()
                thread.join(timeout=3.0)
                server.server_close()

        self.assertEqual(mission.mission_type, "hover_test")
        self.assertEqual(mission.state, "pending_manual_arm")
        self.assertIn(armed.state, {"armed", "running"})
        self.assertEqual(current.state, "completed")
        self.assertEqual(current.artifact_name, "hover.json")

    def test_hover_worker_timeout_persists_partial_artifact_with_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script_dir = root / "hardware" / "bitcraze"
            script_dir.mkdir(parents=True)
            (script_dir / "run_hover_test.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            bitcraze_python = root / "bitcraze-python"
            bitcraze_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            workspace = root / "bitcraze-workspace"
            workspace.mkdir()

            class _LocalRadioProvider:
                def __init__(self) -> None:
                    self.workspace = workspace
                    self.bitcraze_python = bitcraze_python

                def snapshot(self) -> dict[str, object]:
                    return {"radio_ready": True, "radio_version": "3.2", "error": None}

            service = _MODULE.DroneDaemonService(
                repo_root=root,
                env_file=None,
                pose_provider=_MODULE.StubPoseProvider(healthy=True),
                radio_provider=_LocalRadioProvider(),
                skill_layer_mode="bounded_hover_test_only",
            )
            mission = _MODULE.MissionRecord(
                mission_id="DRN-TIMEOUT",
                mission_type="hover_test",
                target_hint="bounded hover test",
                capture_intent="hover_test",
                max_duration_s=5.0,
                return_policy="return_and_land",
                requires_manual_arm=True,
                state="running",
                summary="running",
                created_at="2026-03-26T00:00:00+00:00",
                updated_at="2026-03-26T00:00:00+00:00",
            )

            class _EmptyStream:
                def read(self, _size: int = -1) -> str:
                    return ""

                def close(self) -> None:
                    return None

            class _FakeProcess:
                def __init__(self, command, **_kwargs) -> None:
                    self.command = command
                    self.pid = 4242
                    self.returncode: int | None = None
                    self.stdout = _EmptyStream()
                    self.stderr = _EmptyStream()
                    trace_path = Path(command[command.index("--trace-file") + 1])
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    trace_path.write_text(
                        "\n".join(
                            (
                                json.dumps({"phase": "takeoff", "status": "done"}),
                                json.dumps({"phase": "telemetry_stop", "status": "begin"}),
                            )
                        )
                        + "\n",
                        encoding="utf-8",
                    )

                def wait(self, timeout=None):
                    if self.returncode is not None:
                        return self.returncode
                    raise _MODULE.subprocess.TimeoutExpired(cmd=self.command, timeout=timeout)

                def poll(self):
                    return self.returncode

                def terminate(self) -> None:
                    self.returncode = 130

                def kill(self) -> None:
                    self.returncode = -9

            monotonic_values = iter((0.0, 6.0))
            original_popen = _MODULE.subprocess.Popen
            original_interrupt = service._interrupt_hover_test_process
            original_monotonic = _MODULE.time.monotonic
            try:
                _MODULE.subprocess.Popen = _FakeProcess
                service._interrupt_hover_test_process = staticmethod(  # type: ignore[method-assign]
                    lambda process: setattr(process, "returncode", 130)
                )
                _MODULE.time.monotonic = lambda: next(monotonic_values, 6.0)
                with self.assertRaises(_MODULE.MissionCancelled) as exc_context:
                    service._run_hover_test_worker(mission, stamp="20260326T000000000000Z")
            finally:
                _MODULE.subprocess.Popen = original_popen
                service._interrupt_hover_test_process = original_interrupt  # type: ignore[method-assign]
                _MODULE.time.monotonic = original_monotonic

            artifact_name = exc_context.exception.artifact_name
            self.assertIsNotNone(artifact_name)
            artifact_path = service._artifact_root / str(artifact_name)
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["partial"])
            self.assertEqual(payload["summary"], "Hover test exceeded its bounded runtime. Landing requested.")
            diagnostics = payload["hover_worker_diagnostics"]
            self.assertEqual(diagnostics["last_trace_phase"], "telemetry_stop")
            self.assertEqual(diagnostics["last_trace_status"], "begin")
            self.assertEqual(diagnostics["return_code"], 130)

    def test_hover_worker_drains_stdout_before_wait_deadlock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            script_dir = root / "hardware" / "bitcraze"
            script_dir.mkdir(parents=True)
            (script_dir / "run_hover_test.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            bitcraze_python = root / "bitcraze-python"
            bitcraze_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            workspace = root / "bitcraze-workspace"
            workspace.mkdir()

            class _LocalRadioProvider:
                def __init__(self) -> None:
                    self.workspace = workspace
                    self.bitcraze_python = bitcraze_python

                def snapshot(self) -> dict[str, object]:
                    return {"radio_ready": True, "radio_version": "3.2", "error": None}

            service = _MODULE.DroneDaemonService(
                repo_root=root,
                env_file=None,
                pose_provider=_MODULE.StubPoseProvider(healthy=True),
                radio_provider=_LocalRadioProvider(),
                skill_layer_mode="bounded_hover_test_only",
            )
            mission = _MODULE.MissionRecord(
                mission_id="DRN-DRAIN",
                mission_type="hover_test",
                target_hint="bounded hover test",
                capture_intent="hover_test",
                max_duration_s=5.0,
                return_policy="return_and_land",
                requires_manual_arm=True,
                state="running",
                summary="running",
                created_at="2026-03-26T00:00:00+00:00",
                updated_at="2026-03-26T00:00:00+00:00",
            )

            class _FakeStream:
                def __init__(self, process: "_FakeProcess", payload: str) -> None:
                    self._process = process
                    self._remaining_chunks: list[str] = [payload]

                def read(self, _size: int = -1) -> str:
                    if not self._remaining_chunks:
                        return ""
                    self._process.stdout_drained.set()
                    self._process.returncode = 0
                    return self._remaining_chunks.pop()

                def close(self) -> None:
                    return None

            class _FakeProcess:
                def __init__(self, command, **_kwargs) -> None:
                    self.command = command
                    self.pid = 4343
                    self.returncode: int | None = None
                    self.stdout_drained = threading.Event()
                    payload = json.dumps(
                        {
                            "report": {
                                "status": "completed",
                                "failures": [],
                                "deck_flags": {
                                    "bcAI": True,
                                    "bcFlow2": True,
                                    "bcMultiranger": True,
                                    "bcZRanger2": True,
                                },
                            },
                            "failures": [],
                        }
                    )
                    self.stdout = _FakeStream(self, payload)
                    self.stderr = _FakeStream(self, "")

                def wait(self, timeout=None):
                    if self.returncode is not None:
                        return self.returncode
                    if self.stdout_drained.wait(timeout):
                        return int(self.returncode or 0)
                    raise _MODULE.subprocess.TimeoutExpired(cmd=self.command, timeout=timeout)

                def poll(self):
                    return self.returncode

                def terminate(self) -> None:
                    self.returncode = 0
                    self.stdout_drained.set()

                def kill(self) -> None:
                    self.returncode = -9
                    self.stdout_drained.set()

            original_popen = _MODULE.subprocess.Popen
            try:
                _MODULE.subprocess.Popen = _FakeProcess
                result = service._run_hover_test_worker(mission, stamp="20260326T000000000000Z")
            finally:
                _MODULE.subprocess.Popen = original_popen

            self.assertEqual(result.return_code, 0)
            self.assertEqual(result.report["status"], "completed")
            self.assertEqual(json.loads(result.stdout)["report"]["status"], "completed")

    def test_capture_stationary_inspection_persists_image_without_absolute_camera_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = _MODULE.DroneDaemonService(
                repo_root=root,
                env_file=None,
                pose_provider=_MODULE.StubPoseProvider(healthy=True),
                radio_provider=_HealthyRadioProvider(),
            )
            mission = _MODULE.MissionRecord(
                mission_id="DRN-TEST",
                mission_type="inspect",
                target_hint="desk",
                capture_intent="scene",
                max_duration_s=45.0,
                return_policy="return_and_land",
                requires_manual_arm=True,
                state="running",
                summary="running",
                created_at="2026-03-26T00:00:00+00:00",
                updated_at="2026-03-26T00:00:00+00:00",
            )

            class _FakeCamera:
                def capture_photo(self, *, output_path=None, filename="camera-capture.png"):
                    if output_path is not None:
                        raise AssertionError("expected daemon to persist the image itself")
                    return SimpleNamespace(
                        data=b"\x89PNG\r\n\x1a\nfake",
                        source_device="aideck://192.168.4.1:5000",
                        input_format="aideck-cpx-raw-bayer",
                        content_type="image/png",
                        filename=filename,
                    )

            with patch("twinr.hardware.camera.V4L2StillCamera.from_config", return_value=_FakeCamera()):
                artifact_name = service._capture_stationary_inspection(mission)

            report_path = service._artifact_root / artifact_name
            self.assertTrue(report_path.exists())
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            image_name = str(payload["camera"]["image_file"])
            image_path = service._artifact_root / image_name
            self.assertTrue(image_path.exists())
            self.assertEqual(image_path.read_bytes(), b"\x89PNG\r\n\x1a\nfake")
            self.assertEqual(payload["camera"]["source_device"], "aideck://192.168.4.1:5000")


if __name__ == "__main__":
    unittest.main()
