"""Regression coverage for Twinr's bounded drone-daemon slice."""

from __future__ import annotations

from http.server import ThreadingHTTPServer
import importlib.util
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

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


if __name__ == "__main__":
    unittest.main()
