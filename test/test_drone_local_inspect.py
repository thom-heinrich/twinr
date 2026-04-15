"""Regression coverage for the bounded local inspect mission slice."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.drone_service import RemoteDroneServiceClient


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "hardware" / "ops" / "drone_daemon.py"
_SPEC = importlib.util.spec_from_file_location("twinr_drone_daemon_local_inspect_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE: Any = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class _HealthyRadioProvider:
    workspace = Path("/tmp")
    bitcraze_python = Path("/tmp/python")

    def snapshot(self) -> dict[str, object]:
        return {"radio_ready": True, "radio_version": "3.2", "error": None}


class LocalInspectMissionTests(unittest.TestCase):
    def test_client_helper_emits_local_inspect_mission_payload(self) -> None:
        client = RemoteDroneServiceClient(
            base_url="http://127.0.0.1:8791",
            timeout_s=1.0,
        )
        captured_request: dict[str, object] = {}

        def _fake_request_json(
            method: str,
            route: str,
            *,
            payload: dict[str, object] | None = None,
            query: dict[str, object] | None = None,
            extra_headers: dict[str, str] | None = None,
        ) -> dict[str, object]:
            captured_request["method"] = method
            captured_request["route"] = route
            captured_request["payload"] = payload or {}
            captured_request["query"] = query or {}
            captured_request["extra_headers"] = extra_headers or {}
            return {
                "mission": {
                    "mission_id": "DRN-LOCAL-1",
                    "mission_type": "inspect_local_zone",
                    "state": "pending_manual_arm",
                    "summary": "Mission queued and waiting for local manual arm approval.",
                    "target_hint": "shelf edge",
                    "capture_intent": "object_check",
                    "max_duration_s": 25.0,
                    "return_policy": "return_and_land",
                    "requires_manual_arm": True,
                    "created_at": "2026-04-11T00:00:00+00:00",
                    "updated_at": "2026-04-11T00:00:00+00:00",
                }
            }

        client._request_json = _fake_request_json  # type: ignore[assignment,method-assign]

        mission = client.create_inspect_local_zone_mission(
            target_hint="shelf edge",
            capture_intent="object_check",
            max_duration_s=25.0,
        )

        self.assertEqual(mission.mission_type, "inspect_local_zone")
        self.assertEqual(mission.capture_intent, "object_check")
        self.assertEqual(captured_request["method"], "POST")
        self.assertEqual(captured_request["route"], "/missions")
        self.assertEqual(
            captured_request["payload"],
            {
                "mission_type": "inspect_local_zone",
                "target_hint": "shelf edge",
                "capture_intent": "object_check",
                "max_duration_s": 25.0,
                "return_policy": "return_and_land",
            },
        )

    def test_service_rejects_local_inspect_without_local_flight_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = _MODULE.DroneDaemonService(
                repo_root=Path(temp_dir),
                env_file=None,
                pose_provider=_MODULE.StubPoseProvider(healthy=True),
                radio_provider=_HealthyRadioProvider(),
            )

            with self.assertRaisesRegex(_MODULE.ConflictError, "inspect_local_zone missions require"):
                service.create_mission(
                    {
                        "mission_type": "inspect_local_zone",
                        "target_hint": "shelf edge",
                        "capture_intent": "scene",
                        "max_duration_s": 25.0,
                        "return_policy": "return_and_land",
                    }
                )

    def test_service_executes_stubbed_local_inspect_mission_after_manual_arm(self) -> None:
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
                skill_layer_mode="bounded_local_navigation_only",
            )
            service._run_local_inspect_mission = lambda mission: "inspect-local.json"  # type: ignore[attr-defined]
            service.start()
            try:
                mission = service.create_mission(
                    {
                        "mission_type": "inspect_local_zone",
                        "target_hint": "shelf edge",
                        "capture_intent": "object_check",
                        "max_duration_s": 25.0,
                        "return_policy": "return_and_land",
                    }
                )
                armed = service.manual_arm(mission.mission_id)
                current_payload: dict[str, object] = {}
                for _ in range(40):
                    current_payload = service.mission_payload(mission.mission_id)
                    if str(current_payload.get("state") or "") == "completed":
                        break
                    time.sleep(0.05)
                else:
                    self.fail("local inspect mission did not complete within the bounded wait window")
            finally:
                service.shutdown()

        current = current_payload
        self.assertEqual(mission.mission_type, "inspect_local_zone")
        self.assertEqual(mission.state, "pending_manual_arm")
        self.assertEqual(armed.state, "queued")
        self.assertEqual(current["state"], "completed")
        self.assertEqual(current["artifact_name"], "inspect-local.json")


if __name__ == "__main__":
    unittest.main()
