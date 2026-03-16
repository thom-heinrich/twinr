from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.agent.base_agent.state.snapshot import RuntimeSnapshot
from twinr.ops import health as health_mod


class OpsHealthTests(unittest.TestCase):
    def test_collect_service_health_treats_streaming_loop_as_conversation_loop(self) -> None:
        entries = (
            health_mod._ProcessEntry(
                pid="123",
                command="python -u -m twinr --run-streaming-loop",
            ),
        )

        with mock.patch.object(health_mod, "_list_process_entries", return_value=entries):
            services, probe_ok = health_mod._collect_service_health()

        self.assertTrue(probe_ok)
        conversation = next(service for service in services if service.key == "conversation_loop")
        self.assertTrue(conversation.running)
        self.assertEqual(conversation.count, 1)
        self.assertIn("--run-streaming-loop", conversation.detail)

    def test_collect_system_health_treats_display_companion_lock_owner_as_running_display(self) -> None:
        config = TwinrConfig(
            project_root="/tmp/twinr",
            runtime_state_path="/tmp/twinr/state/runtime-state.json",
        )
        snapshot = RuntimeSnapshot(status="waiting")
        services = (
            health_mod.ServiceHealth(
                key="web",
                label="Web UI",
                running=False,
                count=0,
                detail="Service not detected.",
            ),
            health_mod.ServiceHealth(
                key="conversation_loop",
                label="Conversation loop",
                running=True,
                count=1,
                detail="pid=123 python --run-streaming-loop",
            ),
            health_mod.ServiceHealth(
                key="display",
                label="Display loop",
                running=False,
                count=0,
                detail="Service not detected.",
            ),
        )

        with mock.patch.object(health_mod, "_captured_at", return_value="2026-03-16T18:20:00Z"):
            with mock.patch.object(health_mod, "_read_recent_error_count", return_value=(0, True)):
                with mock.patch.object(health_mod, "_resolve_project_root", return_value=Path("/tmp/twinr")):
                    with mock.patch.object(health_mod, "_read_loadavg", return_value=(0.1, 0.1, 0.1)):
                        with mock.patch.object(health_mod, "_read_memory", return_value=(1024, 900, 12.5)):
                            with mock.patch.object(health_mod, "_read_disk", return_value=(64.0, 40.0, 22.0)):
                                with mock.patch.object(health_mod, "_collect_service_health", return_value=(services, True)):
                                    with mock.patch.object(health_mod, "_read_cpu_temperature_c", return_value=45.0):
                                        with mock.patch.object(health_mod, "_read_hostname", return_value="picarx"):
                                            with mock.patch.object(health_mod, "_read_uptime_seconds", return_value=123.0):
                                                with mock.patch.object(health_mod, "loop_lock_owner", return_value=123):
                                                    health = health_mod.collect_system_health(config, snapshot=snapshot)

        self.assertEqual(health.status, "ok")
        display = next(service for service in health.services if service.key == "display")
        self.assertTrue(display.running)
        self.assertEqual(display.count, 1)
        self.assertIn("display-companion", display.detail)


if __name__ == "__main__":
    unittest.main()
