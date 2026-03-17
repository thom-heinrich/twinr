from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.agent.base_agent.state.snapshot import RuntimeSnapshot
from twinr.display.heartbeat import DisplayHeartbeatStore, build_display_heartbeat
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
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=str(root),
                runtime_state_path=str(root / "state" / "runtime-state.json"),
            )
            snapshot = RuntimeSnapshot(status="waiting")
            DisplayHeartbeatStore.from_config(config).save(
                build_display_heartbeat(
                    runtime_status="waiting",
                    phase="idle",
                    seq=7,
                )
            )
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
                    with mock.patch.object(health_mod, "_resolve_project_root", return_value=root):
                        with mock.patch.object(health_mod, "_read_loadavg", return_value=(0.1, 0.1, 0.1)):
                            with mock.patch.object(health_mod, "_read_memory", return_value=(1024, 900, 12.5)):
                                with mock.patch.object(health_mod, "_read_disk", return_value=(64.0, 40.0, 22.0)):
                                    with mock.patch.object(health_mod, "_collect_service_health", return_value=(services, True)):
                                        with mock.patch.object(health_mod, "_read_cpu_temperature_c", return_value=45.0):
                                            with mock.patch.object(health_mod, "_read_hostname", return_value="picarx"):
                                                with mock.patch.object(health_mod, "_read_uptime_seconds", return_value=123.0):
                                                    with mock.patch.object(health_mod, "loop_lock_owner", return_value=os.getpid()):
                                                        health = health_mod.collect_system_health(config, snapshot=snapshot)

        self.assertEqual(health.status, "ok")
        display = next(service for service in health.services if service.key == "display")
        self.assertTrue(display.running)
        self.assertEqual(display.count, 1)
        self.assertIn("display-companion", display.detail)

    def test_collect_system_health_marks_stale_display_companion_heartbeat_as_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=str(root),
                runtime_state_path=str(root / "state" / "runtime-state.json"),
            )
            snapshot = RuntimeSnapshot(status="waiting")
            stale_time = datetime.now(timezone.utc) - timedelta(seconds=90)
            DisplayHeartbeatStore.from_config(config).save(
                build_display_heartbeat(
                    runtime_status="waiting",
                    phase="rendering",
                    seq=3,
                    pid=123,
                    updated_at=stale_time,
                    last_render_started_at=stale_time,
                )
            )
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
                    with mock.patch.object(health_mod, "_resolve_project_root", return_value=root):
                        with mock.patch.object(health_mod, "_read_loadavg", return_value=(0.1, 0.1, 0.1)):
                            with mock.patch.object(health_mod, "_read_memory", return_value=(1024, 900, 12.5)):
                                with mock.patch.object(health_mod, "_read_disk", return_value=(64.0, 40.0, 22.0)):
                                    with mock.patch.object(health_mod, "_collect_service_health", return_value=(services, True)):
                                        with mock.patch.object(health_mod, "_read_cpu_temperature_c", return_value=45.0):
                                            with mock.patch.object(health_mod, "_read_hostname", return_value="picarx"):
                                                with mock.patch.object(health_mod, "_read_uptime_seconds", return_value=123.0):
                                                    with mock.patch.object(health_mod, "loop_lock_owner", return_value=123):
                                                        health = health_mod.collect_system_health(config, snapshot=snapshot)

        display = next(service for service in health.services if service.key == "display")
        self.assertFalse(display.running)
        self.assertEqual(display.count, 0)
        self.assertIn("stale", display.detail)
        self.assertEqual(health.status, "warn")

    def test_collect_system_health_treats_inflight_rendering_heartbeat_as_running(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=str(root),
                runtime_state_path=str(root / "state" / "runtime-state.json"),
            )
            snapshot = RuntimeSnapshot(status="processing")
            render_started_at = datetime.now(timezone.utc) - timedelta(seconds=20)
            last_completed_at = render_started_at - timedelta(seconds=2)
            DisplayHeartbeatStore.from_config(config).save(
                build_display_heartbeat(
                    runtime_status="processing",
                    phase="rendering",
                    seq=11,
                    pid=123,
                    updated_at=render_started_at,
                    last_render_started_at=render_started_at,
                    last_render_completed_at=last_completed_at,
                )
            )
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
                    with mock.patch.object(health_mod, "_resolve_project_root", return_value=root):
                        with mock.patch.object(health_mod, "_read_loadavg", return_value=(0.1, 0.1, 0.1)):
                            with mock.patch.object(health_mod, "_read_memory", return_value=(1024, 900, 12.5)):
                                with mock.patch.object(health_mod, "_read_disk", return_value=(64.0, 40.0, 22.0)):
                                    with mock.patch.object(health_mod, "_collect_service_health", return_value=(services, True)):
                                        with mock.patch.object(health_mod, "_read_cpu_temperature_c", return_value=45.0):
                                            with mock.patch.object(health_mod, "_read_hostname", return_value="picarx"):
                                                with mock.patch.object(health_mod, "_read_uptime_seconds", return_value=123.0):
                                                    with mock.patch.object(health_mod, "loop_lock_owner", return_value=123):
                                                        health = health_mod.collect_system_health(config, snapshot=snapshot)

        display = next(service for service in health.services if service.key == "display")
        self.assertTrue(display.running)
        self.assertEqual(display.count, 1)
        self.assertIn("display-companion", display.detail)
        self.assertEqual(health.status, "ok")


if __name__ == "__main__":
    unittest.main()
