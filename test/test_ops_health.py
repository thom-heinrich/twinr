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
from twinr.ops.process_memory import ProcessMemoryMetrics, StreamingMemorySnapshot


class OpsHealthTests(unittest.TestCase):
    def setUp(self) -> None:
        health_mod._PROCESS_CACHE = None
        health_mod._PI_STATE_CACHE = None
        health_mod._MEMORY_STATUS_CACHE = None

    def tearDown(self) -> None:
        health_mod._PROCESS_CACHE = None
        health_mod._PI_STATE_CACHE = None
        health_mod._MEMORY_STATUS_CACHE = None

    def collect_health_with_services(
        self,
        config: TwinrConfig,
        *,
        snapshot: RuntimeSnapshot,
        services: tuple[health_mod.ServiceHealth, ...],
        root: Path,
        memory_total_mb: int = 1024,
        memory_available_mb: int = 900,
        memory_used_percent: float = 12.5,
        swap_total_mb: int | None = None,
        swap_used_percent: float | None = None,
    ) -> health_mod.TwinrSystemHealth:
        with mock.patch.object(health_mod, "_captured_at", return_value="2026-03-16T18:20:00Z"):
            with mock.patch.object(health_mod, "_read_recent_error_count", return_value=(0, True)):
                with mock.patch.object(health_mod, "_resolve_project_root", return_value=root):
                    with mock.patch.object(health_mod, "_read_loadavg", return_value=(0.1, 0.1, 0.1)):
                        with mock.patch.object(
                            health_mod,
                            "_read_memory",
                            return_value=(
                                memory_total_mb,
                                memory_available_mb,
                                memory_used_percent,
                                swap_total_mb,
                                None if swap_total_mb is None or swap_used_percent is None else int(
                                    round((swap_total_mb * swap_used_percent) / 100.0)
                                ),
                                swap_used_percent,
                            ),
                        ):
                            with mock.patch.object(health_mod, "_read_disk", return_value=(64.0, 40.0, 22.0)):
                                with mock.patch.object(health_mod, "_read_pressure", return_value=(None, None, None, None, None)):
                                    with mock.patch.object(health_mod, "_collect_service_health", return_value=(services, True)):
                                        with mock.patch.object(
                                            health_mod,
                                            "_read_pi_throttled_state",
                                            return_value=health_mod._PiThrottledState(raw_hex="0x0"),
                                        ):
                                            with mock.patch.object(health_mod, "_read_cpu_temperature_c", return_value=45.0):
                                                with mock.patch.object(health_mod, "_read_hostname", return_value="picarx"):
                                                    with mock.patch.object(health_mod, "_read_uptime_seconds", return_value=123.0):
                                                        return health_mod.collect_system_health(config, snapshot=snapshot)

    def test_collect_service_health_treats_streaming_loop_as_conversation_loop(self) -> None:
        entries = (
            health_mod._ProcessEntry(
                pid="123",
                ppid="1",
                command="python -u -m twinr --run-streaming-loop",
            ),
        )

        with mock.patch.object(health_mod, "_list_process_entries", return_value=entries):
            services, probe_ok = health_mod._collect_service_health(TwinrConfig())

        self.assertTrue(probe_ok)
        conversation = next(service for service in services if service.key == "conversation_loop")
        self.assertTrue(conversation.running)
        self.assertEqual(conversation.count, 1)
        self.assertIn("--run-streaming-loop", conversation.detail)

    def test_collect_service_health_detects_runtime_supervisor(self) -> None:
        entries = (
            health_mod._ProcessEntry(
                pid="321",
                ppid="1",
                command="python -u -m twinr --run-runtime-supervisor",
            ),
        )

        with mock.patch.object(health_mod, "_list_process_entries", return_value=entries):
            services, probe_ok = health_mod._collect_service_health(TwinrConfig())

        self.assertTrue(probe_ok)
        supervisor = next(service for service in services if service.key == "runtime_supervisor")
        self.assertTrue(supervisor.running)
        self.assertEqual(supervisor.count, 1)
        self.assertIn("--run-runtime-supervisor", supervisor.detail)

    def test_collect_service_health_ignores_streaming_worker_with_inherited_parent_argv(self) -> None:
        entries = (
            health_mod._ProcessEntry(
                pid="123",
                ppid="1",
                command="python -u -m twinr --run-streaming-loop",
            ),
            health_mod._ProcessEntry(
                pid="456",
                ppid="123",
                command="python -u -m twinr --run-streaming-loop",
            ),
        )

        with mock.patch.object(health_mod, "_list_process_entries", return_value=entries):
            with mock.patch.object(health_mod, "loop_lock_owner", return_value=123):
                services, probe_ok = health_mod._collect_service_health(TwinrConfig())

        self.assertTrue(probe_ok)
        conversation = next(service for service in services if service.key == "conversation_loop")
        self.assertTrue(conversation.running)
        self.assertEqual(conversation.count, 1)
        self.assertIn("pid=123", conversation.detail)
        self.assertNotIn("pid=456", conversation.detail)

    def test_collect_service_health_ignores_shell_c_script_text_with_streaming_flag(self) -> None:
        entries = (
            health_mod._ProcessEntry(
                pid="123",
                ppid="1",
                command="python -u -m twinr --run-streaming-loop",
                argv=("python", "-u", "-m", "twinr", "--run-streaming-loop"),
            ),
            health_mod._ProcessEntry(
                pid="456",
                ppid="1",
                command="bash -c print --run-streaming-loop",
                argv=("bash", "-c", "print('--run-streaming-loop')"),
            ),
        )

        with mock.patch.object(health_mod, "_list_process_entries", return_value=entries):
            services, probe_ok = health_mod._collect_service_health(TwinrConfig())

        self.assertTrue(probe_ok)
        conversation = next(service for service in services if service.key == "conversation_loop")
        self.assertTrue(conversation.running)
        self.assertEqual(conversation.count, 1)
        self.assertIn("pid=123", conversation.detail)
        self.assertNotIn("pid=456", conversation.detail)

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

            with mock.patch.object(health_mod, "loop_lock_owner", return_value=os.getpid()):
                health = self.collect_health_with_services(
                    config,
                    snapshot=snapshot,
                    services=services,
                    root=root,
                )

        self.assertEqual(health.status, "ok")
        display = next(service for service in health.services if service.key == "display")
        self.assertTrue(display.running)
        self.assertEqual(display.count, 1)
        self.assertIn("display-companion", display.detail)

    def test_collect_system_health_keeps_ok_when_memavailable_has_headroom(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=str(root),
                runtime_state_path=str(root / "state" / "runtime-state.json"),
            )
            snapshot = RuntimeSnapshot(status="waiting")
            services = (
                health_mod.ServiceHealth(
                    key="web",
                    label="Web UI",
                    running=True,
                    count=1,
                    detail="pid=111 python --run-web",
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
                    running=True,
                    count=1,
                    detail="pid=123 display-companion",
                ),
            )

            health = self.collect_health_with_services(
                config,
                snapshot=snapshot,
                services=services,
                root=root,
                memory_total_mb=3796,
                memory_available_mb=693,
                memory_used_percent=81.7,
            )

        self.assertEqual(health.status, "ok")

    def test_collect_system_health_warns_when_memavailable_is_low(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=str(root),
                runtime_state_path=str(root / "state" / "runtime-state.json"),
            )
            snapshot = RuntimeSnapshot(status="waiting")
            services = (
                health_mod.ServiceHealth(
                    key="web",
                    label="Web UI",
                    running=True,
                    count=1,
                    detail="pid=111 python --run-web",
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
                    running=True,
                    count=1,
                    detail="pid=123 display-companion",
                ),
            )

            health = self.collect_health_with_services(
                config,
                snapshot=snapshot,
                services=services,
                root=root,
                memory_total_mb=3796,
                memory_available_mb=400,
                memory_used_percent=78.0,
            )

        self.assertEqual(health.status, "warn")

    def test_collect_system_health_warns_when_swap_is_saturated_despite_memavailable_headroom(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=str(root),
                runtime_state_path=str(root / "state" / "runtime-state.json"),
            )
            snapshot = RuntimeSnapshot(status="waiting")
            services = (
                health_mod.ServiceHealth(
                    key="web",
                    label="Web UI",
                    running=True,
                    count=1,
                    detail="pid=111 python --run-web",
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
                    running=True,
                    count=1,
                    detail="pid=123 display-companion",
                ),
            )

            health = self.collect_health_with_services(
                config,
                snapshot=snapshot,
                services=services,
                root=root,
                memory_total_mb=3796,
                memory_available_mb=545,
                memory_used_percent=85.6,
                swap_total_mb=199,
                swap_used_percent=100.0,
            )

        self.assertEqual(health.status, "warn")

    def test_collect_system_health_exposes_streaming_memory_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=str(root),
                runtime_state_path=str(root / "state" / "runtime-state.json"),
            )
            snapshot = RuntimeSnapshot(status="waiting")
            services = (
                health_mod.ServiceHealth(
                    key="web",
                    label="Web UI",
                    running=True,
                    count=1,
                    detail="pid=111 python --run-web",
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
                    running=True,
                    count=1,
                    detail="pid=123 display-companion",
                ),
            )
            memory_snapshot = StreamingMemorySnapshot(
                schema_version=1,
                captured_at="2026-04-05T00:00:00+00:00",
                pid=123,
                boot_id="boot-1",
                pid_start_ticks=456,
                current_metrics=ProcessMemoryMetrics(
                    vm_rss_kb=1_900_184,
                    anonymous_kb=1_807_636,
                ),
                owner_label="display_companion.hdmi_wayland.native_window_frame",
                owner_detail="phase=display.hdmi_wayland.native_window_frame_presented anonymous_delta_mb=1580",
                owner_rss_delta_kb=1_650_000,
                owner_anonymous_delta_kb=1_580_000,
                phases=(),
            )

            with mock.patch.object(health_mod, "loop_lock_owner", return_value=123):
                with mock.patch.object(health_mod, "load_current_streaming_memory_snapshot", return_value=memory_snapshot):
                    health = self.collect_health_with_services(
                        config,
                        snapshot=snapshot,
                        services=services,
                        root=root,
                    )

        self.assertEqual(
            health.memory_owner_label,
            "display_companion.hdmi_wayland.native_window_frame",
        )
        self.assertEqual(
            health.memory_owner_detail,
            "phase=display.hdmi_wayland.native_window_frame_presented anonymous_delta_mb=1580",
        )
        self.assertEqual(health.memory_owner_rss_mb, 1856)
        self.assertEqual(health.memory_owner_anonymous_mb, 1765)
        self.assertEqual(health.memory_owner_rss_delta_mb, 1611)
        self.assertEqual(health.memory_owner_anonymous_delta_mb, 1543)

    def test_assess_memory_pressure_status_holds_fail_until_recovery_margin_and_hold_time(self) -> None:
        with mock.patch.object(health_mod.time, "monotonic", side_effect=(10.0, 11.0, 25.0, 46.0)):
            self.assertEqual(
                health_mod.assess_memory_pressure_status(
                    memory_available_mb=250,
                    memory_used_percent=93.0,
                ),
                "fail",
            )
            self.assertEqual(
                health_mod.assess_memory_pressure_status(
                    memory_available_mb=257,
                    memory_used_percent=89.0,
                ),
                "fail",
            )
            self.assertEqual(
                health_mod.assess_memory_pressure_status(
                    memory_available_mb=390,
                    memory_used_percent=86.0,
                ),
                "fail",
            )
            self.assertEqual(
                health_mod.assess_memory_pressure_status(
                    memory_available_mb=390,
                    memory_used_percent=86.0,
                ),
                "warn",
            )

    def test_assess_memory_pressure_status_holds_warn_until_sustained_headroom(self) -> None:
        with mock.patch.object(health_mod.time, "monotonic", side_effect=(100.0, 110.0, 131.0)):
            self.assertEqual(
                health_mod.assess_memory_pressure_status(
                    memory_available_mb=400,
                    memory_used_percent=78.0,
                ),
                "warn",
            )
            self.assertEqual(
                health_mod.assess_memory_pressure_status(
                    memory_available_mb=700,
                    memory_used_percent=60.0,
                ),
                "warn",
            )
            self.assertIsNone(
                health_mod.assess_memory_pressure_status(
                    memory_available_mb=700,
                    memory_used_percent=60.0,
                )
            )

    def test_assess_memory_pressure_status_holds_warn_while_swap_remains_saturated(self) -> None:
        with mock.patch.object(health_mod.time, "monotonic", side_effect=(200.0, 230.0, 251.0, 272.0)):
            self.assertEqual(
                health_mod.assess_memory_pressure_status(
                    memory_available_mb=545,
                    memory_used_percent=85.6,
                    swap_total_mb=199,
                    swap_used_percent=100.0,
                ),
                "warn",
            )
            self.assertEqual(
                health_mod.assess_memory_pressure_status(
                    memory_available_mb=900,
                    memory_used_percent=61.0,
                    swap_total_mb=199,
                    swap_used_percent=100.0,
                ),
                "warn",
            )
            self.assertEqual(
                health_mod.assess_memory_pressure_status(
                    memory_available_mb=900,
                    memory_used_percent=61.0,
                    swap_total_mb=199,
                    swap_used_percent=0.0,
                ),
                "warn",
            )
            self.assertIsNone(
                health_mod.assess_memory_pressure_status(
                    memory_available_mb=900,
                    memory_used_percent=61.0,
                    swap_total_mb=199,
                    swap_used_percent=0.0,
                )
            )

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
                    updated_monotonic_ns=0,
                    last_render_started_monotonic_ns=0,
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

            with mock.patch.object(health_mod, "loop_lock_owner", return_value=123):
                health = self.collect_health_with_services(
                    config,
                    snapshot=snapshot,
                    services=services,
                    root=root,
                )

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

            with mock.patch.object(health_mod, "loop_lock_owner", return_value=123):
                health = self.collect_health_with_services(
                    config,
                    snapshot=snapshot,
                    services=services,
                    root=root,
                )

        display = next(service for service in health.services if service.key == "display")
        self.assertTrue(display.running)
        self.assertEqual(display.count, 1)
        self.assertIn("display-companion", display.detail)
        self.assertEqual(health.status, "ok")

    def test_collect_system_health_degrades_supervisor_managed_restart_window_to_warn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=str(root),
                runtime_state_path=str(root / "state" / "runtime-state.json"),
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
                    running=False,
                    count=0,
                    detail="Service not detected.",
                ),
                health_mod.ServiceHealth(
                    key="runtime_supervisor",
                    label="Runtime supervisor",
                    running=True,
                    count=1,
                    detail="pid=333 python --run-runtime-supervisor",
                ),
                health_mod.ServiceHealth(
                    key="display",
                    label="Display loop",
                    running=True,
                    count=1,
                    detail="pid=444 python --run-display-loop",
                ),
            )

            health = self.collect_health_with_services(
                config,
                snapshot=snapshot,
                services=services,
                root=root,
            )

        self.assertEqual(health.status, "warn")

    def test_collect_system_health_fails_when_conversation_loop_is_missing_without_supervisor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=str(root),
                runtime_state_path=str(root / "state" / "runtime-state.json"),
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
                    running=False,
                    count=0,
                    detail="Service not detected.",
                ),
                health_mod.ServiceHealth(
                    key="display",
                    label="Display loop",
                    running=True,
                    count=1,
                    detail="pid=444 python --run-display-loop",
                ),
            )

            health = self.collect_health_with_services(
                config,
                snapshot=snapshot,
                services=services,
                root=root,
            )

        self.assertEqual(health.status, "fail")


if __name__ == "__main__":
    unittest.main()
