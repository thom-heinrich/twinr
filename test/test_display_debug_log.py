from pathlib import Path
from datetime import datetime, timezone
import os
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.display.debug_log import TwinrDisplayDebugLogBuilder
from twinr.agent.base_agent.state.snapshot import RuntimeSnapshot
from twinr.ops.health import ServiceHealth, TwinrSystemHealth
from twinr.ops.events import TwinrOpsEventStore
from twinr.ops.remote_memory_watchdog import (
    RemoteMemoryWatchdogSample,
    RemoteMemoryWatchdogSnapshot,
    RemoteMemoryWatchdogStore,
)
from twinr.ops.usage import TwinrUsageStore


class DisplayDebugLogTests(unittest.TestCase):
    def test_build_sections_groups_system_llm_and_hardware_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            now_text = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            config = TwinrConfig(project_root=temp_dir)
            event_store = TwinrOpsEventStore.from_config(config)
            usage_store = TwinrUsageStore.from_config(config)
            watchdog_store = RemoteMemoryWatchdogStore.from_config(config)

            event_store.append(
                event="remote_memory_watchdog_status_changed",
                message="Remote memory watchdog observed initial state ok.",
                data={"status": "ok"},
            )
            event_store.append(
                event="button_pressed",
                message="Physical button `green` was pressed.",
                data={"button": "green"},
            )
            event_store.append(
                event="turn_started",
                message="Conversation listening window started.",
                data={"request_source": "button"},
            )
            usage_store.append(
                source="streaming_loop",
                request_kind="conversation",
                model="gpt-4o-mini",
                used_web_search=False,
                metadata={
                    "request_source": "button",
                    "transcript": "Wie geht es dir heute?",
                },
            )
            watchdog_store.save(
                RemoteMemoryWatchdogSnapshot(
                    schema_version=1,
                    started_at=now_text,
                    updated_at=now_text,
                    hostname="pi",
                    pid=os.getpid(),
                    interval_s=1.0,
                    history_limit=3600,
                    sample_count=3,
                    failure_count=0,
                    last_ok_at=now_text,
                    last_failure_at=None,
                    artifact_path=str(watchdog_store.path),
                    current=RemoteMemoryWatchdogSample(
                        seq=3,
                        captured_at=now_text,
                        status="ok",
                        ready=True,
                        mode="watchdog_artifact",
                        required=True,
                        latency_ms=14587.7,
                        consecutive_ok=3,
                        consecutive_fail=0,
                    ),
                    recent_samples=(),
                    heartbeat_at=now_text,
                    probe_inflight=True,
                    probe_started_at=now_text,
                    probe_age_s=4.0,
                )
            )

            builder = TwinrDisplayDebugLogBuilder(
                config=config,
                event_store=event_store,
                usage_store=usage_store,
                watchdog_store=watchdog_store,
            )
            sections = builder.build_sections(
                snapshot=RuntimeSnapshot(
                    status="waiting",
                    last_transcript="Wie geht es dir heute?",
                    last_response="Mir geht es gut.",
                ),
                runtime_status="Waiting",
                internet_state="ok",
                ai_state="ok",
                system_state="ok",
                clock_text="09:31",
                health=TwinrSystemHealth(
                    status="ok",
                    captured_at="2026-03-17T09:31:00Z",
                    hostname="pi",
                    cpu_temperature_c=60.4,
                    memory_used_percent=18.3,
                    disk_used_percent=18.8,
                    recent_error_count=4,
                    services=(
                        ServiceHealth(
                            key="conversation_loop",
                            label="Conversation loop",
                            running=True,
                            count=1,
                            detail="pid=111 python --run-streaming-loop",
                        ),
                        ServiceHealth(
                            key="display",
                            label="Display loop",
                            running=True,
                            count=1,
                            detail="pid=111 display-companion",
                        ),
                    ),
                ),
            )

        self.assertEqual([title for title, _lines in sections], ["System Log", "LLM Log", "Hardware Log"])
        self.assertIn("09:31 rt waiting | sys ok", sections[0][1][0])
        self.assertTrue(any("chonky ok" in line for line in sections[0][1]))
        self.assertTrue(any("probe inflight" in line for line in sections[0][1]))
        self.assertTrue(any("errs 4 | conv 1 | disp 1" in line for line in sections[0][1]))
        self.assertTrue(any("mode waiting | src button" in line for line in sections[1][1]))
        self.assertTrue(any("user Wie geht es dir heute?" in line for line in sections[1][1]))
        self.assertTrue(any("conv pid 111 | disp pid 111" in line for line in sections[2][1]))
        self.assertTrue(any("cpu 60C | mem 20% | disk 20%" in line for line in sections[2][1]))

    def test_build_sections_compresses_repeated_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            event_store = TwinrOpsEventStore.from_config(config)
            usage_store = TwinrUsageStore.from_config(config)
            watchdog_store = RemoteMemoryWatchdogStore.from_config(config)
            for _ in range(3):
                event_store.append(
                    event="automation_execution_failed",
                    message="An automation failed during execution.",
                    level="error",
                    data={"name": "Morning Briefing"},
                )

            builder = TwinrDisplayDebugLogBuilder(
                config=config,
                event_store=event_store,
                usage_store=usage_store,
                watchdog_store=watchdog_store,
            )
            sections = builder.build_sections(
                snapshot=RuntimeSnapshot(status="waiting"),
                runtime_status="Waiting",
                internet_state="ok",
                ai_state="ok",
                system_state="ok",
                clock_text="09:31",
            )

        self.assertTrue(any("x3" in line for line in sections[0][1]))

    def test_build_sections_does_not_drift_when_only_watchdog_ages_advance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            now_text = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            config = TwinrConfig(project_root=temp_dir)
            watchdog_store = RemoteMemoryWatchdogStore.from_config(config)
            watchdog_store.save(
                RemoteMemoryWatchdogSnapshot(
                    schema_version=1,
                    started_at=now_text,
                    updated_at=now_text,
                    hostname="pi",
                    pid=os.getpid(),
                    interval_s=1.0,
                    history_limit=3600,
                    sample_count=3,
                    failure_count=0,
                    last_ok_at=now_text,
                    last_failure_at=None,
                    artifact_path=str(watchdog_store.path),
                    current=RemoteMemoryWatchdogSample(
                        seq=3,
                        captured_at=now_text,
                        status="ok",
                        ready=True,
                        mode="watchdog_artifact",
                        required=True,
                        latency_ms=14587.7,
                        consecutive_ok=3,
                        consecutive_fail=0,
                    ),
                    recent_samples=(),
                    heartbeat_at=now_text,
                    probe_inflight=True,
                    probe_started_at=now_text,
                    probe_age_s=4.0,
                )
            )
            builder = TwinrDisplayDebugLogBuilder(
                config=config,
                event_store=TwinrOpsEventStore.from_config(config),
                usage_store=TwinrUsageStore.from_config(config),
                watchdog_store=watchdog_store,
            )

            first = builder.build_sections(
                snapshot=RuntimeSnapshot(status="waiting"),
                runtime_status="Waiting",
                internet_state="ok",
                ai_state="ok",
                system_state="ok",
                clock_text="09:31",
            )
            second = builder.build_sections(
                snapshot=RuntimeSnapshot(status="waiting"),
                runtime_status="Waiting",
                internet_state="ok",
                ai_state="ok",
                system_state="ok",
                clock_text="09:31",
            )

        self.assertEqual(first, second)

    def test_build_sections_does_not_drift_for_minor_host_and_latency_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            now_text = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            config = TwinrConfig(project_root=temp_dir)
            watchdog_store = RemoteMemoryWatchdogStore.from_config(config)
            builder = TwinrDisplayDebugLogBuilder(
                config=config,
                event_store=TwinrOpsEventStore.from_config(config),
                usage_store=TwinrUsageStore.from_config(config),
                watchdog_store=watchdog_store,
            )

            def save_watchdog(latency_ms: float) -> None:
                watchdog_store.save(
                    RemoteMemoryWatchdogSnapshot(
                        schema_version=1,
                        started_at=now_text,
                        updated_at=now_text,
                        hostname="pi",
                        pid=os.getpid(),
                        interval_s=1.0,
                        history_limit=3600,
                        sample_count=3,
                        failure_count=0,
                        last_ok_at=now_text,
                        last_failure_at=None,
                        artifact_path=str(watchdog_store.path),
                        current=RemoteMemoryWatchdogSample(
                            seq=3,
                            captured_at=now_text,
                            status="ok",
                            ready=True,
                            mode="watchdog_artifact",
                            required=True,
                            latency_ms=latency_ms,
                            consecutive_ok=3,
                            consecutive_fail=0,
                        ),
                        recent_samples=(),
                        heartbeat_at=now_text,
                        probe_inflight=True,
                        probe_started_at=now_text,
                        probe_age_s=4.0,
                    )
                )

            def health(cpu: float, mem: float, disk: float) -> TwinrSystemHealth:
                return TwinrSystemHealth(
                    status="ok",
                    captured_at="2026-03-17T09:31:00Z",
                    hostname="pi",
                    cpu_temperature_c=cpu,
                    memory_used_percent=mem,
                    disk_used_percent=disk,
                    recent_error_count=4,
                    services=(
                        ServiceHealth(
                            key="conversation_loop",
                            label="Conversation loop",
                            running=True,
                            count=1,
                            detail="pid=111 python --run-streaming-loop",
                        ),
                        ServiceHealth(
                            key="display",
                            label="Display loop",
                            running=True,
                            count=1,
                            detail="pid=111 display-companion",
                        ),
                    ),
                )

            save_watchdog(15945.0)
            first = builder.build_sections(
                snapshot=RuntimeSnapshot(status="waiting"),
                runtime_status="Waiting",
                internet_state="ok",
                ai_state="ok",
                system_state="ok",
                clock_text="09:31",
                health=health(59.4, 16.8, 18.9),
            )

            save_watchdog(14158.0)
            second = builder.build_sections(
                snapshot=RuntimeSnapshot(status="waiting"),
                runtime_status="Waiting",
                internet_state="ok",
                ai_state="ok",
                system_state="ok",
                clock_text="09:31",
                health=health(58.4, 17.1, 18.9),
            )

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
