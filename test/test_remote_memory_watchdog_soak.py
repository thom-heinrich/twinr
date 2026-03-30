from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from twinr.ops.remote_memory_watchdog_soak import (
    SystemdServiceState,
    WatchdogArtifactState,
    WatchdogSoakSample,
    build_soak_summary,
)


class RemoteMemoryWatchdogSoakTests(unittest.TestCase):
    def test_build_soak_summary_marks_clean_run_as_passing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "report"
            snapshot_path = Path(temp_dir) / "remote_memory_watchdog.json"
            samples = [
                WatchdogSoakSample(
                    observed_at="2026-03-16T18:00:00Z",
                    elapsed_s=0.0,
                    service_running=True,
                    watchdog_ok=True,
                    artifact_fresh=True,
                    stale_seconds=12.0,
                    freshness_source="updated_at",
                    collection_latency_ms=4.0,
                    status_reason="ok",
                    status_flags=["ok"],
                    service=SystemdServiceState(
                        active_state="active",
                        sub_state="running",
                        exec_main_pid=330727,
                        n_restarts=5,
                    ),
                    artifact=WatchdogArtifactState(
                        updated_at="2026-03-16T17:59:48Z",
                        sample_count=40,
                        failure_count=0,
                        last_ok_at="2026-03-16T17:59:48Z",
                        last_failure_at=None,
                        current_status="ok",
                        current_ready=True,
                        current_mode="remote_primary",
                        current_required=True,
                        current_latency_ms=15234.5,
                        current_consecutive_ok=40,
                        current_consecutive_fail=0,
                        current_captured_at="2026-03-16T17:59:48Z",
                        file_mtime="2026-03-16T17:59:48Z",
                        file_size_bytes=4096,
                    ),
                ),
                WatchdogSoakSample(
                    observed_at="2026-03-16T18:00:30Z",
                    elapsed_s=30.0,
                    service_running=True,
                    watchdog_ok=True,
                    artifact_fresh=True,
                    stale_seconds=18.0,
                    freshness_source="updated_at",
                    collection_latency_ms=5.0,
                    status_reason="ok",
                    status_flags=["ok"],
                    service=SystemdServiceState(
                        active_state="active",
                        sub_state="running",
                        exec_main_pid=330727,
                        n_restarts=5,
                    ),
                    artifact=WatchdogArtifactState(
                        updated_at="2026-03-16T18:00:12Z",
                        sample_count=42,
                        failure_count=0,
                        last_ok_at="2026-03-16T18:00:12Z",
                        last_failure_at=None,
                        current_status="ok",
                        current_ready=True,
                        current_mode="remote_primary",
                        current_required=True,
                        current_latency_ms=14980.1,
                        current_consecutive_ok=42,
                        current_consecutive_fail=0,
                        current_captured_at="2026-03-16T18:00:12Z",
                        file_mtime="2026-03-16T18:00:12Z",
                        file_size_bytes=4352,
                    ),
                ),
            ]

            summary = build_soak_summary(
                samples,
                started_at="2026-03-16T18:00:00Z",
                ended_at="2026-03-16T18:00:30Z",
                requested_duration_s=30.0,
                observed_duration_s=30.0,
                interval_s=30.0,
                max_stale_s=180.0,
                min_samples=2,
                require_artifact_progress=True,
                service_name="twinr-remote-memory-watchdog.service",
                snapshot_path=snapshot_path,
                output_dir=output_dir,
                stop_reason="deadline_reached",
                interrupted=False,
                termination_signal=None,
                systemctl_timeout_s=5.0,
            )

        self.assertTrue(summary["all_checks_passed"])
        self.assertEqual(summary["restart_delta"], 0)
        self.assertEqual(summary["exec_main_pid_change_count"], 0)
        self.assertEqual(summary["sample_count_delta"], 2)
        self.assertEqual(summary["failure_count_delta"], 0)
        self.assertEqual(summary["stale_sample_count"], 0)

    def test_build_soak_summary_flags_restart_failure_and_stale_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "report"
            snapshot_path = Path(temp_dir) / "remote_memory_watchdog.json"
            samples = [
                WatchdogSoakSample(
                    observed_at="2026-03-16T18:00:00Z",
                    elapsed_s=0.0,
                    service_running=True,
                    watchdog_ok=True,
                    artifact_fresh=True,
                    stale_seconds=10.0,
                    freshness_source="updated_at",
                    collection_latency_ms=4.0,
                    status_reason="ok",
                    status_flags=["ok"],
                    service=SystemdServiceState(
                        active_state="active",
                        sub_state="running",
                        exec_main_pid=330727,
                        n_restarts=5,
                    ),
                    artifact=WatchdogArtifactState(
                        updated_at="2026-03-16T17:59:50Z",
                        sample_count=40,
                        failure_count=0,
                        last_ok_at="2026-03-16T17:59:50Z",
                        last_failure_at=None,
                        current_status="ok",
                        current_ready=True,
                        current_mode="remote_primary",
                        current_required=True,
                        current_latency_ms=15000.0,
                        current_consecutive_ok=40,
                        current_consecutive_fail=0,
                        current_captured_at="2026-03-16T17:59:50Z",
                        file_mtime="2026-03-16T17:59:50Z",
                        file_size_bytes=4096,
                    ),
                ),
                WatchdogSoakSample(
                    observed_at="2026-03-16T18:01:00Z",
                    elapsed_s=60.0,
                    service_running=False,
                    watchdog_ok=False,
                    artifact_fresh=False,
                    stale_seconds=240.0,
                    freshness_source="updated_at",
                    collection_latency_ms=8.0,
                    status_reason="service_not_running",
                    status_flags=["service_not_running", "watchdog_not_ok", "artifact_stale"],
                    service=SystemdServiceState(
                        active_state="activating",
                        sub_state="auto-restart",
                        exec_main_pid=330999,
                        n_restarts=6,
                    ),
                    artifact=WatchdogArtifactState(
                        updated_at="2026-03-16T17:57:00Z",
                        sample_count=40,
                        failure_count=1,
                        last_ok_at="2026-03-16T17:59:50Z",
                        last_failure_at="2026-03-16T18:00:45Z",
                        current_status="fail",
                        current_ready=False,
                        current_mode="remote_primary",
                        current_required=True,
                        current_latency_ms=17000.0,
                        current_consecutive_ok=0,
                        current_consecutive_fail=1,
                        current_captured_at="2026-03-16T18:00:45Z",
                        file_mtime="2026-03-16T17:57:00Z",
                        file_size_bytes=4096,
                    ),
                ),
            ]

            summary = build_soak_summary(
                samples,
                started_at="2026-03-16T18:00:00Z",
                ended_at="2026-03-16T18:01:00Z",
                requested_duration_s=60.0,
                observed_duration_s=60.0,
                interval_s=30.0,
                max_stale_s=180.0,
                min_samples=2,
                require_artifact_progress=True,
                service_name="twinr-remote-memory-watchdog.service",
                snapshot_path=snapshot_path,
                output_dir=output_dir,
                stop_reason="deadline_reached",
                interrupted=False,
                termination_signal=None,
                systemctl_timeout_s=5.0,
            )

        self.assertFalse(summary["all_checks_passed"])
        self.assertEqual(summary["restart_delta"], 1)
        self.assertEqual(summary["exec_main_pid_change_count"], 1)
        self.assertEqual(summary["failure_count_delta"], 1)
        self.assertEqual(summary["sample_count_delta"], 0)
        self.assertEqual(summary["non_running_sample_count"], 1)
        self.assertEqual(summary["non_ok_sample_count"], 1)
        self.assertEqual(summary["stale_sample_count"], 1)


if __name__ == "__main__":
    unittest.main()
