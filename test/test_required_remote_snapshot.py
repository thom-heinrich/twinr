from datetime import datetime, timedelta, timezone
from dataclasses import replace
from pathlib import Path
import os
import sys
import tempfile
from typing import cast
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.workflows.required_remote_snapshot import (
    _default_watchdog_recovery_starter,
    _required_watchdog_probe_mode,
    assess_required_remote_watchdog_snapshot,
    ensure_required_remote_watchdog_snapshot_ready,
)
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError
from twinr.ops.remote_memory_watchdog import (
    RemoteMemoryWatchdogSample,
    RemoteMemoryWatchdogSnapshot,
)


class _FakeStore:
    def __init__(self, snapshot, *, path: str = "/tmp/remote-memory-watchdog.json") -> None:
        self._snapshot = snapshot
        self.path = Path(path)

    def load(self) -> RemoteMemoryWatchdogSnapshot | None:
        return cast(RemoteMemoryWatchdogSnapshot | None, self._snapshot)


class _SequencedStore:
    def __init__(self, snapshots, *, path: str = "/tmp/remote-memory-watchdog.json") -> None:
        self._snapshots = list(snapshots)
        self._index = 0
        self.path = Path(path)

    def load(self) -> RemoteMemoryWatchdogSnapshot | None:
        if not self._snapshots:
            return None
        if self._index >= len(self._snapshots):
            return cast(RemoteMemoryWatchdogSnapshot | None, self._snapshots[-1])
        snapshot = self._snapshots[self._index]
        self._index += 1
        return cast(RemoteMemoryWatchdogSnapshot | None, snapshot)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _build_snapshot(
    *,
    now: datetime,
    pid: int,
    status: str = "ok",
    ready: bool = True,
    required: bool = True,
    latency_ms: float = 18000.0,
    age_s: float = 1.0,
    heartbeat_age_s: float | None = None,
    probe_inflight: bool = False,
    probe_age_s: float | None = None,
    previous_sample_ages_s: tuple[float, ...] = (),
    probe: dict[str, object] | None = None,
) -> RemoteMemoryWatchdogSnapshot:
    captured_at = now - timedelta(seconds=age_s)
    heartbeat_at = captured_at if heartbeat_age_s is None else now - timedelta(seconds=heartbeat_age_s)
    sample = RemoteMemoryWatchdogSample(
        seq=len(previous_sample_ages_s) + 1,
        captured_at=_iso_utc(captured_at),
        status=status,
        ready=ready,
        mode="remote_primary",
        required=required,
        latency_ms=latency_ms,
        consecutive_ok=1 if status == "ok" else 0,
        consecutive_fail=1 if status == "fail" else 0,
        detail=None if status == "ok" else "remote unavailable",
        probe=probe,
    )
    recent_samples = []
    for seq, previous_age_s in enumerate(sorted(previous_sample_ages_s, reverse=True), start=1):
        previous_captured_at = now - timedelta(seconds=previous_age_s)
        recent_samples.append(
            RemoteMemoryWatchdogSample(
                seq=seq,
                captured_at=_iso_utc(previous_captured_at),
                status=status,
                ready=ready,
                mode="remote_primary",
                required=required,
                latency_ms=latency_ms,
                consecutive_ok=1 if status == "ok" else 0,
                consecutive_fail=1 if status == "fail" else 0,
                detail=None if status == "ok" else "remote unavailable",
            )
        )
    recent_samples.append(sample)
    return RemoteMemoryWatchdogSnapshot(
        schema_version=1,
        started_at=_iso_utc(captured_at - timedelta(seconds=10)),
        updated_at=_iso_utc(captured_at),
        hostname="test-host",
        pid=pid,
        interval_s=1.0,
        history_limit=3600,
        sample_count=len(recent_samples),
        failure_count=0 if status == "ok" else 1,
        last_ok_at=_iso_utc(captured_at) if status == "ok" else None,
        last_failure_at=_iso_utc(captured_at) if status == "fail" else None,
        artifact_path="/tmp/remote-memory-watchdog.json",
        current=sample,
        recent_samples=tuple(recent_samples),
        heartbeat_at=_iso_utc(heartbeat_at),
        probe_inflight=probe_inflight,
        probe_started_at=_iso_utc(now - timedelta(seconds=probe_age_s or 0.0)) if probe_inflight else None,
        probe_age_s=probe_age_s,
    )


class RequiredRemoteWatchdogSnapshotTests(unittest.TestCase):
    def test_default_watchdog_recovery_starter_delegates_to_companion_helper(self) -> None:
        config = TwinrConfig(project_root="/tmp/twinr")

        with mock.patch(
            "twinr.ops.remote_memory_watchdog_companion.ensure_remote_memory_watchdog_process",
            return_value=4321,
        ) as ensure_process:
            owner = _default_watchdog_recovery_starter(config, "/tmp/twinr/.env")

        self.assertEqual(owner, 4321)
        ensure_process.assert_called_once()
        self.assertEqual(ensure_process.call_args.kwargs["env_file"], "/tmp/twinr/.env")

    def test_assess_accepts_recent_ok_snapshot_with_live_pid(self) -> None:
        now = datetime.now(timezone.utc)
        snapshot = _build_snapshot(now=now, pid=os.getpid(), age_s=2.0)
        config = TwinrConfig()

        assessment = assess_required_remote_watchdog_snapshot(
            config,
            now_wall=now,
            store=_FakeStore(snapshot),
        )

        self.assertTrue(assessment.ready)
        self.assertTrue(assessment.pid_alive)
        self.assertEqual(assessment.sample_status, "ok")

    def test_assess_ignores_create_time_drift_when_start_ticks_match(self) -> None:
        now = datetime.now(timezone.utc)
        config = TwinrConfig()
        expected_start_ticks = 123456
        expected_create_time_s = 1_775_405_396.65
        current_create_time_s = expected_create_time_s + 56.0
        snapshot = replace(
            _build_snapshot(now=now, pid=os.getpid(), age_s=2.0),
            pid_starttime_ticks=expected_start_ticks,
            pid_create_time_s=expected_create_time_s,
        )

        with mock.patch(
            "twinr.agent.workflows.required_remote_snapshot._current_boot_id",
            return_value=None,
        ), mock.patch(
            "twinr.agent.workflows.required_remote_snapshot._read_proc_stat_start_ticks",
            return_value=expected_start_ticks,
        ), mock.patch(
            "twinr.agent.workflows.required_remote_snapshot._read_proc_create_time_s",
            return_value=current_create_time_s,
        ):
            assessment = assess_required_remote_watchdog_snapshot(
                config,
                now_wall=now,
                store=_FakeStore(snapshot),
            )

        self.assertTrue(assessment.ready)
        self.assertTrue(assessment.pid_alive)
        self.assertEqual(assessment.sample_status, "ok")

    def test_assess_still_uses_create_time_when_start_ticks_are_missing(self) -> None:
        now = datetime.now(timezone.utc)
        config = TwinrConfig()
        expected_create_time_s = 1_775_405_396.65
        current_create_time_s = expected_create_time_s + 56.0
        snapshot = replace(
            _build_snapshot(now=now, pid=os.getpid(), age_s=2.0),
            pid_starttime_ticks=None,
            pid_create_time_s=expected_create_time_s,
        )

        with mock.patch(
            "twinr.agent.workflows.required_remote_snapshot._current_boot_id",
            return_value=None,
        ), mock.patch(
            "twinr.agent.workflows.required_remote_snapshot._read_proc_create_time_s",
            return_value=current_create_time_s,
        ):
            assessment = assess_required_remote_watchdog_snapshot(
                config,
                now_wall=now,
                store=_FakeStore(snapshot),
            )

        self.assertFalse(assessment.ready)
        self.assertIn("create-time attestation", assessment.detail)

    def test_assess_rejects_non_archive_safe_ok_snapshot(self) -> None:
        now = datetime.now(timezone.utc)
        snapshot = _build_snapshot(
            now=now,
            pid=os.getpid(),
            age_s=2.0,
            probe={
                "warm_result": {
                    "archive_safe": False,
                    "health_tier": "ready",
                    "proof_contract": {
                        "contract_id": "configured_namespace_current_only_readiness",
                    },
                }
            },
        )
        config = TwinrConfig(long_term_memory_remote_watchdog_probe_mode="archive_inclusive")

        assessment = assess_required_remote_watchdog_snapshot(
            config,
            now_wall=now,
            store=_FakeStore(snapshot),
        )

        self.assertFalse(assessment.ready)
        self.assertFalse(assessment.snapshot_stale)
        self.assertIn("archive-safe", assessment.detail.lower())

    def test_assess_accepts_current_only_ok_snapshot_when_current_only_contract_is_configured(self) -> None:
        now = datetime.now(timezone.utc)
        snapshot = _build_snapshot(
            now=now,
            pid=os.getpid(),
            age_s=2.0,
            probe={
                "warm_result": {
                    "archive_safe": False,
                    "health_tier": "ready",
                    "proof_contract": {
                        "contract_id": "configured_namespace_current_only_readiness",
                    },
                }
            },
        )
        config = TwinrConfig(long_term_memory_remote_watchdog_probe_mode="current_only")

        assessment = assess_required_remote_watchdog_snapshot(
            config,
            now_wall=now,
            store=_FakeStore(snapshot),
        )

        self.assertTrue(assessment.ready)
        self.assertEqual(assessment.detail, "ok")

    def test_assess_accepts_current_only_probe_despite_remote_status_false_when_watchdog_artifact_defaults_it(
        self,
    ) -> None:
        now = datetime.now(timezone.utc)
        snapshot = _build_snapshot(
            now=now,
            pid=os.getpid(),
            age_s=2.0,
            probe={
                "remote_status": {
                    "ready": False,
                    "detail": "ChonkyDB instance responded but is not ready.",
                },
                "warm_result": {
                    "archive_safe": False,
                    "health_tier": "ready",
                    "proof_contract": {
                        "contract_id": "configured_namespace_current_only_readiness",
                    },
                },
            },
        )
        config = TwinrConfig(long_term_memory_remote_runtime_check_mode="watchdog_artifact")

        assessment = assess_required_remote_watchdog_snapshot(
            config,
            now_wall=now,
            store=_FakeStore(snapshot),
        )

        self.assertTrue(assessment.ready)
        self.assertEqual(assessment.detail, "ok")

    def test_assess_accepts_ok_snapshot_when_archive_safe_probe_proves_ready_despite_remote_status_false(self) -> None:
        now = datetime.now(timezone.utc)
        snapshot = _build_snapshot(
            now=now,
            pid=os.getpid(),
            age_s=2.0,
            probe={
                "remote_status": {
                    "ready": False,
                    "detail": "ChonkyDB instance responded but is not ready.",
                },
                "warm_result": {
                    "archive_safe": True,
                    "health_tier": "ready",
                },
            },
        )
        config = TwinrConfig()

        assessment = assess_required_remote_watchdog_snapshot(
            config,
            now_wall=now,
            store=_FakeStore(snapshot),
        )

        self.assertTrue(assessment.ready)
        self.assertFalse(assessment.snapshot_stale)
        self.assertEqual(assessment.detail, "ok")

    def test_assess_rejects_missing_snapshot(self) -> None:
        config = TwinrConfig()

        assessment = assess_required_remote_watchdog_snapshot(
            config,
            now_wall=datetime.now(timezone.utc),
            store=_FakeStore(None),
        )

        self.assertFalse(assessment.ready)
        self.assertIn("missing", assessment.detail.lower())

    def test_assess_rejects_dead_watchdog_pid(self) -> None:
        now = datetime.now(timezone.utc)
        snapshot = _build_snapshot(now=now, pid=999999, age_s=1.0)
        config = TwinrConfig()

        assessment = assess_required_remote_watchdog_snapshot(
            config,
            now_wall=now,
            store=_FakeStore(snapshot),
        )

        self.assertFalse(assessment.ready)
        self.assertIn("not alive", assessment.detail.lower())

    def test_assess_rejects_stale_snapshot(self) -> None:
        now = datetime.now(timezone.utc)
        snapshot = _build_snapshot(now=now, pid=os.getpid(), age_s=65.0, latency_ms=18000.0)
        config = TwinrConfig()

        assessment = assess_required_remote_watchdog_snapshot(
            config,
            now_wall=now,
            store=_FakeStore(snapshot),
        )

        self.assertFalse(assessment.ready)
        self.assertIn("stale", assessment.detail.lower())

    def test_assess_accepts_future_dated_wall_timestamps_when_monotonic_fields_are_trusted(self) -> None:
        now = datetime(2026, 4, 4, 17, 3, 3, tzinfo=timezone.utc)
        sample = RemoteMemoryWatchdogSample(
            seq=1,
            captured_at="2026-04-04T17:03:10Z",
            status="ok",
            ready=True,
            mode="remote_primary",
            required=True,
            latency_ms=1200.0,
            consecutive_ok=1,
            consecutive_fail=0,
            captured_monotonic_ns=1_000_000_000,
            detail=None,
        )
        snapshot = RemoteMemoryWatchdogSnapshot(
            schema_version=1,
            started_at="2026-04-04T17:02:00Z",
            updated_at="2026-04-04T17:03:10Z",
            hostname="picarx",
            pid=os.getpid(),
            interval_s=1.0,
            history_limit=3600,
            sample_count=1,
            failure_count=0,
            last_ok_at="2026-04-04T17:03:10Z",
            last_failure_at=None,
            artifact_path="/tmp/remote-memory-watchdog.json",
            current=sample,
            recent_samples=(sample,),
            updated_monotonic_ns=1_000_000_000,
            heartbeat_at="2026-04-04T17:03:11Z",
            heartbeat_monotonic_ns=2_000_000_000,
            boot_id="boot-123",
            pid_starttime_ticks=None,
            pid_create_time_s=None,
            probe_inflight=False,
            probe_started_at=None,
            probe_started_monotonic_ns=None,
            probe_age_s=None,
        )
        config = TwinrConfig()

        with mock.patch(
            "twinr.agent.workflows.required_remote_snapshot._current_boot_id",
            return_value="boot-123",
        ), mock.patch(
            "twinr.agent.workflows.required_remote_snapshot.time.monotonic_ns",
            return_value=4_000_000_000,
        ):
            assessment = assess_required_remote_watchdog_snapshot(
                config,
                now_wall=now,
                store=_FakeStore(snapshot),
            )

        self.assertTrue(assessment.ready)
        self.assertFalse(assessment.snapshot_stale)
        self.assertEqual(assessment.sample_status, "ok")
        self.assertEqual(assessment.sample_age_s, 3.0)
        self.assertEqual(assessment.heartbeat_age_s, 2.0)

    def test_assess_accepts_recent_heartbeat_between_steady_state_samples(self) -> None:
        now = datetime(2026, 3, 20, 17, 35, 0, tzinfo=timezone.utc)
        snapshot = _build_snapshot(
            now=now,
            pid=os.getpid(),
            age_s=7.4,
            latency_ms=1015.4,
            heartbeat_age_s=1.0,
            probe_inflight=False,
        )
        config = TwinrConfig()

        assessment = assess_required_remote_watchdog_snapshot(
            config,
            now_wall=now,
            store=_FakeStore(snapshot),
        )

        self.assertTrue(assessment.ready)
        self.assertFalse(assessment.snapshot_stale)
        self.assertGreater(assessment.sample_age_s or 0.0, assessment.max_sample_age_s)
        self.assertLess(assessment.heartbeat_age_s or 999.0, 5.0)

    def test_assess_rejects_heartbeat_bridge_when_last_sample_is_too_old(self) -> None:
        now = datetime(2026, 3, 20, 17, 35, 0, tzinfo=timezone.utc)
        snapshot = _build_snapshot(
            now=now,
            pid=os.getpid(),
            age_s=14.0,
            latency_ms=1015.4,
            heartbeat_age_s=1.0,
            probe_inflight=False,
        )
        config = TwinrConfig()

        assessment = assess_required_remote_watchdog_snapshot(
            config,
            now_wall=now,
            store=_FakeStore(snapshot),
        )

        self.assertFalse(assessment.ready)
        self.assertTrue(assessment.snapshot_stale)
        self.assertIn("stale", assessment.detail.lower())

    def test_assess_accepts_steady_state_heartbeat_that_exceeds_base_budget_but_not_bridge_budget(
        self,
    ) -> None:
        now = datetime(2026, 3, 20, 19, 29, 46, 24442, tzinfo=timezone.utc)
        snapshot = _build_snapshot(
            now=now,
            pid=os.getpid(),
            age_s=9.024442,
            latency_ms=928.3,
            heartbeat_age_s=7.024442,
            probe_inflight=False,
        )
        config = TwinrConfig()

        assessment = assess_required_remote_watchdog_snapshot(
            config,
            now_wall=now,
            store=_FakeStore(snapshot),
        )

        self.assertTrue(assessment.ready)
        self.assertFalse(assessment.snapshot_stale)
        self.assertGreater(assessment.sample_age_s or 0.0, assessment.max_sample_age_s)

    def test_assess_rejects_steady_state_heartbeat_when_bridge_budget_is_exceeded(self) -> None:
        now = datetime(2026, 3, 20, 19, 29, 46, 24442, tzinfo=timezone.utc)
        snapshot = _build_snapshot(
            now=now,
            pid=os.getpid(),
            age_s=9.024442,
            latency_ms=928.3,
            heartbeat_age_s=12.2,
            probe_inflight=False,
        )
        config = TwinrConfig()

        assessment = assess_required_remote_watchdog_snapshot(
            config,
            now_wall=now,
            store=_FakeStore(snapshot),
        )

        self.assertFalse(assessment.ready)
        self.assertTrue(assessment.snapshot_stale)
        self.assertIn("stale", assessment.detail.lower())

    def test_assess_accepts_recent_heartbeat_when_recent_sample_cycle_is_longer(self) -> None:
        now = datetime(2026, 3, 20, 18, 40, 0, tzinfo=timezone.utc)
        snapshot = _build_snapshot(
            now=now,
            pid=os.getpid(),
            age_s=16.7,
            latency_ms=888.8,
            heartbeat_age_s=1.0,
            probe_inflight=False,
            previous_sample_ages_s=(33.8,),
        )
        config = TwinrConfig()

        assessment = assess_required_remote_watchdog_snapshot(
            config,
            now_wall=now,
            store=_FakeStore(snapshot),
        )

        self.assertTrue(assessment.ready)
        self.assertFalse(assessment.snapshot_stale)

    def test_assess_rejects_recent_heartbeat_when_even_recent_sample_cycle_is_exceeded(self) -> None:
        now = datetime(2026, 3, 20, 18, 40, 0, tzinfo=timezone.utc)
        snapshot = _build_snapshot(
            now=now,
            pid=os.getpid(),
            age_s=24.5,
            latency_ms=888.8,
            heartbeat_age_s=1.0,
            probe_inflight=False,
            previous_sample_ages_s=(41.6,),
        )
        config = TwinrConfig()

        assessment = assess_required_remote_watchdog_snapshot(
            config,
            now_wall=now,
            store=_FakeStore(snapshot),
        )

        self.assertFalse(assessment.ready)
        self.assertTrue(assessment.snapshot_stale)
        self.assertIn("stale", assessment.detail.lower())

    def test_assess_accepts_pi_second_resolution_steady_state_boundary(self) -> None:
        now = datetime(2026, 3, 20, 19, 47, 37, 217594, tzinfo=timezone.utc)
        snapshot = _build_snapshot(
            now=now,
            pid=os.getpid(),
            age_s=21.217594,
            latency_ms=1864.0,
            heartbeat_age_s=8.217594,
            probe_inflight=False,
            previous_sample_ages_s=(78.217594, 59.217594, 40.217594),
        )
        config = TwinrConfig()

        assessment = assess_required_remote_watchdog_snapshot(
            config,
            now_wall=now,
            store=_FakeStore(snapshot),
        )

        self.assertTrue(assessment.ready)
        self.assertFalse(assessment.snapshot_stale)

    def test_assess_accepts_live_inflight_probe_with_fresh_heartbeat(self) -> None:
        now = datetime.now(timezone.utc)
        snapshot = _build_snapshot(
            now=now,
            pid=os.getpid(),
            age_s=55.0,
            latency_ms=18000.0,
            heartbeat_age_s=1.0,
            probe_inflight=True,
            probe_age_s=24.0,
        )
        config = TwinrConfig()

        assessment = assess_required_remote_watchdog_snapshot(
            config,
            now_wall=now,
            store=_FakeStore(snapshot),
        )

        self.assertTrue(assessment.ready)
        self.assertFalse(assessment.snapshot_stale)
        self.assertTrue(assessment.probe_inflight)
        self.assertLess(assessment.heartbeat_age_s or 999.0, 5.0)

    def test_assess_accepts_pi_boundary_inflight_probe_with_recent_heartbeat(self) -> None:
        now = datetime(2026, 3, 20, 13, 14, 22, 565113, tzinfo=timezone.utc)
        sample_at = datetime(2026, 3, 20, 13, 14, 14, tzinfo=timezone.utc)
        sample = RemoteMemoryWatchdogSample(
            seq=1,
            captured_at=_iso_utc(sample_at),
            status="ok",
            ready=True,
            mode="remote_primary",
            required=True,
            latency_ms=1015.4,
            consecutive_ok=1,
            consecutive_fail=0,
            detail=None,
        )
        snapshot = RemoteMemoryWatchdogSnapshot(
            schema_version=1,
            started_at=_iso_utc(sample_at - timedelta(seconds=10)),
            updated_at=_iso_utc(sample_at),
            hostname="test-host",
            pid=os.getpid(),
            interval_s=1.0,
            history_limit=3600,
            sample_count=1,
            failure_count=0,
            last_ok_at=_iso_utc(sample_at),
            last_failure_at=None,
            artifact_path="/tmp/remote-memory-watchdog.json",
            current=sample,
            recent_samples=(sample,),
            heartbeat_at="2026-03-20T13:14:17Z",
            probe_inflight=True,
            probe_started_at="2026-03-20T13:14:14Z",
            probe_age_s=5.0,
        )
        config = TwinrConfig()

        assessment = assess_required_remote_watchdog_snapshot(
            config,
            now_wall=now,
            store=_FakeStore(snapshot),
        )

        self.assertTrue(assessment.ready)
        self.assertFalse(assessment.snapshot_stale)
        self.assertGreater(assessment.sample_age_s or 0.0, assessment.max_sample_age_s)
        self.assertGreater(assessment.heartbeat_age_s or 0.0, 5.0)

    def test_assess_accepts_live_pi_inflight_probe_with_bounded_heartbeat_bridge(self) -> None:
        now = datetime(2026, 3, 20, 19, 36, 49, 2982, tzinfo=timezone.utc)
        snapshot = _build_snapshot(
            now=now,
            pid=os.getpid(),
            age_s=27.002982,
            latency_ms=4933.1,
            heartbeat_age_s=7.002982,
            probe_inflight=True,
            probe_age_s=0.0,
            previous_sample_ages_s=(83.002982, 64.002982, 44.002982),
        )
        config = TwinrConfig()

        assessment = assess_required_remote_watchdog_snapshot(
            config,
            now_wall=now,
            store=_FakeStore(snapshot),
        )

        self.assertTrue(assessment.ready)
        self.assertFalse(assessment.snapshot_stale)
        self.assertTrue(assessment.probe_inflight)
        self.assertGreater(assessment.sample_age_s or 0.0, assessment.max_sample_age_s)

    def test_assess_rejects_inflight_probe_when_heartbeat_is_too_old(self) -> None:
        now = datetime.now(timezone.utc)
        snapshot = _build_snapshot(
            now=now,
            pid=os.getpid(),
            age_s=9.0,
            latency_ms=1015.4,
            heartbeat_age_s=181.2,
            probe_inflight=True,
            probe_age_s=181.2,
        )
        config = TwinrConfig()

        assessment = assess_required_remote_watchdog_snapshot(
            config,
            now_wall=now,
            store=_FakeStore(snapshot),
        )

        self.assertFalse(assessment.ready)
        self.assertTrue(assessment.snapshot_stale)
        self.assertIn("stale", assessment.detail.lower())

    def test_assess_accepts_synthetic_timeout_sample_while_recent_long_success_still_bridges(self) -> None:
        now = datetime(2026, 4, 4, 11, 15, 36, tzinfo=timezone.utc)
        recent_ok = RemoteMemoryWatchdogSample(
            seq=1,
            captured_at=_iso_utc(now - timedelta(seconds=75.0)),
            status="ok",
            ready=True,
            mode="remote_primary",
            required=True,
            latency_ms=150997.2,
            consecutive_ok=1,
            consecutive_fail=0,
            detail=None,
        )
        timed_out = RemoteMemoryWatchdogSample(
            seq=2,
            captured_at=_iso_utc(now - timedelta(seconds=1.0)),
            status="fail",
            ready=False,
            mode="remote_primary",
            required=True,
            latency_ms=15374.9,
            consecutive_ok=0,
            consecutive_fail=1,
            detail="Remote readiness probe exceeded 15.0s and is assumed stuck.",
            probe={
                "watchdog_timeout": {
                    "probe_started_at": _iso_utc(now - timedelta(seconds=55.0)),
                    "probe_age_s": 55.0,
                    "probe_timeout_s": 15.0,
                }
            },
        )
        snapshot = RemoteMemoryWatchdogSnapshot(
            schema_version=1,
            started_at=_iso_utc(now - timedelta(seconds=120.0)),
            updated_at=_iso_utc(now - timedelta(seconds=1.0)),
            hostname="test-host",
            pid=os.getpid(),
            interval_s=1.0,
            history_limit=3600,
            sample_count=2,
            failure_count=1,
            last_ok_at=recent_ok.captured_at,
            last_failure_at=timed_out.captured_at,
            artifact_path="/tmp/remote-memory-watchdog.json",
            current=timed_out,
            recent_samples=(recent_ok, timed_out),
            heartbeat_at=_iso_utc(now - timedelta(seconds=1.0)),
            probe_inflight=True,
            probe_started_at=_iso_utc(now - timedelta(seconds=55.0)),
            probe_age_s=55.0,
        )

        assessment = assess_required_remote_watchdog_snapshot(
            TwinrConfig(),
            now_wall=now,
            store=_FakeStore(snapshot),
        )

        self.assertTrue(assessment.ready)
        self.assertFalse(assessment.snapshot_stale)
        self.assertEqual(assessment.sample_status, "fail")
        self.assertTrue(assessment.probe_inflight)
        self.assertIn("still inflight", assessment.detail.lower())

    def test_assess_accepts_synthetic_timeout_sample_under_startup_probe_budget_without_recent_long_success(
        self,
    ) -> None:
        """Pi repro 2026-04-04: the same timed-out probe later succeeded after 143.3s."""

        now = datetime(2026, 4, 4, 12, 51, 46, tzinfo=timezone.utc)
        recent_ok = RemoteMemoryWatchdogSample(
            seq=25782,
            captured_at="2026-04-04T12:50:32Z",
            status="ok",
            ready=True,
            mode="remote_primary",
            required=True,
            latency_ms=5228.2,
            consecutive_ok=1,
            consecutive_fail=0,
            detail=None,
        )
        timed_out = RemoteMemoryWatchdogSample(
            seq=25783,
            captured_at="2026-04-04T12:50:55Z",
            status="fail",
            ready=False,
            mode="remote_primary",
            required=True,
            latency_ms=15794.8,
            consecutive_ok=0,
            consecutive_fail=1,
            detail="Remote readiness probe exceeded 15.0s and is assumed stuck.",
            probe={
                "watchdog_timeout": {
                    "probe_started_at": "2026-04-04T12:50:39Z",
                    "probe_age_s": 67.0,
                    "probe_timeout_s": 15.0,
                }
            },
        )
        snapshot = RemoteMemoryWatchdogSnapshot(
            schema_version=1,
            started_at="2026-04-04T12:30:00Z",
            updated_at=timed_out.captured_at,
            hostname="picarx",
            pid=os.getpid(),
            interval_s=1.0,
            history_limit=3600,
            sample_count=25783,
            failure_count=741,
            last_ok_at=recent_ok.captured_at,
            last_failure_at=timed_out.captured_at,
            artifact_path="/tmp/remote-memory-watchdog.json",
            current=timed_out,
            recent_samples=(recent_ok, timed_out),
            heartbeat_at="2026-04-04T12:51:46Z",
            probe_inflight=True,
            probe_started_at="2026-04-04T12:50:39Z",
            probe_age_s=67.0,
        )

        assessment = assess_required_remote_watchdog_snapshot(
            TwinrConfig(),
            now_wall=now,
            store=_FakeStore(snapshot),
        )

        self.assertTrue(assessment.ready)
        self.assertFalse(assessment.snapshot_stale)
        self.assertEqual(assessment.sample_status, "fail")
        self.assertTrue(assessment.probe_inflight)
        self.assertIn("still inflight", assessment.detail.lower())

    def test_assess_rejects_synthetic_timeout_sample_once_inflight_probe_exceeds_startup_probe_budget(
        self,
    ) -> None:
        now = datetime(2026, 4, 4, 12, 53, 46, tzinfo=timezone.utc)
        recent_ok = RemoteMemoryWatchdogSample(
            seq=25782,
            captured_at="2026-04-04T12:50:32Z",
            status="ok",
            ready=True,
            mode="remote_primary",
            required=True,
            latency_ms=5228.2,
            consecutive_ok=1,
            consecutive_fail=0,
            detail=None,
        )
        timed_out = RemoteMemoryWatchdogSample(
            seq=25783,
            captured_at="2026-04-04T12:50:55Z",
            status="fail",
            ready=False,
            mode="remote_primary",
            required=True,
            latency_ms=15794.8,
            consecutive_ok=0,
            consecutive_fail=1,
            detail="Remote readiness probe exceeded 15.0s and is assumed stuck.",
            probe={
                "watchdog_timeout": {
                    "probe_started_at": "2026-04-04T12:50:39Z",
                    "probe_age_s": 187.0,
                    "probe_timeout_s": 15.0,
                }
            },
        )
        snapshot = RemoteMemoryWatchdogSnapshot(
            schema_version=1,
            started_at="2026-04-04T12:30:00Z",
            updated_at=timed_out.captured_at,
            hostname="picarx",
            pid=os.getpid(),
            interval_s=1.0,
            history_limit=3600,
            sample_count=25783,
            failure_count=741,
            last_ok_at=recent_ok.captured_at,
            last_failure_at=timed_out.captured_at,
            artifact_path="/tmp/remote-memory-watchdog.json",
            current=timed_out,
            recent_samples=(recent_ok, timed_out),
            heartbeat_at="2026-04-04T12:53:46Z",
            probe_inflight=True,
            probe_started_at="2026-04-04T12:50:39Z",
            probe_age_s=187.0,
        )

        assessment = assess_required_remote_watchdog_snapshot(
            TwinrConfig(),
            now_wall=now,
            store=_FakeStore(snapshot),
        )

        self.assertFalse(assessment.ready)
        self.assertTrue(assessment.snapshot_stale)
        self.assertEqual(assessment.sample_status, "fail")
        self.assertTrue(assessment.probe_inflight)
        self.assertIn("stale", assessment.detail.lower())

    def test_ensure_raises_long_term_remote_unavailable_when_snapshot_fails(self) -> None:
        now = datetime.now(timezone.utc)
        snapshot = _build_snapshot(now=now, pid=os.getpid(), status="fail", ready=False)
        config = TwinrConfig()

        with self.assertRaises(LongTermRemoteUnavailableError):
            ensure_required_remote_watchdog_snapshot_ready(
                config,
                store=_FakeStore(snapshot),
            )

    def test_ensure_waits_briefly_for_starting_watchdog_snapshot(self) -> None:
        now = datetime.now(timezone.utc)
        starting = _build_snapshot(
            now=now,
            pid=os.getpid(),
            status="starting",
            ready=False,
            probe_inflight=True,
            probe_age_s=0.0,
        )
        ok = _build_snapshot(now=now, pid=os.getpid(), status="ok", ready=True)
        config = TwinrConfig(long_term_memory_remote_watchdog_startup_wait_s=1.0)

        with mock.patch("twinr.agent.workflows.required_remote_snapshot.time.sleep", return_value=None):
            assessment = ensure_required_remote_watchdog_snapshot_ready(
                config,
                store=_SequencedStore((starting, ok)),
            )

        self.assertTrue(assessment.ready)
        self.assertEqual(assessment.sample_status, "ok")

    def test_ensure_waits_for_recently_restarted_watchdog_after_one_fresh_fail_sample(self) -> None:
        now = datetime.now(timezone.utc)
        failing = _build_snapshot(
            now=now,
            pid=os.getpid(),
            status="fail",
            ready=False,
            age_s=1.0,
            heartbeat_age_s=1.0,
        )
        ok = _build_snapshot(
            now=now,
            pid=os.getpid(),
            status="ok",
            ready=True,
            age_s=1.0,
            heartbeat_age_s=1.0,
        )
        config = TwinrConfig(long_term_memory_remote_watchdog_startup_wait_s=15.0)

        with mock.patch("twinr.agent.workflows.required_remote_snapshot.time.sleep", return_value=None):
            assessment = ensure_required_remote_watchdog_snapshot_ready(
                config,
                store=_SequencedStore((failing, ok)),
            )

        self.assertTrue(assessment.ready)
        self.assertEqual(assessment.sample_status, "ok")

    def test_ensure_recovers_dead_external_watchdog_owner_before_waiting_for_ready_snapshot(self) -> None:
        now = datetime.now(timezone.utc)
        dead = _build_snapshot(now=now, pid=999999, age_s=1.0)
        starting = _build_snapshot(
            now=now,
            pid=os.getpid(),
            status="starting",
            ready=False,
            probe_inflight=True,
            probe_age_s=0.0,
        )
        ok = _build_snapshot(now=now, pid=os.getpid(), status="ok", ready=True)
        config = TwinrConfig(long_term_memory_remote_watchdog_startup_wait_s=1.0)
        starter_calls: list[str] = []

        def _starter(_config: TwinrConfig, env_file: str) -> int | None:
            starter_calls.append(env_file)
            return 4321

        with mock.patch("twinr.agent.workflows.required_remote_snapshot.time.sleep", return_value=None):
            assessment = ensure_required_remote_watchdog_snapshot_ready(
                config,
                store=_SequencedStore((dead, starting, ok)),
                env_file="/twinr/.env",
                recovery_starter=_starter,
            )

        self.assertTrue(assessment.ready)
        self.assertEqual(assessment.sample_status, "ok")
        self.assertEqual(starter_calls, ["/twinr/.env"])


class TwinrConfigRemoteRuntimeCheckModeTests(unittest.TestCase):
    def test_watchdog_probe_mode_defaults_to_auto(self) -> None:
        self.assertEqual(TwinrConfig().long_term_memory_remote_watchdog_probe_mode, "auto")

    def test_pi_runtime_defaults_to_watchdog_artifact_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_twinr = Path(temp_dir) / "twinr"
            fake_twinr.mkdir(parents=True, exist_ok=True)
            env_path = fake_twinr / ".env"
            env_path.write_text("", encoding="utf-8")

            real_resolve = Path.resolve

            def _fake_resolve(path_obj: Path, *args, **kwargs):
                if path_obj == env_path.parent:
                    return Path("/twinr")
                return real_resolve(path_obj, *args, **kwargs)

            with mock.patch("pathlib.Path.resolve", _fake_resolve):
                config = TwinrConfig.from_env(env_path)

        self.assertEqual(config.long_term_memory_remote_runtime_check_mode, "watchdog_artifact")
        self.assertEqual(config.long_term_memory_remote_watchdog_probe_mode, "auto")
        self.assertEqual(_required_watchdog_probe_mode(config), "current_only")

    def test_explicit_watchdog_probe_mode_is_loaded_from_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "TWINR_LONG_TERM_MEMORY_REMOTE_WATCHDOG_PROBE_MODE=archive_inclusive\n",
                encoding="utf-8",
            )

            config = TwinrConfig.from_env(env_path)

        self.assertEqual(config.long_term_memory_remote_watchdog_probe_mode, "archive_inclusive")
        self.assertEqual(_required_watchdog_probe_mode(config), "archive_inclusive")
