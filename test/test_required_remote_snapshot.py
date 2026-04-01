from datetime import datetime, timedelta, timezone
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

    def test_assess_rejects_non_archive_safe_ok_snapshot(self) -> None:
        now = datetime.now(timezone.utc)
        snapshot = _build_snapshot(
            now=now,
            pid=os.getpid(),
            age_s=2.0,
            probe={
                "warm_result": {
                    "archive_safe": False,
                    "health_tier": "degraded",
                }
            },
        )
        config = TwinrConfig()

        assessment = assess_required_remote_watchdog_snapshot(
            config,
            now_wall=now,
            store=_FakeStore(snapshot),
        )

        self.assertFalse(assessment.ready)
        self.assertFalse(assessment.snapshot_stale)
        self.assertIn("archive-safe", assessment.detail.lower())

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
            heartbeat_age_s=13.2,
            probe_inflight=True,
            probe_age_s=13.2,
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
