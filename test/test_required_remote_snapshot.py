from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.agent.workflows.required_remote_snapshot import (
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

    def load(self):
        return self._snapshot


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
) -> RemoteMemoryWatchdogSnapshot:
    captured_at = now - timedelta(seconds=age_s)
    heartbeat_at = captured_at if heartbeat_age_s is None else now - timedelta(seconds=heartbeat_age_s)
    sample = RemoteMemoryWatchdogSample(
        seq=1,
        captured_at=_iso_utc(captured_at),
        status=status,
        ready=ready,
        mode="remote_primary",
        required=required,
        latency_ms=latency_ms,
        consecutive_ok=1 if status == "ok" else 0,
        consecutive_fail=1 if status == "fail" else 0,
        detail=None if status == "ok" else "remote unavailable",
    )
    return RemoteMemoryWatchdogSnapshot(
        schema_version=1,
        started_at=_iso_utc(captured_at - timedelta(seconds=10)),
        updated_at=_iso_utc(captured_at),
        hostname="test-host",
        pid=pid,
        interval_s=1.0,
        history_limit=3600,
        sample_count=1,
        failure_count=0 if status == "ok" else 1,
        last_ok_at=_iso_utc(captured_at) if status == "ok" else None,
        last_failure_at=_iso_utc(captured_at) if status == "fail" else None,
        artifact_path="/tmp/remote-memory-watchdog.json",
        current=sample,
        recent_samples=(sample,),
        heartbeat_at=_iso_utc(heartbeat_at),
        probe_inflight=probe_inflight,
        probe_started_at=_iso_utc(now - timedelta(seconds=probe_age_s or 0.0)) if probe_inflight else None,
        probe_age_s=probe_age_s,
    )


class RequiredRemoteWatchdogSnapshotTests(unittest.TestCase):
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

    def test_ensure_raises_long_term_remote_unavailable_when_snapshot_fails(self) -> None:
        now = datetime.now(timezone.utc)
        snapshot = _build_snapshot(now=now, pid=os.getpid(), status="fail", ready=False)
        config = TwinrConfig()

        with self.assertRaises(LongTermRemoteUnavailableError):
            ensure_required_remote_watchdog_snapshot_ready(
                config,
                store=_FakeStore(snapshot),
            )


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
