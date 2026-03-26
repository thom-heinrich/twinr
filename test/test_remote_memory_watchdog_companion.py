from pathlib import Path
import tempfile
import unittest
from unittest import mock

from twinr.agent.base_agent.config import TwinrConfig
from twinr.ops.remote_memory_watchdog_companion import ensure_remote_memory_watchdog_process
from twinr.ops.remote_memory_watchdog_state import (
    RemoteMemoryWatchdogStore,
    build_remote_memory_watchdog_bootstrap_snapshot,
)


def _build_config(root: Path) -> TwinrConfig:
    return TwinrConfig(
        project_root=str(root),
        long_term_memory_enabled=True,
        long_term_memory_mode="remote_primary",
        long_term_memory_remote_required=True,
    )


class RemoteMemoryWatchdogCompanionTests(unittest.TestCase):
    def test_existing_external_owner_reseeds_bootstrap_snapshot_when_artifact_pid_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _build_config(root)
            store = RemoteMemoryWatchdogStore.from_config(config)
            stale = build_remote_memory_watchdog_bootstrap_snapshot(
                config,
                pid=111,
                artifact_path=store.path,
                started_at="2026-03-26T10:00:00Z",
                captured_at="2026-03-26T10:00:00Z",
            )
            store.save(stale)
            emitted: list[str] = []

            with mock.patch(
                "twinr.ops.remote_memory_watchdog_companion.loop_lock_owner",
                return_value=222,
            ):
                owner = ensure_remote_memory_watchdog_process(
                    config,
                    env_file=root / ".env",
                    emit=emitted.append,
                )

            self.assertEqual(owner, 222)
            snapshot = store.load()
            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertEqual(snapshot.pid, 222)
            self.assertEqual(snapshot.current.status, "starting")
            self.assertEqual(snapshot.current.detail, "Remote memory watchdog is starting.")
            self.assertIn("remote_memory_watchdog=running:222", emitted)

    def test_spawned_external_owner_seeds_bootstrap_snapshot_before_first_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _build_config(root)
            store = RemoteMemoryWatchdogStore.from_config(config)
            emitted: list[str] = []

            with mock.patch(
                "twinr.ops.remote_memory_watchdog_companion.loop_lock_owner",
                side_effect=[None, 333],
            ), mock.patch(
                "twinr.ops.remote_memory_watchdog_companion.subprocess.Popen",
            ) as popen_mock, mock.patch(
                "twinr.ops.remote_memory_watchdog_companion.time.sleep",
                return_value=None,
            ):
                owner = ensure_remote_memory_watchdog_process(
                    config,
                    env_file=root / ".env",
                    emit=emitted.append,
                    startup_timeout_s=0.2,
                )

            self.assertEqual(owner, 333)
            popen_mock.assert_called_once()
            snapshot = store.load()
            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertEqual(snapshot.pid, 333)
            self.assertEqual(snapshot.current.status, "starting")
            self.assertTrue(snapshot.probe_inflight)
            self.assertEqual(emitted[:2], ["remote_memory_watchdog=spawned", "remote_memory_watchdog=ready:333"])
