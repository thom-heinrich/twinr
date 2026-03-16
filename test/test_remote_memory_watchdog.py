from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError
from twinr.ops import TwinrOpsEventStore
from twinr.ops.remote_memory_watchdog import RemoteMemoryWatchdog, RemoteMemoryWatchdogStore


class _SequencedRemoteService:
    def __init__(self, states: list[str]) -> None:
        self._states = list(states)
        self._index = 0
        self.shutdown_calls = 0

    def remote_required(self) -> bool:
        return True

    def remote_status(self):
        state = self._states[min(self._index, len(self._states) - 1)]
        if state == "ok":
            return SimpleNamespace(mode="remote_primary", ready=True, detail=None)
        return SimpleNamespace(mode="remote_primary", ready=False, detail="remote unavailable")

    def ensure_remote_ready(self) -> None:
        state = self._states[min(self._index, len(self._states) - 1)]
        self._index += 1
        if state != "ok":
            raise LongTermRemoteUnavailableError("remote unavailable")

    def shutdown(self, *, timeout_s: float = 2.0) -> None:
        del timeout_s
        self.shutdown_calls += 1


class _SlowReadyRemoteService(_SequencedRemoteService):
    def __init__(self, *, sleep_s: float) -> None:
        super().__init__(["ok"])
        self._sleep_s = sleep_s

    def ensure_remote_ready(self) -> None:
        time.sleep(self._sleep_s)
        super().ensure_remote_ready()


class RemoteMemoryWatchdogTests(unittest.TestCase):
    def test_store_load_roundtrips_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(root / "state" / "runtime-state.json"),
            )
            store = RemoteMemoryWatchdogStore.from_config(config)
            store.save(
                RemoteMemoryWatchdogSnapshot(
                    schema_version=1,
                    started_at="2026-03-16T18:00:00Z",
                    updated_at="2026-03-16T18:00:05Z",
                    hostname="picarx",
                    pid=123,
                    interval_s=1.0,
                    history_limit=3600,
                    sample_count=5,
                    failure_count=1,
                    last_ok_at="2026-03-16T18:00:05Z",
                    last_failure_at="2026-03-16T17:59:00Z",
                    artifact_path=str(store.path),
                    current=RemoteMemoryWatchdogSample(
                        seq=5,
                        captured_at="2026-03-16T18:00:05Z",
                        status="ok",
                        ready=True,
                        mode="remote_primary",
                        required=True,
                        latency_ms=42.0,
                        consecutive_ok=2,
                        consecutive_fail=0,
                        detail=None,
                    ),
                    recent_samples=(),
                )
            )

            loaded = store.load()

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.current.status, "ok")
        self.assertEqual(loaded.sample_count, 5)
        self.assertEqual(loaded.failure_count, 1)

    def test_probe_once_persists_rolling_snapshot_and_transition_event(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(root / "state" / "runtime-state.json"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_watchdog_history_limit=2,
            )
            event_store = TwinrOpsEventStore.from_config(config)
            service = _SequencedRemoteService(["ok", "fail", "ok"])
            emitted: list[str] = []
            watchdog = RemoteMemoryWatchdog(
                config=config,
                service_factory=lambda: service,
                store=RemoteMemoryWatchdogStore.from_config(config),
                event_store=event_store,
                emit=emitted.append,
            )

            first = watchdog.probe_once()
            second = watchdog.probe_once()
            third = watchdog.probe_once()
            watchdog.close()

            payload = json.loads(watchdog.artifact_path.read_text(encoding="utf-8"))
            status_events = [
                entry
                for entry in event_store.tail(limit=20)
                if entry["event"] == "remote_memory_watchdog_status_changed"
            ]

        self.assertEqual(first.current.status, "ok")
        self.assertEqual(second.current.status, "fail")
        self.assertEqual(third.current.status, "ok")
        self.assertEqual(payload["current"]["status"], "ok")
        self.assertEqual(payload["failure_count"], 1)
        self.assertEqual(len(payload["recent_samples"]), 2)
        self.assertEqual(payload["recent_samples"][0]["status"], "fail")
        self.assertEqual(payload["recent_samples"][1]["status"], "ok")
        self.assertEqual(len(status_events), 3)
        self.assertEqual(json.loads(emitted[-1])["status"], "ok")
        self.assertEqual(service.shutdown_calls, 1)

    def test_probe_once_marks_disabled_when_remote_is_not_required(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(root / "state" / "runtime-state.json"),
            )

            class _DisabledRemoteService:
                def remote_required(self) -> bool:
                    return False

                def remote_status(self):
                    return SimpleNamespace(mode="disabled", ready=False, detail="disabled")

                def ensure_remote_ready(self) -> None:
                    raise AssertionError("disabled remote should not be probed")

                def shutdown(self, *, timeout_s: float = 2.0) -> None:
                    del timeout_s

            watchdog = RemoteMemoryWatchdog(
                config=config,
                service_factory=_DisabledRemoteService,
                store=RemoteMemoryWatchdogStore.from_config(config),
                event_store=TwinrOpsEventStore.from_config(config),
                emit=lambda _line: None,
            )

            snapshot = watchdog.probe_once()

        self.assertEqual(snapshot.current.status, "disabled")
        self.assertFalse(snapshot.current.ready)

    def test_run_emits_per_second_heartbeat_while_deep_probe_is_inflight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(root / "state" / "runtime-state.json"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_watchdog_interval_s=0.01,
            )
            emitted: list[str] = []
            watchdog = RemoteMemoryWatchdog(
                config=config,
                service_factory=lambda: _SlowReadyRemoteService(sleep_s=0.03),
                store=RemoteMemoryWatchdogStore.from_config(config),
                event_store=TwinrOpsEventStore.from_config(config),
                emit=emitted.append,
            )

            result = watchdog.run(duration_s=0.05)

        events = [json.loads(line)["event"] for line in emitted]
        self.assertEqual(result, 0)
        self.assertIn("remote_memory_watchdog_heartbeat", events)
        self.assertIn("remote_memory_watchdog_sample", events)


if __name__ == "__main__":
    unittest.main()
