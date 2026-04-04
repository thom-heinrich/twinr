from __future__ import annotations

import itertools
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import time
from typing import cast
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError
from twinr.ops.events import TwinrOpsEventStore
from twinr.ops.remote_memory_watchdog import (
    RemoteMemoryWatchdog,
    RemoteMemoryWatchdogSample,
    RemoteMemoryWatchdogSnapshot,
    RemoteMemoryWatchdogStore,
    build_remote_memory_watchdog_bootstrap_snapshot,
)


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


class _DeepProbeSequencedRemoteService:
    def __init__(self, states: list[str]) -> None:
        self._states = list(states)
        self._index = 0
        self.shutdown_calls = 0

    def remote_required(self) -> bool:
        return True

    def remote_status(self):
        return SimpleNamespace(mode="remote_primary", ready=True, detail=None)

    def ensure_remote_ready(self) -> None:
        state = self._states[min(self._index, len(self._states) - 1)]
        self._index += 1
        if state != "ok":
            raise LongTermRemoteUnavailableError("remote unavailable")

    def shutdown(self, *, timeout_s: float = 2.0) -> None:
        del timeout_s
        self.shutdown_calls += 1


class _AlwaysFailRemoteService:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def remote_required(self) -> bool:
        return True

    def remote_status(self):
        return SimpleNamespace(mode="remote_primary", ready=True, detail=None)

    def ensure_remote_ready(self) -> None:
        raise LongTermRemoteUnavailableError("remote unavailable")

    def shutdown(self, *, timeout_s: float = 2.0) -> None:
        del timeout_s
        self.shutdown_calls += 1


class _UnexpectedErrorRemoteService:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def remote_required(self) -> bool:
        return True

    def remote_status(self):
        return SimpleNamespace(mode="remote_primary", ready=True, detail=None)

    def ensure_remote_ready(self) -> None:
        raise RuntimeError("boom")

    def shutdown(self, *, timeout_s: float = 2.0) -> None:
        del timeout_s
        self.shutdown_calls += 1


class _CorrelatedFailRemoteService:
    def remote_required(self) -> bool:
        return True

    def remote_status(self):
        return SimpleNamespace(mode="remote_primary", ready=True, detail=None)

    def ensure_remote_ready(self) -> None:
        exc = LongTermRemoteUnavailableError(
            "Failed to persist fine-grained remote long-term memory items "
            "(request_id=ltw-test123, batch=1/1, items=51, bytes=518642)."
        )
        setattr(
            exc,
            "remote_write_context",
            {
                "snapshot_kind": "objects",
                "operation": "store_records_bulk",
                "request_correlation_id": "ltw-test123",
                "batch_index": 1,
                "batch_count": 1,
                "request_item_count": 51,
                "request_bytes": 518642,
            },
        )
        raise exc

    def shutdown(self, *, timeout_s: float = 2.0) -> None:
        del timeout_s


class _ShallowFailThenRecoverRemoteService:
    def __init__(self) -> None:
        self.status_calls = 0
        self.ensure_calls = 0
        self.shutdown_calls = 0

    def remote_required(self) -> bool:
        return True

    def remote_status(self):
        self.status_calls += 1
        if self.status_calls == 1:
            return SimpleNamespace(
                mode="remote_primary",
                ready=False,
                detail="Remote long-term memory is temporarily cooling down after recent failures.",
            )
        return SimpleNamespace(mode="remote_primary", ready=True, detail=None)

    def ensure_remote_ready(self) -> None:
        self.ensure_calls += 1

    def shutdown(self, *, timeout_s: float = 2.0) -> None:
        del timeout_s
        self.shutdown_calls += 1


class _StructuredProbeRemoteService:
    def remote_required(self) -> bool:
        return True

    def remote_status(self):
        return SimpleNamespace(mode="remote_primary", ready=True, detail=None)

    def probe_remote_ready(self):
        class _ProbeResult:
            ready = False
            detail = "Failed to read remote long-term snapshot 'prompt_memory' (ChonkyDBError)."
            remote_status = SimpleNamespace(mode="remote_primary", ready=True, detail=None)

            @staticmethod
            def to_dict() -> dict[str, object]:
                return {
                    "ready": False,
                    "detail": "Failed to read remote long-term snapshot 'prompt_memory' (ChonkyDBError).",
                    "remote_status": {
                        "mode": "remote_primary",
                        "ready": True,
                        "detail": None,
                    },
                    "steps": (
                        {
                            "name": "LongTermRemoteHealthProbe.probe_operational",
                            "status": "fail",
                            "warm_result": {
                                "ready": False,
                                "failed_snapshot_kind": "prompt_memory",
                                "proof_contract": {
                                    "contract_id": "configured_namespace_archive_inclusive_readiness",
                                },
                                "checks": [
                                    {
                                        "store": "prompt_context",
                                        "snapshot_kind": "prompt_memory",
                                        "status": "unavailable",
                                        "selected_source": "pointer_document",
                                        "attempts": [
                                            {
                                                "source": "pointer_document",
                                                "attempt": 1,
                                                "status": "error",
                                                "status_code": 503,
                                            }
                                        ],
                                    }
                                ],
                            },
                        },
                    ),
                    "warm_result": {
                        "ready": False,
                        "failed_snapshot_kind": "prompt_memory",
                        "proof_contract": {
                            "contract_id": "configured_namespace_archive_inclusive_readiness",
                        },
                    },
                }

        return _ProbeResult()

    def shutdown(self, *, timeout_s: float = 2.0) -> None:
        del timeout_s


class _ProbeModeRecordingRemoteService:
    def __init__(self, readiness: list[bool]) -> None:
        self._readiness = list(readiness)
        self._index = 0
        self.calls: list[dict[str, object]] = []
        self.shutdown_calls = 0

    def remote_required(self) -> bool:
        return True

    def remote_status(self):
        return SimpleNamespace(mode="remote_primary", ready=True, detail=None)

    def probe_remote_ready(self, *, bootstrap: bool = True, include_archive: bool = True):
        ready = self._readiness[min(self._index, len(self._readiness) - 1)]
        self._index += 1
        self.calls.append(
            {
                "bootstrap": bootstrap,
                "include_archive": include_archive,
            }
        )

        class _ProbeResult:
            def __init__(self, *, ready: bool, include_archive: bool) -> None:
                self.ready = ready
                self.detail = None if ready else "remote unavailable"
                self.remote_status = SimpleNamespace(mode="remote_primary", ready=True, detail=None)
                self.warm_result = SimpleNamespace(
                    archive_safe=bool(ready and include_archive),
                    health_tier="ready" if ready and include_archive else ("degraded" if ready else "hard_down"),
                )

            def to_dict(self) -> dict[str, object]:
                return {
                    "ready": self.ready,
                    "detail": self.detail,
                    "remote_status": {
                        "mode": "remote_primary",
                        "ready": True,
                        "detail": None,
                    },
                    "steps": (),
                    "warm_result": {
                        "archive_safe": bool(self.warm_result.archive_safe),
                        "health_tier": str(self.warm_result.health_tier),
                    },
                }

        return _ProbeResult(ready=ready, include_archive=include_archive)


class _CurrentOnlyProbeRemoteService:
    def remote_required(self) -> bool:
        return True

    def remote_status(self):
        return SimpleNamespace(mode="remote_primary", ready=True, detail=None)

    def probe_remote_ready(self, *, bootstrap: bool = True, include_archive: bool = True):
        del bootstrap, include_archive

        class _ProbeResult:
            ready = True
            detail = None
            remote_status = SimpleNamespace(mode="remote_primary", ready=True, detail=None)
            warm_result = SimpleNamespace(archive_safe=False, health_tier="degraded")

            @staticmethod
            def to_dict() -> dict[str, object]:
                return {
                    "ready": True,
                    "detail": None,
                    "remote_status": {
                        "mode": "remote_primary",
                        "ready": True,
                        "detail": None,
                    },
                    "steps": (),
                    "warm_result": {
                        "archive_safe": False,
                        "health_tier": "degraded",
                    },
                }

        return _ProbeResult()

    def shutdown(self, *, timeout_s: float = 2.0) -> None:
        del timeout_s


def _object_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return {str(key): item for key, item in value.items()}


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
                        captured_monotonic_ns=1_800_000_000,
                        detail=None,
                    ),
                    recent_samples=(),
                    updated_monotonic_ns=1_800_000_000,
                    heartbeat_at="2026-03-16T18:00:05Z",
                    heartbeat_monotonic_ns=1_900_000_000,
                    boot_id="boot-123",
                    pid_starttime_ticks=456789,
                    pid_create_time_s=1710000000.5,
                    probe_inflight=False,
                    probe_started_at=None,
                    probe_started_monotonic_ns=None,
                    probe_age_s=None,
                )
            )

            loaded = store.load()

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.current.status, "ok")
        self.assertEqual(loaded.sample_count, 5)
        self.assertEqual(loaded.failure_count, 1)
        self.assertEqual(loaded.heartbeat_at, "2026-03-16T18:00:05Z")
        self.assertEqual(loaded.current.captured_monotonic_ns, 1_800_000_000)
        self.assertEqual(loaded.updated_monotonic_ns, 1_800_000_000)
        self.assertEqual(loaded.heartbeat_monotonic_ns, 1_900_000_000)
        self.assertEqual(loaded.boot_id, "boot-123")
        self.assertEqual(loaded.pid_starttime_ticks, 456789)
        self.assertEqual(loaded.pid_create_time_s, 1710000000.5)
        self.assertFalse(loaded.probe_inflight)

    def test_store_save_makes_snapshot_world_readable_for_cross_service_health(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(root / "state" / "runtime-state.json"),
            )
            store = RemoteMemoryWatchdogStore.from_config(config)
            sample = RemoteMemoryWatchdogSample(
                seq=1,
                captured_at="2026-03-16T18:00:05Z",
                status="ok",
                ready=True,
                mode="remote_primary",
                required=True,
                latency_ms=42.0,
                consecutive_ok=1,
                consecutive_fail=0,
                detail=None,
            )
            store.save(
                RemoteMemoryWatchdogSnapshot(
                    schema_version=1,
                    started_at="2026-03-16T18:00:00Z",
                    updated_at="2026-03-16T18:00:05Z",
                    hostname="picarx",
                    pid=123,
                    interval_s=1.0,
                    history_limit=3600,
                    sample_count=1,
                    failure_count=0,
                    last_ok_at="2026-03-16T18:00:05Z",
                    last_failure_at=None,
                    artifact_path=str(store.path),
                    current=sample,
                    recent_samples=(sample,),
                    heartbeat_at="2026-03-16T18:00:05Z",
                    probe_inflight=False,
                    probe_started_at=None,
                    probe_age_s=None,
                )
            )

            mode = store.path.stat().st_mode & 0o777

        self.assertEqual(mode, 0o644)

    def test_snapshot_round_trips_through_json_payload_after_state_helper_extraction(self) -> None:
        sample = RemoteMemoryWatchdogSample(
            seq=5,
            captured_at="2026-03-16T18:00:05Z",
            status="fail",
            ready=False,
            mode="remote_primary",
            required=True,
            latency_ms=42.5,
            consecutive_ok=0,
            consecutive_fail=2,
            detail="remote unavailable",
            probe={
                "remote_write_context": {
                    "request_correlation_id": "ltw-test123",
                    "request_item_count": 51,
                }
            },
        )
        snapshot = RemoteMemoryWatchdogSnapshot(
            schema_version=1,
            started_at="2026-03-16T18:00:00Z",
            updated_at="2026-03-16T18:00:05Z",
            hostname="picarx",
            pid=123,
            interval_s=1.0,
            history_limit=3600,
            sample_count=5,
            failure_count=2,
            last_ok_at=None,
            last_failure_at="2026-03-16T18:00:05Z",
            artifact_path="/tmp/remote_memory_watchdog.json",
            current=sample,
            recent_samples=(sample,),
            heartbeat_at="2026-03-16T18:00:05Z",
            probe_inflight=False,
            probe_started_at=None,
            probe_age_s=None,
        )

        reloaded = RemoteMemoryWatchdogSnapshot.from_dict(json.loads(json.dumps(snapshot.to_dict())))

        self.assertEqual(reloaded.current.probe, sample.probe)
        self.assertIsNone(reloaded.recent_samples[0].probe)
        self.assertEqual(reloaded.failure_count, 2)

    def test_bootstrap_snapshot_builder_stays_reexported_from_main_watchdog_module(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(root / "state" / "runtime-state.json"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
            )
            artifact_path = root / "state" / "ops" / "remote_memory_watchdog.json"

            snapshot = build_remote_memory_watchdog_bootstrap_snapshot(
                config,
                pid=321,
                artifact_path=artifact_path,
                started_at="2026-03-16T18:00:00Z",
                captured_at="2026-03-16T18:00:01Z",
            )

        self.assertEqual(snapshot.pid, 321)
        self.assertEqual(snapshot.current.status, "starting")
        self.assertTrue(snapshot.probe_inflight)
        self.assertEqual(snapshot.artifact_path, str(artifact_path))

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
            first_service = _DeepProbeSequencedRemoteService(["ok", "fail", "ok"])
            second_service = _DeepProbeSequencedRemoteService(["ok"])
            service_instances = [first_service, second_service]
            emitted: list[str] = []
            watchdog = RemoteMemoryWatchdog(
                config=config,
                service_factory=lambda: service_instances.pop(0),
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
        self.assertEqual(first_service.shutdown_calls, 1)
        self.assertEqual(second_service.shutdown_calls, 0)

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

            result = watchdog.run(duration_s=0.2)

        events = [json.loads(line)["event"] for line in emitted]
        self.assertEqual(result, 0)
        self.assertIn("remote_memory_watchdog_heartbeat", events)
        self.assertIn("remote_memory_watchdog_sample", events)

    def test_heartbeat_persists_startup_progress_before_first_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(root / "state" / "runtime-state.json"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
            )
            store = RemoteMemoryWatchdogStore.from_config(config)
            watchdog = RemoteMemoryWatchdog(
                config=config,
                service_factory=lambda: _SlowReadyRemoteService(sleep_s=0.03),
                store=store,
                event_store=TwinrOpsEventStore.from_config(config),
                emit=lambda _line: None,
            )

            watchdog._emit_heartbeat(
                probe_started_at="2026-03-16T18:00:00Z",
                probe_started_monotonic=watchdog._monotonic(),
                probe_inflight=True,
            )
            snapshot = store.load()

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.current.status, "starting")
        self.assertTrue(snapshot.probe_inflight)
        self.assertEqual(snapshot.current.seq, 0)
        self.assertIsNotNone(snapshot.heartbeat_at)

    def test_heartbeat_snapshot_persists_monotonic_and_pid_attestation_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(root / "state" / "runtime-state.json"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
            )
            store = RemoteMemoryWatchdogStore.from_config(config)
            with mock.patch(
                "twinr.ops.remote_memory_watchdog._current_boot_id",
                return_value="boot-xyz",
            ), mock.patch(
                "twinr.ops.remote_memory_watchdog._read_proc_stat_start_ticks",
                return_value=12345,
            ), mock.patch(
                "twinr.ops.remote_memory_watchdog._read_proc_create_time_s",
                return_value=67890.5,
            ):
                watchdog = RemoteMemoryWatchdog(
                    config=config,
                    service_factory=lambda: _SlowReadyRemoteService(sleep_s=0.03),
                    store=store,
                    event_store=TwinrOpsEventStore.from_config(config),
                    emit=lambda _line: None,
                )

            watchdog._emit_heartbeat(
                probe_started_at="2026-03-16T18:00:00Z",
                probe_started_monotonic=watchdog._monotonic(),
                probe_inflight=True,
            )
            snapshot = store.load()

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertIsNotNone(snapshot.current.captured_monotonic_ns)
        self.assertEqual(snapshot.updated_monotonic_ns, snapshot.current.captured_monotonic_ns)
        self.assertIsNotNone(snapshot.heartbeat_monotonic_ns)
        self.assertEqual(snapshot.boot_id, "boot-xyz")
        self.assertEqual(snapshot.pid_starttime_ticks, 12345)
        self.assertEqual(snapshot.pid_create_time_s, 67890.5)
        self.assertIsNotNone(snapshot.probe_started_monotonic_ns)
        self.assertTrue(snapshot.probe_inflight)

    def test_stalled_probe_uses_startup_timeout_before_first_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(root / "state" / "runtime-state.json"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_watchdog_interval_s=1.0,
                long_term_memory_remote_watchdog_probe_timeout_s=15.0,
                long_term_memory_remote_watchdog_startup_probe_timeout_s=45.0,
            )
            store = RemoteMemoryWatchdogStore.from_config(config)
            watchdog = RemoteMemoryWatchdog(
                config=config,
                service_factory=_AlwaysFailRemoteService,
                store=store,
                event_store=TwinrOpsEventStore.from_config(config),
                emit=lambda _line: None,
            )
            watchdog._probe_thread = mock.Mock()
            watchdog._probe_thread.is_alive.return_value = True
            watchdog._probe_started_at = "2026-03-30T14:31:44Z"
            watchdog._probe_started_monotonic = 0.0

            watchdog._mark_stalled_probe_if_needed(now_monotonic=36.9)
            self.assertFalse(watchdog._probe_timeout_reported)
            self.assertIsNone(store.load())
            self.assertEqual(watchdog._effective_probe_timeout_s(), 45.0)

    def test_startup_timeout_defaults_to_cold_probe_estimate_when_not_configured(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(root / "state" / "runtime-state.json"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_watchdog_interval_s=1.0,
                long_term_memory_remote_watchdog_probe_timeout_s=15.0,
                long_term_memory_remote_read_timeout_s=8.0,
                chonkydb_timeout_s=20.0,
            )
            watchdog = RemoteMemoryWatchdog(
                config=config,
                service_factory=_AlwaysFailRemoteService,
                store=RemoteMemoryWatchdogStore.from_config(config),
                event_store=TwinrOpsEventStore.from_config(config),
                emit=lambda _line: None,
            )

            self.assertEqual(watchdog.startup_probe_timeout_s, 180.0)
            self.assertEqual(watchdog._effective_probe_timeout_s(), 180.0)

    def test_stalled_probe_fails_once_startup_timeout_is_exceeded_before_first_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(root / "state" / "runtime-state.json"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_watchdog_interval_s=1.0,
                long_term_memory_remote_watchdog_probe_timeout_s=15.0,
                long_term_memory_remote_watchdog_startup_probe_timeout_s=45.0,
            )
            store = RemoteMemoryWatchdogStore.from_config(config)
            watchdog = RemoteMemoryWatchdog(
                config=config,
                service_factory=_AlwaysFailRemoteService,
                store=store,
                event_store=TwinrOpsEventStore.from_config(config),
                emit=lambda _line: None,
            )
            watchdog._probe_thread = mock.Mock()
            watchdog._probe_thread.is_alive.return_value = True
            watchdog._probe_started_at = "2026-03-30T14:31:44Z"
            watchdog._probe_started_monotonic = 0.0

            watchdog._mark_stalled_probe_if_needed(now_monotonic=45.2)
            snapshot = store.load()
            self.assertTrue(watchdog._probe_timeout_reported)
            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertEqual(snapshot.current.status, "fail")
            self.assertFalse(snapshot.current.ready)
            self.assertIn("45.0s", snapshot.current.detail or "")

    def test_stalled_probe_uses_recent_success_latency_headroom_before_failing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(root / "state" / "runtime-state.json"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_watchdog_interval_s=1.0,
            )
            store = RemoteMemoryWatchdogStore.from_config(config)
            watchdog = RemoteMemoryWatchdog(
                config=config,
                service_factory=_AlwaysFailRemoteService,
                store=store,
                event_store=TwinrOpsEventStore.from_config(config),
                emit=lambda _line: None,
            )
            watchdog._build_sample(
                captured_at="2026-03-30T13:57:39Z",
                status="ok",
                ready=True,
                mode="remote_primary",
                required=True,
                latency_ms=12054.8,
                detail=None,
            )
            watchdog._has_success_since_start = True
            watchdog._probe_thread = mock.Mock()
            watchdog._probe_thread.is_alive.return_value = True
            watchdog._probe_started_at = "2026-03-30T13:57:52Z"
            watchdog._probe_started_monotonic = 0.0

            watchdog._mark_stalled_probe_if_needed(now_monotonic=16.7)
            self.assertFalse(watchdog._probe_timeout_reported)
            self.assertIsNone(store.load())
            self.assertGreater(watchdog._effective_probe_timeout_s(), 16.7)

    def test_stalled_probe_fails_once_recent_success_headroom_is_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(root / "state" / "runtime-state.json"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_watchdog_interval_s=1.0,
            )
            store = RemoteMemoryWatchdogStore.from_config(config)
            watchdog = RemoteMemoryWatchdog(
                config=config,
                service_factory=_AlwaysFailRemoteService,
                store=store,
                event_store=TwinrOpsEventStore.from_config(config),
                emit=lambda _line: None,
            )
            watchdog._build_sample(
                captured_at="2026-03-30T13:57:39Z",
                status="ok",
                ready=True,
                mode="remote_primary",
                required=True,
                latency_ms=12054.8,
                detail=None,
            )
            watchdog._has_success_since_start = True
            watchdog._probe_thread = mock.Mock()
            watchdog._probe_thread.is_alive.return_value = True
            watchdog._probe_started_at = "2026-03-30T13:57:52Z"
            watchdog._probe_started_monotonic = 0.0

            watchdog._mark_stalled_probe_if_needed(now_monotonic=18.2)
            snapshot = store.load()
            self.assertTrue(watchdog._probe_timeout_reported)
            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertEqual(snapshot.current.status, "fail")
            self.assertFalse(snapshot.current.ready)
            self.assertIn("18.1s", snapshot.current.detail or "")

    def test_watchdog_restores_recent_history_from_store_and_uses_it_for_timeout_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(root / "state" / "runtime-state.json"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
            )
            store = RemoteMemoryWatchdogStore.from_config(config)
            persisted_ok = RemoteMemoryWatchdogSample(
                seq=9,
                captured_at="2026-03-30T20:39:30Z",
                status="ok",
                ready=True,
                mode="remote_primary",
                required=True,
                latency_ms=139311.6,
                consecutive_ok=1,
                consecutive_fail=0,
                detail=None,
            )
            store.save(
                RemoteMemoryWatchdogSnapshot(
                    schema_version=1,
                    started_at="2026-03-30T20:36:12Z",
                    updated_at="2026-03-30T20:39:30Z",
                    hostname="picarx",
                    pid=123,
                    interval_s=1.0,
                    history_limit=3600,
                    sample_count=9,
                    failure_count=3,
                    last_ok_at="2026-03-30T20:39:30Z",
                    last_failure_at="2026-03-30T20:37:57Z",
                    artifact_path=str(store.path),
                    current=RemoteMemoryWatchdogSample(
                        seq=0,
                        captured_at="2026-03-30T20:44:06Z",
                        status="starting",
                        ready=False,
                        mode="remote_primary",
                        required=True,
                        latency_ms=0.0,
                        consecutive_ok=0,
                        consecutive_fail=0,
                        detail="Remote memory watchdog is starting.",
                    ),
                    recent_samples=(persisted_ok,),
                    heartbeat_at="2026-03-30T20:44:06Z",
                    probe_inflight=True,
                    probe_started_at="2026-03-30T20:44:06Z",
                    probe_age_s=0.0,
                )
            )

            watchdog = RemoteMemoryWatchdog(
                config=config,
                service_factory=_AlwaysFailRemoteService,
                store=store,
                event_store=TwinrOpsEventStore.from_config(config),
                emit=lambda _line: None,
            )

            self.assertEqual(watchdog._sample_count, 9)
            self.assertEqual(watchdog._failure_count, 3)
            self.assertEqual(watchdog._last_ok_at, "2026-03-30T20:39:30Z")
            self.assertEqual(len(watchdog._recent_samples), 1)
            self.assertGreater(watchdog._effective_probe_timeout_s(), 145.0)

    def test_bootstrap_snapshot_preserves_previous_recent_history(self) -> None:
        config = TwinrConfig(
            project_root="/tmp/twinr",
            long_term_memory_enabled=True,
            long_term_memory_mode="remote_primary",
        )
        previous_sample = RemoteMemoryWatchdogSample(
            seq=4,
            captured_at="2026-03-30T20:39:30Z",
            status="ok",
            ready=True,
            mode="remote_primary",
            required=True,
            latency_ms=139311.6,
            consecutive_ok=1,
            consecutive_fail=0,
            detail=None,
        )
        previous_snapshot = RemoteMemoryWatchdogSnapshot(
            schema_version=1,
            started_at="2026-03-30T20:36:12Z",
            updated_at="2026-03-30T20:39:30Z",
            hostname="picarx",
            pid=111,
            interval_s=1.0,
            history_limit=3600,
            sample_count=4,
            failure_count=3,
            last_ok_at="2026-03-30T20:39:30Z",
            last_failure_at="2026-03-30T20:37:57Z",
            artifact_path="/tmp/twinr/remote_memory_watchdog.json",
            current=previous_sample,
            recent_samples=(previous_sample,),
            heartbeat_at="2026-03-30T20:39:30Z",
            probe_inflight=False,
            probe_started_at=None,
            probe_age_s=None,
        )

        bootstrap = build_remote_memory_watchdog_bootstrap_snapshot(
            config,
            pid=222,
            artifact_path="/tmp/twinr/remote_memory_watchdog.json",
            previous_snapshot=previous_snapshot,
        )

        self.assertEqual(bootstrap.sample_count, 4)
        self.assertEqual(bootstrap.failure_count, 3)
        self.assertEqual(bootstrap.last_ok_at, "2026-03-30T20:39:30Z")
        self.assertEqual(len(bootstrap.recent_samples), 1)
        self.assertEqual(bootstrap.recent_samples[0].latency_ms, 139311.6)
        self.assertEqual(bootstrap.current.status, "starting")

    def test_probe_once_preserves_cached_service_after_expected_remote_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(root / "state" / "runtime-state.json"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
            )
            service = _DeepProbeSequencedRemoteService(["fail", "ok"])
            watchdog = RemoteMemoryWatchdog(
                config=config,
                service_factory=lambda: service,
                store=RemoteMemoryWatchdogStore.from_config(config),
                event_store=TwinrOpsEventStore.from_config(config),
                emit=lambda _line: None,
            )

            first = watchdog.probe_once()
            second = watchdog.probe_once()
            watchdog.close()

        self.assertEqual(first.current.status, "fail")
        self.assertEqual(second.current.status, "ok")
        self.assertEqual(service.shutdown_calls, 1)

    def test_probe_once_recreates_cached_service_after_unexpected_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(root / "state" / "runtime-state.json"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
            )
            failing_service = _UnexpectedErrorRemoteService()
            recovering_service = _SequencedRemoteService(["ok"])
            service_instances = [failing_service, recovering_service]
            watchdog = RemoteMemoryWatchdog(
                config=config,
                service_factory=lambda: service_instances.pop(0),
                store=RemoteMemoryWatchdogStore.from_config(config),
                event_store=TwinrOpsEventStore.from_config(config),
                emit=lambda _line: None,
            )

            first = watchdog.probe_once()
            second = watchdog.probe_once()
            watchdog.close()

        self.assertEqual(first.current.status, "fail")
        self.assertIn("RuntimeError", first.current.detail or "")
        self.assertEqual(second.current.status, "ok")
        self.assertEqual(failing_service.shutdown_calls, 1)
        self.assertEqual(recovering_service.shutdown_calls, 1)

    def test_probe_once_preserves_cached_service_when_shallow_remote_status_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(root / "state" / "runtime-state.json"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
            )
            service = _ShallowFailThenRecoverRemoteService()
            watchdog = RemoteMemoryWatchdog(
                config=config,
                service_factory=lambda: service,
                store=RemoteMemoryWatchdogStore.from_config(config),
                event_store=TwinrOpsEventStore.from_config(config),
                emit=lambda _line: None,
            )

            first = watchdog.probe_once()
            second = watchdog.probe_once()
            watchdog.close()

        self.assertEqual(first.current.status, "fail")
        self.assertIn("cooling down", first.current.detail or "")
        self.assertEqual(second.current.status, "ok")
        self.assertEqual(service.ensure_calls, 1)
        self.assertEqual(service.shutdown_calls, 1)

    def test_probe_once_persists_structured_probe_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(root / "state" / "runtime-state.json"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
            )
            watchdog = RemoteMemoryWatchdog(
                config=config,
                service_factory=_StructuredProbeRemoteService,
                store=RemoteMemoryWatchdogStore.from_config(config),
                event_store=TwinrOpsEventStore.from_config(config),
                emit=lambda _line: None,
            )

            snapshot = watchdog.probe_once()

        self.assertEqual(snapshot.current.status, "fail")
        self.assertIsNotNone(snapshot.current.probe)
        assert snapshot.current.probe is not None
        steps = cast(list[dict[str, object]], snapshot.current.probe["steps"])
        warm_result = _object_dict(steps[0]["warm_result"])
        top_level_warm_result = _object_dict(snapshot.current.probe["warm_result"])
        proof_contract = _object_dict(top_level_warm_result["proof_contract"])
        self.assertEqual(steps[0]["status"], "fail")
        self.assertEqual(warm_result["failed_snapshot_kind"], "[TRUNCATED]")
        self.assertEqual(warm_result["checks"], "[TRUNCATED]")
        self.assertEqual(proof_contract["contract_id"], "configured_namespace_archive_inclusive_readiness")

    def test_heartbeat_snapshot_compacts_historical_probe_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(root / "state" / "runtime-state.json"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
            )
            store = RemoteMemoryWatchdogStore.from_config(config)
            watchdog = RemoteMemoryWatchdog(
                config=config,
                service_factory=_StructuredProbeRemoteService,
                store=store,
                event_store=TwinrOpsEventStore.from_config(config),
                emit=lambda _line: None,
            )

            watchdog.probe_once()
            watchdog._emit_heartbeat(
                probe_started_at="2026-03-16T18:00:00Z",
                probe_started_monotonic=watchdog._monotonic(),
                probe_inflight=False,
            )
            payload = json.loads(store.path.read_text(encoding="utf-8"))

        self.assertIsNotNone(payload["current"]["probe"])
        self.assertEqual(len(payload["recent_samples"]), 1)
        self.assertIsNone(payload["recent_samples"][0]["probe"])

    def test_persisted_snapshot_caps_recent_sample_history_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(root / "state" / "runtime-state.json"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_watchdog_history_limit=200,
            )
            watchdog = RemoteMemoryWatchdog(
                config=config,
                service_factory=lambda: _SequencedRemoteService(["ok"]),
                store=RemoteMemoryWatchdogStore.from_config(config),
                event_store=TwinrOpsEventStore.from_config(config),
                emit=lambda _line: None,
            )

            for _ in range(80):
                watchdog.probe_once()
            payload = json.loads(watchdog.artifact_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["sample_count"], 80)
        self.assertEqual(len(payload["recent_samples"]), 64)

    def test_run_waits_for_keepalive_gap_before_starting_next_deep_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(root / "state" / "runtime-state.json"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_watchdog_interval_s=0.01,
                long_term_memory_remote_keepalive_interval_s=0.2,
            )
            service = _DeepProbeSequencedRemoteService(["ok", "ok", "ok"])
            watchdog = RemoteMemoryWatchdog(
                config=config,
                service_factory=lambda: service,
                store=RemoteMemoryWatchdogStore.from_config(config),
                event_store=TwinrOpsEventStore.from_config(config),
                emit=lambda _line: None,
            )

            result = watchdog.run(duration_s=0.08)
            snapshot = RemoteMemoryWatchdogStore.from_config(config).load()

        self.assertEqual(result, 0)
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.sample_count, 1)
        self.assertEqual(service.shutdown_calls, 1)

    def test_probe_once_keeps_archive_inclusive_probes_until_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(root / "state" / "runtime-state.json"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
            )
            service = _ProbeModeRecordingRemoteService([True, True, False, True])
            watchdog = RemoteMemoryWatchdog(
                config=config,
                service_factory=lambda: service,
                store=RemoteMemoryWatchdogStore.from_config(config),
                event_store=TwinrOpsEventStore.from_config(config),
                emit=lambda _line: None,
            )

            first = watchdog.probe_once()
            second = watchdog.probe_once()
            third = watchdog.probe_once()
            fourth = watchdog.probe_once()
            watchdog.close()

        self.assertEqual(first.current.status, "ok")
        self.assertEqual(second.current.status, "ok")
        self.assertEqual(third.current.status, "fail")
        self.assertEqual(fourth.current.status, "ok")
        self.assertEqual(
            service.calls,
            [
                {"bootstrap": True, "include_archive": True},
                {"bootstrap": False, "include_archive": True},
                {"bootstrap": False, "include_archive": True},
                {"bootstrap": True, "include_archive": True},
            ],
        )

    def test_probe_once_rejects_current_only_attestation_as_degraded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(root / "state" / "runtime-state.json"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
            )
            watchdog = RemoteMemoryWatchdog(
                config=config,
                service_factory=_CurrentOnlyProbeRemoteService,
                store=RemoteMemoryWatchdogStore.from_config(config),
                event_store=TwinrOpsEventStore.from_config(config),
                emit=lambda _line: None,
            )

            snapshot = watchdog.probe_once()

        self.assertEqual(snapshot.current.status, "degraded")
        self.assertFalse(snapshot.current.ready)
        self.assertIn("archive-safe", snapshot.current.detail or "")

    def test_probe_once_timestamps_sample_at_probe_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(root / "state" / "runtime-state.json"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
            )
            monotonic_values = itertools.chain((0.0, 100.0, 104.2, 104.2), itertools.repeat(104.2))
            with mock.patch(
                "twinr.ops.remote_memory_watchdog._utc_now_iso",
                side_effect=("2026-03-16T18:00:00Z", "2026-03-16T18:00:04Z"),
            ):
                watchdog = RemoteMemoryWatchdog(
                    config=config,
                    service_factory=lambda: _DeepProbeSequencedRemoteService(["ok"]),
                    store=RemoteMemoryWatchdogStore.from_config(config),
                    event_store=TwinrOpsEventStore.from_config(config),
                    emit=lambda _line: None,
                    monotonic=lambda: next(monotonic_values),
                )
                snapshot = watchdog.probe_once()

        self.assertEqual(snapshot.current.captured_at, "2026-03-16T18:00:04Z")
        self.assertEqual(snapshot.updated_at, "2026-03-16T18:00:04Z")
        self.assertEqual(snapshot.heartbeat_at, "2026-03-16T18:00:04Z")
        self.assertEqual(snapshot.current.latency_ms, 4200.0)

    def test_probe_once_carries_remote_write_context_into_sample_and_transition_event(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(root / "state" / "runtime-state.json"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
            )
            event_store = TwinrOpsEventStore.from_config(config)
            watchdog = RemoteMemoryWatchdog(
                config=config,
                service_factory=_CorrelatedFailRemoteService,
                store=RemoteMemoryWatchdogStore.from_config(config),
                event_store=event_store,
                emit=lambda _line: None,
            )

            snapshot = watchdog.probe_once()
            events = event_store.tail(limit=5)

        self.assertEqual(snapshot.current.status, "fail")
        self.assertIsNotNone(snapshot.current.probe)
        assert snapshot.current.probe is not None
        remote_write_context = _object_dict(snapshot.current.probe["remote_write_context"])
        self.assertEqual(remote_write_context["request_correlation_id"], "ltw-test123")
        transition_event = next(event for event in events if event["event"] == "remote_memory_watchdog_status_changed")
        transition_data = _object_dict(transition_event["data"])
        self.assertEqual(transition_data["remote_write_context"], "[REDACTED]")


if __name__ == "__main__":
    unittest.main()
