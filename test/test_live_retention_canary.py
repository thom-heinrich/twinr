from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.evaluation import live_retention_canary as canary_mod
from twinr.memory.chonkydb.client import ChonkyDBError
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError
from twinr.ops.remote_memory_watchdog_state import (
    RemoteMemoryWatchdogSample,
    RemoteMemoryWatchdogSnapshot,
    RemoteMemoryWatchdogStore,
)


class _FakeWriterObjectStore:
    def __init__(self) -> None:
        self.committed_ids: tuple[str, ...] = ()

    def commit_active_delta(self, *, object_upserts) -> None:
        self.committed_ids = tuple(item.memory_id for item in object_upserts)

    def load_objects_by_projection_filter(self, *, predicate):
        del predicate
        return (
            SimpleNamespace(memory_id="event:retention_future_appointment"),
            SimpleNamespace(memory_id="episode:retention_old_weather"),
        )

    def load_objects(self):  # pragma: no cover - patched out in the canary
        return ()

    def load_objects_fine_grained(self):  # pragma: no cover - patched out in the canary
        return ()

    def load_current_state_fine_grained(self):  # pragma: no cover - patched out in the canary
        return SimpleNamespace(objects=())

    def load_archived_objects(self):  # pragma: no cover - patched out in the canary
        return ()

    def write_snapshot(self, *args, **kwargs):  # pragma: no cover - patched out in the canary
        del args, kwargs


class _FakeReaderObjectStore:
    def load_current_state_fine_grained(self):
        raise LongTermRemoteUnavailableError(
            "Failed to read remote long-term 'objects' item 'event:retention_future_appointment'."
        )

    def load_archived_objects_fine_grained(self):
        return ()


class _FakeWriterService:
    def __init__(self) -> None:
        self.object_store = _FakeWriterObjectStore()

    def ensure_remote_ready(self) -> None:
        return None

    def run_retention(self):
        self.object_store.load_objects_by_projection_filter(predicate=lambda _item: True)
        return SimpleNamespace(
            archived_objects=(SimpleNamespace(memory_id="episode:retention_old_weather"),),
            pruned_memory_ids=("observation:retention_old_presence",),
        )

    def shutdown(self, *, timeout_s: float = 2.0) -> None:
        del timeout_s


class _FakeReaderService:
    def __init__(self) -> None:
        self.object_store = _FakeReaderObjectStore()

    def ensure_remote_ready(self) -> None:
        return None

    def shutdown(self, *, timeout_s: float = 2.0) -> None:
        del timeout_s


class _FailingWriterObjectStore:
    def commit_active_delta(self, *, object_upserts) -> None:
        del object_upserts
        try:
            raise TimeoutError("write timed out")
        except TimeoutError as root_cause:
            try:
                raise ChonkyDBError(
                    "ChonkyDB request failed for POST /v1/external/records/bulk: timed out"
                ) from root_cause
            except ChonkyDBError as transport_error:
                unavailable = LongTermRemoteUnavailableError(
                    "Failed to persist fine-grained remote long-term memory items "
                    "(request_id=ltw-test123, batch=1/2, items=1, bytes=2378)."
                )
                setattr(
                    unavailable,
                    "remote_write_context",
                    {
                        "snapshot_kind": "objects",
                        "operation": "store_records_bulk",
                        "request_path": "/v1/external/records/bulk",
                        "request_payload_kind": "fine_grained_record_batch",
                        "request_execution_mode": "async",
                        "timeout_s": 180.0,
                        "attempt_count": 3,
                        "request_correlation_id": "ltw-test123",
                        "batch_index": 1,
                        "batch_count": 2,
                        "request_item_count": 1,
                        "request_bytes": 2378,
                    },
                )
                raise unavailable from transport_error


class _FailingWriterService:
    def __init__(self) -> None:
        self.object_store = _FailingWriterObjectStore()

    def ensure_remote_ready(self) -> None:
        return None

    def shutdown(self, *, timeout_s: float = 2.0) -> None:
        del timeout_s


class LiveRetentionCanaryTests(unittest.TestCase):
    def test_run_live_retention_canary_records_failure_stage_and_watchdog_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("TEST=1\n", encoding="utf-8")
            base_config = TwinrConfig(
                project_root=str(root),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=True,
                chonkydb_base_url="https://example.invalid",
                chonkydb_api_key="test-key",
            )
            watchdog_store = RemoteMemoryWatchdogStore.from_config(base_config)
            watchdog_store.save(
                RemoteMemoryWatchdogSnapshot(
                    schema_version=1,
                    started_at="2026-04-04T08:00:00Z",
                    updated_at="2026-04-04T08:00:05Z",
                    hostname="test-host",
                    pid=123,
                    interval_s=1.0,
                    history_limit=64,
                    sample_count=12,
                    failure_count=0,
                    last_ok_at="2026-04-04T08:00:05Z",
                    last_failure_at=None,
                    artifact_path=str(watchdog_store.path),
                    current=RemoteMemoryWatchdogSample(
                        seq=12,
                        captured_at="2026-04-04T08:00:05Z",
                        status="ok",
                        ready=True,
                        mode="remote_primary",
                        required=True,
                        latency_ms=2400.0,
                        consecutive_ok=12,
                        consecutive_fail=0,
                        detail=None,
                        probe={
                            "warm_result": {
                                "probe_mode": "archive_inclusive",
                                "archive_safe": True,
                                "health_tier": "ready",
                                "proof_contract": {
                                    "contract_id": "configured_namespace_archive_inclusive_readiness",
                                },
                            }
                        },
                    ),
                    recent_samples=(),
                )
            )
            writer_service = _FakeWriterService()
            reader_service = _FakeReaderService()

            with mock.patch.object(canary_mod.TwinrConfig, "from_env", return_value=base_config), mock.patch.object(
                canary_mod,
                "_normalize_base_project_root",
                return_value=root,
            ), mock.patch.object(
                canary_mod,
                "_build_isolated_config",
                side_effect=(SimpleNamespace(name="writer"), SimpleNamespace(name="reader")),
            ), mock.patch.object(
                canary_mod.LongTermMemoryService,
                "from_config",
                side_effect=(writer_service, reader_service),
            ):
                result = canary_mod.run_live_retention_canary(
                    env_path=env_path,
                    probe_id="unit_test_retention_canary",
                    write_artifacts=False,
                )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failure_stage, "fresh_reader_load_current_state_fine_grained")
        self.assertIn("event:retention_future_appointment", result.error_message or "")
        self.assertEqual(result.proof_contract["contract_id"], "isolated_namespace_retention_transaction")
        self.assertEqual(
            result.consistency_assessment["relation"],
            "watchdog_ready_canary_failed_non_equivalent",
        )
        self.assertEqual(result.watchdog_observations[-1]["sample_status"], "ok")
        self.assertTrue(result.watchdog_observations[-1]["sample_ready"])
        self.assertEqual(
            [step["name"] for step in result.steps[-2:]],
            ["fresh_reader_ensure_remote_ready", "fresh_reader_load_current_state_fine_grained"],
        )
        self.assertEqual(result.steps[-1]["status"], "fail")

    def test_run_live_retention_canary_carries_remote_write_failure_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("TEST=1\n", encoding="utf-8")
            base_config = TwinrConfig(
                project_root=str(root),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=True,
                chonkydb_base_url="https://example.invalid",
                chonkydb_api_key="test-key",
            )
            watchdog_store = RemoteMemoryWatchdogStore.from_config(base_config)
            watchdog_store.save(
                RemoteMemoryWatchdogSnapshot(
                    schema_version=1,
                    started_at="2026-04-04T08:00:00Z",
                    updated_at="2026-04-04T08:00:05Z",
                    hostname="test-host",
                    pid=123,
                    interval_s=1.0,
                    history_limit=64,
                    sample_count=12,
                    failure_count=0,
                    last_ok_at="2026-04-04T08:00:05Z",
                    last_failure_at=None,
                    artifact_path=str(watchdog_store.path),
                    current=RemoteMemoryWatchdogSample(
                        seq=12,
                        captured_at="2026-04-04T08:00:05Z",
                        status="ok",
                        ready=True,
                        mode="remote_primary",
                        required=True,
                        latency_ms=2400.0,
                        consecutive_ok=12,
                        consecutive_fail=0,
                        detail=None,
                        probe={
                            "warm_result": {
                                "probe_mode": "archive_inclusive",
                                "archive_safe": True,
                                "health_tier": "ready",
                                "proof_contract": {
                                    "contract_id": "configured_namespace_archive_inclusive_readiness",
                                },
                            }
                        },
                    ),
                    recent_samples=(),
                )
            )
            writer_service = _FailingWriterService()

            with mock.patch.object(canary_mod.TwinrConfig, "from_env", return_value=base_config), mock.patch.object(
                canary_mod,
                "_normalize_base_project_root",
                return_value=root,
            ), mock.patch.object(
                canary_mod,
                "_build_isolated_config",
                side_effect=(SimpleNamespace(name="writer"), SimpleNamespace(name="reader")),
            ), mock.patch.object(
                canary_mod.LongTermMemoryService,
                "from_config",
                side_effect=(writer_service,),
            ):
                result = canary_mod.run_live_retention_canary(
                    env_path=env_path,
                    probe_id="unit_test_retention_canary_write_failure",
                    write_artifacts=False,
                )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failure_stage, "seed_retention_objects")
        self.assertEqual(result.root_cause_message, "TimeoutError: write timed out")
        assert result.remote_write_context is not None
        self.assertEqual(result.remote_write_context["request_correlation_id"], "ltw-test123")
        self.assertEqual(result.remote_write_context["request_execution_mode"], "async")
        self.assertEqual(
            [item["type"] for item in result.exception_chain],
            ["LongTermRemoteUnavailableError", "ChonkyDBError", "TimeoutError"],
        )
        failure_step = result.steps[-1]
        evidence = dict(failure_step["evidence"])
        self.assertEqual(evidence["root_cause_message"], "TimeoutError: write timed out")
        self.assertEqual(evidence["remote_write_context"]["request_correlation_id"], "ltw-test123")


if __name__ == "__main__":
    unittest.main()
