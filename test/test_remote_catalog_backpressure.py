from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test.test_longterm_store import _FakeChonkyClient, _config, _source
from twinr.memory.chonkydb.client import ChonkyDBError
from twinr.memory.longterm.core.models import LongTermMemoryObjectV1
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore
from twinr.memory.longterm.storage.store import LongTermStructuredStore


class _AsyncStructuredSnapshotBackpressureClient(_FakeChonkyClient):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(timeout_s=15.0)
        self.job_status_calls: list[str] = []
        self.bulk_execution_modes: list[str] = []
        self.non_catalog_async_batches = 0
        self._pending_job_results: dict[str, dict[str, object]] = {}

    def clone_with_timeout(self, timeout_s: float):
        return _AsyncStructuredSnapshotBackpressureClientView(self, timeout_s=timeout_s)

    def _store_records_bulk_with_timeout(self, request, *, timeout_s: float):
        self.config = SimpleNamespace(timeout_s=float(timeout_s))
        items = tuple(getattr(request, "items", ()))
        execution_mode = str(getattr(request, "execution_mode", "") or "")
        self.bulk_execution_modes.append(execution_mode)
        item_uris = tuple(str(getattr(item, "uri", "") or "") for item in items)
        is_fine_grained_batch = bool(item_uris) and all("/catalog/" not in uri for uri in item_uris)
        if execution_mode == "async" and is_fine_grained_batch:
            self.non_catalog_async_batches += 1
            if self._pending_job_results:
                raise ChonkyDBError(
                    "ChonkyDB request failed for POST /v1/external/records/bulk",
                    status_code=429,
                    response_json={
                        "detail": "queue_saturated",
                        "error": "queue_saturated",
                        "error_type": "RuntimeError",
                        "success": False,
                        "status": 429,
                    },
                )
            accepted = super().store_records_bulk(request)
            job_id = f"job-{len(self._pending_job_results) + self.non_catalog_async_batches}"
            self._pending_job_results[job_id] = dict(accepted)
            return {
                "success": True,
                "job_id": job_id,
                "items": [{} for _ in items],
            }
        return super().store_records_bulk(request)

    def store_records_bulk(self, request):
        return self._store_records_bulk_with_timeout(request, timeout_s=float(self.config.timeout_s))

    def job_status(self, job_id: str):
        self.job_status_calls.append(str(job_id))
        result = self._pending_job_results.pop(str(job_id), None)
        if result is None:
            return {"status": "succeeded", "result": {"items": []}}
        return {"status": "succeeded", "result": result}


class _AsyncStructuredSnapshotBackpressureClientView:
    def __init__(
        self,
        parent: _AsyncStructuredSnapshotBackpressureClient,
        *,
        timeout_s: float,
    ) -> None:
        self._parent = parent
        self.config = SimpleNamespace(timeout_s=float(timeout_s))

    def clone_with_timeout(self, timeout_s: float):
        return self._parent.clone_with_timeout(timeout_s)

    def store_records_bulk(self, request):
        return self._parent._store_records_bulk_with_timeout(request, timeout_s=float(self.config.timeout_s))

    def job_status(self, job_id: str):
        return self._parent.job_status(job_id)

    def fetch_full_document(self, *args, **kwargs):
        return self._parent.fetch_full_document(*args, **kwargs)

    def retrieve(self, request):
        return self._parent.retrieve(request)


class RemoteCatalogBackpressureTests(unittest.TestCase):
    def test_bulk_structured_snapshot_waits_for_async_jobs_before_next_fine_grained_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = _AsyncStructuredSnapshotBackpressureClient()
            config = replace(
                _config(temp_dir),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=True,
                long_term_memory_remote_write_timeout_s=15.0,
                long_term_memory_remote_flush_timeout_s=60.0,
                long_term_memory_remote_retry_attempts=2,
                long_term_memory_remote_retry_backoff_s=0.0,
            )
            remote_state = LongTermRemoteStateStore(
                config=config,
                read_client=cast(object, client),
                write_client=cast(object, client),
                namespace="test-namespace",
            )
            store = LongTermStructuredStore(
                base_path=Path(temp_dir) / "state" / "chonkydb",
                remote_state=remote_state,
            )
            objects = tuple(
                LongTermMemoryObjectV1(
                    memory_id=f"object:{index:03d}",
                    kind="observation",
                    summary=f"Observation {index}",
                    details="x" * 9000,
                    source=_source(),
                )
                for index in range(24)
            )

            store.write_snapshot(objects=objects)
            loaded_ids = tuple(item.memory_id for item in store.load_objects_fine_grained())

        self.assertGreaterEqual(client.non_catalog_async_batches, 2)
        self.assertGreaterEqual(len(client.job_status_calls), 1)
        self.assertFalse(client._pending_job_results)
        self.assertTrue(all(mode == "async" for mode in client.bulk_execution_modes[: client.non_catalog_async_batches]))
        self.assertEqual(loaded_ids, tuple(item.memory_id for item in objects))


if __name__ == "__main__":
    unittest.main()
