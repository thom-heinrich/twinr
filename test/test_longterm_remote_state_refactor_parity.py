from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test.test_longterm_remote_state import FakeOpener  # noqa: E402
from twinr.agent.base_agent.config import TwinrConfig  # noqa: E402
from twinr.memory.chonkydb import ChonkyDBClient, ChonkyDBConnectionConfig  # noqa: E402
from twinr.memory.longterm.storage._remote_state.base import LongTermRemoteStateStoreImpl  # noqa: E402
from twinr.memory.longterm.storage.remote_state import (  # noqa: E402
    LongTermRemoteStateStore,
    remote_snapshot_document_hints_path,
)

_EXPECTED_GOLDEN_DIGESTS = {
    "pointer_read": "c90c04e251aff88257e6b742e278bed7e2fd0185b64152a3f6692d69a27df890",
    "status_ready": "eb65f9d6a2b4f3f46b836fc3968da61bb073b0b3b59b5d159f798c792e0da591",
}


def _normalize_payload(value):
    if is_dataclass(value):
        return _normalize_payload(asdict(value))
    if isinstance(value, Path):
        return "<PATH>"
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if key == "latency_ms":
                normalized[str(key)] = "<latency>"
                continue
            normalized[str(key)] = _normalize_payload(item)
        return normalized
    if isinstance(value, list):
        return [_normalize_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_payload(item) for item in value]
    if hasattr(value, "__dict__") and not isinstance(value, (int, float, bool, str, type(None))):
        return {
            key: _normalize_payload(item)
            for key, item in value.__dict__.items()
            if not key.startswith("_")
        }
    return value


def _config(
    root: str,
    *,
    required: bool = True,
    read_cache_ttl_s: float = 0.0,
) -> TwinrConfig:
    return TwinrConfig(
        project_root=root,
        long_term_memory_path="memory",
        long_term_memory_enabled=True,
        long_term_memory_mode="remote_primary",
        long_term_memory_remote_required=required,
        long_term_memory_migration_enabled=True,
        long_term_memory_remote_namespace="test-namespace",
        long_term_memory_remote_read_cache_ttl_s=read_cache_ttl_s,
        chonkydb_base_url="https://memory.test",
        chonkydb_api_key="secret-key",
    )


def _pointer_read_payload(store_cls: type[LongTermRemoteStateStore]) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as temp_dir:
        config = _config(temp_dir, read_cache_ttl_s=5.0)
        read_opener = FakeOpener()
        read_opener.queue_json(
            {
                "success": True,
                "document_id": "pointer-objects",
                "content": json.dumps(
                    {
                        "schema": "twinr_remote_snapshot_v1",
                        "namespace": "test-namespace",
                        "snapshot_kind": "__pointer__:objects",
                        "updated_at": "2026-03-27T12:00:00+00:00",
                        "body": {
                            "schema": "twinr_remote_snapshot_pointer_v1",
                            "version": 1,
                            "snapshot_kind": "objects",
                            "document_id": "doc-objects",
                        },
                    }
                ),
            }
        )
        for _index in range(2):
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-objects",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "updated_at": "2026-03-27T12:00:01+00:00",
                            "body": {
                                "schema": "object_store",
                                "objects": [{"memory_id": "durable-tea"}],
                            },
                        }
                    ),
                }
            )
        store = store_cls(
            config=config,
            read_client=ChonkyDBClient(
                ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                opener=read_opener,
            ),
            write_client=ChonkyDBClient(
                ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                opener=FakeOpener(),
            ),
        )

        payload = store.load_snapshot(snapshot_kind="objects")
        probe = store.probe_snapshot_load(
            snapshot_kind="objects",
            prefer_cached_document_id=True,
        )
        return {
            "payload": payload,
            "probe": probe.to_dict(include_payload=True),
            "calls": [
                {
                    "method": call["method"],
                    "full_url": call["full_url"],
                    "timeout": call["timeout"],
                }
                for call in read_opener.calls
            ],
        }


def _optional_local_fallback_payload(store_cls: type[LongTermRemoteStateStore]) -> dict[str, object]:
    raise NotImplementedError


def _status_ready_payload(store_cls: type[LongTermRemoteStateStore]) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as temp_dir:
        config = _config(temp_dir)

        class _StatusClient:
            def __init__(self) -> None:
                self.config = SimpleNamespace(timeout_s=3.5)
                self.calls = 0

            def instance(self):
                self.calls += 1
                return SimpleNamespace(ready=True)

            def clone_with_timeout(self, timeout_s: float):
                self.config = SimpleNamespace(timeout_s=timeout_s)
                return self

        read_client = _StatusClient()
        write_client = object()
        store = store_cls(
            config=config,
            read_client=read_client,
            write_client=write_client,
        )
        status = store.status()
        hints_path = remote_snapshot_document_hints_path(config)
        return {
            "status": _normalize_payload(status),
            "read_client_calls": read_client.calls,
            "read_timeout_after_probe": read_client.config.timeout_s,
            "hints_path_name": None if hints_path is None else hints_path.name,
        }


class LongTermRemoteStateRefactorParityTests(unittest.TestCase):
    def test_public_wrapper_preserves_class_module(self) -> None:
        self.assertEqual(
            LongTermRemoteStateStore.__module__,
            "twinr.memory.longterm.storage.remote_state",
        )

    def test_golden_master_hashes_remain_stable(self) -> None:
        cases = {
            "pointer_read": _pointer_read_payload(LongTermRemoteStateStore),
            "status_ready": _status_ready_payload(LongTermRemoteStateStore),
        }
        for name, payload in cases.items():
            with self.subTest(case=name):
                serialized = json.dumps(
                    _normalize_payload(payload),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
                digest = sha256(serialized.encode("utf-8")).hexdigest()
                self.assertEqual(digest, _EXPECTED_GOLDEN_DIGESTS[name])

    def test_public_wrapper_matches_internal_implementation_payloads(self) -> None:
        cases = (
            ("pointer_read", _pointer_read_payload),
            ("status_ready", _status_ready_payload),
        )
        for name, builder in cases:
            with self.subTest(case=name):
                wrapped = _normalize_payload(builder(LongTermRemoteStateStore))
                internal = _normalize_payload(builder(LongTermRemoteStateStoreImpl))
                self.assertEqual(wrapped, internal)
