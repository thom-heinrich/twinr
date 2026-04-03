from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
from typing import Callable, cast
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb import TwinrPersonalGraphStore
from twinr.memory.chonkydb._remote_graph_state import TwinrRemoteGraphState
from twinr.memory.chonkydb.client import ChonkyDBError
from twinr.memory.longterm.core.models import LongTermGraphEdgeCandidateV1
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError


_TEST_CORINNA_PHONE = "5551234"
_TEST_CORINNA_NEIGHBOR_PHONE = "5559988"
_TEST_CORINNA_ALT_PHONE = "5557777"
_TEST_JANINA_PHONE_A = "+15555550011"
_TEST_JANINA_PHONE_B = "+15555550012"


class _FakeGraphClient:
    def __init__(self, *, error_getter) -> None:
        self._error_getter = error_getter
        self._next_document_id = 1
        self.records_by_document_id: dict[str, dict[str, object]] = {}
        self.records_by_uri: dict[str, dict[str, object]] = {}
        self.graph_store_many_calls: list[dict[str, object]] = []
        self.graph_path_calls: list[dict[str, object]] = []
        self.graph_neighbors_calls: list[dict[str, object]] = []
        self.graph_indexes: dict[str, dict[str, object]] = {}

    def _maybe_raise(self) -> None:
        error = self._error_getter()
        if error is not None:
            raise error

    def store_records_bulk(self, request):
        self._maybe_raise()
        items = tuple(getattr(request, "items", ()))
        response_items: list[dict[str, object]] = []
        for item in items:
            document_id = f"doc-{self._next_document_id}"
            self._next_document_id += 1
            record = {
                "document_id": document_id,
                "payload": dict(getattr(item, "payload", {}) or {}),
                "metadata": dict(getattr(item, "metadata", {}) or {}),
                "content": getattr(item, "content", None),
                "uri": getattr(item, "uri", None),
            }
            self.records_by_document_id[document_id] = record
            uri = record.get("uri")
            if isinstance(uri, str) and uri:
                self.records_by_uri[uri] = record
            response_items.append({"document_id": document_id})
        return {"items": response_items}

    def fetch_full_document(self, *, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
        del include_content, max_content_chars
        self._maybe_raise()
        if isinstance(document_id, str) and document_id:
            record = self.records_by_document_id.get(document_id)
            if record is not None:
                return dict(record)
        if isinstance(origin_uri, str) and origin_uri:
            record = self.records_by_uri.get(origin_uri)
            if record is not None:
                return dict(record)
        raise LongTermRemoteUnavailableError("remote document unavailable")

    def graph_store_many(self, request):
        self._maybe_raise()
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        self.graph_store_many_calls.append(dict(payload))
        index_name = str(payload.get("index_name") or "")
        index = self.graph_indexes.setdefault(index_name, {"nodes": set(), "edges": []})
        nodes = index["nodes"]
        edges = index["edges"]
        for item in payload.get("nodes", ()):
            label = str((item or {}).get("label") or "")
            if label:
                nodes.add(label)
        for item in payload.get("edges", ()):
            edge = (
                str((item or {}).get("source_label") or ""),
                str((item or {}).get("edge_type") or ""),
                str((item or {}).get("target_label") or ""),
            )
            if all(edge) and edge not in edges:
                edges.append(edge)
        return {"success": True, "index_name": index_name, "node_count": len(nodes), "edge_count": len(edges)}

    def graph_neighbors(self, request):
        self._maybe_raise()
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        self.graph_neighbors_calls.append(dict(payload))
        index = self.graph_indexes.get(str(payload.get("index_name") or ""), {"edges": []})
        edge_types = {str(item) for item in payload.get("edge_types", ()) if str(item)}
        label = str(payload.get("label_or_id") or "")
        limit = max(1, int(payload.get("limit") or 10))
        neighbors: list[dict[str, object]] = []
        for source, edge_type, target in index["edges"]:
            if source != label:
                continue
            if edge_types and edge_type not in edge_types:
                continue
            neighbors.append({"label": target, "edge_type": edge_type})
            if len(neighbors) >= limit:
                break
        return {"success": True, "neighbors": neighbors}

    def graph_path(self, request):
        self._maybe_raise()
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        self.graph_path_calls.append(dict(payload))
        index = self.graph_indexes.get(str(payload.get("index_name") or ""), {"edges": []})
        source = str(payload.get("source") or "")
        target = str(payload.get("target") or "")
        allowed_edge_types = {str(item) for item in payload.get("edge_types", ()) if str(item)}
        adjacency: dict[str, list[str]] = {}
        for edge_source, edge_type, edge_target in index["edges"]:
            if allowed_edge_types and edge_type not in allowed_edge_types:
                continue
            adjacency.setdefault(edge_source, []).append(edge_target)
        queue: list[tuple[str, list[str]]] = [(source, [source])]
        seen = {source}
        while queue:
            node, path = queue.pop(0)
            if node == target:
                return {"success": True, "path": path}
            for neighbor in adjacency.get(node, ()):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append((neighbor, [*path, neighbor]))
        return {"success": True, "path": []}


class _SyncGraphWriteFailingClient(_FakeGraphClient):
    def __init__(self, *, error_getter) -> None:
        super().__init__(error_getter=error_getter)
        self.bulk_execution_modes: list[str] = []
        self.bulk_timeout_seconds: list[float | None] = []
        self.bulk_call_details: list[dict[str, object]] = []

    def store_records_bulk(self, request):
        self._maybe_raise()
        items = tuple(getattr(request, "items", ()))
        execution_mode = str(getattr(request, "execution_mode", ""))
        timeout_seconds = getattr(request, "timeout_seconds", None)
        snapshot_kinds = {
            str((getattr(item, "metadata", {}) or {}).get("twinr_snapshot_kind") or "")
            for item in items
        }
        uris = tuple(str(getattr(item, "uri", "") or "") for item in items)
        self.bulk_execution_modes.append(execution_mode)
        self.bulk_timeout_seconds.append(None if timeout_seconds is None else float(timeout_seconds))
        self.bulk_call_details.append(
            {
                "execution_mode": execution_mode,
                "timeout_seconds": None if timeout_seconds is None else float(timeout_seconds),
                "snapshot_kinds": tuple(sorted(snapshot_kinds)),
                "uris": uris,
            }
        )
        control_plane_only = bool(uris) and all(
            "/catalog/current" in uri or "/catalog/segment/" in uri for uri in uris
        )
        if (
            execution_mode == "sync"
            and snapshot_kinds & {"graph_nodes", "graph_edges"}
            and not control_plane_only
        ):
            raise LongTermRemoteUnavailableError("graph current-view sync write timed out")
        return super().store_records_bulk(request)


class _TransientGraphQueryFailingClient(_FakeGraphClient):
    def __init__(self, *, path_failures: int = 0, neighbor_failures: int = 0) -> None:
        super().__init__(error_getter=lambda: None)
        self._remaining_path_failures = max(0, int(path_failures))
        self._remaining_neighbor_failures = max(0, int(neighbor_failures))

    def graph_path(self, request):
        if self._remaining_path_failures > 0:
            self._remaining_path_failures -= 1
            raise ChonkyDBError(
                "graph path temporarily unavailable",
                status_code=503,
                response_headers={"Retry-After": "0"},
            )
        return super().graph_path(request)

    def graph_neighbors(self, request):
        if self._remaining_neighbor_failures > 0:
            self._remaining_neighbor_failures -= 1
            raise ChonkyDBError(
                "graph neighbors temporarily unavailable",
                status_code=503,
                response_headers={"Retry-After": "0"},
            )
        return super().graph_neighbors(request)


class _GraphAsyncDocumentIdLagClient(_FakeGraphClient):
    def __init__(self) -> None:
        super().__init__(error_getter=lambda: None)
        self.job_status_calls: list[str] = []
        self.bulk_call_details: list[dict[str, object]] = []
        self.bulk_execution_modes: list[str] = []
        self.bulk_timeout_seconds: list[float | None] = []
        self._next_job_id = 1

    def store_records_bulk(self, request):
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        items = tuple(getattr(request, "items", ()))
        execution_mode = str(getattr(request, "execution_mode", ""))
        timeout_seconds = getattr(request, "timeout_seconds", None)
        snapshot_kinds = {
            str((getattr(item, "metadata", {}) or {}).get("twinr_snapshot_kind") or "")
            for item in items
        }
        uris = tuple(str(getattr(item, "uri", "") or "") for item in items)
        self.bulk_call_details.append(
            {
                "execution_mode": execution_mode,
                "timeout_seconds": None if timeout_seconds is None else float(timeout_seconds),
                "snapshot_kinds": tuple(sorted(snapshot_kinds)),
                "uris": uris,
                "payload": payload,
            }
        )
        self.bulk_execution_modes.append(execution_mode)
        self.bulk_timeout_seconds.append(None if timeout_seconds is None else float(timeout_seconds))
        result = super().store_records_bulk(request)
        if snapshot_kinds & {"graph_nodes", "graph_edges"}:
            job_id = f"job-{self._next_job_id}"
            self._next_job_id += 1
            return {"success": True, "job_id": job_id}
        return result

    def job_status(self, job_id: str):
        self.job_status_calls.append(str(job_id))
        raise AssertionError(f"Graph writes should not poll async job status for projection-complete batches: {job_id}")


class _TimeoutTrackingReadClient:
    def __init__(
        self,
        *,
        timeout_s: float,
        records_by_uri: dict[str, dict[str, object]],
        fetch_calls: list[dict[str, object]] | None = None,
    ) -> None:
        self.config = SimpleNamespace(timeout_s=float(timeout_s))
        self.records_by_uri = records_by_uri
        self.fetch_calls = fetch_calls if fetch_calls is not None else []

    def clone_with_timeout(self, timeout_s: float):
        return _TimeoutTrackingReadClient(
            timeout_s=float(timeout_s),
            records_by_uri=self.records_by_uri,
            fetch_calls=self.fetch_calls,
        )

    def fetch_full_document(self, *, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
        del document_id
        self.fetch_calls.append(
            {
                "timeout_s": float(self.config.timeout_s),
                "origin_uri": origin_uri,
                "include_content": bool(include_content),
                "max_content_chars": int(max_content_chars),
            }
        )
        if isinstance(origin_uri, str) and origin_uri in self.records_by_uri:
            return dict(self.records_by_uri[origin_uri])
        raise LongTermRemoteUnavailableError("remote document unavailable")


class _FakeRemoteState:
    def __init__(self) -> None:
        self.enabled = True
        self.required = False
        self.namespace = "test-namespace"
        self.load_error: Exception | None = None
        self.client = _FakeGraphClient(error_getter=lambda: self.load_error)
        self.read_client = self.client
        self.write_client = self.client
        self.config = SimpleNamespace(
            long_term_memory_migration_enabled=False,
            long_term_memory_migration_batch_size=64,
            long_term_memory_remote_read_timeout_s=8.0,
            long_term_memory_remote_write_timeout_s=15.0,
            long_term_memory_remote_flush_timeout_s=60.0,
            long_term_memory_remote_bulk_request_max_bytes=512 * 1024,
            long_term_memory_remote_shard_max_content_chars=1000,
            long_term_memory_remote_max_content_chars=2_000_000,
            long_term_memory_remote_read_cache_ttl_s=0.0,
        )
        self.snapshots: dict[str, dict[str, object]] = {}
        self.load_calls: list[dict[str, object]] = []
        self.probe_calls: list[dict[str, object]] = []
        self._next_snapshot_document_id = 1
        self._snapshot_uri: Callable[[str], str] | None = self._build_snapshot_uri
        self._extract_snapshot_body: Callable[[object], dict[str, object] | None] | None = (
            self._extract_snapshot_body_impl
        )

    def load_snapshot(self, *, snapshot_kind: str, local_path=None, prefer_cached_document_id: bool = False):
        self.load_calls.append(
            {
                "snapshot_kind": snapshot_kind,
                "local_path": local_path,
                "prefer_cached_document_id": prefer_cached_document_id,
            }
        )
        del local_path
        if self.load_error is not None:
            raise self.load_error
        payload = self.snapshots.get(snapshot_kind)
        return dict(payload) if isinstance(payload, dict) else None

    def probe_snapshot_load(
        self,
        *,
        snapshot_kind: str,
        local_path=None,
        prefer_cached_document_id: bool = False,
        prefer_metadata_only: bool = False,
        fast_fail: bool = False,
    ):
        self.probe_calls.append(
            {
                "snapshot_kind": snapshot_kind,
                "local_path": local_path,
                "prefer_cached_document_id": prefer_cached_document_id,
                "prefer_metadata_only": prefer_metadata_only,
                "fast_fail": fast_fail,
            }
        )
        payload = self.load_snapshot(
            snapshot_kind=snapshot_kind,
            local_path=local_path,
            prefer_cached_document_id=prefer_cached_document_id,
        )
        return SimpleNamespace(
            snapshot_kind=snapshot_kind,
            status="found" if isinstance(payload, dict) else "not_found",
            detail=None,
            payload=payload,
        )

    def _build_snapshot_uri(self, snapshot_kind: str) -> str:
        return f"twinr://longterm/{self.namespace}/{snapshot_kind}"

    def _extract_snapshot_body_impl(self, payload, *, snapshot_kind: str):
        body = payload.get("body") if isinstance(payload, dict) else None
        nested_payload = payload.get("payload") if isinstance(payload, dict) else None
        if not isinstance(body, dict):
            body = nested_payload.get("body") if isinstance(nested_payload, dict) else None
        if not isinstance(body, dict):
            return None
        snapshot_kind_value = (
            payload.get("snapshot_kind")
            if isinstance(payload, dict)
            else None
        )
        if not snapshot_kind_value and isinstance(nested_payload, dict):
            snapshot_kind_value = nested_payload.get("snapshot_kind")
        if str(snapshot_kind_value or "") not in {
            "",
            snapshot_kind,
        }:
            return None
        return dict(body)

    def save_snapshot(self, *, snapshot_kind: str, payload):
        payload_dict = dict(payload)
        self.snapshots[snapshot_kind] = payload_dict
        document_id = f"snapshot-{self._next_snapshot_document_id}"
        self._next_snapshot_document_id += 1
        snapshot_uri_builder = self._snapshot_uri
        assert callable(snapshot_uri_builder)
        snapshot_uri = snapshot_uri_builder(snapshot_kind)
        record = {
            "document_id": document_id,
            "payload": {
                "snapshot_kind": snapshot_kind,
                "body": payload_dict,
            },
            "body": payload_dict,
            "snapshot_kind": snapshot_kind,
            "uri": snapshot_uri,
            "content": json.dumps(payload_dict, ensure_ascii=False),
        }
        self.client.records_by_document_id[document_id] = dict(record)
        self.client.records_by_uri[snapshot_uri] = dict(record)


def _memory_payload(context: str) -> dict[str, object]:
    _header, _sep, payload = context.partition("\n")
    return json.loads(payload)


class TwinrPersonalGraphStoreTests(unittest.TestCase):
    def test_contact_lookup_requires_clarification_for_same_first_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
            )
            store = TwinrPersonalGraphStore.from_config(config)

            first = store.remember_contact(
                given_name="Corinna",
                family_name="Maier",
                phone=_TEST_CORINNA_PHONE,
                role="Physiotherapist",
            )
            second = store.remember_contact(
                given_name="Corinna",
                family_name="Schmidt",
                phone=_TEST_CORINNA_NEIGHBOR_PHONE,
                role="Neighbor",
            )
            lookup = store.lookup_contact(name="Corinna")
            resolved = store.lookup_contact(name="Corinna", role="Physiotherapist")

        self.assertEqual(first.status, "created")
        self.assertEqual(second.status, "created")
        self.assertEqual(lookup.status, "needs_clarification")
        self.assertIn("Corinna", lookup.question or "")
        self.assertEqual(len(lookup.options), 2)
        self.assertEqual(resolved.status, "found")
        self.assertEqual(resolved.match.label, "Corinna Maier")
        self.assertEqual(resolved.match.phones, (_TEST_CORINNA_PHONE,))

    def test_contact_lookup_resolves_exact_contact_label_for_ambiguous_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
            )
            store = TwinrPersonalGraphStore.from_config(config)

            store.remember_contact(
                given_name="Janina",
                family_name="Werner",
                phone=_TEST_JANINA_PHONE_A,
                role="Bekannte",
            )
            store.remember_contact(
                given_name="Janina",
                family_name="Werner privat",
                phone=_TEST_JANINA_PHONE_B,
                role="Privat",
            )

            resolved = store.lookup_contact(
                name="Janina",
                contact_label="Janina Werner privat",
            )

        self.assertEqual(resolved.status, "found")
        assert resolved.match is not None
        self.assertEqual(resolved.match.label, "Janina Werner privat")
        self.assertEqual(resolved.match.phones, (_TEST_JANINA_PHONE_B,))

    def test_contact_lookup_prefers_exact_full_name_over_role_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
            )
            store = TwinrPersonalGraphStore.from_config(config)

            created = store.remember_contact(
                given_name="Anna",
                family_name="Schulz",
                phone="+15555552233",
                role="Tochter",
            )
            resolved = store.lookup_contact(
                name="Anna",
                family_name="Schulz",
                role="daughter",
            )

        self.assertEqual(created.status, "created")
        self.assertEqual(resolved.status, "found")
        assert resolved.match is not None
        self.assertEqual(resolved.match.label, "Anna Schulz")
        self.assertEqual(resolved.match.phones, ("+15555552233",))

    def test_from_config_places_graph_lock_in_runtime_state_lock_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(Path(temp_dir) / "state" / "runtime-state.json"),
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
            )

            store = TwinrPersonalGraphStore.from_config(config)

        self.assertEqual(
            store._lock_path,
            Path(temp_dir) / "state" / "locks" / "twinr_graph_v1.json.lock",
        )

    def test_contact_save_asks_before_merging_new_number_into_existing_named_person(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
            )
            store = TwinrPersonalGraphStore.from_config(config)
            store.remember_contact(
                given_name="Corinna",
                family_name="Maier",
                phone=_TEST_CORINNA_PHONE,
                role="Physiotherapist",
            )

            result = store.remember_contact(given_name="Corinna", phone=_TEST_CORINNA_ALT_PHONE)

        self.assertEqual(result.status, "needs_clarification")
        self.assertIn("Corinna", result.question or "")

    def test_preference_and_plan_feed_prompt_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
                openai_web_search_timezone="Europe/Berlin",
            )
            store = TwinrPersonalGraphStore.from_config(config)
            store.remember_preference(
                category="brand",
                value="Melitta",
                for_product="coffee",
            )
            store.remember_plan(
                summary="go for a walk",
                when_text="today",
            )

            weather_context = store.build_prompt_context("Wie wird das Wetter heute?")
            shopping_context = store.build_prompt_context("Wo kann ich heute Kaffee kaufen?")

        self.assertIsNotNone(weather_context)
        weather_payload = _memory_payload(weather_context or "")
        self.assertEqual(weather_payload["schema"], "twinr_graph_memory_context_v1")
        self.assertIn(
            {"summary": "go for a walk", "when": "today", "date": weather_payload["plans"][0]["date"]},
            weather_payload["plans"],
        )
        self.assertIsNotNone(shopping_context)
        shopping_payload = _memory_payload(shopping_context or "")
        serialized = json.dumps(shopping_payload, ensure_ascii=False)
        self.assertIn("Melitta", serialized)
        self.assertIn("coffee", serialized)

    def test_subtext_payload_ignores_low_information_token_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
                openai_web_search_timezone="Europe/Berlin",
            )
            store = TwinrPersonalGraphStore.from_config(config)
            store.remember_plan(
                summary="Go for a walk in the park",
                when_text="today",
            )

            subtext_payload = store.build_subtext_payload("Wie spät ist es in Tokio?")

        self.assertIsNone(subtext_payload)

    def test_apply_candidate_edges_persists_graph_refs_as_nodes_and_edges(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
            )
            store = TwinrPersonalGraphStore.from_config(config)
            store.apply_candidate_edges(
                (
                    LongTermGraphEdgeCandidateV1(
                        source_ref="user:main",
                        edge_type="social_related_to_user",
                        target_ref="person:janina",
                        confidence=0.95,
                        confirmed_by_user=True,
                        attributes={"relation": "wife"},
                    ),
                    LongTermGraphEdgeCandidateV1(
                        source_ref="person:janina",
                        edge_type="spatial_located_in",
                        target_ref="place:eye_doctor",
                        confidence=0.88,
                        confirmed_by_user=True,
                    ),
                )
            )
            document = store.load_document()

        node_ids = {node.node_id for node in document.nodes}
        edge_types = {edge.edge_type for edge in document.edges}

        self.assertIn("user:main", node_ids)
        self.assertIn("person:janina", node_ids)
        self.assertIn("place:eye_doctor", node_ids)
        self.assertIn("social_related_to_user", edge_types)
        self.assertIn("spatial_located_in", edge_types)

    def test_remote_primary_graph_store_keeps_snapshot_off_disk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )
            store.remember_plan(summary="go for a walk", when_text="today")
            document = store.load_document()
            remote_view = store.probe_remote_current_view()

        self.assertFalse(store.path.exists())
        self.assertTrue(document.edges)
        self.assertIsNotNone(remote_view)
        assert remote_view is not None
        self.assertEqual(remote_view["graph_id"], "graph:user_main")
        self.assertTrue(remote_state.client.graph_store_many_calls)

    def test_ensure_remote_snapshot_seeds_empty_remote_current_view(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )

            created = store.ensure_remote_snapshot()
            remote_view = store.probe_remote_current_view()
            loaded = store.load_document()

        self.assertTrue(created)
        self.assertIsNotNone(remote_view)
        assert remote_view is not None
        self.assertEqual(remote_view["subject_node_id"], "user:main")
        self.assertEqual(loaded.metadata["kind"], "personal_graph")
        self.assertIn("user:main", remote_view["topology_refs"])
        self.assertTrue(remote_state.client.graph_store_many_calls)

    def test_ensure_remote_snapshot_keeps_graph_item_batches_async_while_allowing_control_plane_fast_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _SyncGraphWriteFailingClient(error_getter=lambda: remote_state.load_error)
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )

            created = store.ensure_remote_snapshot()
            remote_view = store.probe_remote_current_view()

        self.assertTrue(created)
        self.assertIsNotNone(remote_view)
        assert remote_view is not None
        self.assertEqual(remote_view["subject_node_id"], "user:main")
        self.assertTrue(remote_state.client.graph_store_many_calls)
        self.assertTrue(remote_state.client.bulk_call_details)
        self.assertTrue(remote_state.client.bulk_execution_modes)
        fine_grained_details = [
            detail
            for detail in remote_state.client.bulk_call_details
            if detail["uris"]
            and not all(
                "/catalog/current" in uri or "/catalog/segment/" in uri
                for uri in detail["uris"]
            )
        ]
        control_plane_details = [
            detail
            for detail in remote_state.client.bulk_call_details
            if detail["uris"]
            and all(
                "/catalog/current" in uri or "/catalog/segment/" in uri
                for uri in detail["uris"]
            )
        ]
        self.assertTrue(fine_grained_details)
        self.assertTrue(all(detail["execution_mode"] == "async" for detail in fine_grained_details))
        self.assertTrue(control_plane_details)
        self.assertTrue(any(detail["execution_mode"] == "sync" for detail in control_plane_details))
        self.assertTrue(remote_state.client.bulk_timeout_seconds)
        self.assertTrue(all(value == 180.0 for value in remote_state.client.bulk_timeout_seconds))

    def test_remember_contact_skips_async_job_document_id_wait_for_graph_catalog_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _GraphAsyncDocumentIdLagClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )

            result = store.remember_contact(
                given_name="Corinna",
                family_name="Maier",
                phone=_TEST_CORINNA_PHONE,
                role="Physiotherapist",
            )
            remote_view = store.load_remote_current_view()

        self.assertEqual(result.status, "created")
        self.assertEqual(remote_state.client.job_status_calls, [])
        self.assertTrue(remote_state.client.bulk_call_details)
        fine_grained_details = [
            detail
            for detail in remote_state.client.bulk_call_details
            if detail["snapshot_kinds"]
            and detail["uris"]
            and not all(
                "/catalog/current" in uri or "/catalog/segment/" in uri
                for uri in detail["uris"]
            )
        ]
        control_plane_details = [
            detail
            for detail in remote_state.client.bulk_call_details
            if detail["snapshot_kinds"]
            and detail["uris"]
            and all(
                "/catalog/current" in uri or "/catalog/segment/" in uri
                for uri in detail["uris"]
            )
        ]
        self.assertTrue(fine_grained_details)
        self.assertTrue(all(detail["execution_mode"] == "async" for detail in fine_grained_details))
        self.assertTrue(control_plane_details)
        self.assertTrue(any(detail["execution_mode"] == "sync" for detail in control_plane_details))
        self.assertTrue(remote_state.client.bulk_timeout_seconds)
        self.assertTrue(all(value == 180.0 for value in remote_state.client.bulk_timeout_seconds))
        self.assertIsNotNone(remote_view)
        assert remote_view is not None
        self.assertEqual(remote_view["subject_node_id"], "user:main")

    def test_ensure_remote_snapshot_repairs_broken_current_view_from_local_graph_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            path = Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json"
            store = TwinrPersonalGraphStore(
                path=path,
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )
            store.remember_plan(summary="go for a walk", when_text="today")
            document_before = store.load_document()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(document_before.to_payload(), ensure_ascii=False), encoding="utf-8")
            graph_store_many_calls_before = len(remote_state.client.graph_store_many_calls)
            subject_uri = store._remote_graph._catalog.item_uri(snapshot_kind="graph_nodes", item_id="user:main")
            subject_record = remote_state.client.records_by_uri.pop(subject_uri)
            remote_state.client.records_by_document_id.pop(str(subject_record.get("document_id") or ""), None)
            node_segment_uris = tuple(
                uri
                for uri in remote_state.client.records_by_uri
                if "/graph_nodes/catalog/segment/" in uri
            )
            for uri in node_segment_uris:
                record = remote_state.client.records_by_uri.pop(uri, None)
                if isinstance(record, dict):
                    remote_state.client.records_by_document_id.pop(str(record.get("document_id") or ""), None)

            repaired = store.ensure_remote_snapshot()
            document_after = store.load_document()

        self.assertTrue(repaired)
        self.assertGreater(len(remote_state.client.graph_store_many_calls), graph_store_many_calls_before)
        self.assertTrue(any(node.node_id == "user:main" for node in document_after.nodes))
        repaired_record = remote_state.client.records_by_uri.get(subject_uri)
        self.assertIsNotNone(repaired_record)
        assert repaired_record is not None
        self.assertEqual(repaired_record["payload"]["item_id"], "user:main")

    def test_ensure_remote_snapshot_fails_closed_for_broken_current_view_without_local_graph_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )
            store.remember_plan(summary="go for a walk", when_text="today")
            self.assertFalse(store.path.exists())
            subject_uri = store._remote_graph._catalog.item_uri(snapshot_kind="graph_nodes", item_id="user:main")
            subject_record = remote_state.client.records_by_uri.pop(subject_uri)
            remote_state.client.records_by_document_id.pop(str(subject_record.get("document_id") or ""), None)
            node_segment_uris = tuple(
                uri
                for uri in remote_state.client.records_by_uri
                if "/graph_nodes/catalog/segment/" in uri
            )
            for uri in node_segment_uris:
                record = remote_state.client.records_by_uri.pop(uri, None)
                if isinstance(record, dict):
                    remote_state.client.records_by_document_id.pop(str(record.get("document_id") or ""), None)

            with self.assertRaises(LongTermRemoteUnavailableError):
                store.ensure_remote_snapshot()

    def test_ensure_remote_snapshot_uses_compatible_current_view_when_direct_heads_lag(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            primary_store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )
            self.assertTrue(primary_store.ensure_remote_snapshot())
            for snapshot_kind in ("graph_nodes", "graph_edges"):
                head_payload = primary_store._remote_graph._catalog.probe_catalog_payload(snapshot_kind=snapshot_kind)
                assert head_payload is not None
                remote_state.save_snapshot(snapshot_kind=snapshot_kind, payload=head_payload)
            graph_store_many_calls_before = len(remote_state.client.graph_store_many_calls)
            remote_state.probe_calls.clear()
            remote_state.load_calls.clear()
            lagging_uris = tuple(
                uri
                for uri in remote_state.client.records_by_uri
                if uri.endswith("/graph_nodes/catalog/current") or uri.endswith("/graph_edges/catalog/current")
            )
            for uri in lagging_uris:
                remote_state.client.records_by_uri.pop(uri, None)

            fresh_store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "fresh" / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )

            created = fresh_store.ensure_remote_snapshot()
            remote_view = fresh_store.load_remote_current_view()

        self.assertFalse(created)
        self.assertEqual(len(remote_state.client.graph_store_many_calls), graph_store_many_calls_before)
        for uri in lagging_uris:
            self.assertNotIn(uri, remote_state.client.records_by_uri)
        self.assertIsNotNone(remote_view)
        assert remote_view is not None
        self.assertEqual(remote_view["subject_node_id"], "user:main")
        self.assertEqual(remote_state.probe_calls, [])
        self.assertEqual(remote_state.load_calls, [])

    def test_ensure_remote_snapshot_repairs_generic_direct_heads_when_edge_snapshot_pointer_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            path = Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json"
            store = TwinrPersonalGraphStore(
                path=path,
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )
            store.remember_plan(summary="go for a walk", when_text="today")
            document_before = store.load_document()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(document_before.to_payload(), ensure_ascii=False), encoding="utf-8")
            graph_store_many_calls_before = len(remote_state.client.graph_store_many_calls)
            for snapshot_kind in ("graph_nodes", "graph_edges"):
                head_uri = store._remote_graph._catalog._catalog_head_uri(snapshot_kind=snapshot_kind)
                head_record = dict(remote_state.client.records_by_uri[head_uri])
                payload = dict(head_record.get("payload") or {})
                generic_payload = {
                    "schema": payload.get("schema"),
                    "version": payload.get("version"),
                    "items_count": payload.get("items_count"),
                    "segments": [],
                }
                head_record["payload"] = generic_payload
                head_record["content"] = json.dumps(generic_payload, ensure_ascii=False)
                remote_state.client.records_by_uri[head_uri] = head_record
                remote_state.client.records_by_document_id[str(head_record.get("document_id") or "")] = head_record
            remote_state.snapshots.pop("graph_edges", None)
            edge_snapshot_uri = remote_state._build_snapshot_uri("graph_edges")
            edge_snapshot_record = remote_state.client.records_by_uri.pop(edge_snapshot_uri, None)
            if isinstance(edge_snapshot_record, dict):
                remote_state.client.records_by_document_id.pop(str(edge_snapshot_record.get("document_id") or ""), None)

            repaired = store.ensure_remote_snapshot()
            remote_view = store.load_remote_current_view()
            edge_head = store._remote_graph._catalog.probe_catalog_payload(snapshot_kind="graph_edges")

        self.assertTrue(repaired)
        self.assertGreater(len(remote_state.client.graph_store_many_calls), graph_store_many_calls_before)
        self.assertIsNotNone(remote_view)
        assert remote_view is not None
        self.assertEqual(remote_view["subject_node_id"], "user:main")
        self.assertIsNotNone(edge_head)
        assert edge_head is not None
        self.assertEqual(edge_head["generation_id"], remote_view["generation_id"])

    def test_ensure_remote_snapshot_repairs_generation_mismatched_current_heads_from_local_graph_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            path = Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json"
            store = TwinrPersonalGraphStore(
                path=path,
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )
            store.remember_plan(summary="go for a walk", when_text="today")
            document_before = store.load_document()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(document_before.to_payload(), ensure_ascii=False), encoding="utf-8")
            graph_store_many_calls_before = len(remote_state.client.graph_store_many_calls)
            for snapshot_kind, bad_generation_id in (
                ("graph_nodes", "gen-bad-node"),
                ("graph_edges", "gen-bad-edge"),
            ):
                head_uri = store._remote_graph._catalog._catalog_head_uri(snapshot_kind=snapshot_kind)
                head_record = dict(remote_state.client.records_by_uri[head_uri])
                payload = dict(head_record.get("payload") or {})
                payload["generation_id"] = bad_generation_id
                head_record["payload"] = payload
                head_record["content"] = json.dumps(payload, ensure_ascii=False)
                remote_state.client.records_by_uri[head_uri] = head_record
                remote_state.client.records_by_document_id[str(head_record.get("document_id") or "")] = head_record

            repaired = store.ensure_remote_snapshot()
            remote_view = store.load_remote_current_view()

        self.assertTrue(repaired)
        self.assertGreater(len(remote_state.client.graph_store_many_calls), graph_store_many_calls_before)
        self.assertIsNotNone(remote_view)
        assert remote_view is not None
        self.assertEqual(remote_view["subject_node_id"], "user:main")
        self.assertNotEqual(remote_view["generation_id"], "gen-bad-node")
        self.assertNotEqual(remote_view["generation_id"], "gen-bad-edge")

    def test_probe_remote_current_view_fast_fails_compatible_snapshot_probe_without_cached_doc_hint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            primary_store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )
            self.assertTrue(primary_store.ensure_remote_snapshot())
            for snapshot_kind in ("graph_nodes", "graph_edges"):
                head_payload = primary_store._remote_graph._catalog.probe_catalog_payload(snapshot_kind=snapshot_kind)
                assert head_payload is not None
                remote_state.save_snapshot(snapshot_kind=snapshot_kind, payload=head_payload)
            lagging_uris = tuple(
                uri
                for uri in remote_state.client.records_by_uri
                if uri.endswith("/graph_nodes/catalog/current") or uri.endswith("/graph_edges/catalog/current")
            )
            for uri in lagging_uris:
                remote_state.client.records_by_uri.pop(uri, None)
            remote_state.probe_calls.clear()
            remote_state.load_calls.clear()
            remote_state._snapshot_uri = None  # type: ignore[method-assign]
            remote_state._extract_snapshot_body = None  # type: ignore[method-assign]

            fresh_store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "fresh" / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )

            remote_view = fresh_store.probe_remote_current_view()

        self.assertIsNotNone(remote_view)
        self.assertEqual(
            {call["snapshot_kind"] for call in remote_state.probe_calls},
            {"graph_nodes", "graph_edges"},
        )
        self.assertTrue(all(call["prefer_cached_document_id"] is False for call in remote_state.probe_calls))

    def test_probe_direct_current_head_payload_read_only_caps_readiness_timeout(self) -> None:
        head_uri = "twinr://longterm/test-namespace/graph_nodes/catalog/current"
        tracking_client = _TimeoutTrackingReadClient(
            timeout_s=12.0,
            records_by_uri={
                head_uri: {
                    "payload": {
                        "schema": "twinr_graph_catalog_head_v1",
                        "subject_node_id": "user:main",
                    }
                }
            },
        )
        remote_state = SimpleNamespace(
            enabled=True,
            required=False,
            namespace="test-namespace",
            read_client=tracking_client,
        )
        graph_state = TwinrRemoteGraphState(remote_state)
        graph_state._catalog = SimpleNamespace(  # type: ignore[assignment]
            _require_client=lambda client, operation: client,
            _catalog_head_uri=lambda snapshot_kind: head_uri if snapshot_kind == "graph_nodes" else "",
            _metadata_only_max_content_chars=lambda: 321,
            _extract_catalog_payload_from_document=lambda snapshot_kind, payload: dict(payload.get("payload") or {}),
        )

        payload = graph_state._probe_direct_current_head_payload_read_only(snapshot_kind="graph_nodes")

        self.assertEqual(payload, {"schema": "twinr_graph_catalog_head_v1", "subject_node_id": "user:main"})
        self.assertEqual(len(tracking_client.fetch_calls), 1)
        self.assertEqual(tracking_client.fetch_calls[0]["timeout_s"], 5.0)
        self.assertEqual(tracking_client.fetch_calls[0]["origin_uri"], head_uri)
        self.assertTrue(tracking_client.fetch_calls[0]["include_content"] is False)

    def test_probe_remote_current_view_retries_full_content_when_metadata_only_graph_heads_get_400(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            primary_store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )
            self.assertTrue(primary_store.ensure_remote_snapshot())
            remote_state.probe_calls.clear()
            remote_state.load_calls.clear()
            original_fetch_full_document = remote_state.client.fetch_full_document
            fetch_calls: list[dict[str, object]] = []

            def _fetch_full_document(*, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
                fetch_calls.append(
                    {
                        "document_id": document_id,
                        "origin_uri": origin_uri,
                        "include_content": include_content,
                        "max_content_chars": max_content_chars,
                    }
                )
                if (
                    not include_content
                    and isinstance(origin_uri, str)
                    and (
                        origin_uri.endswith("/graph_nodes/catalog/current")
                        or origin_uri.endswith("/graph_edges/catalog/current")
                    )
                ):
                    raise ChonkyDBError(
                        "ChonkyDB request failed for GET /v1/external/documents/full",
                        status_code=400,
                        response_json={
                            "detail": "Request validation failed",
                            "success": False,
                        },
                    )
                return original_fetch_full_document(
                    document_id=document_id,
                    origin_uri=origin_uri,
                    include_content=include_content,
                    max_content_chars=max_content_chars,
                )

            remote_state.client.fetch_full_document = _fetch_full_document  # type: ignore[method-assign]
            fresh_store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "fresh" / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )

            remote_view = fresh_store.probe_remote_current_view()

        self.assertIsNotNone(remote_view)
        self.assertEqual(remote_state.probe_calls, [])
        self.assertEqual(remote_state.load_calls, [])
        graph_head_retry_calls = [
            call
            for call in fetch_calls
            if isinstance(call["origin_uri"], str)
            and (
                str(call["origin_uri"]).endswith("/graph_nodes/catalog/current")
                or str(call["origin_uri"]).endswith("/graph_edges/catalog/current")
            )
        ]
        self.assertGreaterEqual(len(graph_head_retry_calls), 4)
        self.assertEqual(sum(1 for call in graph_head_retry_calls if not call["include_content"]), 2)
        self.assertEqual(sum(1 for call in graph_head_retry_calls if call["include_content"]), 2)

    def test_probe_remote_current_view_for_readiness_avoids_full_retry_when_metadata_only_graph_heads_get_400(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            primary_store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )
            self.assertTrue(primary_store.ensure_remote_snapshot())
            original_fetch_full_document = remote_state.client.fetch_full_document
            fetch_calls: list[dict[str, object]] = []

            def _fetch_full_document(*, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
                fetch_calls.append(
                    {
                        "document_id": document_id,
                        "origin_uri": origin_uri,
                        "include_content": include_content,
                        "max_content_chars": max_content_chars,
                    }
                )
                if (
                    not include_content
                    and isinstance(origin_uri, str)
                    and (
                        origin_uri.endswith("/graph_nodes/catalog/current")
                        or origin_uri.endswith("/graph_edges/catalog/current")
                    )
                ):
                    raise ChonkyDBError(
                        "ChonkyDB request failed for GET /v1/external/documents/full",
                        status_code=400,
                        response_json={
                            "detail": "Request validation failed",
                            "success": False,
                        },
                    )
                return original_fetch_full_document(
                    document_id=document_id,
                    origin_uri=origin_uri,
                    include_content=include_content,
                    max_content_chars=max_content_chars,
                )

            remote_state.client.fetch_full_document = _fetch_full_document  # type: ignore[method-assign]
            fresh_store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "fresh" / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )

            remote_view = fresh_store.probe_remote_current_view_for_readiness()

        self.assertIsNotNone(remote_view)
        graph_head_calls = [
            call
            for call in fetch_calls
            if isinstance(call["origin_uri"], str)
            and (
                str(call["origin_uri"]).endswith("/graph_nodes/catalog/current")
                or str(call["origin_uri"]).endswith("/graph_edges/catalog/current")
            )
        ]
        self.assertGreaterEqual(len(graph_head_calls), 1)
        self.assertTrue(all(not call["include_content"] for call in graph_head_calls))

    def test_graph_readiness_bootstrap_keeps_fresh_empty_namespace_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )

            created = store.ensure_remote_snapshot_for_readiness()
            probed = store.probe_remote_current_view_for_readiness()
            loaded = store.load_remote_current_view_for_readiness()

        self.assertFalse(created)
        self.assertIsNotNone(probed)
        self.assertIsNotNone(loaded)
        assert probed is not None
        assert loaded is not None
        self.assertTrue(probed["synthetic_empty"])
        self.assertEqual(probed["subject_node_id"], "user:main")
        self.assertEqual(probed["graph_id"], "graph:user_main")
        self.assertEqual(probed["topology_refs"], {"user:main": "bootstrap_empty:user:main"})
        self.assertEqual(probed["generation_id"], loaded["generation_id"])
        self.assertEqual(probed["topology_index_name"], loaded["topology_index_name"])
        self.assertFalse(remote_state.client.graph_store_many_calls)
        self.assertEqual(
            [
                uri
                for uri in remote_state.client.records_by_uri
                if uri.endswith("/graph_nodes/catalog/current") or uri.endswith("/graph_edges/catalog/current")
            ],
            [],
        )

    def test_graph_readiness_bootstrap_uses_probe_mode_for_empty_graph(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )
            original_fetch_full_document = remote_state.client.fetch_full_document
            fetch_calls: list[dict[str, object]] = []

            def _fetch_full_document(*, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
                fetch_calls.append(
                    {
                        "document_id": document_id,
                        "origin_uri": origin_uri,
                        "include_content": include_content,
                        "max_content_chars": max_content_chars,
                    }
                )
                return original_fetch_full_document(
                    document_id=document_id,
                    origin_uri=origin_uri,
                    include_content=include_content,
                    max_content_chars=max_content_chars,
                )

            remote_state.client.fetch_full_document = _fetch_full_document  # type: ignore[method-assign]

            created = store.ensure_remote_snapshot_for_readiness()

        self.assertFalse(created)
        graph_head_calls = [
            call
            for call in fetch_calls
            if isinstance(call["origin_uri"], str)
            and (
                str(call["origin_uri"]).endswith("/graph_nodes/catalog/current")
                or str(call["origin_uri"]).endswith("/graph_edges/catalog/current")
            )
        ]
        self.assertGreaterEqual(len(graph_head_calls), 1)
        self.assertTrue(all(not call["include_content"] for call in graph_head_calls))

    def test_load_remote_current_view_uses_full_head_when_metadata_only_probe_contract_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            primary_store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )
            self.assertTrue(primary_store.ensure_remote_snapshot())
            remote_state.probe_calls.clear()
            remote_state.load_calls.clear()
            original_fetch_full_document = remote_state.client.fetch_full_document
            fetch_calls: list[dict[str, object]] = []

            def _fetch_full_document(*, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
                fetch_calls.append(
                    {
                        "document_id": document_id,
                        "origin_uri": origin_uri,
                        "include_content": include_content,
                        "max_content_chars": max_content_chars,
                    }
                )
                payload = original_fetch_full_document(
                    document_id=document_id,
                    origin_uri=origin_uri,
                    include_content=include_content,
                    max_content_chars=max_content_chars,
                )
                if (
                    not include_content
                    and isinstance(origin_uri, str)
                    and origin_uri.endswith("/graph_edges/catalog/current")
                    and isinstance(payload, dict)
                ):
                    full_body = payload.get("body")
                    if not isinstance(full_body, dict):
                        nested_payload = payload.get("payload")
                        full_body = nested_payload.get("body") if isinstance(nested_payload, dict) else None
                    stripped_body = {
                        "schema": str(full_body.get("schema") or ""),
                        "version": full_body.get("version"),
                        "items_count": full_body.get("items_count"),
                        "segments": [],
                    }
                    return {
                        "document_id": payload.get("document_id"),
                        "uri": payload.get("uri"),
                        "snapshot_kind": payload.get("snapshot_kind"),
                        "payload": dict(stripped_body),
                        "body": dict(stripped_body),
                        "content": None,
                    }
                return payload

            remote_state.client.fetch_full_document = _fetch_full_document  # type: ignore[method-assign]
            fresh_store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "fresh" / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )

            self.assertIsNone(fresh_store.probe_remote_current_view())
            probe_call_count_after_probe = len(remote_state.probe_calls)
            load_call_count_after_probe = len(remote_state.load_calls)
            remote_view = fresh_store.load_remote_current_view()

        self.assertIsNotNone(remote_view)
        assert remote_view is not None
        self.assertEqual(remote_view["subject_node_id"], "user:main")
        self.assertEqual(remote_view["graph_id"], "graph:user_main")
        self.assertEqual(len(remote_state.probe_calls), probe_call_count_after_probe)
        self.assertEqual(len(remote_state.load_calls), load_call_count_after_probe)
        graph_head_calls = [
            call
            for call in fetch_calls
            if isinstance(call["origin_uri"], str)
            and (
                str(call["origin_uri"]).endswith("/graph_nodes/catalog/current")
                or str(call["origin_uri"]).endswith("/graph_edges/catalog/current")
            )
        ]
        self.assertGreaterEqual(len(graph_head_calls), 4)
        self.assertGreaterEqual(sum(1 for call in graph_head_calls if not call["include_content"]), 2)
        self.assertGreaterEqual(sum(1 for call in graph_head_calls if call["include_content"]), 2)

    def test_remote_graph_load_prefers_remote_current_view_over_stale_local_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            path = Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json"
            store = TwinrPersonalGraphStore(
                path=path,
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )
            store.remember_plan(summary="go for a walk", when_text="today")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "schema": {"name": "twinr_graph", "version": 2},
                        "subject_node_id": "user:main",
                        "graph_id": "graph:user_main",
                        "created_at": "2026-03-14T08:00:00Z",
                        "updated_at": "2026-03-14T08:05:00Z",
                        "nodes": [{"id": "user:main", "type": "user", "label": "Erika"}],
                        "edges": [],
                        "metadata": {"kind": "stale_local_graph"},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            loaded = store.load_document()

        self.assertEqual(loaded.metadata["kind"], "personal_graph")
        self.assertTrue(any(edge.edge_type == "user_plans" for edge in loaded.edges))
        self.assertFalse(path.exists())

    def test_remote_graph_load_uses_topology_refs_when_node_catalog_segments_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )
            store.remember_plan(summary="go for a walk", when_text="today")
            lagging_uris = tuple(
                uri
                for uri in remote_state.client.records_by_uri
                if "/graph_nodes/catalog/segment/" in uri
            )
            for uri in lagging_uris:
                record = remote_state.client.records_by_uri.pop(uri, None)
                if isinstance(record, dict):
                    remote_state.client.records_by_document_id.pop(str(record.get("document_id") or ""), None)

            loaded = store.load_document()

        self.assertEqual(loaded.subject_node_id, "user:main")
        self.assertTrue(any(node.node_id == "user:main" for node in loaded.nodes))
        self.assertTrue(any(edge.edge_type == "user_plans" for edge in loaded.edges))

    def test_remote_graph_load_uses_catalog_projections_when_exact_item_documents_do_not_hydrate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )
            store.remember_plan(summary="go for a walk", when_text="today")
            original_fetch = remote_state.client.fetch_full_document
            unreadable_origin_uris = {
                entry.uri
                for entry in store._remote_graph._catalog.load_catalog_entries(snapshot_kind="graph_nodes")
            }
            unreadable_origin_uris.update(
                entry.uri
                for entry in store._remote_graph._catalog.load_catalog_entries(snapshot_kind="graph_edges")
            )

            def _fetch_without_payload(
                *,
                document_id=None,
                origin_uri=None,
                include_content=True,
                max_content_chars=4000,
            ):
                if origin_uri in unreadable_origin_uris:
                    return {
                        "success": True,
                        "document_id": document_id,
                        "origin_uri": origin_uri,
                        "chunk_count": 0,
                        "chunks": [],
                    }
                return original_fetch(
                    document_id=document_id,
                    origin_uri=origin_uri,
                    include_content=include_content,
                    max_content_chars=max_content_chars,
                )

            remote_state.client.fetch_full_document = _fetch_without_payload
            loaded = store.load_document()

        self.assertEqual(loaded.subject_node_id, "user:main")
        self.assertTrue(any(node.node_id == "user:main" for node in loaded.nodes))
        self.assertTrue(any(edge.edge_type == "user_plans" for edge in loaded.edges))

    def test_remote_current_path_maps_generation_refs_back_to_logical_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )
            store.apply_candidate_edges(
                (
                    LongTermGraphEdgeCandidateV1(
                        source_ref="user:main",
                        edge_type="social_related_to_user",
                        target_ref="person:janina",
                        confidence=0.95,
                        confirmed_by_user=True,
                        attributes={"relation": "wife"},
                    ),
                    LongTermGraphEdgeCandidateV1(
                        source_ref="person:janina",
                        edge_type="spatial_located_in",
                        target_ref="place:eye_doctor",
                        confidence=0.88,
                        confirmed_by_user=True,
                    ),
                )
            )

            path_payload = store.query_remote_current_path(
                source_node_id="user:main",
                target_node_id="place:eye_doctor",
            )
            neighbor_payload = store.query_remote_current_neighbors(node_id="person:janina", limit=4)

        self.assertIsNotNone(path_payload)
        self.assertEqual(path_payload["logical_path"], ["user:main", "person:janina", "place:eye_doctor"])
        self.assertIsNotNone(neighbor_payload)
        self.assertEqual(
            [item["logical_node_id"] for item in neighbor_payload["neighbors"]],
            ["place:eye_doctor"],
        )
        self.assertTrue(remote_state.client.graph_path_calls)
        self.assertTrue(remote_state.client.graph_neighbors_calls)

    def test_build_prompt_context_prefers_remote_query_first_subgraph_and_emits_query_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )
            store.remember_preference(
                category="brand",
                value="Melitta",
                for_product="coffee",
            )
            remote_state.load_calls.clear()

            context = store.build_prompt_context("Wo kann ich heute Kaffee kaufen?")

        self.assertIsNotNone(context)
        payload = _memory_payload(context or "")
        self.assertEqual(payload["query_plan"]["mode"], "remote_query_first_subgraph")
        self.assertIn("Melitta", json.dumps(payload, ensure_ascii=False))
        self.assertTrue(remote_state.client.graph_path_calls)
        self.assertTrue(remote_state.client.graph_neighbors_calls)
        self.assertEqual(remote_state.load_calls, [])

    def test_build_prompt_context_retries_transient_remote_graph_expansion_queries_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _TransientGraphQueryFailingClient(path_failures=1, neighbor_failures=1)
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )
            store.remember_contact(
                given_name="Corinna",
                family_name="Maier",
                phone=_TEST_CORINNA_PHONE,
                role="Physiotherapist",
            )
            remote_state.load_calls.clear()

            context = store.build_prompt_context("Was ist Corinnas Nummer?")

        self.assertIsNotNone(context)
        payload = _memory_payload(context or "")
        self.assertEqual(payload["query_plan"]["mode"], "remote_query_first_subgraph")
        self.assertEqual(payload["query_plan"]["path_query_events"][0]["status"], "retried_ok")
        self.assertEqual(payload["query_plan"]["neighbor_query_events"][0]["status"], "retried_ok")
        self.assertGreaterEqual(len(remote_state.client.graph_path_calls), 2)
        self.assertGreaterEqual(len(remote_state.client.graph_neighbors_calls), 2)
        self.assertEqual(remote_state.load_calls, [])

    def test_select_context_selection_uses_current_head_projections_when_fresh_reader_cannot_read_exact_graph_item_docs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            primary_store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )
            primary_store.remember_contact(
                given_name="Corinna",
                family_name="Maier",
                phone=_TEST_CORINNA_PHONE,
                role="Physiotherapist",
            )
            unreadable_origin_uris = {
                entry.uri
                for snapshot_kind in ("graph_nodes", "graph_edges")
                for entry in primary_store._remote_graph._catalog.load_catalog_entries(snapshot_kind=snapshot_kind)
            }
            unreadable_document_ids = {
                str(entry.document_id)
                for snapshot_kind in ("graph_nodes", "graph_edges")
                for entry in primary_store._remote_graph._catalog.load_catalog_entries(snapshot_kind=snapshot_kind)
                if isinstance(entry.document_id, str) and entry.document_id
            }
            original_fetch = remote_state.client.fetch_full_document
            fetch_calls: list[dict[str, object]] = []

            def _fetch_without_exact_graph_item_documents(
                *,
                document_id=None,
                origin_uri=None,
                include_content=True,
                max_content_chars=4000,
            ):
                fetch_calls.append(
                    {
                        "document_id": document_id,
                        "origin_uri": origin_uri,
                        "include_content": include_content,
                        "max_content_chars": max_content_chars,
                    }
                )
                if (
                    isinstance(document_id, str)
                    and document_id in unreadable_document_ids
                ) or (
                    isinstance(origin_uri, str)
                    and origin_uri in unreadable_origin_uris
                ):
                    raise LongTermRemoteUnavailableError("exact graph item document unavailable")
                return original_fetch(
                    document_id=document_id,
                    origin_uri=origin_uri,
                    include_content=include_content,
                    max_content_chars=max_content_chars,
                )

            remote_state.client.fetch_full_document = _fetch_without_exact_graph_item_documents  # type: ignore[method-assign]
            fresh_store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "fresh" / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )
            fresh_store._remote_graph._catalog.top_catalog_entries = (  # type: ignore[method-assign]
                lambda *, snapshot_kind, limit, eligible=None, preserve_order=False: ()
            )

            selection = fresh_store.select_context_selection("Corinna Maier")

        self.assertIsNotNone(selection.query_plan)
        assert selection.query_plan is not None
        self.assertEqual(selection.query_plan["mode"], "remote_query_first_subgraph")
        self.assertIn("catalog_current_head", selection.query_plan["access_path"])
        self.assertTrue(any(node.label == "Corinna Maier" for node in selection.document.nodes))
        exact_item_fetches = [
            call
            for call in fetch_calls
            if (
                isinstance(call["document_id"], str)
                and str(call["document_id"]) in unreadable_document_ids
            ) or (
                isinstance(call["origin_uri"], str)
                and str(call["origin_uri"]) in unreadable_origin_uris
            )
        ]
        self.assertEqual(exact_item_fetches, [])

    def test_select_context_selection_and_render_prompt_context_reuse_the_same_query_first_subgraph(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )
            store.remember_contact(
                given_name="Corinna",
                family_name="Maier",
                phone=_TEST_CORINNA_PHONE,
                role="Physiotherapist",
            )
            remote_state.load_calls.clear()

            selection = store.select_context_selection("Was ist Corinnas Nummer?")
            context = store.render_prompt_context_selection(
                selection,
                query_text="Was ist Corinnas Nummer?",
            )

        self.assertIsNotNone(context)
        payload = _memory_payload(context or "")
        self.assertEqual(payload["query_plan"]["mode"], "remote_query_first_subgraph")
        self.assertIn("Corinna Maier", json.dumps(payload, ensure_ascii=False))
        self.assertEqual(remote_state.load_calls, [])

    def test_build_prompt_context_keeps_query_first_subgraph_when_graph_expansion_503_persists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _TransientGraphQueryFailingClient(path_failures=3, neighbor_failures=3)
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )
            store.remember_contact(
                given_name="Corinna",
                family_name="Maier",
                phone=_TEST_CORINNA_PHONE,
                role="Physiotherapist",
            )
            remote_state.load_calls.clear()

            context = store.build_prompt_context("Was ist Corinnas Nummer?")

        self.assertIsNotNone(context)
        payload = _memory_payload(context or "")
        self.assertEqual(payload["query_plan"]["mode"], "remote_query_first_subgraph")
        self.assertEqual(payload["query_plan"]["path_query_events"][0]["status"], "degraded")
        self.assertEqual(payload["query_plan"]["neighbor_query_events"][0]["status"], "degraded")
        self.assertEqual(remote_state.load_calls, [])

    def test_build_subtext_payload_prefers_remote_query_first_subgraph_and_keeps_query_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )
            store.remember_plan(
                summary="go for a walk",
                when_text="today",
                details="Prefer a short gentle walk.",
            )
            remote_state.load_calls.clear()

            payload = store.build_subtext_payload("go for a walk today")

        self.assertIsNotNone(payload)
        payload_dict = cast(dict[str, object], payload)  # pylint: disable=unsubscriptable-object
        query_plan = cast(dict[str, object], payload_dict["query_plan"])  # pylint: disable=unsubscriptable-object
        self.assertEqual(query_plan["mode"], "remote_query_first_subgraph")
        self.assertIn("situational_threads", payload_dict)
        self.assertTrue(remote_state.client.graph_path_calls)
        self.assertTrue(remote_state.client.graph_neighbors_calls)
        self.assertEqual(remote_state.load_calls, [])

    def test_rank_preference_prompt_items_keeps_exact_short_domain_term_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )
            store.remember_preference(
                category="brand",
                value="Teekanne",
                for_product="tea",
            )
            store.remember_preference(
                category="store",
                value="Laden Seidel",
                for_product="tea",
            )
            document = store.load_document()
            prompt_preferences = store._prompt_preferences(document, limit=128)
            ranked_preferences = store._rank_preference_prompt_items(
                prompt_preferences,
                query_text="Need tea today; ask what would be practical.",
                limit=3,
                fallback_limit=0,
            )

        self.assertEqual(
            sorted(str(item.get("value", "")) for item in ranked_preferences),
            ["Laden Seidel", "Teekanne"],
        )

    def test_optional_remote_unavailable_error_falls_back_to_empty_graph(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.load_error = LongTermRemoteUnavailableError("remote unavailable")
            store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )

            document = store.load_document()

        self.assertEqual(document.subject_node_id, "user:main")
        self.assertEqual(document.edges, ())
        self.assertFalse(store.path.exists())


if __name__ == "__main__":
    unittest.main()
