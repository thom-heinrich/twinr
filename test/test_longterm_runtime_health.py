from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.longterm.runtime.health import LongTermRemoteHealthProbe
from twinr.memory.longterm.storage.remote_state import (
    LongTermRemoteSnapshotProbe,
    LongTermRemoteStatus,
    LongTermRemoteUnavailableError,
)


class _FakeRemoteState:
    def __init__(self, payloads: dict[str, dict[str, object] | Exception]) -> None:
        self.enabled = True
        self.required = True
        self._payloads = dict(payloads)
        self.calls: list[str] = []

    def status(self) -> LongTermRemoteStatus:
        return LongTermRemoteStatus(mode="remote_primary", ready=True)

    def load_snapshot(self, *, snapshot_kind: str):
        self.calls.append(snapshot_kind)
        payload = self._payloads[snapshot_kind]
        if isinstance(payload, Exception):
            raise payload
        return payload


class _ProbeAwareFakeRemoteState(_FakeRemoteState):
    def __init__(self, payloads: dict[str, dict[str, object] | Exception]) -> None:
        super().__init__(payloads)
        self.probe_calls: list[dict[str, object]] = []

    def probe_snapshot_load(
        self,
        *,
        snapshot_kind: str,
        local_path=None,
        prefer_cached_document_id: bool = False,
        prefer_metadata_only: bool = False,
        fast_fail: bool = False,
    ):
        del local_path, fast_fail
        self.probe_calls.append(
            {
                "snapshot_kind": snapshot_kind,
                "prefer_cached_document_id": prefer_cached_document_id,
                "prefer_metadata_only": prefer_metadata_only,
            }
        )
        payload = self._payloads[snapshot_kind]
        if isinstance(payload, Exception):
            raise payload
        return LongTermRemoteSnapshotProbe(
            snapshot_kind=snapshot_kind,
            status="found",
            latency_ms=1.0,
            selected_source="cached_document",
            payload=dict(payload),
        )


class _FakeGraphStore:
    def __init__(
        self,
        *,
        remote_state: _FakeRemoteState,
        payload: dict[str, object] | Exception | None,
        load_payload: dict[str, object] | Exception | None = None,
    ) -> None:
        self.remote_state = remote_state
        self._payload = payload
        self._load_payload = load_payload
        self.load_calls = 0

    def probe_remote_current_view(self) -> dict[str, object] | None:
        if isinstance(self._payload, Exception):
            raise self._payload
        return None if self._payload is None else dict(self._payload)

    def load_remote_current_view(self) -> dict[str, object] | None:
        self.load_calls += 1
        if isinstance(self._load_payload, Exception):
            raise self._load_payload
        return None if self._load_payload is None else dict(self._load_payload)


class _FakeMidtermStore:
    def __init__(
        self,
        *,
        remote_state: _FakeRemoteState,
        payload: dict[str, object] | Exception | None,
        load_payload: dict[str, object] | Exception | None = None,
    ) -> None:
        self.remote_state = remote_state
        self._payload = payload
        self._load_payload = load_payload
        self.load_calls = 0

    def probe_remote_current_head(self) -> dict[str, object] | None:
        if isinstance(self._payload, Exception):
            raise self._payload
        return None if self._payload is None else dict(self._payload)

    def load_remote_current_head(self) -> dict[str, object] | None:
        self.load_calls += 1
        if isinstance(self._load_payload, Exception):
            raise self._load_payload
        return None if self._load_payload is None else dict(self._load_payload)


class _FakeCurrentHeadStore:
    def __init__(
        self,
        *,
        remote_state: _FakeRemoteState,
        remote_snapshot_kind: str,
        payload: dict[str, object] | Exception | None,
        load_payload: dict[str, object] | Exception | None = None,
    ) -> None:
        self.remote_state = remote_state
        self.remote_snapshot_kind = remote_snapshot_kind
        self._payload = payload
        self._load_payload = load_payload
        self.load_calls = 0

    def probe_remote_current_head(self) -> dict[str, object] | None:
        if isinstance(self._payload, Exception):
            raise self._payload
        return None if self._payload is None else dict(self._payload)

    def load_remote_current_head(self) -> dict[str, object] | None:
        self.load_calls += 1
        if isinstance(self._load_payload, Exception):
            raise self._load_payload
        return None if self._load_payload is None else dict(self._load_payload)


class LongTermRemoteHealthProbeTests(unittest.TestCase):
    def _probe(
        self,
        *,
        prompt_state: _FakeRemoteState,
        object_state: _FakeRemoteState,
        graph_store: _FakeGraphStore,
        midterm_store: _FakeMidtermStore,
        object_store: object | None = None,
        memory_store: object | None = None,
        user_store: object | None = None,
        personality_store: object | None = None,
    ) -> LongTermRemoteHealthProbe:
        return LongTermRemoteHealthProbe(
            prompt_context_store=SimpleNamespace(
                memory_store=memory_store
                or SimpleNamespace(remote_state=prompt_state, remote_snapshot_kind="prompt_memory"),
                user_store=user_store
                or SimpleNamespace(remote_state=prompt_state, remote_snapshot_kind="user_context"),
                personality_store=personality_store
                or SimpleNamespace(remote_state=prompt_state, remote_snapshot_kind="personality_context"),
            ),
            object_store=object_store or SimpleNamespace(remote_state=object_state),
            graph_store=graph_store,
            midterm_store=midterm_store,
        )

    def test_ensure_operational_checks_required_snapshots_and_shards(self) -> None:
        prompt_state = _FakeRemoteState(
            {
                "prompt_memory": {"schema": "prompt_memory", "entries": []},
                "user_context": {"schema": "managed_context", "entries": []},
                "personality_context": {"schema": "managed_context", "entries": []},
            }
        )
        object_state = _FakeRemoteState(
            {
                "objects": {"schema": "twinr_memory_object_catalog_v2", "version": 2, "items": []},
                "conflicts": {"schema": "conflicts", "conflicts": []},
                "archive": {"schema": "twinr_memory_archive_catalog_v2", "version": 2, "items": []},
            }
        )
        graph_state = _FakeRemoteState({})
        graph_store = _FakeGraphStore(
            remote_state=graph_state,
            payload={
                "generation_id": "gen-1",
                "topology_index_name": "twinr_graph_test",
                "subject_node_id": "user:main",
                "graph_id": "graph:user_main",
                "created_at": "2026-03-29T12:00:00Z",
                "updated_at": "2026-03-29T12:00:00Z",
                "topology_refs": {"user:main": "gen-1:user:main"},
            },
        )
        midterm_state = _FakeRemoteState({})
        midterm_store = _FakeMidtermStore(
            remote_state=midterm_state,
            payload={"schema": "twinr_memory_midterm_catalog_v3", "version": 3, "items_count": 0, "segments": []},
        )

        result = self._probe(
            prompt_state=prompt_state,
            object_state=object_state,
            graph_store=graph_store,
            midterm_store=midterm_store,
        ).ensure_operational()

        self.assertEqual(
            result.checked_snapshots,
            (
                "prompt_memory",
                "user_context",
                "personality_context",
                "objects",
                "conflicts",
                "archive",
                "graph",
                "midterm",
            ),
        )

    def test_ensure_operational_raises_when_any_required_shard_fails(self) -> None:
        prompt_state = _FakeRemoteState(
            {
                "prompt_memory": {"schema": "prompt_memory", "entries": []},
                "user_context": {"schema": "managed_context", "entries": []},
                "personality_context": {"schema": "managed_context", "entries": []},
            }
        )
        object_state = _FakeRemoteState(
            {
                "objects": LongTermRemoteUnavailableError("objects catalog unavailable"),
                "conflicts": {"schema": "conflicts", "conflicts": []},
                "archive": {"schema": "twinr_memory_archive_catalog_v2", "version": 2, "items": []},
            }
        )
        graph_state = _FakeRemoteState({})
        graph_store = _FakeGraphStore(
            remote_state=graph_state,
            payload={
                "generation_id": "gen-1",
                "topology_index_name": "twinr_graph_test",
                "subject_node_id": "user:main",
                "graph_id": "graph:user_main",
                "created_at": "2026-03-29T12:00:00Z",
                "updated_at": "2026-03-29T12:00:00Z",
                "topology_refs": {"user:main": "gen-1:user:main"},
            },
        )
        midterm_state = _FakeRemoteState({})
        midterm_store = _FakeMidtermStore(
            remote_state=midterm_state,
            payload={"schema": "twinr_memory_midterm_catalog_v3", "version": 3, "items_count": 0, "segments": []},
        )

        with self.assertRaises(LongTermRemoteUnavailableError):
            self._probe(
                prompt_state=prompt_state,
                object_state=object_state,
                graph_store=graph_store,
                midterm_store=midterm_store,
            ).ensure_operational()

    def test_probe_operational_prefers_cached_document_id_hints_for_probeable_states(self) -> None:
        prompt_state = _ProbeAwareFakeRemoteState(
            {
                "prompt_memory": {"schema": "prompt_memory", "entries": []},
                "user_context": {"schema": "managed_context", "entries": []},
                "personality_context": {"schema": "managed_context", "entries": []},
            }
        )
        object_state = _ProbeAwareFakeRemoteState(
            {
                "objects": {"schema": "twinr_memory_object_catalog_v2", "version": 2, "items": []},
                "conflicts": {"schema": "conflicts", "conflicts": []},
                "archive": {"schema": "twinr_memory_archive_catalog_v2", "version": 2, "items": []},
            }
        )
        graph_state = _ProbeAwareFakeRemoteState({})
        graph_store = _FakeGraphStore(
            remote_state=graph_state,
            payload={
                "generation_id": "gen-1",
                "topology_index_name": "twinr_graph_test",
                "subject_node_id": "user:main",
                "graph_id": "graph:user_main",
                "created_at": "2026-03-29T12:00:00Z",
                "updated_at": "2026-03-29T12:00:00Z",
                "topology_refs": {"user:main": "gen-1:user:main"},
            },
        )
        midterm_state = _ProbeAwareFakeRemoteState({})
        midterm_store = _FakeMidtermStore(
            remote_state=midterm_state,
            payload={"schema": "twinr_memory_midterm_catalog_v3", "version": 3, "items_count": 0, "segments": []},
        )

        result = self._probe(
            prompt_state=prompt_state,
            object_state=object_state,
            graph_store=graph_store,
            midterm_store=midterm_store,
        ).probe_operational()

        self.assertTrue(result.ready)
        self.assertEqual(result.health_tier, "ready")
        self.assertTrue(result.archive_checked)
        self.assertTrue(result.archive_safe)
        proof_contract = result.proof_contract()
        self.assertEqual(proof_contract["contract_id"], "configured_namespace_archive_inclusive_readiness")
        self.assertIn(
            "fresh-reader item lookups after retention",
            proof_contract["operations_not_proved"],
        )
        for state in (prompt_state, object_state):
            self.assertTrue(all(call["prefer_cached_document_id"] for call in state.probe_calls))
            self.assertTrue(all(call["prefer_metadata_only"] for call in state.probe_calls))
        selected_sources = {check.snapshot_kind: check.selected_source for check in result.checks}
        self.assertEqual(selected_sources["graph"], "graph_current_view")
        self.assertEqual(selected_sources["midterm"], "catalog_current_head")

    def test_probe_operational_prefers_object_store_current_contract_over_legacy_snapshot_probe(self) -> None:
        prompt_state = _ProbeAwareFakeRemoteState(
            {
                "prompt_memory": {"schema": "prompt_memory", "entries": []},
                "user_context": {"schema": "managed_context", "entries": []},
                "personality_context": {"schema": "managed_context", "entries": []},
            }
        )
        object_state = _ProbeAwareFakeRemoteState(
            {
                "objects": LongTermRemoteUnavailableError("legacy objects snapshot must not be probed"),
                "conflicts": LongTermRemoteUnavailableError("legacy conflicts snapshot must not be probed"),
                "archive": LongTermRemoteUnavailableError("legacy archive snapshot must not be probed"),
            }
        )
        graph_state = _FakeRemoteState({})
        graph_store = _FakeGraphStore(
            remote_state=graph_state,
            payload={
                "generation_id": "gen-1",
                "topology_index_name": "twinr_graph_test",
                "subject_node_id": "user:main",
                "graph_id": "graph:user_main",
                "created_at": "2026-03-29T12:00:00Z",
                "updated_at": "2026-03-29T12:00:00Z",
                "topology_refs": {"user:main": "gen-1:user:main"},
            },
        )
        midterm_state = _FakeRemoteState({})
        midterm_store = _FakeMidtermStore(
            remote_state=midterm_state,
            payload={"schema": "twinr_memory_midterm_catalog_v3", "version": 3, "items_count": 0, "segments": []},
        )
        object_store = SimpleNamespace(
            remote_state=object_state,
            probe_remote_current_snapshot=lambda *, snapshot_kind: {
                "schema": f"twinr_memory_{snapshot_kind}_catalog_v3",
                "version": 3,
                "items_count": 0,
                "segments": [],
            },
        )

        result = self._probe(
            prompt_state=prompt_state,
            object_state=object_state,
            graph_store=graph_store,
            midterm_store=midterm_store,
            object_store=object_store,
        ).probe_operational()

        self.assertTrue(result.ready)
        self.assertEqual(object_state.probe_calls, [])
        selected_sources = {check.snapshot_kind: check.selected_source for check in result.checks}
        self.assertEqual(selected_sources["objects"], "object_store_current_contract")
        self.assertEqual(selected_sources["conflicts"], "object_store_current_contract")
        self.assertEqual(selected_sources["archive"], "object_store_current_contract")

    def test_probe_operational_prefers_prompt_store_current_contract_over_legacy_snapshot_probe(self) -> None:
        prompt_state = _ProbeAwareFakeRemoteState(
            {
                "prompt_memory": LongTermRemoteUnavailableError("legacy prompt snapshot must not be probed"),
                "user_context": LongTermRemoteUnavailableError("legacy user-context snapshot must not be probed"),
                "personality_context": LongTermRemoteUnavailableError("legacy personality-context snapshot must not be probed"),
            }
        )
        object_state = _ProbeAwareFakeRemoteState(
            {
                "objects": {"schema": "twinr_memory_object_catalog_v2", "version": 2, "items": []},
                "conflicts": {"schema": "conflicts", "conflicts": []},
                "archive": {"schema": "twinr_memory_archive_catalog_v2", "version": 2, "items": []},
            }
        )
        graph_state = _FakeRemoteState({})
        graph_store = _FakeGraphStore(
            remote_state=graph_state,
            payload={
                "generation_id": "gen-1",
                "topology_index_name": "twinr_graph_test",
                "subject_node_id": "user:main",
                "graph_id": "graph:user_main",
                "created_at": "2026-03-29T12:00:00Z",
                "updated_at": "2026-03-29T12:00:00Z",
                "topology_refs": {"user:main": "gen-1:user:main"},
            },
        )
        midterm_state = _FakeRemoteState({})
        midterm_store = _FakeMidtermStore(
            remote_state=midterm_state,
            payload={"schema": "twinr_memory_midterm_catalog_v3", "version": 3, "items_count": 0, "segments": []},
        )

        result = self._probe(
            prompt_state=prompt_state,
            object_state=object_state,
            graph_store=graph_store,
            midterm_store=midterm_store,
            memory_store=_FakeCurrentHeadStore(
                remote_state=prompt_state,
                remote_snapshot_kind="prompt_memory",
                payload={"schema": "twinr_prompt_memory_catalog_v3", "version": 3, "items_count": 0, "segments": []},
            ),
            user_store=_FakeCurrentHeadStore(
                remote_state=prompt_state,
                remote_snapshot_kind="user_context",
                payload={"schema": "twinr_user_context_catalog_v3", "version": 3, "items_count": 0, "segments": []},
            ),
            personality_store=_FakeCurrentHeadStore(
                remote_state=prompt_state,
                remote_snapshot_kind="personality_context",
                payload={"schema": "twinr_personality_context_catalog_v3", "version": 3, "items_count": 0, "segments": []},
            ),
        ).probe_operational()

        self.assertTrue(result.ready)
        self.assertEqual(prompt_state.probe_calls, [])
        selected_sources = {check.snapshot_kind: check.selected_source for check in result.checks}
        self.assertEqual(selected_sources["prompt_memory"], "catalog_current_head")
        self.assertEqual(selected_sources["user_context"], "catalog_current_head")
        self.assertEqual(selected_sources["personality_context"], "catalog_current_head")

    def test_probe_operational_can_skip_archive_for_steady_state(self) -> None:
        prompt_state = _ProbeAwareFakeRemoteState(
            {
                "prompt_memory": {"schema": "prompt_memory", "entries": []},
                "user_context": {"schema": "managed_context", "entries": []},
                "personality_context": {"schema": "managed_context", "entries": []},
            }
        )
        object_state = _ProbeAwareFakeRemoteState(
            {
                "objects": {"schema": "twinr_memory_object_catalog_v2", "version": 2, "items": []},
                "conflicts": {"schema": "conflicts", "conflicts": []},
                "archive": {"schema": "twinr_memory_archive_catalog_v2", "version": 2, "items": []},
            }
        )
        graph_state = _ProbeAwareFakeRemoteState({})
        graph_store = _FakeGraphStore(
            remote_state=graph_state,
            payload={
                "generation_id": "gen-1",
                "topology_index_name": "twinr_graph_test",
                "subject_node_id": "user:main",
                "graph_id": "graph:user_main",
                "created_at": "2026-03-29T12:00:00Z",
                "updated_at": "2026-03-29T12:00:00Z",
                "topology_refs": {"user:main": "gen-1:user:main"},
            },
        )
        midterm_state = _ProbeAwareFakeRemoteState({})
        midterm_store = _FakeMidtermStore(
            remote_state=midterm_state,
            payload={"schema": "twinr_memory_midterm_catalog_v3", "version": 3, "items_count": 0, "segments": []},
        )

        result = self._probe(
            prompt_state=prompt_state,
            object_state=object_state,
            graph_store=graph_store,
            midterm_store=midterm_store,
        ).probe_operational(include_archive=False)

        self.assertTrue(result.ready)
        self.assertEqual(result.health_tier, "degraded")
        self.assertFalse(result.archive_checked)
        self.assertFalse(result.archive_safe)
        proof_contract = result.proof_contract()
        self.assertEqual(proof_contract["contract_id"], "configured_namespace_current_only_readiness")
        self.assertEqual(proof_contract["mutation_scope"], "read_only")
        self.assertNotIn("archive", result.checked_snapshots)
        self.assertEqual(
            [call["snapshot_kind"] for call in object_state.probe_calls],
            ["objects", "conflicts"],
        )

    def test_probe_operational_falls_back_to_loadable_graph_and_midterm_current_heads(self) -> None:
        prompt_state = _ProbeAwareFakeRemoteState(
            {
                "prompt_memory": {"schema": "prompt_memory", "entries": []},
                "user_context": {"schema": "managed_context", "entries": []},
                "personality_context": {"schema": "managed_context", "entries": []},
            }
        )
        object_state = _ProbeAwareFakeRemoteState(
            {
                "objects": {"schema": "twinr_memory_object_catalog_v2", "version": 2, "items": []},
                "conflicts": {"schema": "conflicts", "conflicts": []},
                "archive": {"schema": "twinr_memory_archive_catalog_v2", "version": 2, "items": []},
            }
        )
        graph_state = _FakeRemoteState({})
        graph_store = _FakeGraphStore(
            remote_state=graph_state,
            payload=LongTermRemoteUnavailableError("graph probe lagged"),
            load_payload={
                "generation_id": "gen-1",
                "topology_index_name": "twinr_graph_test",
                "subject_node_id": "user:main",
                "graph_id": "graph:user_main",
                "created_at": "2026-03-29T12:00:00Z",
                "updated_at": "2026-03-29T12:00:00Z",
                "topology_refs": {"user:main": "gen-1:user:main"},
            },
        )
        midterm_state = _FakeRemoteState({})
        midterm_store = _FakeMidtermStore(
            remote_state=midterm_state,
            payload=None,
            load_payload={"schema": "twinr_memory_midterm_catalog_v3", "version": 3, "items_count": 0, "segments": []},
        )

        result = self._probe(
            prompt_state=prompt_state,
            object_state=object_state,
            graph_store=graph_store,
            midterm_store=midterm_store,
        ).probe_operational()

        self.assertTrue(result.ready)
        selected_sources = {check.snapshot_kind: check.selected_source for check in result.checks}
        self.assertEqual(selected_sources["graph"], "graph_current_view_load_contract")
        self.assertEqual(selected_sources["midterm"], "catalog_current_head_load_contract")
        self.assertEqual(graph_store.load_calls, 1)
        self.assertEqual(midterm_store.load_calls, 1)


if __name__ == "__main__":
    unittest.main()
