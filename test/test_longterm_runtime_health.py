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


class LongTermRemoteHealthProbeTests(unittest.TestCase):
    def _probe(
        self,
        *,
        prompt_state: _FakeRemoteState,
        object_state: _FakeRemoteState,
        graph_state: _FakeRemoteState,
        midterm_state: _FakeRemoteState,
    ) -> LongTermRemoteHealthProbe:
        return LongTermRemoteHealthProbe(
            prompt_context_store=SimpleNamespace(
                memory_store=SimpleNamespace(remote_state=prompt_state, remote_snapshot_kind="prompt_memory"),
                user_store=SimpleNamespace(remote_state=prompt_state, remote_snapshot_kind="user_context"),
                personality_store=SimpleNamespace(remote_state=prompt_state, remote_snapshot_kind="personality_context"),
            ),
            object_store=SimpleNamespace(remote_state=object_state),
            graph_store=SimpleNamespace(remote_state=graph_state),
            midterm_store=SimpleNamespace(remote_state=midterm_state),
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
        graph_state = _FakeRemoteState({"graph": {"schema": "graph", "nodes": [], "edges": []}})
        midterm_state = _FakeRemoteState({"midterm": {"schema": "midterm", "packets": []}})

        result = self._probe(
            prompt_state=prompt_state,
            object_state=object_state,
            graph_state=graph_state,
            midterm_state=midterm_state,
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
        graph_state = _FakeRemoteState({"graph": {"schema": "graph", "nodes": [], "edges": []}})
        midterm_state = _FakeRemoteState({"midterm": {"schema": "midterm", "packets": []}})

        with self.assertRaises(LongTermRemoteUnavailableError):
            self._probe(
                prompt_state=prompt_state,
                object_state=object_state,
                graph_state=graph_state,
                midterm_state=midterm_state,
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
        graph_state = _ProbeAwareFakeRemoteState({"graph": {"schema": "graph", "nodes": [], "edges": []}})
        midterm_state = _ProbeAwareFakeRemoteState({"midterm": {"schema": "midterm", "packets": []}})

        result = self._probe(
            prompt_state=prompt_state,
            object_state=object_state,
            graph_state=graph_state,
            midterm_state=midterm_state,
        ).probe_operational()

        self.assertTrue(result.ready)
        self.assertEqual(result.health_tier, "ready")
        self.assertTrue(result.archive_checked)
        self.assertTrue(result.archive_safe)
        for state in (prompt_state, object_state, graph_state, midterm_state):
            self.assertTrue(all(call["prefer_cached_document_id"] for call in state.probe_calls))
            self.assertTrue(all(call["prefer_metadata_only"] for call in state.probe_calls))

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
        graph_state = _ProbeAwareFakeRemoteState({"graph": {"schema": "graph", "nodes": [], "edges": []}})
        midterm_state = _ProbeAwareFakeRemoteState({"midterm": {"schema": "midterm", "packets": []}})

        result = self._probe(
            prompt_state=prompt_state,
            object_state=object_state,
            graph_state=graph_state,
            midterm_state=midterm_state,
        ).probe_operational(include_archive=False)

        self.assertTrue(result.ready)
        self.assertEqual(result.health_tier, "degraded")
        self.assertFalse(result.archive_checked)
        self.assertFalse(result.archive_safe)
        self.assertNotIn("archive", result.checked_snapshots)
        self.assertEqual(
            [call["snapshot_kind"] for call in object_state.probe_calls],
            ["objects", "conflicts"],
        )


if __name__ == "__main__":
    unittest.main()
