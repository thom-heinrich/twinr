from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.longterm.runtime.health import LongTermRemoteHealthProbe
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStatus, LongTermRemoteUnavailableError


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


if __name__ == "__main__":
    unittest.main()
