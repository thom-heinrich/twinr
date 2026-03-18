from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.memory.chonkydb import TwinrPersonalGraphStore
from twinr.memory.longterm.core.models import LongTermGraphEdgeCandidateV1


class _FakeRemoteState:
    def __init__(self) -> None:
        self.enabled = True
        self.config = SimpleNamespace(long_term_memory_migration_enabled=False)
        self.snapshots: dict[str, dict[str, object]] = {}
        self.load_calls: list[dict[str, object]] = []

    def load_snapshot(self, *, snapshot_kind: str, local_path=None, prefer_cached_document_id: bool = False):
        self.load_calls.append(
            {
                "snapshot_kind": snapshot_kind,
                "local_path": local_path,
                "prefer_cached_document_id": prefer_cached_document_id,
            }
        )
        del local_path
        payload = self.snapshots.get(snapshot_kind)
        return dict(payload) if isinstance(payload, dict) else None

    def save_snapshot(self, *, snapshot_kind: str, payload):
        self.snapshots[snapshot_kind] = dict(payload)


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
                phone="01761234",
                role="Physiotherapist",
            )
            second = store.remember_contact(
                given_name="Corinna",
                family_name="Schmidt",
                phone="0309988",
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
        self.assertEqual(resolved.match.phones, ("01761234",))

    def test_contact_save_asks_before_merging_new_number_into_existing_named_person(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
            )
            store = TwinrPersonalGraphStore.from_config(config)
            store.remember_contact(given_name="Corinna", family_name="Maier", phone="01761234", role="Physiotherapist")

            result = store.remember_contact(given_name="Corinna", phone="040998877")

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

        self.assertFalse(store.path.exists())
        self.assertTrue(document.edges)
        self.assertIn("graph", remote_state.snapshots)

    def test_ensure_remote_snapshot_seeds_empty_graph_document(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )

            created = store.ensure_remote_snapshot()

        self.assertTrue(created)
        self.assertIn("graph", remote_state.snapshots)
        self.assertEqual(remote_state.snapshots["graph"]["metadata"]["kind"], "personal_graph")

    def test_remote_graph_reads_prefer_cached_document_id_hint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
                remote_state=remote_state,
            )

            store.ensure_remote_snapshot()

        self.assertTrue(remote_state.load_calls)
        self.assertTrue(remote_state.load_calls[0]["prefer_cached_document_id"])


if __name__ == "__main__":
    unittest.main()
