from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.memory.chonkydb import TwinrPersonalGraphStore
from twinr.memory.longterm.models import LongTermGraphEdgeCandidateV1


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
                phone="5551234",
                role="Physiotherapist",
            )
            second = store.remember_contact(
                given_name="Corinna",
                family_name="Schmidt",
                phone="5559988",
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
        self.assertEqual(resolved.match.phones, ("5551234",))

    def test_contact_save_asks_before_merging_new_number_into_existing_named_person(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
            )
            store = TwinrPersonalGraphStore.from_config(config)
            store.remember_contact(given_name="Corinna", family_name="Maier", phone="5551234", role="Physiotherapist")

            result = store.remember_contact(given_name="Corinna", phone="5557777")

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


if __name__ == "__main__":
    unittest.main()
