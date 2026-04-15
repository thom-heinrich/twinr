from __future__ import annotations

from pathlib import Path
import sys
import unittest
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.chonkydb._remote_graph_state import (
    _filter_edge_payloads_for_seed_nodes,
    _merge_graph_node_candidate_payloads,
    _node_content_for_document,
    _node_metadata,
    _promote_owner_person_candidate_payloads,
    _rerank_graph_node_payloads,
)
from twinr.memory.chonkydb.schema import TwinrGraphDocumentV1, TwinrGraphEdgeV1, TwinrGraphNodeV1
from twinr.memory.longterm.storage.remote_catalog import LongTermRemoteCatalogEntry, LongTermRemoteCatalogStore


class RemoteGraphStateContentTests(unittest.TestCase):
    def _anna_becker_document(self) -> TwinrGraphDocumentV1:
        return TwinrGraphDocumentV1(
            subject_node_id="user:main",
            graph_id="graph:user_main",
            created_at="2026-04-05T00:00:00Z",
            updated_at="2026-04-05T00:00:00Z",
            nodes=(
                TwinrGraphNodeV1(node_id="user:main", node_type="user", label="Erika"),
                TwinrGraphNodeV1(node_id="person:anna_becker", node_type="person", label="Anna Becker"),
                TwinrGraphNodeV1(node_id="person:chris_becker", node_type="person", label="Chris Becker"),
                TwinrGraphNodeV1(
                    node_id="email:anna_becker_example_com",
                    node_type="email",
                    label="anna.becker@example.com",
                    attributes={"canonical": "anna.becker@example.com"},
                ),
                TwinrGraphNodeV1(
                    node_id="phone:chris_becker",
                    node_type="phone",
                    label="+49 30 555 000",
                    attributes={"canonical": "+49 30 555 000"},
                ),
            ),
            edges=(
                TwinrGraphEdgeV1(
                    source_node_id="person:anna_becker",
                    edge_type="general_has_contact_method",
                    target_node_id="email:anna_becker_example_com",
                    attributes={"kind": "email"},
                ),
                TwinrGraphEdgeV1(
                    source_node_id="person:chris_becker",
                    edge_type="general_has_contact_method",
                    target_node_id="phone:chris_becker",
                    attributes={"kind": "phone"},
                ),
            ),
            metadata={"kind": "personal_graph"},
        )

    def test_node_content_includes_adjacent_contact_methods(self) -> None:
        document = self._anna_becker_document()

        person_content = _node_content_for_document(
            document,
            {"id": "person:anna_becker", "type": "person", "label": "Anna Becker"},
        )
        email_content = _node_content_for_document(
            document,
            {
                "id": "email:anna_becker_example_com",
                "type": "email",
                "label": "anna.becker@example.com",
                "attributes": {"canonical": "anna.becker@example.com"},
            },
        )

        self.assertIn("Anna Becker", person_content)
        self.assertIn("email", person_content)
        self.assertIn("anna.becker@example.com", person_content)
        self.assertIn("Anna Becker", email_content)
        self.assertIn("email address", email_content)

    def test_node_metadata_carries_bounded_search_text_for_projection_only_catalog_entries(self) -> None:
        document = self._anna_becker_document()

        metadata = _node_metadata(
            document,
            {"id": "person:anna_becker", "type": "person", "label": "Anna Becker"},
        )
        search_text = str(metadata["search_text"])

        self.assertIn("search_text", metadata)
        self.assertIn("Anna Becker", search_text)
        self.assertIn("email address", search_text)
        self.assertIn("anna.becker@example.com", search_text)
        self.assertLessEqual(len(search_text), 512)

    def test_projection_only_graph_node_catalog_search_prefers_email_person_over_same_surname_noise(self) -> None:
        document = self._anna_becker_document()
        catalog = LongTermRemoteCatalogStore(None)
        chris_entry = LongTermRemoteCatalogEntry(
            snapshot_kind="graph_nodes",
            item_id="person:chris_becker",
            document_id=None,
            uri="twinr://longterm/test/graph_nodes/person%3Achris_becker",
            metadata=_node_metadata(
                document,
                {"id": "person:chris_becker", "type": "person", "label": "Chris Becker"},
            ),
        )
        anna_entry = LongTermRemoteCatalogEntry(
            snapshot_kind="graph_nodes",
            item_id="person:anna_becker",
            document_id=None,
            uri="twinr://longterm/test/graph_nodes/person%3Aanna_becker",
            metadata=_node_metadata(
                document,
                {"id": "person:anna_becker", "type": "person", "label": "Anna Becker"},
            ),
        )

        self.assertIn("anna.becker@example.com", catalog._catalog_entry_search_text(anna_entry))
        selected = catalog._local_search_catalog_entries(
            snapshot_kind="graph_nodes",
            entries=(chris_entry, anna_entry),
            query_text="Which Becker email address should I use?",
            limit=1,
        )

        self.assertEqual(tuple(entry.item_id for entry in selected), ("person:anna_becker",))

    def test_graph_node_rerank_prefers_exact_person_label_over_same_surname_remote_rank(self) -> None:
        document = self._anna_becker_document()
        anna_entry = LongTermRemoteCatalogEntry(
            snapshot_kind="graph_nodes",
            item_id="person:anna_becker",
            document_id=None,
            uri="twinr://longterm/test/graph_nodes/person%3Aanna_becker",
            metadata=_node_metadata(
                document,
                {"id": "person:anna_becker", "type": "person", "label": "Anna Becker"},
            ),
        )
        chris_entry = LongTermRemoteCatalogEntry(
            snapshot_kind="graph_nodes",
            item_id="person:chris_becker",
            document_id=None,
            uri="twinr://longterm/test/graph_nodes/person%3Achris_becker",
            metadata=_node_metadata(
                document,
                {"id": "person:chris_becker", "type": "person", "label": "Chris Becker"},
            ),
        )
        merged_payloads, origins = _merge_graph_node_candidate_payloads(
            initial_payloads=(
                {"id": "person:chris_becker", "type": "person", "label": "Chris Becker"},
            ),
            local_entries=(anna_entry,),
        )

        reranked_payloads, debug = _rerank_graph_node_payloads(
            query_text="What is Anna Becker's email address?",
            candidate_payloads=merged_payloads,
            current_entries_by_item_id={
                "person:anna_becker": anna_entry,
                "person:chris_becker": chris_entry,
            },
            candidate_limit=1,
            origins_by_item_id=origins,
        )

        self.assertEqual(tuple(payload["id"] for payload in reranked_payloads), ("person:anna_becker",))
        self.assertTrue(debug[0]["selected_seed"])
        self.assertEqual(debug[0]["node_id"], "person:anna_becker")
        score_components = cast(dict[str, object], debug[0]["score_components"])
        debug_details = cast(dict[str, object], debug[0]["debug"])
        self.assertTrue(cast(bool, score_components["exact_label_phrase"]))
        self.assertIn("anna.becker@example.com", cast(str, debug_details["search_text_excerpt"]))

    def test_contact_method_seed_promotes_owner_person_node_before_rerank(self) -> None:
        document = self._anna_becker_document()
        anna_person_entry = LongTermRemoteCatalogEntry(
            snapshot_kind="graph_nodes",
            item_id="person:anna_becker",
            document_id=None,
            uri="twinr://longterm/test/graph_nodes/person%3Aanna_becker",
            metadata=_node_metadata(
                document,
                {"id": "person:anna_becker", "type": "person", "label": "Anna Becker"},
            ),
        )
        chris_entry = LongTermRemoteCatalogEntry(
            snapshot_kind="graph_nodes",
            item_id="person:chris_becker",
            document_id=None,
            uri="twinr://longterm/test/graph_nodes/person%3Achris_becker",
            metadata=_node_metadata(
                document,
                {"id": "person:chris_becker", "type": "person", "label": "Chris Becker"},
            ),
        )
        anna_email_entry = LongTermRemoteCatalogEntry(
            snapshot_kind="graph_nodes",
            item_id="email:anna_becker_example_com",
            document_id=None,
            uri="twinr://longterm/test/graph_nodes/email%3Aanna_becker_example_com",
            metadata=_node_metadata(
                document,
                {
                    "id": "email:anna_becker_example_com",
                    "type": "email",
                    "label": "anna.becker@example.com",
                    "attributes": {"canonical": "anna.becker@example.com"},
                },
            ),
        )
        merged_payloads, origins = _merge_graph_node_candidate_payloads(
            initial_payloads=(
                {"id": "person:chris_becker", "type": "person", "label": "Chris Becker"},
            ),
            local_entries=(anna_email_entry,),
        )

        promoted_payloads, promoted_origins, promotion_debug = _promote_owner_person_candidate_payloads(
            candidate_payloads=merged_payloads,
            current_entries_by_item_id={
                "person:anna_becker": anna_person_entry,
                "person:chris_becker": chris_entry,
                "email:anna_becker_example_com": anna_email_entry,
            },
            origins_by_item_id=origins,
        )
        reranked_payloads, debug = _rerank_graph_node_payloads(
            query_text="What is Anna Becker's email address?",
            candidate_payloads=promoted_payloads,
            current_entries_by_item_id={
                "person:anna_becker": anna_person_entry,
                "person:chris_becker": chris_entry,
                "email:anna_becker_example_com": anna_email_entry,
            },
            candidate_limit=1,
            origins_by_item_id=promoted_origins,
        )

        self.assertEqual(
            tuple(payload["id"] for payload in promoted_payloads),
            (
                "person:chris_becker",
                "email:anna_becker_example_com",
                "person:anna_becker",
            ),
        )
        self.assertEqual(
            promotion_debug,
            (
                {
                    "from_node_id": "email:anna_becker_example_com",
                    "from_node_type": "email",
                    "promoted_owner_node_id": "person:anna_becker",
                    "promoted_owner_label": "Anna Becker",
                    "reason": "contact_method_owner_seed",
                },
            ),
        )
        self.assertIn("owner_person_promotion", promoted_origins["person:anna_becker"])
        self.assertEqual(tuple(payload["id"] for payload in reranked_payloads), ("person:anna_becker",))
        self.assertEqual(debug[0]["node_id"], "person:anna_becker")
        self.assertTrue(debug[0]["selected_seed"])

    def test_graph_edge_filter_discards_same_surname_noise_when_seed_nodes_are_reranked(self) -> None:
        selected_edges, deferred_edges = _filter_edge_payloads_for_seed_nodes(
            edge_payloads=(
                {
                    "source": "person:chris_becker",
                    "type": "general_has_contact_method",
                    "target": "phone:chris_becker",
                },
                {
                    "source": "person:anna_becker",
                    "type": "general_has_contact_method",
                    "target": "email:anna_becker_example_com",
                },
            ),
            seed_node_ids=("person:anna_becker",),
            subject_node_id="user:main",
        )

        self.assertEqual(
            tuple(_edge["target"] for _edge in selected_edges),
            ("email:anna_becker_example_com",),
        )
        self.assertEqual(
            tuple(_edge["target"] for _edge in deferred_edges),
            ("phone:chris_becker",),
        )


if __name__ == "__main__":
    unittest.main()
