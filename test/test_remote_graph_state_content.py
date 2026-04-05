from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.chonkydb._remote_graph_state import _node_content_for_document
from twinr.memory.chonkydb.schema import TwinrGraphDocumentV1, TwinrGraphEdgeV1, TwinrGraphNodeV1


class RemoteGraphStateContentTests(unittest.TestCase):
    def test_node_content_includes_adjacent_contact_methods(self) -> None:
        document = TwinrGraphDocumentV1(
            subject_node_id="user:main",
            graph_id="graph:user_main",
            created_at="2026-04-05T00:00:00Z",
            updated_at="2026-04-05T00:00:00Z",
            nodes=(
                TwinrGraphNodeV1(node_id="user:main", node_type="user", label="Erika"),
                TwinrGraphNodeV1(node_id="person:anna_becker", node_type="person", label="Anna Becker"),
                TwinrGraphNodeV1(
                    node_id="email:anna_becker_example_com",
                    node_type="email",
                    label="anna.becker@example.com",
                    attributes={"canonical": "anna.becker@example.com"},
                ),
            ),
            edges=(
                TwinrGraphEdgeV1(
                    source_node_id="person:anna_becker",
                    edge_type="general_has_contact_method",
                    target_node_id="email:anna_becker_example_com",
                    attributes={"kind": "email"},
                ),
            ),
            metadata={"kind": "personal_graph"},
        )

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


if __name__ == "__main__":
    unittest.main()
