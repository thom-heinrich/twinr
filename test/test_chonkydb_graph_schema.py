from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sys
import unittest
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.chonkydb import (
    ChonkyDBClient,
    ChonkyDBConnectionConfig,
    ChonkyDBError,
    ChonkyDBGraphAddEdgeSmartRequest,
    ChonkyDBGraphNeighborsRequest,
    ChonkyDBGraphPathRequest,
    ChonkyDBGraphPatternsRequest,
    ChonkyDBGraphStoreManyEdge,
    ChonkyDBGraphStoreManyNode,
    ChonkyDBGraphStoreManyRequest,
    TWINR_GRAPH_ALLOWED_EDGE_TYPES,
    TWINR_GRAPH_SCHEMA_NAME,
    TWINR_GRAPH_SCHEMA_VERSION,
    TwinrGraphDocumentV1,
    TwinrGraphEdgeV1,
    TwinrGraphNodeV1,
    graph_edge_namespace,
    is_allowed_graph_edge_type,
)


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class FakeOpener:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.responses: list[object] = []

    def queue_json(self, payload: dict[str, object]) -> None:
        self.responses.append(FakeHTTPResponse(payload))

    def queue_http_error(self, status_code: int, payload: dict[str, object]) -> None:
        self.responses.append(
            HTTPError(
                url="https://memory.test/fail",
                code=status_code,
                msg="bad request",
                hdrs=None,
                fp=BytesIO(json.dumps(payload).encode("utf-8")),
            )
        )

    def __call__(self, request, timeout: float):
        self.calls.append(
            {
                "full_url": request.full_url,
                "method": request.get_method(),
                "headers": dict(request.header_items()),
                "body": request.data.decode("utf-8") if request.data else None,
                "timeout": timeout,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class TwinrGraphSchemaTests(unittest.TestCase):
    def test_graph_document_payload_is_versioned_and_structured(self) -> None:
        user = TwinrGraphNodeV1(node_id="user:main", node_type="user", label="Erika")
        brand = TwinrGraphNodeV1(
            node_id="brand:melitta",
            node_type="brand",
            label="Melitta",
            aliases=("Melitta Kaffee",),
        )
        edge = TwinrGraphEdgeV1(
            source_node_id="user:main",
            edge_type="user_prefers",
            target_node_id="brand:melitta",
            confidence=0.86,
            confirmed_by_user=True,
            origin="conversation",
            attributes={"for_product": "coffee"},
        )
        document = TwinrGraphDocumentV1(
            subject_node_id="user:main",
            graph_id="graph:user_main",
            created_at="2026-03-14T08:00:00Z",
            updated_at="2026-03-14T08:05:00Z",
            nodes=(user, brand),
            edges=(edge,),
            metadata={"profile": "main"},
        )

        payload = document.to_payload()

        self.assertEqual(payload["schema"], {"name": TWINR_GRAPH_SCHEMA_NAME, "version": TWINR_GRAPH_SCHEMA_VERSION})
        self.assertEqual(payload["subject_node_id"], "user:main")
        self.assertEqual(payload["nodes"][1]["aliases"], ["Melitta Kaffee"])
        self.assertEqual(payload["edges"][0]["type"], "user_prefers")
        self.assertEqual(payload["edges"][0]["attributes"], {"for_product": "coffee"})

    def test_graph_document_rejects_unknown_edge_type(self) -> None:
        with self.assertRaises(ValueError):
            TwinrGraphEdgeV1(
                source_node_id="user:main",
                edge_type="health_takes_medicine",
                target_node_id="medicine:abc",
            )

    def test_graph_document_rejects_missing_nodes_and_duplicates(self) -> None:
        user = TwinrGraphNodeV1(node_id="user:main", node_type="user", label="Erika")
        twin = TwinrGraphNodeV1(node_id="user:main", node_type="user", label="Duplicate")
        place = TwinrGraphNodeV1(node_id="place:store_z", node_type="place", label="Geschaeft Z")
        edge = TwinrGraphEdgeV1(
            source_node_id="user:main",
            edge_type="user_prefers",
            target_node_id="place:store_z",
        )

        with self.assertRaises(ValueError):
            TwinrGraphDocumentV1(subject_node_id="user:main", nodes=(user, twin), edges=())

        with self.assertRaises(ValueError):
            TwinrGraphDocumentV1(subject_node_id="user:main", nodes=(user,), edges=(edge,))

        document = TwinrGraphDocumentV1(subject_node_id="user:main", nodes=(user, place), edges=(edge,))
        self.assertEqual(document.node("place:store_z").label, "Geschaeft Z")

    def test_edge_catalog_helpers_are_stable(self) -> None:
        self.assertIn("social_related_to_user", TWINR_GRAPH_ALLOWED_EDGE_TYPES)
        self.assertEqual(graph_edge_namespace("spatial_near"), "spatial")
        self.assertTrue(is_allowed_graph_edge_type("general_has_contact_method"))
        self.assertFalse(is_allowed_graph_edge_type("caregiver_knows"))


class ChonkyDBGraphClientTests(unittest.TestCase):
    def test_graph_methods_encode_expected_paths_and_bodies(self) -> None:
        opener = FakeOpener()
        opener.queue_json({"success": True, "stored": True})
        opener.queue_json({"success": True, "edge_created": True})
        opener.queue_json({"success": True, "neighbors": [{"label": "Corinna Maier"}]})
        opener.queue_json({"success": True, "path": ["user:main", "person:corinna_maier"]})
        opener.queue_json({"success": True, "matches": [{"depth": 1}]})
        client = ChonkyDBClient(ChonkyDBConnectionConfig(base_url="https://memory.test"), opener=opener)

        client.graph_store_many(
            ChonkyDBGraphStoreManyRequest(
                index_name="twinr_graph_test",
                nodes=(
                    ChonkyDBGraphStoreManyNode(label="gen1:user:main"),
                    ChonkyDBGraphStoreManyNode(label="gen1:person:corinna_maier"),
                ),
                edges=(
                    ChonkyDBGraphStoreManyEdge(
                        source_label="gen1:user:main",
                        target_label="gen1:person:corinna_maier",
                        edge_type="social_related_to_user",
                    ),
                ),
                timeout_seconds=15,
            )
        )
        client.add_graph_edge_smart(
            ChonkyDBGraphAddEdgeSmartRequest(
                from_ref="user:main",
                to_ref="person:corinna_maier",
                edge_type="social_related_to_user",
            )
        )
        client.graph_neighbors(
            ChonkyDBGraphNeighborsRequest(
                label_or_id="person:corinna_maier",
                edge_types=("general_has_contact_method",),
                with_edges=True,
                limit=4,
            )
        )
        client.graph_path(
            ChonkyDBGraphPathRequest(
                source="user:main",
                target="brand:melitta",
                edge_types=("user_prefers",),
                return_ids=True,
            )
        )
        client.graph_patterns(
            ChonkyDBGraphPatternsRequest(
                patterns=(
                    {"start": "user:main", "edge_type": "user_prefers", "end": "brand:*"},
                ),
                include_content=False,
            )
        )

        self.assertTrue(opener.calls[0]["full_url"].endswith("/v1/graph/store_many"))
        self.assertTrue(opener.calls[1]["full_url"].endswith("/v1/external/graph/edges/smart"))
        self.assertTrue(opener.calls[2]["full_url"].endswith("/v1/external/graph/neighbors"))
        self.assertTrue(opener.calls[3]["full_url"].endswith("/v1/external/graph/path"))
        self.assertTrue(opener.calls[4]["full_url"].endswith("/v1/external/graph/patterns"))
        self.assertEqual(json.loads(opener.calls[0]["body"])["nodes"][0]["label"], "gen1:user:main")
        self.assertEqual(json.loads(opener.calls[1]["body"])["edge_type"], "social_related_to_user")
        self.assertEqual(json.loads(opener.calls[2]["body"])["edge_types"], ["general_has_contact_method"])
        self.assertEqual(json.loads(opener.calls[3]["body"])["target"], "brand:melitta")
        self.assertEqual(json.loads(opener.calls[4]["body"])["patterns"][0]["edge_type"], "user_prefers")

    def test_graph_http_errors_raise_chonkydb_error(self) -> None:
        opener = FakeOpener()
        opener.queue_http_error(409, {"detail": "edge conflict"})
        client = ChonkyDBClient(ChonkyDBConnectionConfig(base_url="https://memory.test"), opener=opener)

        with self.assertRaises(ChonkyDBError) as exc_info:
            client.add_graph_edge_smart(
                {
                    "from_ref": "user:main",
                    "to_ref": "person:corinna_maier",
                    "edge_type": "social_related_to_user",
                }
            )

        self.assertEqual(exc_info.exception.status_code, 409)
        self.assertEqual(exc_info.exception.response_json, {"detail": "edge conflict"})
