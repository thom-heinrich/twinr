from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sys
import unittest
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.memory.chonkydb import (
    ChonkyDBBulkRecordRequest,
    ChonkyDBClient,
    ChonkyDBConnectionConfig,
    ChonkyDBError,
    ChonkyDBRecordItem,
    ChonkyDBRecordRequest,
    ChonkyDBRetrieveRequest,
    chonkydb_data_path,
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


class ChonkyDBClientTests(unittest.TestCase):
    def test_instance_and_auth_use_x_api_key_header(self) -> None:
        opener = FakeOpener()
        opener.queue_json(
            {
                "success": True,
                "service": "ccodex_memory",
                "ready": True,
                "auth_enabled": True,
            }
        )
        opener.queue_json(
            {
                "success": True,
                "auth_enabled": True,
                "scheme": "api_key",
                "header_name": "x-api-key",
                "allow_bearer": True,
                "exempt_paths": ["/v1/health", "/v1/ready"],
                "api_key_configured": True,
            }
        )
        client = ChonkyDBClient(
            ChonkyDBConnectionConfig(
                base_url="https://memory.test/",
                api_key="secret-key",
            ),
            opener=opener,
        )

        instance = client.instance()
        auth = client.auth_info()

        self.assertTrue(instance.ready)
        self.assertEqual(instance.service, "ccodex_memory")
        self.assertEqual(auth.scheme, "api_key")
        self.assertEqual(opener.calls[0]["headers"]["X-api-key"], "secret-key")
        self.assertEqual(opener.calls[1]["headers"]["X-api-key"], "secret-key")

    def test_bearer_auth_header_is_supported(self) -> None:
        opener = FakeOpener()
        opener.queue_json({"success": True, "service": "ccodex_memory", "ready": True, "auth_enabled": True})
        client = ChonkyDBClient(
            ChonkyDBConnectionConfig(
                base_url="https://memory.test",
                api_key="secret-key",
                allow_bearer_auth=True,
            ),
            opener=opener,
        )

        client.instance()

        self.assertEqual(opener.calls[0]["headers"]["Authorization"], "Bearer secret-key")

    def test_retrieve_parses_hits_and_query(self) -> None:
        opener = FakeOpener()
        opener.queue_json(
            {
                "success": True,
                "mode": "advanced",
                "results": [
                    {
                        "payload_id": "payload-1",
                        "doc_id_int": 7,
                        "score": 0.99,
                        "relevance_score": 0.88,
                        "source_index": "vector",
                        "candidate_origin": "vector",
                        "metadata": {"room": "kitchen"},
                    }
                ],
                "indexes_used": ["ccodex_memory_fulltext", "ccodex_memory_vector_768"],
            }
        )
        client = ChonkyDBClient(
            ChonkyDBConnectionConfig(base_url="https://memory.test"),
            opener=opener,
        )

        response = client.retrieve(
            ChonkyDBRetrieveRequest(
                query_text="medication reminder",
                result_limit=2,
                allowed_indexes=("ccodex_memory_fulltext",),
            )
        )

        self.assertTrue(response.success)
        self.assertEqual(response.mode, "advanced")
        self.assertEqual(response.results[0].payload_id, "payload-1")
        self.assertEqual(response.results[0].metadata, {"room": "kitchen"})
        self.assertEqual(response.indexes_used, ("ccodex_memory_fulltext", "ccodex_memory_vector_768"))
        request_body = json.loads(opener.calls[0]["body"])
        self.assertEqual(request_body["query_text"], "medication reminder")
        self.assertEqual(request_body["allowed_indexes"], ["ccodex_memory_fulltext"])

    def test_store_request_and_bulk_request_encode_cleanly(self) -> None:
        opener = FakeOpener()
        opener.queue_json({"success": True, "stored": 1})
        opener.queue_json({"success": True, "stored": 2})
        client = ChonkyDBClient(
            ChonkyDBConnectionConfig(base_url="https://memory.test"),
            opener=opener,
        )

        client.store_record(
            ChonkyDBRecordRequest(
                payload={"note": "Bring medication"},
                metadata={"speaker": "user"},
                tags=("health", "reminder"),
                client_request_id="req-1",
            )
        )
        client.store_records_bulk(
            ChonkyDBBulkRecordRequest(
                items=(
                    ChonkyDBRecordItem(payload={"note": "A"}, metadata={"idx": 1}),
                    ChonkyDBRecordItem(payload={"note": "B"}, metadata={"idx": 2}),
                ),
                client_request_id="bulk-1",
            )
        )

        first_body = json.loads(opener.calls[0]["body"])
        second_body = json.loads(opener.calls[1]["body"])
        self.assertEqual(opener.calls[0]["full_url"], "https://memory.test/v1/external/records/bulk")
        self.assertEqual(first_body["client_request_id"], "req-1")
        self.assertEqual(first_body["items"][0]["payload"], {"note": "Bring medication"})
        self.assertEqual(second_body["client_request_id"], "bulk-1")
        self.assertEqual(len(second_body["items"]), 2)

    def test_list_records_and_full_document_build_query_strings(self) -> None:
        opener = FakeOpener()
        opener.queue_json(
            {
                "success": True,
                "offset": 0,
                "limit": 2,
                "total_count": 9,
                "returned_count": 2,
                "payloads": [
                    {"payload_id": "payload-1", "chonky_id": "1", "metadata": {"probe": True}},
                    {"payload_id": "payload-2", "chonky_id": "2", "metadata": {}},
                ],
            }
        )
        opener.queue_json({"success": True, "document_id": "doc-7"})
        client = ChonkyDBClient(
            ChonkyDBConnectionConfig(base_url="https://memory.test"),
            opener=opener,
        )

        records = client.list_records(limit=2, include_metadata=False)
        document = client.fetch_full_document(document_id="doc-7", include_content=False)

        self.assertEqual(records.returned_count, 2)
        self.assertEqual(records.payloads[0].payload_id, "payload-1")
        self.assertIn("limit=2", opener.calls[0]["full_url"])
        self.assertIn("include_metadata=false", opener.calls[0]["full_url"])
        self.assertIn("document_id=doc-7", opener.calls[1]["full_url"])
        self.assertIn("include_content=false", opener.calls[1]["full_url"])
        self.assertEqual(document["document_id"], "doc-7")

    def test_http_errors_raise_chonkydb_error_with_json_body(self) -> None:
        opener = FakeOpener()
        opener.queue_http_error(
            400,
            {
                "type": "validation_error",
                "detail": "bad request",
            },
        )
        client = ChonkyDBClient(
            ChonkyDBConnectionConfig(base_url="https://memory.test"),
            opener=opener,
        )

        with self.assertRaises(ChonkyDBError) as exc_info:
            client.retrieve({"query_text": "hello"})

        error = exc_info.exception
        self.assertEqual(error.status_code, 400)
        self.assertEqual(error.response_json, {"type": "validation_error", "detail": "bad request"})

    def test_fetch_full_document_requires_identifier(self) -> None:
        client = ChonkyDBClient(ChonkyDBConnectionConfig(base_url="https://memory.test"), opener=FakeOpener())

        with self.assertRaises(ValueError):
            client.fetch_full_document()

    def test_from_twinr_config_and_data_path(self) -> None:
        config = TwinrConfig(
            project_root="/tmp/project",
            long_term_memory_enabled=True,
            long_term_memory_backend="chonkydb",
            long_term_memory_path="state/chonkydb",
            chonkydb_base_url="https://memory.test",
            chonkydb_api_key="secret-key",
            chonkydb_api_key_header="x-api-key",
            chonkydb_allow_bearer_auth=False,
            chonkydb_timeout_s=12.5,
        )

        client = ChonkyDBClient.from_twinr_config(config, opener=FakeOpener())

        self.assertEqual(client.config.base_url, "https://memory.test")
        self.assertEqual(client.config.timeout_s, 12.5)
        self.assertEqual(chonkydb_data_path(config), Path("/tmp/project/state/chonkydb"))
