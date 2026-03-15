from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb.models import (
    ChonkyDBAuthInfo,
    ChonkyDBBulkRecordRequest,
    ChonkyDBConnectionConfig,
    ChonkyDBGraphAddEdgeRequest,
    ChonkyDBGraphAddEdgeSmartRequest,
    ChonkyDBGraphNeighborsRequest,
    ChonkyDBGraphPathRequest,
    ChonkyDBGraphPatternsRequest,
    ChonkyDBInstanceInfo,
    ChonkyDBRecordListResponse,
    ChonkyDBRecordRequest,
    ChonkyDBRetrieveRequest,
    ChonkyDBRetrieveResponse,
    JsonDict,
)


class _ResponseLike(Protocol):
    def __enter__(self) -> "_ResponseLike": ...
    def __exit__(self, exc_type, exc, tb) -> object: ...
    def read(self) -> bytes: ...


UrlopenLike = Callable[[Request, float], _ResponseLike]


def _default_urlopen(request: Request, timeout_s: float) -> _ResponseLike:
    return urlopen(request, timeout=timeout_s)


@dataclass(frozen=True, slots=True)
class ChonkyDBError(RuntimeError):
    message: str
    status_code: int | None = None
    response_text: str | None = None
    response_json: JsonDict | None = None

    def __str__(self) -> str:
        if self.status_code is None:
            return self.message
        return f"{self.message} (status={self.status_code})"


class ChonkyDBClient:
    def __init__(
        self,
        config: ChonkyDBConnectionConfig,
        *,
        opener: UrlopenLike | None = None,
    ) -> None:
        base_url = config.base_url.strip().rstrip("/")
        if not base_url:
            raise ValueError("ChonkyDB base URL is required.")
        self.config = ChonkyDBConnectionConfig(
            base_url=base_url,
            api_key=(config.api_key or "").strip() or None,
            api_key_header=(config.api_key_header or "x-api-key").strip() or "x-api-key",
            allow_bearer_auth=config.allow_bearer_auth,
            timeout_s=max(0.1, float(config.timeout_s)),
        )
        self._opener = opener or _default_urlopen

    @classmethod
    def from_twinr_config(cls, config: TwinrConfig, *, opener: UrlopenLike | None = None) -> "ChonkyDBClient":
        connection = ChonkyDBConnectionConfig(
            base_url=(config.chonkydb_base_url or "").strip(),
            api_key=config.chonkydb_api_key,
            api_key_header=config.chonkydb_api_key_header,
            allow_bearer_auth=config.chonkydb_allow_bearer_auth,
            timeout_s=config.chonkydb_timeout_s,
        )
        return cls(connection, opener=opener)

    def instance(self) -> ChonkyDBInstanceInfo:
        return ChonkyDBInstanceInfo.from_payload(self._request_json("GET", "/v1/external/instance"))

    def auth_info(self) -> ChonkyDBAuthInfo:
        return ChonkyDBAuthInfo.from_payload(self._request_json("GET", "/v1/external/admin/auth"))

    def list_records(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
        include_metadata: bool = True,
    ) -> ChonkyDBRecordListResponse:
        payload = self._request_json(
            "GET",
            "/v1/external/records",
            query={
                "offset": offset,
                "limit": limit,
                "include_metadata": str(include_metadata).lower(),
            },
        )
        return ChonkyDBRecordListResponse.from_payload(payload)

    def get_record(
        self,
        record_id: str,
        *,
        by: str = "payload",
        include_chunks: bool = False,
        include_relationships: bool = False,
        include_document: bool = False,
        include_content: bool = True,
        max_content_chars: int = 4000,
    ) -> JsonDict:
        if not record_id.strip():
            raise ValueError("record_id is required.")
        return self._request_json(
            "GET",
            f"/v1/external/records/{record_id}",
            query={
                "by": by,
                "include_chunks": str(include_chunks).lower(),
                "include_relationships": str(include_relationships).lower(),
                "include_document": str(include_document).lower(),
                "include_content": str(include_content).lower(),
                "max_content_chars": max_content_chars,
            },
        )

    def delete_record(self, record_id: str, *, by: str = "payload") -> JsonDict:
        if not record_id.strip():
            raise ValueError("record_id is required.")
        return self._request_json(
            "DELETE",
            f"/v1/external/records/{record_id}",
            query={"by": by},
        )

    def fetch_full_document(
        self,
        *,
        document_id: str | None = None,
        origin_uri: str | None = None,
        include_content: bool = True,
        max_content_chars: int = 4000,
    ) -> JsonDict:
        if not (document_id or origin_uri):
            raise ValueError("Either document_id or origin_uri is required.")
        return self._request_json(
            "GET",
            "/v1/external/documents/full",
            query={
                "document_id": document_id,
                "origin_uri": origin_uri,
                "include_content": str(include_content).lower(),
                "max_content_chars": max_content_chars,
            },
        )

    def store_record(self, request: ChonkyDBRecordRequest | Mapping[str, object]) -> JsonDict:
        body = request.to_payload() if isinstance(request, ChonkyDBRecordRequest) else dict(request)
        operation = str(body.pop("operation", "store_payload") or "store_payload").strip() or "store_payload"
        execution_mode = str(body.pop("execution_mode", "sync") or "sync").strip() or "sync"
        client_request_id = body.pop("client_request_id", None)
        timeout_seconds = body.pop("timeout_seconds", None)
        bulk_body = {
            "operation": operation,
            "execution_mode": execution_mode,
            "items": [dict(body)],
        }
        if client_request_id is not None:
            bulk_body["client_request_id"] = client_request_id
        if timeout_seconds is not None:
            bulk_body["timeout_seconds"] = timeout_seconds
        return self._request_json("POST", "/v1/external/records/bulk", body=bulk_body)

    def store_records_bulk(self, request: ChonkyDBBulkRecordRequest | Mapping[str, object]) -> JsonDict:
        body = request.to_payload() if isinstance(request, ChonkyDBBulkRecordRequest) else dict(request)
        return self._request_json("POST", "/v1/external/records/bulk", body=body)

    def retrieve(self, request: ChonkyDBRetrieveRequest | Mapping[str, object]) -> ChonkyDBRetrieveResponse:
        body = request.to_payload() if isinstance(request, ChonkyDBRetrieveRequest) else dict(request)
        payload = self._request_json("POST", "/v1/external/retrieve", body=body)
        return ChonkyDBRetrieveResponse.from_payload(payload)

    def add_graph_edge(self, request: ChonkyDBGraphAddEdgeRequest | Mapping[str, object]) -> JsonDict:
        body = request.to_payload() if isinstance(request, ChonkyDBGraphAddEdgeRequest) else dict(request)
        return self._request_json("POST", "/v1/external/graph/edges", body=body)

    def add_graph_edge_smart(self, request: ChonkyDBGraphAddEdgeSmartRequest | Mapping[str, object]) -> JsonDict:
        body = request.to_payload() if isinstance(request, ChonkyDBGraphAddEdgeSmartRequest) else dict(request)
        return self._request_json("POST", "/v1/external/graph/edges/smart", body=body)

    def graph_neighbors(self, request: ChonkyDBGraphNeighborsRequest | Mapping[str, object]) -> JsonDict:
        body = request.to_payload() if isinstance(request, ChonkyDBGraphNeighborsRequest) else dict(request)
        return self._request_json("POST", "/v1/external/graph/neighbors", body=body)

    def graph_path(self, request: ChonkyDBGraphPathRequest | Mapping[str, object]) -> JsonDict:
        body = request.to_payload() if isinstance(request, ChonkyDBGraphPathRequest) else dict(request)
        return self._request_json("POST", "/v1/external/graph/path", body=body)

    def graph_patterns(self, request: ChonkyDBGraphPatternsRequest | Mapping[str, object]) -> JsonDict:
        body = request.to_payload() if isinstance(request, ChonkyDBGraphPatternsRequest) else dict(request)
        return self._request_json("POST", "/v1/external/graph/patterns", body=body)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, object] | None = None,
        body: Mapping[str, object] | None = None,
    ) -> JsonDict:
        url = f"{self.config.base_url}{path}"
        if query:
            filtered_query = [(key, value) for key, value in query.items() if value is not None]
            if filtered_query:
                url = f"{url}?{urlencode(filtered_query)}"

        headers = {"Accept": "application/json"}
        if self.config.api_key:
            if self.config.allow_bearer_auth:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            else:
                headers[self.config.api_key_header] = self.config.api_key

        data: bytes | None = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(dict(body), ensure_ascii=False).encode("utf-8")

        request = Request(url, method=method.upper(), headers=headers, data=data)
        try:
            with self._opener(request, self.config.timeout_s) as response:
                raw = response.read()
        except HTTPError as exc:
            response_text = exc.read().decode("utf-8", errors="ignore")
            response_json = _parse_json_text(response_text)
            raise ChonkyDBError(
                f"ChonkyDB request failed for {method.upper()} {path}",
                status_code=exc.code,
                response_text=response_text,
                response_json=response_json,
            ) from exc
        except URLError as exc:
            raise ChonkyDBError(f"ChonkyDB request failed for {method.upper()} {path}: {exc.reason}") from exc
        except OSError as exc:
            raise ChonkyDBError(f"ChonkyDB request failed for {method.upper()} {path}: {exc}") from exc

        if not raw:
            return {}

        text = raw.decode("utf-8", errors="ignore")
        payload = _parse_json_text(text)
        if payload is None:
            raise ChonkyDBError(
                f"ChonkyDB request returned non-JSON content for {method.upper()} {path}",
                response_text=text,
            )
        return payload


def chonkydb_data_path(config: TwinrConfig) -> Path:
    configured = Path(config.long_term_memory_path)
    if configured.is_absolute():
        return configured
    return Path(config.project_root).resolve() / configured


def _parse_json_text(text: str) -> JsonDict | None:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return dict(payload) if isinstance(payload, dict) else {"value": payload}
