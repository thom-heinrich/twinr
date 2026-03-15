from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Callable, Mapping, Protocol, TypeVar
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener, urlopen

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
    def read(self, amt: int = -1) -> bytes: ...

UrlopenLike = Callable[[Request, float], _ResponseLike]

_ALLOWED_BASE_URL_SCHEMES = frozenset({"http", "https"})
_DEFAULT_TIMEOUT_S = 10.0
_MIN_TIMEOUT_S = 0.1
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_READ_CHUNK_SIZE = 64 * 1024
_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_MODEL_PARSE_EXCEPTIONS = (AssertionError, KeyError, TypeError, ValueError)
_T = TypeVar("_T")

class _NoRedirectHandler(HTTPRedirectHandler):
    # AUDIT-FIX(#1): Refuse server-driven redirects so credentials are not replayed to an unexpected target.
    def redirect_request(self, req, fp, code, msg, hdrs, newurl):  # type: ignore[override]
        raise HTTPError(req.full_url, code, f"Unexpected redirect to {newurl}", hdrs, fp)

_DEFAULT_OPENER = build_opener(ProxyHandler({}), _NoRedirectHandler())

def _default_urlopen(request: Request, timeout_s: float) -> _ResponseLike:
    # AUDIT-FIX(#1): Use a locked-down opener instead of the global urllib opener to avoid proxy autodiscovery and redirects.
    return _DEFAULT_OPENER.open(request, timeout=timeout_s)

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
        # AUDIT-FIX(#1): Canonicalize and validate the base URL so non-HTTP schemes, embedded credentials, and query/fragment abuse are rejected.
        base_url = _normalize_base_url(config.base_url)
        # AUDIT-FIX(#3): Normalize invalid timeout values to a sane finite default instead of crashing during client construction.
        timeout_s = _normalize_timeout_s(config.timeout_s)
        self.config = ChonkyDBConnectionConfig(
            base_url=base_url,
            api_key=_normalize_config_string(config.api_key),
            api_key_header=_normalize_header_name(config.api_key_header or "x-api-key"),
            allow_bearer_auth=bool(config.allow_bearer_auth),
            timeout_s=timeout_s,
        )
        self._opener = opener or _default_urlopen

    @classmethod
    def from_twinr_config(cls, config: TwinrConfig, *, opener: UrlopenLike | None = None) -> "ChonkyDBClient":
        connection = ChonkyDBConnectionConfig(
            base_url=_normalize_config_string(config.chonkydb_base_url) or "",
            api_key=config.chonkydb_api_key,
            api_key_header=config.chonkydb_api_key_header,
            allow_bearer_auth=config.chonkydb_allow_bearer_auth,
            timeout_s=config.chonkydb_timeout_s,
        )
        return cls(connection, opener=opener)

    def instance(self) -> ChonkyDBInstanceInfo:
        payload = self._request_json("GET", "/v1/external/instance")
        # AUDIT-FIX(#4): Wrap schema/shape mismatches from from_payload() so callers always receive a ChonkyDBError.
        return self._parse_model("instance()", ChonkyDBInstanceInfo.from_payload, payload)

    def auth_info(self) -> ChonkyDBAuthInfo:
        payload = self._request_json("GET", "/v1/external/admin/auth")
        # AUDIT-FIX(#4): Wrap schema/shape mismatches from from_payload() so callers always receive a ChonkyDBError.
        return self._parse_model("auth_info()", ChonkyDBAuthInfo.from_payload, payload)

    def list_records(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
        include_metadata: bool = True,
    ) -> ChonkyDBRecordListResponse:
        # AUDIT-FIX(#7): Reject invalid pagination values locally instead of sending malformed requests upstream.
        offset_value = _coerce_int("offset", offset, minimum=0)
        # AUDIT-FIX(#7): Reject invalid pagination values locally instead of sending malformed requests upstream.
        limit_value = _coerce_int("limit", limit, minimum=1)
        payload = self._request_json(
            "GET",
            "/v1/external/records",
            query={
                "offset": offset_value,
                "limit": limit_value,
                "include_metadata": str(include_metadata).lower(),
            },
        )
        # AUDIT-FIX(#4): Wrap schema/shape mismatches from from_payload() so callers always receive a ChonkyDBError.
        return self._parse_model("list_records()", ChonkyDBRecordListResponse.from_payload, payload)

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
        # AUDIT-FIX(#5): Percent-encode record IDs before interpolating them into the URL path.
        # AUDIT-FIX(#7): Reject blank/non-string selector values and invalid max_content_chars locally.
        safe_record_path = _record_path(record_id)
        by_value = _require_non_empty_str("by", by)
        max_content_chars_value = _coerce_int("max_content_chars", max_content_chars, minimum=0)
        return self._request_json(
            "GET",
            safe_record_path,
            query={
                "by": by_value,
                "include_chunks": str(include_chunks).lower(),
                "include_relationships": str(include_relationships).lower(),
                "include_document": str(include_document).lower(),
                "include_content": str(include_content).lower(),
                "max_content_chars": max_content_chars_value,
            },
        )

    def delete_record(self, record_id: str, *, by: str = "payload") -> JsonDict:
        # AUDIT-FIX(#5): Percent-encode record IDs before interpolating them into the URL path.
        # AUDIT-FIX(#7): Reject blank/non-string selector values locally.
        safe_record_path = _record_path(record_id)
        by_value = _require_non_empty_str("by", by)
        return self._request_json(
            "DELETE",
            safe_record_path,
            query={"by": by_value},
        )

    def fetch_full_document(
        self,
        *,
        document_id: str | None = None,
        origin_uri: str | None = None,
        include_content: bool = True,
        max_content_chars: int = 4000,
    ) -> JsonDict:
        # AUDIT-FIX(#7): Treat blank identifiers as absent and reject malformed inputs before issuing a request.
        document_id_value = _normalize_optional_request_string("document_id", document_id)
        # AUDIT-FIX(#7): Treat blank identifiers as absent and reject malformed inputs before issuing a request.
        origin_uri_value = _normalize_optional_request_string("origin_uri", origin_uri)
        if document_id_value is None and origin_uri_value is None:
            raise ValueError("Either document_id or origin_uri is required.")
        # AUDIT-FIX(#7): Reject invalid max_content_chars locally instead of sending malformed requests upstream.
        max_content_chars_value = _coerce_int("max_content_chars", max_content_chars, minimum=0)
        return self._request_json(
            "GET",
            "/v1/external/documents/full",
            query={
                "document_id": document_id_value,
                "origin_uri": origin_uri_value,
                "include_content": str(include_content).lower(),
                "max_content_chars": max_content_chars_value,
            },
        )

    def store_record(self, request: ChonkyDBRecordRequest | Mapping[str, object]) -> JsonDict:
        body = request.to_payload() if isinstance(request, ChonkyDBRecordRequest) else dict(request)
        operation = str(body.pop("operation", "store_payload") or "store_payload").strip() or "store_payload"
        execution_mode = str(body.pop("execution_mode", "sync") or "sync").strip() or "sync"
        client_request_id = body.pop("client_request_id", None)
        timeout_seconds = body.pop("timeout_seconds", None)
        if not body:
            # AUDIT-FIX(#7): Refuse empty single-record writes locally; an empty item is almost certainly a caller bug.
            raise ValueError("store_record request must contain at least one item field.")
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
        # AUDIT-FIX(#4): Wrap schema/shape mismatches from from_payload() so callers always receive a ChonkyDBError.
        return self._parse_model("retrieve()", ChonkyDBRetrieveResponse.from_payload, payload)

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
        # AUDIT-FIX(#7): Validate method/path upfront so malformed internal calls fail deterministically.
        normalized_method = _require_non_empty_str("method", method).upper()
        # AUDIT-FIX(#7): Validate method/path upfront so malformed internal calls fail deterministically.
        normalized_path = _normalize_api_path(path)
        url = f"{self.config.base_url}{normalized_path}"
        if query:
            filtered_query = [(key, value) for key, value in query.items() if value is not None]
            if filtered_query:
                # AUDIT-FIX(#8): Encode repeated query keys correctly instead of collapsing sequences into a string representation.
                url = f"{url}?{urlencode(filtered_query, doseq=True)}"

        headers = {"Accept": "application/json"}
        if self.config.api_key:
            if self.config.allow_bearer_auth:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            else:
                headers[self.config.api_key_header] = self.config.api_key

        data: bytes | None = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            # AUDIT-FIX(#9): Raise a deterministic local validation error when the request body is not JSON-serializable.
            data = _encode_json_body(body)

        request = Request(url, method=normalized_method, headers=headers, data=data)
        try:
            with self._opener(request, self.config.timeout_s) as response:
                # AUDIT-FIX(#6): Bound response size to protect the RPi process from memory exhaustion.
                raw = _read_response_bytes(response)
        except HTTPError as exc:
            # AUDIT-FIX(#6): Bound error-body reads as well, otherwise large error pages can still exhaust memory.
            response_text, response_json = _read_http_error_details(exc)
            raise ChonkyDBError(
                f"ChonkyDB request failed for {normalized_method} {normalized_path}",
                status_code=exc.code,
                response_text=response_text,
                response_json=response_json,
            ) from exc
        except URLError as exc:
            raise ChonkyDBError(
                f"ChonkyDB request failed for {normalized_method} {normalized_path}: {exc.reason}"
            ) from exc
        except OSError as exc:
            raise ChonkyDBError(f"ChonkyDB request failed for {normalized_method} {normalized_path}: {exc}") from exc

        if not raw:
            return {}

        # AUDIT-FIX(#8): Preserve replacement characters in diagnostics instead of silently dropping undecodable bytes.
        text = _decode_response_bytes(raw)
        payload = _parse_json_text(text)
        if payload is None:
            raise ChonkyDBError(
                f"ChonkyDB request returned non-JSON content for {normalized_method} {normalized_path}",
                response_text=text,
            )
        return payload

    def _parse_model(self, operation: str, parser: Callable[[JsonDict], _T], payload: JsonDict) -> _T:
        try:
            return parser(payload)
        except _MODEL_PARSE_EXCEPTIONS as exc:
            raise ChonkyDBError(
                f"ChonkyDB returned an invalid payload for {operation}",
                response_json=payload,
            ) from exc

def chonkydb_data_path(config: TwinrConfig) -> Path:
    # AUDIT-FIX(#2): Validate configured filesystem paths before normalization to avoid TypeError crashes on bad config.
    configured = _coerce_path_config("long_term_memory_path", config.long_term_memory_path)
    # AUDIT-FIX(#2): Resolve the configured path so symlinks and ".." are normalized before containment checks.
    if configured.is_absolute():
        return configured.resolve(strict=False)

    # AUDIT-FIX(#2): Enforce that relative storage paths stay inside project_root after normalization.
    project_root = _coerce_path_config("project_root", config.project_root).resolve(strict=False)
    candidate = (project_root / configured).resolve(strict=False)
    try:
        candidate.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("long_term_memory_path must stay within project_root when configured as a relative path.") from exc
    return candidate

def _parse_json_text(text: str) -> JsonDict | None:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return dict(payload) if isinstance(payload, dict) else {"value": payload}

def _normalize_base_url(raw_base_url: object) -> str:
    if not isinstance(raw_base_url, str):
        raise ValueError("ChonkyDB base URL must be a string.")
    base_url = raw_base_url.strip().rstrip("/")
    if not base_url:
        raise ValueError("ChonkyDB base URL is required.")

    parsed = urlsplit(base_url)
    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_BASE_URL_SCHEMES:
        raise ValueError("ChonkyDB base URL must use http or https.")
    if not parsed.netloc:
        raise ValueError("ChonkyDB base URL must include a host.")
    if parsed.username or parsed.password:
        raise ValueError("ChonkyDB base URL must not contain embedded credentials.")
    if parsed.query or parsed.fragment:
        raise ValueError("ChonkyDB base URL must not contain a query or fragment.")

    return urlunsplit((scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))

def _normalize_header_name(raw_header_name: object) -> str:
    header_name = _normalize_config_string(raw_header_name) or "x-api-key"
    if not _HEADER_NAME_RE.fullmatch(header_name):
        raise ValueError("ChonkyDB API key header contains invalid characters.")
    return header_name

def _normalize_config_string(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None

def _coerce_path_config(name: str, value: object) -> Path:
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{name} must be a non-empty path string.")
        path = Path(normalized)
    else:
        raise ValueError(f"{name} must be a path string.")
    return path.expanduser()

def _normalize_timeout_s(value: object) -> float:
    try:
        timeout_s = float(value)
    except (TypeError, ValueError, OverflowError):
        return _DEFAULT_TIMEOUT_S
    if not math.isfinite(timeout_s):
        return _DEFAULT_TIMEOUT_S
    return max(_MIN_TIMEOUT_S, timeout_s)

def _require_non_empty_str(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a non-empty string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty string.")
    return normalized

def _normalize_optional_request_string(name: str, value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string when provided.")
    normalized = value.strip()
    return normalized or None

def _coerce_int(name: str, value: object, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    try:
        coerced = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer.")
    if minimum is not None and coerced < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return coerced

def _normalize_api_path(path: object) -> str:
    normalized = _require_non_empty_str("path", path)
    parsed = urlsplit(normalized)
    if parsed.scheme or parsed.netloc:
        raise ValueError("path must be a relative API path.")
    if parsed.query or parsed.fragment:
        raise ValueError("path must not include query or fragment components.")
    if not parsed.path.startswith("/"):
        raise ValueError("path must start with '/'.")
    return parsed.path

def _record_path(record_id: object) -> str:
    return f"/v1/external/records/{quote(_require_non_empty_str('record_id', record_id), safe='')}"

def _encode_json_body(body: Mapping[str, object]) -> bytes:
    try:
        serialized = json.dumps(dict(body), ensure_ascii=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("ChonkyDB request body must be JSON-serializable.") from exc
    return serialized.encode("utf-8")

def _read_response_bytes(response: _ResponseLike) -> bytes:
    buffer = bytearray()
    try:
        chunk = response.read(_READ_CHUNK_SIZE)
    except TypeError:
        chunk = response.read()

    while chunk:
        buffer.extend(chunk)
        if len(buffer) > _MAX_RESPONSE_BYTES:
            raise ChonkyDBError(
                f"ChonkyDB response body exceeded {_MAX_RESPONSE_BYTES} bytes."
            )
        try:
            chunk = response.read(_READ_CHUNK_SIZE)
        except TypeError:
            break

    return bytes(buffer)

def _read_http_error_details(exc: HTTPError) -> tuple[str | None, JsonDict | None]:
    response_text: str | None = None
    response_json: JsonDict | None = None
    try:
        raw = _read_response_bytes(exc)
    except ChonkyDBError as read_exc:
        response_text = str(read_exc)
    except OSError:
        response_text = None
    else:
        response_text = _decode_response_bytes(raw)
        response_json = _parse_json_text(response_text)
    finally:
        close = getattr(exc, "close", None)
        if callable(close):
            close()

    return response_text, response_json

def _decode_response_bytes(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")