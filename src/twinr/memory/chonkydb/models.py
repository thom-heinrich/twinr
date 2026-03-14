from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

JsonDict = dict[str, object]


def _merge_extra(payload: JsonDict, extra: Mapping[str, object] | None) -> JsonDict:
    if extra:
        payload.update(dict(extra))
    return payload


def _drop_none(payload: JsonDict) -> JsonDict:
    return {key: value for key, value in payload.items() if value is not None}


@dataclass(frozen=True, slots=True)
class ChonkyDBConnectionConfig:
    base_url: str
    api_key: str | None = None
    api_key_header: str = "x-api-key"
    allow_bearer_auth: bool = False
    timeout_s: float = 20.0


@dataclass(frozen=True, slots=True)
class ChonkyDBInstanceInfo:
    success: bool
    service: str
    ready: bool
    auth_enabled: bool
    raw: JsonDict = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ChonkyDBInstanceInfo":
        raw = dict(payload)
        return cls(
            success=bool(raw.get("success", False)),
            service=str(raw.get("service", "")),
            ready=bool(raw.get("ready", False)),
            auth_enabled=bool(raw.get("auth_enabled", False)),
            raw=raw,
        )


@dataclass(frozen=True, slots=True)
class ChonkyDBAuthInfo:
    success: bool
    auth_enabled: bool
    scheme: str
    header_name: str
    allow_bearer: bool
    exempt_paths: tuple[str, ...]
    api_key_configured: bool
    raw: JsonDict = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ChonkyDBAuthInfo":
        raw = dict(payload)
        exempt_paths = tuple(str(item) for item in raw.get("exempt_paths", ()) if isinstance(item, str))
        return cls(
            success=bool(raw.get("success", False)),
            auth_enabled=bool(raw.get("auth_enabled", False)),
            scheme=str(raw.get("scheme", "")),
            header_name=str(raw.get("header_name", "x-api-key")),
            allow_bearer=bool(raw.get("allow_bearer", False)),
            exempt_paths=exempt_paths,
            api_key_configured=bool(raw.get("api_key_configured", False)),
            raw=raw,
        )


@dataclass(frozen=True, slots=True)
class ChonkyDBRecordItem:
    payload: Mapping[str, object] | None = None
    metadata: Mapping[str, object] | None = None
    content: str | None = None
    uri: str | None = None
    file_path: str | None = None
    file_type: str | None = None
    language: str | None = None
    tags: tuple[str, ...] | None = None
    target_indexes: tuple[str, ...] | None = None
    enable_chunking: bool = True
    enable_llm_analysis: bool | None = None
    llm_focus_areas: tuple[str, ...] | None = None
    include_insights_in_response: bool = False
    timeout_seconds: float | None = None
    targets: tuple[Mapping[str, object], ...] | None = None
    extra: Mapping[str, object] | None = None

    def to_payload(self) -> JsonDict:
        payload = _drop_none(
            {
                "payload": dict(self.payload) if self.payload is not None else None,
                "metadata": dict(self.metadata) if self.metadata is not None else None,
                "content": self.content,
                "uri": self.uri,
                "file_path": self.file_path,
                "file_type": self.file_type,
                "language": self.language,
                "tags": list(self.tags) if self.tags is not None else None,
                "target_indexes": list(self.target_indexes) if self.target_indexes is not None else None,
                "enable_chunking": self.enable_chunking,
                "enable_llm_analysis": self.enable_llm_analysis,
                "llm_focus_areas": list(self.llm_focus_areas) if self.llm_focus_areas is not None else None,
                "include_insights_in_response": self.include_insights_in_response,
                "timeout_seconds": self.timeout_seconds,
                "targets": [dict(item) for item in self.targets] if self.targets is not None else None,
            }
        )
        return _merge_extra(payload, self.extra)


@dataclass(frozen=True, slots=True)
class ChonkyDBRecordRequest(ChonkyDBRecordItem):
    operation: str = "store_payload"
    execution_mode: str = "sync"
    client_request_id: str | None = None

    def to_payload(self) -> JsonDict:
        payload = ChonkyDBRecordItem.to_payload(self)
        payload.update(
            {
                "operation": self.operation,
                "execution_mode": self.execution_mode,
            }
        )
        if self.client_request_id is not None:
            payload["client_request_id"] = self.client_request_id
        return payload


@dataclass(frozen=True, slots=True)
class ChonkyDBBulkRecordRequest:
    items: tuple[ChonkyDBRecordItem, ...]
    operation: str = "store_payload"
    execution_mode: str = "sync"
    timeout_seconds: float | None = None
    client_request_id: str | None = None
    finalize_vector_segments: bool = True
    extra: Mapping[str, object] | None = None

    def to_payload(self) -> JsonDict:
        payload = _drop_none(
            {
                "operation": self.operation,
                "execution_mode": self.execution_mode,
                "items": [item.to_payload() for item in self.items],
                "timeout_seconds": self.timeout_seconds,
                "client_request_id": self.client_request_id,
                "finalize_vector_segments": self.finalize_vector_segments,
            }
        )
        return _merge_extra(payload, self.extra)


@dataclass(frozen=True, slots=True)
class ChonkyDBRetrieveRequest:
    query_text: str | None = None
    mode: str = "advanced"
    result_limit: int = 10
    include_content: bool = False
    include_metadata: bool = True
    content_mode: str | None = None
    max_content_chars: int | None = None
    timeout_seconds: float | None = None
    filters: Mapping[str, object] | None = None
    temporal_start: str | None = None
    temporal_end: str | None = None
    client_request_id: str | None = None
    allowed_indexes: tuple[str, ...] | None = None
    extra: Mapping[str, object] | None = None

    def to_payload(self) -> JsonDict:
        payload = _drop_none(
            {
                "mode": self.mode,
                "query_text": self.query_text,
                "result_limit": self.result_limit,
                "include_content": self.include_content,
                "include_metadata": self.include_metadata,
                "content_mode": self.content_mode,
                "max_content_chars": self.max_content_chars,
                "timeout_seconds": self.timeout_seconds,
                "filters": dict(self.filters) if self.filters is not None else None,
                "temporal_start": self.temporal_start,
                "temporal_end": self.temporal_end,
                "client_request_id": self.client_request_id,
                "allowed_indexes": list(self.allowed_indexes) if self.allowed_indexes is not None else None,
            }
        )
        return _merge_extra(payload, self.extra)


@dataclass(frozen=True, slots=True)
class ChonkyDBGraphAddEdgeRequest:
    from_id: int
    to_id: int
    edge_type: str
    allow_self_loops: bool | None = None
    extra: Mapping[str, object] | None = None

    def to_payload(self) -> JsonDict:
        payload = _drop_none(
            {
                "from_id": self.from_id,
                "to_id": self.to_id,
                "edge_type": self.edge_type,
                "allow_self_loops": self.allow_self_loops,
            }
        )
        return _merge_extra(payload, self.extra)


@dataclass(frozen=True, slots=True)
class ChonkyDBGraphAddEdgeSmartRequest:
    from_ref: str
    to_ref: str
    edge_type: str
    extra: Mapping[str, object] | None = None

    def to_payload(self) -> JsonDict:
        payload = _drop_none(
            {
                "from_ref": self.from_ref,
                "to_ref": self.to_ref,
                "edge_type": self.edge_type,
            }
        )
        return _merge_extra(payload, self.extra)


@dataclass(frozen=True, slots=True)
class ChonkyDBGraphNeighborsRequest:
    index_name: str | None = None
    label_or_id: str | None = None
    label: str | None = None
    direction: str = "both"
    with_edges: bool = False
    return_ids: bool = False
    edge_types: tuple[str, ...] | None = None
    limit: int | None = None
    timeout_seconds: float | None = None
    extra: Mapping[str, object] | None = None

    def to_payload(self) -> JsonDict:
        payload = _drop_none(
            {
                "index_name": self.index_name,
                "label_or_id": self.label_or_id,
                "label": self.label,
                "direction": self.direction,
                "with_edges": self.with_edges,
                "return_ids": self.return_ids,
                "edge_types": list(self.edge_types) if self.edge_types is not None else None,
                "limit": self.limit,
                "timeout_seconds": self.timeout_seconds,
            }
        )
        return _merge_extra(payload, self.extra)


@dataclass(frozen=True, slots=True)
class ChonkyDBGraphPathRequest:
    index_name: str | None = None
    source_label: str | None = None
    source: str | None = None
    target_label: str | None = None
    target: str | None = None
    edge_types: tuple[str, ...] | None = None
    return_ids: bool = False
    extra: Mapping[str, object] | None = None

    def to_payload(self) -> JsonDict:
        payload = _drop_none(
            {
                "index_name": self.index_name,
                "source_label": self.source_label,
                "source": self.source,
                "target_label": self.target_label,
                "target": self.target,
                "edge_types": list(self.edge_types) if self.edge_types is not None else None,
                "return_ids": self.return_ids,
            }
        )
        return _merge_extra(payload, self.extra)


@dataclass(frozen=True, slots=True)
class ChonkyDBGraphPatternsRequest:
    patterns: tuple[Mapping[str, object], ...]
    index_name: str | None = None
    limit: int = 10
    max_depth: int = 5
    include_content: bool = True
    extra: Mapping[str, object] | None = None

    def to_payload(self) -> JsonDict:
        payload = _drop_none(
            {
                "index_name": self.index_name,
                "patterns": [dict(pattern) for pattern in self.patterns],
                "limit": self.limit,
                "max_depth": self.max_depth,
                "include_content": self.include_content,
            }
        )
        return _merge_extra(payload, self.extra)


@dataclass(frozen=True, slots=True)
class ChonkyDBRetrieveHit:
    payload_id: str | None
    doc_id_int: int | None
    score: float | None
    relevance_score: float | None
    source_index: str | None
    candidate_origin: str | None
    metadata: Mapping[str, object] | None
    raw: JsonDict = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ChonkyDBRetrieveHit":
        raw = dict(payload)
        score = raw.get("score")
        relevance_score = raw.get("relevance_score")
        doc_id_int = raw.get("doc_id_int")
        metadata = raw.get("metadata")
        return cls(
            payload_id=str(raw["payload_id"]) if raw.get("payload_id") is not None else None,
            doc_id_int=int(doc_id_int) if isinstance(doc_id_int, (int, float)) else None,
            score=float(score) if isinstance(score, (int, float)) else None,
            relevance_score=float(relevance_score) if isinstance(relevance_score, (int, float)) else None,
            source_index=str(raw["source_index"]) if raw.get("source_index") is not None else None,
            candidate_origin=(
                str(raw["candidate_origin"]) if raw.get("candidate_origin") is not None else None
            ),
            metadata=dict(metadata) if isinstance(metadata, Mapping) else None,
            raw=raw,
        )


@dataclass(frozen=True, slots=True)
class ChonkyDBRetrieveResponse:
    success: bool
    mode: str
    results: tuple[ChonkyDBRetrieveHit, ...]
    indexes_used: tuple[str, ...]
    raw: JsonDict = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ChonkyDBRetrieveResponse":
        raw = dict(payload)
        results = tuple(
            ChonkyDBRetrieveHit.from_payload(item)
            for item in raw.get("results", ())
            if isinstance(item, Mapping)
        )
        indexes_used = tuple(str(item) for item in raw.get("indexes_used", ()) if isinstance(item, str))
        return cls(
            success=bool(raw.get("success", False)),
            mode=str(raw.get("mode", "")),
            results=results,
            indexes_used=indexes_used,
            raw=raw,
        )


@dataclass(frozen=True, slots=True)
class ChonkyDBRecordSummary:
    payload_id: str
    chonky_id: str | None
    metadata: Mapping[str, object] | None
    raw: JsonDict = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ChonkyDBRecordSummary":
        raw = dict(payload)
        metadata = raw.get("metadata")
        return cls(
            payload_id=str(raw.get("payload_id", "")),
            chonky_id=str(raw["chonky_id"]) if raw.get("chonky_id") is not None else None,
            metadata=dict(metadata) if isinstance(metadata, Mapping) else None,
            raw=raw,
        )


@dataclass(frozen=True, slots=True)
class ChonkyDBRecordListResponse:
    success: bool
    offset: int
    limit: int
    total_count: int
    returned_count: int
    payloads: tuple[ChonkyDBRecordSummary, ...]
    raw: JsonDict = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ChonkyDBRecordListResponse":
        raw = dict(payload)
        payloads = tuple(
            ChonkyDBRecordSummary.from_payload(item)
            for item in raw.get("payloads", ())
            if isinstance(item, Mapping)
        )

        def _int_value(key: str) -> int:
            value = raw.get(key, 0)
            return int(value) if isinstance(value, (int, float)) else 0

        return cls(
            success=bool(raw.get("success", False)),
            offset=_int_value("offset"),
            limit=_int_value("limit"),
            total_count=_int_value("total_count"),
            returned_count=_int_value("returned_count"),
            payloads=payloads,
            raw=raw,
        )
