from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field, fields
from datetime import datetime
from math import isfinite
import re
from typing import Mapping
from urllib.parse import urlsplit, urlunsplit

JsonDict = dict[str, object]

_BOOLEAN_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_BOOLEAN_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_HTTP_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


class ChonkyDBValidationError(ValueError):
    """Raised when ChonkyDB request or response models violate the expected contract."""


# AUDIT-FIX(#1): `extra` darf keine reservierten Top-Level-Felder überschreiben und muss string-keyed sein.
def _merge_extra(
    payload: JsonDict,
    extra: Mapping[str, object] | None,
    *,
    reserved_keys: Iterable[str] | None = None,
) -> JsonDict:
    if extra:
        normalized_extra = _normalize_extra_mapping(extra)
        reserved = set(payload) if reserved_keys is None else set(reserved_keys)
        overlapping_keys = sorted(reserved.intersection(normalized_extra))
        if overlapping_keys:
            joined = ", ".join(overlapping_keys)
            raise ChonkyDBValidationError(
                f"extra contains reserved key(s) that would overwrite the base payload: {joined}"
            )
        payload.update(normalized_extra)
    return payload


def _drop_none(payload: JsonDict) -> JsonDict:
    return {key: value for key, value in payload.items() if value is not None}


# AUDIT-FIX(#4): Zentrale, explizite Typ-Normalisierung verhindert stille Bool/Int/Float-Fehlinterpretationen.
def _coerce_bool(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        raise ChonkyDBValidationError(f"{field_name} must be a boolean-compatible 0/1 integer, got {value!r}")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _BOOLEAN_TRUE_VALUES:
            return True
        if normalized in _BOOLEAN_FALSE_VALUES:
            return False
    raise ChonkyDBValidationError(f"{field_name} must be a boolean value, got {type(value).__name__}")


def _coerce_optional_bool(value: object, *, field_name: str) -> bool | None:
    if value is None:
        return None
    return _coerce_bool(value, field_name=field_name)


# AUDIT-FIX(#7): Bool ist in Python ein `int`; deshalb werden bools explizit ausgeschlossen und Zahlen validiert.
def _coerce_int(
    value: object,
    *,
    field_name: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool):
        raise ChonkyDBValidationError(f"{field_name} must be an integer, got bool")
    if isinstance(value, int):
        result = value
    elif isinstance(value, float):
        if not isfinite(value) or not value.is_integer():
            raise ChonkyDBValidationError(f"{field_name} must be an integer-compatible finite number, got {value!r}")
        result = int(value)
    elif isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ChonkyDBValidationError(f"{field_name} must not be blank")
        try:
            result = int(normalized, 10)
        except ValueError as exc:
            raise ChonkyDBValidationError(f"{field_name} must be an integer string, got {value!r}") from exc
    else:
        raise ChonkyDBValidationError(f"{field_name} must be an integer, got {type(value).__name__}")
    if minimum is not None and result < minimum:
        raise ChonkyDBValidationError(f"{field_name} must be >= {minimum}, got {result}")
    return result


def _coerce_optional_int(
    value: object,
    *,
    field_name: str,
    minimum: int | None = None,
) -> int | None:
    if value is None:
        return None
    return _coerce_int(value, field_name=field_name, minimum=minimum)


def _coerce_float(
    value: object,
    *,
    field_name: str,
    minimum: float | None = None,
    allow_zero: bool = True,
) -> float:
    if isinstance(value, bool):
        raise ChonkyDBValidationError(f"{field_name} must be a float, got bool")
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ChonkyDBValidationError(f"{field_name} must not be blank")
        try:
            result = float(normalized)
        except ValueError as exc:
            raise ChonkyDBValidationError(f"{field_name} must be a float string, got {value!r}") from exc
    else:
        raise ChonkyDBValidationError(f"{field_name} must be a float, got {type(value).__name__}")
    if not isfinite(result):
        raise ChonkyDBValidationError(f"{field_name} must be finite, got {value!r}")
    if minimum is not None and result < minimum:
        raise ChonkyDBValidationError(f"{field_name} must be >= {minimum}, got {result}")
    if not allow_zero and result == 0.0:
        raise ChonkyDBValidationError(f"{field_name} must be > 0, got 0")
    return result


def _coerce_optional_float(
    value: object,
    *,
    field_name: str,
    minimum: float | None = None,
    allow_zero: bool = True,
) -> float | None:
    if value is None:
        return None
    return _coerce_float(value, field_name=field_name, minimum=minimum, allow_zero=allow_zero)


def _coerce_required_str(
    value: object,
    *,
    field_name: str,
    strip: bool = True,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise ChonkyDBValidationError(f"{field_name} must be a string, got {type(value).__name__}")
    normalized = value.strip() if strip else value
    if not allow_empty and not normalized:
        raise ChonkyDBValidationError(f"{field_name} must not be blank")
    return normalized


def _coerce_optional_str(
    value: object,
    *,
    field_name: str,
    strip: bool = True,
    empty_as_none: bool = False,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ChonkyDBValidationError(f"{field_name} must be a string, got {type(value).__name__}")
    normalized = value.strip() if strip else value
    if normalized == "" and empty_as_none:
        return None
    return normalized


def _coerce_stringish(
    value: object,
    *,
    field_name: str,
    default: str | None = None,
) -> str | None:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        raise ChonkyDBValidationError(f"{field_name} must be a string, got bool")
    if isinstance(value, (int, float)):
        return str(value)
    raise ChonkyDBValidationError(f"{field_name} must be a string-compatible scalar, got {type(value).__name__}")


# AUDIT-FIX(#5): Sequenzfelder dürfen keine Einzel-Strings annehmen; sonst entstehen Zeichenlisten wie ['a', 'b', 'c'].
def _coerce_optional_str_tuple(
    value: object,
    *,
    field_name: str,
) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)):
        raise ChonkyDBValidationError(
            f"{field_name} must be an iterable of strings, not a single string/bytes value"
        )
    if isinstance(value, Mapping) or not isinstance(value, Iterable):
        raise ChonkyDBValidationError(f"{field_name} must be an iterable of strings")
    result: list[str] = []
    for index, item in enumerate(value):
        result.append(_coerce_required_str(item, field_name=f"{field_name}[{index}]"))
    return tuple(result)


def _coerce_optional_mapping(
    value: object,
    *,
    field_name: str,
) -> Mapping[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ChonkyDBValidationError(f"{field_name} must be a mapping, got {type(value).__name__}")
    return deepcopy(dict(value))


def _coerce_optional_mapping_tuple(
    value: object,
    *,
    field_name: str,
) -> tuple[Mapping[str, object], ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)):
        raise ChonkyDBValidationError(
            f"{field_name} must be an iterable of mappings, not a single string/bytes value"
        )
    if isinstance(value, Mapping) or not isinstance(value, Iterable):
        raise ChonkyDBValidationError(f"{field_name} must be an iterable of mappings")
    result: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        result.append(_coerce_optional_mapping(item, field_name=f"{field_name}[{index}]") or {})
    return tuple(result)


def _coerce_record_item_tuple(
    value: object,
    *,
    field_name: str,
) -> tuple["ChonkyDBRecordItem", ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ChonkyDBValidationError(
            f"{field_name} must be an iterable of ChonkyDBRecordItem instances, not a single string/bytes value"
        )
    if isinstance(value, Mapping) or not isinstance(value, Iterable):
        raise ChonkyDBValidationError(f"{field_name} must be an iterable of ChonkyDBRecordItem instances")
    result: list[ChonkyDBRecordItem] = []
    for index, item in enumerate(value):
        if not isinstance(item, ChonkyDBRecordItem):
            raise ChonkyDBValidationError(
                f"{field_name}[{index}] must be a ChonkyDBRecordItem, got {type(item).__name__}"
            )
        result.append(item)
    return tuple(result)


def _normalize_extra_mapping(extra: Mapping[str, object]) -> JsonDict:
    if not isinstance(extra, Mapping):
        raise ChonkyDBValidationError(f"extra must be a mapping, got {type(extra).__name__}")
    normalized: JsonDict = {}
    for key, value in dict(extra).items():
        if not isinstance(key, str):
            raise ChonkyDBValidationError(f"extra keys must be strings, got {type(key).__name__}")
        key_normalized = key.strip()
        if not key_normalized:
            raise ChonkyDBValidationError("extra keys must not be blank")
        if value is not None:
            normalized[key_normalized] = deepcopy(value)
    return normalized


def _reserved_payload_keys(instance: object) -> set[str]:
    try:
        dataclass_fields = fields(instance)
    except TypeError:
        return set()
    return {
        dataclass_field.name
        for dataclass_field in dataclass_fields
        if dataclass_field.name not in {"extra", "raw"}
    }


# AUDIT-FIX(#6): Zeitfenster werden als timezone-aware ISO-8601/RFC3339 validiert, damit DST- und Naive/aware-Fehler früh scheitern.
def _parse_aware_datetime(value: str, *, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ChonkyDBValidationError(
            f"{field_name} must be an ISO-8601/RFC3339 datetime string with timezone offset"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ChonkyDBValidationError(
            f"{field_name} must include an explicit timezone offset; naive datetimes are not allowed"
        )
    return parsed


# AUDIT-FIX(#2): Basis-URL und Header-Name werden normalisiert/validiert, um Fehlkonfigurationen und Header-Injection zu blockieren.
def _normalize_base_url(value: object) -> str:
    normalized = _coerce_required_str(value, field_name="base_url")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"}:
        raise ChonkyDBValidationError("base_url must use http or https")
    if not parsed.netloc:
        raise ChonkyDBValidationError("base_url must include a host")
    if parsed.username or parsed.password:
        raise ChonkyDBValidationError("base_url must not embed credentials")
    if parsed.query or parsed.fragment:
        raise ChonkyDBValidationError("base_url must not include query parameters or fragments")
    normalized_path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, normalized_path, "", ""))


def _normalize_header_name(value: object) -> str:
    normalized = _coerce_required_str(value, field_name="api_key_header")
    if not _HTTP_HEADER_NAME_RE.fullmatch(normalized):
        raise ChonkyDBValidationError("api_key_header contains invalid HTTP header field-name characters")
    return normalized


def _copy_raw_payload(payload: Mapping[str, object]) -> JsonDict:
    if not isinstance(payload, Mapping):
        raise ChonkyDBValidationError(f"payload must be a mapping, got {type(payload).__name__}")
    return deepcopy(dict(payload))


@dataclass(frozen=True, slots=True)
class ChonkyDBConnectionConfig:
    base_url: str
    api_key: str | None = None
    api_key_header: str = "x-api-key"
    allow_bearer_auth: bool = False
    timeout_s: float = 20.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", _normalize_base_url(self.base_url))  # AUDIT-FIX(#2)
        object.__setattr__(self, "api_key", _coerce_optional_str(self.api_key, field_name="api_key", empty_as_none=True))  # AUDIT-FIX(#2)
        object.__setattr__(self, "api_key_header", _normalize_header_name(self.api_key_header))  # AUDIT-FIX(#2)
        object.__setattr__(self, "allow_bearer_auth", _coerce_bool(self.allow_bearer_auth, field_name="allow_bearer_auth"))  # AUDIT-FIX(#4)
        object.__setattr__(self, "timeout_s", _coerce_float(self.timeout_s, field_name="timeout_s", minimum=0.0, allow_zero=False))  # AUDIT-FIX(#3)


@dataclass(frozen=True, slots=True)
class ChonkyDBInstanceInfo:
    success: bool
    service: str
    ready: bool
    auth_enabled: bool
    raw: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "success", _coerce_bool(self.success, field_name="success"))  # AUDIT-FIX(#4)
        object.__setattr__(self, "service", _coerce_required_str(self.service, field_name="service", allow_empty=True))  # AUDIT-FIX(#4)
        object.__setattr__(self, "ready", _coerce_bool(self.ready, field_name="ready"))  # AUDIT-FIX(#4)
        object.__setattr__(self, "auth_enabled", _coerce_bool(self.auth_enabled, field_name="auth_enabled"))  # AUDIT-FIX(#4)
        object.__setattr__(self, "raw", _copy_raw_payload(self.raw))  # AUDIT-FIX(#8)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ChonkyDBInstanceInfo":
        raw = _copy_raw_payload(payload)  # AUDIT-FIX(#8)
        return cls(
            success=raw.get("success", False),
            service=_coerce_stringish(raw.get("service"), field_name="service", default="") or "",
            ready=raw.get("ready", False),
            auth_enabled=raw.get("auth_enabled", False),
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "success", _coerce_bool(self.success, field_name="success"))  # AUDIT-FIX(#4)
        object.__setattr__(self, "auth_enabled", _coerce_bool(self.auth_enabled, field_name="auth_enabled"))  # AUDIT-FIX(#4)
        object.__setattr__(self, "scheme", _coerce_required_str(self.scheme, field_name="scheme", allow_empty=True))  # AUDIT-FIX(#4)
        object.__setattr__(self, "header_name", _normalize_header_name(self.header_name))  # AUDIT-FIX(#2)
        object.__setattr__(self, "allow_bearer", _coerce_bool(self.allow_bearer, field_name="allow_bearer"))  # AUDIT-FIX(#4)
        object.__setattr__(self, "exempt_paths", _coerce_optional_str_tuple(self.exempt_paths, field_name="exempt_paths") or ())  # AUDIT-FIX(#5)
        object.__setattr__(self, "api_key_configured", _coerce_bool(self.api_key_configured, field_name="api_key_configured"))  # AUDIT-FIX(#4)
        object.__setattr__(self, "raw", _copy_raw_payload(self.raw))  # AUDIT-FIX(#8)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ChonkyDBAuthInfo":
        raw = _copy_raw_payload(payload)  # AUDIT-FIX(#8)
        return cls(
            success=raw.get("success", False),
            auth_enabled=raw.get("auth_enabled", False),
            scheme=_coerce_stringish(raw.get("scheme"), field_name="scheme", default="") or "",
            header_name=_coerce_stringish(raw.get("header_name"), field_name="header_name", default="x-api-key") or "x-api-key",
            allow_bearer=raw.get("allow_bearer", False),
            exempt_paths=raw.get("exempt_paths", ()),
            api_key_configured=raw.get("api_key_configured", False),
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _coerce_optional_mapping(self.payload, field_name="payload"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "metadata", _coerce_optional_mapping(self.metadata, field_name="metadata"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "content", _coerce_optional_str(self.content, field_name="content"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "uri", _coerce_optional_str(self.uri, field_name="uri"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "file_path", _coerce_optional_str(self.file_path, field_name="file_path"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "file_type", _coerce_optional_str(self.file_type, field_name="file_type"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "language", _coerce_optional_str(self.language, field_name="language"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "tags", _coerce_optional_str_tuple(self.tags, field_name="tags"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "target_indexes", _coerce_optional_str_tuple(self.target_indexes, field_name="target_indexes"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "enable_chunking", _coerce_bool(self.enable_chunking, field_name="enable_chunking"))  # AUDIT-FIX(#4)
        object.__setattr__(self, "enable_llm_analysis", _coerce_optional_bool(self.enable_llm_analysis, field_name="enable_llm_analysis"))  # AUDIT-FIX(#4)
        object.__setattr__(self, "llm_focus_areas", _coerce_optional_str_tuple(self.llm_focus_areas, field_name="llm_focus_areas"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "include_insights_in_response", _coerce_bool(self.include_insights_in_response, field_name="include_insights_in_response"))  # AUDIT-FIX(#4)
        object.__setattr__(self, "timeout_seconds", _coerce_optional_float(self.timeout_seconds, field_name="timeout_seconds", minimum=0.0, allow_zero=False))  # AUDIT-FIX(#3)
        object.__setattr__(self, "targets", _coerce_optional_mapping_tuple(self.targets, field_name="targets"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "extra", _normalize_extra_mapping(self.extra) if self.extra is not None else None)  # AUDIT-FIX(#1)

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
        return _merge_extra(payload, self.extra, reserved_keys=_reserved_payload_keys(self))


@dataclass(frozen=True, slots=True)
class ChonkyDBRecordRequest(ChonkyDBRecordItem):
    operation: str = "store_payload"
    execution_mode: str = "sync"
    client_request_id: str | None = None

    def __post_init__(self) -> None:
        ChonkyDBRecordItem.__post_init__(self)
        object.__setattr__(self, "operation", _coerce_required_str(self.operation, field_name="operation"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "execution_mode", _coerce_required_str(self.execution_mode, field_name="execution_mode"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "client_request_id", _coerce_optional_str(self.client_request_id, field_name="client_request_id", empty_as_none=True))  # AUDIT-FIX(#5)

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

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", _coerce_record_item_tuple(self.items, field_name="items"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "operation", _coerce_required_str(self.operation, field_name="operation"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "execution_mode", _coerce_required_str(self.execution_mode, field_name="execution_mode"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "timeout_seconds", _coerce_optional_float(self.timeout_seconds, field_name="timeout_seconds", minimum=0.0, allow_zero=False))  # AUDIT-FIX(#3)
        object.__setattr__(self, "client_request_id", _coerce_optional_str(self.client_request_id, field_name="client_request_id", empty_as_none=True))  # AUDIT-FIX(#5)
        object.__setattr__(self, "finalize_vector_segments", _coerce_bool(self.finalize_vector_segments, field_name="finalize_vector_segments"))  # AUDIT-FIX(#4)
        object.__setattr__(self, "extra", _normalize_extra_mapping(self.extra) if self.extra is not None else None)  # AUDIT-FIX(#1)

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
        return _merge_extra(payload, self.extra, reserved_keys=_reserved_payload_keys(self))


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_text", _coerce_optional_str(self.query_text, field_name="query_text"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "mode", _coerce_required_str(self.mode, field_name="mode"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "result_limit", _coerce_int(self.result_limit, field_name="result_limit", minimum=1))  # AUDIT-FIX(#3)
        object.__setattr__(self, "include_content", _coerce_bool(self.include_content, field_name="include_content"))  # AUDIT-FIX(#4)
        object.__setattr__(self, "include_metadata", _coerce_bool(self.include_metadata, field_name="include_metadata"))  # AUDIT-FIX(#4)
        object.__setattr__(self, "content_mode", _coerce_optional_str(self.content_mode, field_name="content_mode"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "max_content_chars", _coerce_optional_int(self.max_content_chars, field_name="max_content_chars", minimum=1))  # AUDIT-FIX(#3)
        object.__setattr__(self, "timeout_seconds", _coerce_optional_float(self.timeout_seconds, field_name="timeout_seconds", minimum=0.0, allow_zero=False))  # AUDIT-FIX(#3)
        object.__setattr__(self, "filters", _coerce_optional_mapping(self.filters, field_name="filters"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "temporal_start", _coerce_optional_str(self.temporal_start, field_name="temporal_start", empty_as_none=True))  # AUDIT-FIX(#6)
        object.__setattr__(self, "temporal_end", _coerce_optional_str(self.temporal_end, field_name="temporal_end", empty_as_none=True))  # AUDIT-FIX(#6)
        if self.temporal_start is not None:
            start_dt = _parse_aware_datetime(self.temporal_start, field_name="temporal_start")  # AUDIT-FIX(#6)
        else:
            start_dt = None
        if self.temporal_end is not None:
            end_dt = _parse_aware_datetime(self.temporal_end, field_name="temporal_end")  # AUDIT-FIX(#6)
        else:
            end_dt = None
        if start_dt is not None and end_dt is not None and start_dt > end_dt:
            raise ChonkyDBValidationError("temporal_start must be <= temporal_end")
        object.__setattr__(self, "client_request_id", _coerce_optional_str(self.client_request_id, field_name="client_request_id", empty_as_none=True))  # AUDIT-FIX(#5)
        object.__setattr__(self, "allowed_indexes", _coerce_optional_str_tuple(self.allowed_indexes, field_name="allowed_indexes"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "extra", _normalize_extra_mapping(self.extra) if self.extra is not None else None)  # AUDIT-FIX(#1)

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
        return _merge_extra(payload, self.extra, reserved_keys=_reserved_payload_keys(self))


@dataclass(frozen=True, slots=True)
class ChonkyDBGraphAddEdgeRequest:
    from_id: int
    to_id: int
    edge_type: str
    allow_self_loops: bool | None = None
    extra: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "from_id", _coerce_int(self.from_id, field_name="from_id", minimum=0))  # AUDIT-FIX(#7)
        object.__setattr__(self, "to_id", _coerce_int(self.to_id, field_name="to_id", minimum=0))  # AUDIT-FIX(#7)
        object.__setattr__(self, "edge_type", _coerce_required_str(self.edge_type, field_name="edge_type"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "allow_self_loops", _coerce_optional_bool(self.allow_self_loops, field_name="allow_self_loops"))  # AUDIT-FIX(#4)
        object.__setattr__(self, "extra", _normalize_extra_mapping(self.extra) if self.extra is not None else None)  # AUDIT-FIX(#1)

    def to_payload(self) -> JsonDict:
        payload = _drop_none(
            {
                "from_id": self.from_id,
                "to_id": self.to_id,
                "edge_type": self.edge_type,
                "allow_self_loops": self.allow_self_loops,
            }
        )
        return _merge_extra(payload, self.extra, reserved_keys=_reserved_payload_keys(self))


@dataclass(frozen=True, slots=True)
class ChonkyDBGraphAddEdgeSmartRequest:
    from_ref: str
    to_ref: str
    edge_type: str
    extra: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "from_ref", _coerce_required_str(self.from_ref, field_name="from_ref"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "to_ref", _coerce_required_str(self.to_ref, field_name="to_ref"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "edge_type", _coerce_required_str(self.edge_type, field_name="edge_type"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "extra", _normalize_extra_mapping(self.extra) if self.extra is not None else None)  # AUDIT-FIX(#1)

    def to_payload(self) -> JsonDict:
        payload = _drop_none(
            {
                "from_ref": self.from_ref,
                "to_ref": self.to_ref,
                "edge_type": self.edge_type,
            }
        )
        return _merge_extra(payload, self.extra, reserved_keys=_reserved_payload_keys(self))


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "index_name", _coerce_optional_str(self.index_name, field_name="index_name"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "label_or_id", _coerce_optional_str(self.label_or_id, field_name="label_or_id"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "label", _coerce_optional_str(self.label, field_name="label"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "direction", _coerce_required_str(self.direction, field_name="direction"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "with_edges", _coerce_bool(self.with_edges, field_name="with_edges"))  # AUDIT-FIX(#4)
        object.__setattr__(self, "return_ids", _coerce_bool(self.return_ids, field_name="return_ids"))  # AUDIT-FIX(#4)
        object.__setattr__(self, "edge_types", _coerce_optional_str_tuple(self.edge_types, field_name="edge_types"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "limit", _coerce_optional_int(self.limit, field_name="limit", minimum=1))  # AUDIT-FIX(#3)
        object.__setattr__(self, "timeout_seconds", _coerce_optional_float(self.timeout_seconds, field_name="timeout_seconds", minimum=0.0, allow_zero=False))  # AUDIT-FIX(#3)
        object.__setattr__(self, "extra", _normalize_extra_mapping(self.extra) if self.extra is not None else None)  # AUDIT-FIX(#1)

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
        return _merge_extra(payload, self.extra, reserved_keys=_reserved_payload_keys(self))


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "index_name", _coerce_optional_str(self.index_name, field_name="index_name"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "source_label", _coerce_optional_str(self.source_label, field_name="source_label"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "source", _coerce_optional_str(self.source, field_name="source"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "target_label", _coerce_optional_str(self.target_label, field_name="target_label"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "target", _coerce_optional_str(self.target, field_name="target"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "edge_types", _coerce_optional_str_tuple(self.edge_types, field_name="edge_types"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "return_ids", _coerce_bool(self.return_ids, field_name="return_ids"))  # AUDIT-FIX(#4)
        object.__setattr__(self, "extra", _normalize_extra_mapping(self.extra) if self.extra is not None else None)  # AUDIT-FIX(#1)

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
        return _merge_extra(payload, self.extra, reserved_keys=_reserved_payload_keys(self))


@dataclass(frozen=True, slots=True)
class ChonkyDBGraphPatternsRequest:
    patterns: tuple[Mapping[str, object], ...]
    index_name: str | None = None
    limit: int = 10
    max_depth: int = 5
    include_content: bool = True
    extra: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "patterns", _coerce_optional_mapping_tuple(self.patterns, field_name="patterns") or ())  # AUDIT-FIX(#5)
        object.__setattr__(self, "index_name", _coerce_optional_str(self.index_name, field_name="index_name"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "limit", _coerce_int(self.limit, field_name="limit", minimum=1))  # AUDIT-FIX(#3)
        object.__setattr__(self, "max_depth", _coerce_int(self.max_depth, field_name="max_depth", minimum=1))  # AUDIT-FIX(#3)
        object.__setattr__(self, "include_content", _coerce_bool(self.include_content, field_name="include_content"))  # AUDIT-FIX(#4)
        object.__setattr__(self, "extra", _normalize_extra_mapping(self.extra) if self.extra is not None else None)  # AUDIT-FIX(#1)

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
        return _merge_extra(payload, self.extra, reserved_keys=_reserved_payload_keys(self))


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload_id", _coerce_optional_str(self.payload_id, field_name="payload_id"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "doc_id_int", _coerce_optional_int(self.doc_id_int, field_name="doc_id_int", minimum=0))  # AUDIT-FIX(#7)
        object.__setattr__(self, "score", _coerce_optional_float(self.score, field_name="score"))  # AUDIT-FIX(#7)
        object.__setattr__(self, "relevance_score", _coerce_optional_float(self.relevance_score, field_name="relevance_score"))  # AUDIT-FIX(#7)
        object.__setattr__(self, "source_index", _coerce_optional_str(self.source_index, field_name="source_index"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "candidate_origin", _coerce_optional_str(self.candidate_origin, field_name="candidate_origin"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "metadata", _coerce_optional_mapping(self.metadata, field_name="metadata"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "raw", _copy_raw_payload(self.raw))  # AUDIT-FIX(#8)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ChonkyDBRetrieveHit":
        raw = _copy_raw_payload(payload)  # AUDIT-FIX(#8)
        metadata = raw.get("metadata")
        return cls(
            payload_id=_coerce_stringish(raw.get("payload_id"), field_name="payload_id"),
            doc_id_int=raw.get("doc_id_int"),
            score=raw.get("score"),
            relevance_score=raw.get("relevance_score"),
            source_index=_coerce_stringish(raw.get("source_index"), field_name="source_index"),
            candidate_origin=_coerce_stringish(raw.get("candidate_origin"), field_name="candidate_origin"),
            metadata=_coerce_optional_mapping(metadata, field_name="metadata"),
            raw=raw,
        )


@dataclass(frozen=True, slots=True)
class ChonkyDBRetrieveResponse:
    success: bool
    mode: str
    results: tuple[ChonkyDBRetrieveHit, ...]
    indexes_used: tuple[str, ...]
    raw: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "success", _coerce_bool(self.success, field_name="success"))  # AUDIT-FIX(#4)
        object.__setattr__(self, "mode", _coerce_required_str(self.mode, field_name="mode", allow_empty=True))  # AUDIT-FIX(#5)
        if isinstance(self.results, (str, bytes, bytearray)) or isinstance(self.results, Mapping) or not isinstance(self.results, Iterable):
            raise ChonkyDBValidationError("results must be an iterable of ChonkyDBRetrieveHit instances")
        normalized_results: list[ChonkyDBRetrieveHit] = []
        for index, item in enumerate(self.results):
            if not isinstance(item, ChonkyDBRetrieveHit):
                raise ChonkyDBValidationError(
                    f"results[{index}] must be a ChonkyDBRetrieveHit, got {type(item).__name__}"
                )
            normalized_results.append(item)
        object.__setattr__(self, "results", tuple(normalized_results))  # AUDIT-FIX(#5)
        object.__setattr__(self, "indexes_used", _coerce_optional_str_tuple(self.indexes_used, field_name="indexes_used") or ())  # AUDIT-FIX(#5)
        object.__setattr__(self, "raw", _copy_raw_payload(self.raw))  # AUDIT-FIX(#8)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ChonkyDBRetrieveResponse":
        raw = _copy_raw_payload(payload)  # AUDIT-FIX(#8)
        raw_results = raw.get("results", ())
        if isinstance(raw_results, (str, bytes, bytearray)) or isinstance(raw_results, Mapping) or not isinstance(raw_results, Iterable):
            raise ChonkyDBValidationError("results must be an iterable of mappings")
        results_list: list[ChonkyDBRetrieveHit] = []
        for index, item in enumerate(raw_results):
            if not isinstance(item, Mapping):
                raise ChonkyDBValidationError(f"results[{index}] must be a mapping, got {type(item).__name__}")
            results_list.append(ChonkyDBRetrieveHit.from_payload(item))
        results = tuple(results_list)
        return cls(
            success=raw.get("success", False),
            mode=_coerce_stringish(raw.get("mode"), field_name="mode", default="") or "",
            results=results,
            indexes_used=raw.get("indexes_used", ()),
            raw=raw,
        )


@dataclass(frozen=True, slots=True)
class ChonkyDBRecordSummary:
    payload_id: str
    chonky_id: str | None
    metadata: Mapping[str, object] | None
    raw: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload_id", _coerce_required_str(self.payload_id, field_name="payload_id"))  # AUDIT-FIX(#8)
        object.__setattr__(self, "chonky_id", _coerce_optional_str(self.chonky_id, field_name="chonky_id"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "metadata", _coerce_optional_mapping(self.metadata, field_name="metadata"))  # AUDIT-FIX(#5)
        object.__setattr__(self, "raw", _copy_raw_payload(self.raw))  # AUDIT-FIX(#8)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ChonkyDBRecordSummary":
        raw = _copy_raw_payload(payload)  # AUDIT-FIX(#8)
        metadata = raw.get("metadata")
        payload_id = _coerce_stringish(raw.get("payload_id"), field_name="payload_id")
        if payload_id is None:
            raise ChonkyDBValidationError("payload_id is required in record summary")
        return cls(
            payload_id=payload_id,
            chonky_id=_coerce_stringish(raw.get("chonky_id"), field_name="chonky_id"),
            metadata=_coerce_optional_mapping(metadata, field_name="metadata"),
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "success", _coerce_bool(self.success, field_name="success"))  # AUDIT-FIX(#4)
        object.__setattr__(self, "offset", _coerce_int(self.offset, field_name="offset", minimum=0))  # AUDIT-FIX(#7)
        object.__setattr__(self, "limit", _coerce_int(self.limit, field_name="limit", minimum=0))  # AUDIT-FIX(#7)
        object.__setattr__(self, "total_count", _coerce_int(self.total_count, field_name="total_count", minimum=0))  # AUDIT-FIX(#7)
        object.__setattr__(self, "returned_count", _coerce_int(self.returned_count, field_name="returned_count", minimum=0))  # AUDIT-FIX(#7)
        if self.returned_count > self.total_count:
            raise ChonkyDBValidationError("returned_count must be <= total_count")
        if isinstance(self.payloads, (str, bytes, bytearray)) or isinstance(self.payloads, Mapping) or not isinstance(self.payloads, Iterable):
            raise ChonkyDBValidationError("payloads must be an iterable of ChonkyDBRecordSummary instances")
        normalized_payloads: list[ChonkyDBRecordSummary] = []
        for index, item in enumerate(self.payloads):
            if not isinstance(item, ChonkyDBRecordSummary):
                raise ChonkyDBValidationError(
                    f"payloads[{index}] must be a ChonkyDBRecordSummary, got {type(item).__name__}"
                )
            normalized_payloads.append(item)
        object.__setattr__(self, "payloads", tuple(normalized_payloads))  # AUDIT-FIX(#5)
        object.__setattr__(self, "raw", _copy_raw_payload(self.raw))  # AUDIT-FIX(#8)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ChonkyDBRecordListResponse":
        raw = _copy_raw_payload(payload)  # AUDIT-FIX(#8)
        raw_payloads = raw.get("payloads", ())
        if isinstance(raw_payloads, (str, bytes, bytearray)) or isinstance(raw_payloads, Mapping) or not isinstance(raw_payloads, Iterable):
            raise ChonkyDBValidationError("payloads must be an iterable of mappings")
        payload_list: list[ChonkyDBRecordSummary] = []
        for index, item in enumerate(raw_payloads):
            if not isinstance(item, Mapping):
                raise ChonkyDBValidationError(f"payloads[{index}] must be a mapping, got {type(item).__name__}")
            payload_list.append(ChonkyDBRecordSummary.from_payload(item))
        payloads = tuple(payload_list)
        return cls(
            success=raw.get("success", False),
            offset=raw.get("offset", 0),
            limit=raw.get("limit", 0),
            total_count=raw.get("total_count", 0),
            returned_count=raw.get("returned_count", 0),
            payloads=payloads,
            raw=raw,
        )