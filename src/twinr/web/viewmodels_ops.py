from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

_MAX_PRETTY_JSON_CHARS = 8_192  # AUDIT-FIX(#6): Cap rendered payload size to keep diagnostics responsive on RPi-class hardware.
_MAX_SERIALIZATION_DEPTH = 8  # AUDIT-FIX(#2): Bound nested serialization to avoid recursion blowups on malformed payloads.
_REDACTED = "***REDACTED***"

_SECRET_ASSIGNMENT_PATTERNS = (
    re.compile(
        r"(?i)\b(authorization)\s*([:=])\s*(?:bearer|basic)?\s*([A-Za-z0-9._~+/=-]+)"
    ),
    re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|password|passwd|cookie|session(?:id)?|private[_-]?key)\s*([:=])\s*([^\s,;]+)"
    ),
)

_SENSITIVE_NORMALIZED_KEYS = {
    "accesstoken",
    "apikey",
    "apitoken",
    "authorization",
    "clientsecret",
    "cookie",
    "credential",
    "credentials",
    "passwd",
    "password",
    "privatekey",
    "refreshtoken",
    "secret",
    "sessionid",
    "sessiontoken",
    "wifipassword",
}

def _snapshot_iterable(items: Iterable[Any] | None) -> tuple[Any, ...]:
    # AUDIT-FIX(#4): Snapshot iterables defensively so callers can pass generators or None without crashing.
    return () if items is None else tuple(items)

def _coalesce_display(value: Any, default: object) -> Any:
    # AUDIT-FIX(#7): Normalize explicit None/empty-string values to stable UI defaults instead of leaking raw None into templates.
    if value is None:
        return default
    if isinstance(value, str) and value == "":
        return default
    return value

def _safe_field(source: object, name: str, default: Any = None) -> Any:
    # AUDIT-FIX(#3): Tolerate dict-backed, attribute-backed, and partially broken objects without taking down the diagnostics view.
    if isinstance(source, Mapping):
        value = source.get(name, default)
    else:
        try:
            value = getattr(source, name)
        except Exception:
            return default
    return default if value is None else value

def _mapping_view(value: Any) -> Mapping[Any, Any] | None:
    # AUDIT-FIX(#3): Accept mapping-like payloads exposed via dicts, dataclasses, or helper objects with dump methods.
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    for dump_name in ("to_dict", "model_dump"):
        dump = getattr(value, dump_name, None)
        if callable(dump):
            try:
                dumped = dump()
            except Exception:
                return None
            if isinstance(dumped, Mapping):
                return dict(dumped)
            return None
    return None

def _is_sensitive_key(key: str) -> bool:
    # AUDIT-FIX(#1): Detect common credential-bearing field names before structured data reaches any UI or export surface.
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return (
        normalized in _SENSITIVE_NORMALIZED_KEYS
        or (normalized.endswith("token") and not normalized.endswith("tokens"))
        or normalized.endswith("secret")
        or normalized.endswith("password")
        or normalized.endswith("passwd")
        or normalized.endswith("apikey")
        or normalized.endswith("privatekey")
    )

def _redact_string_secrets(text: str) -> str:
    # AUDIT-FIX(#1): Scrub obvious key/value credential patterns from free-form strings such as log messages and repr outputs.
    redacted = text
    for pattern in _SECRET_ASSIGNMENT_PATTERNS:
        redacted = pattern.sub(lambda match: f"{match.group(1)}{match.group(2)} {_REDACTED}", redacted)
    return redacted

def _sanitize_for_display(value: Any, *, _seen: set[int] | None = None, depth: int = 0) -> Any:
    # AUDIT-FIX(#2): Serialize arbitrarily-shaped payloads safely, including mixed keys, nested models, and circular references.
    if isinstance(value, str):
        return _redact_string_secrets(value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if depth >= _MAX_SERIALIZATION_DEPTH:
        return "[max-depth-exceeded]"

    if _seen is None:
        _seen = set()

    obj_id = id(value)
    if obj_id in _seen:
        return "[circular-reference]"

    if isinstance(value, Mapping):
        _seen.add(obj_id)
        try:
            sanitized: dict[str, Any] = {}
            for key, item in value.items():
                key_str = str(key)
                sanitized[key_str] = (
                    _REDACTED
                    if _is_sensitive_key(key_str)
                    else _sanitize_for_display(item, _seen=_seen, depth=depth + 1)
                )
            return sanitized
        finally:
            _seen.discard(obj_id)

    if isinstance(value, (list, tuple, set, frozenset)):
        _seen.add(obj_id)
        try:
            return [_sanitize_for_display(item, _seen=_seen, depth=depth + 1) for item in value]
        finally:
            _seen.discard(obj_id)

    mapping_value = _mapping_view(value)
    if mapping_value is not None:
        return _sanitize_for_display(mapping_value, _seen=_seen, depth=depth + 1)

    return _redact_string_secrets(str(value))

def _safe_pretty_json(value: Any) -> str:
    # AUDIT-FIX(#6): Bound preview size so oversized transcripts or metadata blobs do not freeze low-power support UIs.
    sanitized = _sanitize_for_display(value)
    if isinstance(sanitized, str):
        rendered = sanitized
    else:
        try:
            rendered = json.dumps(
                sanitized,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        except (TypeError, ValueError):
            rendered = _redact_string_secrets(str(sanitized))

    if len(rendered) > _MAX_PRETTY_JSON_CHARS:
        truncated_by = len(rendered) - _MAX_PRETTY_JSON_CHARS
        rendered = f"{rendered[:_MAX_PRETTY_JSON_CHARS]}\n… [truncated {truncated_by} chars]"
    return rendered

def _has_pretty_payload(value: Any) -> bool:
    # AUDIT-FIX(#5): Render all non-empty payload shapes, not only dicts, so diagnostic evidence is not silently lost.
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        return bool(value)
    return True

def _safe_number(value: Any) -> float | None:
    # AUDIT-FIX(#3): Guard numeric formatting against unexpected strings, sentinels, NaN, and infinite values.
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

def _format_log_rows(entries: Iterable[Mapping[str, object]] | None) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for entry in reversed(_snapshot_iterable(entries)):
        entry_mapping = _mapping_view(entry) or {"message": entry}
        data = entry_mapping.get("data")
        rows.append(
            {
                "created_at": _coalesce_display(entry_mapping.get("created_at"), "—"),  # AUDIT-FIX(#7): Keep missing timestamps stable in the UI.
                "level": _coalesce_display(entry_mapping.get("level"), "info"),  # AUDIT-FIX(#7): Keep missing levels stable in the UI.
                "event": _coalesce_display(entry_mapping.get("event"), "unknown"),  # AUDIT-FIX(#7): Keep missing events stable in the UI.
                "message": _redact_string_secrets(str(_coalesce_display(entry_mapping.get("message"), ""))),  # AUDIT-FIX(#1): Prevent free-form log lines from leaking credentials.
                "data_pretty": (_safe_pretty_json(data) if _has_pretty_payload(data) else ""),  # AUDIT-FIX(#2): Serialize arbitrary payloads without crashing.
            }
        )
    return tuple(rows)

def _format_usage_rows(records: Iterable[object] | None) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for record in reversed(_snapshot_iterable(records)):
        token_usage = _mapping_view(_safe_field(record, "token_usage")) or {}
        metadata = _safe_field(record, "metadata")
        used_web_search = _safe_field(record, "used_web_search")
        rows.append(
            {
                "created_at": _coalesce_display(_safe_field(record, "created_at"), "—"),  # AUDIT-FIX(#7): Keep missing timestamps stable in the UI.
                "source": _coalesce_display(_safe_field(record, "source"), "—"),  # AUDIT-FIX(#3): Tolerate schema drift in usage records.
                "request_kind": _coalesce_display(_safe_field(record, "request_kind"), "—"),  # AUDIT-FIX(#3): Tolerate schema drift in usage records.
                "model": _coalesce_display(_safe_field(record, "model"), "unknown"),  # AUDIT-FIX(#7): Preserve legacy fallback behaviour for empty models.
                "response_id": _coalesce_display(_safe_field(record, "response_id"), "—"),  # AUDIT-FIX(#7): Keep missing IDs stable in the UI.
                "request_id": _coalesce_display(_safe_field(record, "request_id"), "—"),  # AUDIT-FIX(#7): Keep missing IDs stable in the UI.
                "used_web_search": (
                    "yes"
                    if used_web_search is True
                    else ("no" if used_web_search is False else "—")
                ),
                "total_tokens": _coalesce_display(_safe_field(record, "total_tokens"), "—"),  # AUDIT-FIX(#3): Preserve zero values while defaulting truly missing totals.
                "input_tokens": _coalesce_display(token_usage.get("input_tokens"), "—"),  # AUDIT-FIX(#3): Handle absent or malformed token usage maps safely.
                "output_tokens": _coalesce_display(token_usage.get("output_tokens"), "—"),  # AUDIT-FIX(#3): Handle absent or malformed token usage maps safely.
                "cached_input_tokens": _coalesce_display(token_usage.get("cached_input_tokens"), "—"),  # AUDIT-FIX(#3): Handle absent or malformed token usage maps safely.
                "reasoning_tokens": _coalesce_display(token_usage.get("reasoning_tokens"), "—"),  # AUDIT-FIX(#3): Handle absent or malformed token usage maps safely.
                "metadata_pretty": (_safe_pretty_json(metadata) if _has_pretty_payload(metadata) else ""),  # AUDIT-FIX(#1): Redact and serialize metadata safely before display.
            }
        )
    return tuple(rows)

def _health_card_detail(health) -> str:
    parts: list[str] = []

    cpu_temperature = _safe_number(_safe_field(health, "cpu_temperature_c"))
    if cpu_temperature is not None:
        parts.append(f"{cpu_temperature:.1f}C")  # AUDIT-FIX(#3): Skip non-numeric health metrics instead of raising during formatting.

    memory_used_percent = _safe_number(_safe_field(health, "memory_used_percent"))
    if memory_used_percent is not None:
        parts.append(f"mem {memory_used_percent:.0f}%")  # AUDIT-FIX(#3): Skip non-numeric health metrics instead of raising during formatting.

    disk_used_percent = _safe_number(_safe_field(health, "disk_used_percent"))
    if disk_used_percent is not None:
        parts.append(f"disk {disk_used_percent:.0f}%")  # AUDIT-FIX(#3): Skip non-numeric health metrics instead of raising during formatting.

    return " · ".join(parts) if parts else "Live Pi snapshot"