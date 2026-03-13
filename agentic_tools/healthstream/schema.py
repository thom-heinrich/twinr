"""
Contract
- Purpose:
  - Stable JSON schema payloads for the `healthstream` tool and its stored events.
  - Designed for LLM agents and future portal ingestion (machine-readable, versioned).
- Inputs (types, units):
  - None (pure metadata).
- Outputs (types, units):
  - JSON-serializable dicts: tool discovery schema + JSON Schema for store/events.
- Invariants:
  - Schema payload must be backward-compatible within a major `schema_version`.
  - Vocab enums must match `validation.ALLOWED_*`.
- Error semantics:
  - Never raises (pure constants); callers treat output as data.
- External boundaries:
  - None.
- Telemetry:
  - None.
"""

##REFACTOR: 2026-01-16##

import copy
import functools
import re
from typing import Any, Dict

from agentic_tools.healthstream.validation import (
    ALLOWED_KINDS,
    ALLOWED_SEVERITIES,
    ALLOWED_STATUSES,
    LINK_TOKEN_RE,
)

try:
    # Optional, schema-safe ECMA-262 pattern constant if the validation module provides it.
    from agentic_tools.healthstream.validation import LINK_TOKEN_PATTERN as _LINK_TOKEN_PATTERN  # type: ignore
except Exception:
    _LINK_TOKEN_PATTERN = None  # type: ignore[assignment]


_EVENT_ID_PATTERN = r"^HS[0-9]{6,}$"
_ISO_ZULU_TIMESTAMP_PATTERN = (
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)


def _python_regex_pattern_to_ecma262(py_pattern: str) -> str:
    """
    Best-effort conversion from a Python `re` pattern to an ECMA-262 compatible
    regex *pattern string* for JSON Schema's `pattern` keyword.

    Constraints:
      - Must never raise.
      - Prefer preserving the original pattern verbatim when it already appears
        portable to avoid unnecessary schema churn.
      - If the pattern appears non-portable (Python-only constructs), fall back to a
        permissive pattern to prevent schema ingestion failures in non-Python validators.
    """
    try:
        pattern = py_pattern or r".*"

        # Fast-path: keep verbatim when no common Python-only constructs appear.
        # Note: JSON Schema `pattern` is not implicitly anchored; preserve that behavior.
        if (
            r"\A" not in pattern
            and r"\Z" not in pattern
            and "(?P<" not in pattern
            and "(?P=" not in pattern
            and "(?(" not in pattern
            and r"\G" not in pattern
        ):
            # Inline flag groups are not reliably supported across JS engines used by validators.
            if re.search(r"\(\?[aiLmsux]+[\):]", pattern) is None:
                return pattern

        # Mostly semantics-preserving conversions:
        # Python: \A, \Z  -> anchors more widely supported in ECMA-262: ^, $
        pattern = pattern.replace(r"\A", "^").replace(r"\Z", "$")

        # Python: (?P<name>...) -> ECMA-262: (?<name>...)
        pattern = re.sub(r"\(\?P<([A-Za-z_][A-Za-z0-9_]*)>", r"(?<\1>", pattern)

        # Python: (?P=name) -> ECMA-262: \k<name>
        pattern = re.sub(
            r"\(\?P=([A-Za-z_][A-Za-z0-9_]*)\)", r"\\k<\1>", pattern
        )

        # Non-portable constructs with no safe representation in JSON Schema pattern:
        if "(?(" in pattern or r"\G" in pattern:
            return r".*"

        # Inline flags (e.g. (?i), (?m:...)) remain a portability risk.
        if re.search(r"\(\?[aiLmsux]+[\):]", pattern) is not None:
            return r".*"

        return pattern
    except Exception:
        return r".*"


def _schema_link_token_pattern() -> str:
    """
    Return an ECMA-262-friendly link token pattern for embedding into JSON Schema.
    Never raises.
    """
    if isinstance(_LINK_TOKEN_PATTERN, str) and _LINK_TOKEN_PATTERN:
        return _LINK_TOKEN_PATTERN
    try:
        return _python_regex_pattern_to_ecma262(getattr(LINK_TOKEN_RE, "pattern", ""))
    except Exception:
        return r".*"


@functools.lru_cache(maxsize=1)
def _build_schema_payload_template() -> Dict[str, Any]:
    """
    Build the schema payload once and cache it. The public API returns a deep copy
    to preserve the historical behavior of returning a fresh dict per call.
    """
    commands = {
        "schema": "Emit tool + store/event JSON schema (machine-readable).",
        "init": "Initialize the store (idempotent).",
        "emit": "Append an event (optionally deduped).",
        "list": "List recent events with filters + paging.",
        "get": "Fetch a single event by id.",
        "prune": "Prune old events (retention).",
    }

    allowed_kinds = sorted(ALLOWED_KINDS)
    allowed_statuses = sorted(ALLOWED_STATUSES)
    allowed_severities = sorted(ALLOWED_SEVERITIES)
    link_token_pattern = _schema_link_token_pattern()

    event_schema: Dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "HealthStreamEvent",
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "created_at", "kind", "status", "severity", "source"],
        "properties": {
            "id": {"type": "string", "pattern": _EVENT_ID_PATTERN},
            "created_at": {"type": "string", "pattern": _ISO_ZULU_TIMESTAMP_PATTERN},
            "kind": {"type": "string", "enum": allowed_kinds},
            "status": {"type": "string", "enum": allowed_statuses},
            "severity": {"type": "string", "enum": allowed_severities},
            "source": {"type": "string", "minLength": 1, "maxLength": 256},
            "text": {"type": "string", "maxLength": 4096},
            "channel": {"type": "string", "maxLength": 64},
            "actor": {"type": "string", "maxLength": 128},
            "origin": {"type": "string", "maxLength": 64},
            "dedupe_key": {"type": "string", "maxLength": 256},
            "artifacts": {
                "type": "array",
                "items": {"type": "string", "maxLength": 512},
                "maxItems": 64,
            },
            "tags": {
                "type": "array",
                "items": {"type": "string", "maxLength": 64},
                "maxItems": 64,
            },
            "links": {
                "type": "array",
                "items": {"type": "string", "pattern": link_token_pattern},
                "maxItems": 64,
            },
            "data": {
                "type": "object",
                "additionalProperties": True,
                "maxProperties": 256,
                "propertyNames": {"type": "string", "maxLength": 256},
            },
        },
    }

    store_schema: Dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "HealthStreamStore",
        "type": "object",
        "additionalProperties": False,
        "required": ["version", "created_at", "updated_at", "revision", "next_event_id", "events"],
        "properties": {
            "version": {"type": "integer", "minimum": 1},
            "created_at": {"type": "string"},
            "updated_at": {"type": "string"},
            "revision": {"type": "integer", "minimum": 0},
            "next_event_id": {"type": "integer", "minimum": 1},
            "events": {
                "type": "object",
                "propertyNames": {"type": "string", "pattern": _EVENT_ID_PATTERN},
                "additionalProperties": event_schema,
            },
        },
    }

    return {
        "ok": True,
        "schema_version": 1,
        "tool": {
            "summary": "Central health/DQ/ops event stream (file-backed, JSON-only).",
            "env": {
                "HEALTHSTREAM_FILE": "Override store path (default: .healthstream.json).",
                "HEALTHSTREAM_LOCK_TIMEOUT_SEC": "Lock acquisition timeout (default: 10).",
            },
            "stdout": "json_only",
            "commands": commands,
        },
        "vocab": {
            "kind": allowed_kinds,
            "status": allowed_statuses,
            "severity": allowed_severities,
            "link_token_pattern": link_token_pattern,
        },
        "json_schema": {
            "event": event_schema,
            "store": store_schema,
        },
    }


def build_schema_payload() -> Dict[str, Any]:
    """
    Return a stable schema payload for tool discovery + portal/agent consumption.
    """
    try:
        return copy.deepcopy(_build_schema_payload_template())
    except Exception:
        template = _build_schema_payload_template()
        return dict(template)
