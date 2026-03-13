"""
Contract
- Purpose:
  - Typed data model for the healthstream store (events + store metadata).
- Inputs (types, units):
  - Event fields are normalized by `validation.normalize_*` helpers.
- Outputs (types, units):
  - `HealthStreamEvent` dataclass (serializable to/from dict).
- Invariants:
  - `created_at` is UTC ISO8601 with Z suffix.
  - `id` is monotonically increasing within a store (HS000001, ...).
- Error semantics:
  - `HealthStreamEvent.from_dict` raises `ValueError` on schema violations.
- External boundaries:
  - None (pure types).
- Telemetry:
  - None.
- Notes:
  - `frozen=True` is shallow: container fields (lists/dicts) remain mutable if mutated in-place.
"""

##REFACTOR: 2026-01-16##

from __future__ import annotations  # NOTE: deprecated in Python 3.14+; kept for cross-version compatibility.

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Mapping, Optional


JsonObject = Dict[str, Any]

_HS_ID_RE = re.compile(r"^HS\d{6,}$")
_UTC_ZERO = timedelta(0)


def _validate_id(value: str) -> None:
    if not _HS_ID_RE.fullmatch(value):
        raise ValueError("event.id must match HS000001-style format (HS + >=6 digits)")


def _validate_created_at(value: str) -> None:
    # Contract requires a UTC ISO8601 timestamp with a trailing "Z".
    if not value.endswith("Z"):
        raise ValueError("event.created_at must be UTC ISO8601 with Z suffix")

    # Python 3.11+ accepts "Z" directly; older versions may not. Keep a safe fallback.
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        try:
            dt = datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError:
            raise ValueError("event.created_at must be a valid ISO8601 UTC timestamp with Z suffix") from None

    # Ensure an actual timestamp (not date-only) and UTC offset.
    if dt.tzinfo is None or dt.utcoffset() != _UTC_ZERO:
        raise ValueError("event.created_at must be UTC ISO8601 with Z suffix")


def _validate_max_str_len(field: str, value: Optional[str], max_len: int) -> None:
    if value is not None and len(value) > max_len:
        raise ValueError(f"event.{field} exceeds max length {max_len}")


def _validate_max_list_len(field: str, value: Optional[List[str]], max_len: int) -> None:
    if value is not None and len(value) > max_len:
        raise ValueError(f"event.{field} exceeds max length {max_len}")


def _validate_max_data_keys(field: str, value: Optional[dict], max_keys: int) -> None:
    if value is not None and len(value) > max_keys:
        raise ValueError(f"event.{field} exceeds max keys {max_keys}")


@dataclass(frozen=True, slots=True)
class HealthStreamEvent:
    id: str
    created_at: str
    kind: str
    status: str
    severity: str
    source: str
    text: Optional[str] = None
    channel: Optional[str] = None
    actor: Optional[str] = None
    origin: Optional[str] = None
    dedupe_key: Optional[str] = None
    artifacts: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    links: Optional[List[str]] = None
    data: Optional[JsonObject] = None

    def __hash__(self) -> int:
        # Hash only immutable fields so hashing is stable even if container fields are mutated in-place.
        return hash(
            (
                self.id,
                self.created_at,
                self.kind,
                self.status,
                self.severity,
                self.source,
                self.text,
                self.channel,
                self.actor,
                self.origin,
                self.dedupe_key,
            )
        )

    def to_dict(self) -> JsonObject:
        out: JsonObject = {
            "id": self.id,
            "created_at": self.created_at,
            "kind": self.kind,
            "status": self.status,
            "severity": self.severity,
            "source": self.source,
        }
        if self.text is not None:
            out["text"] = self.text
        if self.channel is not None:
            out["channel"] = self.channel
        if self.actor is not None:
            out["actor"] = self.actor
        if self.origin is not None:
            out["origin"] = self.origin
        if self.dedupe_key is not None:
            out["dedupe_key"] = self.dedupe_key
        if self.artifacts is not None:
            out["artifacts"] = list(self.artifacts)
        if self.tags is not None:
            out["tags"] = list(self.tags)
        if self.links is not None:
            out["links"] = list(self.links)
        if self.data is not None:
            out["data"] = dict(self.data)
        return out

    @staticmethod
    def from_dict(
        obj: Mapping[str, Any],
        *,
        max_list_len: Optional[int] = None,
        max_data_keys: Optional[int] = None,
        max_str_len: Optional[int] = None,
    ) -> "HealthStreamEvent":
        if not isinstance(obj, Mapping):
            raise ValueError("event must be a mapping")

        idv = obj.get("id")
        created_at = obj.get("created_at")
        kind = obj.get("kind")
        status = obj.get("status")
        severity = obj.get("severity")
        source = obj.get("source")
        if not all(isinstance(x, str) and x.strip() for x in [idv, created_at, kind, status, severity, source]):
            raise ValueError("event missing required string fields: id/created_at/kind/status/severity/source")

        # Enforce contract invariants (format-level guards; monotonicity is store-level).
        _validate_id(idv)
        _validate_created_at(created_at)

        text = obj.get("text")
        channel = obj.get("channel")
        actor = obj.get("actor")
        origin = obj.get("origin")
        dedupe_key = obj.get("dedupe_key")
        artifacts = obj.get("artifacts")
        tags = obj.get("tags")
        links = obj.get("links")
        data = obj.get("data")

        if text is not None and not isinstance(text, str):
            raise ValueError("event.text must be a string when present")
        for k, v in [("channel", channel), ("actor", actor), ("origin", origin), ("dedupe_key", dedupe_key)]:
            if v is not None and not isinstance(v, str):
                raise ValueError(f"event.{k} must be a string when present")

        if artifacts is not None and not (isinstance(artifacts, list) and all(isinstance(x, str) for x in artifacts)):
            raise ValueError("event.artifacts must be a list[str] when present")
        if tags is not None and not (isinstance(tags, list) and all(isinstance(x, str) for x in tags)):
            raise ValueError("event.tags must be a list[str] when present")
        if links is not None and not (isinstance(links, list) and all(isinstance(x, str) for x in links)):
            raise ValueError("event.links must be a list[str] when present")
        if data is not None and not isinstance(data, dict):
            raise ValueError("event.data must be an object when present")

        # Optional defensive limits (default preserves prior behavior).
        if max_str_len is not None:
            _validate_max_str_len("id", idv, max_str_len)
            _validate_max_str_len("created_at", created_at, max_str_len)
            _validate_max_str_len("kind", kind, max_str_len)
            _validate_max_str_len("status", status, max_str_len)
            _validate_max_str_len("severity", severity, max_str_len)
            _validate_max_str_len("source", source, max_str_len)
            _validate_max_str_len("text", text, max_str_len)
            _validate_max_str_len("channel", channel, max_str_len)
            _validate_max_str_len("actor", actor, max_str_len)
            _validate_max_str_len("origin", origin, max_str_len)
            _validate_max_str_len("dedupe_key", dedupe_key, max_str_len)
            if artifacts is not None and any(len(x) > max_str_len for x in artifacts):
                raise ValueError(f"event.artifacts contains element exceeding max length {max_str_len}")
            if tags is not None and any(len(x) > max_str_len for x in tags):
                raise ValueError(f"event.tags contains element exceeding max length {max_str_len}")
            if links is not None and any(len(x) > max_str_len for x in links):
                raise ValueError(f"event.links contains element exceeding max length {max_str_len}")

        if max_list_len is not None:
            _validate_max_list_len("artifacts", artifacts, max_list_len)
            _validate_max_list_len("tags", tags, max_list_len)
            _validate_max_list_len("links", links, max_list_len)

        if max_data_keys is not None:
            _validate_max_data_keys("data", data, max_data_keys)

        return HealthStreamEvent(
            id=str(idv),
            created_at=str(created_at),
            kind=str(kind),
            status=str(status),
            severity=str(severity),
            source=str(source),
            text=str(text) if text is not None else None,
            channel=str(channel) if channel is not None else None,
            actor=str(actor) if actor is not None else None,
            origin=str(origin) if origin is not None else None,
            dedupe_key=str(dedupe_key) if dedupe_key is not None else None,
            artifacts=list(artifacts) if artifacts is not None else None,
            tags=list(tags) if tags is not None else None,
            links=list(links) if links is not None else None,
            data=dict(data) if data is not None else None,
        )
