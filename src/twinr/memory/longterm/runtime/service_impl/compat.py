"""Shared helpers and compatibility types for the long-term runtime service."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import logging
import math

from twinr.memory.longterm.core.models import LongTermMemoryContext
from twinr.memory.longterm.runtime.health import LongTermRemoteWarmResult
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStatus

LOGGER_NAME = "twinr.memory.longterm.runtime.service"
logger = logging.getLogger(LOGGER_NAME)

_TEXT_LIMIT = 320
_SOURCE_LIMIT = 64
_MULTIMODAL_NAME_LIMIT = 128
_MULTIMODAL_MESSAGE_LIMIT = 512
_JSON_STRING_LIMIT = 512
_JSON_ITEMS_LIMIT = 64
_JSON_DEPTH_LIMIT = 5
_DEFAULT_REVIEW_LIMIT = 12
_MAX_REVIEW_LIMIT = 100
_MAX_QUEUE_SIZE = 4096


def _normalize_text(value: object, *, limit: int) -> str:
    """Normalize arbitrary input into one bounded single-line string."""

    if limit <= 0:
        return ""
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    if limit == 1:
        return "…"
    return text[: limit - 1].rstrip() + "…"


def _coerce_positive_int(value: object, *, default: int, maximum: int | None = None) -> int:
    """Coerce config-like input to a positive integer within bounds."""

    try:
        result = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        result = default
    if result <= 0:
        result = default
    if maximum is not None:
        result = min(result, maximum)
    return result


def _coerce_timeout_s(value: object, *, default: float) -> float:
    """Coerce timeout input to finite non-negative seconds."""

    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result) or result < 0:
        return default
    return result


def _writer_state_details(writer: object | None) -> dict[str, object] | None:
    """Return a bounded diagnostic snapshot for one async long-term writer."""

    if writer is None:
        return None
    snapshot = getattr(writer, "snapshot_state", None)
    if not callable(snapshot):
        return None
    try:
        state = snapshot()
    except Exception as exc:
        return {"snapshot_error": type(exc).__name__}
    return {
        "worker_name": getattr(state, "worker_name", None),
        "pending_count": getattr(state, "pending_count", None),
        "inflight_count": getattr(state, "inflight_count", None),
        "dropped_count": getattr(state, "dropped_count", None),
        "last_error_message": getattr(state, "last_error_message", None),
        "accepting": getattr(state, "accepting", None),
        "worker_alive": getattr(state, "worker_alive", None),
    }


def _context_details(context: LongTermMemoryContext) -> dict[str, object]:
    """Summarize which context sections were present for one turn."""

    return {
        "has_subtext_context": bool(getattr(context, "subtext_context", None)),
        "has_topic_context": bool(getattr(context, "topic_context", None)),
        "has_midterm_context": bool(getattr(context, "midterm_context", None)),
        "has_durable_context": bool(getattr(context, "durable_context", None)),
        "has_episodic_context": bool(getattr(context, "episodic_context", None)),
        "has_graph_context": bool(getattr(context, "graph_context", None)),
        "has_conflict_context": bool(getattr(context, "conflict_context", None)),
    }


@lru_cache(maxsize=16)
def _load_timezone(timezone_name: str) -> ZoneInfo:
    """Load and cache one timezone, falling back to UTC on unknown names."""

    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        logger.warning("Unknown timezone %r; falling back to UTC.", timezone_name)
        return ZoneInfo("UTC")


def _normalize_datetime(value: datetime | None, *, timezone_name: str) -> datetime | None:
    """Normalize a datetime into the configured local timezone."""

    if value is None:
        return None
    timezone = _load_timezone(timezone_name)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone)
    return value.astimezone(timezone)


def _serialize_datetime(value: datetime, *, timezone_name: str) -> str:
    """Serialize one datetime after timezone normalization."""

    normalized = _normalize_datetime(value, timezone_name=timezone_name)
    if normalized is None:
        return ""
    return normalized.isoformat()


def _sanitize_jsonish(
    value: Any,
    *,
    timezone_name: str,
    depth: int = 0,
) -> object:
    """Convert nested sensor payloads into bounded JSON-safe values."""

    if depth >= _JSON_DEPTH_LIMIT:
        return _normalize_text(repr(value), limit=_JSON_STRING_LIMIT)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _normalize_text(value, limit=_JSON_STRING_LIMIT)
    if isinstance(value, datetime):
        return _serialize_datetime(value, timezone_name=timezone_name)
    if isinstance(value, bytes):
        return {"__bytes_len__": len(value)}
    if isinstance(value, Mapping):
        sanitized: dict[str, object] = {}
        for index, (key, nested_value) in enumerate(value.items()):
            if index >= _JSON_ITEMS_LIMIT:
                sanitized["__truncated__"] = True
                break
            sanitized[_normalize_text(key, limit=64) or f"key_{index}"] = _sanitize_jsonish(
                nested_value,
                timezone_name=timezone_name,
                depth=depth + 1,
            )
        return sanitized
    if isinstance(value, Iterable):
        sanitized_items: list[object] = []
        for index, item in enumerate(value):
            if index >= _JSON_ITEMS_LIMIT:
                sanitized_items.append("…")
                break
            sanitized_items.append(
                _sanitize_jsonish(
                    item,
                    timezone_name=timezone_name,
                    depth=depth + 1,
                )
            )
        return sanitized_items
    return _normalize_text(repr(value), limit=_JSON_STRING_LIMIT)


def _validate_regular_file_path(path_value: object, *, allow_missing: bool) -> Path | None:
    """Accept only regular, non-symlink file paths for local file access."""

    try:
        if path_value is None:
            return None
        path = Path(path_value).expanduser()  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    try:
        if path.exists():
            if path.is_symlink() or not path.is_file():
                return None
        elif not allow_missing:
            return None
        parent = path.parent
        if parent.exists():
            if parent.is_symlink() or not parent.is_dir():
                return None
    except OSError:
        return None
    return path


def _sort_objects_by_memory_id(items: Iterable[Any]) -> tuple[Any, ...]:
    """Return memory objects sorted by stable memory ID."""

    return tuple(sorted(items, key=lambda row: row.memory_id))


def _sort_conflicts(items: Iterable[Any]) -> tuple[Any, ...]:
    """Return conflict objects sorted by slot and candidate memory ID."""

    return tuple(sorted(items, key=lambda row: (row.slot_key, row.candidate_memory_id)))


@dataclass(frozen=True, slots=True)
class LongTermRemoteReadinessStep:
    """Capture one readiness step executed by the runtime service."""

    name: str
    status: str
    latency_ms: float
    detail: str | None = None
    warm_result: LongTermRemoteWarmResult | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe summary for ops artifacts."""

        payload: dict[str, object] = {
            "name": self.name,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "detail": self.detail,
        }
        if self.warm_result is not None:
            payload["warm_result"] = self.warm_result.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class LongTermRemoteReadinessResult:
    """Capture the full fail-closed remote readiness proof for one probe."""

    ready: bool
    detail: str | None
    remote_status: LongTermRemoteStatus
    steps: tuple[LongTermRemoteReadinessStep, ...]
    warm_result: LongTermRemoteWarmResult | None = None
    total_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe summary for watchdog artifacts and debugging."""

        payload: dict[str, object] = {
            "ready": self.ready,
            "detail": self.detail,
            "remote_status": {
                "mode": self.remote_status.mode,
                "ready": self.remote_status.ready,
                "detail": self.remote_status.detail,
            },
            "steps": [step.to_dict() for step in self.steps],
            "total_latency_ms": self.total_latency_ms,
        }
        if self.warm_result is not None:
            payload["warm_result"] = self.warm_result.to_dict()
        return payload
