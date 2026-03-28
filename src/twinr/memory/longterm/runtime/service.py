"""Compatibility shim for runtime-facing long-term memory service exports.

##REFACTOR: 2026-03-27##

The implementation now lives under ``service_impl/``. Import from this module
exactly as before; no caller changes are required.
"""

from __future__ import annotations

from .service_impl import (
    _DEFAULT_REVIEW_LIMIT,
    _JSON_DEPTH_LIMIT,
    _JSON_ITEMS_LIMIT,
    _JSON_STRING_LIMIT,
    _MAX_QUEUE_SIZE,
    _MAX_REVIEW_LIMIT,
    _MULTIMODAL_MESSAGE_LIMIT,
    _MULTIMODAL_NAME_LIMIT,
    _SOURCE_LIMIT,
    _TEXT_LIMIT,
    LongTermMemoryService,
    LongTermRemoteReadinessResult,
    LongTermRemoteReadinessStep,
    _coerce_positive_int,
    _coerce_timeout_s,
    _context_details,
    _load_timezone,
    _normalize_datetime,
    _normalize_text,
    _sanitize_jsonish,
    _serialize_datetime,
    _sort_conflicts,
    _sort_objects_by_memory_id,
    _validate_regular_file_path,
    _writer_state_details,
)

LongTermMemoryService.__module__ = __name__
LongTermRemoteReadinessStep.__module__ = __name__
LongTermRemoteReadinessResult.__module__ = __name__

__all__ = [
    "_DEFAULT_REVIEW_LIMIT",
    "_JSON_DEPTH_LIMIT",
    "_JSON_ITEMS_LIMIT",
    "_JSON_STRING_LIMIT",
    "_MAX_QUEUE_SIZE",
    "_MAX_REVIEW_LIMIT",
    "_MULTIMODAL_MESSAGE_LIMIT",
    "_MULTIMODAL_NAME_LIMIT",
    "_SOURCE_LIMIT",
    "_TEXT_LIMIT",
    "LongTermMemoryService",
    "LongTermRemoteReadinessResult",
    "LongTermRemoteReadinessStep",
    "_coerce_positive_int",
    "_coerce_timeout_s",
    "_context_details",
    "_load_timezone",
    "_normalize_datetime",
    "_normalize_text",
    "_sanitize_jsonish",
    "_serialize_datetime",
    "_sort_conflicts",
    "_sort_objects_by_memory_id",
    "_validate_regular_file_path",
    "_writer_state_details",
]
