"""Manage runtime-facing reminder and automation mutations and queries."""

# CHANGELOG: 2026-03-27
# BUG-1: Fixed DST-invalid naive datetime normalization that could silently schedule/evaluate automations
#        at non-existent local times around timezone transitions.
# BUG-2: Fixed silent wrong-behavior inputs by normalizing blank event names to None, validating
#        timezone names, canonicalizing weekdays, and rejecting invalid weekday values.
# BUG-3: Fixed caller-mutation races by rejecting unsafely unsnapshotable runtime payloads instead
#        of silently reusing caller-owned mutable objects when deepcopy fails.
# SEC-1: Added bounded text/payload validation to block practical Raspberry-Pi memory/disk DoS via
#        oversized summaries, descriptions, actions, conditions, facts, tags, or error strings.
# SEC-2: Expanded secret redaction and control-character scrubbing for persisted failure text and
#        structured ops-event payloads.
# IMP-1: Added structured ops-event envelopes (schema_version/component/recorded_at/source) for
#        better 2026-grade observability and downstream correlation.
# IMP-2: Added canonical runtime input normalization, including datetime/time convenience inputs,
#        ISO-weekday support, tag dedupe, and safer note/telemetry text compaction.

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from datetime import datetime, time as dt_time, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    from dateutil import tz as _dateutil_tz
except Exception:  # pragma: no cover - optional dependency
    _dateutil_tz = None

from twinr.automations import AutomationCondition, AutomationDefinition
from twinr.automations.store import TimeAutomationMatch
from twinr.memory.reminders import ReminderEntry, format_due_label
from twinr.ops.events import compact_text


logger = logging.getLogger(__name__)

_SECRET_NAME_FRAGMENT = r"(?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password|passwd|authorization)"
_SECRET_JSON_PATTERN = re.compile(
    rf"""(?ix)
    (
        ["']{_SECRET_NAME_FRAGMENT}["']
        \s*:\s*
        ["']
    )
    ([^"']+)
    (["'])
    """
)
_SECRET_KV_PATTERN = re.compile(
    rf"""(?ix)
    (
        \b{_SECRET_NAME_FRAGMENT}\b
        \s*[:=]\s*
    )
    ([^\s,;]+)
    """
)
_BEARER_TOKEN_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_BASIC_TOKEN_PATTERN = re.compile(r"(?i)\bbasic\s+[A-Za-z0-9._~+/=-]+")
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]+")
_CONTROL_CHAR_PRESERVE_NL_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]+")
_SAFE_EVENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")

_OPS_EVENT_SCHEMA_VERSION = 2
_OPS_COMPONENT = "twinr.runtime_automation"

_MAX_KIND_LENGTH = 32
_MAX_SOURCE_LENGTH = 128
_MAX_NAME_LENGTH = 256
_MAX_EVENT_NAME_LENGTH = 256
_MAX_SCHEDULE_LENGTH = 64
_MAX_TIME_OF_DAY_LENGTH = 64
_MAX_TIMEZONE_NAME_LENGTH = 128
_MAX_DUE_AT_LENGTH = 256
_MAX_SUMMARY_LENGTH = 1024
_MAX_DETAILS_LENGTH = 16_384
_MAX_DESCRIPTION_LENGTH = 16_384
_MAX_REQUEST_LENGTH = 16_384
_MAX_ERROR_LENGTH = 4096
_MAX_TAG_LENGTH = 64
_MAX_TAG_COUNT = 32

_MAX_NOTE_CONTENT_LENGTH = 1024
_MAX_NOTE_SUMMARY_LENGTH = 256
_MAX_OPS_MESSAGE_LENGTH = 256
_MAX_OPS_TEXT_LENGTH = 512

_MAX_PAYLOAD_NODES = 8192
_MAX_PAYLOAD_DEPTH = 24
_MAX_PAYLOAD_TEXT_BYTES = 262_144

_ELLIPSIS = "…"


class TwinrRuntimeAutomationMixin:
    """Provide the runtime API for reminders and automations."""

    def _local_timezone_name(self) -> str | None:
        timezone_name = getattr(getattr(self, "config", None), "local_timezone_name", None)
        if not isinstance(timezone_name, str):
            return None
        timezone_name = timezone_name.strip()
        return timezone_name or None

    def _load_zoneinfo(self, timezone_name: str, *, field_name: str) -> ZoneInfo:
        if not isinstance(timezone_name, str):
            raise TypeError(f"{field_name} must be a string.")
        normalized = timezone_name.strip()
        if not normalized:
            raise ValueError(f"{field_name} must not be empty.")
        if len(normalized) > _MAX_TIMEZONE_NAME_LENGTH:
            raise ValueError(
                f"{field_name} must be <= {_MAX_TIMEZONE_NAME_LENGTH} characters."
            )
        try:
            return ZoneInfo(normalized)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Invalid timezone: {normalized!r}.") from exc

    def _normalize_timezone_name(
        self,
        value: str | None,
        *,
        field_name: str,
        fallback_to_local: bool = False,
    ) -> str | None:
        if value is None:
            return self._local_timezone_name() if fallback_to_local else None
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string or None.")
        normalized = value.strip()
        if not normalized:
            return self._local_timezone_name() if fallback_to_local else None
        self._load_zoneinfo(normalized, field_name=field_name)
        return normalized

    def _resolve_imaginary_datetime(self, value: datetime, *, field_name: str) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            return value

        if _dateutil_tz is not None:
            try:
                if not _dateutil_tz.datetime_exists(value):
                    resolved = _dateutil_tz.resolve_imaginary(value)
                    logger.warning(
                        "Resolved imaginary datetime for %s from %s to %s.",
                        field_name,
                        value.isoformat(),
                        resolved.isoformat(),
                    )
                    return resolved
            except Exception:
                logger.exception("Failed DST validity check for %s.", field_name)
                return value
            return value

        try:
            roundtrip = value.astimezone(timezone.utc).astimezone(value.tzinfo)
        except Exception:
            logger.exception("Failed datetime round-trip fallback for %s.", field_name)
            return value

        if roundtrip.replace(tzinfo=None) != value.replace(tzinfo=None):
            logger.warning(
                "Resolved imaginary datetime for %s via UTC round-trip from %s to %s.",
                field_name,
                value.isoformat(),
                roundtrip.isoformat(),
            )
            return roundtrip
        return value

    def _normalize_datetime_arg(self, value: datetime | None, *, field_name: str) -> datetime | None:
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise TypeError(f"{field_name} must be a datetime or None.")
        if value.tzinfo is not None and value.utcoffset() is not None:
            return self._resolve_imaginary_datetime(value, field_name=field_name)

        timezone_name = self._local_timezone_name()
        if timezone_name is None:
            raise ValueError(f"{field_name} must be timezone-aware when no local timezone is configured.")
        zone = self._load_zoneinfo(timezone_name, field_name="local_timezone_name")
        localized = value.replace(tzinfo=zone)
        return self._resolve_imaginary_datetime(localized, field_name=field_name)

    def _strip_controls(self, text: str, *, preserve_newlines: bool) -> str:
        if preserve_newlines:
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            return _CONTROL_CHAR_PRESERVE_NL_PATTERN.sub(" ", text)
        text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        return _CONTROL_CHAR_PATTERN.sub(" ", text)

    def _truncate_automation_text(self, text: str, *, max_length: int) -> str:
        if len(text) <= max_length:
            return text
        if max_length <= len(_ELLIPSIS):
            return _ELLIPSIS[:max_length]
        return text[: max_length - len(_ELLIPSIS)].rstrip() + _ELLIPSIS

    def _sanitize_input_text(
        self,
        value: str | None,
        *,
        field_name: str,
        max_length: int,
        blank_to_none: bool = False,
        allow_blank: bool = False,
        preserve_newlines: bool = False,
    ) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string or None.")
        text = self._strip_controls(value, preserve_newlines=preserve_newlines)
        if preserve_newlines:
            text = text.strip()
        else:
            text = self._compact_text_value(text).strip()
        if not text:
            if blank_to_none:
                return None
            if allow_blank:
                return ""
            raise ValueError(f"{field_name} must not be empty.")
        if len(text) > max_length:
            # BREAKING: oversized runtime fields are now rejected at the boundary instead of being
            # forwarded into stores/logs, preventing practical Pi-side memory and disk exhaustion.
            raise ValueError(f"{field_name} must be <= {max_length} characters.")
        return text

    def _telemetry_text(self, value: object, *, max_length: int = _MAX_OPS_TEXT_LENGTH) -> str:
        text = "" if value is None else str(value)
        text = self._strip_controls(text, preserve_newlines=False)
        text = self._compact_text_value(text).strip()
        return self._truncate_automation_text(text, max_length=max_length)

    def _normalize_automation_required_text(
        self,
        value: str,
        *,
        field_name: str,
        max_length: int,
    ) -> str:
        normalized = self._sanitize_input_text(value, field_name=field_name, max_length=max_length)
        assert normalized is not None
        return normalized

    def _normalize_source(self, value: str | None, *, default: str | None = None) -> str | None:
        normalized = self._sanitize_input_text(
            value,
            field_name="source",
            max_length=_MAX_SOURCE_LENGTH,
            blank_to_none=True,
        )
        if normalized is None:
            return default
        return normalized

    def _normalize_kind(self, value: str | None, *, default: str) -> str:
        normalized = self._sanitize_input_text(
            value,
            field_name="kind",
            max_length=_MAX_KIND_LENGTH,
            blank_to_none=True,
        )
        return normalized or default

    def _normalize_event_name(self, value: str | None) -> str | None:
        return self._sanitize_input_text(
            value,
            field_name="event_name",
            max_length=_MAX_EVENT_NAME_LENGTH,
            blank_to_none=True,
        )

    def _normalize_schedule(self, value: str) -> str:
        return self._normalize_automation_required_text(
            value,
            field_name="schedule",
            max_length=_MAX_SCHEDULE_LENGTH,
        )

    def _normalize_due_at_arg(self, value: str | datetime, *, field_name: str) -> str:
        if isinstance(value, datetime):
            normalized_dt = self._normalize_datetime_arg(value, field_name=field_name)
            assert normalized_dt is not None
            return normalized_dt.isoformat()
        return self._normalize_automation_required_text(
            value,
            field_name=field_name,
            max_length=_MAX_DUE_AT_LENGTH,
        )

    def _normalize_optional_due_at_arg(self, value: str | datetime | None, *, field_name: str) -> str | None:
        if value is None:
            return None
        return self._normalize_due_at_arg(value, field_name=field_name)

    def _normalize_time_of_day_arg(self, value: str | dt_time | None, *, field_name: str) -> str | None:
        if value is None:
            return None
        if isinstance(value, dt_time):
            if value.tzinfo is not None:
                raise ValueError(f"{field_name} must not include timezone information.")
            return value.isoformat()
        return self._normalize_automation_required_text(
            value,
            field_name=field_name,
            max_length=_MAX_TIME_OF_DAY_LENGTH,
        )

    def _normalize_weekdays(self, value: Sequence[int] | None) -> tuple[int, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes, bytearray)):
            raise TypeError("weekdays must be a sequence of integers.")

        raw = tuple(value)
        if not raw:
            return ()
        if len(raw) > 7:
            raise ValueError("weekdays must contain at most 7 entries.")

        normalized_raw: list[int] = []
        for item in raw:
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError("weekdays must contain integers only.")
            normalized_raw.append(item)

        if all(0 <= item <= 6 for item in normalized_raw):
            canonical = tuple(sorted(set(normalized_raw)))
        elif all(1 <= item <= 7 for item in normalized_raw):
            canonical = tuple(sorted({item - 1 for item in normalized_raw}))
        else:
            raise ValueError("weekdays must use either 0..6 (Mon..Sun-style) or 1..7 (ISO weekday) values.")
        return canonical

    def _normalize_tags(self, value: Sequence[str] | None) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes, bytearray)):
            raise TypeError("tags must be a sequence of strings.")

        tags: list[str] = []
        seen: set[str] = set()
        for item in value:
            normalized = self._sanitize_input_text(
                item,
                field_name="tags[]",
                max_length=_MAX_TAG_LENGTH,
                blank_to_none=True,
            )
            if normalized is None:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            tags.append(normalized)
            if len(tags) > _MAX_TAG_COUNT:
                raise ValueError(f"tags must contain at most {_MAX_TAG_COUNT} unique entries.")
        return tuple(tags)

    def _payload_repr_text_bytes(self, value: object) -> int:
        try:
            return len(repr(value).encode("utf-8", "ignore"))
        except Exception:
            return 128

    def _assert_payload_budget(self, value: Any, *, field_name: str) -> None:
        if value is None:
            return

        seen: set[int] = set()
        stack: list[tuple[object, int]] = [(value, 0)]
        nodes = 0
        text_bytes = 0

        while stack:
            current, depth = stack.pop()
            nodes += 1
            if nodes > _MAX_PAYLOAD_NODES:
                raise ValueError(
                    f"{field_name} is too large; maximum node budget is {_MAX_PAYLOAD_NODES}."
                )
            if depth > _MAX_PAYLOAD_DEPTH:
                raise ValueError(
                    f"{field_name} is too deeply nested; maximum depth is {_MAX_PAYLOAD_DEPTH}."
                )

            if isinstance(current, str):
                text_bytes += len(current.encode("utf-8", "ignore"))
            elif isinstance(current, (bytes, bytearray, memoryview)):
                text_bytes += len(current)
            else:
                text_bytes += 0

            if text_bytes > _MAX_PAYLOAD_TEXT_BYTES:
                raise ValueError(
                    f"{field_name} contains too much text/binary data; budget is {_MAX_PAYLOAD_TEXT_BYTES} bytes."
                )

            if isinstance(
                current,
                (
                    str,
                    bytes,
                    bytearray,
                    memoryview,
                    int,
                    float,
                    bool,
                    type(None),
                    datetime,
                    dt_time,
                    ZoneInfo,
                ),
            ):
                continue

            obj_id = id(current)
            if obj_id in seen:
                continue
            seen.add(obj_id)

            if is_dataclass(current):
                for data_field in dataclass_fields(current):
                    stack.append((getattr(current, data_field.name), depth + 1))
                continue

            if isinstance(current, Mapping):
                for key, item in current.items():
                    stack.append((key, depth + 1))
                    stack.append((item, depth + 1))
                continue

            if isinstance(current, (list, tuple, set, frozenset)):
                for item in current:
                    stack.append((item, depth + 1))
                continue

            try:
                state = vars(current)
            except Exception:
                text_bytes += self._payload_repr_text_bytes(current)
                if text_bytes > _MAX_PAYLOAD_TEXT_BYTES:
                    raise ValueError(
                        f"{field_name} contains too much text/binary data; budget is {_MAX_PAYLOAD_TEXT_BYTES} bytes."
                    )
                continue

            stack.append((state, depth + 1))

    def _clone_plain_data(self, value: Any) -> Any:
        if isinstance(value, (str, bytes, int, float, bool, type(None), datetime, dt_time, ZoneInfo)):
            return value
        if isinstance(value, bytearray):
            return bytearray(value)
        if isinstance(value, memoryview):
            return bytes(value)
        if isinstance(value, list):
            return [self._clone_plain_data(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._clone_plain_data(item) for item in value)
        if isinstance(value, dict):
            return {
                self._clone_plain_data(key): self._clone_plain_data(item)
                for key, item in value.items()
            }
        if isinstance(value, set):
            return {self._clone_plain_data(item) for item in value}
        if isinstance(value, frozenset):
            return frozenset(self._clone_plain_data(item) for item in value)
        raise TypeError("Unsupported plain-data type for manual clone.")

    def _snapshot_value(self, value: Any, *, field_name: str) -> Any:
        if value is None:
            return None

        self._assert_payload_budget(value, field_name=field_name)

        try:
            return self._clone_plain_data(value)
        except TypeError:
            pass

        try:
            return deepcopy(value)
        except Exception as exc:
            # BREAKING: inputs that cannot be safely snapshotted are now rejected instead of being
            # silently stored/evaluated by reference, which previously allowed caller-side mutation races.
            raise TypeError(
                f"{field_name} must be immutable plain data or safely deepcopy()-able."
            ) from exc

    def _require_non_empty_identifier(self, value: str, *, field_name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string.")
        normalized = self._compact_text_value(self._strip_controls(value, preserve_newlines=False)).strip()
        if not normalized:
            raise ValueError(f"{field_name} must not be empty.")
        return normalized

    def _require_positive_int(self, value: int, *, field_name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field_name} must be an integer.")
        if value < 1:
            raise ValueError(f"{field_name} must be >= 1.")
        return value

    def _require_non_negative_seconds(self, value: float, *, field_name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field_name} must be a real number.")
        numeric_value = float(value)
        if not math.isfinite(numeric_value) or numeric_value < 0.0:
            raise ValueError(f"{field_name} must be a finite number >= 0.")
        return numeric_value

    def _sanitize_error_text(self, error: object) -> str:
        text = "" if error is None else str(error)
        text = self._strip_controls(text, preserve_newlines=False).strip()
        text = _SECRET_JSON_PATTERN.sub(lambda match: f"{match.group(1)}[REDACTED]{match.group(3)}", text)
        text = _SECRET_KV_PATTERN.sub(lambda match: f"{match.group(1)}[REDACTED]", text)
        text = _BEARER_TOKEN_PATTERN.sub("Bearer [REDACTED]", text)
        text = _BASIC_TOKEN_PATTERN.sub("Basic [REDACTED]", text)
        text = self._compact_text_value(text)
        if not text:
            text = "unknown error"
        return self._truncate_automation_text(text, max_length=_MAX_ERROR_LENGTH)

    def _compact_text_value(self, value: object) -> str:
        text = "" if value is None else str(value)
        try:
            return compact_text(text)
        except Exception:
            return text

    def _safe_isoformat(self, value: object) -> str | None:
        if value is None:
            return None
        isoformat = getattr(value, "isoformat", None)
        if callable(isoformat):
            try:
                return isoformat()
            except Exception:
                logger.exception("Failed to format datetime-like value for ops event payload.")
        return self._telemetry_text(value, max_length=_MAX_OPS_TEXT_LENGTH)

    def _utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _format_due_label_or_fallback(self, due_at: object) -> str:
        if isinstance(due_at, datetime):
            timezone_name = self._local_timezone_name()
            if timezone_name is not None:
                try:
                    return format_due_label(due_at, timezone_name=timezone_name)
                except Exception:
                    logger.exception("Failed to format reminder due label.")
            try:
                return due_at.isoformat()
            except Exception:
                logger.exception("Failed to fallback-format reminder due timestamp.")
        return "the requested time"

    def _safe_remember_note(
        self,
        *,
        kind: str,
        content: str,
        source: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        try:
            safe_kind = self._normalize_kind(kind, default="runtime")
            safe_source = self._normalize_source(source, default="runtime") or "runtime"
            safe_content = self._truncate_automation_text(
                self._telemetry_text(content, max_length=_MAX_NOTE_CONTENT_LENGTH),
                max_length=_MAX_NOTE_CONTENT_LENGTH,
            )
            safe_metadata = None
            if metadata is not None:
                safe_metadata = self._snapshot_value(metadata, field_name="note_metadata")
            self.remember_note(
                kind=safe_kind,
                content=safe_content,
                source=safe_source,
                metadata=safe_metadata,
            )
        except Exception:
            logger.exception("Failed to persist runtime note after successful state mutation.")

    def _append_automation_ops_event(
        self,
        *,
        event: str,
        message: str,
        data: dict[str, object],
        level: str | None = None,
    ) -> None:
        kwargs: dict[str, object] = {
            "event": event if _SAFE_EVENT_NAME_PATTERN.match(event) else "runtime_event",
            "message": self._telemetry_text(message, max_length=_MAX_OPS_MESSAGE_LENGTH),
            "data": {
                "schema_version": _OPS_EVENT_SCHEMA_VERSION,
                "component": _OPS_COMPONENT,
                "recorded_at": self._utc_now_iso(),
                **(self._snapshot_value(data, field_name="ops_event_data") or {}),
            },
        }
        if level is not None:
            kwargs["level"] = level
        try:
            self.ops_events.append(**kwargs)
        except Exception:
            logger.exception("Failed to append ops event after successful state mutation: %s", event)

    def schedule_reminder(
        self,
        *,
        due_at: str | datetime,
        summary: str,
        details: str | None = None,
        kind: str = "reminder",
        source: str = "tool",
        original_request: str | None = None,
    ) -> ReminderEntry:
        """Schedule a reminder or timer and emit best-effort side effects."""

        normalized_due_at = self._normalize_due_at_arg(due_at, field_name="due_at")
        normalized_summary = self._normalize_automation_required_text(
            summary,
            field_name="summary",
            max_length=_MAX_SUMMARY_LENGTH,
        )
        normalized_details = self._sanitize_input_text(
            details,
            field_name="details",
            max_length=_MAX_DETAILS_LENGTH,
            blank_to_none=True,
            preserve_newlines=True,
        )
        normalized_kind = self._normalize_kind(kind, default="reminder")
        normalized_source = self._normalize_source(source, default="tool") or "tool"
        normalized_original_request = self._sanitize_input_text(
            original_request,
            field_name="original_request",
            max_length=_MAX_REQUEST_LENGTH,
            blank_to_none=True,
            preserve_newlines=True,
        )

        entry = self.reminder_store.schedule(
            due_at=normalized_due_at,
            summary=normalized_summary,
            details=normalized_details,
            kind=normalized_kind,
            source=normalized_source,
            original_request=normalized_original_request,
        )

        due_label = self._format_due_label_or_fallback(getattr(entry, "due_at", None))
        entry_summary = self._telemetry_text(
            getattr(entry, "summary", normalized_summary),
            max_length=_MAX_NOTE_SUMMARY_LENGTH,
        )
        self._safe_remember_note(
            kind="reminder",
            content=f"Reminder scheduled for {due_label}: {entry_summary}",
            source=normalized_source,
            metadata={
                "reminder_id": getattr(entry, "reminder_id", None),
                "reminder_kind": getattr(entry, "kind", normalized_kind),
            },
        )
        self._append_automation_ops_event(
            event="reminder_scheduled",
            message="A reminder or timer was scheduled.",
            data={
                "source": normalized_source,
                "reminder_id": getattr(entry, "reminder_id", None),
                "kind": getattr(entry, "kind", normalized_kind),
                "due_at": self._safe_isoformat(getattr(entry, "due_at", None)),
                "summary": self._telemetry_text(
                    getattr(entry, "summary", normalized_summary),
                    max_length=_MAX_OPS_TEXT_LENGTH,
                ),
            },
        )
        return entry

    def list_automation_records(self, *, now: datetime | None = None) -> tuple[dict[str, object], ...]:
        """Return automation records in the tool-facing listing format."""

        return self.automation_store.list_tool_records(
            now=self._normalize_datetime_arg(now, field_name="now")
        )

    def create_time_automation(
        self,
        *,
        name: str,
        actions,
        description: str | None = None,
        enabled: bool = True,
        schedule: str = "once",
        due_at: str | datetime | None = None,
        time_of_day: str | dt_time | None = None,
        weekdays: tuple[int, ...] | list[int] = (),
        timezone_name: str | None = None,
        source: str = "tool",
        tags: tuple[str, ...] | list[str] = (),
    ) -> AutomationDefinition:
        """Create a time-based automation and record the side effects."""

        normalized_name = self._normalize_automation_required_text(
            name,
            field_name="name",
            max_length=_MAX_NAME_LENGTH,
        )
        normalized_description = self._sanitize_input_text(
            description,
            field_name="description",
            max_length=_MAX_DESCRIPTION_LENGTH,
            blank_to_none=True,
            preserve_newlines=True,
        )
        normalized_schedule = self._normalize_schedule(schedule)
        normalized_due_at = self._normalize_optional_due_at_arg(due_at, field_name="due_at")
        normalized_time_of_day = self._normalize_time_of_day_arg(time_of_day, field_name="time_of_day")
        normalized_weekdays = self._normalize_weekdays(weekdays)
        normalized_timezone_name = self._normalize_timezone_name(
            timezone_name,
            field_name="timezone_name",
            fallback_to_local=True,
        )
        normalized_source = self._normalize_source(source, default="tool") or "tool"
        normalized_tags = self._normalize_tags(tags)
        normalized_actions = self._snapshot_value(actions, field_name="actions")

        entry = self.automation_store.create_time_automation(
            name=normalized_name,
            description=normalized_description,
            enabled=enabled,
            schedule=normalized_schedule,
            due_at=normalized_due_at,
            time_of_day=normalized_time_of_day,
            weekdays=normalized_weekdays,
            timezone_name=normalized_timezone_name,
            actions=normalized_actions,
            source=normalized_source,
            tags=normalized_tags,
        )
        trigger = getattr(entry, "trigger", None)
        self._safe_remember_note(
            kind="automation",
            content=f"Time automation created: {self._telemetry_text(getattr(entry, 'name', normalized_name), max_length=_MAX_NOTE_SUMMARY_LENGTH)}",
            source=normalized_source,
            metadata={
                "automation_id": getattr(entry, "automation_id", None),
                "trigger_kind": getattr(trigger, "kind", None),
            },
        )
        self._append_automation_ops_event(
            event="automation_created",
            message="A time-based automation was created.",
            data={
                "source": normalized_source,
                "automation_id": getattr(entry, "automation_id", None),
                "name": self._telemetry_text(getattr(entry, "name", normalized_name)),
                "schedule": getattr(trigger, "schedule", getattr(trigger, "kind", None)),
                "timezone_name": normalized_timezone_name,
            },
        )
        return entry

    def create_if_then_automation(
        self,
        *,
        name: str,
        actions,
        description: str | None = None,
        enabled: bool = True,
        event_name: str | None = None,
        all_conditions: tuple[AutomationCondition, ...] | list[AutomationCondition] = (),
        any_conditions: tuple[AutomationCondition, ...] | list[AutomationCondition] = (),
        cooldown_seconds: float = 0.0,
        source: str = "tool",
        tags: tuple[str, ...] | list[str] = (),
    ) -> AutomationDefinition:
        """Create an event-driven automation and record the side effects."""

        normalized_name = self._normalize_automation_required_text(
            name,
            field_name="name",
            max_length=_MAX_NAME_LENGTH,
        )
        normalized_description = self._sanitize_input_text(
            description,
            field_name="description",
            max_length=_MAX_DESCRIPTION_LENGTH,
            blank_to_none=True,
            preserve_newlines=True,
        )
        normalized_event_name = self._normalize_event_name(event_name)
        cooldown_seconds = self._require_non_negative_seconds(
            cooldown_seconds,
            field_name="cooldown_seconds",
        )
        normalized_source = self._normalize_source(source, default="tool") or "tool"
        normalized_tags = self._normalize_tags(tags)

        entry = self.automation_store.create_if_then_automation(
            name=normalized_name,
            description=normalized_description,
            enabled=enabled,
            event_name=normalized_event_name,
            all_conditions=self._snapshot_value(all_conditions, field_name="all_conditions"),
            any_conditions=self._snapshot_value(any_conditions, field_name="any_conditions"),
            cooldown_seconds=cooldown_seconds,
            actions=self._snapshot_value(actions, field_name="actions"),
            source=normalized_source,
            tags=normalized_tags,
        )
        trigger = getattr(entry, "trigger", None)
        self._safe_remember_note(
            kind="automation",
            content=f"Sensor automation created: {self._telemetry_text(getattr(entry, 'name', normalized_name), max_length=_MAX_NOTE_SUMMARY_LENGTH)}",
            source=normalized_source,
            metadata={
                "automation_id": getattr(entry, "automation_id", None),
                "trigger_kind": getattr(trigger, "kind", None),
            },
        )
        self._append_automation_ops_event(
            event="automation_created",
            message="An if-then automation was created.",
            data={
                "source": normalized_source,
                "automation_id": getattr(entry, "automation_id", None),
                "name": self._telemetry_text(getattr(entry, "name", normalized_name)),
                "event_name": self._telemetry_text(
                    getattr(trigger, "event_name", normalized_event_name),
                    max_length=_MAX_OPS_TEXT_LENGTH,
                ),
                "cooldown_seconds": cooldown_seconds,
            },
        )
        return entry

    def update_automation(
        self,
        automation_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        enabled: bool | None = None,
        trigger=None,
        actions=None,
        source: str | None = None,
        tags: tuple[str, ...] | list[str] | None = None,
    ) -> AutomationDefinition:
        """Update an automation definition and record the side effects."""

        automation_id = self._require_non_empty_identifier(automation_id, field_name="automation_id")
        normalized_name = None
        if name is not None:
            normalized_name = self._normalize_automation_required_text(
                name,
                field_name="name",
                max_length=_MAX_NAME_LENGTH,
            )
        normalized_description = self._sanitize_input_text(
            description,
            field_name="description",
            max_length=_MAX_DESCRIPTION_LENGTH,
            allow_blank=True,
            preserve_newlines=True,
        )
        normalized_source = self._normalize_source(source, default=None)
        normalized_tags = None if tags is None else self._normalize_tags(tags)

        entry = self.automation_store.update(
            automation_id,
            name=normalized_name,
            description=normalized_description,
            enabled=enabled,
            trigger=self._snapshot_value(trigger, field_name="trigger") if trigger is not None else None,
            actions=self._snapshot_value(actions, field_name="actions") if actions is not None else None,
            source=normalized_source,
            tags=normalized_tags,
        )
        trigger_value = getattr(entry, "trigger", None)
        self._safe_remember_note(
            kind="automation",
            content=f"Automation updated: {self._telemetry_text(getattr(entry, 'name', normalized_name) or automation_id, max_length=_MAX_NOTE_SUMMARY_LENGTH)}",
            source=normalized_source or "automation_update",
            metadata={
                "automation_id": getattr(entry, "automation_id", automation_id),
                "trigger_kind": getattr(trigger_value, "kind", None),
            },
        )
        self._append_automation_ops_event(
            event="automation_updated",
            message="An automation was updated.",
            data={
                "source": normalized_source or "automation_update",
                "automation_id": getattr(entry, "automation_id", automation_id),
                "name": self._telemetry_text(getattr(entry, "name", normalized_name) or automation_id),
            },
        )
        return entry

    def delete_automation(self, automation_id: str, *, source: str = "tool") -> AutomationDefinition:
        """Delete an automation and record the side effects."""

        automation_id = self._require_non_empty_identifier(automation_id, field_name="automation_id")
        normalized_source = self._normalize_source(source, default="tool") or "tool"
        entry = self.automation_store.delete(automation_id)
        self._safe_remember_note(
            kind="automation",
            content=f"Automation deleted: {self._telemetry_text(getattr(entry, 'name', automation_id), max_length=_MAX_NOTE_SUMMARY_LENGTH)}",
            source=normalized_source,
            metadata={"automation_id": getattr(entry, "automation_id", automation_id)},
        )
        self._append_automation_ops_event(
            event="automation_deleted",
            message="An automation was deleted.",
            data={
                "source": normalized_source,
                "automation_id": getattr(entry, "automation_id", automation_id),
                "name": self._telemetry_text(getattr(entry, "name", automation_id)),
            },
        )
        return entry

    def due_time_automations(self, *, now: datetime | None = None) -> tuple[AutomationDefinition, ...]:
        """Return time automations that are due at the given moment."""

        return self.automation_store.due_time_automations(
            now=self._normalize_datetime_arg(now, field_name="now")
        )

    def due_time_matches(self, *, now: datetime | None = None) -> tuple[TimeAutomationMatch, ...]:
        """Return due time automations together with the scheduled fire time."""

        return self.automation_store.due_time_matches(
            now=self._normalize_datetime_arg(now, field_name="now")
        )

    def matching_if_then_automations(
        self,
        *,
        facts: dict[str, object],
        event_name: str | None = None,
        now: datetime | None = None,
    ) -> tuple[AutomationDefinition, ...]:
        """Return event automations matching the supplied facts."""

        return self.automation_store.matching_if_then_automations(
            facts=self._snapshot_value(facts, field_name="facts"),
            event_name=self._normalize_event_name(event_name),
            now=self._normalize_datetime_arg(now, field_name="now"),
        )

    def mark_automation_triggered(
        self,
        automation_id: str,
        *,
        triggered_at: datetime | None = None,
        scheduled_for_at: datetime | None = None,
        source: str = "automation_execution",
    ) -> AutomationDefinition:
        """Mark an automation as triggered and record the side effects."""

        automation_id = self._require_non_empty_identifier(automation_id, field_name="automation_id")
        normalized_source = self._normalize_source(source, default="automation_execution") or "automation_execution"
        entry = self.automation_store.mark_triggered(
            automation_id,
            triggered_at=self._normalize_datetime_arg(triggered_at, field_name="triggered_at"),
            scheduled_for_at=self._normalize_datetime_arg(scheduled_for_at, field_name="scheduled_for_at"),
        )
        self._safe_remember_note(
            kind="automation",
            content=f"Automation ran: {self._telemetry_text(getattr(entry, 'name', automation_id), max_length=_MAX_NOTE_SUMMARY_LENGTH)}",
            source=normalized_source,
            metadata={"automation_id": getattr(entry, "automation_id", automation_id)},
        )
        self._append_automation_ops_event(
            event="automation_triggered",
            message="An automation was executed.",
            data={
                "source": normalized_source,
                "automation_id": getattr(entry, "automation_id", automation_id),
                "name": self._telemetry_text(getattr(entry, "name", automation_id)),
                "triggered_at": self._safe_isoformat(getattr(entry, "last_triggered_at", None)),
                "scheduled_for_at": self._safe_isoformat(getattr(entry, "last_scheduled_at", None)),
            },
        )
        return entry

    def reserve_due_reminders(self, *, limit: int = 1) -> tuple[ReminderEntry, ...]:
        """Reserve due reminders for background delivery."""

        return self.reminder_store.reserve_due(limit=self._require_positive_int(limit, field_name="limit"))

    def peek_due_reminders(self, *, limit: int = 1) -> tuple[ReminderEntry, ...]:
        """Peek at due reminders without reserving them."""

        return self.reminder_store.peek_due(limit=self._require_positive_int(limit, field_name="limit"))

    def mark_reminder_delivered(self, reminder_id: str) -> ReminderEntry:
        """Mark a reserved reminder as delivered."""

        reminder_id = self._require_non_empty_identifier(reminder_id, field_name="reminder_id")
        entry = self.reminder_store.mark_delivered(reminder_id)
        self._safe_remember_note(
            kind="reminder",
            content=f"Reminder delivered: {self._telemetry_text(getattr(entry, 'summary', reminder_id), max_length=_MAX_NOTE_SUMMARY_LENGTH)}",
            source="reminder_delivery",
            metadata={"reminder_id": getattr(entry, "reminder_id", reminder_id)},
        )
        self._append_automation_ops_event(
            event="reminder_delivered",
            message="A due reminder was delivered successfully.",
            data={
                "source": "reminder_delivery",
                "reminder_id": getattr(entry, "reminder_id", reminder_id),
                "summary": self._telemetry_text(getattr(entry, "summary", reminder_id)),
            },
        )
        return entry

    def release_reminder_reservation(self, reminder_id: str) -> ReminderEntry:
        """Release a reserved reminder without marking it delivered or failed."""

        reminder_id = self._require_non_empty_identifier(reminder_id, field_name="reminder_id")
        entry = self.reminder_store.release_reservation(reminder_id)
        self._append_automation_ops_event(
            event="reminder_delivery_released",
            message="A reserved reminder was released before delivery started.",
            data={
                "source": "reminder_delivery",
                "reminder_id": getattr(entry, "reminder_id", reminder_id),
                "summary": self._telemetry_text(getattr(entry, "summary", reminder_id)),
            },
        )
        return entry

    def mark_reminder_failed(self, reminder_id: str, *, error: str) -> ReminderEntry:
        """Mark a reserved reminder as failed with sanitized error text."""

        reminder_id = self._require_non_empty_identifier(reminder_id, field_name="reminder_id")
        sanitized_error = self._sanitize_error_text(error)
        entry = self.reminder_store.mark_failed(reminder_id, error=sanitized_error)
        self._append_automation_ops_event(
            event="reminder_delivery_failed",
            level="error",
            message="A due reminder could not be delivered.",
            data={
                "source": "reminder_delivery",
                "reminder_id": getattr(entry, "reminder_id", reminder_id),
                "summary": self._telemetry_text(getattr(entry, "summary", reminder_id)),
                "error": sanitized_error,
            },
        )
        return entry
