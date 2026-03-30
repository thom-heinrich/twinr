# CHANGELOG: 2026-03-30
# BUG-1: Reject non-positive "days" values instead of silently coercing them into a 1-day window.
# BUG-2: Bound event iteration, filter results back to the requested window, sort defensively, deduplicate duplicates, and infer all-day events so outputs stay correct even with imperfect providers.
# SEC-1: Treat calendar payloads as untrusted external input by sanitizing/clipping nested text and container sizes to reduce prompt-injection/context-bloat risk from malicious invites.
# IMP-1: Add bounded transient retry/backoff, richer structured diagnostics, timezone/window metadata, and voice-safe agenda lines for downstream consumers.
# IMP-2: Add frontier request support for read_on_date/read_range plus parameter aliases (days/lookahead_days/limit/max_events/timezone/date/start/end/start_at/end_at).

"""Adapt read-only calendar sources to Twinr integration requests.

This module translates supported calendar operations into bounded
``IntegrationResult`` payloads while centralizing request validation,
timezone resolution, and user-facing agenda summaries.
"""

from __future__ import annotations

import itertools
import logging
import os
import random
import time as time_module
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from twinr.integrations.adapter import IntegrationAdapter
from twinr.integrations.calendar.models import CalendarEvent
from twinr.integrations.models import IntegrationManifest, IntegrationRequest, IntegrationResult


LOGGER = logging.getLogger(__name__)
_INVALID = object()
_MAX_SORT_DATETIME = datetime(9999, 12, 31, 23, 59, 59, tzinfo=UTC)
_DESCRIPTION_FIELDS = frozenset({"description", "details", "notes", "body", "content", "html"})
_TITLE_FIELDS = frozenset({"title", "summary", "subject", "name"})
_LOCATION_FIELDS = frozenset({"location", "where", "place"})
_ID_FIELDS = frozenset({"id", "event_id", "uid", "ical_uid", "calendar_id", "etag"})


def _default_clock() -> datetime:
    """Return the current local wall-clock time as an aware datetime."""

    return datetime.now().astimezone()


def _is_aware_datetime(value: datetime) -> bool:
    """Return True when ``value`` carries a usable timezone offset."""

    return value.tzinfo is not None and value.tzinfo.utcoffset(value) is not None


def _timezone_label(value: tzinfo | None) -> str:
    """Return a stable human-readable name for a timezone."""

    if value is None:
        return "UTC"
    return getattr(value, "key", None) or str(value)


@runtime_checkable
class CalendarReader(Protocol):
    """Describe the read-only calendar source contract used by the adapter.

    Implementations return ``CalendarEvent`` objects that overlap a requested
    time window. Results are expected in start-time order so the adapter can
    truncate them safely for voice and UI consumers.
    """

    def list_events(self, *, start_at: datetime, end_at: datetime, limit: int) -> list[CalendarEvent]:
        """List events overlapping a bounded time window.

        Args:
            start_at: Inclusive window start.
            end_at: Exclusive window end.
            limit: Maximum number of events to return.

        Returns:
            A list of overlapping calendar events ordered by start time.
        """

        ...


@dataclass(frozen=True, slots=True)
class CalendarAdapterSettings:
    """Hold validated limits and timezone overrides for calendar reads.

    Attributes:
        max_events: Maximum number of events exposed in one adapter response.
        default_upcoming_days: Default look-ahead span for upcoming requests.
        max_upcoming_days: Hard cap applied to requested upcoming-day windows.
        timezone_name: Optional IANA timezone name overriding host defaults.
        provider_overfetch: Extra events requested beyond the user-visible limit
            so malformed/duplicate early items do not hide later valid events.
        reader_scan_floor: Minimum number of provider events scanned even for
            tiny user-visible limits such as "next event".
        max_reader_scan_events: Hard cap on provider events scanned for one
            request to protect Raspberry Pi memory/CPU.
        reader_retry_attempts: Number of attempts for transient reader errors.
        reader_retry_base_delay_seconds: Base delay for retry backoff.
        reader_retry_max_delay_seconds: Maximum delay per retry.
        max_event_keys: Maximum mapping keys retained per event payload level.
        max_nested_items: Maximum list-like items retained per payload level.
        max_payload_depth: Maximum recursive payload depth retained.
        max_event_field_chars: Maximum characters retained for general text
            fields.
        max_event_title_chars: Maximum characters retained for title-like text.
        max_event_description_chars: Maximum characters retained for
            description-like text.
        text_scan_multiplier: Caps how much raw text is scanned before
            truncation to avoid spending CPU on giant attacker-controlled blobs.
    """

    max_events: int = 12
    default_upcoming_days: int = 7
    max_upcoming_days: int = 30
    timezone_name: str | None = None
    provider_overfetch: int = 4
    reader_scan_floor: int = 12
    max_reader_scan_events: int = 48
    reader_retry_attempts: int = 2
    reader_retry_base_delay_seconds: float = 0.20
    reader_retry_max_delay_seconds: float = 1.00
    max_event_keys: int = 24
    max_nested_items: int = 16
    max_payload_depth: int = 4
    max_event_field_chars: int = 256
    max_event_title_chars: int = 160
    max_event_description_chars: int = 1200
    text_scan_multiplier: int = 4

    def __post_init__(self) -> None:
        normalized_timezone_name = self.timezone_name.strip() if self.timezone_name is not None else None
        object.__setattr__(self, "timezone_name", normalized_timezone_name or None)

        if normalized_timezone_name:
            try:
                ZoneInfo(normalized_timezone_name)
            except ZoneInfoNotFoundError as exc:
                raise ValueError(f"Unsupported timezone_name: {normalized_timezone_name}") from exc

        if self.max_events < 1:
            raise ValueError("max_events must be >= 1")
        if self.default_upcoming_days < 1:
            raise ValueError("default_upcoming_days must be >= 1")
        if self.max_upcoming_days < 1:
            raise ValueError("max_upcoming_days must be >= 1")
        if self.provider_overfetch < 0:
            raise ValueError("provider_overfetch must be >= 0")
        if self.reader_scan_floor < 1:
            raise ValueError("reader_scan_floor must be >= 1")
        if self.max_reader_scan_events < self.reader_scan_floor:
            raise ValueError("max_reader_scan_events must be >= reader_scan_floor")
        if self.reader_retry_attempts < 1:
            raise ValueError("reader_retry_attempts must be >= 1")
        if self.reader_retry_base_delay_seconds < 0:
            raise ValueError("reader_retry_base_delay_seconds must be >= 0")
        if self.reader_retry_max_delay_seconds < 0:
            raise ValueError("reader_retry_max_delay_seconds must be >= 0")
        if self.max_event_keys < 1:
            raise ValueError("max_event_keys must be >= 1")
        if self.max_nested_items < 1:
            raise ValueError("max_nested_items must be >= 1")
        if self.max_payload_depth < 1:
            raise ValueError("max_payload_depth must be >= 1")
        if self.max_event_field_chars < 16:
            raise ValueError("max_event_field_chars must be >= 16")
        if self.max_event_title_chars < 16:
            raise ValueError("max_event_title_chars must be >= 16")
        if self.max_event_description_chars < self.max_event_field_chars:
            raise ValueError("max_event_description_chars must be >= max_event_field_chars")
        if self.text_scan_multiplier < 1:
            raise ValueError("text_scan_multiplier must be >= 1")


@dataclass(frozen=True, slots=True)
class NormalizedCalendarRequest:
    """Represent a validated calendar request."""

    operation_id: str
    limit: int | None = None
    days: int | None = None
    timezone_name: str | None = None
    target_date: date | None = None
    start_at: datetime | None = None
    end_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class PreparedCalendarEvent:
    """Hold a sanitized event and metadata used for sorting and reporting."""

    payload: dict[str, Any]
    sort_key: datetime
    dedupe_key: str
    agenda_line: str
    text_was_truncated: bool
    event_start_at: datetime | None
    event_end_at: datetime | None


@dataclass(slots=True)
class ReadOnlyCalendarAdapter(IntegrationAdapter):
    """Serve agenda-style calendar reads through the integration interface."""

    manifest: IntegrationManifest
    calendar_reader: CalendarReader
    settings: CalendarAdapterSettings = field(default_factory=CalendarAdapterSettings)
    clock: Callable[[], datetime] = _default_clock

    def __post_init__(self) -> None:
        if not isinstance(self.calendar_reader, CalendarReader):
            raise TypeError("calendar_reader must implement CalendarReader")
        if not callable(self.clock):
            raise TypeError("clock must be callable")

    def execute(self, request: IntegrationRequest) -> IntegrationResult:
        """Execute a supported calendar read operation."""

        try:
            normalized_request = self._normalize_request(request)
            if isinstance(normalized_request, IntegrationResult):
                return normalized_request

            now = self._get_now(timezone_name=normalized_request.timezone_name)

            if normalized_request.operation_id == "read_today":
                return self._read_today(now, limit=normalized_request.limit)

            if normalized_request.operation_id == "read_upcoming":
                return self._read_upcoming(
                    now,
                    days=normalized_request.days or self.settings.default_upcoming_days,
                    limit=normalized_request.limit,
                    summary_kind="upcoming",
                    summary_label="upcoming events",
                )

            if normalized_request.operation_id == "read_next_event":
                return self._read_upcoming(
                    now,
                    days=normalized_request.days or self.settings.default_upcoming_days,
                    limit=1,
                    summary_kind="next_event",
                    summary_label="next event",
                )

            if normalized_request.operation_id == "read_on_date":
                if normalized_request.target_date is None:
                    return self._failure_result(
                        summary="I couldn't understand which day to read.",
                        code="missing_date_parameter",
                        validation_errors=[{"field": "date", "message": "A date is required for read_on_date."}],
                    )
                return self._read_on_date(
                    target_date=normalized_request.target_date,
                    timezone_name=normalized_request.timezone_name,
                    limit=normalized_request.limit,
                )

            if normalized_request.operation_id == "read_range":
                if normalized_request.start_at is None or normalized_request.end_at is None:
                    return self._failure_result(
                        summary="I couldn't understand the calendar time range.",
                        code="missing_time_range",
                        validation_errors=[
                            {"field": "start_at", "message": "A start time is required for read_range."},
                            {"field": "end_at", "message": "An end time is required for read_range."},
                        ],
                    )
                return self._read_range(
                    start_at=normalized_request.start_at,
                    end_at=normalized_request.end_at,
                    limit=normalized_request.limit,
                )

            return self._failure_result(
                summary="I couldn't understand the calendar request.",
                code="unsupported_operation",
            )
        except Exception:
            LOGGER.exception("Calendar adapter execution failed unexpectedly.")
            return self._failure_result(
                summary="I couldn't read the calendar right now.",
                code="calendar_execute_failed",
            )

    def _normalize_request(self, request: IntegrationRequest) -> NormalizedCalendarRequest | IntegrationResult:
        """Validate and normalize an incoming integration request."""

        raw_operation_id = (
            getattr(request, "operation_id", None)
            or getattr(request, "operation", None)
            or getattr(request, "action", None)
        )
        normalized_operation_id = self._normalize_operation_id(raw_operation_id)
        if normalized_operation_id is None:
            return self._failure_result(
                summary="I couldn't understand the calendar request.",
                code="unsupported_operation",
                validation_errors=[
                    {
                        "field": "operation_id",
                        "message": "Supported operations are read_today, read_upcoming, read_next_event, read_on_date, and read_range.",
                    }
                ],
            )

        parameters_obj = getattr(request, "parameters", None)
        if parameters_obj is None:
            parameters: Mapping[str, Any] = {}
        elif isinstance(parameters_obj, Mapping):
            parameters = parameters_obj
        else:
            LOGGER.warning(
                "Calendar request parameters were not a mapping: %s",
                type(parameters_obj).__name__,
            )
            return self._failure_result(
                summary="I couldn't understand the calendar request parameters.",
                code="invalid_request_parameters",
                validation_errors=[
                    {
                        "field": "parameters",
                        "message": "Request parameters must be a mapping object.",
                    }
                ],
            )

        timezone_name = self._parse_timezone_name(
            self._pick_parameter(parameters, "timezone_name", "timezone", "tz")
        )
        if timezone_name is _INVALID:
            return self._failure_result(
                summary="I couldn't understand the requested timezone.",
                code="invalid_timezone_parameter",
                validation_errors=[
                    {
                        "field": "timezone",
                        "message": "Timezone must be a valid IANA timezone name such as Europe/Berlin.",
                    }
                ],
            )

        limit = self._parse_positive_int(
            self._pick_parameter(parameters, "limit", "max_events"),
            allow_missing=True,
        )
        if limit is _INVALID:
            return self._failure_result(
                summary="I couldn't understand how many events to return.",
                code="invalid_limit_parameter",
                validation_errors=[
                    {
                        "field": "limit",
                        "message": "limit must be a positive integer.",
                    }
                ],
            )

        if normalized_operation_id == "read_upcoming":
            # BREAKING: Non-positive "days" values now fail fast instead of being silently coerced into a 1-day window.
            raw_days = self._pick_parameter(parameters, "days", "lookahead_days", "window_days")
            parsed_days = self._parse_positive_int(raw_days, allow_missing=True)
            if parsed_days is _INVALID:
                return self._failure_result(
                    summary="I couldn't understand how many days to look ahead.",
                    code="invalid_days_parameter",
                    validation_errors=[
                        {
                            "field": "days",
                            "message": "days must be a positive integer.",
                        }
                    ],
                )

            return NormalizedCalendarRequest(
                operation_id=normalized_operation_id,
                limit=limit,
                days=parsed_days or self.settings.default_upcoming_days,
                timezone_name=timezone_name,
            )

        if normalized_operation_id == "read_next_event":
            raw_days = self._pick_parameter(parameters, "days", "lookahead_days", "window_days")
            parsed_days = self._parse_positive_int(raw_days, allow_missing=True)
            if parsed_days is _INVALID:
                return self._failure_result(
                    summary="I couldn't understand how many days to look ahead.",
                    code="invalid_days_parameter",
                    validation_errors=[
                        {
                            "field": "days",
                            "message": "days must be a positive integer.",
                        }
                    ],
                )

            return NormalizedCalendarRequest(
                operation_id=normalized_operation_id,
                limit=limit,
                days=parsed_days or self.settings.default_upcoming_days,
                timezone_name=timezone_name,
            )

        if normalized_operation_id == "read_on_date":
            target_timezone = self._resolve_timezone(override_timezone_name=timezone_name)
            parsed_date = self._parse_date_parameter(
                self._pick_parameter(parameters, "date", "day", "on"),
                fallback_timezone=target_timezone,
            )
            if parsed_date is _INVALID:
                return self._failure_result(
                    summary="I couldn't understand which day to read.",
                    code="invalid_date_parameter",
                    validation_errors=[
                        {
                            "field": "date",
                            "message": "date must be an ISO date like 2026-03-30 or an ISO datetime.",
                        }
                    ],
                )

            return NormalizedCalendarRequest(
                operation_id=normalized_operation_id,
                limit=limit,
                timezone_name=timezone_name,
                target_date=parsed_date,
            )

        if normalized_operation_id == "read_range":
            target_timezone = self._resolve_timezone(override_timezone_name=timezone_name)
            start_at = self._parse_datetime_parameter(
                self._pick_parameter(parameters, "start_at", "start"),
                fallback_timezone=target_timezone,
            )
            end_at = self._parse_datetime_parameter(
                self._pick_parameter(parameters, "end_at", "end"),
                fallback_timezone=target_timezone,
            )
            if start_at is _INVALID or end_at is _INVALID:
                return self._failure_result(
                    summary="I couldn't understand the calendar time range.",
                    code="invalid_time_range",
                    validation_errors=[
                        {
                            "field": "start_at",
                            "message": "start_at/start must be an ISO datetime or date.",
                        },
                        {
                            "field": "end_at",
                            "message": "end_at/end must be an ISO datetime or date.",
                        },
                    ],
                )
            if start_at is None or end_at is None:
                return self._failure_result(
                    summary="I couldn't understand the calendar time range.",
                    code="missing_time_range",
                    validation_errors=[
                        {
                            "field": "start_at",
                            "message": "Both start_at and end_at are required for read_range.",
                        },
                        {
                            "field": "end_at",
                            "message": "Both start_at and end_at are required for read_range.",
                        },
                    ],
                )
            if end_at <= start_at:
                return self._failure_result(
                    summary="The requested calendar time range is empty.",
                    code="empty_time_range",
                    validation_errors=[
                        {
                            "field": "end_at",
                            "message": "end_at must be later than start_at.",
                        }
                    ],
                )

            return NormalizedCalendarRequest(
                operation_id=normalized_operation_id,
                limit=limit,
                timezone_name=timezone_name,
                start_at=start_at,
                end_at=end_at,
            )

        return NormalizedCalendarRequest(
            operation_id=normalized_operation_id,
            limit=limit,
            timezone_name=timezone_name,
        )

    def _normalize_operation_id(self, raw_operation_id: Any) -> str | None:
        """Map operation aliases into canonical operation identifiers."""

        if raw_operation_id is None:
            return None

        operation_key = str(raw_operation_id).strip().lower()
        if not operation_key:
            return None

        operation_aliases = {
            "read_today": "read_today",
            "today": "read_today",
            "agenda_today": "read_today",
            "read_upcoming": "read_upcoming",
            "upcoming": "read_upcoming",
            "read_next_event": "read_next_event",
            "next_event": "read_next_event",
            "next": "read_next_event",
            "read_on_date": "read_on_date",
            "read_date": "read_on_date",
            "on_date": "read_on_date",
            "read_range": "read_range",
            "read_window": "read_range",
            "range": "read_range",
        }
        return operation_aliases.get(operation_key)

    def _pick_parameter(self, parameters: Mapping[str, Any], *names: str) -> Any | None:
        """Return the first matching parameter value from ``parameters``."""

        for name in names:
            if name in parameters:
                return parameters[name]
        return None

    def _parse_positive_int(self, raw_value: Any, *, allow_missing: bool) -> int | None | object:
        """Parse a positive integer without silently accepting booleans/zero."""

        if raw_value in (None, ""):
            return None if allow_missing else _INVALID
        if isinstance(raw_value, bool):
            return _INVALID
        if isinstance(raw_value, int):
            return raw_value if raw_value > 0 else _INVALID
        if isinstance(raw_value, float):
            if not raw_value.is_integer():
                return _INVALID
            integer_value = int(raw_value)
            return integer_value if integer_value > 0 else _INVALID
        if isinstance(raw_value, str):
            candidate = raw_value.strip()
            if not candidate:
                return None if allow_missing else _INVALID
            if candidate.startswith("+"):
                candidate = candidate[1:]
            if not candidate.isdigit():
                return _INVALID
            integer_value = int(candidate)
            return integer_value if integer_value > 0 else _INVALID

        try:
            integer_value = int(raw_value)
        except (TypeError, ValueError):
            return _INVALID

        return integer_value if integer_value > 0 else _INVALID

    def _parse_timezone_name(self, raw_value: Any) -> str | None | object:
        """Parse an optional IANA timezone name."""

        if raw_value in (None, ""):
            return None
        if not isinstance(raw_value, str):
            return _INVALID

        timezone_name = raw_value.strip()
        if not timezone_name:
            return None

        try:
            ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            LOGGER.warning("Unsupported request timezone name: %s", timezone_name)
            return _INVALID

        return timezone_name

    def _parse_date_parameter(self, raw_value: Any, *, fallback_timezone: tzinfo) -> date | None | object:
        """Parse a date or ISO datetime into a local calendar date."""

        if raw_value in (None, ""):
            return None
        if isinstance(raw_value, date) and not isinstance(raw_value, datetime):
            return raw_value

        parsed_datetime = self._coerce_datetime(raw_value, fallback_timezone=fallback_timezone)
        if parsed_datetime is not None:
            return parsed_datetime.date()

        if isinstance(raw_value, str):
            candidate = raw_value.strip()
            if not candidate:
                return None
            try:
                return date.fromisoformat(candidate)
            except ValueError:
                return _INVALID

        return _INVALID

    def _parse_datetime_parameter(self, raw_value: Any, *, fallback_timezone: tzinfo) -> datetime | None | object:
        """Parse a request datetime/date into an aware datetime."""

        if raw_value in (None, ""):
            return None
        parsed_datetime = self._coerce_datetime(raw_value, fallback_timezone=fallback_timezone)
        if parsed_datetime is None:
            return _INVALID
        return parsed_datetime

    def _read_today(self, now: datetime, *, limit: int | None = None) -> IntegrationResult:
        """Build a result for the current local calendar day."""

        start_at = datetime.combine(now.date(), time.min, tzinfo=now.tzinfo)
        end_at = start_at + timedelta(days=1)
        return self._build_result(
            start_at=start_at,
            end_at=end_at,
            limit=self._normalize_limit(limit),
            summary_kind="today",
            summary_label="today's agenda",
        )

    def _read_on_date(
        self,
        *,
        target_date: date,
        timezone_name: str | None,
        limit: int | None = None,
    ) -> IntegrationResult:
        """Build a result for one explicit local calendar day."""

        resolved_timezone = self._resolve_timezone(override_timezone_name=timezone_name)
        start_at = datetime.combine(target_date, time.min, tzinfo=resolved_timezone)
        end_at = start_at + timedelta(days=1)
        return self._build_result(
            start_at=start_at,
            end_at=end_at,
            limit=self._normalize_limit(limit),
            summary_kind="date",
            summary_label=target_date.isoformat(),
        )

    def _read_range(
        self,
        *,
        start_at: datetime,
        end_at: datetime,
        limit: int | None = None,
    ) -> IntegrationResult:
        """Build a result for an explicit datetime range."""

        return self._build_result(
            start_at=start_at,
            end_at=end_at,
            limit=self._normalize_limit(limit),
            summary_kind="range",
            summary_label="selected time window",
        )

    def _read_upcoming(
        self,
        now: datetime,
        *,
        days: int,
        limit: int | None = None,
        summary_kind: str = "upcoming",
        summary_label: str = "upcoming events",
    ) -> IntegrationResult:
        """Build a result for a bounded upcoming-events window."""

        bounded_days = max(1, min(days, self.settings.max_upcoming_days))
        return self._build_result(
            start_at=now,
            end_at=now + timedelta(days=bounded_days),
            limit=self._normalize_limit(limit),
            summary_kind=summary_kind,
            summary_label=summary_label,
        )

    def _build_result(
        self,
        *,
        start_at: datetime,
        end_at: datetime,
        limit: int,
        summary_kind: str,
        summary_label: str,
    ) -> IntegrationResult:
        """Read events and convert them into an integration payload."""

        effective_limit = self._normalize_limit(limit)
        provider_limit = self._compute_provider_limit(effective_limit)

        try:
            raw_events = self._list_events_with_retry(
                start_at=start_at,
                end_at=end_at,
                limit=provider_limit,
            )
        except Exception:
            LOGGER.exception("Calendar reader failed while listing events.")
            return self._failure_result(
                summary="I couldn't read the calendar right now.",
                code="calendar_reader_failed",
            )

        if raw_events is None:
            events_iterable = iter(())
        else:
            try:
                events_iterable = iter(raw_events)
            except TypeError:
                LOGGER.exception("Calendar reader returned a non-iterable event collection.")
                return self._failure_result(
                    summary="I couldn't read the calendar right now.",
                    code="calendar_reader_invalid_response",
                )

        prepared_events: list[PreparedCalendarEvent] = []
        seen_event_keys: set[str] = set()
        dropped_events = 0
        duplicate_events = 0
        text_truncated_events = 0
        out_of_range_events = 0
        consumed_events = 0

        for event in itertools.islice(events_iterable, provider_limit):
            consumed_events += 1
            try:
                prepared_event = self._prepare_event(event, fallback_timezone=start_at.tzinfo or UTC)
            except Exception:
                dropped_events += 1
                LOGGER.exception("Calendar event serialization failed.")
                continue

            if not self._event_overlaps_window(
                prepared_event=prepared_event,
                window_start_at=start_at,
                window_end_at=end_at,
            ):
                out_of_range_events += 1
                continue

            if prepared_event.dedupe_key in seen_event_keys:
                duplicate_events += 1
                continue

            seen_event_keys.add(prepared_event.dedupe_key)
            if prepared_event.text_was_truncated:
                text_truncated_events += 1
            prepared_events.append(prepared_event)

        if dropped_events and not prepared_events:
            return self._failure_result(
                summary="I couldn't read the calendar right now.",
                code="calendar_event_serialization_failed",
            )

        prepared_events.sort(key=lambda item: item.sort_key)
        selected_events = prepared_events[:effective_limit]
        has_more = len(prepared_events) > effective_limit or consumed_events >= provider_limit
        serialized_events = [event.payload for event in selected_events]
        agenda_lines = [event.agenda_line for event in selected_events if event.agenda_line]

        return IntegrationResult(
            ok=True,
            summary=self._build_summary(
                summary_kind=summary_kind,
                summary_label=summary_label,
                count=len(serialized_events),
            ),
            details=self._build_details(
                serialized_events=serialized_events,
                agenda_lines=agenda_lines,
                start_at=start_at,
                end_at=end_at,
                dropped_events=dropped_events,
                duplicate_events=duplicate_events,
                text_truncated_events=text_truncated_events,
                out_of_range_events=out_of_range_events,
                provider_limit=provider_limit,
                consumed_events=consumed_events,
                has_more=has_more,
            ),
        )

    def _list_events_with_retry(
        self,
        *,
        start_at: datetime,
        end_at: datetime,
        limit: int,
    ) -> list[CalendarEvent] | Any:
        """List events with bounded retries for transient provider failures."""

        max_attempts = self.settings.reader_retry_attempts
        attempt = 0

        while True:
            attempt += 1
            try:
                return self.calendar_reader.list_events(
                    start_at=start_at,
                    end_at=end_at,
                    limit=limit,
                )
            except Exception as exc:
                if attempt >= max_attempts or not self._is_retryable_reader_error(exc):
                    raise

                delay_seconds = min(
                    self.settings.reader_retry_max_delay_seconds,
                    self.settings.reader_retry_base_delay_seconds * (2 ** (attempt - 1)),
                )
                if delay_seconds > 0:
                    delay_seconds += random.uniform(0.0, delay_seconds / 4)
                    LOGGER.warning(
                        "Transient calendar reader failure (%s). Retrying in %.2fs (attempt %s/%s).",
                        type(exc).__name__,
                        delay_seconds,
                        attempt + 1,
                        max_attempts,
                    )
                    time_module.sleep(delay_seconds)

    def _is_retryable_reader_error(self, exc: Exception) -> bool:
        """Return True for transient calendar reader failures worth retrying."""

        return isinstance(exc, (TimeoutError, ConnectionError, OSError))

    def _compute_provider_limit(self, effective_limit: int) -> int:
        """Compute a bounded provider scan limit."""

        return min(
            self.settings.max_reader_scan_events,
            max(
                self.settings.reader_scan_floor,
                effective_limit + self.settings.provider_overfetch,
            ),
        )

    def _prepare_event(self, event: CalendarEvent, *, fallback_timezone: tzinfo) -> PreparedCalendarEvent:
        """Serialize, sanitize, sort, and summarize one event."""

        if isinstance(event, Mapping):
            serialized_event = event
        else:
            serialized_event = event.as_dict()

        if not isinstance(serialized_event, Mapping):
            raise TypeError("CalendarEvent.as_dict() must return a mapping")

        raw_payload = dict(serialized_event)
        sanitized_payload, text_was_truncated = self._sanitize_payload_value(
            raw_payload,
            depth=0,
            field_name=None,
            fallback_timezone=fallback_timezone,
        )
        if not isinstance(sanitized_payload, dict):
            raise TypeError("Sanitized calendar event payload must be a dict")

        sort_key = self._extract_event_sort_key(
            event=event,
            payload=raw_payload,
            fallback_timezone=fallback_timezone,
        )
        dedupe_key = self._build_dedupe_key(
            payload=sanitized_payload,
            sort_key=sort_key,
            fallback_timezone=fallback_timezone,
        )

        if self._is_all_day_payload(raw_payload):
            sanitized_payload["all_day"] = True

        # BREAKING: Returned event payloads are now sanitized and length-bounded; downstream callers should treat details["events"] as safe summaries, not lossless raw provider blobs.
        # SEC-1: Explicitly annotate tool output as untrusted external content so downstream LLM/tool layers can isolate it from instructions.
        sanitized_payload["content_origin"] = "untrusted_calendar_data"

        event_start_at = self._extract_event_sort_key(
            event=event,
            payload=raw_payload,
            fallback_timezone=fallback_timezone,
        )
        event_end_at = self._extract_payload_datetime(
            raw_payload,
            kind="end",
            fallback_timezone=fallback_timezone,
        )

        agenda_line = self._build_agenda_line(
            payload=sanitized_payload,
            fallback_timezone=fallback_timezone,
        )

        return PreparedCalendarEvent(
            payload=sanitized_payload,
            sort_key=sort_key,
            dedupe_key=dedupe_key,
            agenda_line=agenda_line,
            text_was_truncated=text_was_truncated,
            event_start_at=None if event_start_at == _MAX_SORT_DATETIME else event_start_at,
            event_end_at=event_end_at,
        )

    def _sanitize_payload_value(
        self,
        value: Any,
        *,
        depth: int,
        field_name: str | None,
        fallback_timezone: tzinfo,
    ) -> tuple[Any, bool]:
        """Recursively sanitize event payload data into bounded, JSON-safe values."""

        if depth > self.settings.max_payload_depth:
            return None, True

        if value is None or isinstance(value, bool):
            return value, False

        if isinstance(value, int):
            return value, False

        if isinstance(value, float):
            if value != value or value in (float("inf"), float("-inf")):
                return None, True
            return value, False

        if isinstance(value, datetime):
            normalized_value = value if _is_aware_datetime(value) else value.replace(tzinfo=fallback_timezone)
            return normalized_value.isoformat(), False

        if isinstance(value, date):
            return value.isoformat(), False

        if isinstance(value, str):
            return self._sanitize_text(value, max_chars=self._field_char_limit(field_name))

        if isinstance(value, Mapping):
            sanitized_mapping: dict[str, Any] = {}
            any_truncated = False
            for index, (raw_key, raw_item) in enumerate(value.items()):
                if index >= self.settings.max_event_keys:
                    any_truncated = True
                    break
                sanitized_key, key_truncated = self._sanitize_text(str(raw_key), max_chars=64)
                if not sanitized_key:
                    any_truncated = True
                    continue
                sanitized_item, item_truncated = self._sanitize_payload_value(
                    raw_item,
                    depth=depth + 1,
                    field_name=sanitized_key,
                    fallback_timezone=fallback_timezone,
                )
                any_truncated = any_truncated or key_truncated or item_truncated
                if sanitized_item is not None:
                    sanitized_mapping[sanitized_key] = sanitized_item
            return sanitized_mapping, any_truncated

        if isinstance(value, (list, tuple, set, frozenset)):
            sanitized_items: list[Any] = []
            any_truncated = False
            for index, raw_item in enumerate(value):
                if index >= self.settings.max_nested_items:
                    any_truncated = True
                    break
                sanitized_item, item_truncated = self._sanitize_payload_value(
                    raw_item,
                    depth=depth + 1,
                    field_name=field_name,
                    fallback_timezone=fallback_timezone,
                )
                any_truncated = any_truncated or item_truncated
                if sanitized_item is not None:
                    sanitized_items.append(sanitized_item)
            return sanitized_items, any_truncated

        return self._sanitize_text(str(value), max_chars=self._field_char_limit(field_name))

    def _field_char_limit(self, field_name: str | None) -> int:
        """Return the truncation budget for a text field."""

        normalized_field_name = (field_name or "").strip().lower()
        if normalized_field_name in _DESCRIPTION_FIELDS:
            return self.settings.max_event_description_chars
        if normalized_field_name in _TITLE_FIELDS or normalized_field_name in _LOCATION_FIELDS:
            return self.settings.max_event_title_chars
        return self.settings.max_event_field_chars

    def _sanitize_text(self, text: str, *, max_chars: int) -> tuple[str, bool]:
        """Collapse control characters/whitespace and bound returned text size."""

        scan_budget = max_chars * self.settings.text_scan_multiplier
        truncated = len(text) > scan_budget
        scanned_text = text[:scan_budget]

        cleaned_characters: list[str] = []
        for character in scanned_text:
            if character in ("\n", "\r", "\t"):
                cleaned_characters.append(" ")
                continue

            category = unicodedata.category(character)
            if category.startswith("C"):
                truncated = True
                continue

            cleaned_characters.append(character)

        normalized_text = " ".join("".join(cleaned_characters).split())

        if len(normalized_text) > max_chars:
            normalized_text = normalized_text[: max(1, max_chars - 1)].rstrip() + "…"
            truncated = True

        return normalized_text, truncated

    def _is_all_day_payload(self, payload: Mapping[str, Any]) -> bool:
        """Return True when payload looks like an all-day event."""

        raw_all_day = payload.get("all_day")
        if isinstance(raw_all_day, bool):
            return raw_all_day

        for raw_value in self._iter_payload_datetime_candidates(payload=payload, kind="start"):
            if self._value_looks_like_date_only(raw_value):
                return True

        start_value = payload.get("start")
        if isinstance(start_value, Mapping) and start_value.get("date") and not start_value.get("dateTime"):
            return True

        return False

    def _value_looks_like_date_only(self, value: Any) -> bool:
        """Return True when value looks like a date without a time component."""

        if isinstance(value, date) and not isinstance(value, datetime):
            return True

        if isinstance(value, str):
            candidate = value.strip()
            return len(candidate) == 10 and candidate.count("-") == 2

        if isinstance(value, Mapping):
            date_value = value.get("date")
            date_time_value = value.get("dateTime")
            return bool(date_value) and not date_time_value

        return False

    def _build_dedupe_key(
        self,
        *,
        payload: Mapping[str, Any],
        sort_key: datetime,
        fallback_timezone: tzinfo,
    ) -> str:
        """Build a stable deduplication key for one event."""

        for field_name in _ID_FIELDS:
            raw_id = payload.get(field_name)
            if isinstance(raw_id, str) and raw_id:
                return f"id:{raw_id}"

        end_key = self._extract_event_end_key(payload=payload, fallback_timezone=fallback_timezone)
        title = self._event_title(payload).lower()
        location = self._event_location(payload).lower()
        return f"composite:{sort_key.isoformat()}|{end_key}|{title}|{location}"

    def _extract_event_end_key(self, *, payload: Mapping[str, Any], fallback_timezone: tzinfo) -> str:
        """Extract an ISO-formatted end key for deduplication."""

        end_value = self._extract_payload_datetime(payload, kind="end", fallback_timezone=fallback_timezone)
        return end_value.isoformat() if end_value is not None else ""

    def _extract_event_sort_key(
        self,
        *,
        event: CalendarEvent,
        payload: Mapping[str, Any],
        fallback_timezone: tzinfo,
    ) -> datetime:
        """Extract the event start time used for ordering."""

        for raw_value in self._iter_datetime_candidates(event=event, payload=payload, kind="start"):
            parsed_datetime = self._coerce_datetime(raw_value, fallback_timezone=fallback_timezone)
            if parsed_datetime is not None:
                return parsed_datetime

        return _MAX_SORT_DATETIME

    def _iter_datetime_candidates(
        self,
        *,
        event: CalendarEvent,
        payload: Mapping[str, Any],
        kind: str,
    ) -> list[Any]:
        """Return candidate datetime values from payload keys and object attrs."""

        if kind == "start":
            keys = ("start_at", "starts_at", "start_time", "begin", "start")
            attrs = ("start_at", "starts_at", "start_time", "begin", "start")
        else:
            keys = ("end_at", "ends_at", "end_time", "finish", "end")
            attrs = ("end_at", "ends_at", "end_time", "finish", "end")

        candidates: list[Any] = []

        for key in keys:
            if key in payload:
                candidates.append(payload[key])

        nested_key = payload.get(kind)
        if isinstance(nested_key, Mapping):
            candidates.extend(
                [
                    nested_key.get("dateTime"),
                    nested_key.get("date"),
                    nested_key.get("value"),
                ]
            )

        for attr_name in attrs:
            if hasattr(event, attr_name):
                candidates.append(getattr(event, attr_name))

        return candidates

    def _extract_payload_datetime(
        self,
        payload: Mapping[str, Any],
        *,
        kind: str,
        fallback_timezone: tzinfo,
    ) -> datetime | None:
        """Extract a datetime from a sanitized/raw payload mapping."""

        for raw_value in self._iter_payload_datetime_candidates(payload=payload, kind=kind):
            parsed_datetime = self._coerce_datetime(raw_value, fallback_timezone=fallback_timezone)
            if parsed_datetime is not None:
                return parsed_datetime
        return None

    def _iter_payload_datetime_candidates(self, *, payload: Mapping[str, Any], kind: str) -> list[Any]:
        """Return candidate datetime values from a payload mapping."""

        if kind == "start":
            keys = ("start_at", "starts_at", "start_time", "begin", "start")
        else:
            keys = ("end_at", "ends_at", "end_time", "finish", "end")

        candidates: list[Any] = []
        for key in keys:
            if key in payload:
                candidates.append(payload[key])

        nested_value = payload.get(kind)
        if isinstance(nested_value, Mapping):
            candidates.extend(
                [
                    nested_value.get("dateTime"),
                    nested_value.get("date"),
                    nested_value.get("value"),
                ]
            )
        return candidates

    def _coerce_datetime(self, value: Any, *, fallback_timezone: tzinfo) -> datetime | None:
        """Parse a value into an aware datetime in ``fallback_timezone``."""

        if value is None:
            return None

        if isinstance(value, datetime):
            if _is_aware_datetime(value):
                return value.astimezone(fallback_timezone)
            return value.replace(tzinfo=fallback_timezone)

        if isinstance(value, date):
            return datetime.combine(value, time.min, tzinfo=fallback_timezone)

        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return None
            if candidate.endswith("Z"):
                candidate = f"{candidate[:-1]}+00:00"

            try:
                parsed_datetime = datetime.fromisoformat(candidate)
            except ValueError:
                try:
                    parsed_date = date.fromisoformat(candidate)
                except ValueError:
                    return None
                return datetime.combine(parsed_date, time.min, tzinfo=fallback_timezone)

            if _is_aware_datetime(parsed_datetime):
                return parsed_datetime.astimezone(fallback_timezone)
            return parsed_datetime.replace(tzinfo=fallback_timezone)

        if isinstance(value, Mapping):
            return self._coerce_datetime(
                value.get("dateTime") or value.get("date") or value.get("value"),
                fallback_timezone=fallback_timezone,
            )

        return None

    def _event_title(self, payload: Mapping[str, Any]) -> str:
        """Return a safe display title for an event payload."""

        for field_name in ("title", "summary", "subject", "name"):
            value = payload.get(field_name)
            if isinstance(value, str) and value:
                return value
        return "Untitled event"

    def _event_location(self, payload: Mapping[str, Any]) -> str:
        """Return a safe display location for an event payload."""

        for field_name in ("location", "where", "place"):
            value = payload.get(field_name)
            if isinstance(value, str) and value:
                return value
        return ""

    def _build_agenda_line(self, *, payload: Mapping[str, Any], fallback_timezone: tzinfo) -> str:
        """Build a short voice-safe agenda line for one event."""

        title = self._event_title(payload)
        location = self._event_location(payload)
        start_at = self._extract_payload_datetime(payload, kind="start", fallback_timezone=fallback_timezone)
        end_at = self._extract_payload_datetime(payload, kind="end", fallback_timezone=fallback_timezone)
        all_day = bool(payload.get("all_day"))

        if all_day:
            timing = "All day"
        elif start_at is None:
            timing = "Time unavailable"
        elif end_at is not None and end_at > start_at and start_at.date() == end_at.date():
            timing = f"{start_at:%H:%M}–{end_at:%H:%M}"
        else:
            timing = start_at.strftime("%Y-%m-%d %H:%M")

        agenda_line = f"{timing} — {title}"
        if location:
            agenda_line = f"{agenda_line} @ {location}"
        return agenda_line

    def _event_overlaps_window(
        self,
        *,
        prepared_event: PreparedCalendarEvent,
        window_start_at: datetime,
        window_end_at: datetime,
    ) -> bool:
        """Return True when a prepared event overlaps the requested window."""

        event_start_at = prepared_event.event_start_at
        event_end_at = prepared_event.event_end_at

        if event_start_at is None and event_end_at is None:
            return False
        if event_start_at is None:
            return event_end_at is not None and event_end_at > window_start_at
        if event_end_at is None:
            return window_start_at <= event_start_at < window_end_at

        return event_end_at > window_start_at and event_start_at < window_end_at

    def _normalize_limit(self, limit: int | None) -> int:
        """Clamp a requested limit into the configured calendar bounds."""

        if limit is None:
            return self.settings.max_events

        try:
            normalized_limit = int(limit)
        except (TypeError, ValueError):
            LOGGER.warning("Invalid calendar limit %r; falling back to max_events.", limit)
            return self.settings.max_events

        return max(1, min(normalized_limit, self.settings.max_events))

    def _get_now(self, *, timezone_name: str | None = None) -> datetime:
        """Resolve the current instant into the adapter's effective timezone."""

        raw_now = self.clock()
        if not isinstance(raw_now, datetime):
            raise TypeError("clock must return a datetime")

        resolved_timezone = self._resolve_timezone(now=raw_now, override_timezone_name=timezone_name)
        if _is_aware_datetime(raw_now):
            return raw_now.astimezone(resolved_timezone)

        LOGGER.warning(
            "Calendar clock returned naive datetime; assuming timezone %s.",
            _timezone_label(resolved_timezone),
        )
        return raw_now.replace(tzinfo=resolved_timezone)

    def _resolve_timezone(
        self,
        now: datetime | None = None,
        *,
        override_timezone_name: str | None = None,
    ) -> tzinfo:
        """Resolve the timezone used for calendar window calculations."""

        configured_timezone_name = override_timezone_name or self.settings.timezone_name
        if configured_timezone_name:
            return ZoneInfo(configured_timezone_name)

        env_timezone_name = (os.environ.get("TZ") or "").strip()
        if env_timezone_name:
            env_timezone = self._try_load_zoneinfo(env_timezone_name)
            if env_timezone is not None:
                return env_timezone

        system_timezone = self._read_system_timezone()
        if system_timezone is not None:
            return system_timezone

        if now is not None and _is_aware_datetime(now):
            current_timezone = now.tzinfo
            if current_timezone is not None:
                return current_timezone

        return UTC

    def _read_system_timezone(self) -> tzinfo | None:
        """Read the host's configured timezone when available."""

        localtime_path = Path("/etc/localtime")
        try:
            resolved_localtime = localtime_path.resolve(strict=True)
        except OSError:
            resolved_localtime = None

        if resolved_localtime is not None:
            parts = resolved_localtime.parts
            if "zoneinfo" in parts:
                zoneinfo_index = parts.index("zoneinfo")
                zone_name = "/".join(parts[zoneinfo_index + 1 :])
                loaded_timezone = self._try_load_zoneinfo(zone_name)
                if loaded_timezone is not None:
                    return loaded_timezone

        timezone_file_path = Path("/etc/timezone")
        try:
            timezone_name = timezone_file_path.read_text(encoding="utf-8").strip()
        except OSError:
            timezone_name = ""

        if timezone_name:
            loaded_timezone = self._try_load_zoneinfo(timezone_name)
            if loaded_timezone is not None:
                return loaded_timezone

        return None

    def _try_load_zoneinfo(self, timezone_name: str) -> ZoneInfo | None:
        """Return a ``ZoneInfo`` for ``timezone_name`` or ``None``."""

        try:
            return ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            LOGGER.warning("Ignoring unsupported timezone name: %s", timezone_name)
            return None

    def _build_summary(self, *, summary_kind: str, summary_label: str, count: int) -> str:
        """Build a short voice-safe summary for a calendar response."""

        if summary_kind == "today":
            if count == 0:
                return "No events scheduled for today."
            if count == 1:
                return "1 event scheduled for today."
            return f"{count} events scheduled for today."

        if summary_kind == "next_event":
            if count == 0:
                return "No upcoming events found."
            return "Next event found."

        if summary_kind == "date":
            if count == 0:
                return f"No events scheduled for {summary_label}."
            if count == 1:
                return f"1 event scheduled for {summary_label}."
            return f"{count} events scheduled for {summary_label}."

        if summary_kind == "range":
            if count == 0:
                return "No events found in the selected time window."
            if count == 1:
                return "1 event found in the selected time window."
            return f"{count} events found in the selected time window."

        if count == 0:
            return "No upcoming events found."
        if count == 1:
            return "1 upcoming event found."
        return f"{count} upcoming events found."

    def _build_details(
        self,
        *,
        serialized_events: list[dict[str, Any]],
        agenda_lines: list[str],
        start_at: datetime,
        end_at: datetime,
        dropped_events: int,
        duplicate_events: int,
        text_truncated_events: int,
        out_of_range_events: int,
        provider_limit: int,
        consumed_events: int,
        has_more: bool,
    ) -> dict[str, Any]:
        """Build the structured details payload for a successful read."""

        details: dict[str, Any] = {
            "events": serialized_events,
            "agenda_lines": agenda_lines,
            "count": len(serialized_events),
            "has_more": has_more,
            "partial": bool(dropped_events or duplicate_events or text_truncated_events or out_of_range_events),
            "window": {
                "start_at": start_at.isoformat(),
                "end_at": end_at.isoformat(),
                "timezone": _timezone_label(start_at.tzinfo),
            },
            "reader_diagnostics": {
                "provider_limit": provider_limit,
                "consumed_events": consumed_events,
                "dropped_events": dropped_events,
                "duplicate_events": duplicate_events,
                "text_truncated_events": text_truncated_events,
                "out_of_range_events": out_of_range_events,
            },
            "untrusted_source": True,
        }
        return details

    def _failure_result(
        self,
        *,
        summary: str,
        code: str,
        validation_errors: list[dict[str, str]] | None = None,
    ) -> IntegrationResult:
        """Return a standardized failure result for calendar reads."""

        details: dict[str, Any] = {
            "events": [],
            "agenda_lines": [],
            "count": 0,
            "error_code": code,
        }
        if validation_errors:
            details["validation_errors"] = validation_errors

        return IntegrationResult(
            ok=False,
            summary=summary,
            details=details,
        )