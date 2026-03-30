# CHANGELOG: 2026-03-30
# BUG-1: Fixed missing RRULE/RDATE/EXDATE/RECURRENCE-ID expansion in list_events by switching the primary query path to icalendar + recurring-ical-events.
# BUG-2: Fixed cancelled VEVENTs being surfaced as active events.
# SEC-1: Fixed silent fallback from unresolved explicit TZIDs/X-WR-TIMEZONE to default_timezone; explicit unknown timezones are now rejected instead of shifted.
# IMP-1: Added content-hash query caching and ordered streaming of occurrences for repeated Pi 4 polls.
# IMP-2: Upgraded the parser path to RFC-5545-aware libraries (icalendar>=7.0.3, recurring-ical-events>=3.8.1, tzdata recommended).

"""Parse iCalendar data and expose an ICS-backed calendar reader.

This module converts `.ics` payloads into ``CalendarEvent`` records,
expands recurring events for query windows when modern calendar dependencies
are installed, normalizes timezone handling, and provides a read-only source
used by Twinr's calendar adapter.

Recommended runtime dependencies for the 2026 frontier path:

    pip install "icalendar>=7.0.3" "recurring-ical-events>=3.8.1" "tzdata>=2025.2"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta, tzinfo
import hashlib
import logging
import os
from pathlib import Path
import re
import stat
import threading
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from twinr.integrations.calendar.models import CalendarEvent

try:
    from icalendar import Calendar
except ImportError:
    Calendar = None  # type: ignore[assignment]

try:
    import recurring_ical_events
except ImportError:
    recurring_ical_events = None  # type: ignore[assignment]


_LOGGER = logging.getLogger(__name__)
_MAX_ICS_FILE_BYTES = 2 * 1024 * 1024
_VENDOR_TZID_PREFIX_RE = re.compile(r"^[^A-Za-z]*/")
_DURATION_RE = re.compile(
    r"^(?P<sign>[+-])?P(?:(?P<weeks>\d+)W)?(?:(?P<days>\d+)D)?"
    r"(?:(?:T)(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)
_FRONTIER_REQUIREMENTS = 'Install "icalendar>=7.0.3" and "recurring-ical-events>=3.8.1"; "tzdata" is strongly recommended.'
_FRONTIER_QUERY_AVAILABLE = Calendar is not None and recurring_ical_events is not None
_FRONTIER_PARSE_AVAILABLE = Calendar is not None
_MISSING_FRONTIER_WARNING_EMITTED = False


def unfold_ics_lines(text: str) -> list[str]:
    """Unfold RFC 5545 continuation lines in raw ICS text."""

    lines: list[str] = []
    for raw_line in text.splitlines():
        if raw_line.startswith((" ", "\t")) and lines:
            lines[-1] += raw_line[1:]
        else:
            lines.append(raw_line.rstrip("\r"))
    return lines


def parse_ics_events(text: str, *, default_timezone: tzinfo = UTC) -> list[CalendarEvent]:
    """Parse raw VEVENT blocks from an ICS payload.

    This preserves the historical API shape: it returns one ``CalendarEvent``
    per VEVENT component found in the file and does not expand recurrences.
    ``ICSCalendarSource.list_events()`` is the higher-level API that performs
    occurrence expansion for query windows.
    """

    if _FRONTIER_PARSE_AVAILABLE:
        try:
            calendar = Calendar.from_ical(text.encode("utf-8"))
            events = [
                event
                for component in calendar.walk("VEVENT")
                if (event := _calendar_event_from_component(
                    component,
                    default_timezone=default_timezone,
                    occurrence_scoped_ids=False,
                )) is not None
            ]
            return sorted(events, key=lambda event: event.starts_at)
        except Exception as exc:
            _LOGGER.debug("Falling back to legacy ICS parser after modern parse failure: %s", exc)

    _warn_frontier_dependencies_missing_once()
    return _parse_ics_events_legacy(text, default_timezone=default_timezone)


@dataclass(slots=True)
class ICSCalendarSource:
    """Read events from an ICS payload supplier."""

    loader: Callable[[], str]
    default_timezone: tzinfo = UTC
    _cache_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False, compare=False)
    _cached_text_signature: str | None = field(default=None, init=False, repr=False, compare=False)
    _cached_query: Any | None = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def from_path(cls, path: str | Path, *, default_timezone: tzinfo = UTC) -> "ICSCalendarSource":
        """Build a source that loads ICS text from a filesystem path."""

        file_path = Path(path)
        return cls(
            loader=lambda: _read_ics_text_from_path(file_path),
            default_timezone=default_timezone,
        )

    def list_events(self, *, start_at: datetime, end_at: datetime, limit: int) -> list[CalendarEvent]:
        """List events overlapping a query window.

        The primary 2026 path uses ``icalendar`` + ``recurring-ical-events``
        so RRULE/RDATE/EXDATE/RECURRENCE-ID series are expanded correctly.
        When those dependencies are absent, the source falls back to the
        legacy non-recurring parser to keep the process alive.
        """

        if limit <= 0:
            return []

        try:
            normalised_start = _normalise_query_datetime(start_at, fallback_timezone=self.default_timezone)
            normalised_end = _normalise_query_datetime(end_at, fallback_timezone=self.default_timezone)
            if normalised_end <= normalised_start:
                return []

            raw_text = self.loader()
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            _LOGGER.warning("ICS calendar source could not be loaded: %s", exc)
            return []

        if _FRONTIER_QUERY_AVAILABLE:
            try:
                return self._list_events_frontier(
                    raw_text,
                    start_at=normalised_start,
                    end_at=normalised_end,
                    limit=limit,
                )
            except Exception as exc:
                _LOGGER.warning("ICS calendar source could not be queried with the modern recurrence engine: %s", exc)

        _warn_frontier_dependencies_missing_once()
        try:
            events = parse_ics_events(raw_text, default_timezone=self.default_timezone)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            _LOGGER.warning("ICS calendar source could not be parsed: %s", exc)
            return []

        filtered: list[CalendarEvent] = []
        for event in events:
            try:
                if event.overlaps(normalised_start, normalised_end):
                    filtered.append(event)
                    if len(filtered) >= limit:
                        break
            except Exception as exc:
                _LOGGER.warning("Skipping calendar event during overlap check: %s", exc)
        return filtered

    def _list_events_frontier(
        self,
        raw_text: str,
        *,
        start_at: datetime,
        end_at: datetime,
        limit: int,
    ) -> list[CalendarEvent]:
        """Return overlapping occurrences using the modern RFC-aware stack."""

        query = self._get_or_build_frontier_query(raw_text)
        results: list[CalendarEvent] = []
        seen_ids: set[str] = set()

        def add_component(component: Any, *, ordered_future_stream: bool) -> bool:
            event = _calendar_event_from_component(
                component,
                default_timezone=self.default_timezone,
                occurrence_scoped_ids=True,
            )
            if event is None:
                return False

            if ordered_future_stream and event.starts_at >= end_at and not event.overlaps(start_at, end_at):
                return True

            if not event.overlaps(start_at, end_at):
                return False
            if event.event_id in seen_ids:
                return False

            seen_ids.add(event.event_id)
            results.append(event)
            return len(results) >= limit

        # Events that started before ``start_at`` but are still active at the window
        # boundary would be missed by ``after(start_at)`` alone.
        for component in query.at(start_at):
            if add_component(component, ordered_future_stream=False):
                return sorted(results, key=lambda event: event.starts_at)

        for component in query.after(start_at):
            if add_component(component, ordered_future_stream=True):
                break

        return sorted(results, key=lambda event: event.starts_at)

    def _get_or_build_frontier_query(self, raw_text: str) -> Any:
        """Build or reuse the recurring query object for ``raw_text``."""

        text_signature = _text_signature(raw_text)
        with self._cache_lock:
            if self._cached_text_signature == text_signature and self._cached_query is not None:
                return self._cached_query

        calendar = Calendar.from_ical(raw_text.encode("utf-8"))
        query = recurring_ical_events.of(
            calendar,
            keep_recurrence_attributes=False,
            skip_bad_series=True,
        )

        with self._cache_lock:
            self._cached_text_signature = text_signature
            self._cached_query = query
        return query


def _warn_frontier_dependencies_missing_once() -> None:
    """Log a one-time warning when the frontier parser stack is unavailable."""

    global _MISSING_FRONTIER_WARNING_EMITTED
    if _MISSING_FRONTIER_WARNING_EMITTED or _FRONTIER_QUERY_AVAILABLE:
        return
    _LOGGER.warning(
        "ICS recurrence expansion is running in compatibility fallback mode because modern dependencies are missing. %s",
        _FRONTIER_REQUIREMENTS,
    )
    _MISSING_FRONTIER_WARNING_EMITTED = True


def _text_signature(text: str) -> str:
    """Return a stable content hash for caching parsed calendars."""

    return hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=16).hexdigest()


def _calendar_event_from_component(
    component: Any,
    *,
    default_timezone: tzinfo,
    occurrence_scoped_ids: bool,
) -> CalendarEvent | None:
    """Convert an icalendar VEVENT-like component into a ``CalendarEvent``."""

    if _component_status_is_cancelled(component):
        return None

    raw_start_property = component.get("DTSTART")
    if raw_start_property is None:
        return None

    raw_start_value = _component_temporal_value(raw_start_property)
    starts_at, all_day = _coerce_temporal_value_to_datetime(
        raw_start_value,
        source_property=raw_start_property,
        default_timezone=default_timezone,
    )

    raw_end_property = component.get("DTEND")
    raw_duration_property = component.get("DURATION")

    if raw_end_property is not None:
        raw_end_value = _component_temporal_value(raw_end_property)
        ends_at, _ = _coerce_temporal_value_to_datetime(
            raw_end_value,
            source_property=raw_end_property,
            default_timezone=default_timezone,
        )
        if ends_at <= starts_at:
            raise ValueError("DTEND must be later than DTSTART")
    elif raw_duration_property is not None:
        raw_duration_value = _component_temporal_value(raw_duration_property)
        duration_delta = _coerce_duration_value(raw_duration_value)
        ends_at = starts_at + duration_delta
        if ends_at < starts_at:
            raise ValueError("DURATION must not produce a negative event interval")
    elif all_day:
        ends_at = starts_at + timedelta(days=1)
    else:
        ends_at = starts_at

    summary_text = _component_text(component.get("SUMMARY")) or "Kalendereintrag"
    location = _component_text(component.get("LOCATION"))
    description = _component_text(component.get("DESCRIPTION"))

    if occurrence_scoped_ids:
        event_id = _occurrence_event_id(
            uid=_component_text(component.get("UID")),
            starts_at=starts_at,
            ends_at=ends_at,
            summary=summary_text,
            location=location,
            description=description,
            recurrence_reference=_component_recurrence_reference(component, default_timezone=default_timezone),
        )
    else:
        event_id = _base_event_id(
            uid=_component_text(component.get("UID")),
            starts_at=starts_at,
            ends_at=ends_at,
            summary=summary_text,
            location=location,
            description=description,
            recurrence_reference=_component_recurrence_reference(component, default_timezone=default_timezone),
        )

    return CalendarEvent(
        event_id=event_id,
        summary=summary_text,
        starts_at=starts_at,
        ends_at=ends_at,
        location=location,
        description=description,
        all_day=all_day,
    )


def _component_status_is_cancelled(component: Any) -> bool:
    """Return True for VEVENTs that represent a cancelled occurrence."""

    status = _component_text(component.get("STATUS"))
    return status is not None and status.upper() == "CANCELLED"


def _component_text(value: Any) -> str | None:
    """Convert an icalendar property value into display text."""

    if value is None:
        return None
    if isinstance(value, bytes):
        text_value = value.decode("utf-8", "replace")
    else:
        text_value = str(value)
    stripped = text_value.strip()
    return stripped or None


def _component_temporal_value(value: Any) -> Any:
    """Unwrap icalendar temporal property wrappers."""

    return getattr(value, "dt", value)


def _coerce_temporal_value_to_datetime(
    value: Any,
    *,
    source_property: Any,
    default_timezone: tzinfo,
) -> tuple[datetime, bool]:
    """Normalize an icalendar date/date-time into an aware ``datetime``."""

    if isinstance(value, datetime):
        if _is_aware_datetime(value):
            return value, False

        explicit_tzid = _extract_explicit_tzid(source_property)
        if explicit_tzid is not None:
            # BREAKING: explicit TZIDs that cannot be resolved are rejected instead
            # of being silently interpreted in ``default_timezone``.
            raise ValueError(f"Unknown or unsupported TZID: {explicit_tzid!r}")

        return _attach_timezone(value, default_timezone), False

    if isinstance(value, date):
        return _attach_timezone(datetime.combine(value, time.min), default_timezone), True

    raise ValueError(f"Unsupported iCalendar temporal value: {value!r}")


def _coerce_duration_value(value: Any) -> timedelta:
    """Normalize a duration-like value into ``timedelta``."""

    if isinstance(value, timedelta):
        return value
    if isinstance(value, str):
        return _parse_duration(value)
    raise ValueError(f"Unsupported iCalendar DURATION value: {value!r}")


def _extract_explicit_tzid(source_property: Any) -> str | None:
    """Extract a TZID parameter from an icalendar property wrapper."""

    params = getattr(source_property, "params", None)
    if params is None:
        return None

    try:
        tzid_value = params.get("TZID")
    except Exception:
        return None

    if tzid_value is None:
        return None
    return _component_text(tzid_value)


def _component_recurrence_reference(component: Any, *, default_timezone: tzinfo) -> str | None:
    """Build a stable textual reference for RECURRENCE-ID, when present."""

    raw_property = component.get("RECURRENCE-ID")
    if raw_property is None:
        return None

    raw_value = _component_temporal_value(raw_property)
    if isinstance(raw_value, datetime):
        return _normalise_query_datetime(raw_value, fallback_timezone=default_timezone).isoformat()
    if isinstance(raw_value, date):
        return raw_value.isoformat()

    text_value = _component_text(raw_value)
    return text_value or None


def _occurrence_event_id(
    *,
    uid: str | None,
    starts_at: datetime,
    ends_at: datetime,
    summary: str,
    location: str | None,
    description: str | None,
    recurrence_reference: str | None,
) -> str:
    """Create a unique ID for one concrete occurrence."""

    anchor = recurrence_reference or starts_at.isoformat()
    if uid:
        # BREAKING: occurrence IDs are now scoped to the concrete occurrence
        # start, avoiding UID collisions across recurring series.
        return f"{uid}::{anchor}"

    stable_basis = "|".join(
        [
            summary,
            starts_at.isoformat(),
            ends_at.isoformat(),
            location or "",
            description or "",
            anchor,
        ]
    )
    return f"ics-{hashlib.sha256(stable_basis.encode('utf-8')).hexdigest()[:32]}"


def _base_event_id(
    *,
    uid: str | None,
    starts_at: datetime,
    ends_at: datetime,
    summary: str,
    location: str | None,
    description: str | None,
    recurrence_reference: str | None,
) -> str:
    """Create a stable ID for a raw VEVENT component."""

    if recurrence_reference:
        uid = f"{uid}::{recurrence_reference}" if uid else recurrence_reference
    if uid:
        return uid

    stable_basis = "|".join(
        [
            summary,
            starts_at.isoformat(),
            ends_at.isoformat(),
            location or "",
            description or "",
        ]
    )
    return f"ics-{hashlib.sha256(stable_basis.encode('utf-8')).hexdigest()[:32]}"


def _parse_ics_events_legacy(text: str, *, default_timezone: tzinfo) -> list[CalendarEvent]:
    """Parse VEVENT records from an ICS payload using the compatibility parser."""

    events: list[CalendarEvent] = []
    current: dict[str, tuple[dict[str, str], str]] | None = None
    nested_depth = 0

    for line in unfold_ics_lines(text):
        raw_key, separator, raw_value = line.partition(":")
        if separator and raw_key.upper() == "BEGIN":
            component_name = raw_value.strip().upper()
            if component_name == "VEVENT" and current is None:
                current = {}
                nested_depth = 0
            elif current is not None:
                nested_depth += 1
            continue

        if separator and raw_key.upper() == "END":
            component_name = raw_value.strip().upper()
            if current is None:
                continue
            if nested_depth > 0:
                nested_depth -= 1
                continue
            if component_name == "VEVENT":
                _append_event_if_valid(events, current, default_timezone=default_timezone)
                current = None
            continue

        if current is None or nested_depth > 0 or not separator:
            continue

        parts = _split_ics_key(raw_key)
        key = parts[0].strip().upper()
        params: dict[str, str] = {}
        for part in parts[1:]:
            name, eq, value = part.partition("=")
            if not eq:
                continue
            params[name.strip().upper()] = _clean_param_value(value)
        current[key] = (params, raw_value)

    if current is not None and nested_depth == 0:
        _append_event_if_valid(events, current, default_timezone=default_timezone)

    return sorted(events, key=lambda event: event.starts_at)


def _append_event_if_valid(
    events: list[CalendarEvent],
    fields: dict[str, tuple[dict[str, str], str]],
    *,
    default_timezone: tzinfo,
) -> None:
    """Append a parsed event when the VEVENT fields are valid."""

    try:
        event = _event_from_fields(fields, default_timezone=default_timezone)
    except Exception as exc:
        _LOGGER.warning("Skipping invalid VEVENT in ICS source: %s", exc)
        return
    if event is not None:
        events.append(event)


def _event_from_fields(
    fields: dict[str, tuple[dict[str, str], str]],
    *,
    default_timezone: tzinfo,
) -> CalendarEvent | None:
    """Convert legacy VEVENT fields into a ``CalendarEvent``."""

    status = _optional_text_field(fields.get("STATUS"))
    if status is not None and status.upper() == "CANCELLED":
        return None

    start = fields.get("DTSTART")
    summary = fields.get("SUMMARY")
    if start is None or summary is None:
        return None

    starts_at, all_day = _parse_datetime(start[1], params=start[0], default_timezone=default_timezone)
    end_field = fields.get("DTEND")
    duration_field = fields.get("DURATION")
    ends_at: datetime | None = None

    if end_field is not None:
        ends_at, _ = _parse_datetime(end_field[1], params=end_field[0], default_timezone=default_timezone)
        if ends_at <= starts_at:
            raise ValueError("DTEND must be later than DTSTART")
    elif duration_field is not None:
        ends_at = starts_at + _parse_duration(duration_field[1])
        if ends_at < starts_at:
            raise ValueError("DURATION must not produce a negative event interval")
    elif all_day:
        ends_at = starts_at + timedelta(days=1)
    else:
        ends_at = starts_at

    summary_text = _unescape_ics_text(summary[1]).strip() or "Kalendereintrag"
    location = _optional_text_field(fields.get("LOCATION"))
    description = _optional_text_field(fields.get("DESCRIPTION"))
    recurrence_reference = _legacy_recurrence_reference(fields.get("RECURRENCE-ID"), default_timezone=default_timezone)

    event_id = _base_event_id(
        uid=fields.get("UID", ({}, ""))[1].strip() or None,
        starts_at=starts_at,
        ends_at=ends_at,
        summary=summary_text,
        location=location,
        description=description,
        recurrence_reference=recurrence_reference,
    )

    return CalendarEvent(
        event_id=event_id,
        summary=summary_text,
        starts_at=starts_at,
        ends_at=ends_at,
        location=location,
        description=description,
        all_day=all_day,
    )


def _legacy_recurrence_reference(
    field: tuple[dict[str, str], str] | None,
    *,
    default_timezone: tzinfo,
) -> str | None:
    """Convert a legacy RECURRENCE-ID field into a stable reference string."""

    if field is None:
        return None
    value, all_day = _parse_datetime(field[1], params=field[0], default_timezone=default_timezone)
    if all_day:
        return value.date().isoformat()
    return value.isoformat()


def _parse_datetime(value: str, *, params: dict[str, str], default_timezone: tzinfo) -> tuple[datetime, bool]:
    """Parse an ICS DATE or DATE-TIME value into a normalized datetime."""

    cleaned_value = value.strip()
    if params.get("VALUE", "").upper() == "DATE":
        parsed_date = _parse_date_value(cleaned_value)
        return _attach_timezone(datetime.combine(parsed_date, time.min), default_timezone), True

    timezone_name = params.get("TZID")
    timezone = _timezone_for_name(timezone_name, fallback=default_timezone)
    if cleaned_value.upper().endswith("Z"):
        parsed = _parse_datetime_value(cleaned_value[:-1])
        return parsed.replace(tzinfo=UTC), False
    if len(cleaned_value) == 8:
        parsed_date = _parse_date_value(cleaned_value)
        return _attach_timezone(datetime.combine(parsed_date, time.min), timezone), True

    parsed = _parse_datetime_value(cleaned_value)
    return _attach_timezone(parsed, timezone), False


def _timezone_for_name(name: str | None, *, fallback: tzinfo) -> tzinfo:
    """Resolve an ICS timezone name into a ``tzinfo`` object."""

    if not name:
        return fallback

    cleaned_name = name.strip().strip('"')
    if cleaned_name.upper() in {"UTC", "Z"}:
        return UTC

    candidates = [cleaned_name]
    vendor_stripped = _VENDOR_TZID_PREFIX_RE.sub("", cleaned_name)
    if vendor_stripped and vendor_stripped not in candidates:
        candidates.append(vendor_stripped)
    if "/" in cleaned_name:
        parts = [part for part in cleaned_name.split("/") if part]
        for index in range(1, len(parts) - 1):
            candidate = "/".join(parts[index:])
            if candidate not in candidates:
                candidates.append(candidate)

    for candidate in candidates:
        try:
            return ZoneInfo(candidate)
        except (ValueError, ZoneInfoNotFoundError):
            continue

    # BREAKING: explicit unknown TZIDs are rejected instead of silently being
    # interpreted as ``fallback``.
    raise ValueError(f"Unknown or unsupported TZID: {cleaned_name!r}")


def _normalise_query_datetime(value: datetime, *, fallback_timezone: tzinfo) -> datetime:
    """Normalize a query bound into an aware datetime."""

    if _is_aware_datetime(value):
        return value
    return _attach_timezone(value, fallback_timezone)


def _is_aware_datetime(value: datetime) -> bool:
    """Return True when ``value`` carries a usable timezone offset."""

    return value.tzinfo is not None and value.tzinfo.utcoffset(value) is not None


def _attach_timezone(value: datetime, timezone: tzinfo) -> datetime:
    """Attach ``timezone`` to a naive datetime, validating DST gaps."""

    if _is_aware_datetime(value):
        return value

    aware_fold_0 = value.replace(tzinfo=timezone, fold=0)
    if not isinstance(timezone, ZoneInfo):
        return aware_fold_0

    if _roundtrips_local_time(aware_fold_0, timezone):
        return aware_fold_0

    aware_fold_1 = value.replace(tzinfo=timezone, fold=1)
    if _roundtrips_local_time(aware_fold_1, timezone):
        return aware_fold_1

    raise ValueError(f"Local time {value.isoformat()} does not exist in timezone {timezone!s}")


def _roundtrips_local_time(value: datetime, timezone: ZoneInfo) -> bool:
    """Check whether a timezone attachment preserves the local wall time."""

    round_tripped = value.astimezone(UTC).astimezone(timezone)
    return round_tripped.replace(tzinfo=None) == value.replace(tzinfo=None)


def _parse_date_value(value: str) -> date:
    """Parse an ICS DATE value in ``YYYYMMDD`` format."""

    return datetime.strptime(value, "%Y%m%d").date()


def _parse_datetime_value(value: str) -> datetime:
    """Parse an ICS DATE-TIME value without timezone metadata."""

    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported iCalendar DATE-TIME value: {value!r}")


def _parse_duration(value: str) -> timedelta:
    """Parse an ICS DURATION value into a ``timedelta``."""

    match = _DURATION_RE.fullmatch(value.strip().upper())
    if match is None:
        raise ValueError(f"Unsupported iCalendar DURATION value: {value!r}")

    if not any(match.group(name) for name in ("weeks", "days", "hours", "minutes", "seconds")):
        raise ValueError(f"Unsupported iCalendar DURATION value: {value!r}")

    sign = -1 if match.group("sign") == "-" else 1
    weeks = int(match.group("weeks") or 0)
    days = int(match.group("days") or 0)
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    delta = timedelta(weeks=weeks, days=days, hours=hours, minutes=minutes, seconds=seconds)
    return sign * delta


def _optional_text_field(field: tuple[dict[str, str], str] | None) -> str | None:
    """Decode an optional text field and drop empty results."""

    if field is None:
        return None
    value = _unescape_ics_text(field[1]).strip()
    return value or None


def _unescape_ics_text(value: str) -> str:
    """Decode iCalendar text escape sequences."""

    result: list[str] = []
    index = 0
    while index < len(value):
        char = value[index]
        if char != "\\" or index + 1 >= len(value):
            result.append(char)
            index += 1
            continue

        escaped = value[index + 1]
        if escaped in {"n", "N"}:
            result.append("\n")
        elif escaped == ",":
            result.append(",")
        elif escaped == ";":
            result.append(";")
        elif escaped == "\\":
            result.append("\\")
        else:
            result.append(escaped)
        index += 2
    return "".join(result)


def _split_ics_key(raw_key: str) -> list[str]:
    """Split an ICS property key while respecting quoted parameters."""

    parts: list[str] = []
    current: list[str] = []
    in_quotes = False
    escape = False

    for char in raw_key:
        if escape:
            current.append(char)
            escape = False
            continue
        if char == "\\":
            current.append(char)
            escape = True
            continue
        if char == '"':
            in_quotes = not in_quotes
            current.append(char)
            continue
        if char == ";" and not in_quotes:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)

    parts.append("".join(current))
    return parts


def _clean_param_value(value: str) -> str:
    """Strip whitespace and matching quotes from an ICS parameter value."""

    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] == '"':
        stripped = stripped[1:-1]
    return stripped


def _read_ics_text_from_path(path: Path) -> str:
    """Read UTF-8 ICS text from a bounded regular file path."""

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    fd = os.open(os.fspath(path), flags)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("ICS source path must point to a regular file")
        if metadata.st_size > _MAX_ICS_FILE_BYTES:
            raise ValueError(f"ICS source exceeds {_MAX_ICS_FILE_BYTES} bytes")

        remaining = _MAX_ICS_FILE_BYTES + 1
        chunks: list[bytes] = []
        while remaining > 0:
            chunk = os.read(fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(fd)

    data = b"".join(chunks)
    if len(data) > _MAX_ICS_FILE_BYTES:
        raise ValueError(f"ICS source exceeds {_MAX_ICS_FILE_BYTES} bytes")
    return data.decode("utf-8-sig")