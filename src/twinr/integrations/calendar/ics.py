"""Parse iCalendar data and expose an ICS-backed calendar reader.

This module converts `.ics` payloads into ``CalendarEvent`` records,
normalizes timezone handling for local wall times, and provides a read-only
source used by Twinr's calendar adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, tzinfo
import hashlib
import logging
import os
from pathlib import Path
import re
import stat
from typing import Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from twinr.integrations.calendar.models import CalendarEvent


_LOGGER = logging.getLogger(__name__)  # AUDIT-FIX(#3): Structured logging allows fail-soft recovery without dropping the whole process.
_MAX_ICS_FILE_BYTES = 2 * 1024 * 1024  # AUDIT-FIX(#1): Bound local-file reads to avoid oversized/special-file abuse on the device.
_DURATION_RE = re.compile(
    r"^(?P<sign>[+-])?P(?:(?P<weeks>\d+)W)?(?:(?P<days>\d+)D)?"
    r"(?:(?:T)(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)
_VENDOR_TZID_PREFIX_RE = re.compile(r"^[^A-Za-z]*/")


def unfold_ics_lines(text: str) -> list[str]:
    """Unfold RFC 5545 continuation lines in raw ICS text.

    Args:
        text: Raw iCalendar payload.

    Returns:
        The logical ICS lines with folded continuations joined.
    """

    lines: list[str] = []
    for raw_line in text.splitlines():
        if raw_line.startswith((" ", "\t")) and lines:
            lines[-1] += raw_line[1:]
        else:
            lines.append(raw_line.rstrip("\r"))
    return lines


def parse_ics_events(text: str, *, default_timezone: tzinfo = UTC) -> list[CalendarEvent]:
    """Parse VEVENT records from an ICS payload.

    Args:
        text: Raw iCalendar payload.
        default_timezone: Timezone used for floating or DATE-only values.

    Returns:
        Parsed calendar events sorted by ``starts_at``. Invalid VEVENT blocks
        are skipped instead of aborting the whole parse.
    """

    events: list[CalendarEvent] = []
    current: dict[str, tuple[dict[str, str], str]] | None = None
    nested_depth = 0  # AUDIT-FIX(#2): Ignore nested subcomponents such as VALARM so they cannot overwrite VEVENT fields.

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
                _append_event_if_valid(events, current, default_timezone=default_timezone)  # AUDIT-FIX(#3): One bad VEVENT no longer aborts the full source.
                current = None
            continue

        if current is None or nested_depth > 0 or not separator:
            continue

        parts = _split_ics_key(raw_key)  # AUDIT-FIX(#6): Split parameters without breaking quoted values such as TZID="Europe/Berlin".
        key = parts[0].strip().upper()
        params: dict[str, str] = {}
        for part in parts[1:]:
            name, eq, value = part.partition("=")
            if not eq:
                continue
            params[name.strip().upper()] = _clean_param_value(value)
        current[key] = (params, raw_value)

    if current is not None and nested_depth == 0:
        _append_event_if_valid(events, current, default_timezone=default_timezone)  # AUDIT-FIX(#3): Salvage truncated files that end without END:VEVENT.

    return sorted(events, key=lambda event: event.starts_at)


@dataclass(slots=True)
class ICSCalendarSource:
    """Read events from an ICS payload supplier.

    Attributes:
        loader: Callable returning the raw ICS payload when invoked.
        default_timezone: Timezone used for floating or DATE-only values.
    """

    loader: Callable[[], str]
    default_timezone: tzinfo = UTC

    @classmethod
    def from_path(cls, path: str | Path, *, default_timezone: tzinfo = UTC) -> "ICSCalendarSource":
        """Build a source that loads ICS text from a filesystem path.

        Args:
            path: Local path to the ICS file.
            default_timezone: Timezone used for floating or DATE-only values.

        Returns:
            A source that reloads the file on each query.
        """

        file_path = Path(path)
        return cls(
            loader=lambda: _read_ics_text_from_path(file_path),  # AUDIT-FIX(#1): Harden filesystem reads against symlinks, special files and oversized input.
            default_timezone=default_timezone,
        )

    def list_events(self, *, start_at: datetime, end_at: datetime, limit: int) -> list[CalendarEvent]:
        """List events overlapping a query window.

        Args:
            start_at: Inclusive query-window start.
            end_at: Exclusive query-window end.
            limit: Maximum number of events to return.

        Returns:
            A bounded list of overlapping calendar events.
        """

        if limit <= 0:  # AUDIT-FIX(#4): Negative slicing semantics previously returned the wrong set of events.
            return []

        try:
            normalised_start = _normalise_query_datetime(start_at, fallback_timezone=self.default_timezone)  # AUDIT-FIX(#4): Normalise naive query bounds into the source timezone.
            normalised_end = _normalise_query_datetime(end_at, fallback_timezone=self.default_timezone)
            if normalised_end <= normalised_start:
                return []
            events = parse_ics_events(self.loader(), default_timezone=self.default_timezone)
        except (OSError, UnicodeDecodeError, ValueError) as exc:  # AUDIT-FIX(#3): Calendar source failure degrades to an empty list instead of crashing callers.
            _LOGGER.warning("ICS calendar source could not be loaded or parsed: %s", exc)
            return []

        filtered: list[CalendarEvent] = []
        for event in events:
            try:
                if event.overlaps(normalised_start, normalised_end):
                    filtered.append(event)
                    if len(filtered) >= limit:
                        break
            except Exception as exc:  # AUDIT-FIX(#3): Defensive isolation keeps one malformed event from dropping the full query.
                _LOGGER.warning("Skipping calendar event during overlap check: %s", exc)
        return filtered


def _event_from_fields(
    fields: dict[str, tuple[dict[str, str], str]],
    *,
    default_timezone: tzinfo,
) -> CalendarEvent | None:
    """Convert VEVENT fields into a ``CalendarEvent`` when possible."""

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
        if ends_at <= starts_at:  # AUDIT-FIX(#7): Reject malformed explicit end-times that move backwards or collapse the interval.
            raise ValueError("DTEND must be later than DTSTART")
    elif duration_field is not None:
        ends_at = starts_at + _parse_duration(duration_field[1])  # AUDIT-FIX(#5): Support RFC 5545 DURATION-based VEVENTs.
        if ends_at < starts_at:
            raise ValueError("DURATION must not produce a negative event interval")
    elif all_day:
        ends_at = starts_at + timedelta(days=1)  # AUDIT-FIX(#5): DATE-valued events without DTEND are one day long by definition.
    else:
        ends_at = starts_at  # AUDIT-FIX(#5): DATE-TIME VEVENTs without DTEND or DURATION are zero-duration events by definition.

    summary_text = _unescape_ics_text(summary[1]).strip() or "Kalendereintrag"  # AUDIT-FIX(#8): Decode iCalendar text escapes before surfacing user-visible strings.
    location = _optional_text_field(fields.get("LOCATION"))
    description = _optional_text_field(fields.get("DESCRIPTION"))

    uid = fields.get("UID", ({}, ""))[1].strip()
    recurrence_id = fields.get("RECURRENCE-ID", ({}, ""))[1].strip()
    if recurrence_id:
        uid = f"{uid}::{recurrence_id}" if uid else recurrence_id  # AUDIT-FIX(#9): Recurrence overrides need unique IDs beyond bare UID.
    if not uid:
        stable_basis = "|".join(
            [
                summary_text,
                starts_at.isoformat(),
                ends_at.isoformat() if ends_at is not None else "",
                location or "",
                description or "",
            ]
        )
        uid = f"ics-{hashlib.sha256(stable_basis.encode('utf-8')).hexdigest()[:32]}"  # AUDIT-FIX(#9): Deterministic fallback IDs avoid empty/colliding event identifiers.

    return CalendarEvent(
        event_id=uid,
        summary=summary_text,
        starts_at=starts_at,
        ends_at=ends_at,
        location=location,
        description=description,
        all_day=all_day,
    )


def _parse_datetime(value: str, *, params: dict[str, str], default_timezone: tzinfo) -> tuple[datetime, bool]:
    """Parse an ICS DATE or DATE-TIME value into a normalized datetime."""

    cleaned_value = value.strip()
    if params.get("VALUE", "").upper() == "DATE":
        parsed_date = _parse_date_value(cleaned_value)
        return _attach_timezone(datetime.combine(parsed_date, time.min), default_timezone), True  # AUDIT-FIX(#6): Route all local wall-times through timezone validation.

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

    cleaned_name = name.strip().strip('"')  # AUDIT-FIX(#6): Accept quoted TZID parameter values produced by common calendar exporters.
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

    return fallback


def _append_event_if_valid(
    events: list[CalendarEvent],
    fields: dict[str, tuple[dict[str, str], str]],
    *,
    default_timezone: tzinfo,
) -> None:
    """Append a parsed event when the VEVENT fields are valid."""

    try:
        event = _event_from_fields(fields, default_timezone=default_timezone)
    except Exception as exc:  # AUDIT-FIX(#3): Malformed VEVENTs are isolated and skipped instead of aborting the full import.
        _LOGGER.warning("Skipping invalid VEVENT in ICS source: %s", exc)
        return
    if event is not None:
        events.append(event)


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

    raise ValueError(f"Local time {value.isoformat()} does not exist in timezone {timezone!s}")  # AUDIT-FIX(#6): Reject DST-gap wall-times instead of silently shifting them.


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
