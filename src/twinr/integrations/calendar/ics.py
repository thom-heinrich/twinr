from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from twinr.integrations.calendar.models import CalendarEvent


def unfold_ics_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        if raw_line.startswith((" ", "\t")) and lines:
            lines[-1] += raw_line[1:]
        else:
            lines.append(raw_line.rstrip("\r"))
    return lines


def parse_ics_events(text: str, *, default_timezone: tzinfo = UTC) -> list[CalendarEvent]:
    events: list[CalendarEvent] = []
    current: dict[str, tuple[dict[str, str], str]] | None = None

    for line in unfold_ics_lines(text):
        if line == "BEGIN:VEVENT":
            current = {}
            continue
        if line == "END:VEVENT":
            if current is not None:
                event = _event_from_fields(current, default_timezone=default_timezone)
                if event is not None:
                    events.append(event)
            current = None
            continue
        if current is None or ":" not in line:
            continue

        raw_key, raw_value = line.split(":", 1)
        parts = raw_key.split(";")
        key = parts[0].upper()
        params: dict[str, str] = {}
        for part in parts[1:]:
            if "=" not in part:
                continue
            name, value = part.split("=", 1)
            params[name.upper()] = value
        current[key] = (params, raw_value)

    return sorted(events, key=lambda event: event.starts_at)


@dataclass(slots=True)
class ICSCalendarSource:
    loader: Callable[[], str]
    default_timezone: tzinfo = UTC

    @classmethod
    def from_path(cls, path: str | Path, *, default_timezone: tzinfo = UTC) -> "ICSCalendarSource":
        file_path = Path(path)
        return cls(loader=lambda: file_path.read_text(encoding="utf-8"), default_timezone=default_timezone)

    def list_events(self, *, start_at: datetime, end_at: datetime, limit: int) -> list[CalendarEvent]:
        events = parse_ics_events(self.loader(), default_timezone=self.default_timezone)
        filtered = [event for event in events if event.overlaps(start_at, end_at)]
        return filtered[:limit]


def _event_from_fields(
    fields: dict[str, tuple[dict[str, str], str]],
    *,
    default_timezone: tzinfo,
) -> CalendarEvent | None:
    start = fields.get("DTSTART")
    summary = fields.get("SUMMARY")
    if start is None or summary is None:
        return None

    starts_at, all_day = _parse_datetime(start[1], params=start[0], default_timezone=default_timezone)
    end_field = fields.get("DTEND")
    ends_at = None
    if end_field is not None:
        ends_at, _ = _parse_datetime(end_field[1], params=end_field[0], default_timezone=default_timezone)
    elif all_day:
        ends_at = starts_at + timedelta(days=1)

    uid = fields.get("UID", ({}, summary[1]))[1]
    location = fields.get("LOCATION", ({}, ""))[1].strip() or None
    description = fields.get("DESCRIPTION", ({}, ""))[1].strip() or None
    return CalendarEvent(
        event_id=uid.strip(),
        summary=summary[1].strip(),
        starts_at=starts_at,
        ends_at=ends_at,
        location=location,
        description=description,
        all_day=all_day,
    )


def _parse_datetime(value: str, *, params: dict[str, str], default_timezone: tzinfo) -> tuple[datetime, bool]:
    if params.get("VALUE", "").upper() == "DATE":
        parsed_date = datetime.strptime(value, "%Y%m%d").date()
        return datetime.combine(parsed_date, time.min, tzinfo=default_timezone), True

    timezone_name = params.get("TZID")
    timezone = _timezone_for_name(timezone_name, fallback=default_timezone)
    if value.endswith("Z"):
        parsed = datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        return parsed, False
    if len(value) == 8:
        parsed_date = datetime.strptime(value, "%Y%m%d").date()
        return datetime.combine(parsed_date, time.min, tzinfo=timezone), True

    parsed = datetime.strptime(value, "%Y%m%dT%H%M%S")
    return parsed.replace(tzinfo=timezone), False


def _timezone_for_name(name: str | None, *, fallback: tzinfo) -> tzinfo:
    if not name:
        return fallback
    try:
        return ZoneInfo(name)
    except Exception:
        return fallback
