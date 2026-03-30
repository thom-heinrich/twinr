from datetime import UTC, datetime
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.integrations import IntegrationRequest, ReadOnlyCalendarAdapter, manifest_for_id
from twinr.integrations.calendar import ICSCalendarSource, parse_ics_events
from twinr.integrations.calendar.models import CalendarEvent

_ICS_SAMPLE = """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:event-1
DTSTART:20260313T090000Z
DTEND:20260313T100000Z
SUMMARY:Arzttermin
LOCATION:Praxis
DESCRIPTION:Kontrollbesuch
END:VEVENT
BEGIN:VEVENT
UID:event-2
DTSTART;VALUE=DATE:20260314
SUMMARY:Geburtstag Anna
END:VEVENT
END:VCALENDAR
"""


class ICSCalendarParserTests(unittest.TestCase):
    def test_parse_ics_events_supports_timed_and_all_day_events(self) -> None:
        events = parse_ics_events(_ICS_SAMPLE)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].summary, "Arzttermin")
        self.assertFalse(events[0].all_day)
        self.assertTrue(events[1].all_day)
        self.assertEqual(events[1].starts_at.date().isoformat(), "2026-03-14")

    def test_calendar_event_from_json_bytes_decodes_legacy_payload(self) -> None:
        event = CalendarEvent.from_json_bytes(
            b'{"event_id":"event-1","summary":"Arzttermin","starts_at":"2026-03-13T09:00:00+00:00","ends_at":"2026-03-13T10:00:00+00:00","location":"Praxis","description":"Kontrollbesuch","all_day":false}'
        )

        self.assertEqual(event.event_id, "event-1")
        self.assertEqual(event.summary, "Arzttermin")
        self.assertEqual(event.starts_at, datetime(2026, 3, 13, 9, 0, tzinfo=UTC))
        self.assertEqual(event.ends_at, datetime(2026, 3, 13, 10, 0, tzinfo=UTC))
        self.assertEqual(event.location, "Praxis")


class ReadOnlyCalendarAdapterTests(unittest.TestCase):
    def test_read_today_filters_events_to_current_day(self) -> None:
        manifest = manifest_for_id("calendar_agenda")
        assert manifest is not None
        source = ICSCalendarSource(loader=lambda: _ICS_SAMPLE)
        adapter = ReadOnlyCalendarAdapter(
            manifest=manifest,
            calendar_reader=source,
            clock=lambda: datetime(2026, 3, 13, 8, 0, tzinfo=UTC),
        )

        result = adapter.execute(
            IntegrationRequest(
                integration_id="calendar_agenda",
                operation_id="read_today",
            )
        )

        self.assertEqual(result.details["count"], 1)
        self.assertEqual(result.details["events"][0]["summary"], "Arzttermin")

    def test_read_next_event_returns_first_upcoming_event(self) -> None:
        manifest = manifest_for_id("calendar_agenda")
        assert manifest is not None
        source = ICSCalendarSource(loader=lambda: _ICS_SAMPLE)
        adapter = ReadOnlyCalendarAdapter(
            manifest=manifest,
            calendar_reader=source,
            clock=lambda: datetime(2026, 3, 13, 8, 0, tzinfo=UTC),
        )

        result = adapter.execute(
            IntegrationRequest(
                integration_id="calendar_agenda",
                operation_id="read_next_event",
            )
        )

        self.assertEqual(result.details["count"], 1)
        self.assertEqual(result.details["events"][0]["summary"], "Arzttermin")


if __name__ == "__main__":
    unittest.main()
