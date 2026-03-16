"""Expose Twinr's read-only calendar integration package.

Import calendar models, ICS helpers, and adapter types from this package root
when wiring Twinr integrations or writing focused calendar tests.
"""

from twinr.integrations.calendar.adapter import CalendarAdapterSettings, CalendarReader, ReadOnlyCalendarAdapter
from twinr.integrations.calendar.ics import ICSCalendarSource, parse_ics_events, unfold_ics_lines
from twinr.integrations.calendar.models import CalendarEvent

__all__ = [
    "CalendarAdapterSettings",
    "CalendarEvent",
    "CalendarReader",
    "ICSCalendarSource",
    "ReadOnlyCalendarAdapter",
    "parse_ics_events",
    "unfold_ics_lines",
]
