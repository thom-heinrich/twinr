from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class CalendarEvent:
    event_id: str
    summary: str
    starts_at: datetime
    ends_at: datetime | None = None
    location: str | None = None
    description: str | None = None
    all_day: bool = False

    def overlaps(self, start_at: datetime, end_at: datetime) -> bool:
        event_end = self.ends_at or self.starts_at
        return self.starts_at < end_at and event_end >= start_at

    def as_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "summary": self.summary,
            "starts_at": self.starts_at.isoformat(),
            "ends_at": self.ends_at.isoformat() if self.ends_at else None,
            "location": self.location,
            "description": self.description,
            "all_day": self.all_day,
        }
