from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from typing import Callable, Protocol, runtime_checkable

from twinr.integrations.adapter import IntegrationAdapter
from twinr.integrations.calendar.models import CalendarEvent
from twinr.integrations.models import IntegrationManifest, IntegrationRequest, IntegrationResult


@runtime_checkable
class CalendarReader(Protocol):
    def list_events(self, *, start_at: datetime, end_at: datetime, limit: int) -> list[CalendarEvent]:
        ...


@dataclass(frozen=True, slots=True)
class CalendarAdapterSettings:
    max_events: int = 12
    default_upcoming_days: int = 7


@dataclass(slots=True)
class ReadOnlyCalendarAdapter(IntegrationAdapter):
    manifest: IntegrationManifest
    calendar_reader: CalendarReader
    settings: CalendarAdapterSettings = field(default_factory=CalendarAdapterSettings)
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)

    def execute(self, request: IntegrationRequest) -> IntegrationResult:
        now = self.clock()
        if request.operation_id == "read_today":
            return self._read_today(now)
        if request.operation_id == "read_upcoming":
            days = int(request.parameters.get("days", self.settings.default_upcoming_days))
            return self._read_upcoming(now, days=days)
        if request.operation_id == "read_next_event":
            return self._read_upcoming(now, days=self.settings.default_upcoming_days, limit=1, summary_label="next event")
        raise ValueError(f"Unsupported calendar operation: {request.operation_id}")

    def _read_today(self, now: datetime) -> IntegrationResult:
        start_at = datetime.combine(now.date(), time.min, tzinfo=now.tzinfo)
        end_at = start_at + timedelta(days=1)
        return self._build_result(
            start_at=start_at,
            end_at=end_at,
            limit=self.settings.max_events,
            summary_label="today's agenda",
        )

    def _read_upcoming(
        self,
        now: datetime,
        *,
        days: int,
        limit: int | None = None,
        summary_label: str = "upcoming events",
    ) -> IntegrationResult:
        bounded_days = max(1, min(days, 30))
        return self._build_result(
            start_at=now,
            end_at=now + timedelta(days=bounded_days),
            limit=limit or self.settings.max_events,
            summary_label=summary_label,
        )

    def _build_result(
        self,
        *,
        start_at: datetime,
        end_at: datetime,
        limit: int,
        summary_label: str,
    ) -> IntegrationResult:
        events = self.calendar_reader.list_events(
            start_at=start_at,
            end_at=end_at,
            limit=min(limit, self.settings.max_events),
        )
        return IntegrationResult(
            ok=True,
            summary=f"{len(events)} {summary_label} ready.",
            details={"events": [event.as_dict() for event in events], "count": len(events)},
        )
