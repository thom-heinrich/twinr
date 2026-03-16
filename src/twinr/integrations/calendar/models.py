"""Define the canonical calendar event model used by Twinr integrations.

This module keeps event timestamps timezone-aware, provides overlap checks for
agenda windows, and serializes events into the payload shape returned by the
calendar adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone  # AUDIT-FIX(#1): add UTC normalization support for robust aware-datetime comparisons.


def _is_timezone_aware(value: datetime) -> bool:
    """Return True when ``value`` carries a usable timezone offset."""

    return value.tzinfo is not None and value.utcoffset() is not None  # AUDIT-FIX(#1): reject naive datetimes instead of assuming process-local time.


def _require_aware_datetime(field_name: str, value: datetime) -> None:
    """Validate that ``value`` is a timezone-aware ``datetime`` instance."""

    if not isinstance(value, datetime):  # AUDIT-FIX(#2): fail fast on invalid runtime payloads.
        raise TypeError(f"{field_name} must be a datetime instance")
    if not _is_timezone_aware(value):  # AUDIT-FIX(#1): overlap math and ISO persistence must not accept naive datetimes.
        raise ValueError(f"{field_name} must be timezone-aware")


def _to_utc(value: datetime) -> datetime:
    """Convert an aware datetime to UTC for stable instant comparison."""

    return value.astimezone(timezone.utc)  # AUDIT-FIX(#1): compare absolute instants to avoid DST-fold ambiguity.


@dataclass(frozen=True, slots=True)
class CalendarEvent:
    """Represent one read-only calendar entry.

    Stores the normalized timestamps and text fields shared by ICS parsing,
    runtime filtering, and integration-result serialization. All datetimes
    must be timezone-aware so overlap math stays deterministic across DST
    changes and across hosts.

    Attributes:
        event_id: Stable source identifier or deterministic fallback ID.
        summary: Short user-visible summary text.
        starts_at: Inclusive start instant as a timezone-aware datetime.
        ends_at: Explicit event end when the source provides one.
        location: Optional location text.
        description: Optional longer event description.
        all_day: True when the event represents an all-day calendar entry.
    """

    event_id: str
    summary: str
    starts_at: datetime
    ends_at: datetime | None = None
    location: str | None = None
    description: str | None = None
    all_day: bool = False

    def __post_init__(self) -> None:
        _require_aware_datetime("starts_at", self.starts_at)  # AUDIT-FIX(#1): enforce timezone-aware construction invariants.
        if self.ends_at is not None:
            _require_aware_datetime("ends_at", self.ends_at)  # AUDIT-FIX(#1): explicit end timestamps must remain safely comparable.
            if _to_utc(self.ends_at) < _to_utc(self.starts_at):
                raise ValueError("ends_at must be greater than or equal to starts_at")  # AUDIT-FIX(#2): block impossible negative event ranges.

    def _effective_end(self) -> datetime:
        """Return the end instant used for overlap and serialization logic."""

        if self.ends_at is not None:
            return self.ends_at
        if self.all_day:
            return self.starts_at + timedelta(days=1)  # AUDIT-FIX(#3): single-day all-day events must span the whole local calendar day.
        return self.starts_at

    def overlaps(self, start_at: datetime, end_at: datetime) -> bool:
        """Check whether the event intersects a half-open time window.

        Args:
            start_at: Inclusive query-window start. Must be timezone-aware.
            end_at: Exclusive query-window end. Must be timezone-aware.

        Returns:
            True when the event overlaps the requested window.

        Raises:
            TypeError: If either boundary is not a ``datetime``.
            ValueError: If a boundary is naive or the window is empty.
        """

        _require_aware_datetime("start_at", start_at)  # AUDIT-FIX(#1): reject naive query windows before comparison.
        _require_aware_datetime("end_at", end_at)  # AUDIT-FIX(#1): reject naive query windows before comparison.

        start_utc = _to_utc(start_at)  # AUDIT-FIX(#1): normalize query bounds to UTC for stable ordering.
        end_utc = _to_utc(end_at)  # AUDIT-FIX(#1): normalize query bounds to UTC for stable ordering.
        if end_utc <= start_utc:
            raise ValueError("end_at must be greater than start_at")  # AUDIT-FIX(#2): reject empty or reversed query windows.

        event_start_utc = _to_utc(self.starts_at)  # AUDIT-FIX(#1): normalize event start for stable comparisons.
        event_end_utc = _to_utc(self._effective_end())  # AUDIT-FIX(#1): normalize event end for stable comparisons.
        if event_end_utc == event_start_utc:
            return start_utc <= event_start_utc < end_utc  # AUDIT-FIX(#4): point events overlap only when the instant lies inside the queried window.
        return event_start_utc < end_utc and event_end_utc > start_utc  # AUDIT-FIX(#4): use half-open interval semantics for duration events.

    def as_dict(self) -> dict[str, object]:
        """Serialize the event into the calendar integration result shape.

        Returns:
            A JSON-serializable mapping with ISO-8601 timestamps and optional
            text fields preserved for downstream voice and web consumers.
        """

        effective_end = self._effective_end()  # AUDIT-FIX(#3): persist derived all-day end times explicitly to remove restart ambiguity.
        serialized_end = effective_end.isoformat() if self.ends_at is not None or self.all_day else None
        return {
            "event_id": self.event_id,
            "summary": self.summary,
            "starts_at": self.starts_at.isoformat(),
            "ends_at": serialized_end,
            "location": self.location,
            "description": self.description,
            "all_day": self.all_day,
        }
