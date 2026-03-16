"""Adapt read-only calendar sources to Twinr integration requests.

This module translates supported calendar operations into bounded
``IntegrationResult`` payloads while centralizing request validation,
timezone resolution, and user-facing agenda summaries.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta, tzinfo
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from twinr.integrations.adapter import IntegrationAdapter
from twinr.integrations.calendar.models import CalendarEvent
from twinr.integrations.models import IntegrationManifest, IntegrationRequest, IntegrationResult


LOGGER = logging.getLogger(__name__)


def _default_clock() -> datetime:
    """Return the current local wall-clock time as an aware datetime."""

    # AUDIT-FIX(#1): Use local wall-clock time by default so "today" is evaluated in the household timezone, not hard-coded UTC.
    return datetime.now().astimezone()


def _is_aware_datetime(value: datetime) -> bool:
    """Return True when ``value`` carries a usable timezone offset."""

    # AUDIT-FIX(#1): Centralize aware/naive checks so every calendar window is built from a deterministic timezone state.
    return value.tzinfo is not None and value.tzinfo.utcoffset(value) is not None


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
    """

    max_events: int = 12
    default_upcoming_days: int = 7
    # AUDIT-FIX(#4): Make the look-ahead cap explicit and validated so invalid config cannot create unbounded or negative windows.
    max_upcoming_days: int = 30
    # AUDIT-FIX(#1): Allow an explicit IANA timezone override so DST and "today" boundaries are deterministic across hosts.
    timezone_name: str | None = None

    def __post_init__(self) -> None:
        # AUDIT-FIX(#1): Normalize and validate the configured timezone during startup instead of failing on the first live request.
        normalized_timezone_name = self.timezone_name.strip() if self.timezone_name is not None else None
        object.__setattr__(self, "timezone_name", normalized_timezone_name or None)

        if normalized_timezone_name:
            try:
                ZoneInfo(normalized_timezone_name)
            except ZoneInfoNotFoundError as exc:
                raise ValueError(f"Unsupported timezone_name: {normalized_timezone_name}") from exc

        # AUDIT-FIX(#4): Reject zero or negative limits at construction time so production traffic cannot inherit broken settings.
        if self.max_events < 1:
            raise ValueError("max_events must be >= 1")
        if self.default_upcoming_days < 1:
            raise ValueError("default_upcoming_days must be >= 1")
        if self.max_upcoming_days < 1:
            raise ValueError("max_upcoming_days must be >= 1")


@dataclass(slots=True)
class ReadOnlyCalendarAdapter(IntegrationAdapter):
    """Serve agenda-style calendar reads through the integration interface.

    The adapter accepts high-level operations such as "today", "upcoming", and
    "next event", then queries a ``CalendarReader`` and returns a bounded,
    voice-safe ``IntegrationResult`` payload.

    Attributes:
        manifest: Integration manifest describing the calendar capability.
        calendar_reader: Injected source implementation that lists events.
        settings: Limits and timezone overrides for request handling.
        clock: Injectable clock used to anchor relative time windows.
    """

    manifest: IntegrationManifest
    calendar_reader: CalendarReader
    settings: CalendarAdapterSettings = field(default_factory=CalendarAdapterSettings)
    clock: Callable[[], datetime] = _default_clock

    def __post_init__(self) -> None:
        # AUDIT-FIX(#5): Fail fast on broken dependency injection so miswired adapters die during boot, not during a senior interaction.
        if not isinstance(self.calendar_reader, CalendarReader):
            raise TypeError("calendar_reader must implement CalendarReader")
        if not callable(self.clock):
            raise TypeError("clock must be callable")

    def execute(self, request: IntegrationRequest) -> IntegrationResult:
        """Execute a supported calendar read operation.

        Args:
            request: Integration request describing the calendar operation.

        Returns:
            A success or failure ``IntegrationResult`` with bounded event data.
        """

        # AUDIT-FIX(#2): Convert malformed requests and unsupported operations into safe IntegrationResult failures instead of uncaught exceptions.
        try:
            now = self._get_now()
            operation_id = getattr(request, "operation_id", None)

            if operation_id == "read_today":
                return self._read_today(now)
            if operation_id == "read_upcoming":
                days = self._parse_days(request)
                if days is None:
                    return self._failure_result(
                        summary="I couldn't understand how many days to look ahead.",
                        code="invalid_days_parameter",
                    )
                return self._read_upcoming(now, days=days)
            if operation_id == "read_next_event":
                return self._read_upcoming(
                    now,
                    days=self.settings.default_upcoming_days,
                    limit=1,
                    summary_label="next event",
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

    def _read_today(self, now: datetime) -> IntegrationResult:
        """Build a result for the current local calendar day."""

        # AUDIT-FIX(#1): Build the day window in the resolved local/IANA timezone so "today" remains correct around midnight and DST changes.
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
        """Build a result for a bounded upcoming-events window."""

        # AUDIT-FIX(#4): Clamp look-ahead windows and normalize limits explicitly instead of relying on truthiness or unchecked config.
        bounded_days = max(1, min(days, self.settings.max_upcoming_days))
        return self._build_result(
            start_at=now,
            end_at=now + timedelta(days=bounded_days),
            limit=self._normalize_limit(limit),
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
        """Read events and convert them into an integration payload."""

        effective_limit = self._normalize_limit(limit)

        # AUDIT-FIX(#3): Isolate calendar-provider failures at the adapter boundary so network/provider errors degrade gracefully.
        try:
            raw_events = self.calendar_reader.list_events(
                start_at=start_at,
                end_at=end_at,
                limit=effective_limit,
            )
        except Exception:
            LOGGER.exception("Calendar reader failed while listing events.")
            return self._failure_result(
                summary="I couldn't read the calendar right now.",
                code="calendar_reader_failed",
            )

        try:
            events = [] if raw_events is None else list(raw_events)
        except TypeError:
            LOGGER.exception("Calendar reader returned a non-iterable event collection.")
            return self._failure_result(
                summary="I couldn't read the calendar right now.",
                code="calendar_reader_invalid_response",
            )

        # AUDIT-FIX(#3): Serialize events defensively so one malformed event does not abort the whole calendar response.
        serialized_events: list[dict[str, Any]] = []
        dropped_events = 0
        for event in events[:effective_limit]:
            try:
                serialized_events.append(event.as_dict())
            except Exception:
                dropped_events += 1
                LOGGER.exception("Calendar event serialization failed.")

        if dropped_events and not serialized_events:
            return self._failure_result(
                summary="I couldn't read the calendar right now.",
                code="calendar_event_serialization_failed",
            )

        return IntegrationResult(
            ok=True,
            # AUDIT-FIX(#6): Use direct, non-jargon summaries so downstream voice output is understandable for seniors.
            summary=self._build_summary(summary_label=summary_label, count=len(serialized_events)),
            # AUDIT-FIX(#3): Return partial-success metadata so downstream layers can distinguish missing events from full success.
            details=self._build_details(serialized_events=serialized_events, dropped_events=dropped_events),
        )

    def _parse_days(self, request: IntegrationRequest) -> int | None:
        """Parse the optional upcoming-days parameter from a request."""

        parameters = getattr(request, "parameters", None)
        if parameters is None:
            return self.settings.default_upcoming_days

        # AUDIT-FIX(#2): Validate request.parameters shape before reading "days" so malformed payloads cannot raise at runtime.
        if not isinstance(parameters, Mapping):
            LOGGER.warning(
                "Calendar request parameters were not a mapping: %s",
                type(parameters).__name__,
            )
            return None

        raw_days = parameters.get("days", self.settings.default_upcoming_days)
        if raw_days in (None, ""):
            return self.settings.default_upcoming_days
        if isinstance(raw_days, bool):
            LOGGER.warning("Calendar request days parameter must not be boolean.")
            return None

        try:
            return int(raw_days)
        except (TypeError, ValueError):
            LOGGER.warning("Calendar request days parameter could not be parsed: %r", raw_days)
            return None

    def _normalize_limit(self, limit: int | None) -> int:
        """Clamp a requested limit into the configured calendar bounds."""

        # AUDIT-FIX(#4): Normalize limits numerically; "limit or max_events" is incorrect for 0/False and hides bad internal callers.
        if limit is None:
            return self.settings.max_events

        try:
            normalized_limit = int(limit)
        except (TypeError, ValueError):
            LOGGER.warning("Invalid calendar limit %r; falling back to max_events.", limit)
            return self.settings.max_events

        return max(1, min(normalized_limit, self.settings.max_events))

    def _get_now(self) -> datetime:
        """Resolve the current instant into the adapter's effective timezone."""

        raw_now = self.clock()
        if not isinstance(raw_now, datetime):
            raise TypeError("clock must return a datetime")

        resolved_timezone = self._resolve_timezone(raw_now)
        if _is_aware_datetime(raw_now):
            # AUDIT-FIX(#1): Normalize all calendar calculations into one aware timezone before building time windows.
            return raw_now.astimezone(resolved_timezone)

        LOGGER.warning(
            "Calendar clock returned naive datetime; assuming timezone %s.",
            getattr(resolved_timezone, "key", str(resolved_timezone)),
        )
        # AUDIT-FIX(#1): Repair naive clock outputs deterministically instead of letting downstream code mix naive and aware datetimes.
        return raw_now.replace(tzinfo=resolved_timezone)

    def _resolve_timezone(self, now: datetime | None = None) -> tzinfo:
        """Resolve the timezone used for calendar window calculations."""

        # AUDIT-FIX(#1): Prefer an explicit IANA timezone, then OS timezone data, before falling back to the injected clock timezone.
        configured_timezone_name = self.settings.timezone_name
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
        """Read the host's configured zoneinfo timezone when available."""

        # AUDIT-FIX(#1): Read the host's configured zoneinfo name so future-day windows remain DST-aware on Linux/RPi deployments.
        localtime_path = Path("/etc/localtime")

        try:
            resolved_localtime = localtime_path.resolve(strict=True)
        except OSError:
            return None

        parts = resolved_localtime.parts
        if "zoneinfo" not in parts:
            return None

        zoneinfo_index = parts.index("zoneinfo")
        zone_name = "/".join(parts[zoneinfo_index + 1 :])
        return self._try_load_zoneinfo(zone_name)

    def _try_load_zoneinfo(self, timezone_name: str) -> ZoneInfo | None:
        """Return a ``ZoneInfo`` for ``timezone_name`` or ``None``."""

        # AUDIT-FIX(#1): Treat unverifiable timezone hints as soft failures and continue with safer fallbacks.
        try:
            return ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            LOGGER.warning("Ignoring unsupported timezone name: %s", timezone_name)
            return None

    def _build_summary(self, *, summary_label: str, count: int) -> str:
        """Build a short voice-safe summary for a calendar response."""

        # AUDIT-FIX(#6): Provide explicit empty-state and singular/plural summaries that are suitable for voice feedback.
        if summary_label == "today's agenda":
            if count == 0:
                return "No events scheduled for today."
            if count == 1:
                return "1 event scheduled for today."
            return f"{count} events scheduled for today."

        if summary_label == "next event":
            if count == 0:
                return "No upcoming events found."
            return "Next event found."

        if count == 0:
            return "No upcoming events found."
        if count == 1:
            return "1 upcoming event found."
        return f"{count} upcoming events found."

    def _build_details(
        self,
        *,
        serialized_events: list[dict[str, Any]],
        dropped_events: int,
    ) -> dict[str, Any]:
        """Build the structured details payload for a successful read."""

        # AUDIT-FIX(#3): Preserve partial-success diagnostics without changing the existing success payload shape for healthy reads.
        details: dict[str, Any] = {
            "events": serialized_events,
            "count": len(serialized_events),
        }
        if dropped_events:
            details["dropped_events"] = dropped_events
        return details

    def _failure_result(self, *, summary: str, code: str) -> IntegrationResult:
        """Return a standardized failure result for calendar reads."""

        # AUDIT-FIX(#2): Standardize failure payloads so upstream layers can branch on ok/error_code without exception handling.
        return IntegrationResult(
            ok=False,
            summary=summary,
            details={"events": [], "count": 0, "error_code": code},
        )
