"""Expose managed calendar primitives for future self_coding skills."""

from __future__ import annotations

from typing import NoReturn

from twinr.agent.self_coding.status import CapabilityRiskClass

from .base import SelfCodingModuleFunction, SelfCodingModuleSpec, runtime_unavailable


def _runtime_unavailable(operation: str) -> NoReturn:
    """Raise the canonical runtime-unavailable error for calendar capabilities."""

    runtime_unavailable(operation)
    # AUDIT-FIX(#1): Defensive fallback so these APIs never leak an implicit None
    # if the upstream helper stops raising or is replaced in tests.
    raise RuntimeError(
        f"{operation} is unavailable, but runtime_unavailable() returned unexpectedly"
    )


def _validate_reminder_request(title: str, minutes: int) -> None:
    """Validate reminder inputs before any future runtime dispatch."""

    # AUDIT-FIX(#2): Reject blank titles early so future implementations cannot
    # target an empty or whitespace-only event name.
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string")

    # AUDIT-FIX(#2): Reject bools explicitly because bool is an int subclass,
    # and reject negative offsets because they invert the reminder semantics.
    if isinstance(minutes, bool) or not isinstance(minutes, int) or minutes < 0:
        raise ValueError("minutes must be a non-negative integer")


def today() -> list[dict[str, str]]:
    """Return today's bounded calendar events."""

    _runtime_unavailable("calendar.today")


def next_event() -> dict[str, str] | None:
    """Return the next upcoming calendar event, if any."""

    _runtime_unavailable("calendar.next_event")


def remind_before(title: str, minutes: int) -> str:
    """Create a reminder tied to one calendar event title."""

    _validate_reminder_request(title, minutes)
    _runtime_unavailable("calendar.remind_before")


MODULE_SPEC = SelfCodingModuleSpec(
    capability_id="calendar",
    module_name="calendar",
    # AUDIT-FIX(#3): Keep the module summary aligned with the actual public API,
    # which includes bounded reminder creation in addition to read operations.
    summary=(
        "Read agenda, inspect upcoming events, and create bounded reminders "
        "through the managed calendar integration."
    ),
    risk_class=CapabilityRiskClass.MODERATE,
    public_api=(
        SelfCodingModuleFunction(
            name="today",
            signature="today() -> list[dict[str, str]]",
            summary="Return today's bounded event list with titles and times.",
            returns="a list of JSON-safe event records",
            tags=("read_only", "integration"),
        ),
        SelfCodingModuleFunction(
            name="next_event",
            signature="next_event() -> dict[str, str] | None",
            summary="Return the next upcoming event from the configured agenda.",
            returns="one JSON-safe event record or None",
            tags=("read_only", "integration"),
        ),
        SelfCodingModuleFunction(
            name="remind_before",
            signature="remind_before(title: str, minutes: int) -> str",
            summary="Create one bounded reminder before a named event.",
            returns="a reminder identifier",
            effectful=True,
            tags=("effectful", "integration"),
        ),
    ),
    requires_configuration=True,
    integration_id="calendar_agenda",
    tags=("integration", "calendar", "configured"),
)

# AUDIT-FIX(#4): Freeze the exported symbol list to prevent accidental runtime
# mutation of the module's public surface.
__all__ = ("MODULE_SPEC", "next_event", "remind_before", "today")