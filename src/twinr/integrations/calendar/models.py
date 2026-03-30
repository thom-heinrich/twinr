# CHANGELOG: 2026-03-30
# BUG-1: Blank/whitespace event IDs now get a deterministic fallback ID instead of leaking empty IDs into dedupe/sync layers.
# BUG-2: all_day events now require local-midnight boundaries and a positive span, preventing partial-day events from being mislabeled and misrendered.
# SEC-1: Text fields are size-bounded and control-character-sanitized so hostile calendar feeds cannot easily jam logs/TTS/web views or waste Pi memory.
# IMP-1: Added strict from_mapping/from_json/to_json helpers with optional msgspec fast path and unknown-field rejection.
# IMP-2: Added effective_end/duration helpers plus JSCalendar export for modern JSON-calendar interoperability.

"""Define the canonical calendar event model used by Twinr integrations.

This module keeps event timestamps timezone-aware, provides overlap checks for
agenda windows, serializes events into the legacy calendar-adapter payload
shape, and adds strict import/export helpers for modern JSON calendaring
pipelines.

Public API:
    CalendarEvent(...): immutable canonical event model.
    CalendarEvent.overlaps(start_at, end_at): half-open window intersection.
    CalendarEvent.as_dict(...): legacy adapter payload serialization.
    CalendarEvent.to_json(...): JSON serialization of the legacy payload.
    CalendarEvent.as_jscalendar(): minimal JSCalendar event export.
    CalendarEvent.from_mapping(...): strict legacy payload validation/import.
    CalendarEvent.from_json(...): strict JSON import, optionally accelerated by
        msgspec when available.

Notes:
    * The model intentionally requires timezone-aware datetimes everywhere.
    * all_day events are represented as local-midnight-starting spans.
    * For 2026-style strict JSON schema handling on constrained devices, the
      module uses msgspec when installed but remains fully functional without it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
from typing import Annotated, Self

try:  # Python 3.11+
    from datetime import UTC
except ImportError:  # pragma: no cover - Python 3.10 fallback.
    UTC = timezone.utc

try:
    import msgspec  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional acceleration path.
    msgspec = None

__all__ = ["CalendarEvent"]


# BREAKING: absurdly large or control-character-laden text payloads are now
# rejected/sanitized at the model boundary instead of passing through.
_MAX_EVENT_ID_BYTES = 1_024
_MAX_SUMMARY_BYTES = 512
_MAX_LOCATION_BYTES = 1_024
_MAX_DESCRIPTION_BYTES = 16_384

_ALLOWED_FIELDS = frozenset(
    {
        "event_id",
        "summary",
        "starts_at",
        "ends_at",
        "location",
        "description",
        "all_day",
    }
)

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_LINEBREAK_RE = re.compile(r"\r\n?|\n")
_WHITESPACE_RE = re.compile(r"[ \t]+")

if msgspec is not None:  # pragma: no branch
    AwareDateTime = Annotated[datetime, msgspec.Meta(tz=True)]

    class _CalendarEventPayload(msgspec.Struct, forbid_unknown_fields=True):
        """Strict legacy payload schema used by the optional msgspec fast path."""

        starts_at: AwareDateTime
        event_id: str | None = None
        summary: str = ""
        ends_at: AwareDateTime | None = None
        location: str | None = None
        description: str | None = None
        all_day: bool = False


def _is_timezone_aware(value: datetime) -> bool:
    """Return True when ``value`` carries a usable timezone offset."""

    return value.tzinfo is not None and value.utcoffset() is not None


def _require_aware_datetime(field_name: str, value: datetime) -> None:
    """Validate that ``value`` is an aware ``datetime`` instance."""

    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime instance")
    if not _is_timezone_aware(value):
        raise ValueError(f"{field_name} must be timezone-aware")


def _to_utc(value: datetime) -> datetime:
    """Convert an aware datetime to UTC for stable instant comparison."""

    return value.astimezone(UTC)


def _is_local_midnight(value: datetime) -> bool:
    """Return True when the local wall-clock time is exactly midnight."""

    return (
        value.hour == 0
        and value.minute == 0
        and value.second == 0
        and value.microsecond == 0
    )


def _normalize_line_endings(value: str) -> str:
    """Convert CRLF/CR line endings to LF."""

    return _LINEBREAK_RE.sub("\n", value)


def _sanitize_single_line(value: str) -> str:
    """Normalize control characters and collapse to a single line."""

    value = _normalize_line_endings(value)
    value = value.replace("\n", " ")
    value = _CONTROL_CHARS_RE.sub("", value)
    value = _WHITESPACE_RE.sub(" ", value)
    return value.strip()


def _sanitize_multi_line(value: str) -> str:
    """Normalize control characters while preserving line structure."""

    value = _normalize_line_endings(value)
    value = _CONTROL_CHARS_RE.sub("", value)
    return value.strip()


def _validate_text_bytes(field_name: str, value: str, *, max_bytes: int) -> str:
    """Fail fast on text values that are implausibly large for Pi deployments."""

    encoded = value.encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError(f"{field_name} exceeds {max_bytes} UTF-8 bytes")
    return value


def _normalize_text(
    field_name: str,
    value: object,
    *,
    required: bool,
    multiline: bool,
    max_bytes: int,
    allow_empty: bool = True,
) -> str | None:
    """Validate and sanitize textual runtime payloads."""

    if value is None:
        if required:
            raise TypeError(f"{field_name} must be a string")
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    cleaned = _sanitize_multi_line(value) if multiline else _sanitize_single_line(value)
    if not cleaned and not allow_empty:
        raise ValueError(f"{field_name} must not be empty")
    return _validate_text_bytes(field_name, cleaned, max_bytes=max_bytes)


def _make_fallback_event_id(
    *,
    summary: str,
    starts_at: datetime,
    ends_at: datetime | None,
    location: str | None,
    description: str | None,
    all_day: bool,
) -> str:
    """Build a deterministic fallback ID for feeds that omit or blank out IDs."""

    digest = hashlib.blake2s(digest_size=12)
    for chunk in (
        summary,
        starts_at.isoformat(),
        ends_at.isoformat() if ends_at is not None else "",
        location or "",
        description or "",
        "1" if all_day else "0",
    ):
        digest.update(chunk.encode("utf-8"))
        digest.update(b"\x1f")
    return f"evt_{digest.hexdigest()}"


def _normalize_event_id(
    value: object,
    *,
    summary: str,
    starts_at: datetime,
    ends_at: datetime | None,
    location: str | None,
    description: str | None,
    all_day: bool,
) -> str:
    """Validate ``event_id`` and generate a deterministic fallback when blank."""

    if value is None:
        return _make_fallback_event_id(
            summary=summary,
            starts_at=starts_at,
            ends_at=ends_at,
            location=location,
            description=description,
            all_day=all_day,
        )
    if not isinstance(value, str):
        raise TypeError("event_id must be a string or None")

    cleaned = _sanitize_single_line(value)
    if not cleaned:
        return _make_fallback_event_id(
            summary=summary,
            starts_at=starts_at,
            ends_at=ends_at,
            location=location,
            description=description,
            all_day=all_day,
        )
    return _validate_text_bytes("event_id", cleaned, max_bytes=_MAX_EVENT_ID_BYTES)


def _coerce_aware_datetime(field_name: str, value: object) -> datetime:
    """Parse either an aware ``datetime`` or an aware ISO-8601 string."""

    if isinstance(value, datetime):
        _require_aware_datetime(field_name, value)
        return value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO-8601 datetime string") from exc
        _require_aware_datetime(field_name, parsed)
        return parsed
    raise TypeError(f"{field_name} must be a datetime instance or ISO-8601 string")


def _extract_zone_name(value: datetime) -> str | None:
    """Return an IANA zone name when available."""

    tzinfo = value.tzinfo
    if tzinfo is None:
        return None
    key = getattr(tzinfo, "key", None)
    if isinstance(key, str) and key:
        return key
    return None


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate keys during stdlib JSON parsing."""

    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _format_iso8601_duration(value: timedelta) -> str:
    """Serialize a non-negative ``timedelta`` as a compact ISO-8601 duration."""

    if value < timedelta(0):
        raise ValueError("duration must not be negative")

    total_microseconds = (
        value.days * 86_400_000_000
        + value.seconds * 1_000_000
        + value.microseconds
    )
    days, remainder = divmod(total_microseconds, 86_400_000_000)
    seconds, micros = divmod(remainder, 1_000_000)

    parts: list[str] = ["P"]
    if days:
        parts.append(f"{days}D")

    if seconds or micros or not days:
        parts.append("T")
        if micros:
            seconds_text = f"{seconds}.{micros:06d}".rstrip("0").rstrip(".")
        else:
            seconds_text = str(seconds)
        parts.append(f"{seconds_text}S")

    return "".join(parts)


@dataclass(frozen=True, slots=True)
class CalendarEvent:
    """Represent one immutable canonical calendar entry.

    Attributes:
        event_id: Stable source identifier or deterministic fallback ID.
        summary: Short user-visible summary text.
        starts_at: Inclusive start instant as a timezone-aware datetime.
        ends_at: Explicit event end when the source provides one.
        location: Optional location text.
        description: Optional longer event description.
        all_day: True when the event is a local-date all-day entry.
    """

    event_id: str
    summary: str
    starts_at: datetime
    ends_at: datetime | None = None
    location: str | None = None
    description: str | None = None
    all_day: bool = False

    def __post_init__(self) -> None:
        """Validate invariants and normalize textual fields."""

        _require_aware_datetime("starts_at", self.starts_at)
        if self.ends_at is not None:
            _require_aware_datetime("ends_at", self.ends_at)
        if not isinstance(self.all_day, bool):
            raise TypeError("all_day must be a bool")

        # BREAKING: invalid all_day events are now rejected during
        # construction instead of being accepted and misrendered later.
        if self.all_day:
            if not _is_local_midnight(self.starts_at):
                raise ValueError("all_day starts_at must be aligned to local midnight")
            if self.ends_at is not None:
                if not _is_local_midnight(self.ends_at):
                    raise ValueError("all_day ends_at must be aligned to local midnight")
                if _to_utc(self.ends_at) <= _to_utc(self.starts_at):
                    raise ValueError("all_day ends_at must be greater than starts_at")
        elif self.ends_at is not None and _to_utc(self.ends_at) < _to_utc(self.starts_at):
            raise ValueError("ends_at must be greater than or equal to starts_at")

        normalized_summary = _normalize_text(
            "summary",
            self.summary,
            required=True,
            multiline=False,
            max_bytes=_MAX_SUMMARY_BYTES,
        )
        assert normalized_summary is not None

        normalized_location = _normalize_text(
            "location",
            self.location,
            required=False,
            multiline=False,
            max_bytes=_MAX_LOCATION_BYTES,
        )
        normalized_description = _normalize_text(
            "description",
            self.description,
            required=False,
            multiline=True,
            max_bytes=_MAX_DESCRIPTION_BYTES,
        )
        normalized_event_id = _normalize_event_id(
            self.event_id,
            summary=normalized_summary,
            starts_at=self.starts_at,
            ends_at=self.ends_at,
            location=normalized_location,
            description=normalized_description,
            all_day=self.all_day,
        )

        object.__setattr__(self, "event_id", normalized_event_id)
        object.__setattr__(self, "summary", normalized_summary)
        object.__setattr__(self, "location", normalized_location)
        object.__setattr__(self, "description", normalized_description)

    @property
    def effective_end(self) -> datetime:
        """Return the end instant used for overlap and duration logic."""

        if self.ends_at is not None:
            return self.ends_at
        if self.all_day:
            return self.starts_at + timedelta(days=1)
        return self.starts_at

    @property
    def duration(self) -> timedelta:
        """Return the effective event duration."""

        return self.effective_end - self.starts_at

    @property
    def has_explicit_end(self) -> bool:
        """Return True when the source provided ``ends_at`` explicitly."""

        return self.ends_at is not None

    def overlaps(self, start_at: datetime, end_at: datetime) -> bool:
        """Check whether the event intersects a half-open time window."""

        _require_aware_datetime("start_at", start_at)
        _require_aware_datetime("end_at", end_at)

        start_utc = _to_utc(start_at)
        end_utc = _to_utc(end_at)
        if end_utc <= start_utc:
            raise ValueError("end_at must be greater than start_at")

        event_start_utc = _to_utc(self.starts_at)
        event_end_utc = _to_utc(self.effective_end)
        if event_end_utc == event_start_utc:
            return start_utc <= event_start_utc < end_utc
        return event_start_utc < end_utc and event_end_utc > start_utc

    def as_dict(
        self,
        *,
        include_effective_end: bool = False,
        omit_none: bool = False,
    ) -> dict[str, object]:
        """Serialize the event into the legacy calendar-adapter payload shape.

        Args:
            include_effective_end: When True, always emit ``ends_at`` using the
                derived end for point events too.
            omit_none: When True, drop keys whose value is ``None``.

        Returns:
            A JSON-serializable mapping with ISO-8601 timestamps.
        """

        if include_effective_end:
            serialized_end = self.effective_end.isoformat()
        else:
            serialized_end = (
                self.effective_end.isoformat()
                if self.ends_at is not None or self.all_day
                else None
            )

        payload: dict[str, object] = {
            "event_id": self.event_id,
            "summary": self.summary,
            "starts_at": self.starts_at.isoformat(),
            "ends_at": serialized_end,
            "location": self.location,
            "description": self.description,
            "all_day": self.all_day,
        }
        if omit_none:
            return {key: value for key, value in payload.items() if value is not None}
        return payload

    def to_json(
        self,
        *,
        include_effective_end: bool = False,
        omit_none: bool = False,
    ) -> str:
        """Serialize the legacy adapter payload as UTF-8 JSON text."""

        return self.to_json_bytes(
            include_effective_end=include_effective_end,
            omit_none=omit_none,
        ).decode("utf-8")

    def to_json_bytes(
        self,
        *,
        include_effective_end: bool = False,
        omit_none: bool = False,
    ) -> bytes:
        """Serialize the legacy adapter payload as UTF-8 JSON bytes."""

        payload = self.as_dict(
            include_effective_end=include_effective_end,
            omit_none=omit_none,
        )
        if msgspec is not None:
            return msgspec.json.encode(payload)
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

    def as_jscalendar(self) -> dict[str, object]:
        """Export a minimal JSCalendar-compatible Event object.

        The canonical model stores aware datetimes, while JSCalendar stores a
        local date-time plus an explicit time zone. When the original tzinfo is
        not backed by an IANA zone name, this method falls back to ``Etc/UTC``
        so the export remains unambiguous.
        """

        starts_at = self.starts_at
        zone_name = _extract_zone_name(starts_at)
        if zone_name is None:
            starts_at = starts_at.astimezone(UTC)
            zone_name = "Etc/UTC"

        payload: dict[str, object] = {
            "@type": "Event",
            "uid": self.event_id,
            "title": self.summary,
            "timeZone": zone_name,
            "start": starts_at.replace(tzinfo=None).isoformat(),
            "duration": _format_iso8601_duration(self.duration),
        }
        if self.all_day:
            payload["showWithoutTime"] = True
        if self.location:
            payload["locations"] = {
                "loc-1": {
                    "@type": "Location",
                    "name": self.location,
                }
            }
        if self.description:
            payload["description"] = self.description
        return payload

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        *,
        allow_unknown_fields: bool = False,
    ) -> Self:
        """Build a ``CalendarEvent`` from a legacy adapter payload mapping."""

        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")

        unknown_fields = set(payload) - _ALLOWED_FIELDS
        if unknown_fields and not allow_unknown_fields:
            unknown = ", ".join(sorted(unknown_fields))
            raise ValueError(f"payload contains unknown fields: {unknown}")

        starts_at = _coerce_aware_datetime("starts_at", payload.get("starts_at"))
        ends_raw = payload.get("ends_at")
        ends_at = None if ends_raw is None else _coerce_aware_datetime("ends_at", ends_raw)

        all_day = payload.get("all_day", False)
        if not isinstance(all_day, bool):
            raise TypeError("all_day must be a bool")

        return cls(
            event_id=payload.get("event_id"),  # type: ignore[arg-type]
            summary=payload.get("summary", ""),  # type: ignore[arg-type]
            starts_at=starts_at,
            ends_at=ends_at,
            location=payload.get("location"),  # type: ignore[arg-type]
            description=payload.get("description"),  # type: ignore[arg-type]
            all_day=all_day,
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray | memoryview) -> Self:
        """Build a ``CalendarEvent`` from a JSON-encoded legacy payload."""

        return cls.from_json_bytes(payload)

    @classmethod
    def from_json_bytes(cls, payload: str | bytes | bytearray | memoryview) -> Self:
        """Build a ``CalendarEvent`` from UTF-8 JSON bytes or text."""

        if isinstance(payload, str):
            payload_bytes = payload.encode("utf-8")
        elif isinstance(payload, memoryview):
            payload_bytes = payload.tobytes()
        else:
            payload_bytes = bytes(payload)

        if msgspec is not None:
            parsed = msgspec.json.decode(payload_bytes, type=_CalendarEventPayload)
            return cls(
                event_id=parsed.event_id,  # type: ignore[arg-type]
                summary=parsed.summary,
                starts_at=parsed.starts_at,
                ends_at=parsed.ends_at,
                location=parsed.location,
                description=parsed.description,
                all_day=parsed.all_day,
            )

        decoded = json.loads(payload_bytes, object_pairs_hook=_strict_json_object)
        if not isinstance(decoded, dict):
            raise TypeError("payload must decode to an object")
        return cls.from_mapping(decoded)
