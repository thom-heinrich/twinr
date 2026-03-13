from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import json


def _normalize_text(value: str | None, *, limit: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def _slugify(value: str | None, *, fallback: str) -> str:
    text = "".join(character if character.isalnum() else "_" for character in str(value or "").strip().lower())
    normalized = "_".join(part for part in text.split("_") if part)
    return normalized or fallback


def resolve_timezone(timezone_name: str | None) -> ZoneInfo:
    try:
        return ZoneInfo((timezone_name or "").strip() or "Europe/Berlin")
    except Exception:
        return ZoneInfo("UTC")


def now_in_timezone(timezone_name: str | None) -> datetime:
    return datetime.now(resolve_timezone(timezone_name))


def parse_due_at(value: str, *, timezone_name: str | None) -> datetime:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError("due_at must not be empty")
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("due_at must be a valid ISO 8601 datetime") from exc

    zone = resolve_timezone(timezone_name)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=zone)
    return parsed.astimezone(zone)


def format_due_label(value: datetime, *, timezone_name: str | None) -> str:
    localized = value.astimezone(resolve_timezone(timezone_name))
    return localized.strftime("%A, %d.%m.%Y %H:%M")


@dataclass(frozen=True, slots=True)
class ReminderEntry:
    reminder_id: str
    kind: str
    summary: str
    due_at: datetime
    details: str | None = None
    created_at: datetime = datetime.min
    updated_at: datetime = datetime.min
    source: str = "tool"
    original_request: str | None = None
    delivery_attempts: int = 0
    last_attempted_at: datetime | None = None
    next_attempt_at: datetime | None = None
    delivered_at: datetime | None = None
    last_error: str | None = None

    @property
    def delivered(self) -> bool:
        return self.delivered_at is not None


class ReminderStore:
    def __init__(
        self,
        path: str | Path,
        *,
        timezone_name: str | None = None,
        retry_delay_s: float = 90.0,
        max_entries: int = 48,
    ) -> None:
        self.path = Path(path)
        self.timezone_name = timezone_name or "Europe/Berlin"
        self.retry_delay_s = max(5.0, float(retry_delay_s))
        self.max_entries = max(8, int(max_entries))

    def load_entries(self) -> tuple[ReminderEntry, ...]:
        if not self.path.exists():
            return ()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ()
        items = payload.get("entries", [])
        if not isinstance(items, list):
            return ()
        entries: list[ReminderEntry] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            entry = self._entry_from_payload(item)
            if entry is not None:
                entries.append(entry)
        return tuple(self._sorted_entries(entries))

    def schedule(
        self,
        *,
        due_at: str,
        summary: str,
        details: str | None = None,
        kind: str = "reminder",
        source: str = "tool",
        original_request: str | None = None,
    ) -> ReminderEntry:
        clean_summary = _normalize_text(summary, limit=220)
        clean_details = _normalize_text(details, limit=420) or None
        clean_kind = _slugify(kind, fallback="reminder")
        clean_original_request = _normalize_text(original_request, limit=220) or None
        if not clean_summary:
            raise ValueError("summary must not be empty")

        parsed_due_at = parse_due_at(due_at, timezone_name=self.timezone_name)
        now = now_in_timezone(self.timezone_name)
        if parsed_due_at < now - timedelta(seconds=60):
            raise ValueError("due_at must not be in the past")

        entries = list(self.load_entries())
        fingerprint = (clean_kind, clean_summary.lower(), parsed_due_at.isoformat())
        for index, existing in enumerate(entries):
            existing_fingerprint = (
                existing.kind,
                existing.summary.lower(),
                existing.due_at.isoformat(),
            )
            if existing.delivered or existing_fingerprint != fingerprint:
                continue
            updated = replace(
                existing,
                details=clean_details or existing.details,
                updated_at=now,
                source=source,
                original_request=clean_original_request or existing.original_request,
                last_error=None,
            )
            entries[index] = updated
            self._write_entries(tuple(entries))
            return updated

        created_at = now
        entry = ReminderEntry(
            reminder_id=f"REM-{created_at.strftime('%Y%m%dT%H%M%S%fZ')}",
            kind=clean_kind,
            summary=clean_summary,
            due_at=parsed_due_at,
            details=clean_details,
            created_at=created_at,
            updated_at=created_at,
            source=source,
            original_request=clean_original_request,
        )
        entries.append(entry)
        self._write_entries(tuple(entries))
        return entry

    def reserve_due(self, *, now: datetime | None = None, limit: int = 1) -> tuple[ReminderEntry, ...]:
        current_time = now or now_in_timezone(self.timezone_name)
        entries = list(self.load_entries())
        selected: list[ReminderEntry] = []
        changed = False
        for index, entry in enumerate(entries):
            if len(selected) >= max(1, limit):
                break
            if entry.delivered:
                continue
            if entry.due_at > current_time:
                continue
            if entry.next_attempt_at is not None and entry.next_attempt_at > current_time:
                continue
            reserved = replace(
                entry,
                delivery_attempts=entry.delivery_attempts + 1,
                last_attempted_at=current_time,
                next_attempt_at=current_time + timedelta(seconds=self.retry_delay_s),
                updated_at=current_time,
                last_error=None,
            )
            entries[index] = reserved
            selected.append(reserved)
            changed = True
        if changed:
            self._write_entries(tuple(entries))
        return tuple(selected)

    def mark_delivered(self, reminder_id: str, *, delivered_at: datetime | None = None) -> ReminderEntry:
        current_time = delivered_at or now_in_timezone(self.timezone_name)
        entries = list(self.load_entries())
        for index, entry in enumerate(entries):
            if entry.reminder_id != reminder_id:
                continue
            delivered = replace(
                entry,
                delivered_at=current_time,
                updated_at=current_time,
                next_attempt_at=None,
                last_error=None,
            )
            entries[index] = delivered
            self._write_entries(tuple(entries))
            return delivered
        raise KeyError(f"Unknown reminder_id: {reminder_id}")

    def mark_failed(
        self,
        reminder_id: str,
        *,
        error: str,
        failed_at: datetime | None = None,
    ) -> ReminderEntry:
        current_time = failed_at or now_in_timezone(self.timezone_name)
        entries = list(self.load_entries())
        for index, entry in enumerate(entries):
            if entry.reminder_id != reminder_id:
                continue
            failed = replace(
                entry,
                updated_at=current_time,
                last_error=_normalize_text(error, limit=220) or "unknown error",
                next_attempt_at=current_time + timedelta(seconds=self.retry_delay_s),
            )
            entries[index] = failed
            self._write_entries(tuple(entries))
            return failed
        raise KeyError(f"Unknown reminder_id: {reminder_id}")

    def render_context(self, *, limit: int = 8) -> str | None:
        pending = [entry for entry in self.load_entries() if not entry.delivered]
        if not pending:
            return None
        lines = ["Scheduled reminders and timers:"]
        for entry in pending[:limit]:
            line = f"- [{entry.kind}] {format_due_label(entry.due_at, timezone_name=self.timezone_name)} — {entry.summary}"
            if entry.details and entry.details.lower() != entry.summary.lower():
                line += f" Details: {entry.details}"
            lines.append(line)
        return "\n".join(lines)

    def _entry_from_payload(self, item: dict[str, object]) -> ReminderEntry | None:
        summary = _normalize_text(item.get("summary"), limit=220)
        reminder_id = str(item.get("reminder_id", "")).strip()
        due_at_raw = str(item.get("due_at", "")).strip()
        if not summary or not reminder_id or not due_at_raw:
            return None
        try:
            due_at = parse_due_at(due_at_raw, timezone_name=self.timezone_name)
        except ValueError:
            return None
        return ReminderEntry(
            reminder_id=reminder_id,
            kind=_slugify(item.get("kind"), fallback="reminder"),
            summary=summary,
            due_at=due_at,
            details=_normalize_text(item.get("details"), limit=420) or None,
            created_at=self._parse_timestamp(item.get("created_at")) or due_at,
            updated_at=self._parse_timestamp(item.get("updated_at")) or due_at,
            source=str(item.get("source", "tool")).strip() or "tool",
            original_request=_normalize_text(item.get("original_request"), limit=220) or None,
            delivery_attempts=max(0, int(item.get("delivery_attempts", 0) or 0)),
            last_attempted_at=self._parse_timestamp(item.get("last_attempted_at")),
            next_attempt_at=self._parse_timestamp(item.get("next_attempt_at")),
            delivered_at=self._parse_timestamp(item.get("delivered_at")),
            last_error=_normalize_text(item.get("last_error"), limit=220) or None,
        )

    def _parse_timestamp(self, value: object) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return parse_due_at(text, timezone_name=self.timezone_name)
        except ValueError:
            return None

    def _write_entries(self, entries: tuple[ReminderEntry, ...]) -> None:
        normalized_entries = self._trim_entries(self._sorted_entries(entries))
        payload = {
            "updated_at": now_in_timezone(self.timezone_name).isoformat(),
            "entries": [self._entry_to_payload(entry) for entry in normalized_entries],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _entry_to_payload(self, entry: ReminderEntry) -> dict[str, object]:
        return {
            "reminder_id": entry.reminder_id,
            "kind": entry.kind,
            "summary": entry.summary,
            "details": entry.details,
            "due_at": entry.due_at.isoformat(),
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat(),
            "source": entry.source,
            "original_request": entry.original_request,
            "delivery_attempts": entry.delivery_attempts,
            "last_attempted_at": entry.last_attempted_at.isoformat() if entry.last_attempted_at else None,
            "next_attempt_at": entry.next_attempt_at.isoformat() if entry.next_attempt_at else None,
            "delivered_at": entry.delivered_at.isoformat() if entry.delivered_at else None,
            "last_error": entry.last_error,
        }

    def _sorted_entries(self, entries: tuple[ReminderEntry, ...] | list[ReminderEntry]) -> list[ReminderEntry]:
        return sorted(
            entries,
            key=lambda entry: (
                entry.delivered,
                entry.due_at,
                entry.created_at,
            ),
        )

    def _trim_entries(self, entries: list[ReminderEntry]) -> list[ReminderEntry]:
        if len(entries) <= self.max_entries:
            return entries
        pending = [entry for entry in entries if not entry.delivered]
        delivered = [entry for entry in entries if entry.delivered]
        if len(pending) >= self.max_entries:
            return pending[: self.max_entries]
        remaining = self.max_entries - len(pending)
        return pending + delivered[-remaining:]
