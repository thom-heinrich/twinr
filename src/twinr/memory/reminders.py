"""Persist and render Twinr reminders and timers.

This module owns the file-backed reminder store used by runtime, prompt
assembly, automations, and the web dashboard. It keeps reminder state bounded,
timezone-aware, and safe across concurrent writers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import errno
import fcntl
import json
import math
import os
import stat
import tempfile
import threading
from contextlib import contextmanager
from typing import Iterator
from uuid import uuid4

_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


class ReminderStoreError(RuntimeError):
    """Raise when the reminder store cannot be read or written safely."""

    pass


class ReminderStoreCorruptionError(ReminderStoreError):
    """Raise when reminder-store contents are structurally invalid."""

    pass


class ReminderStoreSecurityError(ReminderStoreError):
    """Raise when the reminder-store path or file is not safe to use."""

    pass


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


def _safe_int(value: object, *, default: int, minimum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def _safe_float(value: object, *, default: float, minimum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return max(minimum, parsed)


def _path_lock_for(path: Path) -> threading.RLock:
    key = os.path.abspath(os.fspath(path))
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[key] = lock
        return lock


def resolve_timezone(timezone_name: str | None) -> ZoneInfo:
    """Resolve Twinr's configured timezone with a safe UTC fallback."""

    try:
        return ZoneInfo((timezone_name or "").strip() or "Europe/Berlin")
    except Exception:
        return ZoneInfo("UTC")


def _localize_naive_datetime(value: datetime, *, zone: ZoneInfo, field_name: str) -> datetime:
    # AUDIT-FIX(#1): Reject ambiguous and nonexistent wall-clock times instead of silently creating invalid reminders.
    candidate_fold_0 = value.replace(tzinfo=zone, fold=0)
    candidate_fold_1 = value.replace(tzinfo=zone, fold=1)
    roundtrip_fold_0 = candidate_fold_0.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None)
    roundtrip_fold_1 = candidate_fold_1.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None)
    valid_fold_0 = roundtrip_fold_0 == value
    valid_fold_1 = roundtrip_fold_1 == value

    if valid_fold_0 and valid_fold_1:
        if candidate_fold_0.utcoffset() != candidate_fold_1.utcoffset():
            raise ValueError(f"{field_name} is ambiguous in timezone {zone.key}; include an explicit UTC offset")
        return candidate_fold_0
    if valid_fold_0:
        return candidate_fold_0
    if valid_fold_1:
        return candidate_fold_1
    raise ValueError(f"{field_name} falls into a nonexistent local time in timezone {zone.key}; include an explicit UTC offset")


def _coerce_datetime(value: datetime, *, timezone_name: str | None, field_name: str) -> datetime:
    # AUDIT-FIX(#8): Normalize externally supplied datetimes so mixed naive/aware values cannot break ordering or retries.
    zone = resolve_timezone(timezone_name)
    if value.tzinfo is None:
        return _localize_naive_datetime(value, zone=zone, field_name=field_name)
    return value.astimezone(zone)


def now_in_timezone(timezone_name: str | None) -> datetime:
    """Return the current aware datetime in the configured timezone."""

    return datetime.now(resolve_timezone(timezone_name))


def parse_due_at(value: str, *, timezone_name: str | None) -> datetime:
    """Parse one reminder due timestamp into the configured timezone.

    Args:
        value: ISO 8601 datetime text. Naive values are interpreted in the
            configured timezone after DST validation.
        timezone_name: Target timezone used for normalization.

    Returns:
        An aware ``datetime`` localized to the configured timezone.

    Raises:
        ValueError: If ``value`` is empty or not a valid ISO 8601 datetime.
    """

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
        # AUDIT-FIX(#1): Validate DST transitions for naive local datetimes before assigning the store timezone.
        return _localize_naive_datetime(parsed, zone=zone, field_name="due_at")
    return parsed.astimezone(zone)


def format_due_label(value: datetime, *, timezone_name: str | None) -> str:
    """Format one due timestamp for user-facing reminder context."""

    # AUDIT-FIX(#8): Accept only normalized aware timestamps for rendering to avoid hidden local-time drift.
    localized = _coerce_datetime(value, timezone_name=timezone_name, field_name="due_at")
    return localized.strftime("%A, %d.%m.%Y %H:%M")


@dataclass(frozen=True, slots=True)
class ReminderEntry:
    """Store one reminder or timer entry.

    Attributes:
        reminder_id: Stable identifier for updates and delivery tracking.
        kind: Normalized reminder category such as ``reminder`` or ``timer``.
        summary: Short user-facing reminder text.
        due_at: Aware due timestamp in the configured timezone.
        details: Optional extra detail shown in prompt context or UI.
        created_at: Creation timestamp for persistence and ordering.
        updated_at: Last mutation timestamp.
        source: Provenance label for the entry.
        original_request: Optional original user request text.
        delivery_attempts: Number of delivery reservations attempted so far.
        last_attempted_at: Timestamp of the latest delivery attempt.
        next_attempt_at: Earliest time the reminder may be retried.
        delivered_at: Timestamp when the reminder was confirmed delivered.
        last_error: Last bounded delivery error message, if any.
    """

    reminder_id: str
    kind: str
    summary: str
    due_at: datetime
    details: str | None = None
    # AUDIT-FIX(#11): Use timezone-aware defaults so ad-hoc ReminderEntry construction cannot trigger naive/aware comparison failures.
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "tool"
    original_request: str | None = None
    delivery_attempts: int = 0
    last_attempted_at: datetime | None = None
    next_attempt_at: datetime | None = None
    delivered_at: datetime | None = None
    last_error: str | None = None

    @property
    def delivered(self) -> bool:
        """Return whether this reminder has already been delivered."""

        return self.delivered_at is not None


class ReminderStore:
    """Manage Twinr's file-backed reminder and timer state.

    The store keeps reminder entries bounded, serializes concurrent access with
    thread and file locks, and renders short reminder context for prompt
    assembly and dashboard consumers.

    Args:
        path: JSON file used for persisted reminder state.
        timezone_name: Timezone used for parsing, ordering, and rendering.
        retry_delay_s: Delay applied after failed delivery attempts.
        max_entries: Maximum number of entries retained in the store.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        timezone_name: str | None = None,
        retry_delay_s: float = 90.0,
        max_entries: int = 48,
    ) -> None:
        raw_path = Path(path).expanduser()
        normalized_path = Path(os.path.abspath(os.fspath(raw_path)))
        if not normalized_path.name:
            raise ValueError("path must point to a file")

        self.path = normalized_path
        # AUDIT-FIX(#4): Use a dedicated lock file plus a per-path in-process lock to serialize all file-backed state mutations.
        self._lock_path = self.path.with_name(f".{self.path.name}.lock")
        self._thread_lock = _path_lock_for(self.path)
        self.timezone_name = (timezone_name or "Europe/Berlin").strip() or "Europe/Berlin"
        # AUDIT-FIX(#6): Invalid .env values must degrade to safe defaults instead of crashing startup.
        self.retry_delay_s = _safe_float(retry_delay_s, default=90.0, minimum=5.0)
        self.max_entries = _safe_int(max_entries, default=48, minimum=8)

    def load_entries(self) -> tuple[ReminderEntry, ...]:
        """Load all persisted reminder entries in normalized order."""

        with self._locked_store():
            return self._load_entries_locked()

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
        """Create or update one pending reminder entry.

        Args:
            due_at: ISO 8601 due timestamp.
            summary: Short reminder text.
            details: Optional longer reminder detail.
            kind: Reminder category label.
            source: Provenance label for the reminder request.
            original_request: Optional original user request text.

        Returns:
            The created or updated ``ReminderEntry``.

        Raises:
            ValueError: If the summary is empty or the due timestamp is invalid
                or already in the past.
        """

        clean_summary = _normalize_text(summary, limit=220)
        clean_details = _normalize_text(details, limit=420) or None
        clean_kind = _slugify(kind, fallback="reminder")
        clean_source = _normalize_text(source, limit=80) or "tool"
        clean_original_request = _normalize_text(original_request, limit=220) or None
        if not clean_summary:
            raise ValueError("summary must not be empty")

        parsed_due_at = parse_due_at(due_at, timezone_name=self.timezone_name)
        now = now_in_timezone(self.timezone_name)
        if parsed_due_at < now - timedelta(seconds=60):
            raise ValueError("due_at must not be in the past")

        with self._locked_store():
            entries = list(self._load_entries_locked())
            fingerprint = (clean_kind, clean_summary.casefold(), parsed_due_at.isoformat())
            for index, existing in enumerate(entries):
                existing_fingerprint = (
                    existing.kind,
                    existing.summary.casefold(),
                    existing.due_at.isoformat(),
                )
                if existing.delivered or existing_fingerprint != fingerprint:
                    continue
                updated = replace(
                    existing,
                    details=clean_details if clean_details is not None else existing.details,
                    updated_at=now,
                    source=clean_source,
                    original_request=clean_original_request or existing.original_request,
                    last_error=None,
                )
                entries[index] = updated
                self._write_entries_locked(tuple(entries))
                return updated

            created_at = now
            entry = ReminderEntry(
                # AUDIT-FIX(#9): Generate reminder IDs in real UTC and add entropy so concurrent schedules cannot collide.
                reminder_id=f"REM-{created_at.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:8]}",
                kind=clean_kind,
                summary=clean_summary,
                due_at=parsed_due_at,
                details=clean_details,
                created_at=created_at,
                updated_at=created_at,
                source=clean_source,
                original_request=clean_original_request,
            )
            entries.append(entry)
            self._write_entries_locked(tuple(entries))
            return entry

    def reserve_due(self, *, now: datetime | None = None, limit: int = 1) -> tuple[ReminderEntry, ...]:
        """Reserve due reminders for one delivery attempt.

        Args:
            now: Current time override used for tests or controlled delivery.
            limit: Maximum number of reminders to reserve.

        Returns:
            A tuple of due reminders with delivery-attempt metadata updated.
        """

        # AUDIT-FIX(#10): Respect zero and negative limits instead of silently reserving one reminder.
        effective_limit = _safe_int(limit, default=1, minimum=0)
        if effective_limit == 0:
            return ()
        current_time = _coerce_datetime(
            now or now_in_timezone(self.timezone_name),
            timezone_name=self.timezone_name,
            field_name="now",
        )
        with self._locked_store():
            entries = list(self._load_entries_locked())
            selected: list[ReminderEntry] = []
            changed = False
            for index, entry in enumerate(entries):
                if len(selected) >= effective_limit:
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
                self._write_entries_locked(tuple(entries))
            return tuple(selected)

    def peek_due(self, *, now: datetime | None = None, limit: int = 1) -> tuple[ReminderEntry, ...]:
        """Return due reminders without mutating delivery state."""

        # AUDIT-FIX(#10): Respect zero and negative limits instead of silently returning one reminder.
        effective_limit = _safe_int(limit, default=1, minimum=0)
        if effective_limit == 0:
            return ()
        current_time = _coerce_datetime(
            now or now_in_timezone(self.timezone_name),
            timezone_name=self.timezone_name,
            field_name="now",
        )
        selected: list[ReminderEntry] = []
        with self._locked_store():
            for entry in self._load_entries_locked():
                if len(selected) >= effective_limit:
                    break
                if entry.delivered:
                    continue
                if entry.due_at > current_time:
                    continue
                if entry.next_attempt_at is not None and entry.next_attempt_at > current_time:
                    continue
                selected.append(entry)
        return tuple(selected)

    def mark_delivered(self, reminder_id: str, *, delivered_at: datetime | None = None) -> ReminderEntry:
        """Mark one reminder as successfully delivered.

        Args:
            reminder_id: Reminder identifier to update.
            delivered_at: Delivery timestamp override.

        Returns:
            The updated ``ReminderEntry``.

        Raises:
            KeyError: If ``reminder_id`` is unknown.
        """

        current_time = _coerce_datetime(
            delivered_at or now_in_timezone(self.timezone_name),
            timezone_name=self.timezone_name,
            field_name="delivered_at",
        )
        with self._locked_store():
            entries = list(self._load_entries_locked())
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
                self._write_entries_locked(tuple(entries))
                return delivered
        raise KeyError(f"Unknown reminder_id: {reminder_id}")

    def mark_failed(
        self,
        reminder_id: str,
        *,
        error: str,
        failed_at: datetime | None = None,
    ) -> ReminderEntry:
        """Record one failed reminder delivery attempt.

        Args:
            reminder_id: Reminder identifier to update.
            error: Short bounded delivery error message.
            failed_at: Failure timestamp override.

        Returns:
            The updated ``ReminderEntry``.

        Raises:
            KeyError: If ``reminder_id`` is unknown.
        """

        current_time = _coerce_datetime(
            failed_at or now_in_timezone(self.timezone_name),
            timezone_name=self.timezone_name,
            field_name="failed_at",
        )
        with self._locked_store():
            entries = list(self._load_entries_locked())
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
                self._write_entries_locked(tuple(entries))
                return failed
        raise KeyError(f"Unknown reminder_id: {reminder_id}")

    def delete(self, reminder_id: str) -> ReminderEntry:
        """Delete one reminder entry by identifier.

        Args:
            reminder_id: Reminder identifier to remove.

        Returns:
            The removed ``ReminderEntry``.

        Raises:
            KeyError: If ``reminder_id`` is unknown.
        """

        with self._locked_store():
            entries = list(self._load_entries_locked())
            for index, entry in enumerate(entries):
                if entry.reminder_id != reminder_id:
                    continue
                removed = entries.pop(index)
                self._write_entries_locked(tuple(entries))
                return removed
        raise KeyError(f"Unknown reminder_id: {reminder_id}")

    def render_context(self, *, limit: int = 8) -> str | None:
        """Render pending reminders into prompt-context text.

        Args:
            limit: Maximum number of pending reminders to include.

        Returns:
            A short reminder summary block, or ``None`` when nothing is
            pending.
        """

        effective_limit = _safe_int(limit, default=8, minimum=1)
        with self._locked_store():
            pending = [entry for entry in self._load_entries_locked() if not entry.delivered]
        if not pending:
            return None
        lines = ["Scheduled reminders and timers:"]
        for entry in pending[:effective_limit]:
            line = f"- [{entry.kind}] {format_due_label(entry.due_at, timezone_name=self.timezone_name)} — {entry.summary}"
            if entry.details and entry.details.casefold() != entry.summary.casefold():
                line += f" Details: {entry.details}"
            lines.append(line)
        return "\n".join(lines)

    def _entry_from_payload(self, item: dict[str, object]) -> ReminderEntry | None:
        summary = _normalize_text(item.get("summary"), limit=220)
        reminder_id = _normalize_text(str(item.get("reminder_id", "")).strip(), limit=96)
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
            source=_normalize_text(item.get("source"), limit=80) or "tool",
            original_request=_normalize_text(item.get("original_request"), limit=220) or None,
            # AUDIT-FIX(#7): Malformed numeric payload fields must not crash store loading.
            delivery_attempts=_safe_int(item.get("delivery_attempts", 0), default=0, minimum=0),
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

    def _load_entries_locked(self) -> tuple[ReminderEntry, ...]:
        payload = self._read_payload_locked()
        if payload is None:
            return ()
        items = payload.get("entries", [])
        if not isinstance(items, list):
            # AUDIT-FIX(#7): Treat structurally invalid JSON as corruption instead of crashing with AttributeError later.
            raise ReminderStoreCorruptionError(f"Reminder store at {self.path} contains an invalid 'entries' payload")
        entries: list[ReminderEntry] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            entry = self._entry_from_payload(item)
            if entry is not None:
                entries.append(entry)
        return tuple(self._sorted_entries(entries))

    def _read_payload_locked(self) -> dict[str, object] | None:
        try:
            raw_payload = self._read_text_file(self.path)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ReminderStoreError(f"Unable to read reminder store at {self.path}") from exc
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            # AUDIT-FIX(#3): Never treat corrupted JSON as an empty store because that would erase pending reminders on the next write.
            raise ReminderStoreCorruptionError(
                f"Reminder store at {self.path} is corrupted and must be repaired before modification"
            ) from exc
        if not isinstance(payload, dict):
            # AUDIT-FIX(#7): Top-level JSON must be validated explicitly because json.loads may return a list or scalar.
            raise ReminderStoreCorruptionError(f"Reminder store at {self.path} must contain a JSON object")
        return payload

    def _write_entries_locked(self, entries: tuple[ReminderEntry, ...]) -> None:
        normalized_entries = self._trim_entries(self._sorted_entries(entries))
        payload = {
            "updated_at": now_in_timezone(self.timezone_name).isoformat(),
            "entries": [self._entry_to_payload(entry) for entry in normalized_entries],
        }
        self._write_payload_locked(payload)

    def _write_payload_locked(self, payload: dict[str, object]) -> None:
        self._ensure_safe_storage_location()
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        temp_path: Path | None = None
        try:
            file_descriptor, temp_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=os.fspath(self.path.parent),
                text=True,
            )
            temp_path = Path(temp_name)
            os.fchmod(file_descriptor, 0o600)
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            # AUDIT-FIX(#2,#3): Persist through a same-directory atomic replace so crashes and symlink swaps cannot truncate the live store.
            os.replace(temp_path, self.path)
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as exc:
            raise ReminderStoreError(f"Unable to persist reminder store at {self.path}") from exc
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

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
            # AUDIT-FIX(#5): Never evict pending reminders just to satisfy the history cap; missing a future reminder is worse than a larger file.
            return pending
        remaining = self.max_entries - len(pending)
        if remaining <= 0:
            return pending
        return pending + delivered[-remaining:]

    @contextmanager
    def _locked_store(self) -> Iterator[None]:
        self._ensure_safe_storage_location()
        with self._thread_lock:
            lock_fd = self._open_lock_file()
            try:
                # AUDIT-FIX(#4): Serialize every load/mutate/write transaction under an OS-level file lock.
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                yield
            finally:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)

    def _ensure_safe_storage_location(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ReminderStoreError(f"Unable to prepare reminder store directory {self.path.parent}") from exc
        # AUDIT-FIX(#2): Validate the entire storage location before any read or write.
        self._assert_safe_directory_chain(self.path.parent)
        self._assert_safe_regular_file(self.path, allow_missing=True)
        self._assert_safe_regular_file(self._lock_path, allow_missing=True)

    def _assert_safe_directory_chain(self, directory: Path) -> None:
        current = directory
        while True:
            try:
                info = current.lstat()
            except OSError as exc:
                raise ReminderStoreError(f"Unable to inspect reminder store directory {current}") from exc
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise ReminderStoreSecurityError(f"Reminder store directory {current} is not a safe directory")
            if current.parent == current:
                break
            current = current.parent

    def _assert_safe_regular_file(self, path: Path, *, allow_missing: bool) -> None:
        try:
            info = path.lstat()
        except FileNotFoundError:
            if allow_missing:
                return
            raise
        except OSError as exc:
            raise ReminderStoreError(f"Unable to inspect reminder store path {path}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ReminderStoreSecurityError(f"Reminder store path {path} is not a regular file")

    def _open_lock_file(self) -> int:
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            return os.open(self._lock_path, flags, 0o600)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ReminderStoreSecurityError(
                    f"Reminder store lock path {self._lock_path} must not be a symlink"
                ) from exc
            raise ReminderStoreError(f"Unable to open reminder store lock file {self._lock_path}") from exc

    def _read_text_file(self, path: Path) -> str:
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            file_descriptor = os.open(path, flags)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ReminderStoreSecurityError(f"Reminder store path {path} must not be a symlink") from exc
            raise
        with os.fdopen(file_descriptor, "r", encoding="utf-8") as handle:
            return handle.read()
