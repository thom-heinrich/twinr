from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import logging
import math
import os
from pathlib import Path
import tempfile
from threading import Lock
from typing import Callable, Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb.client import chonkydb_data_path
from twinr.memory.longterm.ontology import normalize_memory_sensitivity
from twinr.memory.longterm.models import LONGTERM_MEMORY_SENSITIVITY, LongTermProactiveCandidateV1, LongTermProactivePlanV1


_STATE_SCHEMA = "twinr_memory_proactive_state"
_STATE_VERSION = 1
_DEFAULT_HISTORY_LIMIT = 128
_MIN_HISTORY_LIMIT = 16
_MAX_HISTORY_LIMIT = 4096
_MAX_AUDIT_TEXT_CHARS = 512

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_text(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


# AUDIT-FIX(#8): Clamp config/state numerics to safe defaults instead of raising on malformed .env or JSON.
def _coerce_int(value: object, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# AUDIT-FIX(#8): Reject negative counters and absurd history limits so malformed state cannot blow up storage behavior.
def _coerce_non_negative_int(value: object, *, default: int = 0) -> int:
    return max(0, _coerce_int(value, default=default))


# AUDIT-FIX(#8): Keep the on-disk history bounded for RPi storage and malformed config inputs.
def _coerce_history_limit(value: object) -> int:
    limit = _coerce_int(value, default=_DEFAULT_HISTORY_LIMIT)
    return max(_MIN_HISTORY_LIMIT, min(_MAX_HISTORY_LIMIT, limit))


# AUDIT-FIX(#4): Normalize cooldown values before datetime math.
# AUDIT-FIX(#8): Ignore NaN/inf garbage from config instead of crashing.
def _coerce_non_negative_seconds(value: object, *, default: float = 0.0) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(seconds):
        return default
    return max(0.0, seconds)


# AUDIT-FIX(#1): Validate candidate confidence before policy checks.
# AUDIT-FIX(#8): Malformed numeric config/state should fall back safely.
def _coerce_confidence(value: object, *, default: float = 0.0) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(confidence):
        return default
    return confidence


# AUDIT-FIX(#9): Bound audit text persisted to disk to reduce privacy spill and unbounded state growth.
def _truncate_text(value: str | None, *, max_chars: int) -> str | None:
    clean = _normalize_text(value)
    if not clean:
        return None
    if len(clean) <= max_chars:
        return clean
    if max_chars <= 1:
        return "…"
    return clean[: max_chars - 1].rstrip() + "…"


# AUDIT-FIX(#4): Persist and compare only timezone-aware UTC values to avoid naive/aware crashes and DST weirdness.
def _coerce_datetime(
    value: datetime,
    *,
    default_tz: timezone | ZoneInfo = timezone.utc,
) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        value = value.replace(tzinfo=default_tz)
    return value.astimezone(timezone.utc)


# AUDIT-FIX(#4): Parse legacy ISO timestamps defensively and normalize them to UTC.
def _parse_datetime(
    value: object,
    *,
    field_name: str,
    default_tz: timezone | ZoneInfo = timezone.utc,
) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = _normalize_text(str(value)) if value is not None else ""
        if not text:
            raise ValueError(f"{field_name} is required.")
        parsed = datetime.fromisoformat(text)
    return _coerce_datetime(parsed, default_tz=default_tz)


# AUDIT-FIX(#2): Partially corrupt state should not crash the whole feature.
# AUDIT-FIX(#4): Optional timestamps are normalized defensively.
def _parse_optional_datetime(
    value: object,
    *,
    field_name: str,
    default_tz: timezone | ZoneInfo = timezone.utc,
) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        return _parse_datetime(value, field_name=field_name, default_tz=default_tz)
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid %s in proactive state payload.", field_name)
        return None


def _normalized_candidate_id(value: object) -> str:
    return _normalize_text(str(value))


# AUDIT-FIX(#7): Accept only list/tuple memory ID collections so malformed strings do not turn into character soup.
def _coerce_source_memory_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(
        memory_id
        for memory_id in (_normalize_text(str(item)) for item in value)
        if memory_id
    )


def _lexical_absolute_path(path: Path) -> Path:
    if ".." in path.parts:
        raise ValueError(f"Refusing proactive state path containing parent traversal: {path}")
    return Path(os.path.abspath(os.fspath(path.expanduser())))


# AUDIT-FIX(#5): Reject symlinked files/parents so file-backed state cannot be redirected to arbitrary locations.
def _assert_safe_state_file_path(path: Path) -> Path:
    normalized = _lexical_absolute_path(path)
    current = normalized
    while True:
        if current.is_symlink():
            raise ValueError(f"Refusing proactive state path with symlink component: {current}")
        if current.exists():
            if current == normalized and not current.is_file():
                raise ValueError(f"Proactive state path must be a regular file: {normalized}")
            if current != normalized and not current.is_dir():
                raise ValueError(f"Proactive state parent must be a directory: {current}")
        parent = current.parent
        if parent == current:
            break
        current = parent
    return normalized


# AUDIT-FIX(#5): Create parents stepwise and re-check every component instead of trusting mkdir(parents=True) blindly.
def _ensure_safe_directory(path: Path) -> Path:
    normalized = _lexical_absolute_path(path)
    current = Path(normalized.anchor or os.sep)
    for part in normalized.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Refusing proactive state directory with symlink component: {current}")
        if current.exists():
            if not current.is_dir():
                raise ValueError(f"Proactive state directory component is not a directory: {current}")
            continue
        current.mkdir()
    return normalized


# AUDIT-FIX(#5): fsync the parent directory after replace so power loss does not silently drop the rename on ext4.
def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    directory_fd = os.open(os.fspath(path), flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    # AUDIT-FIX(#5): Harden writes against unsafe paths and incomplete persistence on sudden power loss.
    safe_path = _assert_safe_state_file_path(path)
    _ensure_safe_directory(safe_path.parent)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=safe_path.parent,
            prefix=f"{safe_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(os.fspath(temp_path), os.fspath(safe_path))
        _fsync_directory(safe_path.parent)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


# AUDIT-FIX(#2): Quarantine unrecoverable JSON corruption so the feature can self-heal instead of crashing forever.
def _quarantine_corrupt_state_file(path: Path, *, reason: str) -> None:
    safe_path = _assert_safe_state_file_path(path)
    if not safe_path.exists():
        return
    timestamp = _utcnow().strftime("%Y%m%dT%H%M%SZ")
    quarantine_path = safe_path.with_name(f"{safe_path.name}.corrupt.{timestamp}")
    suffix = 0
    while quarantine_path.exists():
        suffix += 1
        quarantine_path = safe_path.with_name(f"{safe_path.name}.corrupt.{timestamp}.{suffix}")
    os.replace(os.fspath(safe_path), os.fspath(quarantine_path))
    _fsync_directory(safe_path.parent)
    logger.warning(
        "Quarantined corrupt proactive state file from %s to %s (%s).",
        safe_path,
        quarantine_path,
        reason,
    )


@dataclass(frozen=True, slots=True)
class LongTermProactiveHistoryEntryV1:
    candidate_id: str
    kind: str
    summary: str
    sensitivity: str = "normal"
    source_memory_ids: tuple[str, ...] = ()
    first_seen_at: datetime = field(default_factory=_utcnow)
    last_seen_at: datetime = field(default_factory=_utcnow)
    last_reserved_at: datetime | None = None
    last_delivered_at: datetime | None = None
    last_skipped_at: datetime | None = None
    last_skip_reason: str | None = None
    last_prompt_text: str | None = None
    delivery_count: int = 0
    skip_count: int = 0

    def __post_init__(self) -> None:
        # AUDIT-FIX(#4): Normalize persisted datetimes so cooldown comparisons stay stable.
        # AUDIT-FIX(#7): Sanitize persisted IDs from malformed payloads.
        # AUDIT-FIX(#9): Bound audit text before it hits disk.
        object.__setattr__(self, "candidate_id", _normalized_candidate_id(self.candidate_id))
        object.__setattr__(self, "kind", _normalize_text(self.kind))
        object.__setattr__(self, "summary", _normalize_text(self.summary))
        object.__setattr__(
            self,
            "source_memory_ids",
            tuple(
                memory_id
                for memory_id in (_normalize_text(memory_id) for memory_id in self.source_memory_ids)
                if memory_id
            ),
        )
        object.__setattr__(self, "first_seen_at", _coerce_datetime(self.first_seen_at))
        object.__setattr__(self, "last_seen_at", _coerce_datetime(self.last_seen_at))
        object.__setattr__(
            self,
            "last_reserved_at",
            _coerce_datetime(self.last_reserved_at) if self.last_reserved_at is not None else None,
        )
        object.__setattr__(
            self,
            "last_delivered_at",
            _coerce_datetime(self.last_delivered_at) if self.last_delivered_at is not None else None,
        )
        object.__setattr__(
            self,
            "last_skipped_at",
            _coerce_datetime(self.last_skipped_at) if self.last_skipped_at is not None else None,
        )
        object.__setattr__(
            self,
            "last_skip_reason",
            _truncate_text(self.last_skip_reason, max_chars=_MAX_AUDIT_TEXT_CHARS),
        )
        object.__setattr__(
            self,
            "last_prompt_text",
            _truncate_text(self.last_prompt_text, max_chars=_MAX_AUDIT_TEXT_CHARS),
        )
        object.__setattr__(self, "sensitivity", normalize_memory_sensitivity(self.sensitivity))
        if not self.candidate_id:
            raise ValueError("candidate_id is required.")
        if not self.kind:
            raise ValueError("kind is required.")
        if not self.summary:
            raise ValueError("summary is required.")
        if self.sensitivity not in LONGTERM_MEMORY_SENSITIVITY:
            raise ValueError(f"sensitivity must be one of: {', '.join(sorted(LONGTERM_MEMORY_SENSITIVITY))}.")
        if self.delivery_count < 0:
            raise ValueError("delivery_count must be non-negative.")
        if self.skip_count < 0:
            raise ValueError("skip_count must be non-negative.")

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "summary": self.summary,
            "sensitivity": self.sensitivity,
            "source_memory_ids": list(self.source_memory_ids),
            "first_seen_at": self.first_seen_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat(),
            "delivery_count": self.delivery_count,
            "skip_count": self.skip_count,
        }
        if self.last_reserved_at is not None:
            payload["last_reserved_at"] = self.last_reserved_at.isoformat()
        if self.last_delivered_at is not None:
            payload["last_delivered_at"] = self.last_delivered_at.isoformat()
        if self.last_skipped_at is not None:
            payload["last_skipped_at"] = self.last_skipped_at.isoformat()
        if self.last_skip_reason is not None:
            payload["last_skip_reason"] = self.last_skip_reason
        if self.last_prompt_text is not None:
            payload["last_prompt_text"] = self.last_prompt_text
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "LongTermProactiveHistoryEntryV1":
        # AUDIT-FIX(#2): Parse legacy/malformed state defensively.
        # AUDIT-FIX(#4): Normalize persisted timestamps during load.
        # AUDIT-FIX(#7): Reject malformed source_memory_ids shapes.
        # AUDIT-FIX(#8): Clamp persisted counters to safe defaults.
        source_memory_ids = _coerce_source_memory_ids(payload.get("source_memory_ids", ()))

        last_seen_at = _parse_datetime(payload.get("last_seen_at"), field_name="last_seen_at")
        first_seen_raw = payload.get("first_seen_at", payload.get("last_seen_at"))
        first_seen_at = _parse_datetime(first_seen_raw, field_name="first_seen_at")

        return cls(
            candidate_id=str(payload.get("candidate_id", "")),
            kind=str(payload.get("kind", "")),
            summary=str(payload.get("summary", "")),
            sensitivity=str(payload.get("sensitivity", "normal")),
            source_memory_ids=source_memory_ids,
            first_seen_at=first_seen_at,
            last_seen_at=last_seen_at,
            last_reserved_at=_parse_optional_datetime(payload.get("last_reserved_at"), field_name="last_reserved_at"),
            last_delivered_at=_parse_optional_datetime(
                payload.get("last_delivered_at"),
                field_name="last_delivered_at",
            ),
            last_skipped_at=_parse_optional_datetime(payload.get("last_skipped_at"), field_name="last_skipped_at"),
            last_skip_reason=str(payload["last_skip_reason"]) if payload.get("last_skip_reason") is not None else None,
            last_prompt_text=str(payload["last_prompt_text"]) if payload.get("last_prompt_text") is not None else None,
            delivery_count=_coerce_non_negative_int(payload.get("delivery_count", 0), default=0),
            skip_count=_coerce_non_negative_int(payload.get("skip_count", 0), default=0),
        )


@dataclass(frozen=True, slots=True)
class LongTermProactiveReservationV1:
    candidate: LongTermProactiveCandidateV1
    reserved_at: datetime

    def __post_init__(self) -> None:
        # AUDIT-FIX(#4): Normalize reservation timestamps immediately so downstream comparisons stay consistent.
        object.__setattr__(self, "reserved_at", _coerce_datetime(self.reserved_at))


@dataclass(slots=True)
class LongTermProactiveStateStore:
    path: Path
    history_limit: int = _DEFAULT_HISTORY_LIMIT
    _lock: Lock = field(default_factory=Lock, repr=False)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "LongTermProactiveStateStore":
        return cls(
            path=chonkydb_data_path(config) / "twinr_memory_proactive_state_v1.json",
            # AUDIT-FIX(#8): Malformed .env values should fall back to sane bounds, not crash the module at startup.
            history_limit=_coerce_history_limit(
                getattr(config, "long_term_memory_proactive_history_limit", _DEFAULT_HISTORY_LIMIT)
            ),
        )

    def load_entries(self) -> tuple[LongTermProactiveHistoryEntryV1, ...]:
        with self._lock:
            return self._load_entries_unlocked()

    # AUDIT-FIX(#2): Keep reads best-effort; corrupted state should degrade to empty history instead of crashing.
    def _load_entries_unlocked(self) -> tuple[LongTermProactiveHistoryEntryV1, ...]:
        try:
            state_path = _assert_safe_state_file_path(self.path)
        except ValueError:
            logger.exception("Refusing unsafe proactive state path: %s", self.path)
            return ()

        if not state_path.exists():
            return ()

        try:
            raw_payload = state_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ()
        except OSError:
            logger.exception("Failed to read proactive state file: %s", state_path)
            return ()

        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            logger.warning("Proactive state file is corrupt and will be quarantined: %s", state_path)
            try:
                _quarantine_corrupt_state_file(state_path, reason=str(exc))
            except Exception:
                logger.exception("Failed to quarantine corrupt proactive state file: %s", state_path)
            return ()

        if not isinstance(payload, dict):
            logger.warning("Ignoring proactive state file with invalid top-level payload: %s", state_path)
            try:
                _quarantine_corrupt_state_file(state_path, reason="top-level JSON payload is not an object")
            except Exception:
                logger.exception("Failed to quarantine invalid proactive state file: %s", state_path)
            return ()

        schema = payload.get("schema", _STATE_SCHEMA)
        if schema != _STATE_SCHEMA:
            logger.warning("Ignoring proactive state file with unexpected schema %r at %s.", schema, state_path)
            return ()

        version = _coerce_int(payload.get("version", _STATE_VERSION), default=_STATE_VERSION)
        if version != _STATE_VERSION:
            logger.warning(
                "Ignoring proactive state file with unexpected version %r at %s.",
                payload.get("version"),
                state_path,
            )
            return ()

        items = payload.get("entries", [])
        if not isinstance(items, list):
            logger.warning("Ignoring malformed proactive history list at %s.", state_path)
            return ()

        entries: list[LongTermProactiveHistoryEntryV1] = []
        needs_repair = False
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                needs_repair = True
                continue
            try:
                entries.append(LongTermProactiveHistoryEntryV1.from_payload(item))
            except Exception:
                logger.warning(
                    "Skipping invalid proactive history entry at index %d from %s.",
                    index,
                    state_path,
                    exc_info=True,
                )
                needs_repair = True

        # AUDIT-FIX(#6): Sort by actual datetime, not ISO text, so mixed offsets cannot corrupt trimming order.
        ranked = tuple(
            sorted(
                entries,
                key=lambda item: (item.last_seen_at, item.candidate_id),
                reverse=True,
            )[: self.history_limit]
        )

        if needs_repair and state_path.exists():
            try:
                self._write_entries_unlocked(ranked)
            except Exception:
                logger.exception("Failed to repair proactive state file after dropping invalid entries: %s", state_path)

        return ranked

    def reserve(
        self,
        candidate: LongTermProactiveCandidateV1,
        *,
        reserved_at: datetime,
    ) -> LongTermProactiveReservationV1:
        safe_reserved_at = _coerce_datetime(reserved_at)
        with self._lock:
            entries = {entry.candidate_id: entry for entry in self._load_entries_unlocked()}
            candidate_id = _normalized_candidate_id(getattr(candidate, "candidate_id", ""))
            existing = entries.get(candidate_id)
            entries[candidate_id] = self._upsert_entry(
                existing=existing,
                candidate=candidate,
                seen_at=safe_reserved_at,
                reserved_at=safe_reserved_at,
            )
            self._write_entries_unlocked(entries.values())
        return LongTermProactiveReservationV1(candidate=candidate, reserved_at=safe_reserved_at)

    # AUDIT-FIX(#3): Select-and-reserve under the same lock to eliminate duplicate reservations on concurrent callers.
    def reserve_first_eligible(
        self,
        *,
        candidates: Iterable[LongTermProactiveCandidateV1],
        reserved_at: datetime,
        is_eligible: Callable[
            [LongTermProactiveCandidateV1, LongTermProactiveHistoryEntryV1 | None],
            bool,
        ],
    ) -> LongTermProactiveReservationV1 | None:
        safe_reserved_at = _coerce_datetime(reserved_at)
        with self._lock:
            entries = {entry.candidate_id: entry for entry in self._load_entries_unlocked()}
            for candidate in candidates:
                candidate_id = _normalized_candidate_id(getattr(candidate, "candidate_id", ""))
                if not candidate_id:
                    continue
                existing = entries.get(candidate_id)
                if not is_eligible(candidate, existing):
                    continue
                entries[candidate_id] = self._upsert_entry(
                    existing=existing,
                    candidate=candidate,
                    seen_at=safe_reserved_at,
                    reserved_at=safe_reserved_at,
                )
                self._write_entries_unlocked(entries.values())
                return LongTermProactiveReservationV1(candidate=candidate, reserved_at=safe_reserved_at)
        return None

    def mark_delivered(
        self,
        *,
        candidate: LongTermProactiveCandidateV1,
        delivered_at: datetime,
        prompt_text: str | None = None,
    ) -> LongTermProactiveHistoryEntryV1:
        safe_delivered_at = _coerce_datetime(delivered_at)
        with self._lock:
            entries = {entry.candidate_id: entry for entry in self._load_entries_unlocked()}
            candidate_id = _normalized_candidate_id(getattr(candidate, "candidate_id", ""))
            existing = entries.get(candidate_id)
            current = self._upsert_entry(existing=existing, candidate=candidate, seen_at=safe_delivered_at)
            entries[candidate_id] = LongTermProactiveHistoryEntryV1(
                candidate_id=current.candidate_id,
                kind=current.kind,
                summary=current.summary,
                sensitivity=current.sensitivity,
                source_memory_ids=current.source_memory_ids,
                first_seen_at=current.first_seen_at,
                last_seen_at=safe_delivered_at,
                last_reserved_at=current.last_reserved_at,
                last_delivered_at=safe_delivered_at,
                last_skipped_at=current.last_skipped_at,
                last_skip_reason=current.last_skip_reason,
                # AUDIT-FIX(#9): Persist only bounded audit text, not arbitrary prompt blobs.
                last_prompt_text=_truncate_text(prompt_text, max_chars=_MAX_AUDIT_TEXT_CHARS) or current.last_prompt_text,
                delivery_count=current.delivery_count + 1,
                skip_count=current.skip_count,
            )
            self._write_entries_unlocked(entries.values())
            return entries[candidate_id]

    def mark_skipped(
        self,
        *,
        candidate: LongTermProactiveCandidateV1,
        skipped_at: datetime,
        reason: str,
    ) -> LongTermProactiveHistoryEntryV1:
        # AUDIT-FIX(#9): Bound skip reasons so storage remains predictable even if upstream passes large strings.
        clean_reason = _truncate_text(reason, max_chars=_MAX_AUDIT_TEXT_CHARS) or "unknown"
        safe_skipped_at = _coerce_datetime(skipped_at)
        with self._lock:
            entries = {entry.candidate_id: entry for entry in self._load_entries_unlocked()}
            candidate_id = _normalized_candidate_id(getattr(candidate, "candidate_id", ""))
            existing = entries.get(candidate_id)
            current = self._upsert_entry(existing=existing, candidate=candidate, seen_at=safe_skipped_at)
            entries[candidate_id] = LongTermProactiveHistoryEntryV1(
                candidate_id=current.candidate_id,
                kind=current.kind,
                summary=current.summary,
                sensitivity=current.sensitivity,
                source_memory_ids=current.source_memory_ids,
                first_seen_at=current.first_seen_at,
                last_seen_at=safe_skipped_at,
                last_reserved_at=current.last_reserved_at,
                last_delivered_at=current.last_delivered_at,
                last_skipped_at=safe_skipped_at,
                last_skip_reason=clean_reason,
                last_prompt_text=current.last_prompt_text,
                delivery_count=current.delivery_count,
                skip_count=current.skip_count + 1,
            )
            self._write_entries_unlocked(entries.values())
            return entries[candidate_id]

    def _upsert_entry(
        self,
        *,
        existing: LongTermProactiveHistoryEntryV1 | None,
        candidate: LongTermProactiveCandidateV1,
        seen_at: datetime,
        reserved_at: datetime | None = None,
    ) -> LongTermProactiveHistoryEntryV1:
        safe_seen_at = _coerce_datetime(seen_at)
        safe_reserved_at = _coerce_datetime(reserved_at) if reserved_at is not None else None
        if existing is None:
            return LongTermProactiveHistoryEntryV1(
                candidate_id=candidate.candidate_id,
                kind=candidate.kind,
                summary=candidate.summary,
                sensitivity=candidate.sensitivity,
                source_memory_ids=_coerce_source_memory_ids(getattr(candidate, "source_memory_ids", ())),
                first_seen_at=safe_seen_at,
                last_seen_at=safe_seen_at,
                last_reserved_at=safe_reserved_at,
            )
        return LongTermProactiveHistoryEntryV1(
            candidate_id=existing.candidate_id,
            kind=candidate.kind,
            summary=candidate.summary,
            sensitivity=candidate.sensitivity,
            source_memory_ids=_coerce_source_memory_ids(getattr(candidate, "source_memory_ids", ())),
            first_seen_at=existing.first_seen_at,
            last_seen_at=safe_seen_at,
            last_reserved_at=safe_reserved_at or existing.last_reserved_at,
            last_delivered_at=existing.last_delivered_at,
            last_skipped_at=existing.last_skipped_at,
            last_skip_reason=existing.last_skip_reason,
            last_prompt_text=existing.last_prompt_text,
            delivery_count=existing.delivery_count,
            skip_count=existing.skip_count,
        )

    def _write_entries_unlocked(self, entries: Iterable[LongTermProactiveHistoryEntryV1]) -> None:
        # AUDIT-FIX(#6): Keep trimming order chronologically correct even if legacy payloads contain mixed offsets.
        ranked = sorted(
            entries,
            key=lambda item: (item.last_seen_at, item.candidate_id),
            reverse=True,
        )[: self.history_limit]
        payload = {
            "schema": _STATE_SCHEMA,
            "version": _STATE_VERSION,
            "entries": [item.to_payload() for item in ranked],
        }
        _write_json_atomic(self.path, payload)


@dataclass(slots=True)
class LongTermProactivePolicy:
    config: TwinrConfig
    state_store: LongTermProactiveStateStore
    blocked_sensitivities: frozenset[str] = frozenset({"private", "sensitive", "critical"})

    def reserve_candidate(
        self,
        *,
        plan: LongTermProactivePlanV1,
        now: datetime | None = None,
    ) -> LongTermProactiveReservationV1 | None:
        if not self.config.long_term_memory_proactive_enabled:
            return None
        current_time = self._current_time(now)
        try:
            # AUDIT-FIX(#3): Atomically reserve the first eligible candidate so concurrent requests cannot double-book it.
            return self.state_store.reserve_first_eligible(
                candidates=plan.candidates,
                reserved_at=current_time,
                is_eligible=lambda candidate, entry: self._candidate_is_eligible(
                    candidate,
                    entry=entry,
                    current_time=current_time,
                ),
            )
        except Exception:
            logger.exception("Failed to reserve proactive memory candidate.")
            return None

    def reserve_specific_candidate(
        self,
        candidate: LongTermProactiveCandidateV1,
        *,
        now: datetime | None = None,
    ) -> LongTermProactiveReservationV1:
        if not self.config.long_term_memory_proactive_enabled:
            raise ValueError("Proactive long-term memory is disabled.")
        current_time = self._current_time(now)
        # AUDIT-FIX(#1): Route explicit reservations through the same privacy/confidence/cooldown gate as auto-selection.
        reservation = self.state_store.reserve_first_eligible(
            candidates=(candidate,),
            reserved_at=current_time,
            is_eligible=lambda item, entry: self._candidate_is_eligible(
                item,
                entry=entry,
                current_time=current_time,
            ),
        )
        if reservation is None:
            raise ValueError("Candidate is not eligible for proactive reservation.")
        return reservation

    def preview_candidate(
        self,
        *,
        plan: LongTermProactivePlanV1,
        now: datetime | None = None,
    ) -> LongTermProactiveCandidateV1 | None:
        if not self.config.long_term_memory_proactive_enabled:
            return None
        current_time = self._current_time(now)
        history = {entry.candidate_id: entry for entry in self.state_store.load_entries()}
        return self._select_candidate(plan.candidates, history=history, current_time=current_time)

    def mark_delivered(
        self,
        reservation: LongTermProactiveReservationV1,
        *,
        delivered_at: datetime | None = None,
        prompt_text: str | None = None,
    ) -> LongTermProactiveHistoryEntryV1:
        current_time = self._current_time(delivered_at)
        try:
            return self.state_store.mark_delivered(
                candidate=reservation.candidate,
                delivered_at=current_time,
                prompt_text=prompt_text,
            )
        except Exception:
            # AUDIT-FIX(#10): Delivery has already happened; return a best-effort in-memory entry instead of crashing upstream.
            logger.exception(
                "Failed to persist proactive delivery state for candidate %s.",
                _normalized_candidate_id(getattr(reservation.candidate, "candidate_id", "")) or "<unknown>",
            )
            return self._build_fallback_history_entry(
                candidate=reservation.candidate,
                reservation=reservation,
                seen_at=current_time,
                delivered_at=current_time,
                prompt_text=prompt_text,
                delivery_count=1,
            )

    def mark_skipped(
        self,
        reservation: LongTermProactiveReservationV1,
        *,
        reason: str,
        skipped_at: datetime | None = None,
    ) -> LongTermProactiveHistoryEntryV1:
        current_time = self._current_time(skipped_at)
        try:
            return self.state_store.mark_skipped(
                candidate=reservation.candidate,
                skipped_at=current_time,
                reason=reason,
            )
        except Exception:
            # AUDIT-FIX(#10): Skip bookkeeping must not crash the dialogue loop when disk state is degraded.
            logger.exception(
                "Failed to persist proactive skip state for candidate %s.",
                _normalized_candidate_id(getattr(reservation.candidate, "candidate_id", "")) or "<unknown>",
            )
            return self._build_fallback_history_entry(
                candidate=reservation.candidate,
                reservation=reservation,
                seen_at=current_time,
                skipped_at=current_time,
                reason=reason,
                skip_count=1,
            )

    def _select_candidate(
        self,
        candidates: Iterable[LongTermProactiveCandidateV1],
        *,
        history: dict[str, LongTermProactiveHistoryEntryV1],
        current_time: datetime,
    ) -> LongTermProactiveCandidateV1 | None:
        for candidate in candidates:
            candidate_id = _normalized_candidate_id(getattr(candidate, "candidate_id", ""))
            if not candidate_id:
                continue
            entry = history.get(candidate_id)
            if self._candidate_is_eligible(candidate, entry=entry, current_time=current_time):
                return candidate
        return None

    # AUDIT-FIX(#1): Centralize privacy/sensitivity validation so every entry point behaves the same.
    # AUDIT-FIX(#4): Use normalized timestamps for cooldown checks.
    # AUDIT-FIX(#8): Use safe numeric coercion for config and candidate values.
    def _candidate_is_eligible(
        self,
        candidate: LongTermProactiveCandidateV1,
        *,
        entry: LongTermProactiveHistoryEntryV1 | None,
        current_time: datetime,
    ) -> bool:
        if not _normalized_candidate_id(getattr(candidate, "candidate_id", "")):
            return False
        if not _normalize_text(getattr(candidate, "kind", "")):
            return False
        if not _normalize_text(getattr(candidate, "summary", "")):
            return False

        confidence = _coerce_confidence(getattr(candidate, "confidence", 0.0), default=0.0)
        if confidence < _coerce_confidence(
            getattr(self.config, "long_term_memory_proactive_min_confidence", 0.0),
            default=0.0,
        ):
            return False

        sensitivity = self._normalized_candidate_sensitivity(candidate)
        if sensitivity is None:
            return False
        if (
            not self.config.long_term_memory_proactive_allow_sensitive
            and sensitivity in self.blocked_sensitivities
        ):
            return False

        if self._within_cooldown(
            entry.last_reserved_at if entry is not None else None,
            current_time=current_time,
            cooldown_s=getattr(self.config, "long_term_memory_proactive_reservation_ttl_s", 0.0),
        ):
            return False
        if self._within_cooldown(
            entry.last_delivered_at if entry is not None else None,
            current_time=current_time,
            cooldown_s=getattr(self.config, "long_term_memory_proactive_repeat_cooldown_s", 0.0),
        ):
            return False
        if self._within_cooldown(
            entry.last_skipped_at if entry is not None else None,
            current_time=current_time,
            cooldown_s=getattr(self.config, "long_term_memory_proactive_skip_cooldown_s", 0.0),
        ):
            return False
        return True

    def _normalized_candidate_sensitivity(self, candidate: LongTermProactiveCandidateV1) -> str | None:
        try:
            # AUDIT-FIX(#1): Normalize candidate sensitivity before policy checks; unknown values fail closed.
            return normalize_memory_sensitivity(str(getattr(candidate, "sensitivity", "normal")))
        except Exception:
            logger.warning(
                "Ignoring proactive candidate with invalid sensitivity %r.",
                getattr(candidate, "sensitivity", None),
            )
            return None

    def _current_time(self, value: datetime | None) -> datetime:
        if value is None:
            return _utcnow()
        return _coerce_datetime(value, default_tz=self._local_timezone())

    def _local_timezone(self) -> timezone | ZoneInfo:
        timezone_name = _normalize_text(str(getattr(self.config, "local_timezone_name", "")))
        if not timezone_name:
            return timezone.utc
        try:
            # AUDIT-FIX(#4): Bad timezone config must degrade to UTC rather than explode in runtime hot paths.
            return ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            logger.warning("Invalid proactive policy timezone %r; falling back to UTC.", timezone_name)
            return timezone.utc

    def _build_fallback_history_entry(
        self,
        *,
        candidate: LongTermProactiveCandidateV1,
        reservation: LongTermProactiveReservationV1,
        seen_at: datetime,
        delivered_at: datetime | None = None,
        skipped_at: datetime | None = None,
        reason: str | None = None,
        prompt_text: str | None = None,
        delivery_count: int = 0,
        skip_count: int = 0,
    ) -> LongTermProactiveHistoryEntryV1:
        return LongTermProactiveHistoryEntryV1(
            candidate_id=candidate.candidate_id,
            kind=candidate.kind,
            summary=candidate.summary,
            sensitivity=str(getattr(candidate, "sensitivity", "normal")),
            source_memory_ids=_coerce_source_memory_ids(getattr(candidate, "source_memory_ids", ())),
            first_seen_at=reservation.reserved_at,
            last_seen_at=seen_at,
            last_reserved_at=reservation.reserved_at,
            last_delivered_at=delivered_at,
            last_skipped_at=skipped_at,
            last_skip_reason=reason,
            last_prompt_text=prompt_text,
            delivery_count=max(0, delivery_count),
            skip_count=max(0, skip_count),
        )

    def _within_cooldown(
        self,
        value: datetime | None,
        *,
        current_time: datetime,
        cooldown_s: float,
    ) -> bool:
        if value is None:
            return False
        return current_time < _coerce_datetime(value) + timedelta(
            seconds=_coerce_non_negative_seconds(cooldown_s, default=0.0)
        )


__all__ = [
    "LongTermProactiveHistoryEntryV1",
    "LongTermProactivePolicy",
    "LongTermProactiveReservationV1",
    "LongTermProactiveStateStore",
]