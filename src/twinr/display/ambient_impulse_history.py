"""Persist reserve-lane exposure history and resolved user reactions."""

# CHANGELOG: 2026-03-28
# BUG-1: Fixed lost-update races by replacing load-modify-save JSON snapshots with transactional SQLite writes.
# BUG-2: Fixed crash/power-loss corruption from non-atomic whole-file overwrites by moving durable storage to SQLite transactions.
# BUG-3: Fixed naive-datetime handling so naive inputs are treated consistently as UTC instead of local wall time.
# BUG-4: Fixed exposure-id collisions during repeated same-second renders by de-duplicating IDs on write.
# SEC-1: Hardened the local store with SQLite defensive mode, trusted_schema=OFF, disabled triggers/views, reduced SQLite runtime limits, and mmap disabled.
# SEC-2: Restricted artifact files to owner-only permissions and quarantined legacy/corrupt payloads instead of reusing them in-place.
# IMP-1: Upgraded the history backend to an indexed SQLite event store with bounded pruning and direct query paths for pending-load/count/resolve.
# IMP-2: Added one-time migration from the legacy JSON payload at the same path, with automatic backup of the previous artifact.
# BREAKING: The on-disk artifact at `path` now uses SQLite database format instead of plain JSON text.

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
from typing import Any

try:  # POSIX/Linux on Raspberry Pi.
    import fcntl
except Exception:  # pragma: no cover - non-POSIX fallback.
    fcntl = None  # type: ignore[assignment]

from twinr.agent.base_agent.config import TwinrConfig

_DEFAULT_HISTORY_PATH = "artifacts/stores/ops/display_ambient_impulse_history.json"
_DEFAULT_MAX_ENTRIES = 256
_DEFAULT_BUSY_TIMEOUT_MS = 5000
_MAX_LEGACY_IMPORT_BYTES = 4 * 1024 * 1024
_ALLOWED_RESPONSE_STATUS = frozenset({"pending", "engaged", "cooled", "avoided", "ignored", "neutral"})
_ALLOWED_RESPONSE_SENTIMENT = frozenset({"positive", "negative", "neutral", "mixed", "unknown"})
_SQLITE_HEADER = b"SQLite format 3\x00"
_SQLITE_USER_VERSION = 1
_SQLITE_JOURNAL_MODE = "TRUNCATE"
_SQLITE_SYNCHRONOUS = "FULL"

_LOGGER = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Return the current UTC wall clock."""

    return datetime.now(timezone.utc)


def _coerce_utc_datetime(value: object | None, *, default: datetime | None = None) -> datetime | None:
    """Normalize one timestamp-like value into an aware UTC datetime."""

    if value is None:
        return default
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return _normalize_timestamp(value) or default


def _normalize_timestamp(value: object | None) -> datetime | None:
    """Parse one optional timestamp into an aware UTC datetime."""

    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_timestamp(value: object) -> str:
    """Serialize one timestamp-like value as UTC ISO-8601 text."""

    normalized = _coerce_utc_datetime(value, default=_utc_now()) or _utc_now()
    return normalized.astimezone(timezone.utc).isoformat()


def _epoch_seconds(value: object) -> float:
    """Convert one timestamp-like value into POSIX epoch seconds."""

    normalized = _coerce_utc_datetime(value, default=_utc_now()) or _utc_now()
    return normalized.timestamp()


def _compact_text(value: object | None, *, max_len: int) -> str:
    """Normalize one bounded single-line text field."""

    if value is None:
        return ""
    compact = " ".join(str(value).split()).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _normalize_mapping(value: Mapping[str, object] | None, *, depth: int = 0) -> dict[str, object] | None:
    """Return one plain JSON-safe mapping copy when present."""

    if value is None or depth >= 4:
        return None
    normalized: dict[str, object] = {}
    for raw_key, raw_value in value.items():
        key = _compact_text(raw_key, max_len=80)
        if not key:
            continue
        if isinstance(raw_value, Mapping):
            child = _normalize_mapping(raw_value, depth=depth + 1)
            if child:
                normalized[key] = child
            continue
        if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes, bytearray)):
            items: list[object] = []
            for item in raw_value[:16]:
                if isinstance(item, Mapping):
                    child = _normalize_mapping(item, depth=depth + 1)
                    if child:
                        items.append(child)
                    continue
                compact = _compact_text(item, max_len=160)
                if compact:
                    items.append(compact)
            if items:
                normalized[key] = items
            continue
        number = _normalize_optional_float(raw_value)
        if number is not None and not isinstance(raw_value, str):
            normalized[key] = number
            continue
        compact = _compact_text(raw_value, max_len=160)
        if compact:
            normalized[key] = compact
    return normalized or None


def _normalize_text_tuple(values: Iterable[object] | None, *, max_items: int, max_len: int) -> tuple[str, ...]:
    """Normalize one bounded ordered text tuple while removing duplicates."""

    if values is None:
        return ()
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        compact = _compact_text(value, max_len=max_len)
        if not compact:
            continue
        key = compact.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(compact)
        if len(ordered) >= max_items:
            break
    return tuple(ordered)


def _normalize_optional_float(value: object | None) -> float | None:
    """Return one optional finite float for persisted payloads."""

    if value is None:
        return None
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalize_response_status(value: object | None) -> str:
    """Normalize one coarse feedback status token."""

    compact = _compact_text(value, max_len=32).lower().replace("-", "_").replace(" ", "_")
    if compact not in _ALLOWED_RESPONSE_STATUS:
        return "pending"
    return compact


def _normalize_response_sentiment(value: object | None) -> str:
    """Normalize one coarse sentiment token."""

    compact = _compact_text(value, max_len=32).lower().replace("-", "_").replace(" ", "_")
    if compact not in _ALLOWED_RESPONSE_SENTIMENT:
        return "unknown"
    return compact


def _normalize_store_path(path: Path | str) -> Path:
    """Return one expanded absolute artifact path."""

    return Path(path).expanduser().resolve()


def _json_dumps_compact(value: object) -> str:
    """Serialize one JSON-safe object into compact UTF-8 text."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_loads_text(value: str | None, *, default: Any) -> Any:
    """Parse one optional JSON text payload with a bounded fallback."""

    text = str(value or "").strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _backup_path_for(path: Path, *, suffix: str) -> Path:
    """Return one non-conflicting sibling backup path."""

    timestamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    candidate = path.with_name(f"{path.name}.{timestamp}.{suffix}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.{timestamp}.{counter}.{suffix}")
        counter += 1
    return candidate


def _best_effort_owner_only(path: Path, *, mode: int) -> None:
    """Apply owner-only permissions when the platform allows it."""

    try:
        path.chmod(mode)
    except OSError:
        return


def _sqlite_pragma_identifier(name: str) -> str:
    """Return one fixed trusted PRAGMA identifier."""

    return name


def _supports_setconfig(op_name: str) -> bool:
    """Return whether the sqlite3 module exposes one db-config constant."""

    return hasattr(sqlite3, op_name) and hasattr(sqlite3.Connection, "setconfig")


def _apply_sqlite_hardening(connection: sqlite3.Connection, *, read_only: bool) -> None:
    """Apply conservative connection hardening suitable for a tiny local store."""

    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA {_sqlite_pragma_identifier('busy_timeout')} = {_DEFAULT_BUSY_TIMEOUT_MS}")
    connection.execute(f"PRAGMA {_sqlite_pragma_identifier('trusted_schema')} = OFF")
    connection.execute(f"PRAGMA {_sqlite_pragma_identifier('mmap_size')} = 0")
    if read_only:
        connection.execute(f"PRAGMA {_sqlite_pragma_identifier('query_only')} = ON")
    if _supports_setconfig("SQLITE_DBCONFIG_DEFENSIVE"):
        try:
            connection.setconfig(sqlite3.SQLITE_DBCONFIG_DEFENSIVE, True)
        except Exception:
            _LOGGER.debug("Could not enable SQLITE_DBCONFIG_DEFENSIVE.", exc_info=True)
    if _supports_setconfig("SQLITE_DBCONFIG_TRUSTED_SCHEMA"):
        try:
            connection.setconfig(sqlite3.SQLITE_DBCONFIG_TRUSTED_SCHEMA, False)
        except Exception:
            _LOGGER.debug("Could not disable SQLITE_DBCONFIG_TRUSTED_SCHEMA.", exc_info=True)
    if _supports_setconfig("SQLITE_DBCONFIG_ENABLE_TRIGGER"):
        try:
            connection.setconfig(sqlite3.SQLITE_DBCONFIG_ENABLE_TRIGGER, False)
        except Exception:
            _LOGGER.debug("Could not disable triggers.", exc_info=True)
    if _supports_setconfig("SQLITE_DBCONFIG_ENABLE_VIEW"):
        try:
            connection.setconfig(sqlite3.SQLITE_DBCONFIG_ENABLE_VIEW, False)
        except Exception:
            _LOGGER.debug("Could not disable views.", exc_info=True)
    for limit_name, limit_value in (
        ("SQLITE_LIMIT_ATTACHED", 0),
        ("SQLITE_LIMIT_LENGTH", 1_000_000),
        ("SQLITE_LIMIT_SQL_LENGTH", 100_000),
        ("SQLITE_LIMIT_COLUMN", 64),
        ("SQLITE_LIMIT_EXPR_DEPTH", 20),
        ("SQLITE_LIMIT_COMPOUND_SELECT", 4),
        ("SQLITE_LIMIT_VARIABLE_NUMBER", 64),
    ):
        if not hasattr(sqlite3, limit_name):
            continue
        try:
            connection.setlimit(getattr(sqlite3, limit_name), limit_value)
        except Exception:
            _LOGGER.debug("Could not apply SQLite limit %s=%s.", limit_name, limit_value, exc_info=True)


def _legacy_payload_to_exposures(payload: object) -> tuple["DisplayAmbientImpulseExposure", ...]:
    """Parse one legacy JSON payload into normalized exposures."""

    raw_entries = payload.get("exposures") if isinstance(payload, Mapping) else None
    if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes, bytearray)):
        return ()
    entries: list[DisplayAmbientImpulseExposure] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            continue
        try:
            entries.append(DisplayAmbientImpulseExposure.from_dict(raw_entry))
        except Exception:
            _LOGGER.warning("Ignoring one invalid legacy display ambient impulse history entry.", exc_info=True)
    return tuple(entries)


def _exposure_id(
    *,
    shown_at: datetime,
    topic_key: str,
    headline: str,
) -> str:
    """Return one stable exposure id for the shown reserve card."""

    digest = hashlib.sha1(f"{shown_at.isoformat()}::{topic_key}::{headline}".encode("utf-8")).hexdigest()
    return f"ambient_exposure:{digest[:16]}"


@dataclass(frozen=True, slots=True)
class DisplayAmbientImpulseExposure:
    """Describe one shown reserve-lane card and its later coarse outcome."""

    exposure_id: str
    source: str
    topic_key: str
    title: str
    headline: str
    body: str
    action: str
    attention_state: str
    shown_at: str
    expires_at: str
    semantic_topic_key: str = ""
    match_anchors: tuple[str, ...] = ()
    response_status: str = "pending"
    response_sentiment: str = "unknown"
    response_at: str | None = None
    response_mode: str = ""
    response_latency_seconds: float | None = None
    response_turn_id: str | None = None
    response_target: str | None = None
    response_summary: str = ""
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        """Normalize one exposure record into a stable bounded contract."""

        object.__setattr__(self, "exposure_id", _compact_text(self.exposure_id, max_len=64))
        object.__setattr__(self, "source", _compact_text(self.source, max_len=48) or "ambient_impulse")
        object.__setattr__(self, "topic_key", _compact_text(self.topic_key, max_len=96).casefold())
        semantic_topic_key = _compact_text(self.semantic_topic_key, max_len=96).casefold()
        object.__setattr__(self, "semantic_topic_key", semantic_topic_key or self.topic_key)
        object.__setattr__(self, "title", _compact_text(self.title, max_len=96))
        object.__setattr__(self, "headline", _compact_text(self.headline, max_len=160))
        object.__setattr__(self, "body", _compact_text(self.body, max_len=160))
        object.__setattr__(self, "action", _compact_text(self.action, max_len=24).lower() or "hint")
        object.__setattr__(
            self,
            "attention_state",
            _compact_text(self.attention_state, max_len=32).lower() or "background",
        )
        shown_at = _coerce_utc_datetime(self.shown_at, default=_utc_now()) or _utc_now()
        expires_at = _coerce_utc_datetime(self.expires_at, default=(shown_at + timedelta(minutes=10))) or (
            shown_at + timedelta(minutes=10)
        )
        object.__setattr__(self, "shown_at", _format_timestamp(shown_at))
        object.__setattr__(self, "expires_at", _format_timestamp(expires_at))
        object.__setattr__(
            self,
            "match_anchors",
            _normalize_text_tuple(self.match_anchors, max_items=8, max_len=160),
        )
        object.__setattr__(self, "response_status", _normalize_response_status(self.response_status))
        object.__setattr__(self, "response_sentiment", _normalize_response_sentiment(self.response_sentiment))
        response_at = _coerce_utc_datetime(self.response_at)
        object.__setattr__(self, "response_at", _format_timestamp(response_at) if response_at is not None else None)
        object.__setattr__(self, "response_mode", _compact_text(self.response_mode, max_len=48).lower())
        latency_seconds = None
        try:
            if self.response_latency_seconds is not None:
                latency_seconds = max(0.0, float(self.response_latency_seconds))
        except (TypeError, ValueError):
            latency_seconds = None
        object.__setattr__(self, "response_latency_seconds", latency_seconds)
        object.__setattr__(self, "response_turn_id", _compact_text(self.response_turn_id, max_len=96) or None)
        object.__setattr__(self, "response_target", _compact_text(self.response_target, max_len=96) or None)
        object.__setattr__(self, "response_summary", _compact_text(self.response_summary, max_len=220))
        object.__setattr__(self, "metadata", _normalize_mapping(self.metadata))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DisplayAmbientImpulseExposure":
        """Build one exposure record from persisted JSON-style data."""

        raw_match_anchors = payload.get("match_anchors")
        raw_metadata = payload.get("metadata")
        return cls(
            exposure_id=_compact_text(payload.get("exposure_id"), max_len=64),
            source=_compact_text(payload.get("source"), max_len=48) or "ambient_impulse",
            topic_key=_compact_text(payload.get("topic_key"), max_len=96),
            semantic_topic_key=_compact_text(payload.get("semantic_topic_key"), max_len=96),
            title=_compact_text(payload.get("title"), max_len=96),
            headline=_compact_text(payload.get("headline"), max_len=160),
            body=_compact_text(payload.get("body"), max_len=160),
            action=_compact_text(payload.get("action"), max_len=24),
            attention_state=_compact_text(payload.get("attention_state"), max_len=32),
            shown_at=_compact_text(payload.get("shown_at"), max_len=64),
            expires_at=_compact_text(payload.get("expires_at"), max_len=64),
            match_anchors=_normalize_text_tuple(
                raw_match_anchors
                if isinstance(raw_match_anchors, Sequence)
                and not isinstance(raw_match_anchors, (str, bytes, bytearray))
                else None,
                max_items=8,
                max_len=160,
            ),
            response_status=_compact_text(payload.get("response_status"), max_len=32),
            response_sentiment=_compact_text(payload.get("response_sentiment"), max_len=32),
            response_at=_compact_text(payload.get("response_at"), max_len=64) or None,
            response_mode=_compact_text(payload.get("response_mode"), max_len=48),
            response_latency_seconds=_normalize_optional_float(payload.get("response_latency_seconds")),
            response_turn_id=_compact_text(payload.get("response_turn_id"), max_len=96) or None,
            response_target=_compact_text(payload.get("response_target"), max_len=96) or None,
            response_summary=_compact_text(payload.get("response_summary"), max_len=220),
            metadata=raw_metadata if isinstance(raw_metadata, Mapping) else None,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the exposure record into a JSON-safe mapping."""

        return asdict(self)

    def shown_at_datetime(self) -> datetime:
        """Return the shown-at timestamp as an aware UTC datetime."""

        return _coerce_utc_datetime(self.shown_at, default=_utc_now()) or _utc_now()

    def semantic_key(self) -> str:
        """Return the grouped semantic topic key for this exposure."""

        return _compact_text(self.semantic_topic_key, max_len=96).casefold() or self.topic_key

    def expires_at_datetime(self) -> datetime:
        """Return the expiry timestamp as an aware UTC datetime."""

        return _coerce_utc_datetime(self.expires_at, default=self.shown_at_datetime()) or self.shown_at_datetime()

    def anchors(self) -> tuple[str, ...]:
        """Return the ordered matching anchors for feedback correlation."""

        return _normalize_text_tuple(
            (self.semantic_topic_key, self.topic_key, self.title, self.headline, self.body, *self.match_anchors),
            max_items=10,
            max_len=160,
        )

    def with_feedback(
        self,
        *,
        response_status: str,
        response_sentiment: str,
        response_at: datetime,
        response_mode: str,
        response_latency_seconds: float | None,
        response_turn_id: str | None,
        response_target: str | None,
        response_summary: str,
    ) -> "DisplayAmbientImpulseExposure":
        """Return one copy annotated with the resolved user reaction."""

        return DisplayAmbientImpulseExposure(
            exposure_id=self.exposure_id,
            source=self.source,
            topic_key=self.topic_key,
            semantic_topic_key=self.semantic_topic_key,
            title=self.title,
            headline=self.headline,
            body=self.body,
            action=self.action,
            attention_state=self.attention_state,
            shown_at=self.shown_at,
            expires_at=self.expires_at,
            match_anchors=self.match_anchors,
            response_status=response_status,
            response_sentiment=response_sentiment,
            response_at=_format_timestamp(response_at),
            response_mode=response_mode,
            response_latency_seconds=response_latency_seconds,
            response_turn_id=response_turn_id,
            response_target=response_target,
            response_summary=response_summary,
            metadata=self.metadata,
        )


@dataclass(slots=True)
class DisplayAmbientImpulseHistoryStore:
    """Persist one bounded history of shown reserve cards and coarse outcomes."""

    path: Path
    max_entries: int = _DEFAULT_MAX_ENTRIES
    _bootstrapped: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayAmbientImpulseHistoryStore":
        """Resolve the history artifact path from configuration."""

        project_root = Path(config.project_root).expanduser().resolve()
        configured = Path(
            getattr(config, "display_ambient_impulse_history_path", _DEFAULT_HISTORY_PATH) or _DEFAULT_HISTORY_PATH
        )
        resolved = configured if configured.is_absolute() else project_root / configured
        max_entries = getattr(config, "display_ambient_impulse_history_max_entries", _DEFAULT_MAX_ENTRIES)
        try:
            bounded_max_entries = max(1, int(max_entries))
        except (TypeError, ValueError):
            bounded_max_entries = _DEFAULT_MAX_ENTRIES
        return cls(path=resolved, max_entries=bounded_max_entries)

    def load(self) -> tuple[DisplayAmbientImpulseExposure, ...]:
        """Load the current exposure history snapshot."""

        if not self._bootstrapped and not _normalize_store_path(self.path).exists():
            return ()
        try:
            self._ensure_storage_ready()
            with self._connect(read_only=True) as connection:
                rows = connection.execute(
                    """
                    SELECT
                        exposure_id,
                        source,
                        topic_key,
                        semantic_topic_key,
                        title,
                        headline,
                        body,
                        action,
                        attention_state,
                        shown_at,
                        expires_at,
                        match_anchors_json,
                        response_status,
                        response_sentiment,
                        response_at,
                        response_mode,
                        response_latency_seconds,
                        response_turn_id,
                        response_target,
                        response_summary,
                        metadata_json
                    FROM ambient_impulse_exposures
                    ORDER BY seq ASC
                    """
                ).fetchall()
        except sqlite3.DatabaseError:
            _LOGGER.warning("Failed to load display ambient impulse history from %s.", self.path, exc_info=True)
            return ()
        return tuple(self._row_to_exposure(row) for row in rows)

    def save_all(self, exposures: Sequence[DisplayAmbientImpulseExposure]) -> tuple[DisplayAmbientImpulseExposure, ...]:
        """Persist one bounded ordered exposure history snapshot."""

        normalized = self._deduplicate_exposures(tuple(exposures)[-max(1, int(self.max_entries)) :])
        self._ensure_storage_ready()
        with self._write_transaction() as connection:
            connection.execute("DELETE FROM ambient_impulse_exposures")
            for exposure in normalized:
                self._insert_exposure(connection, exposure)
            self._prune_to_max_entries(connection)
        return normalized

    def append_exposure(
        self,
        *,
        source: str,
        topic_key: str,
        semantic_topic_key: str | None = None,
        title: str,
        headline: str,
        body: str,
        action: str,
        attention_state: str,
        shown_at: datetime,
        expires_at: datetime,
        match_anchors: Sequence[object] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> DisplayAmbientImpulseExposure:
        """Append one shown reserve-card exposure to the bounded history."""

        normalized_shown_at = _coerce_utc_datetime(shown_at, default=_utc_now()) or _utc_now()
        normalized_expires_at = _coerce_utc_datetime(
            expires_at,
            default=(normalized_shown_at + timedelta(minutes=10)),
        ) or (normalized_shown_at + timedelta(minutes=10))
        base_exposure = DisplayAmbientImpulseExposure(
            exposure_id=_exposure_id(
                shown_at=normalized_shown_at,
                topic_key=_compact_text(topic_key, max_len=96).casefold(),
                headline=_compact_text(headline, max_len=160),
            ),
            source=source,
            topic_key=topic_key,
            semantic_topic_key=semantic_topic_key or topic_key,
            title=title,
            headline=headline,
            body=body,
            action=action,
            attention_state=attention_state,
            shown_at=_format_timestamp(normalized_shown_at),
            expires_at=_format_timestamp(normalized_expires_at),
            match_anchors=_normalize_text_tuple(match_anchors, max_items=8, max_len=160),
            metadata=metadata,
        )
        self._ensure_storage_ready()
        with self._write_transaction() as connection:
            exposure = self._ensure_unique_exposure_id(connection, base_exposure)
            self._insert_exposure(connection, exposure)
            self._prune_to_max_entries(connection)
        return exposure

    def load_pending(
        self,
        *,
        now: datetime,
        max_age_hours: float = 12.0,
        limit: int = 6,
    ) -> tuple[DisplayAmbientImpulseExposure, ...]:
        """Return recent unresolved exposures in newest-first order."""

        if not self._bootstrapped and not _normalize_store_path(self.path).exists():
            return ()
        effective_now = _coerce_utc_datetime(now, default=_utc_now()) or _utc_now()
        minimum_shown_at = effective_now - timedelta(hours=max(0.1, float(max_age_hours)))
        try:
            self._ensure_storage_ready()
            with self._connect(read_only=True) as connection:
                rows = connection.execute(
                    """
                    SELECT
                        exposure_id,
                        source,
                        topic_key,
                        semantic_topic_key,
                        title,
                        headline,
                        body,
                        action,
                        attention_state,
                        shown_at,
                        expires_at,
                        match_anchors_json,
                        response_status,
                        response_sentiment,
                        response_at,
                        response_mode,
                        response_latency_seconds,
                        response_turn_id,
                        response_target,
                        response_summary,
                        metadata_json
                    FROM ambient_impulse_exposures
                    WHERE response_status = 'pending'
                      AND shown_at_epoch <= ?
                      AND shown_at_epoch >= ?
                    ORDER BY shown_at_epoch DESC, seq DESC
                    LIMIT ?
                    """,
                    (
                        effective_now.timestamp(),
                        minimum_shown_at.timestamp(),
                        max(1, int(limit)),
                    ),
                ).fetchall()
        except sqlite3.DatabaseError:
            _LOGGER.warning("Failed to load pending display ambient impulse history from %s.", self.path, exc_info=True)
            return ()
        return tuple(self._row_to_exposure(row) for row in rows)

    def topic_exposure_count(
        self,
        *,
        topic_key: str,
        now: datetime,
        within_days: float = 14.0,
    ) -> int:
        """Return how often the same normalized topic was shown recently."""

        normalized_topic = _compact_text(topic_key, max_len=96).casefold()
        if not normalized_topic:
            return 0
        if not self._bootstrapped and not _normalize_store_path(self.path).exists():
            return 0
        effective_now = _coerce_utc_datetime(now, default=_utc_now()) or _utc_now()
        minimum_shown_at = effective_now - timedelta(days=max(0.1, float(within_days)))
        try:
            self._ensure_storage_ready()
            with self._connect(read_only=True) as connection:
                row = connection.execute(
                    """
                    SELECT COUNT(*) AS exposure_count
                    FROM ambient_impulse_exposures
                    WHERE semantic_topic_key = ?
                      AND shown_at_epoch >= ?
                    """,
                    (normalized_topic, minimum_shown_at.timestamp()),
                ).fetchone()
        except sqlite3.DatabaseError:
            _LOGGER.warning("Failed to count display ambient impulse topic exposures from %s.", self.path, exc_info=True)
            return 0
        return int(row["exposure_count"]) if row is not None else 0

    def resolve_feedback(
        self,
        *,
        exposure_id: str,
        response_status: str,
        response_sentiment: str,
        response_at: datetime,
        response_mode: str,
        response_latency_seconds: float | None,
        response_turn_id: str | None,
        response_target: str | None,
        response_summary: str,
    ) -> DisplayAmbientImpulseExposure | None:
        """Resolve one exposure with the observed later user reaction."""

        normalized_exposure_id = _compact_text(exposure_id, max_len=64)
        if not normalized_exposure_id:
            return None
        normalized_response_at = _coerce_utc_datetime(response_at, default=_utc_now()) or _utc_now()
        self._ensure_storage_ready()
        with self._write_transaction() as connection:
            row = connection.execute(
                """
                SELECT
                    exposure_id,
                    source,
                    topic_key,
                    semantic_topic_key,
                    title,
                    headline,
                    body,
                    action,
                    attention_state,
                    shown_at,
                    expires_at,
                    match_anchors_json,
                    response_status,
                    response_sentiment,
                    response_at,
                    response_mode,
                    response_latency_seconds,
                    response_turn_id,
                    response_target,
                    response_summary,
                    metadata_json
                FROM ambient_impulse_exposures
                WHERE exposure_id = ?
                LIMIT 1
                """,
                (normalized_exposure_id,),
            ).fetchone()
            if row is None:
                return None
            resolved = self._row_to_exposure(row).with_feedback(
                response_status=response_status,
                response_sentiment=response_sentiment,
                response_at=normalized_response_at,
                response_mode=response_mode,
                response_latency_seconds=response_latency_seconds,
                response_turn_id=response_turn_id,
                response_target=response_target,
                response_summary=response_summary,
            )
            connection.execute(
                """
                UPDATE ambient_impulse_exposures
                SET
                    response_status = ?,
                    response_sentiment = ?,
                    response_at = ?,
                    response_at_epoch = ?,
                    response_mode = ?,
                    response_latency_seconds = ?,
                    response_turn_id = ?,
                    response_target = ?,
                    response_summary = ?
                WHERE exposure_id = ?
                """,
                (
                    resolved.response_status,
                    resolved.response_sentiment,
                    resolved.response_at,
                    _epoch_seconds(resolved.response_at) if resolved.response_at is not None else None,
                    resolved.response_mode,
                    resolved.response_latency_seconds,
                    resolved.response_turn_id,
                    resolved.response_target,
                    resolved.response_summary,
                    resolved.exposure_id,
                ),
            )
        return resolved

    def _ensure_storage_ready(self) -> None:
        """Create or migrate the backing SQLite store on first use."""

        if self._bootstrapped:
            return
        self.path = _normalize_store_path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._filesystem_lock():
            if self._bootstrapped:
                return
            imported_exposures = self._migrate_legacy_payload_if_needed()
            with self._connect_raw() as connection:
                _apply_sqlite_hardening(connection, read_only=False)
                self._initialize_database(connection)
                connection.execute("BEGIN IMMEDIATE")
                try:
                    if imported_exposures:
                        connection.execute("DELETE FROM ambient_impulse_exposures")
                        for exposure in self._deduplicate_exposures(imported_exposures):
                            self._insert_exposure(connection, exposure)
                        self._prune_to_max_entries(connection)
                except Exception:
                    connection.execute("ROLLBACK")
                    raise
                else:
                    connection.execute("COMMIT")
            _best_effort_owner_only(self.path, mode=0o600)
            self._bootstrapped = True

    def _migrate_legacy_payload_if_needed(self) -> tuple[DisplayAmbientImpulseExposure, ...]:
        """Migrate the legacy JSON payload at `path` into the SQLite store."""

        if not self.path.exists():
            return ()
        if self.path.is_dir():
            backup_path = _backup_path_for(self.path, suffix="invalid_dir")
            self.path.rename(backup_path)
            _LOGGER.warning(
                "Expected a file at %s but found a directory. Moved it to %s and starting fresh.",
                self.path,
                backup_path,
            )
            return ()
        try:
            with self.path.open("rb") as handle:
                header = handle.read(len(_SQLITE_HEADER))
        except OSError:
            _LOGGER.warning("Could not inspect history artifact %s.", self.path, exc_info=True)
            return ()
        if header == _SQLITE_HEADER:
            return ()
        original_path = self.path
        backup_suffix = "legacy_json"
        imported_exposures: tuple[DisplayAmbientImpulseExposure, ...] = ()
        try:
            if self.path.stat().st_size <= _MAX_LEGACY_IMPORT_BYTES:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                imported_exposures = _legacy_payload_to_exposures(payload)
            else:
                backup_suffix = "oversize_payload"
                _LOGGER.warning(
                    "Skipping legacy ambient impulse history import from %s because it exceeds %s bytes.",
                    self.path,
                    _MAX_LEGACY_IMPORT_BYTES,
                )
        except Exception:
            backup_suffix = "invalid_payload"
            _LOGGER.warning("Could not parse legacy ambient impulse history from %s.", self.path, exc_info=True)
        backup_path = _backup_path_for(self.path, suffix=backup_suffix)
        self.path.replace(backup_path)
        _best_effort_owner_only(backup_path, mode=0o600)
        if imported_exposures:
            _LOGGER.info(
                "Migrated %s display ambient impulse exposure entries from %s into SQLite.",
                len(imported_exposures),
                backup_path,
            )
        else:
            _LOGGER.info(
                "Moved non-SQLite ambient impulse history payload from %s to %s and started with an empty store.",
                original_path,
                backup_path,
            )
        return imported_exposures

    @contextmanager
    def _filesystem_lock(self) -> Iterable[None]:
        """Serialize first-run migration on POSIX filesystems."""

        lock_path = self.path.with_name(f"{self.path.name}.lock")
        file_descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            _best_effort_owner_only(lock_path, mode=0o600)
            if fcntl is not None:
                fcntl.flock(file_descriptor, fcntl.LOCK_EX)
            yield
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(file_descriptor, fcntl.LOCK_UN)
            finally:
                os.close(file_descriptor)

    def _connect_raw(self) -> sqlite3.Connection:
        """Open one raw SQLite connection to the history artifact."""

        return sqlite3.connect(
            self.path,
            timeout=_DEFAULT_BUSY_TIMEOUT_MS / 1000.0,
            isolation_level=None,
            detect_types=0,
            cached_statements=64,
        )

    def _connect(self, *, read_only: bool) -> sqlite3.Connection:
        """Open one hardened SQLite connection."""

        connection = self._connect_raw()
        try:
            _apply_sqlite_hardening(connection, read_only=read_only)
        except Exception:
            connection.close()
            raise
        return connection

    @contextmanager
    def _write_transaction(self) -> Iterable[sqlite3.Connection]:
        """Open one explicit write transaction with immediate locking."""

        with self._connect(read_only=False) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except Exception:
                connection.execute("ROLLBACK")
                raise
            else:
                connection.execute("COMMIT")

    def _initialize_database(self, connection: sqlite3.Connection) -> None:
        """Create the schema and persistent PRAGMA settings when missing."""

        connection.execute(f"PRAGMA journal_mode={_SQLITE_JOURNAL_MODE}")
        connection.execute(f"PRAGMA synchronous={_SQLITE_SYNCHRONOUS}")
        connection.execute("PRAGMA journal_size_limit=0")
        connection.execute(f"PRAGMA user_version={_SQLITE_USER_VERSION}")
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS ambient_impulse_exposures (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            exposure_id TEXT NOT NULL UNIQUE,
            source TEXT NOT NULL,
            topic_key TEXT NOT NULL,
            semantic_topic_key TEXT NOT NULL,
            title TEXT NOT NULL,
            headline TEXT NOT NULL,
            body TEXT NOT NULL,
            action TEXT NOT NULL,
            attention_state TEXT NOT NULL,
            shown_at TEXT NOT NULL,
            shown_at_epoch REAL NOT NULL,
            expires_at TEXT NOT NULL,
            expires_at_epoch REAL NOT NULL,
            match_anchors_json TEXT NOT NULL,
            response_status TEXT NOT NULL CHECK (response_status IN ({",".join(repr(item) for item in sorted(_ALLOWED_RESPONSE_STATUS))})),
            response_sentiment TEXT NOT NULL CHECK (response_sentiment IN ({",".join(repr(item) for item in sorted(_ALLOWED_RESPONSE_SENTIMENT))})),
            response_at TEXT,
            response_at_epoch REAL,
            response_mode TEXT NOT NULL,
            response_latency_seconds REAL,
            response_turn_id TEXT,
            response_target TEXT,
            response_summary TEXT NOT NULL,
            metadata_json TEXT
        ) STRICT
        """
        try:
            connection.execute(create_sql)
        except sqlite3.OperationalError as exc:
            if "STRICT" not in str(exc).upper():
                raise
            connection.execute(create_sql.replace(" STRICT", ""))
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS ambient_impulse_pending_shown_idx
            ON ambient_impulse_exposures (response_status, shown_at_epoch DESC, seq DESC)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS ambient_impulse_semantic_shown_idx
            ON ambient_impulse_exposures (semantic_topic_key, shown_at_epoch DESC, seq DESC)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS ambient_impulse_shown_idx
            ON ambient_impulse_exposures (shown_at_epoch DESC, seq DESC)
            """
        )

    def _row_to_exposure(self, row: sqlite3.Row) -> DisplayAmbientImpulseExposure:
        """Convert one SQLite row into the public exposure record."""

        raw_match_anchors = _json_loads_text(row["match_anchors_json"], default=[])
        raw_metadata = _json_loads_text(row["metadata_json"], default=None)
        return DisplayAmbientImpulseExposure(
            exposure_id=row["exposure_id"],
            source=row["source"],
            topic_key=row["topic_key"],
            semantic_topic_key=row["semantic_topic_key"],
            title=row["title"],
            headline=row["headline"],
            body=row["body"],
            action=row["action"],
            attention_state=row["attention_state"],
            shown_at=row["shown_at"],
            expires_at=row["expires_at"],
            match_anchors=_normalize_text_tuple(
                raw_match_anchors if isinstance(raw_match_anchors, Sequence) and not isinstance(raw_match_anchors, (str, bytes, bytearray)) else None,
                max_items=8,
                max_len=160,
            ),
            response_status=row["response_status"],
            response_sentiment=row["response_sentiment"],
            response_at=row["response_at"],
            response_mode=row["response_mode"],
            response_latency_seconds=_normalize_optional_float(row["response_latency_seconds"]),
            response_turn_id=row["response_turn_id"],
            response_target=row["response_target"],
            response_summary=row["response_summary"],
            metadata=raw_metadata if isinstance(raw_metadata, Mapping) else None,
        )

    def _insert_exposure(self, connection: sqlite3.Connection, exposure: DisplayAmbientImpulseExposure) -> None:
        """Insert one normalized exposure row."""

        metadata_payload = _normalize_mapping(exposure.metadata)
        connection.execute(
            """
            INSERT INTO ambient_impulse_exposures (
                exposure_id,
                source,
                topic_key,
                semantic_topic_key,
                title,
                headline,
                body,
                action,
                attention_state,
                shown_at,
                shown_at_epoch,
                expires_at,
                expires_at_epoch,
                match_anchors_json,
                response_status,
                response_sentiment,
                response_at,
                response_at_epoch,
                response_mode,
                response_latency_seconds,
                response_turn_id,
                response_target,
                response_summary,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                exposure.exposure_id,
                exposure.source,
                exposure.topic_key,
                exposure.semantic_key(),
                exposure.title,
                exposure.headline,
                exposure.body,
                exposure.action,
                exposure.attention_state,
                exposure.shown_at,
                _epoch_seconds(exposure.shown_at),
                exposure.expires_at,
                _epoch_seconds(exposure.expires_at),
                _json_dumps_compact(list(exposure.match_anchors)),
                exposure.response_status,
                exposure.response_sentiment,
                exposure.response_at,
                _epoch_seconds(exposure.response_at) if exposure.response_at is not None else None,
                exposure.response_mode,
                exposure.response_latency_seconds,
                exposure.response_turn_id,
                exposure.response_target,
                exposure.response_summary,
                _json_dumps_compact(metadata_payload) if metadata_payload is not None else None,
            ),
        )

    def _prune_to_max_entries(self, connection: sqlite3.Connection) -> None:
        """Trim the table to the configured bounded history size."""

        max_entries = max(1, int(self.max_entries))
        connection.execute(
            """
            DELETE FROM ambient_impulse_exposures
            WHERE seq NOT IN (
                SELECT seq
                FROM ambient_impulse_exposures
                ORDER BY seq DESC
                LIMIT ?
            )
            """,
            (max_entries,),
        )

    def _deduplicate_exposures(
        self,
        exposures: Sequence[DisplayAmbientImpulseExposure],
    ) -> tuple[DisplayAmbientImpulseExposure, ...]:
        """Ensure unique exposure IDs while preserving exposure order."""

        deduplicated: list[DisplayAmbientImpulseExposure] = []
        seen: set[str] = set()
        for exposure in exposures:
            candidate = exposure
            if candidate.exposure_id in seen or not candidate.exposure_id:
                counter = 1
                while True:
                    suffix = f":{counter}"
                    trimmed = _compact_text(f"{candidate.exposure_id[: max(0, 64 - len(suffix))]}{suffix}", max_len=64)
                    if trimmed and trimmed not in seen:
                        candidate = DisplayAmbientImpulseExposure(
                            exposure_id=trimmed,
                            source=candidate.source,
                            topic_key=candidate.topic_key,
                            semantic_topic_key=candidate.semantic_topic_key,
                            title=candidate.title,
                            headline=candidate.headline,
                            body=candidate.body,
                            action=candidate.action,
                            attention_state=candidate.attention_state,
                            shown_at=candidate.shown_at,
                            expires_at=candidate.expires_at,
                            match_anchors=candidate.match_anchors,
                            response_status=candidate.response_status,
                            response_sentiment=candidate.response_sentiment,
                            response_at=candidate.response_at,
                            response_mode=candidate.response_mode,
                            response_latency_seconds=candidate.response_latency_seconds,
                            response_turn_id=candidate.response_turn_id,
                            response_target=candidate.response_target,
                            response_summary=candidate.response_summary,
                            metadata=candidate.metadata,
                        )
                        break
                    counter += 1
            seen.add(candidate.exposure_id)
            deduplicated.append(candidate)
        return tuple(deduplicated)

    def _ensure_unique_exposure_id(
        self,
        connection: sqlite3.Connection,
        exposure: DisplayAmbientImpulseExposure,
    ) -> DisplayAmbientImpulseExposure:
        """De-duplicate the computed exposure id against existing rows."""

        candidate = exposure
        counter = 0
        while True:
            existing = connection.execute(
                "SELECT 1 FROM ambient_impulse_exposures WHERE exposure_id = ? LIMIT 1",
                (candidate.exposure_id,),
            ).fetchone()
            if existing is None and candidate.exposure_id:
                return candidate
            counter += 1
            suffix = f":{counter}"
            base = exposure.exposure_id or "ambient_exposure"
            trimmed = _compact_text(f"{base[: max(0, 64 - len(suffix))]}{suffix}", max_len=64)
            candidate = DisplayAmbientImpulseExposure(
                exposure_id=trimmed,
                source=exposure.source,
                topic_key=exposure.topic_key,
                semantic_topic_key=exposure.semantic_topic_key,
                title=exposure.title,
                headline=exposure.headline,
                body=exposure.body,
                action=exposure.action,
                attention_state=exposure.attention_state,
                shown_at=exposure.shown_at,
                expires_at=exposure.expires_at,
                match_anchors=exposure.match_anchors,
                response_status=exposure.response_status,
                response_sentiment=exposure.response_sentiment,
                response_at=exposure.response_at,
                response_mode=exposure.response_mode,
                response_latency_seconds=exposure.response_latency_seconds,
                response_turn_id=exposure.response_turn_id,
                response_target=exposure.response_target,
                response_summary=exposure.response_summary,
                metadata=exposure.metadata,
            )
