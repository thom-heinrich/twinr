"""Apply retention and archival policy to long-term memory objects.

This module classifies long-term memories for keeping, expiration,
archival, or pruning with explicit timezone-aware comparisons so runtime
hosts do not silently change retention behavior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from twinr.memory.longterm.core.ontology import kind_matches
from twinr.memory.longterm.core.models import LongTermMemoryObjectV1, LongTermRetentionResultV1

logger = logging.getLogger(__name__)

_DEFAULT_TIMEZONE_NAME = "Europe/Berlin"
_ACTIVE_STATUSES = frozenset({"active", "candidate", "uncertain"})
_TERMINAL_STATUSES = frozenset({"superseded", "invalid", "expired"})


# AUDIT-FIX(#6): Type-guard and trim persisted values before ISO parsing.
# AUDIT-FIX(#2): Preserve timestamp inputs so later logic can compare full datetimes instead of only dates.
def _parse_iso_date(value: str | None) -> datetime | None:
    if value is None or not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        try:
            return datetime.fromisoformat(f"{candidate}T00:00:00")
        except ValueError:
            return None


def _is_date_only_iso(value: str) -> bool:
    candidate = value.strip()
    return (
        len(candidate) == 10
        and candidate[4] == "-"
        and candidate[7] == "-"
        and "T" not in candidate
        and " " not in candidate
    )


def _coerce_bool(value: object, *, default: bool, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    elif isinstance(value, (int, float)):
        return bool(value)

    logger.warning("Invalid %s=%r; falling back to %r", field_name, value, default)
    return default


def _coerce_int(value: object, *, default: int, minimum: int, field_name: str) -> int:
    if isinstance(value, bool):
        logger.warning(
            "Boolean %s=%r is invalid here; falling back to %r",
            field_name,
            value,
            default,
        )
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; falling back to %r", field_name, value, default)
        return default
    if parsed < minimum:
        logger.warning(
            "%s=%r is below minimum %r; clamping to %r",
            field_name,
            value,
            minimum,
            minimum,
        )
        return minimum
    return parsed


def _coerce_timezone_name(value: object, *, default: str) -> str:
    if not isinstance(value, str) or not value.strip():
        logger.warning("Invalid timezone_name=%r; falling back to %r", value, default)
        return default

    candidate = value.strip()
    try:
        ZoneInfo(candidate)
    except ZoneInfoNotFoundError:
        logger.warning("Unknown timezone_name=%r; falling back to %r", candidate, default)
        return default
    return candidate


@dataclass(frozen=True, slots=True)
class LongTermRetentionPolicy:
    """Classify long-term memory objects for keep, expire, archive, or prune.

    Attributes:
        timezone_name: Local timezone used when normalizing naive datetimes.
        mode: Named policy mode used by callers for configuration grouping.
        archive_enabled: Whether archival is preferred over destructive prune
            for supported object classes.
        ephemeral_episode_days: Maximum age for transient episode objects.
        ephemeral_observation_days: Maximum age for transient observations.
        stale_status_archive_days: Maximum age before terminal-status objects
            are archived.
        summary_archive_days: Maximum age before summary objects are archived.
    """

    timezone_name: str = "Europe/Berlin"
    mode: str = "conservative"
    archive_enabled: bool = True
    ephemeral_episode_days: int = 7
    ephemeral_observation_days: int = 2
    stale_status_archive_days: int = 3
    summary_archive_days: int = 30

    # AUDIT-FIX(#3): Coerce stringly .env-style config values up front so retention logic cannot crash or invert booleans.
    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "timezone_name",
            _coerce_timezone_name(self.timezone_name, default=_DEFAULT_TIMEZONE_NAME),
        )
        object.__setattr__(
            self,
            "mode",
            self.mode.strip() if isinstance(self.mode, str) and self.mode.strip() else "conservative",
        )
        object.__setattr__(
            self,
            "archive_enabled",
            _coerce_bool(self.archive_enabled, default=True, field_name="archive_enabled"),
        )
        object.__setattr__(
            self,
            "ephemeral_episode_days",
            _coerce_int(
                self.ephemeral_episode_days,
                default=7,
                minimum=1,
                field_name="ephemeral_episode_days",
            ),
        )
        object.__setattr__(
            self,
            "ephemeral_observation_days",
            _coerce_int(
                self.ephemeral_observation_days,
                default=2,
                minimum=1,
                field_name="ephemeral_observation_days",
            ),
        )
        object.__setattr__(
            self,
            "stale_status_archive_days",
            _coerce_int(
                self.stale_status_archive_days,
                default=3,
                minimum=1,
                field_name="stale_status_archive_days",
            ),
        )
        object.__setattr__(
            self,
            "summary_archive_days",
            _coerce_int(
                self.summary_archive_days,
                default=30,
                minimum=1,
                field_name="summary_archive_days",
            ),
        )

    def _policy_zone(self) -> tzinfo:
        try:
            return ZoneInfo(self.timezone_name)
        except ZoneInfoNotFoundError:
            if self.timezone_name != _DEFAULT_TIMEZONE_NAME:
                logger.warning(
                    "Unknown timezone_name=%r at runtime; falling back to %r",
                    self.timezone_name,
                    _DEFAULT_TIMEZONE_NAME,
                )
                try:
                    return ZoneInfo(_DEFAULT_TIMEZONE_NAME)
                except ZoneInfoNotFoundError:
                    pass
            logger.error(
                "Timezone database unavailable for %r and fallback %r; using UTC",
                self.timezone_name,
                _DEFAULT_TIMEZONE_NAME,
            )
            return timezone.utc

    # AUDIT-FIX(#1): Normalize every datetime through the policy timezone and attach a deterministic zone to naive values.
    def _normalize_datetime(self, value: datetime, *, zone: tzinfo) -> datetime:
        if not isinstance(value, datetime):
            raise TypeError(f"expected datetime, got {type(value)!r}")
        if value.tzinfo is None:
            return value.replace(tzinfo=zone)
        return value.astimezone(zone)

    def _reference_now(self, now: datetime | None) -> datetime:
        zone = self._policy_zone()
        if now is None:
            return datetime.now(zone)
        return self._normalize_datetime(now, zone=zone)

    def _age(self, value: datetime, *, now: datetime, zone: tzinfo) -> timedelta:
        return now - self._normalize_datetime(value, zone=zone)

    def apply(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
        now: datetime | None = None,
    ) -> LongTermRetentionResultV1:
        """Apply the retention policy to the current memory snapshot.

        Args:
            objects: Current long-term memory objects to classify.
            now: Optional reference timestamp. Defaults to the current time in
                the configured timezone.

        Returns:
            A retention result containing kept objects plus any expired,
            archived, or pruned outputs.
        """

        reference = self._reference_now(now)
        reference_utc = reference.astimezone(timezone.utc)
        kept: list[LongTermMemoryObjectV1] = []
        expired: list[LongTermMemoryObjectV1] = []
        archived: list[LongTermMemoryObjectV1] = []
        pruned_ids: list[str] = []

        for item in objects:
            try:
                # AUDIT-FIX(#7): Canonicalize once and persist the canonical form so decision inputs and outputs stay aligned.
                canonical_item = item.canonicalized()
                action = self._classify(item=canonical_item, now=reference)
                if action == "keep":
                    kept.append(canonical_item)
                elif action == "expire":
                    updated = (
                        canonical_item
                        if canonical_item.status == "expired"
                        # AUDIT-FIX(#5): Refresh updated_at when status changes to expired so stale-status archival uses the expiration time.
                        else canonical_item.with_updates(status="expired", updated_at=reference)
                    )
                    kept.append(updated)
                    expired.append(updated)
                elif action == "archive":
                    archived.append(
                        canonical_item.with_updates(
                            archived_at=reference_utc.isoformat(),
                            updated_at=reference,
                        )
                    )
                elif action == "prune":
                    pruned_ids.append(canonical_item.memory_id)
                else:
                    logger.warning(
                        "Unknown retention action=%r for memory_id=%s; keeping object",
                        action,
                        getattr(canonical_item, "memory_id", "<unknown>"),
                    )
                    kept.append(canonical_item)
            except Exception:
                # AUDIT-FIX(#4): Isolate failures per object so one corrupt record cannot abort retention for every other object.
                logger.exception(
                    "Retention policy failed for memory_id=%s; keeping original object",
                    getattr(item, "memory_id", "<unknown>"),
                )
                kept.append(item)

        return LongTermRetentionResultV1(
            kept_objects=tuple(kept),
            expired_objects=tuple(expired),
            pruned_memory_ids=tuple(pruned_ids),
            archived_objects=tuple(archived),
        )

    def _classify(self, *, item: LongTermMemoryObjectV1, now: datetime) -> str:
        zone = self._policy_zone()
        if item.kind == "episode":
            age = self._age(item.updated_at, now=now, zone=zone)
            if age > timedelta(days=self.ephemeral_episode_days):
                return "archive" if self.archive_enabled else "prune"
            return "keep"
        if kind_matches(item.kind, "observation", item.attributes):
            age = self._age(item.updated_at, now=now, zone=zone)
            if age > timedelta(days=self.ephemeral_observation_days):
                return "prune"
            return "keep"
        if item.status in _TERMINAL_STATUSES:
            age = self._age(item.updated_at, now=now, zone=zone)
            if self.archive_enabled and age > timedelta(days=self.stale_status_archive_days):
                return "archive"
            return "keep"
        if item.kind == "summary":
            age = self._age(item.updated_at, now=now, zone=zone)
            if self.archive_enabled and age > timedelta(days=self.summary_archive_days):
                return "archive"
            return "keep"
        if item.valid_to and item.status in _ACTIVE_STATUSES:
            valid_to = _parse_iso_date(item.valid_to)
            if valid_to is None:
                logger.warning(
                    "Invalid valid_to=%r for memory_id=%s; keeping object",
                    item.valid_to,
                    getattr(item, "memory_id", "<unknown>"),
                )
            else:
                # AUDIT-FIX(#2): Keep full timestamp precision for datetime strings; only pure YYYY-MM-DD values use date-level expiry.
                if _is_date_only_iso(item.valid_to):
                    if valid_to.date() < now.date():
                        return "expire"
                elif self._normalize_datetime(valid_to, zone=zone) < now:
                    return "expire"
        return "keep"


__all__ = ["LongTermRetentionPolicy"]
