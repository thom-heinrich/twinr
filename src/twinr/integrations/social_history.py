# CHANGELOG: 2026-03-30
# BUG-1: Fixed calendar-inaccurate 3m/6m/1y lookback windows; these now use exact calendar boundaries instead of fixed-day approximations.
# BUG-2: Preserved terminal import outcomes (including failed) when the integration is disabled; the old code could silently hide the latest failure.
# BUG-3: Rejected stale/out-of-order status regressions and auto-stamped missing running/completed timestamps for consistent persisted state.
# BUG-4: Hardened parsing of persisted settings/enabled flags so malformed store data no longer crashes normalization or truthifies "false".
# SEC-1: Bounded and sanitized untrusted persisted text metadata to reduce downstream stored-XSS/log-forging risk in portal and log surfaces.
# IMP-1: Upgraded the state model to explicit StrEnum-backed source/lookback/status values and added last_import_updated_at for observability.
# IMP-2: Added schema_version and exact lookback_start() while keeping drop-in compatible from_record()/to_record() and lookback_duration.

"""Normalize social-history learning consent and import state.

This module keeps the user-approved social-history configuration independent
from the portal presentation and the WhatsApp runtime import path. The
integration store remains the single persisted source of truth for the opt-in,
selected source, lookback window, and latest import outcome.

Compared with the earlier version, this module now:
- computes calendar-aware lookback boundaries for month/year windows,
- preserves terminal outcomes while disabled,
- rejects stale status regressions from delayed workers,
- records `last_import_updated_at` for better observability,
- stores a lightweight settings `schema_version`, and
- sanitizes persisted free-text metadata before it reaches downstream UIs/logs.
"""

from __future__ import annotations

from calendar import monthrange
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Final

from twinr.integrations.store import ManagedIntegrationConfig


class SocialHistorySource(StrEnum):
    """Allowed social-history import sources."""

    WHATSAPP = "whatsapp"


class SocialHistoryLookbackKey(StrEnum):
    """Allowed bounded lookback windows."""

    LAST_4_WEEKS = "4w"
    LAST_3_MONTHS = "3m"
    LAST_6_MONTHS = "6m"
    LAST_1_YEAR = "1y"


class ImportStatus(StrEnum):
    """Normalized lifecycle states for one social-history import."""

    DISABLED = "disabled"
    IDLE = "idle"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


SOCIAL_HISTORY_LEARNING_INTEGRATION_ID: Final[str] = "social_history_learning"
SOCIAL_HISTORY_SOURCE_WHATSAPP: Final[str] = SocialHistorySource.WHATSAPP
SOCIAL_HISTORY_ALLOWED_SOURCES: Final[tuple[str, ...]] = tuple(
    member.value for member in SocialHistorySource
)
SOCIAL_HISTORY_LOOKBACK_OPTIONS: Final[tuple[tuple[str, str], ...]] = (
    (SocialHistoryLookbackKey.LAST_4_WEEKS.value, "Last 4 weeks"),
    (SocialHistoryLookbackKey.LAST_3_MONTHS.value, "Last 3 months"),
    (SocialHistoryLookbackKey.LAST_6_MONTHS.value, "Last 6 months"),
    (SocialHistoryLookbackKey.LAST_1_YEAR.value, "Last 1 year"),
)
# Legacy fixed-duration compatibility map. Exact calendar-aware boundaries are
# provided by SocialHistoryLearningConfig.lookback_start().
_LOOKBACK_DURATIONS: Final[dict[str, timedelta]] = {
    SocialHistoryLookbackKey.LAST_4_WEEKS.value: timedelta(weeks=4),
    SocialHistoryLookbackKey.LAST_3_MONTHS.value: timedelta(days=90),
    SocialHistoryLookbackKey.LAST_6_MONTHS.value: timedelta(days=183),
    SocialHistoryLookbackKey.LAST_1_YEAR.value: timedelta(days=365),
}
_LOOKBACK_LABELS: Final[dict[SocialHistoryLookbackKey, str]] = {
    SocialHistoryLookbackKey(key): label for key, label in SOCIAL_HISTORY_LOOKBACK_OPTIONS
}
_ALLOWED_IMPORT_STATUSES: Final[frozenset[str]] = frozenset(
    member.value for member in ImportStatus
)
_IN_PROGRESS_STATUSES: Final[frozenset[ImportStatus]] = frozenset(
    {ImportStatus.QUEUED, ImportStatus.RUNNING}
)
_TERMINAL_STATUSES: Final[frozenset[ImportStatus]] = frozenset(
    {ImportStatus.COMPLETED, ImportStatus.PARTIAL, ImportStatus.FAILED}
)
_SUCCESS_STATUSES: Final[frozenset[ImportStatus]] = frozenset(
    {ImportStatus.COMPLETED, ImportStatus.PARTIAL}
)
_RESTARTABLE_STATUSES: Final[frozenset[ImportStatus]] = frozenset(
    _TERMINAL_STATUSES | {ImportStatus.IDLE, ImportStatus.DISABLED}
)
_REQUEST_ID_MAX_LENGTH: Final[int] = 128
_DETAIL_MAX_LENGTH: Final[int] = 512
_ERROR_MAX_LENGTH: Final[int] = 1024
_ISO8601_MAX_LENGTH: Final[int] = 64
_SETTINGS_SCHEMA_VERSION: Final[str] = "2"
_CONTROL_TRANSLATION: Final[dict[int, str]] = {codepoint: " " for codepoint in range(32)}
_CONTROL_TRANSLATION[127] = " "


def _now_utc_iso8601() -> str:
    """Return one normalized UTC timestamp for persisted state."""

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _normalize_settings_mapping(value: object) -> dict[str, object]:
    """Best-effort normalization for persisted settings blobs."""

    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    try:
        return {str(key): item for key, item in dict(value).items()}  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return {}


def _normalize_bool(value: object, *, default: bool = False) -> bool:
    """Normalize permissive persisted booleans."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, float):
        return value != 0.0
    if isinstance(value, str):
        candidate = value.strip().casefold()
        if candidate in {"1", "true", "yes", "on", "enabled"}:
            return True
        if candidate in {"0", "false", "no", "off", "disabled", ""}:
            return False
    return default


def _normalize_text(
    value: object,
    *,
    default: str = "",
    max_length: int | None = None,
) -> str:
    """Normalize one untrusted text value into bounded single-line plain text."""

    if value is None:
        return default
    text = str(value).translate(_CONTROL_TRANSLATION)
    text = " ".join(text.split())
    if max_length is not None:
        text = text[:max_length].rstrip()
    return text or default


def _normalize_optional_text(value: object, *, max_length: int) -> str | None:
    """Normalize one optional untrusted text value."""

    text = _normalize_text(value, max_length=max_length)
    return text or None


def _normalize_source(value: object) -> SocialHistorySource:
    """Normalize one persisted source key."""

    candidate = _normalize_text(
        value,
        default=SocialHistorySource.WHATSAPP.value,
        max_length=32,
    ).casefold()
    try:
        return SocialHistorySource(candidate)
    except ValueError:
        return SocialHistorySource.WHATSAPP


def _normalize_lookback_key(value: object) -> SocialHistoryLookbackKey:
    """Normalize one persisted lookback key."""

    candidate = _normalize_text(
        value,
        default=SocialHistoryLookbackKey.LAST_3_MONTHS.value,
        max_length=8,
    ).casefold()
    try:
        return SocialHistoryLookbackKey(candidate)
    except ValueError:
        return SocialHistoryLookbackKey.LAST_3_MONTHS


def _normalize_iso8601(value: object | None) -> str | None:
    """Normalize one timezone-aware ISO 8601 timestamp into UTC."""

    text = _normalize_text(value, max_length=_ISO8601_MAX_LENGTH)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC).replace(microsecond=0).isoformat()


def _parse_iso8601(value: object | None) -> datetime | None:
    """Parse one normalized ISO 8601 timestamp back into datetime."""

    normalized = _normalize_iso8601(value)
    if normalized is None:
        return None
    return datetime.fromisoformat(normalized)


def _normalize_non_negative_int(value: object, *, default: int = 0) -> int:
    """Normalize one persisted counter into a non-negative integer."""

    try:
        if isinstance(value, bool):
            parsed = int(value)
        elif isinstance(value, int):
            parsed = value
        elif isinstance(value, float):
            parsed = int(value)
        elif isinstance(value, str):
            parsed = int(value.strip())
        else:
            return default
    except (TypeError, ValueError, OverflowError):
        return default
    return max(parsed, 0)


def _normalize_import_status(value: object, *, enabled: bool) -> ImportStatus:
    """Normalize one persisted import status."""

    candidate = _normalize_text(
        value,
        default=ImportStatus.IDLE.value if enabled else ImportStatus.DISABLED.value,
        max_length=32,
    ).casefold()
    try:
        return ImportStatus(candidate)
    except ValueError:
        return ImportStatus.IDLE if enabled else ImportStatus.DISABLED


def _coerce_aware_utc(reference: datetime | None) -> datetime:
    """Coerce one datetime reference into aware UTC."""

    candidate = datetime.now(UTC) if reference is None else reference
    if candidate.tzinfo is None:
        candidate = candidate.replace(tzinfo=UTC)
    return candidate.astimezone(UTC).replace(microsecond=0)


def _subtract_calendar_months(reference: datetime, months: int) -> datetime:
    """Subtract whole calendar months without crossing month boundaries."""

    absolute_month = (reference.year * 12 + (reference.month - 1)) - months
    year, month_index = divmod(absolute_month, 12)
    month = month_index + 1
    day = min(reference.day, monthrange(year, month)[1])
    return reference.replace(year=year, month=month, day=day)


def _lookback_start(reference: datetime, lookback_key: SocialHistoryLookbackKey) -> datetime:
    """Compute the exact calendar-aware lower bound for one lookback selection."""

    if lookback_key is SocialHistoryLookbackKey.LAST_4_WEEKS:
        return reference - timedelta(weeks=4)
    if lookback_key is SocialHistoryLookbackKey.LAST_3_MONTHS:
        return _subtract_calendar_months(reference, 3)
    if lookback_key is SocialHistoryLookbackKey.LAST_6_MONTHS:
        return _subtract_calendar_months(reference, 6)
    if lookback_key is SocialHistoryLookbackKey.LAST_1_YEAR:
        return _subtract_calendar_months(reference, 12)
    raise ValueError(f"Unsupported lookback key: {lookback_key!s}")


def _max_datetime(*values: datetime | None) -> datetime | None:
    """Return the latest non-null datetime."""

    present = [value for value in values if value is not None]
    return max(present) if present else None


@dataclass(frozen=True, slots=True)
class SocialHistoryLearningConfig:
    """Represent one normalized social-history learning consent record."""

    enabled: bool = False
    source: SocialHistorySource = SocialHistorySource.WHATSAPP
    lookback_key: SocialHistoryLookbackKey = SocialHistoryLookbackKey.LAST_3_MONTHS
    last_import_status: ImportStatus = ImportStatus.DISABLED
    last_import_request_id: str | None = None
    last_import_started_at: str | None = None
    last_import_finished_at: str | None = None
    last_import_updated_at: str | None = None
    last_import_error: str | None = None
    last_import_detail: str | None = None
    last_import_messages: int = 0
    last_import_turns: int = 0
    last_import_chats: int = 0
    last_import_oldest_at: str | None = None
    last_import_newest_at: str | None = None

    def __post_init__(self) -> None:
        """Normalize all fields even when the dataclass is constructed directly."""

        enabled = _normalize_bool(self.enabled)
        source = _normalize_source(self.source)
        lookback_key = _normalize_lookback_key(self.lookback_key)
        status = _normalize_import_status(self.last_import_status, enabled=enabled)
        request_id = _normalize_optional_text(
            self.last_import_request_id,
            max_length=_REQUEST_ID_MAX_LENGTH,
        )
        started_at = _normalize_iso8601(self.last_import_started_at)
        finished_at = _normalize_iso8601(self.last_import_finished_at)
        updated_at = _normalize_iso8601(self.last_import_updated_at)
        error = _normalize_optional_text(self.last_import_error, max_length=_ERROR_MAX_LENGTH)
        detail = _normalize_optional_text(self.last_import_detail, max_length=_DETAIL_MAX_LENGTH)
        messages = _normalize_non_negative_int(self.last_import_messages)
        turns = _normalize_non_negative_int(self.last_import_turns)
        chats = _normalize_non_negative_int(self.last_import_chats)
        oldest_at = _normalize_iso8601(self.last_import_oldest_at)
        newest_at = _normalize_iso8601(self.last_import_newest_at)

        if not enabled and status not in _TERMINAL_STATUSES:
            status = ImportStatus.DISABLED
        if status in _IN_PROGRESS_STATUSES:
            finished_at = None
            error = None
        latest_known_at = _max_datetime(
            _parse_iso8601(updated_at),
            _parse_iso8601(finished_at),
            _parse_iso8601(started_at),
        )
        updated_at = latest_known_at.isoformat() if latest_known_at is not None else updated_at

        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "lookback_key", lookback_key)
        object.__setattr__(self, "last_import_status", status)
        object.__setattr__(self, "last_import_request_id", request_id)
        object.__setattr__(self, "last_import_started_at", started_at)
        object.__setattr__(self, "last_import_finished_at", finished_at)
        object.__setattr__(self, "last_import_updated_at", updated_at)
        object.__setattr__(self, "last_import_error", error)
        object.__setattr__(self, "last_import_detail", detail)
        object.__setattr__(self, "last_import_messages", messages)
        object.__setattr__(self, "last_import_turns", turns)
        object.__setattr__(self, "last_import_chats", chats)
        object.__setattr__(self, "last_import_oldest_at", oldest_at)
        object.__setattr__(self, "last_import_newest_at", newest_at)

    @classmethod
    def from_record(
        cls,
        record: ManagedIntegrationConfig | None,
    ) -> "SocialHistoryLearningConfig":
        """Build one normalized config view from a managed integration record."""

        settings = {} if record is None else _normalize_settings_mapping(getattr(record, "settings", {}))
        return cls(
            enabled=False if record is None else _normalize_bool(getattr(record, "enabled", False)),
            source=settings.get("source"),
            lookback_key=settings.get("lookback_key"),
            last_import_status=settings.get("last_import_status"),
            last_import_request_id=settings.get("last_import_request_id"),
            last_import_started_at=settings.get("last_import_started_at"),
            last_import_finished_at=settings.get("last_import_finished_at"),
            last_import_updated_at=settings.get("last_import_updated_at"),
            last_import_error=settings.get("last_import_error"),
            last_import_detail=settings.get("last_import_detail"),
            last_import_messages=settings.get("last_import_messages"),
            last_import_turns=settings.get("last_import_turns"),
            last_import_chats=settings.get("last_import_chats"),
            last_import_oldest_at=settings.get("last_import_oldest_at"),
            last_import_newest_at=settings.get("last_import_newest_at"),
        )

    @property
    def lookback_duration(self) -> timedelta:
        """Return one exact current-duration view of the selected lookback window."""

        reference = _coerce_aware_utc(None)
        return reference - self.lookback_start(reference)

    @property
    def lookback_label(self) -> str:
        """Return the operator-facing label for the selected lookback window."""

        return _LOOKBACK_LABELS[self.lookback_key]

    @property
    def latest_import_event_at(self) -> datetime | None:
        """Return the latest known import event timestamp."""

        return _max_datetime(
            _parse_iso8601(self.last_import_updated_at),
            _parse_iso8601(self.last_import_finished_at),
            _parse_iso8601(self.last_import_started_at),
        )

    def lookback_start(self, reference: datetime | None = None) -> datetime:
        """Return the exact lower-bound timestamp for the selected lookback window."""

        normalized_reference = _coerce_aware_utc(reference)
        return _lookback_start(normalized_reference, self.lookback_key)

    def to_record(self) -> ManagedIntegrationConfig:
        """Serialize the normalized config back into one managed record."""

        settings: dict[str, str] = {
            "schema_version": _SETTINGS_SCHEMA_VERSION,
            "source": self.source.value,
            "lookback_key": self.lookback_key.value,
            "last_import_status": self.last_import_status.value,
            "last_import_messages": str(self.last_import_messages),
            "last_import_turns": str(self.last_import_turns),
            "last_import_chats": str(self.last_import_chats),
        }
        optional_settings = {
            "last_import_request_id": self.last_import_request_id,
            "last_import_started_at": self.last_import_started_at,
            "last_import_finished_at": self.last_import_finished_at,
            "last_import_updated_at": self.last_import_updated_at,
            "last_import_error": self.last_import_error,
            "last_import_detail": self.last_import_detail,
            "last_import_oldest_at": self.last_import_oldest_at,
            "last_import_newest_at": self.last_import_newest_at,
        }
        for key, value in optional_settings.items():
            if value:
                settings[key] = value
        return ManagedIntegrationConfig(
            integration_id=SOCIAL_HISTORY_LEARNING_INTEGRATION_ID,
            enabled=self.enabled,
            settings=settings,
        )


def _is_new_import_run(
    current: SocialHistoryLearningConfig,
    *,
    incoming_status: ImportStatus,
    incoming_request_id: str | None,
    incoming_started_at: str | None,
) -> bool:
    """Return whether one incoming state represents a fresh import run."""

    if incoming_status not in _IN_PROGRESS_STATUSES:
        return False
    if incoming_request_id and incoming_request_id != current.last_import_request_id:
        return True
    if current.last_import_status in _RESTARTABLE_STATUSES:
        if incoming_status is ImportStatus.QUEUED:
            return True
        if incoming_status is ImportStatus.RUNNING:
            return incoming_request_id is not None or incoming_started_at is not None
    return False


def _should_ignore_stale_update(
    current: SocialHistoryLearningConfig,
    *,
    incoming_status: ImportStatus,
    incoming_request_id: str | None,
    incoming_started_at: str | None,
    incoming_finished_at: str | None,
    is_new_run: bool,
) -> bool:
    """Reject delayed worker updates that would regress the persisted lifecycle."""

    if is_new_run:
        return False

    same_request = (
        incoming_request_id is not None
        and incoming_request_id == current.last_import_request_id
    )
    unknown_request = incoming_request_id is None
    current_event_at = current.latest_import_event_at
    incoming_event_at = _max_datetime(
        _parse_iso8601(incoming_finished_at),
        _parse_iso8601(incoming_started_at),
    )

    if current.last_import_status in _TERMINAL_STATUSES:
        if incoming_status in _IN_PROGRESS_STATUSES:
            # Terminal -> in-progress without a new request identity is usually one
            # delayed worker event and should not clobber the latest finished state.
            return True
        if incoming_status in _TERMINAL_STATUSES and (same_request or unknown_request):
            if incoming_event_at is None:
                return True
            if current_event_at is not None and incoming_event_at <= current_event_at:
                return True

    if current.last_import_status is ImportStatus.RUNNING and incoming_status is ImportStatus.QUEUED:
        if unknown_request and incoming_started_at is None:
            return True

    if incoming_event_at is not None and current_event_at is not None:
        if incoming_event_at < current_event_at and (same_request or unknown_request):
            return True
    return False


def social_history_record_with_import_state(
    record: ManagedIntegrationConfig | None,
    *,
    status: str,
    request_id: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    updated_at: str | None = None,
    detail: str | None = None,
    error: str | None = None,
    messages: int | None = None,
    turns: int | None = None,
    chats: int | None = None,
    oldest_at: str | None = None,
    newest_at: str | None = None,
) -> ManagedIntegrationConfig:
    """Return one updated record with normalized import-status metadata."""

    current = SocialHistoryLearningConfig.from_record(record)
    normalized_status = _normalize_import_status(status, enabled=current.enabled)

    # Ignore non-terminal worker churn after opt-out so we do not erase the last
    # meaningful terminal outcome or resurrect in-progress state while disabled.
    if not current.enabled and normalized_status not in _TERMINAL_STATUSES:
        return current.to_record()

    normalized_request_id = _normalize_optional_text(
        request_id,
        max_length=_REQUEST_ID_MAX_LENGTH,
    )
    normalized_started_at = _normalize_iso8601(started_at)
    normalized_finished_at = _normalize_iso8601(finished_at)
    normalized_updated_at = _normalize_iso8601(updated_at) or _now_utc_iso8601()
    normalized_detail = _normalize_optional_text(detail, max_length=_DETAIL_MAX_LENGTH)
    normalized_error = _normalize_optional_text(error, max_length=_ERROR_MAX_LENGTH)
    normalized_oldest_at = _normalize_iso8601(oldest_at)
    normalized_newest_at = _normalize_iso8601(newest_at)

    is_new_run = _is_new_import_run(
        current,
        incoming_status=normalized_status,
        incoming_request_id=normalized_request_id,
        incoming_started_at=normalized_started_at,
    )
    if _should_ignore_stale_update(
        current,
        incoming_status=normalized_status,
        incoming_request_id=normalized_request_id,
        incoming_started_at=normalized_started_at,
        incoming_finished_at=normalized_finished_at,
        is_new_run=is_new_run,
    ):
        return current.to_record()

    resolved_request_id = (
        normalized_request_id
        if normalized_request_id is not None
        else (None if is_new_run else current.last_import_request_id)
    )
    resolved_started_at = (
        normalized_started_at
        or (
            normalized_updated_at
            if normalized_status is ImportStatus.RUNNING
            and (is_new_run or current.last_import_started_at is None)
            else current.last_import_started_at
        )
    )
    if normalized_status is ImportStatus.QUEUED and is_new_run and normalized_started_at is None:
        resolved_started_at = None

    resolved_finished_at = (
        normalized_finished_at
        or (
            normalized_updated_at
            if normalized_status in _TERMINAL_STATUSES
            else current.last_import_finished_at
        )
    )
    if normalized_status in _IN_PROGRESS_STATUSES:
        resolved_finished_at = None

    if normalized_status in _IN_PROGRESS_STATUSES | _SUCCESS_STATUSES:
        resolved_error = None
    elif normalized_status is ImportStatus.FAILED:
        resolved_error = (
            normalized_error
            if normalized_error is not None
            else current.last_import_error
        )
    else:
        resolved_error = current.last_import_error

    resolved_detail = normalized_detail if is_new_run else (
        normalized_detail if normalized_detail is not None else current.last_import_detail
    )

    updated = replace(
        current,
        last_import_status=normalized_status,
        last_import_request_id=resolved_request_id,
        last_import_started_at=resolved_started_at,
        last_import_finished_at=resolved_finished_at,
        last_import_updated_at=(
            normalized_finished_at
            or resolved_finished_at
            or normalized_started_at
            or resolved_started_at
            or normalized_updated_at
        ),
        last_import_error=resolved_error,
        last_import_detail=resolved_detail,
        last_import_messages=(
            current.last_import_messages
            if messages is None
            else _normalize_non_negative_int(messages)
        ),
        last_import_turns=(
            current.last_import_turns
            if turns is None
            else _normalize_non_negative_int(turns)
        ),
        last_import_chats=(
            current.last_import_chats
            if chats is None
            else _normalize_non_negative_int(chats)
        ),
        last_import_oldest_at=normalized_oldest_at or current.last_import_oldest_at,
        last_import_newest_at=normalized_newest_at or current.last_import_newest_at,
    )
    return updated.to_record()