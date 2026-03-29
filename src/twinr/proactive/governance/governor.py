# CHANGELOG: 2026-03-29
# BUG-1: Replaced wall-clock cooldown/TTL/window enforcement with monotonic_ns-based policy math.
# BUG-2: Added hard upper bounds for env/config numeric fields to prevent timedelta overflow and runaway retention.
# BUG-3: Replaced repeated full-history scans with deque/index-based rolling counters for deterministic Pi-class latency.
# BUG-4: Presence-session budgets no longer collapse to the short follow-up timeout when no governor-specific window is configured.
# SEC-1: Sanitized control characters and bounded operator-facing text to reduce log/control-sequence injection risk.
# SEC-2: Canonicalized oversized source identifiers to a bounded digest-backed form to prevent memory amplification.
# IMP-1: Decision responses now expose retry_after_s and blocked_until for better orchestration/backoff.
# IMP-2: Added dedicated presence-session window config aliases while remaining backward-compatible with voice_orchestrator_follow_up_timeout_s.

"""Gate proactive delivery reservations with bounded cooldown policy.

This module owns the in-memory governor used by Twinr runtime workflows to
rate-limit proactive speech prompts, track one active reservation, and retain
bounded delivery history for later audit and policy diagnostics. Import public
types from ``twinr.proactive.governance`` or ``twinr.proactive``.
"""

from __future__ import annotations

import hashlib
import math
import time
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from twinr.agent.base_agent.config import TwinrConfig


_SOURCE_KINDS = frozenset({"social", "reminder", "automation", "longterm"})
_CHANNELS = frozenset({"speech", "print", "display"})
_OUTCOMES = frozenset({"delivered", "skipped"})
_BOOL_TRUE = frozenset({"1", "true", "t", "yes", "y", "on"})
_BOOL_FALSE = frozenset({"0", "false", "f", "no", "n", "off"})

_DEFAULT_TIMEZONE_NAME = "UTC"
_DEFAULT_GLOBAL_PROMPT_COOLDOWN_S = 0.0
_DEFAULT_SOURCE_REPEAT_COOLDOWN_S = 0.0
_DEFAULT_WINDOW_PROMPT_LIMIT = 1
_DEFAULT_WINDOW_S = 60.0
_DEFAULT_PRESENCE_SESSION_PROMPT_LIMIT = 1
_DEFAULT_PRESENCE_GRACE_S = 60.0
_DEFAULT_ACTIVE_RESERVATION_TTL_S = 30.0
_DEFAULT_HISTORY_LIMIT = 128

_MIN_HISTORY_LIMIT = 16
_MAX_HISTORY_LIMIT = 4096
_MAX_COOLDOWN_S = 31_536_000.0  # 365 days
_MAX_WINDOW_S = 31_536_000.0  # 365 days
_MAX_ACTIVE_RESERVATION_TTL_S = 86_400.0  # 24h
_MAX_PROMPT_LIMIT = 100_000

_MAX_SOURCE_ID_CHARS = 256
_MAX_SUMMARY_CHARS = 512
_MAX_REASON_CHARS = 256
_MAX_TOKEN_CHARS = 128
_MIN_IDENTIFIER_HEAD_CHARS = 16

_MIN_RETENTION_S = 60.0
_FINALIZED_CACHE_BUFFER_MULTIPLIER = 2.0
_NS_PER_SECOND = 1_000_000_000


def _strip_control_chars(value: str) -> str:
    """Remove non-whitespace control characters from text."""

    cleaned: list[str] = []
    for char in value:
        if char.isspace():
            cleaned.append(" ")
            continue
        if unicodedata.category(char).startswith("C"):
            continue
        cleaned.append(char)
    return "".join(cleaned)


def _normalize_text(value: str | None) -> str:
    """Normalize optional text into one printable trimmed line."""

    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError("text value must be a str or None.")
    return " ".join(_strip_control_chars(value).split()).strip()


def _clip_text(value: str, *, max_chars: int) -> str:
    """Clip normalized text to one bounded length."""

    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return f"{value[: max_chars - 3]}..."


def _normalize_choice(value: str, *, field_name: str, allowed: frozenset[str]) -> str:
    """Normalize and validate one bounded string enum value."""

    normalized = _normalize_text(value)
    if normalized not in allowed:
        raise ValueError(f"{field_name} must be one of: {', '.join(sorted(allowed))}.")
    return normalized


def _normalize_required_text(value: str, *, field_name: str) -> str:
    """Normalize required text and reject blanks."""

    normalized = _normalize_text(value)
    if not normalized:
        raise ValueError(f"{field_name} is required.")
    return normalized


def _normalize_identifier(value: str, *, field_name: str, max_chars: int = _MAX_SOURCE_ID_CHARS) -> str:
    """Normalize one identifier and bound memory without losing uniqueness."""

    normalized = _normalize_required_text(value, field_name=field_name)
    if len(normalized) <= max_chars:
        return normalized
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    head_chars = max(_MIN_IDENTIFIER_HEAD_CHARS, max_chars - len(digest) - 1)
    # BREAKING: oversized source IDs are canonicalized into prefix+sha256 to stay bounded in memory.
    return f"{normalized[:head_chars]}~{digest}"


def _normalize_bounded_text(
    value: str | None,
    *,
    field_name: str,
    max_chars: int,
    required: bool = False,
) -> str | None:
    """Normalize free text and clip oversized payloads to a safe length."""

    normalized = _normalize_text(value)
    if required and not normalized:
        raise ValueError(f"{field_name} is required.")
    if not normalized:
        return None
    return _clip_text(normalized, max_chars=max_chars)


def _normalize_token(value: str, *, field_name: str) -> str:
    """Normalize a reservation token and reject oversized external values."""

    normalized = _normalize_required_text(value, field_name=field_name)
    if len(normalized) > _MAX_TOKEN_CHARS:
        raise ValueError(f"{field_name} must be <= {_MAX_TOKEN_CHARS} characters.")
    return normalized


def _validate_bool(value: bool, *, field_name: str) -> bool:
    """Require one real boolean value."""

    if isinstance(value, bool):
        return value
    raise TypeError(f"{field_name} must be a bool.")


def _validate_int(value: int, *, field_name: str) -> int:
    """Require one real integer value."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int.")
    return value


def _validate_optional_int(value: int | None, *, field_name: str) -> int | None:
    """Validate one optional integer field."""

    if value is None:
        return None
    return _validate_int(value, field_name=field_name)


def _validate_non_negative_int(value: int, *, field_name: str) -> int:
    """Require one non-negative integer value."""

    number = _validate_int(value, field_name=field_name)
    if number < 0:
        raise ValueError(f"{field_name} must be >= 0.")
    return number


def _validate_aware_datetime(value: datetime, *, field_name: str, target_tz: ZoneInfo | None = None) -> datetime:
    """Require one timezone-aware datetime and normalize it when requested."""

    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")
    if target_tz is None:
        return value
    return value.astimezone(target_tz)


def _coerce_config_bool(value: object, *, default: bool) -> bool:
    """Coerce config-like input to a bounded boolean with fallback."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _BOOL_TRUE:
            return True
        if normalized in _BOOL_FALSE:
            return False
    return default


def _coerce_config_float(
    value: object,
    *,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Coerce config-like input to one finite float within bounds."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not math.isfinite(number):
        number = default
    if minimum is not None and number < minimum:
        number = minimum
    if maximum is not None and number > maximum:
        number = maximum
    return number


def _coerce_config_int(
    value: object,
    *,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Coerce config-like input to one bounded integer."""

    if isinstance(value, bool):
        number = default
    else:
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = default
    if minimum is not None and number < minimum:
        number = minimum
    if maximum is not None and number > maximum:
        number = maximum
    return number


def _seconds_to_ns(seconds: float) -> int:
    """Convert bounded seconds to integer nanoseconds."""

    return max(0, int(seconds * _NS_PER_SECOND))


def _timedelta_to_ns(delta: timedelta) -> int:
    """Convert one timedelta to integer nanoseconds."""

    return (
        ((delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds)
        * 1_000
    )


@dataclass(frozen=True, slots=True)
class _ClockSample:
    """One resolved wall-clock and monotonic time pair."""

    wall_time: datetime
    monotonic_ns: int


@dataclass(frozen=True, slots=True)
class _BlockStatus:
    """Describe one governor block and optional retry timing."""

    reason: str
    retry_after_s: float | None = None


@dataclass(frozen=True, slots=True)
class ProactiveGovernorCandidate:
    """Describe one proactive delivery attempt before reservation.

    Attributes:
        source_kind: Candidate family such as ``social`` or ``longterm``.
        source_id: Stable per-source identifier used for repeat cooldowns.
        summary: Short operator-facing summary kept in governor history.
        channel: Delivery channel currently supported by the governor.
        priority: Higher-level caller priority for downstream orchestration.
        presence_session_id: Optional voice-activation/presence session identifier.
        safety_exempt: Skip normal cooldown checks when True.
        counts_toward_presence_budget: Include the candidate in per-session
            prompt budgets when True.
    """

    source_kind: str
    source_id: str
    summary: str
    channel: str = "speech"
    priority: int = 50
    presence_session_id: int | None = None
    safety_exempt: bool = False
    counts_toward_presence_budget: bool = True

    def __post_init__(self) -> None:
        """Normalize and validate candidate fields after construction."""

        object.__setattr__(
            self,
            "source_kind",
            _normalize_choice(self.source_kind, field_name="source_kind", allowed=_SOURCE_KINDS),
        )
        object.__setattr__(
            self,
            "channel",
            _normalize_choice(self.channel, field_name="channel", allowed=_CHANNELS),
        )
        object.__setattr__(
            self,
            "source_id",
            _normalize_identifier(self.source_id, field_name="source_id"),
        )
        summary = _normalize_bounded_text(
            self.summary,
            field_name="summary",
            max_chars=_MAX_SUMMARY_CHARS,
            required=True,
        )
        object.__setattr__(self, "summary", summary)
        object.__setattr__(
            self,
            "priority",
            _validate_int(self.priority, field_name="priority"),
        )
        object.__setattr__(
            self,
            "presence_session_id",
            _validate_optional_int(self.presence_session_id, field_name="presence_session_id"),
        )
        object.__setattr__(
            self,
            "safety_exempt",
            _validate_bool(self.safety_exempt, field_name="safety_exempt"),
        )
        object.__setattr__(
            self,
            "counts_toward_presence_budget",
            _validate_bool(
                self.counts_toward_presence_budget,
                field_name="counts_toward_presence_budget",
            ),
        )


@dataclass(frozen=True, slots=True)
class ProactiveGovernorReservation:
    """Represent one issued proactive reservation token."""

    candidate: ProactiveGovernorCandidate
    reserved_at: datetime
    reservation_token: str = field(default_factory=lambda: uuid4().hex, repr=False)
    _reserved_monotonic_ns: int = field(default=0, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the reservation payload and its timestamps."""

        if not isinstance(self.candidate, ProactiveGovernorCandidate):
            raise TypeError("candidate must be a ProactiveGovernorCandidate.")
        object.__setattr__(
            self,
            "reserved_at",
            _validate_aware_datetime(self.reserved_at, field_name="reserved_at"),
        )
        object.__setattr__(
            self,
            "reservation_token",
            _normalize_token(self.reservation_token, field_name="reservation_token"),
        )
        object.__setattr__(
            self,
            "_reserved_monotonic_ns",
            _validate_non_negative_int(self._reserved_monotonic_ns, field_name="_reserved_monotonic_ns"),
        )


@dataclass(frozen=True, slots=True)
class ProactiveGovernorHistoryEntry:
    """Persist one finalized proactive outcome for future policy checks."""

    source_kind: str
    source_id: str
    summary: str
    channel: str
    outcome: str
    happened_at: datetime
    priority: int
    presence_session_id: int | None = None
    safety_exempt: bool = False
    counts_toward_presence_budget: bool = True
    reason: str | None = None
    _happened_monotonic_ns: int = field(default=0, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize and validate history fields after construction."""

        object.__setattr__(
            self,
            "source_kind",
            _normalize_choice(self.source_kind, field_name="source_kind", allowed=_SOURCE_KINDS),
        )
        object.__setattr__(
            self,
            "source_id",
            _normalize_identifier(self.source_id, field_name="source_id"),
        )
        summary = _normalize_bounded_text(
            self.summary,
            field_name="summary",
            max_chars=_MAX_SUMMARY_CHARS,
            required=True,
        )
        object.__setattr__(self, "summary", summary)
        object.__setattr__(
            self,
            "channel",
            _normalize_choice(self.channel, field_name="channel", allowed=_CHANNELS),
        )
        object.__setattr__(
            self,
            "outcome",
            _normalize_choice(self.outcome, field_name="outcome", allowed=_OUTCOMES),
        )
        object.__setattr__(
            self,
            "happened_at",
            _validate_aware_datetime(self.happened_at, field_name="happened_at"),
        )
        object.__setattr__(
            self,
            "priority",
            _validate_int(self.priority, field_name="priority"),
        )
        object.__setattr__(
            self,
            "presence_session_id",
            _validate_optional_int(self.presence_session_id, field_name="presence_session_id"),
        )
        object.__setattr__(
            self,
            "safety_exempt",
            _validate_bool(self.safety_exempt, field_name="safety_exempt"),
        )
        object.__setattr__(
            self,
            "counts_toward_presence_budget",
            _validate_bool(
                self.counts_toward_presence_budget,
                field_name="counts_toward_presence_budget",
            ),
        )
        object.__setattr__(
            self,
            "reason",
            _normalize_bounded_text(
                self.reason,
                field_name="reason",
                max_chars=_MAX_REASON_CHARS,
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "_happened_monotonic_ns",
            _validate_non_negative_int(self._happened_monotonic_ns, field_name="_happened_monotonic_ns"),
        )


@dataclass(slots=True)
class ProactiveGovernorDecision:
    """Return the result of one reservation attempt."""

    allowed: bool
    reason: str
    candidate: ProactiveGovernorCandidate
    reservation: ProactiveGovernorReservation | None = None
    # BREAKING: callers may now read retry_after_s / blocked_until for exact backoff handling.
    retry_after_s: float | None = None
    blocked_until: datetime | None = None


@dataclass(slots=True)
class ProactiveGovernor:
    """Enforce proactive speech cooldowns and one in-flight reservation."""

    config: TwinrConfig
    _lock: Lock = field(default_factory=Lock, repr=False)
    _history: deque[ProactiveGovernorHistoryEntry] = field(default_factory=deque, repr=False)
    _active_reservation: ProactiveGovernorReservation | None = field(default=None, repr=False)
    _finalized_reservations: dict[str, ProactiveGovernorHistoryEntry] = field(default_factory=dict, repr=False)
    _latest_delivered_non_exempt_speech: ProactiveGovernorHistoryEntry | None = field(
        default=None,
        repr=False,
    )
    _latest_delivered_source_monotonic_ns: dict[tuple[str, str], int] = field(default_factory=dict, repr=False)
    _speech_delivery_ns: deque[int] = field(default_factory=deque, repr=False)
    _presence_delivery_ns: dict[int, deque[int]] = field(default_factory=dict, repr=False)
    _clock_anchor_wall_time: datetime | None = field(default=None, repr=False)
    _clock_anchor_monotonic_ns: int | None = field(default=None, repr=False)
    _last_effective_monotonic_ns: int | None = field(default=None, repr=False)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "ProactiveGovernor":
        """Build one governor from the canonical Twinr config object."""

        return cls(config=config)

    def try_reserve(
        self,
        candidate: ProactiveGovernorCandidate,
        *,
        now: datetime | None = None,
    ) -> ProactiveGovernorDecision:
        """Attempt to reserve one proactive candidate."""

        if not isinstance(candidate, ProactiveGovernorCandidate):
            raise TypeError("candidate must be a ProactiveGovernorCandidate.")
        with self._lock:
            current = self._resolve_clock_sample_locked(now)
            self._drop_stale_active_reservation_locked(current.monotonic_ns)
            self._purge_runtime_state_locked(current.monotonic_ns)
            active = self._active_reservation
            if active is not None:
                retry_after_s = self._active_reservation_retry_after_s_locked(current.monotonic_ns)
                blocked_until = None
                if retry_after_s is not None:
                    blocked_until = current.wall_time + timedelta(seconds=retry_after_s)
                return ProactiveGovernorDecision(
                    allowed=False,
                    reason="prompt_inflight",
                    candidate=candidate,
                    retry_after_s=retry_after_s,
                    blocked_until=blocked_until,
                )
            if self._governor_enabled() and candidate.channel == "speech":
                block = self._block_status_locked(candidate, current_monotonic_ns=current.monotonic_ns)
                if block is not None:
                    blocked_until = None
                    if block.retry_after_s is not None:
                        blocked_until = current.wall_time + timedelta(seconds=block.retry_after_s)
                    return ProactiveGovernorDecision(
                        allowed=False,
                        reason=block.reason,
                        candidate=candidate,
                        retry_after_s=block.retry_after_s,
                        blocked_until=blocked_until,
                    )
            reservation = ProactiveGovernorReservation(
                candidate=candidate,
                reserved_at=current.wall_time,
                _reserved_monotonic_ns=current.monotonic_ns,
            )
            self._active_reservation = reservation
            return ProactiveGovernorDecision(
                allowed=True,
                reason="allowed",
                candidate=candidate,
                reservation=reservation,
            )

    def cancel(self, reservation: ProactiveGovernorReservation) -> None:
        """Release one active reservation without recording an outcome."""

        if not isinstance(reservation, ProactiveGovernorReservation):
            raise TypeError("reservation must be a ProactiveGovernorReservation.")
        with self._lock:
            if (
                self._active_reservation is not None
                and self._active_reservation.reservation_token == reservation.reservation_token
            ):
                self._active_reservation = None

    def mark_delivered(
        self,
        reservation: ProactiveGovernorReservation,
        *,
        now: datetime | None = None,
    ) -> ProactiveGovernorHistoryEntry:
        """Finalize one reservation as delivered and append history."""

        return self._record_outcome(
            reservation,
            outcome="delivered",
            now=now,
            reason=None,
        )

    def mark_skipped(
        self,
        reservation: ProactiveGovernorReservation,
        *,
        reason: str,
        now: datetime | None = None,
    ) -> ProactiveGovernorHistoryEntry:
        """Finalize one reservation as skipped and append history."""

        normalized_reason = _normalize_bounded_text(
            reason,
            field_name="reason",
            max_chars=_MAX_REASON_CHARS,
            required=False,
        ) or "unknown"
        return self._record_outcome(
            reservation,
            outcome="skipped",
            now=now,
            reason=normalized_reason,
        )

    def history(self) -> tuple[ProactiveGovernorHistoryEntry, ...]:
        """Return the bounded finalized history snapshot."""

        with self._lock:
            self._ensure_history_capacity_locked()
            self._purge_runtime_state_locked(self._current_monotonic_ns_locked())
            return tuple(self._history)

    def _record_outcome(
        self,
        reservation: ProactiveGovernorReservation,
        *,
        outcome: str,
        now: datetime | None,
        reason: str | None,
    ) -> ProactiveGovernorHistoryEntry:
        """Finalize one active reservation and cache the resulting history entry."""

        if not isinstance(reservation, ProactiveGovernorReservation):
            raise TypeError("reservation must be a ProactiveGovernorReservation.")
        with self._lock:
            current = self._resolve_clock_sample_locked(now)
            self._drop_stale_active_reservation_locked(current.monotonic_ns)
            self._purge_runtime_state_locked(current.monotonic_ns)
            existing = self._finalized_reservations.get(reservation.reservation_token)
            if existing is not None:
                return existing
            active = self._active_reservation
            if active is None or active.reservation_token != reservation.reservation_token:
                raise ValueError("reservation is unknown, cancelled, expired, or already finalized.")
            candidate = active.candidate
            entry = ProactiveGovernorHistoryEntry(
                source_kind=candidate.source_kind,
                source_id=candidate.source_id,
                summary=candidate.summary,
                channel=candidate.channel,
                outcome=outcome,
                happened_at=current.wall_time,
                priority=candidate.priority,
                presence_session_id=candidate.presence_session_id,
                safety_exempt=candidate.safety_exempt,
                counts_toward_presence_budget=candidate.counts_toward_presence_budget,
                reason=reason,
                _happened_monotonic_ns=current.monotonic_ns,
            )
            self._active_reservation = None
            self._append_history_locked(entry)
            self._index_history_entry_locked(entry)
            self._finalized_reservations[reservation.reservation_token] = entry
            self._purge_runtime_state_locked(current.monotonic_ns)
            return entry

    def _block_status_locked(
        self,
        candidate: ProactiveGovernorCandidate,
        *,
        current_monotonic_ns: int,
    ) -> _BlockStatus | None:
        """Return the first blocking reason for one speech candidate."""

        if candidate.safety_exempt:
            return None
        global_block = self._global_cooldown_block_locked(current_monotonic_ns=current_monotonic_ns)
        if global_block is not None:
            return global_block
        repeat_block = self._source_repeat_block_locked(candidate, current_monotonic_ns=current_monotonic_ns)
        if repeat_block is not None:
            return repeat_block
        window_block = self._window_budget_block_locked(current_monotonic_ns=current_monotonic_ns)
        if window_block is not None:
            return window_block
        presence_block = self._presence_session_budget_block_locked(
            candidate,
            current_monotonic_ns=current_monotonic_ns,
        )
        if presence_block is not None:
            return presence_block
        return None

    def _global_cooldown_block_locked(self, *, current_monotonic_ns: int) -> _BlockStatus | None:
        """Return a block when the global proactive speech cooldown is active."""

        latest = self._latest_delivered_non_exempt_speech
        if latest is None:
            return None
        cooldown_ns = self._global_cooldown_ns()
        if cooldown_ns <= 0:
            return None
        retry_after_ns = latest._happened_monotonic_ns + cooldown_ns - current_monotonic_ns
        if retry_after_ns <= 0:
            return None
        return _BlockStatus(
            reason="governor_global_prompt_cooldown_active",
            retry_after_s=retry_after_ns / _NS_PER_SECOND,
        )

    def _source_repeat_block_locked(
        self,
        candidate: ProactiveGovernorCandidate,
        *,
        current_monotonic_ns: int,
    ) -> _BlockStatus | None:
        """Return a block when the candidate's source repeat cooldown is active."""

        cooldown_ns = self._source_repeat_cooldown_ns()
        if cooldown_ns <= 0:
            return None
        latest_monotonic_ns = self._latest_delivered_source_monotonic_ns.get(
            (candidate.source_kind, candidate.source_id)
        )
        if latest_monotonic_ns is None:
            return None
        retry_after_ns = latest_monotonic_ns + cooldown_ns - current_monotonic_ns
        if retry_after_ns <= 0:
            return None
        return _BlockStatus(
            reason="governor_source_repeat_cooldown_active",
            retry_after_s=retry_after_ns / _NS_PER_SECOND,
        )

    def _window_budget_block_locked(self, *, current_monotonic_ns: int) -> _BlockStatus | None:
        """Return a block when the rolling speech budget window is exhausted."""

        self._purge_speech_delivery_window_locked(current_monotonic_ns)
        limit = self._window_prompt_limit()
        if len(self._speech_delivery_ns) < limit:
            return None
        retry_after_ns = self._speech_delivery_ns[0] + self._window_ns() - current_monotonic_ns
        if retry_after_ns <= 0:
            return None
        return _BlockStatus(
            reason="governor_window_prompt_budget_exhausted",
            retry_after_s=retry_after_ns / _NS_PER_SECOND,
        )

    def _presence_session_budget_block_locked(
        self,
        candidate: ProactiveGovernorCandidate,
        *,
        current_monotonic_ns: int,
    ) -> _BlockStatus | None:
        """Return a block when the candidate already consumed the session budget."""

        if not candidate.counts_toward_presence_budget:
            return None
        if candidate.presence_session_id is None:
            return None
        session_times = self._presence_delivery_ns.get(candidate.presence_session_id)
        if not session_times:
            return None
        self._purge_presence_window_locked(candidate.presence_session_id, current_monotonic_ns)
        session_times = self._presence_delivery_ns.get(candidate.presence_session_id)
        if not session_times:
            return None
        limit = self._presence_session_prompt_limit()
        if len(session_times) < limit:
            return None
        retry_after_ns = session_times[0] + self._presence_session_window_ns() - current_monotonic_ns
        if retry_after_ns <= 0:
            return None
        return _BlockStatus(
            reason="governor_presence_session_budget_exhausted",
            retry_after_s=retry_after_ns / _NS_PER_SECOND,
        )

    def _drop_stale_active_reservation_locked(self, current_monotonic_ns: int) -> None:
        """Discard one expired in-flight reservation if its TTL elapsed."""

        active = self._active_reservation
        if active is None:
            return
        if current_monotonic_ns >= active._reserved_monotonic_ns + self._active_reservation_ttl_ns():
            self._active_reservation = None

    def _active_reservation_retry_after_s_locked(self, current_monotonic_ns: int) -> float | None:
        """Return remaining seconds until the active reservation TTL elapses."""

        active = self._active_reservation
        if active is None:
            return None
        retry_after_ns = active._reserved_monotonic_ns + self._active_reservation_ttl_ns() - current_monotonic_ns
        return max(0.0, retry_after_ns / _NS_PER_SECOND)

    def _append_history_locked(self, entry: ProactiveGovernorHistoryEntry) -> None:
        """Append one finalized history entry while respecting the configured cap."""

        self._ensure_history_capacity_locked()
        self._history.append(entry)
        history_limit = self._history_limit()
        while len(self._history) > history_limit:
            self._history.popleft()

    def _index_history_entry_locked(self, entry: ProactiveGovernorHistoryEntry) -> None:
        """Update rolling indices for one finalized history entry."""

        if entry.outcome != "delivered" or entry.channel != "speech":
            return
        self._latest_delivered_source_monotonic_ns[(entry.source_kind, entry.source_id)] = (
            entry._happened_monotonic_ns
        )
        if not entry.safety_exempt:
            self._latest_delivered_non_exempt_speech = entry
            self._speech_delivery_ns.append(entry._happened_monotonic_ns)
            if entry.counts_toward_presence_budget and entry.presence_session_id is not None:
                session_times = self._presence_delivery_ns.setdefault(entry.presence_session_id, deque())
                session_times.append(entry._happened_monotonic_ns)

    def _purge_runtime_state_locked(self, current_monotonic_ns: int) -> None:
        """Purge expired rolling-window state and bounded token caches."""

        self._ensure_history_capacity_locked()
        self._purge_speech_delivery_window_locked(current_monotonic_ns)
        self._purge_presence_state_locked(current_monotonic_ns)
        self._purge_source_repeat_index_locked(current_monotonic_ns)
        self._purge_finalized_reservations_locked(current_monotonic_ns)

    def _purge_speech_delivery_window_locked(self, current_monotonic_ns: int) -> None:
        """Drop delivered speech timestamps that left the rolling global window budget."""

        threshold_ns = current_monotonic_ns - self._window_ns()
        while self._speech_delivery_ns and self._speech_delivery_ns[0] < threshold_ns:
            self._speech_delivery_ns.popleft()

    def _purge_presence_window_locked(self, presence_session_id: int, current_monotonic_ns: int) -> None:
        """Drop one session's expired timestamps from the presence budget index."""

        threshold_ns = current_monotonic_ns - self._presence_session_window_ns()
        session_times = self._presence_delivery_ns.get(presence_session_id)
        if session_times is None:
            return
        while session_times and session_times[0] < threshold_ns:
            session_times.popleft()
        if not session_times:
            self._presence_delivery_ns.pop(presence_session_id, None)

    def _purge_presence_state_locked(self, current_monotonic_ns: int) -> None:
        """Drop expired timestamps across all presence-session indices."""

        if not self._presence_delivery_ns:
            return
        for presence_session_id in tuple(self._presence_delivery_ns):
            self._purge_presence_window_locked(presence_session_id, current_monotonic_ns)

    def _purge_source_repeat_index_locked(self, current_monotonic_ns: int) -> None:
        """Drop source-repeat index entries that can no longer affect decisions."""

        cooldown_ns = self._source_repeat_cooldown_ns()
        if cooldown_ns <= 0:
            self._latest_delivered_source_monotonic_ns.clear()
            return
        threshold_ns = current_monotonic_ns - cooldown_ns
        self._latest_delivered_source_monotonic_ns = {
            source_key: delivered_ns
            for source_key, delivered_ns in self._latest_delivered_source_monotonic_ns.items()
            if delivered_ns >= threshold_ns
        }

    def _purge_finalized_reservations_locked(self, current_monotonic_ns: int) -> None:
        """Drop finalized-reservation idempotency entries after the retention buffer."""

        retention_ns = max(
            _seconds_to_ns(_MIN_RETENTION_S),
            int(self._retention_window_ns() * _FINALIZED_CACHE_BUFFER_MULTIPLIER),
            self._active_reservation_ttl_ns(),
        )
        threshold_ns = current_monotonic_ns - retention_ns
        self._finalized_reservations = {
            reservation_token: entry
            for reservation_token, entry in self._finalized_reservations.items()
            if entry._happened_monotonic_ns >= threshold_ns
        }

    def _resolve_clock_sample_locked(self, now: datetime | None) -> _ClockSample:
        """Resolve one wall-clock/monotonic pair for the current call."""

        if now is None:
            wall_time = self._now()
            monotonic_ns = time.monotonic_ns()
        else:
            wall_time = _validate_aware_datetime(
                now,
                field_name="now",
                target_tz=self._timezone(),
            )
            if self._clock_anchor_wall_time is None or self._clock_anchor_monotonic_ns is None:
                self._clock_anchor_wall_time = wall_time
                self._clock_anchor_monotonic_ns = time.monotonic_ns()
            delta_ns = _timedelta_to_ns(wall_time - self._clock_anchor_wall_time)
            monotonic_ns = self._clock_anchor_monotonic_ns + delta_ns
        # BREAKING: explicit backdated `now` inputs are clamped onto a non-decreasing monotonic axis.
        if self._last_effective_monotonic_ns is not None and monotonic_ns < self._last_effective_monotonic_ns:
            monotonic_ns = self._last_effective_monotonic_ns
        self._last_effective_monotonic_ns = monotonic_ns
        return _ClockSample(wall_time=wall_time, monotonic_ns=monotonic_ns)

    def _current_monotonic_ns_locked(self) -> int:
        """Return the current monotonic time while preserving ordering."""

        monotonic_ns = time.monotonic_ns()
        if self._last_effective_monotonic_ns is not None and monotonic_ns < self._last_effective_monotonic_ns:
            monotonic_ns = self._last_effective_monotonic_ns
        self._last_effective_monotonic_ns = monotonic_ns
        return monotonic_ns

    def _timezone(self) -> ZoneInfo:
        """Load the configured local timezone with a safe fallback."""

        timezone_name = getattr(self.config, "local_timezone_name", _DEFAULT_TIMEZONE_NAME)
        normalized_name = _normalize_text(timezone_name) or _DEFAULT_TIMEZONE_NAME
        try:
            return ZoneInfo(normalized_name)
        except ZoneInfoNotFoundError:
            return ZoneInfo(_DEFAULT_TIMEZONE_NAME)

    def _ensure_history_capacity_locked(self) -> None:
        """Enforce the configured bounded history size."""

        history_limit = self._history_limit()
        while len(self._history) > history_limit:
            self._history.popleft()

    def _history_limit(self) -> int:
        """Return the bounded finalized history size."""

        return self._config_int(
            "proactive_governor_history_limit",
            default=_DEFAULT_HISTORY_LIMIT,
            minimum=_MIN_HISTORY_LIMIT,
            maximum=_MAX_HISTORY_LIMIT,
        )

    def _window_prompt_limit(self) -> int:
        """Return the bounded rolling prompt budget."""

        return self._config_int(
            "proactive_governor_window_prompt_limit",
            default=_DEFAULT_WINDOW_PROMPT_LIMIT,
            minimum=1,
            maximum=_MAX_PROMPT_LIMIT,
        )

    def _presence_session_prompt_limit(self) -> int:
        """Return the bounded per-session prompt budget."""

        return self._config_int(
            "proactive_governor_presence_session_prompt_limit",
            default=_DEFAULT_PRESENCE_SESSION_PROMPT_LIMIT,
            minimum=1,
            maximum=_MAX_PROMPT_LIMIT,
        )

    def _global_cooldown_ns(self) -> int:
        """Return the bounded global speech cooldown in nanoseconds."""

        return _seconds_to_ns(
            self._config_float(
                "proactive_governor_global_prompt_cooldown_s",
                default=_DEFAULT_GLOBAL_PROMPT_COOLDOWN_S,
                minimum=0.0,
                maximum=_MAX_COOLDOWN_S,
            )
        )

    def _source_repeat_cooldown_ns(self) -> int:
        """Return the bounded per-source speech cooldown in nanoseconds."""

        return _seconds_to_ns(
            self._config_float(
                "proactive_governor_source_repeat_cooldown_s",
                default=_DEFAULT_SOURCE_REPEAT_COOLDOWN_S,
                minimum=0.0,
                maximum=_MAX_COOLDOWN_S,
            )
        )

    def _window_ns(self) -> int:
        """Return the bounded rolling prompt-window size in nanoseconds."""

        return _seconds_to_ns(
            self._config_float(
                "proactive_governor_window_s",
                default=_DEFAULT_WINDOW_S,
                minimum=1.0,
                maximum=_MAX_WINDOW_S,
            )
        )

    def _presence_session_window_ns(self) -> int:
        """Return the bounded per-session presence window in nanoseconds."""

        explicit_window_s = self._config_float(
            "proactive_governor_presence_session_window_s",
            default=0.0,
            minimum=0.0,
            maximum=_MAX_WINDOW_S,
        )
        if explicit_window_s > 0.0:
            return _seconds_to_ns(explicit_window_s)

        explicit_grace_s = self._config_float(
            "proactive_governor_presence_grace_s",
            default=0.0,
            minimum=0.0,
            maximum=_MAX_WINDOW_S,
        )
        if explicit_grace_s > 0.0:
            return _seconds_to_ns(explicit_grace_s)

        fallback_window_s = max(
            _DEFAULT_PRESENCE_GRACE_S,
            self._config_float(
                "proactive_governor_window_s",
                default=_DEFAULT_WINDOW_S,
                minimum=1.0,
                maximum=_MAX_WINDOW_S,
            ),
            self._config_float(
                "voice_orchestrator_follow_up_timeout_s",
                default=_DEFAULT_PRESENCE_GRACE_S,
                minimum=1.0,
                maximum=_MAX_WINDOW_S,
            ),
        )
        return _seconds_to_ns(fallback_window_s)

    def _active_reservation_ttl_ns(self) -> int:
        """Return the bounded in-flight reservation TTL in nanoseconds."""

        return _seconds_to_ns(
            self._config_float(
                "proactive_governor_active_reservation_ttl_s",
                default=_DEFAULT_ACTIVE_RESERVATION_TTL_S,
                minimum=1.0,
                maximum=_MAX_ACTIVE_RESERVATION_TTL_S,
            )
        )

    def _retention_window_ns(self) -> int:
        """Return the longest policy window that can still affect decisions."""

        return max(
            self._window_ns(),
            self._global_cooldown_ns(),
            self._source_repeat_cooldown_ns(),
            self._presence_session_window_ns(),
            _seconds_to_ns(_MIN_RETENTION_S),
        )

    def _config_bool(self, name: str, *, default: bool) -> bool:
        """Read one boolean governor config field safely."""

        return _coerce_config_bool(getattr(self.config, name, default), default=default)

    def _config_float(
        self,
        name: str,
        *,
        default: float,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        """Read one float governor config field safely."""

        return _coerce_config_float(
            getattr(self.config, name, default),
            default=default,
            minimum=minimum,
            maximum=maximum,
        )

    def _config_int(
        self,
        name: str,
        *,
        default: int,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int:
        """Read one integer governor config field safely."""

        return _coerce_config_int(
            getattr(self.config, name, default),
            default=default,
            minimum=minimum,
            maximum=maximum,
        )

    def _governor_enabled(self) -> bool:
        """Return whether proactive speech governance is enabled."""

        return self._config_bool("proactive_governor_enabled", default=True)

    def _now(self) -> datetime:
        """Return the current time in the governor timezone."""

        return datetime.now(self._timezone())


__all__ = [
    "ProactiveGovernor",
    "ProactiveGovernorCandidate",
    "ProactiveGovernorDecision",
    "ProactiveGovernorHistoryEntry",
    "ProactiveGovernorReservation",
]
