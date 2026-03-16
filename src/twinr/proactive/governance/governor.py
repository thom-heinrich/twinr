"""Gate proactive delivery reservations with bounded cooldown policy.

This module owns the in-memory governor used by Twinr runtime workflows to
rate-limit proactive speech prompts, track one active reservation, and retain
bounded delivery history for later policy checks. Import public types from
``twinr.proactive.governance`` or ``twinr.proactive``.
"""

from __future__ import annotations

import math  # AUDIT-FIX(#4): finite numeric validation for env-derived config values.
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from uuid import uuid4  # AUDIT-FIX(#2): unique reservation tokens prevent stale/synthetic reuse.
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError  # AUDIT-FIX(#3): invalid timezone config now degrades safely.

from twinr.agent.base_agent.config import TwinrConfig


_SOURCE_KINDS = frozenset({"social", "reminder", "automation", "longterm"})
_CHANNELS = frozenset({"speech", "print"})
_OUTCOMES = frozenset({"delivered", "skipped"})
_BOOL_TRUE = frozenset({"1", "true", "t", "yes", "y", "on"})
_BOOL_FALSE = frozenset({"0", "false", "f", "no", "n", "off"})
_DEFAULT_TIMEZONE_NAME = "UTC"  # AUDIT-FIX(#3): safe fallback keeps governor running on bad timezone config.
_DEFAULT_GLOBAL_PROMPT_COOLDOWN_S = 0.0
_DEFAULT_SOURCE_REPEAT_COOLDOWN_S = 0.0
_DEFAULT_WINDOW_PROMPT_LIMIT = 1
_DEFAULT_WINDOW_S = 60.0
_DEFAULT_PRESENCE_SESSION_PROMPT_LIMIT = 1
_DEFAULT_PRESENCE_GRACE_S = 60.0
_DEFAULT_ACTIVE_RESERVATION_TTL_S = 30.0
_DEFAULT_HISTORY_LIMIT = 128
_MIN_HISTORY_LIMIT = 16
_HISTORY_BUFFER_MULTIPLIER = 2.0


def _normalize_text(value: str | None) -> str:
    """Normalize optional text into a single trimmed line."""

    if value is None:
        return ""
    if not isinstance(value, str):  # AUDIT-FIX(#1): reject silent non-string coercion in identifiers and reasons.
        raise TypeError("text value must be a str or None.")
    return " ".join(value.split()).strip()


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


def _validate_aware_datetime(value: datetime, *, field_name: str, target_tz: ZoneInfo | None = None) -> datetime:
    """Require one timezone-aware datetime and normalize it when requested."""

    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:  # AUDIT-FIX(#3): fail fast on naive datetimes before comparisons explode.
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
    if not math.isfinite(number):  # AUDIT-FIX(#4): reject NaN/inf so timedelta math cannot crash.
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


@dataclass(frozen=True, slots=True)
class ProactiveGovernorCandidate:
    """Describe one proactive delivery attempt before reservation.

    Attributes:
        source_kind: Candidate family such as ``social`` or ``longterm``.
        source_id: Stable per-source identifier used for repeat cooldowns.
        summary: Short operator-facing summary kept in governor history.
        channel: Delivery channel currently supported by the governor.
        priority: Higher-level caller priority for downstream orchestration.
        presence_session_id: Optional wakeword/presence session identifier.
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
            _normalize_choice(self.source_kind, field_name="source_kind", allowed=_SOURCE_KINDS),  # AUDIT-FIX(#1): canonicalise enums before policy checks.
        )
        object.__setattr__(
            self,
            "channel",
            _normalize_choice(self.channel, field_name="channel", allowed=_CHANNELS),  # AUDIT-FIX(#1): canonicalise channel before gating logic.
        )
        object.__setattr__(
            self,
            "source_id",
            _normalize_required_text(self.source_id, field_name="source_id"),  # AUDIT-FIX(#6): canonical source IDs keep repeat cooldowns effective.
        )
        object.__setattr__(
            self,
            "summary",
            _normalize_required_text(self.summary, field_name="summary"),  # AUDIT-FIX(#6): store normalized summaries for stable history records.
        )
        object.__setattr__(
            self,
            "priority",
            _validate_int(self.priority, field_name="priority"),  # AUDIT-FIX(#1): reject late int() coercion failures and bool-as-int surprises.
        )
        object.__setattr__(
            self,
            "presence_session_id",
            _validate_optional_int(self.presence_session_id, field_name="presence_session_id"),  # AUDIT-FIX(#1): keep session accounting type-safe.
        )
        object.__setattr__(
            self,
            "safety_exempt",
            _validate_bool(self.safety_exempt, field_name="safety_exempt"),  # AUDIT-FIX(#1): truthy strings like "false" no longer bypass guardrails.
        )
        object.__setattr__(
            self,
            "counts_toward_presence_budget",
            _validate_bool(
                self.counts_toward_presence_budget,
                field_name="counts_toward_presence_budget",
            ),  # AUDIT-FIX(#1): prevent truthy non-bools from corrupting presence budgeting.
        )


@dataclass(frozen=True, slots=True)
class ProactiveGovernorReservation:
    """Represent one issued proactive reservation token.

    The reservation token is the authoritative handle for cancellation and
    finalization. Structural equality of the candidate alone is not enough to
    finalize a reservation.
    """

    candidate: ProactiveGovernorCandidate
    reserved_at: datetime
    reservation_token: str = field(default_factory=lambda: uuid4().hex, repr=False)  # AUDIT-FIX(#2): opaque token hardens reservation ownership.

    def __post_init__(self) -> None:
        """Validate the reservation payload and its timestamp."""

        if not isinstance(self.candidate, ProactiveGovernorCandidate):
            raise TypeError("candidate must be a ProactiveGovernorCandidate.")
        object.__setattr__(
            self,
            "reserved_at",
            _validate_aware_datetime(self.reserved_at, field_name="reserved_at"),  # AUDIT-FIX(#3): reservation timestamps must be safe for comparisons.
        )
        object.__setattr__(
            self,
            "reservation_token",
            _normalize_required_text(self.reservation_token, field_name="reservation_token"),  # AUDIT-FIX(#2): empty tokens are not allowed.
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

    def __post_init__(self) -> None:
        """Normalize and validate history fields after construction."""

        object.__setattr__(
            self,
            "source_kind",
            _normalize_choice(self.source_kind, field_name="source_kind", allowed=_SOURCE_KINDS),  # AUDIT-FIX(#1): history must stay policy-compatible.
        )
        object.__setattr__(
            self,
            "source_id",
            _normalize_required_text(self.source_id, field_name="source_id"),  # AUDIT-FIX(#6): keep historical source matching canonical.
        )
        object.__setattr__(
            self,
            "summary",
            _normalize_required_text(self.summary, field_name="summary"),  # AUDIT-FIX(#6): normalize stored summaries for stable diagnostics.
        )
        object.__setattr__(
            self,
            "channel",
            _normalize_choice(self.channel, field_name="channel", allowed=_CHANNELS),  # AUDIT-FIX(#1): reject corrupt history entries early.
        )
        object.__setattr__(
            self,
            "outcome",
            _normalize_choice(self.outcome, field_name="outcome", allowed=_OUTCOMES),  # AUDIT-FIX(#1): keep outcomes type-safe and bounded.
        )
        object.__setattr__(
            self,
            "happened_at",
            _validate_aware_datetime(self.happened_at, field_name="happened_at"),  # AUDIT-FIX(#3): history timestamps must be timezone-aware.
        )
        object.__setattr__(
            self,
            "priority",
            _validate_int(self.priority, field_name="priority"),  # AUDIT-FIX(#1): prevent corrupt priority values from entering history.
        )
        object.__setattr__(
            self,
            "presence_session_id",
            _validate_optional_int(self.presence_session_id, field_name="presence_session_id"),  # AUDIT-FIX(#1): preserve typed session accounting.
        )
        object.__setattr__(
            self,
            "safety_exempt",
            _validate_bool(self.safety_exempt, field_name="safety_exempt"),  # AUDIT-FIX(#1): exemptions must remain explicit booleans.
        )
        object.__setattr__(
            self,
            "counts_toward_presence_budget",
            _validate_bool(
                self.counts_toward_presence_budget,
                field_name="counts_toward_presence_budget",
            ),  # AUDIT-FIX(#1): keep presence-budget flags deterministic.
        )
        normalized_reason = _normalize_text(self.reason) or None
        object.__setattr__(
            self,
            "reason",
            normalized_reason,  # AUDIT-FIX(#6): normalize skip reasons for consistent audit output.
        )


@dataclass(slots=True)
class ProactiveGovernorDecision:
    """Return the result of one reservation attempt."""

    allowed: bool
    reason: str
    candidate: ProactiveGovernorCandidate
    reservation: ProactiveGovernorReservation | None = None


@dataclass(slots=True)
class ProactiveGovernor:
    """Enforce proactive speech cooldowns and one in-flight reservation.

    The governor is a thin in-memory policy service shared by runtime
    workflows. It normalizes config access, ensures at most one active
    reservation exists, and records bounded delivery history so repeat,
    window, and presence-session budgets stay deterministic.
    """

    config: TwinrConfig
    _lock: Lock = field(default_factory=Lock, repr=False)
    _history: list[ProactiveGovernorHistoryEntry] = field(default_factory=list, repr=False)
    _active_reservation: ProactiveGovernorReservation | None = field(default=None, repr=False)
    _finalized_reservations: dict[str, ProactiveGovernorHistoryEntry] = field(
        default_factory=dict,
        repr=False,
    )  # AUDIT-FIX(#2): idempotent completion tracking prevents duplicate history entries.

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
        """Attempt to reserve one proactive candidate.

        Args:
            candidate: Proposed proactive delivery to gate.
            now: Optional explicit current time. When provided it must already
                be timezone-aware.

        Returns:
            One decision describing whether the caller may proceed. Successful
            decisions carry a reservation token that must later be cancelled or
            finalized.
        """

        current_time = self._resolve_current_time(now)
        with self._lock:
            self._drop_stale_active_reservation(current_time)
            self._trim_history(current_time)
            if self._active_reservation is not None:
                return ProactiveGovernorDecision(
                    allowed=False,
                    reason="prompt_inflight",
                    candidate=candidate,
                )
            if self._governor_enabled() and candidate.channel == "speech":  # AUDIT-FIX(#4): malformed bool config no longer flips behavior unpredictably.
                block_reason = self._block_reason(candidate, current_time=current_time)
                if block_reason is not None:
                    return ProactiveGovernorDecision(
                        allowed=False,
                        reason=block_reason,
                        candidate=candidate,
                    )
            reservation = ProactiveGovernorReservation(candidate=candidate, reserved_at=current_time)
            self._active_reservation = reservation
            return ProactiveGovernorDecision(
                allowed=True,
                reason="allowed",
                candidate=candidate,
                reservation=reservation,
            )

    def cancel(self, reservation: ProactiveGovernorReservation) -> None:
        """Release one active reservation without recording an outcome."""

        with self._lock:
            if (
                self._active_reservation is not None
                and self._active_reservation.reservation_token == reservation.reservation_token
            ):  # AUDIT-FIX(#2): cancel now targets the issued reservation token, not structural equality.
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

        return self._record_outcome(
            reservation,
            outcome="skipped",
            now=now,
            reason=_normalize_text(reason) or "unknown",
        )

    def history(self) -> tuple[ProactiveGovernorHistoryEntry, ...]:
        """Return the bounded finalized history snapshot."""

        with self._lock:
            self._trim_history(self._now())  # AUDIT-FIX(#5): history snapshots no longer retain expired bookkeeping indefinitely.
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

        current_time = self._resolve_current_time(now)
        with self._lock:
            self._drop_stale_active_reservation(current_time)
            existing = self._finalized_reservations.get(reservation.reservation_token)
            if existing is not None:
                return existing  # AUDIT-FIX(#2): repeated completion calls are idempotent.
            if (
                self._active_reservation is None
                or self._active_reservation.reservation_token != reservation.reservation_token
            ):
                raise ValueError("reservation is unknown, cancelled, expired, or already finalized.")  # AUDIT-FIX(#2): reject stale/synthetic completions instead of corrupting history.
            candidate = self._active_reservation.candidate
            entry = ProactiveGovernorHistoryEntry(
                source_kind=candidate.source_kind,
                source_id=candidate.source_id,
                summary=candidate.summary,
                channel=candidate.channel,
                outcome=outcome,
                happened_at=current_time,
                priority=candidate.priority,
                presence_session_id=candidate.presence_session_id,
                safety_exempt=candidate.safety_exempt,
                counts_toward_presence_budget=candidate.counts_toward_presence_budget,
                reason=reason,
            )
            self._active_reservation = None
            self._history.append(entry)
            self._finalized_reservations[reservation.reservation_token] = entry
            self._trim_history(current_time)
            return entry

    def _block_reason(
        self,
        candidate: ProactiveGovernorCandidate,
        *,
        current_time: datetime,
    ) -> str | None:
        """Return the first blocking reason for one speech candidate."""

        if candidate.safety_exempt:
            return None
        if self._global_cooldown_active(current_time=current_time):
            return "governor_global_prompt_cooldown_active"
        if self._source_repeat_cooldown_active(candidate, current_time=current_time):
            return "governor_source_repeat_cooldown_active"
        if self._window_budget_exhausted(current_time=current_time):
            return "governor_window_prompt_budget_exhausted"
        if self._presence_session_budget_exhausted(candidate, current_time=current_time):
            return "governor_presence_session_budget_exhausted"
        return None

    def _global_cooldown_active(self, *, current_time: datetime) -> bool:
        """Return whether the global proactive speech cooldown is active."""

        latest = self._latest_delivered_speech_entry()
        if latest is None:
            return False
        cooldown = self._config_float(
            "proactive_governor_global_prompt_cooldown_s",
            default=_DEFAULT_GLOBAL_PROMPT_COOLDOWN_S,
            minimum=0.0,
        )
        return current_time < latest.happened_at + timedelta(seconds=cooldown)

    def _source_repeat_cooldown_active(
        self,
        candidate: ProactiveGovernorCandidate,
        *,
        current_time: datetime,
    ) -> bool:
        """Return whether the candidate's source-level repeat cooldown is active."""

        cooldown = self._config_float(
            "proactive_governor_source_repeat_cooldown_s",
            default=_DEFAULT_SOURCE_REPEAT_COOLDOWN_S,
            minimum=0.0,
        )
        if cooldown <= 0.0:
            return False
        latest = max(
            (
                entry
                for entry in self._history
                if entry.outcome == "delivered"
                and entry.channel == "speech"
                and entry.source_kind == candidate.source_kind
                and entry.source_id == candidate.source_id
            ),
            key=lambda entry: entry.happened_at,
            default=None,
        )  # AUDIT-FIX(#7): use the newest timestamp, not append order, for repeat cooldowns.
        if latest is None:
            return False
        return current_time < latest.happened_at + timedelta(seconds=cooldown)

    def _window_budget_exhausted(self, *, current_time: datetime) -> bool:
        """Return whether the rolling speech budget window is exhausted."""

        limit = self._config_int(
            "proactive_governor_window_prompt_limit",
            default=_DEFAULT_WINDOW_PROMPT_LIMIT,
            minimum=1,
        )
        window_s = self._config_float(
            "proactive_governor_window_s",
            default=_DEFAULT_WINDOW_S,
            minimum=1.0,
        )
        threshold = current_time - timedelta(seconds=window_s)
        count = sum(
            1
            for entry in self._history
            if entry.outcome == "delivered"
            and entry.channel == "speech"
            and not entry.safety_exempt
            and entry.happened_at >= threshold
        )
        return count >= limit

    def _presence_session_budget_exhausted(
        self,
        candidate: ProactiveGovernorCandidate,
        *,
        current_time: datetime,
    ) -> bool:
        """Return whether the candidate already consumed the current session budget."""

        if not candidate.counts_toward_presence_budget:
            return False
        if candidate.presence_session_id is None:
            return False
        limit = self._config_int(
            "proactive_governor_presence_session_prompt_limit",
            default=_DEFAULT_PRESENCE_SESSION_PROMPT_LIMIT,
            minimum=1,
        )
        window_s = self._config_float(
            "wakeword_presence_grace_s",
            default=_DEFAULT_PRESENCE_GRACE_S,
            minimum=1.0,
        )
        threshold = current_time - timedelta(seconds=window_s)
        count = sum(
            1
            for entry in self._history
            if entry.outcome == "delivered"
            and entry.channel == "speech"
            and not entry.safety_exempt
            and entry.counts_toward_presence_budget
            and entry.presence_session_id == candidate.presence_session_id
            and entry.happened_at >= threshold
        )
        return count >= limit

    def _latest_delivered_speech_entry(self) -> ProactiveGovernorHistoryEntry | None:
        """Return the newest non-exempt delivered speech entry."""

        return max(
            (
                entry
                for entry in self._history
                if entry.outcome == "delivered" and entry.channel == "speech" and not entry.safety_exempt
            ),
            key=lambda entry: entry.happened_at,
            default=None,
        )  # AUDIT-FIX(#7): cooldowns now respect actual timestamps even after clock skew or test-injected times.

    def _drop_stale_active_reservation(self, current_time: datetime) -> None:
        """Discard one expired in-flight reservation if its TTL elapsed."""

        if self._active_reservation is None:
            return
        ttl_s = self._config_float(
            "proactive_governor_active_reservation_ttl_s",
            default=_DEFAULT_ACTIVE_RESERVATION_TTL_S,
            minimum=1.0,
        )
        if current_time >= self._active_reservation.reserved_at + timedelta(seconds=ttl_s):
            self._active_reservation = None  # AUDIT-FIX(#2): expired reservations are invalidated before they can be finalized later.

    def _trim_history(self, current_time: datetime) -> None:
        """Trim history and finalized-token caches to the active policy window."""

        enforcement_window_s = max(
            self._config_float(
                "proactive_governor_window_s",
                default=_DEFAULT_WINDOW_S,
                minimum=1.0,
            ),
            self._config_float(
                "proactive_governor_source_repeat_cooldown_s",
                default=_DEFAULT_SOURCE_REPEAT_COOLDOWN_S,
                minimum=0.0,
            ),
            self._config_float(
                "proactive_governor_global_prompt_cooldown_s",
                default=_DEFAULT_GLOBAL_PROMPT_COOLDOWN_S,
                minimum=0.0,
            ),
            self._config_float(
                "wakeword_presence_grace_s",
                default=_DEFAULT_PRESENCE_GRACE_S,
                minimum=1.0,
            ),
            60.0,
        )
        relevant_threshold = current_time - timedelta(seconds=enforcement_window_s)
        buffer_threshold = current_time - timedelta(seconds=enforcement_window_s * _HISTORY_BUFFER_MULTIPLIER)
        relevant_entries = [
            entry for entry in self._history if entry.happened_at >= relevant_threshold
        ]  # AUDIT-FIX(#5): never evict entries still needed for active rate-limit decisions.
        buffered_older_entries = [
            entry
            for entry in self._history
            if buffer_threshold <= entry.happened_at < relevant_threshold
        ]
        history_limit = self._config_int(
            "proactive_governor_history_limit",
            default=_DEFAULT_HISTORY_LIMIT,
            minimum=_MIN_HISTORY_LIMIT,
        )
        if len(relevant_entries) >= history_limit:
            self._history = relevant_entries
        else:
            keep_older = max(0, history_limit - len(relevant_entries))
            self._history = buffered_older_entries[-keep_older:] + relevant_entries
        self._finalized_reservations = {
            reservation_token: entry
            for reservation_token, entry in self._finalized_reservations.items()
            if entry.happened_at >= buffer_threshold
        }  # AUDIT-FIX(#2): finalized-token cache stays bounded to the same retention window.

    def _resolve_current_time(self, now: datetime | None) -> datetime:
        """Resolve the effective current time in the governor timezone."""

        if now is None:
            return self._now()
        return _validate_aware_datetime(
            now,
            field_name="now",
            target_tz=self._timezone(),
        )  # AUDIT-FIX(#3): explicit times are normalized to one timezone before policy math.

    def _timezone(self) -> ZoneInfo:
        """Load the configured local timezone with a safe fallback."""

        timezone_name = getattr(self.config, "local_timezone_name", _DEFAULT_TIMEZONE_NAME)
        normalized_name = _normalize_text(timezone_name) or _DEFAULT_TIMEZONE_NAME
        try:
            return ZoneInfo(normalized_name)
        except ZoneInfoNotFoundError:
            return ZoneInfo(_DEFAULT_TIMEZONE_NAME)  # AUDIT-FIX(#3): bad timezone config no longer crashes every call site.

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

        return datetime.now(self._timezone())  # AUDIT-FIX(#3): current time always uses a validated timezone.


__all__ = [
    "ProactiveGovernor",
    "ProactiveGovernorCandidate",
    "ProactiveGovernorDecision",
    "ProactiveGovernorHistoryEntry",
    "ProactiveGovernorReservation",
]
