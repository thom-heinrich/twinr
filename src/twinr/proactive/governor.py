from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from zoneinfo import ZoneInfo

from twinr.agent.base_agent.config import TwinrConfig


_SOURCE_KINDS = frozenset({"social", "reminder", "automation", "longterm"})
_CHANNELS = frozenset({"speech", "print"})
_OUTCOMES = frozenset({"delivered", "skipped"})


def _normalize_text(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


@dataclass(frozen=True, slots=True)
class ProactiveGovernorCandidate:
    source_kind: str
    source_id: str
    summary: str
    channel: str = "speech"
    priority: int = 50
    presence_session_id: int | None = None
    safety_exempt: bool = False
    counts_toward_presence_budget: bool = True

    def __post_init__(self) -> None:
        if self.source_kind not in _SOURCE_KINDS:
            raise ValueError(f"source_kind must be one of: {', '.join(sorted(_SOURCE_KINDS))}.")
        if self.channel not in _CHANNELS:
            raise ValueError(f"channel must be one of: {', '.join(sorted(_CHANNELS))}.")
        if not _normalize_text(self.source_id):
            raise ValueError("source_id is required.")
        if not _normalize_text(self.summary):
            raise ValueError("summary is required.")


@dataclass(frozen=True, slots=True)
class ProactiveGovernorReservation:
    candidate: ProactiveGovernorCandidate
    reserved_at: datetime


@dataclass(frozen=True, slots=True)
class ProactiveGovernorHistoryEntry:
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
        if self.outcome not in _OUTCOMES:
            raise ValueError(f"outcome must be one of: {', '.join(sorted(_OUTCOMES))}.")


@dataclass(slots=True)
class ProactiveGovernorDecision:
    allowed: bool
    reason: str
    candidate: ProactiveGovernorCandidate
    reservation: ProactiveGovernorReservation | None = None


@dataclass(slots=True)
class ProactiveGovernor:
    config: TwinrConfig
    _lock: Lock = field(default_factory=Lock, repr=False)
    _history: list[ProactiveGovernorHistoryEntry] = field(default_factory=list, repr=False)
    _active_reservation: ProactiveGovernorReservation | None = field(default=None, repr=False)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "ProactiveGovernor":
        return cls(config=config)

    def try_reserve(
        self,
        candidate: ProactiveGovernorCandidate,
        *,
        now: datetime | None = None,
    ) -> ProactiveGovernorDecision:
        current_time = now or self._now()
        with self._lock:
            self._drop_stale_active_reservation(current_time)
            self._trim_history(current_time)
            if self._active_reservation is not None:
                return ProactiveGovernorDecision(
                    allowed=False,
                    reason="prompt_inflight",
                    candidate=candidate,
                )
            if self.config.proactive_governor_enabled and candidate.channel == "speech":
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
        with self._lock:
            if self._active_reservation == reservation:
                self._active_reservation = None

    def mark_delivered(
        self,
        reservation: ProactiveGovernorReservation,
        *,
        now: datetime | None = None,
    ) -> ProactiveGovernorHistoryEntry:
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
        return self._record_outcome(
            reservation,
            outcome="skipped",
            now=now,
            reason=_normalize_text(reason) or "unknown",
        )

    def history(self) -> tuple[ProactiveGovernorHistoryEntry, ...]:
        with self._lock:
            return tuple(self._history)

    def _record_outcome(
        self,
        reservation: ProactiveGovernorReservation,
        *,
        outcome: str,
        now: datetime | None,
        reason: str | None,
    ) -> ProactiveGovernorHistoryEntry:
        current_time = now or self._now()
        candidate = reservation.candidate
        entry = ProactiveGovernorHistoryEntry(
            source_kind=candidate.source_kind,
            source_id=candidate.source_id,
            summary=candidate.summary,
            channel=candidate.channel,
            outcome=outcome,
            happened_at=current_time,
            priority=int(candidate.priority),
            presence_session_id=candidate.presence_session_id,
            safety_exempt=bool(candidate.safety_exempt),
            counts_toward_presence_budget=bool(candidate.counts_toward_presence_budget),
            reason=reason,
        )
        with self._lock:
            self._drop_stale_active_reservation(current_time)
            if self._active_reservation == reservation:
                self._active_reservation = None
            self._history.append(entry)
            self._trim_history(current_time)
        return entry

    def _block_reason(
        self,
        candidate: ProactiveGovernorCandidate,
        *,
        current_time: datetime,
    ) -> str | None:
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
        latest = self._latest_delivered_speech_entry()
        if latest is None:
            return False
        cooldown = max(0.0, float(self.config.proactive_governor_global_prompt_cooldown_s))
        return current_time < latest.happened_at + timedelta(seconds=cooldown)

    def _source_repeat_cooldown_active(
        self,
        candidate: ProactiveGovernorCandidate,
        *,
        current_time: datetime,
    ) -> bool:
        cooldown = max(0.0, float(self.config.proactive_governor_source_repeat_cooldown_s))
        if cooldown <= 0.0:
            return False
        for entry in reversed(self._history):
            if entry.outcome != "delivered" or entry.channel != "speech":
                continue
            if entry.source_kind != candidate.source_kind or entry.source_id != candidate.source_id:
                continue
            return current_time < entry.happened_at + timedelta(seconds=cooldown)
        return False

    def _window_budget_exhausted(self, *, current_time: datetime) -> bool:
        limit = max(1, int(self.config.proactive_governor_window_prompt_limit))
        window_s = max(1.0, float(self.config.proactive_governor_window_s))
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
        if not candidate.counts_toward_presence_budget:
            return False
        if candidate.presence_session_id is None:
            return False
        limit = max(1, int(self.config.proactive_governor_presence_session_prompt_limit))
        window_s = max(1.0, float(self.config.wakeword_presence_grace_s))
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
        for entry in reversed(self._history):
            if entry.outcome == "delivered" and entry.channel == "speech" and not entry.safety_exempt:
                return entry
        return None

    def _drop_stale_active_reservation(self, current_time: datetime) -> None:
        if self._active_reservation is None:
            return
        ttl_s = max(1.0, float(self.config.proactive_governor_active_reservation_ttl_s))
        if current_time >= self._active_reservation.reserved_at + timedelta(seconds=ttl_s):
            self._active_reservation = None

    def _trim_history(self, current_time: datetime) -> None:
        max_window_s = max(
            float(self.config.proactive_governor_window_s),
            float(self.config.proactive_governor_source_repeat_cooldown_s),
            float(self.config.proactive_governor_global_prompt_cooldown_s),
            float(self.config.wakeword_presence_grace_s),
            60.0,
        )
        threshold = current_time - timedelta(seconds=max_window_s * 2.0)
        self._history = [entry for entry in self._history if entry.happened_at >= threshold]
        history_limit = max(16, int(self.config.proactive_governor_history_limit))
        if len(self._history) > history_limit:
            self._history = self._history[-history_limit:]

    def _now(self) -> datetime:
        return datetime.now(ZoneInfo(self.config.local_timezone_name))


__all__ = [
    "ProactiveGovernor",
    "ProactiveGovernorCandidate",
    "ProactiveGovernorDecision",
    "ProactiveGovernorHistoryEntry",
    "ProactiveGovernorReservation",
]
