"""Plan one full local day of calm HDMI reserve impulses.

The live proactive monitor should not improvise reserve-card timing on every
tick. This module turns the current personality-driven reserve candidates into
one persistent per-day sequence that can be replayed calmly throughout the
day. That keeps scheduling, candidate weighting, and cursor persistence out of
the hot runtime loop while preserving a visible, personality-shaped reserve
surface beside the face.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date as LocalDate, datetime, time as LocalTime, timedelta, timezone
import hashlib
import json
import logging
import math
from pathlib import Path

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.display.reserve_bus_feedback import (
    DisplayReserveBusFeedbackSignal,
    DisplayReserveBusFeedbackStore,
)

from .display_reserve_candidates import load_display_reserve_candidates

_DEFAULT_PLAN_PATH = "artifacts/stores/ops/display_reserve_bus_plan.json"
_DEFAULT_REFRESH_AFTER_LOCAL = "05:30"
_DEFAULT_CANDIDATE_LIMIT = 8
_DEFAULT_ITEMS_PER_DAY = 30
_DEFAULT_TOPIC_GAP = 2
_DEFAULT_SOURCE_GAP = 1
_DEFAULT_FAMILY_GAP = 1
_EMPTY_PLAN_RETRY_S = 60.0
_DEFAULT_MIN_HOLD_S = 4.0 * 60.0
_DEFAULT_BASE_HOLD_S = 8.0 * 60.0
_DEFAULT_MAX_HOLD_S = 12.0 * 60.0

_ACTION_BONUS_S = {
    "silent": -2.0 * 60.0,
    "hint": 0.0,
    "brief_update": 1.0 * 60.0,
    "ask_one": 2.0 * 60.0,
    "invite_follow_up": 3.0 * 60.0,
}
_ATTENTION_BONUS_S = {
    "background": 0.0,
    "growing": 1.0 * 60.0,
    "forming": 2.0 * 60.0,
    "shared_thread": 3.0 * 60.0,
}

_LOGGER = logging.getLogger(__name__)


def _default_local_now() -> datetime:
    """Return the current local wall clock as an aware datetime."""

    return datetime.now().astimezone()


def _utc_now() -> datetime:
    """Return the current UTC wall clock."""

    return datetime.now(timezone.utc)


def _compact_text(value: object | None, *, max_len: int) -> str:
    """Return one bounded single-line text field."""

    if value is None:
        return ""
    compact = " ".join(str(value).split()).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _parse_timestamp(value: object | None) -> datetime | None:
    """Parse one optional ISO-8601 timestamp into an aware datetime."""

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


def _format_timestamp(value: datetime) -> str:
    """Serialize one aware timestamp as UTC ISO-8601 text."""

    return value.astimezone(timezone.utc).isoformat()


def _parse_local_time(value: object | None, *, fallback: str) -> LocalTime:
    """Parse one ``HH:MM`` local time string with a stable fallback."""

    text = str(value or "").strip() or fallback
    hour_text, separator, minute_text = text.partition(":")
    if separator != ":":
        hour_text, minute_text = fallback.split(":", 1)
    try:
        hour = int(hour_text)
        minute = int(minute_text)
    except ValueError:
        fallback_hour, fallback_minute = fallback.split(":", 1)
        return LocalTime(hour=int(fallback_hour), minute=int(fallback_minute))
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        fallback_hour, fallback_minute = fallback.split(":", 1)
        return LocalTime(hour=int(fallback_hour), minute=int(fallback_minute))
    return LocalTime(hour=hour, minute=minute)


def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    """Return one finite bounded integer config value."""

    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def _bounded_seconds(value: object, *, default: float, minimum: float, maximum: float) -> float:
    """Return one finite bounded duration."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return max(minimum, min(maximum, number))


def _normalize_topic_keys(values: Sequence[object] | None) -> tuple[str, ...]:
    """Normalize one ordered set of retired topic keys."""

    if not values:
        return ()
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        compact = _compact_text(value, max_len=96).casefold()
        if not compact or compact in seen:
            continue
        seen.add(compact)
        ordered.append(compact)
    return tuple(ordered)


def _stable_fraction(*parts: object) -> float:
    """Return one deterministic fraction in the inclusive 0..1 range."""

    digest = hashlib.sha1(
        "::".join(_compact_text(part, max_len=160) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") / 4_294_967_295.0


def _default_candidate_loader(
    config: TwinrConfig,
    *,
    local_now: datetime,
    max_items: int,
) -> tuple[AmbientDisplayImpulseCandidate, ...]:
    """Load the current reserve candidates from structured personality state."""

    return load_display_reserve_candidates(
        config,
        local_now=local_now,
        max_items=max_items,
    )


@dataclass(frozen=True, slots=True)
class DisplayReservePlannedItem:
    """Describe one planned ambient reserve-card publication for the day."""

    topic_key: str
    title: str
    source: str
    action: str
    attention_state: str
    eyebrow: str
    headline: str
    body: str
    symbol: str
    accent: str
    reason: str
    candidate_family: str
    salience: float
    hold_seconds: float

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DisplayReservePlannedItem":
        """Build one planned item from JSON-style persisted data."""

        return cls(
            topic_key=_compact_text(payload.get("topic_key"), max_len=96).casefold(),
            title=_compact_text(payload.get("title"), max_len=72),
            source=_compact_text(payload.get("source"), max_len=48),
            action=_compact_text(payload.get("action"), max_len=24).lower() or "hint",
            attention_state=_compact_text(payload.get("attention_state"), max_len=24).lower() or "background",
            eyebrow=_compact_text(payload.get("eyebrow"), max_len=36),
            headline=_compact_text(payload.get("headline"), max_len=128),
            body=_compact_text(payload.get("body"), max_len=128),
            symbol=_compact_text(payload.get("symbol"), max_len=24) or "sparkles",
            accent=_compact_text(payload.get("accent"), max_len=24).lower() or "info",
            reason=_compact_text(payload.get("reason"), max_len=120),
            candidate_family=_compact_text(payload.get("candidate_family"), max_len=40).casefold() or "general",
            salience=float(payload.get("salience", 0.0) or 0.0),
            hold_seconds=_bounded_seconds(
                payload.get("hold_seconds"),
                default=_DEFAULT_BASE_HOLD_S,
                minimum=_DEFAULT_MIN_HOLD_S,
                maximum=_DEFAULT_MAX_HOLD_S,
            ),
        )

    @classmethod
    def from_candidate(
        cls,
        candidate: AmbientDisplayImpulseCandidate,
        *,
        hold_seconds: float,
        reason: str,
    ) -> "DisplayReservePlannedItem":
        """Build one planned item from a live candidate plus planned timing."""

        return cls(
            topic_key=_compact_text(candidate.topic_key, max_len=96).casefold(),
            title=_compact_text(candidate.title, max_len=72),
            source=_compact_text(candidate.source, max_len=48),
            action=_compact_text(candidate.action, max_len=24).lower() or "hint",
            attention_state=_compact_text(candidate.attention_state, max_len=24).lower() or "background",
            eyebrow=_compact_text(candidate.eyebrow, max_len=36),
            headline=_compact_text(candidate.headline, max_len=128),
            body=_compact_text(candidate.body, max_len=128),
            symbol=_compact_text(candidate.symbol, max_len=24) or "sparkles",
            accent=_compact_text(candidate.accent, max_len=24).lower() or "info",
            reason=_compact_text(reason, max_len=120),
            candidate_family=_compact_text(candidate.candidate_family, max_len=40).casefold() or "general",
            salience=max(0.0, float(candidate.salience)),
            hold_seconds=_bounded_seconds(
                hold_seconds,
                default=_DEFAULT_BASE_HOLD_S,
                minimum=_DEFAULT_MIN_HOLD_S,
                maximum=_DEFAULT_MAX_HOLD_S,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the planned item into JSON-safe data."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class DisplayReserveDayPlan:
    """Persist the current local-day reserve publication sequence."""

    local_day: str
    generated_at: str
    cursor: int
    items: tuple[DisplayReservePlannedItem, ...]
    candidate_count: int = 0
    retired_topic_keys: tuple[str, ...] = ()

    @classmethod
    def empty(cls, *, local_day: LocalDate, generated_at: datetime | None = None) -> "DisplayReserveDayPlan":
        """Return one explicit empty plan for the requested local day."""

        return cls(
            local_day=local_day.isoformat(),
            generated_at=_format_timestamp(generated_at or _utc_now()),
            cursor=0,
            items=(),
            candidate_count=0,
            retired_topic_keys=(),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DisplayReserveDayPlan":
        """Build one day plan from persisted JSON-style data."""

        local_day = _compact_text(payload.get("local_day"), max_len=16)
        if not local_day:
            raise ValueError("display reserve day plan requires local_day")
        generated_at = _parse_timestamp(payload.get("generated_at")) or _utc_now()
        cursor = max(0, int(payload.get("cursor", 0) or 0))
        raw_items = payload.get("items")
        if not isinstance(raw_items, Sequence):
            raise ValueError("display reserve day plan items must be a sequence")
        items = tuple(
            DisplayReservePlannedItem.from_dict(entry)
            for entry in raw_items
            if isinstance(entry, Mapping)
        )
        return cls(
            local_day=local_day,
            generated_at=_format_timestamp(generated_at),
            cursor=cursor,
            items=items,
            candidate_count=max(0, int(payload.get("candidate_count", len(items)) or len(items))),
            retired_topic_keys=_normalize_topic_keys(payload.get("retired_topic_keys")),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the plan into JSON-safe data."""

        return {
            "local_day": self.local_day,
            "generated_at": self.generated_at,
            "cursor": self.cursor,
            "candidate_count": self.candidate_count,
            "retired_topic_keys": list(self.retired_topic_keys),
            "items": [item.to_dict() for item in self.items],
        }

    def active_items(self) -> tuple[DisplayReservePlannedItem, ...]:
        """Return the current same-day rotation after answered topics are retired."""

        if not self.retired_topic_keys:
            return self.items
        retired = set(self.retired_topic_keys)
        return tuple(item for item in self.items if item.topic_key not in retired)

    def current_item(self) -> DisplayReservePlannedItem | None:
        """Return the current rotating item for the local day."""

        active_items = self.active_items()
        if not active_items:
            return None
        return active_items[self.cursor % len(active_items)]

    def last_shown_item(self) -> DisplayReservePlannedItem | None:
        """Return the most recently shown item for the current local day."""

        active_items = self.active_items()
        if not active_items or self.cursor <= 0:
            return None
        return active_items[(self.cursor - 1) % len(active_items)]

    def is_exhausted(self) -> bool:
        """Return whether no active same-day rotation items remain."""

        return not bool(self.active_items())

    def advance(self) -> "DisplayReserveDayPlan":
        """Return one copy of the plan with the cursor advanced by one slot."""

        if not self.active_items():
            return self
        return DisplayReserveDayPlan(
            local_day=self.local_day,
            generated_at=self.generated_at,
            cursor=self.cursor + 1,
            items=self.items,
            candidate_count=self.candidate_count,
            retired_topic_keys=self.retired_topic_keys,
        )

    def retire_topics(self, topic_keys: Sequence[object]) -> "DisplayReserveDayPlan":
        """Return one copy with additional answered same-day topics retired."""

        retired_topic_keys = _normalize_topic_keys((*self.retired_topic_keys, *topic_keys))
        if retired_topic_keys == self.retired_topic_keys:
            return self
        return DisplayReserveDayPlan(
            local_day=self.local_day,
            generated_at=self.generated_at,
            cursor=self.cursor,
            items=self.items,
            candidate_count=self.candidate_count,
            retired_topic_keys=retired_topic_keys,
        )


@dataclass(slots=True)
class DisplayReserveDayPlanStore:
    """Read and write the persistent local-day reserve plan artifact."""

    path: Path

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayReserveDayPlanStore":
        """Resolve the reserve-plan artifact path from configuration."""

        project_root = Path(config.project_root).expanduser().resolve()
        configured = Path(
            getattr(config, "display_reserve_bus_plan_path", _DEFAULT_PLAN_PATH) or _DEFAULT_PLAN_PATH
        )
        resolved = configured if configured.is_absolute() else project_root / configured
        return cls(path=resolved)

    def load(self) -> DisplayReserveDayPlan | None:
        """Load the currently persisted day plan, if one exists and parses."""

        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            _LOGGER.warning("Failed to read display reserve day plan from %s.", self.path, exc_info=True)
            return None
        if not isinstance(payload, Mapping):
            _LOGGER.warning(
                "Ignoring invalid display reserve day plan payload at %s because it is not an object.",
                self.path,
            )
            return None
        try:
            return DisplayReserveDayPlan.from_dict(payload)
        except Exception:
            _LOGGER.warning("Ignoring invalid display reserve day plan payload at %s.", self.path, exc_info=True)
            return None

    def save(self, plan: DisplayReserveDayPlan) -> DisplayReserveDayPlan:
        """Persist one full day plan atomically enough for runtime use."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(plan.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return plan

    def clear(self) -> None:
        """Remove the persisted plan artifact when it exists."""

        try:
            self.path.unlink()
        except FileNotFoundError:
            return


@dataclass(slots=True)
class DisplayReserveDayPlanner:
    """Build and persist one deterministic local-day reserve publication plan."""

    store: DisplayReserveDayPlanStore
    feedback_store: DisplayReserveBusFeedbackStore | None = None
    candidate_loader: Callable[..., tuple[AmbientDisplayImpulseCandidate, ...]] = _default_candidate_loader
    local_now: Callable[[], datetime] = _default_local_now

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayReserveDayPlanner":
        """Build the planner and its persistent store from configuration."""

        return cls(
            store=DisplayReserveDayPlanStore.from_config(config),
            feedback_store=DisplayReserveBusFeedbackStore.from_config(config),
        )

    def ensure_plan(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime | None = None,
    ) -> DisplayReserveDayPlan:
        """Load the current plan or build today's plan when none is current."""

        effective_now = (local_now or self.local_now()).astimezone()
        current_day = effective_now.date()
        refresh_after = _parse_local_time(
            getattr(config, "display_reserve_bus_refresh_after_local", _DEFAULT_REFRESH_AFTER_LOCAL),
            fallback=_DEFAULT_REFRESH_AFTER_LOCAL,
        )
        existing = self.store.load()
        feedback_signal = self._load_active_feedback(local_now=effective_now)
        if self._plan_is_current(existing, current_day=current_day, local_now=effective_now, refresh_after=refresh_after):
            if self._feedback_requires_rebuild(existing, feedback_signal=feedback_signal):
                rebuilt = self._build_plan(
                    config=config,
                    local_now=effective_now,
                    local_day=current_day,
                    feedback_signal=feedback_signal,
                    retired_topic_keys=self._retired_topic_keys(
                        existing,
                        feedback_signal=feedback_signal,
                    ),
                )
                return self.store.save(rebuilt)
            if not self._should_retry_empty_current_plan(existing, local_now=effective_now):
                return existing
        rebuilt = self._build_plan(
            config=config,
            local_now=effective_now,
            local_day=current_day,
            feedback_signal=feedback_signal,
        )
        return self.store.save(rebuilt)

    def build_plan_for_day(
        self,
        *,
        config: TwinrConfig,
        local_day: LocalDate,
        local_now: datetime | None = None,
        feedback_signal: DisplayReserveBusFeedbackSignal | None = None,
        retired_topic_keys: frozenset[str] = frozenset(),
    ) -> DisplayReserveDayPlan:
        """Build one unsaved plan for the requested local day.

        The explicit nightly companion planner uses this method to prepare the
        next day ahead of time without replacing the active current-day plan
        immediately. Runtime publication still persists the adopted plan only
        when the next local day actually starts.
        """

        effective_now = (local_now or self.local_now()).astimezone()
        return self._build_plan(
            config=config,
            local_now=effective_now,
            local_day=local_day,
            feedback_signal=feedback_signal,
            retired_topic_keys=retired_topic_keys,
        )

    def peek_next_item(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime | None = None,
    ) -> DisplayReservePlannedItem | None:
        """Return the next planned reserve item for the current local day."""

        return self.ensure_plan(config=config, local_now=local_now).current_item()

    def peek_idle_fill_item(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime | None = None,
    ) -> DisplayReservePlannedItem | None:
        """Return one passive same-day fill item when no active topics remain."""

        plan = self.ensure_plan(config=config, local_now=local_now)
        if not plan.is_exhausted():
            return None
        return plan.last_shown_item()

    def mark_published(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime | None = None,
    ) -> DisplayReserveDayPlan:
        """Advance the current day plan after one successful publication."""

        plan = self.ensure_plan(config=config, local_now=local_now)
        advanced = plan.advance()
        return self.store.save(advanced)

    def _should_retry_empty_current_plan(
        self,
        plan: DisplayReserveDayPlan | None,
        *,
        local_now: datetime,
    ) -> bool:
        """Return whether one current empty plan should be rebuilt now.

        Empty same-day plans are not allowed to stay sticky for the whole day.
        World/personality state may become available a little later than the
        first proactive tick, so the planner retries after a short bounded
        backoff instead of caching blank reserve space until midnight.
        """

        if plan is None or plan.items:
            return False
        if plan.candidate_count > 0:
            return True
        generated_at = _parse_timestamp(plan.generated_at)
        if generated_at is None:
            return True
        age_s = (local_now.astimezone(timezone.utc) - generated_at).total_seconds()
        return age_s >= _EMPTY_PLAN_RETRY_S

    def _plan_is_current(
        self,
        plan: DisplayReserveDayPlan | None,
        *,
        current_day: LocalDate,
        local_now: datetime,
        refresh_after: LocalTime,
    ) -> bool:
        """Return whether one stored plan is still valid for the current local time."""

        if plan is None:
            return False
        if plan.local_day == current_day.isoformat():
            return True
        previous_day = current_day - timedelta(days=1)
        if plan.local_day != previous_day.isoformat():
            return False
        return local_now.timetz().replace(tzinfo=None) < refresh_after

    def _build_plan(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime,
        local_day: LocalDate,
        feedback_signal: DisplayReserveBusFeedbackSignal | None,
        retired_topic_keys: frozenset[str] = frozenset(),
    ) -> DisplayReserveDayPlan:
        """Build one fresh local-day plan from current reserve candidates."""

        candidate_limit = _bounded_int(
            getattr(config, "display_reserve_bus_candidate_limit", _DEFAULT_CANDIDATE_LIMIT),
            default=_DEFAULT_CANDIDATE_LIMIT,
            minimum=1,
            maximum=24,
        )
        items_per_day = _bounded_int(
            getattr(config, "display_reserve_bus_items_per_day", _DEFAULT_ITEMS_PER_DAY),
            default=_DEFAULT_ITEMS_PER_DAY,
            minimum=1,
            maximum=96,
        )
        topic_gap = _bounded_int(
            getattr(config, "display_reserve_bus_topic_gap", _DEFAULT_TOPIC_GAP),
            default=_DEFAULT_TOPIC_GAP,
            minimum=0,
            maximum=8,
        )
        source_gap = _DEFAULT_SOURCE_GAP
        family_gap = _DEFAULT_FAMILY_GAP
        candidates = tuple(
            self.candidate_loader(
                config,
                local_now=local_now,
                max_items=candidate_limit,
            )
        )
        if retired_topic_keys:
            candidates = tuple(
                candidate
                for candidate in candidates
                if candidate.topic_key not in retired_topic_keys
            )
        if not candidates:
            return DisplayReserveDayPlan.empty(
                local_day=local_day,
                generated_at=local_now.astimezone(timezone.utc),
            )
        cycle_slots = self._cycle_slots(
            candidates=candidates,
            configured_slots=items_per_day,
        )
        counts = self._allocate_counts(
            candidates,
            slots=cycle_slots,
            local_day=local_day,
            feedback_signal=feedback_signal,
        )
        ordered = self._schedule_candidates(
            candidates,
            counts=counts,
            local_day=local_day,
            topic_gap=topic_gap,
            source_gap=source_gap,
            family_gap=family_gap,
            feedback_signal=feedback_signal,
        )
        items = tuple(
            DisplayReservePlannedItem.from_candidate(
                candidate,
                hold_seconds=self._hold_seconds_for_candidate(config=config, candidate=candidate),
                reason=f"plan[{index}] {candidate.reason}",
            )
            for index, candidate in enumerate(ordered)
        )
        return DisplayReserveDayPlan(
            local_day=local_day.isoformat(),
            generated_at=_format_timestamp(local_now.astimezone(timezone.utc)),
            cursor=0,
            items=items,
            candidate_count=len(candidates),
            retired_topic_keys=_normalize_topic_keys(retired_topic_keys),
        )

    def _cycle_slots(
        self,
        *,
        candidates: Sequence[AmbientDisplayImpulseCandidate],
        configured_slots: int,
    ) -> int:
        """Return the unique rotation length for the current candidate pool.

        The planner stores one bounded unique cycle and reuses that cycle
        throughout the same local day. Cards do not disappear merely because
        they were shown once; they stay in rotation until real user feedback
        retires them or the next nightly plan replaces the whole day.
        """

        if not candidates:
            return 0
        return min(max(1, int(configured_slots)), len(candidates))

    def _allocate_counts(
        self,
        candidates: Sequence[AmbientDisplayImpulseCandidate],
        *,
        slots: int,
        local_day: LocalDate,
        feedback_signal: DisplayReserveBusFeedbackSignal | None,
    ) -> dict[str, int]:
        """Allocate a bounded number of daily plan slots across candidates."""

        if slots <= 0:
            return {candidate.topic_key: 0 for candidate in candidates}
        if len(candidates) >= slots:
            selected = sorted(
                candidates,
                key=lambda item: (
                    self._candidate_weight(item, feedback_signal=feedback_signal),
                    _stable_fraction(local_day.isoformat(), item.topic_key, "select"),
                ),
                reverse=True,
            )[:slots]
            selected_keys = {candidate.topic_key for candidate in selected}
            return {
                candidate.topic_key: 1 if candidate.topic_key in selected_keys else 0
                for candidate in candidates
            }
        weights = {
            candidate.topic_key: self._candidate_weight(candidate, feedback_signal=feedback_signal)
            for candidate in candidates
        }
        total = sum(weights.values()) or float(len(candidates))
        counts: dict[str, int] = {}
        remainders: list[tuple[float, AmbientDisplayImpulseCandidate]] = []
        remaining_slots = slots
        for candidate in candidates:
            ideal = (weights[candidate.topic_key] / total) * float(slots)
            count = max(1, int(math.floor(ideal)))
            counts[candidate.topic_key] = count
            remaining_slots -= count
            remainders.append((ideal - count, candidate))
        if remaining_slots < 0:
            for _remainder, candidate in sorted(
                remainders,
                key=lambda item: (
                    item[0],
                    _stable_fraction(local_day.isoformat(), item[1].topic_key, "trim"),
                ),
            ):
                if remaining_slots >= 0:
                    break
                if counts[candidate.topic_key] <= 1:
                    continue
                counts[candidate.topic_key] -= 1
                remaining_slots += 1
        if remaining_slots > 0:
            for _remainder, candidate in sorted(
                remainders,
                key=lambda item: (
                    item[0],
                    _stable_fraction(local_day.isoformat(), item[1].topic_key, "fill"),
                ),
                reverse=True,
            ):
                if remaining_slots <= 0:
                    break
                counts[candidate.topic_key] += 1
                remaining_slots -= 1
        return counts

    def _schedule_candidates(
        self,
        candidates: Sequence[AmbientDisplayImpulseCandidate],
        *,
        counts: Mapping[str, int],
        local_day: LocalDate,
        topic_gap: int,
        source_gap: int,
        family_gap: int,
        feedback_signal: DisplayReserveBusFeedbackSignal | None,
    ) -> tuple[AmbientDisplayImpulseCandidate, ...]:
        """Spread repeated candidates across the day without topic clustering."""

        remaining = {candidate.topic_key: max(0, counts.get(candidate.topic_key, 0)) for candidate in candidates}
        ordered: list[AmbientDisplayImpulseCandidate] = []
        while any(remaining.values()):
            recent_topics = tuple(item.topic_key for item in ordered[-topic_gap:]) if topic_gap > 0 else ()
            recent_sources = tuple(item.source for item in ordered[-source_gap:]) if source_gap > 0 else ()
            recent_families = (
                tuple(self._candidate_family(item) for item in ordered[-family_gap:])
                if family_gap > 0
                else ()
            )
            eligible = [
                candidate
                for candidate in candidates
                if remaining.get(candidate.topic_key, 0) > 0
                and candidate.topic_key not in recent_topics
                and candidate.source not in recent_sources
                and self._candidate_family(candidate) not in recent_families
            ]
            if not eligible:
                eligible = [
                    candidate
                    for candidate in candidates
                    if remaining.get(candidate.topic_key, 0) > 0 and candidate.topic_key not in recent_topics
                    and self._candidate_family(candidate) not in recent_families
                ]
            if not eligible:
                eligible = [candidate for candidate in candidates if remaining.get(candidate.topic_key, 0) > 0]
            candidate = max(
                eligible,
                key=lambda item: (
                    remaining[item.topic_key],
                    self._candidate_weight(item, feedback_signal=feedback_signal),
                    _stable_fraction(local_day.isoformat(), item.topic_key, len(ordered)),
                ),
            )
            ordered.append(candidate)
            remaining[candidate.topic_key] -= 1
        return tuple(ordered)

    def _candidate_family(self, candidate: AmbientDisplayImpulseCandidate) -> str:
        """Return one generic family token for reserve-plan mixing."""

        family = _compact_text(getattr(candidate, "candidate_family", None), max_len=40).casefold()
        if family:
            return family
        return _compact_text(candidate.source, max_len=40).casefold() or "general"

    def _candidate_weight(
        self,
        candidate: AmbientDisplayImpulseCandidate,
        *,
        feedback_signal: DisplayReserveBusFeedbackSignal | None,
    ) -> float:
        """Return the relative planning weight for one reserve candidate."""

        action_bonus = {
            "hint": 0.10,
            "brief_update": 0.22,
            "ask_one": 0.30,
            "invite_follow_up": 0.40,
        }.get(candidate.action, 0.10)
        attention_bonus = {
            "background": 0.00,
            "growing": 0.08,
            "forming": 0.16,
            "shared_thread": 0.24,
        }.get(candidate.attention_state, 0.00)
        return (
            1.0
            + max(0.0, float(candidate.salience))
            + action_bonus
            + attention_bonus
            + self._feedback_weight_adjustment(candidate, feedback_signal=feedback_signal)
        )

    def _feedback_weight_adjustment(
        self,
        candidate: AmbientDisplayImpulseCandidate,
        *,
        feedback_signal: DisplayReserveBusFeedbackSignal | None,
    ) -> float:
        """Return one short-lived planning bias from recent reserve feedback."""

        if feedback_signal is None:
            return 0.0
        if candidate.topic_key != feedback_signal.topic_key:
            return 0.0
        intensity = max(0.0, min(1.0, float(feedback_signal.intensity)))
        if feedback_signal.reaction == "immediate_engagement":
            return 0.62 * intensity
        if feedback_signal.reaction == "engaged":
            return 0.38 * intensity
        if feedback_signal.reaction == "cooled":
            return -(0.26 * intensity)
        if feedback_signal.reaction == "avoided":
            return -(0.46 * intensity)
        if feedback_signal.reaction == "ignored":
            return -(0.18 * intensity)
        return 0.0

    def _load_active_feedback(
        self,
        *,
        local_now: datetime,
    ) -> DisplayReserveBusFeedbackSignal | None:
        """Load the current active reserve-bus feedback hint, if any."""

        if self.feedback_store is None:
            return None
        return self.feedback_store.load_active(now=local_now.astimezone(timezone.utc))

    def _feedback_requires_rebuild(
        self,
        plan: DisplayReserveDayPlan | None,
        *,
        feedback_signal: DisplayReserveBusFeedbackSignal | None,
    ) -> bool:
        """Return whether fresh bus feedback should rebuild today's plan."""

        if plan is None or feedback_signal is None:
            return False
        if plan.is_exhausted():
            return False
        generated_at = _parse_timestamp(plan.generated_at)
        if generated_at is None:
            return True
        return feedback_signal.requested_at_datetime() > generated_at

    def _retired_topic_keys(
        self,
        plan: DisplayReserveDayPlan | None,
        *,
        feedback_signal: DisplayReserveBusFeedbackSignal | None,
    ) -> frozenset[str]:
        """Return the same-day topic keys that should leave rotation now.

        Cards stay in the same-day loop until the user actually responds to
        them. Immediate/delayed engagement and explicit cooling/avoidance count
        as answered. Merely being shown once, or later being weakly ignored,
        does not retire the card from today's rotation.
        """

        retired = set(plan.retired_topic_keys if plan is not None else ())
        if feedback_signal is None:
            return frozenset(retired)
        if not feedback_signal.topic_key:
            return frozenset(retired)
        if feedback_signal.reaction in {"immediate_engagement", "engaged", "cooled", "avoided"}:
            retired.add(feedback_signal.topic_key)
        return frozenset(retired)

    def _hold_seconds_for_candidate(
        self,
        *,
        config: TwinrConfig,
        candidate: AmbientDisplayImpulseCandidate,
    ) -> float:
        """Return the desired on-screen duration for one planned candidate."""

        base = _bounded_seconds(
            getattr(config, "display_reserve_bus_base_hold_s", _DEFAULT_BASE_HOLD_S),
            default=_DEFAULT_BASE_HOLD_S,
            minimum=_DEFAULT_MIN_HOLD_S,
            maximum=_DEFAULT_MAX_HOLD_S,
        )
        minimum = _bounded_seconds(
            getattr(config, "display_reserve_bus_min_hold_s", _DEFAULT_MIN_HOLD_S),
            default=_DEFAULT_MIN_HOLD_S,
            minimum=60.0,
            maximum=_DEFAULT_MAX_HOLD_S,
        )
        maximum = _bounded_seconds(
            getattr(config, "display_reserve_bus_max_hold_s", _DEFAULT_MAX_HOLD_S),
            default=_DEFAULT_MAX_HOLD_S,
            minimum=minimum,
            maximum=2.0 * 60.0 * 60.0,
        )
        hold = (
            base
            + _ACTION_BONUS_S.get(candidate.action, 0.0)
            + _ATTENTION_BONUS_S.get(candidate.attention_state, 0.0)
            + (max(0.0, float(candidate.salience)) * 3.0 * 60.0)
        )
        return max(minimum, min(maximum, hold))
