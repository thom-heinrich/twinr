"""Own nightly reserve-lane recalibration and prepared next-day plans.

The right-hand HDMI reserve lane already has a deterministic current-day
planner and a slower companion-flow candidate loader. What was still missing
was one explicit overnight step that:

- reviews recent shown-card outcomes
- runs long-term reflection/world refresh once for the coming day
- prepares the next local-day reserve plan ahead of time
- lets the morning runtime adopt that prepared plan without rebuilding it live

This module keeps that slower daily orchestration out of both the hot
display-publish path and the generic realtime loop.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date as LocalDate, datetime, time as LocalTime, timedelta, timezone
import json
import logging
from pathlib import Path

from twinr.agent.base_agent.config import TwinrConfig
from twinr.display.ambient_impulse_history import DisplayAmbientImpulseHistoryStore
from twinr.memory.longterm.runtime.service import LongTermMemoryService
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError

from .display_reserve_day_plan import (
    DisplayReserveDayPlan,
    DisplayReserveDayPlanStore,
    DisplayReserveDayPlanner,
    DisplayReservePlannedItem,
)
from .display_reserve_learning import (
    DisplayReserveLearningProfile,
    DisplayReserveLearningProfileBuilder,
)

_DEFAULT_PREPARED_PLAN_PATH = "artifacts/stores/ops/display_reserve_bus_plan_prepared.json"
_DEFAULT_MAINTENANCE_STATE_PATH = "artifacts/stores/ops/display_reserve_bus_maintenance.json"
_DEFAULT_NIGHTLY_AFTER_LOCAL = "00:30"
_DEFAULT_OUTCOME_LOOKBACK_DAYS = 2.0

_LOGGER = logging.getLogger(__name__)


def _default_local_now() -> datetime:
    """Return the current local wall clock as an aware datetime."""

    return datetime.now().astimezone()


def _utc_now() -> datetime:
    """Return the current UTC wall clock."""

    return datetime.now(timezone.utc)


def _compact_text(value: object | None, *, max_len: int) -> str:
    """Collapse arbitrary text into one bounded single line."""

    if value is None:
        return ""
    compact = " ".join(str(value).split()).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _parse_timestamp(value: object | None) -> datetime | None:
    """Parse one optional ISO-8601 timestamp into an aware UTC datetime."""

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
    """Parse one bounded ``HH:MM`` local time."""

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


def _resolve_store_path(config: TwinrConfig, attr_name: str, default_path: str) -> Path:
    """Resolve one configured artifact path under the current project root."""

    project_root = Path(config.project_root).expanduser().resolve()
    configured = Path(getattr(config, attr_name, default_path) or default_path)
    return configured if configured.is_absolute() else project_root / configured


def _coerce_iso_day(value: object | None) -> str | None:
    """Normalize one stored local-day string."""

    text = _compact_text(value, max_len=16)
    return text or None


def _topic_key(value: object | None) -> str:
    """Return one normalized topic key."""

    return _compact_text(value, max_len=96).casefold()


@dataclass(frozen=True, slots=True)
class DisplayReserveNightlyOutcomeSummary:
    """Summarize recent reserve-lane outcomes for overnight review."""

    exposure_count: int = 0
    engaged_count: int = 0
    immediate_pickup_count: int = 0
    cooling_count: int = 0
    avoided_count: int = 0
    ignored_count: int = 0
    pending_count: int = 0
    positive_topics: tuple[str, ...] = ()
    cooling_topics: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DisplayReserveNightlyOutcomeSummary":
        """Build one summary from persisted JSON-safe data."""

        positive_topics = payload.get("positive_topics")
        cooling_topics = payload.get("cooling_topics")
        return cls(
            exposure_count=max(0, int(payload.get("exposure_count", 0) or 0)),
            engaged_count=max(0, int(payload.get("engaged_count", 0) or 0)),
            immediate_pickup_count=max(0, int(payload.get("immediate_pickup_count", 0) or 0)),
            cooling_count=max(0, int(payload.get("cooling_count", 0) or 0)),
            avoided_count=max(0, int(payload.get("avoided_count", 0) or 0)),
            ignored_count=max(0, int(payload.get("ignored_count", 0) or 0)),
            pending_count=max(0, int(payload.get("pending_count", 0) or 0)),
            positive_topics=tuple(
                _topic_key(entry)
                for entry in positive_topics
                if _topic_key(entry)
            )
            if isinstance(positive_topics, Sequence)
            else (),
            cooling_topics=tuple(
                _topic_key(entry)
                for entry in cooling_topics
                if _topic_key(entry)
            )
            if isinstance(cooling_topics, Sequence)
            else (),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize one summary into JSON-safe data."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class DisplayReserveNightlyMaintenanceState:
    """Persist the last explicit overnight reserve-lane maintenance result."""

    prepared_local_day: str | None = None
    last_attempted_at: str | None = None
    last_completed_at: str | None = None
    last_status: str = "idle"
    last_error: str | None = None
    reflection_reflected_object_count: int = 0
    reflection_created_summary_count: int = 0
    prepared_candidate_count: int = 0
    prepared_item_count: int = 0
    positive_topics: tuple[str, ...] = ()
    cooling_topics: tuple[str, ...] = ()
    outcome_summary: DisplayReserveNightlyOutcomeSummary = DisplayReserveNightlyOutcomeSummary()

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DisplayReserveNightlyMaintenanceState":
        """Build one state record from JSON-safe persisted data."""

        outcome_payload = payload.get("outcome_summary")
        return cls(
            prepared_local_day=_coerce_iso_day(payload.get("prepared_local_day")),
            last_attempted_at=(
                _format_timestamp(_parse_timestamp(payload.get("last_attempted_at")))
                if _parse_timestamp(payload.get("last_attempted_at")) is not None
                else None
            ),
            last_completed_at=(
                _format_timestamp(_parse_timestamp(payload.get("last_completed_at")))
                if _parse_timestamp(payload.get("last_completed_at")) is not None
                else None
            ),
            last_status=_compact_text(payload.get("last_status"), max_len=24).lower() or "idle",
            last_error=_compact_text(payload.get("last_error"), max_len=240) or None,
            reflection_reflected_object_count=max(
                0,
                int(payload.get("reflection_reflected_object_count", 0) or 0),
            ),
            reflection_created_summary_count=max(
                0,
                int(payload.get("reflection_created_summary_count", 0) or 0),
            ),
            prepared_candidate_count=max(0, int(payload.get("prepared_candidate_count", 0) or 0)),
            prepared_item_count=max(0, int(payload.get("prepared_item_count", 0) or 0)),
            positive_topics=tuple(
                _topic_key(entry)
                for entry in payload.get("positive_topics", ())
                if _topic_key(entry)
            ),
            cooling_topics=tuple(
                _topic_key(entry)
                for entry in payload.get("cooling_topics", ())
                if _topic_key(entry)
            ),
            outcome_summary=DisplayReserveNightlyOutcomeSummary.from_dict(outcome_payload)
            if isinstance(outcome_payload, Mapping)
            else DisplayReserveNightlyOutcomeSummary(),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize one maintenance state into JSON-safe data."""

        payload = asdict(self)
        payload["outcome_summary"] = self.outcome_summary.to_dict()
        return payload


@dataclass(slots=True)
class DisplayReserveNightlyMaintenanceStateStore:
    """Persist the last explicit nightly companion-maintenance state."""

    path: Path

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayReserveNightlyMaintenanceStateStore":
        """Resolve the maintenance-state artifact path from configuration."""

        return cls(
            path=_resolve_store_path(
                config,
                "display_reserve_bus_maintenance_state_path",
                _DEFAULT_MAINTENANCE_STATE_PATH,
            )
        )

    def load(self) -> DisplayReserveNightlyMaintenanceState | None:
        """Load the persisted maintenance state when it exists and parses."""

        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            _LOGGER.warning(
                "Failed to read display reserve maintenance state from %s.",
                self.path,
                exc_info=True,
            )
            return None
        if not isinstance(payload, Mapping):
            return None
        try:
            return DisplayReserveNightlyMaintenanceState.from_dict(payload)
        except Exception:
            _LOGGER.warning(
                "Ignoring invalid display reserve maintenance state at %s.",
                self.path,
                exc_info=True,
            )
            return None

    def save(
        self,
        state: DisplayReserveNightlyMaintenanceState,
    ) -> DisplayReserveNightlyMaintenanceState:
        """Persist one maintenance state atomically enough for runtime use."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return state


@dataclass(frozen=True, slots=True)
class DisplayReserveNightlyMaintenanceResult:
    """Describe one explicit overnight reserve-lane maintenance outcome."""

    action: str
    reason: str
    target_local_day: str | None = None
    plan: DisplayReserveDayPlan | None = None
    state: DisplayReserveNightlyMaintenanceState | None = None


@dataclass(slots=True)
class DisplayReserveCompanionPlanner:
    """Coordinate current-day reserve use with explicit nightly preparation."""

    day_planner: DisplayReserveDayPlanner
    prepared_store: DisplayReserveDayPlanStore
    state_store: DisplayReserveNightlyMaintenanceStateStore
    history_store: DisplayAmbientImpulseHistoryStore
    learning_builder_factory: type[DisplayReserveLearningProfileBuilder] = DisplayReserveLearningProfileBuilder
    long_term_memory_factory: Callable[[TwinrConfig], LongTermMemoryService] = LongTermMemoryService.from_config
    local_now: Callable[[], datetime] = _default_local_now

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayReserveCompanionPlanner":
        """Build the companion planner and its durable stores from config."""

        return cls(
            day_planner=DisplayReserveDayPlanner.from_config(config),
            prepared_store=DisplayReserveDayPlanStore(
                path=_resolve_store_path(
                    config,
                    "display_reserve_bus_prepared_plan_path",
                    _DEFAULT_PREPARED_PLAN_PATH,
                )
            ),
            state_store=DisplayReserveNightlyMaintenanceStateStore.from_config(config),
            history_store=DisplayAmbientImpulseHistoryStore.from_config(config),
        )

    @property
    def store(self) -> DisplayReserveDayPlanStore:
        """Expose the active current-day plan store for existing callers/tests."""

        return self.day_planner.store

    @property
    def feedback_store(self):
        """Expose the short-lived feedback store for existing callers/tests."""

        return self.day_planner.feedback_store

    @property
    def candidate_loader(self):
        """Expose the wrapped day-planner candidate loader for tests."""

        return self.day_planner.candidate_loader

    @candidate_loader.setter
    def candidate_loader(self, value) -> None:
        """Replace the wrapped day-planner candidate loader for tests."""

        self.day_planner.candidate_loader = value

    def ensure_plan(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime | None = None,
    ) -> DisplayReserveDayPlan:
        """Return the active plan, adopting a prepared next-day plan when due."""

        effective_now = (local_now or self.local_now()).astimezone()
        self._clear_stale_prepared_plan(current_day=effective_now.date())
        adopted = self._adopt_prepared_plan_if_due(config=config, local_now=effective_now)
        if adopted is not None:
            return adopted
        return self.day_planner.ensure_plan(config=config, local_now=effective_now)

    def peek_next_item(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime | None = None,
    ) -> DisplayReservePlannedItem | None:
        """Return the next planned item for the current local day."""

        return self.ensure_plan(config=config, local_now=local_now).current_item()

    def peek_idle_fill_item(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime | None = None,
    ) -> DisplayReservePlannedItem | None:
        """Return one passive same-day fill item when the active plan is exhausted."""

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
        """Advance the adopted/current day plan after one successful publish."""

        plan = self.ensure_plan(config=config, local_now=local_now)
        advanced = plan.advance()
        return self.store.save(advanced)

    def maybe_run_nightly_maintenance(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime | None = None,
        search_backend: object | None = None,
    ) -> DisplayReserveNightlyMaintenanceResult:
        """Run one explicit overnight review and prepare the next local day.

        The planner only runs once per target local day after the configured
        nightly cutoff. It executes long-term reflection/world refresh, derives
        a fresh learning profile, summarizes recent reserve-card outcomes, and
        prepares the next day plan into a separate artifact.
        """

        effective_now = (local_now or self.local_now()).astimezone()
        if not bool(getattr(config, "display_reserve_bus_nightly_enabled", True)):
            return DisplayReserveNightlyMaintenanceResult(
                action="inactive",
                reason="nightly_disabled",
            )

        target_day = self._nightly_target_local_day(config=config, local_now=effective_now)
        if target_day is None:
            return DisplayReserveNightlyMaintenanceResult(
                action="skipped",
                reason="not_due",
            )
        existing_state = self.state_store.load()
        prepared = self.prepared_store.load()
        active_plan = self.store.load()
        target_day_text = target_day.isoformat()
        if (
            existing_state is not None
            and existing_state.prepared_local_day == target_day_text
            and existing_state.last_status == "prepared"
            and (
                (prepared is not None and prepared.local_day == target_day_text)
                or (active_plan is not None and active_plan.local_day == target_day_text)
            )
        ):
            return DisplayReserveNightlyMaintenanceResult(
                action="skipped",
                reason="already_prepared",
                target_local_day=target_day_text,
                plan=prepared if prepared is not None else active_plan,
                state=existing_state,
            )

        attempted_at = effective_now.astimezone(timezone.utc)
        try:
            memory_service = self.long_term_memory_factory(config)
            reflection = memory_service.run_reflection(search_backend=search_backend)
            learning_profile = self.learning_builder_factory.from_config(config).build(now=effective_now)
            outcome_summary = self._build_outcome_summary(
                previous_state=existing_state,
                local_now=effective_now,
            )
            plan = self.day_planner.build_plan_for_day(
                config=config,
                local_day=target_day,
                local_now=effective_now,
            )
            prepared_plan = DisplayReserveDayPlan(
                local_day=plan.local_day,
                generated_at=plan.generated_at,
                cursor=0,
                items=plan.items,
                candidate_count=plan.candidate_count,
                retired_topic_keys=plan.retired_topic_keys,
            )
            self.prepared_store.save(prepared_plan)
            state = DisplayReserveNightlyMaintenanceState(
                prepared_local_day=target_day_text,
                last_attempted_at=_format_timestamp(attempted_at),
                last_completed_at=_format_timestamp(_utc_now()),
                last_status="prepared",
                last_error=None,
                reflection_reflected_object_count=len(reflection.reflected_objects),
                reflection_created_summary_count=len(reflection.created_summaries),
                prepared_candidate_count=prepared_plan.candidate_count,
                prepared_item_count=len(prepared_plan.items),
                positive_topics=self._positive_topics_from_profile(learning_profile),
                cooling_topics=self._cooling_topics_from_profile(learning_profile),
                outcome_summary=outcome_summary,
            )
            self.state_store.save(state)
            return DisplayReserveNightlyMaintenanceResult(
                action="prepared",
                reason="prepared_next_day",
                target_local_day=target_day_text,
                plan=prepared_plan,
                state=state,
            )
        except LongTermRemoteUnavailableError:
            failed_state = DisplayReserveNightlyMaintenanceState(
                prepared_local_day=target_day_text,
                last_attempted_at=_format_timestamp(attempted_at),
                last_completed_at=existing_state.last_completed_at if existing_state is not None else None,
                last_status="failed",
                last_error="remote_unavailable",
                reflection_reflected_object_count=0,
                reflection_created_summary_count=0,
                prepared_candidate_count=0,
                prepared_item_count=0,
                positive_topics=existing_state.positive_topics if existing_state is not None else (),
                cooling_topics=existing_state.cooling_topics if existing_state is not None else (),
                outcome_summary=existing_state.outcome_summary
                if existing_state is not None
                else DisplayReserveNightlyOutcomeSummary(),
            )
            self.state_store.save(failed_state)
            raise
        except Exception as exc:
            failed_state = DisplayReserveNightlyMaintenanceState(
                prepared_local_day=target_day_text,
                last_attempted_at=_format_timestamp(attempted_at),
                last_completed_at=existing_state.last_completed_at if existing_state is not None else None,
                last_status="failed",
                last_error=_compact_text(exc, max_len=240) or exc.__class__.__name__,
                reflection_reflected_object_count=0,
                reflection_created_summary_count=0,
                prepared_candidate_count=0,
                prepared_item_count=0,
                positive_topics=existing_state.positive_topics if existing_state is not None else (),
                cooling_topics=existing_state.cooling_topics if existing_state is not None else (),
                outcome_summary=existing_state.outcome_summary
                if existing_state is not None
                else DisplayReserveNightlyOutcomeSummary(),
            )
            self.state_store.save(failed_state)
            raise

    def _nightly_target_local_day(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime,
    ) -> LocalDate | None:
        """Return the local day that should be prepared now, if any."""

        nightly_after = _parse_local_time(
            getattr(config, "display_reserve_bus_nightly_after_local", _DEFAULT_NIGHTLY_AFTER_LOCAL),
            fallback=_DEFAULT_NIGHTLY_AFTER_LOCAL,
        )
        refresh_after = _parse_local_time(
            getattr(config, "display_reserve_bus_refresh_after_local", "05:30"),
            fallback="05:30",
        )
        current_time = local_now.timetz().replace(tzinfo=None)
        if nightly_after < refresh_after:
            if not (nightly_after <= current_time < refresh_after):
                return None
            return local_now.date()
        if current_time >= nightly_after:
            return local_now.date() + timedelta(days=1)
        if current_time < refresh_after:
            return local_now.date()
        return None

    def _adopt_prepared_plan_if_due(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime,
    ) -> DisplayReserveDayPlan | None:
        """Adopt a prepared next-day plan once the current day should start."""

        refresh_after = _parse_local_time(
            getattr(config, "display_reserve_bus_refresh_after_local", "05:30"),
            fallback="05:30",
        )
        current_time = local_now.timetz().replace(tzinfo=None)
        if current_time < refresh_after:
            return None
        prepared = self.prepared_store.load()
        if prepared is None:
            return None
        current_day_text = local_now.date().isoformat()
        if prepared.local_day != current_day_text:
            return None
        existing = self.store.load()
        if existing is not None and existing.local_day == current_day_text:
            self.prepared_store.clear()
            return existing
        adopted = DisplayReserveDayPlan(
            local_day=prepared.local_day,
            generated_at=prepared.generated_at,
            cursor=0,
            items=prepared.items,
            candidate_count=prepared.candidate_count,
            retired_topic_keys=prepared.retired_topic_keys,
        )
        self.prepared_store.clear()
        return self.store.save(adopted)

    def _clear_stale_prepared_plan(self, *, current_day: LocalDate) -> None:
        """Drop prepared plans that missed their day and can no longer be used."""

        prepared = self.prepared_store.load()
        if prepared is None:
            return
        prepared_day = _coerce_iso_day(prepared.local_day)
        if prepared_day is None:
            self.prepared_store.clear()
            return
        if prepared_day < current_day.isoformat():
            self.prepared_store.clear()

    def _build_outcome_summary(
        self,
        *,
        previous_state: DisplayReserveNightlyMaintenanceState | None,
        local_now: datetime,
    ) -> DisplayReserveNightlyOutcomeSummary:
        """Summarize recent shown-card outcomes for the nightly review."""

        if previous_state is not None and previous_state.last_completed_at:
            window_start = _parse_timestamp(previous_state.last_completed_at) or (
                local_now.astimezone(timezone.utc) - timedelta(days=1)
            )
        else:
            window_start = local_now.astimezone(timezone.utc) - timedelta(
                days=_DEFAULT_OUTCOME_LOOKBACK_DAYS
            )
        exposures = [
            exposure
            for exposure in self.history_store.load()
            if exposure.shown_at_datetime() >= window_start
        ]
        positive_topics = Counter[str]()
        cooling_topics = Counter[str]()
        engaged_count = 0
        immediate_pickup_count = 0
        cooling_count = 0
        avoided_count = 0
        ignored_count = 0
        pending_count = 0
        for exposure in exposures:
            status = _compact_text(exposure.response_status, max_len=24).casefold()
            if status == "engaged":
                engaged_count += 1
                positive_topics[exposure.topic_key] += 1
                if exposure.response_mode == "voice_immediate_pickup":
                    immediate_pickup_count += 1
            elif status == "cooled":
                cooling_count += 1
                cooling_topics[exposure.topic_key] += 1
            elif status == "avoided":
                avoided_count += 1
                cooling_topics[exposure.topic_key] += 2
            elif status == "ignored":
                ignored_count += 1
                cooling_topics[exposure.topic_key] += 1
            elif status == "pending":
                pending_count += 1
        return DisplayReserveNightlyOutcomeSummary(
            exposure_count=len(exposures),
            engaged_count=engaged_count,
            immediate_pickup_count=immediate_pickup_count,
            cooling_count=cooling_count,
            avoided_count=avoided_count,
            ignored_count=ignored_count,
            pending_count=pending_count,
            positive_topics=tuple(topic for topic, _count in positive_topics.most_common(4)),
            cooling_topics=tuple(topic for topic, _count in cooling_topics.most_common(4)),
        )

    def _positive_topics_from_profile(
        self,
        profile: DisplayReserveLearningProfile,
    ) -> tuple[str, ...]:
        """Return the strongest pulling topics from the long-horizon profile."""

        ranked = sorted(
            profile.topics.values(),
            key=lambda signal: (
                signal.normalized_score,
                signal.immediate_pickup_weight,
                signal.positive_weight,
            ),
            reverse=True,
        )
        selected: list[str] = []
        for signal in ranked:
            if signal.normalized_score <= 0.0 and signal.immediate_pickup_weight <= 0.0:
                continue
            selected.append(signal.key)
            if len(selected) >= 4:
                break
        return tuple(selected)

    def _cooling_topics_from_profile(
        self,
        profile: DisplayReserveLearningProfile,
    ) -> tuple[str, ...]:
        """Return the strongest cooling topics from the long-horizon profile."""

        ranked = sorted(
            profile.topics.values(),
            key=lambda signal: (
                -signal.normalized_score,
                signal.negative_weight,
                signal.repetition_pressure,
            ),
            reverse=True,
        )
        selected: list[str] = []
        for signal in ranked:
            if signal.normalized_score >= 0.0 and signal.negative_weight <= 0.0:
                continue
            selected.append(signal.key)
            if len(selected) >= 4:
                break
        return tuple(selected)


__all__ = [
    "DisplayReserveCompanionPlanner",
    "DisplayReserveNightlyMaintenanceResult",
    "DisplayReserveNightlyMaintenanceState",
    "DisplayReserveNightlyMaintenanceStateStore",
    "DisplayReserveNightlyOutcomeSummary",
]
