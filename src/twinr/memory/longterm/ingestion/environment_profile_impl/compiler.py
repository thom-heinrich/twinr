"""Compiler orchestration for room-agnostic smart-home environment profiles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.core.models import LongTermMemoryObjectV1, LongTermReflectionResultV1

from .constants import (
    _DEFAULT_ACUTE_EMPIRICAL_Q,
    _DEFAULT_ACUTE_Z_THRESHOLD,
    _DEFAULT_BASELINE_DAYS,
    _DEFAULT_DAY_START_HOUR,
    _DEFAULT_DRIFT_MIN_DAYS,
    _DEFAULT_DRIFT_MIN_SIGMA,
    _DEFAULT_EPOCH_MINUTES,
    _DEFAULT_HISTORY_DAYS,
    _DEFAULT_IQR_MULTIPLIER,
    _DEFAULT_LONG_BASELINE_DAYS,
    _DEFAULT_MIN_BASELINE_DAYS,
    _DEFAULT_MIN_COVERAGE_RATIO,
    _DEFAULT_NIGHT_START_HOUR,
    _DEFAULT_REGIME_ACCEPT_DAYS,
    _DEFAULT_TIMEZONE_NAME,
    _DEFAULT_TRANSITION_WINDOW_S,
)
from .helpers import _normalize_datetime, _resolve_timezone
from .pipeline import (
    build_baselines,
    build_day_profiles,
    build_nodes,
    extract_events,
    node_event_ids,
    update_reference_profile_similarity_markers,
)
from .signals import build_deviations, build_quality_state, build_regime_signals

logger = logging.getLogger("twinr.memory.longterm.ingestion.environment_profile")


@dataclass(frozen=True, slots=True)
class LongTermEnvironmentProfileCompiler:
    """Compile room-agnostic smart-home environment markers and deviations."""

    timezone_name: str = _DEFAULT_TIMEZONE_NAME
    enabled: bool = False
    environment_id: str = "home:main"
    baseline_days: int = _DEFAULT_BASELINE_DAYS
    short_baseline_days: int = _DEFAULT_BASELINE_DAYS
    long_baseline_days: int = _DEFAULT_LONG_BASELINE_DAYS
    history_days: int = _DEFAULT_LONG_BASELINE_DAYS
    min_baseline_days: int = _DEFAULT_MIN_BASELINE_DAYS
    epoch_minutes: int = _DEFAULT_EPOCH_MINUTES
    transition_window_s: float = _DEFAULT_TRANSITION_WINDOW_S
    day_start_hour: int = _DEFAULT_DAY_START_HOUR
    night_start_hour: int = _DEFAULT_NIGHT_START_HOUR
    iqr_multiplier: float = _DEFAULT_IQR_MULTIPLIER
    acute_z_threshold: float = _DEFAULT_ACUTE_Z_THRESHOLD
    acute_empirical_q: float = _DEFAULT_ACUTE_EMPIRICAL_Q
    drift_min_sigma: float = _DEFAULT_DRIFT_MIN_SIGMA
    drift_min_days: int = _DEFAULT_DRIFT_MIN_DAYS
    regime_accept_days: int = _DEFAULT_REGIME_ACCEPT_DAYS
    min_coverage_ratio: float = _DEFAULT_MIN_COVERAGE_RATIO

    def __post_init__(self) -> None:
        """Keep legacy direct-instantiation fields backward compatible."""

        if self.short_baseline_days == _DEFAULT_BASELINE_DAYS and self.baseline_days != _DEFAULT_BASELINE_DAYS:
            object.__setattr__(self, "short_baseline_days", max(7, int(self.baseline_days)))
        if self.long_baseline_days < self.short_baseline_days:
            object.__setattr__(self, "long_baseline_days", max(self.short_baseline_days * 2, _DEFAULT_LONG_BASELINE_DAYS))
        if self.history_days < self.long_baseline_days:
            object.__setattr__(self, "history_days", max(self.long_baseline_days, self.history_days))

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "LongTermEnvironmentProfileCompiler":
        """Build the compiler from Twinr config using the existing sensor-memory gate."""

        baseline_days = max(7, int(getattr(config, "long_term_memory_sensor_baseline_days", _DEFAULT_BASELINE_DAYS) or _DEFAULT_BASELINE_DAYS))
        short_baseline_days = max(
            7,
            int(
                getattr(config, "long_term_memory_environment_short_baseline_days", baseline_days)
                or baseline_days
            ),
        )
        long_baseline_days = max(
            short_baseline_days * 2,
            int(
                getattr(config, "long_term_memory_environment_long_baseline_days", _DEFAULT_LONG_BASELINE_DAYS)
                or _DEFAULT_LONG_BASELINE_DAYS
            ),
        )
        min_days = max(
            3,
            int(
                getattr(config, "long_term_memory_environment_min_baseline_days", getattr(config, "long_term_memory_sensor_min_days_observed", _DEFAULT_MIN_BASELINE_DAYS))
                or _DEFAULT_MIN_BASELINE_DAYS
            ),
        )
        return cls(
            timezone_name=getattr(config, "local_timezone_name", _DEFAULT_TIMEZONE_NAME),
            enabled=bool(getattr(config, "long_term_memory_sensor_memory_enabled", False)),
            baseline_days=baseline_days,
            short_baseline_days=short_baseline_days,
            long_baseline_days=long_baseline_days,
            history_days=max(_DEFAULT_HISTORY_DAYS, long_baseline_days),
            min_baseline_days=min_days,
            acute_z_threshold=max(
                0.5,
                float(
                    getattr(config, "long_term_memory_environment_acute_z_threshold", _DEFAULT_ACUTE_Z_THRESHOLD)
                    or _DEFAULT_ACUTE_Z_THRESHOLD
                ),
            ),
            acute_empirical_q=min(
                0.2,
                max(
                    0.001,
                    float(
                        getattr(config, "long_term_memory_environment_acute_empirical_q", _DEFAULT_ACUTE_EMPIRICAL_Q)
                        or _DEFAULT_ACUTE_EMPIRICAL_Q
                    ),
                ),
            ),
            drift_min_sigma=max(
                0.5,
                float(
                    getattr(config, "long_term_memory_environment_drift_min_sigma", _DEFAULT_DRIFT_MIN_SIGMA)
                    or _DEFAULT_DRIFT_MIN_SIGMA
                ),
            ),
            drift_min_days=max(
                3,
                int(
                    getattr(config, "long_term_memory_environment_drift_min_days", _DEFAULT_DRIFT_MIN_DAYS)
                    or _DEFAULT_DRIFT_MIN_DAYS
                ),
            ),
            regime_accept_days=max(
                5,
                int(
                    getattr(config, "long_term_memory_environment_regime_accept_days", _DEFAULT_REGIME_ACCEPT_DAYS)
                    or _DEFAULT_REGIME_ACCEPT_DAYS
                ),
            ),
            min_coverage_ratio=min(
                1.0,
                max(
                    0.1,
                    float(
                        getattr(config, "long_term_memory_environment_min_coverage_ratio", _DEFAULT_MIN_COVERAGE_RATIO)
                        or _DEFAULT_MIN_COVERAGE_RATIO
                    ),
                ),
            ),
        )

    def compile(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
        now: datetime | None = None,
    ) -> LongTermReflectionResultV1:
        """Compile environment-profile objects from raw smart-home pattern history."""

        empty = LongTermReflectionResultV1(reflected_objects=(), created_summaries=())
        if not self.enabled:
            return empty

        try:
            resolved_timezone = _resolve_timezone(self.timezone_name)
            reference = _normalize_datetime(now or datetime.now(timezone.utc), timezone=resolved_timezone) or datetime.now(timezone.utc)
            reference_day = reference.date()
            events = extract_events(self, objects=objects, timezone=resolved_timezone)
            if not events:
                return empty

            history_start = reference_day - timedelta(days=max(1, self.history_days - 1))
            relevant_events = tuple(
                event
                for event in events
                if history_start <= event.local_day <= reference_day
            )
            if not relevant_events:
                return empty

            nodes = build_nodes(self, events=relevant_events)
            day_profiles, day_event_ids = build_day_profiles(self, events=relevant_events)
            day_profiles = update_reference_profile_similarity_markers(
                self,
                day_profiles=day_profiles,
                events=relevant_events,
                reference_day=reference_day,
            )
            baselines, baseline_event_ids = build_baselines(
                self,
                day_profiles=day_profiles,
                reference_day=reference_day,
                day_event_ids=day_event_ids,
            )
            quality_state = build_quality_state(
                reference=reference,
                reference_day=reference_day,
                nodes=nodes,
                day_profiles=day_profiles,
                baselines=baselines,
                compiler=self,
            )
            deviations = build_deviations(
                reference=reference,
                reference_day=reference_day,
                day_profiles=day_profiles,
                baselines=baselines,
                quality_state=quality_state,
                compiler=self,
            )
            change_point, regime = build_regime_signals(
                reference=reference,
                reference_day=reference_day,
                day_profiles=day_profiles,
                baselines=baselines,
                quality_state=quality_state,
                compiler=self,
            )

            created: list[LongTermMemoryObjectV1] = []
            for node in nodes:
                created.append(
                    node.as_memory_object(
                        event_ids=node_event_ids(node_id=node.node_id, events=relevant_events),
                    )
                )
            for profile in sorted(day_profiles.values(), key=lambda item: item.day):
                created.append(
                    profile.as_memory_object(
                        event_ids=day_event_ids.get(profile.day, ()),
                    )
                )
            for key, baseline in baselines.items():
                created.append(
                    baseline.as_memory_object(
                        valid_from=reference_day - timedelta(days=max(1, baseline.window_days - 1)),
                        valid_to=reference_day,
                        event_ids=baseline_event_ids.get(key, ()),
                    )
                )
            if quality_state is not None:
                created.append(
                    quality_state.as_memory_object(
                        event_ids=day_event_ids.get(reference_day, ()),
                    )
                )
            for deviation in deviations:
                created.append(
                    deviation.as_memory_object(
                        event_ids=day_event_ids.get(reference_day, ()),
                    )
                )
            if change_point is not None:
                created.append(
                    change_point.as_memory_object(
                        event_ids=day_event_ids.get(reference_day, ()),
                    )
                )
            if regime is not None:
                created.append(
                    regime.as_memory_object(
                        event_ids=day_event_ids.get(reference_day, ()),
                    )
                )
            return LongTermReflectionResultV1(reflected_objects=(), created_summaries=tuple(created))
        except Exception:
            logger.exception("Failed to compile smart-home environment profiles.")
            return empty


__all__ = ["LongTermEnvironmentProfileCompiler"]
