"""Datamodels and memory-object renderers for environment-profile compilation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime

from twinr.memory.longterm.core.models import LongTermMemoryObjectV1, LongTermSourceRefV1

from .constants import _MAX_SOURCE_EVENT_IDS, _SMART_HOME_ENVIRONMENT_DOMAIN
from .helpers import _normalize_text, _tokenize_identifier


@dataclass(frozen=True, slots=True)
class SmartHomeEnvironmentEvent:
    """Describe one normalized smart-home event used by the environment compiler."""

    source_event_id: str
    environment_id: str
    node_id: str
    observed_at: datetime
    signal_kind: str
    provider: str = "smart_home"
    route_id: str = ""
    source_entity_id: str = ""
    label: str = ""
    area_label: str = ""
    health_state: str = ""

    def __post_init__(self) -> None:
        """Normalize free-text identifiers and attributes."""

        object.__setattr__(self, "source_event_id", _normalize_text(self.source_event_id))
        object.__setattr__(self, "environment_id", _normalize_text(self.environment_id) or "home:main")
        object.__setattr__(self, "node_id", _normalize_text(self.node_id))
        object.__setattr__(self, "signal_kind", _normalize_text(self.signal_kind))
        object.__setattr__(self, "provider", _normalize_text(self.provider) or "smart_home")
        object.__setattr__(self, "route_id", _normalize_text(self.route_id))
        object.__setattr__(self, "source_entity_id", _normalize_text(self.source_entity_id))
        object.__setattr__(self, "label", _normalize_text(self.label))
        object.__setattr__(self, "area_label", _normalize_text(self.area_label))
        object.__setattr__(self, "health_state", _normalize_text(self.health_state))

    @property
    def local_day(self) -> date:
        """Return the local calendar day of the event."""

        return self.observed_at.date()


@dataclass(frozen=True, slots=True)
class SmartHomeEnvironmentNode:
    """Describe one room-agnostic motion node seen in the environment."""

    environment_id: str
    node_id: str
    provider: str
    source_entity_id: str
    route_id: str
    label: str
    area_label: str
    first_seen_at: datetime
    last_seen_at: datetime
    motion_event_count: int
    active_day_count: int
    last_health_state: str = ""

    def as_memory_object(self, *, event_ids: Sequence[str]) -> LongTermMemoryObjectV1:
        """Render the node summary as one long-term memory summary object."""

        node_token = _tokenize_identifier(self.node_id, fallback="node")
        return LongTermMemoryObjectV1(
            memory_id=f"environment_node:{_tokenize_identifier(self.environment_id, fallback='environment')}:{node_token}",
            kind="summary",
            summary=f"Environment node {node_token} was active in the smart-home profile window.",
            details="Room-agnostic smart-home node summary compiled from motion and health history.",
            source=LongTermSourceRefV1(
                source_type="sensor_memory",
                event_ids=tuple(event_ids[:_MAX_SOURCE_EVENT_IDS]),
                modality="sensor",
            ),
            status="active",
            confidence=0.72,
            sensitivity="low",
            slot_key=f"environment_node:{self.environment_id}:{self.node_id}",
            value_key="environment_node_summary",
            valid_from=self.first_seen_at.date().isoformat(),
            valid_to=self.last_seen_at.date().isoformat(),
            attributes={
                "memory_domain": _SMART_HOME_ENVIRONMENT_DOMAIN,
                "summary_type": "environment_node",
                "environment_id": self.environment_id,
                "node_id": self.node_id,
                "provider": self.provider,
                "source_entity_id": self.source_entity_id,
                "route_id": self.route_id,
                "provider_label": self.label,
                "provider_area_label": self.area_label,
                "first_seen_at": self.first_seen_at.isoformat(),
                "last_seen_at": self.last_seen_at.isoformat(),
                "motion_event_count": self.motion_event_count,
                "active_day_count": self.active_day_count,
                "last_health_state": self.last_health_state or None,
            },
        )


@dataclass(frozen=True, slots=True)
class SmartHomeEnvironmentEpoch:
    """Describe one fixed-width activity epoch."""

    environment_id: str
    epoch_start: datetime
    epoch_width_s: int
    active_node_ids: tuple[str, ...]
    motion_event_count: int
    transition_count: int


@dataclass(frozen=True, slots=True)
class SmartHomeEnvironmentBaselineStat:
    """Describe one rolling baseline stat bundle for a marker."""

    median: float
    iqr: float
    ewma: float
    mad: float
    lower_quantile: float
    upper_quantile: float

    def as_dict(self) -> dict[str, float]:
        """Return the stat bundle as a JSON-safe mapping."""

        return {
            "median": round(self.median, 4),
            "iqr": round(self.iqr, 4),
            "ewma": round(self.ewma, 4),
            "mad": round(self.mad, 4),
            "lower_quantile": round(self.lower_quantile, 4),
            "upper_quantile": round(self.upper_quantile, 4),
        }


@dataclass(frozen=True, slots=True)
class SmartHomeEnvironmentDayProfile:
    """Describe one day-level environment marker vector."""

    environment_id: str
    day: date
    weekday_class: str
    markers: Mapping[str, object]
    quality_flags: tuple[str, ...]
    supporting_ranges: Mapping[str, int]

    def as_memory_object(self, *, event_ids: Sequence[str]) -> LongTermMemoryObjectV1:
        """Render the day profile as one long-term memory summary object."""

        return LongTermMemoryObjectV1(
            memory_id=f"environment_profile:{_tokenize_identifier(self.environment_id, fallback='environment')}:day:{self.day.isoformat()}",
            kind="summary",
            summary="Room-agnostic smart-home environment profile compiled for one day.",
            details="Daily motion-derived smart-home markers for longitudinal routine learning and deviation detection.",
            source=LongTermSourceRefV1(
                source_type="sensor_memory",
                event_ids=tuple(event_ids[:_MAX_SOURCE_EVENT_IDS]),
                modality="sensor",
            ),
            status="active",
            confidence=0.78,
            sensitivity="low",
            slot_key=f"environment_profile:{self.environment_id}:{self.day.isoformat()}",
            value_key="environment_day_profile",
            valid_from=self.day.isoformat(),
            valid_to=self.day.isoformat(),
            attributes={
                "memory_domain": _SMART_HOME_ENVIRONMENT_DOMAIN,
                "summary_type": "environment_day_profile",
                "environment_id": self.environment_id,
                "date": self.day.isoformat(),
                "weekday_class": self.weekday_class,
                "markers": dict(self.markers),
                "quality_flags": list(self.quality_flags),
                "supporting_ranges": dict(self.supporting_ranges),
            },
        )


@dataclass(frozen=True, slots=True)
class SmartHomeEnvironmentBaseline:
    """Describe one rolling baseline built from daily profiles."""

    environment_id: str
    baseline_kind: str
    weekday_class: str
    window_days: int
    sample_count: int
    marker_stats: Mapping[str, SmartHomeEnvironmentBaselineStat]

    def as_memory_object(self, *, valid_from: date, valid_to: date, event_ids: Sequence[str]) -> LongTermMemoryObjectV1:
        """Render the rolling baseline as one long-term memory pattern object."""

        return LongTermMemoryObjectV1(
            memory_id=(
                f"environment_baseline:{_tokenize_identifier(self.environment_id, fallback='environment')}:"
                + (
                    f"{self.weekday_class}:rolling_{self.window_days}d"
                    if self.baseline_kind == "short"
                    else f"{self.baseline_kind}:{self.weekday_class}:rolling_{self.window_days}d"
                )
            ),
            kind="pattern",
            summary=f"Rolling {self.baseline_kind} smart-home environment baseline for {self.weekday_class} days.",
            details="Robust baseline built from prior daily room-agnostic motion markers.",
            source=LongTermSourceRefV1(
                source_type="sensor_memory",
                event_ids=tuple(event_ids[:_MAX_SOURCE_EVENT_IDS]),
                modality="sensor",
            ),
            status="active",
            confidence=min(0.9, 0.42 + (min(self.sample_count, self.window_days) * 0.03)),
            sensitivity="low",
            slot_key=f"environment_baseline:{self.environment_id}:{self.baseline_kind}:{self.weekday_class}",
            value_key="environment_baseline",
            valid_from=valid_from.isoformat(),
            valid_to=valid_to.isoformat(),
            attributes={
                "memory_domain": _SMART_HOME_ENVIRONMENT_DOMAIN,
                "pattern_type": "environment_baseline",
                "environment_id": self.environment_id,
                "baseline_kind": self.baseline_kind,
                "weekday_class": self.weekday_class,
                "window_days": self.window_days,
                "sample_count": self.sample_count,
                "marker_stats": {
                    key: value.as_dict()
                    for key, value in self.marker_stats.items()
                },
            },
        )


@dataclass(frozen=True, slots=True)
class SmartHomeEnvironmentDeviationMetric:
    """Describe one marker that drove a typed deviation."""

    name: str
    observed: float
    baseline_median: float
    delta_ratio: float
    robust_z: float | None = None
    lower_quantile: float | None = None
    upper_quantile: float | None = None

    def as_dict(self) -> dict[str, float | str | None]:
        """Return the metric payload as a JSON-safe mapping."""

        return {
            "name": self.name,
            "observed": round(self.observed, 4),
            "baseline_median": round(self.baseline_median, 4),
            "delta_ratio": round(self.delta_ratio, 4),
            "robust_z": None if self.robust_z is None else round(self.robust_z, 4),
            "lower_quantile": None if self.lower_quantile is None else round(self.lower_quantile, 4),
            "upper_quantile": None if self.upper_quantile is None else round(self.upper_quantile, 4),
        }


@dataclass(frozen=True, slots=True)
class SmartHomeEnvironmentDeviation:
    """Describe one typed environment deviation for the reference day."""

    environment_id: str
    observed_at: datetime
    deviation_type: str
    severity: str
    time_scale: str
    metrics: tuple[SmartHomeEnvironmentDeviationMetric, ...]
    quality_flags: tuple[str, ...]
    blocked_by: tuple[str, ...]
    short_label: str
    human_readable: str

    def as_memory_object(self, *, event_ids: Sequence[str]) -> LongTermMemoryObjectV1:
        """Render the deviation as one long-term memory summary object."""

        return LongTermMemoryObjectV1(
            memory_id=(
                f"environment_deviation:{_tokenize_identifier(self.environment_id, fallback='environment')}:"
                f"{self.deviation_type}:{self.observed_at.date().isoformat()}"
            ),
            kind="summary",
            summary=f"Smart-home environment deviation detected: {self.short_label}.",
            details=self.human_readable,
            source=LongTermSourceRefV1(
                source_type="sensor_memory",
                event_ids=tuple(event_ids[:_MAX_SOURCE_EVENT_IDS]),
                modality="sensor",
            ),
            status="candidate",
            confidence=0.68 if self.severity == "moderate" else 0.78,
            sensitivity="low",
            slot_key=f"environment_deviation:{self.environment_id}:{self.deviation_type}:{self.observed_at.date().isoformat()}",
            value_key=self.deviation_type,
            valid_from=self.observed_at.date().isoformat(),
            valid_to=self.observed_at.date().isoformat(),
            attributes={
                "memory_domain": _SMART_HOME_ENVIRONMENT_DOMAIN,
                "summary_type": "environment_deviation",
                "environment_id": self.environment_id,
                "observed_at": self.observed_at.isoformat(),
                "deviation_type": self.deviation_type,
                "severity": self.severity,
                "time_scale": self.time_scale,
                "markers": [metric.as_dict() for metric in self.metrics],
                "quality_flags": list(self.quality_flags),
                "blocked_by": list(self.blocked_by),
                "explanation": {
                    "short_label": self.short_label,
                    "human_readable": self.human_readable,
                },
            },
        )


@dataclass(frozen=True, slots=True)
class SmartHomeEnvironmentDeviationEvent:
    """Describe one grouped day-level deviation event."""

    environment_id: str
    observed_at: datetime
    classification: str
    severity: str
    time_scale: str
    metrics: tuple[SmartHomeEnvironmentDeviationMetric, ...]
    quality_flags: tuple[str, ...]
    blocked_by: tuple[str, ...]
    short_label: str
    human_readable: str

    def as_memory_object(self, *, event_ids: Sequence[str]) -> LongTermMemoryObjectV1:
        """Render the grouped deviation event as one memory summary."""

        return LongTermMemoryObjectV1(
            memory_id=(
                f"environment_deviation_event:{_tokenize_identifier(self.environment_id, fallback='environment')}:"
                f"{self.classification}:{self.observed_at.date().isoformat()}"
            ),
            kind="summary",
            summary=f"Smart-home environment {self.classification}: {self.short_label}.",
            details=self.human_readable,
            source=LongTermSourceRefV1(
                source_type="sensor_memory",
                event_ids=tuple(event_ids[:_MAX_SOURCE_EVENT_IDS]),
                modality="sensor",
            ),
            status="candidate",
            confidence=0.74 if self.severity == "moderate" else 0.82,
            sensitivity="low",
            slot_key=f"environment_deviation_event:{self.environment_id}:{self.classification}:{self.observed_at.date().isoformat()}",
            value_key=self.classification,
            valid_from=self.observed_at.date().isoformat(),
            valid_to=self.observed_at.date().isoformat(),
            attributes={
                "memory_domain": _SMART_HOME_ENVIRONMENT_DOMAIN,
                "summary_type": "environment_deviation_event",
                "environment_id": self.environment_id,
                "observed_at": self.observed_at.isoformat(),
                "classification": self.classification,
                "severity": self.severity,
                "time_scale": self.time_scale,
                "markers": [metric.as_dict() for metric in self.metrics],
                "quality_flags": list(self.quality_flags),
                "blocked_by": list(self.blocked_by),
                "explanation": {
                    "short_label": self.short_label,
                    "human_readable": self.human_readable,
                },
            },
        )


@dataclass(frozen=True, slots=True)
class SmartHomeEnvironmentQualityState:
    """Describe one bounded interpretation-quality state for the reference day."""

    environment_id: str
    observed_at: datetime
    classification: str
    quality_flags: tuple[str, ...]
    blocked_by: tuple[str, ...]
    evidence_markers: tuple[str, ...]
    human_readable: str

    def as_memory_object(self, *, event_ids: Sequence[str]) -> LongTermMemoryObjectV1:
        """Render the quality state as one summary object."""

        return LongTermMemoryObjectV1(
            memory_id=(
                f"environment_quality_state:{_tokenize_identifier(self.environment_id, fallback='environment')}:"
                f"{self.observed_at.date().isoformat()}"
            ),
            kind="summary",
            summary=f"Smart-home environment quality state for {self.observed_at.date().isoformat()}: {self.classification}.",
            details=self.human_readable,
            source=LongTermSourceRefV1(
                source_type="sensor_memory",
                event_ids=tuple(event_ids[:_MAX_SOURCE_EVENT_IDS]),
                modality="sensor",
            ),
            status="active",
            confidence=0.84,
            sensitivity="low",
            slot_key=f"environment_quality_state:{self.environment_id}",
            value_key=self.observed_at.date().isoformat(),
            valid_from=self.observed_at.date().isoformat(),
            valid_to=self.observed_at.date().isoformat(),
            attributes={
                "memory_domain": _SMART_HOME_ENVIRONMENT_DOMAIN,
                "summary_type": "environment_quality_state",
                "environment_id": self.environment_id,
                "observed_at": self.observed_at.isoformat(),
                "classification": self.classification,
                "quality_flags": list(self.quality_flags),
                "blocked_by": list(self.blocked_by),
                "evidence_markers": list(self.evidence_markers),
            },
        )


@dataclass(frozen=True, slots=True)
class SmartHomeEnvironmentChangePoint:
    """Describe one transition into a changed behavior regime."""

    environment_id: str
    observed_at: datetime
    change_started_on: date
    severity: str
    metrics: tuple[SmartHomeEnvironmentDeviationMetric, ...]
    quality_flags: tuple[str, ...]
    blocked_by: tuple[str, ...]
    human_readable: str

    def as_memory_object(self, *, event_ids: Sequence[str]) -> LongTermMemoryObjectV1:
        """Render the change-point summary as one memory object."""

        return LongTermMemoryObjectV1(
            memory_id=(
                f"environment_change_point:{_tokenize_identifier(self.environment_id, fallback='environment')}:"
                f"{self.observed_at.date().isoformat()}"
            ),
            kind="summary",
            summary="Smart-home environment transition detected.",
            details=self.human_readable,
            source=LongTermSourceRefV1(
                source_type="sensor_memory",
                event_ids=tuple(event_ids[:_MAX_SOURCE_EVENT_IDS]),
                modality="sensor",
            ),
            status="candidate",
            confidence=0.8 if self.severity == "moderate" else 0.86,
            sensitivity="low",
            slot_key=f"environment_change_point:{self.environment_id}",
            value_key=self.observed_at.date().isoformat(),
            valid_from=self.change_started_on.isoformat(),
            valid_to=self.observed_at.date().isoformat(),
            attributes={
                "memory_domain": _SMART_HOME_ENVIRONMENT_DOMAIN,
                "summary_type": "environment_change_point",
                "environment_id": self.environment_id,
                "observed_at": self.observed_at.isoformat(),
                "change_started_on": self.change_started_on.isoformat(),
                "severity": self.severity,
                "markers": [metric.as_dict() for metric in self.metrics],
                "quality_flags": list(self.quality_flags),
                "blocked_by": list(self.blocked_by),
            },
        )


@dataclass(frozen=True, slots=True)
class SmartHomeEnvironmentRegime:
    """Describe one accepted new normal for the environment."""

    environment_id: str
    valid_from_day: date
    observed_at: datetime
    severity: str
    metrics: tuple[SmartHomeEnvironmentDeviationMetric, ...]
    quality_flags: tuple[str, ...]
    human_readable: str

    def as_memory_object(self, *, event_ids: Sequence[str]) -> LongTermMemoryObjectV1:
        """Render the accepted regime as one pattern object."""

        return LongTermMemoryObjectV1(
            memory_id=(
                f"environment_regime:{_tokenize_identifier(self.environment_id, fallback='environment')}:"
                f"{self.valid_from_day.isoformat()}"
            ),
            kind="pattern",
            summary="Accepted smart-home environment regime shift.",
            details=self.human_readable,
            source=LongTermSourceRefV1(
                source_type="sensor_memory",
                event_ids=tuple(event_ids[:_MAX_SOURCE_EVENT_IDS]),
                modality="sensor",
            ),
            status="active",
            confidence=0.82 if self.severity == "moderate" else 0.88,
            sensitivity="low",
            slot_key=f"environment_regime:{self.environment_id}",
            value_key=self.valid_from_day.isoformat(),
            valid_from=self.valid_from_day.isoformat(),
            valid_to=self.observed_at.date().isoformat(),
            attributes={
                "memory_domain": _SMART_HOME_ENVIRONMENT_DOMAIN,
                "pattern_type": "environment_regime",
                "environment_id": self.environment_id,
                "observed_at": self.observed_at.isoformat(),
                "regime_started_on": self.valid_from_day.isoformat(),
                "severity": self.severity,
                "markers": [metric.as_dict() for metric in self.metrics],
                "quality_flags": list(self.quality_flags),
            },
        )
