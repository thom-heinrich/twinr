"""Compile room-agnostic smart-home environment profiles from motion history.

This module turns day-scoped smart-home motion and device-health pattern seeds
into a compact set of longitudinal environment-profile objects:

- node summaries
- daily marker profiles
- rolling baselines
- typed deviations

The compiler is intentionally room-agnostic. Provider labels may be carried as
metadata, but behavior logic is driven by stable node identifiers, temporal
epochs, and transition structure rather than room names.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, tzinfo
from hashlib import sha256
import logging
import math
from statistics import median
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.core.models import LongTermMemoryObjectV1, LongTermReflectionResultV1, LongTermSourceRefV1


logger = logging.getLogger(__name__)

_DEFAULT_TIMEZONE_NAME = "Europe/Berlin"
_DEFAULT_BASELINE_DAYS = 14
_DEFAULT_HISTORY_DAYS = 42
_DEFAULT_MIN_BASELINE_DAYS = 5
_DEFAULT_EPOCH_MINUTES = 5
_DEFAULT_TRANSITION_WINDOW_S = 90.0
_DEFAULT_DAY_START_HOUR = 6
_DEFAULT_NIGHT_START_HOUR = 22
_DEFAULT_IQR_MULTIPLIER = 1.5
_MAX_SOURCE_EVENT_IDS = 32
_SMART_HOME_ENVIRONMENT_DOMAIN = "smart_home_environment"
_MOTION_SIGNAL_TYPE = "motion_node_activity"
_HEALTH_SIGNAL_TYPE = "node_health"
_WEEKDAY_CLASSES = ("all_days", "weekday", "weekend")


def _normalize_text(value: object | None) -> str:
    """Collapse arbitrary input into one bounded line of text."""

    return " ".join(str(value or "").split()).strip()


def _normalize_slug(value: object | None, *, fallback: str) -> str:
    """Return one storage-safe token for identifiers and memory IDs."""

    normalized = _normalize_text(value).lower()
    if not normalized:
        return fallback
    slug_chars = [
        character if character.isalnum() else "_"
        for character in normalized
    ]
    slug = "".join(slug_chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or fallback


def _tokenize_identifier(value: str, *, fallback: str) -> str:
    """Return one compact token safe for use inside long-term memory IDs."""

    slug = _normalize_slug(value, fallback=fallback)
    if len(slug) <= 48:
        return slug
    digest = sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{fallback}_{digest}"


def _coerce_mapping(value: object) -> dict[str, object]:
    """Coerce one mapping-like value into a plain dictionary."""

    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _coerce_float(value: object, *, default: float) -> float:
    """Coerce one numeric input to float with fallback."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _weekday_class(value: date) -> str:
    """Return the shared weekday bucket for one date."""

    return "weekend" if value.weekday() >= 5 else "weekday"


def _resolve_timezone(name: str) -> tzinfo:
    """Resolve one timezone name with bounded fallback behavior."""

    normalized = _normalize_text(name) or _DEFAULT_TIMEZONE_NAME
    try:
        return ZoneInfo(normalized)
    except (ValueError, ZoneInfoNotFoundError):
        logger.warning("Falling back from invalid timezone %r to %r.", normalized, _DEFAULT_TIMEZONE_NAME)
        try:
            return ZoneInfo(_DEFAULT_TIMEZONE_NAME)
        except (ValueError, ZoneInfoNotFoundError):
            return timezone.utc


def _normalize_datetime(value: datetime | None, *, timezone: tzinfo) -> datetime | None:
    """Normalize one datetime into the configured timezone."""

    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone)
    return value.astimezone(timezone)


def _parse_source_event_datetime(event_id: str) -> datetime | None:
    """Extract the timestamp component from one synthetic source event ID."""

    clean = _normalize_text(event_id)
    if not clean:
        return None
    parts = clean.split(":", 2)
    if len(parts) < 2:
        return None
    timestamp = parts[1]
    for pattern in ("%Y%m%dT%H%M%S%f%z", "%Y%m%dT%H%M%S%z"):
        try:
            return datetime.strptime(timestamp, pattern)
        except ValueError:
            continue
    return None


def _quantile(values: Sequence[float], q: float) -> float:
    """Return one bounded linear-interpolated quantile."""

    if not values:
        raise ValueError("values must not be empty.")
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(value) for value in values)
    position = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] + ((ordered[upper_index] - ordered[lower_index]) * weight)


def _iqr(values: Sequence[float]) -> float:
    """Return the interquartile range for one non-empty sample."""

    return _quantile(values, 0.75) - _quantile(values, 0.25)


def _ewma(values: Sequence[float], *, alpha: float = 0.35) -> float:
    """Return one simple exponentially weighted moving average."""

    if not values:
        raise ValueError("values must not be empty.")
    result = float(values[0])
    for value in values[1:]:
        result = (alpha * float(value)) + ((1.0 - alpha) * result)
    return result


def _entropy_from_counts(counts: Mapping[object, int]) -> float:
    """Return the Shannon entropy for one count distribution."""

    total = sum(int(value) for value in counts.values() if int(value) > 0)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        count = int(value)
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Return the cosine similarity between two equal-length vectors."""

    if len(left) != len(right) or not left:
        return None
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    return dot / (left_norm * right_norm)


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

    def as_dict(self) -> dict[str, float]:
        """Return the stat bundle as a JSON-safe mapping."""

        return {
            "median": round(self.median, 4),
            "iqr": round(self.iqr, 4),
            "ewma": round(self.ewma, 4),
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
    weekday_class: str
    window_days: int
    sample_count: int
    marker_stats: Mapping[str, SmartHomeEnvironmentBaselineStat]

    def as_memory_object(self, *, valid_from: date, valid_to: date, event_ids: Sequence[str]) -> LongTermMemoryObjectV1:
        """Render the rolling baseline as one long-term memory pattern object."""

        return LongTermMemoryObjectV1(
            memory_id=(
                f"environment_baseline:{_tokenize_identifier(self.environment_id, fallback='environment')}:"
                f"{self.weekday_class}:rolling_{self.window_days}d"
            ),
            kind="pattern",
            summary=f"Rolling smart-home environment baseline for {self.weekday_class} days.",
            details="Robust baseline built from prior daily room-agnostic motion markers.",
            source=LongTermSourceRefV1(
                source_type="sensor_memory",
                event_ids=tuple(event_ids[:_MAX_SOURCE_EVENT_IDS]),
                modality="sensor",
            ),
            status="active",
            confidence=min(0.9, 0.42 + (min(self.sample_count, self.window_days) * 0.03)),
            sensitivity="low",
            slot_key=f"environment_baseline:{self.environment_id}:{self.weekday_class}",
            value_key="environment_baseline",
            valid_from=valid_from.isoformat(),
            valid_to=valid_to.isoformat(),
            attributes={
                "memory_domain": _SMART_HOME_ENVIRONMENT_DOMAIN,
                "pattern_type": "environment_baseline",
                "environment_id": self.environment_id,
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

    def as_dict(self) -> dict[str, float | str]:
        """Return the metric payload as a JSON-safe mapping."""

        return {
            "name": self.name,
            "observed": round(self.observed, 4),
            "baseline_median": round(self.baseline_median, 4),
            "delta_ratio": round(self.delta_ratio, 4),
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
class LongTermEnvironmentProfileCompiler:
    """Compile room-agnostic smart-home environment markers and deviations."""

    timezone_name: str = _DEFAULT_TIMEZONE_NAME
    enabled: bool = False
    environment_id: str = "home:main"
    baseline_days: int = _DEFAULT_BASELINE_DAYS
    history_days: int = _DEFAULT_HISTORY_DAYS
    min_baseline_days: int = _DEFAULT_MIN_BASELINE_DAYS
    epoch_minutes: int = _DEFAULT_EPOCH_MINUTES
    transition_window_s: float = _DEFAULT_TRANSITION_WINDOW_S
    day_start_hour: int = _DEFAULT_DAY_START_HOUR
    night_start_hour: int = _DEFAULT_NIGHT_START_HOUR
    iqr_multiplier: float = _DEFAULT_IQR_MULTIPLIER

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "LongTermEnvironmentProfileCompiler":
        """Build the compiler from Twinr config using the existing sensor-memory gate."""

        baseline_days = max(7, int(getattr(config, "long_term_memory_sensor_baseline_days", _DEFAULT_BASELINE_DAYS) or _DEFAULT_BASELINE_DAYS))
        min_days = max(3, int(getattr(config, "long_term_memory_sensor_min_days_observed", _DEFAULT_MIN_BASELINE_DAYS) or _DEFAULT_MIN_BASELINE_DAYS))
        return cls(
            timezone_name=getattr(config, "local_timezone_name", _DEFAULT_TIMEZONE_NAME),
            enabled=bool(getattr(config, "long_term_memory_sensor_memory_enabled", False)),
            baseline_days=baseline_days,
            history_days=max(_DEFAULT_HISTORY_DAYS, baseline_days * 3),
            min_baseline_days=min_days,
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
            timezone = _resolve_timezone(self.timezone_name)
            reference = _normalize_datetime(now or datetime.now(timezone.utc), timezone=timezone) or datetime.now(timezone.utc)
            reference_day = reference.date()
            events = self._extract_events(objects=objects, timezone=timezone)
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

            nodes = self._build_nodes(events=relevant_events)
            day_profiles, day_event_ids = self._build_day_profiles(events=relevant_events)
            baselines, baseline_event_ids = self._build_baselines(
                day_profiles=day_profiles,
                reference_day=reference_day,
                day_event_ids=day_event_ids,
            )
            deviations = self._build_deviations(
                reference=reference,
                reference_day=reference_day,
                day_profiles=day_profiles,
                baselines=baselines,
                day_event_ids=day_event_ids,
            )

            created: list[LongTermMemoryObjectV1] = []
            for node in nodes:
                created.append(
                    node.as_memory_object(
                        event_ids=self._node_event_ids(node_id=node.node_id, events=relevant_events),
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
            for deviation in deviations:
                created.append(
                    deviation.as_memory_object(
                        event_ids=day_event_ids.get(reference_day, ()),
                    )
                )
            return LongTermReflectionResultV1(reflected_objects=(), created_summaries=tuple(created))
        except Exception:
            logger.exception("Failed to compile smart-home environment profiles.")
            return empty

    def _extract_events(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
        timezone: tzinfo,
    ) -> tuple[SmartHomeEnvironmentEvent, ...]:
        """Extract normalized environment events from long-term pattern objects."""

        extracted: list[SmartHomeEnvironmentEvent] = []
        seen_source_event_ids: set[str] = set()
        for item in tuple(objects or ()):
            canonical = item.canonicalized()
            if canonical.kind != "pattern" or canonical.status not in {"active", "candidate", "uncertain"}:
                continue
            attrs = _coerce_mapping(canonical.attributes)
            if _normalize_text(attrs.get("memory_domain")).lower() != _SMART_HOME_ENVIRONMENT_DOMAIN:
                continue
            signal_type = _normalize_text(attrs.get("environment_signal_type")).lower()
            if signal_type not in {_MOTION_SIGNAL_TYPE, _HEALTH_SIGNAL_TYPE}:
                continue
            node_id = _normalize_text(attrs.get("node_id"))
            if not node_id:
                continue
            source = getattr(canonical, "source", None)
            raw_event_ids = tuple(getattr(source, "event_ids", ()) or ())
            for raw_event_id in raw_event_ids:
                source_event_id = _normalize_text(raw_event_id)
                if not source_event_id or source_event_id in seen_source_event_ids:
                    continue
                parsed_at = _parse_source_event_datetime(source_event_id)
                if parsed_at is None:
                    continue
                local_observed_at = _normalize_datetime(parsed_at, timezone=timezone)
                if local_observed_at is None:
                    continue
                signal_kind = "motion_detected" if signal_type == _MOTION_SIGNAL_TYPE else "device_health"
                extracted.append(
                    SmartHomeEnvironmentEvent(
                        source_event_id=source_event_id,
                        environment_id=_normalize_text(attrs.get("environment_id")) or self.environment_id,
                        node_id=node_id,
                        observed_at=local_observed_at,
                        signal_kind=signal_kind,
                        provider=_normalize_text(attrs.get("provider")) or "smart_home",
                        route_id=_normalize_text(attrs.get("route_id")),
                        source_entity_id=_normalize_text(attrs.get("source_entity_id")),
                        label=_normalize_text(attrs.get("provider_label")),
                        area_label=_normalize_text(attrs.get("provider_area_label")),
                        health_state=_normalize_text(attrs.get("health_state")).lower(),
                    )
                )
                seen_source_event_ids.add(source_event_id)
        extracted.sort(key=lambda item: (item.observed_at, item.source_event_id))
        return tuple(extracted)

    def _build_nodes(
        self,
        *,
        events: tuple[SmartHomeEnvironmentEvent, ...],
    ) -> tuple[SmartHomeEnvironmentNode, ...]:
        """Build one summary object per observed node."""

        events_by_node: dict[str, list[SmartHomeEnvironmentEvent]] = defaultdict(list)
        for event in events:
            events_by_node[event.node_id].append(event)

        created: list[SmartHomeEnvironmentNode] = []
        for node_id, node_events in events_by_node.items():
            motion_events = [event for event in node_events if event.signal_kind == "motion_detected"]
            if not motion_events:
                continue
            first_event = node_events[0]
            health_events = [event for event in node_events if event.signal_kind == "device_health" and event.health_state]
            created.append(
                SmartHomeEnvironmentNode(
                    environment_id=first_event.environment_id,
                    node_id=node_id,
                    provider=first_event.provider,
                    source_entity_id=first_event.source_entity_id or node_id,
                    route_id=first_event.route_id,
                    label=first_event.label,
                    area_label=first_event.area_label,
                    first_seen_at=motion_events[0].observed_at,
                    last_seen_at=motion_events[-1].observed_at,
                    motion_event_count=len(motion_events),
                    active_day_count=len({event.local_day for event in motion_events}),
                    last_health_state=health_events[-1].health_state if health_events else "",
                )
            )
        created.sort(key=lambda item: (item.provider, item.node_id))
        return tuple(created)

    def _build_day_profiles(
        self,
        *,
        events: tuple[SmartHomeEnvironmentEvent, ...],
    ) -> tuple[dict[date, SmartHomeEnvironmentDayProfile], dict[date, tuple[str, ...]]]:
        """Build day profiles and supporting source event IDs for each day."""

        motion_by_day: dict[date, list[SmartHomeEnvironmentEvent]] = defaultdict(list)
        health_by_day: dict[date, list[SmartHomeEnvironmentEvent]] = defaultdict(list)
        all_event_ids_by_day: dict[date, list[str]] = defaultdict(list)
        for event in events:
            all_event_ids_by_day[event.local_day].append(event.source_event_id)
            if event.signal_kind == "motion_detected":
                motion_by_day[event.local_day].append(event)
            elif event.signal_kind == "device_health":
                health_by_day[event.local_day].append(event)

        profiles: dict[date, SmartHomeEnvironmentDayProfile] = {}
        event_ids: dict[date, tuple[str, ...]] = {}
        known_nodes = {event.node_id for event in events if event.signal_kind == "motion_detected"}
        for day, motion_events in motion_by_day.items():
            motion_events.sort(key=lambda item: (item.observed_at, item.source_event_id))
            epochs, hourly_counts = self._build_epochs(motion_events=motion_events)
            health_events = tuple(sorted(health_by_day.get(day, ()), key=lambda item: (item.observed_at, item.source_event_id)))
            markers = self._build_day_markers(
                day=day,
                motion_events=tuple(motion_events),
                epochs=epochs,
                hourly_counts=hourly_counts,
                health_events=health_events,
                known_nodes=known_nodes,
            )
            quality_flags = self._day_quality_flags(health_events=health_events)
            profiles[day] = SmartHomeEnvironmentDayProfile(
                environment_id=self.environment_id,
                day=day,
                weekday_class=_weekday_class(day),
                markers=markers,
                quality_flags=quality_flags,
                supporting_ranges={
                    "day_start_hour": self.day_start_hour,
                    "night_start_hour": self.night_start_hour,
                },
            )
            deduped_event_ids = list(dict.fromkeys(all_event_ids_by_day.get(day, ())))
            event_ids[day] = tuple(deduped_event_ids[-_MAX_SOURCE_EVENT_IDS:])
        return profiles, event_ids

    def _build_epochs(
        self,
        *,
        motion_events: Sequence[SmartHomeEnvironmentEvent],
    ) -> tuple[tuple[SmartHomeEnvironmentEpoch, ...], list[int]]:
        """Compile fixed-width epochs and hourly event counts for one day."""

        epoch_width_s = self.epoch_minutes * 60
        epochs_by_index: dict[int, set[str]] = defaultdict(set)
        motion_event_counts: Counter[int] = Counter()
        hourly_counts = [0] * 24
        for event in motion_events:
            minute_of_day = (event.observed_at.hour * 60) + event.observed_at.minute
            epoch_index = minute_of_day // self.epoch_minutes
            epochs_by_index[epoch_index].add(event.node_id)
            motion_event_counts[epoch_index] += 1
            hourly_counts[event.observed_at.hour] += 1

        transitions_by_index = Counter()
        for event in self._transition_edges(motion_events):
            epoch_index = ((event[2].hour * 60) + event[2].minute) // self.epoch_minutes
            transitions_by_index[epoch_index] += 1

        first_day = motion_events[0].local_day
        created: list[SmartHomeEnvironmentEpoch] = []
        total_epochs = (24 * 60) // self.epoch_minutes
        for epoch_index in range(total_epochs):
            if epoch_index not in epochs_by_index:
                continue
            epoch_start = datetime.combine(first_day, datetime.min.time(), tzinfo=motion_events[0].observed_at.tzinfo)
            epoch_start += timedelta(minutes=epoch_index * self.epoch_minutes)
            created.append(
                SmartHomeEnvironmentEpoch(
                    environment_id=self.environment_id,
                    epoch_start=epoch_start,
                    epoch_width_s=epoch_width_s,
                    active_node_ids=tuple(sorted(epochs_by_index[epoch_index])),
                    motion_event_count=motion_event_counts[epoch_index],
                    transition_count=transitions_by_index[epoch_index],
                )
            )
        return tuple(created), hourly_counts

    def _build_day_markers(
        self,
        *,
        day: date,
        motion_events: tuple[SmartHomeEnvironmentEvent, ...],
        epochs: tuple[SmartHomeEnvironmentEpoch, ...],
        hourly_counts: list[int],
        health_events: tuple[SmartHomeEnvironmentEvent, ...],
        known_nodes: set[str],
    ) -> dict[str, float | int | None]:
        """Compute the marker vector for one day."""

        total_epochs = (24 * 60) // self.epoch_minutes
        active_epoch_indices = {
            ((epoch.epoch_start.hour * 60) + epoch.epoch_start.minute) // self.epoch_minutes
            for epoch in epochs
        }
        node_counts = Counter(event.node_id for event in motion_events)
        transition_edges = self._transition_edges(motion_events)
        transition_counts = Counter((source, target) for source, target, _ in transition_edges)
        active_flags = [index in active_epoch_indices for index in range(total_epochs)]
        active_epoch_count = len(active_epoch_indices)
        first_activity_minute = None if not motion_events else (motion_events[0].observed_at.hour * 60) + motion_events[0].observed_at.minute
        last_activity_minute = None if not motion_events else (motion_events[-1].observed_at.hour * 60) + motion_events[-1].observed_at.minute
        day_start_epoch = (self.day_start_hour * 60) // self.epoch_minutes
        night_start_epoch = (self.night_start_hour * 60) // self.epoch_minutes
        night_activity_epoch_count = sum(
            1
            for index in active_epoch_indices
            if index < day_start_epoch or index >= night_start_epoch
        )
        longest_daytime_inactivity_epochs = 0
        current_inactivity_epochs = 0
        for index in range(day_start_epoch, night_start_epoch):
            if active_flags[index]:
                longest_daytime_inactivity_epochs = max(longest_daytime_inactivity_epochs, current_inactivity_epochs)
                current_inactivity_epochs = 0
            else:
                current_inactivity_epochs += 1
        longest_daytime_inactivity_epochs = max(longest_daytime_inactivity_epochs, current_inactivity_epochs)

        active_followed_count = 0
        active_to_inactive_count = 0
        motion_burst_count = 0
        for index, active in enumerate(active_flags):
            if active and (index == 0 or not active_flags[index - 1]):
                motion_burst_count += 1
            if active and index + 1 < len(active_flags):
                active_followed_count += 1
                if not active_flags[index + 1]:
                    active_to_inactive_count += 1

        transition_count = sum(transition_counts.values())
        mean_active_node_count = (
            sum(len(epoch.active_node_ids) for epoch in epochs) / len(epochs)
            if epochs
            else 0.0
        )

        health_by_node: dict[str, str] = {}
        for event in health_events:
            if event.health_state:
                health_by_node[event.node_id] = event.health_state
        offline_nodes = {node_id for node_id, state in health_by_node.items() if state == "offline"}
        coverage_denominator = len(known_nodes) if known_nodes else len(node_counts)
        sensor_coverage_ratio = None
        if coverage_denominator > 0:
            sensor_coverage_ratio = max(0.0, (coverage_denominator - len(offline_nodes)) / coverage_denominator)

        return {
            "active_epoch_count_day": active_epoch_count,
            "first_activity_minute_local": first_activity_minute,
            "last_activity_minute_local": last_activity_minute,
            "longest_daytime_inactivity_min": longest_daytime_inactivity_epochs * self.epoch_minutes,
            "night_activity_epoch_count": night_activity_epoch_count,
            "unique_active_node_count_day": len(node_counts),
            "mean_active_node_count_per_active_epoch": round(mean_active_node_count, 4),
            "node_entropy_day": round(_entropy_from_counts(node_counts), 4),
            "dominant_node_share_day": round((max(node_counts.values()) / sum(node_counts.values())) if node_counts else 0.0, 4),
            "transition_count_day": transition_count,
            "transition_entropy_day": round(_entropy_from_counts(transition_counts), 4),
            "fragmentation_index_day": round((active_to_inactive_count / active_followed_count) if active_followed_count else 0.0, 4),
            "motion_burst_count_day": motion_burst_count,
            "circadian_similarity_14d": None,
            "sensor_coverage_ratio_day": None if sensor_coverage_ratio is None else round(sensor_coverage_ratio, 4),
            "hourly_activity_vector": tuple(hourly_counts),
            "profile_day": day.isoformat(),
        }

    def _day_quality_flags(
        self,
        *,
        health_events: tuple[SmartHomeEnvironmentEvent, ...],
    ) -> tuple[str, ...]:
        """Return quality flags for one day profile."""

        flags: list[str] = []
        if not health_events:
            flags.append("sensor_health_unknown")
        if any(event.health_state == "offline" for event in health_events):
            flags.append("device_offline_present")
        return tuple(flags)

    def _build_baselines(
        self,
        *,
        day_profiles: Mapping[date, SmartHomeEnvironmentDayProfile],
        reference_day: date,
        day_event_ids: Mapping[date, tuple[str, ...]],
    ) -> tuple[dict[str, SmartHomeEnvironmentBaseline], dict[str, tuple[str, ...]]]:
        """Build rolling baselines from prior daily profiles."""

        prior_days = sorted(day for day in day_profiles if day < reference_day)
        baselines: dict[str, SmartHomeEnvironmentBaseline] = {}
        baseline_event_ids: dict[str, tuple[str, ...]] = {}
        for weekday_class in _WEEKDAY_CLASSES:
            eligible_days = [
                day
                for day in prior_days
                if (reference_day - day).days <= self.baseline_days
                and (weekday_class == "all_days" or _weekday_class(day) == weekday_class)
            ]
            if len(eligible_days) < self.min_baseline_days:
                continue
            profiles = [day_profiles[day] for day in eligible_days]
            marker_names = [
                "active_epoch_count_day",
                "first_activity_minute_local",
                "last_activity_minute_local",
                "longest_daytime_inactivity_min",
                "night_activity_epoch_count",
                "unique_active_node_count_day",
                "transition_count_day",
                "fragmentation_index_day",
                "sensor_coverage_ratio_day",
            ]
            marker_stats: dict[str, SmartHomeEnvironmentBaselineStat] = {}
            for marker_name in marker_names:
                values = [
                    float(profile.markers[marker_name])
                    for profile in profiles
                    if profile.markers.get(marker_name) is not None
                ]
                if len(values) < self.min_baseline_days:
                    continue
                marker_stats[marker_name] = SmartHomeEnvironmentBaselineStat(
                    median=float(median(values)),
                    iqr=float(_iqr(values)),
                    ewma=float(_ewma(values)),
                )
            if not marker_stats:
                continue
            baselines[weekday_class] = SmartHomeEnvironmentBaseline(
                environment_id=self.environment_id,
                weekday_class=weekday_class,
                window_days=self.baseline_days,
                sample_count=len(eligible_days),
                marker_stats=marker_stats,
            )
            collected_event_ids: list[str] = []
            for day in eligible_days:
                collected_event_ids.extend(day_event_ids.get(day, ()))
            baseline_event_ids[weekday_class] = tuple(list(dict.fromkeys(collected_event_ids))[-_MAX_SOURCE_EVENT_IDS:])

        reference_profile = day_profiles.get(reference_day)
        if reference_profile is not None:
            today_vector = reference_profile.markers.get("hourly_activity_vector")
            if isinstance(today_vector, tuple):
                today_values = [float(value) for value in today_vector]
                baseline = baselines.get(reference_profile.weekday_class) or baselines.get("all_days")
                if baseline is not None:
                    comparison_days = [
                        day
                        for day in prior_days
                        if (reference_day - day).days <= self.baseline_days
                        and (baseline.weekday_class == "all_days" or _weekday_class(day) == baseline.weekday_class)
                    ]
                    prior_vectors = [
                        tuple(float(value) for value in day_profiles[day].markers.get("hourly_activity_vector", ()))
                        for day in comparison_days
                        if isinstance(day_profiles[day].markers.get("hourly_activity_vector"), tuple)
                    ]
                    if prior_vectors and len(prior_vectors[0]) == len(today_values):
                        baseline_vector = [
                            float(median([vector[index] for vector in prior_vectors]))
                            for index in range(len(today_values))
                        ]
                        similarity = _cosine_similarity(today_values, baseline_vector)
                        if similarity is not None:
                            updated_markers = dict(reference_profile.markers)
                            updated_markers["circadian_similarity_14d"] = round(similarity, 4)
                            day_profiles[reference_day] = SmartHomeEnvironmentDayProfile(
                                environment_id=reference_profile.environment_id,
                                day=reference_profile.day,
                                weekday_class=reference_profile.weekday_class,
                                markers=updated_markers,
                                quality_flags=reference_profile.quality_flags,
                                supporting_ranges=reference_profile.supporting_ranges,
                            )
        return baselines, baseline_event_ids

    def _build_deviations(
        self,
        *,
        reference: datetime,
        reference_day: date,
        day_profiles: Mapping[date, SmartHomeEnvironmentDayProfile],
        baselines: Mapping[str, SmartHomeEnvironmentBaseline],
        day_event_ids: Mapping[date, tuple[str, ...]],
    ) -> tuple[SmartHomeEnvironmentDeviation, ...]:
        """Build typed deviations for the reference day against the rolling baseline."""

        profile = day_profiles.get(reference_day)
        if profile is None:
            return ()
        baseline = baselines.get(profile.weekday_class) or baselines.get("all_days")
        if baseline is None:
            return ()

        deviations: list[SmartHomeEnvironmentDeviation] = []
        blocked_by = ("sensor_quality_limited",) if "device_offline_present" in profile.quality_flags else ()
        deviation_specs = (
            ("daily_activity_drop", "active_epoch_count_day", "low", "less activity than usual"),
            ("night_activity_increase", "night_activity_epoch_count", "high", "more night activity than usual"),
            ("late_start_of_day", "first_activity_minute_local", "high", "later start of day than usual"),
            ("early_end_of_day", "last_activity_minute_local", "low", "earlier end of day than usual"),
            ("fragmentation_shift", "fragmentation_index_day", "high", "more fragmented movement than usual"),
            ("possible_sensor_failure", "sensor_coverage_ratio_day", "low", "lower sensor coverage than expected"),
        )
        for deviation_type, marker_name, direction, short_label in deviation_specs:
            metric = self._deviation_metric(
                profile=profile,
                baseline=baseline,
                marker_name=marker_name,
                direction=direction,
            )
            if metric is None:
                continue
            delta_ratio = abs(metric.delta_ratio)
            severity = "high" if delta_ratio >= 0.35 else "moderate"
            deviations.append(
                SmartHomeEnvironmentDeviation(
                    environment_id=self.environment_id,
                    observed_at=reference,
                    deviation_type=deviation_type,
                    severity=severity,
                    time_scale="day",
                    metrics=(metric,),
                    quality_flags=profile.quality_flags,
                    blocked_by=blocked_by,
                    short_label=short_label,
                    human_readable=(
                        "Observed marker drift against the rolling room-agnostic smart-home baseline. "
                        f"Marker {marker_name} moved in the {direction} direction for {profile.day.isoformat()}."
                    ),
                )
            )
        return tuple(deviations)

    def _deviation_metric(
        self,
        *,
        profile: SmartHomeEnvironmentDayProfile,
        baseline: SmartHomeEnvironmentBaseline,
        marker_name: str,
        direction: str,
    ) -> SmartHomeEnvironmentDeviationMetric | None:
        """Return one deviation metric when a marker moved materially."""

        observed_raw = profile.markers.get(marker_name)
        if observed_raw is None:
            return None
        stats = baseline.marker_stats.get(marker_name)
        if stats is None:
            return None
        observed = float(observed_raw)
        spread = max(stats.iqr * self.iqr_multiplier, 1.0 if marker_name.endswith("_count_day") else 0.05)
        delta = observed - stats.median
        if direction == "high" and delta <= spread:
            return None
        if direction == "low" and delta >= -spread:
            return None
        denominator = max(abs(stats.median), 1.0)
        return SmartHomeEnvironmentDeviationMetric(
            name=marker_name,
            observed=observed,
            baseline_median=stats.median,
            delta_ratio=delta / denominator,
        )

    def _node_event_ids(
        self,
        *,
        node_id: str,
        events: Sequence[SmartHomeEnvironmentEvent],
    ) -> tuple[str, ...]:
        """Return bounded supporting event IDs for one node summary."""

        event_ids = [
            event.source_event_id
            for event in events
            if event.node_id == node_id
        ]
        return tuple(list(dict.fromkeys(event_ids))[-_MAX_SOURCE_EVENT_IDS:])

    def _transition_edges(
        self,
        motion_events: Sequence[SmartHomeEnvironmentEvent],
    ) -> tuple[tuple[str, str, datetime], ...]:
        """Return bounded node-to-node transition edges from ordered events."""

        if not motion_events:
            return ()
        ordered = sorted(motion_events, key=lambda item: (item.observed_at, item.source_event_id))
        previous = ordered[0]
        edges: list[tuple[str, str, datetime]] = []
        for current in ordered[1:]:
            delta_s = (current.observed_at - previous.observed_at).total_seconds()
            if current.node_id != previous.node_id and 0.0 <= delta_s <= self.transition_window_s:
                edges.append((previous.node_id, current.node_id, current.observed_at))
            previous = current
        return tuple(edges)


__all__ = [
    "LongTermEnvironmentProfileCompiler",
    "SmartHomeEnvironmentBaseline",
    "SmartHomeEnvironmentBaselineStat",
    "SmartHomeEnvironmentDayProfile",
    "SmartHomeEnvironmentDeviation",
    "SmartHomeEnvironmentDeviationMetric",
    "SmartHomeEnvironmentEpoch",
    "SmartHomeEnvironmentEvent",
    "SmartHomeEnvironmentNode",
]
