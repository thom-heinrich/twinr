"""Event extraction and day/baseline builders for environment profiling."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta, tzinfo
from statistics import median
from typing import TYPE_CHECKING, cast

from twinr.memory.longterm.core.models import LongTermMemoryObjectV1
from twinr.memory.longterm.ingestion.environment_stats import iqr, jensen_shannon_divergence, mad, quantile

from .constants import (
    _BASELINE_MARKER_NAMES,
    _HEALTH_SIGNAL_TYPE,
    _MAX_SOURCE_EVENT_IDS,
    _MOTION_SIGNAL_TYPE,
    _SMART_HOME_ENVIRONMENT_DOMAIN,
    _WEEKDAY_CLASSES,
)
from .helpers import (
    _coerce_mapping,
    _cosine_similarity,
    _entropy_from_counts,
    _ewma,
    _normalize_datetime,
    _normalize_text,
    _parse_source_event_datetime,
    _weekday_class,
)
from .models import (
    SmartHomeEnvironmentBaseline,
    SmartHomeEnvironmentBaselineStat,
    SmartHomeEnvironmentDayProfile,
    SmartHomeEnvironmentEpoch,
    SmartHomeEnvironmentEvent,
    SmartHomeEnvironmentNode,
)

if TYPE_CHECKING:
    from .compiler import LongTermEnvironmentProfileCompiler


def extract_events(
    compiler: LongTermEnvironmentProfileCompiler,
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
                    environment_id=_normalize_text(attrs.get("environment_id")) or compiler.environment_id,
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


def build_nodes(
    compiler: LongTermEnvironmentProfileCompiler,
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


def build_day_profiles(
    compiler: LongTermEnvironmentProfileCompiler,
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
        epochs, hourly_counts = build_epochs(compiler, motion_events=motion_events)
        health_events = tuple(
            sorted(
                health_by_day.get(day, ()),
                key=lambda item: (item.observed_at, item.source_event_id),
            )
        )
        markers = build_day_markers(
            compiler,
            day=day,
            motion_events=tuple(motion_events),
            epochs=epochs,
            hourly_counts=hourly_counts,
            health_events=health_events,
            known_nodes=known_nodes,
        )
        quality_flags = day_quality_flags(health_events=health_events)
        profiles[day] = SmartHomeEnvironmentDayProfile(
            environment_id=compiler.environment_id,
            day=day,
            weekday_class=_weekday_class(day),
            markers=markers,
            quality_flags=quality_flags,
            supporting_ranges={
                "day_start_hour": compiler.day_start_hour,
                "night_start_hour": compiler.night_start_hour,
            },
        )
        deduped_event_ids = list(dict.fromkeys(all_event_ids_by_day.get(day, ())))
        event_ids[day] = tuple(deduped_event_ids[-_MAX_SOURCE_EVENT_IDS:])
    return profiles, event_ids


def build_epochs(
    compiler: LongTermEnvironmentProfileCompiler,
    *,
    motion_events: Sequence[SmartHomeEnvironmentEvent],
) -> tuple[tuple[SmartHomeEnvironmentEpoch, ...], list[int]]:
    """Compile fixed-width epochs and hourly event counts for one day."""

    epoch_width_s = compiler.epoch_minutes * 60
    epochs_by_index: dict[int, set[str]] = defaultdict(set)
    motion_event_counts: Counter[int] = Counter()
    hourly_counts = [0] * 24
    for event in motion_events:
        minute_of_day = (event.observed_at.hour * 60) + event.observed_at.minute
        epoch_index = minute_of_day // compiler.epoch_minutes
        epochs_by_index[epoch_index].add(event.node_id)
        motion_event_counts[epoch_index] += 1
        hourly_counts[event.observed_at.hour] += 1

    transitions_by_index: Counter[int] = Counter()
    for transition in transition_edges(compiler, motion_events):
        epoch_index = ((transition[2].hour * 60) + transition[2].minute) // compiler.epoch_minutes
        transitions_by_index[epoch_index] += 1

    first_day = motion_events[0].local_day
    created: list[SmartHomeEnvironmentEpoch] = []
    total_epochs = (24 * 60) // compiler.epoch_minutes
    for epoch_index in range(total_epochs):
        if epoch_index not in epochs_by_index:
            continue
        epoch_start = datetime.combine(first_day, datetime.min.time(), tzinfo=motion_events[0].observed_at.tzinfo)
        epoch_start += timedelta(minutes=epoch_index * compiler.epoch_minutes)
        created.append(
            SmartHomeEnvironmentEpoch(
                environment_id=compiler.environment_id,
                epoch_start=epoch_start,
                epoch_width_s=epoch_width_s,
                active_node_ids=tuple(sorted(epochs_by_index[epoch_index])),
                motion_event_count=motion_event_counts[epoch_index],
                transition_count=transitions_by_index[epoch_index],
            )
        )
    return tuple(created), hourly_counts


def build_day_markers(
    compiler: LongTermEnvironmentProfileCompiler,
    *,
    day: date,
    motion_events: tuple[SmartHomeEnvironmentEvent, ...],
    epochs: tuple[SmartHomeEnvironmentEpoch, ...],
    hourly_counts: list[int],
    health_events: tuple[SmartHomeEnvironmentEvent, ...],
    known_nodes: set[str],
) -> dict[str, object]:
    """Compute the marker vector for one day."""

    total_epochs = (24 * 60) // compiler.epoch_minutes
    active_epoch_indices = {
        ((epoch.epoch_start.hour * 60) + epoch.epoch_start.minute) // compiler.epoch_minutes
        for epoch in epochs
    }
    node_counts = Counter(event.node_id for event in motion_events)
    transition_edges_found = transition_edges(compiler, motion_events)
    transition_counts = Counter((source, target) for source, target, _ in transition_edges_found)
    active_flags = [index in active_epoch_indices for index in range(total_epochs)]
    active_epoch_count = len(active_epoch_indices)
    first_activity_minute = None if not motion_events else (motion_events[0].observed_at.hour * 60) + motion_events[0].observed_at.minute
    last_activity_minute = None if not motion_events else (motion_events[-1].observed_at.hour * 60) + motion_events[-1].observed_at.minute
    day_start_epoch = (compiler.day_start_hour * 60) // compiler.epoch_minutes
    night_start_epoch = (compiler.night_start_hour * 60) // compiler.epoch_minutes
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
    active_run_lengths: list[int] = []
    current_active_run = 0
    inactive_gap_lengths: list[int] = []
    current_inactive_gap = 0
    for index, active in enumerate(active_flags):
        if active and (index == 0 or not active_flags[index - 1]):
            motion_burst_count += 1
        if active and index + 1 < len(active_flags):
            active_followed_count += 1
            if not active_flags[index + 1]:
                active_to_inactive_count += 1
        if active:
            current_active_run += 1
            if current_inactive_gap > 0:
                inactive_gap_lengths.append(current_inactive_gap)
                current_inactive_gap = 0
        else:
            current_inactive_gap += 1
            if current_active_run > 0:
                active_run_lengths.append(current_active_run)
                current_active_run = 0
    if current_active_run > 0:
        active_run_lengths.append(current_active_run)
    if current_inactive_gap > 0:
        inactive_gap_lengths.append(current_inactive_gap)

    transition_count = sum(transition_counts.values())
    mean_active_node_count = (
        sum(len(epoch.active_node_ids) for epoch in epochs) / len(epochs)
        if epochs
        else 0.0
    )
    interval_minutes = [
        max(0.0, (later.observed_at - earlier.observed_at).total_seconds() / 60.0)
        for earlier, later in zip(motion_events, motion_events[1:])
    ]

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
        "longest_daytime_inactivity_min": longest_daytime_inactivity_epochs * compiler.epoch_minutes,
        "night_activity_epoch_count": night_activity_epoch_count,
        "unique_active_node_count_day": len(node_counts),
        "mean_active_node_count_per_active_epoch": round(mean_active_node_count, 4),
        "node_entropy_day": round(
            _entropy_from_counts(cast(Mapping[object, int], node_counts)),
            4,
        ),
        "dominant_node_share_day": round((max(node_counts.values()) / sum(node_counts.values())) if node_counts else 0.0, 4),
        "transition_count_day": transition_count,
        "transition_entropy_day": round(
            _entropy_from_counts(cast(Mapping[object, int], transition_counts)),
            4,
        ),
        "fragmentation_index_day": round((active_to_inactive_count / active_followed_count) if active_followed_count else 0.0, 4),
        "motion_burst_count_day": motion_burst_count,
        "inter_event_interval_median_day": None if not interval_minutes else round(float(median(interval_minutes)), 4),
        "inter_event_interval_iqr_day": None if len(interval_minutes) < 2 else round(iqr(interval_minutes), 4),
        "active_epoch_run_length_median_day": None if not active_run_lengths else round(float(median(active_run_lengths)) * compiler.epoch_minutes, 4),
        "inactive_gap_entropy_day": round(
                _entropy_from_counts(
                    cast(
                        Mapping[object, int],
                        Counter(int(length) for length in inactive_gap_lengths),
                    )
                ),
                4,
            ),
        "circadian_similarity_14d": None,
        "transition_graph_divergence_14d": None,
        "node_usage_divergence_14d": None,
        "sensor_coverage_ratio_day": None if sensor_coverage_ratio is None else round(sensor_coverage_ratio, 4),
        "hourly_activity_vector": tuple(hourly_counts),
        "profile_day": day.isoformat(),
    }


def update_reference_profile_similarity_markers(
    compiler: LongTermEnvironmentProfileCompiler,
    *,
    day_profiles: Mapping[date, SmartHomeEnvironmentDayProfile],
    events: tuple[SmartHomeEnvironmentEvent, ...],
    reference_day: date,
) -> dict[date, SmartHomeEnvironmentDayProfile]:
    """Update the reference day with similarity and graph-divergence markers."""

    updated = dict(day_profiles)
    reference_profile = updated.get(reference_day)
    if reference_profile is None:
        return updated

    comparison_days = [
        day
        for day in sorted(updated)
        if day < reference_day
        and (reference_day - day).days <= compiler.short_baseline_days
        and _weekday_class(day) == reference_profile.weekday_class
    ]
    if len(comparison_days) < compiler.min_baseline_days:
        comparison_days = [
            day
            for day in sorted(updated)
            if day < reference_day and (reference_day - day).days <= compiler.short_baseline_days
        ]
    if len(comparison_days) < compiler.min_baseline_days:
        return updated

    reference_markers = dict(reference_profile.markers)
    today_vector = reference_markers.get("hourly_activity_vector")
    if isinstance(today_vector, tuple):
        prior_vectors: list[tuple[float, ...]] = []
        for day in comparison_days:
            candidate = updated[day].markers.get("hourly_activity_vector")
            if isinstance(candidate, tuple):
                prior_vectors.append(tuple(float(value) for value in candidate))
        if prior_vectors and len(prior_vectors[0]) == len(today_vector):
            baseline_vector = [
                float(median([vector[index] for vector in prior_vectors]))
                for index in range(len(today_vector))
            ]
            similarity = _cosine_similarity(
                [float(value) for value in today_vector],
                baseline_vector,
            )
            if similarity is not None:
                reference_markers["circadian_similarity_14d"] = round(similarity, 4)

    motion_by_day: dict[date, tuple[SmartHomeEnvironmentEvent, ...]] = {}
    for day in tuple(comparison_days) + (reference_day,):
        day_events = tuple(
            event
            for event in events
            if event.signal_kind == "motion_detected" and event.local_day == day
        )
        if day_events:
            motion_by_day[day] = day_events

    current_motion_events = motion_by_day.get(reference_day, ())
    if current_motion_events:
        current_node_counts = Counter(event.node_id for event in current_motion_events)
        current_transition_counts = Counter(
            f"{source}->{target}"
            for source, target, _ in transition_edges(compiler, current_motion_events)
        )
        comparison_node_counts: Counter[str] = Counter()
        comparison_transition_counts: Counter[str] = Counter()
        for day in comparison_days:
            day_events = motion_by_day.get(day, ())
            comparison_node_counts.update(event.node_id for event in day_events)
            comparison_transition_counts.update(
                f"{source}->{target}"
                for source, target, _ in transition_edges(compiler, day_events)
            )
        node_divergence = jensen_shannon_divergence(
            cast(Mapping[object, float | int], current_node_counts),
            cast(Mapping[object, float | int], comparison_node_counts),
        )
        transition_divergence = jensen_shannon_divergence(
            cast(Mapping[object, float | int], current_transition_counts),
            cast(Mapping[object, float | int], comparison_transition_counts),
        )
        if node_divergence is not None:
            reference_markers["node_usage_divergence_14d"] = round(node_divergence, 4)
        if transition_divergence is not None:
            reference_markers["transition_graph_divergence_14d"] = round(transition_divergence, 4)

    updated[reference_day] = SmartHomeEnvironmentDayProfile(
        environment_id=reference_profile.environment_id,
        day=reference_profile.day,
        weekday_class=reference_profile.weekday_class,
        markers=reference_markers,
        quality_flags=reference_profile.quality_flags,
        supporting_ranges=reference_profile.supporting_ranges,
    )
    return updated


def day_quality_flags(
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


def build_baselines(
    compiler: LongTermEnvironmentProfileCompiler,
    *,
    day_profiles: Mapping[date, SmartHomeEnvironmentDayProfile],
    reference_day: date,
    day_event_ids: Mapping[date, tuple[str, ...]],
) -> tuple[dict[str, SmartHomeEnvironmentBaseline], dict[str, tuple[str, ...]]]:
    """Build rolling baselines from prior daily profiles."""

    prior_days = sorted(day for day in day_profiles if day < reference_day)
    baselines: dict[str, SmartHomeEnvironmentBaseline] = {}
    baseline_event_ids: dict[str, tuple[str, ...]] = {}
    for baseline_kind, window_days in (("short", compiler.short_baseline_days), ("long", compiler.long_baseline_days)):
        for weekday_class in _WEEKDAY_CLASSES:
            eligible_days = [
                day
                for day in prior_days
                if (reference_day - day).days <= window_days
                and (weekday_class == "all_days" or _weekday_class(day) == weekday_class)
            ]
            if len(eligible_days) < compiler.min_baseline_days:
                continue
            profiles = [day_profiles[day] for day in eligible_days]
            marker_stats: dict[str, SmartHomeEnvironmentBaselineStat] = {}
            for marker_name in _BASELINE_MARKER_NAMES:
                values: list[float] = []
                for profile in profiles:
                    marker_value = profile.markers.get(marker_name)
                    if isinstance(marker_value, (int, float)):
                        values.append(float(marker_value))
                if len(values) < compiler.min_baseline_days:
                    continue
                marker_stats[marker_name] = SmartHomeEnvironmentBaselineStat(
                    median=float(median(values)),
                    iqr=float(iqr(values)),
                    ewma=float(_ewma(values)),
                    mad=float(mad(values)),
                    lower_quantile=float(quantile(values, compiler.acute_empirical_q)),
                    upper_quantile=float(quantile(values, 1.0 - compiler.acute_empirical_q)),
                )
            if not marker_stats:
                continue
            key = f"{baseline_kind}:{weekday_class}"
            baselines[key] = SmartHomeEnvironmentBaseline(
                environment_id=compiler.environment_id,
                baseline_kind=baseline_kind,
                weekday_class=weekday_class,
                window_days=window_days,
                sample_count=len(eligible_days),
                marker_stats=marker_stats,
            )
            collected_event_ids: list[str] = []
            for day in eligible_days:
                collected_event_ids.extend(day_event_ids.get(day, ()))
            baseline_event_ids[key] = tuple(list(dict.fromkeys(collected_event_ids))[-_MAX_SOURCE_EVENT_IDS:])
    return baselines, baseline_event_ids


def select_baseline(
    baselines: Mapping[str, SmartHomeEnvironmentBaseline],
    *,
    weekday_class: str,
    baseline_kind: str = "short",
) -> SmartHomeEnvironmentBaseline | None:
    """Select the preferred baseline for one weekday bucket and horizon."""

    return (
        baselines.get(f"{baseline_kind}:{weekday_class}")
        or baselines.get(f"{baseline_kind}:all_days")
        or baselines.get(f"short:{weekday_class}")
        or baselines.get("short:all_days")
        or baselines.get(f"long:{weekday_class}")
        or baselines.get("long:all_days")
    )


def node_event_ids(
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


def transition_edges(
    compiler: LongTermEnvironmentProfileCompiler,
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
        if current.node_id != previous.node_id and 0.0 <= delta_s <= compiler.transition_window_s:
            edges.append((previous.node_id, current.node_id, current.observed_at))
        previous = current
    return tuple(edges)
