"""Deviation, quality-state, and regime analysis for environment profiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import TYPE_CHECKING

from twinr.memory.longterm.ingestion.environment_stats import safe_sigma

from .helpers import _ewma
from .models import (
    SmartHomeEnvironmentBaseline,
    SmartHomeEnvironmentChangePoint,
    SmartHomeEnvironmentDayProfile,
    SmartHomeEnvironmentDeviation,
    SmartHomeEnvironmentDeviationEvent,
    SmartHomeEnvironmentDeviationMetric,
    SmartHomeEnvironmentNode,
    SmartHomeEnvironmentQualityState,
    SmartHomeEnvironmentRegime,
)
from .pipeline import select_baseline

if TYPE_CHECKING:
    from .compiler import LongTermEnvironmentProfileCompiler


def _marker_float(value: object) -> float | None:
    """Return one float marker value when the payload is numeric."""

    if isinstance(value, (int, float)):
        return float(value)
    return None


def build_deviations(
    *,
    reference: datetime,
    reference_day: date,
    day_profiles: Mapping[date, SmartHomeEnvironmentDayProfile],
    baselines: Mapping[str, SmartHomeEnvironmentBaseline],
    quality_state: SmartHomeEnvironmentQualityState | None,
    compiler: LongTermEnvironmentProfileCompiler,
) -> tuple[SmartHomeEnvironmentDeviation | SmartHomeEnvironmentDeviationEvent, ...]:
    """Build typed deviations for the reference day against the rolling baseline."""

    profile = day_profiles.get(reference_day)
    if profile is None:
        return ()
    baseline = select_baseline(
        baselines=baselines,
        weekday_class=profile.weekday_class,
        baseline_kind="short",
    )
    if baseline is None:
        return ()

    deviations: list[SmartHomeEnvironmentDeviation | SmartHomeEnvironmentDeviationEvent] = []
    event_metrics: list[SmartHomeEnvironmentDeviationMetric] = []
    blocked_by = tuple(quality_state.blocked_by) if quality_state is not None else ()
    deviation_specs = (
        ("daily_activity_drop", "active_epoch_count_day", "low", "less activity than usual"),
        ("night_activity_increase", "night_activity_epoch_count", "high", "more night activity than usual"),
        ("late_start_of_day", "first_activity_minute_local", "high", "later start of day than usual"),
        ("early_end_of_day", "last_activity_minute_local", "low", "earlier end of day than usual"),
        ("fragmentation_shift", "fragmentation_index_day", "high", "more fragmented movement than usual"),
        ("transition_graph_shift", "transition_graph_divergence_14d", "high", "changed movement transitions compared with the recent pattern"),
        ("node_usage_shift", "node_usage_divergence_14d", "high", "changed spread of movement across nodes"),
        ("possible_sensor_failure", "sensor_coverage_ratio_day", "low", "lower sensor coverage than expected"),
    )
    for deviation_type, marker_name, direction, short_label in deviation_specs:
        metric = deviation_metric(
            profile=profile,
            baseline=baseline,
            marker_name=marker_name,
            direction=direction,
            compiler=compiler,
        )
        if metric is None:
            continue
        event_metrics.append(metric)
        delta_ratio = abs(metric.delta_ratio)
        severity = "high" if (metric.robust_z is not None and abs(metric.robust_z) >= max(4.0, compiler.acute_z_threshold + 0.5)) or delta_ratio >= 0.45 else "moderate"
        deviations.append(
            SmartHomeEnvironmentDeviation(
                environment_id=compiler.environment_id,
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
    if event_metrics:
        event_severity = "high" if any(metric.robust_z is not None and abs(metric.robust_z) >= 4.0 for metric in event_metrics) or len(event_metrics) >= 3 else "moderate"
        deviations.append(
            SmartHomeEnvironmentDeviationEvent(
                environment_id=compiler.environment_id,
                observed_at=reference,
                classification="acute_deviation",
                severity=event_severity,
                time_scale="day",
                metrics=tuple(event_metrics[:4]),
                quality_flags=profile.quality_flags,
                blocked_by=blocked_by,
                short_label="multiple environment markers differ from the recent pattern",
                human_readable=(
                    "Several room-agnostic smart-home markers differ from the recent pattern for this day. "
                    "Treat this as an acute environment deviation only together with the attached quality and blocker signals."
                ),
            )
        )
    return tuple(deviations)


def deviation_metric(
    *,
    profile: SmartHomeEnvironmentDayProfile,
    baseline: SmartHomeEnvironmentBaseline,
    marker_name: str,
    direction: str,
    compiler: LongTermEnvironmentProfileCompiler,
) -> SmartHomeEnvironmentDeviationMetric | None:
    """Return one deviation metric when a marker moved materially."""

    observed = _marker_float(profile.markers.get(marker_name))
    if observed is None:
        return None
    stats = baseline.marker_stats.get(marker_name)
    if stats is None:
        return None
    baseline_median = float(stats.median)
    robust_sigma = safe_sigma(mad_value=float(stats.mad), iqr_value=float(stats.iqr))
    robust_z = (observed - baseline_median) / robust_sigma
    spread = max(float(stats.iqr) * compiler.iqr_multiplier, 1.0 if marker_name.endswith("_count_day") else 0.05)
    delta = observed - baseline_median
    if direction == "high" and delta <= spread and observed <= float(stats.upper_quantile):
        return None
    if direction == "low" and delta >= -spread and observed >= float(stats.lower_quantile):
        return None
    if direction == "high" and robust_z < compiler.acute_z_threshold and observed <= float(stats.upper_quantile):
        return None
    if direction == "low" and robust_z > -compiler.acute_z_threshold and observed >= float(stats.lower_quantile):
        return None
    denominator = max(abs(baseline_median), robust_sigma, 1.0)
    return SmartHomeEnvironmentDeviationMetric(
        name=marker_name,
        observed=observed,
        baseline_median=baseline_median,
        delta_ratio=delta / denominator,
        robust_z=robust_z,
        lower_quantile=float(stats.lower_quantile),
        upper_quantile=float(stats.upper_quantile),
    )


def build_quality_state(
    *,
    reference: datetime,
    reference_day: date,
    nodes: tuple[SmartHomeEnvironmentNode, ...],
    day_profiles: Mapping[date, SmartHomeEnvironmentDayProfile],
    baselines: Mapping[str, SmartHomeEnvironmentBaseline],
    compiler: LongTermEnvironmentProfileCompiler,
) -> SmartHomeEnvironmentQualityState | None:
    """Build one bounded quality-state summary for the reference day."""

    profile = day_profiles.get(reference_day)
    if profile is None:
        return None
    baseline = select_baseline(
        baselines=baselines,
        weekday_class=profile.weekday_class,
        baseline_kind="short",
    )
    quality_flags = list(profile.quality_flags)
    blocked_by: list[str] = []
    evidence_markers: list[str] = []

    coverage_value = _marker_float(profile.markers.get("sensor_coverage_ratio_day"))
    if coverage_value is not None and coverage_value < compiler.min_coverage_ratio:
        quality_flags.append("low_sensor_coverage")
        blocked_by.append("sensor_quality_limited")
        evidence_markers.append("sensor_coverage_ratio_day")
    if "device_offline_present" in quality_flags:
        blocked_by.append("sensor_quality_limited")
        evidence_markers.append("sensor_coverage_ratio_day")

    recent_node_change_present = any(
        0 <= (reference_day - node.first_seen_at.date()).days <= max(2, compiler.drift_min_days)
        for node in nodes
    )
    if recent_node_change_present:
        quality_flags.append("recent_node_change_present")

    if baseline is not None and likely_visitor_activity(profile=profile, baseline=baseline, compiler=compiler):
        quality_flags.append("possible_visitor_or_multi_person_activity")
        evidence_markers.extend(
            (
                "active_epoch_count_day",
                "unique_active_node_count_day",
                "node_entropy_day",
            )
        )

    deduped_flags = tuple(dict.fromkeys(flag for flag in quality_flags if flag))
    deduped_blocked = tuple(dict.fromkeys(flag for flag in blocked_by if flag))
    deduped_evidence = tuple(dict.fromkeys(marker for marker in evidence_markers if marker))
    classification = "blocked" if deduped_blocked else ("caution" if deduped_flags else "ok")
    if classification == "ok":
        detail = "Environment quality looked good enough for behavior interpretation."
    elif classification == "blocked":
        detail = (
            "Environment quality is limited for behavior interpretation because sensor health or coverage is reduced. "
            f"Flags: {', '.join(deduped_flags) or 'none'}."
        )
    else:
        detail = (
            "Environment quality is usable with caution. "
            f"Flags: {', '.join(deduped_flags) or 'none'}."
        )
    return SmartHomeEnvironmentQualityState(
        environment_id=compiler.environment_id,
        observed_at=reference,
        classification=classification,
        quality_flags=deduped_flags,
        blocked_by=deduped_blocked,
        evidence_markers=deduped_evidence,
        human_readable=detail,
    )


def likely_visitor_activity(
    *,
    profile: SmartHomeEnvironmentDayProfile,
    baseline: SmartHomeEnvironmentBaseline,
    compiler: LongTermEnvironmentProfileCompiler,
) -> bool:
    """Return whether the day looks more like unusual multi-person activity."""

    marker_names = (
        "active_epoch_count_day",
        "unique_active_node_count_day",
        "node_entropy_day",
    )
    positives = 0
    for marker_name in marker_names:
        observed = _marker_float(profile.markers.get(marker_name))
        stats = baseline.marker_stats.get(marker_name)
        if observed is None or stats is None:
            continue
        sigma = safe_sigma(mad_value=float(stats.mad), iqr_value=float(stats.iqr))
        robust_z = (observed - float(stats.median)) / sigma
        if robust_z >= compiler.acute_z_threshold and observed >= float(stats.upper_quantile):
            positives += 1
    return positives >= 2


def build_regime_signals(
    *,
    reference: datetime,
    reference_day: date,
    day_profiles: Mapping[date, SmartHomeEnvironmentDayProfile],
    baselines: Mapping[str, SmartHomeEnvironmentBaseline],
    quality_state: SmartHomeEnvironmentQualityState | None,
    compiler: LongTermEnvironmentProfileCompiler,
) -> tuple[SmartHomeEnvironmentChangePoint | None, SmartHomeEnvironmentRegime | None]:
    """Build transition and accepted-regime signals for the reference day."""

    profile = day_profiles.get(reference_day)
    if profile is None:
        return None, None
    short_baseline = select_baseline(
        baselines=baselines,
        weekday_class=profile.weekday_class,
        baseline_kind="short",
    )
    long_baseline = select_baseline(
        baselines=baselines,
        weekday_class=profile.weekday_class,
        baseline_kind="long",
    )
    if short_baseline is None or long_baseline is None:
        return None, None

    metrics: list[SmartHomeEnvironmentDeviationMetric] = []
    regime_metrics: list[SmartHomeEnvironmentDeviationMetric] = []
    candidate_start_days: list[date] = []
    regime_start_days: list[date] = []
    for marker_name in (
        "active_epoch_count_day",
        "night_activity_epoch_count",
        "first_activity_minute_local",
        "last_activity_minute_local",
        "fragmentation_index_day",
        "transition_graph_divergence_14d",
        "node_usage_divergence_14d",
    ):
        metric, run_start_day, run_length = drift_metric(
            reference_day=reference_day,
            day_profiles=day_profiles,
            long_baseline=long_baseline,
            marker_name=marker_name,
            compiler=compiler,
        )
        if metric is None or run_start_day is None:
            continue
        metrics.append(metric)
        candidate_start_days.append(run_start_day)
        if run_length >= compiler.regime_accept_days:
            regime_metrics.append(metric)
            regime_start_days.append(run_start_day)

    if not metrics:
        return None, None

    blocked_by = tuple(quality_state.blocked_by) if quality_state is not None else ()
    quality_flags = tuple(quality_state.quality_flags) if quality_state is not None else ()
    change_point = SmartHomeEnvironmentChangePoint(
        environment_id=compiler.environment_id,
        observed_at=reference,
        change_started_on=min(candidate_start_days),
        severity="high" if any(metric.robust_z is not None and abs(metric.robust_z) >= 2.5 for metric in metrics) or len(metrics) >= 3 else "moderate",
        metrics=tuple(metrics[:4]),
        quality_flags=quality_flags,
        blocked_by=blocked_by,
        human_readable=(
            "Recent room-agnostic smart-home markers suggest an ongoing transition away from the older behavior regime. "
            "This is a drift candidate, not a confirmed concern on its own."
        ),
    )

    regime = None
    if regime_metrics and not blocked_by:
        regime = SmartHomeEnvironmentRegime(
            environment_id=compiler.environment_id,
            valid_from_day=min(regime_start_days),
            observed_at=reference,
            severity="high" if len(regime_metrics) >= 3 else "moderate",
            metrics=tuple(regime_metrics[:4]),
            quality_flags=quality_flags,
            human_readable=(
                "The environment has stayed on a shifted pattern long enough to treat it as a new normal baseline candidate."
            ),
        )
    return change_point, regime


def drift_metric(
    *,
    reference_day: date,
    day_profiles: Mapping[date, SmartHomeEnvironmentDayProfile],
    long_baseline: SmartHomeEnvironmentBaseline,
    marker_name: str,
    compiler: LongTermEnvironmentProfileCompiler,
) -> tuple[SmartHomeEnvironmentDeviationMetric | None, date | None, int]:
    """Return one drift metric plus the start day and run length when stable."""

    stats = long_baseline.marker_stats.get(marker_name)
    if stats is None:
        return None, None, 0
    series: list[tuple[date, float]] = []
    for day in sorted(day_profiles):
        if day > reference_day or (reference_day - day).days > compiler.long_baseline_days:
            continue
        marker_value = _marker_float(day_profiles[day].markers.get(marker_name))
        if marker_value is None:
            continue
        series.append((day, marker_value))
    if len(series) < max(compiler.min_baseline_days, compiler.drift_min_days + 1):
        return None, None, 0

    sigma = safe_sigma(mad_value=float(stats.mad), iqr_value=float(stats.iqr))
    baseline_median = float(stats.median)
    residuals = [(value - baseline_median) / sigma for _, value in series]
    short_values = [value for _, value in series[-min(7, len(series)):]]
    short_ewma = _ewma(short_values)
    shift_sigma = (short_ewma - baseline_median) / sigma
    if abs(shift_sigma) < compiler.drift_min_sigma:
        return None, None, 0
    direction = 1 if shift_sigma > 0.0 else -1
    if not cusum_change_detected(residuals=residuals, direction=direction, compiler=compiler):
        return None, None, 0

    trailing_days: list[date] = []
    for day, value in reversed(series):
        residual = (value - baseline_median) / sigma
        sign = 1 if residual > 0.0 else -1 if residual < 0.0 else 0
        if sign != direction or abs(residual) < compiler.drift_min_sigma:
            break
        trailing_days.append(day)
    if len(trailing_days) < compiler.drift_min_days:
        return None, None, 0

    metric = SmartHomeEnvironmentDeviationMetric(
        name=marker_name,
        observed=short_ewma,
        baseline_median=baseline_median,
        delta_ratio=(short_ewma - baseline_median) / max(abs(baseline_median), sigma, 1.0),
        robust_z=shift_sigma,
        lower_quantile=float(stats.lower_quantile),
        upper_quantile=float(stats.upper_quantile),
    )
    return metric, min(trailing_days), len(trailing_days)


def cusum_change_detected(
    *,
    residuals: Sequence[float],
    direction: int,
    compiler: LongTermEnvironmentProfileCompiler,
) -> bool:
    """Return whether one simple CUSUM run indicates a regime change."""

    if not residuals or direction == 0:
        return False
    slack = 0.5
    threshold = max(3.5, compiler.drift_min_sigma * float(compiler.drift_min_days))
    positive = 0.0
    negative = 0.0
    for residual in residuals:
        positive = max(0.0, positive + residual - slack)
        negative = min(0.0, negative + residual + slack)
    return positive >= threshold if direction > 0 else abs(negative) >= threshold


__all__ = [
    "build_deviations",
    "build_quality_state",
    "build_regime_signals",
    "cusum_change_detected",
    "deviation_metric",
    "drift_metric",
    "likely_visitor_activity",
]
