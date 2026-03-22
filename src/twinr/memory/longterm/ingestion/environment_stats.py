"""Shared bounded statistics helpers for room-agnostic environment profiling.

The environment-profile compiler uses these helpers to derive robust dispersion
estimates and divergence markers without embedding the math inline in the
orchestration module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from statistics import median


def quantile(values: Sequence[float], q: float) -> float:
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


def iqr(values: Sequence[float]) -> float:
    """Return the interquartile range for one non-empty sample."""

    return quantile(values, 0.75) - quantile(values, 0.25)


def mad(values: Sequence[float]) -> float:
    """Return one median absolute deviation for a non-empty sample."""

    if not values:
        raise ValueError("values must not be empty.")
    center = float(median(float(value) for value in values))
    deviations = [abs(float(value) - center) for value in values]
    return float(median(deviations))


def safe_sigma(*, mad_value: float, iqr_value: float) -> float:
    """Return one bounded robust sigma estimate derived from MAD or IQR."""

    if mad_value > 0.0:
        return max(1.4826 * float(mad_value), 1e-6)
    if iqr_value > 0.0:
        return max(float(iqr_value) / 1.349, 1e-6)
    return 1e-6


def probability_mapping(counts: Mapping[object, float | int]) -> dict[str, float]:
    """Normalize one count mapping into a stable probability mapping."""

    normalized: dict[str, float] = {}
    total = 0.0
    for key, raw in counts.items():
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value <= 0.0:
            continue
        normalized[str(key)] = value
        total += value
    if total <= 0.0:
        return {}
    return {key: (value / total) for key, value in normalized.items()}


def jensen_shannon_divergence(left: Mapping[object, float | int], right: Mapping[object, float | int]) -> float | None:
    """Return the Jensen-Shannon divergence between two bounded distributions."""

    left_probs = probability_mapping(left)
    right_probs = probability_mapping(right)
    if not left_probs or not right_probs:
        return None
    keys = set(left_probs) | set(right_probs)
    if not keys:
        return None
    midpoint = {
        key: 0.5 * (left_probs.get(key, 0.0) + right_probs.get(key, 0.0))
        for key in keys
    }

    def _kl(source: Mapping[str, float], baseline: Mapping[str, float]) -> float:
        value = 0.0
        for key, probability in source.items():
            if probability <= 0.0:
                continue
            baseline_probability = baseline.get(key, 0.0)
            if baseline_probability <= 0.0:
                continue
            value += probability * math.log(probability / baseline_probability, 2)
        return value

    divergence = 0.5 * _kl(left_probs, midpoint) + 0.5 * _kl(right_probs, midpoint)
    return max(0.0, float(divergence))
