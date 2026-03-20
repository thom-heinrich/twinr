"""Keep short RAM-only rolling buffers for multimodal event fusion.

The event-fusion layer operates on short recent history rather than one-off
frames. This module provides the bounded timestamped buffer primitive used by
audio micro-events, vision sequences, and fused claim assembly.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
import math
from typing import Generic, TypeVar


T = TypeVar("T")


def _coerce_timestamp(value: object) -> float:
    """Return one finite non-negative timestamp or raise ``ValueError``."""

    try:
        timestamp = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("observed_at must be a finite non-negative timestamp") from exc
    if not math.isfinite(timestamp) or timestamp < 0.0:
        raise ValueError("observed_at must be a finite non-negative timestamp")
    return timestamp


def _coerce_horizon(value: object) -> float:
    """Return one positive finite buffer horizon in seconds."""

    try:
        horizon = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("horizon_s must be a positive finite number") from exc
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon_s must be a positive finite number")
    return horizon


@dataclass(frozen=True, slots=True)
class TimedSample(Generic[T]):
    """Store one timestamped sample in a rolling buffer."""

    observed_at: float
    value: T

    def __post_init__(self) -> None:
        """Normalize the sample timestamp into one bounded value."""

        object.__setattr__(self, "observed_at", _coerce_timestamp(self.observed_at))


class RollingWindowBuffer(Generic[T]):
    """Keep one bounded monotonic in-memory history of timestamped samples.

    Regressing timestamps are clamped to the latest stored timestamp so callers
    can preserve a monotonic internal sequence even when upstream clocks jitter.
    """

    def __init__(self, *, horizon_s: float) -> None:
        """Initialize one rolling buffer with a strict retention horizon."""

        self.horizon_s = _coerce_horizon(horizon_s)
        self._samples: deque[TimedSample[T]] = deque()

    def __len__(self) -> int:
        """Return the current number of retained samples."""

        return len(self._samples)

    def __iter__(self) -> Iterator[TimedSample[T]]:
        """Iterate over retained samples in timestamp order."""

        return iter(self._samples)

    def append(self, observed_at: float, value: T) -> TimedSample[T]:
        """Append one value and prune samples older than the configured horizon."""

        timestamp = _coerce_timestamp(observed_at)
        latest = self.latest()
        if latest is not None and timestamp < latest.observed_at:
            timestamp = latest.observed_at
        sample = TimedSample(observed_at=timestamp, value=value)
        self._samples.append(sample)
        self.prune(now=timestamp)
        return sample

    def extend(self, samples: Iterable[TimedSample[T] | tuple[float, T]]) -> None:
        """Append many samples while preserving monotonic internal order."""

        for item in samples:
            if isinstance(item, TimedSample):
                self.append(item.observed_at, item.value)
            else:
                observed_at, value = item
                self.append(observed_at, value)

    def latest(self) -> TimedSample[T] | None:
        """Return the latest retained sample when available."""

        return self._samples[-1] if self._samples else None

    def snapshot(self) -> tuple[TimedSample[T], ...]:
        """Return an immutable view of the current retained samples."""

        return tuple(self._samples)

    def since(self, cutoff_s: float) -> tuple[TimedSample[T], ...]:
        """Return retained samples observed on or after one timestamp."""

        cutoff = _coerce_timestamp(cutoff_s)
        return tuple(sample for sample in self._samples if sample.observed_at >= cutoff)

    def between(self, start_s: float, end_s: float) -> tuple[TimedSample[T], ...]:
        """Return retained samples whose timestamps fall inside one closed window."""

        start = _coerce_timestamp(start_s)
        end = _coerce_timestamp(end_s)
        if end < start:
            start, end = end, start
        return tuple(sample for sample in self._samples if start <= sample.observed_at <= end)

    def prune(self, *, now: float | None = None) -> None:
        """Drop samples older than the rolling retention horizon."""

        if not self._samples:
            return
        anchor = _coerce_timestamp(now) if now is not None else self._samples[-1].observed_at
        cutoff = anchor - self.horizon_s
        while self._samples and self._samples[0].observed_at < cutoff:
            self._samples.popleft()

    @property
    def oldest_timestamp(self) -> float | None:
        """Return the oldest retained timestamp when available."""

        return self._samples[0].observed_at if self._samples else None

    @property
    def newest_timestamp(self) -> float | None:
        """Return the newest retained timestamp when available."""

        latest = self.latest()
        return None if latest is None else latest.observed_at


__all__ = [
    "RollingWindowBuffer",
    "TimedSample",
]
