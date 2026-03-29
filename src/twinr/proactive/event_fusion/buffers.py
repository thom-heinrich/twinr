# CHANGELOG: 2026-03-29
# BUG-1: Fixed concurrent read/write races by adding an internal RLock and
#        snapshot-based iteration.
# BUG-2: Fixed timestamp corruption for bounded out-of-order arrivals; small
#        regressions are now inserted in-order instead of always being clamped
#        to the newest sample.
# SEC-1: Added a hard max_samples cap to prevent RAM exhaustion under event
#        storms on Raspberry Pi deployments.
# IMP-1: Added watermark-style late-data handling (late_tolerance_s /
#        late_policy) and optional clock-driven auto-pruning for sparse streams.
# IMP-2: Range queries now use bisect over a sorted timestamp index instead of
#        scanning the whole buffer every time.

"""Keep short RAM-only rolling buffers for multimodal event fusion.

The event-fusion layer operates on short recent history rather than one-off
frames. This module provides the bounded timestamped buffer primitive used by
audio micro-events, vision sequences, and fused claim assembly.

Design notes for 2026 deployments:
- Bounded out-of-order handling via a watermark-style lateness window.
- Thread-safe compound operations for mixed producer / consumer pipelines.
- Hard sample-count cap to survive event storms on edge hardware.
- Optional clock-based pruning so sparse streams do not retain stale history
  forever between appends.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
import math
from threading import RLock
from typing import Generic, Literal, TypeVar


T = TypeVar("T")
LatePolicy = Literal["clamp", "raise"]


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


def _coerce_non_negative_finite(value: object, *, name: str) -> float:
    """Return one finite non-negative number or raise ``ValueError``."""

    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return parsed


def _coerce_max_samples(value: object) -> int | None:
    """Return one positive integer sample cap or ``None`` when explicitly disabled."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("max_samples must be a positive integer or None")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_samples must be a positive integer or None") from exc
    if parsed <= 0:
        raise ValueError("max_samples must be a positive integer or None")
    return parsed


def _coerce_policy(value: object, *, name: str, allowed: tuple[str, ...]) -> str:
    """Return one normalized policy string or raise ``ValueError``."""

    if not isinstance(value, str):
        raise ValueError(f"{name} must be one of: {', '.join(allowed)}")
    normalized = value.strip().lower()
    if normalized not in allowed:
        raise ValueError(f"{name} must be one of: {', '.join(allowed)}")
    return normalized


@dataclass(frozen=True, slots=True)
class TimedSample(Generic[T]):
    """Store one timestamped sample in a rolling buffer."""

    observed_at: float
    value: T

    def __post_init__(self) -> None:
        """Normalize the sample timestamp into one bounded value."""

        object.__setattr__(self, "observed_at", _coerce_timestamp(self.observed_at))


@dataclass(frozen=True, slots=True)
class RollingWindowStats:
    """Expose operational counters for observability and tuning."""

    retained: int
    oldest_timestamp: float | None
    newest_timestamp: float | None
    max_seen_timestamp: float | None
    watermark: float | None
    out_of_order_insertions: int
    late_clamps: int
    capacity_evictions: int
    prune_evictions: int


class RollingWindowBuffer(Generic[T]):
    """Keep one bounded in-memory history of timestamped samples.

    The buffer preserves event-time ordering. Slightly out-of-order samples can be
    inserted at their real timestamp when they fall within ``late_tolerance_s``.
    Samples older than the watermark (``max_seen_timestamp - late_tolerance_s``)
    are either clamped to the watermark or rejected, depending on ``late_policy``.
    """

    def __init__(
        self,
        *,
        horizon_s: float,
        # BREAKING: the buffer is now count-bounded by default to hard-cap RAM
        # on Pi-class devices during event storms. Pass max_samples=None to
        # restore unbounded-by-count behavior.
        max_samples: int | None = 4096,
        # BREAKING: bounded out-of-order handling is enabled by default instead
        # of clamping every regressing timestamp to the newest sample.
        late_tolerance_s: float = 0.250,
        late_policy: LatePolicy = "clamp",
        clock: Callable[[], float] | None = None,
        auto_prune_on_read: bool = True,
    ) -> None:
        """Initialize one rolling buffer.

        Args:
            horizon_s: Retention horizon in seconds.
            max_samples: Hard count cap for retained samples. ``None`` disables
                the cap, though that is discouraged on memory-constrained edge
                devices.
            late_tolerance_s: Allowed event-time disorder before a sample is
                treated as "too late".
            late_policy: ``"clamp"`` keeps too-late samples by moving them to the
                current watermark. ``"raise"`` rejects them with ``ValueError``.
            clock: Optional function returning timestamps in the same timebase as
                ``observed_at``. When provided, read paths can auto-prune stale
                samples even during idle periods.
            auto_prune_on_read: When ``True``, read methods use ``clock`` (when
                available) to prune before returning data.
        """

        self.horizon_s = _coerce_horizon(horizon_s)
        self.max_samples = _coerce_max_samples(max_samples)
        self.late_tolerance_s = _coerce_non_negative_finite(
            late_tolerance_s,
            name="late_tolerance_s",
        )
        self.late_policy = _coerce_policy(
            late_policy,
            name="late_policy",
            allowed=("clamp", "raise"),
        )
        if clock is not None and not callable(clock):
            raise ValueError("clock must be callable when provided")
        self.clock = clock
        self.auto_prune_on_read = bool(auto_prune_on_read)

        self._samples: list[TimedSample[T]] = []
        self._timestamps: list[float] = []
        self._max_seen_timestamp: float | None = None

        self._out_of_order_insertions = 0
        self._late_clamps = 0
        self._capacity_evictions = 0
        self._prune_evictions = 0

        self._lock = RLock()

    def __len__(self) -> int:
        """Return the current number of retained samples."""

        with self._lock:
            self._maybe_prune_on_read_locked(now=None)
            return len(self._samples)

    def __iter__(self) -> Iterator[TimedSample[T]]:
        """Iterate over a stable snapshot of retained samples in timestamp order."""

        return iter(self.snapshot())

    def append(self, observed_at: float, value: T) -> TimedSample[T]:
        """Append one value, handle bounded lateness, and prune expired samples."""

        timestamp = _coerce_timestamp(observed_at)
        with self._lock:
            if self._max_seen_timestamp is not None and timestamp < self._max_seen_timestamp:
                watermark = self._watermark_locked()
                if watermark is not None and timestamp < watermark:
                    if self.late_policy == "raise":
                        raise ValueError(
                            "observed_at is older than the allowed lateness window"
                        )
                    timestamp = watermark
                    self._late_clamps += 1

            sample = TimedSample(observed_at=timestamp, value=value)
            self._insert_sample_locked(sample)

            if self._max_seen_timestamp is None or timestamp > self._max_seen_timestamp:
                self._max_seen_timestamp = timestamp

            anchor = self._max_seen_timestamp if self._max_seen_timestamp is not None else timestamp
            self._prune_locked(now=anchor)
            self._enforce_capacity_locked()
            return sample

    def extend(self, samples: Iterable[TimedSample[T] | tuple[float, T]]) -> None:
        """Append many samples while preserving event-time ordering."""

        for item in samples:
            if isinstance(item, TimedSample):
                self.append(item.observed_at, item.value)
            else:
                observed_at, value = item
                self.append(observed_at, value)

    def latest(self, *, now: float | None = None) -> TimedSample[T] | None:
        """Return the latest retained sample when available."""

        with self._lock:
            self._maybe_prune_on_read_locked(now)
            return self._samples[-1] if self._samples else None

    def snapshot(self, *, now: float | None = None) -> tuple[TimedSample[T], ...]:
        """Return one immutable view of the current retained samples."""

        with self._lock:
            self._maybe_prune_on_read_locked(now)
            return tuple(self._samples)

    def since(self, cutoff_s: float, *, now: float | None = None) -> tuple[TimedSample[T], ...]:
        """Return retained samples observed on or after one timestamp."""

        cutoff = _coerce_timestamp(cutoff_s)
        with self._lock:
            self._maybe_prune_on_read_locked(now)
            index = bisect_left(self._timestamps, cutoff)
            return tuple(self._samples[index:])

    def between(
        self,
        start_s: float,
        end_s: float,
        *,
        now: float | None = None,
    ) -> tuple[TimedSample[T], ...]:
        """Return retained samples whose timestamps fall inside one closed window."""

        start = _coerce_timestamp(start_s)
        end = _coerce_timestamp(end_s)
        if end < start:
            start, end = end, start
        with self._lock:
            self._maybe_prune_on_read_locked(now)
            left = bisect_left(self._timestamps, start)
            right = bisect_right(self._timestamps, end)
            return tuple(self._samples[left:right])

    def prune(self, *, now: float | None = None) -> None:
        """Drop samples older than the rolling retention horizon."""

        with self._lock:
            anchor = self._resolve_anchor_locked(now, for_read=False)
            if anchor is not None:
                self._prune_locked(now=anchor)

    def clear(self) -> None:
        """Reset retained samples and operational counters."""

        with self._lock:
            self._samples.clear()
            self._timestamps.clear()
            self._max_seen_timestamp = None
            self._out_of_order_insertions = 0
            self._late_clamps = 0
            self._capacity_evictions = 0
            self._prune_evictions = 0

    def stats(self, *, now: float | None = None) -> RollingWindowStats:
        """Return one snapshot of current buffer state and counters."""

        with self._lock:
            self._maybe_prune_on_read_locked(now)
            return RollingWindowStats(
                retained=len(self._samples),
                oldest_timestamp=self._timestamps[0] if self._timestamps else None,
                newest_timestamp=self._timestamps[-1] if self._timestamps else None,
                max_seen_timestamp=self._max_seen_timestamp,
                watermark=self._watermark_locked(),
                out_of_order_insertions=self._out_of_order_insertions,
                late_clamps=self._late_clamps,
                capacity_evictions=self._capacity_evictions,
                prune_evictions=self._prune_evictions,
            )

    @property
    def oldest_timestamp(self) -> float | None:
        """Return the oldest retained timestamp when available."""

        with self._lock:
            self._maybe_prune_on_read_locked(now=None)
            return self._timestamps[0] if self._timestamps else None

    @property
    def newest_timestamp(self) -> float | None:
        """Return the newest retained timestamp when available."""

        with self._lock:
            self._maybe_prune_on_read_locked(now=None)
            return self._timestamps[-1] if self._timestamps else None

    @property
    def max_seen_timestamp(self) -> float | None:
        """Return the maximum event-time timestamp ever accepted since last clear."""

        with self._lock:
            return self._max_seen_timestamp

    @property
    def watermark(self) -> float | None:
        """Return the current lateness watermark."""

        with self._lock:
            return self._watermark_locked()

    def _resolve_anchor_locked(self, now: float | None, *, for_read: bool) -> float | None:
        """Resolve one pruning anchor in the same timebase as ``observed_at``."""

        if now is not None:
            return _coerce_timestamp(now)
        if for_read and self.auto_prune_on_read and self.clock is not None:
            return _coerce_timestamp(self.clock())
        if self._max_seen_timestamp is not None:
            return self._max_seen_timestamp
        if self._timestamps:
            return self._timestamps[-1]
        return None

    def _watermark_locked(self) -> float | None:
        """Return the current lateness watermark when one exists."""

        if self._max_seen_timestamp is None:
            return None
        return max(0.0, self._max_seen_timestamp - self.late_tolerance_s)

    def _insert_sample_locked(self, sample: TimedSample[T]) -> None:
        """Insert one sample while preserving sorted timestamp order."""

        if not self._samples or sample.observed_at >= self._timestamps[-1]:
            self._samples.append(sample)
            self._timestamps.append(sample.observed_at)
            return

        index = bisect_right(self._timestamps, sample.observed_at)
        self._samples.insert(index, sample)
        self._timestamps.insert(index, sample.observed_at)
        self._out_of_order_insertions += 1

    def _enforce_capacity_locked(self) -> None:
        """Evict oldest samples when the hard count cap is exceeded."""

        if self.max_samples is None:
            return
        overflow = len(self._samples) - self.max_samples
        if overflow > 0:
            self._evict_oldest_locked(overflow, reason="capacity")

    def _maybe_prune_on_read_locked(self, now: float | None) -> None:
        """Prune on reads when one explicit or configured anchor is available."""

        anchor = self._resolve_anchor_locked(now, for_read=True)
        if anchor is not None:
            self._prune_locked(now=anchor)

    def _prune_locked(self, *, now: float) -> None:
        """Prune expired samples using one already-resolved anchor."""

        if not self._samples:
            return

        anchor = _coerce_timestamp(now)
        cutoff = anchor - self.horizon_s
        if cutoff <= self._timestamps[0]:
            return

        keep_from = bisect_left(self._timestamps, cutoff)
        if keep_from > 0:
            self._evict_oldest_locked(keep_from, reason="prune")

    def _evict_oldest_locked(self, count: int, *, reason: str) -> None:
        """Remove ``count`` oldest samples and update counters."""

        if count <= 0:
            return

        count = min(count, len(self._samples))
        del self._samples[:count]
        del self._timestamps[:count]

        if reason == "capacity":
            self._capacity_evictions += count
        elif reason == "prune":
            self._prune_evictions += count


__all__ = [
    "RollingWindowBuffer",
    "RollingWindowStats",
    "TimedSample",
]