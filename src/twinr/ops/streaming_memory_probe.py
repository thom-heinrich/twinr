"""Throttle process-local streaming memory checkpoints for hot runtime loops.

This helper keeps the shared streaming-memory attribution store useful under
busy Pi runtime threads. Long-lived workers can call ``maybe_record()`` on each
iteration without fsyncing a fresh phase on every pass; the helper records the
latest subsystem phase only when enough wall time elapsed or anonymous memory
grew materially since the previous sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import time
from typing import SupportsFloat, SupportsIndex, cast

from twinr.agent.base_agent.config import TwinrConfig
from twinr.ops.process_memory import (
    ProcessMemoryMetrics,
    StreamingMemoryAttributionStore,
)


_DEFAULT_SAMPLE_INTERVAL_S = 5.0
_DEFAULT_GROWTH_THRESHOLD_KB = 64 * 1024


def _coerce_sample_interval_s(value: object) -> float:
    try:
        interval_s = float(
            cast(SupportsFloat | SupportsIndex | str | bytes | bytearray, value)
        )
    except (TypeError, ValueError):
        return _DEFAULT_SAMPLE_INTERVAL_S
    if not math.isfinite(interval_s) or interval_s <= 0.0:
        return _DEFAULT_SAMPLE_INTERVAL_S
    return max(0.25, interval_s)


def _coerce_growth_threshold_kb(value: object) -> int:
    try:
        threshold_kb = int(
            cast(SupportsIndex | str | bytes | bytearray, value)
        )
    except (TypeError, ValueError):
        return _DEFAULT_GROWTH_THRESHOLD_KB
    return max(1, threshold_kb)


@dataclass(slots=True)
class StreamingMemoryProbe:
    """Record one bounded subsystem checkpoint into the streaming memory store."""

    path: Path | None
    label: str
    owner_label: str
    owner_detail: str
    sample_interval_s: float = _DEFAULT_SAMPLE_INTERVAL_S
    growth_threshold_kb: int = _DEFAULT_GROWTH_THRESHOLD_KB
    _recorded_once: bool = field(default=False, init=False, repr=False)
    _last_recorded_at_monotonic_s: float = field(default=0.0, init=False, repr=False)
    _last_recorded_anonymous_kb: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.sample_interval_s = _coerce_sample_interval_s(self.sample_interval_s)
        self.growth_threshold_kb = _coerce_growth_threshold_kb(self.growth_threshold_kb)

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig | object | None,
        *,
        label: str,
        owner_label: str,
        owner_detail: str,
        sample_interval_s: float = _DEFAULT_SAMPLE_INTERVAL_S,
        growth_threshold_kb: int = _DEFAULT_GROWTH_THRESHOLD_KB,
    ) -> "StreamingMemoryProbe":
        """Build one probe from config when the ops store is available."""

        path: Path | None
        try:
            path = StreamingMemoryAttributionStore.from_config(
                cast(TwinrConfig, config)
            ).path
        except Exception:
            path = None
        return cls(
            path=path,
            label=label,
            owner_label=owner_label,
            owner_detail=owner_detail,
            sample_interval_s=sample_interval_s,
            growth_threshold_kb=growth_threshold_kb,
        )

    def maybe_record(
        self,
        *,
        force: bool = False,
        owner_detail: str | None = None,
    ) -> bool:
        """Record the latest phase when time or anonymous growth warrants it."""

        path = self.path
        if path is None:
            return False

        now = time.monotonic()
        current_metrics = ProcessMemoryMetrics.from_proc()
        current_anonymous_kb = current_metrics.preferred_anonymous_kb()
        due_by_growth = bool(
            current_anonymous_kb is not None
            and self._last_recorded_anonymous_kb is not None
            and (current_anonymous_kb - self._last_recorded_anonymous_kb) >= self.growth_threshold_kb
        )
        due_by_time = (now - self._last_recorded_at_monotonic_s) >= self.sample_interval_s
        if self._recorded_once and not force and not due_by_growth and not due_by_time:
            return False

        snapshot = StreamingMemoryAttributionStore(path).record_phase(
            label=self.label,
            owner_label=self.owner_label,
            owner_detail=owner_detail or self.owner_detail,
            replace=True,
        )
        if snapshot is None:
            return False
        self._recorded_once = True
        self._last_recorded_at_monotonic_s = now
        self._last_recorded_anonymous_kb = snapshot.current_metrics.preferred_anonymous_kb()
        return True


__all__ = ["StreamingMemoryProbe"]
