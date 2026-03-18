"""Plan bounded runtime deadlines across long-term background writers.

This module owns the deterministic timeout planning used by the long-term
runtime service. It converts one total lifecycle timeout into an ordered set of
active writers so each flush call can use the remaining wall-clock deadline
without accidentally reapplying the full timeout per writer.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

from twinr.memory.longterm.runtime.worker import AsyncLongTermWriterState


@dataclass(frozen=True, slots=True)
class LongTermWriterBudget:
    """Describe the flush budget metadata for one active writer."""

    worker_name: str
    timeout_s: float
    pending_count: int
    inflight_count: int
    last_error_message: str | None = None


@dataclass(frozen=True, slots=True)
class LongTermFlushBudgetPlan:
    """Describe one total-deadline split across active long-term writers."""

    total_timeout_s: float
    writer_budgets: tuple[LongTermWriterBudget, ...]


def build_flush_budget_plan(
    *,
    total_timeout_s: float,
    writer_states: Iterable[AsyncLongTermWriterState],
) -> LongTermFlushBudgetPlan:
    """Build the ordered active-writer plan for one total flush deadline.

    Only writers with pending work or a latched error participate in the plan.
    Idle healthy writers are skipped because they do not need any flush
    budget. Each active writer receives the total timeout as its upper bound,
    and the runtime service clamps that bound against the live remaining
    wall-clock deadline before every writer call.
    """

    if not isinstance(total_timeout_s, (int, float)) or isinstance(total_timeout_s, bool):
        raise TypeError("total_timeout_s must be a real number")
    normalized_timeout_s = float(total_timeout_s)
    if not math.isfinite(normalized_timeout_s):
        raise ValueError("total_timeout_s must be finite")
    normalized_timeout_s = max(0.0, normalized_timeout_s)

    active_states = tuple(
        state
        for state in writer_states
        if state.pending_count > 0 or state.last_error_message is not None
    )
    if not active_states:
        return LongTermFlushBudgetPlan(total_timeout_s=normalized_timeout_s, writer_budgets=())

    budgets = tuple(
        LongTermWriterBudget(
            worker_name=state.worker_name,
            timeout_s=normalized_timeout_s,
            pending_count=state.pending_count,
            inflight_count=state.inflight_count,
            last_error_message=state.last_error_message,
        )
        for state in active_states
    )
    return LongTermFlushBudgetPlan(
        total_timeout_s=normalized_timeout_s,
        writer_budgets=budgets,
    )
