# mypy: disable-error-code=attr-defined
"""Lifecycle helpers for the long-term runtime service."""

from __future__ import annotations

import time

from twinr.memory.longterm.runtime.live_object_selectors import select_restart_recall_source_objects
from twinr.memory.longterm.runtime.flush_budget import build_flush_budget_plan
from twinr.memory.longterm.runtime.worker import AsyncLongTermWriterState

from ._typing import ServiceMixinBase
from .compat import _coerce_timeout_s, logger


class LongTermMemoryServiceLifecycleMixin(ServiceMixinBase):
    """Flush and shutdown helpers for bounded background writers."""

    def flush(self, *, timeout_s: float = 2.0) -> bool:
        """Flush active background writers within one true total deadline."""

        resolved_timeout_s = _coerce_timeout_s(timeout_s, default=2.0)
        flush_targets: list[tuple[str, object, AsyncLongTermWriterState]] = []
        if self.writer is not None:
            flush_targets.append(("conversation", self.writer, self.writer.snapshot_state()))
        if self.multimodal_writer is not None:
            flush_targets.append(("multimodal", self.multimodal_writer, self.multimodal_writer.snapshot_state()))

        plan = build_flush_budget_plan(
            total_timeout_s=resolved_timeout_s,
            writer_states=(state for _, _, state in flush_targets),
        )
        budgets_by_name = {
            budget.worker_name: budget
            for budget in plan.writer_budgets
        }
        deadline = time.monotonic() + resolved_timeout_s
        flush_ok = True
        for label, writer, state in flush_targets:
            budget = budgets_by_name.get(getattr(state, "worker_name", ""))
            if budget is None:
                continue
            writer_timeout_s = min(
                budget.timeout_s,
                max(0.0, deadline - time.monotonic()),
            )
            try:
                writer_ok = writer.flush(timeout_s=writer_timeout_s)
            except Exception:
                logger.exception("Failed to flush long-term %s writer.", label)
                writer_ok = False
            flush_ok = flush_ok and writer_ok
        return flush_ok

    def _refresh_restart_recall_packets_locked(self) -> None:
        """Refresh persistent restart-recall packets from the current durable store."""

        if self.restart_recall_policy_compiler is None:
            return
        packets = self.restart_recall_policy_compiler.build_packets(
            objects=self._restart_recall_source_objects_locked(),
        )
        self.midterm_store.replace_packets_with_attribute(
            packets=packets,
            attribute_key="persistence_scope",
            attribute_value="restart_recall",
        )

    def _restart_recall_source_objects_locked(self):
        """Return restart-recall source objects without remote blob hydration.

        Required-remote runtime paths should stay on the shared query-first
        selector or the fresher same-process bridge snapshot.
        """

        bridge_loader = getattr(self.object_store, "_same_process_snapshot_bridge_objects", None)
        if callable(bridge_loader):
            bridge_objects = bridge_loader()
            if bridge_objects is not None:
                return tuple(bridge_objects)
        return tuple(select_restart_recall_source_objects(self.object_store))

    def shutdown(self, *, timeout_s: float = 2.0) -> None:
        """Request bounded shutdown for all configured background writers."""

        resolved_timeout_s = _coerce_timeout_s(timeout_s, default=2.0)
        if self.writer is not None:
            try:
                self.writer.shutdown(timeout_s=resolved_timeout_s)
            except Exception:
                logger.exception("Failed to shutdown long-term conversation writer cleanly.")
        if self.multimodal_writer is not None:
            try:
                self.multimodal_writer.shutdown(timeout_s=resolved_timeout_s)
            except Exception:
                logger.exception("Failed to shutdown long-term multimodal writer cleanly.")
