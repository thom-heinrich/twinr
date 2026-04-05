"""Remote readiness orchestration for the long-term runtime service."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
import time

from twinr.memory.longterm.runtime.health import LongTermRemoteHealthProbe
from twinr.memory.longterm.storage.remote_state import (
    LongTermRemoteStatus,
    LongTermRemoteUnavailableError,
)

from ._typing import ServiceMixinBase
from .compat import LongTermRemoteReadinessResult, LongTermRemoteReadinessStep


class LongTermMemoryServiceReadinessMixin(ServiceMixinBase):
    """Fail-closed remote readiness checks and cache helpers."""

    def probe_remote_ready(
        self,
        *,
        bootstrap: bool = True,
        include_archive: bool = True,
    ) -> LongTermRemoteReadinessResult:
        """Return structured required-remote readiness evidence for runtime use."""

        remote_state = getattr(self.prompt_context_store.memory_store, "remote_state", None)
        if remote_state is None or not remote_state.enabled:
            return LongTermRemoteReadinessResult(
                ready=False,
                detail="Remote-primary long-term memory is disabled.",
                remote_status=LongTermRemoteStatus(mode="disabled", ready=False),
                steps=(),
                total_latency_ms=0.0,
            )
        steps: list[LongTermRemoteReadinessStep] = []
        started = time.monotonic()
        status_started = time.monotonic()
        status = remote_state.status()
        steps.append(
            LongTermRemoteReadinessStep(
                name="remote_status",
                status="ok" if status.ready else "fail",
                latency_ms=round(max(0.0, (time.monotonic() - status_started) * 1000.0), 3),
                detail=status.detail,
            )
        )
        if not status.ready:
            return LongTermRemoteReadinessResult(
                ready=False,
                detail=status.detail or "Remote-primary long-term memory is not ready.",
                remote_status=status,
                steps=tuple(steps),
                total_latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
            )
        with self._store_lock:
            with self._cache_remote_probe_reads():
                if bootstrap:
                    graph_bootstrap = getattr(self.graph_store, "ensure_remote_snapshot_for_readiness", None)
                    if not callable(graph_bootstrap):
                        graph_bootstrap = self.graph_store.ensure_remote_snapshot
                    object_bootstrap = getattr(self.object_store, "ensure_remote_snapshots_for_readiness", None)
                    if not callable(object_bootstrap):
                        object_bootstrap = self.object_store.ensure_remote_snapshots
                    midterm_bootstrap = getattr(self.midterm_store, "ensure_remote_snapshot_for_readiness", None)
                    if not callable(midterm_bootstrap):
                        midterm_bootstrap = self.midterm_store.ensure_remote_snapshot
                    for step_name, callback in (
                        ("prompt_context_store.ensure_remote_snapshots", self.prompt_context_store.ensure_remote_snapshots),
                        ("graph_store.ensure_remote_snapshot_for_readiness", graph_bootstrap),
                        ("object_store.ensure_remote_snapshots_for_readiness", object_bootstrap),
                        ("midterm_store.ensure_remote_snapshot_for_readiness", midterm_bootstrap),
                    ):
                        step_started = time.monotonic()
                        try:
                            callback()
                        except Exception as exc:
                            steps.append(
                                LongTermRemoteReadinessStep(
                                    name=step_name,
                                    status="fail",
                                    latency_ms=round(max(0.0, (time.monotonic() - step_started) * 1000.0), 3),
                                    detail=f"{type(exc).__name__}: {exc}",
                                )
                            )
                            return LongTermRemoteReadinessResult(
                                ready=False,
                                detail=f"{type(exc).__name__}: {exc}",
                                remote_status=status,
                                steps=tuple(steps),
                                total_latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
                            )
                        steps.append(
                            LongTermRemoteReadinessStep(
                                name=step_name,
                                status="ok",
                                latency_ms=round(max(0.0, (time.monotonic() - step_started) * 1000.0), 3),
                            )
                        )
                warm_started = time.monotonic()
                warm_result = LongTermRemoteHealthProbe(
                    prompt_context_store=self.prompt_context_store,
                    object_store=self.object_store,
                    graph_store=self.graph_store,
                    midterm_store=self.midterm_store,
                ).probe_operational(include_archive=include_archive)
                steps.append(
                    LongTermRemoteReadinessStep(
                        name=(
                            "LongTermRemoteHealthProbe.probe_operational"
                            if bootstrap and include_archive
                            else "LongTermRemoteHealthProbe.probe_operational_steady_state"
                        ),
                        status="ok" if warm_result.ready else "fail",
                        latency_ms=round(max(0.0, (time.monotonic() - warm_started) * 1000.0), 3),
                        detail=warm_result.detail,
                        warm_result=warm_result,
                    )
                )
        return LongTermRemoteReadinessResult(
            ready=warm_result.ready,
            detail=warm_result.detail,
            remote_status=status,
            steps=tuple(steps),
            warm_result=warm_result,
            total_latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
        )

    @contextmanager
    def _cache_remote_probe_reads(self) -> Iterator[None]:
        """Reuse snapshot probes only within one bounded readiness pass."""

        with ExitStack() as stack:
            for remote_state in self._iter_unique_remote_states():
                cache_probe_reads = getattr(remote_state, "cache_probe_reads", None)
                if callable(cache_probe_reads):
                    stack.enter_context(cache_probe_reads())
            yield

    def _iter_unique_remote_states(self) -> Iterator[object]:
        """Yield each owned remote-state adapter at most once."""

        seen: set[int] = set()
        for remote_state in (
            getattr(self.prompt_context_store.memory_store, "remote_state", None),
            getattr(self.prompt_context_store.user_store, "remote_state", None),
            getattr(self.prompt_context_store.personality_store, "remote_state", None),
            getattr(self.graph_store, "remote_state", None),
            getattr(self.object_store, "remote_state", None),
            getattr(self.midterm_store, "remote_state", None),
        ):
            if remote_state is None:
                continue
            state_id = id(remote_state)
            if state_id in seen:
                continue
            seen.add(state_id)
            yield remote_state

    def attest_external_remote_ready(self) -> None:
        """Synchronize local remote-state cooldowns to an external proof."""

        for remote_state in self._iter_unique_remote_states():
            attest_external_readiness = getattr(remote_state, "attest_external_readiness", None)
            if callable(attest_external_readiness):
                attest_external_readiness()

    def ensure_remote_ready(self) -> LongTermRemoteReadinessResult:
        """Prove required remote-primary long-term state is ready to use."""

        result = self.probe_remote_ready()
        if result.ready:
            return result
        if self.remote_required():
            raise LongTermRemoteUnavailableError(
                result.detail or "Remote-primary long-term memory is not ready."
            )
        return result

    def remote_required(self) -> bool:
        """Report whether runtime callers must hard-fail on remote loss."""

        remote_state = getattr(self.prompt_context_store.memory_store, "remote_state", None)
        if remote_state is None:
            return bool(
                self.config.long_term_memory_enabled
                and self.config.long_term_memory_mode == "remote_primary"
                and self.config.long_term_memory_remote_required
            )
        return bool(getattr(remote_state, "required", False))

    def remote_status(self) -> LongTermRemoteStatus:
        """Return the effective long-term remote-state status for runtime use."""

        remote_state = getattr(self.prompt_context_store.memory_store, "remote_state", None)
        if remote_state is None:
            if self.config.long_term_memory_enabled and self.config.long_term_memory_mode == "remote_primary":
                return LongTermRemoteStatus(
                    mode="remote_primary",
                    ready=False,
                    detail="Remote long-term memory state is not configured.",
                )
            return LongTermRemoteStatus(mode="disabled", ready=False)
        return remote_state.status()

    @contextmanager
    def _temporary_remote_probe_cache(self) -> Iterator[None]:
        """Reuse identical remote snapshot probes within one foreground turn."""

        with self._cache_remote_probe_reads():
            yield
