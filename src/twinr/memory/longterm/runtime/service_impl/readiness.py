"""Remote readiness orchestration for the long-term runtime service."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import time

from twinr.memory.longterm.runtime.health import LongTermRemoteHealthProbe, LongTermRemoteWarmCheck
from twinr.memory.longterm.storage.remote_state import (
    LongTermRemoteStatus,
    LongTermRemoteUnavailableError,
)

from ._typing import ServiceMixinBase
from .compat import LongTermRemoteReadinessResult, LongTermRemoteReadinessStep

_FAST_TOPIC_READINESS_QUERY = "configured runtime namespace current scope"
_DEFAULT_FAST_TOPIC_READINESS_TIMEOUT_S = 1.2
_DEFAULT_REMOTE_READ_TIMEOUT_S = 8.0
_DEFAULT_WATCHDOG_PROBE_TIMEOUT_S = 15.0
_DEFAULT_WATCHDOG_STARTUP_PROBE_TIMEOUT_S = 45.0
_LOCAL_COOLDOWN_DETAIL = "remote long-term memory is temporarily cooling down after recent failures."
_SHALLOW_STATUS_READY_OVERRIDE_DETAIL = "query_surface_ready_despite_instance_flag_false"


class LongTermMemoryServiceReadinessMixin(ServiceMixinBase):
    """Fail-closed remote readiness checks and cache helpers."""

    def probe_remote_ready(
        self,
        *,
        bootstrap: bool = True,
        include_archive: bool = True,
        external_attestation_probe: bool = False,
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
        status_ready = bool(getattr(status, "ready", False))
        operational_probe_allowed = bool(
            getattr(status, "operational_probe_allowed", status_ready)
        )
        steps.append(
            LongTermRemoteReadinessStep(
                name="remote_status",
                status="ok" if status_ready else ("warn" if operational_probe_allowed else "fail"),
                latency_ms=round(max(0.0, (time.monotonic() - status_started) * 1000.0), 3),
                detail=status.detail,
            )
        )
        if (
            external_attestation_probe
            and not operational_probe_allowed
            and self._is_local_cooldown_status(status)
        ):
            attestation_started = time.monotonic()
            self.attest_external_remote_ready()
            steps.append(
                LongTermRemoteReadinessStep(
                    name="remote_status_external_attestation_reset",
                    status="ok",
                    latency_ms=round(max(0.0, (time.monotonic() - attestation_started) * 1000.0), 3),
                    detail="Cleared local remote cooldown before an external readiness recovery probe.",
                )
            )
            status_started = time.monotonic()
            status = remote_state.status()
            status_ready = bool(getattr(status, "ready", False))
            operational_probe_allowed = bool(
                getattr(status, "operational_probe_allowed", status_ready)
            )
            steps.append(
                LongTermRemoteReadinessStep(
                    name="remote_status_post_attestation",
                    status="ok" if status_ready else ("warn" if operational_probe_allowed else "fail"),
                    latency_ms=round(max(0.0, (time.monotonic() - status_started) * 1000.0), 3),
                    detail=status.detail,
                )
            )
        if not operational_probe_allowed:
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
                if warm_result.ready and self._fast_topic_readiness_required():
                    warm_result, fast_topic_step = self._probe_fast_topic_readiness(
                        warm_result=warm_result,
                        bootstrap=bootstrap,
                    )
                    steps.append(fast_topic_step)
                    if not warm_result.ready:
                        return LongTermRemoteReadinessResult(
                            ready=False,
                            detail=warm_result.detail,
                            remote_status=status,
                            steps=tuple(steps),
                            warm_result=warm_result,
                            total_latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
                        )
        deep_ready_override = self._strong_required_remote_attestation(
            warm_result,
            include_archive=include_archive,
        ) and operational_probe_allowed and not status_ready
        final_ready = bool(warm_result.ready and (status_ready or deep_ready_override))
        final_detail = warm_result.detail
        if deep_ready_override:
            final_detail = self._shallow_status_ready_override_detail(status)
        elif not status_ready:
            final_detail = status.detail or final_detail or "Remote-primary long-term memory is not ready."
        result = LongTermRemoteReadinessResult(
            ready=final_ready,
            detail=final_detail,
            remote_status=status,
            steps=tuple(steps),
            warm_result=warm_result,
            total_latency_ms=round(max(0.0, (time.monotonic() - started) * 1000.0), 3),
        )
        if result.ready:
            self.attest_external_remote_ready()
        return result

    @staticmethod
    def _strong_required_remote_attestation(
        warm_result,
        *,
        include_archive: bool,
    ) -> bool:
        """Return whether one deep readiness result is stronger than `/instance`.

        Live ChonkyDB incidents on 2026-04-11 showed the query/readiness
        surface becoming fully archive-safe before `/v1/external/instance`
        flipped its shallow `ready` bit. When that stronger archive-inclusive
        proof is present, runtime gating must honor it instead of re-failing on
        the weaker shallow flag. The same logic also applies to explicit
        current-only readiness checks: if the configured current-scope
        contract is already green, the lagging shallow bit must not veto it.
        """

        if not bool(getattr(warm_result, "ready", False)):
            return False
        if include_archive and not bool(getattr(warm_result, "archive_safe", False)):
            return False
        health_tier = str(getattr(warm_result, "health_tier", "") or "").strip().lower()
        if include_archive:
            return not health_tier or health_tier == "ready"
        return not health_tier or health_tier == "ready"

    @staticmethod
    def _shallow_status_ready_override_detail(status: LongTermRemoteStatus) -> str:
        """Summarize when deep readiness overrules a shallow unready status."""

        detail = str(getattr(status, "detail", "") or "").strip()
        if detail:
            return f"{_SHALLOW_STATUS_READY_OVERRIDE_DETAIL}: {detail}"
        return _SHALLOW_STATUS_READY_OVERRIDE_DETAIL

    @staticmethod
    def _is_local_cooldown_status(status: LongTermRemoteStatus) -> bool:
        """Return whether one shallow status reflects only the local cooldown gate."""

        detail = str(getattr(status, "detail", "") or "").strip().lower()
        return detail == _LOCAL_COOLDOWN_DETAIL

    def _fast_topic_readiness_required(self) -> bool:
        """Return whether the fast-topic runtime lane must be proved as healthy."""

        return bool(
            self.remote_required()
            and getattr(self.config, "long_term_memory_fast_topic_enabled", True)
        )

    def _fast_topic_readiness_timeout_s(self, *, bootstrap: bool) -> float:
        """Return the bounded fast-topic timeout used only for readiness proofs.

        Runtime answer turns keep using ``long_term_memory_fast_topic_timeout_s``.
        Required-remote readiness is a different contract: it must prove the live
        current-scope top-k route with the normal remote-read budget, while still
        respecting the matching watchdog probe ceiling for the current phase.
        """

        interactive_timeout_s = self._coerce_positive_timeout_s(
            getattr(
                self.config,
                "long_term_memory_fast_topic_timeout_s",
                _DEFAULT_FAST_TOPIC_READINESS_TIMEOUT_S,
            ),
            default=_DEFAULT_FAST_TOPIC_READINESS_TIMEOUT_S,
        )
        remote_read_timeout_s = self._coerce_positive_timeout_s(
            getattr(
                self.config,
                "long_term_memory_remote_read_timeout_s",
                _DEFAULT_REMOTE_READ_TIMEOUT_S,
            ),
            default=_DEFAULT_REMOTE_READ_TIMEOUT_S,
        )
        watchdog_timeout_s = self._coerce_positive_timeout_s(
            getattr(
                self.config,
                (
                    "long_term_memory_remote_watchdog_startup_probe_timeout_s"
                    if bootstrap
                    else "long_term_memory_remote_watchdog_probe_timeout_s"
                ),
                (
                    _DEFAULT_WATCHDOG_STARTUP_PROBE_TIMEOUT_S
                    if bootstrap
                    else _DEFAULT_WATCHDOG_PROBE_TIMEOUT_S
                ),
            ),
            default=(
                _DEFAULT_WATCHDOG_STARTUP_PROBE_TIMEOUT_S
                if bootstrap
                else _DEFAULT_WATCHDOG_PROBE_TIMEOUT_S
            ),
        )
        return max(interactive_timeout_s, min(remote_read_timeout_s, watchdog_timeout_s))

    @staticmethod
    def _coerce_positive_timeout_s(value: object, *, default: float) -> float:
        """Resolve one config-like timeout to a positive float."""

        if isinstance(value, bool):
            resolved = float(value)
        elif isinstance(value, (int, float, str)):
            try:
                resolved = float(value)
            except ValueError:
                return default
        else:
            return default
        if resolved <= 0.0:
            return default
        return resolved

    def _probe_fast_topic_readiness(
        self,
        *,
        warm_result,
        bootstrap: bool,
    ):
        """Prove the current-scope fast-topic retrieval route before attesting readiness."""

        step_name = "object_store.select_fast_topic_objects_readiness"
        started = time.monotonic()
        selector = getattr(self.object_store, "select_fast_topic_objects", None)
        if not callable(selector):
            detail = "Required remote long-term fast-topic selector is unavailable."
            return self._build_fast_topic_readiness_failure(
                warm_result=warm_result,
                detail=detail,
                latency_ms=0.0,
                step_name=step_name,
            )
        try:
            selector(
                query_text=_FAST_TOPIC_READINESS_QUERY,
                limit=1,
                timeout_s=self._fast_topic_readiness_timeout_s(bootstrap=bootstrap),
            )
        except Exception as exc:
            latency_ms = round(max(0.0, (time.monotonic() - started) * 1000.0), 3)
            detail = f"{type(exc).__name__}: {exc}"
            return self._build_fast_topic_readiness_failure(
                warm_result=warm_result,
                detail=detail,
                latency_ms=latency_ms,
                step_name=step_name,
            )
        latency_ms = round(max(0.0, (time.monotonic() - started) * 1000.0), 3)
        updated = replace(
            warm_result,
            checks=warm_result.checks
            + (
                LongTermRemoteWarmCheck(
                    store="object_store",
                    snapshot_kind="fast_topic_route",
                    status="ok",
                    latency_ms=latency_ms,
                    detail="Configured fast-topic retrieval route responded.",
                    selected_source="current_scope_topk_contract",
                ),
            ),
            fast_topic_checked=True,
            fast_topic_ready=True,
        )
        return updated, LongTermRemoteReadinessStep(
            name=step_name,
            status="ok",
            latency_ms=latency_ms,
            detail="Configured fast-topic retrieval route responded.",
        )

    @staticmethod
    def _build_fast_topic_readiness_failure(
        *,
        warm_result,
        detail: str,
        latency_ms: float,
        step_name: str,
    ):
        updated = replace(
            warm_result,
            ready=False,
            detail=detail,
            failed_store="object_store",
            failed_snapshot_kind="fast_topic_route",
            checks=warm_result.checks
            + (
                LongTermRemoteWarmCheck(
                    store="object_store",
                    snapshot_kind="fast_topic_route",
                    status="unavailable",
                    latency_ms=latency_ms,
                    detail=detail,
                    selected_source="current_scope_topk_contract",
                ),
            ),
            health_tier="hard_down",
            fast_topic_checked=True,
            fast_topic_ready=False,
        )
        return updated, LongTermRemoteReadinessStep(
            name=step_name,
            status="fail",
            latency_ms=latency_ms,
            detail=detail,
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
