"""Assemble runtime-facing long-term memory operations.

This module wires together extraction, consolidation, retrieval, retention,
proactive planning, remote readiness checks, and bounded background writers
behind one service used by Twinr runtime loops. Lower-level reasoning,
ingestion, and storage logic stays in their dedicated packages; this module
coordinates those pieces without owning their algorithms.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field  # AUDIT-FIX(#10): keep 3.11-safe dataclass field support for bounded defaults and locks.
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any  # AUDIT-FIX(#12): make fallback return types explicit and 3.11-safe.
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError  # AUDIT-FIX(#9): normalize naive/aware datetimes into the configured local timezone.
import json
import logging
import math
import threading
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.ingestion.backfill import (
    LongTermOpsBackfillRunResult,
    LongTermOpsEventBackfiller,
)
from twinr.memory.context_store import ManagedContextEntry, PersistentMemoryEntry, PromptContextStore
from twinr.memory.chonkydb.personal_graph import TwinrPersonalGraphStore
from twinr.memory.query_normalization import LongTermQueryRewriter
from twinr.memory.longterm.reasoning.consolidator import LongTermMemoryConsolidator
from twinr.memory.longterm.reasoning.conflicts import LongTermConflictResolver
from twinr.memory.longterm.ingestion.extract import LongTermTurnExtractor
from twinr.memory.longterm.ingestion.multimodal import LongTermMultimodalExtractor
from twinr.memory.longterm.core.ontology import kind_matches
from twinr.memory.longterm.core.models import (
    LongTermConsolidationResultV1,
    LongTermConflictQueueItemV1,
    LongTermConflictResolutionV1,
    LongTermConversationTurn,
    LongTermEnqueueResult,
    LongTermMemoryContext,
    LongTermMemoryMutationResultV1,
    LongTermMemoryReviewResultV1,
    LongTermMultimodalEvidence,
    LongTermProactiveCandidateV1,  # AUDIT-FIX(#5): import the proactive candidate model used in public type hints.
    LongTermProactivePlanV1,
    LongTermRetentionResultV1,
    LongTermReflectionResultV1,
)
from twinr.memory.longterm.storage.midterm_store import LongTermMidtermStore
from twinr.memory.longterm.proactive.planner import LongTermProactivePlanner
from twinr.memory.longterm.proactive.state import LongTermProactivePolicy, LongTermProactiveReservationV1, LongTermProactiveStateStore
from twinr.memory.longterm.reasoning.reflect import LongTermMemoryReflector
from twinr.memory.longterm.retrieval.adaptive_policy import LongTermAdaptivePolicyBuilder
from twinr.memory.longterm.retrieval.restart_recall_policy import LongTermRestartRecallPolicyCompiler
from twinr.memory.longterm.retrieval.retriever import LongTermRetriever
from twinr.memory.longterm.reasoning.retention import LongTermRetentionPolicy
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStatus, LongTermRemoteUnavailableError
from twinr.memory.longterm.ingestion.sensor_memory import LongTermSensorMemoryCompiler
from twinr.memory.longterm.storage.store import LongTermStructuredStore
from twinr.memory.longterm.retrieval.subtext import LongTermSubtextBuilder, LongTermSubtextCompiler
from twinr.memory.longterm.reasoning.truth import LongTermTruthMaintainer
from twinr.memory.longterm.runtime.flush_budget import build_flush_budget_plan
from twinr.memory.longterm.runtime.health import LongTermRemoteHealthProbe, LongTermRemoteWarmResult
from twinr.memory.longterm.runtime.worker import AsyncLongTermMemoryWriter, AsyncLongTermMultimodalWriter


logger = logging.getLogger(__name__)

_TEXT_LIMIT = 320
_SOURCE_LIMIT = 64
_MULTIMODAL_NAME_LIMIT = 128
_MULTIMODAL_MESSAGE_LIMIT = 512
_JSON_STRING_LIMIT = 512
_JSON_ITEMS_LIMIT = 64
_JSON_DEPTH_LIMIT = 5
_DEFAULT_REVIEW_LIMIT = 12
_MAX_REVIEW_LIMIT = 100
_MAX_QUEUE_SIZE = 4096


# AUDIT-FIX(#10): harden text normalization against zero/negative limits and silent off-by-one truncation.
def _normalize_text(value: object, *, limit: int) -> str:
    """Normalize arbitrary input into one bounded single-line string."""

    if limit <= 0:
        return ""
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    if limit == 1:
        return "…"
    return text[: limit - 1].rstrip() + "…"


# AUDIT-FIX(#10): clamp queue sizes and review limits to sane positive values on misconfigured .env input.
def _coerce_positive_int(value: object, *, default: int, maximum: int | None = None) -> int:
    """Coerce config-like input to a positive integer within bounds."""

    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    if result <= 0:
        result = default
    if maximum is not None:
        result = min(result, maximum)
    return result


def _coerce_timeout_s(value: object, *, default: float) -> float:
    """Coerce timeout input to finite non-negative seconds."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result) or result < 0:
        return default
    return result


@lru_cache(maxsize=16)
def _load_timezone(timezone_name: str) -> ZoneInfo:
    """Load and cache one timezone, falling back to UTC on unknown names."""

    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        logger.warning("Unknown timezone %r; falling back to UTC.", timezone_name)
        return ZoneInfo("UTC")


# AUDIT-FIX(#9): convert public datetime inputs to timezone-aware local datetimes before scheduling or retention logic.
def _normalize_datetime(value: datetime | None, *, timezone_name: str) -> datetime | None:
    """Normalize a datetime into the configured local timezone."""

    if value is None:
        return None
    timezone = _load_timezone(timezone_name)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone)
    return value.astimezone(timezone)


def _serialize_datetime(value: datetime, *, timezone_name: str) -> str:
    """Serialize one datetime after timezone normalization."""

    normalized = _normalize_datetime(value, timezone_name=timezone_name)
    if normalized is None:
        return ""
    return normalized.isoformat()


# AUDIT-FIX(#6): bound and sanitize multimodal payloads so malformed sensor data cannot poison background persistence.
def _sanitize_jsonish(
    value: Any,
    *,
    timezone_name: str,
    depth: int = 0,
) -> object:
    """Convert nested sensor payloads into bounded JSON-safe values."""

    if depth >= _JSON_DEPTH_LIMIT:
        return _normalize_text(repr(value), limit=_JSON_STRING_LIMIT)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _normalize_text(value, limit=_JSON_STRING_LIMIT)
    if isinstance(value, datetime):
        return _serialize_datetime(value, timezone_name=timezone_name)
    if isinstance(value, bytes):
        return {"__bytes_len__": len(value)}
    if isinstance(value, Mapping):
        sanitized: dict[str, object] = {}
        for index, (key, nested_value) in enumerate(value.items()):
            if index >= _JSON_ITEMS_LIMIT:
                sanitized["__truncated__"] = True
                break
            sanitized[_normalize_text(key, limit=64) or f"key_{index}"] = _sanitize_jsonish(
                nested_value,
                timezone_name=timezone_name,
                depth=depth + 1,
            )
        return sanitized
    if isinstance(value, Iterable):
        sanitized_items: list[object] = []
        for index, item in enumerate(value):
            if index >= _JSON_ITEMS_LIMIT:
                sanitized_items.append("…")
                break
            sanitized_items.append(
                _sanitize_jsonish(
                    item,
                    timezone_name=timezone_name,
                    depth=depth + 1,
                )
            )
        return sanitized_items
    return _normalize_text(repr(value), limit=_JSON_STRING_LIMIT)


# AUDIT-FIX(#7): reject unsafe config-backed file paths used for backfill and local fallback store recreation.
def _validate_regular_file_path(path_value: object, *, allow_missing: bool) -> Path | None:
    """Accept only regular, non-symlink file paths for local file access."""

    try:
        path = Path(path_value).expanduser()
    except (TypeError, ValueError):
        return None
    try:
        if path.exists():
            if path.is_symlink() or not path.is_file():
                return None
        elif not allow_missing:
            return None
        parent = path.parent
        if parent.exists():
            if parent.is_symlink() or not parent.is_dir():
                return None
    except OSError:
        return None
    return path


def _sort_objects_by_memory_id(items: Iterable[Any]) -> tuple[Any, ...]:
    """Return memory objects sorted by stable memory ID."""

    return tuple(sorted(items, key=lambda row: row.memory_id))


def _sort_conflicts(items: Iterable[Any]) -> tuple[Any, ...]:
    """Return conflict objects sorted by slot and candidate memory ID."""

    return tuple(sorted(items, key=lambda row: (row.slot_key, row.candidate_memory_id)))


@dataclass(frozen=True, slots=True)
class LongTermRemoteReadinessStep:
    """Capture one readiness step executed by the runtime service."""

    name: str
    status: str
    latency_ms: float
    detail: str | None = None
    warm_result: LongTermRemoteWarmResult | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe summary for ops artifacts."""

        payload: dict[str, object] = {
            "name": self.name,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "detail": self.detail,
        }
        if self.warm_result is not None:
            payload["warm_result"] = self.warm_result.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class LongTermRemoteReadinessResult:
    """Capture the full fail-closed remote readiness proof for one probe."""

    ready: bool
    detail: str | None
    remote_status: LongTermRemoteStatus
    steps: tuple[LongTermRemoteReadinessStep, ...]
    warm_result: LongTermRemoteWarmResult | None = None
    total_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe summary for watchdog artifacts and debugging."""

        payload: dict[str, object] = {
            "ready": self.ready,
            "detail": self.detail,
            "remote_status": {
                "mode": self.remote_status.mode,
                "ready": self.remote_status.ready,
                "detail": self.remote_status.detail,
            },
            "steps": [step.to_dict() for step in self.steps],
            "total_latency_ms": self.total_latency_ms,
        }
        if self.warm_result is not None:
            payload["warm_result"] = self.warm_result.to_dict()
        return payload


@dataclass(slots=True)
class LongTermMemoryService:
    """Coordinate Twinr long-term memory runtime flows.

    Runtime loops should call this service to build provider context, persist
    conversation turns and multimodal evidence, run maintenance jobs, and
    mutate operator-facing long-term memory state. It centralizes orchestration
    while keeping extraction, reasoning, proactive policy, and storage details
    in their dedicated modules.
    """

    config: TwinrConfig
    prompt_context_store: PromptContextStore
    graph_store: TwinrPersonalGraphStore
    object_store: LongTermStructuredStore
    midterm_store: LongTermMidtermStore
    query_rewriter: LongTermQueryRewriter
    retriever: LongTermRetriever
    extractor: LongTermTurnExtractor
    multimodal_extractor: LongTermMultimodalExtractor
    truth_maintainer: LongTermTruthMaintainer
    consolidator: LongTermMemoryConsolidator
    conflict_resolver: LongTermConflictResolver
    reflector: LongTermMemoryReflector
    sensor_memory: LongTermSensorMemoryCompiler
    ops_backfiller: LongTermOpsEventBackfiller
    planner: LongTermProactivePlanner
    proactive_policy: LongTermProactivePolicy
    retention_policy: LongTermRetentionPolicy
    restart_recall_policy_compiler: LongTermRestartRecallPolicyCompiler | None = None
    writer: AsyncLongTermMemoryWriter | None = None
    multimodal_writer: AsyncLongTermMultimodalWriter | None = None
    _store_lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)  # AUDIT-FIX(#1): serialize file-backed state mutations across foreground calls and background workers.

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        graph_store: TwinrPersonalGraphStore | None = None,
        prompt_context_store: PromptContextStore | None = None,
        extractor: LongTermTurnExtractor | None = None,
    ) -> "LongTermMemoryService":
        """Build the runtime service and its bounded background writers.

        Args:
            config: Runtime configuration used for stores, policies, and
                timezone handling.
            graph_store: Optional prebuilt personal graph store.
            prompt_context_store: Optional prebuilt prompt context store.
            extractor: Optional prebuilt conversation-turn extractor.

        Returns:
            A fully wired long-term memory service ready for runtime use.
        """

        store = prompt_context_store or PromptContextStore.from_config(config)
        graph = graph_store or TwinrPersonalGraphStore.from_config(config)
        object_store = LongTermStructuredStore.from_config(config)
        midterm_store = LongTermMidtermStore.from_config(config)
        extractor = extractor or LongTermTurnExtractor.from_config(config)
        multimodal_extractor = LongTermMultimodalExtractor(timezone_name=config.local_timezone_name)
        truth_maintainer = LongTermTruthMaintainer()
        consolidator = LongTermMemoryConsolidator(truth_maintainer=truth_maintainer)
        conflict_resolver = LongTermConflictResolver()
        reflector = LongTermMemoryReflector.from_config(config)
        sensor_memory = LongTermSensorMemoryCompiler.from_config(config)
        ops_backfiller = LongTermOpsEventBackfiller()
        planner = LongTermProactivePlanner(timezone_name=config.local_timezone_name)
        proactive_state_store = LongTermProactiveStateStore.from_config(config)
        proactive_policy = LongTermProactivePolicy(
            config=config,
            state_store=proactive_state_store,
        )
        retention_policy = LongTermRetentionPolicy(
            timezone_name=config.local_timezone_name,
            mode=config.long_term_memory_retention_mode,
            archive_enabled=config.long_term_memory_archive_enabled,
        )
        subtext_builder = LongTermSubtextBuilder(
            config=config,
            graph_store=graph,
            compiler=LongTermSubtextCompiler.from_config(config),
        )
        retriever = LongTermRetriever(
            config=config,
            prompt_context_store=store,
            graph_store=graph,
            object_store=object_store,
            midterm_store=midterm_store,
            conflict_resolver=conflict_resolver,
            subtext_builder=subtext_builder,
            adaptive_policy_builder=LongTermAdaptivePolicyBuilder(
                proactive_state_store=proactive_state_store,
            ),
        )
        store_lock = threading.RLock()  # AUDIT-FIX(#1): share one re-entrant mutation lock with both background writers.
        # AUDIT-FIX(#10): guard startup against zero/negative queue sizes from configuration.
        queue_size = _coerce_positive_int(
            getattr(config, "long_term_memory_write_queue_size", 1),
            default=1,
            maximum=_MAX_QUEUE_SIZE,
        )
        writer: AsyncLongTermMemoryWriter | None = None
        multimodal_writer: AsyncLongTermMultimodalWriter | None = None
        if config.long_term_memory_enabled and config.long_term_memory_background_store_turns:
            writer = AsyncLongTermMemoryWriter(
                write_callback=lambda item: cls._persist_longterm_turn(
                    config=config,
                    store=store,
                    graph_store=graph,
                    object_store=object_store,
                    midterm_store=midterm_store,
                    extractor=extractor,
                    consolidator=consolidator,
                    reflector=reflector,
                    sensor_memory=sensor_memory,
                    retention_policy=retention_policy,
                    store_lock=store_lock,
                    timezone_name=config.local_timezone_name,
                    item=item,
                ),
                max_queue_size=queue_size,
            )
            multimodal_writer = AsyncLongTermMultimodalWriter(
                write_callback=lambda item: cls._persist_multimodal_evidence(
                    object_store=object_store,
                    midterm_store=midterm_store,
                    multimodal_extractor=multimodal_extractor,
                    consolidator=consolidator,
                    reflector=reflector,
                    sensor_memory=sensor_memory,
                    retention_policy=retention_policy,
                    store_lock=store_lock,
                    timezone_name=config.local_timezone_name,
                    item=item,
                ),
                max_queue_size=queue_size,
            )
        return cls(
            config=config,
            prompt_context_store=store,
            graph_store=graph,
            object_store=object_store,
            midterm_store=midterm_store,
            query_rewriter=LongTermQueryRewriter.from_config(config),
            retriever=retriever,
            extractor=extractor,
            multimodal_extractor=multimodal_extractor,
            truth_maintainer=truth_maintainer,
            consolidator=consolidator,
            conflict_resolver=conflict_resolver,
            reflector=reflector,
            sensor_memory=sensor_memory,
            ops_backfiller=ops_backfiller,
            planner=planner,
            proactive_policy=proactive_policy,
            retention_policy=retention_policy,
            restart_recall_policy_compiler=LongTermRestartRecallPolicyCompiler(),
            writer=writer,
            multimodal_writer=multimodal_writer,
            _store_lock=store_lock,
        )

    def probe_remote_ready(self) -> LongTermRemoteReadinessResult:
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
                for step_name, callback in (
                    ("prompt_context_store.ensure_remote_snapshots", self.prompt_context_store.ensure_remote_snapshots),
                    ("graph_store.ensure_remote_snapshot", self.graph_store.ensure_remote_snapshot),
                    ("object_store.ensure_remote_snapshots", self.object_store.ensure_remote_snapshots),
                    ("midterm_store.ensure_remote_snapshot", self.midterm_store.ensure_remote_snapshot),
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
                ).probe_operational()
                steps.append(
                    LongTermRemoteReadinessStep(
                        name="LongTermRemoteHealthProbe.probe_operational",
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

        seen: set[int] = set()
        with ExitStack() as stack:
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
                cache_probe_reads = getattr(remote_state, "cache_probe_reads", None)
                if callable(cache_probe_reads):
                    stack.enter_context(cache_probe_reads())
            yield

    def ensure_remote_ready(self) -> LongTermRemoteReadinessResult:
        """Prove required remote-primary long-term state is ready to use.

        Raises:
            LongTermRemoteUnavailableError: If required remote state or any
                required snapshot is unavailable.
        """

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

    def build_provider_context(self, query_text: str | None) -> LongTermMemoryContext:
        """Build the normal long-term context injected into provider prompts.

        Args:
            query_text: User query text used for retrieval profiling.

        Returns:
            The best-effort long-term context for provider prompts. Returns an
            empty context if non-remote retrieval fails unexpectedly.

        Raises:
            LongTermRemoteUnavailableError: If required remote-primary state is
                unavailable.
        """

        try:
            query = self.query_rewriter.profile(query_text)
            with self._store_lock:  # AUDIT-FIX(#1): serialize shared file-backed reads and writes against concurrent background workers.
                return self.retriever.build_context(
                    query=query,
                    original_query_text=query_text,
                )
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Failed to build long-term provider context.")  # AUDIT-FIX(#11): degrade to empty context instead of crashing the turn.
            return LongTermMemoryContext()

    def build_tool_provider_context(self, query_text: str | None) -> LongTermMemoryContext:
        """Build a tool-facing context with sensitive details redacted.

        Args:
            query_text: User query text used for retrieval profiling.

        Returns:
            A best-effort context for tool-calling prompts with conflicting
            memories and contact methods filtered out.

        Raises:
            LongTermRemoteUnavailableError: If required remote-primary state is
                unavailable.
        """

        try:
            query = self.query_rewriter.profile(query_text)
            with self._store_lock:  # AUDIT-FIX(#1): serialize shared file-backed reads and writes against concurrent background workers.
                context = self.retriever.build_context(
                    query=query,
                    original_query_text=query_text,
                )
                recall_limit = max(
                    1,
                    _coerce_positive_int(
                        getattr(self.config, "long_term_memory_recall_limit", 1),
                        default=1,
                        maximum=_MAX_REVIEW_LIMIT,
                    ),
                )
                conflict_queue = self.retriever.select_conflict_queue(
                    query=query,
                    limit=recall_limit,
                )
                conflicting_memory_ids = {
                    option.memory_id
                    for item in conflict_queue
                    for option in item.options
                }
                durable_objects = self.retriever.select_durable_objects(
                    query=query,
                    limit=recall_limit,
                )
                filtered_durable_objects = tuple(
                    item
                    for item in durable_objects
                    if not kind_matches(
                        item.kind,
                        "fact",
                        item.attributes,
                        attr_key="fact_type",
                        attr_value="contact_method",
                    )
                    and item.memory_id not in conflicting_memory_ids
                )
                return LongTermMemoryContext(
                    subtext_context=context.subtext_context,
                    midterm_context=context.midterm_context,
                    durable_context=self.retriever._render_durable_context(filtered_durable_objects),
                    episodic_context=context.episodic_context,
                    graph_context=self.graph_store.build_prompt_context(
                        query.retrieval_text or query.original_text,
                        include_contact_methods=False,
                    ),
                    conflict_context=None,
                )
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Failed to build tool-facing long-term provider context.")  # AUDIT-FIX(#11): tool context failure must not take down the request path.
            return LongTermMemoryContext()

    def enqueue_conversation_turn(
        self,
        *,
        transcript: str,
        response: str,
        source: str = "conversation",
    ) -> LongTermEnqueueResult | None:
        """Queue one conversation turn for bounded long-term persistence.

        Args:
            transcript: Normalized user transcript to persist.
            response: Assistant response paired with the transcript.
            source: Source label recorded with the turn.

        Returns:
            Queue admission metadata when a background writer is enabled.
            Returns None when the turn is empty, background writing is
            disabled, or synchronous fallback persistence was required.
        """

        clean_transcript = _normalize_text(transcript, limit=_TEXT_LIMIT)
        clean_response = _normalize_text(response, limit=_TEXT_LIMIT)
        clean_source = _normalize_text(source, limit=_SOURCE_LIMIT) or "conversation"
        if not clean_transcript or not clean_response:
            return None
        # AUDIT-FIX(#6): normalize source text early so queue fallback stores a bounded record.
        item = LongTermConversationTurn(
            transcript=clean_transcript,
            response=clean_response,
            source=clean_source,
        )
        if self.writer is None:
            return None
        try:
            return self.writer.enqueue(item)
        except Exception:
            logger.exception("Failed to enqueue conversation turn; persisting synchronously.")  # AUDIT-FIX(#6): queue failure should not silently drop a senior interaction.
            self._persist_longterm_turn(
                config=self.config,
                store=self.prompt_context_store,
                graph_store=self.graph_store,
                object_store=self.object_store,
                midterm_store=self.midterm_store,
                extractor=self.extractor,
                consolidator=self.consolidator,
                reflector=self.reflector,
                sensor_memory=self.sensor_memory,
                retention_policy=self.retention_policy,
                store_lock=self._store_lock,
                timezone_name=self.config.local_timezone_name,
                item=item,
            )
            return None

    def analyze_conversation_turn(
        self,
        *,
        transcript: str,
        response: str,
    ) -> LongTermConsolidationResultV1:
        """Run extraction and consolidation without mutating stored state."""

        extraction = self.extractor.extract_conversation_turn(
            transcript=transcript,
            response=response,
        )
        with self._store_lock:
            existing_objects = tuple(self.object_store.load_objects())
        return self.consolidator.consolidate(
            extraction=extraction,
            existing_objects=existing_objects,
        )

    def analyze_multimodal_evidence(
        self,
        *,
        event_name: str,
        modality: str,
        source: str = "device_event",
        message: str | None = None,
        data: dict[str, object] | None = None,
    ) -> LongTermConsolidationResultV1:
        """Run multimodal extraction and consolidation without persisting."""

        # AUDIT-FIX(#6): sanitize multimodal inputs before extraction so malformed payloads remain JSON-safe and bounded.
        evidence = self._build_multimodal_evidence(
            event_name=event_name,
            modality=modality,
            source=source,
            message=message,
            data=data,
        )
        extraction = self.multimodal_extractor.extract_evidence(evidence)
        with self._store_lock:
            existing_objects = tuple(self.object_store.load_objects())
        return self.consolidator.consolidate(
            extraction=extraction,
            existing_objects=existing_objects,
        )

    def enqueue_multimodal_evidence(
        self,
        *,
        event_name: str,
        modality: str,
        source: str = "device_event",
        message: str | None = None,
        data: dict[str, object] | None = None,
    ) -> LongTermEnqueueResult | None:
        """Queue one multimodal evidence item for bounded persistence.

        Returns:
            Queue admission metadata when a multimodal writer is enabled.
            Returns None when background writing is disabled or synchronous
            fallback persistence was required.
        """

        if self.multimodal_writer is None:
            return None
        # AUDIT-FIX(#6): sanitize multimodal inputs before extraction so malformed payloads remain JSON-safe and bounded.
        evidence = self._build_multimodal_evidence(
            event_name=event_name,
            modality=modality,
            source=source,
            message=message,
            data=data,
        )
        try:
            return self.multimodal_writer.enqueue(evidence)
        except Exception:
            logger.exception("Failed to enqueue multimodal evidence; persisting synchronously.")  # AUDIT-FIX(#6): keep device events durable even when the async queue is unavailable.
            self._persist_multimodal_evidence(
                object_store=self.object_store,
                midterm_store=self.midterm_store,
                multimodal_extractor=self.multimodal_extractor,
                consolidator=self.consolidator,
                reflector=self.reflector,
                sensor_memory=self.sensor_memory,
                retention_policy=self.retention_policy,
                store_lock=self._store_lock,
                timezone_name=self.config.local_timezone_name,
                item=evidence,
            )
            return None

    # AUDIT-FIX(#4): persist sensor-memory midterm packets everywhere reflection results are applied.
    def run_reflection(self) -> LongTermReflectionResultV1:
        """Run object reflection plus sensor-memory summaries over stored state.

        Returns:
            The combined reflection payload applied to object and midterm
            stores. Returns an empty result on non-remote failures.

        Raises:
            LongTermRemoteUnavailableError: If required remote-primary state is
                unavailable.
        """

        try:
            with self._store_lock:  # AUDIT-FIX(#1): serialize shared file-backed reads and writes against concurrent background workers.
                result = self.reflector.reflect(objects=self.object_store.load_objects())
                if self._has_reflection_payload(result):
                    self.object_store.apply_reflection(result)
                    self.midterm_store.apply_reflection(result)
                sensor_memory_result = self.sensor_memory.compile(objects=self.object_store.load_objects())
                if self._has_reflection_payload(sensor_memory_result):
                    self.object_store.apply_reflection(sensor_memory_result)
                    self.midterm_store.apply_reflection(sensor_memory_result)
                    return LongTermReflectionResultV1(
                        reflected_objects=tuple((*result.reflected_objects, *sensor_memory_result.reflected_objects)),
                        created_summaries=tuple((*result.created_summaries, *sensor_memory_result.created_summaries)),
                        midterm_packets=tuple((*result.midterm_packets, *sensor_memory_result.midterm_packets)),
                    )
                return result
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Long-term reflection failed.")
            return self._empty_reflection_result()

    # AUDIT-FIX(#4): sensor-only reflections now update both object and midterm stores.
    def run_sensor_memory(self, *, now: datetime | None = None) -> LongTermReflectionResultV1:
        """Compile sensor-memory summaries from current long-term objects.

        Args:
            now: Optional local-time reference for routine compilation.

        Returns:
            The applied sensor-memory reflection payload, or an empty result if
            compilation failed without a remote-primary outage.

        Raises:
            LongTermRemoteUnavailableError: If required remote-primary state is
                unavailable.
        """

        normalized_now = _normalize_datetime(now, timezone_name=self.config.local_timezone_name)
        try:
            with self._store_lock:  # AUDIT-FIX(#1): serialize shared file-backed reads and writes against concurrent background workers.
                result = self.sensor_memory.compile(objects=self.object_store.load_objects(), now=normalized_now)
                if self._has_reflection_payload(result):
                    self.object_store.apply_reflection(result)
                    self.midterm_store.apply_reflection(result)
                return result
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Sensor-memory compilation failed.")
            return self._empty_reflection_result()

    # AUDIT-FIX(#8): process backfill batches defensively so one bad item does not abort the entire recovery run.
    def backfill_ops_multimodal_history(
        self,
        *,
        entries: Iterable[Mapping[str, object]] | None = None,
        now: datetime | None = None,
    ) -> LongTermOpsBackfillRunResult:
        """Replay ops-event history into multimodal long-term evidence.

        Args:
            entries: Optional preloaded ops-event entries. When omitted, the
                service loads entries from the configured ops-event store.
            now: Optional local-time reference for sensor-memory compilation.

        Returns:
            Aggregate counters describing what the backfill scanned, generated,
            applied, skipped, and reflected.

        Raises:
            LongTermRemoteUnavailableError: If required remote-primary state is
                unavailable.
        """

        normalized_now = _normalize_datetime(now, timezone_name=self.config.local_timezone_name)
        scanned_events = 0
        generated_evidence = 0
        sensor_observations = 0
        button_interactions = 0
        print_completions = 0
        applied_evidence = 0
        skipped_existing = 0
        reflected_objects = 0
        created_summaries = 0
        reflection_error: str | None = None

        try:
            raw_entries = tuple(entries) if entries is not None else self._load_backfill_entries()
            build = self.ops_backfiller.build_evidence(raw_entries)
            scanned_events = build.scanned_events
            generated_evidence = build.generated_evidence
            sensor_observations = build.sensor_observations
            button_interactions = build.button_interactions
            print_completions = build.print_completions

            with self._store_lock:  # AUDIT-FIX(#1): serialize shared file-backed reads and writes against concurrent background workers.
                existing_objects = tuple(self.object_store.load_objects())
                existing_conflicts = tuple(self.object_store.load_conflicts())
                existing_archived = tuple(self.object_store.load_archived_objects())
                objects_by_id = {item.memory_id: item for item in existing_objects}
                conflicts_by_slot = {item.slot_key: item for item in existing_conflicts}
                seen_turn_ids = {
                    event_id
                    for item in objects_by_id.values()
                    for event_id in item.source.event_ids
                }

                for evidence in build.evidence:
                    try:
                        extraction = self.multimodal_extractor.extract_evidence(evidence)
                        if extraction.turn_id in seen_turn_ids:
                            skipped_existing += 1
                            continue
                        result = self.consolidator.consolidate(
                            extraction=extraction,
                            existing_objects=tuple(objects_by_id.values()),
                        )
                        for item in (*result.episodic_objects, *result.durable_objects, *result.deferred_objects):
                            objects_by_id[item.memory_id] = self.object_store._merge_object(
                                existing=objects_by_id.get(item.memory_id),
                                incoming=item,
                                increment_support=True,
                            )
                        for conflict in result.conflicts:
                            conflicts_by_slot[conflict.slot_key] = conflict
                        seen_turn_ids.add(extraction.turn_id)
                        applied_evidence += 1
                    except Exception as exc:
                        logger.exception("Failed to backfill one multimodal evidence item.")
                        if reflection_error is None:
                            reflection_error = f"{type(exc).__name__}: {exc}"

                current_objects = _sort_objects_by_memory_id(objects_by_id.values())
                current_conflicts = _sort_conflicts(conflicts_by_slot.values())

                reflection_result = self._empty_reflection_result()
                if applied_evidence:
                    try:
                        reflection_result = self.reflector.reflect(objects=current_objects)
                    except Exception as exc:
                        logger.exception("Long-term backfill reflection failed.")
                        if reflection_error is None:
                            reflection_error = f"{type(exc).__name__}: {exc}"
                    else:
                        current_objects = self._merge_reflection_objects(
                            object_store=self.object_store,
                            current_objects=current_objects,
                            reflection=reflection_result,
                        )
                        reflected_objects += len(reflection_result.reflected_objects)
                        created_summaries += len(reflection_result.created_summaries)

                sensor_result = self._empty_reflection_result()
                try:
                    sensor_result = self.sensor_memory.compile(objects=current_objects, now=normalized_now)
                except Exception as exc:
                    logger.exception("Long-term backfill sensor-memory compilation failed.")
                    if reflection_error is None:
                        reflection_error = f"{type(exc).__name__}: {exc}"
                else:
                    if self._has_reflection_payload(sensor_result):
                        current_objects = self._merge_reflection_objects(
                            object_store=self.object_store,
                            current_objects=current_objects,
                            reflection=sensor_result,
                        )
                    reflected_objects += len(sensor_result.reflected_objects)
                    created_summaries += len(sensor_result.created_summaries)

                retention = self._apply_retention_or_keep(
                    retention_policy=self.retention_policy,
                    objects=current_objects,
                )
                should_write_snapshot = (
                    applied_evidence > 0
                    or self._has_reflection_payload(reflection_result)
                    or self._has_reflection_payload(sensor_result)
                    or bool(retention.expired_objects or retention.pruned_memory_ids or retention.archived_objects)
                )
                if should_write_snapshot:
                    self.object_store.write_snapshot(
                        objects=_sort_objects_by_memory_id(retention.kept_objects),
                        conflicts=current_conflicts,
                        archived_objects=self._merge_archived_objects(
                            existing_archived=existing_archived,
                            archived_updates=retention.archived_objects,
                        ),
                    )
                if self._has_reflection_payload(reflection_result):
                    self.midterm_store.apply_reflection(reflection_result)
                if self._has_reflection_payload(sensor_result):
                    self.midterm_store.apply_reflection(sensor_result)
        except LongTermRemoteUnavailableError:
            raise
        except Exception as exc:
            logger.exception("Ops multimodal backfill failed.")
            reflection_error = f"{type(exc).__name__}: {exc}"

        return LongTermOpsBackfillRunResult(
            scanned_events=scanned_events,
            generated_evidence=generated_evidence,
            applied_evidence=applied_evidence,
            skipped_existing=skipped_existing,
            sensor_observations=sensor_observations,
            button_interactions=button_interactions,
            print_completions=print_completions,
            reflected_objects=reflected_objects,
            created_summaries=created_summaries,
            reflection_error=reflection_error,
        )

    def plan_proactive_candidates(
        self,
        *,
        now: datetime | None = None,
        live_facts: Mapping[str, object] | None = None,
    ) -> LongTermProactivePlanV1:
        """Plan proactive candidates from stored memory and live facts.

        Returns:
            A bounded proactive plan. Returns an empty plan if proactive
            planning fails without a remote-primary outage.

        Raises:
            LongTermRemoteUnavailableError: If required remote-primary state is
                unavailable.
        """

        normalized_now = _normalize_datetime(now, timezone_name=self.config.local_timezone_name)
        try:
            with self._store_lock:  # AUDIT-FIX(#1): serialize shared file-backed reads and writes against concurrent background workers.
                return self.planner.plan(
                    objects=self.object_store.load_objects(),
                    now=normalized_now,
                    live_facts=live_facts,
                )
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Proactive planning failed.")  # AUDIT-FIX(#11): proactive failure is optional and should fail closed.
            return LongTermProactivePlanV1(candidates=())

    def reserve_proactive_candidate(
        self,
        *,
        now: datetime | None = None,
        live_facts: Mapping[str, object] | None = None,
    ) -> LongTermProactiveReservationV1 | None:
        """Plan and reserve the next eligible proactive candidate."""

        normalized_now = _normalize_datetime(now, timezone_name=self.config.local_timezone_name)
        try:
            with self._store_lock:  # AUDIT-FIX(#1): serialize shared file-backed reads and writes against concurrent background workers.
                plan = self.planner.plan(
                    objects=self.object_store.load_objects(),
                    now=normalized_now,
                    live_facts=live_facts,
                )
                return self.proactive_policy.reserve_candidate(plan=plan, now=normalized_now)
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Proactive candidate reservation failed.")
            return None

    # AUDIT-FIX(#9): proactive reservation now runs on normalized local datetimes.
    def reserve_specific_proactive_candidate(
        self,
        candidate: LongTermProactiveCandidateV1,
        *,
        now: datetime | None = None,
    ) -> LongTermProactiveReservationV1:
        """Reserve one specific proactive candidate under policy control."""

        normalized_now = _normalize_datetime(now, timezone_name=self.config.local_timezone_name)
        with self._store_lock:
            return self.proactive_policy.reserve_specific_candidate(candidate, now=normalized_now)

    def preview_proactive_candidate(
        self,
        *,
        now: datetime | None = None,
        live_facts: Mapping[str, object] | None = None,
    ) -> LongTermProactiveCandidateV1 | None:
        """Preview the next eligible proactive candidate without reserving it."""

        normalized_now = _normalize_datetime(now, timezone_name=self.config.local_timezone_name)
        try:
            with self._store_lock:  # AUDIT-FIX(#1): serialize shared file-backed reads and writes against concurrent background workers.
                plan = self.planner.plan(
                    objects=self.object_store.load_objects(),
                    now=normalized_now,
                    live_facts=live_facts,
                )
                return self.proactive_policy.preview_candidate(plan=plan, now=normalized_now)
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Proactive candidate preview failed.")
            return None

    # AUDIT-FIX(#12): make best-effort proactive state updates explicit in the return type.
    def mark_proactive_candidate_delivered(
        self,
        reservation: LongTermProactiveReservationV1,
        *,
        delivered_at: datetime | None = None,
        prompt_text: str | None = None,
    ) -> Any | None:
        """Record that a reserved proactive candidate was delivered."""

        normalized_delivered_at = _normalize_datetime(delivered_at, timezone_name=self.config.local_timezone_name)
        try:
            with self._store_lock:  # AUDIT-FIX(#1): serialize shared file-backed reads and writes against concurrent background workers.
                return self.proactive_policy.mark_delivered(
                    reservation,
                    delivered_at=normalized_delivered_at,
                    prompt_text=prompt_text,
                )
        except Exception:
            logger.exception("Failed to mark proactive candidate as delivered.")
            return None

    def mark_proactive_candidate_skipped(
        self,
        reservation: LongTermProactiveReservationV1,
        *,
        reason: str,
        skipped_at: datetime | None = None,
    ) -> Any | None:
        """Record that a reserved proactive candidate was skipped."""

        normalized_skipped_at = _normalize_datetime(skipped_at, timezone_name=self.config.local_timezone_name)
        try:
            with self._store_lock:  # AUDIT-FIX(#1): serialize shared file-backed reads and writes against concurrent background workers.
                return self.proactive_policy.mark_skipped(
                    reservation,
                    reason=reason,
                    skipped_at=normalized_skipped_at,
                )
        except Exception:
            logger.exception("Failed to mark proactive candidate as skipped.")
            return None

    def run_retention(self) -> LongTermRetentionResultV1:
        """Apply retention and archive policy to stored long-term objects.

        Returns:
            The retention result applied to the object store. Returns a keep-all
            result when retention fails without a remote-primary outage.

        Raises:
            LongTermRemoteUnavailableError: If required remote-primary state is
                unavailable.
        """

        try:
            with self._store_lock:  # AUDIT-FIX(#1): serialize shared file-backed reads and writes against concurrent background workers.
                result = self.retention_policy.apply(objects=self.object_store.load_objects())
                self.object_store.apply_retention(result)
                return result
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Long-term retention run failed.")  # AUDIT-FIX(#11): scheduled maintenance should degrade instead of crashing the service.
            return LongTermRetentionResultV1(
                kept_objects=(),
                expired_objects=(),
                pruned_memory_ids=(),
                archived_objects=(),
            )

    def select_conflict_queue(
        self,
        query_text: str | None,
        *,
        limit: int | None = None,
    ) -> tuple[LongTermConflictQueueItemV1, ...]:
        """Select open memory conflicts relevant to one query."""

        try:
            query = self.query_rewriter.profile(query_text)
            normalized_limit = None if limit is None else _coerce_positive_int(limit, default=1, maximum=_MAX_REVIEW_LIMIT)
            with self._store_lock:  # AUDIT-FIX(#1): serialize shared file-backed reads and writes against concurrent background workers.
                return self.retriever.select_conflict_queue(
                    query=query,
                    limit=normalized_limit,
                )
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Failed to select long-term conflict queue.")  # AUDIT-FIX(#11): retrieval helpers must fail safe.
            return ()

    def resolve_conflict(
        self,
        *,
        slot_key: str,
        selected_memory_id: str,
    ) -> LongTermConflictResolutionV1:
        """Resolve one open conflict by selecting the surviving memory."""

        with self._store_lock:
            conflicts = self.object_store.load_conflicts()
            conflict = next((item for item in conflicts if item.slot_key == slot_key), None)
            if conflict is None:
                raise ValueError(f"No open long-term memory conflict found for slot {slot_key!r}.")
            result = self.conflict_resolver.resolve(
                conflict=conflict,
                objects=self.object_store.load_objects(),
                remaining_conflicts=conflicts,
                selected_memory_id=selected_memory_id,
            )
            self.object_store.apply_conflict_resolution(result)
            self._refresh_restart_recall_packets_locked()
            return result

    def review_memory(
        self,
        *,
        query_text: str | None = None,
        status: str | None = None,
        kind: str | None = None,
        include_episodes: bool = False,
        limit: int = _DEFAULT_REVIEW_LIMIT,
    ) -> LongTermMemoryReviewResultV1:
        """Review stored memories with optional query and status filters."""

        normalized_limit = _coerce_positive_int(limit, default=_DEFAULT_REVIEW_LIMIT, maximum=_MAX_REVIEW_LIMIT)
        with self._store_lock:
            return self.object_store.review_objects(
                query_text=query_text,
                status=status,
                kind=kind,
                include_episodes=include_episodes,
                limit=normalized_limit,
            )

    def confirm_memory(self, *, memory_id: str) -> LongTermMemoryMutationResultV1 | LongTermConflictResolutionV1:
        """Confirm one memory, or resolve its conflict if it is disputed."""

        with self._store_lock:
            conflicts = self.object_store.load_conflicts()
            conflict = next(
                (
                    item
                    for item in conflicts
                    if memory_id == item.candidate_memory_id or memory_id in item.existing_memory_ids
                ),
                None,
            )
            if conflict is not None:
                result = self.conflict_resolver.resolve(
                    conflict=conflict,
                    objects=self.object_store.load_objects(),
                    remaining_conflicts=conflicts,
                    selected_memory_id=memory_id,
                )
                self.object_store.apply_conflict_resolution(result)
                self._refresh_restart_recall_packets_locked()
                return result
            result = self.object_store.confirm_object(memory_id)
            self.object_store.apply_memory_mutation(result)
            self._refresh_restart_recall_packets_locked()
            return result

    def invalidate_memory(
        self,
        *,
        memory_id: str,
        reason: str | None = None,
    ) -> LongTermMemoryMutationResultV1:
        """Mark one memory invalid and persist the resulting mutation."""

        with self._store_lock:
            result = self.object_store.invalidate_object(memory_id, reason=reason)
            self.object_store.apply_memory_mutation(result)
            self._refresh_restart_recall_packets_locked()
            return result

    def delete_memory(self, *, memory_id: str) -> LongTermMemoryMutationResultV1:
        """Delete one memory and persist the resulting mutation."""

        with self._store_lock:
            result = self.object_store.delete_object(memory_id)
            self.object_store.apply_memory_mutation(result)
            self._refresh_restart_recall_packets_locked()
            return result

    def store_explicit_memory(
        self,
        *,
        kind: str,
        summary: str,
        details: str | None = None,
    ) -> PersistentMemoryEntry:
        """Store an operator-requested explicit memory in prompt memory."""

        with self._store_lock:
            return self.prompt_context_store.memory_store.remember(
                kind=kind,
                summary=summary,
                details=details,
            )

    def update_user_profile(
        self,
        *,
        category: str,
        instruction: str,
    ) -> ManagedContextEntry:
        """Upsert one managed user-profile instruction entry."""

        with self._store_lock:
            return self.prompt_context_store.user_store.upsert(
                category=category,
                instruction=instruction,
            )

    def update_personality(
        self,
        *,
        category: str,
        instruction: str,
    ) -> ManagedContextEntry:
        """Upsert one managed personality instruction entry."""

        with self._store_lock:
            return self.prompt_context_store.personality_store.upsert(
                category=category,
                instruction=instruction,
            )

    def flush(self, *, timeout_s: float = 2.0) -> bool:
        """Flush active background writers within one true total deadline."""

        resolved_timeout_s = _coerce_timeout_s(timeout_s, default=2.0)  # AUDIT-FIX(#11): harden lifecycle calls against invalid timeout input and writer exceptions.
        flush_targets: list[tuple[str, object, object]] = []
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
            objects=self.object_store.load_objects(),
        )
        self.midterm_store.replace_packets_with_attribute(
            packets=packets,
            attribute_key="persistence_scope",
            attribute_value="restart_recall",
        )

    def shutdown(self, *, timeout_s: float = 2.0) -> None:
        """Request bounded shutdown for all configured background writers."""

        resolved_timeout_s = _coerce_timeout_s(timeout_s, default=2.0)  # AUDIT-FIX(#11): harden lifecycle calls against invalid timeout input and writer exceptions.
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

    @staticmethod
    # AUDIT-FIX(#2): compute reflection/sensor/retention in memory, commit one object snapshot, then update dependent stores.
    # AUDIT-FIX(#3): catch broad callback failures so background workers do not die on one bad turn.
    def _persist_longterm_turn(
        *,
        config: TwinrConfig,
        store: PromptContextStore,
        graph_store: TwinrPersonalGraphStore,
        object_store: LongTermStructuredStore,
        midterm_store: LongTermMidtermStore,
        extractor: LongTermTurnExtractor,
        consolidator: LongTermMemoryConsolidator,
        reflector: LongTermMemoryReflector,
        sensor_memory: LongTermSensorMemoryCompiler,
        retention_policy: LongTermRetentionPolicy,
        store_lock: threading.RLock | None = None,
        timezone_name: str | None = None,
        item: LongTermConversationTurn,
    ) -> PersistentMemoryEntry | None:
        """Persist one conversation turn through the full long-term pipeline.

        Returns:
            An episodic fallback entry when durable persistence degrades and the
            configured mode allows it. Returns None after durable persistence
            succeeds or when remote-primary mode suppresses local fallback.

        Raises:
            LongTermRemoteUnavailableError: If required remote-primary state is
                unavailable during persistence.
        """

        effective_store_lock = store_lock or threading.RLock()
        effective_timezone_name = timezone_name or config.local_timezone_name
        occurred_at = _normalize_datetime(item.created_at, timezone_name=effective_timezone_name) or item.created_at
        reflection = LongTermMemoryService._empty_reflection_result()
        sensor_reflection = LongTermMemoryService._empty_reflection_result()

        try:
            with effective_store_lock:
                existing_objects = tuple(object_store.load_objects())
                existing_conflicts = tuple(object_store.load_conflicts())
                existing_archived = tuple(object_store.load_archived_objects())
                extraction = extractor.extract_conversation_turn(
                    transcript=item.transcript,
                    response=item.response,
                    occurred_at=occurred_at,
                )
                result = consolidator.consolidate(
                    extraction=extraction,
                    existing_objects=existing_objects,
                )
                current_objects, current_conflicts = LongTermMemoryService._merge_consolidation_state(
                    object_store=object_store,
                    existing_objects=existing_objects,
                    existing_conflicts=existing_conflicts,
                    result=result,
                )
                try:
                    reflection = reflector.reflect(objects=current_objects)
                except Exception:
                    logger.exception("Long-term reflection failed during conversation-turn persistence.")
                else:
                    current_objects = LongTermMemoryService._merge_reflection_objects(
                        object_store=object_store,
                        current_objects=current_objects,
                        reflection=reflection,
                    )
                try:
                    sensor_reflection = sensor_memory.compile(objects=current_objects, now=occurred_at)
                except Exception:
                    logger.exception("Sensor-memory compilation failed during conversation-turn persistence.")
                else:
                    if LongTermMemoryService._has_reflection_payload(sensor_reflection):
                        current_objects = LongTermMemoryService._merge_reflection_objects(
                            object_store=object_store,
                            current_objects=current_objects,
                            reflection=sensor_reflection,
                        )
                retention = LongTermMemoryService._apply_retention_or_keep(
                    retention_policy=retention_policy,
                    objects=current_objects,
                )
                object_store.write_snapshot(
                    objects=_sort_objects_by_memory_id(retention.kept_objects),
                    conflicts=current_conflicts,
                    archived_objects=LongTermMemoryService._merge_archived_objects(
                        existing_archived=existing_archived,
                        archived_updates=retention.archived_objects,
                    ),
                )
                try:
                    graph_store.apply_candidate_edges(result.graph_edges)  # AUDIT-FIX(#2): graph write happens only after the durable object snapshot commits.
                except Exception:
                    logger.exception("Graph-store update failed after conversation-turn snapshot commit.")
                try:
                    if LongTermMemoryService._has_reflection_payload(reflection):
                        midterm_store.apply_reflection(reflection)
                    if LongTermMemoryService._has_reflection_payload(sensor_reflection):
                        midterm_store.apply_reflection(sensor_reflection)
                except Exception:
                    logger.exception("Midterm-store update failed after conversation-turn snapshot commit.")
                if config.long_term_memory_mode == "remote_primary":
                    return None
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Long-term conversation-turn persistence failed; falling back to episodic memory.")

        return LongTermMemoryService._persist_episodic_turn(store=store, item=item)

    @staticmethod
    # AUDIT-FIX(#2): multimodal writes now use the same single-snapshot commit pattern as conversation turns.
    # AUDIT-FIX(#3): broad callback handling prevents worker death on malformed evidence or store faults.
    def _persist_multimodal_evidence(
        *,
        object_store: LongTermStructuredStore,
        midterm_store: LongTermMidtermStore,
        multimodal_extractor: LongTermMultimodalExtractor,
        consolidator: LongTermMemoryConsolidator,
        reflector: LongTermMemoryReflector,
        sensor_memory: LongTermSensorMemoryCompiler,
        retention_policy: LongTermRetentionPolicy,
        store_lock: threading.RLock | None = None,
        timezone_name: str | None = None,
        item: LongTermMultimodalEvidence,
    ) -> None:
        """Persist one multimodal evidence item through the long-term pipeline.

        Raises:
            LongTermRemoteUnavailableError: If required remote-primary state is
                unavailable during persistence.
        """

        effective_store_lock = store_lock or threading.RLock()
        effective_timezone_name = timezone_name or "UTC"
        created_at = _normalize_datetime(item.created_at, timezone_name=effective_timezone_name) or item.created_at
        reflection = LongTermMemoryService._empty_reflection_result()
        sensor_reflection = LongTermMemoryService._empty_reflection_result()

        try:
            with effective_store_lock:
                existing_objects = tuple(object_store.load_objects())
                existing_conflicts = tuple(object_store.load_conflicts())
                existing_archived = tuple(object_store.load_archived_objects())
                extraction = multimodal_extractor.extract_evidence(item)
                result = consolidator.consolidate(
                    extraction=extraction,
                    existing_objects=existing_objects,
                )
                current_objects, current_conflicts = LongTermMemoryService._merge_consolidation_state(
                    object_store=object_store,
                    existing_objects=existing_objects,
                    existing_conflicts=existing_conflicts,
                    result=result,
                )
                try:
                    reflection = reflector.reflect(
                        objects=current_objects,
                        include_midterm=LongTermMemoryService._should_include_midterm_in_multimodal_reflection(result),
                    )
                except Exception:
                    logger.exception("Long-term reflection failed during multimodal persistence.")
                else:
                    current_objects = LongTermMemoryService._merge_reflection_objects(
                        object_store=object_store,
                        current_objects=current_objects,
                        reflection=reflection,
                    )
                try:
                    sensor_reflection = sensor_memory.compile(objects=current_objects, now=created_at)
                except Exception:
                    logger.exception("Sensor-memory compilation failed during multimodal persistence.")
                else:
                    if LongTermMemoryService._has_reflection_payload(sensor_reflection):
                        current_objects = LongTermMemoryService._merge_reflection_objects(
                            object_store=object_store,
                            current_objects=current_objects,
                            reflection=sensor_reflection,
                        )
                retention = LongTermMemoryService._apply_retention_or_keep(
                    retention_policy=retention_policy,
                    objects=current_objects,
                )
                object_store.write_snapshot(
                    objects=_sort_objects_by_memory_id(retention.kept_objects),
                    conflicts=current_conflicts,
                    archived_objects=LongTermMemoryService._merge_archived_objects(
                        existing_archived=existing_archived,
                        archived_updates=retention.archived_objects,
                    ),
                )
                if LongTermMemoryService._has_reflection_payload(reflection):
                    midterm_store.apply_reflection(reflection)
                if LongTermMemoryService._has_reflection_payload(sensor_reflection):
                    midterm_store.apply_reflection(sensor_reflection)
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Long-term multimodal persistence failed.")
            return None

    @staticmethod
    # AUDIT-FIX(#7): prefer the configured store, then fall back to a guarded local clone instead of blindly reconstructing it from an unsafe path.
    def _persist_episodic_turn(*, store: PromptContextStore, item: LongTermConversationTurn) -> PersistentMemoryEntry | None:
        """Write an episodic fallback record into prompt memory."""

        quoted_transcript = json.dumps(item.transcript, ensure_ascii=False)
        quoted_response = json.dumps(item.response, ensure_ascii=False)
        summary = f"Conversation about {quoted_transcript}"
        details = f"User said: {quoted_transcript} Twinr answered: {quoted_response}"

        try:
            return store.memory_store.remember(
                kind="episodic_turn",
                summary=summary,
                details=details,
            )
        except LongTermRemoteUnavailableError:
            logger.info("Configured episodic memory store is remote-unavailable; attempting local fallback.")
        except Exception:
            logger.exception("Primary episodic-memory write failed; attempting local fallback.")

        local_memory_store = LongTermMemoryService._clone_local_memory_store(store.memory_store)
        if local_memory_store is None:
            logger.error("Failed to construct local episodic-memory fallback store.")
            return None
        try:
            return local_memory_store.remember(
                kind="episodic_turn",
                summary=summary,
                details=details,
            )
        except Exception:
            logger.exception("Failed to persist episodic fallback memory locally.")
            return None

    @staticmethod
    def _merge_consolidation_state(
        *,
        object_store: LongTermStructuredStore,
        existing_objects: tuple,
        existing_conflicts: tuple,
        result: LongTermConsolidationResultV1,
    ) -> tuple[tuple, tuple]:
        """Merge newly consolidated objects and conflicts into current state."""

        merged_objects = {item.memory_id: item for item in existing_objects}
        for item in (*result.episodic_objects, *result.durable_objects, *result.deferred_objects):
            merged_objects[item.memory_id] = object_store._merge_object(
                existing=merged_objects.get(item.memory_id),
                incoming=item,
                increment_support=True,
            )
        merged_conflicts = {item.slot_key: item for item in existing_conflicts}
        for conflict in result.conflicts:
            merged_conflicts[conflict.slot_key] = conflict
        return (
            _sort_objects_by_memory_id(merged_objects.values()),
            _sort_conflicts(merged_conflicts.values()),
        )

    @staticmethod
    def _merge_reflection_objects(
        *,
        object_store: LongTermStructuredStore,
        current_objects: tuple,
        reflection: LongTermReflectionResultV1,
    ) -> tuple:
        """Merge reflected objects and summaries into the current object set."""

        merged = {item.memory_id: item for item in current_objects}
        for item in (*reflection.reflected_objects, *reflection.created_summaries):
            merged[item.memory_id] = object_store._merge_object(
                existing=merged.get(item.memory_id),
                incoming=item,
                increment_support=False,
            )
        return _sort_objects_by_memory_id(merged.values())

    @staticmethod
    def _empty_reflection_result() -> LongTermReflectionResultV1:
        """Return an empty reflection payload for fail-closed callers."""

        return LongTermReflectionResultV1(reflected_objects=(), created_summaries=(), midterm_packets=())

    @staticmethod
    def _has_reflection_payload(result: LongTermReflectionResultV1) -> bool:
        """Report whether a reflection payload contains any state updates."""

        return bool(result.reflected_objects or result.created_summaries or result.midterm_packets)

    @staticmethod
    def _should_include_midterm_in_multimodal_reflection(result: LongTermConsolidationResultV1) -> bool:
        """Return whether one multimodal batch needs optional midterm compilation.

        Raw multimodal traces such as button, print, camera, and sensor events
        still need deterministic reflection for promotions and summaries, but
        they do not benefit from invoking the slower midterm compiler when the
        batch contains only episodes, observations, and patterns. Richer
        multimodal outputs can opt back into the full reflection path.
        """

        for item in (*result.episodic_objects, *result.durable_objects, *result.deferred_objects):
            if kind_matches(item.kind, "episode", item.attributes):
                continue
            if kind_matches(item.kind, "observation", item.attributes):
                continue
            if kind_matches(item.kind, "pattern", item.attributes):
                continue
            return True
        return False

    @staticmethod
    # AUDIT-FIX(#8): preserve newly consolidated state even when retention logic fails; do not discard the whole batch.
    def _apply_retention_or_keep(
        *,
        retention_policy: LongTermRetentionPolicy,
        objects: tuple,
    ) -> LongTermRetentionResultV1:
        """Apply retention, or keep current objects if retention fails."""

        try:
            return retention_policy.apply(objects=objects)
        except Exception:
            logger.exception("Long-term retention failed; keeping current objects unpruned.")
            return LongTermRetentionResultV1(
                kept_objects=objects,
                expired_objects=(),
                pruned_memory_ids=(),
                archived_objects=(),
            )

    @staticmethod
    def _merge_archived_objects(
        *,
        existing_archived: Iterable[Any],
        archived_updates: Iterable[Any],
    ) -> tuple[Any, ...]:
        """Merge archived-object updates by memory ID."""

        archived = {item.memory_id: item for item in existing_archived}
        for archived_item in archived_updates:
            archived[archived_item.memory_id] = archived_item
        return _sort_objects_by_memory_id(archived.values())

    @staticmethod
    def _clone_local_memory_store(memory_store: Any) -> Any | None:
        """Clone a local prompt-memory store when episodic fallback is needed."""

        path = _validate_regular_file_path(getattr(memory_store, "path", None), allow_missing=True)
        if path is None:
            return None
        try:
            return type(memory_store)(str(path))
        except Exception:
            logger.exception("Failed to clone local prompt memory store for episodic fallback.")
            return None

    # AUDIT-FIX(#7): only read ops-event history from validated regular files.
    def _load_backfill_entries(self) -> tuple[Mapping[str, object], ...]:
        """Load validated ops-event entries for multimodal backfill."""

        try:
            from twinr.ops.events import TwinrOpsEventStore
        except Exception:
            logger.exception("Failed to import TwinrOpsEventStore for ops-event backfill.")
            return ()

        try:
            store = TwinrOpsEventStore.from_config(self.config)
            path = _validate_regular_file_path(getattr(store, "path", None), allow_missing=False)
            if path is None:
                logger.warning("Skipping ops-event backfill because the configured path is unsafe or missing.")
                return ()
            return tuple(self.ops_backfiller.load_entries(str(path)))
        except Exception:
            logger.exception("Failed to load ops-event history for multimodal backfill.")
            return ()

    # AUDIT-FIX(#6): normalize and bound sensor evidence before it touches extractors or persistent stores.
    def _build_multimodal_evidence(
        self,
        *,
        event_name: str,
        modality: str,
        source: str = "device_event",
        message: str | None = None,
        data: Mapping[str, object] | None = None,
    ) -> LongTermMultimodalEvidence:
        """Build a bounded multimodal evidence payload from device data."""

        sanitized_message = _normalize_text(message, limit=_MULTIMODAL_MESSAGE_LIMIT) or None
        sanitized_data = _sanitize_jsonish(
            dict(data or {}),
            timezone_name=self.config.local_timezone_name,
        )
        payload = sanitized_data if isinstance(sanitized_data, dict) else {"value": sanitized_data}
        return LongTermMultimodalEvidence(
            event_name=_normalize_text(event_name, limit=_MULTIMODAL_NAME_LIMIT) or "unknown_event",
            modality=_normalize_text(modality, limit=_SOURCE_LIMIT) or "unknown_modality",
            source=_normalize_text(source, limit=_SOURCE_LIMIT) or "device_event",
            message=sanitized_message,
            data=payload,
        )
