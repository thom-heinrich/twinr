"""Run one large mixed-corpus long-term memory evaluation against remote memory.

This harness composes the existing synthetic recall, multimodal retrieval, and
unified-retrieval goldsets into one intentionally messy shared corpus inside a
single isolated ChonkyDB namespace. The goal is to measure whether Twinr still
retrieves the right evidence when graph contacts, preferences, plans, episodic
turns, multimodal routines, conflicts, midterm packets, and distractors all
coexist in the same store, and whether the same quality survives a fresh-reader
restart.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError, asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb.client import ChonkyDBError
from twinr.memory.chonkydb.personal_graph import TwinrPersonalGraphStore
from twinr.memory.longterm.core.models import (
    LongTermMemoryConflictV1,
    LongTermMemoryObjectV1,
    LongTermMidtermPacketV1,
)
from twinr.memory.longterm.evaluation._unified_retrieval_shared import (
    UnifiedRetrievalGoldsetCaseResult,
    ensure_unified_retrieval_remote_ready,
    seed_unified_retrieval_fixture,
    unified_retrieval_case_profile_memory_type_coverage,
    unified_retrieval_goldset_cases,
    wait_for_unified_retrieval_cases,
)
from twinr.memory.longterm.evaluation.eval import (
    LongTermEvalCaseResult,
    LongTermEvalSummary,
    _StaticQueryRewriter,
    _ContactSeed,
    _EpisodeSeed,
    _PlanSeed,
    _PreferenceSeed,
    _build_eval_cases,
    _run_eval_case_safely,
    _seed_contacts,
    _seed_episodes,
    _seed_plans,
    _seed_preferences,
    _summarize_results,
)
from twinr.memory.longterm.evaluation.live_midterm_acceptance import (
    _build_isolated_config,
    _normalize_base_project_root,
    _safe_namespace_suffix,
    _shutdown_service,
)
from twinr.memory.longterm.evaluation.multimodal_eval import (
    MultimodalEvalCaseResult,
    MultimodalEvalSummary,
    _build_multimodal_eval_cases,
    _run_case,
    _seed_multimodal_store,
)
from twinr.memory.longterm.evaluation.unified_retrieval_benchmark import (
    UnifiedRetrievalBenchmarkAnalysis,
    UnifiedRetrievalBenchmarkSummary,
    benchmark_unified_retrieval_cases,
)
from twinr.memory.longterm.ingestion.extract import LongTermTurnExtractor
from twinr.memory.longterm.reasoning.reflect import LongTermMemoryReflector
from twinr.memory.longterm.runtime.service import LongTermMemoryService
from twinr.memory.longterm.storage.remote_read_diagnostics import extract_remote_write_context
from twinr.memory.longterm.storage._structured_store.snapshots import StructuredStoreCurrentState


_SCHEMA_VERSION = 2
_OPS_ARTIFACT_NAME = "messy_memory_eval.json"
_REPORT_DIR_NAME = "messy_memory_eval"
_DEFAULT_UNIFIED_CASE_PROFILE = "expanded"
_UNIFIED_CASE_TIMEOUT_S = 180.0
_UNIFIED_CASE_POLL_INTERVAL_S = 2.0
_MAX_ERROR_TEXT_CHARS = 240


@dataclass(frozen=True, slots=True)
class _MaterializedMessyCorpus:
    """Hold one locally materialized messy corpus before remote publication."""

    contacts: tuple[_ContactSeed, ...]
    preferences: tuple[_PreferenceSeed, ...]
    preference_memory_count: int
    plans: tuple[_PlanSeed, ...]
    episodes: tuple[_EpisodeSeed, ...]
    multimodal_events: int
    multimodal_episodic_turns: int
    unified_episodic_objects: int
    unified_durable_objects: int
    unified_conflict_count: int
    unified_midterm_packets: int
    graph_document: object
    current_state: StructuredStoreCurrentState
    midterm_packets: tuple[LongTermMidtermPacketV1, ...]


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in stable ISO-8601 form."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one JSON payload atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_CLOEXEC", 0), 0o600)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp_path, path)


def _flatten_suite_case_id(*, suite: str, case_id: str) -> str:
    """Build one stable cross-suite case identifier."""

    return f"{suite}:{' '.join(str(case_id).split()).strip()}"


def _exception_chain(exc: BaseException | None) -> tuple[BaseException, ...]:
    """Return the bounded causal chain for one failure."""

    chain: list[BaseException] = []
    seen: set[int] = set()
    current = exc
    while isinstance(current, BaseException) and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return tuple(chain)


def _clip_text(value: object, *, limit: int = _MAX_ERROR_TEXT_CHARS) -> str | None:
    """Return one bounded string representation for failure artifacts."""

    text = " ".join(str(value or "").split()).strip()
    if not text:
        return None
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _json_safe_mapping(value: Mapping[str, object] | None) -> dict[str, object] | None:
    """Return one JSON-serializable copy of a mapping payload."""

    if not isinstance(value, Mapping):
        return None
    return json.loads(json.dumps(dict(value), ensure_ascii=False))


def _exception_remote_write_context(exc: BaseException | None) -> dict[str, object] | None:
    """Extract the first attached remote-write context from one failure chain."""

    for item in _exception_chain(exc):
        context = extract_remote_write_context(item)
        if isinstance(context, dict) and context:
            return dict(context)
    return None


def _exception_chain_payload(exc: BaseException | None) -> tuple[dict[str, object], ...]:
    """Return a bounded structured exception chain for the messy-eval artifact."""

    payloads: list[dict[str, object]] = []
    for item in _exception_chain(exc):
        payload: dict[str, object] = {"type": type(item).__name__}
        detail = _clip_text(item)
        if detail is not None:
            payload["detail"] = detail
        if isinstance(item, ChonkyDBError):
            if item.status_code is not None:
                payload["status_code"] = int(item.status_code)
            response_json = _json_safe_mapping(item.response_json if isinstance(item.response_json, Mapping) else None)
            if response_json:
                payload["response_json"] = response_json
        payloads.append(payload)
    return tuple(payloads)


@dataclass(frozen=True, slots=True)
class MessyMemoryEvalSeedStats:
    """Describe the mixed-corpus fixture sizes used by the messy evaluation."""

    synthetic_contacts: int
    synthetic_preferences: int
    synthetic_plans: int
    synthetic_episodic_turns: int
    multimodal_events: int
    multimodal_episodic_turns: int
    unified_episodic_objects: int
    unified_durable_objects: int
    unified_conflict_count: int
    unified_midterm_packets: int
    combined_graph_nodes: int
    combined_graph_edges: int

    @property
    def synthetic_total_entries(self) -> int:
        """Return the total number of synthetic entries in the shared corpus."""

        return (
            self.synthetic_contacts
            + self.synthetic_preferences
            + self.synthetic_plans
            + self.synthetic_episodic_turns
        )

    @property
    def multimodal_total_entries(self) -> int:
        """Return the total number of multimodal-eval entries in the shared corpus."""

        return self.multimodal_events + self.multimodal_episodic_turns

    @property
    def unified_total_entries(self) -> int:
        """Return the total number of unified-retrieval structured entries."""

        return (
            self.unified_episodic_objects
            + self.unified_durable_objects
            + self.unified_conflict_count
            + self.unified_midterm_packets
        )

    @property
    def total_seed_entries(self) -> int:
        """Return the total number of seeded entries across all fixture families."""

        return self.synthetic_total_entries + self.multimodal_total_entries + self.unified_total_entries

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serializable representation."""

        return {
            "synthetic_contacts": self.synthetic_contacts,
            "synthetic_preferences": self.synthetic_preferences,
            "synthetic_plans": self.synthetic_plans,
            "synthetic_episodic_turns": self.synthetic_episodic_turns,
            "multimodal_events": self.multimodal_events,
            "multimodal_episodic_turns": self.multimodal_episodic_turns,
            "unified_episodic_objects": self.unified_episodic_objects,
            "unified_durable_objects": self.unified_durable_objects,
            "unified_conflict_count": self.unified_conflict_count,
            "unified_midterm_packets": self.unified_midterm_packets,
            "combined_graph_nodes": self.combined_graph_nodes,
            "combined_graph_edges": self.combined_graph_edges,
            "synthetic_total_entries": self.synthetic_total_entries,
            "multimodal_total_entries": self.multimodal_total_entries,
            "unified_total_entries": self.unified_total_entries,
            "total_seed_entries": self.total_seed_entries,
        }


@dataclass(frozen=True, slots=True)
class MessyMemoryPhaseResult:
    """Capture one writer or fresh-reader phase across all eval suites."""

    phase: str
    synthetic_summary: LongTermEvalSummary
    multimodal_summary: MultimodalEvalSummary
    unified_summary: UnifiedRetrievalBenchmarkSummary
    synthetic_case_results: tuple[LongTermEvalCaseResult, ...]
    multimodal_case_results: tuple[MultimodalEvalCaseResult, ...]
    unified_case_results: tuple[UnifiedRetrievalGoldsetCaseResult, ...]

    @property
    def total_cases(self) -> int:
        """Return the total number of evaluated cases in this phase."""

        return (
            self.synthetic_summary.total_cases
            + self.multimodal_summary.total_cases
            + self.unified_summary.total_cases
        )

    @property
    def passed_cases(self) -> int:
        """Return the number of passing cases in this phase."""

        return (
            self.synthetic_summary.passed_cases
            + self.multimodal_summary.passed_cases
            + self.unified_summary.passed_cases
        )

    @property
    def accuracy(self) -> float:
        """Return the overall pass rate across all suites in this phase."""

        if self.total_cases <= 0:
            return 0.0
        return self.passed_cases / self.total_cases

    @property
    def ready(self) -> bool:
        """Return whether every case passed in this phase."""

        return self.total_cases > 0 and self.passed_cases == self.total_cases

    @property
    def suite_case_counts(self) -> dict[str, int]:
        """Return per-suite case counts for this phase."""

        return {
            "synthetic": self.synthetic_summary.total_cases,
            "multimodal": self.multimodal_summary.total_cases,
            "unified": self.unified_summary.total_cases,
        }

    @property
    def suite_pass_counts(self) -> dict[str, int]:
        """Return per-suite pass counts for this phase."""

        return {
            "synthetic": self.synthetic_summary.passed_cases,
            "multimodal": self.multimodal_summary.passed_cases,
            "unified": self.unified_summary.passed_cases,
        }

    @property
    def category_case_counts(self) -> dict[str, int]:
        """Return combined non-unified category coverage for this phase."""

        combined = dict(self.synthetic_summary.category_case_counts)
        for category, total in self.multimodal_summary.category_case_counts.items():
            combined[category] = combined.get(category, 0) + total
        return combined

    @property
    def category_pass_counts(self) -> dict[str, int]:
        """Return combined non-unified category pass counts for this phase."""

        combined = dict(self.synthetic_summary.category_pass_counts)
        for category, total in self.multimodal_summary.category_pass_counts.items():
            combined[category] = combined.get(category, 0) + total
        return combined

    @property
    def failed_case_ids(self) -> tuple[str, ...]:
        """Return stable cross-suite identifiers for failed cases."""

        failed: list[str] = []
        failed.extend(
            _flatten_suite_case_id(suite="synthetic", case_id=item.case_id)
            for item in self.synthetic_case_results
            if not item.passed
        )
        failed.extend(
            _flatten_suite_case_id(suite="multimodal", case_id=item.case_id)
            for item in self.multimodal_case_results
            if not item.passed
        )
        failed.extend(
            _flatten_suite_case_id(suite="unified", case_id=item.case_id)
            for item in self.unified_case_results
            if not item.passed
        )
        return tuple(sorted(failed))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "phase": self.phase,
            "synthetic_summary": {
                **asdict(self.synthetic_summary),
                "accuracy": self.synthetic_summary.accuracy,
                "category_accuracy": self.synthetic_summary.category_accuracy(),
            },
            "multimodal_summary": {
                **asdict(self.multimodal_summary),
                "accuracy": self.multimodal_summary.accuracy,
            },
            "unified_summary": self.unified_summary.to_dict(),
            "synthetic_case_results": [asdict(item) for item in self.synthetic_case_results],
            "multimodal_case_results": [asdict(item) for item in self.multimodal_case_results],
            "unified_case_results": [item.to_dict() for item in self.unified_case_results],
            "suite_case_counts": self.suite_case_counts,
            "suite_pass_counts": self.suite_pass_counts,
            "category_case_counts": self.category_case_counts,
            "category_pass_counts": self.category_pass_counts,
            "failed_case_ids": list(self.failed_case_ids),
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "accuracy": self.accuracy,
            "ready": self.ready,
        }


@dataclass(frozen=True, slots=True)
class MessyMemoryRestartSummary:
    """Summarize writer-to-fresh-reader quality drift across the mixed corpus."""

    writer_accuracy: float
    fresh_reader_accuracy: float
    accuracy_delta: float
    writer_unified_quality_score: float
    fresh_reader_unified_quality_score: float
    unified_quality_delta: float
    regressed_case_ids: tuple[str, ...] = ()
    recovered_case_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "writer_accuracy": self.writer_accuracy,
            "fresh_reader_accuracy": self.fresh_reader_accuracy,
            "accuracy_delta": self.accuracy_delta,
            "writer_unified_quality_score": self.writer_unified_quality_score,
            "fresh_reader_unified_quality_score": self.fresh_reader_unified_quality_score,
            "unified_quality_delta": self.unified_quality_delta,
            "regressed_case_ids": list(self.regressed_case_ids),
            "recovered_case_ids": list(self.recovered_case_ids),
        }


@dataclass(frozen=True, slots=True)
class MessyMemoryEvalResult:
    """Describe one complete messy mixed-corpus memory evaluation run."""

    probe_id: str
    status: str
    started_at: str
    finished_at: str
    env_path: str
    base_project_root: str
    runtime_namespace: str
    unified_case_profile: str = _DEFAULT_UNIFIED_CASE_PROFILE
    unified_profile_memory_type_coverage: tuple[tuple[str, int], ...] = ()
    writer_root: str | None = None
    fresh_reader_root: str | None = None
    seed_stats: MessyMemoryEvalSeedStats | None = None
    writer_phase: MessyMemoryPhaseResult | None = None
    fresh_reader_phase: MessyMemoryPhaseResult | None = None
    writer_unified_analysis: UnifiedRetrievalBenchmarkAnalysis | None = None
    fresh_reader_unified_analysis: UnifiedRetrievalBenchmarkAnalysis | None = None
    restart_summary: MessyMemoryRestartSummary | None = None
    artifact_path: str | None = None
    report_path: str | None = None
    error_message: str | None = None
    error_remote_write_context: dict[str, object] | None = None
    error_exception_chain: tuple[dict[str, object], ...] = ()
    failure_temp_roots_preserved: bool = False
    schema_version: int = _SCHEMA_VERSION

    @property
    def executed(self) -> bool:
        """Return whether the evaluation completed both phases."""

        return (
            self.status == "ok"
            and self.writer_phase is not None
            and self.fresh_reader_phase is not None
            and self.restart_summary is not None
        )

    @property
    def ready(self) -> bool:
        """Return whether both writer and fresh-reader phases passed completely."""

        return bool(
            self.executed
            and self.writer_phase is not None
            and self.writer_phase.ready
            and self.fresh_reader_phase is not None
            and self.fresh_reader_phase.ready
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "probe_id": self.probe_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "env_path": self.env_path,
            "base_project_root": self.base_project_root,
            "runtime_namespace": self.runtime_namespace,
            "unified_case_profile": self.unified_case_profile,
            "unified_profile_memory_type_coverage": {
                key: value for key, value in self.unified_profile_memory_type_coverage
            },
            "writer_root": self.writer_root,
            "fresh_reader_root": self.fresh_reader_root,
            "seed_stats": self.seed_stats.to_dict() if self.seed_stats is not None else None,
            "writer_phase": self.writer_phase.to_dict() if self.writer_phase is not None else None,
            "fresh_reader_phase": self.fresh_reader_phase.to_dict() if self.fresh_reader_phase is not None else None,
            "writer_unified_analysis": (
                self.writer_unified_analysis.to_dict() if self.writer_unified_analysis is not None else None
            ),
            "fresh_reader_unified_analysis": (
                self.fresh_reader_unified_analysis.to_dict()
                if self.fresh_reader_unified_analysis is not None
                else None
            ),
            "restart_summary": self.restart_summary.to_dict() if self.restart_summary is not None else None,
            "artifact_path": self.artifact_path,
            "report_path": self.report_path,
            "error_message": self.error_message,
            "error_remote_write_context": (
                dict(self.error_remote_write_context)
                if isinstance(self.error_remote_write_context, dict)
                else None
            ),
            "error_exception_chain": [dict(item) for item in self.error_exception_chain],
            "failure_temp_roots_preserved": self.failure_temp_roots_preserved,
            "schema_version": self.schema_version,
            "executed": self.executed,
            "ready": self.ready,
        }


def default_messy_memory_eval_path(project_root: str | Path) -> Path:
    """Return the rolling ops artifact path for the latest messy-eval run."""

    return Path(project_root).expanduser().resolve() / "artifacts" / "stores" / "ops" / _OPS_ARTIFACT_NAME


def default_messy_memory_eval_report_dir(project_root: str | Path) -> Path:
    """Return the per-run report directory for messy-eval snapshots."""

    return Path(project_root).expanduser().resolve() / "artifacts" / "reports" / _REPORT_DIR_NAME


def write_messy_memory_eval_artifacts(
    result: MessyMemoryEvalResult,
    *,
    project_root: str | Path,
) -> MessyMemoryEvalResult:
    """Persist the rolling artifact and one per-run messy-eval snapshot."""

    artifact_path = default_messy_memory_eval_path(project_root)
    report_dir = default_messy_memory_eval_report_dir(project_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{result.probe_id}.json"
    persisted = replace(
        result,
        artifact_path=str(artifact_path),
        report_path=str(report_path),
    )
    payload = persisted.to_dict()
    _atomic_write_json(report_path, payload)
    _atomic_write_json(artifact_path, payload)
    return persisted


def _seed_messy_corpus(
    service: LongTermMemoryService,
    *,
    base_config: TwinrConfig,
    base_project_root: Path,
    local_seed_root: Path,
    remote_namespace: str,
) -> tuple[
    MessyMemoryEvalSeedStats,
    tuple[_ContactSeed, ...],
    tuple[_PreferenceSeed, ...],
    tuple[_PlanSeed, ...],
    tuple[_EpisodeSeed, ...],
]:
    """Materialize the mixed corpus locally, then publish the final state remotely."""

    materialized = _materialize_messy_corpus_locally(
        base_config=base_config,
        base_project_root=base_project_root,
        runtime_root=local_seed_root,
        remote_namespace=remote_namespace,
    )
    _publish_materialized_messy_corpus(service=service, materialized=materialized)
    seed_stats = MessyMemoryEvalSeedStats(
        synthetic_contacts=len(materialized.contacts),
        synthetic_preferences=materialized.preference_memory_count,
        synthetic_plans=len(materialized.plans),
        synthetic_episodic_turns=len(materialized.episodes),
        multimodal_events=materialized.multimodal_events,
        multimodal_episodic_turns=materialized.multimodal_episodic_turns,
        unified_episodic_objects=materialized.unified_episodic_objects,
        unified_durable_objects=materialized.unified_durable_objects,
        unified_conflict_count=materialized.unified_conflict_count,
        unified_midterm_packets=materialized.unified_midterm_packets,
        combined_graph_nodes=len(getattr(materialized.graph_document, "nodes", ())),
        combined_graph_edges=len(getattr(materialized.graph_document, "edges", ())),
    )
    return (
        seed_stats,
        materialized.contacts,
        materialized.preferences,
        materialized.plans,
        materialized.episodes,
    )


def _build_local_seed_config(
    *,
    base_config: TwinrConfig,
    base_project_root: Path,
    runtime_root: Path,
    remote_namespace: str,
) -> TwinrConfig:
    """Return one isolated local-first config for fixture materialization.

    The messy evaluation measures retrieval quality, not remote ingest
    throughput. Materialize the mixed corpus against a local store first, then
    publish the final graph/object/midterm state once into the live namespace.
    """

    isolated = _build_isolated_config(
        base_config=base_config,
        base_project_root=base_project_root,
        runtime_root=runtime_root,
        remote_namespace=f"{remote_namespace}_local_seed",
        background_store_turns=False,
    )
    return replace(
        isolated,
        long_term_memory_mode="local_first",
        long_term_memory_remote_required=False,
        chonkydb_base_url=None,
        chonkydb_api_key=None,
    )


def _build_eval_runtime_config(
    *,
    base_config: TwinrConfig,
    base_project_root: Path,
    runtime_root: Path,
    remote_namespace: str,
) -> TwinrConfig:
    """Return one isolated remote config that isolates retrieval scoring.

    The messy goldset is supposed to measure ChonkyDB-backed recall/precision,
    not optional LLM rewrite/compiler latency. Disable those auxiliary lanes in
    the eval runtime and let the harness inject deterministic canonical query
    rewrites from the fixed goldset cases instead.
    """

    isolated = _build_isolated_config(
        base_config=base_config,
        base_project_root=base_project_root,
        runtime_root=runtime_root,
        remote_namespace=remote_namespace,
        background_store_turns=False,
    )
    return replace(
        isolated,
        long_term_memory_query_rewrite_enabled=False,
        long_term_memory_subtext_compiler_enabled=False,
    )


def _prepare_local_seed_service(service: LongTermMemoryService) -> None:
    """Disable remote-only or benchmark-irrelevant extras during local seeding."""

    extractor = getattr(service, "extractor", None)
    if extractor is not None and hasattr(extractor, "program"):
        try:
            setattr(extractor, "program", None)
        except (FrozenInstanceError, TypeError, AttributeError):
            if is_dataclass(extractor) and isinstance(extractor, LongTermTurnExtractor):
                setattr(service, "extractor", replace(extractor, program=None))
            else:
                raise
    reflector = getattr(service, "reflector", None)
    if reflector is not None and (
        hasattr(reflector, "program") or hasattr(reflector, "midterm_packet_limit")
    ):
        try:
            if hasattr(reflector, "program"):
                setattr(reflector, "program", None)
            if hasattr(reflector, "midterm_packet_limit"):
                setattr(reflector, "midterm_packet_limit", 0)
        except (FrozenInstanceError, TypeError, AttributeError):
            if is_dataclass(reflector) and isinstance(reflector, LongTermMemoryReflector):
                setattr(
                    service,
                    "reflector",
                    replace(
                        reflector,
                        program=None,
                        midterm_packet_limit=0,
                    ),
                )
            else:
                raise
    if hasattr(service, "personality_learning"):
        setattr(service, "personality_learning", None)


def _prepare_eval_phase_service(
    service: LongTermMemoryService,
    *,
    synthetic_cases,
    multimodal_cases,
) -> None:
    """Remove benchmark-irrelevant cache/LLM lanes before case scoring.

    Fixed goldset queries already carry canonical rewrites. Running the live
    query rewriter, materialized provider-answer front, and prepared-context
    front during scoring mutates the remote namespace and can stall the eval on
    auxiliary OpenAI/cache work that is not part of the ChonkyDB retrieval
    contract being measured.
    """

    canonical_queries: dict[str, str] = {}
    for case in (*tuple(synthetic_cases), *tuple(multimodal_cases)):
        raw_query = " ".join(str(getattr(case, "query_text", "") or "").split()).strip()
        canonical_query = " ".join(str(getattr(case, "canonical_query_text", "") or "").split()).strip()
        if raw_query and canonical_query:
            canonical_queries[raw_query] = canonical_query
    setattr(service, "query_rewriter", _StaticQueryRewriter(canonical_queries))
    if hasattr(service, "prepared_context_front"):
        setattr(service, "prepared_context_front", None)
    if hasattr(service, "provider_answer_front"):
        setattr(service, "provider_answer_front", None)


def _materialize_messy_corpus_locally(
    *,
    base_config: TwinrConfig,
    base_project_root: Path,
    runtime_root: Path,
    remote_namespace: str,
) -> _MaterializedMessyCorpus:
    """Build the full messy corpus locally and return the final persisted state."""

    local_service: LongTermMemoryService | None = None
    local_config = _build_local_seed_config(
        base_config=base_config,
        base_project_root=base_project_root,
        runtime_root=runtime_root,
        remote_namespace=remote_namespace,
    )
    try:
        local_service = LongTermMemoryService.from_config(local_config)
        _prepare_local_seed_service(local_service)
        contacts, preference_items, preference_memory_count, plans = _seed_synthetic_graph_locally_then_sync_remote(
            local_service.graph_store
        )
        episodes = tuple(_seed_episodes(local_service))
        multimodal_seed_stats = _seed_multimodal_store(local_service)
        composed_current_state = local_service.object_store.load_current_state_fine_grained_for_write()
        composed_midterm_packets = tuple(local_service.midterm_store.load_packets())
        unified_seed_stats = seed_unified_retrieval_fixture(local_service)
        unified_current_state = local_service.object_store.load_current_state_fine_grained_for_write()
        unified_midterm_packets = tuple(local_service.midterm_store.load_packets())
        merged_current_state = _merge_structured_current_states(
            composed_current_state,
            unified_current_state,
        )
        merged_midterm_packets = _merge_midterm_packets(
            composed_midterm_packets,
            unified_midterm_packets,
        )
        local_service.object_store.write_snapshot(
            objects=merged_current_state.objects,
            conflicts=merged_current_state.conflicts,
            archived_objects=merged_current_state.archived_objects,
        )
        local_service.midterm_store.save_packets(packets=merged_midterm_packets)
        return _MaterializedMessyCorpus(
            contacts=contacts,
            preferences=preference_items,
            preference_memory_count=preference_memory_count,
            plans=plans,
            episodes=episodes,
            multimodal_events=multimodal_seed_stats.multimodal_events,
            multimodal_episodic_turns=multimodal_seed_stats.episodic_turns,
            unified_episodic_objects=unified_seed_stats.episodic_objects,
            unified_durable_objects=unified_seed_stats.durable_objects,
            unified_conflict_count=unified_seed_stats.conflict_count,
            unified_midterm_packets=unified_seed_stats.midterm_packets,
            graph_document=local_service.graph_store.load_document(),
            current_state=merged_current_state,
            midterm_packets=merged_midterm_packets,
        )
    finally:
        _shutdown_service(local_service)


def _publish_materialized_messy_corpus(
    *,
    service: LongTermMemoryService,
    materialized: _MaterializedMessyCorpus,
) -> None:
    """Publish one locally materialized messy corpus into the remote namespace."""

    remote_graph = getattr(service.graph_store, "_remote_graph", None)
    enabled_method = getattr(remote_graph, "enabled", None)
    persist_method = getattr(remote_graph, "persist_document", None)
    if callable(enabled_method) and enabled_method() and callable(persist_method):
        persist_method(document=materialized.graph_document)
    current_state = materialized.current_state
    service.object_store.write_snapshot(
        objects=tuple(getattr(current_state, "objects", ())),
        conflicts=tuple(getattr(current_state, "conflicts", ())),
        archived_objects=tuple(getattr(current_state, "archived_objects", ())),
    )
    service.midterm_store.save_packets(
        packets=tuple(materialized.midterm_packets),
    )


def _seed_synthetic_graph_locally_then_sync_remote(
    graph_store: TwinrPersonalGraphStore,
) -> tuple[
    tuple[_ContactSeed, ...],
    tuple[_PreferenceSeed, ...],
    int,
    tuple[_PlanSeed, ...],
]:
    """Seed graph fixtures locally, then publish the final graph once remotely.

    The messy mixed-corpus harness measures retrieval quality, not per-contact
    write throughput. Building the large synthetic graph through live
    `remember_*` calls would rewrite the full current graph after every single
    contact/preference/plan mutation and can saturate the remote write queue
    before any recall case executes. Seed the graph locally with the same
    graph-store APIs, then persist the finished graph once through the real
    remote graph path.
    """

    local_graph_store = TwinrPersonalGraphStore(
        path=graph_store.path,
        user_label=graph_store.user_label,
        timezone_name=graph_store.timezone_name,
        lock_path=getattr(graph_store, "_lock_path", None),
    )
    contacts = tuple(_seed_contacts(local_graph_store))
    preferences, preference_memory_count = _seed_preferences(local_graph_store)
    preference_items = tuple(preferences)
    plans = tuple(_seed_plans(local_graph_store, list(contacts)))
    remote_graph = getattr(graph_store, "_remote_graph", None)
    enabled_method = getattr(remote_graph, "enabled", None)
    persist_method = getattr(remote_graph, "persist_document", None)
    if callable(enabled_method) and enabled_method() and callable(persist_method):
        document = local_graph_store.load_document()
        persist_method(document=document)
    return contacts, preference_items, preference_memory_count, plans


def _merge_structured_current_states(
    existing: StructuredStoreCurrentState,
    overlay: StructuredStoreCurrentState,
) -> StructuredStoreCurrentState:
    """Merge standalone fixture snapshots so later seeds do not erase earlier ones.

    The mixed-corpus harness composes multiple goldsets inside one namespace.
    Some standalone fixtures still seed via full `write_snapshot(...)` calls,
    which are correct in isolation but would otherwise drop already seeded
    episodic and multimodal objects when composed here.
    """

    return StructuredStoreCurrentState(
        objects=_merge_memory_objects(existing.objects, overlay.objects),
        conflicts=_merge_memory_conflicts(existing.conflicts, overlay.conflicts),
        archived_objects=_merge_memory_objects(existing.archived_objects, overlay.archived_objects),
    )


def _merge_memory_objects(
    *groups: tuple[LongTermMemoryObjectV1, ...],
) -> tuple[LongTermMemoryObjectV1, ...]:
    """Merge structured memory objects by `memory_id`, keeping later overlays."""

    merged: dict[str, LongTermMemoryObjectV1] = {}
    for group in groups:
        for item in group:
            memory_id = " ".join(item.memory_id.split()).strip()
            if memory_id:
                merged[memory_id] = item
    return tuple(sorted(merged.values(), key=lambda row: row.memory_id))


def _merge_memory_conflicts(
    *groups: tuple[LongTermMemoryConflictV1, ...],
) -> tuple[LongTermMemoryConflictV1, ...]:
    """Merge structured conflicts by slot/candidate identity, keeping later overlays."""

    merged: dict[tuple[str, str], LongTermMemoryConflictV1] = {}
    for group in groups:
        for item in group:
            key = (
                " ".join(item.slot_key.split()).strip(),
                " ".join(item.candidate_memory_id.split()).strip(),
            )
            if key[0] and key[1]:
                merged[key] = item
    return tuple(sorted(merged.values(), key=lambda row: (row.slot_key, row.candidate_memory_id)))


def _merge_midterm_packets(
    *groups: tuple[LongTermMidtermPacketV1, ...],
) -> tuple[LongTermMidtermPacketV1, ...]:
    """Merge midterm packets by packet id so composed fixtures preserve prior packets."""

    merged: dict[str, LongTermMidtermPacketV1] = {}
    for group in groups:
        for packet in group:
            packet_id = " ".join(str(getattr(packet, "packet_id", "") or "").split()).strip()
            if packet_id:
                merged[packet_id] = packet
    return tuple(sorted(merged.values(), key=lambda row: row.packet_id))


def _run_phase(
    *,
    service: LongTermMemoryService,
    phase: str,
    unified_case_profile: str,
    contacts: tuple[_ContactSeed, ...],
    preferences: tuple[_PreferenceSeed, ...],
    plans: tuple[_PlanSeed, ...],
    episodes: tuple[_EpisodeSeed, ...],
) -> tuple[MessyMemoryPhaseResult, UnifiedRetrievalBenchmarkAnalysis]:
    """Run one writer or fresh-reader phase across all three eval families."""

    synthetic_cases = tuple(
        _build_eval_cases(
            contacts=list(contacts),
            preferences=list(preferences),
            plans=list(plans),
            episodes=list(episodes),
        )
    )
    multimodal_cases = _build_multimodal_eval_cases()
    _prepare_eval_phase_service(
        service,
        synthetic_cases=synthetic_cases,
        multimodal_cases=multimodal_cases,
    )
    synthetic_results = tuple(
        _run_eval_case_safely(
            service=service,
            graph_store=service.graph_store,
            case=case,
        )
        for case in synthetic_cases
    )
    multimodal_results = tuple(_run_case(service, case) for case in multimodal_cases)
    unified_cases = unified_retrieval_goldset_cases(profile=unified_case_profile)
    unified_results = wait_for_unified_retrieval_cases(
        service=service,
        cases=unified_cases,
        phase=phase,
        timeout_s=_UNIFIED_CASE_TIMEOUT_S,
        poll_interval_s=_UNIFIED_CASE_POLL_INTERVAL_S,
    )
    unified_analysis = benchmark_unified_retrieval_cases(
        cases=unified_cases,
        case_results=unified_results,
    )
    phase_result = MessyMemoryPhaseResult(
        phase=phase,
        synthetic_summary=_summarize_results(synthetic_results),
        multimodal_summary=_summarize(multimodal_results),
        unified_summary=unified_analysis.summary,
        synthetic_case_results=synthetic_results,
        multimodal_case_results=multimodal_results,
        unified_case_results=unified_results,
    )
    return phase_result, unified_analysis


def _summarize(results: tuple[MultimodalEvalCaseResult, ...]) -> MultimodalEvalSummary:
    """Aggregate multimodal per-case outcomes into summary counters."""

    category_case_counts: dict[str, int] = {}
    category_pass_counts: dict[str, int] = {}
    passed = 0
    for result in results:
        category_case_counts[result.category] = category_case_counts.get(result.category, 0) + 1
        if result.passed:
            passed += 1
            category_pass_counts[result.category] = category_pass_counts.get(result.category, 0) + 1
    return MultimodalEvalSummary(
        total_cases=len(results),
        passed_cases=passed,
        category_case_counts=category_case_counts,
        category_pass_counts=category_pass_counts,
    )


def _case_status_map(phase: MessyMemoryPhaseResult) -> dict[str, bool]:
    """Return one cross-suite pass/fail lookup for phase comparison."""

    status: dict[str, bool] = {}
    for synthetic_item in phase.synthetic_case_results:
        status[_flatten_suite_case_id(suite="synthetic", case_id=synthetic_item.case_id)] = synthetic_item.passed
    for multimodal_item in phase.multimodal_case_results:
        status[_flatten_suite_case_id(suite="multimodal", case_id=multimodal_item.case_id)] = multimodal_item.passed
    for unified_item in phase.unified_case_results:
        status[_flatten_suite_case_id(suite="unified", case_id=unified_item.case_id)] = unified_item.passed
    return status


def _build_restart_summary(
    *,
    writer_phase: MessyMemoryPhaseResult,
    fresh_reader_phase: MessyMemoryPhaseResult,
) -> MessyMemoryRestartSummary:
    """Compute writer-to-fresh-reader regressions across the mixed corpus."""

    writer_status = _case_status_map(writer_phase)
    fresh_reader_status = _case_status_map(fresh_reader_phase)
    regressed = sorted(
        case_id
        for case_id in sorted(set(writer_status) | set(fresh_reader_status))
        if writer_status.get(case_id, False) and not fresh_reader_status.get(case_id, False)
    )
    recovered = sorted(
        case_id
        for case_id in sorted(set(writer_status) | set(fresh_reader_status))
        if not writer_status.get(case_id, False) and fresh_reader_status.get(case_id, False)
    )
    return MessyMemoryRestartSummary(
        writer_accuracy=writer_phase.accuracy,
        fresh_reader_accuracy=fresh_reader_phase.accuracy,
        accuracy_delta=fresh_reader_phase.accuracy - writer_phase.accuracy,
        writer_unified_quality_score=writer_phase.unified_summary.overall_quality_score,
        fresh_reader_unified_quality_score=fresh_reader_phase.unified_summary.overall_quality_score,
        unified_quality_delta=(
            fresh_reader_phase.unified_summary.overall_quality_score
            - writer_phase.unified_summary.overall_quality_score
        ),
        regressed_case_ids=tuple(regressed),
        recovered_case_ids=tuple(recovered),
    )


def run_messy_memory_eval(
    *,
    env_path: str | Path = ".env",
    probe_id: str | None = None,
    unified_case_profile: str = _DEFAULT_UNIFIED_CASE_PROFILE,
    write_artifacts: bool = True,
) -> MessyMemoryEvalResult:
    """Run the large mixed-corpus evaluation against real ChonkyDB state."""

    resolved_env_path = Path(env_path).expanduser().resolve(strict=False)
    started_at = _utc_now_iso()
    effective_probe_id = " ".join(
        str(probe_id or f"messy_memory_eval_{started_at.replace(':', '').replace('-', '')}").split()
    ).strip()
    base_config = TwinrConfig.from_env(resolved_env_path)
    base_project_root = _normalize_base_project_root(resolved_env_path, base_config)
    runtime_namespace = f"twinr_messy_memory_eval_{_safe_namespace_suffix(effective_probe_id)}"
    result = MessyMemoryEvalResult(
        probe_id=effective_probe_id,
        status="running",
        started_at=started_at,
        finished_at=started_at,
        env_path=str(resolved_env_path),
        base_project_root=str(base_project_root),
        runtime_namespace=runtime_namespace,
        unified_case_profile=unified_case_profile,
        unified_profile_memory_type_coverage=unified_retrieval_case_profile_memory_type_coverage(
            profile=unified_case_profile
        ),
    )
    writer_service: LongTermMemoryService | None = None
    fresh_reader_service: LongTermMemoryService | None = None
    writer_root: Path | None = None
    fresh_reader_root: Path | None = None

    try:
        if not base_config.chonkydb_base_url or not base_config.chonkydb_api_key:
            raise RuntimeError("ChonkyDB credentials are required for messy memory evaluation.")
        writer_root = Path(tempfile.mkdtemp(prefix=f"{effective_probe_id}_writer_")).resolve(strict=False)
        fresh_reader_root = Path(tempfile.mkdtemp(prefix=f"{effective_probe_id}_reader_")).resolve(strict=False)
        result = replace(
            result,
            writer_root=str(writer_root),
            fresh_reader_root=str(fresh_reader_root),
        )
        writer_config = _build_eval_runtime_config(
            base_config=base_config,
            base_project_root=base_project_root,
            runtime_root=writer_root,
            remote_namespace=runtime_namespace,
        )
        fresh_reader_config = _build_eval_runtime_config(
            base_config=base_config,
            base_project_root=base_project_root,
            runtime_root=fresh_reader_root,
            remote_namespace=runtime_namespace,
        )
        writer_service = LongTermMemoryService.from_config(writer_config)
        fresh_reader_service = LongTermMemoryService.from_config(fresh_reader_config)
        ensure_unified_retrieval_remote_ready(writer_service)
        ensure_unified_retrieval_remote_ready(fresh_reader_service)
        seed_stats, contacts, preferences, plans, episodes = _seed_messy_corpus(
            writer_service,
            base_config=base_config,
            base_project_root=base_project_root,
            local_seed_root=writer_root / "_local_seed_materialize",
            remote_namespace=runtime_namespace,
        )
        writer_phase, writer_analysis = _run_phase(
            service=writer_service,
            phase="writer",
            unified_case_profile=unified_case_profile,
            contacts=contacts,
            preferences=preferences,
            plans=plans,
            episodes=episodes,
        )
        fresh_reader_phase, fresh_reader_analysis = _run_phase(
            service=fresh_reader_service,
            phase="fresh_reader",
            unified_case_profile=unified_case_profile,
            contacts=contacts,
            preferences=preferences,
            plans=plans,
            episodes=episodes,
        )
        restart_summary = _build_restart_summary(
            writer_phase=writer_phase,
            fresh_reader_phase=fresh_reader_phase,
        )
        result = replace(
            result,
            status="ok",
            finished_at=_utc_now_iso(),
            seed_stats=seed_stats,
            writer_phase=writer_phase,
            fresh_reader_phase=fresh_reader_phase,
            writer_unified_analysis=writer_analysis,
            fresh_reader_unified_analysis=fresh_reader_analysis,
            restart_summary=restart_summary,
        )
    except Exception as exc:
        result = replace(
            result,
            status="failed",
            finished_at=_utc_now_iso(),
            error_message=f"{type(exc).__name__}: {exc}",
            error_remote_write_context=_exception_remote_write_context(exc),
            error_exception_chain=_exception_chain_payload(exc),
            failure_temp_roots_preserved=any(
                root is not None and root.exists() for root in (writer_root, fresh_reader_root)
            ),
        )
    finally:
        _shutdown_service(writer_service)
        _shutdown_service(fresh_reader_service)
        if result.status == "ok":
            for root in (writer_root, fresh_reader_root):
                if root is not None:
                    shutil.rmtree(root, ignore_errors=True)

    if write_artifacts:
        result = write_messy_memory_eval_artifacts(result, project_root=base_project_root)
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the messy mixed-corpus runner."""

    parser = argparse.ArgumentParser(description="Run the large messy long-term memory evaluation.")
    parser.add_argument("--env-file", default=".env", help="Path to the Twinr env file with ChonkyDB credentials.")
    parser.add_argument("--probe-id", default=None, help="Optional stable probe id / namespace suffix.")
    parser.add_argument(
        "--unified-case-profile",
        default=_DEFAULT_UNIFIED_CASE_PROFILE,
        choices=("core", "expanded"),
        help="Which unified-retrieval case profile to evaluate inside the mixed corpus.",
    )
    parser.add_argument(
        "--no-write-artifacts",
        action="store_true",
        help="Skip writing the rolling ops artifact and per-run report snapshot.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entrypoint and print the structured messy-eval JSON."""

    args = _build_arg_parser().parse_args(argv)
    result = run_messy_memory_eval(
        env_path=args.env_file,
        probe_id=args.probe_id,
        unified_case_profile=args.unified_case_profile,
        write_artifacts=not args.no_write_artifacts,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.executed else 1


__all__ = [
    "MessyMemoryEvalResult",
    "MessyMemoryEvalSeedStats",
    "MessyMemoryPhaseResult",
    "MessyMemoryRestartSummary",
    "default_messy_memory_eval_path",
    "default_messy_memory_eval_report_dir",
    "run_messy_memory_eval",
    "write_messy_memory_eval_artifacts",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
