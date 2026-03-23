"""Profile long-term provider-context latency with a forensic run pack.

This module runs one or more read-only provider-context retrieval iterations
against the configured Twinr long-term memory stack, captures a forensic
workflow run pack, and writes a condensed latency report that highlights the
slowest spans, decision pivots, and per-query totals. The profiling flow is
meant to answer "where did the time go?" for semantic recall questions without
mutating long-term memory state.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from twinr import TwinrConfig
from twinr.agent.workflows.forensics import (
    WorkflowForensics,
    bind_workflow_forensics,
    workflow_decision,
    workflow_event,
    workflow_span,
)
from twinr.memory.longterm.runtime.service import LongTermMemoryService


_SCHEMA_VERSION = 1
_PROFILE_SCHEMA = "twinr_longterm_latency_profile_v1"
_REPORT_DIR_NAME = "longterm_latency_profile"


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in stable ISO-8601 form."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _coerce_text(value: object | None) -> str:
    """Normalize arbitrary values into one stable single-line string."""

    return " ".join(str(value or "").split()).strip()


def _coerce_optional_text(value: object | None) -> str | None:
    """Normalize optional text values and collapse blanks to ``None``."""

    text = _coerce_text(value)
    return text or None


def _coerce_float(value: object | None, *, default: float = 0.0) -> float:
    """Coerce one numeric value to ``float`` while failing closed."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_mapping(value: object | None) -> dict[str, object] | None:
    """Return one shallow-copied mapping when the input is mapping-like."""

    if not isinstance(value, dict):
        return None
    return dict(value)


def _query_fingerprint(query_text: str | None) -> dict[str, object]:
    """Return a redacted, stable fingerprint for one query string."""

    clean_query = _coerce_text(query_text)
    return {
        "query_chars": len(clean_query),
        "query_sha256": hashlib.sha256(clean_query.encode("utf-8")).hexdigest()[:16] if clean_query else "",
    }


def _profile_id() -> str:
    """Return one stable profile identifier for artifact directories."""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{uuid4().hex[:8]}"


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one JSON payload atomically to disk."""

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


def _read_jsonl_records(path: Path) -> list[dict[str, object]]:
    """Load structured JSONL records from one workflow run pack."""

    records: list[dict[str, object]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        loaded = json.loads(line)
        if isinstance(loaded, dict):
            records.append(dict(loaded))
    return records


def _decision_selected_id(record: dict[str, object]) -> str | None:
    """Extract the selected decision identifier from one workflow record."""

    reason = record.get("reason")
    if not isinstance(reason, dict):
        return None
    selected = reason.get("selected")
    if not isinstance(selected, dict):
        return None
    selected_id = selected.get("id")
    return _coerce_optional_text(selected_id)


@dataclass(frozen=True, slots=True)
class LongTermLatencyProfileQueryRun:
    """Describe one profiled provider-context retrieval iteration."""

    query_index: int
    iteration: int
    trace_id: str
    duration_ms: float
    query_sha256: str
    query_chars: int
    durable_context: bool
    episodic_context: bool
    graph_context: bool
    conflict_context: bool
    midterm_context: bool
    subtext_context: bool
    topic_context: bool = False
    error_type: str | None = None
    error_message: str | None = None
    error_details: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        payload["error_details"] = dict(self.error_details or {})
        return payload


@dataclass(frozen=True, slots=True)
class LongTermLatencySpanStat:
    """Summarize one span name across the full workflow run pack."""

    name: str
    count: int
    total_duration_ms: float
    avg_duration_ms: float
    max_duration_ms: float
    error_count: int = 0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class LongTermLatencyDecisionSummary:
    """Summarize one decision event type across the workflow run pack."""

    msg: str
    count: int
    selected_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "msg": self.msg,
            "count": self.count,
            "selected_counts": dict(sorted(self.selected_counts.items())),
        }


@dataclass(frozen=True, slots=True)
class LongTermLatencyProfileResult:
    """Capture the artifacts and bottleneck summary for one profile run."""

    profile_id: str
    captured_at_utc: str
    env_file: str
    report_dir: str
    workflow_run_id: str
    workflow_run_dir: str
    trace_mode: str
    query_rewrite_enabled: bool
    subtext_compiler_enabled: bool
    background_store_turns_enabled: bool
    read_cache_ttl_s: float
    recall_limit: int
    midterm_limit: int
    query_runs: tuple[LongTermLatencyProfileQueryRun, ...]
    top_spans: tuple[LongTermLatencySpanStat, ...]
    decision_summaries: tuple[LongTermLatencyDecisionSummary, ...]
    exception_counts: dict[str, int]
    workflow_summary: dict[str, object]
    workflow_metrics: dict[str, object]
    profile_target: str = "provider_context"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "schema": _PROFILE_SCHEMA,
            "version": _SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "captured_at_utc": self.captured_at_utc,
            "env_file": self.env_file,
            "report_dir": self.report_dir,
            "workflow_run_id": self.workflow_run_id,
            "workflow_run_dir": self.workflow_run_dir,
            "trace_mode": self.trace_mode,
            "query_rewrite_enabled": self.query_rewrite_enabled,
            "subtext_compiler_enabled": self.subtext_compiler_enabled,
            "background_store_turns_enabled": self.background_store_turns_enabled,
            "read_cache_ttl_s": self.read_cache_ttl_s,
            "recall_limit": self.recall_limit,
            "midterm_limit": self.midterm_limit,
            "query_runs": [item.to_dict() for item in self.query_runs],
            "top_spans": [item.to_dict() for item in self.top_spans],
            "decision_summaries": [item.to_dict() for item in self.decision_summaries],
            "exception_counts": dict(sorted(self.exception_counts.items())),
            "workflow_summary": dict(self.workflow_summary),
            "workflow_metrics": dict(self.workflow_metrics),
            "profile_target": self.profile_target,
        }


def _summarize_span_stats(records: list[dict[str, object]]) -> tuple[LongTermLatencySpanStat, ...]:
    """Aggregate span timings by span name from one workflow run pack."""

    grouped: dict[str, dict[str, float]] = {}
    for record in records:
        kind = _coerce_text(record.get("kind"))
        if kind not in {"span_end", "exception"}:
            continue
        name = _coerce_optional_text(record.get("msg"))
        if kind == "exception":
            details = record.get("details")
            if isinstance(details, dict):
                name = _coerce_optional_text(details.get("span")) or name
        if not name:
            continue
        raw_kpi = record.get("kpi")
        duration_ms = 0.0
        if isinstance(raw_kpi, dict):
            duration_ms = max(0.0, _coerce_float(raw_kpi.get("duration_ms")))
        stats = grouped.setdefault(
            name,
            {
                "count": 0.0,
                "total_duration_ms": 0.0,
                "max_duration_ms": 0.0,
                "error_count": 0.0,
            },
        )
        stats["count"] += 1.0
        stats["total_duration_ms"] += duration_ms
        stats["max_duration_ms"] = max(stats["max_duration_ms"], duration_ms)
        if kind == "exception":
            stats["error_count"] += 1.0
    ranked: list[LongTermLatencySpanStat] = []
    for name, stats in grouped.items():
        count = max(1, int(stats["count"]))
        total_duration_ms = round(stats["total_duration_ms"], 3)
        ranked.append(
            LongTermLatencySpanStat(
                name=name,
                count=count,
                total_duration_ms=total_duration_ms,
                avg_duration_ms=round(total_duration_ms / count, 3),
                max_duration_ms=round(stats["max_duration_ms"], 3),
                error_count=int(stats["error_count"]),
            )
        )
    ranked.sort(key=lambda item: (item.total_duration_ms, item.max_duration_ms, item.count), reverse=True)
    return tuple(ranked[:16])


def _summarize_decisions(records: list[dict[str, object]]) -> tuple[LongTermLatencyDecisionSummary, ...]:
    """Aggregate decision events by message and selected option."""

    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        if _coerce_text(record.get("kind")) != "decision":
            continue
        msg = _coerce_optional_text(record.get("msg"))
        if not msg:
            continue
        grouped_entry = grouped.setdefault(msg, {"count": 0, "selected_counts": {}})
        grouped_entry["count"] += 1
        selected_id = _decision_selected_id(record) or "[none]"
        selected_counts = grouped_entry["selected_counts"]
        selected_counts[selected_id] = int(selected_counts.get(selected_id, 0)) + 1
    ranked = [
        LongTermLatencyDecisionSummary(
            msg=msg,
            count=int(data["count"]),
            selected_counts=dict(data["selected_counts"]),
        )
        for msg, data in grouped.items()
    ]
    ranked.sort(key=lambda item: item.count, reverse=True)
    return tuple(ranked[:16])


def _section_presence(context: object) -> dict[str, bool]:
    """Return the visible section-presence flags for one memory context."""

    return {
        "durable_context": bool(getattr(context, "durable_context", None)),
        "episodic_context": bool(getattr(context, "episodic_context", None)),
        "graph_context": bool(getattr(context, "graph_context", None)),
        "conflict_context": bool(getattr(context, "conflict_context", None)),
        "midterm_context": bool(getattr(context, "midterm_context", None)),
        "subtext_context": bool(getattr(context, "subtext_context", None)),
        "topic_context": bool(getattr(context, "topic_context", None)),
    }


def default_longterm_latency_profile_dir(project_root: Path, *, profile_id: str) -> Path:
    """Return the artifact directory for one long-term latency profile run."""

    return project_root / "artifacts" / "reports" / _REPORT_DIR_NAME / profile_id


def write_longterm_latency_profile_artifacts(result: LongTermLatencyProfileResult) -> Path:
    """Persist one latency profile result payload to its report directory."""

    report_dir = Path(result.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / "profile.json"
    _atomic_write_json(output_path, result.to_dict())
    return output_path


def run_longterm_latency_profile(
    *,
    env_path: str | Path,
    queries: list[str],
    runs_per_query: int = 3,
    keep_query_rewrite: bool = False,
    keep_subtext_compiler: bool = False,
    keep_background_store_turns: bool = False,
    trace_mode: str = "forensic",
    output_dir: str | Path | None = None,
    target: str = "provider_context",
) -> LongTermLatencyProfileResult:
    """Run a latency profile over provider-context retrieval.

    Args:
        env_path: Path to the Twinr ``.env`` file that defines the profiling
            target.
        queries: User-style semantic recall questions to profile.
        runs_per_query: Number of iterations to run per query. Defaults to 3.
        keep_query_rewrite: If true, preserve the configured query rewriter.
            Defaults to false so the profile isolates the retrieval path.
        keep_subtext_compiler: If true, preserve the configured subtext
            compiler. Defaults to false so hidden-personalization work does not
            obscure the retrieval bottleneck.
        keep_background_store_turns: If true, preserve background writers.
            Defaults to false so read-only profiling does not start them.
        trace_mode: Workflow forensics mode. Defaults to ``forensic``.
        output_dir: Optional override for the report directory.

    Returns:
        A structured latency profile result containing the persisted report path
        and the summarized workflow run pack.

    Notes:
        The measured retrieval iterations are read-only. Before the first timed
        iteration, the profiler runs the same required-remote readiness probe
        that bootstraps missing snapshot heads for fresh namespaces, so leading
        repo profiles do not fail on unprovisioned current-scope documents.
    """

    normalized_queries = [_coerce_text(query) for query in queries if _coerce_text(query)]
    if not normalized_queries:
        raise ValueError("At least one non-empty query is required for long-term latency profiling.")

    env_file = Path(env_path).expanduser().resolve(strict=False)
    config = TwinrConfig.from_env(env_file)
    project_root = Path(config.project_root).expanduser().resolve(strict=False)
    profile_id = _profile_id()
    report_dir = (
        Path(output_dir).expanduser().resolve(strict=False)
        if output_dir is not None
        else default_longterm_latency_profile_dir(project_root, profile_id=profile_id)
    )
    workflow_dir = report_dir / "workflow"
    trace_mode_value = _coerce_optional_text(trace_mode) or "forensic"
    profile_config = replace(
        config,
        long_term_memory_query_rewrite_enabled=(
            config.long_term_memory_query_rewrite_enabled if keep_query_rewrite else False
        ),
        long_term_memory_subtext_compiler_enabled=(
            config.long_term_memory_subtext_compiler_enabled if keep_subtext_compiler else False
        ),
        long_term_memory_background_store_turns=(
            config.long_term_memory_background_store_turns if keep_background_store_turns else False
        ),
    )
    normalized_target = _coerce_optional_text(target) or "provider_context"
    if normalized_target not in {"provider_context", "fast_provider_context"}:
        raise ValueError("target must be 'provider_context' or 'fast_provider_context'.")

    tracer = WorkflowForensics(
        project_root=project_root,
        service="longterm-latency-profile",
        enabled=True,
        mode=trace_mode_value,
        base_dir=workflow_dir,
        allow_raw_text=False,
    )
    query_runs: list[LongTermLatencyProfileQueryRun] = []
    service: LongTermMemoryService | None = None
    failure: BaseException | None = None
    try:
        service = LongTermMemoryService.from_config(profile_config)
        # Match runtime startup semantics before timing retrieval iterations.
        # This provisions missing required-remote snapshot heads for fresh
        # namespaces without charging that bootstrap cost to per-query metrics.
        service.probe_remote_ready(bootstrap=True, include_archive=False)
        for query_index, query_text in enumerate(normalized_queries, start=1):
            query_fp = _query_fingerprint(query_text)
            for iteration in range(1, max(1, int(runs_per_query)) + 1):
                trace_id = uuid4().hex
                with bind_workflow_forensics(tracer, trace_id=trace_id):
                    workflow_decision(
                        msg="longterm_latency_profile_iteration_config",
                        question="Which runtime switches are active for this latency profile iteration?",
                        selected={
                            "id": normalized_target,
                            "summary": (
                                "Run read-only provider-context retrieval with bounded profiling switches."
                                if normalized_target == "provider_context"
                                else "Run read-only fast provider-context retrieval with bounded profiling switches."
                            ),
                        },
                        options=[
                            {
                                "id": "retrieval_only_profile",
                                "summary": "Disable optional rewrite/subtext/background writers unless explicitly preserved.",
                                "score_components": {
                                    "query_rewrite_enabled": bool(profile_config.long_term_memory_query_rewrite_enabled),
                                    "subtext_compiler_enabled": bool(profile_config.long_term_memory_subtext_compiler_enabled),
                                    "background_store_turns_enabled": bool(profile_config.long_term_memory_background_store_turns),
                                },
                                "constraints_violated": [],
                            },
                            {
                                "id": "full_runtime_profile",
                                "summary": "Keep every configured runtime feature enabled during profiling.",
                                "score_components": {
                                    "query_rewrite_enabled": bool(config.long_term_memory_query_rewrite_enabled),
                                    "subtext_compiler_enabled": bool(config.long_term_memory_subtext_compiler_enabled),
                                    "background_store_turns_enabled": bool(config.long_term_memory_background_store_turns),
                                },
                                "constraints_violated": [] if keep_query_rewrite and keep_subtext_compiler and keep_background_store_turns else ["disabled_optional_features"],
                            },
                        ],
                        context={
                            "query_index": query_index,
                            "iteration": iteration,
                            "profile_target": normalized_target,
                            **query_fp,
                        },
                        confidence="high",
                        guardrails=[
                            "Do not mutate long-term memory during latency profiling.",
                            "Prefer profiling the retrieval path without optional background work unless explicitly requested.",
                        ],
                        kpi_impact_estimate={
                            "runs_per_query": max(1, int(runs_per_query)),
                            "read_cache_ttl_s": _coerce_float(profile_config.long_term_memory_remote_read_cache_ttl_s),
                        },
                    )
                    with workflow_span(
                        name="longterm_latency_profile_iteration",
                        kind="retrieval",
                        details={
                            "query_index": query_index,
                            "iteration": iteration,
                            "profile_target": normalized_target,
                            **query_fp,
                        },
                    ):
                        started = time.perf_counter()
                        error: BaseException | None = None
                        context: object | None = None
                        try:
                            if normalized_target == "fast_provider_context":
                                context = service.build_fast_provider_context(query_text)
                            else:
                                context = service.build_provider_context(query_text)
                        except BaseException as exc:  # Keep one failed iteration in the forensic artifact instead of aborting the full profile.
                            error = exc
                        duration_ms = round((time.perf_counter() - started) * 1000.0, 3)
                        presence = _section_presence(context) if context is not None else _section_presence(object())
                        error_type = type(error).__name__ if error is not None else None
                        error_message = _coerce_optional_text(str(error)) if error is not None else None
                        error_details = _coerce_mapping(getattr(error, "details", None)) if error is not None else None
                        if error is not None:
                            workflow_event(
                                kind="warning",
                                msg="longterm_latency_profile_iteration_failure",
                                details={
                                    "query_index": query_index,
                                    "iteration": iteration,
                                    "profile_target": normalized_target,
                                    **query_fp,
                                    "error_type": error_type,
                                    "error_message": error_message,
                                    "error_details": dict(error_details or {}),
                                },
                                kpi={"duration_ms": duration_ms},
                            )
                        workflow_event(
                            kind="metric",
                            msg="longterm_latency_profile_iteration_result",
                            details={
                                "query_index": query_index,
                                "iteration": iteration,
                                "profile_target": normalized_target,
                                **query_fp,
                                **presence,
                                "error_type": error_type,
                            },
                            kpi={"duration_ms": duration_ms},
                        )
                query_runs.append(
                    LongTermLatencyProfileQueryRun(
                        query_index=query_index,
                        iteration=iteration,
                        trace_id=trace_id,
                        duration_ms=duration_ms,
                        query_sha256=str(query_fp["query_sha256"]),
                        query_chars=int(query_fp["query_chars"]),
                        durable_context=presence["durable_context"],
                        episodic_context=presence["episodic_context"],
                        graph_context=presence["graph_context"],
                        conflict_context=presence["conflict_context"],
                        midterm_context=presence["midterm_context"],
                        subtext_context=presence["subtext_context"],
                        topic_context=presence["topic_context"],
                        error_type=error_type,
                        error_message=error_message,
                        error_details=dict(error_details or {}) or None,
                    )
                )
    except BaseException as exc:
        failure = exc
    finally:
        if service is not None:
            service.shutdown(timeout_s=1.0)
        tracer.close()

    workflow_run_dir = workflow_dir / tracer.run_id
    if failure is not None:
        raise RuntimeError(f"Long-term latency profile failed; forensic runpack at {workflow_run_dir}") from failure

    workflow_summary = json.loads((workflow_run_dir / "run.summary.json").read_text(encoding="utf-8"))
    workflow_metrics = json.loads((workflow_run_dir / "run.metrics.json").read_text(encoding="utf-8"))
    records = _read_jsonl_records(workflow_run_dir / "run.jsonl")
    result = LongTermLatencyProfileResult(
        profile_id=profile_id,
        captured_at_utc=_utc_now_iso(),
        env_file=str(env_file),
        report_dir=str(report_dir),
        workflow_run_id=tracer.run_id,
        workflow_run_dir=str(workflow_run_dir),
        trace_mode=trace_mode_value,
        query_rewrite_enabled=bool(profile_config.long_term_memory_query_rewrite_enabled),
        subtext_compiler_enabled=bool(profile_config.long_term_memory_subtext_compiler_enabled),
        background_store_turns_enabled=bool(profile_config.long_term_memory_background_store_turns),
        read_cache_ttl_s=_coerce_float(profile_config.long_term_memory_remote_read_cache_ttl_s),
        recall_limit=int(profile_config.long_term_memory_recall_limit),
        midterm_limit=int(profile_config.long_term_memory_midterm_limit),
        query_runs=tuple(query_runs),
        top_spans=_summarize_span_stats(records),
        decision_summaries=_summarize_decisions(records),
        exception_counts=dict(workflow_summary.get("exception_counts") or {}),
        workflow_summary=dict(workflow_summary),
        workflow_metrics=dict(workflow_metrics),
        profile_target=normalized_target,
    )
    write_longterm_latency_profile_artifacts(result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the long-term latency profiler."""

    parser = argparse.ArgumentParser(
        description=(
            "Profile Twinr long-term provider-context latency for one or more semantic queries "
            "and persist a forensic run pack plus condensed bottleneck summary."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env-file", default=".env", help="Path to the Twinr .env file.")
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        required=True,
        help="Semantic recall query to profile. Repeat for multiple queries.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Iterations to run per query. Defaults to 3.",
    )
    parser.add_argument(
        "--keep-query-rewrite",
        action="store_true",
        help="Preserve the configured long-term query rewrite feature.",
    )
    parser.add_argument(
        "--keep-subtext-compiler",
        action="store_true",
        help="Preserve the configured subtext compiler during profiling.",
    )
    parser.add_argument(
        "--keep-background-store-turns",
        action="store_true",
        help="Preserve background long-term writers during profiling.",
    )
    parser.add_argument(
        "--trace-mode",
        default="forensic",
        help="Workflow forensics mode to use. Defaults to forensic.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for the profile report directory.",
    )
    parser.add_argument(
        "--target",
        default="provider_context",
        choices=("provider_context", "fast_provider_context"),
        help="Which read-only context builder to profile. Defaults to provider_context.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and print the profile result as JSON."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    result = run_longterm_latency_profile(
        env_path=args.env_file,
        queries=list(args.queries or []),
        runs_per_query=args.runs,
        keep_query_rewrite=bool(args.keep_query_rewrite),
        keep_subtext_compiler=bool(args.keep_subtext_compiler),
        keep_background_store_turns=bool(args.keep_background_store_turns),
        trace_mode=args.trace_mode,
        output_dir=args.output_dir,
        target=args.target,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
