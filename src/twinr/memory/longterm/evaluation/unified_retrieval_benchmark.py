"""Benchmark unified retrieval quality over the fixed multi-source goldset.

This module turns the deterministic unified-retrieval goldset into a benchmark
that reports how precisely the retriever selects sources, ids, and join
anchors, not just whether the minimum required evidence is present. The goal is
to surface over-selection regressions such as unrelated durable/episodic items
leaking into otherwise correct answers while keeping the same fixed case set
used by the goldset and live acceptance runners.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Mapping

from twinr.memory.longterm.evaluation._unified_retrieval_shared import (
    UnifiedRetrievalGoldsetCase,
    UnifiedRetrievalGoldsetCaseResult,
    unified_retrieval_case_profile_memory_type_coverage,
    unified_retrieval_goldset_cases,
)
from twinr.memory.longterm.evaluation.unified_retrieval_goldset import (
    UnifiedRetrievalGoldsetResult,
    run_unified_retrieval_goldset,
)


_SCHEMA_VERSION = 1
_OPS_ARTIFACT_NAME = "unified_retrieval_benchmark.json"
_REPORT_DIR_NAME = "unified_retrieval_benchmark"
_DEFAULT_CASE_PROFILE = "expanded"


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


def _pair_map(pairs: tuple[tuple[str, tuple[str, ...]], ...]) -> dict[str, tuple[str, ...]]:
    """Return one dictionary view for normalized pair payloads."""

    return {key: tuple(values) for key, values in pairs}


def _flatten_pairs(pairs: tuple[tuple[str, tuple[str, ...]], ...]) -> tuple[str, ...]:
    """Flatten a pair payload into deterministic key/value tokens."""

    tokens = [f"{key}::{value}" for key, values in pairs for value in values]
    return tuple(sorted(tokens))


def _unexpected_pairs(
    expected: tuple[tuple[str, tuple[str, ...]], ...],
    actual: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return pair values that appeared in the actual payload but were not expected."""

    expected_map = _pair_map(expected)
    actual_map = _pair_map(actual)
    unexpected: list[tuple[str, tuple[str, ...]]] = []
    for key, actual_values in sorted(actual_map.items(), key=lambda item: item[0]):
        expected_values = set(expected_map.get(key, ()))
        extras = tuple(value for value in actual_values if value not in expected_values)
        if extras:
            unexpected.append((key, extras))
    return tuple(unexpected)


def _ratio(matched: int, total: int, *, perfect_when_empty: bool) -> float:
    """Return one bounded ratio with explicit empty-set handling."""

    if total <= 0:
        return 1.0 if perfect_when_empty else 0.0
    return matched / total


def _set_precision_recall(expected: set[str], actual: set[str]) -> tuple[float, float]:
    """Return precision and recall for one expected/actual set pair."""

    matched = len(expected & actual)
    precision = _ratio(matched, len(actual), perfect_when_empty=not expected)
    recall = _ratio(matched, len(expected), perfect_when_empty=True)
    return precision, recall


def _case_lookup(cases: tuple[UnifiedRetrievalGoldsetCase, ...]) -> dict[str, UnifiedRetrievalGoldsetCase]:
    """Build a case-id lookup for benchmark analysis."""

    return {case.case_id: case for case in cases}


@dataclass(frozen=True, slots=True)
class UnifiedRetrievalBenchmarkNamedMetric:
    """Describe one named precision/recall metric across the benchmark cases."""

    name: str
    expected_count: int
    actual_count: int
    matched_count: int
    precision: float
    recall: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "name": self.name,
            "expected_count": self.expected_count,
            "actual_count": self.actual_count,
            "matched_count": self.matched_count,
            "precision": self.precision,
            "recall": self.recall,
        }


@dataclass(frozen=True, slots=True)
class UnifiedRetrievalBenchmarkCaseMetrics:
    """Capture per-case quality metrics beyond the goldset pass/fail verdict."""

    case_id: str
    phase: str
    passed: bool
    candidate_source_precision: float
    candidate_source_recall: float
    access_path_precision: float
    access_path_recall: float
    selected_id_precision: float
    selected_id_recall: float
    join_anchor_precision: float
    join_anchor_recall: float
    unexpected_candidate_sources: tuple[str, ...] = ()
    unexpected_access_path: tuple[str, ...] = ()
    forbidden_access_path_hits: tuple[str, ...] = ()
    unexpected_selected_ids: tuple[tuple[str, tuple[str, ...]], ...] = ()
    unexpected_join_anchors: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @property
    def overall_case_score(self) -> float:
        """Return one compact score for operator-friendly case comparisons."""

        metric_values = (
            self.candidate_source_precision,
            self.candidate_source_recall,
            self.access_path_precision,
            self.access_path_recall,
            self.selected_id_precision,
            self.selected_id_recall,
            self.join_anchor_precision,
            self.join_anchor_recall,
        )
        score = sum(metric_values) / len(metric_values)
        if self.forbidden_access_path_hits:
            return 0.0
        return score

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "case_id": self.case_id,
            "phase": self.phase,
            "passed": self.passed,
            "candidate_source_precision": self.candidate_source_precision,
            "candidate_source_recall": self.candidate_source_recall,
            "access_path_precision": self.access_path_precision,
            "access_path_recall": self.access_path_recall,
            "selected_id_precision": self.selected_id_precision,
            "selected_id_recall": self.selected_id_recall,
            "join_anchor_precision": self.join_anchor_precision,
            "join_anchor_recall": self.join_anchor_recall,
            "unexpected_candidate_sources": list(self.unexpected_candidate_sources),
            "unexpected_access_path": list(self.unexpected_access_path),
            "forbidden_access_path_hits": list(self.forbidden_access_path_hits),
            "unexpected_selected_ids": {
                key: list(values) for key, values in self.unexpected_selected_ids
            },
            "unexpected_join_anchors": {
                key: list(values) for key, values in self.unexpected_join_anchors
            },
            "overall_case_score": self.overall_case_score,
        }


@dataclass(frozen=True, slots=True)
class UnifiedRetrievalBenchmarkSummary:
    """Aggregate benchmark metrics across all evaluated cases."""

    total_cases: int
    passed_cases: int
    candidate_source_exact_cases: int
    access_path_exact_cases: int
    selected_id_exact_cases: int
    join_anchor_exact_cases: int
    source_precision_macro: float
    source_recall_macro: float
    access_path_precision_macro: float
    access_path_recall_macro: float
    selected_id_precision_macro: float
    selected_id_recall_macro: float
    join_anchor_precision_macro: float
    join_anchor_recall_macro: float
    path_safety_rate: float
    overall_quality_score: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "candidate_source_exact_cases": self.candidate_source_exact_cases,
            "access_path_exact_cases": self.access_path_exact_cases,
            "selected_id_exact_cases": self.selected_id_exact_cases,
            "join_anchor_exact_cases": self.join_anchor_exact_cases,
            "source_precision_macro": self.source_precision_macro,
            "source_recall_macro": self.source_recall_macro,
            "access_path_precision_macro": self.access_path_precision_macro,
            "access_path_recall_macro": self.access_path_recall_macro,
            "selected_id_precision_macro": self.selected_id_precision_macro,
            "selected_id_recall_macro": self.selected_id_recall_macro,
            "join_anchor_precision_macro": self.join_anchor_precision_macro,
            "join_anchor_recall_macro": self.join_anchor_recall_macro,
            "path_safety_rate": self.path_safety_rate,
            "overall_quality_score": self.overall_quality_score,
        }


@dataclass(frozen=True, slots=True)
class UnifiedRetrievalBenchmarkAnalysis:
    """Bundle the computed benchmark metrics for one evaluated case set."""

    case_metrics: tuple[UnifiedRetrievalBenchmarkCaseMetrics, ...]
    source_metrics: tuple[UnifiedRetrievalBenchmarkNamedMetric, ...]
    access_path_metrics: tuple[UnifiedRetrievalBenchmarkNamedMetric, ...]
    summary: UnifiedRetrievalBenchmarkSummary

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "case_metrics": [item.to_dict() for item in self.case_metrics],
            "source_metrics": [item.to_dict() for item in self.source_metrics],
            "access_path_metrics": [item.to_dict() for item in self.access_path_metrics],
            "summary": self.summary.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class UnifiedRetrievalBenchmarkResult:
    """Describe one complete unified-retrieval benchmark run."""

    probe_id: str
    status: str
    started_at: str
    finished_at: str
    env_path: str
    base_project_root: str
    case_profile: str = _DEFAULT_CASE_PROFILE
    profile_memory_type_coverage: tuple[tuple[str, int], ...] = ()
    goldset_result: UnifiedRetrievalGoldsetResult | None = None
    analysis: UnifiedRetrievalBenchmarkAnalysis | None = None
    artifact_path: str | None = None
    report_path: str | None = None
    error_message: str | None = None
    schema_version: int = _SCHEMA_VERSION

    @property
    def ready(self) -> bool:
        """Return whether the benchmark completed and produced at least one case."""

        return (
            self.status == "ok"
            and self.analysis is not None
            and self.analysis.summary.total_cases > 0
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
            "case_profile": self.case_profile,
            "profile_memory_type_coverage": {
                key: value for key, value in self.profile_memory_type_coverage
            },
            "goldset_result": self.goldset_result.to_dict() if self.goldset_result is not None else None,
            "analysis": self.analysis.to_dict() if self.analysis is not None else None,
            "artifact_path": self.artifact_path,
            "report_path": self.report_path,
            "error_message": self.error_message,
            "schema_version": self.schema_version,
            "ready": self.ready,
        }


def benchmark_unified_retrieval_cases(
    *,
    cases: tuple[UnifiedRetrievalGoldsetCase, ...],
    case_results: tuple[UnifiedRetrievalGoldsetCaseResult, ...],
) -> UnifiedRetrievalBenchmarkAnalysis:
    """Compute quality metrics for one evaluated unified-retrieval case set."""

    case_map = _case_lookup(cases)
    benchmark_cases: list[UnifiedRetrievalBenchmarkCaseMetrics] = []
    for result in case_results:
        case = case_map.get(result.case_id)
        if case is None:
            raise ValueError(f"Unknown unified retrieval benchmark case: {result.case_id!r}")

        expected_sources = set(case.required_candidate_sources)
        actual_sources = set(result.candidate_sources)
        candidate_source_precision, candidate_source_recall = _set_precision_recall(
            expected_sources,
            actual_sources,
        )

        expected_access_path = set(case.required_access_path)
        actual_access_path = set(result.access_path)
        access_path_precision, access_path_recall = _set_precision_recall(
            expected_access_path,
            actual_access_path,
        )

        expected_selected_tokens = set(_flatten_pairs(case.required_selected_ids))
        actual_selected_tokens = set(_flatten_pairs(result.selected_ids))
        selected_id_precision, selected_id_recall = _set_precision_recall(
            expected_selected_tokens,
            actual_selected_tokens,
        )

        expected_join_tokens = set(_flatten_pairs(case.required_join_anchors))
        actual_join_tokens = set(_flatten_pairs(result.join_anchors))
        join_anchor_precision, join_anchor_recall = _set_precision_recall(
            expected_join_tokens,
            actual_join_tokens,
        )

        benchmark_cases.append(
            UnifiedRetrievalBenchmarkCaseMetrics(
                case_id=result.case_id,
                phase=result.phase,
                passed=result.passed,
                candidate_source_precision=candidate_source_precision,
                candidate_source_recall=candidate_source_recall,
                access_path_precision=access_path_precision,
                access_path_recall=access_path_recall,
                selected_id_precision=selected_id_precision,
                selected_id_recall=selected_id_recall,
                join_anchor_precision=join_anchor_precision,
                join_anchor_recall=join_anchor_recall,
                unexpected_candidate_sources=tuple(sorted(actual_sources - expected_sources)),
                unexpected_access_path=tuple(sorted(actual_access_path - expected_access_path)),
                forbidden_access_path_hits=result.forbidden_access_path,
                unexpected_selected_ids=_unexpected_pairs(case.required_selected_ids, result.selected_ids),
                unexpected_join_anchors=_unexpected_pairs(case.required_join_anchors, result.join_anchors),
            )
        )

    source_metrics = _named_metrics(
        names=sorted(
            {
                name
                for case, result in zip(
                    (case_map[item.case_id] for item in case_results),
                    case_results,
                    strict=True,
                )
                for name in (*case.required_candidate_sources, *result.candidate_sources)
            }
        ),
        expected_lookup=lambda case, _result: set(case.required_candidate_sources),
        actual_lookup=lambda _case, result: set(result.candidate_sources),
        cases=case_results,
        case_map=case_map,
    )
    access_path_metrics = _named_metrics(
        names=sorted(
            {
                name
                for case, result in zip(
                    (case_map[item.case_id] for item in case_results),
                    case_results,
                    strict=True,
                )
                for name in (*case.required_access_path, *result.access_path)
            }
        ),
        expected_lookup=lambda case, _result: set(case.required_access_path),
        actual_lookup=lambda _case, result: set(result.access_path),
        cases=case_results,
        case_map=case_map,
    )
    summary = _benchmark_summary(benchmark_cases)
    return UnifiedRetrievalBenchmarkAnalysis(
        case_metrics=tuple(benchmark_cases),
        source_metrics=source_metrics,
        access_path_metrics=access_path_metrics,
        summary=summary,
    )


def _named_metrics(
    *,
    names: list[str],
    expected_lookup,
    actual_lookup,
    cases: tuple[UnifiedRetrievalGoldsetCaseResult, ...],
    case_map: Mapping[str, UnifiedRetrievalGoldsetCase],
) -> tuple[UnifiedRetrievalBenchmarkNamedMetric, ...]:
    """Compute precision/recall for one named case-level feature family."""

    metrics: list[UnifiedRetrievalBenchmarkNamedMetric] = []
    for name in names:
        expected_count = 0
        actual_count = 0
        matched_count = 0
        for result in cases:
            case = case_map[result.case_id]
            expected = expected_lookup(case, result)
            actual = actual_lookup(case, result)
            if name in expected:
                expected_count += 1
            if name in actual:
                actual_count += 1
            if name in expected and name in actual:
                matched_count += 1
        precision = _ratio(matched_count, actual_count, perfect_when_empty=expected_count == 0)
        recall = _ratio(matched_count, expected_count, perfect_when_empty=True)
        metrics.append(
            UnifiedRetrievalBenchmarkNamedMetric(
                name=name,
                expected_count=expected_count,
                actual_count=actual_count,
                matched_count=matched_count,
                precision=precision,
                recall=recall,
            )
        )
    return tuple(metrics)


def _benchmark_summary(
    benchmark_cases: list[UnifiedRetrievalBenchmarkCaseMetrics],
) -> UnifiedRetrievalBenchmarkSummary:
    """Aggregate macro benchmark metrics across all case-level measurements."""

    total_cases = len(benchmark_cases)
    if total_cases <= 0:
        return UnifiedRetrievalBenchmarkSummary(
            total_cases=0,
            passed_cases=0,
            candidate_source_exact_cases=0,
            access_path_exact_cases=0,
            selected_id_exact_cases=0,
            join_anchor_exact_cases=0,
            source_precision_macro=0.0,
            source_recall_macro=0.0,
            access_path_precision_macro=0.0,
            access_path_recall_macro=0.0,
            selected_id_precision_macro=0.0,
            selected_id_recall_macro=0.0,
            join_anchor_precision_macro=0.0,
            join_anchor_recall_macro=0.0,
            path_safety_rate=0.0,
            overall_quality_score=0.0,
        )

    passed_cases = sum(1 for item in benchmark_cases if item.passed)
    candidate_source_exact_cases = sum(
        1
        for item in benchmark_cases
        if not item.unexpected_candidate_sources and item.candidate_source_recall >= 1.0
    )
    access_path_exact_cases = sum(
        1
        for item in benchmark_cases
        if not item.unexpected_access_path
        and not item.forbidden_access_path_hits
        and item.access_path_recall >= 1.0
    )
    selected_id_exact_cases = sum(
        1
        for item in benchmark_cases
        if not item.unexpected_selected_ids and item.selected_id_recall >= 1.0
    )
    join_anchor_exact_cases = sum(
        1
        for item in benchmark_cases
        if not item.unexpected_join_anchors and item.join_anchor_recall >= 1.0
    )
    source_precision_macro = sum(item.candidate_source_precision for item in benchmark_cases) / total_cases
    source_recall_macro = sum(item.candidate_source_recall for item in benchmark_cases) / total_cases
    access_path_precision_macro = sum(item.access_path_precision for item in benchmark_cases) / total_cases
    access_path_recall_macro = sum(item.access_path_recall for item in benchmark_cases) / total_cases
    selected_id_precision_macro = sum(item.selected_id_precision for item in benchmark_cases) / total_cases
    selected_id_recall_macro = sum(item.selected_id_recall for item in benchmark_cases) / total_cases
    join_anchor_precision_macro = sum(item.join_anchor_precision for item in benchmark_cases) / total_cases
    join_anchor_recall_macro = sum(item.join_anchor_recall for item in benchmark_cases) / total_cases
    path_safety_rate = (
        sum(1 for item in benchmark_cases if not item.forbidden_access_path_hits) / total_cases
    )
    overall_quality_score = (
        (
            source_precision_macro
            + source_recall_macro
            + access_path_precision_macro
            + access_path_recall_macro
            + selected_id_precision_macro
            + selected_id_recall_macro
            + join_anchor_precision_macro
            + join_anchor_recall_macro
        )
        / 8.0
    ) * path_safety_rate
    return UnifiedRetrievalBenchmarkSummary(
        total_cases=total_cases,
        passed_cases=passed_cases,
        candidate_source_exact_cases=candidate_source_exact_cases,
        access_path_exact_cases=access_path_exact_cases,
        selected_id_exact_cases=selected_id_exact_cases,
        join_anchor_exact_cases=join_anchor_exact_cases,
        source_precision_macro=source_precision_macro,
        source_recall_macro=source_recall_macro,
        access_path_precision_macro=access_path_precision_macro,
        access_path_recall_macro=access_path_recall_macro,
        selected_id_precision_macro=selected_id_precision_macro,
        selected_id_recall_macro=selected_id_recall_macro,
        join_anchor_precision_macro=join_anchor_precision_macro,
        join_anchor_recall_macro=join_anchor_recall_macro,
        path_safety_rate=path_safety_rate,
        overall_quality_score=overall_quality_score,
    )


def default_unified_retrieval_benchmark_path(project_root: str | Path) -> Path:
    """Return the rolling ops artifact path for the latest benchmark run."""

    return Path(project_root).expanduser().resolve() / "artifacts" / "stores" / "ops" / _OPS_ARTIFACT_NAME


def default_unified_retrieval_benchmark_report_dir(project_root: str | Path) -> Path:
    """Return the per-run report directory for benchmark snapshots."""

    return Path(project_root).expanduser().resolve() / "artifacts" / "reports" / _REPORT_DIR_NAME


def write_unified_retrieval_benchmark_artifacts(
    result: UnifiedRetrievalBenchmarkResult,
    *,
    project_root: str | Path,
) -> UnifiedRetrievalBenchmarkResult:
    """Persist the rolling benchmark artifact and per-run report snapshot."""

    artifact_path = default_unified_retrieval_benchmark_path(project_root)
    report_dir = default_unified_retrieval_benchmark_report_dir(project_root)
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


def run_unified_retrieval_benchmark(
    *,
    env_path: str | Path = ".env",
    probe_id: str | None = None,
    case_profile: str = _DEFAULT_CASE_PROFILE,
    write_artifacts: bool = True,
) -> UnifiedRetrievalBenchmarkResult:
    """Run the fixed unified-retrieval benchmark atop a fresh goldset execution."""

    started_at = _utc_now_iso()
    effective_probe_id = " ".join(
        str(probe_id or f"unified_retrieval_benchmark_{started_at.replace(':', '').replace('-', '')}").split()
    ).strip()
    result = UnifiedRetrievalBenchmarkResult(
        probe_id=effective_probe_id,
        status="running",
        started_at=started_at,
        finished_at=started_at,
        env_path=str(Path(env_path).expanduser().resolve(strict=False)),
        base_project_root="",
        case_profile=case_profile,
        profile_memory_type_coverage=unified_retrieval_case_profile_memory_type_coverage(profile=case_profile),
    )
    try:
        goldset_result = run_unified_retrieval_goldset(
            env_path=env_path,
            probe_id=effective_probe_id,
            case_profile=case_profile,
            write_artifacts=write_artifacts,
        )
        if not goldset_result.ready:
            raise RuntimeError(goldset_result.error_message or "Unified retrieval goldset benchmark preflight failed.")
        analysis = benchmark_unified_retrieval_cases(
            cases=unified_retrieval_goldset_cases(profile=case_profile),
            case_results=goldset_result.case_results,
        )
        result = replace(
            result,
            status="ok",
            finished_at=_utc_now_iso(),
            env_path=goldset_result.env_path,
            base_project_root=goldset_result.base_project_root,
            goldset_result=goldset_result,
            analysis=analysis,
        )
    except Exception as exc:
        result = replace(
            result,
            status="failed",
            finished_at=_utc_now_iso(),
            error_message=f"{type(exc).__name__}: {exc}",
        )

    if write_artifacts:
        project_root = result.base_project_root or Path(env_path).expanduser().resolve(strict=False).parent
        result = write_unified_retrieval_benchmark_artifacts(
            result,
            project_root=project_root,
        )
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the unified retrieval benchmark."""

    parser = argparse.ArgumentParser(description="Run the unified retrieval benchmark over the fixed goldset.")
    parser.add_argument("--env-file", default=".env", help="Path to the Twinr env file with ChonkyDB credentials.")
    parser.add_argument("--probe-id", default=None, help="Optional stable probe id / namespace suffix.")
    parser.add_argument(
        "--case-profile",
        default=_DEFAULT_CASE_PROFILE,
        choices=("core", "expanded"),
        help="Which unified-retrieval case profile to benchmark.",
    )
    parser.add_argument(
        "--no-write-artifacts",
        action="store_true",
        help="Skip writing the rolling ops artifact and per-run report snapshot.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entrypoint and print the structured benchmark JSON."""

    args = _build_arg_parser().parse_args(argv)
    result = run_unified_retrieval_benchmark(
        env_path=args.env_file,
        probe_id=args.probe_id,
        case_profile=args.case_profile,
        write_artifacts=not args.no_write_artifacts,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.ready else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "UnifiedRetrievalBenchmarkAnalysis",
    "UnifiedRetrievalBenchmarkCaseMetrics",
    "UnifiedRetrievalBenchmarkNamedMetric",
    "UnifiedRetrievalBenchmarkResult",
    "UnifiedRetrievalBenchmarkSummary",
    "benchmark_unified_retrieval_cases",
    "default_unified_retrieval_benchmark_path",
    "default_unified_retrieval_benchmark_report_dir",
    "run_unified_retrieval_benchmark",
    "write_unified_retrieval_benchmark_artifacts",
]
