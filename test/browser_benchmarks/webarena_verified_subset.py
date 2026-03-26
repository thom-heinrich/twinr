"""Select and summarize an honest WebArena Verified benchmark slice.

This module keeps the benchmark selection logic separate from the Twinr runner.
It only knows how to load official subset metadata, filter tasks that the
current Twinr adapter can evaluate without faking unsupported trace semantics,
pick a deterministic stratified slice, and summarize the resulting runs.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from typing import Any
import json

from webarena_verified.api import WebArenaVerified
from webarena_verified.types.task import WebArenaVerifiedTask

@dataclass(frozen=True, slots=True)
class WebArenaVerifiedBenchmarkCase:
    """Describe one official task that Twinr can benchmark fairly today."""

    task_id: int
    site_name: str
    expected_action: str
    eval_types: tuple[str, ...]
    intent: str


def load_official_subset_task_ids(*, subset_name: str) -> list[int]:
    """Load task ids from one official WebArena Verified subset file."""

    subset_root = resources.files("webarena_verified").joinpath("assets", "dataset", "subsets")
    subset_path = subset_root.joinpath(f"{subset_name}.json")
    if not subset_path.is_file():
        raise FileNotFoundError(f"Unknown WebArena Verified subset: {subset_name}")
    payload = json.loads(subset_path.read_text(encoding="utf-8"))
    return [int(task_id) for task_id in payload.get("task_ids") or []]


def collect_compatible_cases(
    *,
    benchmark: WebArenaVerified,
    task_ids: Sequence[int],
) -> list[WebArenaVerifiedBenchmarkCase]:
    """Filter official tasks down to the slice Twinr can score honestly today."""

    compatible_cases: list[WebArenaVerifiedBenchmarkCase] = []
    for task_id in task_ids:
        task = benchmark.get_task(int(task_id))
        if not is_currently_supported_task(task=task):
            continue
        compatible_cases.append(
            WebArenaVerifiedBenchmarkCase(
                task_id=int(task.task_id),
                site_name=str(task.sites[0].value),
                expected_action=str(task.expected_action),
                eval_types=tuple(type(evaluator).__name__ for evaluator in task.eval),
                intent=str(task.intent),
            )
        )
    return compatible_cases


def is_currently_supported_task(*, task: WebArenaVerifiedTask) -> bool:
    """Return whether the current Twinr adapter can score this task fairly."""

    eval_types = {type(evaluator).__name__ for evaluator in task.eval}
    return (
        len(task.sites) == 1
        and len(task.start_urls) == 1
        and str(task.expected_action).upper() == "RETRIEVE"
        and eval_types == {"AgentResponseEvaluatorCfg"}
    )


def select_stratified_cases(
    *,
    cases: Sequence[WebArenaVerifiedBenchmarkCase],
    per_site: int,
) -> list[WebArenaVerifiedBenchmarkCase]:
    """Pick a deterministic per-site spread across the compatible case pool."""

    cases_by_site: dict[str, list[WebArenaVerifiedBenchmarkCase]] = defaultdict(list)
    for case in cases:
        cases_by_site[str(case.site_name)].append(case)

    selected_cases: list[WebArenaVerifiedBenchmarkCase] = []
    for site_name in sorted(cases_by_site):
        site_cases = sorted(cases_by_site[site_name], key=lambda case: int(case.task_id))
        for position in choose_evenly_spaced_positions(length=len(site_cases), limit=int(per_site)):
            selected_cases.append(site_cases[position])
    return sorted(selected_cases, key=lambda case: (str(case.site_name), int(case.task_id)))


def choose_evenly_spaced_positions(*, length: int, limit: int) -> list[int]:
    """Choose stable spread-out positions from an ordered task list."""

    if length <= 0 or limit <= 0:
        return []
    if limit >= length:
        return list(range(length))
    if limit == 1:
        return [length // 2]

    targets = [index * (length - 1) / (limit - 1) for index in range(limit)]
    selected: list[int] = []
    used: set[int] = set()
    for target in targets:
        candidate_positions = sorted(range(length), key=lambda pos: (abs(pos - target), pos))
        for position in candidate_positions:
            if position in used:
                continue
            selected.append(position)
            used.add(position)
            break
    return sorted(selected)


def summarize_benchmark_runs(*, runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate top-line and per-site WebArena benchmark metrics."""

    total = len(runs)
    passed = sum(1 for run in runs if bool(run.get("eval_pass")))
    browser_completed = sum(1 for run in runs if bool(run.get("browser_ok")))
    runner_errors = sum(1 for run in runs if str(run.get("runner_status")) != "completed")
    by_site: dict[str, dict[str, Any]] = {}
    for site_name in sorted({str(run.get("site_name") or "unknown") for run in runs}):
        site_runs = [run for run in runs if str(run.get("site_name") or "unknown") == site_name]
        site_total = len(site_runs)
        site_passed = sum(1 for run in site_runs if bool(run.get("eval_pass")))
        site_browser_ok = sum(1 for run in site_runs if bool(run.get("browser_ok")))
        by_site[site_name] = {
            "total": site_total,
            "passed": site_passed,
            "pass_rate": _safe_rate(numerator=site_passed, denominator=site_total),
            "browser_ok": site_browser_ok,
            "browser_ok_rate": _safe_rate(numerator=site_browser_ok, denominator=site_total),
        }

    return {
        "total_tasks": total,
        "passed_tasks": passed,
        "pass_rate": _safe_rate(numerator=passed, denominator=total),
        "browser_ok_tasks": browser_completed,
        "browser_ok_rate": _safe_rate(numerator=browser_completed, denominator=total),
        "runner_errors": runner_errors,
        "by_site": by_site,
    }


def _safe_rate(*, numerator: int, denominator: int) -> float:
    """Return a stable float rate for JSON output."""

    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)
