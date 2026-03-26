#!/usr/bin/env python3
# ruff: noqa: E402
"""Run a repeatable official WebArena Verified subset benchmark through Twinr.

The benchmark stays honest in two ways. First, it selects tasks from the
official WebArena Verified subsets, not a Twinr-owned website or scorer.
Second, it currently filters to the official tasks that Twinr's existing adapter
can evaluate faithfully without pretending to support richer network-event
traces than the browser workspace emits today.
"""

from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any
import argparse
import json
import subprocess
import sys
import time

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from test.browser_benchmarks.webarena_verified_adapter import (
    build_webarena_task_run,
    build_webarena_verified_config,
    evaluate_browser_result,
    load_webarena_verified_task,
    run_twinr_task,
)
from test.browser_benchmarks.webarena_verified_subset import (
    collect_compatible_cases,
    load_official_subset_task_ids,
    select_stratified_cases,
    summarize_benchmark_runs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env", help="Path to the Twinr environment file.")
    parser.add_argument("--environment-label", default="local", help="Label written into the JSON payload.")
    parser.add_argument(
        "--output",
        default="",
        help="Optional explicit JSON output path. Defaults to artifacts/reports/browser_automation/webarena_verified_benchmark_<label>.json",
    )
    parser.add_argument(
        "--subset-name",
        default="webarena-verified-hard",
        help="Official WebArena Verified subset name used as the source pool.",
    )
    parser.add_argument(
        "--per-site",
        type=int,
        default=4,
        help="How many compatible tasks to pick per site from the official subset.",
    )
    parser.add_argument(
        "--site-name",
        action="append",
        default=[],
        help="Optional site filter. Repeat the flag to keep only specific official sites.",
    )
    parser.add_argument("--max-steps", type=int, default=12, help="Twinr browser step budget per task.")
    parser.add_argument("--max-runtime-s", type=float, default=120.0, help="Twinr browser runtime budget per task.")
    parser.add_argument("--shopping-url", default="http://localhost:7770", help="URL for the official shopping site.")
    parser.add_argument("--shopping-admin-url", default="http://localhost:7780", help="URL for the official shopping_admin site.")
    parser.add_argument("--reddit-url", default="http://localhost:9999", help="URL for the official reddit site.")
    parser.add_argument("--gitlab-url", default="http://localhost:8023", help="URL for the official gitlab site.")
    parser.add_argument("--wikipedia-url", default="http://localhost:8888/wikipedia_en_all_maxi_2022-05/A/", help="URL for the official wikipedia site.")
    parser.add_argument("--map-url", default="http://localhost:3000", help="URL for the official map site.")
    parser.add_argument(
        "--start-official-envs",
        action="store_true",
        help="Start all required official site containers before the benchmark.",
    )
    parser.add_argument(
        "--stop-official-envs",
        action="store_true",
        help="Stop the official site containers that this runner started after the benchmark.",
    )
    return parser.parse_args()


def _run_official_env_command(*, command: list[str]) -> None:
    subprocess.run(command, check=True)


def _git_commit(repo_root: Path) -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                capture_output=True,
                check=True,
                text=True,
            )
            .stdout.strip()
        )
    except subprocess.CalledProcessError:
        return "unknown"


def _start_official_envs(*, repo_root: Path, site_names: list[str]) -> list[str]:
    started: list[str] = []
    for site_name in site_names:
        _run_official_env_command(
            command=[
                str(repo_root / ".venv" / "bin" / "webarena-verified"),
                "env",
                "start",
                "--site",
                str(site_name),
            ]
        )
        started.append(site_name)
    return started


def _stop_official_envs(*, repo_root: Path, site_names: list[str]) -> None:
    for site_name in site_names:
        _run_official_env_command(
            command=[
                str(repo_root / ".venv" / "bin" / "webarena-verified"),
                "env",
                "stop",
                "--site",
                str(site_name),
            ]
        )


def main() -> int:
    args = _parse_args()
    repo_root = _REPO_ROOT
    env_path = (repo_root / args.env_file).resolve() if not Path(args.env_file).is_absolute() else Path(args.env_file)

    benchmark_config = build_webarena_verified_config(
        shopping_url=args.shopping_url or None,
        shopping_admin_url=args.shopping_admin_url or None,
        reddit_url=args.reddit_url or None,
        gitlab_url=args.gitlab_url or None,
        wikipedia_url=args.wikipedia_url or None,
        map_url=args.map_url or None,
    )
    subset_task_ids = load_official_subset_task_ids(subset_name=str(args.subset_name))
    benchmark, _ = load_webarena_verified_task(task_id=int(subset_task_ids[0]), config=benchmark_config)
    compatible_cases = collect_compatible_cases(benchmark=benchmark, task_ids=subset_task_ids)
    if args.site_name:
        allowed_sites = {str(site_name) for site_name in args.site_name}
        compatible_cases = [case for case in compatible_cases if str(case.site_name) in allowed_sites]
    selected_cases = select_stratified_cases(cases=compatible_cases, per_site=int(args.per_site))
    if not selected_cases:
        raise SystemExit("No compatible WebArena Verified tasks matched the requested benchmark slice.")
    selected_site_names = sorted({str(case.site_name) for case in selected_cases})

    started_site_names: list[str] = []
    if args.start_official_envs:
        started_site_names = _start_official_envs(repo_root=repo_root, site_names=selected_site_names)

    run_rows: list[dict[str, Any]] = []
    try:
        for case in selected_cases:
            print(
                json.dumps(
                    {
                        "progress": "task_start",
                        "task_id": int(case.task_id),
                        "site_name": str(case.site_name),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            started_at = time.perf_counter()
            task = benchmark.get_task(int(case.task_id))
            try:
                task_run = build_webarena_task_run(task=task, config=benchmark_config)
                browser_result = run_twinr_task(
                    env_file=env_path,
                    task_run=task_run,
                    project_root=repo_root,
                    max_steps=int(args.max_steps),
                    max_runtime_s=float(args.max_runtime_s),
                )
                evaluation = evaluate_browser_result(
                    benchmark=benchmark,
                    task=task,
                    browser_result=browser_result,
                )
                run_rows.append(
                    {
                        "task_id": int(case.task_id),
                        "site_name": str(case.site_name),
                        "expected_action": str(case.expected_action),
                        "intent": str(case.intent),
                        "runner_status": "completed",
                        "duration_s": round(time.perf_counter() - started_at, 3),
                        "browser_ok": bool(browser_result.ok),
                        "browser_status": str(browser_result.status),
                        "browser_summary": str(browser_result.summary),
                        "final_url": str(browser_result.final_url or ""),
                        "visited_urls": list(browser_result.data.get("visited_urls") or []),
                        "answer_markdown": browser_result.data.get("answer_markdown"),
                        "trace_path": browser_result.data.get("trace_path"),
                        "eval_pass": bool(evaluation["pass"]),
                        "eval_score": float(evaluation["eval_result"]["score"]),
                        "eval_status": str(evaluation["eval_result"]["status"]),
                        "agent_response": evaluation["agent_response"],
                        "eval_result": evaluation["eval_result"],
                    }
                )
                print(
                    json.dumps(
                        {
                            "progress": "task_done",
                            "task_id": int(case.task_id),
                            "site_name": str(case.site_name),
                            "eval_pass": bool(evaluation["pass"]),
                            "browser_ok": bool(browser_result.ok),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            except Exception as exc:  # pragma: no cover - exercised in the real runner
                run_rows.append(
                    {
                        "task_id": int(case.task_id),
                        "site_name": str(case.site_name),
                        "expected_action": str(case.expected_action),
                        "intent": str(case.intent),
                        "runner_status": "runner_error",
                        "duration_s": round(time.perf_counter() - started_at, 3),
                        "browser_ok": False,
                        "browser_status": "runner_error",
                        "browser_summary": str(exc),
                        "final_url": "",
                        "visited_urls": [],
                        "answer_markdown": None,
                        "trace_path": None,
                        "eval_pass": False,
                        "eval_score": 0.0,
                        "eval_status": "RUNNER_ERROR",
                        "agent_response": None,
                        "eval_result": None,
                    }
                )
                print(
                    json.dumps(
                        {
                            "progress": "task_error",
                            "task_id": int(case.task_id),
                            "site_name": str(case.site_name),
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        if args.stop_official_envs and started_site_names:
            _stop_official_envs(repo_root=repo_root, site_names=started_site_names)

    output_dir = repo_root / "artifacts" / "reports" / "browser_automation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        Path(args.output)
        if args.output
        else output_dir / f"webarena_verified_benchmark_{args.environment_label}.json"
    )

    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": str(args.environment_label),
        "benchmark": {
            "name": "webarena_verified_hard_agent_response_stratified_v1",
            "source_subset": str(args.subset_name),
            "selection_method": "per_site_even_spacing",
            "compatibility_filters": [
                "single_site",
                "single_start_url",
                "retrieve_only",
                "AgentResponseEvaluatorCfg_only",
            ],
            "per_site": int(args.per_site),
            "selected_site_names": selected_site_names,
            "selected_task_ids": [int(case.task_id) for case in selected_cases],
            "compatible_task_count": len(compatible_cases),
            "selected_task_count": len(selected_cases),
        },
        "versions": {
            "twinr_git_commit": _git_commit(repo_root),
            "webarena_verified": importlib_metadata.version("webarena-verified"),
            "playwright": importlib_metadata.version("playwright"),
        },
        "runner_command": " ".join(str(argument) for argument in sys.argv),
        "results": run_rows,
        "summary": summarize_benchmark_runs(runs=run_rows),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output_path),
                "selected_tasks": len(selected_cases),
                "passed_tasks": payload["summary"]["passed_tasks"],
                "pass_rate": payload["summary"]["pass_rate"],
                "browser_ok_rate": payload["summary"]["browser_ok_rate"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
