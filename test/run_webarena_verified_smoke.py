#!/usr/bin/env python3
# ruff: noqa: E402
"""Run one official WebArena Verified smoke task through Twinr.

This script keeps the benchmark surface external: the official site still runs
from WebArena Verified's Docker image, while the tracked repo code only owns the
thin Twinr adapter plus this repeatable launcher. The default smoke targets task
386 on the public shopping site because it is retrieve-only and evaluates via
the official deterministic evaluator without requiring a HAR trace.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import argparse
import json
import subprocess
import sys

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env", help="Path to the Twinr environment file.")
    parser.add_argument("--environment-label", default="local", help="Label written into the JSON payload.")
    parser.add_argument(
        "--output",
        default="",
        help="Optional explicit JSON output path. Defaults to artifacts/reports/browser_automation/webarena_verified_smoke_<label>.json",
    )
    parser.add_argument("--task-id", type=int, default=386, help="Official WebArena Verified task id to run.")
    parser.add_argument("--max-steps", type=int, default=10, help="Twinr browser step budget.")
    parser.add_argument("--max-runtime-s", type=float, default=90.0, help="Twinr browser runtime budget.")
    parser.add_argument("--shopping-url", default="http://localhost:7770", help="URL for the official shopping site.")
    parser.add_argument("--shopping-admin-url", default="", help="URL for the official shopping_admin site.")
    parser.add_argument("--reddit-url", default="", help="URL for the official reddit site.")
    parser.add_argument("--gitlab-url", default="", help="URL for the official gitlab site.")
    parser.add_argument("--wikipedia-url", default="", help="URL for the official wikipedia site.")
    parser.add_argument("--map-url", default="", help="URL for the official map site.")
    parser.add_argument(
        "--start-official-env",
        default="",
        choices=("", "shopping", "shopping_admin", "reddit", "gitlab", "wikipedia", "map"),
        help="Optional official WebArena Verified site to start before the smoke run.",
    )
    parser.add_argument(
        "--stop-official-env",
        action="store_true",
        help="Stop the official environment named in --start-official-env after the run.",
    )
    return parser.parse_args()


def _run_official_env_command(*, command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    args = _parse_args()
    repo_root = _REPO_ROOT
    env_path = (repo_root / args.env_file).resolve() if not Path(args.env_file).is_absolute() else Path(args.env_file)

    if args.start_official_env:
        _run_official_env_command(
            command=[
                str(repo_root / ".venv" / "bin" / "webarena-verified"),
                "env",
                "start",
                "--site",
                str(args.start_official_env),
            ]
        )

    benchmark_config = build_webarena_verified_config(
        shopping_url=args.shopping_url or None,
        shopping_admin_url=args.shopping_admin_url or None,
        reddit_url=args.reddit_url or None,
        gitlab_url=args.gitlab_url or None,
        wikipedia_url=args.wikipedia_url or None,
        map_url=args.map_url or None,
    )
    benchmark, task = load_webarena_verified_task(task_id=int(args.task_id), config=benchmark_config)
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

    output_dir = repo_root / "artifacts" / "reports" / "browser_automation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        Path(args.output)
        if args.output
        else output_dir / f"webarena_verified_smoke_{args.environment_label}.json"
    )

    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": str(args.environment_label),
        "task": {
            "task_id": int(task.task_id),
            "sites": [str(site.value) for site in task.sites],
            "intent": str(task.intent),
            "start_urls": list(task.start_urls),
        },
        "task_run": {
            "start_url": task_run.start_url,
            "goal": task_run.goal,
            "results_schema": task_run.results_schema,
        },
        "browser_result": {
            "ok": bool(browser_result.ok),
            "status": browser_result.status,
            "summary": browser_result.summary,
            "final_url": browser_result.final_url,
            "error_code": browser_result.error_code,
            "answer_markdown": browser_result.data.get("answer_markdown"),
            "key_points": list(browser_result.data.get("key_points") or []),
            "visited_urls": list(browser_result.data.get("visited_urls") or []),
            "trace_path": browser_result.data.get("trace_path"),
        },
        "evaluation": evaluation,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output_path),
                "task_id": int(task.task_id),
                "browser_ok": bool(browser_result.ok),
                "eval_pass": bool(evaluation["pass"]),
                "eval_score": evaluation["eval_result"]["score"],
            },
            ensure_ascii=False,
        )
    )

    if args.stop_official_env and args.start_official_env:
        _run_official_env_command(
            command=[
                str(repo_root / ".venv" / "bin" / "webarena-verified"),
                "env",
                "stop",
                "--site",
                str(args.start_official_env),
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
