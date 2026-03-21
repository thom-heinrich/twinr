"""Run a bounded co-attention stability soak against live world intelligence.

Purpose
-------
Exercise repeated world-intelligence refresh cycles against a real runtime
environment and verify that ``co_attention`` only grows when new shared
evidence appears. This protects Twinr against a subtle failure mode where
successful but stale feed refreshes would slowly inflate shared-thread state.

Usage
-----
Command-line invocation examples::

    PYTHONPATH=src python3 test/run_world_intelligence_co_attention_soak.py --env-file .env --cycles 3 --force
    PYTHONPATH=src python3 test/run_world_intelligence_co_attention_soak.py --env-file /twinr/.env --cycles 4 --sleep-s 2 --output report.json

Inputs
------
- ``--env-file``: Twinr runtime env file used to build the live config.
- ``--cycles``: Number of refresh cycles to run.
- ``--sleep-s``: Optional pause between cycles.
- ``--force``: Force refresh on every cycle to create a deterministic stress path.
- ``--allow-recalibration``: Also run the slower discovery/recalibration phase inside each cycle.

Outputs
-------
- JSON summary written to stdout.
- Optional JSON artifact written to ``--output``.

Notes
-----
The soak is intentionally bounded. It does not attempt a long-running daemon
loop; it captures enough repeated cycles to detect overreaction in the current
policy implementation.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality.intelligence.service import WorldIntelligenceService
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore


def _utc_now_iso() -> str:
    """Return the current UTC wall clock as ISO-8601 text."""

    return datetime.now(timezone.utc).isoformat()


def _signal_snapshot(signal: object) -> dict[str, object]:
    """Project one interest signal into a compact soak-friendly mapping."""

    return {
        "topic": getattr(signal, "topic", None),
        "engagement_state": getattr(signal, "engagement_state", None),
        "ongoing_interest": getattr(signal, "ongoing_interest", None),
        "ongoing_interest_score": getattr(signal, "ongoing_interest_score", None),
        "co_attention_state": getattr(signal, "co_attention_state", None),
        "co_attention_score": getattr(signal, "co_attention_score", None),
        "co_attention_count": getattr(signal, "co_attention_count", None),
    }


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the bounded soak runner."""

    parser = argparse.ArgumentParser(
        description="Run a bounded world-intelligence co-attention soak."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Twinr env file used to build the live runtime config.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=3,
        help="Number of refresh cycles to run.",
    )
    parser.add_argument(
        "--sleep-s",
        type=float,
        default=1.0,
        help="Pause between cycles in seconds.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force every refresh cycle instead of waiting for due cadence.",
    )
    parser.add_argument(
        "--allow-recalibration",
        action="store_true",
        help="Allow discovery/recalibration inside each cycle instead of running refresh-only soak cycles.",
    )
    parser.add_argument(
        "--top-topics",
        type=int,
        default=8,
        help="How many top interest signals to capture in each cycle snapshot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for the soak artifact.",
    )
    return parser


def run_soak(
    *,
    env_file: Path,
    cycles: int,
    sleep_s: float,
    force: bool,
    top_topics: int,
    allow_recalibration: bool,
) -> dict[str, Any]:
    """Run the bounded soak and return one JSON-safe report mapping."""

    config = TwinrConfig.from_env(env_file)
    remote_state = LongTermRemoteStateStore.from_config(config)
    service = WorldIntelligenceService(config=config, remote_state=remote_state)

    cycle_reports: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    previous_counts: dict[str, int] = {}

    for cycle_index in range(max(1, int(cycles))):
        refresh = service.maybe_refresh(
            force=force,
            allow_recalibration=allow_recalibration,
        )
        state = service.store.load_state(config=config, remote_state=remote_state)
        interest_signals = tuple(state.interest_signals[: max(1, int(top_topics))])
        counts = {
            signal.topic: int(signal.co_attention_count)
            for signal in interest_signals
        }
        no_new_shared_evidence = (
            len(refresh.world_signals) == 0
            and len(refresh.continuity_threads) == 0
        )
        if cycle_index > 0 and no_new_shared_evidence:
            for topic, count in counts.items():
                previous = previous_counts.get(topic)
                if previous is None:
                    continue
                if count > previous:
                    violations.append(
                        {
                            "cycle": cycle_index + 1,
                            "topic": topic,
                            "previous_co_attention_count": previous,
                            "current_co_attention_count": count,
                            "reason": "co_attention_count increased despite no new shared evidence",
                        }
                    )
        cycle_reports.append(
            {
                "cycle": cycle_index + 1,
                "checked_at": _utc_now_iso(),
                "refresh_status": refresh.status,
                "refreshed": refresh.refreshed,
                "force": force,
                "refreshed_subscription_ids": list(refresh.refreshed_subscription_ids),
                "world_signal_count": len(refresh.world_signals),
                "continuity_thread_count": len(refresh.continuity_threads),
                "no_new_shared_evidence": no_new_shared_evidence,
                "interest_signals": [_signal_snapshot(signal) for signal in interest_signals],
            }
        )
        previous_counts = counts
        if cycle_index + 1 < max(1, int(cycles)) and sleep_s > 0:
            time.sleep(max(0.0, float(sleep_s)))

    return {
        "recorded_at": _utc_now_iso(),
        "env_file": str(env_file.resolve()),
        "cycles": max(1, int(cycles)),
        "force": bool(force),
        "allow_recalibration": bool(allow_recalibration),
        "sleep_s": max(0.0, float(sleep_s)),
        "ok": not violations,
        "violations": violations,
        "cycle_reports": cycle_reports,
    }


def main() -> int:
    """Run the CLI entrypoint and emit one JSON report."""

    parser = _build_argument_parser()
    args = parser.parse_args()
    report = run_soak(
        env_file=args.env_file,
        cycles=args.cycles,
        sleep_s=args.sleep_s,
        force=args.force,
        top_topics=args.top_topics,
        allow_recalibration=args.allow_recalibration,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
