#!/usr/bin/env python3
"""Watch and heal drift between the leading repo and the Pi checkout.

Purpose
-------
Run this script on the development machine when `/home/thh/twinr` must stay
authoritative and `/twinr` on the Raspberry Pi must not drift away. The script
mirrors the source-managed part of the repo to the Pi, preserves Pi-local
runtime paths such as `.env`, `.venv`, `state/`, and `artifacts/`, and can run
continuously as a watchdog.

Usage
-----
Command-line invocation examples::

    python3 hardware/ops/watch_pi_repo_mirror.py --once
    python3 hardware/ops/watch_pi_repo_mirror.py --interval-s 5
    python3 hardware/ops/watch_pi_repo_mirror.py --interval-s 5 --metadata-only

Outputs
-------
- One compact JSON line per successful cycle.
- One compact JSON line per failed watchdog attempt in continuous mode.
- Exit code 0 on success, 1 on a failing `--once` probe, 130 on Ctrl+C.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
_MODULE_PATH = PROJECT_ROOT / "src" / "twinr" / "ops" / "pi_repo_mirror.py"
_SPEC = importlib.util.spec_from_file_location("twinr_ops_pi_repo_mirror", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load mirror module from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
DEFAULT_PROTECTED_PATTERNS = _MODULE.DEFAULT_PROTECTED_PATTERNS
PiRepoMirrorCycleResult = _MODULE.PiRepoMirrorCycleResult
PiRepoMirrorWatchdog = _MODULE.PiRepoMirrorWatchdog


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the repo mirror watchdog."""

    parser = argparse.ArgumentParser(
        description="Mirror /home/thh/twinr to the Pi runtime checkout and watchdog drift.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Leading Twinr repo root that is treated as authoritative.",
    )
    parser.add_argument(
        "--pi-env-file",
        type=Path,
        default=PROJECT_ROOT / ".env.pi",
        help="Path to the Pi SSH credential env file.",
    )
    parser.add_argument(
        "--remote-root",
        default="/twinr",
        help="Runtime checkout root on the Raspberry Pi.",
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=5.0,
        help="Watchdog sleep interval between sync cycles.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Optional total watchdog runtime for bounded runs.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=120.0,
        help="Per-rsync timeout in seconds.",
    )
    parser.add_argument(
        "--checksum-every-s",
        type=float,
        default=300.0,
        help="When using --metadata-only, run one full checksum audit this often.",
    )
    checksum_group = parser.add_mutually_exclusive_group()
    checksum_group.add_argument(
        "--checksum-always",
        dest="checksum_always",
        action="store_true",
        default=True,
        help="Compare file contents on every cycle. This is the default.",
    )
    checksum_group.add_argument(
        "--metadata-only",
        dest="checksum_always",
        action="store_false",
        help="Use rsync quick-check (size+mtime) between periodic checksum audits.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run exactly one cycle and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report drift without applying a sync.",
    )
    parser.add_argument(
        "--max-change-lines",
        type=int,
        default=40,
        help="Maximum number of itemized rsync change lines to include in JSON output.",
    )
    parser.add_argument(
        "--protect",
        action="append",
        default=[],
        help="Additional perishable rsync filter pattern to preserve on the Pi.",
    )
    return parser


def main() -> int:
    """Run the repo mirror watchdog and print compact JSON events."""

    args = build_parser().parse_args()
    protected_patterns = tuple(DEFAULT_PROTECTED_PATTERNS) + tuple(args.protect)
    watchdog = PiRepoMirrorWatchdog.from_env(
        project_root=args.project_root,
        pi_env_path=args.pi_env_file,
        remote_root=args.remote_root,
        protected_patterns=protected_patterns,
        timeout_s=args.timeout_s,
    )
    if args.once:
        try:
            result = watchdog.probe_once(
                apply_sync=not args.dry_run,
                checksum=args.checksum_always,
                max_change_lines=args.max_change_lines,
            )
        except KeyboardInterrupt:
            return 130
        except Exception as exc:
            print(json.dumps({"event": "pi_repo_mirror_error", "error": str(exc)}, ensure_ascii=False))
            return 1
        print(_cycle_payload(result), flush=True)
        return 0
    try:
        run_result = watchdog.run(
            interval_s=args.interval_s,
            duration_s=args.duration_s,
            checksum_every_s=None if args.checksum_always else args.checksum_every_s,
            checksum_always=args.checksum_always,
            apply_sync=not args.dry_run,
            max_change_lines=args.max_change_lines,
            on_cycle=lambda cycle: print(_cycle_payload(cycle), flush=True),
            on_error=lambda exc, failure_count: print(
                json.dumps(
                    {
                        "event": "pi_repo_mirror_error",
                        "error": str(exc),
                        "failure_count": failure_count,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            ),
        )
    except KeyboardInterrupt:
        return 130
    print(
        json.dumps(
            {
                "event": "pi_repo_mirror_summary",
                "cycles": run_result.cycles,
                "syncs_applied": run_result.syncs_applied,
                "failures": run_result.failures,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def _cycle_payload(result: PiRepoMirrorCycleResult) -> str:
    payload = asdict(result)
    payload["event"] = "pi_repo_mirror_cycle"
    return json.dumps(payload, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
