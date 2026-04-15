#!/usr/bin/env python3
"""Audit the current authoritative release against the deployed Pi release.

Purpose
-------
Use this operator script when you want one compact view of whether the current
leading-repo release in ``/home/thh/twinr`` matches the persisted deployed
release metadata on the Pi under ``/twinr``. The report joins the local
workspace status, the locally computed release id, the Pi's
``current_release_manifest.json``, and a live checksum drift probe.

Usage
-----
Command-line invocation examples::

    python3 hardware/ops/audit_pi_release.py
    python3 hardware/ops/audit_pi_release.py --max-change-lines 10
    python3 hardware/ops/audit_pi_release.py --remote-root /twinr

Outputs
-------
- One compact JSON object describing the current local-vs-Pi release state.
- Exit code 0 when the local release, remote manifest, and drift probe all agree.
- Exit code 1 when the Pi is missing the manifest, the manifest disagrees, or drift exists.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path

from _repo_python import PROJECT_ROOT, ensure_repo_python

ensure_repo_python()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from twinr.ops.pi_release_audit import audit_pi_release  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the Pi release audit."""

    parser = argparse.ArgumentParser(
        description="Compare the current local release against the deployed Pi release manifest and drift state.",
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
        help="Twinr runtime checkout root on the Raspberry Pi.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=120.0,
        help="Per-SSH command timeout in seconds.",
    )
    parser.add_argument(
        "--max-change-lines",
        type=int,
        default=40,
        help="Maximum number of drift lines to keep in the embedded mirror audit.",
    )
    return parser


def main() -> int:
    """Run the Pi release audit and print one JSON payload."""

    args = build_parser().parse_args()
    result = audit_pi_release(
        project_root=args.project_root,
        pi_env_path=args.pi_env_file,
        remote_root=args.remote_root,
        timeout_s=args.timeout_s,
        max_change_lines=args.max_change_lines,
    )
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0 if result.in_sync else 1


if __name__ == "__main__":
    raise SystemExit(main())
