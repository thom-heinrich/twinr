#!/usr/bin/env python3
"""Run the standalone Twinr status display loop.

Purpose
-------
Launch only the display loop so operators can validate the configured display
backend without starting the full hardware or realtime runtime paths.

Usage
-----
Command-line invocation examples::

    python hardware/display/run_display_loop.py --env-file .env
    python hardware/display/run_display_loop.py --env-file .env --duration 30

Inputs
------
- ``--env-file`` path to the Twinr environment file with display settings
- ``--duration`` optional maximum runtime in seconds

Outputs
-------
- Streams status frames to the configured display until stopped or timed out
- Exit code from ``TwinrStatusDisplayLoop.run()``
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.display import TwinrStatusDisplayLoop


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the display loop runner."""

    parser = argparse.ArgumentParser(description="Run the Twinr status display loop for the configured display backend")
    parser.add_argument("--env-file", default=Path(__file__).resolve().parents[2] / ".env")
    parser.add_argument("--duration", type=float, help="Optional max runtime in seconds")
    return parser


def main() -> int:
    """Run the configured display loop until it exits or times out."""

    args = build_parser().parse_args()
    config = TwinrConfig.from_env(Path(args.env_file))
    loop = TwinrStatusDisplayLoop.from_config(config)
    return loop.run(duration_s=args.duration)


if __name__ == "__main__":
    raise SystemExit(main())
