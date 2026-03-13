#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.display import TwinrStatusDisplayLoop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Twinr Waveshare status display loop")
    parser.add_argument("--env-file", default=Path(__file__).resolve().parents[2] / ".env")
    parser.add_argument("--duration", type=float, help="Optional max runtime in seconds")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = TwinrConfig.from_env(Path(args.env_file))
    loop = TwinrStatusDisplayLoop.from_config(config)
    return loop.run(duration_s=args.duration)


if __name__ == "__main__":
    raise SystemExit(main())
