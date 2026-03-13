#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.display import WaveshareEPD4In2V2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a test card on the Waveshare 4.2 V2 display")
    parser.add_argument("--env-file", default=Path(__file__).resolve().parents[2] / ".env")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = TwinrConfig.from_env(Path(args.env_file))
    display = WaveshareEPD4In2V2.from_config(config)
    display.show_test_pattern()
    print("display_test=ok")
    print(f"display_driver={config.display_driver}")
    print(f"display_busy_gpio={config.display_busy_gpio}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
