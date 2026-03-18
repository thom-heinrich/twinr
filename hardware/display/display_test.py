#!/usr/bin/env python3
"""Render a one-shot Twinr display test card.

Purpose
-------
Confirm that the configured Twinr display backend loads and can render a
known-good test pattern outside the main runtime loop.

Usage
-----
Command-line invocation example::

    python hardware/display/display_test.py --env-file .env

Inputs
------
- ``--env-file`` path to the Twinr environment file with display settings

Outputs
-------
- Renders a display test pattern on success
- Prints the resolved driver and backend-specific output details
- Exit code 0 on success
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.display import create_display_adapter


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the display test script."""

    parser = argparse.ArgumentParser(description="Render a test card on the configured Twinr display backend")
    parser.add_argument("--env-file", default=Path(__file__).resolve().parents[2] / ".env")
    return parser


def main() -> int:
    """Render the display test card and print the resolved configuration."""

    args = build_parser().parse_args()
    config = TwinrConfig.from_env(Path(args.env_file))
    display = create_display_adapter(config, emit=print)
    try:
        display.show_test_pattern()
        print("display_test=ok")
        print(f"display_driver={config.display_driver}")
        if config.display_driver == "hdmi_fbdev":
            print(f"display_fb_path={config.display_fb_path}")
            geometry = getattr(display, "geometry", None)
            if geometry is not None:
                print(f"display_mode={geometry.width}x{geometry.height}")
                print(f"display_bpp={geometry.bits_per_pixel}")
        elif config.display_driver == "hdmi_wayland":
            print(f"display_wayland_display={config.display_wayland_display}")
            print(f"display_wayland_runtime_dir={config.display_wayland_runtime_dir}")
            geometry = getattr(display, "geometry", None)
            if geometry is not None:
                print(f"display_mode={geometry.width}x{geometry.height}")
        else:
            print(f"display_busy_gpio={config.display_busy_gpio}")
    finally:
        display.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
