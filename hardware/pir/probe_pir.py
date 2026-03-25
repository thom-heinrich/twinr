#!/usr/bin/env python3
"""Probe Twinr PIR motion GPIO from the command line.

Purpose
-------
Verify the configured PIR motion input or a temporary GPIO override without
starting the full Twinr runtime.

Usage
-----
Command-line invocation examples::

    python hardware/pir/probe_pir.py --env-file .env --duration 30
    python hardware/pir/probe_pir.py --env-file .env --line 26 --active-high true

Inputs
------
- ``--env-file`` path to the Twinr environment file
- ``--line`` optional GPIO override for ad-hoc probing
- ``--active-high`` / ``--bias`` / ``--debounce-ms`` overrides for probe tuning

Outputs
-------
- Prints the initial PIR level and one line per observed motion event
- Exit code 0 when motion is detected, 1 otherwise
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.hardware.pir import GpioPirMonitor, PirBinding, build_pir_binding


def _parse_bool(raw: str) -> bool:
    """Parse a flexible CLI boolean literal."""

    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {raw}")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the PIR probe script."""

    parser = argparse.ArgumentParser(description="Probe Twinr PIR motion input")
    parser.add_argument("--env-file", default=".env", help="Path to the Twinr .env file")
    parser.add_argument("--chip", help="Override the GPIO chip name")
    parser.add_argument("--line", type=int, help="Override the PIR GPIO line")
    parser.add_argument("--duration", type=float, default=20.0, help="Probe duration in seconds")
    parser.add_argument("--active-high", type=_parse_bool, help="Override PIR polarity")
    parser.add_argument("--bias", help="Override GPIO bias mode")
    parser.add_argument("--debounce-ms", type=int, help="Override PIR debounce in milliseconds")
    return parser


def main() -> int:
    """Run the PIR probe and print observed motion events."""

    args = build_parser().parse_args()
    config = TwinrConfig.from_env(Path(args.env_file))
    binding = PirBinding(name="pir", line_offset=args.line) if args.line is not None else build_pir_binding(config)
    monitor = GpioPirMonitor(
        chip_name=args.chip or config.gpio_chip,
        binding=binding,
        active_high=config.pir_active_high if args.active_high is None else args.active_high,
        bias=args.bias or config.pir_bias,
        debounce_ms=args.debounce_ms or config.pir_debounce_ms,
    )

    print(f"watching chip={monitor.chip_name} line=GPIO{binding.line_offset} duration={args.duration:.1f}s")
    print(
        f"active_high={str(monitor.active_high).lower()} bias={monitor.bias} debounce_ms={monitor.debounce_ms}"
    )
    print("Move in front of the PIR while the probe is running.")

    motion_events = 0
    with monitor:
        print(f"initial_value={monitor.snapshot_value()} motion_detected={str(monitor.motion_detected()).lower()}")
        for event in monitor.iter_events(duration_s=args.duration, poll_timeout=0.25):
            print(
                "event line=GPIO{line} motion={motion} edge={edge} timestamp_ns={ts}".format(
                    line=event.line_offset,
                    motion=str(event.motion_detected).lower(),
                    edge=event.raw_edge,
                    ts=event.timestamp_ns,
                )
            )
            if event.motion_detected:
                motion_events += 1

    if motion_events == 0:
        print("No PIR motion events captured.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
