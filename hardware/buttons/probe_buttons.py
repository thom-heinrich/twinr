#!/usr/bin/env python3
"""Probe Twinr button GPIO lines from the command line.

Purpose
-------
Verify configured or ad-hoc button GPIO mappings on a Raspberry Pi without
starting the full Twinr runtime.

Usage
-----
Command-line invocation examples::

    python hardware/buttons/probe_buttons.py --env-file .env --configured --duration 15
    python hardware/buttons/probe_buttons.py --env-file .env --lines 22,23 --duration 10

Inputs
------
- ``--env-file`` path to the Twinr environment file
- ``--configured`` to use the configured green/yellow button lines
- ``--lines`` to probe ad-hoc GPIO lines instead of configured buttons

Outputs
-------
- Prints one line per observed button event
- Exit code 0 when at least one event is observed, 1 otherwise
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.hardware.buttons import GpioButtonMonitor, build_button_bindings, build_probe_bindings


def _parse_lines(raw: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    """Parse a comma-separated GPIO line override."""

    if raw is None or not raw.strip():
        return default
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the button probe script."""

    parser = argparse.ArgumentParser(description="Probe Twinr GPIO buttons")
    parser.add_argument("--env-file", default=".env", help="Path to the Twinr .env file")
    parser.add_argument("--chip", help="Override the GPIO chip name")
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="How many seconds to wait for button presses",
    )
    parser.add_argument(
        "--lines",
        help="Comma-separated GPIO lines to probe when buttons are not configured",
    )
    parser.add_argument(
        "--configured",
        action="store_true",
        help="Watch the configured green/yellow button GPIO lines instead of probe mode",
    )
    return parser


def main() -> int:
    """Run the GPIO button probe and print observed events."""

    args = build_parser().parse_args()
    config = TwinrConfig.from_env(Path(args.env_file))
    chip_name = args.chip or config.gpio_chip

    if args.configured:
        bindings = build_button_bindings(config)
        if not bindings:
            raise SystemExit(
                "No configured buttons found. Set TWINR_GREEN_BUTTON_GPIO and TWINR_YELLOW_BUTTON_GPIO first."
            )
    else:
        bindings = build_probe_bindings(_parse_lines(args.lines, config.button_probe_lines))

    binding_summary = ", ".join(f"{binding.name}=GPIO{binding.line_offset}" for binding in bindings)
    print(f"watching chip={chip_name} duration={args.duration:.1f}s")
    print(f"bindings: {binding_summary}")
    print(
        f"active_low={str(config.button_active_low).lower()} bias={config.button_bias} debounce_ms={config.button_debounce_ms}"
    )
    print("Press the physical buttons while the probe is running.")

    seen_events = 0
    with GpioButtonMonitor(
        chip_name=chip_name,
        bindings=bindings,
        active_low=config.button_active_low,
        bias=config.button_bias,
        debounce_ms=config.button_debounce_ms,
    ) as monitor:
        for event in monitor.iter_events(duration_s=args.duration):
            seen_events += 1
            print(
                f"event name={event.name} line=GPIO{event.line_offset} action={event.action.value} edge={event.raw_edge} timestamp_ns={event.timestamp_ns}"
            )

    if seen_events == 0:
        print("No button events captured.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
