#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from twinr.config import TwinrConfig
from twinr.hardware.pir import GpioPirMonitor, PirBinding, build_pir_binding


def _parse_bool(raw: str) -> bool:
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {raw}")


def build_parser() -> argparse.ArgumentParser:
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
