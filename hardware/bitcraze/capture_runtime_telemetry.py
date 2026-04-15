#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cflib>=0.1.31",
# ]
# ///
"""Capture one bounded Crazyflie runtime telemetry snapshot.

Purpose
-------
Connect to the Crazyflie over the normal Bitcraze workspace, start one explicit
runtime telemetry profile, wait a short bounded settle window, and emit the
typed snapshot as JSON. This script is the daemon-facing adapter for the
shared telemetry runtime in ``src/twinr/hardware/crazyflie_telemetry.py``.

Usage
-----
Capture one operator snapshot directly::

    /twinr/bitcraze/.venv/bin/python hardware/bitcraze/capture_runtime_telemetry.py \
        --workspace /twinr/bitcraze \
        --profile operator \
        --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from run_hover_test import DEFAULT_URI, _import_cflib  # noqa: E402
from twinr.hardware.crazyflie_telemetry import (  # noqa: E402
    CrazyflieTelemetryRuntime,
    TelemetryProfile,
    snapshot_to_payload,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument(
        "--profile",
        choices=tuple((profile.value for profile in TelemetryProfile)),
        default=TelemetryProfile.OPERATOR.value,
    )
    parser.add_argument("--connect-settle-s", type=float, default=0.6)
    parser.add_argument("--profile-settle-s", type=float, default=0.35)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--period-ms", type=int, default=100)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    workspace = args.workspace.expanduser().resolve(strict=False)
    workspace.mkdir(parents=True, exist_ok=True)
    cache_dir = workspace / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    crtp, crazyflie_cls, log_config_cls, _multiranger_cls, sync_crazyflie_cls, _sync_logger_cls = _import_cflib()
    crtp.init_drivers()
    crazyflie = crazyflie_cls(rw_cache=str(cache_dir))
    telemetry: CrazyflieTelemetryRuntime | None = None

    try:
        with sync_crazyflie_cls(args.uri, cf=crazyflie) as sync_cf:
            if float(args.connect_settle_s) > 0.0:
                time.sleep(float(args.connect_settle_s))
            telemetry = CrazyflieTelemetryRuntime(
                sync_cf,
                log_config_cls,
                profile=TelemetryProfile(args.profile),
                max_samples=max(1, int(args.max_samples)),
                period_in_ms=max(10, int(args.period_ms)),
            )
            telemetry.start()
            if float(args.profile_settle_s) > 0.0:
                time.sleep(float(args.profile_settle_s))
            payload = {
                "uri": str(args.uri),
                "workspace": str(workspace),
                "profile": str(args.profile),
                "telemetry": snapshot_to_payload(telemetry.latest_snapshot()),
            }
    finally:
        if telemetry is not None:
            telemetry.stop()

    if args.json:
        print(json.dumps(payload, sort_keys=True, indent=2))
    else:
        print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
