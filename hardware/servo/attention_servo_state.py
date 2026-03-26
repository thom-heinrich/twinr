#!/usr/bin/env python3
"""Inspect and change Twinr's persisted continuous-servo startup state.

Purpose
-------
Give operators one explicit, bounded control point for the feedbackless
continuous-rotation attention servo. This script only edits the persisted
runtime-state file; the running Twinr process picks those changes up live.

Usage
-----
Command-line invocation examples::

    python3 hardware/servo/attention_servo_state.py status
    python3 hardware/servo/attention_servo_state.py hold-current-zero
    python3 hardware/servo/attention_servo_state.py hold
    python3 hardware/servo/attention_servo_state.py arm
    python3 hardware/servo/attention_servo_state.py return-to-estimated-zero

Inputs
------
- ``--env-file`` path to the Twinr environment file used to resolve
  ``TWINR_ATTENTION_SERVO_STATE_PATH``
- ``--state-path`` optional direct override for the persisted state file
- ``status`` to print the current persisted state
- ``hold-current-zero`` to treat the current physical pose as virtual ``0°``
  and keep startup hold enabled
- ``hold`` to keep startup hold enabled while preserving the current virtual
  heading estimate
- ``arm`` to disable startup hold and let live follow resume
- ``return-to-estimated-zero`` to ask the running runtime to drive back to the
  stored virtual ``0°`` only when the saved uncertainty stays within the
  configured bound

Outputs
-------
- Prints the resolved state path plus the saved hold/heading flags
- Exit code 0 on success
- Exit code 2 when ``arm`` is requested before a zero reference exists
  or when ``return-to-estimated-zero`` is requested before a zero reference
  exists

Notes
-----
This script does not move the servo directly. The running runtime observes the
updated state file and switches between manual hold and live follow on the next
servo update tick.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.servo_state import AttentionServoRuntimeState, AttentionServoStateStore


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the attention-servo state helper."""

    parser = argparse.ArgumentParser(
        description="Inspect or change the persisted startup-hold state for Twinr's continuous attention servo"
    )
    parser.add_argument(
        "--env-file",
        default=Path(__file__).resolve().parents[2] / ".env",
        help="Path to the Twinr .env file used to resolve TWINR_ATTENTION_SERVO_STATE_PATH",
    )
    parser.add_argument(
        "--state-path",
        default=None,
        help="Optional direct override for the persisted attention-servo state file",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="Print the current persisted attention-servo state")
    subparsers.add_parser(
        "hold-current-zero",
        help="Treat the current physical pose as virtual 0° and enable manual hold",
    )
    subparsers.add_parser(
        "hold",
        help="Enable manual hold while preserving the current virtual heading estimate",
    )
    subparsers.add_parser(
        "arm",
        help="Disable manual hold so the running runtime may resume live follow",
    )
    subparsers.add_parser(
        "return-to-estimated-zero",
        help="Ask the runtime to drive back to the stored virtual 0° when uncertainty allows it",
    )
    return parser


def _resolve_state_store(*, env_file: str | Path, state_path_override: str | None) -> AttentionServoStateStore:
    """Resolve the persisted attention-servo state file from CLI inputs."""

    if state_path_override is not None and str(state_path_override).strip():
        return AttentionServoStateStore(Path(str(state_path_override).strip()))
    config = TwinrConfig.from_env(Path(env_file))
    return AttentionServoStateStore(config.attention_servo_state_path)


def _print_state(store: AttentionServoStateStore, state: AttentionServoRuntimeState | None) -> None:
    """Render one short operator-facing snapshot of the resolved state file."""

    print(f"attention_servo_state_path={store.path}")
    if state is None:
        print("attention_servo_state=missing")
        return
    print(f"heading_degrees={state.heading_degrees:.3f}")
    print(f"heading_uncertainty_degrees={state.heading_uncertainty_degrees:.3f}")
    print(f"hold_until_armed={'true' if state.hold_until_armed else 'false'}")
    print(f"return_to_zero_requested={'true' if state.return_to_zero_requested else 'false'}")
    print(f"zero_reference_confirmed={'true' if state.zero_reference_confirmed else 'false'}")
    if state.return_to_zero_requested:
        print("attention_servo_state=returning_to_estimated_zero")
        return
    print("attention_servo_state=manual_hold" if state.hold_until_armed else "attention_servo_state=armed")


def _updated_state_for_command(
    *,
    command: str,
    current_state: AttentionServoRuntimeState,
    updated_at: float,
) -> AttentionServoRuntimeState:
    """Return the new persisted state for one explicit operator command."""

    if command == "hold-current-zero":
        return current_state.adopt_current_as_zero(updated_at=updated_at)
    if command == "hold":
        return current_state.hold_current_heading(updated_at=updated_at)
    if command == "arm":
        if not current_state.zero_reference_confirmed:
            raise ValueError("Cannot arm the continuous attention servo before a zero reference is confirmed")
        return current_state.arm_follow(updated_at=updated_at)
    if command == "return-to-estimated-zero":
        if not current_state.zero_reference_confirmed:
            raise ValueError(
                "Cannot return the continuous attention servo to estimated zero before a zero reference is confirmed"
            )
        return current_state.request_return_to_estimated_zero(updated_at=updated_at)
    raise ValueError(f"Unsupported command: {command}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the attention-servo state helper and print the resulting state."""

    args = build_parser().parse_args(None if argv is None else list(argv))
    store = _resolve_state_store(env_file=args.env_file, state_path_override=args.state_path)
    if args.command == "status":
        _print_state(store, store.load())
        return 0
    current_state = store.load_or_default()
    try:
        updated_state = _updated_state_for_command(
            command=str(args.command),
            current_state=current_state,
            updated_at=time.monotonic(),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    store.save(updated_state)
    _print_state(store, updated_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
