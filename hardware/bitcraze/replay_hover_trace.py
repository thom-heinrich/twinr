#!/usr/bin/env python3
"""Replay one stored hover report through the real bounded hover primitive.

Usage
-----
Replay a previously captured hover report and optional phase trace::

    python3 hardware/bitcraze/replay_hover_trace.py \
        --report-json /tmp/hover-report.json \
        --trace-file /tmp/hover-trace.jsonl \
        --json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from hover_primitive import HoverPrimitiveConfig, StatefulHoverPrimitive  # noqa: E402
from run_hover_test import (  # noqa: E402
    HOVER_RUNTIME_MODE_SITL,
    HOVER_RUNTIME_MODE_HARDWARE,
    HOVER_MICRO_LIFTOFF_HEIGHT_M,
    HOVER_TAKEOFF_TARGET_HEIGHT_TOLERANCE_M,
    _vertical_bootstrap_config_for_runtime_mode,
    _evaluate_primitive_outcome,
    _latest_ground_distance_from_telemetry,
    _latest_ground_distance_from_sitl_telemetry,
    _latest_stability_observation_from_telemetry,
    _stability_config_for_runtime_mode,
    classify_hover_outcome,
    evaluate_hover_stability,
    summarize_hover_telemetry,
)
from twinr.hardware.crazyflie_hover_replay import (  # noqa: E402
    CrazyflieTelemetryReplayRuntime,
    HoverReplayArtifact,
    HoverReplayClock,
    HoverReplayCommander,
    HoverReplayCommanderCommand,
    HoverReplayPhaseEvent,
    HoverReplayTraceWriter,
    _require_float,
    load_hover_replay_artifact,
)


class _ReplayCFHandle:
    """Provide the minimal commander surface required by the hover primitive."""

    def __init__(self, commander: HoverReplayCommander) -> None:
        self.commander = commander


def _coerce_report_failures(report_payload: dict[str, object]) -> tuple[str, ...]:
    raw = report_payload.get("failures")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return ()
    return tuple(str(item) for item in raw)


def run_hover_replay(
    artifact: HoverReplayArtifact,
    *,
    runtime_mode: str,
    setpoint_period_s: float,
    min_clearance_m: float | None = None,
    include_phase_events: bool = False,
    include_command_log: bool = False,
) -> dict[str, object]:
    if not artifact.telemetry_samples:
        raise RuntimeError("hover replay artifact contains no telemetry samples")

    report_payload = artifact.report_payload
    replay_start_timestamp_ms_raw = report_payload.get("replay_start_timestamp_ms")
    replay_samples = artifact.telemetry_samples
    replay_start_timestamp_ms: int | None = None
    if replay_start_timestamp_ms_raw is not None:
        replay_start_timestamp_ms = int(_require_float(replay_start_timestamp_ms_raw, field_name="replay_start_timestamp_ms"))
        replay_samples = tuple(
            sample for sample in artifact.telemetry_samples if int(sample.timestamp_ms) >= replay_start_timestamp_ms
        )
        if not replay_samples:
            raise RuntimeError(
                "hover replay artifact declared replay_start_timestamp_ms="
                f"{replay_start_timestamp_ms} but no telemetry samples remain after trimming"
            )
    target_height_m = _require_float(report_payload["height_m"], field_name="height_m")
    hover_duration_s = _require_float(report_payload["hover_duration_s"], field_name="hover_duration_s")
    takeoff_velocity_mps = _require_float(
        report_payload["takeoff_velocity_mps"],
        field_name="takeoff_velocity_mps",
    )
    land_velocity_mps = _require_float(
        report_payload["land_velocity_mps"],
        field_name="land_velocity_mps",
    )

    clock = HoverReplayClock()
    replay_runtime = CrazyflieTelemetryReplayRuntime(
        replay_samples,
        monotonic=clock.monotonic,
        available_blocks=artifact.available_blocks,
        skipped_blocks=artifact.skipped_blocks,
    )
    commander = HoverReplayCommander(monotonic=clock.monotonic)
    replay_trace = HoverReplayTraceWriter(monotonic=clock.monotonic)
    if runtime_mode == HOVER_RUNTIME_MODE_SITL:
        def ground_distance_provider() -> object:
            return _latest_ground_distance_from_sitl_telemetry(replay_runtime)
    else:
        def ground_distance_provider() -> object:
            return _latest_ground_distance_from_telemetry(replay_runtime)
    primitive = StatefulHoverPrimitive(
        _ReplayCFHandle(commander),
        ground_distance_provider=ground_distance_provider,
        stability_provider=lambda: _latest_stability_observation_from_telemetry(replay_runtime),
        trace_writer=replay_trace,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    primitive_outcome = primitive.run(
        HoverPrimitiveConfig(
            target_height_m=target_height_m,
            hover_duration_s=hover_duration_s,
            takeoff_velocity_mps=takeoff_velocity_mps,
            land_velocity_mps=land_velocity_mps,
            setpoint_period_s=float(setpoint_period_s),
            micro_liftoff_height_m=HOVER_MICRO_LIFTOFF_HEIGHT_M,
            takeoff_confirm_target_height_tolerance_m=HOVER_TAKEOFF_TARGET_HEIGHT_TOLERANCE_M,
            vertical_bootstrap=_vertical_bootstrap_config_for_runtime_mode(
                runtime_mode,
                micro_liftoff_height_m=HOVER_MICRO_LIFTOFF_HEIGHT_M,
                takeoff_confirm_target_height_tolerance_m=HOVER_TAKEOFF_TARGET_HEIGHT_TOLERANCE_M,
            ),
            touchdown_require_supervisor_grounded=(runtime_mode != HOVER_RUNTIME_MODE_SITL),
            touchdown_range_only_confirmation_source=(
                "range_only_sitl" if runtime_mode == HOVER_RUNTIME_MODE_SITL else "range_only"
            ),
            stability=_stability_config_for_runtime_mode(runtime_mode),
        )
    )
    replayed_samples = replay_runtime.snapshot()
    telemetry_summary = summarize_hover_telemetry(
        replayed_samples,
        available_blocks=artifact.available_blocks,
        skipped_blocks=artifact.skipped_blocks,
    )
    failures = tuple(
        dict.fromkeys(
            list(_evaluate_primitive_outcome(primitive_outcome))
            + list(
                evaluate_hover_stability(
                    telemetry_summary,
                    target_height_m=target_height_m,
                    runtime_mode=runtime_mode,
                    min_clearance_m=min_clearance_m,
                )
            )
        )
    )
    replay_status = "completed" if not failures else "unstable"
    replay_outcome_class = classify_hover_outcome(
        status=replay_status,
        primitive_outcome=primitive_outcome,
        failures=failures,
    )
    reported_status = str(report_payload.get("status", "unknown"))
    reported_outcome_class = str(report_payload.get("outcome_class", "unknown"))

    artifact_payload: dict[str, object] = {
        "report_path": artifact.report_path,
        "trace_path": artifact.trace_path,
        "reported_status": reported_status,
        "reported_outcome_class": reported_outcome_class,
        "reported_failures": _coerce_report_failures(report_payload),
        "reported_trace_event_count": len(artifact.trace_events),
        "telemetry_sample_count": len(artifact.telemetry_samples),
        "replay_start_timestamp_ms": replay_start_timestamp_ms,
        "replay_telemetry_sample_count": len(replay_samples),
    }
    replay_payload: dict[str, object] = {
        "status": replay_status,
        "outcome_class": replay_outcome_class,
        "matches_report_outcome_class": replay_outcome_class == reported_outcome_class,
        "primitive_outcome": asdict(primitive_outcome),
        "failures": failures,
        "telemetry_summary": asdict(telemetry_summary),
        "replayed_phase_event_count": len(replay_trace.events),
        "replayed_command_count": len(commander.commands),
    }
    if include_phase_events:
        artifact_payload["trace_events"] = tuple(asdict(event) for event in artifact.trace_events)
        replay_payload["phase_events"] = _serialize_phase_events(replay_trace.events)
    if include_command_log:
        replay_payload["command_log"] = _serialize_commands(commander.commands)
    return {
        "artifact": artifact_payload,
        "replay": replay_payload,
    }


def _serialize_phase_events(events: Sequence[HoverReplayPhaseEvent]) -> tuple[dict[str, object], ...]:
    return tuple(asdict(event) for event in events)


def _serialize_commands(commands: Sequence[HoverReplayCommanderCommand]) -> tuple[dict[str, object], ...]:
    return tuple(asdict(command) for command in commands)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-json", type=Path, required=True, help="Path to a stored hover JSON report")
    parser.add_argument("--trace-file", type=Path, default=None, help="Optional hover worker phase trace JSONL")
    parser.add_argument("--runtime-mode", choices=(HOVER_RUNTIME_MODE_HARDWARE, HOVER_RUNTIME_MODE_SITL), default=HOVER_RUNTIME_MODE_HARDWARE, help="Replay the hover artifact under the hardware or SITL hover contract.")
    parser.add_argument("--setpoint-period-s", type=float, default=0.1, help="Replay setpoint cadence in seconds")
    parser.add_argument("--min-clearance-m", type=float, default=None, help="Optional replay-only in-flight clearance floor used by hover stability evaluation.")
    parser.add_argument("--include-phase-events", action="store_true", help="Include loaded and replayed phase events in JSON output")
    parser.add_argument("--include-command-log", action="store_true", help="Include the replayed commander command log in JSON output")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    artifact = load_hover_replay_artifact(args.report_json, trace_path=args.trace_file)
    payload = run_hover_replay(
        artifact,
        runtime_mode=str(args.runtime_mode),
        setpoint_period_s=float(args.setpoint_period_s),
        min_clearance_m=None if args.min_clearance_m is None else float(args.min_clearance_m),
        include_phase_events=bool(args.include_phase_events),
        include_command_log=bool(args.include_command_log),
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
