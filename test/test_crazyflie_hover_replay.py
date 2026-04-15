"""Regression coverage for Crazyflie hover replay and CrazySim adapter helpers."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from twinr.hardware.crazyflie_hover_replay import (
    CrazyflieTelemetryReplayRuntime,
    HoverReplayClock,
    load_hover_replay_artifact,
)
from twinr.hardware.crazyflie_sim_disturbance import (
    CrazySimDisturbancePlan,
    CrazySimDisturbancePulse,
    load_crazysim_disturbance_plan,
    write_crazysim_disturbance_plan,
)
from twinr.hardware.crazysim_adapter import (
    CrazySimAdapterError,
    CrazySimSingleAgentConfig,
    CrazySimSingleAgentLauncher,
    build_crazysim_single_agent_command,
    resolve_crazysim_checkout,
)


_REPLAY_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "hardware" / "bitcraze" / "replay_hover_trace.py"
_REPLAY_SPEC = importlib.util.spec_from_file_location("bitcraze_replay_hover_trace_script", _REPLAY_SCRIPT_PATH)
assert _REPLAY_SPEC is not None and _REPLAY_SPEC.loader is not None
_REPLAY_MODULE = importlib.util.module_from_spec(_REPLAY_SPEC)
sys.modules[_REPLAY_SPEC.name] = _REPLAY_MODULE
_REPLAY_SPEC.loader.exec_module(_REPLAY_MODULE)

_RUN_HOVER_SIM_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "hardware" / "bitcraze" / "run_hover_sim.py"
_RUN_HOVER_SIM_SPEC = importlib.util.spec_from_file_location("bitcraze_run_hover_sim_script", _RUN_HOVER_SIM_SCRIPT_PATH)
assert _RUN_HOVER_SIM_SPEC is not None and _RUN_HOVER_SIM_SPEC.loader is not None
_RUN_HOVER_SIM_MODULE = importlib.util.module_from_spec(_RUN_HOVER_SIM_SPEC)
sys.modules[_RUN_HOVER_SIM_SPEC.name] = _RUN_HOVER_SIM_MODULE
_RUN_HOVER_SIM_SPEC.loader.exec_module(_RUN_HOVER_SIM_MODULE)


def _supervisor_info(*, is_flying: bool) -> int:
    return (1 << 3) | ((1 << 4) if is_flying else 0)


def _minimal_hover_report_payload() -> dict[str, object]:
    return {
        "report": {
            "height_m": 0.20,
            "hover_duration_s": 0.20,
            "takeoff_velocity_mps": 0.20,
            "land_velocity_mps": 0.20,
            "status": "unstable",
            "outcome_class": "takeoff_failed",
            "failures": ("hover primitive did not report a successful takeoff",),
            "telemetry_summary": {
                "available_blocks": ("hover-sensors", "hover-attitude", "hover-velocity"),
                "skipped_blocks": (),
            },
            "telemetry": (
                {
                    "timestamp_ms": 1000,
                    "block_name": "hover-sensors",
                    "values": {
                        "range.zrange": 30,
                        "motion.squal": 0,
                        "supervisor.info": 0,
                        "pm.vbat": 4.1,
                    },
                },
                {
                    "timestamp_ms": 1000,
                    "block_name": "hover-attitude",
                    "values": {
                        "stabilizer.roll": 0.0,
                        "stabilizer.pitch": 0.0,
                        "stabilizer.yaw": 0.0,
                        "stateEstimate.x": 0.0,
                        "stateEstimate.y": 0.0,
                        "stateEstimate.z": 0.03,
                    },
                },
                {
                    "timestamp_ms": 1000,
                    "block_name": "hover-velocity",
                    "values": {
                        "stateEstimate.vx": 0.0,
                        "stateEstimate.vy": 0.0,
                        "stateEstimate.vz": 0.0,
                        "stabilizer.thrust": 0.0,
                    },
                },
                {
                    "timestamp_ms": 1100,
                    "block_name": "hover-sensors",
                    "values": {
                        "range.zrange": 32,
                        "motion.squal": 0,
                        "supervisor.info": 0,
                        "pm.vbat": 4.0,
                    },
                },
                {
                    "timestamp_ms": 1100,
                    "block_name": "hover-attitude",
                    "values": {
                        "stabilizer.roll": 0.2,
                        "stabilizer.pitch": -0.1,
                        "stabilizer.yaw": 0.0,
                        "stateEstimate.x": 0.0,
                        "stateEstimate.y": 0.0,
                        "stateEstimate.z": 0.03,
                    },
                },
                {
                    "timestamp_ms": 1100,
                    "block_name": "hover-velocity",
                    "values": {
                        "stateEstimate.vx": 0.0,
                        "stateEstimate.vy": 0.0,
                        "stateEstimate.vz": 0.0,
                        "stabilizer.thrust": 15000.0,
                    },
                },
            ),
        }
    }


def _takeoff_aligned_sitl_hover_report_payload(*, replay_start_timestamp_ms: int | None) -> dict[str, object]:
    telemetry: list[dict[str, object]] = []
    samples = [
        (0, 0.02, False),
        (100, 0.02, False),
        (200, 0.02, False),
        (300, 0.02, False),
        (400, 0.02, False),
        (500, 0.03, True),
        (600, 0.09, True),
        (700, 0.12, True),
        (800, 0.18, True),
        (900, 0.22, True),
        (1000, 0.24, True),
        (1100, 0.25, True),
        (1200, 0.25, True),
        (1300, 0.25, True),
        (1400, 0.25, True),
        (1500, 0.25, True),
        (1600, 0.25, True),
        (1700, 0.25, True),
        (1800, 0.25, True),
        (1900, 0.25, True),
        (2000, 0.22, True),
        (2100, 0.18, True),
        (2200, 0.14, True),
        (2300, 0.10, True),
        (2400, 0.07, True),
        (2500, 0.04, True),
        (2600, 0.02, False),
        (2700, 0.01, False),
        (2800, 0.01, False),
        (2900, 0.01, False),
    ]
    for timestamp_ms, z_m, is_flying in samples:
        telemetry.append(
            {
                "timestamp_ms": timestamp_ms,
                "block_name": "hover-attitude",
                "values": {
                    "stabilizer.roll": 0.0,
                    "stabilizer.pitch": 0.0,
                    "stabilizer.yaw": 0.0,
                    "stateEstimate.x": 0.0,
                    "stateEstimate.y": 0.0,
                    "stateEstimate.z": z_m,
                },
                "received_monotonic_s": float(timestamp_ms) / 1000.0,
            }
        )
        telemetry.append(
            {
                "timestamp_ms": timestamp_ms,
                "block_name": "hover-sensors",
                "values": {
                    "range.zrange": 0,
                    "pm.vbat": 4.1,
                    "pm.state": 0,
                    "supervisor.info": _supervisor_info(is_flying=is_flying),
                    "radio.rssi": -40.0,
                    "radio.isConnected": 1,
                },
                "received_monotonic_s": float(timestamp_ms) / 1000.0,
            }
        )
        telemetry.append(
            {
                "timestamp_ms": timestamp_ms,
                "block_name": "hover-velocity",
                "values": {
                    "stateEstimate.vx": 0.0,
                    "stateEstimate.vy": 0.0,
                    "stateEstimate.vz": 0.0,
                    "stabilizer.thrust": 18000.0 if is_flying else 0.0,
                },
                "received_monotonic_s": float(timestamp_ms) / 1000.0,
            }
        )
    report: dict[str, object] = {
        "height_m": 0.25,
        "hover_duration_s": 0.60,
        "takeoff_velocity_mps": 0.20,
        "land_velocity_mps": 0.20,
        "status": "completed",
        "outcome_class": "bounded_hover_ok",
        "failures": (),
        "telemetry_summary": {
            "available_blocks": ("hover-attitude", "hover-sensors", "hover-velocity"),
            "skipped_blocks": (),
        },
        "telemetry": telemetry,
    }
    if replay_start_timestamp_ms is not None:
        report["replay_start_timestamp_ms"] = replay_start_timestamp_ms
    return {"report": report}


def _takeoff_aligned_hardware_hover_report_payload() -> dict[str, object]:
    telemetry: list[dict[str, object]] = []
    samples = [
        (0, 0.02, 0, False),
        (100, 0.02, 0, False),
        (200, 0.03, 0, False),
        (300, 0.04, 0, False),
        (400, 0.06, 0, False),
        (500, 0.08, 60, True),
        (600, 0.09, 80, True),
        (700, 0.10, 80, True),
        (800, 0.10, 80, True),
        (900, 0.10, 80, True),
        (1000, 0.10, 80, True),
        (1100, 0.10, 80, True),
        (1200, 0.10, 80, True),
        (1300, 0.10, 80, True),
        (1400, 0.10, 80, True),
        (1500, 0.10, 80, True),
        (1600, 0.09, 80, True),
        (1700, 0.07, 80, True),
        (1800, 0.04, 60, True),
        (1900, 0.02, 0, False),
        (2000, 0.01, 0, False),
        (2100, 0.01, 0, False),
    ]
    for timestamp_ms, z_m, motion_squal, is_flying in samples:
        telemetry.append(
            {
                "timestamp_ms": timestamp_ms,
                "block_name": "hover-attitude",
                "values": {
                    "stabilizer.roll": 0.0,
                    "stabilizer.pitch": 0.0,
                    "stabilizer.yaw": 0.0,
                    "stateEstimate.x": 0.0,
                    "stateEstimate.y": 0.0,
                    "stateEstimate.z": z_m,
                },
                "received_monotonic_s": float(timestamp_ms) / 1000.0,
            }
        )
        telemetry.append(
            {
                "timestamp_ms": timestamp_ms,
                "block_name": "hover-sensors",
                "values": {
                    "range.zrange": int(z_m * 1000.0),
                    "motion.squal": motion_squal,
                    "pm.vbat": 4.1,
                    "pm.state": 0,
                    "supervisor.info": _supervisor_info(is_flying=is_flying),
                    "radio.rssi": -40.0,
                    "radio.isConnected": 1,
                },
                "received_monotonic_s": float(timestamp_ms) / 1000.0,
            }
        )
        telemetry.append(
            {
                "timestamp_ms": timestamp_ms,
                "block_name": "hover-velocity",
                "values": {
                    "stateEstimate.vx": 0.0,
                    "stateEstimate.vy": 0.0,
                    "stateEstimate.vz": 0.0,
                    "stabilizer.thrust": 18000.0 if is_flying else 0.0,
                },
                "received_monotonic_s": float(timestamp_ms) / 1000.0,
            }
        )
    return {
        "report": {
            "height_m": 0.10,
            "hover_duration_s": 0.30,
            "takeoff_velocity_mps": 0.05,
            "land_velocity_mps": 0.05,
            "status": "completed",
            "outcome_class": "bounded_hover_ok",
            "failures": (),
            "telemetry_summary": {
                "available_blocks": ("hover-sensors", "hover-attitude", "hover-velocity"),
                "skipped_blocks": (),
            },
            "telemetry": telemetry,
        }
    }


class CrazyflieHoverReplayTests(unittest.TestCase):
    def test_load_hover_replay_artifact_parses_report_and_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = root / "hover-report.json"
            report_path.write_text(json.dumps(_minimal_hover_report_payload()), encoding="utf-8")
            trace_path = root / "hover-trace.jsonl"
            trace_path.write_text(
                "\n".join(
                    (
                        json.dumps({"index": 0, "phase": "run_hover_test", "status": "begin", "elapsed_s": 0.0}),
                        json.dumps({"index": 1, "phase": "hover_primitive_takeoff", "status": "begin", "elapsed_s": 0.2}),
                    )
                ),
                encoding="utf-8",
            )

            artifact = load_hover_replay_artifact(report_path, trace_path=trace_path)

        self.assertEqual(len(artifact.telemetry_samples), 6)
        self.assertEqual(artifact.available_blocks, ("hover-sensors", "hover-attitude", "hover-velocity"))
        self.assertEqual(len(artifact.trace_events), 2)
        self.assertEqual(artifact.trace_events[1].phase, "hover_primitive_takeoff")

    def test_replay_runtime_advances_latest_values_and_ages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "hover-report.json"
            report_path.write_text(json.dumps(_minimal_hover_report_payload()), encoding="utf-8")
            artifact = load_hover_replay_artifact(report_path)

        clock = HoverReplayClock()
        runtime = CrazyflieTelemetryReplayRuntime(
            artifact.telemetry_samples,
            monotonic=clock.monotonic,
            available_blocks=artifact.available_blocks,
            skipped_blocks=artifact.skipped_blocks,
        )

        latest_zrange, age_s = runtime.latest_value("range.zrange")
        self.assertEqual(latest_zrange, 30)
        self.assertAlmostEqual(age_s or 0.0, 0.0, places=6)
        self.assertEqual(len(runtime.snapshot()), 3)

        clock.sleep(0.15)
        latest_zrange, age_s = runtime.latest_value("range.zrange")
        self.assertEqual(latest_zrange, 32)
        self.assertAlmostEqual(age_s or 0.0, 0.05, places=6)
        self.assertEqual(len(runtime.snapshot()), 6)

    def test_replay_hover_trace_cli_replays_takeoff_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "hover-report.json"
            report_path.write_text(json.dumps(_minimal_hover_report_payload()), encoding="utf-8")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = _REPLAY_MODULE.main(["--report-json", str(report_path), "--json"])
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["artifact"]["reported_outcome_class"], "takeoff_failed")
        self.assertEqual(payload["replay"]["outcome_class"], "takeoff_failed")
        self.assertTrue(payload["replay"]["matches_report_outcome_class"])
        self.assertTrue(payload["replay"]["primitive_outcome"]["aborted"])
        self.assertFalse(payload["replay"]["primitive_outcome"]["took_off"])

    def test_run_hover_replay_honors_explicit_replay_start_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            anchored_path = root / "anchored.json"
            unanchored_path = root / "unanchored.json"
            anchored_path.write_text(
                json.dumps(_takeoff_aligned_sitl_hover_report_payload(replay_start_timestamp_ms=500)),
                encoding="utf-8",
            )
            unanchored_path.write_text(
                json.dumps(_takeoff_aligned_sitl_hover_report_payload(replay_start_timestamp_ms=None)),
                encoding="utf-8",
            )
            anchored_artifact = load_hover_replay_artifact(anchored_path)
            unanchored_artifact = load_hover_replay_artifact(unanchored_path)

        anchored_payload = _REPLAY_MODULE.run_hover_replay(
            anchored_artifact,
            runtime_mode="sitl",
            setpoint_period_s=0.1,
        )
        unanchored_payload = _REPLAY_MODULE.run_hover_replay(
            unanchored_artifact,
            runtime_mode="sitl",
            setpoint_period_s=0.1,
        )

        self.assertEqual(anchored_payload["artifact"]["replay_start_timestamp_ms"], 500)
        self.assertNotIn(
            "takeoff confirmation failed",
            " ".join(str(item) for item in anchored_payload["replay"]["failures"]),
        )
        self.assertLess(
            int(anchored_payload["artifact"]["replay_telemetry_sample_count"]),
            int(anchored_payload["artifact"]["telemetry_sample_count"]),
        )
        self.assertEqual(unanchored_payload["replay"]["outcome_class"], "takeoff_failed")
        self.assertNotEqual(
            tuple(anchored_payload["replay"]["failures"]),
            tuple(unanchored_payload["replay"]["failures"]),
        )

    def test_run_hover_replay_hardware_uses_vertical_bootstrap_before_hover(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "hover-report.json"
            report_path.write_text(
                json.dumps(_takeoff_aligned_hardware_hover_report_payload()),
                encoding="utf-8",
            )
            artifact = load_hover_replay_artifact(report_path)

        payload = _REPLAY_MODULE.run_hover_replay(
            artifact,
            runtime_mode="hardware",
            setpoint_period_s=0.1,
            include_phase_events=True,
            include_command_log=True,
        )

        self.assertTrue(payload["replay"]["primitive_outcome"]["took_off"])
        self.assertTrue(payload["replay"]["primitive_outcome"]["trim_identified"])
        command_log = payload["replay"]["command_log"]
        self.assertGreaterEqual(len(command_log), 2)
        self.assertEqual(command_log[0]["kind"], "manual")
        first_hover_index = next(
            index
            for index, command in enumerate(command_log)
            if command["kind"] == "hover"
        )
        self.assertGreater(first_hover_index, 0)
        self.assertTrue(
            all(command["kind"] == "manual" for command in command_log[:first_hover_index])
        )
        manual_thrusts = [
            float(command["thrust_percentage"])
            for command in command_log[:first_hover_index]
            if command["kind"] == "manual"
        ]
        self.assertGreaterEqual(len(manual_thrusts), 1)
        self.assertGreater(max(manual_thrusts), min(manual_thrusts))
        phase_events = payload["replay"]["phase_events"]
        self.assertIn(
            ("hover_primitive_vertical_bootstrap", "done"),
            tuple((event["phase"], event["status"]) for event in phase_events),
        )

    def test_run_hover_replay_hardware_blocks_dead_sensor_trace_before_hover_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "hover-report.json"
            report_path.write_text(json.dumps(_minimal_hover_report_payload()), encoding="utf-8")
            artifact = load_hover_replay_artifact(report_path)

        payload = _REPLAY_MODULE.run_hover_replay(
            artifact,
            runtime_mode="hardware",
            setpoint_period_s=0.1,
            include_phase_events=True,
            include_command_log=True,
        )

        self.assertEqual(payload["replay"]["outcome_class"], "takeoff_failed")
        self.assertFalse(payload["replay"]["primitive_outcome"]["took_off"])
        command_log = payload["replay"]["command_log"]
        self.assertGreaterEqual(len(command_log), 1)
        self.assertIn(command_log[0]["kind"], {"manual", "stop"})
        self.assertNotIn("hover", tuple(command["kind"] for command in command_log))
        phase_events = payload["replay"]["phase_events"]
        self.assertIn(
            ("hover_primitive_vertical_bootstrap", "blocked"),
            tuple((event["phase"], event["status"]) for event in phase_events),
        )


class CrazySimAdapterTests(unittest.TestCase):
    def test_resolve_crazysim_checkout_requires_expected_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            firmware_dir = root / "crazyflie-firmware"
            gazebo_launch_script = firmware_dir / "tools" / "crazyflie-simulation" / "simulator_files" / "gazebo" / "launch" / "sitl_singleagent.sh"
            mujoco_launch_script = firmware_dir / "tools" / "crazyflie-simulation" / "simulator_files" / "mujoco" / "launch" / "sitl_singleagent.sh"
            gazebo_launch_script.parent.mkdir(parents=True, exist_ok=True)
            mujoco_launch_script.parent.mkdir(parents=True, exist_ok=True)
            gazebo_launch_script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            mujoco_launch_script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")

            checkout = resolve_crazysim_checkout(root)

        self.assertEqual(checkout.root, root.resolve())
        self.assertEqual(checkout.firmware_dir, firmware_dir.resolve())
        self.assertEqual(checkout.gazebo_launch_script, gazebo_launch_script.resolve())
        self.assertEqual(checkout.mujoco_launch_script, mujoco_launch_script.resolve())
        self.assertEqual(checkout.uri, "udp://127.0.0.1:19850")

    def test_resolve_crazysim_checkout_fails_closed_when_layout_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(CrazySimAdapterError):
                resolve_crazysim_checkout(Path(temp_dir))

    def test_build_crazysim_single_agent_command_uses_validated_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            firmware_dir = root / "crazyflie-firmware"
            gazebo_launch_script = firmware_dir / "tools" / "crazyflie-simulation" / "simulator_files" / "gazebo" / "launch" / "sitl_singleagent.sh"
            mujoco_launch_script = firmware_dir / "tools" / "crazyflie-simulation" / "simulator_files" / "mujoco" / "launch" / "sitl_singleagent.sh"
            gazebo_launch_script.parent.mkdir(parents=True, exist_ok=True)
            mujoco_launch_script.parent.mkdir(parents=True, exist_ok=True)
            gazebo_launch_script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            mujoco_launch_script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            checkout = resolve_crazysim_checkout(root)

        gazebo_command = build_crazysim_single_agent_command(
            CrazySimSingleAgentConfig(
                checkout=checkout,
                backend="gazebo",
                model="crazyflie",
                x_m=1.25,
                y_m=-0.5,
            )
        )
        mujoco_command = build_crazysim_single_agent_command(
            CrazySimSingleAgentConfig(
                checkout=checkout,
                backend="mujoco",
                model="cf2x_T350",
                x_m=0.5,
                y_m=0.25,
                phase_trace_jsonl=root / "trace.jsonl",
                disturbance_runtime_jsonl=root / "runtime.jsonl",
            )
        )

        self.assertEqual(gazebo_command[:4], ("bash", str(checkout.gazebo_launch_script), "-m", "crazyflie"))
        self.assertIn("1.250", gazebo_command)
        self.assertIn("-0.500", gazebo_command)
        self.assertEqual(mujoco_command[:2], ("python3", str(_RUN_HOVER_SIM_SCRIPT_PATH.parents[0] / "launch_crazysim_mujoco.py")))
        self.assertIn("--crazysim-root", mujoco_command)
        self.assertIn(str(checkout.root), mujoco_command)
        self.assertIn("--model-type", mujoco_command)
        self.assertIn("cf2x_T350", mujoco_command)
        self.assertIn("0.500", mujoco_command)
        self.assertIn("0.250", mujoco_command)
        self.assertIn("--wind-speed", mujoco_command)
        self.assertIn("--turbulence", mujoco_command)
        self.assertIn("--phase-trace-jsonl", mujoco_command)
        self.assertIn(str((root / "trace.jsonl").resolve()), mujoco_command)
        self.assertIn("--disturbance-runtime-jsonl", mujoco_command)
        self.assertIn(str((root / "runtime.jsonl").resolve()), mujoco_command)
        self.assertNotIn("--visualize", mujoco_command)

        visualized_mujoco_command = build_crazysim_single_agent_command(
            CrazySimSingleAgentConfig(
                checkout=checkout,
                backend="mujoco",
                model="cf2x_T350",
                visualize=True,
            )
        )
        self.assertIn("--visualize", visualized_mujoco_command)

    def test_crazysim_disturbance_plan_round_trips_via_json(self) -> None:
        plan = CrazySimDisturbancePlan(
            name="forward-impulse",
            description="Apply one bounded forward shove.",
            activation_mode="after_host_phase",
            activation_height_m=0.0,
            activation_phase="hover_primitive_hold",
            activation_status="begin",
            pulses=(
                CrazySimDisturbancePulse(
                    name="forward-impulse",
                    start_s=0.25,
                    duration_s=0.20,
                    target_agent=0,
                    world_force_n=(0.02, 0.0, 0.0),
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_crazysim_disturbance_plan(Path(temp_dir) / "disturbance.json", plan)
            loaded = load_crazysim_disturbance_plan(path)

        self.assertEqual(loaded.name, "forward-impulse")
        self.assertEqual(loaded.activation_mode, "after_host_phase")
        self.assertEqual(loaded.activation_phase, "hover_primitive_hold")
        self.assertEqual(loaded.activation_status, "begin")
        self.assertEqual(len(loaded.pulses), 1)
        self.assertEqual(loaded.pulses[0].world_force_n, (0.02, 0.0, 0.0))

    def test_crazysim_launcher_fails_closed_without_cf2_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            firmware_dir = root / "crazyflie-firmware"
            gazebo_launch_script = firmware_dir / "tools" / "crazyflie-simulation" / "simulator_files" / "gazebo" / "launch" / "sitl_singleagent.sh"
            mujoco_launch_script = firmware_dir / "tools" / "crazyflie-simulation" / "simulator_files" / "mujoco" / "launch" / "sitl_singleagent.sh"
            gazebo_launch_script.parent.mkdir(parents=True, exist_ok=True)
            mujoco_launch_script.parent.mkdir(parents=True, exist_ok=True)
            gazebo_launch_script.write_text("#!/usr/bin/env bash\nsleep 30\n", encoding="utf-8")
            mujoco_launch_script.write_text("#!/usr/bin/env bash\nsleep 30\n", encoding="utf-8")
            checkout = resolve_crazysim_checkout(root)

            launcher = CrazySimSingleAgentLauncher(CrazySimSingleAgentConfig(checkout=checkout, backend="mujoco"))

        with self.assertRaises(CrazySimAdapterError) as exc_info:
            launcher.start()

        self.assertIn("missing CrazySim SITL firmware binary", str(exc_info.exception))

    def test_crazysim_launcher_stop_terminates_child_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            firmware_dir = root / "crazyflie-firmware"
            gazebo_launch_script = firmware_dir / "tools" / "crazyflie-simulation" / "simulator_files" / "gazebo" / "launch" / "sitl_singleagent.sh"
            mujoco_launch_script = firmware_dir / "tools" / "crazyflie-simulation" / "simulator_files" / "mujoco" / "launch" / "sitl_singleagent.sh"
            gazebo_launch_script.parent.mkdir(parents=True, exist_ok=True)
            mujoco_launch_script.parent.mkdir(parents=True, exist_ok=True)
            gazebo_launch_script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            mujoco_launch_script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            checkout = resolve_crazysim_checkout(root)
            pidfile = root / "child.pid"
            launcher = CrazySimSingleAgentLauncher(
                CrazySimSingleAgentConfig(
                    checkout=checkout,
                    backend="mujoco",
                    startup_settle_s=0.1,
                )
            )
            child_code = (
                "from pathlib import Path; import os, time; "
                f"Path({str(pidfile)!r}).write_text(str(os.getpid()), encoding='utf-8'); "
                "time.sleep(30)"
            )
            child_command = (
                "bash",
                "-lc",
                f"python3 -c {shlex.quote(child_code)} & wait",
            )

            with (
                mock.patch.object(CrazySimSingleAgentLauncher, "_validate_runtime_prerequisites"),
                mock.patch.object(CrazySimSingleAgentLauncher, "_cf2_running", return_value=True),
                mock.patch(
                    "twinr.hardware.crazysim_adapter.build_crazysim_single_agent_command",
                    return_value=child_command,
                ),
            ):
                launcher.start()
                deadline = time.monotonic() + 2.0
                while not pidfile.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(pidfile.exists(), "child process never wrote its pidfile")
                child_pid = int(pidfile.read_text(encoding="utf-8").strip())
                self.assertTrue(Path(f"/proc/{child_pid}").exists())
                launcher.stop()
                deadline = time.monotonic() + 2.0
                while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertFalse(
                    Path(f"/proc/{child_pid}").exists(),
                    "launcher.stop() left the spawned child process alive",
                )


class CrazySimHoverRunnerTests(unittest.TestCase):
    def test_probe_udp_uri_ready_uses_official_cflib_scan_contract(self) -> None:
        port_box: dict[str, int] = {}
        ready = threading.Event()
        received = threading.Event()

        def _server() -> None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("127.0.0.1", 0))
            port_box["port"] = int(sock.getsockname()[1])
            ready.set()
            packet, address = sock.recvfrom(1024)
            self.assertEqual(packet, b"\xFF")
            sock.sendto(b"\xF3", address)
            received.set()
            sock.close()

        thread = threading.Thread(target=_server, daemon=True)
        thread.start()
        self.assertTrue(ready.wait(timeout=1.0))

        uri = f"udp://127.0.0.1:{port_box['port']}"
        self.assertTrue(
            _RUN_HOVER_SIM_MODULE._probe_udp_uri_ready(uri, probe_timeout_s=0.2)
        )
        self.assertTrue(received.wait(timeout=1.0))

    def test_wait_for_udp_uri_ready_retries_until_probe_succeeds(self) -> None:
        with mock.patch.object(
            _RUN_HOVER_SIM_MODULE,
            "_probe_udp_uri_ready",
            side_effect=(False, False, True),
        ):
            attempts, elapsed_s = _RUN_HOVER_SIM_MODULE._wait_for_udp_uri_ready(
                "udp://127.0.0.1:19850",
                readiness_timeout_s=1.0,
                probe_timeout_s=0.01,
                probe_period_s=0.01,
            )

        self.assertEqual(attempts, 3)
        self.assertGreaterEqual(elapsed_s, 0.0)

    def test_build_cflib_ready_probe_command_uses_selected_python_and_workspace(self) -> None:
        command = _RUN_HOVER_SIM_MODULE._build_cflib_ready_probe_command(
            python_bin=Path("/tmp/fake-python"),
            uri="udp://127.0.0.1:19850",
            workspace=Path("/tmp/crazysim-workspace"),
        )

        self.assertEqual(command[0], "/tmp/fake-python")
        self.assertEqual(command[1], "-c")
        self.assertIn("SyncCrazyflie", command[2])
        self.assertEqual(command[3], "udp://127.0.0.1:19850")
        self.assertEqual(command[4], "/tmp/crazysim-workspace")

    def test_wait_for_cflib_uri_ready_retries_until_probe_connects(self) -> None:
        completed_timeout = subprocess.TimeoutExpired(cmd=("python",), timeout=0.1)
        completed_failure = subprocess.CompletedProcess(
            args=("python",),
            returncode=2,
            stdout="",
            stderr="connect failed",
        )
        completed_success = subprocess.CompletedProcess(
            args=("python",),
            returncode=0,
            stdout=json.dumps({"connected": True, "cache_dir": "/tmp/cache"}),
            stderr="",
        )
        with mock.patch.object(
            _RUN_HOVER_SIM_MODULE.subprocess,
            "run",
            side_effect=(completed_timeout, completed_failure, completed_success),
        ):
            attempts, elapsed_s, payload = _RUN_HOVER_SIM_MODULE._wait_for_cflib_uri_ready(
                python_bin=Path(sys.executable),
                uri="udp://127.0.0.1:19850",
                workspace=Path("/tmp/crazysim-workspace"),
                readiness_timeout_s=1.0,
                probe_timeout_s=0.1,
                probe_period_s=0.01,
            )

        self.assertEqual(attempts, 3)
        self.assertGreaterEqual(elapsed_s, 0.0)
        self.assertTrue(payload["connected"])
        self.assertEqual(payload["cache_dir"], "/tmp/cache")

    def test_normalize_hover_args_rejects_conflicting_worker_overrides(self) -> None:
        with self.assertRaises(ValueError):
            _RUN_HOVER_SIM_MODULE._normalize_hover_args(("--", "--uri", "udp://127.0.0.1:19850"))
        with self.assertRaises(ValueError):
            _RUN_HOVER_SIM_MODULE._normalize_hover_args(("--", "--runtime-mode", "hardware"))

    def test_build_hover_worker_command_injects_sim_contract(self) -> None:
        command = _RUN_HOVER_SIM_MODULE._build_hover_worker_command(
            python_bin=Path("/tmp/fake-python"),
            uri="udp://127.0.0.1:19850",
            workspace=Path("/tmp/crazysim-workspace"),
            trace_file=Path("/tmp/crazysim-trace.jsonl"),
            hover_args=("--height-m", "0.25", "--hover-duration-s", "1.0"),
        )

        self.assertEqual(command[0], "/tmp/fake-python")
        self.assertIn("--uri", command)
        self.assertIn("udp://127.0.0.1:19850", command)
        self.assertIn("--on-device-failsafe-mode", command)
        self.assertIn("off", command)
        self.assertIn("--runtime-mode", command)
        self.assertIn("sitl", command)
        self.assertIn("--json", command)
        self.assertIn("--trace-file", command)
        self.assertIn("/tmp/crazysim-trace.jsonl", command)

    def test_run_hover_sim_parser_accepts_disturbance_and_wind_args(self) -> None:
        parser = _RUN_HOVER_SIM_MODULE._build_parser()
        args = parser.parse_args(
            [
                "--crazysim-root",
                "/tmp/crazysim",
                "--disturbance-spec-json",
                "/tmp/plan.json",
                "--wind-speed-mps",
                "1.2",
                "--gust-intensity-mps",
                "0.4",
                "--turbulence",
                "moderate",
                "--mass-kg",
                "0.040",
                "--json",
            ]
        )

        self.assertEqual(args.disturbance_spec_json, Path("/tmp/plan.json"))
        self.assertAlmostEqual(args.wind_speed_mps, 1.2)
        self.assertAlmostEqual(args.gust_intensity_mps, 0.4)
        self.assertEqual(args.turbulence, "moderate")
        self.assertAlmostEqual(args.mass_kg, 0.040)


if __name__ == "__main__":
    unittest.main()
