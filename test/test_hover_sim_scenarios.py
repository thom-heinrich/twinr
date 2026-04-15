"""Regression coverage for deterministic CrazySim hover scenarios."""

# pylint: disable=import-error

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from twinr.hardware.crazyflie_hover_replay import load_hover_replay_artifact
from twinr.hardware.crazyflie_hover_recovery import HoverRecoveryWindow
from typing import Any


_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "hardware" / "bitcraze"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from hover_sim_scenarios import (  # noqa: E402
    HoverRecoveryExpectation,
    HoverSimScenarioExpectation,
    apply_hover_sim_scenario,
    evaluate_hover_replay_against_expectation,
    run_hover_sim_scenario,
)  # pylint: disable=import-error
from replay_hover_trace import run_hover_replay  # noqa: E402  # pylint: disable=import-error
from run_hover_test import HOVER_RUNTIME_MODE_HARDWARE, HOVER_RUNTIME_MODE_SITL  # noqa: E402  # pylint: disable=import-error


_RUN_HOVER_SIM_SCENARIOS_SCRIPT_PATH = _SCRIPT_DIR / "run_hover_sim_scenarios.py"
_RUN_HOVER_SIM_SCENARIOS_SPEC = importlib.util.spec_from_file_location(
    "bitcraze_run_hover_sim_scenarios_script",
    _RUN_HOVER_SIM_SCENARIOS_SCRIPT_PATH,
)
assert _RUN_HOVER_SIM_SCENARIOS_SPEC is not None and _RUN_HOVER_SIM_SCENARIOS_SPEC.loader is not None
_RUN_HOVER_SIM_SCENARIOS_MODULE = importlib.util.module_from_spec(_RUN_HOVER_SIM_SCENARIOS_SPEC)
sys.modules[_RUN_HOVER_SIM_SCENARIOS_SPEC.name] = _RUN_HOVER_SIM_SCENARIOS_MODULE
_RUN_HOVER_SIM_SCENARIOS_SPEC.loader.exec_module(_RUN_HOVER_SIM_SCENARIOS_MODULE)

_RUN_HOVER_ACCEPTANCE_GATE_SCRIPT_PATH = _SCRIPT_DIR / "run_hover_acceptance_gate.py"
_RUN_HOVER_ACCEPTANCE_GATE_SPEC = importlib.util.spec_from_file_location(
    "bitcraze_run_hover_acceptance_gate_script",
    _RUN_HOVER_ACCEPTANCE_GATE_SCRIPT_PATH,
)
assert _RUN_HOVER_ACCEPTANCE_GATE_SPEC is not None and _RUN_HOVER_ACCEPTANCE_GATE_SPEC.loader is not None
_RUN_HOVER_ACCEPTANCE_GATE_MODULE = importlib.util.module_from_spec(_RUN_HOVER_ACCEPTANCE_GATE_SPEC)
sys.modules[_RUN_HOVER_ACCEPTANCE_GATE_SPEC.name] = _RUN_HOVER_ACCEPTANCE_GATE_MODULE
_RUN_HOVER_ACCEPTANCE_GATE_SPEC.loader.exec_module(_RUN_HOVER_ACCEPTANCE_GATE_MODULE)


def _supervisor_info(*, is_flying: bool) -> int:
    return (1 << 3) | ((1 << 4) if is_flying else 0)


def _nominal_hover_report_payload() -> dict[str, Any]:
    telemetry: list[dict[str, object]] = []
    samples = [
        (0, 0.00, False),
        (100, 0.04, False),
        (200, 0.08, True),
        (300, 0.09, True),
        (400, 0.10, True),
    ]
    samples.extend((timestamp_ms, 0.10, True) for timestamp_ms in range(500, 3300, 100))
    samples.extend(
        (
            (3300, 0.09, True),
            (3400, 0.08, True),
            (3500, 0.06, True),
            (3600, 0.04, True),
            (3700, 0.03, True),
            (3800, 0.02, False),
            (3900, 0.01, False),
            (4000, 0.01, False),
            (4100, 0.01, False),
            (4200, 0.01, False),
        )
    )
    samples.extend((timestamp_ms, 0.01, False) for timestamp_ms in range(4300, 6200, 100))
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
            }
        )
        telemetry.append(
            {
                "timestamp_ms": timestamp_ms,
                "block_name": "hover-sensors",
                "values": {
                    "range.zrange": int(z_m * 1000.0),
                    "pm.vbat": 4.1,
                    "pm.state": 0,
                    "supervisor.info": _supervisor_info(is_flying=is_flying),
                    "radio.rssi": -40.0,
                    "radio.isConnected": 1,
                    "motion.squal": 80 if is_flying else 0,
                },
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
            }
        )
        telemetry.append(
            {
                "timestamp_ms": timestamp_ms,
                "block_name": "hover-gyro",
                "values": {"gyro.x": 0.0, "gyro.y": 0.0, "gyro.z": 0.0},
            }
        )
        telemetry.append(
            {
                "timestamp_ms": timestamp_ms,
                "block_name": "hover-clearance",
                "values": {
                    "range.front": 700,
                    "range.back": 700,
                    "range.left": 700,
                    "range.right": 700,
                    "range.up": 900,
                },
            }
        )
    return {
        "report": {
            "height_m": 0.10,
            "hover_duration_s": 0.60,
            "takeoff_velocity_mps": 0.50,
            "land_velocity_mps": 0.50,
            "status": "completed",
            "outcome_class": "bounded_hover_ok",
            "failures": (),
            "telemetry_summary": {
                "available_blocks": (
                    "hover-attitude",
                    "hover-sensors",
                    "hover-velocity",
                    "hover-gyro",
                    "hover-clearance",
                ),
                "skipped_blocks": (),
            },
            "telemetry": telemetry,
        }
    }


def _physical_suite_payload(*, matched: bool) -> dict[str, Any]:
    return {
        "output_dir": "/tmp/physical-suite",
        "scenario_names": (
            "physical_forward_impulse_recovery",
            "physical_persistent_forward_abort",
        ),
        "results": (
            {
                "scenario": "physical_forward_impulse_recovery",
                "matches_expectation": matched,
                "actual_outcome_class": "bounded_hover_ok",
            },
        ),
        "all_matched_expectations": matched,
    }


class HoverSimScenarioTests(unittest.TestCase):
    def test_fresh_baseline_contract_summary_unwraps_run_hover_sim_payload(self) -> None:
        report_payload = dict(_nominal_hover_report_payload()["report"])
        report_payload["completed"] = True
        report_payload["landed"] = True
        report_payload["primitive_outcome"] = {
            "took_off": True,
            "stable_hover_established": True,
            "landed": True,
            "forced_motor_cutoff": False,
            "touchdown_confirmation_source": "range_only_sitl",
        }
        wrapped_report = {
            "failures": [],
            "report": report_payload,
        }

        nominal_passed, touchdown_passed, failures = (
            _RUN_HOVER_ACCEPTANCE_GATE_MODULE._fresh_baseline_contract_summary(
                wrapped_report
            )
        )

        self.assertTrue(nominal_passed)
        self.assertTrue(touchdown_passed)
        self.assertEqual(failures, ())

    def test_run_hover_replay_uses_sitl_ground_distance_contract(self) -> None:
        payload = _nominal_hover_report_payload()
        for sample in payload["report"]["telemetry"]:
            if sample["block_name"] == "hover-sensors":
                sample["values"]["range.zrange"] = 0

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "hover-report.json"
            report_path.write_text(json.dumps(payload), encoding="utf-8")
            artifact = load_hover_replay_artifact(report_path)

        sitl_payload = run_hover_replay(
            artifact,
            runtime_mode=HOVER_RUNTIME_MODE_SITL,
            setpoint_period_s=0.1,
        )
        hardware_payload = run_hover_replay(
            artifact,
            runtime_mode=HOVER_RUNTIME_MODE_HARDWARE,
            setpoint_period_s=0.1,
        )

        self.assertEqual(sitl_payload["replay"]["outcome_class"], "bounded_hover_ok")
        self.assertEqual(hardware_payload["replay"]["outcome_class"], "takeoff_failed")

    def test_run_hover_sim_scenarios_preserves_python_bin_symlink_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_venv_python = root / "fake-venv" / "bin" / "python"
            fake_venv_python.parent.mkdir(parents=True, exist_ok=True)
            fake_venv_python.symlink_to(Path(sys.executable))

            normalized = _RUN_HOVER_SIM_SCENARIOS_MODULE._normalize_nonresolving_path(fake_venv_python)

        self.assertEqual(normalized, fake_venv_python)

    def test_write_baseline_hover_report_json_unwraps_run_hover_sim_wrapper(self) -> None:
        wrapped_payload = {
            "failures": [],
            "report": _nominal_hover_report_payload()["report"],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            report_path = _RUN_HOVER_SIM_SCENARIOS_MODULE._write_baseline_hover_report_json(
                wrapped_payload,
                output_dir=output_dir,
            )
            artifact = load_hover_replay_artifact(report_path)

        self.assertTrue(artifact.telemetry_samples)
        self.assertEqual(artifact.report_payload["outcome_class"], "bounded_hover_ok")

    def test_all_supported_scenarios_match_expected_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "hover-report.json"
            report_path.write_text(json.dumps(_nominal_hover_report_payload()), encoding="utf-8")
            artifact = load_hover_replay_artifact(report_path)

        scenario_names = (
            "baseline_nominal",
            "transient_forward_drift_recovery",
            "transient_left_drift_recovery",
            "persistent_forward_drift_abort",
            "zrange_outlier",
            "attitude_spike",
            "wall_proximity",
        )
        for scenario_name in scenario_names:
            result = run_hover_sim_scenario(artifact, scenario_name=scenario_name)
            self.assertTrue(result.matches_expectation, msg=f"{scenario_name}: {result.failure_tuple}")

    def test_wall_proximity_scenario_writes_low_clearance_into_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "hover-report.json"
            report_path.write_text(json.dumps(_nominal_hover_report_payload()), encoding="utf-8")
            artifact = load_hover_replay_artifact(report_path)

        mutated = apply_hover_sim_scenario(artifact, scenario_name="wall_proximity")
        front_values = [
            sample.values["range.front"]
            for sample in mutated.telemetry_samples
            if sample.block_name == "hover-clearance"
        ]
        self.assertIn(120, front_values)

    def test_flow_dropout_scenario_synthesizes_zrange_for_sitl_baseline(self) -> None:
        payload = _nominal_hover_report_payload()
        for sample in payload["report"]["telemetry"]:
            if sample["block_name"] == "hover-sensors":
                sample["values"]["range.zrange"] = 0

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "hover-report.json"
            report_path.write_text(json.dumps(payload), encoding="utf-8")
            artifact = load_hover_replay_artifact(report_path)

        run_hover_sim_scenario(artifact, scenario_name="flow_dropout")
        hover_sensor_zranges = [
            sample.values.get("range.zrange")
            for sample in apply_hover_sim_scenario(artifact, scenario_name="flow_dropout").telemetry_samples
            if sample.block_name == "hover-sensors"
        ]
        self.assertTrue(any(value not in {None, 0} for value in hover_sensor_zranges))

    def test_attitude_spike_scenario_trips_trim_identify_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "hover-report.json"
            report_path.write_text(json.dumps(_nominal_hover_report_payload()), encoding="utf-8")
            artifact = load_hover_replay_artifact(report_path)

        result = run_hover_sim_scenario(artifact, scenario_name="attitude_spike")

        self.assertTrue(result.matches_expectation, msg=str(result.failure_tuple))
        self.assertEqual(result.actual_outcome_class, "takeoff_failed")
        self.assertEqual(result.contract_failures, ())
        mutated_rolls = [
            float(sample.values["stabilizer.roll"])
            for sample in apply_hover_sim_scenario(artifact, scenario_name="attitude_spike").telemetry_samples
            if sample.block_name == "hover-attitude"
        ]
        self.assertTrue(any(abs(value) >= 12.5 for value in mutated_rolls))

    def test_transient_forward_drift_recovery_commands_opposing_correction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "hover-report.json"
            report_path.write_text(json.dumps(_nominal_hover_report_payload()), encoding="utf-8")
            artifact = load_hover_replay_artifact(report_path)

        result = run_hover_sim_scenario(artifact, scenario_name="transient_forward_drift_recovery")

        self.assertTrue(result.matches_expectation, msg=str(result.recovery_failures or result.failure_tuple))
        self.assertEqual(result.actual_outcome_class, "bounded_hover_ok")
        self.assertEqual(result.contract_failures, ())
        self.assertEqual(result.guard_failures, ())
        self.assertEqual(result.recovery_failures, ())
        assert result.guard_metrics is not None
        assert result.recovery_metrics is not None
        self.assertEqual(result.guard_metrics.blocked_count, 0)
        self.assertLess(result.recovery_metrics.min_forward_mps, -0.10)
        self.assertTrue(result.recovery_metrics.recovered_within_window)

    def test_persistent_forward_drift_abort_blocks_on_speed_and_xy_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "hover-report.json"
            report_path.write_text(json.dumps(_nominal_hover_report_payload()), encoding="utf-8")
            artifact = load_hover_replay_artifact(report_path)

        result = run_hover_sim_scenario(artifact, scenario_name="persistent_forward_drift_abort")

        self.assertTrue(result.matches_expectation, msg=str(result.guard_failures or result.failure_tuple))
        self.assertEqual(result.actual_outcome_class, "unstable_hover_aborted")
        self.assertEqual(result.contract_failures, ())
        self.assertEqual(result.guard_failures, ())
        assert result.guard_metrics is not None
        self.assertGreaterEqual(result.guard_metrics.blocked_count, 1)
        self.assertIn("speed", result.guard_metrics.blocked_codes)

    def test_transient_height_drop_recovery_raises_height_command_temporarily(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "hover-report.json"
            report_path.write_text(json.dumps(_nominal_hover_report_payload()), encoding="utf-8")
            artifact = load_hover_replay_artifact(report_path)

        result = run_hover_sim_scenario(artifact, scenario_name="transient_height_drop_recovery")

        self.assertTrue(result.matches_expectation, msg=str(result.recovery_failures or result.failure_tuple))
        self.assertEqual(result.actual_outcome_class, "bounded_hover_ok")
        self.assertEqual(result.contract_failures, ())
        self.assertEqual(result.guard_failures, ())
        self.assertEqual(result.recovery_failures, ())
        command_log_raw = result.replay_payload["replay"]["command_log"]
        hover_heights = [
            float(command["height_m"])
            for command in command_log_raw
            if command.get("kind") == "hover" and command.get("height_m") is not None
        ]
        self.assertTrue(hover_heights)
        self.assertGreater(max(hover_heights), 0.14)

    def test_recovery_contract_fails_closed_when_takeoff_done_is_missing(self) -> None:
        replay_payload = {
            "replay": {
                "outcome_class": "takeoff_failed",
                "failures": (
                    "hover trim identification failed: trim observer has not yet converged",
                ),
                "primitive_outcome": {
                    "aborted": True,
                    "stable_hover_established": False,
                    "touchdown_confirmation_source": "range_only_sitl",
                },
                "phase_events": (
                    {
                        "phase": "hover_primitive_micro_liftoff",
                        "status": "done",
                        "elapsed_s": 0.4,
                    },
                    {
                        "phase": "hover_primitive_abort",
                        "status": "begin",
                        "elapsed_s": 1.2,
                    },
                ),
                "command_log": (
                    {
                        "kind": "hover",
                        "elapsed_s": 0.5,
                        "vx_mps": 0.0,
                        "vy_mps": 0.0,
                        "height_m": 0.10,
                    },
                ),
            }
        }

        evaluation = evaluate_hover_replay_against_expectation(
            replay_payload=replay_payload,
            expectation=HoverSimScenarioExpectation(
                outcome_class="bounded_hover_ok",
                recovery=HoverRecoveryExpectation(
                    disturbance_start_progress=0.15,
                    disturbance_end_progress=0.45,
                    max_recovery_delay_s=0.50,
                    forward_direction="negative",
                    min_forward_abs_mps=0.05,
                ),
            ),
            target_height_m=0.25,
        )

        self.assertIn(
            "hover recovery evaluation requires `hover_primitive_takeoff` `done` in the phase log",
            evaluation.recovery_failures,
        )

    def test_run_hover_sim_scenarios_cli_works_with_stored_baseline_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = root / "hover-report.json"
            report_path.write_text(json.dumps(_nominal_hover_report_payload()), encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = _RUN_HOVER_SIM_SCENARIOS_MODULE.main(
                    [
                        "--baseline-report-json",
                        str(report_path),
                        "--scenario",
                        "baseline_nominal",
                        "--scenario",
                        "attitude_spike",
                        "--scenario",
                        "transient_forward_drift_recovery",
                        "--scenario",
                        "persistent_forward_drift_abort",
                        "--output-dir",
                        str(root / "out"),
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["all_matched_expectations"])
        self.assertEqual(payload["results"][0]["actual_outcome_class"], "bounded_hover_ok")
        self.assertEqual(payload["results"][1]["actual_outcome_class"], "takeoff_failed")
        self.assertEqual(payload["results"][2]["actual_outcome_class"], "bounded_hover_ok")
        self.assertEqual(payload["results"][2]["recovery_failures"], [])
        self.assertEqual(payload["results"][3]["actual_outcome_class"], "unstable_hover_aborted")
        self.assertEqual(payload["results"][3]["guard_failures"], [])

    def test_evaluate_hover_replay_against_expectation_forwards_explicit_recovery_window(self) -> None:
        replay_payload = {
            "replay": {
                "outcome_class": "bounded_hover_ok",
                "failures": (),
                "primitive_outcome": {
                    "aborted": False,
                    "stable_hover_established": True,
                    "touchdown_confirmation_source": "range_only_sitl",
                },
                "phase_events": (
                    {"phase": "hover_primitive_takeoff", "status": "done", "elapsed_s": 2.4},
                    {"phase": "hover_primitive_hold", "status": "done", "elapsed_s": 3.6},
                ),
                "command_log": (
                    {"kind": "hover", "elapsed_s": 2.8, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.25},
                    {"kind": "hover", "elapsed_s": 3.0, "vx_mps": 0.0, "vy_mps": 0.10, "height_m": 0.25},
                    {"kind": "hover", "elapsed_s": 3.2, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.25},
                ),
            }
        }

        evaluation = evaluate_hover_replay_against_expectation(
            replay_payload=replay_payload,
            expectation=HoverSimScenarioExpectation(
                outcome_class="bounded_hover_ok",
                recovery=HoverRecoveryExpectation(
                    disturbance_start_progress=0.15,
                    disturbance_end_progress=0.45,
                    max_recovery_delay_s=0.50,
                    left_direction="positive",
                    min_left_abs_mps=0.05,
                    settle_required_commands=1,
                ),
            ),
            target_height_m=0.25,
            recovery_window=HoverRecoveryWindow(
                disturbance_start_elapsed_s=2.8,
                disturbance_end_elapsed_s=3.15,
                source="physical_runtime",
            ),
        )

        assert evaluation.recovery_metrics is not None
        self.assertEqual(evaluation.recovery_failures, ())
        self.assertEqual(evaluation.recovery_metrics.window_source, "physical_runtime")

    def test_run_hover_acceptance_gate_cli_combines_replay_and_scenario_suite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = root / "hover-report.json"
            physical_suite_json = root / "physical-suite.json"
            report_path.write_text(json.dumps(_nominal_hover_report_payload()), encoding="utf-8")
            physical_suite_json.write_text(
                json.dumps(_physical_suite_payload(matched=True)),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = _RUN_HOVER_ACCEPTANCE_GATE_MODULE.main(
                    [
                        "--replay-case",
                        f"{report_path}||sitl|bounded_hover_ok",
                        "--baseline-report-json",
                        str(report_path),
                        "--physical-suite-json",
                        str(physical_suite_json),
                        "--scenario",
                        "baseline_nominal",
                        "--scenario",
                        "transient_forward_drift_recovery",
                        "--output-dir",
                        str(root / "out"),
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["gate_passed"])
        self.assertFalse(payload["live_flight_eligible"])
        self.assertTrue(payload["all_replays_matched"])
        self.assertTrue(payload["all_physical_matched"])
        self.assertIsNone(payload["fresh_baseline_repeatability_passed"])
        self.assertIsNone(payload["fresh_baseline_touchdown_passed"])
        self.assertTrue(payload["scenario_suite"]["all_matched_expectations"])
        self.assertTrue(payload["physical_suite"]["all_matched_expectations"])
        self.assertEqual(payload["replay_results"][0]["actual_outcome_class"], "bounded_hover_ok")

    def test_run_hover_acceptance_gate_fails_closed_when_replay_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = root / "hover-report.json"
            physical_suite_json = root / "physical-suite.json"
            report_path.write_text(json.dumps(_nominal_hover_report_payload()), encoding="utf-8")
            physical_suite_json.write_text(
                json.dumps(_physical_suite_payload(matched=True)),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = _RUN_HOVER_ACCEPTANCE_GATE_MODULE.main(
                    [
                        "--replay-case",
                        f"{report_path}||sitl|takeoff_failed",
                        "--baseline-report-json",
                        str(report_path),
                        "--physical-suite-json",
                        str(physical_suite_json),
                        "--scenario",
                        "baseline_nominal",
                        "--output-dir",
                        str(root / "out"),
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 1)
        self.assertFalse(payload["gate_passed"])
        self.assertFalse(payload["all_replays_matched"])
        self.assertTrue(payload["scenario_suite"]["all_matched_expectations"])

    def test_run_hover_acceptance_gate_fails_closed_when_physical_suite_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = root / "hover-report.json"
            physical_suite_json = root / "physical-suite.json"
            report_path.write_text(json.dumps(_nominal_hover_report_payload()), encoding="utf-8")
            physical_suite_json.write_text(
                json.dumps(_physical_suite_payload(matched=False)),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = _RUN_HOVER_ACCEPTANCE_GATE_MODULE.main(
                    [
                        "--replay-case",
                        f"{report_path}||sitl|bounded_hover_ok",
                        "--baseline-report-json",
                        str(report_path),
                        "--physical-suite-json",
                        str(physical_suite_json),
                        "--scenario",
                        "baseline_nominal",
                        "--output-dir",
                        str(root / "out"),
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 1)
        self.assertFalse(payload["gate_passed"])
        self.assertTrue(payload["all_replays_matched"])
        self.assertTrue(payload["scenario_suite"]["all_matched_expectations"])
        self.assertFalse(payload["all_physical_matched"])
        self.assertFalse(payload["physical_suite"]["all_matched_expectations"])

    def test_run_hover_acceptance_gate_skips_fresh_suites_when_repeatability_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = root / "hover-report.json"
            report_path.write_text(json.dumps(_nominal_hover_report_payload()), encoding="utf-8")
            stdout = io.StringIO()
            fresh_payload = {
                "run_count": 3,
                "output_dir": str(root / "fresh"),
                "runs": (
                    {
                        "index": 1,
                        "matches_nominal_contract": False,
                        "touchdown_contract_ok": False,
                        "report_json": None,
                        "trace_file": None,
                        "contract_failures": ("sync_connect_timeout",),
                    },
                ),
                "repeatability_passed": False,
                "touchdown_passed": False,
                "selected_baseline_report_json": None,
                "selected_baseline_trace_file": None,
            }
            with (
                mock.patch.object(
                    _RUN_HOVER_ACCEPTANCE_GATE_MODULE,
                    "_run_fresh_baseline_repeatability",
                    return_value=fresh_payload,
                ),
                mock.patch.object(
                    _RUN_HOVER_ACCEPTANCE_GATE_MODULE,
                    "_run_scenario_suite",
                ) as run_scenario_suite,
                mock.patch.object(
                    _RUN_HOVER_ACCEPTANCE_GATE_MODULE,
                    "_run_physical_suite",
                ) as run_physical_suite,
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = _RUN_HOVER_ACCEPTANCE_GATE_MODULE.main(
                    [
                        "--replay-case",
                        f"{report_path}||sitl|bounded_hover_ok",
                        "--crazysim-root",
                        "/tmp/crazysim",
                        "--output-dir",
                        str(root / "out"),
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 1)
        self.assertFalse(payload["gate_passed"])
        self.assertFalse(payload["live_flight_eligible"])
        self.assertFalse(payload["fresh_baseline_repeatability_passed"])
        self.assertFalse(payload["fresh_baseline_touchdown_passed"])
        self.assertTrue(payload["scenario_suite"]["skipped"])
        self.assertTrue(payload["physical_suite"]["skipped"])
        run_scenario_suite.assert_not_called()
        run_physical_suite.assert_not_called()

    def test_run_hover_acceptance_gate_reuses_selected_fresh_baseline_for_scenarios(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = root / "selected-hover-report.json"
            trace_path = root / "selected-hover-trace.jsonl"
            report_path.write_text(json.dumps(_nominal_hover_report_payload()), encoding="utf-8")
            trace_path.write_text("", encoding="utf-8")
            stdout = io.StringIO()
            fresh_payload = {
                "run_count": 3,
                "output_dir": str(root / "fresh"),
                "runs": (),
                "repeatability_passed": True,
                "touchdown_passed": True,
                "selected_baseline_report_json": str(report_path),
                "selected_baseline_trace_file": str(trace_path),
            }
            scenario_payload = {"all_matched_expectations": True}
            physical_payload = _physical_suite_payload(matched=True)
            with (
                mock.patch.object(
                    _RUN_HOVER_ACCEPTANCE_GATE_MODULE,
                    "_run_fresh_baseline_repeatability",
                    return_value=fresh_payload,
                ),
                mock.patch.object(
                    _RUN_HOVER_ACCEPTANCE_GATE_MODULE,
                    "_run_scenario_suite",
                    return_value=scenario_payload,
                ) as run_scenario_suite,
                mock.patch.object(
                    _RUN_HOVER_ACCEPTANCE_GATE_MODULE,
                    "_run_physical_suite",
                    return_value=physical_payload,
                ),
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = _RUN_HOVER_ACCEPTANCE_GATE_MODULE.main(
                    [
                        "--replay-case",
                        "@selected_baseline||sitl|bounded_hover_ok",
                        "--crazysim-root",
                        "/tmp/crazysim",
                        "--output-dir",
                        str(root / "out"),
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["gate_passed"])
        self.assertTrue(payload["live_flight_eligible"])
        self.assertTrue(payload["fresh_baseline_repeatability_passed"])
        self.assertTrue(payload["fresh_baseline_touchdown_passed"])
        run_scenario_suite.assert_called_once()
        self.assertEqual(
            run_scenario_suite.call_args.kwargs["baseline_report_json_override"],
            report_path,
        )
        self.assertEqual(
            run_scenario_suite.call_args.kwargs["baseline_trace_file_override"],
            trace_path,
        )
        self.assertTrue(payload["all_replays_matched"])
        self.assertEqual(
            payload["replay_results"][0]["case"]["report_json"],
            str(report_path),
        )

    def test_run_hover_acceptance_gate_fails_closed_when_selected_baseline_replay_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stdout = io.StringIO()
            fresh_payload = {
                "run_count": 3,
                "output_dir": str(root / "fresh"),
                "runs": (),
                "repeatability_passed": False,
                "touchdown_passed": False,
                "selected_baseline_report_json": None,
                "selected_baseline_trace_file": None,
            }
            with (
                mock.patch.object(
                    _RUN_HOVER_ACCEPTANCE_GATE_MODULE,
                    "_run_fresh_baseline_repeatability",
                    return_value=fresh_payload,
                ),
                mock.patch.object(
                    _RUN_HOVER_ACCEPTANCE_GATE_MODULE,
                    "_run_scenario_suite",
                ) as run_scenario_suite,
                mock.patch.object(
                    _RUN_HOVER_ACCEPTANCE_GATE_MODULE,
                    "_run_physical_suite",
                ) as run_physical_suite,
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = _RUN_HOVER_ACCEPTANCE_GATE_MODULE.main(
                    [
                        "--replay-case",
                        "@selected_baseline||sitl|bounded_hover_ok",
                        "--crazysim-root",
                        "/tmp/crazysim",
                        "--output-dir",
                        str(root / "out"),
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 1)
        self.assertFalse(payload["gate_passed"])
        self.assertFalse(payload["all_replays_matched"])
        self.assertEqual(
            payload["replay_results"][0]["failure"],
            "selected baseline was unavailable for the requested replay case",
        )
        run_scenario_suite.assert_not_called()
        run_physical_suite.assert_not_called()
