"""Regression coverage for real CrazySim disturbance scenario orchestration."""

# pylint: disable=import-error

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from twinr.hardware.crazyflie_hover_recovery import HoverRecoveryWindow


_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "hardware" / "bitcraze"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from hover_physical_disturbance_scenarios import (  # noqa: E402
    _build_run_hover_sim_command,
    hover_physical_disturbance_scenario_names,
    hover_physical_disturbance_spec,
    run_hover_physical_disturbance_scenario,
)
from hover_sim_scenarios import HoverScenarioContractEvaluation  # noqa: E402


_RUN_HOVER_SIM_DISTURBANCES_SCRIPT_PATH = _SCRIPT_DIR / "run_hover_sim_disturbances.py"
_RUN_HOVER_SIM_DISTURBANCES_SPEC = importlib.util.spec_from_file_location(
    "bitcraze_run_hover_sim_disturbances_script",
    _RUN_HOVER_SIM_DISTURBANCES_SCRIPT_PATH,
)
assert (
    _RUN_HOVER_SIM_DISTURBANCES_SPEC is not None
    and _RUN_HOVER_SIM_DISTURBANCES_SPEC.loader is not None
)
_RUN_HOVER_SIM_DISTURBANCES_MODULE = importlib.util.module_from_spec(
    _RUN_HOVER_SIM_DISTURBANCES_SPEC
)
sys.modules[_RUN_HOVER_SIM_DISTURBANCES_SPEC.name] = _RUN_HOVER_SIM_DISTURBANCES_MODULE
_RUN_HOVER_SIM_DISTURBANCES_SPEC.loader.exec_module(_RUN_HOVER_SIM_DISTURBANCES_MODULE)


def _supervisor_info(*, is_flying: bool) -> int:
    return (1 << 3) | ((1 << 4) if is_flying else 0)


def _bounded_hover_report_payload() -> dict[str, object]:
    telemetry: list[dict[str, object]] = []
    samples = (
        (0, 0.00, False),
        (100, 0.04, False),
        (200, 0.08, True),
        (300, 0.10, True),
        (400, 0.10, True),
        (500, 0.10, True),
        (600, 0.08, True),
        (700, 0.04, False),
        (800, 0.02, False),
        (900, 0.01, False),
        (1000, 0.01, False),
        (1100, 0.01, False),
    )
    ground_tail = tuple((timestamp_ms, 0.01, False) for timestamp_ms in range(1200, 5200, 100))
    for timestamp_ms, z_m, is_flying in samples + ground_tail:
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
    return {
        "height_m": 0.10,
        "hover_duration_s": 0.10,
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
            ),
            "skipped_blocks": (),
        },
        "telemetry": telemetry,
    }


class HoverPhysicalDisturbanceScenarioTests(unittest.TestCase):
    def test_supported_scenarios_include_recovery_and_abort_cases(self) -> None:
        names = hover_physical_disturbance_scenario_names()

        self.assertIn("physical_forward_impulse_recovery", names)
        self.assertIn("physical_persistent_forward_abort", names)
        self.assertIn("physical_roll_torque_abort", names)

    def test_hover_recovery_disturbances_arm_only_after_nominal_climb_window(self) -> None:
        forward_spec = hover_physical_disturbance_spec("physical_forward_impulse_recovery")
        persistent_spec = hover_physical_disturbance_spec("physical_persistent_forward_abort")
        roll_abort_spec = hover_physical_disturbance_spec("physical_roll_torque_abort")
        height_spec = hover_physical_disturbance_spec("physical_height_drop_recovery")

        assert forward_spec.disturbance_plan is not None
        assert persistent_spec.disturbance_plan is not None
        assert roll_abort_spec.disturbance_plan is not None
        assert height_spec.disturbance_plan is not None

        self.assertEqual(forward_spec.disturbance_plan.activation_mode, "after_host_phase")
        self.assertEqual(forward_spec.disturbance_plan.activation_phase, "hover_primitive_hold")
        self.assertEqual(forward_spec.disturbance_plan.activation_status, "begin")
        self.assertEqual(height_spec.disturbance_plan.activation_mode, "after_host_phase")
        self.assertEqual(persistent_spec.disturbance_plan.activation_mode, "after_host_phase")
        self.assertEqual(roll_abort_spec.disturbance_plan.activation_mode, "after_airborne")
        self.assertGreaterEqual(forward_spec.disturbance_plan.pulses[0].start_s, 0.20)
        self.assertGreaterEqual(persistent_spec.disturbance_plan.pulses[0].start_s, 0.20)
        self.assertEqual(
            roll_abort_spec.expectation.failure_substrings,
            (
                "supervisor reported unsafe flags",
                "roll reached",
            ),
        )

    def test_build_run_hover_sim_command_threads_disturbance_inputs(self) -> None:
        spec = hover_physical_disturbance_spec("physical_forward_impulse_recovery")
        with tempfile.TemporaryDirectory() as temp_dir:
            command = _build_run_hover_sim_command(
                crazysim_root=Path("/tmp/crazysim"),
                backend="mujoco",
                model="cf2x_T350",
                x_m=0.0,
                y_m=0.0,
                startup_settle_s=3.0,
                hover_timeout_s=30.0,
                workspace=Path(temp_dir) / "workspace",
                python_bin=Path(sys.executable),
                display=":0",
                trace_file=Path(temp_dir) / "trace.jsonl",
                disturbance_spec_json=Path(temp_dir) / "plan.json",
                scenario=spec,
                hover_args=("--height-m", "0.25"),
            )

        self.assertIn("--disturbance-spec-json", command)
        self.assertIn(str(Path(temp_dir) / "plan.json"), command)
        self.assertIn("--wind-speed-mps", command)
        self.assertIn("--python-bin", command)
        self.assertIn("--", command)
        self.assertIn("--height-m", command)

    def test_run_hover_physical_disturbance_scenario_replays_captured_report(self) -> None:
        fake_completed = mock.Mock()
        fake_completed.returncode = 0
        fake_completed.stderr = ""
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
                    {
                        "phase": "hover_primitive_hold",
                        "status": "begin",
                        "elapsed_s": 2.6,
                    },
                ),
            }
        }
        successful_evaluation = HoverScenarioContractEvaluation(
            failure_tuple=(),
            matched_failure_substrings=(),
            missing_failure_substrings=(),
            contract_failures=(),
            guard_failures=(),
            recovery_failures=(),
        )
        runtime_event_log = "\n".join(
            (
                json.dumps(
                    {
                        "kind": "plan_anchor",
                        "plan_name": "forward_impulse",
                        "agent_id": 0,
                        "sim_time_s": 2.0,
                        "host_phase": "hover_primitive_hold",
                        "host_status": "begin",
                        "host_phase_elapsed_s": 2.6,
                    }
                ),
                json.dumps(
                    {
                        "kind": "pulse_active",
                        "plan_name": "forward_impulse",
                        "agent_id": 0,
                        "sim_time_s": 2.2,
                        "pulse_name": "forward_impulse",
                        "elapsed_since_anchor_s": 0.2,
                    }
                ),
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_log = Path(temp_dir) / "disturbance-runtime.jsonl"
            runtime_log.write_text(runtime_event_log, encoding="utf-8")
            fake_completed.stdout = json.dumps(
                {
                    "hover_report": _bounded_hover_report_payload(),
                    "sim": {"disturbance_runtime_jsonl": str(runtime_log)},
                }
            )
            with mock.patch(
                "hover_physical_disturbance_scenarios.subprocess.run",
                return_value=fake_completed,
            ) as subprocess_run, mock.patch(
                "hover_physical_disturbance_scenarios.run_hover_replay",
                return_value=replay_payload,
            ), mock.patch(
                "hover_physical_disturbance_scenarios.evaluate_hover_replay_against_expectation",
                return_value=successful_evaluation,
            ) as evaluation_mock:
                result = run_hover_physical_disturbance_scenario(
                    scenario_name="physical_forward_impulse_recovery",
                    crazysim_root=Path("/tmp/crazysim"),
                    output_dir=Path(temp_dir),
                    hover_args=(),
                )
            self.assertEqual(result.actual_outcome_class, "bounded_hover_ok")
            self.assertTrue(result.matches_expectation)
            self.assertTrue(result.report_json.is_file())
            self.assertIsNotNone(result.disturbance_spec_json)
            self.assertEqual(result.disturbance_runtime_jsonl, runtime_log.resolve())
            assert result.recovery_window is not None
            self.assertEqual(result.recovery_window.source, "physical_runtime")
            self.assertAlmostEqual(
                result.recovery_window.disturbance_start_elapsed_s,
                2.8,
                places=6,
            )
            self.assertAlmostEqual(
                result.recovery_window.disturbance_end_elapsed_s,
                3.15,
                places=6,
            )
            self.assertEqual(result.sim_returncode, 0)
            evaluation_mock.assert_called_once()
            self.assertEqual(
                evaluation_mock.call_args.kwargs["recovery_window"],
                result.recovery_window,
            )
            command = subprocess_run.call_args.args[0]
            workspace_index = command.index("--workspace") + 1
            self.assertTrue(
                str(command[workspace_index]).endswith("/physical_forward_impulse_recovery")
            )

    def test_run_hover_physical_disturbance_scenario_accepts_nonzero_sim_returncode_with_json(self) -> None:
        fake_completed = mock.Mock()
        fake_completed.returncode = 1
        fake_completed.stdout = json.dumps(
            {
                "hover_report": _bounded_hover_report_payload(),
                "sim": {"disturbance_runtime_jsonl": None},
            }
        )
        fake_completed.stderr = "sim hover aborted as expected"
        replay_payload = {
            "replay": {
                "outcome_class": "bounded_hover_ok",
                "failures": (),
                "primitive_outcome": {
                    "aborted": False,
                    "stable_hover_established": True,
                    "touchdown_confirmation_source": "range_only_sitl",
                },
            }
        }
        successful_evaluation = HoverScenarioContractEvaluation(
            failure_tuple=(),
            matched_failure_substrings=(),
            missing_failure_substrings=(),
            contract_failures=(),
            guard_failures=(),
            recovery_failures=(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch(
                "hover_physical_disturbance_scenarios.subprocess.run",
                return_value=fake_completed,
            ), mock.patch(
                "hover_physical_disturbance_scenarios.run_hover_replay",
                return_value=replay_payload,
            ), mock.patch(
                "hover_physical_disturbance_scenarios.evaluate_hover_replay_against_expectation",
                return_value=successful_evaluation,
            ):
                with self.assertRaises(RuntimeError) as exc_info:
                    run_hover_physical_disturbance_scenario(
                        scenario_name="physical_forward_impulse_recovery",
                        crazysim_root=Path("/tmp/crazysim"),
                        output_dir=Path(temp_dir),
                        hover_args=(),
                    )
            self.assertIn("missing runtime evidence", str(exc_info.exception))

    def test_run_hover_sim_disturbances_cli_serializes_results(self) -> None:
        fake_result = mock.Mock()
        fake_result.matches_expectation = True
        fake_result.actual_outcome_class = "bounded_hover_ok"
        fake_result.scenario = hover_physical_disturbance_spec("physical_forward_impulse_recovery")
        fake_result.sim_returncode = 0
        fake_result.report_json = Path("/tmp/report.json")
        fake_result.trace_file = Path("/tmp/trace.jsonl")
        fake_result.disturbance_spec_json = Path("/tmp/plan.json")
        fake_result.disturbance_runtime_jsonl = Path("/tmp/runtime.jsonl")
        fake_result.recovery_window = HoverRecoveryWindow(
            disturbance_start_elapsed_s=2.8,
            disturbance_end_elapsed_s=3.15,
            source="physical_runtime",
        )
        fake_result.evaluation = mock.Mock(
            failure_tuple=(),
            matched_failure_substrings=(),
            missing_failure_substrings=(),
            contract_failures=(),
            guard_failures=(),
            recovery_failures=(),
            guard_metrics=None,
            recovery_metrics=None,
        )
        fake_result.sim_payload = {"hover_report": _bounded_hover_report_payload()}

        stdout = io.StringIO()
        with (
            mock.patch.object(
                _RUN_HOVER_SIM_DISTURBANCES_MODULE,
                "run_hover_physical_disturbance_suite",
                return_value=(fake_result,),
            ) as suite_mock,
            mock.patch("sys.stdout", stdout),
        ):
            exit_code = _RUN_HOVER_SIM_DISTURBANCES_MODULE.main(
                [
                    "--crazysim-root",
                    "/tmp/crazysim",
                    "--scenario",
                    "physical_forward_impulse_recovery",
                    "--json",
                ]
            )
        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["all_matched_expectations"])
        self.assertEqual(payload["results"][0]["actual_outcome_class"], "bounded_hover_ok")
        self.assertEqual(payload["results"][0]["sim_returncode"], 0)
        self.assertEqual(
            suite_mock.call_args.kwargs["workspace"],
            Path(payload["output_dir"]) / "_workspaces",
        )


if __name__ == "__main__":
    unittest.main()
