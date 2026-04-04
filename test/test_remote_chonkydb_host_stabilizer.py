"""Regression tests for remote ChonkyDB host stabilization under contention."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from twinr.ops.remote_chonkydb_host_stabilizer import (
    RemoteHostStabilizationAction,
    RemoteHostTerminatedProcess,
    RemoteHostUnitState,
    apply_remote_host_stabilization,
    build_stale_process_rules,
    build_parser,
    fetch_remote_unit_states,
    probe_public_host_availability,
    stabilize_remote_chonkydb_host,
)
from twinr.ops.remote_chonkydb_repair import (
    ChonkyDBHttpProbeResult,
    RemoteChonkyDBOpsSettings,
)
from twinr.ops.self_coding_pi import PiConnectionSettings


_SETTINGS = RemoteChonkyDBOpsSettings(
    public_base_url="https://tessairact.com:2149",
    public_api_key="secret",
    public_api_key_header="x-api-key",
    ops_public_base_url="https://tessairact.com:2149",
    backend_local_base_url="http://127.0.0.1:3044",
    backend_service="caia-twinr-chonkydb-alt.service",
    runtime_namespace="twinr_longterm_v1:twinr:deff13356ec6",
    ssh=PiConnectionSettings(
        host="thh1986.ddns.net",
        user="thh",
        password="secret",
        port=22,
    ),
)


class RemoteChonkyDBHostStabilizerTests(unittest.TestCase):
    """Cover the orchestration and payload shaping for host stabilization."""

    def test_build_parser_uses_longer_default_ssh_timeout(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.ssh_timeout_s, 180.0)

    def test_build_stale_process_rules_adds_dedicated_backend_data_dir_matcher(self) -> None:
        rules = build_stale_process_rules(_SETTINGS)

        self.assertEqual(rules[0]["label"], "code_graph_benchmark_runner")
        data_rule = next(rule for rule in rules if rule["label"] == "dedicated_backend_data_path_writer")
        self.assertEqual(
            data_rule["required_substrings"],
            ("/home/thh/tessairact/state/offload/chonkydb/twinr_dedicated_3044/data",),
        )
        self.assertEqual(data_rule["minimum_elapsed_s"], 60.0)

    def test_stabilize_remote_host_reports_recovery_and_uses_curated_units(self) -> None:
        system_before_state = (
            RemoteHostUnitState(
                scope="system",
                unit="caia-code-graph-refresh.path",
                enabled_state="enabled",
                load_state="loaded",
                active_state="active",
                sub_state="running",
                result="success",
            ),
        )
        system_after_state = (
            RemoteHostUnitState(
                scope="system",
                unit="caia-code-graph-refresh.path",
                enabled_state="disabled",
                load_state="loaded",
                active_state="inactive",
                sub_state="dead",
                result="success",
            ),
        )
        user_before_state = (
            RemoteHostUnitState(
                scope="user",
                unit="caia-chonkycode-chunks-posttransform.path",
                enabled_state="enabled",
                load_state="loaded",
                active_state="active",
                sub_state="waiting",
                result="success",
            ),
        )
        user_after_state = (
            RemoteHostUnitState(
                scope="user",
                unit="caia-chonkycode-chunks-posttransform.path",
                enabled_state="disabled",
                load_state="loaded",
                active_state="inactive",
                sub_state="dead",
                result="success",
            ),
        )
        with patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.probe_public_host_availability",
            side_effect=[
                ChonkyDBHttpProbeResult(
                    label="public",
                    ok=False,
                    status_code=503,
                    ready=False,
                    detail="slow",
                ),
                ChonkyDBHttpProbeResult(
                    label="public",
                    ok=True,
                    status_code=200,
                    ready=True,
                    detail="ready",
                ),
            ],
        ), patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.fetch_remote_unit_states",
            side_effect=[
                system_before_state,
                user_before_state,
                system_after_state,
                user_after_state,
            ],
        ) as mock_states, patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.apply_remote_host_stabilization",
            return_value=RemoteHostStabilizationAction(
                kill_switch_paths=("/tmp/killswitch_a", "/tmp/killswitch_b"),
                disabled_system_units=("caia-code-graph-refresh.path",),
                disabled_user_units=("caia-chonkycode-chunks-posttransform.path",),
                terminated_processes=(
                    RemoteHostTerminatedProcess(
                        pid=2860300,
                        ppid=1,
                        elapsed_s=345600,
                        command_excerpt="python -m benchmarks.code_graph.benchmark",
                        scope="matched_root",
                        match_label="code_graph_benchmark_runner",
                        root_pid=2860300,
                        termination_signal="SIGKILL",
                    ),
                ),
            ),
        ) as mock_apply, patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.fetch_remote_service_properties",
            return_value={
                "CPUWeight": "10000",
                "StartupCPUWeight": "10000",
                "IOWeight": "10000",
                "StartupIOWeight": "10000",
            },
        ), patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.time.sleep",
        ):
            result = stabilize_remote_chonkydb_host(
                settings=_SETTINGS,
                probe_timeout_s=20.0,
                ssh_timeout_s=60.0,
                settle_s=1.0,
                executor=object(),  # type: ignore[arg-type]
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.diagnosis, "public_recovered_after_host_stabilization")
        self.assertEqual(result.backend_properties["CPUWeight"], "10000")
        self.assertEqual(result.kill_switch_paths, ("/tmp/killswitch_a", "/tmp/killswitch_b"))
        self.assertEqual(result.disabled_system_units, ("caia-code-graph-refresh.path",))
        self.assertEqual(
            result.disabled_user_units,
            ("caia-chonkycode-chunks-posttransform.path",),
        )
        self.assertEqual(len(result.terminated_processes), 1)
        self.assertEqual(result.terminated_processes[0].match_label, "code_graph_benchmark_runner")
        self.assertEqual(mock_states.call_count, 4)
        state_scopes = [call.kwargs["scope"] for call in mock_states.call_args_list]
        self.assertEqual(state_scopes, ["system", "user", "system", "user"])
        self.assertIn(
            "caia-code-graph-refresh.path",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-consumer-portal.service",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-consumer-portal-demo.service",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-chonky-transformer.timer",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-chonkycode-chunks-posttransform.path",
            mock_states.call_args_list[1].kwargs["units"],
        )
        self.assertIn(
            "caia-portal-llm-worker.service",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-consumer-portal.service",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-consumer-portal-demo.service",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-bug-memory-refresh.timer",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-ops-chonky-search-guardrail.service",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-ops-chonky-search-guardrail.timer",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-chonkycode-chunks-posttransform.path",
            mock_apply.call_args.kwargs["user_units"],
        )
        self.assertIn(
            "ollama-gpu.service",
            mock_apply.call_args.kwargs["user_units"],
        )
        self.assertEqual(mock_apply.call_args.kwargs["user_unit_owner"], "thh")
        self.assertEqual(mock_apply.call_args.kwargs["stale_process_min_elapsed_s"], 1800.0)
        self.assertTrue(
            any(
                rule["label"] == "dedicated_backend_data_path_writer"
                and rule["minimum_elapsed_s"] == 60.0
                and rule["required_substrings"]
                == ("/home/thh/tessairact/state/offload/chonkydb/twinr_dedicated_3044/data",)
                for rule in mock_apply.call_args.kwargs["stale_process_rules"]
            )
        )

    def test_apply_remote_host_stabilization_passes_payload_to_remote_helper(self) -> None:
        with patch(
            "twinr.ops.remote_chonkydb_host_stabilizer._run_remote_python_json",
            return_value={
                "kill_switch_paths": ["/tmp/kill_a"],
                "disabled_system_units": ["caia-a.service", "caia-a.timer"],
                "disabled_user_units": ["caia-user.path"],
                "terminated_processes": [
                    {
                        "pid": 2860300,
                        "ppid": 1,
                        "elapsed_s": 345600,
                        "command_excerpt": "python -m benchmarks.code_graph.benchmark",
                        "scope": "matched_root",
                        "match_label": "code_graph_benchmark_runner",
                        "root_pid": 2860300,
                        "termination_signal": "SIGKILL",
                    }
                ],
            },
        ) as mock_run:
            result = apply_remote_host_stabilization(
                executor=object(),  # type: ignore[arg-type]
                backend_service="caia-twinr-chonkydb-alt.service",
                system_units=("caia-a.service", "caia-a.timer"),
                user_units=("caia-user.path",),
                user_unit_owner="thh",
                kill_switch_paths=("/tmp/kill_a",),
                property_assignments={"CPUWeight": "10000"},
                stale_process_rules=(
                    {
                        "label": "code_graph_benchmark_runner",
                        "required_substrings": (
                            "benchmarks.code_graph.benchmark",
                            "run_code_graph_benchmark",
                        ),
                    },
                ),
                stale_process_min_elapsed_s=1800.0,
            )

        self.assertEqual(result.kill_switch_paths, ("/tmp/kill_a",))
        self.assertEqual(result.disabled_system_units, ("caia-a.service", "caia-a.timer"))
        self.assertEqual(result.disabled_user_units, ("caia-user.path",))
        self.assertEqual(len(result.terminated_processes), 1)
        self.assertEqual(result.terminated_processes[0].pid, 2860300)
        self.assertEqual(mock_run.call_args.kwargs["payload"]["backend_service"], "caia-twinr-chonkydb-alt.service")
        self.assertEqual(
            mock_run.call_args.kwargs["payload"]["system_units"],
            ["caia-a.service", "caia-a.timer"],
        )
        self.assertEqual(mock_run.call_args.kwargs["payload"]["user_units"], ["caia-user.path"])
        self.assertEqual(mock_run.call_args.kwargs["payload"]["user_unit_owner"], "thh")
        self.assertEqual(mock_run.call_args.kwargs["payload"]["kill_switch_paths"], ["/tmp/kill_a"])
        self.assertEqual(mock_run.call_args.kwargs["payload"]["property_assignments"], {"CPUWeight": "10000"})
        self.assertEqual(mock_run.call_args.kwargs["payload"]["stale_process_min_elapsed_s"], 1800.0)
        self.assertEqual(
            mock_run.call_args.kwargs["payload"]["stale_process_rules"],
            [
                {
                    "label": "code_graph_benchmark_runner",
                    "required_substrings": (
                        "benchmarks.code_graph.benchmark",
                        "run_code_graph_benchmark",
                    ),
                }
            ],
        )
        self.assertTrue(mock_run.call_args.kwargs["use_sudo"])

    def test_fetch_remote_unit_states_converts_remote_payload(self) -> None:
        with patch(
            "twinr.ops.remote_chonkydb_host_stabilizer._run_remote_python_json",
            return_value={
                "units": [
                    {
                        "scope": "system",
                        "unit": "caia-a.service",
                        "enabled_state": "disabled",
                        "load_state": "loaded",
                        "active_state": "inactive",
                        "sub_state": "dead",
                        "result": "success",
                    }
                ]
            },
        ):
            states = fetch_remote_unit_states(
                executor=object(),  # type: ignore[arg-type]
                units=("caia-a.service",),
                scope="system",
                user_unit_owner="thh",
            )

        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].scope, "system")
        self.assertEqual(states[0].unit, "caia-a.service")
        self.assertEqual(states[0].enabled_state, "disabled")
        self.assertEqual(states[0].sub_state, "dead")

    def test_probe_public_host_availability_uses_query_surface_probe(self) -> None:
        with patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.probe_public_chonkydb",
            return_value=ChonkyDBHttpProbeResult(
                label="public",
                ok=True,
                status_code=200,
                ready=True,
                detail="ready",
                url="https://tessairact.com:2149/v1/external/retrieve/topk_records",
            ),
        ) as mock_probe:
            result = probe_public_host_availability(
                settings=_SETTINGS,
                timeout_s=20.0,
            )

        self.assertTrue(result.ready)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(mock_probe.call_count, 1)
        self.assertEqual(mock_probe.call_args.kwargs["settings"], _SETTINGS)
        self.assertEqual(mock_probe.call_args.kwargs["timeout_s"], 20.0)


if __name__ == "__main__":
    unittest.main()
