"""Regression tests for remote ChonkyDB host stabilization under contention."""

from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from twinr.ops.remote_chonkydb_host_stabilizer import (
    RemoteHostStabilizationAction,
    RemoteHostBootPacingStatus,
    RemoteHostBootPacingStep,
    RemoteHostTerminatedProcess,
    RemoteHostUnitState,
    _HOST_CONTROL_GUARD_UNIT,
    _HOST_BOOT_PACER_SERVICE,
    _HOST_STABILIZER_UNBLOCK_PATH,
    _REMOTE_BOOT_PACER_SCRIPT,
    _REMOTE_SYNC_BOOT_PACING_CODE,
    _REMOTE_STABILIZE_HOST_CODE,
    apply_remote_host_stabilization,
    build_remote_host_boot_pacing_steps,
    ensure_remote_host_boot_pacing_policy,
    build_stale_process_rules,
    build_parser,
    fetch_remote_guard_protected_system_units,
    fetch_remote_unit_states,
    probe_public_host_availability,
    stabilize_remote_chonkydb_host,
    _run_remote_python_json,
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
_BOOT_PACING_STATUS = RemoteHostBootPacingStatus(
    service_name=_HOST_BOOT_PACER_SERVICE,
    script_path="/usr/local/sbin/caia_twinr_host_boot_pacer.py",
    config_path="/etc/twinr/caia_twinr_host_boot_pacer.json",
    release_root="/run/caia/twinr_host_boot_pacer/releases",
    paced_system_units=(),
    paced_user_units=(),
    always_disabled_system_units=(),
    default_target="multi-user.target",
)


class RemoteChonkyDBHostStabilizerTests(unittest.TestCase):
    """Cover the orchestration and payload shaping for host stabilization."""

    def test_run_remote_python_json_sudo_stages_code_and_payload_via_tempfiles(self) -> None:
        captured: dict[str, str] = {}

        class _Executor:
            def run_sudo_ssh(self, script: str) -> subprocess.CompletedProcess[str]:
                captured["script"] = script
                return subprocess.CompletedProcess(
                    args=["ssh"],
                    returncode=0,
                    stdout='{"ok": true}',
                    stderr="",
                )

        payload = _run_remote_python_json(
            executor=_Executor(),  # type: ignore[arg-type]
            code="import json\nimport sys\njson.dump({'ok': True}, sys.stdout)\n",
            payload={"message": "hello"},
            use_sudo=True,
        )

        self.assertEqual(payload, {"ok": True})
        script = captured["script"]
        self.assertIn("tmp_code=$(mktemp)", script)
        self.assertIn("tmp_payload=$(mktemp)", script)
        self.assertIn('python3 "$tmp_code" < "$tmp_payload"', script)
        self.assertNotIn("python3 -c", script)

    def test_build_parser_uses_longer_default_ssh_timeout(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.ssh_timeout_s, 180.0)

    def test_build_stale_process_rules_adds_dedicated_backend_data_dir_matcher(self) -> None:
        rules = build_stale_process_rules(_SETTINGS)

        self.assertEqual(rules[0]["label"], "code_graph_benchmark_runner")
        artifact_ingest_rule = next(
            rule for rule in rules if rule["label"] == "chonkycode_artifact_ingest_runner"
        )
        self.assertEqual(
            artifact_ingest_rule["required_substrings"],
            ("-m chonkycode.cli", "artifact-ingest"),
        )
        self.assertEqual(artifact_ingest_rule["minimum_elapsed_s"], 300.0)
        locomo_eval_rule = next(
            rule for rule in rules if rule["label"] == "ccodex_memory_locomo_eval_runner"
        )
        self.assertEqual(
            locomo_eval_rule["required_substrings"],
            ("benchmarks/ccodex_memory/ccodex_memory_locomo_mc10_eval.py",),
        )
        self.assertEqual(locomo_eval_rule["minimum_elapsed_s"], 300.0)
        unmanaged_api_rule = next(
            rule for rule in rules if rule["label"] == "unmanaged_chonkydb_api_server_listener"
        )
        self.assertEqual(
            unmanaged_api_rule["required_substrings"],
            ("tessairact.automations.helpers.launcher --module chonkydb.api.server",),
        )
        self.assertEqual(
            unmanaged_api_rule["excluded_cgroup_substrings"],
            (
                "/system.slice/caia-twinr-chonkydb-alt.service",
                "/system.slice/caia-ccodex-memory-api.service",
                "/system.slice/caia-chonkycode-api.service",
            ),
        )
        self.assertTrue(unmanaged_api_rule["require_listener"])
        self.assertEqual(unmanaged_api_rule["minimum_elapsed_s"], 60.0)
        data_rule = next(rule for rule in rules if rule["label"] == "dedicated_backend_data_path_writer")
        self.assertEqual(
            data_rule["required_substrings"],
            ("/home/thh/tessairact/state/offload/chonkydb/twinr_dedicated_3044/data",),
        )
        self.assertEqual(data_rule["minimum_elapsed_s"], 60.0)

    def test_fetch_remote_guard_protected_system_units_converts_remote_payload(self) -> None:
        with patch(
            "twinr.ops.remote_chonkydb_host_stabilizer._run_remote_python_json",
            return_value={
                "units": [
                    {
                        "unit": "caia-consumer-portal.service",
                        "reason": "portal must stay live",
                    },
                    {
                        "unit": "caia-code-graph-refresh.timer",
                        "reason": "safety-net timer",
                    },
                ]
            },
        ):
            units = fetch_remote_guard_protected_system_units(
                executor=object(),  # type: ignore[arg-type]
                backend_service="caia-twinr-chonkydb-alt.service",
            )

        self.assertEqual(
            units,
            (
                "caia-consumer-portal.service",
                "caia-code-graph-refresh.timer",
            ),
        )

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
            "twinr.ops.remote_chonkydb_host_stabilizer.ensure_remote_host_control_permit",
        ) as mock_open_permit, patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.remove_remote_host_control_permit",
        ) as mock_close_permit, patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.ensure_remote_host_boot_pacing_policy",
            return_value=_BOOT_PACING_STATUS,
        ), patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.fetch_remote_guard_protected_system_units",
            return_value=(),
        ), patch(
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
        self.assertEqual(mock_open_permit.call_count, 1)
        self.assertEqual(mock_close_permit.call_count, 1)
        self.assertEqual(result.backend_properties["CPUWeight"], "10000")
        self.assertEqual(result.kill_switch_paths, ("/tmp/killswitch_a", "/tmp/killswitch_b"))
        self.assertEqual(result.disabled_system_units, ("caia-code-graph-refresh.path",))
        self.assertEqual(
            result.disabled_user_units,
            ("caia-chonkycode-chunks-posttransform.path",),
        )
        self.assertEqual(len(result.terminated_processes), 1)
        self.assertEqual(result.terminated_processes[0].match_label, "code_graph_benchmark_runner")
        self.assertEqual(result.reactivation_hold_polls, 0)
        self.assertEqual(result.reactivation_hold_elapsed_s, 0.0)
        self.assertEqual(mock_states.call_count, 4)
        state_scopes = [call.kwargs["scope"] for call in mock_states.call_args_list]
        self.assertEqual(state_scopes, ["system", "user", "system", "user"])
        self.assertIn(
            "caia-agent-tools-mcp.service",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-code-graph-refresh.path",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-consumer-portal-config-helper.service",
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
            "caia-consumer-terminald.service",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-ccodex-memory-api.service",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-control-plane-edge.service",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-control-plane-portal.service",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-consumer-portal-live-restart-guard.service",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-consumer-portal-live-restart-guard.timer",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-consumer-portal-live-restart-guard.path",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "codex-portal-live-override.service",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-ccodex-memory-live-restart-guard.service",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-ccodex-memory-live-restart-guard.timer",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-ccodex-memory-live-restart-guard.path",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-chonkycode-api.service",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-external-site.service",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-gpu-embeddings.service",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-if-logic-monitor.service",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-joint-retrieval.service",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-ollama-gpu-proxy.service",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-portal-findings-housekeeper-autopilot.service",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-portal-fixreports-analysis-autopilot.service",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-portal-tasks-review-autopilot.service",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-qwen3tts-api.service",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-repo-script-llm-enricher.service",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-stt.service",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-chonky-transformer.timer",
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            _HOST_CONTROL_GUARD_UNIT,
            mock_states.call_args_list[0].kwargs["units"],
        )
        self.assertIn(
            "caia-chonkycode-chunks-posttransform.path",
            mock_states.call_args_list[1].kwargs["units"],
        )
        self.assertIn(
            "caia-agent-tools-mcp.service",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-portal-llm-worker.service",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-consumer-portal-config-helper.service",
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
            "caia-consumer-terminald.service",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-ccodex-memory-api.service",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-control-plane-edge.service",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-control-plane-portal.service",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-consumer-portal-live-restart-guard.service",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-consumer-portal-live-restart-guard.timer",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-consumer-portal-live-restart-guard.path",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "codex-portal-live-override.service",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-ccodex-memory-live-restart-guard.service",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-ccodex-memory-live-restart-guard.timer",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-ccodex-memory-live-restart-guard.path",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-chonkycode-api.service",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-external-site.service",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-gpu-embeddings.service",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-if-logic-monitor.service",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-joint-retrieval.service",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-ollama-gpu-proxy.service",
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
            "caia-portal-findings-housekeeper-autopilot.service",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-portal-fixreports-analysis-autopilot.service",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-portal-tasks-review-autopilot.service",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-qwen3tts-api.service",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-stt.service",
            mock_apply.call_args.kwargs["system_units"],
        )
        self.assertIn(
            "caia-chonkycode-chunks-posttransform.path",
            mock_apply.call_args.kwargs["user_units"],
        )
        self.assertIn(
            "caia-molt.service",
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
                rule["label"] == "chonkycode_artifact_ingest_runner"
                and rule["minimum_elapsed_s"] == 300.0
                and rule["required_substrings"] == ("-m chonkycode.cli", "artifact-ingest")
                for rule in mock_apply.call_args.kwargs["stale_process_rules"]
            )
        )
        self.assertTrue(
            any(
                rule["label"] == "ccodex_memory_locomo_eval_runner"
                and rule["minimum_elapsed_s"] == 300.0
                and rule["required_substrings"]
                == ("benchmarks/ccodex_memory/ccodex_memory_locomo_mc10_eval.py",)
                for rule in mock_apply.call_args.kwargs["stale_process_rules"]
            )
        )
        self.assertTrue(
            any(
                rule["label"] == "dedicated_backend_data_path_writer"
                and rule["minimum_elapsed_s"] == 60.0
                and rule["required_substrings"]
                == ("/home/thh/tessairact/state/offload/chonkydb/twinr_dedicated_3044/data",)
                for rule in mock_apply.call_args.kwargs["stale_process_rules"]
            )
        )
        self.assertTrue(
            any(
                rule["label"] == "unmanaged_chonkydb_api_server_listener"
                and rule["minimum_elapsed_s"] == 60.0
                and rule["required_substrings"]
                == ("tessairact.automations.helpers.launcher --module chonkydb.api.server",)
                and rule["excluded_cgroup_substrings"]
                == (
                    "/system.slice/caia-twinr-chonkydb-alt.service",
                    "/system.slice/caia-ccodex-memory-api.service",
                    "/system.slice/caia-chonkycode-api.service",
                )
                and rule["require_listener"] is True
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
        self.assertEqual(
            mock_run.call_args.kwargs["payload"]["runtime_block_unblock_path"],
            _HOST_STABILIZER_UNBLOCK_PATH,
        )
        self.assertEqual(mock_run.call_args.kwargs["payload"]["property_assignments"], {"CPUWeight": "10000"})
        self.assertEqual(mock_run.call_args.kwargs["payload"]["stale_process_min_elapsed_s"], 1800.0)
        self.assertEqual(mock_run.call_args.kwargs["payload"]["unit_quiesce_s"], 0.25)
        self.assertEqual(mock_run.call_args.kwargs["payload"]["killed_unit_quiesce_s"], 1.5)
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

    def test_build_remote_host_boot_pacing_steps_orders_boot_release_waves(self) -> None:
        steps = build_remote_host_boot_pacing_steps(
            system_units=(
                "caia-consumer-portal.service",
                "caia-consumer-portal-demo.service",
                "caia-ccodex-memory-api.service",
                "caia-portal-llm-worker.service",
                "caia-repo-script-indexer.service",
                "caia-external-site.service",
                "sunshine-headless.service",
                "gdm.service",
                "caia-code-graph-refresh.timer",
                "caia-consumer-portal-live-restart-guard.timer",
                _HOST_CONTROL_GUARD_UNIT,
            ),
            user_units=("caia-molt.service",),
        )

        self.assertEqual(
            steps[0],
            RemoteHostBootPacingStep(
                scope="system",
                units=("caia-ccodex-memory-api.service", "caia-consumer-portal.service"),
                sleep_before_s=20.0,
                inter_unit_delay_s=8.0,
            ),
        )
        self.assertEqual(
            steps[1],
            RemoteHostBootPacingStep(
                scope="system",
                units=("caia-portal-llm-worker.service", "caia-repo-script-indexer.service"),
                sleep_before_s=30.0,
                inter_unit_delay_s=8.0,
            ),
        )
        self.assertEqual(
            steps[2],
            RemoteHostBootPacingStep(
                scope="system",
                units=("caia-code-graph-refresh.timer",),
                sleep_before_s=20.0,
                inter_unit_delay_s=1.5,
            ),
        )
        self.assertEqual(
            steps[3],
            RemoteHostBootPacingStep(
                scope="system",
                units=("caia-consumer-portal-live-restart-guard.timer", _HOST_CONTROL_GUARD_UNIT),
                sleep_before_s=20.0,
                inter_unit_delay_s=1.5,
            ),
        )
        flattened_units = tuple(unit for step in steps for unit in step.units)
        self.assertNotIn("caia-consumer-portal-demo.service", flattened_units)
        self.assertNotIn("caia-external-site.service", flattened_units)
        self.assertNotIn("caia-molt.service", flattened_units)
        self.assertNotIn("sunshine-headless.service", flattened_units)
        self.assertNotIn("gdm.service", flattened_units)

    def test_ensure_remote_host_boot_pacing_policy_passes_remote_sync_payload(self) -> None:
        with patch(
            "twinr.ops.remote_chonkydb_host_stabilizer._run_remote_python_json",
            return_value={
                "service_name": _HOST_BOOT_PACER_SERVICE,
                "script_path": "/usr/local/sbin/caia_twinr_host_boot_pacer.py",
                "config_path": "/etc/twinr/caia_twinr_host_boot_pacer.json",
                "release_root": "/run/caia/twinr_host_boot_pacer/releases",
                "paced_system_units": [
                    "caia-ccodex-memory-api.service",
                    "caia-consumer-portal.service",
                    "caia-consumer-portal-live-restart-guard.timer",
                ],
                "paced_user_units": [],
                "always_disabled_system_units": [
                    "sunshine-headless.service",
                    "gdm.service",
                ],
                "default_target": "multi-user.target",
            },
        ) as mock_run:
            status = ensure_remote_host_boot_pacing_policy(
                executor=object(),  # type: ignore[arg-type]
                system_units=(
                    "caia-ccodex-memory-api.service",
                    "caia-consumer-portal.service",
                    "caia-consumer-portal-live-restart-guard.timer",
                    "caia-external-site.service",
                    "sunshine-headless.service",
                    "gdm.service",
                ),
                user_units=("caia-molt.service",),
                user_unit_owner="thh",
            )

        self.assertEqual(status.service_name, _HOST_BOOT_PACER_SERVICE)
        self.assertEqual(
            status.paced_system_units,
            (
                "caia-ccodex-memory-api.service",
                "caia-consumer-portal.service",
                "caia-consumer-portal-live-restart-guard.timer",
            ),
        )
        self.assertEqual(status.paced_user_units, ())
        self.assertEqual(
            status.always_disabled_system_units,
            ("sunshine-headless.service", "gdm.service"),
        )
        self.assertEqual(status.default_target, "multi-user.target")
        self.assertIn("ConditionPathExists=", mock_run.call_args.kwargs["code"])
        self.assertEqual(
            mock_run.call_args.kwargs["payload"]["service_name"],
            _HOST_BOOT_PACER_SERVICE,
        )
        self.assertEqual(
            mock_run.call_args.kwargs["payload"]["all_system_units"],
            [
                "caia-ccodex-memory-api.service",
                "caia-consumer-portal.service",
                "caia-consumer-portal-live-restart-guard.timer",
                "caia-external-site.service",
                "sunshine-headless.service",
                "gdm.service",
            ],
        )
        self.assertEqual(
            mock_run.call_args.kwargs["payload"]["always_disabled_system_units"],
            ["sunshine-headless.service", "gdm.service"],
        )
        self.assertEqual(
            mock_run.call_args.kwargs["payload"]["all_user_units"],
            ["caia-molt.service"],
        )
        self.assertEqual(
            mock_run.call_args.kwargs["payload"]["default_target"],
            "multi-user.target",
        )
        self.assertEqual(mock_run.call_args.kwargs["payload"]["user_unit_owner"], "thh")
        self.assertEqual(
            mock_run.call_args.kwargs["payload"]["steps"][0]["units"],
            ["caia-ccodex-memory-api.service", "caia-consumer-portal.service"],
        )
        self.assertEqual(
            mock_run.call_args.kwargs["payload"]["steps"][-1]["units"],
            ["caia-consumer-portal-live-restart-guard.timer"],
        )
        self.assertNotIn(
            "caia-external-site.service",
            tuple(
                unit
                for step in mock_run.call_args.kwargs["payload"]["steps"]
                for unit in step["units"]
            ),
        )
        self.assertTrue(mock_run.call_args.kwargs["use_sudo"])

    def test_remote_boot_pacing_helper_syncs_persistent_dropins_and_service(self) -> None:
        self.assertIn("ConditionPathExists=", _REMOTE_SYNC_BOOT_PACING_CODE)
        self.assertIn("/etc/systemd/system", _REMOTE_SYNC_BOOT_PACING_CODE)
        self.assertIn(".config/systemd/user", _REMOTE_SYNC_BOOT_PACING_CODE)
        self.assertIn('["systemctl", "disable", unit]', _REMOTE_SYNC_BOOT_PACING_CODE)
        self.assertIn('["systemctl", "stop", unit]', _REMOTE_SYNC_BOOT_PACING_CODE)
        self.assertIn('["systemctl", "set-default", default_target]', _REMOTE_SYNC_BOOT_PACING_CODE)
        self.assertIn('["systemctl", "enable", service_name]', _REMOTE_SYNC_BOOT_PACING_CODE)
        self.assertIn("ExecStart=/usr/bin/python3", _REMOTE_SYNC_BOOT_PACING_CODE)
        self.assertIn("Type=oneshot", _REMOTE_SYNC_BOOT_PACING_CODE)
        self.assertIn("WantedBy=multi-user.target", _REMOTE_SYNC_BOOT_PACING_CODE)
        self.assertIn("_touch_release(release_root / scope / unit)", _REMOTE_BOOT_PACER_SCRIPT)
        self.assertIn('["systemctl", "start", unit]', _REMOTE_BOOT_PACER_SCRIPT)
        self.assertIn('"systemctl",', _REMOTE_BOOT_PACER_SCRIPT)
        self.assertIn('"--user",', _REMOTE_BOOT_PACER_SCRIPT)

    def test_remote_helper_imports_regex_for_listener_pid_matching(self) -> None:
        self.assertIn("import re", _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn("re.finditer", _REMOTE_STABILIZE_HOST_CODE)

    def test_remote_helper_installs_runtime_blockers_for_system_and_user_units(self) -> None:
        self.assertIn('_runtime_block_dropin_path(unit)', _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn('_user_runtime_block_dropin_path(unit)', _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn("91-twinr-private-host-stabilizer-block.conf", _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn("ConditionPathExists=", _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn('["systemctl", "daemon-reload"]', _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn("_reload_system_manager()", _REMOTE_STABILIZE_HOST_CODE)
        self.assertNotIn('["systemctl", "mask", unit]', _REMOTE_STABILIZE_HOST_CODE)
        self.assertNotIn('_run_user_systemctl("mask", unit)', _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn('_run_user_systemctl("daemon-reload")', _REMOTE_STABILIZE_HOST_CODE)

    def test_remote_helper_replaces_legacy_shared_blocker_filename(self) -> None:
        self.assertIn("90-twinr-host-stabilizer-block.conf", _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn("91-twinr-private-host-stabilizer-block.conf", _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn("_remove_legacy_system_runtime_blocker(unit)", _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn("_remove_legacy_user_runtime_blocker(unit)", _REMOTE_STABILIZE_HOST_CODE)

    def test_runtime_blocker_uses_private_unblock_path_not_shared_guard_path(self) -> None:
        self.assertEqual(
            _HOST_STABILIZER_UNBLOCK_PATH,
            "/run/caia/maintenance/twinr_private_host_stabilizer_unblock",
        )
        self.assertNotEqual(
            _HOST_STABILIZER_UNBLOCK_PATH,
            "/run/caia/maintenance/twinr_host_stabilizer_unblock",
        )

    def test_remote_helper_excludes_its_own_process_tree_from_stale_kills(self) -> None:
        self.assertIn("exempt_pids = {os.getpid()}", _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn("if pid in exempt_pids:", _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn("if child_pid in target_pids or child_pid in exempt_pids:", _REMOTE_STABILIZE_HOST_CODE)

    def test_remote_helper_bounds_hung_unit_stops_with_kill_fallback(self) -> None:
        self.assertIn("unit_control_timeout_s = 8.0", _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn("unit_kill_timeout_s = 5.0", _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn('["systemctl", "stop", unit]', _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn('["systemctl", "kill", "--kill-who=all", "--signal=SIGKILL", unit]', _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn('"kill",', _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn('"--kill-who=all",', _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn('"--signal=SIGKILL",', _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn("if completed is not None and completed.returncode == 0:", _REMOTE_STABILIZE_HOST_CODE)

    def test_remote_helper_staggers_unit_quiesce_with_longer_pause_after_forced_kill(self) -> None:
        self.assertIn('unit_quiesce_s = max(0.0, float(payload.get("unit_quiesce_s", 0.0) or 0.0))', _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn('float(payload.get("killed_unit_quiesce_s", 0.0) or 0.0)', _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn("def _quiesce_after_unit(*, forced_kill: bool) -> None:", _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn("sleep_s = killed_unit_quiesce_s if forced_kill else unit_quiesce_s", _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn("_quiesce_after_unit(forced_kill=forced_kill)", _REMOTE_STABILIZE_HOST_CODE)

    def test_remote_helper_resets_failed_only_when_units_still_report_failure_state(self) -> None:
        self.assertIn("def _unit_needs_reset(state: dict[str, str]) -> bool:", _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn('if load_state != "loaded":', _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn('if active_state == "failed":', _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn('return result not in {"", "success"}', _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn("def _maybe_reset_failed_system_unit(unit: str, *, label: str) -> None:", _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn("def _maybe_reset_failed_user_unit(unit: str, *, label: str) -> None:", _REMOTE_STABILIZE_HOST_CODE)
        self.assertIn("if not _unit_needs_reset(state):", _REMOTE_STABILIZE_HOST_CODE)

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

    def test_stabilize_remote_host_allows_transient_reactivation_to_settle_within_hold(self) -> None:
        system_before_state = (
            RemoteHostUnitState(
                scope="system",
                unit="caia-consumer-portal.service",
                enabled_state="enabled",
                load_state="loaded",
                active_state="active",
                sub_state="running",
                result="success",
            ),
        )
        user_before_state: tuple[RemoteHostUnitState, ...] = ()
        system_after_first = (
            RemoteHostUnitState(
                scope="system",
                unit="caia-consumer-portal.service",
                enabled_state="disabled",
                load_state="loaded",
                active_state="active",
                sub_state="running",
                result="success",
            ),
        )
        system_after_hold = (
            RemoteHostUnitState(
                scope="system",
                unit="caia-consumer-portal.service",
                enabled_state="disabled",
                load_state="not-found",
                active_state="inactive",
                sub_state="dead",
                result="success",
            ),
        )
        with patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.ensure_remote_host_control_permit",
        ) as mock_open_permit, patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.remove_remote_host_control_permit",
        ) as mock_close_permit, patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.ensure_remote_host_boot_pacing_policy",
            return_value=_BOOT_PACING_STATUS,
        ), patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.fetch_remote_guard_protected_system_units",
            return_value=(),
        ), patch(
            "twinr.ops.remote_chonkydb_host_stabilizer._DEFAULT_REACTIVATION_HOLD_S",
            1.0,
        ), patch(
            "twinr.ops.remote_chonkydb_host_stabilizer._DEFAULT_REACTIVATION_HOLD_POLL_S",
            1.0,
        ), patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.probe_public_host_availability",
            side_effect=[
                ChonkyDBHttpProbeResult(
                    label="public",
                    ok=True,
                    status_code=200,
                    ready=True,
                    detail="ready",
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
                system_after_first,
                user_before_state,
                system_after_hold,
                user_before_state,
            ],
        ) as mock_states, patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.apply_remote_host_stabilization",
            return_value=RemoteHostStabilizationAction(
                kill_switch_paths=("/tmp/kill_a",),
                disabled_system_units=("caia-consumer-portal.service",),
                disabled_user_units=(),
                terminated_processes=(),
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
        self.assertEqual(result.diagnosis, "public_ready_after_host_stabilization")
        self.assertEqual(mock_open_permit.call_count, 1)
        self.assertEqual(mock_close_permit.call_count, 1)
        self.assertEqual(result.system_units_after[0].load_state, "not-found")
        self.assertEqual(result.system_units_after[0].active_state, "inactive")
        self.assertEqual(result.reactivation_hold_polls, 1)
        self.assertEqual(result.reactivation_hold_elapsed_s, 1.0)
        self.assertEqual(mock_apply.call_count, 1)
        self.assertEqual(mock_states.call_count, 6)

    def test_stabilize_remote_host_retries_units_that_stay_reactivated_after_hold(self) -> None:
        system_before_state = (
            RemoteHostUnitState(
                scope="system",
                unit="caia-chonkycode-api.service",
                enabled_state="enabled",
                load_state="loaded",
                active_state="active",
                sub_state="running",
                result="success",
            ),
        )
        user_before_state = (
            RemoteHostUnitState(
                scope="user",
                unit="ollama-gpu.service",
                enabled_state="enabled",
                load_state="loaded",
                active_state="active",
                sub_state="running",
                result="success",
            ),
        )
        system_after_first = (
            RemoteHostUnitState(
                scope="system",
                unit="caia-chonkycode-api.service",
                enabled_state="enabled",
                load_state="loaded",
                active_state="active",
                sub_state="running",
                result="success",
            ),
        )
        user_after_first = (
            RemoteHostUnitState(
                scope="user",
                unit="ollama-gpu.service",
                enabled_state="disabled",
                load_state="loaded",
                active_state="inactive",
                sub_state="dead",
                result="success",
            ),
        )
        system_after_hold = system_after_first
        system_after_second = (
            RemoteHostUnitState(
                scope="system",
                unit="caia-chonkycode-api.service",
                enabled_state="disabled",
                load_state="loaded",
                active_state="inactive",
                sub_state="dead",
                result="success",
            ),
        )
        user_after_second = user_after_first
        with patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.ensure_remote_host_control_permit",
        ) as mock_open_permit, patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.remove_remote_host_control_permit",
        ) as mock_close_permit, patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.ensure_remote_host_boot_pacing_policy",
            return_value=_BOOT_PACING_STATUS,
        ), patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.fetch_remote_guard_protected_system_units",
            return_value=(),
        ), patch(
            "twinr.ops.remote_chonkydb_host_stabilizer._DEFAULT_REACTIVATION_HOLD_S",
            1.0,
        ), patch(
            "twinr.ops.remote_chonkydb_host_stabilizer._DEFAULT_REACTIVATION_HOLD_POLL_S",
            1.0,
        ), patch(
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
                system_after_first,
                user_after_first,
                system_after_hold,
                user_after_first,
                system_after_second,
                user_after_second,
            ],
        ) as mock_states, patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.apply_remote_host_stabilization",
            side_effect=[
                RemoteHostStabilizationAction(
                    kill_switch_paths=("/tmp/kill_a",),
                    disabled_system_units=("caia-chonkycode-api.service",),
                    disabled_user_units=("ollama-gpu.service",),
                    terminated_processes=(),
                ),
                RemoteHostStabilizationAction(
                    kill_switch_paths=("/tmp/kill_a",),
                    disabled_system_units=("caia-chonkycode-api.service",),
                    disabled_user_units=(),
                    terminated_processes=(),
                ),
            ],
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
        self.assertEqual(mock_open_permit.call_count, 1)
        self.assertEqual(mock_close_permit.call_count, 1)
        self.assertEqual(mock_apply.call_count, 2)
        self.assertEqual(result.reactivation_hold_polls, 1)
        self.assertEqual(result.reactivation_hold_elapsed_s, 1.0)
        self.assertEqual(mock_states.call_count, 8)
        self.assertEqual(
            mock_apply.call_args_list[1].kwargs["system_units"],
            ("caia-chonkycode-api.service",),
        )
        self.assertEqual(mock_apply.call_args_list[1].kwargs["user_units"], ())

    def test_stabilize_remote_host_fails_closed_when_conflict_units_reactivate_twice(self) -> None:
        system_before_state = (
            RemoteHostUnitState(
                scope="system",
                unit="caia-consumer-portal.service",
                enabled_state="enabled",
                load_state="loaded",
                active_state="active",
                sub_state="running",
                result="success",
            ),
        )
        user_before_state: tuple[RemoteHostUnitState, ...] = ()
        system_after_first = (
            RemoteHostUnitState(
                scope="system",
                unit="caia-consumer-portal.service",
                enabled_state="enabled",
                load_state="loaded",
                active_state="active",
                sub_state="running",
                result="success",
            ),
        )
        system_after_second = system_after_first
        with patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.ensure_remote_host_control_permit",
        ) as mock_open_permit, patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.remove_remote_host_control_permit",
        ) as mock_close_permit, patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.ensure_remote_host_boot_pacing_policy",
            return_value=_BOOT_PACING_STATUS,
        ), patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.fetch_remote_guard_protected_system_units",
            return_value=(),
        ), patch(
            "twinr.ops.remote_chonkydb_host_stabilizer._DEFAULT_REACTIVATION_HOLD_S",
            1.0,
        ), patch(
            "twinr.ops.remote_chonkydb_host_stabilizer._DEFAULT_REACTIVATION_HOLD_POLL_S",
            1.0,
        ), patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.probe_public_host_availability",
            side_effect=[
                ChonkyDBHttpProbeResult(
                    label="public",
                    ok=True,
                    status_code=200,
                    ready=True,
                    detail="ready",
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
                system_after_first,
                user_before_state,
                system_after_first,
                user_before_state,
                system_after_second,
                user_before_state,
                system_after_second,
                user_before_state,
            ],
        ) as mock_states, patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.apply_remote_host_stabilization",
            side_effect=[
                RemoteHostStabilizationAction(
                    kill_switch_paths=("/tmp/kill_a",),
                    disabled_system_units=("caia-consumer-portal.service",),
                    disabled_user_units=(),
                    terminated_processes=(),
                ),
                RemoteHostStabilizationAction(
                    kill_switch_paths=("/tmp/kill_a",),
                    disabled_system_units=("caia-consumer-portal.service",),
                    disabled_user_units=(),
                    terminated_processes=(),
                ),
            ],
        ), patch(
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

        self.assertFalse(result.ok)
        self.assertEqual(result.diagnosis, "conflict_units_reactivated_after_host_stabilization")
        self.assertEqual(mock_open_permit.call_count, 1)
        self.assertEqual(mock_close_permit.call_count, 1)
        self.assertEqual(result.reactivation_hold_polls, 2)
        self.assertEqual(result.reactivation_hold_elapsed_s, 2.0)
        self.assertEqual(mock_states.call_count, 10)

    def test_stabilize_remote_host_removes_maintenance_permit_when_apply_fails(self) -> None:
        with patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.fetch_remote_guard_protected_system_units",
            return_value=(),
        ), patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.probe_public_host_availability",
            return_value=ChonkyDBHttpProbeResult(
                label="public",
                ok=False,
                status_code=503,
                ready=False,
                detail="slow",
            ),
        ), patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.fetch_remote_unit_states",
            side_effect=[(), ()],
        ), patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.ensure_remote_host_control_permit",
        ) as mock_open_permit, patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.remove_remote_host_control_permit",
        ) as mock_close_permit, patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.ensure_remote_host_boot_pacing_policy",
            return_value=_BOOT_PACING_STATUS,
        ), patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.apply_remote_host_stabilization",
            side_effect=RuntimeError("boom"),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                stabilize_remote_chonkydb_host(
                    settings=_SETTINGS,
                    probe_timeout_s=20.0,
                    ssh_timeout_s=60.0,
                    settle_s=1.0,
                    executor=object(),  # type: ignore[arg-type]
                )

        self.assertEqual(mock_open_permit.call_count, 1)
        self.assertEqual(mock_close_permit.call_count, 1)

    def test_stabilize_remote_host_keeps_guard_required_units_in_system_lane(self) -> None:
        system_before_state = (
            RemoteHostUnitState(
                scope="system",
                unit="caia-consumer-portal.service",
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
                unit="caia-consumer-portal.service",
                enabled_state="disabled",
                load_state="loaded",
                active_state="inactive",
                sub_state="dead",
                result="success",
            ),
        )
        with patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.ensure_remote_host_control_permit",
        ), patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.remove_remote_host_control_permit",
        ), patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.ensure_remote_host_boot_pacing_policy",
            return_value=_BOOT_PACING_STATUS,
        ), patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.fetch_remote_guard_protected_system_units",
            return_value=(
                "caia-consumer-portal.service",
                "caia-code-graph-refresh.timer",
            ),
        ), patch(
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
                (),
                system_after_state,
                (),
            ],
        ) as mock_states, patch(
            "twinr.ops.remote_chonkydb_host_stabilizer.apply_remote_host_stabilization",
            return_value=RemoteHostStabilizationAction(
                kill_switch_paths=("/tmp/kill_a",),
                disabled_system_units=("caia-consumer-portal.service",),
                disabled_user_units=(),
                terminated_processes=(),
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
        filtered_units = mock_apply.call_args.kwargs["system_units"]
        self.assertIn("caia-consumer-portal.service", filtered_units)
        self.assertIn("caia-code-graph-refresh.timer", filtered_units)
        self.assertIn(_HOST_CONTROL_GUARD_UNIT, filtered_units)
        self.assertIn(
            "caia-consumer-portal.service",
            mock_states.call_args_list[0].kwargs["units"],
        )


if __name__ == "__main__":
    unittest.main()
