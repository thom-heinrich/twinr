from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, cast
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.ops.pi_runtime_deploy_remote import (
    _OPS_ARTIFACT_PERMISSION_SPECS,
    _parse_remote_retention_canary_stdout,
    _summarize_retention_canary_failure_payload,
    RetentionCanaryProbeError,
    run_retention_canary_probe,
    wait_for_remote_watchdog_ready,
)


class PiRuntimeDeployRemoteTests(unittest.TestCase):
    def test_ops_artifact_permissions_include_usage_sqlite_sidecars(self) -> None:
        self.assertIn(("usage.jsonl.sqlite3", "600"), _OPS_ARTIFACT_PERMISSION_SPECS)
        self.assertIn(("usage.jsonl.sqlite3.lock", "600"), _OPS_ARTIFACT_PERMISSION_SPECS)

    def test_summarize_retention_canary_failure_payload_prefers_scope_and_stage_details(self) -> None:
        summary = _summarize_retention_canary_failure_payload(
            {
                "status": "failed",
                "failure_stage": "fresh_reader_load_current_state_fine_grained",
                "error_message": "LongTermRemoteUnavailableError: Failed to read remote long-term 'objects' item 'event:retention_future_appointment'.",
                "consistency_assessment": {
                    "relation": "watchdog_ready_canary_failed_non_equivalent",
                    "summary": "The watchdog was green because configured-namespace archive-inclusive warm reads were healthy, but the retention canary failed later on the stricter isolated-namespace write/retention/readback path.",
                },
                "watchdog_observations": [
                    {
                        "sample_status": "ok",
                        "sample_ready": True,
                        "sample_detail": None,
                    }
                ],
            }
        )

        self.assertIn("failure_stage=fresh_reader_load_current_state_fine_grained", summary)
        self.assertIn("relation=watchdog_ready_canary_failed_non_equivalent", summary)
        self.assertIn("watchdog_status=ok", summary)
        self.assertIn("watchdog_ready=true", summary)
        self.assertIn("event:retention_future_appointment", summary)

    def test_parse_remote_retention_canary_stdout_extracts_exit_status_marker(self) -> None:
        exit_status, raw_output = _parse_remote_retention_canary_stdout(
            "__TWINR_RETENTION_EXIT_STATUS__=137\n{\n  \"ready\": false\n}\n"
        )

        self.assertEqual(exit_status, 137)
        self.assertEqual(raw_output, '{\n  "ready": false\n}')

    def test_run_retention_canary_probe_recovers_final_pi_report_after_stdout_timeout(self) -> None:
        probe_id = "deploy_retention_canary_timeout_probe"
        final_payload = {
            "probe_id": probe_id,
            "status": "failed",
            "ready": False,
            "failure_stage": "run_retention",
            "error_message": (
                "LongTermRemoteUnavailableError: Failed to read remote long-term "
                "'objects' item 'episode:retention_old_weather'."
            ),
            "consistency_assessment": {
                "relation": "watchdog_ready_canary_failed_non_equivalent",
                "summary": (
                    "The watchdog was green because configured-namespace archive-inclusive warm reads "
                    "were healthy, but the retention canary failed later on the stricter isolated-namespace "
                    "write/retention/readback path."
                ),
            },
            "watchdog_observations": [
                {
                    "sample_status": "ok",
                    "sample_ready": True,
                }
            ],
        }

        class _FakeRemote:
            timeout_s = 180.0

            def __init__(self) -> None:
                self.calls: list[str] = []
                self.command_calls = 0

            def run_ssh(self, script: str, *, timeout_s: float | None = None):
                del timeout_s
                self.calls.append(script)
                if "twinr.memory.longterm.evaluation.live_retention_canary" in script:
                    self.command_calls += 1
                    raise subprocess.TimeoutExpired(cmd="ssh retention canary", timeout=180)
                return subprocess.CompletedProcess(
                    args=["ssh"],
                    returncode=0,
                    stdout=json.dumps(final_payload),
                    stderr="",
                )

        remote = cast(Any, _FakeRemote())
        with self.assertRaisesRegex(
            RuntimeError,
            "timed out waiting for stdout but completed on the Pi: .*failure_stage=run_retention",
        ):
            run_retention_canary_probe(
                remote=remote,
                remote_root="/twinr",
                env_path="/twinr/.env",
                probe_id=probe_id,
            )
        self.assertEqual(remote.command_calls, 1)
        self.assertTrue(
            any(f"/retention_live_canary/{probe_id}.json" in script for script in remote.calls),
            remote.calls,
        )

    def test_run_retention_canary_probe_reports_exit_status_when_remote_process_dies_without_json(self) -> None:
        probe_id = "deploy_retention_canary_exit_status_probe"

        class _FakeRemote:
            timeout_s = 180.0

            def __init__(self) -> None:
                self.calls: list[str] = []

            def run_ssh(self, script: str, *, timeout_s: float | None = None):
                del timeout_s
                self.calls.append(script)
                if "twinr.memory.longterm.evaluation.live_retention_canary" in script:
                    return subprocess.CompletedProcess(
                        args=["ssh"],
                        returncode=0,
                        stdout="__TWINR_RETENTION_EXIT_STATUS__=137\n",
                        stderr="",
                    )
                return subprocess.CompletedProcess(
                    args=["ssh"],
                    returncode=0,
                    stdout="{}\n",
                    stderr="",
                )

        with self.assertRaisesRegex(
            RuntimeError,
            "retention canary exited without JSON output: .*exit_status=137.*no stdout/stderr payload",
        ):
            run_retention_canary_probe(
                remote=cast(Any, _FakeRemote()),
                remote_root="/twinr",
                env_path="/twinr/.env",
                probe_id=probe_id,
            )

    def test_run_retention_canary_probe_uses_explicit_command_timeout(self) -> None:
        timeouts: list[float | None] = []

        class _FakeRemote:
            timeout_s = 180.0

            def run_ssh(self, script: str, *, timeout_s: float | None = None):
                timeouts.append(timeout_s)
                if "twinr.memory.longterm.evaluation.live_retention_canary" in script:
                    return subprocess.CompletedProcess(
                        args=["ssh"],
                        returncode=0,
                        stdout='{"status": "ok", "ready": true, "probe_id": "p"}\n',
                        stderr="",
                    )
                return subprocess.CompletedProcess(
                    args=["ssh"],
                    returncode=0,
                    stdout="{}\n",
                    stderr="",
                )

        payload = run_retention_canary_probe(
            remote=cast(Any, _FakeRemote()),
            remote_root="/twinr",
            env_path="/twinr/.env",
            probe_id="p",
            command_timeout_s=900.0,
        )

        self.assertTrue(payload["ready"])
        self.assertEqual(timeouts, [900.0])

    def test_run_retention_canary_probe_executes_from_remote_root_with_runtime_pythonpath(self) -> None:
        scripts: list[str] = []

        class _FakeRemote:
            timeout_s = 180.0

            def run_ssh(self, script: str, *, timeout_s: float | None = None):
                del timeout_s
                scripts.append(script)
                return subprocess.CompletedProcess(
                    args=["ssh"],
                    returncode=0,
                    stdout='{"status": "ok", "ready": true, "probe_id": "p"}\n',
                    stderr="",
                )

        payload = run_retention_canary_probe(
            remote=cast(Any, _FakeRemote()),
            remote_root="/twinr",
            env_path=".env",
            probe_id="p",
        )

        self.assertTrue(payload["ready"])
        self.assertEqual(len(scripts), 1)
        self.assertIn("cd /twinr", scripts[0])
        self.assertIn('export PYTHONPATH=/twinr/src:${PYTHONPATH:-""}', scripts[0])

    def test_run_retention_canary_probe_attaches_failed_payload_to_exception(self) -> None:
        failed_payload = {
            "status": "failed",
            "ready": False,
            "failure_stage": "seed_retention_objects",
            "error_message": "LongTermRemoteUnavailableError: Accepted remote long-term 'objects' write could not be read back.",
            "remote_write_context": {
                "operation": "store_records_bulk",
                "request_path": "/v1/external/records/bulk",
            },
        }

        class _FakeRemote:
            timeout_s = 180.0

            def run_ssh(self, script: str, *, timeout_s: float | None = None):
                del script, timeout_s
                return subprocess.CompletedProcess(
                    args=["ssh"],
                    returncode=0,
                    stdout=json.dumps(failed_payload) + "\n",
                    stderr="",
                )

        with self.assertRaises(RetentionCanaryProbeError) as exc_info:
            run_retention_canary_probe(
                remote=cast(Any, _FakeRemote()),
                remote_root="/twinr",
                env_path="/twinr/.env",
                probe_id="failed_payload_probe",
            )

        self.assertEqual(exc_info.exception.payload, failed_payload)

    def test_run_retention_canary_probe_recovers_failed_pi_report_when_inline_stdout_is_malformed(self) -> None:
        probe_id = "deploy_retention_canary_malformed_stdout_probe"
        failed_payload = {
            "probe_id": probe_id,
            "status": "failed",
            "ready": False,
            "failure_stage": "writer_ensure_remote_ready",
            "error_message": "LongTermRemoteUnavailableError: ChonkyDB health check failed (ChonkyDBError).",
        }

        class _FakeRemote:
            timeout_s = 180.0

            def __init__(self) -> None:
                self.calls: list[str] = []

            def run_ssh(self, script: str, *, timeout_s: float | None = None):
                del timeout_s
                self.calls.append(script)
                if "twinr.memory.longterm.evaluation.live_retention_canary" in script:
                    return subprocess.CompletedProcess(
                        args=["ssh"],
                        returncode=0,
                        stdout="__TWINR_RETENTION_EXIT_STATUS__=1\nWARNING: noisy preface before JSON\n",
                        stderr="",
                    )
                return subprocess.CompletedProcess(
                    args=["ssh"],
                    returncode=0,
                    stdout=json.dumps(failed_payload) + "\n",
                    stderr="",
                )

        remote = cast(Any, _FakeRemote())
        with self.assertRaisesRegex(
            RetentionCanaryProbeError,
            "retention canary emitted non-JSON inline output but produced a Pi report: .*writer_ensure_remote_ready",
        ) as exc_info:
            run_retention_canary_probe(
                remote=remote,
                remote_root="/twinr",
                env_path="/twinr/.env",
                probe_id=probe_id,
            )

        self.assertEqual(exc_info.exception.payload, failed_payload)
        self.assertTrue(
            any(f"/retention_live_canary/{probe_id}.json" in script for script in remote.calls),
            remote.calls,
        )

    def test_wait_for_remote_watchdog_ready_uses_freshness_gate_and_timeout_budget(self) -> None:
        payload = {
            "ready": True,
            "detail": "watchdog_ready",
            "sample_captured_at": "2026-04-05T13:48:42Z",
            "sample_fresh_after_gate": True,
        }
        timeouts: list[float | None] = []

        class _FakeRemote:
            timeout_s = 180.0

            def __init__(self) -> None:
                self.scripts: list[str] = []

            def run_ssh(self, script: str, *, timeout_s: float | None = None):
                self.scripts.append(script)
                timeouts.append(timeout_s)
                return subprocess.CompletedProcess(
                    args=["ssh"],
                    returncode=0,
                    stdout=json.dumps(payload) + "\n",
                    stderr="",
                )

        remote = cast(Any, _FakeRemote())
        result = wait_for_remote_watchdog_ready(
            remote=remote,
            remote_root="/twinr",
            env_path="/twinr/.env",
            min_sample_captured_at="2026-04-05T13:46:10Z",
            wait_timeout_s=90.0,
            poll_interval_s=7.0,
        )

        self.assertEqual(result, payload)
        self.assertEqual(timeouts, [120.0])
        self.assertTrue(any("min_sample_captured_at = '2026-04-05T13:46:10Z'" in script for script in remote.scripts))


if __name__ == "__main__":
    unittest.main()
