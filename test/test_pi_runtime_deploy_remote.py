from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.ops.pi_runtime_deploy_remote import (
    _parse_remote_retention_canary_stdout,
    _summarize_retention_canary_failure_payload,
    run_retention_canary_probe,
)


class PiRuntimeDeployRemoteTests(unittest.TestCase):
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
            def __init__(self) -> None:
                self.calls: list[str] = []
                self.command_calls = 0

            def run_ssh(self, script: str):
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

        remote = _FakeRemote()
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
            def __init__(self) -> None:
                self.calls: list[str] = []

            def run_ssh(self, script: str):
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
                remote=_FakeRemote(),
                remote_root="/twinr",
                env_path="/twinr/.env",
                probe_id=probe_id,
            )


if __name__ == "__main__":
    unittest.main()
