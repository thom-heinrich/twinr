"""Regression tests for remote systemd manual-restart protection."""

from __future__ import annotations

import subprocess
import unittest
from typing import Any, cast
from unittest.mock import Mock, patch

from twinr.ops.remote_chonkydb_repair import (
    BackendQuerySurfaceReadinessContract,
    ChonkyDBHttpProbeResult,
    ChonkyDBRemoteServiceState,
    RemoteChonkyDBOpsSettings,
    RemoteChonkyDBRepairPlan,
    repair_remote_chonkydb,
)
from twinr.ops.remote_systemd_restart_guard import (
    RemoteManualRestartProtectionStatus,
    ensure_remote_service_manual_restart_protection,
    guarded_restart_remote_service,
)
from twinr.ops.self_coding_pi import PiConnectionSettings


def _completed(stdout: str = "") -> subprocess.CompletedProcess[str]:
    """Return one successful fake SSH command result."""

    return subprocess.CompletedProcess(
        args=["ssh"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )


def _failed_completed(
    *,
    returncode: int = 1,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    """Return one failed fake SSH command result."""

    return subprocess.CompletedProcess(
        args=["ssh"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


class RemoteSystemdRestartGuardTests(unittest.TestCase):
    """Cover the remote manual-restart protection helper."""

    def test_ensure_manual_restart_protection_verifies_systemd_flags(self) -> None:
        executor = Mock()
        executor.run_sudo_ssh.side_effect = [
            _completed('{"changed": false, "removed_paths": []}'),
            _completed(
                '{"changed": true, "path": '
                '"/etc/systemd/system/caia-twinr-chonkydb-alt.service.d/10-refuse-manual-restart.conf"}'
            ),
            _completed("RefuseManualStart=yes\nRefuseManualStop=yes\n"),
        ]

        status = ensure_remote_service_manual_restart_protection(
            executor=executor,
            service_name="caia-twinr-chonkydb-alt.service",
        )

        self.assertTrue(status.protection_changed)
        self.assertTrue(status.verified)
        scripts = [call.args[0] for call in executor.run_sudo_ssh.call_args_list]
        self.assertIn("/run/systemd/system", scripts[0])
        self.assertIn("RefuseManualStart=no", scripts[0])
        self.assertIn("10-refuse-manual-restart.conf", scripts[1])
        self.assertIn("RefuseManualStart=yes", scripts[1])
        self.assertIn("systemctl show", scripts[2])
        self.assertIn("caia-twinr-chonkydb-alt.service", scripts[2])
        self.assertIn("RefuseManualStart", scripts[2])
        self.assertIn("RefuseManualStop", scripts[2])

    def test_ensure_manual_restart_protection_fails_when_flags_remain_open(self) -> None:
        executor = Mock()
        executor.run_sudo_ssh.side_effect = [
            _completed('{"changed": false, "removed_paths": []}'),
            _completed('{"changed": false, "path": "/tmp/unused.conf"}'),
            _completed("RefuseManualStart=no\nRefuseManualStop=yes\n"),
        ]

        with self.assertRaisesRegex(
            RuntimeError,
            "remote_manual_restart_protection_not_verified",
        ):
            ensure_remote_service_manual_restart_protection(
                executor=executor,
                service_name="caia-twinr-chonkydb-alt.service",
            )

    def test_ensure_manual_restart_protection_removes_stale_allow_manual_override(self) -> None:
        executor = Mock()
        executor.run_sudo_ssh.side_effect = [
            _completed(
                '{"changed": true, "removed_paths": '
                '["/run/systemd/system/caia-twinr-chonkydb-alt.service.d/99-codex-manual-start.conf"]}'
            ),
            _completed('{"changed": false, "path": "/tmp/unused.conf"}'),
            _completed("RefuseManualStart=yes\nRefuseManualStop=yes\n"),
        ]

        status = ensure_remote_service_manual_restart_protection(
            executor=executor,
            service_name="caia-twinr-chonkydb-alt.service",
        )

        self.assertTrue(status.protection_changed)
        self.assertTrue(status.verified)
        scripts = [call.args[0] for call in executor.run_sudo_ssh.call_args_list]
        self.assertIn("/run/systemd/system", scripts[0])
        self.assertIn("RefuseManualStart=no", scripts[0])
        self.assertIn("daemon-reload", scripts[0])
        self.assertIn("systemctl show", scripts[2])

    def test_guarded_restart_uses_bounded_stop_kill_start_sequence(self) -> None:
        executor = Mock()
        executor.run_sudo_ssh.side_effect = [
            _completed(
                '{"changed": true, "path": '
                '"/run/caia/maintenance/twinr_host_control.permit"}'
            ),
            _completed(
                '{"changed": true, "path": '
                '"/etc/systemd/system/caia-twinr-chonkydb-alt.service.d/'
                '10-refuse-manual-restart.conf"}'
            ),
            _completed("RefuseManualStart=no\nRefuseManualStop=no\n"),
            _completed('{"ok": true, "phase": "start", "actions": ["stop", "kill_all_sigkill", "start"]}'),
            _completed(
                '{"changed": true, "path": '
                '"/etc/systemd/system/caia-twinr-chonkydb-alt.service.d/10-refuse-manual-restart.conf"}'
            ),
            _completed("RefuseManualStart=yes\nRefuseManualStop=yes\n"),
            _completed(
                '{"changed": true, "path": '
                '"/run/caia/maintenance/twinr_host_control.permit"}'
            ),
        ]

        guarded_restart_remote_service(
            executor=executor,
            service_name="caia-twinr-chonkydb-alt.service",
        )

        scripts = [call.args[0] for call in executor.run_sudo_ssh.call_args_list]
        self.assertIn("/run/caia/maintenance/twinr_host_control.permit", scripts[0])
        self.assertIn("ALLOW_TWINR_HOST_CONTROL", scripts[0])
        self.assertIn("10-refuse-manual-restart.conf", scripts[1])
        self.assertIn("path.unlink()", scripts[1])
        self.assertIn("systemctl show", scripts[2])
        self.assertIn("RefuseManualStart", scripts[2])
        self.assertIn("'systemctl', 'stop', service_name", scripts[3])
        self.assertIn("'systemctl', 'kill', '--kill-who=all', '--signal=SIGKILL', service_name", scripts[3])
        self.assertIn("'systemctl', 'start', service_name", scripts[3])
        self.assertIn("reset-failed", scripts[3])
        self.assertIn("caia-twinr-chonkydb-alt.service", scripts[3])
        self.assertIn("10-refuse-manual-restart.conf", scripts[4])
        self.assertIn("RefuseManualStart=yes", scripts[4])
        self.assertIn("systemctl show", scripts[5])
        self.assertIn("/run/caia/maintenance/twinr_host_control.permit", scripts[6])

    def test_guarded_restart_raises_on_nonzero_remote_command_and_still_removes_override(self) -> None:
        executor = Mock()
        executor.run_sudo_ssh.side_effect = [
            _completed(
                '{"changed": true, "path": '
                '"/run/caia/maintenance/twinr_host_control.permit"}'
            ),
            _completed(
                '{"changed": true, "path": '
                '"/etc/systemd/system/caia-twinr-chonkydb-alt.service.d/'
                '10-refuse-manual-restart.conf"}'
            ),
            _completed("RefuseManualStart=no\nRefuseManualStop=no\n"),
            _failed_completed(
                stdout='{"ok": false, "phase": "kill_timeout"}',
                stderr="systemctl stop timed out",
            ),
            _completed(
                '{"changed": true, "path": '
                '"/etc/systemd/system/caia-twinr-chonkydb-alt.service.d/10-refuse-manual-restart.conf"}'
            ),
            _completed("RefuseManualStart=yes\nRefuseManualStop=yes\n"),
            _completed(
                '{"changed": true, "path": '
                '"/run/caia/maintenance/twinr_host_control.permit"}'
            ),
        ]

        with self.assertRaisesRegex(RuntimeError, "remote_guarded_restart_failed"):
            guarded_restart_remote_service(
                executor=executor,
                service_name="caia-twinr-chonkydb-alt.service",
            )

        scripts = [call.args[0] for call in executor.run_sudo_ssh.call_args_list]
        self.assertIn("/run/caia/maintenance/twinr_host_control.permit", scripts[0])
        self.assertIn("10-refuse-manual-restart.conf", scripts[1])
        self.assertIn("systemctl show", scripts[2])
        self.assertIn("'systemctl', 'stop', service_name", scripts[3])
        self.assertIn("10-refuse-manual-restart.conf", scripts[4])
        self.assertIn("systemctl show", scripts[5])
        self.assertIn("/run/caia/maintenance/twinr_host_control.permit", scripts[6])

    def test_guarded_restart_fails_fast_when_manual_restart_flags_stay_closed(self) -> None:
        executor = Mock()
        executor.run_sudo_ssh.side_effect = [
            _completed(
                '{"changed": true, "path": '
                '"/run/caia/maintenance/twinr_host_control.permit"}'
            ),
            _completed(
                '{"changed": true, "path": '
                '"/etc/systemd/system/caia-twinr-chonkydb-alt.service.d/'
                '10-refuse-manual-restart.conf"}'
            ),
            _completed("RefuseManualStart=yes\nRefuseManualStop=yes\n"),
            _completed(
                '{"changed": true, "path": '
                '"/etc/systemd/system/caia-twinr-chonkydb-alt.service.d/10-refuse-manual-restart.conf"}'
            ),
            _completed("RefuseManualStart=yes\nRefuseManualStop=yes\n"),
            _completed(
                '{"changed": true, "path": '
                '"/run/caia/maintenance/twinr_host_control.permit"}'
            ),
        ]

        with self.assertRaisesRegex(
            RuntimeError,
            "remote_manual_restart_override_not_verified",
        ):
            guarded_restart_remote_service(
                executor=executor,
                service_name="caia-twinr-chonkydb-alt.service",
            )

        scripts = [call.args[0] for call in executor.run_sudo_ssh.call_args_list]
        self.assertIn("/run/caia/maintenance/twinr_host_control.permit", scripts[0])
        self.assertIn("10-refuse-manual-restart.conf", scripts[1])
        self.assertIn("systemctl show", scripts[2])
        self.assertIn("10-refuse-manual-restart.conf", scripts[3])
        self.assertIn("systemctl show", scripts[4])
        self.assertIn("/run/caia/maintenance/twinr_host_control.permit", scripts[5])


class RemoteChonkyDBRepairProtectionIntegrationTests(unittest.TestCase):
    """Verify the repair flow carries the restart-protection status through."""

    def test_repair_flow_records_restart_protection_status(self) -> None:
        settings = RemoteChonkyDBOpsSettings(
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

        with (
            patch(
                "twinr.ops.remote_chonkydb_repair.ensure_remote_service_manual_restart_protection",
                return_value=RemoteManualRestartProtectionStatus(
                    service_name="caia-twinr-chonkydb-alt.service",
                    protected_dropin_path=(
                        "/etc/systemd/system/caia-twinr-chonkydb-alt.service.d/"
                        "10-refuse-manual-restart.conf"
                    ),
                    protection_changed=True,
                    refuse_manual_start=True,
                    refuse_manual_stop=True,
                ),
            ) as mock_protection,
            patch(
                "twinr.ops.remote_chonkydb_repair.probe_public_chonkydb",
                return_value=ChonkyDBHttpProbeResult(
                    label="public",
                    ok=True,
                    status_code=200,
                    ready=True,
                    detail="ccodex_memory",
                ),
            ),
            patch(
                "twinr.ops.remote_chonkydb_repair.fetch_backend_service_state",
                return_value=ChonkyDBRemoteServiceState(
                    active_state="active",
                    sub_state="running",
                    service_result="success",
                ),
            ),
            patch(
                "twinr.ops.remote_chonkydb_repair.probe_backend_local_chonkydb",
                return_value=ChonkyDBHttpProbeResult(
                    label="backend",
                    ok=True,
                    status_code=200,
                    ready=True,
                    detail="ccodex_memory",
                ),
            ),
            patch(
                "twinr.ops.remote_chonkydb_repair.plan_remote_chonkydb_repair",
                return_value=RemoteChonkyDBRepairPlan(
                    action="none",
                    reason="public_ready",
                ),
            ),
            patch(
                "twinr.ops.remote_chonkydb_repair.fetch_backend_query_surface_readiness_contract",
                return_value=BackendQuerySurfaceReadinessContract(
                    fulltext_rebuild_on_open=False,
                    warmup_fulltext_gate=False,
                    warmup_wait_for_ready=False,
                ),
            ),
            patch(
                "twinr.ops.remote_chonkydb_repair.inspect_backend_data_ownership",
                return_value=None,
            ),
            patch(
                "twinr.ops.remote_chonkydb_repair.inspect_foreign_backend_consumers",
                return_value=(),
            ),
        ):
            result = repair_remote_chonkydb(
                settings=settings,
                probe_timeout_s=2.0,
                ssh_timeout_s=2.0,
                wait_ready_s=2.0,
                poll_interval_s=0.1,
                restart_if_needed=False,
            )

        mock_protection.assert_called_once()
        self.assertTrue(result.ok)
        self.assertTrue(result.restart_protection.verified)
        payload = cast(dict[str, Any], result.to_dict())
        restart_protection = cast(
            dict[str, Any],
            payload["restart_protection"],
        )
        self.assertTrue(bool(restart_protection["protection_changed"]))


if __name__ == "__main__":
    unittest.main()
