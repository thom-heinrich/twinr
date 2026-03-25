from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.ops.pi_repo_mirror import PiRepoMirrorCycleResult
from twinr.ops.pi_runtime_deploy import deploy_pi_runtime


def _completed(
    args: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr=stderr)


class _FakeMirrorWatchdog:
    def __init__(self) -> None:
        self.last_call: dict[str, object] = {}

    def probe_once(
        self,
        *,
        apply_sync: bool = True,
        checksum: bool = True,
        max_change_lines: int = 40,
    ) -> PiRepoMirrorCycleResult:
        self.last_call = {
            "apply_sync": apply_sync,
            "checksum": checksum,
            "max_change_lines": max_change_lines,
        }
        return PiRepoMirrorCycleResult(
            host="192.168.1.95",
            remote_root="/twinr",
            drift_detected=True,
            sync_applied=True,
            checksum_used=True,
            verified_clean=True,
            change_count=1,
            sampled_change_lines=(">f+++++++++ src/twinr/ops/pi_runtime_deploy.py",),
            duration_s=0.42,
        )


class PiRuntimeDeployTests(unittest.TestCase):
    def test_deploy_runs_env_sync_install_restart_and_verification(self) -> None:
        commands: list[list[str]] = []
        envs: list[dict[str, str] | None] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        'PI_HOST="192.168.1.95"',
                        'PI_SSH_USER="thh"',
                        'PI_SSH_PW="chaos"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            local_sha = hashlib.sha256(env_path.read_bytes()).hexdigest()
            mirror = _FakeMirrorWatchdog()

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                envs.append(kwargs.get("env"))
                rendered = " ".join(command)
                if "scp" in command:
                    return _completed(command)
                if "sha256sum /twinr/.env" in rendered:
                    return _completed(command, stdout="")
                if "pip install" in rendered:
                    return _completed(command, stdout="Successfully installed twinr\n")
                if "ActiveState,SubState,UnitFileState,MainPID,ExecMainStatus" in rendered:
                    return _completed(
                        command,
                        stdout=(
                            '[{"name": "twinr-remote-memory-watchdog.service", "active_state": "active",'
                            ' "sub_state": "running", "unit_file_state": "enabled", "main_pid": "101",'
                            ' "exec_main_status": "0"},'
                            ' {"name": "twinr-runtime-supervisor.service", "active_state": "active",'
                            ' "sub_state": "running", "unit_file_state": "enabled", "main_pid": "102",'
                            ' "exec_main_status": "0"},'
                            ' {"name": "twinr-web.service", "active_state": "active",'
                            ' "sub_state": "running", "unit_file_state": "enabled", "main_pid": "103",'
                            ' "exec_main_status": "0"}]\n'
                        ),
                    )
                if "check_pi_openai_env_contract.py" in rendered:
                    return _completed(command, stdout='{"ok": true, "detail": "ready"}\n')
                if "actual_sha=$(sha256sum" in rendered:
                    return _completed(command, stdout="/twinr/.env.deploy-backup-20260324T000000Z")
                return _completed(command)

            result = deploy_pi_runtime(
                project_root=root,
                pi_env_path=pi_env_path,
                subprocess_runner=_runner,
                mirror_watchdog=mirror,
                live_text="Antworte nur mit: ok.",
            )

        self.assertTrue(result.ok)
        self.assertTrue(result.repo_mirror.sync_applied)
        self.assertTrue(result.env_sync is not None)
        assert result.env_sync is not None
        self.assertEqual(result.env_sync.sha256, local_sha)
        self.assertTrue(result.env_sync.changed)
        self.assertTrue(all(state.healthy for state in result.service_states))
        self.assertEqual(result.env_contract, {"ok": True, "detail": "ready"})
        self.assertEqual(mirror.last_call["apply_sync"], True)
        self.assertEqual(mirror.last_call["checksum"], True)
        joined = "\n".join(" ".join(command) for command in commands)
        self.assertIn("sshpass -e scp", joined)
        self.assertIn("pip install --no-deps -e", joined)
        self.assertIn("systemctl daemon-reload", joined)
        self.assertIn("systemctl restart", joined)
        self.assertIn("--live-text", joined)
        self.assertTrue(any(env and env.get("SSHPASS") == "chaos" for env in envs))

    def test_deploy_skips_env_copy_when_remote_checksum_matches(self) -> None:
        commands: list[list[str]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        'PI_HOST="192.168.1.95"',
                        'PI_SSH_USER="thh"',
                        'PI_SSH_PW="chaos"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            local_sha = hashlib.sha256(env_path.read_bytes()).hexdigest()
            mirror = _FakeMirrorWatchdog()

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                rendered = " ".join(command)
                if "sha256sum /twinr/.env" in rendered:
                    return _completed(command, stdout=f"{local_sha}\n")
                if "ActiveState,SubState,UnitFileState,MainPID,ExecMainStatus" in rendered:
                    return _completed(
                        command,
                        stdout='[{"name": "twinr-runtime-supervisor.service", "active_state": "active", "sub_state": "running", "unit_file_state": "enabled", "main_pid": "102", "exec_main_status": "0"}]\n',
                    )
                if "check_pi_openai_env_contract.py" in rendered:
                    return _completed(command, stdout='{"ok": true}\n')
                return _completed(command)

            result = deploy_pi_runtime(
                project_root=root,
                pi_env_path=pi_env_path,
                services=("twinr-runtime-supervisor",),
                subprocess_runner=_runner,
                mirror_watchdog=mirror,
                install_editable=False,
                install_systemd_units=False,
            )

        assert result.env_sync is not None
        self.assertFalse(result.env_sync.changed)
        joined = "\n".join(" ".join(command) for command in commands)
        self.assertNotIn("sshpass -e scp", joined)

    def test_default_deploy_includes_enabled_optional_pi_services(self) -> None:
        commands: list[list[str]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        'PI_HOST="192.168.1.95"',
                        'PI_SSH_USER="thh"',
                        'PI_SSH_PW="chaos"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            ops_root = root / "hardware" / "ops"
            ops_root.mkdir(parents=True, exist_ok=True)
            (ops_root / "twinr-runtime-supervisor.service").write_text(
                "[Service]\nWorkingDirectory=/twinr\nExecStart=/usr/bin/env bash -lc 'cd /twinr && run'\n",
                encoding="utf-8",
            )
            (ops_root / "twinr-remote-memory-watchdog.service").write_text(
                "[Service]\nWorkingDirectory=/twinr\nExecStart=/usr/bin/env bash -lc 'cd /twinr && run'\n",
                encoding="utf-8",
            )
            (ops_root / "twinr-web.service").write_text(
                "[Service]\nWorkingDirectory=/twinr\nExecStart=/usr/bin/env bash -lc 'cd /twinr && run'\n",
                encoding="utf-8",
            )
            (ops_root / "twinr-whatsapp-channel.service").write_text(
                "[Service]\nWorkingDirectory=/twinr\nExecStart=/twinr/.venv/bin/python -m twinr --env-file .env --run-whatsapp-channel\n",
                encoding="utf-8",
            )
            (ops_root / "twinr-orchestrator-server.service").write_text(
                "[Service]\nWorkingDirectory=/home/thh/twinr\nExecStart=/home/thh/twinr/.venv/bin/python -m twinr --run-orchestrator-server\n",
                encoding="utf-8",
            )
            mirror = _FakeMirrorWatchdog()

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                rendered = " ".join(command)
                if "sha256sum /twinr/.env" in rendered:
                    return _completed(command, stdout="")
                if "UnitFileState" in rendered and "ActiveState,SubState" not in rendered:
                    return _completed(
                        command,
                        stdout=(
                            '[{"name": "twinr-remote-memory-watchdog.service", "unit_file_state": "enabled"},'
                            ' {"name": "twinr-runtime-supervisor.service", "unit_file_state": "enabled"},'
                            ' {"name": "twinr-web.service", "unit_file_state": "enabled"},'
                            ' {"name": "twinr-whatsapp-channel.service", "unit_file_state": "enabled"}]\n'
                        ),
                    )
                if "ActiveState,SubState,UnitFileState,MainPID,ExecMainStatus" in rendered:
                    return _completed(
                        command,
                        stdout=(
                            '[{"name": "twinr-remote-memory-watchdog.service", "active_state": "active",'
                            ' "sub_state": "running", "unit_file_state": "enabled", "main_pid": "101",'
                            ' "exec_main_status": "0"},'
                            ' {"name": "twinr-runtime-supervisor.service", "active_state": "active",'
                            ' "sub_state": "running", "unit_file_state": "enabled", "main_pid": "102",'
                            ' "exec_main_status": "0"},'
                            ' {"name": "twinr-web.service", "active_state": "active",'
                            ' "sub_state": "running", "unit_file_state": "enabled", "main_pid": "103",'
                            ' "exec_main_status": "0"},'
                            ' {"name": "twinr-whatsapp-channel.service", "active_state": "active",'
                            ' "sub_state": "running", "unit_file_state": "enabled", "main_pid": "104",'
                            ' "exec_main_status": "0"}]\n'
                        ),
                    )
                if "pip install" in rendered:
                    return _completed(command, stdout="Successfully installed twinr\n")
                if "check_pi_openai_env_contract.py" in rendered:
                    return _completed(command, stdout='{"ok": true}\n')
                if "actual_sha=$(sha256sum" in rendered:
                    return _completed(command, stdout="")
                return _completed(command)

            result = deploy_pi_runtime(
                project_root=root,
                pi_env_path=pi_env_path,
                subprocess_runner=_runner,
                mirror_watchdog=mirror,
            )

        self.assertEqual(
            result.restarted_services,
            (
                "twinr-remote-memory-watchdog.service",
                "twinr-runtime-supervisor.service",
                "twinr-web.service",
                "twinr-whatsapp-channel.service",
            ),
        )
        joined = "\n".join(" ".join(command) for command in commands)
        self.assertIn("twinr-whatsapp-channel.service", joined)
        self.assertNotIn("twinr-orchestrator-server.service", joined)

    def test_rollout_service_adds_disabled_optional_pi_unit_to_default_set(self) -> None:
        commands: list[list[str]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        'PI_HOST="192.168.1.95"',
                        'PI_SSH_USER="thh"',
                        'PI_SSH_PW="chaos"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            ops_root = root / "hardware" / "ops"
            ops_root.mkdir(parents=True, exist_ok=True)
            (ops_root / "twinr-runtime-supervisor.service").write_text(
                "[Service]\nWorkingDirectory=/twinr\nExecStart=/usr/bin/env bash -lc 'cd /twinr && run'\n",
                encoding="utf-8",
            )
            (ops_root / "twinr-remote-memory-watchdog.service").write_text(
                "[Service]\nWorkingDirectory=/twinr\nExecStart=/usr/bin/env bash -lc 'cd /twinr && run'\n",
                encoding="utf-8",
            )
            (ops_root / "twinr-web.service").write_text(
                "[Service]\nWorkingDirectory=/twinr\nExecStart=/usr/bin/env bash -lc 'cd /twinr && run'\n",
                encoding="utf-8",
            )
            (ops_root / "twinr-whatsapp-channel.service").write_text(
                "[Service]\nWorkingDirectory=/twinr\nExecStart=/twinr/.venv/bin/python -m twinr --env-file .env --run-whatsapp-channel\n",
                encoding="utf-8",
            )
            mirror = _FakeMirrorWatchdog()

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                rendered = " ".join(command)
                if "sha256sum /twinr/.env" in rendered:
                    return _completed(command, stdout="")
                if "UnitFileState" in rendered and "ActiveState,SubState" not in rendered:
                    return _completed(
                        command,
                        stdout=(
                            '[{"name": "twinr-remote-memory-watchdog.service", "unit_file_state": "enabled"},'
                            ' {"name": "twinr-runtime-supervisor.service", "unit_file_state": "enabled"},'
                            ' {"name": "twinr-web.service", "unit_file_state": "enabled"},'
                            ' {"name": "twinr-whatsapp-channel.service", "unit_file_state": "disabled"}]\n'
                        ),
                    )
                if "ActiveState,SubState,UnitFileState,MainPID,ExecMainStatus" in rendered:
                    return _completed(
                        command,
                        stdout=(
                            '[{"name": "twinr-remote-memory-watchdog.service", "active_state": "active",'
                            ' "sub_state": "running", "unit_file_state": "enabled", "main_pid": "101",'
                            ' "exec_main_status": "0"},'
                            ' {"name": "twinr-runtime-supervisor.service", "active_state": "active",'
                            ' "sub_state": "running", "unit_file_state": "enabled", "main_pid": "102",'
                            ' "exec_main_status": "0"},'
                            ' {"name": "twinr-web.service", "active_state": "active",'
                            ' "sub_state": "running", "unit_file_state": "enabled", "main_pid": "103",'
                            ' "exec_main_status": "0"},'
                            ' {"name": "twinr-whatsapp-channel.service", "active_state": "active",'
                            ' "sub_state": "running", "unit_file_state": "enabled", "main_pid": "104",'
                            ' "exec_main_status": "0"}]\n'
                        ),
                    )
                if "pip install" in rendered:
                    return _completed(command, stdout="Successfully installed twinr\n")
                if "check_pi_openai_env_contract.py" in rendered:
                    return _completed(command, stdout='{"ok": true}\n')
                if "actual_sha=$(sha256sum" in rendered:
                    return _completed(command, stdout="")
                return _completed(command)

            result = deploy_pi_runtime(
                project_root=root,
                pi_env_path=pi_env_path,
                rollout_services=("twinr-whatsapp-channel",),
                subprocess_runner=_runner,
                mirror_watchdog=mirror,
            )

        self.assertEqual(
            result.restarted_services,
            (
                "twinr-remote-memory-watchdog.service",
                "twinr-runtime-supervisor.service",
                "twinr-web.service",
                "twinr-whatsapp-channel.service",
            ),
        )
        joined = "\n".join(" ".join(command) for command in commands)
        self.assertIn("twinr-whatsapp-channel.service", joined)


if __name__ == "__main__":
    unittest.main()
