from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import tomllib
import unittest
from typing import Any, Sequence
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.ops.pi_repo_mirror import PiRepoMirrorCycleResult
from twinr.ops.pi_runtime_deploy import PiRuntimeDeployError, deploy_pi_runtime
from twinr.ops.pi_runtime_deploy_remote import (
    PiSystemdServiceState,
    PiRemoteExecutor,
    RetentionCanaryProbeError,
    install_editable_package,
    verify_python_import_contract,
)
from twinr.ops.self_coding_pi import load_pi_connection_settings
from twinr.ops.venv_bridged_system_cleanup import find_shadowed_direct_dependency_distributions
from twinr.ops.venv_system_site_bridge import ensure_pi_system_site_packages_bridge
from twinr.ops.venv_wrapper_repair import repair_venv_python_shebangs

_TEST_PI_HOST = "192.0.2.10"
_TEST_PI_SSH_USER = "pi-test-user"
_TEST_PI_SSH_PASSWORD = "placeholder-password"
_TEST_OPENAI_API_KEY = "placeholder-openai-key"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEPLOY_PI_RUNTIME_CLI_PATH = _REPO_ROOT / "hardware" / "ops" / "deploy_pi_runtime.py"
_TEST_PI_IMPORT_MODULES = (
    "dateutil",
    "markupsafe",
    "starlette",
    "pydantic",
    "pydantic_core",
    "urllib3",
    "gpiod",
    "lgpio",
    "pigpio",
    "rapidfuzz",
    "wcwidth",
    "onnx",
    "msgspec",
    "orjson",
    "portalocker",
    "zstandard",
    "h2",
    "opentelemetry.trace",
    "twinr.memory.context_store",
    "twinr.memory.longterm.storage._remote_current_records",
    "twinr.memory.longterm.runtime.health",
)
_TEST_PI_ATTRIBUTE_CONTRACTS: dict[str, Sequence[str]] = {
    "twinr.hardware.camera_ai.adapter_impl.observe:AICameraAdapterObserveMixin": (
        "observe_attention_stream",
        "observe_attention_from_frame_stream",
        "observe_gesture_stream",
        "observe_gesture_from_frame_stream",
    ),
    "twinr.hardware.camera_ai.adapter_impl.perception:AICameraAdapterPerceptionMixin": (
        "observe_perception_stream",
    ),
    "twinr.hardware.camera_ai.adapter_impl.core:LocalAICameraAdapter": (
        "observe_perception_stream",
        "observe_attention_stream",
        "observe_attention_from_frame_stream",
        "observe_gesture_stream",
        "observe_gesture_from_frame_stream",
    ),
}


def _init_git_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Twinr Tests"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "tests@example.com"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=root, check=True)


def _completed(
    args: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr=stderr)


def _load_deploy_pi_runtime_cli_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "test_hardware_ops_deploy_pi_runtime_cli",
        _DEPLOY_PI_RUNTIME_CLI_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load deploy CLI module from {_DEPLOY_PI_RUNTIME_CLI_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_contract_stdout(
    modules: tuple[str, ...] = _TEST_PI_IMPORT_MODULES,
    *,
    python_path: str = "/twinr/.venv/bin/python",
    attribute_contracts: dict[str, Sequence[str]] = _TEST_PI_ATTRIBUTE_CONTRACTS,
    validated_attribute_contracts: tuple[str, ...] | None = None,
) -> str:
    checked_attribute_contracts = tuple(
        f"{target}.{attribute}"
        for target, attributes in attribute_contracts.items()
        for attribute in attributes
    )
    if validated_attribute_contracts is None:
        validated_attribute_contracts = checked_attribute_contracts
    return json.dumps(
        {
            "python_path": python_path,
            "checked_modules": list(modules),
            "imported_modules": list(modules),
            "failed_imports": {},
            "checked_attribute_contracts": list(checked_attribute_contracts),
            "validated_attribute_contracts": list(validated_attribute_contracts),
            "failed_attribute_contracts": {},
            "missing_attributes": {},
            "elapsed_s": 0.123,
        }
    ) + "\n"


def _repo_attestation_stdout(
    *,
    verified_entry_count: int = 2,
    verified_file_count: int = 2,
    verified_symlink_count: int = 0,
    missing_count: int = 0,
    mismatch_count: int = 0,
    sampled_missing_paths: tuple[str, ...] = (),
    sampled_mismatch_details: tuple[str, ...] = (),
) -> str:
    return json.dumps(
        {
            "verified_entry_count": verified_entry_count,
            "verified_file_count": verified_file_count,
            "verified_symlink_count": verified_symlink_count,
            "missing_count": missing_count,
            "mismatch_count": mismatch_count,
            "sampled_missing_paths": list(sampled_missing_paths),
            "sampled_mismatch_details": list(sampled_mismatch_details),
            "elapsed_s": 0.123,
        }
    ) + "\n"


def _normalized_requirement_name(requirement: str) -> str:
    token = re.split(r"[<>=!~;\[\]\s]", str(requirement).strip(), maxsplit=1)[0]
    return token.strip().lower().replace("_", "-")


def _write_dist_metadata(
    *,
    site_packages_dir: Path,
    name: str,
    version: str,
    requires: tuple[str, ...] = (),
) -> None:
    normalized_name = name.replace("-", "_")
    dist_info_dir = site_packages_dir / f"{normalized_name}-{version}.dist-info"
    dist_info_dir.mkdir(parents=True, exist_ok=True)
    metadata_lines = [
        "Metadata-Version: 2.1",
        f"Name: {name}",
        f"Version: {version}",
    ]
    metadata_lines.extend(f"Requires-Dist: {requirement}" for requirement in requires)
    (dist_info_dir / "METADATA").write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")


class _FakeMirrorWatchdog:
    def __init__(self) -> None:
        self.last_call: dict[str, object] = {}
        self.project_root: Path | None = None

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
            host=_TEST_PI_HOST,
            remote_root="/twinr",
            drift_detected=True,
            sync_applied=True,
            checksum_used=True,
            verified_clean=True,
            change_count=1,
            sampled_change_lines=(">f+++++++++ src/twinr/ops/pi_runtime_deploy.py",),
            duration_s=0.42,
        )


class _MutatingSourceMirrorWatchdog:
    def __init__(self, authoritative_root: Path) -> None:
        self.authoritative_root = authoritative_root
        self.project_root = authoritative_root
        self.snapshot_readme: str | None = None
        self.snapshot_root: Path | None = None

    def probe_once(
        self,
        *,
        apply_sync: bool = True,
        checksum: bool = True,
        max_change_lines: int = 40,
    ) -> PiRepoMirrorCycleResult:
        del apply_sync, checksum, max_change_lines
        self.snapshot_root = Path(self.project_root)
        self.snapshot_readme = (self.snapshot_root / "README.md").read_text(encoding="utf-8")
        (self.authoritative_root / "README.md").write_text("changed during deploy\n", encoding="utf-8")
        return PiRepoMirrorCycleResult(
            host=_TEST_PI_HOST,
            remote_root="/twinr",
            drift_detected=True,
            sync_applied=True,
            checksum_used=True,
            verified_clean=True,
            change_count=1,
            sampled_change_lines=(">f+++++++++ README.md",),
            duration_s=0.31,
        )


@contextmanager
def _noop_remote_deploy_lock(*args, **kwargs):
    del args, kwargs
    yield


def _fake_wait_for_services(*, remote, services, wait_timeout_s):
    del remote, wait_timeout_s
    return tuple(
        PiSystemdServiceState(
            name=str(service),
            active_state="active",
            sub_state="running",
            unit_file_state="enabled",
            main_pid=100 + index,
            exec_main_status=0,
            healthy=True,
            load_state="loaded",
            service_type="simple",
            service_result="success",
        )
        for index, service in enumerate(services)
    )


class PiRuntimeDeployTests(unittest.TestCase):
    def setUp(self) -> None:
        self._remote_lock_patcher = mock.patch(
            "twinr.ops.pi_runtime_deploy._remote_deploy_lock",
            _noop_remote_deploy_lock,
        )
        self._remote_lock_patcher.start()
        self._wait_for_services_patcher = mock.patch(
            "twinr.ops.pi_runtime_deploy._wait_for_services",
            _fake_wait_for_services,
        )
        self._wait_for_services_patcher.start()

    def tearDown(self) -> None:
        self._wait_for_services_patcher.stop()
        self._remote_lock_patcher.stop()

    def test_operator_cli_prints_live_progress_to_stderr(self) -> None:
        cli_module = _load_deploy_pi_runtime_cli_module()
        stdout = io.StringIO()
        stderr = io.StringIO()

        def _fake_deploy_pi_runtime(**kwargs):
            progress_callback = kwargs["progress_callback"]
            progress_callback(
                {
                    "kind": "pi_runtime_deploy_progress",
                    "phase": "editable_install",
                    "event": "start",
                    "step": "pip_install_editable",
                }
            )
            progress_callback(
                {
                    "kind": "pi_runtime_deploy_progress",
                    "phase": "editable_install",
                    "event": "end",
                    "step": "pip_install_editable",
                    "elapsed_s": 33.82,
                }
            )
            return object()

        with (
            mock.patch.object(cli_module, "deploy_pi_runtime", side_effect=_fake_deploy_pi_runtime),
            mock.patch.object(cli_module, "asdict", return_value={"ok": True, "duration_s": 1.23}),
            mock.patch.object(
                sys,
                "argv",
                [
                    str(_DEPLOY_PI_RUNTIME_CLI_PATH),
                    "--skip-env-sync",
                    "--skip-env-contract-check",
                ],
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_code = cli_module.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stdout.getvalue()), {"ok": True, "duration_s": 1.23})
        stderr_lines = [json.loads(line) for line in stderr.getvalue().splitlines() if line.strip()]
        self.assertEqual(
            stderr_lines,
            [
                {
                    "kind": "pi_runtime_deploy_progress",
                    "phase": "editable_install",
                    "event": "start",
                    "step": "pip_install_editable",
                },
                {
                    "kind": "pi_runtime_deploy_progress",
                    "phase": "editable_install",
                    "event": "end",
                    "step": "pip_install_editable",
                    "elapsed_s": 33.82,
                },
            ],
        )

    def test_pyproject_declares_direct_runtime_transitive_distributions_explicitly(self) -> None:
        payload = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        dependency_names = {
            _normalized_requirement_name(requirement)
            for requirement in payload["project"]["dependencies"]
        }
        self.assertTrue(
            {
                "markupsafe",
                "pydantic",
                "pydantic-core",
                "python-dateutil",
                "starlette",
                "urllib3",
            }.issubset(dependency_names)
        )

        optional_dependencies = payload["project"]["optional-dependencies"]
        self.assertIn("pi-runtime", optional_dependencies)
        self.assertIn("router-training", optional_dependencies)
        self.assertIn("piaicam-import", optional_dependencies)

    def test_pyproject_pi_runtime_extra_matches_pi_runtime_manifest(self) -> None:
        payload = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        pyproject_requirements = tuple(payload["project"]["optional-dependencies"]["pi-runtime"])
        manifest_requirements = tuple(
            line.strip()
            for line in (_REPO_ROOT / "hardware" / "ops" / "pi_runtime_requirements.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        self.assertEqual(pyproject_requirements, manifest_requirements)

    def test_ensure_pi_system_site_packages_bridge_writes_only_existing_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            site_packages_dir = root / ".venv" / "lib" / "python3.11" / "site-packages"
            existing_dist_packages = root / "usr" / "lib" / "python3" / "dist-packages"
            missing_dist_packages = root / "usr" / "local" / "lib" / "python3.11" / "dist-packages"
            existing_dist_packages.mkdir(parents=True, exist_ok=True)

            result = ensure_pi_system_site_packages_bridge(
                site_packages_dir=site_packages_dir,
                candidate_paths=(existing_dist_packages, missing_dist_packages),
            )

            self.assertTrue(result.changed)
            self.assertEqual(result.active_paths, (str(existing_dist_packages),))
            self.assertEqual(
                Path(result.bridge_path).read_text(encoding="utf-8"),
                f"{existing_dist_packages}\n",
            )

            second_result = ensure_pi_system_site_packages_bridge(
                site_packages_dir=site_packages_dir,
                candidate_paths=(existing_dist_packages, missing_dist_packages),
            )

            self.assertFalse(second_result.changed)
            self.assertEqual(second_result.active_paths, (str(existing_dist_packages),))

    def test_find_shadowed_direct_dependency_distributions_prefers_satisfying_system_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pyproject_path = root / "pyproject.toml"
            pyproject_path.write_text(
                "[project]\nname = 'twinr'\ndependencies = ['PyQt5>=5.15,<6']\n",
                encoding="utf-8",
            )
            venv_site_packages = root / ".venv" / "lib" / "python3.11" / "site-packages"
            system_site_packages = root / "usr" / "lib" / "python3" / "dist-packages"
            _write_dist_metadata(
                site_packages_dir=venv_site_packages,
                name="PyQt5",
                version="5.15.11",
                requires=("PyQt5-sip (>=12.15, <13)", "PyQt5-Qt5 (>=5.15.2, <5.16.0)"),
            )
            _write_dist_metadata(
                site_packages_dir=system_site_packages,
                name="PyQt5",
                version="5.15.9",
                requires=("PyQt5-sip",),
            )

            result = find_shadowed_direct_dependency_distributions(
                project_pyproject=pyproject_path,
                venv_site_packages_dir=venv_site_packages,
                bridged_system_paths=(system_site_packages,),
            )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "PyQt5")
        self.assertEqual(result[0].venv_version, "5.15.11")
        self.assertEqual(result[0].system_version, "5.15.9")

    def test_repair_venv_python_shebangs_rewrites_stale_wrappers_and_activation_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bin_dir = Path(temp_dir) / ".venv" / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            activate = bin_dir / "activate"
            activate.write_text(
                'VIRTUAL_ENV="/home/thh/twinr/.venv"\nexport VIRTUAL_ENV\n',
                encoding="utf-8",
            )
            activate_csh = bin_dir / "activate.csh"
            activate_csh.write_text(
                "setenv VIRTUAL_ENV /home/thh/twinr/.venv\n",
                encoding="utf-8",
            )
            activate_fish = bin_dir / "activate.fish"
            activate_fish.write_text(
                "set -gx VIRTUAL_ENV /home/thh/twinr/.venv\n",
                encoding="utf-8",
            )
            stale = bin_dir / "pytest"
            stale.write_text(
                "#!/home/thh/twinr/.venv/bin/python\nfrom pytest import console_main\n",
                encoding="utf-8",
            )
            current = bin_dir / "pip"
            current.write_text(
                "#!/twinr/.venv/bin/python\nfrom pip._internal.cli.main import main\n",
                encoding="utf-8",
            )
            portable = bin_dir / "uvicorn"
            portable.write_text(
                "#!/usr/bin/env python\nprint('portable')\n",
                encoding="utf-8",
            )
            (bin_dir / "python").symlink_to("python3.11")

            result = repair_venv_python_shebangs(
                bin_dir=bin_dir,
                expected_interpreter="/twinr/.venv/bin/python",
            )
            expected_venv_dir = "/twinr/.venv"
            self.assertEqual(result.checked_files, 6)
            self.assertEqual(result.rewritten_files, 4)
            self.assertEqual(result.sample_paths, ("activate", "activate.csh", "activate.fish", "pytest"))
            self.assertEqual(
                activate.read_text(encoding="utf-8").splitlines()[0],
                f'VIRTUAL_ENV="{expected_venv_dir}"',
            )
            self.assertEqual(
                activate_csh.read_text(encoding="utf-8").splitlines()[0],
                f"setenv VIRTUAL_ENV {expected_venv_dir}",
            )
            self.assertEqual(
                activate_fish.read_text(encoding="utf-8").splitlines()[0],
                f"set -gx VIRTUAL_ENV {expected_venv_dir}",
            )
            self.assertEqual(
                stale.read_text(encoding="utf-8").splitlines()[0],
                "#!/twinr/.venv/bin/python",
            )
            self.assertEqual(
                current.read_text(encoding="utf-8").splitlines()[0],
                "#!/twinr/.venv/bin/python",
            )
            self.assertEqual(
                portable.read_text(encoding="utf-8").splitlines()[0],
                "#!/usr/bin/env python",
            )

    def test_deploy_runs_env_sync_install_restart_and_verification(self) -> None:
        commands: list[list[str]] = []
        envs: list[dict[str, str] | None] = []
        inputs: list[str | None] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("Twinr\n", encoding="utf-8")
            env_path = root / ".env"
            env_path.write_text(f"OPENAI_API_KEY={_TEST_OPENAI_API_KEY}\n", encoding="utf-8")
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            local_sha = hashlib.sha256(env_path.read_bytes()).hexdigest()
            _init_git_repo(root)
            mirror = _FakeMirrorWatchdog()

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                envs.append(kwargs.get("env"))
                inputs.append(kwargs.get("input"))
                rendered = " ".join(command)
                if "scp" in command:
                    return _completed(command)
                if "sha256sum /twinr/.env" in rendered:
                    return _completed(command, stdout="")
                if "pip install" in rendered:
                    return _completed(command, stdout="Successfully installed twinr\n")
                if "repair_venv_python_shebangs" in rendered:
                    return _completed(
                        command,
                        stdout='{"checked_files": 3, "rewritten_files": 1, "sample_paths": ["pytest"]}\n',
                    )
                if "venv_system_site_bridge.py" in rendered:
                    return _completed(
                        command,
                        stdout=(
                            '{"bridge_path": "/twinr/.venv/lib/python3.11/site-packages/'
                            'twinr_pi_system_site.pth", "active_paths": ["/usr/lib/python3/dist-packages"],'
                            ' "changed": true}\n'
                        ),
                    )
                if "repo_attestation_manifest_path" in rendered:
                    return _completed(
                        command,
                        stdout=_repo_attestation_stdout(
                            verified_entry_count=1,
                            verified_file_count=1,
                        ),
                    )
                if "importlib.import_module" in rendered:
                    return _completed(command, stdout=_import_contract_stdout())
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
                if "twinr.memory.longterm.evaluation.live_retention_canary" in rendered:
                    return _completed(
                        command,
                        stdout=(
                            '{"status": "ok", "ready": true, '
                            '"archived_memory_ids": ["episode:retention_old_weather"], '
                            '"pruned_memory_ids": ["observation:retention_old_presence"], '
                            '"fresh_kept_ids": ["event:retention_future_appointment"], '
                            '"fresh_archived_ids": ["episode:retention_old_weather"]}\n'
                        ),
                    )
                if "actual_sha=$(sha256sum" in rendered:
                    return _completed(command, stdout="/twinr/.env.deploy-backup-20260324T000000Z")
                return _completed(command)

            result = deploy_pi_runtime(
                project_root=root,
                pi_env_path=pi_env_path,
                subprocess_runner=_runner,
                mirror_watchdog=mirror,
                live_text="Antworte nur mit: ok.",
                verify_retention_canary=True,
            )

        self.assertTrue(result.ok)
        self.assertRegex(result.release_id, r"^[0-9a-f]{64}$")
        self.assertRegex(result.source_commit, r"^[0-9a-f]{40}$")
        self.assertTrue(result.repo_mirror.sync_applied)
        self.assertEqual(result.repo_attestation.verified_entry_count, 1)
        self.assertEqual(result.repo_attestation.mismatch_count, 0)
        self.assertTrue(result.release_manifest_sync.remote_path.endswith("current_release_manifest.json"))
        self.assertTrue(result.env_sync is not None)
        assert result.env_sync is not None
        self.assertEqual(result.env_sync.sha256, local_sha)
        self.assertTrue(result.env_sync.changed)
        self.assertTrue(all(state.healthy for state in result.service_states))
        assert result.import_contract is not None
        self.assertEqual(result.import_contract.checked_modules, _TEST_PI_IMPORT_MODULES)
        self.assertIn(
            "twinr.hardware.camera_ai.adapter_impl.observe:AICameraAdapterObserveMixin.observe_gesture_stream",
            result.import_contract.validated_attribute_contracts,
        )
        self.assertEqual(result.env_contract, {"ok": True, "detail": "ready"})
        self.assertIsNotNone(result.retention_canary)
        assert result.retention_canary is not None
        self.assertTrue(result.retention_canary["ready"])
        self.assertEqual(mirror.last_call["apply_sync"], True)
        self.assertEqual(mirror.last_call["checksum"], True)
        joined = "\n".join(" ".join(command) for command in commands)
        self.assertIn("-type d -name __pycache__ -exec sudo chown -R", joined)
        self.assertIn("compileall -q -f --invalidation-mode checked-hash", joined)
        self.assertIn("sshpass -d 0 scp", joined)
        self.assertIn("pip install --no-deps -e", joined)
        self.assertIn("repair_venv_python_shebangs", joined)
        self.assertIn("repo_attestation_manifest_path", joined)
        self.assertIn("current_release_manifest.json", joined)
        self.assertIn("systemctl daemon-reload", joined)
        self.assertIn("systemctl restart", joined)
        self.assertIn("--live-text", joined)
        self.assertIn("twinr.memory.longterm.evaluation.live_retention_canary", joined)
        assert result.bytecode_refresh_summary is not None
        self.assertIn("checked-hash bytecode", result.bytecode_refresh_summary)
        self.assertTrue(any(value == _TEST_PI_SSH_PASSWORD for value in inputs))
        self.assertFalse(any(env and env.get("SSHPASS") for env in envs))
        assert result.editable_install_summary is not None
        self.assertIn("normalized 1 stale venv entrypoint file", result.editable_install_summary)
        self.assertIn("bridged 1 Pi system site-package path into the venv", result.editable_install_summary)

    def test_deploy_fails_closed_when_repo_attestation_finds_stale_remote_file(self) -> None:
        commands: list[list[str]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("Twinr\n", encoding="utf-8")
            env_path = root / ".env"
            env_path.write_text(f"OPENAI_API_KEY={_TEST_OPENAI_API_KEY}\n", encoding="utf-8")
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                "\n".join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            _init_git_repo(root)
            mirror = _FakeMirrorWatchdog()

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                rendered = " ".join(command)
                if "sha256sum /twinr/.env" in rendered:
                    return _completed(command, stdout="")
                if "repo_attestation_manifest_path" in rendered:
                    return _completed(
                        command,
                        stdout=_repo_attestation_stdout(
                            verified_entry_count=1,
                            verified_file_count=0,
                            mismatch_count=1,
                            sampled_mismatch_details=(
                                "README.md: expected sha256 local, got remote",
                            ),
                        ),
                    )
                return _completed(command)

            with self.assertRaises(PiRuntimeDeployError) as exc_info:
                deploy_pi_runtime(
                    project_root=root,
                    pi_env_path=pi_env_path,
                    services=("twinr-runtime-supervisor",),
                    subprocess_runner=_runner,
                    mirror_watchdog=mirror,
                    install_editable=False,
                    install_systemd_units=False,
                    verify_env_contract=False,
                )

        self.assertEqual(exc_info.exception.phase, "repo_attestation")
        self.assertIn("1 mismatched", str(exc_info.exception))
        joined = "\n".join(" ".join(command) for command in commands)
        self.assertIn("repo_attestation_manifest_path", joined)
        self.assertNotIn("systemctl restart", joined)

    def test_deploy_uses_repo_snapshot_when_authoritative_tree_mutates_during_mirror(self) -> None:
        commands: list[list[str]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("Twinr\n", encoding="utf-8")
            env_path = root / ".env"
            env_path.write_text(f"OPENAI_API_KEY={_TEST_OPENAI_API_KEY}\n", encoding="utf-8")
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                "\n".join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            _init_git_repo(root)
            mirror = _MutatingSourceMirrorWatchdog(root)

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                rendered = " ".join(command)
                if "repo_attestation_manifest_path" in rendered:
                    return _completed(
                        command,
                        stdout=_repo_attestation_stdout(
                            verified_entry_count=1,
                            verified_file_count=1,
                        ),
                    )
                if "importlib.import_module" in rendered:
                    return _completed(command, stdout=_import_contract_stdout())
                if "actual_sha=$(sha256sum" in rendered:
                    return _completed(command, stdout="")
                return _completed(command)

            result = deploy_pi_runtime(
                project_root=root,
                pi_env_path=pi_env_path,
                services=("twinr-runtime-supervisor",),
                sync_env=False,
                install_editable=False,
                install_systemd_units=False,
                verify_env_contract=False,
                verify_retention_canary=False,
                subprocess_runner=_runner,
                mirror_watchdog=mirror,
            )

            self.assertTrue(result.ok)
            self.assertEqual((root / "README.md").read_text(encoding="utf-8"), "changed during deploy\n")
            self.assertEqual(mirror.snapshot_readme, "Twinr\n")
            self.assertIsNotNone(mirror.snapshot_root)
            assert mirror.snapshot_root is not None
            self.assertNotEqual(mirror.snapshot_root, root.resolve())
            self.assertEqual(mirror.snapshot_root.name, "authoritative_repo")

        joined = "\n".join(" ".join(command) for command in commands)
        self.assertIn("repo_attestation_manifest_path", joined)

    def test_deploy_retargets_default_watchdog_to_repo_snapshot(self) -> None:
        commands: list[list[str]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("Twinr\n", encoding="utf-8")
            env_path = root / ".env"
            env_path.write_text(f"OPENAI_API_KEY={_TEST_OPENAI_API_KEY}\n", encoding="utf-8")
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                "\n".join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            _init_git_repo(root)
            mirror = _FakeMirrorWatchdog()

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                rendered = " ".join(command)
                if "sha256sum /twinr/.env" in rendered:
                    return _completed(command, stdout="")
                if "repo_attestation_manifest_path" in rendered:
                    return _completed(command, stdout=_repo_attestation_stdout())
                if "importlib.import_module" in rendered:
                    return _completed(command, stdout=_import_contract_stdout())
                if "ActiveState,SubState,UnitFileState,MainPID,ExecMainStatus" in rendered:
                    return _completed(
                        command,
                        stdout='[{"name": "twinr-runtime-supervisor.service", "active_state": "active", "sub_state": "running", "unit_file_state": "enabled", "main_pid": "102", "exec_main_status": "0"}]\n',
                    )
                if "check_pi_openai_env_contract.py" in rendered:
                    return _completed(command, stdout='{"ok": true}\n')
                if "actual_sha=$(sha256sum" in rendered:
                    return _completed(command, stdout="")
                return _completed(command)

            with mock.patch("twinr.ops.pi_runtime_deploy.PiRepoMirrorWatchdog.from_env", return_value=mirror):
                result = deploy_pi_runtime(
                    project_root=root,
                    pi_env_path=pi_env_path,
                    services=("twinr-runtime-supervisor",),
                    subprocess_runner=_runner,
                    install_editable=False,
                    install_systemd_units=False,
                )

        self.assertTrue(result.ok)
        assert mirror.project_root is not None
        self.assertEqual(mirror.project_root.name, "authoritative_repo")
        self.assertTrue(getattr(mirror, "_authoritative_source_is_snapshot"))

    def test_install_editable_package_syncs_only_pending_project_dependencies(self) -> None:
        commands: list[list[str]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                rendered = " ".join(command)
                if "pip install --no-deps -e" in rendered:
                    return _completed(command, stdout="Successfully installed twinr\n")
                if "project_pyproject = Path" in rendered:
                    return _completed(
                        command,
                        stdout='{"requirements": ["defusedxml>=0.7.1,<1", "feedparser>=6.0.12,<7"]}\n',
                    )
                if "pip install defusedxml>=0.7.1,<1 feedparser>=6.0.12,<7" in rendered:
                    return _completed(
                        command,
                        stdout="Successfully installed defusedxml feedparser sgmllib3k\n",
                    )
                if "repair_venv_python_shebangs" in rendered:
                    return _completed(
                        command,
                        stdout='{"checked_files": 2, "rewritten_files": 0, "sample_paths": []}\n',
                    )
                if "venv_system_site_bridge.py" in rendered:
                    return _completed(
                        command,
                        stdout=(
                            '{"bridge_path": "/twinr/.venv/lib/python3.11/site-packages/'
                            'twinr_pi_system_site.pth", "active_paths": ["/usr/lib/python3/dist-packages"],'
                            ' "changed": false}\n'
                        ),
                    )
                return _completed(command)

            remote = PiRemoteExecutor(
                settings=load_pi_connection_settings(pi_env_path),
                subprocess_runner=_runner,
                timeout_s=30,
            )
            summary = install_editable_package(
                remote=remote,
                remote_root="/twinr",
                install_with_deps=False,
            )

        joined = "\n".join(" ".join(command) for command in commands)
        self.assertIn("pip install --no-deps -e", joined)
        self.assertIn("project_pyproject = Path", joined)
        self.assertIn("pip install", joined)
        self.assertIn("defusedxml>=0.7.1,<1", joined)
        self.assertIn("feedparser>=6.0.12,<7", joined)
        self.assertNotIn("pip install -e /twinr", joined)
        self.assertIn("installed 2 mirrored project dependencies", summary)
        self.assertIn("verified 2 venv entrypoint file(s)", summary)
        self.assertIn("verified Pi system site-package bridge for 1 path", summary)

    def test_install_editable_package_removes_venv_shadowed_system_dependency_before_checks(self) -> None:
        commands: list[list[str]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            def _runner(args, **kwargs):
                del kwargs
                command = [str(part) for part in args]
                commands.append(command)
                rendered = " ".join(command)
                if "venv_system_site_bridge.py" in rendered:
                    return _completed(
                        command,
                        stdout=(
                            '{"bridge_path": "/twinr/.venv/lib/python3.11/site-packages/'
                            'twinr_pi_system_site.pth", "active_paths": ["/usr/lib/python3/dist-packages"],'
                            ' "changed": false}\n'
                        ),
                    )
                if "venv_bridged_system_cleanup.py" in rendered:
                    return _completed(
                        command,
                        stdout=(
                            '[{"name": "PyQt5", "venv_version": "5.15.11",'
                            ' "venv_path": "/twinr/.venv/lib/python3.11/site-packages",'
                            ' "system_version": "5.15.9",'
                            ' "system_path": "/usr/lib/python3/dist-packages"}]\n'
                        ),
                    )
                if "pip uninstall -y PyQt5" in rendered:
                    return _completed(command, stdout="Successfully uninstalled PyQt5-5.15.11\n")
                if "pip install --no-deps -e" in rendered:
                    return _completed(command, stdout="Successfully installed twinr\n")
                if "pip install --dry-run --quiet --report -" in rendered:
                    return _completed(command, stdout='{"ok": true, "pending": []}\n')
                if '"$remote_python" -m pip check' in rendered:
                    return _completed(command, stdout="")
                if "repair_venv_python_shebangs" in rendered:
                    return _completed(
                        command,
                        stdout='{"checked_files": 2, "rewritten_files": 0, "sample_paths": []}\n',
                    )
                return _completed(command)

            remote = PiRemoteExecutor(
                settings=load_pi_connection_settings(pi_env_path),
                subprocess_runner=_runner,
                timeout_s=30,
            )
            summary = install_editable_package(
                remote=remote,
                remote_root="/twinr",
                install_with_deps=False,
            )

        rendered_commands = [" ".join(command) for command in commands]
        cleanup_index = next(
            index for index, rendered in enumerate(rendered_commands) if "venv_bridged_system_cleanup.py" in rendered
        )
        uninstall_index = next(
            index for index, rendered in enumerate(rendered_commands) if "pip uninstall -y PyQt5" in rendered
        )
        install_index = next(
            index for index, rendered in enumerate(rendered_commands) if "pip install --no-deps -e" in rendered
        )
        dry_run_index = next(
            index
            for index, rendered in enumerate(rendered_commands)
            if "project_name = normalize_name" in rendered
        )
        pip_check_index = next(
            index for index, rendered in enumerate(rendered_commands) if '"$remote_python" -m pip check' in rendered
        )
        self.assertLess(cleanup_index, uninstall_index)
        self.assertLess(uninstall_index, install_index)
        self.assertLess(install_index, dry_run_index)
        self.assertLess(dry_run_index, pip_check_index)
        self.assertIn("removed 1 venv-shadowed direct dependency", summary)
        self.assertIn("PyQt5 (venv 5.15.11 -> system 5.15.9)", summary)

    def test_install_editable_package_emits_progress_for_long_substeps(self) -> None:
        commands: list[list[str]] = []
        progress_events: list[dict[str, object]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                rendered = " ".join(command)
                if "venv_system_site_bridge.py" in rendered:
                    return _completed(
                        command,
                        stdout=(
                            '{"bridge_path": "/twinr/.venv/lib/python3.11/site-packages/'
                            'twinr_pi_system_site.pth", "active_paths": [], "changed": false}\n'
                        ),
                    )
                if "pip install --no-deps -e" in rendered:
                    return _completed(command, stdout="Successfully installed twinr\n")
                if "project_name = normalize_name" in rendered:
                    return _completed(command, stdout='{"ok": true, "pending": []}\n')
                if '"$remote_python" -m pip check' in rendered:
                    return _completed(command, stdout="")
                if "repair_venv_python_shebangs" in rendered:
                    return _completed(
                        command,
                        stdout='{"checked_files": 2, "rewritten_files": 0, "sample_paths": []}\n',
                    )
                return _completed(command)

            remote = PiRemoteExecutor(
                settings=load_pi_connection_settings(pi_env_path),
                subprocess_runner=_runner,
                timeout_s=30,
            )
            install_editable_package(
                remote=remote,
                remote_root="/twinr",
                install_with_deps=False,
                progress_callback=progress_events.append,
            )

        editable_events = [
            (str(event.get("event", "")), str(event.get("step", "")))
            for event in progress_events
            if event.get("phase") == "editable_install" and event.get("step")
        ]
        self.assertEqual(
            editable_events,
            [
                ("start", "ensure_remote_venv"),
                ("end", "ensure_remote_venv"),
                ("start", "bridge_system_site_packages"),
                ("end", "bridge_system_site_packages"),
                ("start", "cleanup_shadowed_system_packages"),
                ("end", "cleanup_shadowed_system_packages"),
                ("start", "pip_install_editable"),
                ("end", "pip_install_editable"),
                ("start", "sync_runtime_dependencies"),
                ("end", "sync_runtime_dependencies"),
                ("start", "pip_check"),
                ("end", "pip_check"),
                ("start", "repair_venv_entrypoints"),
                ("end", "repair_venv_entrypoints"),
            ],
        )
        self.assertTrue(all(event.get("kind") == "pi_runtime_deploy_progress" for event in progress_events))

    def test_verify_python_import_contract_attests_all_requested_modules(self) -> None:
        commands: list[list[str]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                rendered = " ".join(command)
                if "importlib.import_module" in rendered:
                    return _completed(command, stdout=_import_contract_stdout())
                return _completed(command)

            remote = PiRemoteExecutor(
                settings=load_pi_connection_settings(pi_env_path),
                subprocess_runner=_runner,
                timeout_s=30,
            )
            result = verify_python_import_contract(
                remote=remote,
                remote_python="/twinr/.venv/bin/python",
                modules=_TEST_PI_IMPORT_MODULES,
            )

        self.assertEqual(result.checked_modules, _TEST_PI_IMPORT_MODULES)
        self.assertEqual(result.imported_modules, _TEST_PI_IMPORT_MODULES)
        self.assertEqual(result.python_path, "/twinr/.venv/bin/python")
        joined = "\n".join(" ".join(command) for command in commands)
        self.assertIn("opentelemetry.trace", joined)
        self.assertIn("twinr.memory.context_store", joined)
        self.assertIn("twinr.memory.longterm.runtime.health", joined)
        self.assertIn("importlib.import_module", joined)

    def test_verify_python_import_contract_validates_required_attributes(self) -> None:
        commands: list[list[str]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                rendered = " ".join(command)
                if "importlib.import_module" in rendered:
                    return _completed(command, stdout=_import_contract_stdout())
                return _completed(command)

            remote = PiRemoteExecutor(
                settings=load_pi_connection_settings(pi_env_path),
                subprocess_runner=_runner,
                timeout_s=30,
            )
            result = verify_python_import_contract(
                remote=remote,
                remote_python="/twinr/.venv/bin/python",
                modules=_TEST_PI_IMPORT_MODULES,
                attribute_contracts=_TEST_PI_ATTRIBUTE_CONTRACTS,
            )

        self.assertIn(
            "twinr.hardware.camera_ai.adapter_impl.core:LocalAICameraAdapter.observe_attention_stream",
            result.checked_attribute_contracts,
        )
        self.assertIn(
            "twinr.hardware.camera_ai.adapter_impl.observe:AICameraAdapterObserveMixin.observe_gesture_stream",
            result.validated_attribute_contracts,
        )
        joined = "\n".join(" ".join(command) for command in commands)
        self.assertIn("observe_gesture_stream", joined)
        self.assertIn("AICameraAdapterObserveMixin", joined)

    def test_verify_python_import_contract_raises_on_failed_module(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                rendered = " ".join(command)
                if "importlib.import_module" in rendered:
                    payload = {
                        "python_path": "/twinr/.venv/bin/python",
                        "checked_modules": ["rapidfuzz", "wcwidth"],
                        "imported_modules": ["rapidfuzz"],
                        "failed_imports": {"wcwidth": "No module named 'wcwidth'"},
                        "elapsed_s": 0.1,
                    }
                    return _completed(command, stdout=json.dumps(payload) + "\n")
                return _completed(command)

            remote = PiRemoteExecutor(
                settings=load_pi_connection_settings(pi_env_path),
                subprocess_runner=_runner,
                timeout_s=30,
            )
            with self.assertRaisesRegex(RuntimeError, "wcwidth"):
                verify_python_import_contract(
                    remote=remote,
                    remote_python="/twinr/.venv/bin/python",
                    modules=("rapidfuzz", "wcwidth"),
                )

    def test_verify_python_import_contract_raises_on_missing_required_attribute(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                rendered = " ".join(command)
                if "importlib.import_module" in rendered:
                    return _completed(
                        command,
                        stdout=_import_contract_stdout(
                            modules=("rapidfuzz",),
                            attribute_contracts={
                                "twinr.hardware.camera_ai.adapter_impl.observe:AICameraAdapterObserveMixin": (
                                    "observe_gesture_stream",
                                ),
                            },
                            validated_attribute_contracts=(),
                        ),
                    )
                return _completed(command)

            remote = PiRemoteExecutor(
                settings=load_pi_connection_settings(pi_env_path),
                subprocess_runner=_runner,
                timeout_s=30,
            )
            with self.assertRaisesRegex(RuntimeError, "observe_gesture_stream"):
                verify_python_import_contract(
                    remote=remote,
                    remote_python="/twinr/.venv/bin/python",
                    modules=("rapidfuzz",),
                    attribute_contracts={
                        "twinr.hardware.camera_ai.adapter_impl.observe:AICameraAdapterObserveMixin": (
                            "observe_gesture_stream",
                        ),
                    },
                )

    def test_deploy_installs_optional_browser_automation_runtime_support(self) -> None:
        commands: list[list[str]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(f"OPENAI_API_KEY={_TEST_OPENAI_API_KEY}\n", encoding="utf-8")
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            browser_root = root / "browser_automation"
            browser_root.mkdir(parents=True, exist_ok=True)
            (browser_root / "runtime_requirements.txt").write_text(
                "playwright>=1.58,<2\n",
                encoding="utf-8",
            )
            (browser_root / "playwright_browsers.txt").write_text(
                "chromium\n",
                encoding="utf-8",
            )
            _init_git_repo(root)
            mirror = _FakeMirrorWatchdog()

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                rendered = " ".join(command)
                if "sha256sum /twinr/.env" in rendered:
                    return _completed(command, stdout="")
                if 'pip install -r "$requirements_path"' in rendered and "browser_automation" in rendered:
                    return _completed(command, stdout="Successfully installed playwright\n")
                if "playwright install" in rendered:
                    return _completed(command, stdout="Downloaded Chromium\n")
                if "pip install --no-deps -e" in rendered:
                    return _completed(command, stdout="Successfully installed twinr\n")
                if "repo_attestation_manifest_path" in rendered:
                    return _completed(command, stdout=_repo_attestation_stdout())
                if "importlib.import_module" in rendered:
                    return _completed(command, stdout=_import_contract_stdout())
                if "ActiveState,SubState,UnitFileState,MainPID,ExecMainStatus" in rendered:
                    return _completed(
                        command,
                        stdout='[{"name": "twinr-runtime-supervisor.service", "active_state": "active", "sub_state": "running", "unit_file_state": "enabled", "main_pid": "102", "exec_main_status": "0"}]\n',
                    )
                if "check_pi_openai_env_contract.py" in rendered:
                    return _completed(command, stdout='{"ok": true}\n')
                if "actual_sha=$(sha256sum" in rendered:
                    return _completed(command, stdout="")
                return _completed(command)

            result = deploy_pi_runtime(
                project_root=root,
                pi_env_path=pi_env_path,
                services=("twinr-runtime-supervisor",),
                subprocess_runner=_runner,
                mirror_watchdog=mirror,
            )

        self.assertTrue(result.ok)
        joined = "\n".join(" ".join(command) for command in commands)
        self.assertIn("browser_automation/runtime_requirements.txt", joined)
        self.assertIn("browser_automation/playwright_browsers.txt", joined)
        self.assertIn("pip install -r \"$requirements_path\"", joined)
        self.assertIn("deps_summary = run", joined)
        self.assertIn("install_summary = run", joined)
        self.assertNotIn("browser_automation-runtime_requirements.txt", joined)
        self.assertNotIn("browser_automation-playwright_browsers.txt", joined)

    def test_deploy_installs_optional_pi_runtime_requirements_manifest(self) -> None:
        commands: list[list[str]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(f"OPENAI_API_KEY={_TEST_OPENAI_API_KEY}\n", encoding="utf-8")
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            ops_root = root / "hardware" / "ops"
            ops_root.mkdir(parents=True, exist_ok=True)
            (ops_root / "pi_runtime_requirements.txt").write_text(
                "rapidfuzz>=3.14,<4\nwcwidth>=0.6,<1\n",
                encoding="utf-8",
            )
            _init_git_repo(root)
            mirror = _FakeMirrorWatchdog()

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                rendered = " ".join(command)
                if "sha256sum /twinr/.env" in rendered:
                    return _completed(command, stdout="")
                if 'pip install -r "$requirements_path"' in rendered and "pi_runtime" in rendered:
                    return _completed(command, stdout="Successfully installed rapidfuzz wcwidth\n")
                if "pip install --no-deps -e" in rendered:
                    return _completed(command, stdout="Successfully installed twinr\n")
                if "repair_venv_python_shebangs" in rendered:
                    return _completed(
                        command,
                        stdout='{"checked_files": 2, "rewritten_files": 0, "sample_paths": []}\n',
                    )
                if "venv_system_site_bridge.py" in rendered:
                    return _completed(
                        command,
                        stdout=(
                            '{"bridge_path": "/twinr/.venv/lib/python3.11/site-packages/'
                            'twinr_pi_system_site.pth", "active_paths": ["/usr/lib/python3/dist-packages"],'
                            ' "changed": false}\n'
                        ),
                    )
                if "repo_attestation_manifest_path" in rendered:
                    return _completed(command, stdout=_repo_attestation_stdout())
                if "importlib.import_module" in rendered:
                    return _completed(command, stdout=_import_contract_stdout())
                if "ActiveState,SubState,UnitFileState,MainPID,ExecMainStatus" in rendered:
                    return _completed(
                        command,
                        stdout='[{"name": "twinr-runtime-supervisor.service", "active_state": "active", "sub_state": "running", "unit_file_state": "enabled", "main_pid": "102", "exec_main_status": "0"}]\n',
                    )
                if "check_pi_openai_env_contract.py" in rendered:
                    return _completed(command, stdout='{"ok": true}\n')
                if "actual_sha=$(sha256sum" in rendered:
                    return _completed(command, stdout="")
                return _completed(command)

            result = deploy_pi_runtime(
                project_root=root,
                pi_env_path=pi_env_path,
                services=("twinr-runtime-supervisor",),
                subprocess_runner=_runner,
                mirror_watchdog=mirror,
            )

        self.assertTrue(result.ok)
        joined = "\n".join(" ".join(command) for command in commands)
        self.assertIn("hardware/ops/pi_runtime_requirements.txt", joined)
        self.assertIn('echo "[pi_runtime] installing python requirements"', joined)
        assert result.editable_install_summary is not None
        self.assertIn("Successfully installed rapidfuzz wcwidth", result.editable_install_summary)

    def test_deploy_skips_env_copy_when_remote_checksum_matches(self) -> None:
        commands: list[list[str]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(f"OPENAI_API_KEY={_TEST_OPENAI_API_KEY}\n", encoding="utf-8")
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            local_sha = hashlib.sha256(env_path.read_bytes()).hexdigest()
            _init_git_repo(root)
            mirror = _FakeMirrorWatchdog()

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                rendered = " ".join(command)
                if "sha256sum /twinr/.env" in rendered:
                    return _completed(command, stdout=f"{local_sha}\n")
                if "repo_attestation_manifest_path" in rendered:
                    return _completed(command, stdout=_repo_attestation_stdout())
                if "importlib.import_module" in rendered:
                    return _completed(command, stdout=_import_contract_stdout())
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
        self.assertNotIn("authoritative.env", joined)

    def test_deploy_emits_phase_progress_events(self) -> None:
        progress_events: list[dict[str, object]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(f"OPENAI_API_KEY={_TEST_OPENAI_API_KEY}\n", encoding="utf-8")
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            _init_git_repo(root)
            mirror = _FakeMirrorWatchdog()

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                rendered = " ".join(command)
                if "sha256sum /twinr/.env" in rendered:
                    return _completed(command, stdout="")
                if "repo_attestation_manifest_path" in rendered:
                    return _completed(command, stdout=_repo_attestation_stdout())
                if "importlib.import_module" in rendered:
                    return _completed(command, stdout=_import_contract_stdout())
                if "ActiveState,SubState,UnitFileState,MainPID,ExecMainStatus" in rendered:
                    return _completed(
                        command,
                        stdout='[{"name": "twinr-runtime-supervisor.service", "active_state": "active", "sub_state": "running", "unit_file_state": "enabled", "main_pid": "102", "exec_main_status": "0"}]\n',
                    )
                return _completed(command)

            result = deploy_pi_runtime(
                project_root=root,
                pi_env_path=pi_env_path,
                services=("twinr-runtime-supervisor",),
                subprocess_runner=_runner,
                mirror_watchdog=mirror,
                install_editable=False,
                install_systemd_units=False,
                verify_env_contract=False,
                verify_retention_canary=False,
                progress_callback=progress_events.append,
            )

        self.assertTrue(result.ok)
        phase_events = [
            (str(event.get("event", "")), str(event.get("phase", "")))
            for event in progress_events
            if "step" not in event
        ]
        self.assertIn(("start", "repo_snapshot"), phase_events)
        self.assertIn(("end", "repo_snapshot"), phase_events)
        self.assertIn(("start", "repo_mirror"), phase_events)
        self.assertIn(("end", "repo_mirror"), phase_events)
        self.assertIn(("start", "repo_attestation"), phase_events)
        self.assertIn(("end", "repo_attestation"), phase_events)
        self.assertIn(("start", "release_manifest_sync"), phase_events)
        self.assertIn(("end", "release_manifest_sync"), phase_events)
        self.assertIn(("start", "python_import_contract"), phase_events)
        self.assertIn(("end", "python_import_contract"), phase_events)
        self.assertTrue(all(event.get("kind") == "pi_runtime_deploy_progress" for event in progress_events))

    def test_deploy_repairs_shared_state_permissions_before_service_restart(self) -> None:
        commands: list[list[str]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(f"OPENAI_API_KEY={_TEST_OPENAI_API_KEY}\n", encoding="utf-8")
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            _init_git_repo(root)
            mirror = _FakeMirrorWatchdog()

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                rendered = " ".join(command)
                if "sha256sum /twinr/.env" in rendered:
                    return _completed(command, stdout="")
                if "repo_attestation_manifest_path" in rendered:
                    return _completed(command, stdout=_repo_attestation_stdout())
                if "importlib.import_module" in rendered:
                    return _completed(command, stdout=_import_contract_stdout())
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

        self.assertTrue(result.ok)
        joined = "\n".join(" ".join(command) for command in commands)
        self.assertIn('install -d -m 700 -o "$owner_user" -g "$owner_user" "$state_dir"', joined)
        self.assertIn('repair_path_if_exists "$state_dir/automations.json" 600', joined)
        self.assertIn('repair_path_if_exists "$state_dir/automations.json.bak" 600', joined)
        self.assertIn('repair_path_if_exists "$state_dir/automations.json.lock" 600', joined)
        self.assertIn('repair_path_if_exists "$state_dir/user_discovery.json" 600', joined)
        self.assertIn('verify_path_if_exists "$state_dir/automations.json" 600', joined)
        self.assertIn('verify_path_if_exists "$state_dir/automations.json.bak" 600', joined)
        self.assertIn('verify_path_if_exists "$state_dir/automations.json.lock" 600', joined)
        self.assertIn('verify_path_if_exists "$state_dir/user_discovery.json" 600', joined)
        self.assertIn('install -d -m 700 -o "$owner_user" -g "$owner_user" "$ops_dir"', joined)
        self.assertIn('repair_path_if_exists "$ops_dir/events.jsonl" 666', joined)
        self.assertIn('repair_path_if_exists "$ops_dir/.events.jsonl.lock" 666', joined)
        self.assertIn('repair_path_if_exists "$ops_dir/remote_memory_watchdog.json" 644', joined)
        self.assertIn('repair_path_if_exists "$ops_dir/display_ambient_impulse.json" 644', joined)
        self.assertIn('repair_path_if_exists "$ops_dir/display_heartbeat.json" 644', joined)
        self.assertIn('repair_path_if_exists "$ops_dir/display_render_state.json" 644', joined)
        self.assertIn('repair_path_if_exists "$ops_dir/streaming_memory_segments.json" 644', joined)
        self.assertIn('verify_mode_if_exists "$ops_dir/events.jsonl" 666', joined)
        self.assertIn('verify_mode_if_exists "$ops_dir/.events.jsonl.lock" 666', joined)
        self.assertIn('verify_mode_if_exists "$ops_dir/remote_memory_watchdog.json" 644', joined)
        self.assertIn('verify_mode_if_exists "$ops_dir/display_ambient_impulse.json" 644', joined)
        self.assertIn('verify_mode_if_exists "$ops_dir/display_heartbeat.json" 644', joined)
        self.assertIn('verify_mode_if_exists "$ops_dir/display_render_state.json" 644', joined)
        self.assertIn('verify_mode_if_exists "$ops_dir/streaming_memory_segments.json" 644', joined)
        self.assertLess(joined.index('repair_path_if_exists "$state_dir/automations.json" 600'), joined.index("sudo systemctl restart"))
        self.assertLess(joined.index('repair_path_if_exists "$ops_dir/events.jsonl" 666'), joined.index("sudo systemctl restart"))
        self.assertLess(joined.index('repair_path_if_exists "$ops_dir/remote_memory_watchdog.json" 644'), joined.index("sudo systemctl restart"))
        self.assertLess(joined.index('repair_path_if_exists "$ops_dir/display_heartbeat.json" 644'), joined.index("sudo systemctl restart"))
        self.assertLess(joined.index('repair_path_if_exists "$ops_dir/display_render_state.json" 644'), joined.index("sudo systemctl restart"))
        self.assertLess(joined.index("sudo systemctl restart"), joined.index('verify_path_if_exists "$state_dir/automations.json" 600'))
        self.assertLess(joined.index("sudo systemctl restart"), joined.index('verify_mode_if_exists "$ops_dir/events.jsonl" 666'))
        self.assertLess(joined.index("sudo systemctl restart"), joined.index('verify_mode_if_exists "$ops_dir/remote_memory_watchdog.json" 644'))
        self.assertLess(joined.index("sudo systemctl restart"), joined.index('verify_mode_if_exists "$ops_dir/display_heartbeat.json" 644'))
        self.assertLess(joined.index("sudo systemctl restart"), joined.index('verify_mode_if_exists "$ops_dir/display_render_state.json" 644'))

    def test_deploy_rebases_repo_owned_workflow_trace_env_path_for_pi_sync(self) -> None:
        synced_env_snapshots: dict[str, str] = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    (
                        f"OPENAI_API_KEY={_TEST_OPENAI_API_KEY}",
                        "TWINR_WORKFLOW_TRACE_ENABLED=true",
                        "TWINR_WORKFLOW_TRACE_MODE=forensic",
                        "TWINR_WORKFLOW_TRACE_DIR=/home/thh/twinr/state/forensics/workflow_host_voice",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            _init_git_repo(root)
            mirror = _FakeMirrorWatchdog()

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                rendered = " ".join(command)
                if "scp" in command:
                    local_path = Path(command[-2])
                    synced_env_snapshots[local_path.name] = local_path.read_text(encoding="utf-8")
                    return _completed(command)
                if "sha256sum /twinr/.env" in rendered:
                    return _completed(command, stdout="")
                if "repo_attestation_manifest_path" in rendered:
                    return _completed(command, stdout=_repo_attestation_stdout())
                if "importlib.import_module" in rendered:
                    return _completed(command, stdout=_import_contract_stdout())
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
                verify_retention_canary=False,
            )

        self.assertTrue(result.ok)
        synced_env = synced_env_snapshots["authoritative.env"]
        self.assertIn(
            "TWINR_WORKFLOW_TRACE_DIR=/twinr/state/forensics/workflow_host_voice",
            synced_env,
        )

    def test_retention_canary_emits_remote_probe_heartbeat_progress(self) -> None:
        progress_events: list[dict[str, object]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(f"OPENAI_API_KEY={_TEST_OPENAI_API_KEY}\n", encoding="utf-8")
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            _init_git_repo(root)
            mirror = _FakeMirrorWatchdog()

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                rendered = " ".join(command)
                if "sha256sum /twinr/.env" in rendered:
                    return _completed(command, stdout="")
                if "repo_attestation_manifest_path" in rendered:
                    return _completed(command, stdout=_repo_attestation_stdout())
                if "importlib.import_module" in rendered:
                    return _completed(command, stdout=_import_contract_stdout())
                if "ActiveState,SubState,UnitFileState,MainPID,ExecMainStatus" in rendered:
                    return _completed(
                        command,
                        stdout='[{"name": "twinr-runtime-supervisor.service", "active_state": "active", "sub_state": "running", "unit_file_state": "enabled", "main_pid": "102", "exec_main_status": "0"}]\n',
                    )
                return _completed(command)

            with mock.patch("twinr.ops.pi_runtime_deploy._RETENTION_CANARY_HEARTBEAT_S", 0.01), mock.patch(
                "twinr.ops.pi_runtime_deploy._run_retention_canary_probe",
                side_effect=lambda **_kwargs: (time.sleep(0.03) or {"status": "ok", "ready": True, "report_path": "/twinr/report.json"}),
            ):
                result = deploy_pi_runtime(
                    project_root=root,
                    pi_env_path=pi_env_path,
                    services=("twinr-runtime-supervisor",),
                    subprocess_runner=_runner,
                    mirror_watchdog=mirror,
                    install_editable=False,
                    install_systemd_units=False,
                    verify_env_contract=False,
                    verify_retention_canary=True,
                    progress_callback=progress_events.append,
                )

        self.assertTrue(result.ok)
        remote_probe_events = [
            event
            for event in progress_events
            if event.get("phase") == "retention_canary" and event.get("step") == "remote_probe"
        ]
        self.assertTrue(any(event.get("event") == "start" for event in remote_probe_events))
        self.assertTrue(any(event.get("event") == "heartbeat" for event in remote_probe_events))
        self.assertTrue(any(event.get("event") == "end" for event in remote_probe_events))

    def test_deploy_passes_dedicated_retention_canary_timeout(self) -> None:
        captured_command_timeout_s: list[float] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(f"OPENAI_API_KEY={_TEST_OPENAI_API_KEY}\n", encoding="utf-8")
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            _init_git_repo(root)
            mirror = _FakeMirrorWatchdog()

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                rendered = " ".join(command)
                if "sha256sum /twinr/.env" in rendered:
                    return _completed(command, stdout="")
                if "repo_attestation_manifest_path" in rendered:
                    return _completed(command, stdout=_repo_attestation_stdout())
                if "importlib.import_module" in rendered:
                    return _completed(command, stdout=_import_contract_stdout())
                if "ActiveState,SubState,UnitFileState,MainPID,ExecMainStatus" in rendered:
                    return _completed(
                        command,
                        stdout='[{"name": "twinr-runtime-supervisor.service", "active_state": "active", "sub_state": "running", "unit_file_state": "enabled", "main_pid": "102", "exec_main_status": "0"}]\n',
                    )
                return _completed(command)

            with mock.patch(
                "twinr.ops.pi_runtime_deploy._run_retention_canary_probe",
                side_effect=lambda **kwargs: (
                    captured_command_timeout_s.append(float(kwargs["command_timeout_s"]))
                    or {"status": "ok", "ready": True, "report_path": "/twinr/report.json"}
                ),
            ):
                result = deploy_pi_runtime(
                    project_root=root,
                    pi_env_path=pi_env_path,
                    services=("twinr-runtime-supervisor",),
                    subprocess_runner=_runner,
                    mirror_watchdog=mirror,
                    install_editable=False,
                    install_systemd_units=False,
                    verify_env_contract=False,
                    verify_retention_canary=True,
                    retention_canary_timeout_s=600.0,
                )

        self.assertTrue(result.ok)
        self.assertEqual(captured_command_timeout_s, [600.0])

    def test_deploy_recovers_retention_canary_after_remote_host_stabilization(self) -> None:
        progress_events: list[dict[str, object]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(f"OPENAI_API_KEY={_TEST_OPENAI_API_KEY}\n", encoding="utf-8")
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            _init_git_repo(root)
            mirror = _FakeMirrorWatchdog()

            failed_payload = {
                "status": "failed",
                "ready": False,
                "failure_stage": "seed_retention_objects",
                "error_message": (
                    "LongTermRemoteUnavailableError: Accepted remote long-term 'objects' write "
                    "could not be read back: Remote write attestation observed the accepted payload "
                    "without a stable document id."
                ),
                "remote_write_context": {
                    "operation": "store_records_bulk",
                    "request_path": "/v1/external/records/bulk",
                    "request_execution_mode": "async",
                },
            }
            successful_retry_payload = {
                "status": "ok",
                "ready": True,
                "report_path": "/twinr/artifacts/reports/retention_live_canary/retry.json",
            }

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                rendered = " ".join(command)
                if "sha256sum /twinr/.env" in rendered:
                    return _completed(command, stdout="")
                if "repo_attestation_manifest_path" in rendered:
                    return _completed(command, stdout=_repo_attestation_stdout())
                if "importlib.import_module" in rendered:
                    return _completed(command, stdout=_import_contract_stdout())
                if "ActiveState,SubState,UnitFileState,MainPID,ExecMainStatus" in rendered:
                    return _completed(
                        command,
                        stdout='[{"name": "twinr-runtime-supervisor.service", "active_state": "active", "sub_state": "running", "unit_file_state": "enabled", "main_pid": "102", "exec_main_status": "0"}]\n',
                    )
                return _completed(command)

            with mock.patch(
                "twinr.ops.pi_runtime_deploy._run_retention_canary_probe",
                side_effect=[
                    RetentionCanaryProbeError("retention canary failed", payload=failed_payload),
                    successful_retry_payload,
                ],
            ) as probe_mock, mock.patch(
                "twinr.ops.pi_runtime_deploy._diagnose_retention_canary_host_contention",
                return_value={
                    "available": True,
                    "contention_detected": True,
                    "contention_signals": ["backend_query_unhealthy", "active_system_conflicts"],
                },
            ) as diagnose_mock, mock.patch(
                "twinr.ops.pi_runtime_deploy._stabilize_retention_canary_host",
                return_value={
                    "ok": True,
                    "diagnosis": "public_recovered_after_host_stabilization",
                },
            ) as stabilize_mock, mock.patch(
                "twinr.ops.pi_runtime_deploy.time.time_ns",
                return_value=1,
            ):
                result = deploy_pi_runtime(
                    project_root=root,
                    pi_env_path=pi_env_path,
                    services=("twinr-runtime-supervisor",),
                    subprocess_runner=_runner,
                    mirror_watchdog=mirror,
                    install_editable=False,
                    install_systemd_units=False,
                    verify_env_contract=False,
                    verify_retention_canary=True,
                    progress_callback=progress_events.append,
                )

        self.assertTrue(result.ok)
        assert result.retention_canary is not None
        self.assertTrue(result.retention_canary["ready"])
        recovery = result.retention_canary.get("host_contention_recovery")
        self.assertIsInstance(recovery, dict)
        assert isinstance(recovery, dict)
        self.assertEqual(recovery.get("retry_probe_id"), "deploy_retention_canary_1_after_host_stabilization")
        self.assertEqual(probe_mock.call_count, 2)
        diagnose_mock.assert_called_once()
        stabilize_mock.assert_called_once()
        self.assertEqual(
            stabilize_mock.call_args.kwargs["diagnosis"]["contention_signals"],
            ["backend_query_unhealthy", "active_system_conflicts"],
        )
        self.assertEqual(stabilize_mock.call_args.kwargs["ssh_timeout_s"], 180.0)
        retention_events = [
            event
            for event in progress_events
            if event.get("phase") == "retention_canary"
        ]
        self.assertTrue(
            any(event.get("step") == "host_contention_diagnosis" for event in retention_events)
        )
        self.assertTrue(
            any(event.get("step") == "host_contention_stabilization" for event in retention_events)
        )

    def test_deploy_waits_for_fresh_watchdog_sample_after_backend_repair_before_retry(self) -> None:
        progress_events: list[dict[str, object]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(f"OPENAI_API_KEY={_TEST_OPENAI_API_KEY}\n", encoding="utf-8")
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            _init_git_repo(root)
            mirror = _FakeMirrorWatchdog()

            failed_payload = {
                "status": "failed",
                "ready": False,
                "failure_stage": "seed_retention_objects",
                "error_message": "LongTermRemoteUnavailableError: Failed to persist fine-grained remote long-term memory items.",
                "root_cause_message": (
                    "ChonkyDBError: ChonkyDB request failed for POST "
                    "/v1/external/records/bulk (status=429)"
                ),
                "remote_write_context": {
                    "operation": "store_records_bulk",
                    "request_path": "/v1/external/records/bulk",
                    "request_execution_mode": "async",
                },
                "exception_chain": [
                    {
                        "type": "LongTermRemoteUnavailableError",
                        "detail": "Failed to persist fine-grained remote long-term memory items.",
                    },
                    {
                        "type": "ChonkyDBError",
                        "detail": "ChonkyDB request failed for POST /v1/external/records/bulk (status=429)",
                        "status_code": 429,
                        "response_json": {
                            "detail": "queue_saturated",
                            "error": "queue_saturated",
                        },
                    },
                ],
            }
            successful_retry_payload = {
                "status": "ok",
                "ready": True,
                "report_path": "/twinr/artifacts/reports/retention_live_canary/retry.json",
            }

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                rendered = " ".join(command)
                if "sha256sum /twinr/.env" in rendered:
                    return _completed(command, stdout="")
                if "repo_attestation_manifest_path" in rendered:
                    return _completed(command, stdout=_repo_attestation_stdout())
                if "importlib.import_module" in rendered:
                    return _completed(command, stdout=_import_contract_stdout())
                if "ActiveState,SubState,UnitFileState,MainPID,ExecMainStatus" in rendered:
                    return _completed(
                        command,
                        stdout='[{"name": "twinr-runtime-supervisor.service", "active_state": "active", "sub_state": "running", "unit_file_state": "enabled", "main_pid": "102", "exec_main_status": "0"}]\n',
                    )
                return _completed(command)

            with mock.patch(
                "twinr.ops.pi_runtime_deploy._run_retention_canary_probe",
                side_effect=[
                    RetentionCanaryProbeError("retention canary failed", payload=failed_payload),
                    successful_retry_payload,
                ],
            ) as probe_mock, mock.patch(
                "twinr.ops.pi_runtime_deploy._diagnose_retention_canary_host_contention",
                return_value={
                    "available": True,
                    "contention_detected": True,
                    "contention_signals": ["public_query_unhealthy", "backend_query_unhealthy"],
                },
            ), mock.patch(
                "twinr.ops.pi_runtime_deploy._stabilize_retention_canary_host",
                return_value={
                    "ok": True,
                    "diagnosis": "public_recovered_after_host_stabilization_and_backend_repair",
                    "backend_repair": {"ok": True, "action_taken": "restart_backend_service"},
                },
            ), mock.patch(
                "twinr.ops.pi_runtime_deploy._wait_for_remote_watchdog_ready",
                return_value={
                    "ready": True,
                    "detail": "watchdog_ready",
                    "sample_captured_at": "2026-04-05T13:48:42Z",
                    "sample_fresh_after_gate": True,
                },
            ) as watchdog_wait_mock, mock.patch(
                "twinr.ops.pi_runtime_deploy.time.time_ns",
                return_value=1,
            ):
                result = deploy_pi_runtime(
                    project_root=root,
                    pi_env_path=pi_env_path,
                    services=("twinr-runtime-supervisor",),
                    subprocess_runner=_runner,
                    mirror_watchdog=mirror,
                    install_editable=False,
                    install_systemd_units=False,
                    verify_env_contract=False,
                    verify_retention_canary=True,
                    progress_callback=progress_events.append,
                )

        self.assertTrue(result.ok)
        assert result.retention_canary is not None
        recovery = result.retention_canary.get("host_contention_recovery")
        self.assertIsInstance(recovery, dict)
        assert isinstance(recovery, dict)
        self.assertEqual(
            recovery["post_repair_watchdog_readiness"]["sample_captured_at"],
            "2026-04-05T13:48:42Z",
        )
        self.assertEqual(probe_mock.call_count, 2)
        watchdog_wait_mock.assert_called_once()
        wait_kwargs = watchdog_wait_mock.call_args.kwargs
        self.assertEqual(wait_kwargs["remote_root"], "/twinr")
        self.assertEqual(wait_kwargs["env_path"], "/twinr/.env")
        self.assertRegex(wait_kwargs["min_sample_captured_at"], r"^\d{4}-\d{2}-\d{2}T.*Z$")
        retention_events = [
            event
            for event in progress_events
            if event.get("phase") == "retention_canary"
        ]
        self.assertTrue(
            any(event.get("step") == "post_repair_watchdog_readiness" for event in retention_events)
        )

    def test_deploy_skips_retry_when_post_repair_watchdog_never_republishes_ready_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(f"OPENAI_API_KEY={_TEST_OPENAI_API_KEY}\n", encoding="utf-8")
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            _init_git_repo(root)
            mirror = _FakeMirrorWatchdog()

            failed_payload = {
                "status": "failed",
                "ready": False,
                "failure_stage": "seed_retention_objects",
                "error_message": "LongTermRemoteUnavailableError: Failed to persist fine-grained remote long-term memory items.",
                "root_cause_message": (
                    "ChonkyDBError: ChonkyDB request failed for POST "
                    "/v1/external/records/bulk (status=429)"
                ),
                "remote_write_context": {
                    "operation": "store_records_bulk",
                    "request_path": "/v1/external/records/bulk",
                    "request_execution_mode": "async",
                },
                "exception_chain": [
                    {
                        "type": "LongTermRemoteUnavailableError",
                        "detail": "Failed to persist fine-grained remote long-term memory items.",
                    },
                    {
                        "type": "ChonkyDBError",
                        "detail": "ChonkyDB request failed for POST /v1/external/records/bulk (status=429)",
                        "status_code": 429,
                        "response_json": {
                            "detail": "queue_saturated",
                            "error": "queue_saturated",
                        },
                    },
                ],
            }

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                rendered = " ".join(command)
                if "sha256sum /twinr/.env" in rendered:
                    return _completed(command, stdout="")
                if "repo_attestation_manifest_path" in rendered:
                    return _completed(command, stdout=_repo_attestation_stdout())
                if "importlib.import_module" in rendered:
                    return _completed(command, stdout=_import_contract_stdout())
                if "ActiveState,SubState,UnitFileState,MainPID,ExecMainStatus" in rendered:
                    return _completed(
                        command,
                        stdout='[{"name": "twinr-runtime-supervisor.service", "active_state": "active", "sub_state": "running", "unit_file_state": "enabled", "main_pid": "102", "exec_main_status": "0"}]\n',
                    )
                return _completed(command)

            with mock.patch(
                "twinr.ops.pi_runtime_deploy._run_retention_canary_probe",
                side_effect=RetentionCanaryProbeError("retention canary failed", payload=failed_payload),
            ) as probe_mock, mock.patch(
                "twinr.ops.pi_runtime_deploy._diagnose_retention_canary_host_contention",
                return_value={
                    "available": True,
                    "contention_detected": True,
                    "contention_signals": ["public_query_unhealthy", "backend_query_unhealthy"],
                },
            ), mock.patch(
                "twinr.ops.pi_runtime_deploy._stabilize_retention_canary_host",
                return_value={
                    "ok": True,
                    "diagnosis": "public_recovered_after_host_stabilization_and_backend_repair",
                    "backend_repair": {"ok": True, "action_taken": "restart_backend_service"},
                },
            ), mock.patch(
                "twinr.ops.pi_runtime_deploy._wait_for_remote_watchdog_ready",
                return_value={
                    "ready": False,
                    "detail": "watchdog_not_ready",
                    "sample_captured_at": "2026-04-05T13:46:46Z",
                    "sample_fresh_after_gate": False,
                },
            ):
                with self.assertRaises(PiRuntimeDeployError) as exc_info:
                    deploy_pi_runtime(
                        project_root=root,
                        pi_env_path=pi_env_path,
                        services=("twinr-runtime-supervisor",),
                        subprocess_runner=_runner,
                        mirror_watchdog=mirror,
                        install_editable=False,
                        install_systemd_units=False,
                        verify_env_contract=False,
                        verify_retention_canary=True,
                    )

        self.assertEqual(exc_info.exception.phase, "retention_canary")
        self.assertEqual(probe_mock.call_count, 1)
        self.assertIn("fresh ready sample", str(exc_info.exception))

    def test_deploy_skips_host_stabilization_when_retention_failure_is_not_contention_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(f"OPENAI_API_KEY={_TEST_OPENAI_API_KEY}\n", encoding="utf-8")
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            _init_git_repo(root)
            mirror = _FakeMirrorWatchdog()

            failed_payload = {
                "status": "failed",
                "ready": False,
                "failure_stage": "fresh_reader_load_current_state_fine_grained",
                "error_message": "LongTermRemoteUnavailableError: readback failed on a fresh reader.",
            }

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                rendered = " ".join(command)
                if "sha256sum /twinr/.env" in rendered:
                    return _completed(command, stdout="")
                if "repo_attestation_manifest_path" in rendered:
                    return _completed(command, stdout=_repo_attestation_stdout())
                if "importlib.import_module" in rendered:
                    return _completed(command, stdout=_import_contract_stdout())
                if "ActiveState,SubState,UnitFileState,MainPID,ExecMainStatus" in rendered:
                    return _completed(
                        command,
                        stdout='[{"name": "twinr-runtime-supervisor.service", "active_state": "active", "sub_state": "running", "unit_file_state": "enabled", "main_pid": "102", "exec_main_status": "0"}]\n',
                    )
                return _completed(command)

            with mock.patch(
                "twinr.ops.pi_runtime_deploy._run_retention_canary_probe",
                side_effect=RetentionCanaryProbeError("retention canary failed", payload=failed_payload),
            ), mock.patch(
                "twinr.ops.pi_runtime_deploy._diagnose_retention_canary_host_contention",
                return_value={
                    "available": True,
                    "contention_detected": True,
                    "contention_signals": ["backend_query_unhealthy"],
                },
            ) as diagnose_mock, mock.patch(
                "twinr.ops.pi_runtime_deploy._stabilize_retention_canary_host",
            ) as stabilize_mock:
                with self.assertRaises(PiRuntimeDeployError) as exc_info:
                    deploy_pi_runtime(
                        project_root=root,
                        pi_env_path=pi_env_path,
                        services=("twinr-runtime-supervisor",),
                        subprocess_runner=_runner,
                        mirror_watchdog=mirror,
                        install_editable=False,
                        install_systemd_units=False,
                        verify_env_contract=False,
                        verify_retention_canary=True,
                    )

        self.assertEqual(exc_info.exception.phase, "retention_canary")
        diagnose_mock.assert_called_once()
        stabilize_mock.assert_not_called()

    def test_default_deploy_includes_enabled_optional_pi_services(self) -> None:
        commands: list[list[str]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(f"OPENAI_API_KEY={_TEST_OPENAI_API_KEY}\n", encoding="utf-8")
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
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
            _init_git_repo(root)
            mirror = _FakeMirrorWatchdog()

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                rendered = " ".join(command)
                if "sha256sum /twinr/.env" in rendered:
                    return _completed(command, stdout="")
                if "is-enabled" in rendered:
                    return _completed(
                        command,
                        stdout=(
                            '[{"name": "twinr-remote-memory-watchdog.service", "state": "enabled", "returncode": 0},'
                            ' {"name": "twinr-runtime-supervisor.service", "state": "enabled", "returncode": 0},'
                            ' {"name": "twinr-web.service", "state": "enabled", "returncode": 0},'
                            ' {"name": "twinr-whatsapp-channel.service", "state": "enabled", "returncode": 0}]\n'
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
                if "repo_attestation_manifest_path" in rendered:
                    return _completed(command, stdout=_repo_attestation_stdout())
                if "importlib.import_module" in rendered:
                    return _completed(command, stdout=_import_contract_stdout())
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
            env_path.write_text(f"OPENAI_API_KEY={_TEST_OPENAI_API_KEY}\n", encoding="utf-8")
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
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
            _init_git_repo(root)
            mirror = _FakeMirrorWatchdog()

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                rendered = " ".join(command)
                if "sha256sum /twinr/.env" in rendered:
                    return _completed(command, stdout="")
                if "is-enabled" in rendered:
                    return _completed(
                        command,
                        stdout=(
                            '[{"name": "twinr-remote-memory-watchdog.service", "state": "enabled", "returncode": 0},'
                            ' {"name": "twinr-runtime-supervisor.service", "state": "enabled", "returncode": 0},'
                            ' {"name": "twinr-web.service", "state": "enabled", "returncode": 0},'
                            ' {"name": "twinr-whatsapp-channel.service", "state": "disabled", "returncode": 1}]\n'
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
                if "repo_attestation_manifest_path" in rendered:
                    return _completed(command, stdout=_repo_attestation_stdout())
                if "importlib.import_module" in rendered:
                    return _completed(command, stdout=_import_contract_stdout())
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

    def test_default_deploy_includes_linked_optional_pi_unit(self) -> None:
        commands: list[list[str]] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(f"OPENAI_API_KEY={_TEST_OPENAI_API_KEY}\n", encoding="utf-8")
            pi_env_path = root / ".env.pi"
            pi_env_path.write_text(
                '\n'.join(
                    (
                        f'PI_HOST="{_TEST_PI_HOST}"',
                        f'PI_SSH_USER="{_TEST_PI_SSH_USER}"',
                        f'PI_SSH_PW="{_TEST_PI_SSH_PASSWORD}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            ops_root = root / "hardware" / "ops"
            ops_root.mkdir(parents=True, exist_ok=True)
            (ops_root / "twinr-runtime-supervisor.service").write_text(
                "[Install]\nWantedBy=multi-user.target\n[Service]\nWorkingDirectory=/twinr\nExecStart=/usr/bin/env bash -lc 'cd /twinr && run'\n",
                encoding="utf-8",
            )
            (ops_root / "twinr-remote-memory-watchdog.service").write_text(
                "[Install]\nWantedBy=multi-user.target\n[Service]\nWorkingDirectory=/twinr\nExecStart=/usr/bin/env bash -lc 'cd /twinr && run'\n",
                encoding="utf-8",
            )
            (ops_root / "twinr-web.service").write_text(
                "[Install]\nWantedBy=multi-user.target\n[Service]\nWorkingDirectory=/twinr\nExecStart=/usr/bin/env bash -lc 'cd /twinr && run'\n",
                encoding="utf-8",
            )
            (ops_root / "twinr-whatsapp-channel.service").write_text(
                "[Install]\nWantedBy=multi-user.target\n[Service]\nWorkingDirectory=/twinr\nExecStart=/twinr/.venv/bin/python -m twinr --env-file .env --run-whatsapp-channel\n",
                encoding="utf-8",
            )
            _init_git_repo(root)
            mirror = _FakeMirrorWatchdog()

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                rendered = " ".join(command)
                if "sha256sum /twinr/.env" in rendered:
                    return _completed(command, stdout="")
                if "is-enabled" in rendered:
                    return _completed(
                        command,
                        stdout=(
                            '[{"name": "twinr-remote-memory-watchdog.service", "state": "enabled", "returncode": 0},'
                            ' {"name": "twinr-runtime-supervisor.service", "state": "enabled", "returncode": 0},'
                            ' {"name": "twinr-web.service", "state": "enabled", "returncode": 0},'
                            ' {"name": "twinr-whatsapp-channel.service", "state": "linked", "returncode": 1}]\n'
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
                if "repo_attestation_manifest_path" in rendered:
                    return _completed(command, stdout=_repo_attestation_stdout())
                if "importlib.import_module" in rendered:
                    return _completed(command, stdout=_import_contract_stdout())
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


if __name__ == "__main__":
    unittest.main()
