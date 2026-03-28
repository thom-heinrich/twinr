from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.ops.pi_repo_mirror import PiRepoMirrorCycleResult
from twinr.ops.pi_runtime_deploy import deploy_pi_runtime
from twinr.ops.pi_runtime_deploy_remote import (
    PiRemoteExecutor,
    install_editable_package,
    verify_python_import_contract,
)
from twinr.ops.self_coding_pi import load_pi_connection_settings
from twinr.ops.venv_system_site_bridge import ensure_pi_system_site_packages_bridge
from twinr.ops.venv_wrapper_repair import repair_venv_python_shebangs

_TEST_PI_HOST = "192.0.2.10"
_TEST_PI_SSH_USER = "pi-test-user"
_TEST_PI_SSH_PASSWORD = "placeholder-password"
_TEST_OPENAI_API_KEY = "placeholder-openai-key"
_REPO_ROOT = Path(__file__).resolve().parents[1]
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
)


def _completed(
    args: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr=stderr)


def _import_contract_stdout(
    modules: tuple[str, ...] = _TEST_PI_IMPORT_MODULES,
    *,
    python_path: str = "/twinr/.venv/bin/python",
) -> str:
    return json.dumps(
        {
            "python_path": python_path,
            "checked_modules": list(modules),
            "imported_modules": list(modules),
            "failed_imports": {},
            "elapsed_s": 0.123,
        }
    ) + "\n"


def _normalized_requirement_name(requirement: str) -> str:
    token = re.split(r"[<>=!~;\[\]\s]", str(requirement).strip(), maxsplit=1)[0]
    return token.strip().lower().replace("_", "-")


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


class PiRuntimeDeployTests(unittest.TestCase):
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

    def test_repair_venv_python_shebangs_rewrites_only_stale_venv_wrappers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bin_dir = Path(temp_dir) / ".venv" / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
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
            self.assertEqual(result.checked_files, 3)
            self.assertEqual(result.rewritten_files, 1)
            self.assertEqual(result.sample_paths, ("pytest",))
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
        assert result.import_contract is not None
        self.assertEqual(result.import_contract.checked_modules, _TEST_PI_IMPORT_MODULES)
        self.assertEqual(result.env_contract, {"ok": True, "detail": "ready"})
        self.assertEqual(mirror.last_call["apply_sync"], True)
        self.assertEqual(mirror.last_call["checksum"], True)
        joined = "\n".join(" ".join(command) for command in commands)
        self.assertIn("sshpass -e scp", joined)
        self.assertIn("pip install --no-deps -e", joined)
        self.assertIn("repair_venv_python_shebangs", joined)
        self.assertIn("systemctl daemon-reload", joined)
        self.assertIn("systemctl restart", joined)
        self.assertIn("--live-text", joined)
        self.assertTrue(any(env and env.get("SSHPASS") == _TEST_PI_SSH_PASSWORD for env in envs))
        assert result.editable_install_summary is not None
        self.assertIn("normalized 1 stale venv wrapper shebang", result.editable_install_summary)
        self.assertIn("bridged 1 Pi system site-package path into the venv", result.editable_install_summary)

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
        self.assertIn("verified 2 venv wrapper shebang(s)", summary)
        self.assertIn("verified Pi system site-package bridge for 1 path", summary)

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
        self.assertIn("importlib.import_module", joined)

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
            mirror = _FakeMirrorWatchdog()

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                rendered = " ".join(command)
                if "sha256sum /twinr/.env" in rendered:
                    return _completed(command, stdout="")
                if "runtime_requirements.txt" in rendered:
                    return _completed(command, stdout="Successfully installed playwright\n")
                if "playwright install" in rendered:
                    return _completed(command, stdout="Downloaded Chromium\n")
                if "pip install --no-deps -e" in rendered:
                    return _completed(command, stdout="Successfully installed twinr\n")
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
        self.assertIn("pip install -r \"$requirements_path\"", joined)
        self.assertIn("-m playwright install", joined)

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
            mirror = _FakeMirrorWatchdog()

            def _runner(args, **kwargs):
                command = [str(part) for part in args]
                commands.append(command)
                rendered = " ".join(command)
                if "sha256sum /twinr/.env" in rendered:
                    return _completed(command, stdout=f"{local_sha}\n")
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
        self.assertNotIn("sshpass -e scp", joined)

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

    def test_default_deploy_repairs_masked_optional_pi_unit_with_existing_enable_link(self) -> None:
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
                            '[{"name": "twinr-remote-memory-watchdog.service", "unit_file_state": "enabled", "install_link_present": true},'
                            ' {"name": "twinr-runtime-supervisor.service", "unit_file_state": "enabled", "install_link_present": true},'
                            ' {"name": "twinr-web.service", "unit_file_state": "enabled", "install_link_present": true},'
                            ' {"name": "twinr-whatsapp-channel.service", "unit_file_state": "masked", "install_link_present": true}]\n'
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
