"""Bootstrap the Pi-side self_coding Codex runtime from the leading repo.

This module owns the reproducible operator workflow that prepares `/twinr` for
SDK-backed self_coding compiles: install system prerequisites, sync the pinned
bridge files, copy the local Codex auth/config into the Pi user's codex home,
and run the explicit Twinr self-test remotely.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any

_SubprocessRunner = Any
_CODEX_VERSION_PATTERN = re.compile(r"codex-cli\s+([0-9][0-9A-Za-z.\-]+)")


@dataclass(frozen=True, slots=True)
class PiConnectionSettings:
    """Hold the SSH credentials used for the Pi acceptance instance."""

    host: str
    user: str
    password: str


@dataclass(frozen=True, slots=True)
class PiBootstrapResult:
    """Summarize one Pi bootstrap run for operator logging and scripts."""

    ready: bool
    host: str
    remote_root: str
    codex_cli_version: str
    self_test_output: str


def load_pi_connection_settings(env_path: str | Path) -> PiConnectionSettings:
    """Load `PI_HOST`, `PI_SSH_USER`, and `PI_SSH_PW` from a dotenv file."""

    values = _read_env_values(Path(env_path))
    host = str(values.get("PI_HOST", "")).strip()
    user = str(values.get("PI_SSH_USER", "")).strip()
    password = str(values.get("PI_SSH_PW", "")).strip()
    if not host:
        raise ValueError("PI_HOST is missing from the Pi env file")
    if not user:
        raise ValueError("PI_SSH_USER is missing from the Pi env file")
    if not password:
        raise ValueError("PI_SSH_PW is missing from the Pi env file")
    return PiConnectionSettings(host=host, user=user, password=password)


def bootstrap_self_coding_pi(
    *,
    project_root: str | Path,
    pi_env_path: str | Path,
    remote_root: str = "/twinr",
    local_codex_home: str | Path | None = None,
    subprocess_runner: _SubprocessRunner = subprocess.run,
) -> PiBootstrapResult:
    """Install Pi prerequisites and prove the remote self_coding Codex path.

    Args:
        project_root: Leading-repo root on the development machine.
        pi_env_path: Path to `.env.pi`.
        remote_root: Runtime checkout root on the Pi.
        local_codex_home: Optional override for the source Codex home.
        subprocess_runner: Injectable subprocess runner for tests.

    Returns:
        A normalized bootstrap result with the installed CLI version and the
        remote self-test output.
    """

    root = Path(project_root).resolve()
    settings = load_pi_connection_settings(pi_env_path)
    codex_home = _default_codex_home() if local_codex_home is None else Path(local_codex_home).expanduser()
    auth_file = codex_home / "auth.json"
    config_file = codex_home / "config.toml"
    if not auth_file.exists() or not auth_file.is_file():
        raise ValueError(f"local Codex auth file is missing: {auth_file}")
    if not config_file.exists() or not config_file.is_file():
        raise ValueError(f"local Codex config file is missing: {config_file}")

    bridge_root = root / "src" / "twinr" / "agent" / "self_coding" / "codex_driver" / "sdk_bridge"
    if not bridge_root.exists() or not bridge_root.is_dir():
        raise ValueError(f"local SDK bridge directory is missing: {bridge_root}")
    if not (bridge_root / "package.json").exists():
        raise ValueError(f"local SDK bridge package.json is missing: {bridge_root / 'package.json'}")

    codex_version_output = _run_local(
        ["codex", "--version"],
        subprocess_runner=subprocess_runner,
    )
    codex_cli_version = _parse_codex_cli_version(codex_version_output.stdout)

    env = _sshpass_env(settings.password)
    remote_spec = f"{settings.user}@{settings.host}"
    remote_bridge_root = f"{remote_root.rstrip('/')}/src/twinr/agent/self_coding/codex_driver/sdk_bridge/"
    remote_env_file = f"{remote_root.rstrip('/')}/.env"
    remote_python = f"{remote_root.rstrip('/')}/.venv/bin/python"

    _run_local(
        [
            "sshpass",
            "-e",
            "rsync",
            "-az",
            "--delete",
            "-e",
            "ssh -o StrictHostKeyChecking=no",
            f"{bridge_root}/",
            f"{remote_spec}:{remote_bridge_root}",
        ],
        subprocess_runner=subprocess_runner,
        env=env,
    )
    _run_local(
        [
            "sshpass",
            "-e",
            "scp",
            "-o",
            "StrictHostKeyChecking=no",
            str(auth_file),
            str(config_file),
            f"{remote_spec}:/tmp/",
        ],
        subprocess_runner=subprocess_runner,
        env=env,
    )
    _run_local(
        [
            "sshpass",
            "-e",
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            remote_spec,
            _remote_install_command(
                codex_cli_version=codex_cli_version,
                remote_root=remote_root,
            ),
        ],
        subprocess_runner=subprocess_runner,
        env=env,
    )
    _run_local(
        [
            "sshpass",
            "-e",
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            remote_spec,
            _remote_auth_sync_command(user=settings.user),
        ],
        subprocess_runner=subprocess_runner,
        env=env,
    )
    self_test = _run_local(
        [
            "sshpass",
            "-e",
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            remote_spec,
            _remote_self_test_command(
                remote_root=remote_root,
                remote_python=remote_python,
                remote_env_file=remote_env_file,
            ),
        ],
        subprocess_runner=subprocess_runner,
        env=env,
    )
    output = (self_test.stdout or self_test.stderr or "").strip()
    return PiBootstrapResult(
        ready=True,
        host=settings.host,
        remote_root=remote_root,
        codex_cli_version=codex_cli_version,
        self_test_output=output,
    )


def _run_local(
    args: list[str],
    *,
    subprocess_runner: _SubprocessRunner,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess_runner(
        args,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        env=env,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip() or f"command failed: {' '.join(args)}"
        raise RuntimeError(message)
    return completed


def _sshpass_env(password: str) -> dict[str, str]:
    env = dict(os.environ)
    env["SSHPASS"] = password
    return env


def _default_codex_home() -> Path:
    configured = os.environ.get("CODEX_HOME", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".codex"


def _parse_codex_cli_version(text: str) -> str:
    match = _CODEX_VERSION_PATTERN.search(str(text or ""))
    if match is None:
        raise RuntimeError(f"could not parse codex version from: {text!r}")
    return match.group(1)


def _remote_install_command(
    *,
    codex_cli_version: str,
    remote_root: str,
) -> str:
    remote_root_q = shlex.quote(remote_root.rstrip("/"))
    remote_bridge_q = shlex.quote(f"{remote_root.rstrip('/')}/src/twinr/agent/self_coding/codex_driver/sdk_bridge")
    version_q = shlex.quote(codex_cli_version)
    return (
        "set -euo pipefail; "
        "if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; "
        "then sudo apt-get update && sudo apt-get install -y nodejs npm; fi; "
        f"sudo npm install -g @openai/codex@{version_q}; "
        f"cd {remote_bridge_q}; npm ci; "
        f"cd {remote_root_q}"
    )


def _remote_auth_sync_command(*, user: str) -> str:
    home_q = shlex.quote(f"/home/{user}")
    return (
        "set -euo pipefail; "
        f"install -d -m 700 {home_q}/.codex; "
        f"mv /tmp/auth.json {home_q}/.codex/auth.json; "
        f"mv /tmp/config.toml {home_q}/.codex/config.toml; "
        f"chmod 600 {home_q}/.codex/auth.json {home_q}/.codex/config.toml"
    )


def _remote_self_test_command(
    *,
    remote_root: str,
    remote_python: str,
    remote_env_file: str,
) -> str:
    remote_root_q = shlex.quote(remote_root.rstrip("/"))
    remote_python_q = shlex.quote(remote_python)
    remote_env_file_q = shlex.quote(remote_env_file)
    return (
        "set -euo pipefail; "
        f"cd {remote_root_q}; PYTHONPATH=src {remote_python_q} -m twinr --env-file {remote_env_file_q} "
        "--self-coding-codex-self-test --self-coding-live-auth-check"
    )


def _read_env_values(path: Path) -> dict[str, str]:
    if not path.exists() or not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8-sig")
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = _parse_env_value(raw_value)
    return values


def _parse_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        return ""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value


__all__ = [
    "PiBootstrapResult",
    "PiConnectionSettings",
    "bootstrap_self_coding_pi",
    "load_pi_connection_settings",
]
