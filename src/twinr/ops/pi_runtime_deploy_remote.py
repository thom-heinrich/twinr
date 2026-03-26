"""Remote execution and verification helpers for Pi runtime deploys.

This module owns the SSH/SCP-side primitives used by the operator-facing Pi
deploy flow. Besides core repo/env/systemd steps, it also installs optional
runtime support files that live inside mirrored local workspaces such as the
ignored ``browser_automation/`` tree.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any, Sequence

from twinr.ops.self_coding_pi import PiConnectionSettings


_SubprocessRunner = Any


@dataclass(frozen=True, slots=True)
class PiSyncedFileResult:
    """Describe one authoritative file sync from the leading repo to the Pi."""

    local_path: str
    remote_path: str
    sha256: str
    changed: bool
    backup_path: str | None


@dataclass(frozen=True, slots=True)
class PiSystemdServiceState:
    """Summarize one productive service state on the Pi."""

    name: str
    active_state: str
    sub_state: str
    unit_file_state: str
    main_pid: int | None
    exec_main_status: int | None
    healthy: bool


class PiRemoteExecutor:
    """Run bounded SSH/SCP commands against the Pi acceptance host."""

    def __init__(
        self,
        *,
        settings: PiConnectionSettings,
        subprocess_runner: _SubprocessRunner,
        timeout_s: float,
    ) -> None:
        self.settings = settings
        self._subprocess_runner = subprocess_runner
        self.timeout_s = timeout_s

    @property
    def remote_spec(self) -> str:
        """Return the canonical ``user@host`` SSH target string."""

        return f"{self.settings.user}@{self.settings.host}"

    def run_ssh(self, script: str) -> subprocess.CompletedProcess[str]:
        """Run one remote bash script over SSH and capture UTF-8 text output."""

        return self._run_local(
            [
                "sshpass",
                "-e",
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "ConnectTimeout=10",
                self.remote_spec,
                "bash -lc " + shlex.quote("set -euo pipefail; " + script),
            ]
        )

    def run_scp(self, local_path: Path, remote_path: str) -> subprocess.CompletedProcess[str]:
        """Copy one local file to the Pi over SCP."""

        return self._run_local(
            [
                "sshpass",
                "-e",
                "scp",
                "-o",
                "StrictHostKeyChecking=no",
                str(local_path),
                f"{self.remote_spec}:{remote_path}",
            ]
        )

    def _run_local(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        completed = self._subprocess_runner(
            args,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=sshpass_env(self.settings.password),
            timeout=self.timeout_s,
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "").strip()
            if not message:
                message = f"command failed: {' '.join(args)}"
            raise RuntimeError(message)
        return completed


def sync_authoritative_file(
    *,
    remote: PiRemoteExecutor,
    local_path: Path,
    remote_path: str,
    mode: str,
) -> PiSyncedFileResult:
    """Copy one authoritative local file onto the Pi with checksum verification."""

    local_bytes = local_path.read_bytes()
    local_sha = hashlib.sha256(local_bytes).hexdigest()
    remote_sha = read_remote_sha256(remote=remote, remote_path=remote_path)
    if remote_sha == local_sha:
        return PiSyncedFileResult(
            local_path=str(local_path),
            remote_path=remote_path,
            sha256=local_sha,
            changed=False,
            backup_path=None,
        )

    remote_temp = f"/tmp/twinr-deploy-{time.time_ns()}-{local_path.name}"
    backup_suffix = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    remote.run_scp(local_path, remote_temp)
    completed = remote.run_ssh(
        "\n".join(
            (
                f"target={shlex.quote(remote_path)}",
                f"tmp={shlex.quote(remote_temp)}",
                f"expected_sha={shlex.quote(local_sha)}",
                f"mode={shlex.quote(mode)}",
                f"backup=\"$target.deploy-backup-{backup_suffix}\"",
                "if [ -e \"$target\" ]; then cp \"$target\" \"$backup\"; else backup=\"\"; fi",
                "install -D -m \"$mode\" \"$tmp\" \"$target\"",
                "rm -f \"$tmp\"",
                "actual_sha=$(sha256sum \"$target\" | awk '{print $1}')",
                "if [ \"$actual_sha\" != \"$expected_sha\" ]; then",
                "  echo \"remote checksum mismatch after sync\" >&2",
                "  exit 1",
                "fi",
                "printf '%s' \"$backup\"",
            )
        )
    )
    backup_path = (completed.stdout or "").strip() or None
    return PiSyncedFileResult(
        local_path=str(local_path),
        remote_path=remote_path,
        sha256=local_sha,
        changed=True,
        backup_path=backup_path,
    )


def read_remote_sha256(*, remote: PiRemoteExecutor, remote_path: str) -> str | None:
    """Return the remote file checksum if the file exists, otherwise ``None``."""

    completed = remote.run_ssh(
        f"if [ -f {shlex.quote(remote_path)} ]; then sha256sum {shlex.quote(remote_path)} | awk '{{print $1}}'; fi"
    )
    checksum = (completed.stdout or "").strip()
    return checksum or None


def install_editable_package(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    install_with_deps: bool,
) -> str:
    """Ensure the Pi venv exists and refresh the editable Twinr install."""

    remote_python = f"{remote_root}/.venv/bin/python"
    pip_args = "-e \"$remote_root\"" if install_with_deps else "--no-deps -e \"$remote_root\""
    completed = remote.run_ssh(
        "\n".join(
            (
                f"remote_root={shlex.quote(remote_root)}",
                f"remote_python={shlex.quote(remote_python)}",
                "if [ ! -x \"$remote_python\" ]; then python3 -m venv \"$remote_root/.venv\"; fi",
                f"\"$remote_python\" -m pip install {pip_args}",
            )
        )
    )
    return summarize_output(completed)


def install_browser_automation_runtime_support(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    install_python_requirements: bool,
    install_playwright_browsers: bool,
) -> str:
    """Install mirrored browser-automation runtime requirements on the Pi.

    Args:
        remote: Remote executor targeting the Pi acceptance host.
        remote_root: Mirrored Twinr checkout root on the Pi.
        install_python_requirements: Whether to install Python packages from
            ``browser_automation/runtime_requirements.txt``.
        install_playwright_browsers: Whether to install Playwright browser
            binaries listed in ``browser_automation/playwright_browsers.txt``.

    Returns:
        One compact summary of the remote install output.
    """

    remote_python = f"{remote_root}/.venv/bin/python"
    requirements_path = f"{remote_root}/browser_automation/runtime_requirements.txt"
    browsers_path = f"{remote_root}/browser_automation/playwright_browsers.txt"
    lines = [
        f"remote_root={shlex.quote(remote_root)}",
        f"remote_python={shlex.quote(remote_python)}",
        f"requirements_path={shlex.quote(requirements_path)}",
        f"browsers_path={shlex.quote(browsers_path)}",
        "if [ ! -x \"$remote_python\" ]; then python3 -m venv \"$remote_root/.venv\"; fi",
    ]
    if install_python_requirements:
        lines.extend(
            (
                'test -s "$requirements_path"',
                'echo "[browser_automation] installing python requirements"',
                '"$remote_python" -m pip install -r "$requirements_path"',
            )
        )
    if install_playwright_browsers:
        lines.extend(
            (
                'test -s "$browsers_path"',
                "browser_names=()",
                'while IFS= read -r raw_line; do',
                '  line="${raw_line%%#*}"',
                '  set -- $line',
                '  if [ "$#" -eq 0 ]; then',
                "    continue",
                "  fi",
                '  browser_names+=("$1")',
                'done < "$browsers_path"',
                'if [ "${#browser_names[@]}" -eq 0 ]; then',
                '  echo "playwright browser manifest did not contain any browser names" >&2',
                "  exit 1",
                "fi",
                'echo "[browser_automation] installing Playwright browsers: ${browser_names[*]}"',
                '"$remote_python" -m playwright install "${browser_names[@]}"',
            )
        )
    completed = remote.run_ssh("\n".join(lines))
    return summarize_output(completed)


def install_service_units(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    services: Sequence[str],
) -> None:
    """Install the productive service units from the mirrored repo onto the Pi."""

    lines = [f"remote_root={shlex.quote(remote_root)}"]
    for service_name in services:
        source_path = shlex.quote(f"{remote_root}/hardware/ops/{service_name}")
        target_path = shlex.quote(f"/etc/systemd/system/{service_name}")
        lines.extend(
            (
                f"test -f {source_path}",
                f"sudo install -m 644 {source_path} {target_path}",
            )
        )
    services_arg = " ".join(shlex.quote(name) for name in services)
    lines.extend(
        (
            "sudo systemctl daemon-reload",
            f"sudo systemctl enable {services_arg}",
        )
    )
    remote.run_ssh("\n".join(lines))


def restart_services(*, remote: PiRemoteExecutor, services: Sequence[str]) -> None:
    """Restart the selected productive services on the Pi."""

    services_arg = " ".join(shlex.quote(name) for name in services)
    remote.run_ssh(f"sudo systemctl restart {services_arg}")


def wait_for_services(
    *,
    remote: PiRemoteExecutor,
    services: Sequence[str],
    wait_timeout_s: float,
) -> tuple[PiSystemdServiceState, ...]:
    """Poll the productive services until they report healthy or time out."""

    deadline = time.monotonic() + wait_timeout_s
    latest_states: tuple[PiSystemdServiceState, ...] = ()
    while True:
        latest_states = load_service_states(remote=remote, services=services)
        if latest_states and all(state.healthy for state in latest_states):
            return latest_states
        if time.monotonic() >= deadline:
            break
        time.sleep(1.0)
    failing = [state for state in latest_states if not state.healthy]
    failing_names = ", ".join(state.name for state in failing) or ", ".join(services)
    journal_excerpt = load_journal_excerpt(remote=remote, service_name=failing[0].name if failing else services[0])
    raise RuntimeError(
        f"services did not become healthy within {wait_timeout_s:.1f}s: {failing_names}\n{journal_excerpt}"
    )


def load_service_states(
    *,
    remote: PiRemoteExecutor,
    services: Sequence[str],
) -> tuple[PiSystemdServiceState, ...]:
    """Load the current ``systemctl show`` snapshot for the requested services."""

    script = "\n".join(
        (
            "python3 - <<'PY'",
            "import json",
            "import subprocess",
            f"services = {json.dumps(list(services), ensure_ascii=False)}",
            "payload = []",
            "for name in services:",
            "    completed = subprocess.run(",
            "        ['systemctl', 'show', name, '--property=ActiveState,SubState,UnitFileState,MainPID,ExecMainStatus'],",
            "        check=False,",
            "        capture_output=True,",
            "        text=True,",
            "        encoding='utf-8',",
            "        errors='replace',",
            "    )",
            "    values = {}",
            "    for raw_line in completed.stdout.splitlines():",
            "        if '=' not in raw_line:",
            "            continue",
            "        key, value = raw_line.split('=', 1)",
            "        values[key] = value",
            "    payload.append(",
            "        {",
            "            'name': name,",
            "            'active_state': values.get('ActiveState', ''),",
            "            'sub_state': values.get('SubState', ''),",
            "            'unit_file_state': values.get('UnitFileState', ''),",
            "            'main_pid': values.get('MainPID', ''),",
            "            'exec_main_status': values.get('ExecMainStatus', ''),",
            "        }",
            "    )",
            "print(json.dumps(payload, ensure_ascii=False))",
            "PY",
        )
    )
    completed = remote.run_ssh(script)
    raw_payload = json.loads((completed.stdout or "[]").strip() or "[]")
    states: list[PiSystemdServiceState] = []
    for item in raw_payload:
        active_state = str(item.get("active_state", "")).strip()
        sub_state = str(item.get("sub_state", "")).strip()
        states.append(
            PiSystemdServiceState(
                name=str(item.get("name", "")).strip(),
                active_state=active_state,
                sub_state=sub_state,
                unit_file_state=str(item.get("unit_file_state", "")).strip(),
                main_pid=parse_optional_int(item.get("main_pid")),
                exec_main_status=parse_optional_int(item.get("exec_main_status")),
                healthy=active_state == "active" and sub_state == "running",
            )
        )
    return tuple(states)


def load_journal_excerpt(
    *,
    remote: PiRemoteExecutor,
    service_name: str,
    lines: int = 40,
) -> str:
    """Return a short recent journal excerpt for one failing service."""

    completed = remote.run_ssh(
        f"journalctl -u {shlex.quote(service_name)} -n {int(lines)} --no-pager --output cat || true"
    )
    text = (completed.stdout or completed.stderr or "").strip()
    return summarize_text(text)


def run_env_contract_probe(
    *,
    remote: PiRemoteExecutor,
    remote_root: str,
    env_path: str,
    live_text: str | None,
    live_search: str | None,
) -> dict[str, object]:
    """Run the bounded Pi env-contract probe and parse its JSON result."""

    remote_python = f"{remote_root}/.venv/bin/python"
    args = [
        shlex.quote(remote_python),
        shlex.quote(f"{remote_root}/hardware/ops/check_pi_openai_env_contract.py"),
        "--env-file",
        shlex.quote(env_path),
    ]
    if live_text is not None:
        args.extend(("--live-text", shlex.quote(live_text)))
    if live_search is not None:
        args.extend(("--live-search", shlex.quote(live_search)))
    completed = remote.run_ssh(" ".join(args))
    return json.loads((completed.stdout or "").strip() or "{}")


def parse_optional_int(value: object) -> int | None:
    """Normalize systemd numeric fields to ``int | None``."""

    text = str(value or "").strip()
    if not text:
        return None
    try:
        number = int(text)
    except ValueError:
        return None
    return number if number > 0 else None


def summarize_output(completed: subprocess.CompletedProcess[str]) -> str:
    """Return one compact summary string for a completed subprocess."""

    text = "\n".join(
        part.strip()
        for part in (completed.stdout or "", completed.stderr or "")
        if part and part.strip()
    )
    return summarize_text(text)


def summarize_text(text: str, *, max_lines: int = 12, max_chars: int = 1200) -> str:
    """Collapse multi-line command output into a bounded summary string."""

    normalized = "\n".join(line.rstrip() for line in str(text or "").splitlines() if line.strip())
    if not normalized:
        return ""
    lines = normalized.splitlines()
    if len(lines) > max_lines:
        normalized = "\n".join(lines[-max_lines:])
    if len(normalized) > max_chars:
        normalized = normalized[-max_chars:]
    return normalized


def sshpass_env(password: str) -> dict[str, str]:
    """Return the local environment with ``SSHPASS`` injected."""

    import os

    env = dict(os.environ)
    env["SSHPASS"] = password
    return env


__all__ = [
    "PiRemoteExecutor",
    "PiSyncedFileResult",
    "PiSystemdServiceState",
    "install_browser_automation_runtime_support",
    "install_editable_package",
    "install_service_units",
    "load_journal_excerpt",
    "load_service_states",
    "parse_optional_int",
    "read_remote_sha256",
    "restart_services",
    "run_env_contract_probe",
    "sshpass_env",
    "summarize_output",
    "summarize_text",
    "sync_authoritative_file",
    "wait_for_services",
]
