"""Protect remote systemd services from accidental manual restarts.

Twinr's dedicated remote ChonkyDB backend is a required-remote dependency.
If an operator or agent restarts the systemd unit mid-request, Twinr sees a
hard 503 and must fail closed. This helper installs a persistent drop-in that
refuses manual starts/stops by default and opens a temporary override only for
an explicitly controlled restart path.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import shlex
import subprocess
from typing import Protocol


_PROTECTED_DROPIN_NAME = "10-refuse-manual-restart.conf"
_TEMP_OVERRIDE_DROPIN_NAME = (
    "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz-temp-manual-restart-override.conf"
)
_PROTECTED_DROPIN_CONTENT = "[Unit]\nRefuseManualStart=yes\nRefuseManualStop=yes\n"
_TEMP_OVERRIDE_DROPIN_CONTENT = "[Unit]\nRefuseManualStart=no\nRefuseManualStop=no\n"
_RESTART_STOP_WAIT_S = 12.0
_RESTART_KILL_WAIT_S = 8.0


class RemoteSudoExecutor(Protocol):
    """Minimal executor contract for remote sudo shell commands."""

    def run_sudo_ssh(self, script: str) -> subprocess.CompletedProcess[str]:
        """Run one remote shell script under sudo."""


@dataclass(frozen=True, slots=True)
class RemoteManualRestartProtectionStatus:
    """Describe the enforced manual-restart protection for one remote unit."""

    service_name: str
    protected_dropin_path: str
    protection_changed: bool
    refuse_manual_start: bool
    refuse_manual_stop: bool

    @property
    def verified(self) -> bool:
        """Return whether the remote systemd manager now refuses manual restarts."""

        return self.refuse_manual_start and self.refuse_manual_stop


def ensure_remote_service_manual_restart_protection(
    *,
    executor: RemoteSudoExecutor,
    service_name: str,
) -> RemoteManualRestartProtectionStatus:
    """Install and verify the persistent manual-restart protection drop-in."""

    removed_conflicting_overrides = _remove_conflicting_manual_restart_overrides(
        executor=executor,
        service_name=service_name,
    )
    protected_path = _dropin_path(
        service_name=service_name,
        dropin_name=_PROTECTED_DROPIN_NAME,
    )
    changed = _sync_remote_dropin(
        executor=executor,
        dropin_path=protected_path,
        contents=_PROTECTED_DROPIN_CONTENT,
    )
    flags = _fetch_manual_restart_flags(
        executor=executor,
        service_name=service_name,
    )
    status = RemoteManualRestartProtectionStatus(
        service_name=service_name,
        protected_dropin_path=protected_path,
        protection_changed=changed or removed_conflicting_overrides,
        refuse_manual_start=flags["RefuseManualStart"],
        refuse_manual_stop=flags["RefuseManualStop"],
    )
    if not status.verified:
        raise RuntimeError(
            "remote_manual_restart_protection_not_verified:"
            f"service={service_name}:"
            f"refuse_manual_start={status.refuse_manual_start}:"
            f"refuse_manual_stop={status.refuse_manual_stop}"
        )
    return status


def guarded_restart_remote_service(
    *,
    executor: RemoteSudoExecutor,
    service_name: str,
) -> None:
    """Restart one protected remote unit via a temporary allow-manual override.

    A plain ``systemctl restart`` can trap Twinr in a long outage when the
    dedicated ChonkyDB service hangs in ``stop-sigterm`` while shutting down its
    docid mapping. Use a bounded stop/kill/start sequence instead so operator
    repairs fail fast instead of burning the full systemd stop timeout.
    """

    temp_override_path = _dropin_path(
        service_name=service_name,
        dropin_name=_TEMP_OVERRIDE_DROPIN_NAME,
    )
    _sync_remote_dropin(
        executor=executor,
        dropin_path=temp_override_path,
        contents=_TEMP_OVERRIDE_DROPIN_CONTENT,
    )
    try:
        completed = executor.run_sudo_ssh(
            _python_bounded_restart_script(
                service_name=service_name,
                graceful_stop_wait_s=_RESTART_STOP_WAIT_S,
                kill_wait_s=_RESTART_KILL_WAIT_S,
            )
        )
        _raise_for_remote_command_failure(
            completed=completed,
            failure_prefix="remote_guarded_restart_failed",
        )
    finally:
        _remove_remote_dropin(
            executor=executor,
            dropin_path=temp_override_path,
        )


def _dropin_path(*, service_name: str, dropin_name: str) -> str:
    """Return the absolute drop-in path for one remote service."""

    return f"/etc/systemd/system/{service_name}.d/{dropin_name}"


def _remove_conflicting_manual_restart_overrides(
    *,
    executor: RemoteSudoExecutor,
    service_name: str,
) -> bool:
    """Remove stale allow-manual overrides that still defeat the protected unit.

    The persistent protection lives under ``/etc``, but previous repair runs or
    external operator tooling can leave a runtime drop-in under ``/run`` that
    reopens manual starts/stops. As long as those temporary ``RefuseManual*=no``
    overrides exist, ``systemctl show`` quite correctly reports the unit as
    unprotected and the repair flow fail-closes forever. Clean those temporary
    overrides first, then verify the durable protected state.
    """

    completed = executor.run_sudo_ssh(
        _python_conflicting_override_cleanup_script(service_name=service_name)
    )
    payload = _parse_json_stdout(completed.stdout)
    return bool(payload.get("changed", False))


def _sync_remote_dropin(
    *,
    executor: RemoteSudoExecutor,
    dropin_path: str,
    contents: str,
) -> bool:
    """Write one remote drop-in only when its contents changed."""

    completed = executor.run_sudo_ssh(
        _python_dropin_sync_script(
            dropin_path=dropin_path,
            contents=contents,
        )
    )
    payload = _parse_json_stdout(completed.stdout)
    return bool(payload.get("changed", False))


def _remove_remote_dropin(
    *,
    executor: RemoteSudoExecutor,
    dropin_path: str,
) -> bool:
    """Remove one remote drop-in and reload systemd only when it existed."""

    completed = executor.run_sudo_ssh(
        _python_dropin_remove_script(dropin_path=dropin_path)
    )
    payload = _parse_json_stdout(completed.stdout)
    return bool(payload.get("changed", False))


def _fetch_manual_restart_flags(
    *,
    executor: RemoteSudoExecutor,
    service_name: str,
) -> dict[str, bool]:
    """Return the live RefuseManualStart/Stop flags from remote systemd."""

    completed = executor.run_sudo_ssh(
        "systemctl show "
        + shlex.quote(service_name)
        + " --no-pager -p RefuseManualStart -p RefuseManualStop"
    )
    values: dict[str, bool] = {
        "RefuseManualStart": False,
        "RefuseManualStop": False,
    }
    for raw_line in str(completed.stdout or "").splitlines():
        key, separator, value = raw_line.partition("=")
        if not separator:
            continue
        normalized = str(value or "").strip().lower()
        values[str(key).strip()] = normalized in {"1", "yes", "true"}
    return values


def _parse_json_stdout(stdout: str) -> dict[str, object]:
    """Decode a tiny JSON payload returned from the remote helper snippets."""

    try:
        payload = json.loads(str(stdout or "").strip())
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(f"invalid_remote_json:{exc}:stdout={stdout!r}") from exc
    if not isinstance(payload, dict):  # pragma: no cover - defensive guard
        raise RuntimeError(f"invalid_remote_json_type:{type(payload).__name__}")
    return payload


def _raise_for_remote_command_failure(
    *,
    completed: subprocess.CompletedProcess[str],
    failure_prefix: str,
) -> None:
    """Raise a compact error when one remote command exits non-zero."""

    if int(completed.returncode) == 0:
        return
    stdout = str(completed.stdout or "").strip()
    stderr = str(completed.stderr or "").strip()
    raise RuntimeError(
        f"{failure_prefix}:returncode={completed.returncode}:stdout={stdout!r}:stderr={stderr!r}"
    )


def _python_dropin_sync_script(*, dropin_path: str, contents: str) -> str:
    """Return the remote Python snippet that syncs one drop-in file."""

    return (
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        "import json\n"
        "import os\n"
        "import subprocess\n"
        "import tempfile\n"
        f"path = Path({dropin_path!r})\n"
        f"contents = {contents!r}\n"
        "path.parent.mkdir(parents=True, exist_ok=True)\n"
        "changed = True\n"
        "try:\n"
        "    current = path.read_text(encoding='utf-8')\n"
        "except FileNotFoundError:\n"
        "    current = None\n"
        "if current == contents:\n"
        "    changed = False\n"
        "else:\n"
        "    fd, tmp_name = tempfile.mkstemp(prefix=path.name + '.', dir=str(path.parent), text=True)\n"
        "    os.close(fd)\n"
        "    tmp_path = Path(tmp_name)\n"
        "    try:\n"
        "        tmp_path.write_text(contents, encoding='utf-8')\n"
        "        os.chmod(tmp_path, 0o644)\n"
        "        os.replace(tmp_path, path)\n"
        "    finally:\n"
        "        if tmp_path.exists():\n"
        "            tmp_path.unlink()\n"
        "if changed:\n"
        "    subprocess.run(['systemctl', 'daemon-reload'], check=True)\n"
        "print(json.dumps({'changed': changed, 'path': str(path)}))\n"
        "PY"
    )


def _python_bounded_restart_script(
    *,
    service_name: str,
    graceful_stop_wait_s: float,
    kill_wait_s: float,
) -> str:
    """Return the remote Python snippet for a bounded stop/kill/start bounce."""

    return (
        "python3 - <<'PY'\n"
        "import json\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        f"service_name = {service_name!r}\n"
        f"graceful_stop_wait_s = float({float(graceful_stop_wait_s)!r})\n"
        f"kill_wait_s = float({float(kill_wait_s)!r})\n"
        "poll_interval_s = 0.5\n"
        "\n"
        "def _show() -> dict[str, str]:\n"
        "    completed = subprocess.run(\n"
        "        [\n"
        "            'systemctl',\n"
        "            'show',\n"
        "            service_name,\n"
        "            '--no-pager',\n"
        "            '-p', 'ActiveState',\n"
        "            '-p', 'SubState',\n"
        "            '-p', 'Result',\n"
        "            '-p', 'MainPID',\n"
        "        ],\n"
        "        capture_output=True,\n"
        "        text=True,\n"
        "        check=False,\n"
        "    )\n"
        "    values: dict[str, str] = {}\n"
        "    for raw_line in str(completed.stdout or '').splitlines():\n"
        "        key, separator, value = raw_line.partition('=')\n"
        "        if separator:\n"
        "            values[str(key).strip()] = str(value).strip()\n"
        "    return values\n"
        "\n"
        "def _main_pid(state: dict[str, str]) -> int:\n"
        "    raw_value = str(state.get('MainPID', '0') or '0').strip()\n"
        "    try:\n"
        "        return int(raw_value)\n"
        "    except ValueError:\n"
        "        return 0\n"
        "\n"
        "def _stopped(state: dict[str, str]) -> bool:\n"
        "    return str(state.get('ActiveState', '')).strip() in {'inactive', 'failed'} and _main_pid(state) <= 0\n"
        "\n"
        "def _wait_stopped(timeout_s: float) -> tuple[bool, dict[str, str]]:\n"
        "    deadline = time.monotonic() + max(0.0, float(timeout_s))\n"
        "    latest = _show()\n"
        "    while True:\n"
        "        if _stopped(latest):\n"
        "            return True, latest\n"
        "        if time.monotonic() >= deadline:\n"
        "            return False, latest\n"
        "        time.sleep(poll_interval_s)\n"
        "        latest = _show()\n"
        "\n"
        "actions: list[str] = []\n"
        "state = _show()\n"
        "if str(state.get('ActiveState', '')).strip() in {'active', 'activating', 'reloading', 'deactivating'} or _main_pid(state) > 0:\n"
        "    stop_completed = subprocess.run(\n"
        "        ['systemctl', 'stop', service_name],\n"
        "        capture_output=True,\n"
        "        text=True,\n"
        "        check=False,\n"
        "    )\n"
        "    actions.append('stop')\n"
        "    if stop_completed.returncode != 0:\n"
        "        json.dump(\n"
        "            {\n"
        "                'ok': False,\n"
        "                'phase': 'stop_failed',\n"
        "                'actions': actions,\n"
        "                'state': _show(),\n"
        "                'stdout': str(stop_completed.stdout or ''),\n"
        "                'stderr': str(stop_completed.stderr or ''),\n"
        "            },\n"
        "            sys.stdout,\n"
        "        )\n"
        "        raise SystemExit(int(stop_completed.returncode) or 1)\n"
        "    stopped, state = _wait_stopped(graceful_stop_wait_s)\n"
        "    if not stopped:\n"
        "        kill_completed = subprocess.run(\n"
        "            ['systemctl', 'kill', '--kill-who=all', '--signal=SIGKILL', service_name],\n"
        "            capture_output=True,\n"
        "            text=True,\n"
        "            check=False,\n"
        "        )\n"
        "        actions.append('kill_all_sigkill')\n"
        "        if kill_completed.returncode != 0:\n"
        "            json.dump(\n"
        "                {\n"
        "                    'ok': False,\n"
        "                    'phase': 'kill_failed',\n"
        "                    'actions': actions,\n"
        "                    'state': _show(),\n"
        "                    'stdout': str(kill_completed.stdout or ''),\n"
        "                    'stderr': str(kill_completed.stderr or ''),\n"
        "                },\n"
        "                sys.stdout,\n"
        "            )\n"
        "            raise SystemExit(int(kill_completed.returncode) or 1)\n"
        "        stopped, state = _wait_stopped(kill_wait_s)\n"
        "        if not stopped:\n"
        "            json.dump(\n"
        "                {\n"
        "                    'ok': False,\n"
        "                    'phase': 'kill_timeout',\n"
        "                    'actions': actions,\n"
        "                    'state': state,\n"
        "                },\n"
        "                sys.stdout,\n"
        "            )\n"
        "            raise SystemExit(1)\n"
        "if str(state.get('Result', '')).strip() == 'failed' or str(state.get('ActiveState', '')).strip() == 'failed':\n"
        "    reset_completed = subprocess.run(\n"
        "        ['systemctl', 'reset-failed', service_name],\n"
        "        capture_output=True,\n"
        "        text=True,\n"
        "        check=False,\n"
        "    )\n"
        "    actions.append('reset_failed')\n"
        "    if reset_completed.returncode != 0:\n"
        "        json.dump(\n"
        "            {\n"
        "                'ok': False,\n"
        "                'phase': 'reset_failed_error',\n"
        "                'actions': actions,\n"
        "                'state': _show(),\n"
        "                'stdout': str(reset_completed.stdout or ''),\n"
        "                'stderr': str(reset_completed.stderr or ''),\n"
        "            },\n"
        "            sys.stdout,\n"
        "        )\n"
        "        raise SystemExit(int(reset_completed.returncode) or 1)\n"
        "start_completed = subprocess.run(\n"
        "    ['systemctl', 'start', service_name],\n"
        "    capture_output=True,\n"
        "    text=True,\n"
        "    check=False,\n"
        ")\n"
        "actions.append('start')\n"
        "payload = {\n"
        "    'ok': start_completed.returncode == 0,\n"
        "    'phase': 'start',\n"
        "    'actions': actions,\n"
        "    'state': _show(),\n"
        "    'stdout': str(start_completed.stdout or ''),\n"
        "    'stderr': str(start_completed.stderr or ''),\n"
        "}\n"
        "json.dump(payload, sys.stdout)\n"
        "raise SystemExit(int(start_completed.returncode))\n"
        "PY"
    )


def _python_dropin_remove_script(*, dropin_path: str) -> str:
    """Return the remote Python snippet that removes one drop-in file."""

    return (
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        "import json\n"
        "import os\n"
        "import subprocess\n"
        f"path = Path({dropin_path!r})\n"
        "changed = False\n"
        "if path.exists():\n"
        "    path.unlink()\n"
        "    changed = True\n"
        "if changed:\n"
        "    subprocess.run(['systemctl', 'daemon-reload'], check=True)\n"
        "print(json.dumps({'changed': changed, 'path': str(path)}))\n"
        "PY"
    )


def _python_conflicting_override_cleanup_script(*, service_name: str) -> str:
    """Return the remote Python snippet that removes stale allow-manual overrides."""

    return (
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        "import json\n"
        "import subprocess\n"
        f"service_name = {service_name!r}\n"
        "dropin_dirs = [\n"
        "    Path('/run/systemd/system') / f'{service_name}.d',\n"
        "    Path('/etc/systemd/system') / f'{service_name}.d',\n"
        "]\n"
        "changed = False\n"
        "removed_paths = []\n"
        "for dropin_dir in dropin_dirs:\n"
        "    if not dropin_dir.is_dir():\n"
        "        continue\n"
        "    for path in sorted(dropin_dir.glob('*.conf')):\n"
        "        try:\n"
        "            contents = path.read_text(encoding='utf-8')\n"
        "        except UnicodeDecodeError:\n"
        "            continue\n"
        "        if 'RefuseManualStart=no' not in contents or 'RefuseManualStop=no' not in contents:\n"
        "            continue\n"
        "        path.unlink()\n"
        "        removed_paths.append(str(path))\n"
        "        changed = True\n"
        "if changed:\n"
        "    subprocess.run(['systemctl', 'daemon-reload'], check=True)\n"
        "print(json.dumps({'changed': changed, 'removed_paths': removed_paths}))\n"
        "PY"
    )
