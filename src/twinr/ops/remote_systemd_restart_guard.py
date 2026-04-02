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
    """Restart one protected remote unit via a temporary allow-manual override."""

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
        executor.run_sudo_ssh("systemctl restart " + shlex.quote(service_name))
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
