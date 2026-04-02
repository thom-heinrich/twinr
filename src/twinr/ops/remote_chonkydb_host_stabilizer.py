"""Stabilize the dedicated Twinr ChonkyDB host under shared-host contention.

Purpose
-------
Use this module when the public Twinr ChonkyDB endpoint is reachable but
becomes slow or unstable because unrelated host workloads reclaim CPU and I/O
from the dedicated backend service. The stabilizer raises the backend service
weight and quiesces a curated set of non-Twinr CAIA system units plus
user-session units that were proven to starve or directly share the dedicated
Twinr backend during live incidents.

Outputs
-------
- one JSON-serializable result object with public probe latency before/after
  the stabilization action, using the public `/instance` contract rather than
  the stricter current-scope query-surface canary
- the backend service CPU/IO weights after stabilization
- the before/after enabled/active state for the curated system- and user-unit
  conflict sets
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any, Mapping

from twinr.ops.remote_chonkydb_repair import (
    ChonkyDBHttpProbeResult,
    RemoteChonkyDBExecutor,
    RemoteChonkyDBOpsSettings,
    _probe_http_json,
    load_remote_chonkydb_ops_settings,
)


_DEFAULT_SETTLE_S = 8.0
_DEFAULT_CHONKY_PROPERTIES: dict[str, str] = {
    "CPUWeight": "10000",
    "StartupCPUWeight": "10000",
    "IOWeight": "10000",
    "StartupIOWeight": "10000",
}
_DEFAULT_KILLSWITCH_PATHS = (
    "/home/thh/Desktop/tessairact_local/tessairact/state/systemd_agent/killswitch_codex_traces_pipeline",
    "/home/thh/Desktop/tessairact_local/tessairact/state/systemd_agent/killswitch_user_policy_loop",
)
_DEFAULT_SYSTEM_CONFLICTING_UNITS = (
    "caia-codex-traces-pipeline.service",
    "caia-codex-traces-pipeline.timer",
    "caia-artifacts-stores-backup.service",
    "caia-artifacts-stores-backup.timer",
    "caia-code-graph-refresh.service",
    "caia-code-graph-refresh.timer",
    "caia-code-graph-refresh.path",
    "caia-agent-user-policy-loop.service",
    "caia-agent-user-policy-loop.timer",
    "caia-agent-user-simulation-loop.service",
    "caia-agent-user-simulation-loop.timer",
    "caia-agent-user-preferences.service",
    "caia-agent-user-preferences.timer",
    "caia-agent-repo-rollup-12h.service",
    "caia-agent-repo-rollup-12h.timer",
    "caia-agent-repo-rollup-15m.service",
    "caia-agent-repo-rollup-15m.timer",
    "caia-agent-repo-rollup-1d.service",
    "caia-agent-repo-rollup-1d.timer",
    "caia-agent-repo-rollup-2h.service",
    "caia-agent-repo-rollup-2h.timer",
    "caia-agent-repo-rollup-6h.service",
    "caia-agent-repo-rollup-6h.timer",
    "caia-agent-repo-rollup-freshness-watchdog.service",
    "caia-agent-repo-rollup-freshness-watchdog.timer",
    "caia-artifact-memory-ingest.service",
    "caia-repo-script-indexer.service",
    "caia-agent-audit.service",
    "caia-agent-audit-checkpoint.timer",
    "caia-agent-audit-insights.timer",
    "caia-agent-audit-prune.timer",
    "caia-agent-audit-watchdog.timer",
    "caia-wiki-sphinx-doc-agent.service",
    "caia-wiki-doc-maintainer.timer",
    "caia-wiki-doc-jobs-cleanup.timer",
    "caia-code-change-review.timer",
    "caia-chonky-transformer.service",
    "caia-chonky-transformer.timer",
    "caia-bug-blackbook-refresh.service",
    "caia-bug-blackbook-refresh.timer",
    "caia-bug-memory-refresh.service",
    "caia-bug-memory-refresh.timer",
    "caia-ops-chonkbin-writer-matrix-canary.service",
    "caia-ops-chonkbin-writer-matrix-canary.timer",
    "caia-portal-kg-agent-autopilot.service",
    "caia-portal-llm-worker.service",
    "caia-repo-script-llm-enricher.service",
)
_DEFAULT_USER_CONFLICTING_UNITS = (
    "caia-chonkycode-chunks-posttransform.service",
    "caia-chonkycode-chunks-posttransform.timer",
    "caia-chonkycode-chunks-posttransform.path",
    "caia-chonkycode-meta-graph-ssot.service",
    "caia-chonkycode-meta-graph-ssot.timer",
)
_REMOTE_UNIT_STATE_CODE = """
import json
import os
import pwd
import subprocess
import sys


def _user_env(owner: str) -> dict[str, str]:
    values = dict(os.environ)
    resolved_owner = owner or values.get("USER", "")
    uid = pwd.getpwnam(resolved_owner).pw_uid
    values.setdefault("XDG_RUNTIME_DIR", f"/run/user/{uid}")
    values.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{uid}/bus")
    return values


def _run(scope: str, owner: str, *args: str) -> subprocess.CompletedProcess[str]:
    command = ["systemctl"]
    environment = None
    if scope == "user":
        command.append("--user")
        environment = _user_env(owner)
    command.extend(list(args))
    return subprocess.run(command, capture_output=True, text=True, check=False, env=environment)


def _enabled_state(scope: str, owner: str, unit: str) -> str:
    completed = _run(scope, owner, "is-enabled", unit)
    value = str(completed.stdout or completed.stderr or "").strip()
    return value or "unknown"


def _show_map(scope: str, owner: str, unit: str) -> dict[str, str]:
    completed = _run(
        scope,
        owner,
        "show",
        unit,
        "--no-pager",
        "-p",
        "LoadState",
        "-p",
        "ActiveState",
        "-p",
        "SubState",
        "-p",
        "Result",
    )
    values: dict[str, str] = {}
    for raw_line in str(completed.stdout or "").splitlines():
        key, separator, value = raw_line.partition("=")
        if separator:
            values[str(key).strip()] = str(value).strip()
    return values


payload = json.loads(sys.stdin.read() or "{}")
scope = str(payload.get("scope", "system") or "system").strip().lower()
owner = str(payload.get("user_unit_owner", "") or "").strip()
units = tuple(str(item).strip() for item in payload.get("units", []) if str(item).strip())
result = []
for unit in units:
    show = _show_map(scope, owner, unit)
    result.append(
        {
            "scope": scope,
            "unit": unit,
            "enabled_state": _enabled_state(scope, owner, unit),
            "load_state": str(show.get("LoadState", "")).strip(),
            "active_state": str(show.get("ActiveState", "")).strip(),
            "sub_state": str(show.get("SubState", "")).strip(),
            "result": str(show.get("Result", "")).strip(),
        }
    )
json.dump({"units": result}, sys.stdout)
"""
_REMOTE_SERVICE_PROPERTIES_CODE = """
import json
import subprocess
import sys

payload = json.loads(sys.stdin.read() or "{}")
service_name = str(payload.get("service_name", "")).strip()
properties = tuple(str(item).strip() for item in payload.get("properties", []) if str(item).strip())
command = ["systemctl", "show", service_name, "--no-pager"]
for key in properties:
    command.extend(["-p", key])
completed = subprocess.run(command, capture_output=True, text=True, check=False)
values: dict[str, str] = {}
for raw_line in str(completed.stdout or "").splitlines():
    name, separator, value = raw_line.partition("=")
    if separator:
        values[str(name).strip()] = str(value).strip()
json.dump({"properties": values}, sys.stdout)
"""
_REMOTE_STABILIZE_HOST_CODE = """
import json
import os
import pwd
from pathlib import Path
import subprocess
import sys

payload = json.loads(sys.stdin.read() or "{}")
backend_service = str(payload.get("backend_service", "")).strip()
system_units = tuple(str(item).strip() for item in payload.get("system_units", []) if str(item).strip())
user_units = tuple(str(item).strip() for item in payload.get("user_units", []) if str(item).strip())
user_unit_owner = str(payload.get("user_unit_owner", "") or "").strip()
kill_switch_paths = tuple(
    str(item).strip() for item in payload.get("kill_switch_paths", []) if str(item).strip()
)
property_map = {
    str(key).strip(): str(value).strip()
    for key, value in dict(payload.get("property_assignments") or {}).items()
    if str(key).strip()
}

for raw_path in kill_switch_paths:
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

if backend_service and property_map:
    command = ["systemctl", "set-property", backend_service]
    for key, value in property_map.items():
        command.append(f"{key}={value}")
    subprocess.run(command, check=True)

disabled_system_units = []
for unit in system_units:
    subprocess.run(["systemctl", "disable", "--now", unit], capture_output=True, text=True, check=False)
    disabled_system_units.append(unit)


def _run_user_systemctl(*args: str) -> subprocess.CompletedProcess[str]:
    uid = pwd.getpwnam(user_unit_owner).pw_uid
    command = [
        "runuser",
        "-u",
        user_unit_owner,
        "--",
        "env",
        f"XDG_RUNTIME_DIR=/run/user/{uid}",
        f"DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/{uid}/bus",
        "systemctl",
        "--user",
        *list(args),
    ]
    return subprocess.run(command, capture_output=True, text=True, check=False)


disabled_user_units = []
for unit in user_units:
    _run_user_systemctl("disable", "--now", unit)
    disabled_user_units.append(unit)

json.dump(
    {
        "kill_switch_paths": list(kill_switch_paths),
        "disabled_system_units": disabled_system_units,
        "disabled_user_units": disabled_user_units,
        "property_assignments": property_map,
    },
    sys.stdout,
)
"""


@dataclass(frozen=True, slots=True)
class RemoteHostUnitState:
    """Describe the enabled and runtime state of one remote systemd unit."""

    scope: str
    unit: str
    enabled_state: str
    load_state: str
    active_state: str
    sub_state: str
    result: str


@dataclass(frozen=True, slots=True)
class RemoteChonkyDBHostStabilizationResult:
    """Summarize one host-stabilization run for the Twinr backend."""

    ok: bool
    diagnosis: str
    elapsed_s: float
    public_before: ChonkyDBHttpProbeResult
    public_after: ChonkyDBHttpProbeResult
    backend_service: str
    backend_properties: Mapping[str, str]
    system_units_before: tuple[RemoteHostUnitState, ...]
    system_units_after: tuple[RemoteHostUnitState, ...]
    user_units_before: tuple[RemoteHostUnitState, ...]
    user_units_after: tuple[RemoteHostUnitState, ...]
    kill_switch_paths: tuple[str, ...]
    disabled_system_units: tuple[str, ...]
    disabled_user_units: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation of the stabilization result."""

        return {
            "ok": self.ok,
            "diagnosis": self.diagnosis,
            "elapsed_s": self.elapsed_s,
            "public_before": asdict(self.public_before),
            "public_after": asdict(self.public_after),
            "backend_service": self.backend_service,
            "backend_properties": dict(self.backend_properties),
            "system_units_before": [asdict(item) for item in self.system_units_before],
            "system_units_after": [asdict(item) for item in self.system_units_after],
            "user_units_before": [asdict(item) for item in self.user_units_before],
            "user_units_after": [asdict(item) for item in self.user_units_after],
            "kill_switch_paths": list(self.kill_switch_paths),
            "disabled_system_units": list(self.disabled_system_units),
            "disabled_user_units": list(self.disabled_user_units),
        }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for remote-host stabilization."""

    parser = argparse.ArgumentParser(
        description=(
            "Prioritize Twinr's dedicated remote ChonkyDB backend by disabling a curated set "
            "of conflicting shared-host CAIA jobs and raising the backend CPU/IO weights."
        ),
    )
    project_root = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--env-file",
        type=Path,
        default=project_root / ".env",
        help="Twinr runtime env file that defines the public ChonkyDB contract.",
    )
    parser.add_argument(
        "--ops-env-file",
        type=Path,
        default=project_root / ".env.chonkydb",
        help="Operator env file that defines backend SSH and service provenance.",
    )
    parser.add_argument(
        "--probe-timeout-s",
        type=float,
        default=20.0,
        help="Timeout in seconds for individual public probes.",
    )
    parser.add_argument(
        "--ssh-timeout-s",
        type=float,
        default=60.0,
        help="Timeout in seconds for individual backend SSH commands.",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=_DEFAULT_SETTLE_S,
        help="Seconds to wait after disabling conflicts before probing again.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the host-stabilization workflow and print JSON."""

    args = build_parser().parse_args(argv)
    settings = load_remote_chonkydb_ops_settings(
        env_file=args.env_file,
        ops_env_file=args.ops_env_file,
    )
    result = stabilize_remote_chonkydb_host(
        settings=settings,
        probe_timeout_s=args.probe_timeout_s,
        ssh_timeout_s=args.ssh_timeout_s,
        settle_s=args.settle_s,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False))
    return 0 if result.ok else 1


def stabilize_remote_chonkydb_host(
    *,
    settings: RemoteChonkyDBOpsSettings,
    probe_timeout_s: float,
    ssh_timeout_s: float,
    settle_s: float,
    executor: RemoteChonkyDBExecutor | None = None,
    subprocess_runner: Any = subprocess.run,
) -> RemoteChonkyDBHostStabilizationResult:
    """Quiesce shared-host system/user conflict units and prioritize the backend."""

    started = time.perf_counter()
    if executor is None:
        executor = RemoteChonkyDBExecutor(
            settings=settings.ssh,
            subprocess_runner=subprocess_runner,
            timeout_s=ssh_timeout_s,
        )
    public_before = probe_public_host_availability(
        settings=settings,
        timeout_s=probe_timeout_s,
    )
    system_units_before = fetch_remote_unit_states(
        executor=executor,
        units=_DEFAULT_SYSTEM_CONFLICTING_UNITS,
        scope="system",
        user_unit_owner=settings.ssh.user,
    )
    user_units_before = fetch_remote_unit_states(
        executor=executor,
        units=_DEFAULT_USER_CONFLICTING_UNITS,
        scope="user",
        user_unit_owner=settings.ssh.user,
    )
    action = apply_remote_host_stabilization(
        executor=executor,
        backend_service=settings.backend_service,
        system_units=_DEFAULT_SYSTEM_CONFLICTING_UNITS,
        user_units=_DEFAULT_USER_CONFLICTING_UNITS,
        user_unit_owner=settings.ssh.user,
        kill_switch_paths=_DEFAULT_KILLSWITCH_PATHS,
        property_assignments=_DEFAULT_CHONKY_PROPERTIES,
    )
    if settle_s > 0:
        time.sleep(float(settle_s))
    public_after = probe_public_host_availability(
        settings=settings,
        timeout_s=probe_timeout_s,
    )
    system_units_after = fetch_remote_unit_states(
        executor=executor,
        units=_DEFAULT_SYSTEM_CONFLICTING_UNITS,
        scope="system",
        user_unit_owner=settings.ssh.user,
    )
    user_units_after = fetch_remote_unit_states(
        executor=executor,
        units=_DEFAULT_USER_CONFLICTING_UNITS,
        scope="user",
        user_unit_owner=settings.ssh.user,
    )
    backend_properties = fetch_remote_service_properties(
        executor=executor,
        service_name=settings.backend_service,
        properties=tuple(_DEFAULT_CHONKY_PROPERTIES),
    )
    elapsed_s = round(time.perf_counter() - started, 3)
    diagnosis = "public_ready_after_host_stabilization"
    if not public_before.ready and public_after.ready:
        diagnosis = "public_recovered_after_host_stabilization"
    elif not public_after.ready:
        diagnosis = "public_still_unhealthy_after_host_stabilization"
    return RemoteChonkyDBHostStabilizationResult(
        ok=public_after.ready,
        diagnosis=diagnosis,
        elapsed_s=elapsed_s,
        public_before=public_before,
        public_after=public_after,
        backend_service=settings.backend_service,
        backend_properties=backend_properties,
        system_units_before=system_units_before,
        system_units_after=system_units_after,
        user_units_before=user_units_before,
        user_units_after=user_units_after,
        kill_switch_paths=action["kill_switch_paths"],
        disabled_system_units=action["disabled_system_units"],
        disabled_user_units=action["disabled_user_units"],
    )


def probe_public_host_availability(
    *,
    settings: RemoteChonkyDBOpsSettings,
    timeout_s: float,
) -> ChonkyDBHttpProbeResult:
    """Probe the public `/instance` surface for host-availability recovery.

    The host stabilizer is about reclaiming CPU/I/O for Twinr's dedicated
    backend. A strict current-scope query canary belongs to
    `remote_chonkydb_repair.py`, where contract-level remote-memory readiness is
    diagnosed separately. Here we only need a stable public liveness probe that
    does not false-fail because one namespace/scope has no current head yet.
    """

    return _probe_http_json(
        label="public",
        url=settings.public_base_url.rstrip("/") + "/v1/external/instance",
        headers={
            "Accept": "application/json",
            settings.public_api_key_header: settings.public_api_key,
        },
        timeout_s=timeout_s,
    )


def fetch_remote_unit_states(
    *,
    executor: RemoteChonkyDBExecutor,
    units: tuple[str, ...],
    scope: str,
    user_unit_owner: str,
) -> tuple[RemoteHostUnitState, ...]:
    """Return enabled/runtime state for one curated remote systemd scope."""

    payload = _run_remote_python_json(
        executor=executor,
        code=_REMOTE_UNIT_STATE_CODE,
        payload={
            "units": list(units),
            "scope": scope,
            "user_unit_owner": user_unit_owner,
        },
        use_sudo=False,
    )
    raw_items = payload.get("units")
    if not isinstance(raw_items, list):  # pragma: no cover - defensive guard
        raise RuntimeError(f"invalid_remote_unit_state_payload:{payload!r}")
    states: list[RemoteHostUnitState] = []
    for item in raw_items:
        if not isinstance(item, dict):  # pragma: no cover - defensive guard
            raise RuntimeError(f"invalid_remote_unit_state_item:{item!r}")
        states.append(
            RemoteHostUnitState(
                scope=str(item.get("scope", scope)).strip() or scope,
                unit=str(item.get("unit", "")).strip(),
                enabled_state=str(item.get("enabled_state", "")).strip(),
                load_state=str(item.get("load_state", "")).strip(),
                active_state=str(item.get("active_state", "")).strip(),
                sub_state=str(item.get("sub_state", "")).strip(),
                result=str(item.get("result", "")).strip(),
            )
        )
    return tuple(states)


def fetch_remote_service_properties(
    *,
    executor: RemoteChonkyDBExecutor,
    service_name: str,
    properties: tuple[str, ...],
) -> dict[str, str]:
    """Fetch a small property map for one remote service."""

    payload = _run_remote_python_json(
        executor=executor,
        code=_REMOTE_SERVICE_PROPERTIES_CODE,
        payload={
            "service_name": service_name,
            "properties": list(properties),
        },
        use_sudo=False,
    )
    raw_map = payload.get("properties")
    if not isinstance(raw_map, dict):  # pragma: no cover - defensive guard
        raise RuntimeError(f"invalid_remote_service_properties_payload:{payload!r}")
    return {str(key).strip(): str(value).strip() for key, value in raw_map.items()}


def apply_remote_host_stabilization(
    *,
    executor: RemoteChonkyDBExecutor,
    backend_service: str,
    system_units: tuple[str, ...],
    user_units: tuple[str, ...],
    user_unit_owner: str,
    kill_switch_paths: tuple[str, ...],
    property_assignments: Mapping[str, str],
) -> dict[str, tuple[str, ...]]:
    """Touch kill-switches, raise backend weights, and disable conflict units."""

    payload = _run_remote_python_json(
        executor=executor,
        code=_REMOTE_STABILIZE_HOST_CODE,
        payload={
            "backend_service": backend_service,
            "system_units": list(system_units),
            "user_units": list(user_units),
            "user_unit_owner": user_unit_owner,
            "kill_switch_paths": list(kill_switch_paths),
            "property_assignments": dict(property_assignments),
        },
        use_sudo=True,
    )
    raw_paths = payload.get("kill_switch_paths")
    raw_system_units = payload.get("disabled_system_units")
    raw_user_units = payload.get("disabled_user_units")
    if (
        not isinstance(raw_paths, list)
        or not isinstance(raw_system_units, list)
        or not isinstance(raw_user_units, list)
    ):  # pragma: no cover
        raise RuntimeError(f"invalid_remote_stabilize_payload:{payload!r}")
    return {
        "kill_switch_paths": tuple(str(item).strip() for item in raw_paths if str(item).strip()),
        "disabled_system_units": tuple(
            str(item).strip() for item in raw_system_units if str(item).strip()
        ),
        "disabled_user_units": tuple(
            str(item).strip() for item in raw_user_units if str(item).strip()
        ),
    }


def _run_remote_python_json(
    *,
    executor: RemoteChonkyDBExecutor,
    code: str,
    payload: Mapping[str, object],
    use_sudo: bool,
) -> dict[str, object]:
    """Run one tiny remote Python helper and decode the JSON response."""

    input_text = json.dumps(payload, ensure_ascii=False)
    if use_sudo:
        # `run_sudo_ssh()` cannot pass a second stdin payload after the password,
        # so re-run through a sudo shell command that forwards JSON via a heredoc.
        completed = executor.run_sudo_ssh(
            "python3 - <<'PY'\n"
            "import subprocess\n"
            "import sys\n"
            f"code = {code!r}\n"
            f"payload = {input_text!r}\n"
            "completed = subprocess.run(\n"
            "    ['python3', '-c', code],\n"
            "    input=payload,\n"
            "    text=True,\n"
            "    capture_output=True,\n"
            "    check=False,\n"
            ")\n"
            "sys.stdout.write(completed.stdout)\n"
            "sys.stderr.write(completed.stderr)\n"
            "raise SystemExit(completed.returncode)\n"
            "PY"
        )
        stdout = str(completed.stdout or "").strip()
    else:
        completed = executor.run_ssh(
            "python3 -c " + shlex.quote(code),
            input_text=input_text,
        )
        stdout = str(completed.stdout or "").strip()
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(f"invalid_remote_json:{exc}:stdout={stdout!r}") from exc
    if not isinstance(data, dict):  # pragma: no cover - defensive guard
        raise RuntimeError(f"invalid_remote_json_type:{type(data).__name__}")
    return data
