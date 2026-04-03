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
  the stabilization action, using the same public empty-scope-safe query
  surface probe as the repair helper instead of a weaker `/instance`-only
  liveness check
- the backend service CPU/IO weights after stabilization
- the before/after enabled/active state for the curated system- and user-unit
  conflict sets
- the bounded stale-process cleanup summary for proven long-running
  user-session benchmark workers that bypass systemd unit control
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
    load_remote_chonkydb_ops_settings,
    probe_public_chonkydb,
)


_DEFAULT_SETTLE_S = 8.0
_DEFAULT_SSH_TIMEOUT_S = 180.0
_DEFAULT_CHONKY_PROPERTIES: dict[str, str] = {
    "CPUWeight": "10000",
    "StartupCPUWeight": "10000",
    "IOWeight": "10000",
    "StartupIOWeight": "10000",
}
_DEFAULT_STALE_PROCESS_MIN_ELAPSED_S = 1800.0
_DEFAULT_KILLSWITCH_PATHS = (
    "/home/thh/Desktop/tessairact_local/tessairact/state/systemd_agent/killswitch_codex_traces_pipeline",
    "/home/thh/Desktop/tessairact_local/tessairact/state/systemd_agent/killswitch_user_policy_loop",
)
_DEFAULT_SYSTEM_CONFLICTING_UNITS = (
    "caia-consumer-portal.service",
    "caia-consumer-portal-demo.service",
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
    "caia-ops-chonky-search-guardrail.service",
    "caia-ops-chonky-search-guardrail.timer",
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
    "ollama-gpu.service",
)
_DEFAULT_STALE_PROCESS_RULES = (
    {
        "label": "code_graph_benchmark_runner",
        "required_substrings": (
            "benchmarks.code_graph.benchmark",
            "run_code_graph_benchmark",
        ),
    },
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
import signal
import subprocess
import sys
import time

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
stale_process_rules = tuple(
    item for item in payload.get("stale_process_rules", []) if isinstance(item, dict)
)
stale_process_min_elapsed_s = max(0.0, float(payload.get("stale_process_min_elapsed_s", 0.0) or 0.0))

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
    subprocess.run(["systemctl", "stop", unit], capture_output=True, text=True, check=False)
    subprocess.run(["systemctl", "disable", unit], capture_output=True, text=True, check=False)
    subprocess.run(
        ["systemctl", "mask", "--runtime", unit],
        capture_output=True,
        text=True,
        check=False,
    )
    subprocess.run(["systemctl", "reset-failed", unit], capture_output=True, text=True, check=False)
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
    _run_user_systemctl("stop", unit)
    _run_user_systemctl("disable", unit)
    _run_user_systemctl("mask", "--runtime", unit)
    _run_user_systemctl("reset-failed", unit)
    disabled_user_units.append(unit)


def _list_process_table() -> tuple[dict[int, dict[str, object]], dict[int, set[int]]]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,etimes=,args="],
        capture_output=True,
        text=True,
        check=False,
    )
    processes: dict[int, dict[str, object]] = {}
    children: dict[int, set[int]] = {}
    for raw_line in str(completed.stdout or "").splitlines():
        line = str(raw_line).strip()
        if not line:
            continue
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            elapsed_s = int(parts[2])
        except ValueError:
            continue
        args = str(parts[3]).strip()
        processes[pid] = {
            "pid": pid,
            "ppid": ppid,
            "elapsed_s": elapsed_s,
            "args": args,
        }
        children.setdefault(ppid, set()).add(pid)
    return processes, children


def _command_excerpt(raw_value: object, *, limit: int = 240) -> str:
    value = str(raw_value or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def _collect_stale_process_targets() -> list[dict[str, object]]:
    if not stale_process_rules or stale_process_min_elapsed_s <= 0:
        return []
    processes, children = _list_process_table()
    matched_roots: dict[int, dict[str, object]] = {}
    for process in processes.values():
        args = str(process.get("args", "") or "")
        elapsed_s = int(process.get("elapsed_s", 0) or 0)
        if elapsed_s < stale_process_min_elapsed_s:
            continue
        for rule in stale_process_rules:
            label = str(rule.get("label", "") or "").strip()
            required = tuple(
                str(item).strip()
                for item in rule.get("required_substrings", [])
                if str(item).strip()
            )
            if required and all(marker in args for marker in required):
                matched_roots[int(process["pid"])] = {
                    "match_label": label,
                    "required_substrings": required,
                }
                break
    if not matched_roots:
        return []
    target_pids = set(matched_roots)
    root_by_pid = {pid: pid for pid in matched_roots}
    queue = list(matched_roots)
    while queue:
        current = queue.pop()
        for child_pid in children.get(current, set()):
            if child_pid in target_pids:
                continue
            target_pids.add(child_pid)
            root_by_pid[child_pid] = root_by_pid[current]
            queue.append(child_pid)
    return [
        {
            "pid": pid,
            "ppid": int(processes.get(pid, {}).get("ppid", 0) or 0),
            "elapsed_s": int(processes.get(pid, {}).get("elapsed_s", 0) or 0),
            "command_excerpt": _command_excerpt(processes.get(pid, {}).get("args", "")),
            "scope": "matched_root" if pid in matched_roots else "descendant",
            "match_label": str(
                matched_roots.get(root_by_pid.get(pid, pid), {}).get("match_label", "") or ""
            ),
            "root_pid": int(root_by_pid.get(pid, pid)),
        }
        for pid in sorted(target_pids)
    ]


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_stale_processes() -> list[dict[str, object]]:
    targets = _collect_stale_process_targets()
    if not targets:
        return []
    by_pid = {int(item["pid"]): dict(item) for item in targets}
    for pid in tuple(by_pid):
        try:
            os.kill(pid, signal.SIGTERM)
            by_pid[pid]["termination_signal"] = "SIGTERM"
        except ProcessLookupError:
            by_pid[pid]["termination_signal"] = "already_gone"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        remaining = [pid for pid in by_pid if _process_exists(pid)]
        if not remaining:
            break
        time.sleep(0.2)
    remaining = [pid for pid in by_pid if _process_exists(pid)]
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
            by_pid[pid]["termination_signal"] = "SIGKILL"
        except ProcessLookupError:
            by_pid[pid]["termination_signal"] = "already_gone"
    return [by_pid[pid] for pid in sorted(by_pid)]


terminated_processes = _terminate_stale_processes()

json.dump(
    {
        "kill_switch_paths": list(kill_switch_paths),
        "disabled_system_units": disabled_system_units,
        "disabled_user_units": disabled_user_units,
        "terminated_processes": terminated_processes,
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
class RemoteHostTerminatedProcess:
    """Describe one stale remote process that the stabilizer terminated."""

    pid: int
    ppid: int
    elapsed_s: int
    command_excerpt: str
    scope: str
    match_label: str
    root_pid: int
    termination_signal: str


@dataclass(frozen=True, slots=True)
class RemoteHostStabilizationAction:
    """Capture the concrete remote actions executed during stabilization."""

    kill_switch_paths: tuple[str, ...]
    disabled_system_units: tuple[str, ...]
    disabled_user_units: tuple[str, ...]
    terminated_processes: tuple[RemoteHostTerminatedProcess, ...]


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
    terminated_processes: tuple[RemoteHostTerminatedProcess, ...]

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
            "terminated_processes": [asdict(item) for item in self.terminated_processes],
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
        default=_DEFAULT_SSH_TIMEOUT_S,
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
        stale_process_rules=_DEFAULT_STALE_PROCESS_RULES,
        stale_process_min_elapsed_s=_DEFAULT_STALE_PROCESS_MIN_ELAPSED_S,
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
        kill_switch_paths=action.kill_switch_paths,
        disabled_system_units=action.disabled_system_units,
        disabled_user_units=action.disabled_user_units,
        terminated_processes=action.terminated_processes,
    )


def probe_public_host_availability(
    *,
    settings: RemoteChonkyDBOpsSettings,
    timeout_s: float,
) -> ChonkyDBHttpProbeResult:
    """Probe the public query surface with empty-scope-safe readiness logic."""

    return probe_public_chonkydb(
        settings=settings,
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
    stale_process_rules: tuple[Mapping[str, object], ...],
    stale_process_min_elapsed_s: float,
) -> RemoteHostStabilizationAction:
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
            "stale_process_rules": [dict(item) for item in stale_process_rules],
            "stale_process_min_elapsed_s": float(stale_process_min_elapsed_s),
        },
        use_sudo=True,
    )
    raw_paths = payload.get("kill_switch_paths")
    raw_system_units = payload.get("disabled_system_units")
    raw_user_units = payload.get("disabled_user_units")
    raw_processes = payload.get("terminated_processes")
    if (
        not isinstance(raw_paths, list)
        or not isinstance(raw_system_units, list)
        or not isinstance(raw_user_units, list)
        or not isinstance(raw_processes, list)
    ):  # pragma: no cover
        raise RuntimeError(f"invalid_remote_stabilize_payload:{payload!r}")
    terminated_processes: list[RemoteHostTerminatedProcess] = []
    for item in raw_processes:
        if not isinstance(item, dict):  # pragma: no cover - defensive guard
            raise RuntimeError(f"invalid_remote_terminated_process_item:{item!r}")
        terminated_processes.append(
            RemoteHostTerminatedProcess(
                pid=int(item.get("pid", 0) or 0),
                ppid=int(item.get("ppid", 0) or 0),
                elapsed_s=int(item.get("elapsed_s", 0) or 0),
                command_excerpt=str(item.get("command_excerpt", "")).strip(),
                scope=str(item.get("scope", "")).strip(),
                match_label=str(item.get("match_label", "")).strip(),
                root_pid=int(item.get("root_pid", 0) or 0),
                termination_signal=str(item.get("termination_signal", "")).strip(),
            )
        )
    return RemoteHostStabilizationAction(
        kill_switch_paths=tuple(str(item).strip() for item in raw_paths if str(item).strip()),
        disabled_system_units=tuple(
            str(item).strip() for item in raw_system_units if str(item).strip()
        ),
        disabled_user_units=tuple(
            str(item).strip() for item in raw_user_units if str(item).strip()
        ),
        terminated_processes=tuple(terminated_processes),
    )


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
