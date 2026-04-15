"""Stabilize the dedicated Twinr ChonkyDB host under shared-host contention.

Purpose
-------
Use this module when the public Twinr ChonkyDB endpoint is reachable but
becomes slow or unstable because unrelated host workloads reclaim CPU and I/O
from the dedicated backend service. The stabilizer raises the backend service
weight, quiesces a curated set of non-Twinr CAIA system units plus user-session
units, and permanently disables proven host-wide display crashers that were
shown to OOM or fork-storm the shared thh1986 box during live incidents.

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
  user-session benchmark workers and direct dedicated-data-path writers that
  bypass systemd unit control
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
from urllib.parse import urlsplit

from twinr.ops.remote_chonkydb_repair import (
    ChonkyDBHttpProbeResult,
    RemoteChonkyDBExecutor,
    RemoteChonkyDBOpsSettings,
    load_remote_chonkydb_ops_settings,
    probe_public_chonkydb,
)
from twinr.ops.remote_systemd_restart_guard import (
    ensure_remote_host_control_permit,
    remove_remote_host_control_permit,
)


_DEFAULT_SETTLE_S = 8.0
_DEFAULT_SSH_TIMEOUT_S = 180.0
_DEFAULT_UNIT_QUIESCE_S = 0.25
_DEFAULT_KILLED_UNIT_QUIESCE_S = 1.5
_DEDICATED_DATA_PATH_PROCESS_MIN_ELAPSED_S = 60.0
_DEFAULT_FOREIGN_HEAVY_PROCESS_MIN_ELAPSED_S = 300.0
_DEFAULT_REACTIVATION_HOLD_S = 15.0
_DEFAULT_REACTIVATION_HOLD_POLL_S = 2.0
_HOST_CONTROL_GUARD_UNIT = "caia-twinr-host-control-guard.service"
# Live 2026-04-11 proof: the external CAIA host-control guard can write the
# historical shared unblock path and thereby reopen the same conflict units
# Twinr just quiesced. Keep the runtime blockers tied to a Twinr-private token
# so foreign guard actions cannot silently dissolve the quiet-host lane.
_HOST_STABILIZER_UNBLOCK_PATH = "/run/caia/maintenance/twinr_private_host_stabilizer_unblock"
_HOST_STABILIZER_BLOCK_DROPIN_NAME = "91-twinr-private-host-stabilizer-block.conf"
_LEGACY_HOST_STABILIZER_BLOCK_DROPIN_NAME = "90-twinr-host-stabilizer-block.conf"
_HOST_BOOT_PACER_SERVICE = "caia-twinr-host-boot-pacer.service"
_HOST_BOOT_PACER_SCRIPT_PATH = "/usr/local/sbin/caia_twinr_host_boot_pacer.py"
_HOST_BOOT_PACER_CONFIG_PATH = "/etc/twinr/caia_twinr_host_boot_pacer.json"
_HOST_BOOT_PACER_RELEASE_ROOT = "/run/caia/twinr_host_boot_pacer/releases"
_HOST_BOOT_PACER_DROPIN_NAME = "92-twinr-host-boot-pacer.conf"
_HOST_BOOT_PACER_INITIAL_DELAY_S = 20.0
_HOST_BOOT_PACER_EARLY_GAP_S = 8.0
_HOST_BOOT_PACER_MID_DELAY_S = 30.0
_HOST_BOOT_PACER_MID_GAP_S = 8.0
_HOST_BOOT_PACER_LATE_DELAY_S = 30.0
_HOST_BOOT_PACER_LATE_GAP_S = 4.0
_HOST_BOOT_PACER_TIMER_DELAY_S = 20.0
_HOST_BOOT_PACER_TIMER_GAP_S = 1.5
_HOST_BOOT_PACER_GUARD_DELAY_S = 20.0
_HOST_BOOT_PACER_GUARD_GAP_S = 1.5
_DEFAULT_CHONKY_PROPERTIES: dict[str, str] = {
    "CPUWeight": "10000",
    "StartupCPUWeight": "10000",
    "IOWeight": "10000",
    "StartupIOWeight": "10000",
}
_DEFAULT_STALE_PROCESS_MIN_ELAPSED_S = 1800.0
_REACTIVATED_ACTIVE_STATES = frozenset({"active", "activating", "reloading"})
_DEFAULT_KILLSWITCH_PATHS = (
    "/home/thh/Desktop/tessairact_local/tessairact/state/systemd_agent/killswitch_codex_traces_pipeline",
    "/home/thh/Desktop/tessairact_local/tessairact/state/systemd_agent/killswitch_user_policy_loop",
)
_DEFAULT_SYSTEM_RESTART_GUARD_UNITS = (
    # Live 2026-04-11 proof: these external CAIA guard units re-enable or start
    # the portal/ccodex lanes that Twinr just quiesced, which re-saturates the
    # dedicated backend and recreates Pi required-remote hangs.
    "caia-consumer-portal-live-restart-guard.service",
    "caia-consumer-portal-live-restart-guard.timer",
    "caia-consumer-portal-live-restart-guard.path",
    "codex-portal-live-override.service",
    "caia-ccodex-memory-live-restart-guard.service",
    "caia-ccodex-memory-live-restart-guard.timer",
    "caia-ccodex-memory-live-restart-guard.path",
)
_DEFAULT_ALWAYS_DISABLED_SYSTEM_UNITS = (
    # Live 2026-04-12 proof: sunshine-headless restart-looped more than 60
    # times, each restart re-ran six `runuser` launches from
    # sunshine-headless-run.sh, and both sunshine/gdm were OOM-killed on the
    # same boot. This host is Twinr's dedicated remote memory box, not a
    # desktop seat.
    "sunshine-headless.service",
    "gdm.service",
)
_HOST_SAFE_DEFAULT_TARGET = "multi-user.target"
_DEFAULT_BOOT_PACING_EXCLUDED_UNITS = (
    "caia-molt.service",
    "caia-consumer-portal-demo.service",
    "caia-external-site.service",
    "caia-ollama-gpu-proxy.service",
    *_DEFAULT_ALWAYS_DISABLED_SYSTEM_UNITS,
)
_DEFAULT_BOOT_PACING_EARLY_SYSTEM_UNITS = (
    "caia-ccodex-memory-api.service",
    "caia-control-plane-portal.service",
    "caia-control-plane-edge.service",
    "caia-consumer-terminald.service",
    "caia-consumer-portal.service",
    "caia-consumer-portal-demo.service",
    "caia-external-site.service",
    "caia-chonkycode-api.service",
)
_DEFAULT_BOOT_PACING_MID_SYSTEM_UNITS = (
    "caia-artifact-memory-ingest.service",
    "caia-portal-llm-worker.service",
    "caia-stt.service",
    "caia-qwen3tts-api.service",
    "caia-gpu-embeddings.service",
    "caia-ollama-gpu-proxy.service",
    "caia-joint-retrieval.service",
    "caia-repo-script-indexer.service",
    "caia-repo-script-llm-enricher.service",
    "caia-agent-tools-mcp.service",
)
_DEFAULT_BOOT_PACING_GUARD_UNITS = (
    *_DEFAULT_SYSTEM_RESTART_GUARD_UNITS,
    "caia-twinr-host-control-guard.timer",
    "caia-twinr-host-control-guard.path",
    _HOST_CONTROL_GUARD_UNIT,
)
_DEFAULT_SYSTEM_CONFLICTING_UNITS = (
    "caia-agent-tools-mcp.service",
    "caia-consumer-portal-config-helper.service",
    "caia-consumer-portal.service",
    "caia-consumer-portal-demo.service",
    "caia-consumer-terminald.service",
    "caia-ccodex-memory-api.service",
    "caia-chonkycode-api.service",
    "caia-control-plane-edge.service",
    "caia-control-plane-portal.service",
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
    "caia-external-site.service",
    "caia-gpu-embeddings.service",
    "caia-if-logic-monitor.service",
    "caia-joint-retrieval.service",
    "caia-ollama-gpu-proxy.service",
    "caia-portal-findings-housekeeper-autopilot.service",
    "caia-portal-fixreports-analysis-autopilot.service",
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
    "caia-portal-tasks-review-autopilot.service",
    "caia-qwen3tts-api.service",
    "caia-repo-script-llm-enricher.service",
    "caia-stt.service",
    *_DEFAULT_ALWAYS_DISABLED_SYSTEM_UNITS,
    *_DEFAULT_SYSTEM_RESTART_GUARD_UNITS,
)
_DEFAULT_USER_CONFLICTING_UNITS = (
    "caia-molt.service",
    "caia-chonkycode-chunks-posttransform.service",
    "caia-chonkycode-chunks-posttransform.timer",
    "caia-chonkycode-chunks-posttransform.path",
    "caia-chonkycode-meta-graph-ssot.service",
    "caia-chonkycode-meta-graph-ssot.timer",
    "ollama-gpu.service",
)
_DEFAULT_STALE_PROCESS_RULES: tuple[dict[str, object], ...] = (
    {
        "label": "code_graph_benchmark_runner",
        "required_substrings": (
            "benchmarks.code_graph.benchmark",
            "run_code_graph_benchmark",
        ),
    },
    {
        "label": "chonkycode_artifact_ingest_runner",
        "minimum_elapsed_s": _DEFAULT_FOREIGN_HEAVY_PROCESS_MIN_ELAPSED_S,
        "required_substrings": (
            "-m chonkycode.cli",
            "artifact-ingest",
        ),
    },
    {
        "label": "ccodex_memory_locomo_eval_runner",
        "minimum_elapsed_s": _DEFAULT_FOREIGN_HEAVY_PROCESS_MIN_ELAPSED_S,
        "required_substrings": (
            "benchmarks/ccodex_memory/ccodex_memory_locomo_mc10_eval.py",
        ),
    },
)
_UNMANAGED_CHONKYDB_API_SERVER_MIN_ELAPSED_S = 60.0


def build_stale_process_rules(settings: RemoteChonkyDBOpsSettings) -> tuple[dict[str, object], ...]:
    """Return the stale-process rules for one dedicated backend configuration.

    Interactive repair or benchmark jobs can bypass systemd entirely and touch
    the same dedicated Twinr ChonkyDB data directory directly. Those jobs are
    invisible to the existing foreign-consumer-by-base-url checks, so the
    stabilizer also matches the derived dedicated data-path substring.
    """

    rules: list[dict[str, object]] = [dict(rule) for rule in _DEFAULT_STALE_PROCESS_RULES]
    managed_cgroup_markers = _managed_chonkydb_api_cgroup_markers(settings)
    if managed_cgroup_markers:
        rules.append(
            {
                "label": "unmanaged_chonkydb_api_server_listener",
                "minimum_elapsed_s": _UNMANAGED_CHONKYDB_API_SERVER_MIN_ELAPSED_S,
                "required_substrings": (
                    "tessairact.automations.helpers.launcher --module chonkydb.api.server",
                ),
                "excluded_cgroup_substrings": managed_cgroup_markers,
                "require_listener": True,
            }
        )
    data_dir = _dedicated_backend_data_dir(settings.backend_local_base_url)
    if data_dir:
        rules.append(
            {
                "label": "dedicated_backend_data_path_writer",
                "minimum_elapsed_s": _DEDICATED_DATA_PATH_PROCESS_MIN_ELAPSED_S,
                "required_substrings": (data_dir,),
            }
        )
    return tuple(rules)


def _managed_chonkydb_api_cgroup_markers(
    settings: RemoteChonkyDBOpsSettings,
) -> tuple[str, ...]:
    """Return the cgroup markers for loopback API servers we intentionally keep.

    The shared thh1986 host can legitimately run more than one ChonkyDB API
    service under systemd. What we need to kill are the unmanaged strays that
    bypass service ownership and consume the same host CPU budget. The remote
    helper therefore excludes these known managed service cgroups before it
    considers a loopback API listener stale.
    """

    managed_services = (
        settings.backend_service,
        "caia-ccodex-memory-api.service",
        "caia-chonkycode-api.service",
    )
    markers: list[str] = []
    for service_name in managed_services:
        normalized_name = str(service_name or "").strip()
        if not normalized_name:
            continue
        markers.append(f"/system.slice/{normalized_name}")
    return tuple(dict.fromkeys(markers))


def _dedicated_backend_data_dir(backend_local_base_url: str) -> str | None:
    """Derive the standard dedicated backend data directory from the loopback URL."""

    parsed = urlsplit(str(backend_local_base_url or "").strip())
    try:
        port = parsed.port
    except ValueError:
        return None
    if port is None:
        return None
    return f"/home/thh/tessairact/state/offload/chonkydb/twinr_dedicated_{port}/data"


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
_REMOTE_GUARD_PROTECTED_UNITS_CODE = """
import json
from pathlib import Path
import subprocess
import sys

payload = json.loads(sys.stdin.read() or "{}")
backend_service = str(payload.get("backend_service", "")).strip()
guard_unit = str(payload.get("guard_unit", "")).strip()


def _show_property(unit: str, property_name: str) -> str:
    completed = subprocess.run(
        ["systemctl", "show", unit, "--no-pager", "-p", property_name],
        capture_output=True,
        text=True,
        check=False,
    )
    for raw_line in str(completed.stdout or "").splitlines():
        key, separator, value = raw_line.partition("=")
        if separator and str(key).strip() == property_name:
            return str(value).strip()
    return ""


working_directory = _show_property(guard_unit, "WorkingDirectory")
if not working_directory:
    raise RuntimeError(f"missing_guard_workdir:{guard_unit}")
profile_path = Path(working_directory) / "systemd" / "unit_activation_profile.json"
payload = json.loads(profile_path.read_text(encoding="utf-8"))
required_units = payload.get("required_units")
unit_map = payload.get("units")
if not isinstance(required_units, list):
    raise RuntimeError(f"invalid_guard_required_units:{profile_path}")
if not isinstance(unit_map, dict):
    raise RuntimeError(f"invalid_guard_unit_map:{profile_path}")
items = []
for raw_unit in required_units:
    unit_name = str(raw_unit or "").strip()
    if not unit_name or unit_name == backend_service:
        continue
    directive = unit_map.get(unit_name)
    if not isinstance(directive, dict):
        raise RuntimeError(f"missing_guard_unit_directive:{unit_name}:{profile_path}")
    if directive.get("enabled") is not True or directive.get("ensure_active") is not True:
        continue
    items.append(
        {
            "unit": unit_name,
            "reason": str(directive.get("reason") or "").strip(),
        }
    )
json.dump(
    {
        "guard_unit": guard_unit,
        "profile_path": str(profile_path),
        "units": items,
    },
    sys.stdout,
)
"""
_REMOTE_BOOT_PACER_SCRIPT = """
import argparse
import json
import os
from pathlib import Path
import pwd
import subprocess
import time


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if completed.returncode == 0:
        return
    detail = str(completed.stderr or completed.stdout or "").strip()
    raise RuntimeError(detail or f"command_failed:{command!r}:rc={completed.returncode}")


def _clean_release_root(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
            continue
        if path.is_dir():
            path.rmdir()
    root.rmdir()


def _touch_release(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("release\\n", encoding="utf-8")


def _user_env(owner: str) -> dict[str, str]:
    uid = pwd.getpwnam(owner).pw_uid
    values = dict(os.environ)
    values["XDG_RUNTIME_DIR"] = f"/run/user/{uid}"
    values["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path=/run/user/{uid}/bus"
    return values


def _start_unit(*, scope: str, owner: str, unit: str) -> None:
    if scope == "system":
        _run(["systemctl", "reset-failed", unit])
        _run(["systemctl", "start", unit])
        return
    environment = _user_env(owner)
    _run(
        [
            "runuser",
            "-u",
            owner,
            "--",
            "env",
            f"XDG_RUNTIME_DIR={environment['XDG_RUNTIME_DIR']}",
            f"DBUS_SESSION_BUS_ADDRESS={environment['DBUS_SESSION_BUS_ADDRESS']}",
            "systemctl",
            "--user",
            "reset-failed",
            unit,
        ],
        env=environment,
    )
    _run(
        [
            "runuser",
            "-u",
            owner,
            "--",
            "env",
            f"XDG_RUNTIME_DIR={environment['XDG_RUNTIME_DIR']}",
            f"DBUS_SESSION_BUS_ADDRESS={environment['DBUS_SESSION_BUS_ADDRESS']}",
            "systemctl",
            "--user",
            "start",
            unit,
        ],
        env=environment,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    owner = str(payload.get("user_unit_owner", "") or "").strip()
    release_root = Path(str(payload.get("release_root", "") or "").strip())
    _clean_release_root(release_root)
    release_root.mkdir(parents=True, exist_ok=True)
    for raw_step in payload.get("steps", []):
        sleep_before_s = max(0.0, float(raw_step.get("sleep_before_s", 0.0) or 0.0))
        if sleep_before_s > 0:
            time.sleep(sleep_before_s)
        scope = str(raw_step.get("scope", "system") or "system").strip().lower()
        units = tuple(str(item).strip() for item in raw_step.get("units", []) if str(item).strip())
        inter_unit_delay_s = max(0.0, float(raw_step.get("inter_unit_delay_s", 0.0) or 0.0))
        for index, unit in enumerate(units):
            _touch_release(release_root / scope / unit)
            _start_unit(scope=scope, owner=owner, unit=unit)
            if inter_unit_delay_s > 0 and index + 1 < len(units):
                time.sleep(inter_unit_delay_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
_REMOTE_SYNC_BOOT_PACING_CODE = """
import json
import os
from pathlib import Path
import pwd
import subprocess
import sys

payload = json.loads(sys.stdin.read() or "{}")
service_name = str(payload.get("service_name", "") or "").strip()
script_path = Path(str(payload.get("script_path", "") or "").strip())
config_path = Path(str(payload.get("config_path", "") or "").strip())
release_root = str(payload.get("release_root", "") or "").strip()
dropin_name = str(payload.get("dropin_name", "") or "").strip()
user_unit_owner = str(payload.get("user_unit_owner", "") or "").strip()
script_contents = str(payload.get("script_contents", "") or "")
default_target = str(payload.get("default_target", "") or "").strip()
boot_steps = tuple(item for item in payload.get("steps", []) if isinstance(item, dict))
all_system_units = tuple(
    str(item).strip()
    for item in payload.get("all_system_units", [])
    if str(item).strip()
)
always_disabled_system_units = tuple(
    str(item).strip()
    for item in payload.get("always_disabled_system_units", [])
    if str(item).strip()
)
all_user_units = tuple(
    str(item).strip()
    for item in payload.get("all_user_units", [])
    if str(item).strip()
)


def _write_text(
    path: Path,
    contents: str,
    *,
    mode: int,
    owner: str | None = None,
) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    changed = True
    if path.exists():
        changed = path.read_text(encoding="utf-8") != contents
    if changed:
        path.write_text(contents, encoding="utf-8")
    os.chmod(path, mode)
    if owner:
        owner_entry = pwd.getpwnam(owner)
        os.chown(path, owner_entry.pw_uid, owner_entry.pw_gid)
    return changed


def _remove_file(path: Path) -> bool:
    if not path.exists():
        return False
    path.unlink()
    parent = path.parent
    while parent.name.endswith(".d") and parent.exists():
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent
    return True


def _dropin_path(*, scope: str, unit: str) -> Path:
    if scope == "system":
        return Path("/etc/systemd/system") / f"{unit}.d" / dropin_name
    owner_entry = pwd.getpwnam(user_unit_owner)
    return (
        Path(owner_entry.pw_dir)
        / ".config/systemd/user"
        / f"{unit}.d"
        / dropin_name
    )


def _dropin_contents(*, scope: str, unit: str) -> str:
    return (
        "[Unit]\\n"
        f"ConditionPathExists={release_root}/{scope}/{unit}\\n"
    )


def _user_bus_exists() -> bool:
    if not user_unit_owner:
        return False
    uid = pwd.getpwnam(user_unit_owner).pw_uid
    return Path(f"/run/user/{uid}/bus").exists()


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode == 0:
        return
    detail = str(completed.stderr or completed.stdout or "").strip()
    raise RuntimeError(detail or f"command_failed:{command!r}:rc={completed.returncode}")


def _run_user(*args: str) -> None:
    owner_entry = pwd.getpwnam(user_unit_owner)
    runtime_dir = f"/run/user/{owner_entry.pw_uid}"
    command = [
        "runuser",
        "-u",
        user_unit_owner,
        "--",
        "env",
        f"XDG_RUNTIME_DIR={runtime_dir}",
        f"DBUS_SESSION_BUS_ADDRESS=unix:path={runtime_dir}/bus",
        "systemctl",
        "--user",
        *list(args),
    ]
    _run(command)


paced_system_units = []
paced_user_units = []
for step in boot_steps:
    scope = str(step.get("scope", "system") or "system").strip().lower()
    units = tuple(str(item).strip() for item in step.get("units", []) if str(item).strip())
    if scope == "system":
        paced_system_units.extend(units)
    else:
        paced_user_units.extend(units)
paced_system_units = list(dict.fromkeys(paced_system_units))
paced_user_units = list(dict.fromkeys(paced_user_units))

system_changed = False
user_changed = False
for unit in all_system_units:
    path = _dropin_path(scope="system", unit=unit)
    if unit in paced_system_units:
        system_changed = _write_text(
            path,
            _dropin_contents(scope="system", unit=unit),
            mode=0o644,
        ) or system_changed
    else:
        system_changed = _remove_file(path) or system_changed

for unit in all_user_units:
    path = _dropin_path(scope="user", unit=unit)
    if unit in paced_user_units:
        user_changed = _write_text(
            path,
            _dropin_contents(scope="user", unit=unit),
            mode=0o644,
            owner=user_unit_owner,
        ) or user_changed
    else:
        user_changed = _remove_file(path) or user_changed

script_changed = _write_text(script_path, script_contents, mode=0o755)
config_payload = {
    "release_root": release_root,
    "user_unit_owner": user_unit_owner,
    "steps": list(boot_steps),
}
config_changed = _write_text(
    config_path,
    json.dumps(config_payload, ensure_ascii=False, indent=2) + "\\n",
    mode=0o644,
)
service_contents = (
    "[Unit]\\n"
    "Description=Twinr host boot pacer for staged CAIA startup\\n"
    "After=network-online.target caia-twinr-chonkydb-alt.service\\n"
    "Wants=network-online.target caia-twinr-chonkydb-alt.service\\n\\n"
    "[Service]\\n"
    "Type=oneshot\\n"
    f"ExecStart=/usr/bin/python3 {script_path} --config {config_path}\\n"
    "RemainAfterExit=yes\\n\\n"
    "[Install]\\n"
    "WantedBy=multi-user.target\\n"
)
service_path = Path("/etc/systemd/system") / service_name
service_changed = _write_text(service_path, service_contents, mode=0o644)
if system_changed or script_changed or config_changed or service_changed:
    _run(["systemctl", "daemon-reload"])
for unit in always_disabled_system_units:
    _run(["systemctl", "disable", unit])
    _run(["systemctl", "stop", unit])
if default_target:
    _run(["systemctl", "set-default", default_target])
_run(["systemctl", "enable", service_name])
if user_changed and _user_bus_exists():
    _run_user("daemon-reload")

json.dump(
    {
        "service_name": service_name,
        "script_path": str(script_path),
        "config_path": str(config_path),
        "release_root": release_root,
        "paced_system_units": paced_system_units,
        "paced_user_units": paced_user_units,
        "always_disabled_system_units": list(always_disabled_system_units),
        "default_target": default_target,
    },
    sys.stdout,
)
"""
_REMOTE_STABILIZE_HOST_CODE = """
import json
import os
import pwd
from pathlib import Path
import re
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
unit_quiesce_s = max(0.0, float(payload.get("unit_quiesce_s", 0.0) or 0.0))
killed_unit_quiesce_s = max(
    unit_quiesce_s,
    float(payload.get("killed_unit_quiesce_s", 0.0) or 0.0),
)
runtime_block_unblock_path = str(payload.get("runtime_block_unblock_path", "") or "").strip()
runtime_block_dropin_name = "91-twinr-private-host-stabilizer-block.conf"
legacy_runtime_block_dropin_name = "90-twinr-host-stabilizer-block.conf"
unit_control_timeout_s = 8.0
unit_kill_timeout_s = 5.0

for raw_path in kill_switch_paths:
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

def _run_bounded(command: list[str], *, timeout_s: float) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return None


def _command_detail(completed: subprocess.CompletedProcess[str] | None) -> str:
    if completed is None:
        return "timeout"
    detail = str(completed.stderr or completed.stdout or "").strip()
    return detail or f"rc={completed.returncode}"


def _require_success(
    completed: subprocess.CompletedProcess[str] | None,
    *,
    label: str,
) -> subprocess.CompletedProcess[str]:
    if completed is None:
        raise RuntimeError(f"{label}:timeout")
    if completed.returncode != 0:
        raise RuntimeError(f"{label}:rc={completed.returncode}:detail={_command_detail(completed)}")
    return completed


if backend_service and property_map:
    command = ["systemctl", "set-property", backend_service]
    for key, value in property_map.items():
        command.append(f"{key}={value}")
    subprocess.run(command, check=True)


def _show_map_from_completed(completed: subprocess.CompletedProcess[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in str(completed.stdout or "").splitlines():
        key, separator, value = raw_line.partition("=")
        if separator:
            values[str(key).strip()] = str(value).strip()
    return values


def _show_system_unit_state(unit: str) -> dict[str, str]:
    completed = _require_success(
        _run_bounded(
            [
                "systemctl",
                "show",
                unit,
                "--no-pager",
                "-p",
                "LoadState",
                "-p",
                "ActiveState",
                "-p",
                "Result",
            ],
            timeout_s=unit_control_timeout_s,
        ),
        label=f"systemctl_show_failed:{unit}",
    )
    return _show_map_from_completed(completed)


def _unit_needs_reset(state: dict[str, str]) -> bool:
    load_state = str(state.get("LoadState", "")).strip().lower()
    active_state = str(state.get("ActiveState", "")).strip().lower()
    result = str(state.get("Result", "")).strip().lower()
    if load_state != "loaded":
        return False
    if active_state == "failed":
        return True
    return result not in {"", "success"}


def _maybe_reset_failed_system_unit(unit: str, *, label: str) -> None:
    state = _show_system_unit_state(unit)
    if not _unit_needs_reset(state):
        return
    _require_success(
        _run_bounded(["systemctl", "reset-failed", unit], timeout_s=unit_control_timeout_s),
        label=label,
    )


def _runtime_block_dropin_path(unit: str, *, dropin_name: str = runtime_block_dropin_name) -> Path:
    return Path("/run/systemd/system") / f"{unit}.d" / dropin_name


def _legacy_runtime_block_dropin_path(unit: str) -> Path:
    return _runtime_block_dropin_path(unit, dropin_name=legacy_runtime_block_dropin_name)


def _user_runtime_block_dropin_path(
    unit: str,
    *,
    dropin_name: str = runtime_block_dropin_name,
) -> Path:
    uid = pwd.getpwnam(user_unit_owner).pw_uid
    return Path(f"/run/user/{uid}/systemd/user") / f"{unit}.d" / dropin_name


def _legacy_user_runtime_block_dropin_path(unit: str) -> Path:
    return _user_runtime_block_dropin_path(unit, dropin_name=legacy_runtime_block_dropin_name)


def _ensure_runtime_blocker_removed() -> None:
    if not runtime_block_unblock_path:
        return
    Path(runtime_block_unblock_path).unlink(missing_ok=True)


def _ensure_system_runtime_blocker(unit: str) -> None:
    if not runtime_block_unblock_path:
        return
    path = _runtime_block_dropin_path(unit)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "[Unit]\\n"
        f"ConditionPathExists={runtime_block_unblock_path}\\n",
        encoding="utf-8",
    )


def _reload_system_manager() -> None:
    _require_success(
        _run_bounded(["systemctl", "daemon-reload"], timeout_s=unit_control_timeout_s),
        label="systemctl_daemon_reload_failed",
    )


def _remove_legacy_system_runtime_blocker(unit: str) -> None:
    _legacy_runtime_block_dropin_path(unit).unlink(missing_ok=True)


def _ensure_user_runtime_blocker(unit: str) -> None:
    if not runtime_block_unblock_path:
        return
    path = _user_runtime_block_dropin_path(unit)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "[Unit]\\n"
        f"ConditionPathExists={runtime_block_unblock_path}\\n",
        encoding="utf-8",
    )


def _quiesce_after_unit(*, forced_kill: bool) -> None:
    sleep_s = killed_unit_quiesce_s if forced_kill else unit_quiesce_s
    if sleep_s > 0:
        time.sleep(sleep_s)


def _stop_system_unit(unit: str) -> bool:
    completed = _run_bounded(["systemctl", "stop", unit], timeout_s=unit_control_timeout_s)
    if completed is not None and completed.returncode == 0:
        return False
    _require_success(
        _run_bounded(
            ["systemctl", "kill", "--kill-who=all", "--signal=SIGKILL", unit],
            timeout_s=unit_kill_timeout_s,
        ),
        label=f"systemctl_kill_failed:{unit}",
    )
    _maybe_reset_failed_system_unit(
        unit,
        label=f"systemctl_reset_failed_after_kill:{unit}",
    )
    return True


disabled_system_units = []
for unit in system_units:
    _ensure_runtime_blocker_removed()
    _remove_legacy_system_runtime_blocker(unit)
    _ensure_system_runtime_blocker(unit)
    _reload_system_manager()
    forced_kill = _stop_system_unit(unit)
    _require_success(
        _run_bounded(["systemctl", "disable", unit], timeout_s=unit_control_timeout_s),
        label=f"systemctl_disable_failed:{unit}",
    )
    _maybe_reset_failed_system_unit(
        unit,
        label=f"systemctl_reset_failed:{unit}",
    )
    disabled_system_units.append(unit)
    _quiesce_after_unit(forced_kill=forced_kill)


def _run_user_systemctl(
    *args: str,
    timeout_s: float = unit_control_timeout_s,
) -> subprocess.CompletedProcess[str] | None:
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
    return _run_bounded(command, timeout_s=timeout_s)


def _reload_user_manager() -> None:
    _require_success(
        _run_user_systemctl("daemon-reload"),
        label="user_systemctl_daemon_reload_failed",
    )


def _remove_legacy_user_runtime_blocker(unit: str) -> None:
    legacy_user_runtime_blocker_path = _legacy_user_runtime_block_dropin_path(unit)
    legacy_user_runtime_blocker_path.unlink(missing_ok=True)


def _show_user_unit_state(unit: str) -> dict[str, str]:
    completed = _require_success(
        _run_user_systemctl(
            "show",
            unit,
            "--no-pager",
            "-p",
            "LoadState",
            "-p",
            "ActiveState",
            "-p",
            "Result",
        ),
        label=f"user_systemctl_show_failed:{unit}",
    )
    return _show_map_from_completed(completed)


def _maybe_reset_failed_user_unit(unit: str, *, label: str) -> None:
    state = _show_user_unit_state(unit)
    if not _unit_needs_reset(state):
        return
    _require_success(
        _run_user_systemctl("reset-failed", unit),
        label=label,
    )


def _stop_user_unit(unit: str) -> bool:
    completed = _run_user_systemctl("stop", unit)
    if completed is not None and completed.returncode == 0:
        return False
    _require_success(
        _run_user_systemctl(
            "kill",
            "--kill-who=all",
            "--signal=SIGKILL",
            unit,
            timeout_s=unit_kill_timeout_s,
        ),
        label=f"user_systemctl_kill_failed:{unit}",
    )
    _maybe_reset_failed_user_unit(
        unit,
        label=f"user_systemctl_reset_failed_after_kill:{unit}",
    )
    return True


disabled_user_units = []
for unit in user_units:
    _ensure_runtime_blocker_removed()
    _remove_legacy_user_runtime_blocker(unit)
    _ensure_user_runtime_blocker(unit)
    _reload_user_manager()
    forced_kill = _stop_user_unit(unit)
    _require_success(
        _run_user_systemctl("disable", unit),
        label=f"user_systemctl_disable_failed:{unit}",
    )
    _maybe_reset_failed_user_unit(
        unit,
        label=f"user_systemctl_reset_failed:{unit}",
    )
    disabled_user_units.append(unit)
    _quiesce_after_unit(forced_kill=forced_kill)


def _list_process_table() -> tuple[dict[int, dict[str, object]], dict[int, set[int]]]:
    def _listener_ports_by_pid() -> dict[int, tuple[int, ...]]:
        completed = subprocess.run(
            ["ss", "-tlnpH"],
            capture_output=True,
            text=True,
            check=False,
        )
        listeners: dict[int, set[int]] = {}
        for raw_line in str(completed.stdout or "").splitlines():
            line = str(raw_line).strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            local_address = str(parts[3]).strip()
            users_field = str(parts[-1]).strip()
            try:
                port = int(local_address.rsplit(":", 1)[-1])
            except ValueError:
                continue
            for match in re.finditer(r"pid=(\d+)", users_field):
                listeners.setdefault(int(match.group(1)), set()).add(port)
        return {pid: tuple(sorted(ports)) for pid, ports in listeners.items()}

    def _read_cgroup(pid: int) -> str:
        try:
            with open(f"/proc/{pid}/cgroup", "r", encoding="utf-8") as handle:
                return handle.read().strip()
        except OSError:
            return ""

    listener_ports = _listener_ports_by_pid()
    completed = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,user=,etimes=,args="],
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
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            elapsed_s = int(parts[3])
        except ValueError:
            continue
        user = str(parts[2]).strip()
        args = str(parts[4]).strip()
        processes[pid] = {
            "pid": pid,
            "ppid": ppid,
            "user": user,
            "elapsed_s": elapsed_s,
            "args": args,
            "cgroup": _read_cgroup(pid),
            "listener_ports": listener_ports.get(pid, ()),
        }
        children.setdefault(ppid, set()).add(pid)
    return processes, children


def _command_excerpt(raw_value: object, *, limit: int = 240) -> str:
    value = str(raw_value or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def _collect_stale_process_targets() -> list[dict[str, object]]:
    if not stale_process_rules:
        return []
    processes, children = _list_process_table()
    exempt_pids = {os.getpid()}
    cursor = os.getpid()
    while cursor > 0:
        process = processes.get(cursor)
        if process is None:
            break
        parent_pid = int(process.get("ppid", 0) or 0)
        if parent_pid <= 0 or parent_pid in exempt_pids:
            break
        exempt_pids.add(parent_pid)
        cursor = parent_pid
    matched_roots: dict[int, dict[str, object]] = {}
    for process in processes.values():
        pid = int(process.get("pid", 0) or 0)
        if pid in exempt_pids:
            continue
        args = str(process.get("args", "") or "")
        elapsed_s = int(process.get("elapsed_s", 0) or 0)
        for rule in stale_process_rules:
            label = str(rule.get("label", "") or "").strip()
            required = tuple(
                str(item).strip()
                for item in rule.get("required_substrings", [])
                if str(item).strip()
            )
            excluded_cgroup_substrings = tuple(
                str(item).strip()
                for item in rule.get("excluded_cgroup_substrings", [])
                if str(item).strip()
            )
            require_listener = bool(rule.get("require_listener"))
            minimum_elapsed_s = float(rule.get("minimum_elapsed_s", stale_process_min_elapsed_s) or 0.0)
            if elapsed_s < minimum_elapsed_s:
                continue
            cgroup = str(process.get("cgroup", "") or "")
            if excluded_cgroup_substrings and any(
                marker in cgroup for marker in excluded_cgroup_substrings
            ):
                continue
            listener_ports = tuple(int(item) for item in process.get("listener_ports", ()) if int(item) > 0)
            if require_listener and not listener_ports:
                continue
            if required and all(marker in args for marker in required):
                matched_roots[int(process["pid"])] = {
                    "match_label": label,
                    "minimum_elapsed_s": minimum_elapsed_s,
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
            if child_pid in target_pids or child_pid in exempt_pids:
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
class RemoteHostQuietHoldObservation:
    """Capture the bounded reactivation hold that follows a stop/mask pass.

    Some systemd units briefly reappear as active or enabled while their stop
    jobs, restarts, or unit garbage collection settle. The stabilizer must
    distinguish that transient churn from a persistent quiet-host violation
    before it retries or fails closed.
    """

    system_units: tuple[RemoteHostUnitState, ...]
    user_units: tuple[RemoteHostUnitState, ...]
    system_reactivated: tuple[RemoteHostUnitState, ...]
    user_reactivated: tuple[RemoteHostUnitState, ...]
    polls: int
    elapsed_s: float


@dataclass(frozen=True, slots=True)
class RemoteHostBootPacingStep:
    """Describe one boot-time release wave for paced host units."""

    scope: str
    units: tuple[str, ...]
    sleep_before_s: float
    inter_unit_delay_s: float


@dataclass(frozen=True, slots=True)
class RemoteHostBootPacingStatus:
    """Summarize the persistent boot-pacing policy synced to the remote host."""

    service_name: str
    script_path: str
    config_path: str
    release_root: str
    paced_system_units: tuple[str, ...]
    paced_user_units: tuple[str, ...]
    always_disabled_system_units: tuple[str, ...]
    default_target: str


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
    reactivation_hold_polls: int
    reactivation_hold_elapsed_s: float

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
            "reactivation_hold_polls": self.reactivation_hold_polls,
            "reactivation_hold_elapsed_s": self.reactivation_hold_elapsed_s,
        }


def _collect_reactivated_units(
    states: tuple[RemoteHostUnitState, ...],
) -> tuple[RemoteHostUnitState, ...]:
    """Return units that still violate the quiet-host hold after stabilization.

    Conflict units are expected to end the hold pass inactive and not explicitly
    enabled. Static units are allowed to remain `static`, but any unit that is
    still `enabled` or currently active is a real hold violation because it can
    continue to steal CPU/IO from the dedicated 3044 backend.
    """

    violations: list[RemoteHostUnitState] = []
    for state in states:
        if state.active_state in _REACTIVATED_ACTIVE_STATES or state.enabled_state == "enabled":
            violations.append(state)
    return tuple(violations)


def _merge_stabilization_actions(
    first: RemoteHostStabilizationAction,
    second: RemoteHostStabilizationAction | None,
) -> RemoteHostStabilizationAction:
    """Merge two stabilization passes into one operator-facing action summary."""

    if second is None:
        return first

    def _dedupe(values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item for item in values if item))

    process_by_key: dict[tuple[int, str, str], RemoteHostTerminatedProcess] = {}
    for item in (*first.terminated_processes, *second.terminated_processes):
        key = (item.pid, item.match_label, item.termination_signal)
        process_by_key[key] = item

    return RemoteHostStabilizationAction(
        kill_switch_paths=_dedupe(first.kill_switch_paths + second.kill_switch_paths),
        disabled_system_units=_dedupe(first.disabled_system_units + second.disabled_system_units),
        disabled_user_units=_dedupe(first.disabled_user_units + second.disabled_user_units),
        terminated_processes=tuple(process_by_key.values()),
    )


def _observe_quiet_host_hold(
    *,
    executor: RemoteChonkyDBExecutor,
    system_units: tuple[RemoteHostUnitState, ...],
    user_units: tuple[RemoteHostUnitState, ...],
    user_unit_owner: str,
    hold_s: float,
    poll_interval_s: float,
) -> RemoteHostQuietHoldObservation:
    """Poll a short bounded window before treating reactivation as persistent."""

    current_system_units = system_units
    current_user_units = user_units
    system_reactivated = _collect_reactivated_units(current_system_units)
    user_reactivated = _collect_reactivated_units(current_user_units)
    remaining_s = max(0.0, float(hold_s))
    poll_s = max(0.0, float(poll_interval_s))
    polls = 0
    elapsed_s = 0.0
    while (system_reactivated or user_reactivated) and remaining_s > 0.0 and poll_s > 0.0:
        sleep_s = min(poll_s, remaining_s)
        time.sleep(sleep_s)
        elapsed_s += sleep_s
        remaining_s = max(0.0, remaining_s - sleep_s)
        current_system_units = fetch_remote_unit_states(
            executor=executor,
            units=tuple(item.unit for item in current_system_units),
            scope="system",
            user_unit_owner=user_unit_owner,
        )
        current_user_units = fetch_remote_unit_states(
            executor=executor,
            units=tuple(item.unit for item in current_user_units),
            scope="user",
            user_unit_owner=user_unit_owner,
        )
        polls += 1
        system_reactivated = _collect_reactivated_units(current_system_units)
        user_reactivated = _collect_reactivated_units(current_user_units)
    return RemoteHostQuietHoldObservation(
        system_units=current_system_units,
        user_units=current_user_units,
        system_reactivated=system_reactivated,
        user_reactivated=user_reactivated,
        polls=polls,
        elapsed_s=round(elapsed_s, 3),
    )


def _dedupe_nonempty_units(units: tuple[str, ...]) -> tuple[str, ...]:
    """Keep first-seen unit order while dropping empty or duplicate names."""

    return tuple(dict.fromkeys(unit for unit in units if unit))


def _ordered_subset(
    units: tuple[str, ...],
    preferred_units: tuple[str, ...],
) -> tuple[str, ...]:
    """Return the units that appear in both tuples, preserving preferred order."""

    unit_set = set(units)
    return tuple(unit for unit in preferred_units if unit in unit_set)


def build_remote_host_boot_pacing_steps(
    *,
    system_units: tuple[str, ...],
    user_units: tuple[str, ...],
) -> tuple[RemoteHostBootPacingStep, ...]:
    """Build the paced boot-release waves for the shared thh1986 host.

    The controlled reboot failures proved that many CAIA units come back via
    dependencies, timers, paths, or restart guards even when their own unit
    file state says `disabled`. Keep them behind persistent `ConditionPathExists`
    drop-ins under `/etc`, then release them in dependency-aware waves after
    the dedicated Twinr backend is already back on its feet. Units that were
    proven to destabilize the host itself stay excluded and are disabled
    persistently instead of being re-released.
    """

    excluded_units = set(_DEFAULT_BOOT_PACING_EXCLUDED_UNITS)
    paced_system_units = _dedupe_nonempty_units(
        tuple(unit for unit in system_units if unit not in excluded_units)
    )
    paced_user_units = _dedupe_nonempty_units(
        tuple(unit for unit in user_units if unit not in excluded_units)
    )
    service_units = tuple(unit for unit in paced_system_units if unit.endswith(".service"))
    timer_path_units = tuple(
        unit
        for unit in paced_system_units
        if unit.endswith(".timer") or unit.endswith(".path")
    )
    early_service_units = _ordered_subset(
        service_units,
        _DEFAULT_BOOT_PACING_EARLY_SYSTEM_UNITS,
    )
    mid_service_units = _ordered_subset(
        service_units,
        _DEFAULT_BOOT_PACING_MID_SYSTEM_UNITS,
    )
    guard_units = _ordered_subset(
        paced_system_units,
        _DEFAULT_BOOT_PACING_GUARD_UNITS,
    )
    used_system_units = {
        *early_service_units,
        *mid_service_units,
        *guard_units,
    }
    late_service_units = tuple(
        unit for unit in service_units if unit not in used_system_units
    )
    timer_path_release_units = tuple(
        unit for unit in timer_path_units if unit not in set(guard_units)
    )
    steps: list[RemoteHostBootPacingStep] = []

    def _append_step(
        *,
        scope: str,
        units: tuple[str, ...],
        sleep_before_s: float,
        inter_unit_delay_s: float,
    ) -> None:
        normalized_units = _dedupe_nonempty_units(units)
        if not normalized_units:
            return
        steps.append(
            RemoteHostBootPacingStep(
                scope=scope,
                units=normalized_units,
                sleep_before_s=float(sleep_before_s),
                inter_unit_delay_s=float(inter_unit_delay_s),
            )
        )

    _append_step(
        scope="system",
        units=early_service_units,
        sleep_before_s=_HOST_BOOT_PACER_INITIAL_DELAY_S,
        inter_unit_delay_s=_HOST_BOOT_PACER_EARLY_GAP_S,
    )
    _append_step(
        scope="system",
        units=mid_service_units,
        sleep_before_s=_HOST_BOOT_PACER_MID_DELAY_S,
        inter_unit_delay_s=_HOST_BOOT_PACER_MID_GAP_S,
    )
    _append_step(
        scope="system",
        units=late_service_units,
        sleep_before_s=_HOST_BOOT_PACER_LATE_DELAY_S,
        inter_unit_delay_s=_HOST_BOOT_PACER_LATE_GAP_S,
    )
    _append_step(
        scope="user",
        units=paced_user_units,
        sleep_before_s=_HOST_BOOT_PACER_LATE_DELAY_S,
        inter_unit_delay_s=_HOST_BOOT_PACER_LATE_GAP_S,
    )
    _append_step(
        scope="system",
        units=timer_path_release_units,
        sleep_before_s=_HOST_BOOT_PACER_TIMER_DELAY_S,
        inter_unit_delay_s=_HOST_BOOT_PACER_TIMER_GAP_S,
    )
    _append_step(
        scope="system",
        units=guard_units,
        sleep_before_s=_HOST_BOOT_PACER_GUARD_DELAY_S,
        inter_unit_delay_s=_HOST_BOOT_PACER_GUARD_GAP_S,
    )
    return tuple(steps)


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
    # Live 2026-04-11 outage proof: the remote host-control guard's
    # required_units set can itself contain the long-running CAIA services that
    # saturate the dedicated Twinr backend host. Twinr's required remote memory
    # must take precedence during host-contention stabilization, so quiesce the
    # full proven conflict lane instead of excluding guard-protected units.
    system_units = tuple(dict.fromkeys((*_DEFAULT_SYSTEM_CONFLICTING_UNITS, _HOST_CONTROL_GUARD_UNIT)))
    public_before = probe_public_host_availability(
        settings=settings,
        timeout_s=probe_timeout_s,
    )
    system_units_before = fetch_remote_unit_states(
        executor=executor,
        units=system_units,
        scope="system",
        user_unit_owner=settings.ssh.user,
    )
    user_units_before = fetch_remote_unit_states(
        executor=executor,
        units=_DEFAULT_USER_CONFLICTING_UNITS,
        scope="user",
        user_unit_owner=settings.ssh.user,
    )
    system_units_after = system_units_before
    user_units_after = user_units_before
    system_reactivated: tuple[RemoteHostUnitState, ...] = ()
    user_reactivated: tuple[RemoteHostUnitState, ...] = ()
    reactivation_hold_polls = 0
    reactivation_hold_elapsed_s = 0.0
    recovery_action: RemoteHostStabilizationAction | None = None
    public_after = public_before
    backend_properties: Mapping[str, str] = {}
    maintenance_permit_open = False
    try:
        ensure_remote_host_control_permit(
            executor=executor,
        )
        maintenance_permit_open = True
        ensure_remote_host_boot_pacing_policy(
            executor=executor,
            system_units=tuple(
                item.unit for item in system_units_before if item.load_state != "not-found"
            ),
            user_units=tuple(
                item.unit for item in user_units_before if item.load_state != "not-found"
            ),
            user_unit_owner=settings.ssh.user,
        )
        action = apply_remote_host_stabilization(
            executor=executor,
            backend_service=settings.backend_service,
            system_units=system_units,
            user_units=_DEFAULT_USER_CONFLICTING_UNITS,
            user_unit_owner=settings.ssh.user,
            kill_switch_paths=_DEFAULT_KILLSWITCH_PATHS,
            property_assignments=_DEFAULT_CHONKY_PROPERTIES,
            stale_process_rules=build_stale_process_rules(settings),
            stale_process_min_elapsed_s=_DEFAULT_STALE_PROCESS_MIN_ELAPSED_S,
        )
        if settle_s > 0:
            time.sleep(float(settle_s))
        system_units_after = fetch_remote_unit_states(
            executor=executor,
            units=system_units,
            scope="system",
            user_unit_owner=settings.ssh.user,
        )
        user_units_after = fetch_remote_unit_states(
            executor=executor,
            units=_DEFAULT_USER_CONFLICTING_UNITS,
            scope="user",
            user_unit_owner=settings.ssh.user,
        )
        system_reactivated = _collect_reactivated_units(system_units_after)
        user_reactivated = _collect_reactivated_units(user_units_after)
        quiet_hold = _observe_quiet_host_hold(
            executor=executor,
            system_units=system_units_after,
            user_units=user_units_after,
            user_unit_owner=settings.ssh.user,
            hold_s=_DEFAULT_REACTIVATION_HOLD_S,
            poll_interval_s=_DEFAULT_REACTIVATION_HOLD_POLL_S,
        )
        system_units_after = quiet_hold.system_units
        user_units_after = quiet_hold.user_units
        system_reactivated = quiet_hold.system_reactivated
        user_reactivated = quiet_hold.user_reactivated
        reactivation_hold_polls = quiet_hold.polls
        reactivation_hold_elapsed_s = quiet_hold.elapsed_s
        if system_reactivated or user_reactivated:
            recovery_action = apply_remote_host_stabilization(
                executor=executor,
                backend_service=settings.backend_service,
                system_units=tuple(item.unit for item in system_reactivated),
                user_units=tuple(item.unit for item in user_reactivated),
                user_unit_owner=settings.ssh.user,
                kill_switch_paths=_DEFAULT_KILLSWITCH_PATHS,
                property_assignments=_DEFAULT_CHONKY_PROPERTIES,
                stale_process_rules=build_stale_process_rules(settings),
                stale_process_min_elapsed_s=_DEFAULT_STALE_PROCESS_MIN_ELAPSED_S,
            )
            if settle_s > 0:
                time.sleep(float(settle_s))
            system_units_after = fetch_remote_unit_states(
                executor=executor,
                units=system_units,
                scope="system",
                user_unit_owner=settings.ssh.user,
            )
            user_units_after = fetch_remote_unit_states(
                executor=executor,
                units=_DEFAULT_USER_CONFLICTING_UNITS,
                scope="user",
                user_unit_owner=settings.ssh.user,
            )
            quiet_hold = _observe_quiet_host_hold(
                executor=executor,
                system_units=system_units_after,
                user_units=user_units_after,
                user_unit_owner=settings.ssh.user,
                hold_s=_DEFAULT_REACTIVATION_HOLD_S,
                poll_interval_s=_DEFAULT_REACTIVATION_HOLD_POLL_S,
            )
            system_units_after = quiet_hold.system_units
            user_units_after = quiet_hold.user_units
            system_reactivated = quiet_hold.system_reactivated
            user_reactivated = quiet_hold.user_reactivated
            reactivation_hold_polls += quiet_hold.polls
            reactivation_hold_elapsed_s = round(
                reactivation_hold_elapsed_s + quiet_hold.elapsed_s,
                3,
            )
        public_after = probe_public_host_availability(
            settings=settings,
            timeout_s=probe_timeout_s,
        )
        backend_properties = fetch_remote_service_properties(
            executor=executor,
            service_name=settings.backend_service,
            properties=tuple(_DEFAULT_CHONKY_PROPERTIES),
        )
    finally:
        if maintenance_permit_open:
            remove_remote_host_control_permit(
                executor=executor,
            )
    elapsed_s = round(time.perf_counter() - started, 3)
    diagnosis = "public_ready_after_host_stabilization"
    if system_reactivated or user_reactivated:
        diagnosis = "conflict_units_reactivated_after_host_stabilization"
    elif not public_before.ready and public_after.ready:
        diagnosis = "public_recovered_after_host_stabilization"
    elif not public_after.ready:
        diagnosis = "public_still_unhealthy_after_host_stabilization"
    merged_action = _merge_stabilization_actions(action, recovery_action)
    return RemoteChonkyDBHostStabilizationResult(
        ok=public_after.ready and not system_reactivated and not user_reactivated,
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
        kill_switch_paths=merged_action.kill_switch_paths,
        disabled_system_units=merged_action.disabled_system_units,
        disabled_user_units=merged_action.disabled_user_units,
        terminated_processes=merged_action.terminated_processes,
        reactivation_hold_polls=reactivation_hold_polls,
        reactivation_hold_elapsed_s=reactivation_hold_elapsed_s,
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


def fetch_remote_guard_protected_system_units(
    *,
    executor: RemoteChonkyDBExecutor,
    backend_service: str,
) -> tuple[str, ...]:
    """Return host-guard required system units for operator diagnostics.

    The thh1986 host-control guard restores its own required_units SSOT from the
    remote TESSAIRACT activation profile. Twinr still surfaces that set for
    operator forensics, but host-contention stabilization no longer treats it
    as an exclusion lane because those same units can be the proven contention
    source for Twinr's required dedicated backend.
    """

    payload = _run_remote_python_json(
        executor=executor,
        code=_REMOTE_GUARD_PROTECTED_UNITS_CODE,
        payload={
            "backend_service": backend_service,
            "guard_unit": _HOST_CONTROL_GUARD_UNIT,
        },
        use_sudo=False,
    )
    raw_items = payload.get("units")
    if not isinstance(raw_items, list):  # pragma: no cover - defensive guard
        raise RuntimeError(f"invalid_remote_guard_protected_units_payload:{payload!r}")
    units: list[str] = []
    for item in raw_items:
        if not isinstance(item, dict):  # pragma: no cover - defensive guard
            raise RuntimeError(f"invalid_remote_guard_protected_unit_item:{item!r}")
        unit_name = str(item.get("unit", "")).strip()
        if unit_name:
            units.append(unit_name)
    return tuple(dict.fromkeys(units))


def ensure_remote_host_boot_pacing_policy(
    *,
    executor: RemoteChonkyDBExecutor,
    system_units: tuple[str, ...],
    user_units: tuple[str, ...],
    user_unit_owner: str,
) -> RemoteHostBootPacingStatus:
    """Install the reboot-persistent boot pacing policy on the remote host."""

    steps = build_remote_host_boot_pacing_steps(
        system_units=system_units,
        user_units=user_units,
    )
    system_unit_set = set(system_units)
    always_disabled_system_units = tuple(
        unit for unit in _DEFAULT_ALWAYS_DISABLED_SYSTEM_UNITS if unit in system_unit_set
    )
    payload = _run_remote_python_json(
        executor=executor,
        code=_REMOTE_SYNC_BOOT_PACING_CODE,
        payload={
            "service_name": _HOST_BOOT_PACER_SERVICE,
            "script_path": _HOST_BOOT_PACER_SCRIPT_PATH,
            "config_path": _HOST_BOOT_PACER_CONFIG_PATH,
            "release_root": _HOST_BOOT_PACER_RELEASE_ROOT,
            "dropin_name": _HOST_BOOT_PACER_DROPIN_NAME,
            "user_unit_owner": user_unit_owner,
            "script_contents": _REMOTE_BOOT_PACER_SCRIPT,
            "default_target": _HOST_SAFE_DEFAULT_TARGET,
            "steps": [
                {
                    "scope": step.scope,
                    "units": list(step.units),
                    "sleep_before_s": step.sleep_before_s,
                    "inter_unit_delay_s": step.inter_unit_delay_s,
                }
                for step in steps
            ],
            "all_system_units": list(system_units),
            "always_disabled_system_units": list(always_disabled_system_units),
            "all_user_units": list(user_units),
        },
        use_sudo=True,
    )
    raw_system_units = payload.get("paced_system_units")
    raw_user_units = payload.get("paced_user_units")
    raw_always_disabled_system_units = payload.get("always_disabled_system_units")
    if (
        not isinstance(raw_system_units, list)
        or not isinstance(raw_user_units, list)
        or not isinstance(raw_always_disabled_system_units, list)
    ):
        raise RuntimeError(f"invalid_remote_boot_pacing_payload:{payload!r}")
    return RemoteHostBootPacingStatus(
        service_name=str(payload.get("service_name", "")).strip(),
        script_path=str(payload.get("script_path", "")).strip(),
        config_path=str(payload.get("config_path", "")).strip(),
        release_root=str(payload.get("release_root", "")).strip(),
        paced_system_units=tuple(
            str(item).strip() for item in raw_system_units if str(item).strip()
        ),
        paced_user_units=tuple(
            str(item).strip() for item in raw_user_units if str(item).strip()
        ),
        always_disabled_system_units=tuple(
            str(item).strip()
            for item in raw_always_disabled_system_units
            if str(item).strip()
        ),
        default_target=str(payload.get("default_target", "")).strip(),
    )


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
    """Touch kill-switches, raise backend weights, and disable conflict units.

    The remote helper deliberately inserts a short per-unit quiesce pause after
    each stop/disable step, and a longer pause after kill-fallbacks, so the
    shared host can reclaim cgroup resources incrementally instead of taking one
    large burst of unit-control churn.
    """

    payload = _run_remote_python_json(
        executor=executor,
        code=_REMOTE_STABILIZE_HOST_CODE,
        payload={
            "backend_service": backend_service,
            "system_units": list(system_units),
            "user_units": list(user_units),
            "user_unit_owner": user_unit_owner,
            "kill_switch_paths": list(kill_switch_paths),
            "runtime_block_unblock_path": _HOST_STABILIZER_UNBLOCK_PATH,
            "property_assignments": dict(property_assignments),
            "stale_process_rules": [dict(item) for item in stale_process_rules],
            "stale_process_min_elapsed_s": float(stale_process_min_elapsed_s),
            "unit_quiesce_s": _DEFAULT_UNIT_QUIESCE_S,
            "killed_unit_quiesce_s": _DEFAULT_KILLED_UNIT_QUIESCE_S,
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
        # Stage code and JSON payload into root-owned temp files so the remote
        # sudo helper does not depend on fragile nested shell/Python quoting.
        completed = executor.run_sudo_ssh(
            _build_sudo_remote_python_helper_script(
                code=code,
                payload_text=input_text,
            )
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


def _build_sudo_remote_python_helper_script(*, code: str, payload_text: str) -> str:
    """Return a remote sudo helper that runs Python from staged temp files."""

    normalized_code = code if code.endswith("\n") else code + "\n"
    normalized_payload = payload_text if payload_text.endswith("\n") else payload_text + "\n"
    code_delimiter = _select_heredoc_delimiter(
        prefix="TWINR_REMOTE_CODE",
        content=normalized_code,
    )
    payload_delimiter = _select_heredoc_delimiter(
        prefix="TWINR_REMOTE_PAYLOAD",
        content=normalized_payload,
    )
    return (
        "tmp_code=$(mktemp)\n"
        "tmp_payload=$(mktemp)\n"
        "cleanup() {\n"
        "    rm -f \"$tmp_code\" \"$tmp_payload\"\n"
        "}\n"
        "trap cleanup EXIT\n"
        f"cat > \"$tmp_code\" <<'{code_delimiter}'\n"
        f"{normalized_code}"
        f"{code_delimiter}\n"
        f"cat > \"$tmp_payload\" <<'{payload_delimiter}'\n"
        f"{normalized_payload}"
        f"{payload_delimiter}\n"
        "python3 \"$tmp_code\" < \"$tmp_payload\"\n"
    )


def _select_heredoc_delimiter(*, prefix: str, content: str) -> str:
    """Pick a heredoc delimiter that does not occur inside the content."""

    delimiter = prefix
    suffix = 0
    while delimiter in content:
        suffix += 1
        delimiter = f"{prefix}_{suffix}"
    return delimiter
