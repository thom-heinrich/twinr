"""Diagnose and optionally repair the dedicated remote ChonkyDB backend.

This module separates three failure layers for Twinr's remote-memory backend:
the public Twinr-facing URL, the dedicated backend systemd unit on `thh1986`,
and the backend's local loopback HTTP surface on `127.0.0.1:3044`.

The repair flow exists to prevent blind restarts. If the public endpoint is
down but the backend loopback instance is still healthy, restarting the backend
would only add avoidable downtime while leaving the real proxy/routing fault
unfixed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any, Mapping
import urllib.error
import urllib.request

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.storage._remote_state.shared import _remote_namespace_for_config
from twinr.ops.pi_runtime_deploy_remote import PiRemoteExecutor
from twinr.ops.remote_systemd_restart_guard import (
    RemoteManualRestartProtectionStatus,
    ensure_remote_service_manual_restart_protection,
    guarded_restart_remote_service,
)
from twinr.ops.self_coding_pi import PiConnectionSettings


_SYSTEMD_HEALTHY_SUBSTATES = frozenset({"running", "listening", "exited"})


@dataclass(frozen=True, slots=True)
class RemoteChonkyDBOpsSettings:
    """Hold the local and remote operator settings for ChonkyDB repair."""

    public_base_url: str
    public_api_key: str
    public_api_key_header: str
    ops_public_base_url: str
    backend_local_base_url: str
    backend_service: str
    runtime_namespace: str
    ssh: PiConnectionSettings


@dataclass(frozen=True, slots=True)
class ChonkyDBHttpProbeResult:
    """Describe one HTTP probe result for the public or backend endpoint."""

    label: str
    ok: bool
    status_code: int
    ready: bool
    detail: str
    url: str = ""
    payload: dict[str, object] | None = None
    error: str | None = None

    @classmethod
    def skipped(cls, *, label: str, detail: str) -> ChonkyDBHttpProbeResult:
        """Return a normalized placeholder probe that was intentionally skipped."""

        return cls(
            label=label,
            ok=False,
            status_code=0,
            ready=False,
            detail=detail,
        )


@dataclass(frozen=True, slots=True)
class ChonkyDBRemoteServiceState:
    """Summarize the dedicated backend service state on the remote host."""

    active_state: str
    sub_state: str
    service_result: str
    load_state: str = ""
    exec_main_pid: int | None = None
    exec_main_status: int | None = None
    active_enter_timestamp: str = ""
    exec_main_start_timestamp: str = ""
    exec_main_exit_timestamp: str = ""

    @property
    def healthy(self) -> bool:
        """Return whether the systemd unit is loaded and in a healthy substate."""

        if self.load_state and self.load_state != "loaded":
            return False
        return self.active_state == "active" and self.sub_state in _SYSTEMD_HEALTHY_SUBSTATES


@dataclass(frozen=True, slots=True)
class RemoteChonkyDBRepairPlan:
    """Describe the bounded action Twinr should take for one outage state."""

    action: str
    reason: str


@dataclass(frozen=True, slots=True)
class RemoteChonkyDBRepairResult:
    """Describe one diagnose-and-repair run against the remote backend."""

    ok: bool
    action_taken: str
    diagnosis: str
    elapsed_s: float
    restart_protection: RemoteManualRestartProtectionStatus
    public_before: ChonkyDBHttpProbeResult
    backend_service_before: ChonkyDBRemoteServiceState
    backend_before: ChonkyDBHttpProbeResult
    public_after: ChonkyDBHttpProbeResult
    backend_service_after: ChonkyDBRemoteServiceState
    backend_after: ChonkyDBHttpProbeResult
    plan: RemoteChonkyDBRepairPlan

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation of the nested repair result."""

        return {
            "ok": self.ok,
            "action_taken": self.action_taken,
            "diagnosis": self.diagnosis,
            "elapsed_s": self.elapsed_s,
            "restart_protection": asdict(self.restart_protection),
            "public_before": asdict(self.public_before),
            "backend_service_before": asdict(self.backend_service_before),
            "backend_before": asdict(self.backend_before),
            "public_after": asdict(self.public_after),
            "backend_service_after": asdict(self.backend_service_after),
            "backend_after": asdict(self.backend_after),
            "plan": asdict(self.plan),
        }


class RemoteChonkyDBExecutor:
    """Run bounded SSH commands against the dedicated remote ChonkyDB host."""

    def __init__(
        self,
        *,
        settings: PiConnectionSettings,
        subprocess_runner: Any = subprocess.run,
        timeout_s: float = 60.0,
    ) -> None:
        self.settings = settings
        self._subprocess_runner = subprocess_runner
        self.timeout_s = float(timeout_s)
        self._delegate = PiRemoteExecutor(
            settings=settings,
            subprocess_runner=subprocess_runner,
            timeout_s=timeout_s,
        )

    def run_ssh(self, script: str, *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
        """Run one remote shell script without exposing SSH secrets in the environment."""

        args = [
            "ssh",
            *self._delegate._ssh_common_args(for_scp=False),  # pylint: disable=protected-access
            self._delegate.remote_spec,
            "bash -lc " + shlex.quote("set -euo pipefail; " + script),
        ]
        return _run_local_command(
            args,
            password=self.settings.password,
            input_text=input_text,
            subprocess_runner=self._subprocess_runner,
            timeout_s=self.timeout_s,
        )

    def run_sudo_ssh(self, script: str) -> subprocess.CompletedProcess[str]:
        """Run one remote shell script under sudo using stdin for the password."""

        password = str(self.settings.password or "").strip()
        if not password:
            raise RuntimeError("remote sudo requires TWINR_CHONKYDB_OPS_BACKEND_SSH_PW")
        return self.run_ssh(
            "sudo -S -p '' bash -lc " + shlex.quote("set -euo pipefail; " + script),
            input_text=password + "\n",
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the remote ChonkyDB repair helper."""

    parser = argparse.ArgumentParser(
        description=(
            "Diagnose the public Twinr ChonkyDB endpoint against the backend service and "
            "restart the backend only when it is the proven failing layer."
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
        help="Timeout in seconds for individual public or backend HTTP probes.",
    )
    parser.add_argument(
        "--ssh-timeout-s",
        type=float,
        default=60.0,
        help="Timeout in seconds for individual backend SSH commands.",
    )
    parser.add_argument(
        "--wait-ready-s",
        type=float,
        default=120.0,
        help="Maximum wait time after a restart for the public endpoint to recover.",
    )
    parser.add_argument(
        "--poll-interval-s",
        type=float,
        default=3.0,
        help="Polling interval while waiting for recovery after a restart.",
    )
    parser.add_argument(
        "--no-restart",
        action="store_true",
        help="Only diagnose the failure layers; never restart the backend service.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the remote ChonkyDB diagnose-and-repair flow and print JSON."""

    args = build_parser().parse_args(argv)
    settings = load_remote_chonkydb_ops_settings(
        env_file=args.env_file,
        ops_env_file=args.ops_env_file,
    )
    result = repair_remote_chonkydb(
        settings=settings,
        probe_timeout_s=args.probe_timeout_s,
        ssh_timeout_s=args.ssh_timeout_s,
        wait_ready_s=args.wait_ready_s,
        poll_interval_s=args.poll_interval_s,
        restart_if_needed=not args.no_restart,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False))
    return 0 if result.ok else 1


def load_remote_chonkydb_ops_settings(
    *,
    env_file: Path,
    ops_env_file: Path,
) -> RemoteChonkyDBOpsSettings:
    """Load Twinr runtime and backend operator settings from the repo env files."""

    config = TwinrConfig.from_env(env_file)
    ops_values = _read_env_values(ops_env_file)
    backend_host = _require_env_value(
        ops_values,
        "TWINR_CHONKYDB_OPS_BACKEND_SSH_HOST",
        source_path=ops_env_file,
    )
    backend_user = _require_env_value(
        ops_values,
        "TWINR_CHONKYDB_OPS_BACKEND_SSH_USER",
        source_path=ops_env_file,
    )
    backend_password = _require_env_value(
        ops_values,
        "TWINR_CHONKYDB_OPS_BACKEND_SSH_PW",
        source_path=ops_env_file,
    )
    backend_service = _require_env_value(
        ops_values,
        "TWINR_CHONKYDB_OPS_BACKEND_SERVICE",
        source_path=ops_env_file,
    )
    backend_local_base_url = _normalize_base_url(
        _require_env_value(
            ops_values,
            "TWINR_CHONKYDB_OPS_BACKEND_LOCAL_BASE_URL",
            source_path=ops_env_file,
        )
    )
    runtime_public_base_url = _normalize_base_url(str(config.chonkydb_base_url or ""))
    if not runtime_public_base_url:
        raise ValueError(f"{env_file} does not define TWINR_CHONKYDB_BASE_URL")
    public_api_key = str(config.chonkydb_api_key or "").strip()
    if not public_api_key:
        raise ValueError(f"{env_file} does not define TWINR_CHONKYDB_API_KEY")
    public_api_key_header = str(config.chonkydb_api_key_header or "x-api-key").strip() or "x-api-key"
    raw_port = str(ops_values.get("TWINR_CHONKYDB_OPS_BACKEND_SSH_PORT", "22") or "22").strip()
    try:
        backend_port = int(raw_port)
    except ValueError as exc:
        raise ValueError(f"{ops_env_file} has an invalid backend SSH port: {raw_port!r}") from exc
    return RemoteChonkyDBOpsSettings(
        public_base_url=runtime_public_base_url,
        public_api_key=public_api_key,
        public_api_key_header=public_api_key_header,
        ops_public_base_url=_normalize_base_url(
            str(ops_values.get("TWINR_CHONKYDB_OPS_PUBLIC_BASE_URL", "") or "")
        ),
        backend_local_base_url=backend_local_base_url,
        backend_service=backend_service,
        runtime_namespace=_remote_namespace_for_config(config),
        ssh=PiConnectionSettings(
            host=backend_host,
            user=backend_user,
            password=backend_password,
            port=backend_port,
        ),
    )


def plan_remote_chonkydb_repair(
    *,
    public_probe: ChonkyDBHttpProbeResult,
    backend_service: ChonkyDBRemoteServiceState,
    backend_probe: ChonkyDBHttpProbeResult,
) -> RemoteChonkyDBRepairPlan:
    """Return the bounded repair action for the current outage evidence."""

    if public_probe.ready:
        return RemoteChonkyDBRepairPlan(action="none", reason="public_ready")
    if not backend_service.healthy:
        return RemoteChonkyDBRepairPlan(
            action="restart_backend_service",
            reason="backend_service_inactive",
        )
    if backend_probe.ready:
        return RemoteChonkyDBRepairPlan(
            action="none",
            reason="public_proxy_unhealthy",
        )
    return RemoteChonkyDBRepairPlan(
        action="restart_backend_service",
        reason="backend_local_unhealthy",
    )


def repair_remote_chonkydb(
    *,
    settings: RemoteChonkyDBOpsSettings,
    probe_timeout_s: float,
    ssh_timeout_s: float,
    wait_ready_s: float,
    poll_interval_s: float,
    restart_if_needed: bool,
    subprocess_runner: Any = subprocess.run,
) -> RemoteChonkyDBRepairResult:
    """Diagnose and optionally repair the dedicated remote ChonkyDB backend."""

    started = time.perf_counter()
    executor = RemoteChonkyDBExecutor(
        settings=settings.ssh,
        subprocess_runner=subprocess_runner,
        timeout_s=ssh_timeout_s,
    )
    restart_protection = ensure_remote_service_manual_restart_protection(
        executor=executor,
        service_name=settings.backend_service,
    )
    public_before = probe_public_chonkydb(
        settings=settings,
        timeout_s=probe_timeout_s,
    )
    backend_service_before = fetch_backend_service_state(
        executor=executor,
        service_name=settings.backend_service,
    )
    backend_before = probe_backend_local_chonkydb(
        executor=executor,
        settings=settings,
        timeout_s=probe_timeout_s,
    )
    plan = plan_remote_chonkydb_repair(
        public_probe=public_before,
        backend_service=backend_service_before,
        backend_probe=backend_before,
    )
    action_taken = "none"
    public_after = public_before
    backend_service_after = backend_service_before
    backend_after = backend_before

    if plan.action == "restart_backend_service":
        if restart_if_needed:
            restart_backend_service(
                executor=executor,
                service_name=settings.backend_service,
            )
            action_taken = "restart_backend_service"
            public_after, backend_service_after, backend_after = wait_for_backend_recovery(
                executor=executor,
                settings=settings,
                probe_timeout_s=probe_timeout_s,
                wait_ready_s=wait_ready_s,
                poll_interval_s=poll_interval_s,
            )
        else:
            action_taken = "restart_required_but_skipped"

    elapsed_s = round(time.perf_counter() - started, 3)
    ok = public_after.ready
    diagnosis = plan.reason
    if plan.reason == "public_proxy_unhealthy" and not public_after.ready:
        diagnosis = "public_proxy_unhealthy"
    elif action_taken == "restart_required_but_skipped":
        diagnosis = "backend_restart_required"
    elif action_taken == "restart_backend_service" and not public_after.ready:
        diagnosis = "backend_restart_did_not_restore_public_health"
    return RemoteChonkyDBRepairResult(
        ok=ok,
        action_taken=action_taken,
        diagnosis=diagnosis,
        elapsed_s=elapsed_s,
        restart_protection=restart_protection,
        public_before=public_before,
        backend_service_before=backend_service_before,
        backend_before=backend_before,
        public_after=public_after,
        backend_service_after=backend_service_after,
        backend_after=backend_after,
        plan=plan,
    )


def probe_public_chonkydb(
    *,
    settings: RemoteChonkyDBOpsSettings,
    timeout_s: float,
) -> ChonkyDBHttpProbeResult:
    """Probe the public Twinr-facing ChonkyDB endpoint using runtime auth."""

    instance_probe = _probe_http_json(
        label="public",
        url=settings.public_base_url.rstrip("/") + "/v1/external/instance",
        headers={
            "Accept": "application/json",
            settings.public_api_key_header: settings.public_api_key,
        },
        timeout_s=timeout_s,
    )
    if not instance_probe.ready:
        return instance_probe
    query_probe = _probe_http_json(
        label="public",
        url=settings.public_base_url.rstrip("/") + "/v1/external/retrieve/topk_records",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            settings.public_api_key_header: settings.public_api_key,
        },
        timeout_s=min(float(timeout_s), 10.0),
        method="POST",
        json_body=_query_surface_canary_payload(runtime_namespace=settings.runtime_namespace),
    )
    return _require_query_surface_ready(instance_probe=instance_probe, query_probe=query_probe)


def fetch_backend_service_state(
    *,
    executor: RemoteChonkyDBExecutor,
    service_name: str,
) -> ChonkyDBRemoteServiceState:
    """Return the current systemd state for the dedicated backend service."""

    completed = executor.run_sudo_ssh(
        "systemctl show "
        + shlex.quote(service_name)
        + " --no-pager -p LoadState -p ActiveState -p SubState -p Result "
        + "-p ExecMainPID -p ExecMainStatus -p ActiveEnterTimestamp "
        + "-p ExecMainStartTimestamp -p ExecMainExitTimestamp"
    )
    values = _parse_key_value_lines(completed.stdout)
    return ChonkyDBRemoteServiceState(
        load_state=str(values.get("LoadState", "")).strip(),
        active_state=str(values.get("ActiveState", "")).strip(),
        sub_state=str(values.get("SubState", "")).strip(),
        service_result=str(values.get("Result", "")).strip(),
        exec_main_pid=_parse_optional_int(values.get("ExecMainPID")),
        exec_main_status=_parse_optional_int(values.get("ExecMainStatus")),
        active_enter_timestamp=str(values.get("ActiveEnterTimestamp", "")).strip(),
        exec_main_start_timestamp=str(values.get("ExecMainStartTimestamp", "")).strip(),
        exec_main_exit_timestamp=str(values.get("ExecMainExitTimestamp", "")).strip(),
    )


def probe_backend_local_chonkydb(
    *,
    executor: RemoteChonkyDBExecutor,
    settings: RemoteChonkyDBOpsSettings,
    timeout_s: float,
) -> ChonkyDBHttpProbeResult:
    """Probe the backend loopback instance on the remote host."""

    env_output = executor.run_sudo_ssh(
        "systemctl show -p Environment --no-pager " + shlex.quote(settings.backend_service)
    ).stdout
    env_map = _parse_systemd_environment_output(env_output)
    api_key = (
        str(env_map.get("CHONKDB_API_KEY") or env_map.get("CCODEX_MEMORY_API_KEY") or "").strip()
        or settings.public_api_key
    )
    api_key_header = (
        str(
            env_map.get("CHONKDB_API_KEY_HEADER")
            or env_map.get("CCODEX_MEMORY_API_KEY_HEADER")
            or ""
        ).strip()
        or settings.public_api_key_header
    )
    if not api_key:
        return ChonkyDBHttpProbeResult(
            label="backend",
            ok=False,
            status_code=0,
            ready=False,
            detail="missing_backend_auth_material",
            url=settings.backend_local_base_url.rstrip("/") + "/v1/external/instance",
            error="systemd service environment does not expose a backend API key",
        )
    instance_probe_payload = {
        "url": settings.backend_local_base_url.rstrip("/") + "/v1/external/instance",
        "api_key": api_key,
        "api_key_header": api_key_header,
        "timeout_s": float(timeout_s),
    }
    completed = executor.run_ssh(
        "python3 -c " + shlex.quote(_REMOTE_HTTP_PROBE_CODE),
        input_text=json.dumps(instance_probe_payload, ensure_ascii=False),
    )
    instance_probe = _probe_result_from_remote_payload(json.loads(completed.stdout))
    if not instance_probe.ready:
        return instance_probe
    query_probe_payload = {
        "url": settings.backend_local_base_url.rstrip("/") + "/v1/external/retrieve/topk_records",
        "api_key": api_key,
        "api_key_header": api_key_header,
        "timeout_s": min(float(timeout_s), 10.0),
        "method": "POST",
        "json_body": _query_surface_canary_payload(runtime_namespace=settings.runtime_namespace),
    }
    query_completed = executor.run_ssh(
        "python3 -c " + shlex.quote(_REMOTE_HTTP_PROBE_CODE),
        input_text=json.dumps(query_probe_payload, ensure_ascii=False),
    )
    query_probe = _probe_result_from_remote_payload(json.loads(query_completed.stdout))
    return _require_query_surface_ready(instance_probe=instance_probe, query_probe=query_probe)


def restart_backend_service(
    *,
    executor: RemoteChonkyDBExecutor,
    service_name: str,
) -> None:
    """Restart the dedicated backend service under sudo."""

    guarded_restart_remote_service(
        executor=executor,
        service_name=service_name,
    )


def wait_for_backend_recovery(
    *,
    executor: RemoteChonkyDBExecutor,
    settings: RemoteChonkyDBOpsSettings,
    probe_timeout_s: float,
    wait_ready_s: float,
    poll_interval_s: float,
) -> tuple[ChonkyDBHttpProbeResult, ChonkyDBRemoteServiceState, ChonkyDBHttpProbeResult]:
    """Wait until the public endpoint recovers or the recovery budget expires."""

    deadline = time.perf_counter() + max(0.0, float(wait_ready_s))
    latest_public = ChonkyDBHttpProbeResult.skipped(
        label="public",
        detail="recovery_not_probed",
    )
    latest_service = ChonkyDBRemoteServiceState(
        load_state="",
        active_state="",
        sub_state="",
        service_result="",
    )
    latest_backend = ChonkyDBHttpProbeResult.skipped(
        label="backend",
        detail="recovery_not_probed",
    )
    while True:
        latest_service = fetch_backend_service_state(
            executor=executor,
            service_name=settings.backend_service,
        )
        latest_backend = probe_backend_local_chonkydb(
            executor=executor,
            settings=settings,
            timeout_s=probe_timeout_s,
        )
        latest_public = probe_public_chonkydb(
            settings=settings,
            timeout_s=probe_timeout_s,
        )
        if latest_public.ready:
            return latest_public, latest_service, latest_backend
        if time.perf_counter() >= deadline:
            return latest_public, latest_service, latest_backend
        time.sleep(max(0.1, float(poll_interval_s)))


def _probe_http_json(
    *,
    label: str,
    url: str,
    headers: Mapping[str, str],
    timeout_s: float,
    method: str = "GET",
    json_body: Mapping[str, Any] | None = None,
) -> ChonkyDBHttpProbeResult:
    """Run one local HTTP JSON probe and normalize the response."""

    body_bytes = None
    if json_body is not None:
        body_bytes = json.dumps(dict(json_body), ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, headers=dict(headers), method=str(method or "GET").upper(), data=body_bytes)
    try:
        with urllib.request.urlopen(request, timeout=float(timeout_s)) as response:
            body_text = response.read().decode("utf-8", errors="replace")
            status_code = int(getattr(response, "status", 0) or 0)
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        status_code = int(exc.code or 0)
    except Exception as exc:
        return ChonkyDBHttpProbeResult(
            label=label,
            ok=False,
            status_code=0,
            ready=False,
            detail=str(exc).strip() or type(exc).__name__,
            url=url,
            error=type(exc).__name__,
        )
    payload = _parse_json_payload(body_text)
    ready = bool(payload.get("ready", False))
    detail = _probe_detail(payload=payload, body_text=body_text, status_code=status_code)
    return ChonkyDBHttpProbeResult(
        label=label,
        ok=status_code == 200,
        status_code=status_code,
        ready=bool(status_code == 200 and ready),
        detail=detail,
        url=url,
        payload=payload if payload else None,
    )


def _probe_result_from_remote_payload(payload: Mapping[str, Any]) -> ChonkyDBHttpProbeResult:
    """Normalize one remote JSON probe payload into the local dataclass."""

    body_payload = payload.get("payload")
    typed_payload = dict(body_payload) if isinstance(body_payload, Mapping) else None
    status_code = _parse_optional_int(payload.get("status"))
    detail = str(payload.get("detail") or "").strip()
    if not detail and typed_payload:
        detail = _probe_detail(
            payload=typed_payload,
            body_text=json.dumps(typed_payload, ensure_ascii=False),
            status_code=int(status_code or 0),
        )
    if not detail:
        detail = str(payload.get("error") or "remote_probe_failed").strip()
    ready = bool(typed_payload and typed_payload.get("ready", False))
    return ChonkyDBHttpProbeResult(
        label="backend",
        ok=int(status_code or 0) == 200,
        status_code=int(status_code or 0),
        ready=bool(int(status_code or 0) == 200 and ready),
        detail=detail,
        url=str(payload.get("url") or ""),
        payload=typed_payload,
        error=str(payload.get("error") or "").strip() or None,
    )


def _query_surface_canary_payload(*, runtime_namespace: str) -> dict[str, object]:
    """Return a tiny current-scope query canary for ChonkyDB read readiness."""

    return {
        "query_text": "memory probe",
        "result_limit": 1,
        "include_content": False,
        "include_metadata": False,
        "max_content_chars": 128,
        "scope_ref": "longterm:objects:current",
        "namespace": str(runtime_namespace or "").strip(),
    }


def _require_query_surface_ready(
    *,
    instance_probe: ChonkyDBHttpProbeResult,
    query_probe: ChonkyDBHttpProbeResult,
) -> ChonkyDBHttpProbeResult:
    """Only mark ChonkyDB ready when the query surface works after `/instance`."""

    if query_probe.ok:
        return instance_probe
    return ChonkyDBHttpProbeResult(
        label=instance_probe.label,
        ok=False,
        status_code=query_probe.status_code,
        ready=False,
        detail=f"query_surface_unhealthy: {query_probe.detail}",
        url=query_probe.url,
        payload=query_probe.payload,
        error=query_probe.error,
    )


def _probe_detail(
    *,
    payload: Mapping[str, Any],
    body_text: str,
    status_code: int,
) -> str:
    """Return one compact operator detail string for a probe result."""

    for key in ("detail", "error", "title", "service", "status"):
        value = payload.get(key)
        text = str(value or "").strip()
        if text:
            return text
    body_head = " ".join(str(body_text or "").split())[:220]
    if body_head:
        return body_head
    return f"http_{status_code}"


def _parse_json_payload(body_text: str) -> dict[str, object]:
    """Parse a JSON payload or return an empty mapping on malformed bodies."""

    try:
        payload = json.loads(body_text)
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _parse_key_value_lines(text: str) -> dict[str, str]:
    """Parse simple `KEY=VALUE` output blocks such as `systemctl show`."""

    mapping: dict[str, str] = {}
    for raw_line in str(text or "").splitlines():
        key, separator, value = raw_line.partition("=")
        if separator:
            mapping[str(key).strip()] = str(value).strip()
    return mapping


def _parse_systemd_environment_output(text: str) -> dict[str, str]:
    """Parse `systemctl show -p Environment` output into one env mapping."""

    raw_value = str(text or "").strip()
    if raw_value.startswith("Environment="):
        raw_value = raw_value[len("Environment=") :]
    if not raw_value:
        return {}
    try:
        tokens = shlex.split(raw_value)
    except ValueError:
        tokens = [part for part in raw_value.split(" ") if part]
    env_map: dict[str, str] = {}
    for token in tokens:
        key, separator, value = str(token).partition("=")
        if separator:
            env_map[str(key)] = str(value)
    return env_map


def _read_env_values(path: Path) -> dict[str, str]:
    """Parse one dotenv file into a plain mapping without shell evaluation."""

    if not path.exists():
        raise FileNotFoundError(f"env file not found: {path}")
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[str(key).strip()] = str(value).strip().strip("'").strip('"')
    return values


def _normalize_base_url(raw_value: str) -> str:
    """Normalize a base URL to a stripped form without a trailing slash."""

    return str(raw_value or "").strip().rstrip("/")


def _require_env_value(values: Mapping[str, str], key: str, *, source_path: Path) -> str:
    """Return one required env value or raise a precise configuration error."""

    value = str(values.get(key, "")).strip()
    if not value:
        raise ValueError(f"{source_path} is missing {key}")
    return value


def _parse_optional_int(value: object) -> int | None:
    """Convert a systemd field to `int` when it carries a real number."""

    text = str(value or "").strip()
    if not text or text in {"n/a", "0"}:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _run_local_command(
    args: list[str],
    *,
    password: str | None,
    input_text: str | None,
    subprocess_runner: Any,
    timeout_s: float,
) -> subprocess.CompletedProcess[str]:
    """Run one local subprocess, using `sshpass -d` when a password is present."""

    if password:
        return _run_local_with_sshpass(
            args,
            password=password,
            input_text=input_text,
            subprocess_runner=subprocess_runner,
            timeout_s=timeout_s,
        )
    return _run_local(
        args,
        input_text=input_text,
        pass_fds=(),
        subprocess_runner=subprocess_runner,
        timeout_s=timeout_s,
    )


def _run_local_with_sshpass(
    args: list[str],
    *,
    password: str,
    input_text: str | None,
    subprocess_runner: Any,
    timeout_s: float,
) -> subprocess.CompletedProcess[str]:
    """Run one local subprocess while keeping the SSH password off stdin."""

    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, (password + "\n").encode("utf-8"))
        os.close(write_fd)
        write_fd = -1
        return _run_local(
            ["sshpass", "-d", str(read_fd), *args],
            input_text=input_text,
            pass_fds=(read_fd,),
            subprocess_runner=subprocess_runner,
            timeout_s=timeout_s,
        )
    finally:
        if write_fd >= 0:
            os.close(write_fd)
        os.close(read_fd)


def _run_local(
    args: list[str],
    *,
    input_text: str | None,
    pass_fds: tuple[int, ...],
    subprocess_runner: Any,
    timeout_s: float,
) -> subprocess.CompletedProcess[str]:
    """Run one local subprocess and raise a concise error on failure."""

    completed = subprocess_runner(
        args,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        input=input_text,
        timeout=float(timeout_s),
        pass_fds=pass_fds,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip()
        if not message:
            message = f"command failed: {' '.join(args)}"
        raise RuntimeError(message)
    return completed


_REMOTE_HTTP_PROBE_CODE = """
from __future__ import annotations

import json
import urllib.error
import urllib.request
import sys

payload = json.load(sys.stdin)
url = str(payload.get("url") or "").strip()
api_key = str(payload.get("api_key") or "").strip()
api_key_header = str(payload.get("api_key_header") or "x-api-key").strip() or "x-api-key"
timeout_s = float(payload.get("timeout_s") or 20.0)
method = str(payload.get("method") or "GET").strip().upper() or "GET"
json_body = payload.get("json_body")
headers = {"Accept": "application/json"}
if api_key:
    headers[api_key_header] = api_key
body_bytes = None
if json_body is not None:
    headers["Content-Type"] = "application/json"
    body_bytes = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
request = urllib.request.Request(url, headers=headers, method=method, data=body_bytes)
try:
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body_text = response.read().decode("utf-8", errors="replace")
        status = int(getattr(response, "status", 0) or 0)
except urllib.error.HTTPError as exc:
    body_text = exc.read().decode("utf-8", errors="replace")
    status = int(exc.code or 0)
except Exception as exc:
    print(json.dumps({"url": url, "status": 0, "detail": str(exc), "error": type(exc).__name__}))
    raise SystemExit(0)
try:
    parsed = json.loads(body_text)
    payload_dict = dict(parsed) if isinstance(parsed, dict) else {}
except Exception:
    payload_dict = {}
detail = ""
for key in ("detail", "error", "title", "service", "status"):
    text = str(payload_dict.get(key) or "").strip()
    if text:
        detail = text
        break
if not detail:
    detail = " ".join(body_text.split())[:220]
print(json.dumps({"url": url, "status": status, "detail": detail, "payload": payload_dict}, ensure_ascii=False))
"""
