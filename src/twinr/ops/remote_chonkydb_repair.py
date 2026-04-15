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
from twinr.memory.chonkydb.client import ChonkyDBError
from twinr.memory.longterm.storage._remote_state.shared import is_explicit_remote_transient_detail
from twinr.memory.longterm.storage._remote_state.shared import _remote_namespace_for_config
from twinr.ops.pi_runtime_deploy_remote import PiRemoteExecutor
from twinr.ops.remote_systemd_restart_guard import (
    RemoteManualRestartProtectionStatus,
    ensure_remote_service_manual_restart_protection,
    guarded_restart_remote_service,
)
from twinr.ops.self_coding_pi import PiConnectionSettings


_SYSTEMD_HEALTHY_SUBSTATES = frozenset({"running", "listening", "exited"})
_DEFAULT_SSH_TIMEOUT_S = 180.0


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
class ForeignBackendConsumer:
    """Describe one non-Twinr unit still pointed at the dedicated backend."""

    unit_name: str
    active_state: str
    sub_state: str
    configured_base_url: str = ""
    fragment_path: str = ""
    coupled_to_backend_service: bool = False

    @property
    def active(self) -> bool:
        """Return whether the foreign consumer is currently live."""

        return self.active_state == "active" and self.sub_state in _SYSTEMD_HEALTHY_SUBSTATES


@dataclass(frozen=True, slots=True)
class BackendDataOwnershipState:
    """Describe whether the dedicated backend data dir has owner/group drift."""

    data_dir: str
    expected_user: str
    expected_group: str
    mismatched_entry_count: int
    sample_entries: tuple[str, ...] = ()
    truncated: bool = False
    error: str | None = None

    @property
    def has_drift(self) -> bool:
        """Return whether at least one data-dir entry uses the wrong owner/group."""

        return int(self.mismatched_entry_count) > 0


@dataclass(frozen=True, slots=True)
class BackendQuerySurfaceReadinessContract:
    """Describe whether the backend startup contract can prove query readiness.

    The dedicated Twinr backend relies on `CHONKY_FT_REBUILD_ON_OPEN` for
    correctness, but the live `token_fast` serving contract must not block on
    unrelated fulltext warmup. Twinr also requires the sanctioned
    payload-read startup lane to stay enabled; disabling that gate strands the
    dedicated backend on startup even though current-head reads are part of the
    required remote-memory contract.
    """

    fulltext_rebuild_on_open: bool
    warmup_fulltext_gate: bool
    warmup_wait_for_ready: bool
    ready_default_scope: str = "full"
    serving_contract_scope: str = "full"
    warmup_wait_ready_timeout_s: float | None = None
    warmup_payload_read_path: bool = False
    warmup_payload_read_timeout_s: float | None = None

    @property
    def requires_full_scope_warmup(self) -> bool:
        """Return whether the configured serving contract blocks on full scope."""

        scopes = {
            str(self.ready_default_scope or "").strip().lower() or "full",
            str(self.serving_contract_scope or "").strip().lower() or "full",
        }
        return "full" in scopes

    @property
    def unsafe_reason(self) -> str | None:
        """Return the proven unsafe startup mismatch when present."""

        if self.fulltext_rebuild_on_open and self.requires_full_scope_warmup:
            if not self.warmup_fulltext_gate:
                return "fulltext_rebuild_on_open_without_query_gate"
            if not self.warmup_wait_for_ready:
                return "fulltext_rebuild_on_open_without_ready_wait"
            if (
                self.warmup_wait_ready_timeout_s is not None
                and float(self.warmup_wait_ready_timeout_s) <= 0.0
            ):
                return "fulltext_rebuild_on_open_without_ready_timeout"
            return None
        if self.fulltext_rebuild_on_open and self.warmup_fulltext_gate:
            return "token_fast_serving_contract_blocked_by_fulltext_gate"
        if (
            self.fulltext_rebuild_on_open
            and self.warmup_wait_for_ready
            and self.warmup_wait_ready_timeout_s is not None
            and float(self.warmup_wait_ready_timeout_s) <= 0.0
        ):
            return "fulltext_rebuild_on_open_without_ready_timeout"
        return None

    @property
    def unsafe(self) -> bool:
        """Return whether Twinr has proven the startup contract unsafe."""

        return self.unsafe_reason is not None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "fulltext_rebuild_on_open": self.fulltext_rebuild_on_open,
            "warmup_fulltext_gate": self.warmup_fulltext_gate,
            "warmup_wait_for_ready": self.warmup_wait_for_ready,
            "ready_default_scope": self.ready_default_scope,
            "serving_contract_scope": self.serving_contract_scope,
            "requires_full_scope_warmup": self.requires_full_scope_warmup,
            "warmup_wait_ready_timeout_s": self.warmup_wait_ready_timeout_s,
            "warmup_payload_read_path": self.warmup_payload_read_path,
            "warmup_payload_read_timeout_s": self.warmup_payload_read_timeout_s,
            "unsafe": self.unsafe,
            "unsafe_reason": self.unsafe_reason,
        }


@dataclass(frozen=True, slots=True)
class BackendDataOwnershipRepairStatus:
    """Describe whether Twinr repaired data-dir owner/group drift."""

    data_dir: str
    expected_user: str
    expected_group: str
    changed_entry_count: int
    error: str | None = None

    @property
    def changed(self) -> bool:
        """Return whether at least one entry was repaired."""

        return int(self.changed_entry_count) > 0


@dataclass(frozen=True, slots=True)
class BackendPayloadReadWarmupRepairStatus:
    """Describe whether Twinr disabled the proven payload-read startup gate."""

    dropin_path: str
    changed: bool
    contract: BackendQuerySurfaceReadinessContract

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "dropin_path": self.dropin_path,
            "changed": self.changed,
            "contract": self.contract.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BackendPayloadSyncBulkApiReadyContract:
    """Describe whether payload sync bulk incorrectly waits for full API ready."""

    ready_default_scope: str
    payload_sync_bulk_require_api_ready: bool

    @property
    def unsafe_reason(self) -> str | None:
        """Return the proven unsafe contract mismatch when present."""

        if self.payload_sync_bulk_require_api_ready and self.ready_default_scope == "full":
            return "sync_bulk_api_waits_for_full_ready"
        return None

    @property
    def unsafe(self) -> bool:
        """Return whether the dedicated Twinr write surface is incorrectly gated."""

        return self.unsafe_reason is not None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "ready_default_scope": self.ready_default_scope,
            "payload_sync_bulk_require_api_ready": self.payload_sync_bulk_require_api_ready,
            "unsafe": self.unsafe,
            "unsafe_reason": self.unsafe_reason,
        }


@dataclass(frozen=True, slots=True)
class BackendPayloadSyncBulkApiReadyRepairStatus:
    """Describe whether Twinr disabled the sync-bulk API ready gate."""

    dropin_path: str
    changed: bool
    contract: BackendPayloadSyncBulkApiReadyContract

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "dropin_path": self.dropin_path,
            "changed": self.changed,
            "contract": self.contract.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BackendVectorWarmupContract:
    """Describe whether the vector warmup budget can satisfy full ready."""

    ready_default_scope: str
    serving_contract_scope: str
    warmup_wait_for_ready: bool
    warmup_wait_ready_timeout_s: float | None = None
    warmup_vector_open_timeout_s: float | None = None

    @property
    def requires_full_scope_warmup(self) -> bool:
        """Return whether the backend is configured to wait for full readiness."""

        if not self.warmup_wait_for_ready:
            return False
        scopes = {
            str(self.ready_default_scope or "").strip().lower(),
            str(self.serving_contract_scope or "").strip().lower(),
        }
        return "full" in scopes

    @property
    def unsafe_reason(self) -> str | None:
        """Return the proven unsafe vector warmup contract mismatch when present."""

        if not self.requires_full_scope_warmup:
            return None
        ready_budget = self.warmup_wait_ready_timeout_s
        vector_budget = self.warmup_vector_open_timeout_s
        if ready_budget is None or float(ready_budget) <= 0.0:
            return None
        if vector_budget is None or float(vector_budget) <= 0.0:
            return None
        if float(vector_budget) < float(ready_budget):
            return "vector_open_timeout_shorter_than_full_ready_budget"
        return None

    @property
    def unsafe(self) -> bool:
        """Return whether the vector warmup budget is provably too short."""

        return self.unsafe_reason is not None

    @property
    def target_vector_open_timeout_s(self) -> float | None:
        """Return the minimum safe vector-open timeout for the live contract."""

        if not self.requires_full_scope_warmup:
            return None
        ready_budget = self.warmup_wait_ready_timeout_s
        if ready_budget is None or float(ready_budget) <= 0.0:
            return None
        vector_budget = self.warmup_vector_open_timeout_s
        if vector_budget is None:
            return float(ready_budget)
        return max(float(vector_budget), float(ready_budget))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "ready_default_scope": self.ready_default_scope,
            "serving_contract_scope": self.serving_contract_scope,
            "warmup_wait_for_ready": self.warmup_wait_for_ready,
            "warmup_wait_ready_timeout_s": self.warmup_wait_ready_timeout_s,
            "warmup_vector_open_timeout_s": self.warmup_vector_open_timeout_s,
            "requires_full_scope_warmup": self.requires_full_scope_warmup,
            "unsafe": self.unsafe,
            "unsafe_reason": self.unsafe_reason,
            "target_vector_open_timeout_s": self.target_vector_open_timeout_s,
        }


@dataclass(frozen=True, slots=True)
class BackendVectorWarmupRepairStatus:
    """Describe whether Twinr repaired the vector warmup timeout budget."""

    dropin_path: str
    changed: bool
    contract: BackendVectorWarmupContract
    target_vector_open_timeout_s: float | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "dropin_path": self.dropin_path,
            "changed": self.changed,
            "contract": self.contract.to_dict(),
            "target_vector_open_timeout_s": self.target_vector_open_timeout_s,
        }


@dataclass(frozen=True, slots=True)
class BackendQuerySurfaceReadinessRepairStatus:
    """Describe whether Twinr had to harden the backend readiness contract."""

    dropin_path: str
    changed: bool
    contract: BackendQuerySurfaceReadinessContract

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "dropin_path": self.dropin_path,
            "changed": self.changed,
            "contract": self.contract.to_dict(),
        }


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
    foreign_consumers: tuple[ForeignBackendConsumer, ...] = ()
    backend_data_ownership: BackendDataOwnershipState | None = None
    foreign_consumer_inspection_error: str | None = None

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
            "foreign_consumers": [asdict(consumer) for consumer in self.foreign_consumers],
            "backend_data_ownership": (
                asdict(self.backend_data_ownership)
                if self.backend_data_ownership is not None
                else None
            ),
            "foreign_consumer_inspection_error": self.foreign_consumer_inspection_error,
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
        default=_DEFAULT_SSH_TIMEOUT_S,
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
    foreign_consumers: tuple[ForeignBackendConsumer, ...] = (),
    backend_data_ownership: BackendDataOwnershipState | None = None,
    backend_readiness_contract: BackendQuerySurfaceReadinessContract | None = None,
    backend_vector_warmup_contract: BackendVectorWarmupContract | None = None,
) -> RemoteChonkyDBRepairPlan:
    """Return the bounded repair action for the current outage evidence."""

    if public_probe.ready:
        return RemoteChonkyDBRepairPlan(action="none", reason="public_ready")
    payload_read_gate_outage = _payload_read_gate_causes_outage(
        contract=backend_readiness_contract,
        backend_service=backend_service,
        public_probe=public_probe,
        backend_probe=backend_probe,
    )
    query_surface_contract_outage = _query_surface_contract_causes_outage(
        contract=backend_readiness_contract,
        backend_service=backend_service,
        public_probe=public_probe,
        backend_probe=backend_probe,
    )
    vector_warmup_timeout_outage = _vector_warmup_timeout_contract_causes_outage(
        contract=backend_vector_warmup_contract,
        backend_service=backend_service,
        public_probe=public_probe,
        backend_probe=backend_probe,
    )
    if (
        query_surface_contract_outage
        and backend_data_ownership is not None
        and backend_data_ownership.has_drift
    ):
        return RemoteChonkyDBRepairPlan(
            action="repair_backend_startup_contract_and_data_ownership_then_restart_backend_service",
            reason="backend_query_surface_contract_and_data_permission_drift",
        )
    if query_surface_contract_outage:
        return RemoteChonkyDBRepairPlan(
            action="repair_backend_startup_contract_and_restart_backend_service",
            reason="backend_query_surface_contract",
        )
    if payload_read_gate_outage and backend_data_ownership is not None and backend_data_ownership.has_drift:
        return RemoteChonkyDBRepairPlan(
            action="repair_backend_startup_contract_and_data_ownership_then_restart_backend_service",
            reason="backend_payload_read_gate_and_data_permission_drift",
        )
    if payload_read_gate_outage:
        return RemoteChonkyDBRepairPlan(
            action="repair_backend_startup_contract_and_restart_backend_service",
            reason="backend_payload_read_gate",
        )
    if vector_warmup_timeout_outage and backend_data_ownership is not None and backend_data_ownership.has_drift:
        return RemoteChonkyDBRepairPlan(
            action="repair_backend_startup_contract_and_data_ownership_then_restart_backend_service",
            reason="backend_vector_warmup_timeout_budget_and_data_permission_drift",
        )
    if vector_warmup_timeout_outage:
        return RemoteChonkyDBRepairPlan(
            action="repair_backend_startup_contract_and_restart_backend_service",
            reason="backend_vector_warmup_timeout_budget",
        )
    if backend_data_ownership is not None and backend_data_ownership.has_drift:
        return RemoteChonkyDBRepairPlan(
            action="repair_backend_data_ownership_and_restart_backend_service",
            reason="backend_data_permission_drift",
        )
    if not backend_service.healthy:
        return RemoteChonkyDBRepairPlan(
            action="restart_backend_service",
            reason="backend_service_inactive",
        )
    if not backend_probe.ready and any(consumer.active for consumer in foreign_consumers):
        return RemoteChonkyDBRepairPlan(
            action="none",
            reason="backend_foreign_consumer_contention",
        )
    if _backend_probe_indicates_active_but_unresponsive_service(
        backend_service=backend_service,
        backend_probe=backend_probe,
    ):
        return RemoteChonkyDBRepairPlan(
            action="restart_backend_service",
            reason="backend_active_but_unresponsive",
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


def _payload_read_gate_causes_outage(
    *,
    contract: BackendQuerySurfaceReadinessContract | None,
    backend_service: ChonkyDBRemoteServiceState,
    public_probe: ChonkyDBHttpProbeResult,
    backend_probe: ChonkyDBHttpProbeResult,
) -> bool:
    """Return whether a disabled payload-read startup contract is the outage."""

    if contract is None:
        return False
    if not backend_service.healthy:
        return False
    if not contract.warmup_wait_for_ready or contract.warmup_payload_read_path:
        return False
    return is_explicit_service_warmup_detail(
        public_probe.detail,
    ) and is_explicit_service_warmup_detail(backend_probe.detail)


def _query_surface_contract_causes_outage(
    *,
    contract: BackendQuerySurfaceReadinessContract | None,
    backend_service: ChonkyDBRemoteServiceState,
    public_probe: ChonkyDBHttpProbeResult,
    backend_probe: ChonkyDBHttpProbeResult,
) -> bool:
    """Return whether the query-surface readiness contract itself is unsafe."""

    if contract is None or not contract.unsafe:
        return False
    if not backend_service.healthy:
        return False
    if contract.unsafe_reason == "fulltext_rebuild_on_open_without_query_gate":
        return False
    return is_explicit_service_warmup_detail(
        public_probe.detail,
    ) and is_explicit_service_warmup_detail(backend_probe.detail)


def _vector_warmup_timeout_contract_causes_outage(
    *,
    contract: BackendVectorWarmupContract | None,
    backend_service: ChonkyDBRemoteServiceState,
    public_probe: ChonkyDBHttpProbeResult,
    backend_probe: ChonkyDBHttpProbeResult,
) -> bool:
    """Return whether a too-short vector budget strands full-scope startup."""

    if contract is None or not contract.unsafe:
        return False
    if not backend_service.healthy:
        return False
    return is_explicit_service_warmup_detail(
        public_probe.detail,
    ) and is_explicit_service_warmup_detail(backend_probe.detail)


def is_explicit_service_warmup_detail(detail: str | None) -> bool:
    """Return whether one backend detail explicitly reports startup warmup."""

    return is_explicit_remote_transient_detail(detail)


def extract_backend_problem_detail(exc: BaseException | None) -> str | None:
    """Return one bounded backend problem-detail string from the exception chain."""

    if exc is None:
        return None
    for item in _exception_chain(exc):
        if not isinstance(item, ChonkyDBError):
            continue
        response_json = item.response_json if isinstance(item.response_json, dict) else None
        if response_json is not None:
            detail = _normalize_problem_detail(response_json.get("detail"))
            if detail:
                return detail
        detail = _normalize_problem_detail(item.response_text)
        if detail:
            return detail
    return None


def _exception_chain(exc: BaseException) -> tuple[BaseException, ...]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return tuple(chain)


def _normalize_problem_detail(value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) > 240:
        return text[:240].rstrip()
    return text


def _backend_probe_indicates_active_but_unresponsive_service(
    *,
    backend_service: ChonkyDBRemoteServiceState,
    backend_probe: ChonkyDBHttpProbeResult,
) -> bool:
    """Return whether the backend stayed active while its loopback HTTP surface stopped answering.

    Live 2026-04-09 incident evidence showed the dedicated backend keeping an
    `active/running` systemd state after an internal DocID mapping persistence
    failure while `127.0.0.1:3044` timed out completely. Treat that outage
    shape as a distinct operator diagnosis instead of collapsing it into the
    generic local-unhealthy bucket.
    """

    if not backend_service.healthy or backend_probe.ready:
        return False
    if int(backend_probe.status_code or 0) != 0:
        return False
    if str(backend_probe.error or "").strip() == "TimeoutError":
        return True
    detail = str(backend_probe.detail or "").strip().lower()
    return any(
        fragment in detail
        for fragment in (
            "timed out",
            "timeout",
            "connection refused",
            "connection reset",
        )
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
    backend_data_ownership = inspect_backend_data_ownership(
        executor=executor,
        service_name=settings.backend_service,
    )
    backend_readiness_contract = fetch_backend_query_surface_readiness_contract(
        executor=executor,
        service_name=settings.backend_service,
    )
    backend_vector_warmup_contract = fetch_backend_vector_warmup_contract(
        executor=executor,
        service_name=settings.backend_service,
    )
    foreign_consumers: tuple[ForeignBackendConsumer, ...] = ()
    foreign_consumer_inspection_error: str | None = None
    try:
        foreign_consumers = inspect_foreign_backend_consumers(
            executor=executor,
            settings=settings,
        )
    except Exception as exc:  # pragma: no cover - bounded live fallback
        foreign_consumer_inspection_error = f"{type(exc).__name__}: {exc}"
    plan = plan_remote_chonkydb_repair(
        public_probe=public_before,
        backend_service=backend_service_before,
        backend_probe=backend_before,
        foreign_consumers=foreign_consumers,
        backend_data_ownership=backend_data_ownership,
        backend_readiness_contract=backend_readiness_contract,
        backend_vector_warmup_contract=backend_vector_warmup_contract,
    )
    action_taken = "none"
    public_after = public_before
    backend_service_after = backend_service_before
    backend_after = backend_before

    restart_actions = {
        "restart_backend_service",
        "repair_backend_startup_contract_and_restart_backend_service",
        "repair_backend_data_ownership_and_restart_backend_service",
        "repair_backend_startup_contract_and_data_ownership_then_restart_backend_service",
    }
    if plan.action in restart_actions:
        if restart_if_needed:
            ensure_backend_query_surface_readiness_contract(
                executor=executor,
                service_name=settings.backend_service,
            )
            ensure_backend_payload_read_warmup_contract(
                executor=executor,
                service_name=settings.backend_service,
            )
            ensure_backend_vector_warmup_timeout_contract(
                executor=executor,
                service_name=settings.backend_service,
            )
            ensure_backend_payload_sync_bulk_api_ready_contract(
                executor=executor,
                service_name=settings.backend_service,
            )
            if backend_data_ownership is not None and backend_data_ownership.has_drift:
                repair_backend_data_ownership(
                    executor=executor,
                    service_name=settings.backend_service,
                )
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
    elif plan.reason == "backend_foreign_consumer_contention":
        diagnosis = "backend_foreign_consumer_contention"
    elif action_taken == "restart_required_but_skipped":
        if plan.reason == "backend_active_but_unresponsive":
            diagnosis = "backend_active_but_unresponsive_restart_required"
        else:
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
        foreign_consumers=foreign_consumers,
        backend_data_ownership=backend_data_ownership,
        foreign_consumer_inspection_error=foreign_consumer_inspection_error,
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
    if not _should_probe_query_surface_after_instance_probe(instance_probe):
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


def _fetch_backend_service_env_map(
    *,
    executor: RemoteChonkyDBExecutor,
    service_name: str,
) -> dict[str, str]:
    """Return the live `systemd` environment mapping for one backend service."""

    env_output = executor.run_sudo_ssh(
        "systemctl show -p Environment --no-pager " + shlex.quote(service_name)
    ).stdout
    return _parse_systemd_environment_output(env_output)


def probe_backend_local_chonkydb(
    *,
    executor: RemoteChonkyDBExecutor,
    settings: RemoteChonkyDBOpsSettings,
    timeout_s: float,
) -> ChonkyDBHttpProbeResult:
    """Probe the backend loopback instance on the remote host."""

    env_map = _fetch_backend_service_env_map(
        executor=executor,
        service_name=settings.backend_service,
    )
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
    if not _should_probe_query_surface_after_instance_probe(instance_probe):
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


def inspect_backend_data_ownership(
    *,
    executor: RemoteChonkyDBExecutor,
    service_name: str,
) -> BackendDataOwnershipState | None:
    """Inspect the backend data dir for owner/group drift versus the service account."""

    completed = executor.run_sudo_ssh(
        "systemctl show -p User -p Group -p Environment --no-pager " + shlex.quote(service_name)
    )
    values = _parse_key_value_lines(completed.stdout)
    env_map = _parse_systemd_environment_output(str(values.get("Environment", "")))
    data_dir = str(
        env_map.get("CHONKDB_DATA_DIR")
        or env_map.get("CHONKYDB_DATA_PATH")
        or env_map.get("CCODEX_MEMORY_DATA_DIR")
        or ""
    ).strip()
    expected_user = str(values.get("User", "")).strip() or "root"
    expected_group = str(values.get("Group", "")).strip() or expected_user
    if not data_dir:
        return BackendDataOwnershipState(
            data_dir="",
            expected_user=expected_user,
            expected_group=expected_group,
            mismatched_entry_count=0,
            error="missing_backend_data_dir",
        )
    payload = {
        "data_dir": data_dir,
        "expected_user": expected_user,
        "expected_group": expected_group,
        "sample_limit": 12,
    }
    probe_completed = executor.run_ssh(
        "python3 -c " + shlex.quote(_REMOTE_DATA_OWNERSHIP_PROBE_CODE),
        input_text=json.dumps(payload, ensure_ascii=False),
    )
    parsed = json.loads(probe_completed.stdout)
    sample_entries = parsed.get("sample_entries")
    return BackendDataOwnershipState(
        data_dir=str(parsed.get("data_dir") or data_dir).strip(),
        expected_user=str(parsed.get("expected_user") or expected_user).strip() or expected_user,
        expected_group=str(parsed.get("expected_group") or expected_group).strip() or expected_group,
        mismatched_entry_count=int(parsed.get("mismatched_entry_count") or 0),
        sample_entries=tuple(
            str(entry).strip()
            for entry in (sample_entries if isinstance(sample_entries, list) else [])
            if str(entry).strip()
        ),
        truncated=bool(parsed.get("truncated")),
        error=str(parsed.get("error") or "").strip() or None,
    )


def repair_backend_data_ownership(
    *,
    executor: RemoteChonkyDBExecutor,
    service_name: str,
) -> BackendDataOwnershipRepairStatus:
    """Repair data-dir owner/group drift to the backend service account."""

    current = inspect_backend_data_ownership(
        executor=executor,
        service_name=service_name,
    )
    if current is None:
        return BackendDataOwnershipRepairStatus(
            data_dir="",
            expected_user="",
            expected_group="",
            changed_entry_count=0,
            error="missing_backend_data_ownership_state",
        )
    if not current.has_drift or current.error:
        return BackendDataOwnershipRepairStatus(
            data_dir=current.data_dir,
            expected_user=current.expected_user,
            expected_group=current.expected_group,
            changed_entry_count=0,
            error=current.error,
        )
    payload = {
        "data_dir": current.data_dir,
        "expected_user": current.expected_user,
        "expected_group": current.expected_group,
    }
    completed = executor.run_sudo_ssh(
        _python_backend_data_ownership_repair_script(payload=payload),
    )
    parsed = _parse_json_payload(completed.stdout)
    return BackendDataOwnershipRepairStatus(
        data_dir=str(parsed.get("data_dir") or current.data_dir).strip(),
        expected_user=str(parsed.get("expected_user") or current.expected_user).strip()
        or current.expected_user,
        expected_group=str(parsed.get("expected_group") or current.expected_group).strip()
        or current.expected_group,
        changed_entry_count=int(_parse_optional_int(parsed.get("changed_entry_count")) or 0),
        error=str(parsed.get("error") or "").strip() or None,
    )


def inspect_foreign_backend_consumers(
    *,
    executor: RemoteChonkyDBExecutor,
    settings: RemoteChonkyDBOpsSettings,
) -> tuple[ForeignBackendConsumer, ...]:
    """Return active or configured non-Twinr units still pointed at the backend."""

    payload = {
        "backend_service": settings.backend_service,
        "backend_local_base_url": settings.backend_local_base_url,
    }
    completed = executor.run_ssh(
        "python3 -c " + shlex.quote(_REMOTE_FOREIGN_CONSUMER_CODE),
        input_text=json.dumps(payload, ensure_ascii=False),
    )
    parsed = json.loads(completed.stdout)
    consumers_payload = parsed.get("consumers")
    if not isinstance(consumers_payload, list):
        return ()
    consumers: list[ForeignBackendConsumer] = []
    for raw_entry in consumers_payload:
        if not isinstance(raw_entry, Mapping):
            continue
        consumers.append(
            ForeignBackendConsumer(
                unit_name=str(raw_entry.get("unit_name") or "").strip(),
                active_state=str(raw_entry.get("active_state") or "").strip(),
                sub_state=str(raw_entry.get("sub_state") or "").strip(),
                configured_base_url=str(raw_entry.get("configured_base_url") or "").strip(),
                fragment_path=str(raw_entry.get("fragment_path") or "").strip(),
                coupled_to_backend_service=bool(raw_entry.get("coupled_to_backend_service")),
            )
        )
    return tuple(
        sorted(
            consumers,
            key=lambda consumer: (
                not consumer.active,
                consumer.unit_name,
            ),
        )
    )


def ensure_backend_query_surface_readiness_contract(
    *,
    executor: RemoteChonkyDBExecutor,
    service_name: str,
) -> BackendQuerySurfaceReadinessRepairStatus:
    """Repair the proven unsafe fulltext-startup mismatch on the remote host.

    Twinr observed repeated outages where systemd reported the dedicated
    ChonkyDB unit as `active/running`, yet both the local loopback probe and
    the public URL timed out or returned `503 Upstream unavailable or
    restarting`. Remote evidence showed the service still running fulltext
    rebuild/open work under `CHONKY_FT_REBUILD_ON_OPEN=1` while the query gate
    remained disabled. This helper writes one persistent drop-in only when that
    exact mismatch is present, then relies on the guarded restart path to make
    the corrected env contract take effect.
    """

    env_map = _fetch_backend_service_env_map(
        executor=executor,
        service_name=service_name,
    )
    contract = _backend_query_surface_readiness_contract_from_env(env_map)
    dropin_path = f"/etc/systemd/system/{service_name}.d/40-twinr-query-surface-readiness.conf"
    if not contract.unsafe:
        return BackendQuerySurfaceReadinessRepairStatus(
            dropin_path=dropin_path,
            changed=False,
            contract=contract,
        )
    readiness_timeout_s = (
        float(contract.warmup_wait_ready_timeout_s)
        if contract.warmup_wait_ready_timeout_s is not None
        and float(contract.warmup_wait_ready_timeout_s) > 0.0
        else 180.0
    )
    dropin_content = _backend_query_surface_readiness_dropin_content(
        contract=contract,
        readiness_timeout_s=readiness_timeout_s,
    )
    completed = executor.run_sudo_ssh(
        _python_service_dropin_sync_script(
            dropin_path=dropin_path,
            dropin_content=dropin_content,
        )
    )
    payload = _parse_json_payload(completed.stdout)
    changed = bool(payload.get("changed"))
    return BackendQuerySurfaceReadinessRepairStatus(
        dropin_path=str(payload.get("path") or dropin_path).strip() or dropin_path,
        changed=changed,
        contract=contract,
    )


def fetch_backend_query_surface_readiness_contract(
    *,
    executor: RemoteChonkyDBExecutor,
    service_name: str,
) -> BackendQuerySurfaceReadinessContract:
    """Return the live startup-readiness contract from the backend env."""

    return _backend_query_surface_readiness_contract_from_env(
        _fetch_backend_service_env_map(
            executor=executor,
            service_name=service_name,
        )
    )


def ensure_backend_payload_read_warmup_contract(
    *,
    executor: RemoteChonkyDBExecutor,
    service_name: str,
) -> BackendPayloadReadWarmupRepairStatus:
    """Restore the required payload-read startup contract for Twinr."""

    contract = fetch_backend_query_surface_readiness_contract(
        executor=executor,
        service_name=service_name,
    )
    dropin_path = f"/etc/systemd/system/{service_name}.d/45-twinr-disable-payload-read-gate.conf"
    payload_timeout_s = (
        float(contract.warmup_payload_read_timeout_s)
        if contract.warmup_payload_read_timeout_s is not None
        and float(contract.warmup_payload_read_timeout_s) > 0.0
        else 30.0
    )
    if contract.warmup_payload_read_path and payload_timeout_s > 0.0:
        return BackendPayloadReadWarmupRepairStatus(
            dropin_path=dropin_path,
            changed=False,
            contract=contract,
        )
    completed = executor.run_sudo_ssh(
        _python_service_dropin_sync_script(
            dropin_path=dropin_path,
            dropin_content=_backend_payload_read_warmup_dropin_content(
                payload_timeout_s=payload_timeout_s,
            ),
        )
    )
    payload = _parse_json_payload(completed.stdout)
    return BackendPayloadReadWarmupRepairStatus(
        dropin_path=str(payload.get("path") or dropin_path).strip() or dropin_path,
        changed=bool(payload.get("changed")),
        contract=contract,
    )


def ensure_backend_payload_sync_bulk_api_ready_contract(
    *,
    executor: RemoteChonkyDBExecutor,
    service_name: str,
) -> BackendPayloadSyncBulkApiReadyRepairStatus:
    """Disable the over-strict sync-bulk API ready gate for Twinr dedicated writes."""

    contract = _backend_payload_sync_bulk_api_ready_contract_from_env(
        _fetch_backend_service_env_map(
            executor=executor,
            service_name=service_name,
        )
    )
    dropin_path = f"/etc/systemd/system/{service_name}.d/46-twinr-disable-sync-bulk-api-ready-gate.conf"
    if not contract.unsafe:
        return BackendPayloadSyncBulkApiReadyRepairStatus(
            dropin_path=dropin_path,
            changed=False,
            contract=contract,
        )
    completed = executor.run_sudo_ssh(
        _python_service_dropin_sync_script(
            dropin_path=dropin_path,
            dropin_content=_backend_payload_sync_bulk_api_ready_dropin_content(),
        )
    )
    payload = _parse_json_payload(completed.stdout)
    return BackendPayloadSyncBulkApiReadyRepairStatus(
        dropin_path=str(payload.get("path") or dropin_path).strip() or dropin_path,
        changed=bool(payload.get("changed")),
        contract=contract,
    )


def fetch_backend_vector_warmup_contract(
    *,
    executor: RemoteChonkyDBExecutor,
    service_name: str,
) -> BackendVectorWarmupContract:
    """Return the live vector-warmup budget contract from the backend env."""

    return _backend_vector_warmup_contract_from_env(
        _fetch_backend_service_env_map(
            executor=executor,
            service_name=service_name,
        )
    )


def ensure_backend_vector_warmup_timeout_contract(
    *,
    executor: RemoteChonkyDBExecutor,
    service_name: str,
) -> BackendVectorWarmupRepairStatus:
    """Align vector warmup timeout with the full-scope ready-wait budget."""

    contract = fetch_backend_vector_warmup_contract(
        executor=executor,
        service_name=service_name,
    )
    dropin_path = f"/etc/systemd/system/{service_name}.d/52-twinr-vector-warmup-budget.conf"
    target_timeout_s = contract.target_vector_open_timeout_s
    if not contract.unsafe or target_timeout_s is None:
        return BackendVectorWarmupRepairStatus(
            dropin_path=dropin_path,
            changed=False,
            contract=contract,
            target_vector_open_timeout_s=target_timeout_s,
        )
    completed = executor.run_sudo_ssh(
        _python_service_dropin_sync_script(
            dropin_path=dropin_path,
            dropin_content=_backend_vector_warmup_timeout_dropin_content(
                vector_open_timeout_s=target_timeout_s,
            ),
        )
    )
    payload = _parse_json_payload(completed.stdout)
    return BackendVectorWarmupRepairStatus(
        dropin_path=str(payload.get("path") or dropin_path).strip() or dropin_path,
        changed=bool(payload.get("changed")),
        contract=contract,
        target_vector_open_timeout_s=target_timeout_s,
    )


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


def _should_probe_query_surface_after_instance_probe(instance_probe: ChonkyDBHttpProbeResult) -> bool:
    """Return whether the live query canary should run after `/instance`.

    Live 2026-04-11 evidence showed the dedicated backend continuing to return
    `503 Service warmup in progress` on `/v1/external/instance` even while the
    real `catalog/current` query surface was already resolving current heads.
    Keep probing the authoritative query path for explicit warmup responses so
    repair/recovery flows do not keep treating a usable backend as still down.
    """

    if instance_probe.ok:
        return True
    return is_explicit_service_warmup_detail(instance_probe.detail)


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

    if query_probe.ok or _query_probe_empty_but_healthy(query_probe):
        if instance_probe.ready:
            return instance_probe
        detail = str(instance_probe.detail or "").strip()
        if detail:
            detail = f"query_surface_ready_despite_instance_flag_false: {detail}"
        else:
            detail = "query_surface_ready_despite_instance_flag_false"
        return ChonkyDBHttpProbeResult(
            label=instance_probe.label,
            ok=True,
            status_code=200,
            ready=True,
            detail=detail,
            url=query_probe.url or instance_probe.url,
            payload=query_probe.payload or instance_probe.payload,
        )
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


def _query_probe_empty_but_healthy(query_probe: ChonkyDBHttpProbeResult) -> bool:
    """Return whether one query probe proved an empty but healthy current scope.

    A fresh namespace or a snapshot kind without a current head can
    legitimately return `404 document_not_found` even though the public query
    surface and backend are healthy. Treat that exact contract as green so
    operator repair/stabilization tools do not trigger blind restarts on empty
    scopes.
    """

    if int(query_probe.status_code or 0) != 404:
        return False
    payload = dict(query_probe.payload or {})
    detail_candidates = (
        str(query_probe.detail or "").strip().lower(),
        str(payload.get("detail") or "").strip().lower(),
        str(payload.get("error") or "").strip().lower(),
        str(payload.get("title") or "").strip().lower(),
        str(payload.get("error_type") or "").strip().lower(),
    )
    return "document_not_found" in detail_candidates


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


def _backend_query_surface_readiness_contract_from_env(
    env_map: Mapping[str, str],
) -> BackendQuerySurfaceReadinessContract:
    """Summarize the startup env flags that govern query-surface readiness."""

    return BackendQuerySurfaceReadinessContract(
        fulltext_rebuild_on_open=_parse_env_flag(
            env_map.get("CHONKY_FT_REBUILD_ON_OPEN"),
            default=False,
        ),
        warmup_fulltext_gate=_parse_env_flag(
            env_map.get("CHONKY_API_WARMUP_FULLTEXT_GATE"),
            default=False,
        ),
        warmup_wait_for_ready=_parse_env_flag(
            env_map.get("CHONKY_API_WARMUP_WAIT_FOR_READY"),
            default=False,
        ),
        ready_default_scope=str(env_map.get("CHONKY_API_READY_DEFAULT_SCOPE") or "").strip().lower()
        or "full",
        serving_contract_scope=str(env_map.get("CHONKY_API_SERVING_CONTRACT_SCOPE") or "").strip().lower()
        or str(env_map.get("CHONKY_API_READY_DEFAULT_SCOPE") or "").strip().lower()
        or "full",
        warmup_wait_ready_timeout_s=_parse_optional_float(
            env_map.get("CHONKY_API_WARMUP_WAIT_READY_TIMEOUT_S"),
        ),
        warmup_payload_read_path=_parse_env_flag(
            env_map.get("CHONKY_API_WARMUP_PAYLOAD_READ_PATH"),
            default=False,
        ),
        warmup_payload_read_timeout_s=_parse_optional_float(
            env_map.get("CHONKY_API_WARMUP_PAYLOAD_READ_TIMEOUT_S"),
        ),
    )


def _backend_payload_sync_bulk_api_ready_contract_from_env(
    env_map: Mapping[str, str],
) -> BackendPayloadSyncBulkApiReadyContract:
    """Summarize the startup env flags that govern sync-bulk write readiness."""

    return BackendPayloadSyncBulkApiReadyContract(
        ready_default_scope=str(env_map.get("CHONKY_API_READY_DEFAULT_SCOPE") or "").strip().lower() or "full",
        payload_sync_bulk_require_api_ready=_parse_env_flag(
            env_map.get("CHONKY_API_PAYLOADS_SYNC_BULK_REQUIRE_API_READY"),
            default=True,
        ),
    )


def _backend_vector_warmup_contract_from_env(
    env_map: Mapping[str, str],
) -> BackendVectorWarmupContract:
    """Summarize the vector warmup budget that gates full-scope readiness."""

    return BackendVectorWarmupContract(
        ready_default_scope=str(env_map.get("CHONKY_API_READY_DEFAULT_SCOPE") or "").strip().lower() or "full",
        serving_contract_scope=str(env_map.get("CHONKY_API_SERVING_CONTRACT_SCOPE") or "").strip().lower() or "full",
        warmup_wait_for_ready=_parse_env_flag(
            env_map.get("CHONKY_API_WARMUP_WAIT_FOR_READY"),
            default=False,
        ),
        warmup_wait_ready_timeout_s=_parse_optional_float(
            env_map.get("CHONKY_API_WARMUP_WAIT_READY_TIMEOUT_S"),
        ),
        warmup_vector_open_timeout_s=_parse_optional_float(
            env_map.get("CHONKY_API_WARMUP_VECTOR_OPEN_TIMEOUT_S"),
        ),
    )


def _backend_query_surface_readiness_dropin_content(
    *,
    contract: BackendQuerySurfaceReadinessContract,
    readiness_timeout_s: float,
) -> str:
    """Render the hardened startup contract for the dedicated backend service."""

    gate_value = "1" if contract.requires_full_scope_warmup else "0"
    return (
        "[Service]\n"
        "# Keep the query surface on the configured serving contract.\n"
        f"Environment=CHONKY_API_WARMUP_FULLTEXT_GATE={gate_value}\n"
        "Environment=CHONKY_API_WARMUP_WAIT_FOR_READY=1\n"
        f"Environment=CHONKY_API_WARMUP_WAIT_READY_TIMEOUT_S={int(max(1.0, readiness_timeout_s))}\n"
    )


def _backend_payload_read_warmup_dropin_content(*, payload_timeout_s: float) -> str:
    """Render the sanctioned Twinr payload-read startup contract."""

    return (
        "[Service]\n"
        "# Keep the Twinr dedicated backend on the sanctioned payload-read startup lane.\n"
        "Environment=CHONKY_API_WARMUP_PAYLOAD_READ_PATH=1\n"
        f"Environment=CHONKY_API_WARMUP_PAYLOAD_READ_TIMEOUT_S={int(max(1.0, payload_timeout_s))}\n"
    )


def _backend_payload_sync_bulk_api_ready_dropin_content() -> str:
    """Render the Twinr-specific override that disables sync-bulk API ready gating."""

    return (
        "[Service]\n"
        "# Twinr writes must not wait for unrelated full-scope startup warmup.\n"
        "Environment=CHONKY_API_PAYLOADS_SYNC_BULK_REQUIRE_API_READY=0\n"
    )


def _backend_vector_warmup_timeout_dropin_content(*, vector_open_timeout_s: float) -> str:
    """Render the Twinr-specific override that widens vector warmup budget."""

    return (
        "[Service]\n"
        "# Full-scope warmup must not abort vector open before ready-wait expires.\n"
        f"Environment=CHONKY_API_WARMUP_VECTOR_OPEN_TIMEOUT_S={int(max(1.0, vector_open_timeout_s))}\n"
    )


def _python_service_dropin_sync_script(
    *,
    dropin_path: str,
    dropin_content: str,
) -> str:
    """Return one remote Python script that writes the readiness drop-in."""

    return (
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        "import json\n"
        f"path = Path({dropin_path!r})\n"
        f"content = {dropin_content!r}\n"
        "path.parent.mkdir(parents=True, exist_ok=True)\n"
        "existing = path.read_text(encoding='utf-8') if path.exists() else ''\n"
        "changed = existing != content\n"
        "if changed:\n"
        "    path.write_text(content, encoding='utf-8')\n"
        "print(json.dumps({'path': str(path), 'changed': changed}, ensure_ascii=False))\n"
        "PY\n"
        "systemctl daemon-reload"
    )


def _python_backend_data_ownership_repair_script(*, payload: Mapping[str, object]) -> str:
    """Return one remote Python script that repairs data-dir owner/group drift."""

    return (
        "python3 - <<'PY'\n"
        "from __future__ import annotations\n"
        "import grp\n"
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        "import pwd\n"
        f"payload = json.loads({json.dumps(dict(payload), ensure_ascii=False)!r})\n"
        "data_dir = str(payload.get('data_dir') or '').strip()\n"
        "expected_user = str(payload.get('expected_user') or '').strip() or 'root'\n"
        "expected_group = str(payload.get('expected_group') or '').strip() or expected_user\n"
        "result = {\n"
        "    'data_dir': data_dir,\n"
        "    'expected_user': expected_user,\n"
        "    'expected_group': expected_group,\n"
        "    'changed_entry_count': 0,\n"
        "    'error': '',\n"
        "}\n"
        "if not data_dir:\n"
        "    result['error'] = 'missing_data_dir'\n"
        "    print(json.dumps(result, ensure_ascii=False))\n"
        "    raise SystemExit(0)\n"
        "path = Path(data_dir)\n"
        "if not path.exists():\n"
        "    result['error'] = 'data_dir_not_found'\n"
        "    print(json.dumps(result, ensure_ascii=False))\n"
        "    raise SystemExit(0)\n"
        "try:\n"
        "    expected_uid = pwd.getpwnam(expected_user).pw_uid\n"
        "    expected_gid = grp.getgrnam(expected_group).gr_gid\n"
        "except KeyError as exc:\n"
        "    result['error'] = f'missing_identity:{exc}'\n"
        "    print(json.dumps(result, ensure_ascii=False))\n"
        "    raise SystemExit(0)\n"
        "def _repair_one(raw_path: str) -> None:\n"
        "    stat_result = os.lstat(raw_path)\n"
        "    if int(stat_result.st_uid) == int(expected_uid) and int(stat_result.st_gid) == int(expected_gid):\n"
        "        return\n"
        "    os.chown(raw_path, int(expected_uid), int(expected_gid), follow_symlinks=False)\n"
        "    result['changed_entry_count'] += 1\n"
        "try:\n"
        "    _repair_one(data_dir)\n"
        "    stack = [data_dir]\n"
        "    while stack:\n"
        "        current = stack.pop()\n"
        "        with os.scandir(current) as entries:\n"
        "            for entry in entries:\n"
        "                _repair_one(entry.path)\n"
        "                if entry.is_dir(follow_symlinks=False):\n"
        "                    stack.append(entry.path)\n"
        "except OSError as exc:\n"
        "    result['error'] = f'{type(exc).__name__}:{exc}'\n"
        "print(json.dumps(result, ensure_ascii=False))\n"
        "PY"
    )


def _parse_env_flag(value: object, *, default: bool) -> bool:
    """Parse one boolean-ish env value."""

    text = str(value or "").strip().lower()
    if not text:
        return bool(default)
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _parse_optional_float(value: object) -> float | None:
    """Parse one optional numeric env value into `float`."""

    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


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


_REMOTE_DATA_OWNERSHIP_PROBE_CODE = """
from __future__ import annotations

import grp
import json
import os
from pathlib import Path
import pwd
import sys

payload = json.load(sys.stdin)
data_dir = str(payload.get("data_dir") or "").strip()
expected_user = str(payload.get("expected_user") or "").strip() or "root"
expected_group = str(payload.get("expected_group") or "").strip() or expected_user
sample_limit = max(1, int(payload.get("sample_limit") or 12))
result = {
    "data_dir": data_dir,
    "expected_user": expected_user,
    "expected_group": expected_group,
    "mismatched_entry_count": 0,
    "sample_entries": [],
    "truncated": False,
    "error": "",
}
if not data_dir:
    result["error"] = "missing_data_dir"
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0)
path = Path(data_dir)
if not path.exists():
    result["error"] = "data_dir_not_found"
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0)
try:
    expected_uid = pwd.getpwnam(expected_user).pw_uid
except KeyError:
    result["error"] = f"user_not_found:{expected_user}"
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0)
try:
    expected_gid = grp.getgrnam(expected_group).gr_gid
except KeyError:
    result["error"] = f"group_not_found:{expected_group}"
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0)


def _owner_name(uid: int) -> str:
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def _group_name(gid: int) -> str:
    try:
        return grp.getgrgid(gid).gr_name
    except KeyError:
        return str(gid)


def _record_if_mismatch(raw_path: str, stat_result: os.stat_result) -> bool:
    owner_ok = int(stat_result.st_uid) == int(expected_uid) and int(stat_result.st_gid) == int(expected_gid)
    if owner_ok:
        return False
    result["mismatched_entry_count"] += 1
    if len(result["sample_entries"]) < sample_limit:
        result["sample_entries"].append(
            f"{_owner_name(int(stat_result.st_uid))}:{_group_name(int(stat_result.st_gid))} {raw_path}"
        )
    if len(result["sample_entries"]) >= sample_limit:
        result["truncated"] = True
        return True
    return False


try:
    if _record_if_mismatch(data_dir, os.lstat(data_dir)):
        print(json.dumps(result, ensure_ascii=False))
        raise SystemExit(0)
    stack = [data_dir]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                stat_result = entry.stat(follow_symlinks=False)
                if _record_if_mismatch(entry.path, stat_result):
                    print(json.dumps(result, ensure_ascii=False))
                    raise SystemExit(0)
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
except OSError as exc:
    result["error"] = f"{type(exc).__name__}:{exc}"

print(json.dumps(result, ensure_ascii=False))
"""


_REMOTE_FOREIGN_CONSUMER_CODE = """
from __future__ import annotations

import json
import shlex
import subprocess
import sys


def _parse_key_value_lines(text: str) -> dict[str, str]:
    mapping = {}
    for raw_line in str(text or "").splitlines():
        key, separator, value = raw_line.partition("=")
        if separator:
            mapping[str(key).strip()] = str(value).strip()
    return mapping


def _parse_environment(raw_value: str) -> dict[str, str]:
    text = str(raw_value or "").strip()
    if text.startswith("Environment="):
        text = text[len("Environment="):]
    if not text:
        return {}
    try:
        tokens = shlex.split(text)
    except ValueError:
        tokens = [part for part in text.split(" ") if part]
    mapping = {}
    for token in tokens:
        key, separator, value = str(token).partition("=")
        if separator:
            mapping[str(key)] = str(value)
    return mapping


payload = json.load(sys.stdin)
backend_service = str(payload.get("backend_service") or "").strip()
backend_local_base_url = str(payload.get("backend_local_base_url") or "").strip().rstrip("/")
list_completed = subprocess.run(
    ["systemctl", "list-unit-files", "--type=service", "--no-legend", "--plain"],
    check=False,
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
)
if list_completed.returncode != 0:
    print(json.dumps({"consumers": []}, ensure_ascii=False))
    raise SystemExit(0)

consumers = []
seen_units = set()
for raw_line in list_completed.stdout.splitlines():
    unit_name = str(raw_line.split(None, 1)[0] if raw_line.split(None, 1) else "").strip()
    if not unit_name or not unit_name.endswith(".service") or unit_name == backend_service:
        continue
    if unit_name in seen_units:
        continue
    seen_units.add(unit_name)
    show_completed = subprocess.run(
        [
            "systemctl",
            "show",
            "--no-pager",
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "FragmentPath",
            "-p",
            "Environment",
            "-p",
            "After",
            "-p",
            "Wants",
            "-p",
            "Requires",
            "-p",
            "PartOf",
            unit_name,
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if show_completed.returncode != 0:
        continue
    values = _parse_key_value_lines(show_completed.stdout)
    env_map = _parse_environment(values.get("Environment", ""))
    configured_base_url = ""
    for key, value in env_map.items():
        if not str(key).endswith("_BASE_URL"):
            continue
        normalized_value = str(value).strip().rstrip("/")
        if normalized_value == backend_local_base_url:
            configured_base_url = str(value).strip()
            break
    relation_fields = " ".join(
        str(values.get(field, "")).strip()
        for field in ("After", "Wants", "Requires", "PartOf")
    )
    coupled_to_backend_service = backend_service in relation_fields.split()
    if not configured_base_url and not coupled_to_backend_service:
        continue
    consumers.append(
        {
            "unit_name": unit_name,
            "active_state": str(values.get("ActiveState", "")).strip(),
            "sub_state": str(values.get("SubState", "")).strip(),
            "configured_base_url": configured_base_url,
            "fragment_path": str(values.get("FragmentPath", "")).strip(),
            "coupled_to_backend_service": coupled_to_backend_service,
        }
    )

consumers.sort(key=lambda item: (item.get("unit_name") or ""))
print(json.dumps({"consumers": consumers}, ensure_ascii=False))
"""
