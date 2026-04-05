"""Diagnose and recover retention-canary failures caused by backend host contention.

Purpose
-------
The deploy-time retention canary is supposed to prove that Twinr's remote
memory can still write, retain, and read back from a fresh isolated namespace.
When that proof fails because the dedicated backend host is saturated by
shared-host workloads, operators need one bounded recovery path that proves the
host state first, stabilizes only the proven shared-host contention layer, and,
when the backend service still remains unhealthy afterwards, applies one guarded
backend repair before the canary is retried once.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import subprocess
from typing import Mapping

from twinr.ops.remote_chonkydb_host_stabilizer import (
    _DEFAULT_SYSTEM_CONFLICTING_UNITS,
    _DEFAULT_USER_CONFLICTING_UNITS,
    fetch_remote_service_properties,
    fetch_remote_unit_states,
    stabilize_remote_chonkydb_host,
)
from twinr.ops.remote_chonkydb_repair import (
    RemoteChonkyDBExecutor,
    fetch_backend_service_state,
    load_remote_chonkydb_ops_settings,
    probe_backend_local_chonkydb,
    probe_public_chonkydb,
    repair_remote_chonkydb,
)


_HIGH_BACKEND_MEMORY_CURRENT_BYTES = 2_000_000_000
_HIGH_BACKEND_TASKS_CURRENT = 256
_SYSTEMD_HEALTHY_SUBSTATES = frozenset({"running", "listening", "exited"})
_HOST_CONTENTION_FAILURE_MARKERS = (
    "queue_saturated",
    "payload_sync_bulk_busy",
    "upstream unavailable or restarting",
    "without a stable document id",
    "could not be read back",
)


def _parse_optional_int(value: object) -> int | None:
    """Return one positive integer when the remote property is parseable."""

    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = int(text)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _iter_error_fragments(value: object) -> list[str]:
    """Flatten nested failure payloads into comparable lowercase fragments."""

    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Mapping):
        mapping_fragments: list[str] = []
        for key, nested in value.items():
            key_text = str(key).strip()
            if key_text:
                mapping_fragments.append(key_text)
            mapping_fragments.extend(_iter_error_fragments(nested))
        return mapping_fragments
    if isinstance(value, list):
        sequence_fragments: list[str] = []
        for item in value:
            sequence_fragments.extend(_iter_error_fragments(item))
        return sequence_fragments
    text = str(value).strip()
    return [text] if text else []


def _service_payload_healthy(payload: object) -> bool:
    """Return whether one serialized service-state payload is healthy."""

    if not isinstance(payload, Mapping):
        return False
    load_state = str(payload.get("load_state") or "").strip()
    active_state = str(payload.get("active_state") or "").strip()
    sub_state = str(payload.get("sub_state") or "").strip()
    if load_state and load_state != "loaded":
        return False
    return active_state == "active" and sub_state in _SYSTEMD_HEALTHY_SUBSTATES


def _probe_payload_ready(payload: object) -> bool:
    """Return whether one serialized HTTP probe payload is ready."""

    return isinstance(payload, Mapping) and bool(payload.get("ready"))


def _stabilization_needs_backend_repair(
    *,
    diagnosis: Mapping[str, object] | None,
    stabilization: Mapping[str, object],
) -> bool:
    """Return whether one failed stabilization pass should escalate to repair."""

    if bool(stabilization.get("ok")):
        return False
    if _probe_payload_ready(stabilization.get("public_after")):
        return False
    if diagnosis is None:
        return True
    if not _service_payload_healthy(diagnosis.get("backend_service")):
        return True
    return not _probe_payload_ready(diagnosis.get("backend_probe"))


def _refresh_backend_repair_diagnosis(
    *,
    project_root: Path,
    probe_timeout_s: float,
    ssh_timeout_s: float,
    diagnosis: Mapping[str, object] | None,
    stabilization: Mapping[str, object],
) -> tuple[Mapping[str, object] | None, dict[str, object] | None]:
    """Return the repair-gate diagnosis after one failed stabilization pass.

    The pre-stabilization diagnosis can age out while the host stabilizer is
    quiescing foreign work. When the public surface still remains unhealthy
    afterwards, re-probe the backend before deciding that guarded repair is
    unnecessary; otherwise a backend that turned unhealthy during the
    stabilization window can be missed exactly as the live retention canary
    retries.
    """

    if bool(stabilization.get("ok")):
        return diagnosis, None
    if _probe_payload_ready(stabilization.get("public_after")):
        return diagnosis, None
    try:
        refreshed = diagnose_retention_canary_host_contention(
            project_root=project_root,
            probe_timeout_s=probe_timeout_s,
            ssh_timeout_s=ssh_timeout_s,
        )
    except Exception as exc:  # pragma: no cover - defensive observability.
        error_payload = {
            "available": False,
            "contention_detected": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        return None, error_payload
    if isinstance(refreshed, Mapping) and refreshed:
        return refreshed, dict(refreshed)
    empty_payload = {
        "available": False,
        "contention_detected": False,
        "error": "empty_post_stabilization_diagnosis",
    }
    return None, empty_payload


def diagnose_retention_canary_host_contention(
    *,
    project_root: str | Path,
    probe_timeout_s: float = 10.0,
    ssh_timeout_s: float = 60.0,
) -> dict[str, object]:
    """Return one structured diagnosis for the dedicated ChonkyDB host.

    The diagnosis is intentionally read-only. It proves whether the public
    query surface, backend loopback query surface, backend service resource
    counters, and the curated shared-host conflict-unit sets currently support
    a contention explanation for a failed retention canary.
    """

    resolved_root = Path(project_root).resolve()
    env_file = resolved_root / ".env"
    ops_env_file = resolved_root / ".env.chonkydb"
    if not env_file.is_file():
        return {
            "available": False,
            "contention_detected": False,
            "error": f"missing_env_file:{env_file}",
        }
    if not ops_env_file.is_file():
        return {
            "available": False,
            "contention_detected": False,
            "error": f"missing_ops_env_file:{ops_env_file}",
        }
    settings = load_remote_chonkydb_ops_settings(
        env_file=env_file,
        ops_env_file=ops_env_file,
    )
    executor = RemoteChonkyDBExecutor(
        settings=settings.ssh,
        subprocess_runner=subprocess.run,
        timeout_s=ssh_timeout_s,
    )
    public_probe = probe_public_chonkydb(settings=settings, timeout_s=probe_timeout_s)
    backend_service = fetch_backend_service_state(
        executor=executor,
        service_name=settings.backend_service,
    )
    backend_probe = probe_backend_local_chonkydb(
        executor=executor,
        settings=settings,
        timeout_s=probe_timeout_s,
    )
    backend_properties = fetch_remote_service_properties(
        executor=executor,
        service_name=settings.backend_service,
        properties=(
            "MemoryCurrent",
            "MemoryPeak",
            "MemorySwapCurrent",
            "TasksCurrent",
            "NRestarts",
            "CPUWeight",
            "IOWeight",
            "ActiveEnterTimestamp",
        ),
    )
    system_units = fetch_remote_unit_states(
        executor=executor,
        units=_DEFAULT_SYSTEM_CONFLICTING_UNITS,
        scope="system",
        user_unit_owner=settings.ssh.user,
    )
    user_units = fetch_remote_unit_states(
        executor=executor,
        units=_DEFAULT_USER_CONFLICTING_UNITS,
        scope="user",
        user_unit_owner=settings.ssh.user,
    )
    active_system_conflicts = tuple(
        unit.unit
        for unit in system_units
        if unit.active_state in {"active", "activating", "reloading"}
    )
    active_user_conflicts = tuple(
        unit.unit
        for unit in user_units
        if unit.active_state in {"active", "activating", "reloading"}
    )
    memory_current = _parse_optional_int(backend_properties.get("MemoryCurrent"))
    tasks_current = _parse_optional_int(backend_properties.get("TasksCurrent"))
    contention_signals: list[str] = []
    if not public_probe.ready:
        contention_signals.append("public_query_unhealthy")
    if not backend_probe.ready:
        contention_signals.append("backend_query_unhealthy")
    if memory_current is not None and memory_current >= _HIGH_BACKEND_MEMORY_CURRENT_BYTES:
        contention_signals.append("backend_memory_current_high")
    if tasks_current is not None and tasks_current >= _HIGH_BACKEND_TASKS_CURRENT:
        contention_signals.append("backend_tasks_current_high")
    if active_system_conflicts:
        contention_signals.append("active_system_conflicts")
    if active_user_conflicts:
        contention_signals.append("active_user_conflicts")
    return {
        "available": True,
        "backend_service_name": settings.backend_service,
        "public_probe": asdict(public_probe),
        "backend_service": asdict(backend_service),
        "backend_probe": asdict(backend_probe),
        "backend_properties": dict(backend_properties),
        "active_system_conflicts": list(active_system_conflicts),
        "active_user_conflicts": list(active_user_conflicts),
        "contention_signals": contention_signals,
        "contention_detected": bool(contention_signals),
    }


def retention_canary_host_recovery_eligible(
    *,
    canary_payload: Mapping[str, object] | None,
    diagnosis: Mapping[str, object] | None,
) -> bool:
    """Return whether one failed canary should trigger bounded host recovery."""

    if not isinstance(canary_payload, Mapping) or not isinstance(diagnosis, Mapping):
        return False
    if not bool(diagnosis.get("contention_detected")):
        return False
    failure_stage = str(canary_payload.get("failure_stage") or "").strip().lower()
    if failure_stage != "seed_retention_objects":
        return False
    remote_write_context = canary_payload.get("remote_write_context")
    if not isinstance(remote_write_context, Mapping):
        return False
    if str(remote_write_context.get("operation") or "").strip() != "store_records_bulk":
        return False
    if str(remote_write_context.get("request_path") or "").strip() != "/v1/external/records/bulk":
        return False
    error_fragments = [
        *_iter_error_fragments(canary_payload.get("error_message")),
        *_iter_error_fragments(canary_payload.get("root_cause_message")),
        *_iter_error_fragments(canary_payload.get("exception_chain")),
    ]
    normalized_error = " ".join(fragment.lower() for fragment in error_fragments if fragment).strip()
    return any(marker in normalized_error for marker in _HOST_CONTENTION_FAILURE_MARKERS)


def stabilize_retention_canary_host(
    *,
    project_root: str | Path,
    probe_timeout_s: float = 10.0,
    ssh_timeout_s: float = 60.0,
    settle_s: float = 8.0,
    repair_wait_ready_s: float = 120.0,
    repair_poll_interval_s: float = 3.0,
    diagnosis: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Run one bounded host recovery pass for the dedicated backend.

    The first step always applies the host stabilizer so proven shared-host
    pressure is reduced before any backend restart is considered. If the public
    query surface still remains unhealthy afterwards and the evidence points at
    the backend service itself, escalate once into the guarded backend-repair
    workflow.
    """

    resolved_root = Path(project_root).resolve()
    settings = load_remote_chonkydb_ops_settings(
        env_file=resolved_root / ".env",
        ops_env_file=resolved_root / ".env.chonkydb",
    )
    stabilization = stabilize_remote_chonkydb_host(
        settings=settings,
        probe_timeout_s=probe_timeout_s,
        ssh_timeout_s=ssh_timeout_s,
        settle_s=settle_s,
    )
    result = stabilization.to_dict()
    repair_diagnosis, refreshed_diagnosis = _refresh_backend_repair_diagnosis(
        project_root=resolved_root,
        probe_timeout_s=probe_timeout_s,
        ssh_timeout_s=ssh_timeout_s,
        diagnosis=diagnosis,
        stabilization=result,
    )
    if refreshed_diagnosis is not None:
        result["diagnosis_after_stabilization"] = refreshed_diagnosis
    if not _stabilization_needs_backend_repair(
        diagnosis=repair_diagnosis,
        stabilization=result,
    ):
        return result
    repair = repair_remote_chonkydb(
        settings=settings,
        probe_timeout_s=probe_timeout_s,
        ssh_timeout_s=ssh_timeout_s,
        wait_ready_s=repair_wait_ready_s,
        poll_interval_s=repair_poll_interval_s,
        restart_if_needed=True,
    )
    result["backend_repair"] = repair.to_dict()
    result["ok"] = bool(repair.ok)
    result["diagnosis"] = (
        "public_recovered_after_host_stabilization_and_backend_repair"
        if repair.ok
        else str(repair.diagnosis or result.get("diagnosis") or "").strip()
    )
    return result
