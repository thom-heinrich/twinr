from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from shutil import disk_usage, which
import socket
import subprocess

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.runtime_state import RuntimeSnapshot
from twinr.ops.events import TwinrOpsEventStore
from twinr.ops.paths import resolve_ops_paths_for_config


@dataclass(frozen=True, slots=True)
class ServiceHealth:
    key: str
    label: str
    running: bool
    count: int
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TwinrSystemHealth:
    status: str
    captured_at: str
    hostname: str
    uptime_seconds: float | None = None
    load_1m: float | None = None
    load_5m: float | None = None
    load_15m: float | None = None
    cpu_temperature_c: float | None = None
    memory_total_mb: int | None = None
    memory_available_mb: int | None = None
    memory_used_percent: float | None = None
    disk_total_gb: float | None = None
    disk_free_gb: float | None = None
    disk_used_percent: float | None = None
    runtime_status: str | None = None
    runtime_updated_at: str | None = None
    runtime_error: str | None = None
    recent_error_count: int = 0
    services: tuple[ServiceHealth, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["services"] = [service.to_dict() for service in self.services]
        return payload


def collect_system_health(
    config: TwinrConfig,
    *,
    snapshot: RuntimeSnapshot | None = None,
    event_store: TwinrOpsEventStore | None = None,
) -> TwinrSystemHealth:
    events = (event_store or TwinrOpsEventStore.from_config(config)).tail(limit=100)
    error_count = sum(1 for entry in events if str(entry.get("level", "")).lower() == "error")
    project_root = resolve_ops_paths_for_config(config).project_root
    load_1m, load_5m, load_15m = _read_loadavg()
    memory_total_mb, memory_available_mb, memory_used_percent = _read_memory()
    disk_total_gb, disk_free_gb, disk_used_percent = _read_disk(project_root)
    services = _collect_service_health()

    status = "ok"
    if any(service.label == "Conversation loop" and not service.running for service in services):
        status = "warn"
    if memory_used_percent is not None and memory_used_percent >= 90:
        status = "fail"
    elif status == "ok" and memory_used_percent is not None and memory_used_percent >= 80:
        status = "warn"
    if disk_used_percent is not None and disk_used_percent >= 95:
        status = "fail"
    elif status == "ok" and disk_used_percent is not None and disk_used_percent >= 85:
        status = "warn"
    cpu_temperature_c = _read_cpu_temperature_c()
    if cpu_temperature_c is not None and cpu_temperature_c >= 82:
        status = "fail"
    elif status == "ok" and cpu_temperature_c is not None and cpu_temperature_c >= 72:
        status = "warn"
    if (snapshot or RuntimeSnapshot()).error_message:
        status = "warn" if status == "ok" else status

    return TwinrSystemHealth(
        status=status,
        captured_at=_captured_at(),
        hostname=socket.gethostname(),
        uptime_seconds=_read_uptime_seconds(),
        load_1m=load_1m,
        load_5m=load_5m,
        load_15m=load_15m,
        cpu_temperature_c=cpu_temperature_c,
        memory_total_mb=memory_total_mb,
        memory_available_mb=memory_available_mb,
        memory_used_percent=memory_used_percent,
        disk_total_gb=disk_total_gb,
        disk_free_gb=disk_free_gb,
        disk_used_percent=disk_used_percent,
        runtime_status=(snapshot.status if snapshot is not None else None),
        runtime_updated_at=(snapshot.updated_at if snapshot is not None else None),
        runtime_error=(snapshot.error_message if snapshot is not None else None),
        recent_error_count=error_count,
        services=services,
    )


def _captured_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_uptime_seconds() -> float | None:
    path = Path("/proc/uptime")
    if not path.exists():
        return None
    try:
        return float(path.read_text(encoding="utf-8").split()[0])
    except Exception:
        return None


def _read_loadavg() -> tuple[float | None, float | None, float | None]:
    path = Path("/proc/loadavg")
    if not path.exists():
        return (None, None, None)
    try:
        parts = path.read_text(encoding="utf-8").split()
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except Exception:
        return (None, None, None)


def _read_memory() -> tuple[int | None, int | None, float | None]:
    path = Path("/proc/meminfo")
    if not path.exists():
        return (None, None, None)
    values: dict[str, int] = {}
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if ":" not in raw_line:
                continue
            key, rest = raw_line.split(":", 1)
            parts = rest.strip().split()
            if not parts:
                continue
            values[key] = int(parts[0])
    except Exception:
        return (None, None, None)
    total_kib = values.get("MemTotal")
    available_kib = values.get("MemAvailable")
    if total_kib is None or available_kib is None or total_kib <= 0:
        return (None, None, None)
    total_mb = int(total_kib / 1024)
    available_mb = int(available_kib / 1024)
    used_percent = round(((total_kib - available_kib) / total_kib) * 100.0, 1)
    return (total_mb, available_mb, used_percent)


def _read_disk(project_root: Path) -> tuple[float | None, float | None, float | None]:
    try:
        usage = disk_usage(project_root)
    except Exception:
        return (None, None, None)
    total_gb = round(usage.total / (1024**3), 1)
    free_gb = round(usage.free / (1024**3), 1)
    used_percent = round(((usage.total - usage.free) / usage.total) * 100.0, 1) if usage.total else None
    return (total_gb, free_gb, used_percent)


def _read_cpu_temperature_c() -> float | None:
    path = Path("/sys/class/thermal/thermal_zone0/temp")
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8").strip()
        return round(int(raw) / 1000.0, 1)
    except Exception:
        return None


def _collect_service_health() -> tuple[ServiceHealth, ...]:
    service_specs = (
        ("web", "Web UI", "--run-web"),
        (
            "conversation_loop",
            "Conversation loop",
            "--run-realtime-loop|--run-hardware-loop|--run-streaming-loop",
        ),
        ("display", "Display loop", "--run-display-loop"),
    )
    rows: list[ServiceHealth] = []
    for key, label, pattern in service_specs:
        matches = _find_processes(pattern)
        rows.append(
            ServiceHealth(
                key=key,
                label=label,
                running=bool(matches),
                count=len(matches),
                detail=", ".join(matches[:2]) if matches else "No matching process found.",
            )
        )
    return tuple(rows)


def _find_processes(pattern: str) -> tuple[str, ...]:
    if which("pgrep"):
        try:
            result = subprocess.run(
                ["pgrep", "-af", "--", pattern],
                capture_output=True,
                text=True,
                check=False,
                timeout=2.0,
            )
            if result.returncode == 0:
                return tuple(
                    line.strip()
                    for line in result.stdout.splitlines()
                    if line.strip() and "pgrep -af --" not in line
                )
        except Exception:
            pass
    if which("ps") is None:
        return ()
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,args"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
    except Exception:
        return ()
    if result.returncode != 0:
        return ()
    alternatives = tuple(part for part in pattern.split("|") if part)
    matches = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and "ps -eo pid,args" not in line and any(part in line for part in alternatives)
    ]
    return tuple(matches)
