# CHANGELOG: 2026-03-30
# BUG-1: Fixed the degraded-mode process probe. The old `pgrep -af` fallback
# expected a `pid ppid cmd` layout even though `pgrep -af` emits `pid cmd`, and
# also called `_command_matches()` with a `str` instead of `_ProcessEntry`.
# Under probe fallback, service detection could fail or raise.
# BUG-2: Fixed high-frequency health polling overhead. The old path spawned `ps`
# and then re-opened `/proc/<pid>/cmdline` for every process on every call.
# This caused avoidable latency spikes on a Raspberry Pi under UI polling.
# SEC-1: Removed PATH-dependent process inspection for core health collection and
# restricted optional binary resolution to trusted system directories. This
# avoids executing attacker-controlled lookalikes from an injected PATH.
# SEC-2: Runtime error text is now redacted and length-bounded before being
# returned in health payloads, reducing practical secret leakage through ops UIs
# and support exports.
# IMP-1: Upgraded host/process telemetry to prefer psutil when available, with a
# procfs fallback, bounded caches, and race-safer process iteration.
# IMP-2: Added Raspberry Pi-specific throttling telemetry (`vcgencmd
# get_throttled`), swap telemetry, and Linux PSI pressure metrics so operator
# health surfaces capture the signals that actually predict degraded UX on a Pi.
# BREAKING: `TwinrSystemHealth.to_dict()` now includes additional optional
# telemetry keys (`swap_*`, `*_pressure_*`, `pi_throttled_*`). Consumers with a
# strict schema must accept these extra fields.

"""Sample bounded Twinr host and service health for ops surfaces.

This module reads runtime snapshot data, host metrics, process presence,
companion-loop locks, and recent ops errors to produce dashboard-friendly
health snapshots.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from shutil import disk_usage
import os
import re
import shlex
import socket
import subprocess
import threading
import time
import warnings

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.state.snapshot import RuntimeSnapshot
from twinr.ops.events import TwinrOpsEventStore
from twinr.ops.locks import loop_lock_owner
from twinr.ops.paths import resolve_ops_paths_for_config

try:
    import psutil as _psutil  # type: ignore[import-not-found]  # pylint: disable=import-error
except Exception:  # pragma: no cover - optional dependency at runtime.
    _psutil = None


_PROCESS_CACHE_TTL_SECONDS = 1.0
_PI_STATE_CACHE_TTL_SECONDS = 1.0
_VCGENCMD_TIMEOUT_SECONDS = 0.25
_MAX_SERVICE_DETAIL_ITEMS = 2
_MAX_SERVICE_DETAIL_LENGTH = 160
_MAX_RUNTIME_ERROR_LENGTH = 400
_ERROR_LEVELS = frozenset({"error", "critical", "fatal"})
_MEMORY_WARN_AVAILABLE_MB = 512
_MEMORY_FAIL_AVAILABLE_MB = 256
_MEMORY_WARN_USED_PERCENT = 80.0
_MEMORY_FAIL_USED_PERCENT = 90.0
_DISK_WARN_USED_PERCENT = 85.0
_DISK_FAIL_USED_PERCENT = 95.0
_CPU_TEMPERATURE_WARN_C = 80.0
_CPU_TEMPERATURE_FAIL_C = 85.0
_PSI_WARN_AVG10_PERCENT = 1.0
_PSI_FAIL_AVG10_PERCENT = 5.0
_SAFE_BINARY_DIRS = (
    "/usr/bin",
    "/bin",
    "/usr/sbin",
    "/sbin",
    "/usr/local/bin",
    "/opt/vc/bin",
)
_SHELL_EXECUTABLES = frozenset({"ash", "bash", "dash", "ksh", "sh", "zsh"})
_PI_THROTTLED_FLAGS = {
    0: "undervoltage-now",
    1: "arm-frequency-capped-now",
    2: "throttled-now",
    3: "soft-temp-limit-now",
    16: "undervoltage-occurred",
    17: "arm-frequency-capped-occurred",
    18: "throttling-occurred",
    19: "soft-temp-limit-occurred",
}
_ASSIGNMENT_SECRET_PATTERN = re.compile(
    r"(?i)\b(authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|secret|password|passwd)\b\s*[:=]\s*([\"'])?[^\s,\"']+\2?"
)
_BEARER_SECRET_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+\b")
_URL_CREDENTIAL_PATTERN = re.compile(
    r"([a-z][a-z0-9+.-]*://)([^/\s:@]+):([^/\s@]+)@",
    re.IGNORECASE,
)
_SERVICE_SPECS = (
    ("web", "Web UI", ("--run-web",)),
    ("conversation_loop", "Conversation loop", ("--run-streaming-loop",)),
    ("runtime_supervisor", "Runtime supervisor", ("--run-runtime-supervisor",)),
    ("display", "Display loop", ("--run-display-loop",)),
)
_CACHE_LOCK = threading.RLock()
_PROCESS_CACHE: "_TimedCache | None" = None
_PI_STATE_CACHE: "_TimedCache | None" = None


@dataclass(frozen=True, slots=True)
class ServiceHealth:
    """Represent the observed state of one Twinr background service."""

    key: str
    label: str
    running: bool
    count: int
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TwinrSystemHealth:
    """Collect one bounded snapshot of Twinr host and runtime health."""

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
    swap_total_mb: int | None = None
    swap_used_mb: int | None = None
    swap_used_percent: float | None = None
    disk_total_gb: float | None = None
    disk_free_gb: float | None = None
    disk_used_percent: float | None = None
    cpu_pressure_some_10s_pct: float | None = None
    memory_pressure_some_10s_pct: float | None = None
    memory_pressure_full_10s_pct: float | None = None
    io_pressure_some_10s_pct: float | None = None
    io_pressure_full_10s_pct: float | None = None
    runtime_status: str | None = None
    runtime_updated_at: str | None = None
    runtime_error: str | None = None
    recent_error_count: int = 0
    pi_throttled_raw: str | None = None
    pi_throttled_flags: tuple[str, ...] = ()
    services: tuple[ServiceHealth, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["pi_throttled_flags"] = list(self.pi_throttled_flags)
        payload["services"] = [service.to_dict() for service in self.services]
        return payload


@dataclass(frozen=True, slots=True)
class _ProcessEntry:
    pid: str
    ppid: str
    command: str
    argv: tuple[str, ...] = ()
    create_time: float | None = None


@dataclass(frozen=True, slots=True)
class _PiThrottledState:
    raw_hex: str | None = None
    flags: tuple[str, ...] = ()

    @property
    def undervoltage_now(self) -> bool:
        return "undervoltage-now" in self.flags

    @property
    def throttled_now(self) -> bool:
        return (
            "throttled-now" in self.flags
            or "arm-frequency-capped-now" in self.flags
        )

    @property
    def degraded(self) -> bool:
        return bool(self.flags)

    @property
    def sticky_only(self) -> bool:
        return self.degraded and not (self.undervoltage_now or self.throttled_now)


@dataclass(frozen=True, slots=True)
class _TimedCache:
    captured_monotonic: float
    value: object


def collect_system_health(
    config: TwinrConfig,
    *,
    snapshot: RuntimeSnapshot | None = None,
    event_store: TwinrOpsEventStore | None = None,
) -> TwinrSystemHealth:
    """Collect a bounded Twinr host and service health snapshot.

    Args:
        config: Twinr runtime configuration.
        snapshot: Optional already-loaded runtime snapshot to fold into the
            returned health state.
        event_store: Optional ops event store override for recent error counts.

    Returns:
        A ``TwinrSystemHealth`` payload suitable for the web UI or support
        export paths.
    """

    captured_at = _captured_at()
    recent_error_count, recent_errors_ok = _read_recent_error_count(config, event_store)
    project_root = _resolve_project_root(config)
    project_root_ok = project_root is not None
    load_1m, load_5m, load_15m = _read_loadavg()
    (
        memory_total_mb,
        memory_available_mb,
        memory_used_percent,
        swap_total_mb,
        swap_used_mb,
        swap_used_percent,
    ) = _read_memory()
    disk_total_gb, disk_free_gb, disk_used_percent = _read_disk(project_root)
    (
        cpu_pressure_some_10s_pct,
        memory_pressure_some_10s_pct,
        memory_pressure_full_10s_pct,
        io_pressure_some_10s_pct,
        io_pressure_full_10s_pct,
    ) = _read_pressure()
    services, service_probe_ok = _collect_service_health(config)
    if service_probe_ok:
        services = _apply_display_companion_health(config, services)

    runtime_status = (
        _normalize_optional_text(getattr(snapshot, "status", None))
        if snapshot is not None
        else None
    )
    runtime_updated_at = (
        _normalize_optional_text(getattr(snapshot, "updated_at", None))
        if snapshot is not None
        else None
    )
    runtime_error = (
        _sanitize_runtime_error(getattr(snapshot, "error_message", None))
        if snapshot is not None
        else None
    )

    pi_state = _read_pi_throttled_state()
    cpu_temperature_c = _read_cpu_temperature_c()

    status = "ok"
    if not recent_errors_ok or not service_probe_ok or not project_root_ok:
        status = _escalate_status(status, "warn")

    display_service = _get_service(services, "display")
    if service_probe_ok:
        conversation_status = _conversation_loop_status_level(services)
        if conversation_status is not None:
            status = _escalate_status(status, conversation_status)

    if service_probe_ok and display_service is not None and display_service.count != 1:
        status = _escalate_status(status, "warn")

    runtime_status_level = _normalize_health_status(runtime_status)
    if runtime_status_level is not None:
        status = _escalate_status(status, runtime_status_level)

    memory_status = assess_memory_pressure_status(
        memory_available_mb=memory_available_mb,
        memory_used_percent=memory_used_percent,
        memory_pressure_full_10s_pct=memory_pressure_full_10s_pct,
    )
    if memory_status is not None:
        status = _escalate_status(status, memory_status)

    io_pressure_status = _assess_pressure_status(io_pressure_full_10s_pct)
    if io_pressure_status is not None:
        status = _escalate_status(status, io_pressure_status)

    if disk_used_percent is not None and disk_used_percent >= _DISK_FAIL_USED_PERCENT:
        status = _escalate_status(status, "fail")
    elif disk_used_percent is not None and disk_used_percent >= _DISK_WARN_USED_PERCENT:
        status = _escalate_status(status, "warn")

    temperature_status = _assess_temperature_status(cpu_temperature_c)
    if temperature_status is not None:
        status = _escalate_status(status, temperature_status)

    pi_status = _assess_pi_throttled_status(pi_state)
    if pi_status is not None:
        status = _escalate_status(status, pi_status)

    if runtime_error:
        status = _escalate_status(status, "warn")

    return TwinrSystemHealth(
        status=status,
        captured_at=captured_at,
        hostname=_read_hostname(),
        uptime_seconds=_read_uptime_seconds(),
        load_1m=load_1m,
        load_5m=load_5m,
        load_15m=load_15m,
        cpu_temperature_c=cpu_temperature_c,
        memory_total_mb=memory_total_mb,
        memory_available_mb=memory_available_mb,
        memory_used_percent=memory_used_percent,
        swap_total_mb=swap_total_mb,
        swap_used_mb=swap_used_mb,
        swap_used_percent=swap_used_percent,
        disk_total_gb=disk_total_gb,
        disk_free_gb=disk_free_gb,
        disk_used_percent=disk_used_percent,
        cpu_pressure_some_10s_pct=cpu_pressure_some_10s_pct,
        memory_pressure_some_10s_pct=memory_pressure_some_10s_pct,
        memory_pressure_full_10s_pct=memory_pressure_full_10s_pct,
        io_pressure_some_10s_pct=io_pressure_some_10s_pct,
        io_pressure_full_10s_pct=io_pressure_full_10s_pct,
        runtime_status=runtime_status,
        runtime_updated_at=runtime_updated_at,
        runtime_error=runtime_error,
        recent_error_count=recent_error_count,
        pi_throttled_raw=pi_state.raw_hex,
        pi_throttled_flags=pi_state.flags,
        services=services,
    )


def _captured_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def _read_uptime_seconds() -> float | None:
    path = Path("/proc/uptime")
    if not path.exists():
        return None
    try:
        return float(path.read_text(encoding="utf-8").split()[0])
    except Exception:
        return None


def _read_loadavg() -> tuple[float | None, float | None, float | None]:
    try:
        load_1m, load_5m, load_15m = os.getloadavg()
        return (round(load_1m, 2), round(load_5m, 2), round(load_15m, 2))
    except Exception:
        return (None, None, None)


def _read_memory() -> tuple[int | None, int | None, float | None, int | None, int | None, float | None]:
    if _psutil is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                memory = _psutil.virtual_memory()
                swap = _psutil.swap_memory()
            return (
                int(memory.total / 1024 / 1024),
                int(memory.available / 1024 / 1024),
                _clamp_percentage(float(memory.percent)),
                int(swap.total / 1024 / 1024),
                int(swap.used / 1024 / 1024),
                _clamp_percentage(float(swap.percent)),
            )
        except Exception:
            pass

    path = Path("/proc/meminfo")
    if not path.exists():
        return (None, None, None, None, None, None)

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
        return (None, None, None, None, None, None)

    total_kib = values.get("MemTotal")
    available_kib = values.get("MemAvailable")
    swap_total_kib = values.get("SwapTotal")
    swap_free_kib = values.get("SwapFree")
    if total_kib is None or available_kib is None or total_kib <= 0:
        return (None, None, None, None, None, None)

    swap_used_percent = None
    swap_used_mb = None
    swap_total_mb = None
    if swap_total_kib is not None and swap_total_kib >= 0:
        swap_total_mb = int(swap_total_kib / 1024)
        if swap_free_kib is not None:
            used_kib = max(0, swap_total_kib - swap_free_kib)
            swap_used_mb = int(used_kib / 1024)
            if swap_total_kib > 0:
                swap_used_percent = _clamp_percentage((used_kib / swap_total_kib) * 100.0)
            else:
                swap_used_percent = 0.0

    return (
        int(total_kib / 1024),
        int(available_kib / 1024),
        _clamp_percentage(((total_kib - available_kib) / total_kib) * 100.0),
        swap_total_mb,
        swap_used_mb,
        swap_used_percent,
    )


def assess_memory_pressure_status(
    *,
    memory_available_mb: int | None,
    memory_used_percent: float | None,
    memory_pressure_full_10s_pct: float | None = None,
) -> str | None:
    """Classify memory pressure for operator-facing health surfaces.

    Linux exposes ``MemAvailable`` as the estimate of how much memory can still
    be used for new work without swapping, which is a better operator signal
    than raw used percentage once page cache and reclaimable memory dominate.
    If Linux PSI is available, sustained ``memory/full`` stalls indicate the
    system is spending time fully blocked on reclaim or swap pressure and should
    outrank raw capacity percentages.
    """

    pressure_status = _assess_pressure_status(memory_pressure_full_10s_pct)
    if pressure_status is not None:
        return pressure_status

    if memory_available_mb is not None:
        if memory_available_mb <= _MEMORY_FAIL_AVAILABLE_MB:
            return "fail"
        if memory_available_mb <= _MEMORY_WARN_AVAILABLE_MB:
            return "warn"
        return None

    if memory_used_percent is None:
        return None
    if memory_used_percent >= _MEMORY_FAIL_USED_PERCENT:
        return "fail"
    if memory_used_percent >= _MEMORY_WARN_USED_PERCENT:
        return "warn"
    return None


def _assess_pressure_status(pressure_avg10_pct: float | None) -> str | None:
    if pressure_avg10_pct is None:
        return None
    if pressure_avg10_pct >= _PSI_FAIL_AVG10_PERCENT:
        return "fail"
    if pressure_avg10_pct >= _PSI_WARN_AVG10_PERCENT:
        return "warn"
    return None


def _read_disk(project_root: Path | None) -> tuple[float | None, float | None, float | None]:
    target_path = _coerce_disk_usage_path(project_root)
    if target_path is None:
        return (None, None, None)
    try:
        usage = disk_usage(target_path)
    except Exception:
        return (None, None, None)
    total_gb = round(usage.total / (1024**3), 1)
    free_gb = round(usage.free / (1024**3), 1)
    used_percent = (
        _clamp_percentage(((usage.total - usage.free) / usage.total) * 100.0)
        if usage.total
        else None
    )
    return (total_gb, free_gb, used_percent)


def _read_cpu_temperature_c() -> float | None:
    if _psutil is not None and hasattr(_psutil, "sensors_temperatures"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sensors = _psutil.sensors_temperatures(fahrenheit=False)
        except Exception:
            sensors = {}
        temperature = _extract_temperature_from_sensors(sensors)
        if temperature is not None:
            return temperature

    path = Path("/sys/class/thermal/thermal_zone0/temp")
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8").strip()
        return round(int(raw) / 1000.0, 1)
    except Exception:
        return None


def _extract_temperature_from_sensors(
    sensors: dict[str, list[object]] | None,
) -> float | None:
    if not sensors:
        return None

    preferred_keys = (
        "cpu_thermal",
        "soc_thermal",
        "cpu-thermal",
        "soc-thermal",
        "cpu",
        "soc",
        "thermal",
    )
    candidates: list[float] = []

    def _extend(entries: list[object]) -> None:
        for entry in entries:
            current = getattr(entry, "current", None)
            if isinstance(current, (int, float)):
                candidates.append(float(current))

    lowered = {str(key).lower(): value for key, value in sensors.items()}
    for key in preferred_keys:
        entries = lowered.get(key)
        if entries:
            _extend(entries)
            if candidates:
                return round(max(candidates), 1)

    for entries in lowered.values():
        _extend(entries)
    if not candidates:
        return None
    return round(max(candidates), 1)


def _read_pressure() -> tuple[float | None, float | None, float | None, float | None, float | None]:
    cpu_pressure_some = _read_psi_avg10(Path("/proc/pressure/cpu"), "some")
    memory_pressure_some = _read_psi_avg10(Path("/proc/pressure/memory"), "some")
    memory_pressure_full = _read_psi_avg10(Path("/proc/pressure/memory"), "full")
    io_pressure_some = _read_psi_avg10(Path("/proc/pressure/io"), "some")
    io_pressure_full = _read_psi_avg10(Path("/proc/pressure/io"), "full")
    return (
        cpu_pressure_some,
        memory_pressure_some,
        memory_pressure_full,
        io_pressure_some,
        io_pressure_full,
    )


def _read_psi_avg10(path: Path, section: str) -> float | None:
    if not path.exists():
        return None
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line.startswith(f"{section} "):
                continue
            values = dict(
                part.split("=", 1)
                for part in line.split()[1:]
                if "=" in part
            )
            avg10 = values.get("avg10")
            if avg10 is None:
                return None
            return _clamp_percentage(float(avg10))
    except Exception:
        return None
    return None


def _collect_service_health(config: TwinrConfig | None = None) -> tuple[tuple[ServiceHealth, ...], bool]:
    process_entries = _get_cached_process_entries()
    if process_entries is None:
        unavailable_rows = tuple(
            ServiceHealth(
                key=key,
                label=label,
                running=False,
                count=0,
                detail=_build_service_detail((), probe_ok=False),
            )
            for key, label, _alternatives in _SERVICE_SPECS
        )
        return (unavailable_rows, False)

    parent_by_pid = {entry.pid: entry.ppid for entry in process_entries}
    rows: list[ServiceHealth] = []
    for key, label, alternatives in _SERVICE_SPECS:
        matching_entries = tuple(
            entry
            for entry in process_entries
            if _command_matches(entry, alternatives)
        )
        if key == "conversation_loop":
            matching_entries = _filter_conversation_loop_entries(
                config,
                matching_entries,
                parent_by_pid=parent_by_pid,
            )

        matches = tuple(
            _format_process_detail(
                entry.pid,
                entry.command,
                argv=entry.argv,
                create_time=entry.create_time,
            )
            for entry in sorted(matching_entries, key=_sort_process_entry_key)
        )
        rows.append(
            ServiceHealth(
                key=key,
                label=label,
                running=bool(matches),
                count=len(matches),
                detail=_build_service_detail(matches, probe_ok=True),
            )
        )
    return (tuple(rows), True)


def _sort_process_entry_key(entry: _ProcessEntry) -> tuple[int, float]:
    try:
        pid_key = int(entry.pid)
    except Exception:
        pid_key = 0
    create_time = entry.create_time if entry.create_time is not None else 0.0
    return (pid_key, create_time)


def _get_cached_process_entries() -> tuple[_ProcessEntry, ...] | None:
    global _PROCESS_CACHE
    now = time.monotonic()
    with _CACHE_LOCK:
        cache = _PROCESS_CACHE
        if cache is not None and (now - cache.captured_monotonic) <= _PROCESS_CACHE_TTL_SECONDS:
            return cache.value

    entries = _list_process_entries()
    if entries is None:
        return None

    with _CACHE_LOCK:
        _PROCESS_CACHE = _TimedCache(captured_monotonic=now, value=entries)
    return entries


def _list_process_entries() -> tuple[_ProcessEntry, ...] | None:
    if _psutil is not None:
        entries = _list_process_entries_psutil()
        if entries is not None:
            return entries
    return _list_process_entries_procfs()


def _list_process_entries_psutil() -> tuple[_ProcessEntry, ...] | None:
    try:
        entries: list[_ProcessEntry] = []
        for proc in _psutil.process_iter(
            ["pid", "ppid", "cmdline", "name", "create_time"],
            ad_value=None,
        ):
            info = proc.info
            pid = info.get("pid")
            if pid is None:
                continue
            ppid = info.get("ppid")
            raw_argv = info.get("cmdline") or ()
            argv = tuple(str(part) for part in raw_argv if part)
            name = str(info.get("name") or "").strip()
            command = " ".join(argv) if argv else name
            if not command:
                continue
            create_time = info.get("create_time")
            entries.append(
                _ProcessEntry(
                    pid=str(pid),
                    ppid=str(ppid or 0),
                    command=command,
                    argv=argv,
                    create_time=float(create_time) if isinstance(create_time, (int, float)) else None,
                )
            )
        return tuple(entries)
    except Exception:
        return None


def _list_process_entries_procfs() -> tuple[_ProcessEntry, ...] | None:
    proc_root = Path("/proc")
    if not proc_root.exists():
        return None

    entries: list[_ProcessEntry] = []
    try:
        proc_paths = tuple(proc_root.iterdir())
    except Exception:
        return None

    for proc_path in proc_paths:
        pid = proc_path.name
        if not pid.isdigit():
            continue
        try:
            ppid = _read_proc_ppid(proc_path) or "0"
            argv = _read_process_argv(pid)
            command = " ".join(argv) if argv else _read_proc_comm(proc_path)
            if not command:
                continue
            entries.append(
                _ProcessEntry(
                    pid=pid,
                    ppid=ppid,
                    command=command,
                    argv=argv,
                    create_time=_read_proc_create_time(proc_path),
                )
            )
        except Exception:
            continue
    return tuple(entries)


def _read_proc_ppid(proc_path: Path) -> str | None:
    status_path = proc_path / "status"
    if not status_path.exists():
        return None
    try:
        for raw_line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not raw_line.startswith("PPid:"):
                continue
            value = raw_line.split(":", 1)[1].strip()
            return value if value.isdigit() else None
    except Exception:
        return None
    return None


def _read_proc_comm(proc_path: Path) -> str:
    try:
        return (proc_path / "comm").read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return ""


def _read_proc_create_time(proc_path: Path) -> float | None:
    stat_path = proc_path / "stat"
    if not stat_path.exists():
        return None
    try:
        data = stat_path.read_text(encoding="utf-8", errors="replace").strip()
        if not data:
            return None
        right_paren = data.rfind(")")
        if right_paren < 0:
            return None
        rest = data[right_paren + 2 :].split()
        if len(rest) < 20:
            return None
        start_ticks = int(rest[19])
        hz = os.sysconf("SC_CLK_TCK")
        boot_time = _read_boot_time_epoch()
        if hz <= 0 or boot_time is None:
            return None
        return boot_time + (start_ticks / hz)
    except Exception:
        return None


def _read_boot_time_epoch() -> float | None:
    path = Path("/proc/stat")
    if not path.exists():
        return None
    try:
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not raw_line.startswith("btime "):
                continue
            return float(raw_line.split()[1])
    except Exception:
        return None
    return None


def _read_process_argv(pid: str) -> tuple[str, ...]:
    path = Path("/proc") / pid / "cmdline"
    try:
        raw = path.read_bytes()
    except Exception:
        return ()
    if not raw:
        return ()
    return tuple(
        part.decode("utf-8", errors="replace")
        for part in raw.split(b"\0")
        if part
    )


def _command_tokens(entry: _ProcessEntry) -> tuple[str, ...]:
    if entry.argv:
        tokens = entry.argv
    else:
        try:
            tokens = tuple(shlex.split(entry.command))
        except Exception:
            tokens = tuple(entry.command.split())

    if len(tokens) >= 2 and tokens[1] == "-c" and Path(tokens[0]).name in _SHELL_EXECUTABLES:
        return tokens[:2]
    return tokens


def _command_matches(entry: _ProcessEntry, alternatives: tuple[str, ...]) -> bool:
    tokens = _command_tokens(entry)
    if not tokens:
        return False
    return any(token in alternatives for token in tokens)


def _format_process_detail(
    pid: str,
    command: str,
    *,
    argv: tuple[str, ...] = (),
    create_time: float | None = None,
) -> str:
    tokens = argv or _command_tokens(_ProcessEntry(pid=pid, ppid="0", command=command))
    executable = Path(tokens[0]).name if tokens else "<unknown>"
    safe_flags = [token for token in tokens[1:] if token.startswith("--run-")]
    detail = f"pid={pid} {executable}"
    if safe_flags:
        detail = f"{detail} {' '.join(safe_flags[:3])}"
    if create_time is not None:
        try:
            created_at = datetime.fromtimestamp(create_time, tz=timezone.utc)
            detail = f"{detail} started={created_at.isoformat(timespec='seconds')}"
        except Exception:
            pass
    return _truncate_text(detail, _MAX_SERVICE_DETAIL_LENGTH)


def _build_service_detail(matches: tuple[str, ...], *, probe_ok: bool) -> str:
    if not probe_ok:
        return "Process listing unavailable."
    if not matches:
        return "Service not detected."
    preview = ", ".join(matches[:_MAX_SERVICE_DETAIL_ITEMS])
    if len(matches) > _MAX_SERVICE_DETAIL_ITEMS:
        preview = f"{preview}, +{len(matches) - _MAX_SERVICE_DETAIL_ITEMS} more"
    return preview


def _filter_conversation_loop_entries(
    config: TwinrConfig | None,
    entries: tuple[_ProcessEntry, ...],
    *,
    parent_by_pid: dict[str, str],
) -> tuple[_ProcessEntry, ...]:
    if not entries or config is None:
        return entries
    owner_pid = _resolve_streaming_loop_owner_pid(config, entries)
    if owner_pid is None:
        return entries

    filtered = tuple(
        entry
        for entry in entries
        if entry.pid == owner_pid or not _pid_descends_from(entry.pid, owner_pid, parent_by_pid)
    )
    return filtered or entries


def _resolve_streaming_loop_owner_pid(
    config: TwinrConfig,
    entries: tuple[_ProcessEntry, ...],
) -> str | None:
    owner_pid = loop_lock_owner(config, "streaming-loop")
    owner_text = str(owner_pid).strip() if owner_pid is not None else ""
    if not owner_text.isdigit():
        return None
    for entry in entries:
        if entry.pid == owner_text:
            return owner_text
    return None


def _pid_descends_from(pid: str, ancestor_pid: str, parent_by_pid: dict[str, str]) -> bool:
    current_pid = pid
    visited: set[str] = set()
    while current_pid and current_pid not in visited:
        visited.add(current_pid)
        parent_pid = parent_by_pid.get(current_pid)
        if not parent_pid:
            return False
        if parent_pid == ancestor_pid:
            return True
        current_pid = parent_pid
    return False


def _apply_display_companion_health(
    config: TwinrConfig,
    services: tuple[ServiceHealth, ...],
) -> tuple[ServiceHealth, ...]:
    # Import lazily so display heartbeat helpers can import ops locks without
    # pulling health back in through twinr.ops.__init__ during package init.
    from twinr.display.heartbeat import assess_display_companion_health

    display_service = _get_service(services, "display")
    if display_service is None:
        return services
    if display_service.running or display_service.count != 0:
        return services

    companion = assess_display_companion_health(config, loop_owner_fn=loop_lock_owner)
    if companion.owner_pid is None:
        return services

    updated: list[ServiceHealth] = []
    for service in services:
        if service.key != "display":
            updated.append(service)
            continue
        updated.append(
            ServiceHealth(
                key=service.key,
                label=service.label,
                running=companion.running,
                count=companion.count,
                detail=_truncate_text(
                    _sanitize_text(companion.detail),
                    _MAX_SERVICE_DETAIL_LENGTH,
                ),
            )
        )
    return tuple(updated)


def _read_recent_error_count(
    config: TwinrConfig,
    event_store: TwinrOpsEventStore | None,
) -> tuple[int, bool]:
    try:
        store = event_store if event_store is not None else TwinrOpsEventStore.from_config(config)
        events = store.tail(limit=100)
    except Exception:
        return (0, False)

    count = 0
    try:
        for entry in events:
            if not isinstance(entry, dict):
                continue
            level = str(entry.get("level", "")).strip().lower()
            if level in _ERROR_LEVELS:
                count += 1
    except Exception:
        return (count, False)
    return (count, True)


def _resolve_project_root(config: TwinrConfig) -> Path | None:
    try:
        project_root = Path(resolve_ops_paths_for_config(config).project_root)
    except Exception:
        return None
    try:
        return project_root if project_root.exists() else None
    except OSError:
        return None


def _coerce_disk_usage_path(project_root: Path | None) -> Path | None:
    candidate = Path("/") if project_root is None else Path(project_root).expanduser()
    for path in (candidate, *candidate.parents):
        try:
            if path.exists():
                return path
        except OSError:
            continue
    root = Path("/")
    try:
        return root if root.exists() else None
    except OSError:
        return None


def _read_pi_throttled_state() -> _PiThrottledState:
    global _PI_STATE_CACHE
    now = time.monotonic()
    with _CACHE_LOCK:
        cache = _PI_STATE_CACHE
        if cache is not None and (now - cache.captured_monotonic) <= _PI_STATE_CACHE_TTL_SECONDS:
            return cache.value

    state = _probe_pi_throttled_state()
    with _CACHE_LOCK:
        _PI_STATE_CACHE = _TimedCache(captured_monotonic=now, value=state)
    return state


def _probe_pi_throttled_state() -> _PiThrottledState:
    vcgencmd_binary = _resolve_binary("vcgencmd")
    if vcgencmd_binary is None:
        return _PiThrottledState()

    try:
        result = subprocess.run(
            [vcgencmd_binary, "get_throttled"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=_VCGENCMD_TIMEOUT_SECONDS,
        )
    except Exception:
        return _PiThrottledState()

    if result.returncode != 0:
        return _PiThrottledState()

    output = result.stdout.strip()
    match = re.search(r"throttled=(0x[0-9a-fA-F]+)", output)
    if match is None:
        return _PiThrottledState()

    raw_hex = match.group(1).lower()
    try:
        value = int(raw_hex, 16)
    except ValueError:
        return _PiThrottledState()

    flags = tuple(
        label
        for bit, label in _PI_THROTTLED_FLAGS.items()
        if value & (1 << bit)
    )
    return _PiThrottledState(raw_hex=raw_hex, flags=flags)


def _assess_temperature_status(cpu_temperature_c: float | None) -> str | None:
    if cpu_temperature_c is None:
        return None
    if cpu_temperature_c >= _CPU_TEMPERATURE_FAIL_C:
        return "fail"
    if cpu_temperature_c >= _CPU_TEMPERATURE_WARN_C:
        return "warn"
    return None


def _assess_pi_throttled_status(state: _PiThrottledState) -> str | None:
    if state.undervoltage_now or state.throttled_now:
        return "fail"
    if state.degraded:
        return "warn"
    return None


def _normalize_optional_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        normalized = (
            value.replace(tzinfo=timezone.utc)
            if value.tzinfo is None
            else value.astimezone(timezone.utc)
        )
        return normalized.isoformat()
    text = _sanitize_text(str(value))
    return text or None


def _sanitize_runtime_error(value: object) -> str | None:
    if value is None:
        return None
    text = _sanitize_text(str(value))
    if not text:
        return None
    return _truncate_text(text, _MAX_RUNTIME_ERROR_LENGTH)


def _sanitize_text(text: str) -> str:
    sanitized = text.replace("\r", " ").replace("\n", " ").strip()
    if not sanitized:
        return ""
    sanitized = _URL_CREDENTIAL_PATTERN.sub(
        r"\1<redacted>:<redacted>@",
        sanitized,
    )
    sanitized = _BEARER_SECRET_PATTERN.sub("Bearer <redacted>", sanitized)
    sanitized = _ASSIGNMENT_SECRET_PATTERN.sub(
        lambda match: f"{match.group(1)}=<redacted>",
        sanitized,
    )
    return " ".join(sanitized.split())


def _normalize_health_status(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"ok", "ready", "healthy"}:
        return "ok"
    if normalized in {"warn", "warning", "degraded", "starting"}:
        return "warn"
    if normalized in {"fail", "failed", "error", "critical", "panic"}:
        return "fail"
    return None


def _get_service(
    services: tuple[ServiceHealth, ...],
    key: str,
) -> ServiceHealth | None:
    for service in services:
        if service.key == key:
            return service
    return None


def _conversation_loop_status_level(services: tuple[ServiceHealth, ...]) -> str | None:
    conversation_service = _get_service(services, "conversation_loop")
    if _service_running_exactly_once(conversation_service):
        return None

    runtime_supervisor = _get_service(services, "runtime_supervisor")
    if _service_running_exactly_once(runtime_supervisor):
        return "warn"
    return "fail"


def _service_running_exactly_once(service: ServiceHealth | None) -> bool:
    return service is not None and service.running and service.count == 1


def _clamp_percentage(value: float | None) -> float | None:
    if value is None:
        return None
    return round(min(100.0, max(0.0, value)), 1)


def _truncate_text(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 1].rstrip()}…"


def _escalate_status(current: str, candidate: str) -> str:
    severity_order = {"ok": 0, "warn": 1, "fail": 2}
    return candidate if severity_order[candidate] > severity_order[current] else current


def _resolve_binary(name: str) -> str | None:
    if not name or "/" in name or "\x00" in name:
        return None
    for directory in _SAFE_BINARY_DIRS:
        candidate = Path(directory) / name
        try:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        except OSError:
            continue
    return None
