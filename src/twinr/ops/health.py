from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from shutil import disk_usage, which
import socket
import subprocess

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.state.snapshot import RuntimeSnapshot
from twinr.ops.events import TwinrOpsEventStore
from twinr.ops.paths import resolve_ops_paths_for_config


_PROCESS_LIST_TIMEOUT_SECONDS = 0.75  # AUDIT-FIX(#4): Bound synchronous process inspection on the async hot path.
_PGREP_TIMEOUT_SECONDS = 0.5  # AUDIT-FIX(#4): Keep fallback probing short to limit event-loop stalls.
_MAX_SERVICE_DETAIL_ITEMS = 2
_MAX_SERVICE_DETAIL_LENGTH = 160
_ERROR_LEVELS = frozenset({"error", "critical", "fatal"})  # AUDIT-FIX(#7): Count severe events beyond only "error".


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


@dataclass(frozen=True, slots=True)
class _ProcessEntry:
    pid: str
    command: str


def collect_system_health(
    config: TwinrConfig,
    *,
    snapshot: RuntimeSnapshot | None = None,
    event_store: TwinrOpsEventStore | None = None,
) -> TwinrSystemHealth:
    captured_at = _captured_at()  # AUDIT-FIX(#1): Capture once so partial failures still return a coherent snapshot.
    recent_error_count, recent_errors_ok = _read_recent_error_count(
        config,
        event_store,
    )  # AUDIT-FIX(#1): Best-effort ops-log reads must not crash health collection.
    project_root = _resolve_project_root(config)  # AUDIT-FIX(#1): Config/path resolution can fail; degrade instead of raising.
    project_root_ok = project_root is not None  # AUDIT-FIX(#8): Surface an invalid project root as degraded health even when disk stats can fall back.
    load_1m, load_5m, load_15m = _read_loadavg()
    memory_total_mb, memory_available_mb, memory_used_percent = _read_memory()
    disk_total_gb, disk_free_gb, disk_used_percent = _read_disk(project_root)
    services, service_probe_ok = _collect_service_health()  # AUDIT-FIX(#4): Probe processes once with bounded latency.

    runtime_status = (
        _normalize_optional_text(getattr(snapshot, "status", None))
        if snapshot is not None
        else None
    )  # AUDIT-FIX(#7): Normalize non-string snapshot values for JSON safety.
    runtime_updated_at = (
        _normalize_optional_text(getattr(snapshot, "updated_at", None))
        if snapshot is not None
        else None
    )  # AUDIT-FIX(#7): Normalize datetimes to UTC-safe ISO strings.
    runtime_error = (
        _normalize_optional_text(getattr(snapshot, "error_message", None))
        if snapshot is not None
        else None
    )  # AUDIT-FIX(#6): Avoid fragile zero-arg RuntimeSnapshot construction.

    status = "ok"
    if not recent_errors_ok or not service_probe_ok or not project_root_ok:
        status = _escalate_status(status, "warn")  # AUDIT-FIX(#1): Partial health-capture failures must degrade, not disappear.

    conversation_service = _get_service(services, "conversation_loop")
    display_service = _get_service(services, "display")
    if service_probe_ok and (
        conversation_service is None
        or not conversation_service.running
        or conversation_service.count != 1
    ):
        status = _escalate_status(status, "fail")  # AUDIT-FIX(#2): Core conversation service must exist exactly once.
    if (
        service_probe_ok
        and display_service is not None
        and display_service.count != 1
    ):
        status = _escalate_status(status, "warn")  # AUDIT-FIX(#11): Display-path degradation must surface separately from total failure.

    runtime_status_level = _normalize_health_status(runtime_status)
    if runtime_status_level is not None:
        status = _escalate_status(status, runtime_status_level)  # AUDIT-FIX(#3): Propagate runtime snapshot state into overall health.

    if memory_used_percent is not None and memory_used_percent >= 90:
        status = _escalate_status(status, "fail")
    elif memory_used_percent is not None and memory_used_percent >= 80:
        status = _escalate_status(status, "warn")

    if disk_used_percent is not None and disk_used_percent >= 95:
        status = _escalate_status(status, "fail")
    elif disk_used_percent is not None and disk_used_percent >= 85:
        status = _escalate_status(status, "warn")

    cpu_temperature_c = _read_cpu_temperature_c()
    if cpu_temperature_c is not None and cpu_temperature_c >= 82:
        status = _escalate_status(status, "fail")
    elif cpu_temperature_c is not None and cpu_temperature_c >= 72:
        status = _escalate_status(status, "warn")

    if runtime_error:
        status = _escalate_status(status, "warn")

    return TwinrSystemHealth(
        status=status,
        captured_at=captured_at,
        hostname=_read_hostname(),  # AUDIT-FIX(#1): Hostname lookup should not be able to break health reporting.
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
        runtime_status=runtime_status,
        runtime_updated_at=runtime_updated_at,
        runtime_error=runtime_error,
        recent_error_count=recent_error_count,
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
    used_percent = _clamp_percentage(
        ((total_kib - available_kib) / total_kib) * 100.0,
    )  # AUDIT-FIX(#8): Clamp anomalous percentages into a valid telemetry range.
    return (total_mb, available_mb, used_percent)


def _read_disk(project_root: Path | None) -> tuple[float | None, float | None, float | None]:
    target_path = _coerce_disk_usage_path(project_root)  # AUDIT-FIX(#8): Fall back to a real on-disk path instead of returning no disk telemetry.
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
    )  # AUDIT-FIX(#8): Keep disk percentages bounded and JSON-safe.
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


def _collect_service_health() -> tuple[tuple[ServiceHealth, ...], bool]:
    service_specs = (
        ("web", "Web UI", ("--run-web",)),
        (
            "conversation_loop",
            "Conversation loop",
            ("--run-realtime-loop", "--run-hardware-loop", "--run-streaming-loop"),
        ),
        ("display", "Display loop", ("--run-display-loop",)),
    )
    process_entries = _list_process_entries()
    rows: list[ServiceHealth] = []

    if process_entries is not None:
        for key, label, alternatives in service_specs:
            matches = tuple(
                _format_process_detail(entry.pid, entry.command)
                for entry in process_entries
                if _command_matches(entry.command, alternatives)
            )  # AUDIT-FIX(#10): Match exact argv tokens to avoid substring false positives.
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

    fallback_rows = []
    for key, label, alternatives in service_specs:
        matches = _find_processes(alternatives)
        if matches is None:
            unavailable_rows = [
                ServiceHealth(
                    key=service_key,
                    label=service_label,
                    running=False,
                    count=0,
                    detail=_build_service_detail((), probe_ok=False),  # AUDIT-FIX(#11): Distinguish probe failure from an actually stopped service.
                )
                for service_key, service_label, _service_alternatives in service_specs
            ]
            return (tuple(unavailable_rows), False)
        fallback_rows.append(
            ServiceHealth(
                key=key,
                label=label,
                running=bool(matches),
                count=len(matches),
                detail=_build_service_detail(matches, probe_ok=True),
            )
        )
    return (tuple(fallback_rows), True)


def _find_processes(alternatives: tuple[str, ...]) -> tuple[str, ...] | None:
    pgrep_binary = _resolve_binary("pgrep")  # AUDIT-FIX(#9): Use the absolute binary path returned by which().
    if pgrep_binary is None:
        return None
    pattern = "|".join(alternatives)
    try:
        result = subprocess.run(
            [pgrep_binary, "-af", "--", pattern],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=_PGREP_TIMEOUT_SECONDS,
        )
    except Exception:
        return None
    if result.returncode not in (0, 1):
        return None

    matches: list[str] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid, command = _split_process_line(stripped)
        if pid is None or command is None:
            continue
        if not _command_matches(command, alternatives):
            continue  # AUDIT-FIX(#10): Verify exact flag tokens even when pgrep returns regex-adjacent matches.
        matches.append(
            _format_process_detail(pid, command),
        )  # AUDIT-FIX(#5): Return sanitized process summaries instead of raw argv.
    return tuple(matches)


def _list_process_entries() -> tuple[_ProcessEntry, ...] | None:
    ps_binary = _resolve_binary("ps")  # AUDIT-FIX(#9): Use the resolved absolute binary path instead of PATH lookup at execution time.
    if ps_binary is None:
        return None
    try:
        result = subprocess.run(
            [ps_binary, "-eo", "pid=,args="],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=_PROCESS_LIST_TIMEOUT_SECONDS,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None

    entries: list[_ProcessEntry] = []
    for raw_line in result.stdout.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        pid, command = _split_process_line(stripped)
        if pid is None or command is None:
            continue
        entries.append(_ProcessEntry(pid=pid, command=command))
    return tuple(entries)


def _split_process_line(line: str) -> tuple[str | None, str | None]:
    parts = line.split(maxsplit=1)
    if len(parts) != 2:
        return (None, None)
    pid, command = parts[0], parts[1].strip()
    if not pid.isdigit() or not command:
        return (None, None)
    return (pid, command)


def _command_matches(command: str, alternatives: tuple[str, ...]) -> bool:
    tokens = command.split()
    if not tokens:
        return False
    return any(token in alternatives for token in tokens)


def _format_process_detail(pid: str, command: str) -> str:
    tokens = command.split()
    executable = Path(tokens[0]).name if tokens else "<unknown>"
    safe_flags = [token for token in tokens[1:] if token.startswith("--run-")]
    detail = f"pid={pid} {executable}"
    if safe_flags:
        detail = f"{detail} {' '.join(safe_flags[:3])}"
    return _truncate_text(
        detail,
        _MAX_SERVICE_DETAIL_LENGTH,
    )  # AUDIT-FIX(#5): Expose only minimal, non-secret process context in health payloads.


def _build_service_detail(matches: tuple[str, ...], *, probe_ok: bool) -> str:
    if not probe_ok:
        return "Process listing unavailable."
    if not matches:
        return "Service not detected."
    preview = ", ".join(matches[:_MAX_SERVICE_DETAIL_ITEMS])
    if len(matches) > _MAX_SERVICE_DETAIL_ITEMS:
        preview = f"{preview}, +{len(matches) - _MAX_SERVICE_DETAIL_ITEMS} more"
    return preview


def _read_recent_error_count(
    config: TwinrConfig,
    event_store: TwinrOpsEventStore | None,
) -> tuple[int, bool]:
    try:
        store = (
            event_store
            if event_store is not None
            else TwinrOpsEventStore.from_config(config)
        )  # AUDIT-FIX(#6): Use explicit None checks so an injected store is never ignored by truthiness.
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
        return project_root if project_root.exists() else None  # AUDIT-FIX(#8): Treat a missing configured root as degraded and let disk stats fall back safely.
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
    text = str(value).strip()
    return text or None


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
    return which(name)  # AUDIT-FIX(#9): Resolve once and execute absolute paths to avoid PATH-hijack surprises.