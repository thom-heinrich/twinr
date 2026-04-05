"""Persist process-local memory attribution for the live streaming loop.

This module samples Linux procfs memory counters from inside the live Twinr
process and keeps one bounded attribution snapshot under the ops store. The
snapshot is intentionally phase-oriented: startup and heavy-subsystem
boundaries record current RSS/anonymous memory so later health surfaces can
name the concrete owner path behind a memory-pressure warning instead of only
showing host-level symptoms.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import threading

from twinr.agent.base_agent.config import TwinrConfig
from twinr.ops.paths import resolve_ops_paths_for_config


_DEFAULT_FILE_NAME = "streaming_memory_segments.json"
_SCHEMA_VERSION = 1
_MAX_PHASES = 24
_MAX_LABEL_CHARS = 80
_MAX_DETAIL_CHARS = 240
_FILE_MODE = 0o600
_STORE_LOCK = threading.RLock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compact_text(value: object | None, *, limit: int) -> str | None:
    if value is None:
        return None
    text = "".join(ch if ch.isprintable() else " " for ch in str(value))
    compact = " ".join(text.split()).strip()
    if not compact:
        return None
    if len(compact) <= limit:
        return compact
    if limit <= 1:
        return compact[:limit]
    return compact[: limit - 1].rstrip() + "…"


def _normalize_optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            return int(value)
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    if isinstance(value, (bytes, bytearray)):
        try:
            return int(value.decode("utf-8", errors="replace").strip())
        except ValueError:
            return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _resolve_default_path(config: TwinrConfig) -> Path:
    return resolve_ops_paths_for_config(config).ops_store_root / _DEFAULT_FILE_NAME


def _read_proc_lines(pid: int | None, relative_path: str) -> list[str]:
    target = Path("/proc") / ("self" if pid is None else str(int(pid))) / relative_path
    try:
        return target.read_text(encoding="utf-8", errors="replace").splitlines()
    except (FileNotFoundError, OSError, ValueError):
        return []


def _read_boot_id() -> str | None:
    path = Path("/proc/sys/kernel/random/boot_id")
    try:
        value = path.read_text(encoding="utf-8", errors="replace").strip()
    except (FileNotFoundError, OSError):
        return None
    return value or None


def _read_process_start_ticks(pid: int | None) -> int | None:
    stat_lines = _read_proc_lines(pid, "stat")
    if not stat_lines:
        return None
    data = stat_lines[0].strip()
    if not data:
        return None
    right_paren = data.rfind(")")
    if right_paren < 0:
        return None
    rest = data[right_paren + 2 :].split()
    if len(rest) < 20:
        return None
    try:
        return int(rest[19])
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class ProcessMemoryMetrics:
    """Capture one procfs-backed memory snapshot for a single process."""

    vm_rss_kb: int | None = None
    rss_anon_kb: int | None = None
    rss_file_kb: int | None = None
    vm_swap_kb: int | None = None
    threads: int | None = None
    smaps_rss_kb: int | None = None
    pss_anon_kb: int | None = None
    private_dirty_kb: int | None = None
    anonymous_kb: int | None = None
    smaps_swap_kb: int | None = None

    @classmethod
    def from_proc(cls, pid: int | None = None) -> "ProcessMemoryMetrics":
        """Read one bounded Linux procfs memory sample."""

        status_values: dict[str, int] = {}
        for raw_line in _read_proc_lines(pid, "status"):
            if ":" not in raw_line:
                continue
            key, raw_value = raw_line.split(":", 1)
            parts = raw_value.strip().split()
            if not parts:
                continue
            try:
                status_values[key] = int(parts[0])
            except ValueError:
                continue

        smaps_values: dict[str, int] = {}
        for raw_line in _read_proc_lines(pid, "smaps_rollup"):
            if ":" not in raw_line:
                continue
            key, raw_value = raw_line.split(":", 1)
            parts = raw_value.strip().split()
            if not parts:
                continue
            try:
                smaps_values[key] = int(parts[0])
            except ValueError:
                continue

        return cls(
            vm_rss_kb=status_values.get("VmRSS"),
            rss_anon_kb=status_values.get("RssAnon"),
            rss_file_kb=status_values.get("RssFile"),
            vm_swap_kb=status_values.get("VmSwap"),
            threads=status_values.get("Threads"),
            smaps_rss_kb=smaps_values.get("Rss"),
            pss_anon_kb=smaps_values.get("Pss_Anon"),
            private_dirty_kb=smaps_values.get("Private_Dirty"),
            anonymous_kb=smaps_values.get("Anonymous"),
            smaps_swap_kb=smaps_values.get("Swap"),
        )

    def preferred_rss_kb(self) -> int | None:
        return self.smaps_rss_kb if self.smaps_rss_kb is not None else self.vm_rss_kb

    def preferred_anonymous_kb(self) -> int | None:
        for candidate in (
            self.anonymous_kb,
            self.private_dirty_kb,
            self.pss_anon_kb,
            self.rss_anon_kb,
        ):
            if candidate is not None:
                return candidate
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "vm_rss_kb": self.vm_rss_kb,
            "rss_anon_kb": self.rss_anon_kb,
            "rss_file_kb": self.rss_file_kb,
            "vm_swap_kb": self.vm_swap_kb,
            "threads": self.threads,
            "smaps_rss_kb": self.smaps_rss_kb,
            "pss_anon_kb": self.pss_anon_kb,
            "private_dirty_kb": self.private_dirty_kb,
            "anonymous_kb": self.anonymous_kb,
            "smaps_swap_kb": self.smaps_swap_kb,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ProcessMemoryMetrics":
        return cls(
            vm_rss_kb=_normalize_optional_int(payload.get("vm_rss_kb")),
            rss_anon_kb=_normalize_optional_int(payload.get("rss_anon_kb")),
            rss_file_kb=_normalize_optional_int(payload.get("rss_file_kb")),
            vm_swap_kb=_normalize_optional_int(payload.get("vm_swap_kb")),
            threads=_normalize_optional_int(payload.get("threads")),
            smaps_rss_kb=_normalize_optional_int(payload.get("smaps_rss_kb")),
            pss_anon_kb=_normalize_optional_int(payload.get("pss_anon_kb")),
            private_dirty_kb=_normalize_optional_int(payload.get("private_dirty_kb")),
            anonymous_kb=_normalize_optional_int(payload.get("anonymous_kb")),
            smaps_swap_kb=_normalize_optional_int(payload.get("smaps_swap_kb")),
        )


@dataclass(frozen=True, slots=True)
class StreamingMemoryPhase:
    """Record one named attribution boundary inside the live process."""

    label: str
    owner_label: str | None
    owner_detail: str | None
    captured_at: str
    pid: int
    boot_id: str | None
    pid_start_ticks: int | None
    metrics: ProcessMemoryMetrics
    rss_delta_kb: int | None = None
    anonymous_delta_kb: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "owner_label": self.owner_label,
            "owner_detail": self.owner_detail,
            "captured_at": self.captured_at,
            "pid": self.pid,
            "boot_id": self.boot_id,
            "pid_start_ticks": self.pid_start_ticks,
            "metrics": self.metrics.to_dict(),
            "rss_delta_kb": self.rss_delta_kb,
            "anonymous_delta_kb": self.anonymous_delta_kb,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "StreamingMemoryPhase | None":
        metrics_raw = payload.get("metrics")
        if not isinstance(metrics_raw, dict):
            return None
        pid = _normalize_optional_int(payload.get("pid"))
        if pid is None:
            return None
        label = _compact_text(payload.get("label"), limit=_MAX_LABEL_CHARS)
        if not label:
            return None
        return cls(
            label=label,
            owner_label=_compact_text(payload.get("owner_label"), limit=_MAX_LABEL_CHARS),
            owner_detail=_compact_text(payload.get("owner_detail"), limit=_MAX_DETAIL_CHARS),
            captured_at=_compact_text(payload.get("captured_at"), limit=64) or _utc_now_iso(),
            pid=pid,
            boot_id=_compact_text(payload.get("boot_id"), limit=80),
            pid_start_ticks=_normalize_optional_int(payload.get("pid_start_ticks")),
            metrics=ProcessMemoryMetrics.from_dict(metrics_raw),
            rss_delta_kb=_normalize_optional_int(payload.get("rss_delta_kb")),
            anonymous_delta_kb=_normalize_optional_int(payload.get("anonymous_delta_kb")),
        )


@dataclass(frozen=True, slots=True)
class StreamingMemorySnapshot:
    """Persist the newest bounded attribution state for the streaming loop."""

    schema_version: int
    captured_at: str
    pid: int
    boot_id: str | None
    pid_start_ticks: int | None
    current_metrics: ProcessMemoryMetrics
    owner_label: str | None
    owner_detail: str | None
    owner_rss_delta_kb: int | None
    owner_anonymous_delta_kb: int | None
    phases: tuple[StreamingMemoryPhase, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "captured_at": self.captured_at,
            "pid": self.pid,
            "boot_id": self.boot_id,
            "pid_start_ticks": self.pid_start_ticks,
            "current_metrics": self.current_metrics.to_dict(),
            "owner_label": self.owner_label,
            "owner_detail": self.owner_detail,
            "owner_rss_delta_kb": self.owner_rss_delta_kb,
            "owner_anonymous_delta_kb": self.owner_anonymous_delta_kb,
            "phases": [phase.to_dict() for phase in self.phases],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "StreamingMemorySnapshot | None":
        metrics_raw = payload.get("current_metrics")
        phases_raw = payload.get("phases")
        pid = _normalize_optional_int(payload.get("pid"))
        if pid is None or not isinstance(metrics_raw, dict) or not isinstance(phases_raw, list):
            return None
        phases: list[StreamingMemoryPhase] = []
        for item in phases_raw[:_MAX_PHASES]:
            if not isinstance(item, dict):
                continue
            phase = StreamingMemoryPhase.from_dict(item)
            if phase is not None:
                phases.append(phase)
        return cls(
            schema_version=_normalize_optional_int(payload.get("schema_version")) or _SCHEMA_VERSION,
            captured_at=_compact_text(payload.get("captured_at"), limit=64) or _utc_now_iso(),
            pid=pid,
            boot_id=_compact_text(payload.get("boot_id"), limit=80),
            pid_start_ticks=_normalize_optional_int(payload.get("pid_start_ticks")),
            current_metrics=ProcessMemoryMetrics.from_dict(metrics_raw),
            owner_label=_compact_text(payload.get("owner_label"), limit=_MAX_LABEL_CHARS),
            owner_detail=_compact_text(payload.get("owner_detail"), limit=_MAX_DETAIL_CHARS),
            owner_rss_delta_kb=_normalize_optional_int(payload.get("owner_rss_delta_kb")),
            owner_anonymous_delta_kb=_normalize_optional_int(payload.get("owner_anonymous_delta_kb")),
            phases=tuple(phases),
        )


def _identity_matches(
    snapshot: StreamingMemorySnapshot | None,
    *,
    pid: int,
    boot_id: str | None,
    pid_start_ticks: int | None,
) -> bool:
    if snapshot is None:
        return False
    if snapshot.pid != pid:
        return False
    if snapshot.boot_id and boot_id and snapshot.boot_id != boot_id:
        return False
    if snapshot.pid_start_ticks is not None and pid_start_ticks is not None:
        return snapshot.pid_start_ticks == pid_start_ticks
    return True


def _synthesized_owner_detail(
    *,
    label: str,
    owner_detail: str | None,
    metrics: ProcessMemoryMetrics,
    rss_delta_kb: int | None,
    anonymous_delta_kb: int | None,
) -> str:
    parts = []
    if owner_detail:
        parts.append(owner_detail)
    parts.append(f"phase={label}")
    preferred_rss_kb = metrics.preferred_rss_kb()
    preferred_anon_kb = metrics.preferred_anonymous_kb()
    if preferred_rss_kb is not None:
        parts.append(f"rss_mb={int(round(preferred_rss_kb / 1024.0))}")
    if preferred_anon_kb is not None:
        parts.append(f"anonymous_mb={int(round(preferred_anon_kb / 1024.0))}")
    if rss_delta_kb is not None:
        parts.append(f"rss_delta_mb={int(round(rss_delta_kb / 1024.0))}")
    if anonymous_delta_kb is not None:
        parts.append(f"anonymous_delta_mb={int(round(anonymous_delta_kb / 1024.0))}")
    return _compact_text(" ".join(parts), limit=_MAX_DETAIL_CHARS) or label


def _select_owner(phases: tuple[StreamingMemoryPhase, ...]) -> tuple[str | None, str | None, int | None, int | None]:
    if len(phases) <= 1:
        return (None, None, None, None)
    best_phase: StreamingMemoryPhase | None = None
    best_key: tuple[int, int] | None = None
    for phase in phases[1:]:
        anon_delta = int(phase.anonymous_delta_kb or 0)
        rss_delta = int(phase.rss_delta_kb or 0)
        key = (anon_delta, rss_delta)
        if best_key is None or key > best_key:
            best_key = key
            best_phase = phase
    if best_phase is None:
        return (None, None, None, None)
    owner_label = best_phase.owner_label or best_phase.label
    owner_detail = _synthesized_owner_detail(
        label=best_phase.label,
        owner_detail=best_phase.owner_detail,
        metrics=best_phase.metrics,
        rss_delta_kb=best_phase.rss_delta_kb,
        anonymous_delta_kb=best_phase.anonymous_delta_kb,
    )
    return (
        owner_label,
        owner_detail,
        best_phase.rss_delta_kb,
        best_phase.anonymous_delta_kb,
    )


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(serialized)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    os.chmod(temp_path, _FILE_MODE)
    os.replace(temp_path, path)


@dataclass(slots=True)
class StreamingMemoryAttributionStore:
    """Read and write the bounded live streaming-memory attribution snapshot."""

    path: Path

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "StreamingMemoryAttributionStore":
        return cls(_resolve_default_path(config))

    def load(self) -> StreamingMemorySnapshot | None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, TypeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        return StreamingMemorySnapshot.from_dict(payload)

    def record_phase(
        self,
        *,
        label: str,
        owner_label: str | None = None,
        owner_detail: str | None = None,
        replace: bool = False,
        reset: bool = False,
        pid: int | None = None,
    ) -> StreamingMemorySnapshot | None:
        """Capture one named phase and refresh the bounded attribution snapshot."""

        normalized_label = _compact_text(label, limit=_MAX_LABEL_CHARS)
        if not normalized_label:
            return None

        current_pid = int(os.getpid() if pid is None else pid)
        boot_id = _read_boot_id()
        pid_start_ticks = _read_process_start_ticks(current_pid)
        metrics = ProcessMemoryMetrics.from_proc(current_pid)

        with _STORE_LOCK:
            snapshot = None if reset else self.load()
            if not _identity_matches(
                snapshot,
                pid=current_pid,
                boot_id=boot_id,
                pid_start_ticks=pid_start_ticks,
            ):
                snapshot = None

            phases = list(snapshot.phases if snapshot is not None else ())
            if replace:
                phases = [phase for phase in phases if phase.label != normalized_label]

            baseline_phase = phases[0] if phases else None
            baseline_metrics = baseline_phase.metrics if baseline_phase is not None else metrics
            preferred_rss_kb = metrics.preferred_rss_kb()
            baseline_rss_kb = baseline_metrics.preferred_rss_kb()
            preferred_anon_kb = metrics.preferred_anonymous_kb()
            baseline_anon_kb = baseline_metrics.preferred_anonymous_kb()
            rss_delta_kb = (
                int(preferred_rss_kb - baseline_rss_kb)
                if preferred_rss_kb is not None and baseline_rss_kb is not None
                else None
            )
            anonymous_delta_kb = (
                int(preferred_anon_kb - baseline_anon_kb)
                if preferred_anon_kb is not None and baseline_anon_kb is not None
                else None
            )
            phase = StreamingMemoryPhase(
                label=normalized_label,
                owner_label=_compact_text(owner_label, limit=_MAX_LABEL_CHARS),
                owner_detail=_compact_text(owner_detail, limit=_MAX_DETAIL_CHARS),
                captured_at=_utc_now_iso(),
                pid=current_pid,
                boot_id=boot_id,
                pid_start_ticks=pid_start_ticks,
                metrics=metrics,
                rss_delta_kb=rss_delta_kb,
                anonymous_delta_kb=anonymous_delta_kb,
            )
            phases.append(phase)
            if len(phases) > _MAX_PHASES:
                phases = [phases[0], *phases[-(_MAX_PHASES - 1) :]]

            owner_label_value, owner_detail_value, owner_rss_delta_kb, owner_anon_delta_kb = _select_owner(tuple(phases))
            updated = StreamingMemorySnapshot(
                schema_version=_SCHEMA_VERSION,
                captured_at=phase.captured_at,
                pid=current_pid,
                boot_id=boot_id,
                pid_start_ticks=pid_start_ticks,
                current_metrics=metrics,
                owner_label=owner_label_value,
                owner_detail=owner_detail_value,
                owner_rss_delta_kb=owner_rss_delta_kb,
                owner_anonymous_delta_kb=owner_anon_delta_kb,
                phases=tuple(phases),
            )
            _atomic_write_json(self.path, updated.to_dict())
            return updated


def record_streaming_memory_phase(
    config: TwinrConfig,
    *,
    label: str,
    owner_label: str | None = None,
    owner_detail: str | None = None,
    replace: bool = False,
    reset: bool = False,
) -> StreamingMemorySnapshot | None:
    """Record one phase to the default per-config streaming-memory snapshot."""

    return StreamingMemoryAttributionStore.from_config(config).record_phase(
        label=label,
        owner_label=owner_label,
        owner_detail=owner_detail,
        replace=replace,
        reset=reset,
    )


def record_streaming_memory_phase_best_effort(
    config: TwinrConfig,
    *,
    label: str,
    owner_label: str | None = None,
    owner_detail: str | None = None,
    replace: bool = False,
    reset: bool = False,
) -> StreamingMemorySnapshot | None:
    """Record one phase, but never let observability break the runtime path."""

    try:
        return record_streaming_memory_phase(
            config,
            label=label,
            owner_label=owner_label,
            owner_detail=owner_detail,
            replace=replace,
            reset=reset,
        )
    except Exception:
        return None


def load_current_streaming_memory_snapshot(
    config: TwinrConfig,
    *,
    expected_pid: int | None = None,
) -> StreamingMemorySnapshot | None:
    """Load the latest attribution snapshot for the current live streaming PID."""

    snapshot = StreamingMemoryAttributionStore.from_config(config).load()
    if snapshot is None:
        return None
    if expected_pid is None:
        return snapshot
    if snapshot.pid != expected_pid:
        return None
    current_boot_id = _read_boot_id()
    current_start_ticks = _read_process_start_ticks(expected_pid)
    if snapshot.boot_id and current_boot_id and snapshot.boot_id != current_boot_id:
        return None
    if snapshot.pid_start_ticks is not None and current_start_ticks is not None:
        if snapshot.pid_start_ticks != current_start_ticks:
            return None
    return snapshot


__all__ = [
    "ProcessMemoryMetrics",
    "StreamingMemoryAttributionStore",
    "StreamingMemoryPhase",
    "StreamingMemorySnapshot",
    "load_current_streaming_memory_snapshot",
    "record_streaming_memory_phase_best_effort",
    "record_streaming_memory_phase",
]
