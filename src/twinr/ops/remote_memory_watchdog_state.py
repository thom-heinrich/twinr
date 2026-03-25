"""Persisted state models and storage helpers for the remote-memory watchdog."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket

from twinr.agent.base_agent.config import TwinrConfig
from twinr.ops.events import compact_text
from twinr.ops.paths import resolve_ops_paths_for_config


DEFAULT_WATCHDOG_INTERVAL_S = 1.0
DEFAULT_HISTORY_LIMIT = 3600
SNAPSHOT_SCHEMA_VERSION = 1
PERSISTED_RECENT_SAMPLE_LIMIT = 64
STARTING_SAMPLE_DETAIL = "Remote memory watchdog is starting."


def _coerce_float_value(value: object, *, default: float) -> float:
    """Return one finite-ish float from JSON-like payload values."""

    if isinstance(value, bool):
        return default
    if not isinstance(value, (int, float, str)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int_value(value: object, *, default: int) -> int:
    """Return one integer from JSON-like payload values."""

    if isinstance(value, bool):
        return default
    if not isinstance(value, (int, float, str)):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_object_dict(value: object) -> dict[str, object] | None:
    """Copy one generic mapping into a JSON-serializable object dict."""

    if not isinstance(value, dict):
        return None
    return {str(key): item for key, item in value.items()}


def utc_now_iso() -> str:
    """Return the current UTC timestamp in a stable ISO-8601 form."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def coerce_interval_s(value: object, *, default: float) -> float:
    """Normalize the watchdog interval to a finite positive float."""

    interval_s = _coerce_float_value(value, default=default)
    if interval_s <= 0.0:
        return default
    return interval_s


def coerce_history_limit(value: object, *, default: int) -> int:
    """Normalize the rolling sample-history length."""

    history_limit = _coerce_int_value(value, default=default)
    if history_limit <= 0:
        return default
    return history_limit


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one JSON payload atomically to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode("utf-8")
    file_mode = 0o644
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_CLOEXEC", 0), file_mode)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
        os.fchmod(fd, file_mode)
    finally:
        os.close(fd)
    os.replace(tmp_path, path)
    os.chmod(path, file_mode)


@dataclass(frozen=True, slots=True)
class RemoteMemoryWatchdogSample:
    """Capture one remote-memory probe result."""

    seq: int
    captured_at: str
    status: str
    ready: bool
    mode: str
    required: bool
    latency_ms: float
    consecutive_ok: int
    consecutive_fail: int
    detail: str | None = None
    probe: dict[str, object] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RemoteMemoryWatchdogSample":
        """Hydrate one sample from persisted JSON data."""

        probe_payload = _coerce_object_dict(payload.get("probe"))
        return cls(
            seq=_coerce_int_value(payload.get("seq", 0) or 0, default=0),
            captured_at=str(payload.get("captured_at", "") or ""),
            status=str(payload.get("status", "unknown") or "unknown"),
            ready=bool(payload.get("ready", False)),
            mode=str(payload.get("mode", "unknown") or "unknown"),
            required=bool(payload.get("required", False)),
            latency_ms=_coerce_float_value(payload.get("latency_ms", 0.0) or 0.0, default=0.0),
            consecutive_ok=_coerce_int_value(payload.get("consecutive_ok", 0) or 0, default=0),
            consecutive_fail=_coerce_int_value(payload.get("consecutive_fail", 0) or 0, default=0),
            detail=cls._coerce_optional_text(payload.get("detail")),
            probe=probe_payload,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    @staticmethod
    def _coerce_optional_text(value: object) -> str | None:
        text = compact_text(str(value or "").strip(), limit=240)
        return text or None


def compact_history_sample(sample: RemoteMemoryWatchdogSample) -> RemoteMemoryWatchdogSample:
    """Drop heavy nested probe payloads from persisted history samples."""

    if sample.probe is None:
        return sample
    return replace(sample, probe=None)


@dataclass(frozen=True, slots=True)
class RemoteMemoryWatchdogSnapshot:
    """Persisted rolling watchdog state."""

    schema_version: int
    started_at: str
    updated_at: str
    hostname: str
    pid: int
    interval_s: float
    history_limit: int
    sample_count: int
    failure_count: int
    last_ok_at: str | None
    last_failure_at: str | None
    artifact_path: str
    current: RemoteMemoryWatchdogSample
    recent_samples: tuple[RemoteMemoryWatchdogSample, ...]
    heartbeat_at: str | None = None
    probe_inflight: bool = False
    probe_started_at: str | None = None
    probe_age_s: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RemoteMemoryWatchdogSnapshot":
        """Hydrate one persisted snapshot from JSON data."""

        current_payload = payload.get("current")
        if not isinstance(current_payload, dict):
            raise ValueError("Remote memory watchdog snapshot is missing `current`.")
        raw_recent_samples = payload.get("recent_samples")
        if raw_recent_samples is None:
            raw_recent_samples = ()
        if not isinstance(raw_recent_samples, list):
            raise ValueError("Remote memory watchdog snapshot has malformed `recent_samples`.")
        return cls(
            schema_version=_coerce_int_value(payload.get("schema_version", 0) or 0, default=0),
            started_at=str(payload.get("started_at", "") or ""),
            updated_at=str(payload.get("updated_at", "") or ""),
            hostname=str(payload.get("hostname", "") or ""),
            pid=_coerce_int_value(payload.get("pid", 0) or 0, default=0),
            interval_s=_coerce_float_value(payload.get("interval_s", 0.0) or 0.0, default=0.0),
            history_limit=_coerce_int_value(payload.get("history_limit", 0) or 0, default=0),
            sample_count=_coerce_int_value(payload.get("sample_count", 0) or 0, default=0),
            failure_count=_coerce_int_value(payload.get("failure_count", 0) or 0, default=0),
            last_ok_at=RemoteMemoryWatchdogSample._coerce_optional_text(payload.get("last_ok_at")),
            last_failure_at=RemoteMemoryWatchdogSample._coerce_optional_text(payload.get("last_failure_at")),
            artifact_path=str(payload.get("artifact_path", "") or ""),
            current=RemoteMemoryWatchdogSample.from_dict(current_payload),
            recent_samples=tuple(
                RemoteMemoryWatchdogSample.from_dict(item)
                for item in raw_recent_samples
                if isinstance(item, dict)
            ),
            heartbeat_at=RemoteMemoryWatchdogSample._coerce_optional_text(payload.get("heartbeat_at")),
            probe_inflight=bool(payload.get("probe_inflight", False)),
            probe_started_at=RemoteMemoryWatchdogSample._coerce_optional_text(payload.get("probe_started_at")),
            probe_age_s=cls._coerce_optional_float(payload.get("probe_age_s")),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        payload["current"] = self.current.to_dict()
        payload["recent_samples"] = [sample.to_dict() for sample in self.recent_samples]
        return payload

    @staticmethod
    def _coerce_optional_float(value: object) -> float | None:
        parsed = _coerce_float_value(value, default=-1.0)
        if not parsed >= 0.0:
            return None
        return parsed


class RemoteMemoryWatchdogStore:
    """Persist the current rolling watchdog snapshot to disk."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "RemoteMemoryWatchdogStore":
        """Build the canonical watchdog-state store for one Twinr config."""

        ops_paths = resolve_ops_paths_for_config(config)
        return cls(ops_paths.ops_store_root / "remote_memory_watchdog.json")

    def save(self, snapshot: RemoteMemoryWatchdogSnapshot) -> None:
        """Persist the latest rolling snapshot atomically."""

        atomic_write_json(self.path, snapshot.to_dict())

    def load(self) -> RemoteMemoryWatchdogSnapshot | None:
        """Load the persisted rolling snapshot when it exists."""

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Remote memory watchdog snapshot is not valid JSON: {self.path}") from exc
        except OSError as exc:
            raise RuntimeError(f"Remote memory watchdog snapshot is unreadable: {self.path}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Remote memory watchdog snapshot has an invalid top-level payload: {self.path}")
        try:
            return RemoteMemoryWatchdogSnapshot.from_dict(payload)
        except ValueError as exc:
            raise RuntimeError(f"Remote memory watchdog snapshot is malformed: {self.path}") from exc


def build_starting_sample(config: TwinrConfig, *, captured_at: str) -> RemoteMemoryWatchdogSample:
    """Return the synthetic startup sample used before the first deep probe."""

    required = bool(
        config.long_term_memory_enabled
        and config.long_term_memory_mode == "remote_primary"
        and config.long_term_memory_remote_required
    )
    return RemoteMemoryWatchdogSample(
        seq=0,
        captured_at=captured_at,
        status="starting",
        ready=False,
        mode="remote_primary" if required else "disabled",
        required=required,
        latency_ms=0.0,
        consecutive_ok=0,
        consecutive_fail=0,
        detail=STARTING_SAMPLE_DETAIL,
    )


def build_remote_memory_watchdog_bootstrap_snapshot(
    config: TwinrConfig,
    *,
    pid: int,
    artifact_path: str | Path,
    started_at: str | None = None,
    captured_at: str | None = None,
) -> RemoteMemoryWatchdogSnapshot:
    """Build the supervisor-seeded startup snapshot for a fresh watchdog child."""

    resolved_started_at = str(started_at or utc_now_iso())
    resolved_captured_at = str(captured_at or resolved_started_at)
    interval_s = coerce_interval_s(
        getattr(config, "long_term_memory_remote_watchdog_interval_s", DEFAULT_WATCHDOG_INTERVAL_S),
        default=DEFAULT_WATCHDOG_INTERVAL_S,
    )
    history_limit = coerce_history_limit(
        getattr(config, "long_term_memory_remote_watchdog_history_limit", DEFAULT_HISTORY_LIMIT),
        default=DEFAULT_HISTORY_LIMIT,
    )
    return RemoteMemoryWatchdogSnapshot(
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        started_at=resolved_started_at,
        updated_at=resolved_captured_at,
        hostname=socket.gethostname(),
        pid=int(pid),
        interval_s=interval_s,
        history_limit=history_limit,
        sample_count=0,
        failure_count=0,
        last_ok_at=None,
        last_failure_at=None,
        artifact_path=str(artifact_path),
        current=build_starting_sample(config, captured_at=resolved_captured_at),
        recent_samples=(),
        heartbeat_at=resolved_captured_at,
        probe_inflight=True,
        probe_started_at=resolved_captured_at,
        probe_age_s=0.0,
    )


__all__ = [
    "DEFAULT_HISTORY_LIMIT",
    "DEFAULT_WATCHDOG_INTERVAL_S",
    "PERSISTED_RECENT_SAMPLE_LIMIT",
    "RemoteMemoryWatchdogSample",
    "RemoteMemoryWatchdogSnapshot",
    "RemoteMemoryWatchdogStore",
    "SNAPSHOT_SCHEMA_VERSION",
    "STARTING_SAMPLE_DETAIL",
    "atomic_write_json",
    "build_remote_memory_watchdog_bootstrap_snapshot",
    "build_starting_sample",
    "coerce_history_limit",
    "coerce_interval_s",
    "compact_history_sample",
    "utc_now_iso",
]
