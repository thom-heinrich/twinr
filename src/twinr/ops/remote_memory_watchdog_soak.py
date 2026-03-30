"""Record a bounded soak observation for the remote-memory watchdog.

Run this module during Pi acceptance to prove that the dedicated
`twinr-remote-memory-watchdog.service` keeps running, avoids new restarts, and
continues to write fresh successful watchdog snapshots over a multi-hour window.

Usage
-----
::

    PYTHONPATH=src python3 -m twinr.ops.remote_memory_watchdog_soak \
        --project-root /twinr \
        --duration-s 14400 \
        --interval-s 30
"""

# CHANGELOG: 2026-03-30
# BUG-1: Reject NaN/Inf CLI timing inputs; they previously allowed non-terminating
#        soaks or nonsensical freshness checks.
# BUG-2: A soak no longer passes on a single healthy sample; the final verdict now
#        requires a minimum sample count plus observed watchdog-artifact progress.
# BUG-3: systemctl polling now uses explicit timeouts and a non-catch-up schedule,
#        preventing bus stalls from hanging the soak or causing burst re-sampling.
# BUG-4: Reusing an output directory now resets prior run artifacts instead of
#        silently appending old samples into a new soak report.
# SEC-1: Artifact writes now use unique temp files, O_NOFOLLOW append opens,
#        fsync, and 0600/0700 permissions to block symlink clobbering in writable
#        artifact trees.
# IMP-1: Added signal-aware early termination with a final checkpointed summary and
#        explicit stop_reason / interrupted metadata.
# IMP-2: Added frontier observability metrics: collection latency, sample spacing,
#        artifact-progress evidence, filesystem freshness fallback, and reason-
#        coded sample failures.

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

DEFAULT_DURATION_S = 4.0 * 60.0 * 60.0
DEFAULT_INTERVAL_S = 30.0
DEFAULT_MAX_STALE_S = 180.0
DEFAULT_MIN_SAMPLES = 2
DEFAULT_SYSTEMCTL_TIMEOUT_S = 5.0
DEFAULT_SERVICE_NAME = "twinr-remote-memory-watchdog.service"


def _utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in a stable ISO-8601 form."""

    return _isoformat_utc(_utc_now())


def _isoformat_utc(value: datetime) -> str:
    """Serialize one datetime in UTC with second precision."""

    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_utc_iso(value: str | None) -> datetime | None:
    """Parse one persisted UTC-ish timestamp."""

    if not value:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_positive_float(value: object, *, default: float) -> float:
    """Clamp one CLI float input to a positive finite value."""

    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(normalized) or normalized <= 0.0:
        return default
    return normalized


def _normalize_positive_int(value: object, *, default: int) -> int:
    """Clamp one CLI integer input to a positive finite value."""

    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return default
    if normalized <= 0:
        return default
    return normalized


def _round_or_none(value: float | None) -> float | None:
    """Round one floating-point value when present."""

    if value is None:
        return None
    return round(value, 3)


def _percentile(values: list[float], percentile: float) -> float | None:
    """Compute one linear-interpolated percentile in [0.0, 1.0]."""

    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction)


def _numeric_distribution(prefix: str, values: list[float]) -> dict[str, float | None]:
    """Summarize one numeric series with frontier-friendly distribution stats."""

    return {
        f"{prefix}_min": _round_or_none(min(values) if values else None),
        f"{prefix}_p50": _round_or_none(_percentile(values, 0.50)),
        f"{prefix}_p95": _round_or_none(_percentile(values, 0.95)),
        f"{prefix}_p99": _round_or_none(_percentile(values, 0.99)),
        f"{prefix}_max": _round_or_none(max(values) if values else None),
    }


def _count_changes(values: list[object]) -> int:
    """Count value transitions across a sequence."""

    return sum(1 for previous, current in zip(values, values[1:]) if current != previous)


def _count_increases(values: list[int]) -> int:
    """Count strictly increasing steps across a sequence."""

    return sum(1 for previous, current in zip(values, values[1:]) if current > previous)


def _ensure_private_dir(path: Path) -> None:
    """Create one directory tree and tighten permissions when possible."""

    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise RuntimeError(f"Output path is not a directory: {path}")
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def _fsync_directory(path: Path) -> None:
    """Best-effort fsync of one directory for durable rename visibility."""

    try:
        dir_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def _write_text_secure_atomic(path: Path, content: str) -> None:
    """Write one text file atomically with durable flushes."""

    _ensure_private_dir(path.parent)
    fd = -1
    tmp_path: Path | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        tmp_path = Path(tmp_name)
        try:
            os.fchmod(fd, 0o600)
        except OSError:
            pass
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        _fsync_directory(path.parent)
    finally:
        if fd >= 0:
            os.close(fd)
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON file atomically."""

    _write_text_secure_atomic(
        path,
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )


def _open_text_append_nofollow(path: Path):
    """Open one append-only text file without following a final symlink."""

    _ensure_private_dir(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
    except OSError:
        pass
    return os.fdopen(fd, "a", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON line durably to a bounded artifact log."""

    with _open_text_append_nofollow(path) as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _default_snapshot_path(project_root: Path) -> Path:
    """Return the canonical watchdog artifact path for one project root."""

    return project_root / "artifacts" / "stores" / "ops" / "remote_memory_watchdog.json"


def _default_output_dir(project_root: Path, *, started_at: datetime) -> Path:
    """Return the default soak-report directory."""

    stamp = started_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return project_root / "artifacts" / "reports" / "remote_memory_watchdog_soak" / f"{stamp}-pid{os.getpid()}"


def _systemctl_show(service_name: str, *, timeout_s: float) -> dict[str, str]:
    """Load the minimal service state from `systemctl show`."""

    timeout_s = max(0.1, timeout_s)
    env = os.environ.copy()
    env.setdefault("SYSTEMD_PAGER", "")
    env.setdefault("SYSTEMD_COLORS", "0")
    env.setdefault("LC_ALL", "C")
    env.setdefault("LANG", "C")
    env.setdefault("SYSTEMD_BUS_TIMEOUT", f"{timeout_s:g}s")
    try:
        result = subprocess.run(
            [
                "systemctl",
                "show",
                "--no-pager",
                service_name,
                "-p",
                "ActiveState",
                "-p",
                "SubState",
                "-p",
                "ExecMainPID",
                "-p",
                "NRestarts",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_s + 0.5,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"systemctl show timed out after {timeout_s:g}s") from exc
    if result.returncode != 0:
        detail = " ".join((result.stderr or result.stdout or "").split()).strip() or "systemctl show failed"
        raise RuntimeError(detail)
    payload: dict[str, str] = {}
    for raw_line in result.stdout.splitlines():
        key, separator, value = raw_line.partition("=")
        if separator:
            payload[key] = value.strip()
    return payload


def _as_optional_int(value: object) -> int | None:
    """Convert one field to an int when possible."""

    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_optional_finite_float(value: object) -> float | None:
    """Convert one field to a finite float when possible."""

    try:
        parsed = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    if parsed is None or not math.isfinite(parsed):
        return None
    return parsed


@dataclass(frozen=True, slots=True)
class SystemdServiceState:
    """Capture the current systemd state for the watchdog service."""

    active_state: str
    sub_state: str
    exec_main_pid: int | None
    n_restarts: int | None

    @classmethod
    def from_systemctl_show(cls, payload: dict[str, str]) -> "SystemdServiceState":
        """Hydrate one service-state snapshot from `systemctl show` output."""

        return cls(
            active_state=str(payload.get("ActiveState", "") or ""),
            sub_state=str(payload.get("SubState", "") or ""),
            exec_main_pid=_as_optional_int(payload.get("ExecMainPID")),
            n_restarts=_as_optional_int(payload.get("NRestarts")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    @property
    def running(self) -> bool:
        """Report whether systemd still considers the unit healthy and running."""

        return self.active_state == "active" and self.sub_state == "running"


@dataclass(frozen=True, slots=True)
class WatchdogArtifactState:
    """Capture the persisted rolling watchdog snapshot state."""

    updated_at: str | None
    sample_count: int | None
    failure_count: int | None
    last_ok_at: str | None
    last_failure_at: str | None
    current_status: str | None
    current_ready: bool | None
    current_mode: str | None
    current_required: bool | None
    current_latency_ms: float | None
    current_consecutive_ok: int | None
    current_consecutive_fail: int | None
    current_captured_at: str | None
    file_mtime: str | None
    file_size_bytes: int | None

    @classmethod
    def from_json_payload(
        cls,
        payload: dict[str, Any],
        *,
        file_mtime: str | None,
        file_size_bytes: int | None,
    ) -> "WatchdogArtifactState":
        """Hydrate one artifact-state snapshot from the rolling JSON file."""

        current = payload.get("current")
        if not isinstance(current, dict):
            current = {}
        return cls(
            updated_at=str(payload.get("updated_at", "") or "") or None,
            sample_count=_as_optional_int(payload.get("sample_count")),
            failure_count=_as_optional_int(payload.get("failure_count")),
            last_ok_at=str(payload.get("last_ok_at", "") or "") or None,
            last_failure_at=str(payload.get("last_failure_at", "") or "") or None,
            current_status=str(current.get("status", "") or "") or None,
            current_ready=current.get("ready") if isinstance(current.get("ready"), bool) else None,
            current_mode=str(current.get("mode", "") or "") or None,
            current_required=current.get("required") if isinstance(current.get("required"), bool) else None,
            current_latency_ms=_as_optional_finite_float(current.get("latency_ms")),
            current_consecutive_ok=_as_optional_int(current.get("consecutive_ok")),
            current_consecutive_fail=_as_optional_int(current.get("consecutive_fail")),
            current_captured_at=str(current.get("captured_at", "") or "") or None,
            file_mtime=file_mtime,
            file_size_bytes=file_size_bytes,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class WatchdogSoakSample:
    """Capture one soak-observation sample."""

    observed_at: str
    elapsed_s: float
    service_running: bool
    watchdog_ok: bool
    artifact_fresh: bool
    stale_seconds: float | None
    freshness_source: str | None
    collection_latency_ms: float
    status_reason: str
    status_flags: list[str]
    service: SystemdServiceState | None
    artifact: WatchdogArtifactState | None
    service_error: str | None = None
    artifact_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        payload["service"] = self.service.to_dict() if self.service is not None else None
        payload["artifact"] = self.artifact.to_dict() if self.artifact is not None else None
        return payload


@dataclass(slots=True)
class StopRequest:
    """Represent an asynchronous request to stop the soak cleanly."""

    requested: bool = False
    signal_name: str | None = None

    def request(self, signum: int) -> None:
        """Register one stop request from a process signal."""

        if not self.requested:
            try:
                self.signal_name = signal.Signals(signum).name
            except ValueError:
                self.signal_name = f"SIG{signum}"
        self.requested = True

    @property
    def stop_reason(self) -> str | None:
        """Return the normalized stop reason when stopping was requested."""

        if not self.requested:
            return None
        if self.signal_name:
            return f"signal:{self.signal_name.lower()}"
        return "signal"


def _install_signal_handlers(stop_request: StopRequest) -> dict[int, Any]:
    """Install signal handlers that ask the soak loop to stop gracefully."""

    previous: dict[int, Any] = {}

    def _handler(signum: int, _frame: object) -> None:
        stop_request.request(signum)

    for candidate in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if candidate is None:
            continue
        try:
            previous[int(candidate)] = signal.getsignal(candidate)
            signal.signal(candidate, _handler)
        except (OSError, RuntimeError, ValueError):
            continue
    return previous


def _restore_signal_handlers(previous: dict[int, Any]) -> None:
    """Restore any signal handlers replaced by `_install_signal_handlers`."""

    for signum, handler in previous.items():
        try:
            signal.signal(signum, handler)
        except (OSError, RuntimeError, ValueError):
            continue


def load_watchdog_artifact_state(snapshot_path: Path) -> WatchdogArtifactState | None:
    """Load the persisted watchdog state when it exists."""

    try:
        with snapshot_path.open("r", encoding="utf-8") as handle:
            raw_payload = handle.read()
            file_stat = os.fstat(handle.fileno())
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError(f"Watchdog artifact is unreadable: {snapshot_path}") from exc
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Watchdog artifact is not valid JSON: {snapshot_path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Watchdog artifact has an invalid payload: {snapshot_path}")
    file_mtime = _isoformat_utc(datetime.fromtimestamp(file_stat.st_mtime, tz=timezone.utc))
    return WatchdogArtifactState.from_json_payload(
        payload,
        file_mtime=file_mtime,
        file_size_bytes=int(file_stat.st_size),
    )


def _artifact_freshness_reference(artifact_state: WatchdogArtifactState) -> tuple[datetime | None, str | None]:
    """Select the strongest freshness timestamp available in one artifact."""

    candidates = (
        ("updated_at", artifact_state.updated_at),
        ("current_captured_at", artifact_state.current_captured_at),
        ("file_mtime", artifact_state.file_mtime),
        ("last_ok_at", artifact_state.last_ok_at),
    )
    for source, raw_value in candidates:
        parsed = _parse_utc_iso(raw_value)
        if parsed is not None:
            return parsed, source
    return None, None


def _sample_status_flags(
    *,
    service_state: SystemdServiceState | None,
    artifact_state: WatchdogArtifactState | None,
    service_error: str | None,
    artifact_error: str | None,
    service_running: bool,
    watchdog_ok: bool,
    artifact_fresh: bool,
) -> list[str]:
    """Derive reason-coded status flags for one sample."""

    flags: list[str] = []
    if service_error:
        flags.append("service_error")
    elif service_state is None:
        flags.append("service_missing")
    elif not service_running:
        flags.append("service_not_running")

    if artifact_error:
        flags.append("artifact_error")
    elif artifact_state is None:
        flags.append("artifact_missing")
    else:
        if not watchdog_ok:
            flags.append("watchdog_not_ok")
        if not artifact_fresh:
            flags.append("artifact_stale")

    if not flags:
        flags.append("ok")
    return flags


def collect_soak_sample(
    *,
    service_name: str,
    snapshot_path: Path,
    started_monotonic: float,
    max_stale_s: float,
    systemctl_timeout_s: float,
    observed_at: datetime | None = None,
    observed_monotonic: float | None = None,
) -> WatchdogSoakSample:
    """Collect one combined service-plus-artifact soak sample."""

    observed = observed_at or _utc_now()
    observed_monotonic = observed_monotonic if observed_monotonic is not None else time.monotonic()
    collection_started_monotonic = observed_monotonic
    service_state: SystemdServiceState | None = None
    artifact_state: WatchdogArtifactState | None = None
    service_error: str | None = None
    artifact_error: str | None = None

    try:
        service_state = SystemdServiceState.from_systemctl_show(
            _systemctl_show(service_name, timeout_s=systemctl_timeout_s)
        )
    except Exception as exc:
        service_error = " ".join(str(exc).split()).strip() or type(exc).__name__

    try:
        artifact_state = load_watchdog_artifact_state(snapshot_path)
    except Exception as exc:
        artifact_error = " ".join(str(exc).split()).strip() or type(exc).__name__

    stale_seconds: float | None = None
    freshness_source: str | None = None
    if artifact_state is not None:
        freshness_reference, freshness_source = _artifact_freshness_reference(artifact_state)
        if freshness_reference is not None:
            stale_seconds = round(max(0.0, (observed - freshness_reference).total_seconds()), 3)

    service_running = service_state.running if service_state is not None else False
    watchdog_ok = bool(
        artifact_state is not None
        and artifact_state.current_status == "ok"
        and artifact_state.current_ready is True
    )
    artifact_fresh = stale_seconds is not None and stale_seconds <= max_stale_s
    status_flags = _sample_status_flags(
        service_state=service_state,
        artifact_state=artifact_state,
        service_error=service_error,
        artifact_error=artifact_error,
        service_running=service_running,
        watchdog_ok=watchdog_ok,
        artifact_fresh=artifact_fresh,
    )

    return WatchdogSoakSample(
        observed_at=_isoformat_utc(observed),
        elapsed_s=round(max(0.0, observed_monotonic - started_monotonic), 3),
        service_running=service_running,
        watchdog_ok=watchdog_ok,
        artifact_fresh=artifact_fresh,
        stale_seconds=stale_seconds,
        freshness_source=freshness_source,
        collection_latency_ms=round(max(0.0, (time.monotonic() - collection_started_monotonic) * 1000.0), 3),
        status_reason=status_flags[0],
        status_flags=status_flags,
        service=service_state,
        artifact=artifact_state,
        service_error=service_error,
        artifact_error=artifact_error,
    )


def _artifact_progress_metrics(samples: list[WatchdogSoakSample]) -> dict[str, Any]:
    """Compute artifact-progression evidence across one soak run."""

    updated_at_values = [sample.artifact.updated_at for sample in samples if sample.artifact and sample.artifact.updated_at]
    current_captured_values = [
        sample.artifact.current_captured_at
        for sample in samples
        if sample.artifact and sample.artifact.current_captured_at
    ]
    file_mtime_values = [sample.artifact.file_mtime for sample in samples if sample.artifact and sample.artifact.file_mtime]
    last_ok_values = [sample.artifact.last_ok_at for sample in samples if sample.artifact and sample.artifact.last_ok_at]
    sample_count_values = [sample.artifact.sample_count for sample in samples if sample.artifact and sample.artifact.sample_count is not None]

    updated_at_change_count = _count_changes(updated_at_values)
    current_captured_at_change_count = _count_changes(current_captured_values)
    file_mtime_change_count = _count_changes(file_mtime_values)
    last_ok_at_change_count = _count_changes(last_ok_values)
    sample_count_increase_count = _count_increases([int(value) for value in sample_count_values])

    artifact_progress_observed = any(
        (
            updated_at_change_count > 0,
            current_captured_at_change_count > 0,
            file_mtime_change_count > 0,
            last_ok_at_change_count > 0,
            sample_count_increase_count > 0,
        )
    )

    return {
        "artifact_updated_at_change_count": updated_at_change_count,
        "artifact_current_captured_at_change_count": current_captured_at_change_count,
        "artifact_file_mtime_change_count": file_mtime_change_count,
        "artifact_last_ok_at_change_count": last_ok_at_change_count,
        "artifact_sample_count_increase_count": sample_count_increase_count,
        "artifact_progress_observed": artifact_progress_observed,
    }


def build_soak_summary(
    samples: list[WatchdogSoakSample],
    *,
    started_at: str,
    ended_at: str,
    requested_duration_s: float,
    observed_duration_s: float,
    interval_s: float,
    max_stale_s: float,
    min_samples: int,
    require_artifact_progress: bool,
    service_name: str,
    snapshot_path: Path,
    output_dir: Path,
    stop_reason: str,
    interrupted: bool,
    termination_signal: str | None,
    systemctl_timeout_s: float,
) -> dict[str, Any]:
    """Summarize the recorded soak run into one operator-facing JSON payload."""

    first_service = next((sample.service for sample in samples if sample.service is not None), None)
    final_service = next((sample.service for sample in reversed(samples) if sample.service is not None), None)
    first_artifact = next((sample.artifact for sample in samples if sample.artifact is not None), None)
    final_artifact = next((sample.artifact for sample in reversed(samples) if sample.artifact is not None), None)

    pid_values = [
        sample.service.exec_main_pid
        for sample in samples
        if sample.service is not None and sample.service.exec_main_pid is not None
    ]
    pid_change_count = _count_changes(pid_values)

    restart_delta = None
    if first_service is not None and final_service is not None:
        if first_service.n_restarts is not None and final_service.n_restarts is not None:
            restart_delta = final_service.n_restarts - first_service.n_restarts

    failure_count_delta = None
    if first_artifact is not None and final_artifact is not None:
        if first_artifact.failure_count is not None and final_artifact.failure_count is not None:
            failure_count_delta = final_artifact.failure_count - first_artifact.failure_count

    sample_count_delta = None
    if first_artifact is not None and final_artifact is not None:
        if first_artifact.sample_count is not None and final_artifact.sample_count is not None:
            sample_count_delta = final_artifact.sample_count - first_artifact.sample_count

    stale_values = [sample.stale_seconds for sample in samples if sample.stale_seconds is not None]
    latency_values = [sample.collection_latency_ms for sample in samples]
    watchdog_latency_values = [
        sample.artifact.current_latency_ms
        for sample in samples
        if sample.artifact is not None and sample.artifact.current_latency_ms is not None
    ]
    sample_spacing_values = [
        round(max(0.0, current.elapsed_s - previous.elapsed_s), 3)
        for previous, current in zip(samples, samples[1:])
    ]

    non_running_sample_count = sum(1 for sample in samples if not sample.service_running)
    non_ok_sample_count = sum(1 for sample in samples if not sample.watchdog_ok)
    stale_sample_count = sum(1 for sample in samples if not sample.artifact_fresh)
    service_error_count = sum(1 for sample in samples if sample.service_error)
    artifact_error_count = sum(1 for sample in samples if sample.artifact_error)

    progress_metrics = _artifact_progress_metrics(samples)
    min_samples_ok = len(samples) >= min_samples
    artifact_progress_ok = (not require_artifact_progress) or bool(progress_metrics["artifact_progress_observed"])

    reason_counts = dict(sorted(Counter(sample.status_reason for sample in samples).items()))
    flag_counts = dict(sorted(Counter(flag for sample in samples for flag in sample.status_flags).items()))

    checks_passed = all(
        (
            bool(samples),
            not interrupted,
            stop_reason == "deadline_reached",
            min_samples_ok,
            artifact_progress_ok,
            non_running_sample_count == 0,
            non_ok_sample_count == 0,
            stale_sample_count == 0,
            service_error_count == 0,
            artifact_error_count == 0,
            restart_delta == 0,
            pid_change_count == 0,
            failure_count_delta in (None, 0),
            sample_count_delta is None or sample_count_delta >= 0,
        )
    )

    payload: dict[str, Any] = {
        "run_id": output_dir.name,
        "started_at": started_at,
        "ended_at": ended_at,
        "requested_duration_s": round(requested_duration_s, 3),
        "observed_duration_s": round(observed_duration_s, 3),
        "interval_s": round(interval_s, 3),
        "max_stale_s": round(max_stale_s, 3),
        "systemctl_timeout_s": round(systemctl_timeout_s, 3),
        "min_samples_required": min_samples,
        "require_artifact_progress": require_artifact_progress,
        "min_samples_ok": min_samples_ok,
        "service_name": service_name,
        "snapshot_path": str(snapshot_path),
        "output_dir": str(output_dir),
        "sample_total": len(samples),
        "all_checks_passed": checks_passed,
        "stop_reason": stop_reason,
        "interrupted": interrupted,
        "termination_signal": termination_signal,
        "non_running_sample_count": non_running_sample_count,
        "non_ok_sample_count": non_ok_sample_count,
        "stale_sample_count": stale_sample_count,
        "service_error_count": service_error_count,
        "artifact_error_count": artifact_error_count,
        "status_reason_counts": reason_counts,
        "status_flag_counts": flag_counts,
        "baseline_exec_main_pid": first_service.exec_main_pid if first_service is not None else None,
        "final_exec_main_pid": final_service.exec_main_pid if final_service is not None else None,
        "exec_main_pid_change_count": pid_change_count,
        "baseline_restart_count": first_service.n_restarts if first_service is not None else None,
        "final_restart_count": final_service.n_restarts if final_service is not None else None,
        "restart_delta": restart_delta,
        "baseline_sample_count": first_artifact.sample_count if first_artifact is not None else None,
        "final_sample_count": final_artifact.sample_count if final_artifact is not None else None,
        "sample_count_delta": sample_count_delta,
        "baseline_failure_count": first_artifact.failure_count if first_artifact is not None else None,
        "final_failure_count": final_artifact.failure_count if final_artifact is not None else None,
        "failure_count_delta": failure_count_delta,
        "last_ok_at": final_artifact.last_ok_at if final_artifact is not None else None,
        "last_failure_at": final_artifact.last_failure_at if final_artifact is not None else None,
        "freshness_source_final": samples[-1].freshness_source if samples else None,
        "final_artifact_file_mtime": final_artifact.file_mtime if final_artifact is not None else None,
        "final_artifact_file_size_bytes": final_artifact.file_size_bytes if final_artifact is not None else None,
    }
    payload.update(progress_metrics)
    payload.update(_numeric_distribution("sample_stale_s", [float(value) for value in stale_values]))
    payload.update(_numeric_distribution("collection_latency_ms", [float(value) for value in latency_values]))
    payload.update(_numeric_distribution("watchdog_latency_ms", [float(value) for value in watchdog_latency_values]))
    payload.update(_numeric_distribution("sample_spacing_s", [float(value) for value in sample_spacing_values]))
    return payload


def _status_payload(
    samples: list[WatchdogSoakSample],
    *,
    complete: bool,
    started_at: str,
    ended_at: str,
    requested_duration_s: float,
    observed_duration_s: float,
    interval_s: float,
    max_stale_s: float,
    min_samples: int,
    require_artifact_progress: bool,
    service_name: str,
    snapshot_path: Path,
    output_dir: Path,
    stop_reason: str,
    interrupted: bool,
    termination_signal: str | None,
    systemctl_timeout_s: float,
) -> dict[str, Any]:
    """Build one status payload for both progress and final summary files."""

    payload = build_soak_summary(
        samples,
        started_at=started_at,
        ended_at=ended_at,
        requested_duration_s=requested_duration_s,
        observed_duration_s=observed_duration_s,
        interval_s=interval_s,
        max_stale_s=max_stale_s,
        min_samples=min_samples,
        require_artifact_progress=require_artifact_progress,
        service_name=service_name,
        snapshot_path=snapshot_path,
        output_dir=output_dir,
        stop_reason=stop_reason,
        interrupted=interrupted,
        termination_signal=termination_signal,
        systemctl_timeout_s=systemctl_timeout_s,
    )
    payload["complete"] = complete
    payload["last_observed_at"] = samples[-1].observed_at if samples else None
    payload["samples_path"] = str(output_dir / "samples.jsonl")
    return payload


def _reset_run_artifacts(output_dir: Path) -> None:
    """Reset run-local artifacts so one rerun cannot mix with old samples."""

    # BREAKING: Re-running into an existing output directory now starts a fresh
    # soak report instead of appending into a previous run's samples.jsonl.

    _ensure_private_dir(output_dir)
    for name in ("samples.jsonl", "status.json", "summary.json"):
        path = output_dir / name
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeError(f"Unable to reset old soak artifact: {path}") from exc
    _fsync_directory(output_dir)


def _emit_progress(sample: WatchdogSoakSample) -> None:
    """Emit one compact progress line to stderr for human operators."""

    progress = {
        "observed_at": sample.observed_at,
        "elapsed_s": sample.elapsed_s,
        "status_reason": sample.status_reason,
        "service_running": sample.service_running,
        "watchdog_ok": sample.watchdog_ok,
        "artifact_fresh": sample.artifact_fresh,
        "stale_seconds": sample.stale_seconds,
        "freshness_source": sample.freshness_source,
        "collection_latency_ms": sample.collection_latency_ms,
    }
    print(json.dumps(progress, ensure_ascii=True, sort_keys=True), file=sys.stderr, flush=True)


def run_soak(
    *,
    project_root: Path,
    service_name: str,
    snapshot_path: Path,
    output_dir: Path,
    duration_s: float,
    interval_s: float,
    max_stale_s: float,
    min_samples: int,
    require_artifact_progress: bool,
    systemctl_timeout_s: float,
    print_progress: bool,
) -> int:
    """Run one bounded soak observation and persist rolling artifacts."""

    started_dt = _utc_now()
    started_at = _isoformat_utc(started_dt)
    started_monotonic = time.monotonic()
    deadline_monotonic = started_monotonic + duration_s

    _reset_run_artifacts(output_dir)
    samples_path = output_dir / "samples.jsonl"
    status_path = output_dir / "status.json"
    samples: list[WatchdogSoakSample] = []

    stop_request = StopRequest()
    previous_signal_handlers = _install_signal_handlers(stop_request)
    try:
        next_tick = started_monotonic
        stop_reason = "deadline_reached"
        termination_signal: str | None = None

        while True:
            now_monotonic = time.monotonic()
            if stop_request.requested:
                next_tick = min(next_tick, now_monotonic)
            if now_monotonic < next_tick:
                time.sleep(min(0.5, next_tick - now_monotonic))
                continue

            observed_monotonic = time.monotonic()
            sample = collect_soak_sample(
                service_name=service_name,
                snapshot_path=snapshot_path,
                started_monotonic=started_monotonic,
                max_stale_s=max_stale_s,
                systemctl_timeout_s=systemctl_timeout_s,
                observed_at=_utc_now(),
                observed_monotonic=observed_monotonic,
            )
            samples.append(sample)
            _append_jsonl(samples_path, sample.to_dict())
            if print_progress:
                _emit_progress(sample)

            _atomic_write_json(
                status_path,
                _status_payload(
                    samples,
                    complete=False,
                    started_at=started_at,
                    ended_at=sample.observed_at,
                    requested_duration_s=duration_s,
                    observed_duration_s=sample.elapsed_s,
                    interval_s=interval_s,
                    max_stale_s=max_stale_s,
                    min_samples=min_samples,
                    require_artifact_progress=require_artifact_progress,
                    service_name=service_name,
                    snapshot_path=snapshot_path,
                    output_dir=output_dir,
                    stop_reason=stop_request.stop_reason or "running",
                    interrupted=stop_request.requested,
                    termination_signal=stop_request.signal_name,
                    systemctl_timeout_s=systemctl_timeout_s,
                ),
            )

            if stop_request.requested:
                stop_reason = stop_request.stop_reason or "signal"
                termination_signal = stop_request.signal_name
                break
            if time.monotonic() >= deadline_monotonic:
                stop_reason = "deadline_reached"
                break

            next_tick = time.monotonic() + interval_s

    finally:
        _restore_signal_handlers(previous_signal_handlers)

    final_observed_duration_s = samples[-1].elapsed_s if samples else 0.0
    summary_payload = _status_payload(
        samples,
        complete=True,
        started_at=started_at,
        ended_at=samples[-1].observed_at if samples else _utc_now_iso(),
        requested_duration_s=duration_s,
        observed_duration_s=final_observed_duration_s,
        interval_s=interval_s,
        max_stale_s=max_stale_s,
        min_samples=min_samples,
        require_artifact_progress=require_artifact_progress,
        service_name=service_name,
        snapshot_path=snapshot_path,
        output_dir=output_dir,
        stop_reason=stop_reason,
        interrupted=stop_reason != "deadline_reached",
        termination_signal=termination_signal,
        systemctl_timeout_s=systemctl_timeout_s,
    )
    summary_payload["project_root"] = str(project_root)
    _atomic_write_json(status_path, summary_payload)
    _atomic_write_json(output_dir / "summary.json", summary_payload)
    print(json.dumps(summary_payload, ensure_ascii=True, sort_keys=True))
    return 0 if summary_payload["all_checks_passed"] else 1


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", help="Twinr project root that owns the watchdog artifacts.")
    parser.add_argument(
        "--service-name",
        default=DEFAULT_SERVICE_NAME,
        help="Systemd service to observe.",
    )
    parser.add_argument(
        "--snapshot-path",
        default=None,
        help="Override the remote-memory watchdog snapshot path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the soak report. Defaults under artifacts/reports/remote_memory_watchdog_soak.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=DEFAULT_DURATION_S,
        help="Total soak duration in seconds.",
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=DEFAULT_INTERVAL_S,
        help="Sampling interval in seconds.",
    )
    parser.add_argument(
        "--max-stale-s",
        type=float,
        default=DEFAULT_MAX_STALE_S,
        help="Maximum allowed age of the saved watchdog snapshot before it counts as drift.",
    )
    parser.add_argument(
        "--systemctl-timeout-s",
        type=float,
        default=DEFAULT_SYSTEMCTL_TIMEOUT_S,
        help="Timeout for each systemctl poll in seconds.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help="Minimum number of collected samples required for a passing soak.",
    )
    parser.add_argument(
        "--require-artifact-progress",
        dest="require_artifact_progress",
        action="store_true",
        default=True,
        help="Require observed watchdog-artifact progression (timestamps, counts, or file mtime).",
    )
    parser.add_argument(
        "--no-require-artifact-progress",
        dest="require_artifact_progress",
        action="store_false",
        help="Allow a soak to pass without observed artifact progression.",
    )
    parser.add_argument(
        "--print-progress",
        action="store_true",
        help="Emit one compact progress JSON line per sample to stderr.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the bounded soak recorder from the command line."""

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).expanduser().resolve(strict=False)
    duration_s = _normalize_positive_float(args.duration_s, default=DEFAULT_DURATION_S)
    interval_s = _normalize_positive_float(args.interval_s, default=DEFAULT_INTERVAL_S)
    max_stale_s = _normalize_positive_float(args.max_stale_s, default=DEFAULT_MAX_STALE_S)
    systemctl_timeout_s = _normalize_positive_float(
        args.systemctl_timeout_s,
        default=DEFAULT_SYSTEMCTL_TIMEOUT_S,
    )
    min_samples = _normalize_positive_int(args.min_samples, default=DEFAULT_MIN_SAMPLES)

    snapshot_path = (
        Path(args.snapshot_path).expanduser().resolve(strict=False)
        if args.snapshot_path
        else _default_snapshot_path(project_root)
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve(strict=False)
        if args.output_dir
        else _default_output_dir(project_root, started_at=_utc_now())
    )

    # BREAKING: A soak now fails unless it captures at least `--min-samples`
    # samples and, by default, observes actual artifact progression during the run.
    return run_soak(
        project_root=project_root,
        service_name=str(args.service_name),
        snapshot_path=snapshot_path,
        output_dir=output_dir,
        duration_s=duration_s,
        interval_s=interval_s,
        max_stale_s=max_stale_s,
        min_samples=min_samples,
        require_artifact_progress=bool(args.require_artifact_progress),
        systemctl_timeout_s=systemctl_timeout_s,
        print_progress=bool(args.print_progress),
    )


if __name__ == "__main__":
    raise SystemExit(main())