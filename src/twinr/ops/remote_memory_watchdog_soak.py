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

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from typing import Any


DEFAULT_DURATION_S = 4.0 * 60.0 * 60.0
DEFAULT_INTERVAL_S = 30.0
DEFAULT_MAX_STALE_S = 180.0
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
    if normalized <= 0.0:
        return default
    return normalized


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON file atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON line to a bounded artifact log."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def _default_snapshot_path(project_root: Path) -> Path:
    """Return the canonical watchdog artifact path for one project root."""

    return project_root / "artifacts" / "stores" / "ops" / "remote_memory_watchdog.json"


def _default_output_dir(project_root: Path, *, started_at: datetime) -> Path:
    """Return the default soak-report directory."""

    stamp = started_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return project_root / "artifacts" / "reports" / "remote_memory_watchdog_soak" / stamp


def _systemctl_show(service_name: str) -> dict[str, str]:
    """Load the minimal service state from `systemctl show`."""

    result = subprocess.run(
        [
            "systemctl",
            "show",
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
    )
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

    @classmethod
    def from_json_payload(cls, payload: dict[str, Any]) -> "WatchdogArtifactState":
        """Hydrate one artifact-state snapshot from the rolling JSON file."""

        current = payload.get("current")
        if not isinstance(current, dict):
            current = {}
        latency_ms = current.get("latency_ms")
        try:
            parsed_latency = float(latency_ms) if latency_ms is not None else None
        except (TypeError, ValueError):
            parsed_latency = None
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
            current_latency_ms=parsed_latency,
            current_consecutive_ok=_as_optional_int(current.get("consecutive_ok")),
            current_consecutive_fail=_as_optional_int(current.get("consecutive_fail")),
            current_captured_at=str(current.get("captured_at", "") or "") or None,
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


def load_watchdog_artifact_state(snapshot_path: Path) -> WatchdogArtifactState | None:
    """Load the persisted watchdog state when it exists."""

    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Watchdog artifact is not valid JSON: {snapshot_path}") from exc
    except OSError as exc:
        raise RuntimeError(f"Watchdog artifact is unreadable: {snapshot_path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Watchdog artifact has an invalid payload: {snapshot_path}")
    return WatchdogArtifactState.from_json_payload(payload)


def collect_soak_sample(
    *,
    service_name: str,
    snapshot_path: Path,
    started_at: datetime,
    max_stale_s: float,
    observed_at: datetime | None = None,
) -> WatchdogSoakSample:
    """Collect one combined service-plus-artifact soak sample."""

    observed = observed_at or _utc_now()
    service_state: SystemdServiceState | None = None
    artifact_state: WatchdogArtifactState | None = None
    service_error: str | None = None
    artifact_error: str | None = None

    try:
        service_state = SystemdServiceState.from_systemctl_show(_systemctl_show(service_name))
    except Exception as exc:
        service_error = " ".join(str(exc).split()).strip() or type(exc).__name__

    try:
        artifact_state = load_watchdog_artifact_state(snapshot_path)
    except Exception as exc:
        artifact_error = " ".join(str(exc).split()).strip() or type(exc).__name__

    stale_seconds: float | None = None
    if artifact_state is not None:
        updated_at = _parse_utc_iso(artifact_state.updated_at)
        if updated_at is not None:
            stale_seconds = round(max(0.0, (observed - updated_at).total_seconds()), 3)

    service_running = service_state.running if service_state is not None else False
    watchdog_ok = bool(
        artifact_state is not None
        and artifact_state.current_status == "ok"
        and artifact_state.current_ready is True
    )
    artifact_fresh = stale_seconds is not None and stale_seconds <= max_stale_s

    return WatchdogSoakSample(
        observed_at=_isoformat_utc(observed),
        elapsed_s=round(max(0.0, (observed - started_at).total_seconds()), 3),
        service_running=service_running,
        watchdog_ok=watchdog_ok,
        artifact_fresh=artifact_fresh,
        stale_seconds=stale_seconds,
        service=service_state,
        artifact=artifact_state,
        service_error=service_error,
        artifact_error=artifact_error,
    )


def build_soak_summary(
    samples: list[WatchdogSoakSample],
    *,
    started_at: str,
    ended_at: str,
    duration_s: float,
    interval_s: float,
    max_stale_s: float,
    service_name: str,
    snapshot_path: Path,
    output_dir: Path,
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
    pid_change_count = sum(1 for previous, current in zip(pid_values, pid_values[1:]) if current != previous)

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
    latency_values = [
        sample.artifact.current_latency_ms
        for sample in samples
        if sample.artifact is not None and sample.artifact.current_latency_ms is not None
    ]

    non_running_sample_count = sum(1 for sample in samples if not sample.service_running)
    non_ok_sample_count = sum(1 for sample in samples if not sample.watchdog_ok)
    stale_sample_count = sum(1 for sample in samples if not sample.artifact_fresh)
    service_error_count = sum(1 for sample in samples if sample.service_error)
    artifact_error_count = sum(1 for sample in samples if sample.artifact_error)

    sample_progress_ok = sample_count_delta is None or sample_count_delta > 0 or len(samples) < 2

    checks_passed = all(
        (
            bool(samples),
            non_running_sample_count == 0,
            non_ok_sample_count == 0,
            stale_sample_count == 0,
            service_error_count == 0,
            artifact_error_count == 0,
            restart_delta == 0,
            pid_change_count == 0,
            failure_count_delta in (None, 0),
            sample_progress_ok,
        )
    )

    return {
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_s": round(duration_s, 3),
        "interval_s": round(interval_s, 3),
        "max_stale_s": round(max_stale_s, 3),
        "service_name": service_name,
        "snapshot_path": str(snapshot_path),
        "output_dir": str(output_dir),
        "sample_total": len(samples),
        "all_checks_passed": checks_passed,
        "non_running_sample_count": non_running_sample_count,
        "non_ok_sample_count": non_ok_sample_count,
        "stale_sample_count": stale_sample_count,
        "service_error_count": service_error_count,
        "artifact_error_count": artifact_error_count,
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
        "max_observed_stale_s": round(max(stale_values), 3) if stale_values else None,
        "max_latency_ms": round(max(latency_values), 3) if latency_values else None,
    }


def _status_payload(
    samples: list[WatchdogSoakSample],
    *,
    complete: bool,
    started_at: str,
    ended_at: str,
    duration_s: float,
    interval_s: float,
    max_stale_s: float,
    service_name: str,
    snapshot_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build one status payload for both progress and final summary files."""

    payload = build_soak_summary(
        samples,
        started_at=started_at,
        ended_at=ended_at,
        duration_s=duration_s,
        interval_s=interval_s,
        max_stale_s=max_stale_s,
        service_name=service_name,
        snapshot_path=snapshot_path,
        output_dir=output_dir,
    )
    payload["complete"] = complete
    payload["last_observed_at"] = samples[-1].observed_at if samples else None
    payload["samples_path"] = str(output_dir / "samples.jsonl")
    return payload


def run_soak(
    *,
    project_root: Path,
    service_name: str,
    snapshot_path: Path,
    output_dir: Path,
    duration_s: float,
    interval_s: float,
    max_stale_s: float,
) -> int:
    """Run one bounded soak observation and persist rolling artifacts."""

    started_dt = _utc_now()
    started_at = _isoformat_utc(started_dt)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    status_path = output_dir / "status.json"
    samples: list[WatchdogSoakSample] = []

    deadline = time.monotonic() + duration_s
    next_tick = time.monotonic()

    while True:
        now_monotonic = time.monotonic()
        if now_monotonic < next_tick:
            time.sleep(min(0.5, next_tick - now_monotonic))
            continue

        sample = collect_soak_sample(
            service_name=service_name,
            snapshot_path=snapshot_path,
            started_at=started_dt,
            max_stale_s=max_stale_s,
        )
        samples.append(sample)
        _append_jsonl(samples_path, sample.to_dict())
        _atomic_write_json(
            status_path,
            _status_payload(
                samples,
                complete=False,
                started_at=started_at,
                ended_at=sample.observed_at,
                duration_s=sample.elapsed_s,
                interval_s=interval_s,
                max_stale_s=max_stale_s,
                service_name=service_name,
                snapshot_path=snapshot_path,
                output_dir=output_dir,
            ),
        )

        if time.monotonic() >= deadline:
            break
        next_tick += interval_s

    final_duration_s = samples[-1].elapsed_s if samples else 0.0
    summary_payload = _status_payload(
        samples,
        complete=True,
        started_at=started_at,
        ended_at=samples[-1].observed_at if samples else _utc_now_iso(),
        duration_s=final_duration_s,
        interval_s=interval_s,
        max_stale_s=max_stale_s,
        service_name=service_name,
        snapshot_path=snapshot_path,
        output_dir=output_dir,
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
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the bounded soak recorder from the command line."""

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).expanduser().resolve(strict=False)
    duration_s = _normalize_positive_float(args.duration_s, default=DEFAULT_DURATION_S)
    interval_s = _normalize_positive_float(args.interval_s, default=DEFAULT_INTERVAL_S)
    max_stale_s = _normalize_positive_float(args.max_stale_s, default=DEFAULT_MAX_STALE_S)

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

    return run_soak(
        project_root=project_root,
        service_name=str(args.service_name),
        snapshot_path=snapshot_path,
        output_dir=output_dir,
        duration_s=duration_s,
        interval_s=interval_s,
        max_stale_s=max_stale_s,
    )


if __name__ == "__main__":
    raise SystemExit(main())
