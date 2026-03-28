#!/usr/bin/env python3
# CHANGELOG: 2026-03-27
# BUG-1: Replaced wall-clock timing with monotonic/perf-counter timing so bounded duration and elapsed values are not broken by clock jumps.
# BUG-2: Replaced drift-prone sleep-after-work cadence with absolute-deadline scheduling so short gestures are less likely to be missed by undersampling.
# BUG-3: Flush partial probe output on SIGINT/SIGTERM and defensive-copy the cached raw pose snapshot to avoid losing data or emitting later-mutated state.
# SEC-1: Replaced direct Path.write_text() output with secure tempfile + atomic os.replace() commit (0600 perms) to avoid symlink overwrite/partial-file hazards.
# IMP-1: Added structured telemetry fields for schema/session correlation, UTC timestamp, observe latency, schedule lag, and raw-vs-final mismatch deltas.
# IMP-2: Added structured stderr lifecycle events so the probe is easier to correlate with modern observability pipelines without changing stdout JSONL samples.
# /// script
# requires-python = ">=3.11"
# ///

"""Capture a bounded live gesture probe from Twinr's Pi AI-camera path.

Purpose
-------
Record the raw MediaPipe pose/fine-hand result that the local AI-camera adapter
cached for the current frame *and* the final composed Twinr observation that
runtime layers consume. This gives one calibration/debug view that answers
whether a missing gesture was lost in the recognizer itself or only later in
surface stabilization / HDMI acknowledgement policy.

Usage
-----
Command-line invocation examples::

    PYTHONPATH=src python3 hardware/piaicam/probe_live_gestures.py --env-file /twinr/.env
    PYTHONPATH=src python3 hardware/piaicam/probe_live_gestures.py --duration-s 8 --interval-s 0.2
    PYTHONPATH=src python3 hardware/piaicam/probe_live_gestures.py --output /tmp/twinr_gesture_probe.jsonl

Inputs
------
- ``--env-file`` Twinr env file used to build the local AI-camera adapter
- ``--duration-s`` bounded probe duration
- ``--interval-s`` bounded capture cadence between observations
- ``--output`` optional JSONL path for the collected probe lines

Outputs
-------
- Prints one JSON line per sampled frame to stdout
- Optionally writes the same JSON lines to ``--output``
- Prints structured lifecycle/error events to stderr
- Exit code 0 when the bounded probe completes, 1 on setup/runtime failure

Notes
-----
This script is intentionally diagnostic-only. It reads the adapter's cached
raw pose result after each observation so operator calibration can compare raw
gesture inference with the final observation contract without teaching that
debug path to the production runtime.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import tempfile
import time
from typing import TYPE_CHECKING, Any, TextIO
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if TYPE_CHECKING:
    from twinr.hardware.camera_ai.adapter import LocalAICameraAdapter
    from twinr.hardware.camera_ai.models import AICameraObservation


SCHEMA_VERSION = "2026-03-27.1"
_STOP_SIGNAL: int | None = None


@dataclass(frozen=True, slots=True)
class GestureProbeSample:
    """Describe one raw-plus-final gesture sample from the local camera path."""

    # BREAKING: Output JSON now includes extra timing/correlation fields so
    # downstream tooling can diagnose cadence drift, latency, and raw-vs-final
    # mismatches directly from the JSONL stream.
    schema_version: str
    session_id: str
    host: str
    pid: int
    sampled_at_utc: str
    index: int
    elapsed_s: float
    observe_latency_ms: float
    schedule_lag_ms: float
    target_interval_s: float
    person_count: int
    final_coarse: str | None
    final_coarse_conf: float | None
    final_fine: str | None
    final_fine_conf: float | None
    raw_coarse: str | None
    raw_coarse_conf: float | None
    raw_fine: str | None
    raw_fine_conf: float | None
    coarse_changed_from_raw: bool | None
    fine_changed_from_raw: bool | None
    coarse_conf_delta: float | None
    fine_conf_delta: float | None
    hand_near: bool
    showing_intent: bool | None
    camera_error: str | None


class AtomicJsonlSink:
    """Write JSONL safely to a temporary file and atomically publish on close."""

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._tmp_path: Path | None = None
        self._fh: TextIO | None = None

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def open(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self._path.name}.",
            suffix=".tmp",
            dir=str(self._path.parent),
            text=True,
        )
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
        except OSError:
            pass
        self._tmp_path = Path(tmp_name)
        self._fh = os.fdopen(fd, "w", encoding="utf-8", buffering=1)

    def write_line(self, line: str) -> None:
        if self._fh is None:
            return
        self._fh.write(line)
        self._fh.write("\n")

    def close(self, *, commit: bool) -> None:
        close_error: Exception | None = None
        replace_error: Exception | None = None

        if self._fh is not None:
            try:
                self._fh.flush()
                os.fsync(self._fh.fileno())
            except OSError:
                pass
            except Exception as exc:  # pragma: no cover - defensive cleanup path
                close_error = exc
            finally:
                try:
                    self._fh.close()
                except Exception as exc:  # pragma: no cover - defensive cleanup path
                    close_error = close_error or exc
                self._fh = None

        if self._tmp_path is not None:
            try:
                if commit and self._path is not None:
                    os.replace(self._tmp_path, self._path)
                else:
                    self._tmp_path.unlink(missing_ok=True)
            except Exception as exc:  # pragma: no cover - filesystem dependent
                replace_error = exc
            finally:
                self._tmp_path = None

        if close_error is not None:
            raise close_error
        if replace_error is not None:
            raise replace_error


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the bounded live gesture probe."""

    parser = argparse.ArgumentParser(
        description="Capture bounded live gesture probe samples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", default=".env", help="Twinr env file path")
    parser.add_argument("--duration-s", type=float, default=12.0, help="Total bounded probe duration")
    parser.add_argument("--interval-s", type=float, default=0.2, help="Target cadence between observations")
    parser.add_argument("--output", type=Path, help="Optional JSONL output path")
    return parser


def _coerce_duration(value: float) -> float:
    return max(0.5, min(60.0, float(value)))


def _coerce_interval(value: float) -> float:
    return max(0.05, min(2.0, float(value)))


def _normalize_output_path(path: Path) -> Path:
    """Return an absolute output path without resolving the final symlink target."""

    path = path.expanduser()
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _handle_stop_signal(signum: int, _frame: object) -> None:
    global _STOP_SIGNAL
    _STOP_SIGNAL = signum


# BREAKING: SIGTERM is now trapped so partial probe data can be flushed and the
# process exits cleanly instead of being terminated mid-write.
def _install_signal_handlers() -> None:
    for signum_name in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, signum_name, None)
        if signum is None:
            continue
        signal.signal(signum, _handle_stop_signal)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_float(value: Any, *, digits: int = 6) -> float | None:
    if value is None:
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(coerced):
        return None
    return round(coerced, digits)


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_label(value: Any) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    if raw is None:
        return None
    return str(raw)


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _delta(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    return round(current - baseline, 6)


def _changed(current: str | None, baseline: str | None) -> bool | None:
    if current is None or baseline is None:
        return None
    return current != baseline


def _deepcopy_or_self(value: Any) -> Any:
    if value is None:
        return None
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _snapshot_raw_pose(adapter: Any) -> Any:
    """Prefer a public accessor if one appears later; otherwise snapshot the cache."""

    for accessor_name in ("snapshot_pose_result", "get_last_pose_result"):
        accessor = getattr(adapter, accessor_name, None)
        if callable(accessor):
            return _deepcopy_or_self(accessor())

    public_attr = getattr(adapter, "last_pose_result", None)
    if public_attr is not None:
        return _deepcopy_or_self(public_attr)

    return _deepcopy_or_self(getattr(adapter, "_last_pose_result", None))


def _build_sample(
    *,
    session_id: str,
    host: str,
    pid: int,
    started_ns: int,
    interval_s: float,
    index: int,
    sample_started_ns: int,
    observe_latency_ns: int,
    schedule_lag_ns: int,
    observation: AICameraObservation,
    raw_pose: Any,
) -> GestureProbeSample:
    """Convert one adapter observation plus cached raw pose into JSON-safe fields."""

    final_coarse = _safe_label(getattr(observation, "gesture_event", None))
    final_coarse_conf = _safe_float(getattr(observation, "gesture_confidence", None))
    final_fine = _safe_label(getattr(observation, "fine_hand_gesture", None))
    final_fine_conf = _safe_float(getattr(observation, "fine_hand_gesture_confidence", None))

    raw_coarse = _safe_label(None if raw_pose is None else getattr(raw_pose, "gesture_event", None))
    raw_coarse_conf = _safe_float(None if raw_pose is None else getattr(raw_pose, "gesture_confidence", None))
    raw_fine = _safe_label(None if raw_pose is None else getattr(raw_pose, "fine_hand_gesture", None))
    raw_fine_conf = _safe_float(None if raw_pose is None else getattr(raw_pose, "fine_hand_gesture_confidence", None))

    return GestureProbeSample(
        schema_version=SCHEMA_VERSION,
        session_id=session_id,
        host=host,
        pid=pid,
        sampled_at_utc=_utc_now_iso(),
        index=index,
        elapsed_s=round((sample_started_ns - started_ns) / 1_000_000_000, 6),
        observe_latency_ms=round(observe_latency_ns / 1_000_000, 3),
        schedule_lag_ms=round(schedule_lag_ns / 1_000_000, 3),
        target_interval_s=round(interval_s, 6),
        person_count=_safe_int(getattr(observation, "person_count", 0), default=0),
        final_coarse=final_coarse,
        final_coarse_conf=final_coarse_conf,
        final_fine=final_fine,
        final_fine_conf=final_fine_conf,
        raw_coarse=raw_coarse,
        raw_coarse_conf=raw_coarse_conf,
        raw_fine=raw_fine,
        raw_fine_conf=raw_fine_conf,
        coarse_changed_from_raw=_changed(final_coarse, raw_coarse),
        fine_changed_from_raw=_changed(final_fine, raw_fine),
        coarse_conf_delta=_delta(final_coarse_conf, raw_coarse_conf),
        fine_conf_delta=_delta(final_fine_conf, raw_fine_conf),
        hand_near=bool(getattr(observation, "hand_or_object_near_camera", False)),
        showing_intent=getattr(observation, "showing_intent_likely", None),
        camera_error=_safe_text(getattr(observation, "camera_error", None)),
    )


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _emit_stderr_event(kind: str, **fields: Any) -> None:
    payload = {
        "kind": kind,
        "sampled_at_utc": _utc_now_iso(),
        **fields,
    }
    print(_json_dumps(payload), file=sys.stderr, flush=True)


def _signal_name(signum: int | None) -> str | None:
    if signum is None:
        return None
    try:
        return signal.Signals(signum).name
    except ValueError:
        return str(signum)


def _ensure_runtime_python() -> None:
    """Reject interpreters older than the Twinr runtime support floor."""

    if sys.version_info[:2] < (3, 11):
        raise RuntimeError(
            "probe_live_gestures_requires_python_3_11:"
            f"detected_{sys.version.split()[0]}"
        )


def _load_twinr_runtime():
    """Import Twinr runtime modules lazily so --help can run without them."""

    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    try:
        from twinr.agent.base_agent.config import TwinrConfig
        from twinr.hardware.camera_ai.adapter import LocalAICameraAdapter
    except Exception as exc:
        raise RuntimeError(f"probe_live_gestures_runtime_import_failed:{type(exc).__name__}:{exc}") from exc
    return TwinrConfig, LocalAICameraAdapter


def main() -> int:
    """Run the bounded gesture probe and emit JSONL samples."""

    global _STOP_SIGNAL
    _STOP_SIGNAL = None

    args = build_parser().parse_args()
    duration_s = _coerce_duration(args.duration_s)
    interval_s = _coerce_interval(args.interval_s)
    output_path = _normalize_output_path(args.output) if args.output else None
    env_file = str(Path(args.env_file).expanduser())
    session_id = uuid.uuid4().hex
    host = socket.gethostname()
    pid = os.getpid()

    sink = AtomicJsonlSink(output_path)
    adapter: LocalAICameraAdapter | None = None
    started_ns: int | None = None
    sample_count = 0
    exit_code = 0

    _install_signal_handlers()

    try:
        sink.open()
        _ensure_runtime_python()
        TwinrConfig, LocalAICameraAdapter = _load_twinr_runtime()
        config = TwinrConfig.from_env(env_file)
        adapter = LocalAICameraAdapter.from_config(config)
        started_ns = time.monotonic_ns()
        duration_ns = int(duration_s * 1_000_000_000)
        interval_ns = int(interval_s * 1_000_000_000)
        index = 0

        while _STOP_SIGNAL is None and (time.monotonic_ns() - started_ns) < duration_ns:
            target_start_ns = started_ns + index * interval_ns
            sample_started_ns = time.monotonic_ns()
            schedule_lag_ns = max(0, sample_started_ns - target_start_ns)

            observe_started_ns = time.perf_counter_ns()
            observation = adapter.observe()
            observe_latency_ns = time.perf_counter_ns() - observe_started_ns
            raw_pose = _snapshot_raw_pose(adapter)
            sample_captured_ns = time.monotonic_ns()

            sample = _build_sample(
                session_id=session_id,
                host=host,
                pid=pid,
                started_ns=started_ns,
                interval_s=interval_s,
                index=index,
                sample_started_ns=sample_captured_ns,
                observe_latency_ns=observe_latency_ns,
                schedule_lag_ns=schedule_lag_ns,
                observation=observation,
                raw_pose=raw_pose,
            )
            line = _json_dumps(asdict(sample))
            print(line, flush=True)
            sink.write_line(line)
            sample_count += 1
            index += 1

            if _STOP_SIGNAL is not None:
                break

            next_target_ns = started_ns + index * interval_ns
            sleep_ns = next_target_ns - time.monotonic_ns()
            if sleep_ns > 0:
                time.sleep(sleep_ns / 1_000_000_000)

        if _STOP_SIGNAL is not None:
            exit_code = 130
            _emit_stderr_event(
                "probe_interrupted",
                session_id=session_id,
                signal=_signal_name(_STOP_SIGNAL),
                samples=sample_count,
            )

    except KeyboardInterrupt:
        exit_code = 130
        _emit_stderr_event(
            "probe_interrupted",
            session_id=session_id,
            signal="SIGINT",
            samples=sample_count,
        )
    except Exception as exc:
        exit_code = 1
        _emit_stderr_event(
            "probe_failed",
            session_id=session_id,
            error_type=type(exc).__name__,
            error=str(exc),
            samples=sample_count,
        )
    finally:
        adapter_close_error: Exception | None = None
        if adapter is not None:
            try:
                adapter.close()
            except Exception as exc:  # pragma: no cover - depends on adapter impl
                adapter_close_error = exc

        commit_output = output_path is not None and sample_count > 0
        try:
            sink.close(commit=commit_output)
        except Exception as exc:
            if exit_code == 0:
                exit_code = 1
            _emit_stderr_event(
                "probe_output_commit_failed",
                session_id=session_id,
                error_type=type(exc).__name__,
                error=str(exc),
            )

        if adapter_close_error is not None:
            if exit_code == 0:
                exit_code = 1
            _emit_stderr_event(
                "probe_close_failed",
                session_id=session_id,
                error_type=type(adapter_close_error).__name__,
                error=str(adapter_close_error),
            )

    if exit_code == 0 and started_ns is not None:
        _emit_stderr_event(
            "probe_completed",
            session_id=session_id,
            samples=sample_count,
            elapsed_s=round((time.monotonic_ns() - started_ns) / 1_000_000_000, 6),
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
