"""Replay bounded Crazyflie hover telemetry without live hardware.

This module provides the deterministic data plane behind Twinr's hover replay
lane. It loads stored hover reports/traces, replays telemetry samples against a
fake monotonic clock, and records synthetic commander/trace outputs so the
existing hover primitive can be exercised without touching a real drone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from twinr.hardware.crazyflie_telemetry import CrazyflieTelemetrySample


@dataclass(frozen=True, slots=True)
class HoverReplayPhaseEvent:
    """Represent one replayable worker/primitive phase event."""

    index: int
    phase: str
    status: str
    elapsed_s: float
    ts_utc: str | None = None
    message: str | None = None
    data: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HoverReplayArtifact:
    """Own one stored hover report plus its optional phase trace."""

    report_path: str
    report_payload: dict[str, object]
    telemetry_samples: tuple[CrazyflieTelemetrySample, ...]
    available_blocks: tuple[str, ...]
    skipped_blocks: tuple[str, ...]
    trace_path: str | None = None
    trace_events: tuple[HoverReplayPhaseEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class HoverReplayCommanderCommand:
    """Record one synthetic commander action emitted during replay."""

    elapsed_s: float
    kind: str
    roll_deg: float | None = None
    pitch_deg: float | None = None
    vx_mps: float | None = None
    vy_mps: float | None = None
    yawrate_dps: float | None = None
    height_m: float | None = None
    thrust_percentage: float | None = None
    rate_mode: bool | None = None


class HoverReplayClock:
    """Drive replay time deterministically without wall-clock sleeps."""

    def __init__(self, *, start_s: float = 0.0) -> None:
        self._time_s = max(0.0, float(start_s))

    def monotonic(self) -> float:
        """Return the current replay time."""

        return self._time_s

    def sleep(self, seconds: float) -> None:
        """Advance replay time by the requested bounded duration."""

        bounded = max(0.0, float(seconds))
        self._time_s += bounded


class CrazyflieTelemetryReplayRuntime:
    """Replay telemetry samples through the runtime's `latest_value` contract."""

    def __init__(
        self,
        samples: Sequence[CrazyflieTelemetrySample],
        *,
        monotonic: Callable[[], float],
        available_blocks: Sequence[str] = (),
        skipped_blocks: Sequence[str] = (),
    ) -> None:
        self._samples = tuple(
            sorted(
                samples,
                key=lambda sample: (
                    int(sample.timestamp_ms),
                    float(sample.received_monotonic_s or 0.0),
                    str(sample.block_name),
                ),
            )
        )
        self._monotonic = monotonic
        self.available_blocks = tuple(dict.fromkeys(str(name) for name in available_blocks))
        self.skipped_blocks = tuple(dict.fromkeys(str(name) for name in skipped_blocks))
        self._first_timestamp_ms = self._samples[0].timestamp_ms if self._samples else 0
        self._cursor = 0
        self._latest_by_key: dict[str, tuple[float | int | None, float]] = {}

    def _elapsed_s_for_sample(self, sample: CrazyflieTelemetrySample) -> float:
        return max(0.0, (int(sample.timestamp_ms) - int(self._first_timestamp_ms)) / 1000.0)

    def _advance(self) -> None:
        current_elapsed_s = max(0.0, float(self._monotonic()))
        while self._cursor < len(self._samples):
            sample = self._samples[self._cursor]
            sample_elapsed_s = self._elapsed_s_for_sample(sample)
            if sample_elapsed_s > current_elapsed_s:
                break
            for key, value in sample.values.items():
                self._latest_by_key[str(key)] = (value, sample_elapsed_s)
            self._cursor += 1

    def latest_value(self, key: str) -> tuple[float | int | None, float | None]:
        """Return the latest replayed value plus its age in seconds."""

        self._advance()
        current_elapsed_s = max(0.0, float(self._monotonic()))
        latest = self._latest_by_key.get(str(key))
        if latest is None:
            return (None, None)
        value, sample_elapsed_s = latest
        return (value, max(0.0, current_elapsed_s - sample_elapsed_s))

    def snapshot(self) -> tuple[CrazyflieTelemetrySample, ...]:
        """Return every telemetry sample already visible at the current replay time."""

        self._advance()
        return self._samples[: self._cursor]

    @property
    def total_sample_count(self) -> int:
        """Return the total number of replayable telemetry samples."""

        return len(self._samples)


class HoverReplayCommander:
    """Capture replayed hover/stop commands without sending them anywhere."""

    def __init__(self, *, monotonic: Callable[[], float]) -> None:
        self._monotonic = monotonic
        self._commands: list[HoverReplayCommanderCommand] = []

    def send_hover_setpoint(self, vx: float, vy: float, yawrate: float, zdistance: float) -> None:
        self._commands.append(
            HoverReplayCommanderCommand(
                elapsed_s=max(0.0, float(self._monotonic())),
                kind="hover",
                vx_mps=float(vx),
                vy_mps=float(vy),
                yawrate_dps=float(yawrate),
                height_m=float(zdistance),
            )
        )

    def send_setpoint_manual(
        self,
        roll: float,
        pitch: float,
        yawrate: float,
        thrust_percentage: float,
        rate: bool,
    ) -> None:
        """Capture one replayed manual commander command."""

        self._commands.append(
            HoverReplayCommanderCommand(
                elapsed_s=max(0.0, float(self._monotonic())),
                kind="manual",
                roll_deg=float(roll),
                pitch_deg=float(pitch),
                yawrate_dps=float(yawrate),
                thrust_percentage=float(thrust_percentage),
                rate_mode=bool(rate),
            )
        )

    def send_stop_setpoint(self) -> None:
        self._commands.append(
            HoverReplayCommanderCommand(
                elapsed_s=max(0.0, float(self._monotonic())),
                kind="stop",
            )
        )

    def send_notify_setpoint_stop(self) -> None:
        self._commands.append(
            HoverReplayCommanderCommand(
                elapsed_s=max(0.0, float(self._monotonic())),
                kind="notify_stop",
            )
        )

    @property
    def commands(self) -> tuple[HoverReplayCommanderCommand, ...]:
        """Return the immutable replayed commander log."""

        return tuple(self._commands)


class HoverReplayTraceWriter:
    """Capture synthetic primitive trace events during replay runs."""

    def __init__(self, *, monotonic: Callable[[], float]) -> None:
        self._monotonic = monotonic
        self._events: list[HoverReplayPhaseEvent] = []

    def emit(
        self,
        phase: str,
        *,
        status: str,
        message: str | None = None,
        data: Mapping[str, object] | None = None,
    ) -> None:
        self._events.append(
            HoverReplayPhaseEvent(
                index=len(self._events),
                phase=str(phase),
                status=str(status),
                elapsed_s=max(0.0, float(self._monotonic())),
                message=message,
                data=dict(data or {}),
            )
        )

    @property
    def events(self) -> tuple[HoverReplayPhaseEvent, ...]:
        """Return the immutable replayed trace log."""

        return tuple(self._events)


def _require_float(raw: object, *, field_name: str) -> float:
    if isinstance(raw, bool):
        return float(int(raw))
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be numeric, got {raw!r}") from exc
    raise ValueError(f"{field_name} must be numeric, got {type(raw).__name__}")


def _require_int(raw: object, *, field_name: str) -> int:
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if raw.is_integer():
            return int(raw)
        raise ValueError(f"{field_name} must be an integer-compatible number, got {raw!r}")
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be integer-compatible, got {raw!r}") from exc
    raise ValueError(f"{field_name} must be integer-compatible, got {type(raw).__name__}")


def _coerce_telemetry_sample(raw: Mapping[str, object]) -> CrazyflieTelemetrySample:
    timestamp_ms = _require_int(raw["timestamp_ms"], field_name="timestamp_ms")
    block_name = str(raw["block_name"])
    values_raw = raw.get("values")
    if not isinstance(values_raw, Mapping):
        raise ValueError("telemetry sample values must be a mapping")
    values: dict[str, float | int | None] = {}
    for key, value in values_raw.items():
        normalized: float | int | None
        if value is None:
            normalized = None
        elif isinstance(value, bool):
            normalized = int(value)
        elif isinstance(value, int):
            normalized = int(value)
        elif isinstance(value, float):
            normalized = float(value)
        else:
            raise ValueError(f"unsupported telemetry value type for {key!r}: {type(value).__name__}")
        values[str(key)] = normalized
    received_monotonic_s_raw = raw.get("received_monotonic_s")
    received_monotonic_s = (
        None
        if received_monotonic_s_raw is None
        else _require_float(received_monotonic_s_raw, field_name="received_monotonic_s")
    )
    return CrazyflieTelemetrySample(
        timestamp_ms=timestamp_ms,
        block_name=block_name,
        values=values,
        received_monotonic_s=received_monotonic_s,
    )


def _coerce_phase_event(raw: Mapping[str, object]) -> HoverReplayPhaseEvent:
    index_raw = raw.get("index")
    elapsed_raw = raw.get("elapsed_s")
    phase_raw = raw.get("phase")
    status_raw = raw.get("status")
    if index_raw is None or elapsed_raw is None or phase_raw is None or status_raw is None:
        raise ValueError("phase event is missing one of: index, elapsed_s, phase, status")
    data_raw = raw.get("data")
    if data_raw is not None and not isinstance(data_raw, Mapping):
        raise ValueError("phase event data must be a mapping when present")
    return HoverReplayPhaseEvent(
        index=_require_int(index_raw, field_name="index"),
        phase=str(phase_raw),
        status=str(status_raw),
        elapsed_s=_require_float(elapsed_raw, field_name="elapsed_s"),
        ts_utc=None if raw.get("ts_utc") is None else str(raw.get("ts_utc")),
        message=None if raw.get("message") is None else str(raw.get("message")),
        data=dict(data_raw or {}),
    )


def load_hover_trace_events(path: Path) -> tuple[HoverReplayPhaseEvent, ...]:
    """Load one hover phase trace JSONL file."""

    resolved = path.expanduser().resolve(strict=True)
    events: list[HoverReplayPhaseEvent] = []
    for line_no, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        raw = json.loads(stripped)
        if not isinstance(raw, Mapping):
            raise ValueError(f"{resolved}:{line_no}: trace line must be a JSON object")
        events.append(_coerce_phase_event(raw))
    return tuple(events)


def load_hover_telemetry_jsonl(path: Path) -> tuple[CrazyflieTelemetrySample, ...]:
    """Load one telemetry-only JSONL file into normalized telemetry samples."""

    resolved = path.expanduser().resolve(strict=True)
    samples: list[CrazyflieTelemetrySample] = []
    for line_no, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        raw = json.loads(stripped)
        if not isinstance(raw, Mapping):
            raise ValueError(f"{resolved}:{line_no}: telemetry line must be a JSON object")
        samples.append(_coerce_telemetry_sample(raw))
    return tuple(samples)


def load_hover_replay_artifact(
    report_path: Path,
    *,
    trace_path: Path | None = None,
) -> HoverReplayArtifact:
    """Load one stored hover report and its optional phase trace."""

    resolved_report = report_path.expanduser().resolve(strict=True)
    report_wrapper_raw = json.loads(resolved_report.read_text(encoding="utf-8"))
    if not isinstance(report_wrapper_raw, Mapping):
        raise ValueError(f"{resolved_report}: hover replay report must be a JSON object")

    raw_report = report_wrapper_raw.get("report", report_wrapper_raw)
    if not isinstance(raw_report, Mapping):
        raise ValueError(f"{resolved_report}: hover replay payload must contain a 'report' object")
    telemetry_raw = raw_report.get("telemetry")
    if not isinstance(telemetry_raw, Sequence):
        raise ValueError(f"{resolved_report}: hover replay report must contain a telemetry sequence")
    telemetry_samples = tuple(_coerce_telemetry_sample(sample) for sample in telemetry_raw if isinstance(sample, Mapping))
    if len(telemetry_samples) != len(telemetry_raw):
        raise ValueError(f"{resolved_report}: every telemetry item must be an object")

    telemetry_summary_raw = raw_report.get("telemetry_summary")
    available_blocks: tuple[str, ...] = ()
    skipped_blocks: tuple[str, ...] = ()
    if isinstance(telemetry_summary_raw, Mapping):
        available_raw = telemetry_summary_raw.get("available_blocks")
        skipped_raw = telemetry_summary_raw.get("skipped_blocks")
        if isinstance(available_raw, Sequence) and not isinstance(available_raw, (str, bytes, bytearray)):
            available_blocks = tuple(str(item) for item in available_raw)
        if isinstance(skipped_raw, Sequence) and not isinstance(skipped_raw, (str, bytes, bytearray)):
            skipped_blocks = tuple(str(item) for item in skipped_raw)
    if not available_blocks:
        available_blocks = tuple(dict.fromkeys(sample.block_name for sample in telemetry_samples))

    trace_events = () if trace_path is None else load_hover_trace_events(trace_path)

    return HoverReplayArtifact(
        report_path=str(resolved_report),
        report_payload=dict(raw_report),
        telemetry_samples=telemetry_samples,
        available_blocks=available_blocks,
        skipped_blocks=skipped_blocks,
        trace_path=None if trace_path is None else str(trace_path.expanduser().resolve(strict=True)),
        trace_events=trace_events,
    )
