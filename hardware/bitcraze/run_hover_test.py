#!/usr/bin/env python3
# CHANGELOG: 2026-03-27
# BUG-1: The script claimed to be a bounded indoor hover primitive but silently clamped user input and had no upper bounds on height/duration/velocity; a typo could therefore trigger a materially different flight than requested.
# BUG-2: The success path unconditionally reported landed=True and completed=True when stability checks passed, even if the hover primitive outcome did not actually confirm takeoff/landing.
# BUG-3: Invalid --require-deck values escaped before the main exception handler, causing an uncaught traceback instead of a normal operator-facing failure.
# SEC-1: --trace-file accepted arbitrary paths and followed the final path component blindly; on a Raspberry Pi service account this can be abused to append to attacker-chosen files via symlink/special-file tricks.
# IMP-1: Replace blocking one-shot SyncLogger reads with bounded callback-based log sampling and add preflight gating on supervisor.info, radio.isConnected, and pm.state.
# IMP-2: Add graceful SIGTERM/SIGHUP aborts, adaptive telemetry sample budgeting, richer radio/supervisor evidence, and stricter argument validation for a truly bounded acceptance primitive.
# BUG-4: Ground-start lateral clearance used a heuristic floor bypass while twinrFs armed laterals as soon as it observed ~8 cm, so host takeoff and firmware failsafe could fight during ascent. Preflight, takeoff, and on-device arming now share one explicit active-height contract.
# BUG-5: Hover acceptance used to treat any liftoff plus landing as good enough even when the craft never reached target height or was already drifting hard. Acceptance now fails closed on inadequate achieved height, excessive drift, excessive attitude, and excessive horizontal speed.
# BUG-6: Landing could still classify success before the supervisor grounded state caught up, which allowed a short post-touchdown re-hop. Hover reports now distinguish range-only touchdown failures from true grounded completion.
# BUG-7: The in-flight stability contract lived only in post-flight summary grading, so visibly unstable hover could continue for seconds before landing. The worker now uses the primitive's live stabilize/guard path and reports explicit bounded-hover outcome classes.
# BUG-8: Real hardware still used hover-mode for the first micro-liftoff, so dead `range.zrange`/`motion.squal` truth could produce a real climb before the host knew takeoff had failed. The hardware worker now enables a bounded raw-thrust micro-liftoff that must prove fresh z-range and flow liveness before hover-mode is allowed.
# BUG-9: The first closed-loop hardware bootstrap was still reference-limited, so under missing lift progress it timed out around ~46% thrust instead of deterministically driving to the configured ceiling. The bootstrap now adds an explicit progress-drive term that escalates thrust to ceiling under missing range-rise.
# BUG-10: Preflight power/supervisor state used one arbitrary log packet, so transient `pm.state`/start-state flaps could randomly block reruns. Preflight now samples one bounded status window, uses the latest sample as truth, and surfaces any state flapping explicitly.
# BUG-11: Hover reports used the inferred airborne window as their only flow truth, so real raw `motion.squal` evidence during failed takeoff could be mislabeled as "flow never nonzero". The report now distinguishes raw-flow truth from missing airborne-window truth and surfaces takeoff lateral evidence separately.
# BREAKING: Numeric CLI inputs are now validated and rejected instead of being silently clamped.
# BREAKING: The bounded hover primitive now enforces maximum indoor-safety envelopes for height, hover duration, and vertical velocities.
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cflib>=0.1.31",
# ]
# ///
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import signal
import stat
import sys
import threading
import time
from types import FrameType
from typing import Any, Iterable, Mapping, TypeAlias, TypeVar, TypedDict, cast
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
from hover_primitive import HoverEstimatorSettlingConfig, HoverEstimatorSettlingReport, HoverGroundDistanceObservation, HoverLinkHealthConfig, HoverPreArmConfig, HoverPreArmSnapshot, HoverPrimitiveConfig, HoverPrimitiveOutcome, HoverStabilityConfig, HoverStabilityObservation, HoverVerticalBootstrapConfig, HOVER_MICRO_LIFTOFF_HEIGHT_M, HOVER_STABILITY_MAX_ATTITUDE_ABS_DEG, HOVER_STABILITY_MAX_HORIZONTAL_SPEED_MPS, HOVER_STABILITY_MAX_XY_DRIFT_M, HOVER_TAKEOFF_CONFIRM_MAX_AGE_S, HOVER_TAKEOFF_TARGET_HEIGHT_TOLERANCE_M, StatefulHoverPrimitive, apply_hover_pre_arm, compute_takeoff_active_height_m, wait_for_estimator_settle  # noqa: E402
from on_device_failsafe import (  # noqa: E402
    ON_DEVICE_FAILSAFE_COMMAND_PROTOCOL_VERSION,
    ON_DEVICE_FAILSAFE_CRITICAL_BATTERY_V,
    ON_DEVICE_FAILSAFE_DEBUG_FLAG_ATTITUDE_READY,
    ON_DEVICE_FAILSAFE_DEBUG_FLAG_DISTURBANCE_VALID,
    ON_DEVICE_FAILSAFE_DEBUG_FLAG_FLOW_READY,
    ON_DEVICE_FAILSAFE_DEBUG_FLAG_HOVER_THRUST_VALID,
    ON_DEVICE_FAILSAFE_DEBUG_FLAG_RANGE_READY,
    ON_DEVICE_FAILSAFE_DEBUG_FLAG_THRUST_AT_CEILING,
    ON_DEVICE_FAILSAFE_DEBUG_FLAG_TOUCHDOWN_BY_RANGE,
    ON_DEVICE_FAILSAFE_DEBUG_FLAG_TOUCHDOWN_BY_SUPERVISOR,
    ON_DEVICE_FAILSAFE_HEARTBEAT_TIMEOUT_S,
    ON_DEVICE_FAILSAFE_LOW_BATTERY_V,
    ON_DEVICE_FAILSAFE_MIN_UP_CLEARANCE_M,
    OnDeviceFailsafeConfig,
    OnDeviceFailsafeHeartbeatSession,
    OnDeviceFailsafeSessionReport,
    OnDeviceHoverIntent,
    OnDeviceHoverResult,
    probe_on_device_failsafe,
)
from twinr.hardware.crazyflie_start_contract import (  # noqa: E402
    StartEnvelopeConfig,
    evaluate_start_clearance_envelope,
    lateral_clearance_gate_active,
)
from twinr.hardware.crazyflie_trusted_state import LateralTrustConfig  # noqa: E402
from twinr.hardware.crazyflie_telemetry import (  # noqa: E402
    CompositeTraceWriter,
    CrazyflieTelemetryRuntime,
    CrazyflieTelemetrySample,
    CrazyflieTelemetrySummary,
    TelemetryProfile,
    profile_log_blocks,
    summarize_crazyflie_telemetry,
)
DEFAULT_URI = 'radio://0/80/2M'
DEFAULT_REQUIRED_DECKS = ('bcFlow2', 'bcZRanger2')
DECK_NAME_ALIASES = {'flow2': 'bcFlow2', 'flow': 'bcFlow2', 'bcflow2': 'bcFlow2', 'zranger2': 'bcZRanger2', 'zranger': 'bcZRanger2', 'bczranger2': 'bcZRanger2', 'multiranger': 'bcMultiranger', 'multi-ranger': 'bcMultiranger', 'bcmultiranger': 'bcMultiranger', 'aideck': 'bcAI', 'ai': 'bcAI', 'bcai': 'bcAI'}
DECK_PARAM_NAMES = ('bcMultiranger', 'bcFlow2', 'bcZRanger2', 'bcAI')
HOVER_TELEMETRY_PERIOD_MS = 100
HOVER_TELEMETRY_STARTUP_SETTLE_S = 0.2
HOVER_TELEMETRY_MAX_SAMPLES = 400
HOVER_TELEMETRY_MAX_SAMPLES_CAP = 5000
HOVER_TELEMETRY_AIRBORNE_MIN_ALTITUDE_M = 0.02
HOVER_TELEMETRY_WINDOW_PADDING_MS = HOVER_TELEMETRY_PERIOD_MS
HOVER_RANGE_INVALID_MM = 32000.0
HOVER_PREFLIGHT_CLEARANCE_SAMPLE_S = 0.4
HOVER_PREFLIGHT_CLEARANCE_PERIOD_S = 0.1
HOVER_PREFLIGHT_LOG_TIMEOUT_S = 2.0
HOVER_PREFLIGHT_LOG_WINDOW_S = 0.5
HOVER_PREFLIGHT_LOG_MIN_SAMPLES = 3
HOVER_PREFLIGHT_LOG_WAIT_SLICE_S = 0.05
HOVER_DEFAULT_MIN_CLEARANCE_M = 0.35
HOVER_STABILITY_ALTITUDE_MARGIN_M = 0.25
HOVER_STABILITY_MIN_UNDER_LOAD_VBAT_V = 3.5
HOVER_DEFAULT_ESTIMATOR = 2
HOVER_DEFAULT_CONTROLLER = 1
HOVER_DEFAULT_MOTION_DISABLE = 0
HOVER_DEFAULT_ON_DEVICE_FAILSAFE_MODE = 'required'
HOVER_RUNTIME_MODE_HARDWARE = 'hardware'
HOVER_RUNTIME_MODE_SITL = 'sitl'
HOVER_HARDWARE_VERTICAL_BOOTSTRAP_MIN_THRUST_PERCENT = 36.0
HOVER_HARDWARE_VERTICAL_BOOTSTRAP_FEEDFORWARD_THRUST_PERCENT = 40.0
HOVER_HARDWARE_VERTICAL_BOOTSTRAP_MAX_THRUST_PERCENT = 52.0
HOVER_HARDWARE_VERTICAL_BOOTSTRAP_REFERENCE_DURATION_S = 0.75
HOVER_HARDWARE_VERTICAL_BOOTSTRAP_PROGRESS_TO_CEILING_S = 0.35
HOVER_HARDWARE_VERTICAL_BOOTSTRAP_TIMEOUT_S = 1.0
HOVER_HARDWARE_VERTICAL_BOOTSTRAP_HEIGHT_GAIN_PER_M = 120.0
HOVER_HARDWARE_VERTICAL_BOOTSTRAP_VZ_GAIN_PER_MPS = 45.0
HOVER_HARDWARE_VERTICAL_BOOTSTRAP_MIN_RANGE_RISE_M = 0.02
HOVER_HARDWARE_VERTICAL_BOOTSTRAP_MAX_CEILING_WITHOUT_PROGRESS_S = 0.35
HOVER_HARDWARE_VERTICAL_BOOTSTRAP_REQUIRED_LIVENESS_SAMPLES = 1
HOVER_HARDWARE_VERTICAL_BOOTSTRAP_MIN_MOTION_SQUAL = 1
HOVER_MIN_HEIGHT_M = 0.1
HOVER_MAX_HEIGHT_M = 0.5
HOVER_MIN_DURATION_S = 1.0
HOVER_MAX_DURATION_S = 10.0
HOVER_MIN_VERTICAL_VELOCITY_MPS = 0.05
HOVER_MAX_VERTICAL_VELOCITY_MPS = 0.5
HOVER_MAX_CONNECT_SETTLE_S = 5.0
HOVER_MAX_ESTIMATOR_SETTLE_TIMEOUT_S = 20.0
HOVER_MAX_CLEARANCE_M = 4.0
HOVER_MAX_MIN_BATTERY_LEVEL = 100
HOVER_MAX_BATTERY_V = 5.0
HOVER_MAX_FAILSAFE_HEARTBEAT_TIMEOUT_S = 5.0
HOVER_PRE_FLIGHT_SLACK_S = 2.0
POWER_STATE_NAMES = {0: 'battery', 1: 'charging', 2: 'charged', 3: 'low_power', 4: 'shutdown'}
_SUPERVISOR_CAN_ARM_MASK = 1 << 0
_SUPERVISOR_IS_ARMED_MASK = 1 << 1
_SUPERVISOR_AUTO_ARM_MASK = 1 << 2
_SUPERVISOR_CAN_FLY_MASK = 1 << 3
_SUPERVISOR_IS_FLYING_MASK = 1 << 4
_SUPERVISOR_TUMBLED_MASK = 1 << 5
_SUPERVISOR_LOCKED_MASK = 1 << 6
_SUPERVISOR_CRASHED_MASK = 1 << 7
_SUPERVISOR_HL_FLYING_MASK = 1 << 8
_SUPERVISOR_HL_TRAJECTORY_FINISHED_MASK = 1 << 9
_SUPERVISOR_HL_DISABLED_MASK = 1 << 10
_SUPERVISOR_FLAG_BITS = (('tumbled', _SUPERVISOR_TUMBLED_MASK), ('locked', _SUPERVISOR_LOCKED_MASK), ('crashed', _SUPERVISOR_CRASHED_MASK))
_ON_DEVICE_DEBUG_FLAGS = (
    (ON_DEVICE_FAILSAFE_DEBUG_FLAG_RANGE_READY, "range_ready"),
    (ON_DEVICE_FAILSAFE_DEBUG_FLAG_FLOW_READY, "flow_ready"),
    (ON_DEVICE_FAILSAFE_DEBUG_FLAG_THRUST_AT_CEILING, "thrust_at_ceiling"),
    (ON_DEVICE_FAILSAFE_DEBUG_FLAG_HOVER_THRUST_VALID, "hover_thrust_valid"),
    (ON_DEVICE_FAILSAFE_DEBUG_FLAG_DISTURBANCE_VALID, "disturbance_valid"),
    (ON_DEVICE_FAILSAFE_DEBUG_FLAG_TOUCHDOWN_BY_RANGE, "touchdown_by_range"),
    (ON_DEVICE_FAILSAFE_DEBUG_FLAG_TOUCHDOWN_BY_SUPERVISOR, "touchdown_by_supervisor"),
    (ON_DEVICE_FAILSAFE_DEBUG_FLAG_ATTITUDE_READY, "attitude_ready"),
)
_ON_DEVICE_LATERAL_COMMAND_SOURCES = {
    0: "none",
    1: "mission_takeoff",
    2: "mission_hover",
    3: "mission_landing",
    4: "failsafe_brake",
    5: "failsafe_descend",
    6: "touchdown_confirm",
}

HoverTelemetrySample: TypeAlias = CrazyflieTelemetrySample
HoverTelemetrySummary: TypeAlias = CrazyflieTelemetrySummary
summarize_hover_telemetry = summarize_crazyflie_telemetry
_T = TypeVar("_T")


def _normalize_runtime_mode(raw: object) -> str:
    value = str(raw or HOVER_RUNTIME_MODE_HARDWARE).strip().lower()
    if value not in {HOVER_RUNTIME_MODE_HARDWARE, HOVER_RUNTIME_MODE_SITL}:
        raise ValueError(
            "runtime_mode must be one of "
            f"{HOVER_RUNTIME_MODE_HARDWARE!r} or {HOVER_RUNTIME_MODE_SITL!r}; got {value!r}"
        )
    return value


def _telemetry_profile_for_runtime_mode(runtime_mode: str) -> TelemetryProfile:
    if runtime_mode == HOVER_RUNTIME_MODE_SITL:
        return TelemetryProfile.HOVER_ACCEPTANCE_SITL
    return TelemetryProfile.HOVER_ACCEPTANCE


def _touchdown_requires_supervisor_grounded_for_runtime_mode(runtime_mode: str) -> bool:
    if runtime_mode == HOVER_RUNTIME_MODE_SITL:
        return False
    return True


def _link_health_config_for_runtime_mode(runtime_mode: str) -> HoverLinkHealthConfig | None:
    """Return the runtime link-health contract for one hover lane.

    Local CrazySim SITL uses CFLib's UDP transport and host scheduler timing,
    not a real Crazyradio path. The latency callbacks are still useful for
    live-radio protection, but they are not a trustworthy abort signal for
    local SITL acceptance and scenario replay.
    """

    if runtime_mode == HOVER_RUNTIME_MODE_SITL:
        return None
    return HoverLinkHealthConfig()


def _hover_telemetry_blocks_for_runtime_mode(runtime_mode: str) -> tuple[object, ...]:
    return profile_log_blocks(_telemetry_profile_for_runtime_mode(runtime_mode))


def _preflight_log_variables_for_runtime_mode(runtime_mode: str) -> tuple[str, ...]:
    if runtime_mode == HOVER_RUNTIME_MODE_SITL:
        return (
            'pm.vbat',
            'pm.state',
            'supervisor.info',
            'radio.isConnected',
            'range.zrange',
        )
    return (
        'pm.vbat',
        'pm.batteryLevel',
        'pm.state',
        'supervisor.info',
        'radio.isConnected',
        'range.zrange',
        'motion.squal',
    )


def _default_required_decks_for_runtime_mode(runtime_mode: str) -> tuple[str, ...]:
    if runtime_mode == HOVER_RUNTIME_MODE_SITL:
        return ()
    return DEFAULT_REQUIRED_DECKS


def _stability_config_for_runtime_mode(runtime_mode: str) -> HoverStabilityConfig:
    """Build one shared hover-stability contract for the selected runtime mode."""

    require_motion_squal = runtime_mode != HOVER_RUNTIME_MODE_SITL
    return HoverStabilityConfig(
        require_motion_squal=require_motion_squal,
        lateral_trust=LateralTrustConfig(
            require_motion_squal=require_motion_squal,
        ),
    )


def _vertical_bootstrap_config_for_runtime_mode(
    runtime_mode: str,
    *,
    micro_liftoff_height_m: float,
    takeoff_confirm_target_height_tolerance_m: float,
) -> HoverVerticalBootstrapConfig | None:
    """Return the bounded hardware-only vertical bootstrap contract."""

    if runtime_mode == HOVER_RUNTIME_MODE_SITL:
        return None
    bounded_micro_liftoff_height_m = max(0.0, float(micro_liftoff_height_m))
    bounded_target_height_tolerance_m = max(
        0.0,
        float(takeoff_confirm_target_height_tolerance_m),
    )
    return HoverVerticalBootstrapConfig(
        target_height_m=bounded_micro_liftoff_height_m,
        min_thrust_percentage=HOVER_HARDWARE_VERTICAL_BOOTSTRAP_MIN_THRUST_PERCENT,
        feedforward_thrust_percentage=HOVER_HARDWARE_VERTICAL_BOOTSTRAP_FEEDFORWARD_THRUST_PERCENT,
        max_thrust_percentage=HOVER_HARDWARE_VERTICAL_BOOTSTRAP_MAX_THRUST_PERCENT,
        reference_duration_s=HOVER_HARDWARE_VERTICAL_BOOTSTRAP_REFERENCE_DURATION_S,
        progress_to_ceiling_s=HOVER_HARDWARE_VERTICAL_BOOTSTRAP_PROGRESS_TO_CEILING_S,
        max_duration_s=HOVER_HARDWARE_VERTICAL_BOOTSTRAP_TIMEOUT_S,
        height_gain_per_m=HOVER_HARDWARE_VERTICAL_BOOTSTRAP_HEIGHT_GAIN_PER_M,
        vertical_speed_gain_per_mps=HOVER_HARDWARE_VERTICAL_BOOTSTRAP_VZ_GAIN_PER_MPS,
        min_range_height_m=bounded_micro_liftoff_height_m,
        max_range_height_m=bounded_micro_liftoff_height_m + bounded_target_height_tolerance_m,
        min_range_rise_m=HOVER_HARDWARE_VERTICAL_BOOTSTRAP_MIN_RANGE_RISE_M,
        max_observation_age_s=HOVER_TAKEOFF_CONFIRM_MAX_AGE_S,
        max_ceiling_without_progress_s=HOVER_HARDWARE_VERTICAL_BOOTSTRAP_MAX_CEILING_WITHOUT_PROGRESS_S,
        required_liveness_samples=HOVER_HARDWARE_VERTICAL_BOOTSTRAP_REQUIRED_LIVENESS_SAMPLES,
        require_motion_squal_liveness=True,
        min_motion_squal=HOVER_HARDWARE_VERTICAL_BOOTSTRAP_MIN_MOTION_SQUAL,
    )


class HoverTelemetryCollector(CrazyflieTelemetryRuntime):
    """Preserve the historic hover-worker telemetry collector API.

    The actual telemetry ownership now lives in
    ``twinr.hardware.crazyflie_telemetry.CrazyflieTelemetryRuntime``. This thin
    compatibility shim keeps the existing worker/test call surface stable while
    routing all runtime collection through the shared implementation.
    """

    def __init__(
        self,
        sync_cf: Any,
        log_config_cls: Any,
        *,
        period_in_ms: int = HOVER_TELEMETRY_PERIOD_MS,
        max_samples: int = HOVER_TELEMETRY_MAX_SAMPLES,
        profile: TelemetryProfile = TelemetryProfile.HOVER_ACCEPTANCE,
    ) -> None:
        super().__init__(
            sync_cf,
            log_config_cls,
            profile=profile,
            max_samples=max_samples,
            period_in_ms=period_in_ms,
        )

@dataclass(frozen=True)
class HoverPowerSnapshot:
    vbat_v: float | None
    battery_level: int | None
    state: int | None
    observed_states: tuple[int, ...] = ()
    sample_count: int = 0

@dataclass(frozen=True)
class HoverStatusSnapshot:
    supervisor_info: int | None
    can_arm: bool | None
    is_armed: bool | None
    auto_arm: bool | None
    can_fly: bool | None
    is_flying: bool | None
    tumbled: bool | None
    locked: bool | None
    crashed: bool | None
    hl_flying: bool | None
    hl_trajectory_finished: bool | None
    hl_disabled: bool | None
    radio_connected: bool | None
    zrange_m: float | None
    motion_squal: int | None
    observed_can_arm: tuple[bool, ...] = ()
    observed_can_fly: tuple[bool, ...] = ()
    observed_is_armed: tuple[bool, ...] = ()
    observed_supervisor_info: tuple[int, ...] = ()
    sample_count: int = 0

@dataclass(frozen=True)
class HoverClearanceSnapshot:
    front_m: float | None
    back_m: float | None
    left_m: float | None
    right_m: float | None
    up_m: float | None
    down_m: float | None

@dataclass(frozen=True)
class HoverWorkerTraceEvent:
    index: int
    pid: int
    ts_utc: str
    elapsed_s: float
    phase: str
    status: str
    message: str | None = None
    data: dict[str, object] | None = None


class HoverRuntimeConfig(TypedDict):
    runtime_mode: str
    uri: str
    workspace: Path
    height_m: float
    hover_duration_s: float
    takeoff_velocity_mps: float
    land_velocity_mps: float
    connect_settle_s: float
    min_vbat_v: float
    min_battery_level: int
    min_clearance_m: float
    stabilizer_estimator: int
    stabilizer_controller: int
    motion_disable: int
    estimator_settle_timeout_s: float
    on_device_failsafe_mode: str
    on_device_failsafe_heartbeat_timeout_s: float
    on_device_failsafe_low_battery_v: float
    on_device_failsafe_critical_battery_v: float
    on_device_failsafe_min_up_clearance_m: float
    required_decks: tuple[str, ...]


class HoverWorkerTraceWriter:

    def __init__(self, path: Path | None) -> None:
        self.path = self._normalize_path(path)
        self._lock = threading.Lock()
        self._start_monotonic = time.monotonic()
        self._next_index = 0
        self._events: list[HoverWorkerTraceEvent] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._probe_path()

    @staticmethod
    def _normalize_path(path: Path | None) -> Path | None:
        if path is None:
            return None
        candidate = path.expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        return candidate

    def _probe_path(self) -> None:
        fd = self._open_fd()
        os.close(fd)

    def _open_fd(self) -> int:
        if self.path is None:
            raise RuntimeError('trace file path is not configured')
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        nofollow_flag = getattr(os, 'O_NOFOLLOW', 0)
        fd = os.open(str(self.path), flags | nofollow_flag, 384)
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            os.close(fd)
            raise RuntimeError(f'trace path `{self.path}` must be a regular file')
        return fd

    def emit(self, phase: str, *, status: str, message: str | None=None, data: Mapping[str, object] | None=None) -> None:
        with self._lock:
            event = HoverWorkerTraceEvent(index=self._next_index, pid=os.getpid(), ts_utc=datetime.now(timezone.utc).isoformat(), elapsed_s=max(0.0, time.monotonic() - self._start_monotonic), phase=str(phase), status=str(status), message=str(message) if message is not None else None, data=dict(data) if data is not None else None)
            self._next_index += 1
            self._events.append(event)
            if self.path is None:
                return
            encoded = json.dumps(asdict(event), sort_keys=True, default=str)
            fd = self._open_fd()
            try:
                with os.fdopen(fd, 'a', encoding='utf-8') as handle:
                    handle.write(encoded)
                    handle.write('\n')
                    handle.flush()
                    try:
                        os.fsync(handle.fileno())
                    except OSError:
                        pass
            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise

    @property
    def events(self) -> tuple[HoverWorkerTraceEvent, ...]:
        return tuple(self._events)

    @property
    def start_monotonic_s(self) -> float:
        return float(self._start_monotonic)

class GracefulSignalAbortContext:

    def __init__(self, trace_writer: HoverWorkerTraceWriter | None=None) -> None:
        self._trace_writer = trace_writer
        self._previous_handlers: dict[int, Any] = {}
        self.last_signal_name: str | None = None
        self._signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, 'SIGHUP'):
            self._signals.append(signal.SIGHUP)

    def __enter__(self) -> GracefulSignalAbortContext:
        for signum in self._signals:
            self._previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle_signal)
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        for signum, previous in self._previous_handlers.items():
            signal.signal(signum, previous)
        self._previous_handlers.clear()

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        self.last_signal_name = signal.Signals(signum).name
        if self._trace_writer is not None:
            self._trace_writer.emit('signal', status='error', message=self.last_signal_name, data={'signum': signum})
        raise KeyboardInterrupt(self.last_signal_name)

@dataclass(frozen=True)
class HoverTestReport:
    uri: str
    workspace: str
    height_m: float
    hover_duration_s: float
    takeoff_velocity_mps: float
    land_velocity_mps: float
    connect_settle_s: float
    min_vbat_v: float
    min_battery_level: int
    min_clearance_m: float
    deck_flags: dict[str, int | None]
    required_decks: tuple[str, ...]
    clearance_snapshot: HoverClearanceSnapshot | None
    status_snapshot: HoverStatusSnapshot | None
    pre_arm_snapshot: HoverPreArmSnapshot | None
    estimator_settle: HoverEstimatorSettlingReport | None
    power: HoverPowerSnapshot
    status: str
    outcome_class: str
    completed: bool
    landed: bool
    interrupted: bool
    primitive_outcome: HoverPrimitiveOutcome | None
    telemetry: tuple[HoverTelemetrySample, ...]
    telemetry_summary: HoverTelemetrySummary | None
    replay_start_timestamp_ms: int | None
    failures: tuple[str, ...]
    recommendations: tuple[str, ...]
    on_device_failsafe: OnDeviceFailsafeSessionReport | None = None


def classify_hover_outcome(
    *,
    status: str,
    primitive_outcome: HoverPrimitiveOutcome | None,
    failures: Iterable[str],
) -> str:
    """Map one hover worker result into a stable operator-facing outcome class."""

    normalized_status = str(status or "").strip().lower()
    failure_tuple = tuple(str(item) for item in failures)
    if normalized_status == "blocked":
        return "blocked_preflight"
    if normalized_status == "interrupted":
        return "interrupted"
    if primitive_outcome is None:
        return "blocked_preflight"
    if not primitive_outcome.took_off:
        return "takeoff_failed"
    if primitive_outcome.forced_motor_cutoff or primitive_outcome.touchdown_confirmation_source == "timeout_forced_cutoff":
        return "touchdown_not_confirmed"
    if failure_tuple:
        return "unstable_hover_aborted"
    return "bounded_hover_ok"

def normalize_required_deck_name(value: str) -> str:
    normalized = str(value or '').strip().lower()
    if not normalized:
        raise ValueError('required deck names must not be empty')
    try:
        return DECK_NAME_ALIASES[normalized]
    except KeyError as exc:
        allowed = ', '.join(sorted(dict.fromkeys(DECK_NAME_ALIASES)))
        raise ValueError(f'unsupported deck name `{value}`; choose one of: {allowed}') from exc

def _power_state_name(state: int | None) -> str:
    if state is None:
        return 'unknown'
    return POWER_STATE_NAMES.get(int(state), f'unknown_{state}')


def _dedupe_preserve_order(values: Iterable[_T]) -> tuple[_T, ...]:
    ordered: list[_T] = []
    for value in values:
        if value not in ordered:
            ordered.append(value)
    return tuple(ordered)


def _format_power_state_sequence(states: Iterable[int]) -> str:
    ordered_states = _dedupe_preserve_order(int(state) for state in states)
    return " -> ".join(
        f"{_power_state_name(int(state))} ({int(state)})" for state in ordered_states
    )


def _format_bool_sequence(states: Iterable[bool]) -> str:
    ordered_states = _dedupe_preserve_order(bool(state) for state in states)
    return " -> ".join("true" if bool(state) else "false" for state in ordered_states)


def _bool_from_bitfield(value: int | None, mask: int) -> bool | None:
    if value is None:
        return None
    return int(value) & int(mask) != 0

def _normalize_range_mm_to_m(raw: object) -> float | None:
    numeric = _normalize_numeric_value(raw)
    if numeric is None:
        return None
    value_mm = float(numeric)
    if value_mm <= 0.0 or value_mm >= HOVER_RANGE_INVALID_MM:
        return None
    return value_mm / 1000.0

def _decode_status_snapshot(
    sample: Mapping[str, object],
    *,
    observed_can_arm: tuple[bool, ...] = (),
    observed_can_fly: tuple[bool, ...] = (),
    observed_is_armed: tuple[bool, ...] = (),
    observed_supervisor_info: tuple[int, ...] = (),
    sample_count: int = 0,
) -> HoverStatusSnapshot:
    supervisor_info_raw = _normalize_numeric_value(sample.get('supervisor.info'))
    supervisor_info = int(supervisor_info_raw) if supervisor_info_raw is not None else None
    radio_connected_raw = _normalize_numeric_value(sample.get('radio.isConnected'))
    radio_connected = None if radio_connected_raw is None else bool(int(radio_connected_raw))
    motion_squal_raw = _normalize_numeric_value(sample.get('motion.squal'))
    motion_squal = int(motion_squal_raw) if motion_squal_raw is not None else None
    return HoverStatusSnapshot(supervisor_info=supervisor_info, can_arm=_bool_from_bitfield(supervisor_info, _SUPERVISOR_CAN_ARM_MASK), is_armed=_bool_from_bitfield(supervisor_info, _SUPERVISOR_IS_ARMED_MASK), auto_arm=_bool_from_bitfield(supervisor_info, _SUPERVISOR_AUTO_ARM_MASK), can_fly=_bool_from_bitfield(supervisor_info, _SUPERVISOR_CAN_FLY_MASK), is_flying=_bool_from_bitfield(supervisor_info, _SUPERVISOR_IS_FLYING_MASK), tumbled=_bool_from_bitfield(supervisor_info, _SUPERVISOR_TUMBLED_MASK), locked=_bool_from_bitfield(supervisor_info, _SUPERVISOR_LOCKED_MASK), crashed=_bool_from_bitfield(supervisor_info, _SUPERVISOR_CRASHED_MASK), hl_flying=_bool_from_bitfield(supervisor_info, _SUPERVISOR_HL_FLYING_MASK), hl_trajectory_finished=_bool_from_bitfield(supervisor_info, _SUPERVISOR_HL_TRAJECTORY_FINISHED_MASK), hl_disabled=_bool_from_bitfield(supervisor_info, _SUPERVISOR_HL_DISABLED_MASK), radio_connected=radio_connected, zrange_m=_normalize_range_mm_to_m(sample.get('range.zrange')), motion_squal=motion_squal, observed_can_arm=observed_can_arm, observed_can_fly=observed_can_fly, observed_is_armed=observed_is_armed, observed_supervisor_info=observed_supervisor_info, sample_count=sample_count)

def evaluate_hover_preflight(*, runtime_mode: str, deck_flags: dict[str, int | None], required_decks: tuple[str, ...], power: HoverPowerSnapshot, status_snapshot: HoverStatusSnapshot | None, clearance_snapshot: HoverClearanceSnapshot | None, min_vbat_v: float, min_battery_level: int, min_clearance_m: float, lateral_clearance_arm_height_m: float) -> list[str]:
    """Return blocking preflight failures for one bounded hover attempt.

    Live takeoff now fails closed on the full observed obstacle envelope.
    Ground-start lateral clearance is not deferred anymore on the host side;
    if the start area is too tight, the worker must not arm into takeoff.
    """
    failures: list[str] = []
    for deck_name in required_decks:
        if deck_flags.get(deck_name) != 1:
            failures.append(f'required deck {deck_name} is not detected')
    if power.vbat_v is None:
        failures.append('battery voltage is unavailable')
    elif power.vbat_v < min_vbat_v:
        failures.append(f'battery voltage {power.vbat_v:.2f} V is below the {min_vbat_v:.2f} V hover gate')
    if power.battery_level is None and runtime_mode != HOVER_RUNTIME_MODE_SITL:
        failures.append('battery level is unavailable')
    elif power.battery_level is not None and power.battery_level < min_battery_level:
        failures.append(f'battery level {power.battery_level}% is below the {min_battery_level}% hover gate')
    if power.observed_states and len(set(power.observed_states)) > 1:
        failures.append(
            "battery power state flapped during preflight window: "
            f"{_format_power_state_sequence(power.observed_states)}"
        )
    if power.state is None:
        failures.append('battery power state is unavailable')
    elif power.state != 0:
        failures.append(f'battery power state is `{_power_state_name(power.state)}` ({power.state}) instead of `battery` (0)')
    if status_snapshot is None:
        failures.append('supervisor status is unavailable')
    else:
        if status_snapshot.observed_can_arm and len(set(status_snapshot.observed_can_arm)) > 1:
            failures.append(
                "supervisor can_arm flapped during preflight window: "
                f"{_format_bool_sequence(status_snapshot.observed_can_arm)}"
            )
        if status_snapshot.observed_can_fly and len(set(status_snapshot.observed_can_fly)) > 1:
            failures.append(
                "supervisor can_fly flapped during preflight window: "
                f"{_format_bool_sequence(status_snapshot.observed_can_fly)}"
            )
        if status_snapshot.observed_is_armed and len(set(status_snapshot.observed_is_armed)) > 1:
            failures.append(
                "supervisor is_armed flapped during preflight window: "
                f"{_format_bool_sequence(status_snapshot.observed_is_armed)}"
            )
        if status_snapshot.supervisor_info is None:
            failures.append('supervisor.info is unavailable')
        if status_snapshot.radio_connected is False:
            failures.append('radio link is not marked connected by the firmware')
        elif status_snapshot.radio_connected is None:
            failures.append('radio connection state is unavailable')
        if status_snapshot.can_fly is False:
            failures.append('supervisor reports the Crazyflie is not ready to fly')
        if status_snapshot.tumbled:
            failures.append('supervisor reports the Crazyflie is tumbled')
        if status_snapshot.locked:
            failures.append('supervisor reports the Crazyflie is locked and must be restarted')
        if status_snapshot.crashed:
            failures.append('supervisor reports the Crazyflie is in a crashed state')
    if clearance_snapshot is not None:
        measured_down_m = clearance_snapshot.down_m
        if status_snapshot is not None and status_snapshot.zrange_m is not None:
            measured_down_m = float(status_snapshot.zrange_m)
        failures.extend(
            evaluate_start_clearance_envelope(
                front_m=clearance_snapshot.front_m,
                back_m=clearance_snapshot.back_m,
                left_m=clearance_snapshot.left_m,
                right_m=clearance_snapshot.right_m,
                up_m=clearance_snapshot.up_m,
                config=StartEnvelopeConfig(
                    min_clearance_m=min_clearance_m,
                    require_lateral_clearance=(
                        runtime_mode != HOVER_RUNTIME_MODE_SITL
                        and lateral_clearance_gate_active(
                            down_m=measured_down_m,
                            is_flying=None if status_snapshot is None else status_snapshot.is_flying,
                            active_height_m=lateral_clearance_arm_height_m,
                        )
                    ),
                    require_up_clearance=True,
                ),
            )
        )
    return failures

def recommendations_for_report(report: HoverTestReport) -> tuple[str, ...]:
    recommendations: list[str] = []
    if report.failures:
        recommendations.extend(report.failures)
    if report.outcome_class == 'blocked_preflight':
        recommendations.append('Charge the battery, confirm Flow/Z-ranger deck presence, and clear supervisor faults before retrying the hover test.')
    elif report.outcome_class == 'interrupted':
        recommendations.append('Confirm the Crazyflie is on the ground before retrying the hover test.')
    elif report.outcome_class == 'takeoff_failed':
        recommendations.append('Takeoff never entered a valid bounded hover regime; inspect the ascent telemetry before retrying.')
    elif report.outcome_class == 'touchdown_not_confirmed':
        recommendations.append('Landing reached the motor-cutoff recovery path without confirmed grounded state; inspect the touchdown telemetry before retrying.')
    elif report.outcome_class == 'unstable_hover_aborted':
        recommendations.append('Hover completed but acceptance gates failed; inspect the telemetry and primitive outcome before retrying.')
    elif report.outcome_class == 'bounded_hover_ok':
        recommendations.append('Hover test completed and landed. The bounded flight primitive looks ready.')
    if report.status_snapshot is not None and report.status_snapshot.locked:
        recommendations.append('Restart the Crazyflie or perform crash recovery before the next flight attempt.')
    if report.on_device_failsafe is not None:
        if report.on_device_failsafe.mode == 'required' and (not report.on_device_failsafe.availability.loaded):
            recommendations.append('Flash or load the Twinr on-device failsafe app before the next hover attempt.')
        recommendations.extend(report.on_device_failsafe.failures)
    return tuple(dict.fromkeys(recommendations))

def _normalize_numeric_value(raw: object) -> float | int | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, Integral):
        return int(raw)
    if isinstance(raw, Real):
        value = float(raw)
        if math.isnan(value) or math.isinf(value):
            return None
        if value.is_integer():
            return int(value)
        return value
    try:
        value = float(cast(Any, raw))
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    if value.is_integer():
        return int(value)
    return value

def evaluate_hover_stability(summary: HoverTelemetrySummary, *, target_height_m: float, runtime_mode: str, min_clearance_m: float | None = None) -> list[str]:
    failures: list[str] = []
    if summary.analysis_window_status == 'raw_samples_missing_airborne_window':
        failures.append(
            'telemetry was captured during the hover test, but neither downward '
            'range nor state-estimate height entered the airborne telemetry window'
        )
    elif summary.analysis_window_status == 'no_samples':
        failures.append('no telemetry samples were captured during the hover test')
    if runtime_mode != HOVER_RUNTIME_MODE_SITL and not summary.flow_observed:
        if summary.analysis_window_status == 'raw_samples_missing_airborne_window' and summary.raw_flow_observed:
            failures.append(
                'raw optical-flow quality was present, but the hover test never established an airborne telemetry window'
            )
        else:
            failures.append('optical-flow quality never became nonzero during the hover test')
    if runtime_mode != HOVER_RUNTIME_MODE_SITL and not summary.zrange_observed:
        failures.append('downward z-range never produced a nonzero reading during the hover test')
    if summary.supervisor_flags_seen:
        failures.append(f"supervisor reported unsafe flags during the hover test: {', '.join(summary.supervisor_flags_seen)}")
    if summary.radio_disconnect_seen:
        failures.append('firmware reported radio disconnection during the hover test')
    if summary.height_sensor_untrusted_samples > 0:
        failures.append(
            "hover height telemetry became untrusted in "
            f"{summary.height_sensor_untrusted_samples} samples due to downward-range/state-estimate disagreement"
        )
    max_safe_altitude_m = max(HOVER_TELEMETRY_AIRBORNE_MIN_ALTITUDE_M, float(target_height_m) + HOVER_STABILITY_ALTITUDE_MARGIN_M)
    if summary.trusted_height_max_m is None:
        failures.append('trusted hover height never became available during the hover test')
    elif summary.trusted_height_max_m > max_safe_altitude_m:
        failures.append(
            f'hover altitude reached {summary.trusted_height_max_m:.2f} m which exceeds the '
            f'{max_safe_altitude_m:.2f} m safety ceiling for a {target_height_m:.2f} m hover'
        )
    target_height_gate_m = compute_takeoff_active_height_m(float(target_height_m))
    if summary.trusted_height_max_m is not None and summary.trusted_height_max_m < target_height_gate_m:
        failures.append(
            f'hover reached only {summary.trusted_height_max_m:.2f} m which is below the '
            f'{target_height_gate_m:.2f} m target-height gate for a {target_height_m:.2f} m hover'
        )
    if summary.roll_abs_max_deg is not None and summary.roll_abs_max_deg > HOVER_STABILITY_MAX_ATTITUDE_ABS_DEG:
        failures.append(f'roll reached {summary.roll_abs_max_deg:.2f} deg which exceeds the {HOVER_STABILITY_MAX_ATTITUDE_ABS_DEG:.2f} deg hover stability limit')
    if summary.pitch_abs_max_deg is not None and summary.pitch_abs_max_deg > HOVER_STABILITY_MAX_ATTITUDE_ABS_DEG:
        failures.append(f'pitch reached {summary.pitch_abs_max_deg:.2f} deg which exceeds the {HOVER_STABILITY_MAX_ATTITUDE_ABS_DEG:.2f} deg hover stability limit')
    if summary.xy_drift_m is not None and summary.xy_drift_m > HOVER_STABILITY_MAX_XY_DRIFT_M:
        failures.append(f'xy drift reached {summary.xy_drift_m:.2f} m which exceeds the {HOVER_STABILITY_MAX_XY_DRIFT_M:.2f} m hover stability limit')
    if summary.horizontal_speed_max_mps is not None and summary.horizontal_speed_max_mps > HOVER_STABILITY_MAX_HORIZONTAL_SPEED_MPS:
        failures.append(f'horizontal speed reached {summary.horizontal_speed_max_mps:.2f} m/s which exceeds the {HOVER_STABILITY_MAX_HORIZONTAL_SPEED_MPS:.2f} m/s hover stability limit')
    bounded_min_clearance_m = None if min_clearance_m is None else max(0.0, float(min_clearance_m))
    if bounded_min_clearance_m is not None and bounded_min_clearance_m > 0.0:
        clearance_pairs: tuple[tuple[str, float | None], ...] = (
            ('front', summary.front_min_m),
            ('back', summary.back_min_m),
            ('left', summary.left_min_m),
            ('right', summary.right_min_m),
            ('up', summary.up_min_m),
        )
        for direction_name, observed_clearance_m in clearance_pairs:
            if observed_clearance_m is not None and observed_clearance_m < bounded_min_clearance_m:
                failures.append(
                    f'{direction_name} clearance reached {observed_clearance_m:.2f} m which is below the '
                    f'{bounded_min_clearance_m:.2f} m hover safety floor'
                )
    if summary.battery_min_v is not None and summary.battery_min_v < HOVER_STABILITY_MIN_UNDER_LOAD_VBAT_V:
        failures.append(f'battery sagged to {summary.battery_min_v:.2f} V under load which is below the {HOVER_STABILITY_MIN_UNDER_LOAD_VBAT_V:.2f} V hover safety floor')
    return failures

def _evaluate_primitive_outcome(primitive_outcome: HoverPrimitiveOutcome | None) -> list[str]:
    if primitive_outcome is None:
        return ['hover primitive did not return an outcome']
    failures: list[str] = []
    if not primitive_outcome.took_off:
        failures.append('hover primitive did not report a successful takeoff')
    elif not primitive_outcome.stable_hover_established:
        failures.append('hover primitive never established one bounded stable-hover window before hold')
    if primitive_outcome.abort_reason:
        failures.append(str(primitive_outcome.abort_reason))
    if not primitive_outcome.landed:
        failures.append('hover primitive did not report a completed landing')
    if primitive_outcome.touchdown_confirmation_source == 'timeout_forced_cutoff':
        failures.append('hover primitive had to force the final motor cutoff because touchdown was not jointly confirmed')
    return failures


def _bounded_hover_timeout_s(
    *,
    height_m: float,
    hover_duration_s: float,
    takeoff_velocity_mps: float,
    land_velocity_mps: float,
    micro_liftoff_height_m: float,
) -> float:
    """Return one bounded wall-clock timeout for the on-device hover mission."""

    bounded_height_m = max(0.0, float(height_m))
    bounded_micro_liftoff_height_m = max(0.0, float(micro_liftoff_height_m))
    bounded_takeoff_velocity_mps = max(float(takeoff_velocity_mps), 1e-6)
    bounded_land_velocity_mps = max(float(land_velocity_mps), 1e-6)
    takeoff_buffer_s = max(
        bounded_height_m / bounded_takeoff_velocity_mps,
        bounded_micro_liftoff_height_m / bounded_takeoff_velocity_mps,
    )
    landing_buffer_s = max(bounded_height_m / bounded_land_velocity_mps, 0.5)
    return takeoff_buffer_s + float(hover_duration_s) + landing_buffer_s + HOVER_PRE_FLIGHT_SLACK_S


def _primitive_outcome_from_on_device_hover(
    result: OnDeviceHoverResult,
    *,
    target_height_m: float,
) -> HoverPrimitiveOutcome:
    """Project one on-device hover result onto the shared hover outcome surface."""

    final_status = result.final_status
    final_phase = final_status.state_name if final_status is not None else "on_device_missing_status"
    commanded_max_height_m = float(target_height_m)
    if final_status is not None and final_status.commanded_height_mm is not None:
        commanded_max_height_m = max(
            commanded_max_height_m,
            float(final_status.commanded_height_mm) / 1000.0,
        )
    status_context = _format_on_device_status_context(final_status)
    abort_reason: str | None = None
    if result.failures:
        abort_reason = "; ".join(dict.fromkeys(str(failure) for failure in result.failures))
        if status_context:
            abort_reason = f"{abort_reason}; on-device status: {status_context}"
    elif final_status is not None and final_status.state_name.startswith("failsafe_"):
        abort_reason = f"on-device hover mission fell into {final_status.state_name}:{final_status.reason_name}"
        if status_context:
            abort_reason = f"{abort_reason}; {status_context}"
    touchdown_distance_m: float | None = None
    if result.landed and final_status is not None:
        touchdown_distance_m = float(final_status.down_range_mm) / 1000.0
    return HoverPrimitiveOutcome(
        final_phase=final_phase,
        took_off=result.took_off,
        landed=result.landed,
        aborted=abort_reason is not None,
        abort_reason=abort_reason,
        commanded_max_height_m=commanded_max_height_m,
        setpoint_count=max(0, len(result.observed_state_names)),
        forced_motor_cutoff=False,
        touchdown_confirmation_source="on_device" if result.landed else None,
        touchdown_distance_m=touchdown_distance_m,
        touchdown_supervisor_grounded=result.landed,
        stable_hover_established=result.qualified_hover_reached,
        trim_identified=result.qualified_hover_reached,
        qualified_hover_reached=result.qualified_hover_reached,
        landing_trim_identified=result.landing_reached,
        abort_phase=final_phase if abort_reason is not None else None,
    )


def _format_on_device_status_context(status: Any | None) -> str | None:
    """Summarize the final on-device mission truth without inventing host logic."""

    if status is None:
        return None
    parts: list[str] = []
    if getattr(status, "reason_name", None) not in {None, "none"}:
        parts.append(f"reason={status.reason_name}")
    debug_flags = getattr(status, "debug_flags", None)
    if debug_flags is not None:
        active_flags = [
            name
            for bit, name in _ON_DEVICE_DEBUG_FLAGS
            if int(debug_flags) & int(bit)
        ]
        if active_flags:
            parts.append(
                f"debug_flags=0x{int(debug_flags):02x}({','.join(active_flags)})"
            )
        else:
            parts.append(f"debug_flags=0x{int(debug_flags):02x}")
    motion_squal = getattr(status, "motion_squal", None)
    if motion_squal is not None:
        parts.append(f"motion_squal={int(motion_squal)}")
    up_range_mm = getattr(status, "up_range_mm", None)
    if up_range_mm is not None:
        parts.append(f"up_range_m={float(up_range_mm) / 1000.0:.3f}")
    hover_thrust_permille = getattr(status, "hover_thrust_permille", None)
    if hover_thrust_permille is not None:
        parts.append(f"hover_thrust_permille={int(hover_thrust_permille)}")
    return "; ".join(parts) if parts else None


def _find_replay_start_timestamp_ms(
    trace_writer: HoverWorkerTraceWriter,
    telemetry_samples: Iterable[HoverTelemetrySample],
    *,
    phase: str = "hover_primitive_takeoff",
    status: str = "begin",
) -> int | None:
    """Map one primitive phase begin marker onto the first matching telemetry sample.

    Replay must start when the primitive actually entered takeoff, not at the
    beginning of preflight telemetry collection. The worker trace and telemetry
    samples share the same monotonic clock during capture, so the first sample
    at or after the traced phase boundary gives one explicit replay anchor.
    """

    target_event = next(
        (
            event
            for event in trace_writer.events
            if event.phase == phase and event.status == status
        ),
        None,
    )
    if target_event is None:
        return None
    anchor_monotonic_s = trace_writer.start_monotonic_s + float(target_event.elapsed_s)
    for sample in telemetry_samples:
        if sample.received_monotonic_s is None:
            continue
        if float(sample.received_monotonic_s) >= anchor_monotonic_s:
            return int(sample.timestamp_ms)
    return None

def _estimate_telemetry_max_samples(*, height_m: float, hover_duration_s: float, takeoff_velocity_mps: float, land_velocity_mps: float, period_in_ms: int, block_count: int) -> int:
    takeoff_s = max(0.0, float(height_m) / max(float(takeoff_velocity_mps), 1e-06))
    landing_s = max(0.0, float(height_m) / max(float(land_velocity_mps), 1e-06))
    total_s = takeoff_s + float(hover_duration_s) + landing_s + HOVER_TELEMETRY_STARTUP_SETTLE_S + HOVER_PRE_FLIGHT_SLACK_S
    events_per_block = max(1, math.ceil(total_s * 1000.0 / max(int(period_in_ms), 10)))
    estimated = int(events_per_block * max(1, int(block_count)) * 1.2) + 32
    return max(HOVER_TELEMETRY_MAX_SAMPLES, min(estimated, HOVER_TELEMETRY_MAX_SAMPLES_CAP))

def _import_cflib():
    # pylint: disable=import-error
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.log import LogConfig
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.crazyflie.syncLogger import SyncLogger
    from cflib.utils.multiranger import Multiranger
    return (cflib.crtp, Crazyflie, LogConfig, Multiranger, SyncCrazyflie, SyncLogger)

def _read_deck_flags(sync_cf, deck_names: Iterable[str]) -> dict[str, int | None]:
    flags: dict[str, int | None] = {}
    for deck_name in deck_names:
        value: int | None
        try:
            raw = sync_cf.cf.param.get_value(f'deck.{deck_name}')
        except Exception:
            value = None
        else:
            try:
                value = int(str(raw).strip())
            except (TypeError, ValueError):
                value = None
        flags[deck_name] = value
    return flags

def _read_single_log_sample(sync_cf, log_config_cls, *, name: str, variables: tuple[str, ...], period_in_ms: int=100, timeout_s: float=HOVER_PREFLIGHT_LOG_TIMEOUT_S) -> dict[str, object]:
    config = log_config_cls(name=name, period_in_ms=max(10, int(period_in_ms)))
    for variable_name in variables:
        config.add_variable(variable_name)
    received = threading.Event()
    payload: dict[str, object] = {}
    errors: list[str] = []

    def _on_data(timestamp: int, data: Mapping[str, object], log_block: Any) -> None:
        if received.is_set():
            return
        payload.update(dict(data))
        received.set()

    def _on_error(log_conf: Any, message: str) -> None:
        errors.append(str(message))
        received.set()
    sync_cf.cf.log.add_config(config)
    config.data_received_cb.add_callback(_on_data)
    error_cb = getattr(config, 'error_cb', None)
    if error_cb is not None:
        error_cb.add_callback(_on_error)
    try:
        config.start()
        if not received.wait(max(0.1, float(timeout_s))):
            raise TimeoutError(f'{name} did not yield a log sample within {timeout_s:.2f}s')
        if errors and (not payload):
            raise RuntimeError(f'{name} failed before producing data: {errors[0]}')
        if not payload:
            raise RuntimeError(f'{name} yielded no data')
        return payload
    finally:
        try:
            config.stop()
        except Exception:
            pass
        try:
            config.delete()
        except Exception:
            pass
        try:
            config.data_received_cb.remove_callback(_on_data)
        except Exception:
            pass
        if error_cb is not None:
            try:
                error_cb.remove_callback(_on_error)
            except Exception:
                pass


def _read_log_window_samples(
    sync_cf,
    log_config_cls,
    *,
    name: str,
    variables: tuple[str, ...],
    period_in_ms: int = 100,
    timeout_s: float = HOVER_PREFLIGHT_LOG_TIMEOUT_S,
    window_s: float = HOVER_PREFLIGHT_LOG_WINDOW_S,
    min_samples: int = HOVER_PREFLIGHT_LOG_MIN_SAMPLES,
) -> tuple[dict[str, object], ...]:
    """Collect a bounded sequence of log samples and return them in arrival order.

    Preflight decisions must not hinge on one arbitrary packet. This helper
    waits for the first sample, then captures one short bounded window so the
    latest sample becomes the authoritative start state while any flapping is
    still preserved as evidence.
    """

    config = log_config_cls(name=name, period_in_ms=max(10, int(period_in_ms)))
    for variable_name in variables:
        config.add_variable(variable_name)
    first_received = threading.Event()
    activity = threading.Event()
    samples: list[dict[str, object]] = []
    errors: list[str] = []
    lock = threading.Lock()

    def _on_data(timestamp: int, data: Mapping[str, object], log_block: Any) -> None:
        del timestamp, log_block
        with lock:
            samples.append(dict(data))
        first_received.set()
        activity.set()

    def _on_error(log_conf: Any, message: str) -> None:
        del log_conf
        errors.append(str(message))
        activity.set()
        first_received.set()

    sync_cf.cf.log.add_config(config)
    config.data_received_cb.add_callback(_on_data)
    error_cb = getattr(config, 'error_cb', None)
    if error_cb is not None:
        error_cb.add_callback(_on_error)
    try:
        config.start()
        if not first_received.wait(max(0.1, float(timeout_s))):
            raise TimeoutError(f'{name} did not yield a first log sample within {timeout_s:.2f}s')
        if errors and not samples:
            raise RuntimeError(f'{name} failed before producing data: {errors[0]}')
        deadline_s = time.monotonic() + max(0.0, float(window_s))
        while time.monotonic() < deadline_s:
            remaining_s = max(0.0, deadline_s - time.monotonic())
            activity.wait(min(max(HOVER_PREFLIGHT_LOG_WAIT_SLICE_S, 0.01), remaining_s))
            activity.clear()
            if errors:
                raise RuntimeError(f'{name} failed during preflight window: {errors[0]}')
        with lock:
            captured = tuple(samples)
        if not captured:
            raise RuntimeError(f'{name} yielded no data')
        if len(captured) < max(1, int(min_samples)):
            raise RuntimeError(
                f'{name} yielded only {len(captured)} samples during the '
                f'{float(window_s):.2f}s preflight window'
            )
        return captured
    finally:
        try:
            config.stop()
        except Exception:
            pass
        try:
            config.delete()
        except Exception:
            pass
        try:
            config.data_received_cb.remove_callback(_on_data)
        except Exception:
            pass
        if error_cb is not None:
            try:
                error_cb.remove_callback(_on_error)
            except Exception:
                pass


def _read_preflight_snapshots(sync_cf, log_config_cls, *, runtime_mode: str) -> tuple[HoverPowerSnapshot, HoverStatusSnapshot]:
    samples = _read_log_window_samples(
        sync_cf,
        log_config_cls,
        name='hover-preflight',
        variables=_preflight_log_variables_for_runtime_mode(runtime_mode),
        period_in_ms=100,
        timeout_s=HOVER_PREFLIGHT_LOG_TIMEOUT_S,
    )
    sample = samples[-1]
    vbat_raw = _normalize_numeric_value(sample.get('pm.vbat'))
    battery_level_raw = _normalize_numeric_value(sample.get('pm.batteryLevel'))
    state_raw = _normalize_numeric_value(sample.get('pm.state'))
    observed_states = tuple(
        int(state_numeric)
        for state_numeric in (
            _normalize_numeric_value(window_sample.get('pm.state'))
            for window_sample in samples
        )
        if state_numeric is not None
    )
    observed_supervisor_info = tuple(
        int(supervisor_numeric)
        for supervisor_numeric in (
            _normalize_numeric_value(window_sample.get('supervisor.info'))
            for window_sample in samples
        )
        if supervisor_numeric is not None
    )
    observed_can_arm = tuple(
        bool(int(info_value) & _SUPERVISOR_CAN_ARM_MASK)
        for info_value in observed_supervisor_info
    )
    observed_can_fly = tuple(
        bool(int(info_value) & _SUPERVISOR_CAN_FLY_MASK)
        for info_value in observed_supervisor_info
    )
    observed_is_armed = tuple(
        bool(int(info_value) & _SUPERVISOR_IS_ARMED_MASK)
        for info_value in observed_supervisor_info
    )
    power = HoverPowerSnapshot(
        vbat_v=float(vbat_raw) if vbat_raw is not None else None,
        battery_level=int(battery_level_raw) if battery_level_raw is not None else None,
        state=int(state_raw) if state_raw is not None else None,
        observed_states=observed_states,
        sample_count=len(samples),
    )
    return (
        power,
        _decode_status_snapshot(
            sample,
            observed_can_arm=observed_can_arm,
            observed_can_fly=observed_can_fly,
            observed_is_armed=observed_is_armed,
            observed_supervisor_info=observed_supervisor_info,
            sample_count=len(samples),
        ),
    )

def _normalize_clearance_value(raw: object) -> float | None:
    numeric = _normalize_numeric_value(raw)
    if numeric is None:
        return None
    value = float(numeric)
    if value <= 0.0:
        return None
    return value

def _read_clearance_snapshot(sync_cf, multiranger_cls) -> HoverClearanceSnapshot:
    readings: dict[str, list[float]] = {'front': [], 'back': [], 'left': [], 'right': [], 'up': [], 'down': []}
    start = time.monotonic()
    with multiranger_cls(sync_cf, zranger=True) as multiranger:
        while time.monotonic() - start < HOVER_PREFLIGHT_CLEARANCE_SAMPLE_S:
            for direction_name in readings:
                value = _normalize_clearance_value(getattr(multiranger, direction_name))
                if value is not None:
                    readings[direction_name].append(value)
            time.sleep(HOVER_PREFLIGHT_CLEARANCE_PERIOD_S)
    return HoverClearanceSnapshot(front_m=min(readings['front']) if readings['front'] else None, back_m=min(readings['back']) if readings['back'] else None, left_m=min(readings['left']) if readings['left'] else None, right_m=min(readings['right']) if readings['right'] else None, up_m=min(readings['up']) if readings['up'] else None, down_m=min(readings['down']) if readings['down'] else None)

def _latest_ground_distance_from_telemetry(telemetry: HoverTelemetryCollector | None) -> HoverGroundDistanceObservation:
    if telemetry is None:
        return HoverGroundDistanceObservation(distance_m=None, age_s=None)
    latest_value, age_s = telemetry.latest_value('range.zrange')
    supervisor_value, supervisor_age_s = telemetry.latest_value('supervisor.info')
    is_flying: bool | None = None
    if supervisor_value is not None:
        is_flying = int(supervisor_value) & _SUPERVISOR_IS_FLYING_MASK != 0
    if latest_value is None:
        return HoverGroundDistanceObservation(distance_m=None, age_s=None, is_flying=is_flying, supervisor_age_s=supervisor_age_s)
    numeric = float(latest_value)
    if numeric <= 0.0 or numeric >= HOVER_RANGE_INVALID_MM:
        return HoverGroundDistanceObservation(distance_m=None, age_s=age_s, is_flying=is_flying, supervisor_age_s=supervisor_age_s)
    return HoverGroundDistanceObservation(distance_m=numeric / 1000.0, age_s=age_s, is_flying=is_flying, supervisor_age_s=supervisor_age_s)


def _latest_ground_distance_from_sitl_telemetry(telemetry: HoverTelemetryCollector | None) -> HoverGroundDistanceObservation:
    if telemetry is None:
        return HoverGroundDistanceObservation(distance_m=None, age_s=None)
    z_estimate_value, z_estimate_age_s = telemetry.latest_value('stateEstimate.z')
    supervisor_value, supervisor_age_s = telemetry.latest_value('supervisor.info')
    supervisor_info = int(supervisor_value) if supervisor_value is not None else None
    is_flying: bool | None = None
    if supervisor_info is not None:
        is_flying = int(supervisor_info) & _SUPERVISOR_IS_FLYING_MASK != 0
    if z_estimate_value is None:
        return HoverGroundDistanceObservation(distance_m=None, age_s=None, is_flying=is_flying, supervisor_age_s=supervisor_age_s, supervisor_info=supervisor_info)
    return HoverGroundDistanceObservation(distance_m=max(0.0, float(z_estimate_value)), age_s=z_estimate_age_s, is_flying=is_flying, supervisor_age_s=supervisor_age_s, supervisor_info=supervisor_info)


def _latest_stability_observation_from_telemetry(
    telemetry: HoverTelemetryCollector | None,
) -> HoverStabilityObservation:
    """Project the shared telemetry runtime into one bounded hover-stability snapshot."""

    if telemetry is None:
        return HoverStabilityObservation(height_m=None, height_age_s=None)

    zrange_raw, zrange_age_s = telemetry.latest_value("range.zrange")
    z_estimate_raw, z_estimate_age_s = telemetry.latest_value("stateEstimate.z")
    x_raw, x_age_s = telemetry.latest_value("stateEstimate.x")
    y_raw, y_age_s = telemetry.latest_value("stateEstimate.y")
    vx_raw, vx_age_s = telemetry.latest_value("stateEstimate.vx")
    vy_raw, vy_age_s = telemetry.latest_value("stateEstimate.vy")
    vz_raw, vz_age_s = telemetry.latest_value("stateEstimate.vz")
    roll_raw, roll_age_s = telemetry.latest_value("stabilizer.roll")
    pitch_raw, pitch_age_s = telemetry.latest_value("stabilizer.pitch")
    yaw_raw, yaw_age_s = telemetry.latest_value("stabilizer.yaw")
    motion_squal_raw, motion_squal_age_s = telemetry.latest_value("motion.squal")
    supervisor_raw, supervisor_age_s = telemetry.latest_value("supervisor.info")

    zrange_m: float | None = None
    if zrange_raw is not None:
        zrange_numeric = float(zrange_raw)
        if 0.0 < zrange_numeric < HOVER_RANGE_INVALID_MM:
            zrange_m = zrange_numeric / 1000.0

    pose_age_s: float | None = None
    if x_age_s is not None and y_age_s is not None:
        pose_age_s = max(float(x_age_s), float(y_age_s))
    velocity_age_s: float | None = None
    if vx_age_s is not None and vy_age_s is not None:
        velocity_age_s = max(float(vx_age_s), float(vy_age_s))
    attitude_age_s: float | None = None
    if roll_age_s is not None and pitch_age_s is not None:
        attitude_age_s = max(float(roll_age_s), float(pitch_age_s))

    supervisor_info = int(supervisor_raw) if supervisor_raw is not None else None
    is_flying = None if supervisor_info is None else bool(supervisor_info & _SUPERVISOR_IS_FLYING_MASK)

    return HoverStabilityObservation(
        height_m=zrange_m,
        height_age_s=zrange_age_s,
        z_estimate_m=float(z_estimate_raw) if z_estimate_raw is not None else None,
        z_estimate_age_s=z_estimate_age_s,
        x_m=float(x_raw) if x_raw is not None else None,
        y_m=float(y_raw) if y_raw is not None else None,
        pose_age_s=pose_age_s,
        vx_mps=float(vx_raw) if vx_raw is not None else None,
        vy_mps=float(vy_raw) if vy_raw is not None else None,
        velocity_age_s=velocity_age_s,
        vz_mps=float(vz_raw) if vz_raw is not None else None,
        vz_age_s=vz_age_s,
        roll_deg=float(roll_raw) if roll_raw is not None else None,
        pitch_deg=float(pitch_raw) if pitch_raw is not None else None,
        yaw_deg=float(yaw_raw) if yaw_raw is not None else None,
        yaw_age_s=yaw_age_s,
        attitude_age_s=attitude_age_s,
        motion_squal=int(motion_squal_raw) if motion_squal_raw is not None else None,
        motion_squal_age_s=motion_squal_age_s,
        is_flying=is_flying,
        supervisor_age_s=supervisor_age_s,
        supervisor_info=supervisor_info,
    )

def run_hover_test(*, runtime_mode: str, uri: str, workspace: Path, height_m: float, hover_duration_s: float, takeoff_velocity_mps: float, land_velocity_mps: float, connect_settle_s: float, min_vbat_v: float, min_battery_level: int, min_clearance_m: float, stabilizer_estimator: int, stabilizer_controller: int, motion_disable: int, estimator_settle_timeout_s: float, on_device_failsafe_mode: str, on_device_failsafe_heartbeat_timeout_s: float, on_device_failsafe_low_battery_v: float, on_device_failsafe_critical_battery_v: float, on_device_failsafe_min_up_clearance_m: float, required_decks: tuple[str, ...], trace_writer: HoverWorkerTraceWriter | None=None) -> HoverTestReport:
    runtime_mode = _normalize_runtime_mode(runtime_mode)
    trace = trace_writer or HoverWorkerTraceWriter(None)
    trace.emit('run_hover_test', status='begin', data={'runtime_mode': runtime_mode, 'uri': uri, 'workspace': str(workspace), 'height_m': height_m, 'hover_duration_s': hover_duration_s})
    trace.emit('cflib_import', status='begin')
    crtp, crazyflie_cls, log_config_cls, multiranger_cls, sync_crazyflie_cls, sync_logger_cls = _import_cflib()
    trace.emit('cflib_import', status='done')
    workspace = workspace.expanduser()
    workspace.mkdir(parents=True, exist_ok=True)
    workspace = workspace.resolve()
    cache_dir = workspace / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    trace.emit('workspace_ready', status='done', data={'workspace': str(workspace), 'cache_dir': str(cache_dir)})
    trace.emit('crtp_init', status='begin')
    crtp.init_drivers()
    trace.emit('crtp_init', status='done')
    crazyflie = crazyflie_cls(rw_cache=str(cache_dir))
    sync_context = sync_crazyflie_cls(uri, cf=crazyflie)
    sync_cf = None
    entered_sync_context = False
    deck_flags: dict[str, int | None] = {}
    power = HoverPowerSnapshot(vbat_v=None, battery_level=None, state=None)
    status_snapshot: HoverStatusSnapshot | None = None
    clearance_snapshot: HoverClearanceSnapshot | None = None
    pre_arm_snapshot: HoverPreArmSnapshot | None = None
    estimator_settle: HoverEstimatorSettlingReport | None = None
    primitive_outcome: HoverPrimitiveOutcome | None = None
    telemetry_samples: tuple[HoverTelemetrySample, ...] = ()
    telemetry_summary: HoverTelemetrySummary | None = None
    on_device_failsafe_report: OnDeviceFailsafeSessionReport | None = None
    took_off = False
    landed = False
    primitive: StatefulHoverPrimitive | None = None
    telemetry = None
    on_device_failsafe_session: OnDeviceFailsafeHeartbeatSession | None = None
    runtime_trace: HoverWorkerTraceWriter | CompositeTraceWriter = trace

    def _stop_telemetry() -> None:
        if telemetry is None:
            return
        trace.emit('telemetry_stop', status='begin')
        telemetry.stop()
        trace.emit('telemetry_stop', status='done')

    def _snapshot_telemetry() -> tuple[HoverTelemetrySample, ...]:
        trace.emit('telemetry_snapshot', status='begin')
        samples = telemetry.snapshot() if telemetry is not None else ()
        trace.emit('telemetry_snapshot', status='done', data={'sample_count': len(samples)})
        return samples

    def _build_report(*, status: str, completed: bool, landed_flag: bool, interrupted: bool, failures: Iterable[str], telemetry_payload: tuple[HoverTelemetrySample, ...]=(), telemetry_summary_payload: HoverTelemetrySummary | None=None) -> HoverTestReport:
        failure_tuple = tuple(failures)
        replay_start_timestamp_ms = _find_replay_start_timestamp_ms(
            trace,
            telemetry_payload,
        )
        report = HoverTestReport(uri=uri, workspace=str(workspace), height_m=height_m, hover_duration_s=hover_duration_s, takeoff_velocity_mps=takeoff_velocity_mps, land_velocity_mps=land_velocity_mps, connect_settle_s=connect_settle_s, min_vbat_v=min_vbat_v, min_battery_level=min_battery_level, min_clearance_m=min_clearance_m, deck_flags=deck_flags, required_decks=required_decks, clearance_snapshot=clearance_snapshot, status_snapshot=status_snapshot, pre_arm_snapshot=pre_arm_snapshot, estimator_settle=estimator_settle, power=power, status=status, outcome_class=classify_hover_outcome(status=status, primitive_outcome=primitive_outcome, failures=failure_tuple), completed=completed, landed=landed_flag, interrupted=interrupted, primitive_outcome=primitive_outcome, telemetry=telemetry_payload, telemetry_summary=telemetry_summary_payload, replay_start_timestamp_ms=replay_start_timestamp_ms, failures=failure_tuple, recommendations=(), on_device_failsafe=on_device_failsafe_report)
        trace.emit('report_build', status='begin', data={'status': status})
        final_report = replace(report, recommendations=recommendations_for_report(report))
        trace.emit('report_build', status='done', data={'status': final_report.status})
        return final_report
    try:
        trace.emit('sync_connect', status='begin', data={'uri': uri})
        sync_cf = sync_context.__enter__()
        entered_sync_context = True
        trace.emit('sync_connect', status='done')
        if connect_settle_s > 0:
            trace.emit('connect_settle', status='begin', data={'sleep_s': connect_settle_s})
            time.sleep(connect_settle_s)
            trace.emit('connect_settle', status='done')
        trace.emit('deck_flags', status='begin')
        deck_flags = _read_deck_flags(sync_cf, DECK_PARAM_NAMES)
        trace.emit('deck_flags', status='done', data={'deck_flags': deck_flags})
        if deck_flags.get('bcMultiranger') == 1:
            trace.emit('clearance_snapshot', status='begin')
            clearance_snapshot = _read_clearance_snapshot(sync_cf, multiranger_cls)
            trace.emit('clearance_snapshot', status='done', data=asdict(clearance_snapshot))
        else:
            trace.emit('clearance_snapshot', status='skipped', data={'reason': 'multiranger_not_detected'})
        trace.emit('preflight_snapshot', status='begin')
        power, status_snapshot = _read_preflight_snapshots(sync_cf, log_config_cls, runtime_mode=runtime_mode)
        trace.emit('preflight_snapshot', status='done', data={'vbat_v': power.vbat_v, 'battery_level': power.battery_level, 'power_state': power.state, 'power_state_name': _power_state_name(power.state), 'observed_power_states': power.observed_states, 'preflight_sample_count': power.sample_count, 'supervisor_info': None if status_snapshot is None else status_snapshot.supervisor_info, 'observed_supervisor_info': () if status_snapshot is None else status_snapshot.observed_supervisor_info, 'observed_can_arm': () if status_snapshot is None else status_snapshot.observed_can_arm, 'observed_can_fly': () if status_snapshot is None else status_snapshot.observed_can_fly, 'observed_is_armed': () if status_snapshot is None else status_snapshot.observed_is_armed, 'can_fly': None if status_snapshot is None else status_snapshot.can_fly, 'radio_connected': None if status_snapshot is None else status_snapshot.radio_connected, 'zrange_m': None if status_snapshot is None else status_snapshot.zrange_m})
        lateral_clearance_arm_height_m = compute_takeoff_active_height_m(height_m)
        failures = evaluate_hover_preflight(runtime_mode=runtime_mode, deck_flags=deck_flags, required_decks=required_decks, power=power, status_snapshot=status_snapshot, clearance_snapshot=clearance_snapshot, min_vbat_v=min_vbat_v, min_battery_level=min_battery_level, min_clearance_m=min_clearance_m, lateral_clearance_arm_height_m=lateral_clearance_arm_height_m)
        if failures:
            trace.emit('preflight', status='blocked', data={'failures': failures})
            return _build_report(status='blocked', completed=False, landed_flag=False, interrupted=False, failures=failures)
        trace.emit('preflight', status='done')
        hover_telemetry_blocks = _hover_telemetry_blocks_for_runtime_mode(runtime_mode)
        telemetry_max_samples = _estimate_telemetry_max_samples(height_m=height_m, hover_duration_s=hover_duration_s, takeoff_velocity_mps=takeoff_velocity_mps, land_velocity_mps=land_velocity_mps, period_in_ms=HOVER_TELEMETRY_PERIOD_MS, block_count=len(hover_telemetry_blocks))
        telemetry = HoverTelemetryCollector(sync_cf, log_config_cls, max_samples=telemetry_max_samples, profile=_telemetry_profile_for_runtime_mode(runtime_mode))
        runtime_trace = CompositeTraceWriter(trace, telemetry)
        pre_arm_snapshot = apply_hover_pre_arm(
            sync_cf,
            config=HoverPreArmConfig(
                estimator=stabilizer_estimator,
                controller=stabilizer_controller,
                motion_disable=motion_disable,
                require_motion_disable_param=(runtime_mode != HOVER_RUNTIME_MODE_SITL),
            ),
            trace_writer=runtime_trace,
        )
        if pre_arm_snapshot.failures:
            return _build_report(status='blocked', completed=False, landed_flag=False, interrupted=False, failures=pre_arm_snapshot.failures)
        estimator_settle = wait_for_estimator_settle(
            sync_cf,
            log_config_cls,
            sync_logger_cls,
            config=HoverEstimatorSettlingConfig(
                timeout_s=estimator_settle_timeout_s,
                require_motion_squal=(runtime_mode != HOVER_RUNTIME_MODE_SITL),
                require_ground_range=(runtime_mode != HOVER_RUNTIME_MODE_SITL),
            ),
            trace_writer=runtime_trace,
        )
        if not estimator_settle.stable:
            return _build_report(status='blocked', completed=False, landed_flag=False, interrupted=False, failures=estimator_settle.failures)
        if runtime_mode == HOVER_RUNTIME_MODE_HARDWARE and on_device_failsafe_mode == 'off':
            return _build_report(
                status='blocked',
                completed=False,
                landed_flag=False,
                interrupted=False,
                failures=('hardware bounded hover now requires the on-device twinrFs mission lane; --on-device-failsafe-mode=off is unsupported',),
            )
        if on_device_failsafe_mode != 'off':
            trace.emit('on_device_failsafe_probe', status='begin', data={'mode': on_device_failsafe_mode})
            availability = probe_on_device_failsafe(sync_cf)
            trace.emit('on_device_failsafe_probe', status='done', data={'loaded': availability.loaded, 'protocol_version': availability.protocol_version, 'state': availability.state_name, 'reason': availability.reason_name})
            on_device_failsafe_config = OnDeviceFailsafeConfig(
                heartbeat_timeout_s=on_device_failsafe_heartbeat_timeout_s,
                low_battery_v=on_device_failsafe_low_battery_v,
                critical_battery_v=on_device_failsafe_critical_battery_v,
                min_clearance_m=min_clearance_m,
                min_up_clearance_m=on_device_failsafe_min_up_clearance_m,
                descent_rate_mps=land_velocity_mps,
                arm_lateral_clearance=False,
            )
            on_device_failsafe_report = OnDeviceFailsafeSessionReport(
                mode=on_device_failsafe_mode,
                config=on_device_failsafe_config,
                availability=availability,
                session_id=None,
                started=False,
                closed=False,
                disabled_cleanly=False,
                packets_sent=0,
                status_packets_received=0,
                heartbeat_deadline_misses=0,
                started_monotonic_s=None,
                closed_monotonic_s=None,
                last_heartbeat_sent_monotonic_s=None,
                last_status_received_monotonic_s=None,
                last_status=None,
                observed_state_names=(),
                observed_reason_names=(),
                link_metrics=None,
                failures=availability.failures,
            )
            if runtime_mode == HOVER_RUNTIME_MODE_HARDWARE:
                if not availability.loaded:
                    return _build_report(
                        status='blocked',
                        completed=False,
                        landed_flag=False,
                        interrupted=False,
                        failures=('required on-device hover app `twinrFs` is not loaded on the Crazyflie firmware',),
                    )
                if availability.protocol_version is None or int(availability.protocol_version) < ON_DEVICE_FAILSAFE_COMMAND_PROTOCOL_VERSION:
                    return _build_report(
                        status='blocked',
                        completed=False,
                        landed_flag=False,
                        interrupted=False,
                        failures=(
                            'hardware bounded hover requires twinrFs protocol '
                            f'>= {ON_DEVICE_FAILSAFE_COMMAND_PROTOCOL_VERSION}, '
                            f'got {availability.protocol_version!r}',
                        ),
                    )
            elif not availability.loaded and on_device_failsafe_mode == 'required':
                return _build_report(status='blocked', completed=False, landed_flag=False, interrupted=False, failures=('required on-device failsafe app `twinrFs` is not loaded on the Crazyflie firmware',))
            if availability.loaded:
                on_device_failsafe_session = OnDeviceFailsafeHeartbeatSession(sync_cf, mode=on_device_failsafe_mode, config=on_device_failsafe_config, availability=availability, trace_writer=runtime_trace)
                on_device_failsafe_session.start()
                on_device_failsafe_report = on_device_failsafe_session.report()
        trace.emit('telemetry_start', status='begin')
        telemetry.start()
        trace.emit('telemetry_start', status='done', data={'available_blocks': telemetry.available_blocks, 'skipped_blocks': telemetry.skipped_blocks, 'max_samples': telemetry_max_samples})
        if HOVER_TELEMETRY_STARTUP_SETTLE_S > 0:
            trace.emit('telemetry_settle', status='begin', data={'sleep_s': HOVER_TELEMETRY_STARTUP_SETTLE_S})
            time.sleep(HOVER_TELEMETRY_STARTUP_SETTLE_S)
            trace.emit('telemetry_settle', status='done')
        if runtime_mode == HOVER_RUNTIME_MODE_SITL:
            def _ground_distance_provider() -> HoverGroundDistanceObservation:
                return _latest_ground_distance_from_sitl_telemetry(telemetry)
        elif hasattr(telemetry, 'ground_distance_observation'):
            _ground_distance_provider = telemetry.ground_distance_observation
        else:
            def _ground_distance_provider() -> HoverGroundDistanceObservation:
                return _latest_ground_distance_from_telemetry(telemetry)
        link_health_provider = telemetry.link_health_observation if hasattr(telemetry, 'link_health_observation') else None
        stability_config = _stability_config_for_runtime_mode(runtime_mode)
        micro_liftoff_height_m = HOVER_MICRO_LIFTOFF_HEIGHT_M
        takeoff_confirm_target_height_tolerance_m = HOVER_TAKEOFF_TARGET_HEIGHT_TOLERANCE_M
        if runtime_mode == HOVER_RUNTIME_MODE_HARDWARE:
            if on_device_failsafe_session is None:
                return _build_report(
                    status='blocked',
                    completed=False,
                    landed_flag=False,
                    interrupted=False,
                    failures=('hardware bounded hover requires an active on-device heartbeat session',),
                )
            hover_intent = OnDeviceHoverIntent(
                target_height_m=height_m,
                hover_duration_s=hover_duration_s,
                takeoff_ramp_s=max(
                    micro_liftoff_height_m / max(float(takeoff_velocity_mps), 1e-6),
                    0.1,
                ),
                micro_liftoff_height_m=micro_liftoff_height_m,
                target_tolerance_m=takeoff_confirm_target_height_tolerance_m,
            )
            trace.emit(
                'on_device_hover_start',
                status='begin',
                data={
                    'target_height_m': hover_intent.target_height_m,
                    'hover_duration_s': hover_intent.hover_duration_s,
                    'micro_liftoff_height_m': hover_intent.micro_liftoff_height_m,
                    'takeoff_ramp_s': hover_intent.takeoff_ramp_s,
                    'target_tolerance_m': hover_intent.target_tolerance_m,
                },
            )
            on_device_failsafe_session.start_bounded_hover(hover_intent)
            on_device_hover_result = on_device_failsafe_session.wait_for_bounded_hover_result(
                timeout_s=_bounded_hover_timeout_s(
                    height_m=height_m,
                    hover_duration_s=hover_duration_s,
                    takeoff_velocity_mps=takeoff_velocity_mps,
                    land_velocity_mps=land_velocity_mps,
                    micro_liftoff_height_m=micro_liftoff_height_m,
                )
            )
            primitive_outcome = _primitive_outcome_from_on_device_hover(
                on_device_hover_result,
                target_height_m=height_m,
            )
            on_device_failsafe_report = on_device_failsafe_session.report()
            trace.emit(
                'on_device_hover_start',
                status='done',
                data={
                    'final_phase': primitive_outcome.final_phase,
                    'took_off': primitive_outcome.took_off,
                    'landed': primitive_outcome.landed,
                    'qualified_hover_reached': primitive_outcome.qualified_hover_reached,
                },
            )
        else:
            trace.emit('hover_primitive_create', status='begin')
            primitive = StatefulHoverPrimitive(
                sync_cf,
                ground_distance_provider=_ground_distance_provider,
                stability_provider=lambda: _latest_stability_observation_from_telemetry(telemetry),
                link_health_provider=link_health_provider,
                trace_writer=runtime_trace,
            )
            trace.emit('hover_primitive_create', status='done', data={'stability_guard_enabled': True})

            def _arm_lateral_clearance_after_takeoff() -> None:
                if on_device_failsafe_session is None:
                    return
                trace.emit('on_device_failsafe_lateral_clearance', status='begin', data={'armed': True, 'active_height_m': lateral_clearance_arm_height_m})
                on_device_failsafe_session.set_lateral_clearance_armed(True)
                trace.emit('on_device_failsafe_lateral_clearance', status='done', data={'armed': True, 'active_height_m': lateral_clearance_arm_height_m})

            primitive_outcome = primitive.run(
                HoverPrimitiveConfig(
                    target_height_m=height_m,
                    hover_duration_s=hover_duration_s,
                    takeoff_velocity_mps=takeoff_velocity_mps,
                    land_velocity_mps=land_velocity_mps,
                    micro_liftoff_height_m=micro_liftoff_height_m,
                    takeoff_confirm_target_height_tolerance_m=takeoff_confirm_target_height_tolerance_m,
                    touchdown_require_supervisor_grounded=(
                        _touchdown_requires_supervisor_grounded_for_runtime_mode(runtime_mode)
                    ),
                    touchdown_range_only_confirmation_source=(
                        "range_only_sitl" if runtime_mode == HOVER_RUNTIME_MODE_SITL else "range_only"
                    ),
                    link_health=_link_health_config_for_runtime_mode(runtime_mode),
                    stability=stability_config,
                    vertical_bootstrap=_vertical_bootstrap_config_for_runtime_mode(
                        runtime_mode,
                        micro_liftoff_height_m=micro_liftoff_height_m,
                        takeoff_confirm_target_height_tolerance_m=takeoff_confirm_target_height_tolerance_m,
                    ),
                ),
                after_takeoff=_arm_lateral_clearance_after_takeoff,
            )
        took_off = primitive_outcome.took_off
        landed = primitive_outcome.landed
        _stop_telemetry()
    except KeyboardInterrupt:
        trace.emit('worker_interrupt', status='error', message='KeyboardInterrupt')
        try:
            _stop_telemetry()
        except Exception as exc:
            trace.emit('telemetry_stop', status='error', message=f'{exc.__class__.__name__}:{exc}')
        telemetry_samples = _snapshot_telemetry()
        trace.emit('telemetry_summary', status='begin')
        telemetry_summary = summarize_hover_telemetry(telemetry_samples, available_blocks=telemetry.available_blocks if telemetry is not None else (), skipped_blocks=telemetry.skipped_blocks if telemetry is not None else ())
        trace.emit('telemetry_summary', status='done', data={'sample_count': telemetry_summary.sample_count})
        if primitive is not None:
            took_off = took_off or primitive.took_off
            landed = landed or primitive.landed
        if on_device_failsafe_session is not None:
            on_device_failsafe_report = on_device_failsafe_session.report()
        return _build_report(status='interrupted', completed=False, landed_flag=landed, interrupted=True, failures=('hover test interrupted; landing requested',), telemetry_payload=telemetry_samples, telemetry_summary_payload=telemetry_summary)
    except Exception as exc:
        trace.emit('run_hover_test_exception', status='error', message=f'{exc.__class__.__name__}:{exc}')
        try:
            _stop_telemetry()
        except Exception as stop_exc:
            trace.emit('telemetry_stop', status='error', message=f'{stop_exc.__class__.__name__}:{stop_exc}')
        if primitive is not None:
            took_off = took_off or primitive.took_off
            landed = landed or primitive.landed
        raise
    finally:
        if on_device_failsafe_session is not None:
            disable_on_close = bool(landed or not took_off)
            trace.emit('on_device_failsafe_release', status='begin', data={'disable_on_close': disable_on_close, 'took_off': took_off, 'landed': landed})
            on_device_failsafe_session.close(disable=disable_on_close)
            on_device_failsafe_report = on_device_failsafe_session.report()
            trace.emit('on_device_failsafe_release', status='done', data={'disabled_cleanly': on_device_failsafe_report.disabled_cleanly, 'status_packets_received': on_device_failsafe_report.status_packets_received})
        if entered_sync_context:
            trace.emit('sync_disconnect', status='begin')
            try:
                suppressed = bool(sync_context.__exit__(*sys.exc_info()))
            except BaseException as exc:
                trace.emit('sync_disconnect', status='error', message=f'{exc.__class__.__name__}:{exc}')
                raise
            else:
                trace.emit('sync_disconnect', status='done', data={'suppressed': suppressed})
    telemetry_samples = _snapshot_telemetry()
    trace.emit('telemetry_summary', status='begin')
    telemetry_summary = summarize_hover_telemetry(telemetry_samples, available_blocks=telemetry.available_blocks if telemetry is not None else (), skipped_blocks=telemetry.skipped_blocks if telemetry is not None else ())
    trace.emit('telemetry_summary', status='done', data={'sample_count': telemetry_summary.sample_count, 'raw_sample_count': telemetry_summary.raw_sample_count, 'airborne_window_detected': telemetry_summary.airborne_window_detected, 'analysis_window_status': telemetry_summary.analysis_window_status, 'flow_observed': telemetry_summary.flow_observed, 'raw_flow_observed': telemetry_summary.raw_flow_observed, 'zrange_observed': telemetry_summary.zrange_observed, 'stable_supervisor': telemetry_summary.stable_supervisor, 'radio_disconnect_seen': telemetry_summary.radio_disconnect_seen, 'takeoff_lateral_classification': telemetry_summary.takeoff_lateral_classification, 'takeoff_commanded_vx_abs_max_mps': telemetry_summary.takeoff_commanded_vx_abs_max_mps, 'takeoff_commanded_vy_abs_max_mps': telemetry_summary.takeoff_commanded_vy_abs_max_mps, 'takeoff_estimated_horizontal_speed_max_mps': telemetry_summary.takeoff_estimated_horizontal_speed_max_mps})
    stability_failures = evaluate_hover_stability(telemetry_summary, target_height_m=height_m, runtime_mode=runtime_mode)
    primitive_failures = list(_evaluate_primitive_outcome(primitive_outcome))
    all_failures = tuple(primitive_failures + stability_failures)
    trace.emit('stability_eval', status='done', data={'failures': all_failures})
    return _build_report(status='completed' if not all_failures else 'unstable', completed=not all_failures, landed_flag=bool(primitive_outcome.landed) if primitive_outcome is not None else landed, interrupted=False, failures=all_failures, telemetry_payload=telemetry_samples, telemetry_summary_payload=telemetry_summary)

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run a bounded Crazyflie hover test.')
    parser.add_argument('--runtime-mode', choices=(HOVER_RUNTIME_MODE_HARDWARE, HOVER_RUNTIME_MODE_SITL), default=HOVER_RUNTIME_MODE_HARDWARE, help='Hover runtime mode: real hardware or CrazySim SITL (default: hardware).')
    parser.add_argument('--uri', default=DEFAULT_URI, help='Crazyflie radio URI (default: radio://0/80/2M)')
    parser.add_argument('--workspace', default='/twinr/bitcraze', help='Bitcraze workspace root for cflib cache files')
    parser.add_argument('--height-m', type=float, default=0.25, help='Hover height in meters (default: 0.25)')
    parser.add_argument('--hover-duration-s', type=float, default=3.0, help='Hover hold time in seconds (default: 3.0)')
    parser.add_argument('--takeoff-velocity-mps', type=float, default=0.2, help='Takeoff velocity in meters per second (default: 0.2)')
    parser.add_argument('--land-velocity-mps', type=float, default=0.2, help='Landing velocity in meters per second (default: 0.2)')
    parser.add_argument('--connect-settle-s', type=float, default=1.0, help='Initial wait after connect before preflight and takeoff (default: 1.0)')
    parser.add_argument('--min-vbat-v', type=float, default=3.8, help='Minimum battery voltage for hover acceptance (default: 3.8)')
    parser.add_argument('--min-battery-level', type=int, default=20, help='Minimum battery level percent for hover acceptance (default: 20)')
    parser.add_argument('--min-clearance-m', type=float, default=HOVER_DEFAULT_MIN_CLEARANCE_M, help='Minimum observed Multi-ranger clearance in meters when that deck is present (default: 0.35)')
    parser.add_argument('--stabilizer-estimator', type=int, default=HOVER_DEFAULT_ESTIMATOR, help='Deterministic stabilizer.estimator value for hover pre-arm (default: 2)')
    parser.add_argument('--stabilizer-controller', type=int, default=HOVER_DEFAULT_CONTROLLER, help='Deterministic stabilizer.controller value for hover pre-arm (default: 1)')
    parser.add_argument('--motion-disable', type=int, default=HOVER_DEFAULT_MOTION_DISABLE, help='Deterministic motion.disable value for hover pre-arm (default: 0)')
    parser.add_argument('--estimator-settle-timeout-s', type=float, default=HoverEstimatorSettlingConfig().timeout_s, help='Maximum seconds to wait for Kalman settle before takeoff (default: 5.0)')
    parser.add_argument('--on-device-failsafe-mode', choices=('required', 'preferred', 'off'), default=HOVER_DEFAULT_ON_DEVICE_FAILSAFE_MODE, help='Whether the hover worker requires the Twinr firmware failsafe app (default: required).')
    parser.add_argument('--on-device-failsafe-heartbeat-timeout-s', type=float, default=ON_DEVICE_FAILSAFE_HEARTBEAT_TIMEOUT_S, help='Firmware heartbeat timeout for the on-device failsafe in seconds (default: 0.35).')
    parser.add_argument('--on-device-failsafe-low-battery-v', type=float, default=ON_DEVICE_FAILSAFE_LOW_BATTERY_V, help='Firmware low-battery safe-land threshold in volts (default: 3.55).')
    parser.add_argument('--on-device-failsafe-critical-battery-v', type=float, default=ON_DEVICE_FAILSAFE_CRITICAL_BATTERY_V, help='Firmware critical-battery safe-land threshold in volts (default: 3.35).')
    parser.add_argument('--on-device-failsafe-min-up-clearance-m', type=float, default=ON_DEVICE_FAILSAFE_MIN_UP_CLEARANCE_M, help='Firmware upward-clearance trigger in meters (default: 0.25).')
    parser.add_argument('--require-deck', action='append', default=[], help='Require one deck flag (examples: flow2, zranger2, multiranger, aideck); repeat as needed')
    parser.add_argument('--trace-file', default='', help='Optional JSONL trace path for worker-phase diagnostics and timeout forensics.')
    parser.add_argument('--json', action='store_true', help='Emit the full report as JSON')
    return parser

def _validate_bounded_number(name: str, raw_value: float, *, minimum: float | None=None, maximum: float | None=None, unit: str='') -> float:
    value = float(raw_value)
    unit_suffix = f' {unit}' if unit else ''
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be >= {minimum:.2f}{unit_suffix}; got {value:.2f}{unit_suffix}')
    if maximum is not None and value > maximum:
        raise ValueError(f'{name} must be <= {maximum:.2f}{unit_suffix}; got {value:.2f}{unit_suffix}')
    return value

def _validate_runtime_config(args: argparse.Namespace) -> HoverRuntimeConfig:
    runtime_mode = _normalize_runtime_mode(getattr(args, 'runtime_mode', HOVER_RUNTIME_MODE_HARDWARE))
    normalized_decks: list[str] = []
    for raw_name in args.require_deck or []:
        if not str(raw_name or '').strip():
            continue
        deck_name = normalize_required_deck_name(raw_name)
        if deck_name not in normalized_decks:
            normalized_decks.append(deck_name)
    required_decks = tuple(normalized_decks) or _default_required_decks_for_runtime_mode(runtime_mode)
    height_m = _validate_bounded_number('height_m', args.height_m, minimum=HOVER_MIN_HEIGHT_M, maximum=HOVER_MAX_HEIGHT_M, unit='m')
    hover_duration_s = _validate_bounded_number('hover_duration_s', args.hover_duration_s, minimum=HOVER_MIN_DURATION_S, maximum=HOVER_MAX_DURATION_S, unit='s')
    takeoff_velocity_mps = _validate_bounded_number('takeoff_velocity_mps', args.takeoff_velocity_mps, minimum=HOVER_MIN_VERTICAL_VELOCITY_MPS, maximum=HOVER_MAX_VERTICAL_VELOCITY_MPS, unit='m/s')
    land_velocity_mps = _validate_bounded_number('land_velocity_mps', args.land_velocity_mps, minimum=HOVER_MIN_VERTICAL_VELOCITY_MPS, maximum=HOVER_MAX_VERTICAL_VELOCITY_MPS, unit='m/s')
    connect_settle_s = _validate_bounded_number('connect_settle_s', args.connect_settle_s, minimum=0.0, maximum=HOVER_MAX_CONNECT_SETTLE_S, unit='s')
    min_vbat_v = _validate_bounded_number('min_vbat_v', args.min_vbat_v, minimum=0.0, maximum=HOVER_MAX_BATTERY_V, unit='V')
    min_battery_level = int(args.min_battery_level)
    if not 0 <= min_battery_level <= HOVER_MAX_MIN_BATTERY_LEVEL:
        raise ValueError(f'min_battery_level must be between 0 and {HOVER_MAX_MIN_BATTERY_LEVEL}; got {min_battery_level}')
    min_clearance_m = _validate_bounded_number('min_clearance_m', args.min_clearance_m, minimum=0.0, maximum=HOVER_MAX_CLEARANCE_M, unit='m')
    estimator_settle_timeout_s = _validate_bounded_number('estimator_settle_timeout_s', args.estimator_settle_timeout_s, minimum=0.5, maximum=HOVER_MAX_ESTIMATOR_SETTLE_TIMEOUT_S, unit='s')
    heartbeat_timeout_s = _validate_bounded_number('on_device_failsafe_heartbeat_timeout_s', args.on_device_failsafe_heartbeat_timeout_s, minimum=0.05, maximum=HOVER_MAX_FAILSAFE_HEARTBEAT_TIMEOUT_S, unit='s')
    failsafe_low_battery_v = _validate_bounded_number('on_device_failsafe_low_battery_v', args.on_device_failsafe_low_battery_v, minimum=0.0, maximum=HOVER_MAX_BATTERY_V, unit='V')
    failsafe_critical_battery_v = _validate_bounded_number('on_device_failsafe_critical_battery_v', args.on_device_failsafe_critical_battery_v, minimum=0.0, maximum=HOVER_MAX_BATTERY_V, unit='V')
    if failsafe_low_battery_v < failsafe_critical_battery_v:
        raise ValueError('on_device_failsafe_low_battery_v must be >= on_device_failsafe_critical_battery_v')
    failsafe_min_up_clearance_m = _validate_bounded_number('on_device_failsafe_min_up_clearance_m', args.on_device_failsafe_min_up_clearance_m, minimum=0.0, maximum=HOVER_MAX_CLEARANCE_M, unit='m')
    return {'runtime_mode': runtime_mode, 'uri': str(args.uri).strip() or DEFAULT_URI, 'workspace': Path(str(args.workspace).strip() or '/twinr/bitcraze'), 'height_m': height_m, 'hover_duration_s': hover_duration_s, 'takeoff_velocity_mps': takeoff_velocity_mps, 'land_velocity_mps': land_velocity_mps, 'connect_settle_s': connect_settle_s, 'min_vbat_v': min_vbat_v, 'min_battery_level': min_battery_level, 'min_clearance_m': min_clearance_m, 'stabilizer_estimator': int(args.stabilizer_estimator), 'stabilizer_controller': int(args.stabilizer_controller), 'motion_disable': int(args.motion_disable), 'estimator_settle_timeout_s': estimator_settle_timeout_s, 'on_device_failsafe_mode': str(args.on_device_failsafe_mode).strip() or HOVER_DEFAULT_ON_DEVICE_FAILSAFE_MODE, 'on_device_failsafe_heartbeat_timeout_s': heartbeat_timeout_s, 'on_device_failsafe_low_battery_v': failsafe_low_battery_v, 'on_device_failsafe_critical_battery_v': failsafe_critical_battery_v, 'on_device_failsafe_min_up_clearance_m': failsafe_min_up_clearance_m, 'required_decks': required_decks}

def _print_human_report(report: HoverTestReport) -> None:
    print(f'status={report.status}')
    print(f'outcome_class={report.outcome_class}')
    print(f'uri={report.uri}')
    print(f'workspace={report.workspace}')
    print(f'height_m={report.height_m}')
    print(f'hover_duration_s={report.hover_duration_s}')
    print(f'power.vbat_v={report.power.vbat_v}')
    print(f'power.battery_level={report.power.battery_level}')
    print(f'power.state={report.power.state}')
    print(f'power.state_name={_power_state_name(report.power.state)}')
    print(f'min_clearance_m={report.min_clearance_m}')
    if report.status_snapshot is not None:
        print(f'status.supervisor_info={report.status_snapshot.supervisor_info}')
        print(f'status.can_arm={str(report.status_snapshot.can_arm).lower()}')
        print(f'status.can_fly={str(report.status_snapshot.can_fly).lower()}')
        print(f'status.is_flying={str(report.status_snapshot.is_flying).lower()}')
        print(f'status.tumbled={str(report.status_snapshot.tumbled).lower()}')
        print(f'status.locked={str(report.status_snapshot.locked).lower()}')
        print(f'status.crashed={str(report.status_snapshot.crashed).lower()}')
        print(f'status.radio_connected={str(report.status_snapshot.radio_connected).lower()}')
        print(f'status.zrange_m={report.status_snapshot.zrange_m}')
        print(f'status.motion_squal={report.status_snapshot.motion_squal}')
    if report.clearance_snapshot is not None:
        print(f'clearance.front_m={report.clearance_snapshot.front_m}')
        print(f'clearance.back_m={report.clearance_snapshot.back_m}')
        print(f'clearance.left_m={report.clearance_snapshot.left_m}')
        print(f'clearance.right_m={report.clearance_snapshot.right_m}')
        print(f'clearance.up_m={report.clearance_snapshot.up_m}')
        print(f'clearance.down_m={report.clearance_snapshot.down_m}')
    if report.pre_arm_snapshot is not None:
        print(f'pre_arm.estimator={report.pre_arm_snapshot.estimator}')
        print(f'pre_arm.controller={report.pre_arm_snapshot.controller}')
        print(f'pre_arm.motion_disable={report.pre_arm_snapshot.motion_disable}')
        print(f'pre_arm.kalman_reset_after={report.pre_arm_snapshot.kalman_reset_after}')
        print(f'pre_arm.verified={str(report.pre_arm_snapshot.verified).lower()}')
    if report.estimator_settle is not None:
        print(f'settle.stable={str(report.estimator_settle.stable).lower()}')
        print(f'settle.sample_count={report.estimator_settle.sample_count}')
        print(f'settle.var_px_span={report.estimator_settle.var_px_span}')
        print(f'settle.var_py_span={report.estimator_settle.var_py_span}')
        print(f'settle.var_pz_span={report.estimator_settle.var_pz_span}')
        print(f'settle.motion_squal_mean={report.estimator_settle.motion_squal_mean}')
        print(f'settle.motion_squal_nonzero_ratio={report.estimator_settle.motion_squal_nonzero_ratio}')
        print(f'settle.zrange_min_m={report.estimator_settle.zrange_min_m}')
    if report.primitive_outcome is not None:
        print(f'primitive.final_phase={report.primitive_outcome.final_phase}')
        print(f'primitive.took_off={str(report.primitive_outcome.took_off).lower()}')
        print(f'primitive.landed={str(report.primitive_outcome.landed).lower()}')
        print(f'primitive.stable_hover_established={str(report.primitive_outcome.stable_hover_established).lower()}')
        print(f'primitive.touchdown_confirmation_source={report.primitive_outcome.touchdown_confirmation_source}')
        print(f'primitive.touchdown_distance_m={report.primitive_outcome.touchdown_distance_m}')
        print(f'primitive.touchdown_supervisor_grounded={str(report.primitive_outcome.touchdown_supervisor_grounded).lower()}')
        print(f'primitive.commanded_max_height_m={report.primitive_outcome.commanded_max_height_m}')
        print(f'primitive.setpoint_count={report.primitive_outcome.setpoint_count}')
    if report.on_device_failsafe is not None:
        print(f'on_device_failsafe.mode={report.on_device_failsafe.mode}')
        print(f'on_device_failsafe.loaded={str(report.on_device_failsafe.availability.loaded).lower()}')
        print(f'on_device_failsafe.started={str(report.on_device_failsafe.started).lower()}')
        print(f'on_device_failsafe.disabled_cleanly={str(report.on_device_failsafe.disabled_cleanly).lower()}')
        print(f'on_device_failsafe.status_packets_received={report.on_device_failsafe.status_packets_received}')
        if report.on_device_failsafe.last_status is not None:
            print(f'on_device_failsafe.last_state={report.on_device_failsafe.last_status.state_name}')
            print(f'on_device_failsafe.last_reason={report.on_device_failsafe.last_status.reason_name}')
            print(f'on_device_failsafe.last_vbat_mv={report.on_device_failsafe.last_status.vbat_mv}')
            print(f'on_device_failsafe.last_min_clearance_mm={report.on_device_failsafe.last_status.min_clearance_mm}')
    if report.telemetry_summary is not None:
        print(f'telemetry.sample_count={report.telemetry_summary.sample_count}')
        print(f'telemetry.analysis_window_status={report.telemetry_summary.analysis_window_status}')
        print(f"telemetry.available_blocks={','.join(report.telemetry_summary.available_blocks)}")
        if report.telemetry_summary.skipped_blocks:
            print(f"telemetry.skipped_blocks={','.join(report.telemetry_summary.skipped_blocks)}")
        print(f'telemetry.flow_observed={str(report.telemetry_summary.flow_observed).lower()}')
        print(f'telemetry.raw_flow_observed={str(report.telemetry_summary.raw_flow_observed).lower()}')
        print(f'telemetry.zrange_observed={str(report.telemetry_summary.zrange_observed).lower()}')
        print(f'telemetry.xy_drift_m={report.telemetry_summary.xy_drift_m}')
        print(f'telemetry.z_drift_m={report.telemetry_summary.z_drift_m}')
        print(f'telemetry.z_span_m={report.telemetry_summary.z_span_m}')
        print(f'telemetry.trusted_height_min_m={report.telemetry_summary.trusted_height_min_m}')
        print(f'telemetry.trusted_height_max_m={report.telemetry_summary.trusted_height_max_m}')
        print(f'telemetry.height_sensor_disagreement_max_m={report.telemetry_summary.height_sensor_disagreement_max_m}')
        print(f'telemetry.height_sensor_untrusted_samples={report.telemetry_summary.height_sensor_untrusted_samples}')
        print(f'telemetry.horizontal_speed_max_mps={report.telemetry_summary.horizontal_speed_max_mps}')
        print(f'telemetry.flow_squal_mean={report.telemetry_summary.flow_squal_mean}')
        print(f'telemetry.raw_flow_squal_mean={report.telemetry_summary.raw_flow_squal_mean}')
        print(f'telemetry.motion_delta_x_abs_max={report.telemetry_summary.motion_delta_x_abs_max}')
        print(f'telemetry.motion_delta_y_abs_max={report.telemetry_summary.motion_delta_y_abs_max}')
        print(f'telemetry.takeoff_lateral_classification={report.telemetry_summary.takeoff_lateral_classification}')
        takeoff_sources = ','.join(
            _ON_DEVICE_LATERAL_COMMAND_SOURCES.get(code, f'unknown_{code}')
            for code in report.telemetry_summary.takeoff_command_source_codes_seen
        )
        print(f'telemetry.takeoff_command_sources={takeoff_sources}')
        print(f'telemetry.takeoff_commanded_vx_abs_max_mps={report.telemetry_summary.takeoff_commanded_vx_abs_max_mps}')
        print(f'telemetry.takeoff_commanded_vy_abs_max_mps={report.telemetry_summary.takeoff_commanded_vy_abs_max_mps}')
        print(f'telemetry.takeoff_estimated_vx_abs_max_mps={report.telemetry_summary.takeoff_estimated_vx_abs_max_mps}')
        print(f'telemetry.takeoff_estimated_vy_abs_max_mps={report.telemetry_summary.takeoff_estimated_vy_abs_max_mps}')
        print(f'telemetry.takeoff_estimated_horizontal_speed_max_mps={report.telemetry_summary.takeoff_estimated_horizontal_speed_max_mps}')
        print(f'telemetry.takeoff_disturbance_estimate_vx_abs_max_mps={report.telemetry_summary.takeoff_disturbance_estimate_vx_abs_max_mps}')
        print(f'telemetry.takeoff_disturbance_estimate_vy_abs_max_mps={report.telemetry_summary.takeoff_disturbance_estimate_vy_abs_max_mps}')
        print(f'telemetry.front_min_m={report.telemetry_summary.front_min_m}')
        print(f'telemetry.back_min_m={report.telemetry_summary.back_min_m}')
        print(f'telemetry.left_min_m={report.telemetry_summary.left_min_m}')
        print(f'telemetry.right_min_m={report.telemetry_summary.right_min_m}')
        print(f'telemetry.up_min_m={report.telemetry_summary.up_min_m}')
        print(f'telemetry.thrust_max={report.telemetry_summary.thrust_max}')
        print(f'telemetry.gyro_abs_max_dps={report.telemetry_summary.gyro_abs_max_dps}')
        print(f'telemetry.radio_rssi_latest_dbm={report.telemetry_summary.radio_rssi_latest_dbm}')
        print(f'telemetry.radio_disconnect_seen={str(report.telemetry_summary.radio_disconnect_seen).lower()}')
        print(f'telemetry.battery_drop_v={report.telemetry_summary.battery_drop_v}')
        if report.telemetry_summary.supervisor_flags_seen:
            print(f"telemetry.supervisor_flags={','.join(report.telemetry_summary.supervisor_flags_seen)}")
    for deck_name, flag in sorted(report.deck_flags.items()):
        print(f"deck.{deck_name}={(flag if flag is not None else 'unknown')}")
    for failure in report.failures:
        print(f'failure={failure}')
    for recommendation in report.recommendations:
        print(f'recommendation={recommendation}')

def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        trace_writer = HoverWorkerTraceWriter(Path(str(args.trace_file).strip()) if str(args.trace_file).strip() else None)
    except Exception as exc:
        failure_message = f'invalid_trace_file:{exc.__class__.__name__}:{exc}'
        if args.json:
            print(json.dumps({'report': None, 'failures': [failure_message]}, indent=2, sort_keys=True))
        else:
            print(f'failure={failure_message}')
        return 1
    trace_writer.emit('main', status='begin', data={'json': bool(args.json)})
    try:
        config = _validate_runtime_config(args)
    except Exception as exc:
        failure_message = f'invalid_hover_test_arguments:{exc.__class__.__name__}:{exc}'
        trace_writer.emit('main', status='error', message=failure_message)
        if args.json:
            print(json.dumps({'report': None, 'failures': [failure_message]}, indent=2, sort_keys=True))
        else:
            print(f'failure={failure_message}')
        trace_writer.emit('main', status='done', data={'exit_code': 1})
        return 1
    signal_context = GracefulSignalAbortContext(trace_writer)
    try:
        with signal_context:
            report = run_hover_test(trace_writer=trace_writer, **config)
    except KeyboardInterrupt:
        signal_name = signal_context.last_signal_name or 'KeyboardInterrupt'
        failure_message = f'hover_test_interrupted:{signal_name}'
        trace_writer.emit('main', status='error', message=failure_message)
        if args.json:
            print(json.dumps({'report': None, 'failures': [failure_message]}, indent=2, sort_keys=True))
        else:
            print(f'failure={failure_message}')
        trace_writer.emit('main', status='done', data={'exit_code': 130})
        return 130
    except Exception as exc:
        failure_message = f'hover_test_exception:{exc.__class__.__name__}:{exc}'
        failure_payload = {'report': None, 'failures': [failure_message]}
        trace_writer.emit('main', status='error', message=failure_message)
        if args.json:
            trace_writer.emit('json_emit', status='begin', data={'kind': 'failure'})
            print(json.dumps(failure_payload, indent=2, sort_keys=True))
            trace_writer.emit('json_emit', status='done', data={'kind': 'failure'})
        else:
            print(f'failure={failure_message}')
        trace_writer.emit('main', status='done', data={'exit_code': 1})
        return 1
    if args.json:
        trace_writer.emit('json_emit', status='begin', data={'kind': 'success', 'status': report.status})
        print(json.dumps({'report': asdict(report), 'failures': list(report.failures)}, indent=2, sort_keys=True))
        trace_writer.emit('json_emit', status='done', data={'kind': 'success', 'status': report.status})
    else:
        _print_human_report(report)
    exit_code = 1
    if report.status == 'completed':
        exit_code = 0
    elif report.status == 'interrupted':
        exit_code = 130
    trace_writer.emit('main', status='done', data={'exit_code': exit_code, 'report_status': report.status})
    return exit_code
if __name__ == '__main__':
    raise SystemExit(main())
