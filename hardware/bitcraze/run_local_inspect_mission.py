#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Run one bounded Crazyflie local inspect mission.

Purpose
-------
Execute Twinr's first autonomous indoor inspect slice on top of the existing
Crazyflie hover/failsafe stack. The worker proves preflight, estimator settle,
on-device failsafe, bounded local navigation, image capture, and deterministic
landing in one mission lane.

Usage
-----
Run one local inspect mission directly for bench checks::

    python3 hardware/bitcraze/run_local_inspect_mission.py \
        --repo-root /home/thh/twinr \
        --workspace /twinr/bitcraze \
        --bitcraze-python /twinr/bitcraze/.venv/bin/python \
        --artifact-root /tmp/drone-artifacts \
        --image-name inspect.png \
        --json

Notes
-----
This worker intentionally reuses the bounded hover-test helpers for controller
setup, estimator gating, and on-device failsafe handling while consuming the
shared runtime telemetry lane from `src/twinr/hardware/crazyflie_telemetry.py`.
It adds only the missing mission-specific layers: local free-space planning,
bounded horizontal translation, and bounded still capture at the chosen hover
anchor.
"""
# P0-ARCH-1: 2026-04-14 hardware inspect missions are blocked until they are
# migrated onto the single on-device twinrFs hover/control owner. Leaving the
# older host-side hover primitive alive here would violate Twinr's single
# implementation rule for live Crazyflie takeoff/hover control.

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import site
import signal
import subprocess
import sys
import time
from types import FrameType
from typing import Any, Iterable, TypeAlias, TypedDict

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from hover_primitive import (  # noqa: E402
    HoverEstimatorSettlingConfig,
    HoverEstimatorSettlingReport,
    HoverLinkHealthConfig,
    HoverPreArmConfig,
    HoverPreArmSnapshot,
    HoverPrimitiveConfig,
    HoverPrimitiveOutcome,
    HoverTranslationConfig,
    StatefulHoverPrimitive,
    apply_hover_pre_arm,
    compute_takeoff_active_height_m,
    wait_for_estimator_settle,
)
from local_navigation import (  # noqa: E402
    LocalInspectNavigationPlan,
    LocalInspectNavigationPolicy,
    plan_local_inspect_navigation,
)
from on_device_failsafe import (  # noqa: E402
    ON_DEVICE_FAILSAFE_CRITICAL_BATTERY_V,
    ON_DEVICE_FAILSAFE_HEARTBEAT_TIMEOUT_S,
    ON_DEVICE_FAILSAFE_LOW_BATTERY_V,
    ON_DEVICE_FAILSAFE_MIN_UP_CLEARANCE_M,
    OnDeviceFailsafeConfig,
    OnDeviceFailsafeHeartbeatSession,
    OnDeviceFailsafeSessionReport,
    probe_on_device_failsafe,
)
from twinr.hardware.crazyflie_telemetry import (  # noqa: E402
    CompositeTraceWriter,
    CrazyflieTelemetryRuntime,
    CrazyflieTelemetrySample,
    CrazyflieTelemetrySummary,
    TelemetryProfile,
    profile_log_blocks,
    summarize_crazyflie_telemetry,
)
from run_hover_test import (  # noqa: E402
    DEFAULT_URI,
    HOVER_DEFAULT_ESTIMATOR,
    HOVER_DEFAULT_MOTION_DISABLE,
    HOVER_DEFAULT_ON_DEVICE_FAILSAFE_MODE,
    HOVER_RUNTIME_MODE_HARDWARE,
    HOVER_RUNTIME_MODE_SITL,
    HOVER_DEFAULT_CONTROLLER,
    HOVER_DEFAULT_MIN_CLEARANCE_M,
    HOVER_MAX_BATTERY_V,
    HOVER_MAX_CLEARANCE_M,
    HOVER_MAX_CONNECT_SETTLE_S,
    HOVER_MAX_ESTIMATOR_SETTLE_TIMEOUT_S,
    HOVER_MAX_FAILSAFE_HEARTBEAT_TIMEOUT_S,
    HOVER_MAX_HEIGHT_M,
    HOVER_MAX_MIN_BATTERY_LEVEL,
    HOVER_MAX_VERTICAL_VELOCITY_MPS,
    HOVER_MIN_HEIGHT_M,
    HOVER_MIN_VERTICAL_VELOCITY_MPS,
    HoverClearanceSnapshot,
    HoverPowerSnapshot,
    HoverStatusSnapshot,
    HoverWorkerTraceWriter,
    _default_required_decks_for_runtime_mode,
    _estimate_telemetry_max_samples,
    _evaluate_primitive_outcome,
    _import_cflib,
    _latest_stability_observation_from_telemetry,
    _normalize_runtime_mode,
    _power_state_name,
    _read_clearance_snapshot,
    _read_deck_flags,
    _read_preflight_snapshots,
    _stability_config_for_runtime_mode,
    evaluate_hover_preflight,
    evaluate_hover_stability,
    normalize_required_deck_name,
)

HoverTelemetryCollector: TypeAlias = CrazyflieTelemetryRuntime
HoverTelemetrySample: TypeAlias = CrazyflieTelemetrySample
HoverTelemetrySummary: TypeAlias = CrazyflieTelemetrySummary
summarize_hover_telemetry = summarize_crazyflie_telemetry

LOCAL_INSPECT_DEFAULT_HEIGHT_M = 0.25
LOCAL_INSPECT_DEFAULT_TAKEOFF_VELOCITY_MPS = 0.20
LOCAL_INSPECT_DEFAULT_LAND_VELOCITY_MPS = 0.20
LOCAL_INSPECT_DEFAULT_TRANSLATION_VELOCITY_MPS = 0.15
LOCAL_INSPECT_DEFAULT_NOMINAL_TRANSLATION_M = 0.20
LOCAL_INSPECT_DEFAULT_MIN_TRANSLATION_M = 0.10
LOCAL_INSPECT_DEFAULT_MAX_TRANSLATION_M = 0.30
LOCAL_INSPECT_DEFAULT_SETTLE_S = 0.35
LOCAL_INSPECT_DEFAULT_CAPTURE_DWELL_S = 0.30
LOCAL_INSPECT_DEFAULT_MIN_VBAT_V = 3.8
LOCAL_INSPECT_DEFAULT_MIN_BATTERY_LEVEL = 20
LOCAL_INSPECT_DEFAULT_CONNECT_SETTLE_S = 1.0


@dataclass(frozen=True)
class LocalInspectCaptureSummary:
    """Summarize the captured still produced during the inspect mission."""

    image_file: str
    source_device: str
    input_format: str | None
    content_type: str
    size_bytes: int


@dataclass(frozen=True)
class LocalInspectMissionReport:
    """Summarize one bounded local inspect mission run."""

    uri: str
    workspace: str
    target_hint: str
    capture_intent: str
    height_m: float
    deck_flags: dict[str, int | None]
    required_decks: tuple[str, ...]
    clearance_snapshot: HoverClearanceSnapshot | None
    status_snapshot: HoverStatusSnapshot | None
    pre_arm_snapshot: HoverPreArmSnapshot | None
    estimator_settle: HoverEstimatorSettlingReport | None
    power: HoverPowerSnapshot
    navigation_plan: LocalInspectNavigationPlan | None
    capture: LocalInspectCaptureSummary | None
    status: str
    completed: bool
    landed: bool
    interrupted: bool
    primitive_outcome: HoverPrimitiveOutcome | None
    telemetry: tuple[HoverTelemetrySample, ...]
    telemetry_summary: HoverTelemetrySummary | None
    failures: tuple[str, ...]
    recommendations: tuple[str, ...]
    on_device_failsafe: OnDeviceFailsafeSessionReport | None = None


class LocalInspectWorkerConfig(TypedDict):
    repo_root: Path
    env_file: Path | None
    runtime_mode: str
    uri: str
    workspace: Path
    bitcraze_python: Path
    artifact_root: Path
    image_name: str
    target_hint: str
    capture_intent: str
    height_m: float
    takeoff_velocity_mps: float
    land_velocity_mps: float
    translation_velocity_mps: float
    nominal_translation_m: float
    min_translation_m: float
    max_translation_m: float
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
    hover_settle_s: float
    capture_dwell_s: float
    required_decks: tuple[str, ...]


class GracefulSignalAbortContext:
    """Raise ``KeyboardInterrupt`` on termination signals during a mission run."""

    def __init__(self, trace_writer: HoverWorkerTraceWriter | None = None) -> None:
        self._trace_writer = trace_writer
        self._previous_handlers: dict[int, Any] = {}
        self.last_signal_name: str | None = None
        self._signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGHUP"):
            self._signals.append(signal.SIGHUP)

    def __enter__(self) -> "GracefulSignalAbortContext":
        for signum in self._signals:
            self._previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle_signal)
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        for signum, previous in self._previous_handlers.items():
            signal.signal(signum, previous)
        self._previous_handlers.clear()

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        del frame
        self.last_signal_name = signal.Signals(signum).name
        if self._trace_writer is not None:
            self._trace_writer.emit(
                "signal",
                status="error",
                message=self.last_signal_name,
                data={"signum": signum},
            )
        raise KeyboardInterrupt(self.last_signal_name)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_src_path(repo_root: Path) -> None:
    src_root = repo_root.resolve(strict=False) / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


def _safe_dir_permissions(path: Path, *, mode: int = 0o750) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        return


def _safe_file_permissions(path: Path, *, mode: int = 0o600) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        return


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _safe_dir_permissions(path.parent)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_bytes(data)
    _safe_file_permissions(temp_path)
    temp_path.replace(path)
    _safe_file_permissions(path)


def _build_twinr_config(*, repo_root: Path, env_file: Path | None) -> Any:
    _ensure_src_path(repo_root)
    from twinr.agent.base_agent.config import TwinrConfig

    if env_file is None:
        return TwinrConfig(project_root=str(repo_root))
    return TwinrConfig.from_env(env_file)


def _bridge_bitcraze_site_packages(bitcraze_python: Path) -> tuple[str, ...]:
    """Add the isolated Bitcraze workspace packages to the current interpreter."""

    command = [
        str(bitcraze_python),
        "-c",
        (
            "import json, site, sysconfig; "
            "paths = []; "
            "paths.extend(site.getsitepackages()); "
            "user = site.getusersitepackages(); "
            "paths.append(user); "
            "paths.append(sysconfig.get_path('purelib')); "
            "paths.append(sysconfig.get_path('platlib')); "
            "seen = []; "
            "for path in paths: "
            "  if path and path not in seen: "
            "    seen.append(path); "
            "print(json.dumps(seen))"
        ),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10.0,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip() or "site_path_discovery_failed"
        raise RuntimeError(f"bitcraze_site_packages_failed:{stderr}")
    try:
        paths = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("bitcraze_site_packages_invalid_json") from exc
    if not isinstance(paths, list) or not paths:
        raise RuntimeError("bitcraze_site_packages_missing")
    normalized_paths: list[str] = []
    for raw_path in paths:
        path = str(raw_path or "").strip()
        if not path:
            continue
        site.addsitedir(path)
        normalized_paths.append(path)
    if not normalized_paths:
        raise RuntimeError("bitcraze_site_packages_missing")
    return tuple(normalized_paths)


def _coerce_capture_bytes(raw: object) -> bytes:
    """Normalize one camera payload into bytes."""

    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, bytearray):
        return bytes(raw)
    if raw is None:
        return b""
    if isinstance(raw, str):
        return raw.encode("utf-8")
    if isinstance(raw, memoryview):
        return raw.tobytes()
    if not isinstance(raw, Iterable):
        raise TypeError(f"unsupported_capture_payload:{type(raw).__name__}")
    return bytes(raw)


def _build_navigation_policy(config: LocalInspectWorkerConfig) -> LocalInspectNavigationPolicy:
    """Construct one validated local inspect policy from worker config."""

    return LocalInspectNavigationPolicy(
        target_height_m=config["height_m"],
        nominal_translation_m=config["nominal_translation_m"],
        min_translation_m=config["min_translation_m"],
        max_translation_m=config["max_translation_m"],
        required_post_move_clearance_m=config["min_clearance_m"],
        translation_velocity_mps=config["translation_velocity_mps"],
        hover_settle_s=config["hover_settle_s"],
        capture_dwell_s=config["capture_dwell_s"],
    ).normalized()


def _default_local_inspect_required_decks_for_runtime_mode(runtime_mode: str) -> tuple[str, ...]:
    """Return the mission-specific required deck contract for one runtime mode."""

    if runtime_mode == HOVER_RUNTIME_MODE_SITL:
        return _default_required_decks_for_runtime_mode(runtime_mode)
    hover_required_decks = list(_default_required_decks_for_runtime_mode(runtime_mode))
    if "bcMultiranger" not in hover_required_decks:
        hover_required_decks.append("bcMultiranger")
    return tuple(hover_required_decks)


def _report_recommendations(report: LocalInspectMissionReport) -> tuple[str, ...]:
    """Render operator-facing recommendations for one mission report."""

    recommendations = list(report.failures)
    if report.status == "blocked":
        recommendations.append(
            "Resolve deck, battery, estimator, or failsafe gates before retrying the local inspect mission."
        )
    elif report.status == "interrupted":
        recommendations.append("Confirm the Crazyflie is safely on the ground before retrying the mission.")
    elif report.status == "unstable":
        recommendations.append(
            "Mission landed but stability gates failed; inspect telemetry and primitive outcome before retrying."
        )
    elif report.status == "completed":
        recommendations.append("Local inspect mission completed and landed.")
    if report.navigation_plan is not None:
        recommendations.append(report.navigation_plan.reason)
    if report.capture is not None:
        recommendations.append(
            f"Inspect capture stored as {report.capture.image_file} from {report.capture.source_device}."
        )
    if report.on_device_failsafe is not None:
        recommendations.extend(report.on_device_failsafe.failures)
    return tuple(dict.fromkeys(recommendations))


def _build_report(
    *,
    config: LocalInspectWorkerConfig,
    power: HoverPowerSnapshot,
    deck_flags: dict[str, int | None],
    clearance_snapshot: HoverClearanceSnapshot | None,
    status_snapshot: HoverStatusSnapshot | None,
    pre_arm_snapshot: HoverPreArmSnapshot | None,
    estimator_settle: HoverEstimatorSettlingReport | None,
    navigation_plan: LocalInspectNavigationPlan | None,
    capture: LocalInspectCaptureSummary | None,
    status: str,
    completed: bool,
    landed: bool,
    interrupted: bool,
    primitive_outcome: HoverPrimitiveOutcome | None,
    telemetry: tuple[HoverTelemetrySample, ...],
    telemetry_summary: HoverTelemetrySummary | None,
    failures: Iterable[str],
    on_device_failsafe: OnDeviceFailsafeSessionReport | None,
) -> LocalInspectMissionReport:
    """Build one final worker report and attach recommendations."""

    report = LocalInspectMissionReport(
        uri=config["uri"],
        workspace=str(config["workspace"]),
        target_hint=config["target_hint"],
        capture_intent=config["capture_intent"],
        height_m=config["height_m"],
        deck_flags=deck_flags,
        required_decks=config["required_decks"],
        clearance_snapshot=clearance_snapshot,
        status_snapshot=status_snapshot,
        pre_arm_snapshot=pre_arm_snapshot,
        estimator_settle=estimator_settle,
        power=power,
        navigation_plan=navigation_plan,
        capture=capture,
        status=status,
        completed=completed,
        landed=landed,
        interrupted=interrupted,
        primitive_outcome=primitive_outcome,
        telemetry=telemetry,
        telemetry_summary=telemetry_summary,
        failures=tuple(failures),
        recommendations=(),
        on_device_failsafe=on_device_failsafe,
    )
    return replace(report, recommendations=_report_recommendations(report))


def run_local_inspect_mission(
    *,
    config: LocalInspectWorkerConfig,
    trace_writer: HoverWorkerTraceWriter | None = None,
) -> LocalInspectMissionReport:
    """Execute one bounded takeoff-navigate-capture-land inspect mission."""

    trace = trace_writer or HoverWorkerTraceWriter(None)
    trace.emit(
        "run_local_inspect_mission",
        status="begin",
        data={
            "uri": config["uri"],
            "workspace": str(config["workspace"]),
            "target_hint": config["target_hint"],
            "capture_intent": config["capture_intent"],
        },
    )
    if config["runtime_mode"] == HOVER_RUNTIME_MODE_HARDWARE:
        trace.emit(
            "run_local_inspect_mission",
            status="blocked",
            message=(
                "hardware local inspect remains disabled until it is migrated onto "
                "the on-device twinrFs mission lane"
            ),
        )
        return _build_report(
            config=config,
            power=HoverPowerSnapshot(vbat_v=None, battery_level=None, state=None),
            deck_flags={},
            clearance_snapshot=None,
            status_snapshot=None,
            pre_arm_snapshot=None,
            estimator_settle=None,
            navigation_plan=None,
            capture=None,
            status="blocked",
            completed=False,
            landed=False,
            interrupted=False,
            primitive_outcome=None,
            telemetry=(),
            telemetry_summary=None,
            failures=(
                "hardware local inspect is blocked until the mission is migrated "
                "onto the on-device twinrFs hover/control lane",
            ),
            on_device_failsafe=None,
        )

    repo_root = config["repo_root"].resolve(strict=False)
    workspace = config["workspace"].expanduser().resolve(strict=False)
    workspace.mkdir(parents=True, exist_ok=True)
    cache_dir = workspace / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = config["artifact_root"].expanduser().resolve(strict=False)
    artifact_root.mkdir(parents=True, exist_ok=True)
    _safe_dir_permissions(artifact_root)

    trace.emit("bitcraze_site_packages", status="begin")
    bitcraze_paths = _bridge_bitcraze_site_packages(config["bitcraze_python"])
    trace.emit(
        "bitcraze_site_packages",
        status="done",
        data={"path_count": len(bitcraze_paths)},
    )

    twinr_config = _build_twinr_config(repo_root=repo_root, env_file=config["env_file"])
    from twinr.hardware.camera import V4L2StillCamera

    navigation_policy = _build_navigation_policy(config)
    crtp, crazyflie_cls, log_config_cls, multiranger_cls, sync_crazyflie_cls, sync_logger_cls = _import_cflib()

    deck_flags: dict[str, int | None] = {}
    power = HoverPowerSnapshot(vbat_v=None, battery_level=None, state=None)
    status_snapshot: HoverStatusSnapshot | None = None
    clearance_snapshot: HoverClearanceSnapshot | None = None
    pre_arm_snapshot: HoverPreArmSnapshot | None = None
    estimator_settle: HoverEstimatorSettlingReport | None = None
    navigation_plan: LocalInspectNavigationPlan | None = None
    capture_summary: LocalInspectCaptureSummary | None = None
    primitive_outcome: HoverPrimitiveOutcome | None = None
    telemetry_samples: tuple[HoverTelemetrySample, ...] = ()
    telemetry_summary: HoverTelemetrySummary | None = None
    on_device_failsafe_report: OnDeviceFailsafeSessionReport | None = None
    primitive: StatefulHoverPrimitive | None = None
    telemetry: HoverTelemetryCollector | None = None
    on_device_failsafe_session: OnDeviceFailsafeHeartbeatSession | None = None
    in_air = False
    landed = False
    runtime_trace = trace

    def _stop_telemetry() -> None:
        if telemetry is None:
            return
        trace.emit("telemetry_stop", status="begin")
        telemetry.stop()
        trace.emit("telemetry_stop", status="done")

    def _snapshot_telemetry() -> tuple[HoverTelemetrySample, ...]:
        if telemetry is None:
            return ()
        trace.emit("telemetry_snapshot", status="begin")
        samples = telemetry.snapshot()
        trace.emit("telemetry_snapshot", status="done", data={"sample_count": len(samples)})
        return samples

    trace.emit("crtp_init", status="begin")
    crtp.init_drivers()
    trace.emit("crtp_init", status="done")

    crazyflie = crazyflie_cls(rw_cache=str(cache_dir))
    sync_context = sync_crazyflie_cls(config["uri"], cf=crazyflie)
    sync_cf = None
    entered_sync_context = False

    try:
        trace.emit("sync_connect", status="begin", data={"uri": config["uri"]})
        sync_cf = sync_context.__enter__()
        entered_sync_context = True
        trace.emit("sync_connect", status="done")

        if config["connect_settle_s"] > 0:
            trace.emit("connect_settle", status="begin", data={"sleep_s": config["connect_settle_s"]})
            time.sleep(config["connect_settle_s"])
            trace.emit("connect_settle", status="done")

        trace.emit("deck_flags", status="begin")
        deck_flags = _read_deck_flags(sync_cf, ("bcFlow2", "bcZRanger2", "bcMultiranger", "bcAI"))
        trace.emit("deck_flags", status="done", data={"deck_flags": deck_flags})

        if deck_flags.get("bcMultiranger") == 1:
            trace.emit("clearance_snapshot", status="begin")
            clearance_snapshot = _read_clearance_snapshot(sync_cf, multiranger_cls)
            trace.emit("clearance_snapshot", status="done", data=asdict(clearance_snapshot))
        else:
            trace.emit("clearance_snapshot", status="blocked", message="bcMultiranger is required")
            return _build_report(
                config=config,
                power=power,
                deck_flags=deck_flags,
                clearance_snapshot=None,
                status_snapshot=None,
                pre_arm_snapshot=None,
                estimator_settle=None,
                navigation_plan=None,
                capture=None,
                status="blocked",
                completed=False,
                landed=False,
                interrupted=False,
                primitive_outcome=None,
                telemetry=(),
                telemetry_summary=None,
                failures=("required deck bcMultiranger is not detected",),
                on_device_failsafe=None,
            )

        trace.emit("preflight_snapshot", status="begin")
        power, status_snapshot = _read_preflight_snapshots(
            sync_cf,
            log_config_cls,
            runtime_mode=config["runtime_mode"],
        )
        trace.emit(
            "preflight_snapshot",
            status="done",
            data={
                "vbat_v": power.vbat_v,
                "battery_level": power.battery_level,
                "power_state": power.state,
                "power_state_name": _power_state_name(power.state),
            },
        )
        lateral_clearance_arm_height_m = compute_takeoff_active_height_m(config["height_m"])
        preflight_failures = evaluate_hover_preflight(
            runtime_mode=config["runtime_mode"],
            deck_flags=deck_flags,
            required_decks=config["required_decks"],
            power=power,
            status_snapshot=status_snapshot,
            clearance_snapshot=clearance_snapshot,
            min_vbat_v=config["min_vbat_v"],
            min_battery_level=config["min_battery_level"],
            min_clearance_m=config["min_clearance_m"],
            lateral_clearance_arm_height_m=lateral_clearance_arm_height_m,
        )
        if preflight_failures:
            trace.emit("preflight", status="blocked", data={"failures": preflight_failures})
            return _build_report(
                config=config,
                power=power,
                deck_flags=deck_flags,
                clearance_snapshot=clearance_snapshot,
                status_snapshot=status_snapshot,
                pre_arm_snapshot=None,
                estimator_settle=None,
                navigation_plan=None,
                capture=None,
                status="blocked",
                completed=False,
                landed=False,
                interrupted=False,
                primitive_outcome=None,
                telemetry=(),
                telemetry_summary=None,
                failures=preflight_failures,
                on_device_failsafe=None,
            )

        telemetry_max_samples = _estimate_telemetry_max_samples(
            height_m=config["height_m"],
            hover_duration_s=max(config["capture_dwell_s"], config["hover_settle_s"]),
            takeoff_velocity_mps=config["takeoff_velocity_mps"],
            land_velocity_mps=config["land_velocity_mps"],
            period_in_ms=100,
            block_count=len(profile_log_blocks(TelemetryProfile.INSPECT_LOCAL_ZONE)),
        )
        telemetry = CrazyflieTelemetryRuntime(
            sync_cf,
            log_config_cls,
            profile=TelemetryProfile.INSPECT_LOCAL_ZONE,
            max_samples=telemetry_max_samples,
            period_in_ms=100,
        )
        runtime_trace = CompositeTraceWriter(trace, telemetry)

        pre_arm_snapshot = apply_hover_pre_arm(
            sync_cf,
            config=HoverPreArmConfig(
                estimator=config["stabilizer_estimator"],
                controller=config["stabilizer_controller"],
                motion_disable=config["motion_disable"],
            ),
            trace_writer=runtime_trace,
        )
        if pre_arm_snapshot.failures:
            return _build_report(
                config=config,
                power=power,
                deck_flags=deck_flags,
                clearance_snapshot=clearance_snapshot,
                status_snapshot=status_snapshot,
                pre_arm_snapshot=pre_arm_snapshot,
                estimator_settle=None,
                navigation_plan=None,
                capture=None,
                status="blocked",
                completed=False,
                landed=False,
                interrupted=False,
                primitive_outcome=None,
                telemetry=(),
                telemetry_summary=None,
                failures=pre_arm_snapshot.failures,
                on_device_failsafe=None,
            )

        estimator_settle = wait_for_estimator_settle(
            sync_cf,
            log_config_cls,
            sync_logger_cls,
            config=HoverEstimatorSettlingConfig(timeout_s=config["estimator_settle_timeout_s"]),
            trace_writer=runtime_trace,
        )
        if not estimator_settle.stable:
            return _build_report(
                config=config,
                power=power,
                deck_flags=deck_flags,
                clearance_snapshot=clearance_snapshot,
                status_snapshot=status_snapshot,
                pre_arm_snapshot=pre_arm_snapshot,
                estimator_settle=estimator_settle,
                navigation_plan=None,
                capture=None,
                status="blocked",
                completed=False,
                landed=False,
                interrupted=False,
                primitive_outcome=None,
                telemetry=(),
                telemetry_summary=None,
                failures=estimator_settle.failures,
                on_device_failsafe=None,
            )

        if config["on_device_failsafe_mode"] != "off":
            trace.emit("on_device_failsafe_probe", status="begin", data={"mode": config["on_device_failsafe_mode"]})
            availability = probe_on_device_failsafe(sync_cf)
            trace.emit(
                "on_device_failsafe_probe",
                status="done",
                data={
                    "loaded": availability.loaded,
                    "protocol_version": availability.protocol_version,
                    "state": availability.state_name,
                    "reason": availability.reason_name,
                },
            )
            on_device_failsafe_config = OnDeviceFailsafeConfig(
                heartbeat_timeout_s=config["on_device_failsafe_heartbeat_timeout_s"],
                low_battery_v=config["on_device_failsafe_low_battery_v"],
                critical_battery_v=config["on_device_failsafe_critical_battery_v"],
                min_clearance_m=config["min_clearance_m"],
                min_up_clearance_m=config["on_device_failsafe_min_up_clearance_m"],
                arm_lateral_clearance=False,
            )
            on_device_failsafe_report = OnDeviceFailsafeSessionReport(
                mode=config["on_device_failsafe_mode"],
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
            if not availability.loaded and config["on_device_failsafe_mode"] == "required":
                return _build_report(
                    config=config,
                    power=power,
                    deck_flags=deck_flags,
                    clearance_snapshot=clearance_snapshot,
                    status_snapshot=status_snapshot,
                    pre_arm_snapshot=pre_arm_snapshot,
                    estimator_settle=estimator_settle,
                    navigation_plan=None,
                    capture=None,
                    status="blocked",
                    completed=False,
                    landed=False,
                    interrupted=False,
                    primitive_outcome=None,
                    telemetry=(),
                    telemetry_summary=None,
                    failures=(
                        "required on-device failsafe app `twinrFs` is not loaded on the Crazyflie firmware",
                    ),
                    on_device_failsafe=on_device_failsafe_report,
                )
            if availability.loaded:
                on_device_failsafe_session = OnDeviceFailsafeHeartbeatSession(
                    sync_cf,
                    mode=config["on_device_failsafe_mode"],
                    config=on_device_failsafe_config,
                    availability=availability,
                    trace_writer=runtime_trace,
                )
                on_device_failsafe_session.start()
                on_device_failsafe_report = on_device_failsafe_session.report()

        trace.emit("telemetry_start", status="begin")
        telemetry.start()
        trace.emit(
            "telemetry_start",
            status="done",
            data={"available_blocks": telemetry.available_blocks, "skipped_blocks": telemetry.skipped_blocks},
        )

        navigation_plan = plan_local_inspect_navigation(
            clearance_snapshot=asdict(clearance_snapshot),
            policy=navigation_policy,
        )
        trace.emit(
            "navigation_plan",
            status="done",
            data={
                "decision": navigation_plan.decision,
                "reason": navigation_plan.reason,
                "candidate_count": len(navigation_plan.candidates),
            },
        )

        primitive = StatefulHoverPrimitive(
            sync_cf,
            ground_distance_provider=telemetry.ground_distance_observation,
            stability_provider=lambda: _latest_stability_observation_from_telemetry(telemetry),
            link_health_provider=telemetry.link_health_observation,
            trace_writer=runtime_trace,
        )
        stability_config = _stability_config_for_runtime_mode(config["runtime_mode"])
        landing_config = HoverPrimitiveConfig(
            target_height_m=config["height_m"],
            hover_duration_s=0.0,
            takeoff_velocity_mps=config["takeoff_velocity_mps"],
            land_velocity_mps=config["land_velocity_mps"],
            link_health=HoverLinkHealthConfig(),
        )
        primitive.takeoff(
            target_height_m=navigation_plan.target_height_m,
            velocity_mps=config["takeoff_velocity_mps"],
            stability_config=stability_config,
            link_health_config=HoverLinkHealthConfig(),
        )
        if on_device_failsafe_session is not None:
            trace.emit(
                "on_device_failsafe_lateral_clearance",
                status="begin",
                data={"armed": True, "active_height_m": lateral_clearance_arm_height_m},
            )
            on_device_failsafe_session.set_lateral_clearance_armed(True)
            trace.emit(
                "on_device_failsafe_lateral_clearance",
                status="done",
                data={"armed": True, "active_height_m": lateral_clearance_arm_height_m},
            )
        in_air = True
        primitive.stabilize_hover(
            target_height_m=navigation_plan.target_height_m,
            stability_config=stability_config,
            link_health_config=HoverLinkHealthConfig(),
        )
        primitive.hold_for(
            navigation_plan.hover_settle_s,
            link_health_config=HoverLinkHealthConfig(),
            stability_config=stability_config,
        )

        selected_translation = navigation_plan.selected_translation
        if selected_translation is not None:
            primitive.translate_body(
                HoverTranslationConfig(
                    forward_m=selected_translation.forward_m,
                    left_m=selected_translation.left_m,
                    velocity_mps=navigation_plan.translation_velocity_mps,
                    target_height_m=navigation_plan.target_height_m,
                    settle_duration_s=navigation_plan.hover_settle_s,
                    link_health=HoverLinkHealthConfig(),
                    stability=stability_config,
                )
            )

        if navigation_plan.capture_dwell_s > 0:
            primitive.hold_for(
                navigation_plan.capture_dwell_s,
                link_health_config=HoverLinkHealthConfig(),
                stability_config=stability_config,
            )

        trace.emit("camera_capture", status="begin", data={"image_name": config["image_name"]})
        camera = V4L2StillCamera.from_config(twinr_config)
        capture = camera.capture_photo(filename=config["image_name"])
        capture_bytes = _coerce_capture_bytes(getattr(capture, "data", b""))
        _atomic_write_bytes(artifact_root / config["image_name"], capture_bytes)
        capture_summary = LocalInspectCaptureSummary(
            image_file=config["image_name"],
            source_device=str(getattr(capture, "source_device", "")),
            input_format=getattr(capture, "input_format", None),
            content_type=str(getattr(capture, "content_type", "application/octet-stream")),
            size_bytes=len(capture_bytes),
        )
        trace.emit(
            "camera_capture",
            status="done",
            data={
                "source_device": capture_summary.source_device,
                "input_format": capture_summary.input_format,
                "size_bytes": capture_summary.size_bytes,
            },
        )

        primitive.land(landing_config)
        landed = primitive.landed
        primitive_outcome = primitive.current_outcome(
            final_phase="captured_and_landed",
            aborted=False,
            abort_reason=None,
        )
        _stop_telemetry()
        telemetry_samples = _snapshot_telemetry()
        telemetry_summary = summarize_hover_telemetry(
            telemetry_samples,
            available_blocks=telemetry.available_blocks if telemetry is not None else (),
            skipped_blocks=telemetry.skipped_blocks if telemetry is not None else (),
        )
        stability_failures = evaluate_hover_stability(
            telemetry_summary,
            target_height_m=config["height_m"],
            runtime_mode=config["runtime_mode"],
        )
        primitive_failures = _evaluate_primitive_outcome(primitive_outcome)
        all_failures = tuple(primitive_failures + stability_failures)
        status = "completed" if not all_failures else "unstable"
        return _build_report(
            config=config,
            power=power,
            deck_flags=deck_flags,
            clearance_snapshot=clearance_snapshot,
            status_snapshot=status_snapshot,
            pre_arm_snapshot=pre_arm_snapshot,
            estimator_settle=estimator_settle,
            navigation_plan=navigation_plan,
            capture=capture_summary,
            status=status,
            completed=not all_failures,
            landed=landed,
            interrupted=False,
            primitive_outcome=primitive_outcome,
            telemetry=telemetry_samples,
            telemetry_summary=telemetry_summary,
            failures=all_failures,
            on_device_failsafe=on_device_failsafe_report,
        )
    except KeyboardInterrupt:
        trace.emit("mission_interrupt", status="error", message="KeyboardInterrupt")
        interrupted = True
        failures = ("local inspect mission interrupted; landing requested",)
        status = "interrupted"
    except Exception as exc:
        trace.emit("run_local_inspect_mission_exception", status="error", message=f"{exc.__class__.__name__}:{exc}")
        interrupted = False
        failures = (f"local inspect mission failed: {exc.__class__.__name__}:{exc}",)
        status = "failed"
    finally:
        if primitive is not None and in_air and not primitive.landed:
            landing_config = HoverPrimitiveConfig(
                target_height_m=config["height_m"],
                hover_duration_s=0.0,
                takeoff_velocity_mps=config["takeoff_velocity_mps"],
                land_velocity_mps=config["land_velocity_mps"],
                link_health=HoverLinkHealthConfig(),
            )
            trace.emit("forced_landing", status="begin", data={"status": locals().get("status", "unknown")})
            primitive.land(landing_config)
            landed = primitive.landed
            trace.emit("forced_landing", status="done", data={"landed": landed})
        if telemetry is not None:
            try:
                _stop_telemetry()
            except Exception as exc:
                trace.emit("telemetry_stop", status="error", message=f"{exc.__class__.__name__}:{exc}")
        telemetry_samples = _snapshot_telemetry()
        telemetry_summary = summarize_hover_telemetry(
            telemetry_samples,
            available_blocks=telemetry.available_blocks if telemetry is not None else (),
            skipped_blocks=telemetry.skipped_blocks if telemetry is not None else (),
        )
        if on_device_failsafe_session is not None:
            disable_on_close = bool(landed or not in_air)
            trace.emit("on_device_failsafe_release", status="begin", data={"disable_on_close": disable_on_close})
            on_device_failsafe_session.close(disable=disable_on_close)
            on_device_failsafe_report = on_device_failsafe_session.report()
            trace.emit("on_device_failsafe_release", status="done")
        if entered_sync_context:
            trace.emit("sync_disconnect", status="begin")
            sync_context.__exit__(*sys.exc_info())
            trace.emit("sync_disconnect", status="done")

    if primitive is not None:
        primitive_outcome = primitive.current_outcome(
            final_phase="interrupted_landed" if status == "interrupted" and landed else "error_landed" if landed else "error",
            aborted=status == "failed",
            abort_reason=failures[0] if failures else None,
        )
    return _build_report(
        config=config,
        power=power,
        deck_flags=deck_flags,
        clearance_snapshot=clearance_snapshot,
        status_snapshot=status_snapshot,
        pre_arm_snapshot=pre_arm_snapshot,
        estimator_settle=estimator_settle,
        navigation_plan=navigation_plan,
        capture=capture_summary,
        status=status,
        completed=False,
        landed=landed,
        interrupted=interrupted,
        primitive_outcome=primitive_outcome,
        telemetry=telemetry_samples,
        telemetry_summary=telemetry_summary,
        failures=failures,
        on_device_failsafe=on_device_failsafe_report,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a bounded Crazyflie local inspect mission.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2], help="Leading repo root.")
    parser.add_argument("--env-file", type=Path, help="Optional Twinr env file for camera configuration.")
    parser.add_argument(
        "--runtime-mode",
        choices=(HOVER_RUNTIME_MODE_HARDWARE, HOVER_RUNTIME_MODE_SITL),
        default=HOVER_RUNTIME_MODE_HARDWARE,
        help="Mission runtime mode: real hardware or CrazySim SITL (default: hardware).",
    )
    parser.add_argument("--uri", default=DEFAULT_URI, help="Crazyflie radio URI.")
    parser.add_argument("--workspace", default="/twinr/bitcraze", help="Bitcraze workspace root.")
    parser.add_argument("--bitcraze-python", type=Path, required=True, help="Interpreter from the Bitcraze workspace venv.")
    parser.add_argument("--artifact-root", type=Path, required=True, help="Artifact root for captured images and traces.")
    parser.add_argument("--image-name", required=True, help="Relative filename for the captured still image.")
    parser.add_argument("--target-hint", default="local inspect zone", help="High-level target description for the mission artifact.")
    parser.add_argument("--capture-intent", default="scene", choices=("scene", "object_check", "look_closer"), help="Inspect capture intent.")
    parser.add_argument("--height-m", type=float, default=LOCAL_INSPECT_DEFAULT_HEIGHT_M, help="Mission hover height in meters.")
    parser.add_argument("--takeoff-velocity-mps", type=float, default=LOCAL_INSPECT_DEFAULT_TAKEOFF_VELOCITY_MPS, help="Takeoff velocity in meters per second.")
    parser.add_argument("--land-velocity-mps", type=float, default=LOCAL_INSPECT_DEFAULT_LAND_VELOCITY_MPS, help="Landing velocity in meters per second.")
    parser.add_argument("--translation-velocity-mps", type=float, default=LOCAL_INSPECT_DEFAULT_TRANSLATION_VELOCITY_MPS, help="Horizontal translation velocity in meters per second.")
    parser.add_argument("--nominal-translation-m", type=float, default=LOCAL_INSPECT_DEFAULT_NOMINAL_TRANSLATION_M, help="Preferred translation length in meters.")
    parser.add_argument("--min-translation-m", type=float, default=LOCAL_INSPECT_DEFAULT_MIN_TRANSLATION_M, help="Minimum accepted translation length in meters.")
    parser.add_argument("--max-translation-m", type=float, default=LOCAL_INSPECT_DEFAULT_MAX_TRANSLATION_M, help="Maximum translation length in meters.")
    parser.add_argument("--connect-settle-s", type=float, default=LOCAL_INSPECT_DEFAULT_CONNECT_SETTLE_S, help="Initial wait after radio connect.")
    parser.add_argument("--min-vbat-v", type=float, default=LOCAL_INSPECT_DEFAULT_MIN_VBAT_V, help="Minimum battery voltage gate.")
    parser.add_argument("--min-battery-level", type=int, default=LOCAL_INSPECT_DEFAULT_MIN_BATTERY_LEVEL, help="Minimum battery level gate.")
    parser.add_argument("--min-clearance-m", type=float, default=HOVER_DEFAULT_MIN_CLEARANCE_M, help="Minimum post-move clearance contract.")
    parser.add_argument("--stabilizer-estimator", type=int, default=HOVER_DEFAULT_ESTIMATOR, help="Deterministic stabilizer.estimator value.")
    parser.add_argument("--stabilizer-controller", type=int, default=HOVER_DEFAULT_CONTROLLER, help="Deterministic stabilizer.controller value.")
    parser.add_argument("--motion-disable", type=int, default=HOVER_DEFAULT_MOTION_DISABLE, help="Deterministic motion.disable value.")
    parser.add_argument("--estimator-settle-timeout-s", type=float, default=HoverEstimatorSettlingConfig().timeout_s, help="Maximum estimator-settle wait before takeoff.")
    parser.add_argument("--on-device-failsafe-mode", choices=("required", "preferred", "off"), default=HOVER_DEFAULT_ON_DEVICE_FAILSAFE_MODE, help="Whether the on-device failsafe app is required.")
    parser.add_argument("--on-device-failsafe-heartbeat-timeout-s", type=float, default=ON_DEVICE_FAILSAFE_HEARTBEAT_TIMEOUT_S, help="Heartbeat timeout for the on-device failsafe app.")
    parser.add_argument("--on-device-failsafe-low-battery-v", type=float, default=ON_DEVICE_FAILSAFE_LOW_BATTERY_V, help="Low-battery threshold for the on-device failsafe app.")
    parser.add_argument("--on-device-failsafe-critical-battery-v", type=float, default=ON_DEVICE_FAILSAFE_CRITICAL_BATTERY_V, help="Critical-battery threshold for the on-device failsafe app.")
    parser.add_argument("--on-device-failsafe-min-up-clearance-m", type=float, default=ON_DEVICE_FAILSAFE_MIN_UP_CLEARANCE_M, help="Upward-clearance threshold for the on-device failsafe app.")
    parser.add_argument("--hover-settle-s", type=float, default=LOCAL_INSPECT_DEFAULT_SETTLE_S, help="Bounded settle time after takeoff and translation.")
    parser.add_argument("--capture-dwell-s", type=float, default=LOCAL_INSPECT_DEFAULT_CAPTURE_DWELL_S, help="Hover dwell time immediately before capture.")
    parser.add_argument("--require-deck", action="append", default=[], help="Require one deck flag; repeat as needed.")
    parser.add_argument("--trace-file", default="", help="Optional JSONL trace path.")
    parser.add_argument("--json", action="store_true", help="Emit the full report as JSON.")
    return parser


def _validate_bounded_number(
    name: str,
    raw_value: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    unit: str = "",
) -> float:
    value = float(raw_value)
    unit_suffix = f" {unit}" if unit else ""
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum:.2f}{unit_suffix}; got {value:.2f}{unit_suffix}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum:.2f}{unit_suffix}; got {value:.2f}{unit_suffix}")
    return value


def _validate_runtime_config(args: argparse.Namespace) -> LocalInspectWorkerConfig:
    runtime_mode = _normalize_runtime_mode(getattr(args, "runtime_mode", HOVER_RUNTIME_MODE_HARDWARE))
    normalized_decks: list[str] = []
    for raw_name in args.require_deck or []:
        if not str(raw_name or "").strip():
            continue
        deck_name = normalize_required_deck_name(raw_name)
        if deck_name not in normalized_decks:
            normalized_decks.append(deck_name)
    required_decks = tuple(normalized_decks) or _default_local_inspect_required_decks_for_runtime_mode(runtime_mode)
    height_m = _validate_bounded_number("height_m", args.height_m, minimum=HOVER_MIN_HEIGHT_M, maximum=HOVER_MAX_HEIGHT_M, unit="m")
    takeoff_velocity_mps = _validate_bounded_number("takeoff_velocity_mps", args.takeoff_velocity_mps, minimum=HOVER_MIN_VERTICAL_VELOCITY_MPS, maximum=HOVER_MAX_VERTICAL_VELOCITY_MPS, unit="m/s")
    land_velocity_mps = _validate_bounded_number("land_velocity_mps", args.land_velocity_mps, minimum=HOVER_MIN_VERTICAL_VELOCITY_MPS, maximum=HOVER_MAX_VERTICAL_VELOCITY_MPS, unit="m/s")
    translation_velocity_mps = _validate_bounded_number("translation_velocity_mps", args.translation_velocity_mps, minimum=0.05, maximum=HOVER_MAX_VERTICAL_VELOCITY_MPS, unit="m/s")
    nominal_translation_m = _validate_bounded_number("nominal_translation_m", args.nominal_translation_m, minimum=0.05, maximum=HOVER_MAX_CLEARANCE_M, unit="m")
    min_translation_m = _validate_bounded_number("min_translation_m", args.min_translation_m, minimum=0.05, maximum=HOVER_MAX_CLEARANCE_M, unit="m")
    max_translation_m = _validate_bounded_number("max_translation_m", args.max_translation_m, minimum=min_translation_m, maximum=HOVER_MAX_CLEARANCE_M, unit="m")
    connect_settle_s = _validate_bounded_number("connect_settle_s", args.connect_settle_s, minimum=0.0, maximum=HOVER_MAX_CONNECT_SETTLE_S, unit="s")
    min_vbat_v = _validate_bounded_number("min_vbat_v", args.min_vbat_v, minimum=0.0, maximum=HOVER_MAX_BATTERY_V, unit="V")
    min_battery_level = int(args.min_battery_level)
    if not 0 <= min_battery_level <= HOVER_MAX_MIN_BATTERY_LEVEL:
        raise ValueError(f"min_battery_level must be between 0 and {HOVER_MAX_MIN_BATTERY_LEVEL}; got {min_battery_level}")
    min_clearance_m = _validate_bounded_number("min_clearance_m", args.min_clearance_m, minimum=0.0, maximum=HOVER_MAX_CLEARANCE_M, unit="m")
    estimator_settle_timeout_s = _validate_bounded_number("estimator_settle_timeout_s", args.estimator_settle_timeout_s, minimum=0.5, maximum=HOVER_MAX_ESTIMATOR_SETTLE_TIMEOUT_S, unit="s")
    heartbeat_timeout_s = _validate_bounded_number("on_device_failsafe_heartbeat_timeout_s", args.on_device_failsafe_heartbeat_timeout_s, minimum=0.05, maximum=HOVER_MAX_FAILSAFE_HEARTBEAT_TIMEOUT_S, unit="s")
    failsafe_low_battery_v = _validate_bounded_number("on_device_failsafe_low_battery_v", args.on_device_failsafe_low_battery_v, minimum=0.0, maximum=HOVER_MAX_BATTERY_V, unit="V")
    failsafe_critical_battery_v = _validate_bounded_number("on_device_failsafe_critical_battery_v", args.on_device_failsafe_critical_battery_v, minimum=0.0, maximum=HOVER_MAX_BATTERY_V, unit="V")
    if failsafe_low_battery_v < failsafe_critical_battery_v:
        raise ValueError("on_device_failsafe_low_battery_v must be >= on_device_failsafe_critical_battery_v")
    failsafe_min_up_clearance_m = _validate_bounded_number("on_device_failsafe_min_up_clearance_m", args.on_device_failsafe_min_up_clearance_m, minimum=0.0, maximum=HOVER_MAX_CLEARANCE_M, unit="m")
    hover_settle_s = _validate_bounded_number("hover_settle_s", args.hover_settle_s, minimum=0.0, maximum=10.0, unit="s")
    capture_dwell_s = _validate_bounded_number("capture_dwell_s", args.capture_dwell_s, minimum=0.0, maximum=10.0, unit="s")
    return {
        "repo_root": Path(args.repo_root).resolve(strict=False),
        "env_file": None if args.env_file is None else Path(args.env_file).resolve(strict=False),
        "runtime_mode": runtime_mode,
        "uri": str(args.uri).strip() or DEFAULT_URI,
        "workspace": Path(str(args.workspace).strip() or "/twinr/bitcraze"),
        "bitcraze_python": Path(args.bitcraze_python).resolve(strict=False),
        "artifact_root": Path(args.artifact_root).resolve(strict=False),
        "image_name": str(args.image_name).strip() or "camera-capture.png",
        "target_hint": str(args.target_hint).strip() or "local inspect zone",
        "capture_intent": str(args.capture_intent).strip() or "scene",
        "height_m": height_m,
        "takeoff_velocity_mps": takeoff_velocity_mps,
        "land_velocity_mps": land_velocity_mps,
        "translation_velocity_mps": translation_velocity_mps,
        "nominal_translation_m": nominal_translation_m,
        "min_translation_m": min_translation_m,
        "max_translation_m": max_translation_m,
        "connect_settle_s": connect_settle_s,
        "min_vbat_v": min_vbat_v,
        "min_battery_level": min_battery_level,
        "min_clearance_m": min_clearance_m,
        "stabilizer_estimator": int(args.stabilizer_estimator),
        "stabilizer_controller": int(args.stabilizer_controller),
        "motion_disable": int(args.motion_disable),
        "estimator_settle_timeout_s": estimator_settle_timeout_s,
        "on_device_failsafe_mode": str(args.on_device_failsafe_mode).strip() or HOVER_DEFAULT_ON_DEVICE_FAILSAFE_MODE,
        "on_device_failsafe_heartbeat_timeout_s": heartbeat_timeout_s,
        "on_device_failsafe_low_battery_v": failsafe_low_battery_v,
        "on_device_failsafe_critical_battery_v": failsafe_critical_battery_v,
        "on_device_failsafe_min_up_clearance_m": failsafe_min_up_clearance_m,
        "hover_settle_s": hover_settle_s,
        "capture_dwell_s": capture_dwell_s,
        "required_decks": required_decks,
    }


def _print_human_report(report: LocalInspectMissionReport) -> None:
    print(f"status={report.status}")
    print(f"target_hint={report.target_hint}")
    print(f"capture_intent={report.capture_intent}")
    print(f"workspace={report.workspace}")
    print(f"height_m={report.height_m}")
    print(f"power.vbat_v={report.power.vbat_v}")
    print(f"power.battery_level={report.power.battery_level}")
    if report.navigation_plan is not None:
        print(f"navigation.decision={report.navigation_plan.decision}")
        print(f"navigation.reason={report.navigation_plan.reason}")
        if report.navigation_plan.selected_translation is not None:
            translation = report.navigation_plan.selected_translation
            print(f"navigation.direction={translation.direction}")
            print(f"navigation.forward_m={translation.forward_m}")
            print(f"navigation.left_m={translation.left_m}")
    if report.capture is not None:
        print(f"capture.image_file={report.capture.image_file}")
        print(f"capture.source_device={report.capture.source_device}")
        print(f"capture.input_format={report.capture.input_format}")
        print(f"capture.bytes={report.capture.size_bytes}")
    if report.telemetry_summary is not None:
        print(f"telemetry.sample_count={report.telemetry_summary.sample_count}")
        print(f"telemetry.zrange_observed={str(report.telemetry_summary.zrange_observed).lower()}")
        print(f"telemetry.flow_observed={str(report.telemetry_summary.flow_observed).lower()}")
    for failure in report.failures:
        print(f"failure={failure}")
    for recommendation in report.recommendations:
        print(f"recommendation={recommendation}")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        trace_writer = HoverWorkerTraceWriter(
            Path(str(args.trace_file).strip()) if str(args.trace_file).strip() else None
        )
    except Exception as exc:
        failure_message = f"invalid_trace_file:{exc.__class__.__name__}:{exc}"
        if args.json:
            print(json.dumps({"report": None, "failures": [failure_message]}, indent=2, sort_keys=True))
        else:
            print(f"failure={failure_message}")
        return 1

    trace_writer.emit("main", status="begin", data={"json": bool(args.json)})
    try:
        config = _validate_runtime_config(args)
    except Exception as exc:
        failure_message = f"invalid_local_inspect_arguments:{exc.__class__.__name__}:{exc}"
        trace_writer.emit("main", status="error", message=failure_message)
        if args.json:
            print(json.dumps({"report": None, "failures": [failure_message]}, indent=2, sort_keys=True))
        else:
            print(f"failure={failure_message}")
        trace_writer.emit("main", status="done", data={"exit_code": 1})
        return 1

    signal_context = GracefulSignalAbortContext(trace_writer)
    with signal_context:
        report = run_local_inspect_mission(config=config, trace_writer=trace_writer)

    payload = {"report": asdict(report), "failures": list(report.failures)}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_human_report(report)
    exit_code = 0 if report.status in {"completed", "unstable"} else 1
    trace_writer.emit("main", status="done", data={"exit_code": exit_code, "status": report.status})
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
