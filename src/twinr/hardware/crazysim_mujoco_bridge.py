"""Bridge Twinr's MuJoCo launch lane into an operator-managed CrazySim checkout.

This module keeps one CrazySim simulator implementation. It does not fork or
reimplement CrazySim physics. Instead it:

1. starts the official `cf2` SITL firmware binary from the validated checkout
2. loads the official CrazySim MuJoCo Python entrypoint from that checkout
3. optionally injects one explicit physical disturbance schedule into the live
   MuJoCo `qfrc_applied` path before every physics step
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import ModuleType
from typing import TextIO, Sequence

import numpy as np

from twinr.hardware.crazyflie_sim_disturbance import (
    CrazySimDisturbancePlan,
    CrazySimDisturbanceRuntimeEvent,
    load_crazysim_disturbance_plan,
)


_DEFAULT_FIRMWARE_PORT = 19950
_DEFAULT_TIMESTEP_S = 0.001
_DEFAULT_MODEL_TYPE = "cf2x_T350"
_CRAZYSIM_MUJOCO_ENTRYPOINT = (
    "crazyflie-firmware",
    "tools",
    "crazyflie-simulation",
    "simulator_files",
    "mujoco",
    "crazysim.py",
)


class CrazySimMuJoCoBridgeError(RuntimeError):
    """Raise when the MuJoCo bridge cannot start or validate CrazySim."""


def _typed_vector3(values: Sequence[float]) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def _host_elapsed_s(host_event: dict[str, object] | None) -> float | None:
    if host_event is None:
        return None
    elapsed_s = host_event.get("elapsed_s")
    if not isinstance(elapsed_s, (int, float)):
        raise CrazySimMuJoCoBridgeError("host phase trace event is missing numeric `elapsed_s`")
    return float(elapsed_s)


class _DisturbanceRuntimeWriter:
    """Persist deterministic disturbance runtime evidence as JSONL."""

    def __init__(self, path: Path | None) -> None:
        self._path = None if path is None else Path(path).expanduser().resolve()
        self._handle: TextIO | None = None

    def emit(self, event: CrazySimDisturbanceRuntimeEvent) -> None:
        """Append one JSONL event and flush it immediately."""

        if self._path is None:
            return
        if self._handle is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("a", encoding="utf-8")
        self._handle.write(json.dumps(event.to_payload(), sort_keys=True))
        self._handle.write("\n")
        self._handle.flush()

    def close(self) -> None:
        """Close the backing file handle when it exists."""

        if self._handle is None:
            return
        self._handle.close()
        self._handle = None


class _HostPhaseTraceMonitor:
    """Tail one hover-worker trace JSONL until a specific phase event appears."""

    def __init__(self, *, path: Path, phase: str, status: str) -> None:
        self._path = Path(path).expanduser().resolve()
        self._phase = str(phase)
        self._status = str(status)
        self._offset = 0
        self._matched_event: dict[str, object] | None = None

    def poll(self) -> dict[str, object] | None:
        """Return the first matching event once it has been written to disk."""

        if self._matched_event is not None:
            return dict(self._matched_event)
        if not self._path.exists():
            return None
        with self._path.open("r", encoding="utf-8") as handle:
            handle.seek(self._offset)
            while True:
                line_start = handle.tell()
                raw_line = handle.readline()
                if not raw_line:
                    break
                if not raw_line.endswith("\n"):
                    handle.seek(line_start)
                    break
                stripped = raw_line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValueError("hover trace JSONL lines must be objects")
                phase = str(payload.get("phase") or "").strip()
                status = str(payload.get("status") or "").strip()
                if phase != self._phase or status != self._status:
                    continue
                elapsed_raw = payload.get("elapsed_s")
                if not isinstance(elapsed_raw, (int, float)):
                    raise ValueError("hover trace event is missing numeric `elapsed_s`")
                self._matched_event = {
                    "phase": phase,
                    "status": status,
                    "elapsed_s": float(elapsed_raw),
                }
                break
            self._offset = handle.tell()
        if self._matched_event is None:
            return None
        return dict(self._matched_event)


class CrazySimPhysicalDisturbanceInjector:
    """Apply one explicit disturbance schedule inside the live CrazySim step loop."""

    def __init__(
        self,
        plan: CrazySimDisturbancePlan,
        *,
        phase_trace_jsonl: Path | None = None,
        runtime_log_jsonl: Path | None = None,
    ) -> None:
        self._plan = plan
        self._activation_anchor_by_agent: dict[int, float] = {}
        self._activation_host_event_by_agent: dict[int, dict[str, object]] = {}
        self._started_pulses: set[tuple[int, str]] = set()
        self._finished_pulses: set[tuple[int, str]] = set()
        self._runtime_writer = _DisturbanceRuntimeWriter(runtime_log_jsonl)
        if self._plan.activation_mode == "after_host_phase":
            if phase_trace_jsonl is None:
                raise CrazySimMuJoCoBridgeError(
                    "after_host_phase disturbances require --phase-trace-jsonl"
                )
            assert self._plan.activation_phase is not None
            assert self._plan.activation_status is not None
            self._phase_monitor: _HostPhaseTraceMonitor | None = _HostPhaseTraceMonitor(
                path=phase_trace_jsonl,
                phase=self._plan.activation_phase,
                status=self._plan.activation_status,
            )
        else:
            self._phase_monitor = None

    def close(self) -> None:
        """Flush and close the runtime evidence stream."""

        self._runtime_writer.close()

    def _activation_anchor_s(self, agent: object) -> float | None:
        agent_id = int(getattr(agent, "agent_id"))
        existing = self._activation_anchor_by_agent.get(agent_id)
        if existing is not None:
            return existing
        if self._plan.activation_mode == "immediate":
            self._activation_anchor_by_agent[agent_id] = 0.0
            return 0.0
        body_id = int(getattr(agent, "_body_id"))
        data = getattr(agent, "data")
        z_m = float(data.xpos[body_id][2])
        host_event: dict[str, object] | None = None
        if self._plan.activation_mode == "after_host_phase":
            assert self._phase_monitor is not None
            host_event = self._phase_monitor.poll()
            if host_event is None:
                return None
        elif z_m < float(self._plan.activation_height_m):
            return None
        anchor_s = float(data.time)
        self._activation_anchor_by_agent[agent_id] = anchor_s
        if host_event is not None:
            self._activation_host_event_by_agent[agent_id] = host_event
        self._runtime_writer.emit(
            CrazySimDisturbanceRuntimeEvent(
                kind="plan_anchor",
                plan_name=self._plan.name,
                agent_id=agent_id,
                sim_time_s=anchor_s,
                z_m=z_m,
                host_phase=None if host_event is None else str(host_event["phase"]),
                host_status=None if host_event is None else str(host_event["status"]),
                host_phase_elapsed_s=(
                    _host_elapsed_s(host_event)
                ),
            )
        )
        print(
            "[twinr-crazysim] disturbance plan "
            f"`{self._plan.name}` armed for agent {agent_id} at sim_time={anchor_s:.3f}s "
            f"z_m={z_m:.3f}",
            flush=True,
        )
        return anchor_s

    def apply(self, agent: object) -> None:
        """Accumulate active world forces and body torques for one MuJoCo step."""

        anchor_s = self._activation_anchor_s(agent)
        if anchor_s is None:
            return
        agent_id = int(getattr(agent, "agent_id"))
        data = getattr(agent, "data")
        body_id = int(getattr(agent, "_body_id"))
        qvel_adr = int(getattr(agent, "_qvel_adr"))
        elapsed_s = float(data.time) - float(anchor_s)
        rotation_matrix = data.xmat[body_id].reshape(3, 3)
        total_world_force = np.zeros(3, dtype=float)
        total_body_torque = np.zeros(3, dtype=float)
        active_pulse_names: list[str] = []

        for pulse in self._plan.pulses:
            if int(pulse.target_agent) != agent_id:
                continue
            if not (float(pulse.start_s) <= elapsed_s <= float(pulse.end_s)):
                pulse_key = (agent_id, pulse.name)
                if pulse_key in self._started_pulses and pulse_key not in self._finished_pulses:
                    print(
                        "[twinr-crazysim] disturbance pulse "
                        f"`{pulse.name}` finished for agent {agent_id} at elapsed={elapsed_s:.3f}s",
                        flush=True,
                    )
                    self._finished_pulses.add(pulse_key)
                    host_event = self._activation_host_event_by_agent.get(agent_id)
                    self._runtime_writer.emit(
                        CrazySimDisturbanceRuntimeEvent(
                            kind="pulse_finished",
                            plan_name=self._plan.name,
                            agent_id=agent_id,
                            sim_time_s=float(data.time),
                            z_m=float(data.xpos[body_id][2]),
                            pulse_name=pulse.name,
                            elapsed_since_anchor_s=elapsed_s,
                            host_phase=None if host_event is None else str(host_event["phase"]),
                            host_status=None if host_event is None else str(host_event["status"]),
                            host_phase_elapsed_s=(
                                _host_elapsed_s(host_event)
                            ),
                            world_force_n=_typed_vector3(pulse.world_force_n),
                            body_force_n=_typed_vector3(pulse.body_force_n),
                            body_torque_nm=_typed_vector3(pulse.body_torque_nm),
                        )
                    )
                continue
            pulse_key = (agent_id, pulse.name)
            if pulse_key not in self._started_pulses:
                print(
                    "[twinr-crazysim] disturbance pulse "
                    f"`{pulse.name}` active for agent {agent_id} at elapsed={elapsed_s:.3f}s",
                    flush=True,
                )
                self._started_pulses.add(pulse_key)
                host_event = self._activation_host_event_by_agent.get(agent_id)
                self._runtime_writer.emit(
                    CrazySimDisturbanceRuntimeEvent(
                        kind="pulse_active",
                        plan_name=self._plan.name,
                        agent_id=agent_id,
                        sim_time_s=float(data.time),
                        z_m=float(data.xpos[body_id][2]),
                        pulse_name=pulse.name,
                        elapsed_since_anchor_s=elapsed_s,
                        host_phase=None if host_event is None else str(host_event["phase"]),
                        host_status=None if host_event is None else str(host_event["status"]),
                        host_phase_elapsed_s=(
                            _host_elapsed_s(host_event)
                        ),
                        world_force_n=_typed_vector3(pulse.world_force_n),
                        body_force_n=_typed_vector3(pulse.body_force_n),
                        body_torque_nm=_typed_vector3(pulse.body_torque_nm),
                    )
                )
            active_pulse_names.append(pulse.name)
            total_world_force += np.asarray(pulse.world_force_n, dtype=float)
            body_force = np.asarray(pulse.body_force_n, dtype=float)
            if np.any(body_force):
                total_world_force += rotation_matrix @ body_force
            total_body_torque += np.asarray(pulse.body_torque_nm, dtype=float)

        if not active_pulse_names:
            return
        data.qfrc_applied[qvel_adr : qvel_adr + 3] += total_world_force
        data.qfrc_applied[qvel_adr + 3 : qvel_adr + 6] += total_body_torque


def _load_crazysim_module(crazysim_root: Path) -> ModuleType:
    """Load the official CrazySim MuJoCo module from one validated checkout."""

    module_path = crazysim_root.joinpath(*_CRAZYSIM_MUJOCO_ENTRYPOINT)
    if not module_path.is_file():
        raise CrazySimMuJoCoBridgeError(f"missing CrazySim MuJoCo entrypoint: {module_path}")
    spec = importlib.util.spec_from_file_location("twinr_loaded_crazysim_mujoco", module_path)
    if spec is None or spec.loader is None:
        raise CrazySimMuJoCoBridgeError(f"failed to import CrazySim MuJoCo module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install_disturbance_patch(
    crazysim_module: ModuleType,
    *,
    disturbance_plan: CrazySimDisturbancePlan | None,
    phase_trace_jsonl: Path | None,
    runtime_log_jsonl: Path | None,
) -> CrazySimPhysicalDisturbanceInjector | None:
    """Monkeypatch CrazySim's aero-effects lane so disturbances stay in one loop."""

    if disturbance_plan is None:
        return None
    injector = CrazySimPhysicalDisturbanceInjector(
        disturbance_plan,
        phase_trace_jsonl=phase_trace_jsonl,
        runtime_log_jsonl=runtime_log_jsonl,
    )
    original_apply_aero_effects = crazysim_module.DroneAgent.apply_aero_effects

    def _patched_apply_aero_effects(agent: object) -> None:
        original_apply_aero_effects(agent)
        injector.apply(agent)

    crazysim_module.DroneAgent.apply_aero_effects = _patched_apply_aero_effects
    return injector


def _build_crazysim_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        "crazysim.py",
        "--model-type",
        str(args.model_type),
        "--port",
        str(int(args.port)),
        "--dt",
        f"{float(args.dt):.6f}",
    ]
    if bool(args.visualize):
        argv.append("--vis")
    if bool(args.sensor_noise):
        argv.append("--sensor-noise")
    if bool(args.ground_effect):
        argv.append("--ground-effect")
    if bool(args.flowdeck):
        argv.append("--flowdeck")
    if bool(args.downwash):
        argv.append("--downwash")
    if args.mass is not None:
        argv.extend(("--mass", f"{float(args.mass):.6f}"))
    if args.scene is not None:
        argv.extend(("--scene", str(Path(args.scene).expanduser().resolve())))
    if float(args.wind_speed) > 0.0:
        argv.extend(("--wind-speed", f"{float(args.wind_speed):.6f}"))
    if float(args.wind_direction) != 0.0:
        argv.extend(("--wind-direction", f"{float(args.wind_direction):.6f}"))
    if float(args.gust_intensity) > 0.0:
        argv.extend(("--gust-intensity", f"{float(args.gust_intensity):.6f}"))
    if str(args.turbulence) != "none":
        argv.extend(("--turbulence", str(args.turbulence)))
    argv.extend(("--", f"{float(args.x_m):.3f},{float(args.y_m):.3f}"))
    return argv


def _start_cf2_process(
    *,
    crazysim_root: Path,
    firmware_port: int,
) -> subprocess.Popen[str]:
    firmware_dir = crazysim_root / "crazyflie-firmware"
    cf2_binary = firmware_dir / "sitl_make" / "build" / "cf2"
    if not cf2_binary.is_file():
        raise CrazySimMuJoCoBridgeError(
            "missing CrazySim SITL firmware binary: "
            f"{cf2_binary} (build {firmware_dir}/sitl_make/build/cf2 first)"
        )
    subprocess.run(("pkill", "-x", "cf2"), check=False, capture_output=True, text=True)
    time.sleep(1.0)
    working_dir = firmware_dir / "sitl_make" / "build" / "0"
    working_dir.mkdir(parents=True, exist_ok=True)
    stdout_handle = (working_dir / "out.log").open("w", encoding="utf-8")
    stderr_handle = (working_dir / "error.log").open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            (str(cf2_binary), str(int(firmware_port))),
            cwd=str(working_dir),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
    except Exception:
        stdout_handle.close()
        stderr_handle.close()
        raise
    process._twinr_stdout_handle = stdout_handle  # type: ignore[attr-defined]
    process._twinr_stderr_handle = stderr_handle  # type: ignore[attr-defined]
    return process


def _stop_cf2_process(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    stdout_handle = getattr(process, "_twinr_stdout_handle", None)
    stderr_handle = getattr(process, "_twinr_stderr_handle", None)
    if stdout_handle is not None:
        stdout_handle.close()
    if stderr_handle is not None:
        stderr_handle.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crazysim-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=_DEFAULT_FIRMWARE_PORT)
    parser.add_argument("--model-type", default=_DEFAULT_MODEL_TYPE)
    parser.add_argument("--x-m", type=float, default=0.0)
    parser.add_argument("--y-m", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=_DEFAULT_TIMESTEP_S)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--mass", type=float, default=None)
    parser.add_argument("--scene", type=Path, default=None)
    parser.add_argument("--sensor-noise", action="store_true")
    parser.add_argument("--ground-effect", action="store_true")
    parser.add_argument("--flowdeck", action="store_true")
    parser.add_argument("--downwash", action="store_true")
    parser.add_argument("--wind-speed", type=float, default=0.0)
    parser.add_argument("--wind-direction", type=float, default=0.0)
    parser.add_argument("--gust-intensity", type=float, default=0.0)
    parser.add_argument(
        "--turbulence",
        choices=("none", "light", "moderate", "severe"),
        default="none",
    )
    parser.add_argument("--disturbance-spec-json", type=Path, default=None)
    parser.add_argument("--phase-trace-jsonl", type=Path, default=None)
    parser.add_argument("--disturbance-runtime-jsonl", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    crazysim_root = Path(args.crazysim_root).expanduser().resolve(strict=True)
    disturbance_plan = (
        None
        if args.disturbance_spec_json is None
        else load_crazysim_disturbance_plan(Path(args.disturbance_spec_json))
    )
    crazysim_module = _load_crazysim_module(crazysim_root)
    injector = _install_disturbance_patch(
        crazysim_module,
        disturbance_plan=disturbance_plan,
        phase_trace_jsonl=(
            None
            if args.phase_trace_jsonl is None
            else Path(args.phase_trace_jsonl)
        ),
        runtime_log_jsonl=(
            None
            if args.disturbance_runtime_jsonl is None
            else Path(args.disturbance_runtime_jsonl)
        ),
    )
    cf2_process = _start_cf2_process(
        crazysim_root=crazysim_root,
        firmware_port=int(args.port),
    )
    time.sleep(1.0)

    def _terminate(_signum: int, _frame: object | None) -> None:
        _stop_cf2_process(cf2_process)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    original_argv = list(sys.argv)
    sys.argv = _build_crazysim_argv(args)
    try:
        crazysim_module.main()
    finally:
        sys.argv = original_argv
        if injector is not None:
            injector.close()
        _stop_cf2_process(cf2_process)
    return 0
