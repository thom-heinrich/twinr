"""Launch and validate a CrazySim workspace for Twinr hover simulation.

Twinr does not vendor CrazySim. This module provides one strict adapter around
an operator-managed CrazySim checkout so replay/simulation tooling can fail
closed with actionable errors instead of guessing workspace layouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Literal, Mapping


DEFAULT_CRAZYSIM_URI = "udp://127.0.0.1:19850"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TWINR_MUJOCO_LAUNCHER = _REPO_ROOT / "hardware" / "bitcraze" / "launch_crazysim_mujoco.py"
_CRAZYSIM_GAZEBO_SINGLE_AGENT_LAUNCH = (
    "crazyflie-firmware",
    "tools",
    "crazyflie-simulation",
    "simulator_files",
    "gazebo",
    "launch",
    "sitl_singleagent.sh",
)
_CRAZYSIM_MUJOCO_SINGLE_AGENT_LAUNCH = (
    "crazyflie-firmware",
    "tools",
    "crazyflie-simulation",
    "simulator_files",
    "mujoco",
    "launch",
    "sitl_singleagent.sh",
)


class CrazySimAdapterError(RuntimeError):
    """Raise when a CrazySim checkout or launch path is invalid."""


@dataclass(frozen=True, slots=True)
class CrazySimCheckout:
    """Describe one validated CrazySim workspace checkout."""

    root: Path
    firmware_dir: Path
    gazebo_launch_script: Path
    mujoco_launch_script: Path
    uri: str = DEFAULT_CRAZYSIM_URI

    @property
    def cf2_binary(self) -> Path:
        """Return the expected CrazySim firmware binary path."""

        return self.firmware_dir / "sitl_make" / "build" / "cf2"


@dataclass(frozen=True, slots=True)
class CrazySimSingleAgentConfig:
    """Describe one bounded CrazySim single-agent launch."""

    checkout: CrazySimCheckout
    backend: Literal["gazebo", "mujoco"] = "gazebo"
    model: str = "crazyflie"
    x_m: float = 0.0
    y_m: float = 0.0
    startup_settle_s: float = 2.0
    extra_env: Mapping[str, str] = field(default_factory=dict)
    visualize: bool = False
    disturbance_spec_json: Path | None = None
    phase_trace_jsonl: Path | None = None
    disturbance_runtime_jsonl: Path | None = None
    wind_speed_mps: float = 0.0
    wind_direction_deg: float = 0.0
    gust_intensity_mps: float = 0.0
    turbulence: Literal["none", "light", "moderate", "severe"] = "none"
    mass_kg: float | None = None
    sensor_noise: bool = False
    ground_effect: bool = False
    flowdeck: bool = False
    downwash: bool = False


def resolve_crazysim_checkout(root: Path) -> CrazySimCheckout:
    """Validate one CrazySim checkout rooted at `root`."""

    resolved_root = root.expanduser().resolve(strict=True)
    firmware_dir = resolved_root / "crazyflie-firmware"
    gazebo_launch_script = resolved_root.joinpath(*_CRAZYSIM_GAZEBO_SINGLE_AGENT_LAUNCH)
    mujoco_launch_script = resolved_root.joinpath(*_CRAZYSIM_MUJOCO_SINGLE_AGENT_LAUNCH)
    failures: list[str] = []
    if not firmware_dir.is_dir():
        failures.append(f"missing CrazySim firmware checkout: {firmware_dir}")
    if not gazebo_launch_script.is_file():
        failures.append(f"missing CrazySim Gazebo single-agent launcher: {gazebo_launch_script}")
    if not mujoco_launch_script.is_file():
        failures.append(f"missing CrazySim MuJoCo single-agent launcher: {mujoco_launch_script}")
    if failures:
        raise CrazySimAdapterError("; ".join(failures))
    return CrazySimCheckout(
        root=resolved_root,
        firmware_dir=firmware_dir,
        gazebo_launch_script=gazebo_launch_script,
        mujoco_launch_script=mujoco_launch_script,
    )


def build_crazysim_single_agent_command(config: CrazySimSingleAgentConfig) -> tuple[str, ...]:
    """Build the exact single-agent launch command for one CrazySim checkout."""

    if config.backend == "gazebo":
        launch_script = config.checkout.gazebo_launch_script
        return (
            "bash",
            str(launch_script),
            "-m",
            str(config.model),
            "-x",
            f"{float(config.x_m):.3f}",
            "-y",
            f"{float(config.y_m):.3f}",
        )
    if config.backend != "mujoco":
        raise CrazySimAdapterError(f"unsupported CrazySim backend: {config.backend}")
    command: list[str] = [
        "python3",
        str(_TWINR_MUJOCO_LAUNCHER),
        "--crazysim-root",
        str(config.checkout.root),
        "--model-type",
        str(config.model),
        "--x-m",
        f"{float(config.x_m):.3f}",
        "--y-m",
        f"{float(config.y_m):.3f}",
        "--wind-speed",
        f"{float(config.wind_speed_mps):.6f}",
        "--wind-direction",
        f"{float(config.wind_direction_deg):.6f}",
        "--gust-intensity",
        f"{float(config.gust_intensity_mps):.6f}",
        "--turbulence",
        str(config.turbulence),
    ]
    if config.visualize:
        command.append("--visualize")
    if config.disturbance_spec_json is not None:
        command.extend(
            (
                "--disturbance-spec-json",
                str(Path(config.disturbance_spec_json).expanduser().resolve()),
            )
        )
    if config.phase_trace_jsonl is not None:
        command.extend(
            (
                "--phase-trace-jsonl",
                str(Path(config.phase_trace_jsonl).expanduser().resolve()),
            )
        )
    if config.disturbance_runtime_jsonl is not None:
        command.extend(
            (
                "--disturbance-runtime-jsonl",
                str(Path(config.disturbance_runtime_jsonl).expanduser().resolve()),
            )
        )
    if config.mass_kg is not None:
        command.extend(("--mass", f"{float(config.mass_kg):.6f}"))
    if config.sensor_noise:
        command.append("--sensor-noise")
    if config.ground_effect:
        command.append("--ground-effect")
    if config.flowdeck:
        command.append("--flowdeck")
    if config.downwash:
        command.append("--downwash")
    return tuple(command)


class CrazySimSingleAgentLauncher:
    """Own one CrazySim single-agent subprocess lifetime."""

    def __init__(self, config: CrazySimSingleAgentConfig) -> None:
        self._config = config
        self._process: subprocess.Popen[str] | None = None

    @property
    def uri(self) -> str:
        """Return the Twinr-facing URI for the simulated Crazyflie."""

        return self._config.checkout.uri

    @property
    def process(self) -> subprocess.Popen[str] | None:
        """Expose the underlying subprocess for diagnostics."""

        return self._process

    @staticmethod
    def _python3_has_modules(*module_names: str) -> bool:
        command = [
            "python3",
            "-c",
            "import importlib.util, sys; "
            "mods = sys.argv[1:]; "
            "missing = [name for name in mods if importlib.util.find_spec(name) is None]; "
            "raise SystemExit(0 if not missing else 1)",
            *module_names,
        ]
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def _validate_runtime_prerequisites(self) -> None:
        checkout = self._config.checkout
        failures: list[str] = []
        if not checkout.cf2_binary.is_file():
            failures.append(
                "missing CrazySim SITL firmware binary: "
                f"{checkout.cf2_binary} (build /tmp/crazysim/crazyflie-firmware/sitl_make/build/cf2 first)"
            )
        if self._config.backend == "gazebo" and shutil.which("gz") is None:
            failures.append("missing Gazebo CLI 'gz' on PATH for CrazySim Gazebo backend")
        if self._config.backend == "mujoco":
            if not self._python3_has_modules("mujoco", "tomli"):
                failures.append(
                    "missing Python packages 'mujoco' and/or 'tomli' on the system python3 path "
                    "for CrazySim MuJoCo backend"
                )
            if not _TWINR_MUJOCO_LAUNCHER.is_file():
                failures.append(f"missing Twinr MuJoCo launcher bridge: {_TWINR_MUJOCO_LAUNCHER}")
        if failures:
            raise CrazySimAdapterError("; ".join(failures))

    @staticmethod
    def _cf2_running() -> bool:
        result = subprocess.run(
            ("pgrep", "-x", "cf2"),
            check=False,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())

    @staticmethod
    def _signal_process_group(process: subprocess.Popen[str], sig: signal.Signals) -> None:
        """Signal the whole launcher process group, not only the wrapper shell."""

        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError as exc:
            if process.poll() is None:
                raise CrazySimAdapterError(
                    f"failed to signal CrazySim process group {process.pid} with {sig.name}"
                ) from exc

    def _stop_process(self, process: subprocess.Popen[str]) -> None:
        """Stop one launcher-owned process group and wait for real exit."""

        if process.poll() is None:
            self._signal_process_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._signal_process_group(process, signal.SIGKILL)
                process.wait(timeout=5)

    def start(self) -> None:
        """Launch CrazySim and fail closed if it exits immediately."""

        if self._process is not None:
            raise CrazySimAdapterError("CrazySim launcher is already running")
        self._validate_runtime_prerequisites()
        env = dict(**self._config.extra_env)
        self._process = subprocess.Popen(
            build_crazysim_single_agent_command(self._config),
            cwd=str(self._config.checkout.firmware_dir),
            env=None if not env else {**os.environ, **env},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            text=True,
        )
        settle_s = max(0.0, float(self._config.startup_settle_s))
        if settle_s > 0.0:
            time.sleep(settle_s)
        if self._process.poll() is not None:
            stdout_text, stderr_text = self._process.communicate(timeout=5)
            self._process = None
            raise CrazySimAdapterError(
                "CrazySim exited during startup: "
                f"stdout={stdout_text.strip()!r} stderr={stderr_text.strip()!r}"
            )
        if not self._cf2_running():
            self._stop_process(self._process)
            stdout_text, stderr_text = self._process.communicate(timeout=5)
            self._process = None
            raise CrazySimAdapterError(
                "CrazySim launcher stayed up but no live cf2 process appeared: "
                f"stdout={stdout_text.strip()!r} stderr={stderr_text.strip()!r}"
            )

    def stop(self) -> None:
        """Terminate the CrazySim process and fail only on real stop errors."""

        process = self._process
        self._process = None
        if process is None:
            return
        self._stop_process(process)
