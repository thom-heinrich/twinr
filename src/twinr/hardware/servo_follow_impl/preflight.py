"""Fail closed when the Pi GPIO environment already owns the servo line."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Callable

from .constants import (
    _CONFLICTING_GPIO_OVERLAY_NAMES,
    _DEFAULT_SERVO_DRIVER,
    _SERVO_PRECHECK_TIMEOUT_S,
)

if TYPE_CHECKING:
    from .config import AttentionServoConfig

def _run_best_effort_text_command(args: list[str], *, timeout_s: float = _SERVO_PRECHECK_TIMEOUT_S) -> str | None:
    """Return stdout for one short best-effort probe command."""

    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout

def _detect_conflicting_servo_gpio_environment(
    *,
    gpio: int,
    run_text_command: Callable[[list[str]], str | None] = _run_best_effort_text_command,
) -> list[str]:
    """Describe live GPIO conflict evidence that makes servo startup unsafe."""

    checked_gpio = int(gpio)
    gpio_token = f"gpio={checked_gpio}"
    pin_token = f"pin={checked_gpio}"
    conflicts: list[str] = []
    overlay_output = run_text_command(["dtoverlay", "-l"])
    if overlay_output:
        for raw_line in overlay_output.splitlines():
            line = raw_line.strip()
            if ":" not in line:
                continue
            descriptor = line.split(":", 1)[1].strip()
            if not descriptor or (gpio_token not in descriptor and pin_token not in descriptor):
                continue
            overlay_name = descriptor.split()[0].strip().lower()
            if overlay_name in _CONFLICTING_GPIO_OVERLAY_NAMES:
                conflicts.append(f"overlay {descriptor}")
    process_output = run_text_command(["ps", "-eo", "pid=,stat=,args="])
    if process_output:
        for raw_line in process_output.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            pid, stat, args = parts
            lowered_args = args.lower()
            if "dtoverlay" not in lowered_args:
                continue
            if "pwm-pio" in lowered_args:
                if gpio_token in lowered_args or pin_token in lowered_args:
                    conflicts.append(f"process pid={pid} stat={stat} args={args}")
                continue
            if "dtoverlay -r" in lowered_args and conflicts:
                conflicts.append(f"process pid={pid} stat={stat} args={args}")
    return conflicts

def _assert_servo_gpio_environment_ready(config: "AttentionServoConfig") -> None:
    """Fail closed when a proven conflicting GPIO overlay/process state is present."""

    driver = str(config.driver or _DEFAULT_SERVO_DRIVER).strip().lower() or _DEFAULT_SERVO_DRIVER
    if driver in {"pololu_maestro", "peer_pololu_maestro"} or getattr(config, "maestro_transport_requested", False):
        return
    if not config.enabled or config.gpio is None:
        return
    conflicts = _detect_conflicting_servo_gpio_environment(gpio=int(config.gpio))
    if not conflicts:
        return
    rendered_conflicts = "; ".join(conflicts[:4])
    if len(conflicts) > 4:
        rendered_conflicts += f"; +{len(conflicts) - 4} more"
    raise RuntimeError(
        f"Refusing attention servo startup on GPIO{int(config.gpio)} due to conflicting overlay/process state: "
        f"{rendered_conflicts}"
    )
