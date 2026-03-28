"""Expose the decomposed attention-servo implementation package."""

from __future__ import annotations

from .config import AttentionServoConfig
from .controller import AttentionServoController
from .preflight import _detect_conflicting_servo_gpio_environment
from .types import AttentionServoDecision
from .writers import (
    LGPIOPWMServoPulseWriter,
    LGPIOServoPulseWriter,
    PigpioServoPulseWriter,
    SysfsPWMServoPulseWriter,
    TwinrKernelServoPulseWriter,
    _SysfsPWMDescriptor,
    _default_pulse_writer_for_config,
)

__all__ = [
    "AttentionServoConfig",
    "AttentionServoController",
    "AttentionServoDecision",
    "LGPIOPWMServoPulseWriter",
    "LGPIOServoPulseWriter",
    "PigpioServoPulseWriter",
    "SysfsPWMServoPulseWriter",
    "TwinrKernelServoPulseWriter",
    "_SysfsPWMDescriptor",
    "_default_pulse_writer_for_config",
    "_detect_conflicting_servo_gpio_environment",
]
