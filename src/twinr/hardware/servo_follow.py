"""Expose the stable public attention-servo API over the decomposed stack.

This module is intentionally thin. The implementation now lives under
``twinr.hardware.servo_follow_impl`` so the public import path stays stable
while the follow stack remains separated by concern.
"""

from __future__ import annotations

##REFACTOR: 2026-03-27##

from twinr.hardware.servo_maestro import PololuMaestroServoPulseWriter
from twinr.hardware.servo_peer import PeerPololuMaestroServoPulseWriter

from .servo_follow_impl import (
    AttentionServoConfig,
    AttentionServoController,
    AttentionServoDecision,
    LGPIOPWMServoPulseWriter,
    LGPIOServoPulseWriter,
    PigpioServoPulseWriter,
    SysfsPWMServoPulseWriter,
    TwinrKernelServoPulseWriter,
    _SysfsPWMDescriptor,
    _default_pulse_writer_for_config,
    _detect_conflicting_servo_gpio_environment,
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
]

for _exported in (
    AttentionServoConfig,
    AttentionServoController,
    AttentionServoDecision,
    LGPIOPWMServoPulseWriter,
    LGPIOServoPulseWriter,
    PigpioServoPulseWriter,
    SysfsPWMServoPulseWriter,
    TwinrKernelServoPulseWriter,
    PololuMaestroServoPulseWriter,
    PeerPololuMaestroServoPulseWriter,
    _SysfsPWMDescriptor,
    _default_pulse_writer_for_config,
    _detect_conflicting_servo_gpio_environment,
):
    try:
        _exported.__module__ = __name__
    except (AttributeError, TypeError):
        continue
