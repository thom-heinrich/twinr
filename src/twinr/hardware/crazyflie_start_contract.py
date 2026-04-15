"""Own the bounded live start-clearance contract for Crazyflie hover.

This module keeps the start envelope out of worker/orchestration scripts. It
does not decide how to fly; it decides whether a bounded indoor hover attempt
may arm into takeoff at all given the currently observed obstacle envelope.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StartEnvelopeConfig:
    """Describe the minimum obstacle clearance required before live takeoff."""

    min_clearance_m: float
    require_lateral_clearance: bool = True
    require_up_clearance: bool = True


def lateral_clearance_gate_active(
    *,
    down_m: float | None,
    is_flying: bool | None,
    active_height_m: float,
) -> bool:
    """Return whether lateral pre-arm clearance is trustworthy enough to enforce.

    Near the floor, side-facing ToF sensors can report ground geometry instead
    of real lateral obstacles. The bounded hover lane therefore only trusts the
    lateral start envelope once the shared takeoff-active-height contract has
    been reached, or when the supervisor already reports flight.
    """

    if is_flying is True:
        return True
    if down_m is None:
        return False
    return float(down_m) >= max(0.0, float(active_height_m))


def evaluate_start_clearance_envelope(
    *,
    front_m: float | None,
    back_m: float | None,
    left_m: float | None,
    right_m: float | None,
    up_m: float | None,
    config: StartEnvelopeConfig,
) -> tuple[str, ...]:
    """Return hard pre-arm failures for the observed clearance envelope."""

    failures: list[str] = []
    directions: list[tuple[str, float | None]] = []
    if config.require_lateral_clearance:
        directions.extend(
            (
                ("front", front_m),
                ("back", back_m),
                ("left", left_m),
                ("right", right_m),
            )
        )
    if config.require_up_clearance:
        directions.append(("up", up_m))

    min_clearance_m = max(0.0, float(config.min_clearance_m))
    for direction_name, value in directions:
        if value is None:
            continue
        if float(value) < min_clearance_m:
            failures.append(
                f"{direction_name} clearance {float(value):.2f} m is below the "
                f"{min_clearance_m:.2f} m hover gate"
            )
    return tuple(failures)
