"""Expose passive-infrared motion primitives for future self_coding skills."""

from __future__ import annotations

from typing import NoReturn  # AUDIT-FIX(#1): Make the unavailability path explicit and type-safe.

from twinr.agent.self_coding.status import CapabilityRiskClass

from .base import SelfCodingModuleFunction, SelfCodingModuleSpec, runtime_unavailable


# AUDIT-FIX(#1): Centralize the placeholder behavior and guarantee that these wrappers never fall through with an invalid return type.
def _raise_runtime_unavailable(capability_name: str) -> NoReturn:
    """Raise the project-standard capability-unavailable error without fallthrough."""

    runtime_unavailable(capability_name)
    raise RuntimeError(
        f"runtime_unavailable({capability_name!r}) returned unexpectedly; "
        "PIR capability wrappers must not fall through."
    )


# AUDIT-FIX(#3): Document the current runtime-unavailable behavior so callers can implement graceful fallback paths.
def motion_detected() -> bool:
    """Return whether the PIR sensor currently reports motion.

    Raises the project-standard runtime-unavailable error until a PIR runtime
    backend is configured. Callers must handle capability absence.
    """

    _raise_runtime_unavailable("pir.motion_detected")


# AUDIT-FIX(#3): Document the current runtime-unavailable behavior so callers can implement graceful fallback paths.
def time_since_last_motion() -> float:
    """Return the seconds since the most recent PIR motion signal.

    Raises the project-standard runtime-unavailable error until a PIR runtime
    backend is configured. Callers must handle capability absence.
    """

    _raise_runtime_unavailable("pir.time_since_last_motion")


# AUDIT-FIX(#2): Align self-coding metadata with the real contract so generated skills do not assume a guaranteed successful sensor read.
MODULE_SPEC = SelfCodingModuleSpec(
    capability_id="pir",
    module_name="pir",
    summary=(
        "Observe passive infrared motion signals from the configured sensor "
        "when a PIR runtime backend is available."
    ),
    risk_class=CapabilityRiskClass.LOW,
    public_api=(
        SelfCodingModuleFunction(
            name="motion_detected",
            signature="motion_detected() -> bool",
            summary=(
                "Return whether the PIR sensor currently reports motion, or "
                "raise a runtime-unavailable error when no PIR backend is active."
            ),
            returns=(
                "True when motion is active and False otherwise; raises when "
                "the PIR capability is unavailable."
            ),
            tags=("read_only", "presence"),
        ),
        SelfCodingModuleFunction(
            name="time_since_last_motion",
            signature="time_since_last_motion() -> float",
            summary=(
                "Return the bounded age of the last motion event in seconds, or "
                "raise a runtime-unavailable error when no PIR backend is active."
            ),
            returns=(
                "seconds since the last accepted PIR event; raises when the PIR "
                "capability is unavailable."
            ),
            tags=("read_only", "presence"),
        ),
    ),
    tags=("sensor", "builtin", "read_only", "presence"),
)

__all__ = ["MODULE_SPEC", "motion_detected", "time_since_last_motion"]