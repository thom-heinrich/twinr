"""Expose Twinr safety context and effectful self-disable hooks for self_coding skills."""  # AUDIT-FIX(#2): Clarify that this module is not purely read-only.

from __future__ import annotations

from typing import Final, NoReturn  # AUDIT-FIX(#1,#4): Make fail-closed control flow explicit and exported bindings harder to rebind accidentally.

from twinr.agent.self_coding.status import CapabilityRiskClass

from .base import SelfCodingModuleFunction, SelfCodingModuleSpec, runtime_unavailable


def _runtime_bound_unavailable(api_name: str) -> NoReturn:  # AUDIT-FIX(#1): Centralize fail-closed behavior for runtime-only safety APIs.
    """Fail closed for runtime-only safety APIs when no live runtime adapter is present."""
    runtime_unavailable(api_name)
    raise RuntimeError(f"{api_name} is unavailable in this runtime")  # AUDIT-FIX(#1): Guarantee non-return even if the shared helper is patched to return.


def night_mode() -> bool:
    """Return whether Twinr currently considers the device to be in night mode via the live runtime adapter."""  # AUDIT-FIX(#3): Surface runtime-bound semantics to introspection users.
    _runtime_bound_unavailable("safety.night_mode")  # AUDIT-FIX(#1): Preserve the bool contract by failing closed.


def log_event(event: str, severity: str = "info") -> None:
    """Write one bounded safety log event via the live runtime adapter."""  # AUDIT-FIX(#3): Surface runtime-bound semantics to introspection users.
    _runtime_bound_unavailable("safety.log_event")  # AUDIT-FIX(#1): Prevent silent no-op success for safety logging.


def disable_skill(reason: str) -> None:
    """Disable the current learned skill with one explicit reason via the live runtime adapter."""  # AUDIT-FIX(#3): Surface runtime-bound semantics to introspection users.
    _runtime_bound_unavailable("safety.disable_skill")  # AUDIT-FIX(#1): Prevent silent no-op success for self-disable safety guards.


MODULE_SPEC: Final[SelfCodingModuleSpec] = SelfCodingModuleSpec(  # AUDIT-FIX(#4): Reduce accidental rebinding of exported capability metadata.
    capability_id="safety",
    module_name="safety",
    summary="Expose Twinr safety context plus runtime-bound self-protection actions.",  # AUDIT-FIX(#2): Reflect that this module is not read-only.
    risk_class=CapabilityRiskClass.HIGH,
    public_api=(
        SelfCodingModuleFunction(
            name="night_mode",
            signature="night_mode() -> bool",
            summary="Return whether Twinr currently wants quiet, low-disturbance behavior; fails closed when the live runtime adapter is absent.",  # AUDIT-FIX(#3): Make failure semantics explicit to planners and generated code.
            returns="True when night-mode suppression is active",
            tags=("read_only", "safety"),
        ),
        SelfCodingModuleFunction(
            name="log_event",
            signature='log_event(event: str, severity: str = "info") -> None',
            summary="Record one bounded safety-relevant note for operators and health tracking; fails closed when the live runtime adapter is absent.",  # AUDIT-FIX(#3): Make failure semantics explicit to planners and generated code.
            effectful=True,
            tags=("effectful", "safety"),
        ),
        SelfCodingModuleFunction(
            name="disable_skill",
            signature="disable_skill(reason: str) -> None",
            summary="Disable the current learned skill when a hard safety guard trips; fails closed when the live runtime adapter is absent.",  # AUDIT-FIX(#3): Make failure semantics explicit to planners and generated code.
            effectful=True,
            tags=("effectful", "safety"),
        ),
    ),
    tags=("safety", "builtin", "policy"),
)

__all__: Final[tuple[str, ...]] = ("MODULE_SPEC", "disable_skill", "log_event", "night_mode")  # AUDIT-FIX(#4): Use an immutable export surface.