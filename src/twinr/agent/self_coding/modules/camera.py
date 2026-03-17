"""Expose semantic camera primitives for future self_coding skills."""

from __future__ import annotations

from typing import NoReturn  # AUDIT-FIX(#1): Make the raise-only control flow explicit and type-safe in Python 3.11.

from twinr.agent.self_coding.status import CapabilityRiskClass

from .base import SelfCodingModuleFunction, SelfCodingModuleSpec, runtime_unavailable


# AUDIT-FIX(#1): Centralize the unavailable-runtime path so these stubs can never silently fall through as None.
def _raise_runtime_unavailable(symbol: str) -> NoReturn:
    result = runtime_unavailable(symbol)
    if isinstance(result, BaseException):
        raise result
    # AUDIT-FIX(#1): Fail closed if the helper ever stops raising; returning None here would violate the advertised API contract.
    raise RuntimeError(
        f"{symbol} unexpectedly returned from runtime_unavailable(); "
        "the capability stub must raise to preserve the advertised API contract."
    )


def is_anyone_visible() -> bool:
    """Return whether at least one person is visible in the current camera view."""

    # AUDIT-FIX(#1): Route through the explicit NoReturn guard so this stub cannot accidentally return None.
    _raise_runtime_unavailable("camera.is_anyone_visible")


def count_persons() -> int:
    # AUDIT-FIX(#4): Remove the undocumented "bounded" promise; only the non-negative count contract is defined here.
    """Return the current non-negative visible-person count.

    Any upper bound is implementation-defined by the live camera backend.
    """

    # AUDIT-FIX(#1): Route through the explicit NoReturn guard so this stub cannot accidentally return None.
    _raise_runtime_unavailable("camera.count_persons")


def is_user_alone() -> bool:
    # AUDIT-FIX(#2): Document this as a heuristic privacy hint, not as a hard guarantee that spoken output is safe.
    """Return whether the camera heuristically suggests the user is alone.

    This is a best-effort privacy hint, not a hard guarantee that spoken output is private.
    """

    # AUDIT-FIX(#1): Route through the explicit NoReturn guard so this stub cannot accidentally return None.
    _raise_runtime_unavailable("camera.is_user_alone")


MODULE_SPEC = SelfCodingModuleSpec(
    capability_id="camera",
    module_name="camera",
    # AUDIT-FIX(#3): Make the runtime-gated placeholder nature explicit so registries and code generators do not read this as an always-live backend.
    summary="Describe runtime-gated semantic camera-side presence and visibility signals.",
    risk_class=CapabilityRiskClass.MODERATE,
    public_api=(
        SelfCodingModuleFunction(
            name="is_anyone_visible",
            signature="is_anyone_visible() -> bool",
            summary="Return whether the camera currently sees at least one person.",
            returns="True when a person is visible and False otherwise",
            tags=("read_only", "presence"),
        ),
        SelfCodingModuleFunction(
            name="count_persons",
            signature="count_persons() -> int",
            # AUDIT-FIX(#4): Keep the public contract aligned with the function docstring and avoid an undefined upper-bound promise.
            summary="Return the current non-negative number of visible people.",
            returns="a non-negative visible-person count; any upper bound is implementation-defined",
            tags=("read_only", "presence"),
        ),
        SelfCodingModuleFunction(
            name="is_user_alone",
            signature="is_user_alone() -> bool",
            # AUDIT-FIX(#2): Prevent generated code from treating this signal as a privacy/security boundary.
            summary="Return whether the camera heuristically suggests that no additional people are visible near the user.",
            returns="True when the view looks private enough for spoken output as a best-effort hint, not a privacy guarantee",
            tags=("read_only", "privacy"),
        ),
    ),
    tags=("sensor", "builtin", "read_only", "vision"),
)

__all__ = ["MODULE_SPEC", "count_persons", "is_anyone_visible", "is_user_alone"]