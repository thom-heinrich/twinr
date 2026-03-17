"""Expose bounded rule-composition primitives for future self_coding skills."""

from __future__ import annotations

import inspect  # AUDIT-FIX(#3): derive public signature metadata from live callables to prevent drift.
from collections.abc import Callable, Sequence
from typing import Any, NoReturn  # AUDIT-FIX(#1): make the fail-closed helper contract explicit.

from twinr.agent.self_coding.status import CapabilityRiskClass

from .base import SelfCodingModuleFunction, SelfCodingModuleSpec, runtime_unavailable


def _raise_runtime_unavailable(symbol: str) -> NoReturn:
    """Guarantee that descriptor-only entry points fail closed."""
    runtime_unavailable(symbol)  # AUDIT-FIX(#1): preserve the shared base failure path for unavailable capabilities.
    raise RuntimeError(
        f"{symbol} is unavailable in this runtime; runtime_unavailable() returned instead of raising."
    )  # AUDIT-FIX(#1): prevent silent success or wrong return values if the shared helper is misconfigured.


def _signature_text(name: str, func: Callable[..., Any]) -> str:
    """Render a stable public signature string from the live callable."""
    signature_text = str(inspect.signature(func, eval_str=True))  # AUDIT-FIX(#3): bind metadata to the live callable instead of a duplicated string.
    signature_text = signature_text.replace("collections.abc.", "").replace("typing.", "")  # AUDIT-FIX(#3): normalize 3.11 type rendering.
    return f"{name}{signature_text}"


def _resolve_rules_risk_class() -> CapabilityRiskClass:
    """Treat effectful automation control as non-LOW risk when the enum supports it."""
    for candidate_name in ("MEDIUM", "MODERATE", "HIGH"):  # AUDIT-FIX(#2): prefer a safer tier for effectful control primitives.
        candidate = getattr(CapabilityRiskClass, candidate_name, None)
        if candidate is not None:
            return candidate
    return CapabilityRiskClass.LOW  # AUDIT-FIX(#2): keep backward compatibility if older enums expose only LOW.


_RULES_RISK_CLASS = _resolve_rules_risk_class()


def when(condition: Callable[..., Any], action: Callable[..., Any], *, cooldown: str | None = None) -> str:
    """Register one condition-action rule with an optional cooldown."""

    _raise_runtime_unavailable("rules.when")  # AUDIT-FIX(#1): guarantee this effectful stub cannot fail open.


def when_all(
    conditions: Sequence[Callable[..., Any]],
    action: Callable[..., Any],
    *,
    cooldown: str | None = None,
) -> str:
    """Register one rule that requires every condition to pass."""

    _raise_runtime_unavailable("rules.when_all")  # AUDIT-FIX(#1): guarantee this effectful stub cannot fail open.


def cooldown(rule_id: str, interval: str) -> None:
    """Apply or update a cooldown window for one existing rule."""

    _raise_runtime_unavailable("rules.cooldown")  # AUDIT-FIX(#1): guarantee this effectful stub cannot appear to succeed.


def suppress(rule_id: str, duration: str) -> None:
    """Suppress one existing rule for a bounded duration."""

    _raise_runtime_unavailable("rules.suppress")  # AUDIT-FIX(#1): guarantee this effectful stub cannot appear to succeed.


MODULE_SPEC = SelfCodingModuleSpec(
    capability_id="rules",
    module_name="rules",
    summary="Compose bounded trigger and condition logic for automation-first skills.",
    risk_class=_RULES_RISK_CLASS,  # AUDIT-FIX(#2): avoid under-classifying effectful automation control as LOW when safer tiers exist.
    public_api=(
        SelfCodingModuleFunction(
            name="when",
            signature=_signature_text("when", when),  # AUDIT-FIX(#3): keep published API metadata synchronized with the live callable.
            summary="Register one condition-action rule with an optional cooldown.",
            returns="a stable rule identifier",
            effectful=True,
            tags=("effectful", "control"),
        ),
        SelfCodingModuleFunction(
            name="when_all",
            signature=_signature_text("when_all", when_all),  # AUDIT-FIX(#3): keep published API metadata synchronized with the live callable.
            summary="Register one rule that runs only when every condition passes.",
            returns="a stable rule identifier",
            effectful=True,
            tags=("effectful", "control"),
        ),
        SelfCodingModuleFunction(
            name="cooldown",
            signature=_signature_text("cooldown", cooldown),  # AUDIT-FIX(#3): keep published API metadata synchronized with the live callable.
            summary="Apply or update a bounded cooldown interval for an existing rule.",
            effectful=True,
            tags=("effectful", "control"),
        ),
        SelfCodingModuleFunction(
            name="suppress",
            signature=_signature_text("suppress", suppress),  # AUDIT-FIX(#3): keep published API metadata synchronized with the live callable.
            summary="Temporarily suppress an existing rule for a bounded duration.",
            effectful=True,
            tags=("effectful", "control"),
        ),
    ),
    tags=("control", "builtin", "rules"),
)

__all__ = ["MODULE_SPEC", "cooldown", "suppress", "when", "when_all"]