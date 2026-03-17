"""Centralize the background automation tool-call broker policy."""

from __future__ import annotations

import inspect  # AUDIT-FIX(#5): Validate resolved handlers against the broker payload contract.
import re  # AUDIT-FIX(#1/#2): Enforce canonical safe tool-name syntax.

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field  # AUDIT-FIX(#2): Cache normalized names for constant-time membership tests.
from typing import Any


class AutomationToolBrokerPolicyError(RuntimeError):
    """Base error for automation tool broker policy failures."""  # AUDIT-FIX(#4): Give callers a stable base exception to catch.


class AutomationToolBrokerPolicyConfigError(AutomationToolBrokerPolicyError):
    """Raised when the policy is configured with invalid allowed tool names."""  # AUDIT-FIX(#1): Fail fast on bad allowlist configuration.


class AutomationToolBrokerPolicyToolNameError(AutomationToolBrokerPolicyError):
    """Raised when a requested tool name is invalid."""  # AUDIT-FIX(#2): Separate malformed input from policy denials.


class AutomationToolBrokerPolicyDeniedError(AutomationToolBrokerPolicyError):
    """Raised when a tool is not allowed by policy."""  # AUDIT-FIX(#4): Distinguish deny decisions from resolution failures.


class AutomationToolBrokerPolicyHandlerUnavailableError(AutomationToolBrokerPolicyError):
    """Raised when an allowed tool handler cannot be resolved safely."""  # AUDIT-FIX(#3/#4): Provide a predictable handler-resolution failure mode.


_TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")  # AUDIT-FIX(#1/#2): Restrict names to canonical lowercase snake_case identifiers.
_SIGNATURE_BIND_SENTINEL = object()  # AUDIT-FIX(#5): Use a single sentinel for safe signature binding checks.


def _normalize_allowed_tool_names(allowed_tool_names: Iterable[str]) -> tuple[str, ...]:
    """Validate and canonicalize the configured allowlist."""  # AUDIT-FIX(#1): Centralize strict allowlist validation.
    if isinstance(allowed_tool_names, (str, bytes)):
        raise AutomationToolBrokerPolicyConfigError(
            "allowed_tool_names must be an iterable of lowercase snake_case tool-name strings, not a single string value"
        )

    normalized_names: set[str] = set()
    for raw_name in allowed_tool_names:
        if not isinstance(raw_name, str):
            raise AutomationToolBrokerPolicyConfigError("allowed_tool_names entries must be strings")
        normalized_name = raw_name.strip()
        if not normalized_name:
            continue
        if not _TOOL_NAME_PATTERN.fullmatch(normalized_name):
            raise AutomationToolBrokerPolicyConfigError(
                "allowed_tool_names entries must use lowercase snake_case tool names"
            )
        normalized_names.add(normalized_name)

    if not normalized_names:
        raise AutomationToolBrokerPolicyConfigError(
            "allowed_tool_names must contain at least one non-empty tool name"
        )

    return tuple(sorted(normalized_names))


def _normalize_tool_name(tool_name: object) -> str | None:
    """Return a canonical tool name, or None for invalid input."""  # AUDIT-FIX(#2): Avoid arbitrary-object coercion via str().
    if not isinstance(tool_name, str):
        return None
    normalized_name = tool_name.strip()
    if not normalized_name:
        return None
    return normalized_name if _TOOL_NAME_PATTERN.fullmatch(normalized_name) else None


def _require_tool_name(tool_name: object) -> str:
    """Return a canonical tool name or raise a typed validation error."""  # AUDIT-FIX(#2/#4): Fail explicitly for malformed tool requests.
    normalized_name = _normalize_tool_name(tool_name)
    if normalized_name is None:
        raise AutomationToolBrokerPolicyToolNameError(
            "tool_name must be a non-empty lowercase snake_case string"
        )
    return normalized_name


def _validate_handler_signature(handler: Callable[[dict[str, Any]], Any], tool_name: str) -> None:
    """Ensure the resolved handler accepts one positional payload argument."""  # AUDIT-FIX(#5): Catch contract mismatches before runtime invocation.
    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):
        return

    try:
        signature.bind(_SIGNATURE_BIND_SENTINEL)
    except TypeError as exc:
        raise AutomationToolBrokerPolicyHandlerUnavailableError(
            f"Automation tool_call handler has an incompatible signature for tool `{tool_name}`"
        ) from exc


@dataclass(frozen=True, slots=True)
class AutomationToolBrokerPolicy:
    """Decide which realtime tool handlers background automations may call."""

    allowed_tool_names: tuple[str, ...] = (
        "inspect_camera",
        "print_receipt",
        "run_self_coding_skill_scheduled",
        "run_self_coding_skill_sensor",
        "search_live_info",
    )
    _allowed_tool_name_set: frozenset[str] = field(init=False, repr=False)  # AUDIT-FIX(#2): Preserve public tuple API while using normalized set membership internally.

    def __post_init__(self) -> None:
        normalized = _normalize_allowed_tool_names(self.allowed_tool_names)  # AUDIT-FIX(#1): Reject invalid allowlist entries instead of coercing them silently.
        object.__setattr__(self, "allowed_tool_names", normalized)
        object.__setattr__(self, "_allowed_tool_name_set", frozenset(normalized))  # AUDIT-FIX(#2): Cache normalized membership state once at construction time.

    def is_allowed(self, tool_name: str) -> bool:
        normalized_name = _normalize_tool_name(tool_name)  # AUDIT-FIX(#2): Treat invalid or malformed tool names as disallowed.
        return normalized_name in self._allowed_tool_name_set if normalized_name is not None else False

    def resolve_handler(self, tool_executor: object, tool_name: str) -> Callable[[dict[str, Any]], Any]:
        normalized_name = _require_tool_name(tool_name)  # AUDIT-FIX(#2): Reject malformed tool names with a typed policy error.
        if normalized_name not in self._allowed_tool_name_set:
            raise AutomationToolBrokerPolicyDeniedError(
                f"Automation tool_call is not allowed for tool `{normalized_name}`"
            )  # AUDIT-FIX(#4): Use a domain-specific deny exception.

        if tool_executor is None:
            raise AutomationToolBrokerPolicyHandlerUnavailableError(
                f"Automation tool_call handler is unavailable for tool `{normalized_name}`"
            )  # AUDIT-FIX(#3/#4): Fail predictably for a missing executor reference.

        handler_name = f"handle_{normalized_name}"
        try:
            handler = getattr(tool_executor, handler_name, None)  # AUDIT-FIX(#3): Wrap dynamic handler lookup so descriptor/__getattr__ failures do not leak raw exceptions.
        except Exception as exc:
            raise AutomationToolBrokerPolicyHandlerUnavailableError(
                f"Automation tool_call handler lookup failed for tool `{normalized_name}`"
            ) from exc

        if not callable(handler):
            raise AutomationToolBrokerPolicyHandlerUnavailableError(
                f"Automation tool_call handler is unavailable for tool `{normalized_name}`"
            )  # AUDIT-FIX(#4): Preserve clear semantics for missing or non-callable handlers.

        _validate_handler_signature(handler, normalized_name)  # AUDIT-FIX(#5): Ensure the resolved handler can actually consume the broker payload.
        return handler


_DEFAULT_AUTOMATION_TOOL_BROKER_POLICY = AutomationToolBrokerPolicy()


def default_automation_tool_broker_policy() -> AutomationToolBrokerPolicy:
    """Return the canonical background automation broker policy."""

    return _DEFAULT_AUTOMATION_TOOL_BROKER_POLICY