"""Centralize the background automation tool-call broker policy."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AutomationToolBrokerPolicy:
    """Decide which realtime tool handlers background automations may call."""

    allowed_tool_names: tuple[str, ...] = (
        "inspect_camera",
        "print_receipt",
        "search_live_info",
    )

    def __post_init__(self) -> None:
        normalized = tuple(sorted({str(name or "").strip() for name in self.allowed_tool_names if str(name or "").strip()}))
        object.__setattr__(self, "allowed_tool_names", normalized)

    def is_allowed(self, tool_name: str) -> bool:
        return str(tool_name or "").strip() in self.allowed_tool_names

    def resolve_handler(self, tool_executor: object, tool_name: str) -> Callable[[dict[str, Any]], Any]:
        normalized_name = str(tool_name or "").strip()
        if not self.is_allowed(normalized_name):
            raise RuntimeError(f"Automation tool_call is not allowed for tool `{normalized_name}`")
        handler = getattr(tool_executor, f"handle_{normalized_name}", None)
        if not callable(handler):
            raise RuntimeError(f"Automation tool_call handler is unavailable for tool `{normalized_name}`")
        return handler


_DEFAULT_AUTOMATION_TOOL_BROKER_POLICY = AutomationToolBrokerPolicy()


def default_automation_tool_broker_policy() -> AutomationToolBrokerPolicy:
    """Return the canonical background automation broker policy."""

    return _DEFAULT_AUTOMATION_TOOL_BROKER_POLICY
