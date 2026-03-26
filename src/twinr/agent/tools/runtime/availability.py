"""Resolve runtime-available realtime tools from local integration readiness.

This module keeps capability gating separate from the static realtime tool
registry. The registry still defines the full canonical tool surface, while
this module removes tools that the current local runtime cannot safely execute.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from twinr.browser_automation import probe_browser_automation
from twinr.integrations import SmartHomeIntegrationAdapter, build_smart_home_hub_adapter
from twinr.channels.whatsapp.config import WhatsAppChannelConfig

from .registry import bind_realtime_tool_handlers, realtime_tool_names

if TYPE_CHECKING:
    from collections.abc import Sequence

    from twinr.agent.base_agent.config import TwinrConfig
    from twinr.agent.tools.runtime.registry import RealtimeToolHandler


_SMART_HOME_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "list_smart_home_entities",
        "read_smart_home_state",
        "control_smart_home_entities",
        "read_smart_home_sensor_stream",
    }
)
_BROWSER_AUTOMATION_TOOL_NAMES: frozenset[str] = frozenset({"browser_automation"})
_WHATSAPP_TOOL_NAMES: frozenset[str] = frozenset({"send_whatsapp_message"})


def available_realtime_tool_names(
    config: TwinrConfig,
    *,
    tool_names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return the canonical realtime tools that are runnable on this device.

    The local runtime must not advertise smart-home tools unless the managed
    smart-home adapter is actually ready. Advertising a tool that can only fail
    at execution time produces avoidable live-turn tool crashes.
    """

    requested_tool_names = _normalize_requested_tool_names(tool_names)
    if not requested_tool_names:
        return ()
    available_names = requested_tool_names
    if any(name in _SMART_HOME_TOOL_NAMES for name in available_names) and not _smart_home_tools_ready(config):
        available_names = tuple(name for name in available_names if name not in _SMART_HOME_TOOL_NAMES)
    if any(name in _BROWSER_AUTOMATION_TOOL_NAMES for name in available_names) and not _browser_automation_tools_ready(config):
        available_names = tuple(name for name in available_names if name not in _BROWSER_AUTOMATION_TOOL_NAMES)
    if any(name in _WHATSAPP_TOOL_NAMES for name in available_names) and not _whatsapp_tools_ready(config):
        available_names = tuple(name for name in available_names if name not in _WHATSAPP_TOOL_NAMES)
    return available_names


def bind_available_realtime_tool_handlers(
    handler_owner: object,
    *,
    config: TwinrConfig,
    tool_names: Sequence[str] | None = None,
) -> dict[str, RealtimeToolHandler]:
    """Bind only the realtime handlers that are currently safe to expose."""

    available_names = available_realtime_tool_names(config, tool_names=tool_names)
    if not available_names:
        return {}
    bound_handlers = bind_realtime_tool_handlers(handler_owner)
    return {name: bound_handlers[name] for name in available_names}


def _normalize_requested_tool_names(tool_names: Sequence[str] | None) -> tuple[str, ...]:
    raw_tool_names = realtime_tool_names() if tool_names is None else tuple(tool_names)
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_name in raw_tool_names:
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("Tool names must be non-empty strings")
        if raw_name in seen:
            continue
        seen.add(raw_name)
        normalized.append(raw_name)
    return tuple(normalized)


def _smart_home_tools_ready(config: TwinrConfig) -> bool:
    project_root = Path(str(getattr(config, "project_root", ".") or ".")).resolve()
    try:
        adapter = build_smart_home_hub_adapter(project_root)
    except Exception:
        return False
    return isinstance(adapter, SmartHomeIntegrationAdapter)


def _browser_automation_tools_ready(config: TwinrConfig) -> bool:
    project_root = Path(str(getattr(config, "project_root", ".") or ".")).resolve()
    try:
        return probe_browser_automation(config=config, project_root=project_root).available
    except Exception:
        return False


def _whatsapp_tools_ready(config: TwinrConfig) -> bool:
    try:
        whatsapp_config = WhatsAppChannelConfig.from_twinr_config(config)
    except Exception:
        return False
    return (whatsapp_config.auth_dir / "creds.json").is_file()


__all__ = [
    "available_realtime_tool_names",
    "bind_available_realtime_tool_handlers",
]
