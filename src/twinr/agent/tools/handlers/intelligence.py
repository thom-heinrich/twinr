"""Handle RSS/world-intelligence tool calls for Twinr live sessions.

This handler exposes the explicit tool surface for configuring Twinr's
RSS-backed place/world awareness. It keeps payload validation and JSON-safe
serialization local to the tool boundary while delegating durable state changes
to the runtime-facing personality/memory services.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from twinr.agent.personality.intelligence import WorldIntelligenceConfigRequest

from .handler_telemetry import emit_best_effort, record_event_best_effort
from .support import ArgumentValidationError, optional_bool, optional_float, require_sensitive_voice_confirmation

_MAX_TEXT_LENGTH = 256


def _ensure_mapping(arguments: dict[str, object]) -> dict[str, object]:
    """Validate that one tool payload is a JSON-like object."""

    if not isinstance(arguments, Mapping):
        raise RuntimeError("Tool arguments must be an object.")
    return dict(arguments)


def _text_argument(
    arguments: Mapping[str, object],
    key: str,
    *,
    required: bool = False,
    max_length: int = _MAX_TEXT_LENGTH,
) -> str | None:
    """Read one bounded text argument from a tool payload."""

    raw_value = arguments.get(key)
    if raw_value is None:
        text = ""
    elif isinstance(raw_value, str):
        text = raw_value.strip()
    else:
        raise RuntimeError(f"Invalid `{key}` value.")
    if not text:
        if required:
            raise RuntimeError(f"Cannot continue without `{key}`.")
        return None
    if len(text) > max_length:
        raise RuntimeError(f"`{key}` exceeds the maximum supported length.")
    return text


def _string_list_argument(
    arguments: Mapping[str, object],
    key: str,
    *,
    max_items: int = 16,
) -> tuple[str, ...]:
    """Read one bounded string-list argument from a tool payload."""

    raw_value = arguments.get(key)
    if raw_value is None:
        return ()
    if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes, bytearray)):
        raise RuntimeError(f"`{key}` must be a list of strings.")
    items: list[str] = []
    for index, item in enumerate(raw_value):
        if not isinstance(item, str):
            raise RuntimeError(f"`{key}[{index}]` must be a string.")
        normalized = item.strip()
        if not normalized:
            raise RuntimeError(f"`{key}[{index}]` cannot be blank.")
        items.append(normalized)
        if len(items) >= max_items:
            break
    return tuple(items)


def _serialize_subscription(subscription: object) -> dict[str, object]:
    """Serialize one world-intelligence subscription for tool responses."""

    return {
        "subscription_id": str(getattr(subscription, "subscription_id", "")),
        "label": str(getattr(subscription, "label", "")),
        "feed_url": str(getattr(subscription, "feed_url", "")),
        "scope": str(getattr(subscription, "scope", "")),
        "region": getattr(subscription, "region", None),
        "topics": list(getattr(subscription, "topics", ()) or ()),
        "priority": getattr(subscription, "priority", None),
        "active": bool(getattr(subscription, "active", False)),
        "refresh_interval_hours": getattr(subscription, "refresh_interval_hours", None),
        "last_checked_at": getattr(subscription, "last_checked_at", None),
        "last_refreshed_at": getattr(subscription, "last_refreshed_at", None),
        "last_error": getattr(subscription, "last_error", None),
    }


def _serialize_refresh(refresh: object | None) -> dict[str, object] | None:
    """Serialize one optional refresh result for tool responses."""

    if refresh is None:
        return None
    return {
        "status": str(getattr(refresh, "status", "")),
        "refreshed": bool(getattr(refresh, "refreshed", False)),
        "refreshed_subscription_ids": list(getattr(refresh, "refreshed_subscription_ids", ()) or ()),
        "world_signal_count": len(getattr(refresh, "world_signals", ()) or ()),
        "continuity_thread_count": len(getattr(refresh, "continuity_threads", ()) or ()),
        "errors": list(getattr(refresh, "errors", ()) or ()),
        "checked_at": getattr(refresh, "checked_at", None),
    }


def _safe_emit(owner: Any, key: str, value: object) -> None:
    """Emit one best-effort telemetry key/value pair."""

    emit_best_effort(owner, f"{key}={value}")


def _safe_record_event(owner: Any, event_name: str, message: str, **metadata: object) -> None:
    """Record one best-effort tool event."""

    record_event_best_effort(owner, event_name, message, dict(metadata))


def handle_configure_world_intelligence(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Configure Twinr's RSS-backed world-intelligence subscriptions.

    Args:
        owner: Tool executor owner exposing runtime access, backend access, and
            telemetry hooks.
        arguments: Tool payload describing one explicit action such as
            subscribing feeds, discovering feeds, listing subscriptions,
            deactivating feeds, or forcing a refresh.

    Returns:
        JSON-safe payload summarizing the resulting subscriptions and any
        immediate refresh output.

    Raises:
        SensitiveActionConfirmationRequired: If persistent configuration
            changes need explicit confirmation for the current speaker.
        RuntimeError: If the payload is malformed or runtime configuration
            fails.
    """

    arguments = _ensure_mapping(arguments)
    action = _text_argument(arguments, "action", required=True) or "list"
    require_sensitive_voice_confirmation(
        owner,
        dict(arguments),
        action_label="change world intelligence subscriptions",
    )
    try:
        priority = optional_float(dict(arguments), "priority", default=0.6)
        refresh_interval_hours = optional_float(dict(arguments), "refresh_interval_hours", default=72.0)
        auto_subscribe = optional_bool(dict(arguments), "auto_subscribe", default=True)
        refresh_after_change = optional_bool(dict(arguments), "refresh_after_change", default=False)
    except ArgumentValidationError as exc:
        raise RuntimeError(str(exc)) from exc

    request = WorldIntelligenceConfigRequest(
        action=action,
        query=_text_argument(arguments, "query"),
        label=_text_argument(arguments, "label"),
        location_hint=_text_argument(arguments, "location_hint"),
        region=_text_argument(arguments, "region"),
        topics=_string_list_argument(arguments, "topics"),
        feed_urls=_string_list_argument(arguments, "feed_urls"),
        subscription_refs=_string_list_argument(arguments, "subscription_refs"),
        scope=_text_argument(arguments, "scope") or "topic",
        priority=priority,
        refresh_interval_hours=int(refresh_interval_hours),
        auto_subscribe=True if auto_subscribe is None else auto_subscribe,
        refresh_after_change=False if refresh_after_change is None else refresh_after_change,
        created_by="tool",
    )
    result = owner.runtime.configure_world_intelligence(
        request=request,
        search_backend=getattr(owner, "print_backend", None),
    )
    _safe_emit(owner, "world_intelligence_tool_call", action)
    _safe_record_event(
        owner,
        "world_intelligence_configured",
        "World intelligence subscriptions were updated.",
        action=action,
        subscription_count=len(getattr(result, "subscriptions", ()) or ()),
        discovered_feeds=len(getattr(result, "discovered_feed_urls", ()) or ()),
    )
    return {
        "status": str(getattr(result, "status", "ok")),
        "action": str(getattr(result, "action", action)),
        "subscriptions": [
            _serialize_subscription(subscription)
            for subscription in (getattr(result, "subscriptions", ()) or ())
        ],
        "subscription_count": len(getattr(result, "subscriptions", ()) or ()),
        "discovered_feed_urls": list(getattr(result, "discovered_feed_urls", ()) or ()),
        "refresh": _serialize_refresh(getattr(result, "refresh", None)),
    }
