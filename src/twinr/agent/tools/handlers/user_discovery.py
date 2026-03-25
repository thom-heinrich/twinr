"""Handle guided user-discovery tool calls for Twinr's companion setup flow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from twinr.memory.user_discovery import UserDiscoveryFact, UserDiscoveryMemoryRoute

from .handler_telemetry import emit_best_effort, record_event_best_effort
from .support import require_sensitive_voice_confirmation

_ALLOWED_ACTIONS = frozenset(
    {
        "start_or_resume",
        "answer",
        "skip_topic",
        "pause_session",
        "snooze",
        "status",
        "review_profile",
        "replace_fact",
        "delete_fact",
    }
)
_ALLOWED_FACT_STORAGES = frozenset({"user_profile", "personality"})
_ALLOWED_ROUTE_KINDS = frozenset({"user_profile", "personality", "contact", "preference", "plan", "durable_memory"})
_MAX_ACTION_LENGTH = 32
_MAX_TOPIC_ID_LENGTH = 48
_MAX_FACT_ID_LENGTH = 40
_MAX_FACT_ITEMS = 8
_MAX_FACT_TEXT_LENGTH = 220
_MAX_EVENT_VALUE_LENGTH = 160


def _ensure_mapping(arguments: dict[str, object]) -> dict[str, object]:
    if not isinstance(arguments, Mapping):
        raise RuntimeError("Tool arguments must be an object.")
    return dict(arguments)


def _get_optional_text(
    arguments: Mapping[str, object],
    key: str,
    *,
    max_length: int,
) -> str | None:
    raw_value = arguments.get(key)
    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        raise RuntimeError(f"Invalid `{key}` value.")
    compact = " ".join(raw_value.split()).strip()
    if not compact:
        return None
    if len(compact) > max_length:
        raise RuntimeError(f"`{key}` exceeds the maximum supported length.")
    return compact


def _get_optional_bool(arguments: Mapping[str, object], key: str) -> bool | None:
    raw_value = arguments.get(key)
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        return raw_value
    raise RuntimeError(f"Invalid `{key}` value.")


def _get_optional_int(
    arguments: Mapping[str, object],
    key: str,
    *,
    minimum: int,
    maximum: int,
) -> int | None:
    raw_value = arguments.get(key)
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        raise RuntimeError(f"Invalid `{key}` value.")
    if isinstance(raw_value, int):
        number = raw_value
    elif isinstance(raw_value, float):
        if not raw_value.is_integer():
            raise RuntimeError(f"Invalid `{key}` value.")
        number = int(raw_value)
    elif isinstance(raw_value, str):
        try:
            number = int(raw_value)
        except ValueError as exc:
            raise RuntimeError(f"Invalid `{key}` value.") from exc
    else:
        raise RuntimeError(f"Invalid `{key}` value.")
    if number < minimum or number > maximum:
        raise RuntimeError(f"`{key}` must be between {minimum} and {maximum}.")
    return number


def _normalize_action(value: str | None) -> str:
    compact = " ".join(str(value or "").split()).strip().lower().replace("-", "_").replace(" ", "_")
    if compact not in _ALLOWED_ACTIONS:
        raise RuntimeError("Unsupported `action`.")
    return compact


def _parse_learned_facts(arguments: Mapping[str, object]) -> tuple[UserDiscoveryFact, ...]:
    raw_value = arguments.get("learned_facts")
    if raw_value is None:
        return ()
    if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes, bytearray)):
        raise RuntimeError("`learned_facts` must be a list.")
    facts: list[UserDiscoveryFact] = []
    for index, item in enumerate(raw_value[:_MAX_FACT_ITEMS]):
        if not isinstance(item, Mapping):
            raise RuntimeError(f"Invalid `learned_facts[{index}]` value.")
        storage = _get_optional_text(item, "storage", max_length=24) or "user_profile"
        normalized_storage = storage.lower().replace("-", "_").replace(" ", "_")
        if normalized_storage not in _ALLOWED_FACT_STORAGES:
            raise RuntimeError(f"Unsupported `learned_facts[{index}].storage`.")
        text = _get_optional_text(item, "text", max_length=_MAX_FACT_TEXT_LENGTH)
        if text is None:
            continue
        facts.append(UserDiscoveryFact(storage=normalized_storage, text=text))
    return tuple(facts)


def _parse_memory_routes(arguments: Mapping[str, object]) -> tuple[UserDiscoveryMemoryRoute, ...]:
    raw_value = arguments.get("memory_routes")
    if raw_value is None:
        return ()
    if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes, bytearray)):
        raise RuntimeError("`memory_routes` must be a list.")
    routes: list[UserDiscoveryMemoryRoute] = []
    for index, item in enumerate(raw_value[:_MAX_FACT_ITEMS]):
        if not isinstance(item, Mapping):
            raise RuntimeError(f"Invalid `memory_routes[{index}]` value.")
        route_kind = _get_optional_text(item, "route_kind", max_length=32)
        normalized_route_kind = (route_kind or "").lower().replace("-", "_").replace(" ", "_")
        if normalized_route_kind not in _ALLOWED_ROUTE_KINDS:
            raise RuntimeError(f"Unsupported `memory_routes[{index}].route_kind`.")
        routes.append(UserDiscoveryMemoryRoute.from_dict(item))
    return tuple(route for route in routes if not route.is_empty())


def _safe_emit(owner: Any, key: str, value: object) -> None:
    compact = " ".join(str(value).split()).strip()
    if len(compact) > _MAX_EVENT_VALUE_LENGTH:
        compact = compact[: _MAX_EVENT_VALUE_LENGTH - 3].rstrip() + "..."
    emit_best_effort(owner, f"{key}={compact or '-'}")


def _safe_record_event(owner: Any, event_name: str, message: str, **metadata: object) -> None:
    record_event_best_effort(owner, event_name, message, dict(metadata))


def handle_manage_user_discovery(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Advance Twinr's guided get-to-know-you flow and persist high-value facts."""

    arguments = _ensure_mapping(arguments)
    action = _normalize_action(_get_optional_text(arguments, "action", max_length=_MAX_ACTION_LENGTH))
    topic_id = _get_optional_text(arguments, "topic_id", max_length=_MAX_TOPIC_ID_LENGTH)
    fact_id = _get_optional_text(arguments, "fact_id", max_length=_MAX_FACT_ID_LENGTH)
    learned_facts = _parse_learned_facts(arguments)
    memory_routes = _parse_memory_routes(arguments)
    topic_complete = _get_optional_bool(arguments, "topic_complete")
    permission_granted = _get_optional_bool(arguments, "permission_granted")
    snooze_days = _get_optional_int(arguments, "snooze_days", minimum=1, maximum=14)

    if learned_facts or memory_routes or action in {"replace_fact", "delete_fact"}:
        require_sensitive_voice_confirmation(
            owner,
            arguments,
            action_label="save get-to-know-you details",
        )

    result = owner.runtime.manage_user_discovery(
        action=action,
        topic_id=topic_id,
        learned_facts=learned_facts,
        memory_routes=memory_routes,
        fact_id=fact_id,
        topic_complete=topic_complete,
        permission_granted=permission_granted,
        snooze_days=snooze_days,
    )
    payload = result.to_dict()
    _safe_emit(owner, "user_discovery_tool_call", True)
    _safe_emit(owner, "user_discovery_action", action)
    if payload.get("topic_id"):
        _safe_emit(owner, "user_discovery_topic", payload["topic_id"])
    _safe_emit(owner, "user_discovery_mode", payload.get("response_mode", "status"))
    _safe_record_event(
        owner,
        "user_discovery_updated",
        "Twinr updated the guided user-discovery flow.",
        action=action,
        topic_id=payload.get("topic_id"),
        response_mode=payload.get("response_mode"),
        facts_saved=payload.get("facts_saved", 0),
    )
    return payload


__all__ = ["handle_manage_user_discovery"]
