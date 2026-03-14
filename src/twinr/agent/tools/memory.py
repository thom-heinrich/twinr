from __future__ import annotations

from typing import Any

from twinr.agent.tools.support import require_sensitive_voice_confirmation


def handle_remember_memory(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    require_sensitive_voice_confirmation(owner, arguments, action_label="save durable memory")
    kind = str(arguments.get("kind", "")).strip() or "memory"
    summary = str(arguments.get("summary", "")).strip()
    details = str(arguments.get("details", "")).strip()
    if not summary:
        raise RuntimeError("remember_memory requires `summary`")

    entry = owner.runtime.store_durable_memory(
        kind=kind,
        summary=summary,
        details=details or None,
    )
    owner.runtime.remember_note(
        kind="fact",
        content=f"Saved memory: {entry.summary}",
        source="remember_memory",
        metadata={"memory_kind": entry.kind, "memory_id": entry.entry_id},
    )
    owner.emit("memory_tool_call=true")
    owner.emit(f"memory_saved={entry.summary}")
    owner._record_event(
        "memory_saved",
        "Important user-requested memory was stored in MEMORY.md.",
        kind=entry.kind,
        summary=entry.summary,
    )
    return {
        "status": "saved",
        "kind": entry.kind,
        "summary": entry.summary,
        "memory_id": entry.entry_id,
    }


def handle_remember_contact(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    require_sensitive_voice_confirmation(owner, arguments, action_label="save a contact")
    given_name = str(arguments.get("given_name", "")).strip()
    if not given_name:
        raise RuntimeError("remember_contact requires `given_name`")
    result = owner.runtime.remember_contact(
        given_name=given_name,
        family_name=str(arguments.get("family_name", "")).strip() or None,
        phone=str(arguments.get("phone", "")).strip() or None,
        email=str(arguments.get("email", "")).strip() or None,
        role=str(arguments.get("role", "")).strip() or None,
        relation=str(arguments.get("relation", "")).strip() or None,
        notes=str(arguments.get("notes", "")).strip() or None,
        source="remember_contact",
    )
    owner.emit("graph_contact_tool_call=true")
    if result.status == "needs_clarification":
        owner.emit("graph_contact_clarification=true")
        return {
            "status": "needs_clarification",
            "question": result.question,
            "options": [
                {"label": option.label, "role": option.role, "phones": list(option.phones), "emails": list(option.emails)}
                for option in result.options
            ],
        }
    owner.emit(f"graph_contact_saved={result.label}")
    owner._record_event(
        "graph_contact_saved",
        "Structured contact memory was stored.",
        label=result.label,
        status=result.status,
    )
    return {
        "status": result.status,
        "label": result.label,
        "node_id": result.node_id,
    }


def handle_lookup_contact(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    name = str(arguments.get("name", "")).strip()
    if not name:
        raise RuntimeError("lookup_contact requires `name`")
    result = owner.runtime.lookup_contact(
        name=name,
        family_name=str(arguments.get("family_name", "")).strip() or None,
        role=str(arguments.get("role", "")).strip() or None,
    )
    owner.emit("graph_contact_lookup=true")
    if result.status == "not_found":
        return {"status": "not_found", "name": name}
    if result.status == "needs_clarification":
        owner.emit("graph_contact_clarification=true")
        return {
            "status": "needs_clarification",
            "question": result.question,
            "options": [
                {"label": option.label, "role": option.role, "phones": list(option.phones), "emails": list(option.emails)}
                for option in result.options
            ],
        }
    assert result.match is not None
    return {
        "status": "found",
        "label": result.match.label,
        "role": result.match.role,
        "phones": list(result.match.phones),
        "emails": list(result.match.emails),
    }


def handle_remember_preference(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    require_sensitive_voice_confirmation(owner, arguments, action_label="save a preference")
    category = str(arguments.get("category", "")).strip()
    value = str(arguments.get("value", "")).strip()
    if not category or not value:
        raise RuntimeError("remember_preference requires `category` and `value`")
    sentiment = str(arguments.get("sentiment", "")).strip() or "prefer"
    if sentiment == "usually_buy_at":
        sentiment = "usually_buy_at"
    result = owner.runtime.remember_preference(
        category=category,
        value=value,
        for_product=str(arguments.get("for_product", "")).strip() or None,
        sentiment=sentiment,
        details=str(arguments.get("details", "")).strip() or None,
        source="remember_preference",
    )
    owner.emit("graph_preference_tool_call=true")
    owner.emit(f"graph_preference_saved={result.label}")
    owner._record_event(
        "graph_preference_saved",
        "Structured preference memory was stored.",
        label=result.label,
        edge_type=result.edge_type,
    )
    return {
        "status": result.status,
        "label": result.label,
        "node_id": result.node_id,
        "edge_type": result.edge_type,
    }


def handle_remember_plan(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    require_sensitive_voice_confirmation(owner, arguments, action_label="save a future plan")
    summary = str(arguments.get("summary", "")).strip()
    if not summary:
        raise RuntimeError("remember_plan requires `summary`")
    result = owner.runtime.remember_plan(
        summary=summary,
        when_text=str(arguments.get("when", "")).strip() or None,
        details=str(arguments.get("details", "")).strip() or None,
        source="remember_plan",
    )
    owner.emit("graph_plan_tool_call=true")
    owner.emit(f"graph_plan_saved={result.label}")
    owner._record_event(
        "graph_plan_saved",
        "Structured plan memory was stored.",
        label=result.label,
        edge_type=result.edge_type,
    )
    return {
        "status": result.status,
        "label": result.label,
        "node_id": result.node_id,
        "edge_type": result.edge_type,
    }


def handle_update_user_profile(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    require_sensitive_voice_confirmation(owner, arguments, action_label="change the saved user profile")
    category = str(arguments.get("category", "")).strip()
    instruction = str(arguments.get("instruction", "")).strip()
    if not category or not instruction:
        raise RuntimeError("update_user_profile requires `category` and `instruction`")

    entry = owner.runtime.update_user_profile_context(
        category=category,
        instruction=instruction,
    )
    owner.runtime.remember_note(
        kind="preference",
        content=f"User profile update ({entry.key}): {entry.instruction}",
        source="update_user_profile",
        metadata={"category": entry.key},
    )
    owner.emit("user_profile_tool_call=true")
    owner.emit(f"user_profile_update={entry.key}")
    owner._record_event(
        "user_profile_updated",
        "Stable user profile context was updated from an explicit user request.",
        category=entry.key,
    )
    return {
        "status": "updated",
        "category": entry.key,
        "instruction": entry.instruction,
    }


def handle_update_personality(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    require_sensitive_voice_confirmation(owner, arguments, action_label="change Twinr's future behavior")
    category = str(arguments.get("category", "")).strip()
    instruction = str(arguments.get("instruction", "")).strip()
    if not category or not instruction:
        raise RuntimeError("update_personality requires `category` and `instruction`")

    entry = owner.runtime.update_personality_context(
        category=category,
        instruction=instruction,
    )
    owner.runtime.remember_note(
        kind="preference",
        content=f"Behavior update ({entry.key}): {entry.instruction}",
        source="update_personality",
        metadata={"category": entry.key},
    )
    owner.emit("personality_tool_call=true")
    owner.emit(f"personality_update={entry.key}")
    owner._record_event(
        "personality_updated",
        "Twinr personality context was updated from an explicit user request.",
        category=entry.key,
    )
    return {
        "status": "updated",
        "category": entry.key,
        "instruction": entry.instruction,
    }
