"""Handle self-coding feasibility and requirements-dialogue tool calls."""

from __future__ import annotations

from typing import Any

from twinr.agent.self_coding import (
    SelfCodingCapabilityRegistry,
    SelfCodingFeasibilityChecker,
    SelfCodingLearningFlow,
    SelfCodingRequirementsDialogue,
    SelfCodingStore,
    SkillSpec,
    SkillTriggerSpec,
)

_MAX_TELEMETRY_VALUE_LENGTH = 256


def handle_propose_skill_learning(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Start a self-coding feasibility check and requirements dialogue."""

    name = _require_text(arguments.get("name"), field_name="name")
    action = _require_text(arguments.get("action"), field_name="action")
    capabilities = _require_string_list(arguments.get("capabilities"), field_name="capabilities")
    if not capabilities:
        raise RuntimeError("propose_skill_learning requires at least one capability")
    request_summary = _optional_text(arguments.get("request_summary")) or action
    draft_spec = SkillSpec(
        name=name,
        action=action,
        trigger=SkillTriggerSpec(
            mode=_optional_text(arguments.get("trigger_mode")) or "push",
            conditions=tuple(_require_string_list(arguments.get("trigger_conditions"), field_name="trigger_conditions")),
        ),
        skill_id=_optional_text(arguments.get("skill_id")) or "",
        scope=_require_mapping(arguments.get("scope"), field_name="scope"),
        constraints=tuple(_require_string_list(arguments.get("constraints"), field_name="constraints")),
        capabilities=tuple(capabilities),
    )
    update = _resolve_learning_flow(owner).start_request(draft_spec, request_summary=request_summary)
    if update.session is not None:
        _safe_emit(owner, "self_coding_tool_call=true")
        _safe_emit_kv(owner, "self_coding_dialogue", update.session.session_id)
    _safe_record_event(
        owner,
        "self_coding_learning_requested",
        "Realtime tool evaluated a self-coding learning request.",
        phase=update.phase,
        outcome=update.feasibility.outcome.value,
        skill_name=name,
    )
    return _learning_update_payload(update)


def handle_answer_skill_question(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Continue an active self-coding requirements dialogue."""

    session_id = _require_text(arguments.get("session_id"), field_name="session_id")
    response: dict[str, object] = {}
    for key in (
        "use_default",
        "trigger_mode",
        "trigger_conditions",
        "scope",
        "constraints",
        "action",
        "answer_summary",
        "confirmed",
    ):
        if key in arguments:
            response[key] = arguments[key]
    update = _resolve_learning_flow(owner).answer_question(session_id, response)
    _safe_emit(owner, "self_coding_tool_call=true")
    _safe_emit_kv(owner, "self_coding_dialogue", session_id)
    _safe_record_event(
        owner,
        "self_coding_learning_answered",
        "Realtime tool advanced a self-coding requirements dialogue.",
        phase=update.phase,
        outcome=update.feasibility.outcome.value,
        session_id=session_id,
    )
    return _learning_update_payload(update)


def _resolve_learning_flow(owner: Any) -> SelfCodingLearningFlow:
    flow = getattr(owner, "_self_coding_learning_flow", None)
    if isinstance(flow, SelfCodingLearningFlow):
        return flow
    store = getattr(owner, "_self_coding_store", None)
    if not isinstance(store, SelfCodingStore):
        store = SelfCodingStore.from_config(owner.config)
    registry = getattr(owner, "_self_coding_capability_registry", None)
    if not isinstance(registry, SelfCodingCapabilityRegistry):
        registry = SelfCodingCapabilityRegistry.from_config(owner.config)
    checker = getattr(owner, "_self_coding_feasibility_checker", None)
    if not isinstance(checker, SelfCodingFeasibilityChecker):
        checker = SelfCodingFeasibilityChecker(registry)
    dialogue = getattr(owner, "_self_coding_requirements_dialogue", None)
    if not isinstance(dialogue, SelfCodingRequirementsDialogue):
        dialogue = SelfCodingRequirementsDialogue()
    flow = SelfCodingLearningFlow(store=store, checker=checker, dialogue=dialogue)
    setattr(owner, "_self_coding_learning_flow", flow)
    setattr(owner, "_self_coding_store", store)
    setattr(owner, "_self_coding_capability_registry", registry)
    setattr(owner, "_self_coding_feasibility_checker", checker)
    setattr(owner, "_self_coding_requirements_dialogue", dialogue)
    return flow


def _learning_update_payload(update) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": update.phase,
        "recommended_reply": update.recommended_reply,
        "feasibility": update.feasibility.to_payload(),
        "summary": update.feasibility.summary,
        "outcome": update.feasibility.outcome.value,
        "reasons": list(update.feasibility.reasons),
        "missing_capabilities": list(update.feasibility.missing_capabilities),
        "suggested_target": None if update.feasibility.suggested_target is None else update.feasibility.suggested_target.value,
    }
    if update.prompt:
        payload["prompt"] = update.prompt
    if update.session is not None:
        payload["session_id"] = update.session.session_id
        payload["remaining_questions"] = max(0, 3 - len(update.session.answered_question_ids))
        payload["dialogue_status"] = update.session.status.value
    if update.skill_spec is not None:
        payload["skill_spec"] = update.skill_spec.to_payload()
    return payload


def _require_text(raw_value: object, *, field_name: str) -> str:
    text = str(raw_value or "").strip()
    if not text:
        raise RuntimeError(f"{field_name} must not be empty")
    return text


def _optional_text(raw_value: object) -> str | None:
    text = str(raw_value or "").strip()
    return text or None


def _require_mapping(raw_value: object, *, field_name: str) -> dict[str, object]:
    if raw_value is None:
        return {}
    if not isinstance(raw_value, dict):
        raise RuntimeError(f"{field_name} must be a JSON object")
    return dict(raw_value)


def _require_string_list(raw_value: object, *, field_name: str) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, (str, bytes, bytearray)):
        raw_items = (raw_value,)
    elif isinstance(raw_value, (list, tuple)):
        raw_items = raw_value
    else:
        raise RuntimeError(f"{field_name} must be a list of strings")
    values: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        value = str(raw_item or "").strip()
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _safe_emit(owner: Any, event: str) -> None:
    try:
        owner.emit(event)
    except Exception:
        return


def _safe_emit_kv(owner: Any, key: str, value: object) -> None:
    safe_value = " ".join(str(value or "").split())[:_MAX_TELEMETRY_VALUE_LENGTH]
    _safe_emit(owner, f"{key}={safe_value}")


def _safe_record_event(owner: Any, event_name: str, description: str, **payload: object) -> None:
    try:
        owner._record_event(event_name, description, **payload)
    except Exception:
        return
