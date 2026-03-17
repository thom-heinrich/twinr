"""Handle self-coding feasibility, activation, and rollback tool calls."""

from __future__ import annotations

import logging
import math  # AUDIT-FIX(#5): reject non-finite numeric values inside JSON-like scope payloads.
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Callable, TypedDict, TypeVar

from twinr.agent.self_coding import (
    SelfCodingActivationService,
    SelfCodingCapabilityRegistry,
    SelfCodingCompileWorker,
    SelfCodingFeasibilityChecker,
    SelfCodingHealthService,
    SelfCodingLearningFlow,
    SelfCodingRequirementsDialogue,
    SelfCodingSkillExecutionService,
    SelfCodingStore,
    SkillSpec,
    SkillTriggerSpec,
)
from twinr.automations import AutomationStore

_LOGGER = logging.getLogger(__name__)
_MAX_TELEMETRY_VALUE_LENGTH = 256
_MAX_TEXT_LENGTH = 4096
_MAX_ID_LENGTH = 256
_MAX_LIST_ITEMS = 64
_MAX_LIST_ITEM_LENGTH = 256
_MAX_MAPPING_ENTRIES = 64
_MAX_MAPPING_KEY_LENGTH = 128
_MAX_JSON_DEPTH = 6  # AUDIT-FIX(#5): bound nested JSON-like structures to protect RPi memory and serialization budgets.
_MAX_RUNTIME_SIGNATURE_ITEMS = 128  # AUDIT-FIX(#6): fingerprint a broader config surface so cache invalidation survives in-place config updates.
_MAX_RUNTIME_SIGNATURE_REPR_LENGTH = 256
_DEFAULT_REMAINING_QUESTIONS = 3
_SELF_CODING_RUNTIME_LOCK = threading.RLock()
_T = TypeVar("_T")


class _SelfCodingRuntime(TypedDict):
    store: SelfCodingStore
    registry: SelfCodingCapabilityRegistry
    checker: SelfCodingFeasibilityChecker
    dialogue: SelfCodingRequirementsDialogue
    compile_worker: SelfCodingCompileWorker
    automation_store: AutomationStore
    activation_service: SelfCodingActivationService
    health_service: SelfCodingHealthService
    skill_execution_service: SelfCodingSkillExecutionService
    flow: SelfCodingLearningFlow


class _SelfCodingToolInputError(RuntimeError):
    """Raised when tool arguments are invalid but the process is otherwise healthy."""


def handle_propose_skill_learning(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Start a self-coding feasibility check and requirements dialogue."""

    def _operation() -> dict[str, object]:
        name = _require_text(arguments.get("name"), field_name="name", max_length=_MAX_ID_LENGTH)
        action = _require_text(arguments.get("action"), field_name="action", max_length=_MAX_TEXT_LENGTH)
        capabilities = _require_string_list(
            arguments.get("capabilities"),
            field_name="capabilities",
            max_items=_MAX_LIST_ITEMS,
            max_item_length=_MAX_LIST_ITEM_LENGTH,
        )
        if not capabilities:
            raise _SelfCodingToolInputError("propose_skill_learning requires at least one capability")
        request_summary = _optional_text(
            arguments.get("request_summary"),
            field_name="request_summary",
            max_length=_MAX_TEXT_LENGTH,
        ) or action
        draft_spec = SkillSpec(
            name=name,
            action=action,
            trigger=SkillTriggerSpec(
                mode=_optional_text(arguments.get("trigger_mode"), field_name="trigger_mode", max_length=64) or "push",
                conditions=tuple(
                    _require_string_list(
                        arguments.get("trigger_conditions"),
                        field_name="trigger_conditions",
                        max_items=_MAX_LIST_ITEMS,
                        max_item_length=_MAX_LIST_ITEM_LENGTH,
                    )
                ),
            ),
            skill_id=_optional_text(arguments.get("skill_id"), field_name="skill_id", max_length=_MAX_ID_LENGTH) or "",
            scope=_require_mapping(arguments.get("scope"), field_name="scope"),
            constraints=tuple(
                _require_string_list(
                    arguments.get("constraints"),
                    field_name="constraints",
                    max_items=_MAX_LIST_ITEMS,
                    max_item_length=_MAX_LIST_ITEM_LENGTH,
                )
            ),
            capabilities=tuple(capabilities),
        )
        update = _resolve_learning_flow(owner).start_request(draft_spec, request_summary=request_summary)
        if getattr(update, "session", None) is not None:
            _safe_emit_kv(owner, "self_coding_dialogue", getattr(update.session, "session_id", None))  # AUDIT-FIX(#4): telemetry must tolerate partial internal result objects.
        _safe_record_event(
            owner,
            "self_coding_learning_requested",
            "Realtime tool evaluated a self-coding learning request.",
            phase=_enum_value(getattr(update, "phase", None)),  # AUDIT-FIX(#4): avoid crashes when internal enums are downgraded to raw strings or missing.
            outcome=_enum_value(getattr(getattr(update, "feasibility", None), "outcome", None)),
            skill_name=name,
        )
        return _learning_update_payload(update)

    return _run_tool_call(owner, "propose_skill_learning", _operation)


def handle_answer_skill_question(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Continue an active self-coding requirements dialogue."""

    def _operation() -> dict[str, object]:
        session_id = _require_text(arguments.get("session_id"), field_name="session_id", max_length=_MAX_ID_LENGTH)
        response = _build_answer_response(arguments)
        update = _resolve_learning_flow(owner).answer_question(session_id, response)
        _safe_emit_kv(owner, "self_coding_dialogue", session_id)
        _safe_record_event(
            owner,
            "self_coding_learning_answered",
            "Realtime tool advanced a self-coding requirements dialogue.",
            phase=_enum_value(getattr(update, "phase", None)),  # AUDIT-FIX(#4): avoid crashes when internal enums are downgraded to raw strings or missing.
            outcome=_enum_value(getattr(getattr(update, "feasibility", None), "outcome", None)),
            session_id=session_id,
        )
        return _learning_update_payload(update)

    return _run_tool_call(owner, "answer_skill_question", _operation)


def handle_confirm_skill_activation(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Enable one staged learned skill version after explicit confirmation."""

    def _operation() -> dict[str, object]:
        job_id = _require_text(arguments.get("job_id"), field_name="job_id", max_length=_MAX_ID_LENGTH)
        confirmed = _coerce_bool(arguments.get("confirmed"), field_name="confirmed")  # AUDIT-FIX(#2): require an explicit confirmation bit instead of silently treating omission as a decline.
        activation = _resolve_activation_service(owner).confirm_activation(job_id=job_id, confirmed=confirmed)
        activation_skill_id = getattr(activation, "skill_id", None)  # AUDIT-FIX(#4): telemetry and audit logging must tolerate partial activation payloads.
        activation_version = getattr(activation, "version", None)
        _safe_emit_kv(owner, "self_coding_activation", activation_skill_id)
        status_value = _enum_value(getattr(activation, "status", None))
        if confirmed and status_value in {"activated", "active", "enabled"}:
            event_name = "self_coding_skill_activated"
            description = "Realtime tool enabled a staged self-coding skill version."
        elif not confirmed:
            event_name = "self_coding_skill_activation_declined"
            description = "Realtime tool declined activation of a staged self-coding skill version."
        else:
            event_name = "self_coding_skill_activation_processed"
            description = "Realtime tool processed a staged self-coding skill activation request."
        _safe_record_event(
            owner,
            event_name,
            description,
            job_id=job_id,
            confirmed=confirmed,
            skill_id=activation_skill_id,
            version=activation_version,
            status=status_value,
        )
        return _activation_payload(owner, activation)

    return _run_tool_call(owner, "confirm_skill_activation", _operation)


def handle_rollback_skill_activation(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Restore one earlier learned skill version and pause the current one."""

    def _operation() -> dict[str, object]:
        skill_id = _require_text(arguments.get("skill_id"), field_name="skill_id", max_length=_MAX_ID_LENGTH)
        target_version = _optional_positive_int(arguments.get("target_version"), field_name="target_version")
        activation = _resolve_activation_service(owner).rollback_activation(
            skill_id=skill_id,
            target_version=target_version,
        )
        activation_skill_id = getattr(activation, "skill_id", None)  # AUDIT-FIX(#4): telemetry and audit logging must tolerate partial activation payloads.
        _safe_emit_kv(owner, "self_coding_activation", activation_skill_id)
        _safe_record_event(
            owner,
            "self_coding_skill_rolled_back",
            "Realtime tool rolled a self-coding skill back to an earlier version.",
            skill_id=activation_skill_id,
            version=getattr(activation, "version", None),
        )
        return _activation_payload(owner, activation)

    return _run_tool_call(owner, "rollback_skill_activation", _operation)


def handle_pause_skill_activation(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Pause one active learned skill version."""

    def _operation() -> dict[str, object]:
        skill_id = _require_text(arguments.get("skill_id"), field_name="skill_id", max_length=_MAX_ID_LENGTH)
        version = _optional_positive_int(arguments.get("version"), field_name="version")
        if version is None:
            raise _SelfCodingToolInputError("version must be a positive integer")
        reason = _optional_text(arguments.get("reason"), field_name="reason", max_length=_MAX_LIST_ITEM_LENGTH) or "operator_pause"
        activation = _resolve_activation_service(owner).pause_activation(skill_id=skill_id, version=version, reason=reason)
        activation_skill_id = getattr(activation, "skill_id", None)  # AUDIT-FIX(#4): telemetry and audit logging must tolerate partial activation payloads.
        _safe_emit_kv(owner, "self_coding_activation", activation_skill_id)
        _safe_record_event(
            owner,
            "self_coding_skill_paused",
            "Realtime tool paused a learned self-coding skill version.",
            skill_id=activation_skill_id,
            version=getattr(activation, "version", None),
            reason=reason,
        )
        return _activation_payload(owner, activation)

    return _run_tool_call(owner, "pause_skill_activation", _operation)


def handle_reactivate_skill_activation(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Re-enable one paused learned skill version."""

    def _operation() -> dict[str, object]:
        skill_id = _require_text(arguments.get("skill_id"), field_name="skill_id", max_length=_MAX_ID_LENGTH)
        version = _optional_positive_int(arguments.get("version"), field_name="version")
        if version is None:
            raise _SelfCodingToolInputError("version must be a positive integer")
        activation = _resolve_activation_service(owner).reactivate_activation(skill_id=skill_id, version=version)
        activation_skill_id = getattr(activation, "skill_id", None)  # AUDIT-FIX(#4): telemetry and audit logging must tolerate partial activation payloads.
        _safe_emit_kv(owner, "self_coding_activation", activation_skill_id)
        _safe_record_event(
            owner,
            "self_coding_skill_reactivated",
            "Realtime tool re-enabled a paused self-coding skill version.",
            skill_id=activation_skill_id,
            version=getattr(activation, "version", None),
        )
        return _activation_payload(owner, activation)

    return _run_tool_call(owner, "reactivate_skill_activation", _operation)


def handle_run_self_coding_skill_scheduled(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Execute one hidden scheduled trigger for an active self-coding skill."""

    def _operation() -> dict[str, object]:
        skill_id = _require_text(arguments.get("skill_id"), field_name="skill_id", max_length=_MAX_ID_LENGTH)
        version = _optional_positive_int(arguments.get("version"), field_name="version")
        if version is None:
            raise _SelfCodingToolInputError("version must be a positive integer")
        trigger_id = _require_text(arguments.get("trigger_id"), field_name="trigger_id", max_length=_MAX_ID_LENGTH)
        result = _require_result_mapping(  # AUDIT-FIX(#4): normalize internal execution results before reading fields or returning the payload.
            ensure_self_coding_runtime(owner)["skill_execution_service"].execute_scheduled(
                owner,
                skill_id=skill_id,
                version=version,
                trigger_id=trigger_id,
            ),
            operation="run_self_coding_skill_scheduled",
        )
        _safe_record_event(
            owner,
            "self_coding_skill_scheduled_executed",
            "Background automation executed a scheduled self-coding skill trigger.",
            skill_id=skill_id,
            version=version,
            trigger_id=trigger_id,
            delivered=bool(result.get("delivered", False)),
        )
        return result

    return _run_tool_call(owner, "run_self_coding_skill_scheduled", _operation)


def handle_run_self_coding_skill_sensor(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Execute one hidden sensor trigger for an active self-coding skill."""

    def _operation() -> dict[str, object]:
        skill_id = _require_text(arguments.get("skill_id"), field_name="skill_id", max_length=_MAX_ID_LENGTH)
        version = _optional_positive_int(arguments.get("version"), field_name="version")
        if version is None:
            raise _SelfCodingToolInputError("version must be a positive integer")
        trigger_id = _require_text(arguments.get("trigger_id"), field_name="trigger_id", max_length=_MAX_ID_LENGTH)
        event_name = _optional_text(arguments.get("event_name"), field_name="event_name", max_length=_MAX_ID_LENGTH)
        result = _require_result_mapping(  # AUDIT-FIX(#4): normalize internal execution results before reading fields or returning the payload.
            ensure_self_coding_runtime(owner)["skill_execution_service"].execute_sensor_event(
                owner,
                skill_id=skill_id,
                version=version,
                trigger_id=trigger_id,
                event_name=event_name,
            ),
            operation="run_self_coding_skill_sensor",
        )
        _safe_record_event(
            owner,
            "self_coding_skill_sensor_executed",
            "Background automation executed a sensor-triggered self-coding skill handler.",
            skill_id=skill_id,
            version=version,
            trigger_id=trigger_id,
            sensor_event_name=event_name,
            delivered=bool(result.get("delivered", False)),
        )
        return result

    return _run_tool_call(owner, "run_self_coding_skill_sensor", _operation)


def ensure_self_coding_runtime(owner: Any) -> _SelfCodingRuntime:
    """Build and cache the focused self-coding helpers on one runtime owner."""

    with _SELF_CODING_RUNTIME_LOCK:
        config = _require_owner_config(owner)
        runtime_signature = _runtime_signature(config)
        cached_runtime_signature = getattr(owner, "_self_coding_runtime_signature", None)

        store = getattr(owner, "_self_coding_store", None)
        if runtime_signature != cached_runtime_signature or not isinstance(store, SelfCodingStore):
            store = SelfCodingStore.from_config(config)

        registry = getattr(owner, "_self_coding_capability_registry", None)
        if runtime_signature != cached_runtime_signature or not isinstance(registry, SelfCodingCapabilityRegistry):
            registry = SelfCodingCapabilityRegistry.from_config(config)

        checker = getattr(owner, "_self_coding_feasibility_checker", None)
        if not isinstance(checker, SelfCodingFeasibilityChecker) or _get_dependency(checker, "registry") is not registry:
            checker = SelfCodingFeasibilityChecker(registry)

        dialogue = getattr(owner, "_self_coding_requirements_dialogue", None)
        if not isinstance(dialogue, SelfCodingRequirementsDialogue):
            dialogue = SelfCodingRequirementsDialogue()

        compile_worker = getattr(owner, "_self_coding_compile_worker", None)
        if not isinstance(compile_worker, SelfCodingCompileWorker) or _get_dependency(compile_worker, "store") is not store:
            compile_worker = SelfCodingCompileWorker(store=store)

        automation_store = getattr(owner, "_self_coding_automation_store", None)
        if runtime_signature != cached_runtime_signature or not isinstance(automation_store, AutomationStore):
            automation_store = AutomationStore(
                config.automation_store_path,
                timezone_name=config.local_timezone_name,
                max_entries=config.automation_max_entries,
            )

        activation_service = getattr(owner, "_self_coding_activation_service", None)
        if (
            not isinstance(activation_service, SelfCodingActivationService)
            or _get_dependency(activation_service, "store") is not store
            or _get_dependency(activation_service, "automation_store") is not automation_store
        ):
            activation_service = SelfCodingActivationService(store=store, automation_store=automation_store)

        health_service = getattr(owner, "_self_coding_health_service", None)
        if (
            not isinstance(health_service, SelfCodingHealthService)
            or _get_dependency(health_service, "store") is not store
            or _get_dependency(health_service, "activation_service") is not activation_service
        ):
            health_service = SelfCodingHealthService(store=store, activation_service=activation_service)

        skill_execution_service = getattr(owner, "_self_coding_skill_execution_service", None)
        if (
            not isinstance(skill_execution_service, SelfCodingSkillExecutionService)
            or _get_dependency(skill_execution_service, "store") is not store
            or _get_dependency(skill_execution_service, "health_service") is not health_service
        ):
            skill_execution_service = SelfCodingSkillExecutionService(store=store, health_service=health_service)

        flow = getattr(owner, "_self_coding_learning_flow", None)
        if (
            not isinstance(flow, SelfCodingLearningFlow)
            or _get_dependency(flow, "store") is not store
            or _get_dependency(flow, "checker") is not checker
            or _get_dependency(flow, "dialogue") is not dialogue
            or _get_dependency(flow, "compile_worker") is not compile_worker
        ):
            flow = SelfCodingLearningFlow(store=store, checker=checker, dialogue=dialogue, compile_worker=compile_worker)

        setattr(owner, "_self_coding_runtime_signature", runtime_signature)
        setattr(owner, "_self_coding_store", store)
        setattr(owner, "_self_coding_capability_registry", registry)
        setattr(owner, "_self_coding_feasibility_checker", checker)
        setattr(owner, "_self_coding_requirements_dialogue", dialogue)
        setattr(owner, "_self_coding_compile_worker", compile_worker)
        setattr(owner, "_self_coding_automation_store", automation_store)
        setattr(owner, "_self_coding_activation_service", activation_service)
        setattr(owner, "_self_coding_health_service", health_service)
        setattr(owner, "_self_coding_skill_execution_service", skill_execution_service)
        setattr(owner, "_self_coding_learning_flow", flow)
        return {
            "store": store,
            "registry": registry,
            "checker": checker,
            "dialogue": dialogue,
            "compile_worker": compile_worker,
            "automation_store": automation_store,
            "activation_service": activation_service,
            "health_service": health_service,
            "skill_execution_service": skill_execution_service,
            "flow": flow,
        }


def _resolve_learning_flow(owner: Any) -> SelfCodingLearningFlow:
    return ensure_self_coding_runtime(owner)["flow"]


def _resolve_activation_service(owner: Any) -> SelfCodingActivationService:
    return ensure_self_coding_runtime(owner)["activation_service"]


def _learning_update_payload(update: Any) -> dict[str, object]:
    feasibility = getattr(update, "feasibility", None)
    if feasibility is None:
        raise RuntimeError("learning update is missing feasibility data")  # AUDIT-FIX(#4): fail predictably instead of crashing on malformed internal service payloads.
    suggested_target = getattr(feasibility, "suggested_target", None)
    payload: dict[str, object] = {
        "status": _enum_value(getattr(update, "phase", None)),  # AUDIT-FIX(#4): tolerate enum or string status payloads from internal services.
        "recommended_reply": _internal_text(getattr(update, "recommended_reply", None), max_length=_MAX_TEXT_LENGTH),
        "feasibility": feasibility.to_payload(),
        "summary": _internal_text(getattr(feasibility, "summary", None), max_length=_MAX_TEXT_LENGTH),
        "outcome": _enum_value(getattr(feasibility, "outcome", None)),
        "reasons": _coerce_internal_string_list(getattr(feasibility, "reasons", None)),  # AUDIT-FIX(#4): degrade safely if internal list-like fields are malformed.
        "missing_capabilities": _coerce_internal_string_list(
            getattr(feasibility, "missing_capabilities", None),
            max_item_length=_MAX_ID_LENGTH,
        ),
        "suggested_target": None if suggested_target is None else _enum_value(suggested_target),
    }
    prompt = _internal_text(getattr(update, "prompt", None), max_length=_MAX_TEXT_LENGTH)
    if prompt is not None:
        payload["prompt"] = prompt
    session = getattr(update, "session", None)
    if session is not None:
        payload["session_id"] = _internal_text(getattr(session, "session_id", None), max_length=_MAX_ID_LENGTH)
        payload["remaining_questions"] = _remaining_questions(session)
        payload["dialogue_status"] = _enum_value(getattr(session, "status", None))  # AUDIT-FIX(#4): tolerate enum or string dialogue status payloads from internal services.
    skill_spec = getattr(update, "skill_spec", None)
    if skill_spec is not None:
        payload["skill_spec"] = skill_spec.to_payload()
    compile_job = getattr(update, "compile_job", None)
    if compile_job is not None:
        payload["compile_job_id"] = _internal_text(getattr(compile_job, "job_id", None), max_length=_MAX_ID_LENGTH)
        payload["compile_job_status"] = _enum_value(getattr(compile_job, "status", None))  # AUDIT-FIX(#4): tolerate enum or string compile status payloads from internal services.
    return payload


def _activation_payload(owner: Any, activation: Any) -> dict[str, object]:
    automation_store = getattr(owner, "_self_coding_automation_store", None)
    raw_metadata = getattr(activation, "metadata", None)
    metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    automation_ids = _coerce_internal_string_list(  # AUDIT-FIX(#3): internal activation metadata must degrade safely instead of being validated like external caller input.
        metadata.get("automation_ids"),
        max_item_length=_MAX_ID_LENGTH,
    )
    if not automation_ids:
        single_automation_id = _internal_text(metadata.get("automation_id"), max_length=_MAX_ID_LENGTH)
        if single_automation_id:
            automation_ids = [single_automation_id]
    automation_enabled = False
    if isinstance(automation_store, AutomationStore) and automation_ids:
        enabled_values: list[bool] = []
        for automation_id in automation_ids:
            try:
                entry = automation_store.get(automation_id)
            except Exception as exc:
                _safe_record_event(
                    owner,
                    "self_coding_automation_lookup_failed",
                    "Realtime tool could not read the automation state for a self-coding activation.",
                    skill_id=getattr(activation, "skill_id", None),
                    automation_id=automation_id,
                    error_type=type(exc).__name__,
                    error_message=_truncate_text(str(exc), _MAX_TELEMETRY_VALUE_LENGTH),
                )
                entry = None
            enabled_values.append(bool(getattr(entry, "enabled", False)) if entry is not None else False)
        automation_enabled = bool(enabled_values) and all(enabled_values)
    return {
        "status": _enum_value(getattr(activation, "status", None)),  # AUDIT-FIX(#4): tolerate enum or string activation status payloads from internal services.
        "skill_id": _internal_text(getattr(activation, "skill_id", None), max_length=_MAX_ID_LENGTH),
        "skill_name": _internal_text(getattr(activation, "skill_name", None), max_length=_MAX_TEXT_LENGTH),
        "version": getattr(activation, "version", None),
        "job_id": _internal_text(getattr(activation, "job_id", None), max_length=_MAX_ID_LENGTH),
        "artifact_id": _internal_text(getattr(activation, "artifact_id", None), max_length=_MAX_ID_LENGTH),
        "automation_id": automation_ids[0] if automation_ids else None,
        "automation_ids": automation_ids,
        "automation_enabled": automation_enabled,
        "activated_at": _format_datetime(owner, "activated_at", getattr(activation, "activated_at", None)),
        "feedback_due_at": _format_datetime(owner, "feedback_due_at", getattr(activation, "feedback_due_at", None)),
    }


def _require_text(raw_value: object, *, field_name: str, max_length: int = _MAX_TEXT_LENGTH) -> str:
    text = _coerce_text(raw_value, field_name=field_name, max_length=max_length)
    if text is None:
        raise _SelfCodingToolInputError(f"{field_name} must not be empty")
    return text


def _optional_text(
    raw_value: object,
    *,
    field_name: str,
    max_length: int = _MAX_TEXT_LENGTH,
) -> str | None:
    return _coerce_text(raw_value, field_name=field_name, max_length=max_length)


def _require_mapping(raw_value: object, *, field_name: str) -> dict[str, object]:
    if raw_value is None:
        return {}
    if not isinstance(raw_value, Mapping):
        raise _SelfCodingToolInputError(f"{field_name} must be a JSON object")
    return _sanitize_mapping(raw_value, field_name=field_name, depth=0)  # AUDIT-FIX(#5): recursively sanitize nested values instead of accepting arbitrary objects verbatim.


def _require_string_list(
    raw_value: object,
    *,
    field_name: str,
    max_items: int = _MAX_LIST_ITEMS,
    max_item_length: int = _MAX_LIST_ITEM_LENGTH,
) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, (str, bytes, bytearray)):
        raw_items = (raw_value,)
    elif isinstance(raw_value, (list, tuple)):
        raw_items = raw_value
    else:
        raise _SelfCodingToolInputError(f"{field_name} must be a list of strings")
    if len(raw_items) > max_items:
        raise _SelfCodingToolInputError(f"{field_name} must contain at most {max_items} items")
    values: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        value = _coerce_text(raw_item, field_name=field_name, max_length=max_item_length)
        if value is None:
            continue
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _build_answer_response(arguments: Mapping[str, object]) -> dict[str, object]:
    response: dict[str, object] = {}
    if "use_default" in arguments:
        response["use_default"] = _coerce_bool(arguments.get("use_default"), field_name="use_default")
    if "trigger_mode" in arguments:
        response["trigger_mode"] = _optional_text(arguments.get("trigger_mode"), field_name="trigger_mode", max_length=64)
    if "trigger_conditions" in arguments:
        response["trigger_conditions"] = _require_string_list(
            arguments.get("trigger_conditions"),
            field_name="trigger_conditions",
            max_items=_MAX_LIST_ITEMS,
            max_item_length=_MAX_LIST_ITEM_LENGTH,
        )
    if "scope" in arguments:
        response["scope"] = _require_mapping(arguments.get("scope"), field_name="scope")
    if "constraints" in arguments:
        response["constraints"] = _require_string_list(
            arguments.get("constraints"),
            field_name="constraints",
            max_items=_MAX_LIST_ITEMS,
            max_item_length=_MAX_LIST_ITEM_LENGTH,
        )
    if "action" in arguments:
        response["action"] = _optional_text(arguments.get("action"), field_name="action", max_length=_MAX_TEXT_LENGTH)
    if "answer_summary" in arguments:
        response["answer_summary"] = _optional_text(
            arguments.get("answer_summary"),
            field_name="answer_summary",
            max_length=_MAX_TEXT_LENGTH,
        )
    if "confirmed" in arguments:
        response["confirmed"] = _coerce_bool(arguments.get("confirmed"), field_name="confirmed")
    if not response or all(value is None for value in response.values()):
        raise _SelfCodingToolInputError("answer_skill_question requires at least one answer field")  # AUDIT-FIX(#1): reject empty dialogue answers before they silently advance or invalidate the requirements flow.
    return response


def _run_tool_call(owner: Any, operation: str, callback: Callable[[], _T]) -> _T:
    with _SELF_CODING_RUNTIME_LOCK:
        _safe_emit(owner, "self_coding_tool_call=true")
        try:
            return callback()
        except _SelfCodingToolInputError as exc:
            _safe_record_event(
                owner,
                "self_coding_tool_input_rejected",
                "Realtime tool rejected invalid self-coding input.",
                operation=operation,
                error_message=_truncate_text(str(exc), _MAX_TELEMETRY_VALUE_LENGTH),
            )
            raise RuntimeError(str(exc)) from None
        except Exception as exc:
            _safe_record_event(
                owner,
                "self_coding_tool_failed",
                "Realtime tool failed while handling a self-coding request.",
                operation=operation,
                error_type=type(exc).__name__,
                error_message=_truncate_text(str(exc), _MAX_TELEMETRY_VALUE_LENGTH),
            )
            raise RuntimeError("The requested self-coding action could not be completed right now.") from exc


def _coerce_text(
    raw_value: object,
    *,
    field_name: str,
    max_length: int,
) -> str | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        text = raw_value.strip()
    elif isinstance(raw_value, (bytes, bytearray)):
        try:
            text = bytes(raw_value).decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise _SelfCodingToolInputError(f"{field_name} must be valid UTF-8 text") from exc
    else:
        raise _SelfCodingToolInputError(f"{field_name} must be a string")
    if not text:
        return None
    if len(text) > max_length:
        raise _SelfCodingToolInputError(f"{field_name} must be at most {max_length} characters")
    return text


def _optional_positive_int(raw_value: object, *, field_name: str) -> int | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        raise _SelfCodingToolInputError(f"{field_name} must be a positive integer")
    if isinstance(raw_value, int):
        value = raw_value
    else:
        text = _coerce_text(raw_value, field_name=field_name, max_length=32)
        if text is None or not text.isdigit():
            raise _SelfCodingToolInputError(f"{field_name} must be a positive integer")
        value = int(text)
    if value <= 0:
        raise _SelfCodingToolInputError(f"{field_name} must be a positive integer")
    return value


def _coerce_bool(raw_value: object, *, field_name: str, default: bool | None = None) -> bool:
    if raw_value is None:
        if default is None:
            raise _SelfCodingToolInputError(f"{field_name} must be a boolean")
        return default
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, int):
        if raw_value in (0, 1):
            return bool(raw_value)
        raise _SelfCodingToolInputError(f"{field_name} must be a boolean")
    text = _coerce_text(raw_value, field_name=field_name, max_length=16)
    if text is None:
        if default is None:
            raise _SelfCodingToolInputError(f"{field_name} must be a boolean")
        return default
    normalized = text.lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise _SelfCodingToolInputError(f"{field_name} must be a boolean")


def _remaining_questions(session: Any) -> int:
    answered_question_ids = getattr(session, "answered_question_ids", ()) or ()
    if isinstance(answered_question_ids, (list, tuple, set, frozenset)):
        answered = len(answered_question_ids)
    else:
        answered = 0  # AUDIT-FIX(#4): tolerate malformed internal session state instead of failing while computing telemetry/payload fields.
    for attribute_name in ("total_questions", "required_questions", "question_count"):
        raw_total = getattr(session, attribute_name, None)
        if isinstance(raw_total, int) and raw_total >= 0:
            return max(0, raw_total - answered)
    questions = getattr(session, "questions", None)
    if isinstance(questions, (list, tuple)):
        return max(0, len(questions) - answered)
    return max(0, _DEFAULT_REMAINING_QUESTIONS - answered)


def _require_owner_config(owner: Any) -> Any:
    config = getattr(owner, "config", None)
    if config is None:
        raise RuntimeError("owner config is not available")
    return config


def _runtime_signature(config: Any) -> tuple[object, ...]:  # AUDIT-FIX(#6): broaden runtime cache invalidation beyond a tiny fixed config subset.
    raw_state = getattr(config, "__dict__", None)
    if isinstance(raw_state, Mapping):
        items = tuple(
            (key, raw_state[key])
            for key in sorted(raw_state)
            if not str(key).startswith("_")
        )[:_MAX_RUNTIME_SIGNATURE_ITEMS]
        return (type(config), tuple((str(key), _freeze_signature_value(value, depth=0)) for key, value in items))  # AUDIT-FIX(#6): fingerprint public config values so in-place updates rebuild runtime helpers.
    return (
        type(config),
        getattr(config, "automation_store_path", None),
        getattr(config, "local_timezone_name", None),
        getattr(config, "automation_max_entries", None),
    )


def _freeze_signature_value(raw_value: object, *, depth: int) -> object:  # AUDIT-FIX(#6): create a bounded, hashable fingerprint for public config values.
    if depth >= _MAX_JSON_DEPTH:
        return "<max-depth>"  # AUDIT-FIX(#6): cap recursive config fingerprinting to predictable cost on RPi.
    if raw_value is None or isinstance(raw_value, (bool, int, str)):
        return raw_value
    if isinstance(raw_value, float):
        return raw_value if math.isfinite(raw_value) else "<non-finite-float>"
    if isinstance(raw_value, (bytes, bytearray)):
        return _internal_text(raw_value, max_length=_MAX_RUNTIME_SIGNATURE_REPR_LENGTH)
    if isinstance(raw_value, Mapping):
        items = []
        for index, (key, value) in enumerate(sorted(raw_value.items(), key=lambda item: str(item[0]))):
            if index >= _MAX_MAPPING_ENTRIES:
                break
            items.append((str(key), _freeze_signature_value(value, depth=depth + 1)))
        return tuple(items)
    if isinstance(raw_value, (list, tuple, set, frozenset)):
        values: list[object] = []
        for index, value in enumerate(raw_value):
            if index >= _MAX_LIST_ITEMS:
                break
            values.append(_freeze_signature_value(value, depth=depth + 1))
        return tuple(values)
    return _safe_repr(raw_value, max_length=_MAX_RUNTIME_SIGNATURE_REPR_LENGTH)  # AUDIT-FIX(#6): fallback to a bounded representation for exotic config values without crashing fingerprint generation.


def _get_dependency(instance: object, attribute_name: str) -> object:
    try:
        return getattr(instance, attribute_name)
    except Exception:
        return object()


def _enum_value(raw_value: object) -> str:
    value = getattr(raw_value, "value", raw_value)
    return _safe_stringify(value, max_length=_MAX_TEXT_LENGTH).strip()


def _internal_text(raw_value: object, *, max_length: int) -> str | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        text = raw_value.strip()
    elif isinstance(raw_value, (bytes, bytearray)):
        try:
            text = bytes(raw_value).decode("utf-8").strip()
        except UnicodeDecodeError:
            return None
    else:
        return None
    if not text:
        return None
    return text[:max_length]


def _coerce_internal_string_list(  # AUDIT-FIX(#3): parse internal metadata leniently so bad store payloads do not surface as caller input errors.
    raw_value: object,
    *,
    max_items: int = _MAX_LIST_ITEMS,
    max_item_length: int = _MAX_LIST_ITEM_LENGTH,
) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, (str, bytes, bytearray)):
        raw_items: tuple[object, ...] = (raw_value,)
    elif isinstance(raw_value, (list, tuple, set, frozenset)):
        raw_items = tuple(raw_value)
    else:
        return []
    values: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        if len(values) >= max_items:
            break
        value = _internal_text(raw_item, max_length=max_item_length)
        if value is None or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _sanitize_mapping(raw_value: Mapping[object, object], *, field_name: str, depth: int) -> dict[str, object]:  # AUDIT-FIX(#5): recursively normalize and bound nested mapping payloads.
    if depth >= _MAX_JSON_DEPTH:
        raise _SelfCodingToolInputError(f"{field_name} must not exceed {_MAX_JSON_DEPTH} levels")
    if len(raw_value) > _MAX_MAPPING_ENTRIES:
        raise _SelfCodingToolInputError(f"{field_name} must contain at most {_MAX_MAPPING_ENTRIES} entries")
    normalized: dict[str, object] = {}
    for raw_key, value in raw_value.items():
        key = _require_text(raw_key, field_name=f"{field_name} key", max_length=_MAX_MAPPING_KEY_LENGTH)
        if key in normalized:
            raise _SelfCodingToolInputError(f"{field_name} contains duplicate keys after normalization")  # AUDIT-FIX(#5): reject silent overwrites caused by whitespace-normalized duplicate keys.
        normalized[key] = _sanitize_json_value(value, field_name=f"{field_name}.{key}", depth=depth + 1)
    return normalized


def _sanitize_json_value(raw_value: object, *, field_name: str, depth: int) -> object:  # AUDIT-FIX(#5): enforce JSON-compatible nested values before they reach persistent state.
    if depth > _MAX_JSON_DEPTH:
        raise _SelfCodingToolInputError(f"{field_name} must not exceed {_MAX_JSON_DEPTH} levels")
    if raw_value is None or isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, float):
        if not math.isfinite(raw_value):
            raise _SelfCodingToolInputError(f"{field_name} must be a finite number")  # AUDIT-FIX(#5): block NaN/inf values that break JSON serialization and downstream stores.
        return raw_value
    if isinstance(raw_value, str):
        if len(raw_value) > _MAX_TEXT_LENGTH:
            raise _SelfCodingToolInputError(f"{field_name} must be at most {_MAX_TEXT_LENGTH} characters")
        return raw_value
    if isinstance(raw_value, (bytes, bytearray)):
        try:
            text = bytes(raw_value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _SelfCodingToolInputError(f"{field_name} must be valid UTF-8 text") from exc
        if len(text) > _MAX_TEXT_LENGTH:
            raise _SelfCodingToolInputError(f"{field_name} must be at most {_MAX_TEXT_LENGTH} characters")
        return text
    if isinstance(raw_value, Mapping):
        return _sanitize_mapping(raw_value, field_name=field_name, depth=depth)
    if isinstance(raw_value, (list, tuple)):
        if len(raw_value) > _MAX_LIST_ITEMS:
            raise _SelfCodingToolInputError(f"{field_name} must contain at most {_MAX_LIST_ITEMS} items")
        return [_sanitize_json_value(value, field_name=f"{field_name}[{index}]", depth=depth + 1) for index, value in enumerate(raw_value)]
    raise _SelfCodingToolInputError(f"{field_name} must contain only JSON-compatible values")


def _require_result_mapping(raw_value: object, *, operation: str) -> dict[str, object]:  # AUDIT-FIX(#4): reject malformed internal execution payloads with a stable error boundary.
    if not isinstance(raw_value, Mapping):
        raise RuntimeError(f"{operation} returned an invalid result payload")
    return dict(raw_value)


def _format_datetime(owner: Any, field_name: str, raw_value: object) -> str | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, datetime):
        _safe_record_event(
            owner,
            "self_coding_invalid_datetime",
            "Realtime tool received a non-datetime activation timestamp.",
            field_name=field_name,
            value_type=type(raw_value).__name__,
        )
        return None
    if raw_value.tzinfo is None or raw_value.utcoffset() is None:
        _safe_record_event(
            owner,
            "self_coding_invalid_datetime",
            "Realtime tool received a naive activation timestamp.",
            field_name=field_name,
        )
        return None
    return raw_value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _truncate_text(text: str, max_length: int) -> str:
    normalized = " ".join(text.split())
    return normalized[:max_length]


def _safe_repr(raw_value: object, *, max_length: int) -> str:  # AUDIT-FIX(#6): fingerprint exotic config values without letting repr() failures bubble.
    try:
        text = repr(raw_value)
    except Exception:
        text = f"<unrepresentable:{type(raw_value).__name__}>"
    return _truncate_text(text, max_length)


def _safe_stringify(raw_value: object, *, max_length: int) -> str:  # AUDIT-FIX(#7): protect telemetry from broken __str__ implementations and falsey-value loss.
    if raw_value is None:
        return ""
    if isinstance(raw_value, str):
        text = raw_value
    elif isinstance(raw_value, (bytes, bytearray)):
        try:
            text = bytes(raw_value).decode("utf-8")
        except UnicodeDecodeError:
            return ""
    else:
        try:
            text = str(raw_value)
        except Exception:
            text = f"<unprintable:{type(raw_value).__name__}>"  # AUDIT-FIX(#7): keep telemetry best-effort even when custom __str__ implementations explode.
    return _truncate_text(text, max_length)


def _safe_emit(owner: Any, event: str) -> None:
    try:
        owner.emit(event)
    except Exception as exc:
        _LOGGER.debug("self-coding telemetry emit failed: %s", type(exc).__name__)


def _safe_emit_kv(owner: Any, key: str, value: object) -> None:
    safe_value = _safe_stringify(value, max_length=_MAX_TELEMETRY_VALUE_LENGTH)  # AUDIT-FIX(#7): preserve falsey values like 0/False and prevent telemetry stringification from becoming fatal.
    _safe_emit(owner, f"{key}={safe_value}")


def _safe_record_event(owner: Any, event_name: str, description: str, **payload: object) -> None:
    try:
        owner._record_event(event_name, description, **payload)
    except Exception as exc:
        _LOGGER.debug("self-coding event recording failed: %s", type(exc).__name__)