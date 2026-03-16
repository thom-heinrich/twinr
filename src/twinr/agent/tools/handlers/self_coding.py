"""Handle self-coding feasibility, activation, and rollback tool calls."""

from __future__ import annotations

import logging  # AUDIT-FIX(#10): add lightweight diagnostics for best-effort telemetry failures.
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Callable, TypedDict, TypeVar

from twinr.agent.self_coding import (
    SelfCodingActivationService,
    SelfCodingCapabilityRegistry,
    SelfCodingCompileWorker,
    SelfCodingFeasibilityChecker,
    SelfCodingLearningFlow,
    SelfCodingRequirementsDialogue,
    SelfCodingStore,
    SkillSpec,
    SkillTriggerSpec,
)
from twinr.automations import AutomationStore

_LOGGER = logging.getLogger(__name__)
_MAX_TELEMETRY_VALUE_LENGTH = 256
_MAX_TEXT_LENGTH = 4096  # AUDIT-FIX(#6): bound free-form text to protect RPi memory/log budgets.
_MAX_ID_LENGTH = 256
_MAX_LIST_ITEMS = 64
_MAX_LIST_ITEM_LENGTH = 256
_MAX_MAPPING_ENTRIES = 64
_MAX_MAPPING_KEY_LENGTH = 128
_DEFAULT_REMAINING_QUESTIONS = 3
_SELF_CODING_RUNTIME_LOCK = threading.RLock()  # AUDIT-FIX(#2): serialize runtime/tool mutations around shared file-backed state.
_T = TypeVar("_T")


class _SelfCodingRuntime(TypedDict):  # AUDIT-FIX(#4): typed runtime bundle removes ignore-based casts around guarded runtime construction.
    store: SelfCodingStore
    registry: SelfCodingCapabilityRegistry
    checker: SelfCodingFeasibilityChecker
    dialogue: SelfCodingRequirementsDialogue
    compile_worker: SelfCodingCompileWorker
    automation_store: AutomationStore
    activation_service: SelfCodingActivationService
    flow: SelfCodingLearningFlow


class _SelfCodingToolInputError(RuntimeError):  # AUDIT-FIX(#4): distinguish caller input errors from unexpected subsystem failures.
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
        if update.session is not None:
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

    return _run_tool_call(owner, "propose_skill_learning", _operation)  # AUDIT-FIX(#4): guard service/config failures with a stable error boundary.


def handle_answer_skill_question(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Continue an active self-coding requirements dialogue."""

    def _operation() -> dict[str, object]:
        session_id = _require_text(arguments.get("session_id"), field_name="session_id", max_length=_MAX_ID_LENGTH)
        response = _build_answer_response(arguments)  # AUDIT-FIX(#6): validate dialogue answers before passing them downstream.
        update = _resolve_learning_flow(owner).answer_question(session_id, response)
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

    return _run_tool_call(owner, "answer_skill_question", _operation)  # AUDIT-FIX(#4): prevent raw service exceptions from escaping.


def handle_confirm_skill_activation(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Enable one staged learned skill version after explicit confirmation."""

    def _operation() -> dict[str, object]:
        job_id = _require_text(arguments.get("job_id"), field_name="job_id", max_length=_MAX_ID_LENGTH)
        confirmed = _coerce_bool(arguments.get("confirmed"), field_name="confirmed", default=False)  # AUDIT-FIX(#1): strict confirmation parsing blocks accidental activation on "false"/"0"/"no".
        activation = _resolve_activation_service(owner).confirm_activation(job_id=job_id, confirmed=confirmed)
        _safe_emit_kv(owner, "self_coding_activation", activation.skill_id)
        status_value = _enum_value(getattr(activation, "status", None))
        if confirmed and status_value in {"activated", "active", "enabled"}:
            event_name = "self_coding_skill_activated"
            description = "Realtime tool enabled a staged self-coding skill version."
        elif not confirmed:
            event_name = "self_coding_skill_activation_declined"  # AUDIT-FIX(#8): keep audit trail truthful when confirmation is withheld.
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
            skill_id=activation.skill_id,
            version=activation.version,
            status=status_value,
        )
        return _activation_payload(owner, activation)

    return _run_tool_call(owner, "confirm_skill_activation", _operation)  # AUDIT-FIX(#4): stabilize activation failures for callers.


def handle_rollback_skill_activation(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Restore one earlier learned skill version and pause the current one."""

    def _operation() -> dict[str, object]:
        skill_id = _require_text(arguments.get("skill_id"), field_name="skill_id", max_length=_MAX_ID_LENGTH)
        target_version = _optional_positive_int(arguments.get("target_version"), field_name="target_version")  # AUDIT-FIX(#5): reject bools, garbage strings, and non-positive versions.
        activation = _resolve_activation_service(owner).rollback_activation(
            skill_id=skill_id,
            target_version=target_version,
        )
        _safe_emit_kv(owner, "self_coding_activation", activation.skill_id)
        _safe_record_event(
            owner,
            "self_coding_skill_rolled_back",
            "Realtime tool rolled a self-coding skill back to an earlier version.",
            skill_id=activation.skill_id,
            version=activation.version,
        )
        return _activation_payload(owner, activation)

    return _run_tool_call(owner, "rollback_skill_activation", _operation)  # AUDIT-FIX(#4): stabilize rollback failures for callers.


def ensure_self_coding_runtime(owner: Any) -> _SelfCodingRuntime:
    """Build and cache the focused self-coding helpers on one runtime owner."""

    with _SELF_CODING_RUNTIME_LOCK:  # AUDIT-FIX(#2): owner-local caches are mutated under one lock to avoid cross-wired helpers.
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
            checker = SelfCodingFeasibilityChecker(registry)  # AUDIT-FIX(#3): rebuild dependency-bound helpers when their upstream object changes.

        dialogue = getattr(owner, "_self_coding_requirements_dialogue", None)
        if not isinstance(dialogue, SelfCodingRequirementsDialogue):
            dialogue = SelfCodingRequirementsDialogue()

        compile_worker = getattr(owner, "_self_coding_compile_worker", None)
        if not isinstance(compile_worker, SelfCodingCompileWorker) or _get_dependency(compile_worker, "store") is not store:
            compile_worker = SelfCodingCompileWorker(store=store)  # AUDIT-FIX(#3): keep compile worker pinned to the active store.

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
            activation_service = SelfCodingActivationService(store=store, automation_store=automation_store)  # AUDIT-FIX(#3): rebuild activation service when either backing store changes.

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
        setattr(owner, "_self_coding_learning_flow", flow)
        return {
            "store": store,
            "registry": registry,
            "checker": checker,
            "dialogue": dialogue,
            "compile_worker": compile_worker,
            "automation_store": automation_store,
            "activation_service": activation_service,
            "flow": flow,
        }


def _resolve_learning_flow(owner: Any) -> SelfCodingLearningFlow:
    return ensure_self_coding_runtime(owner)["flow"]


def _resolve_activation_service(owner: Any) -> SelfCodingActivationService:
    return ensure_self_coding_runtime(owner)["activation_service"]


def _learning_update_payload(update: Any) -> dict[str, object]:
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
        payload["remaining_questions"] = _remaining_questions(update.session)  # AUDIT-FIX(#9): derive remaining questions from session metadata when available.
        payload["dialogue_status"] = update.session.status.value
    if update.skill_spec is not None:
        payload["skill_spec"] = update.skill_spec.to_payload()
    if update.compile_job is not None:
        payload["compile_job_id"] = update.compile_job.job_id
        payload["compile_job_status"] = update.compile_job.status.value
    return payload


def _activation_payload(owner: Any, activation: Any) -> dict[str, object]:
    automation_store = getattr(owner, "_self_coding_automation_store", None)
    raw_metadata = getattr(activation, "metadata", None)
    metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}  # AUDIT-FIX(#7): tolerate malformed metadata payloads instead of crashing.
    automation_id = _internal_text(metadata.get("automation_id"), max_length=_MAX_ID_LENGTH) or ""
    automation_enabled = False
    if isinstance(automation_store, AutomationStore) and automation_id:
        try:
            entry = automation_store.get(automation_id)
        except Exception as exc:  # AUDIT-FIX(#7): avoid failing the whole activation payload on ancillary automation lookup errors.
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
        automation_enabled = bool(getattr(entry, "enabled", False)) if entry is not None else False
    return {
        "status": activation.status.value,
        "skill_id": activation.skill_id,
        "skill_name": activation.skill_name,
        "version": activation.version,
        "job_id": activation.job_id,
        "artifact_id": activation.artifact_id,
        "automation_id": automation_id or None,
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
    mapping = dict(raw_value)
    if len(mapping) > _MAX_MAPPING_ENTRIES:
        raise _SelfCodingToolInputError(f"{field_name} must contain at most {_MAX_MAPPING_ENTRIES} entries")
    normalized: dict[str, object] = {}
    for raw_key, value in mapping.items():
        key = _require_text(raw_key, field_name=f"{field_name} key", max_length=_MAX_MAPPING_KEY_LENGTH)  # AUDIT-FIX(#6): reject non-string keys instead of silently stringifying arbitrary objects.
        normalized[key] = value
    return normalized


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
        value = _coerce_text(raw_item, field_name=field_name, max_length=max_item_length)  # AUDIT-FIX(#6): keep list elements textual and size-bounded.
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
        except Exception as exc:  # AUDIT-FIX(#4): convert unexpected service/config/store failures into a stable tool error.
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
    # AUDIT-FIX(#6): reject arbitrary object stringification and enforce input size bounds.
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


def _coerce_bool(raw_value: object, *, field_name: str, default: bool | None = None) -> bool:
    # AUDIT-FIX(#1): parse booleans explicitly so "false" does not become True via Python truthiness.
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


def _optional_positive_int(raw_value: object, *, field_name: str) -> int | None:
    # AUDIT-FIX(#5): accept only positive integer versions and reject bool/int coercion traps.
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


def _remaining_questions(session: Any) -> int:
    # AUDIT-FIX(#9): prefer dialogue-provided totals over a hard-coded question count.
    answered = len(getattr(session, "answered_question_ids", ()) or ())
    for attribute_name in ("total_questions", "required_questions", "question_count"):
        raw_total = getattr(session, attribute_name, None)
        if isinstance(raw_total, int) and raw_total >= 0:
            return max(0, raw_total - answered)
    questions = getattr(session, "questions", None)
    if isinstance(questions, (list, tuple)):
        return max(0, len(questions) - answered)
    return max(0, _DEFAULT_REMAINING_QUESTIONS - answered)


def _require_owner_config(owner: Any) -> Any:
    # AUDIT-FIX(#4): fail fast with a controlled error when the runtime owner is misconfigured.
    config = getattr(owner, "config", None)
    if config is None:
        raise RuntimeError("owner config is not available")
    return config


def _runtime_signature(config: Any) -> tuple[object, object, object, object]:
    return (
        id(config),  # AUDIT-FIX(#3): rebuild cached helpers if the config object itself is replaced.
        getattr(config, "automation_store_path", None),
        getattr(config, "local_timezone_name", None),
        getattr(config, "automation_max_entries", None),
    )


def _get_dependency(instance: object, attribute_name: str) -> object:
    try:
        return getattr(instance, attribute_name)
    except Exception:
        return object()


def _enum_value(raw_value: object) -> str:
    value = getattr(raw_value, "value", raw_value)
    return str(value or "").strip()


def _internal_text(raw_value: object, *, max_length: int) -> str | None:
    # AUDIT-FIX(#7): internal payload decoration should degrade safely instead of turning malformed metadata into a hard failure.
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


def _format_datetime(owner: Any, field_name: str, raw_value: object) -> str | None:
    # AUDIT-FIX(#7): never emit ambiguous naive timestamps as if they were trustworthy activation times.
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


def _safe_emit(owner: Any, event: str) -> None:
    try:
        owner.emit(event)
    except Exception as exc:
        _LOGGER.debug("self-coding telemetry emit failed: %s", type(exc).__name__)  # AUDIT-FIX(#10): preserve silent best-effort behavior while leaving breadcrumbs for operators.


def _safe_emit_kv(owner: Any, key: str, value: object) -> None:
    safe_value = _truncate_text(str(value or ""), _MAX_TELEMETRY_VALUE_LENGTH)
    _safe_emit(owner, f"{key}={safe_value}")


def _safe_record_event(owner: Any, event_name: str, description: str, **payload: object) -> None:
    try:
        owner._record_event(event_name, description, **payload)
    except Exception as exc:
        _LOGGER.debug("self-coding event recording failed: %s", type(exc).__name__)  # AUDIT-FIX(#10): keep telemetry non-fatal but diagnosable.