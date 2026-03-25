"""Build and cache self-coding runtime helpers for realtime tool handlers."""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping
from typing import Any, TypedDict

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
)
from twinr.automations import AutomationStore

_MAX_ID_LENGTH = 256
_MAX_JSON_DEPTH = 6
_MAX_LIST_ITEMS = 64
_MAX_MAPPING_ENTRIES = 64
_MAX_RUNTIME_SIGNATURE_ITEMS = 128
_MAX_RUNTIME_SIGNATURE_REPR_LENGTH = 256

SELF_CODING_RUNTIME_LOCK = threading.RLock()


class SelfCodingRuntime(TypedDict):
    """Cached self-coding runtime helpers stored on one owner."""

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


def ensure_self_coding_runtime(owner: Any) -> SelfCodingRuntime:
    """Build and cache the focused self-coding helpers on one runtime owner."""

    with SELF_CODING_RUNTIME_LOCK:
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


def resolve_learning_flow(owner: Any) -> SelfCodingLearningFlow:
    """Return the cached self-coding learning flow for one owner."""

    return ensure_self_coding_runtime(owner)["flow"]


def resolve_activation_service(owner: Any) -> SelfCodingActivationService:
    """Return the cached self-coding activation service for one owner."""

    return ensure_self_coding_runtime(owner)["activation_service"]


def ensure_confirm_activation_job_ready(owner: Any, *, job_id: str) -> str:
    """Best-effort compile a queued self-coding job before activation."""

    runtime = ensure_self_coding_runtime(owner)
    store = runtime["store"]
    compile_worker = runtime["compile_worker"]
    job = store.load_job(job_id)
    status = getattr(getattr(job, "status", None), "value", getattr(job, "status", ""))
    normalized_status = _truncate_text(str(status).strip(), _MAX_ID_LENGTH)
    if normalized_status == "queued":
        completed = compile_worker.run_job(job.job_id)
        completed_job_id = _bounded_text(getattr(completed, "job_id", None), max_length=_MAX_ID_LENGTH)
        if completed_job_id:
            return completed_job_id
    return job_id


def _require_owner_config(owner: Any) -> Any:
    config = getattr(owner, "config", None)
    if config is None:
        raise RuntimeError("owner config is not available")
    return config


def _runtime_signature(config: Any) -> tuple[object, ...]:
    """Fingerprint public config values so cached helpers rebuild after drift."""

    raw_state = getattr(config, "__dict__", None)
    if isinstance(raw_state, Mapping):
        items = tuple(
            (key, raw_state[key])
            for key in sorted(raw_state)
            if not str(key).startswith("_")
        )[:_MAX_RUNTIME_SIGNATURE_ITEMS]
        return (
            type(config),
            tuple((str(key), _freeze_signature_value(value, depth=0)) for key, value in items),
        )
    return (
        type(config),
        getattr(config, "automation_store_path", None),
        getattr(config, "local_timezone_name", None),
        getattr(config, "automation_max_entries", None),
    )


def _freeze_signature_value(raw_value: object, *, depth: int) -> object:
    """Create a bounded, hashable fingerprint for config values."""

    if depth >= _MAX_JSON_DEPTH:
        return "<max-depth>"
    if raw_value is None or isinstance(raw_value, (bool, int, str)):
        return raw_value
    if isinstance(raw_value, float):
        return raw_value if math.isfinite(raw_value) else "<non-finite-float>"
    if isinstance(raw_value, (bytes, bytearray)):
        return _bounded_text(raw_value, max_length=_MAX_RUNTIME_SIGNATURE_REPR_LENGTH)
    if isinstance(raw_value, Mapping):
        items: list[tuple[str, object]] = []
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
    return _safe_repr(raw_value, max_length=_MAX_RUNTIME_SIGNATURE_REPR_LENGTH)


def _get_dependency(instance: object, attribute_name: str) -> object:
    try:
        return getattr(instance, attribute_name)
    except Exception:
        return object()


def _bounded_text(raw_value: object, *, max_length: int) -> str | None:
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


def _truncate_text(text: str, max_length: int) -> str:
    normalized = " ".join(text.split())
    return normalized[:max_length]


def _safe_repr(raw_value: object, *, max_length: int) -> str:
    try:
        text = repr(raw_value)
    except Exception:
        text = f"<unrepresentable:{type(raw_value).__name__}>"
    return _truncate_text(text, max_length)


__all__ = [
    "SELF_CODING_RUNTIME_LOCK",
    "SelfCodingRuntime",
    "ensure_confirm_activation_job_ready",
    "ensure_self_coding_runtime",
    "resolve_activation_service",
    "resolve_learning_flow",
]
