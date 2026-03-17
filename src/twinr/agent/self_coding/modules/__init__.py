"""Expose the curated MVP self_coding module-library surface.

These modules are the trusted, Codex-readable shim layer for the current ASE
MVP. They intentionally expose semantic primitives only; real execution stays
disabled until the later sandboxed skill runner exists.
"""

from __future__ import annotations

from twinr.agent.self_coding.modules.base import (
    SelfCodingModuleFunction,
    SelfCodingModuleRuntimeUnavailableError,
    SelfCodingModuleSpec,
)
from twinr.agent.self_coding.modules.calendar import MODULE_SPEC as CALENDAR_MODULE
from twinr.agent.self_coding.modules.camera import MODULE_SPEC as CAMERA_MODULE
from twinr.agent.self_coding.modules.email import MODULE_SPEC as EMAIL_MODULE
from twinr.agent.self_coding.modules.llm_call import MODULE_SPEC as LLM_CALL_MODULE
from twinr.agent.self_coding.modules.memory import MODULE_SPEC as MEMORY_MODULE
from twinr.agent.self_coding.modules.pir import MODULE_SPEC as PIR_MODULE
from twinr.agent.self_coding.modules.rules import MODULE_SPEC as RULES_MODULE
from twinr.agent.self_coding.modules.safety import MODULE_SPEC as SAFETY_MODULE
from twinr.agent.self_coding.modules.scheduler import MODULE_SPEC as SCHEDULER_MODULE
from twinr.agent.self_coding.modules.speaker import MODULE_SPEC as SPEAKER_MODULE
from twinr.agent.self_coding.modules.web_search import MODULE_SPEC as WEB_SEARCH_MODULE

MODULE_LIBRARY: tuple[SelfCodingModuleSpec, ...] = (
    CAMERA_MODULE,
    PIR_MODULE,
    SPEAKER_MODULE,
    WEB_SEARCH_MODULE,
    LLM_CALL_MODULE,
    MEMORY_MODULE,
    SCHEDULER_MODULE,
    RULES_MODULE,
    SAFETY_MODULE,
    EMAIL_MODULE,
    CALENDAR_MODULE,
)

_MODULES_BY_ID = {spec.capability_id: spec for spec in MODULE_LIBRARY}
_MODULES_BY_NAME = {spec.module_name: spec for spec in MODULE_LIBRARY}


def module_spec_for(capability_or_module_id: str) -> SelfCodingModuleSpec | None:
    """Return one curated module spec by capability id or module name."""

    normalized = str(capability_or_module_id or "").strip().lower()
    if not normalized:
        return None
    return _MODULES_BY_ID.get(normalized) or _MODULES_BY_NAME.get(normalized)


__all__ = [
    "MODULE_LIBRARY",
    "SelfCodingModuleFunction",
    "SelfCodingModuleRuntimeUnavailableError",
    "SelfCodingModuleSpec",
    "module_spec_for",
]
