"""Turn compiled skill packages into hidden automation entries."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import is_dataclass, replace
from datetime import datetime
import hashlib
import json
import re
from typing import Any

from twinr.agent.self_coding.sandbox.policy import CapabilityBrokerManifest
from twinr.agent.self_coding.runtime.contracts import SkillPackage
from twinr.automations import (
    AutomationAction,
    AutomationDefinition,
    TimeAutomationTrigger,
)
from twinr.automations.sensors import build_sensor_trigger

# AUDIT-FIX(#1): Constrain generated automation IDs to a filesystem- and log-safe subset.
_SAFE_AUTOMATION_ID_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")
# AUDIT-FIX(#6): Strip control characters from human-readable fields before persisting them.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1F\x7F]")

# AUDIT-FIX(#1): Keep raw untrusted identifiers bounded so malformed packages cannot generate pathological metadata.
_MAX_RAW_IDENTIFIER_LENGTH = 512
# AUDIT-FIX(#1): Bound each generated automation ID component so the final ID stays within common persistence limits.
_MAX_SAFE_ID_COMPONENT_LENGTH = 96
# AUDIT-FIX(#6): Keep hidden automation labels bounded for admin UIs and logs.
_MAX_DISPLAY_TEXT_LENGTH = 160
# AUDIT-FIX(#6): Keep descriptive metadata bounded and log-safe.
_MAX_DESCRIPTION_LENGTH = 2048
# AUDIT-FIX(#6): Prevent oversized module names from poisoning persisted metadata.
_MAX_ENTRY_MODULE_LENGTH = 255


def _validate_version(value: int) -> int:
    """Return a validated package version."""
    # AUDIT-FIX(#2): Reject lossy int(...) coercions such as True -> 1 or 1.9 -> 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"version must be a non-bool int, got {type(value).__name__}")
    if value < 0:
        raise ValueError("version must be >= 0")
    return value


def _require_identifier_text(
    value: str,
    *,
    field_name: str,
    max_length: int = _MAX_RAW_IDENTIFIER_LENGTH,
) -> str:
    """Validate a raw identifier-like text value without rewriting its semantic contents."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a str, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    if len(value) > max_length:
        raise ValueError(f"{field_name} exceeds max length {max_length}")
    if _CONTROL_CHAR_RE.search(value):
        raise ValueError(f"{field_name} contains control characters")
    return value


def _sanitize_display_text(
    value: str,
    *,
    field_name: str,
    max_length: int = _MAX_DISPLAY_TEXT_LENGTH,
    allow_empty: bool = False,
) -> str:
    """Normalize human-readable text for hidden admin surfaces."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a str, got {type(value).__name__}")
    # AUDIT-FIX(#6): Remove control characters and collapse whitespace before names/descriptions hit logs or admin UIs.
    sanitized = " ".join(_CONTROL_CHAR_RE.sub(" ", value).split())
    if not sanitized and not allow_empty:
        raise ValueError(f"{field_name} must not be blank")
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip()
    return sanitized


def _sanitize_optional_display_text(
    value: str | None,
    *,
    field_name: str,
    max_length: int = _MAX_DESCRIPTION_LENGTH,
) -> str | None:
    """Normalize optional human-readable text."""
    if value is None:
        return None
    sanitized = _sanitize_display_text(
        value,
        field_name=field_name,
        max_length=max_length,
        allow_empty=True,
    )
    return sanitized or None


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _safe_automation_id_component(value: str, *, field_name: str) -> str:
    """Return a deterministic safe component for automation IDs."""
    raw_value = _require_identifier_text(value, field_name=field_name)
    safe_value = _SAFE_AUTOMATION_ID_COMPONENT_RE.sub("_", raw_value.strip())
    safe_value = re.sub(r"_+", "_", safe_value).strip("._-")
    digest = _stable_hash(raw_value)

    # AUDIT-FIX(#1): Add a hash suffix whenever sanitization changes the original token so different raw IDs do not collapse.
    needs_hash_suffix = not safe_value or safe_value != raw_value or len(safe_value) > _MAX_SAFE_ID_COMPONENT_LENGTH
    if not safe_value:
        safe_value = "id"
    if needs_hash_suffix:
        prefix = safe_value[: _MAX_SAFE_ID_COMPONENT_LENGTH - len(digest) - 1].rstrip("._-")
        if not prefix:
            prefix = "id"
        safe_value = f"{prefix}_{digest}"
    return safe_value


def _as_tuple(value: Any, *, field_name: str) -> tuple[Any, ...]:
    """Return an iterable value as a tuple with explicit validation."""
    if value is None:
        raise ValueError(f"{field_name} must not be None")
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of trigger objects, not {type(value).__name__}")
    try:
        return tuple(value)
    except TypeError as exc:  # pragma: no cover - defensive contract guard
        raise TypeError(f"{field_name} must be iterable") from exc


def _find_duplicates(values: Iterable[str]) -> tuple[str, ...]:
    """Return duplicates while preserving first-seen order."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)


def _stable_unique_sorted(values: Iterable[str], *, field_name: str) -> list[str]:
    """Return a deterministic, duplicate-free, sorted list of strings."""
    materialized = list(values)
    duplicates = _find_duplicates(materialized)
    if duplicates:
        raise ValueError(f"{field_name} contains duplicate values: {', '.join(repr(value) for value in duplicates)}")
    return sorted(materialized)


def _clone_trigger(trigger: Any) -> Any:
    """Clone a trigger definition without assuming one concrete implementation type."""
    # AUDIT-FIX(#5): Keep clone semantics but avoid hard-wiring dataclasses.replace() onto every future trigger class.
    if is_dataclass(trigger) and not isinstance(trigger, type):
        return replace(trigger)
    return deepcopy(trigger)


def _validate_scheduled_trigger(
    trigger: Any,
    *,
    skill_id: str,
    trigger_id: str,
) -> None:
    """Validate timezone-sensitive schedule fields before they are persisted."""
    due_at = getattr(trigger, "due_at", None)
    time_of_day = getattr(trigger, "time_of_day", None)
    weekdays = getattr(trigger, "weekdays", None)
    timezone_name = getattr(trigger, "timezone_name", None)

    # AUDIT-FIX(#4): Reject naive datetimes early so DST/UTC conversion bugs do not silently shift senior reminders.
    if isinstance(due_at, datetime) and (
        due_at.tzinfo is None or due_at.tzinfo.utcoffset(due_at) is None
    ):
        raise ValueError(
            f"Scheduled trigger {trigger_id!r} for skill {skill_id!r} uses a timezone-naive due_at"
        )

    # AUDIT-FIX(#4): Require an explicit timezone when scheduling local clock times or weekday recurrences.
    if (time_of_day is not None or weekdays) and (
        not isinstance(timezone_name, str) or not timezone_name.strip()
    ):
        raise ValueError(
            f"Scheduled trigger {trigger_id!r} for skill {skill_id!r} requires timezone_name"
        )

    if isinstance(timezone_name, str) and timezone_name.strip():
        _require_identifier_text(
            timezone_name,
            field_name=f"scheduled trigger {trigger_id!r} timezone_name",
            max_length=128,
        )


def skill_package_automation_entries(
    *,
    skill_id: str,
    skill_name: str,
    version: int,
    package: SkillPackage,
) -> tuple[AutomationDefinition, ...]:
    """Return disabled hidden automations that drive one activated skill package."""

    # AUDIT-FIX(#2): Validate version once and reuse the canonical integer everywhere.
    validated_version = _validate_version(version)
    raw_skill_id = _require_identifier_text(skill_id, field_name="skill_id")
    # AUDIT-FIX(#1): Generate safe automation IDs even when upstream IDs include spaces, slashes, or long AI-generated text.
    safe_skill_component = _safe_automation_id_component(raw_skill_id, field_name="skill_id")
    # AUDIT-FIX(#6): Normalize human-readable names/descriptions before they reach internal UIs and logs.
    safe_skill_name = _sanitize_display_text(skill_name, field_name="skill_name")
    description = _sanitize_optional_display_text(
        getattr(package, "description", None),
        field_name="package.description",
    ) or ""
    scheduled_triggers = _as_tuple(
        package.scheduled_triggers,
        field_name="package.scheduled_triggers",
    )
    sensor_triggers = _as_tuple(
        package.sensor_triggers,
        field_name="package.sensor_triggers",
    )

    entries: list[AutomationDefinition] = []
    # AUDIT-FIX(#3): Detect ID collisions before a later store silently overwrites one hidden automation with another.
    seen_automation_ids: set[str] = set()
    base_id = f"ase_{safe_skill_component}_v{validated_version}"

    for trigger in scheduled_triggers:
        raw_trigger_id = _require_identifier_text(
            trigger.trigger_id,
            field_name="scheduled trigger_id",
        )
        trigger_label = _sanitize_display_text(
            raw_trigger_id,
            field_name=f"scheduled trigger {raw_trigger_id!r} label",
            max_length=80,
        )
        safe_trigger_component = _safe_automation_id_component(
            raw_trigger_id,
            field_name=f"scheduled trigger {raw_trigger_id!r} trigger_id",
        )
        automation_id = f"{base_id}_schedule_{safe_trigger_component}"
        if automation_id in seen_automation_ids:
            raise ValueError(f"Duplicate generated automation_id {automation_id!r} for skill {raw_skill_id!r}")
        seen_automation_ids.add(automation_id)

        _validate_scheduled_trigger(
            trigger,
            skill_id=raw_skill_id,
            trigger_id=raw_trigger_id,
        )

        try:
            automation_trigger = TimeAutomationTrigger(
                schedule=trigger.schedule,
                due_at=trigger.due_at,
                time_of_day=trigger.time_of_day,
                weekdays=deepcopy(trigger.weekdays),
                timezone_name=trigger.timezone_name,
            )
        except Exception as exc:  # pragma: no cover - depends on upstream trigger model validation
            raise ValueError(
                f"Invalid scheduled trigger {raw_trigger_id!r} for skill {raw_skill_id!r}"
            ) from exc

        entries.append(
            AutomationDefinition(
                automation_id=automation_id,
                name=f"{safe_skill_name} schedule {trigger_label}",
                description=description,
                enabled=False,
                trigger=automation_trigger,
                actions=(
                    AutomationAction(
                        kind="tool_call",
                        tool_name="run_self_coding_skill_scheduled",
                        payload={
                            "skill_id": raw_skill_id,
                            "version": validated_version,
                            "trigger_id": raw_trigger_id,
                        },
                        enabled=True,
                    ),
                ),
                source="self_coding",
                tags=("self_coding", "skill_package", "hidden"),
            )
        )

    for trigger in sensor_triggers:
        raw_trigger_id = _require_identifier_text(
            trigger.trigger_id,
            field_name="sensor trigger_id",
        )
        trigger_label = _sanitize_display_text(
            raw_trigger_id,
            field_name=f"sensor trigger {raw_trigger_id!r} label",
            max_length=80,
        )
        safe_trigger_component = _safe_automation_id_component(
            raw_trigger_id,
            field_name=f"sensor trigger {raw_trigger_id!r} trigger_id",
        )
        automation_id = f"{base_id}_sensor_{safe_trigger_component}"
        if automation_id in seen_automation_ids:
            raise ValueError(f"Duplicate generated automation_id {automation_id!r} for skill {raw_skill_id!r}")
        seen_automation_ids.add(automation_id)

        try:
            sensor_trigger = build_sensor_trigger(
                trigger.sensor_trigger_kind,
                hold_seconds=trigger.hold_seconds,
                cooldown_seconds=trigger.cooldown_seconds,
            )
            automation_trigger = _clone_trigger(sensor_trigger)
        except Exception as exc:  # pragma: no cover - depends on upstream trigger builder behavior
            # AUDIT-FIX(#5): Re-raise with skill/trigger context so activation failures are diagnosable in production.
            raise ValueError(
                f"Invalid sensor trigger {raw_trigger_id!r} for skill {raw_skill_id!r}"
            ) from exc

        event_name = getattr(sensor_trigger, "event_name", None)
        if event_name is not None:
            event_name = _require_identifier_text(
                event_name,
                field_name=f"sensor trigger {raw_trigger_id!r} event_name",
            )

        entries.append(
            AutomationDefinition(
                automation_id=automation_id,
                name=f"{safe_skill_name} sensor {trigger_label}",
                description=description,
                enabled=False,
                trigger=automation_trigger,
                actions=(
                    AutomationAction(
                        kind="tool_call",
                        tool_name="run_self_coding_skill_sensor",
                        payload={
                            "skill_id": raw_skill_id,
                            "version": validated_version,
                            "trigger_id": raw_trigger_id,
                            "event_name": event_name,
                        },
                        enabled=True,
                    ),
                ),
                source="self_coding",
                tags=("self_coding", "skill_package", "hidden"),
            )
        )
    return tuple(entries)


def skill_package_activation_metadata(
    *,
    package: SkillPackage,
    automation_entries: tuple[AutomationDefinition, ...],
    policy_manifest: CapabilityBrokerManifest | None = None,
) -> dict[str, Any]:
    """Return stable metadata persisted on activation records for skill packages."""

    entry_module = _require_identifier_text(
        package.entry_module,
        field_name="package.entry_module",
        max_length=_MAX_ENTRY_MODULE_LENGTH,
    )
    automation_ids = _stable_unique_sorted(
        [
            _require_identifier_text(
                entry.automation_id,
                field_name="automation entry automation_id",
            )
            for entry in _as_tuple(automation_entries, field_name="automation_entries")
        ],
        field_name="automation_entries.automation_id",
    )
    scheduled_trigger_ids = _stable_unique_sorted(
        [
            _require_identifier_text(
                trigger.trigger_id,
                field_name="scheduled trigger_id",
            )
            for trigger in _as_tuple(
                package.scheduled_triggers,
                field_name="package.scheduled_triggers",
            )
        ],
        field_name="package.scheduled_triggers.trigger_id",
    )
    sensor_trigger_ids = _stable_unique_sorted(
        [
            _require_identifier_text(
                trigger.trigger_id,
                field_name="sensor trigger_id",
            )
            for trigger in _as_tuple(
                package.sensor_triggers,
                field_name="package.sensor_triggers",
            )
        ],
        field_name="package.sensor_triggers.trigger_id",
    )

    metadata = {
        "artifact_kind": "skill_package",
        # AUDIT-FIX(#6): Validate persisted module metadata before it is stored.
        "entry_module": entry_module,
        # AUDIT-FIX(#7): Canonicalize metadata collections so repeated activations produce stable persisted records.
        "automation_ids": automation_ids,
        "scheduled_trigger_ids": scheduled_trigger_ids,
        "sensor_trigger_ids": sensor_trigger_ids,
    }
    if policy_manifest is not None:
        try:
            sandbox_policy_manifest = json.loads(
                json.dumps(policy_manifest.to_payload(), sort_keys=True)
            )
            # AUDIT-FIX(#8): Sort and duplicate-check allowed methods so persisted policy metadata is deterministic.
            allowed_methods = _stable_unique_sorted(
                [
                    _require_identifier_text(
                        method,
                        field_name="policy_manifest.allowed_methods[]",
                        max_length=32,
                    )
                    for method in policy_manifest.allowed_methods
                ],
                field_name="policy_manifest.allowed_methods",
            )
        except Exception as exc:  # pragma: no cover - depends on upstream manifest implementation
            # AUDIT-FIX(#8): Surface policy manifest serialization/ordering problems with explicit activation context.
            raise ValueError("Invalid capability broker policy manifest") from exc
        metadata["sandbox_policy_manifest"] = sandbox_policy_manifest
        metadata["sandbox_allowed_methods"] = allowed_methods
    return metadata


__all__ = [
    "skill_package_activation_metadata",
    "skill_package_automation_entries",
]