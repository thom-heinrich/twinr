"""Build operator-safe automation page models and form persistence helpers.

This module groups stored automations into UI families, renders stable edit
sections, and validates web form submissions before they reach `AutomationStore`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePath
from threading import RLock
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from twinr.automations import (
    AutomationAction,
    AutomationDefinition,
    AutomationStore,
    IfThenAutomationTrigger,
    SensorTriggerSpec,
    TimeAutomationTrigger,
    build_sensor_trigger,
    describe_sensor_trigger,
    describe_sensor_trigger_text,
    supported_sensor_trigger_kinds,
)
from twinr.integrations import IntegrationAutomationFamilyBlock
from twinr.memory.reminders import format_due_label, now_in_timezone, parse_due_at
from twinr.web.support.store import FileBackedSetting

_BOOL_OPTIONS = (("true", "Enabled"), ("false", "Disabled"))
_DELIVERY_OPTIONS = (("spoken", "Speak it"), ("printed", "Print it"))
_CONTENT_MODE_OPTIONS = (
    ("llm_prompt", "Generate fresh content"),
    ("static_text", "Use this exact text"),
)
_TIME_SCHEDULE_OPTIONS = (
    ("once", "One time"),
    ("daily", "Every day"),
    ("weekly", "Specific weekdays"),
)
_WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_WEEKDAY_ALIASES = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "mo": 0,
    "mon": 0,
    "monday": 0,
    "montag": 0,
    "di": 1,
    "tu": 1,
    "tue": 1,
    "tues": 1,
    "tuesday": 1,
    "dienstag": 1,
    "mi": 2,
    "we": 2,
    "wed": 2,
    "wednesday": 2,
    "mittwoch": 2,
    "do": 3,
    "th": 3,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "thursday": 3,
    "donnerstag": 3,
    "fr": 4,
    "fri": 4,
    "friday": 4,
    "freitag": 4,
    "sa": 5,
    "sat": 5,
    "saturday": 5,
    "samstag": 5,
    "so": 6,
    "su": 6,
    "sun": 6,
    "sunday": 6,
    "sonntag": 6,
}

_ALLOWED_DELIVERIES = {"spoken", "printed"}
_ALLOWED_CONTENT_MODES = {"llm_prompt", "static_text"}
_ALLOWED_TIME_SCHEDULES = {"once", "daily", "weekly"}
_DEFAULT_SENSOR_COOLDOWN_SECONDS = 60.0
_DEFAULT_TIMEZONE_NAME = "UTC"
_MAX_NAME_LENGTH = 120
_MAX_DESCRIPTION_LENGTH = 220
_MAX_CONTENT_LENGTH = 4000
_STORE_MUTATION_LOCK = RLock()


@dataclass(frozen=True, slots=True)
class AutomationFamilyDefinition:
    """Describe one automation family shown on the automations page."""

    key: str
    title: str
    summary: str
    detail: str
    create_supported: bool = False
    reserved_note: str | None = None
    status_key: str | None = None
    status_label: str | None = None


@dataclass(frozen=True, slots=True)
class AutomationFamilyCard:
    """Summarize one automation family for the overview cards."""

    key: str
    title: str
    count: int
    summary: str
    detail: str
    status_key: str
    status_label: str


@dataclass(frozen=True, slots=True)
class AutomationRow:
    """Represent one stored automation row in the family tables."""

    automation_id: str
    name: str
    description: str | None
    family_key: str
    family_title: str
    enabled: bool
    status_key: str
    status_label: str
    trigger_summary: str
    action_summary: str
    next_run_label: str | None
    last_triggered_label: str | None
    source: str
    tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AutomationFamilySection:
    """Group one family definition with its rendered rows."""

    definition: AutomationFamilyDefinition
    rows: tuple[AutomationRow, ...]


@dataclass(frozen=True, slots=True)
class AutomationFormSection:
    """Describe one titled automation form section."""

    title: str
    description: str
    fields: tuple[FileBackedSetting, ...]


def build_automation_page_context(
    store: AutomationStore,
    *,
    timezone_name: str,
    edit_ref: str | None = None,
    integration_blocks: tuple[IntegrationAutomationFamilyBlock, ...] = (),
) -> dict[str, Any]:
    """Build the template context for the automations page.

    Args:
        store: Automation store backing the page.
        timezone_name: IANA timezone used for operator-facing schedule labels.
        edit_ref: Optional automation id or unique name to open in edit mode.
        integration_blocks: Extra family definitions provided by integrations.

    Returns:
        Template-ready context dictionary for `automations_page.html`.
    """

    # AUDIT-FIX(#3): Serialize same-process reads against file-backed mutations to avoid torn snapshots.
    with _STORE_MUTATION_LOCK:
        entries = store.load_entries()
    definitions = _family_definitions(integration_blocks=integration_blocks)
    grouped_rows = _build_family_rows(
        entries,
        store=store,
        timezone_name=timezone_name,
        integration_blocks=integration_blocks,
    )
    family_sections = tuple(
        AutomationFamilySection(definition=definition, rows=grouped_rows.get(definition.key, ()))
        for definition in definitions
        if definition.key != "other" or grouped_rows.get("other")
    )
    family_cards = tuple(
        AutomationFamilyCard(
            key=definition.key,
            title=definition.title,
            count=len(grouped_rows.get(definition.key, ())),
            summary=definition.summary,
            detail=definition.detail,
            status_key=_definition_status_key(definition, has_rows=bool(grouped_rows.get(definition.key))),
            status_label=_definition_status_label(definition, has_rows=bool(grouped_rows.get(definition.key))),
        )
        for definition in definitions
        if definition.key != "other" or grouped_rows.get("other")
    )
    edit_entry = _find_entry(entries, edit_ref)
    # AUDIT-FIX(#2): Do not open reserved or non-operator-managed automations in the generic edit forms.
    editing_time_entry = edit_entry if _can_operator_edit(edit_entry, expected_kind="time") else None
    editing_sensor_entry = edit_entry if _can_operator_edit(edit_entry, expected_kind="sensor") else None
    sensor_options = _sensor_trigger_options()
    return {
        # AUDIT-FIX(#9): Expose only the store filename, not the full on-device filesystem path.
        "automation_store_path": _store_path_label(store),
        "automation_entries_total": len(entries),
        "family_cards": family_cards,
        "family_sections": family_sections,
        "time_form_mode": "edit" if editing_time_entry is not None else "create",
        "sensor_form_mode": "edit" if editing_sensor_entry is not None else "create",
        "time_form_title": "Edit scheduled automation" if editing_time_entry is not None else "Add scheduled automation",
        "sensor_form_title": "Edit sensor automation" if editing_sensor_entry is not None else "Add sensor automation",
        "editing_time_id": editing_time_entry.automation_id if editing_time_entry is not None else "",
        "editing_sensor_id": editing_sensor_entry.automation_id if editing_sensor_entry is not None else "",
        "editing_time_name": editing_time_entry.name if editing_time_entry is not None else None,
        "editing_sensor_name": editing_sensor_entry.name if editing_sensor_entry is not None else None,
        "time_form_sections": _time_form_sections(editing_time_entry, timezone_name=timezone_name),
        "sensor_form_sections": _sensor_form_sections(editing_sensor_entry, sensor_options=sensor_options),
        "sensor_trigger_options": sensor_options,
        "supported_sensor_labels": tuple(label for _key, label in sensor_options),
    }


def save_time_automation(
    store: AutomationStore,
    form: dict[str, str],
    *,
    timezone_name: str,
) -> AutomationDefinition:
    """Create or update one scheduled automation from web form data."""

    automation_id = _normalize_inline_text(form.get("automation_id"))
    # AUDIT-FIX(#4): Validate persisted text without silently truncating or flattening operator-provided content.
    name = _required_text(form.get("name"), field_name="name", limit=_MAX_NAME_LENGTH)
    description = _optional_input_text(
        form.get("description"),
        field_name="description",
        limit=_MAX_DESCRIPTION_LENGTH,
        multiline=True,
    )
    schedule = _parse_choice(form.get("schedule"), field_name="schedule", allowed=_ALLOWED_TIME_SCHEDULES, default="once")
    due_at = _optional_input_text(form.get("due_at"), field_name="due at", limit=80)
    time_of_day = _optional_input_text(form.get("time_of_day"), field_name="time of day", limit=8)
    weekdays = _parse_weekdays(form.get("weekdays_text", ""))
    enabled = _parse_bool(form.get("enabled"), default=True)
    tags = _parse_tags(form.get("tags_text", ""))
    actions = (_build_action(form),)
    # AUDIT-FIX(#1): Validate schedule/time/timezone inputs and clear irrelevant fields before persisting.
    configured_timezone = _validated_timezone_name(timezone_name, field_name="configured timezone")
    timezone_value = _validated_timezone_name(
        form.get("timezone_name"),
        field_name="timezone",
        fallback=configured_timezone,
    )
    schedule, due_at, time_of_day, weekdays = _normalize_time_schedule(
        schedule=schedule,
        due_at=due_at,
        time_of_day=time_of_day,
        weekdays=weekdays,
        timezone_name=timezone_value,
    )
    if automation_id:
        # AUDIT-FIX(#3): Serialize read-modify-write operations against the file-backed store.
        with _STORE_MUTATION_LOCK:
            entry = store.get(automation_id)
            if entry is None:
                raise ValueError("Unknown automation.")
            # AUDIT-FIX(#2): Only operator-managed scheduled automations may be edited from this form.
            _ensure_operator_editable(entry, expected_kind="time")
            trigger = TimeAutomationTrigger(
                schedule=schedule,
                due_at=due_at,
                time_of_day=time_of_day,
                weekdays=weekdays,
                timezone_name=timezone_value,
            )
            try:
                return store.update(
                    automation_id,
                    name=name,
                    description=description,
                    enabled=enabled,
                    trigger=trigger,
                    actions=actions,
                    tags=tags,
                )
            except KeyError as exc:
                raise ValueError("Unknown automation.") from exc
    # AUDIT-FIX(#3): Serialize create operations against the file-backed store.
    with _STORE_MUTATION_LOCK:
        return store.create_time_automation(
            name=name,
            description=description,
            enabled=enabled,
            schedule=schedule,
            due_at=due_at,
            time_of_day=time_of_day,
            weekdays=weekdays,
            timezone_name=timezone_value,
            actions=actions,
            source="web_ui",
            tags=tags,
        )


def save_sensor_automation(store: AutomationStore, form: dict[str, str]) -> AutomationDefinition:
    """Create or update one sensor automation from web form data."""

    automation_id = _normalize_inline_text(form.get("automation_id"))
    # AUDIT-FIX(#4): Validate persisted text without silently truncating or flattening operator-provided content.
    name = _required_text(form.get("name"), field_name="name", limit=_MAX_NAME_LENGTH)
    description = _optional_input_text(
        form.get("description"),
        field_name="description",
        limit=_MAX_DESCRIPTION_LENGTH,
        multiline=True,
    )
    trigger_kind = _required_text(form.get("trigger_kind"), field_name="trigger kind", limit=80).lower()
    hold_seconds = _parse_nonnegative_float(form.get("hold_seconds"), default=0.0)
    # AUDIT-FIX(#6): Match the backend default to the UI default and reject non-finite numeric values.
    cooldown_seconds = _parse_nonnegative_float(
        form.get("cooldown_seconds"),
        default=_DEFAULT_SENSOR_COOLDOWN_SECONDS,
    )
    enabled = _parse_bool(form.get("enabled"), default=True)
    tags = _parse_tags(form.get("tags_text", ""))
    actions = (_build_action(form),)
    trigger = build_sensor_trigger(
        trigger_kind,
        hold_seconds=hold_seconds,
        cooldown_seconds=cooldown_seconds,
    )
    if automation_id:
        # AUDIT-FIX(#3): Serialize read-modify-write operations against the file-backed store.
        with _STORE_MUTATION_LOCK:
            entry = store.get(automation_id)
            if entry is None:
                raise ValueError("Unknown automation.")
            # AUDIT-FIX(#2): Only operator-managed sensor automations may be edited from this form.
            _ensure_operator_editable(entry, expected_kind="sensor")
            try:
                return store.update(
                    automation_id,
                    name=name,
                    description=description,
                    enabled=enabled,
                    trigger=trigger,
                    actions=actions,
                    tags=tags,
                )
            except KeyError as exc:
                raise ValueError("Unknown automation.") from exc
    # AUDIT-FIX(#3): Serialize create operations against the file-backed store.
    with _STORE_MUTATION_LOCK:
        return store.create_if_then_automation(
            name=name,
            description=description,
            enabled=enabled,
            actions=actions,
            event_name=trigger.event_name,
            all_conditions=trigger.all_conditions,
            any_conditions=trigger.any_conditions,
            cooldown_seconds=trigger.cooldown_seconds,
            source="web_ui",
            tags=tags,
        )


def toggle_automation_enabled(store: AutomationStore, automation_id: str) -> AutomationDefinition:
    """Flip the enabled flag for one operator-editable automation."""

    # AUDIT-FIX(#3): Serialize toggles against concurrent writes in the file-backed store.
    with _STORE_MUTATION_LOCK:
        entry = store.get(automation_id)
        if entry is None:
            raise ValueError("Unknown automation.")
        # AUDIT-FIX(#2): Prevent generic UI controls from mutating non-operator-managed automations.
        _ensure_operator_editable(entry)
        try:
            return store.update(automation_id, enabled=not _coerce_bool(getattr(entry, 'enabled', False), default=False))
        except KeyError as exc:
            raise ValueError("Unknown automation.") from exc


def delete_automation(store: AutomationStore, automation_id: str) -> AutomationDefinition:
    """Delete one operator-editable automation."""

    # AUDIT-FIX(#3): Serialize deletes against concurrent writes in the file-backed store.
    with _STORE_MUTATION_LOCK:
        entry = store.get(automation_id)
        if entry is None:
            raise ValueError("Unknown automation.")
        # AUDIT-FIX(#2): Prevent generic UI controls from deleting non-operator-managed automations.
        _ensure_operator_editable(entry)
        try:
            return store.delete(automation_id)
        except KeyError as exc:
            raise ValueError("Unknown automation.") from exc


def _family_definitions(
    *,
    integration_blocks: tuple[IntegrationAutomationFamilyBlock, ...] = (),
) -> tuple[AutomationFamilyDefinition, ...]:
    """Return the built-in plus integration-provided automation families."""

    definitions = [
        AutomationFamilyDefinition(
            key="time",
            title="Scheduled",
            summary="Run at a fixed time",
            detail="Daily briefings, one-off notices, or recurring printed content.",
            create_supported=True,
        ),
        AutomationFamilyDefinition(
            key="sensor",
            title="Sensor-triggered",
            summary="Run when motion, camera, or VAD facts match",
            detail="Idle-time reactions driven by PIR, camera, or background microphone facts.",
            create_supported=True,
        ),
    ]
    definitions.extend(
        AutomationFamilyDefinition(
            key=block.key,
            title=block.title,
            summary=block.summary,
            detail=block.detail,
            reserved_note=block.operator_note,
            status_key=block.status_key,
            status_label=block.status_label,
        )
        for block in integration_blocks
    )
    definitions.append(
        AutomationFamilyDefinition(
            key="other",
            title="Other rules",
            summary="Fallback bucket for custom rules",
            detail="This captures stored automation types that do not fit the current operator forms yet.",
        )
    )
    return tuple(definitions)


def _build_family_rows(
    entries: tuple[AutomationDefinition, ...],
    *,
    store: AutomationStore,
    timezone_name: str,
    integration_blocks: tuple[IntegrationAutomationFamilyBlock, ...] = (),
) -> dict[str, tuple[AutomationRow, ...]]:
    """Group stored automations into template-ready family rows."""

    grouped: dict[str, list[AutomationRow]] = {}
    definitions_by_key = {
        definition.key: definition for definition in _family_definitions(integration_blocks=integration_blocks)
    }
    current_time = _safe_now_in_timezone(timezone_name)
    for entry in entries:
        family_key = _family_key_for_entry(entry, integration_blocks=integration_blocks)
        family_title = definitions_by_key.get(family_key, AutomationFamilyDefinition(
            key=family_key,
            title=family_key.replace("_", " ").title(),
            summary="",
            detail="",
        )).title
        # AUDIT-FIX(#5): Isolate malformed stored automations so one bad record cannot take down the whole page.
        next_run = _safe_next_run_at(store, entry, now=current_time)
        entry_enabled = _coerce_bool(getattr(entry, "enabled", False), default=False)
        entry_name = _normalize_inline_text(getattr(entry, "name", "")) or str(getattr(entry, "automation_id", ""))
        grouped.setdefault(family_key, []).append(
            AutomationRow(
                automation_id=str(getattr(entry, "automation_id", "")),
                name=entry_name or "Unnamed automation",
                description=getattr(entry, "description", None) or None,
                family_key=family_key,
                family_title=family_title,
                enabled=entry_enabled,
                status_key="ok" if entry_enabled else "muted",
                status_label="Enabled" if entry_enabled else "Disabled",
                trigger_summary=_safe_trigger_summary(entry, timezone_name=timezone_name),
                action_summary=_safe_action_summary(getattr(entry, "actions", ()) or ()),
                next_run_label=_safe_due_label(next_run, timezone_name=timezone_name),
                last_triggered_label=_safe_due_label(
                    getattr(entry, "last_triggered_at", None),
                    timezone_name=timezone_name,
                ),
                source=str(getattr(entry, "source", "") or ""),
                tags=tuple(str(tag) for tag in (getattr(entry, "tags", ()) or ()) if str(tag).strip()),
            )
        )
    return {
        key: tuple(sorted(rows, key=lambda item: item.name.lower()))
        for key, rows in grouped.items()
    }


def _trigger_summary(entry: AutomationDefinition, *, timezone_name: str) -> str:
    """Summarize one automation trigger for operator display."""

    trigger = entry.trigger
    try:
        sensor_text = describe_sensor_trigger_text(trigger)
    except Exception:
        sensor_text = None
    if sensor_text:
        return sensor_text[0].upper() + sensor_text[1:]
    if not isinstance(trigger, TimeAutomationTrigger):
        return f"{str(getattr(trigger, 'kind', 'custom')).replace('_', ' ').title()} trigger"
    effective_timezone = _coerce_timezone_name(
        getattr(trigger, "timezone_name", None) or timezone_name,
        fallback=_coerce_timezone_name(timezone_name, fallback=_DEFAULT_TIMEZONE_NAME),
    )
    if trigger.schedule == "once":
        if not getattr(trigger, "due_at", None):
            return "One time schedule (missing date/time)"
        try:
            due_at = parse_due_at(str(trigger.due_at), timezone_name=effective_timezone)
        except Exception:
            return "One time schedule (invalid date/time)"
        due_label = _safe_due_label(due_at, timezone_name=effective_timezone)
        return f"One time at {due_label or 'unknown time'}"
    time_label = _validated_time_of_day_or_fallback(getattr(trigger, "time_of_day", None))
    if trigger.schedule == "daily":
        return f"Every day at {time_label}"
    weekday_labels = ", ".join(_weekday_label(index, lower=False) for index in (getattr(trigger, "weekdays", ()) or ()))
    if not weekday_labels:
        return f"Weekly at {time_label}"
    return f"Every {weekday_labels} at {time_label}"


def _action_summary(actions: tuple[AutomationAction, ...]) -> str:
    """Summarize enabled actions for operator display."""

    enabled_actions = [action for action in actions if _coerce_bool(getattr(action, "enabled", True), default=True)]
    if not enabled_actions:
        return "No enabled action"
    parts: list[str] = []
    for action in enabled_actions:
        if action.kind == "llm_prompt":
            payload = _coerce_mapping(getattr(action, "payload", {}))
            # AUDIT-FIX(#7): Coerce legacy payload booleans/strings safely instead of treating 'false' as truthy.
            delivery = _coerce_choice(payload.get("delivery", "spoken"), allowed=_ALLOWED_DELIVERIES, default="spoken")
            search = _coerce_bool(payload.get("allow_web_search", False), default=False)
            label = "Speak generated content" if delivery == "spoken" else "Print generated content"
            if search:
                label += " with web search"
            if getattr(action, "text", None):
                label += f": {_optional_text(action.text, limit=120)}"
            parts.append(label)
            continue
        if action.kind == "say":
            parts.append(f"Speak: {_optional_text(getattr(action, 'text', ''), limit=120) or 'No text'}")
            continue
        if action.kind == "print":
            parts.append(f"Print: {_optional_text(getattr(action, 'text', ''), limit=120) or 'No text'}")
            continue
        tool_name = _optional_text(getattr(action, "tool_name", None), limit=80) or "tool"
        parts.append(f"Tool call: {tool_name}")
    return "; ".join(part for part in parts if part)


def _time_form_sections(
    entry: AutomationDefinition | None,
    *,
    timezone_name: str,
) -> tuple[AutomationFormSection, ...]:
    """Build the scheduled-automation form sections."""

    trigger = entry.trigger if entry is not None and isinstance(entry.trigger, TimeAutomationTrigger) else None
    action_defaults = _action_defaults(entry)
    return (
        AutomationFormSection(
            title="When it should run",
            description="Use one time, daily, or weekly schedules. Weekly accepts names like mon, wed, fri.",
            fields=(
                FileBackedSetting(
                    key="name",
                    label="Name",
                    value=entry.name if entry is not None else "",
                    help_text="Short operator-facing label.",
                    placeholder="Morning weather briefing",
                ),
                FileBackedSetting(
                    key="enabled",
                    label="Enabled",
                    value="true" if (entry.enabled if entry is not None else True) else "false",
                    input_type="select",
                    options=_BOOL_OPTIONS,
                    help_text="Disabled automations stay stored but will not fire.",
                ),
                FileBackedSetting(
                    key="description",
                    label="Description",
                    value=(entry.description or "") if entry is not None else "",
                    help_text="Optional short note for caregivers and operators.",
                    input_type="textarea",
                    rows=3,
                    wide=True,
                    placeholder="Tell Thom the weather each morning.",
                ),
                FileBackedSetting(
                    key="schedule",
                    label="Schedule",
                    value=trigger.schedule if trigger is not None else "once",
                    input_type="select",
                    options=_TIME_SCHEDULE_OPTIONS,
                    help_text="Choose one time, daily, or weekly.",
                ),
                FileBackedSetting(
                    key="due_at",
                    label="Due at",
                    value=str(trigger.due_at or "") if trigger is not None else "",
                    help_text="Only used for one-time schedules. Use ISO local time such as 2026-03-14T08:00+01:00.",
                    placeholder="2026-03-14T08:00+01:00",
                ),
                FileBackedSetting(
                    key="time_of_day",
                    label="Time of day",
                    value=str(trigger.time_of_day or "") if trigger is not None else "08:00",
                    help_text="Used for daily or weekly schedules.",
                    placeholder="08:00",
                ),
                FileBackedSetting(
                    key="weekdays_text",
                    label="Weekdays",
                    value=_format_weekdays(trigger.weekdays) if trigger is not None else "",
                    help_text="Only used for weekly schedules. Example: mon, wed, fri.",
                    placeholder="mon, wed, fri",
                ),
                FileBackedSetting(
                    key="timezone_name",
                    label="Timezone",
                    value=(trigger.timezone_name if trigger is not None else timezone_name) or timezone_name,
                    help_text="Leave this on the local timezone unless there is a clear reason not to.",
                    placeholder="Europe/Berlin",
                ),
                FileBackedSetting(
                    key="tags_text",
                    label="Tags",
                    value=", ".join(entry.tags) if entry is not None else "",
                    help_text="Optional comma-separated tags for later filtering.",
                    placeholder="daily, weather",
                ),
            ),
        ),
        AutomationFormSection(
            title="What Twinr should do",
            description="Use generated content for live jobs like weather, news, or headlines. Use exact text for fixed announcements.",
            fields=(
                FileBackedSetting(
                    key="delivery",
                    label="Delivery",
                    value=action_defaults["delivery"],
                    input_type="select",
                    options=_DELIVERY_OPTIONS,
                    help_text="Speak the result aloud or print it on paper.",
                ),
                FileBackedSetting(
                    key="content_mode",
                    label="Content mode",
                    value=action_defaults["content_mode"],
                    input_type="select",
                    options=_CONTENT_MODE_OPTIONS,
                    help_text="Generated content can use live web search when needed.",
                ),
                FileBackedSetting(
                    key="allow_web_search",
                    label="Allow web search",
                    value=action_defaults["allow_web_search"],
                    input_type="select",
                    options=_BOOL_OPTIONS,
                    help_text="Enable for current information such as weather, headlines, transport, or opening hours.",
                ),
                FileBackedSetting(
                    key="content",
                    label="Content or prompt",
                    value=action_defaults["content"],
                    help_text="For generated content, write the prompt Twinr should use later. For exact delivery, write the final text.",
                    input_type="textarea",
                    rows=5,
                    wide=True,
                    placeholder="Tell Thom the morning weather in Schwarzenbek in clear German.",
                ),
            ),
        ),
    )


def _sensor_form_sections(
    entry: AutomationDefinition | None,
    *,
    sensor_options: tuple[tuple[str, str], ...],
) -> tuple[AutomationFormSection, ...]:
    """Build the sensor-automation form sections."""

    sensor_trigger = _safe_describe_sensor_trigger(entry.trigger) if entry is not None else None
    action_defaults = _action_defaults(entry)
    return (
        AutomationFormSection(
            title="When it should react",
            description="Sensor automations only run while Twinr is idle. Use hold and cooldown to avoid chatter.",
            fields=(
                FileBackedSetting(
                    key="name",
                    label="Name",
                    value=entry.name if entry is not None else "",
                    help_text="Short operator-facing label.",
                    placeholder="Person near device greeting",
                ),
                FileBackedSetting(
                    key="enabled",
                    label="Enabled",
                    value="true" if (entry.enabled if entry is not None else True) else "false",
                    input_type="select",
                    options=_BOOL_OPTIONS,
                    help_text="Disabled automations stay stored but will not fire.",
                ),
                FileBackedSetting(
                    key="description",
                    label="Description",
                    value=(entry.description or "") if entry is not None else "",
                    help_text="Optional short note for caregivers and operators.",
                    input_type="textarea",
                    rows=3,
                    wide=True,
                    placeholder="Greet when someone is visible near Twinr.",
                ),
                FileBackedSetting(
                    key="trigger_kind",
                    label="Sensor trigger",
                    value=sensor_trigger.trigger_kind if sensor_trigger is not None else "pir_motion_detected",
                    input_type="select",
                    options=sensor_options,
                    help_text="These are stable high-level signals, not raw camera or microphone data.",
                ),
                FileBackedSetting(
                    key="hold_seconds",
                    label="Hold seconds",
                    value=_format_float(sensor_trigger.hold_seconds) if sensor_trigger is not None else "",
                    help_text="Required for quiet/no-motion triggers. Optional for camera presence triggers.",
                    placeholder="30",
                ),
                FileBackedSetting(
                    key="cooldown_seconds",
                    label="Cooldown seconds",
                    value=_format_float(sensor_trigger.cooldown_seconds) if sensor_trigger is not None else "60",
                    help_text="Minimum wait before the same automation can fire again.",
                    placeholder="60",
                ),
                FileBackedSetting(
                    key="tags_text",
                    label="Tags",
                    value=", ".join(entry.tags) if entry is not None else "",
                    help_text="Optional comma-separated tags for later filtering.",
                    placeholder="sensor, proactive",
                ),
            ),
        ),
        AutomationFormSection(
            title="What Twinr should do",
            description="Generated content works well for short spoken notices. Printed delivery is useful for bounded paper output.",
            fields=(
                FileBackedSetting(
                    key="delivery",
                    label="Delivery",
                    value=action_defaults["delivery"],
                    input_type="select",
                    options=_DELIVERY_OPTIONS,
                    help_text="Speak the result aloud or print it on paper.",
                ),
                FileBackedSetting(
                    key="content_mode",
                    label="Content mode",
                    value=action_defaults["content_mode"],
                    input_type="select",
                    options=_CONTENT_MODE_OPTIONS,
                    help_text="Generated content can use live web search when needed.",
                ),
                FileBackedSetting(
                    key="allow_web_search",
                    label="Allow web search",
                    value=action_defaults["allow_web_search"],
                    input_type="select",
                    options=_BOOL_OPTIONS,
                    help_text="Enable only when the automation really needs current external information.",
                ),
                FileBackedSetting(
                    key="content",
                    label="Content or prompt",
                    value=action_defaults["content"],
                    help_text="For generated content, write the prompt Twinr should use later. For exact delivery, write the final text.",
                    input_type="textarea",
                    rows=5,
                    wide=True,
                    placeholder="Say that Twinr is ready to help if someone is standing nearby.",
                ),
            ),
        ),
    )


def _action_defaults(entry: AutomationDefinition | None) -> dict[str, str]:
    """Extract edit-form defaults from an automation's primary action."""

    defaults = {
        "delivery": "spoken",
        "content_mode": "llm_prompt",
        "allow_web_search": "false",
        "content": "",
    }
    if entry is None:
        return defaults
    # AUDIT-FIX(#7): Guard against empty action lists and malformed legacy payloads when populating edit forms.
    action = _primary_action(entry)
    if action is None:
        return defaults
    if action.kind == "llm_prompt":
        payload = _coerce_mapping(getattr(action, "payload", {}))
        return {
            "delivery": _coerce_choice(payload.get("delivery", "spoken"), allowed=_ALLOWED_DELIVERIES, default="spoken"),
            "content_mode": "llm_prompt",
            "allow_web_search": "true" if _coerce_bool(payload.get("allow_web_search", False), default=False) else "false",
            "content": str(getattr(action, "text", "") or ""),
        }
    return {
        "delivery": "printed" if action.kind == "print" else "spoken",
        "content_mode": "static_text",
        "allow_web_search": "false",
        "content": str(getattr(action, "text", "") or ""),
    }


def _build_action(form: dict[str, str]) -> AutomationAction:
    """Build one automation action from submitted form data."""

    delivery = _parse_choice(form.get("delivery"), field_name="delivery", allowed=_ALLOWED_DELIVERIES, default="spoken")
    content_mode = _parse_choice(
        form.get("content_mode"),
        field_name="content mode",
        allowed=_ALLOWED_CONTENT_MODES,
        default="llm_prompt",
    )
    # AUDIT-FIX(#4): Preserve operator-entered line breaks and reject overlong action content instead of mutating it.
    content = _required_text(form.get("content"), field_name="content", limit=_MAX_CONTENT_LENGTH, multiline=True)
    allow_web_search = _parse_bool(form.get("allow_web_search"), default=False)
    if content_mode == "static_text":
        return AutomationAction(kind="print" if delivery == "printed" else "say", text=content)
    return AutomationAction(
        kind="llm_prompt",
        text=content,
        payload={
            "delivery": delivery,
            "allow_web_search": allow_web_search,
        },
    )


def _family_key_for_entry(
    entry: AutomationDefinition,
    *,
    integration_blocks: tuple[IntegrationAutomationFamilyBlock, ...],
) -> str:
    """Classify one automation into a page family."""

    source = str(getattr(entry, "source", "") or "")
    for block in integration_blocks:
        if any(prefix and source.startswith(prefix) for prefix in block.source_prefixes):
            return block.key
    if _is_sensor_entry(entry):
        return "sensor"
    if isinstance(entry.trigger, TimeAutomationTrigger):
        return "time"
    return "other"


def _definition_status_key(definition: AutomationFamilyDefinition, *, has_rows: bool) -> str:
    """Return the status key for one automation family card."""

    if definition.status_key:
        return definition.status_key
    return "ok" if has_rows else ("muted" if definition.create_supported else "warn")


def _definition_status_label(definition: AutomationFamilyDefinition, *, has_rows: bool) -> str:
    """Return the status label for one automation family card."""

    if definition.status_label:
        return definition.status_label
    if has_rows:
        return "Configured"
    if definition.create_supported:
        return "Ready"
    return "Reserved"


def _is_sensor_entry(entry: AutomationDefinition | None) -> bool:
    """Return whether an automation uses a supported sensor trigger."""

    if entry is None:
        return False
    if not isinstance(entry.trigger, IfThenAutomationTrigger):
        return False
    return _safe_describe_sensor_trigger(entry.trigger) is not None


def _find_entry(entries: tuple[AutomationDefinition, ...], edit_ref: str | None) -> AutomationDefinition | None:
    """Find the requested automation by id or unique name."""

    lookup = _normalize_inline_text(edit_ref)
    if not lookup:
        return None
    for entry in entries:
        if str(getattr(entry, "automation_id", "")) == lookup:
            return entry
    # AUDIT-FIX(#8): Only fall back to name lookup when the name is unique, so duplicate labels cannot target the wrong rule.
    name_matches = [
        entry
        for entry in entries
        if _normalize_inline_text(getattr(entry, "name", "")) == lookup
    ]
    if len(name_matches) == 1:
        return name_matches[0]
    return None


def _required_text(
    value: object,
    *,
    field_name: str,
    limit: int = 420,
    multiline: bool = False,
) -> str:
    """Normalize and require bounded non-empty text input."""

    # AUDIT-FIX(#4): Reject overlong persisted input instead of silently truncating it.
    text = _normalize_multiline_text(value) if multiline else _normalize_inline_text(value)
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    if len(text) > limit:
        raise ValueError(f"{field_name} must be {limit} characters or fewer")
    return text


def _optional_input_text(
    value: object,
    *,
    field_name: str,
    limit: int,
    multiline: bool = False,
) -> str | None:
    """Normalize optional bounded text input from the web form."""

    # AUDIT-FIX(#4): Preserve formatting for stored multiline text while validating size explicitly.
    text = _normalize_multiline_text(value) if multiline else _normalize_inline_text(value)
    if not text:
        return None
    if len(text) > limit:
        raise ValueError(f"{field_name} must be {limit} characters or fewer")
    return text


def _optional_text(value: object, *, limit: int) -> str:
    """Collapse and trim text for short operator summaries."""

    text = " ".join(str(value or "").split()).strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def _parse_bool(value: object, *, default: bool) -> bool:
    """Parse a strict boolean form field."""

    text = str(value or "").strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError("Boolean field must use true or false")


def _parse_nonnegative_float(value: object, *, default: float) -> float:
    """Parse a finite non-negative float form field."""

    text = str(value or "").strip()
    if not text:
        return default
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError("Numeric field must be a number") from exc
    # AUDIT-FIX(#6): Reject NaN/Inf so cooldown and hold timings stay schedulable and printable.
    if not math.isfinite(parsed):
        raise ValueError("Numeric field must be finite")
    if parsed < 0:
        raise ValueError("Numeric field must not be negative")
    return parsed


def _parse_tags(value: object) -> tuple[str, ...]:
    """Parse comma- or newline-separated tags into a stable tuple."""

    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").replace("\n", ",")
    items = [_normalize_inline_text(item) for item in text.split(",")]
    seen: set[str] = set()
    tags: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        tags.append(item)
    return tuple(tags)


def _parse_weekdays(value: object) -> tuple[int, ...]:
    """Parse weekday tokens into sorted weekday indexes."""

    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").replace("\n", ",").replace(";", ",")
    if not text.strip():
        return ()
    values: set[int] = set()
    for chunk in text.split(","):
        token = chunk.strip().lower()
        if not token:
            continue
        if token not in _WEEKDAY_ALIASES:
            raise ValueError("Weekdays must use names like mon, wed, fri or numbers 0-6")
        values.add(_WEEKDAY_ALIASES[token])
    return tuple(sorted(values))


def _format_weekdays(values: tuple[int, ...]) -> str:
    """Format stored weekday indexes for the edit form."""

    # AUDIT-FIX(#5): Ignore malformed weekday indexes instead of crashing the edit form.
    return ", ".join(_weekday_label(index, lower=True) for index in values)


def _format_float(value: float) -> str:
    """Format a stored numeric value for form display."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return ""
    # AUDIT-FIX(#5): Render malformed persisted numbers safely instead of raising during form population.
    if not math.isfinite(parsed):
        return ""
    if parsed.is_integer():
        return str(int(parsed))
    return f"{parsed:g}"


def _normalize_inline_text(value: object) -> str:
    """Collapse arbitrary input into single-line text."""

    return " ".join(str(value or "").split()).strip()


def _normalize_multiline_text(value: object) -> str:
    """Normalize newlines in multiline text input."""

    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _parse_choice(value: object, *, field_name: str, allowed: set[str], default: str) -> str:
    """Parse one allowed choice value with a default."""

    text = _normalize_inline_text(value).lower()
    if not text:
        return default
    if text not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise ValueError(f"{field_name} must be one of: {allowed_list}")
    return text


def _coerce_choice(value: object, *, allowed: set[str], default: str) -> str:
    """Coerce one choice-like value into an allowed set."""

    text = _normalize_inline_text(value).lower()
    if text in allowed:
        return text
    return default


# AUDIT-FIX(#1): Validate IANA timezone names before persisting them into schedules.
def _validated_timezone_name(
    value: object,
    *,
    field_name: str,
    fallback: str | None = None,
) -> str:
    """Validate and return an IANA timezone name."""

    text = _normalize_inline_text(value) or (fallback or "")
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    try:
        ZoneInfo(text)
    except ZoneInfoNotFoundError as exc:
        raise ValueError("Timezone must be a valid IANA timezone such as Europe/Berlin") from exc
    return text


def _coerce_timezone_name(value: object, *, fallback: str) -> str:
    """Return the first valid timezone name among the candidates."""

    text = _normalize_inline_text(value)
    for candidate in (text, fallback, _DEFAULT_TIMEZONE_NAME):
        if not candidate:
            continue
        try:
            ZoneInfo(candidate)
        except ZoneInfoNotFoundError:
            continue
        return candidate
    return _DEFAULT_TIMEZONE_NAME


# AUDIT-FIX(#1): Normalize schedule-specific fields so irrelevant stale values are not stored.
def _normalize_time_schedule(
    *,
    schedule: str,
    due_at: str | None,
    time_of_day: str | None,
    weekdays: tuple[int, ...],
    timezone_name: str,
) -> tuple[str, str, str, tuple[int, ...]]:
    """Normalize schedule-specific fields for time automations."""

    if schedule == "once":
        if not due_at:
            raise ValueError("Due at must not be empty for one-time schedules")
        return schedule, _validated_due_at(due_at, timezone_name=timezone_name), "", ()
    validated_time = _validated_time_of_day(time_of_day)
    if schedule == "daily":
        return schedule, "", validated_time, ()
    if not weekdays:
        raise ValueError("Weekly schedules must include at least one weekday")
    return schedule, "", validated_time, weekdays


# AUDIT-FIX(#1): Canonicalize one-time schedule timestamps through the shared parser.
def _validated_due_at(value: str, *, timezone_name: str) -> str:
    """Canonicalize one one-time due timestamp."""

    try:
        due_at = parse_due_at(value, timezone_name=timezone_name)
    except Exception as exc:
        raise ValueError("Due at must use an ISO date/time such as 2026-03-14T08:00+01:00") from exc
    if due_at is None or not hasattr(due_at, 'isoformat'):
        raise ValueError("Due at must use an ISO date/time such as 2026-03-14T08:00+01:00")
    return due_at.isoformat()


# AUDIT-FIX(#1): Reject invalid daily/weekly clock strings at save time instead of later in the scheduler.
def _validated_time_of_day(value: str | None) -> str:
    """Validate a daily or weekly clock value."""

    if not value:
        raise ValueError("Time of day must not be empty")
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            parsed = datetime.strptime(value, fmt).time()
        except ValueError:
            continue
        return parsed.strftime(fmt)
    raise ValueError("Time of day must use HH:MM or HH:MM:SS")


def _validated_time_of_day_or_fallback(value: object) -> str:
    """Return a valid clock string or a display fallback."""

    text = _normalize_inline_text(value)
    if not text:
        return "unknown time"
    try:
        return _validated_time_of_day(text)
    except ValueError:
        return text


# AUDIT-FIX(#5): Fall back to a safe timezone so broken config does not take down the page.
def _safe_now_in_timezone(timezone_name: str):
    """Return the current time in a safe timezone fallback chain."""

    safe_timezone = _coerce_timezone_name(timezone_name, fallback=_DEFAULT_TIMEZONE_NAME)
    try:
        return now_in_timezone(safe_timezone)
    except Exception:
        return now_in_timezone(_DEFAULT_TIMEZONE_NAME)


# AUDIT-FIX(#5): Format persisted datetimes defensively so malformed records degrade gracefully.
def _safe_due_label(value: object, *, timezone_name: str) -> str | None:
    """Format a due label without letting bad data crash the page."""

    if value is None:
        return None
    safe_timezone = _coerce_timezone_name(timezone_name, fallback=_DEFAULT_TIMEZONE_NAME)
    try:
        return format_due_label(value, timezone_name=safe_timezone)
    except Exception:
        return None


# AUDIT-FIX(#5): Prevent one bad automation from breaking next-run calculation for the whole list.
def _safe_next_run_at(store: AutomationStore, entry: AutomationDefinition, *, now: object):
    """Compute `next_run_at` without letting one entry fail the page."""

    try:
        return store.engine.next_run_at(entry, now=now)
    except Exception:
        return None


# AUDIT-FIX(#5): Contain trigger-summary failures to the affected automation row.
def _safe_trigger_summary(entry: AutomationDefinition, *, timezone_name: str) -> str:
    """Summarize one trigger with a guarded fallback."""

    try:
        return _trigger_summary(entry, timezone_name=timezone_name)
    except Exception:
        return "Trigger configuration error"


# AUDIT-FIX(#5): Contain action-summary failures to the affected automation row.
def _safe_action_summary(actions: tuple[AutomationAction, ...]) -> str:
    """Summarize actions with a guarded fallback."""

    try:
        return _action_summary(actions)
    except Exception:
        return "Action configuration error"


# AUDIT-FIX(#5): Guard sensor-trigger introspection against malformed stored trigger payloads.
def _safe_describe_sensor_trigger(trigger: object):
    """Describe a sensor trigger with a guarded fallback."""

    try:
        return describe_sensor_trigger(trigger)
    except Exception:
        return None


def _primary_action(entry: AutomationDefinition | None) -> AutomationAction | None:
    """Return the first enabled action, or the first stored action."""

    if entry is None:
        return None
    actions = tuple(getattr(entry, "actions", ()) or ())
    if not actions:
        return None
    for candidate in actions:
        if _coerce_bool(getattr(candidate, "enabled", True), default=True):
            return candidate
    return actions[0]


def _coerce_mapping(value: object) -> Mapping[str, Any]:
    """Return a mapping value or an empty mapping."""

    if isinstance(value, Mapping):
        return value
    return {}


# AUDIT-FIX(#7): Coerce legacy string payloads and malformed booleans without crashing or misreporting.
def _coerce_bool(value: object, *, default: bool) -> bool:
    """Coerce legacy or malformed bool-like payload values."""

    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return default
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return default
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return default


def _weekday_label(index: object, *, lower: bool) -> str:
    """Return one weekday label or a safe fallback label."""

    if isinstance(index, bool):
        value = int(index)
    elif isinstance(index, int):
        value = index
    else:
        try:
            value = int(index)
        except (TypeError, ValueError):
            value = -1
    if 0 <= value < len(_WEEKDAY_NAMES):
        label = _WEEKDAY_NAMES[value][:3]
    else:
        label = f"day {value}"
    return label.lower() if lower else label


def _sensor_trigger_options() -> tuple[tuple[str, str], ...]:
    """Return supported sensor trigger choices for the edit form."""

    options: list[tuple[str, str]] = []
    for trigger_kind in supported_sensor_trigger_kinds():
        try:
            label = SensorTriggerSpec(trigger_kind).label
        except Exception:
            label = trigger_kind.replace("_", " ").title()
        options.append((trigger_kind, label))
    return tuple(options)


# AUDIT-FIX(#9): Reduce UI-visible filesystem disclosure to a filename only.
def _store_path_label(store: AutomationStore) -> str:
    """Reduce the automation store path to a UI-safe filename."""

    path_text = str(getattr(store, "path", "") or "")
    if not path_text:
        return ""
    return PurePath(path_text).name or path_text


# AUDIT-FIX(#2): Treat only web-form-managed automations as editable through the generic operator forms.
def _can_operator_edit(entry: AutomationDefinition | None, *, expected_kind: str | None = None) -> bool:
    """Return whether the generic web UI may edit this automation."""

    if entry is None:
        return False
    source = str(getattr(entry, "source", "") or "").strip()
    if source not in {"", "web_ui"}:
        return False
    if expected_kind == "time":
        return isinstance(entry.trigger, TimeAutomationTrigger)
    if expected_kind == "sensor":
        return _is_sensor_entry(entry)
    return True


def _ensure_operator_editable(entry: AutomationDefinition, *, expected_kind: str | None = None) -> None:
    """Raise when an automation is not editable from the web UI."""

    if not _can_operator_edit(entry, expected_kind=expected_kind):
        if str(getattr(entry, "source", "") or "").strip() not in {"", "web_ui"}:
            raise ValueError("This automation is managed outside the web form and cannot be changed here.")
        if expected_kind == "time":
            raise ValueError("This form can only edit scheduled automations.")
        if expected_kind == "sensor":
            raise ValueError("This form can only edit sensor automations.")
        raise ValueError("This automation cannot be changed here.")
