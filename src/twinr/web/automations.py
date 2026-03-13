from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
from twinr.web.store import FileBackedSetting

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


@dataclass(frozen=True, slots=True)
class AutomationFamilyDefinition:
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
    key: str
    title: str
    count: int
    summary: str
    detail: str
    status_key: str
    status_label: str


@dataclass(frozen=True, slots=True)
class AutomationRow:
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
    definition: AutomationFamilyDefinition
    rows: tuple[AutomationRow, ...]


@dataclass(frozen=True, slots=True)
class AutomationFormSection:
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
    editing_time_entry = (
        edit_entry if edit_entry is not None and isinstance(edit_entry.trigger, TimeAutomationTrigger) else None
    )
    editing_sensor_entry = edit_entry if _is_sensor_entry(edit_entry) else None
    sensor_options = tuple(
        (
            trigger_kind,
            SensorTriggerSpec(trigger_kind).label.capitalize(),
        )
        for trigger_kind in supported_sensor_trigger_kinds()
    )
    return {
        "automation_store_path": str(store.path),
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
    automation_id = str(form.get("automation_id", "")).strip()
    name = _required_text(form.get("name"), field_name="name")
    description = _optional_text(form.get("description"), limit=220)
    schedule = str(form.get("schedule", "once") or "once").strip().lower()
    due_at = _optional_text(form.get("due_at"), limit=80)
    time_of_day = _optional_text(form.get("time_of_day"), limit=8)
    weekdays = _parse_weekdays(form.get("weekdays_text", ""))
    timezone_value = _optional_text(form.get("timezone_name"), limit=64) or timezone_name
    enabled = _parse_bool(form.get("enabled"), default=True)
    tags = _parse_tags(form.get("tags_text", ""))
    actions = (_build_action(form),)
    if automation_id:
        entry = store.get(automation_id)
        if entry is None:
            raise ValueError("Unknown automation.")
        trigger = TimeAutomationTrigger(
            schedule=schedule,
            due_at=due_at,
            time_of_day=time_of_day,
            weekdays=weekdays,
            timezone_name=timezone_value,
        )
        return store.update(
            automation_id,
            name=name,
            description=description,
            enabled=enabled,
            trigger=trigger,
            actions=actions,
            tags=tags,
        )
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
    automation_id = str(form.get("automation_id", "")).strip()
    name = _required_text(form.get("name"), field_name="name")
    description = _optional_text(form.get("description"), limit=220)
    trigger_kind = _required_text(form.get("trigger_kind"), field_name="trigger kind").lower()
    hold_seconds = _parse_nonnegative_float(form.get("hold_seconds"), default=0.0)
    cooldown_seconds = _parse_nonnegative_float(form.get("cooldown_seconds"), default=0.0)
    enabled = _parse_bool(form.get("enabled"), default=True)
    tags = _parse_tags(form.get("tags_text", ""))
    actions = (_build_action(form),)
    trigger = build_sensor_trigger(
        trigger_kind,
        hold_seconds=hold_seconds,
        cooldown_seconds=cooldown_seconds,
    )
    if automation_id:
        entry = store.get(automation_id)
        if entry is None:
            raise ValueError("Unknown automation.")
        return store.update(
            automation_id,
            name=name,
            description=description,
            enabled=enabled,
            trigger=trigger,
            actions=actions,
            tags=tags,
        )
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
    entry = store.get(automation_id)
    if entry is None:
        raise ValueError("Unknown automation.")
    return store.update(automation_id, enabled=not entry.enabled)


def delete_automation(store: AutomationStore, automation_id: str) -> AutomationDefinition:
    try:
        return store.delete(automation_id)
    except KeyError as exc:
        raise ValueError("Unknown automation.") from exc


def _family_definitions(
    *,
    integration_blocks: tuple[IntegrationAutomationFamilyBlock, ...] = (),
) -> tuple[AutomationFamilyDefinition, ...]:
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
    grouped: dict[str, list[AutomationRow]] = {}
    current_time = now_in_timezone(timezone_name)
    for entry in entries:
        family_key = _family_key_for_entry(entry, integration_blocks=integration_blocks)
        family_title = next(
            (
                definition.title
                for definition in _family_definitions(integration_blocks=integration_blocks)
                if definition.key == family_key
            ),
            family_key.replace("_", " ").title(),
        )
        next_run = store.engine.next_run_at(entry, now=current_time)
        next_run_label = format_due_label(next_run, timezone_name=timezone_name) if next_run is not None else None
        last_triggered_label = (
            format_due_label(entry.last_triggered_at, timezone_name=timezone_name)
            if entry.last_triggered_at is not None
            else None
        )
        status_key = "ok" if entry.enabled else "muted"
        status_label = "Enabled" if entry.enabled else "Disabled"
        grouped.setdefault(family_key, []).append(
            AutomationRow(
                automation_id=entry.automation_id,
                name=entry.name,
                description=entry.description,
                family_key=family_key,
                family_title=family_title,
                enabled=entry.enabled,
                status_key=status_key,
                status_label=status_label,
                trigger_summary=_trigger_summary(entry, timezone_name=timezone_name),
                action_summary=_action_summary(entry.actions),
                next_run_label=next_run_label,
                last_triggered_label=last_triggered_label,
                source=entry.source,
                tags=entry.tags,
            )
        )
    return {
        key: tuple(sorted(rows, key=lambda item: item.name.lower()))
        for key, rows in grouped.items()
    }


def _trigger_summary(entry: AutomationDefinition, *, timezone_name: str) -> str:
    trigger = entry.trigger
    sensor_text = describe_sensor_trigger_text(trigger)
    if sensor_text:
        return sensor_text[0].upper() + sensor_text[1:]
    if not isinstance(trigger, TimeAutomationTrigger):
        return f"{trigger.kind.replace('_', ' ').title()} trigger"
    if trigger.schedule == "once":
        due_at = parse_due_at(str(trigger.due_at), timezone_name=trigger.timezone_name or timezone_name)
        return f"One time at {format_due_label(due_at, timezone_name=trigger.timezone_name or timezone_name)}"
    if trigger.schedule == "daily":
        return f"Every day at {trigger.time_of_day}"
    weekday_labels = ", ".join(_WEEKDAY_NAMES[index][:3] for index in trigger.weekdays)
    return f"Every {weekday_labels} at {trigger.time_of_day}"


def _action_summary(actions: tuple[AutomationAction, ...]) -> str:
    enabled_actions = [action for action in actions if action.enabled]
    if not enabled_actions:
        return "No enabled action"
    parts: list[str] = []
    for action in enabled_actions:
        if action.kind == "llm_prompt":
            delivery = str(action.payload.get("delivery", "spoken")).strip().lower() or "spoken"
            search = bool(action.payload.get("allow_web_search", False))
            label = "Speak generated content" if delivery == "spoken" else "Print generated content"
            if search:
                label += " with web search"
            if action.text:
                label += f": {_optional_text(action.text, limit=120)}"
            parts.append(label)
            continue
        if action.kind == "say":
            parts.append(f"Speak: {_optional_text(action.text, limit=120)}")
            continue
        if action.kind == "print":
            parts.append(f"Print: {_optional_text(action.text, limit=120)}")
            continue
        tool_name = _optional_text(action.tool_name, limit=80) or "tool"
        parts.append(f"Tool call: {tool_name}")
    return "; ".join(part for part in parts if part)


def _time_form_sections(
    entry: AutomationDefinition | None,
    *,
    timezone_name: str,
) -> tuple[AutomationFormSection, ...]:
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
    sensor_trigger = describe_sensor_trigger(entry.trigger) if entry is not None else None
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
    if entry is None:
        return {
            "delivery": "spoken",
            "content_mode": "llm_prompt",
            "allow_web_search": "false",
            "content": "",
        }
    action = next((candidate for candidate in entry.actions if candidate.enabled), entry.actions[0])
    if action.kind == "llm_prompt":
        return {
            "delivery": str(action.payload.get("delivery", "spoken")).strip().lower() or "spoken",
            "content_mode": "llm_prompt",
            "allow_web_search": "true" if bool(action.payload.get("allow_web_search", False)) else "false",
            "content": action.text or "",
        }
    return {
        "delivery": "printed" if action.kind == "print" else "spoken",
        "content_mode": "static_text",
        "allow_web_search": "false",
        "content": action.text or "",
    }


def _build_action(form: dict[str, str]) -> AutomationAction:
    delivery = str(form.get("delivery", "spoken") or "spoken").strip().lower()
    content_mode = str(form.get("content_mode", "llm_prompt") or "llm_prompt").strip().lower()
    content = _required_text(form.get("content"), field_name="content")
    allow_web_search = _parse_bool(form.get("allow_web_search"), default=False)
    if delivery not in {"spoken", "printed"}:
        raise ValueError("delivery must be spoken or printed")
    if content_mode == "static_text":
        return AutomationAction(kind="print" if delivery == "printed" else "say", text=content)
    if content_mode != "llm_prompt":
        raise ValueError("content mode must be llm_prompt or static_text")
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
    for block in integration_blocks:
        if any(entry.source.startswith(prefix) for prefix in block.source_prefixes):
            return block.key
    if _is_sensor_entry(entry):
        return "sensor"
    if isinstance(entry.trigger, TimeAutomationTrigger):
        return "time"
    return "other"


def _definition_status_key(definition: AutomationFamilyDefinition, *, has_rows: bool) -> str:
    if definition.status_key:
        return definition.status_key
    return "ok" if has_rows else ("muted" if definition.create_supported else "warn")


def _definition_status_label(definition: AutomationFamilyDefinition, *, has_rows: bool) -> str:
    if definition.status_label:
        return definition.status_label
    if has_rows:
        return "Configured"
    if definition.create_supported:
        return "Ready"
    return "Reserved"


def _is_sensor_entry(entry: AutomationDefinition | None) -> bool:
    if entry is None:
        return False
    if not isinstance(entry.trigger, IfThenAutomationTrigger):
        return False
    return describe_sensor_trigger(entry.trigger) is not None


def _find_entry(entries: tuple[AutomationDefinition, ...], edit_ref: str | None) -> AutomationDefinition | None:
    lookup = str(edit_ref or "").strip()
    if not lookup:
        return None
    for entry in entries:
        if entry.automation_id == lookup or entry.name == lookup:
            return entry
    return None


def _required_text(value: object, *, field_name: str) -> str:
    text = _optional_text(value, limit=420)
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


def _optional_text(value: object, *, limit: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def _parse_bool(value: object, *, default: bool) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError("Boolean field must use true or false")


def _parse_nonnegative_float(value: object, *, default: float) -> float:
    text = str(value or "").strip()
    if not text:
        return default
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError("Numeric field must be a number") from exc
    if parsed < 0:
        raise ValueError("Numeric field must not be negative")
    return parsed


def _parse_tags(value: object) -> tuple[str, ...]:
    text = str(value or "").replace("\n", ",")
    items = [item.strip() for item in text.split(",")]
    return tuple(item for item in items if item)


def _parse_weekdays(value: object) -> tuple[int, ...]:
    text = str(value or "").replace(";", ",")
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
    return ", ".join(_WEEKDAY_NAMES[index][:3].lower() for index in values)


def _format_float(value: float) -> str:
    if int(value) == value:
        return str(int(value))
    return f"{value:g}"
