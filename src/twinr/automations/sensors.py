from __future__ import annotations

from dataclasses import dataclass

from twinr.automations.store import AutomationCondition, IfThenAutomationTrigger

_SUPPORTED_SENSOR_TRIGGER_KINDS = (
    "pir_motion_detected",
    "pir_no_motion",
    "vad_speech_detected",
    "vad_quiet",
    "camera_person_visible",
    "camera_hand_or_object_near_camera",
)

_IMMEDIATE_SENSOR_EVENTS = {
    "pir_motion_detected": "pir.motion_detected",
    "vad_speech_detected": "vad.speech_detected",
    "camera_person_visible": "camera.person_visible",
    "camera_hand_or_object_near_camera": "camera.hand_or_object_near_camera",
}

_SENSOR_LABELS = {
    "pir_motion_detected": "PIR motion is detected",
    "pir_no_motion": "there has been no PIR motion",
    "vad_speech_detected": "speech is detected by the background microphone",
    "vad_quiet": "the background microphone has been quiet",
    "camera_person_visible": "a person is visible in the camera view",
    "camera_hand_or_object_near_camera": "a hand or object is near the camera",
}


@dataclass(frozen=True, slots=True)
class SensorTriggerSpec:
    trigger_kind: str
    hold_seconds: float = 0.0
    cooldown_seconds: float = 0.0

    @property
    def label(self) -> str:
        return _SENSOR_LABELS.get(self.trigger_kind, self.trigger_kind.replace("_", " "))


def supported_sensor_trigger_kinds() -> tuple[str, ...]:
    return _SUPPORTED_SENSOR_TRIGGER_KINDS


def build_sensor_trigger(
    trigger_kind: str,
    *,
    hold_seconds: float = 0.0,
    cooldown_seconds: float = 0.0,
) -> IfThenAutomationTrigger:
    normalized_kind = str(trigger_kind or "").strip().lower()
    if normalized_kind not in _SUPPORTED_SENSOR_TRIGGER_KINDS:
        raise ValueError(f"Unsupported sensor trigger kind: {trigger_kind}")

    normalized_hold = max(0.0, float(hold_seconds or 0.0))
    normalized_cooldown = max(0.0, float(cooldown_seconds or 0.0))

    if normalized_kind in {"pir_no_motion", "vad_quiet"} and normalized_hold <= 0:
        raise ValueError(f"{normalized_kind} requires hold_seconds greater than zero")

    if normalized_kind in _IMMEDIATE_SENSOR_EVENTS and normalized_hold <= 0:
        return IfThenAutomationTrigger(
            event_name=_IMMEDIATE_SENSOR_EVENTS[normalized_kind],
            all_conditions=_base_conditions(normalized_kind),
            cooldown_seconds=normalized_cooldown,
        )

    return IfThenAutomationTrigger(
        event_name=None,
        all_conditions=_duration_conditions(normalized_kind, hold_seconds=normalized_hold),
        cooldown_seconds=normalized_cooldown,
    )


def describe_sensor_trigger(trigger: object) -> SensorTriggerSpec | None:
    if not isinstance(trigger, IfThenAutomationTrigger):
        return None

    for trigger_kind, event_name in _IMMEDIATE_SENSOR_EVENTS.items():
        if trigger.event_name != event_name:
            continue
        if _contains_all_conditions(trigger.all_conditions, _base_conditions(trigger_kind)):
            return SensorTriggerSpec(
                trigger_kind=trigger_kind,
                hold_seconds=0.0,
                cooldown_seconds=float(trigger.cooldown_seconds),
            )

    for trigger_kind in (
        "pir_no_motion",
        "vad_quiet",
        "camera_person_visible",
        "camera_hand_or_object_near_camera",
    ):
        duration_key = _duration_fact_key(trigger_kind)
        hold_seconds = _extract_duration_threshold(trigger.all_conditions, duration_key)
        if hold_seconds is None:
            continue
        if _contains_all_conditions(trigger.all_conditions, _duration_conditions(trigger_kind, hold_seconds=hold_seconds)):
            return SensorTriggerSpec(
                trigger_kind=trigger_kind,
                hold_seconds=hold_seconds,
                cooldown_seconds=float(trigger.cooldown_seconds),
            )
    return None


def describe_sensor_trigger_text(trigger: object) -> str | None:
    spec = describe_sensor_trigger(trigger)
    if spec is None:
        return None
    if spec.hold_seconds > 0:
        return f"when {spec.label} for {spec.hold_seconds:g}s"
    return f"when {spec.label}"


def _base_conditions(trigger_kind: str) -> tuple[AutomationCondition, ...]:
    if trigger_kind == "pir_motion_detected":
        return (AutomationCondition(key="pir.motion_detected", operator="truthy"),)
    if trigger_kind == "vad_speech_detected":
        return (AutomationCondition(key="vad.speech_detected", operator="truthy"),)
    if trigger_kind == "camera_person_visible":
        return (AutomationCondition(key="camera.person_visible", operator="truthy"),)
    if trigger_kind == "camera_hand_or_object_near_camera":
        return (AutomationCondition(key="camera.hand_or_object_near_camera", operator="truthy"),)
    return ()


def _duration_conditions(trigger_kind: str, *, hold_seconds: float) -> tuple[AutomationCondition, ...]:
    if trigger_kind == "pir_no_motion":
        return (
            AutomationCondition(key="pir.motion_detected", operator="falsy"),
            AutomationCondition(key="pir.no_motion_for_s", operator="gte", value=hold_seconds),
        )
    if trigger_kind == "vad_quiet":
        return (
            AutomationCondition(key="vad.quiet", operator="truthy"),
            AutomationCondition(key="vad.quiet_for_s", operator="gte", value=hold_seconds),
        )
    if trigger_kind == "camera_person_visible":
        return (
            AutomationCondition(key="camera.person_visible", operator="truthy"),
            AutomationCondition(key="camera.person_visible_for_s", operator="gte", value=hold_seconds),
        )
    if trigger_kind == "camera_hand_or_object_near_camera":
        return (
            AutomationCondition(key="camera.hand_or_object_near_camera", operator="truthy"),
            AutomationCondition(
                key="camera.hand_or_object_near_camera_for_s",
                operator="gte",
                value=hold_seconds,
            ),
        )
    return _base_conditions(trigger_kind)


def _duration_fact_key(trigger_kind: str) -> str:
    if trigger_kind == "pir_no_motion":
        return "pir.no_motion_for_s"
    if trigger_kind == "vad_quiet":
        return "vad.quiet_for_s"
    if trigger_kind == "camera_person_visible":
        return "camera.person_visible_for_s"
    return "camera.hand_or_object_near_camera_for_s"


def _contains_all_conditions(
    current_conditions: tuple[AutomationCondition, ...],
    expected_conditions: tuple[AutomationCondition, ...],
) -> bool:
    return all(condition in current_conditions for condition in expected_conditions)


def _extract_duration_threshold(
    conditions: tuple[AutomationCondition, ...],
    duration_key: str,
) -> float | None:
    for condition in conditions:
        if condition.key != duration_key or condition.operator not in {"gt", "gte"}:
            continue
        try:
            return max(0.0, float(condition.value or 0.0))
        except (TypeError, ValueError):
            return None
    return None
