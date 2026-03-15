from __future__ import annotations

from collections.abc import Iterable  # AUDIT-FIX(#3): validate persisted condition containers without assuming a tuple payload.
from dataclasses import dataclass
import math  # AUDIT-FIX(#1): reject NaN/inf and other non-finite numeric inputs deterministically.

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

# AUDIT-FIX(#6): use clearer user-facing wording and avoid unnecessary sensor jargon in generated descriptions.
_SENSOR_LABELS = {
    "pir_motion_detected": "motion is detected by the motion sensor",
    "pir_no_motion": "no motion has been detected by the motion sensor",
    "vad_speech_detected": "speech is detected by the room microphone",
    "vad_quiet": "the room microphone has been quiet",
    "camera_person_visible": "a person is visible in the camera view",
    "camera_hand_or_object_near_camera": "a hand or object is near the camera",
}

# AUDIT-FIX(#5): centralize trigger metadata so unknown kinds fail closed instead of silently falling through to the wrong fact key.
_DURATION_FACT_KEYS = {
    "pir_no_motion": "pir.no_motion_for_s",
    "vad_quiet": "vad.quiet_for_s",
    "camera_person_visible": "camera.person_visible_for_s",
    "camera_hand_or_object_near_camera": "camera.hand_or_object_near_camera_for_s",
}
_DURATION_TRIGGER_KINDS = (
    "pir_no_motion",
    "vad_quiet",
    "camera_person_visible",
    "camera_hand_or_object_near_camera",
)
_REQUIRED_POSITIVE_HOLD_KINDS = frozenset({"pir_no_motion", "vad_quiet"})
_MISSING = object()
_NO_VALUE_FINGERPRINT = "__NO_VALUE__"


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


# AUDIT-FIX(#1): parse numeric input strictly to avoid silent bool/negative coercion and raw float() exceptions leaking upstream.
def _parse_seconds(
    value: object,
    *,
    field_name: str,
    default: float = 0.0,
) -> float:
    if value is None:
        return default

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        raw_value: object = stripped
    else:
        raw_value = value

    if isinstance(raw_value, bool):
        raise ValueError(f"{field_name} must be a non-negative finite number")

    try:
        seconds = float(raw_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a non-negative finite number") from exc

    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError(f"{field_name} must be a non-negative finite number")

    return seconds


# AUDIT-FIX(#3): fail closed while describing corrupted persisted state instead of crashing on bad cooldown/threshold values.
def _parse_optional_seconds(value: object, *, field_name: str) -> float | None:
    try:
        return _parse_seconds(value, field_name=field_name)
    except ValueError:
        return None


def build_sensor_trigger(
    trigger_kind: str,
    *,
    hold_seconds: float = 0.0,
    cooldown_seconds: float = 0.0,
) -> IfThenAutomationTrigger:
    normalized_kind = str(trigger_kind or "").strip().lower()
    if normalized_kind not in _SUPPORTED_SENSOR_TRIGGER_KINDS:
        raise ValueError(f"Unsupported sensor trigger kind: {trigger_kind}")

    normalized_hold = _parse_seconds(hold_seconds, field_name="hold_seconds")  # AUDIT-FIX(#1)
    normalized_cooldown = _parse_seconds(cooldown_seconds, field_name="cooldown_seconds")  # AUDIT-FIX(#1)

    if normalized_kind in _REQUIRED_POSITIVE_HOLD_KINDS and normalized_hold <= 0.0:
        raise ValueError(f"{normalized_kind} requires hold_seconds greater than zero")

    if normalized_kind in _IMMEDIATE_SENSOR_EVENTS and normalized_hold <= 0.0:
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


# AUDIT-FIX(#2,#3,#4): only describe exact, lossless trigger shapes and degrade safely when persisted state is malformed.
def describe_sensor_trigger(trigger: object) -> SensorTriggerSpec | None:
    if not isinstance(trigger, IfThenAutomationTrigger):
        return None

    all_conditions = _normalize_condition_sequence(getattr(trigger, "all_conditions", ()))
    if all_conditions is None:
        return None

    cooldown_seconds_value = _parse_optional_seconds(
        getattr(trigger, "cooldown_seconds", 0.0),
        field_name="cooldown_seconds",
    )
    if cooldown_seconds_value is None:
        return None

    event_name = _normalize_event_name(getattr(trigger, "event_name", None))
    if event_name is _MISSING:
        return None

    for trigger_kind, immediate_event_name in _IMMEDIATE_SENSOR_EVENTS.items():
        normalized_immediate_event_name = _normalize_event_name(immediate_event_name)
        if event_name != normalized_immediate_event_name:
            continue
        if _conditions_semantically_equal(all_conditions, _base_conditions(trigger_kind)):
            return SensorTriggerSpec(
                trigger_kind=trigger_kind,
                hold_seconds=0.0,
                cooldown_seconds=cooldown_seconds_value,
            )

    for trigger_kind in _DURATION_TRIGGER_KINDS:
        if event_name is not None:
            continue
        duration_key = _duration_fact_key(trigger_kind)
        hold_seconds_value = _extract_duration_threshold(all_conditions, duration_key)
        if hold_seconds_value is None:
            continue
        if _conditions_semantically_equal(
            all_conditions,
            _duration_conditions(trigger_kind, hold_seconds=hold_seconds_value),
        ):
            return SensorTriggerSpec(
                trigger_kind=trigger_kind,
                hold_seconds=hold_seconds_value,
                cooldown_seconds=cooldown_seconds_value,
            )
    return None


def describe_sensor_trigger_text(trigger: object) -> str | None:
    spec = describe_sensor_trigger(trigger)
    if spec is None:
        return None
    if spec.hold_seconds > 0:
        return f"when {spec.label} for {_format_seconds_text(spec.hold_seconds)}"
    return f"when {spec.label}"


# AUDIT-FIX(#6): render hold durations in plain language instead of terse engineering shorthand like "5s".
def _format_seconds_text(seconds: float) -> str:
    if float(seconds).is_integer():
        whole_seconds = int(seconds)
        unit = "second" if whole_seconds == 1 else "seconds"
        return f"{whole_seconds} {unit}"
    return f"{seconds:g} seconds"


# AUDIT-FIX(#5): construct conditions through a helper so omitted values stay omitted and internal mappings remain explicit.
def _make_condition(key: str, operator: str, value: object = _MISSING) -> AutomationCondition:
    if value is _MISSING:
        return AutomationCondition(key=key, operator=operator)
    return AutomationCondition(key=key, operator=operator, value=value)


def _base_conditions(trigger_kind: str) -> tuple[AutomationCondition, ...]:
    if trigger_kind == "pir_motion_detected":
        return (_make_condition("pir.motion_detected", "truthy"),)
    if trigger_kind == "vad_speech_detected":
        return (_make_condition("vad.speech_detected", "truthy"),)
    if trigger_kind == "camera_person_visible":
        return (_make_condition("camera.person_visible", "truthy"),)
    if trigger_kind == "camera_hand_or_object_near_camera":
        return (_make_condition("camera.hand_or_object_near_camera", "truthy"),)
    raise ValueError(f"Unsupported sensor trigger kind: {trigger_kind}")  # AUDIT-FIX(#5)


def _duration_conditions(trigger_kind: str, *, hold_seconds: float) -> tuple[AutomationCondition, ...]:
    if trigger_kind == "pir_no_motion":
        return (
            _make_condition("pir.motion_detected", "falsy"),
            _make_condition("pir.no_motion_for_s", "gte", hold_seconds),
        )
    if trigger_kind == "vad_quiet":
        return (
            _make_condition("vad.quiet", "truthy"),
            _make_condition("vad.quiet_for_s", "gte", hold_seconds),
        )
    if trigger_kind == "camera_person_visible":
        return (
            _make_condition("camera.person_visible", "truthy"),
            _make_condition("camera.person_visible_for_s", "gte", hold_seconds),
        )
    if trigger_kind == "camera_hand_or_object_near_camera":
        return (
            _make_condition("camera.hand_or_object_near_camera", "truthy"),
            _make_condition(
                "camera.hand_or_object_near_camera_for_s",
                "gte",
                hold_seconds,
            ),
        )
    raise ValueError(f"Unsupported sensor trigger kind: {trigger_kind}")  # AUDIT-FIX(#5)


def _duration_fact_key(trigger_kind: str) -> str:
    try:
        return _DURATION_FACT_KEYS[trigger_kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported sensor trigger kind: {trigger_kind}") from exc  # AUDIT-FIX(#5)


# AUDIT-FIX(#3): tolerate tuple/list/set/generator payloads from persisted state, but reject anything that is not a real condition object.
def _normalize_condition_sequence(conditions: object) -> tuple[AutomationCondition, ...] | None:
    if conditions is None or isinstance(conditions, (str, bytes)):
        return None
    if not isinstance(conditions, Iterable):
        return None

    normalized_conditions: list[AutomationCondition] = []
    for condition in conditions:
        if not isinstance(condition, AutomationCondition):
            return None
        normalized_conditions.append(condition)
    return tuple(normalized_conditions)


# AUDIT-FIX(#3): accept blank persisted event names as "no event" for backward compatibility, but reject non-string payloads.
def _normalize_event_name(value: object) -> str | None | object:
    if value is None:
        return None
    if not isinstance(value, str):
        return _MISSING
    normalized_value = value.strip()
    if not normalized_value:
        return None
    return normalized_value


# AUDIT-FIX(#2): require an exact semantic condition match so extra guards are not silently dropped on round-trip edits.
def _conditions_semantically_equal(
    current_conditions: tuple[AutomationCondition, ...],
    expected_conditions: tuple[AutomationCondition, ...],
) -> bool:
    try:
        current_signatures = tuple(sorted(_condition_signature(condition) for condition in current_conditions))
        expected_signatures = tuple(sorted(_condition_signature(condition) for condition in expected_conditions))
    except (TypeError, ValueError):
        return False
    return current_signatures == expected_signatures


def _condition_signature(condition: AutomationCondition) -> tuple[str, str, str]:
    key = getattr(condition, "key", None)
    operator = getattr(condition, "operator", None)
    if not isinstance(key, str) or not isinstance(operator, str):
        raise TypeError("Invalid AutomationCondition payload")

    normalized_key = key.strip()
    normalized_operator = operator.strip().lower()
    if not normalized_key or not normalized_operator:
        raise ValueError("Invalid AutomationCondition payload")

    value = getattr(condition, "value", _MISSING)
    return (
        normalized_key,
        normalized_operator,
        _condition_value_fingerprint(normalized_operator, value),
    )


def _condition_value_fingerprint(operator: str, value: object) -> str:
    if value is _MISSING or value is None:
        return _NO_VALUE_FINGERPRINT

    if operator in {"truthy", "falsy"} and isinstance(value, str) and not value.strip():
        return _NO_VALUE_FINGERPRINT

    if operator in {"gt", "gte", "lt", "lte"} and not isinstance(value, bool):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError):
            pass
        else:
            if math.isfinite(numeric_value):
                return f"float:{numeric_value:.17g}"

    return f"{type(value).__name__}:{value!r}"


# AUDIT-FIX(#4): require exactly one finite, positive duration threshold; anything ambiguous or zero-like is treated as invalid state.
def _extract_duration_threshold(
    conditions: tuple[AutomationCondition, ...],
    duration_key: str,
) -> float | None:
    threshold: float | None = None
    for condition in conditions:
        if getattr(condition, "key", None) != duration_key:
            continue
        if getattr(condition, "operator", None) != "gte":
            continue

        candidate_threshold = _parse_optional_seconds(
            getattr(condition, "value", None),
            field_name=duration_key,
        )
        if candidate_threshold is None or candidate_threshold <= 0.0:
            return None
        if threshold is not None:
            return None
        threshold = candidate_threshold
    return threshold
