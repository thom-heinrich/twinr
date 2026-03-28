"""Build and describe canonical sensor-based automation triggers.

This module translates operator-facing sensor trigger kinds into
``IfThenAutomationTrigger`` objects and back again. The helpers here are the
single source of truth for supported sensor-trigger shapes that the web UI and
tooling may round-trip safely.
"""

from __future__ import annotations

# CHANGELOG: 2026-03-28
# BUG-1: fixed `smart_home_motion_cleared` so it no longer depends on the global
#        `smart_home.motion_detected` fact; the canonical trigger is now the
#        clear-event edge itself. This prevents false negatives in multi-sensor
#        homes where one sensor clears while another still reports motion.
# BUG-2: fixed lossy round-trips by rejecting trigger descriptions whenever
#        persisted `any_conditions` are present or malformed; those conditions
#        are part of real trigger semantics in Twinr's automation engine.
# SEC-1: reject unbounded hold/cooldown durations early. Extremely large finite
#        values can later overflow downstream `timedelta`/datetime arithmetic and
#        turn a malformed UI payload into a practical automation-engine DoS.
# IMP-1: replaced scattered trigger metadata and `if` chains with a typed
#        catalog so supported kinds, labels, events, and condition shapes stay in
#        one place and fail closed.
# IMP-2: upgraded canonical immediate triggers to event-first shapes. This aligns
#        with 2026 home-automation practice, where edge events are modeled
#        separately from sustained state/duration filters.
# IMP-3: improved text descriptions to include cooldowns and more human-readable
#        durations.
# BREAKING: immediate triggers now serialize canonically as `event_name` plus an
#           empty `all_conditions` tuple instead of duplicating redundant truthy
#           state guards. `describe_sensor_trigger()` remains backward-compatible
#           with legacy persisted shapes and will normalize them on the next save.
# BREAKING: hold/cooldown values above 366 days are now rejected instead of being
#           accepted and failing later in downstream runtime code.

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
import math

from twinr.automations.store import AutomationCondition, IfThenAutomationTrigger


_MAX_SENSOR_DURATION_SECONDS = 366.0 * 24.0 * 60.0 * 60.0
_MISSING = object()
_NO_VALUE_FINGERPRINT = "__NO_VALUE__"


class SensorTriggerKind(StrEnum):
    PIR_MOTION_DETECTED = "pir_motion_detected"
    PIR_NO_MOTION = "pir_no_motion"
    VAD_SPEECH_DETECTED = "vad_speech_detected"
    VAD_QUIET = "vad_quiet"
    CAMERA_PERSON_VISIBLE = "camera_person_visible"
    CAMERA_HAND_OR_OBJECT_NEAR_CAMERA = "camera_hand_or_object_near_camera"
    SMART_HOME_MOTION_DETECTED = "smart_home_motion_detected"
    SMART_HOME_MOTION_CLEARED = "smart_home_motion_cleared"
    SMART_HOME_BUTTON_PRESSED = "smart_home_button_pressed"
    SMART_HOME_DEVICE_OFFLINE = "smart_home_device_offline"
    SMART_HOME_ALARM_TRIGGERED = "smart_home_alarm_triggered"


@dataclass(frozen=True, slots=True)
class SensorTriggerSpec:
    """Describe a supported sensor trigger in a UI-friendly form.

    Attributes:
        trigger_kind: Canonical sensor trigger identifier.
        hold_seconds: Required duration threshold for sustained triggers.
        cooldown_seconds: Minimum quiet period after a match fires again.
    """

    trigger_kind: str
    hold_seconds: float = 0.0
    cooldown_seconds: float = 0.0

    @property
    def label(self) -> str:
        """Return the human-readable label for this trigger kind."""

        metadata = _TRIGGERS.get(_coerce_trigger_kind(self.trigger_kind))
        if metadata is None:
            return self.trigger_kind.replace("_", " ")
        return metadata.label


@dataclass(frozen=True, slots=True)
class _SensorTriggerMetadata:
    kind: SensorTriggerKind
    label: str
    event_name: str | None = None
    state_key: str | None = None
    state_operator: str | None = None
    duration_fact_key: str | None = None
    requires_positive_hold: bool = False
    legacy_event_conditions: tuple[AutomationCondition, ...] = ()

    @property
    def supports_immediate(self) -> bool:
        return self.event_name is not None

    @property
    def supports_hold(self) -> bool:
        return self.duration_fact_key is not None


def _truthy(key: str) -> AutomationCondition:
    return AutomationCondition(key=key, operator="truthy")


def _falsy(key: str) -> AutomationCondition:
    return AutomationCondition(key=key, operator="falsy")


def _gte(key: str, value: float) -> AutomationCondition:
    return AutomationCondition(key=key, operator="gte", value=value)


_TRIGGERS: dict[SensorTriggerKind, _SensorTriggerMetadata] = {
    SensorTriggerKind.PIR_MOTION_DETECTED: _SensorTriggerMetadata(
        kind=SensorTriggerKind.PIR_MOTION_DETECTED,
        label="motion is detected by the motion sensor",
        event_name="pir.motion_detected",
        state_key="pir.motion_detected",
        state_operator="truthy",
    ),
    SensorTriggerKind.PIR_NO_MOTION: _SensorTriggerMetadata(
        kind=SensorTriggerKind.PIR_NO_MOTION,
        label="no motion has been detected by the motion sensor",
        state_key="pir.motion_detected",
        state_operator="falsy",
        duration_fact_key="pir.no_motion_for_s",
        requires_positive_hold=True,
    ),
    SensorTriggerKind.VAD_SPEECH_DETECTED: _SensorTriggerMetadata(
        kind=SensorTriggerKind.VAD_SPEECH_DETECTED,
        label="speech is detected by the room microphone",
        event_name="vad.speech_detected",
        state_key="vad.speech_detected",
        state_operator="truthy",
    ),
    SensorTriggerKind.VAD_QUIET: _SensorTriggerMetadata(
        kind=SensorTriggerKind.VAD_QUIET,
        label="the room microphone has been quiet",
        state_key="vad.quiet",
        state_operator="truthy",
        duration_fact_key="vad.quiet_for_s",
        requires_positive_hold=True,
    ),
    SensorTriggerKind.CAMERA_PERSON_VISIBLE: _SensorTriggerMetadata(
        kind=SensorTriggerKind.CAMERA_PERSON_VISIBLE,
        label="a person is visible in the camera view",
        event_name="camera.person_visible",
        state_key="camera.person_visible",
        state_operator="truthy",
        duration_fact_key="camera.person_visible_for_s",
    ),
    SensorTriggerKind.CAMERA_HAND_OR_OBJECT_NEAR_CAMERA: _SensorTriggerMetadata(
        kind=SensorTriggerKind.CAMERA_HAND_OR_OBJECT_NEAR_CAMERA,
        label="a hand or object is near the camera",
        event_name="camera.hand_or_object_near_camera",
        state_key="camera.hand_or_object_near_camera",
        state_operator="truthy",
        duration_fact_key="camera.hand_or_object_near_camera_for_s",
    ),
    SensorTriggerKind.SMART_HOME_MOTION_DETECTED: _SensorTriggerMetadata(
        kind=SensorTriggerKind.SMART_HOME_MOTION_DETECTED,
        label="motion is detected by a smart-home sensor",
        event_name="smart_home.motion_detected",
        state_key="smart_home.motion_detected",
        state_operator="truthy",
    ),
    SensorTriggerKind.SMART_HOME_MOTION_CLEARED: _SensorTriggerMetadata(
        kind=SensorTriggerKind.SMART_HOME_MOTION_CLEARED,
        label="motion has cleared on a smart-home sensor",
        event_name="smart_home.motion_cleared",
        # Legacy persisted shape retained for backward description/migration only.
        legacy_event_conditions=(_falsy("smart_home.motion_detected"),),
    ),
    SensorTriggerKind.SMART_HOME_BUTTON_PRESSED: _SensorTriggerMetadata(
        kind=SensorTriggerKind.SMART_HOME_BUTTON_PRESSED,
        label="a smart-home button was pressed",
        event_name="smart_home.button_pressed",
        state_key="smart_home.button_pressed",
        state_operator="truthy",
    ),
    SensorTriggerKind.SMART_HOME_DEVICE_OFFLINE: _SensorTriggerMetadata(
        kind=SensorTriggerKind.SMART_HOME_DEVICE_OFFLINE,
        label="a smart-home device went offline",
        event_name="smart_home.device_offline",
        state_key="smart_home.device_offline",
        state_operator="truthy",
    ),
    SensorTriggerKind.SMART_HOME_ALARM_TRIGGERED: _SensorTriggerMetadata(
        kind=SensorTriggerKind.SMART_HOME_ALARM_TRIGGERED,
        label="a smart-home alarm was triggered",
        event_name="smart_home.alarm_triggered",
        state_key="smart_home.alarm_triggered",
        state_operator="truthy",
    ),
}

_SUPPORTED_SENSOR_TRIGGER_KINDS = tuple(kind.value for kind in SensorTriggerKind)


def supported_sensor_trigger_kinds() -> tuple[str, ...]:
    """Return the canonical sensor trigger kinds accepted by this package."""

    return _SUPPORTED_SENSOR_TRIGGER_KINDS


def build_sensor_trigger(
    trigger_kind: str,
    *,
    hold_seconds: float = 0.0,
    cooldown_seconds: float = 0.0,
) -> IfThenAutomationTrigger:
    """Build a canonical ``IfThenAutomationTrigger`` for a sensor trigger kind.

    Args:
        trigger_kind: Sensor trigger kind from the supported trigger catalog.
        hold_seconds: Required sustained duration for duration-based triggers.
        cooldown_seconds: Minimum delay before the trigger may match again.

    Returns:
        A normalized ``IfThenAutomationTrigger`` that matches Twinr runtime
        facts and events for the requested sensor kind.

    Raises:
        ValueError: If the trigger kind is unsupported or the durations are
            invalid for that kind.
    """

    normalized_kind = _require_trigger_kind(trigger_kind)
    metadata = _TRIGGERS[normalized_kind]

    normalized_hold = _parse_seconds(hold_seconds, field_name="hold_seconds")
    normalized_cooldown = _parse_seconds(cooldown_seconds, field_name="cooldown_seconds")

    if normalized_hold > 0.0 and not metadata.supports_hold:
        raise ValueError(f"{normalized_kind.value} does not support hold_seconds")

    if metadata.requires_positive_hold and normalized_hold <= 0.0:
        raise ValueError(f"{normalized_kind.value} requires hold_seconds greater than zero")

    if normalized_hold > 0.0:
        return IfThenAutomationTrigger(
            event_name=None,
            all_conditions=_duration_conditions(metadata, hold_seconds=normalized_hold),
            cooldown_seconds=normalized_cooldown,
        )

    if metadata.supports_immediate:
        return IfThenAutomationTrigger(
            event_name=metadata.event_name,
            all_conditions=(),
            cooldown_seconds=normalized_cooldown,
        )

    raise ValueError(f"{normalized_kind.value} requires hold_seconds greater than zero")


def describe_sensor_trigger(trigger: object) -> SensorTriggerSpec | None:
    """Recover a ``SensorTriggerSpec`` from a canonical if/then trigger.

    Returns ``None`` whenever the trigger cannot be mapped back to one of the
    supported sensor shapes without losing meaning.
    """

    if not isinstance(trigger, IfThenAutomationTrigger):
        return None

    all_conditions = _normalize_condition_sequence(getattr(trigger, "all_conditions", ()))
    if all_conditions is None:
        return None

    any_conditions = _normalize_condition_sequence(getattr(trigger, "any_conditions", ()))
    if any_conditions is None or any_conditions:
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

    for metadata in _TRIGGERS.values():
        if not metadata.supports_immediate:
            continue
        if event_name != _normalize_event_name(metadata.event_name):
            continue
        if _matches_immediate_shape(metadata, all_conditions):
            return SensorTriggerSpec(
                trigger_kind=metadata.kind.value,
                hold_seconds=0.0,
                cooldown_seconds=cooldown_seconds_value,
            )

    if event_name is not None:
        return None

    for metadata in _TRIGGERS.values():
        if not metadata.supports_hold:
            continue
        hold_seconds_value = _extract_duration_threshold(all_conditions, metadata.duration_fact_key)
        if hold_seconds_value is None:
            continue
        if _conditions_semantically_equal(
            all_conditions,
            _duration_conditions(metadata, hold_seconds=hold_seconds_value),
        ):
            return SensorTriggerSpec(
                trigger_kind=metadata.kind.value,
                hold_seconds=hold_seconds_value,
                cooldown_seconds=cooldown_seconds_value,
            )
    return None


def describe_sensor_trigger_text(trigger: object) -> str | None:
    """Render a supported sensor trigger as plain English text."""

    spec = describe_sensor_trigger(trigger)
    if spec is None:
        return None

    description = f"when {spec.label}"
    if spec.hold_seconds > 0.0:
        description = f"{description} for {_format_duration_text(spec.hold_seconds)}"
    if spec.cooldown_seconds > 0.0:
        description = f"{description} (at most once every {_format_duration_text(spec.cooldown_seconds)})"
    return description


def _coerce_trigger_kind(value: object) -> SensorTriggerKind | None:
    if isinstance(value, SensorTriggerKind):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if not text:
        return None
    try:
        return SensorTriggerKind(text)
    except ValueError:
        return None


def _require_trigger_kind(value: object) -> SensorTriggerKind:
    trigger_kind = _coerce_trigger_kind(value)
    if trigger_kind is None:
        raise ValueError(f"Unsupported sensor trigger kind: {value}")
    return trigger_kind


def _parse_seconds(
    value: object,
    *,
    field_name: str,
    default: float = 0.0,
) -> float:
    """Parse a non-negative finite duration value."""

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

    if seconds > _MAX_SENSOR_DURATION_SECONDS:
        raise ValueError(
            f"{field_name} must be less than or equal to {_format_duration_text(_MAX_SENSOR_DURATION_SECONDS)}"
        )

    return seconds


def _parse_optional_seconds(value: object, *, field_name: str) -> float | None:
    """Parse seconds for best-effort reads from possibly corrupted state."""

    try:
        return _parse_seconds(value, field_name=field_name)
    except ValueError:
        return None


def _base_conditions(metadata: _SensorTriggerMetadata) -> tuple[AutomationCondition, ...]:
    if metadata.state_key is None or metadata.state_operator is None:
        return ()
    return (_make_condition(metadata.state_key, metadata.state_operator),)


def _duration_conditions(
    metadata: _SensorTriggerMetadata,
    *,
    hold_seconds: float,
) -> tuple[AutomationCondition, ...]:
    if metadata.state_key is None or metadata.state_operator is None or metadata.duration_fact_key is None:
        raise ValueError(f"Unsupported sensor trigger kind: {metadata.kind.value}")

    return (
        _make_condition(metadata.state_key, metadata.state_operator),
        _make_condition(metadata.duration_fact_key, "gte", hold_seconds),
    )


def _matches_immediate_shape(
    metadata: _SensorTriggerMetadata,
    current_conditions: tuple[AutomationCondition, ...],
) -> bool:
    if not current_conditions:
        return True
    canonical_conditions = _base_conditions(metadata)
    if canonical_conditions and _conditions_semantically_equal(current_conditions, canonical_conditions):
        return True
    if metadata.legacy_event_conditions and _conditions_semantically_equal(
        current_conditions,
        metadata.legacy_event_conditions,
    ):
        return True
    return False


def _normalize_event_name(value: object) -> str | None | object:
    if value is None:
        return None
    if not isinstance(value, str):
        return _MISSING

    text = value.strip()
    if not text:
        return None

    parts: list[str] = []
    for character in text.lower():
        if character.isalnum():
            parts.append(character)
            continue
        if character in {".", "_"}:
            if parts and parts[-1] != character:
                parts.append(character)
            continue
        if parts and parts[-1] != "_":
            parts.append("_")

    normalized = "".join(parts).strip("._")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    while ".." in normalized:
        normalized = normalized.replace("..", ".")
    normalized = normalized.replace("._", ".").replace("_.", ".")
    return normalized or None


def _make_condition(key: str, operator: str, value: object = _MISSING) -> AutomationCondition:
    if value is _MISSING:
        return AutomationCondition(key=key, operator=operator)
    return AutomationCondition(key=key, operator=operator, value=value)


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


def _format_duration_text(seconds: float) -> str:
    if float(seconds).is_integer():
        whole_seconds = int(seconds)
        for unit_seconds, unit_name in (
            (24 * 60 * 60, "day"),
            (60 * 60, "hour"),
            (60, "minute"),
            (1, "second"),
        ):
            if whole_seconds >= unit_seconds and whole_seconds % unit_seconds == 0:
                amount = whole_seconds // unit_seconds
                suffix = "" if amount == 1 else "s"
                return f"{amount} {unit_name}{suffix}"
        unit = "second" if whole_seconds == 1 else "seconds"
        return f"{whole_seconds} {unit}"
    return f"{seconds:g} seconds"