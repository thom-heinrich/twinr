"""Configuration for the proactive camera stabilization surface."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

from ..gesture_calibration import GestureCalibrationProfile
from .coercion import coerce_optional_ratio
from .validation import (
    coerce_non_negative_float,
    coerce_positive_float,
    coerce_positive_int,
    require_bounded_float,
    require_bounded_ratio,
    require_non_negative_float,
    require_positive_int,
)

# CHANGELOG: 2026-03-29
# BUG-1: from_config no longer falls back to the slow proactive capture cadence for
#        center-follow refresh; the old fallback could inflate smoothing to 10s and
#        deadband to 25%, making person anchors visibly stale on Pi deployments.
# BUG-2: reject malformed gesture_calibration objects and non-finite numeric values
#        eagerly; the old class could defer failures and freeze automation with NaN/Inf.
# SEC-1: harden runtime-config ingestion against NaN/Inf config poisoning and bool/int
#        type confusion for counters, preventing practical denial-of-service-by-config.
# IMP-1: add a 2026-style speed-adaptive center-filter surface (one_euro / velocity /
#        legacy_ema) while preserving the legacy alpha/deadband/window compatibility API.
# IMP-2: broaden from_config to accept object or mapping sources and expose structured
#        filter export helpers for downstream MediaPipe / edge-runtime integration.

MIN_CENTER_SMOOTHING_ALPHA = 0.1
MAX_CENTER_SMOOTHING_ALPHA = 1.0
MIN_CENTER_DEADBAND = 0.0
MAX_CENTER_DEADBAND = 0.25
MIN_CENTER_SMOOTHING_WINDOW_S = 0.1
MAX_CENTER_SMOOTHING_WINDOW_S = 10.0

DEFAULT_DISPLAY_ATTENTION_REFRESH_INTERVAL_S = 0.8
MAX_DYNAMIC_CENTER_DEADBAND_DEFAULT = 0.08
MAX_DYNAMIC_CENTER_SMOOTHING_WINDOW_S_DEFAULT = 2.5

PRIMARY_PERSON_CENTER_FILTER_LEGACY_EMA = "legacy_ema"
PRIMARY_PERSON_CENTER_FILTER_ONE_EURO = "one_euro"
PRIMARY_PERSON_CENTER_FILTER_VELOCITY = "velocity"
_ALLOWED_PRIMARY_PERSON_CENTER_FILTERS = frozenset(
    {
        PRIMARY_PERSON_CENTER_FILTER_LEGACY_EMA,
        PRIMARY_PERSON_CENTER_FILTER_ONE_EURO,
        PRIMARY_PERSON_CENTER_FILTER_VELOCITY,
    }
)

MIN_PRIMARY_PERSON_CENTER_FILTER_FREQUENCY_HZ = 0.5
MAX_PRIMARY_PERSON_CENTER_FILTER_FREQUENCY_HZ = 60.0
MIN_PRIMARY_PERSON_CENTER_FILTER_MIN_CUTOFF_HZ = 0.01
MAX_PRIMARY_PERSON_CENTER_FILTER_MIN_CUTOFF_HZ = 10.0
MIN_PRIMARY_PERSON_CENTER_FILTER_BETA = 0.0
MAX_PRIMARY_PERSON_CENTER_FILTER_BETA = 100.0
MIN_PRIMARY_PERSON_CENTER_FILTER_DERIVATIVE_CUTOFF_HZ = 0.01
MAX_PRIMARY_PERSON_CENTER_FILTER_DERIVATIVE_CUTOFF_HZ = 10.0
MIN_PRIMARY_PERSON_CENTER_FILTER_WINDOW_SIZE = 1
MAX_PRIMARY_PERSON_CENTER_FILTER_WINDOW_SIZE = 120
MIN_PRIMARY_PERSON_CENTER_FILTER_VELOCITY_SCALE = 0.01
MAX_PRIMARY_PERSON_CENTER_FILTER_VELOCITY_SCALE = 1000.0

_MISSING = object()

_POSITIVE_INT_FIELDS = frozenset(
    {
        "person_visible_on_samples",
        "person_visible_off_samples",
        "looking_toward_device_on_samples",
        "looking_toward_device_off_samples",
        "person_near_device_on_samples",
        "person_near_device_off_samples",
        "engaged_with_device_on_samples",
        "engaged_with_device_off_samples",
        "showing_intent_on_samples",
        "showing_intent_off_samples",
        "hand_or_object_near_camera_on_samples",
        "hand_or_object_near_camera_off_samples",
        "fine_hand_explicit_confirm_samples",
        "object_on_samples",
        "object_off_samples",
        "primary_person_center_filter_window_size",
    }
)

_NON_NEGATIVE_FLOAT_FIELDS = frozenset(
    {
        "person_visible_unknown_hold_s",
        "person_visible_event_cooldown_s",
        "person_recently_visible_window_s",
        "person_returned_absence_s",
        "looking_toward_device_unknown_hold_s",
        "person_near_device_unknown_hold_s",
        "engaged_with_device_unknown_hold_s",
        "showing_intent_unknown_hold_s",
        "showing_intent_event_cooldown_s",
        "hand_or_object_near_camera_unknown_hold_s",
        "hand_or_object_near_camera_event_cooldown_s",
        "motion_event_cooldown_s",
        "gesture_event_cooldown_s",
        "fine_hand_explicit_hold_s",
        "object_unknown_hold_s",
        "secondary_unknown_hold_s",
    }
)

_BOUNDED_RATIO_FIELDS = {
    "fine_hand_explicit_min_confidence": (0.0, 1.0),
    "primary_person_center_smoothing_alpha": (
        MIN_CENTER_SMOOTHING_ALPHA,
        MAX_CENTER_SMOOTHING_ALPHA,
    ),
    "primary_person_center_deadband": (
        MIN_CENTER_DEADBAND,
        MAX_CENTER_DEADBAND,
    ),
    "primary_person_center_filter_min_allowed_object_scale": (0.0, 1.0),
}

_BOUNDED_FLOAT_FIELDS = {
    "primary_person_center_smoothing_window_s": (
        MIN_CENTER_SMOOTHING_WINDOW_S,
        MAX_CENTER_SMOOTHING_WINDOW_S,
    ),
    "primary_person_center_filter_frequency_hz": (
        MIN_PRIMARY_PERSON_CENTER_FILTER_FREQUENCY_HZ,
        MAX_PRIMARY_PERSON_CENTER_FILTER_FREQUENCY_HZ,
    ),
    "primary_person_center_filter_min_cutoff_hz": (
        MIN_PRIMARY_PERSON_CENTER_FILTER_MIN_CUTOFF_HZ,
        MAX_PRIMARY_PERSON_CENTER_FILTER_MIN_CUTOFF_HZ,
    ),
    "primary_person_center_filter_beta": (
        MIN_PRIMARY_PERSON_CENTER_FILTER_BETA,
        MAX_PRIMARY_PERSON_CENTER_FILTER_BETA,
    ),
    "primary_person_center_filter_derivative_cutoff_hz": (
        MIN_PRIMARY_PERSON_CENTER_FILTER_DERIVATIVE_CUTOFF_HZ,
        MAX_PRIMARY_PERSON_CENTER_FILTER_DERIVATIVE_CUTOFF_HZ,
    ),
    "primary_person_center_filter_velocity_scale": (
        MIN_PRIMARY_PERSON_CENTER_FILTER_VELOCITY_SCALE,
        MAX_PRIMARY_PERSON_CENTER_FILTER_VELOCITY_SCALE,
    ),
}

_BOOL_FIELDS = frozenset({"primary_person_center_filter_disable_value_scaling"})

_CONFIG_ALIASES = {
    "person_returned_absence_s": ("proactive_person_returned_absence_s",),
    "fine_hand_explicit_hold_s": ("proactive_local_camera_fine_hand_explicit_hold_s",),
    "fine_hand_explicit_confirm_samples": (
        "proactive_local_camera_fine_hand_explicit_confirm_samples",
    ),
    "fine_hand_explicit_min_confidence": (
        "proactive_local_camera_fine_hand_explicit_min_confidence",
    ),
    "primary_person_center_smoothing_alpha": (
        "proactive_local_camera_primary_person_center_smoothing_alpha",
    ),
    "primary_person_center_deadband": (
        "proactive_local_camera_primary_person_center_deadband",
    ),
    "primary_person_center_smoothing_window_s": (
        "proactive_local_camera_primary_person_center_smoothing_window_s",
    ),
    "primary_person_center_filter": (
        "proactive_local_camera_primary_person_center_filter",
    ),
    "primary_person_center_filter_frequency_hz": (
        "proactive_local_camera_primary_person_center_filter_frequency_hz",
    ),
    "primary_person_center_filter_min_cutoff_hz": (
        "proactive_local_camera_primary_person_center_filter_min_cutoff_hz",
    ),
    "primary_person_center_filter_beta": (
        "proactive_local_camera_primary_person_center_filter_beta",
    ),
    "primary_person_center_filter_derivative_cutoff_hz": (
        "proactive_local_camera_primary_person_center_filter_derivative_cutoff_hz",
    ),
    "primary_person_center_filter_window_size": (
        "proactive_local_camera_primary_person_center_filter_window_size",
    ),
    "primary_person_center_filter_velocity_scale": (
        "proactive_local_camera_primary_person_center_filter_velocity_scale",
    ),
    "primary_person_center_filter_min_allowed_object_scale": (
        "proactive_local_camera_primary_person_center_filter_min_allowed_object_scale",
    ),
    "primary_person_center_filter_disable_value_scaling": (
        "proactive_local_camera_primary_person_center_filter_disable_value_scaling",
    ),
}


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _require_plain_positive_int(value: object, *, field_name: str) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an int, got bool.")
    require_positive_int(value, field_name=field_name)


def _require_bounded_positive_int(
    value: object,
    *,
    field_name: str,
    minimum: int = 1,
    maximum: int | None = None,
) -> None:
    _require_plain_positive_int(value, field_name=field_name)
    int_value = int(value)
    if int_value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}.")
    if maximum is not None and int_value > maximum:
        raise ValueError(f"{field_name} must be <= {maximum}.")


def _require_finite_non_negative_float(value: object, *, field_name: str) -> None:
    if not _is_finite_number(value):
        raise ValueError(f"{field_name} must be a finite number.")
    require_non_negative_float(float(value), field_name=field_name)


def _require_finite_bounded_float(
    value: object,
    *,
    field_name: str,
    minimum: float,
    maximum: float,
) -> None:
    if not _is_finite_number(value):
        raise ValueError(f"{field_name} must be a finite number.")
    require_bounded_float(float(value), field_name=field_name, minimum=minimum, maximum=maximum)


def _require_finite_bounded_ratio(
    value: object,
    *,
    field_name: str,
    minimum: float,
    maximum: float,
) -> None:
    if not _is_finite_number(value):
        raise ValueError(f"{field_name} must be a finite number.")
    require_bounded_ratio(float(value), field_name=field_name, minimum=minimum, maximum=maximum)


def _read_config_value(config: object, *names: str, default: object = _MISSING) -> object:
    if config is None:
        return default
    for name in names:
        if isinstance(config, Mapping) and name in config:
            return config[name]
        value = getattr(config, name, _MISSING)
        if value is not _MISSING:
            return value
    return default


def _coerce_runtime_positive_float(value: object, *, default: float) -> float:
    if isinstance(value, bool):
        return default
    coerced = coerce_positive_float(value, default=default)
    if not _is_finite_number(coerced):
        return default
    coerced_float = float(coerced)
    return coerced_float if coerced_float > 0.0 else default


def _coerce_runtime_non_negative_float(value: object, *, default: float) -> float:
    if isinstance(value, bool):
        return default
    coerced = coerce_non_negative_float(value, default=default)
    if not _is_finite_number(coerced):
        return default
    coerced_float = float(coerced)
    return coerced_float if coerced_float >= 0.0 else default


def _coerce_runtime_positive_int(value: object, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    coerced = coerce_positive_int(value, default=default)
    if isinstance(coerced, bool):
        return default
    try:
        coerced_int = int(coerced)
    except (TypeError, ValueError):
        return default
    return coerced_int if coerced_int > 0 else default


def _coerce_runtime_bounded_float(
    value: object,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool):
        return default
    try:
        coerced_float = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(coerced_float):
        return default
    return min(maximum, max(minimum, coerced_float))


def _coerce_runtime_bounded_ratio(
    value: object,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    ratio = coerce_optional_ratio(value)
    if ratio is None or not _is_finite_number(ratio):
        return default
    return min(maximum, max(minimum, float(ratio)))


def _coerce_runtime_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _normalize_filter_kind(value: object, *, default: str) -> str:
    if not isinstance(value, str):
        return default
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized if normalized in _ALLOWED_PRIMARY_PERSON_CENTER_FILTERS else default


def _alpha_from_cutoff(*, frequency_hz: float, cutoff_hz: float) -> float:
    safe_frequency_hz = max(MIN_PRIMARY_PERSON_CENTER_FILTER_FREQUENCY_HZ, float(frequency_hz))
    safe_cutoff_hz = max(MIN_PRIMARY_PERSON_CENTER_FILTER_MIN_CUTOFF_HZ, float(cutoff_hz))
    sample_period_s = 1.0 / safe_frequency_hz
    tau = 1.0 / (2.0 * math.pi * safe_cutoff_hz)
    alpha = 1.0 / (1.0 + (tau / sample_period_s))
    return min(MAX_CENTER_SMOOTHING_ALPHA, max(MIN_CENTER_SMOOTHING_ALPHA, alpha))


def _config_names_for(field_name: str) -> tuple[str, ...]:
    return (
        field_name,
        f"proactive_camera_surface_{field_name}",
        f"proactive_local_camera_{field_name}",
        *_CONFIG_ALIASES.get(field_name, ()),
    )


def _apply_runtime_overrides(config: object, kwargs: dict[str, object]) -> dict[str, object]:
    if config is None:
        return kwargs

    for field_name in _POSITIVE_INT_FIELDS:
        raw = _read_config_value(config, *_config_names_for(field_name), default=_MISSING)
        if raw is not _MISSING:
            kwargs[field_name] = _coerce_runtime_positive_int(raw, default=int(kwargs[field_name]))

    for field_name in _NON_NEGATIVE_FLOAT_FIELDS:
        raw = _read_config_value(config, *_config_names_for(field_name), default=_MISSING)
        if raw is not _MISSING:
            kwargs[field_name] = _coerce_runtime_non_negative_float(raw, default=float(kwargs[field_name]))

    for field_name, (minimum, maximum) in _BOUNDED_RATIO_FIELDS.items():
        raw = _read_config_value(config, *_config_names_for(field_name), default=_MISSING)
        if raw is not _MISSING:
            kwargs[field_name] = _coerce_runtime_bounded_ratio(
                raw,
                default=float(kwargs[field_name]),
                minimum=minimum,
                maximum=maximum,
            )

    for field_name, (minimum, maximum) in _BOUNDED_FLOAT_FIELDS.items():
        raw = _read_config_value(config, *_config_names_for(field_name), default=_MISSING)
        if raw is not _MISSING:
            kwargs[field_name] = _coerce_runtime_bounded_float(
                raw,
                default=float(kwargs[field_name]),
                minimum=minimum,
                maximum=maximum,
            )

    for field_name in _BOOL_FIELDS:
        raw = _read_config_value(config, *_config_names_for(field_name), default=_MISSING)
        if raw is not _MISSING:
            kwargs[field_name] = _coerce_runtime_bool(raw, default=bool(kwargs[field_name]))

    raw_filter_kind = _read_config_value(
        config,
        *_config_names_for("primary_person_center_filter"),
        default=_MISSING,
    )
    if raw_filter_kind is not _MISSING:
        kwargs["primary_person_center_filter"] = _normalize_filter_kind(
            raw_filter_kind,
            default=str(kwargs["primary_person_center_filter"]),
        )

    raw_calibration = _read_config_value(config, "gesture_calibration", default=_MISSING)
    if raw_calibration is not _MISSING and isinstance(raw_calibration, GestureCalibrationProfile):
        kwargs["gesture_calibration"] = raw_calibration

    return kwargs


@dataclass(frozen=True, slots=True)
class ProactiveCameraSurfaceConfig:
    """Store stabilization rules for camera-derived automation signals."""

    person_visible_on_samples: int = 1
    person_visible_off_samples: int = 2
    person_visible_unknown_hold_s: float = 9.0
    person_visible_event_cooldown_s: float = 9.0
    person_recently_visible_window_s: float = 30.0
    person_returned_absence_s: float = 20.0 * 60.0
    looking_toward_device_on_samples: int = 1
    looking_toward_device_off_samples: int = 2
    looking_toward_device_unknown_hold_s: float = 9.0
    person_near_device_on_samples: int = 1
    person_near_device_off_samples: int = 2
    person_near_device_unknown_hold_s: float = 9.0
    engaged_with_device_on_samples: int = 1
    engaged_with_device_off_samples: int = 2
    engaged_with_device_unknown_hold_s: float = 9.0
    showing_intent_on_samples: int = 1
    showing_intent_off_samples: int = 2
    showing_intent_unknown_hold_s: float = 9.0
    showing_intent_event_cooldown_s: float = 9.0
    hand_or_object_near_camera_on_samples: int = 1
    hand_or_object_near_camera_off_samples: int = 2
    hand_or_object_near_camera_unknown_hold_s: float = 9.0
    hand_or_object_near_camera_event_cooldown_s: float = 9.0
    motion_event_cooldown_s: float = 9.0
    gesture_event_cooldown_s: float = 9.0
    fine_hand_explicit_hold_s: float = 0.45
    fine_hand_explicit_confirm_samples: int = 1
    fine_hand_explicit_min_confidence: float = 0.72
    gesture_calibration: GestureCalibrationProfile = field(default_factory=GestureCalibrationProfile.defaults)
    primary_person_center_smoothing_alpha: float = 0.58
    primary_person_center_deadband: float = 0.028
    primary_person_center_smoothing_window_s: float = 1.4
    object_on_samples: int = 2
    object_off_samples: int = 2
    object_unknown_hold_s: float = 9.0
    secondary_unknown_hold_s: float = 9.0
    primary_person_center_filter: str = PRIMARY_PERSON_CENTER_FILTER_LEGACY_EMA
    primary_person_center_filter_frequency_hz: float = 1.25
    primary_person_center_filter_min_cutoff_hz: float = 0.625
    primary_person_center_filter_beta: float = 0.375
    primary_person_center_filter_derivative_cutoff_hz: float = 1.0
    primary_person_center_filter_window_size: int = 3
    primary_person_center_filter_velocity_scale: float = 10.0
    primary_person_center_filter_min_allowed_object_scale: float = 0.01
    primary_person_center_filter_disable_value_scaling: bool = False

    def __post_init__(self) -> None:
        """Reject malformed debounce, cooldown, and filter configuration eagerly."""

        _require_plain_positive_int(self.person_visible_on_samples, field_name="person_visible_on_samples")
        _require_plain_positive_int(self.person_visible_off_samples, field_name="person_visible_off_samples")
        _require_finite_non_negative_float(
            self.person_visible_unknown_hold_s,
            field_name="person_visible_unknown_hold_s",
        )
        _require_finite_non_negative_float(
            self.person_visible_event_cooldown_s,
            field_name="person_visible_event_cooldown_s",
        )
        _require_finite_non_negative_float(
            self.person_recently_visible_window_s,
            field_name="person_recently_visible_window_s",
        )
        _require_finite_non_negative_float(
            self.person_returned_absence_s,
            field_name="person_returned_absence_s",
        )
        _require_plain_positive_int(
            self.looking_toward_device_on_samples,
            field_name="looking_toward_device_on_samples",
        )
        _require_plain_positive_int(
            self.looking_toward_device_off_samples,
            field_name="looking_toward_device_off_samples",
        )
        _require_finite_non_negative_float(
            self.looking_toward_device_unknown_hold_s,
            field_name="looking_toward_device_unknown_hold_s",
        )
        _require_plain_positive_int(
            self.person_near_device_on_samples,
            field_name="person_near_device_on_samples",
        )
        _require_plain_positive_int(
            self.person_near_device_off_samples,
            field_name="person_near_device_off_samples",
        )
        _require_finite_non_negative_float(
            self.person_near_device_unknown_hold_s,
            field_name="person_near_device_unknown_hold_s",
        )
        _require_plain_positive_int(
            self.engaged_with_device_on_samples,
            field_name="engaged_with_device_on_samples",
        )
        _require_plain_positive_int(
            self.engaged_with_device_off_samples,
            field_name="engaged_with_device_off_samples",
        )
        _require_finite_non_negative_float(
            self.engaged_with_device_unknown_hold_s,
            field_name="engaged_with_device_unknown_hold_s",
        )
        _require_plain_positive_int(
            self.showing_intent_on_samples,
            field_name="showing_intent_on_samples",
        )
        _require_plain_positive_int(
            self.showing_intent_off_samples,
            field_name="showing_intent_off_samples",
        )
        _require_finite_non_negative_float(
            self.showing_intent_unknown_hold_s,
            field_name="showing_intent_unknown_hold_s",
        )
        _require_finite_non_negative_float(
            self.showing_intent_event_cooldown_s,
            field_name="showing_intent_event_cooldown_s",
        )
        _require_plain_positive_int(
            self.hand_or_object_near_camera_on_samples,
            field_name="hand_or_object_near_camera_on_samples",
        )
        _require_plain_positive_int(
            self.hand_or_object_near_camera_off_samples,
            field_name="hand_or_object_near_camera_off_samples",
        )
        _require_finite_non_negative_float(
            self.hand_or_object_near_camera_unknown_hold_s,
            field_name="hand_or_object_near_camera_unknown_hold_s",
        )
        _require_finite_non_negative_float(
            self.hand_or_object_near_camera_event_cooldown_s,
            field_name="hand_or_object_near_camera_event_cooldown_s",
        )
        _require_finite_non_negative_float(
            self.motion_event_cooldown_s,
            field_name="motion_event_cooldown_s",
        )
        _require_finite_non_negative_float(
            self.gesture_event_cooldown_s,
            field_name="gesture_event_cooldown_s",
        )
        _require_finite_non_negative_float(
            self.fine_hand_explicit_hold_s,
            field_name="fine_hand_explicit_hold_s",
        )
        _require_plain_positive_int(
            self.fine_hand_explicit_confirm_samples,
            field_name="fine_hand_explicit_confirm_samples",
        )
        _require_finite_bounded_ratio(
            self.fine_hand_explicit_min_confidence,
            field_name="fine_hand_explicit_min_confidence",
            minimum=0.0,
            maximum=1.0,
        )
        if not isinstance(self.gesture_calibration, GestureCalibrationProfile):
            raise TypeError(
                "gesture_calibration must be a GestureCalibrationProfile, "
                f"got {type(self.gesture_calibration).__name__}."
            )
        _require_finite_bounded_ratio(
            self.primary_person_center_smoothing_alpha,
            field_name="primary_person_center_smoothing_alpha",
            minimum=MIN_CENTER_SMOOTHING_ALPHA,
            maximum=MAX_CENTER_SMOOTHING_ALPHA,
        )
        _require_finite_bounded_ratio(
            self.primary_person_center_deadband,
            field_name="primary_person_center_deadband",
            minimum=MIN_CENTER_DEADBAND,
            maximum=MAX_CENTER_DEADBAND,
        )
        _require_finite_bounded_float(
            self.primary_person_center_smoothing_window_s,
            field_name="primary_person_center_smoothing_window_s",
            minimum=MIN_CENTER_SMOOTHING_WINDOW_S,
            maximum=MAX_CENTER_SMOOTHING_WINDOW_S,
        )
        _require_plain_positive_int(self.object_on_samples, field_name="object_on_samples")
        _require_plain_positive_int(self.object_off_samples, field_name="object_off_samples")
        _require_finite_non_negative_float(self.object_unknown_hold_s, field_name="object_unknown_hold_s")
        _require_finite_non_negative_float(self.secondary_unknown_hold_s, field_name="secondary_unknown_hold_s")

        normalized_filter_kind = _normalize_filter_kind(
            self.primary_person_center_filter,
            default=self.primary_person_center_filter,
        )
        if normalized_filter_kind not in _ALLOWED_PRIMARY_PERSON_CENTER_FILTERS:
            allowed = ", ".join(sorted(_ALLOWED_PRIMARY_PERSON_CENTER_FILTERS))
            raise ValueError(
                "primary_person_center_filter must be one of "
                f"{allowed!s}; got {self.primary_person_center_filter!r}."
            )

        _require_finite_bounded_float(
            self.primary_person_center_filter_frequency_hz,
            field_name="primary_person_center_filter_frequency_hz",
            minimum=MIN_PRIMARY_PERSON_CENTER_FILTER_FREQUENCY_HZ,
            maximum=MAX_PRIMARY_PERSON_CENTER_FILTER_FREQUENCY_HZ,
        )
        _require_finite_bounded_float(
            self.primary_person_center_filter_min_cutoff_hz,
            field_name="primary_person_center_filter_min_cutoff_hz",
            minimum=MIN_PRIMARY_PERSON_CENTER_FILTER_MIN_CUTOFF_HZ,
            maximum=MAX_PRIMARY_PERSON_CENTER_FILTER_MIN_CUTOFF_HZ,
        )
        _require_finite_bounded_float(
            self.primary_person_center_filter_beta,
            field_name="primary_person_center_filter_beta",
            minimum=MIN_PRIMARY_PERSON_CENTER_FILTER_BETA,
            maximum=MAX_PRIMARY_PERSON_CENTER_FILTER_BETA,
        )
        _require_finite_bounded_float(
            self.primary_person_center_filter_derivative_cutoff_hz,
            field_name="primary_person_center_filter_derivative_cutoff_hz",
            minimum=MIN_PRIMARY_PERSON_CENTER_FILTER_DERIVATIVE_CUTOFF_HZ,
            maximum=MAX_PRIMARY_PERSON_CENTER_FILTER_DERIVATIVE_CUTOFF_HZ,
        )
        _require_bounded_positive_int(
            self.primary_person_center_filter_window_size,
            field_name="primary_person_center_filter_window_size",
            minimum=MIN_PRIMARY_PERSON_CENTER_FILTER_WINDOW_SIZE,
            maximum=MAX_PRIMARY_PERSON_CENTER_FILTER_WINDOW_SIZE,
        )
        _require_finite_bounded_float(
            self.primary_person_center_filter_velocity_scale,
            field_name="primary_person_center_filter_velocity_scale",
            minimum=MIN_PRIMARY_PERSON_CENTER_FILTER_VELOCITY_SCALE,
            maximum=MAX_PRIMARY_PERSON_CENTER_FILTER_VELOCITY_SCALE,
        )
        _require_finite_bounded_ratio(
            self.primary_person_center_filter_min_allowed_object_scale,
            field_name="primary_person_center_filter_min_allowed_object_scale",
            minimum=0.0,
            maximum=1.0,
        )
        if not isinstance(self.primary_person_center_filter_disable_value_scaling, bool):
            raise TypeError(
                "primary_person_center_filter_disable_value_scaling must be a bool."
            )

    def resolved_primary_person_center_smoothing_alpha(
        self,
        *,
        frequency_hz: float | None = None,
    ) -> float:
        """Return the effective alpha for legacy consumers.

        Legacy EMA consumers keep using ``primary_person_center_smoothing_alpha``.
        Newer consumers can select the speed-adaptive filter described by
        ``primary_person_center_filter`` and still request a compatible alpha.
        """

        if self.primary_person_center_filter == PRIMARY_PERSON_CENTER_FILTER_LEGACY_EMA:
            return float(self.primary_person_center_smoothing_alpha)

        effective_frequency_hz = (
            self.primary_person_center_filter_frequency_hz
            if frequency_hz is None
            else _coerce_runtime_bounded_float(
                frequency_hz,
                default=self.primary_person_center_filter_frequency_hz,
                minimum=MIN_PRIMARY_PERSON_CENTER_FILTER_FREQUENCY_HZ,
                maximum=MAX_PRIMARY_PERSON_CENTER_FILTER_FREQUENCY_HZ,
            )
        )

        if self.primary_person_center_filter == PRIMARY_PERSON_CENTER_FILTER_ONE_EURO:
            return _alpha_from_cutoff(
                frequency_hz=effective_frequency_hz,
                cutoff_hz=self.primary_person_center_filter_min_cutoff_hz,
            )

        return float(self.primary_person_center_smoothing_alpha)

    def primary_person_center_filter_config(self) -> dict[str, object]:
        """Export the center-filter configuration for downstream runtime wiring."""

        config: dict[str, object] = {
            "filter": self.primary_person_center_filter,
            "deadband": float(self.primary_person_center_deadband),
            "legacy_alpha": self.resolved_primary_person_center_smoothing_alpha(),
            "legacy_smoothing_window_s": float(self.primary_person_center_smoothing_window_s),
        }

        if self.primary_person_center_filter == PRIMARY_PERSON_CENTER_FILTER_ONE_EURO:
            config.update(
                {
                    "frequency": float(self.primary_person_center_filter_frequency_hz),
                    "min_cutoff": float(self.primary_person_center_filter_min_cutoff_hz),
                    "beta": float(self.primary_person_center_filter_beta),
                    # MediaPipe's proto field is spelled "derivate_cutoff".
                    "derivate_cutoff": float(self.primary_person_center_filter_derivative_cutoff_hz),
                    "min_allowed_object_scale": float(
                        self.primary_person_center_filter_min_allowed_object_scale
                    ),
                    "disable_value_scaling": bool(
                        self.primary_person_center_filter_disable_value_scaling
                    ),
                }
            )
        elif self.primary_person_center_filter == PRIMARY_PERSON_CENTER_FILTER_VELOCITY:
            config.update(
                {
                    "window_size": int(self.primary_person_center_filter_window_size),
                    "velocity_scale": float(self.primary_person_center_filter_velocity_scale),
                    "min_allowed_object_scale": float(
                        self.primary_person_center_filter_min_allowed_object_scale
                    ),
                    "disable_value_scaling": bool(
                        self.primary_person_center_filter_disable_value_scaling
                    ),
                }
            )

        return config

    def as_dict(self) -> dict[str, object]:
        """Return a serialization-friendly view of this config."""

        return asdict(self)

    @classmethod
    def from_config(cls, config: object) -> "ProactiveCameraSurfaceConfig":
        """Build one cadence-aware camera surface config from Twinr config.

        ``config`` may be a regular object with attributes or a mapping-like config
        loaded from TOML/YAML/JSON.
        """

        interval_s = _coerce_runtime_positive_float(
            _read_config_value(config, "proactive_capture_interval_s", default=6.0),
            default=6.0,
        )
        default_attention_refresh_s = max(
            MIN_CENTER_SMOOTHING_WINDOW_S,
            min(DEFAULT_DISPLAY_ATTENTION_REFRESH_INTERVAL_S, interval_s),
        )
        attention_refresh_s = _coerce_runtime_positive_float(
            _read_config_value(
                config,
                "display_attention_refresh_interval_s",
                "proactive_display_attention_refresh_interval_s",
                default=default_attention_refresh_s,
            ),
            default=default_attention_refresh_s,
        )

        unknown_hold_s = max(interval_s + 1.0, interval_s * 1.5)
        cooldown_s = max(interval_s, interval_s * 1.5)
        gesture_cooldown_s = max(0.8, min(2.0, attention_refresh_s * 2.0))
        fine_hand_explicit_hold_s = max(0.2, min(0.8, attention_refresh_s * 0.75))

        fine_hand_explicit_min_confidence = _coerce_runtime_bounded_ratio(
            _read_config_value(
                config,
                "proactive_local_camera_fine_hand_explicit_min_confidence",
                default=0.72,
            ),
            default=0.72,
            minimum=0.0,
            maximum=1.0,
        )

        center_smoothing_window_s = max(
            MIN_CENTER_SMOOTHING_WINDOW_S,
            min(MAX_DYNAMIC_CENTER_SMOOTHING_WINDOW_S_DEFAULT, attention_refresh_s * 1.75),
        )
        center_deadband = max(
            0.012,
            min(MAX_DYNAMIC_CENTER_DEADBAND_DEFAULT, attention_refresh_s * 0.05),
        )

        center_filter_frequency_hz = min(
            MAX_PRIMARY_PERSON_CENTER_FILTER_FREQUENCY_HZ,
            max(MIN_PRIMARY_PERSON_CENTER_FILTER_FREQUENCY_HZ, 1.0 / attention_refresh_s),
        )
        center_filter_min_cutoff_hz = min(
            1.2,
            max(0.35, 0.5 / attention_refresh_s),
        )
        center_filter_beta = min(
            1.25,
            max(0.15, 0.3 / attention_refresh_s),
        )
        center_filter_window_size = min(
            9,
            max(3, int(round(center_smoothing_window_s / attention_refresh_s)) + 1),
        )
        center_filter_velocity_scale = min(
            24.0,
            max(4.0, 8.0 / attention_refresh_s),
        )

        kwargs: dict[str, object] = {
            "person_visible_unknown_hold_s": unknown_hold_s,
            "person_visible_event_cooldown_s": cooldown_s,
            "person_recently_visible_window_s": max(30.0, interval_s * 5.0),
            "person_returned_absence_s": _coerce_runtime_positive_float(
                _read_config_value(config, "proactive_person_returned_absence_s", default=20.0 * 60.0),
                default=20.0 * 60.0,
            ),
            "looking_toward_device_unknown_hold_s": unknown_hold_s,
            "person_near_device_unknown_hold_s": unknown_hold_s,
            "engaged_with_device_unknown_hold_s": unknown_hold_s,
            "showing_intent_unknown_hold_s": unknown_hold_s,
            "showing_intent_event_cooldown_s": cooldown_s,
            "hand_or_object_near_camera_unknown_hold_s": unknown_hold_s,
            "hand_or_object_near_camera_event_cooldown_s": cooldown_s,
            "motion_event_cooldown_s": cooldown_s,
            "gesture_event_cooldown_s": gesture_cooldown_s,
            "fine_hand_explicit_hold_s": _coerce_runtime_non_negative_float(
                _read_config_value(
                    config,
                    "proactive_local_camera_fine_hand_explicit_hold_s",
                    default=fine_hand_explicit_hold_s,
                ),
                default=fine_hand_explicit_hold_s,
            ),
            "fine_hand_explicit_confirm_samples": _coerce_runtime_positive_int(
                _read_config_value(
                    config,
                    "proactive_local_camera_fine_hand_explicit_confirm_samples",
                    default=1,
                ),
                default=1,
            ),
            "fine_hand_explicit_min_confidence": fine_hand_explicit_min_confidence,
            "gesture_calibration": GestureCalibrationProfile.from_runtime_config(config),
            "primary_person_center_smoothing_alpha": 0.76,
            "primary_person_center_deadband": center_deadband,
            "primary_person_center_smoothing_window_s": center_smoothing_window_s,
            "object_unknown_hold_s": unknown_hold_s,
            "secondary_unknown_hold_s": unknown_hold_s,
            "primary_person_center_filter": PRIMARY_PERSON_CENTER_FILTER_LEGACY_EMA,
            "primary_person_center_filter_frequency_hz": center_filter_frequency_hz,
            "primary_person_center_filter_min_cutoff_hz": center_filter_min_cutoff_hz,
            "primary_person_center_filter_beta": center_filter_beta,
            "primary_person_center_filter_derivative_cutoff_hz": 1.0,
            "primary_person_center_filter_window_size": center_filter_window_size,
            "primary_person_center_filter_velocity_scale": center_filter_velocity_scale,
            "primary_person_center_filter_min_allowed_object_scale": 0.01,
            "primary_person_center_filter_disable_value_scaling": False,
        }

        kwargs = _apply_runtime_overrides(config, kwargs)
        if kwargs["primary_person_center_filter"] == PRIMARY_PERSON_CENTER_FILTER_ONE_EURO:
            kwargs["primary_person_center_smoothing_alpha"] = _alpha_from_cutoff(
                frequency_hz=float(kwargs["primary_person_center_filter_frequency_hz"]),
                cutoff_hz=float(kwargs["primary_person_center_filter_min_cutoff_hz"]),
            )
        return cls(**kwargs)
