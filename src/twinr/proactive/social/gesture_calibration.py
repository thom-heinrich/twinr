# CHANGELOG: 2026-03-29
# BUG-1: Prevent malformed numeric values (NaN/Infinity/overflow) in operator JSON from crashing calibration loading.
# BUG-2: Fix gesture-name normalization so common operator labels like Thumb_Up, Thumb_Down, Pointing_Up, Victory, OK and MiddleFinger map to Twinr enums instead of being silently ignored.
# BUG-3: Make GestureCalibrationProfile genuinely immutable; the previous frozen dataclass still exposed a mutable fine_hand dict that could be changed accidentally at runtime.
# SEC-1: Replace check-then-read file handling with bounded single-open loading, regular-file checks, and symlink refusal where the platform supports O_NOFOLLOW.
# SEC-2: Enforce strict JSON parsing for calibration files by rejecting duplicate keys and non-finite constants, preventing local config-based DoS and ambiguous overrides.
# IMP-1: Add schema-versioned named calibration profiles plus runtime/env path and profile selection for per-user / per-room deployments.
# IMP-2: Extend per-gesture policy with temporal hysteresis metadata (release_samples, cooldown_s, max_tracking_gap_s) for modern event-level gating.
# IMP-3: Add load diagnostics (issues, schema_version, active_profile, source_bytes) and wider bounded timing windows suitable for slower senior-user gesture holds.

"""
Load bounded per-gesture calibration for the social camera surface.

Twinr's live Pi gesture path should not rely on one global confirmation and
confidence threshold for every fine hand symbol. `OK_SIGN` and
`MIDDLE_FINGER` are materially easier to confuse than `THUMBS_UP`, while
`PEACE_SIGN` often benefits from a slightly longer confirmation window than a
generic `POINTING` pose.

This helper keeps that policy in one place and optionally loads an
operator-edited calibration file from `state/mediapipe/gesture_calibration.json`.

Supported file formats:
- Legacy schema: a flat object or an object with a top-level `fine_hand`.
- Schema v2: base `fine_hand` overrides plus named `profiles` and an optional
  `default_profile`, selectable through `config.gesture_calibration_profile`
  or `TWINR_GESTURE_CALIBRATION_PROFILE`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import errno
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Final, Mapping, SupportsFloat, SupportsIndex, cast

from twinr.agent.base_agent.config import TwinrConfig

from .engine import SocialFineHandGesture


_DEFAULT_CALIBRATION_PATH: Final[str] = "state/mediapipe/gesture_calibration.json"
_DEFAULT_MAX_CALIBRATION_BYTES: Final[int] = 64 * 1024
_MAX_CALIBRATION_BYTES: Final[int] = 1 * 1024 * 1024
_CURRENT_SCHEMA_VERSION: Final[int] = 2
_SUPPORTED_SCHEMA_VERSIONS: Final[frozenset[int]] = frozenset({1, 2})
_ENV_CALIBRATION_PATH: Final[str] = "TWINR_GESTURE_CALIBRATION_PATH"
_ENV_CALIBRATION_PROFILE: Final[str] = "TWINR_GESTURE_CALIBRATION_PROFILE"

_DEFAULT_MIN_CONFIDENCE: Final[float] = 0.72
_DEFAULT_CONFIRM_SAMPLES: Final[int] = 1
_DEFAULT_HOLD_S: Final[float] = 0.45

_MAX_CONFIRM_SAMPLES: Final[int] = 600
_MAX_HOLD_S: Final[float] = 5.0
_MAX_MIN_VISIBLE_S: Final[float] = 12.0
_MAX_RELEASE_SAMPLES: Final[int] = 600
_MAX_COOLDOWN_S: Final[float] = 12.0
_MAX_TRACKING_GAP_S: Final[float] = 1.0


@dataclass(frozen=True, slots=True)
class FineHandGesturePolicy:
    """Describe bounded user-facing acceptance rules for one hand symbol."""

    min_confidence: float
    confirm_samples: int
    hold_s: float
    min_visible_s: float = 0.0
    release_samples: int = 1
    cooldown_s: float = 0.0
    max_tracking_gap_s: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_confidence", _clamp_ratio(self.min_confidence, default=_DEFAULT_MIN_CONFIDENCE))
        object.__setattr__(
            self,
            "confirm_samples",
            _clamp_int(self.confirm_samples, default=_DEFAULT_CONFIRM_SAMPLES, minimum=1, maximum=_MAX_CONFIRM_SAMPLES),
        )
        object.__setattr__(
            self,
            "hold_s",
            _clamp_float(self.hold_s, default=_DEFAULT_HOLD_S, minimum=0.0, maximum=_MAX_HOLD_S),
        )
        object.__setattr__(
            self,
            "min_visible_s",
            _clamp_float(self.min_visible_s, default=0.0, minimum=0.0, maximum=_MAX_MIN_VISIBLE_S),
        )
        object.__setattr__(
            self,
            "release_samples",
            _clamp_int(self.release_samples, default=1, minimum=1, maximum=_MAX_RELEASE_SAMPLES),
        )
        object.__setattr__(
            self,
            "cooldown_s",
            _clamp_float(self.cooldown_s, default=0.0, minimum=0.0, maximum=_MAX_COOLDOWN_S),
        )
        object.__setattr__(
            self,
            "max_tracking_gap_s",
            _clamp_float(self.max_tracking_gap_s, default=0.0, minimum=0.0, maximum=_MAX_TRACKING_GAP_S),
        )

    def to_jsonable(self) -> dict[str, float | int]:
        """Return one deterministic JSON-serializable representation."""

        return {
            "min_confidence": self.min_confidence,
            "confirm_samples": self.confirm_samples,
            "hold_s": self.hold_s,
            "min_visible_s": self.min_visible_s,
            "release_samples": self.release_samples,
            "cooldown_s": self.cooldown_s,
            "max_tracking_gap_s": self.max_tracking_gap_s,
        }


@dataclass(frozen=True, slots=True)
class GestureCalibrationProfile:
    """Store calibrated per-gesture acceptance rules for the social layer."""

    fine_hand: Mapping[SocialFineHandGesture, FineHandGesturePolicy] = field(default_factory=dict)
    source_path: str | None = None
    issues: tuple[str, ...] = ()
    schema_version: int | None = None
    active_profile: str | None = None
    source_bytes: int | None = None

    def __post_init__(self) -> None:
        normalized: dict[SocialFineHandGesture, FineHandGesturePolicy] = {}
        for gesture, policy in dict(self.fine_hand).items():
            if not isinstance(gesture, SocialFineHandGesture):
                coerced_gesture = _coerce_fine_hand_gesture(gesture)
                if coerced_gesture is None:
                    continue
                gesture = coerced_gesture
            if not isinstance(policy, FineHandGesturePolicy):
                if isinstance(policy, dict):
                    policy = FineHandGesturePolicy(
                        min_confidence=policy.get("min_confidence"),
                        confirm_samples=policy.get("confirm_samples"),
                        hold_s=policy.get("hold_s"),
                        min_visible_s=policy.get("min_visible_s", 0.0),
                        release_samples=policy.get("release_samples", 1),
                        cooldown_s=policy.get("cooldown_s", 0.0),
                        max_tracking_gap_s=policy.get("max_tracking_gap_s", 0.0),
                    )
                else:
                    continue
            normalized[gesture] = policy
        # BREAKING: `fine_hand` is materialized as an immutable mapping proxy so the
        # frozen profile is actually immutable and safe to share across threads.
        object.__setattr__(self, "fine_hand", MappingProxyType(normalized))
        object.__setattr__(self, "issues", tuple(str(issue) for issue in self.issues if str(issue).strip()))
        object.__setattr__(self, "active_profile", _normalize_profile_name(self.active_profile))
        if self.schema_version is not None:
            schema_version = _coerce_int(self.schema_version, default=1)
            object.__setattr__(self, "schema_version", max(1, schema_version))
        if self.source_bytes is not None:
            object.__setattr__(self, "source_bytes", max(0, _coerce_int(self.source_bytes, default=0)))

    @classmethod
    def defaults(cls) -> "GestureCalibrationProfile":
        """Return one conservative built-in fine-hand calibration profile."""

        return cls(
            fine_hand={
                SocialFineHandGesture.THUMBS_UP: FineHandGesturePolicy(0.68, 1, 0.35, 1.0),
                SocialFineHandGesture.THUMBS_DOWN: FineHandGesturePolicy(0.78, 1, 0.35, 1.0),
                SocialFineHandGesture.POINTING: FineHandGesturePolicy(0.70, 1, 0.32),
                SocialFineHandGesture.PEACE_SIGN: FineHandGesturePolicy(0.78, 1, 0.40, 1.0),
                SocialFineHandGesture.OK_SIGN: FineHandGesturePolicy(0.86, 1, 0.46),
                SocialFineHandGesture.MIDDLE_FINGER: FineHandGesturePolicy(0.90, 1, 0.28),
            }
        )

    @classmethod
    def from_runtime_config(cls, config: TwinrConfig | object) -> "GestureCalibrationProfile":
        """Load one optional runtime calibration file with conservative fallback."""

        defaults = cls.defaults()
        return cls.from_path(
            _resolve_calibration_path(config),
            defaults=defaults,
            selected_profile=_resolve_requested_profile(config),
            max_bytes=_resolve_max_bytes(config),
        )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        defaults: "GestureCalibrationProfile | None" = None,
        selected_profile: str | None = None,
        max_bytes: int = _DEFAULT_MAX_CALIBRATION_BYTES,
    ) -> "GestureCalibrationProfile":
        """Load one calibration file from disk with strict parsing and safe fallback."""

        defaults_profile = defaults or cls.defaults()
        calibration_path = _absolutize_path(path)
        selected_profile = _normalize_profile_name(selected_profile)
        max_bytes = _clamp_int(
            max_bytes,
            default=_DEFAULT_MAX_CALIBRATION_BYTES,
            minimum=4 * 1024,
            maximum=_MAX_CALIBRATION_BYTES,
        )

        raw_text, read_issues, source_bytes, missing = _read_text_file_safely(calibration_path, max_bytes=max_bytes)
        if raw_text is None:
            if missing and not read_issues:
                return defaults_profile
            return cls(
                fine_hand=dict(defaults_profile.fine_hand),
                source_path=str(calibration_path),
                issues=read_issues,
                active_profile=selected_profile,
                source_bytes=source_bytes,
            )

        payload, parse_issues = _parse_calibration_payload(raw_text)
        if payload is None:
            return cls(
                fine_hand=dict(defaults_profile.fine_hand),
                source_path=str(calibration_path),
                issues=read_issues + parse_issues,
                active_profile=selected_profile,
                source_bytes=source_bytes,
            )

        schema_version, schema_issues = _extract_schema_version(payload)
        if schema_version is None:
            return cls(
                fine_hand=dict(defaults_profile.fine_hand),
                source_path=str(calibration_path),
                issues=read_issues + parse_issues + schema_issues,
                active_profile=selected_profile,
                source_bytes=source_bytes,
            )
        if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
            return cls(
                fine_hand=dict(defaults_profile.fine_hand),
                source_path=str(calibration_path),
                issues=read_issues + parse_issues + schema_issues + (
                    f"unsupported schema_version={schema_version}; supported={sorted(_SUPPORTED_SCHEMA_VERSIONS)}",
                ),
                schema_version=schema_version,
                active_profile=selected_profile,
                source_bytes=source_bytes,
            )

        merged_base, merge_issues = _merge_fine_hand_policies(
            defaults_profile.fine_hand,
            payload,
            source_label="fine_hand",
        )

        merged_policies = merged_base
        active_profile = selected_profile
        profile_issues: tuple[str, ...] = ()
        if schema_version >= 2:
            default_profile = None
            if isinstance(payload, dict):
                default_profile = _normalize_profile_name(payload.get("default_profile"))
            merged_policies, profile_issues, active_profile = _apply_named_profile(
                merged_base,
                payload,
                requested_profile=selected_profile or default_profile,
            )

        return cls(
            fine_hand=merged_policies,
            source_path=str(calibration_path),
            issues=read_issues + parse_issues + schema_issues + merge_issues + profile_issues,
            schema_version=schema_version,
            active_profile=active_profile,
            source_bytes=source_bytes,
        )

    def fine_hand_policy(
        self,
        gesture: SocialFineHandGesture,
        *,
        fallback_min_confidence: float,
        fallback_confirm_samples: int,
        fallback_hold_s: float,
        fallback_min_visible_s: float = 0.0,
        fallback_release_samples: int = 1,
        fallback_cooldown_s: float = 0.0,
        fallback_max_tracking_gap_s: float = 0.0,
    ) -> FineHandGesturePolicy:
        """Return the calibrated policy for one gesture with bounded fallback."""

        calibrated = self.fine_hand.get(gesture)
        if calibrated is not None:
            return calibrated
        return FineHandGesturePolicy(
            min_confidence=fallback_min_confidence,
            confirm_samples=fallback_confirm_samples,
            hold_s=fallback_hold_s,
            min_visible_s=fallback_min_visible_s,
            release_samples=fallback_release_samples,
            cooldown_s=fallback_cooldown_s,
            max_tracking_gap_s=fallback_max_tracking_gap_s,
        )

    def to_jsonable(self) -> dict[str, object]:
        """Return one deterministic JSON-serializable representation."""

        fine_hand_payload: dict[str, object] = {}
        for gesture in sorted(self.fine_hand, key=lambda item: str(getattr(item, "value", item.name))):
            gesture_name = str(getattr(gesture, "value", gesture.name)).lower()
            fine_hand_payload[gesture_name] = self.fine_hand[gesture].to_jsonable()
        return {
            "schema_version": _CURRENT_SCHEMA_VERSION,
            "fine_hand": fine_hand_payload,
        }


def _apply_named_profile(
    base_policies: Mapping[SocialFineHandGesture, FineHandGesturePolicy],
    payload: object,
    *,
    requested_profile: str | None,
) -> tuple[dict[SocialFineHandGesture, FineHandGesturePolicy], tuple[str, ...], str | None]:
    """Merge one named profile from a schema-versioned payload onto base policies."""

    if requested_profile is None:
        return dict(base_policies), (), None
    if not isinstance(payload, dict):
        return dict(base_policies), (f"ignored requested calibration profile {requested_profile!r}: root payload is not an object",), None

    profiles_payload = payload.get("profiles")
    if profiles_payload is None:
        return dict(base_policies), (f"ignored requested calibration profile {requested_profile!r}: file has no 'profiles' section",), None
    if not isinstance(profiles_payload, dict):
        return dict(base_policies), ("ignored invalid 'profiles' section: expected an object",), None

    actual_profile_name, profile_payload = _lookup_mapping_entry(profiles_payload, requested_profile)
    if actual_profile_name is None:
        return dict(base_policies), (f"requested calibration profile {requested_profile!r} was not found",), None

    merged, issues = _merge_fine_hand_policies(
        base_policies,
        profile_payload,
        source_label=f"profiles.{actual_profile_name}",
    )
    return merged, issues, actual_profile_name


def _merge_fine_hand_policies(
    defaults: Mapping[SocialFineHandGesture, FineHandGesturePolicy],
    payload: object,
    *,
    source_label: str,
) -> tuple[dict[SocialFineHandGesture, FineHandGesturePolicy], tuple[str, ...]]:
    """Merge one JSON payload onto the built-in fine-hand defaults."""

    merged = dict(defaults)
    issues: list[str] = []
    fine_hand_payload = payload
    if isinstance(payload, dict):
        fine_hand_payload = payload.get("fine_hand", payload)
    if fine_hand_payload is None:
        return merged, ()
    if not isinstance(fine_hand_payload, dict):
        return merged, (f"ignored invalid {source_label}: expected an object",)

    for raw_name, raw_policy in fine_hand_payload.items():
        gesture = _coerce_fine_hand_gesture(raw_name)
        if gesture is None:
            issues.append(f"ignored unknown fine-hand gesture {raw_name!r} in {source_label}")
            continue
        if gesture in {SocialFineHandGesture.NONE, SocialFineHandGesture.UNKNOWN}:
            issues.append(f"ignored non-actionable fine-hand gesture {raw_name!r} in {source_label}")
            continue
        if not isinstance(raw_policy, dict):
            issues.append(f"ignored invalid policy for {gesture.name} in {source_label}: expected an object")
            continue

        fallback = merged.get(gesture, FineHandGesturePolicy(_DEFAULT_MIN_CONFIDENCE, 1, _DEFAULT_HOLD_S))
        merged[gesture] = FineHandGesturePolicy(
            min_confidence=_coerce_float(raw_policy.get("min_confidence"), default=fallback.min_confidence),
            confirm_samples=_coerce_int(raw_policy.get("confirm_samples"), default=fallback.confirm_samples),
            hold_s=_coerce_float(raw_policy.get("hold_s"), default=fallback.hold_s),
            min_visible_s=_coerce_float(raw_policy.get("min_visible_s"), default=fallback.min_visible_s),
            release_samples=_coerce_int(raw_policy.get("release_samples"), default=fallback.release_samples),
            cooldown_s=_coerce_float(raw_policy.get("cooldown_s"), default=fallback.cooldown_s),
            max_tracking_gap_s=_coerce_float(
                raw_policy.get("max_tracking_gap_s"),
                default=fallback.max_tracking_gap_s,
            ),
        )

    return merged, tuple(issues)


def _coerce_fine_hand_gesture(value: object) -> SocialFineHandGesture | None:
    """Normalize one token into a known fine-hand gesture enum."""

    normalized = _normalize_token(value)
    if not normalized:
        return None

    aliases = _fine_hand_alias_map()
    gesture = aliases.get(normalized) or aliases.get(normalized.replace("_", ""))
    if gesture is not None:
        return gesture

    try:
        return SocialFineHandGesture(normalized)
    except (TypeError, ValueError):
        pass
    try:
        return SocialFineHandGesture[normalized.upper()]
    except (KeyError, TypeError):
        return None


@lru_cache(maxsize=1)
def _fine_hand_alias_map() -> dict[str, SocialFineHandGesture]:
    """Build one robust alias map for operator- and MediaPipe-style gesture names."""

    aliases: dict[str, SocialFineHandGesture] = {}
    for gesture in SocialFineHandGesture:
        _register_aliases(aliases, gesture, getattr(gesture, "name", ""))
        _register_aliases(aliases, gesture, getattr(gesture, "value", ""))

    _bind_aliases(aliases, SocialFineHandGesture.PEACE_SIGN, "peace", "victory", "victory_sign", "peace_sign", "v_sign")
    _bind_aliases(aliases, SocialFineHandGesture.POINTING, "point", "pointing", "point_up", "pointing_up", "index_point")
    _bind_aliases(aliases, SocialFineHandGesture.THUMBS_UP, "thumb_up", "thumbs_up", "thumbsup", "thumbs-up")
    _bind_aliases(aliases, SocialFineHandGesture.THUMBS_DOWN, "thumb_down", "thumbs_down", "thumbsdown", "thumbs-down")
    _bind_aliases(aliases, SocialFineHandGesture.OK_SIGN, "ok", "okay", "ok_sign", "okay_sign", "oksign")
    _bind_aliases(aliases, SocialFineHandGesture.MIDDLE_FINGER, "middle", "middle_finger", "middlefinger")
    return aliases


def _register_aliases(
    aliases: dict[str, SocialFineHandGesture],
    gesture: SocialFineHandGesture,
    *raw_values: object,
) -> None:
    """Register normalized aliases for one gesture."""

    for raw in raw_values:
        normalized = _normalize_token(raw)
        if not normalized:
            continue
        aliases.setdefault(normalized, gesture)
        collapsed = normalized.replace("_", "")
        if collapsed:
            aliases.setdefault(collapsed, gesture)


def _bind_aliases(
    aliases: dict[str, SocialFineHandGesture],
    gesture: SocialFineHandGesture,
    *raw_values: object,
) -> None:
    """Bind explicit operator-friendly aliases to one gesture."""

    _register_aliases(aliases, gesture, *raw_values)


def _normalize_token(value: object) -> str:
    """Normalize one token into a lowercase underscore form."""

    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    parts: list[str] = []
    current: list[str] = []
    for char in raw:
        if char.isalnum():
            current.append(char)
            continue
        if current:
            parts.append("".join(current))
            current.clear()
    if current:
        parts.append("".join(current))
    return "_".join(part for part in parts if part)


def _normalize_profile_name(value: object) -> str | None:
    """Normalize one requested profile name while preserving readable keys."""

    text = str(value or "").strip()
    return text or None


def _lookup_mapping_entry(mapping: Mapping[object, object], requested_key: str) -> tuple[str | None, object | None]:
    """Look up one mapping key with exact-then-normalized matching."""

    if requested_key in mapping:
        return requested_key, mapping[requested_key]

    normalized_requested = _normalize_token(requested_key)
    for raw_key, raw_value in mapping.items():
        candidate = str(raw_key)
        if _normalize_token(candidate) == normalized_requested:
            return candidate, raw_value
    return None, None


def _absolutize_path(path: str | Path) -> Path:
    """Return one absolute path without resolving the final component's symlink."""

    candidate = Path(path).expanduser()
    return Path(os.path.abspath(os.fspath(candidate)))


def _resolve_calibration_path(config: TwinrConfig | object) -> Path:
    """Resolve the calibration path from env/config with project-root fallback."""

    raw_override = os.environ.get(_ENV_CALIBRATION_PATH) or getattr(config, "gesture_calibration_path", None)
    if raw_override:
        return _absolutize_path(raw_override)

    project_root = _absolutize_path(getattr(config, "project_root", ".") or ".")
    return _absolutize_path(project_root / _DEFAULT_CALIBRATION_PATH)


def _resolve_requested_profile(config: TwinrConfig | object) -> str | None:
    """Resolve one requested calibration profile from env/config."""

    return _normalize_profile_name(
        os.environ.get(_ENV_CALIBRATION_PROFILE) or getattr(config, "gesture_calibration_profile", None)
    )


def _resolve_max_bytes(config: TwinrConfig | object) -> int:
    """Resolve one bounded calibration file size limit."""

    return _clamp_int(
        getattr(config, "gesture_calibration_max_bytes", _DEFAULT_MAX_CALIBRATION_BYTES),
        default=_DEFAULT_MAX_CALIBRATION_BYTES,
        minimum=4 * 1024,
        maximum=_MAX_CALIBRATION_BYTES,
    )


def _read_text_file_safely(path: Path, *, max_bytes: int) -> tuple[str | None, tuple[str, ...], int | None, bool]:
    """Read one UTF-8 regular file with bounded size and symlink hardening."""

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC

    use_no_follow = os.name == "posix" and hasattr(os, "O_NOFOLLOW")
    if use_no_follow:
        flags |= os.O_NOFOLLOW
    else:
        try:
            if path.is_symlink():
                return None, (f"refused symlink gesture calibration file: {path}",), None, False
        except OSError as exc:
            return None, (f"failed to inspect gesture calibration path {path}: {exc}",), None, False

    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        return None, (), None, True
    except NotADirectoryError:
        return None, (), None, True
    except OSError as exc:
        if use_no_follow and exc.errno == errno.ELOOP:
            return None, (f"refused symlink gesture calibration file: {path}",), None, False
        return None, (f"failed to open gesture calibration file {path}: {exc}",), None, False

    try:
        with os.fdopen(fd, "rb", closefd=True) as handle:
            metadata = os.fstat(handle.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                return None, (f"gesture calibration path is not a regular file: {path}",), int(metadata.st_size), False
            if metadata.st_size > max_bytes:
                return None, (
                    f"gesture calibration file exceeds {max_bytes} bytes: {path}",
                ), int(metadata.st_size), False

            data = handle.read(max_bytes + 1)
            if len(data) > max_bytes:
                return None, (
                    f"gesture calibration file exceeds {max_bytes} bytes while reading: {path}",
                ), len(data), False
    except OSError as exc:
        return None, (f"failed to read gesture calibration file {path}: {exc}",), None, False

    try:
        return data.decode("utf-8"), (), len(data), False
    except UnicodeDecodeError as exc:
        return None, (f"gesture calibration file is not valid UTF-8: {path} ({exc})",), len(data), False


def _parse_calibration_payload(raw_text: str) -> tuple[object | None, tuple[str, ...]]:
    """Parse one calibration JSON document with strict duplicate/non-finite rejection."""

    try:
        payload = json.loads(
            raw_text,
            object_pairs_hook=_reject_duplicate_object_pairs,
            parse_constant=_reject_invalid_json_constant,
        )
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        return None, (f"failed to parse gesture calibration JSON: {exc}",)
    return payload, ()


def _reject_duplicate_object_pairs(pairs: list[tuple[object, object]]) -> dict[object, object]:
    """Reject duplicate JSON keys so operator mistakes are never ambiguous."""

    result: dict[object, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_invalid_json_constant(token: str) -> float:
    """Reject NaN/Infinity constants that Python's JSON decoder otherwise accepts."""

    raise ValueError(f"non-finite JSON constant {token!r} is not allowed")


def _extract_schema_version(payload: object) -> tuple[int | None, tuple[str, ...]]:
    """Extract one supported schema version from the calibration payload."""

    if not isinstance(payload, dict):
        return None, ("gesture calibration root must be a JSON object",)

    if "schema_version" not in payload:
        inferred_version = 2 if ("profiles" in payload or "default_profile" in payload) else 1
        return inferred_version, ()

    raw_schema_version = payload.get("schema_version")
    schema_version = _coerce_int(raw_schema_version, default=-1)
    if schema_version < 1:
        return None, (f"invalid schema_version {raw_schema_version!r}",)
    return schema_version, ()


def _coerce_float(value: object, *, default: float) -> float:
    """Return one finite float with a safe fallback."""

    if value is None or isinstance(value, bool):
        return default
    try:
        numeric = float(cast(str | bytes | bytearray | SupportsFloat | SupportsIndex, value))
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def _coerce_int(value: object, *, default: int) -> int:
    """Return one finite integer with a safe fallback."""

    if value is None or isinstance(value, bool):
        return default
    try:
        numeric = int(cast(str | bytes | bytearray | SupportsIndex, value))
    except (TypeError, ValueError, OverflowError):
        return default
    return numeric


def _clamp_ratio(value: object, *, default: float) -> float:
    """Clamp one ratio-like value into [0.0, 1.0]."""

    numeric = _coerce_float(value, default=default)
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def _clamp_float(value: object, *, default: float, minimum: float, maximum: float) -> float:
    """Clamp one finite float into a closed interval."""

    numeric = _coerce_float(value, default=default)
    if numeric < minimum:
        return minimum
    if numeric > maximum:
        return maximum
    return numeric


def _clamp_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    """Clamp one finite integer into a closed interval."""

    numeric = _coerce_int(value, default=default)
    if numeric < minimum:
        return minimum
    if numeric > maximum:
        return maximum
    return numeric


DEFAULT_HDMI_ACK_FINE_HAND_POLICIES: Final[dict[SocialFineHandGesture, FineHandGesturePolicy]] = {
    # These Pi-facing HDMI acknowledgement floors are intentionally lower than the
    # broader social defaults. They preserve the accepted gesture baseline for the
    # authoritative live stream without loosening the wider social camera policy.
    SocialFineHandGesture.THUMBS_UP: FineHandGesturePolicy(0.48, 1, 0.35, 1.0),
    SocialFineHandGesture.THUMBS_DOWN: FineHandGesturePolicy(0.37, 1, 0.35, 1.0),
    SocialFineHandGesture.PEACE_SIGN: FineHandGesturePolicy(0.60, 1, 0.40, 1.0),
}


__all__ = [
    "DEFAULT_HDMI_ACK_FINE_HAND_POLICIES",
    "FineHandGesturePolicy",
    "GestureCalibrationProfile",
]
