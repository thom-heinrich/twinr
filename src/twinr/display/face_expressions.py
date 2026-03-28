# CHANGELOG: 2026-03-28
# BUG-1: clear() is now source-aware and lock-protected so one producer no longer removes another producer's cue by accident.
# BUG-2: free-form numeric inputs and naive datetimes are normalized safely; inf/NaN and local-time drift no longer crash or skew TTLs.
# SEC-1: cue persistence no longer delegates to a predictable temporary filename; writes use a sidecar lock, unique temp files, fsync, and atomic replace.
# SEC-2: source labels are sanitized and length-bounded to avoid control-character pollution and trivial artifact bloat.
# IMP-1: added affective control primitives (valence/arousal + intensity-aware presets) for low-latency non-verbal backchanneling.
# IMP-2: added strict validation, richer aliases, direct-constructor normalization, and source-scoped clear/preview helpers.

"""Producer-facing helpers for HDMI face expressions.

Other Twinr modules should not have to write raw JSON cue payloads just to make
the HDMI face look left, raise its brows, or smile briefly. This module keeps a
small combinable expression API on top of the file-backed cue store so runtime
producers can express intent in one call.
"""

from __future__ import annotations

from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import math
import os
from os import PathLike
from pathlib import Path
import re
import tempfile
from typing import Protocol

from twinr.display.face_cues import DisplayFaceCue, DisplayFaceCueStore

try:  # Raspberry Pi / Linux fast path.
    import fcntl
except ImportError:  # pragma: no cover - non-Unix fallback.
    fcntl = None


_LOGGER = logging.getLogger(__name__)

_MAX_GAZE_AXIS = 3
_MAX_HEAD_AXIS = 2
_MIN_HOLD_SECONDS = 0.1
_DEFAULT_MAX_HOLD_SECONDS = 300.0
_DEFAULT_LOCK_SUFFIX = ".lock"
_MAX_SOURCE_LENGTH = 64
_SOURCE_ALLOWED_CHARS = re.compile(r"[^A-Za-z0-9._:/-]+")

_UNSET = object()


class DisplayFaceExpressionInputError(ValueError):
    """Raised when a producer-facing face-expression input is invalid."""


class DisplayFaceBrowStyle(str, Enum):
    """Supported HDMI eyebrow shapes for external expression triggers."""

    STRAIGHT = "straight"
    INWARD_TILT = "inward_tilt"
    OUTWARD_TILT = "outward_tilt"
    ROOF = "roof"
    RAISED = "raised"
    SOFT = "soft"


class DisplayFaceMouthStyle(str, Enum):
    """Supported HDMI mouth shapes for external expression triggers."""

    NEUTRAL = "neutral"
    SMILE = "smile"
    SAD = "sad"
    THINKING = "thinking"
    PURSED = "pursed"
    SCRUNCHED = "scrunched"
    OPEN = "open"
    SPEAK = "speak"


class DisplayFaceGazeDirection(str, Enum):
    """Discrete gaze directions in 45-degree steps plus center."""

    CENTER = "center"
    UP = "up"
    UP_RIGHT = "up_right"
    RIGHT = "right"
    DOWN_RIGHT = "down_right"
    DOWN = "down"
    DOWN_LEFT = "down_left"
    LEFT = "left"
    UP_LEFT = "up_left"

    def axes(self) -> tuple[int, int]:
        """Return the normalized cue axes for this direction."""

        return _GAZE_AXES[self]


class DisplayFaceEmotion(str, Enum):
    """Producer-facing emotion/state presets for HDMI expressions."""

    CALM = "calm"
    HAPPY = "happy"
    SAD = "sad"
    THOUGHTFUL = "thoughtful"
    CURIOUS = "curious"
    FOCUSED = "focused"
    LISTENING = "listening"
    SPEAKING = "speaking"
    REASSURING = "reassuring"
    ALERT = "alert"


_GAZE_AXES: dict[DisplayFaceGazeDirection, tuple[int, int]] = {
    DisplayFaceGazeDirection.CENTER: (0, 0),
    DisplayFaceGazeDirection.UP: (0, -2),
    DisplayFaceGazeDirection.UP_RIGHT: (2, -2),
    DisplayFaceGazeDirection.RIGHT: (2, 0),
    DisplayFaceGazeDirection.DOWN_RIGHT: (2, 2),
    DisplayFaceGazeDirection.DOWN: (0, 2),
    DisplayFaceGazeDirection.DOWN_LEFT: (-2, 2),
    DisplayFaceGazeDirection.LEFT: (-2, 0),
    DisplayFaceGazeDirection.UP_LEFT: (-2, -2),
}

_GAZE_ALIASES = {
    "forward": DisplayFaceGazeDirection.CENTER,
    "centre": DisplayFaceGazeDirection.CENTER,
    "upper_right": DisplayFaceGazeDirection.UP_RIGHT,
    "upper_left": DisplayFaceGazeDirection.UP_LEFT,
    "lower_right": DisplayFaceGazeDirection.DOWN_RIGHT,
    "lower_left": DisplayFaceGazeDirection.DOWN_LEFT,
    "up-right": DisplayFaceGazeDirection.UP_RIGHT,
    "down-right": DisplayFaceGazeDirection.DOWN_RIGHT,
    "down-left": DisplayFaceGazeDirection.DOWN_LEFT,
    "up-left": DisplayFaceGazeDirection.UP_LEFT,
    "upright": DisplayFaceGazeDirection.UP_RIGHT,
    "upleft": DisplayFaceGazeDirection.UP_LEFT,
    "downright": DisplayFaceGazeDirection.DOWN_RIGHT,
    "downleft": DisplayFaceGazeDirection.DOWN_LEFT,
}

_BROW_ALIASES = {
    "flat": DisplayFaceBrowStyle.STRAIGHT,
    "focus": DisplayFaceBrowStyle.INWARD_TILT,
    "focused": DisplayFaceBrowStyle.INWARD_TILT,
    "concern": DisplayFaceBrowStyle.SOFT,
    "concerned": DisplayFaceBrowStyle.SOFT,
}

_MOUTH_ALIASES = {
    "line": DisplayFaceMouthStyle.NEUTRAL,
    "concern": DisplayFaceMouthStyle.SAD,
    "concerned": DisplayFaceMouthStyle.SAD,
    "talk": DisplayFaceMouthStyle.SPEAK,
    "talking": DisplayFaceMouthStyle.SPEAK,
    "speaking": DisplayFaceMouthStyle.SPEAK,
}

_EMOTION_ALIASES = {
    "neutral": DisplayFaceEmotion.CALM,
    "engaged": DisplayFaceEmotion.LISTENING,
    "speak": DisplayFaceEmotion.SPEAKING,
    "talking": DisplayFaceEmotion.SPEAKING,
    "thinking": DisplayFaceEmotion.THOUGHTFUL,
    "comforting": DisplayFaceEmotion.REASSURING,
    "attentive": DisplayFaceEmotion.LISTENING,
}


class _DisplayConfigLike(Protocol):
    """Describe the minimal config surface needed by expression controllers."""

    project_root: str | PathLike[str]


def _normalize_label(value: object | None) -> str:
    """Normalize one optional free-form enum label."""

    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _raise_or_none(*, strict: bool, message: str) -> None:
    if strict:
        raise DisplayFaceExpressionInputError(message)
    _LOGGER.warning(message)
    return None


def _normalize_now(value: datetime | None) -> datetime:
    """Return one aware UTC timestamp, treating naive values as already-UTC."""

    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _coerce_unit_interval(
    value: float | int | str | None,
    *,
    field_name: str,
    default: float,
    strict: bool,
) -> float:
    """Normalize one scalar to [0.0, 1.0]."""

    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        _raise_or_none(strict=strict, message=f"Invalid {field_name!r}: {value!r}.")
        return default
    if not math.isfinite(parsed):
        _raise_or_none(strict=strict, message=f"Non-finite {field_name!r}: {value!r}.")
        return default
    return max(0.0, min(1.0, parsed))


def _coerce_signed_scalar(
    value: float | int | str,
    *,
    field_name: str,
    minimum: float,
    maximum: float,
    strict: bool,
) -> float:
    """Normalize one scalar to [minimum, maximum]."""

    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        _raise_or_none(strict=strict, message=f"Invalid {field_name!r}: {value!r}.")
        return 0.0
    if not math.isfinite(parsed):
        _raise_or_none(strict=strict, message=f"Non-finite {field_name!r}: {value!r}.")
        return 0.0
    return max(minimum, min(maximum, parsed))


def _coerce_axis(
    value: object | None,
    *,
    field_name: str,
    maximum: int,
    strict: bool,
) -> int:
    """Normalize one signed cue-axis value into a bounded integer."""

    if value is None:
        return 0
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        _raise_or_none(strict=strict, message=f"Invalid {field_name!r}: {value!r}.")
        return 0
    if not math.isfinite(parsed):
        _raise_or_none(strict=strict, message=f"Non-finite {field_name!r}: {value!r}.")
        return 0
    return max(-maximum, min(maximum, int(round(parsed))))


def _coerce_optional_bool(
    value: object | None,
    *,
    field_name: str,
    strict: bool,
) -> bool | None:
    """Normalize one optional boolean input."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    label = _normalize_label(value)
    if label in {"1", "true", "yes", "on"}:
        return True
    if label in {"0", "false", "no", "off"}:
        return False
    _raise_or_none(strict=strict, message=f"Invalid {field_name!r}: {value!r}.")
    return None


def _sanitize_source(value: object | None) -> str:
    """Return one bounded, printable producer-source label."""

    text = str(value or "").strip()
    if not text:
        return "external"
    text = "".join(ch for ch in text if ch.isprintable() and ch not in {"\r", "\n", "\t"})
    text = _SOURCE_ALLOWED_CHARS.sub("_", text).strip("._:/- ")
    text = text[:_MAX_SOURCE_LENGTH]
    return text or "external"


def _coerce_gaze_direction(
    value: DisplayFaceGazeDirection | str | None,
    *,
    strict: bool,
) -> DisplayFaceGazeDirection | None:
    """Parse one optional gaze direction value."""

    if value is None or isinstance(value, DisplayFaceGazeDirection):
        return value
    label = _normalize_label(value)
    if not label:
        return None
    alias = _GAZE_ALIASES.get(label)
    if alias is not None:
        return alias
    try:
        return DisplayFaceGazeDirection(label)
    except ValueError:
        _raise_or_none(strict=strict, message=f"Invalid 'gaze': {value!r}.")
        return None


def _coerce_brow_style(
    value: DisplayFaceBrowStyle | str | None,
    *,
    strict: bool,
) -> DisplayFaceBrowStyle | None:
    """Parse one optional brow style value."""

    if value is None or isinstance(value, DisplayFaceBrowStyle):
        return value
    label = _normalize_label(value)
    if not label:
        return None
    alias = _BROW_ALIASES.get(label)
    if alias is not None:
        return alias
    try:
        return DisplayFaceBrowStyle(label)
    except ValueError:
        _raise_or_none(strict=strict, message=f"Invalid 'brows': {value!r}.")
        return None


def _coerce_mouth_style(
    value: DisplayFaceMouthStyle | str | None,
    *,
    strict: bool,
) -> DisplayFaceMouthStyle | None:
    """Parse one optional mouth style value."""

    if value is None or isinstance(value, DisplayFaceMouthStyle):
        return value
    label = _normalize_label(value)
    if not label:
        return None
    alias = _MOUTH_ALIASES.get(label)
    if alias is not None:
        return alias
    try:
        return DisplayFaceMouthStyle(label)
    except ValueError:
        _raise_or_none(strict=strict, message=f"Invalid 'mouth': {value!r}.")
        return None


def _coerce_emotion(
    value: DisplayFaceEmotion | str | None,
    *,
    strict: bool,
) -> DisplayFaceEmotion | None:
    """Parse one optional emotion preset value."""

    if value is None or isinstance(value, DisplayFaceEmotion):
        return value
    label = _normalize_label(value)
    if not label:
        return None
    alias = _EMOTION_ALIASES.get(label)
    if alias is not None:
        return alias
    try:
        return DisplayFaceEmotion(label)
    except ValueError:
        _raise_or_none(strict=strict, message=f"Invalid 'emotion': {value!r}.")
        return None


def _normalize_hold_seconds(
    hold_seconds: float | int | str | None,
    *,
    default_ttl_s: float,
    maximum_ttl_s: float,
    strict: bool,
) -> float:
    """Return one safe hold duration for persisted cues."""

    default = max(_MIN_HOLD_SECONDS, min(maximum_ttl_s, float(default_ttl_s)))
    if hold_seconds is None:
        return default
    try:
        parsed = float(hold_seconds)
    except (TypeError, ValueError, OverflowError):
        _raise_or_none(strict=strict, message=f"Invalid 'hold_seconds': {hold_seconds!r}.")
        return default
    if not math.isfinite(parsed):
        _raise_or_none(strict=strict, message=f"Non-finite 'hold_seconds': {hold_seconds!r}.")
        return default
    return max(_MIN_HOLD_SECONDS, min(maximum_ttl_s, parsed))


def _parse_timestamp(value: str | None) -> datetime | None:
    """Parse one ISO-8601 timestamp into aware UTC."""

    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _cue_signature_without_timestamps(cue: DisplayFaceCue) -> tuple[object, ...]:
    """Return a semantic cue signature without volatile timestamps."""

    return (
        _sanitize_source(cue.source),
        cue.gaze_x,
        cue.gaze_y,
        cue.head_dx,
        cue.head_dy,
        cue.mouth,
        cue.brows,
        cue.blink,
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write one artifact with a unique temp file and atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            except OSError:
                pass
            finally:
                os.close(dir_fd)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


@dataclass(frozen=True, slots=True)
class DisplayFaceExpression:
    """Describe one producer-facing HDMI face expression."""

    gaze: DisplayFaceGazeDirection | str = DisplayFaceGazeDirection.CENTER
    mouth: DisplayFaceMouthStyle | str | None = None
    brows: DisplayFaceBrowStyle | str | None = None
    blink: bool | str | None = None
    head_dx: int | float | str = 0
    head_dy: int | float | str = 0

    def __post_init__(self) -> None:
        """Normalize direct constructor calls into the stable expression model."""

        object.__setattr__(
            self,
            "gaze",
            _coerce_gaze_direction(self.gaze, strict=True) or DisplayFaceGazeDirection.CENTER,
        )
        object.__setattr__(self, "mouth", _coerce_mouth_style(self.mouth, strict=True))
        object.__setattr__(self, "brows", _coerce_brow_style(self.brows, strict=True))
        object.__setattr__(self, "blink", _coerce_optional_bool(self.blink, field_name="blink", strict=True))
        object.__setattr__(self, "head_dx", _coerce_axis(self.head_dx, field_name="head_dx", maximum=_MAX_HEAD_AXIS, strict=True))
        object.__setattr__(self, "head_dy", _coerce_axis(self.head_dy, field_name="head_dy", maximum=_MAX_HEAD_AXIS, strict=True))

    @classmethod
    def from_inputs(
        cls,
        *,
        gaze: DisplayFaceGazeDirection | str | None = None,
        mouth: DisplayFaceMouthStyle | str | None = None,
        brows: DisplayFaceBrowStyle | str | None = None,
        blink: bool | str | None = None,
        head_dx: int | float | str = 0,
        head_dy: int | float | str = 0,
        strict: bool = True,
    ) -> "DisplayFaceExpression":
        """Build one expression directly from producer-facing inputs."""

        return cls(
            gaze=_coerce_gaze_direction(gaze, strict=strict) or DisplayFaceGazeDirection.CENTER,
            mouth=_coerce_mouth_style(mouth, strict=strict),
            brows=_coerce_brow_style(brows, strict=strict),
            blink=_coerce_optional_bool(blink, field_name="blink", strict=strict),
            head_dx=_coerce_axis(head_dx, field_name="head_dx", maximum=_MAX_HEAD_AXIS, strict=strict),
            head_dy=_coerce_axis(head_dy, field_name="head_dy", maximum=_MAX_HEAD_AXIS, strict=strict),
        )

    @classmethod
    def from_emotion(
        cls,
        emotion: DisplayFaceEmotion | str,
        *,
        intensity: float | int | str = 1.0,
        gaze: DisplayFaceGazeDirection | str | None = None,
        mouth: DisplayFaceMouthStyle | str | None = None,
        brows: DisplayFaceBrowStyle | str | None = None,
        blink: bool | str | None = None,
        head_dx: int | float | str | object = _UNSET,
        head_dy: int | float | str | object = _UNSET,
        strict: bool = True,
    ) -> "DisplayFaceExpression":
        """Build one expression from an emotion preset plus optional overrides."""

        parsed_emotion = _coerce_emotion(emotion, strict=strict)
        if parsed_emotion is None:
            if strict:
                raise DisplayFaceExpressionInputError(f"Invalid 'emotion': {emotion!r}.")
            parsed_emotion = DisplayFaceEmotion.CALM
        preset = _emotion_expression(parsed_emotion, intensity=intensity, strict=strict)
        return preset.with_overrides(
            gaze=gaze,
            mouth=mouth if mouth is not None else _UNSET,
            brows=brows if brows is not None else _UNSET,
            blink=blink if blink is not None else _UNSET,
            head_dx=head_dx,
            head_dy=head_dy,
            strict=strict,
        )

    @classmethod
    def from_affect(
        cls,
        *,
        valence: float | int | str,
        arousal: float | int | str,
        blink: bool | str | None = None,
        head_dx: int | float | str | object = _UNSET,
        head_dy: int | float | str | object = _UNSET,
        strict: bool = True,
    ) -> "DisplayFaceExpression":
        """Project one continuous affect state into the supported discrete face cue."""

        signed_valence = _coerce_signed_scalar(
            valence,
            field_name="valence",
            minimum=-1.0,
            maximum=1.0,
            strict=strict,
        )
        unit_arousal = _coerce_unit_interval(
            arousal,
            field_name="arousal",
            default=0.0,
            strict=strict,
        )

        if signed_valence >= 0.5:
            base = cls(
                gaze=DisplayFaceGazeDirection.UP_RIGHT if unit_arousal >= 0.8 else DisplayFaceGazeDirection.CENTER,
                mouth=DisplayFaceMouthStyle.OPEN if unit_arousal >= 0.85 else DisplayFaceMouthStyle.SMILE,
                brows=DisplayFaceBrowStyle.RAISED if unit_arousal >= 0.35 else DisplayFaceBrowStyle.SOFT,
                head_dx=1 if unit_arousal >= 0.8 else 0,
                head_dy=-1 if unit_arousal >= 0.55 else 0,
            )
        elif signed_valence <= -0.5:
            base = cls(
                gaze=DisplayFaceGazeDirection.CENTER if unit_arousal >= 0.7 else DisplayFaceGazeDirection.DOWN,
                mouth=DisplayFaceMouthStyle.SCRUNCHED if unit_arousal >= 0.8 else DisplayFaceMouthStyle.SAD,
                brows=DisplayFaceBrowStyle.INWARD_TILT if unit_arousal >= 0.5 else DisplayFaceBrowStyle.SOFT,
                head_dx=0,
                head_dy=0 if unit_arousal >= 0.7 else 1,
            )
        else:
            base = cls(
                gaze=DisplayFaceGazeDirection.CENTER if unit_arousal >= 0.4 else DisplayFaceGazeDirection.UP_LEFT,
                mouth=DisplayFaceMouthStyle.OPEN if unit_arousal >= 0.9 else (DisplayFaceMouthStyle.NEUTRAL if unit_arousal >= 0.35 else DisplayFaceMouthStyle.THINKING),
                brows=DisplayFaceBrowStyle.RAISED if unit_arousal >= 0.8 else (DisplayFaceBrowStyle.STRAIGHT if unit_arousal >= 0.35 else DisplayFaceBrowStyle.ROOF),
                head_dx=0 if unit_arousal >= 0.35 else -1,
                head_dy=0,
            )

        return base.with_overrides(
            blink=blink if blink is not None else _UNSET,
            head_dx=head_dx,
            head_dy=head_dy,
            strict=strict,
        )

    def with_overrides(
        self,
        *,
        gaze: DisplayFaceGazeDirection | str | None = None,
        mouth: DisplayFaceMouthStyle | str | None | object = _UNSET,
        brows: DisplayFaceBrowStyle | str | None | object = _UNSET,
        blink: bool | str | None | object = _UNSET,
        head_dx: int | float | str | object = _UNSET,
        head_dy: int | float | str | object = _UNSET,
        strict: bool = True,
    ) -> "DisplayFaceExpression":
        """Return one new expression with bounded field overrides applied."""

        resolved_mouth = self.mouth if mouth is _UNSET else _coerce_mouth_style(mouth, strict=strict)
        resolved_brows = self.brows if brows is _UNSET else _coerce_brow_style(brows, strict=strict)
        resolved_blink = self.blink if blink is _UNSET else _coerce_optional_bool(blink, field_name="blink", strict=strict)
        resolved_head_dx = self.head_dx if head_dx is _UNSET else _coerce_axis(head_dx, field_name="head_dx", maximum=_MAX_HEAD_AXIS, strict=strict)
        resolved_head_dy = self.head_dy if head_dy is _UNSET else _coerce_axis(head_dy, field_name="head_dy", maximum=_MAX_HEAD_AXIS, strict=strict)
        return DisplayFaceExpression(
            gaze=_coerce_gaze_direction(gaze, strict=strict) or self.gaze,
            mouth=resolved_mouth,
            brows=resolved_brows,
            blink=resolved_blink,
            head_dx=resolved_head_dx,
            head_dy=resolved_head_dy,
        )

    def to_cue(self, *, source: str = "external") -> DisplayFaceCue:
        """Translate the producer-facing expression into one persisted cue."""

        gaze_x, gaze_y = self.gaze.axes()
        return DisplayFaceCue(
            source=_sanitize_source(source),
            gaze_x=gaze_x,
            gaze_y=gaze_y,
            head_dx=self.head_dx,
            head_dy=self.head_dy,
            mouth=self.mouth.value if self.mouth is not None else None,
            brows=self.brows.value if self.brows is not None else None,
            blink=self.blink,
        )

    def signature(self) -> tuple[object, ...]:
        """Return one stable expression signature for testing or dedupe."""

        return (
            self.gaze.value,
            self.mouth.value if self.mouth is not None else None,
            self.brows.value if self.brows is not None else None,
            self.blink,
            self.head_dx,
            self.head_dy,
        )


def _emotion_expression(
    emotion: DisplayFaceEmotion,
    *,
    intensity: float | int | str = 1.0,
    strict: bool,
) -> DisplayFaceExpression:
    """Return one intensity-aware preset expression for a supported emotion."""

    level = _coerce_unit_interval(intensity, field_name="intensity", default=1.0, strict=strict)

    if emotion == DisplayFaceEmotion.HAPPY:
        return DisplayFaceExpression(
            gaze=DisplayFaceGazeDirection.UP_RIGHT if level >= 0.8 else DisplayFaceGazeDirection.CENTER,
            mouth=DisplayFaceMouthStyle.OPEN if level >= 0.9 else DisplayFaceMouthStyle.SMILE,
            brows=DisplayFaceBrowStyle.RAISED if level >= 0.45 else DisplayFaceBrowStyle.SOFT,
            head_dx=1 if level >= 0.85 else 0,
            head_dy=-1 if level >= 0.55 else 0,
        )
    if emotion == DisplayFaceEmotion.SAD:
        return DisplayFaceExpression(
            gaze=DisplayFaceGazeDirection.DOWN,
            mouth=DisplayFaceMouthStyle.SAD,
            brows=DisplayFaceBrowStyle.INWARD_TILT if level >= 0.7 else DisplayFaceBrowStyle.SOFT,
            head_dx=0,
            head_dy=1 if level >= 0.35 else 0,
        )
    if emotion == DisplayFaceEmotion.THOUGHTFUL:
        return DisplayFaceExpression(
            gaze=DisplayFaceGazeDirection.UP_LEFT,
            mouth=DisplayFaceMouthStyle.THINKING,
            brows=DisplayFaceBrowStyle.ROOF,
            head_dx=-1 if level >= 0.3 else 0,
            head_dy=0,
        )
    if emotion == DisplayFaceEmotion.CURIOUS:
        return DisplayFaceExpression(
            gaze=DisplayFaceGazeDirection.UP_RIGHT if level >= 0.75 else DisplayFaceGazeDirection.RIGHT,
            mouth=DisplayFaceMouthStyle.PURSED,
            brows=DisplayFaceBrowStyle.RAISED,
            head_dx=1 if level >= 0.4 else 0,
            head_dy=0,
        )
    if emotion == DisplayFaceEmotion.FOCUSED:
        return DisplayFaceExpression(
            gaze=DisplayFaceGazeDirection.CENTER,
            mouth=DisplayFaceMouthStyle.PURSED if level >= 0.8 else DisplayFaceMouthStyle.NEUTRAL,
            brows=DisplayFaceBrowStyle.INWARD_TILT if level >= 0.4 else DisplayFaceBrowStyle.STRAIGHT,
            head_dx=0,
            head_dy=0,
        )
    if emotion == DisplayFaceEmotion.LISTENING:
        return DisplayFaceExpression(
            gaze=DisplayFaceGazeDirection.CENTER,
            mouth=DisplayFaceMouthStyle.NEUTRAL,
            brows=DisplayFaceBrowStyle.SOFT if level >= 0.2 else DisplayFaceBrowStyle.STRAIGHT,
            blink=False if level >= 0.7 else None,
            head_dx=0,
            head_dy=0,
        )
    if emotion == DisplayFaceEmotion.SPEAKING:
        return DisplayFaceExpression(
            gaze=DisplayFaceGazeDirection.CENTER,
            mouth=DisplayFaceMouthStyle.OPEN if level >= 0.8 else DisplayFaceMouthStyle.SPEAK,
            brows=DisplayFaceBrowStyle.RAISED if level >= 0.65 else DisplayFaceBrowStyle.STRAIGHT,
            head_dx=0,
            head_dy=-1 if level >= 0.85 else 0,
        )
    if emotion == DisplayFaceEmotion.REASSURING:
        return DisplayFaceExpression(
            gaze=DisplayFaceGazeDirection.CENTER,
            mouth=DisplayFaceMouthStyle.SMILE,
            brows=DisplayFaceBrowStyle.SOFT,
            head_dx=0,
            head_dy=-1 if level >= 0.4 else 0,
        )
    if emotion == DisplayFaceEmotion.ALERT:
        return DisplayFaceExpression(
            gaze=DisplayFaceGazeDirection.UP if level >= 0.6 else DisplayFaceGazeDirection.CENTER,
            mouth=DisplayFaceMouthStyle.OPEN if level >= 0.8 else DisplayFaceMouthStyle.NEUTRAL,
            brows=DisplayFaceBrowStyle.RAISED,
            blink=False if level >= 0.5 else None,
            head_dx=0,
            head_dy=0,
        )
    return DisplayFaceExpression(
        gaze=DisplayFaceGazeDirection.CENTER,
        mouth=DisplayFaceMouthStyle.NEUTRAL,
        brows=DisplayFaceBrowStyle.STRAIGHT,
        head_dx=0,
        head_dy=0,
    )


@dataclass(slots=True)
class DisplayFaceExpressionController:
    """Persist producer-facing HDMI face expressions through the cue store."""

    store: DisplayFaceCueStore
    default_source: str = "external"
    # BREAKING: invalid free-form labels now raise by default instead of silently
    # degrading to center/None. Pass strict_inputs=False to retain legacy behavior.
    strict_inputs: bool = True
    max_hold_seconds: float = _DEFAULT_MAX_HOLD_SECONDS

    @classmethod
    def from_config(
        cls,
        config: _DisplayConfigLike,
        *,
        default_source: str = "external",
        strict_inputs: bool = True,
        max_hold_seconds: float = _DEFAULT_MAX_HOLD_SECONDS,
    ) -> "DisplayFaceExpressionController":
        """Resolve the configured cue store and build one controller."""

        return cls(
            store=DisplayFaceCueStore.from_config(config),
            default_source=default_source,
            strict_inputs=strict_inputs,
            max_hold_seconds=max_hold_seconds,
        )

    def show(
        self,
        *,
        gaze: DisplayFaceGazeDirection | str | None = None,
        mouth: DisplayFaceMouthStyle | str | None = None,
        brows: DisplayFaceBrowStyle | str | None = None,
        blink: bool | str | None = None,
        head_dx: int | float | str = 0,
        head_dy: int | float | str = 0,
        emotion: DisplayFaceEmotion | str | None = None,
        intensity: float | int | str = 1.0,
        source: str | None = None,
        hold_seconds: float | int | str | None = None,
        now: datetime | None = None,
        strict: bool | None = None,
    ) -> DisplayFaceCue:
        """Persist one bounded expression cue from high-level input values."""

        expression = self.build_expression(
            gaze=gaze,
            mouth=mouth,
            brows=brows,
            blink=blink,
            head_dx=head_dx,
            head_dy=head_dy,
            emotion=emotion,
            intensity=intensity,
            strict=strict,
        )
        return self.show_expression(
            expression,
            source=source,
            hold_seconds=hold_seconds,
            now=now,
            strict=strict,
        )

    def show_affect(
        self,
        *,
        valence: float | int | str,
        arousal: float | int | str,
        blink: bool | str | None = None,
        head_dx: int | float | str | object = _UNSET,
        head_dy: int | float | str | object = _UNSET,
        source: str | None = None,
        hold_seconds: float | int | str | None = None,
        now: datetime | None = None,
        strict: bool | None = None,
    ) -> DisplayFaceCue:
        """Persist one expression projected from a continuous affect state."""

        effective_strict = self.strict_inputs if strict is None else strict
        expression = DisplayFaceExpression.from_affect(
            valence=valence,
            arousal=arousal,
            blink=blink,
            head_dx=head_dx,
            head_dy=head_dy,
            strict=effective_strict,
        )
        return self.show_expression(
            expression,
            source=source,
            hold_seconds=hold_seconds,
            now=now,
            strict=effective_strict,
        )

    def show_expression(
        self,
        expression: DisplayFaceExpression,
        *,
        source: str | None = None,
        hold_seconds: float | int | str | None = None,
        now: datetime | None = None,
        strict: bool | None = None,
    ) -> DisplayFaceCue:
        """Persist one already-built expression."""

        if not isinstance(expression, DisplayFaceExpression):
            raise TypeError(
                "expression must be a DisplayFaceExpression instance; "
                f"got {type(expression).__name__}."
            )
        effective_strict = self.strict_inputs if strict is None else strict
        return self._save_cue(
            expression.to_cue(source=(source or self.default_source)),
            hold_seconds=hold_seconds,
            now=now,
            strict=effective_strict,
        )

    def preview(
        self,
        *,
        gaze: DisplayFaceGazeDirection | str | None = None,
        mouth: DisplayFaceMouthStyle | str | None = None,
        brows: DisplayFaceBrowStyle | str | None = None,
        blink: bool | str | None = None,
        head_dx: int | float | str = 0,
        head_dy: int | float | str = 0,
        emotion: DisplayFaceEmotion | str | None = None,
        intensity: float | int | str = 1.0,
        source: str | None = None,
        strict: bool | None = None,
    ) -> DisplayFaceCue:
        """Build one normalized cue without persisting it."""

        expression = self.build_expression(
            gaze=gaze,
            mouth=mouth,
            brows=brows,
            blink=blink,
            head_dx=head_dx,
            head_dy=head_dy,
            emotion=emotion,
            intensity=intensity,
            strict=strict,
        )
        return expression.to_cue(source=(source or self.default_source))

    # BREAKING: clear() is now source-scoped by default to avoid deleting another
    # producer's newer cue. Pass force=True for the old global-clear behavior.
    def clear(
        self,
        *,
        source: str | None = None,
        force: bool = False,
        now: datetime | None = None,
    ) -> bool:
        """Remove the active cue owned by this source, or force-clear any cue."""

        effective_now = _normalize_now(now)
        expected_source = _sanitize_source(source or self.default_source)
        path = self.store.path
        with self._artifact_lock():
            if not force:
                current = self.store.load_active(now=effective_now)
                if current is None:
                    return False
                if _sanitize_source(current.source) != expected_source:
                    _LOGGER.debug(
                        "Skip clear for display face cue at %s because current source %r is not owned by %r.",
                        path,
                        current.source,
                        expected_source,
                    )
                    return False
            try:
                path.unlink()
            except FileNotFoundError:
                return False
            return True

    def build_expression(
        self,
        *,
        gaze: DisplayFaceGazeDirection | str | None = None,
        mouth: DisplayFaceMouthStyle | str | None = None,
        brows: DisplayFaceBrowStyle | str | None = None,
        blink: bool | str | None = None,
        head_dx: int | float | str = 0,
        head_dy: int | float | str = 0,
        emotion: DisplayFaceEmotion | str | None = None,
        intensity: float | int | str = 1.0,
        strict: bool | None = None,
    ) -> DisplayFaceExpression:
        """Normalize one call's expression inputs into a stable model."""

        effective_strict = self.strict_inputs if strict is None else strict
        parsed_emotion = _coerce_emotion(emotion, strict=effective_strict)
        if emotion is not None:
            return DisplayFaceExpression.from_emotion(
                parsed_emotion or DisplayFaceEmotion.CALM,
                intensity=intensity,
                gaze=gaze,
                mouth=mouth,
                brows=brows,
                blink=blink,
                head_dx=head_dx,
                head_dy=head_dy,
                strict=effective_strict,
            )
        return DisplayFaceExpression.from_inputs(
            gaze=gaze,
            mouth=mouth,
            brows=brows,
            blink=blink,
            head_dx=head_dx,
            head_dy=head_dy,
            strict=effective_strict,
        )

    @property
    def lock_path(self) -> Path:
        """Return the sidecar lock path used for atomic inter-process updates."""

        return self.store.path.with_name(f"{self.store.path.name}{_DEFAULT_LOCK_SUFFIX}")

    @contextmanager
    def _artifact_lock(self):
        """Provide one inter-process lock around cue mutations on Linux/Pi."""

        lock_path = self.lock_path
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+b") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _save_cue(
        self,
        cue: DisplayFaceCue,
        *,
        hold_seconds: float | int | str | None,
        now: datetime | None,
        strict: bool,
    ) -> DisplayFaceCue:
        """Persist one normalized face cue with lock + secure atomic write."""

        effective_now = _normalize_now(now)
        effective_hold_seconds = _normalize_hold_seconds(
            hold_seconds,
            default_ttl_s=self.store.default_ttl_s,
            maximum_ttl_s=self.max_hold_seconds,
            strict=strict,
        )
        normalized = DisplayFaceCue.from_dict(
            cue.to_dict(),
            fallback_updated_at=effective_now,
            default_ttl_s=effective_hold_seconds,
        )

        path = self.store.path
        payload = json.dumps(
            normalized.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        with self._artifact_lock():
            current = self.store.load_active(now=effective_now)
            if current is not None:
                current_expires = _parse_timestamp(current.expires_at)
                next_expires = _parse_timestamp(normalized.expires_at)
                if (
                    _cue_signature_without_timestamps(current) == _cue_signature_without_timestamps(normalized)
                    and current_expires is not None
                    and next_expires is not None
                    and current_expires >= next_expires
                ):
                    return current
            _atomic_write_bytes(path, payload)
        return normalized
