"""Producer-facing helpers for HDMI face expressions.

Other Twinr modules should not have to write raw JSON cue payloads just to make
the HDMI face look left, raise its brows, or smile briefly. This module keeps a
small combinable expression API on top of the file-backed cue store so runtime
producers can express intent in one call.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from twinr.agent.base_agent.config import TwinrConfig
from twinr.display.face_cues import DisplayFaceCue, DisplayFaceCueStore


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
    """Optional emotion presets that producers can refine with overrides."""

    CALM = "calm"
    HAPPY = "happy"
    SAD = "sad"
    THOUGHTFUL = "thoughtful"
    CURIOUS = "curious"
    FOCUSED = "focused"


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
    "up-right": DisplayFaceGazeDirection.UP_RIGHT,
    "down-right": DisplayFaceGazeDirection.DOWN_RIGHT,
    "down-left": DisplayFaceGazeDirection.DOWN_LEFT,
    "up-left": DisplayFaceGazeDirection.UP_LEFT,
}


def _normalize_label(value: object | None) -> str:
    """Normalize one optional free-form enum label."""

    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _coerce_gaze_direction(value: DisplayFaceGazeDirection | str | None) -> DisplayFaceGazeDirection | None:
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
        return None


def _coerce_brow_style(value: DisplayFaceBrowStyle | str | None) -> DisplayFaceBrowStyle | None:
    """Parse one optional brow style value."""

    if value is None or isinstance(value, DisplayFaceBrowStyle):
        return value
    label = _normalize_label(value)
    if not label:
        return None
    try:
        return DisplayFaceBrowStyle(label)
    except ValueError:
        return None


def _coerce_mouth_style(value: DisplayFaceMouthStyle | str | None) -> DisplayFaceMouthStyle | None:
    """Parse one optional mouth style value."""

    if value is None or isinstance(value, DisplayFaceMouthStyle):
        return value
    label = _normalize_label(value)
    if not label:
        return None
    try:
        return DisplayFaceMouthStyle(label)
    except ValueError:
        return None


def _coerce_emotion(value: DisplayFaceEmotion | str | None) -> DisplayFaceEmotion | None:
    """Parse one optional emotion preset value."""

    if value is None or isinstance(value, DisplayFaceEmotion):
        return value
    label = _normalize_label(value)
    if not label:
        return None
    try:
        return DisplayFaceEmotion(label)
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class DisplayFaceExpression:
    """Describe one producer-facing HDMI face expression."""

    gaze: DisplayFaceGazeDirection = DisplayFaceGazeDirection.CENTER
    mouth: DisplayFaceMouthStyle | None = None
    brows: DisplayFaceBrowStyle | None = None
    blink: bool | None = None
    head_dx: int = 0
    head_dy: int = 0

    @classmethod
    def from_emotion(
        cls,
        emotion: DisplayFaceEmotion | str,
        *,
        gaze: DisplayFaceGazeDirection | str | None = None,
        mouth: DisplayFaceMouthStyle | str | None = None,
        brows: DisplayFaceBrowStyle | str | None = None,
        blink: bool | None = None,
        head_dx: int = 0,
        head_dy: int = 0,
    ) -> "DisplayFaceExpression":
        """Build one expression from a preset emotion plus optional overrides."""

        preset = _emotion_expression(_coerce_emotion(emotion))
        return cls(
            gaze=_coerce_gaze_direction(gaze) or preset.gaze,
            mouth=_coerce_mouth_style(mouth) or preset.mouth,
            brows=_coerce_brow_style(brows) or preset.brows,
            blink=blink if blink is not None else preset.blink,
            head_dx=head_dx if head_dx else preset.head_dx,
            head_dy=head_dy if head_dy else preset.head_dy,
        )

    def to_cue(self, *, source: str = "external") -> DisplayFaceCue:
        """Translate the producer-facing expression into one persisted cue."""

        gaze_x, gaze_y = self.gaze.axes()
        return DisplayFaceCue(
            source=str(source or "external").strip() or "external",
            gaze_x=gaze_x,
            gaze_y=gaze_y,
            head_dx=self.head_dx,
            head_dy=self.head_dy,
            mouth=self.mouth.value if self.mouth is not None else None,
            brows=self.brows.value if self.brows is not None else None,
            blink=self.blink,
        )


def _emotion_expression(emotion: DisplayFaceEmotion | None) -> DisplayFaceExpression:
    """Return one conservative preset expression for an optional emotion."""

    if emotion == DisplayFaceEmotion.HAPPY:
        return DisplayFaceExpression(
            gaze=DisplayFaceGazeDirection.CENTER,
            mouth=DisplayFaceMouthStyle.SMILE,
            brows=DisplayFaceBrowStyle.RAISED,
        )
    if emotion == DisplayFaceEmotion.SAD:
        return DisplayFaceExpression(
            gaze=DisplayFaceGazeDirection.DOWN,
            mouth=DisplayFaceMouthStyle.SAD,
            brows=DisplayFaceBrowStyle.SOFT,
        )
    if emotion == DisplayFaceEmotion.THOUGHTFUL:
        return DisplayFaceExpression(
            gaze=DisplayFaceGazeDirection.UP_LEFT,
            mouth=DisplayFaceMouthStyle.THINKING,
            brows=DisplayFaceBrowStyle.ROOF,
        )
    if emotion == DisplayFaceEmotion.CURIOUS:
        return DisplayFaceExpression(
            gaze=DisplayFaceGazeDirection.RIGHT,
            mouth=DisplayFaceMouthStyle.PURSED,
            brows=DisplayFaceBrowStyle.RAISED,
        )
    if emotion == DisplayFaceEmotion.FOCUSED:
        return DisplayFaceExpression(
            gaze=DisplayFaceGazeDirection.CENTER,
            mouth=DisplayFaceMouthStyle.NEUTRAL,
            brows=DisplayFaceBrowStyle.INWARD_TILT,
        )
    return DisplayFaceExpression(
        gaze=DisplayFaceGazeDirection.CENTER,
        mouth=DisplayFaceMouthStyle.NEUTRAL,
        brows=DisplayFaceBrowStyle.STRAIGHT,
    )


@dataclass(slots=True)
class DisplayFaceExpressionController:
    """Persist producer-facing HDMI face expressions through the cue store."""

    store: DisplayFaceCueStore
    default_source: str = "external"

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        default_source: str = "external",
    ) -> "DisplayFaceExpressionController":
        """Resolve the configured cue store and build one controller."""

        return cls(store=DisplayFaceCueStore.from_config(config), default_source=default_source)

    def show(
        self,
        *,
        gaze: DisplayFaceGazeDirection | str | None = None,
        mouth: DisplayFaceMouthStyle | str | None = None,
        brows: DisplayFaceBrowStyle | str | None = None,
        blink: bool | None = None,
        head_dx: int = 0,
        head_dy: int = 0,
        emotion: DisplayFaceEmotion | str | None = None,
        source: str | None = None,
        hold_seconds: float | None = None,
        now: datetime | None = None,
    ) -> DisplayFaceCue:
        """Persist one bounded expression cue from high-level input values."""

        expression = self._expression_from_inputs(
            gaze=gaze,
            mouth=mouth,
            brows=brows,
            blink=blink,
            head_dx=head_dx,
            head_dy=head_dy,
            emotion=emotion,
        )
        return self.store.save(
            expression.to_cue(source=(source or self.default_source)),
            hold_seconds=hold_seconds,
            now=now,
        )

    def show_expression(
        self,
        expression: DisplayFaceExpression,
        *,
        source: str | None = None,
        hold_seconds: float | None = None,
        now: datetime | None = None,
    ) -> DisplayFaceCue:
        """Persist one already-built expression."""

        return self.store.save(
            expression.to_cue(source=(source or self.default_source)),
            hold_seconds=hold_seconds,
            now=now,
        )

    def clear(self) -> None:
        """Remove the currently active external expression cue."""

        self.store.clear()

    def _expression_from_inputs(
        self,
        *,
        gaze: DisplayFaceGazeDirection | str | None,
        mouth: DisplayFaceMouthStyle | str | None,
        brows: DisplayFaceBrowStyle | str | None,
        blink: bool | None,
        head_dx: int,
        head_dy: int,
        emotion: DisplayFaceEmotion | str | None,
    ) -> DisplayFaceExpression:
        """Normalize one call's expression inputs into a stable model."""

        parsed_emotion = _coerce_emotion(emotion)
        if parsed_emotion is not None:
            return DisplayFaceExpression.from_emotion(
                parsed_emotion,
                gaze=gaze,
                mouth=mouth,
                brows=brows,
                blink=blink,
                head_dx=head_dx,
                head_dy=head_dy,
            )
        return DisplayFaceExpression(
            gaze=_coerce_gaze_direction(gaze) or DisplayFaceGazeDirection.CENTER,
            mouth=_coerce_mouth_style(mouth),
            brows=_coerce_brow_style(brows),
            blink=blink,
            head_dx=head_dx,
            head_dy=head_dy,
        )
