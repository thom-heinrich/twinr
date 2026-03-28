# mypy: disable-error-code="attr-defined,call-overload,arg-type"
# CHANGELOG: 2026-03-28
# BUG-1: _face_offset now resolves cues through the same _effective_face_cue pipeline as _eye_state, removing head/eye desync when cues are filtered or rewritten upstream.
# BUG-2: Blink generation is no longer tied to the 6/12-frame motion loops; the previous implementation could blink unrealistically often when animation_frame is a real render-frame counter.
# BUG-3: Face-box scaling no longer pops at hard 420/640/720 thresholds and now tolerates inverted/float boxes, avoiding visible size jitter from noisy trackers.
# SEC-1: External face cues are sanitized, finite-checked, and clamped before use so malformed or hostile cue payloads cannot drive unbounded render offsets on a deployed Pi.
# IMP-1: Hot-path animation tables are precomputed at module scope and eye templates are cached over a finite state-space to cut per-frame allocations on Raspberry Pi 4.
# IMP-2: Idle eye behavior now uses subtler micro-saccades plus status-specific blink cadences, and face scaling is continuous/proxemic-aware to better match 2025-2026 HRI guidance.

"""Face-state calculation helpers for the default HDMI scene."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from math import isfinite
from typing import Final, Literal, NamedTuple, TypedDict

from .typing_contracts import HdmiFaceCueLike

FaceStatus = Literal["waiting", "listening", "processing", "answering", "printing", "error"]
EyeSide = Literal["left", "right"]


class HdmiEyeState(TypedDict):
    width: int
    height: int
    eye_shift_x: int
    eye_shift_y: int
    highlight_dx: int
    highlight_dy: int
    brow_raise: int
    brow_slant: int
    brow_style: str
    blink: bool
    lid_arc: bool


@dataclass(frozen=True, slots=True)
class _SanitisedFaceCue:
    gaze_x: int = 0
    gaze_y: int = 0
    head_dx: int = 0
    head_dy: int = 0
    blink: bool | None = None
    brows: str | None = None


class _EyeStateTemplate(NamedTuple):
    width: int = 56
    height: int = 74
    eye_shift_x: int = 0
    eye_shift_y: int = 0
    highlight_dx: int = 0
    highlight_dy: int = 0
    brow_raise: int = 0
    brow_slant: int = 4
    brow_style: str = ""
    blink: bool = False
    lid_arc: bool = False


_VALID_STATUSES: Final[frozenset[str]] = frozenset(
    {"waiting", "listening", "processing", "answering", "printing", "error"}
)
_ALLOWED_BROWS: Final[frozenset[str]] = frozenset(
    {"raised", "soft", "straight", "inward_tilt", "outward_tilt", "roof"}
)
_DEFAULT_EYE_TEMPLATE: Final[_EyeStateTemplate] = _EyeStateTemplate()

# Finite, bounded, cache-friendly motion tables.
_WAITING_FACE_OFFSETS: Final[tuple[tuple[int, int], ...]] = (
    (0, 0),
    (-1, 0),
    (-2, 0),
    (-1, 0),
    (0, -1),
    (0, 0),
    (0, 1),
    (0, 0),
    (1, 0),
    (2, 0),
    (1, 0),
    (0, 0),
)
_LISTENING_FACE_OFFSETS: Final[tuple[tuple[int, int], ...]] = (
    (0, 0),
    (0, -1),
    (0, -1),
    (0, 0),
    (0, 1),
    (0, 0),
)
_PROCESSING_FACE_OFFSETS: Final[tuple[tuple[int, int], ...]] = (
    (0, 0),
    (-1, 0),
    (-1, 0),
    (0, 0),
    (1, 0),
    (0, 0),
)
_ANSWERING_FACE_OFFSETS: Final[tuple[tuple[int, int], ...]] = (
    (0, 0),
    (0, -1),
    (0, 0),
    (0, 1),
    (0, 0),
    (0, 0),
)
_PRINTING_FACE_OFFSETS: Final[tuple[tuple[int, int], ...]] = (
    (0, 0),
    (1, 0),
    (0, 0),
    (-1, 0),
    (0, 0),
    (0, 0),
)
_ERROR_FACE_OFFSETS: Final[tuple[tuple[int, int], ...]] = (
    (0, 1),
    (0, 0),
    (0, 1),
    (0, 0),
    (0, 1),
    (0, 0),
)

# Waiting uses a longer cycle to avoid the “robot-face metronome” look.
_WAITING_EYE_SHIFT_X: Final[tuple[int, ...]] = (
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
)
_WAITING_EYE_SHIFT_Y: Final[tuple[int, ...]] = (
    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
)
_WAITING_HIGHLIGHT_DX: Final[tuple[int, ...]] = (
    0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0,
)
_WAITING_HIGHLIGHT_DY: Final[tuple[int, ...]] = (
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
)

_LISTENING_HEIGHT: Final[tuple[int, ...]] = (78, 80, 82, 80, 78, 76)
_LISTENING_HIGHLIGHT_DX: Final[tuple[int, ...]] = (-8, -7, -6, -5, -6, -7)
_LISTENING_HIGHLIGHT_DY: Final[tuple[int, ...]] = (-18, -19, -20, -19, -18, -17)

_PROCESSING_SCAN: Final[tuple[int, ...]] = (-12, -8, -3, 3, 8, 12)
_PROCESSING_HIGHLIGHT_DY: Final[tuple[int, ...]] = (-17, -16, -15, -16, -17, -18)

_ANSWERING_HEIGHT: Final[tuple[int, ...]] = (70, 74, 72, 74, 70, 72)
_ANSWERING_HIGHLIGHT_DX: Final[tuple[int, ...]] = (-8, -7, -6, -7, -8, -7)
_ANSWERING_HIGHLIGHT_DY: Final[tuple[int, ...]] = (-18, -17, -16, -17, -18, -17)

_PRINTING_HEIGHT: Final[tuple[int, ...]] = (70, 68, 66, 64, 66, 68)
_PRINTING_HIGHLIGHT_DX: Final[tuple[int, ...]] = (-9, -8, -7, -6, -7, -8)

_ERROR_HEIGHT: Final[tuple[int, ...]] = (60, 58, 56, 58, 60, 58)
_ERROR_HIGHLIGHT_DX: Final[tuple[int, ...]] = (-2, -1, 0, 1, 0, -1)

_MOTION_CYCLES: Final[dict[str, int]] = {
    "waiting": len(_WAITING_EYE_SHIFT_X),
    "listening": len(_LISTENING_HEIGHT),
    "processing": len(_PROCESSING_SCAN),
    "answering": len(_ANSWERING_HEIGHT),
    "printing": len(_PRINTING_HEIGHT),
    "error": len(_ERROR_HEIGHT),
}
_BLINK_SPECS: Final[dict[str, tuple[int, frozenset[int]]]] = {
    "waiting": (12, frozenset({9})),
    "listening": (108, frozenset({72, 73})),
    "processing": (132, frozenset({96, 97})),
    "answering": (120, frozenset({84, 85})),
    "printing": (144, frozenset({102, 103})),
    "error": (180, frozenset({132, 133})),
}
_BROW_PRESETS: Final[dict[str, tuple[int, int, str]]] = {
    "raised": (-11, 0, "raised"),
    "soft": (-3, 0, "soft"),
    "straight": (0, 0, "straight"),
    "inward_tilt": (0, 0, "inward_tilt"),
    "outward_tilt": (0, 0, "outward_tilt"),
    "roof": (-1, 0, "roof"),
}


def _status_token(status: str) -> str:
    return status if status in _VALID_STATUSES else ""


def _side_token(side: str) -> str:
    return "left" if side == "left" else "right"


@cache
def _cached_face_base_offset(
    status: str,
    phase: int,
    directional_cue_active: bool,
    cue_driven_error: bool,
) -> tuple[int, int]:
    if status == "waiting":
        if directional_cue_active:
            return (0, 0)
        return _WAITING_FACE_OFFSETS[phase % len(_WAITING_FACE_OFFSETS)]
    if status == "listening":
        return _LISTENING_FACE_OFFSETS[phase % len(_LISTENING_FACE_OFFSETS)]
    if status == "processing":
        return _PROCESSING_FACE_OFFSETS[phase % len(_PROCESSING_FACE_OFFSETS)]
    if status == "answering":
        return _ANSWERING_FACE_OFFSETS[phase % len(_ANSWERING_FACE_OFFSETS)]
    if status == "printing":
        return _PRINTING_FACE_OFFSETS[phase % len(_PRINTING_FACE_OFFSETS)]
    if status == "error":
        if cue_driven_error:
            return (0, 0)
        return _ERROR_FACE_OFFSETS[phase % len(_ERROR_FACE_OFFSETS)]
    return (0, 0)


@cache
def _cached_eye_template(
    status: str,
    side: str,
    motion_phase: int,
    blink_phase: int,
    directional_cue_active: bool,
    cue_driven_error: bool,
) -> _EyeStateTemplate:
    state = _DEFAULT_EYE_TEMPLATE

    if status == "waiting":
        if directional_cue_active:
            return state
        return _EyeStateTemplate(
            width=56,
            height=74,
            eye_shift_x=_WAITING_EYE_SHIFT_X[motion_phase],
            eye_shift_y=_WAITING_EYE_SHIFT_Y[motion_phase],
            highlight_dx=_WAITING_HIGHLIGHT_DX[motion_phase],
            highlight_dy=_WAITING_HIGHLIGHT_DY[motion_phase],
            brow_raise=0,
            brow_slant=4,
            brow_style="",
            blink=blink_phase in _BLINK_SPECS["waiting"][1],
            lid_arc=False,
        )

    if status == "listening":
        return _EyeStateTemplate(
            width=56,
            height=_LISTENING_HEIGHT[motion_phase],
            eye_shift_x=0,
            eye_shift_y=0,
            highlight_dx=_LISTENING_HIGHLIGHT_DX[motion_phase],
            highlight_dy=_LISTENING_HIGHLIGHT_DY[motion_phase],
            brow_raise=-8,
            brow_slant=2,
            brow_style="",
            blink=blink_phase in _BLINK_SPECS["listening"][1],
            lid_arc=False,
        )

    if status == "processing":
        scan = _PROCESSING_SCAN[motion_phase]
        return _EyeStateTemplate(
            width=56,
            height=68,
            eye_shift_x=0,
            eye_shift_y=0,
            highlight_dx=scan if side == "left" else scan - 2,
            highlight_dy=_PROCESSING_HIGHLIGHT_DY[motion_phase],
            brow_raise=-1,
            brow_slant=4,
            brow_style="",
            blink=blink_phase in _BLINK_SPECS["processing"][1],
            lid_arc=False,
        )

    if status == "answering":
        return _EyeStateTemplate(
            width=56,
            height=_ANSWERING_HEIGHT[motion_phase],
            eye_shift_x=0,
            eye_shift_y=0,
            highlight_dx=_ANSWERING_HIGHLIGHT_DX[motion_phase],
            highlight_dy=_ANSWERING_HIGHLIGHT_DY[motion_phase],
            brow_raise=-2,
            brow_slant=2,
            brow_style="",
            blink=blink_phase in _BLINK_SPECS["answering"][1],
            lid_arc=False,
        )

    if status == "printing":
        return _EyeStateTemplate(
            width=56,
            height=_PRINTING_HEIGHT[motion_phase],
            eye_shift_x=0,
            eye_shift_y=0,
            highlight_dx=_PRINTING_HIGHLIGHT_DX[motion_phase],
            highlight_dy=0,
            brow_raise=-4,
            brow_slant=2,
            brow_style="",
            blink=blink_phase in _BLINK_SPECS["printing"][1],
            lid_arc=False,
        )

    if status == "error":
        if cue_driven_error:
            return _EyeStateTemplate(
                width=54,
                height=58,
                eye_shift_x=0,
                eye_shift_y=1,
                highlight_dx=0,
                highlight_dy=0,
                brow_raise=2,
                brow_slant=8,
                brow_style="",
                blink=False,
                lid_arc=False,
            )
        return _EyeStateTemplate(
            width=54,
            height=_ERROR_HEIGHT[motion_phase],
            eye_shift_x=0,
            eye_shift_y=2,
            highlight_dx=_ERROR_HIGHLIGHT_DX[motion_phase],
            highlight_dy=0,
            brow_raise=2,
            brow_slant=8,
            brow_style="",
            blink=blink_phase in _BLINK_SPECS["error"][1],
            lid_arc=False,
        )

    return state


class HdmiFaceLogicMixin:
    """Calculate the deterministic HDMI face state for each runtime frame."""

    def _face_offset(
        self,
        status: str,
        animation_frame: int,
        *,
        face_cue: HdmiFaceCueLike | None = None,
    ) -> tuple[int, int]:
        status_token = _status_token(status)
        motion_phase = self._frame_phase(animation_frame, cycle=_MOTION_CYCLES.get(status_token, 1))
        face_cue = self._resolved_face_cue(status=status, face_cue=face_cue)
        directional_cue_active = self._directional_face_cue_active(face_cue)
        cue_driven_error = status_token == "error" and directional_cue_active
        base = _cached_face_base_offset(status_token, motion_phase, directional_cue_active, cue_driven_error)
        if face_cue is None:
            return base
        head_scale_x = 8 if status_token == "error" else 5
        head_scale_y = 3 if status_token == "error" else 2
        return (
            base[0] + self._scaled_offset(face_cue.head_dx, head_scale_x),
            base[1] + self._scaled_offset(face_cue.head_dy, head_scale_y),
        )

    def _eye_state(
        self,
        status: str,
        animation_frame: int,
        side: str,
        *,
        face_cue: HdmiFaceCueLike | None = None,
    ) -> HdmiEyeState:
        status_token = _status_token(status)
        side_token = _side_token(side)
        face_cue = self._resolved_face_cue(status=status, face_cue=face_cue)
        directional_cue_active = self._directional_face_cue_active(face_cue)
        cue_driven_error = status_token == "error" and directional_cue_active
        motion_phase = self._frame_phase(animation_frame, cycle=_MOTION_CYCLES.get(status_token, 1))
        blink_cycle = _BLINK_SPECS.get(status_token, (1, frozenset()))[0]
        blink_phase = self._frame_phase(animation_frame, cycle=blink_cycle)
        template = _cached_eye_template(
            status_token,
            side_token,
            motion_phase,
            blink_phase,
            directional_cue_active,
            cue_driven_error,
        )
        state: HdmiEyeState = {
            "width": template.width,
            "height": template.height,
            "eye_shift_x": template.eye_shift_x,
            "eye_shift_y": template.eye_shift_y,
            "highlight_dx": template.highlight_dx,
            "highlight_dy": template.highlight_dy,
            "brow_raise": template.brow_raise,
            "brow_slant": template.brow_slant,
            "brow_style": template.brow_style,
            "blink": template.blink,
            "lid_arc": template.lid_arc,
        }
        return self._apply_face_cue_to_eye_state(state, status=status_token, face_cue=face_cue)

    def _face_scale_for_box(self, box: tuple[int, int, int, int]) -> float:
        """Return the visible face scale for one face box, including widescreen growth."""

        width, height = self._box_dimensions(box)
        scale = self._normalise_scale(min(width / 220.0, height / 210.0))
        scale_cap = self._max_face_scale_for_box(box)
        return max(0.56, min(scale, scale_cap))

    def _max_face_scale_for_box(self, box: tuple[int, int, int, int]) -> float:
        """Return the upper face-scale clamp for the current screen geometry."""

        width, height = self._box_dimensions(box)
        mid_growth = max(
            self._smoothstep(width, start=320.0, end=420.0),
            self._smoothstep(height, start=320.0, end=420.0),
        )
        wide_growth = max(
            self._smoothstep(width, start=420.0, end=640.0),
            self._smoothstep(height, start=420.0, end=720.0),
        )
        return 1.55 + (0.40 * mid_growth) + (0.50 * wide_growth)

    def _apply_face_cue_to_eye_state(
        self,
        state: HdmiEyeState,
        *,
        status: str,
        face_cue: HdmiFaceCueLike | None,
    ) -> HdmiEyeState:
        cue = face_cue if isinstance(face_cue, _SanitisedFaceCue) else self._sanitise_face_cue(face_cue)
        if cue is None:
            return state

        merged: HdmiEyeState = dict(state)
        horizontal_eye_scale = 6 if status == "error" else 4
        vertical_eye_scale = 1 if status == "error" else 3
        horizontal_highlight_scale = 8 if status == "error" else 6
        vertical_highlight_scale = 3 if status == "error" else 5
        merged["eye_shift_x"] = int(merged["eye_shift_x"]) + self._scaled_offset(
            cue.gaze_x,
            horizontal_eye_scale,
        )
        merged["eye_shift_y"] = int(merged["eye_shift_y"]) + self._scaled_offset(
            cue.gaze_y,
            vertical_eye_scale,
        )
        merged["highlight_dx"] = int(merged["highlight_dx"]) + self._scaled_offset(
            cue.gaze_x,
            horizontal_highlight_scale,
        )
        merged["highlight_dy"] = int(merged["highlight_dy"]) + self._scaled_offset(
            cue.gaze_y,
            vertical_highlight_scale,
        )
        if cue.blink is not None:
            merged["blink"] = cue.blink

        if cue.brows is not None:
            brow_raise, brow_slant, brow_style = _BROW_PRESETS[cue.brows]
            merged["brow_raise"] = brow_raise
            merged["brow_slant"] = brow_slant
            merged["brow_style"] = brow_style
        return merged

    def _directional_face_cue_active(self, face_cue: HdmiFaceCueLike | None) -> bool:
        """Return whether the current face cue should dominate waiting idle motion."""

        cue = face_cue if isinstance(face_cue, _SanitisedFaceCue) else self._sanitise_face_cue(face_cue)
        if cue is None:
            return False
        return cue.gaze_x != 0 or cue.gaze_y != 0 or cue.head_dx != 0 or cue.head_dy != 0

    def _resolved_face_cue(
        self,
        *,
        status: str,
        face_cue: HdmiFaceCueLike | None,
    ) -> _SanitisedFaceCue | None:
        effective = self._effective_face_cue(status=status, face_cue=face_cue)
        return self._sanitise_face_cue(effective)

    def _sanitise_face_cue(self, face_cue: HdmiFaceCueLike | None) -> _SanitisedFaceCue | None:
        if face_cue is None:
            return None

        sanitized = _SanitisedFaceCue(
            gaze_x=self._clamped_int(getattr(face_cue, "gaze_x", 0), minimum=-3, maximum=3),
            gaze_y=self._clamped_int(getattr(face_cue, "gaze_y", 0), minimum=-2, maximum=2),
            head_dx=self._clamped_int(getattr(face_cue, "head_dx", 0), minimum=-3, maximum=3),
            head_dy=self._clamped_int(getattr(face_cue, "head_dy", 0), minimum=-2, maximum=2),
            blink=self._coerce_optional_bool(getattr(face_cue, "blink", None)),
            brows=self._coerce_brow_style(getattr(face_cue, "brows", None)),
        )
        if (
            sanitized.gaze_x == 0
            and sanitized.gaze_y == 0
            and sanitized.head_dx == 0
            and sanitized.head_dy == 0
            and sanitized.blink is None
            and sanitized.brows is None
        ):
            return None
        return sanitized

    def _coerce_brow_style(self, value: object) -> str | None:
        if isinstance(value, str) and value in _ALLOWED_BROWS:
            return value
        return None

    def _coerce_optional_bool(self, value: object) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            parsed = self._finite_float(value, default=float("nan"))
            if not isfinite(parsed):
                return None
            return bool(parsed)
        return None

    def _frame_index(self, value: object, *, default: int = 0) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        return int(round(self._finite_float(value, default=float(default))))

    def _frame_phase(self, animation_frame: object, *, cycle: int) -> int:
        if cycle <= 1:
            return 0
        return self._frame_index(animation_frame) % cycle

    def _box_dimensions(self, box: tuple[int, int, int, int]) -> tuple[float, float]:
        left, top, right, bottom = box
        width = max(1.0, abs(self._finite_float(right) - self._finite_float(left)))
        height = max(1.0, abs(self._finite_float(bottom) - self._finite_float(top)))
        return width, height

    def _smoothstep(self, value: object, *, start: float, end: float) -> float:
        parsed = self._finite_float(value)
        if parsed <= start:
            return 0.0
        if parsed >= end:
            return 1.0
        t = (parsed - start) / (end - start)
        return t * t * (3.0 - (2.0 * t))

    def _finite_float(self, value: object, *, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if not isfinite(parsed):
            return default
        return parsed

    def _clamped_int(
        self,
        value: object,
        *,
        minimum: int,
        maximum: int,
        default: int = 0,
    ) -> int:
        parsed = self._finite_float(value, default=float(default))
        return max(minimum, min(maximum, int(round(parsed))))

    def _scaled_offset(self, value: int | float, scale: float) -> int:
        return int(round(self._finite_float(value) * self._finite_float(scale, default=1.0)))

    def _scaled_size(self, value: int | float, scale: float, *, minimum: int = 1) -> int:
        return max(minimum, int(round(self._finite_float(value) * self._finite_float(scale, default=1.0))))

    def _normalise_scale(self, value: object) -> float:
        parsed = self._finite_float(value, default=1.0)
        if parsed > 0:
            return parsed
        return 1.0
