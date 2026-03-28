# mypy: disable-error-code="attr-defined,call-overload,arg-type"
# CHANGELOG: 2026-03-28
# BUG-1: Guard external eye-state payloads against missing keys / non-numeric values that could crash HDMI rendering.
# BUG-2: Fix silent misrendering from truthy string booleans (e.g. "False") in blink/lid flags.
# SEC-1: Clamp untrusted geometry and animation inputs to prevent render-path CPU blowups on Raspberry Pi deployments.
# IMP-1: Add a normalized render contract (status, mouth style, eye state) so expression control stays modular and resilient.
# IMP-2: Add an optional aggdraw backend for anti-aliased arcs/lines/rounded rectangles while preserving Pillow drop-in compatibility.
"""Face drawing helpers for the default HDMI scene renderer."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from .typing_contracts import HdmiCanvasDrawSurface, HdmiFaceCueLike

try:  # Optional 2026-quality backend for anti-aliased vector primitives on PIL images.
    import aggdraw as _aggdraw
except Exception:  # pragma: no cover - optional dependency
    _aggdraw = None


_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)

_VALID_STATUSES = {
    "waiting",
    "listening",
    "processing",
    "answering",
    "printing",
    "error",
}
# BREAKING: Alias statuses such as "speaking" and "thinking" now map to
# canonical render states instead of falling through to the default fallback mouth.
_STATUS_ALIASES = {
    "idle": "waiting",
    "ready": "waiting",
    "pending": "waiting",
    "hearing": "listening",
    "thinking": "processing",
    "working": "processing",
    "speaking": "answering",
    "talking": "answering",
    "responding": "answering",
    "output": "printing",
    "failed": "error",
    "fault": "error",
    "alert": "error",
}

_VALID_MOUTH_STYLES = {
    "neutral",
    "smile",
    "open",
    "pursed",
    "speak",
    "sad",
    "thinking",
    "scrunched",
}
_MOUTH_STYLE_ALIASES = {
    "closed": "neutral",
    "flat": "neutral",
    "happy": "smile",
    "talk": "speak",
    "speaking": "speak",
    "frown": "sad",
    "think": "thinking",
    "squint": "scrunched",
}

_VALID_BROW_STYLES = {"", "soft", "roof", "inward_tilt", "outward_tilt"}

_TRUE_STRINGS = {"1", "true", "yes", "on", "y", "t"}
_FALSE_STRINGS = {"0", "false", "no", "off", "n", "f", ""}

# Canonical face extents in the unscaled design space. Used to bound scale and
# prevent oversized geometry from bleeding across the scene or burning CPU.
_FACE_HALF_WIDTH_UNITS = 112
_FACE_TOP_UNITS = 112
_FACE_BOTTOM_UNITS = 88

_COORD_LIMIT = 8_192
_FRAME_LIMIT = 1_000_000
_JITTER_LIMIT = 24


@dataclass(frozen=True, slots=True)
class _FaceEyeState:
    blink: bool = False
    width: int = 52
    height: int = 36
    eye_shift_x: int = 0
    eye_shift_y: int = 0
    lid_arc: bool = False
    brow_raise: int = 0
    brow_slant: int = 0
    brow_style: str = ""
    highlight_dx: int = -6
    highlight_dy: int = -6


class _PillowSurfaceOps:
    def __init__(self, draw: object) -> None:
        self._draw = draw

    def line(self, xy: tuple[int, ...] | tuple[int, int, int, int], *, fill: tuple[int, int, int], width: int) -> None:
        self._draw.line(xy, fill=fill, width=width)

    def arc(
        self,
        xy: tuple[int, int, int, int],
        *,
        start: int,
        end: int,
        fill: tuple[int, int, int],
        width: int,
    ) -> None:
        self._draw.arc(xy, start=start, end=end, fill=fill, width=width)

    def ellipse(
        self,
        xy: tuple[int, int, int, int],
        *,
        fill: tuple[int, int, int] | None = None,
        outline: tuple[int, int, int] | None = None,
        width: int = 1,
    ) -> None:
        kwargs: dict[str, object] = {}
        if fill is not None:
            kwargs["fill"] = fill
        if outline is not None:
            kwargs["outline"] = outline
            kwargs["width"] = width
        self._draw.ellipse(xy, **kwargs)

    def rounded_rectangle(
        self,
        xy: tuple[int, int, int, int],
        *,
        radius: int,
        fill: tuple[int, int, int] | None = None,
        outline: tuple[int, int, int] | None = None,
        width: int = 1,
    ) -> None:
        kwargs: dict[str, object] = {"radius": radius}
        if fill is not None:
            kwargs["fill"] = fill
        if outline is not None:
            kwargs["outline"] = outline
            kwargs["width"] = width
        self._draw.rounded_rectangle(xy, **kwargs)

    def flush(self) -> None:
        return None


class _AggdrawSurfaceOps:
    def __init__(self, surface: Any) -> None:
        self._surface = surface
        self._pen_cache: dict[tuple[tuple[int, int, int], int], Any] = {}
        self._brush_cache: dict[tuple[int, int, int], Any] = {}
        try:
            self._surface.setantialias(True)
        except Exception:
            pass

    def _pen(self, color: tuple[int, int, int], width: int) -> Any:
        key = (color, max(1, int(width)))
        pen = self._pen_cache.get(key)
        if pen is None:
            pen = _aggdraw.Pen(color, key[1])
            self._pen_cache[key] = pen
        return pen

    def _brush(self, color: tuple[int, int, int]) -> Any:
        brush = self._brush_cache.get(color)
        if brush is None:
            brush = _aggdraw.Brush(color)
            self._brush_cache[color] = brush
        return brush

    def line(self, xy: tuple[int, ...] | tuple[int, int, int, int], *, fill: tuple[int, int, int], width: int) -> None:
        self._surface.line(xy, self._pen(fill, width))

    def arc(
        self,
        xy: tuple[int, int, int, int],
        *,
        start: int,
        end: int,
        fill: tuple[int, int, int],
        width: int,
    ) -> None:
        self._surface.arc(xy, start, end, self._pen(fill, width))

    def ellipse(
        self,
        xy: tuple[int, int, int, int],
        *,
        fill: tuple[int, int, int] | None = None,
        outline: tuple[int, int, int] | None = None,
        width: int = 1,
    ) -> None:
        pen = self._pen(outline, width) if outline is not None else None
        brush = self._brush(fill) if fill is not None else None
        self._surface.ellipse(xy, pen=pen, brush=brush)

    def rounded_rectangle(
        self,
        xy: tuple[int, int, int, int],
        *,
        radius: int,
        fill: tuple[int, int, int] | None = None,
        outline: tuple[int, int, int] | None = None,
        width: int = 1,
    ) -> None:
        pen = self._pen(outline, width) if outline is not None else None
        brush = self._brush(fill) if fill is not None else None
        self._surface.rounded_rectangle(xy, radius, pen=pen, brush=brush)

    def flush(self) -> None:
        self._surface.flush()


class HdmiFaceRenderingMixin:
    """Draw the HDMI face from deterministic eye, brow, and mouth state."""

    def _draw_face(
        self,
        draw: HdmiCanvasDrawSurface,
        *,
        box: tuple[int, int, int, int],
        status: str,
        animation_frame: int,
        face_cue: HdmiFaceCueLike | None,
    ) -> None:
        safe_box = self._normalise_box(box)
        if safe_box is None:
            return
        surface = self._surface_ops(draw)
        safe_status = self._normalise_status(status)
        safe_frame = self._coerce_int(animation_frame, default=0, minimum=-_FRAME_LIMIT, maximum=_FRAME_LIMIT)
        try:
            scale = self._bounded_face_scale(safe_box, self._face_scale_for_box(safe_box))
            left, top, right, bottom = safe_box
            center_x = (left + right) // 2
            center_y = top + ((bottom - top) // 2) - self._scaled_offset(6, scale)
            self._draw_face_features(
                surface,
                center_x=center_x,
                center_y=center_y,
                status=safe_status,
                animation_frame=safe_frame,
                scale=scale,
                face_cue=face_cue,
            )
        finally:
            surface.flush()

    def _draw_face_features(
        self,
        draw: object,
        *,
        center_x: int,
        center_y: int,
        status: str,
        animation_frame: int,
        scale: float,
        face_cue: HdmiFaceCueLike | None,
    ) -> None:
        safe_scale = self._normalise_scale(scale)
        jitter_x, jitter_y = self._safe_face_offset(status, animation_frame, face_cue)
        jitter_x = self._scaled_offset(jitter_x, safe_scale)
        jitter_y = self._scaled_offset(jitter_y, safe_scale)

        left_eye = (
            center_x - self._scaled_offset(72, safe_scale) + jitter_x,
            center_y - self._scaled_offset(24, safe_scale) + jitter_y,
        )
        right_eye = (
            center_x + self._scaled_offset(72, safe_scale) + jitter_x,
            center_y - self._scaled_offset(24, safe_scale) + jitter_y,
        )
        self._draw_face_eye(
            draw,
            left_eye,
            status=status,
            side="left",
            animation_frame=animation_frame,
            scale=safe_scale,
            face_cue=face_cue,
        )
        self._draw_face_eye(
            draw,
            right_eye,
            status=status,
            side="right",
            animation_frame=animation_frame,
            scale=safe_scale,
            face_cue=face_cue,
        )
        self._draw_face_mouth(
            draw,
            center_x=center_x + jitter_x,
            center_y=center_y + self._scaled_offset(56, safe_scale) + jitter_y,
            status=status,
            animation_frame=animation_frame,
            scale=safe_scale,
            face_cue=face_cue,
        )

    def _draw_face_eye(
        self,
        draw: object,
        origin: tuple[int, int],
        *,
        status: str,
        side: str,
        animation_frame: int,
        scale: float,
        face_cue: HdmiFaceCueLike | None,
    ) -> None:
        center_x, center_y = origin
        eye = self._safe_eye_state(status, animation_frame, side, face_cue)
        line_width = self._scaled_size(4, scale, minimum=2)
        self._draw_face_brow(
            draw,
            center_x=center_x,
            center_y=center_y,
            scale=scale,
            side=side,
            eye=eye,
            line_width=line_width,
        )

        if eye.blink:
            draw.arc(
                (
                    center_x - self._scaled_offset(26, scale),
                    center_y - self._scaled_offset(8, scale),
                    center_x + self._scaled_offset(26, scale),
                    center_y + self._scaled_offset(10, scale),
                ),
                start=200,
                end=340,
                fill=_WHITE,
                width=self._scaled_size(5, scale, minimum=2),
            )
            return

        width = self._scaled_size(eye.width, scale, minimum=8)
        height = self._scaled_size(eye.height, scale, minimum=8)
        offset_x = self._scaled_offset(eye.eye_shift_x, scale)
        offset_y = self._scaled_offset(eye.eye_shift_y, scale)
        box = (
            center_x - (width // 2) + offset_x,
            center_y - (height // 2) + offset_y,
            center_x + (width // 2) + offset_x,
            center_y + (height // 2) + offset_y,
        )
        draw.ellipse(box, fill=_WHITE)
        self._draw_face_eye_highlights(draw, box, eye, scale=scale)

        if eye.lid_arc:
            draw.arc(
                (
                    box[0] + self._scaled_offset(4, scale),
                    box[1] - self._scaled_offset(10, scale),
                    box[2] - self._scaled_offset(4, scale),
                    box[1] + self._scaled_offset(18, scale),
                ),
                start=180,
                end=360,
                fill=_WHITE,
                width=self._scaled_size(3, scale, minimum=2),
            )

    def _draw_face_brow(
        self,
        draw: object,
        *,
        center_x: int,
        center_y: int,
        scale: float,
        side: str,
        eye: _FaceEyeState,
        line_width: int,
    ) -> None:
        """Draw one stable eyebrow using either the legacy or external style path."""

        brow_y = center_y - self._scaled_offset(52, scale) + self._scaled_offset(eye.brow_raise, scale)
        half_width = self._scaled_offset(24, scale)
        left_x = center_x - half_width
        right_x = center_x + half_width
        style = eye.brow_style
        if not style:
            slant = self._scaled_offset(eye.brow_slant, scale)
            if side == "left":
                draw.line(
                    (left_x, brow_y + slant, right_x, brow_y - slant),
                    fill=_WHITE,
                    width=line_width,
                )
            else:
                draw.line(
                    (left_x, brow_y - slant, right_x, brow_y + slant),
                    fill=_WHITE,
                    width=line_width,
                )
            return

        if style == "soft":
            draw.arc(
                (
                    left_x,
                    brow_y - self._scaled_offset(10, scale),
                    right_x,
                    brow_y + self._scaled_offset(10, scale),
                ),
                start=190,
                end=350,
                fill=_WHITE,
                width=line_width,
            )
            return

        if style == "roof":
            peak_y = brow_y - self._scaled_offset(10, scale)
            draw.line(
                (
                    left_x,
                    brow_y + self._scaled_offset(4, scale),
                    center_x,
                    peak_y,
                    right_x,
                    brow_y + self._scaled_offset(4, scale),
                ),
                fill=_WHITE,
                width=line_width,
            )
            return

        if style == "inward_tilt":
            rise = self._scaled_offset(7, scale)
            if side == "left":
                draw.line((left_x, brow_y - rise, right_x, brow_y + rise), fill=_WHITE, width=line_width)
            else:
                draw.line((left_x, brow_y + rise, right_x, brow_y - rise), fill=_WHITE, width=line_width)
            return

        if style == "outward_tilt":
            rise = self._scaled_offset(7, scale)
            if side == "left":
                draw.line((left_x, brow_y + rise, right_x, brow_y - rise), fill=_WHITE, width=line_width)
            else:
                draw.line((left_x, brow_y - rise, right_x, brow_y + rise), fill=_WHITE, width=line_width)
            return

        draw.line((left_x, brow_y, right_x, brow_y), fill=_WHITE, width=line_width)

    def _draw_face_mouth(
        self,
        draw: object,
        *,
        center_x: int,
        center_y: int,
        status: str,
        animation_frame: int,
        scale: float,
        face_cue: HdmiFaceCueLike | None,
    ) -> None:
        mouth_style = self._safe_face_cue_mouth(face_cue)
        if mouth_style is not None:
            self._draw_face_cue_mouth(
                draw,
                center_x=center_x,
                center_y=center_y,
                animation_frame=animation_frame,
                scale=scale,
                mouth_style=mouth_style,
            )
            return
        line_width = self._scaled_size(4, scale, minimum=2)
        if status == "waiting":
            if self._safe_directional_face_cue_active(face_cue):
                sway = 0
                smile_width = 28
                smile_height = 16
            else:
                frame = animation_frame % 12
                sway = (-2, -1, 0, 1, 2, 1, 0, -1, -2, -1, 0, 1)[frame]
                smile_width = (26, 28, 30, 31, 30, 28, 26, 25, 26, 28, 30, 29)[frame]
                smile_height = (14, 15, 17, 18, 17, 16, 15, 14, 15, 16, 18, 17)[frame]
            draw.arc(
                (
                    center_x - self._scaled_offset(smile_width, scale),
                    center_y - self._scaled_offset(12, scale) + self._scaled_offset(sway, scale),
                    center_x + self._scaled_offset(smile_width, scale),
                    center_y + self._scaled_offset(smile_height, scale) + self._scaled_offset(sway, scale),
                ),
                start=24,
                end=156,
                fill=_WHITE,
                width=line_width,
            )
            return
        if status == "listening":
            frame = animation_frame % 6
            openness = (12, 16, 20, 18, 14, 10)[frame]
            mouth_width = (9, 10, 11, 11, 10, 9)[frame]
            mouth_lift = (0, -1, -2, -1, 0, 1)[frame]
            draw.ellipse(
                (
                    center_x - self._scaled_offset(mouth_width, scale),
                    center_y - self._scaled_offset(8, scale) + self._scaled_offset(mouth_lift, scale),
                    center_x + self._scaled_offset(mouth_width, scale),
                    center_y + self._scaled_offset(openness, scale) + self._scaled_offset(mouth_lift, scale),
                ),
                outline=_WHITE,
                width=line_width,
            )
            return
        if status == "processing":
            frame = animation_frame % 6
            offset_y = (-2, -1, 0, 1, 2, 1)[frame]
            mouth_width = (24, 22, 20, 22, 24, 26)[frame]
            draw.line(
                (
                    center_x - self._scaled_offset(mouth_width, scale),
                    center_y + self._scaled_offset(4 + offset_y, scale),
                    center_x - self._scaled_offset(4, scale),
                    center_y + self._scaled_offset(2 + offset_y, scale),
                ),
                fill=_WHITE,
                width=line_width,
            )
            draw.line(
                (
                    center_x + self._scaled_offset(4, scale),
                    center_y + self._scaled_offset(2 + offset_y, scale),
                    center_x + self._scaled_offset(mouth_width, scale),
                    center_y + self._scaled_offset(4 + offset_y, scale),
                ),
                fill=_WHITE,
                width=line_width,
            )
            return
        if status == "answering":
            frame = animation_frame % 6
            openness = (7, 10, 13, 11, 8, 12)[frame]
            mouth_width = (18, 20, 22, 21, 19, 20)[frame]
            draw.rounded_rectangle(
                (
                    center_x - self._scaled_offset(mouth_width, scale),
                    center_y - self._scaled_offset(2, scale),
                    center_x + self._scaled_offset(mouth_width, scale),
                    center_y + self._scaled_offset(openness, scale),
                ),
                radius=self._scaled_size(8, scale, minimum=2),
                outline=_WHITE,
                width=line_width,
            )
            return
        if status == "printing":
            frame = animation_frame % 6
            lift = (0, -2, -1, 0, 1, 0)[frame]
            smile_width = (26, 28, 30, 30, 28, 26)[frame]
            draw.arc(
                (
                    center_x - self._scaled_offset(smile_width, scale),
                    center_y - self._scaled_offset(6, scale) + self._scaled_offset(lift, scale),
                    center_x + self._scaled_offset(smile_width, scale),
                    center_y + self._scaled_offset(16, scale) + self._scaled_offset(lift, scale),
                ),
                start=12,
                end=168,
                fill=_WHITE,
                width=line_width,
            )
            return
        if status == "error":
            frame = animation_frame % 6
            wobble = (0, 1, 2, 1, 0, -1)[frame]
            draw.arc(
                (
                    center_x - self._scaled_offset(22, scale),
                    center_y + self._scaled_offset(6 + wobble, scale),
                    center_x + self._scaled_offset(22, scale),
                    center_y + self._scaled_offset(18 + wobble, scale),
                ),
                start=200,
                end=340,
                fill=_WHITE,
                width=line_width,
            )
            return

        draw.arc(
            (
                center_x - self._scaled_offset(22, scale),
                center_y - self._scaled_offset(4, scale),
                center_x + self._scaled_offset(22, scale),
                center_y + self._scaled_offset(10, scale),
            ),
            start=20,
            end=160,
            fill=_WHITE,
            width=line_width,
        )

    def _draw_face_cue_mouth(
        self,
        draw: object,
        *,
        center_x: int,
        center_y: int,
        animation_frame: int,
        scale: float,
        mouth_style: str,
    ) -> None:
        line_width = self._scaled_size(4, scale, minimum=2)
        if mouth_style == "neutral":
            draw.line(
                (
                    center_x - self._scaled_offset(22, scale),
                    center_y + self._scaled_offset(4, scale),
                    center_x + self._scaled_offset(22, scale),
                    center_y + self._scaled_offset(4, scale),
                ),
                fill=_WHITE,
                width=line_width,
            )
            return
        if mouth_style == "smile":
            sway = (-1, 0, 1, 0)[animation_frame % 4]
            draw.arc(
                (
                    center_x - self._scaled_offset(28, scale),
                    center_y - self._scaled_offset(10, scale) + self._scaled_offset(sway, scale),
                    center_x + self._scaled_offset(28, scale),
                    center_y + self._scaled_offset(16, scale) + self._scaled_offset(sway, scale),
                ),
                start=20,
                end=160,
                fill=_WHITE,
                width=line_width,
            )
            return
        if mouth_style == "open":
            draw.ellipse(
                (
                    center_x - self._scaled_offset(10, scale),
                    center_y - self._scaled_offset(8, scale),
                    center_x + self._scaled_offset(10, scale),
                    center_y + self._scaled_offset(16, scale),
                ),
                outline=_WHITE,
                width=line_width,
            )
            return
        if mouth_style == "pursed":
            pulse = (0, 1, 0, -1)[animation_frame % 4]
            draw.rounded_rectangle(
                (
                    center_x - self._scaled_offset(12, scale),
                    center_y - self._scaled_offset(4, scale),
                    center_x + self._scaled_offset(12, scale),
                    center_y + self._scaled_offset(10 + pulse, scale),
                ),
                radius=self._scaled_size(8, scale, minimum=2),
                outline=_WHITE,
                width=line_width,
            )
            return
        if mouth_style == "speak":
            openness = (10, 14, 18, 12)[animation_frame % 4]
            draw.rounded_rectangle(
                (
                    center_x - self._scaled_offset(18, scale),
                    center_y - self._scaled_offset(4, scale),
                    center_x + self._scaled_offset(18, scale),
                    center_y + self._scaled_offset(openness, scale),
                ),
                radius=self._scaled_size(8, scale, minimum=2),
                outline=_WHITE,
                width=line_width,
            )
            return
        if mouth_style == "sad":
            draw.arc(
                (
                    center_x - self._scaled_offset(22, scale),
                    center_y + self._scaled_offset(8, scale),
                    center_x + self._scaled_offset(22, scale),
                    center_y + self._scaled_offset(20, scale),
                ),
                start=200,
                end=340,
                fill=_WHITE,
                width=line_width,
            )
            return
        if mouth_style == "thinking":
            drift = (-1, 0, 1, 0)[animation_frame % 4]
            draw.line(
                (
                    center_x - self._scaled_offset(22, scale),
                    center_y + self._scaled_offset(6, scale),
                    center_x - self._scaled_offset(4, scale),
                    center_y + self._scaled_offset(2 + drift, scale),
                    center_x + self._scaled_offset(12, scale),
                    center_y + self._scaled_offset(6 - drift, scale),
                    center_x + self._scaled_offset(24, scale),
                    center_y + self._scaled_offset(4, scale),
                ),
                fill=_WHITE,
                width=line_width,
            )
            return
        if mouth_style == "scrunched":
            flutter = (-1, 0, 1, 0)[animation_frame % 4]
            draw.line(
                (
                    center_x - self._scaled_offset(18, scale),
                    center_y + self._scaled_offset(5 + flutter, scale),
                    center_x - self._scaled_offset(8, scale),
                    center_y + self._scaled_offset(1, scale),
                    center_x,
                    center_y + self._scaled_offset(7 - flutter, scale),
                    center_x + self._scaled_offset(8, scale),
                    center_y + self._scaled_offset(1, scale),
                    center_x + self._scaled_offset(18, scale),
                    center_y + self._scaled_offset(5 + flutter, scale),
                ),
                fill=_WHITE,
                width=line_width,
            )
            return
        draw.line(
            (
                center_x - self._scaled_offset(22, scale),
                center_y + self._scaled_offset(4, scale),
                center_x + self._scaled_offset(22, scale),
                center_y + self._scaled_offset(4, scale),
            ),
            fill=_WHITE,
            width=line_width,
        )

    def _draw_face_eye_highlights(
        self,
        draw: object,
        box: tuple[int, int, int, int],
        eye: _FaceEyeState,
        *,
        scale: float,
    ) -> None:
        center_x = (box[0] + box[2]) // 2
        center_y = (box[1] + box[3]) // 2
        main_x = center_x + self._scaled_offset(eye.highlight_dx, scale)
        main_y = center_y + self._scaled_offset(eye.highlight_dy, scale)
        main_radius = self._scaled_size(8, scale, minimum=2)
        secondary_x_offset = self._scaled_offset(10, scale)
        secondary_y_offset = self._scaled_offset(8, scale)
        secondary_width = self._scaled_size(6, scale, minimum=2)
        secondary_height = self._scaled_size(6, scale, minimum=2)
        draw.ellipse(
            (main_x - main_radius, main_y - main_radius, main_x + main_radius, main_y + main_radius),
            fill=_BLACK,
        )
        draw.ellipse(
            (
                main_x + secondary_x_offset,
                main_y + secondary_y_offset,
                main_x + secondary_x_offset + secondary_width,
                main_y + secondary_y_offset + secondary_height,
            ),
            fill=_BLACK,
        )

    def _surface_ops(self, draw: object) -> _PillowSurfaceOps | _AggdrawSurfaceOps:
        if _aggdraw is not None:
            # Direct aggdraw surface supplied by caller.
            if hasattr(draw, "flush") and hasattr(draw, "setantialias"):
                try:
                    return _AggdrawSurfaceOps(draw)
                except Exception:
                    pass
            # Common Pillow path: ImageDraw.Draw(image) exposes the backing image via _image.
            base_image = getattr(draw, "_image", None)
            if base_image is not None:
                try:
                    return _AggdrawSurfaceOps(_aggdraw.Draw(base_image))
                except Exception:
                    pass
        return _PillowSurfaceOps(draw)

    def _normalise_box(self, box: tuple[int, int, int, int] | object) -> tuple[int, int, int, int] | None:
        if not isinstance(box, tuple) or len(box) != 4:
            return None
        left = self._coerce_int(box[0], default=0, minimum=-_COORD_LIMIT, maximum=_COORD_LIMIT)
        top = self._coerce_int(box[1], default=0, minimum=-_COORD_LIMIT, maximum=_COORD_LIMIT)
        right = self._coerce_int(box[2], default=0, minimum=-_COORD_LIMIT, maximum=_COORD_LIMIT)
        bottom = self._coerce_int(box[3], default=0, minimum=-_COORD_LIMIT, maximum=_COORD_LIMIT)
        if right < left:
            left, right = right, left
        if bottom < top:
            top, bottom = bottom, top
        if right == left or bottom == top:
            return None
        return (left, top, right, bottom)

    def _bounded_face_scale(self, box: tuple[int, int, int, int], scale: float) -> float:
        safe_scale = self._normalise_scale(scale)
        width = max(1, box[2] - box[0])
        height = max(1, box[3] - box[1])
        width_cap = width / float(_FACE_HALF_WIDTH_UNITS * 2)
        height_cap = height / float(_FACE_TOP_UNITS + _FACE_BOTTOM_UNITS)
        return max(0.05, min(safe_scale, width_cap, height_cap))

    def _normalise_status(self, status: object) -> str:
        text = self._safe_text(status, fallback="waiting")
        return _STATUS_ALIASES.get(text, text)

    def _safe_face_cue_mouth(self, face_cue: HdmiFaceCueLike | None) -> str | None:
        if face_cue is None:
            return None
        try:
            raw = getattr(face_cue, "mouth", None)
        except Exception:
            return None
        if raw is None:
            return None
        text = self._safe_text(raw, fallback="")
        text = _MOUTH_STYLE_ALIASES.get(text, text)
        return text if text in _VALID_MOUTH_STYLES else None

    def _safe_eye_state(
        self,
        status: str,
        animation_frame: int,
        side: str,
        face_cue: HdmiFaceCueLike | None,
    ) -> _FaceEyeState:
        try:
            raw = self._eye_state(status, animation_frame, side, face_cue=face_cue)
        except Exception:
            raw = {}
        return self._normalise_eye_state(raw)

    def _safe_face_offset(
        self,
        status: str,
        animation_frame: int,
        face_cue: HdmiFaceCueLike | None,
    ) -> tuple[int, int]:
        try:
            raw_x, raw_y = self._face_offset(status, animation_frame, face_cue=face_cue)
        except Exception:
            return (0, 0)
        safe_x = self._coerce_int(raw_x, default=0, minimum=-_JITTER_LIMIT, maximum=_JITTER_LIMIT)
        safe_y = self._coerce_int(raw_y, default=0, minimum=-_JITTER_LIMIT, maximum=_JITTER_LIMIT)
        return (safe_x, safe_y)

    def _safe_directional_face_cue_active(self, face_cue: HdmiFaceCueLike | None) -> bool:
        try:
            return bool(self._directional_face_cue_active(face_cue))
        except Exception:
            return False

    def _normalise_eye_state(self, raw: object) -> _FaceEyeState:
        if not isinstance(raw, Mapping):
            return _FaceEyeState()
        brow_style = self._safe_text(raw.get("brow_style", ""), fallback="")
        if brow_style not in _VALID_BROW_STYLES:
            brow_style = ""
        return _FaceEyeState(
            blink=self._coerce_bool(raw.get("blink"), default=False),
            width=self._coerce_int(raw.get("width"), default=52, minimum=16, maximum=88),
            height=self._coerce_int(raw.get("height"), default=36, minimum=12, maximum=72),
            eye_shift_x=self._coerce_int(raw.get("eye_shift_x"), default=0, minimum=-18, maximum=18),
            eye_shift_y=self._coerce_int(raw.get("eye_shift_y"), default=0, minimum=-14, maximum=14),
            lid_arc=self._coerce_bool(raw.get("lid_arc"), default=False),
            brow_raise=self._coerce_int(raw.get("brow_raise"), default=0, minimum=-20, maximum=20),
            brow_slant=self._coerce_int(raw.get("brow_slant"), default=0, minimum=-20, maximum=20),
            brow_style=brow_style,
            highlight_dx=self._coerce_int(raw.get("highlight_dx"), default=-6, minimum=-18, maximum=18),
            highlight_dy=self._coerce_int(raw.get("highlight_dy"), default=-6, minimum=-18, maximum=18),
        )

    def _coerce_bool(self, value: object, *, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                return default
            return bool(value)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in _TRUE_STRINGS:
                return True
            if text in _FALSE_STRINGS:
                return False
            return default
        if value is None:
            return default
        return bool(value)

    def _coerce_int(
        self,
        value: object,
        *,
        default: int,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int:
        try:
            if isinstance(value, bool):
                coerced = int(value)
            elif isinstance(value, int):
                coerced = value
            elif isinstance(value, float):
                if not math.isfinite(value):
                    raise ValueError("non-finite float")
                coerced = int(round(value))
            elif isinstance(value, str):
                text = value.strip()
                if not text or len(text) > 32:
                    raise ValueError("unsupported integer string")
                coerced = int(float(text)) if any(ch in text for ch in ".eE") else int(text, 10)
            else:
                coerced = int(value)  # type: ignore[arg-type]
        except Exception:
            coerced = default
        if minimum is not None and coerced < minimum:
            coerced = minimum
        if maximum is not None and coerced > maximum:
            coerced = maximum
        return coerced

    def _safe_text(self, value: object, *, fallback: str) -> str:
        if value is None:
            return fallback
        try:
            text = str(value).strip().lower()
        except Exception:
            return fallback
        if not text:
            return fallback
        if len(text) > 64:
            return fallback
        return text