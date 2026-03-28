# CHANGELOG: 2026-03-28
# BUG-1: Fixed crescent rendering so it no longer punches an opaque black hole into non-black / RGBA scenes.
# BUG-2: Unknown ornament names no longer fail as a silent wrong-render; they warn once and degrade to sparkles.
# SEC-1: No practically exploitable issue found in this isolated renderer; no security code change required here.
# IMP-1: Added local RGBA supersampled ornament compositing for higher-quality edges on Pillow-backed HDMI canvases.
# IMP-2: Added typed ornament normalization, progress clamping, theme color hooks, and cutout-color override hooks.

# mypy: disable-error-code="attr-defined,call-overload,arg-type"
"""Ambient ornament rendering helpers for the default HDMI scene."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union, cast

from .typing_contracts import HdmiAmbientMomentLike, HdmiCanvasDrawSurface

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python < 3.11 compatibility shim.
    class StrEnum(str, Enum):
        pass

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - Pillow may be absent in some import-only test environments.
    Image = None
    ImageDraw = None

if Image is not None:
    _LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)
else:  # pragma: no cover - only used when Pillow is unavailable.
    _LANCZOS = None


class _AmbientOrnament(StrEnum):
    HEART = "heart"
    CRESCENT = "crescent"
    WAVE_MARKS = "wave_marks"
    CROWN = "crown"
    DOT_CLUSTER = "dot_cluster"
    SPARKLES = "sparkles"


@dataclass(frozen=True)
class _AmbientPlacement:
    dx: int
    dy: int
    extent_x: int
    extent_y: int
    renderer_name: str
    animated: bool = False


@dataclass(frozen=True)
class _AmbientOverlayContext:
    draw: HdmiCanvasDrawSurface
    center_x: int
    center_y: int
    scale: float
    tile: Any
    base_image: Any
    left: int
    top: int
    supersample: int


_AMBIENT_PLACEMENTS: dict[_AmbientOrnament, _AmbientPlacement] = {
    _AmbientOrnament.HEART: _AmbientPlacement(76, -84, 26, 40, "_draw_ambient_heart"),
    _AmbientOrnament.CRESCENT: _AmbientPlacement(-76, -78, 30, 28, "_draw_ambient_crescent"),
    _AmbientOrnament.WAVE_MARKS: _AmbientPlacement(98, -4, 34, 30, "_draw_ambient_wave_marks"),
    _AmbientOrnament.CROWN: _AmbientPlacement(0, -96, 28, 26, "_draw_ambient_crown"),
    _AmbientOrnament.DOT_CLUSTER: _AmbientPlacement(-78, -58, 28, 28, "_draw_ambient_dot_cluster", animated=True),
    _AmbientOrnament.SPARKLES: _AmbientPlacement(72, -68, 42, 30, "_draw_ambient_sparkles", animated=True),
}

_AMBIENT_UNKNOWN_ORNAMENTS_WARNED: set[str] = set()
_AmbientColor = Union[tuple[int, int, int], tuple[int, int, int, int]]


class HdmiAmbientRenderingMixin:
    """Render rare idle-only ambient ornaments around the HDMI face."""

    def _draw_ambient_moment(
        self,
        draw: HdmiCanvasDrawSurface,
        *,
        box: tuple[int, int, int, int],
        animation_frame: int,
        ambient_moment: HdmiAmbientMomentLike,
    ) -> None:
        """Draw a tiny ornament for one active idle ambient moment."""

        left, top, right, bottom = box
        scale = self._face_scale_for_box(box)
        center_x = (left + right) // 2
        center_y = top + ((bottom - top) // 2) - self._scaled_offset(6, scale)

        ornament = self._resolve_ambient_ornament(getattr(ambient_moment, "ornament", None))
        progress = self._normalized_ambient_progress(getattr(ambient_moment, "progress", 0.0))
        placement = _AMBIENT_PLACEMENTS[ornament]

        self._draw_ambient_ornament(
            draw,
            placement=placement,
            center_x=center_x + self._scaled_offset(placement.dx, scale),
            center_y=center_y + self._scaled_offset(placement.dy, scale),
            scale=scale,
            progress=progress,
            animation_frame=animation_frame,
        )

    def _draw_ambient_ornament(
        self,
        draw: HdmiCanvasDrawSurface,
        *,
        placement: _AmbientPlacement,
        center_x: int,
        center_y: int,
        scale: float,
        progress: float,
        animation_frame: int,
    ) -> None:
        renderer = getattr(self, placement.renderer_name)
        overlay = self._begin_ambient_overlay(
            draw,
            center_x=center_x,
            center_y=center_y,
            scale=scale,
            half_width=self._ambient_extent_px(placement.extent_x, scale, minimum=8),
            half_height=self._ambient_extent_px(placement.extent_y, scale, minimum=8),
        )

        if overlay is None:
            if placement.animated:
                renderer(
                    draw,
                    center_x=center_x,
                    center_y=center_y,
                    scale=scale,
                    progress=progress,
                    animation_frame=animation_frame,
                )
            else:
                renderer(
                    draw,
                    center_x=center_x,
                    center_y=center_y,
                    scale=scale,
                    progress=progress,
                )
            return

        if placement.animated:
            renderer(
                overlay.draw,
                center_x=overlay.center_x,
                center_y=overlay.center_y,
                scale=overlay.scale,
                progress=progress,
                animation_frame=animation_frame,
            )
        else:
            renderer(
                overlay.draw,
                center_x=overlay.center_x,
                center_y=overlay.center_y,
                scale=overlay.scale,
                progress=progress,
            )
        self._commit_ambient_overlay(overlay)

    def _resolve_ambient_ornament(self, ornament: object) -> _AmbientOrnament:
        if isinstance(ornament, _AmbientOrnament):
            return ornament
        if isinstance(ornament, str):
            try:
                return _AmbientOrnament(ornament)
            except ValueError:
                pass
        self._warn_unknown_ambient_ornament(ornament)
        return _AmbientOrnament.SPARKLES

    def _normalized_ambient_progress(self, progress: object) -> float:
        try:
            value = float(progress)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(value):
            return 0.0
        return max(0.0, min(1.0, value))

    def _warn_unknown_ambient_ornament(self, ornament: object) -> None:
        key = repr(ornament)
        if key in _AMBIENT_UNKNOWN_ORNAMENTS_WARNED:
            return
        _AMBIENT_UNKNOWN_ORNAMENTS_WARNED.add(key)
        warnings.warn(
            f"Unknown HDMI ambient ornament {ornament!r}; falling back to sparkle ornament.",
            RuntimeWarning,
            stacklevel=3,
        )

    def _ambient_ornament_rgb(self) -> tuple[int, int, int]:
        """Theme hook for ornament color; subclasses may override."""
        return (255, 255, 255)

    def _ambient_cutout_rgb(self) -> tuple[int, int, int]:
        """Fallback cutout color when alpha compositing is unavailable."""
        return (0, 0, 0)

    def _ambient_supersample_factor(self, scale: float) -> int:
        """Small local supersampling pass; tuned to stay inexpensive on Pi-class CPUs."""
        return 2 if scale > 0 else 1

    def _ambient_fill(
        self,
        draw: object,
        *,
        alpha: int = 255,
    ) -> _AmbientColor:
        rgb = self._ambient_ornament_rgb()
        clamped_alpha = max(0, min(255, int(round(alpha))))
        if self._draw_supports_alpha(draw):
            return (rgb[0], rgb[1], rgb[2], clamped_alpha)
        return rgb

    def _ambient_clear_fill(self, draw: object) -> _AmbientColor:
        if self._draw_supports_alpha(draw) and bool(getattr(draw, "_twinr_ambient_overlay", False)):
            return (0, 0, 0, 0)
        return self._ambient_cutout_rgb()

    def _draw_supports_alpha(self, draw: object) -> bool:
        mode = getattr(draw, "mode", None)
        return isinstance(mode, str) and "A" in mode.upper()

    def _ambient_extent_px(self, units: int, scale: float, *, minimum: int) -> int:
        return max(minimum, abs(int(self._scaled_offset(units, scale))))

    def _extract_pillow_image(self, draw: object) -> Optional[Any]:
        if Image is None or ImageDraw is None:
            return None
        image = getattr(draw, "_image", None)
        if image is None:
            return None
        if not hasattr(image, "paste") or not hasattr(image, "size"):
            return None
        return image

    def _begin_ambient_overlay(
        self,
        draw: HdmiCanvasDrawSurface,
        *,
        center_x: int,
        center_y: int,
        scale: float,
        half_width: int,
        half_height: int,
    ) -> Optional[_AmbientOverlayContext]:
        base_image = self._extract_pillow_image(draw)
        if base_image is None or Image is None or ImageDraw is None:
            return None

        supersample = max(1, int(self._ambient_supersample_factor(scale)))
        pad = max(2, self._scaled_size(2, scale, minimum=2))

        left = max(0, center_x - half_width - pad)
        top = max(0, center_y - half_height - pad)
        right = min(base_image.size[0], center_x + half_width + pad + 1)
        bottom = min(base_image.size[1], center_y + half_height + pad + 1)
        if right <= left or bottom <= top:
            return None

        tile_width = right - left
        tile_height = bottom - top
        tile = Image.new("RGBA", (tile_width * supersample, tile_height * supersample), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(tile, "RGBA")
        try:
            setattr(overlay_draw, "_twinr_ambient_overlay", True)
        except Exception:
            pass

        return _AmbientOverlayContext(
            draw=cast(HdmiCanvasDrawSurface, overlay_draw),
            center_x=(center_x - left) * supersample,
            center_y=(center_y - top) * supersample,
            scale=scale * supersample,
            tile=tile,
            base_image=base_image,
            left=left,
            top=top,
            supersample=supersample,
        )

    def _commit_ambient_overlay(self, overlay: _AmbientOverlayContext) -> None:
        tile = overlay.tile
        if tile.getbbox() is None:
            return

        if overlay.supersample > 1 and _LANCZOS is not None:
            tile = tile.resize(
                (
                    max(1, tile.size[0] // overlay.supersample),
                    max(1, tile.size[1] // overlay.supersample),
                ),
                resample=_LANCZOS,
            )

        base_mode = getattr(overlay.base_image, "mode", "")
        if base_mode in {"RGBA", "LA", "RGBa", "La"} and hasattr(overlay.base_image, "alpha_composite"):
            overlay.base_image.alpha_composite(tile, dest=(overlay.left, overlay.top))
            return
        overlay.base_image.paste(tile, (overlay.left, overlay.top), tile)

    def _draw_ambient_sparkles(
        self,
        draw: object,
        *,
        center_x: int,
        center_y: int,
        scale: float,
        progress: float,
        animation_frame: int,
    ) -> None:
        """Draw a small twinkling sparkle cluster near the waiting face."""

        progress = self._normalized_ambient_progress(progress)
        drift_y = self._scaled_offset((0.45 - progress) * 10.0, scale)
        pulse = (0, 1, 2, 1)[animation_frame % 4]
        size = self._scaled_size(5 + pulse, scale, minimum=3)
        line_width = self._scaled_size(2, scale, minimum=1)
        self._draw_star(
            draw,
            center_x=center_x,
            center_y=center_y + drift_y,
            size=size,
            line_width=line_width,
        )
        self._draw_star(
            draw,
            center_x=center_x + self._scaled_offset(22, scale),
            center_y=center_y - self._scaled_offset(12, scale) + drift_y,
            size=max(2, size - self._scaled_size(2, scale, minimum=1)),
            line_width=line_width,
        )

    def _draw_ambient_dot_cluster(
        self,
        draw: object,
        *,
        center_x: int,
        center_y: int,
        scale: float,
        progress: float,
        animation_frame: int,
    ) -> None:
        """Draw a tiny drifting dot cluster for the curious idle moment."""

        progress = self._normalized_ambient_progress(progress)
        lift = self._scaled_offset(progress * 10.0, scale)
        wobble = (-1, 0, 1, 0)[animation_frame % 4]
        radius = self._scaled_size(4, scale, minimum=2)
        fill = self._ambient_fill(draw)
        offsets = (
            (-10, 3),
            (1, -5),
            (12, 5),
        )
        for index, (offset_x, offset_y) in enumerate(offsets):
            jitter = wobble if index % 2 == 0 else -wobble
            cx = center_x + self._scaled_offset(offset_x + jitter, scale)
            cy = center_y + self._scaled_offset(offset_y, scale) - lift
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill)

    def _draw_ambient_heart(
        self,
        draw: object,
        *,
        center_x: int,
        center_y: int,
        scale: float,
        progress: float,
    ) -> None:
        """Draw a tiny floating heart near the idle face."""

        progress = self._normalized_ambient_progress(progress)
        fill = self._ambient_fill(draw)
        lift = self._scaled_offset(progress * 16.0, scale)
        top_y = center_y - self._scaled_offset(4, scale) - lift
        lobe_radius = self._scaled_size(7, scale, minimum=4)
        draw.ellipse(
            (
                center_x - (lobe_radius * 2),
                top_y - lobe_radius,
                center_x,
                top_y + lobe_radius,
            ),
            fill=fill,
        )
        draw.ellipse(
            (
                center_x,
                top_y - lobe_radius,
                center_x + (lobe_radius * 2),
                top_y + lobe_radius,
            ),
            fill=fill,
        )
        draw.polygon(
            (
                (center_x - (lobe_radius * 2), top_y + self._scaled_offset(2, scale)),
                (center_x + (lobe_radius * 2), top_y + self._scaled_offset(2, scale)),
                (center_x, top_y + (lobe_radius * 3)),
            ),
            fill=fill,
        )

    def _draw_ambient_crescent(
        self,
        draw: object,
        *,
        center_x: int,
        center_y: int,
        scale: float,
        progress: float,
    ) -> None:
        """Draw a tiny crescent moon for the sleepy idle moment."""

        progress = self._normalized_ambient_progress(progress)
        drift_y = self._scaled_offset(progress * 8.0, scale)
        radius = self._scaled_size(10, scale, minimum=5)
        box = (
            center_x - radius,
            center_y - radius - drift_y,
            center_x + radius,
            center_y + radius - drift_y,
        )
        draw.ellipse(box, fill=self._ambient_fill(draw))
        mask_offset = self._scaled_offset(5, scale)
        draw.ellipse(
            (
                box[0] + mask_offset,
                box[1] - self._scaled_offset(1, scale),
                box[2] + mask_offset,
                box[3] - self._scaled_offset(1, scale),
            ),
            fill=self._ambient_clear_fill(draw),
        )

    def _draw_ambient_wave_marks(
        self,
        draw: object,
        *,
        center_x: int,
        center_y: int,
        scale: float,
        progress: float,
    ) -> None:
        """Draw two tiny outward arcs that read like a gentle hello-wave."""

        progress = self._normalized_ambient_progress(progress)
        drift_x = self._scaled_offset(progress * 6.0, scale)
        line_width = self._scaled_size(2, scale, minimum=1)
        fill = self._ambient_fill(draw)
        step_y = self._scaled_offset(12, scale)
        for index, width in enumerate((18, 12)):
            offset_y = index * step_y
            draw.arc(
                (
                    center_x - self._scaled_offset(width, scale) + drift_x,
                    center_y - self._scaled_offset(10, scale) + offset_y,
                    center_x + self._scaled_offset(width, scale) + drift_x,
                    center_y + self._scaled_offset(10, scale) + offset_y,
                ),
                start=280,
                end=80,
                fill=fill,
                width=line_width,
            )

    def _draw_ambient_crown(
        self,
        draw: object,
        *,
        center_x: int,
        center_y: int,
        scale: float,
        progress: float,
    ) -> None:
        """Draw a tiny crown above the face for the playful proud moment."""

        progress = self._normalized_ambient_progress(progress)
        fill = self._ambient_fill(draw)
        lift = self._scaled_offset((0.5 - progress) * 6.0, scale)
        half_width = self._scaled_offset(16, scale)
        height = self._scaled_offset(12, scale)
        line_width = self._scaled_size(2, scale, minimum=1)
        points = (
            (center_x - half_width, center_y + height - lift),
            (center_x - self._scaled_offset(9, scale), center_y - self._scaled_offset(2, scale) - lift),
            (center_x, center_y + self._scaled_offset(2, scale) - lift),
            (center_x + self._scaled_offset(9, scale), center_y - self._scaled_offset(2, scale) - lift),
            (center_x + half_width, center_y + height - lift),
        )
        draw.line(points, fill=fill, width=line_width)
        draw.line(
            (
                center_x - half_width,
                center_y + height - lift,
                center_x + half_width,
                center_y + height - lift,
            ),
            fill=fill,
            width=line_width,
        )
        jewel_radius = self._scaled_size(2, scale, minimum=1)
        for jewel_x, jewel_y in points[1:4]:
            draw.ellipse(
                (
                    jewel_x - jewel_radius,
                    jewel_y - jewel_radius,
                    jewel_x + jewel_radius,
                    jewel_y + jewel_radius,
                ),
                fill=fill,
            )

    def _draw_star(
        self,
        draw: object,
        *,
        center_x: int,
        center_y: int,
        size: int,
        line_width: int,
    ) -> None:
        """Draw one tiny four-point star used by ambient sparkles."""

        fill = self._ambient_fill(draw)
        draw.line(
            (center_x - size, center_y, center_x + size, center_y),
            fill=fill,
            width=line_width,
        )
        draw.line(
            (center_x, center_y - size, center_x, center_y + size),
            fill=fill,
            width=line_width,
        )
        diagonal = max(1, int(round(size * 0.7)))
        draw.line(
            (center_x - diagonal, center_y - diagonal, center_x + diagonal, center_y + diagonal),
            fill=fill,
            width=max(1, line_width - 1),
        )
        draw.line(
            (center_x - diagonal, center_y + diagonal, center_x + diagonal, center_y - diagonal),
            fill=fill,
            width=max(1, line_width - 1),
        )