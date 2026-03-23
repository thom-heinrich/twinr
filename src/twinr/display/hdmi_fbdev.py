"""Render Twinr status screens on an HDMI framebuffer.

This backend targets the Raspberry Pi HDMI path exposed via ``/dev/fb0``. It
keeps Twinr's status-loop contract identical to the e-paper path while
rendering a calmer, larger, full-color screen for the HDMI panel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
import fcntl
import math
import os
from pathlib import Path
import subprocess
from threading import RLock
from typing import Any

from twinr.agent.base_agent.config import TwinrConfig
from twinr.display.ambient_impulse_cues import DisplayAmbientImpulseCue
from twinr.display.contracts import DisplayLogSections, DisplayStateFields
from twinr.display.debug_signals import DisplayDebugSignal
from twinr.display.emoji_cues import DisplayEmojiCue
from twinr.display.face_cues import DisplayFaceCue
from twinr.display.hdmi_default_scene import (
    HdmiDefaultSceneRenderer,
    display_state_value,
    order_state_fields,
    state_field_value,
    state_value_color,
    status_accent_color,
    status_headline,
    status_helper_text,
    time_value,
)
from twinr.display.presentation_cues import DisplayPresentationCue


_FBIOGET_VSCREENINFO = 0x4600
_FBIOGET_FSCREENINFO = 0x4602
_SUPPORTED_ROTATIONS = (0, 90, 180, 270)
_STATE_CARD_ORDER = ("Status", "Internet", "AI", "System", "Zeit", "Hinweis")


@dataclass(frozen=True, slots=True)
class FramebufferBitfield:
    """Describe one framebuffer color/transparency channel."""

    offset: int
    length: int
    msb_right: int


@dataclass(frozen=True, slots=True)
class FramebufferGeometry:
    """Describe the active framebuffer memory layout."""

    width: int
    height: int
    bits_per_pixel: int
    line_length: int
    red: FramebufferBitfield
    green: FramebufferBitfield
    blue: FramebufferBitfield
    transp: FramebufferBitfield

    @property
    def bytes_per_pixel(self) -> int:
        return max(1, self.bits_per_pixel // 8)

    @property
    def frame_size_bytes(self) -> int:
        return self.line_length * self.height


@dataclass(slots=True)
class HdmiFramebufferDisplay:
    """Render Twinr status frames into the configured HDMI framebuffer."""

    framebuffer_path: Path = Path("/dev/fb0")
    driver: str = "hdmi_fbdev"
    width: int = 0
    height: int = 0
    rotation_degrees: int = 0
    layout_mode: str = "default"
    emit: Callable[[str], None] | None = None
    _geometry: FramebufferGeometry | None = field(default=None, init=False, repr=False)
    _framebuffer: Any | None = field(default=None, init=False, repr=False)
    _font_cache: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _emoji_cache: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _emoji_font_path_cache: Path | None = field(default=None, init=False, repr=False)
    _emoji_font_path_resolved: bool = field(default=False, init=False, repr=False)
    _emoji_font_missing_reported: bool = field(default=False, init=False, repr=False)
    _lock: object = field(default_factory=RLock, init=False, repr=False)
    _last_rendered_status: str | None = field(default=None, init=False, repr=False)
    _default_scene_renderer: HdmiDefaultSceneRenderer | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.framebuffer_path = self.framebuffer_path.expanduser()
        self.rotation_degrees = self.rotation_degrees % 360
        self.layout_mode = self._normalise_layout_mode(self.layout_mode)
        self._default_scene_renderer = HdmiDefaultSceneRenderer(self)
        if self.driver != "hdmi_fbdev":
            raise RuntimeError(f"HdmiFramebufferDisplay does not support driver `{self.driver}`.")
        if self.rotation_degrees not in _SUPPORTED_ROTATIONS:
            raise RuntimeError("Display rotation must be one of 0, 90, 180, or 270 degrees.")

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        emit: Callable[[str], None] | None = None,
    ) -> "HdmiFramebufferDisplay":
        """Build an HDMI framebuffer adapter from Twinr configuration."""

        return cls(
            framebuffer_path=Path(getattr(config, "display_fb_path", "/dev/fb0") or "/dev/fb0"),
            driver=config.display_driver,
            width=max(0, int(getattr(config, "display_width", 0) or 0)),
            height=max(0, int(getattr(config, "display_height", 0) or 0)),
            rotation_degrees=config.display_rotation_degrees,
            layout_mode=config.display_layout,
            emit=emit,
        )

    @property
    def geometry(self) -> FramebufferGeometry:
        """Return the active framebuffer geometry, probing it on demand."""

        if self._geometry is None:
            self._geometry = self._read_framebuffer_geometry()
            self._safe_emit(
                " ".join(
                    (
                        "display_framebuffer=ready",
                        f"path={self.framebuffer_path}",
                        f"size={self._geometry.width}x{self._geometry.height}",
                        f"bpp={self._geometry.bits_per_pixel}",
                    )
                )
            )
        return self._geometry

    @property
    def framebuffer_size(self) -> tuple[int, int]:
        """Return the physical framebuffer resolution."""

        geometry = self.geometry
        return (geometry.width, geometry.height)

    @property
    def canvas_size(self) -> tuple[int, int]:
        """Return the logical render canvas before rotation is applied."""

        width, height = self.framebuffer_size
        if self.rotation_degrees in (90, 270):
            return (height, width)
        return (width, height)

    def show_test_pattern(self) -> None:
        """Render and display the HDMI smoke-test card."""

        image = self.render_test_image()
        self.show_image(image)

    def supports_idle_waiting_animation(self) -> bool:
        """Return whether this backend can animate the idle waiting face."""

        return self.layout_mode == "default"

    def show_status(
        self,
        status: str,
        *,
        headline: str | None = None,
        ticker_text: str | None = None,
        details: tuple[str, ...] = (),
        state_fields: DisplayStateFields = (),
        log_sections: DisplayLogSections = (),
        debug_signals: tuple[DisplayDebugSignal, ...] = (),
        animation_frame: int = 0,
        face_cue: DisplayFaceCue | None = None,
        emoji_cue: DisplayEmojiCue | None = None,
        ambient_impulse_cue: DisplayAmbientImpulseCue | None = None,
        presentation_cue: DisplayPresentationCue | None = None,
    ) -> None:
        """Render and display one runtime status frame."""

        image = self.render_status_image(
            status=status,
            headline=headline,
            ticker_text=ticker_text,
            details=details,
            state_fields=state_fields,
            log_sections=log_sections,
            debug_signals=debug_signals,
            animation_frame=animation_frame,
            face_cue=face_cue,
            emoji_cue=emoji_cue,
            ambient_impulse_cue=ambient_impulse_cue,
            presentation_cue=presentation_cue,
        )
        self.show_image(image)
        self._last_rendered_status = self._normalise_text(status, fallback="status").lower() or "status"

    def show_image(self, image: object) -> None:
        """Write one prepared frame into the framebuffer."""

        with self._lock:
            prepared = self._prepare_image(image)
            geometry = self.geometry
            payload = self._pack_framebuffer_bytes(prepared, geometry)
            framebuffer = self._open_framebuffer()
            framebuffer.seek(0)
            framebuffer.write(payload)
            framebuffer.flush()

    def render_test_image(self):
        """Build a full-screen HDMI validation card."""

        image, draw = self._new_canvas()
        width, height = image.size
        header_font = self._font(24, bold=True)
        title_font = self._font(54, bold=True)
        body_font = self._font(24, bold=False)
        caption_font = self._font(18, bold=False)

        self._draw_background(draw, width=width, height=height, accent=(33, 121, 219))
        draw.rounded_rectangle((28, 22, width - 28, 88), radius=26, fill=(18, 40, 78))
        draw.text((48, 38), "TWINR HDMI TEST", fill=(248, 246, 240), font=header_font)
        draw.text((46, 120), "Display ist bereit", fill=(25, 35, 48), font=title_font)
        draw.text((48, 194), "Dieser Bildschirm nutzt den neuen HDMI-Pfad.", fill=(49, 63, 83), font=body_font)

        card_top = 266
        card_height = 128
        card_gap = 18
        card_width = (width - 96 - (card_gap * 2)) // 3
        swatches = (
            ("Signal", (33, 121, 219), "HDMI"),
            ("Modus", (28, 160, 126), f"{self.geometry.width} x {self.geometry.height}"),
            ("Ausgabe", (228, 152, 34), str(self.framebuffer_path)),
        )
        for index, swatch in enumerate(swatches):
            title, color, value = swatch
            left = 32 + index * (card_width + card_gap)
            right = left + card_width
            draw.rounded_rectangle((left, card_top, right, card_top + card_height), radius=24, fill=(248, 246, 240))
            draw.rounded_rectangle((left + 18, card_top + 18, left + 74, card_top + 74), radius=18, fill=color)
            draw.text((left + 96, card_top + 20), title, fill=(88, 97, 112), font=caption_font)
            draw.text((left + 96, card_top + 54), value, fill=(25, 35, 48), font=body_font)

        draw.text(
            (48, height - 46),
            "Wenn du diese Karte siehst, nutzt Twinr nicht mehr den alten Waveshare-GPIO-Pfad.",
            fill=(82, 95, 114),
            font=caption_font,
        )
        return image

    def render_status_image(
        self,
        *,
        status: str,
        headline: str | None,
        details: tuple[str, ...],
        state_fields: DisplayStateFields,
        log_sections: DisplayLogSections,
        debug_signals: tuple[DisplayDebugSignal, ...] = (),
        animation_frame: int = 0,
        ticker_text: str | None = None,
        face_cue: DisplayFaceCue | None = None,
        emoji_cue: DisplayEmojiCue | None = None,
        ambient_impulse_cue: DisplayAmbientImpulseCue | None = None,
        presentation_cue: DisplayPresentationCue | None = None,
        render_now: datetime | None = None,
    ):
        """Render one HDMI-friendly status frame."""

        try:
            from PIL import ImageDraw
        except ImportError as exc:  # pragma: no cover - environment issue
            raise RuntimeError("Pillow is required for Twinr HDMI rendering.") from exc

        image, draw = self._new_canvas()
        del ImageDraw
        width, height = image.size
        safe_status = self._normalise_text(status, fallback="waiting").lower() or "waiting"
        safe_headline = self._status_headline(safe_status, fallback=headline)
        helper_text = self._status_helper_text(safe_status)
        ordered_fields = self._ordered_state_fields(state_fields, details)

        accent = self._status_accent_color(safe_status)
        self._draw_background(draw, width=width, height=height, accent=accent)
        if self.layout_mode == "debug_log":
            self._draw_debug_log_screen(
                draw,
                width=width,
                height=height,
                status=safe_status,
                headline=safe_headline,
                helper_text=helper_text,
                state_fields=ordered_fields,
                log_sections=log_sections,
                debug_signals=debug_signals,
            )
            return image

        self._draw_default_screen(
            image,
            draw,
            width=width,
            height=height,
            status=safe_status,
            headline=safe_headline,
            ticker_text=ticker_text,
            helper_text=helper_text,
            state_fields=ordered_fields,
            debug_signals=debug_signals,
            animation_frame=animation_frame,
            face_cue=face_cue,
            emoji_cue=emoji_cue,
            ambient_impulse_cue=ambient_impulse_cue,
            presentation_cue=presentation_cue,
            render_now=render_now,
        )
        return image

    def close(self) -> None:
        """Release the open framebuffer handle."""

        framebuffer = self._framebuffer
        self._framebuffer = None
        if framebuffer is None:
            return
        try:
            framebuffer.close()
        except Exception:
            return

    def _scene_renderer(self) -> HdmiDefaultSceneRenderer:
        renderer = self._default_scene_renderer
        if renderer is None:
            renderer = HdmiDefaultSceneRenderer(self)
            self._default_scene_renderer = renderer
        return renderer

    def _draw_default_screen(
        self,
        image: object,
        draw: object,
        *,
        width: int,
        height: int,
        status: str,
        headline: str,
        helper_text: str,
        state_fields: DisplayStateFields,
        debug_signals: tuple[DisplayDebugSignal, ...],
        animation_frame: int,
        ticker_text: str | None = None,
        face_cue: DisplayFaceCue | None = None,
        emoji_cue: DisplayEmojiCue | None = None,
        ambient_impulse_cue: DisplayAmbientImpulseCue | None = None,
        presentation_cue: DisplayPresentationCue | None = None,
        render_now: datetime | None = None,
    ) -> None:
        self._scene_renderer().draw(
            image=image,
            draw=draw,
            width=width,
            height=height,
            status=status,
            headline=headline,
            ticker_text=ticker_text,
            helper_text=helper_text,
            state_fields=state_fields,
            debug_signals=debug_signals,
            animation_frame=animation_frame,
            face_cue=face_cue,
            emoji_cue=emoji_cue,
            ambient_impulse_cue=ambient_impulse_cue,
            presentation_cue=presentation_cue,
            ambient_now=render_now,
        )

    def _state_field_value(
        self,
        state_fields: DisplayStateFields,
        name: str | tuple[str, ...],
        *,
        fallback: str = "--",
    ) -> str:
        return state_field_value(self._normalise_text, state_fields, name, fallback=fallback)

    def _display_state_value(self, field_name: str, value: str) -> str:
        return display_state_value(self._normalise_text, field_name, value)

    def _draw_debug_log_screen(
        self,
        draw: object,
        *,
        width: int,
        height: int,
        status: str,
        headline: str,
        helper_text: str,
        state_fields: DisplayStateFields,
        log_sections: DisplayLogSections,
        debug_signals: tuple[DisplayDebugSignal, ...] = (),
    ) -> None:
        del debug_signals
        header_font = self._font(20, bold=True)
        status_font = self._font(38, bold=True)
        helper_font = self._font(18, bold=False)
        title_font = self._font(18, bold=True)
        line_font = self._font(15, bold=False)

        sidebar_width = 232
        draw.rounded_rectangle((20, 18, width - 20, height - 18), radius=28, fill=(248, 246, 240))
        draw.rounded_rectangle((20, 18, sidebar_width, height - 18), radius=28, fill=(25, 37, 58))
        draw.text((42, 40), "TWINR Diagnose", fill=(243, 240, 232), font=header_font)
        draw.text((42, 92), headline, fill=(255, 255, 255), font=status_font)
        draw.text((42, 144), helper_text, fill=(206, 214, 224), font=helper_font)
        self._draw_state_cards(
            draw,
            left=36,
            top=222,
            width=sidebar_width - 52,
            height=220,
            state_fields=state_fields,
            compact=True,
        )

        cards_left = sidebar_width + 18
        cards_right = width - 32
        cards_top = 32
        card_gap = 14
        top_card_height = 160
        bottom_card_top = cards_top + top_card_height + card_gap
        half_width = (cards_right - cards_left - card_gap) // 2
        card_specs = (
            (cards_left, cards_top, cards_left + half_width, cards_top + top_card_height),
            (cards_left + half_width + card_gap, cards_top, cards_right, cards_top + top_card_height),
            (cards_left, bottom_card_top, cards_right, height - 32),
        )
        fallback_sections = (
            ("System", ("Noch keine Systemereignisse.",)),
            ("Hardware", ("Noch keine Hardwareereignisse.",)),
            ("Sprachlog", ("Noch keine Sprachereignisse.",)),
        )
        sections = tuple(log_sections[:3]) if log_sections else fallback_sections
        while len(sections) < 3:
            sections += (fallback_sections[len(sections)],)

        for (left, top, right, bottom), section in zip(card_specs, sections):
            title, lines = section
            draw.rounded_rectangle((left, top, right, bottom), radius=24, fill=(255, 251, 244))
            draw.text((left + 18, top + 14), title, fill=(28, 40, 56), font=title_font)
            draw.line((left + 18, top + 44, right - 18, top + 44), fill=(218, 210, 197), width=2)
            line_y = top + 58
            max_width = max(40, right - left - 36)
            line_height = self._text_height(draw, font=line_font) + 6
            max_lines = max(1, (bottom - line_y - 12) // max(line_height, 1))
            wrapped = self._wrapped_lines(draw, lines, max_width=max_width, font=line_font, max_lines=max_lines)
            for line in wrapped:
                draw.text((left + 18, line_y), line, fill=(72, 82, 96), font=line_font)
                line_y += line_height

        draw.text((cards_left, height - 22), f"Status: {headline} | Modus: {status}", fill=(98, 108, 123), font=helper_font)

    def _draw_background(self, draw: object, *, width: int, height: int, accent: tuple[int, int, int]) -> None:
        bands = 18
        top_color = (252, 247, 237)
        bottom_color = (234, 226, 211)
        for index in range(bands):
            y0 = int((height / bands) * index)
            y1 = int((height / bands) * (index + 1))
            ratio = index / max(1, bands - 1)
            color = tuple(
                int((top * (1.0 - ratio)) + (bottom * ratio))
                for top, bottom in zip(top_color, bottom_color)
            )
            draw.rectangle((0, y0, width, y1), fill=color)
        draw.ellipse((width - 260, -90, width + 100, 230), fill=(*accent, 40))
        draw.ellipse((-120, height - 220, 300, height + 120), fill=(255, 255, 255))

    def _draw_state_cards(
        self,
        draw: object,
        *,
        left: int,
        top: int,
        width: int,
        height: int,
        state_fields: DisplayStateFields,
        compact: bool = False,
    ) -> None:
        filtered = tuple(field for field in state_fields if field[0] in _STATE_CARD_ORDER)
        if not filtered:
            return
        columns = 2 if compact or len(filtered) > 3 else len(filtered)
        columns = max(1, min(columns, 2))
        rows = math.ceil(len(filtered) / columns)
        gap = 12
        card_width = (width - (gap * (columns - 1))) // columns
        card_height = (height - (gap * (rows - 1))) // rows
        label_font = self._font(16 if compact else 15, bold=True)
        value_font = self._font(18 if compact else 22, bold=False)
        for index, field in enumerate(filtered):
            column = index % columns
            row = index // columns
            card_left = left + column * (card_width + gap)
            card_top = top + row * (card_height + gap)
            card_right = card_left + card_width
            card_bottom = card_top + card_height
            color = self._state_value_color(field[1])
            draw.rounded_rectangle((card_left, card_top, card_right, card_bottom), radius=18, fill=(255, 252, 247))
            draw.rounded_rectangle((card_left + 14, card_top + 14, card_left + 46, card_top + 46), radius=12, fill=color)
            draw.text((card_left + 58, card_top + 15), field[0], fill=(96, 101, 112), font=label_font)
            value = self._truncate_text(draw, field[1], max_width=card_width - 74, font=value_font)
            draw.text((card_left + 58, card_top + 43), value, fill=(23, 34, 45), font=value_font)

    def _ordered_state_fields(
        self,
        state_fields: DisplayStateFields,
        details: tuple[str, ...],
    ) -> DisplayStateFields:
        return order_state_fields(self._normalise_text, state_fields, details)

    def _status_headline(self, status: str, *, fallback: str | None) -> str:
        return status_headline(self._normalise_text, status, fallback=fallback)

    def _status_helper_text(self, status: str) -> str:
        return status_helper_text(status)

    def _status_accent_color(self, status: str) -> tuple[int, int, int]:
        return status_accent_color(status)

    def _state_value_color(self, value: str) -> tuple[int, int, int]:
        return state_value_color(self._normalise_text, value)

    def _time_value(self, state_fields: DisplayStateFields) -> str:
        return time_value(state_fields)

    def _normalise_layout_mode(self, value: object) -> str:
        normalized = self._normalise_text(value, fallback="default").lower() or "default"
        if normalized == "debug_face":
            return "debug_log"
        if normalized not in {"default", "debug_log"}:
            raise RuntimeError(f"Unsupported display layout: {normalized}")
        return normalized

    def _normalise_text(self, value: object, *, fallback: str) -> str:
        if value is None:
            return fallback
        compact = " ".join("".join(ch if ch.isprintable() else " " for ch in str(value)).split())
        return compact or fallback

    def _new_canvas(self):
        try:
            from PIL import Image, ImageDraw
        except ImportError as exc:  # pragma: no cover - environment issue
            raise RuntimeError("Pillow is required for Twinr HDMI rendering.") from exc
        image = Image.new("RGB", self.canvas_size, (255, 255, 255))
        draw = ImageDraw.Draw(image, "RGBA")
        return image, draw

    def _font(self, size: int, *, bold: bool) -> object:
        cache_key = f"{size}:{int(bool(bold))}"
        cached = self._font_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            from PIL import ImageFont
        except ImportError as exc:  # pragma: no cover - environment issue
            raise RuntimeError("Pillow is required for Twinr HDMI font rendering.") from exc
        font_candidates = (
            (
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            )
            if bold
            else (
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            )
        )
        font = None
        for candidate in font_candidates:
            try:
                font = ImageFont.truetype(candidate, size=max(8, int(size)))
                break
            except OSError:
                continue
        if font is None:
            font = ImageFont.load_default()
        self._font_cache[cache_key] = font
        return font

    def _text_width(self, draw: object, text: str, *, font: object | None = None) -> int:
        if not text:
            return 0
        return draw.textbbox((0, 0), text, font=font)[2]

    def _text_height(self, draw: object, *, font: object | None = None) -> int:
        return draw.textbbox((0, 0), "Hg", font=font)[3]

    def _truncate_text(self, draw: object, text: str, *, max_width: int, font: object | None = None) -> str:
        compact = self._normalise_text(text, fallback="")
        if max_width <= 0 or self._text_width(draw, compact, font=font) <= max_width:
            return compact
        ellipsis = "..."
        while compact and self._text_width(draw, compact + ellipsis, font=font) > max_width:
            compact = compact[:-1].rstrip()
        return (compact + ellipsis).strip() if compact else ellipsis

    def _render_emoji_glyph(self, emoji: str, *, target_size: int) -> object | None:
        """Render one real Unicode emoji into a bounded transparent RGBA image."""

        compact = self._normalise_text(emoji, fallback="")
        if not compact or target_size <= 0:
            return None
        cache_key = f"{compact}:{int(target_size)}"
        cached = self._emoji_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            return None
        font_path = self._resolve_emoji_font_path()
        if font_path is None:
            return None
        try:
            font = ImageFont.truetype(str(font_path), 109)
        except OSError:
            self._emit_missing_emoji_font_once(f"display_emoji_font=unreadable path={font_path}")
            return None
        canvas = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), compact, font=font, embedded_color=True)
        bbox = canvas.getbbox()
        if bbox is None:
            return None
        glyph = canvas.crop(bbox)
        fitted = glyph.resize((target_size, target_size), resample=Image.Resampling.LANCZOS)
        self._emoji_cache[cache_key] = fitted
        return fitted.copy()

    def _resolve_emoji_font_path(self) -> Path | None:
        """Return one usable emoji font path, caching the first successful probe."""

        if self._emoji_font_path_resolved:
            return self._emoji_font_path_cache
        for candidate in self._emoji_font_candidates():
            if candidate.is_file():
                self._emoji_font_path_cache = candidate
                self._emoji_font_path_resolved = True
                return candidate
        self._emoji_font_path_resolved = True
        self._emit_missing_emoji_font_once(
            "display_emoji_font=missing expected=/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
        )
        return None

    def _emoji_font_candidates(self) -> tuple[Path, ...]:
        """Return prioritized emoji-font candidates for Linux HDMI deployments."""

        candidates: list[Path] = []
        override = str(os.environ.get("TWINR_DISPLAY_EMOJI_FONT_PATH", "")).strip()
        if override:
            candidates.append(Path(override).expanduser())
        candidates.extend(
            (
                Path("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"),
                Path("/usr/local/share/fonts/NotoColorEmoji.ttf"),
                Path("/usr/local/share/fonts/truetype/noto/NotoColorEmoji.ttf"),
            )
        )
        fontconfig_match = self._emoji_font_path_from_fontconfig()
        if fontconfig_match is not None:
            candidates.append(fontconfig_match)
        deduped: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return tuple(deduped)

    def _emoji_font_path_from_fontconfig(self) -> Path | None:
        """Ask fontconfig for a likely emoji font path when common paths are absent."""

        try:
            result = subprocess.run(
                ("fc-match", "--format=%{file}\n", "Noto Color Emoji"),
                check=False,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        lines = (result.stdout or "").strip().splitlines()
        if not lines:
            return None
        path = Path(lines[0]).expanduser()
        if "emoji" not in str(path).lower():
            return None
        return path

    def _emit_missing_emoji_font_once(self, line: str) -> None:
        """Emit one bounded HDMI telemetry line for missing or unreadable emoji fonts."""

        if self._emoji_font_missing_reported:
            return
        self._emoji_font_missing_reported = True
        self._safe_emit(line)

    def _wrapped_lines(
        self,
        draw: object,
        lines: tuple[str, ...],
        *,
        max_width: int,
        font: object,
        max_lines: int,
    ) -> tuple[str, ...]:
        wrapped: list[str] = []
        for source in lines:
            text = self._normalise_text(source, fallback="")
            if not text:
                continue
            words = text.split()
            current = ""
            for word in words:
                candidate = word if not current else f"{current} {word}"
                if self._text_width(draw, candidate, font=font) <= max_width:
                    current = candidate
                    continue
                if not current:
                    wrapped.append(self._truncate_text(draw, word, max_width=max_width, font=font))
                    if len(wrapped) >= max_lines:
                        return tuple(wrapped[:max_lines])
                    current = ""
                    continue
                if current:
                    wrapped.append(self._truncate_text(draw, current, max_width=max_width, font=font))
                    if len(wrapped) >= max_lines:
                        return tuple(wrapped[:max_lines])
                current = word
            if current:
                wrapped.append(self._truncate_text(draw, current, max_width=max_width, font=font))
            if len(wrapped) >= max_lines:
                return tuple(wrapped[:max_lines])
        return tuple(wrapped[:max_lines])

    def _prepare_image(self, image: object):
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - environment issue
            raise RuntimeError("Pillow is required for Twinr HDMI rendering.") from exc
        if not isinstance(image, Image.Image):
            raise RuntimeError("HDMI display expected a Pillow image.")
        prepared = image.convert("RGBA")
        if prepared.size != self.canvas_size:
            prepared = prepared.resize(self.canvas_size)
        if self.rotation_degrees:
            prepared = prepared.rotate(self.rotation_degrees, expand=True)
        if prepared.size != self.framebuffer_size:
            prepared = prepared.resize(self.framebuffer_size)
        return prepared

    def _open_framebuffer(self):
        framebuffer = self._framebuffer
        if framebuffer is not None:
            return framebuffer
        try:
            framebuffer = self.framebuffer_path.open("r+b", buffering=0)
        except OSError as exc:
            raise RuntimeError(f"Unable to open framebuffer `{self.framebuffer_path}`.") from exc
        self._framebuffer = framebuffer
        return framebuffer

    def _read_framebuffer_geometry(self) -> FramebufferGeometry:
        try:
            with self.framebuffer_path.open("rb", buffering=0) as framebuffer:
                vinfo = bytearray(160)
                finfo = bytearray(80)
                fcntl.ioctl(framebuffer, _FBIOGET_VSCREENINFO, vinfo, True)
                fcntl.ioctl(framebuffer, _FBIOGET_FSCREENINFO, finfo, True)
        except OSError as exc:
            raise RuntimeError(f"Unable to read framebuffer info from `{self.framebuffer_path}`.") from exc

        import struct

        width, height, _xvirt, _yvirt, _xoff, _yoff, bits_per_pixel, _gray = struct.unpack_from("8I", vinfo, 0)
        line_length = struct.unpack_from("I", finfo, 48)[0]
        red = FramebufferBitfield(*struct.unpack_from("3I", vinfo, 32))
        green = FramebufferBitfield(*struct.unpack_from("3I", vinfo, 44))
        blue = FramebufferBitfield(*struct.unpack_from("3I", vinfo, 56))
        transp = FramebufferBitfield(*struct.unpack_from("3I", vinfo, 68))

        geometry = FramebufferGeometry(
            width=int(width),
            height=int(height),
            bits_per_pixel=int(bits_per_pixel),
            line_length=int(line_length),
            red=red,
            green=green,
            blue=blue,
            transp=transp,
        )
        if geometry.width <= 0 or geometry.height <= 0:
            raise RuntimeError(f"Framebuffer `{self.framebuffer_path}` returned an invalid resolution.")
        if geometry.bits_per_pixel not in {16, 32}:
            raise RuntimeError(
                f"Framebuffer `{self.framebuffer_path}` uses unsupported {geometry.bits_per_pixel} bpp; expected 16 or 32."
            )
        for field in (geometry.red, geometry.green, geometry.blue, geometry.transp):
            if field.msb_right != 0:
                raise RuntimeError(f"Framebuffer `{self.framebuffer_path}` exposes an unsupported channel layout.")
        return geometry

    def _pack_framebuffer_bytes(self, image: object, geometry: FramebufferGeometry) -> bytes:
        rgba_image = image.convert("RGBA")
        pixels = rgba_image.load()
        output = bytearray(geometry.frame_size_bytes)
        for row in range(geometry.height):
            target_start = row * geometry.line_length
            for column in range(geometry.width):
                red, green, blue, alpha = pixels[column, row]
                packed = 0
                for value, field in (
                    (red, geometry.red),
                    (green, geometry.green),
                    (blue, geometry.blue),
                    (alpha, geometry.transp),
                ):
                    if field.length <= 0:
                        continue
                    scaled = ((int(value) * ((1 << field.length) - 1)) + 127) // 255
                    packed |= scaled << field.offset
                byte_offset = target_start + (column * geometry.bytes_per_pixel)
                output[byte_offset : byte_offset + geometry.bytes_per_pixel] = packed.to_bytes(
                    geometry.bytes_per_pixel,
                    "little",
                )
        return bytes(output)

    def _safe_emit(self, line: str) -> None:
        compact = self._normalise_text(line, fallback="")
        if not compact or self.emit is None:
            return
        try:
            self.emit(compact[:160])
        except Exception:
            return
