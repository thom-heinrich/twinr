# CHANGELOG: 2026-03-28
# BUG-1: Replaced fixed-offset framebuffer ioctl parsing with native-ABI ctypes structs; fixes broken line_length parsing on 32-bit Raspberry Pi OS.
# BUG-2: Honour framebuffer xoffset/yoffset and virtual-memory layout; fixes writes landing in the wrong visible page on panned or double-buffered fbdev setups.
# BUG-3: Handle short writes explicitly instead of assuming one write syscall copies a full frame.
# SEC-1: Harden framebuffer opening with O_NOFOLLOW/CLOEXEC and character-device validation by default; prevents arbitrary file corruption if the configured path is compromised.
# IMP-1: Added zero-copy mmap output, unchanged-frame suppression, and dirty-row writes to cut CPU, memory bandwidth, and visible tearing on Raspberry Pi 4 deployments.
# IMP-2: Added modern fast pixel packers (Pillow raw modes for common 24/32bpp, optional NumPy vector path, resilient generic fallback), 24bpp support, trusted fontconfig probing, and reopen/reprobe recovery.

"""Render Twinr status screens on an HDMI framebuffer.

This backend targets the Raspberry Pi HDMI path exposed via ``/dev/fb0``. It
keeps Twinr's status-loop contract identical to the e-paper path while
rendering a calmer, larger, full-color screen for the HDMI panel.
"""

from __future__ import annotations

from collections.abc import Callable
import ctypes
from dataclasses import dataclass, field
from datetime import datetime
import errno
import fcntl
import math
import mmap
import os
from pathlib import Path
import shutil
import stat
import subprocess
from threading import RLock

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
from twinr.display.service_connect_cues import DisplayServiceConnectCue
from twinr.display.wake_cues import DisplayWakeCue


_FBIOGET_VSCREENINFO = 0x4600
_FBIOGET_FSCREENINFO = 0x4602
_SUPPORTED_ROTATIONS = (0, 90, 180, 270)
_STATE_CARD_ORDER = ("Status", "Internet", "AI", "System", "Zeit", "Hinweis")
_FB_TYPE_FOURCC = 5
_FB_VISUAL_FOURCC = 6
_TRUSTED_SYSTEM_PATH = "/usr/local/bin:/usr/bin:/bin"
_SUPPORTED_PILLOW_RAW_MODES = frozenset({"RGB", "BGR", "RGBX", "BGRX", "XRGB", "XBGR", "RGBA", "BGRA", "ABGR"})


class _CFbBitfield(ctypes.Structure):
    _fields_ = (
        ("offset", ctypes.c_uint32),
        ("length", ctypes.c_uint32),
        ("msb_right", ctypes.c_uint32),
    )


class _CFbFixScreeninfo(ctypes.Structure):
    _fields_ = (
        ("id", ctypes.c_char * 16),
        ("smem_start", ctypes.c_ulong),
        ("smem_len", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("type_aux", ctypes.c_uint32),
        ("visual", ctypes.c_uint32),
        ("xpanstep", ctypes.c_uint16),
        ("ypanstep", ctypes.c_uint16),
        ("ywrapstep", ctypes.c_uint16),
        ("line_length", ctypes.c_uint32),
        ("mmio_start", ctypes.c_ulong),
        ("mmio_len", ctypes.c_uint32),
        ("accel", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint16),
        ("reserved", ctypes.c_uint16 * 2),
    )


class _CFbVarScreeninfo(ctypes.Structure):
    _fields_ = (
        ("xres", ctypes.c_uint32),
        ("yres", ctypes.c_uint32),
        ("xres_virtual", ctypes.c_uint32),
        ("yres_virtual", ctypes.c_uint32),
        ("xoffset", ctypes.c_uint32),
        ("yoffset", ctypes.c_uint32),
        ("bits_per_pixel", ctypes.c_uint32),
        ("grayscale", ctypes.c_uint32),
        ("red", _CFbBitfield),
        ("green", _CFbBitfield),
        ("blue", _CFbBitfield),
        ("transp", _CFbBitfield),
        ("nonstd", ctypes.c_uint32),
        ("activate", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("accel_flags", ctypes.c_uint32),
        ("pixclock", ctypes.c_uint32),
        ("left_margin", ctypes.c_uint32),
        ("right_margin", ctypes.c_uint32),
        ("upper_margin", ctypes.c_uint32),
        ("lower_margin", ctypes.c_uint32),
        ("hsync_len", ctypes.c_uint32),
        ("vsync_len", ctypes.c_uint32),
        ("sync", ctypes.c_uint32),
        ("vmode", ctypes.c_uint32),
        ("rotate", ctypes.c_uint32),
        ("colorspace", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 4),
    )


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
    virtual_width: int = 0
    virtual_height: int = 0
    xoffset: int = 0
    yoffset: int = 0
    bits_per_pixel: int = 0
    line_length: int = 0
    memory_length: int = 0
    framebuffer_type: int = 0
    visual: int = 0
    capabilities: int = 0
    grayscale: int = 0
    red: FramebufferBitfield = field(default_factory=lambda: FramebufferBitfield(offset=0, length=0, msb_right=0))
    green: FramebufferBitfield = field(default_factory=lambda: FramebufferBitfield(offset=0, length=0, msb_right=0))
    blue: FramebufferBitfield = field(default_factory=lambda: FramebufferBitfield(offset=0, length=0, msb_right=0))
    transp: FramebufferBitfield = field(default_factory=lambda: FramebufferBitfield(offset=0, length=0, msb_right=0))

    def __post_init__(self) -> None:
        """Preserve the historical minimal constructor contract.

        Older callers only supplied the visible geometry plus channel bitfields.
        Fill the newer low-level framebuffer metadata from those visible values
        when it was not provided explicitly.
        """

        width = max(1, int(self.width))
        height = max(1, int(self.height))
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)

        bytes_per_pixel = max(1, (int(self.bits_per_pixel) + 7) // 8) if int(self.bits_per_pixel) > 0 else 0
        virtual_width = int(self.virtual_width) if int(self.virtual_width) > 0 else width
        virtual_height = int(self.virtual_height) if int(self.virtual_height) > 0 else height
        line_length = int(self.line_length) if int(self.line_length) > 0 else (width * bytes_per_pixel)
        memory_length = int(self.memory_length) if int(self.memory_length) > 0 else (line_length * virtual_height)

        object.__setattr__(self, "virtual_width", virtual_width)
        object.__setattr__(self, "virtual_height", virtual_height)
        object.__setattr__(self, "line_length", line_length)
        object.__setattr__(self, "memory_length", memory_length)

    @property
    def bytes_per_pixel(self) -> int:
        return max(1, (self.bits_per_pixel + 7) // 8)

    @property
    def xoffset_bytes(self) -> int:
        return self.xoffset * self.bytes_per_pixel

    @property
    def visible_offset_bytes(self) -> int:
        return self.yoffset * self.line_length

    @property
    def row_pixel_bytes(self) -> int:
        return self.width * self.bytes_per_pixel

    @property
    def frame_size_bytes(self) -> int:
        return self.line_length * self.height

    @property
    def required_memory_bytes(self) -> int:
        return self.visible_offset_bytes + self.frame_size_bytes

    @property
    def mapping_size_bytes(self) -> int:
        if self.memory_length > 0:
            return max(self.memory_length, self.required_memory_bytes)
        return self.required_memory_bytes


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
    _framebuffer_fd: int | None = field(default=None, init=False, repr=False)
    _framebuffer_map: mmap.mmap | None = field(default=None, init=False, repr=False)
    _framebuffer_map_length: int = field(default=0, init=False, repr=False)
    _font_cache: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _font_path_cache: dict[str, Path | None] = field(default_factory=dict, init=False, repr=False)
    _emoji_cache: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _emoji_font_path_cache: Path | None = field(default=None, init=False, repr=False)
    _emoji_font_path_resolved: bool = field(default=False, init=False, repr=False)
    _emoji_font_missing_reported: bool = field(default=False, init=False, repr=False)
    _non_device_override_reported: bool = field(default=False, init=False, repr=False)
    _lock: object = field(default_factory=RLock, init=False, repr=False)
    _last_rendered_status: str | None = field(default=None, init=False, repr=False)
    _default_scene_renderer: HdmiDefaultSceneRenderer | None = field(default=None, init=False, repr=False)
    _last_payload: bytes | None = field(default=None, init=False, repr=False)
    _reported_io_backend: str | None = field(default=None, init=False, repr=False)
    _reported_pack_backend: str | None = field(default=None, init=False, repr=False)
    _legacy_open_framebuffer_func: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.framebuffer_path = self.framebuffer_path.expanduser()
        self.rotation_degrees = self.rotation_degrees % 360
        self.layout_mode = self._normalise_layout_mode(self.layout_mode)
        self._default_scene_renderer = HdmiDefaultSceneRenderer(self)
        self._legacy_open_framebuffer_func = getattr(self._open_framebuffer, "__func__", self._open_framebuffer)
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
                        f"virtual={self._geometry.virtual_width}x{self._geometry.virtual_height}",
                        f"offset={self._geometry.xoffset},{self._geometry.yoffset}",
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

    def refresh_geometry(self) -> FramebufferGeometry:
        """Re-probe framebuffer geometry after a modeset, hotplug, or recovery."""

        with self._lock:
            self._reset_framebuffer_state(clear_geometry=True)
            return self.geometry

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
        wake_cue: DisplayWakeCue | None = None,
        emoji_cue: DisplayEmojiCue | None = None,
        ambient_impulse_cue: DisplayAmbientImpulseCue | None = None,
        service_connect_cue: DisplayServiceConnectCue | None = None,
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
            wake_cue=wake_cue,
            emoji_cue=emoji_cue,
            ambient_impulse_cue=ambient_impulse_cue,
            service_connect_cue=service_connect_cue,
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
            if self._last_payload is not None and payload == self._last_payload:
                return
            compatibility_open = self._open_framebuffer
            compatibility_open_func = getattr(compatibility_open, "__func__", compatibility_open)
            if compatibility_open_func is not self._legacy_open_framebuffer_func:
                framebuffer = self._open_framebuffer()
                framebuffer.seek(0)
                framebuffer.write(payload)
                flush = getattr(framebuffer, "flush", None)
                if callable(flush):
                    flush()
                self._last_payload = payload
                return
            try:
                self._write_framebuffer_payload(payload, geometry)
            except OSError as exc:
                self._safe_emit(f"display_framebuffer=retry reason={exc.__class__.__name__}")
                self._reset_framebuffer_state(clear_geometry=True)
                geometry = self.geometry
                prepared = self._prepare_image(image)
                payload = self._pack_framebuffer_bytes(prepared, geometry)
                self._write_framebuffer_payload(payload, geometry)
            self._last_payload = payload

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
        wake_cue: DisplayWakeCue | None = None,
        emoji_cue: DisplayEmojiCue | None = None,
        ambient_impulse_cue: DisplayAmbientImpulseCue | None = None,
        service_connect_cue: DisplayServiceConnectCue | None = None,
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
            wake_cue=wake_cue,
            emoji_cue=emoji_cue,
            ambient_impulse_cue=ambient_impulse_cue,
            service_connect_cue=service_connect_cue,
            presentation_cue=presentation_cue,
            render_now=render_now,
        )
        return image

    def close(self) -> None:
        """Release the open framebuffer handle."""

        with self._lock:
            self._reset_framebuffer_state(clear_geometry=False)

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
        wake_cue: DisplayWakeCue | None = None,
        emoji_cue: DisplayEmojiCue | None = None,
        ambient_impulse_cue: DisplayAmbientImpulseCue | None = None,
        service_connect_cue: DisplayServiceConnectCue | None = None,
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
            wake_cue=wake_cue,
            emoji_cue=emoji_cue,
            ambient_impulse_cue=ambient_impulse_cue,
            service_connect_cue=service_connect_cue,
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
        filtered = tuple(state_field for state_field in state_fields if state_field[0] in _STATE_CARD_ORDER)
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
        for index, state_field in enumerate(filtered):
            column = index % columns
            row = index // columns
            card_left = left + column * (card_width + gap)
            card_top = top + row * (card_height + gap)
            card_right = card_left + card_width
            card_bottom = card_top + card_height
            color = self._state_value_color(state_field[1])
            draw.rounded_rectangle((card_left, card_top, card_right, card_bottom), radius=18, fill=(255, 252, 247))
            draw.rounded_rectangle((card_left + 14, card_top + 14, card_left + 46, card_top + 46), radius=12, fill=color)
            draw.text((card_left + 58, card_top + 15), state_field[0], fill=(96, 101, 112), font=label_font)
            value = self._truncate_text(draw, state_field[1], max_width=card_width - 74, font=value_font)
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
        font = None
        for candidate in self._font_candidates(bold=bold):
            try:
                font = ImageFont.truetype(str(candidate), size=max(8, int(size)))
                break
            except OSError:
                continue
        if font is None:
            font = ImageFont.load_default()
        self._font_cache[cache_key] = font
        return font

    def _font_candidates(self, *, bold: bool) -> tuple[Path, ...]:
        cache_key = "bold" if bold else "regular"
        cached = self._font_path_cache.get(cache_key)
        if cache_key in self._font_path_cache:
            return tuple(path for path in (cached,) if path is not None)
        candidates: list[Path] = []
        if bold:
            candidates.extend(
                (
                    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
                    Path("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"),
                    Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
                    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                )
            )
        else:
            candidates.extend(
                (
                    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                    Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
                    Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
                    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
                )
            )
        query = "DejaVu Sans:style=Bold" if bold else "DejaVu Sans:style=Book"
        fontconfig_match = self._font_path_from_fontconfig(query)
        if fontconfig_match is not None:
            candidates.append(fontconfig_match)
        deduped: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if candidate.is_file():
                deduped.append(candidate)
        self._font_path_cache[cache_key] = deduped[0] if deduped else None
        return tuple(deduped)

    def _font_path_from_fontconfig(self, query: str) -> Path | None:
        executable = self._fc_match_executable()
        if executable is None:
            return None
        try:
            result = subprocess.run(
                (executable, "--format=%{file}\n", query),
                check=False,
                capture_output=True,
                text=True,
                timeout=1.0,
                env={"LANG": "C", "LC_ALL": "C", "PATH": _TRUSTED_SYSTEM_PATH},
            )
        except (OSError, subprocess.SubprocessError):
            return None
        lines = (result.stdout or "").strip().splitlines()
        if not lines:
            return None
        path = Path(lines[0]).expanduser()
        return path if path.is_file() else None

    def _fc_match_executable(self) -> str | None:
        for candidate in ("/usr/bin/fc-match", "/bin/fc-match", "/usr/local/bin/fc-match"):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        discovered = shutil.which("fc-match", path=_TRUSTED_SYSTEM_PATH)
        return discovered if discovered else None

    def _text_width(self, draw: object, text: str, *, font: object | None = None) -> int:
        if not text:
            return 0
        left, _top, right, _bottom = draw.textbbox((0, 0), text, font=font)
        return max(0, right - left)

    def _text_height(self, draw: object, *, font: object | None = None) -> int:
        left, top, right, bottom = draw.textbbox((0, 0), "Hg", font=font)
        del left, right
        return max(0, bottom - top)

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
        resampling = self._pil_resampling()
        fitted = glyph.resize((target_size, target_size), resample=resampling.LANCZOS)
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

        executable = self._fc_match_executable()
        if executable is None:
            return None
        try:
            result = subprocess.run(
                (executable, "--format=%{file}\n", "Noto Color Emoji"),
                check=False,
                capture_output=True,
                text=True,
                timeout=1.0,
                env={"LANG": "C", "LC_ALL": "C", "PATH": _TRUSTED_SYSTEM_PATH},
            )
        except (OSError, subprocess.SubprocessError):
            return None
        lines = (result.stdout or "").strip().splitlines()
        if not lines:
            return None
        path = Path(lines[0]).expanduser()
        if "emoji" not in str(path).lower():
            return None
        return path if path.is_file() else None

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
        resampling = self._pil_resampling()
        prepared = image.convert("RGBA")
        if prepared.size != self.canvas_size:
            prepared = prepared.resize(self.canvas_size, resample=resampling.LANCZOS)
        if self.rotation_degrees:
            prepared = prepared.rotate(self.rotation_degrees, expand=True)
        if prepared.size != self.framebuffer_size:
            prepared = prepared.resize(self.framebuffer_size, resample=resampling.LANCZOS)
        return prepared

    def _pil_resampling(self):
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - environment issue
            raise RuntimeError("Pillow is required for Twinr HDMI rendering.") from exc
        return getattr(Image, "Resampling", Image)

    def _open_framebuffer_fd(self) -> int:
        framebuffer_fd = self._framebuffer_fd
        if framebuffer_fd is not None:
            return framebuffer_fd
        framebuffer_fd = self._open_checked_path(os.O_RDWR)
        self._framebuffer_fd = framebuffer_fd
        return framebuffer_fd

    def _open_framebuffer(self):
        """Open one legacy file-like framebuffer handle for compatibility callers."""

        return os.fdopen(os.dup(self._open_framebuffer_fd()), "r+b", buffering=0)

    def _open_checked_path(self, base_flags: int) -> int:
        flags = base_flags
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(self.framebuffer_path, flags)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise RuntimeError(f"Refusing to follow symlink for framebuffer `{self.framebuffer_path}`.") from exc
            if exc.errno == errno.ENOENT and Path("/dev/dri/card0").exists():
                raise RuntimeError(
                    f"Unable to open framebuffer `{self.framebuffer_path}`. "
                    "This system appears to expose DRM/KMS (`/dev/dri/card0`) but not a fbdev node. "
                    "Use a DRM/KMS backend or enable fbdev emulation for compatibility."
                ) from exc
            raise RuntimeError(f"Unable to open framebuffer `{self.framebuffer_path}`.") from exc
        try:
            self._validate_framebuffer_fd(fd)
            return fd
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise

    def _validate_framebuffer_fd(self, fd: int) -> None:
        stats = os.fstat(fd)
        if stat.S_ISCHR(stats.st_mode):
            return
        if self._allow_non_device_framebuffer():
            if not self._non_device_override_reported:
                self._non_device_override_reported = True
                self._safe_emit(f"display_framebuffer=non_device_override path={self.framebuffer_path}")
            return
        # BREAKING: Non-device framebuffer targets (for example a regular file used as a fake framebuffer in tests)
        # are now refused by default. Set TWINR_DISPLAY_ALLOW_NON_DEVICE_FB=1 to opt back into the old behaviour.
        raise RuntimeError(
            f"Refusing non-device framebuffer path `{self.framebuffer_path}`. "
            "Expected a character device such as /dev/fb0."
        )

    def _allow_non_device_framebuffer(self) -> bool:
        return str(os.environ.get("TWINR_DISPLAY_ALLOW_NON_DEVICE_FB", "")).strip().lower() in {"1", "true", "yes", "on"}

    def _read_framebuffer_geometry(self) -> FramebufferGeometry:
        try:
            fd = self._open_checked_path(os.O_RDONLY)
        except RuntimeError:
            raise
        try:
            vinfo_buffer = bytearray(ctypes.sizeof(_CFbVarScreeninfo))
            finfo_buffer = bytearray(ctypes.sizeof(_CFbFixScreeninfo))
            fcntl.ioctl(fd, _FBIOGET_VSCREENINFO, vinfo_buffer, True)
            fcntl.ioctl(fd, _FBIOGET_FSCREENINFO, finfo_buffer, True)
        except OSError as exc:
            raise RuntimeError(f"Unable to read framebuffer info from `{self.framebuffer_path}`.") from exc
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

        vinfo = _CFbVarScreeninfo.from_buffer_copy(vinfo_buffer)
        finfo = _CFbFixScreeninfo.from_buffer_copy(finfo_buffer)

        geometry = FramebufferGeometry(
            width=int(vinfo.xres),
            height=int(vinfo.yres),
            virtual_width=int(vinfo.xres_virtual),
            virtual_height=int(vinfo.yres_virtual),
            xoffset=int(vinfo.xoffset),
            yoffset=int(vinfo.yoffset),
            bits_per_pixel=int(vinfo.bits_per_pixel),
            line_length=int(finfo.line_length),
            memory_length=int(finfo.smem_len),
            framebuffer_type=int(finfo.type),
            visual=int(finfo.visual),
            capabilities=int(finfo.capabilities),
            grayscale=int(vinfo.grayscale),
            red=FramebufferBitfield(int(vinfo.red.offset), int(vinfo.red.length), int(vinfo.red.msb_right)),
            green=FramebufferBitfield(int(vinfo.green.offset), int(vinfo.green.length), int(vinfo.green.msb_right)),
            blue=FramebufferBitfield(int(vinfo.blue.offset), int(vinfo.blue.length), int(vinfo.blue.msb_right)),
            transp=FramebufferBitfield(int(vinfo.transp.offset), int(vinfo.transp.length), int(vinfo.transp.msb_right)),
        )
        self._validate_geometry(geometry)
        return geometry

    def _validate_geometry(self, geometry: FramebufferGeometry) -> None:
        if geometry.width <= 0 or geometry.height <= 0:
            raise RuntimeError(f"Framebuffer `{self.framebuffer_path}` returned an invalid resolution.")
        if geometry.bits_per_pixel not in {16, 24, 32}:
            raise RuntimeError(
                f"Framebuffer `{self.framebuffer_path}` uses unsupported {geometry.bits_per_pixel} bpp; expected 16, 24, or 32."
            )
        if geometry.framebuffer_type == _FB_TYPE_FOURCC or geometry.visual == _FB_VISUAL_FOURCC:
            fourcc = geometry.grayscale.to_bytes(4, "little", signed=False).decode("latin1", errors="replace")
            raise RuntimeError(
                f"Framebuffer `{self.framebuffer_path}` exposes a FOURCC pixel format `{fourcc}`; "
                "this fbdev compatibility backend currently supports only RGB bitfield framebuffers."
            )
        for color_field in (geometry.red, geometry.green, geometry.blue, geometry.transp):
            if color_field.msb_right != 0:
                raise RuntimeError(f"Framebuffer `{self.framebuffer_path}` exposes an unsupported channel layout.")
        if geometry.line_length <= 0:
            raise RuntimeError(f"Framebuffer `{self.framebuffer_path}` returned an invalid line_length.")
        if geometry.virtual_width <= 0:
            raise RuntimeError(f"Framebuffer `{self.framebuffer_path}` returned an invalid virtual width.")
        if geometry.virtual_height <= 0:
            raise RuntimeError(f"Framebuffer `{self.framebuffer_path}` returned an invalid virtual height.")
        if geometry.xoffset < 0 or geometry.yoffset < 0:
            raise RuntimeError(f"Framebuffer `{self.framebuffer_path}` returned negative offsets.")
        if geometry.xoffset > geometry.virtual_width or geometry.yoffset > geometry.virtual_height:
            raise RuntimeError(f"Framebuffer `{self.framebuffer_path}` returned offsets outside the virtual framebuffer.")
        if geometry.row_pixel_bytes + geometry.xoffset_bytes > geometry.line_length:
            raise RuntimeError(
                f"Framebuffer `{self.framebuffer_path}` returned a visible row wider than line_length permits."
            )
        if geometry.memory_length > 0 and geometry.required_memory_bytes > geometry.memory_length:
            raise RuntimeError(
                f"Framebuffer `{self.framebuffer_path}` returned insufficient memory for the visible page "
                f"({geometry.required_memory_bytes}>{geometry.memory_length})."
            )

    def _pack_framebuffer_bytes(self, image: object, geometry: FramebufferGeometry) -> bytes:
        packers = (
            self._pack_framebuffer_bytes_pillow,
            self._pack_framebuffer_bytes_numpy,
            self._pack_framebuffer_bytes_generic,
        )
        last_error: Exception | None = None
        for packer in packers:
            try:
                payload, backend = packer(image, geometry)
                self._report_pack_backend_once(backend, geometry)
                return payload
            except Exception as exc:
                last_error = exc
                continue
        raise RuntimeError(f"Unable to pack framebuffer pixels for `{self.framebuffer_path}`.") from last_error

    def _pack_framebuffer_bytes_pillow(self, image: object, geometry: FramebufferGeometry) -> tuple[bytes, str]:
        raw_mode_info = self._pillow_raw_mode_for_geometry(geometry)
        if raw_mode_info is None:
            raise ValueError("no compatible Pillow raw mode")
        source_mode, raw_mode = raw_mode_info
        source = image.convert(source_mode)
        packed = source.tobytes("raw", raw_mode)
        if geometry.line_length == geometry.row_pixel_bytes and geometry.xoffset_bytes == 0:
            return packed, f"pillow:{raw_mode.lower()}"
        output = bytearray(geometry.frame_size_bytes)
        source_row_bytes = geometry.row_pixel_bytes
        for row in range(geometry.height):
            src_start = row * source_row_bytes
            dst_start = (row * geometry.line_length) + geometry.xoffset_bytes
            output[dst_start : dst_start + source_row_bytes] = packed[src_start : src_start + source_row_bytes]
        return bytes(output), f"pillow:{raw_mode.lower()}"

    def _pack_framebuffer_bytes_numpy(self, image: object, geometry: FramebufferGeometry) -> tuple[bytes, str]:
        try:
            import numpy as np
        except ImportError as exc:
            raise ValueError("numpy unavailable") from exc

        rgba_image = image.convert("RGBA")
        rgba = np.asarray(rgba_image, dtype=np.uint8)
        packed = np.zeros((geometry.height, geometry.width), dtype=np.uint32)
        for channel_index, color_field in enumerate((geometry.red, geometry.green, geometry.blue, geometry.transp)):
            if color_field.length <= 0:
                continue
            channel = rgba[:, :, channel_index].astype(np.uint32, copy=False)
            max_value = (1 << color_field.length) - 1
            scaled = ((channel * max_value) + 127) // 255
            packed |= scaled << color_field.offset

        if geometry.bytes_per_pixel == 2:
            pixel_rows = packed.astype("<u2", copy=False).view(np.uint8).reshape(geometry.height, geometry.width * 2)
        elif geometry.bytes_per_pixel == 3:
            pixel_rows = packed.astype("<u4", copy=False).view(np.uint8).reshape(geometry.height, geometry.width, 4)
            pixel_rows = pixel_rows[:, :, :3].reshape(geometry.height, geometry.width * 3)
        elif geometry.bytes_per_pixel == 4:
            pixel_rows = packed.astype("<u4", copy=False).view(np.uint8).reshape(geometry.height, geometry.width * 4)
        else:
            raise ValueError("unsupported bytes_per_pixel")

        if geometry.line_length == geometry.row_pixel_bytes and geometry.xoffset_bytes == 0:
            return pixel_rows.tobytes(order="C"), "numpy"

        output = np.zeros((geometry.height, geometry.line_length), dtype=np.uint8)
        output[:, geometry.xoffset_bytes : geometry.xoffset_bytes + geometry.row_pixel_bytes] = pixel_rows
        return output.tobytes(order="C"), "numpy"

    def _pack_framebuffer_bytes_generic(self, image: object, geometry: FramebufferGeometry) -> tuple[bytes, str]:
        rgba_image = image.convert("RGBA")
        pixels = rgba_image.load()
        output = bytearray(geometry.frame_size_bytes)
        for row in range(geometry.height):
            row_start = (row * geometry.line_length) + geometry.xoffset_bytes
            for column in range(geometry.width):
                red, green, blue, alpha = pixels[column, row]
                packed = 0
                for value, color_field in (
                    (red, geometry.red),
                    (green, geometry.green),
                    (blue, geometry.blue),
                    (alpha, geometry.transp),
                ):
                    if color_field.length <= 0:
                        continue
                    scaled = ((int(value) * ((1 << color_field.length) - 1)) + 127) // 255
                    packed |= scaled << color_field.offset
                byte_offset = row_start + (column * geometry.bytes_per_pixel)
                output[byte_offset : byte_offset + geometry.bytes_per_pixel] = packed.to_bytes(
                    geometry.bytes_per_pixel,
                    "little",
                )
        return bytes(output), "generic"

    def _pillow_raw_mode_for_geometry(self, geometry: FramebufferGeometry) -> tuple[str, str] | None:
        bytes_per_pixel = geometry.bytes_per_pixel
        if bytes_per_pixel not in {3, 4}:
            return None
        positions = ["X"] * bytes_per_pixel
        for letter, color_field in (("R", geometry.red), ("G", geometry.green), ("B", geometry.blue), ("A", geometry.transp)):
            if color_field.length <= 0:
                continue
            if color_field.length != 8 or color_field.offset % 8 != 0:
                return None
            byte_index = color_field.offset // 8
            if byte_index < 0 or byte_index >= bytes_per_pixel:
                return None
            if positions[byte_index] != "X" and positions[byte_index] != letter:
                return None
            positions[byte_index] = letter
        raw_mode = "".join(positions)
        if raw_mode not in _SUPPORTED_PILLOW_RAW_MODES:
            return None
        if bytes_per_pixel == 3:
            return ("RGB", raw_mode)
        if "A" in raw_mode:
            return ("RGBA", raw_mode)
        return ("RGB", raw_mode)

    def _report_pack_backend_once(self, backend: str, geometry: FramebufferGeometry) -> None:
        if self._reported_pack_backend == backend:
            return
        self._reported_pack_backend = backend
        self._safe_emit(
            " ".join(
                (
                    f"display_framebuffer_pack={backend}",
                    f"bpp={geometry.bits_per_pixel}",
                    f"stride={geometry.line_length}",
                )
            )
        )

    def _write_framebuffer_payload(self, payload: bytes, geometry: FramebufferGeometry) -> None:
        spans = self._changed_row_spans(payload, self._last_payload, geometry)
        if spans == ():
            return
        framebuffer_map = self._framebuffer_mapping(geometry)
        if framebuffer_map is not None:
            self._report_io_backend_once("mmap", geometry)
            self._write_payload_to_mmap(framebuffer_map, payload, geometry, spans)
            return
        self._report_io_backend_once("write", geometry)
        fd = self._open_framebuffer_fd()
        self._write_payload_to_fd(fd, payload, geometry, spans)

    def _framebuffer_mapping(self, geometry: FramebufferGeometry) -> mmap.mmap | None:
        if not self._prefer_mmap():
            return None
        existing = self._framebuffer_map
        if existing is not None:
            if self._framebuffer_map_length >= geometry.mapping_size_bytes:
                return existing
            self._close_framebuffer_mapping()
        fd = self._open_framebuffer_fd()
        try:
            framebuffer_map = mmap.mmap(fd, geometry.mapping_size_bytes, access=mmap.ACCESS_WRITE)
        except (BufferError, OSError, ValueError):
            return None
        self._framebuffer_map = framebuffer_map
        self._framebuffer_map_length = geometry.mapping_size_bytes
        return framebuffer_map

    def _prefer_mmap(self) -> bool:
        return str(os.environ.get("TWINR_DISPLAY_DISABLE_MMAP", "")).strip().lower() not in {"1", "true", "yes", "on"}

    def _write_payload_to_mmap(
        self,
        framebuffer_map: mmap.mmap,
        payload: bytes,
        geometry: FramebufferGeometry,
        spans: tuple[tuple[int, int], ...] | None,
    ) -> None:
        base_offset = geometry.visible_offset_bytes
        if spans is None:
            framebuffer_map[base_offset : base_offset + len(payload)] = payload
            return
        payload_view = memoryview(payload)
        for start_row, end_row in spans:
            source_start = start_row * geometry.line_length
            source_end = end_row * geometry.line_length
            target_start = base_offset + source_start
            framebuffer_map[target_start : target_start + (source_end - source_start)] = payload_view[source_start:source_end]

    def _write_payload_to_fd(
        self,
        fd: int,
        payload: bytes,
        geometry: FramebufferGeometry,
        spans: tuple[tuple[int, int], ...] | None,
    ) -> None:
        base_offset = geometry.visible_offset_bytes
        payload_view = memoryview(payload)
        if spans is None:
            os.lseek(fd, base_offset, os.SEEK_SET)
            self._write_all(fd, payload_view)
            return
        for start_row, end_row in spans:
            source_start = start_row * geometry.line_length
            source_end = end_row * geometry.line_length
            os.lseek(fd, base_offset + source_start, os.SEEK_SET)
            self._write_all(fd, payload_view[source_start:source_end])

    def _changed_row_spans(
        self,
        payload: bytes,
        previous_payload: bytes | None,
        geometry: FramebufferGeometry,
    ) -> tuple[tuple[int, int], ...] | None:
        if self._force_full_refresh():
            return None
        if previous_payload is None or len(previous_payload) != len(payload):
            return None
        if payload == previous_payload:
            return ()
        stride = geometry.line_length
        current_view = memoryview(payload)
        previous_view = memoryview(previous_payload)
        spans: list[tuple[int, int]] = []
        start_row: int | None = None
        changed_rows = 0
        for row in range(geometry.height):
            row_start = row * stride
            row_end = row_start + stride
            if current_view[row_start:row_end] != previous_view[row_start:row_end]:
                changed_rows += 1
                if start_row is None:
                    start_row = row
                continue
            if start_row is not None:
                spans.append((start_row, row))
                start_row = None
        if start_row is not None:
            spans.append((start_row, geometry.height))
        if not spans:
            return ()
        if changed_rows * 4 >= geometry.height * 3:
            return None
        return tuple(spans)

    def _force_full_refresh(self) -> bool:
        return str(os.environ.get("TWINR_DISPLAY_FORCE_FULL_REFRESH", "")).strip().lower() in {"1", "true", "yes", "on"}

    def _report_io_backend_once(self, backend: str, geometry: FramebufferGeometry) -> None:
        if self._reported_io_backend == backend:
            return
        self._reported_io_backend = backend
        self._safe_emit(
            " ".join(
                (
                    f"display_framebuffer_io={backend}",
                    f"offset={geometry.visible_offset_bytes}",
                    f"stride={geometry.line_length}",
                )
            )
        )

    def _write_all(self, fd: int, payload: memoryview) -> None:
        written_total = 0
        while written_total < len(payload):
            written = os.write(fd, payload[written_total:])
            if written <= 0:
                raise OSError("short framebuffer write")
            written_total += written

    def _reset_framebuffer_state(self, *, clear_geometry: bool) -> None:
        self._close_framebuffer_mapping()
        framebuffer_fd = self._framebuffer_fd
        self._framebuffer_fd = None
        self._last_payload = None
        self._reported_io_backend = None
        if clear_geometry:
            self._geometry = None
        if framebuffer_fd is None:
            return
        try:
            os.close(framebuffer_fd)
        except OSError:
            return

    def _close_framebuffer_mapping(self) -> None:
        framebuffer_map = self._framebuffer_map
        self._framebuffer_map = None
        self._framebuffer_map_length = 0
        if framebuffer_map is None:
            return
        try:
            framebuffer_map.close()
        except Exception:
            return

    def _safe_emit(self, line: str) -> None:
        compact = self._normalise_text(line, fallback="")
        if not compact or self.emit is None:
            return
        try:
            self.emit(compact[:160])
        except Exception:
            return
