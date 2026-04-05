# CHANGELOG: 2026-03-28
# BUG-1: Fixed symbol/headline overlap when panel.symbol is present without panel.eyebrow.
# BUG-2: Fixed emoji glyph and halo overflow that could bleed outside small reserve boxes.
# BUG-3: Added the missing "alert" reserve-card accent so alert panels no longer render as neutral.
# BUG-4: Avoided silent loss of image-card placeholders when image_data_url is present but no image surface is supplied.
# SEC-1: Added strict inline-image validation, decompression-bomb guards, MIME allowlisting, byte/pixel caps, and safe bounded compositing for data-URL images.
# IMP-1: Added height-aware adaptive text fitting for text and image reserve cards.
# IMP-2: Added small LRU caches for rendered emoji glyphs and decoded inline images to reduce repeated work on Raspberry Pi 4.
# IMP-3: Added bounded rounded-mask image placement and fallback placeholders to keep reserve rendering deterministic on constrained HDMI scenes.
# BREAKING: Oversized or unsupported inline images are now rejected and render a placeholder instead of being forwarded blindly to downstream decoders.

"""Reserve-panel rendering helpers for the default HDMI scene."""

from __future__ import annotations

import base64
import binascii
import io
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import unquote_to_bytes

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from .models import (
    _PROMPT_MODE_LINE_GAP,
    _PROMPT_MODE_SECTION_GAP,
    HdmiStatusPanelModel,
)
from .typing_contracts import (
    HdmiEmojiCueLike,
    HdmiImageSurface,
    HdmiPanelDrawSurface,
)

# BREAKING: inline reserve images are now restricted to bounded raster MIME types.
_ALLOWED_INLINE_IMAGE_MIME_TYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
        "image/bmp",
        "image/avif",
    }
)
_INLINE_IMAGE_MAX_DATA_URL_CHARS = 2_000_000
_INLINE_IMAGE_MAX_DECODED_BYTES = 1_500_000
_INLINE_IMAGE_MAX_PIXELS = 2_097_152
_INLINE_IMAGE_MAX_DIMENSION = 2048
_INLINE_IMAGE_CACHE_SIZE = 8
_EMOJI_GLYPH_CACHE_SIZE = 64
_CACHED_INLINE_IMAGE_MAX_SIDE = 1024
_MIN_QR_SIDE = 72


@dataclass(slots=True)
class _PanelTextLayout:
    headline_font: object
    body_font: object
    headline_lines: list[str]
    body_lines: list[str]


@dataclass(slots=True)
class _ReservePanelCacheState:
    lock: threading.RLock = field(default_factory=threading.RLock)
    emoji_glyphs: OrderedDict[tuple[str, int], HdmiImageSurface] = field(default_factory=OrderedDict)
    inline_images: OrderedDict[str, HdmiImageSurface | None] = field(default_factory=OrderedDict)


class HdmiReservePanelRenderingMixin:
    """Render the right-hand reserve area and its bounded reserve-card modes."""

    def _draw_emoji_reserve(
        self,
        image: HdmiImageSurface,
        draw: HdmiPanelDrawSurface,
        *,
        box: tuple[int, int, int, int],
        emoji_cue: HdmiEmojiCueLike,
    ) -> None:
        """Draw one real Unicode emoji in the reserved right-hand HDMI area."""

        del draw
        left, top, right, bottom, width, height = self._box_metrics(box)
        if width <= 0 or height <= 0:
            return

        inset = min(14, max(4, min(width, height) // 10))
        max_glyph_side = min(width, height) - (inset * 2)
        if max_glyph_side < 16:
            return
        target_size = min(148, max_glyph_side)

        emoji_image = self._cached_emoji_glyph(emoji_cue.glyph(), target_size)
        if emoji_image is None:
            return

        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay, "RGBA")

        halo_padding = min(13, max(4, min(width, height) // 12))
        halo_size = min(width, height) - (halo_padding * 2)
        if halo_size > 8:
            halo_left = max(0, (width - halo_size) // 2)
            halo_top = max(0, (height - halo_size) // 2)
            overlay_draw.ellipse(
                (halo_left, halo_top, halo_left + halo_size, halo_top + halo_size),
                fill=self._emoji_accent_fill(emoji_cue.accent),
            )

        fitted = emoji_image
        if fitted.width > max_glyph_side or fitted.height > max_glyph_side:
            fitted = ImageOps.contain(
                fitted,
                (max_glyph_side, max_glyph_side),
                method=Image.Resampling.LANCZOS,
            )
        paste_left = max(0, (width - fitted.width) // 2)
        paste_top = max(0, (height - fitted.height) // 2)
        overlay.paste(fitted, (paste_left, paste_top), fitted)
        image.paste(overlay, (left, top), overlay)

    def _emoji_accent_fill(self, accent: str) -> tuple[int, int, int, int]:
        """Return a soft reserve-halo color for one emoji accent token."""

        mapping = {
            "neutral": (255, 255, 255, 18),
            "info": (90, 132, 196, 38),
            "success": (92, 232, 148, 40),
            "warm": (255, 179, 87, 42),
            "alert": (244, 114, 90, 42),
        }
        return mapping.get(self._normalize_accent(accent), (255, 255, 255, 18))

    def _draw_status_panel(
        self,
        draw: HdmiPanelDrawSurface,
        *,
        image: HdmiImageSurface | None = None,
        box: tuple[int, int, int, int],
        panel: HdmiStatusPanelModel,
        compact: bool,
    ) -> None:
        """Draw one small calm reserve card for an active ambient impulse cue."""

        left, top, right, bottom, width, height = self._box_metrics(box)
        if width <= 0 or height <= 0:
            return

        radius = self._clamped_radius(width, height, 18 if compact else 24)
        normalized_box = (left, top, right, bottom)

        if not self._panel_has_visible_content(panel):
            draw.rounded_rectangle(
                normalized_box,
                radius=radius,
                fill=(0, 0, 0),
                outline=(94, 100, 112),
                width=2,
            )
            return

        draw.rounded_rectangle(
            normalized_box,
            radius=radius,
            fill=(0, 0, 0),
            outline=(255, 255, 255),
            width=2,
        )
        accent_fill = self._panel_accent_fill(panel.accent)

        if panel.image_data_url:
            self._draw_status_panel_with_image(
                image=image,
                draw=draw,
                box=normalized_box,
                panel=panel,
                compact=compact,
                accent_fill=accent_fill,
            )
            return

        prompt_mode = bool(panel.prompt_mode)
        if not prompt_mode:
            self._draw_panel_accent_marker(draw, box=normalized_box, accent_fill=accent_fill, compact=compact)

        eyebrow_text = self._bounded_text(panel.eyebrow, 48)
        headline_text = self._bounded_text(panel.headline, 160)
        helper_text = self._bounded_text(panel.helper_text, 280)
        symbol_text = self._bounded_text(panel.symbol, 12)

        eyebrow_font = self.tools._font(13 if compact else 15, bold=True)
        padding_x = 16 if compact else 20
        padding_y = 16 if compact else 18
        text_left = left + padding_x
        text_top = top + padding_y
        inner_width = max(56, width - (padding_x * 2))

        if symbol_text and not prompt_mode:
            symbol_font = self.tools._font(22 if compact else 26, bold=False)
            symbol_width = int(round(self.tools._text_width(draw, symbol_text, font=symbol_font)))
            symbol_height = self.tools._text_height(draw, font=symbol_font)
            draw.text((text_left, text_top - 2), symbol_text, fill=accent_fill, font=symbol_font)
            if eyebrow_text:
                eyebrow_x = text_left + symbol_width + 10
            elif inner_width - symbol_width - 10 >= 96:
                text_left += symbol_width + 10
                inner_width = max(56, inner_width - symbol_width - 10)
                eyebrow_x = text_left
            else:
                text_top += symbol_height + 8
                eyebrow_x = text_left
        else:
            eyebrow_x = text_left

        if eyebrow_text:
            draw.text((eyebrow_x, text_top), eyebrow_text, fill=(182, 182, 182), font=eyebrow_font)
            text_top += self.tools._text_height(draw, font=eyebrow_font) + (12 if compact else 14)

        if prompt_mode:
            prompt_layout = self._fit_prompt_mode_layout(
                draw,
                panel=panel,
                inner_width=inner_width,
                available_top=text_top,
                available_bottom=bottom - padding_y,
                compact=compact,
            )
            headline_font = prompt_layout.headline_font
            body_font = prompt_layout.body_font
            headline_lines = prompt_layout.headline_lines
            body_lines = prompt_layout.body_lines
        else:
            layout = self._fit_text_block_layout(
                draw,
                headline=headline_text,
                helper_text=helper_text,
                max_width=inner_width,
                available_height=max(0, (bottom - padding_y) - text_top),
                compact=compact,
                candidate_sizes=self._text_layout_candidates(compact=compact, image_card=False),
                max_headline_lines=2,
                max_body_lines=2 if compact else 3,
                line_gap=2,
                section_gap=6,
            )
            headline_font = layout.headline_font
            body_font = layout.body_font
            headline_lines = layout.headline_lines
            body_lines = layout.body_lines

        line_gap = _PROMPT_MODE_LINE_GAP if prompt_mode else 2
        for index, line in enumerate(headline_lines):
            draw.text((text_left, text_top), line, fill=(255, 255, 255), font=headline_font)
            text_top += self.tools._text_height(draw, font=headline_font)
            if index < len(headline_lines) - 1:
                text_top += line_gap
        if body_lines:
            text_top += _PROMPT_MODE_SECTION_GAP if prompt_mode else 6
        for index, line in enumerate(body_lines):
            draw.text(
                (text_left, text_top),
                line,
                fill=(255, 255, 255) if prompt_mode else (214, 214, 214),
                font=body_font,
            )
            text_top += self.tools._text_height(draw, font=body_font)
            if index < len(body_lines) - 1:
                text_top += line_gap

    def _draw_status_panel_with_image(
        self,
        image: HdmiImageSurface | None,
        draw: HdmiPanelDrawSurface,
        *,
        box: tuple[int, int, int, int],
        panel: HdmiStatusPanelModel,
        compact: bool,
        accent_fill: tuple[int, int, int],
    ) -> None:
        """Draw one reserve card that includes a pairing QR or other bounded image."""

        left, top, right, bottom, width, height = self._box_metrics(box)
        if width <= 0 or height <= 0:
            return

        padding_x = 16 if compact else 20
        padding_y = 16 if compact else 18
        text_left = left + padding_x
        text_top = top + padding_y
        inner_width = max(56, width - (padding_x * 2))
        eyebrow_font = self.tools._font(13 if compact else 15, bold=True)

        self._draw_panel_accent_marker(draw, box=box, accent_fill=accent_fill, compact=compact)

        eyebrow_text = self._bounded_text(panel.eyebrow, 48)
        headline_text = self._bounded_text(panel.headline, 140)
        helper_text = self._bounded_text(panel.helper_text, 220)

        if eyebrow_text:
            draw.text((text_left, text_top), eyebrow_text, fill=(182, 182, 182), font=eyebrow_font)
            text_top += self.tools._text_height(draw, font=eyebrow_font) + (10 if compact else 12)

        layout = self._fit_image_card_layout(
            draw,
            headline=headline_text,
            helper_text=helper_text,
            max_width=inner_width,
            available_height=max(0, (bottom - padding_y) - text_top),
            compact=compact,
        )

        headline_font = layout.headline_font
        body_font = layout.body_font
        headline_lines = layout.headline_lines
        body_lines = layout.body_lines

        for index, line in enumerate(headline_lines):
            draw.text((text_left, text_top), line, fill=(255, 255, 255), font=headline_font)
            text_top += self.tools._text_height(draw, font=headline_font)
            if index < len(headline_lines) - 1:
                text_top += 2
        if body_lines:
            text_top += 6
        for index, line in enumerate(body_lines):
            draw.text((text_left, text_top), line, fill=(214, 214, 214), font=body_font)
            text_top += self.tools._text_height(draw, font=body_font)
            if index < len(body_lines) - 1:
                text_top += 2

        available_top = text_top + 12
        available_bottom = bottom - padding_y
        available_height = max(0, available_bottom - available_top)
        qr_side = min(inner_width, available_height)
        if qr_side < _MIN_QR_SIDE:
            return

        qr_left = left + max(0, ((right - left) - qr_side) // 2)
        qr_top = available_bottom - qr_side
        qr_box = (qr_left, qr_top, qr_left + qr_side, qr_top + qr_side)
        qr_radius = self._clamped_radius(qr_side, qr_side, 16 if compact else 18)
        draw.rounded_rectangle(
            qr_box,
            radius=qr_radius,
            fill=(255, 255, 255),
            outline=(84, 84, 84),
            width=2,
        )
        if image is not None and self._paste_inline_image_safe(
            image,
            box=qr_box,
            image_data_url=panel.image_data_url,
            corner_radius=qr_radius,
        ):
            return

        placeholder_font = self.tools._font(14 if compact else 18, bold=True)
        placeholder = "QR UNAVAILABLE"
        draw.text(
            ((qr_box[0] + qr_box[2]) // 2, (qr_box[1] + qr_box[3]) // 2),
            placeholder,
            fill=(42, 42, 42),
            font=placeholder_font,
            anchor="mm",
        )

    def _panel_accent_fill(self, accent: str) -> tuple[int, int, int]:
        """Return a calm accent color for the ambient reserve card."""

        mapping = {
            "neutral": (224, 224, 224),
            "info": (112, 168, 255),
            "success": (116, 242, 170),
            "warm": (255, 190, 98),
            "alert": (255, 129, 102),
        }
        return mapping.get(self._normalize_accent(accent), (224, 224, 224))

    def _draw_panel_accent_marker(
        self,
        draw: HdmiPanelDrawSurface,
        *,
        box: tuple[int, int, int, int],
        accent_fill: tuple[int, int, int],
        compact: bool,
    ) -> None:
        left, top, right, bottom, width, height = self._box_metrics(box)
        if width <= 0 or height <= 0:
            return
        marker_size = 8 if compact else 10
        inset = 12
        marker_size = min(marker_size, max(4, width - (inset * 2)), max(4, height - (inset * 2)))
        marker_left = min(max(left + inset, left), right - marker_size)
        marker_top = min(max(top + inset, top), bottom - marker_size)
        if marker_left >= right or marker_top >= bottom:
            return
        draw.rounded_rectangle(
            (marker_left, marker_top, marker_left + marker_size, marker_top + marker_size),
            radius=max(2, marker_size // 2),
            fill=accent_fill,
        )

    def _fit_image_card_layout(
        self,
        draw: HdmiPanelDrawSurface,
        *,
        headline: str,
        helper_text: str,
        max_width: int,
        available_height: int,
        compact: bool,
    ) -> _PanelTextLayout:
        return self._fit_text_block_layout(
            draw,
            headline=headline,
            helper_text=helper_text,
            max_width=max_width,
            available_height=available_height,
            compact=compact,
            candidate_sizes=self._text_layout_candidates(compact=compact, image_card=True),
            max_headline_lines=2,
            max_body_lines=2,
            line_gap=2,
            section_gap=6,
            reserve_bottom_space=_MIN_QR_SIDE + 12,
        )

    def _fit_text_block_layout(
        self,
        draw: HdmiPanelDrawSurface,
        *,
        headline: str,
        helper_text: str,
        max_width: int,
        available_height: int,
        compact: bool,
        candidate_sizes: Iterable[tuple[int, int]],
        max_headline_lines: int,
        max_body_lines: int,
        line_gap: int,
        section_gap: int,
        reserve_bottom_space: int = 0,
    ) -> _PanelTextLayout:
        available_height = max(0, available_height)
        best: _PanelTextLayout | None = None
        best_overflow: int | None = None

        for headline_size, body_size in candidate_sizes:
            headline_font = self.tools._font(headline_size, bold=True)
            body_font = self.tools._font(body_size, bold=False)
            headline_lines = self.tools._wrapped_lines(
                draw,
                (headline,),
                max_width=max_width,
                font=headline_font,
                max_lines=max_headline_lines,
            )
            body_lines = self.tools._wrapped_lines(
                draw,
                (helper_text,),
                max_width=max_width,
                font=body_font,
                max_lines=max_body_lines,
            )
            layout = _PanelTextLayout(
                headline_font=headline_font,
                body_font=body_font,
                headline_lines=headline_lines,
                body_lines=body_lines,
            )
            text_height = self._text_layout_height(
                draw,
                layout,
                line_gap=line_gap,
                section_gap=section_gap,
            )
            overflow = max(0, (text_height + reserve_bottom_space) - available_height)
            if overflow == 0:
                return layout
            if best is None or best_overflow is None or overflow < best_overflow:
                best = layout
                best_overflow = overflow

        if best is None:
            fallback_headline_font = self.tools._font(16 if compact else 18, bold=True)
            fallback_body_font = self.tools._font(12 if compact else 13, bold=False)
            return _PanelTextLayout(
                headline_font=fallback_headline_font,
                body_font=fallback_body_font,
                headline_lines=[],
                body_lines=[],
            )

        reduced = self._reduce_layout_to_fit(
            draw,
            layout=best,
            max_width=max_width,
            headline=headline,
            helper_text=helper_text,
            available_height=available_height,
            line_gap=line_gap,
            section_gap=section_gap,
            reserve_bottom_space=reserve_bottom_space,
        )
        return reduced

    def _reduce_layout_to_fit(
        self,
        draw: HdmiPanelDrawSurface,
        *,
        layout: _PanelTextLayout,
        max_width: int,
        headline: str,
        helper_text: str,
        available_height: int,
        line_gap: int,
        section_gap: int,
        reserve_bottom_space: int,
    ) -> _PanelTextLayout:
        headline_font = layout.headline_font
        body_font = layout.body_font

        options = [
            (2, len(layout.body_lines)),
            (2, 1),
            (2, 0),
            (1, 1),
            (1, 0),
            (0, 0),
        ]
        for headline_max_lines, body_max_lines in options:
            headline_lines = (
                self.tools._wrapped_lines(
                    draw,
                    (headline,),
                    max_width=max_width,
                    font=headline_font,
                    max_lines=headline_max_lines,
                )
                if headline_max_lines > 0
                else []
            )
            body_lines = (
                self.tools._wrapped_lines(
                    draw,
                    (helper_text,),
                    max_width=max_width,
                    font=body_font,
                    max_lines=body_max_lines,
                )
                if body_max_lines > 0
                else []
            )
            candidate = _PanelTextLayout(
                headline_font=headline_font,
                body_font=body_font,
                headline_lines=headline_lines,
                body_lines=body_lines,
            )
            if (
                self._text_layout_height(draw, candidate, line_gap=line_gap, section_gap=section_gap)
                + reserve_bottom_space
                <= available_height
            ):
                return candidate
        return _PanelTextLayout(
            headline_font=headline_font,
            body_font=body_font,
            headline_lines=layout.headline_lines[:1],
            body_lines=[],
        )

    def _text_layout_height(
        self,
        draw: HdmiPanelDrawSurface,
        layout: _PanelTextLayout,
        *,
        line_gap: int,
        section_gap: int,
    ) -> int:
        height = 0
        if layout.headline_lines:
            headline_line_height = self.tools._text_height(draw, font=layout.headline_font)
            height += headline_line_height * len(layout.headline_lines)
            if len(layout.headline_lines) > 1:
                height += line_gap * (len(layout.headline_lines) - 1)
        if layout.body_lines:
            if layout.headline_lines:
                height += section_gap
            body_line_height = self.tools._text_height(draw, font=layout.body_font)
            height += body_line_height * len(layout.body_lines)
            if len(layout.body_lines) > 1:
                height += line_gap * (len(layout.body_lines) - 1)
        return height

    def _text_layout_candidates(self, *, compact: bool, image_card: bool) -> tuple[tuple[int, int], ...]:
        if image_card:
            return (
                (20, 13),
                (18, 12),
                (16, 12),
                (15, 11),
            ) if compact else (
                (24, 15),
                (22, 14),
                (20, 13),
                (18, 12),
                (16, 12),
            )
        return (
            (22, 15),
            (20, 14),
            (18, 13),
            (16, 12),
        ) if compact else (
            (28, 18),
            (26, 17),
            (24, 16),
            (22, 15),
            (20, 14),
            (18, 13),
        )

    def _paste_inline_image_safe(
        self,
        image: HdmiImageSurface,
        *,
        box: tuple[int, int, int, int],
        image_data_url: str | None,
        corner_radius: int,
    ) -> bool:
        if not image_data_url:
            return False

        decoded = self._decoded_inline_image(image_data_url)
        if decoded is None:
            return False

        left, top, right, bottom, width, height = self._box_metrics(box)
        if width <= 0 or height <= 0:
            return False

        inset = min(12, max(6, min(width, height) // 12))
        target_size = (max(1, width - (inset * 2)), max(1, height - (inset * 2)))
        fitted = ImageOps.contain(
            decoded.copy(),
            target_size,
            method=self._inline_image_resample(decoded),
        )

        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        paste_left = max(0, (width - fitted.width) // 2)
        paste_top = max(0, (height - fitted.height) // 2)
        overlay.paste(fitted, (paste_left, paste_top), fitted)

        mask = Image.new("L", (width, height), 0)
        mask_draw = ImageDraw.Draw(mask)
        inner_radius = self._clamped_radius(width, height, max(2, corner_radius - 2))
        mask_draw.rounded_rectangle((0, 0, width, height), radius=inner_radius, fill=255)

        clipped = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        clipped.paste(overlay, (0, 0), mask)
        image.paste(clipped, (left, top), clipped)
        return True

    def _decoded_inline_image(self, image_data_url: str) -> HdmiImageSurface | None:
        if not image_data_url or len(image_data_url) > _INLINE_IMAGE_MAX_DATA_URL_CHARS:
            return None

        cache_state = self._reserve_panel_cache_state()
        with cache_state.lock:
            cached = cache_state.inline_images.get(image_data_url)
            if cached is not None or image_data_url in cache_state.inline_images:
                cache_state.inline_images.move_to_end(image_data_url)
                return cached.copy() if cached is not None else None

        decoded = self._decode_inline_image_uncached(image_data_url)

        with cache_state.lock:
            cache_state.inline_images[image_data_url] = decoded.copy() if decoded is not None else None
            cache_state.inline_images.move_to_end(image_data_url)
            while len(cache_state.inline_images) > _INLINE_IMAGE_CACHE_SIZE:
                cache_state.inline_images.popitem(last=False)

        return decoded

    def _decode_inline_image_uncached(self, image_data_url: str) -> HdmiImageSurface | None:
        media_type, payload = self._decode_data_url(image_data_url)
        if media_type is None or payload is None:
            return None
        if media_type not in _ALLOWED_INLINE_IMAGE_MIME_TYPES:
            # BREAKING: vector/unknown inline image types are rejected here instead of being delegated blindly.
            return None
        if len(payload) > _INLINE_IMAGE_MAX_DECODED_BYTES:
            return None

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(payload)) as source:
                    width, height = source.size
                    if width <= 0 or height <= 0:
                        return None
                    if width > _INLINE_IMAGE_MAX_DIMENSION or height > _INLINE_IMAGE_MAX_DIMENSION:
                        return None
                    if width * height > _INLINE_IMAGE_MAX_PIXELS:
                        return None
                    if getattr(source, "is_animated", False):
                        source.seek(0)
                    prepared = ImageOps.exif_transpose(source)
                    rgba = prepared.convert("RGBA")
        except (
            OSError,
            ValueError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
            UnidentifiedImageError,
        ):
            return None

        if rgba.width > _CACHED_INLINE_IMAGE_MAX_SIDE or rgba.height > _CACHED_INLINE_IMAGE_MAX_SIDE:
            rgba = ImageOps.contain(
                rgba,
                (_CACHED_INLINE_IMAGE_MAX_SIDE, _CACHED_INLINE_IMAGE_MAX_SIDE),
                method=Image.Resampling.LANCZOS,
            )
        return rgba

    def _decode_data_url(self, image_data_url: str) -> tuple[str | None, bytes | None]:
        if image_data_url[:5].lower() != "data:":
            return None, None
        try:
            header, encoded = image_data_url[5:].split(",", 1)
        except ValueError:
            return None, None

        media_type = header.split(";", 1)[0].strip().lower() or "text/plain"
        is_base64 = ";base64" in header.lower()
        try:
            if is_base64:
                payload = base64.b64decode(encoded.strip(), validate=True)
            else:
                payload = unquote_to_bytes(encoded)
        except (binascii.Error, ValueError):
            return None, None
        return media_type, payload

    def _inline_image_resample(self, image: HdmiImageSurface) -> int:
        if image.mode in {"1", "L", "P"}:
            color_count = image.getcolors(maxcolors=8)
            if color_count is not None and len(color_count) <= 4:
                return Image.Resampling.NEAREST
        return Image.Resampling.LANCZOS

    def _cached_emoji_glyph(self, glyph: str, target_size: int) -> HdmiImageSurface | None:
        if not glyph:
            return None
        key = (glyph, target_size)
        cache_state = self._reserve_panel_cache_state()
        with cache_state.lock:
            cached = cache_state.emoji_glyphs.get(key)
            if cached is not None:
                cache_state.emoji_glyphs.move_to_end(key)
                return cached

        emoji_image = self.tools._render_emoji_glyph(glyph, target_size=target_size)
        if emoji_image is None:
            return None

        with cache_state.lock:
            cache_state.emoji_glyphs[key] = emoji_image
            cache_state.emoji_glyphs.move_to_end(key)
            while len(cache_state.emoji_glyphs) > _EMOJI_GLYPH_CACHE_SIZE:
                cache_state.emoji_glyphs.popitem(last=False)
        return emoji_image

    def _reserve_panel_cache_state(self) -> _ReservePanelCacheState:
        state = getattr(self, "_hdmi_reserve_panel_cache_state", None)
        if state is None:
            state = _ReservePanelCacheState()
            setattr(self, "_hdmi_reserve_panel_cache_state", state)
        return state

    def _panel_has_visible_content(self, panel: HdmiStatusPanelModel) -> bool:
        return any(
            (
                panel.eyebrow,
                panel.headline,
                panel.helper_text,
                panel.symbol,
                panel.image_data_url,
            )
        )

    def _bounded_text(self, value: str | None, limit: int) -> str:
        if not value:
            return ""
        collapsed = " ".join(str(value).split())
        if len(collapsed) <= limit:
            return collapsed
        return collapsed[: max(1, limit - 1)].rstrip() + "…"

    def _normalize_accent(self, accent: str | None) -> str:
        return (accent or "neutral").strip().lower()

    def _box_metrics(self, box: tuple[int, int, int, int]) -> tuple[int, int, int, int, int, int]:
        left, top, right, bottom = box
        width = max(0, right - left)
        height = max(0, bottom - top)
        return left, top, right, bottom, width, height

    def _clamped_radius(self, width: int, height: int, preferred: int) -> int:
        if width <= 0 or height <= 0:
            return 0
        return max(0, min(preferred, width // 2, height // 2))