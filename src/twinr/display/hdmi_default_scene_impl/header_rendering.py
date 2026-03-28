# mypy: disable-error-code="attr-defined,call-overload,arg-type"
# CHANGELOG: 2026-03-28
# BUG-1: Signal pills could exceed the available row width and render outside the header on narrow boxes or long labels.
# BUG-2: Header and ticker text could overflow their bounding boxes because width budgeting was not clamped end-to-end.
# SEC-1: Untrusted remote strings (e.g. ticker/news/status feeds) were rendered without control-character hygiene or length caps, enabling practical UI spoofing and render-loop DoS on a Pi 4.
# IMP-1: Added bounded text-measurement/truncation caches and binary-search fitting to cut per-frame CPU cost on Raspberry Pi 4.
# IMP-2: Added responsive slot-based layout, grapheme-aware truncation when `regex` is installed, and optional RTL-aware drawing on Pillow builds with libraqm.

"""Header and ticker rendering helpers for the default HDMI scene."""

from __future__ import annotations

import unicodedata
from typing import Any

try:
    import regex as _regex
except Exception:  # pragma: no cover - optional frontier dependency
    _regex = None

from .models import (
    _DEFAULT_HEADER_SIGNAL_ROWS,
    HdmiHeaderModel,
    HdmiNewsTickerModel,
)
from .typing_contracts import HdmiHeaderSignalLike, HdmiPanelDrawSurface

_RENDER_CACHE_LIMIT = 1024
_MAX_UI_TEXT_GRAPHEMES = 512
_MAX_SIGNAL_LABEL_GRAPHEMES = 64
_ELLIPSIS = "…"
_BIDI_OVERRIDE_CHARS = {
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",
    "\u2066",
    "\u2067",
    "\u2068",
    "\u2069",
}
_RTL_BIDI_CLASSES = {"R", "AL", "AN"}


class HdmiHeaderRenderingMixin:
    """Render the bounded HDMI header and bottom ticker surfaces."""

    def _draw_twinr_header(
        self,
        draw: HdmiPanelDrawSurface,
        *,
        box: tuple[int, int, int, int],
        header: HdmiHeaderModel,
    ) -> None:
        left, top, right, bottom = box
        if right <= left or bottom <= top:
            return

        box_height = max(32, bottom - top)
        main_row_height = box_height
        debug_lane_top = bottom
        debug_lane_bottom = bottom
        signal_row_count = max(0, _DEFAULT_HEADER_SIGNAL_ROWS)
        if signal_row_count > 0:
            signal_row_gap = 6
            signal_lane_padding_bottom = 8
            max_pill_height = 20 if box_height >= 84 else 18
            signal_lane_height = max(
                38,
                (max_pill_height * signal_row_count) + (signal_row_gap * max(0, signal_row_count - 1)),
            )
            debug_lane_bottom = bottom - signal_lane_padding_bottom
            debug_lane_top = max(top + 30, debug_lane_bottom - signal_lane_height)
            main_row_height = max(32, debug_lane_top - top - 4)

        header_font = self.tools._font(24 if main_row_height >= 50 else 18, bold=True)
        state_font = self.tools._font(22 if main_row_height >= 50 else 17, bold=True)
        time_font = self.tools._font(22 if main_row_height >= 50 else 16, bold=False)
        system_label_font = self.tools._font(12 if main_row_height >= 50 else 10, bold=True)
        system_value_font = self.tools._font(22 if main_row_height >= 50 else 18, bold=True)

        header_height = self._measure_text_height(draw, font=header_font)
        state_height = self._measure_text_height(draw, font=state_font)
        time_height = self._measure_text_height(draw, font=time_font)
        system_label_height = self._measure_text_height(draw, font=system_label_font)
        system_value_height = self._measure_text_height(draw, font=system_value_font)

        label_y = top + max(4, (main_row_height - header_height) // 2) - 1
        state_y = top + max(4, (main_row_height - state_height) // 2) - 1
        time_y = top + max(4, (main_row_height - time_height) // 2) - 1
        system_label_y = top + max(4, (main_row_height - system_label_height) // 2) - 2
        system_value_y = top + max(4, (main_row_height - system_value_height) // 2) - 1

        draw.rounded_rectangle(box, radius=20, fill=(0, 0, 0), outline=(255, 255, 255), width=2)

        brand_text = self._normalise_ui_text(header.brand, fallback="")
        state_text_raw = self._normalise_ui_text(header.state, fallback="")
        time_text = self._normalise_ui_text(header.time_value, fallback="")
        system_value_text = self._normalise_ui_text(header.system_value, fallback="")
        system_label_text = "SYSTEM"

        brand_x = left + 20
        brand_width = self._measure_text_width(draw, brand_text, font=header_font) if brand_text else 0
        if brand_text:
            self._draw_ui_text(draw, (brand_x, label_y), brand_text, fill=(255, 255, 255), font=header_font)

        time_width = self._measure_text_width(draw, time_text, font=time_font) if time_text else 0
        system_label_width = (
            self._measure_text_width(draw, system_label_text, font=system_label_font)
            if system_label_text
            else 0
        )
        system_value_width = (
            self._measure_text_width(draw, system_value_text, font=system_value_font)
            if system_value_text
            else 0
        )
        system_gap = 8 if box_height >= 50 else 6
        right_group_gap = 14 if box_height >= 50 else 10
        right_group_width = system_label_width + system_gap + system_value_width + right_group_gap + time_width
        system_x = right - 22 - right_group_width
        if system_label_text:
            self._draw_ui_text(
                draw,
                (system_x, system_label_y),
                system_label_text,
                fill=(204, 204, 204),
                font=system_label_font,
            )
        if system_value_text:
            self._draw_ui_text(
                draw,
                (system_x + system_label_width + system_gap, system_value_y),
                system_value_text,
                fill=header.system_accent,
                font=system_value_font,
            )
        if time_text:
            self._draw_ui_text(
                draw,
                (right - 22 - time_width, time_y),
                time_text,
                fill=(188, 188, 188),
                font=time_font,
            )

        state_left = left + 20 + brand_width + 26
        state_right = system_x - 24
        if state_right > state_left:
            state_text = self._fit_text(draw, state_text_raw, max_width=state_right - state_left, font=state_font)
            if state_text:
                state_width = self._measure_text_width(draw, state_text, font=state_font)
                state_x = state_left + max(0, ((state_right - state_left) - state_width) // 2)
                self._draw_ui_text(draw, (state_x, state_y), state_text, fill=(255, 255, 255), font=state_font)

        if signal_row_count > 0:
            self._draw_header_signal_lane(
                draw,
                left=left + 18,
                top=debug_lane_top,
                right=right - 18,
                bottom=debug_lane_bottom,
                signals=header.debug_signals,
            )

    def _draw_header_signal_lane(
        self,
        draw: HdmiPanelDrawSurface,
        *,
        left: int,
        top: int,
        right: int,
        bottom: int,
        signals: tuple[HdmiHeaderSignalLike, ...],
    ) -> None:
        """Draw a bounded wrapped signal lane under the HDMI status header."""

        max_rows = max(0, _DEFAULT_HEADER_SIGNAL_ROWS)
        if not signals or right <= left or bottom <= top or max_rows <= 0:
            return

        font = self.tools._font(12 if (bottom - top) >= 18 else 10, bold=True)
        text_height = self._measure_text_height(draw, font=font)
        gap = 8
        row_gap = 6
        lane_height = max(0, bottom - top)
        pill_height = max(
            16,
            min(
                max(16, (lane_height - (row_gap * max(0, max_rows - 1))) // max_rows),
                text_height + 8,
            ),
        )
        total_rows_height = (pill_height * max_rows) + (row_gap * max(0, max_rows - 1))
        y = top + max(0, (lane_height - total_rows_height) // 2)
        available_width = max(0, right - left)
        if available_width <= 0:
            return

        rendered_rows: list[list[tuple[str, tuple[int, int, int], int]]] = [[]]
        row_widths: list[int] = [0]
        overflow = 0

        for index, signal in enumerate(signals):
            label_raw = self._normalise_ui_text(
                signal.label,
                fallback=signal.key.upper(),
                uppercase=True,
                max_graphemes=_MAX_SIGNAL_LABEL_GRAPHEMES,
            )
            if not label_raw:
                continue
            label = label_raw
            pill_width = self._measure_text_width(draw, label, font=font) + 18
            if pill_width > available_width:
                label = self._fit_text(
                    draw,
                    label_raw,
                    max_width=max(0, available_width - 18),
                    font=font,
                    uppercase=True,
                    max_graphemes=_MAX_SIGNAL_LABEL_GRAPHEMES,
                )
                pill_width = self._measure_text_width(draw, label, font=font) + 18 if label else 0
            if not label or pill_width <= 0:
                continue
            color = self._header_signal_color(signal.accent)
            required_width = pill_width if not rendered_rows[-1] else pill_width + gap
            if row_widths[-1] + required_width <= available_width:
                rendered_rows[-1].append((label, color, pill_width))
                row_widths[-1] += required_width
                continue
            if len(rendered_rows) < max_rows:
                rendered_rows.append([(label, color, pill_width)])
                row_widths.append(pill_width)
                continue
            overflow = len(signals) - index
            break

        if overflow > 0:
            last_row = rendered_rows[-1]
            while True:
                overflow_label = f"+{overflow}"
                overflow_width = self._measure_text_width(draw, overflow_label, font=font) + 18
                required_width = overflow_width if not last_row else overflow_width + gap
                if row_widths[-1] + required_width <= available_width:
                    last_row.append((overflow_label, (214, 214, 214), overflow_width))
                    row_widths[-1] += required_width
                    break
                if not last_row:
                    break
                _removed_label, _removed_color, _removed_width = last_row.pop()
                overflow += 1
                row_widths[-1] = sum(item[2] for item in last_row) + (gap * max(0, len(last_row) - 1))

        for row_index, rendered_labels in enumerate(rendered_rows[:max_rows]):
            if not rendered_labels:
                continue
            x = left
            row_y = y + (row_index * (pill_height + row_gap))
            for label, color, pill_width in rendered_labels:
                draw.rounded_rectangle(
                    (x, row_y, x + pill_width, row_y + pill_height),
                    radius=min(11, pill_height // 2),
                    fill=(12, 12, 12),
                    outline=color,
                    width=2,
                )
                text_y = row_y + max(1, (pill_height - text_height) // 2) - 1
                self._draw_ui_text(draw, (x + 9, text_y), label, fill=color, font=font)
                x += pill_width + gap

    def _header_signal_color(self, accent: str) -> tuple[int, int, int]:
        """Return the header-pill outline/text color for one signal accent."""

        mapping = {
            "neutral": (214, 214, 214),
            "info": (114, 168, 255),
            "success": (116, 242, 170),
            "warning": (255, 196, 104),
            "alert": (255, 134, 110),
        }
        return mapping.get(accent, (214, 214, 214))

    def _draw_news_ticker(
        self,
        draw: HdmiPanelDrawSurface,
        *,
        box: tuple[int, int, int, int],
        ticker: HdmiNewsTickerModel,
    ) -> None:
        """Draw one calm bottom-bar headline ticker."""

        left, top, right, bottom = box
        if right <= left or bottom <= top:
            return

        box_height = max(28, bottom - top)
        compact = box_height < 42 or (right - left) < 420
        label_font = self.tools._font(14 if compact else 16, bold=True)
        text_font = self.tools._font(15 if compact else 20, bold=False)

        label_height = self._measure_text_height(draw, font=label_font)
        text_height = self._measure_text_height(draw, font=text_font)
        label_y = top + max(4, (box_height - label_height) // 2) - 1
        text_y = top + max(4, (box_height - text_height) // 2) - 1

        draw.rounded_rectangle(box, radius=18 if compact else 20, fill=(0, 0, 0), outline=(255, 255, 255), width=2)

        inner_left = left + 18
        inner_right = right - 18
        inner_width = max(0, inner_right - inner_left)
        gap = 14 if compact else 18

        min_text_width = min(max(32, inner_width // 2), inner_width)
        label_cap = max(0, min(max(28, inner_width // 3), inner_width - min_text_width - gap))
        label_text = self._fit_text(draw, ticker.label, max_width=label_cap, font=label_font)
        label_width = self._measure_text_width(draw, label_text, font=label_font) if label_text else 0

        text_left = inner_left
        if label_text:
            text_left += label_width + gap
            if text_left >= inner_right:
                label_text = ""
                label_width = 0
                text_left = inner_left

        if label_text:
            self._draw_ui_text(draw, (inner_left, label_y), label_text, fill=(168, 168, 168), font=label_font)

        available_width = max(0, inner_right - text_left)
        rendered = self._fit_text(draw, ticker.text, max_width=available_width, font=text_font)
        if rendered:
            self._draw_ui_text(draw, (text_left, text_y), rendered, fill=(255, 255, 255), font=text_font)

    def _render_cache(self) -> dict[str, dict[object, object]]:
        cache = getattr(self, "_hdmi_header_render_cache", None)
        if not isinstance(cache, dict):
            cache = {"width": {}, "height": {}, "fit": {}}
            setattr(self, "_hdmi_header_render_cache", cache)
        return cache

    def _bounded_cache_store(self, cache: dict[object, object], key: object, value: object) -> object:
        if len(cache) >= _RENDER_CACHE_LIMIT:
            cache.clear()
        cache[key] = value
        return value

    def _draw_ui_text(
        self,
        draw: HdmiPanelDrawSurface,
        xy: tuple[int, int],
        text: str,
        *,
        fill: Any,
        font: Any,
    ) -> None:
        if not text:
            return

        direction = self._preferred_text_direction(text)
        direction_supported = self._directional_text_support()
        if direction is not None and direction_supported is not False:
            try:
                draw.text(xy, text, fill=fill, font=font, direction=direction)
                self._set_directional_text_support(True)
                return
            except Exception:
                self._set_directional_text_support(False)
        draw.text(xy, text, fill=fill, font=font)

    def _preferred_text_direction(self, text: str) -> str | None:
        for character in text:
            bidi_class = unicodedata.bidirectional(character)
            if bidi_class in _RTL_BIDI_CLASSES:
                return "rtl"
            if bidi_class == "L":
                return "ltr"
        return None

    def _measure_text_width(self, draw: HdmiPanelDrawSurface, text: str, *, font: Any) -> int:
        if not text:
            return 0

        direction = self._preferred_text_direction(text)
        cache = self._render_cache()["width"]
        key = (id(font), direction, text)
        cached = cache.get(key)
        if cached is not None:
            return int(cached)

        width = 0
        textlength = getattr(draw, "textlength", None)
        direction_supported = self._directional_text_support()
        if callable(textlength):
            try:
                kwargs = {"font": font}
                if direction is not None and direction_supported is not False:
                    kwargs["direction"] = direction
                width = int(round(textlength(text, **kwargs)))
                if "direction" in kwargs:
                    self._set_directional_text_support(True)
            except Exception:
                if direction is not None:
                    self._set_directional_text_support(False)
                width = 0
        if width <= 0:
            width = int(self.tools._text_width(draw, text, font=font))
        return int(self._bounded_cache_store(cache, key, width))

    def _directional_text_support(self) -> bool | None:
        return getattr(self, "_hdmi_directional_text_supported", None)

    def _set_directional_text_support(self, supported: bool) -> None:
        setattr(self, "_hdmi_directional_text_supported", supported)

    def _measure_text_height(self, draw: HdmiPanelDrawSurface, *, font: Any) -> int:
        cache = self._render_cache()["height"]
        key = id(font)
        cached = cache.get(key)
        if cached is not None:
            return int(cached)
        height = int(self.tools._text_height(draw, font=font))
        return int(self._bounded_cache_store(cache, key, height))

    def _fit_text(
        self,
        draw: HdmiPanelDrawSurface,
        text: str,
        *,
        max_width: int,
        font: Any,
        fallback: str = "",
        uppercase: bool = False,
        max_graphemes: int = _MAX_UI_TEXT_GRAPHEMES,
    ) -> str:
        if max_width <= 0:
            return ""

        clean = self._normalise_ui_text(
            text,
            fallback=fallback,
            uppercase=uppercase,
            max_graphemes=max_graphemes,
        )
        if not clean:
            return ""

        cache = self._render_cache()["fit"]
        key = (id(font), clean, max_width)
        cached = cache.get(key)
        if isinstance(cached, str):
            return cached

        if self._measure_text_width(draw, clean, font=font) <= max_width:
            return str(self._bounded_cache_store(cache, key, clean))

        ellipsis_width = self._measure_text_width(draw, _ELLIPSIS, font=font)
        if ellipsis_width > max_width:
            return ""

        graphemes = self._split_graphemes(clean)
        if not graphemes:
            return ""

        low = 0
        high = len(graphemes)
        best = _ELLIPSIS
        while low < high:
            mid = (low + high + 1) // 2
            candidate = "".join(graphemes[:mid]).rstrip()
            if not candidate:
                high = mid - 1
                continue
            rendered = f"{candidate}{_ELLIPSIS}"
            if self._measure_text_width(draw, rendered, font=font) <= max_width:
                best = rendered
                low = mid
            else:
                high = mid - 1
        return str(self._bounded_cache_store(cache, key, best))

    def _normalise_ui_text(
        self,
        value: object,
        *,
        fallback: str = "",
        uppercase: bool = False,
        max_graphemes: int = _MAX_UI_TEXT_GRAPHEMES,
    ) -> str:
        text = self.tools._normalise_text(value, fallback=fallback)
        if not text:
            return ""

        text = "".join(" " if character in "\r\n\t\f\v" else character for character in text)
        text = "".join(character for character in text if character not in _BIDI_OVERRIDE_CHARS)
        text = "".join(
            character
            for character in text
            if unicodedata.category(character) != "Cc" or character == " "
        )
        text = " ".join(text.split())
        if not text:
            return ""

        text = self._limit_graphemes(text, max_graphemes)
        if uppercase:
            text = text.upper()
        return text

    def _limit_graphemes(self, text: str, max_graphemes: int) -> str:
        if max_graphemes <= 0 or not text:
            return ""
        if _regex is None:
            return text[:max_graphemes]
        graphemes = _regex.findall(r"\X", text)
        if len(graphemes) <= max_graphemes:
            return text
        return "".join(graphemes[:max_graphemes])

    def _split_graphemes(self, text: str) -> list[str]:
        if not text:
            return []
        if _regex is None:
            return list(text)
        return list(_regex.findall(r"\X", text))
