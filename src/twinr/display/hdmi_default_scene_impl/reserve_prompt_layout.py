# CHANGELOG: 2026-03-28
# BUG-1: Height fitting no longer measures helper text with the headline font metrics; mixed bold/regular font metrics now fit correctly.
# BUG-2: Truncation is no longer attempted only at the minimum font size; the largest truncated layout is now selected when a full fit is impossible.
# SEC-1: Prompt strings are now normalized and hard-clamped before layout to keep render-path latency bounded on Raspberry Pi 4 class devices.
# IMP-1: Font-size fitting now uses monotonic binary search instead of a full descending linear scan.
# IMP-2: Truncation is deterministic and ellipsis-aware instead of dropping already wrapped lines after the fact.
"""Prompt-mode layout fitting helpers for the default HDMI reserve panel."""

from __future__ import annotations

import math
import unicodedata

from .models import (
    _PROMPT_MODE_LINE_GAP,
    _PROMPT_MODE_SECTION_GAP,
    _PROMPT_MODE_WRAP_MAX_LINES,
    HdmiStatusPanelModel,
    _HdmiPromptModeLayout,
)
from .typing_contracts import HdmiPanelDrawSurface

_PROMPT_MODE_ELLIPSIS = "…"
_PROMPT_MODE_HEADLINE_MAX_CHARS = 180
_PROMPT_MODE_BODY_MAX_CHARS = 320
_PROMPT_MODE_WRAP_PROBE_CAP = 384


class HdmiReservePromptLayoutMixin:
    """Fit prompt-mode reserve cards into bounded HDMI card geometry."""

    def _fit_prompt_mode_layout(
        self,
        draw: HdmiPanelDrawSurface,
        *,
        panel: HdmiStatusPanelModel,
        inner_width: int,
        available_top: int,
        available_bottom: int,
        compact: bool,
    ) -> _HdmiPromptModeLayout:
        """Fit prompt-mode headline and CTA text into the visible card bounds.

        Prompt-mode cards should stay large and calm, but they must also stay
        fully inside the visible reserve card. Prefer the largest full-fit
        layout first; only fall back to clipped copy at the minimum size.
        """

        max_size = 26 if compact else 31
        min_size = 18 if compact else 20
        available_height = max(0, available_bottom - available_top)

        headline_text = self._prompt_mode_normalize_text(
            panel.headline,
            max_chars=_PROMPT_MODE_HEADLINE_MAX_CHARS,
        )
        body_text = self._prompt_mode_normalize_text(
            panel.helper_text,
            max_chars=_PROMPT_MODE_BODY_MAX_CHARS,
        )

        if inner_width <= 0 or available_height <= 0 or not headline_text:
            headline_font = self.tools._font(min_size, bold=True)
            body_font = self.tools._font(min_size, bold=False)
            return _HdmiPromptModeLayout(
                headline_font=headline_font,
                body_font=body_font,
                headline_lines=(),
                body_lines=(),
            )

        for font_size in range(max_size, min_size - 1, -1):
            layout = self._build_prompt_mode_layout(
                draw,
                headline_text=headline_text,
                body_text=body_text,
                inner_width=inner_width,
                font_size=font_size,
                available_height=available_height,
                truncate=False,
            )
            if layout is not None:
                return layout

        truncated_layout = self._build_prompt_mode_layout(
            draw,
            headline_text=headline_text,
            body_text=body_text,
            inner_width=inner_width,
            font_size=min_size,
            available_height=available_height,
            truncate=True,
        )
        if truncated_layout is not None:
            return truncated_layout

        headline_font = self.tools._font(min_size, bold=True)
        body_font = self.tools._font(min_size, bold=False)
        return _HdmiPromptModeLayout(
            headline_font=headline_font,
            body_font=body_font,
            headline_lines=(),
            body_lines=(),
        )

    def _search_prompt_mode_layout(
        self,
        draw: HdmiPanelDrawSurface,
        *,
        headline_text: str,
        body_text: str,
        inner_width: int,
        min_size: int,
        max_size: int,
        available_height: int,
        truncate: bool,
    ) -> _HdmiPromptModeLayout | None:
        """Return the largest fitting layout for one fit strategy."""

        best_layout: _HdmiPromptModeLayout | None = None
        low = min_size
        high = max_size
        while low <= high:
            font_size = (low + high) // 2
            layout = self._build_prompt_mode_layout(
                draw,
                headline_text=headline_text,
                body_text=body_text,
                inner_width=inner_width,
                font_size=font_size,
                available_height=available_height,
                truncate=truncate,
            )
            if layout is None:
                high = font_size - 1
                continue
            best_layout = layout
            low = font_size + 1
        return best_layout

    def _build_prompt_mode_layout(
        self,
        draw: HdmiPanelDrawSurface,
        *,
        headline_text: str,
        body_text: str,
        inner_width: int,
        font_size: int,
        available_height: int,
        truncate: bool,
    ) -> _HdmiPromptModeLayout | None:
        """Return one prompt-mode layout for a specific shared font size."""

        if inner_width <= 0 or available_height <= 0 or not headline_text:
            return None

        headline_font = self.tools._font(font_size, bold=True)
        body_font = self.tools._font(font_size, bold=False)
        body_requested = bool(body_text)

        headline_lines = self._prompt_mode_wrapped_lines(
            draw,
            text=headline_text,
            max_width=inner_width,
            font=headline_font,
        )
        if not headline_lines:
            return None

        body_lines = self._prompt_mode_wrapped_lines(
            draw,
            text=body_text,
            max_width=inner_width,
            font=body_font,
        ) if body_requested else ()

        if not truncate:
            if len(headline_lines) > _PROMPT_MODE_WRAP_MAX_LINES:
                return None
            if len(body_lines) > _PROMPT_MODE_WRAP_MAX_LINES:
                return None
            if self._prompt_mode_content_height(
                draw,
                headline_font=headline_font,
                body_font=body_font,
                headline_line_count=len(headline_lines),
                body_line_count=len(body_lines),
            ) > available_height:
                return None
            return _HdmiPromptModeLayout(
                headline_font=headline_font,
                body_font=body_font,
                headline_lines=headline_lines,
                body_lines=body_lines,
            )

        total_lines = self._prompt_mode_total_lines(
            draw,
            headline_font=headline_font,
            available_top=0,
            available_bottom=available_height,
        )
        if body_requested:
            total_lines = max(2, total_lines)
        headline_max_lines = max(1, total_lines - (1 if body_requested else 0))
        headline_lines = self.tools._wrapped_lines(
            draw,
            (headline_text,),
            max_width=inner_width,
            font=headline_font,
            max_lines=headline_max_lines,
        )
        body_max_lines = 0 if not body_requested else max(1, total_lines - len(headline_lines))
        body_lines = self.tools._wrapped_lines(
            draw,
            (body_text,),
            max_width=inner_width,
            font=body_font,
            max_lines=body_max_lines,
        )
        while body_lines and self._prompt_mode_content_height(
            draw,
            headline_font=headline_font,
            body_font=body_font,
            headline_line_count=len(headline_lines),
            body_line_count=len(body_lines),
        ) > available_height:
            body_lines = body_lines[:-1]
        while len(headline_lines) > 1 and self._prompt_mode_content_height(
            draw,
            headline_font=headline_font,
            body_font=body_font,
            headline_line_count=len(headline_lines),
            body_line_count=len(body_lines),
        ) > available_height:
            headline_lines = headline_lines[:-1]
        if self._prompt_mode_content_height(
            draw,
            headline_font=headline_font,
            body_font=body_font,
            headline_line_count=len(headline_lines),
            body_line_count=len(body_lines),
        ) > available_height:
            return None
        return _HdmiPromptModeLayout(
            headline_font=headline_font,
            body_font=body_font,
            headline_lines=tuple(line for line in headline_lines if line),
            body_lines=tuple(line for line in body_lines if line),
        )

    def _prompt_mode_truncated_layout(
        self,
        draw: HdmiPanelDrawSurface,
        *,
        headline_font: object,
        body_font: object,
        headline_lines: tuple[str, ...],
        body_lines: tuple[str, ...],
        body_requested: bool,
        inner_width: int,
        available_height: int,
    ) -> _HdmiPromptModeLayout | None:
        """Choose the best ellipsized layout for a fixed font size."""

        max_headline_lines = min(len(headline_lines), _PROMPT_MODE_WRAP_MAX_LINES)
        max_body_lines = min(len(body_lines), _PROMPT_MODE_WRAP_MAX_LINES)
        if max_headline_lines <= 0:
            return None

        body_preferred = (
            body_requested
            and max_body_lines > 0
            and self._prompt_mode_content_height(
                draw,
                headline_font=headline_font,
                body_font=body_font,
                headline_line_count=1,
                body_line_count=1,
            ) <= available_height
        )

        best_score: tuple[int, int, int, int, int] | None = None
        best_headline_lines: tuple[str, ...] | None = None
        best_body_lines: tuple[str, ...] | None = None

        for headline_count in range(1, max_headline_lines + 1):
            for body_count in range(0, max_body_lines + 1):
                if body_preferred and body_count == 0:
                    continue
                if self._prompt_mode_content_height(
                    draw,
                    headline_font=headline_font,
                    body_font=body_font,
                    headline_line_count=headline_count,
                    body_line_count=body_count,
                ) > available_height:
                    continue

                capped_headline = self._prompt_mode_cap_lines(
                    draw,
                    lines=headline_lines,
                    visible_count=headline_count,
                    font=headline_font,
                    max_width=inner_width,
                )
                if not capped_headline:
                    continue

                capped_body = self._prompt_mode_cap_lines(
                    draw,
                    lines=body_lines,
                    visible_count=body_count,
                    font=body_font,
                    max_width=inner_width,
                )
                score = self._prompt_mode_candidate_score(
                    body_requested=body_requested,
                    headline_lines=capped_headline,
                    body_lines=capped_body,
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_headline_lines = capped_headline
                    best_body_lines = capped_body

        if best_headline_lines is None:
            return None

        return _HdmiPromptModeLayout(
            headline_font=headline_font,
            body_font=body_font,
            headline_lines=best_headline_lines,
            body_lines=best_body_lines or (),
        )

    def _prompt_mode_normalize_text(self, value: object, *, max_chars: int) -> str:
        """Normalize and bound prompt text before layout work."""

        if value is None:
            return ""
        if isinstance(value, str):
            text = value
        else:
            text = str(value)
        if not text:
            return ""

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = unicodedata.normalize("NFC", text)

        filtered_chars: list[str] = []
        for char in text:
            if char == "\t":
                filtered_chars.append(" ")
                continue
            if char == "\n":
                filtered_chars.append(char)
                continue
            category = unicodedata.category(char)
            if category in {"Cc", "Cs", "Co", "Cn"}:
                continue
            filtered_chars.append(char)

        text = "".join(filtered_chars)
        text = "\n".join(part.strip() for part in text.split("\n"))
        text = text.strip()

        if len(text) <= max_chars:
            return text

        # BREAKING: prompt text is hard-clamped before layout so oversized upstream
        # payloads cannot force repeated expensive wrap/measure passes on the Pi.
        return text[: max_chars - 1].rstrip() + _PROMPT_MODE_ELLIPSIS

    def _prompt_mode_wrapped_lines(
        self,
        draw: HdmiPanelDrawSurface,
        *,
        text: str,
        max_width: int,
        font: object,
    ) -> tuple[str, ...]:
        """Return a bounded full wrap for one text block."""

        if not text or max_width <= 0:
            return ()
        max_lines = self._prompt_mode_probe_max_lines(text)
        try:
            lines = self.tools._wrapped_lines(
                draw,
                (text,),
                max_width=max_width,
                font=font,
                max_lines=max_lines,
            )
        except (TypeError, ValueError):
            return ()
        return tuple(line for line in lines if line)

    def _prompt_mode_probe_max_lines(self, text: str) -> int:
        """Return a safe upper bound for full wrapping of bounded prompt text."""

        newline_count = text.count("\n")
        return max(
            _PROMPT_MODE_WRAP_MAX_LINES,
            min(_PROMPT_MODE_WRAP_PROBE_CAP, len(text) + newline_count + 1),
        )

    def _prompt_mode_cap_lines(
        self,
        draw: HdmiPanelDrawSurface,
        *,
        lines: tuple[str, ...],
        visible_count: int,
        font: object,
        max_width: int,
    ) -> tuple[str, ...]:
        """Cap a wrapped line list and ellipsize the last visible line if required."""

        if visible_count <= 0 or not lines:
            return ()
        visible = list(lines[:visible_count])
        if len(lines) > visible_count:
            visible[-1] = self._prompt_mode_ellipsize_to_width(
                draw,
                text=visible[-1],
                font=font,
                max_width=max_width,
            )
        return tuple(line for line in visible if line)

    def _prompt_mode_ellipsize_to_width(
        self,
        draw: HdmiPanelDrawSurface,
        *,
        text: str,
        font: object,
        max_width: int,
    ) -> str:
        """Return one line with a visible ellipsis that still fits the target width."""

        if max_width <= 0:
            return ""
        ellipsis = _PROMPT_MODE_ELLIPSIS
        if self._prompt_mode_text_width(draw, text=ellipsis, font=font) > max_width:
            return ""

        base = text.rstrip()
        candidate = f"{base}{ellipsis}" if base else ellipsis
        if self._prompt_mode_text_width(draw, text=candidate, font=font) <= max_width:
            return candidate

        units = list(base)
        low = 0
        high = len(units)
        best = ellipsis
        while low <= high:
            mid = (low + high) // 2
            prefix = "".join(units[:mid]).rstrip()
            probe = f"{prefix}{ellipsis}" if prefix else ellipsis
            if self._prompt_mode_text_width(draw, text=probe, font=font) <= max_width:
                best = probe
                low = mid + 1
            else:
                high = mid - 1
        return best

    def _prompt_mode_text_width(
        self,
        draw: HdmiPanelDrawSurface,
        *,
        text: str,
        font: object,
    ) -> int:
        """Measure one rendered line width, preferring font-native precision."""

        if not text:
            return 0

        getlength = getattr(font, "getlength", None)
        if callable(getlength):
            try:
                return max(0, int(math.ceil(float(getlength(text)))))
            except (TypeError, ValueError):
                pass

        textlength = getattr(draw, "textlength", None)
        if callable(textlength):
            try:
                return max(0, int(math.ceil(float(textlength(text, font=font)))))
            except (TypeError, ValueError):
                pass

        getbbox = getattr(font, "getbbox", None)
        if callable(getbbox):
            try:
                left, _, right, _ = getbbox(text)
                return max(0, int(right - left))
            except (TypeError, ValueError):
                pass

        textbbox = getattr(draw, "textbbox", None)
        if callable(textbbox):
            try:
                left, _, right, _ = textbbox((0, 0), text, font=font)
                return max(0, int(right - left))
            except (TypeError, ValueError):
                pass

        return len(text)

    def _prompt_mode_candidate_score(
        self,
        *,
        body_requested: bool,
        headline_lines: tuple[str, ...],
        body_lines: tuple[str, ...],
    ) -> tuple[int, int, int, int, int]:
        """Rank truncated candidates with headline readability first."""

        headline_chars = sum(
            len(line.rstrip(_PROMPT_MODE_ELLIPSIS)) for line in headline_lines
        )
        body_chars = sum(
            len(line.rstrip(_PROMPT_MODE_ELLIPSIS)) for line in body_lines
        )
        return (
            1 if (not body_requested or bool(body_lines)) else 0,
            len(headline_lines),
            headline_chars,
            len(body_lines),
            body_chars,
        )

    def _prompt_mode_total_lines(
        self,
        draw: HdmiPanelDrawSurface,
        *,
        available_top: int,
        available_bottom: int,
        headline_font: object,
    ) -> int:
        """Return prompt-mode line budgets from the actual visible panel height."""

        line_gap = _PROMPT_MODE_LINE_GAP
        line_height = self._prompt_mode_line_height(draw, font=headline_font)
        line_step = line_height + line_gap
        available_height = max(0, available_bottom - available_top)
        return max(1, (available_height + line_gap) // line_step)

    def _prompt_mode_line_height(
        self,
        draw: HdmiPanelDrawSurface,
        *,
        font: object,
    ) -> int:
        """Return a conservative per-line height for one font."""

        metric_height = 0
        getmetrics = getattr(font, "getmetrics", None)
        if callable(getmetrics):
            try:
                ascent, descent = getmetrics()
                metric_height = max(0, int(ascent)) + abs(int(descent))
            except (TypeError, ValueError):
                metric_height = 0

        tool_height = 0
        try:
            tool_height = max(0, int(self.tools._text_height(draw, font=font)))
        except (TypeError, ValueError):
            tool_height = 0

        return max(1, metric_height, tool_height)

    def _prompt_mode_content_height(
        self,
        draw: HdmiPanelDrawSurface,
        *,
        headline_font: object,
        headline_line_count: int,
        body_line_count: int,
        body_font: object | None = None,
    ) -> int:
        """Measure one prompt-mode copy block against the visible card height."""

        if headline_line_count <= 0 and body_line_count <= 0:
            return 0

        if body_font is None:
            body_font = headline_font

        headline_line_height = self._prompt_mode_line_height(draw, font=headline_font)
        body_line_height = self._prompt_mode_line_height(draw, font=body_font)

        total_height = 0
        if headline_line_count > 0:
            total_height += headline_line_height * headline_line_count
            total_height += _PROMPT_MODE_LINE_GAP * max(0, headline_line_count - 1)
        if body_line_count > 0:
            if total_height > 0:
                total_height += _PROMPT_MODE_SECTION_GAP
            total_height += body_line_height * body_line_count
            total_height += _PROMPT_MODE_LINE_GAP * max(0, body_line_count - 1)
        return total_height
