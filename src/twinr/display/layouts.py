# CHANGELOG: 2026-03-28
# BUG-1: Fixed debug_log section sizing that could push later sections off-screen when 4+ sections were supplied.
# BUG-2: Fixed silent API data loss by rendering state_fields instead of ignoring them.
# SEC-1: Hardened rendering against log/status text spoofing and Pi-side UI stalls by stripping bidi/control chars and bounding rendered content.
# IMP-1: Replaced vendor-private text measurement dependency with Pillow-compatible bbox/length helpers and adaptive wrapping.
# IMP-2: Added a layout registry plus responsive section packing and overflow summaries for constrained e-paper surfaces.

"""Compose Twinr status-card layout variants on the e-paper canvas.

This module keeps panel-layout concerns separate from the Waveshare transport
adapter so new display surfaces can evolve without mixing into vendor import
or hardware recovery logic.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Iterable, Sequence
from itertools import islice
from typing import Literal, Protocol

StateFields = tuple[tuple[str, str], ...]
LogSections = tuple[tuple[str, tuple[str, ...]], ...]
LayoutMode = Literal["default", "debug_log"]

_ELLIPSIS = "…"
_HEADER_HEIGHT = 58
_DEFAULT_MARGIN_X = 18
_SECTION_GAP = 4
_SECTION_MIN_HEIGHT = 50
_STATE_CHIP_HEIGHT = 18
_STATE_CHIP_GAP_X = 6
_STATE_CHIP_GAP_Y = 4
_STATE_CHIP_PADDING_X = 6

_MAX_STATE_FIELDS = 6
_MAX_DETAILS = 5
_MAX_LOG_SECTIONS = 8
_MAX_LOG_LINES_PER_SECTION = 18

_MAX_HEADLINE_CHARS = 160
_MAX_STATUS_CHARS = 48
_MAX_DETAIL_CHARS = 180
_MAX_STATE_KEY_CHARS = 28
_MAX_STATE_VALUE_CHARS = 40
_MAX_LOG_TITLE_CHARS = 60
_MAX_LOG_LINE_CHARS = 220

_BIDI_CONTROL_CHARS = frozenset(
    {
        "\u200e",  # LRM
        "\u200f",  # RLM
        "\u202a",  # LRE
        "\u202b",  # RLE
        "\u202c",  # PDF
        "\u202d",  # LRO
        "\u202e",  # RLO
        "\u2066",  # LRI
        "\u2067",  # RLI
        "\u2068",  # FSI
        "\u2069",  # PDI
    }
)


class StatusCardDisplay(Protocol):
    """Structural display contract for layout rendering."""

    def _font(self, size: int, bold: bool = False) -> object: ...
    def _draw_face(
        self,
        draw: object,
        *,
        status: str,
        animation_frame: int,
        center_x: int,
        center_y: int,
        scale: float,
    ) -> None: ...
    def _draw_details_footer(
        self,
        draw: object,
        *,
        details: tuple[str, ...],
        canvas_width: int,
        canvas_height: int,
    ) -> None: ...


class CanvasDraw(Protocol):
    """Minimal draw contract used by this module."""

    def rectangle(
        self,
        xy: tuple[int, int, int, int],
        fill: int | None = None,
        outline: int | None = None,
        width: int = 1,
    ) -> None: ...
    def text(
        self,
        xy: tuple[float, float],
        text: str,
        fill: int,
        font: object,
        anchor: str | None = None,
    ) -> None: ...
    def line(
        self,
        xy: tuple[int, int, int, int],
        fill: int | None = None,
        width: int = 1,
    ) -> None: ...


LayoutRenderer = Callable[
    [StatusCardDisplay, CanvasDraw, str, str, tuple[str, ...], StateFields, LogSections, int, int, int],
    None,
]

_LAYOUT_RENDERERS: dict[str, LayoutRenderer] = {}


def register_status_card_layout(name: str, renderer: LayoutRenderer) -> None:
    """Register a new layout renderer by name."""
    normalized_name = " ".join(str(name).split())
    if not normalized_name:
        raise ValueError("Layout name must not be empty.")
    _LAYOUT_RENDERERS[normalized_name] = renderer


def draw_status_card(
    display: StatusCardDisplay,
    draw: CanvasDraw,
    *,
    layout_mode: str,
    status: str,
    headline: str,
    details: tuple[str, ...],
    state_fields: StateFields,
    log_sections: LogSections,
    animation_frame: int,
    canvas_width: int,
    canvas_height: int,
) -> None:
    """Draw one of the supported status-card layouts onto the canvas."""
    normalized_layout = " ".join(str(layout_mode).split())
    renderer = _LAYOUT_RENDERERS.get(normalized_layout)
    if renderer is None:
        raise RuntimeError(f"Unsupported display layout: {layout_mode}")

    safe_status = _sanitize_single_line(status, max_chars=_MAX_STATUS_CHARS)
    safe_headline = _sanitize_single_line(headline, max_chars=_MAX_HEADLINE_CHARS)
    safe_details = _prepare_details(details)
    safe_state_fields = _prepare_state_fields(state_fields)
    safe_log_sections = _prepare_log_sections(log_sections)

    renderer(
        display,
        draw,
        safe_status,
        safe_headline,
        safe_details,
        safe_state_fields,
        safe_log_sections,
        animation_frame,
        canvas_width,
        canvas_height,
    )


def _draw_default_status_card(
    display: StatusCardDisplay,
    draw: CanvasDraw,
    status: str,
    headline: str,
    details: tuple[str, ...],
    state_fields: StateFields,
    log_sections: LogSections,
    animation_frame: int,
    canvas_width: int,
    canvas_height: int,
) -> None:
    del log_sections

    brand_font = display._font(28, bold=True)
    status_font = display._font(24, bold=True)
    chip_font = display._font(12, bold=False)

    draw.rectangle((0, 0, canvas_width - 1, canvas_height - 1), fill=255)
    draw.rectangle((0, 0, canvas_width - 1, _HEADER_HEIGHT), fill=0)
    draw.text((_DEFAULT_MARGIN_X, 10), "TWINR", fill=255, font=brand_font)

    label_text = _truncate_text(
        draw,
        headline or "Status",
        max_width=max(canvas_width - 170, 110),
        font=status_font,
    )
    label_width = _text_width(draw, label_text, font=status_font)
    draw.text(
        (canvas_width - label_width - _DEFAULT_MARGIN_X, 12),
        label_text,
        fill=255,
        font=status_font,
    )

    if state_fields:
        _draw_state_field_chips(
            draw,
            state_fields=state_fields,
            font=chip_font,
            left=_DEFAULT_MARGIN_X,
            top=_HEADER_HEIGHT + 8,
            max_width=canvas_width - (_DEFAULT_MARGIN_X * 2),
            max_rows=1,
            fill=255,
            outline=0,
            text_fill=0,
        )

    display._draw_face(
        draw,
        status=status,
        animation_frame=animation_frame,
        center_x=canvas_width // 2,
        center_y=(canvas_height // 2) + (20 if state_fields else 16),
        scale=1.0,
    )
    display._draw_details_footer(
        draw,
        details=details,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )


def _draw_debug_log_status_card(
    display: StatusCardDisplay,
    draw: CanvasDraw,
    status: str,
    headline: str,
    details: tuple[str, ...],
    state_fields: StateFields,
    log_sections: LogSections,
    animation_frame: int,
    canvas_width: int,
    canvas_height: int,
) -> None:
    del status, details, animation_frame

    title_font = display._font(18, bold=True)
    section_font = display._font(14, bold=True)
    line_font = display._font(11, bold=False)
    chip_font = display._font(11, bold=False)

    draw.rectangle((0, 0, canvas_width - 1, canvas_height - 1), fill=255)
    header_text = _truncate_text(
        draw,
        f"TWINR Debug Log | {headline or 'System'}",
        max_width=max(canvas_width - 24, 40),
        font=title_font,
    )
    draw.text((12, 8), header_text, fill=0, font=title_font)
    draw.line((12, 32, canvas_width - 12, 32), fill=0, width=2)

    top = 38
    if state_fields:
        top = _draw_state_field_chips(
            draw,
            state_fields=state_fields,
            font=chip_font,
            left=12,
            top=top,
            max_width=canvas_width - 24,
            max_rows=2,
            fill=255,
            outline=0,
            text_fill=0,
        )
        top += 4

    sections_to_draw = _fit_sections_to_canvas(
        log_sections if log_sections else (("Activity", ("No log data available.",)),),
        available_height=max(canvas_height - top - 6, 1),
    )

    bottom_margin = 6
    available_height = max(canvas_height - top - bottom_margin, 1)
    section_count = len(sections_to_draw)
    inner_available = max(available_height - (_SECTION_GAP * max(section_count - 1, 0)), section_count)
    base_height = max(inner_available // max(section_count, 1), 1)
    remainder = max(inner_available - (base_height * section_count), 0)

    section_top = top
    for index, section in enumerate(sections_to_draw):
        section_height = base_height + (1 if index < remainder else 0)
        section_bottom = min(section_top + section_height, canvas_height - bottom_margin)
        _draw_log_section(
            draw,
            title_font=section_font,
            line_font=line_font,
            left=12,
            top=section_top,
            right=canvas_width - 12,
            bottom=section_bottom,
            title=section[0],
            lines=section[1],
        )
        section_top = section_bottom + _SECTION_GAP


def _prepare_details(details: Iterable[object]) -> tuple[str, ...]:
    result: list[str] = []
    for raw_detail in islice(details, _MAX_DETAILS):
        detail = _sanitize_single_line(raw_detail, max_chars=_MAX_DETAIL_CHARS)
        if detail:
            result.append(detail)
    return tuple(result)


def _prepare_state_fields(state_fields: Iterable[tuple[object, object]]) -> StateFields:
    result: list[tuple[str, str]] = []
    for key, value in islice(state_fields, _MAX_STATE_FIELDS):
        safe_key = _sanitize_single_line(key, max_chars=_MAX_STATE_KEY_CHARS)
        safe_value = _sanitize_single_line(value, max_chars=_MAX_STATE_VALUE_CHARS)
        if safe_key or safe_value:
            result.append((safe_key, safe_value))
    return tuple(result)


def _prepare_log_sections(log_sections: Iterable[tuple[object, Iterable[object]]]) -> LogSections:
    result: list[tuple[str, tuple[str, ...]]] = []
    for raw_title, raw_lines in islice(log_sections, _MAX_LOG_SECTIONS):
        title = _sanitize_single_line(raw_title, max_chars=_MAX_LOG_TITLE_CHARS) or "Section"
        lines: list[str] = []
        for raw_line in islice(raw_lines, _MAX_LOG_LINES_PER_SECTION):
            line = _sanitize_single_line(raw_line, max_chars=_MAX_LOG_LINE_CHARS)
            if line:
                lines.append(line)
        result.append((title, tuple(lines) or ("No recent entries.",)))
    return tuple(result)


def _sanitize_single_line(value: object, *, max_chars: int) -> str:
    text = "" if value is None else str(value)

    # Limit worst-case CPU work before Unicode normalization/truncation.
    raw_limit = max(max_chars * 4, 64)
    if len(text) > raw_limit:
        text = text[:raw_limit]

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    sanitized_chars: list[str] = []
    for char in text:
        if char in _BIDI_CONTROL_CHARS:
            continue
        if char == "\n" or char == "\t":
            sanitized_chars.append(" ")
            continue
        codepoint = ord(char)
        if codepoint < 32 or codepoint == 127:
            sanitized_chars.append(" ")
            continue
        sanitized_chars.append(char)

    collapsed = " ".join("".join(sanitized_chars).split()).strip()
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 1].rstrip() + _ELLIPSIS


def _fit_sections_to_canvas(
    sections: Sequence[tuple[str, tuple[str, ...]]],
    *,
    available_height: int,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    section_list = list(sections)
    if not section_list:
        return (("Activity", ("No log data available.",)),)

    max_fit = max(1, min(len(section_list), (available_height + _SECTION_GAP) // (_SECTION_MIN_HEIGHT + _SECTION_GAP)))
    if len(section_list) <= max_fit:
        return tuple(section_list)

    visible_slots = max_fit
    if visible_slots <= 1:
        hidden_titles = [title for title, _ in section_list]
        summary_lines = hidden_titles[:3]
        if len(hidden_titles) > 3:
            summary_lines.append(f"+{len(hidden_titles) - 3} more sections")
        return (
            (
                f"+{len(hidden_titles)} sections",
                tuple(summary_lines) or ("Hidden due to limited space.",),
            ),
        )

    visible_sections = section_list[: visible_slots - 1]
    hidden_sections = section_list[visible_slots - 1 :]
    summary_lines = [title for title, _ in hidden_sections[:3]]
    remaining_titles = len(hidden_sections) - len(summary_lines)
    if remaining_titles > 0:
        summary_lines.append(f"+{remaining_titles} more sections")
    summary_section = (
        f"+{len(hidden_sections)} more sections",
        tuple(summary_lines) or ("Hidden due to limited space.",),
    )
    return tuple(visible_sections + [summary_section])


def _draw_log_section(
    draw: CanvasDraw,
    *,
    title_font: object,
    line_font: object,
    left: int,
    top: int,
    right: int,
    bottom: int,
    title: str,
    lines: tuple[str, ...],
) -> None:
    if bottom <= top:
        return

    _draw_box(draw, (left, top, right, bottom), fill=255, outline=0, radius=6, width=1)

    inner_left = left + 4
    inner_right = right - 4
    title_y = top + 5

    draw.text(
        (inner_left + 2, title_y),
        _truncate_text(draw, title, max_width=max(inner_right - inner_left - 4, 20), font=title_font),
        fill=0,
        font=title_font,
    )

    separator_y = title_y + _line_height(draw, title_font) + 2
    draw.line((inner_left + 2, separator_y, inner_right - 2, separator_y), fill=0, width=1)

    text_top = separator_y + 5
    text_width = max(inner_right - inner_left - 4, 20)
    line_step = _line_height(draw, line_font) + 2
    max_lines = max((bottom - text_top - 4) // max(line_step, 1), 1)

    wrapped_lines: list[str] = []
    for raw_line in lines:
        wrapped = _wrap_text(draw, raw_line, max_width=text_width, font=line_font)
        wrapped_lines.extend(wrapped or ("",))

    visible_lines = _limit_lines(
        draw,
        wrapped_lines or ["No recent entries."],
        max_lines=max_lines,
        max_width=text_width,
        font=line_font,
    )

    line_y = text_top
    for line in visible_lines:
        draw.text((inner_left + 4, line_y), line, fill=0, font=line_font)
        line_y += line_step


def _draw_state_field_chips(
    draw: CanvasDraw,
    *,
    state_fields: StateFields,
    font: object,
    left: int,
    top: int,
    max_width: int,
    max_rows: int,
    fill: int,
    outline: int,
    text_fill: int,
) -> int:
    chip_top = top
    chip_left = left
    rows_used = 1
    line_height = _line_height(draw, font)

    items: list[str] = []
    for key, value in state_fields:
        if key and value:
            items.append(f"{key}: {value}")
        elif key:
            items.append(key)
        elif value:
            items.append(value)

    hidden_items = 0
    for index, item in enumerate(items):
        text_max_width = max(max_width - (_STATE_CHIP_PADDING_X * 2), 16)
        chip_text = _truncate_text(draw, item, max_width=text_max_width, font=font)
        chip_width = min(
            _text_width(draw, chip_text, font=font) + (_STATE_CHIP_PADDING_X * 2),
            max_width,
        )

        if chip_left > left and chip_left + chip_width > left + max_width:
            if rows_used >= max_rows:
                hidden_items = len(items) - index
                break
            rows_used += 1
            chip_left = left
            chip_top += _STATE_CHIP_HEIGHT + _STATE_CHIP_GAP_Y

        _draw_box(
            draw,
            (
                chip_left,
                chip_top,
                chip_left + chip_width,
                chip_top + _STATE_CHIP_HEIGHT,
            ),
            fill=fill,
            outline=outline,
            radius=5,
            width=1,
        )
        text_y = chip_top + max((_STATE_CHIP_HEIGHT - line_height) // 2 - 1, 0)
        draw.text((chip_left + _STATE_CHIP_PADDING_X, text_y), chip_text, fill=text_fill, font=font)
        chip_left += chip_width + _STATE_CHIP_GAP_X

    if hidden_items > 0:
        overflow_text = f"+{hidden_items}"
        chip_text = _truncate_text(
            draw,
            overflow_text,
            max_width=max(max_width - (_STATE_CHIP_PADDING_X * 2), 16),
            font=font,
        )
        chip_width = min(
            _text_width(draw, chip_text, font=font) + (_STATE_CHIP_PADDING_X * 2),
            max_width,
        )
        if chip_left > left and chip_left + chip_width > left + max_width:
            if rows_used < max_rows:
                rows_used += 1
                chip_left = left
                chip_top += _STATE_CHIP_HEIGHT + _STATE_CHIP_GAP_Y
            else:
                return chip_top + _STATE_CHIP_HEIGHT

        _draw_box(
            draw,
            (
                chip_left,
                chip_top,
                chip_left + chip_width,
                chip_top + _STATE_CHIP_HEIGHT,
            ),
            fill=fill,
            outline=outline,
            radius=5,
            width=1,
        )
        text_y = chip_top + max((_STATE_CHIP_HEIGHT - line_height) // 2 - 1, 0)
        draw.text((chip_left + _STATE_CHIP_PADDING_X, text_y), chip_text, fill=text_fill, font=font)

    return chip_top + _STATE_CHIP_HEIGHT


def _wrap_text(
    draw: CanvasDraw,
    text: str,
    *,
    max_width: int,
    font: object,
) -> tuple[str, ...]:
    normalized = " ".join(text.split())
    if not normalized:
        return tuple()

    words = normalized.split(" ")
    lines: list[str] = []
    current = ""

    for word in words:
        if not current:
            if _fits_width(draw, word, max_width=max_width, font=font):
                current = word
                continue

            broken = _break_token(draw, word, max_width=max_width, font=font)
            if broken:
                lines.extend(broken[:-1])
                current = broken[-1]
            continue

        candidate = f"{current} {word}"
        if _fits_width(draw, candidate, max_width=max_width, font=font):
            current = candidate
            continue

        lines.append(current)
        if _fits_width(draw, word, max_width=max_width, font=font):
            current = word
            continue

        broken = _break_token(draw, word, max_width=max_width, font=font)
        if broken:
            lines.extend(broken[:-1])
            current = broken[-1]
        else:
            current = word

    if current:
        lines.append(current)

    return tuple(lines)


def _break_token(
    draw: CanvasDraw,
    token: str,
    *,
    max_width: int,
    font: object,
) -> list[str]:
    if not token:
        return []
    if _fits_width(draw, token, max_width=max_width, font=font):
        return [token]

    pieces: list[str] = []
    remainder = token
    while remainder:
        cut = _largest_fitting_prefix(draw, remainder, max_width=max_width, font=font)
        if cut <= 0:
            cut = 1

        piece = remainder[:cut]
        pieces.append(piece)
        remainder = remainder[cut:]

        if _fits_width(draw, remainder, max_width=max_width, font=font):
            if remainder:
                pieces.append(remainder)
            break

    return pieces


def _largest_fitting_prefix(
    draw: CanvasDraw,
    text: str,
    *,
    max_width: int,
    font: object,
) -> int:
    low = 0
    high = len(text)
    best = 0
    while low <= high:
        mid = (low + high) // 2
        sample = text[:mid]
        if _fits_width(draw, sample, max_width=max_width, font=font):
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


def _limit_lines(
    draw: CanvasDraw,
    lines: Sequence[str],
    *,
    max_lines: int,
    max_width: int,
    font: object,
) -> tuple[str, ...]:
    if max_lines <= 0:
        return tuple()
    if len(lines) <= max_lines:
        return tuple(lines)

    visible = list(lines[:max_lines])
    final_line = visible[-1].rstrip()
    overflow_candidate = f"{final_line} {_ELLIPSIS}".strip()
    visible[-1] = _truncate_text(draw, overflow_candidate, max_width=max_width, font=font)
    if not visible[-1]:
        visible[-1] = _truncate_text(draw, _ELLIPSIS, max_width=max_width, font=font)
    if not visible[-1].endswith(_ELLIPSIS):
        visible[-1] = _truncate_text(draw, f"{visible[-1]} {_ELLIPSIS}", max_width=max_width, font=font)
    return tuple(visible)


def _truncate_text(
    draw: CanvasDraw,
    text: str,
    *,
    max_width: int,
    font: object,
) -> str:
    normalized = " ".join(text.split()).strip()
    if not normalized or max_width <= 0:
        return ""
    if _fits_width(draw, normalized, max_width=max_width, font=font):
        return normalized

    ellipsis_width = _text_width(draw, _ELLIPSIS, font=font)
    if ellipsis_width > max_width:
        return ""

    low = 0
    high = len(normalized)
    best = _ELLIPSIS
    while low <= high:
        mid = (low + high) // 2
        candidate = normalized[:mid].rstrip()
        if candidate:
            candidate = f"{candidate}{_ELLIPSIS}"
        else:
            candidate = _ELLIPSIS

        if _fits_width(draw, candidate, max_width=max_width, font=font):
            best = candidate
            low = mid + 1
        else:
            high = mid - 1

    return best


def _fits_width(
    draw: CanvasDraw,
    text: str,
    *,
    max_width: int,
    font: object,
) -> bool:
    return _text_width(draw, text, font=font) <= max_width


def _text_width(draw: CanvasDraw, text: str, *, font: object) -> int:
    if not text:
        return 0

    textbbox = getattr(draw, "textbbox", None)
    if callable(textbbox):
        left, _, right, _ = textbbox((0, 0), text, font=font)
        return max(int(round(right - left)), 0)

    textlength = getattr(draw, "textlength", None)
    if callable(textlength):
        return max(int(round(textlength(text, font=font))), 0)

    getbbox = getattr(font, "getbbox", None)
    if callable(getbbox):
        left, _, right, _ = getbbox(text)
        return max(int(round(right - left)), 0)

    return len(text) * 8


def _line_height(draw: CanvasDraw, font: object) -> int:
    probe = "Ag"

    textbbox = getattr(draw, "textbbox", None)
    if callable(textbbox):
        _, top, _, bottom = textbbox((0, 0), probe, font=font)
        return max(int(round(bottom - top)), 1)

    getbbox = getattr(font, "getbbox", None)
    if callable(getbbox):
        _, top, _, bottom = getbbox(probe)
        return max(int(round(bottom - top)), 1)

    return 12


def _draw_box(
    draw: CanvasDraw,
    box: tuple[int, int, int, int],
    *,
    fill: int | None,
    outline: int | None,
    radius: int,
    width: int,
) -> None:
    rounded_rectangle = getattr(draw, "rounded_rectangle", None)
    if callable(rounded_rectangle):
        rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
        return
    draw.rectangle(box, fill=fill, outline=outline, width=width)


register_status_card_layout("default", _draw_default_status_card)
register_status_card_layout("debug_log", _draw_debug_log_status_card)
