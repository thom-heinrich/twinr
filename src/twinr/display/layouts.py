"""Compose Twinr status-card layout variants on the e-paper canvas.

This module keeps panel-layout concerns separate from the Waveshare transport
adapter so new display surfaces can evolve without mixing into vendor import
or hardware recovery logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from twinr.display.waveshare_v2 import WaveshareEPD4In2V2


StateFields = tuple[tuple[str, str], ...]
LogSections = tuple[tuple[str, tuple[str, ...]], ...]


def draw_status_card(
    display: "WaveshareEPD4In2V2",
    draw: object,
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

    if layout_mode == "default":
        _draw_default_status_card(
            display,
            draw,
            status=status,
            headline=headline,
            details=details,
            animation_frame=animation_frame,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )
        return
    if layout_mode == "debug_log":
        _draw_debug_log_status_card(
            display,
            draw,
            headline=headline,
            log_sections=log_sections,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )
        return
    raise RuntimeError(f"Unsupported display layout: {layout_mode}")


def _draw_default_status_card(
    display: "WaveshareEPD4In2V2",
    draw: object,
    *,
    status: str,
    headline: str,
    details: tuple[str, ...],
    animation_frame: int,
    canvas_width: int,
    canvas_height: int,
) -> None:
    brand_font = display._font(28, bold=True)
    status_font = display._font(24, bold=True)
    draw.rectangle((0, 0, canvas_width - 1, canvas_height - 1), fill=255)
    draw.rectangle((0, 0, canvas_width - 1, 58), fill=0)
    draw.text((18, 10), "TWINR", fill=255, font=brand_font)
    status_label = " ".join(headline.split())
    label_text = display._truncate_text(
        draw,
        status_label,
        max_width=max(canvas_width - 170, 110),
        font=status_font,
    )
    label_width = display._text_width(draw, label_text, font=status_font)
    draw.text((canvas_width - label_width - 18, 12), label_text, fill=255, font=status_font)
    display._draw_face(
        draw,
        status=status,
        animation_frame=animation_frame,
        center_x=canvas_width // 2,
        center_y=(canvas_height // 2) + 16,
        scale=1.0,
    )
    display._draw_details_footer(
        draw,
        details=details,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )


def _draw_debug_log_status_card(
    display: "WaveshareEPD4In2V2",
    draw: object,
    *,
    headline: str,
    log_sections: LogSections,
    canvas_width: int,
    canvas_height: int,
) -> None:
    title_font = display._font(18, bold=True)
    section_font = display._font(14, bold=True)
    line_font = display._font(11, bold=False)

    draw.rectangle((0, 0, canvas_width - 1, canvas_height - 1), fill=255)
    header_text = display._truncate_text(
        draw,
        f"TWINR Debug Log | {headline}",
        max_width=canvas_width - 24,
        font=title_font,
    )
    draw.text((12, 8), header_text, fill=0, font=title_font)
    draw.line((12, 32, canvas_width - 12, 32), fill=0, width=2)

    top = 38
    bottom_margin = 6
    section_gap = 4
    section_count = max(len(log_sections), 1)
    available_height = canvas_height - top - bottom_margin - (section_gap * max(section_count - 1, 0))
    section_height = max(74, available_height // section_count)

    for index, section in enumerate(log_sections):
        title, lines = section
        section_top = top + (index * (section_height + section_gap))
        section_bottom = min(section_top + section_height, canvas_height - bottom_margin)
        draw.text(
            (14, section_top),
            display._truncate_text(draw, title, max_width=canvas_width - 28, font=section_font),
            fill=0,
            font=section_font,
        )
        draw.line((14, section_top + 18, canvas_width - 14, section_top + 18), fill=0, width=1)
        line_height = display._text_height(draw, font=line_font) + 2
        line_y = section_top + 22
        max_lines = max((section_bottom - line_y) // max(line_height, 1), 1)
        for line in lines[:max_lines]:
            text = display._truncate_text(draw, line, max_width=canvas_width - 28, font=line_font)
            draw.text((16, line_y), text, fill=0, font=line_font)
            line_y += line_height
