"""Compose the modular default HDMI scene for Twinr's fullscreen surface.

This module keeps the senior-facing HDMI layout separate from framebuffer or
Wayland transport concerns. It exposes explicit scene, layout, and card models
so future capability cards, panel morphs, and richer animations can grow
without bloating the backend adapters.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from twinr.display.contracts import DisplayStateFields
from twinr.display.face_cues import DisplayFaceCue
from twinr.display.hdmi_ambient_moments import HdmiAmbientMoment, HdmiAmbientMomentDirector
from twinr.display.hdmi_presentation_graph import (
    HdmiPresentationNode,
    HdmiPresentationSceneGraph,
    HdmiPresentationSceneGraphBuilder,
)
from twinr.display.presentation_cues import DisplayPresentationCue


_STATE_CARD_ORDER = ("Status", "Internet", "AI", "System", "Zeit", "Hinweis")
_DETAIL_MAX_LINES = 3


class _HdmiSceneTools(Protocol):
    """Text and sanitization helpers supplied by the active HDMI backend."""

    def _font(self, size: int, *, bold: bool) -> object: ...

    def _text_width(self, draw: object, text: str, *, font: object | None = None) -> int: ...

    def _text_height(self, draw: object, *, font: object | None = None) -> int: ...

    def _truncate_text(self, draw: object, text: str, *, max_width: int, font: object | None = None) -> str: ...

    def _wrapped_lines(
        self,
        draw: object,
        lines: tuple[str, ...],
        *,
        max_width: int,
        font: object,
        max_lines: int,
    ) -> tuple[str, ...]: ...

    def _normalise_text(self, value: object, *, fallback: str) -> str: ...


@dataclass(frozen=True, slots=True)
class HdmiSceneCard:
    """Describe one status card in the right-hand HDMI status panel."""

    key: str
    label: str
    value: str
    accent: tuple[int, int, int]
    detail_lines: tuple[str, ...] = ()
    emphasis: float = 1.0
    column_span: int = 1
    row_span: int = 1


@dataclass(frozen=True, slots=True)
class HdmiStatusPanelModel:
    """Prepared content for the right-hand status panel."""

    eyebrow: str
    headline: str
    helper_text: str
    cards: tuple[HdmiSceneCard, ...]


@dataclass(frozen=True, slots=True)
class HdmiNewsTickerModel:
    """Prepared content for the bottom HDMI news ticker."""

    label: str
    text: str


@dataclass(frozen=True, slots=True)
class HdmiDefaultSceneLayout:
    """Geometry contract for the default Twinr HDMI scene."""

    header_box: tuple[int, int, int, int]
    face_box: tuple[int, int, int, int]
    panel_box: tuple[int, int, int, int]
    ticker_box: tuple[int, int, int, int]
    ticker_reserved: bool
    compact_panel: bool


@dataclass(frozen=True, slots=True)
class HdmiDefaultScene:
    """Full scene state for one rendered HDMI frame."""

    status: str
    animation_frame: int
    layout: HdmiDefaultSceneLayout
    time_value: str
    panel: HdmiStatusPanelModel
    ticker: HdmiNewsTickerModel | None = None
    face_cue: DisplayFaceCue | None = None
    ambient_moment: HdmiAmbientMoment | None = None
    presentation_graph: HdmiPresentationSceneGraph | None = None


def order_state_fields(
    normalise_text: Callable[..., str],
    state_fields: DisplayStateFields,
    details: tuple[str, ...],
) -> DisplayStateFields:
    """Keep state fields stable for rendering or synthesize short detail rows."""

    if state_fields:
        order = {name: index for index, name in enumerate(_STATE_CARD_ORDER)}
        return tuple(sorted(state_fields, key=lambda item: order.get(item[0], len(order))))

    synthesized = []
    for index, detail in enumerate(details[:_DETAIL_MAX_LINES]):
        synthesized.append((f"Info {index + 1}", normalise_text(detail, fallback="")))
    return tuple(synthesized)


def state_field_value(
    normalise_text: Callable[..., str],
    state_fields: DisplayStateFields,
    name: str | tuple[str, ...],
    *,
    fallback: str = "--",
) -> str:
    """Return one display state-field value with stable fallback behavior."""

    names = (name,) if isinstance(name, str) else name
    for field_name, value in state_fields:
        if field_name in names:
            return normalise_text(value, fallback=fallback)
    return fallback


def display_state_value(normalise_text: Callable[..., str], field_name: str, value: str) -> str:
    """Translate mixed-language runtime values into stable HDMI copy."""

    compact = normalise_text(value, fallback="--")
    normalized = compact.lower()
    if field_name in {"Internet", "Network"}:
        mapping = {
            "ok": "Online",
            "fehlt": "Offline",
            "?": "Unknown",
            "wartet": "Waiting",
        }
        return mapping.get(normalized, compact)
    if field_name == "AI":
        mapping = {
            "ok": "Ready",
            "fehlt": "Missing",
            "?": "Unknown",
            "wartet": "Waiting",
            "achtung": "Warning",
            "fehler": "Error",
        }
        return mapping.get(normalized, compact)
    if field_name == "System":
        mapping = {
            "ok": "OK",
            "fehler": "Error",
            "achtung": "Warning",
            "warm": "Warm",
            "?": "Unknown",
        }
        return mapping.get(normalized, compact)
    if field_name in {"Zeit", "Time"}:
        return compact

    generic = {
        "bereit": "Ready",
        "wartet": "Waiting",
        "fehlt": "Missing",
        "achtung": "Warning",
        "fehler": "Error",
        "status wird aktualisiert": "Updating status",
        "status nicht verfügbar": "Status unavailable",
        "?": "Unknown",
    }
    return generic.get(normalized, compact)


def status_headline(normalise_text: Callable[..., str], status: str, *, fallback: str | None) -> str:
    """Return the senior-facing main headline for one runtime status."""

    mapping = {
        "waiting": "Waiting",
        "listening": "Listening",
        "processing": "Thinking",
        "answering": "Speaking",
        "printing": "Printing",
        "error": "Check system",
    }
    if status in mapping:
        return mapping[status]
    return normalise_text(fallback, fallback=status.replace("_", " ").title())


def status_helper_text(status: str) -> str:
    """Return a short, calm helper sentence for the active status."""

    mapping = {
        "waiting": "Press the green button and speak naturally.",
        "listening": "Listening now. Speak at your own pace.",
        "processing": "Thinking for a moment.",
        "answering": "Speaking now.",
        "printing": "Preparing the print.",
        "error": "Please check the system in debug view.",
    }
    return mapping.get(status, "Twinr is updating the current status.")


def status_accent_color(status: str) -> tuple[int, int, int]:
    """Return the accent color associated with one runtime status."""

    mapping = {
        "waiting": (90, 132, 196),
        "listening": (36, 163, 130),
        "processing": (226, 164, 51),
        "answering": (82, 114, 222),
        "printing": (197, 128, 35),
        "error": (205, 89, 74),
    }
    return mapping.get(status, (102, 126, 150))


def state_value_color(normalise_text: Callable[..., str], value: str) -> tuple[int, int, int]:
    """Return the accent color for a specific panel-card value."""

    normalized = normalise_text(value, fallback="").lower()
    if normalized in {"ok", "bereit", "ready", "online"}:
        return (40, 167, 117)
    if normalized in {"fehler", "fehlt", "error", "offline", "missing"}:
        return (205, 89, 74)
    if normalized in {"achtung", "warm", "warning"}:
        return (228, 152, 34)
    if normalized in {"?", "wartet", "waiting", "unknown"}:
        return (130, 145, 166)
    return (90, 132, 196)


def time_value(state_fields: DisplayStateFields) -> str:
    """Return the visible clock value for the header."""

    for name, value in state_fields:
        if name in {"Zeit", "Time"}:
            return value
    return "--:--"


@dataclass(slots=True)
class HdmiDefaultSceneRenderer:
    """Render Twinr's modular default HDMI scene."""

    tools: _HdmiSceneTools
    ambient_director: HdmiAmbientMomentDirector = field(default_factory=HdmiAmbientMomentDirector)

    def draw(
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
        animation_frame: int,
        ticker_text: str | None = None,
        face_cue: DisplayFaceCue | None = None,
        presentation_cue: DisplayPresentationCue | None = None,
        presentation_now: datetime | None = None,
        ambient_now: datetime | None = None,
    ) -> None:
        """Draw one full default HDMI scene onto the provided canvas."""

        scene = self.build_scene(
            width=width,
            height=height,
            status=status,
            headline=headline,
            ticker_text=ticker_text,
            helper_text=helper_text,
            state_fields=state_fields,
            animation_frame=animation_frame,
            face_cue=face_cue,
            presentation_cue=presentation_cue,
            presentation_now=presentation_now,
            ambient_now=ambient_now,
        )
        draw.rectangle((0, 0, width, height), fill=(0, 0, 0))
        self._draw_twinr_header(draw, box=scene.layout.header_box, time_value=scene.time_value)
        self._draw_face(
            draw,
            box=scene.layout.face_box,
            status=scene.status,
            animation_frame=scene.animation_frame,
            face_cue=scene.face_cue,
        )
        if scene.ambient_moment is not None:
            self._draw_ambient_moment(
                draw,
                box=scene.layout.face_box,
                animation_frame=scene.animation_frame,
                ambient_moment=scene.ambient_moment,
            )
        graph = scene.presentation_graph
        if graph is not None and graph.active_node is not None:
            self._draw_presentation_surface(
                image,
                draw,
                presentation=graph.active_node,
                queued_count=len(graph.queued_cards),
            )
            return
        self._draw_status_panel(draw, box=scene.layout.panel_box, panel=scene.panel, compact=scene.layout.compact_panel)
        if scene.ticker is not None:
            self._draw_news_ticker(draw, box=scene.layout.ticker_box, ticker=scene.ticker)

    def build_scene(
        self,
        *,
        width: int,
        height: int,
        status: str,
        headline: str,
        helper_text: str,
        state_fields: DisplayStateFields,
        animation_frame: int,
        ticker_text: str | None = None,
        face_cue: DisplayFaceCue | None = None,
        presentation_cue: DisplayPresentationCue | None = None,
        presentation_now: datetime | None = None,
        ambient_now: datetime | None = None,
    ) -> HdmiDefaultScene:
        """Build the scene model so layout and content can evolve independently."""

        ticker = self._build_ticker_model(ticker_text)
        layout = self._layout_for_size(width=width, height=height, reserve_ticker=ticker is not None)
        presentation_graph = self._presentation_graph(
            cue=presentation_cue,
            layout=layout,
            now=presentation_now,
        )
        ambient_moment = self._ambient_moment(
            status=status,
            face_cue=face_cue,
            presentation_graph=presentation_graph,
            ambient_now=ambient_now,
        )
        resolved_face_cue = face_cue
        if resolved_face_cue is None and presentation_graph is not None:
            resolved_face_cue = presentation_graph.face_cue
        if resolved_face_cue is None and ambient_moment is not None:
            resolved_face_cue = ambient_moment.face_cue
        return HdmiDefaultScene(
            status=status,
            animation_frame=animation_frame,
            layout=layout,
            time_value=time_value(state_fields),
            panel=self._build_panel_model(
                status=status,
                headline=headline,
                helper_text=helper_text,
                state_fields=state_fields,
            ),
            ticker=None if presentation_graph is not None else ticker,
            face_cue=resolved_face_cue,
            ambient_moment=ambient_moment,
            presentation_graph=presentation_graph,
        )

    def _layout_for_size(self, *, width: int, height: int, reserve_ticker: bool) -> HdmiDefaultSceneLayout:
        if width < 560 or height < 320:
            header_box = (12, 10, width - 12, 56)
            if reserve_ticker:
                ticker_box = (12, height - 54, width - 12, height - 12)
                content_bottom = ticker_box[1] - 10
            else:
                ticker_box = (12, height - 12, width - 12, height - 12)
                content_bottom = height - 12
            face_box_width = max(94, min(126, int(width * 0.33)))
            face_box = (18, 70, 18 + face_box_width, content_bottom)
            panel_box = (face_box[2] + 14, 68, width - 18, content_bottom)
        else:
            header_box = (24, 18, width - 24, 72)
            if reserve_ticker:
                ticker_box = (24, height - 74, width - 24, height - 24)
                content_bottom = ticker_box[1] - 16
            else:
                ticker_box = (24, height - 24, width - 24, height - 24)
                content_bottom = height - 24
            face_box_width = max(250, min(324, int(width * 0.39)))
            face_box = (38, 90, 38 + face_box_width, content_bottom)
            panel_box = (face_box[2] + 20, 90, width - 36, content_bottom)
        compact_panel = self._should_use_compact_panel(panel_box=panel_box, reserve_ticker=reserve_ticker)
        return HdmiDefaultSceneLayout(
            header_box=header_box,
            face_box=face_box,
            panel_box=panel_box,
            ticker_box=ticker_box,
            ticker_reserved=reserve_ticker,
            compact_panel=compact_panel,
        )

    def _should_use_compact_panel(
        self,
        *,
        panel_box: tuple[int, int, int, int],
        reserve_ticker: bool,
    ) -> bool:
        """Return whether the status panel must collapse to its compact card mode."""

        panel_width = panel_box[2] - panel_box[0]
        panel_height = panel_box[3] - panel_box[1]
        if panel_width < 320 or panel_height < 240:
            return True
        if reserve_ticker and panel_height < 320:
            return True
        return False

    def _build_panel_model(
        self,
        *,
        status: str,
        headline: str,
        helper_text: str,
        state_fields: DisplayStateFields,
    ) -> HdmiStatusPanelModel:
        normalise_text = self.tools._normalise_text
        cards = (
            HdmiSceneCard(
                key="network",
                label="NETWORK",
                value=display_state_value(normalise_text, "Internet", state_field_value(normalise_text, state_fields, "Internet")),
                accent=state_value_color(normalise_text, state_field_value(normalise_text, state_fields, "Internet")),
            ),
            HdmiSceneCard(
                key="ai",
                label="AI",
                value=display_state_value(normalise_text, "AI", state_field_value(normalise_text, state_fields, "AI")),
                accent=state_value_color(normalise_text, state_field_value(normalise_text, state_fields, "AI")),
            ),
            HdmiSceneCard(
                key="system",
                label="SYSTEM",
                value=display_state_value(normalise_text, "System", state_field_value(normalise_text, state_fields, "System")),
                accent=state_value_color(normalise_text, state_field_value(normalise_text, state_fields, "System")),
            ),
            HdmiSceneCard(
                key="time",
                label="TIME",
                value=display_state_value(
                    normalise_text,
                    "Zeit",
                    state_field_value(normalise_text, state_fields, ("Zeit", "Time")),
                ),
                accent=status_accent_color(status),
            ),
        )
        return HdmiStatusPanelModel(
            eyebrow="STATUS",
            headline=headline,
            helper_text=helper_text,
            cards=cards,
        )

    def _build_ticker_model(self, ticker_text: str | None) -> HdmiNewsTickerModel | None:
        """Return the optional bottom-bar ticker model."""

        compact = self.tools._normalise_text(ticker_text, fallback="")
        if not compact:
            return None
        return HdmiNewsTickerModel(label="NEWS", text=compact)

    def _presentation_graph(
        self,
        *,
        cue: DisplayPresentationCue | None,
        layout: HdmiDefaultSceneLayout,
        now: datetime | None,
    ) -> HdmiPresentationSceneGraph | None:
        """Resolve the optional presentation cue into a dedicated scene graph."""

        builder = HdmiPresentationSceneGraphBuilder()
        return builder.build(
            cue=cue,
            face_box=layout.face_box,
            panel_box=layout.panel_box,
            now=now,
        )

    def _ambient_moment(
        self,
        *,
        status: str,
        face_cue: DisplayFaceCue | None,
        presentation_graph: HdmiPresentationSceneGraph | None,
        ambient_now: datetime | None,
    ) -> HdmiAmbientMoment | None:
        """Resolve the optional idle-only ambient moment for the waiting face."""

        return self.ambient_director.resolve(
            status=status,
            now=ambient_now,
            face_cue_active=face_cue is not None,
            presentation_active=presentation_graph is not None,
        )

    def _draw_twinr_header(self, draw: object, *, box: tuple[int, int, int, int], time_value: str) -> None:
        left, top, right, bottom = box
        box_height = max(32, bottom - top)
        header_font = self.tools._font(24 if box_height >= 50 else 18, bold=True)
        time_font = self.tools._font(22 if box_height >= 50 else 16, bold=False)
        label_y = top + max(4, (box_height - self.tools._text_height(draw, font=header_font)) // 2) - 1
        time_y = top + max(4, (box_height - self.tools._text_height(draw, font=time_font)) // 2) - 1

        draw.rounded_rectangle(box, radius=20, fill=(0, 0, 0), outline=(255, 255, 255), width=2)
        draw.text((left + 20, label_y), "TWINR", fill=(255, 255, 255), font=header_font)
        time_width = self.tools._text_width(draw, time_value, font=time_font)
        draw.text((right - 22 - time_width, time_y), time_value, fill=(188, 188, 188), font=time_font)

    def _draw_news_ticker(
        self,
        draw: object,
        *,
        box: tuple[int, int, int, int],
        ticker: HdmiNewsTickerModel,
    ) -> None:
        """Draw one calm bottom-bar headline ticker."""

        left, top, right, bottom = box
        box_height = max(28, bottom - top)
        compact = box_height < 42 or (right - left) < 420
        label_font = self.tools._font(14 if compact else 16, bold=True)
        text_font = self.tools._font(15 if compact else 20, bold=False)
        label_height = self.tools._text_height(draw, font=label_font)
        text_height = self.tools._text_height(draw, font=text_font)
        label_y = top + max(4, (box_height - label_height) // 2) - 1
        text_y = top + max(4, (box_height - text_height) // 2) - 1
        draw.rounded_rectangle(box, radius=18 if compact else 20, fill=(0, 0, 0), outline=(255, 255, 255), width=2)
        draw.text((left + 18, label_y), ticker.label, fill=(168, 168, 168), font=label_font)
        label_width = self.tools._text_width(draw, ticker.label, font=label_font)
        text_left = left + 18 + label_width + (14 if compact else 18)
        available_width = max(48, right - text_left - 18)
        rendered = self.tools._truncate_text(draw, ticker.text, max_width=available_width, font=text_font)
        draw.text((text_left, text_y), rendered, fill=(255, 255, 255), font=text_font)

    def _draw_status_panel(
        self,
        draw: object,
        *,
        box: tuple[int, int, int, int],
        panel: HdmiStatusPanelModel,
        compact: bool,
    ) -> None:
        left, top, right, bottom = box
        panel_width = right - left
        panel_padding = 16 if compact else 24
        panel_inner_width = panel_width - (panel_padding * 2)

        eyebrow_font = self.tools._font(14 if compact else 16, bold=True)
        headline_font = self.tools._font(28 if compact else 44, bold=True)
        helper_font = self.tools._font(13 if compact else 19, bold=False)

        draw.rounded_rectangle(box, radius=30, fill=(4, 4, 4), outline=(255, 255, 255), width=3)
        header_label_y = top + (16 if compact else 22)
        headline_y = top + (40 if compact else 56)
        helper_y = top + (76 if compact else 118)
        draw.text((left + panel_padding, header_label_y), panel.eyebrow, fill=(160, 160, 160), font=eyebrow_font)
        headline_text = self.tools._truncate_text(draw, panel.headline, max_width=panel_inner_width, font=headline_font)
        draw.text((left + panel_padding, headline_y), headline_text, fill=(255, 255, 255), font=headline_font)

        helper_lines = self.tools._wrapped_lines(
            draw,
            (panel.helper_text,),
            max_width=panel_inner_width,
            font=helper_font,
            max_lines=1 if compact else 2,
        )
        line_height = self.tools._text_height(draw, font=helper_font) + 6
        for line in helper_lines:
            draw.text((left + panel_padding, helper_y), line, fill=(214, 214, 214), font=helper_font)
            helper_y += line_height

        divider_y = helper_y + (8 if compact else 10)
        draw.line((left + panel_padding, divider_y, right - panel_padding, divider_y), fill=(88, 88, 88), width=2)
        rows_top = divider_y + (12 if compact else 18)
        rows_bottom = bottom - (16 if compact else 22)
        if rows_bottom <= rows_top:
            return
        rows_box = (
            left + panel_padding,
            rows_top,
            right - panel_padding,
            rows_bottom,
        )
        if compact or self._rows_need_compact(rows_box):
            self._draw_compact_cards(draw, box=rows_box, cards=panel.cards)
            return
        self._draw_cards(draw, box=rows_box, cards=panel.cards)

    def _rows_need_compact(self, box: tuple[int, int, int, int]) -> bool:
        """Return whether the remaining status-card area is too small for a two-by-two grid."""

        left, top, right, bottom = box
        available_width = max(0, right - left)
        available_height = max(0, bottom - top)
        if available_width <= 0 or available_height <= 0:
            return True
        # The full 2x2 card layout needs enough vertical space for two rows,
        # their gap, and label/value padding. With less space, the compact
        # summary rows remain readable and avoid overlapping chrome.
        return available_height < 124

    def _draw_presentation_surface(
        self,
        image: object,
        draw: object,
        *,
        presentation: HdmiPresentationNode,
        queued_count: int,
    ) -> None:
        """Draw one expanding presentation surface over the default scene."""

        left, top, right, bottom = presentation.box
        compact = (right - left) < 360 or (bottom - top) < 240
        padding = 16 if compact else 24
        eyebrow_font = self.tools._font(14 if compact else 17, bold=True)
        title_font = self.tools._font(22 if compact else 40, bold=True)
        subtitle_font = self.tools._font(13 if compact else 18, bold=False)
        body_font = self.tools._font(14 if compact else 20, bold=False)
        label = "SHOWING" if presentation.kind == "image" else "FOCUS"
        label = label if queued_count <= 0 else f"{label} +{queued_count}"

        draw.rounded_rectangle(
            presentation.box,
            radius=32 if not compact else 20,
            fill=(2, 2, 2),
            outline=presentation.accent,
            width=3,
        )
        draw.rounded_rectangle(
            (left + 12, top + 12, left + 20, top + 20),
            radius=4,
            fill=presentation.accent,
        )
        text_left = left + padding
        text_top = top + padding
        inner_width = max(80, right - left - (padding * 2))
        draw.text((text_left + 18, text_top), label, fill=(176, 176, 176), font=eyebrow_font)
        text_top += self.tools._text_height(draw, font=eyebrow_font) + (10 if compact else 14)

        if presentation.title and presentation.chrome_progress >= 0.12:
            wrapped_title = self.tools._wrapped_lines(
                draw,
                (presentation.title,),
                max_width=inner_width,
                font=title_font,
                max_lines=2,
            )
            title_line_height = self.tools._text_height(draw, font=title_font) + 6
            for line in wrapped_title:
                draw.text((text_left, text_top), line, fill=(255, 255, 255), font=title_font)
                text_top += title_line_height

        if presentation.subtitle and presentation.content_progress > 0.0:
            subtitle_lines = self.tools._wrapped_lines(
                draw,
                (presentation.subtitle,),
                max_width=inner_width,
                font=subtitle_font,
                max_lines=2,
            )
            subtitle_line_height = self.tools._text_height(draw, font=subtitle_font) + 4
            for line in subtitle_lines:
                draw.text((text_left, text_top), line, fill=(214, 214, 214), font=subtitle_font)
                text_top += subtitle_line_height
            text_top += 4

        content_bottom = bottom - padding
        if presentation.kind == "image" and presentation.content_progress > 0.0:
            text_top = self._draw_presentation_image_block(
                image,
                draw,
                presentation=presentation,
                box=(text_left, text_top, right - padding, content_bottom),
                compact=compact,
            )

        remaining_top = min(text_top + (8 if compact else 14), content_bottom)
        remaining_box = (text_left, remaining_top, right - padding, content_bottom)
        self._draw_presentation_body(
            draw,
            body_lines=presentation.body_lines,
            box=remaining_box,
            font=body_font,
            compact=compact,
            reveal_progress=presentation.body_progress,
        )

    def _draw_presentation_image_block(
        self,
        image: object,
        draw: object,
        *,
        presentation: HdmiPresentationNode,
        box: tuple[int, int, int, int],
        compact: bool,
    ) -> int:
        """Paste one local image into the presentation surface and return the next text Y."""

        left, top, right, bottom = box
        available_width = max(60, right - left)
        available_height = max(60, bottom - top)
        image_height = int(available_height * (0.62 if not compact else 0.54))
        image_box = (left, top, right, min(bottom, top + image_height))
        draw.rounded_rectangle(image_box, radius=18 if not compact else 12, fill=(8, 8, 8), outline=(84, 84, 84), width=2)
        pasted = self._paste_local_image(image, box=image_box, image_path=presentation.image_path)
        if not pasted:
            placeholder_font = self.tools._font(16 if compact else 24, bold=True)
            placeholder = "IMAGE UNAVAILABLE"
            placeholder_width = self.tools._text_width(draw, placeholder, font=placeholder_font)
            placeholder_height = self.tools._text_height(draw, font=placeholder_font)
            draw.text(
                (
                    left + max(0, (available_width - placeholder_width) // 2),
                    top + max(0, ((image_box[3] - image_box[1]) - placeholder_height) // 2),
                ),
                placeholder,
                fill=(170, 170, 170),
                font=placeholder_font,
            )
        return image_box[3]

    def _draw_presentation_body(
        self,
        draw: object,
        *,
        body_lines: tuple[str, ...],
        box: tuple[int, int, int, int],
        font: object,
        compact: bool,
        reveal_progress: float,
    ) -> None:
        """Draw the bounded body copy for a fullscreen presentation."""

        left, top, right, bottom = box
        if top >= bottom or right <= left or reveal_progress <= 0.0:
            return
        max_lines = 3 if compact else 4
        wrapped = self.tools._wrapped_lines(
            draw,
            body_lines,
            max_width=max(60, right - left),
            font=font,
            max_lines=max_lines,
        )
        line_height = self.tools._text_height(draw, font=font) + (4 if compact else 8)
        visible_lines = max(1, min(max_lines, int(round(reveal_progress * max_lines))))
        for index, line in enumerate(wrapped[:visible_lines]):
            y = top + (index * line_height)
            if y + line_height > bottom:
                break
            draw.text((left, y), line, fill=(236, 236, 236), font=font)

    def _paste_local_image(
        self,
        image: object,
        *,
        box: tuple[int, int, int, int],
        image_path: str | None,
    ) -> bool:
        """Paste one bounded local image into the target box."""

        if not image_path:
            return False
        try:
            from PIL import Image, ImageOps
        except ImportError:
            return False
        try:
            with Image.open(image_path) as opened:
                prepared = ImageOps.contain(
                    opened.convert("RGB"),
                    (max(1, box[2] - box[0] - 12), max(1, box[3] - box[1] - 12)),
                )
        except Exception:
            return False
        target_left = box[0] + max(0, ((box[2] - box[0]) - prepared.width) // 2)
        target_top = box[1] + max(0, ((box[3] - box[1]) - prepared.height) // 2)
        image.paste(prepared, (target_left, target_top))
        return True

    def _draw_cards(
        self,
        draw: object,
        *,
        box: tuple[int, int, int, int],
        cards: tuple[HdmiSceneCard, ...],
    ) -> None:
        left, top, right, bottom = box
        label_font = self.tools._font(15, bold=True)
        value_font = self.tools._font(20, bold=False)
        column_gap = 14
        row_gap = 12
        card_width = max(72, (right - left - column_gap) // 2)
        card_height = max(56, min(72, (bottom - top - row_gap) // 2))

        for index, card in enumerate(cards[:4]):
            column = index % 2
            row = index // 2
            row_left = left + column * (card_width + column_gap)
            row_top = top + row * (card_height + row_gap)
            row_right = row_left + card_width
            row_bottom = row_top + card_height
            self._draw_card(
                draw,
                box=(row_left, row_top, row_right, row_bottom),
                card=card,
                compact=False,
                label_font=label_font,
                value_font=value_font,
            )

    def _draw_compact_cards(
        self,
        draw: object,
        *,
        box: tuple[int, int, int, int],
        cards: tuple[HdmiSceneCard, ...],
    ) -> None:
        left, top, right, bottom = box
        available_height = max(0, bottom - top)
        if available_height < 56:
            label_font = self.tools._font(11, bold=True)
            value_font = self.tools._font(13, bold=False)
            summary = " | ".join(card.value for card in cards[:4])
            draw.rounded_rectangle((left, top, right, bottom), radius=12, fill=(0, 0, 0), outline=(104, 104, 104), width=2)
            draw.text((left + 10, top + 6), "LIVE", fill=(156, 156, 156), font=label_font)
            rendered = self.tools._truncate_text(draw, summary, max_width=max(24, right - left - 20), font=value_font)
            draw.text((left + 10, top + 18), rendered, fill=(255, 255, 255), font=value_font)
            return

        row_specs = (
            HdmiSceneCard(
                key="network_ai",
                label="NETWORK / AI",
                value=" / ".join(card.value for card in cards[:2]),
                accent=cards[0].accent if cards else (104, 104, 104),
            ),
            HdmiSceneCard(
                key="system_time",
                label="SYSTEM / TIME",
                value=" / ".join(card.value for card in cards[2:4]),
                accent=cards[2].accent if len(cards) > 2 else (104, 104, 104),
            ),
        )
        gap = 8
        row_height = max(24, min(40, (available_height - gap) // 2))
        label_font = self.tools._font(12, bold=True)
        value_font = self.tools._font(14, bold=False)
        for index, card in enumerate(row_specs):
            row_top = top + index * (row_height + gap)
            row_bottom = min(bottom, row_top + row_height)
            self._draw_card(
                draw,
                box=(left, row_top, right, row_bottom),
                card=card,
                compact=True,
                label_font=label_font,
                value_font=value_font,
            )

    def _draw_card(
        self,
        draw: object,
        *,
        box: tuple[int, int, int, int],
        card: HdmiSceneCard,
        compact: bool,
        label_font: object,
        value_font: object,
    ) -> None:
        left, top, right, bottom = box
        radius = 14 if compact else 16
        draw.rounded_rectangle(box, radius=radius, fill=(0, 0, 0), outline=(104, 104, 104), width=2)
        accent_left = left + 10
        accent_top = top + (11 if compact else 12)
        accent_size = 6 if compact else 8
        draw.rounded_rectangle(
            (accent_left, accent_top, accent_left + accent_size, accent_top + accent_size),
            radius=3,
            fill=card.accent,
        )
        label_x = left + (24 if compact else 16)
        label_y = top + (8 if compact else 12)
        value_y = top + (20 if compact else 34)
        draw.text((label_x, label_y), card.label, fill=(156, 156, 156), font=label_font)
        rendered_value = self.tools._truncate_text(
            draw,
            card.value,
            max_width=max(36, right - left - 24),
            font=value_font,
        )
        draw.text((left + 12, value_y), rendered_value, fill=(255, 255, 255), font=value_font)

    def _draw_face(
        self,
        draw: object,
        *,
        box: tuple[int, int, int, int],
        status: str,
        animation_frame: int,
        face_cue: DisplayFaceCue | None,
    ) -> None:
        left, top, right, bottom = box
        scale = self._normalise_scale(min((right - left) / 220.0, (bottom - top) / 210.0))
        center_x = (left + right) // 2
        center_y = top + ((bottom - top) // 2) - self._scaled_offset(6, scale)
        self._draw_face_features(
            draw,
            center_x=center_x,
            center_y=center_y,
            status=status,
            animation_frame=animation_frame,
            scale=max(0.56, min(scale, 1.55)),
            face_cue=face_cue,
        )

    def _draw_ambient_moment(
        self,
        draw: object,
        *,
        box: tuple[int, int, int, int],
        animation_frame: int,
        ambient_moment: HdmiAmbientMoment,
    ) -> None:
        """Draw a tiny ornament for one active idle ambient moment."""

        left, top, right, bottom = box
        scale = self._normalise_scale(min((right - left) / 220.0, (bottom - top) / 210.0))
        scale = max(0.56, min(scale, 1.55))
        center_x = (left + right) // 2
        center_y = top + ((bottom - top) // 2) - self._scaled_offset(6, scale)
        if ambient_moment.ornament == "heart":
            self._draw_ambient_heart(
                draw,
                center_x=center_x + self._scaled_offset(76, scale),
                center_y=center_y - self._scaled_offset(84, scale),
                scale=scale,
                progress=ambient_moment.progress,
            )
            return
        if ambient_moment.ornament == "crescent":
            self._draw_ambient_crescent(
                draw,
                center_x=center_x - self._scaled_offset(76, scale),
                center_y=center_y - self._scaled_offset(78, scale),
                scale=scale,
                progress=ambient_moment.progress,
            )
            return
        if ambient_moment.ornament == "wave_marks":
            self._draw_ambient_wave_marks(
                draw,
                center_x=center_x + self._scaled_offset(98, scale),
                center_y=center_y - self._scaled_offset(4, scale),
                scale=scale,
                progress=ambient_moment.progress,
            )
            return
        if ambient_moment.ornament == "crown":
            self._draw_ambient_crown(
                draw,
                center_x=center_x,
                center_y=center_y - self._scaled_offset(96, scale),
                scale=scale,
                progress=ambient_moment.progress,
            )
            return
        if ambient_moment.ornament == "dot_cluster":
            self._draw_ambient_dot_cluster(
                draw,
                center_x=center_x - self._scaled_offset(78, scale),
                center_y=center_y - self._scaled_offset(58, scale),
                scale=scale,
                progress=ambient_moment.progress,
                animation_frame=animation_frame,
            )
            return
        self._draw_ambient_sparkles(
            draw,
            center_x=center_x + self._scaled_offset(72, scale),
            center_y=center_y - self._scaled_offset(68, scale),
            scale=scale,
            progress=ambient_moment.progress,
            animation_frame=animation_frame,
        )

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

        drift_y = self._scaled_offset((0.45 - progress) * 10.0, scale)
        pulse = (0, 1, 2, 1)[animation_frame % 4]
        size = self._scaled_size(5 + pulse, scale, minimum=3)
        line_width = self._scaled_size(2, scale, minimum=1)
        self._draw_star(draw, center_x=center_x, center_y=center_y + drift_y, size=size, line_width=line_width)
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

        lift = self._scaled_offset(progress * 10.0, scale)
        wobble = (-1, 0, 1, 0)[animation_frame % 4]
        radius = self._scaled_size(4, scale, minimum=2)
        offsets = (
            (-10, 3),
            (1, -5),
            (12, 5),
        )
        for index, (offset_x, offset_y) in enumerate(offsets):
            jitter = wobble if index % 2 == 0 else -wobble
            cx = center_x + self._scaled_offset(offset_x + jitter, scale)
            cy = center_y + self._scaled_offset(offset_y, scale) - lift
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(255, 255, 255))

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
            fill=(255, 255, 255),
        )
        draw.ellipse(
            (
                center_x,
                top_y - lobe_radius,
                center_x + (lobe_radius * 2),
                top_y + lobe_radius,
            ),
            fill=(255, 255, 255),
        )
        draw.polygon(
            (
                (center_x - (lobe_radius * 2), top_y + self._scaled_offset(2, scale)),
                (center_x + (lobe_radius * 2), top_y + self._scaled_offset(2, scale)),
                (center_x, top_y + (lobe_radius * 3)),
            ),
            fill=(255, 255, 255),
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

        drift_y = self._scaled_offset(progress * 8.0, scale)
        radius = self._scaled_size(10, scale, minimum=5)
        box = (
            center_x - radius,
            center_y - radius - drift_y,
            center_x + radius,
            center_y + radius - drift_y,
        )
        draw.ellipse(box, fill=(255, 255, 255))
        mask_offset = self._scaled_offset(5, scale)
        draw.ellipse(
            (
                box[0] + mask_offset,
                box[1] - self._scaled_offset(1, scale),
                box[2] + mask_offset,
                box[3] - self._scaled_offset(1, scale),
            ),
            fill=(0, 0, 0),
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

        drift_x = self._scaled_offset(progress * 6.0, scale)
        line_width = self._scaled_size(2, scale, minimum=1)
        for index, width in enumerate((18, 12)):
            offset_y = index * self._scaled_offset(12, scale)
            draw.arc(
                (
                    center_x - self._scaled_offset(width, scale) + drift_x,
                    center_y - self._scaled_offset(10, scale) + offset_y,
                    center_x + self._scaled_offset(width, scale) + drift_x,
                    center_y + self._scaled_offset(10, scale) + offset_y,
                ),
                start=280,
                end=80,
                fill=(255, 255, 255),
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
        draw.line(points, fill=(255, 255, 255), width=line_width)
        draw.line(
            (
                center_x - half_width,
                center_y + height - lift,
                center_x + half_width,
                center_y + height - lift,
            ),
            fill=(255, 255, 255),
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
                fill=(255, 255, 255),
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

        draw.line((center_x - size, center_y, center_x + size, center_y), fill=(255, 255, 255), width=line_width)
        draw.line((center_x, center_y - size, center_x, center_y + size), fill=(255, 255, 255), width=line_width)
        diagonal = max(1, int(round(size * 0.7)))
        draw.line(
            (center_x - diagonal, center_y - diagonal, center_x + diagonal, center_y + diagonal),
            fill=(255, 255, 255),
            width=max(1, line_width - 1),
        )
        draw.line(
            (center_x - diagonal, center_y + diagonal, center_x + diagonal, center_y - diagonal),
            fill=(255, 255, 255),
            width=max(1, line_width - 1),
        )

    def _draw_face_features(
        self,
        draw: object,
        *,
        center_x: int,
        center_y: int,
        status: str,
        animation_frame: int,
        scale: float,
        face_cue: DisplayFaceCue | None,
    ) -> None:
        safe_scale = self._normalise_scale(scale)
        jitter_x, jitter_y = self._face_offset(status, animation_frame, face_cue=face_cue)
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
        face_cue: DisplayFaceCue | None,
    ) -> None:
        center_x, center_y = origin
        eye = self._eye_state(status, animation_frame, side, face_cue=face_cue)
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

        if bool(eye["blink"]):
            draw.arc(
                (
                    center_x - self._scaled_offset(26, scale),
                    center_y - self._scaled_offset(8, scale),
                    center_x + self._scaled_offset(26, scale),
                    center_y + self._scaled_offset(10, scale),
                ),
                start=200,
                end=340,
                fill=(255, 255, 255),
                width=self._scaled_size(5, scale, minimum=2),
            )
            return

        width = self._scaled_size(int(eye["width"]), scale, minimum=8)
        height = self._scaled_size(int(eye["height"]), scale, minimum=8)
        offset_x = self._scaled_offset(int(eye["eye_shift_x"]), scale)
        offset_y = self._scaled_offset(int(eye["eye_shift_y"]), scale)
        box = (
            center_x - (width // 2) + offset_x,
            center_y - (height // 2) + offset_y,
            center_x + (width // 2) + offset_x,
            center_y + (height // 2) + offset_y,
        )
        draw.ellipse(box, fill=(255, 255, 255))
        self._draw_face_eye_highlights(draw, box, eye, scale=scale)

        if bool(eye["lid_arc"]):
            draw.arc(
                (
                    box[0] + self._scaled_offset(4, scale),
                    box[1] - self._scaled_offset(10, scale),
                    box[2] - self._scaled_offset(4, scale),
                    box[1] + self._scaled_offset(18, scale),
                ),
                start=180,
                end=360,
                fill=(255, 255, 255),
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
        eye: dict[str, object],
        line_width: int,
    ) -> None:
        """Draw one stable eyebrow using either the legacy or external style path."""

        brow_y = center_y - self._scaled_offset(52, scale) + self._scaled_offset(int(eye["brow_raise"]), scale)
        half_width = self._scaled_offset(24, scale)
        left_x = center_x - half_width
        right_x = center_x + half_width
        style = str(eye.get("brow_style") or "")
        if not style:
            slant = self._scaled_offset(int(eye["brow_slant"]), scale)
            if side == "left":
                draw.line(
                    (left_x, brow_y + slant, right_x, brow_y - slant),
                    fill=(255, 255, 255),
                    width=line_width,
                )
            else:
                draw.line(
                    (left_x, brow_y - slant, right_x, brow_y + slant),
                    fill=(255, 255, 255),
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
                fill=(255, 255, 255),
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
                fill=(255, 255, 255),
                width=line_width,
            )
            return

        if style == "inward_tilt":
            rise = self._scaled_offset(7, scale)
            if side == "left":
                draw.line((left_x, brow_y - rise, right_x, brow_y + rise), fill=(255, 255, 255), width=line_width)
            else:
                draw.line((left_x, brow_y + rise, right_x, brow_y - rise), fill=(255, 255, 255), width=line_width)
            return

        if style == "outward_tilt":
            rise = self._scaled_offset(7, scale)
            if side == "left":
                draw.line((left_x, brow_y + rise, right_x, brow_y - rise), fill=(255, 255, 255), width=line_width)
            else:
                draw.line((left_x, brow_y - rise, right_x, brow_y + rise), fill=(255, 255, 255), width=line_width)
            return

        draw.line((left_x, brow_y, right_x, brow_y), fill=(255, 255, 255), width=line_width)

    def _draw_face_mouth(
        self,
        draw: object,
        *,
        center_x: int,
        center_y: int,
        status: str,
        animation_frame: int,
        scale: float,
        face_cue: DisplayFaceCue | None,
    ) -> None:
        if face_cue is not None and face_cue.mouth:
            self._draw_face_cue_mouth(
                draw,
                center_x=center_x,
                center_y=center_y,
                animation_frame=animation_frame,
                scale=scale,
                mouth_style=face_cue.mouth,
            )
            return
        line_width = self._scaled_size(4, scale, minimum=2)
        if status == "waiting":
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
                fill=(255, 255, 255),
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
                outline=(255, 255, 255),
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
                fill=(255, 255, 255),
                width=line_width,
            )
            draw.line(
                (
                    center_x + self._scaled_offset(4, scale),
                    center_y + self._scaled_offset(2 + offset_y, scale),
                    center_x + self._scaled_offset(mouth_width, scale),
                    center_y + self._scaled_offset(4 + offset_y, scale),
                ),
                fill=(255, 255, 255),
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
                outline=(255, 255, 255),
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
                fill=(255, 255, 255),
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
                fill=(255, 255, 255),
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
            fill=(255, 255, 255),
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
                fill=(255, 255, 255),
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
                fill=(255, 255, 255),
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
                outline=(255, 255, 255),
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
                outline=(255, 255, 255),
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
                outline=(255, 255, 255),
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
                fill=(255, 255, 255),
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
                fill=(255, 255, 255),
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
                fill=(255, 255, 255),
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
            fill=(255, 255, 255),
            width=line_width,
        )

    def _draw_face_eye_highlights(
        self,
        draw: object,
        box: tuple[int, int, int, int],
        eye: dict[str, object],
        *,
        scale: float,
    ) -> None:
        center_x = (box[0] + box[2]) // 2
        center_y = (box[1] + box[3]) // 2
        main_x = center_x + self._scaled_offset(int(eye["highlight_dx"]), scale)
        main_y = center_y + self._scaled_offset(int(eye["highlight_dy"]), scale)
        main_radius = self._scaled_size(8, scale, minimum=2)
        secondary_x_offset = self._scaled_offset(10, scale)
        secondary_y_offset = self._scaled_offset(8, scale)
        secondary_width = self._scaled_size(6, scale, minimum=2)
        secondary_height = self._scaled_size(6, scale, minimum=2)
        draw.ellipse((main_x - main_radius, main_y - main_radius, main_x + main_radius, main_y + main_radius), fill=(0, 0, 0))
        draw.ellipse(
            (
                main_x + secondary_x_offset,
                main_y + secondary_y_offset,
                main_x + secondary_x_offset + secondary_width,
                main_y + secondary_y_offset + secondary_height,
            ),
            fill=(0, 0, 0),
        )

    def _face_offset(
        self,
        status: str,
        animation_frame: int,
        *,
        face_cue: DisplayFaceCue | None = None,
    ) -> tuple[int, int]:
        if status == "waiting":
            base = (
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
            )[animation_frame % 12]
        elif status == "listening":
            base = ((0, 0), (0, -1), (0, -1), (0, 0), (0, 1), (0, 0))[animation_frame % 6]
        elif status == "processing":
            base = ((0, 0), (-1, 0), (-1, 0), (0, 0), (1, 0), (0, 0))[animation_frame % 6]
        elif status == "answering":
            base = ((0, 0), (0, -1), (0, 0), (0, 1), (0, 0), (0, 0))[animation_frame % 6]
        elif status == "printing":
            base = ((0, 0), (1, 0), (0, 0), (-1, 0), (0, 0), (0, 0))[animation_frame % 6]
        elif status == "error":
            base = ((0, 1), (0, 0), (0, 1), (0, 0), (0, 1), (0, 0))[animation_frame % 6]
        else:
            base = (0, 0)
        if face_cue is None:
            return base
        return (base[0] + face_cue.head_dx, base[1] + face_cue.head_dy)

    def _eye_state(
        self,
        status: str,
        animation_frame: int,
        side: str,
        *,
        face_cue: DisplayFaceCue | None = None,
    ) -> dict[str, object]:
        state: dict[str, object] = {
            "width": 56,
            "height": 74,
            "eye_shift_x": 0,
            "eye_shift_y": 0,
            "highlight_dx": -10,
            "highlight_dy": -18,
            "brow_raise": 0,
            "brow_slant": 4,
            "brow_style": "",
            "blink": False,
            "lid_arc": False,
        }

        if status == "waiting":
            frame = animation_frame % 12
            # HDMI renders the face white-on-black, so calm idle motion reads
            # best when the eye shape stays stable and only gaze/blink drift.
            state["eye_shift_y"] = (-1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)[frame]
            state["highlight_dx"] = (-10, -8, -6, -3, 1, 5, 8, 5, 1, -3, -7, -9)[frame]
            state["blink"] = frame == 9
        elif status == "listening":
            frame = animation_frame % 6
            state["height"] = (78, 80, 82, 80, 78, 76)[frame]
            state["highlight_dx"] = (-8, -7, -6, -5, -6, -7)[frame]
            state["highlight_dy"] = (-18, -19, -20, -19, -18, -17)[frame]
            state["brow_raise"] = -8
            state["brow_slant"] = 2
            state["blink"] = frame == 5
        elif status == "processing":
            frame = animation_frame % 6
            scan = (-12, -8, -3, 3, 8, 12)[frame]
            state["height"] = 68
            state["highlight_dx"] = scan if side == "left" else scan - 2
            state["highlight_dy"] = (-17, -16, -15, -16, -17, -18)[frame]
            state["brow_raise"] = -1
            state["brow_slant"] = 4
        elif status == "answering":
            frame = animation_frame % 6
            state["height"] = (70, 74, 72, 74, 70, 72)[frame]
            state["highlight_dx"] = (-8, -7, -6, -7, -8, -7)[frame]
            state["highlight_dy"] = (-18, -17, -16, -17, -18, -17)[frame]
            state["brow_raise"] = -2
            state["brow_slant"] = 2
        elif status == "printing":
            frame = animation_frame % 6
            state["height"] = (70, 68, 66, 64, 66, 68)[frame]
            state["highlight_dx"] = (-9, -8, -7, -6, -7, -8)[frame]
            state["brow_raise"] = -4
            state["brow_slant"] = 2
            state["blink"] = frame == 4
        elif status == "error":
            frame = animation_frame % 6
            state["width"] = 54
            state["height"] = (60, 58, 56, 58, 60, 58)[frame]
            state["highlight_dx"] = (-12, -11, -10, -9, -10, -11)[frame]
            state["highlight_dy"] = -14
            state["brow_raise"] = 2
            state["brow_slant"] = 8
            state["eye_shift_y"] = 2
            state["blink"] = frame == 3
        return self._apply_face_cue_to_eye_state(state, face_cue=face_cue)

    def _apply_face_cue_to_eye_state(
        self,
        state: dict[str, object],
        *,
        face_cue: DisplayFaceCue | None,
    ) -> dict[str, object]:
        if face_cue is None:
            return state

        merged = dict(state)
        merged["highlight_dx"] = int(merged["highlight_dx"]) + (face_cue.gaze_x * 6)
        merged["highlight_dy"] = int(merged["highlight_dy"]) + (face_cue.gaze_y * 5)
        if face_cue.blink is not None:
            merged["blink"] = face_cue.blink

        if face_cue.brows == "raised":
            merged["brow_raise"] = -11
            merged["brow_slant"] = 0
            merged["brow_style"] = "raised"
        elif face_cue.brows == "soft":
            merged["brow_raise"] = -3
            merged["brow_slant"] = 0
            merged["brow_style"] = "soft"
        elif face_cue.brows == "straight":
            merged["brow_raise"] = 0
            merged["brow_slant"] = 0
            merged["brow_style"] = "straight"
        elif face_cue.brows == "inward_tilt":
            merged["brow_raise"] = 0
            merged["brow_slant"] = 0
            merged["brow_style"] = "inward_tilt"
        elif face_cue.brows == "outward_tilt":
            merged["brow_raise"] = 0
            merged["brow_slant"] = 0
            merged["brow_style"] = "outward_tilt"
        elif face_cue.brows == "roof":
            merged["brow_raise"] = -1
            merged["brow_slant"] = 0
            merged["brow_style"] = "roof"
        return merged

    def _scaled_offset(self, value: int | float, scale: float) -> int:
        return int(round(float(value) * scale))

    def _scaled_size(self, value: int | float, scale: float, *, minimum: int = 1) -> int:
        return max(minimum, int(round(float(value) * scale)))

    def _normalise_scale(self, value: object) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 1.0
        if parsed > 0:
            return parsed
        return 1.0
