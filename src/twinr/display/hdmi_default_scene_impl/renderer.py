"""Public renderer implementation for the decomposed HDMI default scene."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from twinr.display.hdmi_ambient_moments import HdmiAmbientMomentDirector

from .ambient_rendering import HdmiAmbientRenderingMixin
from .face_logic import HdmiFaceLogicMixin
from .face_rendering import HdmiFaceRenderingMixin
from .models import _HdmiSceneTools
from .panel_rendering import HdmiPanelRenderingMixin
from .presentation_rendering import HdmiPresentationRenderingMixin
from .scene_builder import HdmiSceneBuilderMixin
from .typing_contracts import (
    HdmiCanvasDrawSurface,
    HdmiFaceCueLike,
    HdmiImageSurface,
)


@dataclass(slots=True)
class HdmiDefaultSceneRenderer(
    HdmiSceneBuilderMixin,
    HdmiPanelRenderingMixin,
    HdmiPresentationRenderingMixin,
    HdmiAmbientRenderingMixin,
    HdmiFaceLogicMixin,
    HdmiFaceRenderingMixin,
):
    """Render Twinr's modular default HDMI scene."""

    tools: _HdmiSceneTools
    ambient_director: HdmiAmbientMomentDirector = field(default_factory=HdmiAmbientMomentDirector)

    def draw(
        self,
        image: HdmiImageSurface,
        draw: HdmiCanvasDrawSurface,
        *,
        width: int,
        height: int,
        status: str,
        headline: str,
        helper_text: str,
        state_fields,
        debug_signals=(),
        animation_frame: int = 0,
        ticker_text: str | None = None,
        face_cue: HdmiFaceCueLike | None = None,
        emoji_cue=None,
        ambient_impulse_cue=None,
        service_connect_cue=None,
        presentation_cue=None,
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
            debug_signals=debug_signals,
            animation_frame=animation_frame,
            face_cue=face_cue,
            emoji_cue=emoji_cue,
            ambient_impulse_cue=ambient_impulse_cue,
            service_connect_cue=service_connect_cue,
            presentation_cue=presentation_cue,
            presentation_now=presentation_now,
            ambient_now=ambient_now,
        )
        draw.rectangle((0, 0, width, height), fill=(0, 0, 0))
        self._draw_twinr_header(draw, box=scene.layout.header_box, header=scene.header)
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
        reserve_bus = scene.reserve_bus
        if reserve_bus is None or reserve_bus.owner != "emoji":
            self._draw_status_panel(
                draw,
                image=image,
                box=scene.layout.panel_box,
                panel=scene.panel,
                compact=scene.layout.compact_panel,
            )
        if reserve_bus is not None and reserve_bus.owner == "emoji" and reserve_bus.emoji_cue is not None:
            self._draw_emoji_reserve(image, draw, box=scene.layout.panel_box, emoji_cue=reserve_bus.emoji_cue)
        if scene.ticker is not None:
            self._draw_news_ticker(draw, box=scene.layout.ticker_box, ticker=scene.ticker)
