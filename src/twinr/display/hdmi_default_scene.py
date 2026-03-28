"""Expose the stable public HDMI default-scene API over the decomposed stack.

This module is intentionally thin. The implementation now lives under
``twinr.display.hdmi_default_scene_impl`` so the public import path stays
stable while the scene renderer stays split by concern.
"""

from __future__ import annotations

##REFACTOR: 2026-03-27##

from .hdmi_default_scene_impl.models import (
    HdmiDefaultScene,
    HdmiDefaultSceneLayout,
    HdmiHeaderModel,
    HdmiNewsTickerModel,
    HdmiSceneCard,
    HdmiStatusPanelModel,
    display_state_value,
    order_state_fields,
    state_field_value,
    state_value_color,
    status_accent_color,
    status_headline,
    status_helper_text,
    time_value,
)
from .hdmi_default_scene_impl.renderer import (
    HdmiDefaultSceneRenderer as _HdmiDefaultSceneRendererImpl,
)


class HdmiDefaultSceneRenderer(_HdmiDefaultSceneRendererImpl):
    """Preserve the legacy module import path for the HDMI default scene renderer."""


__all__ = [
    "HdmiDefaultScene",
    "HdmiDefaultSceneLayout",
    "HdmiDefaultSceneRenderer",
    "HdmiHeaderModel",
    "HdmiNewsTickerModel",
    "HdmiSceneCard",
    "HdmiStatusPanelModel",
    "display_state_value",
    "order_state_fields",
    "state_field_value",
    "state_value_color",
    "status_accent_color",
    "status_headline",
    "status_helper_text",
    "time_value",
]

for _exported in (
    HdmiDefaultScene,
    HdmiDefaultSceneLayout,
    HdmiDefaultSceneRenderer,
    HdmiHeaderModel,
    HdmiNewsTickerModel,
    HdmiSceneCard,
    HdmiStatusPanelModel,
):
    try:
        _exported.__module__ = __name__
    except (AttributeError, TypeError):
        continue
