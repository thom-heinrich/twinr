"""Expose the decomposed HDMI default-scene implementation package lazily."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import (
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
    from .renderer import HdmiDefaultSceneRenderer

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

_EXPORTS = {
    "HdmiDefaultScene": "twinr.display.hdmi_default_scene_impl.models",
    "HdmiDefaultSceneLayout": "twinr.display.hdmi_default_scene_impl.models",
    "HdmiDefaultSceneRenderer": "twinr.display.hdmi_default_scene_impl.renderer",
    "HdmiHeaderModel": "twinr.display.hdmi_default_scene_impl.models",
    "HdmiNewsTickerModel": "twinr.display.hdmi_default_scene_impl.models",
    "HdmiSceneCard": "twinr.display.hdmi_default_scene_impl.models",
    "HdmiStatusPanelModel": "twinr.display.hdmi_default_scene_impl.models",
    "display_state_value": "twinr.display.hdmi_default_scene_impl.models",
    "order_state_fields": "twinr.display.hdmi_default_scene_impl.models",
    "state_field_value": "twinr.display.hdmi_default_scene_impl.models",
    "state_value_color": "twinr.display.hdmi_default_scene_impl.models",
    "status_accent_color": "twinr.display.hdmi_default_scene_impl.models",
    "status_headline": "twinr.display.hdmi_default_scene_impl.models",
    "status_helper_text": "twinr.display.hdmi_default_scene_impl.models",
    "time_value": "twinr.display.hdmi_default_scene_impl.models",
}


def __getattr__(name: str) -> object:
    """Resolve public HDMI scene symbols lazily from their owning modules."""

    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
