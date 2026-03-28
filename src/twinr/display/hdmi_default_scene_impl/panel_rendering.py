"""Aggregate the focused HDMI header and reserve-panel rendering mixins."""

from __future__ import annotations

from .header_rendering import HdmiHeaderRenderingMixin
from .reserve_panel_rendering import HdmiReservePanelRenderingMixin
from .reserve_prompt_layout import HdmiReservePromptLayoutMixin


class HdmiPanelRenderingMixin(
    HdmiHeaderRenderingMixin,
    HdmiReservePromptLayoutMixin,
    HdmiReservePanelRenderingMixin,
):
    """Aggregate the HDMI header and reserve-panel rendering helpers."""
