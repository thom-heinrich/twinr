"""Construct the configured Twinr display backend."""

from __future__ import annotations

from collections.abc import Callable

from twinr.agent.base_agent.config import TwinrConfig
from twinr.display.contracts import TwinrDisplayAdapter
from twinr.display.hdmi_fbdev import HdmiFramebufferDisplay
from twinr.display.hdmi_wayland import HdmiWaylandDisplay
from twinr.display.waveshare_v2 import WaveshareEPD4In2V2


def create_display_adapter(
    config: TwinrConfig,
    *,
    emit: Callable[[str], None] | None = None,
) -> TwinrDisplayAdapter:
    """Build the display adapter selected by ``config.display_driver``."""

    driver = str(getattr(config, "display_driver", "") or "").strip().lower()
    if driver == "hdmi_wayland":
        return HdmiWaylandDisplay.from_config(config, emit=emit)
    if driver == "hdmi_fbdev":
        return HdmiFramebufferDisplay.from_config(config, emit=emit)
    if driver == "waveshare_4in2_v2":
        return WaveshareEPD4In2V2.from_config(config, emit=emit)
    supported = ", ".join(getattr(config, "supported_display_drivers", ("hdmi_wayland", "hdmi_fbdev", "waveshare_4in2_v2")))
    raise RuntimeError(f"Unsupported display driver `{driver or '<empty>'}`. Supported drivers: {supported}.")
