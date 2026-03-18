"""Present Twinr status screens as a visible fullscreen Wayland surface."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from twinr.agent.base_agent.config import TwinrConfig
from twinr.display.hdmi_fbdev import (
    _SUPPORTED_ROTATIONS,
    FramebufferBitfield,
    FramebufferGeometry,
    HdmiFramebufferDisplay,
)
from twinr.display.wayland_env import apply_wayland_environment
from twinr.display.wayland_surface_host import HdmiWaylandSurfaceHost


@dataclass(slots=True)
class HdmiWaylandDisplay(HdmiFramebufferDisplay):
    """Render Twinr HDMI frames into a fullscreen Wayland window."""

    driver: str = "hdmi_wayland"
    wayland_display: str = "wayland-0"
    wayland_runtime_dir: str | None = None
    _qt_modules: tuple[Any, Any, Any] | None = field(default=None, init=False, repr=False)
    _qt_app: Any | None = field(default=None, init=False, repr=False)
    _qt_window: Any | None = field(default=None, init=False, repr=False)
    _qt_image_label: Any | None = field(default=None, init=False, repr=False)
    _qt_image_bytes: bytes | None = field(default=None, init=False, repr=False)
    _surface_host: HdmiWaylandSurfaceHost | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.framebuffer_path = self.framebuffer_path.expanduser()
        self.rotation_degrees = self.rotation_degrees % 360
        self.layout_mode = self._normalise_layout_mode(self.layout_mode)
        self.wayland_display = str(self.wayland_display or "wayland-0").strip() or "wayland-0"
        self.wayland_runtime_dir = (
            str(Path(self.wayland_runtime_dir).expanduser())
            if self.wayland_runtime_dir
            else None
        )
        self.width = max(1, int(self.width or 800))
        self.height = max(1, int(self.height or 480))
        if self.driver != "hdmi_wayland":
            raise RuntimeError(f"HdmiWaylandDisplay does not support driver `{self.driver}`.")
        if self.rotation_degrees not in _SUPPORTED_ROTATIONS:
            raise RuntimeError("Display rotation must be one of 0, 90, 180, or 270 degrees.")
        self._surface_host = HdmiWaylandSurfaceHost(emit=self.emit)

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        emit: Callable[[str], None] | None = None,
    ) -> "HdmiWaylandDisplay":
        """Build a Wayland HDMI adapter from Twinr configuration."""

        return cls(
            framebuffer_path=Path(getattr(config, "display_fb_path", "/dev/fb0") or "/dev/fb0"),
            driver=config.display_driver,
            width=max(1, int(getattr(config, "display_width", 0) or 800)),
            height=max(1, int(getattr(config, "display_height", 0) or 480)),
            rotation_degrees=config.display_rotation_degrees,
            layout_mode=config.display_layout,
            wayland_display=getattr(config, "display_wayland_display", "wayland-0") or "wayland-0",
            wayland_runtime_dir=getattr(config, "display_wayland_runtime_dir", None),
            emit=emit,
        )

    @property
    def geometry(self) -> FramebufferGeometry:
        """Return the logical window geometry used for rendering."""

        if self._geometry is None:
            self._geometry = FramebufferGeometry(
                width=self.width,
                height=self.height,
                bits_per_pixel=32,
                line_length=self.width * 4,
                red=FramebufferBitfield(offset=16, length=8, msb_right=0),
                green=FramebufferBitfield(offset=8, length=8, msb_right=0),
                blue=FramebufferBitfield(offset=0, length=8, msb_right=0),
                transp=FramebufferBitfield(offset=24, length=8, msb_right=0),
            )
        return self._geometry

    def show_image(self, image: object) -> None:
        """Blit one prepared frame into the fullscreen Wayland window."""

        with self._lock:
            prepared = self._prepare_image(image)
            qt_core, qt_gui, qt_widgets = self._load_qt()
            host = self._surface_host or HdmiWaylandSurfaceHost(emit=self.emit)
            self._surface_host = host
            width, height = prepared.size
            self._qt_image_bytes = prepared.tobytes("raw", "RGBA")
            socket_path = apply_wayland_environment(
                self.wayland_display,
                configured_runtime_dir=self.wayland_runtime_dir,
            )
            host.show_raster_image(
                qt_core=qt_core,
                qt_gui=qt_gui,
                qt_widgets=qt_widgets,
                rgba_bytes=self._qt_image_bytes,
                size=(width, height),
                socket_path=socket_path,
            )
            self._qt_app = host.app
            self._qt_window = host.window
            self._qt_image_label = host.image_label
            self.tick()

    def tick(self) -> None:
        """Keep the Wayland event queue responsive between rerenders."""

        app = self._qt_app
        if app is None and self._surface_host is None:
            return
        try:
            if self._surface_host is not None:
                self._surface_host.tick()
            elif app is not None:
                app.processEvents()
        except Exception:
            return

    def close(self) -> None:
        """Release the Wayland window cleanly."""

        host = self._surface_host
        if host is not None:
            host.close()
        self._qt_modules = None
        self._qt_app = None
        self._qt_window = None
        self._qt_image_label = None
        self._qt_image_bytes = None
        self._surface_host = None
        HdmiFramebufferDisplay.close(self)

    def _load_qt(self) -> tuple[Any, Any, Any]:
        modules = self._qt_modules
        if modules is not None:
            return modules
        apply_wayland_environment(
            self.wayland_display,
            configured_runtime_dir=self.wayland_runtime_dir,
        )
        try:
            from PyQt5 import QtCore, QtGui, QtWidgets
        except ImportError as exc:  # pragma: no cover - environment issue
            raise RuntimeError("PyQt5 is required for the hdmi_wayland display backend.") from exc
        self._qt_modules = (QtCore, QtGui, QtWidgets)
        return self._qt_modules
