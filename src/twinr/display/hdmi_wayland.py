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
            app, _window, image_label = self._ensure_window(qt_core, qt_widgets, prepared.size)
            width, height = prepared.size
            self._qt_image_bytes = prepared.tobytes("raw", "RGBA")
            qimage = qt_gui.QImage(
                self._qt_image_bytes,
                width,
                height,
                width * 4,
                qt_gui.QImage.Format_RGBA8888,
            )
            pixmap = qt_gui.QPixmap.fromImage(qimage)
            image_label.setPixmap(pixmap)
            image_label.resize(width, height)
            image_label.show()
            if hasattr(app, "processEvents"):
                app.processEvents()
            self.tick()

    def tick(self) -> None:
        """Keep the Wayland event queue responsive between rerenders."""

        app = self._qt_app
        if app is None:
            return
        try:
            app.processEvents()
        except Exception:
            return

    def close(self) -> None:
        """Release the Wayland window cleanly."""

        window = self._qt_window
        if window is not None and hasattr(window, "close"):
            try:
                window.close()
            except Exception:
                pass
        app = self._qt_app
        if app is not None and hasattr(app, "processEvents"):
            try:
                app.processEvents()
            except Exception:
                pass
        self._qt_modules = None
        self._qt_app = None
        self._qt_window = None
        self._qt_image_label = None
        self._qt_image_bytes = None
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

    def _ensure_window(self, qt_core: Any, qt_widgets: Any, size: tuple[int, int]):
        window = self._qt_window
        current_size = tuple(int(part) for part in size)
        image_label = self._qt_image_label
        if window is not None and image_label is not None:
            return (self._qt_app, window, image_label)

        app = qt_widgets.QApplication.instance()
        if app is None:
            app = qt_widgets.QApplication([])
            if hasattr(app, "setQuitOnLastWindowClosed"):
                app.setQuitOnLastWindowClosed(False)

        window = qt_widgets.QWidget()
        window.setWindowTitle("TWINR")
        flags = qt_core.Qt.Window | qt_core.Qt.FramelessWindowHint | qt_core.Qt.WindowStaysOnTopHint
        if hasattr(window, "setWindowFlags"):
            window.setWindowFlags(flags)
        if hasattr(window, "setStyleSheet"):
            window.setStyleSheet("background: rgb(10, 18, 32);")

        image_label = qt_widgets.QLabel(window)
        if hasattr(image_label, "setScaledContents"):
            image_label.setScaledContents(True)
        if hasattr(image_label, "setAlignment"):
            image_label.setAlignment(qt_core.Qt.AlignCenter)
        image_label.setGeometry(0, 0, current_size[0], current_size[1])
        if hasattr(window, "resize"):
            window.resize(current_size[0], current_size[1])
        window.showFullScreen()
        if hasattr(window, "raise_"):
            window.raise_()
        if hasattr(window, "activateWindow"):
            window.activateWindow()

        self._qt_app = app
        self._qt_window = window
        self._qt_image_label = image_label
        socket_path = apply_wayland_environment(
            self.wayland_display,
            configured_runtime_dir=self.wayland_runtime_dir,
        )
        self._safe_emit(
            " ".join(
                (
                    "display_wayland=ready",
                    f"socket={socket_path}",
                    f"size={current_size[0]}x{current_size[1]}",
                    "toolkit=pyqt5",
                )
            )
        )
        return (app, window, image_label)
