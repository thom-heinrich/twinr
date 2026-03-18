"""Own the native Wayland window used by Twinr's HDMI presentation path."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class HdmiWaylandSurfaceHost:
    """Keep the native Wayland window separate from image rendering logic."""

    emit: Callable[[str], None] | None = None
    _qt_app: Any | None = field(default=None, init=False, repr=False)
    _qt_window: Any | None = field(default=None, init=False, repr=False)
    _qt_image_label: Any | None = field(default=None, init=False, repr=False)
    _last_surface_kind: str = field(default="none", init=False, repr=False)

    @property
    def app(self) -> Any | None:
        return self._qt_app

    @property
    def window(self) -> Any | None:
        return self._qt_window

    @property
    def image_label(self) -> Any | None:
        return self._qt_image_label

    def show_raster_image(
        self,
        *,
        qt_core: Any,
        qt_gui: Any,
        qt_widgets: Any,
        rgba_bytes: bytes,
        size: tuple[int, int],
        socket_path: Path,
    ) -> None:
        """Show one RGBA frame on the raster surface."""

        app, _window, image_label = self._ensure_window(
            qt_core=qt_core,
            qt_widgets=qt_widgets,
            size=size,
            socket_path=socket_path,
        )
        width, height = size
        qimage = qt_gui.QImage(
            rgba_bytes,
            width,
            height,
            width * 4,
            qt_gui.QImage.Format_RGBA8888,
        )
        pixmap = qt_gui.QPixmap.fromImage(qimage)
        image_label.setPixmap(pixmap)
        image_label.resize(width, height)
        image_label.show()
        self._last_surface_kind = "raster"
        if hasattr(app, "processEvents"):
            app.processEvents()

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
        self._qt_app = None
        self._qt_window = None
        self._qt_image_label = None
        self._last_surface_kind = "none"

    def _ensure_window(
        self,
        *,
        qt_core: Any,
        qt_widgets: Any,
        size: tuple[int, int],
        socket_path: Path,
    ) -> tuple[Any, Any, Any]:
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
        self._safe_emit(
            " ".join(
                (
                    "display_wayland=ready",
                    f"socket={socket_path}",
                    f"size={current_size[0]}x{current_size[1]}",
                    "toolkit=pyqt5",
                    "surface_host=qt_raster",
                )
            )
        )
        return (app, window, image_label)

    def _safe_emit(self, line: str) -> None:
        if self.emit is None:
            return
        try:
            self.emit(line)
        except Exception:
            return
