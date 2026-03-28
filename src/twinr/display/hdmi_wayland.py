# CHANGELOG: 2026-03-28
# BUG-1: close() and tick() now share the same lock as show_image() to remove teardown/render races.
# BUG-2: tick() no longer swallows every Qt exception; display freezes now fail loudly instead of silently.
# BUG-3: direct construction now accepts str framebuffer paths and None/empty rotation values without crashing.
# SEC-1: WAYLAND_DISPLAY and the resolved Wayland runtime/socket are now validated against practical local spoofing / misrouting risks.
# IMP-1: Prefer Qt 6 bindings (PySide6, then PyQt6) with PyQt5 fallback; Qt 5.15 is no longer the default path.
# IMP-2: Replaced per-frame opaque host handoff with a persistent fullscreen Qt widget, duplicate-frame suppression, and explicit screen-optimized pixmap updates.
# BREAKING: show_image(), tick(), and close() now require the main thread once Qt state exists, because QWidget/QPixmap are not safe off the GUI thread.
# BREAKING: insecure WAYLAND_DISPLAY values or weak XDG runtime permissions now raise RuntimeError unless TWINR_ALLOW_INSECURE_WAYLAND=1 is set.

"""Present Twinr status screens as a visible fullscreen Wayland surface."""

from __future__ import annotations

import importlib
import os
import re
import stat
import sys
import threading
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


_WAYLAND_DISPLAY_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_ALLOW_INSECURE_WAYLAND_ENV = "TWINR_ALLOW_INSECURE_WAYLAND"
_QT_BINDING_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("pyside6", "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets"),
    ("pyqt6", "PyQt6.QtCore", "PyQt6.QtGui", "PyQt6.QtWidgets"),
    ("pyqt5", "PyQt5.QtCore", "PyQt5.QtGui", "PyQt5.QtWidgets"),
)


def _qt_enum(root: Any, name: str, nested_enum: str | None = None) -> Any:
    value = getattr(root, name, None)
    if value is not None:
        return value
    if nested_enum:
        nested = getattr(root, nested_enum, None)
        if nested is not None:
            value = getattr(nested, name, None)
            if value is not None:
                return value
    raise AttributeError(f"Qt enum `{name}` is not available on {root!r}.")


def _qimage_format_rgba8888(qt_gui: Any) -> Any:
    image_cls = qt_gui.QImage
    direct = getattr(image_cls, "Format_RGBA8888", None)
    if direct is not None:
        return direct
    enum_root = getattr(image_cls, "Format", None)
    if enum_root is not None:
        nested = getattr(enum_root, "Format_RGBA8888", None)
        if nested is not None:
            return nested
    raise RuntimeError("The available Qt build does not expose QImage.Format_RGBA8888.")


def _ordered_qt_specs(preferred_api: str) -> tuple[tuple[str, str, str, str], ...]:
    specs = list(_QT_BINDING_SPECS)
    if preferred_api != "auto":
        specs.sort(key=lambda spec: 0 if spec[0] == preferred_api else 1)
        return tuple(specs)

    loaded: list[tuple[str, str, str, str]] = []
    unloaded: list[tuple[str, str, str, str]] = []
    for spec in specs:
        if any(module_name in sys.modules for module_name in spec[1:]):
            loaded.append(spec)
        else:
            unloaded.append(spec)
    return tuple(loaded + unloaded)


def _build_wayland_window_class(qt_core: Any, qt_gui: Any, qt_widgets: Any) -> type[Any]:
    QWidget = qt_widgets.QWidget
    QPixmap = qt_gui.QPixmap
    QImage = qt_gui.QImage
    QPainter = qt_gui.QPainter
    QColor = qt_gui.QColor
    QCursor = qt_gui.QCursor

    frameless = _qt_enum(qt_core.Qt, "FramelessWindowHint", "WindowType")
    stays_on_top = _qt_enum(qt_core.Qt, "WindowStaysOnTopHint", "WindowType")
    opaque_paint = _qt_enum(qt_core.Qt, "WA_OpaquePaintEvent", "WidgetAttribute")
    no_system_background = _qt_enum(qt_core.Qt, "WA_NoSystemBackground", "WidgetAttribute")
    blank_cursor = _qt_enum(qt_core.Qt, "BlankCursor", "CursorShape")
    smooth_pixmap_transform = _qt_enum(qt_gui.QPainter, "SmoothPixmapTransform", "RenderHint")
    rgba8888 = _qimage_format_rgba8888(qt_gui)

    class TwinrWaylandRasterWindow(QWidget):
        def __init__(self, logical_width: int, logical_height: int) -> None:
            super().__init__(None)
            self._pixmap = QPixmap()
            self._upload_bytes: bytes | None = None
            self._background = QColor(0, 0, 0)
            self.setObjectName("TwinrWaylandDisplay")
            self.setWindowTitle("Twinr")
            self.setWindowFlag(frameless, True)
            self.setWindowFlag(stays_on_top, True)
            self.setAttribute(opaque_paint, True)
            self.setAttribute(no_system_background, True)
            self.setAutoFillBackground(False)
            self.setCursor(QCursor(blank_cursor))
            self.resize(max(1, logical_width), max(1, logical_height))

        def set_frame(self, rgba_bytes: bytes, width: int, height: int) -> None:
            if width <= 0 or height <= 0:
                return
            stride = width * 4
            self._upload_bytes = rgba_bytes
            image = QImage(self._upload_bytes, width, height, stride, rgba8888)
            self._pixmap = QPixmap.fromImage(image)
            self.update()

        def clear_frame(self) -> None:
            self._upload_bytes = None
            self._pixmap = QPixmap()
            self.update()

        def ensure_visible(self) -> None:
            if not self.isVisible():
                self.showFullScreen()
            self.raise_()
            self.activateWindow()

        def paintEvent(self, event: Any) -> None:  # pragma: no cover - requires Qt runtime
            painter = QPainter(self)
            painter.fillRect(self.rect(), self._background)
            if not self._pixmap.isNull():
                painter.setRenderHint(smooth_pixmap_transform, False)
                painter.drawPixmap(self.rect(), self._pixmap)

    return TwinrWaylandRasterWindow


def _supports_native_raster_window(qt_gui: Any, qt_widgets: Any) -> bool:
    widget_cls = getattr(qt_widgets, "QWidget", None)
    if widget_cls is None:
        return False
    if any(getattr(qt_gui, name, None) is None for name in ("QPainter", "QColor", "QCursor")):
        return False
    return all(
        hasattr(widget_cls, attribute)
        for attribute in ("setWindowFlag", "setAttribute", "setAutoFillBackground", "setCursor")
    )


@dataclass(slots=True)
class HdmiWaylandDisplay(HdmiFramebufferDisplay):
    """Render Twinr HDMI frames into a fullscreen Wayland window."""

    driver: str = "hdmi_wayland"
    wayland_display: str = "wayland-0"
    wayland_runtime_dir: str | None = None
    qt_api: str = "auto"
    _qt_binding_name: str | None = field(default=None, init=False, repr=False)
    _qt_modules: tuple[Any, Any, Any] | None = field(default=None, init=False, repr=False)
    _qt_app: Any | None = field(default=None, init=False, repr=False)
    _qt_window: Any | None = field(default=None, init=False, repr=False)
    _qt_window_class: type[Any] | None = field(default=None, init=False, repr=False)
    _qt_image_label: Any | None = field(default=None, init=False, repr=False)
    _qt_image_bytes: bytes | None = field(default=None, init=False, repr=False)
    _last_frame_size: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _last_socket_path: str | None = field(default=None, init=False, repr=False)
    _owns_qt_app: bool = field(default=False, init=False, repr=False)
    _qt_thread_ident: int | None = field(default=None, init=False, repr=False)
    _surface_host: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.framebuffer_path = Path(self.framebuffer_path).expanduser()
        self.rotation_degrees = int(self.rotation_degrees or 0) % 360
        self.layout_mode = self._normalise_layout_mode(self.layout_mode)
        self.wayland_display = str(self.wayland_display or "wayland-0").strip() or "wayland-0"
        self.wayland_runtime_dir = (
            str(Path(self.wayland_runtime_dir).expanduser())
            if self.wayland_runtime_dir
            else None
        )
        self.qt_api = str(self.qt_api or "auto").strip().lower() or "auto"
        self.width = max(1, int(self.width or 800))
        self.height = max(1, int(self.height or 480))

        if self.driver != "hdmi_wayland":
            raise RuntimeError(f"HdmiWaylandDisplay does not support driver `{self.driver}`.")
        if self.rotation_degrees not in _SUPPORTED_ROTATIONS:
            raise RuntimeError("Display rotation must be one of 0, 90, 180, or 270 degrees.")
        if self.qt_api not in {"auto", "pyside6", "pyqt6", "pyqt5"}:
            raise RuntimeError("display_qt_api must be one of auto, pyside6, pyqt6, or pyqt5.")
        self._validate_wayland_display_name(self.wayland_display)

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
            driver=getattr(config, "display_driver", "hdmi_wayland") or "hdmi_wayland",
            width=max(1, int(getattr(config, "display_width", 0) or 800)),
            height=max(1, int(getattr(config, "display_height", 0) or 480)),
            rotation_degrees=int(getattr(config, "display_rotation_degrees", 0) or 0),
            layout_mode=getattr(config, "display_layout", None),
            wayland_display=getattr(config, "display_wayland_display", "wayland-0") or "wayland-0",
            wayland_runtime_dir=getattr(config, "display_wayland_runtime_dir", None),
            qt_api=getattr(config, "display_qt_api", "auto") or "auto",
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
            self._assert_gui_thread("HdmiWaylandDisplay.show_image")
            prepared = self._prepare_image(image)
            width, height = prepared.size
            rgba_bytes = prepared.tobytes("raw", "RGBA")
            qt_core, qt_gui, qt_widgets = self._load_qt()
            if self._qt_image_bytes == rgba_bytes and self._last_frame_size == (width, height):
                self.tick()
                return
            if not _supports_native_raster_window(qt_gui, qt_widgets):
                socket_path = Path(self._prepare_wayland_socket())
                host = self._surface_host
                if not isinstance(host, HdmiWaylandSurfaceHost):
                    host = HdmiWaylandSurfaceHost(emit=self.emit)
                    self._surface_host = host
                host.show_raster_image(
                    qt_core=qt_core,
                    qt_gui=qt_gui,
                    qt_widgets=qt_widgets,
                    rgba_bytes=rgba_bytes,
                    size=(width, height),
                    socket_path=socket_path,
                )
                self._qt_app = host.app
                self._qt_window = host.window
                self._qt_image_label = host.image_label
                self._qt_image_bytes = rgba_bytes
                self._last_frame_size = (width, height)
                self.tick()
                return
            window = self._ensure_qt_window()
            window.ensure_visible()

            window.set_frame(rgba_bytes, width, height)
            self._qt_image_label = None
            self._qt_image_bytes = rgba_bytes
            self._last_frame_size = (width, height)
            self.tick()

    def tick(self) -> None:
        """Keep the Wayland event queue responsive between rerenders."""

        with self._lock:
            app = self._qt_app
            if app is None:
                return
            self._assert_gui_thread("HdmiWaylandDisplay.tick")
            try:
                app.processEvents()
            except Exception as exc:  # pragma: no cover - requires Qt runtime
                self._emit_message(f"hdmi_wayland tick failed: {exc}")
                raise

    def close(self) -> None:
        """Release the Wayland window cleanly."""

        error: Exception | None = None
        with self._lock:
            if self._qt_app is not None or self._qt_window is not None:
                self._assert_gui_thread("HdmiWaylandDisplay.close")
            try:
                if isinstance(self._surface_host, HdmiWaylandSurfaceHost):
                    self._surface_host.close()
                elif self._qt_window is not None:
                    self._qt_window.clear_frame()
                    self._qt_window.close()
                if self._qt_app is not None:
                    self._qt_app.processEvents()
            except Exception as exc:  # pragma: no cover - requires Qt runtime
                error = exc
                self._emit_message(f"hdmi_wayland close failed: {exc}")
            finally:
                self._qt_window = None
                self._qt_image_label = None
                self._qt_image_bytes = None
                self._last_frame_size = None
                self._last_socket_path = None
                self._surface_host = None
                self._qt_app = None
                self._owns_qt_app = False
                self._qt_thread_ident = None
                self._qt_modules = None
                self._qt_binding_name = None
                self._qt_window_class = None
                HdmiFramebufferDisplay.close(self)
        if error is not None:
            raise error

    def _ensure_qt_window(self) -> Any:
        socket_path = self._prepare_wayland_socket()
        qt_core, qt_gui, qt_widgets = self._load_qt()
        self._ensure_qt_app(qt_widgets)

        if self._qt_window_class is None:
            self._qt_window_class = _build_wayland_window_class(qt_core, qt_gui, qt_widgets)

        window = self._qt_window
        if window is None:
            window = self._qt_window_class(self.width, self.height)
            self._qt_window = window
            self._surface_host = window
            self._emit_message(
                f"hdmi_wayland ready via {self._qt_binding_name or 'qt'} on {socket_path}"
            )
        return window

    def _prepare_wayland_socket(self) -> str:
        if self._last_socket_path:
            cached = Path(self._last_socket_path)
            if cached.exists():
                self._validate_wayland_socket(cached)
                return self._last_socket_path

        os.environ.setdefault("QT_QPA_PLATFORM", "wayland")
        socket_path = apply_wayland_environment(
            self.wayland_display,
            configured_runtime_dir=self.wayland_runtime_dir,
        )
        socket_path = Path(str(socket_path)).expanduser()
        self._validate_wayland_socket(socket_path)
        self._last_socket_path = str(socket_path)
        return self._last_socket_path

    def _ensure_qt_app(self, qt_widgets: Any) -> Any:
        app = qt_widgets.QApplication.instance()
        if app is None:
            app = qt_widgets.QApplication(["twinr-hdmi-wayland"])
            app.setQuitOnLastWindowClosed(False)
            self._owns_qt_app = True
        else:
            self._owns_qt_app = False

        platform_name = ""
        platform_name_getter = getattr(app, "platformName", None)
        if callable(platform_name_getter):
            try:
                platform_name = str(platform_name_getter() or "").lower()
            except Exception:
                platform_name = ""
        if platform_name and "wayland" not in platform_name:
            raise RuntimeError(
                "hdmi_wayland requires a Wayland Qt platform plugin, "
                f"but the existing QApplication is using `{platform_name}`."
            )

        self._qt_app = app
        return app

    def _load_qt(self) -> tuple[Any, Any, Any]:
        modules = self._qt_modules
        if modules is not None:
            return modules

        self._prepare_wayland_socket()
        last_error: ImportError | None = None
        for binding_name, core_name, gui_name, widgets_name in _ordered_qt_specs(self.qt_api):
            try:
                qt_core = importlib.import_module(core_name)
                qt_gui = importlib.import_module(gui_name)
                qt_widgets = importlib.import_module(widgets_name)
            except ImportError as exc:
                last_error = exc
                continue

            self._qt_binding_name = binding_name
            self._qt_modules = (qt_core, qt_gui, qt_widgets)
            return self._qt_modules

        raise RuntimeError(
            "A Qt binding is required for the hdmi_wayland backend. "
            "Install PySide6 (preferred), PyQt6, or PyQt5."
        ) from last_error

    def _validate_wayland_display_name(self, name: str) -> None:
        # BREAKING: reject path-based / traversal-style WAYLAND_DISPLAY values by default.
        if os.environ.get(_ALLOW_INSECURE_WAYLAND_ENV) == "1":
            return
        if "/" in name or not _WAYLAND_DISPLAY_RE.fullmatch(name):
            raise RuntimeError(
                "Unsafe WAYLAND_DISPLAY value. Only plain socket names such as `wayland-0` "
                f"are accepted by default; got `{name}`. Set {_ALLOW_INSECURE_WAYLAND_ENV}=1 "
                "to bypass this check."
            )

    def _validate_wayland_socket(self, socket_path: Path) -> None:
        # BREAKING: fail closed on weak or spoofable Wayland runtime/socket paths unless explicitly overridden.
        if not socket_path.exists():
            raise RuntimeError(f"Wayland socket does not exist: {socket_path}")

        runtime_dir = socket_path.parent
        runtime_stat = runtime_dir.stat()
        socket_stat = socket_path.stat()
        current_uid = os.getuid()
        allow_insecure = os.environ.get(_ALLOW_INSECURE_WAYLAND_ENV) == "1"

        if not stat.S_ISDIR(runtime_stat.st_mode):
            raise RuntimeError(f"Wayland runtime path is not a directory: {runtime_dir}")
        if not stat.S_ISSOCK(socket_stat.st_mode):
            raise RuntimeError(f"Wayland socket path is not a Unix socket: {socket_path}")

        if allow_insecure:
            return

        runtime_mode = stat.S_IMODE(runtime_stat.st_mode)
        if runtime_dir.is_symlink():
            raise RuntimeError(f"Wayland runtime dir must not be a symlink: {runtime_dir}")
        if socket_path.is_symlink():
            raise RuntimeError(f"Wayland socket must not be a symlink: {socket_path}")
        if runtime_stat.st_uid != current_uid and current_uid != 0:
            raise RuntimeError(
                f"Wayland runtime dir `{runtime_dir}` is owned by uid {runtime_stat.st_uid}, "
                f"expected uid {current_uid}."
            )
        if runtime_mode != 0o700:
            raise RuntimeError(
                f"Wayland runtime dir `{runtime_dir}` has mode {runtime_mode:o}; expected 700."
            )

    def _assert_gui_thread(self, operation: str) -> None:
        # Keep all Qt access on one dedicated thread, but allow the display companion to own that thread.
        current_ident = threading.get_ident()
        owner_ident = self._qt_thread_ident
        if owner_ident is None:
            self._qt_thread_ident = current_ident
            return
        if owner_ident != current_ident:
            raise RuntimeError(
                f"{operation} must run on the Qt/Wayland display thread."
            )

    def _emit_message(self, message: str) -> None:
        emit = self.emit
        if emit is None:
            return
        try:
            emit(message)
        except Exception:
            return
