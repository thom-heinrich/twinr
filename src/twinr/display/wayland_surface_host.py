# CHANGELOG: 2026-03-28
# BUG-1: Honor socket_path before QApplication creation so the host binds to the intended Wayland compositor instead of silently using the ambient/default socket.
# BUG-2: Replace manual child geometry with a zero-margin layout so the raster surface really fills the fullscreen window across resizes, fullscreen transitions, and HiDPI/fractional-scaling setups.
# BUG-3: Validate frame size and RGBA buffer length before constructing QImage to prevent corrupt output, undefined reads, and hard crashes on malformed frames.
# BUG-4: Refuse GUI initialization/rendering from a non-GUI thread, turning an intermittent Qt crash/race into a deterministic runtime error.
# BUG-5: Emit truthful toolkit/platform metadata instead of the hard-coded toolkit=pyqt5 false positive.
# SEC-1: Sanitize emitted key=value telemetry so crafted socket paths cannot inject extra log/telemetry lines.
# SEC-2: Reject control characters in socket_path and short frame buffers before they reach the native Qt/Wayland layer.
# IMP-1: Upgrade internals to be Qt6-first while remaining compatible with PySide6/PyQt6/PyQt5/PySide2 enum layouts.
# IMP-2: Force a native top-level window, choose the best matching screen for the requested HDMI frame size, and keep the window non-activating/no-focus for passive presentation usage.
# IMP-3: Add accessibility/test identifiers and richer Wayland readiness metadata for modern embedded deployment and diagnostics.
# BREAKING: If a QWidget-compatible QApplication is not available, if the requested Wayland socket changes after Qt GUI startup, or if rendering is attempted from a non-GUI thread, this host now raises RuntimeError instead of silently rendering to the wrong display or crashing later.

"""Own the native Wayland window used by Twinr's HDMI presentation path."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote


@dataclass(slots=True)
class HdmiWaylandSurfaceHost:
    """Keep the native Wayland window separate from image rendering logic."""

    emit: Callable[[str], None] | None = None
    _qt_app: Any | None = field(default=None, init=False, repr=False)
    _qt_window: Any | None = field(default=None, init=False, repr=False)
    _qt_image_label: Any | None = field(default=None, init=False, repr=False)
    _qt_core_module: Any | None = field(default=None, init=False, repr=False)
    _last_surface_kind: str = field(default="none", init=False, repr=False)
    _wayland_display: str | None = field(default=None, init=False, repr=False)
    _toolkit_name: str = field(default="qt_python", init=False, repr=False)

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
        rgba_bytes: bytes | bytearray | memoryview,
        size: tuple[int, int],
        socket_path: Path,
    ) -> None:
        """Show one RGBA frame on the raster surface."""

        width, height = self._validate_size(size)
        frame_bytes = self._normalize_rgba_bytes(rgba_bytes, width=width, height=height)
        app, window, image_label = self._ensure_window(
            qt_core=qt_core,
            qt_widgets=qt_widgets,
            size=(width, height),
            socket_path=socket_path,
        )
        self._assert_gui_thread(qt_core=qt_core, app=app)

        qimage_format = self._resolve_qimage_format(qt_gui)
        qimage = qt_gui.QImage(
            frame_bytes,
            width,
            height,
            width * 4,
            qimage_format,
        )
        if hasattr(qimage, "isNull") and qimage.isNull():
            raise RuntimeError("Failed to construct QImage from the supplied RGBA frame.")

        pixmap = qt_gui.QPixmap.fromImage(qimage)
        if hasattr(pixmap, "isNull") and pixmap.isNull():
            raise RuntimeError("Failed to convert RGBA frame to a screen pixmap.")

        device_pixel_ratio = self._window_device_pixel_ratio(window)
        if device_pixel_ratio > 0.0 and hasattr(pixmap, "setDevicePixelRatio"):
            try:
                pixmap.setDevicePixelRatio(device_pixel_ratio)
            except Exception:
                pass

        image_label.setPixmap(pixmap)
        image_label.show()
        if hasattr(window, "showFullScreen"):
            window.showFullScreen()

        self._last_surface_kind = "raster"
        self._process_events(app)

    def tick(self) -> None:
        """Keep the Wayland event queue responsive between rerenders."""

        app = self._qt_app
        if app is None:
            return
        self._process_events(app)

    def close(self) -> None:
        """Release the Wayland window cleanly."""

        label = self._qt_image_label
        if label is not None:
            if hasattr(label, "clear"):
                try:
                    label.clear()
                except Exception:
                    pass
            if hasattr(label, "hide"):
                try:
                    label.hide()
                except Exception:
                    pass
            if hasattr(label, "deleteLater"):
                try:
                    label.deleteLater()
                except Exception:
                    pass

        window = self._qt_window
        if window is not None:
            if hasattr(window, "hide"):
                try:
                    window.hide()
                except Exception:
                    pass
            if hasattr(window, "close"):
                try:
                    window.close()
                except Exception:
                    pass
            if hasattr(window, "deleteLater"):
                try:
                    window.deleteLater()
                except Exception:
                    pass

        app = self._qt_app
        self._process_events(app)

        self._qt_app = None
        self._qt_window = None
        self._qt_image_label = None
        self._qt_core_module = None
        self._last_surface_kind = "none"

    def _ensure_window(
        self,
        *,
        qt_core: Any,
        qt_widgets: Any,
        size: tuple[int, int],
        socket_path: Path,
    ) -> tuple[Any, Any, Any]:
        requested_size = self._validate_size(size)
        requested_display = self._normalize_wayland_display(socket_path)

        app = qt_widgets.QApplication.instance()
        self._assert_gui_thread(qt_core=qt_core, app=app)

        if app is not None and not isinstance(app, qt_widgets.QApplication):
            raise RuntimeError(
                "A Qt application instance already exists, but it is not a QApplication. "
                "HdmiWaylandSurfaceHost requires QtWidgets/QApplication."
            )

        if app is None:
            self._configure_wayland_environment(requested_display)
            app = qt_widgets.QApplication([])
            if hasattr(app, "setQuitOnLastWindowClosed"):
                app.setQuitOnLastWindowClosed(False)

        self._qt_core_module = qt_core
        self._toolkit_name = self._detect_toolkit_name(qt_core=qt_core, qt_widgets=qt_widgets)
        self._validate_wayland_binding(app=app, requested_display=requested_display)

        window = self._qt_window
        image_label = self._qt_image_label
        if window is not None and image_label is not None:
            self._qt_app = app
            return (app, window, image_label)

        window = qt_widgets.QWidget()
        image_label = qt_widgets.QLabel(window)

        self._configure_window(
            app=app,
            qt_core=qt_core,
            qt_widgets=qt_widgets,
            window=window,
            requested_size=requested_size,
        )
        self._configure_image_label(qt_core=qt_core, qt_widgets=qt_widgets, image_label=image_label)

        if hasattr(qt_widgets, "QVBoxLayout"):
            layout = qt_widgets.QVBoxLayout(window)
            if hasattr(layout, "setContentsMargins"):
                layout.setContentsMargins(0, 0, 0, 0)
            if hasattr(layout, "setSpacing"):
                layout.setSpacing(0)
            if hasattr(layout, "addWidget"):
                layout.addWidget(image_label)
        else:
            logical_width, logical_height = self._logical_window_size(app=app, requested_size=requested_size)
            if hasattr(image_label, "setGeometry"):
                image_label.setGeometry(0, 0, logical_width, logical_height)

        if hasattr(window, "showFullScreen"):
            window.showFullScreen()
        if hasattr(window, "raise_"):
            try:
                window.raise_()
            except Exception:
                pass

        self._qt_app = app
        self._qt_window = window
        self._qt_image_label = image_label
        self._wayland_display = requested_display

        self._emit_status(
            display_wayland="ready",
            socket=requested_display,
            size=f"{requested_size[0]}x{requested_size[1]}",
            toolkit=self._toolkit_name,
            platform=self._platform_name(app),
            screen=self._screen_name_for_window(window),
            surface_host="qt_raster_native",
            focus="passive",
        )
        self._process_events(app)
        return (app, window, image_label)

    def _configure_window(
        self,
        *,
        app: Any,
        qt_core: Any,
        qt_widgets: Any,
        window: Any,
        requested_size: tuple[int, int],
    ) -> None:
        if hasattr(window, "setWindowTitle"):
            window.setWindowTitle("TWINR")
        if hasattr(window, "setObjectName"):
            window.setObjectName("twinr_hdmi_surface")

        flags = self._combine_flags(
            self._qt_enum(qt_core, "WindowType", "Window"),
            self._qt_enum(qt_core, "WindowType", "FramelessWindowHint"),
            self._qt_enum(qt_core, "WindowType", "WindowStaysOnTopHint"),
            self._qt_enum(qt_core, "WindowType", "WindowDoesNotAcceptFocus"),
        )
        if flags is not None and hasattr(window, "setWindowFlags"):
            window.setWindowFlags(flags)

        for attribute_name in (
            "WA_NativeWindow",
            "WA_DontCreateNativeAncestors",
            "WA_ShowWithoutActivating",
        ):
            self._set_widget_attribute(window, qt_core, attribute_name, True)

        no_focus = self._qt_enum(qt_core, "FocusPolicy", "NoFocus")
        if no_focus is not None and hasattr(window, "setFocusPolicy"):
            try:
                window.setFocusPolicy(no_focus)
            except Exception:
                pass

        if hasattr(window, "setStyleSheet"):
            window.setStyleSheet("background: rgb(10, 18, 32);")

        if hasattr(window, "setAccessibleName"):
            window.setAccessibleName("TWINR HDMI Surface")
        if hasattr(window, "setAccessibleDescription"):
            window.setAccessibleDescription(
                "Dedicated fullscreen Wayland surface for the Twinr HDMI presentation path."
            )
        if hasattr(window, "setAccessibleIdentifier"):
            window.setAccessibleIdentifier("twinr.hdmi_surface")

        target_screen = self._select_best_screen(app=app, requested_size=requested_size)
        if target_screen is not None and hasattr(window, "setScreen"):
            try:
                window.setScreen(target_screen)
            except Exception:
                pass

        logical_width, logical_height = self._logical_window_size(
            app=app,
            requested_size=requested_size,
            target_screen=target_screen,
        )
        if hasattr(window, "resize"):
            window.resize(logical_width, logical_height)

        if hasattr(window, "winId"):
            try:
                window.winId()
            except Exception:
                pass

        window_handle = window.windowHandle() if hasattr(window, "windowHandle") else None
        if target_screen is not None and window_handle is not None and hasattr(window_handle, "setScreen"):
            try:
                window_handle.setScreen(target_screen)
            except Exception:
                pass

    def _configure_image_label(self, *, qt_core: Any, qt_widgets: Any, image_label: Any) -> None:
        if hasattr(image_label, "setObjectName"):
            image_label.setObjectName("twinr_hdmi_surface_image")
        if hasattr(image_label, "setScaledContents"):
            image_label.setScaledContents(True)
        if hasattr(image_label, "setAlignment"):
            alignment = self._qt_enum(qt_core, "AlignmentFlag", "AlignCenter")
            if alignment is not None:
                image_label.setAlignment(alignment)
        if hasattr(image_label, "setFocusPolicy"):
            no_focus = self._qt_enum(qt_core, "FocusPolicy", "NoFocus")
            if no_focus is not None:
                try:
                    image_label.setFocusPolicy(no_focus)
                except Exception:
                    pass
        if hasattr(qt_widgets, "QSizePolicy") and hasattr(image_label, "setSizePolicy"):
            size_policy_class = qt_widgets.QSizePolicy
            expanding = getattr(size_policy_class, "Expanding", None)
            if expanding is None:
                policy_enum = getattr(size_policy_class, "Policy", None)
                if policy_enum is not None:
                    expanding = getattr(policy_enum, "Expanding", None)
            if expanding is not None:
                try:
                    image_label.setSizePolicy(expanding, expanding)
                except Exception:
                    pass
        if hasattr(image_label, "setAccessibleName"):
            image_label.setAccessibleName("TWINR HDMI Frame")
        if hasattr(image_label, "setAccessibleDescription"):
            image_label.setAccessibleDescription(
                "Raster image label used to present the current Twinr HDMI frame."
            )
        if hasattr(image_label, "setAccessibleIdentifier"):
            image_label.setAccessibleIdentifier("twinr.hdmi_surface.image")

    def _validate_wayland_binding(self, *, app: Any, requested_display: str) -> None:
        active_display = self._wayland_display or os.environ.get("WAYLAND_DISPLAY") or "wayland-0"
        if active_display != requested_display:
            raise RuntimeError(
                "The requested Wayland socket differs from the one already bound into this "
                "process. Qt cannot retarget the compositor after QApplication startup."
            )

        platform_name = self._platform_name(app)
        if platform_name and "wayland" not in platform_name.lower():
            raise RuntimeError(
                f"Qt started on platform '{platform_name}', but HdmiWaylandSurfaceHost requires a native Wayland platform."
            )

    def _configure_wayland_environment(self, requested_display: str) -> None:
        os.environ["QT_QPA_PLATFORM"] = "wayland"
        os.environ["WAYLAND_DISPLAY"] = requested_display

    def _assert_gui_thread(self, *, qt_core: Any, app: Any | None) -> None:
        if app is None:
            return

        qthread_class = getattr(qt_core, "QThread", None)
        app_thread_getter = getattr(app, "thread", None)
        if qthread_class is None or app_thread_getter is None:
            return

        try:
            current_qt_thread = qthread_class.currentThread()
            app_qt_thread = app_thread_getter()
        except Exception:
            return

        if app_qt_thread is not None and current_qt_thread is not None and app_qt_thread != current_qt_thread:
            raise RuntimeError("HdmiWaylandSurfaceHost must be called from the Qt GUI thread.")

    def _select_best_screen(self, *, app: Any, requested_size: tuple[int, int]) -> Any | None:
        screens_getter = getattr(app, "screens", None)
        if screens_getter is None:
            return None
        try:
            screens = list(screens_getter())
        except Exception:
            return None
        if not screens:
            return None
        if len(screens) == 1:
            return screens[0]

        requested_width, requested_height = requested_size
        best_screen: Any | None = None
        best_score: float | None = None
        for screen in screens:
            score = self._screen_match_score(
                screen=screen,
                requested_width=requested_width,
                requested_height=requested_height,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_screen = screen
        return best_screen

    def _screen_match_score(self, *, screen: Any, requested_width: int, requested_height: int) -> float:
        candidate_sizes: list[tuple[int, int]] = []
        geometry_getter = getattr(screen, "geometry", None)
        if geometry_getter is not None:
            try:
                geometry = geometry_getter()
            except Exception:
                geometry = None
            if geometry is not None and hasattr(geometry, "width") and hasattr(geometry, "height"):
                candidate_sizes.append((int(geometry.width()), int(geometry.height())))

        available_geometry_getter = getattr(screen, "availableGeometry", None)
        if available_geometry_getter is not None:
            try:
                available_geometry = available_geometry_getter()
            except Exception:
                available_geometry = None
            if available_geometry is not None and hasattr(available_geometry, "width") and hasattr(available_geometry, "height"):
                candidate_sizes.append((int(available_geometry.width()), int(available_geometry.height())))

        device_pixel_ratio = self._screen_device_pixel_ratio(screen)
        if device_pixel_ratio > 0.0:
            derived_sizes: list[tuple[int, int]] = []
            for width, height in candidate_sizes:
                derived_sizes.append((width, height))
                derived_sizes.append(
                    (
                        max(1, int(round(width * device_pixel_ratio))),
                        max(1, int(round(height * device_pixel_ratio))),
                    )
                )
            candidate_sizes = derived_sizes

        if not candidate_sizes:
            return float("inf")

        return min(
            abs(width - requested_width) + abs(height - requested_height)
            for width, height in candidate_sizes
        )

    def _logical_window_size(
        self,
        *,
        app: Any,
        requested_size: tuple[int, int],
        target_screen: Any | None = None,
    ) -> tuple[int, int]:
        width, height = requested_size
        screen = target_screen or self._select_best_screen(app=app, requested_size=requested_size)
        device_pixel_ratio = self._screen_device_pixel_ratio(screen)
        if device_pixel_ratio <= 0.0:
            device_pixel_ratio = 1.0
        return (
            max(1, int(round(width / device_pixel_ratio))),
            max(1, int(round(height / device_pixel_ratio))),
        )

    def _window_device_pixel_ratio(self, window: Any) -> float:
        if window is None:
            return 1.0
        window_handle = window.windowHandle() if hasattr(window, "windowHandle") else None
        ratio_getter = getattr(window_handle, "devicePixelRatio", None)
        if ratio_getter is None:
            return 1.0
        try:
            value = float(ratio_getter())
        except Exception:
            return 1.0
        return value if value > 0.0 else 1.0

    def _screen_device_pixel_ratio(self, screen: Any | None) -> float:
        if screen is None:
            return 1.0
        ratio_getter = getattr(screen, "devicePixelRatio", None)
        if ratio_getter is None:
            return 1.0
        try:
            value = float(ratio_getter())
        except Exception:
            return 1.0
        return value if value > 0.0 else 1.0

    def _screen_name_for_window(self, window: Any) -> str | None:
        if window is None or not hasattr(window, "windowHandle"):
            return None
        try:
            window_handle = window.windowHandle()
        except Exception:
            return None
        if window_handle is None or not hasattr(window_handle, "screen"):
            return None
        try:
            screen = window_handle.screen()
        except Exception:
            return None
        if screen is None:
            return None
        name_getter = getattr(screen, "name", None)
        if name_getter is None:
            return None
        try:
            name = name_getter()
        except Exception:
            return None
        return str(name) if name else None

    def _normalize_rgba_bytes(self, rgba_bytes: bytes | bytearray | memoryview, *, width: int, height: int) -> bytes | bytearray:
        expected_size = width * height * 4

        if isinstance(rgba_bytes, memoryview):
            actual_size = rgba_bytes.nbytes
            if actual_size < expected_size:
                raise ValueError(
                    f"RGBA frame is too short: expected at least {expected_size} bytes, got {actual_size}."
                )
            return rgba_bytes[:expected_size].tobytes()

        if isinstance(rgba_bytes, (bytes, bytearray)):
            actual_size = len(rgba_bytes)
            if actual_size < expected_size:
                raise ValueError(
                    f"RGBA frame is too short: expected at least {expected_size} bytes, got {actual_size}."
                )
            if actual_size == expected_size:
                return rgba_bytes
            return rgba_bytes[:expected_size]

        try:
            coerced = bytes(rgba_bytes)
        except Exception as exc:
            raise TypeError("rgba_bytes must be bytes-like.") from exc

        actual_size = len(coerced)
        if actual_size < expected_size:
            raise ValueError(
                f"RGBA frame is too short: expected at least {expected_size} bytes, got {actual_size}."
            )
        return coerced[:expected_size]

    def _normalize_wayland_display(self, socket_path: Path) -> str:
        display = str(socket_path)
        if not display:
            raise ValueError("socket_path must not be empty.")
        if any(ord(char) < 32 for char in display):
            raise ValueError("socket_path must not contain control characters.")
        return display

    def _validate_size(self, size: tuple[int, int]) -> tuple[int, int]:
        if len(size) != 2:
            raise ValueError("size must contain exactly two integers: (width, height).")
        width = int(size[0])
        height = int(size[1])
        if width <= 0 or height <= 0:
            raise ValueError(f"size must be positive, got {size!r}.")
        return (width, height)

    def _resolve_qimage_format(self, qt_gui: Any) -> Any:
        qimage_class = qt_gui.QImage
        if hasattr(qimage_class, "Format_RGBA8888"):
            return qimage_class.Format_RGBA8888
        qimage_format_enum = getattr(qimage_class, "Format", None)
        if qimage_format_enum is not None and hasattr(qimage_format_enum, "Format_RGBA8888"):
            return qimage_format_enum.Format_RGBA8888
        raise RuntimeError("This Qt binding does not expose QImage.Format_RGBA8888.")

    def _qt_enum(self, qt_core: Any, enum_group: str, member_name: str) -> Any | None:
        qt_namespace = getattr(qt_core, "Qt", None)
        if qt_namespace is None:
            return None
        if hasattr(qt_namespace, member_name):
            return getattr(qt_namespace, member_name)
        enum_container = getattr(qt_namespace, enum_group, None)
        if enum_container is None:
            return None
        return getattr(enum_container, member_name, None)

    def _combine_flags(self, *flags: Any) -> Any | None:
        combined: Any | None = None
        for flag in flags:
            if flag is None:
                continue
            combined = flag if combined is None else (combined | flag)
        return combined

    def _set_widget_attribute(self, widget: Any, qt_core: Any, attribute_name: str, enabled: bool) -> None:
        attribute = self._qt_enum(qt_core, "WidgetAttribute", attribute_name)
        if attribute is None or not hasattr(widget, "setAttribute"):
            return
        try:
            widget.setAttribute(attribute, enabled)
        except Exception:
            return

    def _platform_name(self, app: Any) -> str:
        platform_name_getter = getattr(app, "platformName", None)
        if platform_name_getter is None:
            return "unknown"
        try:
            platform_name = platform_name_getter()
        except Exception:
            return "unknown"
        return str(platform_name) if platform_name else "unknown"

    def _detect_toolkit_name(self, *, qt_core: Any, qt_widgets: Any) -> str:
        module_names = " ".join(
            (
                str(getattr(qt_core, "__name__", "")),
                str(getattr(qt_widgets, "__name__", "")),
            )
        ).lower()
        for toolkit_name in ("pyside6", "pyqt6", "pyside2", "pyqt5"):
            if toolkit_name in module_names:
                return toolkit_name
        return "qt_python"

    def _emit_status(self, **fields: Any) -> None:
        parts: list[str] = []
        for key, value in fields.items():
            if value is None:
                continue
            sanitized_key = str(key).strip().replace(" ", "_")
            sanitized_value = self._sanitize_emit_value(value)
            parts.append(f"{sanitized_key}={sanitized_value}")
        self._safe_emit(" ".join(parts))

    def _sanitize_emit_value(self, value: Any) -> str:
        return quote(str(value), safe="/:._-")

    def _process_events(self, app: Any | None) -> None:
        if app is None or not hasattr(app, "processEvents"):
            return
        try:
            app.processEvents()
        except Exception:
            return

    def _safe_emit(self, line: str) -> None:
        if self.emit is None:
            return
        try:
            self.emit(line)
        except Exception:
            return
