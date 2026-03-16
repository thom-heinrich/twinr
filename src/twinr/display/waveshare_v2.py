"""Render Twinr status cards on the Waveshare 4.2 V2 e-paper panel.

This adapter validates Twinr's configured GPIO and vendor-driver layout before
loading the vendor module. It owns image preparation, status-card rendering,
and best-effort hardware cleanup for the panel.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone  # AUDIT-FIX(#7): Use aware local timestamps instead of naive localtime.
import gc
from importlib import import_module
import importlib.util  # AUDIT-FIX(#1): Load vendor modules from exact files instead of mutating sys.path.
from pathlib import Path
from threading import RLock  # AUDIT-FIX(#3): Serialise shared mutable state and global import mutations.
import logging  # AUDIT-FIX(#4): Emit diagnosable recovery logs for hardware/display faults.
import sys
import time

from twinr.agent.base_agent.config import TwinrConfig


_LOGGER = logging.getLogger(__name__)
_IMPORT_LOCK = RLock()
_SUPPORTED_ROTATIONS = {0, 90, 180, 270}


@dataclass(slots=True)
class WaveshareEPD4In2V2:
    """Render Twinr status frames on a Waveshare 4.2 V2 panel.

    The adapter keeps hardware interaction serialized per instance and
    validates vendor-driver expectations before importing the Waveshare
    modules.

    Attributes:
        project_root: Twinr project root used to resolve vendor paths safely.
        vendor_dir: Directory that contains the ``waveshare_epd`` vendor
            package.
        driver: Configured driver identifier. Only ``waveshare_4in2_v2`` is
            supported.
        spi_bus: SPI bus index expected by the installed vendor driver.
        spi_device: SPI device index expected by the installed vendor driver.
        cs_gpio: Chip-select GPIO number.
        dc_gpio: Data/command GPIO number.
        reset_gpio: Reset GPIO number.
        busy_gpio: Busy GPIO number.
        width: Logical display width before rotation.
        height: Logical display height before rotation.
        rotation_degrees: Rotation applied before panel upload.
        full_refresh_interval: Number of partial renders between full
            refreshes.
    """

    project_root: Path
    vendor_dir: Path
    driver: str = "waveshare_4in2_v2"
    spi_bus: int = 0
    spi_device: int = 0
    cs_gpio: int = 8
    dc_gpio: int = 25
    reset_gpio: int = 17
    busy_gpio: int = 24
    width: int = 400
    height: int = 300
    rotation_degrees: int = 270
    full_refresh_interval: int = 0
    _driver_module: object | None = field(default=None, init=False, repr=False)
    _epdconfig_module: object | None = field(default=None, init=False, repr=False)  # AUDIT-FIX(#5): Keep vendor transport module for deterministic cleanup.
    _epd: object | None = field(default=None, init=False, repr=False)
    _render_count: int = field(default=0, init=False, repr=False)
    _font_cache: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _lock: object = field(default_factory=RLock, init=False, repr=False)  # AUDIT-FIX(#3): Protect shared driver/font state.

    # AUDIT-FIX(#1): Canonicalise and validate file-system inputs early so the driver cannot import code outside the Twinr project tree.
    # AUDIT-FIX(#6): Fail fast on invalid config values instead of crashing later inside PIL or vendor code.
    def __post_init__(self) -> None:
        self.project_root = self.project_root.expanduser().resolve(strict=False)
        self.vendor_dir = self._resolve_vendor_dir(self.vendor_dir)
        self.rotation_degrees = self.rotation_degrees % 360

        if self.width <= 0 or self.height <= 0:
            raise RuntimeError("Display width and height must be positive integers.")
        if self.full_refresh_interval < 0:
            raise RuntimeError("Display full_refresh_interval must be >= 0.")
        if self.rotation_degrees not in _SUPPORTED_ROTATIONS:
            raise RuntimeError(
                "Display rotation must be one of 0, 90, 180, or 270 degrees."
            )
        if self.spi_bus < 0 or self.spi_device < 0:
            raise RuntimeError("SPI bus and device must be >= 0.")

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "WaveshareEPD4In2V2":
        """Build a display adapter from Twinr configuration.

        Args:
            config: Runtime configuration with display GPIO, SPI, and sizing
                values.

        Returns:
            A validated ``WaveshareEPD4In2V2`` instance.

        Raises:
            RuntimeError: If the configured display GPIOs conflict with other
                runtime GPIO assignments.
        """
        conflicts = config.display_gpio_conflicts()
        if conflicts:
            raise RuntimeError(
                "Display GPIO configuration is invalid: " + "; ".join(conflicts)
            )
        return cls(
            project_root=Path(config.project_root),
            # AUDIT-FIX(#1): Keep the raw configured vendor path and let __post_init__ resolve it safely against project_root.
            vendor_dir=Path(config.display_vendor_dir),
            driver=config.display_driver,
            spi_bus=config.display_spi_bus,
            spi_device=config.display_spi_device,
            cs_gpio=config.display_cs_gpio,
            dc_gpio=config.display_dc_gpio,
            reset_gpio=config.display_reset_gpio,
            busy_gpio=config.display_busy_gpio,
            width=config.display_width,
            height=config.display_height,
            rotation_degrees=config.display_rotation_degrees,
            full_refresh_interval=config.display_full_refresh_interval,
        )

    @property
    def vendor_package_dir(self) -> Path:
        """Return the directory that should contain the vendor package."""
        return self.vendor_dir / "waveshare_epd"

    @property
    def canvas_size(self) -> tuple[int, int]:
        """Return the logical canvas size before rotation."""
        return (self.width, self.height)

    @property
    def allowed_image_sizes(self) -> tuple[tuple[int, int], ...]:
        """Return accepted logical and rotated image sizes."""
        sizes = {self.canvas_size}
        if self.rotation_degrees in (90, 270):
            sizes.add((self.height, self.width))
        return tuple(sorted(sizes))

    def show_test_pattern(self) -> None:
        """Render and display the built-in hardware smoke-test card."""
        image = self.render_test_image()
        self.show_image(image, clear_first=True)

    def show_status(
        self,
        status: str,
        *,
        headline: str | None = None,
        details: tuple[str, ...] = (),
        animation_frame: int = 0,
    ) -> None:
        """Render and display one runtime status frame.

        Args:
            status: Canonical runtime status key.
            headline: Optional headline override shown in the top bar.
            details: Up to four short footer/detail lines.
            animation_frame: Precomputed animation frame index.
        """
        image = self.render_status_image(
            status=status,
            headline=headline,
            details=details,
            animation_frame=animation_frame,
        )
        self.show_image(image, clear_first=False)

    # AUDIT-FIX(#3): Guard mutable driver state with a per-instance lock so concurrent callers cannot double-initialise the panel or corrupt render counters.
    # AUDIT-FIX(#4): Reset and retry once after display failures to recover from transient SPI/EPD faults.
    def show_image(self, image: object, *, clear_first: bool) -> None:
        """Upload a prepared image to the panel with one recovery retry.

        Args:
            image: Pillow image or image-like object accepted by the adapter.
            clear_first: Whether to clear the panel before rendering.

        Raises:
            RuntimeError: If the panel cannot render the image after one
                recovery attempt.
        """
        with self._lock:
            prepared_image = self._prepare_image(image)
            self._validate_prepared_image(prepared_image)
            last_error: Exception | None = None
            started_at = time.monotonic()

            for attempt in range(2):
                try:
                    epd = self._get_epd()
                    self._display_prepared_image(epd, prepared_image, clear_first=clear_first)
                    if attempt > 0:
                        _LOGGER.info(
                            "E-paper render recovered after %.3fs.",
                            time.monotonic() - started_at,
                        )
                    return
                except Exception as exc:  # pragma: no cover - exercised via hardware fault paths.
                    last_error = exc
                    _LOGGER.warning(
                        "E-paper render attempt %s failed after %.3fs; resetting driver state.",
                        attempt + 1,
                        time.monotonic() - started_at,
                        exc_info=exc,
                    )
                    self._reset_driver_state()

                clear_first = True

            raise RuntimeError(
                "E-paper display update failed after one recovery attempt."
            ) from last_error

    # AUDIT-FIX(#3): Keep shutdown atomic relative to display calls.
    # AUDIT-FIX(#5): Perform best-effort hardware cleanup instead of only suppressing sleep() and leaking SPI/GPIO resources.
    def close(self) -> None:
        """Release vendor-driver resources and reset local adapter state."""
        with self._lock:
            self._shutdown_hardware()
            self._driver_module = None
            self._epdconfig_module = None
            self._epd = None
            self._render_count = 0

    # AUDIT-FIX(#3): Protect font-cache mutation and render helpers from concurrent access.
    # AUDIT-FIX(#7): Use an aware local datetime so the rendered test timestamp is unambiguous across DST/timezone changes.
    def render_test_image(self) -> object:
        """Build the built-in black-and-white hardware smoke-test card."""
        with self._lock:
            image, draw = self._new_canvas()
            canvas_width, canvas_height = image.size
            title_font = self._font(28, bold=True)
            body_font = self._font(20, bold=False)
            now_text = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            draw.rectangle((0, 0, canvas_width - 1, canvas_height - 1), outline=0, width=5)
            draw.rectangle((0, 0, canvas_width - 1, 58), fill=0)
            draw.text((16, 12), "TWINR E-PAPER V2", fill=255, font=title_font)
            draw.text((18, 78), now_text, fill=0, font=body_font)
            draw.text((18, 116), "BLACK / WHITE TEST", fill=0, font=body_font)
            draw.text((18, 152), "Display path is ready.", fill=0, font=body_font)
            draw.rectangle((20, 210, 120, 290), fill=0)
            draw.rectangle((140, 210, 240, 290), outline=0, width=3)
            return image

    # AUDIT-FIX(#3): Protect font-cache mutation and render helpers from concurrent access.
    # AUDIT-FIX(#2): Normalise status/headline/detail payloads so error rendering does not crash on non-string inputs.
    def render_status_image(
        self,
        *,
        status: str,
        headline: str | None,
        details: tuple[str, ...],
        animation_frame: int = 0,
    ) -> object:
        """Build a status card image without sending it to the panel.

        Args:
            status: Canonical runtime status key.
            headline: Optional headline override shown in the top bar.
            details: Footer/detail lines for health and time labels.
            animation_frame: Precomputed animation frame index.

        Returns:
            A Pillow image object sized for the configured panel.
        """
        with self._lock:
            safe_status = self._normalise_text(status, fallback="status")
            safe_headline = self._normalise_text(headline, fallback=safe_status)
            safe_details = self._normalise_details(details)
            safe_animation_frame = self._normalise_animation_frame(animation_frame)

            image, draw = self._new_canvas()
            canvas_width, canvas_height = image.size
            brand_font = self._font(28, bold=True)
            status_font = self._font(24, bold=True)
            draw.rectangle((0, 0, canvas_width - 1, canvas_height - 1), fill=255)
            draw.rectangle((0, 0, canvas_width - 1, 58), fill=0)
            draw.text((18, 10), "TWINR", fill=255, font=brand_font)
            status_label = " ".join(safe_headline.split())
            label_text = self._truncate_text(
                draw,
                status_label,
                max_width=max(canvas_width - 170, 110),
                font=status_font,
            )
            label_width = self._text_width(draw, label_text, font=status_font)
            draw.text((canvas_width - label_width - 18, 12), label_text, fill=255, font=status_font)

            self._draw_face(
                draw,
                status=safe_status.lower(),
                animation_frame=safe_animation_frame,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
            )
            self._draw_details_footer(
                draw,
                details=safe_details,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
            )
            return image

    # AUDIT-FIX(#1): Replace sys.path mutation with exact-file imports after validating the vendor package layout and origin paths.
    # AUDIT-FIX(#3): Serialize module loading globally because sys.modules is process-global state.
    # AUDIT-FIX(#6): Fail fast on non-default SPI settings when the vendor package cannot prove it is configured for them.
    def _load_driver_module(self):
        if self._driver_module is not None:
            return self._driver_module

        if self.driver != "waveshare_4in2_v2":
            raise RuntimeError(f"Unsupported display driver: {self.driver}")

        package_dir = self._validate_vendor_layout()
        package_init = package_dir / "__init__.py"
        epdconfig_path = package_dir / "epdconfig.py"
        driver_path = package_dir / "epd4in2_V2.py"

        with _IMPORT_LOCK:
            epdconfig_module = None
            self._load_exact_vendor_module(
                module_name="waveshare_epd",
                module_path=package_init,
                is_package=True,
            )
            epdconfig_module = self._load_exact_vendor_module(
                module_name="waveshare_epd.epdconfig",
                module_path=epdconfig_path,
            )
            try:
                self._load_exact_vendor_module(
                    module_name="waveshare_epd.epd4in2_V2",
                    module_path=driver_path,
                )
                epdconfig = import_module("waveshare_epd.epdconfig")
                module = import_module("waveshare_epd.epd4in2_V2")
            except Exception as exc:
                self._cleanup_failed_vendor_import(epdconfig_module)
                raise RuntimeError(
                    "Display vendor files are incomplete or failed to import. "
                    "Run `hardware/display/setup_display.sh` again."
                ) from exc

        self._validate_vendor_config(epdconfig)
        self._validate_driver_module_origin(module, driver_path)
        self._epdconfig_module = epdconfig
        self._driver_module = module
        return module

    def _cleanup_failed_vendor_import(self, epdconfig_module: object | None) -> None:
        cleanup = getattr(epdconfig_module, "module_exit", None)
        if callable(cleanup):
            with suppress(Exception):
                cleanup(cleanup=True)
        self._drop_cached_vendor_modules()

    def _validate_vendor_config(self, epdconfig: object) -> None:
        expected = {
            "RST_PIN": self.reset_gpio,
            "DC_PIN": self.dc_gpio,
            "CS_PIN": self.cs_gpio,
            "BUSY_PIN": self.busy_gpio,
        }
        mismatches = []
        for name, expected_value in expected.items():
            actual_value = getattr(epdconfig, name, None)
            if actual_value != expected_value:
                mismatches.append(f"{name}={actual_value} (expected {expected_value})")

        # AUDIT-FIX(#6): Detect non-default SPI settings that would otherwise be accepted but silently ignored by the vendor package.
        optional_spi = {
            "SPI_BUS": self.spi_bus,
            "SPI_DEVICE": self.spi_device,
        }
        unverifiable_spi = []
        for name, expected_value in optional_spi.items():
            if hasattr(epdconfig, name):
                actual_value = getattr(epdconfig, name, None)
                if actual_value != expected_value:
                    mismatches.append(f"{name}={actual_value} (expected {expected_value})")
            elif expected_value != 0:
                unverifiable_spi.append(f"{name}={expected_value}")

        if mismatches:
            mismatch_text = ", ".join(mismatches)
            raise RuntimeError(
                "Installed display driver pins do not match Twinr config: "
                f"{mismatch_text}. Run `hardware/display/setup_display.sh` again."
            )

        if unverifiable_spi:
            unverifiable_text = ", ".join(unverifiable_spi)
            raise RuntimeError(
                "Configured SPI bus/device cannot be verified against the installed vendor driver: "
                f"{unverifiable_text}. Use SPI 0:0 or patch the vendor driver during setup."
            )

    # AUDIT-FIX(#8): Convert missing Pillow dependency into a clear runtime error instead of a raw ImportError from deep inside a render path.
    def _new_canvas(self) -> tuple[object, object]:
        try:
            from PIL import Image, ImageDraw
        except Exception as exc:
            raise RuntimeError(
                "Pillow is required for Twinr e-paper rendering."
            ) from exc

        image = Image.new("1", self.canvas_size, 255)
        return image, ImageDraw.Draw(image)

    def _draw_face(
        self,
        draw: object,
        *,
        status: str,
        animation_frame: int,
        canvas_width: int,
        canvas_height: int,
    ) -> None:
        face_center_x = canvas_width // 2
        face_center_y = (canvas_height // 2) + 16
        jitter_x, jitter_y = self._face_offset(status, animation_frame)

        left_eye = (face_center_x - 72 + jitter_x, face_center_y - 24 + jitter_y)
        right_eye = (face_center_x + 72 + jitter_x, face_center_y - 24 + jitter_y)
        self._draw_eye(draw, left_eye, status=status, side="left", animation_frame=animation_frame)
        self._draw_eye(draw, right_eye, status=status, side="right", animation_frame=animation_frame)
        self._draw_mouth(
            draw,
            center_x=face_center_x + jitter_x,
            center_y=face_center_y + 56 + jitter_y,
            status=status,
            animation_frame=animation_frame,
        )

    # AUDIT-FIX(#2): Only render sanitised detail lines so exception objects or control characters cannot break the status footer.
    def _draw_details_footer(
        self,
        draw: object,
        *,
        details: tuple[str, ...],
        canvas_width: int,
        canvas_height: int,
    ) -> None:
        rows = self._footer_rows(details)
        if not rows:
            return
        if len(rows) == 1 and len(rows[0]) == 1:
            self._draw_single_footer_line(
                draw,
                line=rows[0][0],
                canvas_width=canvas_width,
                canvas_height=canvas_height,
            )
            return
        self._draw_footer_grid(
            draw,
            rows=rows,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )

    def _draw_single_footer_line(
        self,
        draw: object,
        *,
        line: str,
        canvas_width: int,
        canvas_height: int,
    ) -> None:
        footer_font = self._font(18, bold=False)
        divider_y = canvas_height - 54
        draw.line((28, divider_y, canvas_width - 28, divider_y), fill=0, width=2)
        text_y = divider_y + 8
        left_text, right_text = self._split_footer_parts(line)
        right_width = self._text_width(draw, right_text, font=footer_font)
        right_margin = 24
        right_x = max(canvas_width - right_width - right_margin, 24)
        single_line_left_width = max(right_x - 36, 120)
        left_lines = self._wrap_footer_left(
            draw,
            left_text,
            font=footer_font,
            full_width=canvas_width - 48,
            final_width=single_line_left_width,
        )
        line_height = self._text_height(draw, font=footer_font)
        for index, left_line in enumerate(left_lines):
            line_y = text_y + (index * (line_height + 2))
            max_width = single_line_left_width if index == (len(left_lines) - 1) else (canvas_width - 48)
            trimmed = self._truncate_text(draw, left_line, max_width=max_width, font=footer_font)
            draw.text((24, line_y), trimmed, fill=0, font=footer_font)
            if right_text and index == (len(left_lines) - 1):
                draw.text((right_x, line_y), right_text, fill=0, font=footer_font)

    def _draw_footer_grid(
        self,
        draw: object,
        *,
        rows: tuple[tuple[str, ...], ...],
        canvas_width: int,
        canvas_height: int,
    ) -> None:
        footer_font = self._font(16, bold=False)
        line_height = self._text_height(draw, font=footer_font)
        row_gap = 4
        padding_top = 8
        padding_bottom = 8
        footer_height = padding_top + (len(rows) * line_height) + (max(len(rows) - 1, 0) * row_gap) + padding_bottom
        divider_y = canvas_height - footer_height
        draw.line((28, divider_y, canvas_width - 28, divider_y), fill=0, width=2)
        text_y = divider_y + padding_top
        left_x = 24
        column_gap = 16
        content_width = canvas_width - (left_x * 2)
        column_width = max((content_width - column_gap) // 2, 96)
        right_x = left_x + column_width + column_gap

        for row_index, row in enumerate(rows):
            line_y = text_y + (row_index * (line_height + row_gap))
            left_text = self._truncate_text(draw, row[0], max_width=content_width if len(row) == 1 else column_width, font=footer_font)
            draw.text((left_x, line_y), left_text, fill=0, font=footer_font)
            if len(row) > 1:
                right_text = self._truncate_text(draw, row[1], max_width=column_width, font=footer_font)
                draw.text((right_x, line_y), right_text, fill=0, font=footer_font)

    def _footer_rows(self, details: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
        lines = self._normalise_details(details)
        if not lines:
            return ()
        if len(lines) == 1:
            return ((lines[0],),)
        capped = lines[:4]
        return tuple(tuple(capped[index : index + 2]) for index in range(0, len(capped), 2))

    def _draw_eye(
        self,
        draw: object,
        origin: tuple[int, int],
        *,
        status: str,
        side: str,
        animation_frame: int,
    ) -> None:
        center_x, center_y = origin
        eye = self._eye_state(status, animation_frame, side)

        brow_y = center_y - 52 + int(eye["brow_raise"])
        if side == "left":
            draw.line(
                (
                    center_x - 24,
                    brow_y + int(eye["brow_slant"]),
                    center_x + 24,
                    brow_y - int(eye["brow_slant"]),
                ),
                fill=0,
                width=4,
            )
        else:
            draw.line(
                (
                    center_x - 24,
                    brow_y - int(eye["brow_slant"]),
                    center_x + 24,
                    brow_y + int(eye["brow_slant"]),
                ),
                fill=0,
                width=4,
            )

        if bool(eye["blink"]):
            draw.arc(
                (
                    center_x - 26,
                    center_y - 8,
                    center_x + 26,
                    center_y + 10,
                ),
                start=200,
                end=340,
                fill=0,
                width=5,
            )
            return

        width = int(eye["width"])
        height = int(eye["height"])
        offset_x = int(eye["eye_shift_x"])
        offset_y = int(eye["eye_shift_y"])
        box = (
            center_x - (width // 2) + offset_x,
            center_y - (height // 2) + offset_y,
            center_x + (width // 2) + offset_x,
            center_y + (height // 2) + offset_y,
        )
        draw.ellipse(box, fill=0)
        self._draw_eye_highlights(draw, box, eye)

        if bool(eye["lid_arc"]):
            draw.arc(
                (
                    box[0] + 4,
                    box[1] - 10,
                    box[2] - 4,
                    box[1] + 18,
                ),
                start=180,
                end=360,
                fill=0,
                width=3,
            )

    def _draw_mouth(
        self,
        draw: object,
        *,
        center_x: int,
        center_y: int,
        status: str,
        animation_frame: int,
    ) -> None:
        if status == "waiting":
            sway = (-1, 0, 1, 0, -1, 0)[animation_frame % 6]
            draw.arc(
                (center_x - 24, center_y - 10 + sway, center_x + 24, center_y + 12 + sway),
                start=18,
                end=162,
                fill=0,
                width=4,
            )
            return
        if status == "listening":
            openness = (14, 18, 14, 12)[animation_frame % 4]
            draw.ellipse((center_x - 10, center_y - 8, center_x + 10, center_y + openness), outline=0, width=4)
            return
        if status == "processing":
            offset_y = (-1, 0, 1, 0)[animation_frame % 4]
            left_segment = (
                center_x - 22,
                center_y + 4 + offset_y,
                center_x - 4,
                center_y + 2 + offset_y,
            )
            right_segment = (
                center_x + 4,
                center_y + 2 + offset_y,
                center_x + 22,
                center_y + 4 + offset_y,
            )
            draw.line(left_segment, fill=0, width=4)
            draw.line(right_segment, fill=0, width=4)
            return
        if status == "answering":
            openness = (8, 11, 7, 10)[animation_frame % 4]
            draw.rounded_rectangle(
                (center_x - 22, center_y - 2, center_x + 22, center_y + openness),
                radius=8,
                outline=0,
                width=4,
            )
            return
        if status == "printing":
            lift = (0, -1, 0, 1)[animation_frame % 4]
            draw.arc(
                (center_x - 28, center_y - 6 + lift, center_x + 28, center_y + 16 + lift),
                start=12,
                end=168,
                fill=0,
                width=4,
            )
            return
        if status == "error":
            draw.arc(
                (center_x - 22, center_y + 6, center_x + 22, center_y + 18),
                start=200,
                end=340,
                fill=0,
                width=4,
            )
            return
        draw.arc((center_x - 20, center_y - 8, center_x + 20, center_y + 8), start=20, end=160, fill=0, width=4)

    def _prepare_image(self, image: object):
        if hasattr(image, "rotate") and hasattr(image, "size"):
            width, height = image.size
            if (width, height) == self.canvas_size and self.rotation_degrees != 0:
                return image.rotate(self.rotation_degrees, expand=True)
        return image

    # AUDIT-FIX(#4): Validate the prepared image before touching hardware so bad caller input does not masquerade as an SPI/display failure.
    def _validate_prepared_image(self, image: object) -> None:
        size = getattr(image, "size", None)
        if not isinstance(size, tuple) or len(size) != 2:
            raise RuntimeError(
                "Display image must expose a two-value `.size` attribute."
            )

        width = int(size[0])
        height = int(size[1])
        if (width, height) not in self.allowed_image_sizes:
            raise RuntimeError(
                "Display image size "
                f"{(width, height)} does not match expected sizes {self.allowed_image_sizes}."
            )

    # AUDIT-FIX(#4): Keep the success/failure boundary narrow so render counters are only mutated after a successful hardware update.
    def _display_prepared_image(self, epd: object, prepared_image: object, *, clear_first: bool) -> None:
        if clear_first or self._render_count == 0:
            self._init_full(epd)
            if clear_first and hasattr(epd, "Clear"):
                epd.Clear()
            prepared = epd.getbuffer(prepared_image)
            epd.display(prepared)
            self._render_count = 1
            return

        if self.full_refresh_interval > 0 and self._render_count % self.full_refresh_interval == 0:
            self._init_full(epd)
            prepared = epd.getbuffer(prepared_image)
            epd.display(prepared)
            self._render_count += 1
            return

        if hasattr(epd, "display_Partial"):
            prepared = epd.getbuffer(prepared_image)
            epd.display_Partial(prepared)
        elif hasattr(epd, "display_Fast") and hasattr(epd, "init_fast"):
            self._init_fast(epd)
            prepared = epd.getbuffer(prepared_image)
            epd.display_Fast(prepared)
        else:
            self._init_full(epd)
            prepared = epd.getbuffer(prepared_image)
            epd.display(prepared)
        self._render_count += 1

    # AUDIT-FIX(#3): Only instantiate the vendor driver once per wrapper instance and validate its shape before use.
    def _get_epd(self):
        if self._epd is None:
            module = self._load_driver_module()
            if not hasattr(module, "EPD"):
                raise RuntimeError("Display driver module does not expose EPD().")
            self._epd = module.EPD()
        return self._epd

    def _init_full(self, epd: object) -> None:
        if not hasattr(epd, "init"):
            raise RuntimeError("Display driver instance does not expose init().")
        epd.init()

    def _init_fast(self, epd: object) -> None:
        if not hasattr(epd, "init_fast"):
            raise RuntimeError("Display driver instance does not expose init_fast().")
        speed_mode = getattr(epd, "Seconds_1_5S", 0)
        epd.init_fast(speed_mode)

    def _face_offset(self, status: str, animation_frame: int) -> tuple[int, int]:
        if status == "waiting":
            return ((0, 0), (-2, 0), (2, 0), (0, -1), (0, 1), (0, 0))[animation_frame % 6]
        if status == "listening":
            return ((0, 0), (0, -1), (0, 0), (0, 1))[animation_frame % 4]
        if status == "processing":
            return ((0, 0), (-1, 0), (1, 0), (0, 0))[animation_frame % 4]
        if status == "answering":
            return ((0, 0), (0, -1), (0, 0), (0, 1))[animation_frame % 4]
        if status == "printing":
            return ((0, 0), (1, 0), (0, 0), (-1, 0))[animation_frame % 4]
        if status == "error":
            return ((0, 1), (0, 0), (0, 1), (0, 0))[animation_frame % 4]
        return (0, 0)

    def _eye_state(self, status: str, animation_frame: int, side: str) -> dict[str, int | bool]:
        state: dict[str, int | bool] = {
            "width": 56,
            "height": 74,
            "eye_shift_x": 0,
            "eye_shift_y": 0,
            "highlight_dx": -10,
            "highlight_dy": -18,
            "brow_raise": 0,
            "brow_slant": 4,
            "blink": False,
            "lid_arc": False,
        }

        if status == "waiting":
            looks = (-10, -5, 4, 8, 0, -2)
            state["highlight_dx"] = looks[animation_frame % 6]
            state["eye_shift_y"] = (-1, 0, 0, 0, 1, 0)[animation_frame % 6]
            state["blink"] = animation_frame == 4
            return state

        if status == "listening":
            state["height"] = (78, 82, 78, 74)[animation_frame % 4]
            state["highlight_dx"] = (-8, -6, -8, -10)[animation_frame % 4]
            state["brow_raise"] = -8
            state["brow_slant"] = 2
            state["lid_arc"] = True
            state["blink"] = animation_frame == 3
            return state

        if status == "processing":
            gaze = (-12, -4, 4, 12)[animation_frame % 4]
            state["highlight_dx"] = gaze if side == "left" else gaze - 2
            state["height"] = 68
            state["brow_raise"] = -1
            state["brow_slant"] = 4
            state["lid_arc"] = True
            return state

        if status == "answering":
            state["height"] = (70, 74, 70, 72)[animation_frame % 4]
            state["highlight_dx"] = (-8, -6, -8, -7)[animation_frame % 4]
            state["brow_raise"] = -2
            state["brow_slant"] = 2
            return state

        if status == "printing":
            state["height"] = (70, 66, 70, 62)[animation_frame % 4]
            state["highlight_dx"] = (-9, -8, -7, -6)[animation_frame % 4]
            state["brow_raise"] = -4
            state["brow_slant"] = 2
            state["blink"] = animation_frame == 3
            return state

        if status == "error":
            state["height"] = (60, 56, 60, 58)[animation_frame % 4]
            state["highlight_dx"] = (-12, -11, -10, -11)[animation_frame % 4]
            state["highlight_dy"] = -14
            state["brow_raise"] = 2
            state["brow_slant"] = 8
            state["eye_shift_y"] = 2
            state["blink"] = animation_frame == 2
            return state

        return state

    def _draw_eye_highlights(self, draw: object, box: tuple[int, int, int, int], eye: dict[str, int | bool]) -> None:
        center_x = (box[0] + box[2]) // 2
        center_y = (box[1] + box[3]) // 2
        main_x = center_x + int(eye["highlight_dx"])
        main_y = center_y + int(eye["highlight_dy"])
        draw.ellipse((main_x - 8, main_y - 8, main_x + 8, main_y + 8), fill=255)
        draw.ellipse((main_x + 10, main_y + 8, main_x + 16, main_y + 14), fill=255)

    def _split_footer_parts(self, text: str) -> tuple[str, str]:
        compact = text.strip()
        if compact.endswith(")") and " (" in compact:
            prefix, suffix = compact.rsplit(" (", 1)
            return prefix.strip(), f"({suffix}"
        return compact, ""

    def _wrap_footer_left(
        self,
        draw: object,
        text: str,
        *,
        font: object | None,
        full_width: int,
        final_width: int,
    ) -> tuple[str, ...]:
        compact = text.strip()
        if self._text_width(draw, compact, font=font) <= final_width:
            return (compact,)
        parts = [part.strip() for part in compact.split("|") if part.strip()]
        if len(parts) < 2:
            first = self._truncate_text(draw, compact, max_width=full_width, font=font)
            return (first,)
        first_line_parts: list[str] = []
        for part in parts:
            candidate_parts = first_line_parts + [part]
            candidate_text = " | ".join(candidate_parts)
            if first_line_parts and self._text_width(draw, candidate_text, font=font) > full_width:
                break
            first_line_parts = candidate_parts
        if not first_line_parts:
            first_line_parts = [parts[0]]
        remaining = parts[len(first_line_parts):]
        if not remaining:
            return (" | ".join(first_line_parts),)
        second_line = " | ".join(remaining)
        return (
            " | ".join(first_line_parts),
            self._truncate_text(draw, second_line, max_width=final_width, font=font),
        )

    # AUDIT-FIX(#3): Serialise cache access to avoid duplicate font loads and shared-cache races under concurrent render calls.
    # AUDIT-FIX(#8): Convert missing Pillow dependency into a clear runtime error while retaining graceful font fallback when files are missing.
    def _font(self, size: int, *, bold: bool) -> object:
        cache_key = f"{'bold' if bold else 'regular'}:{max(8, size)}"
        with self._lock:
            cached = self._font_cache.get(cache_key)
            if cached is not None:
                return cached

            try:
                from PIL import ImageFont
            except Exception as exc:
                raise RuntimeError(
                    "Pillow is required for Twinr e-paper font rendering."
                ) from exc

            candidates = (
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            ) if bold else (
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            )
            font = None
            for candidate in candidates:
                if not Path(candidate).exists():
                    continue
                with suppress(Exception):
                    font = ImageFont.truetype(candidate, size=max(8, size))
                    break
            if font is None:
                font = ImageFont.load_default()
            self._font_cache[cache_key] = font
            return font

    # AUDIT-FIX(#9): Keep layout fallbacks narrow; earlier image/draw validation now catches gross misuse before these helpers need to suppress anything.
    def _text_width(self, draw: object, text: str, *, font: object | None = None) -> int:
        if not text:
            return 0
        width = len(text) * 6
        with suppress(Exception):
            text_box = draw.textbbox((0, 0), text, font=font)
            width = int(text_box[2] - text_box[0])
        return width

    # AUDIT-FIX(#9): Keep layout fallbacks narrow; earlier image/draw validation now catches gross misuse before these helpers need to suppress anything.
    def _text_height(self, draw: object, *, font: object | None = None) -> int:
        height = 12
        with suppress(Exception):
            text_box = draw.textbbox((0, 0), "Hg", font=font)
            height = int(text_box[3] - text_box[1])
        return height

    def _truncate_text(self, draw: object, text: str, *, max_width: int, font: object | None = None) -> str:
        compact = text.strip()
        if self._text_width(draw, compact, font=font) <= max_width:
            return compact
        ellipsis = "..."
        while compact and self._text_width(draw, compact + ellipsis, font=font) > max_width:
            compact = compact[:-1].rstrip()
        return (compact + ellipsis) if compact else ellipsis

    # AUDIT-FIX(#1): Keep vendor imports pinned to project-local paths and reject path traversal outside project_root.
    def _resolve_vendor_dir(self, vendor_dir: Path) -> Path:
        candidate = vendor_dir.expanduser()
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        resolved = candidate.resolve(strict=False)
        if not resolved.is_relative_to(self.project_root):
            raise RuntimeError(
                "Display vendor directory must stay inside the Twinr project root."
            )
        return resolved

    # AUDIT-FIX(#1): Reject symlinked or missing vendor entrypoints before importing executable code from disk.
    def _validate_vendor_layout(self) -> Path:
        if self.vendor_dir.is_symlink():
            raise RuntimeError("Display vendor directory must not be a symlink.")

        package_dir = self.vendor_package_dir
        if not package_dir.exists():
            raise RuntimeError(
                "Display vendor files are missing. Run `hardware/display/setup_display.sh` first."
            )
        if not package_dir.is_dir():
            raise RuntimeError("Display vendor package path is invalid.")
        if package_dir.is_symlink():
            raise RuntimeError("Display vendor package path must not be a symlink.")

        resolved_package_dir = package_dir.resolve(strict=True)
        if not resolved_package_dir.is_relative_to(self.project_root):
            raise RuntimeError(
                "Display vendor package must stay inside the Twinr project root."
            )

        required_files = (
            resolved_package_dir / "__init__.py",
            resolved_package_dir / "epdconfig.py",
            resolved_package_dir / "epd4in2_V2.py",
        )
        for required_file in required_files:
            if not required_file.exists():
                raise RuntimeError(
                    "Display vendor files are incomplete. "
                    "Run `hardware/display/setup_display.sh` again."
                )
            if not required_file.is_file():
                raise RuntimeError(f"Display vendor file path is invalid: {required_file.name}.")
            if required_file.is_symlink():
                raise RuntimeError(
                    f"Display vendor file must not be a symlink: {required_file.name}."
                )

        return resolved_package_dir

    # AUDIT-FIX(#1): Import the vendor package from exact files instead of mutating sys.path.
    def _load_exact_vendor_module(
        self,
        *,
        module_name: str,
        module_path: Path,
        is_package: bool = False,
    ) -> object:
        resolved_path = module_path.resolve(strict=True)
        existing = sys.modules.get(module_name)
        if existing is not None and self._module_matches_path(existing, resolved_path):
            return existing

        if existing is not None:
            sys.modules.pop(module_name, None)

        spec = importlib.util.spec_from_file_location(
            module_name,
            str(resolved_path),
            submodule_search_locations=[str(resolved_path.parent)] if is_package else None,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load display vendor module: {module_name}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            gc.collect()
            raise
        return module

    def _module_matches_path(self, module: object, expected_path: Path) -> bool:
        module_file = getattr(module, "__file__", None)
        if not module_file:
            return False
        return Path(module_file).resolve(strict=False) == expected_path.resolve(strict=False)

    def _validate_driver_module_origin(self, module: object, expected_path: Path) -> None:
        if not self._module_matches_path(module, expected_path.resolve(strict=True)):
            raise RuntimeError(
                "Loaded display driver module origin does not match the validated vendor path."
            )

    # AUDIT-FIX(#5): On failures or shutdown, put the panel to sleep and close the vendor transport layer when available.
    def _shutdown_hardware(self) -> None:
        epd = self._epd
        epdconfig = self._epdconfig_module

        if epd is not None and hasattr(epd, "sleep"):
            with suppress(Exception):
                epd.sleep()

        if epdconfig is not None and hasattr(epdconfig, "module_exit"):
            module_exit = getattr(epdconfig, "module_exit")
            with suppress(Exception):
                try:
                    module_exit(cleanup=True)
                except TypeError:
                    module_exit()

    # AUDIT-FIX(#4): After any hardware fault, fully discard cached driver state so the next attempt starts from a clean init path.
    def _reset_driver_state(self) -> None:
        self._shutdown_hardware()
        self._drop_cached_vendor_modules()

    def _drop_cached_vendor_modules(self) -> None:
        """Discard cached vendor modules so the next render uses a fresh import."""

        self._epdconfig_module = None
        self._driver_module = None
        self._epd = None
        self._render_count = 0
        for module_name in (
            "waveshare_epd.epd4in2_V2",
            "waveshare_epd.epdconfig",
            "waveshare_epd",
        ):
            sys.modules.pop(module_name, None)
        gc.collect()

    # AUDIT-FIX(#2): Collapse whitespace and coerce arbitrary values into safe display strings, especially on error-reporting paths.
    def _normalise_text(self, value: object, *, fallback: str = "") -> str:
        if value is None:
            text = fallback
        else:
            text = str(value)
        text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        return " ".join(text.split())

    # AUDIT-FIX(#2): Accept string, iterable, or arbitrary objects for detail payloads so callers can pass exception objects without crashing the UI path.
    def _normalise_details(self, details: object) -> tuple[str, ...]:
        if details is None:
            return ()
        if isinstance(details, str):
            text = self._normalise_text(details)
            return (text,) if text else ()

        with suppress(TypeError):
            normalised = tuple(
                text
                for text in (self._normalise_text(item) for item in details)
                if text
            )
            return normalised

        text = self._normalise_text(details)
        return (text,) if text else ()

    # AUDIT-FIX(#2): Coerce animation_frame defensively so status rendering keeps working even when callers pass floats, strings, or None.
    def _normalise_animation_frame(self, value: object) -> int:
        with suppress(Exception):
            return int(value)
        return 0
