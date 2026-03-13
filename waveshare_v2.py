from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
import sys
import time

from twinr.agent.base_agent.config import TwinrConfig


@dataclass(slots=True)
class WaveshareEPD4In2V2:
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
    _driver_module: object | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "WaveshareEPD4In2V2":
        return cls(
            project_root=Path(config.project_root),
            vendor_dir=Path(config.project_root) / config.display_vendor_dir,
            driver=config.display_driver,
            spi_bus=config.display_spi_bus,
            spi_device=config.display_spi_device,
            cs_gpio=config.display_cs_gpio,
            dc_gpio=config.display_dc_gpio,
            reset_gpio=config.display_reset_gpio,
            busy_gpio=config.display_busy_gpio,
            width=config.display_width,
            height=config.display_height,
        )

    @property
    def vendor_package_dir(self) -> Path:
        return self.vendor_dir / "waveshare_epd"

    @property
    def canvas_size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def show_test_pattern(self) -> None:
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
        image = self.render_status_image(
            status=status,
            headline=headline,
            details=details,
            animation_frame=animation_frame,
        )
        self.show_image(image, clear_first=False)

    def show_image(self, image: object, *, clear_first: bool) -> None:
        module = self._load_driver_module()
        epd = module.EPD()
        epd.init()
        try:
            if clear_first and hasattr(epd, "Clear"):
                epd.Clear()
            epd.display(epd.getbuffer(self._prepare_image(image)))
        finally:
            with suppress(Exception):
                epd.sleep()

    def render_test_image(self) -> object:
        image, draw = self._new_canvas()
        canvas_width, canvas_height = image.size
        draw.rectangle((0, 0, canvas_width - 1, canvas_height - 1), outline=0, width=5)
        draw.rectangle((0, 0, canvas_width - 1, 48), fill=0)
        draw.text((14, 14), "TWINR E-PAPER V2", fill=255)
        draw.text((14, 72), time.strftime("%Y-%m-%d %H:%M:%S"), fill=0)
        draw.text((14, 110), "BLACK / WHITE TEST", fill=0)
        draw.text((14, 148), "Display path is ready.", fill=0)
        draw.rectangle((20, 210, 120, 290), fill=0)
        draw.rectangle((140, 210, 240, 290), outline=0, width=3)
        return image

    def render_status_image(
        self,
        *,
        status: str,
        headline: str | None,
        details: tuple[str, ...],
        animation_frame: int = 0,
    ) -> object:
        image, draw = self._new_canvas()
        canvas_width, canvas_height = image.size
        draw.rounded_rectangle((0, 0, canvas_width - 1, canvas_height - 1), radius=18, outline=0, width=4)
        draw.rectangle((0, 0, canvas_width - 1, 42), fill=0)
        draw.text((18, 12), "TWINR", fill=255)
        status_label = (headline or status).upper()
        label_width = min(len(status_label) * 10, canvas_width - 120)
        draw.text((canvas_width - label_width - 20, 12), status_label[:14], fill=255)

        self._draw_face(
            draw,
            status=status.lower(),
            animation_frame=animation_frame,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )
        return image

    def _load_driver_module(self):
        if self._driver_module is not None:
            return self._driver_module

        if self.driver != "waveshare_4in2_v2":
            raise RuntimeError(f"Unsupported display driver: {self.driver}")
        if not self.vendor_package_dir.exists():
            raise RuntimeError(
                "Display vendor files are missing. Run `hardware/display/setup_display.sh` first."
            )

        vendor_parent = self.vendor_dir
        for module_name in [
            "waveshare_epd",
            "waveshare_epd.epdconfig",
            "waveshare_epd.epd4in2_V2",
        ]:
            sys.modules.pop(module_name, None)

        sys.path.insert(0, str(vendor_parent))
        try:
            try:
                epdconfig = import_module("waveshare_epd.epdconfig")
                module = import_module("waveshare_epd.epd4in2_V2")
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Display vendor files are incomplete. Run `hardware/display/setup_display.sh` again."
                ) from exc
        finally:
            sys.path.pop(0)

        self._validate_vendor_config(epdconfig)
        self._driver_module = module
        return module

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

        if mismatches:
            mismatch_text = ", ".join(mismatches)
            raise RuntimeError(
                "Installed display driver pins do not match Twinr config: "
                f"{mismatch_text}. Run `hardware/display/setup_display.sh` again."
            )

    def _new_canvas(self) -> tuple[object, object]:
        from PIL import Image, ImageDraw

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

        left_eye = (face_center_x - 116 + jitter_x, face_center_y - 46 + jitter_y)
        right_eye = (face_center_x + 12 + jitter_x, face_center_y - 46 + jitter_y)
        self._draw_eye(draw, left_eye, status=status, side="left", animation_frame=animation_frame)
        self._draw_eye(draw, right_eye, status=status, side="right", animation_frame=animation_frame)
        self._draw_mouth(
            draw,
            center_x=face_center_x + jitter_x,
            center_y=face_center_y + 56 + jitter_y,
            status=status,
            animation_frame=animation_frame,
        )

    def _draw_eye(
        self,
        draw: object,
        origin: tuple[int, int],
        *,
        status: str,
        side: str,
        animation_frame: int,
    ) -> None:
        x, y = origin
        width = 104
        height = 70
        blink = self._is_blinking(status, animation_frame)
        if blink:
            draw.arc((x, y + 18, x + width, y + 42), start=200, end=340, fill=0, width=6)
            return

        draw.ellipse((x, y, x + width, y + height), outline=0, width=5)
        pupil_dx, pupil_dy = self._pupil_offset(status, animation_frame, side)
        pupil_box = (
            x + 34 + pupil_dx,
            y + 20 + pupil_dy,
            x + 64 + pupil_dx,
            y + 50 + pupil_dy,
        )
        draw.ellipse(pupil_box, fill=0)
        draw.ellipse((pupil_box[0] + 5, pupil_box[1] + 5, pupil_box[0] + 12, pupil_box[1] + 12), fill=255)
        if status in {"processing", "printing"}:
            draw.arc((x + 12, y - 6, x + width - 12, y + 24), start=180, end=360, fill=0, width=4)
        if status == "error":
            draw.arc((x + 12, y - 2, x + width - 12, y + 18), start=180, end=330, fill=0, width=4)

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
            draw.arc((center_x - 34, center_y - 12, center_x + 34, center_y + 20), start=15, end=165, fill=0, width=4)
            return
        if status == "listening":
            draw.ellipse((center_x - 12, center_y - 10, center_x + 12, center_y + 16), outline=0, width=4)
            return
        if status == "processing":
            wave_y = center_y + ((animation_frame % 2) * 2)
            draw.arc((center_x - 30, wave_y - 8, center_x - 2, wave_y + 12), start=0, end=180, fill=0, width=4)
            draw.arc((center_x, wave_y - 8, center_x + 28, wave_y + 12), start=180, end=360, fill=0, width=4)
            return
        if status == "answering":
            openness = 10 + (animation_frame % 3) * 5
            draw.rounded_rectangle(
                (center_x - 18, center_y - 6, center_x + 18, center_y + openness),
                radius=8,
                outline=0,
                width=4,
            )
            return
        if status == "printing":
            draw.arc((center_x - 38, center_y - 4, center_x + 38, center_y + 24), start=10, end=170, fill=0, width=5)
            return
        if status == "error":
            draw.arc((center_x - 32, center_y + 4, center_x + 32, center_y + 26), start=200, end=340, fill=0, width=4)
            return
        draw.line((center_x - 22, center_y, center_x + 22, center_y), fill=0, width=4)

    def _prepare_image(self, image: object):
        if hasattr(image, "rotate") and hasattr(image, "size"):
            width, height = image.size
            if (width, height) == self.canvas_size:
                return image.rotate(90, expand=True)
        return image

    def _face_offset(self, status: str, animation_frame: int) -> tuple[int, int]:
        if status != "waiting":
            return (0, 0)
        offsets = ((0, 0), (-3, 0), (2, 1), (0, -1), (3, 0), (0, 0))
        return offsets[animation_frame % len(offsets)]

    def _is_blinking(self, status: str, animation_frame: int) -> bool:
        if status == "waiting":
            return animation_frame % 8 == 6
        if status == "printing":
            return animation_frame % 6 == 3
        return False

    def _pupil_offset(self, status: str, animation_frame: int, side: str) -> tuple[int, int]:
        if status == "waiting":
            cycle = ((0, 0), (-4, 0), (3, 1), (0, -1), (2, 0), (0, 0))
            return cycle[animation_frame % len(cycle)]
        if status == "listening":
            return (0, 2)
        if status == "processing":
            return ((-5, -2) if side == "left" else (5, -2))
        if status == "answering":
            return ((-2, 1) if side == "left" else (2, 1))
        if status == "printing":
            return ((-3, 0) if side == "left" else (3, 0))
        if status == "error":
            return (0, 0)
        return (0, 0)
