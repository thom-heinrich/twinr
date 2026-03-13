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
    rotation_degrees: int = 270
    _driver_module: object | None = field(default=None, init=False, repr=False)
    _epd: object | None = field(default=None, init=False, repr=False)
    _render_count: int = field(default=0, init=False, repr=False)

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
            rotation_degrees=config.display_rotation_degrees,
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
        epd = self._get_epd()
        prepared_image = self._prepare_image(image)

        if clear_first or self._render_count == 0:
            self._init_full(epd)
            if clear_first and hasattr(epd, "Clear"):
                epd.Clear()
            prepared = epd.getbuffer(prepared_image)
            epd.display(prepared)
            self._render_count = 1
            return

        if self._render_count % 48 == 0:
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

    def close(self) -> None:
        if self._epd is None:
            return
        with suppress(Exception):
            self._epd.sleep()
        self._epd = None
        self._render_count = 0

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
        draw.rectangle((0, 0, canvas_width - 1, canvas_height - 1), fill=255)
        draw.rectangle((0, 0, canvas_width - 1, 42), fill=0)
        draw.text((18, 12), "TWINR", fill=255)
        status_label = (headline or status).upper()
        label_text = status_label[:16]
        label_width = min(len(label_text) * 10, canvas_width - 120)
        draw.text((canvas_width - label_width - 20, 12), label_text, fill=255)

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
        face_center_y = (canvas_height // 2) + 22
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

        brow_y = center_y - 52 + eye["brow_raise"]
        if side == "left":
            draw.line(
                (
                    center_x - 24,
                    brow_y + eye["brow_slant"],
                    center_x + 24,
                    brow_y - eye["brow_slant"],
                ),
                fill=0,
                width=4,
            )
        else:
            draw.line(
                (
                    center_x - 24,
                    brow_y - eye["brow_slant"],
                    center_x + 24,
                    brow_y + eye["brow_slant"],
                ),
                fill=0,
                width=4,
            )

        if eye["blink"]:
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

        width = eye["width"]
        height = eye["height"]
        offset_x = eye["eye_shift_x"]
        offset_y = eye["eye_shift_y"]
        box = (
            center_x - (width // 2) + offset_x,
            center_y - (height // 2) + offset_y,
            center_x + (width // 2) + offset_x,
            center_y + (height // 2) + offset_y,
        )
        draw.ellipse(box, fill=0)
        self._draw_eye_highlights(draw, box, eye)

        if eye["lid_arc"]:
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
            wave_y = center_y + (-1, 1, -1, 1)[animation_frame % 4]
            draw.arc((center_x - 24, wave_y - 6, center_x - 2, wave_y + 8), start=0, end=180, fill=0, width=4)
            draw.arc((center_x, wave_y - 6, center_x + 22, wave_y + 8), start=180, end=360, fill=0, width=4)
            return
        if status == "answering":
            openness = (12, 18, 10, 16)[animation_frame % 4]
            draw.rounded_rectangle(
                (center_x - 16, center_y - 6, center_x + 16, center_y + openness),
                radius=8,
                outline=0,
                width=4,
            )
            return
        if status == "printing":
            lift = (0, -1, 0, 1)[animation_frame % 4]
            draw.arc((center_x - 28, center_y - 6 + lift, center_x + 28, center_y + 16 + lift), start=12, end=168, fill=0, width=4)
            return
        if status == "error":
            draw.arc((center_x - 22, center_y + 6, center_x + 22, center_y + 18), start=200, end=340, fill=0, width=4)
            return
        draw.arc((center_x - 20, center_y - 8, center_x + 20, center_y + 8), start=20, end=160, fill=0, width=4)

    def _prepare_image(self, image: object):
        if hasattr(image, "rotate") and hasattr(image, "size"):
            width, height = image.size
            if (width, height) == self.canvas_size:
                return image.rotate(self.rotation_degrees % 360, expand=True)
        return image

    def _get_epd(self):
        if self._epd is None:
            module = self._load_driver_module()
            self._epd = module.EPD()
        return self._epd

    def _init_full(self, epd: object) -> None:
        epd.init()

    def _init_fast(self, epd: object) -> None:
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
            state["brow_raise"] = -2
            state["brow_slant"] = 7
            state["lid_arc"] = True
            return state

        if status == "answering":
            state["height"] = (70, 74, 70, 72)[animation_frame % 4]
            state["highlight_dx"] = (-8, -6, -8, -7)[animation_frame % 4]
            state["brow_raise"] = -3
            state["brow_slant"] = 3
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
