import builtins
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from types import SimpleNamespace

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.display import WaveshareEPD4In2V2
from twinr.config import TwinrConfig


def _prepared_image(width: int = 300, height: int = 400):
    return SimpleNamespace(size=(width, height))


def _prepared_image(width: int = 300, height: int = 400):
    return SimpleNamespace(size=(width, height))


class WaveshareDisplayTests(unittest.TestCase):
    def test_split_footer_parts_separates_time_suffix(self) -> None:
        display = WaveshareEPD4In2V2(
            project_root=Path("."),
            vendor_dir=Path("hardware/display/vendor"),
            width=400,
            height=300,
            rotation_degrees=270,
        )

        left, right = display._split_footer_parts("Internet ok | AI ok | System ok (12:34)")

        self.assertEqual(left, "Internet ok | AI ok | System ok")
        self.assertEqual(right, "(12:34)")

    def test_render_status_image_draws_health_footer(self) -> None:
        display = WaveshareEPD4In2V2(
            project_root=Path("."),
            vendor_dir=Path("hardware/display/vendor"),
            width=400,
            height=300,
            rotation_degrees=270,
        )

        plain = display.render_status_image(
            status="waiting",
            headline="Waiting",
            details=(),
            animation_frame=0,
        )
        with_footer = display.render_status_image(
            status="waiting",
            headline="Waiting",
            details=("Betrieb ok | 54C",),
            animation_frame=0,
        )

        plain_footer = plain.crop((24, 244, 376, 296))
        rendered_footer = with_footer.crop((24, 244, 376, 296))

        self.assertNotEqual(list(plain_footer.getdata()), list(rendered_footer.getdata()))

    def test_wrap_footer_left_splits_long_health_line_before_time_suffix(self) -> None:
        display = WaveshareEPD4In2V2(
            project_root=Path("."),
            vendor_dir=Path("hardware/display/vendor"),
            width=400,
            height=300,
            rotation_degrees=270,
        )
        _image, draw = display._new_canvas()
        footer_font = display._font(18, bold=False)

        lines = display._wrap_footer_left(
            draw,
            "Internet ok | AI ok | System ok",
            font=footer_font,
            full_width=240,
            final_width=160,
        )

        self.assertEqual(lines, ("Internet ok | AI ok", "System ok"))

    def test_footer_rows_keep_all_health_details(self) -> None:
        display = WaveshareEPD4In2V2(
            project_root=Path("."),
            vendor_dir=Path("hardware/display/vendor"),
            width=400,
            height=300,
            rotation_degrees=270,
        )

        rows = display._footer_rows(("Internet ok", "AI ok", "System ok", "Zeit 12:34"))

        self.assertEqual(
            rows,
            (
                ("Internet ok", "AI ok"),
                ("System ok", "Zeit 12:34"),
            ),
        )

    def test_render_status_image_uses_second_footer_row_for_multiple_details(self) -> None:
        display = WaveshareEPD4In2V2(
            project_root=Path("."),
            vendor_dir=Path("hardware/display/vendor"),
            width=400,
            height=300,
            rotation_degrees=270,
        )

        one_line = display.render_status_image(
            status="waiting",
            headline="Waiting",
            details=("Internet ok",),
            animation_frame=0,
        )
        multi_line = display.render_status_image(
            status="waiting",
            headline="Waiting",
            details=("Internet ok", "AI ok", "System ok", "Zeit 12:34"),
            animation_frame=0,
        )

        one_line_lower = one_line.crop((24, 270, 376, 296))
        multi_line_lower = multi_line.crop((24, 270, 376, 296))
        one_line_right = one_line.crop((210, 248, 376, 296))
        multi_line_right = multi_line.crop((210, 248, 376, 296))

        self.assertGreater(
            sum(1 for pixel in multi_line_lower.getdata() if pixel == 0),
            sum(1 for pixel in one_line_lower.getdata() if pixel == 0),
        )
        self.assertGreater(
            sum(1 for pixel in multi_line_right.getdata() if pixel == 0),
            sum(1 for pixel in one_line_right.getdata() if pixel == 0),
        )

    def test_render_status_image_debug_log_draws_section_titles_and_entries(self) -> None:
        display = WaveshareEPD4In2V2(
            project_root=Path("."),
            vendor_dir=Path("hardware/display/vendor"),
            width=400,
            height=300,
            rotation_degrees=270,
            layout_mode="debug_log",
        )

        image = display.render_status_image(
            status="waiting",
            headline="Waiting",
            details=("Internet ok", "AI ok", "System ok", "Zeit 12:34"),
            log_sections=(
                ("System Log", ("09:31 now Waiting", "net ok | ai ok | sys ok")),
                ("LLM Log", ("user Wie geht es dir heute?", "ai Mir geht es gut.")),
                ("Hardware Log", ("button green", "remote ok")),
            ),
            animation_frame=0,
        )

        header_region = image.crop((14, 8, 386, 42))
        system_region = image.crop((14, 48, 386, 118))
        lower_region = image.crop((14, 120, 386, 290))

        self.assertGreater(sum(1 for pixel in header_region.getdata() if pixel == 0), 200)
        self.assertGreater(sum(1 for pixel in system_region.getdata() if pixel == 0), 300)
        self.assertGreater(sum(1 for pixel in lower_region.getdata() if pixel == 0), 1000)

    def test_prepare_image_applies_configured_rotation(self) -> None:
        display = WaveshareEPD4In2V2(
            project_root=Path("."),
            vendor_dir=Path("hardware/display/vendor"),
            width=400,
            height=300,
            rotation_degrees=270,
        )

        image = Image.new("1", display.canvas_size, 255)

        rotated = display._prepare_image(image)

        self.assertEqual(rotated.size, (300, 400))

    def test_load_driver_module_requires_matching_pins(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            vendor_dir = Path(temp_dir) / "vendor" / "waveshare_epd"
            vendor_dir.mkdir(parents=True)
            (vendor_dir / "__init__.py").write_text("", encoding="utf-8")
            (vendor_dir / "epdconfig.py").write_text(
                textwrap.dedent(
                    """
                    RST_PIN = 99
                    DC_PIN = 25
                    CS_PIN = 8
                    BUSY_PIN = 24
                    PWR_PIN = 18
                    """
                ),
                encoding="utf-8",
            )
            (vendor_dir / "epd4in2_V2.py").write_text(
                "class EPD:\n    pass\n",
                encoding="utf-8",
            )

            display = WaveshareEPD4In2V2(
                project_root=Path(temp_dir),
                vendor_dir=Path(temp_dir) / "vendor",
            )

            with self.assertRaisesRegex(RuntimeError, "do not match Twinr config"):
                display._load_driver_module()

    def test_from_config_rejects_display_gpio_collision(self) -> None:
        config = TwinrConfig(
            project_root=".",
            green_button_gpio=23,
            yellow_button_gpio=24,
            display_busy_gpio=24,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Display BUSY GPIO 24 collides with yellow button GPIO 24.",
        ):
            WaveshareEPD4In2V2.from_config(config)

    def test_from_config_applies_display_layout_mode(self) -> None:
        config = TwinrConfig(
            project_root=".",
            display_layout="debug_log",
        )

        display = WaveshareEPD4In2V2.from_config(config)

        self.assertEqual(display.layout_mode, "debug_log")

    def test_load_driver_module_releases_failed_vendor_import_before_retry(self) -> None:
        sentinel_name = "__twinr_display_vendor_pin_busy__"
        with tempfile.TemporaryDirectory() as temp_dir:
            vendor_dir = Path(temp_dir) / "vendor" / "waveshare_epd"
            vendor_dir.mkdir(parents=True)
            (vendor_dir / "__init__.py").write_text("", encoding="utf-8")
            (vendor_dir / "epdconfig.py").write_text(
                textwrap.dedent(
                    f"""
                    import builtins

                    _SENTINEL = "{sentinel_name}"
                    if getattr(builtins, _SENTINEL, False):
                        raise RuntimeError("pin busy")
                    setattr(builtins, _SENTINEL, True)

                    RST_PIN = 17
                    DC_PIN = 25
                    CS_PIN = 8
                    BUSY_PIN = 24
                    PWR_PIN = 18

                    def module_exit(cleanup=False):
                        setattr(builtins, _SENTINEL, False)
                    """
                ),
                encoding="utf-8",
            )
            driver_path = vendor_dir / "epd4in2_V2.py"
            driver_path.write_text(
                textwrap.dedent(
                    """
                    from . import epdconfig

                    raise RuntimeError("driver import failed")
                    """
                ),
                encoding="utf-8",
            )

            display = WaveshareEPD4In2V2(
                project_root=Path(temp_dir),
                vendor_dir=Path(temp_dir) / "vendor",
            )

            try:
                with self.assertRaisesRegex(RuntimeError, "failed to import"):
                    display._load_driver_module()

                driver_path.write_text(
                    textwrap.dedent(
                        """
                        from . import epdconfig

                        class EPD:
                            pass
                        """
                    ),
                    encoding="utf-8",
                )

                module = display._load_driver_module()

                self.assertTrue(hasattr(module, "EPD"))
            finally:
                if hasattr(builtins, sentinel_name):
                    delattr(builtins, sentinel_name)

    def test_show_image_calls_vendor_driver_methods(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            vendor_dir = Path(temp_dir) / "vendor" / "waveshare_epd"
            vendor_dir.mkdir(parents=True)
            (vendor_dir / "__init__.py").write_text("", encoding="utf-8")
            (vendor_dir / "epdconfig.py").write_text(
                textwrap.dedent(
                    """
                    RST_PIN = 17
                    DC_PIN = 25
                    CS_PIN = 8
                    BUSY_PIN = 24
                    PWR_PIN = 18
                    """
                ),
                encoding="utf-8",
            )
            (vendor_dir / "epd4in2_V2.py").write_text(
                textwrap.dedent(
                    """
                    EVENTS = []

                    class EPD:
                        width = 400
                        height = 300

                        def init(self):
                            EVENTS.append("init")

                        def init_fast(self, mode):
                            EVENTS.append(("init_fast", mode))

                        def Clear(self):
                            EVENTS.append("clear")

                        def getbuffer(self, image):
                            EVENTS.append(("buffer", image))
                            return b"buffer"

                        def display(self, buffer):
                            EVENTS.append(("display", buffer))

                        def display_Fast(self, buffer):
                            EVENTS.append(("display_fast", buffer))

                        def sleep(self):
                            EVENTS.append("sleep")
                    """
                ),
                encoding="utf-8",
            )

            display = WaveshareEPD4In2V2(
                project_root=Path(temp_dir),
                vendor_dir=Path(temp_dir) / "vendor",
            )
            token = _prepared_image()

            display.show_image(token, clear_first=True)

            module = display._load_driver_module()
            self.assertEqual(
                module.EVENTS,
                [
                    "init",
                    "clear",
                    ("buffer", token),
                    ("display", b"buffer"),
                ],
            )

    def test_show_image_prefers_fast_refresh_over_partial_after_first_render(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            vendor_dir = Path(temp_dir) / "vendor" / "waveshare_epd"
            vendor_dir.mkdir(parents=True)
            (vendor_dir / "__init__.py").write_text("", encoding="utf-8")
            (vendor_dir / "epdconfig.py").write_text(
                textwrap.dedent(
                    """
                    RST_PIN = 17
                    DC_PIN = 25
                    CS_PIN = 8
                    BUSY_PIN = 24
                    PWR_PIN = 18
                    """
                ),
                encoding="utf-8",
            )
            (vendor_dir / "epd4in2_V2.py").write_text(
                textwrap.dedent(
                    """
                    EVENTS = []

                    class EPD:
                        width = 400
                        height = 300
                        Seconds_1_5S = 7

                        def init(self):
                            EVENTS.append("init")

                        def init_fast(self, mode):
                            EVENTS.append(("init_fast", mode))

                        def getbuffer(self, image):
                            EVENTS.append(("buffer", image))
                            return image

                        def display(self, buffer):
                            EVENTS.append(("display", buffer))

                        def display_Partial(self, buffer):
                            EVENTS.append(("display_partial", buffer))

                        def display_Fast(self, buffer):
                            EVENTS.append(("display_fast", buffer))

                        def sleep(self):
                            EVENTS.append("sleep")
                    """
                ),
                encoding="utf-8",
            )

            display = WaveshareEPD4In2V2(
                project_root=Path(temp_dir),
                vendor_dir=Path(temp_dir) / "vendor",
            )

            first = _prepared_image()
            second = _prepared_image()

            display.show_image(first, clear_first=False)
            display.show_image(second, clear_first=False)

            module = display._load_driver_module()
            self.assertEqual(
                module.EVENTS,
                [
                    "init",
                    ("buffer", first),
                    ("display", first),
                    ("init_fast", 7),
                    ("buffer", second),
                    ("display_fast", second),
                ],
            )

    def test_show_image_falls_back_to_fast_when_partial_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            vendor_dir = Path(temp_dir) / "vendor" / "waveshare_epd"
            vendor_dir.mkdir(parents=True)
            (vendor_dir / "__init__.py").write_text("", encoding="utf-8")
            (vendor_dir / "epdconfig.py").write_text(
                textwrap.dedent(
                    """
                    RST_PIN = 17
                    DC_PIN = 25
                    CS_PIN = 8
                    BUSY_PIN = 24
                    PWR_PIN = 18
                    """
                ),
                encoding="utf-8",
            )
            (vendor_dir / "epd4in2_V2.py").write_text(
                textwrap.dedent(
                    """
                    EVENTS = []

                    class EPD:
                        width = 400
                        height = 300
                        Seconds_1_5S = 7

                        def init(self):
                            EVENTS.append("init")

                        def init_fast(self, mode):
                            EVENTS.append(("init_fast", mode))

                        def getbuffer(self, image):
                            EVENTS.append(("buffer", image))
                            return image

                        def display(self, buffer):
                            EVENTS.append(("display", buffer))

                        def display_Fast(self, buffer):
                            EVENTS.append(("display_fast", buffer))

                        def sleep(self):
                            EVENTS.append("sleep")
                    """
                ),
                encoding="utf-8",
            )

            display = WaveshareEPD4In2V2(
                project_root=Path(temp_dir),
                vendor_dir=Path(temp_dir) / "vendor",
            )

            first = _prepared_image()
            second = _prepared_image()

            display.show_image(first, clear_first=False)
            display.show_image(second, clear_first=False)

            module = display._load_driver_module()
            self.assertEqual(
                module.EVENTS,
                [
                    "init",
                    ("buffer", first),
                    ("display", first),
                    ("init_fast", 7),
                    ("buffer", second),
                    ("display_fast", second),
                ],
            )

    def test_show_image_times_out_stuck_vendor_busy_wait(self) -> None:
        sentinel_name = "__twinr_display_busy_timeout_events__"
        with tempfile.TemporaryDirectory() as temp_dir:
            vendor_dir = Path(temp_dir) / "vendor" / "waveshare_epd"
            vendor_dir.mkdir(parents=True)
            (vendor_dir / "__init__.py").write_text("", encoding="utf-8")
            (vendor_dir / "epdconfig.py").write_text(
                textwrap.dedent(
                    f"""
                    import builtins

                    _EVENTS = getattr(builtins, "{sentinel_name}", None)
                    if _EVENTS is None:
                        _EVENTS = []
                        setattr(builtins, "{sentinel_name}", _EVENTS)
                    RST_PIN = 17
                    DC_PIN = 25
                    CS_PIN = 8
                    BUSY_PIN = 24
                    PWR_PIN = 18

                    def digital_read(pin):
                        _EVENTS.append(("digital_read", pin))
                        return 1

                    def delay_ms(delay):
                        _EVENTS.append(("delay_ms", delay))

                    def module_exit(cleanup=False):
                        _EVENTS.append(("module_exit", cleanup))
                    """
                ),
                encoding="utf-8",
            )
            (vendor_dir / "epd4in2_V2.py").write_text(
                textwrap.dedent(
                    f"""
                    import builtins

                    _EVENTS = getattr(builtins, "{sentinel_name}", None)
                    if _EVENTS is None:
                        _EVENTS = []
                        setattr(builtins, "{sentinel_name}", _EVENTS)

                    class EPD:
                        width = 400
                        height = 300
                        busy_pin = 24

                        def init(self):
                            _EVENTS.append("init")
                            self.ReadBusy()

                        def getbuffer(self, image):
                            _EVENTS.append(("buffer", image))
                            return image

                        def display(self, buffer):
                            _EVENTS.append(("display", buffer))

                        def sleep(self):
                            _EVENTS.append("sleep")
                    """
                ),
                encoding="utf-8",
            )

            display = WaveshareEPD4In2V2(
                project_root=Path(temp_dir),
                vendor_dir=Path(temp_dir) / "vendor",
                busy_timeout_s=0.01,
            )

            try:
                with self.assertRaisesRegex(RuntimeError, "failed after one recovery attempt"):
                    display.show_image(_prepared_image(), clear_first=False)

                events = getattr(builtins, sentinel_name)
                self.assertEqual(events.count("init"), 2)
                self.assertEqual(events.count("sleep"), 2)
                self.assertEqual(events.count(("module_exit", True)), 2)
            finally:
                if hasattr(builtins, sentinel_name):
                    delattr(builtins, sentinel_name)

    def test_show_image_can_force_full_refresh_on_configured_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            vendor_dir = Path(temp_dir) / "vendor" / "waveshare_epd"
            vendor_dir.mkdir(parents=True)
            (vendor_dir / "__init__.py").write_text("", encoding="utf-8")
            (vendor_dir / "epdconfig.py").write_text(
                textwrap.dedent(
                    """
                    RST_PIN = 17
                    DC_PIN = 25
                    CS_PIN = 8
                    BUSY_PIN = 24
                    PWR_PIN = 18
                    """
                ),
                encoding="utf-8",
            )
            (vendor_dir / "epd4in2_V2.py").write_text(
                textwrap.dedent(
                    """
                    EVENTS = []

                    class EPD:
                        width = 400
                        height = 300

                        def init(self):
                            EVENTS.append("init")

                        def getbuffer(self, image):
                            EVENTS.append(("buffer", image))
                            return image

                        def display(self, buffer):
                            EVENTS.append(("display", buffer))

                        def display_Partial(self, buffer):
                            EVENTS.append(("display_partial", buffer))
                    """
                ),
                encoding="utf-8",
            )

            display = WaveshareEPD4In2V2(
                project_root=Path(temp_dir),
                vendor_dir=Path(temp_dir) / "vendor",
                full_refresh_interval=2,
            )

            first = _prepared_image()
            second = _prepared_image()
            third = _prepared_image()

            display.show_image(first, clear_first=False)
            display.show_image(second, clear_first=False)
            display.show_image(third, clear_first=False)

            module = display._load_driver_module()
            self.assertEqual(
                module.EVENTS,
                [
                    "init",
                    ("buffer", first),
                    ("display", first),
                    ("buffer", second),
                    ("display_partial", second),
                    "init",
                    ("buffer", third),
                    ("display", third),
                ],
            )

    def test_show_image_debug_log_uses_fast_refresh_after_first_render(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            vendor_dir = Path(temp_dir) / "vendor" / "waveshare_epd"
            vendor_dir.mkdir(parents=True)
            (vendor_dir / "__init__.py").write_text("", encoding="utf-8")
            (vendor_dir / "epdconfig.py").write_text(
                textwrap.dedent(
                    """
                    RST_PIN = 17
                    DC_PIN = 25
                    CS_PIN = 8
                    BUSY_PIN = 24
                    PWR_PIN = 18
                    """
                ),
                encoding="utf-8",
            )
            (vendor_dir / "epd4in2_V2.py").write_text(
                textwrap.dedent(
                    """
                    EVENTS = []

                    class EPD:
                        width = 400
                        height = 300
                        Seconds_1_5S = 7

                        def init(self):
                            EVENTS.append("init")

                        def init_fast(self, mode):
                            EVENTS.append(("init_fast", mode))

                        def getbuffer(self, image):
                            EVENTS.append(("buffer", image))
                            return image

                        def display(self, buffer):
                            EVENTS.append(("display", buffer))

                        def display_Partial(self, buffer):
                            EVENTS.append(("display_partial", buffer))

                        def display_Fast(self, buffer):
                            EVENTS.append(("display_fast", buffer))
                    """
                ),
                encoding="utf-8",
            )

            display = WaveshareEPD4In2V2(
                project_root=Path(temp_dir),
                vendor_dir=Path(temp_dir) / "vendor",
                layout_mode="debug_log",
            )

            first = _prepared_image()
            second = _prepared_image()

            display.show_image(first, clear_first=False)
            display.show_image(second, clear_first=False)

            module = display._load_driver_module()
            self.assertEqual(
                module.EVENTS,
                [
                    "init",
                    ("buffer", first),
                    ("display", first),
                    ("init_fast", 7),
                    ("buffer", second),
                    ("display_fast", second),
                ],
            )


if __name__ == "__main__":
    unittest.main()
