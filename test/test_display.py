from pathlib import Path
import sys
import tempfile
import textwrap
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.display import WaveshareEPD4In2V2


class WaveshareDisplayTests(unittest.TestCase):
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
            token = object()

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

    def test_show_image_prefers_partial_refresh_after_first_render(self) -> None:
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

            display.show_image("first", clear_first=False)
            display.show_image("second", clear_first=False)

            module = display._load_driver_module()
            self.assertEqual(
                module.EVENTS,
                [
                    "init",
                    ("buffer", "first"),
                    ("display", "first"),
                    ("buffer", "second"),
                    ("display_partial", "second"),
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

            display.show_image("first", clear_first=False)
            display.show_image("second", clear_first=False)

            module = display._load_driver_module()
            self.assertEqual(
                module.EVENTS,
                [
                    "init",
                    ("buffer", "first"),
                    ("display", "first"),
                    ("init_fast", 7),
                    ("buffer", "second"),
                    ("display_fast", "second"),
                ],
            )


if __name__ == "__main__":
    unittest.main()
