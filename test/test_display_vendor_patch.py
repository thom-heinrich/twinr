from pathlib import Path
import importlib.util
import sys
import tempfile
import types
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "hardware" / "display" / "vendor_patch.py"
MODULE_SPEC = importlib.util.spec_from_file_location("twinr_display_vendor_patch", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
vendor_patch = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(vendor_patch)


SAMPLE_EPDCONFIG = """import os
import logging
import sys
import time
import subprocess

from ctypes import *

logger = logging.getLogger(__name__)


class RaspberryPi:
    RST_PIN  = 17
    DC_PIN   = 25
    CS_PIN   = 8
    BUSY_PIN = 24
    PWR_PIN  = 18

    def __init__(self):
        import spidev
        import gpiozero

        self.SPI = spidev.SpiDev()
        self.GPIO_RST_PIN    = gpiozero.LED(self.RST_PIN)
        self.GPIO_DC_PIN     = gpiozero.LED(self.DC_PIN)
        # self.GPIO_CS_PIN     = gpiozero.LED(self.CS_PIN)
        self.GPIO_PWR_PIN    = gpiozero.LED(self.PWR_PIN)
        self.GPIO_BUSY_PIN   = gpiozero.Button(self.BUSY_PIN, pull_up = False)

    def module_init(self, cleanup=False):
        if cleanup:
            return 0
        self.SPI.open(0, 0)
        return 0

    def module_exit(self, cleanup=False):
        self.SPI.close()
        if cleanup:
            self.GPIO_RST_PIN.close()
            self.GPIO_DC_PIN.close()
            self.GPIO_PWR_PIN.close()
            self.GPIO_BUSY_PIN.close()


if sys.version_info[0] == 2:
    process = subprocess.Popen("cat /proc/cpuinfo | grep Raspberry", shell=True, stdout=subprocess.PIPE)
else:
    process = subprocess.Popen("cat /proc/cpuinfo | grep Raspberry", shell=True, stdout=subprocess.PIPE, text=True)
output, _ = process.communicate()
if sys.version_info[0] == 2:
    output = output.decode(sys.stdout.encoding)

if "Raspberry" in output:
    implementation = RaspberryPi()
"""


class VendorPatchTests(unittest.TestCase):
    def test_patch_updates_pins_and_spi_bus(self) -> None:
        patched = vendor_patch.patch_waveshare_epdconfig(
            SAMPLE_EPDCONFIG,
            pin_values={"RST_PIN": 5, "DC_PIN": 6, "CS_PIN": 7, "BUSY_PIN": 8},
            spi_bus=1,
            spi_device=2,
        )

        self.assertIn("RST_PIN  = 5", patched)
        self.assertIn("DC_PIN   = 6", patched)
        self.assertIn("CS_PIN   = 7", patched)
        self.assertIn("BUSY_PIN = 8", patched)
        self.assertIn("self.SPI.open(1, 2)", patched)
        self.assertIn("_safe_close_resource", patched)
        self.assertIn("self.SPI = None", patched)

    def test_patch_closes_partial_gpio_resources_on_constructor_failure(self) -> None:
        patched = vendor_patch.patch_waveshare_epdconfig(
            SAMPLE_EPDCONFIG,
            pin_values={"RST_PIN": 17, "DC_PIN": 25, "CS_PIN": 8, "BUSY_PIN": 24},
            spi_bus=0,
            spi_device=0,
        )
        class_source = patched.split("if sys.version_info[0] == 2:")[0]
        created_devices: list[object] = []

        class _FakeSpi:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        class _FakeDevice:
            def __init__(self, pin: int, *, should_fail: bool = False) -> None:
                self.pin = pin
                self.closed = False
                created_devices.append(self)
                if should_fail:
                    raise RuntimeError("gpio busy")

            def close(self) -> None:
                self.closed = True

        led_calls = {"count": 0}

        def _fake_led(pin: int) -> _FakeDevice:
            led_calls["count"] += 1
            return _FakeDevice(pin, should_fail=led_calls["count"] == 2)

        def _fake_button(pin: int, pull_up: bool = False) -> _FakeDevice:
            del pull_up
            return _FakeDevice(pin)

        fake_spidev = types.SimpleNamespace(SpiDev=_FakeSpi)
        fake_gpiozero = types.SimpleNamespace(LED=_fake_led, Button=_fake_button)
        original_spidev = sys.modules.get("spidev")
        original_gpiozero = sys.modules.get("gpiozero")
        sys.modules["spidev"] = fake_spidev
        sys.modules["gpiozero"] = fake_gpiozero
        namespace: dict[str, object] = {}
        try:
            exec(class_source, namespace, namespace)
            raspberry_pi = namespace["RaspberryPi"]
            with self.assertRaisesRegex(RuntimeError, "gpio busy"):
                raspberry_pi()
        finally:
            if original_spidev is None:
                sys.modules.pop("spidev", None)
            else:
                sys.modules["spidev"] = original_spidev
            if original_gpiozero is None:
                sys.modules.pop("gpiozero", None)
            else:
                sys.modules["gpiozero"] = original_gpiozero

        self.assertGreaterEqual(len(created_devices), 1)
        self.assertTrue(created_devices[0].closed)

    def test_write_vendor_package_writes_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            original_download = vendor_patch.download_text

            def _fake_download(url: str) -> str:
                if url == vendor_patch.EPDCONFIG_URL:
                    return SAMPLE_EPDCONFIG
                if url == vendor_patch.DRIVER_URL:
                    return "class EPD:\n    pass\n"
                raise AssertionError(url)

            vendor_patch.download_text = _fake_download
            try:
                vendor_patch.write_vendor_package(
                    vendor_dir=temp_path / "vendor" / "waveshare_epd",
                    pin_values={"RST_PIN": 5, "DC_PIN": 6, "CS_PIN": 7, "BUSY_PIN": 8},
                    spi_bus=1,
                    spi_device=2,
                )
            finally:
                vendor_patch.download_text = original_download

            self.assertTrue((temp_path / "vendor" / "waveshare_epd" / "__init__.py").is_file())
            epdconfig = (temp_path / "vendor" / "waveshare_epd" / "epdconfig.py").read_text(encoding="utf-8")
            self.assertIn("self.SPI.open(1, 2)", epdconfig)
            self.assertIn("_safe_close_resource", epdconfig)


if __name__ == "__main__":
    unittest.main()
