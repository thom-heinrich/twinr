#!/usr/bin/env python3
"""Download and patch the Waveshare vendor display package for Twinr.

This script owns the Pi-side vendor-package generation used by
`hardware/display/setup_display.sh`. It keeps the generated driver under
`state/display/vendor/`, applies Twinr's configured GPIO/SPI wiring, and
hardens the upstream Raspberry Pi transport constructor so partial gpiozero
allocation failures release already-claimed pins before re-raising.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping
import urllib.request


EPDCONFIG_URL = (
    "https://raw.githubusercontent.com/waveshareteam/e-Paper/master/"
    "RaspberryPi_JetsonNano/python/lib/waveshare_epd/epdconfig.py"
)
DRIVER_URL = (
    "https://raw.githubusercontent.com/waveshareteam/e-Paper/master/"
    "RaspberryPi_JetsonNano/python/lib/waveshare_epd/epd4in2_V2.py"
)
PIN_KEYS = ("RST_PIN", "DC_PIN", "CS_PIN", "BUSY_PIN")
_SAFE_CLOSE_HELPER = """
def _safe_close_resource(resource):
    if resource is None:
        return
    try:
        resource.close()
    except Exception:
        pass
"""
_RPI_INIT_START = "    def __init__(self):\n"


def download_text(url: str) -> str:
    """Fetch one UTF-8 vendor source file with bounded network behavior."""

    with urllib.request.urlopen(url, timeout=20) as response:
        return response.read().decode("utf-8", "ignore")


def patch_waveshare_epdconfig(
    source_text: str,
    *,
    pin_values: Mapping[str, int | str],
    spi_bus: int,
    spi_device: int,
) -> str:
    """Patch Waveshare's `epdconfig.py` for Twinr's runtime invariants."""

    patched = _patch_pin_assignments(source_text, pin_values)
    patched = _patch_spi_open(patched, spi_bus=spi_bus, spi_device=spi_device)
    patched = _inject_gc_import(patched)
    patched = _inject_safe_close_helper(patched)
    patched = _patch_raspberrypi_constructor(patched)
    return patched


def write_vendor_package(
    *,
    vendor_dir: Path,
    pin_values: Mapping[str, int | str],
    spi_bus: int,
    spi_device: int,
) -> None:
    """Download the vendor package, apply Twinr patches, and write it to disk."""

    vendor_dir.mkdir(parents=True, exist_ok=True)
    epdconfig = patch_waveshare_epdconfig(
        download_text(EPDCONFIG_URL),
        pin_values=pin_values,
        spi_bus=spi_bus,
        spi_device=spi_device,
    )
    driver = download_text(DRIVER_URL)
    (vendor_dir / "__init__.py").write_text("", encoding="utf-8")
    (vendor_dir / "epdconfig.py").write_text(epdconfig, encoding="utf-8")
    (vendor_dir / "epd4in2_V2.py").write_text(driver, encoding="utf-8")


def _patch_pin_assignments(
    source_text: str,
    pin_values: Mapping[str, int | str],
) -> str:
    """Rewrite the Raspberry Pi GPIO constants in the downloaded vendor file."""

    patched_lines: list[str] = []
    for line in source_text.splitlines():
        stripped = line.strip()
        left, separator, _right = stripped.partition("=")
        key = left.strip()
        if separator and key in pin_values:
            indent = line[: len(line) - len(line.lstrip())]
            patched_lines.append(f"{indent}{key:<8} = {pin_values[key]}")
            continue
        patched_lines.append(line)
    return "\n".join(patched_lines) + "\n"


def _patch_spi_open(source_text: str, *, spi_bus: int, spi_device: int) -> str:
    """Rewrite the upstream default SPI bus/device assignment."""

    desired_line = f"            self.SPI.open({spi_bus}, {spi_device})"
    if desired_line in source_text:
        return source_text
    patched_lines: list[str] = []
    replaced = False
    for line in source_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("self.SPI.open(") and not replaced:
            indent = line[: len(line) - len(line.lstrip())]
            patched_lines.append(f"{indent}self.SPI.open({spi_bus}, {spi_device})")
            replaced = True
            continue
        patched_lines.append(line)
    if not replaced:
        raise RuntimeError("Waveshare epdconfig.py no longer exposes the expected SPI open call")
    return "\n".join(patched_lines) + "\n"


def _inject_safe_close_helper(source_text: str) -> str:
    """Inject a best-effort resource closer once into the vendor module."""

    if "_safe_close_resource(" in source_text:
        return source_text
    anchor = "logger = logging.getLogger(__name__)\n"
    if anchor not in source_text:
        raise RuntimeError("Waveshare epdconfig.py no longer exposes the expected logger anchor")
    return source_text.replace(anchor, anchor + _SAFE_CLOSE_HELPER + "\n", 1)


def _inject_gc_import(source_text: str) -> str:
    """Ensure the vendor module imports `gc` for partial-init cleanup."""

    if "import gc\n" in source_text:
        return source_text
    anchor = "import logging\n"
    if anchor not in source_text:
        raise RuntimeError("Waveshare epdconfig.py no longer exposes the expected import anchor")
    return source_text.replace(anchor, anchor + "import gc\n", 1)


def _patch_raspberrypi_constructor(source_text: str) -> str:
    """Harden the Raspberry Pi transport constructor against partial failures."""

    if "self.SPI = None" in source_text and "_safe_close_resource(self.GPIO_RST_PIN)" in source_text:
        return source_text
    start_index = source_text.find(_RPI_INIT_START)
    end_index = source_text.find("\n    def ", start_index + len(_RPI_INIT_START))
    if start_index == -1 or end_index == -1:
        raise RuntimeError(
            "Waveshare epdconfig.py no longer matches the expected RaspberryPi.__init__ block"
        )
    replacement = """
    def __init__(self):
        import spidev
        import gpiozero

        self.SPI = None
        self.GPIO_RST_PIN = None
        self.GPIO_DC_PIN = None
        self.GPIO_PWR_PIN = None
        self.GPIO_BUSY_PIN = None
        try:
            self.SPI = spidev.SpiDev()
            self.GPIO_RST_PIN = gpiozero.LED(self.RST_PIN)
            self.GPIO_DC_PIN = gpiozero.LED(self.DC_PIN)
            # self.GPIO_CS_PIN     = gpiozero.LED(self.CS_PIN)
            self.GPIO_PWR_PIN = gpiozero.LED(self.PWR_PIN)
            self.GPIO_BUSY_PIN = gpiozero.Button(self.BUSY_PIN, pull_up=False)
        except Exception:
            _safe_close_resource(self.GPIO_RST_PIN)
            _safe_close_resource(self.GPIO_DC_PIN)
            _safe_close_resource(self.GPIO_PWR_PIN)
            _safe_close_resource(self.GPIO_BUSY_PIN)
            _safe_close_resource(self.SPI)
            gc.collect()
            raise
"""
    return source_text[:start_index] + replacement + source_text[end_index:]


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for vendor-package generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-dir", required=True, help="Target directory that will contain waveshare_epd/")
    parser.add_argument("--spi-bus", type=int, default=0)
    parser.add_argument("--spi-device", type=int, default=0)
    parser.add_argument("--reset-gpio", type=int, required=True)
    parser.add_argument("--dc-gpio", type=int, required=True)
    parser.add_argument("--cs-gpio", type=int, required=True)
    parser.add_argument("--busy-gpio", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    """Run the vendor-package download and patch flow."""

    args = _parse_args()
    pin_values = {
        "RST_PIN": args.reset_gpio,
        "DC_PIN": args.dc_gpio,
        "CS_PIN": args.cs_gpio,
        "BUSY_PIN": args.busy_gpio,
    }
    for key in PIN_KEYS:
        if key not in pin_values:
            raise RuntimeError(f"Missing pin mapping for {key}")
    write_vendor_package(
        vendor_dir=Path(args.vendor_dir) / "waveshare_epd",
        pin_values=pin_values,
        spi_bus=args.spi_bus,
        spi_device=args.spi_device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
