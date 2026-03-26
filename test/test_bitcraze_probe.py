"""Regression coverage for the Bitcraze Crazyradio probe helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "hardware" / "bitcraze" / "probe_crazyradio.py"
_SPEC = importlib.util.spec_from_file_location("bitcraze_probe_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class BitcrazeProbeTests(unittest.TestCase):
    def test_parse_proc_mounts_keeps_all_mountpoints(self) -> None:
        mounts = _MODULE.parse_proc_mounts(
            "/dev/sda /media/thh/Crazyradio2 vfat rw 0 0\n"
            "/dev/sda /run/user/1000/gvfs vfat rw 0 0\n"
        )

        self.assertEqual(
            mounts["/dev/sda"],
            ["/media/thh/Crazyradio2", "/run/user/1000/gvfs"],
        )

    def test_classify_pa_emulation(self) -> None:
        mode = _MODULE.classify_usb_mode("1915", "7777", info_uf2_present=False)
        self.assertEqual(mode, "pa_emulation")

    def test_classify_bootloader_when_uf2_volume_is_present(self) -> None:
        mode = _MODULE.classify_usb_mode("35f0", "bad2", info_uf2_present=True)
        self.assertEqual(mode, "uf2_bootloader")

    def test_classify_native_crazyradio2_without_uf2_volume(self) -> None:
        mode = _MODULE.classify_usb_mode("35f0", "bad2", info_uf2_present=False)
        self.assertEqual(mode, "crazyradio2_native")

    def test_probe_workspace_reports_missing_venv_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = _MODULE.probe_workspace(Path(temp_dir), try_cflib_access=False)

        self.assertTrue(probe.exists)
        self.assertIsNone(probe.venv_python)
        self.assertIsNone(probe.cflib_version)
        self.assertFalse(probe.radio_access_attempted)

    def test_validate_report_fails_when_required_mode_is_missing(self) -> None:
        device = _MODULE.BitcrazeUsbDevice(
            sysfs_path="/sys/bus/usb/devices/1-9",
            vendor_id="35f0",
            product_id="bad2",
            manufacturer="Bitcraze AB",
            product="Crazyradio 2.0",
            serial="049867FCE657E611",
            tty_nodes=("/dev/ttyACM0",),
            block_devices=("/dev/sda",),
            mountpoints=("/media/thh/Crazyradio2",),
            info_uf2_present=True,
            mode="uf2_bootloader",
            recommendation="flash the pinned PA emulation UF2 for cflib compatibility",
        )
        workspace = _MODULE.WorkspaceProbe(
            workspace="/twinr/bitcraze",
            exists=True,
            venv_python="/twinr/bitcraze/.venv/bin/python",
            cflib_version="0.1.31",
            cfclient_version="2025.12.1",
            radio_access_attempted=False,
            radio_access_ok=False,
            radio_access_error=None,
            detected_radio_serials=(),
            radio_version=None,
        )

        failures = _MODULE.validate_report(
            devices=[device],
            workspace_probe=workspace,
            expect_mode="pa_emulation",
            require_cflib_access=False,
        )

        self.assertEqual(failures, ["expected mode pa_emulation, found uf2_bootloader"])
