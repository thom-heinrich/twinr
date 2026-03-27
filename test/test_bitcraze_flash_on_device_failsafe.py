"""Regression coverage for the on-device failsafe flash/probe helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import unittest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "hardware" / "bitcraze" / "flash_on_device_failsafe.py"
_SPEC = importlib.util.spec_from_file_location("bitcraze_flash_on_device_failsafe_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE: Any = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class OnDeviceFailsafeFlashHelperTests(unittest.TestCase):
    def test_build_cfloader_command_supports_warm_and_cold_boot(self) -> None:
        command = _MODULE.build_cfloader_command(
            cfloader_executable=Path("/tmp/cfloader"),
            binary_path=Path("/tmp/twinr.bin"),
            uri="radio://0/80/2M",
            boot_mode="warm",
        )

        self.assertEqual(
            command,
            (
                "/tmp/cfloader",
                "-w",
                "radio://0/80/2M",
                "flash",
                "/tmp/twinr.bin",
                "stm32-fw",
            ),
        )
        self.assertEqual(
            _MODULE.build_cfloader_command(
                cfloader_executable=Path("/tmp/cfloader"),
                binary_path=Path("/tmp/twinr.bin"),
                uri="radio://0/80/2M",
                boot_mode="cold",
            ),
            (
                "/tmp/cfloader",
                "flash",
                "/tmp/twinr.bin",
                "stm32-fw",
            ),
        )

    def test_flash_on_device_failsafe_reports_successful_probe(self) -> None:
        observed_commands: list[list[str]] = []

        def _runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            observed_commands.append(list(args[0]))
            return subprocess.CompletedProcess(args[0], 0, stdout="flash ok\n", stderr="")

        original_probe = _MODULE._probe_loaded_failsafe
        try:
            _MODULE._probe_loaded_failsafe = lambda **_kwargs: _MODULE.OnDeviceFailsafeAvailability(
                loaded=True,
                protocol_version=1,
                enabled=1,
                state_code=1,
                state_name="monitoring",
                reason_code=0,
                reason_name="none",
                failures=(),
            )
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_root = Path(temp_dir)
                binary = temp_root / "twinr.bin"
                cfloader = temp_root / "cfloader"
                binary.write_bytes(b"fw")
                cfloader.write_text("#!/bin/sh\n")
                report = _MODULE.flash_on_device_failsafe(
                    uri="radio://0/80/2M",
                    workspace=temp_root,
                    binary_path=binary,
                    cfloader_executable=cfloader,
                    runner=_runner,
                )
        finally:
            _MODULE._probe_loaded_failsafe = original_probe

        self.assertTrue(report.flashed)
        self.assertTrue(report.probe_attempted)
        self.assertIsNotNone(report.availability)
        self.assertTrue(report.availability.loaded)
        self.assertEqual(report.failures, ())
        self.assertEqual(
            observed_commands,
            [[str(cfloader), "-w", "radio://0/80/2M", "flash", str(binary), "stm32-fw"]],
        )

    def test_flash_on_device_failsafe_reports_missing_post_flash_probe(self) -> None:
        def _runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args[0], 0, stdout="flash ok\n", stderr="")

        original_probe = _MODULE._probe_loaded_failsafe
        try:
            _MODULE._probe_loaded_failsafe = lambda **_kwargs: _MODULE.OnDeviceFailsafeAvailability(
                loaded=False,
                protocol_version=None,
                enabled=None,
                state_code=None,
                state_name=None,
                reason_code=None,
                reason_name=None,
                failures=("twinrFs.protocolVersion:missing",),
            )
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_root = Path(temp_dir)
                binary = temp_root / "twinr.bin"
                cfloader = temp_root / "cfloader"
                binary.write_bytes(b"fw")
                cfloader.write_text("#!/bin/sh\n")
                report = _MODULE.flash_on_device_failsafe(
                    workspace=temp_root,
                    binary_path=binary,
                    cfloader_executable=cfloader,
                    runner=_runner,
                )
        finally:
            _MODULE._probe_loaded_failsafe = original_probe

        self.assertTrue(report.flashed)
        self.assertTrue(any("protocolVersion" in failure for failure in report.failures))
        self.assertIn("post-flash probe did not see the `twinrFs` firmware app", report.failures)

    def test_probe_loaded_failsafe_reconnects_and_uses_probe_module(self) -> None:
        class _FakeCRTP:
            @staticmethod
            def init_drivers() -> None:
                return None

        class _FakeCrazyflie:
            def __init__(self, *, rw_cache: str) -> None:
                self.rw_cache = rw_cache
                self.param = object()

        class _FakeSyncCrazyflie:
            def __init__(self, uri: str, *, cf: object) -> None:
                self.uri = uri
                self.cf = cf

            def __enter__(self) -> "_FakeSyncCrazyflie":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

        original_import = _MODULE._import_cflib
        original_probe = _MODULE.probe_on_device_failsafe
        try:
            _MODULE._import_cflib = lambda: (_FakeCRTP, _FakeCrazyflie, _FakeSyncCrazyflie)
            _MODULE.probe_on_device_failsafe = lambda sync_cf: _MODULE.OnDeviceFailsafeAvailability(
                loaded=True,
                protocol_version=1,
                enabled=1,
                state_code=1,
                state_name="monitoring",
                reason_code=0,
                reason_name="none",
                failures=(str(sync_cf.cf.rw_cache),),
            )
            with tempfile.TemporaryDirectory() as temp_dir:
                availability = _MODULE._probe_loaded_failsafe(
                    uri="radio://0/80/2M",
                    workspace=Path(temp_dir),
                    settle_s=0.0,
                    sleep=lambda _seconds: None,
                )
        finally:
            _MODULE._import_cflib = original_import
            _MODULE.probe_on_device_failsafe = original_probe

        self.assertTrue(availability.loaded)
        self.assertTrue(any(".cache" in failure for failure in availability.failures))


if __name__ == "__main__":
    unittest.main()
