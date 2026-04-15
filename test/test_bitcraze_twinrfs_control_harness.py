"""Compile and exercise the pure C twinrFs control helpers through one harness."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import ClassVar
import unittest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS_SOURCE = _REPO_ROOT / "test" / "fixtures" / "twinrfs_control_harness.c"
_CONTROL_ROOT = _REPO_ROOT / "hardware" / "bitcraze" / "twinr_on_device_failsafe" / "src"
_VERTICAL_SOURCE = _CONTROL_ROOT / "twinr_on_device_failsafe_vertical_control.c"
_DISTURBANCE_SOURCE = _CONTROL_ROOT / "twinr_on_device_failsafe_disturbance_control.c"


class TwinrFsControlHarnessTests(unittest.TestCase):
    _temp_dir: ClassVar[tempfile.TemporaryDirectory[str]]
    _binary_path: ClassVar[Path]

    @classmethod
    def setUpClass(cls) -> None:
        compiler = shutil.which("cc")
        if compiler is None:
            raise RuntimeError("cc is required for the twinrFs control harness test")
        cls._temp_dir = tempfile.TemporaryDirectory()
        cls._binary_path = Path(cls._temp_dir.name) / "twinrfs_control_harness"
        subprocess.run(
            [
                compiler,
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                str(_CONTROL_ROOT),
                str(_HARNESS_SOURCE),
                str(_VERTICAL_SOURCE),
                str(_DISTURBANCE_SOURCE),
                "-lm",
                "-o",
                str(cls._binary_path),
            ],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp_dir.cleanup()

    def _run_scenario(self, scenario: str) -> dict[str, object]:
        completed = subprocess.run(
            [str(self._binary_path), scenario],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    def _payload_bool(self, payload: dict[str, object], key: str) -> bool:
        value = payload[key]
        if not isinstance(value, bool):
            raise AssertionError(f"{key} is not a bool: {value!r}")
        return value

    def _payload_int(self, payload: dict[str, object], key: str) -> int:
        value = payload[key]
        if not isinstance(value, int):
            raise AssertionError(f"{key} is not an int: {value!r}")
        return value

    def _payload_float(self, payload: dict[str, object], key: str) -> float:
        value = payload[key]
        if not isinstance(value, float):
            raise AssertionError(f"{key} is not a float: {value!r}")
        return value

    def test_vertical_handoff_succeeds_without_abort(self) -> None:
        payload = self._run_scenario("vertical_handoff")

        self.assertTrue(self._payload_bool(payload, "handoff"))
        self.assertFalse(self._payload_bool(payload, "abort"))
        self.assertEqual(self._payload_int(payload, "progress_class"), 3)
        self.assertEqual(self._payload_int(payload, "ceiling_ms"), 0)
        self.assertEqual(self._payload_int(payload, "battery_limited"), 0)

    def test_vertical_controller_reaches_ceiling_and_aborts_without_progress(self) -> None:
        payload = self._run_scenario("vertical_ceiling_abort")

        self.assertFalse(self._payload_bool(payload, "handoff"))
        self.assertTrue(self._payload_bool(payload, "abort"))
        self.assertEqual(self._payload_int(payload, "progress_class"), 4)
        self.assertTrue(self._payload_bool(payload, "thrust_at_ceiling"))
        self.assertGreaterEqual(self._payload_int(payload, "ceiling_ms"), 350)

    def test_disturbance_controller_classifies_recoverable_bias(self) -> None:
        payload = self._run_scenario("disturbance_recoverable")

        self.assertTrue(self._payload_bool(payload, "valid"))
        self.assertTrue(self._payload_bool(payload, "recoverable"))
        self.assertFalse(self._payload_bool(payload, "abort"))
        self.assertEqual(self._payload_int(payload, "severity_class"), 1)
        self.assertGreater(self._payload_int(payload, "severity_permille"), 0)
        self.assertEqual(self._payload_int(payload, "near_ground"), 0)
        self.assertLess(self._payload_float(payload, "vx_command"), 0.0)
        self.assertGreater(self._payload_float(payload, "vy_command"), 0.0)

    def test_disturbance_controller_aborts_after_nonrecoverable_bias_persists(self) -> None:
        payload = self._run_scenario("disturbance_nonrecoverable")

        self.assertTrue(self._payload_bool(payload, "valid"))
        self.assertFalse(self._payload_bool(payload, "recoverable"))
        self.assertTrue(self._payload_bool(payload, "abort"))
        self.assertEqual(self._payload_int(payload, "severity_class"), 2)
        self.assertEqual(self._payload_int(payload, "nonrecoverable_count"), 5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
