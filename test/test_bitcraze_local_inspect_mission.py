"""Regression coverage for the bounded Crazyflie local-inspect worker."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "hardware"
    / "bitcraze"
    / "run_local_inspect_mission.py"
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
_SPEC = importlib.util.spec_from_file_location(
    "bitcraze_local_inspect_mission_module",
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class LocalInspectConfigTests(unittest.TestCase):
    def test_validate_runtime_config_defaults_to_hardware_runtime(self) -> None:
        parser = _MODULE._build_parser()
        args = parser.parse_args(
            [
                "--bitcraze-python",
                "/tmp/bitcraze-python",
                "--artifact-root",
                "/tmp/drone-artifacts",
                "--image-name",
                "inspect.png",
            ]
        )

        config = _MODULE._validate_runtime_config(args)

        self.assertEqual(config["runtime_mode"], _MODULE.HOVER_RUNTIME_MODE_HARDWARE)
        self.assertEqual(config["required_decks"], ("bcFlow2", "bcZRanger2", "bcMultiranger"))

    def test_validate_runtime_config_defaults_to_no_required_decks_in_sitl(self) -> None:
        parser = _MODULE._build_parser()
        args = parser.parse_args(
            [
                "--runtime-mode",
                "sitl",
                "--bitcraze-python",
                "/tmp/bitcraze-python",
                "--artifact-root",
                "/tmp/drone-artifacts",
                "--image-name",
                "inspect.png",
            ]
        )

        config = _MODULE._validate_runtime_config(args)

        self.assertEqual(config["runtime_mode"], _MODULE.HOVER_RUNTIME_MODE_SITL)
        self.assertEqual(config["required_decks"], ())

    def test_run_local_inspect_mission_blocks_hardware_until_on_device_migration(self) -> None:
        report = _MODULE.run_local_inspect_mission(
            config={
                "repo_root": Path("/home/thh/twinr"),
                "env_file": None,
                "runtime_mode": _MODULE.HOVER_RUNTIME_MODE_HARDWARE,
                "uri": "radio://0/80/2M",
                "workspace": Path("/tmp/bitcraze-workspace"),
                "bitcraze_python": Path("/tmp/bitcraze-python"),
                "artifact_root": Path("/tmp/drone-artifacts"),
                "image_name": "inspect.png",
                "target_hint": "chair",
                "capture_intent": "still",
                "height_m": 0.25,
                "takeoff_velocity_mps": 0.2,
                "land_velocity_mps": 0.2,
                "translation_velocity_mps": 0.15,
                "nominal_translation_m": 0.2,
                "min_translation_m": 0.1,
                "max_translation_m": 0.3,
                "connect_settle_s": 0.0,
                "min_vbat_v": 3.8,
                "min_battery_level": 20,
                "min_clearance_m": 0.35,
                "stabilizer_estimator": 2,
                "stabilizer_controller": 1,
                "motion_disable": 0,
                "estimator_settle_timeout_s": 5.0,
                "on_device_failsafe_mode": "required",
                "on_device_failsafe_heartbeat_timeout_s": 0.35,
                "on_device_failsafe_low_battery_v": 3.55,
                "on_device_failsafe_critical_battery_v": 3.35,
                "on_device_failsafe_min_up_clearance_m": 0.25,
                "hover_settle_s": 0.35,
                "capture_dwell_s": 0.3,
                "required_decks": ("bcFlow2", "bcZRanger2", "bcMultiranger"),
            }
        )

        self.assertEqual(report.status, "blocked")
        self.assertIn("hardware local inspect is blocked", report.failures[0])


if __name__ == "__main__":
    unittest.main()
