"""Regression coverage for the Bitcraze firmware promotion helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
from typing import Any
import unittest


_POLICY_PATH = Path(__file__).resolve().parents[1] / "hardware" / "bitcraze" / "firmware_release_policy.py"
_POLICY_SPEC = importlib.util.spec_from_file_location("bitcraze_firmware_release_policy_for_promotion_test", _POLICY_PATH)
assert _POLICY_SPEC is not None and _POLICY_SPEC.loader is not None
_POLICY_MODULE: Any = importlib.util.module_from_spec(_POLICY_SPEC)
sys.modules[_POLICY_SPEC.name] = _POLICY_MODULE
_POLICY_SPEC.loader.exec_module(_POLICY_MODULE)

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "hardware" / "bitcraze" / "promote_on_device_failsafe_release.py"
_SPEC = importlib.util.spec_from_file_location("bitcraze_promote_release_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE: Any = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class PromoteOnDeviceFailsafeReleaseTests(unittest.TestCase):
    def test_promote_bench_release_writes_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            artifact_path = temp_root / "twinr_on_device_failsafe.bin"
            artifact_path.write_bytes(b"firmware")
            app_root = temp_root / "app"
            app_root.mkdir()
            (app_root / "main.c").write_text("int main(void) { return 0; }\n", encoding="utf-8")
            build_attestation = _POLICY_MODULE.create_build_attestation(
                artifact_path=artifact_path,
                firmware_root=temp_root / "crazyflie-firmware",
                firmware_git_commit="abc123def4567890",
                firmware_git_tag="2025.12.1",
                expected_firmware_revision="2025.12.1",
                app_root=app_root,
                app_tree_sha256=_POLICY_MODULE.compute_tree_sha256(app_root),
                toolchain="arm-none-eabi-gcc",
            )
            source_attestation_path = temp_root / "build-attestation.json"
            _POLICY_MODULE.write_attestation(source_attestation_path, build_attestation)

            report = _MODULE.promote_on_device_failsafe_release(
                artifact_path=artifact_path,
                source_attestation_path=source_attestation_path,
                lane="bench",
                release_id="bench-r1",
                approved_by="tester",
                reason="bench validation passed",
                validation_evidence=("test:test_bitcraze_promote_on_device_failsafe_release.py",),
                output_path=temp_root / "bench-release.json",
            )

            written = _POLICY_MODULE.load_attestation(Path(report.output_path))

        self.assertEqual(report.lane, "bench")
        self.assertEqual(report.release_id, "bench-r1")
        self.assertEqual(written.lane, "bench")
        self.assertEqual(written.release_id, "bench-r1")


if __name__ == "__main__":
    unittest.main()
