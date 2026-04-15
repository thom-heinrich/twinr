"""Regression coverage for the Bitcraze firmware release policy helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
from typing import Any
import unittest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "hardware" / "bitcraze" / "firmware_release_policy.py"
_SPEC = importlib.util.spec_from_file_location("bitcraze_firmware_release_policy_module_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE: Any = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class BitcrazeFirmwareReleasePolicyTests(unittest.TestCase):
    def _create_build_attestation(self, temp_root: Path, *, with_patch: bool = False) -> tuple[Path, Path]:
        artifact_path = temp_root / "twinr_on_device_failsafe.bin"
        artifact_path.write_bytes(b"firmware")
        app_root = temp_root / "app"
        app_root.mkdir()
        (app_root / "main.c").write_text("int main(void) { return 0; }\n", encoding="utf-8")
        patch_paths: tuple[Path, ...] = ()
        if with_patch:
            patch_path = temp_root / "zranger.patch"
            patch_path.write_text("diff --git a/a b/a\n", encoding="utf-8")
            patch_paths = (patch_path,)
        attestation = _MODULE.create_build_attestation(
            artifact_path=artifact_path,
            firmware_root=temp_root / "crazyflie-firmware",
            firmware_git_commit="abc123def4567890",
            firmware_git_tag="2025.12.1",
            expected_firmware_revision="2025.12.1",
            app_root=app_root,
            app_tree_sha256=_MODULE.compute_tree_sha256(app_root),
            toolchain="arm-none-eabi-gcc",
            firmware_patch_paths=patch_paths,
        )
        attestation_path = temp_root / "build-attestation.json"
        _MODULE.write_attestation(attestation_path, attestation)
        return artifact_path, attestation_path

    def test_authorize_dev_lane_requires_matching_build_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            artifact_path, attestation_path = self._create_build_attestation(temp_root)

            decision = _MODULE.authorize_flash_request(
                lane="dev",
                device_role="dev",
                artifact_path=artifact_path,
                artifact_sha256=_MODULE.sha256_file(artifact_path),
                artifact_size_bytes=artifact_path.stat().st_size,
                artifact_source_kind="bin",
                artifact_release=None,
                artifact_repository=None,
                artifact_platform=None,
                artifact_target=None,
                artifact_firmware_type=None,
                attestation_path=attestation_path,
            )

        self.assertEqual(decision.lane, "dev")
        self.assertEqual(decision.device_role, "dev")
        self.assertEqual(decision.attestation_kind, _MODULE.BUILD_ATTESTATION_KIND)
        self.assertEqual(decision.source_firmware_git_tag, "2025.12.1")

    def test_build_attestation_round_trips_patch_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            _, attestation_path = self._create_build_attestation(temp_root, with_patch=True)

            loaded = _MODULE.load_attestation(attestation_path)

        self.assertEqual(len(loaded.source.firmware_patch_paths), 1)
        self.assertEqual(len(loaded.source.firmware_patch_sha256s), 1)

    def test_operator_promotion_requires_bench_release_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            artifact_path, attestation_path = self._create_build_attestation(temp_root)

            with self.assertRaises(_MODULE.FirmwareReleasePolicyError):
                _MODULE.create_release_attestation(
                    artifact_path=artifact_path,
                    lane="operator",
                    release_id="operator-r1",
                    source_attestation=_MODULE.load_attestation(attestation_path),
                    source_attestation_path=attestation_path,
                    approved_by="tester",
                    reason="promotion attempt",
                    validation_evidence=("test:test_bitcraze_firmware_release_policy.py",),
                )

    def test_recovery_lane_rejects_raw_bin(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            artifact_path, _ = self._create_build_attestation(temp_root)

            with self.assertRaises(_MODULE.FirmwareReleasePolicyError):
                _MODULE.authorize_flash_request(
                    lane="recovery",
                    device_role="operator",
                    artifact_path=artifact_path,
                    artifact_sha256=_MODULE.sha256_file(artifact_path),
                    artifact_size_bytes=artifact_path.stat().st_size,
                    artifact_source_kind="bin",
                    artifact_release=None,
                    artifact_repository=None,
                    artifact_platform=None,
                    artifact_target=None,
                    artifact_firmware_type=None,
                    attestation_path=None,
                )


if __name__ == "__main__":
    unittest.main()
