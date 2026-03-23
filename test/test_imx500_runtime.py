from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.camera_ai.config import AICameraAdapterConfig
from twinr.hardware.camera_ai.imx500_runtime import IMX500RuntimeSessionManager, NetworkSession
from twinr.runtime_paths import prime_raspberry_pi_system_site_packages


class RuntimePathsTests(unittest.TestCase):
    def test_prime_raspberry_pi_system_site_packages_appends_missing_paths_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            device_model = root / "model"
            device_model.write_bytes(b"Raspberry Pi 5")
            system_a = root / "system-a"
            system_b = root / "system-b"
            missing = root / "missing"
            system_a.mkdir()
            system_b.mkdir()
            sys_path = ["/venv/site-packages", str(system_a)]

            added = prime_raspberry_pi_system_site_packages(
                sys_path=sys_path,
                candidate_paths=(system_a, system_b, missing),
                device_model_path=device_model,
            )

        self.assertEqual(added, (str(system_b),))
        self.assertEqual(sys_path, ["/venv/site-packages", str(system_a), str(system_b)])

    def test_prime_raspberry_pi_system_site_packages_skips_non_pi_hosts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            device_model = root / "model"
            device_model.write_bytes(b"Generic Linux Board")
            system_path = root / "system"
            system_path.mkdir()
            sys_path = ["/venv/site-packages"]

            added = prime_raspberry_pi_system_site_packages(
                sys_path=sys_path,
                candidate_paths=(system_path,),
                device_model_path=device_model,
            )

        self.assertEqual(added, ())
        self.assertEqual(sys_path, ["/venv/site-packages"])


class IMX500RuntimeTests(unittest.TestCase):
    def test_load_detection_runtime_primes_pi_system_paths_before_imports(self) -> None:
        manager = IMX500RuntimeSessionManager(config=AICameraAdapterConfig())
        picamera2_module = SimpleNamespace(Picamera2=object())
        imx500_module = SimpleNamespace(IMX500=object())

        with patch("twinr.hardware.camera_ai.imx500_runtime.prime_raspberry_pi_system_site_packages") as prime:
            with patch(
                "twinr.hardware.camera_ai.imx500_runtime.importlib.import_module",
                side_effect=[picamera2_module, imx500_module],
            ) as import_module:
                runtime = manager.load_detection_runtime()

        prime.assert_called_once_with()
        self.assertEqual(import_module.call_args_list[0].args, ("picamera2",))
        self.assertEqual(import_module.call_args_list[1].args, ("picamera2.devices.imx500",))
        self.assertIs(runtime["Picamera2"], picamera2_module.Picamera2)
        self.assertIs(runtime["IMX500"], imx500_module.IMX500)

    def test_load_pose_postprocess_primes_pi_system_paths_before_import(self) -> None:
        manager = IMX500RuntimeSessionManager(config=AICameraAdapterConfig())
        helper = object()
        module = SimpleNamespace(postprocess_higherhrnet=helper)

        with patch("twinr.hardware.camera_ai.imx500_runtime.prime_raspberry_pi_system_site_packages") as prime:
            with patch(
                "twinr.hardware.camera_ai.imx500_runtime.importlib.import_module",
                return_value=module,
            ) as import_module:
                loaded = manager.load_pose_postprocess()

        prime.assert_called_once_with()
        import_module.assert_called_once_with("picamera2.devices.imx500.postprocess_highernet")
        self.assertIs(loaded, helper)

    def test_capture_metadata_reduces_frame_rate_in_low_light(self) -> None:
        manager = IMX500RuntimeSessionManager(
            config=AICameraAdapterConfig(
                frame_rate=15,
                low_light_frame_rate=6,
                low_light_lux_threshold=1.2,
                low_light_recover_lux_threshold=2.2,
                metadata_wait_s=0.2,
            ),
            sleep_fn=lambda _seconds: None,
        )
        picam2 = _FakePicamera2(
            [
                {
                    "CnnOutputTensor": [1],
                    "Lux": 0.54,
                    "ExposureTime": 66379,
                    "AnalogueGain": 8.0,
                    "FrameDuration": 66657,
                }
            ]
        )
        session = NetworkSession(
            network_path="detector.rpk",
            task_name="detection",
            picam2=picam2,
            imx500=object(),
            input_size=(320, 320),
            configured_frame_rate=15.0,
        )
        manager._session = session

        metadata = manager.capture_metadata(session, observed_at=0.0)

        self.assertEqual(metadata["Lux"], 0.54)
        self.assertEqual(
            picam2.set_control_calls,
            [{"FrameRate": 6.0, "FrameDurationLimits": (166667, 166667)}],
        )
        self.assertAlmostEqual(session.configured_frame_rate, 6.0, places=3)
        metrics = manager.last_camera_metrics()
        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertTrue(metrics["camera_low_light_mode"])
        self.assertFalse(metrics["camera_manual_low_light_mode"])
        self.assertTrue(metrics["camera_exposure_saturated"])
        self.assertAlmostEqual(metrics["camera_lux"], 0.54, places=3)
        self.assertAlmostEqual(metrics["camera_configured_frame_rate"], 6.0, places=3)

    def test_capture_metadata_enables_manual_exposure_when_auto_preview_stays_capped(self) -> None:
        manager = IMX500RuntimeSessionManager(
            config=AICameraAdapterConfig(
                frame_rate=15,
                low_light_frame_rate=6,
                low_light_lux_threshold=1.2,
                low_light_recover_lux_threshold=2.2,
                low_light_manual_exposure_ratio=0.9,
                low_light_manual_analogue_gain=8.0,
                low_light_auto_exposure_cap_ratio=0.6,
                metadata_wait_s=0.2,
            ),
            sleep_fn=lambda _seconds: None,
        )
        picam2 = _FakePicamera2(
            [
                {
                    "CnnOutputTensor": [1],
                    "Lux": 0.54,
                    "ExposureTime": 66657,
                    "AnalogueGain": 8.0,
                    "FrameDuration": 166662,
                }
            ]
        )
        session = NetworkSession(
            network_path="detector.rpk",
            task_name="detection",
            picam2=picam2,
            imx500=object(),
            input_size=(320, 320),
            configured_frame_rate=6.0,
        )
        manager._session = session

        manager.capture_metadata(session, observed_at=0.0)

        self.assertEqual(
            picam2.set_control_calls,
            [
                {
                    "AeEnable": False,
                    "FrameRate": 6.0,
                    "FrameDurationLimits": (166667, 166667),
                    "ExposureTime": 150000,
                    "AnalogueGain": 8.0,
                }
            ],
        )
        self.assertTrue(session.manual_low_light_mode)
        metrics = manager.last_camera_metrics()
        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertTrue(metrics["camera_low_light_mode"])
        self.assertTrue(metrics["camera_manual_low_light_mode"])
        self.assertTrue(metrics["camera_auto_exposure_capped"])

    def test_capture_metadata_restores_default_frame_rate_after_lux_recovers(self) -> None:
        manager = IMX500RuntimeSessionManager(
            config=AICameraAdapterConfig(
                frame_rate=15,
                low_light_frame_rate=6,
                low_light_lux_threshold=1.2,
                low_light_recover_lux_threshold=2.2,
                metadata_wait_s=0.2,
            ),
            sleep_fn=lambda _seconds: None,
        )
        picam2 = _FakePicamera2(
            [
                {
                    "CnnOutputTensor": [1],
                    "Lux": 3.1,
                    "ExposureTime": 22000,
                    "AnalogueGain": 1.4,
                    "FrameDuration": 166666,
                }
            ]
        )
        session = NetworkSession(
            network_path="detector.rpk",
            task_name="detection",
            picam2=picam2,
            imx500=object(),
            input_size=(320, 320),
            configured_frame_rate=6.0,
        )
        manager._session = session

        manager.capture_metadata(session, observed_at=0.0)

        self.assertEqual(
            picam2.set_control_calls,
            [{"FrameRate": 15.0, "FrameDurationLimits": (66667, 66667)}],
        )
        self.assertAlmostEqual(session.configured_frame_rate, 15.0, places=3)
        metrics = manager.last_camera_metrics()
        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertFalse(metrics["camera_low_light_mode"])
        self.assertAlmostEqual(metrics["camera_lux"], 3.1, places=3)

    def test_capture_metadata_restores_auto_exposure_after_manual_low_light_recovers(self) -> None:
        manager = IMX500RuntimeSessionManager(
            config=AICameraAdapterConfig(
                frame_rate=15,
                low_light_frame_rate=6,
                low_light_lux_threshold=1.2,
                low_light_recover_lux_threshold=2.2,
                metadata_wait_s=0.2,
            ),
            sleep_fn=lambda _seconds: None,
        )
        picam2 = _FakePicamera2(
            [
                {
                    "CnnOutputTensor": [1],
                    "Lux": 3.1,
                    "ExposureTime": 149976,
                    "AnalogueGain": 8.0,
                    "FrameDuration": 166662,
                }
            ]
        )
        session = NetworkSession(
            network_path="detector.rpk",
            task_name="detection",
            picam2=picam2,
            imx500=object(),
            input_size=(320, 320),
            configured_frame_rate=6.0,
            manual_low_light_mode=True,
        )
        manager._session = session

        manager.capture_metadata(session, observed_at=0.0)

        self.assertEqual(
            picam2.set_control_calls,
            [{"AeEnable": True, "FrameRate": 15.0, "FrameDurationLimits": (66667, 66667)}],
        )
        self.assertAlmostEqual(session.configured_frame_rate, 15.0, places=3)
        self.assertFalse(session.manual_low_light_mode)
        metrics = manager.last_camera_metrics()
        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertFalse(metrics["camera_manual_low_light_mode"])
        self.assertFalse(metrics["camera_low_light_mode"])
        self.assertFalse(metrics["camera_auto_exposure_capped"])


class _FakePicamera2:
    def __init__(self, metadata_frames: list[dict[str, object]]) -> None:
        self._metadata_frames = list(metadata_frames)
        self.set_control_calls: list[dict[str, object]] = []

    def capture_metadata(self) -> dict[str, object]:
        if len(self._metadata_frames) > 1:
            return dict(self._metadata_frames.pop(0))
        return dict(self._metadata_frames[0])

    def set_controls(self, values: dict[str, object]) -> None:
        self.set_control_calls.append(dict(values))


if __name__ == "__main__":
    unittest.main()
