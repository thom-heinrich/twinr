from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.camera import V4L2StillCamera


class V4L2StillCameraTests(unittest.TestCase):
    def test_capture_photo_writes_png_and_tracks_input_format(self) -> None:
        camera = V4L2StillCamera(
            device="/dev/video0",
            width=640,
            height=480,
            framerate=30,
            ffmpeg_path="ffmpeg",
            input_format="bayer_grbg8",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "capture.png"
            with patch("twinr.hardware.camera.shutil.which", return_value="/usr/bin/ffmpeg"):
                with patch(
                    "twinr.hardware.camera.subprocess.run",
                    return_value=SimpleNamespace(returncode=0, stdout=b"PNGDATA", stderr=b""),
                ) as run_mock:
                    capture = camera.capture_photo(output_path=output_path, filename="capture.png")
                    written_bytes = output_path.read_bytes()

        self.assertEqual(capture.data, b"PNGDATA")
        self.assertEqual(capture.input_format, "bayer_grbg8")
        self.assertEqual(written_bytes, b"PNGDATA")
        command = run_mock.call_args.args[0]
        self.assertIn("-input_format", command)
        self.assertIn("bayer_grbg8", command)

    def test_capture_photo_falls_back_to_additional_formats(self) -> None:
        camera = V4L2StillCamera(
            device="/dev/video0",
            width=640,
            height=480,
            framerate=30,
        )

        responses = [
            SimpleNamespace(returncode=1, stdout=b"", stderr=b"default failed"),
            SimpleNamespace(returncode=0, stdout=b"PNGDATA", stderr=b""),
        ]

        with patch("twinr.hardware.camera.shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("twinr.hardware.camera.subprocess.run", side_effect=responses) as run_mock:
                capture = camera.capture_photo()

        self.assertEqual(capture.data, b"PNGDATA")
        self.assertEqual(capture.input_format, "yuyv422")
        self.assertEqual(run_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
