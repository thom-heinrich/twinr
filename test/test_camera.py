from contextlib import contextmanager
from pathlib import Path
import base64
import socket
import struct
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import patch
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.aideck_camera import AIDeckStillCamera, AIDeckStreamStalledError
from twinr.hardware.camera import (
    CameraCaptureFailedError,
    CameraCaptureTimeoutError,
    CameraConfigurationError,
    RPiCamStillCamera,
    V4L2StillCamera,
)

_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl9l2sAAAAASUVORK5CYII="
)
_RAW_BAYER_BYTES = bytes(
    [
        10,
        40,
        20,
        50,
        60,
        90,
        70,
        100,
        30,
        80,
        40,
        90,
        100,
        140,
        110,
        150,
    ]
)


class _FakeUrlopenResponse:
    def __init__(self, payload: bytes, *, content_type: str = "image/png") -> None:
        self._payload = payload
        self.headers = {"Content-Type": content_type}

    def __enter__(self) -> "_FakeUrlopenResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> Literal[False]:
        return False

    def read(self) -> bytes:
        return self._payload


class _FakeSocket:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0
        self.timeout: float | None = None

    def __enter__(self) -> "_FakeSocket":
        return self

    def __exit__(self, exc_type, exc, traceback) -> Literal[False]:
        return False

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def recv(self, size: int) -> bytes:
        if self._offset >= len(self._payload):
            return b""
        end = min(len(self._payload), self._offset + size)
        chunk = self._payload[self._offset : end]
        self._offset = end
        return chunk

    def recv_into(self, buffer, size: int) -> int:
        chunk = self.recv(size)
        view = memoryview(buffer)
        view[: len(chunk)] = chunk
        return len(chunk)


class _TimeoutOnRecvSocket(_FakeSocket):
    def __init__(self) -> None:
        super().__init__(b"")

    def recv(self, size: int) -> bytes:
        del size
        raise socket.timeout("timed out")


def _pack_aideck_stream(*, width: int, height: int, depth: int, encoding: int, image_bytes: bytes) -> bytes:
    header = struct.pack("<HBB", 13, 0x63, 0x05) + struct.pack(
        "<BHHBBI",
        0xBC,
        width,
        height,
        depth,
        encoding,
        len(image_bytes),
    )
    chunk = struct.pack("<HBB", len(image_bytes) + 2, 0x00, 0x00) + image_bytes
    return header + chunk


def _png_dimensions(payload: bytes) -> tuple[int, int]:
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    chunk_length = struct.unpack(">I", payload[8:12])[0]
    assert chunk_length == 13
    assert payload[12:16] == b"IHDR"
    width, height = struct.unpack(">II", payload[16:24])
    return width, height


def _png_rgb_rows(payload: bytes) -> list[bytes]:
    width, _height = _png_dimensions(payload)
    offset = 8
    compressed = bytearray()
    while offset < len(payload):
        chunk_length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        chunk_payload = payload[offset + 8 : offset + 8 + chunk_length]
        offset += 12 + chunk_length
        if chunk_type == b"IDAT":
            compressed.extend(chunk_payload)
        if chunk_type == b"IEND":
            break
    decompressed = zlib.decompress(bytes(compressed))
    rows: list[bytes] = []
    cursor = 0
    row_bytes = width * 3
    while cursor < len(decompressed):
        filter_type = decompressed[cursor]
        assert filter_type == 0
        cursor += 1
        row = bytes(decompressed[cursor : cursor + row_bytes])
        rows.append(row)
        cursor += len(row)
    return rows


class V4L2StillCameraTests(unittest.TestCase):
    @staticmethod
    def _which(binary_name: str) -> str | None:
        if binary_name == "ffmpeg":
            return "/usr/bin/ffmpeg"
        if binary_name == "rpicam-still":
            return "/usr/bin/rpicam-still"
        return None

    def test_capture_photo_writes_png_and_tracks_input_format(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            camera = V4L2StillCamera(
                device="/dev/video0",
                width=640,
                height=480,
                framerate=30,
                ffmpeg_path="ffmpeg",
                input_format="bayer_grbg8",
                output_root=temp_dir,
            )
            output_path = Path(temp_dir) / "capture.png"
            with patch("twinr.hardware.camera.shutil.which", return_value="/usr/bin/ffmpeg"):
                with patch(
                    "twinr.hardware.camera.subprocess.run",
                    return_value=SimpleNamespace(returncode=0, stdout=_PNG_BYTES, stderr=b""),
                ) as run_mock:
                    capture = camera.capture_photo(output_path=output_path, filename="capture.png")
                    written_bytes = output_path.read_bytes()

        self.assertEqual(capture.data, _PNG_BYTES)
        self.assertEqual(capture.input_format, "bayer_grbg8")
        self.assertEqual(written_bytes, _PNG_BYTES)
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
            SimpleNamespace(returncode=0, stdout=_PNG_BYTES, stderr=b""),
        ]

        with patch("twinr.hardware.camera.shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("twinr.hardware.camera.subprocess.run", side_effect=responses) as run_mock:
                capture = camera.capture_photo()

        self.assertEqual(capture.data, _PNG_BYTES)
        self.assertEqual(capture.input_format, "mjpeg")
        self.assertEqual(run_mock.call_count, 2)

    def test_capture_photo_does_not_switch_to_rpicam_still_after_v4l2_failure(self) -> None:
        camera = V4L2StillCamera(
            device="/dev/video0",
            width=640,
            height=480,
            framerate=30,
        )

        responses = [
            SimpleNamespace(returncode=1, stdout=b"", stderr=b"Device or resource busy"),
            SimpleNamespace(returncode=1, stdout=b"", stderr=b"Device or resource busy"),
            SimpleNamespace(returncode=1, stdout=b"", stderr=b"Device or resource busy"),
            SimpleNamespace(returncode=1, stdout=b"", stderr=b"Device or resource busy"),
        ]

        with patch("twinr.hardware.camera.shutil.which", side_effect=self._which):
            with patch("twinr.hardware.camera.subprocess.run", side_effect=responses) as run_mock:
                with self.assertRaises(CameraCaptureFailedError):
                    camera.capture_photo()

        self.assertEqual(run_mock.call_count, 4)
        self.assertTrue(
            all(call.args[0][0] == "/usr/bin/ffmpeg" for call in run_mock.call_args_list)
        )

    def test_capture_photo_raises_configuration_error_when_ffmpeg_is_missing(self) -> None:
        camera = V4L2StillCamera(
            device="/dev/video0",
            width=640,
            height=480,
            framerate=30,
        )

        with patch("twinr.hardware.camera.shutil.which", return_value=None):
            with self.assertRaises(CameraConfigurationError) as context:
                camera.capture_photo()

        self.assertIn("rpicam://<index>", str(context.exception))

    def test_capture_photo_raises_timeout_for_v4l2_lane(self) -> None:
        camera = V4L2StillCamera(
            device="/dev/video0",
            width=640,
            height=480,
            framerate=30,
        )

        with patch("twinr.hardware.camera.shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch(
                "twinr.hardware.camera.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=10.0),
            ):
                with self.assertRaises(CameraCaptureTimeoutError):
                    camera.capture_photo()

    def test_capture_photo_rejects_unicam_v4l2_lane_with_configuration_error(self) -> None:
        camera = V4L2StillCamera(
            device="/dev/video0",
            width=640,
            height=480,
            framerate=30,
        )

        def which_side_effect(binary: str) -> str | None:
            if binary == "v4l2-ctl":
                return "/usr/bin/v4l2-ctl"
            if binary == "ffmpeg":
                return "/usr/bin/ffmpeg"
            return None

        v4l2_result = SimpleNamespace(
            returncode=0,
            stdout=b"Driver name   : unicam\nCard type     : unicam-image\n",
            stderr=b"",
        )

        with patch("twinr.hardware.camera.shutil.which", side_effect=which_side_effect):
            with patch("twinr.hardware.camera.subprocess.run", return_value=v4l2_result) as run_mock:
                with self.assertRaises(CameraConfigurationError) as context:
                    camera.capture_photo()

        self.assertIn("rpicam://<index>", str(context.exception))
        self.assertEqual(run_mock.call_count, 1)
        self.assertEqual(run_mock.call_args.args[0][0], "/usr/bin/v4l2-ctl")

    def test_from_config_uses_explicit_rpicam_device_uri(self) -> None:
        config = SimpleNamespace(
            camera_device="rpicam://0",
            camera_width=640,
            camera_height=480,
            camera_framerate=30,
            camera_input_format=None,
            camera_ffmpeg_path="ffmpeg",
            camera_proxy_snapshot_url=None,
            camera_capture_timeout_seconds=4.0,
            camera_capture_output_dir=None,
        )

        camera = V4L2StillCamera.from_config(cast(TwinrConfig, config))

        self.assertIsInstance(camera, RPiCamStillCamera)

    def test_explicit_rpicam_camera_capture_uses_rpicam_still(self) -> None:
        camera = RPiCamStillCamera(
            camera_index=0,
            width=640,
            height=480,
            capture_timeout_seconds=4.0,
        )

        def run_side_effect(command: list[str], **_kwargs):
            self.assertEqual(command[0], "/usr/bin/rpicam-still")
            output_path = Path(command[command.index("--output") + 1])
            output_path.write_bytes(_PNG_BYTES)
            return SimpleNamespace(returncode=0, stdout=b"{}", stderr=b"")

        with patch("twinr.hardware.camera.shutil.which", side_effect=self._which):
            with patch("twinr.hardware.camera.subprocess.run", side_effect=run_side_effect):
                capture = camera.capture_photo()

        self.assertEqual(capture.data, _PNG_BYTES)
        self.assertEqual(capture.input_format, "rpicam-still")
        self.assertEqual(capture.source_device, "rpicam://0")

    def test_from_config_fetches_snapshot_from_peer_proxy_when_url_is_configured(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimpleNamespace(
                camera_device="/dev/video0",
                camera_width=640,
                camera_height=480,
                camera_framerate=30,
                camera_input_format=None,
                camera_ffmpeg_path="ffmpeg",
                camera_proxy_snapshot_url="http://10.42.0.2:8767/snapshot.png",
                camera_capture_timeout_seconds=9.0,
                camera_capture_output_dir=temp_dir,
            )
            camera = V4L2StillCamera.from_config(cast(TwinrConfig, config))
            proxy_camera = cast(Any, camera)
            output_path = Path(temp_dir) / "proxy-capture.png"

            proxy_camera._http_client = None
            proxy_camera._urllib_opener = SimpleNamespace(
                open=lambda *_args, **_kwargs: _FakeUrlopenResponse(_PNG_BYTES)
            )

            with patch.object(
                camera,
                "_fetch_snapshot_payload_urllib",
                return_value=(_PNG_BYTES, "image/png"),
            ) as fetch_mock:
                capture = camera.capture_photo(output_path=output_path, filename="proxy-capture.png")
                written_bytes = output_path.read_bytes()

        self.assertEqual(capture.data, _PNG_BYTES)
        self.assertEqual(capture.input_format, "http-snapshot")
        self.assertEqual(capture.source_device, "http://10.42.0.2:8767/snapshot.png")
        self.assertEqual(written_bytes, _PNG_BYTES)
        request_url = fetch_mock.call_args.args[0]
        self.assertIn("width=640", request_url)
        self.assertIn("height=480", request_url)
        self.assertIn("timeout_ms=9000", request_url)

    def test_snapshot_proxy_timeout_maps_socket_timeout_to_camera_timeout(self) -> None:
        config = SimpleNamespace(
            camera_device="/dev/video0",
            camera_width=640,
            camera_height=480,
            camera_framerate=30,
            camera_input_format=None,
            camera_ffmpeg_path="ffmpeg",
            camera_proxy_snapshot_url="http://10.42.0.2:8767/snapshot.png",
            camera_capture_timeout_seconds=4.0,
            camera_capture_output_dir=None,
        )
        camera = V4L2StillCamera.from_config(cast(TwinrConfig, config))
        proxy_camera = cast(Any, camera)

        proxy_camera._http_client = None
        proxy_camera._urllib_opener = SimpleNamespace()

        with patch.object(
            camera,
            "_fetch_snapshot_payload_urllib",
            side_effect=CameraCaptureTimeoutError("timed out"),
        ):
            with self.assertRaises(CameraCaptureTimeoutError):
                camera.capture_photo()

    def test_from_config_uses_aideck_stream_camera_for_aideck_device_uri(self) -> None:
        config = SimpleNamespace(
            camera_device="aideck://192.168.4.1:5000",
            camera_width=640,
            camera_height=480,
            camera_framerate=30,
            camera_input_format=None,
            camera_ffmpeg_path="ffmpeg",
            camera_proxy_snapshot_url=None,
            camera_capture_timeout_seconds=4.0,
            camera_capture_output_dir=None,
        )

        camera = V4L2StillCamera.from_config(cast(TwinrConfig, config))

        self.assertIsInstance(camera, AIDeckStillCamera)

    def test_from_config_prefers_explicit_aideck_device_over_snapshot_proxy(self) -> None:
        config = SimpleNamespace(
            camera_device="aideck://192.168.4.1:5000",
            camera_width=640,
            camera_height=480,
            camera_framerate=30,
            camera_input_format=None,
            camera_ffmpeg_path="ffmpeg",
            camera_proxy_snapshot_url="http://10.42.0.2:8767/snapshot.png",
            camera_capture_timeout_seconds=4.0,
            camera_capture_output_dir=None,
        )

        camera = V4L2StillCamera.from_config(cast(TwinrConfig, config))

        self.assertIsInstance(camera, AIDeckStillCamera)

    def test_aideck_raw_stream_is_converted_to_png(self) -> None:
        camera = AIDeckStillCamera(device="aideck://192.168.4.1:5000", capture_timeout_seconds=3.0)

        with patch.object(
            camera,
            "_read_frame_with_retry",
            return_value=(4, 4, 1, 0, _RAW_BAYER_BYTES),
        ):
            capture = camera.capture_photo(filename="vision.png")

        self.assertEqual(capture.content_type, "image/png")
        self.assertEqual(capture.filename, "vision.png")
        self.assertEqual(capture.input_format, "aideck-cpx-raw-bayer")
        self.assertEqual(capture.source_device, "aideck://192.168.4.1:5000")
        self.assertEqual(_png_dimensions(capture.data), (4, 4))
        rows = _png_rgb_rows(capture.data)
        self.assertEqual(len(rows), 4)
        self.assertNotEqual(rows[1][3:6], rows[1][6:9])

    def test_aideck_jpeg_stream_passes_through_jpeg_bytes(self) -> None:
        jpeg_bytes = b"\xff\xd8JPEG-DATA\xff\xd9"
        camera = AIDeckStillCamera(device="aideck://192.168.4.1:5000", capture_timeout_seconds=3.0)

        with (
            patch.object(
                camera,
                "_read_frame_with_retry",
                return_value=(324, 244, 1, 1, jpeg_bytes),
            ),
            patch("twinr.hardware.aideck_camera._validate_jpeg", return_value=None),
        ):
            capture = camera.capture_photo(filename="vision.png")

        self.assertEqual(capture.data, jpeg_bytes)
        self.assertEqual(capture.content_type, "image/jpeg")
        self.assertEqual(capture.filename, "vision.jpg")
        self.assertEqual(capture.input_format, "aideck-cpx-jpeg")

    def test_aideck_capture_uses_wifi_manager_when_present(self) -> None:
        events: list[str] = []

        class FakeWifiManager:
            @contextmanager
            def ensure_stream_ready(self, host: str, port: int):
                events.append(f"enter:{host}:{port}")
                try:
                    yield
                finally:
                    events.append("exit")

        camera = AIDeckStillCamera(
            device="aideck://192.168.4.1:5000",
            capture_timeout_seconds=3.0,
            wifi_connection_manager=cast(Any, FakeWifiManager()),
        )

        with patch.object(
            camera,
            "_read_frame",
            side_effect=lambda _deadline: events.append("connect") or (4, 4, 1, 0, _RAW_BAYER_BYTES),
        ):
            capture = camera.capture_photo(filename="vision.jpg")

        self.assertEqual(capture.content_type, "image/png")
        self.assertEqual(events, ["enter:192.168.4.1:5000", "connect", "exit"])

    def test_aideck_retries_once_after_transient_timeout(self) -> None:
        camera = AIDeckStillCamera(device="aideck://192.168.4.1:5000", capture_timeout_seconds=2.0)
        attempts = [
            socket.timeout("timed out"),
            (4, 4, 1, 0, _RAW_BAYER_BYTES),
        ]

        with patch("twinr.hardware.aideck_camera.time.sleep") as sleep_mock:
            with patch.object(camera, "_read_frame", side_effect=attempts):
                capture = camera.capture_photo(filename="vision.jpg")

        self.assertEqual(capture.content_type, "image/png")
        sleep_mock.assert_called_once_with(1.0)

    def test_aideck_timeout_maps_to_camera_timeout(self) -> None:
        camera = AIDeckStillCamera(device="aideck://192.168.4.1:5000", capture_timeout_seconds=2.0)

        with patch(
            "twinr.hardware.aideck_camera.socket.create_connection",
            side_effect=socket.timeout("timed out"),
        ):
            with self.assertRaises(CameraCaptureTimeoutError):
                camera.capture_photo()

    def test_aideck_stalled_stream_maps_to_precise_capture_failure(self) -> None:
        camera = AIDeckStillCamera(device="aideck://192.168.4.1:5000", capture_timeout_seconds=2.0)

        with patch.object(
            camera,
            "_read_frame_with_retry",
            side_effect=AIDeckStreamStalledError(
                camera._stream_stalled_message("stalled before one full frame arrived")
            ),
        ):
            with self.assertRaises(CameraCaptureFailedError) as context:
                camera.capture_photo()

        self.assertIn("stalled before one full frame arrived", str(context.exception))


if __name__ == "__main__":
    unittest.main()
