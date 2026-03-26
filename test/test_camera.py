from contextlib import contextmanager
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import patch
from urllib.error import URLError
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.aideck_camera import AIDeckStillCamera
from twinr.hardware.camera import CameraCaptureFailedError, CameraCaptureTimeoutError, V4L2StillCamera

_PNG_BYTES = b"\x89PNG\r\n\x1a\nPNGDATA"


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
        self.assertEqual(capture.input_format, "yuyv422")
        self.assertEqual(run_mock.call_count, 2)

    def test_capture_photo_uses_rpicam_still_fallback_for_unicam_busy_device(self) -> None:
        camera = V4L2StillCamera(
            device="/dev/video0",
            width=640,
            height=480,
            framerate=30,
        )

        def run_side_effect(command: list[str], **_kwargs):
            if command[0] == "/usr/bin/ffmpeg":
                return SimpleNamespace(returncode=1, stdout=b"", stderr=b"Device or resource busy")
            if command[0] == "/usr/bin/rpicam-still":
                output_path = Path(command[command.index("--output") + 1])
                output_path.write_bytes(_PNG_BYTES)
                return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
            raise AssertionError(command)

        with patch("twinr.hardware.camera.shutil.which", side_effect=self._which):
            with patch.object(camera, "_device_sysfs_name", return_value="unicam-image"):
                with patch("twinr.hardware.camera.subprocess.run", side_effect=run_side_effect) as run_mock:
                    capture = camera.capture_photo()

        self.assertEqual(capture.data, _PNG_BYTES)
        self.assertEqual(capture.input_format, "rpicam-still")
        self.assertEqual(capture.source_device, "/dev/video0")
        self.assertEqual(run_mock.call_count, 4)
        self.assertEqual(run_mock.call_args.args[0][0], "/usr/bin/rpicam-still")

    def test_capture_photo_uses_rpicam_still_fallback_when_default_video0_is_missing(self) -> None:
        camera = V4L2StillCamera(
            device="/dev/video0",
            width=640,
            height=480,
            framerate=30,
        )

        def run_side_effect(command: list[str], **_kwargs):
            if command[0] == "/usr/bin/ffmpeg":
                return SimpleNamespace(
                    returncode=1,
                    stdout=b"",
                    stderr=b"Cannot open video device /dev/video0: No such file or directory",
                )
            if command[0] == "/usr/bin/rpicam-still":
                output_path = Path(command[command.index("--output") + 1])
                output_path.write_bytes(_PNG_BYTES)
                return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
            raise AssertionError(command)

        with patch("twinr.hardware.camera.shutil.which", side_effect=self._which):
            with patch.object(camera, "_device_sysfs_name", return_value=""):
                with patch("twinr.hardware.camera.subprocess.run", side_effect=run_side_effect) as run_mock:
                    capture = camera.capture_photo()

        self.assertEqual(capture.data, _PNG_BYTES)
        self.assertEqual(capture.input_format, "rpicam-still")
        self.assertEqual(capture.source_device, "/dev/video0")
        self.assertEqual(run_mock.call_count, 4)
        self.assertEqual(run_mock.call_args.args[0][0], "/usr/bin/rpicam-still")

    def test_capture_photo_raises_timeout_when_busy_fallback_is_not_available(self) -> None:
        camera = V4L2StillCamera(
            device="/dev/video0",
            width=640,
            height=480,
            framerate=30,
        )

        with patch("twinr.hardware.camera.shutil.which", side_effect=lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None):
            with patch.object(camera, "_device_sysfs_name", return_value="unicam-image"):
                with patch(
                    "twinr.hardware.camera.subprocess.run",
                    side_effect=subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=10.0),
                ):
                    with self.assertRaises(CameraCaptureTimeoutError):
                        camera.capture_photo()

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
            output_path = Path(temp_dir) / "proxy-capture.png"

            with patch(
                "twinr.hardware.camera.urlopen",
                return_value=_FakeUrlopenResponse(_PNG_BYTES),
            ) as urlopen_mock:
                capture = camera.capture_photo(output_path=output_path, filename="proxy-capture.png")
                written_bytes = output_path.read_bytes()

        self.assertEqual(capture.data, _PNG_BYTES)
        self.assertEqual(capture.input_format, "http-snapshot")
        self.assertEqual(capture.source_device, "http://10.42.0.2:8767/snapshot.png")
        self.assertEqual(written_bytes, _PNG_BYTES)
        request = urlopen_mock.call_args.args[0]
        self.assertIn("width=640", request.full_url)
        self.assertIn("height=480", request.full_url)
        self.assertIn("timeout_ms=9000", request.full_url)
        self.assertEqual(urlopen_mock.call_args.kwargs["timeout"], 9.0)

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

        with patch(
            "twinr.hardware.camera.urlopen",
            side_effect=URLError(socket.timeout("timed out")),
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
        raw_bytes = bytes(
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
        camera = AIDeckStillCamera(device="aideck://192.168.4.1:5000", capture_timeout_seconds=3.0)

        with patch(
            "twinr.hardware.aideck_camera.socket.create_connection",
            return_value=_FakeSocket(
                _pack_aideck_stream(
                    width=4,
                    height=4,
                    depth=1,
                    encoding=0,
                    image_bytes=raw_bytes,
                )
            ),
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

        with patch(
            "twinr.hardware.aideck_camera.socket.create_connection",
            return_value=_FakeSocket(
                _pack_aideck_stream(
                    width=324,
                    height=244,
                    depth=1,
                    encoding=1,
                    image_bytes=jpeg_bytes,
                )
            ),
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

        def _connect(*_args, **_kwargs):
            events.append("connect")
            return _FakeSocket(
                _pack_aideck_stream(
                    width=324,
                    height=244,
                    depth=1,
                    encoding=1,
                    image_bytes=b"\xff\xd8JPEG-DATA\xff\xd9",
                )
            )

        with patch("twinr.hardware.aideck_camera.socket.create_connection", side_effect=_connect):
            capture = camera.capture_photo(filename="vision.jpg")

        self.assertEqual(capture.content_type, "image/jpeg")
        self.assertEqual(events, ["enter:192.168.4.1:5000", "connect", "exit"])

    def test_aideck_retries_once_after_transient_timeout(self) -> None:
        camera = AIDeckStillCamera(device="aideck://192.168.4.1:5000", capture_timeout_seconds=2.0)
        attempts = [
            socket.timeout("timed out"),
            _FakeSocket(
                _pack_aideck_stream(
                    width=324,
                    height=244,
                    depth=1,
                    encoding=1,
                    image_bytes=b"\xff\xd8JPEG-DATA\xff\xd9",
                )
            ),
        ]

        with patch("twinr.hardware.aideck_camera.time.sleep") as sleep_mock:
            with patch("twinr.hardware.aideck_camera.socket.create_connection", side_effect=attempts):
                capture = camera.capture_photo(filename="vision.jpg")

        self.assertEqual(capture.content_type, "image/jpeg")
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

        with patch(
            "twinr.hardware.aideck_camera.socket.create_connection",
            return_value=_TimeoutOnRecvSocket(),
        ):
            with self.assertRaises(CameraCaptureFailedError) as context:
                camera.capture_photo()

        self.assertIn("accepted the TCP connection but sent no frame bytes", str(context.exception))


if __name__ == "__main__":
    unittest.main()
