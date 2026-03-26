"""Capture bounded still frames from a Bitcraze AI-Deck WiFi stream.

This adapter speaks the official CPX-over-TCP framing used by Bitcraze's
AI-Deck WiFi image streamer example. It connects to one ``aideck://host:port``
stream, reads one complete frame, and converts the result into a standard image
payload that Twinr's existing still-camera consumers can forward to higher
runtime layers unchanged.

Two stream modes are supported:

- raw Bayer frames are debayered into RGB and returned as ``image/png``
- JPEG frames are passed through as ``image/jpeg``
"""

from __future__ import annotations

from contextlib import nullcontext
import socket
import struct
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit
import zlib

from twinr.hardware.camera import (
    _DEFAULT_CAPTURE_TIMEOUT_SECONDS,
    CameraCaptureFailedError,
    CameraCaptureTimeoutError,
    CameraConfigurationError,
    CapturedPhoto,
    V4L2StillCamera,
)
from twinr.hardware.aideck_wifi import AIDeckWifiConnectionManager, AIDeckWifiHandoverError

_AIDECK_DEVICE_SCHEME = "aideck"
_AIDECK_DEFAULT_PORT = 5000
_AIDECK_DEFAULT_WIDTH = 324
_AIDECK_DEFAULT_HEIGHT = 244
_AIDECK_HEADER_MAGIC = 0xBC
_AIDECK_RAW_ENCODING = 0
_AIDECK_JPEG_ENCODING = 1
_AIDECK_MAX_FRAME_BYTES = 1_000_000
_AIDECK_MAX_DIMENSION = 2048
_AIDECK_STREAM_INPUT_FORMAT = "aideck-cpx-stream"
_AIDECK_RAW_OUTPUT_FORMAT = "aideck-cpx-raw-bayer"
_AIDECK_JPEG_OUTPUT_FORMAT = "aideck-cpx-jpeg"
_AIDECK_CAPTURE_ATTEMPTS = 2
_AIDECK_CAPTURE_RETRY_DELAY_SECONDS = 1.0
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SOI = b"\xff\xd8"
_JPEG_EOI = b"\xff\xd9"


class AIDeckStreamStalledError(RuntimeError):
    """Raised when the AI-Deck stream accepts TCP but does not deliver frame bytes."""


class AIDeckStillCamera(V4L2StillCamera):
    """Capture one still frame from an AI-Deck WiFi image stream."""

    def __init__(
        self,
        *,
        device: str,
        capture_timeout_seconds: float = _DEFAULT_CAPTURE_TIMEOUT_SECONDS,
        output_root: str | Path | None = None,
        wifi_connection_manager: AIDeckWifiConnectionManager | None = None,
    ) -> None:
        self.device, self.stream_host, self.stream_port = self._parse_device_uri(device)
        self.width = _AIDECK_DEFAULT_WIDTH
        self.height = _AIDECK_DEFAULT_HEIGHT
        self.framerate = 1
        self.ffmpeg_path = "aideck"
        self.input_format = _AIDECK_STREAM_INPUT_FORMAT
        self.capture_timeout_seconds = self._validate_positive_float(
            "capture_timeout_seconds",
            capture_timeout_seconds,
        )
        self.output_root = self._normalize_output_root(output_root)
        self._capture_lock = threading.Lock()
        self._wifi_connection_manager = wifi_connection_manager

    def capture_photo(
        self,
        *,
        output_path: str | Path | None = None,
        filename: str = "camera-capture.png",
    ) -> CapturedPhoto:
        """Capture one AI-Deck frame and return it as a standard photo payload."""

        safe_filename = self._sanitize_filename(filename)
        with self._capture_lock:
            try:
                frame = self._read_frame_with_retry()
            except AIDeckStreamStalledError as exc:
                raise CameraCaptureFailedError(
                    str(exc),
                    user_safe_message=(
                        "The AI camera connected but did not deliver an image. "
                        "Please power-cycle the drone and try again."
                    ),
                ) from exc
            except (socket.timeout, TimeoutError) as exc:
                raise CameraCaptureTimeoutError(
                    (
                        f"AI-Deck stream timed out for {self.device} "
                        f"after {self.capture_timeout_seconds:.1f}s"
                    ),
                    user_safe_message="The AI camera took too long to respond. Please try again.",
                ) from exc
            except OSError as exc:
                raise CameraCaptureFailedError(
                    f"AI-Deck stream failed for {self.device}: {exc}",
                    user_safe_message="The AI camera is currently unavailable. Please try again.",
                ) from exc
            except AIDeckWifiHandoverError as exc:
                raise CameraCaptureFailedError(
                    f"AI-Deck WiFi handover failed for {self.device}: {exc}",
                    user_safe_message="The AI camera network could not be prepared. Please try again.",
                ) from exc
            except (struct.error, ValueError) as exc:
                raise CameraCaptureFailedError(
                    f"AI-Deck stream returned invalid data for {self.device}: {exc}",
                    user_safe_message="The AI camera returned an invalid image. Please try again.",
                ) from exc

        capture = self._capture_from_frame(frame=frame, safe_filename=safe_filename)
        if output_path is not None:
            self._write_output_file(output_path, capture.filename, capture.data)
        return capture

    def _read_frame_with_retry(self) -> tuple[int, int, int, int, bytes]:
        """Read one frame, retrying once after a short transient-capture stall."""

        last_error: Exception | None = None
        for attempt in range(_AIDECK_CAPTURE_ATTEMPTS):
            try:
                handover = (
                    self._wifi_connection_manager.ensure_stream_ready(self.stream_host, self.stream_port)
                    if self._wifi_connection_manager is not None
                    else nullcontext()
                )
                with handover:
                    return self._read_frame()
            except (socket.timeout, TimeoutError, OSError, struct.error, ValueError) as exc:
                last_error = exc
                if attempt + 1 >= _AIDECK_CAPTURE_ATTEMPTS:
                    raise
                time.sleep(_AIDECK_CAPTURE_RETRY_DELAY_SECONDS)
        raise RuntimeError("AI-Deck capture retry state became inconsistent") from last_error

    def _read_frame(self) -> tuple[int, int, int, int, bytes]:
        """Open the stream once, read one complete frame, and return its payload."""

        with socket.create_connection(
            (self.stream_host, self.stream_port),
            timeout=self.capture_timeout_seconds,
        ) as connection:
            connection.settimeout(self.capture_timeout_seconds)
            try:
                packet_length, _routing, _function = self._read_packet_info(connection)
            except (socket.timeout, TimeoutError) as exc:
                raise AIDeckStreamStalledError(self._stream_stalled_message("sent no frame bytes")) from exc
            try:
                header_payload = self._read_exact(connection, packet_length - 2)
                if len(header_payload) != 11:
                    raise ValueError(f"unexpected AI-Deck header length: {len(header_payload)}")
                magic, width, height, depth, encoding, size = struct.unpack("<BHHBBI", header_payload)
                if magic != _AIDECK_HEADER_MAGIC:
                    raise ValueError(f"unexpected AI-Deck header magic: 0x{magic:02x}")
                if width <= 0 or width > _AIDECK_MAX_DIMENSION:
                    raise ValueError(f"unexpected AI-Deck frame width: {width}")
                if height <= 0 or height > _AIDECK_MAX_DIMENSION:
                    raise ValueError(f"unexpected AI-Deck frame height: {height}")
                if size <= 0 or size > _AIDECK_MAX_FRAME_BYTES:
                    raise ValueError(f"unexpected AI-Deck frame size: {size}")

                remaining = size
                payload = bytearray()
                while remaining > 0:
                    packet_length, _dst, _src = self._read_packet_info(connection)
                    chunk = self._read_exact(connection, packet_length - 2)
                    if not chunk:
                        raise ValueError("AI-Deck stream returned an empty image chunk")
                    payload.extend(chunk)
                    remaining -= len(chunk)
                if remaining < 0:
                    raise ValueError("AI-Deck stream exceeded the announced frame size")
                return width, height, depth, encoding, bytes(payload)
            except (socket.timeout, TimeoutError) as exc:
                raise AIDeckStreamStalledError(self._stream_stalled_message("stalled before one full frame arrived")) from exc

    def _stream_stalled_message(self, detail: str) -> str:
        """Describe the proven AI-Deck state where TCP is up but the stream is silent."""

        return (
            f"AI-Deck streamer on {self.device} accepted the TCP connection but {detail} "
            f"within {self.capture_timeout_seconds:.1f}s."
        )

    @staticmethod
    def _read_packet_info(connection: socket.socket) -> tuple[int, int, int]:
        """Read one CPX packet header and validate its declared payload length."""

        raw = AIDeckStillCamera._read_exact(connection, 4)
        packet_length, route_field_1, route_field_2 = struct.unpack("<HBB", raw)
        if packet_length < 2:
            raise ValueError(f"unexpected AI-Deck packet length: {packet_length}")
        return packet_length, route_field_1, route_field_2

    @staticmethod
    def _read_exact(connection: socket.socket, size: int) -> bytes:
        """Read exactly ``size`` bytes from the socket or raise on EOF."""

        if size < 0:
            raise ValueError(f"cannot read a negative byte count: {size}")
        payload = bytearray()
        while len(payload) < size:
            chunk = connection.recv(size - len(payload))
            if not chunk:
                raise ValueError("AI-Deck stream closed before one frame was complete")
            payload.extend(chunk)
        return bytes(payload)

    def _capture_from_frame(
        self,
        *,
        frame: tuple[int, int, int, int, bytes],
        safe_filename: str,
    ) -> CapturedPhoto:
        """Normalize one raw AI-Deck frame into the shared ``CapturedPhoto`` contract."""

        width, height, depth, encoding, payload = frame
        if encoding == _AIDECK_RAW_ENCODING:
            image_bytes = _encode_png_rgb(
                width,
                height,
                _debayer_bayer_bg_to_rgb(width, height, depth, payload),
            )
            filename = _filename_with_extension(safe_filename, ".png")
            return CapturedPhoto(
                data=image_bytes,
                content_type="image/png",
                filename=filename,
                source_device=self.device,
                input_format=_AIDECK_RAW_OUTPUT_FORMAT,
            )
        if encoding == _AIDECK_JPEG_ENCODING:
            if not payload.startswith(_JPEG_SOI) or not payload.endswith(_JPEG_EOI):
                raise ValueError("AI-Deck JPEG frame did not contain valid JPEG markers")
            filename = _filename_with_extension(safe_filename, ".jpg")
            return CapturedPhoto(
                data=payload,
                content_type="image/jpeg",
                filename=filename,
                source_device=self.device,
                input_format=_AIDECK_JPEG_OUTPUT_FORMAT,
            )
        raise ValueError(f"unsupported AI-Deck frame encoding: {encoding}")

    @classmethod
    def _parse_device_uri(cls, device: str) -> tuple[str, str, int]:
        """Validate and normalize one ``aideck://host[:port]`` device URI."""

        normalized = cls._validate_non_empty_text("device", device)
        parsed = urlsplit(normalized)
        if parsed.scheme != _AIDECK_DEVICE_SCHEME:
            raise CameraConfigurationError(f"device must use {_AIDECK_DEVICE_SCHEME}://host[:port]")
        if not parsed.hostname:
            raise CameraConfigurationError("AI-Deck device must include a host")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise CameraConfigurationError("AI-Deck device only supports aideck://host[:port]")
        port = parsed.port or _AIDECK_DEFAULT_PORT
        if port <= 0 or port > 65535:
            raise CameraConfigurationError("AI-Deck device port must stay between 1 and 65535")
        normalized_uri = f"{_AIDECK_DEVICE_SCHEME}://{parsed.hostname}:{port}"
        return normalized_uri, parsed.hostname, port


def _filename_with_extension(filename: str, extension: str) -> str:
    """Return one sanitized filename with the required extension."""

    path = Path(filename)
    if path.suffix.casefold() == extension.casefold():
        return path.name
    return f"{path.stem}{extension}"


def _debayer_bayer_bg_to_rgb(width: int, height: int, depth: int, payload: bytes) -> bytes:
    """Convert one 8-bit Bayer BG frame into packed RGB bytes."""

    if depth != 1:
        raise ValueError(f"unsupported AI-Deck raw depth: {depth}")
    expected_size = width * height
    if len(payload) != expected_size:
        raise ValueError(
            f"AI-Deck raw frame size mismatch: expected {expected_size} bytes, got {len(payload)}"
        )
    raw = payload
    rgb = bytearray(width * height * 3)

    def pixel(x: int, y: int) -> int:
        clamped_x = 0 if x < 0 else width - 1 if x >= width else x
        clamped_y = 0 if y < 0 else height - 1 if y >= height else y
        return raw[clamped_y * width + clamped_x]

    write_index = 0
    for y in range(height):
        even_row = (y % 2) == 0
        for x in range(width):
            even_col = (x % 2) == 0
            center = pixel(x, y)
            if even_row and even_col:
                blue = center
                green = (pixel(x - 1, y) + pixel(x + 1, y) + pixel(x, y - 1) + pixel(x, y + 1)) // 4
                red = (pixel(x - 1, y - 1) + pixel(x + 1, y - 1) + pixel(x - 1, y + 1) + pixel(x + 1, y + 1)) // 4
            elif even_row:
                green = center
                blue = (pixel(x - 1, y) + pixel(x + 1, y)) // 2
                red = (pixel(x, y - 1) + pixel(x, y + 1)) // 2
            elif even_col:
                green = center
                red = (pixel(x - 1, y) + pixel(x + 1, y)) // 2
                blue = (pixel(x, y - 1) + pixel(x, y + 1)) // 2
            else:
                red = center
                green = (pixel(x - 1, y) + pixel(x + 1, y) + pixel(x, y - 1) + pixel(x, y + 1)) // 4
                blue = (pixel(x - 1, y - 1) + pixel(x + 1, y - 1) + pixel(x - 1, y + 1) + pixel(x + 1, y + 1)) // 4
            rgb[write_index : write_index + 3] = bytes((red, green, blue))
            write_index += 3
    return bytes(rgb)


def _encode_png_rgb(width: int, height: int, rgb_bytes: bytes) -> bytes:
    """Encode packed 8-bit RGB bytes into one PNG image without external deps."""

    expected_size = width * height * 3
    if len(rgb_bytes) != expected_size:
        raise ValueError(f"PNG RGB payload size mismatch: expected {expected_size}, got {len(rgb_bytes)}")
    row_bytes = width * 3
    image_rows = bytearray()
    for offset in range(0, len(rgb_bytes), row_bytes):
        image_rows.append(0)
        image_rows.extend(rgb_bytes[offset : offset + row_bytes])
    compressed = zlib.compress(bytes(image_rows), level=6)
    return b"".join(
        (
            _PNG_SIGNATURE,
            _png_chunk(
                b"IHDR",
                struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
            ),
            _png_chunk(b"IDAT", compressed),
            _png_chunk(b"IEND", b""),
        )
    )


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    """Build one PNG chunk with the required CRC trailer."""

    return b"".join(
        (
            struct.pack(">I", len(payload)),
            chunk_type,
            payload,
            struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF),
        )
    )
