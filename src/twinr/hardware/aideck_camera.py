"""Capture bounded still frames from a Bitcraze AI-Deck WiFi stream.

This adapter speaks the official CPX-over-TCP framing used by Bitcraze's
AI-Deck WiFi image streamer example. It connects to one ``aideck://host:port``
stream, reads one complete frame, and converts the result into a standard image
payload that Twinr's existing still-camera consumers can forward to higher
runtime layers unchanged.

Two stream modes are supported:

- raw Bayer frames are debayered into PNG
- JPEG frames are validated and returned as ``image/jpeg``

Deployment note:
- 2026 frontier path on Raspberry Pi 4 is ``numpy`` + ``opencv-python-headless``
  for C-accelerated demosaicing and in-memory PNG/JPEG handling.
- If those packages are unavailable, the adapter falls back to a zero-external-
  dependency path for raw Bayer -> PNG conversion and a standards-aware JPEG
  parser for baseline validation.
"""

# CHANGELOG: 2026-03-28
# BUG-1: Enforced a capture-wide deadline across retries instead of spending the full timeout per attempt.
# BUG-2: Decoded CPX routing headers, skipped non-image packets, and re-synchronized on valid image headers.
# BUG-3: Validated RAW frame headers against width/height/depth before payload decode to stop malformed frames early.
# BUG-4: Moved AIDeckWifiHandoverError handling ahead of generic OSError handling so its specific remediation is preserved.
# BUG-5: Kept frame normalization and output-file writes inside the capture lock to avoid same-camera race conditions.
# SEC-1: Enforced the documented CPX WiFi MTU, version, source, destination, and function checks for incoming packets.
# SEC-2: Replaced JPEG marker-only checks with structural validation plus decode validation when OpenCV/Pillow is available.
# IMP-1: Prefer NumPy + OpenCV edge-aware demosaicing and in-memory PNG encoding on Raspberry Pi.
# IMP-2: Switched socket reads to recv_into-based exact reads and added noise-tolerant header scanning for unstable streams.
# IMP-3: Added optional Pillow fallback for JPEG validation when OpenCV is unavailable.

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from io import BytesIO
import socket
import struct
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit
import zlib

try:  # 2026 frontier path on Pi 4
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore[assignment]

try:  # 2026 frontier path on Pi 4
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    np = None  # type: ignore[assignment]

try:  # optional validation fallback
    from PIL import Image, UnidentifiedImageError  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Image = None  # type: ignore[assignment]
    UnidentifiedImageError = OSError  # type: ignore[assignment]

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
_AIDECK_HEADER_PAYLOAD_BYTES = 11

# CPX/WiFi transport constants from the official Bitcraze implementation.
_AIDECK_CPX_VERSION = 0
_AIDECK_CPX_HEADER_BYTES = 2
_AIDECK_CPX_WIFI_MTU_BYTES = 1022
_AIDECK_CPX_MAX_DATA_BYTES = _AIDECK_CPX_WIFI_MTU_BYTES - _AIDECK_CPX_HEADER_BYTES
_AIDECK_CPX_TARGET_WIFI_HOST = 3
_AIDECK_CPX_TARGET_GAP8 = 4
_AIDECK_CPX_FUNCTION_APP = 5
_AIDECK_MAX_HEADER_SCAN_PACKETS = 64

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SOI = b"\xff\xd8"
_JPEG_EOI = b"\xff\xd9"
_JPEG_SOS = 0xDA
_JPEG_SOF_MARKERS = frozenset(
    {
        0xC0,  # baseline DCT
        0xC1,  # extended sequential DCT
        0xC2,  # progressive DCT
        0xC3,  # lossless sequential
        0xC5,  # differential sequential DCT
        0xC6,  # differential progressive DCT
        0xC7,  # differential lossless
        0xC9,  # extended sequential DCT, arithmetic
        0xCA,  # progressive DCT, arithmetic
        0xCB,  # lossless, arithmetic
        0xCD,  # differential sequential DCT, arithmetic
        0xCE,  # differential progressive DCT, arithmetic
        0xCF,  # differential lossless, arithmetic
    }
)
_JPEG_MARKERS_WITHOUT_LENGTH = frozenset({0x01, 0xD8, 0xD9, *range(0xD0, 0xD8)})


@dataclass(frozen=True, slots=True)
class _CPXRoute:
    destination: int
    source: int
    last_packet: bool
    reserved: bool
    function: int
    version: int


@dataclass(frozen=True, slots=True)
class _CPXPacket:
    route: _CPXRoute
    payload: bytes


@dataclass(frozen=True, slots=True)
class _AIDeckImageHeader:
    width: int
    height: int
    depth: int
    encoding: int
    size: int
    route: _CPXRoute


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
        deadline = time.monotonic() + self.capture_timeout_seconds

        with self._capture_lock:
            try:
                frame = self._read_frame_with_retry(deadline)
            except AIDeckWifiHandoverError as exc:
                raise CameraCaptureFailedError(
                    f"AI-Deck WiFi handover failed for {self.device}: {exc}",
                    user_safe_message="The AI camera network could not be prepared. Please try again.",
                ) from exc
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
            except (struct.error, ValueError) as exc:
                raise CameraCaptureFailedError(
                    f"AI-Deck stream returned invalid data for {self.device}: {exc}",
                    user_safe_message="The AI camera returned an invalid image. Please try again.",
                ) from exc

            capture = self._capture_from_frame(frame=frame, safe_filename=safe_filename)
            if output_path is not None:
                self._write_output_file(output_path, capture.filename, capture.data)
            return capture

    def _read_frame_with_retry(self, deadline: float) -> tuple[int, int, int, int, bytes]:
        """Read one frame, retrying once after a short transient-capture stall."""

        last_error: Exception | None = None
        for attempt in range(_AIDECK_CAPTURE_ATTEMPTS):
            self._ensure_time_remaining(deadline)
            try:
                handover = (
                    self._wifi_connection_manager.ensure_stream_ready(self.stream_host, self.stream_port)
                    if self._wifi_connection_manager is not None
                    else nullcontext()
                )
                with handover:
                    self._ensure_time_remaining(deadline)
                    return self._read_frame(deadline)
            except (AIDeckStreamStalledError, socket.timeout, TimeoutError, OSError, struct.error, ValueError) as exc:
                last_error = exc
                if attempt + 1 >= _AIDECK_CAPTURE_ATTEMPTS:
                    raise
                remaining_after_failure = deadline - time.monotonic()
                if remaining_after_failure <= 0:
                    raise TimeoutError("AI-Deck capture deadline expired before retry") from exc
                time.sleep(min(_AIDECK_CAPTURE_RETRY_DELAY_SECONDS, remaining_after_failure))
        raise RuntimeError("AI-Deck capture retry state became inconsistent") from last_error

    def _read_frame(self, deadline: float) -> tuple[int, int, int, int, bytes]:
        """Open the stream once, read one complete frame, and return its payload."""

        timeout = self._remaining_timeout(deadline)
        with socket.create_connection(
            (self.stream_host, self.stream_port),
            timeout=timeout,
        ) as connection:
            connection.settimeout(timeout)
            try:
                header = self._read_image_header(connection, deadline)
                payload = self._read_image_payload(
                    connection,
                    deadline,
                    header=header,
                )
            except (socket.timeout, TimeoutError) as exc:
                raise AIDeckStreamStalledError(
                    self._stream_stalled_message("stalled before one full frame arrived")
                ) from exc

        self.width = header.width
        self.height = header.height
        return header.width, header.height, header.depth, header.encoding, payload

    def _read_image_header(self, connection: socket.socket, deadline: float) -> _AIDeckImageHeader:
        """Scan the stream until a valid APP image header packet is found."""

        for _ in range(_AIDECK_MAX_HEADER_SCAN_PACKETS):
            packet = self._read_packet(connection, deadline)
            if not self._is_expected_image_packet(packet.route):
                continue
            if len(packet.payload) != _AIDECK_HEADER_PAYLOAD_BYTES:
                continue
            try:
                return self._parse_image_header(packet)
            except ValueError:
                # Resynchronize by scanning forward to the next plausible header packet.
                continue
        raise AIDeckStreamStalledError(
            self._stream_stalled_message("did not produce a valid image header")
        )

    def _read_image_payload(
        self,
        connection: socket.socket,
        deadline: float,
        *,
        header: _AIDeckImageHeader,
    ) -> bytes:
        """Read APP packets until the announced frame payload has been assembled."""

        remaining = header.size
        payload = bytearray()

        while remaining > 0:
            packet = self._read_packet(connection, deadline)
            if not self._is_expected_image_packet(packet.route, expected_version=header.route.version):
                continue
            chunk = packet.payload
            if not chunk:
                raise ValueError("AI-Deck stream returned an empty image chunk")
            if len(chunk) > remaining:
                raise ValueError("AI-Deck stream exceeded the announced frame size")
            payload.extend(chunk)
            remaining -= len(chunk)

        return bytes(payload)

    def _read_packet(self, connection: socket.socket, deadline: float) -> _CPXPacket:
        """Read one CPX packet from the WiFi/TCP transport."""

        packet_length, route = self._read_packet_info(connection, deadline)
        payload = self._read_exact(connection, packet_length - _AIDECK_CPX_HEADER_BYTES, deadline)
        return _CPXPacket(route=route, payload=payload)

    def _stream_stalled_message(self, detail: str) -> str:
        """Describe the proven AI-Deck state where TCP is up but the stream is silent."""

        return (
            f"AI-Deck streamer on {self.device} accepted the TCP connection but {detail} "
            f"within {self.capture_timeout_seconds:.1f}s."
        )

    def _parse_image_header(self, packet: _CPXPacket) -> _AIDeckImageHeader:
        """Validate and unpack one APP image header packet."""

        magic, width, height, depth, encoding, size = struct.unpack("<BHHBBI", packet.payload)
        if magic != _AIDECK_HEADER_MAGIC:
            raise ValueError(f"unexpected AI-Deck header magic: 0x{magic:02x}")
        if width <= 0 or width > _AIDECK_MAX_DIMENSION:
            raise ValueError(f"unexpected AI-Deck frame width: {width}")
        if height <= 0 or height > _AIDECK_MAX_DIMENSION:
            raise ValueError(f"unexpected AI-Deck frame height: {height}")
        if size <= 0 or size > _AIDECK_MAX_FRAME_BYTES:
            raise ValueError(f"unexpected AI-Deck frame size: {size}")
        if encoding not in {_AIDECK_RAW_ENCODING, _AIDECK_JPEG_ENCODING}:
            raise ValueError(f"unsupported AI-Deck frame encoding: {encoding}")
        if encoding == _AIDECK_RAW_ENCODING:
            if depth != 1:
                raise ValueError(f"unsupported AI-Deck raw depth: {depth}")
            expected_size = width * height * depth
            if size != expected_size:
                raise ValueError(
                    f"AI-Deck raw frame size mismatch: expected {expected_size} bytes from header, got {size}"
                )

        return _AIDeckImageHeader(
            width=width,
            height=height,
            depth=depth,
            encoding=encoding,
            size=size,
            route=packet.route,
        )

    @staticmethod
    def _is_expected_image_packet(route: _CPXRoute, *, expected_version: int = _AIDECK_CPX_VERSION) -> bool:
        """Return whether one CPX route matches the documented AI-Deck image stream route."""

        return (
            route.version == expected_version
            and route.function == _AIDECK_CPX_FUNCTION_APP
            and route.source == _AIDECK_CPX_TARGET_GAP8
            and route.destination == _AIDECK_CPX_TARGET_WIFI_HOST
        )

    @staticmethod
    def _read_packet_info(connection: socket.socket, deadline: float) -> tuple[int, _CPXRoute]:
        """Read one CPX WiFi packet header and validate its declared payload length."""

        raw = AIDeckStillCamera._read_exact(connection, 4, deadline)
        packet_length, route_field_1, route_field_2 = struct.unpack("<HBB", raw)
        if packet_length < _AIDECK_CPX_HEADER_BYTES:
            raise ValueError(f"unexpected AI-Deck packet length: {packet_length}")
        if packet_length > _AIDECK_CPX_WIFI_MTU_BYTES:
            raise ValueError(
                f"AI-Deck packet length {packet_length} exceeded the documented WiFi MTU "
                f"{_AIDECK_CPX_WIFI_MTU_BYTES}"
            )

        route = _CPXRoute(
            destination=route_field_1 & 0x07,
            source=(route_field_1 >> 3) & 0x07,
            last_packet=bool((route_field_1 >> 6) & 0x01),
            reserved=bool((route_field_1 >> 7) & 0x01),
            function=route_field_2 & 0x3F,
            version=(route_field_2 >> 6) & 0x03,
        )
        return packet_length, route

    @staticmethod
    def _read_exact(connection: socket.socket, size: int, deadline: float) -> bytes:
        """Read exactly ``size`` bytes from the socket or raise on EOF."""

        if size < 0:
            raise ValueError(f"cannot read a negative byte count: {size}")
        if size == 0:
            return b""

        buffer = bytearray(size)
        view = memoryview(buffer)
        received = 0
        while received < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("AI-Deck stream read exceeded the capture deadline")
            connection.settimeout(remaining)
            chunk_size = connection.recv_into(view[received:], size - received)
            if chunk_size == 0:
                raise ValueError("AI-Deck stream closed before one frame was complete")
            received += chunk_size
        return bytes(buffer)

    @staticmethod
    def _remaining_timeout(deadline: float) -> float:
        """Return remaining time until the capture deadline."""

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("AI-Deck capture deadline expired")
        return remaining

    @staticmethod
    def _ensure_time_remaining(deadline: float) -> None:
        """Raise a timeout when no capture budget remains."""

        if deadline - time.monotonic() <= 0:
            raise TimeoutError("AI-Deck capture deadline expired")

    def _capture_from_frame(
        self,
        *,
        frame: tuple[int, int, int, int, bytes],
        safe_filename: str,
    ) -> CapturedPhoto:
        """Normalize one raw AI-Deck frame into the shared ``CapturedPhoto`` contract."""

        width, height, depth, encoding, payload = frame
        if encoding == _AIDECK_RAW_ENCODING:
            image_bytes = _raw_bayer_bggr_to_png(width, height, depth, payload)
            filename = _filename_with_extension(safe_filename, ".png")
            return CapturedPhoto(
                data=image_bytes,
                content_type="image/png",
                filename=filename,
                source_device=self.device,
                input_format=_AIDECK_RAW_OUTPUT_FORMAT,
            )
        if encoding == _AIDECK_JPEG_ENCODING:
            _validate_jpeg(width, height, payload)
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

        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            normalized_host = f"[{host}]"
        else:
            normalized_host = host
        normalized_uri = f"{_AIDECK_DEVICE_SCHEME}://{normalized_host}:{port}"
        return normalized_uri, host, port


def _filename_with_extension(filename: str, extension: str) -> str:
    """Return one sanitized filename with the required extension."""

    path = Path(filename)
    if path.suffix.casefold() == extension.casefold():
        return path.name
    return f"{path.stem}{extension}"


def _raw_bayer_bggr_to_png(width: int, height: int, depth: int, payload: bytes) -> bytes:
    """Convert one 8-bit Bayer BGGR frame into PNG bytes."""

    if depth != 1:
        raise ValueError(f"unsupported AI-Deck raw depth: {depth}")
    expected_size = width * height
    if len(payload) != expected_size:
        raise ValueError(
            f"AI-Deck raw frame size mismatch: expected {expected_size} bytes, got {len(payload)}"
        )

    image_bytes = _try_encode_png_with_opencv(width, height, payload)
    if image_bytes is not None:
        return image_bytes

    rgb = _debayer_bayer_bggr_to_rgb(width, height, payload)
    return _encode_png_rgb(width, height, rgb)


def _try_encode_png_with_opencv(width: int, height: int, payload: bytes) -> bytes | None:
    """Use OpenCV for fast edge-aware Bayer demosaicing and PNG encoding when available."""

    if cv2 is None or np is None:  # pragma: no cover - dependency dependent
        return None

    bayer = np.frombuffer(payload, dtype=np.uint8).reshape((height, width))
    conversion_code = _select_opencv_bggr_demosaic_code()
    color_bgr = cv2.cvtColor(bayer, conversion_code)
    ok, encoded = cv2.imencode(
        ".png",
        color_bgr,
        [int(cv2.IMWRITE_PNG_COMPRESSION), 1],
    )
    if not ok:
        raise ValueError("OpenCV failed to encode the AI-Deck raw frame as PNG")
    return bytes(encoded)


def _select_opencv_bggr_demosaic_code() -> int:
    """Select the best available OpenCV demosaicing code for BGGR Bayer input."""

    preferred_names = (
        "COLOR_BayerBGGR2BGR_EA",
        "COLOR_BayerRG2BGR_EA",   # alias for BGGR in some builds
        "COLOR_BayerBGGR2BGR",
        "COLOR_BayerRG2BGR",      # alias for BGGR in some builds
    )
    for name in preferred_names:
        code = getattr(cv2, name, None)
        if code is not None:
            return int(code)
    raise ValueError("OpenCV does not expose a supported BGGR Bayer conversion code")


def _debayer_bayer_bg_to_rgb(width: int, height: int, depth: int, payload: bytes) -> bytes:
    """Backwards-compatible wrapper for 8-bit Bayer BGGR -> packed RGB conversion."""

    if depth != 1:
        raise ValueError(f"unsupported AI-Deck raw depth: {depth}")
    expected_size = width * height
    if len(payload) != expected_size:
        raise ValueError(
            f"AI-Deck raw frame size mismatch: expected {expected_size} bytes, got {len(payload)}"
        )
    return _debayer_bayer_bggr_to_rgb(width, height, payload)


def _debayer_bayer_bggr_to_rgb(width: int, height: int, payload: bytes) -> bytes:
    """Convert one 8-bit Bayer BGGR frame into packed RGB bytes."""

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
                red = (
                    pixel(x - 1, y - 1)
                    + pixel(x + 1, y - 1)
                    + pixel(x - 1, y + 1)
                    + pixel(x + 1, y + 1)
                ) // 4
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
                blue = (
                    pixel(x - 1, y - 1)
                    + pixel(x + 1, y - 1)
                    + pixel(x - 1, y + 1)
                    + pixel(x + 1, y + 1)
                ) // 4
            rgb[write_index : write_index + 3] = bytes((red, green, blue))
            write_index += 3
    return bytes(rgb)


def _validate_jpeg(width: int, height: int, payload: bytes) -> None:
    """Perform structural and decode validation for one JPEG payload."""

    if not payload.startswith(_JPEG_SOI) or not payload.endswith(_JPEG_EOI):
        raise ValueError("AI-Deck JPEG frame did not contain valid JPEG boundary markers")

    actual_width, actual_height, _precision = _extract_jpeg_dimensions(payload)
    if actual_width != width or actual_height != height:
        raise ValueError(
            f"AI-Deck JPEG dimensions mismatch: header announced {width}x{height}, "
            f"JPEG contains {actual_width}x{actual_height}"
        )

    if cv2 is not None and np is not None:  # pragma: no cover - dependency dependent
        decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError("AI-Deck JPEG payload could not be decoded by OpenCV")
        return

    if Image is not None:  # pragma: no cover - dependency dependent
        try:
            with Image.open(BytesIO(payload), formats=("JPEG",)) as image:
                image.load()
                if image.size != (width, height):
                    raise ValueError(
                        f"AI-Deck JPEG dimensions mismatch after Pillow decode: "
                        f"expected {width}x{height}, got {image.size[0]}x{image.size[1]}"
                    )
                return
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("AI-Deck JPEG payload could not be decoded by Pillow") from exc


def _extract_jpeg_dimensions(payload: bytes) -> tuple[int, int, int]:
    """Extract JPEG dimensions from the first SOF segment."""

    if len(payload) < 4 or not payload.startswith(_JPEG_SOI):
        raise ValueError("invalid JPEG header")

    index = 2
    length = len(payload)

    while index < length:
        while index < length and payload[index] != 0xFF:
            index += 1
        if index >= length:
            break

        while index < length and payload[index] == 0xFF:
            index += 1
        if index >= length:
            break

        marker = payload[index]
        index += 1

        if marker in _JPEG_MARKERS_WITHOUT_LENGTH:
            if marker == 0xD9:
                break
            continue
        if marker == _JPEG_SOS:
            break
        if index + 2 > length:
            raise ValueError("truncated JPEG segment length")

        segment_length = struct.unpack_from(">H", payload, index)[0]
        if segment_length < 2:
            raise ValueError(f"invalid JPEG segment length: {segment_length}")
        segment_end = index + segment_length
        if segment_end > length:
            raise ValueError("truncated JPEG segment")

        if marker in _JPEG_SOF_MARKERS:
            if segment_length < 7:
                raise ValueError("truncated JPEG SOF segment")
            precision = payload[index + 2]
            height = struct.unpack_from(">H", payload, index + 3)[0]
            width = struct.unpack_from(">H", payload, index + 5)[0]
            if width <= 0 or height <= 0:
                raise ValueError("JPEG contained non-positive dimensions")
            return width, height, precision

        index = segment_end

    raise ValueError("JPEG did not contain a start-of-frame segment")


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

    compressed = zlib.compress(bytes(image_rows), level=1)
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