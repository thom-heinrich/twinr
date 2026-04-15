"""Capture bounded still photos from local devices, Pi camera stacks, AI-Deck streams, or a proxy.

This module wraps ``ffmpeg`` for direct V4L2 capture, supports explicit
``rpicam-still`` capture for Raspberry Pi libcamera stacks through
``rpicam://`` or ``libcamera://`` device URIs, and can also fetch one bounded
still frame from a peer HTTP snapshot proxy or a Bitcraze AI-Deck WiFi stream
while preserving the same upstream ``CapturedPhoto`` contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Iterable, Sequence
import asyncio
import ipaddress
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

try:  # Optional frontier transport; stdlib fallback remains available.
    import httpx  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    httpx = None  # type: ignore[assignment]

from twinr.agent.base_agent.config import TwinrConfig

# CHANGELOG: 2026-03-28
# BUG-1: Removed implicit V4L2-to-rpicam backend switching so still capture now
#        stays on the configured lane and fails closed on lane mismatch.
# BUG-2: Added MJPEG to FFmpeg V4L2 probing for broader USB/UVC compatibility
#        without crossing over into the Pi libcamera backend.
# SEC-1: Bounded HTTP snapshot reads, disabled implicit environment proxies, and disabled redirects for peer snapshot fetches.
# SEC-2: Validated PNG geometry from remote snapshots before accepting payloads, blocking oversized-image and decompression-bomb style memory abuse.
# IMP-1: Added a first-class rpicam:// / libcamera:// backend with metadata-rich still capture via rpicam-still.
# IMP-2: Enriched CapturedPhoto with optional width/height/backend/metadata/timestamp fields while keeping the old constructor call pattern compatible.
# IMP-3: Prefer /dev/shm for temporary Pi camera files to reduce SD-card I/O and latency.
# IMP-4: Fail closed early when a configured `/dev/video*` node is really the Pi libcamera/unicam lane, so Twinr surfaces the explicit `rpicam://` contract instead of burning the full ffmpeg timeout.

_DEFAULT_FFMPEG_FORMAT_CANDIDATES = (None, "mjpeg", "yuyv422", "bayer_grbg8")
_DEFAULT_CAPTURE_TIMEOUT_SECONDS = 10.0
_DEFAULT_CAPTURE_FILENAME = "camera-capture.png"
_DEFAULT_HTTP_CHUNK_SIZE = 64 * 1024
_DEFAULT_HTTP_MAX_RESPONSE_BYTES_CAP = 64 * 1024 * 1024
_DEFAULT_HTTP_MIN_RESPONSE_BYTES_CAP = 1 * 1024 * 1024
_DEFAULT_HTTP_BYTES_PER_PIXEL_CAP = 6
_DEFAULT_HTTP_EXTRA_BYTES_CAP = 256 * 1024
_DEFAULT_HTTP_ERROR_BODY_BYTES = 8 * 1024
_MAX_ERROR_TEXT_LENGTH = 512
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_MIN_HEADER_BYTES = 24
_RPICAM_STILL_BINARY = "rpicam-still"
_RPICAM_STILL_INPUT_FORMAT = "rpicam-still"
_RPICAM_STILL_DEVICE_SCHEME = "rpicam://"
_LIBCAMERA_DEVICE_SCHEME = "libcamera://"
_RPICAM_STILL_MIN_TIMEOUT_MS = 1000
_RPICAM_STILL_MAX_TIMEOUT_MS = 5000
_UNICAM_IMAGE_MARKER = "unicam-image"
_UNICAM_DRIVER_MARKER = "driver name   : unicam"
_V4L2_IDENTITY_TIMEOUT_SECONDS = 1.5
_HTTP_SNAPSHOT_INPUT_FORMAT = "http-snapshot"
_HTTP_ALLOWED_SCHEMES = frozenset({"http", "https"})
_HTTP_LOCAL_HOSTS = frozenset({"localhost"})
_HTTP_LOCAL_HOST_SUFFIXES = (
    ".local",
    ".localdomain",
    ".lan",
    ".home",
    ".internal",
    ".corp",
    ".home.arpa",
    ".fritz.box",
)
_AIDECK_DEVICE_SCHEME = "aideck://"
_DEV_SHM_DIR = Path("/dev/shm")
_FFMPEG_BACKEND_LABEL = "ffmpeg"
_RPICAM_BACKEND_LABEL = "rpicam-still"
_HTTPX_BACKEND_LABEL = "httpx"
_URLLIB_BACKEND_LABEL = "urllib"


class CameraError(RuntimeError):
    """Represent a camera failure that is safe to surface upstream."""

    def __init__(self, message: str, *, user_safe_message: str | None = None) -> None:
        super().__init__(message)
        self.user_safe_message = user_safe_message or "The camera is currently unavailable. Please try again."


class CameraConfigurationError(CameraError):
    """Represent invalid local camera configuration or missing dependencies."""


class CameraCaptureTimeoutError(CameraError):
    """Represent a still capture that exceeded the configured timeout."""


class CameraCaptureFailedError(CameraError):
    """Represent a capture failure after the camera process started."""


class _NoRedirectHandler(HTTPRedirectHandler):
    """Reject redirects for peer snapshot fetches."""

    def redirect_request(self, req, fp, code, msg, hdrs, newurl):  # type: ignore[override]
        raise HTTPError(req.full_url, code, "Redirects are disabled for camera snapshot requests", hdrs, fp)


@dataclass(frozen=True, slots=True)
class CapturedPhoto:
    """Represent one captured still image and its provenance metadata."""

    data: bytes
    content_type: str
    filename: str
    source_device: str
    input_format: str | None = None
    width: int | None = None
    height: int | None = None
    capture_timestamp_ns: int | None = None
    metadata: dict[str, object] | None = None
    backend: str | None = None


class V4L2StillCamera:
    """Capture one still frame from a local camera device.

    The primary path uses bounded ``ffmpeg`` reads from one configured V4L2
    device. Raspberry Pi libcamera stacks use the dedicated
    ``RPiCamStillCamera`` path instead of switching backends implicitly from a
    ``/dev/video*`` node at capture time.
    """

    _device_locks: ClassVar[dict[str, threading.Lock]] = {}
    _device_locks_guard: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        *,
        device: str,
        width: int,
        height: int,
        framerate: int,
        ffmpeg_path: str = "ffmpeg",
        input_format: str | None = None,
        capture_timeout_seconds: float = _DEFAULT_CAPTURE_TIMEOUT_SECONDS,
        output_root: str | Path | None = None,
    ) -> None:
        self.device = self._validate_device(device)
        self.width = self._validate_positive_int("width", width)
        self.height = self._validate_positive_int("height", height)
        self.framerate = self._validate_positive_int("framerate", framerate)
        self.ffmpeg_path = self._validate_non_empty_text("ffmpeg_path", ffmpeg_path)
        self.input_format = self._normalize_input_format(input_format)
        self.capture_timeout_seconds = self._validate_positive_float(
            "capture_timeout_seconds", capture_timeout_seconds
        )
        self.output_root = self._normalize_output_root(output_root)
        self._capture_lock = self._get_device_lock(self.device)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "V4L2StillCamera":
        """Build a still camera from ``TwinrConfig`` values."""

        device = getattr(config, "camera_device", None)
        normalized_device = device.strip() if isinstance(device, str) else device

        if isinstance(normalized_device, str) and normalized_device.lower().startswith(_AIDECK_DEVICE_SCHEME):
            from twinr.hardware.aideck_camera import AIDeckStillCamera
            from twinr.hardware.aideck_wifi import AIDeckWifiConnectionManager

            return AIDeckStillCamera(
                device=normalized_device,
                capture_timeout_seconds=getattr(
                    config, "camera_capture_timeout_seconds", _DEFAULT_CAPTURE_TIMEOUT_SECONDS
                ),
                output_root=getattr(config, "camera_capture_output_dir", None),
                wifi_connection_manager=AIDeckWifiConnectionManager.from_config(config),
            )

        if isinstance(normalized_device, str) and cls._is_rpicam_device_string(normalized_device):
            return RPiCamStillCamera(
                camera_index=cls._parse_rpicam_camera_index(
                    None,
                    device=normalized_device if isinstance(normalized_device, str) else None,
                ),
                width=config.camera_width,
                height=config.camera_height,
                capture_timeout_seconds=getattr(
                    config, "camera_capture_timeout_seconds", _DEFAULT_CAPTURE_TIMEOUT_SECONDS
                ),
                output_root=getattr(config, "camera_capture_output_dir", None),
                device_identifier=normalized_device if isinstance(normalized_device, str) else None,
            )

        snapshot_url = getattr(config, "camera_proxy_snapshot_url", None)
        if isinstance(snapshot_url, str) and snapshot_url.strip():
            return SnapshotProxyStillCamera(
                snapshot_url=snapshot_url,
                width=config.camera_width,
                height=config.camera_height,
                capture_timeout_seconds=getattr(
                    config, "camera_capture_timeout_seconds", _DEFAULT_CAPTURE_TIMEOUT_SECONDS
                ),
                output_root=getattr(config, "camera_capture_output_dir", None),
                max_response_bytes=getattr(config, "camera_proxy_max_response_bytes", None),
                allow_public_hosts=bool(getattr(config, "camera_proxy_allow_public_hosts", False)),
                allow_env_proxies=bool(getattr(config, "camera_proxy_allow_env_proxies", False)),
            )

        return cls(
            device=config.camera_device,
            width=config.camera_width,
            height=config.camera_height,
            framerate=config.camera_framerate,
            ffmpeg_path=config.camera_ffmpeg_path,
            input_format=config.camera_input_format,
            capture_timeout_seconds=getattr(
                config, "camera_capture_timeout_seconds", _DEFAULT_CAPTURE_TIMEOUT_SECONDS
            ),
            output_root=getattr(config, "camera_capture_output_dir", None),
        )

    async def capture_photo_async(
        self,
        *,
        output_path: str | Path | None = None,
        filename: str = _DEFAULT_CAPTURE_FILENAME,
    ) -> CapturedPhoto:
        """Capture one still photo without blocking the caller's event loop."""
        return await asyncio.to_thread(self.capture_photo, output_path=output_path, filename=filename)

    def capture_photo(
        self,
        *,
        output_path: str | Path | None = None,
        filename: str = _DEFAULT_CAPTURE_FILENAME,
    ) -> CapturedPhoto:
        """Capture one still photo and optionally persist it to disk."""

        safe_filename = self._sanitize_filename(filename)
        attempted_formats = tuple(self._candidate_input_formats())
        errors: list[str] = []
        timeout_error: CameraCaptureTimeoutError | None = None
        self._fail_if_pi_libcamera_v4l2_lane_mismatch()
        ffmpeg_binary = shutil.which(self.ffmpeg_path)
        if ffmpeg_binary is None:
            raise CameraConfigurationError(
                (
                    f"ffmpeg was not found on PATH: {self.ffmpeg_path}. "
                    "For Raspberry Pi libcamera cameras, configure an explicit "
                    "rpicam://<index> or libcamera://<index> camera_device instead of a /dev/video* node."
                ),
                user_safe_message="The camera software is not installed correctly.",
            )

        with self._capture_lock:
            for candidate in attempted_formats:
                label = candidate or "default"
                try:
                    result = subprocess.run(
                        self._build_command(ffmpeg_binary, candidate),
                        check=False,
                        capture_output=True,
                        stdin=subprocess.DEVNULL,
                        timeout=self.capture_timeout_seconds,
                    )
                except subprocess.TimeoutExpired:
                    timeout_error = CameraCaptureTimeoutError(
                        (
                            f"Camera capture timed out for {self.device} using format {label} "
                            f"after {self.capture_timeout_seconds:.1f}s"
                        ),
                        user_safe_message="The camera took too long to respond. Please try again.",
                    )
                    errors.append(f"{label}: timeout after {self.capture_timeout_seconds:.1f}s")
                    break
                except OSError as exc:
                    raise CameraCaptureFailedError(
                        f"Unable to execute ffmpeg for camera capture: {exc}",
                        user_safe_message="The camera could not be started. Please try again.",
                    ) from exc

                if result.returncode == 0 and result.stdout and self._is_png_bytes(result.stdout):
                    width, height = self._validate_png_payload(result.stdout, source_label=self.device)
                    capture = CapturedPhoto(
                        data=bytes(result.stdout),
                        content_type="image/png",
                        filename=safe_filename,
                        source_device=self.device,
                        input_format=candidate,
                        width=width,
                        height=height,
                        capture_timestamp_ns=time.time_ns(),
                        metadata=None,
                        backend=_FFMPEG_BACKEND_LABEL,
                    )
                    if output_path is not None:
                        self._write_output_file(output_path, safe_filename, capture.data)
                    return capture

                stderr = self._summarize_process_error(result.stderr)
                if result.returncode == 0 and result.stdout and not self._is_png_bytes(result.stdout):
                    stderr = "ffmpeg returned non-PNG bytes"
                errors.append(f"{label}: {stderr}")
            if timeout_error is not None:
                raise timeout_error

        candidates = ", ".join(fmt or "default" for fmt in attempted_formats)
        raise CameraCaptureFailedError(
            f"Camera capture failed for {self.device} with formats [{candidates}]: {' | '.join(errors)}",
            user_safe_message="The camera could not take a photo right now. Please try again.",
        )

    def _fail_if_pi_libcamera_v4l2_lane_mismatch(self) -> None:
        """Reject Pi libcamera/unicam V4L2 nodes that require the explicit rpicam lane."""

        v4l2_binary = shutil.which("v4l2-ctl")
        if v4l2_binary is None:
            return
        try:
            result = subprocess.run(
                [v4l2_binary, "-d", self.device, "-D"],
                check=False,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                timeout=min(self.capture_timeout_seconds, _V4L2_IDENTITY_TIMEOUT_SECONDS),
            )
        except (OSError, subprocess.SubprocessError):
            return
        probe_text = self._decode_process_output(result.stdout, result.stderr)
        if result.returncode == 0 and self._looks_like_pi_libcamera_v4l2_identity(probe_text):
            raise CameraConfigurationError(
                (
                    f"Camera device {self.device} is backed by the Raspberry Pi libcamera/unicam stack. "
                    "Configure an explicit rpicam://<index> or libcamera://<index> camera_device instead of a /dev/video* node."
                ),
                user_safe_message="The camera is configured on the wrong lane. Please switch Twinr to the Raspberry Pi camera device.",
            )

    def _candidate_input_formats(self) -> Sequence[str | None]:
        return (self.input_format,) if self.input_format else _DEFAULT_FFMPEG_FORMAT_CANDIDATES

    def _build_command(self, ffmpeg_binary: str, input_format: str | None) -> list[str]:
        command = [
            ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "video4linux2",
            "-framerate",
            str(self.framerate),
            "-video_size",
            f"{self.width}x{self.height}",
        ]
        if input_format:
            command.extend(["-input_format", input_format])
        command.extend(["-i", self.device, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"])
        return command

    @staticmethod
    def _decode_process_output(stdout: object, stderr: object) -> str:
        parts: list[str] = []
        for value in (stdout, stderr):
            if isinstance(value, bytes):
                parts.append(value.decode("utf-8", errors="replace"))
            elif isinstance(value, str):
                parts.append(value)
        return "\n".join(part for part in parts if part).lower()

    @staticmethod
    def _looks_like_pi_libcamera_v4l2_identity(probe_text: str) -> bool:
        normalized = str(probe_text or "").strip().lower()
        return _UNICAM_IMAGE_MARKER in normalized or _UNICAM_DRIVER_MARKER in normalized

    def _capture_with_rpicam_still(
        self,
        *,
        safe_filename: str,
        output_path: str | Path | None,
        source_device: str,
        prior_errors: Sequence[str],
    ) -> CapturedPhoto:
        binary = shutil.which(_RPICAM_STILL_BINARY)
        if binary is None:
            raise CameraConfigurationError(
                f"{_RPICAM_STILL_BINARY} was not found on PATH",
                user_safe_message="The camera software is not installed correctly.",
            )

        temp_dir = self._preferred_temp_dir()
        temp_path: str | None = None
        try:
            fd, temp_path = tempfile.mkstemp(prefix=".camera-rpicam-", suffix=".png", dir=str(temp_dir))
            os.close(fd)
            result = subprocess.run(
                self._build_rpicam_command(binary, temp_path),
                check=False,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                timeout=max(self.capture_timeout_seconds, 3.0),
            )
            if result.returncode != 0:
                stderr = self._summarize_process_error(result.stderr)
                prior_text = f"; prior_errors={' | '.join(prior_errors)}" if prior_errors else ""
                raise CameraCaptureFailedError(
                    f"Pi camera capture failed for {source_device}: {stderr}{prior_text}",
                    user_safe_message="The camera could not take a photo right now. Please try again.",
                )

            data = Path(temp_path).read_bytes()
            width, height = self._validate_png_payload(data, source_label=source_device)
            metadata = self._parse_json_metadata(result.stdout)
            capture_timestamp_ns = self._extract_capture_timestamp_ns(metadata) or time.time_ns()
            capture = CapturedPhoto(
                data=data,
                content_type="image/png",
                filename=safe_filename,
                source_device=source_device,
                input_format=_RPICAM_STILL_INPUT_FORMAT,
                width=width,
                height=height,
                capture_timestamp_ns=capture_timestamp_ns,
                metadata=metadata,
                backend=_RPICAM_BACKEND_LABEL,
            )
            if output_path is not None:
                self._write_output_file(output_path, safe_filename, capture.data)
            return capture
        except subprocess.TimeoutExpired as exc:
            raise CameraCaptureTimeoutError(
                f"Pi camera capture timed out for {source_device} after {self.capture_timeout_seconds:.1f}s",
                user_safe_message="The camera took too long to respond. Please try again.",
            ) from exc
        except OSError as exc:
            raise CameraCaptureFailedError(
                f"Pi camera capture could not read captured output for {source_device}: {exc}",
                user_safe_message="The camera could not take a photo right now. Please try again.",
            ) from exc
        finally:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass

    def _build_rpicam_command(self, binary: str, output_path: str) -> list[str]:
        command = [
            binary,
            "-v",
            "0",
            "--nopreview",
            "--immediate",
            "--timeout",
            f"{self._rpicam_still_timeout_ms()}ms",
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            "--encoding",
            "png",
            "--output",
            output_path,
            "--metadata",
            "-",
            "--metadata-format",
            "json",
        ]
        camera_index = self._get_rpicam_camera_index()
        if camera_index is not None:
            command.extend(["--camera", str(camera_index)])
        return command

    def _get_rpicam_camera_index(self) -> int | None:
        return getattr(self, "rpicam_camera_index", None)

    def _preferred_temp_dir(self) -> Path:
        if _DEV_SHM_DIR.is_dir() and os.access(_DEV_SHM_DIR, os.W_OK):
            return _DEV_SHM_DIR
        return Path(tempfile.gettempdir())

    def _rpicam_still_timeout_ms(self) -> int:
        timeout_ms = int(self.capture_timeout_seconds * 1000.0)
        if timeout_ms < _RPICAM_STILL_MIN_TIMEOUT_MS:
            return _RPICAM_STILL_MIN_TIMEOUT_MS
        if timeout_ms > _RPICAM_STILL_MAX_TIMEOUT_MS:
            return _RPICAM_STILL_MAX_TIMEOUT_MS
        return timeout_ms

    @classmethod
    def _get_device_lock(cls, device: str) -> threading.Lock:
        with cls._device_locks_guard:
            lock = cls._device_locks.get(device)
            if lock is None:
                lock = threading.Lock()
                cls._device_locks[device] = lock
            return lock

    def _write_output_file(self, output_path: str | Path, filename: str, data: bytes) -> None:
        target = self._resolve_output_file(output_path, filename)
        parent_dir = target.parent
        temp_path: str | None = None
        try:
            parent_dir.mkdir(parents=True, exist_ok=True)
            if target.is_symlink():
                raise CameraCaptureFailedError(
                    f"Refusing to overwrite symlink with captured image: {target}",
                    user_safe_message="The photo could not be saved safely.",
                )
            if target.exists() and target.is_dir():
                raise CameraCaptureFailedError(
                    f"Refusing to overwrite directory with captured image: {target}",
                    user_safe_message="The photo could not be saved.",
                )

            fd, temp_path = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(parent_dir))
            with os.fdopen(fd, "wb") as temp_file:
                temp_file.write(data)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, target)
            self._sync_directory(parent_dir)
        except CameraError:
            raise
        except OSError as exc:
            raise CameraCaptureFailedError(
                f"Unable to persist captured image to {target}: {exc}",
                user_safe_message="The photo was taken but could not be saved.",
            ) from exc
        finally:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass

    def _resolve_output_file(self, output_path: str | Path, filename: str) -> Path:
        requested_path = Path(output_path).expanduser()
        if requested_path.exists() and requested_path.is_dir():
            requested_path = requested_path / filename

        if requested_path.is_absolute():
            if self.output_root is None:
                raise CameraCaptureFailedError(
                    "Absolute output paths are disabled unless camera_capture_output_dir is configured.",
                    user_safe_message="The photo save location is not allowed.",
                )
            base_dir = self.output_root
            candidate = requested_path
        else:
            base_dir = self.output_root or Path.cwd().resolve(strict=False)
            candidate = base_dir / requested_path

        resolved_target = candidate.resolve(strict=False)
        try:
            resolved_target.relative_to(base_dir)
        except ValueError as exc:
            raise CameraCaptureFailedError(
                f"Refusing to write camera output outside the allowed directory: {resolved_target}",
                user_safe_message="The photo save location is not allowed.",
            ) from exc
        return resolved_target

    @staticmethod
    def _sync_directory(directory: Path) -> None:
        try:
            directory_fd = os.open(str(directory), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        except OSError:
            pass
        finally:
            os.close(directory_fd)

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        candidate = Path(filename or _DEFAULT_CAPTURE_FILENAME).name.strip()
        if not candidate:
            candidate = _DEFAULT_CAPTURE_FILENAME
        stem = Path(candidate).stem or "camera-capture"
        safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
        safe_stem = safe_stem.strip("._-") or "camera-capture"
        return f"{safe_stem}.png"

    @staticmethod
    def _is_png_bytes(data: bytes) -> bool:
        return data.startswith(_PNG_SIGNATURE)

    @classmethod
    def _validate_png_payload(
        cls,
        data: bytes,
        *,
        source_label: str,
        expected_max_width: int | None = None,
        expected_max_height: int | None = None,
        max_pixels: int | None = None,
    ) -> tuple[int, int]:
        width, height = cls._parse_png_size(data)
        if expected_max_width is not None and width > expected_max_width:
            raise CameraCaptureFailedError(
                f"PNG width {width} exceeds configured bound {expected_max_width} for {source_label}",
                user_safe_message="The camera returned an invalid photo. Please try again.",
            )
        if expected_max_height is not None and height > expected_max_height:
            raise CameraCaptureFailedError(
                f"PNG height {height} exceeds configured bound {expected_max_height} for {source_label}",
                user_safe_message="The camera returned an invalid photo. Please try again.",
            )
        if max_pixels is not None and width * height > max_pixels:
            raise CameraCaptureFailedError(
                f"PNG pixels {width * height} exceed configured bound {max_pixels} for {source_label}",
                user_safe_message="The camera returned an invalid photo. Please try again.",
            )
        return width, height

    @classmethod
    def _parse_png_size(cls, data: bytes | bytearray) -> tuple[int, int]:
        if len(data) < _PNG_MIN_HEADER_BYTES or not bytes(data).startswith(_PNG_SIGNATURE):
            raise CameraCaptureFailedError(
                "The camera returned invalid PNG bytes.",
                user_safe_message="The camera returned an invalid photo. Please try again.",
            )
        if data[8:12] != b"\x00\x00\x00\r" or data[12:16] != b"IHDR":
            raise CameraCaptureFailedError(
                "The camera returned a PNG without a valid IHDR header.",
                user_safe_message="The camera returned an invalid photo. Please try again.",
            )
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        if width <= 0 or height <= 0:
            raise CameraCaptureFailedError(
                f"The camera returned an invalid PNG size {width}x{height}.",
                user_safe_message="The camera returned an invalid photo. Please try again.",
            )
        return width, height

    @staticmethod
    def _summarize_process_error(stderr: bytes | None) -> str:
        if not stderr:
            return "empty output"
        decoded = stderr.decode("utf-8", errors="replace")
        cleaned = " ".join("".join(ch if ch.isprintable() else " " for ch in decoded).split())
        if len(cleaned) > _MAX_ERROR_TEXT_LENGTH:
            return f"{cleaned[: _MAX_ERROR_TEXT_LENGTH - 3].rstrip()}..."
        return cleaned or "empty output"

    @classmethod
    def _summarize_exception_message(cls, exc: BaseException) -> str:
        return cls._summarize_process_error(str(exc).encode("utf-8", errors="replace"))

    @staticmethod
    def _validate_non_empty_text(name: str, value: str) -> str:
        if not isinstance(value, str):
            raise CameraConfigurationError(f"{name} must be a string")
        normalized = value.strip()
        if not normalized:
            raise CameraConfigurationError(f"{name} must not be empty")
        if any(ch in normalized for ch in ("\x00", "\r", "\n")):
            raise CameraConfigurationError(f"{name} must not contain control characters")
        return normalized

    @classmethod
    def _validate_device(cls, device: str) -> str:
        normalized = cls._validate_non_empty_text("device", device)
        if not Path(normalized).is_absolute() or not normalized.startswith("/dev/"):
            raise CameraConfigurationError("device must be an absolute V4L2 path under /dev")
        return normalized

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise CameraConfigurationError(f"{name} must be an integer")
        if value <= 0:
            raise CameraConfigurationError(f"{name} must be greater than zero")
        return value

    @staticmethod
    def _validate_positive_float(name: str, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CameraConfigurationError(f"{name} must be a number")
        normalized = float(value)
        if normalized <= 0.0:
            raise CameraConfigurationError(f"{name} must be greater than zero")
        return normalized

    @classmethod
    def _normalize_input_format(cls, input_format: str | None) -> str | None:
        if input_format is None:
            return None
        return cls._validate_non_empty_text("input_format", input_format)

    @staticmethod
    def _normalize_output_root(output_root: str | Path | None) -> Path | None:
        if output_root is None:
            return None
        if isinstance(output_root, Path):
            candidate = output_root
        elif isinstance(output_root, str):
            normalized = output_root.strip()
            if not normalized:
                return None
            candidate = Path(normalized)
        else:
            raise CameraConfigurationError("output_root must be a path-like value")
        return candidate.expanduser().resolve(strict=False)

    @staticmethod
    def _normalize_optional_bool(name: str, value: bool | None) -> bool | None:
        if value is None:
            return None
        if not isinstance(value, bool):
            raise CameraConfigurationError(f"{name} must be a boolean if provided")
        return value

    @staticmethod
    def _normalize_optional_non_negative_int(name: str, value: int | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise CameraConfigurationError(f"{name} must be an integer if provided")
        if value < 0:
            raise CameraConfigurationError(f"{name} must not be negative")
        return value

    @classmethod
    def _is_rpicam_device_string(cls, device: str) -> bool:
        normalized = cls._validate_non_empty_text("device", device).casefold()
        return normalized.startswith(_RPICAM_STILL_DEVICE_SCHEME) or normalized.startswith(_LIBCAMERA_DEVICE_SCHEME)

    @classmethod
    def _parse_rpicam_camera_index(cls, value: object, *, device: str | None = None) -> int:
        candidate = value
        if candidate is None and device is not None and cls._is_rpicam_device_string(device):
            candidate = device.split("://", 1)[1]
        if candidate is None:
            return 0
        if isinstance(candidate, bool):
            raise CameraConfigurationError("camera_rpicam_index must be an integer")
        if isinstance(candidate, int):
            if candidate < 0:
                raise CameraConfigurationError("camera_rpicam_index must not be negative")
            return candidate
        if isinstance(candidate, str):
            normalized = candidate.strip()
            if not normalized:
                return 0
            if not normalized.isdecimal():
                raise CameraConfigurationError("camera_rpicam_index must be a non-negative integer")
            return int(normalized)
        raise CameraConfigurationError("camera_rpicam_index must be an integer")

    @staticmethod
    def _parse_json_metadata(stdout: bytes | None) -> dict[str, object] | None:
        if not stdout:
            return None
        text = stdout.decode("utf-8", errors="replace").strip()
        if not text:
            return None
        candidate_texts = [text]
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            candidate_texts.append(text[start : end + 1])
        for candidate in candidate_texts:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        return {"raw_metadata_text": text[:_MAX_ERROR_TEXT_LENGTH]}

    @staticmethod
    def _extract_capture_timestamp_ns(metadata: dict[str, object] | None) -> int | None:
        if not metadata:
            return None
        for key in (
            "SensorTimestamp",
            "SensorTimestampNs",
            "sensor_timestamp",
            "sensor_timestamp_ns",
            "timestamp_ns",
            "TimestampNs",
        ):
            value = metadata.get(key)
            if isinstance(value, int) and value > 0:
                return value
            if isinstance(value, str) and value.isdecimal():
                return int(value)
        return None


class RPiCamStillCamera(V4L2StillCamera):
    """Capture one still frame directly from the Raspberry Pi libcamera stack."""

    def __init__(
        self,
        *,
        camera_index: int,
        width: int,
        height: int,
        capture_timeout_seconds: float = _DEFAULT_CAPTURE_TIMEOUT_SECONDS,
        output_root: str | Path | None = None,
        device_identifier: str | None = None,
    ) -> None:
        self.camera_index = self._parse_rpicam_camera_index(camera_index)
        self.device = (
            self._validate_non_empty_text("device_identifier", device_identifier)
            if isinstance(device_identifier, str) and device_identifier.strip()
            else f"{_RPICAM_STILL_DEVICE_SCHEME}{self.camera_index}"
        )
        self.width = self._validate_positive_int("width", width)
        self.height = self._validate_positive_int("height", height)
        self.framerate = 1
        self.ffmpeg_path = _RPICAM_STILL_BINARY
        self.input_format = _RPICAM_STILL_INPUT_FORMAT
        self.capture_timeout_seconds = self._validate_positive_float(
            "capture_timeout_seconds", capture_timeout_seconds
        )
        self.output_root = self._normalize_output_root(output_root)
        self.rpicam_camera_index = self.camera_index
        self._capture_lock = self._get_device_lock(self.device)

    def capture_photo(
        self,
        *,
        output_path: str | Path | None = None,
        filename: str = _DEFAULT_CAPTURE_FILENAME,
    ) -> CapturedPhoto:
        safe_filename = self._sanitize_filename(filename)
        with self._capture_lock:
            return self._capture_with_rpicam_still(
                safe_filename=safe_filename,
                output_path=output_path,
                source_device=self.device,
                prior_errors=(),
            )


class SnapshotProxyStillCamera(V4L2StillCamera):
    """Fetch bounded still images from a peer HTTP snapshot proxy."""

    def __init__(
        self,
        *,
        snapshot_url: str,
        width: int,
        height: int,
        capture_timeout_seconds: float = _DEFAULT_CAPTURE_TIMEOUT_SECONDS,
        output_root: str | Path | None = None,
        max_response_bytes: int | None = None,
        allow_public_hosts: bool = False,
        allow_env_proxies: bool = False,
    ) -> None:
        self.allow_public_hosts = bool(allow_public_hosts)
        self.allow_env_proxies = bool(allow_env_proxies)
        self.snapshot_url = self._validate_snapshot_url(
            snapshot_url,
            allow_public_hosts=self.allow_public_hosts,
        )
        self.device = self.snapshot_url
        self.width = self._validate_positive_int("width", width)
        self.height = self._validate_positive_int("height", height)
        self.framerate = 1
        self.ffmpeg_path = "http"
        self.input_format = _HTTP_SNAPSHOT_INPUT_FORMAT
        self.capture_timeout_seconds = self._validate_positive_float(
            "capture_timeout_seconds", capture_timeout_seconds
        )
        self.output_root = self._normalize_output_root(output_root)
        self.max_response_bytes = self._normalize_max_response_bytes(max_response_bytes)
        self._capture_lock = threading.Lock()
        self._http_client = self._build_httpx_client() if httpx is not None else None
        self._urllib_opener = None if self._http_client is not None else self._build_urllib_opener()

    def close(self) -> None:
        client = getattr(self, "_http_client", None)
        if client is not None:
            client.close()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def capture_photo(
        self,
        *,
        output_path: str | Path | None = None,
        filename: str = _DEFAULT_CAPTURE_FILENAME,
    ) -> CapturedPhoto:
        safe_filename = self._sanitize_filename(filename)
        request_url = self._build_snapshot_request_url()
        headers = {
            "Accept": "image/png",
            "Accept-Encoding": "identity",
            "User-Agent": "twinr-peer-camera-client/2",
        }

        with self._capture_lock:
            if self._http_client is not None:
                payload, raw_content_type = self._fetch_snapshot_payload_httpx(request_url, headers)
                backend = _HTTPX_BACKEND_LABEL
            else:
                payload, raw_content_type = self._fetch_snapshot_payload_urllib(request_url, headers)
                backend = _URLLIB_BACKEND_LABEL

        width, height = self._validate_png_payload(
            payload,
            source_label=self.snapshot_url,
            expected_max_width=self.width,
            expected_max_height=self.height,
            max_pixels=self.width * self.height,
        )
        metadata: dict[str, object] = {
            "http_content_type": self._normalize_content_type(raw_content_type),
            "backend": backend,
        }
        capture = CapturedPhoto(
            data=payload,
            content_type="image/png",
            filename=safe_filename,
            source_device=self.snapshot_url,
            input_format=_HTTP_SNAPSHOT_INPUT_FORMAT,
            width=width,
            height=height,
            capture_timestamp_ns=time.time_ns(),
            metadata=metadata,
            backend=backend,
        )
        if output_path is not None:
            self._write_output_file(output_path, safe_filename, capture.data)
        return capture

    def _fetch_snapshot_payload_httpx(self, request_url: str, headers: dict[str, str]) -> tuple[bytes, str]:
        assert self._http_client is not None
        deadline = time.monotonic() + self.capture_timeout_seconds
        try:
            with self._http_client.stream("GET", request_url, headers=headers) as response:
                if 300 <= response.status_code < 400:
                    raise CameraCaptureFailedError(
                        f"Camera snapshot proxy attempted redirect for {self.snapshot_url}; redirects are disabled.",
                        user_safe_message="The proxy camera returned an unexpected response. Please try again.",
                    )
                if response.status_code >= 400:
                    detail = self._read_error_text_from_chunks(response.iter_bytes(), deadline)
                    raise CameraCaptureFailedError(
                        f"Camera snapshot proxy returned HTTP {response.status_code} for {self.snapshot_url}: {detail}",
                        user_safe_message="The proxy camera could not take a photo right now. Please try again.",
                    )
                content_length = self._parse_content_length(response.headers.get("Content-Length"))
                payload = self._read_png_bytes_from_chunks(
                    response.iter_bytes(),
                    deadline=deadline,
                    source_label=self.snapshot_url,
                    content_length=content_length,
                )
                return payload, response.headers.get("Content-Type", "")
        except CameraError:
            raise
        except httpx.TimeoutException as exc:  # type: ignore[union-attr]
            raise CameraCaptureTimeoutError(
                f"Camera snapshot proxy timed out for {self.snapshot_url} after {self.capture_timeout_seconds:.1f}s",
                user_safe_message="The proxy camera took too long to respond. Please try again.",
            ) from exc
        except httpx.RequestError as exc:  # type: ignore[union-attr]
            raise CameraCaptureFailedError(
                f"Camera snapshot proxy request failed for {self.snapshot_url}: {self._summarize_exception_message(exc)}",
                user_safe_message="The proxy camera is currently unavailable. Please try again.",
            ) from exc

    def _fetch_snapshot_payload_urllib(self, request_url: str, headers: dict[str, str]) -> tuple[bytes, str]:
        assert self._urllib_opener is not None
        request = Request(request_url, headers=headers, method="GET")
        deadline = time.monotonic() + self.capture_timeout_seconds
        try:
            with self._urllib_opener.open(request, timeout=self.capture_timeout_seconds) as response:
                content_length = self._parse_content_length(response.headers.get("Content-Length"))
                payload = self._read_png_bytes_from_chunks(
                    self._iter_response_chunks(response),
                    deadline=deadline,
                    source_label=self.snapshot_url,
                    content_length=content_length,
                )
                return payload, response.headers.get("Content-Type", "")
        except HTTPError as exc:
            detail = self._summarize_process_error(exc.read(_DEFAULT_HTTP_ERROR_BODY_BYTES))
            if 300 <= exc.code < 400:
                raise CameraCaptureFailedError(
                    f"Camera snapshot proxy attempted redirect for {self.snapshot_url}; redirects are disabled.",
                    user_safe_message="The proxy camera returned an unexpected response. Please try again.",
                ) from exc
            raise CameraCaptureFailedError(
                f"Camera snapshot proxy returned HTTP {exc.code} for {self.snapshot_url}: {detail}",
                user_safe_message="The proxy camera could not take a photo right now. Please try again.",
            ) from exc
        except URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, (TimeoutError, socket.timeout)):
                raise CameraCaptureTimeoutError(
                    f"Camera snapshot proxy timed out for {self.snapshot_url} after {self.capture_timeout_seconds:.1f}s",
                    user_safe_message="The proxy camera took too long to respond. Please try again.",
                ) from exc
            raise CameraCaptureFailedError(
                f"Camera snapshot proxy request failed for {self.snapshot_url}: {self._summarize_reason(reason)}",
                user_safe_message="The proxy camera is currently unavailable. Please try again.",
            ) from exc
        except TimeoutError as exc:
            raise CameraCaptureTimeoutError(
                f"Camera snapshot proxy timed out for {self.snapshot_url} after {self.capture_timeout_seconds:.1f}s",
                user_safe_message="The proxy camera took too long to respond. Please try again.",
            ) from exc
        except OSError as exc:
            raise CameraCaptureFailedError(
                f"Camera snapshot proxy request failed for {self.snapshot_url}: {exc}",
                user_safe_message="The proxy camera is currently unavailable. Please try again.",
            ) from exc

    def _build_snapshot_request_url(self) -> str:
        parsed = urlsplit(self.snapshot_url)
        params = [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=False)
            if key not in {"width", "height", "timeout_ms"}
        ]
        params.extend(
            (
                ("width", str(self.width)),
                ("height", str(self.height)),
                ("timeout_ms", str(int(self.capture_timeout_seconds * 1000.0))),
            )
        )
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(params), ""))

    def _build_httpx_client(self):
        if httpx is None:
            return None
        timeout = httpx.Timeout(  # type: ignore[union-attr]
            connect=self.capture_timeout_seconds,
            read=self.capture_timeout_seconds,
            write=self.capture_timeout_seconds,
            pool=self.capture_timeout_seconds,
        )
        return httpx.Client(  # type: ignore[union-attr]
            timeout=timeout,
            follow_redirects=False,
            trust_env=self.allow_env_proxies,
            headers={"Accept-Encoding": "identity"},
        )

    def _build_urllib_opener(self):
        proxy_handler = ProxyHandler(None if self.allow_env_proxies else {})
        return build_opener(proxy_handler, _NoRedirectHandler())

    def _normalize_max_response_bytes(self, max_response_bytes: int | None) -> int:
        if max_response_bytes is None:
            dynamic_cap = self.width * self.height * _DEFAULT_HTTP_BYTES_PER_PIXEL_CAP + _DEFAULT_HTTP_EXTRA_BYTES_CAP
            return max(_DEFAULT_HTTP_MIN_RESPONSE_BYTES_CAP, min(_DEFAULT_HTTP_MAX_RESPONSE_BYTES_CAP, dynamic_cap))
        return self._validate_positive_int("max_response_bytes", max_response_bytes)

    def _read_png_bytes_from_chunks(
        self,
        chunks: Iterable[bytes],
        *,
        deadline: float,
        source_label: str,
        content_length: int | None,
    ) -> bytes:
        if content_length is not None and content_length > self.max_response_bytes:
            raise CameraCaptureFailedError(
                f"Camera snapshot payload for {source_label} exceeded the configured size limit: {content_length} > {self.max_response_bytes}",
                user_safe_message="The proxy camera returned an invalid photo. Please try again.",
            )

        payload = bytearray()
        checked_header = False
        for chunk in chunks:
            if time.monotonic() > deadline:
                raise CameraCaptureTimeoutError(
                    f"Camera snapshot proxy timed out for {source_label} after {self.capture_timeout_seconds:.1f}s",
                    user_safe_message="The proxy camera took too long to respond. Please try again.",
                )
            if not chunk:
                continue
            payload.extend(chunk)
            if len(payload) > self.max_response_bytes:
                raise CameraCaptureFailedError(
                    f"Camera snapshot payload for {source_label} exceeded the configured size limit of {self.max_response_bytes} bytes",
                    user_safe_message="The proxy camera returned an invalid photo. Please try again.",
                )
            if not checked_header and len(payload) >= _PNG_MIN_HEADER_BYTES:
                width, height = self._parse_png_size(payload)
                self._validate_png_payload(
                    bytes(payload[:_PNG_MIN_HEADER_BYTES]),
                    source_label=source_label,
                    expected_max_width=self.width,
                    expected_max_height=self.height,
                    max_pixels=self.width * self.height,
                )
                checked_header = True
                if width > self.width or height > self.height:
                    raise CameraCaptureFailedError(
                        f"Camera snapshot PNG dimensions {width}x{height} exceed requested bounds {self.width}x{self.height} for {source_label}",
                        user_safe_message="The proxy camera returned an invalid photo. Please try again.",
                    )
        if not payload:
            raise CameraCaptureFailedError(
                f"Camera snapshot proxy returned an empty response for {source_label}",
                user_safe_message="The proxy camera returned an invalid photo. Please try again.",
            )
        return bytes(payload)

    def _read_error_text_from_chunks(self, chunks: Iterable[bytes], deadline: float) -> str:
        buffer = bytearray()
        for chunk in chunks:
            if time.monotonic() > deadline or len(buffer) >= _DEFAULT_HTTP_ERROR_BODY_BYTES:
                break
            if not chunk:
                continue
            remaining = _DEFAULT_HTTP_ERROR_BODY_BYTES - len(buffer)
            buffer.extend(chunk[:remaining])
        return self._summarize_process_error(bytes(buffer))

    @staticmethod
    def _iter_response_chunks(response) -> Iterable[bytes]:
        while True:
            chunk = response.read(_DEFAULT_HTTP_CHUNK_SIZE)
            if not chunk:
                break
            yield chunk

    @classmethod
    def _validate_snapshot_url(cls, snapshot_url: str, *, allow_public_hosts: bool) -> str:
        normalized = cls._validate_non_empty_text("snapshot_url", snapshot_url)
        parsed = urlsplit(normalized)
        if parsed.scheme not in _HTTP_ALLOWED_SCHEMES:
            raise CameraConfigurationError("snapshot_url must use http or https")
        if not parsed.netloc or not parsed.hostname:
            raise CameraConfigurationError("snapshot_url must include a host")
        if parsed.username or parsed.password:
            raise CameraConfigurationError("snapshot_url must not include embedded credentials")
        if parsed.fragment:
            raise CameraConfigurationError("snapshot_url must not include a fragment")
        # BREAKING: public snapshot hosts now require explicit opt-in because peer camera snapshots should stay on private links by default.
        if not allow_public_hosts and not cls._is_local_snapshot_host(parsed.hostname):
            raise CameraConfigurationError(
                "snapshot_url must target a local/private host unless camera_proxy_allow_public_hosts is enabled"
            )
        return normalized

    @staticmethod
    def _is_local_snapshot_host(hostname: str) -> bool:
        host = hostname.strip().rstrip(".").casefold()
        if not host:
            return False
        if host in _HTTP_LOCAL_HOSTS:
            return True
        if any(host.endswith(suffix) for suffix in _HTTP_LOCAL_HOST_SUFFIXES):
            return True
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return "." not in host
        return address.is_private or address.is_loopback or address.is_link_local or address.is_reserved

    @staticmethod
    def _parse_content_length(content_length: str | None) -> int | None:
        if content_length is None:
            return None
        normalized = content_length.strip()
        if not normalized or not normalized.isdecimal():
            return None
        return int(normalized)

    @staticmethod
    def _normalize_content_type(content_type: str) -> str:
        return content_type.split(";", 1)[0].strip().casefold()

    @classmethod
    def _summarize_reason(cls, reason: object) -> str:
        if isinstance(reason, bytes):
            return cls._summarize_process_error(reason)
        return cls._summarize_process_error(str(reason).encode("utf-8", errors="replace"))
