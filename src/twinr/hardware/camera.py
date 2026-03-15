from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Sequence
import asyncio
import os
import shutil
import subprocess
import tempfile
import threading

from twinr.agent.base_agent.config import TwinrConfig

_DEFAULT_FFMPEG_FORMAT_CANDIDATES = (None, "yuyv422", "bayer_grbg8")
_DEFAULT_CAPTURE_TIMEOUT_SECONDS = 10.0
_DEFAULT_CAPTURE_FILENAME = "camera-capture.png"
_MAX_ERROR_TEXT_LENGTH = 512
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class CameraError(RuntimeError):  # AUDIT-FIX(#6): Use structured camera exceptions so callers can recover differently from config, timeout, and capture failures.
    def __init__(self, message: str, *, user_safe_message: str | None = None) -> None:
        super().__init__(message)
        self.user_safe_message = user_safe_message or "The camera is currently unavailable. Please try again."


class CameraConfigurationError(CameraError):
    pass


class CameraCaptureTimeoutError(CameraError):
    pass


class CameraCaptureFailedError(CameraError):
    pass


@dataclass(frozen=True, slots=True)
class CapturedPhoto:
    data: bytes
    content_type: str
    filename: str
    source_device: str
    input_format: str | None = None


class V4L2StillCamera:
    _device_locks: ClassVar[dict[str, threading.Lock]] = {}  # AUDIT-FIX(#2): Serialize access per device across all instances in this process to avoid V4L2 EBUSY races.
    _device_locks_guard: ClassVar[threading.Lock] = threading.Lock()  # AUDIT-FIX(#2): Protect lock creation so concurrent constructors do not race and create duplicate locks.

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
        self.device = self._validate_device(device)  # AUDIT-FIX(#4): Reject invalid or surprising device identifiers up front.
        self.width = self._validate_positive_int("width", width)  # AUDIT-FIX(#4): Fail fast on invalid dimensions.
        self.height = self._validate_positive_int("height", height)  # AUDIT-FIX(#4): Fail fast on invalid dimensions.
        self.framerate = self._validate_positive_int("framerate", framerate)  # AUDIT-FIX(#4): Reject invalid frame rates.
        self.ffmpeg_path = self._validate_non_empty_text("ffmpeg_path", ffmpeg_path)  # AUDIT-FIX(#4): Prevent empty executable paths.
        self.input_format = self._normalize_input_format(input_format)  # AUDIT-FIX(#4): Normalize optional format strings safely.
        self.capture_timeout_seconds = self._validate_positive_float(
            "capture_timeout_seconds", capture_timeout_seconds
        )  # AUDIT-FIX(#1): Bound camera calls so ffmpeg cannot hang forever.
        self.output_root = self._normalize_output_root(output_root)  # AUDIT-FIX(#3): Support path confinement for written captures.
        self._capture_lock = self._get_device_lock(self.device)  # AUDIT-FIX(#2): Reuse the per-device lock for this camera instance.

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "V4L2StillCamera":
        return cls(
            device=config.camera_device,
            width=config.camera_width,
            height=config.camera_height,
            framerate=config.camera_framerate,
            ffmpeg_path=config.camera_ffmpeg_path,
            input_format=config.camera_input_format,
            capture_timeout_seconds=getattr(
                config, "camera_capture_timeout_seconds", _DEFAULT_CAPTURE_TIMEOUT_SECONDS
            ),  # AUDIT-FIX(#1): Keep the old config schema while allowing bounded camera calls.
            output_root=getattr(config, "camera_capture_output_dir", None),  # AUDIT-FIX(#3): Optional allowlist root for persisted photos.
        )

    async def capture_photo_async(
        self,
        *,
        output_path: str | Path | None = None,
        filename: str = _DEFAULT_CAPTURE_FILENAME,
    ) -> CapturedPhoto:
        return await asyncio.to_thread(self.capture_photo, output_path=output_path, filename=filename)  # AUDIT-FIX(#5): Provide a non-blocking entrypoint for the single-process async stack.

    def capture_photo(
        self,
        *,
        output_path: str | Path | None = None,
        filename: str = _DEFAULT_CAPTURE_FILENAME,
    ) -> CapturedPhoto:
        safe_filename = self._sanitize_filename(filename)  # AUDIT-FIX(#7): Strip path/control characters from returned filenames.
        ffmpeg_binary = shutil.which(self.ffmpeg_path)
        if ffmpeg_binary is None:
            raise CameraConfigurationError(
                f"ffmpeg was not found on PATH: {self.ffmpeg_path}",
                user_safe_message="The camera software is not installed correctly.",
            )

        attempted_formats = tuple(self._candidate_input_formats())
        errors: list[str] = []
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
                except subprocess.TimeoutExpired as exc:
                    raise CameraCaptureTimeoutError(
                        (
                            f"Camera capture timed out for {self.device} using format {label} "
                            f"after {self.capture_timeout_seconds:.1f}s"
                        ),
                        user_safe_message="The camera took too long to respond. Please try again.",
                    ) from exc
                except OSError as exc:
                    raise CameraCaptureFailedError(
                        f"Unable to execute ffmpeg for camera capture: {exc}",
                        user_safe_message="The camera could not be started. Please try again.",
                    ) from exc

                if result.returncode == 0 and result.stdout and self._is_png_bytes(result.stdout):  # AUDIT-FIX(#7): Verify ffmpeg really returned a PNG payload.
                    capture = CapturedPhoto(
                        data=bytes(result.stdout),
                        content_type="image/png",
                        filename=safe_filename,
                        source_device=self.device,
                        input_format=candidate,
                    )
                    if output_path is not None:
                        self._write_output_file(output_path, safe_filename, capture.data)
                    return capture

                stderr = self._summarize_process_error(result.stderr)  # AUDIT-FIX(#6): Normalize and truncate raw ffmpeg stderr before surfacing it.
                if result.returncode == 0 and result.stdout and not self._is_png_bytes(result.stdout):
                    stderr = "ffmpeg returned non-PNG bytes"
                errors.append(f"{label}: {stderr}")

        candidates = ", ".join(fmt or "default" for fmt in attempted_formats)
        raise CameraCaptureFailedError(
            f"Camera capture failed for {self.device} with formats [{candidates}]: {' | '.join(errors)}",
            user_safe_message="The camera could not take a photo right now. Please try again.",
        )

    def _candidate_input_formats(self) -> Sequence[str | None]:
        if self.input_format:
            return (self.input_format,)
        return _DEFAULT_FFMPEG_FORMAT_CANDIDATES

    def _build_command(self, ffmpeg_binary: str, input_format: str | None) -> list[str]:
        command = [
            ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",  # AUDIT-FIX(#1): Prevent ffmpeg from inheriting stdin and waiting for terminal input.
            "-f",
            "video4linux2",
            "-framerate",
            str(self.framerate),
            "-video_size",
            f"{self.width}x{self.height}",
        ]
        if input_format:
            command.extend(["-input_format", input_format])
        command.extend(
            [
                "-i",
                self.device,
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-",
            ]
        )
        return command

    @classmethod
    def _get_device_lock(cls, device: str) -> threading.Lock:
        with cls._device_locks_guard:
            lock = cls._device_locks.get(device)
            if lock is None:
                lock = threading.Lock()
                cls._device_locks[device] = lock
            return lock

    def _write_output_file(self, output_path: str | Path, filename: str, data: bytes) -> None:
        target = self._resolve_output_file(output_path, filename)  # AUDIT-FIX(#3): Constrain writes to an allowed directory tree.
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
            os.replace(temp_path, target)  # AUDIT-FIX(#2): Use atomic replace so readers never observe a partial image file.
            self._sync_directory(parent_dir)  # AUDIT-FIX(#2): Flush directory metadata for better restart resilience after power loss.
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
        safe_stem = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in stem)
        safe_stem = safe_stem.strip("._-") or "camera-capture"
        return f"{safe_stem}.png"

    @staticmethod
    def _is_png_bytes(data: bytes) -> bool:
        return data.startswith(_PNG_SIGNATURE)

    @staticmethod
    def _summarize_process_error(stderr: bytes | None) -> str:
        if not stderr:
            return "empty output"
        decoded = stderr.decode("utf-8", errors="replace")
        cleaned = " ".join("".join(character if character.isprintable() else " " for character in decoded).split())
        if len(cleaned) > _MAX_ERROR_TEXT_LENGTH:
            return f"{cleaned[: _MAX_ERROR_TEXT_LENGTH - 3].rstrip()}..."
        return cleaned or "empty output"

    @staticmethod
    def _validate_non_empty_text(name: str, value: str) -> str:
        if not isinstance(value, str):
            raise CameraConfigurationError(f"{name} must be a string")
        normalized = value.strip()
        if not normalized:
            raise CameraConfigurationError(f"{name} must not be empty")
        if any(character in normalized for character in ("\x00", "\r", "\n")):
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
        normalized = cls._validate_non_empty_text("input_format", input_format)
        return normalized

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