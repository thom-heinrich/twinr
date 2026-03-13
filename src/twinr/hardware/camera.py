from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import shutil
import subprocess

from twinr.agent.base_agent.config import TwinrConfig

_DEFAULT_FFMPEG_FORMAT_CANDIDATES = (None, "yuyv422", "bayer_grbg8")


@dataclass(frozen=True, slots=True)
class CapturedPhoto:
    data: bytes
    content_type: str
    filename: str
    source_device: str
    input_format: str | None = None


class V4L2StillCamera:
    def __init__(
        self,
        *,
        device: str,
        width: int,
        height: int,
        framerate: int,
        ffmpeg_path: str = "ffmpeg",
        input_format: str | None = None,
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.framerate = framerate
        self.ffmpeg_path = ffmpeg_path
        self.input_format = input_format.strip() if input_format else None

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "V4L2StillCamera":
        return cls(
            device=config.camera_device,
            width=config.camera_width,
            height=config.camera_height,
            framerate=config.camera_framerate,
            ffmpeg_path=config.camera_ffmpeg_path,
            input_format=config.camera_input_format,
        )

    def capture_photo(
        self,
        *,
        output_path: str | Path | None = None,
        filename: str = "camera-capture.png",
    ) -> CapturedPhoto:
        ffmpeg_binary = shutil.which(self.ffmpeg_path)
        if ffmpeg_binary is None:
            raise RuntimeError(f"ffmpeg was not found on PATH: {self.ffmpeg_path}")

        attempted_formats = self._candidate_input_formats()
        errors: list[str] = []
        for candidate in attempted_formats:
            result = subprocess.run(
                self._build_command(ffmpeg_binary, candidate),
                check=False,
                capture_output=True,
            )
            if result.returncode == 0 and result.stdout:
                capture = CapturedPhoto(
                    data=bytes(result.stdout),
                    content_type="image/png",
                    filename=filename,
                    source_device=self.device,
                    input_format=candidate,
                )
                if output_path is not None:
                    output_file = Path(output_path)
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    output_file.write_bytes(capture.data)
                return capture
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            label = candidate or "default"
            errors.append(f"{label}: {stderr or 'empty output'}")

        candidates = ", ".join(fmt or "default" for fmt in attempted_formats)
        raise RuntimeError(
            f"Camera capture failed for {self.device} with formats [{candidates}]: {' | '.join(errors)}"
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
