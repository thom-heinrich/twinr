"""Render bounded media clips into reusable WAV payloads for workflow audio."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
import shutil
import subprocess

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.audio import normalize_wav_playback_level

_DEFAULT_RENDER_TIMEOUT_S = 15.0
_DEFAULT_OUTPUT_SAMPLE_RATE_HZ = 24000
_DEFAULT_OUTPUT_CHANNELS = 1
_RIFF_PREFIX = b"RIFF"
_MAX_ERROR_TEXT_CHARS = 200
_CACHE_LOCK = Lock()
_RENDERED_CLIP_CACHE: dict[tuple[object, ...], bytes] = {}


@dataclass(frozen=True, slots=True)
class RenderedAudioClipSpec:
    """Describe one bounded rendered audio clip."""

    relative_path: Path
    clip_start_s: float
    clip_duration_s: float
    fade_in_duration_s: float
    fade_out_start_s: float
    fade_out_duration_s: float
    output_gain: float
    output_sample_rate_hz: int = _DEFAULT_OUTPUT_SAMPLE_RATE_HZ
    output_channels: int = _DEFAULT_OUTPUT_CHANNELS
    normalize_max_gain: float = 1.0
    render_timeout_s: float = _DEFAULT_RENDER_TIMEOUT_S


class RenderedAudioClipError(RuntimeError):
    """Represent a rendered clip failure that is safe to surface."""


class RenderedAudioClipConfigurationError(RenderedAudioClipError):
    """Represent invalid rendered-clip configuration."""


class RenderedAudioClipRenderError(RenderedAudioClipError):
    """Represent a bounded clip render failure."""


def build_rendered_audio_clip_wav_bytes(
    config: TwinrConfig,
    spec: RenderedAudioClipSpec,
) -> bytes | None:
    """Return one rendered WAV payload for the configured clip spec."""

    source_path = _resolve_clip_path(config, spec)
    if source_path is None:
        return None
    ffmpeg_binary = shutil.which(str(getattr(config, "camera_ffmpeg_path", "ffmpeg") or "ffmpeg"))
    if ffmpeg_binary is None:
        raise RenderedAudioClipConfigurationError("ffmpeg_missing")
    cache_key = _cache_key(source_path=source_path, spec=spec)
    with _CACHE_LOCK:
        cached = _RENDERED_CLIP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    command = _build_ffmpeg_command(ffmpeg_binary=ffmpeg_binary, source_path=source_path, spec=spec)
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=max(1.0, float(spec.render_timeout_s)),
        )
    except subprocess.TimeoutExpired as exc:
        raise RenderedAudioClipRenderError("render_timeout") from exc
    except OSError as exc:
        raise RenderedAudioClipRenderError(f"render_exec_failed:{type(exc).__name__}") from exc
    if result.returncode != 0:
        raise RenderedAudioClipRenderError(_summarize_process_error(result.stderr))
    if not result.stdout.startswith(_RIFF_PREFIX):
        raise RenderedAudioClipRenderError("render_invalid_wav")
    rendered = normalize_wav_playback_level(result.stdout, max_gain=max(1.0, float(spec.normalize_max_gain)))
    with _CACHE_LOCK:
        _RENDERED_CLIP_CACHE[cache_key] = rendered
    return rendered


def iter_wav_bytes_chunks(wav_bytes: bytes, *, chunk_size: int = 16384):
    """Yield one WAV payload in bounded chunks for interruptible playback."""

    normalized_chunk_size = max(1024, int(chunk_size))
    for start in range(0, len(wav_bytes), normalized_chunk_size):
        yield wav_bytes[start : start + normalized_chunk_size]


def _resolve_clip_path(config: TwinrConfig, spec: RenderedAudioClipSpec) -> Path | None:
    project_root = Path(str(getattr(config, "project_root", ".") or ".")).expanduser().resolve(strict=False)
    candidate = (project_root / spec.relative_path).resolve(strict=False)
    if candidate.is_file():
        return candidate
    return None


def _build_ffmpeg_command(
    *,
    ffmpeg_binary: str,
    source_path: Path,
    spec: RenderedAudioClipSpec,
) -> list[str]:
    return [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{float(spec.clip_start_s):.3f}",
        "-t",
        f"{float(spec.clip_duration_s):.3f}",
        "-i",
        str(source_path),
        "-vn",
        "-sn",
        "-dn",
        "-af",
        (
            f"volume={float(spec.output_gain):.3f},"
            f"afade=t=in:st=0:d={float(spec.fade_in_duration_s):.3f},"
            f"afade=t=out:st={float(spec.fade_out_start_s):.3f}:d={float(spec.fade_out_duration_s):.3f}"
        ),
        "-ac",
        str(int(spec.output_channels)),
        "-ar",
        str(int(spec.output_sample_rate_hz)),
        "-f",
        "wav",
        "-",
    ]


def _cache_key(*, source_path: Path, spec: RenderedAudioClipSpec) -> tuple[object, ...]:
    stat = source_path.stat()
    return (
        str(source_path),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        str(spec.relative_path),
        float(spec.clip_start_s),
        float(spec.clip_duration_s),
        float(spec.fade_in_duration_s),
        float(spec.fade_out_start_s),
        float(spec.fade_out_duration_s),
        float(spec.output_gain),
        int(spec.output_sample_rate_hz),
        int(spec.output_channels),
        float(spec.normalize_max_gain),
    )


def _summarize_process_error(stderr: bytes) -> str:
    text = stderr.decode("utf-8", errors="ignore").strip()
    if not text:
        return "render_failed"
    text = text.replace("\r", " ").replace("\n", " ").strip() or "render_failed"
    if len(text) > _MAX_ERROR_TEXT_CHARS:
        return f"{text[: _MAX_ERROR_TEXT_CHARS - 3]}..."
    return text
