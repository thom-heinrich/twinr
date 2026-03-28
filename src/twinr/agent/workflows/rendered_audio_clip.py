# CHANGELOG: 2026-03-28
# BUG-1: Stop producing non-seekable/stdout WAV headers with placeholder RIFF sizes by rendering to a seekable temporary file before reading bytes back.
# BUG-2: Deduplicate concurrent renders for the same clip spec so multi-threaded callers do not spawn redundant ffmpeg jobs and spike CPU/latency on Raspberry Pi 4.
# BUG-3: Bound cache growth and predicted render size so repeated or adversarially varied clip specs cannot exhaust memory on-device.
# SEC-1: Prevent path traversal / absolute-path escape; rendered media must resolve inside configured allowlisted roots.
# SEC-2: Redact surfaced ffmpeg stderr so local filesystem paths are not leaked to callers.
# IMP-1: Use high-quality time-stretching with ffmpeg+librubberband when available, with automatic fallback to daisy-chained atempo filters.
# IMP-2: Produce deterministic PCM WAV output with explicit mono 16-bit formatting, metadata stripping, validation, and high-quality resampling.

"""Render bounded media clips into reusable WAV payloads for workflow audio."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import io
import math
from pathlib import Path
import re
from threading import Event, Lock
import shutil
import subprocess
import tempfile
import wave

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.audio import normalize_wav_playback_level

_DEFAULT_RENDER_TIMEOUT_S = 15.0
_DEFAULT_OUTPUT_SAMPLE_RATE_HZ = 24000
_DEFAULT_OUTPUT_CHANNELS = 1
_DEFAULT_CACHE_MAX_ENTRIES = 32
_DEFAULT_CACHE_MAX_BYTES = 32 * 1024 * 1024
_DEFAULT_MAX_OUTPUT_BYTES = 16 * 1024 * 1024
_DEFAULT_FILTER_PROBE_TIMEOUT_S = 4.0
_DEFAULT_MAX_ERROR_TEXT_CHARS = 200
_DEFAULT_MAX_RENDER_DURATION_S = 300.0
_DEFAULT_MIN_PLAYBACK_SPEED = 0.5
_DEFAULT_MAX_PLAYBACK_SPEED = 4.0
_PCM_SAMPLE_WIDTH_BYTES = 2
_RIFF_PREFIX = b"RIFF"
_WAVE_PREFIX = b"WAVE"
_CACHE_LOCK = Lock()
_RENDERED_CLIP_CACHE: OrderedDict[tuple[object, ...], bytes] = OrderedDict()
_RENDERED_CLIP_CACHE_BYTES = 0
_INFLIGHT_RENDERS: dict[tuple[object, ...], "_InFlightRender"] = {}
_FILTER_SUPPORT_CACHE: dict[tuple[str, str], bool] = {}
_ABSOLUTE_PATH_RE = re.compile(r"/[^\s'\":]+")


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
    playback_speed: float = 1.0
    output_sample_rate_hz: int = _DEFAULT_OUTPUT_SAMPLE_RATE_HZ
    output_channels: int = _DEFAULT_OUTPUT_CHANNELS
    normalize_max_gain: float = 1.0
    render_timeout_s: float = _DEFAULT_RENDER_TIMEOUT_S


@dataclass(slots=True)
class _InFlightRender:
    event: Event = field(default_factory=Event)
    result: bytes | None = None
    error: BaseException | None = None


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

    validated_spec = _validate_spec(config=config, spec=spec)
    source_path = _resolve_clip_path(config, validated_spec)
    if source_path is None:
        return None
    ffmpeg_binary = _resolve_ffmpeg_binary(config)

    cache_key = _cache_key(source_path=source_path, spec=validated_spec)
    cached = _get_cached_render(cache_key)
    if cached is not None:
        return cached

    inflight, is_leader = _acquire_inflight_render(cache_key)
    if not is_leader:
        inflight.event.wait()
        if inflight.error is not None:
            raise inflight.error
        if inflight.result is None:
            raise RenderedAudioClipRenderError("render_missing_result")
        return inflight.result

    try:
        rendered = _render_clip_wav_bytes(
            config=config,
            ffmpeg_binary=ffmpeg_binary,
            source_path=source_path,
            spec=validated_spec,
        )
        inflight.result = rendered
        _store_cached_render(config=config, cache_key=cache_key, rendered=rendered)
        return rendered
    except BaseException as exc:  # pragma: no cover - re-raised after signalling waiters
        inflight.error = exc
        raise
    finally:
        inflight.event.set()
        with _CACHE_LOCK:
            _INFLIGHT_RENDERS.pop(cache_key, None)


def iter_wav_bytes_chunks(wav_bytes: bytes, *, chunk_size: int = 16384):
    """Yield one WAV payload in bounded chunks for interruptible playback."""

    normalized_chunk_size = max(1024, int(chunk_size))
    for start in range(0, len(wav_bytes), normalized_chunk_size):
        yield wav_bytes[start : start + normalized_chunk_size]


def _resolve_ffmpeg_binary(config: TwinrConfig) -> str:
    configured = (
        getattr(config, "audio_ffmpeg_path", None)
        or getattr(config, "camera_ffmpeg_path", None)
        or "ffmpeg"
    )
    ffmpeg_binary = shutil.which(str(configured))
    if ffmpeg_binary is None:
        raise RenderedAudioClipConfigurationError("ffmpeg_missing")
    return ffmpeg_binary


def _validate_spec(config: TwinrConfig, spec: RenderedAudioClipSpec) -> RenderedAudioClipSpec:
    clip_start_s = _require_finite_float("clip_start_s", spec.clip_start_s, minimum=0.0)
    clip_duration_s = _require_finite_float("clip_duration_s", spec.clip_duration_s, minimum=0.001)
    fade_in_duration_s = _require_finite_float("fade_in_duration_s", spec.fade_in_duration_s, minimum=0.0)
    fade_out_start_s = _require_finite_float("fade_out_start_s", spec.fade_out_start_s, minimum=0.0)
    fade_out_duration_s = _require_finite_float("fade_out_duration_s", spec.fade_out_duration_s, minimum=0.0)
    output_gain = _require_finite_float("output_gain", spec.output_gain, minimum=0.0)
    playback_speed = _normalize_playback_speed(spec.playback_speed)
    output_sample_rate_hz = _require_int(
        "output_sample_rate_hz",
        spec.output_sample_rate_hz,
        minimum=8000,
        maximum=192000,
    )
    output_channels = _require_int("output_channels", spec.output_channels, minimum=1, maximum=2)
    normalize_max_gain = _require_finite_float("normalize_max_gain", spec.normalize_max_gain, minimum=1.0)
    render_timeout_s = _require_finite_float("render_timeout_s", spec.render_timeout_s, minimum=1.0)

    expected_output_duration_s = clip_duration_s / playback_speed
    fade_in_duration_s, fade_out_start_s, fade_out_duration_s = _normalize_fade_timing(
        output_duration_s=expected_output_duration_s,
        fade_in_duration_s=fade_in_duration_s,
        fade_out_start_s=fade_out_start_s,
        fade_out_duration_s=fade_out_duration_s,
    )

    max_render_duration_s = _config_float(
        config,
        "rendered_audio_clip_max_duration_s",
        _DEFAULT_MAX_RENDER_DURATION_S,
        minimum=1.0,
    )
    if expected_output_duration_s > max_render_duration_s + 1e-6:
        raise RenderedAudioClipConfigurationError("render_duration_too_large")

    max_output_bytes = _config_int(
        config,
        "rendered_audio_clip_max_output_bytes",
        _DEFAULT_MAX_OUTPUT_BYTES,
        minimum=65536,
    )
    estimated_output_bytes = _estimate_output_wav_bytes(
        duration_s=expected_output_duration_s,
        sample_rate_hz=output_sample_rate_hz,
        channels=output_channels,
    )
    if estimated_output_bytes > max_output_bytes:
        raise RenderedAudioClipConfigurationError("render_output_too_large")

    return RenderedAudioClipSpec(
        relative_path=Path(spec.relative_path),
        clip_start_s=clip_start_s,
        clip_duration_s=clip_duration_s,
        fade_in_duration_s=fade_in_duration_s,
        fade_out_start_s=fade_out_start_s,
        fade_out_duration_s=fade_out_duration_s,
        output_gain=output_gain,
        playback_speed=playback_speed,
        output_sample_rate_hz=output_sample_rate_hz,
        output_channels=output_channels,
        normalize_max_gain=normalize_max_gain,
        render_timeout_s=render_timeout_s,
    )


def _normalize_fade_timing(
    *,
    output_duration_s: float,
    fade_in_duration_s: float,
    fade_out_start_s: float,
    fade_out_duration_s: float,
) -> tuple[float, float, float]:
    bounded_output_duration_s = max(0.0, float(output_duration_s))
    normalized_fade_in_duration_s = min(fade_in_duration_s, bounded_output_duration_s)
    normalized_fade_out_duration_s = min(fade_out_duration_s, bounded_output_duration_s)
    latest_fade_out_start_s = max(0.0, bounded_output_duration_s - normalized_fade_out_duration_s)
    normalized_fade_out_start_s = min(fade_out_start_s, latest_fade_out_start_s)
    return (
        normalized_fade_in_duration_s,
        normalized_fade_out_start_s,
        normalized_fade_out_duration_s,
    )


def _estimate_output_wav_bytes(*, duration_s: float, sample_rate_hz: int, channels: int) -> int:
    pcm_bytes = math.ceil(duration_s * float(sample_rate_hz) * float(channels) * _PCM_SAMPLE_WIDTH_BYTES)
    return 4096 + max(0, int(pcm_bytes))


def _resolve_clip_path(config: TwinrConfig, spec: RenderedAudioClipSpec) -> Path | None:
    project_root = Path(str(getattr(config, "project_root", ".") or ".")).expanduser().resolve(strict=False)
    requested_relative_path = Path(spec.relative_path)

    # BREAKING: clip paths are now confined to explicit allowlisted roots; absolute paths and '..' escapes outside those roots are rejected.
    candidate = (project_root / requested_relative_path).expanduser().resolve(strict=False)
    allowed_roots = _allowed_clip_roots(config=config, project_root=project_root)
    if not any(_is_path_within(candidate, root) for root in allowed_roots):
        raise RenderedAudioClipConfigurationError("clip_path_outside_allowed_roots")
    if candidate.is_file():
        return candidate
    return None


def _allowed_clip_roots(*, config: TwinrConfig, project_root: Path) -> tuple[Path, ...]:
    raw_roots = getattr(config, "rendered_audio_clip_allowed_roots", None)
    if raw_roots is None:
        return (project_root,)
    resolved_roots: list[Path] = []
    for raw_root in raw_roots:
        root_path = Path(str(raw_root)).expanduser()
        if not root_path.is_absolute():
            root_path = project_root / root_path
        resolved_root = root_path.resolve(strict=False)
        resolved_roots.append(resolved_root)
    if not resolved_roots:
        resolved_roots.append(project_root)
    return tuple(resolved_roots)


def _build_ffmpeg_command(
    *,
    ffmpeg_binary: str,
    output_path: Path,
    source_path: Path,
    spec: RenderedAudioClipSpec,
) -> list[str]:
    audio_filter = _build_audio_filter(ffmpeg_binary=ffmpeg_binary, spec=spec)
    return [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{spec.clip_start_s:.6f}",
        "-t",
        f"{spec.clip_duration_s:.6f}",
        "-i",
        str(source_path),
        "-map",
        "0:a:0",
        "-map_metadata",
        "-1",
        "-fflags",
        "+bitexact",
        "-flags",
        "+bitexact",
        "-vn",
        "-sn",
        "-dn",
        "-af",
        audio_filter,
        "-ac",
        str(spec.output_channels),
        "-ar",
        str(spec.output_sample_rate_hz),
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        "-y",
        str(output_path),
    ]


def _render_clip_wav_bytes(
    *,
    config: TwinrConfig,
    ffmpeg_binary: str,
    source_path: Path,
    spec: RenderedAudioClipSpec,
) -> bytes:
    max_output_bytes = _config_int(
        config,
        "rendered_audio_clip_max_output_bytes",
        _DEFAULT_MAX_OUTPUT_BYTES,
        minimum=65536,
    )

    with tempfile.TemporaryDirectory(prefix="twinr-rendered-audio-") as temp_dir:
        output_path = Path(temp_dir) / "rendered.wav"
        command = _build_ffmpeg_command(
            ffmpeg_binary=ffmpeg_binary,
            output_path=output_path,
            source_path=source_path,
            spec=spec,
        )
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                timeout=spec.render_timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RenderedAudioClipRenderError("render_timeout") from exc
        except OSError as exc:
            raise RenderedAudioClipRenderError(f"render_exec_failed:{type(exc).__name__}") from exc

        if result.returncode != 0:
            raise RenderedAudioClipRenderError(_summarize_process_error(result.stderr, source_path=source_path))
        if not output_path.is_file():
            raise RenderedAudioClipRenderError("render_missing_output")

        rendered_size = output_path.stat().st_size
        if rendered_size <= 0:
            raise RenderedAudioClipRenderError("render_empty_output")
        if rendered_size > max_output_bytes:
            raise RenderedAudioClipRenderError("render_output_too_large")

        rendered = output_path.read_bytes()

    if not rendered.startswith(_RIFF_PREFIX) or _WAVE_PREFIX not in rendered[:16]:
        raise RenderedAudioClipRenderError("render_invalid_wav")

    normalized = normalize_wav_playback_level(rendered, max_gain=spec.normalize_max_gain)
    if len(normalized) > max_output_bytes:
        raise RenderedAudioClipRenderError("render_output_too_large")
    _validate_wav_bytes(
        normalized,
        expected_sample_rate_hz=spec.output_sample_rate_hz,
        expected_channels=spec.output_channels,
    )
    return normalized


def _cache_key(*, source_path: Path, spec: RenderedAudioClipSpec) -> tuple[object, ...]:
    stat = source_path.stat()
    return (
        str(source_path),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        str(spec.relative_path),
        spec.clip_start_s,
        spec.clip_duration_s,
        spec.fade_in_duration_s,
        spec.fade_out_start_s,
        spec.fade_out_duration_s,
        spec.output_gain,
        spec.playback_speed,
        spec.output_sample_rate_hz,
        spec.output_channels,
        spec.normalize_max_gain,
    )


def _get_cached_render(cache_key: tuple[object, ...]) -> bytes | None:
    with _CACHE_LOCK:
        cached = _RENDERED_CLIP_CACHE.get(cache_key)
        if cached is None:
            return None
        _RENDERED_CLIP_CACHE.move_to_end(cache_key)
        return cached


def _store_cached_render(*, config: TwinrConfig, cache_key: tuple[object, ...], rendered: bytes) -> None:
    global _RENDERED_CLIP_CACHE_BYTES

    cache_max_entries = _config_int(
        config,
        "rendered_audio_clip_cache_max_entries",
        _DEFAULT_CACHE_MAX_ENTRIES,
        minimum=1,
    )
    cache_max_bytes = _config_int(
        config,
        "rendered_audio_clip_cache_max_bytes",
        _DEFAULT_CACHE_MAX_BYTES,
        minimum=65536,
    )

    with _CACHE_LOCK:
        existing = _RENDERED_CLIP_CACHE.pop(cache_key, None)
        if existing is not None:
            _RENDERED_CLIP_CACHE_BYTES -= len(existing)

        _RENDERED_CLIP_CACHE[cache_key] = rendered
        _RENDERED_CLIP_CACHE_BYTES += len(rendered)

        while len(_RENDERED_CLIP_CACHE) > cache_max_entries or _RENDERED_CLIP_CACHE_BYTES > cache_max_bytes:
            _, evicted = _RENDERED_CLIP_CACHE.popitem(last=False)
            _RENDERED_CLIP_CACHE_BYTES -= len(evicted)


def _acquire_inflight_render(cache_key: tuple[object, ...]) -> tuple[_InFlightRender, bool]:
    with _CACHE_LOCK:
        inflight = _INFLIGHT_RENDERS.get(cache_key)
        if inflight is not None:
            return inflight, False
        inflight = _InFlightRender()
        _INFLIGHT_RENDERS[cache_key] = inflight
        return inflight, True


def _build_audio_filter(*, ffmpeg_binary: str, spec: RenderedAudioClipSpec) -> str:
    filter_parts: list[str] = [f"volume={spec.output_gain:.6f}"]

    if not math.isclose(spec.playback_speed, 1.0, rel_tol=0.0, abs_tol=0.0005):
        if _ffmpeg_supports_feature(ffmpeg_binary, "rubberband_filter"):
            filter_parts.append(
                "rubberband="
                f"tempo={spec.playback_speed:.6f}:"
                "transients=crisp:"
                "detector=compound:"
                "phase=laminar:"
                "window=short:"
                "smoothing=on:"
                "formant=preserved:"
                "pitchq=quality"
            )
        else:
            filter_parts.extend(_build_atempo_filter_chain(spec.playback_speed))

    if spec.fade_in_duration_s > 0.0:
        filter_parts.append(f"afade=t=in:st=0:d={spec.fade_in_duration_s:.6f}")
    if spec.fade_out_duration_s > 0.0:
        filter_parts.append(f"afade=t=out:st={spec.fade_out_start_s:.6f}:d={spec.fade_out_duration_s:.6f}")

    filter_parts.extend(_build_resample_filters(ffmpeg_binary=ffmpeg_binary, spec=spec))
    return ",".join(filter_parts)


def _build_atempo_filter_chain(playback_speed: float) -> list[str]:
    remaining = playback_speed
    filters: list[str] = []
    while remaining > 2.0 + 1e-6:
        filters.append("atempo=2.000000")
        remaining /= 2.0
    while remaining < 0.5 - 1e-6:
        filters.append("atempo=0.500000")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.6f}")
    return filters


def _build_resample_filters(*, ffmpeg_binary: str, spec: RenderedAudioClipSpec) -> list[str]:
    channel_layout = "mono" if spec.output_channels == 1 else "stereo"
    filters: list[str] = []
    if _ffmpeg_supports_feature(ffmpeg_binary, "soxr_resampler"):
        filters.append(
            "aresample="
            f"{spec.output_sample_rate_hz}:"
            "resampler=soxr:"
            "precision=20"
        )
    else:
        filters.append(
            "aresample="
            f"{spec.output_sample_rate_hz}:"
            "filter_size=32:"
            "phase_shift=10:"
            "linear_interp=1:"
            "exact_rational=1:"
            "dither_method=triangular_hp"
        )
    filters.append(f"aformat=sample_fmts=s16:channel_layouts={channel_layout}")
    return filters


def _ffmpeg_supports_feature(ffmpeg_binary: str, feature_name: str) -> bool:
    cache_key = (ffmpeg_binary, feature_name)
    cached = _FILTER_SUPPORT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if feature_name == "rubberband_filter":
        command = [ffmpeg_binary, "-hide_banner", "-filters"]
        expected_text = " rubberband "
    elif feature_name == "soxr_resampler":
        command = [ffmpeg_binary, "-hide_banner", "-h", "filter=aresample"]
        expected_text = " soxr "
    else:  # pragma: no cover - internal misuse guard
        raise RuntimeError(f"unknown ffmpeg feature probe: {feature_name}")

    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=_DEFAULT_FILTER_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        supported = False
    else:
        listing = ((result.stdout or b"") + b"\n" + (result.stderr or b"")).decode("utf-8", errors="ignore")
        supported = result.returncode == 0 and expected_text in listing

    _FILTER_SUPPORT_CACHE[cache_key] = supported
    return supported


def _normalize_playback_speed(playback_speed: float) -> float:
    normalized = float(playback_speed)
    if not math.isfinite(normalized):
        raise RenderedAudioClipConfigurationError("playback_speed_invalid")
    if not _DEFAULT_MIN_PLAYBACK_SPEED <= normalized <= _DEFAULT_MAX_PLAYBACK_SPEED:
        raise RenderedAudioClipConfigurationError("playback_speed_out_of_range")
    return normalized


def _validate_wav_bytes(
    wav_bytes: bytes,
    *,
    expected_sample_rate_hz: int,
    expected_channels: int,
) -> None:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            if wav_file.getframerate() != expected_sample_rate_hz:
                raise RenderedAudioClipRenderError("render_unexpected_sample_rate")
            if wav_file.getnchannels() != expected_channels:
                raise RenderedAudioClipRenderError("render_unexpected_channel_count")
            if wav_file.getsampwidth() != _PCM_SAMPLE_WIDTH_BYTES:
                raise RenderedAudioClipRenderError("render_unexpected_sample_width")
            if wav_file.getnframes() < 0:
                raise RenderedAudioClipRenderError("render_invalid_frame_count")
    except wave.Error as exc:
        raise RenderedAudioClipRenderError("render_invalid_wav") from exc


def _summarize_process_error(stderr: bytes | None, *, source_path: Path) -> str:
    text = (stderr or b"").decode("utf-8", errors="ignore").strip()
    if not text:
        return "render_failed"
    text = text.replace("\r", " ").replace("\n", " ").strip()
    text = text.replace(str(source_path), "<clip>")
    text = _ABSOLUTE_PATH_RE.sub("<path>", text)
    if len(text) > _DEFAULT_MAX_ERROR_TEXT_CHARS:
        return f"{text[: _DEFAULT_MAX_ERROR_TEXT_CHARS - 3]}..."
    return text or "render_failed"


def _is_path_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _require_finite_float(
    name: str,
    value: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise RenderedAudioClipConfigurationError(f"{name}_invalid")
    if minimum is not None and normalized < minimum:
        raise RenderedAudioClipConfigurationError(f"{name}_out_of_range")
    if maximum is not None and normalized > maximum:
        raise RenderedAudioClipConfigurationError(f"{name}_out_of_range")
    return normalized


def _require_int(
    name: str,
    value: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    normalized = int(value)
    if minimum is not None and normalized < minimum:
        raise RenderedAudioClipConfigurationError(f"{name}_out_of_range")
    if maximum is not None and normalized > maximum:
        raise RenderedAudioClipConfigurationError(f"{name}_out_of_range")
    return normalized


def _config_int(
    config: TwinrConfig,
    attr_name: str,
    default: int,
    *,
    minimum: int | None = None,
) -> int:
    raw_value = getattr(config, attr_name, default)
    value = int(raw_value)
    if minimum is not None and value < minimum:
        value = minimum
    return value


def _config_float(
    config: TwinrConfig,
    attr_name: str,
    default: float,
    *,
    minimum: float | None = None,
) -> float:
    raw_value = getattr(config, attr_name, default)
    value = float(raw_value)
    if not math.isfinite(value):
        value = default
    if minimum is not None and value < minimum:
        value = minimum
    return value