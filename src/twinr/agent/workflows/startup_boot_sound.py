"""Render and play Twinr's bounded startup boot sound clip.

This helper keeps the orchestration loops thin while packaging the startup
audio asset lookup, bounded ``ffmpeg`` render, fade profile, and queued
speaker playback behind one focused workflow-local module.
"""

from __future__ import annotations

from pathlib import Path
from threading import Thread
from typing import Callable

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.workflows.playback_coordinator import PlaybackCoordinator, PlaybackPriority
from twinr.agent.workflows.rendered_audio_clip import RenderedAudioClipSpec, build_rendered_audio_clip_wav_bytes

_BOOT_SOUND_SPEC = RenderedAudioClipSpec(
    relative_path=Path("media") / "boot.mp3",
    clip_start_s=4.0,
    clip_duration_s=7.0,
    fade_in_duration_s=1.0,
    fade_out_start_s=4.0,
    fade_out_duration_s=3.0,
    output_gain=0.1,
    normalize_max_gain=1.0,
)
_MAX_ERROR_TEXT_CHARS = 200


class StartupBootSoundError(RuntimeError):
    """Represent a boot-sound startup failure that is safe to surface."""


class StartupBootSoundConfigurationError(StartupBootSoundError):
    """Represent invalid local startup-boot-sound configuration."""


class StartupBootSoundRenderError(StartupBootSoundError):
    """Represent a bounded render/playback failure for the boot clip."""


def start_startup_boot_sound(
    *,
    config: TwinrConfig,
    playback_coordinator: PlaybackCoordinator,
    emit: Callable[[str], None] | None = None,
) -> Thread | None:
    """Start boot-sound playback in the background so loop startup stays responsive."""

    if _resolve_boot_sound_path(config) is None:
        return None
    worker = Thread(
        target=lambda: play_startup_boot_sound(
            config=config,
            playback_coordinator=playback_coordinator,
            emit=emit,
        ),
        name="twinr-startup-boot-sound",
        daemon=True,
    )
    worker.start()
    return worker


def play_startup_boot_sound(
    *,
    config: TwinrConfig,
    playback_coordinator: PlaybackCoordinator,
    emit: Callable[[str], None] | None = None,
) -> bool:
    """Render and queue the boot-sound clip once through the playback coordinator."""

    try:
        wav_bytes = build_startup_boot_sound_wav_bytes(config)
    except StartupBootSoundError as exc:
        _safe_emit(emit, f"boot_sound_failed={_summarize_error_text(exc)}")
        return False
    if wav_bytes is None:
        return False
    try:
        playback_coordinator.play_wav_bytes(
            owner="startup_boot_sound",
            priority=PlaybackPriority.FEEDBACK,
            wav_bytes=wav_bytes,
        )
    except Exception as exc:  # AUDIT-FIX(#1): Startup earcons must not abort the main loop if the speaker path is unavailable.
        _safe_emit(emit, f"boot_sound_failed={type(exc).__name__}")
        return False
    _safe_emit(emit, "boot_sound=played")
    return True


def build_startup_boot_sound_wav_bytes(config: TwinrConfig) -> bytes | None:
    """Return the normalized WAV payload for the startup boot clip."""

    try:
        return build_rendered_audio_clip_wav_bytes(config, _BOOT_SOUND_SPEC)
    except Exception as exc:
        if isinstance(exc, StartupBootSoundError):
            raise
        name = type(exc).__name__
        if name.endswith("ConfigurationError"):
            raise StartupBootSoundConfigurationError(str(exc)) from exc
        if name.endswith("RenderError"):
            raise StartupBootSoundRenderError(str(exc)) from exc
        raise


def _resolve_boot_sound_path(config: TwinrConfig) -> Path | None:
    project_root = Path(str(getattr(config, "project_root", ".") or ".")).expanduser().resolve(strict=False)
    candidate = (project_root / _BOOT_SOUND_SPEC.relative_path).resolve(strict=False)
    if candidate.is_file():
        return candidate
    return None


def _summarize_error_text(error: object) -> str:
    text = str(error).replace("\r", " ").replace("\n", " ").strip() or "unknown"
    if len(text) > _MAX_ERROR_TEXT_CHARS:
        return f"{text[: _MAX_ERROR_TEXT_CHARS - 3]}..."
    return text


def _safe_emit(emit: Callable[[str], None] | None, line: str) -> None:
    if not callable(emit):
        return
    try:
        emit(line)
    except Exception:
        return
