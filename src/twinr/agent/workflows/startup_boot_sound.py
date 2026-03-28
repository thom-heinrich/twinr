"""Render and play Twinr's bounded startup boot sound clip.

This helper keeps the orchestration loops thin while packaging the startup
audio asset lookup, bounded ``ffmpeg`` render, fade profile, and queued
speaker playback behind one focused workflow-local module.
"""

# CHANGELOG: 2026-03-28
# BUG-1: Direct callers of play_startup_boot_sound()/build_startup_boot_sound_wav_bytes()
# now handle a missing boot asset consistently instead of falling through into backend
# render exceptions.
# BUG-2: All render/playback failures are now contained and classified, so worker threads
# no longer die with uncaught exceptions on stderr.
# BUG-3: Concurrent startup requests are coalesced, and duplicate boot-sound enqueues are
# suppressed for a short replay window to prevent overlapping earcons.
# SEC-1: Boot assets must now resolve inside project_root, be regular readable files, and
# stay below bounded size limits; emitted failure reasons are safe tokens instead of raw
# backend exception text.
# IMP-1: Rendered WAV bytes are validated, cached in memory, and persisted atomically on
# disk keyed by asset+spec fingerprint to avoid repeated ffmpeg cold starts on Raspberry Pi 4.
# IMP-2: Worker startup is explicit (no lambda target), cache-directory discovery is adaptive,
# and emit telemetry is more structured.
# BREAKING: media/boot.mp3 may no longer be an external symlink that resolves outside
# project_root; oversized or non-regular assets are rejected.

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Thread, current_thread
from time import monotonic
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

_BOOT_SOUND_OWNER = "startup_boot_sound"
_BOOT_SOUND_WORKER_NAME = "twinr-startup-boot-sound"
_CACHE_NAMESPACE = "startup_boot_sound"
_CACHE_VERSION = 2
_MAX_BOOT_SOUND_SOURCE_BYTES = 16 * 1024 * 1024
_MAX_RENDERED_WAV_BYTES = 12 * 1024 * 1024
_MIN_RENDERED_WAV_DURATION_S = 0.25
_MAX_RENDERED_WAV_DURATION_S = float(_BOOT_SOUND_SPEC.clip_duration_s) + 0.5
_MIN_RENDERED_WAV_RATE_HZ = 8_000
_MAX_RENDERED_WAV_RATE_HZ = 192_000
_REPLAY_SUPPRESSION_WINDOW_S = float(_BOOT_SOUND_SPEC.clip_duration_s) + 1.0

_render_cache_lock = Lock()
_render_cache: dict[str, bytes] = {}

_render_build_lock = Lock()

_start_lock = Lock()
_start_worker: Thread | None = None

_playback_lock = Lock()
_last_playback_by_fingerprint: dict[str, float] = {}


class StartupBootSoundError(RuntimeError):
    """Represent a boot-sound startup failure that is safe to surface."""


class StartupBootSoundConfigurationError(StartupBootSoundError):
    """Represent invalid local startup-boot-sound configuration."""


class StartupBootSoundRenderError(StartupBootSoundError):
    """Represent a bounded render/playback failure for the boot clip."""


@dataclass(frozen=True)
class _ResolvedBootSoundAsset:
    project_root: Path
    asset_path: Path
    asset_size: int
    asset_mtime_ns: int
    fingerprint: str
    cache_path: Path | None


def start_startup_boot_sound(
    *,
    config: TwinrConfig,
    playback_coordinator: PlaybackCoordinator,
    emit: Callable[[str], None] | None = None,
) -> Thread | None:
    """Start boot-sound playback in the background so loop startup stays responsive."""

    try:
        asset = _resolve_boot_sound_asset(config)
    except StartupBootSoundError as exc:
        _emit_failure(emit, exc)
        return None
    if asset is None:
        return None

    with _start_lock:
        global _start_worker
        if _start_worker is not None and _start_worker.is_alive():
            return _start_worker
        worker = Thread(
            target=_startup_boot_sound_worker,
            kwargs={
                "config": config,
                "playback_coordinator": playback_coordinator,
                "emit": emit,
            },
            name=_BOOT_SOUND_WORKER_NAME,
            daemon=True,
        )
        _start_worker = worker

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
        asset, wav_bytes, source = _prepare_startup_boot_sound(config)
    except StartupBootSoundError as exc:
        _emit_failure(emit, exc)
        return False
    except Exception as exc:  # Defensive containment: this helper must never escape in startup paths.
        _emit_failure(emit, StartupBootSoundRenderError(_unexpected_error_token(exc)))
        return False

    if asset is None or wav_bytes is None:
        return False

    with _playback_lock:
        # BREAKING: identical boot clips requested again inside the replay window are coalesced.
        if _should_suppress_recent_playback(asset.fingerprint):
            _safe_emit(emit, "boot_sound=deduped")
            return False
        try:
            playback_coordinator.play_wav_bytes(
                owner=_BOOT_SOUND_OWNER,
                priority=PlaybackPriority.FEEDBACK,
                wav_bytes=wav_bytes,
            )
        except Exception as exc:  # AUDIT-FIX(#1): Startup earcons must not abort the main loop if the speaker path is unavailable.
            _emit_failure(emit, StartupBootSoundRenderError(f"playback_{_normalize_error_token(type(exc).__name__)}"))
            return False
        _mark_recent_playback(asset.fingerprint)

    _safe_emit(emit, "boot_sound=played")
    _safe_emit(emit, f"boot_sound_source={source}")
    return True


def build_startup_boot_sound_wav_bytes(config: TwinrConfig) -> bytes | None:
    """Return the normalized WAV payload for the startup boot clip."""

    asset, wav_bytes, _ = _prepare_startup_boot_sound(config)
    if asset is None:
        return None
    return wav_bytes


def _startup_boot_sound_worker(
    *,
    config: TwinrConfig,
    playback_coordinator: PlaybackCoordinator,
    emit: Callable[[str], None] | None,
) -> None:
    try:
        play_startup_boot_sound(
            config=config,
            playback_coordinator=playback_coordinator,
            emit=emit,
        )
    finally:
        with _start_lock:
            global _start_worker
            if _start_worker is current_thread():
                _start_worker = None


def _prepare_startup_boot_sound(config: TwinrConfig) -> tuple[_ResolvedBootSoundAsset | None, bytes | None, str]:
    asset = _resolve_boot_sound_asset(config)
    if asset is None:
        return None, None, "missing"

    cached_bytes = _get_memory_cached_wav_bytes(asset.fingerprint)
    if cached_bytes is not None:
        return asset, cached_bytes, "memory_cache"

    cached_bytes = _load_disk_cached_wav_bytes(asset)
    if cached_bytes is not None:
        _store_memory_cached_wav_bytes(asset.fingerprint, cached_bytes)
        return asset, cached_bytes, "disk_cache"

    with _render_build_lock:
        cached_bytes = _get_memory_cached_wav_bytes(asset.fingerprint)
        if cached_bytes is not None:
            return asset, cached_bytes, "memory_cache"

        cached_bytes = _load_disk_cached_wav_bytes(asset)
        if cached_bytes is not None:
            _store_memory_cached_wav_bytes(asset.fingerprint, cached_bytes)
            return asset, cached_bytes, "disk_cache"

        wav_bytes = _render_startup_boot_sound_wav_bytes(config)
        wav_bytes = _validate_rendered_wav_bytes(wav_bytes)
        _store_memory_cached_wav_bytes(asset.fingerprint, wav_bytes)
        _persist_disk_cached_wav_bytes(asset, wav_bytes)
        return asset, wav_bytes, "render"


def _render_startup_boot_sound_wav_bytes(config: TwinrConfig) -> bytes:
    try:
        wav_bytes = build_rendered_audio_clip_wav_bytes(config, _BOOT_SOUND_SPEC)
    except StartupBootSoundError:
        raise
    except Exception as exc:
        name = type(exc).__name__
        if name.endswith("ConfigurationError"):
            raise StartupBootSoundConfigurationError("renderer_configuration_error") from exc
        if name.endswith("RenderError"):
            raise StartupBootSoundRenderError("renderer_render_error") from exc
        if isinstance(exc, (PermissionError, IsADirectoryError, NotADirectoryError)):
            raise StartupBootSoundConfigurationError("renderer_input_unreadable") from exc
        if isinstance(exc, FileNotFoundError):
            raise StartupBootSoundRenderError("renderer_backend_missing") from exc
        raise StartupBootSoundRenderError(_unexpected_error_token(exc)) from exc

    if wav_bytes is None:
        raise StartupBootSoundRenderError("renderer_empty_result")
    if isinstance(wav_bytes, memoryview):
        wav_bytes = wav_bytes.tobytes()
    elif isinstance(wav_bytes, bytearray):
        wav_bytes = bytes(wav_bytes)
    elif not isinstance(wav_bytes, bytes):
        raise StartupBootSoundRenderError("renderer_non_bytes_result")

    return wav_bytes


def _resolve_boot_sound_asset(config: TwinrConfig) -> _ResolvedBootSoundAsset | None:
    project_root = Path(str(getattr(config, "project_root", ".") or ".")).expanduser().resolve(strict=False)
    candidate = project_root / _BOOT_SOUND_SPEC.relative_path

    try:
        candidate_lstat = candidate.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise StartupBootSoundConfigurationError("boot_asset_unreadable") from exc

    try:
        resolved_candidate = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        if stat.S_ISLNK(candidate_lstat.st_mode):
            raise StartupBootSoundConfigurationError("boot_asset_broken_symlink") from exc
        return None
    except OSError as exc:
        raise StartupBootSoundConfigurationError("boot_asset_unreadable") from exc

    # BREAKING: media/boot.mp3 must resolve inside project_root; external symlink targets are rejected.
    try:
        resolved_candidate.relative_to(project_root)
    except ValueError as exc:
        raise StartupBootSoundConfigurationError("boot_asset_outside_project_root") from exc

    try:
        stat_result = resolved_candidate.stat()
    except OSError as exc:
        raise StartupBootSoundConfigurationError("boot_asset_unreadable") from exc

    if not stat.S_ISREG(stat_result.st_mode):
        raise StartupBootSoundConfigurationError("boot_asset_not_regular_file")
    if stat_result.st_size <= 0:
        raise StartupBootSoundConfigurationError("boot_asset_empty")
    # BREAKING: oversized boot assets are rejected before they ever reach the renderer.
    if stat_result.st_size > _MAX_BOOT_SOUND_SOURCE_BYTES:
        raise StartupBootSoundConfigurationError("boot_asset_too_large")

    fingerprint = _compute_boot_sound_fingerprint(
        asset_path=resolved_candidate,
        asset_size=stat_result.st_size,
        asset_mtime_ns=stat_result.st_mtime_ns,
    )
    cache_path = _resolve_boot_sound_cache_path(config, project_root, fingerprint)
    return _ResolvedBootSoundAsset(
        project_root=project_root,
        asset_path=resolved_candidate,
        asset_size=stat_result.st_size,
        asset_mtime_ns=stat_result.st_mtime_ns,
        fingerprint=fingerprint,
        cache_path=cache_path,
    )


def _compute_boot_sound_fingerprint(*, asset_path: Path, asset_size: int, asset_mtime_ns: int) -> str:
    payload = {
        "cache_version": _CACHE_VERSION,
        "asset_path": str(asset_path),
        "asset_size": asset_size,
        "asset_mtime_ns": asset_mtime_ns,
        "spec": {
            "relative_path": str(_BOOT_SOUND_SPEC.relative_path),
            "clip_start_s": float(_BOOT_SOUND_SPEC.clip_start_s),
            "clip_duration_s": float(_BOOT_SOUND_SPEC.clip_duration_s),
            "fade_in_duration_s": float(_BOOT_SOUND_SPEC.fade_in_duration_s),
            "fade_out_start_s": float(_BOOT_SOUND_SPEC.fade_out_start_s),
            "fade_out_duration_s": float(_BOOT_SOUND_SPEC.fade_out_duration_s),
            "output_gain": float(_BOOT_SOUND_SPEC.output_gain),
            "normalize_max_gain": float(_BOOT_SOUND_SPEC.normalize_max_gain),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _resolve_boot_sound_cache_path(config: TwinrConfig, project_root: Path, fingerprint: str) -> Path | None:
    for attr_name in (
        "startup_boot_sound_cache_dir",
        "rendered_audio_cache_dir",
        "cache_dir",
        "runtime_dir",
        "state_dir",
        "tmp_dir",
    ):
        raw_value = getattr(config, attr_name, None)
        if not raw_value:
            continue
        base_dir = Path(str(raw_value)).expanduser().resolve(strict=False)
        return base_dir / "twinr" / _CACHE_NAMESPACE / f"{fingerprint}.wav"

    return project_root / ".cache" / "twinr" / _CACHE_NAMESPACE / f"{fingerprint}.wav"


def _get_memory_cached_wav_bytes(fingerprint: str) -> bytes | None:
    with _render_cache_lock:
        return _render_cache.get(fingerprint)


def _store_memory_cached_wav_bytes(fingerprint: str, wav_bytes: bytes) -> None:
    with _render_cache_lock:
        _render_cache.clear()
        _render_cache[fingerprint] = wav_bytes


def _load_disk_cached_wav_bytes(asset: _ResolvedBootSoundAsset) -> bytes | None:
    if asset.cache_path is None:
        return None
    try:
        wav_bytes = asset.cache_path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError:
        return None

    try:
        return _validate_rendered_wav_bytes(wav_bytes)
    except StartupBootSoundError:
        _best_effort_unlink(asset.cache_path)
        return None


def _persist_disk_cached_wav_bytes(asset: _ResolvedBootSoundAsset, wav_bytes: bytes) -> None:
    if asset.cache_path is None:
        return

    try:
        asset.cache_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(asset.cache_path.parent),
            prefix=f"{asset.cache_path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(wav_bytes)
            handle.flush()
            os.fsync(handle.fileno())
            tmp_path = Path(handle.name)
        os.replace(tmp_path, asset.cache_path)
        try:
            os.chmod(asset.cache_path, 0o600)
        except OSError:
            pass
    except OSError:
        if tmp_path is not None:
            _best_effort_unlink(tmp_path)


def _validate_rendered_wav_bytes(wav_bytes: bytes) -> bytes:
    if not isinstance(wav_bytes, bytes):
        raise StartupBootSoundRenderError("invalid_wav_type")
    if not wav_bytes:
        raise StartupBootSoundRenderError("empty_wav")
    if len(wav_bytes) > _MAX_RENDERED_WAV_BYTES:
        raise StartupBootSoundRenderError("wav_too_large")

    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
            channel_count = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate_hz = handle.getframerate()
            frame_count = handle.getnframes()
    except (wave.Error, EOFError) as exc:
        raise StartupBootSoundRenderError("invalid_wav_payload") from exc

    if channel_count not in (1, 2):
        raise StartupBootSoundRenderError("unsupported_wav_channels")
    if not 1 <= sample_width <= 4:
        raise StartupBootSoundRenderError("unsupported_wav_sample_width")
    if not _MIN_RENDERED_WAV_RATE_HZ <= sample_rate_hz <= _MAX_RENDERED_WAV_RATE_HZ:
        raise StartupBootSoundRenderError("unsupported_wav_sample_rate")
    if frame_count <= 0:
        raise StartupBootSoundRenderError("empty_wav_frame_count")

    duration_s = frame_count / float(sample_rate_hz)
    if duration_s < _MIN_RENDERED_WAV_DURATION_S:
        raise StartupBootSoundRenderError("wav_too_short")
    if duration_s > _MAX_RENDERED_WAV_DURATION_S:
        raise StartupBootSoundRenderError("wav_too_long")

    return wav_bytes


def _should_suppress_recent_playback(fingerprint: str) -> bool:
    last_playback_s = _last_playback_by_fingerprint.get(fingerprint)
    if last_playback_s is None:
        return False
    return (monotonic() - last_playback_s) < _REPLAY_SUPPRESSION_WINDOW_S


def _mark_recent_playback(fingerprint: str) -> None:
    _last_playback_by_fingerprint.clear()
    _last_playback_by_fingerprint[fingerprint] = monotonic()


def _unexpected_error_token(exc: Exception) -> str:
    return f"unexpected_{_normalize_error_token(type(exc).__name__)}"


def _emit_failure(emit: Callable[[str], None] | None, error: StartupBootSoundError) -> None:
    token = _normalize_error_token(str(error) or type(error).__name__)
    _safe_emit(emit, f"boot_sound_failed={token}")


def _normalize_error_token(token: str) -> str:
    normalized = []
    previous_was_separator = False
    for char in token.lower():
        if char.isalnum():
            normalized.append(char)
            previous_was_separator = False
            continue
        if previous_was_separator:
            continue
        normalized.append("_")
        previous_was_separator = True
    collapsed = "".join(normalized).strip("_")
    return collapsed[:80] or "unknown"


def _best_effort_unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        return


def _safe_emit(emit: Callable[[str], None] | None, line: str) -> None:
    if not callable(emit):
        return
    try:
        emit(line)
    except Exception:
        return