# CHANGELOG: 2026-03-29
# BUG-1: Parse opt-in booleans safely so "false"/"0"/"off" no longer enables audio capture persistence.
# BUG-2: Eliminate silent filename collisions and same-second retention misordering by using microsecond timestamps plus ordered unique artifact IDs.
# BUG-3: Prevent debug-store exceptions from taking down the live voice gateway; persist_capture is now best-effort and returns safe error metadata on store-local failures.
# BUG-4: Serialize writes/trimming across threads and processes to remove retention races under concurrent captures.
# SEC-1: Make artifact dirs/files private (0700/0600 by default) and switch to secure atomic temp-file writes with fsync for WAV/JSON outputs.
# SEC-2: Reduce partial-write and temp-name exposure by using mkstemp()+os.replace() and advisory locking.
# IMP-1: Add structured sidecar JSON manifests for auditability, provenance, and replay without loading WAV blobs.
# IMP-2: Add optional byte-budget retention and full SHA-256 metadata while preserving existing return keys.
# BREAKING: persist_capture now degrades to safe error metadata on store-local failures instead of propagating exceptions from the optional debug sink.

"""Persist bounded opt-in voice-gateway audio artifacts for forensic replay.

The live transcript-first gateway normally stores only text-level evidence. When
timing- or hardware-sensitive wake failures still need stronger proof, operators
can explicitly enable this store to keep a short rolling window of WAV captures
for the same bounded decision windows that already land in the transcript debug
stream.

This implementation is best-effort by design: local storage failures are turned
into safe error metadata so the voice gateway itself does not fail because an
optional forensic sink had a transient filesystem problem.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import uuid

try:
    import fcntl
except ImportError:  # pragma: no cover - not expected on Raspberry Pi / Unix.
    fcntl = None  # type: ignore[assignment]

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.audio import AmbientAudioCaptureWindow, pcm16_to_wav_bytes


_DEFAULT_DIR_NAME = "voice_gateway_audio"
_DEFAULT_MAX_FILES = 24
_DEFAULT_DIR_MODE = 0o700
_DEFAULT_FILE_MODE = 0o600
_LOCK_FILE_NAME = ".voice_audio_debug.lock"
_MANIFEST_SUFFIX = ".json"
_THREAD_LOCK = threading.Lock()


def _utc_now() -> datetime:
    """Return one timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


def _utc_timestamp_token(at: datetime | None = None) -> str:
    """Return one filesystem-safe UTC timestamp with microsecond precision."""

    instant = at or _utc_now()
    return instant.strftime("%Y%m%dT%H%M%S_%fZ")


def _ordered_unique_token() -> str:
    """Return one time-ordered unique token when supported by this Python."""

    uuid7_factory = getattr(uuid, "uuid7", None)
    if callable(uuid7_factory):
        return uuid7_factory().hex  # pylint: disable=not-callable
    return uuid.uuid4().hex


def _default_project_root(config: TwinrConfig | object) -> Path:
    """Prefer the configured project root and otherwise stay under the cwd."""

    raw_project_root = str(getattr(config, "project_root", "") or "").strip()
    if raw_project_root and raw_project_root != ".":
        candidate = Path(raw_project_root).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve(strict=False)
        else:
            candidate = candidate.resolve(strict=False)
        return candidate
    return Path.cwd().resolve(strict=False)


def _sanitize_token(
    value: object,
    *,
    fallback: str,
    max_length: int = 48,
) -> str:
    """Return one short filename-safe token without regex product logic."""

    raw = str(value or "").strip()
    if not raw:
        return fallback
    sanitized = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in raw
    ).strip("_")
    if not sanitized:
        return fallback
    return sanitized[:max(1, max_length)]


def _coerce_bool(value: object, *, default: bool) -> bool:
    """Parse booleans robustly from native or env-style config values."""

    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on", "enabled"}:
        return True
    if text in {"", "0", "false", "f", "no", "n", "off", "disabled"}:
        return False
    return default


def _coerce_int(
    value: object,
    *,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Parse one integer config value with clamping and safe fallback."""

    try:
        parsed = int(str(value).strip()) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _coerce_optional_int(
    value: object,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    """Parse one optional integer config value with clamping."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = int(text)
    except (TypeError, ValueError):
        return None
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _coerce_mode(value: object, *, default: int) -> int:
    """Parse octal or decimal file modes safely."""

    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        parsed = int(text, 0)
    except (TypeError, ValueError):
        return default
    parsed &= 0o777
    return parsed or default


def _capture_duration_ms(capture: AmbientAudioCaptureWindow) -> int:
    """Return duration_ms from capture metadata or derive it from PCM geometry."""

    sample = getattr(capture, "sample", None)
    raw_duration_ms = getattr(sample, "duration_ms", None)
    try:
        if raw_duration_ms is not None:
            duration_ms = int(raw_duration_ms)
            if duration_ms >= 0:
                return duration_ms
    except (TypeError, ValueError):
        pass

    pcm_bytes = bytes(getattr(capture, "pcm_bytes", b"") or b"")
    sample_rate = int(getattr(capture, "sample_rate", 0) or 0)
    channels = int(getattr(capture, "channels", 0) or 0)
    if sample_rate <= 0 or channels <= 0:
        return 0
    bytes_per_frame = 2 * channels
    if bytes_per_frame <= 0:
        return 0
    frame_count = len(pcm_bytes) // bytes_per_frame
    return int((frame_count * 1000) / sample_rate)


def _json_bytes(payload: dict[str, object]) -> bytes:
    """Encode one manifest payload deterministically as UTF-8 JSON."""

    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_all(fd: int, payload: bytes) -> None:
    """Write the full payload to one raw file descriptor."""

    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short write while persisting audio artifact")
        view = view[written:]


def _fsync_directory(path: Path) -> None:
    """Best-effort fsync for the parent directory after atomic replace."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _ensure_private_directory(path: Path, *, mode: int) -> None:
    """Create the artifact directory and best-effort enforce private perms."""

    path.mkdir(parents=True, exist_ok=True, mode=mode)
    try:
        if not path.is_symlink():
            path.chmod(mode)
    except OSError:
        pass


def _atomic_write_bytes(path: Path, payload: bytes, *, mode: int) -> None:
    """Write one file atomically and durably within its target directory."""

    fd, temp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)
    try:
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, mode)
        except OSError:
            pass
        try:
            _write_all(fd, payload)
            os.fsync(fd)
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
        if not hasattr(os, "fchmod"):
            try:
                temp_path.chmod(mode)
            except OSError:
                pass
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        raise


@contextmanager
def _advisory_file_lock(lock_path: Path, *, mode: int):
    """Serialize cross-process store mutations with one advisory lock file."""

    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    fd = os.open(lock_path, flags, mode)
    try:
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, mode)
        except OSError:
            pass
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        try:
            os.close(fd)
        except OSError:
            pass


@dataclass(frozen=True, slots=True)
class VoiceAudioDebugArtifactStoreConfig:
    """Store bounded audio-artifact settings for the live voice gateway."""

    dir_path: Path
    enabled: bool = False
    max_files: int = _DEFAULT_MAX_FILES
    max_total_bytes: int | None = None
    write_manifest: bool = True
    dir_mode: int = _DEFAULT_DIR_MODE
    file_mode: int = _DEFAULT_FILE_MODE

    @classmethod
    def from_config(cls, config: TwinrConfig | object) -> "VoiceAudioDebugArtifactStoreConfig":
        """Build one artifact-store config from the canonical Twinr config."""

        project_root = _default_project_root(config)
        default_dir = project_root / "artifacts" / "stores" / "ops" / _DEFAULT_DIR_NAME
        raw_dir = getattr(config, "voice_orchestrator_audio_debug_dir", None) or default_dir
        dir_path = Path(raw_dir).expanduser()
        if not dir_path.is_absolute():
            dir_path = (project_root / dir_path).resolve(strict=False)
        else:
            dir_path = dir_path.resolve(strict=False)

        return cls(
            dir_path=dir_path,
            enabled=_coerce_bool(
                getattr(config, "voice_orchestrator_audio_debug_enabled", False),
                default=False,
            ),
            max_files=_coerce_int(
                getattr(
                    config,
                    "voice_orchestrator_audio_debug_max_files",
                    _DEFAULT_MAX_FILES,
                ),
                default=_DEFAULT_MAX_FILES,
                minimum=4,
            ),
            max_total_bytes=_coerce_optional_int(
                getattr(config, "voice_orchestrator_audio_debug_max_total_bytes", None),
                minimum=1,
            ),
            write_manifest=_coerce_bool(
                getattr(config, "voice_orchestrator_audio_debug_write_manifest", True),
                default=True,
            ),
            dir_mode=_coerce_mode(
                getattr(config, "voice_orchestrator_audio_debug_dir_mode", _DEFAULT_DIR_MODE),
                default=_DEFAULT_DIR_MODE,
            ),
            file_mode=_coerce_mode(
                getattr(config, "voice_orchestrator_audio_debug_file_mode", _DEFAULT_FILE_MODE),
                default=_DEFAULT_FILE_MODE,
            ),
        )


class VoiceAudioDebugArtifactStore:
    """Persist a short rolling set of WAV artifacts for voice debug windows."""

    def __init__(self, *, config: VoiceAudioDebugArtifactStoreConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: TwinrConfig | object) -> "VoiceAudioDebugArtifactStore":
        """Build one artifact store rooted under Twinr's ops directory."""

        return cls(config=VoiceAudioDebugArtifactStoreConfig.from_config(config))

    def persist_capture(
        self,
        *,
        capture: AmbientAudioCaptureWindow,
        session_id: str | None,
        trace_id: str | None,
        stage: str,
        outcome: str,
    ) -> dict[str, object] | None:
        """Persist one bounded capture window and return safe artifact metadata."""

        if not self.config.enabled:
            return None

        try:
            now = _utc_now()
            duration_ms = _capture_duration_ms(capture)
            wav_bytes = pcm16_to_wav_bytes(
                capture.pcm_bytes,
                sample_rate=capture.sample_rate,
                channels=capture.channels,
            )
            payload_sha = hashlib.sha256(wav_bytes).hexdigest()
            artifact_id = _ordered_unique_token()
            file_name = "_".join(
                (
                    _utc_timestamp_token(now),
                    _sanitize_token(stage, fallback="stage", max_length=24),
                    _sanitize_token(outcome, fallback="outcome", max_length=24),
                    _sanitize_token(session_id, fallback="session", max_length=16),
                    _sanitize_token(trace_id, fallback="trace", max_length=16),
                    artifact_id[:12],
                    payload_sha[:12],
                )
            )
            artifact_path = self.config.dir_path / f"{file_name}.wav"
            manifest_path = artifact_path.with_suffix(_MANIFEST_SUFFIX)
            created_at_iso = now.isoformat().replace("+00:00", "Z")

            manifest_payload: dict[str, object] = {
                "schema_version": 1,
                "artifact_id": artifact_id,
                "created_at_utc": created_at_iso,
                "audio_artifact_path": str(artifact_path),
                "audio_artifact_sha256": payload_sha,
                "audio_artifact_bytes": len(wav_bytes),
                "audio_artifact_duration_ms": duration_ms,
                "sample_rate": int(capture.sample_rate),
                "channels": int(capture.channels),
                "stage": str(stage),
                "outcome": str(outcome),
                "session_id": session_id,
                "trace_id": trace_id,
            }

            _ensure_private_directory(self.config.dir_path, mode=self.config.dir_mode)
            with _THREAD_LOCK:
                with _advisory_file_lock(
                    self.config.dir_path / _LOCK_FILE_NAME,
                    mode=self.config.file_mode,
                ):
                    _atomic_write_bytes(
                        artifact_path,
                        wav_bytes,
                        mode=self.config.file_mode,
                    )
                    if self.config.write_manifest:
                        _atomic_write_bytes(
                            manifest_path,
                            _json_bytes(manifest_payload),
                            mode=self.config.file_mode,
                        )
                    self._trim_locked()

            metadata: dict[str, object] = {
                "audio_artifact_id": artifact_id,
                "audio_artifact_path": str(artifact_path),
                "audio_artifact_sha256": payload_sha[:16],
                "audio_artifact_sha256_full": payload_sha,
                "audio_artifact_bytes": len(wav_bytes),
                "audio_artifact_duration_ms": duration_ms,
                "audio_artifact_created_at_utc": created_at_iso,
            }
            if self.config.write_manifest:
                metadata["audio_artifact_metadata_path"] = str(manifest_path)
            return metadata
        except Exception as exc:
            return {
                "audio_artifact_error": "persist_failed",
                "audio_artifact_error_type": exc.__class__.__name__,
            }

    def _artifact_bundle_size(self, wav_path: Path) -> int:
        """Return the total size of a WAV artifact plus its sidecar manifest."""

        total_bytes = 0
        for path in (wav_path, wav_path.with_suffix(_MANIFEST_SUFFIX)):
            try:
                total_bytes += path.lstat().st_size
            except OSError:
                continue
        return total_bytes

    def _delete_artifact_bundle(self, wav_path: Path) -> None:
        """Delete one WAV artifact and its optional sidecar manifest."""

        for path in (wav_path, wav_path.with_suffix(_MANIFEST_SUFFIX)):
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                continue

    def _wav_artifact_paths(self) -> list[Path]:
        """List persisted WAV artifacts in retention order."""

        if not self.config.dir_path.is_dir():
            return []
        wav_paths: list[Path] = []
        try:
            with os.scandir(self.config.dir_path) as entries:
                for entry in entries:
                    if not entry.name.endswith(".wav"):
                        continue
                    try:
                        if not entry.is_file(follow_symlinks=False):
                            continue
                    except OSError:
                        continue
                    wav_paths.append(Path(entry.path))
        except OSError:
            return []
        wav_paths.sort(key=lambda candidate: candidate.name)
        return wav_paths

    def _trim(self) -> None:
        """Keep only the newest bounded number of persisted WAV artifacts."""

        if not self.config.dir_path.is_dir():
            return
        with _THREAD_LOCK:
            with _advisory_file_lock(
                self.config.dir_path / _LOCK_FILE_NAME,
                mode=self.config.file_mode,
            ):
                self._trim_locked()

    def _trim_locked(self) -> None:
        """Keep only the newest bounded number of persisted artifact bundles."""

        wav_paths = self._wav_artifact_paths()
        if not wav_paths:
            return

        delete_count = max(0, len(wav_paths) - self.config.max_files)
        for wav_path in wav_paths[:delete_count]:
            self._delete_artifact_bundle(wav_path)

        if self.config.max_total_bytes is None:
            return

        remaining_paths = self._wav_artifact_paths()
        total_bytes = sum(self._artifact_bundle_size(path) for path in remaining_paths)
        if total_bytes <= self.config.max_total_bytes:
            return

        while len(remaining_paths) > 1 and total_bytes > self.config.max_total_bytes:
            wav_path = remaining_paths.pop(0)
            total_bytes -= self._artifact_bundle_size(wav_path)
            self._delete_artifact_bundle(wav_path)


__all__ = [
    "VoiceAudioDebugArtifactStore",
    "VoiceAudioDebugArtifactStoreConfig",
]
