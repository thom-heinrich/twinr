"""Persist bounded opt-in voice-gateway audio artifacts for forensic replay.

The live transcript-first gateway normally stores only text-level evidence. When
timing- or hardware-sensitive wake failures still need stronger proof, operators
can explicitly enable this store to keep a short rolling window of WAV captures
for the same bounded decision windows that already land in the transcript debug
stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.audio import AmbientAudioCaptureWindow, pcm16_to_wav_bytes


_DEFAULT_DIR_NAME = "voice_gateway_audio"
_DEFAULT_MAX_FILES = 24


def _utc_timestamp_token() -> str:
    """Return one filesystem-safe UTC timestamp for artifact names."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def _sanitize_token(value: object, *, fallback: str) -> str:
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
    return sanitized[:48]


@dataclass(frozen=True, slots=True)
class VoiceAudioDebugArtifactStoreConfig:
    """Store bounded audio-artifact settings for the live voice gateway."""

    dir_path: Path
    enabled: bool = False
    max_files: int = _DEFAULT_MAX_FILES

    @classmethod
    def from_config(cls, config: TwinrConfig | object) -> "VoiceAudioDebugArtifactStoreConfig":
        """Build one artifact-store config from the canonical Twinr config."""

        default_dir = _default_project_root(config) / "artifacts" / "stores" / "ops" / _DEFAULT_DIR_NAME
        raw_dir = getattr(config, "voice_orchestrator_audio_debug_dir", None) or default_dir
        dir_path = Path(raw_dir).expanduser()
        if not dir_path.is_absolute():
            dir_path = (Path(getattr(config, "project_root", ".") or ".") / dir_path).resolve(strict=False)
        return cls(
            dir_path=dir_path,
            enabled=bool(getattr(config, "voice_orchestrator_audio_debug_enabled", False)),
            max_files=max(
                4,
                int(
                    getattr(
                        config,
                        "voice_orchestrator_audio_debug_max_files",
                        _DEFAULT_MAX_FILES,
                    )
                    or _DEFAULT_MAX_FILES
                ),
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
        wav_bytes = pcm16_to_wav_bytes(
            capture.pcm_bytes,
            sample_rate=capture.sample_rate,
            channels=capture.channels,
        )
        payload_sha = hashlib.sha256(wav_bytes).hexdigest()
        file_name = "_".join(
            (
                _utc_timestamp_token(),
                _sanitize_token(stage, fallback="stage"),
                _sanitize_token(outcome, fallback="outcome"),
                _sanitize_token(session_id, fallback="session"),
                _sanitize_token(trace_id, fallback="trace"),
                payload_sha[:12],
            )
        )
        artifact_path = self.config.dir_path / f"{file_name}.wav"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(wav_bytes)
        self._trim()
        return {
            "audio_artifact_path": str(artifact_path),
            "audio_artifact_sha256": payload_sha[:16],
            "audio_artifact_bytes": len(wav_bytes),
            "audio_artifact_duration_ms": int(capture.sample.duration_ms),
        }

    def _trim(self) -> None:
        """Keep only the newest bounded number of persisted WAV artifacts."""

        if not self.config.dir_path.is_dir():
            return
        wav_paths = sorted(
            self.config.dir_path.glob("*.wav"),
            key=lambda candidate: candidate.name,
        )
        overflow = len(wav_paths) - self.config.max_files
        if overflow <= 0:
            return
        for path in wav_paths[:overflow]:
            try:
                path.unlink()
            except OSError:
                continue


__all__ = [
    "VoiceAudioDebugArtifactStore",
    "VoiceAudioDebugArtifactStoreConfig",
]
