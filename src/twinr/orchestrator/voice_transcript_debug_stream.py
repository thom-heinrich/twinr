"""Persist bounded textual voice-gateway transcript evidence.

The live transcript-first gateway on thh1986 must make its raw STT decisions
inspectable after the fact. This module appends one compact JSONL record per
transcribed voice decision window so operators can see exactly what the
gateway heard without persisting room audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from twinr.agent.base_agent.config import TwinrConfig


_DEFAULT_FILE_NAME = "voice_gateway_transcripts.jsonl"
_DEFAULT_MAX_LINES = 4096
_DEFAULT_TRIM_EVERY = 64
_MAX_TEXT_LEN = 4096


def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bounded_text(value: object, *, limit: int = _MAX_TEXT_LEN) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return _bounded_text(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return _bounded_text(repr(value))


def _default_project_root(config: TwinrConfig | object) -> Path:
    """Prefer the configured project root and otherwise stay under the cwd.

    The live gateway may be started from a temporary env file outside the
    leading repo. For transcript evidence we still want the artifact anchored
    under the actual repo/runtime checkout, not under whatever fallback
    `runtime_state_path` happens to point to.
    """

    raw_project_root = str(getattr(config, "project_root", "") or "").strip()
    if raw_project_root and raw_project_root != ".":
        candidate = Path(raw_project_root).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve(strict=False)
        else:
            candidate = candidate.resolve(strict=False)
        return candidate
    return Path.cwd().resolve(strict=False)


@dataclass(frozen=True, slots=True)
class VoiceTranscriptDebugStreamConfig:
    """Store bounded transcript-debug stream settings."""

    path: Path
    enabled: bool = True
    max_lines: int = _DEFAULT_MAX_LINES
    trim_every: int = _DEFAULT_TRIM_EVERY

    @classmethod
    def from_config(cls, config: TwinrConfig | object) -> "VoiceTranscriptDebugStreamConfig":
        """Build one bounded transcript-debug config from Twinr runtime config."""

        project_path = _default_project_root(config) / "artifacts" / "stores" / "ops" / _DEFAULT_FILE_NAME
        raw_path = getattr(config, "voice_orchestrator_transcript_debug_path", project_path)
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (Path(getattr(config, "project_root", ".") or ".") / path).resolve(strict=False)
        return cls(
            path=path,
            enabled=bool(getattr(config, "voice_orchestrator_transcript_debug_enabled", True)),
            max_lines=max(
                256,
                int(
                    getattr(
                        config,
                        "voice_orchestrator_transcript_debug_max_lines",
                        _DEFAULT_MAX_LINES,
                    )
                    or _DEFAULT_MAX_LINES
                ),
            ),
            trim_every=max(
                1,
                int(
                    getattr(
                        config,
                        "voice_orchestrator_transcript_debug_trim_every",
                        _DEFAULT_TRIM_EVERY,
                    )
                    or _DEFAULT_TRIM_EVERY
                ),
            ),
        )


class VoiceTranscriptDebugStream:
    """Append bounded transcript evidence for the live voice gateway."""

    def __init__(self, *, config: VoiceTranscriptDebugStreamConfig) -> None:
        self.config = config
        self._writes_since_trim = 0

    @classmethod
    def from_config(cls, config: TwinrConfig | object) -> "VoiceTranscriptDebugStream":
        """Build one transcript-debug stream rooted under Twinr's ops store."""

        return cls(config=VoiceTranscriptDebugStreamConfig.from_config(config))

    def append_entry(
        self,
        *,
        session_id: str | None,
        trace_id: str | None,
        state: str,
        backend: str,
        stage: str,
        outcome: str,
        transcript: str | None = None,
        matched_phrase: str | None = None,
        remaining_text: str | None = None,
        detector_label: str | None = None,
        score: float | None = None,
        sample: dict[str, object] | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        """Append one bounded transcript-debug record."""

        if not self.config.enabled:
            return
        entry = {
            "created_at": _utc_now_iso_z(),
            "session_id": _bounded_text(session_id, limit=128) if session_id else None,
            "trace_id": _bounded_text(trace_id, limit=128) if trace_id else None,
            "state": _bounded_text(state, limit=64) or "unknown",
            "backend": _bounded_text(backend, limit=64) or "unknown",
            "stage": _bounded_text(stage, limit=96) or "unknown",
            "outcome": _bounded_text(outcome, limit=128) or "unknown",
            "transcript": _bounded_text(transcript) if transcript is not None else None,
            "matched_phrase": _bounded_text(matched_phrase, limit=128) if matched_phrase else None,
            "remaining_text": _bounded_text(remaining_text) if remaining_text is not None else None,
            "detector_label": _bounded_text(detector_label, limit=128) if detector_label else None,
            "score": score if score is None or math.isfinite(float(score)) else str(score),
            "sample": _json_safe(sample or {}),
            "details": _json_safe(details or {}),
        }
        self.config.path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True, allow_nan=False))
            handle.write("\n")
        self._writes_since_trim += 1
        if self._writes_since_trim >= self.config.trim_every:
            self._trim()

    def _trim(self) -> None:
        """Keep only the newest bounded slice of transcript-debug records."""

        self._writes_since_trim = 0
        if not self.config.path.is_file():
            return
        try:
            lines = self.config.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        if len(lines) <= self.config.max_lines:
            return
        kept = lines[-self.config.max_lines :]
        self.config.path.write_text("\n".join(kept) + "\n", encoding="utf-8")


__all__ = [
    "VoiceTranscriptDebugStream",
    "VoiceTranscriptDebugStreamConfig",
]
