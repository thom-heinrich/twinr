"""Persist a bounded continuous debug stream for HDMI attention refresh.

This module keeps first-run evidence for the local eye-follow path separate
from the normal changed-only ops events. Each attention refresh may append one
compact JSONL tick containing outcome codes, stage timings, and the current
camera/target/cue state so short-lived dropouts can be diagnosed after the
fact instead of only while the Pi is actively failing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import math

from twinr.agent.base_agent.config import TwinrConfig
from twinr.ops.paths import resolve_ops_paths_for_config


_DEFAULT_FILE_NAME = "attention_debug.jsonl"
_DEFAULT_MAX_LINES = 2048
_DEFAULT_TRIM_EVERY = 64
_MAX_TEXT_LEN = 240


def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _compact_text(value: object, *, limit: int = _MAX_TEXT_LEN) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3].rstrip() + "..."


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return _compact_text(value)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return _compact_text(repr(value))


@dataclass(frozen=True, slots=True)
class AttentionDebugStreamConfig:
    """Store bounded attention-debug stream settings."""

    path: Path
    enabled: bool = True
    max_lines: int = _DEFAULT_MAX_LINES
    trim_every: int = _DEFAULT_TRIM_EVERY

    @classmethod
    def from_config(cls, config: TwinrConfig | object) -> "AttentionDebugStreamConfig":
        """Build a bounded debug-stream config from runtime config."""

        project_path = resolve_ops_paths_for_config(config).ops_store_root / _DEFAULT_FILE_NAME
        raw_path = getattr(config, "display_attention_debug_stream_path", project_path)
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (Path(getattr(config, "project_root", ".") or ".") / path).resolve(strict=False)
        return cls(
            path=path,
            enabled=bool(getattr(config, "display_attention_debug_stream_enabled", True)),
            max_lines=max(128, int(getattr(config, "display_attention_debug_stream_max_lines", _DEFAULT_MAX_LINES) or _DEFAULT_MAX_LINES)),
            trim_every=max(1, int(getattr(config, "display_attention_debug_stream_trim_every", _DEFAULT_TRIM_EVERY) or _DEFAULT_TRIM_EVERY)),
        )


class AttentionDebugStream:
    """Append bounded continuous attention-debug ticks to a JSONL file."""

    def __init__(self, *, config: AttentionDebugStreamConfig) -> None:
        self.config = config
        self._writes_since_trim = 0

    @classmethod
    def from_config(cls, config: TwinrConfig | object) -> "AttentionDebugStream":
        """Build one debug stream rooted under Twinr's ops store."""

        return cls(config=AttentionDebugStreamConfig.from_config(config))

    def append_tick(
        self,
        *,
        outcome: str,
        observed_at: float | None,
        data: dict[str, object] | None = None,
    ) -> None:
        """Append one bounded attention-debug tick."""

        if not self.config.enabled:
            return
        entry = {
            "created_at": _utc_now_iso_z(),
            "outcome": _compact_text(outcome, limit=96) or "unknown",
            "observed_at": observed_at,
            "data": _json_safe(data or {}),
        }
        self.config.path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True, allow_nan=False))
            handle.write("\n")
        self._writes_since_trim += 1
        if self._writes_since_trim >= self.config.trim_every:
            self._trim()

    def _trim(self) -> None:
        """Keep only the newest bounded slice of debug ticks."""

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
    "AttentionDebugStream",
    "AttentionDebugStreamConfig",
]
