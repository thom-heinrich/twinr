"""Persist the last successfully rendered display state for ops forensics.

The display loop already persists heartbeat liveness, but heartbeat alone does
not prove which footer/header values were actually shown on the physical
surface. This module stores one bounded artifact with the most recent rendered
state fields plus the health verdict that produced them, so "display says
ERROR" can be verified without guessing from unrelated runtime snapshots.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from twinr.agent.base_agent.config import TwinrConfig


_DISPLAY_RENDER_STATE_RELATIVE_PATH = Path("artifacts") / "stores" / "ops" / "display_render_state.json"
_SCHEMA_VERSION = 2
_DEFAULT_FILE_MODE = 0o600
_MAX_ARTIFACT_BYTES = 64 * 1024
_MAX_STATUS_CHARS = 32
_MAX_LAYOUT_CHARS = 32
_MAX_HEADLINE_CHARS = 96
_MAX_DETAIL_ITEMS = 4
_MAX_DETAIL_CHARS = 64
_MAX_STATE_FIELDS = 12
_MAX_FIELD_LABEL_CHARS = 32
_MAX_FIELD_VALUE_CHARS = 96
_MAX_ERROR_CHARS = 240
_MAX_OPERATOR_STATUS_CHARS = 16
_MAX_REASON_CODES = 8
_MAX_REASON_CODE_CHARS = 48
_MAX_REASON_DETAIL_CHARS = 240


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_text(value: object | None, *, max_chars: int, fallback: str | None = None) -> str | None:
    if value is None:
        return fallback
    text = "".join(ch if ch.isprintable() else " " for ch in str(value))
    compact = " ".join(text.split()).strip()
    if max_chars > 0 and len(compact) > max_chars:
        compact = compact[:max_chars].rstrip()
    if compact:
        return compact
    return fallback


def _normalize_details(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values[:_MAX_DETAIL_ITEMS]:
        item = _normalize_text(value, max_chars=_MAX_DETAIL_CHARS)
        if item:
            normalized.append(item)
    return tuple(normalized)


def _fsync_directory(path: Path) -> None:
    try:
        directory_fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        try:
            os.close(directory_fd)
        except OSError:
            pass


@dataclass(frozen=True, slots=True)
class DisplayRenderStateField:
    """Represent one state-field row exactly as rendered."""

    label: str
    value: str

    @classmethod
    def from_pair(cls, label: object, value: object) -> "DisplayRenderStateField | None":
        normalized_label = _normalize_text(label, max_chars=_MAX_FIELD_LABEL_CHARS)
        normalized_value = _normalize_text(value, max_chars=_MAX_FIELD_VALUE_CHARS)
        if not normalized_label or not normalized_value:
            return None
        return cls(label=normalized_label, value=normalized_value)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DisplayRenderStateSnapshot:
    """Capture the last display frame that was successfully presented."""

    rendered_at: str
    layout: str
    runtime_status: str
    headline: str
    details: tuple[str, ...]
    state_fields: tuple[DisplayRenderStateField, ...]
    health_status: str | None = None
    snapshot_status: str | None = None
    snapshot_stale: bool = False
    snapshot_error: str | None = None
    runtime_error: str | None = None
    operator_status: str | None = None
    operator_reason_codes: tuple[str, ...] = ()
    operator_reason_detail: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "rendered_at": self.rendered_at,
            "layout": self.layout,
            "runtime_status": self.runtime_status,
            "headline": self.headline,
            "details": list(self.details),
            "state_fields": [item.to_dict() for item in self.state_fields],
            "health_status": self.health_status,
            "snapshot_status": self.snapshot_status,
            "snapshot_stale": self.snapshot_stale,
            "snapshot_error": self.snapshot_error,
            "runtime_error": self.runtime_error,
            "operator_status": self.operator_status,
            "operator_reason_codes": list(self.operator_reason_codes),
            "operator_reason_detail": self.operator_reason_detail,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "DisplayRenderStateSnapshot":
        state_fields_raw = payload.get("state_fields")
        normalized_fields: list[DisplayRenderStateField] = []
        if isinstance(state_fields_raw, list):
            for item in state_fields_raw[:_MAX_STATE_FIELDS]:
                if not isinstance(item, dict):
                    continue
                normalized = DisplayRenderStateField.from_pair(item.get("label"), item.get("value"))
                if normalized is not None:
                    normalized_fields.append(normalized)
        details_raw = payload.get("details")
        details = _normalize_details(details_raw if isinstance(details_raw, list) else ())
        operator_reason_codes_raw = payload.get("operator_reason_codes")
        operator_reason_codes: list[str] = []
        if isinstance(operator_reason_codes_raw, list):
            for item in operator_reason_codes_raw[:_MAX_REASON_CODES]:
                normalized_code = _normalize_text(item, max_chars=_MAX_REASON_CODE_CHARS)
                if normalized_code:
                    operator_reason_codes.append(normalized_code)
        return cls(
            rendered_at=_normalize_text(payload.get("rendered_at"), max_chars=64, fallback=_utc_now().isoformat())
            or _utc_now().isoformat(),
            layout=_normalize_text(payload.get("layout"), max_chars=_MAX_LAYOUT_CHARS, fallback="default") or "default",
            runtime_status=_normalize_text(
                payload.get("runtime_status"),
                max_chars=_MAX_STATUS_CHARS,
                fallback="error",
            )
            or "error",
            headline=_normalize_text(payload.get("headline"), max_chars=_MAX_HEADLINE_CHARS, fallback="") or "",
            details=details,
            state_fields=tuple(normalized_fields),
            health_status=_normalize_text(payload.get("health_status"), max_chars=_MAX_STATUS_CHARS),
            snapshot_status=_normalize_text(payload.get("snapshot_status"), max_chars=_MAX_STATUS_CHARS),
            snapshot_stale=bool(payload.get("snapshot_stale", False)),
            snapshot_error=_normalize_text(payload.get("snapshot_error"), max_chars=_MAX_ERROR_CHARS),
            runtime_error=_normalize_text(payload.get("runtime_error"), max_chars=_MAX_ERROR_CHARS),
            operator_status=_normalize_text(
                payload.get("operator_status"),
                max_chars=_MAX_OPERATOR_STATUS_CHARS,
            ),
            operator_reason_codes=tuple(operator_reason_codes),
            operator_reason_detail=_normalize_text(
                payload.get("operator_reason_detail"),
                max_chars=_MAX_REASON_DETAIL_CHARS,
            ),
        )


@dataclass(slots=True)
class DisplayRenderStateStore:
    """Read and write the bounded rendered-display-state artifact."""

    path: Path
    file_mode: int = _DEFAULT_FILE_MODE
    max_artifact_bytes: int = _MAX_ARTIFACT_BYTES

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayRenderStateStore":
        project_root = Path(config.project_root).expanduser().resolve()
        return cls(project_root / _DISPLAY_RENDER_STATE_RELATIVE_PATH)

    def load(self) -> DisplayRenderStateSnapshot | None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
        return DisplayRenderStateSnapshot.from_dict(payload)

    def save(
        self,
        *,
        layout: str,
        runtime_status: str,
        headline: str,
        details: tuple[str, ...],
        state_fields: tuple[tuple[str, str], ...],
        health_status: str | None,
        snapshot_status: str | None,
        snapshot_stale: bool,
        snapshot_error: str | None,
        runtime_error: str | None,
        operator_status: str | None,
        operator_reason_codes: tuple[str, ...],
        operator_reason_detail: str | None,
    ) -> DisplayRenderStateSnapshot:
        normalized_fields: list[DisplayRenderStateField] = []
        for label, value in state_fields[:_MAX_STATE_FIELDS]:
            normalized = DisplayRenderStateField.from_pair(label, value)
            if normalized is not None:
                normalized_fields.append(normalized)
        snapshot = DisplayRenderStateSnapshot(
            rendered_at=_utc_now().isoformat(),
            layout=_normalize_text(layout, max_chars=_MAX_LAYOUT_CHARS, fallback="default") or "default",
            runtime_status=_normalize_text(runtime_status, max_chars=_MAX_STATUS_CHARS, fallback="error") or "error",
            headline=_normalize_text(headline, max_chars=_MAX_HEADLINE_CHARS, fallback="") or "",
            details=_normalize_details(details),
            state_fields=tuple(normalized_fields),
            health_status=_normalize_text(health_status, max_chars=_MAX_STATUS_CHARS),
            snapshot_status=_normalize_text(snapshot_status, max_chars=_MAX_STATUS_CHARS),
            snapshot_stale=bool(snapshot_stale),
            snapshot_error=_normalize_text(snapshot_error, max_chars=_MAX_ERROR_CHARS),
            runtime_error=_normalize_text(runtime_error, max_chars=_MAX_ERROR_CHARS),
            operator_status=_normalize_text(operator_status, max_chars=_MAX_OPERATOR_STATUS_CHARS),
            operator_reason_codes=tuple(
                normalized
                for normalized in (
                    _normalize_text(value, max_chars=_MAX_REASON_CODE_CHARS)
                    for value in operator_reason_codes[:_MAX_REASON_CODES]
                )
                if normalized
            ),
            operator_reason_detail=_normalize_text(operator_reason_detail, max_chars=_MAX_REASON_DETAIL_CHARS),
        )
        payload = json.dumps(snapshot.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        if len(payload) > self.max_artifact_bytes:
            raise ValueError("Display render-state artifact exceeds the configured size limit.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent))
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_path, self.file_mode)
            os.replace(temp_path, self.path)
            _fsync_directory(self.path.parent)
        finally:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
        return snapshot
