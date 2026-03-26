"""Persist active service-connect reserve cues for Twinr's HDMI info lane.

Service onboarding is stronger than ordinary ambient reserve content because
the user may need to scan a QR code or read a short status while they are in a
voice-triggered connect flow. This module keeps that right-lane payload
bounded, file-backed, and independent from the generic runtime snapshot.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path

from twinr.agent.base_agent.config import TwinrConfig


_DEFAULT_SERVICE_CONNECT_TTL_S = 20.0
_DEFAULT_SERVICE_CONNECT_PATH = "artifacts/stores/ops/display_service_connect.json"
_ALLOWED_ACCENTS = frozenset({"neutral", "info", "success", "warm", "alert"})

_LOGGER = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Return the current UTC wall clock."""

    return datetime.now(timezone.utc)


def _normalize_timestamp(value: object | None) -> datetime | None:
    """Parse one optional timestamp into an aware UTC datetime."""

    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    """Serialize one aware timestamp as UTC ISO-8601 text."""

    return value.astimezone(timezone.utc).isoformat()


def _compact_text(value: object | None, *, max_len: int) -> str:
    """Normalize one bounded display text field."""

    if value is None:
        return ""
    text = "".join(ch if ch.isprintable() else " " for ch in str(value))
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _normalize_accent(value: object | None) -> str:
    """Normalize one optional visual accent token."""

    compact = _compact_text(value, max_len=24).lower().replace("-", "_").replace(" ", "_")
    if compact not in _ALLOWED_ACCENTS:
        return "info"
    return compact


@dataclass(frozen=True, slots=True)
class DisplayServiceConnectCue:
    """Describe one active service-connect reserve cue for the HDMI panel."""

    source: str = "service_connect"
    updated_at: str | None = None
    expires_at: str | None = None
    service_id: str = ""
    service_label: str = ""
    phase: str = ""
    summary: str = ""
    detail: str = ""
    qr_image_data_url: str | None = None
    accent: str = "info"

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _compact_text(self.source, max_len=80) or "service_connect")
        object.__setattr__(self, "service_id", _compact_text(self.service_id, max_len=48).casefold())
        object.__setattr__(self, "service_label", _compact_text(self.service_label, max_len=48))
        object.__setattr__(self, "phase", _compact_text(self.phase, max_len=48).lower())
        object.__setattr__(self, "summary", _compact_text(self.summary, max_len=96))
        object.__setattr__(self, "detail", _compact_text(self.detail, max_len=180))
        object.__setattr__(
            self,
            "qr_image_data_url",
            _compact_text(self.qr_image_data_url, max_len=100_000) or None,
        )
        object.__setattr__(self, "accent", _normalize_accent(self.accent))

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        fallback_updated_at: datetime | None = None,
        default_ttl_s: float = _DEFAULT_SERVICE_CONNECT_TTL_S,
    ) -> "DisplayServiceConnectCue":
        """Build one normalized service-connect cue from JSON-style data."""

        safe_now = (fallback_updated_at or _utc_now()).astimezone(timezone.utc)
        updated_at = _normalize_timestamp(payload.get("updated_at")) or safe_now
        expires_at = _normalize_timestamp(payload.get("expires_at"))
        if expires_at is None:
            expires_at = updated_at + timedelta(seconds=max(0.1, float(default_ttl_s)))
        return cls(
            source=_compact_text(payload.get("source"), max_len=80) or "service_connect",
            updated_at=_format_timestamp(updated_at),
            expires_at=_format_timestamp(expires_at),
            service_id=_compact_text(payload.get("service_id"), max_len=48).casefold(),
            service_label=_compact_text(payload.get("service_label"), max_len=48),
            phase=_compact_text(payload.get("phase"), max_len=48).lower(),
            summary=_compact_text(payload.get("summary"), max_len=96),
            detail=_compact_text(payload.get("detail"), max_len=180),
            qr_image_data_url=_compact_text(payload.get("qr_image_data_url"), max_len=100_000) or None,
            accent=_normalize_accent(payload.get("accent")),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the cue into a JSON-safe mapping."""

        return asdict(self)

    def is_active(self, *, now: datetime | None = None) -> bool:
        """Return whether the cue should still affect the HDMI scene."""

        expires_at = _normalize_timestamp(self.expires_at)
        if expires_at is None:
            return True
        return expires_at >= (now or _utc_now()).astimezone(timezone.utc)

    def signature(self) -> tuple[object, ...]:
        """Return one stable render-signature fragment for this cue."""

        qr_signature = None
        if self.qr_image_data_url:
            qr_signature = (len(self.qr_image_data_url), self.qr_image_data_url[:64])
        return (
            self.source,
            self.updated_at,
            self.expires_at,
            self.service_id,
            self.service_label,
            self.phase,
            self.summary,
            self.detail,
            qr_signature,
            self.accent,
        )


@dataclass(slots=True)
class DisplayServiceConnectCueStore:
    """Read and write the optional HDMI service-connect cue artifact."""

    path: Path
    default_ttl_s: float = _DEFAULT_SERVICE_CONNECT_TTL_S

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayServiceConnectCueStore":
        """Resolve the service-connect cue path from Twinr configuration."""

        project_root = Path(config.project_root).expanduser().resolve()
        configured_path = Path(
            getattr(config, "display_service_connect_path", _DEFAULT_SERVICE_CONNECT_PATH)
            or _DEFAULT_SERVICE_CONNECT_PATH
        )
        resolved_path = configured_path if configured_path.is_absolute() else project_root / configured_path
        return cls(path=resolved_path)

    def load(self) -> DisplayServiceConnectCue | None:
        """Load the current service-connect cue, if one exists and parses."""

        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            _LOGGER.warning("Failed to read display service-connect cue from %s.", self.path, exc_info=True)
            return None
        if not isinstance(payload, dict):
            _LOGGER.warning(
                "Ignoring invalid display service-connect cue payload at %s because it is not an object.",
                self.path,
            )
            return None
        fallback_updated_at = datetime.fromtimestamp(self.path.stat().st_mtime, tz=timezone.utc)
        try:
            return DisplayServiceConnectCue.from_dict(
                payload,
                fallback_updated_at=fallback_updated_at,
                default_ttl_s=self.default_ttl_s,
            )
        except Exception:
            _LOGGER.warning("Ignoring invalid display service-connect cue payload at %s.", self.path, exc_info=True)
            return None

    def load_active(self, *, now: datetime | None = None) -> DisplayServiceConnectCue | None:
        """Load the current cue only when it is still active."""

        cue = self.load()
        if cue is None or not cue.is_active(now=now):
            return None
        return cue

    def save(
        self,
        cue: DisplayServiceConnectCue,
        *,
        hold_seconds: float | None = None,
        now: datetime | None = None,
    ) -> DisplayServiceConnectCue:
        """Persist one service-connect cue with a bounded expiry."""

        written_at = (now or _utc_now()).astimezone(timezone.utc)
        ttl_s = self.default_ttl_s if hold_seconds is None else max(0.1, float(hold_seconds))
        normalized = DisplayServiceConnectCue(
            source=cue.source,
            updated_at=_format_timestamp(written_at),
            expires_at=_format_timestamp(written_at + timedelta(seconds=ttl_s)),
            service_id=cue.service_id,
            service_label=cue.service_label,
            phase=cue.phase,
            summary=cue.summary,
            detail=cue.detail,
            qr_image_data_url=cue.qr_image_data_url,
            accent=cue.accent,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(normalized.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return normalized

    def clear(self) -> None:
        """Remove the persisted cue artifact when it exists."""

        try:
            self.path.unlink()
        except FileNotFoundError:
            return


__all__ = [
    "DisplayServiceConnectCue",
    "DisplayServiceConnectCueStore",
]
