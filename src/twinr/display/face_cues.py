"""Persist optional HDMI face-expression cues for the display loop.

The senior-facing HDMI face should react to explicit external triggers without
teaching the generic runtime snapshot schema about gaze, brows, or mouth
semantics. This module stores one small optional cue payload that the display
loop can merge into the active status animation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path

from twinr.agent.base_agent.config import TwinrConfig


_DEFAULT_FACE_CUE_TTL_S = 4.0
_DEFAULT_FACE_CUE_PATH = "artifacts/stores/ops/display_face_cue.json"
_MAX_AXIS = 2
_MOUTH_ALIASES = {
    "concern": "sad",
    "line": "neutral",
}
_ALLOWED_MOUTHS = frozenset(
    {
        "neutral",
        "smile",
        "sad",
        "thinking",
        "pursed",
        "scrunched",
        "open",
        "speak",
    }
)
_BROW_ALIASES = {
    "concern": "inward_tilt",
    "flat": "straight",
    "focus": "inward_tilt",
}
_ALLOWED_BROWS = frozenset(
    {
        "straight",
        "inward_tilt",
        "outward_tilt",
        "roof",
        "raised",
        "soft",
    }
)

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
    """Serialize one aware timestamp as ISO-8601 text."""

    return value.astimezone(timezone.utc).isoformat()


def _clamp_axis(value: object | None) -> int:
    """Normalize one signed cue-axis value into the supported range."""

    try:
        parsed = int(round(float(value or 0)))
    except (TypeError, ValueError):
        return 0
    return max(-_MAX_AXIS, min(_MAX_AXIS, parsed))


def _normalize_style(
    value: object | None,
    *,
    allowed: frozenset[str],
    aliases: Mapping[str, str] | None = None,
) -> str | None:
    """Normalize one optional face-style label against an explicit allow-list."""

    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        return None
    if aliases is not None:
        text = aliases.get(text, text)
    if text not in allowed:
        return None
    return text


def _normalize_optional_bool(value: object | None) -> bool | None:
    """Normalize one optional boolean face flag."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


@dataclass(frozen=True, slots=True)
class DisplayFaceCue:
    """Describe one optional external face-expression cue."""

    source: str = "external"
    updated_at: str | None = None
    expires_at: str | None = None
    gaze_x: int = 0
    gaze_y: int = 0
    head_dx: int = 0
    head_dy: int = 0
    mouth: str | None = None
    brows: str | None = None
    blink: bool | None = None

    def __post_init__(self) -> None:
        """Normalize direct constructor calls into the canonical cue vocabulary."""

        object.__setattr__(self, "source", str(self.source or "").strip() or "external")
        object.__setattr__(self, "gaze_x", _clamp_axis(self.gaze_x))
        object.__setattr__(self, "gaze_y", _clamp_axis(self.gaze_y))
        object.__setattr__(self, "head_dx", _clamp_axis(self.head_dx))
        object.__setattr__(self, "head_dy", _clamp_axis(self.head_dy))
        object.__setattr__(self, "mouth", _normalize_style(self.mouth, allowed=_ALLOWED_MOUTHS, aliases=_MOUTH_ALIASES))
        object.__setattr__(self, "brows", _normalize_style(self.brows, allowed=_ALLOWED_BROWS, aliases=_BROW_ALIASES))
        object.__setattr__(self, "blink", _normalize_optional_bool(self.blink))

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        fallback_updated_at: datetime | None = None,
        default_ttl_s: float = _DEFAULT_FACE_CUE_TTL_S,
    ) -> "DisplayFaceCue":
        """Build one normalized face cue from JSON-style data."""

        safe_now = (fallback_updated_at or _utc_now()).astimezone(timezone.utc)
        updated_at = _normalize_timestamp(payload.get("updated_at")) or safe_now
        expires_at = _normalize_timestamp(payload.get("expires_at"))
        if expires_at is None:
            expires_at = updated_at + timedelta(seconds=max(0.1, float(default_ttl_s)))
        source = str(payload.get("source", "") or "").strip() or "external"
        return cls(
            source=source,
            updated_at=_format_timestamp(updated_at),
            expires_at=_format_timestamp(expires_at),
            gaze_x=_clamp_axis(payload.get("gaze_x")),
            gaze_y=_clamp_axis(payload.get("gaze_y")),
            head_dx=_clamp_axis(payload.get("head_dx")),
            head_dy=_clamp_axis(payload.get("head_dy")),
            mouth=_normalize_style(payload.get("mouth"), allowed=_ALLOWED_MOUTHS, aliases=_MOUTH_ALIASES),
            brows=_normalize_style(payload.get("brows"), allowed=_ALLOWED_BROWS, aliases=_BROW_ALIASES),
            blink=_normalize_optional_bool(payload.get("blink")),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the cue into a JSON-safe mapping."""

        return asdict(self)

    def is_active(self, *, now: datetime | None = None) -> bool:
        """Return whether the cue should still influence the HDMI face."""

        expires_at = _normalize_timestamp(self.expires_at)
        if expires_at is None:
            return True
        return expires_at >= (now or _utc_now()).astimezone(timezone.utc)

    def signature(self) -> tuple[object, ...]:
        """Return a stable render-signature fragment for this cue."""

        return (
            self.source,
            self.updated_at,
            self.expires_at,
            self.gaze_x,
            self.gaze_y,
            self.head_dx,
            self.head_dy,
            self.mouth,
            self.brows,
            self.blink,
        )


@dataclass(slots=True)
class DisplayFaceCueStore:
    """Read and write the optional face-cue artifact."""

    path: Path
    default_ttl_s: float = _DEFAULT_FACE_CUE_TTL_S

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayFaceCueStore":
        """Resolve the face-cue path from Twinr configuration."""

        project_root = Path(config.project_root).expanduser().resolve()
        configured_path = Path(
            getattr(config, "display_face_cue_path", _DEFAULT_FACE_CUE_PATH) or _DEFAULT_FACE_CUE_PATH
        )
        resolved_path = configured_path if configured_path.is_absolute() else project_root / configured_path
        return cls(path=resolved_path, default_ttl_s=float(getattr(config, "display_face_cue_ttl_s", _DEFAULT_FACE_CUE_TTL_S) or _DEFAULT_FACE_CUE_TTL_S))

    def load(self) -> DisplayFaceCue | None:
        """Load the current face cue, if one exists and parses."""

        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            _LOGGER.warning("Failed to read display face cue from %s.", self.path, exc_info=True)
            return None
        if not isinstance(payload, dict):
            _LOGGER.warning("Ignoring invalid display face cue payload at %s because it is not an object.", self.path)
            return None
        fallback_updated_at = datetime.fromtimestamp(self.path.stat().st_mtime, tz=timezone.utc)
        try:
            return DisplayFaceCue.from_dict(
                payload,
                fallback_updated_at=fallback_updated_at,
                default_ttl_s=self.default_ttl_s,
            )
        except Exception:
            _LOGGER.warning("Ignoring invalid display face cue payload at %s.", self.path, exc_info=True)
            return None

    def load_active(self, *, now: datetime | None = None) -> DisplayFaceCue | None:
        """Load the current cue only when it is still active."""

        cue = self.load()
        if cue is None or not cue.is_active(now=now):
            return None
        return cue

    def save(
        self,
        cue: DisplayFaceCue,
        *,
        hold_seconds: float | None = None,
        now: datetime | None = None,
    ) -> DisplayFaceCue:
        """Persist one normalized face cue atomically."""

        effective_now = (now or _utc_now()).astimezone(timezone.utc)
        effective_ttl_s = max(0.1, float(self.default_ttl_s if hold_seconds is None else hold_seconds))
        normalized = DisplayFaceCue.from_dict(
            cue.to_dict(),
            fallback_updated_at=effective_now,
            default_ttl_s=effective_ttl_s,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.parent / f".{self.path.name}.tmp"
        tmp_path.write_text(
            json.dumps(normalized.to_dict(), ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)
        return normalized

    def clear(self) -> None:
        """Remove the current face cue artifact if one exists."""

        try:
            self.path.unlink()
        except FileNotFoundError:
            return
