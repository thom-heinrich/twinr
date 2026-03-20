"""Persist and normalize optional HDMI emoji cues.

This module keeps the right-hand HDMI reserve area on a structured, bounded
cue contract instead of letting random text or ad-hoc renderer state leak into
the main senior-facing surface. Producers can request one real Unicode emoji
such as a thumbs-up or a waving hand without teaching the generic runtime
snapshot schema about emoji semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import json
import logging
from pathlib import Path

from twinr.agent.base_agent.config import TwinrConfig


_DEFAULT_EMOJI_CUE_TTL_S = 6.0
_DEFAULT_EMOJI_CUE_PATH = "artifacts/stores/ops/display_emoji.json"
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
    """Serialize one aware timestamp as ISO-8601 text."""

    return value.astimezone(timezone.utc).isoformat()


def _compact_text(value: object | None, *, max_len: int = 80) -> str:
    """Normalize one bounded text field."""

    if value is None:
        return ""
    text = "".join(ch if ch.isprintable() else " " for ch in str(value))
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


class DisplayEmojiSymbol(str, Enum):
    """Supported right-hand HDMI emoji symbols."""

    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    WAVING_HAND = "waving_hand"
    RAISED_HAND = "raised_hand"
    POINTING_HAND = "pointing_hand"
    OK_HAND = "ok_hand"
    HEART = "heart"
    CHECK = "check"
    QUESTION = "question"
    EXCLAMATION = "exclamation"
    WARNING = "warning"
    SPARKLES = "sparkles"

    def glyph(self) -> str:
        """Return the real Unicode emoji glyph for this symbol."""

        return _EMOJI_GLYPHS[self]


_EMOJI_GLYPHS: dict[DisplayEmojiSymbol, str] = {
    DisplayEmojiSymbol.THUMBS_UP: "👍",
    DisplayEmojiSymbol.THUMBS_DOWN: "👎",
    DisplayEmojiSymbol.WAVING_HAND: "👋",
    DisplayEmojiSymbol.RAISED_HAND: "✋",
    DisplayEmojiSymbol.POINTING_HAND: "👉",
    DisplayEmojiSymbol.OK_HAND: "👌",
    DisplayEmojiSymbol.HEART: "❤️",
    DisplayEmojiSymbol.CHECK: "✅",
    DisplayEmojiSymbol.QUESTION: "❓",
    DisplayEmojiSymbol.EXCLAMATION: "❗",
    DisplayEmojiSymbol.WARNING: "⚠️",
    DisplayEmojiSymbol.SPARKLES: "✨",
}


def _normalize_symbol(value: object | None) -> DisplayEmojiSymbol | None:
    """Normalize one optional emoji symbol token."""

    if value is None:
        return None
    if isinstance(value, DisplayEmojiSymbol):
        return value
    compact = _compact_text(value, max_len=24).lower().replace("-", "_").replace(" ", "_")
    if not compact:
        return None
    try:
        return DisplayEmojiSymbol(compact)
    except ValueError:
        return None


def _normalize_accent(value: object | None) -> str:
    """Normalize one optional emoji accent token."""

    compact = _compact_text(value, max_len=24).lower().replace("-", "_").replace(" ", "_")
    if compact not in _ALLOWED_ACCENTS:
        return "neutral"
    return compact


@dataclass(frozen=True, slots=True)
class DisplayEmojiCue:
    """Describe one optional right-hand HDMI emoji cue."""

    source: str = "external"
    updated_at: str | None = None
    expires_at: str | None = None
    symbol: str = DisplayEmojiSymbol.HEART.value
    accent: str = "neutral"

    def __post_init__(self) -> None:
        """Normalize direct constructor calls into the canonical cue contract."""

        object.__setattr__(self, "source", _compact_text(self.source, max_len=80) or "external")
        normalized_symbol = _normalize_symbol(self.symbol) or DisplayEmojiSymbol.HEART
        object.__setattr__(self, "symbol", normalized_symbol.value)
        object.__setattr__(self, "accent", _normalize_accent(self.accent))

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        fallback_updated_at: datetime | None = None,
        default_ttl_s: float = _DEFAULT_EMOJI_CUE_TTL_S,
    ) -> "DisplayEmojiCue":
        """Build one normalized emoji cue from JSON-style data."""

        safe_now = (fallback_updated_at or _utc_now()).astimezone(timezone.utc)
        updated_at = _normalize_timestamp(payload.get("updated_at")) or safe_now
        expires_at = _normalize_timestamp(payload.get("expires_at"))
        if expires_at is None:
            expires_at = updated_at + timedelta(seconds=max(0.1, float(default_ttl_s)))
        symbol = _normalize_symbol(payload.get("symbol")) or DisplayEmojiSymbol.HEART
        return cls(
            source=_compact_text(payload.get("source"), max_len=80) or "external",
            updated_at=_format_timestamp(updated_at),
            expires_at=_format_timestamp(expires_at),
            symbol=symbol.value,
            accent=_normalize_accent(payload.get("accent")),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the cue into a JSON-safe mapping."""

        return asdict(self)

    def is_active(self, *, now: datetime | None = None) -> bool:
        """Return whether the cue should still influence the HDMI scene."""

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
            self.symbol,
            self.accent,
        )

    def glyph(self) -> str:
        """Return the rendered Unicode glyph for this cue."""

        return (_normalize_symbol(self.symbol) or DisplayEmojiSymbol.HEART).glyph()


@dataclass(slots=True)
class DisplayEmojiCueStore:
    """Read and write the optional HDMI emoji cue artifact."""

    path: Path
    default_ttl_s: float = _DEFAULT_EMOJI_CUE_TTL_S

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayEmojiCueStore":
        """Resolve the emoji-cue path from Twinr configuration."""

        project_root = Path(config.project_root).expanduser().resolve()
        configured_path = Path(
            getattr(config, "display_emoji_cue_path", _DEFAULT_EMOJI_CUE_PATH) or _DEFAULT_EMOJI_CUE_PATH
        )
        resolved_path = configured_path if configured_path.is_absolute() else project_root / configured_path
        return cls(
            path=resolved_path,
            default_ttl_s=float(
                getattr(config, "display_emoji_cue_ttl_s", _DEFAULT_EMOJI_CUE_TTL_S) or _DEFAULT_EMOJI_CUE_TTL_S
            ),
        )

    def load(self) -> DisplayEmojiCue | None:
        """Load the current emoji cue, if one exists and parses."""

        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            _LOGGER.warning("Failed to read display emoji cue from %s.", self.path, exc_info=True)
            return None
        if not isinstance(payload, dict):
            _LOGGER.warning("Ignoring invalid display emoji cue payload at %s because it is not an object.", self.path)
            return None
        fallback_updated_at = datetime.fromtimestamp(self.path.stat().st_mtime, tz=timezone.utc)
        try:
            return DisplayEmojiCue.from_dict(
                payload,
                fallback_updated_at=fallback_updated_at,
                default_ttl_s=self.default_ttl_s,
            )
        except Exception:
            _LOGGER.warning("Ignoring invalid display emoji cue payload at %s.", self.path, exc_info=True)
            return None

    def load_active(self, *, now: datetime | None = None) -> DisplayEmojiCue | None:
        """Load the current emoji cue only when it is still active."""

        cue = self.load()
        if cue is None or not cue.is_active(now=now):
            return None
        return cue

    def save(
        self,
        cue: DisplayEmojiCue,
        *,
        hold_seconds: float | None = None,
        now: datetime | None = None,
    ) -> DisplayEmojiCue:
        """Persist one normalized emoji cue atomically."""

        safe_now = (now or _utc_now()).astimezone(timezone.utc)
        ttl_s = max(0.1, float(self.default_ttl_s if hold_seconds is None else hold_seconds))
        normalized = DisplayEmojiCue(
            source=cue.source,
            updated_at=_format_timestamp(safe_now),
            expires_at=_format_timestamp(safe_now + timedelta(seconds=ttl_s)),
            symbol=cue.symbol,
            accent=cue.accent,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp_path.write_text(
            json.dumps(normalized.to_dict(), ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)
        return normalized

    def clear(self) -> None:
        """Remove the current emoji cue artifact if one exists."""

        try:
            self.path.unlink()
        except FileNotFoundError:
            return


@dataclass(slots=True)
class DisplayEmojiController:
    """Persist producer-facing HDMI emoji cues through the cue store."""

    store: DisplayEmojiCueStore
    default_source: str = "external"

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        default_source: str = "external",
    ) -> "DisplayEmojiController":
        """Resolve the configured emoji cue store and build one controller."""

        return cls(store=DisplayEmojiCueStore.from_config(config), default_source=default_source)

    def show_symbol(
        self,
        symbol: DisplayEmojiSymbol | str,
        *,
        accent: str = "neutral",
        source: str | None = None,
        hold_seconds: float | None = None,
        now: datetime | None = None,
    ) -> DisplayEmojiCue:
        """Persist one explicit emoji symbol cue."""

        normalized_symbol = _normalize_symbol(symbol) or DisplayEmojiSymbol.HEART
        return self.store.save(
            DisplayEmojiCue(
                source=source or self.default_source,
                symbol=normalized_symbol.value,
                accent=accent,
            ),
            hold_seconds=hold_seconds,
            now=now,
        )

    def clear(self) -> None:
        """Remove the currently active emoji cue."""

        self.store.clear()
