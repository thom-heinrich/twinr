"""Persist short ambient display impulses for Twinr's HDMI reserve area.

The default HDMI scene already has a dedicated right-hand reserve area used by
emoji acknowledgements and richer presentation cards. This module adds a small
text-plus-emoji cue contract for quieter companion impulses that should remain
outside the generic runtime snapshot schema.

The contract is intentionally bounded:

- one active impulse at a time
- short text fields only
- positive visual accents
- explicit expiry so stale companion nudges disappear on their own
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path

from twinr.agent.base_agent.config import TwinrConfig
from twinr.display.emoji_cues import DisplayEmojiSymbol

_DEFAULT_AMBIENT_IMPULSE_TTL_S = 18.0
_DEFAULT_AMBIENT_IMPULSE_PATH = "artifacts/stores/ops/display_ambient_impulse.json"
_ALLOWED_ACCENTS = frozenset({"neutral", "info", "success", "warm"})
_ALLOWED_ACTIONS = frozenset({"hint", "brief_update", "ask_one", "invite_follow_up"})

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


def _normalize_symbol(value: object | None) -> DisplayEmojiSymbol:
    """Normalize one optional symbol token into a supported emoji symbol."""

    if isinstance(value, DisplayEmojiSymbol):
        return value
    compact = _compact_text(value, max_len=24).lower().replace("-", "_").replace(" ", "_")
    if not compact:
        return DisplayEmojiSymbol.SPARKLES
    try:
        return DisplayEmojiSymbol(compact)
    except ValueError:
        return DisplayEmojiSymbol.SPARKLES


def _normalize_accent(value: object | None) -> str:
    """Normalize one optional visual accent token."""

    compact = _compact_text(value, max_len=24).lower().replace("-", "_").replace(" ", "_")
    if compact not in _ALLOWED_ACCENTS:
        return "info"
    return compact


def _normalize_action(value: object | None) -> str:
    """Normalize one optional engagement-action token."""

    compact = _compact_text(value, max_len=24).lower().replace("-", "_").replace(" ", "_")
    if compact not in _ALLOWED_ACTIONS:
        return "hint"
    return compact


@dataclass(frozen=True, slots=True)
class DisplayAmbientImpulseCue:
    """Describe one active ambient HDMI reserve-card impulse."""

    source: str = "external"
    updated_at: str | None = None
    expires_at: str | None = None
    topic_key: str = ""
    semantic_topic_key: str = ""
    eyebrow: str = ""
    headline: str = ""
    body: str = ""
    symbol: str = DisplayEmojiSymbol.SPARKLES.value
    accent: str = "info"
    action: str = "hint"
    attention_state: str = "background"

    def __post_init__(self) -> None:
        """Normalize direct constructor calls into the canonical cue contract."""

        object.__setattr__(self, "source", _compact_text(self.source, max_len=80) or "external")
        object.__setattr__(self, "topic_key", _compact_text(self.topic_key, max_len=96).casefold())
        semantic_topic_key = _compact_text(self.semantic_topic_key, max_len=96).casefold()
        object.__setattr__(self, "semantic_topic_key", semantic_topic_key or self.topic_key)
        object.__setattr__(self, "eyebrow", _compact_text(self.eyebrow, max_len=36))
        object.__setattr__(self, "headline", _compact_text(self.headline, max_len=128))
        object.__setattr__(self, "body", _compact_text(self.body, max_len=128))
        object.__setattr__(self, "symbol", _normalize_symbol(self.symbol).value)
        object.__setattr__(self, "accent", _normalize_accent(self.accent))
        object.__setattr__(self, "action", _normalize_action(self.action))
        object.__setattr__(self, "attention_state", _compact_text(self.attention_state, max_len=32).lower() or "background")

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        fallback_updated_at: datetime | None = None,
        default_ttl_s: float = _DEFAULT_AMBIENT_IMPULSE_TTL_S,
    ) -> "DisplayAmbientImpulseCue":
        """Build one normalized ambient impulse from JSON-style data."""

        safe_now = (fallback_updated_at or _utc_now()).astimezone(timezone.utc)
        updated_at = _normalize_timestamp(payload.get("updated_at")) or safe_now
        expires_at = _normalize_timestamp(payload.get("expires_at"))
        if expires_at is None:
            expires_at = updated_at + timedelta(seconds=max(0.1, float(default_ttl_s)))
        return cls(
            source=_compact_text(payload.get("source"), max_len=80) or "external",
            updated_at=_format_timestamp(updated_at),
            expires_at=_format_timestamp(expires_at),
            topic_key=_compact_text(payload.get("topic_key"), max_len=96).casefold(),
            semantic_topic_key=_compact_text(payload.get("semantic_topic_key"), max_len=96).casefold(),
            eyebrow=_compact_text(payload.get("eyebrow"), max_len=36),
            headline=_compact_text(payload.get("headline"), max_len=128),
            body=_compact_text(payload.get("body"), max_len=128),
            symbol=_normalize_symbol(payload.get("symbol")).value,
            accent=_normalize_accent(payload.get("accent")),
            action=_normalize_action(payload.get("action")),
            attention_state=_compact_text(payload.get("attention_state"), max_len=32).lower() or "background",
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

        return (
            self.source,
            self.updated_at,
            self.expires_at,
            self.topic_key,
            self.semantic_topic_key,
            self.eyebrow,
            self.headline,
            self.body,
            self.symbol,
            self.accent,
            self.action,
            self.attention_state,
        )

    def glyph(self) -> str:
        """Return the rendered Unicode glyph for this cue."""

        return _normalize_symbol(self.symbol).glyph()


@dataclass(slots=True)
class DisplayAmbientImpulseCueStore:
    """Read and write the optional ambient-impulse cue artifact."""

    path: Path
    default_ttl_s: float = _DEFAULT_AMBIENT_IMPULSE_TTL_S

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayAmbientImpulseCueStore":
        """Resolve the ambient-impulse cue path from Twinr configuration."""

        project_root = Path(config.project_root).expanduser().resolve()
        configured_path = Path(
            getattr(config, "display_ambient_impulse_path", _DEFAULT_AMBIENT_IMPULSE_PATH)
            or _DEFAULT_AMBIENT_IMPULSE_PATH
        )
        resolved_path = configured_path if configured_path.is_absolute() else project_root / configured_path
        return cls(
            path=resolved_path,
            default_ttl_s=float(
                getattr(
                    config,
                    "display_ambient_impulse_ttl_s",
                    _DEFAULT_AMBIENT_IMPULSE_TTL_S,
                )
                or _DEFAULT_AMBIENT_IMPULSE_TTL_S
            ),
        )

    def load(self) -> DisplayAmbientImpulseCue | None:
        """Load the current ambient impulse cue, if one exists and parses."""

        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            _LOGGER.warning("Failed to read display ambient impulse cue from %s.", self.path, exc_info=True)
            return None
        if not isinstance(payload, dict):
            _LOGGER.warning(
                "Ignoring invalid display ambient impulse cue payload at %s because it is not an object.",
                self.path,
            )
            return None
        fallback_updated_at = datetime.fromtimestamp(self.path.stat().st_mtime, tz=timezone.utc)
        try:
            return DisplayAmbientImpulseCue.from_dict(
                payload,
                fallback_updated_at=fallback_updated_at,
                default_ttl_s=self.default_ttl_s,
            )
        except Exception:
            _LOGGER.warning("Ignoring invalid display ambient impulse cue payload at %s.", self.path, exc_info=True)
            return None

    def load_active(self, *, now: datetime | None = None) -> DisplayAmbientImpulseCue | None:
        """Load the current cue only when it is still active."""

        cue = self.load()
        if cue is None or not cue.is_active(now=now):
            return None
        return cue

    def save(
        self,
        cue: DisplayAmbientImpulseCue,
        *,
        hold_seconds: float | None = None,
        now: datetime | None = None,
    ) -> DisplayAmbientImpulseCue:
        """Persist one ambient impulse cue with a bounded expiry."""

        written_at = (now or _utc_now()).astimezone(timezone.utc)
        ttl_s = self.default_ttl_s if hold_seconds is None else max(0.1, float(hold_seconds))
        normalized = DisplayAmbientImpulseCue(
            source=cue.source,
            updated_at=_format_timestamp(written_at),
            expires_at=_format_timestamp(written_at + timedelta(seconds=ttl_s)),
            topic_key=cue.topic_key,
            semantic_topic_key=cue.semantic_topic_key,
            eyebrow=cue.eyebrow,
            headline=cue.headline,
            body=cue.body,
            symbol=cue.symbol,
            accent=cue.accent,
            action=cue.action,
            attention_state=cue.attention_state,
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


@dataclass(slots=True)
class DisplayAmbientImpulseController:
    """Persist short-lived ambient impulses for the HDMI reserve card."""

    store: DisplayAmbientImpulseCueStore
    default_source: str = "ambient_impulse"

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        default_source: str = "ambient_impulse",
    ) -> "DisplayAmbientImpulseController":
        """Build one controller from the configured cue store."""

        return cls(
            store=DisplayAmbientImpulseCueStore.from_config(config),
            default_source=_compact_text(default_source, max_len=80) or "ambient_impulse",
        )

    def show_impulse(
        self,
        *,
        topic_key: str,
        semantic_topic_key: str | None = None,
        eyebrow: str,
        headline: str,
        body: str,
        symbol: DisplayEmojiSymbol | str = DisplayEmojiSymbol.SPARKLES,
        accent: str = "info",
        action: str = "hint",
        attention_state: str = "background",
        hold_seconds: float | None = None,
        source: str | None = None,
        now: datetime | None = None,
    ) -> DisplayAmbientImpulseCue:
        """Persist one active ambient impulse cue."""

        cue = DisplayAmbientImpulseCue(
            source=_compact_text(source, max_len=80) or self.default_source,
            topic_key=topic_key,
            semantic_topic_key=_compact_text(semantic_topic_key, max_len=96).casefold(),
            eyebrow=eyebrow,
            headline=headline,
            body=body,
            symbol=_normalize_symbol(symbol).value,
            accent=accent,
            action=action,
            attention_state=attention_state,
        )
        return self.store.save(
            cue,
            hold_seconds=hold_seconds,
            now=now,
        )
