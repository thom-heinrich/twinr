"""Persist and produce optional HDMI presentation cues.

The default HDMI scene should be able to expand beyond the small status panel
without teaching the generic runtime snapshot schema about images, focus cards,
or other presentation-only concepts. This module stores one bounded optional
presentation payload and offers a producer-facing controller so other Twinr
modules can trigger it cleanly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path

from twinr.agent.base_agent.config import TwinrConfig


_DEFAULT_PRESENTATION_TTL_S = 20.0
_DEFAULT_PRESENTATION_PATH = "artifacts/stores/ops/display_presentation.json"
_ALLOWED_KINDS = frozenset({"image", "rich_card"})
_ALLOWED_ACCENTS = frozenset({"alert", "info", "success", "warm"})
_ALLOWED_FACE_EMOTIONS = frozenset({"calm", "happy", "sad", "thoughtful", "curious", "focused"})
_MAX_CARDS = 4
_MAX_BODY_LINES = 5
_MAX_TEXT_LEN = 140
_MORPH_DURATION_S = 0.85

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


def _compact_text(value: object | None, *, max_len: int = _MAX_TEXT_LEN) -> str:
    """Normalize one bounded display string."""

    if value is None:
        return ""
    text = "".join(ch if ch.isprintable() else " " for ch in str(value))
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _normalize_kind(value: object | None) -> str:
    """Normalize one presentation kind label."""

    text = _compact_text(value, max_len=24).lower().replace("-", "_").replace(" ", "_")
    if text not in _ALLOWED_KINDS:
        return "rich_card"
    return text


def _normalize_accent(value: object | None) -> str:
    """Normalize one accent label."""

    text = _compact_text(value, max_len=24).lower().replace("-", "_").replace(" ", "_")
    if text not in _ALLOWED_ACCENTS:
        return "info"
    return text


def _normalize_body_lines(value: object | None) -> tuple[str, ...]:
    """Normalize one bounded body-line list."""

    if value is None:
        return ()
    if isinstance(value, str):
        candidates: Sequence[object] = (value,)
    elif isinstance(value, Sequence):
        candidates = value
    else:
        return ()
    lines: list[str] = []
    for item in candidates:
        compact = _compact_text(item)
        if compact:
            lines.append(compact)
        if len(lines) >= _MAX_BODY_LINES:
            break
    return tuple(lines)


def _normalize_image_path(value: object | None) -> str | None:
    """Normalize one optional local image path."""

    compact = _compact_text(value, max_len=512)
    if not compact:
        return None
    return str(Path(compact).expanduser())


def _normalize_card_key(value: object | None, *, fallback: str = "primary") -> str:
    """Normalize one short presentation-card key for durable storage."""

    compact = _compact_text(value, max_len=48).lower().replace("-", "_").replace(" ", "_")
    return compact or fallback


def _normalize_priority(value: object | None, *, default: int = 100) -> int:
    """Normalize one bounded card priority."""

    try:
        parsed = int(round(float(default if value is None else value)))
    except (TypeError, ValueError):
        return default
    return max(0, min(999, parsed))


def _normalize_face_emotion(value: object | None) -> str | None:
    """Normalize one optional face-emotion label for presentation sync."""

    compact = _compact_text(value, max_len=24).lower().replace("-", "_").replace(" ", "_")
    if compact not in _ALLOWED_FACE_EMOTIONS:
        return None
    return compact


@dataclass(frozen=True, slots=True)
class DisplayPresentationCardCue:
    """Describe one bounded presentation card inside the active HDMI scene."""

    key: str = "primary"
    kind: str = "rich_card"
    title: str = ""
    subtitle: str = ""
    body_lines: tuple[str, ...] = ()
    image_path: str | None = None
    accent: str = "info"
    priority: int = 100
    face_emotion: str | None = None

    def __post_init__(self) -> None:
        """Normalize direct constructor calls into the canonical card contract."""

        object.__setattr__(self, "key", _normalize_card_key(self.key))
        object.__setattr__(self, "kind", _normalize_kind(self.kind))
        object.__setattr__(self, "title", _compact_text(self.title))
        object.__setattr__(self, "subtitle", _compact_text(self.subtitle))
        object.__setattr__(self, "body_lines", _normalize_body_lines(self.body_lines))
        object.__setattr__(self, "image_path", _normalize_image_path(self.image_path))
        object.__setattr__(self, "accent", _normalize_accent(self.accent))
        object.__setattr__(self, "priority", _normalize_priority(self.priority))
        object.__setattr__(self, "face_emotion", _normalize_face_emotion(self.face_emotion))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object], *, fallback_key: str = "primary") -> "DisplayPresentationCardCue":
        """Build one normalized presentation card from JSON-style data."""

        return cls(
            key=_normalize_card_key(payload.get("key"), fallback=fallback_key),
            kind=_normalize_kind(payload.get("kind")),
            title=_compact_text(payload.get("title")),
            subtitle=_compact_text(payload.get("subtitle")),
            body_lines=_normalize_body_lines(payload.get("body_lines") or payload.get("body")),
            image_path=_normalize_image_path(payload.get("image_path")),
            accent=_normalize_accent(payload.get("accent")),
            priority=_normalize_priority(payload.get("priority")),
            face_emotion=_normalize_face_emotion(payload.get("face_emotion")),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the card into a JSON-safe mapping."""

        return asdict(self)

    def signature(self) -> tuple[object, ...]:
        """Return one stable signature fragment for this card."""

        return (
            self.key,
            self.kind,
            self.title,
            self.subtitle,
            self.body_lines,
            self.image_path,
            self.accent,
            self.priority,
            self.face_emotion,
        )


def _normalize_cards(value: object | None) -> tuple[DisplayPresentationCardCue, ...]:
    """Normalize one optional card sequence into bounded card cues."""

    if value is None:
        return ()
    if isinstance(value, DisplayPresentationCardCue):
        return (value,)
    if isinstance(value, Mapping):
        return (DisplayPresentationCardCue.from_dict(value),)
    if isinstance(value, str) or not isinstance(value, Sequence):
        return ()
    cards: list[DisplayPresentationCardCue] = []
    for index, item in enumerate(value):
        if len(cards) >= _MAX_CARDS:
            break
        fallback_key = f"card_{index + 1}"
        if isinstance(item, DisplayPresentationCardCue):
            cards.append(item)
            continue
        if isinstance(item, Mapping):
            cards.append(DisplayPresentationCardCue.from_dict(item, fallback_key=fallback_key))
    return tuple(cards)


@dataclass(frozen=True, slots=True)
class DisplayPresentationCue:
    """Describe one optional HDMI presentation overlay."""

    source: str = "external"
    updated_at: str | None = None
    expires_at: str | None = None
    kind: str = "rich_card"
    title: str = ""
    subtitle: str = ""
    body_lines: tuple[str, ...] = ()
    image_path: str | None = None
    accent: str = "info"
    cards: tuple[DisplayPresentationCardCue, ...] = ()
    active_card_key: str | None = None

    def __post_init__(self) -> None:
        """Normalize direct constructor calls into the canonical cue contract."""

        object.__setattr__(self, "source", _compact_text(self.source, max_len=80) or "external")
        object.__setattr__(self, "kind", _normalize_kind(self.kind))
        object.__setattr__(self, "title", _compact_text(self.title))
        object.__setattr__(self, "subtitle", _compact_text(self.subtitle))
        object.__setattr__(self, "body_lines", _normalize_body_lines(self.body_lines))
        object.__setattr__(self, "image_path", _normalize_image_path(self.image_path))
        object.__setattr__(self, "accent", _normalize_accent(self.accent))
        object.__setattr__(self, "cards", _normalize_cards(self.cards))
        object.__setattr__(
            self,
            "active_card_key",
            _normalize_card_key(self.active_card_key, fallback="") or None,
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        fallback_updated_at: datetime | None = None,
        default_ttl_s: float = _DEFAULT_PRESENTATION_TTL_S,
    ) -> "DisplayPresentationCue":
        """Build one normalized presentation cue from JSON-style data."""

        safe_now = (fallback_updated_at or _utc_now()).astimezone(timezone.utc)
        updated_at = _normalize_timestamp(payload.get("updated_at")) or safe_now
        expires_at = _normalize_timestamp(payload.get("expires_at"))
        if expires_at is None:
            expires_at = updated_at + timedelta(seconds=max(0.1, float(default_ttl_s)))
        return cls(
            source=_compact_text(payload.get("source"), max_len=80) or "external",
            updated_at=_format_timestamp(updated_at),
            expires_at=_format_timestamp(expires_at),
            kind=_normalize_kind(payload.get("kind")),
            title=_compact_text(payload.get("title")),
            subtitle=_compact_text(payload.get("subtitle")),
            body_lines=_normalize_body_lines(payload.get("body_lines") or payload.get("body")),
            image_path=_normalize_image_path(payload.get("image_path")),
            accent=_normalize_accent(payload.get("accent")),
            cards=_normalize_cards(payload.get("cards")),
            active_card_key=_normalize_card_key(payload.get("active_card_key"), fallback="") or None,
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

    def transition_progress(self, *, now: datetime | None = None, duration_s: float = _MORPH_DURATION_S) -> float:
        """Return the bounded morph progress for this cue's expand animation."""

        updated_at = _normalize_timestamp(self.updated_at)
        if updated_at is None:
            return 1.0
        safe_duration = max(0.05, float(duration_s))
        elapsed_s = ((now or _utc_now()).astimezone(timezone.utc) - updated_at).total_seconds()
        if elapsed_s <= 0:
            return 0.0
        if elapsed_s >= safe_duration:
            return 1.0
        return max(0.0, min(1.0, elapsed_s / safe_duration))

    def transition_bucket(
        self,
        *,
        now: datetime | None = None,
        duration_s: float = _MORPH_DURATION_S,
        buckets: int = 12,
    ) -> int:
        """Return a stable render bucket for morph progress."""

        safe_buckets = max(1, int(buckets))
        return min(safe_buckets, int(round(self.transition_progress(now=now, duration_s=duration_s) * safe_buckets)))

    def signature(self) -> tuple[object, ...]:
        """Return a stable render-signature fragment for this cue."""

        return (
            self.source,
            self.updated_at,
            self.expires_at,
            self.kind,
            self.title,
            self.subtitle,
            self.body_lines,
            self.image_path,
            self.accent,
            tuple(card.signature() for card in self.normalized_cards()),
            self.active_card_key,
        )

    def normalized_cards(self) -> tuple[DisplayPresentationCardCue, ...]:
        """Return the bounded cards represented by this cue."""

        if self.cards:
            return self.cards
        if not any((self.title, self.subtitle, self.body_lines, self.image_path)):
            return ()
        synthesized = DisplayPresentationCardCue(
            key=_normalize_card_key(self.active_card_key, fallback="primary"),
            kind=self.kind,
            title=self.title,
            subtitle=self.subtitle,
            body_lines=self.body_lines,
            image_path=self.image_path,
            accent=self.accent,
        )
        return (synthesized,)

    def active_card(self) -> DisplayPresentationCardCue | None:
        """Return the selected active card for the current cue."""

        cards = self.normalized_cards()
        if not cards:
            return None
        if self.active_card_key:
            for card in cards:
                if card.key == self.active_card_key:
                    return card
        best = cards[0]
        for card in cards[1:]:
            if card.priority > best.priority:
                best = card
        return best

    def queued_cards(self) -> tuple[DisplayPresentationCardCue, ...]:
        """Return the non-active cards sorted by descending priority."""

        active = self.active_card()
        if active is None:
            return ()
        queued = [card for card in self.normalized_cards() if card.key != active.key]
        queued.sort(key=lambda card: card.priority, reverse=True)
        return tuple(queued)

    def transition_stage(self, *, now: datetime | None = None, duration_s: float = _MORPH_DURATION_S) -> str:
        """Return the semantic transition stage for this cue."""

        progress = self.transition_progress(now=now, duration_s=duration_s)
        if progress <= 0.0:
            return "docked"
        if progress < 0.22:
            return "lifting"
        if progress < 0.82:
            return "expanding"
        return "focused"

    def telemetry_signature(self, *, now: datetime | None = None, duration_s: float = _MORPH_DURATION_S) -> tuple[object, ...] | None:
        """Return a semantic signature suitable for low-noise telemetry."""

        active = self.active_card()
        if active is None:
            return None
        return (
            active.key,
            active.kind,
            active.priority,
            self.transition_stage(now=now, duration_s=duration_s),
            len(self.queued_cards()),
        )


@dataclass(slots=True)
class DisplayPresentationStore:
    """Read and write the optional presentation-cue artifact."""

    path: Path
    default_ttl_s: float = _DEFAULT_PRESENTATION_TTL_S

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayPresentationStore":
        """Resolve the presentation-cue path from Twinr configuration."""

        project_root = Path(config.project_root).expanduser().resolve()
        configured_path = Path(
            getattr(config, "display_presentation_path", _DEFAULT_PRESENTATION_PATH) or _DEFAULT_PRESENTATION_PATH
        )
        resolved_path = configured_path if configured_path.is_absolute() else project_root / configured_path
        return cls(
            path=resolved_path,
            default_ttl_s=float(
                getattr(config, "display_presentation_ttl_s", _DEFAULT_PRESENTATION_TTL_S)
                or _DEFAULT_PRESENTATION_TTL_S
            ),
        )

    def load(self) -> DisplayPresentationCue | None:
        """Load the current presentation cue, if one exists and parses."""

        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            _LOGGER.warning("Failed to read display presentation cue from %s.", self.path, exc_info=True)
            return None
        if not isinstance(payload, dict):
            _LOGGER.warning(
                "Ignoring invalid display presentation cue payload at %s because it is not an object.",
                self.path,
            )
            return None
        fallback_updated_at = datetime.fromtimestamp(self.path.stat().st_mtime, tz=timezone.utc)
        try:
            return DisplayPresentationCue.from_dict(
                payload,
                fallback_updated_at=fallback_updated_at,
                default_ttl_s=self.default_ttl_s,
            )
        except Exception:
            _LOGGER.warning("Ignoring invalid display presentation cue payload at %s.", self.path, exc_info=True)
            return None

    def load_active(self, *, now: datetime | None = None) -> DisplayPresentationCue | None:
        """Load the current cue only when it is still active."""

        cue = self.load()
        if cue is None or not cue.is_active(now=now):
            return None
        return cue

    def save(
        self,
        cue: DisplayPresentationCue,
        *,
        hold_seconds: float | None = None,
        now: datetime | None = None,
    ) -> DisplayPresentationCue:
        """Persist one normalized presentation cue atomically."""

        effective_now = (now or _utc_now()).astimezone(timezone.utc)
        effective_ttl_s = max(0.1, float(self.default_ttl_s if hold_seconds is None else hold_seconds))
        normalized = DisplayPresentationCue.from_dict(
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
        """Remove the current presentation cue artifact if one exists."""

        try:
            self.path.unlink()
        except FileNotFoundError:
            return


@dataclass(slots=True)
class DisplayPresentationController:
    """Persist producer-facing HDMI presentation cues without raw JSON."""

    store: DisplayPresentationStore
    default_source: str = "external"

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        default_source: str = "external",
    ) -> "DisplayPresentationController":
        """Resolve the configured presentation store and build one controller."""

        return cls(store=DisplayPresentationStore.from_config(config), default_source=default_source)

    def show_rich_card(
        self,
        *,
        key: str = "primary",
        title: str,
        subtitle: str = "",
        body_lines: Sequence[str] = (),
        accent: str = "info",
        priority: int = 100,
        face_emotion: str | None = None,
        source: str | None = None,
        hold_seconds: float | None = None,
        now: datetime | None = None,
    ) -> DisplayPresentationCue:
        """Persist one fullscreen rich-card presentation cue."""

        card = DisplayPresentationCardCue(
            key=key,
            kind="rich_card",
            title=title,
            subtitle=subtitle,
            body_lines=tuple(body_lines),
            accent=accent,
            priority=priority,
            face_emotion=face_emotion,
        )
        return self.show_scene(
            cards=(card,),
            active_card_key=card.key,
            source=source,
            hold_seconds=hold_seconds,
            now=now,
        )

    def show_image(
        self,
        *,
        key: str = "primary",
        image_path: str,
        title: str = "",
        subtitle: str = "",
        body_lines: Sequence[str] = (),
        accent: str = "info",
        priority: int = 100,
        face_emotion: str | None = None,
        source: str | None = None,
        hold_seconds: float | None = None,
        now: datetime | None = None,
    ) -> DisplayPresentationCue:
        """Persist one fullscreen image presentation cue."""

        card = DisplayPresentationCardCue(
            key=key,
            kind="image",
            title=title,
            subtitle=subtitle,
            body_lines=tuple(body_lines),
            image_path=image_path,
            accent=accent,
            priority=priority,
            face_emotion=face_emotion,
        )
        return self.show_scene(
            cards=(card,),
            active_card_key=card.key,
            source=source,
            hold_seconds=hold_seconds,
            now=now,
        )

    def show_scene(
        self,
        *,
        cards: Sequence[DisplayPresentationCardCue | Mapping[str, object]],
        active_card_key: str | None = None,
        source: str | None = None,
        hold_seconds: float | None = None,
        now: datetime | None = None,
    ) -> DisplayPresentationCue:
        """Persist one multi-card presentation scene with explicit priority."""

        normalized_cards = _normalize_cards(cards)
        active = None
        if active_card_key:
            normalized_key = _normalize_card_key(active_card_key)
            for card in normalized_cards:
                if card.key == normalized_key:
                    active = card
                    break
        if active is None and normalized_cards:
            active = normalized_cards[0]
            for card in normalized_cards[1:]:
                if card.priority > active.priority:
                    active = card
        cue = DisplayPresentationCue(
            source=source or self.default_source,
            kind=active.kind if active is not None else "rich_card",
            title=active.title if active is not None else "",
            subtitle=active.subtitle if active is not None else "",
            body_lines=active.body_lines if active is not None else (),
            image_path=active.image_path if active is not None else None,
            accent=active.accent if active is not None else "info",
            cards=normalized_cards,
            active_card_key=active.key if active is not None else None,
        )
        return self.store.save(cue, hold_seconds=hold_seconds, now=now)

    def clear(self) -> None:
        """Remove the currently active presentation cue."""

        self.store.clear()
