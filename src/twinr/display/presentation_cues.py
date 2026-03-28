# CHANGELOG: 2026-03-28
# BUG-1: Fixed concurrent-writer races caused by a shared temp filename; saves now use an inter-process lock and a unique temp file.
# BUG-2: Fixed unstable relative image paths; image paths are now resolved against project_root before persistence.
# BUG-3: Fixed silent persistence of empty/invalid scenes; producer-facing APIs now reject scenes with no valid cards.
# BUG-4: Fixed duplicate card-key collisions; duplicate keys are deterministically uniquified instead of shadowing each other.
# SEC-1: Restricted image_path to approved local roots/extensions and optional file-size/existence checks to block arbitrary file reads and device/proc paths.
# SEC-2: Added bounded store-size checks plus durable fsync-backed writes to reduce corruption, memory pressure, and stale-on-crash artifacts.
# IMP-1: Added schema_version, sequence, face_emotion mirroring, and stricter normalization so the artifact contract is explicit and evolvable.
# IMP-2: Removed deep-copy-heavy dataclasses.asdict() serialization in favor of shallow manual wire serialization for lower overhead on Pi-class hardware.

"""Persist and produce optional HDMI presentation cues.

The default HDMI scene should be able to expand beyond the small status panel
without teaching the generic runtime snapshot schema about images, focus cards,
or other presentation-only concepts. This module stores one bounded optional
presentation payload and offers a producer-facing controller so other Twinr
modules can trigger it cleanly.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from os import PathLike
from pathlib import Path
import tempfile
from typing import Protocol

try:  # Unix on Raspberry Pi OS.
    import fcntl
except ImportError:  # pragma: no cover - not expected on Pi, kept for portability.
    fcntl = None


_SCHEMA_VERSION = 2
_DEFAULT_PRESENTATION_TTL_S = 20.0
_DEFAULT_PRESENTATION_MAX_TTL_S = 3600.0
_DEFAULT_PRESENTATION_PATH = "artifacts/stores/ops/display_presentation.json"
_DEFAULT_MAX_STORE_BYTES = 64 * 1024
_DEFAULT_MAX_IMAGE_BYTES = 15 * 1024 * 1024
_DEFAULT_STORE_FILE_MODE = 0o600

_ALLOWED_KINDS = frozenset({"image", "rich_card"})
_ALLOWED_ACCENTS = frozenset({"alert", "info", "success", "warm"})
_ALLOWED_FACE_EMOTIONS = frozenset({"calm", "happy", "sad", "thoughtful", "curious", "focused"})
_ALLOWED_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"})

_MAX_CARDS = 4
_MAX_BODY_LINES = 5
_MAX_TEXT_LEN = 140
_MORPH_DURATION_S = 0.85

_LOGGER = logging.getLogger(__name__)


class _DisplayConfigLike(Protocol):
    """Describe the minimal config surface needed by the presentation store."""

    project_root: str | PathLike[str]


@dataclass(frozen=True, slots=True)
class _DisplayPresentationPolicy:
    """Bound validation and persistence behavior for one store instance."""

    project_root: Path
    allowed_image_roots: tuple[Path, ...]
    allowed_image_extensions: frozenset[str]
    require_existing_images: bool
    max_image_bytes: int
    max_store_bytes: int
    max_ttl_s: float
    file_mode: int
    clear_expired_on_load: bool


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
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
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


def _normalize_card_key(value: object | None, *, fallback: str = "primary") -> str:
    """Normalize one short presentation-card key for durable storage."""

    compact = _compact_text(value, max_len=48).lower().replace("-", "_").replace(" ", "_")
    return compact or fallback


def _normalize_priority(value: object | None, *, default: int = 100) -> int:
    """Normalize one bounded card priority."""

    numeric: float | str
    numeric = default if value is None else (int(value) if isinstance(value, bool) else str(value))
    try:
        parsed = int(round(float(numeric)))
    except (TypeError, ValueError):
        return default
    return max(0, min(999, parsed))


def _normalize_face_emotion(value: object | None) -> str | None:
    """Normalize one optional face-emotion label for presentation sync."""

    compact = _compact_text(value, max_len=24).lower().replace("-", "_").replace(" ", "_")
    if compact not in _ALLOWED_FACE_EMOTIONS:
        return None
    return compact


def _normalize_schema_version(value: object | None) -> int:
    """Normalize one persisted schema version."""

    try:
        parsed = int(value if value is not None else _SCHEMA_VERSION)
    except (TypeError, ValueError):
        return _SCHEMA_VERSION
    return max(1, parsed)


def _normalize_sequence(value: object | None) -> int:
    """Normalize one persisted cue sequence number."""

    try:
        parsed = int(value if value is not None else 0)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _normalize_positive_float(
    value: object | None,
    *,
    default: float,
    minimum: float,
    maximum: float | None = None,
) -> float:
    """Normalize one bounded positive floating-point value."""

    try:
        parsed = float(value if value is not None else default)
    except (TypeError, ValueError):
        parsed = default
    if parsed < minimum:
        parsed = minimum
    if maximum is not None and parsed > maximum:
        parsed = maximum
    return parsed


def _normalize_positive_int(
    value: object | None,
    *,
    default: int,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """Normalize one bounded positive integer value."""

    try:
        parsed = int(value if value is not None else default)
    except (TypeError, ValueError):
        parsed = default
    if parsed < minimum:
        parsed = minimum
    if maximum is not None and parsed > maximum:
        parsed = maximum
    return parsed


def _normalize_bool(value: object | None, *, default: bool) -> bool:
    """Normalize one boolean-like value."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _coerce_path(value: object, *, project_root: Path) -> Path:
    """Resolve one configured path relative to the project root."""

    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else project_root / raw
    return candidate.resolve(strict=False)


def _coerce_path_list(value: object | None, *, project_root: Path) -> tuple[Path, ...]:
    """Normalize one optional configured path list."""

    if value is None:
        return ()
    if isinstance(value, (str, os.PathLike)):
        return (_coerce_path(value, project_root=project_root),)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        roots: list[Path] = []
        for item in value:
            if item in (None, ""):
                continue
            roots.append(_coerce_path(item, project_root=project_root))
        return tuple(roots)
    return ()


def _coerce_extension_set(value: object | None) -> frozenset[str]:
    """Normalize one optional configured image-extension allow-list."""

    if value is None:
        return _ALLOWED_IMAGE_EXTENSIONS
    if isinstance(value, str):
        items: Sequence[object] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = value
    else:
        return _ALLOWED_IMAGE_EXTENSIONS
    normalized = {
        f".{_compact_text(item, max_len=16).lower().lstrip('.')}"
        for item in items
        if _compact_text(item, max_len=16)
    }
    return frozenset(normalized or _ALLOWED_IMAGE_EXTENSIONS)


def _policy_from_config(config: _DisplayConfigLike) -> _DisplayPresentationPolicy:
    """Build one normalization and persistence policy from Twinr config."""

    project_root = Path(config.project_root).expanduser().resolve()
    temp_root = Path(tempfile.gettempdir()).expanduser().resolve()
    configured_roots = _coerce_path_list(
        getattr(config, "display_presentation_allowed_image_roots", None)
        or getattr(config, "display_presentation_assets_root", None)
        or getattr(config, "display_presentation_image_root", None),
        project_root=project_root,
    )
    default_roots: tuple[Path, ...]
    if configured_roots:
        default_roots = configured_roots
    elif temp_root == project_root:
        default_roots = (project_root,)
    else:
        # Keep repo-local assets working by default while still allowing the
        # transient /tmp images used by visual QC and operator-driven previews.
        default_roots = (project_root, temp_root)
    return _DisplayPresentationPolicy(
        project_root=project_root,
        allowed_image_roots=default_roots,
        allowed_image_extensions=_coerce_extension_set(
            getattr(config, "display_presentation_allowed_image_extensions", None)
        ),
        require_existing_images=_normalize_bool(
            getattr(config, "display_presentation_require_existing_images", None),
            default=False,
        ),
        max_image_bytes=_normalize_positive_int(
            getattr(config, "display_presentation_max_image_bytes", None),
            default=_DEFAULT_MAX_IMAGE_BYTES,
            minimum=1,
        ),
        max_store_bytes=_normalize_positive_int(
            getattr(config, "display_presentation_max_store_bytes", None),
            default=_DEFAULT_MAX_STORE_BYTES,
            minimum=1024,
        ),
        max_ttl_s=_normalize_positive_float(
            getattr(config, "display_presentation_max_ttl_s", None),
            default=_DEFAULT_PRESENTATION_MAX_TTL_S,
            minimum=0.1,
        ),
        file_mode=_normalize_positive_int(
            getattr(config, "display_presentation_store_file_mode", None),
            default=_DEFAULT_STORE_FILE_MODE,
            minimum=0,
            maximum=0o777,
        ),
        clear_expired_on_load=_normalize_bool(
            getattr(config, "display_presentation_clear_expired_on_load", None),
            default=True,
        ),
    )


def _resolve_candidate_path(text: str, *, policy: _DisplayPresentationPolicy) -> Path:
    """Resolve one candidate image path against the configured project root."""

    raw = Path(text).expanduser()
    candidate = raw if raw.is_absolute() else policy.project_root / raw
    return candidate.resolve(strict=False)


def _is_within_root(candidate: Path, root: Path) -> bool:
    """Return whether one resolved path stays under one resolved root."""

    try:
        return candidate.is_relative_to(root)
    except AttributeError:  # pragma: no cover - for very old Python only.
        return candidate == root or root in candidate.parents


def _normalize_image_path(
    value: object | None,
    *,
    policy: _DisplayPresentationPolicy | None = None,
    require_existing: bool | None = None,
) -> str | None:
    """Normalize one optional local image path."""

    compact = _compact_text(value, max_len=512)
    if not compact:
        return None
    if policy is None:
        return str(Path(compact).expanduser())

    candidate = _resolve_candidate_path(compact, policy=policy)
    if not any(_is_within_root(candidate, root) for root in policy.allowed_image_roots):
        raise ValueError(
            f"image_path {candidate} is outside the configured presentation asset roots: "
            f"{', '.join(str(root) for root in policy.allowed_image_roots)}"
        )

    suffix = candidate.suffix.lower()
    if suffix not in policy.allowed_image_extensions:
        raise ValueError(
            f"image_path {candidate} uses unsupported extension {suffix or '<none>'}; "
            f"allowed: {', '.join(sorted(policy.allowed_image_extensions))}"
        )

    must_exist = policy.require_existing_images if require_existing is None else require_existing
    if candidate.exists():
        if not candidate.is_file():
            raise ValueError(f"image_path {candidate} is not a regular file")
        size_bytes = candidate.stat().st_size
        if size_bytes > policy.max_image_bytes:
            raise ValueError(
                f"image_path {candidate} is {size_bytes} bytes, exceeding the "
                f"{policy.max_image_bytes}-byte limit"
            )
    elif must_exist:
        raise ValueError(f"image_path {candidate} does not exist")

    return str(candidate)


def _wire_card_dict(card: "DisplayPresentationCardCue") -> dict[str, object]:
    """Serialize one card into a JSON-safe mapping."""

    return {
        "key": card.key,
        "kind": card.kind,
        "title": card.title,
        "subtitle": card.subtitle,
        "body_lines": list(card.body_lines),
        "image_path": card.image_path,
        "accent": card.accent,
        "priority": card.priority,
        "face_emotion": card.face_emotion,
    }


def _wire_cue_dict(cue: "DisplayPresentationCue") -> dict[str, object]:
    """Serialize one cue into a JSON-safe mapping."""

    return {
        "schema_version": cue.schema_version,
        "sequence": cue.sequence,
        "source": cue.source,
        "updated_at": cue.updated_at,
        "expires_at": cue.expires_at,
        "kind": cue.kind,
        "title": cue.title,
        "subtitle": cue.subtitle,
        "body_lines": list(cue.body_lines),
        "image_path": cue.image_path,
        "accent": cue.accent,
        "face_emotion": cue.face_emotion,
        "cards": [_wire_card_dict(card) for card in cue.cards],
        "active_card_key": cue.active_card_key,
    }


def _select_active_card(
    cards: Sequence["DisplayPresentationCardCue"],
    active_card_key: str | None,
) -> "DisplayPresentationCardCue | None":
    """Select one active card from a bounded card list."""

    if not cards:
        return None
    if active_card_key:
        normalized_key = _normalize_card_key(active_card_key, fallback="")
        for card in cards:
            if card.key == normalized_key:
                return card
    best = cards[0]
    for card in cards[1:]:
        if card.priority > best.priority:
            best = card
    return best


def _uniquify_card_keys(cards: Sequence["DisplayPresentationCardCue"]) -> tuple["DisplayPresentationCardCue", ...]:
    """Ensure each card key is unique while preserving order."""

    seen: set[str] = set()
    unique_cards: list[DisplayPresentationCardCue] = []
    for card in cards:
        key = card.key
        if key not in seen:
            seen.add(key)
            unique_cards.append(card)
            continue
        suffix = 2
        while True:
            candidate = _normalize_card_key(f"{key}_{suffix}", fallback=key)
            if candidate not in seen:
                _LOGGER.warning(
                    "Duplicate display presentation card key %r detected; renamed later card to %r.",
                    key,
                    candidate,
                )
                seen.add(candidate)
                unique_cards.append(replace(card, key=candidate))
                break
            suffix += 1
    return tuple(unique_cards)


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
        # BREAKING: cards with no visible content and no face_emotion are rejected instead of being silently persisted.
        if not any((self.title, self.subtitle, self.body_lines, self.image_path, self.face_emotion)):
            raise ValueError("DisplayPresentationCardCue must contain text, image_path, or face_emotion")

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        fallback_key: str = "primary",
        policy: _DisplayPresentationPolicy | None = None,
    ) -> "DisplayPresentationCardCue":
        """Build one normalized presentation card from JSON-style data."""

        kind = _normalize_kind(payload.get("kind"))
        image_path = _normalize_image_path(
            payload.get("image_path"),
            policy=policy,
            require_existing=None,
        )
        return cls(
            key=_normalize_card_key(payload.get("key"), fallback=fallback_key),
            kind=kind,
            title=_compact_text(payload.get("title")),
            subtitle=_compact_text(payload.get("subtitle")),
            body_lines=_normalize_body_lines(payload.get("body_lines") or payload.get("body")),
            image_path=image_path,
            accent=_normalize_accent(payload.get("accent")),
            priority=_normalize_priority(payload.get("priority")),
            face_emotion=_normalize_face_emotion(payload.get("face_emotion")),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the card into a JSON-safe mapping."""

        return _wire_card_dict(self)

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


def _normalize_cards(
    value: object | None,
    *,
    policy: _DisplayPresentationPolicy | None = None,
    strict: bool = False,
) -> tuple[DisplayPresentationCardCue, ...]:
    """Normalize one optional card sequence into bounded card cues."""

    if value is None:
        return ()
    if isinstance(value, DisplayPresentationCardCue):
        raw_items: Sequence[object] = (value,)
    elif isinstance(value, Mapping):
        raw_items = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        raw_items = value
    else:
        return ()

    cards: list[DisplayPresentationCardCue] = []
    for index, item in enumerate(raw_items):
        if len(cards) >= _MAX_CARDS:
            break
        fallback_key = f"card_{index + 1}"
        try:
            if isinstance(item, DisplayPresentationCardCue):
                card = (
                    DisplayPresentationCardCue.from_dict(item.to_dict(), fallback_key=item.key, policy=policy)
                    if policy is not None
                    else item
                )
                cards.append(card)
                continue
            if isinstance(item, Mapping):
                cards.append(DisplayPresentationCardCue.from_dict(item, fallback_key=fallback_key, policy=policy))
        except ValueError:
            if strict:
                raise
            _LOGGER.warning("Dropping invalid display presentation card at index %s.", index, exc_info=True)
    return _uniquify_card_keys(cards)


@dataclass(frozen=True, slots=True)
class DisplayPresentationCue:
    """Describe one optional HDMI presentation overlay."""

    schema_version: int = _SCHEMA_VERSION
    sequence: int = 0
    source: str = "external"
    updated_at: str | None = None
    expires_at: str | None = None
    kind: str = "rich_card"
    title: str = ""
    subtitle: str = ""
    body_lines: tuple[str, ...] = ()
    image_path: str | None = None
    accent: str = "info"
    face_emotion: str | None = None
    cards: tuple[DisplayPresentationCardCue, ...] = ()
    active_card_key: str | None = None

    def __post_init__(self) -> None:
        """Normalize direct constructor calls into the canonical cue contract."""

        updated_at = _normalize_timestamp(self.updated_at)
        expires_at = _normalize_timestamp(self.expires_at)
        object.__setattr__(self, "schema_version", _normalize_schema_version(self.schema_version))
        object.__setattr__(self, "sequence", _normalize_sequence(self.sequence))
        object.__setattr__(self, "source", _compact_text(self.source, max_len=80) or "external")
        object.__setattr__(self, "updated_at", _format_timestamp(updated_at) if updated_at is not None else None)
        object.__setattr__(self, "expires_at", _format_timestamp(expires_at) if expires_at is not None else None)
        object.__setattr__(self, "kind", _normalize_kind(self.kind))
        object.__setattr__(self, "title", _compact_text(self.title))
        object.__setattr__(self, "subtitle", _compact_text(self.subtitle))
        object.__setattr__(self, "body_lines", _normalize_body_lines(self.body_lines))
        object.__setattr__(self, "image_path", _normalize_image_path(self.image_path))
        object.__setattr__(self, "accent", _normalize_accent(self.accent))
        object.__setattr__(self, "face_emotion", _normalize_face_emotion(self.face_emotion))
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
        policy: _DisplayPresentationPolicy | None = None,
        strict: bool = False,
    ) -> "DisplayPresentationCue":
        """Build one normalized presentation cue from JSON-style data."""

        safe_now = (fallback_updated_at or _utc_now()).astimezone(timezone.utc)
        effective_ttl_s = _normalize_positive_float(
            default_ttl_s,
            default=_DEFAULT_PRESENTATION_TTL_S,
            minimum=0.1,
            maximum=(policy.max_ttl_s if policy is not None else _DEFAULT_PRESENTATION_MAX_TTL_S),
        )

        cards = _normalize_cards(payload.get("cards"), policy=policy, strict=strict)
        active_card = _select_active_card(cards, payload.get("active_card_key"))

        updated_at = _normalize_timestamp(payload.get("updated_at")) or safe_now
        expires_at = _normalize_timestamp(payload.get("expires_at"))
        if expires_at is None or expires_at < updated_at:
            expires_at = updated_at + timedelta(seconds=effective_ttl_s)

        if active_card is not None:
            kind = active_card.kind
            title = active_card.title
            subtitle = active_card.subtitle
            body_lines = active_card.body_lines
            image_path = active_card.image_path
            accent = active_card.accent
            face_emotion = active_card.face_emotion
            active_card_key = active_card.key
        else:
            top_kind = _normalize_kind(payload.get("kind"))
            kind = top_kind
            title = _compact_text(payload.get("title"))
            subtitle = _compact_text(payload.get("subtitle"))
            body_lines = _normalize_body_lines(payload.get("body_lines") or payload.get("body"))
            image_path = _normalize_image_path(
                payload.get("image_path"),
                policy=policy,
                require_existing=None,
            )
            accent = _normalize_accent(payload.get("accent"))
            face_emotion = _normalize_face_emotion(payload.get("face_emotion"))
            active_card_key = _normalize_card_key(payload.get("active_card_key"), fallback="") or None

        return cls(
            schema_version=_normalize_schema_version(payload.get("schema_version")),
            sequence=_normalize_sequence(payload.get("sequence")),
            source=_compact_text(payload.get("source"), max_len=80) or "external",
            updated_at=_format_timestamp(updated_at),
            expires_at=_format_timestamp(expires_at),
            kind=kind,
            title=title,
            subtitle=subtitle,
            body_lines=body_lines,
            image_path=image_path,
            accent=accent,
            face_emotion=face_emotion,
            cards=cards,
            active_card_key=active_card_key,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the cue into a JSON-safe mapping."""

        return _wire_cue_dict(self)

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
            self.schema_version,
            self.sequence,
            self.source,
            self.updated_at,
            self.expires_at,
            self.kind,
            self.title,
            self.subtitle,
            self.body_lines,
            self.image_path,
            self.accent,
            self.face_emotion,
            tuple(card.signature() for card in self.normalized_cards()),
            self.active_card_key,
        )

    def normalized_cards(self) -> tuple[DisplayPresentationCardCue, ...]:
        """Return the bounded cards represented by this cue."""

        if self.cards:
            return self.cards
        if not any((self.title, self.subtitle, self.body_lines, self.image_path, self.face_emotion)):
            return ()
        synthesized = DisplayPresentationCardCue(
            key=_normalize_card_key(self.active_card_key, fallback="primary"),
            kind=self.kind,
            title=self.title,
            subtitle=self.subtitle,
            body_lines=self.body_lines,
            image_path=self.image_path,
            accent=self.accent,
            face_emotion=self.face_emotion,
        )
        return (synthesized,)

    def active_card(self) -> DisplayPresentationCardCue | None:
        """Return the selected active card for the current cue."""

        return _select_active_card(self.normalized_cards(), self.active_card_key)

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

    def telemetry_signature(
        self,
        *,
        now: datetime | None = None,
        duration_s: float = _MORPH_DURATION_S,
    ) -> tuple[object, ...] | None:
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
            self.sequence,
        )


def _json_dumps(payload: Mapping[str, object]) -> str:
    """Serialize one payload deterministically and compactly."""

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _fsync_directory(path: Path) -> None:
    """Best-effort fsync for one directory to harden atomic replace on Unix."""

    flags = getattr(os, "O_RDONLY", 0)
    with suppress(OSError):
        fd = os.open(path, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


@contextmanager
def _exclusive_lock(lock_path: Path, *, file_mode: int) -> Iterator[None]:
    """Acquire one advisory inter-process lock for writers and destructive clears."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        with suppress(OSError):
            os.chmod(lock_path, file_mode)
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(slots=True)
class DisplayPresentationStore:
    """Read and write the optional presentation-cue artifact."""

    path: Path
    default_ttl_s: float = _DEFAULT_PRESENTATION_TTL_S
    policy: _DisplayPresentationPolicy | None = None

    @classmethod
    def from_config(cls, config: _DisplayConfigLike) -> "DisplayPresentationStore":
        """Resolve the presentation-cue path from Twinr configuration."""

        project_root = Path(config.project_root).expanduser().resolve()
        raw_path = getattr(config, "display_presentation_path", None)
        configured_path = (
            Path(raw_path).expanduser()
            if raw_path not in (None, "")
            else Path(_DEFAULT_PRESENTATION_PATH)
        )
        resolved_path = configured_path if configured_path.is_absolute() else project_root / configured_path
        policy = _policy_from_config(config)
        return cls(
            path=resolved_path.resolve(strict=False),
            default_ttl_s=_normalize_positive_float(
                getattr(config, "display_presentation_ttl_s", None),
                default=_DEFAULT_PRESENTATION_TTL_S,
                minimum=0.1,
                maximum=policy.max_ttl_s,
            ),
            policy=policy,
        )

    @property
    def _lock_path(self) -> Path:
        """Return the sidecar lock path used for writer coordination."""

        return self.path.with_name(f".{self.path.name}.lock")

    def _read_raw_payload(self) -> tuple[dict[str, object] | None, datetime | None]:
        """Read one raw JSON payload with bounded size checks."""

        if not self.path.exists():
            return None, None
        try:
            with self.path.open("rb") as handle:
                stat_result = os.fstat(handle.fileno())
                max_store_bytes = (
                    self.policy.max_store_bytes if self.policy is not None else _DEFAULT_MAX_STORE_BYTES
                )
                if stat_result.st_size > max_store_bytes:
                    _LOGGER.warning(
                        "Ignoring oversized display presentation cue at %s (%s bytes > %s).",
                        self.path,
                        stat_result.st_size,
                        max_store_bytes,
                    )
                    return None, None
                data = handle.read(max_store_bytes + 1)
        except FileNotFoundError:
            return None, None
        except Exception:
            _LOGGER.warning("Failed to read display presentation cue from %s.", self.path, exc_info=True)
            return None, None

        if len(data) > max_store_bytes:
            _LOGGER.warning(
                "Ignoring oversized display presentation cue at %s because the read exceeded %s bytes.",
                self.path,
                max_store_bytes,
            )
            return None, None

        try:
            payload = json.loads(data.decode("utf-8"))
        except Exception:
            _LOGGER.warning("Failed to decode display presentation cue from %s.", self.path, exc_info=True)
            return None, None

        if not isinstance(payload, dict):
            _LOGGER.warning(
                "Ignoring invalid display presentation cue payload at %s because it is not an object.",
                self.path,
            )
            return None, None

        fallback_updated_at = datetime.fromtimestamp(stat_result.st_mtime, tz=timezone.utc)
        return payload, fallback_updated_at

    def _load_unlocked(self) -> DisplayPresentationCue | None:
        """Load the current presentation cue without acquiring the write lock."""

        payload, fallback_updated_at = self._read_raw_payload()
        if payload is None:
            return None
        try:
            cue = DisplayPresentationCue.from_dict(
                payload,
                fallback_updated_at=fallback_updated_at,
                default_ttl_s=self.default_ttl_s,
                policy=self.policy,
                strict=False,
            )
        except Exception:
            _LOGGER.warning("Ignoring invalid display presentation cue payload at %s.", self.path, exc_info=True)
            return None

        if cue.schema_version > _SCHEMA_VERSION:
            _LOGGER.warning(
                "Loaded display presentation cue schema_version=%s newer than supported=%s.",
                cue.schema_version,
                _SCHEMA_VERSION,
            )
        return cue

    def load(self) -> DisplayPresentationCue | None:
        """Load the current presentation cue, if one exists and parses."""

        return self._load_unlocked()

    def load_active(self, *, now: datetime | None = None) -> DisplayPresentationCue | None:
        """Load the current cue only when it is still active."""

        cue = self.load()
        if cue is None:
            return None
        if cue.is_active(now=now):
            return cue
        if self.policy is not None and self.policy.clear_expired_on_load:
            with _exclusive_lock(self._lock_path, file_mode=self.policy.file_mode):
                latest = self._load_unlocked()
                if latest is not None and not latest.is_active(now=now):
                    with suppress(FileNotFoundError):
                        self.path.unlink()
        return None

    def save(
        self,
        cue: DisplayPresentationCue,
        *,
        hold_seconds: float | None = None,
        now: datetime | None = None,
    ) -> DisplayPresentationCue:
        """Persist one normalized presentation cue atomically and durably."""

        policy = self.policy or _DisplayPresentationPolicy(
            project_root=self.path.parent.resolve(strict=False),
            allowed_image_roots=(self.path.parent.resolve(strict=False),),
            allowed_image_extensions=_ALLOWED_IMAGE_EXTENSIONS,
            require_existing_images=False,
            max_image_bytes=_DEFAULT_MAX_IMAGE_BYTES,
            max_store_bytes=_DEFAULT_MAX_STORE_BYTES,
            max_ttl_s=_DEFAULT_PRESENTATION_MAX_TTL_S,
            file_mode=_DEFAULT_STORE_FILE_MODE,
            clear_expired_on_load=True,
        )
        effective_now = (now or _utc_now()).astimezone(timezone.utc)
        effective_ttl_s = _normalize_positive_float(
            self.default_ttl_s if hold_seconds is None else hold_seconds,
            default=self.default_ttl_s,
            minimum=0.1,
            maximum=policy.max_ttl_s,
        )

        wire_payload = cue.to_dict()
        stamp_new_lifetime = hold_seconds is not None or _normalize_timestamp(wire_payload.get("updated_at")) is None
        if stamp_new_lifetime:
            wire_payload["updated_at"] = _format_timestamp(effective_now)
            wire_payload["expires_at"] = _format_timestamp(effective_now + timedelta(seconds=effective_ttl_s))
        elif _normalize_timestamp(wire_payload.get("expires_at")) is None:
            wire_payload["expires_at"] = _format_timestamp(effective_now + timedelta(seconds=effective_ttl_s))
        wire_payload["schema_version"] = _SCHEMA_VERSION

        self.path.parent.mkdir(parents=True, exist_ok=True)

        with _exclusive_lock(self._lock_path, file_mode=policy.file_mode):
            current = self._load_unlocked()
            wire_payload["sequence"] = (current.sequence + 1) if current is not None else 1

            normalized = DisplayPresentationCue.from_dict(
                wire_payload,
                fallback_updated_at=effective_now,
                default_ttl_s=effective_ttl_s,
                policy=policy,
                strict=True,
            )
            serialized = _json_dumps(normalized.to_dict())
            encoded = serialized.encode("utf-8")
            if len(encoded) > policy.max_store_bytes:
                raise ValueError(
                    f"Serialized presentation cue is {len(encoded)} bytes, exceeding the "
                    f"{policy.max_store_bytes}-byte limit"
                )

            fd, tmp_name = tempfile.mkstemp(
                dir=str(self.path.parent),
                prefix=f".{self.path.name}.",
                suffix=".tmp",
            )
            tmp_path = Path(tmp_name)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                with suppress(OSError):
                    os.chmod(tmp_path, policy.file_mode)
                os.replace(tmp_path, self.path)
                with suppress(OSError):
                    os.chmod(self.path, policy.file_mode)
                _fsync_directory(self.path.parent)
            finally:
                with suppress(FileNotFoundError):
                    tmp_path.unlink()

        return normalized

    def clear(self) -> None:
        """Remove the current presentation cue artifact if one exists."""

        policy = self.policy or _DisplayPresentationPolicy(
            project_root=self.path.parent.resolve(strict=False),
            allowed_image_roots=(self.path.parent.resolve(strict=False),),
            allowed_image_extensions=_ALLOWED_IMAGE_EXTENSIONS,
            require_existing_images=False,
            max_image_bytes=_DEFAULT_MAX_IMAGE_BYTES,
            max_store_bytes=_DEFAULT_MAX_STORE_BYTES,
            max_ttl_s=_DEFAULT_PRESENTATION_MAX_TTL_S,
            file_mode=_DEFAULT_STORE_FILE_MODE,
            clear_expired_on_load=True,
        )
        with _exclusive_lock(self._lock_path, file_mode=policy.file_mode):
            with suppress(FileNotFoundError):
                self.path.unlink()
            _fsync_directory(self.path.parent)


@dataclass(slots=True)
class DisplayPresentationController:
    """Persist producer-facing HDMI presentation cues without raw JSON."""

    store: DisplayPresentationStore
    default_source: str = "external"

    @classmethod
    def from_config(
        cls,
        config: _DisplayConfigLike,
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

        normalized_cards = _normalize_cards(cards, policy=self.store.policy, strict=True)
        # BREAKING: empty scenes now raise instead of persisting an active-but-empty cue artifact.
        if not normalized_cards:
            raise ValueError("Display presentation scenes must contain at least one valid card")

        active = _select_active_card(normalized_cards, active_card_key)
        if active is None:
            raise ValueError("Display presentation scenes must contain an active card")

        cue = DisplayPresentationCue(
            source=source or self.default_source,
            kind=active.kind,
            title=active.title,
            subtitle=active.subtitle,
            body_lines=active.body_lines,
            image_path=active.image_path,
            accent=active.accent,
            face_emotion=active.face_emotion,
            cards=normalized_cards,
            active_card_key=active.key,
        )
        return self.store.save(cue, hold_seconds=hold_seconds, now=now)

    def clear(self) -> None:
        """Remove the currently active presentation cue."""

        self.store.clear()
