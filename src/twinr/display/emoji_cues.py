# CHANGELOG: 2026-03-28
# BUG-1: show_symbol()/from_dict() now accept real Unicode emoji glyphs (for example 👍, 👋, ❤️)
#         instead of silently collapsing unsupported glyph inputs to "heart".
# BUG-2: save() no longer uses one shared *.tmp path; concurrent writers are serialized with a lock
#         and each save uses a unique secure temp file, fixing FileNotFoundError/lost-update races.
# BUG-3: load() no longer races against clear()/save() while stat'ing the artifact; reads are lock-aware
#         and use metadata from the already-open file descriptor.
# SEC-1: load() refuses symlinks/non-regular files and caps artifact size, preventing practical local
#         filesystem abuse (symlink/FIFO tricks, oversized file DoS) on shared Raspberry Pi deployments.
# SEC-2: writes are durable as well as atomic: temp file is fsync()'d, then os.replace()'d, then the
#         parent directory is fsync()'d, reducing silent cue loss after power cuts.
# IMP-1: cue TTL is now bounded and configurable via display_emoji_cue_max_ttl_s to stop stale cues from
#         pinning the senior-facing HDMI surface indefinitely after producer bugs or bad payloads.
# IMP-2: producer ergonomics upgraded with alias normalization ("thumbsup", "wave", glyphs, etc.) while
#         the persisted contract stays canonical and backward compatible.
# BUG-4: signature() now excludes updated_at/expires_at so cue lifetime refreshes do not
#         force semantically identical HDMI reserve rerenders on hdmi_wayland.

"""Persist and normalize optional HDMI emoji cues.

This module keeps the right-hand HDMI reserve area on a structured, bounded
cue contract instead of letting random text or ad-hoc renderer state leak into
the main senior-facing surface. Producers can request one real Unicode emoji
such as a thumbs-up or a waving hand without teaching the generic runtime
snapshot schema about emoji semantics.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import errno
import json
import logging
import math
import os
from os import PathLike
from pathlib import Path
import stat
import tempfile
from typing import Protocol
import unicodedata

try:
    import fcntl
except ImportError:  # pragma: no cover - not expected on Raspberry Pi / Linux.
    fcntl = None


_DEFAULT_EMOJI_CUE_TTL_S = 6.0
_DEFAULT_EMOJI_CUE_MAX_TTL_S = 300.0  # BREAKING: explicit TTLs are now capped by default.
_DEFAULT_EMOJI_CUE_PATH = "artifacts/stores/ops/display_emoji.json"
_ALLOWED_ACCENTS = frozenset({"neutral", "info", "success", "warm", "alert"})
_MAX_CUE_FILE_BYTES = 4096
_MAX_SOURCE_LEN = 80
_MAX_SYMBOL_TOKEN_LEN = 24
_MAX_ACCENT_LEN = 24
_MAX_ALLOWED_MAX_TTL_S = 86_400.0
_VARIATION_SELECTORS = frozenset({"\ufe0f", "\ufe0e"})

_LOGGER = logging.getLogger(__name__)


class _DisplayConfigLike(Protocol):
    """Describe the minimal config surface needed by the emoji cue store."""

    project_root: str | PathLike[str]


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


def _compact_text(value: object | None, *, max_len: int = _MAX_SOURCE_LEN) -> str:
    """Normalize one bounded text field."""

    if value is None:
        return ""
    text = "".join(ch if ch.isprintable() else " " for ch in str(value))
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _normalize_alias_key(value: object | None, *, max_len: int = _MAX_SYMBOL_TOKEN_LEN) -> str:
    """Normalize one textual symbol/accent alias into a comparable key."""

    compact = _compact_text(value, max_len=max_len)
    if not compact:
        return ""
    normalized = unicodedata.normalize("NFKC", compact).strip().lower()
    normalized = "".join(ch for ch in normalized if ch not in _VARIATION_SELECTORS)
    normalized = normalized.replace("-", "_").replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _strip_emoji_variants(value: str) -> str:
    """Strip text/emoji presentation selectors from one glyph string."""

    return "".join(ch for ch in unicodedata.normalize("NFKC", value) if ch not in _VARIATION_SELECTORS)


def _coerce_positive_seconds(
    value: object | None,
    *,
    default: float,
    minimum: float = 0.1,
    maximum: float | None = None,
) -> float:
    """Normalize one TTL-like value into a finite bounded positive float."""

    try:
        seconds = float(default if value is None else value)
    except (TypeError, ValueError):
        seconds = float(default)
    if not math.isfinite(seconds):
        seconds = float(default)
    seconds = max(minimum, seconds)
    if maximum is not None:
        seconds = min(seconds, maximum)
    return seconds


class DisplayEmojiSymbol(str, Enum):
    """Supported right-hand HDMI emoji symbols."""

    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    WAVING_HAND = "waving_hand"
    RAISED_HAND = "raised_hand"
    POINTING_HAND = "pointing_hand"
    VICTORY_HAND = "victory_hand"
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
    DisplayEmojiSymbol.RAISED_HAND: "🖐️",
    DisplayEmojiSymbol.POINTING_HAND: "👉",
    DisplayEmojiSymbol.VICTORY_HAND: "✌️",
    DisplayEmojiSymbol.OK_HAND: "👌",
    DisplayEmojiSymbol.HEART: "❤️",
    DisplayEmojiSymbol.CHECK: "✅",
    DisplayEmojiSymbol.QUESTION: "❓",
    DisplayEmojiSymbol.EXCLAMATION: "❗",
    DisplayEmojiSymbol.WARNING: "⚠️",
    DisplayEmojiSymbol.SPARKLES: "✨",
}

_SYMBOL_ALIASES: dict[str, DisplayEmojiSymbol] = {}


def _register_symbol_aliases(symbol: DisplayEmojiSymbol, *aliases: str) -> None:
    """Register one symbol's accepted token/glyph aliases."""

    for alias in aliases:
        key = _normalize_alias_key(alias)
        if key:
            _SYMBOL_ALIASES[key] = symbol
    glyph = symbol.glyph()
    _SYMBOL_ALIASES[glyph] = symbol
    _SYMBOL_ALIASES[_strip_emoji_variants(glyph)] = symbol


for _symbol, _aliases in {
    DisplayEmojiSymbol.THUMBS_UP: ("thumbs_up", "thumbsup", "+1", "like", "approve"),
    DisplayEmojiSymbol.THUMBS_DOWN: ("thumbs_down", "thumbsdown", "-1", "dislike"),
    DisplayEmojiSymbol.WAVING_HAND: ("waving_hand", "wave", "waving", "hello", "bye"),
    DisplayEmojiSymbol.RAISED_HAND: ("raised_hand", "raise_hand", "hand_up", "stop"),
    DisplayEmojiSymbol.POINTING_HAND: ("pointing_hand", "point", "point_right"),
    DisplayEmojiSymbol.VICTORY_HAND: ("victory_hand", "peace", "v_sign", "victory"),
    DisplayEmojiSymbol.OK_HAND: ("ok_hand", "ok", "okay"),
    DisplayEmojiSymbol.HEART: ("heart", "love"),
    DisplayEmojiSymbol.CHECK: ("check", "check_mark", "checkmark", "tick", "done"),
    DisplayEmojiSymbol.QUESTION: ("question", "question_mark", "help", "unknown"),
    DisplayEmojiSymbol.EXCLAMATION: ("exclamation", "exclamation_mark", "important"),
    DisplayEmojiSymbol.WARNING: ("warning", "alert", "caution"),
    DisplayEmojiSymbol.SPARKLES: ("sparkles", "sparkle", "celebrate"),
}.items():
    _register_symbol_aliases(_symbol, *_aliases)


def _normalize_symbol(value: object | None) -> DisplayEmojiSymbol | None:
    """Normalize one optional emoji symbol token or real glyph."""

    if value is None:
        return None
    if isinstance(value, DisplayEmojiSymbol):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in _SYMBOL_ALIASES:
            return _SYMBOL_ALIASES[stripped]
        stripped = _strip_emoji_variants(stripped)
        if stripped in _SYMBOL_ALIASES:
            return _SYMBOL_ALIASES[stripped]
    alias = _normalize_alias_key(value)
    if not alias:
        return None
    return _SYMBOL_ALIASES.get(alias)


def _normalize_accent(value: object | None) -> str:
    """Normalize one optional emoji accent token."""

    compact = _normalize_alias_key(value, max_len=_MAX_ACCENT_LEN)
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

        object.__setattr__(self, "source", _compact_text(self.source, max_len=_MAX_SOURCE_LEN) or "external")
        object.__setattr__(self, "symbol", (_normalize_symbol(self.symbol) or DisplayEmojiSymbol.HEART).value)
        object.__setattr__(self, "accent", _normalize_accent(self.accent))

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        fallback_updated_at: datetime | None = None,
        default_ttl_s: float = _DEFAULT_EMOJI_CUE_TTL_S,
        max_ttl_s: float = _DEFAULT_EMOJI_CUE_MAX_TTL_S,
    ) -> "DisplayEmojiCue":
        """Build one normalized emoji cue from JSON-style data."""

        safe_now = (fallback_updated_at or _utc_now()).astimezone(timezone.utc)
        updated_at = _normalize_timestamp(payload.get("updated_at")) or safe_now
        default_ttl_s = _coerce_positive_seconds(default_ttl_s, default=_DEFAULT_EMOJI_CUE_TTL_S)
        max_ttl_s = _coerce_positive_seconds(
            max_ttl_s,
            default=_DEFAULT_EMOJI_CUE_MAX_TTL_S,
            maximum=_MAX_ALLOWED_MAX_TTL_S,
        )

        parsed_expires_at = _normalize_timestamp(payload.get("expires_at"))
        expires_at = parsed_expires_at or (updated_at + timedelta(seconds=default_ttl_s))
        expires_at = min(expires_at, updated_at + timedelta(seconds=max_ttl_s))

        raw_symbol = payload.get("symbol")
        normalized_symbol = _normalize_symbol(raw_symbol)
        # BREAKING: explicitly invalid stored symbols are now rejected instead of
        # silently turning into a heart cue.
        if raw_symbol is not None and normalized_symbol is None:
            raise ValueError(f"Unsupported display emoji symbol: {raw_symbol!r}")

        return cls(
            source=_compact_text(payload.get("source"), max_len=_MAX_SOURCE_LEN) or "external",
            updated_at=_format_timestamp(updated_at),
            expires_at=_format_timestamp(expires_at),
            symbol=(normalized_symbol or DisplayEmojiSymbol.HEART).value,
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
    max_ttl_s: float = _DEFAULT_EMOJI_CUE_MAX_TTL_S
    max_file_bytes: int = _MAX_CUE_FILE_BYTES
    _lock_path: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize constructor inputs and derive the lock-file path."""

        self.path = Path(self.path).expanduser()
        self.default_ttl_s = _coerce_positive_seconds(self.default_ttl_s, default=_DEFAULT_EMOJI_CUE_TTL_S)
        self.max_ttl_s = _coerce_positive_seconds(
            self.max_ttl_s,
            default=_DEFAULT_EMOJI_CUE_MAX_TTL_S,
            maximum=_MAX_ALLOWED_MAX_TTL_S,
        )
        self.max_file_bytes = max(512, int(self.max_file_bytes))
        self._lock_path = self.path.with_name(f"{self.path.name}.lock")

    @classmethod
    def from_config(cls, config: _DisplayConfigLike) -> "DisplayEmojiCueStore":
        """Resolve the emoji-cue path from Twinr configuration."""

        project_root = Path(config.project_root).expanduser().resolve()
        configured_path = Path(
            getattr(config, "display_emoji_cue_path", _DEFAULT_EMOJI_CUE_PATH) or _DEFAULT_EMOJI_CUE_PATH
        )
        resolved_path = configured_path if configured_path.is_absolute() else project_root / configured_path
        return cls(
            path=resolved_path,
            default_ttl_s=getattr(config, "display_emoji_cue_ttl_s", _DEFAULT_EMOJI_CUE_TTL_S),
            max_ttl_s=getattr(config, "display_emoji_cue_max_ttl_s", _DEFAULT_EMOJI_CUE_MAX_TTL_S),
            max_file_bytes=getattr(config, "display_emoji_cue_max_bytes", _MAX_CUE_FILE_BYTES),
        )

    @contextmanager
    def _lock(self, *, shared: bool) -> Iterator[None]:
        """Serialize readers/writers across processes using an advisory lock file."""

        if not self.path.parent.exists():
            raise FileNotFoundError(self.path.parent)
        if fcntl is None:  # pragma: no cover - not expected on Raspberry Pi / Linux.
            yield
            return

        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(self._lock_path, flags, 0o600)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise RuntimeError(f"Unsafe display emoji lock path: {self._lock_path}") from exc
            raise

        try:
            fcntl.flock(fd, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _fsync_parent_dir(self) -> None:
        """Best-effort fsync of the parent directory for rename/unlink durability."""

        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
        try:
            fd = os.open(self.path.parent, flags)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    def _read_payload_bytes(self) -> tuple[bytes, datetime] | None:
        """Open the cue file without following a final-component symlink and read it safely."""

        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(self.path, flags)
        except FileNotFoundError:
            return None
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENXIO}:
                _LOGGER.warning("Ignoring unsafe display emoji cue path at %s.", self.path)
                return None
            raise

        try:
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):
                _LOGGER.warning("Ignoring display emoji cue at %s because it is not a regular file.", self.path)
                return None
            if file_stat.st_size > self.max_file_bytes:
                _LOGGER.warning(
                    "Ignoring display emoji cue at %s because it exceeds %s bytes.",
                    self.path,
                    self.max_file_bytes,
                )
                return None

            chunks = bytearray()
            while True:
                block = os.read(fd, min(4096, self.max_file_bytes + 1 - len(chunks)))
                if not block:
                    break
                chunks.extend(block)
                if len(chunks) > self.max_file_bytes:
                    _LOGGER.warning(
                        "Ignoring display emoji cue at %s because it exceeds %s bytes while reading.",
                        self.path,
                        self.max_file_bytes,
                    )
                    return None

            updated_at = datetime.fromtimestamp(file_stat.st_mtime_ns / 1_000_000_000, tz=timezone.utc)
            return bytes(chunks), updated_at
        finally:
            os.close(fd)

    def load(self) -> DisplayEmojiCue | None:
        """Load the current emoji cue, if one exists and parses."""

        if not self.path.parent.exists():
            return None

        try:
            with self._lock(shared=True):
                result = self._read_payload_bytes()
                if result is None:
                    return None
                raw, fallback_updated_at = result
                try:
                    payload = json.loads(raw)
                except Exception:
                    _LOGGER.warning("Failed to parse display emoji cue from %s.", self.path, exc_info=True)
                    return None
                if not isinstance(payload, dict):
                    _LOGGER.warning(
                        "Ignoring invalid display emoji cue payload at %s because it is not an object.",
                        self.path,
                    )
                    return None
                try:
                    return DisplayEmojiCue.from_dict(
                        payload,
                        fallback_updated_at=fallback_updated_at,
                        default_ttl_s=self.default_ttl_s,
                        max_ttl_s=self.max_ttl_s,
                    )
                except Exception:
                    _LOGGER.warning("Ignoring invalid display emoji cue payload at %s.", self.path, exc_info=True)
                    return None
        except FileNotFoundError:
            return None
        except RuntimeError:
            _LOGGER.warning("Unsafe display emoji lock path at %s.", self._lock_path)
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
        """Persist one normalized emoji cue atomically and durably."""

        safe_now = (now or _utc_now()).astimezone(timezone.utc)
        ttl_s = _coerce_positive_seconds(
            hold_seconds,
            default=self.default_ttl_s,
            maximum=self.max_ttl_s,
        )
        normalized = DisplayEmojiCue(
            source=cue.source,
            updated_at=_format_timestamp(safe_now),
            expires_at=_format_timestamp(safe_now + timedelta(seconds=ttl_s)),
            symbol=cue.symbol,
            accent=cue.accent,
        )
        encoded = json.dumps(normalized.to_dict(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > self.max_file_bytes:
            raise ValueError(f"Display emoji cue payload exceeds {self.max_file_bytes} bytes after serialization.")

        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        with self._lock(shared=False):
            fd, tmp_name = tempfile.mkstemp(
                prefix=f"{self.path.name}.",
                suffix=".tmp",
                dir=self.path.parent,
                text=False,
            )
            try:
                with os.fdopen(fd, "wb") as tmp_file:
                    tmp_file.write(encoded)
                    tmp_file.flush()
                    os.fsync(tmp_file.fileno())
                os.replace(tmp_name, self.path)
                self._fsync_parent_dir()
            except Exception:
                try:
                    Path(tmp_name).unlink(missing_ok=True)
                except Exception:
                    _LOGGER.warning("Failed to clean up temporary emoji cue file %s.", tmp_name, exc_info=True)
                raise

        return normalized

    def clear(self) -> None:
        """Remove the current emoji cue artifact if one exists."""

        if not self.path.parent.exists():
            return

        try:
            with self._lock(shared=False):
                try:
                    path_stat = self.path.lstat()
                except FileNotFoundError:
                    return
                if stat.S_ISDIR(path_stat.st_mode):
                    _LOGGER.warning("Refusing to clear display emoji cue at %s because it is a directory.", self.path)
                    return
                self.path.unlink(missing_ok=True)
                self._fsync_parent_dir()
        except FileNotFoundError:
            return
        except RuntimeError:
            _LOGGER.warning("Unsafe display emoji lock path at %s.", self._lock_path)
            return


@dataclass(slots=True)
class DisplayEmojiController:
    """Persist producer-facing HDMI emoji cues through the cue store."""

    store: DisplayEmojiCueStore
    default_source: str = "external"

    @classmethod
    def from_config(
        cls,
        config: _DisplayConfigLike,
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

        normalized_symbol = _normalize_symbol(symbol)
        if normalized_symbol is None:
            _LOGGER.warning(
                "Unsupported display emoji symbol %r; falling back to %s.",
                symbol,
                DisplayEmojiSymbol.HEART.value,
            )
            normalized_symbol = DisplayEmojiSymbol.HEART

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
