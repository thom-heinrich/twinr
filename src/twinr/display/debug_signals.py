"""Persist bounded HDMI header debug signals for sensor-fusion inspection.

The senior-facing HDMI surface stays face-first, but operator debugging still
benefits from a tiny explicit signal lane that can mirror currently active
camera and fusion states such as ``MOTION_STILL`` or ``POSSIBLE_FALL`` without
teaching the generic runtime snapshot schema about those transient internals.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path

from twinr.agent.base_agent.config import TwinrConfig


_DEFAULT_DEBUG_SIGNAL_TTL_S = 6.0
_DEFAULT_DEBUG_SIGNAL_PATH = "artifacts/stores/ops/display_debug_signals.json"
_ALLOWED_ACCENTS = frozenset({"neutral", "info", "success", "warning", "alert"})

_LOGGER = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Return the current UTC wall clock."""

    return datetime.now(timezone.utc)


def _normalize_timestamp(value: object | None) -> datetime | None:
    """Parse one optional timestamp into an aware UTC ``datetime``."""

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


def _normalize_text(value: object | None, *, fallback: str = "") -> str:
    """Return one bounded printable text token."""

    compact = " ".join(str(value or "").split()).strip()
    return compact or fallback


def _normalize_key(value: object | None) -> str:
    """Normalize one signal key into a stable snake-style token."""

    text = _normalize_text(value).lower().replace("-", "_").replace(" ", "_")
    text = "".join(ch for ch in text if ch.isalnum() or ch == "_")
    return text or "signal"


def _normalize_label(value: object | None, *, fallback: str) -> str:
    """Normalize one signal label into a stable uppercase pill label."""

    compact = _normalize_text(value, fallback=fallback).upper().replace("-", "_").replace(" ", "_")
    compact = "".join(ch for ch in compact if ch.isalnum() or ch == "_")
    return compact or fallback


def _normalize_accent(value: object | None) -> str:
    """Normalize one signal accent token."""

    compact = _normalize_text(value).lower()
    if compact not in _ALLOWED_ACCENTS:
        return "neutral"
    return compact


@dataclass(frozen=True, slots=True)
class DisplayDebugSignal:
    """Describe one bounded HDMI header debug-signal pill."""

    key: str
    label: str
    accent: str = "neutral"
    priority: int = 0

    def __post_init__(self) -> None:
        """Normalize the signal into a stable display-safe payload."""

        normalized_key = _normalize_key(self.key)
        normalized_label = _normalize_label(self.label, fallback=normalized_key.upper())
        object.__setattr__(self, "key", normalized_key)
        object.__setattr__(self, "label", normalized_label)
        object.__setattr__(self, "accent", _normalize_accent(self.accent))
        object.__setattr__(self, "priority", int(self.priority))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DisplayDebugSignal":
        """Build one signal from JSON-style data."""

        return cls(
            key=_normalize_text(payload.get("key"), fallback="signal"),
            label=_normalize_text(payload.get("label"), fallback="SIGNAL"),
            accent=_normalize_text(payload.get("accent"), fallback="neutral"),
            priority=int(payload.get("priority", 0) or 0),
        )

    def signature(self) -> tuple[object, ...]:
        """Return a stable render-signature fragment for this signal."""

        return (self.key, self.label, self.accent, self.priority)


@dataclass(frozen=True, slots=True)
class DisplayDebugSignalSnapshot:
    """Describe one short-lived batch of active HDMI debug signals."""

    source: str = "external"
    updated_at: str | None = None
    expires_at: str | None = None
    signals: tuple[DisplayDebugSignal, ...] = ()

    def __post_init__(self) -> None:
        """Normalize the snapshot and deduplicate signals by key."""

        object.__setattr__(self, "source", _normalize_text(self.source, fallback="external"))
        deduped: list[DisplayDebugSignal] = []
        seen_keys: set[str] = set()
        for signal in self.signals:
            normalized = signal if isinstance(signal, DisplayDebugSignal) else DisplayDebugSignal.from_dict(signal)
            if normalized.key in seen_keys:
                continue
            seen_keys.add(normalized.key)
            deduped.append(normalized)
        object.__setattr__(self, "signals", tuple(deduped))

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        fallback_updated_at: datetime | None = None,
        default_ttl_s: float = _DEFAULT_DEBUG_SIGNAL_TTL_S,
    ) -> "DisplayDebugSignalSnapshot":
        """Build one normalized snapshot from JSON-style data."""

        safe_now = (fallback_updated_at or _utc_now()).astimezone(timezone.utc)
        updated_at = _normalize_timestamp(payload.get("updated_at")) or safe_now
        expires_at = _normalize_timestamp(payload.get("expires_at"))
        if expires_at is None:
            expires_at = updated_at + timedelta(seconds=max(0.1, float(default_ttl_s)))
        raw_signals = payload.get("signals")
        signal_items = raw_signals if isinstance(raw_signals, list) else []
        return cls(
            source=_normalize_text(payload.get("source"), fallback="external"),
            updated_at=_format_timestamp(updated_at),
            expires_at=_format_timestamp(expires_at),
            signals=tuple(
                DisplayDebugSignal.from_dict(item)
                for item in signal_items
                if isinstance(item, Mapping)
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the snapshot into a JSON-safe mapping."""

        payload = asdict(self)
        payload["signals"] = [asdict(signal) for signal in self.signals]
        return payload

    def is_active(self, *, now: datetime | None = None) -> bool:
        """Return whether the snapshot should still affect the HDMI header."""

        expires_at = _normalize_timestamp(self.expires_at)
        if expires_at is None:
            return True
        return expires_at >= (now or _utc_now()).astimezone(timezone.utc)

    def signature(self) -> tuple[object, ...]:
        """Return a stable render-signature fragment for the whole batch."""

        return tuple(signal.signature() for signal in self.signals)


@dataclass(slots=True)
class DisplayDebugSignalStore:
    """Read and write the optional HDMI debug-signal artifact."""

    path: Path
    default_ttl_s: float = _DEFAULT_DEBUG_SIGNAL_TTL_S

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayDebugSignalStore":
        """Resolve the debug-signal path from Twinr configuration."""

        project_root = Path(config.project_root).expanduser().resolve()
        configured_path = Path(
            getattr(config, "display_debug_signal_path", _DEFAULT_DEBUG_SIGNAL_PATH) or _DEFAULT_DEBUG_SIGNAL_PATH
        )
        resolved_path = configured_path if configured_path.is_absolute() else project_root / configured_path
        return cls(
            path=resolved_path,
            default_ttl_s=float(
                getattr(config, "display_debug_signal_ttl_s", _DEFAULT_DEBUG_SIGNAL_TTL_S)
                or _DEFAULT_DEBUG_SIGNAL_TTL_S
            ),
        )

    def load(self) -> DisplayDebugSignalSnapshot | None:
        """Load the current debug-signal snapshot, if one exists and parses."""

        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            _LOGGER.warning("Failed to read display debug signals from %s.", self.path, exc_info=True)
            return None
        if not isinstance(payload, dict):
            _LOGGER.warning(
                "Ignoring invalid display debug signal payload at %s because it is not an object.",
                self.path,
            )
            return None
        fallback_updated_at = datetime.fromtimestamp(self.path.stat().st_mtime, tz=timezone.utc)
        try:
            return DisplayDebugSignalSnapshot.from_dict(
                payload,
                fallback_updated_at=fallback_updated_at,
                default_ttl_s=self.default_ttl_s,
            )
        except Exception:
            _LOGGER.warning("Ignoring invalid display debug signal payload at %s.", self.path, exc_info=True)
            return None

    def load_active(self, *, now: datetime | None = None) -> DisplayDebugSignalSnapshot | None:
        """Load the current snapshot only while it is still active."""

        snapshot = self.load()
        if snapshot is None or not snapshot.is_active(now=now):
            return None
        return snapshot

    def save(
        self,
        snapshot: DisplayDebugSignalSnapshot,
        *,
        hold_seconds: float | None = None,
        now: datetime | None = None,
    ) -> DisplayDebugSignalSnapshot:
        """Persist one normalized debug-signal snapshot atomically."""

        safe_now = (now or _utc_now()).astimezone(timezone.utc)
        ttl_s = max(0.1, float(self.default_ttl_s if hold_seconds is None else hold_seconds))
        normalized = DisplayDebugSignalSnapshot(
            source=snapshot.source,
            updated_at=_format_timestamp(safe_now),
            expires_at=_format_timestamp(safe_now + timedelta(seconds=ttl_s)),
            signals=snapshot.signals,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp_path.write_text(
            json.dumps(normalized.to_dict(), ensure_ascii=True, separators=(",", ":")),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)
        return normalized
