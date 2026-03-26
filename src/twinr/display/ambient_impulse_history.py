"""Persist reserve-lane exposure history and resolved user reactions.

The active ambient-cue artifact only answers one question: what is visible
right now? It should stay tiny and render-focused.

This module owns the separate durable history of reserve-card exposures so
other layers can learn from them without coupling that learning logic into the
display renderer or the proactive publish loop.

The contract stays bounded:

- append one exposure when a reserve card is actually shown
- keep a short rolling history instead of unbounded telemetry
- resolve an exposure later with one coarse reaction outcome
- never store raw transcripts or free-form logs here
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import math
from pathlib import Path

from twinr.agent.base_agent.config import TwinrConfig

_DEFAULT_HISTORY_PATH = "artifacts/stores/ops/display_ambient_impulse_history.json"
_DEFAULT_MAX_ENTRIES = 256
_ALLOWED_RESPONSE_STATUS = frozenset({"pending", "engaged", "cooled", "avoided", "ignored", "neutral"})
_ALLOWED_RESPONSE_SENTIMENT = frozenset({"positive", "negative", "neutral", "mixed", "unknown"})

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
    """Normalize one bounded single-line text field."""

    if value is None:
        return ""
    compact = " ".join(str(value).split()).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _normalize_mapping(value: Mapping[str, object] | None) -> dict[str, object] | None:
    """Return one plain JSON-safe mapping copy when present."""

    if value is None:
        return None
    normalized: dict[str, object] = {}
    for raw_key, raw_value in value.items():
        key = _compact_text(raw_key, max_len=80)
        if not key:
            continue
        if isinstance(raw_value, Mapping):
            child = _normalize_mapping(raw_value)
            if child:
                normalized[key] = child
            continue
        if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes, bytearray)):
            items: list[object] = []
            for item in raw_value:
                compact = _compact_text(item, max_len=160)
                if compact:
                    items.append(compact)
            if items:
                normalized[key] = items
            continue
        compact = _compact_text(raw_value, max_len=160)
        if compact:
            normalized[key] = compact
    return normalized or None


def _normalize_text_tuple(values: Iterable[object] | None, *, max_items: int, max_len: int) -> tuple[str, ...]:
    """Normalize one bounded ordered text tuple while removing duplicates."""

    if values is None:
        return ()
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        compact = _compact_text(value, max_len=max_len)
        if not compact:
            continue
        key = compact.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(compact)
        if len(ordered) >= max_items:
            break
    return tuple(ordered)


def _normalize_optional_float(value: object | None) -> float | None:
    """Return one optional finite float for persisted payloads."""

    if value is None:
        return None
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalize_response_status(value: object | None) -> str:
    """Normalize one coarse feedback status token."""

    compact = _compact_text(value, max_len=32).lower().replace("-", "_").replace(" ", "_")
    if compact not in _ALLOWED_RESPONSE_STATUS:
        return "pending"
    return compact


def _normalize_response_sentiment(value: object | None) -> str:
    """Normalize one coarse sentiment token."""

    compact = _compact_text(value, max_len=32).lower().replace("-", "_").replace(" ", "_")
    if compact not in _ALLOWED_RESPONSE_SENTIMENT:
        return "unknown"
    return compact


def _exposure_id(
    *,
    shown_at: datetime,
    topic_key: str,
    headline: str,
) -> str:
    """Return one stable exposure id for the shown reserve card."""

    digest = hashlib.sha1(f"{shown_at.isoformat()}::{topic_key}::{headline}".encode("utf-8")).hexdigest()
    return f"ambient_exposure:{digest[:16]}"


@dataclass(frozen=True, slots=True)
class DisplayAmbientImpulseExposure:
    """Describe one shown reserve-lane card and its later coarse outcome."""

    exposure_id: str
    source: str
    topic_key: str
    title: str
    headline: str
    body: str
    action: str
    attention_state: str
    shown_at: str
    expires_at: str
    semantic_topic_key: str = ""
    match_anchors: tuple[str, ...] = ()
    response_status: str = "pending"
    response_sentiment: str = "unknown"
    response_at: str | None = None
    response_mode: str = ""
    response_latency_seconds: float | None = None
    response_turn_id: str | None = None
    response_target: str | None = None
    response_summary: str = ""
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        """Normalize one exposure record into a stable bounded contract."""

        object.__setattr__(self, "exposure_id", _compact_text(self.exposure_id, max_len=64))
        object.__setattr__(self, "source", _compact_text(self.source, max_len=48) or "ambient_impulse")
        object.__setattr__(self, "topic_key", _compact_text(self.topic_key, max_len=96).casefold())
        semantic_topic_key = _compact_text(self.semantic_topic_key, max_len=96).casefold()
        object.__setattr__(self, "semantic_topic_key", semantic_topic_key or self.topic_key)
        object.__setattr__(self, "title", _compact_text(self.title, max_len=96))
        object.__setattr__(self, "headline", _compact_text(self.headline, max_len=160))
        object.__setattr__(self, "body", _compact_text(self.body, max_len=160))
        object.__setattr__(self, "action", _compact_text(self.action, max_len=24).lower() or "hint")
        object.__setattr__(
            self,
            "attention_state",
            _compact_text(self.attention_state, max_len=32).lower() or "background",
        )
        shown_at = _normalize_timestamp(self.shown_at) or _utc_now()
        expires_at = _normalize_timestamp(self.expires_at) or (shown_at + timedelta(minutes=10))
        object.__setattr__(self, "shown_at", _format_timestamp(shown_at))
        object.__setattr__(self, "expires_at", _format_timestamp(expires_at))
        object.__setattr__(
            self,
            "match_anchors",
            _normalize_text_tuple(self.match_anchors, max_items=8, max_len=160),
        )
        object.__setattr__(self, "response_status", _normalize_response_status(self.response_status))
        object.__setattr__(self, "response_sentiment", _normalize_response_sentiment(self.response_sentiment))
        response_at = _normalize_timestamp(self.response_at)
        object.__setattr__(self, "response_at", _format_timestamp(response_at) if response_at is not None else None)
        object.__setattr__(self, "response_mode", _compact_text(self.response_mode, max_len=48).lower())
        latency_seconds = None
        try:
            if self.response_latency_seconds is not None:
                latency_seconds = max(0.0, float(self.response_latency_seconds))
        except (TypeError, ValueError):
            latency_seconds = None
        object.__setattr__(self, "response_latency_seconds", latency_seconds)
        object.__setattr__(self, "response_turn_id", _compact_text(self.response_turn_id, max_len=96) or None)
        object.__setattr__(self, "response_target", _compact_text(self.response_target, max_len=96) or None)
        object.__setattr__(self, "response_summary", _compact_text(self.response_summary, max_len=220))
        object.__setattr__(self, "metadata", _normalize_mapping(self.metadata))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DisplayAmbientImpulseExposure":
        """Build one exposure record from persisted JSON-style data."""

        raw_match_anchors = payload.get("match_anchors")
        raw_metadata = payload.get("metadata")
        return cls(
            exposure_id=_compact_text(payload.get("exposure_id"), max_len=64),
            source=_compact_text(payload.get("source"), max_len=48) or "ambient_impulse",
            topic_key=_compact_text(payload.get("topic_key"), max_len=96),
            semantic_topic_key=_compact_text(payload.get("semantic_topic_key"), max_len=96),
            title=_compact_text(payload.get("title"), max_len=96),
            headline=_compact_text(payload.get("headline"), max_len=160),
            body=_compact_text(payload.get("body"), max_len=160),
            action=_compact_text(payload.get("action"), max_len=24),
            attention_state=_compact_text(payload.get("attention_state"), max_len=32),
            shown_at=_compact_text(payload.get("shown_at"), max_len=64),
            expires_at=_compact_text(payload.get("expires_at"), max_len=64),
            match_anchors=_normalize_text_tuple(
                raw_match_anchors
                if isinstance(raw_match_anchors, Sequence)
                and not isinstance(raw_match_anchors, (str, bytes, bytearray))
                else None,
                max_items=8,
                max_len=160,
            ),
            response_status=_compact_text(payload.get("response_status"), max_len=32),
            response_sentiment=_compact_text(payload.get("response_sentiment"), max_len=32),
            response_at=_compact_text(payload.get("response_at"), max_len=64) or None,
            response_mode=_compact_text(payload.get("response_mode"), max_len=48),
            response_latency_seconds=_normalize_optional_float(payload.get("response_latency_seconds")),
            response_turn_id=_compact_text(payload.get("response_turn_id"), max_len=96) or None,
            response_target=_compact_text(payload.get("response_target"), max_len=96) or None,
            response_summary=_compact_text(payload.get("response_summary"), max_len=220),
            metadata=raw_metadata if isinstance(raw_metadata, Mapping) else None,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the exposure record into a JSON-safe mapping."""

        return asdict(self)

    def shown_at_datetime(self) -> datetime:
        """Return the shown-at timestamp as an aware UTC datetime."""

        return _normalize_timestamp(self.shown_at) or _utc_now()

    def semantic_key(self) -> str:
        """Return the grouped semantic topic key for this exposure."""

        return _compact_text(self.semantic_topic_key, max_len=96).casefold() or self.topic_key

    def expires_at_datetime(self) -> datetime:
        """Return the expiry timestamp as an aware UTC datetime."""

        return _normalize_timestamp(self.expires_at) or self.shown_at_datetime()

    def anchors(self) -> tuple[str, ...]:
        """Return the ordered matching anchors for feedback correlation."""

        return _normalize_text_tuple(
            (self.semantic_topic_key, self.topic_key, self.title, self.headline, self.body, *self.match_anchors),
            max_items=10,
            max_len=160,
        )

    def with_feedback(
        self,
        *,
        response_status: str,
        response_sentiment: str,
        response_at: datetime,
        response_mode: str,
        response_latency_seconds: float | None,
        response_turn_id: str | None,
        response_target: str | None,
        response_summary: str,
    ) -> "DisplayAmbientImpulseExposure":
        """Return one copy annotated with the resolved user reaction."""

        return DisplayAmbientImpulseExposure(
            exposure_id=self.exposure_id,
            source=self.source,
            topic_key=self.topic_key,
            semantic_topic_key=self.semantic_topic_key,
            title=self.title,
            headline=self.headline,
            body=self.body,
            action=self.action,
            attention_state=self.attention_state,
            shown_at=self.shown_at,
            expires_at=self.expires_at,
            match_anchors=self.match_anchors,
            response_status=response_status,
            response_sentiment=response_sentiment,
            response_at=_format_timestamp(response_at),
            response_mode=response_mode,
            response_latency_seconds=response_latency_seconds,
            response_turn_id=response_turn_id,
            response_target=response_target,
            response_summary=response_summary,
            metadata=self.metadata,
        )


@dataclass(slots=True)
class DisplayAmbientImpulseHistoryStore:
    """Persist one bounded history of shown reserve cards and coarse outcomes."""

    path: Path
    max_entries: int = _DEFAULT_MAX_ENTRIES

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayAmbientImpulseHistoryStore":
        """Resolve the history artifact path from configuration."""

        project_root = Path(config.project_root).expanduser().resolve()
        configured = Path(
            getattr(config, "display_ambient_impulse_history_path", _DEFAULT_HISTORY_PATH)
            or _DEFAULT_HISTORY_PATH
        )
        resolved = configured if configured.is_absolute() else project_root / configured
        return cls(path=resolved, max_entries=_DEFAULT_MAX_ENTRIES)

    def load(self) -> tuple[DisplayAmbientImpulseExposure, ...]:
        """Load the current exposure history snapshot."""

        if not self.path.exists():
            return ()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            _LOGGER.warning("Failed to read display ambient impulse history from %s.", self.path, exc_info=True)
            return ()
        raw_entries = payload.get("exposures") if isinstance(payload, Mapping) else None
        if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes, bytearray)):
            _LOGGER.warning(
                "Ignoring invalid display ambient impulse history payload at %s because exposures is not a sequence.",
                self.path,
            )
            return ()
        entries: list[DisplayAmbientImpulseExposure] = []
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, Mapping):
                continue
            try:
                entries.append(DisplayAmbientImpulseExposure.from_dict(raw_entry))
            except Exception:
                _LOGGER.warning(
                    "Ignoring invalid display ambient impulse history entry at %s.",
                    self.path,
                    exc_info=True,
                )
        return tuple(entries)

    def save_all(self, exposures: Sequence[DisplayAmbientImpulseExposure]) -> tuple[DisplayAmbientImpulseExposure, ...]:
        """Persist one bounded ordered exposure history snapshot."""

        normalized = tuple(exposures)[-max(1, int(self.max_entries)) :]
        payload = {"exposures": [entry.to_dict() for entry in normalized]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return normalized

    def append_exposure(
        self,
        *,
        source: str,
        topic_key: str,
        semantic_topic_key: str | None = None,
        title: str,
        headline: str,
        body: str,
        action: str,
        attention_state: str,
        shown_at: datetime,
        expires_at: datetime,
        match_anchors: Sequence[object] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> DisplayAmbientImpulseExposure:
        """Append one shown reserve-card exposure to the bounded history."""

        exposure = DisplayAmbientImpulseExposure(
            exposure_id=_exposure_id(
                shown_at=shown_at.astimezone(timezone.utc),
                topic_key=_compact_text(topic_key, max_len=96).casefold(),
                headline=_compact_text(headline, max_len=160),
            ),
            source=source,
            topic_key=topic_key,
            semantic_topic_key=semantic_topic_key or topic_key,
            title=title,
            headline=headline,
            body=body,
            action=action,
            attention_state=attention_state,
            shown_at=_format_timestamp(shown_at.astimezone(timezone.utc)),
            expires_at=_format_timestamp(expires_at.astimezone(timezone.utc)),
            match_anchors=_normalize_text_tuple(match_anchors, max_items=8, max_len=160),
            metadata=metadata,
        )
        history = self.load()
        self.save_all((*history, exposure))
        return exposure

    def load_pending(
        self,
        *,
        now: datetime,
        max_age_hours: float = 12.0,
        limit: int = 6,
    ) -> tuple[DisplayAmbientImpulseExposure, ...]:
        """Return recent unresolved exposures in newest-first order."""

        effective_now = now.astimezone(timezone.utc)
        minimum_shown_at = effective_now - timedelta(hours=max(0.1, float(max_age_hours)))
        pending = [
            exposure
            for exposure in self.load()
            if exposure.response_status == "pending"
            and exposure.shown_at_datetime() <= effective_now
            and exposure.shown_at_datetime() >= minimum_shown_at
        ]
        pending.sort(key=lambda entry: entry.shown_at_datetime(), reverse=True)
        return tuple(pending[: max(1, int(limit))])

    def topic_exposure_count(
        self,
        *,
        topic_key: str,
        now: datetime,
        within_days: float = 14.0,
    ) -> int:
        """Return how often the same normalized topic was shown recently."""

        normalized_topic = _compact_text(topic_key, max_len=96).casefold()
        if not normalized_topic:
            return 0
        effective_now = now.astimezone(timezone.utc)
        minimum_shown_at = effective_now - timedelta(days=max(0.1, float(within_days)))
        return sum(
            1
            for exposure in self.load()
            if exposure.semantic_topic_key == normalized_topic and exposure.shown_at_datetime() >= minimum_shown_at
        )

    def resolve_feedback(
        self,
        *,
        exposure_id: str,
        response_status: str,
        response_sentiment: str,
        response_at: datetime,
        response_mode: str,
        response_latency_seconds: float | None,
        response_turn_id: str | None,
        response_target: str | None,
        response_summary: str,
    ) -> DisplayAmbientImpulseExposure | None:
        """Resolve one exposure with the observed later user reaction."""

        exposures = list(self.load())
        resolved: DisplayAmbientImpulseExposure | None = None
        for index, exposure in enumerate(exposures):
            if exposure.exposure_id != exposure_id:
                continue
            resolved = exposure.with_feedback(
                response_status=response_status,
                response_sentiment=response_sentiment,
                response_at=response_at,
                response_mode=response_mode,
                response_latency_seconds=response_latency_seconds,
                response_turn_id=response_turn_id,
                response_target=response_target,
                response_summary=response_summary,
            )
            exposures[index] = resolved
            break
        if resolved is None:
            return None
        self.save_all(tuple(exposures))
        return resolved


__all__ = ["DisplayAmbientImpulseExposure", "DisplayAmbientImpulseHistoryStore"]
