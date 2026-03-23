"""Persist short-lived reserve-bus feedback that can reshape today's plan.

The ambient impulse history keeps a durable record of what Twinr showed and how
the user later reacted. That is valuable for long-term personality learning,
but the HDMI reserve lane also benefits from one faster loop: when a visible
card gets picked up immediately, the remaining cards for the current day should
adapt sooner.

This module owns that bounded short-lived signal:

- one latest reserve-bus feedback hint at a time
- explicit expiry so the effect fades away naturally
- generic reaction labels such as ``immediate_engagement`` or ``ignored``
- no transcript storage and no topic-specific rules
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
import math
from pathlib import Path

from twinr.agent.base_agent.config import TwinrConfig

_DEFAULT_FEEDBACK_PATH = "artifacts/stores/ops/display_reserve_bus_feedback.json"
_DEFAULT_FEEDBACK_TTL_S = 4.0 * 60.0 * 60.0
_ALLOWED_REACTIONS = frozenset(
    {
        "immediate_engagement",
        "engaged",
        "cooled",
        "avoided",
        "ignored",
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
    """Serialize one aware timestamp as UTC ISO-8601 text."""

    return value.astimezone(timezone.utc).isoformat()


def _compact_text(value: object | None, *, max_len: int) -> str:
    """Collapse one arbitrary value into bounded single-line text."""

    if value is None:
        return ""
    compact = " ".join(str(value).split()).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _normalize_reaction(value: object | None) -> str:
    """Normalize one bounded reserve-bus reaction token."""

    compact = _compact_text(value, max_len=40).lower().replace("-", "_").replace(" ", "_")
    if compact not in _ALLOWED_REACTIONS:
        return "engaged"
    return compact


def _normalize_intensity(value: object | None) -> float:
    """Clamp one optional feedback intensity into the inclusive 0..1 range."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


@dataclass(frozen=True, slots=True)
class DisplayReserveBusFeedbackSignal:
    """Describe one short-lived reserve-bus feedback hint.

    Attributes:
        source: Producer of the feedback hint.
        requested_at: UTC timestamp when the hint was recorded.
        expires_at: UTC timestamp after which the hint should no longer bias
            reserve planning.
        topic_key: Stable normalized topic key this hint refers to.
        reaction: Coarse reaction token such as ``immediate_engagement``.
        intensity: Relative strength of the reaction in the 0..1 range.
        reason: Short auditable explanation for the hint.
    """

    source: str = "display_reserve_card"
    requested_at: str | None = None
    expires_at: str | None = None
    topic_key: str = ""
    reaction: str = "engaged"
    intensity: float = 0.0
    reason: str = ""

    def __post_init__(self) -> None:
        """Normalize direct constructor calls into the canonical contract."""

        object.__setattr__(self, "source", _compact_text(self.source, max_len=80) or "display_reserve_card")
        object.__setattr__(self, "topic_key", _compact_text(self.topic_key, max_len=96).casefold())
        object.__setattr__(self, "reaction", _normalize_reaction(self.reaction))
        object.__setattr__(self, "intensity", _normalize_intensity(self.intensity))
        requested_at = _normalize_timestamp(self.requested_at) or _utc_now()
        expires_at = _normalize_timestamp(self.expires_at) or (
            requested_at + timedelta(seconds=_DEFAULT_FEEDBACK_TTL_S)
        )
        object.__setattr__(self, "requested_at", _format_timestamp(requested_at))
        object.__setattr__(self, "expires_at", _format_timestamp(expires_at))
        object.__setattr__(self, "reason", _compact_text(self.reason, max_len=160))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DisplayReserveBusFeedbackSignal":
        """Build one feedback hint from persisted JSON-style data."""

        return cls(
            source=_compact_text(payload.get("source"), max_len=80),
            requested_at=_compact_text(payload.get("requested_at"), max_len=64) or None,
            expires_at=_compact_text(payload.get("expires_at"), max_len=64) or None,
            topic_key=_compact_text(payload.get("topic_key"), max_len=96),
            reaction=_compact_text(payload.get("reaction"), max_len=40),
            intensity=payload.get("intensity"),
            reason=_compact_text(payload.get("reason"), max_len=160),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the feedback hint into JSON-safe data."""

        return asdict(self)

    def requested_at_datetime(self) -> datetime:
        """Return the requested-at timestamp as an aware UTC datetime."""

        return _normalize_timestamp(self.requested_at) or _utc_now()

    def expires_at_datetime(self) -> datetime:
        """Return the expiry timestamp as an aware UTC datetime."""

        return _normalize_timestamp(self.expires_at) or (
            self.requested_at_datetime() + timedelta(seconds=_DEFAULT_FEEDBACK_TTL_S)
        )

    def is_active(self, *, now: datetime | None = None) -> bool:
        """Return whether this hint should still bias reserve planning."""

        return self.expires_at_datetime() >= (now or _utc_now()).astimezone(timezone.utc)


@dataclass(slots=True)
class DisplayReserveBusFeedbackStore:
    """Read and write one short-lived reserve-bus feedback hint."""

    path: Path
    default_ttl_s: float = _DEFAULT_FEEDBACK_TTL_S

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayReserveBusFeedbackStore":
        """Resolve the feedback artifact path from configuration."""

        project_root = Path(config.project_root).expanduser().resolve()
        configured = Path(
            getattr(config, "display_reserve_bus_feedback_path", _DEFAULT_FEEDBACK_PATH)
            or _DEFAULT_FEEDBACK_PATH
        )
        resolved = configured if configured.is_absolute() else project_root / configured
        return cls(path=resolved, default_ttl_s=_DEFAULT_FEEDBACK_TTL_S)

    def load(self) -> DisplayReserveBusFeedbackSignal | None:
        """Load the current reserve-bus feedback hint, if present."""

        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            _LOGGER.warning("Failed to read display reserve bus feedback from %s.", self.path, exc_info=True)
            return None
        if not isinstance(payload, Mapping):
            _LOGGER.warning(
                "Ignoring invalid display reserve bus feedback payload at %s because it is not an object.",
                self.path,
            )
            return None
        try:
            return DisplayReserveBusFeedbackSignal.from_dict(payload)
        except Exception:
            _LOGGER.warning("Ignoring invalid display reserve bus feedback payload at %s.", self.path, exc_info=True)
            return None

    def load_active(self, *, now: datetime | None = None) -> DisplayReserveBusFeedbackSignal | None:
        """Load the current hint only when it is still active."""

        signal = self.load()
        if signal is None:
            return None
        if not signal.is_active(now=now):
            return None
        return signal

    def save(self, signal: DisplayReserveBusFeedbackSignal) -> DisplayReserveBusFeedbackSignal:
        """Persist one reserve-bus feedback hint."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(signal.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return signal

    def clear(self) -> None:
        """Remove the persisted feedback artifact when it exists."""

        try:
            self.path.unlink()
        except FileNotFoundError:
            return

    def record_reaction(
        self,
        *,
        topic_key: str,
        reaction: str,
        intensity: float,
        reason: str,
        now: datetime,
        source: str = "display_reserve_card",
        ttl_s: float | None = None,
    ) -> DisplayReserveBusFeedbackSignal:
        """Persist one new short-lived reserve-bus reaction hint."""

        effective_now = now.astimezone(timezone.utc)
        lifetime_s = max(60.0, float(ttl_s if ttl_s is not None else self.default_ttl_s))
        signal = DisplayReserveBusFeedbackSignal(
            source=source,
            requested_at=_format_timestamp(effective_now),
            expires_at=_format_timestamp(effective_now + timedelta(seconds=lifetime_s)),
            topic_key=topic_key,
            reaction=reaction,
            intensity=intensity,
            reason=reason,
        )
        return self.save(signal)


__all__ = ["DisplayReserveBusFeedbackSignal", "DisplayReserveBusFeedbackStore"]
