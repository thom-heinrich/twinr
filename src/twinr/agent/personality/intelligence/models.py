"""Define persistent RSS/world-intelligence models for evolving personality.

The world-intelligence layer keeps three concerns explicit:

- durable feed subscriptions chosen by the installer, the user, or future
  reflection/recalibration passes
- refresh/discovery runtime state that limits how often Twinr looks for new
  sources or polls existing feeds
- slowly learned calibration signals that summarize which topics and regions
  deserve ongoing source coverage
- reflected situational-awareness threads that condense repeated feed updates
  into calmer place/world context over days or weeks
- bounded refresh outputs that can be converted into prompt-time world and
  continuity context without rewriting core character
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

DEFAULT_WORLD_INTELLIGENCE_SUBSCRIPTIONS_KIND = "agent_world_intelligence_subscriptions_v1"
DEFAULT_WORLD_INTELLIGENCE_STATE_KIND = "agent_world_intelligence_state_v1"

_ALLOWED_WORLD_ACTIONS = frozenset(
    {"list", "subscribe", "discover", "deactivate", "refresh_now"}
)
_ALLOWED_WORLD_SCOPES = frozenset({"local", "regional", "national", "global", "topic"})
_ALLOWED_ENGAGEMENT_STATES = frozenset({"resonant", "warm", "uncertain", "cooling", "avoid"})
_ALLOWED_ONGOING_INTEREST_STATES = frozenset({"active", "growing", "peripheral"})
_ALLOWED_CO_ATTENTION_STATES = frozenset({"latent", "forming", "shared_thread"})


def _clean_text(value: object | None) -> str:
    """Normalize one free-form text value into a trimmed single line."""

    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _optional_text(value: object | None) -> str | None:
    """Return one normalized string or ``None`` for blank input."""

    normalized = _clean_text(value)
    return normalized or None


def _normalize_string_tuple(value: object | None, *, field_name: str) -> tuple[str, ...]:
    """Normalize a sequence of strings while rejecting blank items."""

    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a sequence of strings.")
    items: list[str] = []
    for index, item in enumerate(value):
        normalized = _clean_text(item)
        if not normalized:
            raise ValueError(f"{field_name}[{index}] cannot be blank.")
        items.append(normalized)
    return tuple(items)


def _normalize_float(value: object | None, *, field_name: str, default: float) -> float:
    """Normalize one bounded 0..1 float field."""

    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _normalize_int(
    value: object | None,
    *,
    field_name: str,
    default: int,
    minimum: int,
) -> int:
    """Normalize one positive integer field."""

    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc
    return max(minimum, parsed)


def _legacy_default_engagement_score(payload: Mapping[str, object]) -> float:
    """Derive one bounded engagement default for older stored interest signals.

    Older snapshots may not yet carry explicit engagement fields. In that case
    we backfill them from the already persisted salience/confidence/evidence
    structure so the policy can become useful immediately without requiring a
    manual reseed.
    """

    salience = _normalize_float(payload.get("salience"), field_name="salience", default=0.5)
    confidence = _normalize_float(payload.get("confidence"), field_name="confidence", default=0.5)
    evidence_count = _normalize_int(
        payload.get("evidence_count"),
        field_name="evidence_count",
        default=1,
        minimum=1,
    )
    explicit = bool(payload.get("explicit", False))
    base = 0.38 + (salience * 0.24) + (confidence * 0.22) + (min(evidence_count, 4) * 0.05)
    if explicit:
        base = max(base, 0.86)
    return _normalize_float(base, field_name="engagement_score", default=0.5)


def _mapping_items(value: object | None, *, field_name: str) -> tuple[Mapping[str, object], ...]:
    """Normalize one payload field into a tuple of mappings."""

    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a sequence of mappings.")
    items: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{field_name}[{index}] must be a mapping.")
        items.append(item)
    return tuple(items)


def _required_mapping_text(
    payload: Mapping[str, object],
    *,
    field_name: str,
    aliases: tuple[str, ...] = (),
) -> str:
    """Read one required normalized text field from a payload mapping."""

    for key in (field_name,) + aliases:
        normalized = _clean_text(payload.get(key))
        if normalized:
            return normalized
    raise ValueError(f"{field_name} is required.")


def _derive_engagement_state(
    *,
    engagement_score: float,
    engagement_count: int,
    positive_signal_count: int,
    exposure_count: int,
    non_reengagement_count: int,
    deflection_count: int,
    explicit: bool,
) -> str:
    """Classify one topic's current engagement state from bounded evidence.

    The states intentionally separate steady positive pull from evidence of
    cooling:

    - ``resonant``: the topic repeatedly draws the user back in
    - ``warm``: the topic is still a healthy ongoing interest
    - ``uncertain``: there is too little evidence either way
    - ``cooling``: repeated exposure is not converting into renewed uptake
    - ``avoid``: explicit or repeated stronger deflection says Twinr should back off
    """

    if deflection_count >= 2:
        return "avoid"
    if deflection_count >= 1 and explicit and positive_signal_count == 0:
        return "avoid"
    if deflection_count >= 1:
        return "cooling"
    if (
        exposure_count >= 2
        and non_reengagement_count >= 2
        and non_reengagement_count >= max(1, positive_signal_count)
    ):
        return "cooling"
    if explicit and exposure_count >= 2 and non_reengagement_count >= 1 and positive_signal_count == 0:
        return "cooling"
    if engagement_score >= 0.86 or engagement_count >= 4 or positive_signal_count >= 3:
        return "resonant"
    if engagement_score >= 0.62 or engagement_count >= 2 or positive_signal_count >= 1 or explicit:
        return "warm"
    return "uncertain"


def _derive_ongoing_interest_score(
    *,
    salience: float,
    engagement_score: float,
    engagement_state: str | None,
    engagement_count: int,
    positive_signal_count: int,
    non_reengagement_count: int,
    deflection_count: int,
    explicit: bool,
) -> float:
    """Summarize whether one topic keeps feeling worth following over time."""

    normalized_state = _clean_text(engagement_state).casefold()
    positive_pull = (
        (engagement_score * 0.7)
        + (salience * 0.18)
        + (min(max(positive_signal_count, 0), 4) * 0.04)
        + (min(max(engagement_count, 0), 5) * 0.03)
    )
    if explicit and positive_signal_count > 0 and deflection_count == 0:
        positive_pull = max(positive_pull, 0.76)
    penalty = (non_reengagement_count * 0.11) + (deflection_count * 0.18)
    if normalized_state == "cooling":
        penalty += 0.18
    elif normalized_state == "avoid":
        penalty += 0.38
    return _normalize_float(
        positive_pull - penalty,
        field_name="ongoing_interest_score",
        default=0.5,
    )


def _derive_ongoing_interest_state(
    *,
    ongoing_interest_score: float,
    engagement_state: str | None,
    engagement_count: int,
    positive_signal_count: int,
    non_reengagement_count: int,
    deflection_count: int,
) -> str:
    """Classify how alive one topic currently feels in Twinr's attention."""

    normalized_state = _clean_text(engagement_state).casefold()
    if normalized_state == "avoid" or deflection_count >= 1:
        return "peripheral"
    if normalized_state == "cooling" and (non_reengagement_count >= 1 or ongoing_interest_score < 0.62):
        return "peripheral"
    if (
        normalized_state == "uncertain"
        and positive_signal_count == 0
        and engagement_count <= 1
        and ongoing_interest_score < 0.68
    ):
        return "peripheral"
    if (
        ongoing_interest_score >= 0.84
        or (normalized_state == "resonant" and positive_signal_count >= 2)
        or engagement_count >= 5
    ):
        return "active"
    if (
        ongoing_interest_score >= 0.58
        or normalized_state in {"warm", "resonant"}
        or positive_signal_count >= 1
        or engagement_count >= 2
    ):
        return "growing"
    return "peripheral"


def _derive_co_attention_score(
    *,
    ongoing_interest_score: float,
    ongoing_interest: str,
    engagement_state: str | None,
    co_attention_count: int,
    non_reengagement_count: int,
    deflection_count: int,
) -> float:
    """Summarize whether a topic has become a shared ongoing thread."""

    normalized_state = _clean_text(engagement_state).casefold()
    base = (ongoing_interest_score * 0.66) + (min(max(co_attention_count, 0), 4) * 0.12)
    if ongoing_interest == "active":
        base += 0.12
    elif ongoing_interest == "growing":
        base += 0.05
    penalty = (non_reengagement_count * 0.05) + (deflection_count * 0.12)
    if normalized_state == "cooling":
        penalty += 0.18
    elif normalized_state == "avoid":
        penalty += 0.36
    return _normalize_float(
        base - penalty,
        field_name="co_attention_score",
        default=0.0,
    )


def _derive_co_attention_state(
    *,
    co_attention_score: float,
    co_attention_count: int,
    ongoing_interest: str,
    engagement_state: str | None,
) -> str:
    """Classify whether Twinr and the user now share an ongoing thread."""

    normalized_state = _clean_text(engagement_state).casefold()
    if normalized_state == "avoid":
        return "latent"
    if (
        co_attention_count >= 2
        and ongoing_interest == "active"
        and normalized_state not in {"cooling", "avoid"}
        and co_attention_score >= 0.78
    ):
        return "shared_thread"
    if co_attention_score >= 0.86 and ongoing_interest != "peripheral":
        return "shared_thread"
    if (
        co_attention_count >= 1
        and ongoing_interest in {"active", "growing"}
        and normalized_state not in {"cooling", "avoid"}
    ):
        return "forming"
    return "latent"


@dataclass(frozen=True, slots=True)
class WorldFeedItem:
    """Describe one fetched RSS or Atom item before prompt conversion."""

    feed_url: str
    source: str
    title: str
    link: str | None = None
    published_at: str | None = None

    def __post_init__(self) -> None:
        """Normalize one fetched feed item."""

        object.__setattr__(self, "feed_url", _required_mapping_text({"feed_url": self.feed_url}, field_name="feed_url"))
        object.__setattr__(self, "source", _required_mapping_text({"source": self.source}, field_name="source"))
        object.__setattr__(self, "title", _required_mapping_text({"title": self.title}, field_name="title"))
        object.__setattr__(self, "link", _optional_text(self.link))
        object.__setattr__(self, "published_at", _optional_text(self.published_at))

    def to_payload(self) -> dict[str, object]:
        """Serialize the feed item into a JSON-safe mapping."""

        payload: dict[str, object] = {
            "feed_url": self.feed_url,
            "source": self.source,
            "title": self.title,
        }
        if self.link is not None:
            payload["link"] = self.link
        if self.published_at is not None:
            payload["published_at"] = self.published_at
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "WorldFeedItem":
        """Build a fetched feed item from one payload mapping."""

        return cls(
            feed_url=_required_mapping_text(payload, field_name="feed_url"),
            source=_required_mapping_text(payload, field_name="source"),
            title=_required_mapping_text(payload, field_name="title"),
            link=payload.get("link"),
            published_at=payload.get("published_at"),
        )


@dataclass(frozen=True, slots=True)
class WorldFeedSubscription:
    """Persist one ongoing RSS/Atom subscription for world awareness."""

    subscription_id: str
    label: str
    feed_url: str
    scope: str = "topic"
    region: str | None = None
    topics: tuple[str, ...] = ()
    priority: float = 0.6
    base_priority: float | None = None
    active: bool = True
    refresh_interval_hours: int = 72
    base_refresh_interval_hours: int | None = None
    source_page_url: str | None = None
    source_title: str | None = None
    created_by: str = "installer"
    created_at: str | None = None
    updated_at: str | None = None
    last_checked_at: str | None = None
    last_refreshed_at: str | None = None
    last_error: str | None = None
    last_item_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize one persisted feed subscription."""

        object.__setattr__(
            self,
            "subscription_id",
            _required_mapping_text({"subscription_id": self.subscription_id}, field_name="subscription_id"),
        )
        object.__setattr__(self, "label", _required_mapping_text({"label": self.label}, field_name="label"))
        object.__setattr__(self, "feed_url", _required_mapping_text({"feed_url": self.feed_url}, field_name="feed_url"))
        normalized_scope = (_clean_text(self.scope).casefold() or "topic")
        if normalized_scope not in _ALLOWED_WORLD_SCOPES:
            raise ValueError(f"scope must be one of {sorted(_ALLOWED_WORLD_SCOPES)}.")
        object.__setattr__(self, "scope", normalized_scope)
        object.__setattr__(self, "region", _optional_text(self.region))
        object.__setattr__(self, "topics", _normalize_string_tuple(self.topics, field_name="topics"))
        normalized_priority = _normalize_float(self.priority, field_name="priority", default=0.6)
        object.__setattr__(self, "priority", normalized_priority)
        object.__setattr__(
            self,
            "base_priority",
            _normalize_float(
                normalized_priority if self.base_priority is None else self.base_priority,
                field_name="base_priority",
                default=normalized_priority,
            ),
        )
        object.__setattr__(self, "active", bool(self.active))
        normalized_refresh_interval_hours = _normalize_int(
            self.refresh_interval_hours,
            field_name="refresh_interval_hours",
            default=72,
            minimum=24,
        )
        object.__setattr__(
            self,
            "refresh_interval_hours",
            normalized_refresh_interval_hours,
        )
        object.__setattr__(
            self,
            "base_refresh_interval_hours",
            _normalize_int(
                normalized_refresh_interval_hours
                if self.base_refresh_interval_hours is None
                else self.base_refresh_interval_hours,
                field_name="base_refresh_interval_hours",
                default=normalized_refresh_interval_hours,
                minimum=24,
            ),
        )
        object.__setattr__(self, "source_page_url", _optional_text(self.source_page_url))
        object.__setattr__(self, "source_title", _optional_text(self.source_title))
        object.__setattr__(self, "created_by", _required_mapping_text({"created_by": self.created_by}, field_name="created_by"))
        object.__setattr__(self, "created_at", _optional_text(self.created_at))
        object.__setattr__(self, "updated_at", _optional_text(self.updated_at))
        object.__setattr__(self, "last_checked_at", _optional_text(self.last_checked_at))
        object.__setattr__(self, "last_refreshed_at", _optional_text(self.last_refreshed_at))
        object.__setattr__(self, "last_error", _optional_text(self.last_error))
        object.__setattr__(
            self,
            "last_item_ids",
            _normalize_string_tuple(self.last_item_ids, field_name="last_item_ids"),
        )

    def to_payload(self) -> dict[str, object]:
        """Serialize the subscription into a JSON-safe mapping."""

        payload: dict[str, object] = {
            "subscription_id": self.subscription_id,
            "label": self.label,
            "feed_url": self.feed_url,
            "scope": self.scope,
            "topics": list(self.topics),
            "priority": self.priority,
            "base_priority": self.base_priority,
            "active": self.active,
            "refresh_interval_hours": self.refresh_interval_hours,
            "base_refresh_interval_hours": self.base_refresh_interval_hours,
            "created_by": self.created_by,
            "last_item_ids": list(self.last_item_ids),
        }
        if self.region is not None:
            payload["region"] = self.region
        if self.source_page_url is not None:
            payload["source_page_url"] = self.source_page_url
        if self.source_title is not None:
            payload["source_title"] = self.source_title
        if self.created_at is not None:
            payload["created_at"] = self.created_at
        if self.updated_at is not None:
            payload["updated_at"] = self.updated_at
        if self.last_checked_at is not None:
            payload["last_checked_at"] = self.last_checked_at
        if self.last_refreshed_at is not None:
            payload["last_refreshed_at"] = self.last_refreshed_at
        if self.last_error is not None:
            payload["last_error"] = self.last_error
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "WorldFeedSubscription":
        """Build a subscription from one snapshot payload item."""

        return cls(
            subscription_id=_required_mapping_text(payload, field_name="subscription_id", aliases=("id",)),
            label=_required_mapping_text(payload, field_name="label", aliases=("title",)),
            feed_url=_required_mapping_text(payload, field_name="feed_url"),
            scope=payload.get("scope", "topic"),
            region=payload.get("region"),
            topics=payload.get("topics"),
            priority=payload.get("priority"),
            base_priority=payload.get("base_priority"),
            active=payload.get("active", True),
            refresh_interval_hours=payload.get("refresh_interval_hours"),
            base_refresh_interval_hours=payload.get("base_refresh_interval_hours"),
            source_page_url=payload.get("source_page_url"),
            source_title=payload.get("source_title"),
            created_by=_clean_text(payload.get("created_by")) or "installer",
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
            last_checked_at=payload.get("last_checked_at"),
            last_refreshed_at=payload.get("last_refreshed_at"),
            last_error=payload.get("last_error"),
            last_item_ids=payload.get("last_item_ids"),
        )


@dataclass(frozen=True, slots=True)
class WorldInterestSignal:
    """Capture one slowly learned topic/region interest for source calibration.

    ``engagement_score`` remains the compact bounded numeric summary used by
    ranking paths, while ``engagement_state`` captures the coarser Twinr policy
    stance:

    - ``resonant`` / ``warm`` mean the topic still actively pulls the user in
    - ``uncertain`` means Twinr should stay lightweight and observant
    - ``cooling`` means repeated exposure is not turning into renewed uptake
    - ``avoid`` means Twinr should actively back off

    Two slower derivative layers make the policy reusable across prompting and
    RSS calibration:

    - ``ongoing_interest`` says whether the topic currently feels active,
      growing, or merely peripheral in Twinr's continuing attention
    - ``co_attention`` says whether the topic is still latent, is becoming a
      shared thread, or has become a durable shared running topic between the
      user and Twinr's world-intelligence layer
    """

    signal_id: str
    topic: str
    summary: str
    region: str | None = None
    scope: str = "topic"
    salience: float = 0.5
    confidence: float = 0.5
    engagement_score: float = 0.5
    engagement_state: str | None = None
    ongoing_interest_score: float | None = None
    ongoing_interest: str | None = None
    co_attention_score: float | None = None
    co_attention_state: str | None = None
    co_attention_count: int = 0
    evidence_count: int = 1
    engagement_count: int = 1
    positive_signal_count: int = 1
    exposure_count: int = 1
    non_reengagement_count: int = 0
    deflection_count: int = 0
    explicit: bool = False
    source_event_ids: tuple[str, ...] = ()
    updated_at: str | None = None

    def __post_init__(self) -> None:
        """Normalize one world-interest signal."""

        object.__setattr__(self, "signal_id", _required_mapping_text({"signal_id": self.signal_id}, field_name="signal_id"))
        object.__setattr__(self, "topic", _required_mapping_text({"topic": self.topic}, field_name="topic"))
        object.__setattr__(self, "summary", _required_mapping_text({"summary": self.summary}, field_name="summary"))
        object.__setattr__(self, "region", _optional_text(self.region))
        normalized_scope = (_clean_text(self.scope).casefold() or "topic")
        if normalized_scope not in _ALLOWED_WORLD_SCOPES:
            raise ValueError(f"scope must be one of {sorted(_ALLOWED_WORLD_SCOPES)}.")
        object.__setattr__(self, "scope", normalized_scope)
        object.__setattr__(self, "salience", _normalize_float(self.salience, field_name="salience", default=0.5))
        object.__setattr__(self, "confidence", _normalize_float(self.confidence, field_name="confidence", default=0.5))
        object.__setattr__(
            self,
            "engagement_score",
            _normalize_float(self.engagement_score, field_name="engagement_score", default=0.5),
        )
        object.__setattr__(
            self,
            "evidence_count",
            _normalize_int(self.evidence_count, field_name="evidence_count", default=1, minimum=1),
        )
        object.__setattr__(
            self,
            "engagement_count",
            _normalize_int(self.engagement_count, field_name="engagement_count", default=1, minimum=0),
        )
        object.__setattr__(
            self,
            "positive_signal_count",
            _normalize_int(self.positive_signal_count, field_name="positive_signal_count", default=1, minimum=0),
        )
        object.__setattr__(
            self,
            "exposure_count",
            _normalize_int(self.exposure_count, field_name="exposure_count", default=1, minimum=0),
        )
        object.__setattr__(
            self,
            "non_reengagement_count",
            _normalize_int(
                self.non_reengagement_count,
                field_name="non_reengagement_count",
                default=0,
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "deflection_count",
            _normalize_int(self.deflection_count, field_name="deflection_count", default=0, minimum=0),
        )
        object.__setattr__(self, "explicit", bool(self.explicit))
        normalized_state = _optional_text(self.engagement_state)
        if normalized_state is not None:
            normalized_state = normalized_state.casefold()
            if normalized_state not in _ALLOWED_ENGAGEMENT_STATES:
                raise ValueError(f"engagement_state must be one of {sorted(_ALLOWED_ENGAGEMENT_STATES)}.")
        else:
            normalized_state = _derive_engagement_state(
                engagement_score=self.engagement_score,
                engagement_count=self.engagement_count,
                positive_signal_count=self.positive_signal_count,
                exposure_count=self.exposure_count,
                non_reengagement_count=self.non_reengagement_count,
                deflection_count=self.deflection_count,
                explicit=self.explicit,
            )
        object.__setattr__(self, "engagement_state", normalized_state)
        derived_ongoing_interest_score = _derive_ongoing_interest_score(
            salience=self.salience,
            engagement_score=self.engagement_score,
            engagement_state=normalized_state,
            engagement_count=self.engagement_count,
            positive_signal_count=self.positive_signal_count,
            non_reengagement_count=self.non_reengagement_count,
            deflection_count=self.deflection_count,
            explicit=self.explicit,
        )
        object.__setattr__(
            self,
            "ongoing_interest_score",
            _normalize_float(
                derived_ongoing_interest_score if self.ongoing_interest_score is None else self.ongoing_interest_score,
                field_name="ongoing_interest_score",
                default=derived_ongoing_interest_score,
            ),
        )
        normalized_ongoing_interest = _optional_text(self.ongoing_interest)
        if normalized_ongoing_interest is not None:
            normalized_ongoing_interest = normalized_ongoing_interest.casefold()
            if normalized_ongoing_interest not in _ALLOWED_ONGOING_INTEREST_STATES:
                raise ValueError(
                    f"ongoing_interest must be one of {sorted(_ALLOWED_ONGOING_INTEREST_STATES)}."
                )
        else:
            normalized_ongoing_interest = _derive_ongoing_interest_state(
                ongoing_interest_score=self.ongoing_interest_score,
                engagement_state=normalized_state,
                engagement_count=self.engagement_count,
                positive_signal_count=self.positive_signal_count,
                non_reengagement_count=self.non_reengagement_count,
                deflection_count=self.deflection_count,
            )
        object.__setattr__(self, "ongoing_interest", normalized_ongoing_interest)
        object.__setattr__(
            self,
            "co_attention_count",
            _normalize_int(
                self.co_attention_count,
                field_name="co_attention_count",
                default=0,
                minimum=0,
            ),
        )
        derived_co_attention_score = _derive_co_attention_score(
            ongoing_interest_score=self.ongoing_interest_score,
            ongoing_interest=self.ongoing_interest,
            engagement_state=normalized_state,
            co_attention_count=self.co_attention_count,
            non_reengagement_count=self.non_reengagement_count,
            deflection_count=self.deflection_count,
        )
        object.__setattr__(
            self,
            "co_attention_score",
            _normalize_float(
                derived_co_attention_score if self.co_attention_score is None else self.co_attention_score,
                field_name="co_attention_score",
                default=derived_co_attention_score,
            ),
        )
        normalized_co_attention_state = _optional_text(self.co_attention_state)
        if normalized_co_attention_state is not None:
            normalized_co_attention_state = normalized_co_attention_state.casefold()
            if normalized_co_attention_state not in _ALLOWED_CO_ATTENTION_STATES:
                raise ValueError(
                    f"co_attention_state must be one of {sorted(_ALLOWED_CO_ATTENTION_STATES)}."
                )
        else:
            normalized_co_attention_state = _derive_co_attention_state(
                co_attention_score=self.co_attention_score,
                co_attention_count=self.co_attention_count,
                ongoing_interest=self.ongoing_interest,
                engagement_state=normalized_state,
            )
        object.__setattr__(self, "co_attention_state", normalized_co_attention_state)
        object.__setattr__(
            self,
            "source_event_ids",
            _normalize_string_tuple(self.source_event_ids, field_name="source_event_ids"),
        )
        object.__setattr__(self, "updated_at", _optional_text(self.updated_at))

    def to_payload(self) -> dict[str, object]:
        """Serialize one calibration signal into a JSON-safe mapping."""

        payload: dict[str, object] = {
            "signal_id": self.signal_id,
            "topic": self.topic,
            "summary": self.summary,
            "scope": self.scope,
            "salience": self.salience,
            "confidence": self.confidence,
            "engagement_score": self.engagement_score,
            "engagement_state": self.engagement_state,
            "ongoing_interest_score": self.ongoing_interest_score,
            "ongoing_interest": self.ongoing_interest,
            "co_attention_score": self.co_attention_score,
            "co_attention_state": self.co_attention_state,
            "co_attention_count": self.co_attention_count,
            "evidence_count": self.evidence_count,
            "engagement_count": self.engagement_count,
            "positive_signal_count": self.positive_signal_count,
            "exposure_count": self.exposure_count,
            "non_reengagement_count": self.non_reengagement_count,
            "deflection_count": self.deflection_count,
            "explicit": self.explicit,
            "source_event_ids": list(self.source_event_ids),
        }
        if self.region is not None:
            payload["region"] = self.region
        if self.updated_at is not None:
            payload["updated_at"] = self.updated_at
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "WorldInterestSignal":
        """Build one calibration signal from stored payload data."""

        legacy_engagement_count = (
            payload.get("engagement_count")
            if "engagement_count" in payload
            else payload.get("evidence_count")
        )
        positive_signal_count = (
            payload.get("positive_signal_count")
            if "positive_signal_count" in payload
            else legacy_engagement_count
        )
        engagement_score = (
            payload.get("engagement_score")
            if "engagement_score" in payload
            else _legacy_default_engagement_score(payload)
        )
        engagement_count = (
            payload.get("engagement_count")
            if "engagement_count" in payload
            else legacy_engagement_count
        )
        exposure_count = payload.get("exposure_count")
        if exposure_count is None:
            fallback_positive_count = _normalize_int(
                positive_signal_count,
                field_name="positive_signal_count",
                default=1,
                minimum=0,
            )
            exposure_count = max(1, fallback_positive_count)
        return cls(
            signal_id=_required_mapping_text(payload, field_name="signal_id", aliases=("id",)),
            topic=_required_mapping_text(payload, field_name="topic"),
            summary=_required_mapping_text(payload, field_name="summary", aliases=("description",)),
            region=payload.get("region"),
            scope=payload.get("scope", "topic"),
            salience=payload.get("salience"),
            confidence=payload.get("confidence"),
            engagement_score=engagement_score,
            engagement_state=payload.get("engagement_state"),
            ongoing_interest_score=payload.get("ongoing_interest_score"),
            ongoing_interest=payload.get("ongoing_interest"),
            co_attention_score=payload.get("co_attention_score"),
            co_attention_state=payload.get("co_attention_state"),
            co_attention_count=payload.get("co_attention_count", 0),
            evidence_count=payload.get("evidence_count"),
            engagement_count=engagement_count,
            positive_signal_count=positive_signal_count,
            exposure_count=exposure_count,
            non_reengagement_count=payload.get("non_reengagement_count", 0),
            deflection_count=payload.get("deflection_count", 0),
            explicit=payload.get("explicit", False),
            source_event_ids=payload.get("source_event_ids"),
            updated_at=payload.get("updated_at"),
        )


@dataclass(frozen=True, slots=True)
class SituationalAwarenessThread:
    """Persist one condensed place/world thread built from repeated feed updates."""

    thread_id: str
    title: str
    summary: str
    topic: str
    region: str | None = None
    scope: str = "topic"
    salience: float = 0.5
    update_count: int = 1
    recent_titles: tuple[str, ...] = ()
    source_labels: tuple[str, ...] = ()
    supporting_item_ids: tuple[str, ...] = ()
    updated_at: str | None = None
    review_at: str | None = None

    def __post_init__(self) -> None:
        """Normalize one awareness thread payload."""

        object.__setattr__(self, "thread_id", _required_mapping_text({"thread_id": self.thread_id}, field_name="thread_id"))
        object.__setattr__(self, "title", _required_mapping_text({"title": self.title}, field_name="title"))
        object.__setattr__(self, "summary", _required_mapping_text({"summary": self.summary}, field_name="summary"))
        object.__setattr__(self, "topic", _required_mapping_text({"topic": self.topic}, field_name="topic"))
        object.__setattr__(self, "region", _optional_text(self.region))
        normalized_scope = (_clean_text(self.scope).casefold() or "topic")
        if normalized_scope not in _ALLOWED_WORLD_SCOPES:
            raise ValueError(f"scope must be one of {sorted(_ALLOWED_WORLD_SCOPES)}.")
        object.__setattr__(self, "scope", normalized_scope)
        object.__setattr__(self, "salience", _normalize_float(self.salience, field_name="salience", default=0.5))
        object.__setattr__(
            self,
            "update_count",
            _normalize_int(self.update_count, field_name="update_count", default=1, minimum=1),
        )
        object.__setattr__(self, "recent_titles", _normalize_string_tuple(self.recent_titles, field_name="recent_titles"))
        object.__setattr__(self, "source_labels", _normalize_string_tuple(self.source_labels, field_name="source_labels"))
        object.__setattr__(
            self,
            "supporting_item_ids",
            _normalize_string_tuple(self.supporting_item_ids, field_name="supporting_item_ids"),
        )
        object.__setattr__(self, "updated_at", _optional_text(self.updated_at))
        object.__setattr__(self, "review_at", _optional_text(self.review_at))

    def to_payload(self) -> dict[str, object]:
        """Serialize one awareness thread into a JSON-safe mapping."""

        payload: dict[str, object] = {
            "thread_id": self.thread_id,
            "title": self.title,
            "summary": self.summary,
            "topic": self.topic,
            "scope": self.scope,
            "salience": self.salience,
            "update_count": self.update_count,
            "recent_titles": list(self.recent_titles),
            "source_labels": list(self.source_labels),
            "supporting_item_ids": list(self.supporting_item_ids),
        }
        if self.region is not None:
            payload["region"] = self.region
        if self.updated_at is not None:
            payload["updated_at"] = self.updated_at
        if self.review_at is not None:
            payload["review_at"] = self.review_at
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "SituationalAwarenessThread":
        """Build one awareness thread from stored payload data."""

        return cls(
            thread_id=_required_mapping_text(payload, field_name="thread_id", aliases=("id",)),
            title=_required_mapping_text(payload, field_name="title", aliases=("name",)),
            summary=_required_mapping_text(payload, field_name="summary", aliases=("description",)),
            topic=_required_mapping_text(payload, field_name="topic"),
            region=payload.get("region"),
            scope=payload.get("scope", "topic"),
            salience=payload.get("salience"),
            update_count=payload.get("update_count"),
            recent_titles=payload.get("recent_titles"),
            source_labels=payload.get("source_labels"),
            supporting_item_ids=payload.get("supporting_item_ids"),
            updated_at=payload.get("updated_at"),
            review_at=payload.get("review_at"),
        )


@dataclass(frozen=True, slots=True)
class WorldIntelligenceState:
    """Persist global timing, calibration, and awareness state."""

    schema_version: int = 1
    last_discovered_at: str | None = None
    last_refreshed_at: str | None = None
    last_recalibrated_at: str | None = None
    discovery_interval_hours: int = 336
    recalibration_interval_hours: int = 336
    last_discovery_query: str | None = None
    interest_signals: tuple[WorldInterestSignal, ...] = ()
    awareness_threads: tuple[SituationalAwarenessThread, ...] = ()

    def __post_init__(self) -> None:
        """Normalize the global state snapshot."""

        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "last_discovered_at", _optional_text(self.last_discovered_at))
        object.__setattr__(self, "last_refreshed_at", _optional_text(self.last_refreshed_at))
        object.__setattr__(self, "last_recalibrated_at", _optional_text(self.last_recalibrated_at))
        object.__setattr__(
            self,
            "discovery_interval_hours",
            _normalize_int(
                self.discovery_interval_hours,
                field_name="discovery_interval_hours",
                default=336,
                minimum=168,
            ),
        )
        object.__setattr__(
            self,
            "recalibration_interval_hours",
            _normalize_int(
                self.recalibration_interval_hours,
                field_name="recalibration_interval_hours",
                default=336,
                minimum=168,
            ),
        )
        object.__setattr__(self, "last_discovery_query", _optional_text(self.last_discovery_query))
        object.__setattr__(self, "interest_signals", tuple(self.interest_signals))
        object.__setattr__(self, "awareness_threads", tuple(self.awareness_threads))

    def to_payload(self) -> dict[str, object]:
        """Serialize the state snapshot into a JSON-safe mapping."""

        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "discovery_interval_hours": self.discovery_interval_hours,
            "recalibration_interval_hours": self.recalibration_interval_hours,
            "interest_signals": [item.to_payload() for item in self.interest_signals],
            "awareness_threads": [item.to_payload() for item in self.awareness_threads],
        }
        if self.last_discovered_at is not None:
            payload["last_discovered_at"] = self.last_discovered_at
        if self.last_refreshed_at is not None:
            payload["last_refreshed_at"] = self.last_refreshed_at
        if self.last_recalibrated_at is not None:
            payload["last_recalibrated_at"] = self.last_recalibrated_at
        if self.last_discovery_query is not None:
            payload["last_discovery_query"] = self.last_discovery_query
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "WorldIntelligenceState":
        """Build a global state snapshot from one payload mapping."""

        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            last_discovered_at=payload.get("last_discovered_at"),
            last_refreshed_at=payload.get("last_refreshed_at"),
            last_recalibrated_at=payload.get("last_recalibrated_at"),
            discovery_interval_hours=payload.get("discovery_interval_hours"),
            recalibration_interval_hours=payload.get("recalibration_interval_hours"),
            last_discovery_query=payload.get("last_discovery_query"),
            interest_signals=tuple(
                WorldInterestSignal.from_payload(item)
                for item in _mapping_items(payload.get("interest_signals"), field_name="interest_signals")
            ),
            awareness_threads=tuple(
                SituationalAwarenessThread.from_payload(item)
                for item in _mapping_items(payload.get("awareness_threads"), field_name="awareness_threads")
            ),
        )


@dataclass(frozen=True, slots=True)
class WorldIntelligenceConfigRequest:
    """Describe one installer/tool request that changes world intelligence."""

    action: str
    query: str | None = None
    label: str | None = None
    location_hint: str | None = None
    region: str | None = None
    topics: tuple[str, ...] = ()
    feed_urls: tuple[str, ...] = ()
    subscription_refs: tuple[str, ...] = ()
    scope: str = "topic"
    priority: float = 0.6
    refresh_interval_hours: int = 72
    auto_subscribe: bool = True
    refresh_after_change: bool = False
    created_by: str = "tool"

    def __post_init__(self) -> None:
        """Normalize one config request before execution."""

        normalized_action = _clean_text(self.action).casefold()
        if normalized_action not in _ALLOWED_WORLD_ACTIONS:
            raise ValueError(f"action must be one of {sorted(_ALLOWED_WORLD_ACTIONS)}.")
        object.__setattr__(self, "action", normalized_action)
        object.__setattr__(self, "query", _optional_text(self.query))
        object.__setattr__(self, "label", _optional_text(self.label))
        object.__setattr__(self, "location_hint", _optional_text(self.location_hint))
        object.__setattr__(self, "region", _optional_text(self.region))
        object.__setattr__(self, "topics", _normalize_string_tuple(self.topics, field_name="topics"))
        object.__setattr__(self, "feed_urls", _normalize_string_tuple(self.feed_urls, field_name="feed_urls"))
        object.__setattr__(
            self,
            "subscription_refs",
            _normalize_string_tuple(self.subscription_refs, field_name="subscription_refs"),
        )
        normalized_scope = (_clean_text(self.scope).casefold() or "topic")
        if normalized_scope not in _ALLOWED_WORLD_SCOPES:
            raise ValueError(f"scope must be one of {sorted(_ALLOWED_WORLD_SCOPES)}.")
        object.__setattr__(self, "scope", normalized_scope)
        object.__setattr__(self, "priority", _normalize_float(self.priority, field_name="priority", default=0.6))
        object.__setattr__(
            self,
            "refresh_interval_hours",
            _normalize_int(
                self.refresh_interval_hours,
                field_name="refresh_interval_hours",
                default=72,
                minimum=24,
            ),
        )
        object.__setattr__(self, "auto_subscribe", bool(self.auto_subscribe))
        object.__setattr__(self, "refresh_after_change", bool(self.refresh_after_change))
        object.__setattr__(self, "created_by", _required_mapping_text({"created_by": self.created_by}, field_name="created_by"))


@dataclass(frozen=True, slots=True)
class WorldIntelligenceRefreshResult:
    """Return the bounded outcome of one feed refresh pass."""

    status: str
    refreshed: bool
    subscriptions: tuple[WorldFeedSubscription, ...] = ()
    world_signals: tuple[object, ...] = ()
    continuity_threads: tuple[object, ...] = ()
    awareness_threads: tuple[SituationalAwarenessThread, ...] = ()
    refreshed_subscription_ids: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    checked_at: str | None = None

    def __post_init__(self) -> None:
        """Normalize one refresh result."""

        object.__setattr__(self, "status", _required_mapping_text({"status": self.status}, field_name="status"))
        object.__setattr__(self, "refreshed", bool(self.refreshed))
        object.__setattr__(self, "subscriptions", tuple(self.subscriptions))
        object.__setattr__(self, "world_signals", tuple(self.world_signals))
        object.__setattr__(self, "continuity_threads", tuple(self.continuity_threads))
        object.__setattr__(self, "awareness_threads", tuple(self.awareness_threads))
        object.__setattr__(
            self,
            "refreshed_subscription_ids",
            _normalize_string_tuple(self.refreshed_subscription_ids, field_name="refreshed_subscription_ids"),
        )
        object.__setattr__(self, "errors", _normalize_string_tuple(self.errors, field_name="errors"))
        object.__setattr__(self, "checked_at", _optional_text(self.checked_at))


@dataclass(frozen=True, slots=True)
class WorldIntelligenceConfigResult:
    """Return the outcome of one subscription/discovery mutation request."""

    status: str
    action: str
    subscriptions: tuple[WorldFeedSubscription, ...] = ()
    discovered_feed_urls: tuple[str, ...] = ()
    message: str | None = None
    refresh: WorldIntelligenceRefreshResult | None = None

    def __post_init__(self) -> None:
        """Normalize one config result."""

        object.__setattr__(self, "status", _required_mapping_text({"status": self.status}, field_name="status"))
        object.__setattr__(self, "action", _required_mapping_text({"action": self.action}, field_name="action"))
        object.__setattr__(self, "subscriptions", tuple(self.subscriptions))
        object.__setattr__(
            self,
            "discovered_feed_urls",
            _normalize_string_tuple(self.discovered_feed_urls, field_name="discovered_feed_urls"),
        )
        object.__setattr__(self, "message", _optional_text(self.message))
        if self.refresh is not None and not isinstance(self.refresh, WorldIntelligenceRefreshResult):
            raise ValueError("refresh must be a WorldIntelligenceRefreshResult.")
