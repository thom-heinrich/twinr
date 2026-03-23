"""Convert durable reflection outputs into right-lane companion candidates.

The ambient reserve lane should not only echo prompt-time mindshare or urgent
memory conflicts. Reflection produces slower, often richer signals about
ongoing threads, preferences, routines, and recent environment summaries. This
module turns those durable reflection artifacts into bounded display
conversation openers without coupling the display runtime to reflection
internals.

Candidate derivation stays generic:

- reflection summaries come from persisted long-term summary objects
- midterm packets come from persisted reflection/restart-recall packets
- filtering relies on structured kinds/attributes, not named topics
- output is still just a candidate pool; ranking and scheduling happen
  elsewhere
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.memory.longterm.core.models import LongTermMemoryObjectV1, LongTermMidtermPacketV1
from twinr.memory.longterm.runtime.service import LongTermMemoryService

_DEFAULT_CANDIDATE_LIMIT = 3
_DEFAULT_MAX_AGE_DAYS = 14.0
_ALLOWED_SENSITIVITY = frozenset({"low", "normal"})
_CONTINUITY_PACKET_KINDS = frozenset({"recent_turn_continuity", "conversation_context"})
_DEVICE_PACKET_KINDS = frozenset({"device_context", "device_interaction"})
_SUPPRESSED_PACKET_KINDS = frozenset({"interaction_quality"})
_META_PACKET_KINDS = frozenset({"conversation_state", "policy_context"})


def _utc_now() -> datetime:
    """Return the current UTC wall clock."""

    return datetime.now(timezone.utc)


def _compact_text(value: object | None, *, max_len: int) -> str:
    """Collapse one value into bounded single-line text."""

    if value is None:
        return ""
    compact = " ".join(str(value).split()).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _mapping(value: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return one mapping or an empty mapping."""

    return value if isinstance(value, Mapping) else {}


def _coerce_days(value: object, *, default: float, minimum: float, maximum: float) -> float:
    """Coerce one config-like day value into a bounded float."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return max(minimum, min(maximum, number))


def _normalize_label(value: object | None, *, title_case: bool = False, max_len: int = 72) -> str:
    """Normalize one generic label for visible topic anchors."""

    text = _compact_text(value, max_len=max_len).replace("_", " ").replace(":", " ")
    if title_case:
        return text.title()
    return text


def _sentence_label(value: object | None, *, max_len: int = 72) -> str:
    """Normalize one label while only capitalizing the first visible letter."""

    text = _normalize_label(value, title_case=False, max_len=max_len)
    if not text:
        return ""
    return text[:1].upper() + text[1:]


def _primary_hint_phrase(query_hints: Sequence[object] | None, *, max_len: int = 48) -> str:
    """Return one compact semantic phrase from bounded query hints."""

    if not query_hints:
        return ""
    for raw_hint in query_hints:
        normalized = _normalize_label(raw_hint, title_case=False, max_len=max_len)
        if " " in normalized:
            return normalized
    parts: list[str] = []
    for raw_hint in query_hints[:3]:
        normalized = _normalize_label(raw_hint, title_case=False, max_len=24)
        if not normalized:
            continue
        candidate = " ".join((*parts, normalized))
        if len(candidate) > max_len and parts:
            break
        parts.append(normalized)
    return " ".join(parts)


def _topic_key(*values: object | None) -> str:
    """Return one stable normalized topic key."""

    for value in values:
        compact = _compact_text(value, max_len=96).casefold()
        if compact:
            return compact
    return ""


def _recency_score(updated_at: datetime, *, now: datetime, max_age_days: float) -> float:
    """Return one bounded 0..1 recency score within the configured window."""

    age_days = max(0.0, (now - updated_at.astimezone(timezone.utc)).total_seconds() / 86_400.0)
    return max(0.0, 1.0 - (age_days / max(1.0, max_age_days)))


@dataclass(frozen=True, slots=True)
class DisplayReserveReflectionConfig:
    """Store bounded configuration for reflection-derived reserve candidates."""

    candidate_limit: int = _DEFAULT_CANDIDATE_LIMIT
    max_age_days: float = _DEFAULT_MAX_AGE_DAYS

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayReserveReflectionConfig":
        """Resolve reflection-candidate limits from the runtime configuration."""

        return cls(
            candidate_limit=max(
                1,
                int(
                    getattr(
                        config,
                        "display_reserve_bus_reflection_candidate_limit",
                        _DEFAULT_CANDIDATE_LIMIT,
                    )
                ),
            ),
            max_age_days=_coerce_days(
                getattr(config, "display_reserve_bus_reflection_max_age_days", _DEFAULT_MAX_AGE_DAYS),
                default=_DEFAULT_MAX_AGE_DAYS,
                minimum=1.0,
                maximum=90.0,
            ),
        )


def _summary_title(item: LongTermMemoryObjectV1) -> str:
    """Return one stable topic title for a reflection summary object."""

    attributes = _mapping(item.attributes)
    for value in (
        attributes.get("person_name"),
        attributes.get("environment_id"),
        item.summary,
        attributes.get("slot_key"),
        item.slot_key,
        item.value_key,
    ):
        label = _normalize_label(value, title_case=True)
        if label:
            return label
    return _compact_text(item.summary, max_len=72)


def _summary_hook_hint(item: LongTermMemoryObjectV1) -> str:
    """Return one compact user-facing hook for a reflection summary."""

    attributes = _mapping(item.attributes)
    for value in (
        attributes.get("display_anchor"),
        item.summary,
        item.details,
    ):
        text = _compact_text(value, max_len=160)
        if text:
            return text
    return ""


def _summary_display_copy(item: LongTermMemoryObjectV1, *, title: str) -> tuple[str, str]:
    """Return one deterministic display-safe fallback copy for reflection summaries."""

    attributes = _mapping(item.attributes)
    summary_type = _compact_text(attributes.get("summary_type"), max_len=32).casefold()
    memory_domain = _compact_text(attributes.get("memory_domain"), max_len=48).casefold()
    anchor = _sentence_label(title, max_len=72) or "das"
    if summary_type == "thread" or _compact_text(attributes.get("person_name"), max_len=80):
        return (
            _compact_text(f"Wollen wir bei {anchor} kurz anknuepfen?", max_len=112),
            _compact_text("Da haengt zwischen uns noch ein kleiner Faden.", max_len=112),
        )
    if memory_domain == "smart_home_environment" or summary_type == "environment_reflection":
        return (
            _compact_text(f"Rund um {anchor} schaue ich heute noch einmal hin.", max_len=112),
            _compact_text("Ein kurzer Blick darauf reicht mir erstmal.", max_len=112),
        )
    return (
        _compact_text(f"Zu {anchor} wuerde ich gern kurz anknuepfen.", max_len=112),
        _compact_text("Ein kleiner Nachtrag dazu waere schon gut.", max_len=112),
    )


def _summary_action(item: LongTermMemoryObjectV1) -> tuple[str, str]:
    """Return one bounded action/attention pair for a summary object."""

    attributes = _mapping(item.attributes)
    summary_type = _compact_text(attributes.get("summary_type"), max_len=32).casefold()
    memory_domain = _compact_text(attributes.get("memory_domain"), max_len=48).casefold()
    if summary_type == "thread" or _compact_text(attributes.get("person_name"), max_len=80):
        return ("ask_one", "shared_thread")
    if memory_domain == "smart_home_environment" or summary_type == "environment_reflection":
        return ("brief_update", "growing")
    return ("hint", "forming")


def _summary_candidate(item: LongTermMemoryObjectV1, *, now: datetime, max_age_days: float) -> AmbientDisplayImpulseCandidate | None:
    """Convert one reflection summary object into a reserve candidate."""

    if item.kind != "summary" or item.status != "active":
        return None
    if _compact_text(item.sensitivity, max_len=24).casefold() not in _ALLOWED_SENSITIVITY:
        return None
    if item.updated_at.astimezone(timezone.utc) < now - timedelta(days=max_age_days):
        return None
    title = _summary_title(item)
    topic_key = _topic_key(item.slot_key, item.value_key, item.memory_id, title)
    if not topic_key:
        return None
    action, attention_state = _summary_action(item)
    headline, body = _summary_display_copy(item, title=title)
    recency = _recency_score(item.updated_at, now=now, max_age_days=max_age_days)
    salience = min(0.98, max(0.42, (float(item.confidence) * 0.62) + (recency * 0.24)))
    details = _compact_text(item.details, max_len=140)
    return AmbientDisplayImpulseCandidate(
        topic_key=topic_key,
        title=title,
        source="reflection_summary",
        action=action,
        attention_state=attention_state,
        salience=salience,
        eyebrow="",
        headline=headline,
        body=body,
        symbol="sparkles",
        accent="warm" if attention_state == "shared_thread" else "info",
        reason="reflection_summary",
        candidate_family="reflection",
        generation_context={
            "candidate_family": "reflection_summary",
            "display_anchor": title,
            "hook_hint": _summary_hook_hint(item),
            "reflection_kind": _compact_text(_mapping(item.attributes).get("summary_type"), max_len=40) or "summary",
            "topic_title": title,
            "summary": _compact_text(item.summary, max_len=180),
            "details": details,
            "memory_domain": _compact_text(_mapping(item.attributes).get("memory_domain"), max_len=48),
        },
    )


def _packet_title(packet: LongTermMidtermPacketV1) -> str:
    """Return one stable topic title for a midterm packet."""

    attributes = _mapping(packet.attributes)
    kind = _compact_text(packet.kind, max_len=40).casefold()
    if kind in _CONTINUITY_PACKET_KINDS:
        continuity_anchor = (
            _sentence_label(attributes.get("display_anchor"), max_len=72)
            or _sentence_label(attributes.get("transcript_excerpt"), max_len=72)
            or _sentence_label(_primary_hint_phrase(packet.query_hints, max_len=48), max_len=72)
        )
        if continuity_anchor:
            return continuity_anchor
    for value in (
        attributes.get("person_name"),
        attributes.get("environment_id"),
        _primary_hint_phrase(packet.query_hints, max_len=48),
        packet.summary,
    ):
        label = _sentence_label(value, max_len=72)
        if label:
            return label
    return _compact_text(packet.summary, max_len=72)


def _packet_action(packet: LongTermMidtermPacketV1) -> tuple[str, str]:
    """Return one bounded action/attention pair for a midterm packet."""

    attributes = _mapping(packet.attributes)
    packet_scope = _compact_text(attributes.get("packet_scope"), max_len=48).casefold()
    persistence_scope = _compact_text(attributes.get("persistence_scope"), max_len=48).casefold()
    kind = _compact_text(packet.kind, max_len=40).casefold()
    if (
        persistence_scope == "restart_recall"
        or kind in _SUPPRESSED_PACKET_KINDS
        or kind in _META_PACKET_KINDS
        or kind in _DEVICE_PACKET_KINDS
    ):
        return ("silent", "background")
    if kind in _CONTINUITY_PACKET_KINDS:
        return ("ask_one", "shared_thread")
    if kind == "preference":
        return ("ask_one", "forming")
    if packet_scope == "recent_environment_reflection":
        return ("brief_update", "growing")
    if kind == "interaction":
        return ("hint", "forming")
    return ("brief_update", "forming")


def _continuity_packet_has_displayable_anchor(packet: LongTermMidtermPacketV1) -> bool:
    """Return whether one continuity packet has a display-worthy semantic anchor.

    Continuity packets are useful only when they preserve a concrete thread the
    user could still recognize later. Token-only turn residue like farewells or
    generic closure cues should stay in memory, but they should not become
    visible reserve-lane conversation openers.
    """

    attributes = _mapping(packet.attributes)
    if _compact_text(attributes.get("display_anchor"), max_len=72):
        return True
    if _compact_text(attributes.get("person_name"), max_len=72):
        return True
    if _compact_text(attributes.get("environment_id"), max_len=72):
        return True
    for raw_hint in packet.query_hints:
        normalized = _normalize_label(raw_hint, title_case=False, max_len=72)
        if normalized and " " in normalized:
            return True
    return False


def _packet_candidate_family(packet: LongTermMidtermPacketV1) -> str:
    """Return one generic family label for reflection-derived packets."""

    attributes = _mapping(packet.attributes)
    packet_scope = _compact_text(attributes.get("packet_scope"), max_len=48).casefold()
    kind = _compact_text(packet.kind, max_len=40).casefold()
    if kind in _CONTINUITY_PACKET_KINDS:
        return "reflection_thread"
    if kind == "preference":
        return "reflection_preference"
    if packet_scope == "recent_environment_reflection" or kind in _DEVICE_PACKET_KINDS:
        return "reflection_context"
    return "reflection"


def _packet_display_anchor(packet: LongTermMidtermPacketV1, *, title: str) -> str:
    """Return one compact user-facing anchor for a reflection packet."""

    attributes = _mapping(packet.attributes)
    kind = _compact_text(packet.kind, max_len=40).casefold()
    if kind in _CONTINUITY_PACKET_KINDS:
        return (
            _sentence_label(attributes.get("display_anchor"), max_len=72)
            or _sentence_label(attributes.get("transcript_excerpt"), max_len=72)
            or _sentence_label(_primary_hint_phrase(packet.query_hints, max_len=48), max_len=72)
            or _sentence_label(title, max_len=72)
        )
    return (
        _sentence_label(title, max_len=72)
        or _sentence_label(_primary_hint_phrase(packet.query_hints, max_len=48), max_len=72)
    )


def _packet_hook_hint(packet: LongTermMidtermPacketV1) -> str:
    """Return one compact user-facing conversational hook for a reflection packet."""

    attributes = _mapping(packet.attributes)
    for value in (
        attributes.get("response_excerpt"),
        attributes.get("transcript_excerpt"),
        _primary_hint_phrase(packet.query_hints, max_len=96),
        packet.summary,
    ):
        text = _compact_text(value, max_len=160)
        if text:
            return text
    return ""


def _packet_display_copy(
    packet: LongTermMidtermPacketV1,
    *,
    title: str,
    action: str,
    attention_state: str,
    candidate_family: str,
) -> tuple[str, str]:
    """Return one deterministic display-safe fallback copy for reflection packets."""

    anchor = _packet_display_anchor(packet, title=title) or "das"
    if candidate_family == "reflection_thread":
        if action == "ask_one":
            return (
                _compact_text(f"Wollen wir bei {anchor} kurz anknuepfen?", max_len=112),
                _compact_text("Da ist zwischen uns noch etwas offen.", max_len=112),
            )
        return (
            _compact_text(f"Bei {anchor} bleibe ich noch kurz dran.", max_len=112),
            _compact_text("Der Faden ist noch nicht ganz zu Ende.", max_len=112),
        )
    if candidate_family == "reflection_preference":
        return (
            _compact_text(f"Bei {anchor} moechte ich dich noch etwas besser verstehen.", max_len=112),
            _compact_text("Ein kleiner Hinweis reicht mir da schon.", max_len=112),
        )
    if attention_state == "growing":
        return (
            _compact_text(f"Rund um {anchor} schaue ich heute noch einmal hin.", max_len=112),
            _compact_text("Ein kurzer Blick darauf waere gut.", max_len=112),
        )
    return (
        _compact_text(f"Zu {anchor} wuerde ich gern kurz anknuepfen.", max_len=112),
        _compact_text("Ein kleiner Nachtrag dazu waere schon hilfreich.", max_len=112),
    )


def _packet_topic_key(packet: LongTermMidtermPacketV1, *, title: str) -> str:
    """Return one stable semantic topic key for one reflection packet."""

    attributes = _mapping(packet.attributes)
    primary_hint = _primary_hint_phrase(packet.query_hints, max_len=48)
    kind = _compact_text(packet.kind, max_len=40).casefold()
    return _topic_key(
        attributes.get("person_name"),
        attributes.get("environment_id"),
        primary_hint,
        title if kind in _CONTINUITY_PACKET_KINDS else None,
        packet.packet_id,
        packet.kind,
    )


def _packet_salience(packet: LongTermMidtermPacketV1, *, now: datetime, max_age_days: float) -> float:
    """Return one bounded salience score for one reflection packet."""

    attributes = _mapping(packet.attributes)
    packet_scope = _compact_text(attributes.get("packet_scope"), max_len=48).casefold()
    kind = _compact_text(packet.kind, max_len=40).casefold()
    recency = _recency_score(packet.updated_at, now=now, max_age_days=max_age_days)
    if kind in _CONTINUITY_PACKET_KINDS:
        return min(0.96, max(0.48, 0.48 + (recency * 0.34)))
    if kind == "preference":
        return min(0.92, max(0.4, 0.4 + (recency * 0.22)))
    if packet_scope == "recent_environment_reflection":
        return min(0.88, max(0.36, 0.36 + (recency * 0.22)))
    if kind in _DEVICE_PACKET_KINDS:
        return min(0.72, max(0.24, 0.24 + (recency * 0.16)))
    return min(0.84, max(0.3, 0.3 + (recency * 0.2)))


def _packet_symbol_and_accent(packet: LongTermMidtermPacketV1, *, attention_state: str) -> tuple[str, str]:
    """Return one visual style pair for a reflection packet."""

    kind = _compact_text(packet.kind, max_len=40).casefold()
    if attention_state == "shared_thread":
        return ("question", "warm")
    if kind == "preference":
        return ("sparkles", "warm")
    if kind in _DEVICE_PACKET_KINDS:
        return ("sparkles", "info")
    return ("sparkles", "info")


def _packet_candidate(packet: LongTermMidtermPacketV1, *, now: datetime, max_age_days: float) -> AmbientDisplayImpulseCandidate | None:
    """Convert one eligible midterm packet into a reserve candidate."""

    if _compact_text(packet.sensitivity, max_len=24).casefold() not in _ALLOWED_SENSITIVITY:
        return None
    if packet.updated_at.astimezone(timezone.utc) < now - timedelta(days=max_age_days):
        return None
    action, attention_state = _packet_action(packet)
    if action == "silent":
        return None
    kind = _compact_text(packet.kind, max_len=40).casefold()
    if kind in _CONTINUITY_PACKET_KINDS and not _continuity_packet_has_displayable_anchor(packet):
        return None
    title = _packet_title(packet)
    topic_key = _packet_topic_key(packet, title=title)
    if not topic_key:
        return None
    salience = _packet_salience(packet, now=now, max_age_days=max_age_days)
    details = _compact_text(packet.details, max_len=140)
    symbol, accent = _packet_symbol_and_accent(packet, attention_state=attention_state)
    candidate_family = _packet_candidate_family(packet)
    headline, body = _packet_display_copy(
        packet,
        title=title,
        action=action,
        attention_state=attention_state,
        candidate_family=candidate_family,
    )
    packet_attributes = _mapping(packet.attributes)
    return AmbientDisplayImpulseCandidate(
        topic_key=topic_key,
        title=title,
        source="reflection_midterm",
        action=action,
        attention_state=attention_state,
        salience=salience,
        eyebrow="",
        headline=headline,
        body=body,
        symbol=symbol,
        accent=accent,
        reason="reflection_midterm",
        candidate_family=candidate_family,
        generation_context={
            "candidate_family": candidate_family,
            "display_anchor": _packet_display_anchor(packet, title=title),
            "hook_hint": _packet_hook_hint(packet),
            "reflection_kind": _compact_text(packet.kind, max_len=48),
            "topic_title": title,
            "summary": _compact_text(packet.summary, max_len=180),
            "details": details,
            "query_hints": tuple(_compact_text(value, max_len=48) for value in packet.query_hints[:4]),
            "attributes": {key: _compact_text(value, max_len=80) for key, value in packet_attributes.items()},
        },
    )


def _candidate_rank_key(candidate: AmbientDisplayImpulseCandidate) -> tuple[float, str, str]:
    """Return one stable descending rank key for reflection candidates."""

    return (float(candidate.salience), candidate.attention_state, candidate.topic_key)


def load_display_reserve_reflection_candidates(
    memory_service: LongTermMemoryService,
    *,
    config: TwinrConfig,
    local_now: datetime,
    max_items: int,
) -> tuple[AmbientDisplayImpulseCandidate, ...]:
    """Load reflection-derived reserve candidates from long-term durable state."""

    resolved = DisplayReserveReflectionConfig.from_config(config)
    limited_max = max(1, min(int(max_items), resolved.candidate_limit))
    effective_now = local_now.astimezone(timezone.utc)
    candidates: list[AmbientDisplayImpulseCandidate] = []
    for item in memory_service.object_store.load_objects():
        candidate = _summary_candidate(item, now=effective_now, max_age_days=resolved.max_age_days)
        if candidate is not None:
            candidates.append(candidate)
    for packet in memory_service.midterm_store.load_packets():
        candidate = _packet_candidate(packet, now=effective_now, max_age_days=resolved.max_age_days)
        if candidate is not None:
            candidates.append(candidate)
    deduped: dict[str, AmbientDisplayImpulseCandidate] = {}
    for candidate in candidates:
        current = deduped.get(candidate.topic_key)
        if current is None or _candidate_rank_key(candidate) > _candidate_rank_key(current):
            deduped[candidate.topic_key] = candidate
    ranked = sorted(deduped.values(), key=_candidate_rank_key, reverse=True)
    return tuple(ranked[:limited_max])


__all__ = [
    "DisplayReserveReflectionConfig",
    "load_display_reserve_reflection_candidates",
]
