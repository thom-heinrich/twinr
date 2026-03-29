# CHANGELOG: 2026-03-29
# BUG-1: Respect max_items<=0 and allow config candidate_limit=0; the old code always returned at least one candidate.
# BUG-2: Replace first-non-empty topic keys with visible semantic keys to fix silent dedupe collisions and duplicate cards.
# BUG-3: Skip malformed persisted records instead of letting one bad record blank the whole reflection lane.
# BUG-4: device_context/device_interaction packets were unintentionally suppressed despite dedicated context handling; they can now surface again.
# SEC-1: Redact PII-like literals, strip control/BiDi characters, and stop using raw transcript excerpts as visible anchors.
# IMP-1: Upgrade salience to a multi-factor score (recency, confidence, anchor quality, content prior, corroboration).
# IMP-2: Add semantic merge + soft family diversity so the right lane is less repetitive.
# IMP-3: Harden datetime/query-hint coercion for mixed legacy data on long-lived Raspberry Pi deployments.

"""Convert durable reflection outputs into right-lane companion candidates."""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.memory.longterm.core.models import LongTermMemoryObjectV1, LongTermMidtermPacketV1
from twinr.memory.longterm.runtime.service import LongTermMemoryService

logger = logging.getLogger(__name__)

_DEFAULT_CANDIDATE_LIMIT = 3
_DEFAULT_MAX_AGE_DAYS = 14.0
_DEFAULT_FAMILY_SOFT_CAP = 1

_ALLOWED_SENSITIVITY = frozenset({"low", "normal"})
_ALLOWED_VISIBLE_ANCHOR_ORIGINS = frozenset({"display_anchor", "person_name", "environment_id"})
_CONTINUITY_PACKET_KINDS = frozenset({"recent_turn_continuity", "conversation_context"})
_HIDDEN_CONTINUITY_PACKET_KINDS = frozenset({"recent_turn_continuity"})
_DEVICE_PACKET_KINDS = frozenset({"device_context", "device_interaction"})
_SUPPRESSED_PACKET_KINDS = frozenset({"interaction_quality"})
_META_PACKET_KINDS = frozenset({"conversation_state", "policy_context"})

_BIDI_AND_FORMAT_CODEPOINTS = frozenset(
    {
        "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
        "\u2066", "\u2067", "\u2068", "\u2069",
        "\u200e", "\u200f", "\u061c", "\ufeff",
    }
)
_EMAIL_RE = re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")
_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+\b")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_UUID_RE = re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b")
_LONG_NUMBER_RE = re.compile(r"(?<!\d)(?:\+?\d[\d .()/:-]{6,}\d)(?!\d)")
_TOKEN_RE = re.compile(r"(?i)\b(?:api[_-]?key|token|secret|bearer)\s*[:=]\s*[^\s,;]+")
_GENERIC_ANCHOR_RE = re.compile(r"(?i)^(?:das|dies|diese?s?|etwas|thema|sache|nachtrag|update|info)$")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _local_timezone() -> tzinfo:
    return datetime.now().astimezone().tzinfo or timezone.utc


def _safe_str(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:  # pragma: no cover
        try:
            return repr(value)
        except Exception:
            return ""


def _strip_controls_and_bidi(text: str) -> str:
    parts: list[str] = []
    for char in text:
        if char in _BIDI_AND_FORMAT_CODEPOINTS:
            continue
        category = unicodedata.category(char)
        if category.startswith("C") and char not in {"\n", "\r", "\t"}:
            continue
        parts.append(char)
    return "".join(parts)


def _redact_sensitive_text(text: str) -> str:
    text = _TOKEN_RE.sub("[redacted-secret]", text)
    text = _EMAIL_RE.sub("[redacted-email]", text)
    text = _URL_RE.sub("[redacted-link]", text)
    text = _IPV4_RE.sub("[redacted-ip]", text)
    text = _UUID_RE.sub("[redacted-id]", text)
    text = _LONG_NUMBER_RE.sub("[redacted-number]", text)
    return text


def _compact_text(value: object | None, *, max_len: int, redact: bool = False) -> str:
    if value is None:
        return ""
    raw = _safe_str(value)
    if len(raw) > max(4096, max_len * 8):
        raw = raw[: max(4096, max_len * 8)]
    compact = " ".join(_strip_controls_and_bidi(raw).split()).strip()
    if redact and compact:
        compact = " ".join(_redact_sensitive_text(compact).split()).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _mapping(value: Mapping[str, object] | None) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Sequence[object] | None) -> tuple[object, ...]:
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return ()
    if isinstance(value, Sequence):
        return tuple(value)
    return ()


def _coerce_days(value: object, *, default: float, minimum: float, maximum: float) -> float:
    if not isinstance(value, (int, float, str)):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    if math.isinf(number):
        return maximum if number > 0 else minimum
    return max(minimum, min(maximum, number))


def _coerce_probability(value: object, *, default: float = 0.5) -> float:
    if not isinstance(value, (int, float, str)):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return max(0.0, min(1.0, number))


def _to_utc(value: datetime, *, assume_tz: tzinfo) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"Expected datetime, got {type(value)!r}")
    if value.tzinfo is None:
        value = value.replace(tzinfo=assume_tz)
    return value.astimezone(timezone.utc)


def _normalize_label(value: object | None, *, max_len: int = 72, redact: bool = False) -> str:
    return _compact_text(value, max_len=max_len, redact=redact).replace("_", " ").replace(":", " ")


def _sentence_label(value: object | None, *, max_len: int = 72, redact: bool = False) -> str:
    text = _normalize_label(value, max_len=max_len, redact=redact)
    return text[:1].upper() + text[1:] if text else ""


def _normalized_key_part(value: object | None, *, max_len: int = 96) -> str:
    text = _compact_text(value, max_len=max_len).casefold()
    return re.sub(r"\s+", " ", text).strip()


def _join_topic_key(*values: object | None) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for value in values:
        part = _normalized_key_part(value)
        if not part or part in seen:
            continue
        seen.add(part)
        parts.append(part)
    return " | ".join(parts)


def _primary_hint_phrase(query_hints: Sequence[object] | None, *, max_len: int = 48) -> str:
    hints = _sequence(query_hints)
    if not hints:
        return ""
    for raw_hint in hints:
        normalized = _normalize_label(raw_hint, max_len=max_len, redact=True)
        if " " in normalized:
            return normalized
    parts: list[str] = []
    for raw_hint in hints[:3]:
        normalized = _normalize_label(raw_hint, max_len=24, redact=True)
        if not normalized:
            continue
        candidate = " ".join((*parts, normalized))
        if len(candidate) > max_len and parts:
            break
        parts.append(normalized)
    return " ".join(parts)


def _looks_like_safe_visible_anchor(text: str) -> bool:
    if not text:
        return False
    lowered = text.casefold()
    if lowered.startswith("[redacted-") or lowered in {"[redacted]", "redacted"}:
        return False
    if _GENERIC_ANCHOR_RE.match(lowered):
        return False
    letters = sum(char.isalpha() for char in text)
    digits = sum(char.isdigit() for char in text)
    punctuation = sum(not char.isalnum() and not char.isspace() for char in text)
    return len(text) >= 2 and letters > 0 and digits <= letters and punctuation <= max(6, len(text) // 3)


def _anchor_quality(display_anchor: str, *, origin: str) -> float:
    if not display_anchor:
        return 0.0
    if origin in {"display_anchor", "person_name", "environment_id"}:
        return 1.0
    if origin == "transcript_excerpt":
        return 0.55
    return 0.35


def _half_life_days_for_family(candidate_family: str) -> float:
    if candidate_family == "reflection_thread":
        return 4.0
    if candidate_family == "reflection_context":
        return 3.0
    if candidate_family == "reflection_preference":
        return 7.0
    return 5.0


def _recency_score(updated_at: datetime, *, now: datetime, candidate_family: str, max_age_days: float) -> float:
    age_days = max(0.0, (now - updated_at).total_seconds() / 86_400.0)
    if age_days >= max_age_days:
        return 0.0
    half_life = min(max_age_days, _half_life_days_for_family(candidate_family))
    return max(0.0, min(1.0, math.exp(-math.log(2.0) * age_days / max(0.1, half_life))))


def _content_type_prior(candidate_family: str, *, action: str, attention_state: str) -> float:
    if candidate_family == "reflection_thread":
        return 0.92 if action == "ask_one" else 0.82
    if candidate_family == "reflection_preference":
        return 0.84
    if candidate_family == "reflection_context":
        return 0.78 if attention_state == "growing" else 0.72
    return 0.68


def _compute_salience(
    *,
    candidate_family: str,
    updated_at: datetime,
    now: datetime,
    max_age_days: float,
    confidence: float,
    display_anchor: str,
    anchor_origin: str,
    action: str,
    attention_state: str,
    corroboration_count: int = 1,
) -> float:
    score = (
        0.42 * _recency_score(updated_at, now=now, candidate_family=candidate_family, max_age_days=max_age_days)
        + 0.24 * confidence
        + 0.18 * _anchor_quality(display_anchor, origin=anchor_origin)
        + 0.16 * _content_type_prior(candidate_family, action=action, attention_state=attention_state)
        + min(0.12, max(0.0, (corroboration_count - 1) * 0.06))
    )
    family_cap = {
        "reflection_thread": 0.78,
        "reflection_preference": 0.74,
        "reflection_context": 0.70,
        "reflection_summary": 0.72,
    }.get(candidate_family, 0.78)
    return max(0.18, min(family_cap, score))


@dataclass(frozen=True, slots=True)
class DisplayReserveReflectionConfig:
    candidate_limit: int = _DEFAULT_CANDIDATE_LIMIT
    max_age_days: float = _DEFAULT_MAX_AGE_DAYS
    family_soft_cap: int = _DEFAULT_FAMILY_SOFT_CAP

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayReserveReflectionConfig":
        raw_limit = getattr(
            config,
            "display_reserve_bus_reflection_candidate_limit",
            _DEFAULT_CANDIDATE_LIMIT,
        )
        try:
            candidate_limit = int(raw_limit)
        except (TypeError, ValueError):
            candidate_limit = _DEFAULT_CANDIDATE_LIMIT
        return cls(
            candidate_limit=max(0, candidate_limit),
            max_age_days=_coerce_days(
                getattr(config, "display_reserve_bus_reflection_max_age_days", _DEFAULT_MAX_AGE_DAYS),
                default=_DEFAULT_MAX_AGE_DAYS,
                minimum=1.0,
                maximum=90.0,
            ),
            family_soft_cap=max(
                1,
                int(
                    getattr(
                        config,
                        "display_reserve_bus_reflection_family_soft_cap",
                        _DEFAULT_FAMILY_SOFT_CAP,
                    )
                ),
            ),
        )


def _summary_title(item: LongTermMemoryObjectV1) -> str:
    attributes = _mapping(item.attributes)
    for value in (
        attributes.get("display_anchor"),
        attributes.get("person_name"),
        attributes.get("environment_id"),
    ):
        label = _normalize_label(value, max_len=72, redact=True)
        if _looks_like_safe_visible_anchor(label):
            return label
    return ""


def _summary_hook_hint(item: LongTermMemoryObjectV1) -> str:
    attributes = _mapping(item.attributes)
    for value in (attributes.get("display_anchor"), item.summary, item.details):
        text = _compact_text(value, max_len=160, redact=True)
        if text:
            return text
    return ""


def _summary_action(item: LongTermMemoryObjectV1) -> tuple[str, str]:
    attributes = _mapping(item.attributes)
    summary_type = _compact_text(attributes.get("summary_type"), max_len=32).casefold()
    memory_domain = _compact_text(attributes.get("memory_domain"), max_len=48).casefold()
    if summary_type == "thread" or _compact_text(attributes.get("person_name"), max_len=80, redact=True):
        return ("ask_one", "shared_thread")
    if memory_domain == "smart_home_environment" or summary_type == "environment_reflection":
        return ("brief_update", "growing")
    return ("hint", "forming")


def _summary_candidate_family(item: LongTermMemoryObjectV1) -> str:
    attributes = _mapping(item.attributes)
    summary_type = _compact_text(attributes.get("summary_type"), max_len=32).casefold()
    memory_domain = _compact_text(attributes.get("memory_domain"), max_len=48).casefold()
    if summary_type == "thread" or _compact_text(attributes.get("person_name"), max_len=80, redact=True):
        return "reflection_thread"
    if memory_domain == "smart_home_environment" or summary_type == "environment_reflection":
        return "reflection_context"
    return "reflection_summary"


def _summary_card_intent(item: LongTermMemoryObjectV1, *, title: str) -> dict[str, str]:
    attributes = _mapping(item.attributes)
    summary_type = _compact_text(attributes.get("summary_type"), max_len=32).casefold()
    memory_domain = _compact_text(attributes.get("memory_domain"), max_len=48).casefold()
    anchor = _compact_text(title, max_len=96, redact=True) or "dem Thema"
    if summary_type == "thread" or _compact_text(attributes.get("person_name"), max_len=80, redact=True):
        return {
            "topic_semantics": f"frueherer gemeinsamer Faden zu {anchor}",
            "statement_intent": f"Twinr soll ruhig daran erinnern, dass es zu {anchor} noch einen gemeinsamen Gespraechsfaden gibt.",
            "cta_intent": "Zu einem kurzen Rueckblick, Update oder Weiterreden einladen.",
            "relationship_stance": "ruhiger Rueckbezug statt Befund oder Ticket",
        }
    if memory_domain == "smart_home_environment" or summary_type == "environment_reflection":
        return {
            "topic_semantics": f"kuerzlich beobachteter Kontext rund um {anchor}",
            "statement_intent": f"Twinr soll einen konkreten kuerzlichen Kontext zu {anchor} ruhig ansprechen.",
            "cta_intent": "Zu einer kurzen Einordnung oder Reaktion einladen.",
            "relationship_stance": "sachlich, alltagsnah und nicht technisch",
        }
    return {
        "topic_semantics": f"reflektierter Nachtrag zu {anchor}",
        "statement_intent": f"Twinr soll einen ruhigen Nachtrag oder Rueckbezug zu {anchor} anstossen.",
        "cta_intent": "Zu einer kurzen Reaktion oder Ergaenzung einladen.",
        "relationship_stance": "ruhig und persoenlich statt meta oder abstrakt",
    }


def _summary_display_copy(item: LongTermMemoryObjectV1, *, title: str) -> tuple[str, str]:
    attributes = _mapping(item.attributes)
    summary_type = _compact_text(attributes.get("summary_type"), max_len=32).casefold()
    memory_domain = _compact_text(attributes.get("memory_domain"), max_len=48).casefold()
    anchor = _sentence_label(title, max_len=72, redact=True) or "das"
    if summary_type == "thread" or _compact_text(attributes.get("person_name"), max_len=80, redact=True):
        return (
            _compact_text(f"Bei {anchor} ist zwischen uns noch etwas offen.", max_len=112, redact=True),
            _compact_text("Wollen wir kurz darueber reden?", max_len=112, redact=True),
        )
    if memory_domain == "smart_home_environment" or summary_type == "environment_reflection":
        return (
            _compact_text(f"Rund um {anchor} schaue ich heute noch einmal hin.", max_len=112, redact=True),
            _compact_text("Magst du kurz was dazu sagen?", max_len=112, redact=True),
        )
    return (
        _compact_text(f"Zu {anchor} habe ich noch einen kleinen Nachtrag im Kopf.", max_len=112, redact=True),
        _compact_text("Wollen wir kurz darueber reden?", max_len=112, redact=True),
    )


def _summary_candidate(item: LongTermMemoryObjectV1, *, now: datetime, max_age_days: float) -> AmbientDisplayImpulseCandidate | None:
    if item.kind != "summary" or item.status != "active":
        return None
    if _compact_text(item.sensitivity, max_len=24).casefold() not in _ALLOWED_SENSITIVITY:
        return None
    updated_at = _to_utc(item.updated_at, assume_tz=timezone.utc)
    if updated_at < now - timedelta(days=max_age_days):
        return None
    title = _summary_title(item)
    if not title:
        return None

    action, attention_state = _summary_action(item)
    candidate_family = _summary_candidate_family(item)
    attributes = _mapping(item.attributes)
    anchor_origin = "display_anchor" if _compact_text(attributes.get("display_anchor"), max_len=72, redact=True) else (
        "person_name" if _compact_text(attributes.get("person_name"), max_len=72, redact=True) else "environment_id"
    )
    confidence = _coerce_probability(getattr(item, "confidence", 0.5), default=0.5)
    headline, body = _summary_display_copy(item, title=title)
    details = _compact_text(item.details, max_len=140, redact=True)

    return AmbientDisplayImpulseCandidate(
        topic_key=_summary_semantic_topic_key(item, title=title),
        semantic_topic_key=_summary_semantic_topic_key(item, title=title),
        title=title,
        source="reflection_summary",
        action=action,
        attention_state=attention_state,
        salience=_compute_salience(
            candidate_family=candidate_family,
            updated_at=updated_at,
            now=now,
            max_age_days=max_age_days,
            confidence=confidence,
            display_anchor=title,
            anchor_origin=anchor_origin,
            action=action,
            attention_state=attention_state,
        ),
        eyebrow="",
        headline=headline,
        body=body,
        symbol="sparkles",
        accent="warm" if attention_state == "shared_thread" else "info",
        reason="reflection_summary",
        candidate_family=candidate_family,
        generation_context={
            "candidate_family": candidate_family,
            "display_anchor": title,
            "display_anchor_origin": anchor_origin,
            "hook_hint": _summary_hook_hint(item),
            "card_intent": _summary_card_intent(item, title=title),
            "reflection_kind": _compact_text(attributes.get("summary_type"), max_len=40) or "summary",
            "topic_title": title,
            "summary": _compact_text(item.summary, max_len=180, redact=True),
            "details": details,
            "memory_domain": _compact_text(attributes.get("memory_domain"), max_len=48, redact=True),
            "source_text_trust": "untrusted_memory",
            "salience_factors": {
                "confidence": confidence,
                "recency": _recency_score(updated_at, now=now, candidate_family=candidate_family, max_age_days=max_age_days),
                "anchor_quality": _anchor_quality(title, origin=anchor_origin),
                "content_type_prior": _content_type_prior(candidate_family, action=action, attention_state=attention_state),
            },
        },
    )


def _summary_semantic_topic_key(item: LongTermMemoryObjectV1, *, title: str) -> str:
    """Return one stable grouped semantic key for a reflection summary."""

    return (
        _join_topic_key(item.slot_key)
        or _join_topic_key(item.memory_id)
        or _join_topic_key(item.value_key)
        or _join_topic_key(title)
    )


def _packet_title(packet: LongTermMidtermPacketV1) -> str:
    attributes = _mapping(packet.attributes)
    for value in (
        attributes.get("display_anchor"),
        attributes.get("person_name"),
        attributes.get("environment_id"),
        packet.summary,
    ):
        label = _sentence_label(value, max_len=72, redact=True)
        if label and _looks_like_safe_visible_anchor(label):
            return label
    return ""


def _packet_anchor_origin(packet: LongTermMidtermPacketV1) -> str:
    attributes = _mapping(packet.attributes)
    if _sentence_label(attributes.get("display_anchor"), max_len=72, redact=True):
        return "display_anchor"
    if _sentence_label(attributes.get("person_name"), max_len=72, redact=True):
        return "person_name"
    if _sentence_label(attributes.get("environment_id"), max_len=72, redact=True):
        return "environment_id"
    if _sentence_label(attributes.get("transcript_excerpt"), max_len=72, redact=True):
        return "transcript_excerpt"
    return "unknown"


def _packet_explicit_display_anchor(packet: LongTermMidtermPacketV1) -> str:
    attributes = _mapping(packet.attributes)
    for value in (
        attributes.get("display_anchor"),
        attributes.get("person_name"),
        attributes.get("environment_id"),
    ):
        label = _sentence_label(value, max_len=72, redact=True)
        if _looks_like_safe_visible_anchor(label):
            return label
    return ""


def _packet_structured_topic_anchor(packet: LongTermMidtermPacketV1) -> str:
    return _packet_explicit_display_anchor(packet)


def _packet_action(packet: LongTermMidtermPacketV1) -> tuple[str, str]:
    attributes = _mapping(packet.attributes)
    packet_scope = _compact_text(attributes.get("packet_scope"), max_len=48).casefold()
    persistence_scope = _compact_text(attributes.get("persistence_scope"), max_len=48).casefold()
    kind = _compact_text(packet.kind, max_len=40).casefold()

    if persistence_scope == "restart_recall" or kind in _SUPPRESSED_PACKET_KINDS or kind in _META_PACKET_KINDS:
        return ("silent", "background")
    if kind in _CONTINUITY_PACKET_KINDS:
        return ("ask_one", "shared_thread")
    if kind == "preference":
        return ("ask_one", "forming")
    # BREAKING: device_context/device_interaction packets can now surface when they
    # carry a structured anchor, instead of being forced silent.
    if packet_scope == "recent_environment_reflection" or kind in _DEVICE_PACKET_KINDS:
        return ("brief_update", "growing")
    if kind == "interaction":
        return ("hint", "forming")
    return ("brief_update", "forming")


def _packet_allowed_for_visible_lane(packet: LongTermMidtermPacketV1) -> bool:
    return _compact_text(packet.kind, max_len=40).casefold() not in _HIDDEN_CONTINUITY_PACKET_KINDS


def _packet_candidate_family(packet: LongTermMidtermPacketV1) -> str:
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
    del title
    if _compact_text(packet.kind, max_len=40).casefold() in _CONTINUITY_PACKET_KINDS:
        return _packet_structured_topic_anchor(packet)
    return _packet_explicit_display_anchor(packet)


def _packet_hook_hint(packet: LongTermMidtermPacketV1) -> str:
    attributes = _mapping(packet.attributes)
    for value in (
        attributes.get("response_excerpt"),
        attributes.get("transcript_excerpt"),
        _primary_hint_phrase(packet.query_hints, max_len=96),
        packet.summary,
    ):
        text = _compact_text(value, max_len=160, redact=True)
        if text:
            return text
    return ""


def _packet_display_goal(candidate_family: str) -> str:
    if candidate_family == "reflection_thread":
        return "call_back_to_earlier_conversation"
    if candidate_family == "reflection_preference":
        return "surface_personal_preference"
    if candidate_family == "reflection_context":
        return "raise_recent_context"
    return "raise_reflection_follow_up"


def _packet_topic_summary(*, candidate_family: str, display_anchor: str, anchor_origin: str) -> str:
    anchor = _compact_text(display_anchor, max_len=96, redact=True) or "das Thema"
    if candidate_family == "reflection_thread":
        if anchor_origin in {"display_anchor", "person_name", "environment_id"}:
            return _compact_text(
                f"Frueherer Gespraechsfaden zu {anchor}; ruhig daran anknuepfen, nicht wie neue Diagnose oder Stoerungsmeldung formulieren.",
                max_len=180,
                redact=True,
            )
        return _compact_text(
            f"Frueherer offener Gespraechsfaden zu {anchor}; eher wie natuerliches Nachfassen als wie neue Meldung formulieren.",
            max_len=180,
            redact=True,
        )
    if candidate_family == "reflection_preference":
        return _compact_text(
            f"Gelerntes ueber {anchor}; alltagsnah und persoenlich, nicht technisch oder etikettenhaft formulieren.",
            max_len=180,
            redact=True,
        )
    return _compact_text(
        f"Reflektierter Anlass zu {anchor}; alltagsnah und konkret formulieren.",
        max_len=180,
        redact=True,
    )


def _packet_card_intent(*, candidate_family: str, display_anchor: str, anchor_origin: str) -> dict[str, str]:
    del anchor_origin
    anchor = _compact_text(display_anchor, max_len=96, redact=True) or "dem Thema"
    if candidate_family == "reflection_thread":
        return {
            "topic_semantics": f"frueherer Gespraechsfaden zu {anchor}",
            "statement_intent": f"Twinr soll ruhig an einen frueheren Gespraechsfaden zu {anchor} anknuepfen.",
            "cta_intent": "Zu einem kurzen Weiterreden, Update oder Nachfassen einladen.",
            "relationship_stance": "ruhiger Rueckbezug statt Diagnose, Stoerung oder Supportfall",
        }
    if candidate_family == "reflection_preference":
        return {
            "topic_semantics": f"gelerntes persoenliches Detail zu {anchor}",
            "statement_intent": f"Twinr soll ein gelerntes persoenliches Detail zu {anchor} alltagsnah ansprechen.",
            "cta_intent": "Zu einer kurzen Bestaetigung, Ergaenzung oder Nuancierung einladen.",
            "relationship_stance": "persoenlich und aufmerksam statt etikettenhaft",
        }
    if candidate_family == "reflection_context":
        return {
            "topic_semantics": f"kuerzlicher Kontext rund um {anchor}",
            "statement_intent": f"Twinr soll einen kuerzlichen Kontext zu {anchor} ruhig und konkret ansprechen.",
            "cta_intent": "Zu einer kurzen Einordnung oder Reaktion einladen.",
            "relationship_stance": "alltagsnah und nicht technisch",
        }
    return {
        "topic_semantics": f"reflektierter Anlass zu {anchor}",
        "statement_intent": f"Twinr soll einen reflektierten Anlass zu {anchor} natuerlich und konkret ansprechen.",
        "cta_intent": "Zu einer kurzen Reaktion oder Ergaenzung einladen.",
        "relationship_stance": "ruhig und konkret statt meta",
    }


def _packet_display_copy(
    packet: LongTermMidtermPacketV1,
    *,
    title: str,
    action: str,
    attention_state: str,
    candidate_family: str,
) -> tuple[str, str]:
    anchor = _packet_display_anchor(packet, title=title) or "das"
    if candidate_family == "reflection_thread":
        if action == "ask_one":
            return (
                _compact_text(f"Bei {anchor} ist zwischen uns noch etwas offen.", max_len=112, redact=True),
                _compact_text("Wollen wir kurz darueber reden?", max_len=112, redact=True),
            )
        return (
            _compact_text(f"Bei {anchor} bleibe ich noch kurz dran.", max_len=112, redact=True),
            _compact_text("Magst du kurz was dazu sagen?", max_len=112, redact=True),
        )
    if candidate_family == "reflection_preference":
        return (
            _compact_text(f"Bei {anchor} moechte ich dich noch etwas besser verstehen.", max_len=112, redact=True),
            _compact_text("Magst du mir kurz mehr dazu sagen?", max_len=112, redact=True),
        )
    if attention_state == "growing":
        return (
            _compact_text(f"Rund um {anchor} schaue ich heute noch einmal hin.", max_len=112, redact=True),
            _compact_text("Wollen wir kurz draufschauen?", max_len=112, redact=True),
        )
    return (
        _compact_text(f"Zu {anchor} habe ich noch einen kleinen Nachtrag im Kopf.", max_len=112, redact=True),
        _compact_text("Magst du kurz was dazu sagen?", max_len=112, redact=True),
    )


def _packet_symbol_and_accent(packet: LongTermMidtermPacketV1, *, attention_state: str) -> tuple[str, str]:
    kind = _compact_text(packet.kind, max_len=40).casefold()
    if attention_state == "shared_thread":
        return ("question", "warm")
    if kind == "preference":
        return ("sparkles", "warm")
    return ("sparkles", "info")


def _packet_candidate(packet: LongTermMidtermPacketV1, *, now: datetime, max_age_days: float) -> AmbientDisplayImpulseCandidate | None:
    if _compact_text(packet.sensitivity, max_len=24).casefold() not in _ALLOWED_SENSITIVITY:
        return None
    if not _packet_allowed_for_visible_lane(packet):
        return None

    updated_at = _to_utc(packet.updated_at, assume_tz=timezone.utc)
    if updated_at < now - timedelta(days=max_age_days):
        return None

    action, attention_state = _packet_action(packet)
    if action == "silent":
        return None

    kind = _compact_text(packet.kind, max_len=40).casefold()
    if kind in _CONTINUITY_PACKET_KINDS and not _packet_structured_topic_anchor(packet):
        return None

    title = _packet_title(packet)
    display_anchor = _packet_display_anchor(packet, title=title)
    anchor_origin = _packet_anchor_origin(packet)

    # BREAKING: transcript excerpts can no longer become visible topic anchors.
    if not display_anchor or anchor_origin not in _ALLOWED_VISIBLE_ANCHOR_ORIGINS:
        return None

    candidate_family = _packet_candidate_family(packet)
    packet_attributes = _mapping(packet.attributes)
    confidence = _coerce_probability(packet_attributes.get("confidence"), default=0.62)
    headline, body = _packet_display_copy(
        packet,
        title=display_anchor,
        action=action,
        attention_state=attention_state,
        candidate_family=candidate_family,
    )
    symbol, accent = _packet_symbol_and_accent(packet, attention_state=attention_state)
    details = _compact_text(packet.details, max_len=140, redact=True)

    return AmbientDisplayImpulseCandidate(
        topic_key=_join_topic_key(display_anchor),
        semantic_topic_key=_packet_semantic_topic_key(packet, display_anchor=display_anchor, title=title),
        title=display_anchor,
        source="reflection_midterm",
        action=action,
        attention_state=attention_state,
        salience=_compute_salience(
            candidate_family=candidate_family,
            updated_at=updated_at,
            now=now,
            max_age_days=max_age_days,
            confidence=confidence,
            display_anchor=display_anchor,
            anchor_origin=anchor_origin,
            action=action,
            attention_state=attention_state,
        ),
        eyebrow="",
        headline=headline,
        body=body,
        symbol=symbol,
        accent=accent,
        reason="reflection_midterm",
        candidate_family=candidate_family,
        generation_context={
            "candidate_family": candidate_family,
            "display_goal": _packet_display_goal(candidate_family),
            "display_anchor": display_anchor,
            "display_anchor_origin": anchor_origin,
            "hook_hint": _packet_hook_hint(packet),
            "card_intent": _packet_card_intent(
                candidate_family=candidate_family,
                display_anchor=display_anchor,
                anchor_origin=anchor_origin,
            ),
            "topic_summary": _packet_topic_summary(
                candidate_family=candidate_family,
                display_anchor=display_anchor,
                anchor_origin=anchor_origin,
            ),
            "reflection_kind": _compact_text(packet.kind, max_len=48, redact=True),
            "topic_title": display_anchor,
            "summary": _compact_text(packet.summary, max_len=180, redact=True),
            "details": details,
            "query_hints": tuple(_compact_text(value, max_len=48, redact=True) for value in _sequence(packet.query_hints)[:4]),
            "attributes": {
                key: _compact_text(value, max_len=80, redact=True)
                for key, value in packet_attributes.items()
                if key in {"display_anchor", "person_name", "environment_id", "packet_scope", "persistence_scope", "confidence"}
            },
            "transcript_excerpt": _compact_text(packet_attributes.get("transcript_excerpt"), max_len=160, redact=True),
            "response_excerpt": _compact_text(packet_attributes.get("response_excerpt"), max_len=160, redact=True),
            "source_text_trust": "untrusted_memory",
            "salience_factors": {
                "confidence": confidence,
                "recency": _recency_score(updated_at, now=now, candidate_family=candidate_family, max_age_days=max_age_days),
                "anchor_quality": _anchor_quality(display_anchor, origin=anchor_origin),
                "content_type_prior": _content_type_prior(candidate_family, action=action, attention_state=attention_state),
            },
        },
    )


def _packet_semantic_topic_key(
    packet: LongTermMidtermPacketV1,
    *,
    display_anchor: str,
    title: str,
) -> str:
    """Return one grouped semantic key for a reflection packet."""

    return _join_topic_key(display_anchor) or _join_topic_key(title) or _join_topic_key(packet.packet_id, packet.kind)


def _safe_convert_summary(item: LongTermMemoryObjectV1, *, now: datetime, max_age_days: float) -> AmbientDisplayImpulseCandidate | None:
    try:
        return _summary_candidate(item, now=now, max_age_days=max_age_days)
    except Exception:
        logger.warning("Skipping malformed reflection summary object", exc_info=True)
        return None


def _safe_convert_packet(packet: LongTermMidtermPacketV1, *, now: datetime, max_age_days: float) -> AmbientDisplayImpulseCandidate | None:
    try:
        return _packet_candidate(packet, now=now, max_age_days=max_age_days)
    except Exception:
        logger.warning("Skipping malformed reflection midterm packet", exc_info=True)
        return None


def _candidate_source_priority(candidate: AmbientDisplayImpulseCandidate) -> int:
    if candidate.source == "reflection_summary":
        return 2
    if candidate.source == "reflection_midterm":
        return 1
    return 0


def _candidate_rank_key(candidate: AmbientDisplayImpulseCandidate) -> tuple[float, int, int, str]:
    attention_priority = {"shared_thread": 3, "growing": 2, "forming": 1, "background": 0}.get(candidate.attention_state, 0)
    return (float(candidate.salience), attention_priority, _candidate_source_priority(candidate), candidate.topic_key)


def _candidate_semantic_key(candidate: AmbientDisplayImpulseCandidate) -> str:
    return candidate.semantic_key() or _join_topic_key(candidate.title)


def _merge_generation_context(
    base: Mapping[str, Any] | None,
    incoming: Mapping[str, Any] | None,
    *,
    corroboration_count: int,
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base or {})
    for key, value in (incoming or {}).items():
        if key not in merged or not merged[key]:
            merged[key] = value
    merged["corroboration_count"] = corroboration_count
    return merged


def _merge_candidate_group(group: Sequence[AmbientDisplayImpulseCandidate]) -> AmbientDisplayImpulseCandidate:
    best = max(group, key=_candidate_rank_key)
    corroboration_count = len(group)
    merged_context = _merge_generation_context(
        best.generation_context if isinstance(best.generation_context, Mapping) else {},
        {},
        corroboration_count=corroboration_count,
    )
    return AmbientDisplayImpulseCandidate(
        topic_key=best.topic_key,
        semantic_topic_key=best.semantic_topic_key,
        title=best.title,
        source=best.source,
        action=best.action,
        attention_state=best.attention_state,
        salience=min(0.99, float(best.salience) + min(0.12, max(0.0, (corroboration_count - 1) * 0.06))),
        eyebrow=best.eyebrow,
        headline=best.headline,
        body=best.body,
        symbol=best.symbol,
        accent=best.accent,
        reason=best.reason,
        candidate_family=best.candidate_family,
        generation_context=merged_context,
    )


def _select_diverse_candidates(
    ranked: Sequence[AmbientDisplayImpulseCandidate],
    *,
    limit: int,
    family_soft_cap: int,
) -> tuple[AmbientDisplayImpulseCandidate, ...]:
    if limit <= 0:
        return ()
    selected: list[AmbientDisplayImpulseCandidate] = []
    deferred: list[AmbientDisplayImpulseCandidate] = []
    family_counts: dict[str, int] = {}

    for candidate in ranked:
        family = _compact_text(candidate.candidate_family, max_len=64) or "reflection"
        if family_counts.get(family, 0) < family_soft_cap:
            selected.append(candidate)
            family_counts[family] = family_counts.get(family, 0) + 1
            if len(selected) >= limit:
                return tuple(selected[:limit])
        else:
            deferred.append(candidate)

    for candidate in deferred:
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return tuple(selected[:limit])


def load_display_reserve_reflection_candidates(
    memory_service: LongTermMemoryService,
    *,
    config: TwinrConfig,
    local_now: datetime,
    max_items: int,
) -> tuple[AmbientDisplayImpulseCandidate, ...]:
    resolved = DisplayReserveReflectionConfig.from_config(config)
    if max_items <= 0 or resolved.candidate_limit <= 0:
        return ()

    limited_max = min(int(max_items), resolved.candidate_limit)
    if limited_max <= 0:
        return ()

    effective_now = _to_utc(local_now, assume_tz=_local_timezone())
    grouped: dict[str, list[AmbientDisplayImpulseCandidate]] = {}

    for item in memory_service.object_store.load_objects():
        candidate = _safe_convert_summary(item, now=effective_now, max_age_days=resolved.max_age_days)
        if candidate is not None:
            grouped.setdefault(_candidate_semantic_key(candidate), []).append(candidate)

    for packet in memory_service.midterm_store.load_packets():
        candidate = _safe_convert_packet(packet, now=effective_now, max_age_days=resolved.max_age_days)
        if candidate is not None:
            grouped.setdefault(_candidate_semantic_key(candidate), []).append(candidate)

    merged = [_merge_candidate_group(group) for group in grouped.values()]
    ranked = sorted(merged, key=_candidate_rank_key, reverse=True)
    return _select_diverse_candidates(ranked, limit=limited_max, family_soft_cap=resolved.family_soft_cap)


__all__ = [
    "DisplayReserveReflectionConfig",
    "load_display_reserve_reflection_candidates",
]
