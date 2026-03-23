"""Build personality-shaped ambient display copy from Twinr's companion state.

The display reserve should feel like a living conversational opening, not like
an internal topic label. This module therefore turns structured mindshare items
plus positive-engagement policy into short, readable prompts that carry a clear
topic anchor and at least a little of Twinr's own tone. The output should stay
calm and legible, but it should not read like a blank notification template.

The logic stays generic and policy-driven:

- no named-topic hardcoding
- no regex topic parsing
- no hidden prompt text in the runtime loop
- deterministic bounded variation from stable item keys and the local day
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib

from twinr.agent.personality.positive_engagement import PositiveEngagementTopicPolicy
from twinr.agent.personality.self_expression import CompanionMindshareItem


@dataclass(frozen=True, slots=True)
class AmbientDisplayImpulseCopy:
    """Describe the user-facing copy for one reserve-lane impulse."""

    eyebrow: str = ""
    headline: str = ""
    body: str = ""
    symbol: str = "question"
    accent: str = "info"


def _normalized_text(value: object | None) -> str:
    """Collapse arbitrary text into one compact single line."""

    return " ".join(str(value or "").split()).strip()


def _truncate_text(value: object | None, *, max_len: int) -> str:
    """Return one bounded display-safe text field."""

    compact = _normalized_text(value)
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _stable_fraction(*parts: object) -> float:
    """Return one deterministic 0..1 fraction for bounded copy variation."""

    digest = hashlib.sha1(
        "::".join(_normalized_text(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") / 4_294_967_295.0


def _source_family(source: object | None) -> str:
    """Map one raw mindshare source onto a generic copy family."""

    normalized = _normalized_text(source).casefold()
    if normalized in {"continuity", "relationship"}:
        return "memory"
    if normalized == "place":
        return "place"
    if normalized in {"situational_awareness", "regional_news", "local_news", "world"}:
        return "world"
    return "general"


def _topic_phrase(item: CompanionMindshareItem) -> str:
    """Return one bounded human-readable topic phrase for direct questions."""

    return _truncate_text(item.title, max_len=52)


def _accent_for_policy(
    policy: PositiveEngagementTopicPolicy,
    *,
    source_family: str,
) -> str:
    """Choose one calm accent token for the reserve card."""

    if policy.attention_state == "shared_thread":
        return "warm"
    if source_family == "memory":
        return "warm"
    if policy.action == "brief_update":
        return "success"
    return "info"


def _symbol_for_policy(
    policy: PositiveEngagementTopicPolicy,
    *,
    source_family: str,
) -> str:
    """Choose one semantic symbol token for the impulse."""

    if source_family == "memory":
        return "heart"
    if policy.action in {"ask_one", "invite_follow_up"}:
        return "question"
    return "sparkles"


def _memory_question(
    topic: str,
    *,
    policy: PositiveEngagementTopicPolicy,
    seed: float,
) -> str:
    """Render one personal-thread question that helps Twinr remember better."""

    if policy.action == "hint":
        variants = (
            f"Soll ich bei {topic} weiter hinschauen?",
            f"Bei {topic} ist fuer mich noch etwas offen. Soll ich dranbleiben?",
        )
    elif policy.action == "brief_update":
        variants = (
            f"Sollen wir bei {topic} spaeter kurz anknuepfen?",
            f"Zu {topic} wuerde ich spaeter gern noch einmal kurz zurueckkommen.",
        )
    elif policy.action == "ask_one":
        variants = (
            f"Wie ist es bei {topic} weitergegangen?",
            f"Zu {topic} fehlt mir noch etwas. Magst du mich kurz auf Stand bringen?",
        )
    else:
        variants = (
            f"Bei {topic} habe ich noch ein Fragezeichen. Wie ging es weiter?",
            f"Zu {topic} fehlt mir noch ein Stueck. Magst du mich kurz auf Stand bringen?",
        )
    return variants[1 if seed >= 0.5 else 0]


def _world_question(
    topic: str,
    *,
    policy: PositiveEngagementTopicPolicy,
    source_family: str,
    seed: float,
) -> str:
    """Render one world/place-style conversational opening."""

    if policy.action == "hint":
        variants = (
            f"Soll ich {topic} heute weiter im Blick behalten?",
            f"Bei {topic} schaue ich gerade noch einmal hin. Soll ich dranbleiben?",
        )
    elif policy.action == "brief_update":
        variants = (
            f"Bei {topic} tut sich gerade etwas. Soll ich spaeter kurz updaten?",
            f"Zu {topic} koennte ich spaeter ein kurzes Update geben. Waere das gut?",
        )
    elif source_family == "world":
        variants = (
            f"Bei {topic} ist gerade wieder Bewegung drin. Wie siehst du das?",
            f"{topic} laesst mich heute nicht ganz los. Wie schaust du darauf?",
        )
    else:
        variants = (
            f"Rund um {topic} ist gerade etwas in Bewegung. Wie siehst du das?",
            f"Bei {topic} lohnt heute ein zweiter Blick. Was meinst du?",
        )
    return variants[1 if seed >= 0.5 else 0]


def _general_question(
    topic: str,
    *,
    policy: PositiveEngagementTopicPolicy,
    seed: float,
) -> str:
    """Render one generic but still concrete conversational opening."""

    if policy.action == "brief_update":
        variants = (
            f"Soll ich bei {topic} spaeter noch einmal kurz nachfassen?",
            f"Zu {topic} waere spaeter ein kleiner zweiter Blick gut. Passt das?",
        )
    else:
        variants = (
            f"Bei {topic} habe ich noch ein kleines Fragezeichen. Wie siehst du das?",
            f"Zu {topic} wuerde ich gern kurz deine Sicht hoeren. Was meinst du?",
        )
    return variants[1 if seed >= 0.5 else 0]


def _helper_text(
    topic: str,
    *,
    policy: PositiveEngagementTopicPolicy,
    source_family: str,
    seed: float,
) -> str:
    """Render one short supportive helper line under the main question."""

    if source_family == "memory":
        variants = (
            "Da fehlt mir noch ein klares Bild.",
            "Da habe ich noch ein kleines Fragezeichen.",
        )
        return variants[1 if seed >= 0.5 else 0]
    if policy.action == "hint":
        variants = (
            "Ein kurzer Hinweis reicht mir da schon.",
            "Ein kleiner Kompass waere dafuer schon genug.",
        )
        return variants[1 if seed >= 0.5 else 0]
    if policy.attention_state == "shared_thread":
        variants = (
            "Das ist gerade so ein kleiner Faden zwischen uns.",
            "Da bleibe ich ruhig, aber ziemlich wach dran.",
        )
        return variants[1 if seed >= 0.5 else 0]
    variants = (
        "Dein Blick darauf interessiert mich mehr als die Schlagzeile.",
        "Da wuerde ich gern kurz deine Sicht hoeren.",
    )
    return variants[1 if seed >= 0.5 else 0]


def build_ambient_display_impulse_copy(
    item: CompanionMindshareItem,
    policy: PositiveEngagementTopicPolicy,
    *,
    local_now: datetime | None,
) -> AmbientDisplayImpulseCopy:
    """Build one question-first display impulse from current companion state."""

    topic = _topic_phrase(item)
    source_family = _source_family(item.source)
    seed = _stable_fraction(
        item.title,
        item.source,
        policy.action,
        policy.attention_state,
        local_now.date() if local_now else "",
    )
    if source_family == "memory":
        headline = _memory_question(topic, policy=policy, seed=seed)
    elif source_family in {"world", "place"}:
        headline = _world_question(
            topic,
            policy=policy,
            source_family=source_family,
            seed=seed,
        )
    else:
        headline = _general_question(topic, policy=policy, seed=seed)
    return AmbientDisplayImpulseCopy(
        eyebrow="",
        headline=_truncate_text(headline, max_len=112),
        body=_truncate_text(
            _helper_text(topic, policy=policy, source_family=source_family, seed=seed),
            max_len=96,
        ),
        symbol=_symbol_for_policy(policy, source_family=source_family),
        accent=_accent_for_policy(policy, source_family=source_family),
    )
