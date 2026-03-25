"""Build personality-shaped ambient display copy from Twinr's companion state.

The display reserve should feel like a living conversational opening, not like
an internal topic label. This module therefore turns structured mindshare items
plus positive-engagement policy into short, readable prompts that carry a clear
topic anchor and at least a little of Twinr's own tone. The visible contract is
explicit: the large headline should stand on its own as the explanatory
statement, while the smaller second line acts as the call to action.

The logic stays generic and policy-driven:

- no named-topic hardcoding
- no regex topic parsing
- no hidden prompt text in the runtime loop
- deterministic bounded variation from stable item keys and the local day
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from twinr.agent.personality._display_utils import (
    normalized_text as _normalized_text,
    stable_fraction as _stable_fraction,
    truncate_text as _truncate_text,
)
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


def _memory_statement(
    topic: str,
    *,
    policy: PositiveEngagementTopicPolicy,
    seed: float,
) -> str:
    """Render one personal-thread statement for the main reserve headline."""

    if policy.action == "hint":
        variants = (
            f"Bei {topic} ist fuer mich noch etwas offen.",
            f"Zu {topic} fehlt mir noch ein kleines Stueck.",
        )
    elif policy.action == "brief_update":
        variants = (
            f"Zu {topic} habe ich noch einen kleinen Nachtrag im Kopf.",
            f"Bei {topic} bin ich gedanklich noch nicht ganz fertig.",
        )
    elif policy.action == "ask_one":
        variants = (
            f"Zu {topic} fehlt mir noch etwas.",
            f"Bei {topic} habe ich noch ein kleines Fragezeichen.",
        )
    else:
        variants = (
            f"Bei {topic} habe ich noch ein kleines Fragezeichen.",
            f"Zu {topic} fehlt mir noch ein Stueck.",
        )
    return variants[1 if seed >= 0.5 else 0]


def _world_statement(
    topic: str,
    *,
    policy: PositiveEngagementTopicPolicy,
    source_family: str,
    seed: float,
) -> str:
    """Render one world/place-style statement for the main reserve headline."""

    if policy.action == "hint":
        variants = (
            f"Bei {topic} bleibe ich heute kurz dran.",
            f"Zu {topic} schaue ich gerade noch einmal hin.",
        )
    elif policy.action == "brief_update":
        variants = (
            f"Bei {topic} tut sich gerade etwas.",
            f"Rund um {topic} ist heute etwas in Bewegung.",
        )
    elif source_family == "world":
        variants = (
            f"Bei {topic} ist gerade wieder Bewegung drin.",
            f"{topic} laesst mich heute nicht ganz los.",
        )
    else:
        variants = (
            f"Rund um {topic} ist gerade etwas in Bewegung.",
            f"Bei {topic} lohnt heute ein zweiter Blick.",
        )
    return variants[1 if seed >= 0.5 else 0]


def _general_statement(
    topic: str,
    *,
    policy: PositiveEngagementTopicPolicy,
    seed: float,
) -> str:
    """Render one generic but still concrete statement for the main headline."""

    if policy.action == "brief_update":
        variants = (
            f"Zu {topic} waere spaeter ein kleiner zweiter Blick gut.",
            f"Bei {topic} haenge ich gedanklich noch kurz dran.",
        )
    else:
        variants = (
            f"Bei {topic} habe ich noch ein kleines Fragezeichen.",
            f"Zu {topic} ist mir noch etwas offen geblieben.",
        )
    return variants[1 if seed >= 0.5 else 0]


def _call_to_action(
    topic: str,
    *,
    policy: PositiveEngagementTopicPolicy,
    source_family: str,
    seed: float,
) -> str:
    """Render one short CTA line under the explanatory headline."""

    if source_family == "memory":
        if policy.action == "brief_update":
            variants = (
                "Wollen wir spaeter kurz darueber reden?",
                "Magst du spaeter kurz mehr dazu sagen?",
            )
        elif policy.action == "hint":
            variants = (
                "Magst du mir kurz etwas dazu sagen?",
                "Wollen wir kurz darueber reden?",
            )
        else:
            variants = (
                "Magst du mich kurz auf Stand bringen?",
                "Wollen wir kurz darueber reden?",
            )
        return variants[1 if seed >= 0.5 else 0]
    if policy.action == "brief_update":
        variants = (
            "Soll ich dir spaeter kurz mehr dazu sagen?",
            "Wollen wir spaeter kurz darueber reden?",
        )
        return variants[1 if seed >= 0.5 else 0]
    if policy.action == "hint":
        variants = (
            "Magst du kurz was dazu sagen?",
            "Wollen wir kurz darueber reden?",
        )
        return variants[1 if seed >= 0.5 else 0]
    if policy.attention_state == "shared_thread":
        variants = (
            "Wollen wir kurz darueber reden?",
            "Was meinst du dazu?",
        )
        return variants[1 if seed >= 0.5 else 0]
    variants = (
        "Was meinst du dazu?",
        "Magst du kurz was dazu sagen?",
    )
    return variants[1 if seed >= 0.5 else 0]


def build_ambient_display_impulse_copy(
    item: CompanionMindshareItem,
    policy: PositiveEngagementTopicPolicy,
    *,
    local_now: datetime | None,
) -> AmbientDisplayImpulseCopy:
    """Build one statement-first display impulse from current companion state."""

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
        headline = _memory_statement(topic, policy=policy, seed=seed)
    elif source_family in {"world", "place"}:
        headline = _world_statement(
            topic,
            policy=policy,
            source_family=source_family,
            seed=seed,
        )
    else:
        headline = _general_statement(topic, policy=policy, seed=seed)
    return AmbientDisplayImpulseCopy(
        eyebrow="",
        headline=_truncate_text(headline, max_len=112),
        body=_truncate_text(
            _call_to_action(topic, policy=policy, source_family=source_family, seed=seed),
            max_len=96,
        ),
        symbol=_symbol_for_policy(policy, source_family=source_family),
        accent=_accent_for_policy(policy, source_family=source_family),
    )
