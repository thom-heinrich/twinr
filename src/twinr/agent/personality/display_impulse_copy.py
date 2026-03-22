"""Build question-first ambient display copy from Twinr's companion state.

The display reserve should feel like a living conversational opening, not like
an internal topic label. This module therefore turns structured mindshare items
plus positive-engagement policy into short, readable prompts that can either
invite the user's opinion or ask for a little more context so Twinr can
remember a personal thread better.

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
            f"Soll ich mir zu {topic} etwas genauer merken?",
            f"Ist {topic} etwas, das ich weiter gut im Kopf behalten soll?",
        )
    elif policy.action == "brief_update":
        variants = (
            f"Willst du mir zu {topic} noch ein kleines Update geben?",
            f"Magst du mir kurz sagen, wie es bei {topic} steht?",
        )
    elif policy.action == "ask_one":
        variants = (
            f"Magst du mir zu {topic} noch etwas erzählen?",
            f"Wie ist es bei {topic} weitergegangen?",
        )
    else:
        variants = (
            f"Wie ist es bei {topic} inzwischen weitergegangen?",
            f"Magst du mir zu {topic} noch etwas mehr erzählen?",
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
            f"Soll ich {topic} weiter im Blick behalten?",
            f"Willst du, dass ich bei {topic} weiter aufpasse?",
        )
    elif policy.action == "brief_update":
        variants = (
            f"Soll ich dir spaeter kurz sagen, was sich bei {topic} getan hat?",
            f"Willst du spaeter ein kleines Update zu {topic}?",
        )
    elif source_family == "world":
        variants = (
            f"Ich habe zu {topic} heute etwas gelesen. Was meinst du?",
            f"Bei {topic} ist gerade wieder Bewegung drin. Wie siehst du das?",
        )
    else:
        variants = (
            f"Was meinst du zu {topic}?",
            f"Wie siehst du {topic} gerade?",
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
            f"Willst du spaeter kurz auf {topic} schauen?",
            f"Soll ich zu {topic} spaeter kurz nachhaken?",
        )
    else:
        variants = (
            f"Was meinst du zu {topic}?",
            f"Wie siehst du {topic} gerade?",
        )
    return variants[1 if seed >= 0.5 else 0]


def _helper_text(
    *,
    policy: PositiveEngagementTopicPolicy,
    source_family: str,
    seed: float,
) -> str:
    """Render one short supportive helper line under the main question."""

    if source_family == "memory":
        variants = (
            "Dann kann ich mir den Faden besser merken.",
            "Dann halte ich mir das genauer fest.",
        )
        return variants[1 if seed >= 0.5 else 0]
    if policy.action == "hint":
        variants = (
            "Ein kurzes Ja oder Nein reicht mir schon.",
            "Ein kleiner Hinweis genuegt mir.",
        )
        return variants[1 if seed >= 0.5 else 0]
    if policy.attention_state == "shared_thread":
        variants = (
            "Dann halte ich den Faden fuer uns warm.",
            "Dann weiss ich besser, ob ich dranbleiben soll.",
        )
        return variants[1 if seed >= 0.5 else 0]
    variants = (
        "Dann weiss ich, ob ich weiter darauf achten soll.",
        "Dann kann ich mich daran besser orientieren.",
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
            _helper_text(policy=policy, source_family=source_family, seed=seed),
            max_len=96,
        ),
        symbol=_symbol_for_policy(policy, source_family=source_family),
        accent=_accent_for_policy(policy, source_family=source_family),
    )
