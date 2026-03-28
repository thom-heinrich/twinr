# CHANGELOG: 2026-03-27
# BUG-1: Stop emitting empty or grammatically broken copy when raw titles are blank, quoted,
#        heavily punctuated, or injected into case-sensitive German sentence frames.
# BUG-2: Make visible length limiting display-width aware (grapheme-safe when wcwidth is present)
#        so emoji/CJK/wide glyphs do not overflow a Pi display or get cut mid-cluster.
# SEC-1: Strip bidirectional override/isolate controls plus control characters from untrusted
#        mindshare text before it reaches the display surface.
# IMP-1: Replace binary two-variant branching with a larger deterministic template bank and
#        independent stable rotation for headline/body to reduce repetitive copy.
# IMP-2: Add translation hooks and Unicode-native copy so the module is ready for gettext/Babel/
#        CLDR-based localization without changing the public builder API.

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

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
import gettext
import unicodedata

from twinr.agent.personality._display_utils import (
    normalized_text as _normalized_text,
    stable_fraction as _stable_fraction,
    truncate_text as _truncate_text,
)
from twinr.agent.personality.positive_engagement import PositiveEngagementTopicPolicy
from twinr.agent.personality.self_expression import CompanionMindshareItem

try:  # Optional, tiny dependency that materially improves small-display clipping.
    from wcwidth import iter_graphemes as _iter_graphemes
    from wcwidth import wcswidth as _wcswidth
except Exception:  # pragma: no cover - optional dependency.
    _iter_graphemes = None
    _wcswidth = None


__all__ = [
    "AmbientDisplayImpulseCopy",
    "build_ambient_display_impulse_copy",
    "configure_ambient_display_translation",
]


_TRANSLATE: Callable[[str], str] = gettext.gettext

# Unicode bidi controls that can visually reorder surrounding text.
_BIDI_CONTROL_CODEPOINTS: tuple[tuple[int, int], ...] = (
    (0x202A, 0x202E),  # embeddings / overrides
    (0x2066, 0x2069),  # isolates
)
_BIDI_CONTROL_CHARS = frozenset(
    chr(codepoint)
    for start, end in _BIDI_CONTROL_CODEPOINTS
    for codepoint in range(start, end + 1)
)

# Invisible formatting characters that are not needed for this compact display copy.
# Intentionally keep ZWJ / ZWNJ because they can be semantically important in names.
_EXTRA_DISALLOWED_FORMAT_CHARS = frozenset(
    {
        "\u00ad",  # soft hyphen
        "\u200b",  # zero width space
        "\u2060",  # word joiner
        "\ufeff",  # BOM / zero width no-break space
    }
)

_MAX_TOPIC_COLS = 52
_MAX_HEADLINE_COLS = 112
_MAX_BODY_COLS = 96
_STABLE_TEXT_KEY_LIMIT = 256

_DEFAULT_TOPIC_FALLBACK = "dieses Thema"


@dataclass(frozen=True, slots=True)
class AmbientDisplayImpulseCopy:
    """Describe the user-facing copy for one reserve-lane impulse."""

    eyebrow: str = ""
    headline: str = ""
    body: str = ""
    symbol: str = "question"
    accent: str = "info"


def configure_ambient_display_translation(gettext_fn: Callable[[str], str] | None) -> None:
    """Register one translation function for all visible template strings.

    The public builder signature stays unchanged. Callers that already have a
    gettext/Babel-backed translation function can install it once at startup.
    """

    global _TRANSLATE
    _TRANSLATE = gettext_fn or gettext.gettext


def _t(message: str) -> str:
    """Translate one template string if a translator was configured."""

    return _TRANSLATE(message)


def _safe_format(template: str, /, **kwargs: str) -> str:
    """Format one translated template without letting catalog mistakes crash UI."""

    try:
        return template.format(**kwargs)
    except Exception:
        return template


def _safe_text(value: object | None) -> str:
    """Normalize one arbitrary input into compact, display-safe Unicode text."""

    raw = unicodedata.normalize("NFC", _normalized_text(value))
    if not raw:
        return ""

    cleaned: list[str] = []
    for char in raw:
        if char in _BIDI_CONTROL_CHARS or char in _EXTRA_DISALLOWED_FORMAT_CHARS:
            continue

        category = unicodedata.category(char)
        if category in {"Cc", "Cs", "Co", "Cn"}:
            if char.isspace():
                cleaned.append(" ")
            continue

        cleaned.append(" " if char.isspace() else char)

    compact = " ".join("".join(cleaned).split())
    return compact.strip()


def _display_width(text: str) -> int:
    """Return approximate occupied display columns for one string."""

    if not text:
        return 0
    if _wcswidth is None:
        return len(text)
    measured = _wcswidth(text)
    return measured if measured >= 0 else len(text)


def _graphemes(text: str) -> Iterable[str]:
    """Iterate user-perceived grapheme groups where possible."""

    if _iter_graphemes is None:
        return tuple(text)
    return _iter_graphemes(text)


def _truncate_display_text(text: str, *, max_cols: int) -> str:
    """Clip one string to the available display width.

    When wcwidth is available this respects grapheme clusters and printable
    width. Without it, we fall back to the repository's existing truncation
    helper to preserve previous runtime behavior.
    """

    text = _safe_text(text)
    if not text or max_cols <= 0:
        return ""

    if _iter_graphemes is None or _wcswidth is None:
        return _truncate_text(text, max_len=max_cols)

    if _display_width(text) <= max_cols:
        return text

    ellipsis = "…"
    budget = max(1, max_cols - _display_width(ellipsis))

    parts: list[str] = []
    used = 0
    for grapheme in _graphemes(text):
        width = _display_width(grapheme)
        if width < 0:
            width = len(grapheme)
        if used + width > budget:
            break
        parts.append(grapheme)
        used += width

    clipped = "".join(parts).rstrip(" ,;:.-")
    if not clipped:
        clipped = next(iter(_graphemes(text)), "")[:1]
    return clipped + ellipsis


def _strip_wrapping_noise(text: str) -> str:
    """Remove common outer quoting / bracket noise from upstream titles."""

    stripped = text.strip()
    stripped = stripped.strip("„“‚‘'\"`´[](){}<>")
    return stripped.rstrip(" .!?…,:;").strip()


def _source_family(source: object | None) -> str:
    """Map one raw mindshare source onto a generic copy family."""

    normalized = _safe_text(source).casefold()
    if normalized in {"continuity", "relationship"}:
        return "memory"
    if normalized == "place":
        return "place"
    if normalized in {"situational_awareness", "regional_news", "local_news", "world"}:
        return "world"
    return "general"


def _topic_phrase(item: CompanionMindshareItem) -> str:
    """Return one bounded human-readable topic phrase for direct display."""

    raw_title = _safe_text(getattr(item, "title", ""))
    topic = _strip_wrapping_noise(raw_title)
    if not topic:
        topic = _DEFAULT_TOPIC_FALLBACK
    return _truncate_display_text(topic, max_cols=_MAX_TOPIC_COLS)


def _topic_anchor(topic: str) -> str:
    """Render the display-safe quoted topic anchor used inside templates."""

    topic = _truncate_display_text(topic, max_cols=_MAX_TOPIC_COLS)
    if not topic:
        topic = _DEFAULT_TOPIC_FALLBACK
    return f"„{topic}“"


def _policy_action(policy: PositiveEngagementTopicPolicy | None) -> str:
    """Read the policy action defensively."""

    return _safe_text(getattr(policy, "action", "")).casefold()


def _attention_state(policy: PositiveEngagementTopicPolicy | None) -> str:
    """Read the policy attention state defensively."""

    return _safe_text(getattr(policy, "attention_state", "")).casefold()


def _stable_pick(
    variants: tuple[str, ...],
    *,
    salt: str,
    item: CompanionMindshareItem,
    source_family: str,
    policy: PositiveEngagementTopicPolicy,
    local_now: datetime | None,
) -> str:
    """Pick one deterministic variant from a bounded template bank."""

    if not variants:
        return ""

    title_key = _safe_text(getattr(item, "title", ""))[:_STABLE_TEXT_KEY_LIMIT]
    source_key = _safe_text(getattr(item, "source", ""))[:64]
    day_key = local_now.date().isoformat() if local_now else ""
    fraction = _stable_fraction(
        salt,
        title_key,
        source_key,
        source_family,
        _policy_action(policy),
        _attention_state(policy),
        day_key,
    )
    index = min(int(fraction * len(variants)), len(variants) - 1)
    return variants[index]


def _accent_for_policy(
    policy: PositiveEngagementTopicPolicy,
    *,
    source_family: str,
) -> str:
    """Choose one calm accent token for the reserve card."""

    if _attention_state(policy) == "shared_thread":
        return "warm"
    if source_family == "memory":
        return "warm"
    if _policy_action(policy) == "brief_update":
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
    if _policy_action(policy) in {"ask_one", "invite_follow_up"}:
        return "question"
    return "sparkles"


def _memory_headline_templates(
    *,
    action: str,
    attention_state: str,
) -> tuple[str, ...]:
    """Return context-appropriate headline templates for personal threads."""

    if attention_state == "shared_thread":
        return (
            "Bei {topic} würde ich gern den Faden wieder aufnehmen.",
            "{topic} würde ich gern mit dir weiterdenken.",
            "Zu {topic} würde ich gern noch kurz anschließen.",
        )
    if action == "hint":
        return (
            "Zu {topic} fehlt mir noch ein kleines Stück.",
            "Bei {topic} ist für mich noch etwas offen.",
            "{topic} habe ich noch nicht ganz zu Ende gedacht.",
        )
    if action == "brief_update":
        return (
            "Zu {topic} habe ich noch einen kleinen Nachtrag.",
            "{topic} ist für mich noch nicht ganz abgeschlossen.",
            "Bei {topic} hänge ich gedanklich noch kurz fest.",
        )
    if action == "ask_one":
        return (
            "Zu {topic} habe ich noch eine kleine Frage.",
            "Bei {topic} ist für mich noch ein Punkt offen.",
            "{topic} würde ich gern noch kurz mit dir klären.",
        )
    return (
        "Zu {topic} ist mir noch etwas offen geblieben.",
        "Bei {topic} habe ich noch ein kleines Fragezeichen.",
        "{topic} würde ich gern noch einmal kurz aufnehmen.",
    )


def _world_headline_templates(
    *,
    action: str,
    source_family: str,
    attention_state: str,
) -> tuple[str, ...]:
    """Return context-appropriate headline templates for world/place topics."""

    if attention_state == "shared_thread":
        return (
            "Zu {topic} würde ich gern kurz mit dir hinschauen.",
            "{topic} würde ich gern mit dir einordnen.",
            "Rund um {topic} würde ich gern kurz gemeinsam prüfen.",
        )
    if action == "hint":
        return (
            "Rund um {topic} schaue ich heute noch einmal hin.",
            "Zu {topic} bleibe ich heute kurz dran.",
            "{topic} möchte ich heute noch einmal prüfen.",
        )
    if action == "brief_update":
        return (
            "Rund um {topic} bewegt sich heute etwas.",
            "Bei {topic} ist heute etwas in Bewegung.",
            "Zu {topic} gibt es wohl gerade einen kleinen Nachtrag.",
        )
    if source_family == "world":
        return (
            "Rund um {topic} ist heute wieder Bewegung.",
            "{topic} lässt mich heute nicht ganz los.",
            "Zu {topic} lohnt heute ein zweiter Blick.",
        )
    return (
        "Rund um {topic} lohnt heute ein zweiter Blick.",
        "{topic} schaue ich mir heute noch einmal an.",
        "Zu {topic} bleibe ich heute kurz aufmerksam.",
    )


def _general_headline_templates(
    *,
    action: str,
    attention_state: str,
) -> tuple[str, ...]:
    """Return generic headline templates that still stay concrete."""

    if attention_state == "shared_thread":
        return (
            "Zu {topic} würde ich gern kurz mit dir weiterdenken.",
            "{topic} würde ich gern kurz zusammen einordnen.",
            "Bei {topic} würde ich gern noch kurz anschließen.",
        )
    if action == "brief_update":
        return (
            "Zu {topic} wäre später ein kurzer zweiter Blick gut.",
            "Bei {topic} habe ich noch einen offenen Punkt.",
            "{topic} würde ich später gern noch einmal ansehen.",
        )
    if action == "hint":
        return (
            "Zu {topic} habe ich noch einen kleinen Punkt im Kopf.",
            "Bei {topic} ist mir noch etwas offen geblieben.",
            "{topic} würde ich gern noch kurz sortieren.",
        )
    return (
        "Zu {topic} ist mir noch etwas offen geblieben.",
        "Bei {topic} habe ich noch ein kleines Fragezeichen.",
        "{topic} würde ich gern noch kurz einordnen.",
    )


def _call_to_action_templates(
    *,
    action: str,
    attention_state: str,
    source_family: str,
) -> tuple[str, ...]:
    """Return short, low-pressure CTA templates under the explanatory line."""

    if source_family == "memory":
        if action == "brief_update":
            return (
                "Wenn du magst, reden wir später kurz darüber.",
                "Magst du mir später zwei Sätze dazu sagen?",
                "Sag mir später gern kurz mehr dazu.",
            )
        if action == "hint":
            return (
                "Wenn du magst, sagen wir kurz etwas dazu.",
                "Magst du mir kurz den Stand sagen?",
                "Lass uns das kurz zusammen ansehen.",
            )
        return (
            "Magst du mich kurz auf den Stand bringen?",
            "Wenn du magst, reden wir kurz darüber.",
            "Sag mir gern kurz, was wichtig ist.",
        )

    if action == "brief_update":
        return (
            "Soll ich dir später kurz sagen, was ich dazu sehe?",
            "Wenn du magst, schauen wir später noch einmal hin.",
            "Lass uns später kurz darauf zurückkommen.",
        )
    if action == "hint":
        return (
            "Magst du kurz sagen, was daran wichtig ist?",
            "Wenn du magst, schauen wir kurz gemeinsam hin.",
            "Lass uns das kurz einordnen.",
        )
    if attention_state == "shared_thread":
        return (
            "Wenn du magst, reden wir kurz darüber.",
            "Was ist dir daran gerade wichtig?",
            "Sollen wir den Punkt kurz zusammen ansehen?",
        )
    return (
        "Was ist dir daran wichtig?",
        "Magst du kurz etwas dazu sagen?",
        "Sollen wir kurz darüber sprechen?",
    )


def _headline_for_context(
    topic_anchor: str,
    *,
    policy: PositiveEngagementTopicPolicy,
    source_family: str,
    item: CompanionMindshareItem,
    local_now: datetime | None,
) -> str:
    """Render one bounded explanatory headline for the reserve card."""

    action = _policy_action(policy)
    attention_state = _attention_state(policy)

    if source_family == "memory":
        templates = _memory_headline_templates(
            action=action,
            attention_state=attention_state,
        )
    elif source_family in {"world", "place"}:
        templates = _world_headline_templates(
            action=action,
            source_family=source_family,
            attention_state=attention_state,
        )
    else:
        templates = _general_headline_templates(
            action=action,
            attention_state=attention_state,
        )

    template = _stable_pick(
        templates,
        salt="headline",
        item=item,
        source_family=source_family,
        policy=policy,
        local_now=local_now,
    )
    return _safe_format(_t(template), topic=topic_anchor)


def _body_for_context(
    *,
    policy: PositiveEngagementTopicPolicy,
    source_family: str,
    item: CompanionMindshareItem,
    local_now: datetime | None,
) -> str:
    """Render one short CTA line under the explanatory headline."""

    templates = _call_to_action_templates(
        action=_policy_action(policy),
        attention_state=_attention_state(policy),
        source_family=source_family,
    )
    template = _stable_pick(
        templates,
        salt="body",
        item=item,
        source_family=source_family,
        policy=policy,
        local_now=local_now,
    )
    return _safe_format(_t(template))


def build_ambient_display_impulse_copy(
    item: CompanionMindshareItem,
    policy: PositiveEngagementTopicPolicy,
    *,
    local_now: datetime | None,
) -> AmbientDisplayImpulseCopy:
    """Build one statement-first display impulse from current companion state."""

    source_family = _source_family(getattr(item, "source", None))
    topic = _topic_phrase(item)
    topic_anchor = _topic_anchor(topic)

    headline = _headline_for_context(
        topic_anchor,
        policy=policy,
        source_family=source_family,
        item=item,
        local_now=local_now,
    )
    body = _body_for_context(
        policy=policy,
        source_family=source_family,
        item=item,
        local_now=local_now,
    )

    return AmbientDisplayImpulseCopy(
        eyebrow="",
        headline=_truncate_display_text(headline, max_cols=_MAX_HEADLINE_COLS),
        body=_truncate_display_text(body, max_cols=_MAX_BODY_COLS),
        symbol=_symbol_for_policy(policy, source_family=source_family),
        accent=_accent_for_policy(policy, source_family=source_family),
    )