"""Turn user-discovery invites into reserve-lane candidate cards."""

# CHANGELOG: 2026-03-29
# BUG-1: max_items is now enforced; max_items <= 0 no longer leaks a candidate card.
# BUG-2: review-stage discovery cards now get review-specific anchors and copy instead of silently falling back to opener phrasing.
# BUG-3: malformed invite fields and non-finite salience no longer crash candidate generation or poison ranking inputs.
# SEC-1: all user-adjacent display/prompt text is normalized to NFKC and stripped of control/invisible formatting characters before propagation.
# SEC-2: topic keys, labels, and prompt payloads are length-bounded and canonicalized to prevent key pollution and oversized UI/prompt payloads.
# IMP-1: invite data is snapshotted into a typed, immutable internal shape before card generation.
# IMP-2: invite loading now fails closed with logging so a single bad invite cannot take down the ambient display path.

from __future__ import annotations

import logging
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal, TypedDict

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.memory.user_discovery import UserDiscoveryService

LOGGER = logging.getLogger(__name__)

PromptStage = Literal["opener", "follow_up", "review"]


class CardIntent(TypedDict):
    topic_semantics: str
    statement_intent: str
    cta_intent: str
    relationship_stance: str


class GenerationContext(TypedDict):
    candidate_family: str
    display_goal: str
    invite_kind: str
    phase: str
    topic_id: str
    topic_label: str
    display_label: str
    session_minutes: int
    display_prompt_stage: PromptStage
    display_anchor: str
    hook_hint: str
    topic_summary: str
    card_intent: CardIntent
    raw_invite_headline: str
    raw_invite_body: str


@dataclass(frozen=True, slots=True)
class _InviteSnapshot:
    invite_kind: str
    phase: str
    topic_id: str
    topic_label: str
    display_topic_label: str
    session_minutes: int
    display_prompt_stage: PromptStage
    salience: float
    headline: str
    body: str
    reason: str


_TITLE_MAX_LEN: Final[int] = 72
_LABEL_MAX_LEN: Final[int] = 96
_HEADLINE_MAX_LEN: Final[int] = 112
_BODY_MAX_LEN: Final[int] = 112
_REASON_MAX_LEN: Final[int] = 120
_ANCHOR_MAX_LEN: Final[int] = 96
_HOOK_MAX_LEN: Final[int] = 160
_INTENT_MAX_LEN: Final[int] = 160
_TOKEN_MAX_LEN: Final[int] = 48
_SESSION_MINUTES_MAX: Final[int] = 24 * 60

_TOPIC_PROMPT_ANCHORS: Final[dict[str, dict[PromptStage, str]]] = {
    "basics": {
        "opener": "dein Name",
        "follow_up": "ein paar Grundlinien aus deinem Alltag",
        "review": "was ich zu deinem Namen und deiner Anrede schon verstanden habe",
    },
    "companion_style": {
        "opener": "der Ton zwischen uns",
        "follow_up": "wie sich unser Ton gut anfuehlt",
        "review": "was ich ueber den Ton zwischen uns verstanden habe",
    },
    "social": {
        "opener": "die Menschen, die dir wichtig sind",
        "follow_up": "dein Umfeld",
        "review": "was ich ueber dein Umfeld verstanden habe",
    },
    "interests": {
        "opener": "deine Interessen",
        "follow_up": "das, was dich wirklich packt",
        "review": "was ich ueber deine Interessen verstanden habe",
    },
    "hobbies": {
        "opener": "deine Hobbys",
        "follow_up": "das, was dir Freude macht",
        "review": "was ich ueber das, was dir Freude macht, verstanden habe",
    },
    "routines": {
        "opener": "dein Alltag",
        "follow_up": "dein Tagesrhythmus",
        "review": "was ich ueber deinen Tagesrhythmus verstanden habe",
    },
    "pets": {
        "opener": "Tiere in deinem Alltag",
        "follow_up": "die Rolle von Tieren in deinem Leben",
        "review": "was ich ueber Tiere in deinem Alltag verstanden habe",
    },
    "no_goes": {
        "opener": "Dinge, die ich lieber lasse",
        "follow_up": "Grenzen, die ich besser spueren sollte",
        "review": "was ich ueber deine Grenzen verstanden habe",
    },
    "health": {
        "opener": "dein Umgang mit Gesundheitsthemen",
        "follow_up": "was ich bei Gesundheitsthemen beachten sollte",
        "review": "was ich ueber deinen Umgang mit Gesundheitsthemen verstanden habe",
    },
}

_TOPIC_PROMPT_SENTENCES: Final[dict[str, dict[PromptStage, str]]] = {
    "basics": {
        "opener": "Ich moechte wissen, wie ich dich ansprechen soll.",
        "follow_up": "Ich habe schon etwas von dir gelernt und moechte das Bild noch runder machen.",
        "review": "Ich moechte kurz abgleichen, ob ich deinen Namen und deine Anrede noch richtig im Kopf habe.",
    },
    "companion_style": {
        "opener": "Ich moechte wissen, wie ich mit dir reden soll.",
        "follow_up": "Ich glaube, ich kann noch feiner verstehen, was sich im Ton zwischen uns gut anfuehlt.",
        "review": "Ich moechte kurz abgleichen, ob mein Ton fuer dich noch passt.",
    },
    "social": {
        "opener": "Ich moechte wissen, wer dir wichtig ist.",
        "follow_up": "Bei deinem Umfeld habe ich noch einen kleinen offenen Faden.",
        "review": "Ich moechte kurz abgleichen, ob ich dein Umfeld noch richtig im Blick habe.",
    },
    "interests": {
        "opener": "Ich moechte besser verstehen, was dich wirklich interessiert.",
        "follow_up": "Ich glaube, bei deinen Interessen fehlt mir noch ein spannender Winkel.",
        "review": "Ich moechte kurz abgleichen, ob ich deine Interessen noch richtig verstanden habe.",
    },
    "hobbies": {
        "opener": "Ich moechte wissen, was du gern machst.",
        "follow_up": "Ich wuerde gern noch besser verstehen, was dir im Alltag Freude macht.",
        "review": "Ich moechte kurz abgleichen, was dir im Alltag noch immer Freude macht.",
    },
    "routines": {
        "opener": "Ich moechte verstehen, wie dein Alltag meistens aussieht.",
        "follow_up": "Bei deinem Tagesrhythmus fehlt mir, glaube ich, noch ein kleines Stueck.",
        "review": "Ich moechte kurz abgleichen, ob ich deinen Tagesrhythmus noch richtig verstehe.",
    },
    "pets": {
        "opener": "Ich moechte wissen, ob Tiere in deinem Alltag eine Rolle spielen.",
        "follow_up": "Bei Tieren in deinem Alltag habe ich noch einen kleinen offenen Faden.",
        "review": "Ich moechte kurz abgleichen, ob Tiere in deinem Alltag noch eine Rolle spielen.",
    },
    "no_goes": {
        "opener": "Ich moechte wissen, was ich lieber lassen soll.",
        "follow_up": "Ich glaube, ich kann noch feiner verstehen, was ich besser bleiben lasse.",
        "review": "Ich moechte kurz abgleichen, ob ich deine Grenzen noch richtig beachte.",
    },
    "health": {
        "opener": "Ich moechte wissen, wie vorsichtig ich bei Gesundheitsthemen sein soll.",
        "follow_up": "Ich moechte vorsichtig besser verstehen, was ich bei Gesundheitsthemen beachten soll.",
        "review": "Ich moechte kurz abgleichen, ob ich bei Gesundheitsthemen noch passend vorsichtig bin.",
    },
}

_TOPIC_CARD_INTENTS: Final[dict[str, CardIntent]] = {
    "basics": {
        "topic_semantics": "bevorzugte Anrede und Namensform",
        "statement_intent": "Twinr will wissen, wie es den Nutzer ansprechen soll.",
        "cta_intent": "Den Nutzer bitten, den passenden Namen oder die passende Anrede zu nennen.",
        "relationship_stance": "ruhiges Kennenlernen ohne Setup-Ton",
    },
    "companion_style": {
        "topic_semantics": "bevorzugter Ton im Gespraech",
        "statement_intent": "Twinr will verstehen, wie es mit dem Nutzer reden soll.",
        "cta_intent": "Den Nutzer nach dem passenden Ton oder Umgang fragen.",
        "relationship_stance": "aufmerksames Abstimmen ohne Formalitaetsjargon",
    },
    "social": {
        "topic_semantics": "wichtige Menschen im Leben des Nutzers",
        "statement_intent": "Twinr will wissen, wer dem Nutzer wichtig ist.",
        "cta_intent": "Den Nutzer einladen, kurz von wichtigen Menschen zu erzaehlen.",
        "relationship_stance": "warm und persoenlich, aber nicht aufdringlich",
    },
    "interests": {
        "topic_semantics": "echte Interessen des Nutzers",
        "statement_intent": "Twinr moechte besser verstehen, was den Nutzer wirklich interessiert.",
        "cta_intent": "Den Nutzer bitten, ein Thema oder Interesse kurz zu nennen.",
        "relationship_stance": "neugierig, alltagsnah und ohne Profiling-Sprache",
    },
    "hobbies": {
        "topic_semantics": "Dinge, die der Nutzer gern macht",
        "statement_intent": "Twinr moechte wissen, was der Nutzer gern macht.",
        "cta_intent": "Den Nutzer einladen, von einer liebsten Beschaeftigung zu erzaehlen.",
        "relationship_stance": "locker und interessiert",
    },
    "routines": {
        "topic_semantics": "typischer Alltag des Nutzers",
        "statement_intent": "Twinr moechte verstehen, wie der Alltag des Nutzers meistens aussieht.",
        "cta_intent": "Den Nutzer nach einem typischen Tagesmuster oder einer Gewohnheit fragen.",
        "relationship_stance": "ruhiges Kennenlernen des Alltags",
    },
    "pets": {
        "topic_semantics": "Tiere im Alltag des Nutzers",
        "statement_intent": "Twinr moechte wissen, ob Tiere im Alltag des Nutzers eine Rolle spielen.",
        "cta_intent": "Den Nutzer bitten, kurz von einem Tier oder dessen Rolle zu erzaehlen.",
        "relationship_stance": "freundlich und alltagsnah",
    },
    "no_goes": {
        "topic_semantics": "Dinge, die Twinr lieber lassen soll",
        "statement_intent": "Twinr will wissen, was es beim Nutzer lieber lassen soll.",
        "cta_intent": "Den Nutzer einladen, eine Grenze oder ein No-Go kurz zu nennen.",
        "relationship_stance": "respektvoll und entlastend",
    },
    "health": {
        "topic_semantics": "vorsichtiger Umgang mit Gesundheitsthemen",
        "statement_intent": "Twinr will wissen, wie vorsichtig es bei Gesundheitsthemen sein soll.",
        "cta_intent": "Den Nutzer bitten, kurz zu sagen, wie vorsichtig Twinr dabei sein soll.",
        "relationship_stance": "behutsam und ruhig",
    },
}

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")
_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")


def _first_nonempty(*values: str, default: str = "") -> str:
    for value in values:
        if value:
            return value
    return default


def _sanitize_text(value: object | None, *, max_len: int) -> str:
    text = "" if value is None else str(value)
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    cleaned: list[str] = []
    for char in text:
        if char.isspace():
            cleaned.append(" ")
            continue

        category = unicodedata.category(char)
        if category.startswith("C"):
            continue

        cleaned.append(char)

    compact = _WHITESPACE_RE.sub(" ", "".join(cleaned)).strip()
    if len(compact) <= max_len:
        return compact
    if max_len <= 3:
        return compact[:max_len]
    return compact[: max_len - 3].rstrip() + "..."


def _slug_token(value: object | None, *, fallback: str, max_len: int = _TOKEN_MAX_LEN) -> str:
    text = _sanitize_text(value, max_len=max_len * 8)
    if not text:
        return fallback

    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.encode("ascii", "ignore").decode("ascii").casefold()
    text = _TOKEN_RE.sub("_", text).strip("_")
    if not text:
        return fallback
    return text[:max_len].rstrip("_") or fallback


def _safe_int(
    value: object | None,
    *,
    default: int = 0,
    min_value: int = 0,
    max_value: int | None = None,
) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default
    if parsed < min_value:
        return min_value
    if max_value is not None and parsed > max_value:
        return max_value
    return parsed


def _safe_salience(value: object | None, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(parsed):
        return default
    return max(0.0, parsed)


def _prompt_stage(value: object | None) -> PromptStage:
    stage = _slug_token(value, fallback="opener", max_len=24)
    if stage in {"opener", "follow_up", "review"}:
        return stage
    return "opener"


def _snapshot_invite(invite: object) -> _InviteSnapshot:
    return _InviteSnapshot(
        invite_kind=_slug_token(getattr(invite, "invite_kind", None), fallback="topic_invite", max_len=32),
        phase=_slug_token(getattr(invite, "phase", None), fallback="unknown_phase", max_len=32),
        topic_id=_slug_token(getattr(invite, "topic_id", None), fallback="", max_len=32),
        topic_label=_sanitize_text(getattr(invite, "topic_label", None), max_len=_LABEL_MAX_LEN),
        display_topic_label=_sanitize_text(getattr(invite, "display_topic_label", None), max_len=_TITLE_MAX_LEN),
        session_minutes=_safe_int(
            getattr(invite, "session_minutes", None),
            default=0,
            min_value=0,
            max_value=_SESSION_MINUTES_MAX,
        ),
        display_prompt_stage=_prompt_stage(getattr(invite, "display_prompt_stage", None)),
        salience=_safe_salience(getattr(invite, "salience", None)),
        headline=_sanitize_text(getattr(invite, "headline", None), max_len=_HEADLINE_MAX_LEN),
        body=_sanitize_text(getattr(invite, "body", None), max_len=_BODY_MAX_LEN),
        reason=_sanitize_text(getattr(invite, "reason", None), max_len=_REASON_MAX_LEN),
    )


def _prompt_anchor(invite: _InviteSnapshot) -> str:
    """Return one human-facing anchor for prompt-time discovery cards."""

    if invite.invite_kind == "review_profile":
        return "was ich schon ueber dich gelernt habe"

    anchor_options = _TOPIC_PROMPT_ANCHORS.get(invite.topic_id, {})
    anchor = anchor_options.get(invite.display_prompt_stage) or anchor_options.get("opener")
    if anchor:
        return _sanitize_text(anchor, max_len=_ANCHOR_MAX_LEN)

    return _sanitize_text(
        _first_nonempty(
            invite.display_topic_label,
            invite.topic_label,
            default="etwas Persoenliches ueber dich",
        ),
        max_len=_ANCHOR_MAX_LEN,
    )


def _prompt_hook_hint(invite: _InviteSnapshot, *, anchor: str) -> str:
    """Return one user-facing hook that avoids setup labels and UI wording."""

    if invite.invite_kind == "review_profile":
        return _sanitize_text(
            "Ich moechte mit dir abgleichen, was ich schon ueber dich weiss und was davon noch passt.",
            max_len=_HOOK_MAX_LEN,
        )

    sentence_options = _TOPIC_PROMPT_SENTENCES.get(invite.topic_id, {})
    sentence = sentence_options.get(invite.display_prompt_stage) or sentence_options.get("opener")
    if sentence:
        return _sanitize_text(sentence, max_len=_HOOK_MAX_LEN)

    if invite.display_prompt_stage == "review":
        return _sanitize_text(
            f"Ich moechte kurz abgleichen, was ich zu {anchor} schon verstanden habe.",
            max_len=_HOOK_MAX_LEN,
        )

    return _sanitize_text(f"Ich moechte noch besser verstehen, {anchor}.", max_len=_HOOK_MAX_LEN)


def _card_intent(invite: _InviteSnapshot, *, anchor: str) -> CardIntent:
    """Return structured semantic card intent for one discovery invite."""

    if invite.invite_kind == "review_profile":
        return {
            "topic_semantics": "gemeinsamer Abgleich von bereits Gelerntem",
            "statement_intent": "Twinr will mit dem Nutzer abgleichen, was es schon ueber ihn gelernt hat.",
            "cta_intent": "Den Nutzer einladen, Gelerntes zu bestaetigen, zu korrigieren oder zu aktualisieren.",
            "relationship_stance": "ruhiger Rueckblick statt Profilpflege-Sprache",
        }

    base_intent = _TOPIC_CARD_INTENTS.get(invite.topic_id)
    if base_intent is None:
        topic_semantics = _sanitize_text(anchor, max_len=80) or "persoenliches Kennenlernen"
        base_intent = {
            "topic_semantics": topic_semantics,
            "statement_intent": _sanitize_text(
                f"Twinr will den Nutzer besser verstehen, besonders bei {topic_semantics}.",
                max_len=_INTENT_MAX_LEN,
            ),
            "cta_intent": "Den Nutzer zu einer kurzen persoenlichen Antwort einladen.",
            "relationship_stance": "ruhiges Kennenlernen ohne Meta-Sprache",
        }

    if invite.display_prompt_stage == "review":
        topic_semantics = _sanitize_text(base_intent["topic_semantics"], max_len=80) or "persoenliches Kennenlernen"
        return {
            "topic_semantics": topic_semantics,
            "statement_intent": _sanitize_text(
                f"Twinr will bestaetigen, ob sein bisheriges Verstaendnis rund um {topic_semantics} noch stimmt.",
                max_len=_INTENT_MAX_LEN,
            ),
            "cta_intent": "Den Nutzer einladen, kurz zu bestaetigen, zu korrigieren oder zu aktualisieren.",
            "relationship_stance": "ruhiger Abgleich statt erneuter Erstfrage",
        }

    return {
        "topic_semantics": _sanitize_text(base_intent["topic_semantics"], max_len=80),
        "statement_intent": _sanitize_text(base_intent["statement_intent"], max_len=_INTENT_MAX_LEN),
        "cta_intent": _sanitize_text(base_intent["cta_intent"], max_len=_INTENT_MAX_LEN),
        "relationship_stance": _sanitize_text(base_intent["relationship_stance"], max_len=80),
    }


def _topic_key_topic_token(invite: _InviteSnapshot, *, prompt_anchor: str) -> str:
    if invite.invite_kind == "review_profile":
        return "review_profile"
    if invite.topic_id:
        return invite.topic_id

    return _slug_token(
        _first_nonempty(invite.display_topic_label, invite.topic_label, prompt_anchor, default="unknown_topic"),
        fallback="unknown_topic",
        max_len=_TOKEN_MAX_LEN,
    )


def _build_generation_context(
    invite: _InviteSnapshot,
    *,
    title: str,
    prompt_anchor: str,
    prompt_hook_hint: str,
    card_intent: CardIntent,
) -> GenerationContext:
    topic_label = _sanitize_text(
        _first_nonempty(invite.topic_label, title, prompt_anchor),
        max_len=_LABEL_MAX_LEN,
    )
    return {
        "candidate_family": "user_discovery",
        "display_goal": "invite_user_discovery",
        "invite_kind": invite.invite_kind,
        "phase": invite.phase,
        "topic_id": invite.topic_id or _slug_token(topic_label, fallback="unknown_topic", max_len=_TOKEN_MAX_LEN),
        "topic_label": topic_label,
        "display_label": title,
        "session_minutes": invite.session_minutes,
        "display_prompt_stage": invite.display_prompt_stage,
        "display_anchor": prompt_anchor,
        "hook_hint": prompt_hook_hint,
        "topic_summary": prompt_hook_hint,
        "card_intent": card_intent,
        # BREAKING: raw_invite_* is now sanitized, NFKC-normalized single-line text rather than byte-preserving upstream text.
        "raw_invite_headline": invite.headline,
        "raw_invite_body": invite.body,
    }


def load_display_reserve_user_discovery_candidates(
    config: TwinrConfig,
    *,
    local_now: datetime,
    max_items: int,
) -> tuple[AmbientDisplayImpulseCandidate, ...]:
    """Expose up to one due get-to-know-you invitation candidate."""

    if _safe_int(max_items, default=0, min_value=0) <= 0:
        return ()

    try:
        raw_invite = UserDiscoveryService.from_config(config).build_invitation(now=local_now)
    except Exception:
        LOGGER.exception("Failed to build user-discovery invitation candidate.")
        return ()

    if raw_invite is None:
        return ()

    try:
        invite = _snapshot_invite(raw_invite)
        prompt_anchor = _prompt_anchor(invite)
        prompt_hook_hint = _prompt_hook_hint(invite, anchor=prompt_anchor)
        card_intent = _card_intent(invite, anchor=prompt_anchor)

        title = _sanitize_text(
            _first_nonempty(
                invite.display_topic_label,
                invite.topic_label,
                prompt_anchor,
                default="Kennenlernen",
            ),
            max_len=_TITLE_MAX_LEN,
        )
        headline = _sanitize_text(
            _first_nonempty(invite.headline, prompt_hook_hint, title),
            max_len=_HEADLINE_MAX_LEN,
        )
        body = _sanitize_text(
            _first_nonempty(invite.body, prompt_hook_hint),
            max_len=_BODY_MAX_LEN,
        )
        reason = _sanitize_text(
            _first_nonempty(invite.reason, prompt_hook_hint),
            max_len=_REASON_MAX_LEN,
        )
        topic_token = _topic_key_topic_token(invite, prompt_anchor=prompt_anchor)

        # BREAKING: topic_key is now canonicalized and slugged.
        # This intentionally invalidates older unsafe cache keys and prevents collisions/key pollution from malformed topic metadata.
        topic_key = f"user_discovery:{invite.phase}:{topic_token}"

        candidate = AmbientDisplayImpulseCandidate(
            topic_key=topic_key,
            title=title,
            source="user_discovery",
            action="ask_one",
            attention_state="forming" if invite.phase == "initial_setup" else "growing",
            salience=invite.salience,
            eyebrow="",
            headline=headline,
            body=body,
            symbol="question",
            accent="warm",
            reason=reason,
            candidate_family="user_discovery",
            generation_context=_build_generation_context(
                invite,
                title=title,
                prompt_anchor=prompt_anchor,
                prompt_hook_hint=prompt_hook_hint,
                card_intent=card_intent,
            ),
        )
    except Exception:
        LOGGER.exception("Failed to convert user-discovery invitation into an ambient display candidate.")
        return ()

    return (candidate,)


__all__ = ["load_display_reserve_user_discovery_candidates"]    