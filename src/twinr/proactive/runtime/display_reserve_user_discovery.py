"""Turn user-discovery invites into reserve-lane candidate cards."""

from __future__ import annotations

from datetime import datetime

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.memory.user_discovery import UserDiscoveryService

_TOPIC_PROMPT_ANCHORS = {
    "basics": {
        "opener": "dein Name",
        "follow_up": "ein paar Grundlinien aus deinem Alltag",
    },
    "companion_style": {
        "opener": "der Ton zwischen uns",
        "follow_up": "wie sich unser Ton gut anfuehlt",
    },
    "social": {
        "opener": "die Menschen, die dir wichtig sind",
        "follow_up": "dein Umfeld",
    },
    "interests": {
        "opener": "deine Interessen",
        "follow_up": "das, was dich wirklich packt",
    },
    "hobbies": {
        "opener": "deine Hobbys",
        "follow_up": "das, was dir Freude macht",
    },
    "routines": {
        "opener": "dein Alltag",
        "follow_up": "dein Tagesrhythmus",
    },
    "pets": {
        "opener": "Tiere in deinem Alltag",
        "follow_up": "die Rolle von Tieren in deinem Leben",
    },
    "no_goes": {
        "opener": "Dinge, die ich lieber lasse",
        "follow_up": "Grenzen, die ich besser spueren sollte",
    },
    "health": {
        "opener": "dein Umgang mit Gesundheitsthemen",
        "follow_up": "was ich bei Gesundheitsthemen beachten sollte",
    },
}

_TOPIC_PROMPT_SENTENCES = {
    "basics": {
        "opener": "Ich moechte wissen, wie ich dich ansprechen soll.",
        "follow_up": "Ich habe schon etwas von dir gelernt und moechte das Bild noch runder machen.",
    },
    "companion_style": {
        "opener": "Ich moechte wissen, wie ich mit dir reden soll.",
        "follow_up": "Ich glaube, ich kann noch feiner verstehen, was sich im Ton zwischen uns gut anfuehlt.",
    },
    "social": {
        "opener": "Ich moechte wissen, wer dir wichtig ist.",
        "follow_up": "Bei deinem Umfeld habe ich noch einen kleinen offenen Faden.",
    },
    "interests": {
        "opener": "Ich moechte besser verstehen, was dich wirklich interessiert.",
        "follow_up": "Ich glaube, bei deinen Interessen fehlt mir noch ein spannender Winkel.",
    },
    "hobbies": {
        "opener": "Ich moechte wissen, was du gern machst.",
        "follow_up": "Ich wuerde gern noch besser verstehen, was dir im Alltag Freude macht.",
    },
    "routines": {
        "opener": "Ich moechte verstehen, wie dein Alltag meistens aussieht.",
        "follow_up": "Bei deinem Tagesrhythmus fehlt mir, glaube ich, noch ein kleines Stueck.",
    },
    "pets": {
        "opener": "Ich moechte wissen, ob Tiere in deinem Alltag eine Rolle spielen.",
        "follow_up": "Bei Tieren in deinem Alltag habe ich noch einen kleinen offenen Faden.",
    },
    "no_goes": {
        "opener": "Ich moechte wissen, was ich lieber lassen soll.",
        "follow_up": "Ich glaube, ich kann noch feiner verstehen, was ich besser bleiben lasse.",
    },
    "health": {
        "opener": "Ich moechte wissen, wie vorsichtig ich bei Gesundheitsthemen sein soll.",
        "follow_up": "Ich moechte vorsichtig besser verstehen, was ich bei Gesundheitsthemen beachten soll.",
    },
}

_TOPIC_CARD_INTENTS = {
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


def _compact_text(value: object | None, *, max_len: int) -> str:
    compact = " ".join(str(value or "").split()).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "..."


def _prompt_stage(invite: object) -> str:
    """Return the prompt stage the discovery service selected for this invite."""

    stage = _compact_text(getattr(invite, "display_prompt_stage", None), max_len=24).casefold()
    return stage if stage in {"opener", "follow_up", "review"} else "opener"


def _prompt_anchor(invite: object) -> str:
    """Return one human-facing anchor for prompt-time discovery cards."""

    invite_kind = _compact_text(getattr(invite, "invite_kind", None), max_len=32)
    if invite_kind == "review_profile":
        return "was ich schon ueber dich gelernt habe"
    topic_id = _compact_text(getattr(invite, "topic_id", None), max_len=32).casefold()
    stage = _prompt_stage(invite)
    anchor_options = _TOPIC_PROMPT_ANCHORS.get(topic_id, {})
    anchor = anchor_options.get(stage) or anchor_options.get("opener")
    if anchor:
        return anchor
    return _compact_text(getattr(invite, "display_topic_label", None), max_len=96)


def _prompt_hook_hint(invite: object, *, anchor: str) -> str:
    """Return one user-facing hook that avoids setup labels and UI wording."""

    invite_kind = _compact_text(getattr(invite, "invite_kind", None), max_len=32)
    if invite_kind == "review_profile":
        return _compact_text(
            "Ich moechte mit dir abgleichen, was ich schon ueber dich weiss und was davon noch passt.",
            max_len=160,
        )
    topic_id = _compact_text(getattr(invite, "topic_id", None), max_len=32).casefold()
    stage = _prompt_stage(invite)
    sentence_options = _TOPIC_PROMPT_SENTENCES.get(topic_id, {})
    sentence = sentence_options.get(stage) or sentence_options.get("opener")
    if sentence:
        return _compact_text(sentence, max_len=160)
    return _compact_text(f"Ich moechte noch besser verstehen, {anchor}.", max_len=160)


def _card_intent(invite: object) -> dict[str, str]:
    """Return structured semantic card intent for one discovery invite."""

    invite_kind = _compact_text(getattr(invite, "invite_kind", None), max_len=32)
    if invite_kind == "review_profile":
        return {
            "topic_semantics": "gemeinsamer Abgleich von bereits Gelerntem",
            "statement_intent": "Twinr will mit dem Nutzer abgleichen, was es schon ueber ihn gelernt hat.",
            "cta_intent": "Den Nutzer einladen, Gelerntes zu bestaetigen, zu korrigieren oder zu aktualisieren.",
            "relationship_stance": "ruhiger Rueckblick statt Profilpflege-Sprache",
        }
    topic_id = _compact_text(getattr(invite, "topic_id", None), max_len=32).casefold()
    card_intent = _TOPIC_CARD_INTENTS.get(topic_id)
    if card_intent is not None:
        return dict(card_intent)
    anchor = _prompt_anchor(invite)
    return {
        "topic_semantics": _compact_text(anchor, max_len=80) or "persoenliches Kennenlernen",
        "statement_intent": _compact_text(f"Twinr will den Nutzer besser verstehen, besonders bei {anchor}.", max_len=160),
        "cta_intent": "Den Nutzer zu einer kurzen persoenlichen Antwort einladen.",
        "relationship_stance": "ruhiges Kennenlernen ohne Meta-Sprache",
    }


def load_display_reserve_user_discovery_candidates(
    config: TwinrConfig,
    *,
    local_now: datetime,
    max_items: int,
) -> tuple[AmbientDisplayImpulseCandidate, ...]:
    """Expose at most one due get-to-know-you invitation candidate."""

    del max_items
    invite = UserDiscoveryService.from_config(config).build_invitation(now=local_now)
    if invite is None:
        return ()
    topic_key = f"user_discovery:{invite.phase}:{invite.topic_id}"
    prompt_anchor = _prompt_anchor(invite)
    prompt_hook_hint = _prompt_hook_hint(invite, anchor=prompt_anchor)
    card_intent = _card_intent(invite)
    return (
        AmbientDisplayImpulseCandidate(
            topic_key=topic_key,
            title=invite.display_topic_label,
            source="user_discovery",
            action="ask_one",
            attention_state="forming" if invite.phase == "initial_setup" else "growing",
            salience=float(invite.salience),
            eyebrow="",
            headline=_compact_text(invite.headline, max_len=112),
            body=_compact_text(invite.body, max_len=112),
            symbol="question",
            accent="warm",
            reason=_compact_text(invite.reason, max_len=120),
            candidate_family="user_discovery",
            generation_context={
                "candidate_family": "user_discovery",
                "display_goal": "invite_user_discovery",
                "invite_kind": invite.invite_kind,
                "phase": invite.phase,
                "topic_id": invite.topic_id,
                "topic_label": invite.topic_label,
                "display_label": invite.display_topic_label,
                "session_minutes": invite.session_minutes,
                "display_prompt_stage": _prompt_stage(invite),
                "display_anchor": prompt_anchor,
                "hook_hint": prompt_hook_hint,
                "topic_summary": prompt_hook_hint,
                "card_intent": card_intent,
                "raw_invite_headline": _compact_text(invite.headline, max_len=120),
                "raw_invite_body": _compact_text(invite.body, max_len=140),
            },
        ),
    )


__all__ = ["load_display_reserve_user_discovery_candidates"]
