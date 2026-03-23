"""Expose the active reserve-lane card as bounded provider-grounding context.

Twinr's fast supervisor and search lanes should understand when the user is
reacting to the currently visible right-hand reserve card. This module keeps
that display-to-dialog bridge out of the larger runtime context mixin:

- read the active reserve-card cue from the existing display cue store
- normalize only the tiny user-relevant fields
- emit one short model-facing grounding message that says, in effect,
  ``AUF DEINEM SCREEN STEHT GERADE: ...`` so upstream lanes can anchor deictic
  or slightly noisy follow-ups to the visible card
"""

from __future__ import annotations

from twinr.display.ambient_impulse_cues import DisplayAmbientImpulseCue, DisplayAmbientImpulseCueStore


def _compact_text(value: object | None, *, max_len: int) -> str:
    """Collapse arbitrary text into one bounded single-line string."""

    if value is None:
        return ""
    compact = " ".join(str(value).split()).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _cue_summary(cue: DisplayAmbientImpulseCue) -> tuple[str, str, str]:
    """Return the bounded semantic fields worth surfacing to provider lanes."""

    topic = _compact_text(cue.topic_key, max_len=96)
    headline = _compact_text(cue.headline, max_len=128)
    body = _compact_text(cue.body, max_len=128)
    return topic, headline, body


def build_active_display_grounding_message(config) -> str | None:
    """Return one narrow grounding message for the currently visible reserve cue.

    The message is intentionally explicit about the visible screen content so
    the model can ground turns like "das da" or a slightly garbled repeat of a
    shown topic. It must still never override a clearly different explicit user
    request.
    """

    try:
        store = DisplayAmbientImpulseCueStore.from_config(config)
        cue = store.load_active()
    except Exception:
        return None
    if cue is None:
        return None
    topic, headline, body = _cue_summary(cue)
    if not any((topic, headline, body)):
        return None

    parts = ["AUF DEINEM SCREEN STEHT GERADE eine sichtbare Themenkarte."]
    if topic:
        parts.append(f"Sichtbarer Themenanker: {topic}.")
    if headline:
        parts.append(f"Sichtbare Überschrift: {headline}.")
    if body:
        parts.append(f"Sichtbarer Zusatz: {body}.")
    parts.append(
        "Wenn der Nutzer sich wahrscheinlich auf diese sichtbare Karte bezieht, behandle diesen Screen-Inhalt als primären Deutungsanker."
    )
    parts.append(
        "Wenn ein einzelnes Wort im Transkript wie leichte ASR-Störung oder ein fehlerhafter Nah-Treffer wirkt, formuliere goal und prompt mit dem sichtbaren Thema statt mit dem verrauschten Wort."
    )
    parts.append(
        "Überschreibe damit aber keine klar anderslautende explizite Anfrage des Nutzers."
    )
    return " ".join(parts)

def build_active_display_grounding_instruction_overlay(config) -> str | None:
    """Return one authoritative turn overlay for display-grounded supervisor calls.

    The fast supervisor already sees the active cue in the conversation
    context. This overlay repeats only the bounded essential parts as
    authoritative turn guidance so display-grounded routing does not depend on
    lower-priority context ordering alone.
    """

    grounding = build_active_display_grounding_message(config)
    if not grounding:
        return None
    return " ".join(
        (
            grounding,
            "Für diesen Turn ist diese sichtbare Karte autoritativer situativer Kontext.",
            "Wenn der Nutzer wahrscheinlich auf diese sichtbare Karte reagiert, halte das sichtbare Thema in spoken_ack, spoken_reply, goal und prompt explizit sichtbar.",
            "Wenn der Nutzer eine kurze soziale oder erklärende Anschlussfrage zu dieser sichtbaren Karte stellt, ist meist eine direkte, natürliche Companion-Antwort passender als end_conversation.",
            "Wähle end_conversation nur bei klarem Stoppsignal des Nutzers, nicht bloß wegen einer kurzen Rückfrage zur sichtbaren Karte.",
        )
    )


__all__ = [
    "build_active_display_grounding_instruction_overlay",
    "build_active_display_grounding_message",
]
