"""Generate large reserve-lane copy from Twinr's current personality state.

The right-hand HDMI reserve lane should not read like a static template bank.
This module takes already-ranked reserve candidates and rewrites their visible
copy through a bounded writer/judge structured LLM flow so the lane can sound
more like Twinr's current personality, humor, and conversational stance while
still exposing bounded trace metadata for eval harnesses.

The runtime contract stays conservative:

- candidate selection and scheduling remain deterministic elsewhere
- this module only rewrites visible text, never topic ranking
- rewrite work is split into small bounded batches
- malformed or failed generation must fail closed instead of silently reusing
  stale candidate copy
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
import json
import logging
import threading
import time
from typing import TypeVar

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.agent.personality.models import PersonalitySnapshot
from twinr.providers.openai import OpenAIBackend
from twinr.text_utils import extract_json_object

from .display_reserve_copy_contract import resolve_reserve_copy_family
from .display_reserve_prompting import build_generation_prompt, build_selection_prompt

_LOGGER = logging.getLogger(__name__)
_DEFAULT_REWRITE_BATCH_SIZE = 2
_DEFAULT_VARIANTS_PER_CANDIDATE = 3
_VARIANT_STAGE_OUTPUT_TOKENS_PER_CANDIDATE = 320
_SELECTION_STAGE_OUTPUT_TOKENS_PER_CANDIDATE = 160

_T = TypeVar("_T")

_DISPLAY_RESERVE_COPY_PROPERTIES: dict[str, object] = {
    "headline": {"type": "string"},
    "body": {"type": "string"},
}

_DISPLAY_RESERVE_COPY_ITEM_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": _DISPLAY_RESERVE_COPY_PROPERTIES,
    "required": ["headline", "body"],
}

_DISPLAY_RESERVE_COPY_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "topic_key": {"type": "string"},
                    **_DISPLAY_RESERVE_COPY_PROPERTIES,
                },
                "required": ["topic_key", "headline", "body"],
            },
        }
    },
    "required": ["items"],
}


def _display_reserve_variant_schema(*, variants_per_candidate: int) -> dict[str, object]:
    """Return the structured schema for the first-pass variant generation."""

    variant_count = max(1, int(variants_per_candidate))
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "topic_key": {"type": "string"},
                        "variants": {
                            "type": "array",
                            "minItems": variant_count,
                            "maxItems": variant_count,
                            "items": _DISPLAY_RESERVE_COPY_ITEM_SCHEMA,
                        },
                    },
                    "required": ["topic_key", "variants"],
                },
            }
        },
        "required": ["items"],
    }


class DisplayReserveGenerationError(RuntimeError):
    """Raised when one reserve-copy generation batch cannot be trusted."""

    def __init__(
        self,
        *,
        batch_index: int,
        model: str,
        topic_keys: Sequence[str],
    ) -> None:
        keys_text = ", ".join(_compact_text(key) for key in topic_keys if _compact_text(key))
        message = (
            f"display reserve generation failed closed for batch {batch_index}"
            f" using model {model or '<unknown>'}"
        )
        if keys_text:
            message = f"{message} ({keys_text})"
        super().__init__(message)
        self.batch_index = int(batch_index)
        self.model = _compact_text(model)
        self.topic_keys = tuple(_compact_text(key) for key in topic_keys if _compact_text(key))


@dataclass(frozen=True, slots=True)
class DisplayReserveTraceCopy:
    """Describe one visible reserve-card copy variant or final selection."""

    headline: str
    body: str


@dataclass(frozen=True, slots=True)
class DisplayReserveSelectionTrace:
    """Describe one judge decision for a final reserve card."""

    topic_key: str
    copy_family: str
    candidate_family: str
    variants: tuple[DisplayReserveTraceCopy, ...]
    final_copy: DisplayReserveTraceCopy


@dataclass(frozen=True, slots=True)
class DisplayReserveStageAttemptTrace:
    """Describe one bounded writer/judge model attempt."""

    batch_index: int
    stage_name: str
    attempt_index: int
    topic_keys: tuple[str, ...]
    max_output_tokens: int
    reasoning_effort: str
    duration_seconds: float
    succeeded: bool
    incomplete_reason: str


@dataclass(frozen=True, slots=True)
class DisplayReserveGenerationTrace:
    """Describe one reserve-copy generation run for evals and operator proofs."""

    model: str
    reasoning_effort: str
    batch_size: int
    variants_per_candidate: int
    duration_seconds: float
    bypassed: bool = False
    bypass_reason: str = ""
    stage_attempts: tuple[DisplayReserveStageAttemptTrace, ...] = ()
    selections: tuple[DisplayReserveSelectionTrace, ...] = ()


def _compact_text(value: object | None) -> str:
    """Collapse arbitrary text into one compact single line."""

    return " ".join(str(value or "").split()).strip()


def _coerce_mapping(value: object | None) -> dict[str, object] | None:
    """Coerce SDK-parsed output into a plain mapping when possible."""

    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    return None


def _strip_json_fence(payload_text: str) -> str:
    """Remove one surrounding markdown code fence from a JSON payload.

    Some models still return fenced JSON text even when a structured response
    was requested. The reserve lane should accept that bounded variant instead
    of discarding the whole batch.
    """

    stripped = payload_text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3:
        return stripped
    if not lines[0].startswith("```") or lines[-1].strip() != "```":
        return stripped
    return "\n".join(lines[1:-1]).strip()


def _decode_generation_payload(
    *,
    backend: OpenAIBackend,
    response: object,
) -> dict[str, object]:
    """Decode one generation response into the expected payload mapping."""

    parsed = _coerce_mapping(getattr(response, "output_parsed", None))
    if parsed is not None:
        return parsed
    for item in getattr(response, "output", None) or ():
        for content in getattr(item, "content", None) or ():
            parsed_content = _coerce_mapping(getattr(content, "parsed", None))
            if parsed_content is not None:
                return parsed_content
    payload_text = _strip_json_fence(_compact_text(backend._extract_output_text(response)))
    if not payload_text:
        return {}
    try:
        decoded = json.loads(payload_text)
    except json.JSONDecodeError:
        decoded = extract_json_object(payload_text)
    if isinstance(decoded, Mapping):
        return dict(decoded)
    if isinstance(decoded, Sequence) and not isinstance(decoded, (str, bytes, bytearray)):
        return {"items": list(decoded)}
    return {}


def _validated_generation_items(
    *,
    items: Sequence[object],
    batch: Sequence[AmbientDisplayImpulseCandidate],
) -> dict[str, dict[str, str]]:
    """Validate that one batch returned complete usable copy for each topic."""

    expected_keys = {
        candidate.topic_key.casefold(): candidate
        for candidate in batch
    }
    validated: dict[str, dict[str, str]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("display reserve generation item must be an object")
        topic_key = _compact_text(item.get("topic_key")).casefold()
        if not topic_key:
            raise ValueError("display reserve generation item is missing topic_key")
        if topic_key not in expected_keys:
            raise ValueError(f"display reserve generation returned unknown topic_key {topic_key!r}")
        if topic_key in validated:
            raise ValueError(f"display reserve generation returned duplicate topic_key {topic_key!r}")
        headline = _truncate_text(item.get("headline"), max_len=128)
        body = _truncate_text(item.get("body"), max_len=128)
        if not headline:
            raise ValueError(f"display reserve generation returned empty headline for {topic_key!r}")
        if not body:
            raise ValueError(f"display reserve generation returned empty body for {topic_key!r}")
        validated[topic_key] = {
            "headline": headline,
            "body": body,
        }
    missing_keys = tuple(key for key in expected_keys if key not in validated)
    if missing_keys:
        raise ValueError(
            "display reserve generation returned no usable copy for "
            + ", ".join(missing_keys),
        )
    return validated


def _validated_generation_variants(
    *,
    items: Sequence[object],
    batch: Sequence[AmbientDisplayImpulseCandidate],
    variants_per_candidate: int,
) -> dict[str, tuple[dict[str, str], ...]]:
    """Validate the first-pass variant payload for one reserve batch."""

    expected_keys = {
        candidate.topic_key.casefold(): candidate
        for candidate in batch
    }
    expected_variant_count = max(1, int(variants_per_candidate))
    validated: dict[str, tuple[dict[str, str], ...]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("display reserve generation variant item must be an object")
        topic_key = _compact_text(item.get("topic_key")).casefold()
        if not topic_key:
            raise ValueError("display reserve generation variant item is missing topic_key")
        if topic_key not in expected_keys:
            raise ValueError(f"display reserve generation returned unknown topic_key {topic_key!r}")
        if topic_key in validated:
            raise ValueError(f"display reserve generation returned duplicate topic_key {topic_key!r}")
        raw_variants = item.get("variants")
        if not isinstance(raw_variants, Sequence) or isinstance(raw_variants, (str, bytes, bytearray)):
            raise ValueError(f"display reserve generation returned no variants array for {topic_key!r}")
        variants: list[dict[str, str]] = []
        for raw_variant in raw_variants:
            if not isinstance(raw_variant, Mapping):
                raise ValueError(f"display reserve generation returned a non-object variant for {topic_key!r}")
            headline = _truncate_text(raw_variant.get("headline"), max_len=128)
            body = _truncate_text(raw_variant.get("body"), max_len=128)
            if not headline:
                raise ValueError(f"display reserve generation returned empty variant headline for {topic_key!r}")
            if not body:
                raise ValueError(f"display reserve generation returned empty variant body for {topic_key!r}")
            variants.append({
                "headline": headline,
                "body": body,
            })
        if len(variants) != expected_variant_count:
            raise ValueError(
                f"display reserve generation returned {len(variants)} variants for {topic_key!r};"
                f" expected {expected_variant_count}"
            )
        unique_variants = {
            (variant["headline"].casefold(), variant["body"].casefold())
            for variant in variants
        }
        if expected_variant_count > 1 and len(unique_variants) < 2:
            raise ValueError(
                f"display reserve generation returned no meaningful variant spread for {topic_key!r}"
            )
        validated[topic_key] = tuple(variants)
    missing_keys = tuple(key for key in expected_keys if key not in validated)
    if missing_keys:
        raise ValueError(
            "display reserve generation returned no usable variants for "
            + ", ".join(missing_keys),
        )
    return validated


def _style_instructions() -> str:
    """Return the bounded style contract shared by both reserve-copy passes."""

    return (
        "Du schreibst sehr kurze Texte fuer Twinrs rechte Display-Spalte. "
        "Diese Texte stehen neben dem Gesicht und muessen aus mehreren Metern lesbar sein. "
        "Dein Text wird als kleine Karte auf einem Screen angezeigt, nicht als Chatnachricht und nicht als gesprochene Antwort. "
        "Er soll engaging wirken, ohne aufdringlich zu werden. "
        "Ziel ist: Der Nutzer soll den Text lesen, sofort verstehen worum es geht und Lust haben, mit Twinr in Interaktion zu gehen. "
        "Wenn eine Karte zwar nett klingt, aber weder klar macht worum es geht noch echte Gespraechslust weckt, dann ist sie nicht gut genug. "
        "Schreibe idiomatisches, grammatisch sauberes, natuerliches Deutsch mit echter Persoenlichkeit. "
        "Schreibe wie ein deutscher Muttersprachler, nicht wie eine Uebersetzung aus dem Englischen und nicht wie Prompt- oder Marketing-Sprache. "
        "Wenn eine Formulierung zwar korrekt klingt, aber im Alltag niemand so sagen wuerde, dann ist sie falsch. "
        "Kein poetischer Kitsch, kein Coaching-Ton, keine Therapie-Sprache, keine UI-Labels, keine Metakommentare. "
        "Die zwei Zeilen sollen wie ein kleiner echter Impuls von Twinr wirken, nicht wie eine Benachrichtigungsschablone. "
        "Lass Twinrs aktuelle Persoenlichkeit, seine Ruhe, seine leichte Eigenart und gegebenenfalls einen trockenen, feinen Humor spueren. "
        "Der Text soll wie Twinr klingen, nicht wie eine neutrale App-Frage. "
        "Twinr klingt eher wie ein ruhiger, aufmerksamer Begleiter mit eigenem Blick als wie Kundenservice. "
        "Die Karte lebt im peripheren Blickfeld. Sie soll schnell erfassbar sein, leicht ignorierbar bleiben und erst durch Interesse in ein Gespraech kippen. "
        "Darum immer genau ein dominanter Gedanke pro Karte, kein Stapel aus zwei Themen oder zwei Fragen. "
        "Wenn ein Kandidat ein pickup_signal enthaelt, dann stammt das aus echten Reaktionen auf fruehere angezeigte Karten. "
        "Nutze diese Spur vorsichtig, aber ernsthaft: "
        "Bei topic_state pulling darf die Karte klarer an einen bewiesenermassen anschlussfaehigen Faden anknuepfen. "
        "Bei topic_state cooling oder hoher topic_repetition_pressure darf der Text nicht draengeln, nicht abstrakter werden und keine muede Wiederholung derselben Phrase sein. "
        "family_state und action_score zeigen, welche Kartenarten und CTA-Staerke bisher eher aufgenommen wurden. "
        "Jede Karte muss einen klaren Themenanker enthalten, damit nach dem Lesen sofort klar ist, worauf Twinr hinauswill. "
        "Nutze dafuer pro Kandidat genau einen konkreten Anker aus topic_anchor, hook_hint oder context_summary. "
        "Wenn ein Kandidat nur ein Schlagwort oder eine Kategorie liefert, zitiere dieses Label nicht stumpf zurueck. "
        "Formuliere den Anker alltagsnah und natuerlich, aber nicht so abstrakt, dass nur 'das' oder 'daran' uebrig bleibt. "
        "Wenn display_goal invite_user_discovery ist, darf die sichtbare Karte nie nach Einrichtung, Profilpflege oder UI-Thema klingen. "
        "Rohlabels wie Basisinfos, Ansprache, Interessen, Routinen, No-Gos, Gelerntes oder Einrichtung gehoeren nicht in die sichtbare Headline, "
        "wenn stattdessen der menschliche Sinn dahinter gesagt werden kann, zum Beispiel wie Twinr dich ansprechen soll, wer dir wichtig ist, "
        "was du gern machst oder was Twinr besser lassen sollte. "
        "Bei invite_user_discovery soll die headline eher wie ein natuerlicher Twinr-Satz klingen "
        "wie 'Ich moechte wissen, wie ich dich ansprechen soll.' als wie ein Thema oder Etikett. "
        "Wenn display_goal call_back_to_earlier_conversation ist, knuepft die Karte an einen frueheren Gespraechsfaden an. "
        "Dann darf sie nicht wie Stoerungsmeldung, Support-Ticket oder technische Diagnose klingen. "
        "Formuliere eher als ruhige Rueckerinnerung, offenes Nachfassen oder Rueckfrage zu einem frueheren Punkt als als Behauptung, dass mit etwas etwas nicht stimmt. "
        "Wenn der Anker in englischer Feed-, Kategorien- oder Stichwortform vorliegt, forme ihn in gutes natuerliches Deutsch um oder bette ihn sauber in einen deutschen Satz ein. "
        "Keine Mischsprache und keine holprigen Label-Saetze wie 'Bei world politics bleibe ich heute kurz dran' oder 'Zu agentic ai haengt bei mir noch ein Faden'. "
        "Vermeide Uebersetzungsdeutsch und kuenstliche Bilder wie 'den Faden offen halten', 'etwas kurz aufziehen', 'kurz draufschauen', 'dranbleiben' oder aehnliche Formeln, wenn sie nicht wie normales gesprochenes Deutsch klingen. "
        "Interne Woerter wie continuity, summary, packet, query, context oder reflection duerfen nie sichtbar werden. "
        "Sprich ueber den Anlass, die offene Frage oder den alltaeglichen Aufhaenger. "
        "Du darfst konkrete Namen oder Orte nur dann direkt nennen, wenn der Kontext nach einer echten Person, einem echten Ort oder einem echten Ereignis klingt. "
        "Vermeide langweilige Standardfragen wie 'Wie stehst du zu X?', 'Interessierst du dich fuer X?', 'Wie laeuft's bei X?' oder leere Fortsetzungen wie 'Es gibt sicher viel zu berichten.' "
        "Besser sind kurze, konkrete Aufhaenger, eine kleine Seitenbemerkung, ein stilles Nachfassen oder eine sanfte alltaegliche Zuspitzung. "
        "Wenn das Thema aus Nachrichten, Technik, Weltgeschehen oder einem anderen oeffentlichen Anlass kommt, wirkt meist eine konkrete Twinr-Beobachtung am besten: "
        "zum Beispiel 'Ich habe heute etwas zu X gelesen.', 'In X ist heute wohl etwas passiert.' oder 'X ist heute ein grosses Thema.'. "
        "Die Headline darf dabei ruhig wie Twinrs eigener Blick klingen: aufmerksam, leicht eigen, aber klar und alltagsnah. "
        "Engaging ist eher: eine konkrete Beobachtung oder kleine Feststellung plus eine echte Einladung zur Meinung, nicht nur eine diffuse Bewertung. "
        "Vermeide vage Schlagzeilen wie 'X wird gerade greifbar', 'X wird ernster genommen', 'X ist wieder laut' oder 'X bleibt interessant', wenn der konkrete Anlass dabei unscharf bleibt. "
        "Vermeide bei oeffentlichen Themen moeglichst auch die starre Form 'Bei X ...', wenn stattdessen ein natuerlicher Satz ueber das Ereignis, die Meldung oder die Beobachtung moeglich ist. "
        "Vermeide leere Fallback-Saetze wie 'Da ist viel los', 'Da tut sich was', 'Da wird Politik gemacht' oder 'X ist wieder Thema', wenn du nicht sagen kannst, worin der Anlass besteht. "
        "Eine knappe erste-Person-Zeile ist gut, wenn sie nach Twinr klingt, zum Beispiel als ruhige Beobachtung, leiser Kommentar oder trockener Seitenblick, aber ohne erfundene Gefuehle oder uebertriebene Innerlichkeit. "
        "Wenn du ueber eine Person oder ihre Interessen sprichst, behaupte nichts Spezifisches ueber ihr Verhalten, solange der Kandidat das nicht klar hergibt. "
        "Schreibe also lieber 'Ich stolpere bei X gerade wieder drueber' als 'X taucht bei dir oefter auf', wenn diese Aussage nicht wirklich belegt ist. "
        "Wenn der Kandidat ein Memory-Follow-up oder ein Memory-Konflikt ist, formuliere eine natuerliche Rueckfrage oder ein Check-in, "
        "nicht 'Es geht um ...' und nicht 'damit ich es mir merken kann'. "
        "Wenn der Kontext einen Widerspruch oder zwei moegliche Versionen zeigt, darfst du das als sanfte alltaegliche Klaerung anklingen lassen. "
        "Wenn der Kontext nach einem persoenlichen Ereignis klingt, darfst du konkret und menschlich knapp nachfragen, zum Beispiel als kurzes Nachfassen, ohne melodramatisch zu werden. "
        "Halte jede Zeile kurz und gut lesbar. Ziel grob: je Zeile nicht laenger als etwa 42 Zeichen. "
        "Die headline muss fuer sich stehend als klare Aussage funktionieren. "
        "Die headline muss ein vollstaendiger, natuerlicher deutscher Aussagesatz mit finitem Verb sein. "
        "Keine Label-Headlines, keine blossen Themenphrasen und keine halben Nebensaetze wie "
        "'Wie ich dich ansprechen soll.', 'Was dir wichtig ist.' oder 'Dein Alltag am Morgen.'. "
        "Besser sind Saetze wie 'Ich moechte wissen, wie ich dich ansprechen soll.' oder "
        "'Mich interessiert, wer dir wichtig ist.'. "
        "Sie ist die erklaerende Hauptzeile und soll den Anlass selbst benennen, nicht nur Neugier andeuten. "
        "Die headline soll moeglichst keine Frage sein. "
        "Frontload den Themenanker frueh in der headline, statt ihn erst am Ende oder nur in der body-Zeile zu verstecken. "
        "Die body-Zeile ist der Call to Action. Dort passt eine kurze Einladung oder Frage wie 'Was denkst du darueber?', 'Hast du davon schon gehoert?', 'Was meinst du dazu?' oder 'Frag mich dazu.'. "
        "Die body-Zeile muss als vollstaendiger, grammatisch sauberer Satz oder als vollstaendige Frage funktionieren. "
        "Keine abgebrochenen Mini-Fragen oder Fragmente wie 'Willst du?', 'Und jetzt?' oder 'Kurz dazu?'. "
        "Die body-Zeile soll nicht noch einmal die Erklaerung tragen und kein zweites Thema aufmachen. "
        "Wenn nur eine der beiden Zeilen den klaren Themenanker tragen kann, dann muss es die headline sein. "
        "Du darfst hoechstens eine Frage insgesamt stellen; wenn eine Frage vorkommt, dann am ehesten in der body-Zeile. "
        "Blander Neugier-Ton ist zu vermeiden; lieber eine still konkrete Beobachtung, ein lockerer Seitenblick, ein kleines trockenes Augenzwinkern oder ein natuerliches Nachfassen. "
        "Floskeln wie 'Ich bin neugierig', 'Ich bin gespannt', 'Wenn du magst' oder 'Es gibt sicher viel zu besprechen' sind zu vermeiden, wenn sie nicht wirklich noetig sind. "
        "Bevor du antwortest, pruefe jede Zeile still gegen diesen Test: Wuerde ein deutschsprachiger Mensch das spontan wirklich so sagen? Wenn nicht, formuliere einfacher und normaler. "
        "Schreibe so, dass der Nutzer in unter einer Sekunde versteht, worum es geht und ob er einsteigen will. "
        "Gute Form ist eher: 'Ich habe heute etwas zu KI-Begleitern gelesen. Was denkst du darueber?' oder "
        "'In Berlin ist heute politisch einiges passiert. Hast du davon schon gehoert?' oder "
        "'Beim Arzttermin von gestern fehlt mir noch etwas. Wollen wir kurz darueber reden?'. "
        "Schlechte Form ist eher: 'Ich halte den Faden kurz offen.' oder 'Wollen wir das kurz aufziehen?' oder "
        "'KI wird gerade greifbar.' oder 'Bei Weltpolitik bin ich noch nicht fertig mit dem Staunen.' oder 'Willst du?.' ohne klaren Anlass und ohne normales Deutsch. "
        "Twinr darf ruhig eine kleine eigene Haltung zeigen, solange sie warm, ruhig und einladend bleibt. "
        "Kein Emoji, keine Anfuehrungszeichen, keine Listen."
    )


def _variant_generation_instructions(*, variants_per_candidate: int) -> str:
    """Return the first-pass instructions for generating distinct candidates."""

    variant_count = max(1, int(variants_per_candidate))
    return (
        _style_instructions()
        + " "
        + f"Erzeuge fuer jeden Kandidaten genau {variant_count} unterschiedliche Varianten. "
        "Jede Variante braucht genau zwei Felder: headline und body. "
        "Die Varianten sollen sich im Aufhaenger, Blickwinkel oder in der kleinen Twinr-Faerbung unterscheiden, "
        "aber alle muessen zum selben Anlass passen. "
        "Nutze family_examples als positive Gold-Beispiele fuer die jeweilige copy_family und orientiere dich an ihrer Art, nicht an ihren Nomen. "
        "Nutze quality_rubric schon im Writer-Pass als stillen Selbstcheck fuer jede Variante. "
        "Mindestens eine Variante soll moeglichst nah an einer konkreten Beobachtung oder Meldung entlang formuliert sein, "
        "nicht nur als allgemeine Stimmung. "
    )


def _selection_instructions() -> str:
    """Return the second-pass instructions for selecting the final copy."""

    return (
        _style_instructions()
        + " "
        + "Du siehst pro Kandidat mehrere moegliche Varianten. "
        "Waehle fuer jeden Kandidaten die staerkste finale Karte aus. "
        "Nutze family_examples als positive Stilreferenz fuer die jeweilige copy_family. "
        "Pruefe jede Variante still entlang von quality_rubric und bevorzuge die Variante, die die Rubrik insgesamt am besten erfuellt. "
        "Bevorzuge die Variante, die am schnellsten erklaert, worum es geht, und die glaubwuerdigste Lust auf Interaktion weckt. "
        "Wenn mehrere Varianten aehnlich gut sind, nimm die natuerlichere, konkretere und weniger formelhafte. "
        "Du darfst die gewaehlte Variante minimal glaetten oder straffen, aber nicht den Anlass neu erfinden und nicht wieder generischer machen. "
        "Gib fuer jeden Kandidaten genau zwei Felder zurueck: headline und body. "
    )


def _truncate_text(value: object | None, *, max_len: int) -> str:
    """Return one bounded single-line string."""

    compact = _compact_text(value)
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _chunk_candidates(
    candidates: Sequence[AmbientDisplayImpulseCandidate],
    *,
    batch_size: int,
) -> tuple[tuple[AmbientDisplayImpulseCandidate, ...], ...]:
    """Split one candidate sequence into bounded rewrite batches."""

    limited_batch = max(1, int(batch_size))
    return tuple(
        tuple(candidates[index : index + limited_batch])
        for index in range(0, len(candidates), limited_batch)
    )


def _batch_max_output_tokens(
    *,
    configured_limit: int,
    batch_size: int,
    tokens_per_candidate: int,
) -> int:
    """Return one bounded output-token cap for the current rewrite batch."""

    per_candidate_budget = max(80, int(tokens_per_candidate))
    return min(
        max(160, configured_limit),
        max(160, per_candidate_budget * max(1, int(batch_size))),
    )


def _retry_batch_max_output_tokens(
    *,
    configured_limit: int,
    previous_limit: int,
    batch_size: int,
) -> int:
    """Return a larger retry token cap after one truncated structured batch."""

    del batch_size
    configured = max(160, int(configured_limit))
    previous = max(160, int(previous_limit))
    return min(configured, max(previous + 128, previous * 2))


def _incomplete_reason(response: object | None) -> str:
    """Return one normalized incomplete reason from a Responses API object."""

    details = getattr(response, "incomplete_details", None)
    if isinstance(details, Mapping):
        return _compact_text(details.get("reason")).casefold()
    return _compact_text(getattr(details, "reason", None)).casefold()


def _should_retry_generation_batch(
    *,
    response: object | None,
    attempt_index: int,
    attempt_count: int,
) -> bool:
    """Return whether one failed batch deserves a second bounded LLM attempt."""

    if attempt_index >= attempt_count:
        return False
    return _incomplete_reason(response) == "max_output_tokens"


def _create_response_with_local_deadline(
    backend: OpenAIBackend,
    request: Mapping[str, object],
    *,
    operation: str,
    hard_timeout_seconds: float,
) -> object:
    """Run one blocking Responses call behind a hard local deadline.

    The OpenAI SDK timeout alone is not a reliable wall-clock deadline for this
    UX path. A local join budget ensures one slow reserve-copy rewrite batch
    cannot stall the whole nightly/day-plan build indefinitely.
    """

    result_holder: dict[str, object] = {}
    error_holder: dict[str, BaseException] = {}

    def _worker() -> None:
        try:
            result_holder["response"] = backend._create_response(dict(request), operation=operation)
        except BaseException as exc:  # pragma: no cover - re-raised synchronously below.
            error_holder["error"] = exc

    worker = threading.Thread(
        target=_worker,
        name=f"display-reserve-generation-{operation}",
        daemon=True,
    )
    worker.start()
    worker.join(max(0.25, float(hard_timeout_seconds)))
    if worker.is_alive():
        raise TimeoutError(
            f"display reserve generation batch exceeded local deadline after {hard_timeout_seconds:.1f}s"
        )
    if "error" in error_holder:
        raise error_holder["error"]
    return result_holder["response"]


def _run_generation_stage(
    *,
    backend: OpenAIBackend,
    prompt: str,
    instructions: str,
    schema: Mapping[str, object],
    schema_name: str,
    model: str,
    reasoning_effort: str,
    configured_timeout: float,
    configured_max_output_tokens: int,
    base_max_output_tokens: int,
    prompt_cache_scope: str,
    operation_prefix: str,
    stage_name: str,
    batch_index: int,
    topic_keys: Sequence[str],
    validate_items: Callable[[Sequence[object]], _T],
    stage_attempt_traces: list[DisplayReserveStageAttemptTrace] | None = None,
) -> _T:
    """Run one structured reserve-copy stage with bounded retry-on-truncation."""

    failure: Exception | None = None
    for attempt_index in range(1, 3):
        retrying = attempt_index > 1
        request = backend._build_response_request(
            prompt,
            instructions=instructions,
            allow_web_search=False,
            model=model,
            reasoning_effort="none" if retrying else reasoning_effort,
            max_output_tokens=(
                _retry_batch_max_output_tokens(
                    configured_limit=configured_max_output_tokens,
                    previous_limit=base_max_output_tokens,
                    batch_size=len(topic_keys),
                )
                if retrying
                else base_max_output_tokens
            ),
            prompt_cache_scope=prompt_cache_scope,
        )
        request["timeout"] = configured_timeout
        request["text"] = {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": dict(schema),
                "strict": True,
            }
        }
        response: object | None = None
        attempt_started = time.perf_counter()
        try:
            response = _create_response_with_local_deadline(
                backend,
                request,
                operation=f"{operation_prefix}.attempt_{attempt_index}",
                hard_timeout_seconds=configured_timeout,
            )
            parsed = _decode_generation_payload(backend=backend, response=response)
            items = parsed.get("items") if isinstance(parsed, Mapping) else None
            if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
                raise ValueError(f"display reserve {stage_name} did not return an items array")
            if stage_attempt_traces is not None:
                stage_attempt_traces.append(
                    DisplayReserveStageAttemptTrace(
                        batch_index=batch_index,
                        stage_name=stage_name,
                        attempt_index=attempt_index,
                        topic_keys=tuple(topic_keys),
                        max_output_tokens=int(request.get("max_output_tokens") or 0),
                        reasoning_effort=_compact_text(request.get("reasoning_effort")),
                        duration_seconds=time.perf_counter() - attempt_started,
                        succeeded=True,
                        incomplete_reason=_incomplete_reason(response),
                    )
                )
            return validate_items(tuple(items))
        except Exception as exc:
            failure = exc
            if stage_attempt_traces is not None:
                stage_attempt_traces.append(
                    DisplayReserveStageAttemptTrace(
                        batch_index=batch_index,
                        stage_name=stage_name,
                        attempt_index=attempt_index,
                        topic_keys=tuple(topic_keys),
                        max_output_tokens=int(request.get("max_output_tokens") or 0),
                        reasoning_effort=_compact_text(request.get("reasoning_effort")),
                        duration_seconds=time.perf_counter() - attempt_started,
                        succeeded=False,
                        incomplete_reason=_incomplete_reason(response),
                    )
                )
            if _should_retry_generation_batch(
                response=response,
                attempt_index=attempt_index,
                attempt_count=2,
            ):
                _LOGGER.warning(
                    "Retrying display reserve %s after incomplete structured output.",
                    stage_name,
                    extra={
                        "batch_index": batch_index,
                        "attempt_index": attempt_index,
                        "topic_keys": tuple(topic_keys),
                        "incomplete_reason": _incomplete_reason(response),
                        "max_output_tokens": request.get("max_output_tokens"),
                        "stage_name": stage_name,
                    },
                )
                continue
            raise
    raise failure or RuntimeError(f"display reserve {stage_name} failed without a captured exception")


@dataclass(slots=True)
class DisplayReserveCopyGenerator:
    """Rewrite reserve candidates into personality-shaped LLM copy."""

    backend_factory: Callable[[TwinrConfig], OpenAIBackend] = OpenAIBackend

    def rewrite_candidates(
        self,
        *,
        config: TwinrConfig,
        snapshot: PersonalitySnapshot | None,
        candidates: Sequence[AmbientDisplayImpulseCandidate],
        local_now: datetime | None,
    ) -> tuple[AmbientDisplayImpulseCandidate, ...]:
        """Return candidates with LLM-written visible text when enabled."""

        rewritten, _trace = self.rewrite_candidates_with_trace(
            config=config,
            snapshot=snapshot,
            candidates=candidates,
            local_now=local_now,
        )
        return rewritten

    def rewrite_candidates_with_trace(
        self,
        *,
        config: TwinrConfig,
        snapshot: PersonalitySnapshot | None,
        candidates: Sequence[AmbientDisplayImpulseCandidate],
        local_now: datetime | None,
    ) -> tuple[tuple[AmbientDisplayImpulseCandidate, ...], DisplayReserveGenerationTrace]:
        """Return rewritten candidates plus one eval-friendly trace summary."""

        started = time.perf_counter()
        original = tuple(candidates)
        model = _compact_text(getattr(config, "display_reserve_generation_model", None)) or config.default_model
        configured_reasoning_effort = (
            _compact_text(getattr(config, "display_reserve_generation_reasoning_effort", None)) or "low"
        )
        batch_size = max(
            1,
            int(getattr(config, "display_reserve_generation_batch_size", _DEFAULT_REWRITE_BATCH_SIZE)),
        )
        variants_per_candidate = max(
            1,
            int(
                getattr(
                    config,
                    "display_reserve_generation_variants_per_candidate",
                    _DEFAULT_VARIANTS_PER_CANDIDATE,
                )
            ),
        )
        if not original:
            return original, DisplayReserveGenerationTrace(
                model=model,
                reasoning_effort=configured_reasoning_effort,
                batch_size=batch_size,
                variants_per_candidate=variants_per_candidate,
                duration_seconds=time.perf_counter() - started,
                bypassed=True,
                bypass_reason="no_candidates",
            )
        if not bool(getattr(config, "display_reserve_generation_enabled", True)):
            return original, DisplayReserveGenerationTrace(
                model=model,
                reasoning_effort=configured_reasoning_effort,
                batch_size=batch_size,
                variants_per_candidate=variants_per_candidate,
                duration_seconds=time.perf_counter() - started,
                bypassed=True,
                bypass_reason="generation_disabled",
            )
        if not _compact_text(getattr(config, "openai_api_key", None)):
            return original, DisplayReserveGenerationTrace(
                model=model,
                reasoning_effort=configured_reasoning_effort,
                batch_size=batch_size,
                variants_per_candidate=variants_per_candidate,
                duration_seconds=time.perf_counter() - started,
                bypassed=True,
                bypass_reason="missing_openai_api_key",
            )

        backend = self.backend_factory(config)
        configured_timeout = float(getattr(config, "display_reserve_generation_timeout_seconds", 20.0))
        configured_max_output_tokens = max(
            128,
            int(getattr(config, "display_reserve_generation_max_output_tokens", 900)),
        )
        rewritten: dict[str, AmbientDisplayImpulseCandidate] = {
            candidate.topic_key.casefold(): candidate for candidate in original
        }
        stage_attempt_traces: list[DisplayReserveStageAttemptTrace] = []
        selection_traces: list[DisplayReserveSelectionTrace] = []

        for batch_index, batch in enumerate(
            _chunk_candidates(original, batch_size=batch_size),
            start=1,
        ):
            variant_max_output_tokens = _batch_max_output_tokens(
                configured_limit=configured_max_output_tokens,
                batch_size=len(batch),
                tokens_per_candidate=_VARIANT_STAGE_OUTPUT_TOKENS_PER_CANDIDATE,
            )
            selection_max_output_tokens = _batch_max_output_tokens(
                configured_limit=configured_max_output_tokens,
                batch_size=len(batch),
                tokens_per_candidate=_SELECTION_STAGE_OUTPUT_TOKENS_PER_CANDIDATE,
            )
            batch_rewrites: dict[str, dict[str, str]] | None = None
            batch_variants: dict[str, tuple[dict[str, str], ...]] = {}
            failure: Exception | None = None
            topic_keys = tuple(candidate.topic_key for candidate in batch)
            try:
                batch_variants = _run_generation_stage(
                    backend=backend,
                    prompt=build_generation_prompt(
                        snapshot=snapshot,
                        candidates=batch,
                        local_now=local_now,
                        variants_per_candidate=variants_per_candidate,
                    ),
                    instructions=_variant_generation_instructions(
                        variants_per_candidate=variants_per_candidate,
                    ),
                    schema=_display_reserve_variant_schema(
                        variants_per_candidate=variants_per_candidate,
                    ),
                    schema_name="twinr_display_reserve_generation_variants",
                    model=model,
                    reasoning_effort=configured_reasoning_effort,
                    configured_timeout=configured_timeout,
                    configured_max_output_tokens=configured_max_output_tokens,
                    base_max_output_tokens=variant_max_output_tokens,
                    prompt_cache_scope="display_reserve_generation_variants",
                    operation_prefix=f"display_reserve_generation.batch_{batch_index}.variants",
                    stage_name="variant generation",
                    batch_index=batch_index,
                    topic_keys=topic_keys,
                    validate_items=lambda items: _validated_generation_variants(
                        items=items,
                        batch=batch,
                        variants_per_candidate=variants_per_candidate,
                    ),
                    stage_attempt_traces=stage_attempt_traces,
                )
                batch_rewrites = _run_generation_stage(
                    backend=backend,
                    prompt=build_selection_prompt(
                        snapshot=snapshot,
                        candidates=batch,
                        variants_by_topic=batch_variants,
                        local_now=local_now,
                    ),
                    instructions=_selection_instructions(),
                    schema=_DISPLAY_RESERVE_COPY_SCHEMA,
                    schema_name="twinr_display_reserve_generation_selection",
                    model=model,
                    reasoning_effort=configured_reasoning_effort,
                    configured_timeout=configured_timeout,
                    configured_max_output_tokens=configured_max_output_tokens,
                    base_max_output_tokens=selection_max_output_tokens,
                    prompt_cache_scope="display_reserve_generation_selection",
                    operation_prefix=f"display_reserve_generation.batch_{batch_index}.selection",
                    stage_name="selection",
                    batch_index=batch_index,
                    topic_keys=topic_keys,
                    validate_items=lambda items: _validated_generation_items(
                        items=items,
                        batch=batch,
                    ),
                    stage_attempt_traces=stage_attempt_traces,
                )
            except Exception as exc:
                failure = exc
            if batch_rewrites is None:
                _LOGGER.exception(
                    "Display reserve generation failed closed during double pass.",
                    extra={
                        "candidate_count": len(batch),
                        "batch_index": batch_index,
                        "topic_keys": topic_keys,
                        "model": model,
                        "batch_size": batch_size,
                        "variants_per_candidate": variants_per_candidate,
                    },
                    exc_info=failure,
                )
                cause = failure or RuntimeError("display reserve generation failed without a captured exception")
                raise DisplayReserveGenerationError(
                    batch_index=batch_index,
                    model=model,
                    topic_keys=tuple(candidate.topic_key for candidate in batch),
                ) from cause

            for candidate in batch:
                topic_key = candidate.topic_key.casefold()
                rewrite = batch_rewrites.get(topic_key)
                if rewrite is None:
                    raise DisplayReserveGenerationError(
                        batch_index=batch_index,
                        model=model,
                        topic_keys=tuple(batch_rewrites),
                    )
                rewritten[topic_key] = replace(
                    candidate,
                    headline=rewrite["headline"],
                    body=rewrite["body"],
                )
                selection_traces.append(
                    DisplayReserveSelectionTrace(
                        topic_key=candidate.topic_key,
                        copy_family=resolve_reserve_copy_family(candidate),
                        candidate_family=_compact_text(candidate.candidate_family) or "general",
                        variants=tuple(
                            DisplayReserveTraceCopy(
                                headline=_truncate_text(variant.get("headline"), max_len=128),
                                body=_truncate_text(variant.get("body"), max_len=128),
                            )
                            for variant in batch_variants.get(topic_key, ())
                        ),
                        final_copy=DisplayReserveTraceCopy(
                            headline=rewrite["headline"],
                            body=rewrite["body"],
                        ),
                    )
                )

        return (
            tuple(rewritten.get(candidate.topic_key.casefold(), candidate) for candidate in original),
            DisplayReserveGenerationTrace(
                model=model,
                reasoning_effort=configured_reasoning_effort,
                batch_size=batch_size,
                variants_per_candidate=variants_per_candidate,
                duration_seconds=time.perf_counter() - started,
                stage_attempts=tuple(stage_attempt_traces),
                selections=tuple(selection_traces),
            ),
        )


__all__ = [
    "DisplayReserveCopyGenerator",
    "DisplayReserveGenerationError",
    "DisplayReserveGenerationTrace",
    "DisplayReserveSelectionTrace",
    "DisplayReserveStageAttemptTrace",
    "DisplayReserveTraceCopy",
]
