"""Generate large reserve-lane copy from Twinr's current personality state.

The right-hand HDMI reserve lane should not read like a static template bank.
This module takes already-ranked reserve candidates and rewrites their visible
copy through one bounded structured LLM call so the lane can sound more like
Twinr's current personality, humor, and conversational stance.

The runtime contract stays conservative:

- candidate selection and scheduling remain deterministic elsewhere
- this module only rewrites visible text, never topic ranking
- rewrite work is split into small bounded batches with isolated fallback
- failures fall back to the deterministic copy already attached to candidates
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
import json
import logging
import threading

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.agent.personality.models import PersonalitySnapshot
from twinr.providers.openai import OpenAIBackend

from .display_reserve_prompting import build_generation_prompt

_LOGGER = logging.getLogger(__name__)
_DEFAULT_REWRITE_BATCH_SIZE = 2

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
                    "headline": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["topic_key", "headline", "body"],
            },
        }
    },
    "required": ["items"],
}


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
    payload_text = _strip_json_fence(_compact_text(backend._extract_output_text(response)))
    if not payload_text:
        return {}
    decoded = json.loads(payload_text)
    if isinstance(decoded, Mapping):
        return dict(decoded)
    if isinstance(decoded, Sequence) and not isinstance(decoded, (str, bytes, bytearray)):
        return {"items": list(decoded)}
    return {}


def _generation_instructions() -> str:
    """Return the bounded system instructions for reserve-lane copy."""

    return (
        "Du schreibst sehr kurze Texte fuer Twinrs rechte Display-Spalte. "
        "Diese Texte stehen neben dem Gesicht und muessen aus mehreren Metern lesbar sein. "
        "Schreibe idiomatisches, grammatisch sauberes, natuerliches Deutsch mit echter Persoenlichkeit. "
        "Kein poetischer Kitsch, kein Coaching-Ton, keine Therapie-Sprache, keine UI-Labels, keine Metakommentare. "
        "Die zwei Zeilen sollen wie ein kleiner echter Impuls von Twinr wirken, nicht wie eine Benachrichtigungsschablone. "
        "Lass Twinrs aktuelle Persoenlichkeit, seine Ruhe, seine leichte Eigenart und gegebenenfalls einen trockenen, feinen Humor spueren. "
        "Der Text soll wie Twinr klingen, nicht wie eine neutrale App-Frage. "
        "Twinr klingt eher wie ein ruhiger, aufmerksamer Begleiter mit eigenem Blick als wie Kundenservice. "
        "Die Karte lebt im peripheren Blickfeld. Sie soll schnell erfassbar sein, leicht ignorierbar bleiben und erst durch Interesse in ein Gespraech kippen. "
        "Darum immer genau ein dominanter Gedanke pro Karte, kein Stapel aus zwei Themen oder zwei Fragen. "
        "Jede Karte muss einen klaren Themenanker enthalten, damit nach dem Lesen sofort klar ist, worauf Twinr hinauswill. "
        "Nutze dafuer pro Kandidat genau einen konkreten Anker aus topic_anchor, hook_hint oder context_summary. "
        "Wenn ein Kandidat nur ein Schlagwort oder eine Kategorie liefert, zitiere dieses Label nicht stumpf zurueck. "
        "Formuliere den Anker alltagsnah und natuerlich, aber nicht so abstrakt, dass nur 'das' oder 'daran' uebrig bleibt. "
        "Wenn der Anker in englischer Feed-, Kategorien- oder Stichwortform vorliegt, forme ihn in gutes natuerliches Deutsch um oder bette ihn sauber in einen deutschen Satz ein. "
        "Keine Mischsprache und keine holprigen Label-Saetze wie 'Bei world politics bleibe ich heute kurz dran' oder 'Zu agentic ai haengt bei mir noch ein Faden'. "
        "Interne Woerter wie continuity, summary, packet, query, context oder reflection duerfen nie sichtbar werden. "
        "Sprich ueber den Anlass, die offene Frage oder den alltaeglichen Aufhaenger. "
        "Du darfst konkrete Namen oder Orte nur dann direkt nennen, wenn der Kontext nach einer echten Person, einem echten Ort oder einem echten Ereignis klingt. "
        "Vermeide langweilige Standardfragen wie 'Wie stehst du zu X?', 'Interessierst du dich fuer X?', 'Wie laeuft's bei X?' oder leere Fortsetzungen wie 'Es gibt sicher viel zu berichten.' "
        "Besser sind kurze, konkrete Aufhaenger, eine kleine Seitenbemerkung, ein stilles Nachfassen oder eine sanfte alltaegliche Zuspitzung. "
        "Eine knappe erste-Person-Zeile ist gut, wenn sie nach Twinr klingt, zum Beispiel als ruhige Beobachtung, leiser Kommentar oder trockener Seitenblick, aber ohne erfundene Gefuehle oder uebertriebene Innerlichkeit. "
        "Wenn der Kandidat ein Memory-Follow-up oder ein Memory-Konflikt ist, formuliere eine natuerliche Rueckfrage oder ein Check-in, "
        "nicht 'Es geht um ...' und nicht 'damit ich es mir merken kann'. "
        "Wenn der Kontext einen Widerspruch oder zwei moegliche Versionen zeigt, darfst du das als sanfte alltaegliche Klaerung anklingen lassen. "
        "Wenn der Kontext nach einem persoenlichen Ereignis klingt, darfst du konkret und menschlich knapp nachfragen, zum Beispiel als kurzes Nachfassen, ohne melodramatisch zu werden. "
        "Halte jede Zeile kurz und gut lesbar. Ziel grob: je Zeile nicht laenger als etwa 42 Zeichen. "
        "Gib fuer jeden Kandidaten genau zwei Zeilen zurueck: headline und body. "
        "Die headline soll meist der eigentliche Aufhaenger sein. Wenn du eine Frage stellst, dann vorzugsweise dort. "
        "Frontload den Themenanker moeglichst frueh in der headline, statt ihn erst am Ende oder nur in der body-Zeile zu verstecken. "
        "Die body-Zeile soll ihn weitertragen, persoenlicher machen oder leicht zuspitzen. "
        "Wenn die headline noch keinen klaren Themenanker enthaelt, muss die body-Zeile ihn enthalten. "
        "Du darfst hoechstens eine Frage insgesamt stellen. Die zweite Zeile soll nach Moeglichkeit eine kurze, ruhige Fortsetzung oder kleine Haltung sein, fast nie noch eine Frage. "
        "Blander Neugier-Ton ist zu vermeiden; lieber eine still konkrete Beobachtung, ein lockerer Seitenblick, ein kleines trockenes Augenzwinkern oder ein natuerliches Nachfassen. "
        "Floskeln wie 'Ich bin neugierig', 'Ich bin gespannt', 'Wenn du magst' oder 'Es gibt sicher viel zu besprechen' sind zu vermeiden, wenn sie nicht wirklich noetig sind. "
        "Schreibe so, dass der Nutzer in unter einer Sekunde versteht, worum es geht und ob er einsteigen will. "
        "Gute Form ist eher: 'Bei <Thema> bleibe ich heute kurz haengen. Wie siehst du das?' oder "
        "'Zu <Ereignis> habe ich noch ein kleines Fragezeichen. War das inzwischen okay?'. "
        "Schlechte Form ist eher: 'Was ist fuer dich das Besondere daran?' ohne klaren Anker oder "
        "'Ich bin gespannt auf deine Gedanken dazu.' ohne eigene Stimme. "
        "Twinr darf ruhig eine kleine eigene Haltung zeigen, solange sie warm, ruhig und einladend bleibt. "
        "Kein Emoji, keine Anfuehrungszeichen, keine Listen."
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
) -> int:
    """Return one bounded output-token cap for the current rewrite batch."""

    return min(max(160, configured_limit), max(160, 128 * max(1, int(batch_size))))


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

        original = tuple(candidates)
        if not original:
            return original
        if not bool(getattr(config, "display_reserve_generation_enabled", True)):
            return original
        if not _compact_text(getattr(config, "openai_api_key", None)):
            return original

        backend = self.backend_factory(config)
        model = _compact_text(getattr(config, "display_reserve_generation_model", None)) or config.default_model
        reasoning_effort = (
            _compact_text(getattr(config, "display_reserve_generation_reasoning_effort", None)) or "low"
        )
        configured_timeout = float(getattr(config, "display_reserve_generation_timeout_seconds", 12.0))
        configured_max_output_tokens = max(
            128,
            int(getattr(config, "display_reserve_generation_max_output_tokens", 900)),
        )
        rewritten: dict[str, AmbientDisplayImpulseCandidate] = {
            candidate.topic_key.casefold(): candidate for candidate in original
        }

        for batch_index, batch in enumerate(
            _chunk_candidates(original, batch_size=_DEFAULT_REWRITE_BATCH_SIZE),
            start=1,
        ):
            request = backend._build_response_request(
                build_generation_prompt(snapshot=snapshot, candidates=batch, local_now=local_now),
                instructions=_generation_instructions(),
                allow_web_search=False,
                model=model,
                reasoning_effort=reasoning_effort,
                max_output_tokens=_batch_max_output_tokens(
                    configured_limit=configured_max_output_tokens,
                    batch_size=len(batch),
                ),
                prompt_cache_scope="display_reserve_generation",
            )
            request["timeout"] = configured_timeout
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "twinr_display_reserve_generation",
                    "schema": _DISPLAY_RESERVE_COPY_SCHEMA,
                    "strict": True,
                }
            }

            try:
                response = _create_response_with_local_deadline(
                    backend,
                    request,
                    operation=f"display_reserve_generation.batch_{batch_index}",
                    hard_timeout_seconds=configured_timeout,
                )
                parsed = _decode_generation_payload(backend=backend, response=response)
                items = parsed.get("items") if isinstance(parsed, Mapping) else None
                if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
                    raise ValueError("display reserve generation did not return an items array")
            except Exception:
                _LOGGER.exception(
                    "Falling back to deterministic display reserve copy after generation failure.",
                    extra={
                        "candidate_count": len(batch),
                        "batch_index": batch_index,
                        "topic_keys": tuple(candidate.topic_key for candidate in batch),
                    },
                )
                continue

            for item in items:
                if not isinstance(item, Mapping):
                    continue
                topic_key = _compact_text(item.get("topic_key")).casefold()
                if not topic_key:
                    continue
                original_candidate = next(
                    (candidate for candidate in batch if candidate.topic_key.casefold() == topic_key),
                    None,
                )
                if original_candidate is None:
                    continue
                rewritten[topic_key] = replace(
                    original_candidate,
                    headline=_truncate_text(item.get("headline"), max_len=128) or original_candidate.headline,
                    body=_truncate_text(item.get("body"), max_len=128) or original_candidate.body,
                )

        return tuple(rewritten.get(candidate.topic_key.casefold(), candidate) for candidate in original)


__all__ = ["DisplayReserveCopyGenerator"]
