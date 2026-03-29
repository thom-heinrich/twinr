# CHANGELOG: 2026-03-29
# BUG-1: Fixed fallback parsing of fenced JSON responses. The old path compacted
#        text before stripping markdown fences, so valid ```json ... ``` payloads
#        could fail closed even though a usable structured object was present.
# BUG-2: Fixed credential detection so generation no longer disables itself when
#        the API key is supplied via OPENAI_API_KEY in the environment.
# BUG-3: Added duplicate topic-key detection per batch to prevent ambiguous
#        rewrite assignment from silently corrupting final reserve-card copy.
# SEC-1: Set store=False on all reserve-copy Responses API calls so personal
#        snapshot/context payloads are not retained by default by the provider.
# SEC-2: Bounded in-flight local-deadline worker threads to stop repeated slow
#        upstream calls from piling up and exhausting a Raspberry Pi deployment.
# IMP-1: Added separate writer/judge model routing, reasoning, and verbosity
#        controls so the rewrite path can use 2026-era specialized generation.
# IMP-2: Tightened schemas and post-validation to enforce display-aware line
#        lengths, improving readability from several meters on HDMI reserve lane.
# IMP-3: Added bounded parallel batch execution plus richer per-attempt token/
#        cache telemetry for eval harnesses and production latency tuning.

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
from concurrent import futures
from contextvars import copy_context
from dataclasses import dataclass, replace
from datetime import datetime
import json
import logging
import os
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
_DEFAULT_MAX_PARALLEL_BATCHES = 1
_DEFAULT_VARIANTS_PER_CANDIDATE = 3
_DEFAULT_HEADLINE_MAX_CHARS = 56
_DEFAULT_BODY_MAX_CHARS = 72
_DEFAULT_WRITER_VERBOSITY = "low"
_DEFAULT_JUDGE_VERBOSITY = "low"
_DEFAULT_LOCAL_DEADLINE_MAX_INFLIGHT = 4
_VARIANT_STAGE_OUTPUT_TOKENS_PER_CANDIDATE = 320
_SELECTION_STAGE_OUTPUT_TOKENS_PER_CANDIDATE = 160
_LOCAL_DEADLINE_WORKER_SLOTS = threading.BoundedSemaphore(
    _DEFAULT_LOCAL_DEADLINE_MAX_INFLIGHT
)

_T = TypeVar("_T")


def _display_reserve_copy_properties(
    *,
    headline_max_chars: int,
    body_max_chars: int,
) -> dict[str, object]:
    """Return the visible copy schema properties for one reserve card."""

    return {
        "headline": {
            "type": "string",
            "minLength": 1,
            "maxLength": max(8, int(headline_max_chars)),
        },
        "body": {
            "type": "string",
            "minLength": 1,
            "maxLength": max(8, int(body_max_chars)),
        },
    }


def _display_reserve_copy_item_schema(
    *,
    headline_max_chars: int,
    body_max_chars: int,
) -> dict[str, object]:
    """Return the structured schema for one visible copy object."""

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": _display_reserve_copy_properties(
            headline_max_chars=headline_max_chars,
            body_max_chars=body_max_chars,
        ),
        "required": ["headline", "body"],
    }


def _display_reserve_copy_schema(
    *,
    headline_max_chars: int,
    body_max_chars: int,
) -> dict[str, object]:
    """Return the structured schema for final selected reserve copy."""

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
                        "topic_key": {"type": "string", "minLength": 1},
                        **_display_reserve_copy_properties(
                            headline_max_chars=headline_max_chars,
                            body_max_chars=body_max_chars,
                        ),
                    },
                    "required": ["topic_key", "headline", "body"],
                },
            }
        },
        "required": ["items"],
    }


def _display_reserve_variant_schema(
    *,
    variants_per_candidate: int,
    headline_max_chars: int,
    body_max_chars: int,
) -> dict[str, object]:
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
                        "topic_key": {"type": "string", "minLength": 1},
                        "variants": {
                            "type": "array",
                            "minItems": variant_count,
                            "maxItems": variant_count,
                            "items": _display_reserve_copy_item_schema(
                                headline_max_chars=headline_max_chars,
                                body_max_chars=body_max_chars,
                            ),
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
        keys_text = ", ".join(
            _compact_text(key) for key in topic_keys if _compact_text(key)
        )
        message = (
            f"display reserve generation failed closed for batch {batch_index}"
            f" using model {model or '<unknown>'}"
        )
        if keys_text:
            message = f"{message} ({keys_text})"
        super().__init__(message)
        self.batch_index = int(batch_index)
        self.model = _compact_text(model)
        self.topic_keys = tuple(
            _compact_text(key) for key in topic_keys if _compact_text(key)
        )


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
    cached_prompt_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


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
    writer_model: str = ""
    judge_model: str = ""
    writer_reasoning_effort: str = ""
    judge_reasoning_effort: str = ""
    max_parallel_batches: int = 1


@dataclass(frozen=True, slots=True)
class _BatchRewriteResult:
    """Hold one completed reserve-copy batch rewrite."""

    batch_index: int
    rewritten: dict[str, AmbientDisplayImpulseCandidate]
    stage_attempts: tuple[DisplayReserveStageAttemptTrace, ...]
    selections: tuple[DisplayReserveSelectionTrace, ...]


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


def _getenv_text(name: str) -> str:
    """Return one compact environment variable value."""

    return _compact_text(os.getenv(name))


def _has_openai_credentials(config: TwinrConfig) -> bool:
    """Return whether reserve-copy generation appears to have API credentials."""

    if _compact_text(getattr(config, "openai_api_key", None)):
        return True
    return bool(_getenv_text("OPENAI_API_KEY"))


def _positive_int(
    value: object | None,
    *,
    default: int,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    """Normalize one positive integer configuration value."""

    try:
        normalized = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        normalized = int(default)
    normalized = max(int(minimum), normalized)
    if maximum is not None:
        normalized = min(int(maximum), normalized)
    return normalized


def _positive_float(
    value: object | None,
    *,
    default: float,
    minimum: float = 0.25,
    maximum: float | None = None,
) -> float:
    """Normalize one positive float configuration value."""

    try:
        normalized = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        normalized = float(default)
    normalized = max(float(minimum), normalized)
    if maximum is not None:
        normalized = min(float(maximum), normalized)
    return normalized


def _normalized_verbosity(value: object | None, *, default: str = "") -> str:
    """Return a supported text verbosity hint or an empty string."""

    compact = _compact_text(value).casefold()
    if compact in {"low", "medium", "high"}:
        return compact
    default_compact = _compact_text(default).casefold()
    return default_compact if default_compact in {"low", "medium", "high"} else ""


def _supports_text_verbosity(model: str) -> bool:
    """Return whether the current model likely supports text verbosity hints."""

    normalized = _compact_text(model).casefold()
    return normalized.startswith("gpt-5")


def _maybe_text_verbosity(
    model: str, configured_value: object | None, *, default: str
) -> str:
    """Return a text verbosity hint only when the target model likely supports it."""

    if not _supports_text_verbosity(model):
        return ""
    return _normalized_verbosity(configured_value, default=default)


def _safe_get(source: object | None, *path: str) -> object | None:
    """Read nested mapping or attribute values without raising."""

    current = source
    for key in path:
        if current is None:
            return None
        if isinstance(current, Mapping):
            current = current.get(key)
            continue
        current = getattr(current, key, None)
    return current


def _safe_int(value: object | None) -> int:
    """Coerce a numeric SDK field into int when possible."""

    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _response_usage_stats(response: object | None) -> tuple[int, int, int]:
    """Return input tokens, output tokens, and cached prompt tokens."""

    usage = getattr(response, "usage", None)
    input_tokens = _safe_int(_safe_get(usage, "input_tokens"))
    if input_tokens <= 0:
        input_tokens = _safe_int(_safe_get(usage, "prompt_tokens"))
    output_tokens = _safe_int(_safe_get(usage, "output_tokens"))
    if output_tokens <= 0:
        output_tokens = _safe_int(_safe_get(usage, "completion_tokens"))
    cached_prompt_tokens = _safe_int(
        _safe_get(usage, "input_tokens_details", "cached_tokens")
    )
    if cached_prompt_tokens <= 0:
        cached_prompt_tokens = _safe_int(
            _safe_get(usage, "prompt_tokens_details", "cached_tokens")
        )
    return input_tokens, output_tokens, cached_prompt_tokens


def _strip_json_fence(payload_text: str) -> str:
    """Remove one surrounding markdown code fence from a JSON payload.

    Some models still return fenced JSON text even when a structured response
    was requested. The reserve lane should accept that bounded variant instead
    of discarding the whole batch.
    """

    stripped = payload_text.strip()
    if not stripped.startswith("```") or not stripped.endswith("```"):
        return stripped
    inner = stripped[3:-3].strip()
    if inner.casefold().startswith("json"):
        inner = inner[4:].lstrip()
    return inner


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
    payload_text = _strip_json_fence(str(backend._extract_output_text(response) or "").strip())
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


def _truncate_text(value: object | None, *, max_len: int) -> str:
    """Return one bounded single-line string with readable truncation."""

    compact = _compact_text(value).strip(' "\'„“”‚‘’')
    if len(compact) <= max_len:
        return compact
    cutoff = compact.rfind(" ", 0, max_len - 1)
    if cutoff >= max(10, max_len // 2):
        compact = compact[:cutoff]
    else:
        compact = compact[: max_len - 1]
    return compact.rstrip(" ,;:-.") + "…"


def _ensure_unique_batch_topic_keys(
    batch: Sequence[AmbientDisplayImpulseCandidate],
) -> None:
    """Reject one batch when topic keys are not unique after normalization."""

    seen: set[str] = set()
    duplicates: list[str] = []
    for candidate in batch:
        normalized = _compact_text(candidate.topic_key).casefold()
        if not normalized:
            raise ValueError(
                "display reserve generation candidate is missing topic_key"
            )
        if normalized in seen:
            duplicates.append(candidate.topic_key)
            continue
        seen.add(normalized)
    if duplicates:
        raise ValueError(
            "display reserve generation batch contains duplicate topic_keys: "
            + ", ".join(
                _compact_text(value) for value in duplicates if _compact_text(value)
            )
        )


def _validated_generation_items(
    *,
    items: Sequence[object],
    batch: Sequence[AmbientDisplayImpulseCandidate],
    headline_max_chars: int,
    body_max_chars: int,
) -> dict[str, dict[str, str]]:
    """Validate that one batch returned complete usable copy for each topic."""

    expected_keys = {
        candidate.topic_key.casefold(): candidate for candidate in batch
    }
    validated: dict[str, dict[str, str]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("display reserve generation item must be an object")
        topic_key = _compact_text(item.get("topic_key")).casefold()
        if not topic_key:
            raise ValueError("display reserve generation item is missing topic_key")
        if topic_key not in expected_keys:
            raise ValueError(
                f"display reserve generation returned unknown topic_key {topic_key!r}"
            )
        if topic_key in validated:
            raise ValueError(
                f"display reserve generation returned duplicate topic_key {topic_key!r}"
            )
        headline = _truncate_text(item.get("headline"), max_len=headline_max_chars)
        body = _truncate_text(item.get("body"), max_len=body_max_chars)
        if not headline:
            raise ValueError(
                f"display reserve generation returned empty headline for {topic_key!r}"
            )
        if not body:
            raise ValueError(
                f"display reserve generation returned empty body for {topic_key!r}"
            )
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
    headline_max_chars: int,
    body_max_chars: int,
) -> dict[str, tuple[dict[str, str], ...]]:
    """Validate the first-pass variant payload for one reserve batch."""

    expected_keys = {
        candidate.topic_key.casefold(): candidate for candidate in batch
    }
    expected_variant_count = max(1, int(variants_per_candidate))
    validated: dict[str, tuple[dict[str, str], ...]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError(
                "display reserve generation variant item must be an object"
            )
        topic_key = _compact_text(item.get("topic_key")).casefold()
        if not topic_key:
            raise ValueError(
                "display reserve generation variant item is missing topic_key"
            )
        if topic_key not in expected_keys:
            raise ValueError(
                f"display reserve generation returned unknown topic_key {topic_key!r}"
            )
        if topic_key in validated:
            raise ValueError(
                f"display reserve generation returned duplicate topic_key {topic_key!r}"
            )
        raw_variants = item.get("variants")
        if not isinstance(raw_variants, Sequence) or isinstance(
            raw_variants, (str, bytes, bytearray)
        ):
            raise ValueError(
                f"display reserve generation returned no variants array for {topic_key!r}"
            )
        variants: list[dict[str, str]] = []
        for raw_variant in raw_variants:
            if not isinstance(raw_variant, Mapping):
                raise ValueError(
                    f"display reserve generation returned a non-object variant for {topic_key!r}"
                )
            headline = _truncate_text(
                raw_variant.get("headline"), max_len=headline_max_chars
            )
            body = _truncate_text(raw_variant.get("body"), max_len=body_max_chars)
            if not headline:
                raise ValueError(
                    f"display reserve generation returned empty variant headline for {topic_key!r}"
                )
            if not body:
                raise ValueError(
                    f"display reserve generation returned empty variant body for {topic_key!r}"
                )
            variants.append(
                {
                    "headline": headline,
                    "body": body,
                }
            )
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
    failure: Exception | None,
    attempt_index: int,
    attempt_count: int,
) -> bool:
    """Return whether one failed batch deserves a second bounded LLM attempt."""

    if attempt_index >= attempt_count:
        return False
    if _incomplete_reason(response) == "max_output_tokens":
        return True
    if isinstance(failure, (TimeoutError, ConnectionError)):
        return True
    return False


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

    acquire_timeout = max(0.25, float(hard_timeout_seconds))
    if not _LOCAL_DEADLINE_WORKER_SLOTS.acquire(timeout=acquire_timeout):
        raise TimeoutError(
            "display reserve generation could not acquire a local deadline worker slot"
        )

    result_holder: dict[str, object] = {}
    error_holder: dict[str, BaseException] = {}
    current_context = copy_context()

    def _worker_call() -> None:
        result_holder["response"] = backend._create_response(
            dict(request), operation=operation
        )

    def _worker() -> None:
        try:
            current_context.run(_worker_call)
        except BaseException as exc:  # pragma: no cover - re-raised synchronously below.
            error_holder["error"] = exc
        finally:
            _LOCAL_DEADLINE_WORKER_SLOTS.release()

    worker = threading.Thread(
        target=_worker,
        name=f"display-reserve-generation-{operation}",
        daemon=True,
    )
    worker.start()
    worker.join(acquire_timeout)
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
    verbosity: str,
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
    retry_due_to_truncation = False
    for attempt_index in range(1, 3):
        retrying = attempt_index > 1
        request = backend._build_response_request(
            prompt,
            instructions=instructions,
            allow_web_search=False,
            model=model,
            reasoning_effort="none" if retry_due_to_truncation else reasoning_effort,
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
        request["store"] = False
        request["text"] = {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": dict(schema),
                "strict": True,
            }
        }
        if verbosity:
            text_config = dict(request["text"])
            text_config["verbosity"] = verbosity
            request["text"] = text_config

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
            if not isinstance(items, Sequence) or isinstance(
                items, (str, bytes, bytearray)
            ):
                raise ValueError(
                    f"display reserve {stage_name} did not return an items array"
                )
            input_tokens, output_tokens, cached_prompt_tokens = _response_usage_stats(
                response
            )
            if stage_attempt_traces is not None:
                stage_attempt_traces.append(
                    DisplayReserveStageAttemptTrace(
                        batch_index=batch_index,
                        stage_name=stage_name,
                        attempt_index=attempt_index,
                        topic_keys=tuple(topic_keys),
                        max_output_tokens=int(request.get("max_output_tokens") or 0),
                        reasoning_effort=_compact_text(
                            request.get("reasoning_effort")
                        ),
                        duration_seconds=time.perf_counter() - attempt_started,
                        succeeded=True,
                        incomplete_reason=_incomplete_reason(response),
                        cached_prompt_tokens=cached_prompt_tokens,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        model=model,
                    )
                )
            return validate_items(tuple(items))
        except Exception as exc:
            failure = exc
            input_tokens, output_tokens, cached_prompt_tokens = _response_usage_stats(
                response
            )
            if stage_attempt_traces is not None:
                stage_attempt_traces.append(
                    DisplayReserveStageAttemptTrace(
                        batch_index=batch_index,
                        stage_name=stage_name,
                        attempt_index=attempt_index,
                        topic_keys=tuple(topic_keys),
                        max_output_tokens=int(request.get("max_output_tokens") or 0),
                        reasoning_effort=_compact_text(
                            request.get("reasoning_effort")
                        ),
                        duration_seconds=time.perf_counter() - attempt_started,
                        succeeded=False,
                        incomplete_reason=_incomplete_reason(response),
                        cached_prompt_tokens=cached_prompt_tokens,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        model=model,
                    )
                )
            if _should_retry_generation_batch(
                response=response,
                failure=exc,
                attempt_index=attempt_index,
                attempt_count=2,
            ):
                retry_due_to_truncation = (
                    _incomplete_reason(response) == "max_output_tokens"
                )
                _LOGGER.warning(
                    "Retrying display reserve %s after a bounded stage failure.",
                    stage_name,
                    extra={
                        "batch_index": batch_index,
                        "attempt_index": attempt_index,
                        "topic_keys": tuple(topic_keys),
                        "incomplete_reason": _incomplete_reason(response),
                        "max_output_tokens": request.get("max_output_tokens"),
                        "stage_name": stage_name,
                        "model": model,
                        "retry_due_to_truncation": retry_due_to_truncation,
                        "failure_type": type(exc).__name__,
                    },
                )
                continue
            raise
    raise failure or RuntimeError(
        f"display reserve {stage_name} failed without a captured exception"
    )


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
        base_model = (
            _compact_text(getattr(config, "display_reserve_generation_model", None))
            or _compact_text(getattr(config, "default_model", None))
        )
        configured_reasoning_effort = (
            _compact_text(
                getattr(config, "display_reserve_generation_reasoning_effort", None)
            )
            or "low"
        )
        writer_model = (
            _compact_text(getattr(config, "display_reserve_generation_writer_model", None))
            or base_model
        )
        judge_model = (
            _compact_text(getattr(config, "display_reserve_generation_judge_model", None))
            or writer_model
        )
        writer_reasoning_effort = (
            _compact_text(
                getattr(config, "display_reserve_generation_writer_reasoning_effort", None)
            )
            or configured_reasoning_effort
        )
        judge_reasoning_effort = (
            _compact_text(
                getattr(config, "display_reserve_generation_judge_reasoning_effort", None)
            )
            or configured_reasoning_effort
        )
        batch_size = _positive_int(
            getattr(config, "display_reserve_generation_batch_size", _DEFAULT_REWRITE_BATCH_SIZE),
            default=_DEFAULT_REWRITE_BATCH_SIZE,
            minimum=1,
            maximum=16,
        )
        max_parallel_batches = _positive_int(
            getattr(
                config,
                "display_reserve_generation_max_parallel_batches",
                _DEFAULT_MAX_PARALLEL_BATCHES,
            ),
            default=_DEFAULT_MAX_PARALLEL_BATCHES,
            minimum=1,
            maximum=8,
        )
        variants_per_candidate = _positive_int(
            getattr(
                config,
                "display_reserve_generation_variants_per_candidate",
                _DEFAULT_VARIANTS_PER_CANDIDATE,
            ),
            default=_DEFAULT_VARIANTS_PER_CANDIDATE,
            minimum=1,
            maximum=6,
        )
        headline_max_chars = _positive_int(
            getattr(
                config,
                "display_reserve_generation_headline_max_chars",
                _DEFAULT_HEADLINE_MAX_CHARS,
            ),
            default=_DEFAULT_HEADLINE_MAX_CHARS,
            minimum=24,
            maximum=128,
        )
        body_max_chars = _positive_int(
            getattr(
                config,
                "display_reserve_generation_body_max_chars",
                _DEFAULT_BODY_MAX_CHARS,
            ),
            default=_DEFAULT_BODY_MAX_CHARS,
            minimum=24,
            maximum=160,
        )
        writer_verbosity = _maybe_text_verbosity(
            writer_model,
            getattr(config, "display_reserve_generation_writer_verbosity", None),
            default=_DEFAULT_WRITER_VERBOSITY,
        )
        judge_verbosity = _maybe_text_verbosity(
            judge_model,
            getattr(config, "display_reserve_generation_judge_verbosity", None),
            default=_DEFAULT_JUDGE_VERBOSITY,
        )

        if not original:
            return original, DisplayReserveGenerationTrace(
                model=base_model,
                reasoning_effort=configured_reasoning_effort,
                batch_size=batch_size,
                variants_per_candidate=variants_per_candidate,
                duration_seconds=time.perf_counter() - started,
                bypassed=True,
                bypass_reason="no_candidates",
                writer_model=writer_model,
                judge_model=judge_model,
                writer_reasoning_effort=writer_reasoning_effort,
                judge_reasoning_effort=judge_reasoning_effort,
                max_parallel_batches=max_parallel_batches,
            )
        if not bool(getattr(config, "display_reserve_generation_enabled", True)):
            return original, DisplayReserveGenerationTrace(
                model=base_model,
                reasoning_effort=configured_reasoning_effort,
                batch_size=batch_size,
                variants_per_candidate=variants_per_candidate,
                duration_seconds=time.perf_counter() - started,
                bypassed=True,
                bypass_reason="generation_disabled",
                writer_model=writer_model,
                judge_model=judge_model,
                writer_reasoning_effort=writer_reasoning_effort,
                judge_reasoning_effort=judge_reasoning_effort,
                max_parallel_batches=max_parallel_batches,
            )
        if not _has_openai_credentials(config):
            return original, DisplayReserveGenerationTrace(
                model=base_model,
                reasoning_effort=configured_reasoning_effort,
                batch_size=batch_size,
                variants_per_candidate=variants_per_candidate,
                duration_seconds=time.perf_counter() - started,
                bypassed=True,
                bypass_reason="missing_openai_credentials",
                writer_model=writer_model,
                judge_model=judge_model,
                writer_reasoning_effort=writer_reasoning_effort,
                judge_reasoning_effort=judge_reasoning_effort,
                max_parallel_batches=max_parallel_batches,
            )
        if not writer_model or not judge_model:
            return original, DisplayReserveGenerationTrace(
                model=base_model,
                reasoning_effort=configured_reasoning_effort,
                batch_size=batch_size,
                variants_per_candidate=variants_per_candidate,
                duration_seconds=time.perf_counter() - started,
                bypassed=True,
                bypass_reason="missing_model",
                writer_model=writer_model,
                judge_model=judge_model,
                writer_reasoning_effort=writer_reasoning_effort,
                judge_reasoning_effort=judge_reasoning_effort,
                max_parallel_batches=max_parallel_batches,
            )

        configured_timeout = _positive_float(
            getattr(config, "display_reserve_generation_timeout_seconds", 20.0),
            default=20.0,
            minimum=1.0,
            maximum=120.0,
        )
        configured_max_output_tokens = max(
            128,
            int(getattr(config, "display_reserve_generation_max_output_tokens", 900)),
        )
        rewritten: dict[str, AmbientDisplayImpulseCandidate] = {
            candidate.topic_key.casefold(): candidate for candidate in original
        }

        batches = _chunk_candidates(original, batch_size=batch_size)

        def _rewrite_batch(
            batch_index: int,
            batch: tuple[AmbientDisplayImpulseCandidate, ...],
        ) -> _BatchRewriteResult:
            topic_keys = tuple(candidate.topic_key for candidate in batch)
            try:
                _ensure_unique_batch_topic_keys(batch)
                backend = self.backend_factory(config)
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
                stage_attempt_traces: list[DisplayReserveStageAttemptTrace] = []
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
                        headline_max_chars=headline_max_chars,
                        body_max_chars=body_max_chars,
                    ),
                    schema_name="twinr_display_reserve_generation_variants",
                    model=writer_model,
                    reasoning_effort=writer_reasoning_effort,
                    verbosity=writer_verbosity,
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
                        headline_max_chars=headline_max_chars,
                        body_max_chars=body_max_chars,
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
                    schema=_display_reserve_copy_schema(
                        headline_max_chars=headline_max_chars,
                        body_max_chars=body_max_chars,
                    ),
                    schema_name="twinr_display_reserve_generation_selection",
                    model=judge_model,
                    reasoning_effort=judge_reasoning_effort,
                    verbosity=judge_verbosity,
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
                        headline_max_chars=headline_max_chars,
                        body_max_chars=body_max_chars,
                    ),
                    stage_attempt_traces=stage_attempt_traces,
                )

                rewritten_batch: dict[str, AmbientDisplayImpulseCandidate] = {}
                selection_traces: list[DisplayReserveSelectionTrace] = []
                for candidate in batch:
                    topic_key = candidate.topic_key.casefold()
                    rewrite = batch_rewrites.get(topic_key)
                    if rewrite is None:
                        raise DisplayReserveGenerationError(
                            batch_index=batch_index,
                            model=judge_model,
                            topic_keys=tuple(candidate.topic_key for candidate in batch),
                        )
                    rewritten_batch[topic_key] = replace(
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
                                    headline=_truncate_text(
                                        variant.get("headline"),
                                        max_len=headline_max_chars,
                                    ),
                                    body=_truncate_text(
                                        variant.get("body"),
                                        max_len=body_max_chars,
                                    ),
                                )
                                for variant in batch_variants.get(topic_key, ())
                            ),
                            final_copy=DisplayReserveTraceCopy(
                                headline=rewrite["headline"],
                                body=rewrite["body"],
                            ),
                        )
                    )
                return _BatchRewriteResult(
                    batch_index=batch_index,
                    rewritten=rewritten_batch,
                    stage_attempts=tuple(stage_attempt_traces),
                    selections=tuple(selection_traces),
                )
            except DisplayReserveGenerationError:
                raise
            except Exception as failure:
                _LOGGER.exception(
                    "Display reserve generation failed closed during double pass.",
                    extra={
                        "candidate_count": len(batch),
                        "batch_index": batch_index,
                        "topic_keys": topic_keys,
                        "model": judge_model or writer_model or base_model,
                        "writer_model": writer_model,
                        "judge_model": judge_model,
                        "batch_size": batch_size,
                        "variants_per_candidate": variants_per_candidate,
                    },
                    exc_info=failure,
                )
                cause = failure or RuntimeError(
                    "display reserve generation failed without a captured exception"
                )
                raise DisplayReserveGenerationError(
                    batch_index=batch_index,
                    model=judge_model or writer_model or base_model,
                    topic_keys=topic_keys,
                ) from cause

        batch_results: list[_BatchRewriteResult] = []
        if len(batches) <= 1 or max_parallel_batches <= 1:
            for batch_index, batch in enumerate(batches, start=1):
                batch_results.append(_rewrite_batch(batch_index, batch))
        else:
            with futures.ThreadPoolExecutor(
                max_workers=max_parallel_batches,
                thread_name_prefix="display-reserve-batch",
            ) as executor:
                for window_start in range(0, len(batches), max_parallel_batches):
                    window = list(
                        enumerate(
                            batches[window_start : window_start + max_parallel_batches],
                            start=window_start + 1,
                        )
                    )
                    future_to_index: dict[futures.Future[_BatchRewriteResult], int] = {}
                    for batch_index, batch in window:
                        batch_context = copy_context()
                        future = executor.submit(
                            batch_context.run, _rewrite_batch, batch_index, batch
                        )
                        future_to_index[future] = batch_index
                    try:
                        for future in futures.as_completed(future_to_index):
                            batch_results.append(future.result())
                    except Exception:
                        for future in future_to_index:
                            future.cancel()
                        raise

        stage_attempt_traces: list[DisplayReserveStageAttemptTrace] = []
        selection_traces: list[DisplayReserveSelectionTrace] = []
        for result in sorted(batch_results, key=lambda item: item.batch_index):
            rewritten.update(result.rewritten)
            stage_attempt_traces.extend(result.stage_attempts)
            selection_traces.extend(result.selections)

        return (
            tuple(
                rewritten.get(candidate.topic_key.casefold(), candidate)
                for candidate in original
            ),
            DisplayReserveGenerationTrace(
                model=base_model,
                reasoning_effort=configured_reasoning_effort,
                batch_size=batch_size,
                variants_per_candidate=variants_per_candidate,
                duration_seconds=time.perf_counter() - started,
                stage_attempts=tuple(stage_attempt_traces),
                selections=tuple(selection_traces),
                writer_model=writer_model,
                judge_model=judge_model,
                writer_reasoning_effort=writer_reasoning_effort,
                judge_reasoning_effort=judge_reasoning_effort,
                max_parallel_batches=max_parallel_batches,
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
