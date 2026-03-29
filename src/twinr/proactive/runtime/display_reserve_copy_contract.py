"""Own reusable copy assets for reserve-card writer and judge prompts.

This module keeps stable family examples and rubric assets out of the runtime
batching/orchestration files. It also exposes a schema-versioned prompt
contract that is friendly to structured prompting, family-specific evaluation,
and prompt-caching-friendly stable prefixes.
"""

# CHANGELOG: 2026-03-29
# BUG-1: reserve_copy_examples_for_family() now normalizes aliases, mixed case,
#        whitespace, and separator variants instead of silently falling back to
#        "world" for inputs like "Memory", "memory-follow-up", or enum values.
# BUG-2: resolve_reserve_copy_family() now accepts explicit family overrides and
#        robustly normalizes spaced/hyphenated/enum metadata, avoiding silent
#        misrouting to the wrong family.
# SEC-1: Hardened routing against control characters, zero-width characters,
#        oversized metadata, accidental global asset mutation, and runtime
#        import coupling to heavier orchestration modules.
# IMP-1: Added a schema-versioned reserve_copy_contract_payload() with stable
#        ordering so prompts can keep static assets at the prefix for better
#        cache reuse and structured prompting.
# IMP-2: Added family-specific rubric overlays and upgraded all German gold
#        examples/rubrics to idiomatic UTF-8 text for higher-fidelity copy.

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate


_PROMPT_ASSETS_SCHEMA_VERSION: Final[str] = "2026-03-29.reserve-copy.v2"

_FAMILY_WORLD: Final[str] = "world"
_FAMILY_MEMORY: Final[str] = "memory"
_FAMILY_DISCOVERY: Final[str] = "discovery"
_FAMILY_REFLECTION: Final[str] = "reflection"
_DEFAULT_FAMILY: Final[str] = _FAMILY_WORLD

_ROUTING_TEXT_MAX_LEN: Final[int] = 96
_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")
_MULTI_UNDERSCORE_RE = re.compile(r"_+")


@runtime_checkable
class ReserveCopyCandidateLike(Protocol):
    """Structural type for the candidate attributes used in this module."""

    candidate_family: object | None
    source: object | None
    generation_context: object | None


def _extract_text(value: object | None) -> str:
    """Safely coerce simple text-like values without invoking arbitrary __str__."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    nested_value = getattr(value, "value", None)
    if isinstance(nested_value, str):
        return nested_value
    if isinstance(nested_value, bytes):
        return nested_value.decode("utf-8", errors="replace")
    if isinstance(value, (bool, int, float)):
        return str(value)
    return ""


def _strip_control_chars(text: str) -> str:
    """Replace control/format characters with spaces for safe routing."""

    if not text:
        return ""
    return "".join(
        " " if unicodedata.category(char).startswith("C") else char
        for char in text
    )


def _compact_text(value: object | None) -> str:
    """Collapse text into one trimmed single line with Unicode normalization."""

    text = unicodedata.normalize("NFKC", _extract_text(value))
    text = _strip_control_chars(text)
    text = " ".join(text.split()).strip()
    if len(text) > _ROUTING_TEXT_MAX_LEN:
        return text[:_ROUTING_TEXT_MAX_LEN].rstrip()
    return text


def _normalize_key(value: object | None) -> str:
    """Normalize routing keys across spaces, hyphens, case, and enum values."""

    text = _compact_text(value).casefold()
    if not text:
        return ""
    text = _NON_ALNUM_RE.sub("_", text)
    return _MULTI_UNDERSCORE_RE.sub("_", text).strip("_")


def _normalize_family_name(copy_family: object | None) -> str:
    """Normalize a family/alias into one canonical family name."""

    key = _normalize_key(copy_family)
    if not key:
        return _DEFAULT_FAMILY
    return _FAMILY_ALIASES.get(key, _DEFAULT_FAMILY)


@dataclass(frozen=True, slots=True)
class ReserveCopyExample:
    """Describe one small gold example for a reserve-card family."""

    use_for: str
    headline: str
    body: str

    def to_prompt_dict(self) -> dict[str, str]:
        """Serialize the example into one prompt-facing mapping."""

        return {
            "use_for": self.use_for,
            "headline": self.headline,
            "body": self.body,
        }


@dataclass(frozen=True, slots=True)
class ReserveCopyRubricCriterion:
    """Describe one explicit quality dimension for writer/judge prompts."""

    key: str
    question: str
    prefer: str
    avoid: str

    def to_prompt_dict(self) -> dict[str, str]:
        """Serialize the criterion into one prompt-facing mapping."""

        return {
            "key": self.key,
            "question": self.question,
            "prefer": self.prefer,
            "avoid": self.avoid,
        }


_FAMILY_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        # Canonical families
        _FAMILY_WORLD: _FAMILY_WORLD,
        _FAMILY_MEMORY: _FAMILY_MEMORY,
        _FAMILY_DISCOVERY: _FAMILY_DISCOVERY,
        _FAMILY_REFLECTION: _FAMILY_REFLECTION,
        # World/public-topic aliases
        "public": _FAMILY_WORLD,
        "public_topic": _FAMILY_WORLD,
        "public_topics": _FAMILY_WORLD,
        "world_topic": _FAMILY_WORLD,
        "news": _FAMILY_WORLD,
        "current_events": _FAMILY_WORLD,
        # Memory/follow-up aliases
        "follow_up": _FAMILY_MEMORY,
        "followup": _FAMILY_MEMORY,
        "memory_follow_up": _FAMILY_MEMORY,
        "memory_conflict": _FAMILY_MEMORY,
        "conflict": _FAMILY_MEMORY,
        "relationship": _FAMILY_MEMORY,
        # Discovery aliases
        "user_discovery": _FAMILY_DISCOVERY,
        "discover": _FAMILY_DISCOVERY,
        "invite_user_discovery": _FAMILY_DISCOVERY,
        # Reflection aliases
        "reflect": _FAMILY_REFLECTION,
        "call_back_to_earlier_conversation": _FAMILY_REFLECTION,
        "callback_to_earlier_conversation": _FAMILY_REFLECTION,
        "call_back": _FAMILY_REFLECTION,
        "callback": _FAMILY_REFLECTION,
    }
)

_FAMILY_EXAMPLES: Final[Mapping[str, tuple[ReserveCopyExample, ...]]] = MappingProxyType(
    {
        _FAMILY_WORLD: (
            ReserveCopyExample(
                use_for="Öffentliches Thema mit konkreter Beobachtung",
                headline="Ich habe heute etwas zu KI-Begleitern gelesen.",
                body="Was denkst du darüber?",
            ),
            ReserveCopyExample(
                use_for="Politische Meldung mit klarem Anlass",
                headline="In Berlin ist heute politisch einiges passiert.",
                body="Hast du davon schon gehört?",
            ),
            ReserveCopyExample(
                use_for="Öffentliche Debatte mit kleiner Einordnung",
                headline="Reparaturen sind heute wieder ein großes Thema.",
                body="Ist das für dich sinnvoll oder eher Symbolik?",
            ),
        ),
        _FAMILY_MEMORY: (
            ReserveCopyExample(
                use_for="Persönliches Nachfassen nach einem Ereignis",
                headline="Beim Arzttermin von gestern ist für mich noch etwas offen.",
                body="Wollen wir kurz darüber reden?",
            ),
            ReserveCopyExample(
                use_for="Sanfte Klärung bei zwei möglichen Versionen",
                headline="Ich habe dazu zwei Versionen im Kopf.",
                body="Magst du mir kurz sagen, was stimmt?",
            ),
            ReserveCopyExample(
                use_for="Ruhige Rückfrage zu einem persönlichen Faden",
                headline="Von letzter Woche ist für mich noch etwas offen.",
                body="Was ist dir dazu noch wichtig?",
            ),
        ),
        _FAMILY_DISCOVERY: (
            ReserveCopyExample(
                use_for="Natürliche Frage nach der richtigen Ansprache",
                headline="Ich möchte wissen, wie ich dich ansprechen soll.",
                body="Wie soll ich dich nennen?",
            ),
            ReserveCopyExample(
                use_for="Interesse an einer Alltagsgewohnheit",
                headline="Mich interessiert, was dir morgens gut tut.",
                body="Was gehört für dich dazu?",
            ),
            ReserveCopyExample(
                use_for="Freundliche Grenze oder Vorliebe kennenlernen",
                headline="Ich würde gern wissen, was Twinr besser lassen soll.",
                body="Was ist dir da wichtig?",
            ),
        ),
        _FAMILY_REFLECTION: (
            ReserveCopyExample(
                use_for="Ruhiger Rückbezug auf ein früheres Gespräch",
                headline="Ich denke noch an unser Gespräch über Twinr.",
                body="Was ist dir davon hängen geblieben?",
            ),
            ReserveCopyExample(
                use_for="Kleiner Anschluss an einen frischen konkreten Punkt",
                headline="Dein Gedanke zum Arzttermin ist mir geblieben.",
                body="Wollen wir da kurz weitermachen?",
            ),
            ReserveCopyExample(
                use_for="Rückfrage zu einem offenen Thema von gestern",
                headline="Ich habe dein Thema von gestern noch im Kopf.",
                body="Was ist seitdem dazu passiert?",
            ),
        ),
    }
)

_QUALITY_RUBRIC: Final[tuple[ReserveCopyRubricCriterion, ...]] = (
    ReserveCopyRubricCriterion(
        key="idiomatisches_deutsch",
        question="Klingt die Karte wie normales, spontanes Deutsch?",
        prefer="einfache, natürliche Sätze ohne Übersetzungs- oder Labelton",
        avoid="Mischsprache, UI-Wörter, Halbsätze oder holprige Formeln",
    ),
    ReserveCopyRubricCriterion(
        key="sofortige_klarheit",
        question="Ist nach der Headline sofort klar, worum es geht?",
        prefer="ein klar benannter Anlass mit frühem Themenanker",
        avoid="vage Stimmung, Etiketten oder ein versteckter Anlass erst in der Body-Zeile",
    ),
    ReserveCopyRubricCriterion(
        key="interaktionsreiz",
        question="Macht die Body-Zeile Lust, darauf zu reagieren?",
        prefer="eine echte Anschlussfrage oder Einladung zur Meinung",
        avoid="bloße Nettigkeit, leere CTA-Floskeln oder eine zweite Erklärzeile",
    ),
    ReserveCopyRubricCriterion(
        key="twinr_stimme",
        question="Klingt der Text ruhig, warm und leicht eigen wie Twinr?",
        prefer="aufmerksam, unaufgeregt, leise eigen, eher Begleiter als Service",
        avoid="Marketing, Coaching, Kundenservice oder übertriebene Innerlichkeit",
    ),
    ReserveCopyRubricCriterion(
        key="screen_tauglichkeit",
        question="Funktioniert die Karte als kurzer HDMI-Impuls aus dem Augenwinkel?",
        prefer="ein dominanter Gedanke, kurze Zeilen, schnell erfassbar",
        avoid="überladene Karten, doppelte Themen oder verschachtelte Sätze",
    ),
)

_FAMILY_RUBRIC_OVERLAYS: Final[Mapping[str, tuple[ReserveCopyRubricCriterion, ...]]] = (
    MappingProxyType(
        {
            _FAMILY_WORLD: (
                ReserveCopyRubricCriterion(
                    key="oeffentlicher_anlass",
                    question="Ist der öffentliche Anlass konkret genug, um sofort ein Bild zu erzeugen?",
                    prefer="ein klarer Beobachtungspunkt oder Tagesanlass",
                    avoid="abstrakte Weltlage, Generalisierung oder bloßes Schlagwort",
                ),
            ),
            _FAMILY_MEMORY: (
                ReserveCopyRubricCriterion(
                    key="erinnerungstreue",
                    question="Bleibt die Karte treu zu Bekanntem, ohne neue Fakten zu erfinden?",
                    prefer="vorsichtige Bezugnahme auf bereits Bekanntes",
                    avoid="ausgedachte Details, falsche Sicherheit oder neue Behauptungen",
                ),
            ),
            _FAMILY_DISCOVERY: (
                ReserveCopyRubricCriterion(
                    key="leicht_zu_beantworten",
                    question="Ist die Frage leicht und konkret genug, um spontan beantwortet zu werden?",
                    prefer="eine kleine, gut beantwortbare Frage",
                    avoid="zu breite Selbstreflexion oder mehrere Fragen auf einmal",
                ),
            ),
            _FAMILY_REFLECTION: (
                ReserveCopyRubricCriterion(
                    key="gespraechskontinuitaet",
                    question="Wirkt die Karte wie ein echter Anschluss an einen früheren Faden?",
                    prefer="klare Kontinuität mit sanfter Fortsetzung",
                    avoid="Themenbruch, Wiederholung ohne Fortschritt oder unnötige Erklärung",
                ),
            ),
        }
    )
)


@cache
def _serialized_examples_for_family(copy_family: str) -> tuple[dict[str, str], ...]:
    family = _normalize_family_name(copy_family)
    return tuple(example.to_prompt_dict() for example in _FAMILY_EXAMPLES[family])


@cache
def _serialized_base_rubric() -> tuple[dict[str, str], ...]:
    return tuple(criterion.to_prompt_dict() for criterion in _QUALITY_RUBRIC)


@cache
def _serialized_family_rubric(copy_family: str) -> tuple[dict[str, str], ...]:
    family = _normalize_family_name(copy_family)
    return tuple(
        criterion.to_prompt_dict()
        for criterion in _FAMILY_RUBRIC_OVERLAYS.get(family, ())
    )


def _copy_prompt_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """Return detached prompt rows so callers cannot mutate module globals."""

    return [dict(row) for row in rows]


def _families_in_batch(
    candidates: Sequence["AmbientDisplayImpulseCandidate | ReserveCopyCandidateLike"],
) -> tuple[str, ...]:
    """Resolve and sort unique families present in a batch."""

    families = {resolve_reserve_copy_family(candidate) for candidate in candidates}
    return tuple(sorted(families))


def resolve_reserve_copy_family(
    candidate: "AmbientDisplayImpulseCandidate | ReserveCopyCandidateLike",
) -> str:
    """Map one reserve candidate onto the shared prompt example families."""

    raw_context = getattr(candidate, "generation_context", None)
    context = raw_context if isinstance(raw_context, Mapping) else {}

    for key in ("reserve_copy_family", "copy_family", "prompt_copy_family"):
        explicit_family = _normalize_key(context.get(key))
        if explicit_family in _FAMILY_ALIASES:
            return _FAMILY_ALIASES[explicit_family]

    display_goal = _normalize_key(context.get("display_goal"))
    candidate_family = _normalize_key(getattr(candidate, "candidate_family", None))
    source = _normalize_key(getattr(candidate, "source", None))

    if display_goal in _FAMILY_ALIASES:
        return _FAMILY_ALIASES[display_goal]
    if candidate_family in _FAMILY_ALIASES:
        return _FAMILY_ALIASES[candidate_family]
    if source in _FAMILY_ALIASES:
        return _FAMILY_ALIASES[source]

    candidate_family_tokens = set(candidate_family.split("_")) if candidate_family else set()
    source_tokens = set(source.split("_")) if source else set()

    if (
        "discovery" in candidate_family_tokens
        or "discovery" in source_tokens
        or source == "user_discovery"
    ):
        return _FAMILY_DISCOVERY
    if (
        "reflection" in candidate_family_tokens
        or "reflection" in source_tokens
        or source.startswith("reflection")
    ):
        return _FAMILY_REFLECTION
    if (
        "memory" in candidate_family_tokens
        or "memory" in source_tokens
        or "follow" in candidate_family_tokens
        or "follow" in source_tokens
        or "conflict" in candidate_family_tokens
        or "conflict" in source_tokens
        or source.startswith("memory")
        or source == "relationship"
    ):
        return _FAMILY_MEMORY
    return _FAMILY_WORLD


def reserve_copy_examples_for_family(copy_family: str) -> tuple[ReserveCopyExample, ...]:
    """Return the small gold example set for one family or accepted alias."""

    return _FAMILY_EXAMPLES[_normalize_family_name(copy_family)]


def reserve_copy_examples_payload(
    candidates: Sequence["AmbientDisplayImpulseCandidate | ReserveCopyCandidateLike"],
) -> dict[str, list[dict[str, str]]]:
    """Serialize family examples for the families present in the batch."""

    return {
        family: _copy_prompt_rows(_serialized_examples_for_family(family))
        for family in _families_in_batch(candidates)
    }


def reserve_copy_family_rubric_payload(copy_family: str) -> list[dict[str, str]]:
    """Return the family-specific rubric overlay for one family."""

    return _copy_prompt_rows(_serialized_family_rubric(copy_family))


def reserve_copy_rubric_payload(copy_family: str | None = None) -> list[dict[str, str]]:
    """Return the shared rubric, optionally extended with a family overlay."""

    rubric = _copy_prompt_rows(_serialized_base_rubric())
    if copy_family is None:
        return rubric
    rubric.extend(reserve_copy_family_rubric_payload(copy_family))
    return rubric


def reserve_copy_contract_payload(
    candidates: Sequence["AmbientDisplayImpulseCandidate | ReserveCopyCandidateLike"],
    *,
    include_family_rubric_overlays: bool = True,
    default_family_on_empty: str | None = _DEFAULT_FAMILY,
) -> dict[str, Any]:
    """Return a schema-versioned prompt contract for reserve-card prompting."""

    families = _families_in_batch(candidates)
    if not families and default_family_on_empty is not None:
        families = (_normalize_family_name(default_family_on_empty),)

    payload: dict[str, Any] = {
        "schema_version": _PROMPT_ASSETS_SCHEMA_VERSION,
        "families": list(families),
        "examples_by_family": {
            family: _copy_prompt_rows(_serialized_examples_for_family(family))
            for family in families
        },
        "quality_rubric": _copy_prompt_rows(_serialized_base_rubric()),
    }
    if include_family_rubric_overlays:
        family_rubric_overlays: dict[str, list[dict[str, str]]] = {}
        for family in families:
            overlay = _serialized_family_rubric(family)
            if overlay:
                family_rubric_overlays[family] = _copy_prompt_rows(overlay)
        if family_rubric_overlays:
            payload["family_rubric_overlays"] = family_rubric_overlays
    return payload


__all__ = [
    "ReserveCopyCandidateLike",
    "ReserveCopyExample",
    "ReserveCopyRubricCriterion",
    "reserve_copy_contract_payload",
    "reserve_copy_examples_for_family",
    "reserve_copy_examples_payload",
    "reserve_copy_family_rubric_payload",
    "reserve_copy_rubric_payload",
    "resolve_reserve_copy_family",
]