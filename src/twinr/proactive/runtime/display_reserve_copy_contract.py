"""Own reusable copy assets for reserve-card writer and judge prompts.

This module keeps stable family examples and the shared quality rubric out of
the runtime batching/orchestration files. Prompting can then attach a compact
positive copy contract to each batch instead of growing one larger monolithic
style instruction string.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate


def _compact_text(value: object | None) -> str:
    """Collapse arbitrary text into one trimmed single line."""

    return " ".join(str(value or "").split()).strip()


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


_FAMILY_EXAMPLES: dict[str, tuple[ReserveCopyExample, ...]] = {
    "world": (
        ReserveCopyExample(
            use_for="oeffentliches Thema mit konkreter Beobachtung",
            headline="Ich habe heute etwas zu KI-Begleitern gelesen.",
            body="Was denkst du darueber?",
        ),
        ReserveCopyExample(
            use_for="politische Meldung mit klarem Anlass",
            headline="In Berlin ist heute politisch einiges passiert.",
            body="Hast du davon schon gehoert?",
        ),
        ReserveCopyExample(
            use_for="oeffentliche Debatte mit kleiner Einordnung",
            headline="Reparaturen sind heute wieder ein grosses Thema.",
            body="Ist das fuer dich sinnvoll oder eher Symbolik?",
        ),
    ),
    "memory": (
        ReserveCopyExample(
            use_for="persoenliches Nachfassen nach einem Ereignis",
            headline="Beim Arzttermin von gestern fehlt mir noch etwas.",
            body="Wollen wir kurz darueber reden?",
        ),
        ReserveCopyExample(
            use_for="sanfte Klaerung bei zwei moeglichen Versionen",
            headline="Ich habe da zwei Versionen im Kopf.",
            body="Magst du mir kurz sagen, was stimmt?",
        ),
        ReserveCopyExample(
            use_for="ruhige Rueckfrage zu einem persoenlichen Faden",
            headline="Von letzter Woche ist fuer mich noch etwas offen.",
            body="Was ist dir dazu noch wichtig?",
        ),
    ),
    "discovery": (
        ReserveCopyExample(
            use_for="natuerliche Frage nach der richtigen Ansprache",
            headline="Ich moechte wissen, wie ich dich ansprechen soll.",
            body="Wie soll ich dich nennen?",
        ),
        ReserveCopyExample(
            use_for="Interesse an einer Alltagsgewohnheit",
            headline="Mich interessiert, was dir morgens gut tut.",
            body="Was gehoert fuer dich dazu?",
        ),
        ReserveCopyExample(
            use_for="freundliche Grenze oder Vorliebe kennenlernen",
            headline="Ich wuerde gern wissen, was Twinr besser lassen soll.",
            body="Was ist dir da wichtig?",
        ),
    ),
    "reflection": (
        ReserveCopyExample(
            use_for="ruhiger Rueckbezug auf ein frueheres Gespraech",
            headline="Ich denke noch an unser Gespraech ueber Twinr.",
            body="Was ist dir davon haengengeblieben?",
        ),
        ReserveCopyExample(
            use_for="kleiner Anschluss an einen frischen konkreten Punkt",
            headline="Dein Gedanke zum Arzttermin ist mir geblieben.",
            body="Wollen wir da kurz weitermachen?",
        ),
        ReserveCopyExample(
            use_for="Rueckfrage zu einem offenen Thema von gestern",
            headline="Ich habe dein Thema von gestern noch im Kopf.",
            body="Was ist seitdem dazu passiert?",
        ),
    ),
}

_QUALITY_RUBRIC: tuple[ReserveCopyRubricCriterion, ...] = (
    ReserveCopyRubricCriterion(
        key="idiomatisches_deutsch",
        question="Klingt die Karte wie normales, spontanes Deutsch?",
        prefer="einfache, natuerliche Saetze ohne Uebersetzungs- oder Labelton",
        avoid="Mischsprache, UI-Woerter, Halbsaetze, holprige Formeln",
    ),
    ReserveCopyRubricCriterion(
        key="sofortige_klarheit",
        question="Ist nach der Headline sofort klar, worum es geht?",
        prefer="ein klar benannter Anlass mit fruehem Themenanker",
        avoid="vage Stimmung, Etiketten oder versteckter Anlass erst in der Body-Zeile",
    ),
    ReserveCopyRubricCriterion(
        key="interaktionsreiz",
        question="Macht die Body-Zeile Lust, darauf zu reagieren?",
        prefer="eine echte Anschlussfrage oder Einladung zur Meinung",
        avoid="blosse Nettigkeit, leere CTA-Floskeln oder zweite Erklaerungszeile",
    ),
    ReserveCopyRubricCriterion(
        key="twinr_stimme",
        question="Klingt der Text ruhig, warm und leicht eigen wie Twinr?",
        prefer="aufmerksam, unaufgeregt, leise eigen, eher Begleiter als Service",
        avoid="Marketing, Coaching, Kundenservice oder uebertriebene Innerlichkeit",
    ),
    ReserveCopyRubricCriterion(
        key="screen_tauglichkeit",
        question="Funktioniert die Karte als kurzer HDMI-Impuls aus dem Augenwinkel?",
        prefer="ein dominanter Gedanke, kurze Zeilen, schnell erfassbar",
        avoid="ueberladene Karten, doppelte Themen oder verschachtelte Saetze",
    ),
)


def resolve_reserve_copy_family(candidate: AmbientDisplayImpulseCandidate) -> str:
    """Map one reserve candidate onto the shared prompt example families."""

    context = candidate.generation_context if isinstance(candidate.generation_context, Mapping) else {}
    display_goal = _compact_text(context.get("display_goal")).casefold()
    candidate_family = _compact_text(candidate.candidate_family).casefold()
    source = _compact_text(candidate.source).casefold()

    if display_goal == "invite_user_discovery" or "discovery" in candidate_family or source == "user_discovery":
        return "discovery"
    if (
        display_goal == "call_back_to_earlier_conversation"
        or "reflection" in candidate_family
        or source.startswith("reflection")
    ):
        return "reflection"
    if (
        "memory" in candidate_family
        or "follow_up" in candidate_family
        or "conflict" in candidate_family
        or source.startswith("memory")
        or source == "relationship"
    ):
        return "memory"
    return "world"


def reserve_copy_examples_for_family(copy_family: str) -> tuple[ReserveCopyExample, ...]:
    """Return the small gold example set for one normalized copy family."""

    return _FAMILY_EXAMPLES.get(copy_family, _FAMILY_EXAMPLES["world"])


def reserve_copy_examples_payload(
    candidates: Sequence[AmbientDisplayImpulseCandidate],
) -> dict[str, list[dict[str, str]]]:
    """Serialize family examples for the families present in the batch."""

    families = {
        resolve_reserve_copy_family(candidate)
        for candidate in candidates
    }
    return {
        family: [example.to_prompt_dict() for example in reserve_copy_examples_for_family(family)]
        for family in sorted(families)
    }


def reserve_copy_rubric_payload() -> list[dict[str, str]]:
    """Return the prompt-facing quality rubric in stable priority order."""

    return [criterion.to_prompt_dict() for criterion in _QUALITY_RUBRIC]


__all__ = [
    "ReserveCopyExample",
    "ReserveCopyRubricCriterion",
    "reserve_copy_examples_for_family",
    "reserve_copy_examples_payload",
    "reserve_copy_rubric_payload",
    "resolve_reserve_copy_family",
]
