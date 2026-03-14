from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import re
from zoneinfo import ZoneInfo

from twinr.memory.longterm.models import (
    LongTermGraphEdgeCandidateV1,
    LongTermMemoryObjectV1,
    LongTermSourceRefV1,
    LongTermTurnExtractionV1,
)


_RELATION_TO_EDGE = {
    "wife": ("social_family_of", "wife"),
    "husband": ("social_family_of", "husband"),
    "daughter": ("social_family_of", "daughter"),
    "son": ("social_family_of", "son"),
    "mother": ("social_family_of", "mother"),
    "father": ("social_family_of", "father"),
    "sister": ("social_family_of", "sister"),
    "brother": ("social_family_of", "brother"),
    "friend": ("social_friend_of", "friend"),
    "neighbor": ("general_related_to", "neighbor"),
    "neighbour": ("general_related_to", "neighbor"),
    "physiotherapist": ("social_supports_user_as", "physiotherapist"),
    "caregiver": ("social_supports_user_as", "caregiver"),
    "doctor": ("social_supports_user_as", "doctor"),
}

_RELATION_PATTERN = re.compile(
    r"\bmy\s+(?P<relation>wife|husband|daughter|son|mother|father|sister|brother|friend|neighbor|neighbour|physiotherapist|caregiver|doctor)\s+(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
)
_AT_PLACE_PATTERN = re.compile(
    r"\b(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+at\s+(?P<place>[^.,;]+?)(?:\s+and\s+is\s+getting\s+(?P<treatment>[^.,;]+))?(?:[.,;]|$)"
)
_GETTING_PATTERN = re.compile(
    r"\b(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+getting\s+(?P<treatment>[^.,;]+?)(?:[.,;]|$)"
)
_DAY_WORDS = {
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
}
_WEATHER_WORDS = {
    "warm",
    "hot",
    "cold",
    "cool",
    "sunny",
    "rainy",
    "windy",
    "mild",
    "cloudy",
    "bright",
}
_MEDICAL_TERMS = {"doctor", "clinic", "treatment", "laser", "hospital", "appointment", "surgery"}


def _normalize_text(value: str | None, *, limit: int | None = None) -> str:
    text = " ".join(str(value or "").split()).strip()
    if limit is None or len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def _slugify(value: str, *, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", _normalize_text(value).lower()).strip("_")
    return slug or fallback


def _extract_date(text: str, *, occurred_at: datetime, timezone_name: str) -> str | None:
    lower = text.lower()
    now = occurred_at.astimezone(ZoneInfo(timezone_name))
    if "today" in lower:
        return now.date().isoformat()
    if "tomorrow" in lower:
        return (now.date() + timedelta(days=1)).isoformat()
    match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if match is not None:
        return match.group(1)
    return None


def _quoted(value: str) -> str:
    return '"' + value.replace('"', '\\"') + '"'


@dataclass(frozen=True, slots=True)
class LongTermTurnExtractor:
    timezone_name: str = "Europe/Berlin"

    def extract_conversation_turn(
        self,
        *,
        transcript: str,
        response: str,
        occurred_at: datetime | None = None,
        turn_id: str | None = None,
        source: str = "conversation_turn",
    ) -> LongTermTurnExtractionV1:
        clean_transcript = _normalize_text(transcript, limit=500)
        clean_response = _normalize_text(response, limit=500)
        if not clean_transcript:
            raise ValueError("transcript is required.")
        occurred = occurred_at or datetime.now(ZoneInfo(self.timezone_name))
        resolved_turn_id = turn_id or f"turn:{occurred.astimezone(ZoneInfo(self.timezone_name)).strftime('%Y%m%dT%H%M%S%z')}"
        source_ref = LongTermSourceRefV1(
            source_type=source,
            event_ids=(resolved_turn_id,),
            speaker="user",
            modality="voice",
        )
        episode = LongTermMemoryObjectV1(
            memory_id=f"episode:{_slugify(resolved_turn_id, fallback='turn')}",
            kind="episode",
            summary="Conversation turn recorded for long-term memory.",
            details=f"User said: {_quoted(clean_transcript)} Assistant answered: {_quoted(clean_response)}",
            source=source_ref,
            status="candidate",
            confidence=1.0,
            sensitivity="normal",
            slot_key=f"episode:{resolved_turn_id}",
            value_key=_slugify(clean_transcript, fallback="episode"),
            valid_from=_extract_date(clean_transcript, occurred_at=occurred, timezone_name=self.timezone_name),
            valid_to=_extract_date(clean_transcript, occurred_at=occurred, timezone_name=self.timezone_name),
            attributes={
                "raw_transcript": clean_transcript,
                "raw_response": clean_response,
            },
        )

        candidates: list[LongTermMemoryObjectV1] = []
        graph_edges: list[LongTermGraphEdgeCandidateV1] = []
        known_people: dict[str, str] = {}

        for match in _RELATION_PATTERN.finditer(clean_transcript):
            relation = _normalize_text(match.group("relation")).lower()
            name = _normalize_text(match.group("name"), limit=100)
            person_ref = f"person:{_slugify(name, fallback='person')}"
            known_people[name] = person_ref
            edge_type, relation_label = _RELATION_TO_EDGE.get(relation, ("general_related_to", relation))
            memory_id = f"fact:{_slugify(f'{name}_{relation_label}', fallback='relationship')}"
            candidates.append(
                LongTermMemoryObjectV1(
                    memory_id=memory_id,
                    kind="relationship_fact",
                    summary=f"{name} is the user's {relation_label}.",
                    details=f"Derived from a user statement in {_quoted(clean_transcript)}.",
                    source=source_ref,
                    status="candidate",
                    confidence=0.98,
                    sensitivity="private",
                    slot_key=f"relationship:user:main:{relation_label}",
                    value_key=person_ref,
                    attributes={
                        "person_ref": person_ref,
                        "person_name": name,
                        "relation": relation_label,
                    },
                )
            )
            graph_edges.append(
                LongTermGraphEdgeCandidateV1(
                    source_ref="user:main",
                    edge_type=edge_type,
                    target_ref=person_ref,
                    confidence=0.98,
                    confirmed_by_user=True,
                    attributes={
                        "relation": relation_label,
                        "origin_memory_id": memory_id,
                    },
                )
            )

        for observation in self._extract_environment_observations(
            transcript=clean_transcript,
            occurred_at=occurred,
            source_ref=source_ref,
        ):
            candidates.append(observation)

        for event in self._extract_person_events(
            transcript=clean_transcript,
            occurred_at=occurred,
            source_ref=source_ref,
            known_people=known_people,
        ):
            candidates.extend(event["objects"])
            graph_edges.extend(event["edges"])

        return LongTermTurnExtractionV1(
            turn_id=resolved_turn_id,
            occurred_at=occurred,
            episode=episode,
            candidate_objects=tuple(candidates),
            graph_edges=tuple(graph_edges),
        )

    def _extract_environment_observations(
        self,
        *,
        transcript: str,
        occurred_at: datetime,
        source_ref: LongTermSourceRefV1,
    ) -> list[LongTermMemoryObjectV1]:
        lower_tokens = {token.strip(".,;:!?").lower() for token in transcript.split()}
        observations: list[LongTermMemoryObjectV1] = []
        date_key = _extract_date(transcript, occurred_at=occurred_at, timezone_name=self.timezone_name)
        day_words = sorted(word for word in _DAY_WORDS if word in lower_tokens)
        for day_word in day_words:
            observations.append(
                LongTermMemoryObjectV1(
                    memory_id=f"observation:{_slugify(f'{day_word}_{date_key or 'day'}', fallback='day')}",
                    kind="situational_observation",
                    summary=f"The user described the day as {day_word}.",
                    details=f"Observed in {_quoted(transcript)}.",
                    source=source_ref,
                    confidence=0.72,
                    sensitivity="low",
                    slot_key=f"observation:day_name:{date_key or 'unknown'}",
                    value_key=day_word,
                    valid_from=date_key,
                    valid_to=date_key,
                    attributes={"topic": "day_name"},
                )
            )
        weather_words = sorted(word for word in _WEATHER_WORDS if word in lower_tokens)
        for weather_word in weather_words:
            observations.append(
                LongTermMemoryObjectV1(
                    memory_id=f"observation:{_slugify(f'{weather_word}_{date_key or 'weather'}', fallback='weather')}",
                    kind="situational_observation",
                    summary=f"The user described the day as {weather_word}.",
                    details=f"Observed in {_quoted(transcript)}.",
                    source=source_ref,
                    confidence=0.7,
                    sensitivity="low",
                    slot_key=f"observation:weather:{date_key or 'unknown'}",
                    value_key=weather_word,
                    valid_from=date_key,
                    valid_to=date_key,
                    attributes={"topic": "weather"},
                )
            )
        return observations

    def _extract_person_events(
        self,
        *,
        transcript: str,
        occurred_at: datetime,
        source_ref: LongTermSourceRefV1,
        known_people: dict[str, str],
    ) -> list[dict[str, tuple[LongTermMemoryObjectV1, ...] | tuple[LongTermGraphEdgeCandidateV1, ...]]]:
        person_events: dict[str, dict[str, str]] = {}
        for match in _AT_PLACE_PATTERN.finditer(transcript):
            name = _normalize_text(match.group("name"), limit=100)
            place = _normalize_text(match.group("place"), limit=120)
            treatment = _normalize_text(match.group("treatment"), limit=120)
            event = person_events.setdefault(name, {})
            if place:
                event["place"] = place
            if treatment:
                event["treatment"] = treatment
        for match in _GETTING_PATTERN.finditer(transcript):
            name = _normalize_text(match.group("name"), limit=100)
            treatment = _normalize_text(match.group("treatment"), limit=120)
            if not treatment:
                continue
            event = person_events.setdefault(name, {})
            event.setdefault("treatment", treatment)

        results: list[dict[str, tuple[LongTermMemoryObjectV1, ...] | tuple[LongTermGraphEdgeCandidateV1, ...]]] = []
        date_key = _extract_date(transcript, occurred_at=occurred_at, timezone_name=self.timezone_name)
        for name, details in person_events.items():
            person_ref = known_people.get(name, f"person:{_slugify(name, fallback='person')}")
            treatment = details.get("treatment")
            place = details.get("place")
            if not treatment and not place:
                continue
            event_slug = _slugify(
                f"{name}_{treatment or place or 'event'}_{date_key or occurred_at.date().isoformat()}",
                fallback="event",
            )
            event_ref = f"event:{event_slug}"
            summary = self._event_summary(name=name, place=place, treatment=treatment, date_key=date_key)
            kind = "medical_event" if self._looks_medical(place=place, treatment=treatment) else "event_fact"
            memory = LongTermMemoryObjectV1(
                memory_id=f"fact:{event_slug}",
                kind=kind,
                summary=summary,
                details=f"Derived from a user statement in {_quoted(transcript)}.",
                source=source_ref,
                confidence=0.9,
                sensitivity="medical" if kind == "medical_event" else "private",
                slot_key=f"event:{person_ref}:{_slugify(treatment or place or 'event', fallback='event')}:{date_key or 'open'}",
                value_key=event_ref,
                valid_from=date_key,
                valid_to=date_key,
                attributes={
                    "person_ref": person_ref,
                    "person_name": name,
                    "place": place,
                    "treatment": treatment,
                    "event_ref": event_ref,
                },
            )
            edges = [
                LongTermGraphEdgeCandidateV1(
                    source_ref=event_ref,
                    edge_type="general_related_to",
                    target_ref=person_ref,
                    confidence=0.9,
                    attributes={"origin_memory_id": memory.memory_id},
                    valid_from=date_key,
                    valid_to=date_key,
                )
            ]
            if date_key:
                edges.append(
                    LongTermGraphEdgeCandidateV1(
                        source_ref=event_ref,
                        edge_type="temporal_occurs_on",
                        target_ref=f"day:{date_key}",
                        confidence=0.95,
                        attributes={"origin_memory_id": memory.memory_id},
                        valid_from=date_key,
                        valid_to=date_key,
                    )
                )
            results.append({"objects": (memory,), "edges": tuple(edges)})
        return results

    def _event_summary(
        self,
        *,
        name: str,
        place: str | None,
        treatment: str | None,
        date_key: str | None,
    ) -> str:
        date_suffix = f" on {date_key}" if date_key else ""
        if place and treatment:
            return f"{name} has {treatment} at {place}{date_suffix}."
        if treatment:
            return f"{name} is receiving {treatment}{date_suffix}."
        return f"{name} is at {place}{date_suffix}."

    def _looks_medical(self, *, place: str | None, treatment: str | None) -> bool:
        haystack = f"{place or ''} {treatment or ''}".lower()
        return any(term in haystack for term in _MEDICAL_TERMS)


__all__ = ["LongTermTurnExtractor"]
