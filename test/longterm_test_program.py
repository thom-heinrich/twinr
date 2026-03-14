from __future__ import annotations

from datetime import datetime
from typing import Mapping

from twinr.memory.longterm.extract import LongTermStructuredTurnProgram, LongTermTurnExtractor


def _janina_full_payload(date_key: str) -> dict[str, object]:
    return {
        "objects": [
            {
                "kind": "relationship_fact",
                "summary": "Janina is the user's wife.",
                "details": "Derived from the user's statement.",
                "confidence": 0.98,
                "sensitivity": "private",
                "slot_key": "relationship:user:main:wife",
                "value_key": "person:janina",
                "attributes": {
                    "person_ref": "person:janina",
                    "person_name": "Janina",
                    "relation": "wife",
                },
            },
            {
                "kind": "situational_observation",
                "summary": "The user described the day as sunday.",
                "details": "Observed in the transcript.",
                "confidence": 0.72,
                "sensitivity": "low",
                "slot_key": f"observation:day_name:{date_key}",
                "value_key": "sunday",
                "valid_from": date_key,
                "valid_to": date_key,
                "attributes": {
                    "topic": "day_name",
                },
            },
            {
                "kind": "situational_observation",
                "summary": "The user described the day as warm.",
                "details": "Observed in the transcript.",
                "confidence": 0.7,
                "sensitivity": "low",
                "slot_key": f"observation:weather:{date_key}",
                "value_key": "warm",
                "valid_from": date_key,
                "valid_to": date_key,
                "attributes": {
                    "topic": "weather",
                },
            },
            {
                "kind": "medical_event",
                "summary": f"Janina has eye laser treatment at the eye doctor on {date_key}.",
                "details": "Directly stated by the user.",
                "confidence": 0.93,
                "sensitivity": "medical",
                "slot_key": f"event:person:janina:eye_laser_treatment:{date_key}",
                "value_key": "eye_laser_treatment",
                "valid_from": date_key,
                "valid_to": date_key,
                "attributes": {
                    "person_ref": "person:janina",
                    "person_name": "Janina",
                    "place": "eye doctor",
                    "place_ref": "place:eye_doctor",
                    "treatment": "eye laser treatment",
                },
            },
        ],
        "graph_edges": [
            {
                "source_ref": "user:main",
                "edge_type": "social_family_of",
                "target_ref": "person:janina",
                "confidence": 0.98,
                "confirmed_by_user": True,
                "attributes": {
                    "relation": "wife",
                },
            },
            {
                "source_ref": "person:janina",
                "edge_type": "general_related_to",
                "target_ref": "place:eye_doctor",
                "confidence": 0.9,
                "confirmed_by_user": True,
                "attributes": {
                    "origin": "medical_event",
                },
            },
            {
                "source_ref": "event:person:janina:eye_laser_treatment",
                "edge_type": "temporal_occurs_on",
                "target_ref": f"day:{date_key}",
                "confidence": 0.9,
                "confirmed_by_user": True,
                "attributes": {},
            },
        ],
    }


def _janina_today_payload(date_key: str) -> dict[str, object]:
    return {
        "objects": [
            {
                "kind": "relationship_fact",
                "summary": "Janina is the user's wife.",
                "details": "Derived from the user's statement.",
                "confidence": 0.98,
                "sensitivity": "private",
                "slot_key": "relationship:user:main:wife",
                "value_key": "person:janina",
                "attributes": {
                    "person_ref": "person:janina",
                    "person_name": "Janina",
                    "relation": "wife",
                },
            },
            {
                "kind": "medical_event",
                "summary": f"Janina has an appointment at the eye doctor on {date_key}.",
                "details": "Directly stated by the user.",
                "confidence": 0.9,
                "sensitivity": "medical",
                "slot_key": f"event:person:janina:eye_doctor:{date_key}",
                "value_key": "eye_doctor",
                "valid_from": date_key,
                "valid_to": date_key,
                "attributes": {
                    "person_ref": "person:janina",
                    "person_name": "Janina",
                    "place": "eye doctor",
                    "place_ref": "place:eye_doctor",
                },
            },
        ],
        "graph_edges": [
            {
                "source_ref": "user:main",
                "edge_type": "social_family_of",
                "target_ref": "person:janina",
                "confidence": 0.98,
                "confirmed_by_user": True,
                "attributes": {
                    "relation": "wife",
                },
            },
            {
                "source_ref": "event:person:janina:eye_doctor",
                "edge_type": "temporal_occurs_on",
                "target_ref": f"day:{date_key}",
                "confidence": 0.9,
                "confirmed_by_user": True,
                "attributes": {},
            },
        ],
    }


def _janina_laser_today_payload(date_key: str) -> dict[str, object]:
    return {
        "objects": [
            {
                "kind": "relationship_fact",
                "summary": "Janina is the user's wife.",
                "details": "Derived from the user's statement.",
                "confidence": 0.98,
                "sensitivity": "private",
                "slot_key": "relationship:user:main:wife",
                "value_key": "person:janina",
                "attributes": {
                    "person_ref": "person:janina",
                    "person_name": "Janina",
                    "relation": "wife",
                },
            },
            {
                "kind": "medical_event",
                "summary": f"Janina has eye laser treatment on {date_key}.",
                "details": "Directly stated by the user.",
                "confidence": 0.92,
                "sensitivity": "medical",
                "slot_key": f"event:person:janina:eye_laser_treatment:{date_key}",
                "value_key": "eye_laser_treatment",
                "valid_from": date_key,
                "valid_to": date_key,
                "attributes": {
                    "person_ref": "person:janina",
                    "person_name": "Janina",
                    "treatment": "eye laser treatment",
                },
            },
        ],
        "graph_edges": [
            {
                "source_ref": "user:main",
                "edge_type": "social_family_of",
                "target_ref": "person:janina",
                "confidence": 0.98,
                "confirmed_by_user": True,
                "attributes": {
                    "relation": "wife",
                },
            },
            {
                "source_ref": "event:person:janina:eye_laser_treatment",
                "edge_type": "temporal_occurs_on",
                "target_ref": f"day:{date_key}",
                "confidence": 0.9,
                "confirmed_by_user": True,
                "attributes": {},
            },
        ],
    }


def _warm_today_payload(date_key: str) -> dict[str, object]:
    return {
        "objects": [
            {
                "kind": "situational_observation",
                "summary": "The user described the day as warm.",
                "details": "Observed in the transcript.",
                "confidence": 0.7,
                "sensitivity": "low",
                "slot_key": f"observation:weather:{date_key}",
                "value_key": "warm",
                "valid_from": date_key,
                "valid_to": date_key,
                "attributes": {
                    "topic": "weather",
                },
            },
        ],
        "graph_edges": [],
    }


class StubStructuredTurnProgram(LongTermStructuredTurnProgram):
    def extract_turn(
        self,
        *,
        transcript: str,
        response: str,
        occurred_at: datetime,
        turn_id: str,
        timezone_name: str,
    ) -> Mapping[str, object]:
        del response, turn_id, timezone_name
        date_key = occurred_at.date().isoformat()
        normalized = " ".join(transcript.split()).strip().lower()
        if "my wife janina is at the eye doctor and is getting eye laser treatment" in normalized:
            return _janina_full_payload(date_key)
        if "my wife janina is getting eye laser treatment today" in normalized:
            return _janina_laser_today_payload(date_key)
        if "my wife janina is at the eye doctor today" in normalized:
            return _janina_today_payload(date_key)
        if normalized == "today is warm." or normalized == "today is warm":
            return _warm_today_payload(date_key)
        return {"objects": [], "graph_edges": []}


def make_test_extractor(*, timezone_name: str = "Europe/Berlin") -> LongTermTurnExtractor:
    return LongTermTurnExtractor(
        timezone_name=timezone_name,
        program=StubStructuredTurnProgram(),
    )
