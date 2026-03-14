from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Mapping, Protocol
from zoneinfo import ZoneInfo

from twinr.agent.base_agent.config import TwinrConfig
from twinr.llm_json import request_structured_json_object
from twinr.memory.chonkydb.schema import TWINR_GRAPH_ALLOWED_EDGE_TYPES
from twinr.memory.longterm.models import (
    LongTermGraphEdgeCandidateV1,
    LongTermMemoryObjectV1,
    LongTermSourceRefV1,
    LongTermTurnExtractionV1,
)
from twinr.text_utils import collapse_whitespace, slugify_identifier, truncate_text

if TYPE_CHECKING:
    from twinr.providers.openai import OpenAIBackend


_MEMORY_KIND_PREFIX = {
    "relationship_fact": "fact",
    "contact_method_fact": "fact",
    "preference_fact": "fact",
    "plan_fact": "plan",
    "situational_observation": "observation",
    "medical_event": "event",
    "event_fact": "event",
}

_EXTRACTION_OBJECT_KINDS = (
    "relationship_fact",
    "contact_method_fact",
    "preference_fact",
    "plan_fact",
    "situational_observation",
    "medical_event",
    "event_fact",
)


def _normalize_text(value: str | None, *, limit: int | None = None) -> str:
    return truncate_text(value, limit=limit)


def _quoted(value: str) -> str:
    return '"' + value.replace('"', '\\"') + '"'


class LongTermStructuredTurnProgram(Protocol):
    def extract_turn(
        self,
        *,
        transcript: str,
        response: str,
        occurred_at: datetime,
        turn_id: str,
        timezone_name: str,
    ) -> Mapping[str, object]:
        ...


@dataclass(frozen=True, slots=True)
class OpenAIStructuredTurnProgram:
    backend: "OpenAIBackend"

    def extract_turn(
        self,
        *,
        transcript: str,
        response: str,
        occurred_at: datetime,
        turn_id: str,
        timezone_name: str,
    ) -> Mapping[str, object]:
        from twinr.providers.openai.structured import request_structured_json_object

        return request_structured_json_object(
            self.backend,
            prompt="\n".join(
                (
                    "Compile long-term memory candidates from one assistant turn.",
                    "Internal memory must stay in canonical English.",
                    "Split one turn into multiple atomic memories when needed.",
                    "Only extract facts that are explicit or directly stated.",
                    "Do not infer hidden medical diagnoses, feelings, or unstated background facts.",
                    "Keep names, dates, places, phone numbers, and relation labels precise.",
                    f"Turn id: {turn_id}",
                    f"Occurred at: {occurred_at.astimezone(ZoneInfo(timezone_name)).isoformat()}",
                    f"Timezone: {timezone_name}",
                    f"User transcript: {_quoted(transcript)}",
                    f"Assistant response: {_quoted(response)}",
                )
            ),
            instructions="\n".join(
                (
                    "Return one strict JSON object only.",
                    "Use canonical English in summaries, details, slot keys, value keys, and attributes.",
                    "Use graph refs like person:janina, place:eye_doctor, phone:+15555551234, day:2026-03-14.",
                    "If a person relation is explicit, emit both a relationship memory object and a graph edge.",
                    "If an event has a specific day, set valid_from and valid_to to that date and emit a temporal_occurs_on edge to day:<date>.",
                    "If no durable candidate is justified, return empty arrays.",
                )
            ),
            schema_name="twinr_long_term_turn_extraction_v1",
            schema=_turn_extraction_schema(),
            model=self.backend.config.default_model,
            reasoning_effort="low",
            max_output_tokens=1200,
        )


@dataclass(frozen=True, slots=True)
class LongTermTurnExtractor:
    timezone_name: str = "Europe/Berlin"
    program: LongTermStructuredTurnProgram | None = None

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        program: LongTermStructuredTurnProgram | None = None,
    ) -> "LongTermTurnExtractor":
        if program is not None:
            return cls(
                timezone_name=config.local_timezone_name,
                program=program,
            )
        extractor_program: LongTermStructuredTurnProgram | None = None
        if config.openai_api_key:
            from twinr.providers.openai import OpenAIBackend

            extractor_program = OpenAIStructuredTurnProgram(
                backend=OpenAIBackend(
                    config,
                    base_instructions="",
                )
            )
        return cls(
            timezone_name=config.local_timezone_name,
            program=extractor_program,
        )

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
        episode_attributes: dict[str, object] = {
            "raw_transcript": clean_transcript,
            "raw_response": clean_response,
            "structured_extraction_status": "disabled" if self.program is None else "ready",
        }
        candidate_objects: tuple[LongTermMemoryObjectV1, ...] = ()
        graph_edges: tuple[LongTermGraphEdgeCandidateV1, ...] = ()
        if self.program is not None:
            try:
                payload = self.program.extract_turn(
                    transcript=clean_transcript,
                    response=clean_response,
                    occurred_at=occurred,
                    turn_id=resolved_turn_id,
                    timezone_name=self.timezone_name,
                )
                candidate_objects = self._candidate_objects_from_payload(
                    payload=payload,
                    source_ref=source_ref,
                )
                graph_edges = self._graph_edges_from_payload(payload)
            except Exception as exc:
                episode_attributes["structured_extraction_status"] = "failed"
                episode_attributes["structured_extraction_error"] = _normalize_text(str(exc), limit=220)
        episode = LongTermMemoryObjectV1(
            memory_id=f"episode:{slugify_identifier(resolved_turn_id, fallback='turn')}",
            kind="episode",
            summary="Conversation turn recorded for long-term memory.",
            details=f"User said: {_quoted(clean_transcript)} Assistant answered: {_quoted(clean_response)}",
            source=source_ref,
            status="candidate",
            confidence=1.0,
            sensitivity="normal",
            slot_key=f"episode:{resolved_turn_id}",
            value_key=slugify_identifier(clean_transcript, fallback="episode"),
            valid_from=None,
            valid_to=None,
            attributes=episode_attributes,
        )
        return LongTermTurnExtractionV1(
            turn_id=resolved_turn_id,
            occurred_at=occurred,
            episode=episode,
            candidate_objects=candidate_objects,
            graph_edges=graph_edges,
        )

    def _candidate_objects_from_payload(
        self,
        *,
        payload: Mapping[str, object],
        source_ref: LongTermSourceRefV1,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        objects_payload = payload.get("objects")
        if not isinstance(objects_payload, list):
            return ()
        candidates: list[LongTermMemoryObjectV1] = []
        for index, raw_item in enumerate(objects_payload):
            if not isinstance(raw_item, Mapping):
                continue
            kind = _normalize_text(str(raw_item.get("kind", "")), limit=64)
            summary = _normalize_text(str(raw_item.get("summary", "")), limit=220)
            slot_key = _normalize_text(str(raw_item.get("slot_key", "")), limit=220)
            value_key = _normalize_text(str(raw_item.get("value_key", "")), limit=220)
            if kind not in _EXTRACTION_OBJECT_KINDS or not summary or not slot_key or not value_key:
                continue
            attributes = self._normalize_attributes(raw_item.get("attributes"))
            confidence = raw_item.get("confidence", 0.5)
            sensitivity = _normalize_text(str(raw_item.get("sensitivity", "normal")), limit=32) or "normal"
            prefix = _MEMORY_KIND_PREFIX.get(kind, "memory")
            stable_basis = f"{slot_key}:{value_key}"
            memory_id = f"{prefix}:{slugify_identifier(stable_basis, fallback=f'item_{index + 1}')}"
            candidates.append(
                LongTermMemoryObjectV1(
                    memory_id=memory_id,
                    kind=kind,
                    summary=summary,
                    details=_normalize_text(raw_item.get("details"), limit=320) or None,
                    source=source_ref,
                    status="candidate",
                    confidence=float(confidence) if isinstance(confidence, (int, float)) else 0.5,
                    sensitivity=sensitivity if sensitivity else "normal",
                    slot_key=slot_key,
                    value_key=value_key,
                    valid_from=self._optional_text(raw_item.get("valid_from")),
                    valid_to=self._optional_text(raw_item.get("valid_to")),
                    attributes=attributes or None,
                )
            )
        return tuple(candidates)

    def _graph_edges_from_payload(
        self,
        payload: Mapping[str, object],
    ) -> tuple[LongTermGraphEdgeCandidateV1, ...]:
        edges_payload = payload.get("graph_edges")
        if not isinstance(edges_payload, list):
            return ()
        edges: list[LongTermGraphEdgeCandidateV1] = []
        for raw_edge in edges_payload:
            if not isinstance(raw_edge, Mapping):
                continue
            edge_type = _normalize_text(str(raw_edge.get("edge_type", "")), limit=80)
            if edge_type not in TWINR_GRAPH_ALLOWED_EDGE_TYPES:
                continue
            source_ref = _normalize_text(str(raw_edge.get("source_ref", "")), limit=160)
            target_ref = _normalize_text(str(raw_edge.get("target_ref", "")), limit=160)
            if not source_ref or not target_ref:
                continue
            confidence = raw_edge.get("confidence", 0.5)
            edges.append(
                LongTermGraphEdgeCandidateV1(
                    source_ref=source_ref,
                    edge_type=edge_type,
                    target_ref=target_ref,
                    confidence=float(confidence) if isinstance(confidence, (int, float)) else 0.5,
                    confirmed_by_user=bool(raw_edge.get("confirmed_by_user", True)),
                    valid_from=self._optional_text(raw_edge.get("valid_from")),
                    valid_to=self._optional_text(raw_edge.get("valid_to")),
                    attributes=self._normalize_attributes(raw_edge.get("attributes")) or None,
                )
            )
        return tuple(edges)

    def _normalize_attributes(self, value: object) -> dict[str, str]:
        if not isinstance(value, Mapping):
            return {}
        normalized: dict[str, str] = {}
        for key, raw in value.items():
            clean_key = collapse_whitespace(str(key))
            clean_value = self._optional_text(raw)
            if clean_key and clean_value:
                normalized[clean_key] = clean_value
        return normalized

    def _optional_text(self, value: object) -> str | None:
        if value is None:
            return None
        clean_value = _normalize_text(str(value), limit=160)
        return clean_value or None


def _turn_extraction_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": list(_EXTRACTION_OBJECT_KINDS),
                        },
                        "summary": {"type": "string"},
                        "details": {"type": "string"},
                        "confidence": {"type": "number"},
                        "sensitivity": {
                            "type": "string",
                            "enum": ["low", "normal", "private", "medical", "sensitive"],
                        },
                        "slot_key": {"type": "string"},
                        "value_key": {"type": "string"},
                        "valid_from": {"type": "string"},
                        "valid_to": {"type": "string"},
                        "attributes": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                    },
                    "required": [
                        "kind",
                        "summary",
                        "confidence",
                        "sensitivity",
                        "slot_key",
                        "value_key",
                        "attributes",
                    ],
                },
            },
            "graph_edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "source_ref": {"type": "string"},
                        "edge_type": {
                            "type": "string",
                            "enum": sorted(TWINR_GRAPH_ALLOWED_EDGE_TYPES),
                        },
                        "target_ref": {"type": "string"},
                        "confidence": {"type": "number"},
                        "confirmed_by_user": {"type": "boolean"},
                        "valid_from": {"type": "string"},
                        "valid_to": {"type": "string"},
                        "attributes": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                    },
                    "required": [
                        "source_ref",
                        "edge_type",
                        "target_ref",
                        "confidence",
                        "confirmed_by_user",
                        "attributes",
                    ],
                },
            },
        },
        "required": ["objects", "graph_edges"],
    }


__all__ = [
    "LongTermStructuredTurnProgram",
    "LongTermTurnExtractor",
    "OpenAIStructuredTurnProgram",
]
