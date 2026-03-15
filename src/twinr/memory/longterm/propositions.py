from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Mapping, Protocol
from zoneinfo import ZoneInfo

from twinr.agent.base_agent.config import TwinrConfig
from twinr.llm_json import request_structured_json_object
from twinr.memory.chonkydb.schema import TWINR_GRAPH_ALLOWED_EDGE_TYPES, normalize_graph_edge_type
from twinr.memory.longterm.ontology import LONGTERM_GENERIC_KINDS, memory_kind_prefix, normalize_memory_kind
from twinr.memory.longterm.models import (
    LongTermGraphEdgeCandidateV1,
    LongTermMemoryObjectV1,
    LongTermSourceRefV1,
)
from twinr.text_utils import collapse_whitespace, slugify_identifier, truncate_text

if TYPE_CHECKING:
    from twinr.providers.openai import OpenAIBackend


_PROPOSITION_KINDS = tuple(sorted(LONGTERM_GENERIC_KINDS - {"episode"}))
_SENSITIVITIES = ("low", "normal", "private", "sensitive", "critical")
_SOURCE_CHANNELS = ("user_transcript", "assistant_response", "both")


def _normalize_text(value: str | None, *, limit: int | None = None) -> str:
    return truncate_text(value, limit=limit)


def _quoted(value: str) -> str:
    return '"' + value.replace('"', '\\"') + '"'


def _nullable_string_schema() -> dict[str, object]:
    return {
        "anyOf": [
            {"type": "string"},
            {"type": "null"},
        ]
    }


def _attribute_entries_schema() -> dict[str, object]:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["key", "value"],
        },
    }


def _normalize_attribute_entries(value: object) -> dict[str, str]:
    if isinstance(value, list):
        normalized: dict[str, str] = {}
        for item in value:
            if not isinstance(item, Mapping):
                continue
            clean_key = collapse_whitespace(str(item.get("key", "")))
            clean_value = _normalize_optional_text(item.get("value"))
            if clean_key and clean_value:
                normalized[clean_key] = clean_value
        return normalized
    if not isinstance(value, Mapping):
        return {}
    normalized = {}
    for key, item in value.items():
        clean_key = collapse_whitespace(str(key))
        clean_value = _normalize_optional_text(item)
        if clean_key and clean_value:
            normalized[clean_key] = clean_value
    return normalized


def _normalize_optional_text(value: object) -> str | None:
    if value is None:
        return None
    clean_value = _normalize_text(str(value), limit=160)
    return clean_value or None


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
    model: str | None = None
    max_output_tokens: int = 2200

    def extract_turn(
        self,
        *,
        transcript: str,
        response: str,
        occurred_at: datetime,
        turn_id: str,
        timezone_name: str,
    ) -> Mapping[str, object]:
        return request_structured_json_object(
            self.backend,
            prompt="\n".join(
                (
                    "Decompose one assistant turn into atomic long-term memory propositions.",
                    "Internal memory must stay in canonical English.",
                    "One rich turn may yield multiple propositions.",
                    "Capture only what is explicit or directly stated.",
                    "Primary evidence is the user transcript. Use the assistant response only as secondary context.",
                    "Do not infer hidden diagnoses, motives, feelings, or unstated background facts.",
                    "Use broad kinds only: fact, event, observation, plan, summary.",
                    "Keep predicates short, stable, and canonical English.",
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
                    "Return one proposition per atomic memory-worthy claim.",
                    "Do not return empty arrays when the turn contains explicit relationships, events, or observations worth remembering.",
                    "Set source_channel to user_transcript when the claim comes from the user transcript, assistant_response when it comes only from the assistant response, or both when both say the same thing.",
                    "Never create a durable proposition from assistant wording alone unless it strictly restates an explicit fact from the user transcript.",
                    "Use graph refs like user:main, person:janina, place:eye_doctor, day:2026-03-15 when they are directly grounded by the turn.",
                    "When a graph edge is grounded in a single proposition, include that proposition_id in edge attributes.",
                    "Use subject_ref for the main entity being described.",
                    "Use object_ref for stable entity targets and value_text for literal values such as warm or Sunday.",
                    "Encode attributes as arrays of {key, value} pairs.",
                    "If a person relation is explicit, include both a proposition and a graph edge.",
                    "If an event is tied to a specific day, include valid_from, valid_to, and a temporal_occurs_on graph edge.",
                )
            ),
            schema_name="twinr_long_term_turn_propositions_v1",
            schema=_turn_proposition_schema(),
            model=self.model or self.backend.config.default_model,
            reasoning_effort="low",
            max_output_tokens=max(1200, self.max_output_tokens),
        )


@dataclass(frozen=True, slots=True)
class LongTermTurnPropositionV1:
    proposition_id: str
    kind: str
    summary: str
    predicate: str
    confidence: float
    sensitivity: str
    source_channel: str
    details: str | None = None
    subject_ref: str | None = None
    object_ref: str | None = None
    value_text: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    attributes: Mapping[str, object] | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, object], *, index: int) -> "LongTermTurnPropositionV1 | None":
        kind = _normalize_text(str(payload.get("kind", "")), limit=64)
        summary = _normalize_text(str(payload.get("summary", "")), limit=220)
        predicate = _normalize_text(str(payload.get("predicate", "")), limit=120)
        if not kind or not summary or not predicate:
            return None
        confidence = payload.get("confidence", 0.5)
        if not isinstance(confidence, (int, float)):
            confidence = 0.5
        sensitivity = _normalize_text(str(payload.get("sensitivity", "normal")), limit=32) or "normal"
        source_channel = _normalize_text(str(payload.get("source_channel", "user_transcript")), limit=32) or "user_transcript"
        proposition_id = _normalize_text(str(payload.get("proposition_id", "")), limit=120) or f"proposition:{index + 1}"
        return cls(
            proposition_id=proposition_id,
            kind=kind,
            summary=summary,
            predicate=predicate,
            confidence=float(confidence),
            sensitivity=sensitivity,
            source_channel=source_channel,
            details=_normalize_optional_text(payload.get("details")),
            subject_ref=_normalize_optional_text(payload.get("subject_ref")),
            object_ref=_normalize_optional_text(payload.get("object_ref")),
            value_text=_normalize_optional_text(payload.get("value_text")),
            valid_from=_normalize_optional_text(payload.get("valid_from")),
            valid_to=_normalize_optional_text(payload.get("valid_to")),
            attributes=_normalize_attribute_entries(payload.get("attributes")) or None,
        )


@dataclass(frozen=True, slots=True)
class LongTermTurnPropositionBundleV1:
    propositions: tuple[LongTermTurnPropositionV1, ...]
    graph_edges: tuple[LongTermGraphEdgeCandidateV1, ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "LongTermTurnPropositionBundleV1":
        raw_propositions = payload.get("propositions")
        propositions: list[LongTermTurnPropositionV1] = []
        if isinstance(raw_propositions, list):
            for index, raw_item in enumerate(raw_propositions):
                if not isinstance(raw_item, Mapping):
                    continue
                proposition = LongTermTurnPropositionV1.from_payload(raw_item, index=index)
                if proposition is not None:
                    propositions.append(proposition)
        raw_edges = payload.get("graph_edges")
        graph_edges: list[LongTermGraphEdgeCandidateV1] = []
        if isinstance(raw_edges, list):
            for raw_edge in raw_edges:
                if not isinstance(raw_edge, Mapping):
                    continue
                edge_type = _normalize_text(str(raw_edge.get("edge_type", "")), limit=80)
                attributes = _normalize_attribute_entries(raw_edge.get("attributes"))
                normalized_edge_type, normalized_attributes = normalize_graph_edge_type(edge_type, attributes)
                if normalized_edge_type not in TWINR_GRAPH_ALLOWED_EDGE_TYPES:
                    continue
                source_ref = _normalize_optional_text(raw_edge.get("source_ref"))
                target_ref = _normalize_optional_text(raw_edge.get("target_ref"))
                if not source_ref or not target_ref:
                    continue
                confidence = raw_edge.get("confidence", 0.5)
                if not isinstance(confidence, (int, float)):
                    confidence = 0.5
                graph_edges.append(
                    LongTermGraphEdgeCandidateV1(
                        source_ref=source_ref,
                        edge_type=normalized_edge_type,
                        target_ref=target_ref,
                        confidence=float(confidence),
                        confirmed_by_user=bool(raw_edge.get("confirmed_by_user", True)),
                        valid_from=_normalize_optional_text(raw_edge.get("valid_from")),
                        valid_to=_normalize_optional_text(raw_edge.get("valid_to")),
                        attributes=normalized_attributes or None,
                    )
                )
        return cls(
            propositions=tuple(propositions),
            graph_edges=tuple(graph_edges),
        )


@dataclass(frozen=True, slots=True)
class LongTermTurnPropositionCompiler:
    def compile(
        self,
        *,
        bundle: LongTermTurnPropositionBundleV1,
        source_ref: LongTermSourceRefV1,
    ) -> tuple[tuple[LongTermMemoryObjectV1, ...], tuple[LongTermGraphEdgeCandidateV1, ...]]:
        objects: list[LongTermMemoryObjectV1] = []
        accepted_proposition_ids: set[str] = set()
        for index, proposition in enumerate(bundle.propositions):
            if proposition.source_channel == "assistant_response":
                continue
            normalized_kind, normalized_attributes = normalize_memory_kind(
                proposition.kind,
                proposition.attributes,
            )
            if normalized_kind not in _PROPOSITION_KINDS:
                continue
            attributes = dict(normalized_attributes or {})
            if proposition.subject_ref:
                attributes.setdefault("subject_ref", proposition.subject_ref)
            if proposition.object_ref:
                attributes.setdefault("object_ref", proposition.object_ref)
            if proposition.value_text:
                attributes.setdefault("value_text", proposition.value_text)
            attributes.setdefault("predicate", proposition.predicate)
            slot_key = self._build_slot_key(
                kind=normalized_kind,
                proposition=proposition,
            )
            value_key = self._build_value_key(proposition)
            memory_id = self._build_memory_id(
                kind=normalized_kind,
                slot_key=slot_key,
                value_key=value_key,
                fallback_index=index,
            )
            objects.append(
                LongTermMemoryObjectV1(
                    memory_id=memory_id,
                    kind=normalized_kind,
                    summary=proposition.summary,
                    details=proposition.details,
                    source=source_ref,
                    status="candidate",
                    confidence=proposition.confidence,
                    sensitivity=proposition.sensitivity,
                    slot_key=slot_key,
                    value_key=value_key,
                    valid_from=proposition.valid_from,
                    valid_to=proposition.valid_to,
                    attributes=attributes or None,
                )
            )
            accepted_proposition_ids.add(proposition.proposition_id)
        graph_edges = tuple(
            edge
            for edge in bundle.graph_edges
            if self._edge_is_allowed(edge=edge, accepted_proposition_ids=accepted_proposition_ids)
        )
        return tuple(objects), graph_edges

    def _build_slot_key(
        self,
        *,
        kind: str,
        proposition: LongTermTurnPropositionV1,
    ) -> str:
        parts = [kind]
        if proposition.subject_ref:
            parts.append(proposition.subject_ref)
        parts.append(slugify_identifier(proposition.predicate, fallback="predicate"))
        if proposition.valid_from:
            parts.append(proposition.valid_from)
        if proposition.valid_to and proposition.valid_to != proposition.valid_from:
            parts.append(proposition.valid_to)
        return ":".join(part for part in parts if part)

    def _build_value_key(self, proposition: LongTermTurnPropositionV1) -> str:
        if proposition.object_ref:
            return proposition.object_ref
        if proposition.value_text:
            return slugify_identifier(proposition.value_text, fallback="value")
        return slugify_identifier(proposition.summary, fallback="value")

    def _build_memory_id(
        self,
        *,
        kind: str,
        slot_key: str,
        value_key: str,
        fallback_index: int,
    ) -> str:
        prefix = memory_kind_prefix(kind)
        stable_basis = f"{slot_key}:{value_key}"
        return f"{prefix}:{slugify_identifier(stable_basis, fallback=f'item_{fallback_index + 1}')}"

    def _edge_is_allowed(
        self,
        *,
        edge: LongTermGraphEdgeCandidateV1,
        accepted_proposition_ids: set[str],
    ) -> bool:
        attributes = dict(edge.attributes or {})
        for key in ("proposition_id", "origin_proposition_id", "event_ref"):
            referenced = _normalize_optional_text(attributes.get(key))
            if referenced is None:
                continue
            return referenced in accepted_proposition_ids
        if not accepted_proposition_ids:
            return True
        return True


def structured_turn_program_from_config(
    config: TwinrConfig,
) -> LongTermStructuredTurnProgram | None:
    if not config.openai_api_key:
        return None
    from twinr.providers.openai import OpenAIBackend

    return OpenAIStructuredTurnProgram(
        backend=OpenAIBackend(
            replace(
                config,
                openai_realtime_language="en",
                openai_reasoning_effort="low",
            ),
            base_instructions="",
        ),
        model=config.long_term_memory_turn_extractor_model,
        max_output_tokens=config.long_term_memory_turn_extractor_max_output_tokens,
    )


def _turn_proposition_schema() -> dict[str, object]:
    nullable_string = _nullable_string_schema()
    attributes_schema = _attribute_entries_schema()
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "propositions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "proposition_id": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": list(_PROPOSITION_KINDS),
                        },
                        "summary": {"type": "string"},
                        "details": nullable_string,
                        "predicate": {"type": "string"},
                        "confidence": {"type": "number"},
                        "sensitivity": {
                            "type": "string",
                            "enum": list(_SENSITIVITIES),
                        },
                        "source_channel": {
                            "type": "string",
                            "enum": list(_SOURCE_CHANNELS),
                        },
                        "subject_ref": nullable_string,
                        "object_ref": nullable_string,
                        "value_text": nullable_string,
                        "valid_from": nullable_string,
                        "valid_to": nullable_string,
                        "attributes": attributes_schema,
                    },
                    "required": [
                        "proposition_id",
                        "kind",
                        "summary",
                        "details",
                        "predicate",
                        "confidence",
                        "sensitivity",
                        "source_channel",
                        "subject_ref",
                        "object_ref",
                        "value_text",
                        "valid_from",
                        "valid_to",
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
                        "valid_from": nullable_string,
                        "valid_to": nullable_string,
                        "attributes": attributes_schema,
                    },
                    "required": [
                        "source_ref",
                        "edge_type",
                        "target_ref",
                        "confidence",
                        "confirmed_by_user",
                        "valid_from",
                        "valid_to",
                        "attributes",
                    ],
                },
            },
        },
        "required": ["propositions", "graph_edges"],
    }


__all__ = [
    "LongTermStructuredTurnProgram",
    "LongTermTurnPropositionBundleV1",
    "LongTermTurnPropositionCompiler",
    "OpenAIStructuredTurnProgram",
    "_turn_proposition_schema",
    "structured_turn_program_from_config",
]
