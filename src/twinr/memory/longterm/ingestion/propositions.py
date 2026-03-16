from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone, tzinfo
from typing import TYPE_CHECKING, Mapping, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from twinr.agent.base_agent.config import TwinrConfig
from twinr.llm_json import request_structured_json_object
from twinr.memory.chonkydb.schema import TWINR_GRAPH_ALLOWED_EDGE_TYPES, normalize_graph_edge_type
from twinr.memory.longterm.core.ontology import LONGTERM_GENERIC_KINDS, memory_kind_prefix, normalize_memory_kind
from twinr.memory.longterm.core.models import (
    LongTermGraphEdgeCandidateV1,
    LongTermMemoryObjectV1,
    LongTermSourceRefV1,
)
from twinr.text_utils import collapse_whitespace, slugify_identifier, truncate_text

if TYPE_CHECKING:
    from twinr.providers.openai import OpenAIBackend


LOGGER = logging.getLogger(__name__)

_PROPOSITION_KINDS = tuple(sorted(LONGTERM_GENERIC_KINDS - {"episode"}))
_SENSITIVITIES = ("low", "normal", "private", "sensitive", "critical")
_SOURCE_CHANNELS = ("user_transcript", "assistant_response", "both")
_DEFAULT_TIMEZONE_NAME = "UTC"
_DEFAULT_CONFIDENCE = 0.5
_MAX_PROMPT_TEXT_CHARS = 6000
_MAX_PROMPT_FIELD_CHARS = 120


def _normalize_text(value: str | None, *, limit: int | None = None) -> str:
    # AUDIT-FIX(#6): Collapse hidden whitespace before truncation so refs/keys/ids do not diverge only by control characters.
    if value is None:
        return ""
    return truncate_text(collapse_whitespace(str(value)), limit=limit)


def _quoted(value: str) -> str:
    # AUDIT-FIX(#4): Use JSON string encoding instead of ad-hoc quote replacement so control characters and newlines stay safely delimited in the prompt.
    return json.dumps(value, ensure_ascii=False)


def _empty_turn_payload() -> dict[str, object]:
    return {
        "propositions": [],
        "graph_edges": [],
    }


def _normalize_prompt_text(value: str | None, *, limit: int = _MAX_PROMPT_TEXT_CHARS) -> str:
    # AUDIT-FIX(#4): Bound and normalize prompt payloads so accidental long turns or malformed transcripts do not explode token usage or prompt structure.
    return truncate_text(collapse_whitespace(value or ""), limit=limit)


def _normalize_positive_int(value: object, *, default: int) -> int:
    # AUDIT-FIX(#7): Coerce token budgets defensively so None/strings/invalid env-derived values cannot crash max() or pass nonsense downstream.
    if isinstance(value, int) and not isinstance(value, bool):
        return max(1, value)
    try:
        return max(1, int(str(value)))
    except (TypeError, ValueError):
        return max(1, default)


def _normalize_confidence(value: object, *, default: float = _DEFAULT_CONFIDENCE) -> float:
    # AUDIT-FIX(#5): Clamp to a finite probability-like range so malformed model payloads cannot inject NaN/inf/out-of-range scores into ranking logic.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    normalized = float(value)
    if not math.isfinite(normalized):
        return default
    if normalized < 0.0:
        return 0.0
    if normalized > 1.0:
        return 1.0
    return normalized


def _normalize_bool(value: object, *, default: bool = False) -> bool:
    # AUDIT-FIX(#5): Avoid bool("false") == True, which would silently mark unconfirmed graph edges as confirmed.
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = _normalize_text(value, limit=16).lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        return default
    if isinstance(value, int):
        return value != 0
    return default


def _normalize_enum_text(
    value: object,
    *,
    allowed: tuple[str, ...],
    default: str | None,
    limit: int,
) -> str | None:
    normalized = _normalize_text(None if value is None else str(value), limit=limit)
    if not normalized:
        return default
    if normalized in allowed:
        return normalized
    return default


def _normalize_identifier_text(value: object, *, limit: int) -> str | None:
    normalized = _normalize_text(None if value is None else str(value), limit=limit)
    return normalized or None


def _normalize_iso_temporal_text(value: object) -> str | None:
    # AUDIT-FIX(#2): Reject non-ISO or timezone-naive datetime payloads so stored temporal ranges are canonical and not host-timezone dependent.
    normalized = _normalize_identifier_text(value, limit=64)
    if normalized is None:
        return None
    try:
        parsed_date = date.fromisoformat(normalized)
    except ValueError:
        parsed_date = None
    if parsed_date is not None:
        return parsed_date.isoformat()
    try:
        parsed_datetime = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed_datetime.tzinfo is None or parsed_datetime.utcoffset() is None:
        return None
    return parsed_datetime.isoformat()


def _temporal_sort_key(value: str | None) -> tuple[int, str] | None:
    if value is None:
        return None
    try:
        parsed_date = date.fromisoformat(value)
    except ValueError:
        parsed_date = None
    if parsed_date is not None:
        return (0, parsed_date.isoformat())
    try:
        parsed_datetime = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed_datetime.tzinfo is None or parsed_datetime.utcoffset() is None:
        return None
    return (1, parsed_datetime.astimezone(timezone.utc).isoformat())


def _normalize_valid_window(valid_from: str | None, valid_to: str | None) -> tuple[str | None, str | None]:
    # AUDIT-FIX(#5): Normalize reversed ranges instead of storing logically impossible intervals.
    from_key = _temporal_sort_key(valid_from)
    to_key = _temporal_sort_key(valid_to)
    if from_key is not None and to_key is not None and to_key < from_key:
        return valid_to, valid_from
    return valid_from, valid_to


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


def _resolve_timezone(timezone_name: str) -> tuple[str, tzinfo]:
    # AUDIT-FIX(#2): Invalid zone keys or missing zoneinfo data must not abort extraction; fall back to UTC deterministically.
    normalized = _normalize_text(timezone_name, limit=64) or _DEFAULT_TIMEZONE_NAME
    try:
        return normalized, ZoneInfo(normalized)
    except ZoneInfoNotFoundError:
        LOGGER.warning("Invalid timezone_name for long-term memory extraction: %s; falling back to %s", normalized, _DEFAULT_TIMEZONE_NAME)
    try:
        return _DEFAULT_TIMEZONE_NAME, ZoneInfo(_DEFAULT_TIMEZONE_NAME)
    except ZoneInfoNotFoundError:
        LOGGER.warning("ZoneInfo database unavailable; falling back to built-in UTC timezone for long-term memory extraction")
        return _DEFAULT_TIMEZONE_NAME, timezone.utc


def _ensure_aware_datetime(value: datetime) -> datetime:
    # AUDIT-FIX(#2): astimezone() treats naive datetimes as system-local time; attach UTC explicitly so RPi host settings cannot skew memory timestamps.
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value


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
        # AUDIT-FIX(#2): Normalize temporal context before prompt construction so invalid timezones or naive datetimes do not raise or silently depend on host locale.
        resolved_timezone_name, resolved_timezone = _resolve_timezone(timezone_name)
        resolved_occurred_at = _ensure_aware_datetime(occurred_at).astimezone(resolved_timezone)
        normalized_turn_id = _normalize_text(turn_id, limit=_MAX_PROMPT_FIELD_CHARS) or "turn"
        normalized_transcript = _normalize_prompt_text(transcript)
        normalized_response = _normalize_prompt_text(response)
        if not normalized_transcript and not normalized_response:
            return _empty_turn_payload()
        try:
            payload = request_structured_json_object(
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
                        f"Turn id: {normalized_turn_id}",
                        f"Occurred at: {resolved_occurred_at.isoformat()}",
                        f"Timezone: {resolved_timezone_name}",
                        f"User transcript: {_quoted(normalized_transcript)}",
                        f"Assistant response: {_quoted(normalized_response)}",
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
                max_output_tokens=max(1200, _normalize_positive_int(self.max_output_tokens, default=2200)),
            )
        except Exception:
            # AUDIT-FIX(#1): Memory extraction is auxiliary; provider/network/schema failures must degrade to an empty payload instead of crashing the turn path.
            LOGGER.exception("Long-term memory turn extraction failed")
            return _empty_turn_payload()
        if not isinstance(payload, Mapping):
            # AUDIT-FIX(#1): Defensively reject malformed helper returns so downstream parsing always sees the documented shape.
            LOGGER.warning("Long-term memory turn extraction returned a non-mapping payload: %r", type(payload))
            return _empty_turn_payload()
        return payload


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
        # AUDIT-FIX(#5): Validate model-produced proposition fields before they reach durable memory candidate objects.
        kind = _normalize_enum_text(payload.get("kind"), allowed=_PROPOSITION_KINDS, default=None, limit=64)
        summary = _normalize_text(str(payload.get("summary", "")), limit=220)
        predicate = _normalize_text(str(payload.get("predicate", "")), limit=120)
        if not kind or not summary or not predicate:
            return None
        confidence = _normalize_confidence(payload.get("confidence", _DEFAULT_CONFIDENCE))
        sensitivity = _normalize_enum_text(
            payload.get("sensitivity", "normal"),
            allowed=_SENSITIVITIES,
            default="normal",
            limit=32,
        ) or "normal"
        source_channel = _normalize_enum_text(
            payload.get("source_channel", "assistant_response"),
            allowed=_SOURCE_CHANNELS,
            default="assistant_response",
            limit=32,
        ) or "assistant_response"
        proposition_id = _normalize_identifier_text(payload.get("proposition_id"), limit=120) or f"proposition:{index + 1}"
        valid_from = _normalize_iso_temporal_text(payload.get("valid_from"))
        valid_to = _normalize_iso_temporal_text(payload.get("valid_to"))
        valid_from, valid_to = _normalize_valid_window(valid_from, valid_to)
        return cls(
            proposition_id=proposition_id,
            kind=kind,
            summary=summary,
            predicate=predicate,
            confidence=confidence,
            sensitivity=sensitivity,
            source_channel=source_channel,
            details=_normalize_optional_text(payload.get("details")),
            subject_ref=_normalize_identifier_text(payload.get("subject_ref"), limit=160),
            object_ref=_normalize_identifier_text(payload.get("object_ref"), limit=160),
            value_text=_normalize_optional_text(payload.get("value_text")),
            valid_from=valid_from,
            valid_to=valid_to,
            attributes=_normalize_attribute_entries(payload.get("attributes")) or None,
        )


@dataclass(frozen=True, slots=True)
class LongTermTurnPropositionBundleV1:
    propositions: tuple[LongTermTurnPropositionV1, ...]
    graph_edges: tuple[LongTermGraphEdgeCandidateV1, ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "LongTermTurnPropositionBundleV1":
        # AUDIT-FIX(#5): Apply the same defensive coercion to graph-edge payloads because structured outputs do not guarantee semantic correctness.
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
                raw_attributes = _normalize_attribute_entries(raw_edge.get("attributes"))
                normalized_edge_type, normalized_attributes = normalize_graph_edge_type(edge_type, raw_attributes)
                if normalized_edge_type not in TWINR_GRAPH_ALLOWED_EDGE_TYPES:
                    continue
                source_ref = _normalize_identifier_text(raw_edge.get("source_ref"), limit=160)
                target_ref = _normalize_identifier_text(raw_edge.get("target_ref"), limit=160)
                if not source_ref or not target_ref:
                    continue
                valid_from = _normalize_iso_temporal_text(raw_edge.get("valid_from"))
                valid_to = _normalize_iso_temporal_text(raw_edge.get("valid_to"))
                valid_from, valid_to = _normalize_valid_window(valid_from, valid_to)
                edge_attributes = dict(raw_attributes)
                # AUDIT-FIX(#3): Preserve proposition grounding attributes even if edge-type normalization discards unknown keys.
                edge_attributes.update(normalized_attributes or {})
                graph_edges.append(
                    LongTermGraphEdgeCandidateV1(
                        source_ref=source_ref,
                        edge_type=normalized_edge_type,
                        target_ref=target_ref,
                        confidence=_normalize_confidence(raw_edge.get("confidence", _DEFAULT_CONFIDENCE)),
                        confirmed_by_user=_normalize_bool(raw_edge.get("confirmed_by_user"), default=False),
                        valid_from=valid_from,
                        valid_to=valid_to,
                        attributes=edge_attributes or None,
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
        accepted_refs: set[str] = set()
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
                    confidence=_normalize_confidence(proposition.confidence),
                    sensitivity=proposition.sensitivity,
                    slot_key=slot_key,
                    value_key=value_key,
                    valid_from=proposition.valid_from,
                    valid_to=proposition.valid_to,
                    attributes=attributes or None,
                )
            )
            accepted_proposition_ids.add(proposition.proposition_id)
            if proposition.subject_ref:
                accepted_refs.add(proposition.subject_ref)
            if proposition.object_ref:
                accepted_refs.add(proposition.object_ref)
            if proposition.valid_from:
                accepted_refs.add(f"day:{proposition.valid_from}")
            if proposition.valid_to:
                accepted_refs.add(f"day:{proposition.valid_to}")
        graph_edges = tuple(
            edge
            for edge in bundle.graph_edges
            if self._edge_is_allowed(
                edge=edge,
                accepted_proposition_ids=accepted_proposition_ids,
                accepted_refs=accepted_refs,
            )
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
        accepted_refs: set[str],
    ) -> bool:
        # AUDIT-FIX(#3): Only persist graph edges that are explicitly grounded by an accepted proposition; otherwise hallucinated relations can leak into long-term memory.
        if not accepted_proposition_ids:
            return False
        attributes = dict(edge.attributes or {})
        for key in ("proposition_id", "origin_proposition_id"):
            referenced = _normalize_identifier_text(attributes.get(key), limit=120)
            if referenced is not None:
                return referenced in accepted_proposition_ids
        return edge.source_ref in accepted_refs and edge.target_ref in accepted_refs


def structured_turn_program_from_config(
    config: TwinrConfig,
) -> LongTermStructuredTurnProgram | None:
    # AUDIT-FIX(#7): Normalize env-derived config before constructing the extractor so blank keys or invalid token budgets do not fail at runtime.
    if not _normalize_text(config.openai_api_key, limit=512):
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
        max_output_tokens=_normalize_positive_int(
            config.long_term_memory_turn_extractor_max_output_tokens,
            default=2200,
        ),
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
