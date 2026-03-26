"""Extract long-term memory candidates from a single conversation turn.

This module normalizes a user transcript and assistant response, optionally
asks a structured turn program for proposition output, and emits the episode,
candidate objects, and graph edges expected by the long-term memory runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from itertools import islice
import logging
import math
from typing import Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb.schema import TWINR_GRAPH_ALLOWED_EDGE_TYPES, normalize_graph_edge_type
from twinr.memory.longterm.core.ontology import LONGTERM_GENERIC_KINDS, memory_kind_prefix, normalize_memory_kind
from twinr.memory.longterm.core.models import (
    LongTermGraphEdgeCandidateV1,
    LongTermMemoryObjectV1,
    LongTermSourceRefV1,
    LongTermTurnExtractionV1,
)
from twinr.memory.longterm.ingestion.propositions import (
    LongTermStructuredTurnProgram,
    LongTermTurnPropositionBundleV1,
    LongTermTurnPropositionCompiler,
    OpenAIStructuredTurnProgram,
    _turn_proposition_schema,
    structured_turn_program_from_config,
)
from twinr.text_utils import collapse_whitespace, slugify_identifier, truncate_text

_EXTRACTION_OBJECT_KINDS = tuple(sorted(LONGTERM_GENERIC_KINDS - {"episode"}))
_DEFAULT_TIMEZONE_NAME = "Europe/Berlin"
_MAX_SOURCE_LEN = 80
_MAX_TURN_ID_LEN = 160
_MAX_REFERENCE_LEN = 160
_MAX_ATTRIBUTE_KEY_LEN = 64
_MAX_OBJECTS = 32
_MAX_GRAPH_EDGES = 64
_MAX_ATTRIBUTES = 32
_LOGGER = logging.getLogger(__name__)
_TRUE_TOKENS = frozenset({"1", "true", "yes", "y", "on", "confirmed"})
_FALSE_TOKENS = frozenset({"0", "false", "no", "n", "off", "unconfirmed"})
_ALLOWED_IDENTIFIER_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ":._/-"
)


def _strip_control_chars(value: str) -> str:
    """Replace non-printable characters with spaces."""
    return "".join(char if char.isprintable() else " " for char in value)


def _normalize_text(value: object | None, *, limit: int | None = None) -> str:
    """Normalize text for durable turn-memory storage."""
    # AUDIT-FIX(#7): Collapse whitespace and strip control characters so blank/dirty payloads do not create junk memories.
    clean_value = collapse_whitespace(_strip_control_chars("" if value is None else str(value)))
    truncated_value = truncate_text(clean_value, limit=limit)
    return collapse_whitespace(_strip_control_chars(truncated_value or ""))


def _normalize_identifier(value: object | None, *, limit: int, fallback: str) -> str:
    """Normalize externally supplied identifiers into bounded safe text."""
    # AUDIT-FIX(#8): Sanitize externally supplied identifiers before they enter persisted state and logs.
    text = _normalize_text(value, limit=limit)
    if not text:
        return fallback
    sanitized = "".join(char if char in _ALLOWED_IDENTIFIER_CHARS else "-" for char in text)
    while "--" in sanitized:
        sanitized = sanitized.replace("--", "-")
    sanitized = sanitized.strip("-")
    return sanitized or fallback


def _quoted(value: str) -> str:
    """Quote text for human-readable episode details."""
    # AUDIT-FIX(#6): Escape backslashes before quotes so stored details remain parseable after text sanitation.
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _stable_digest(value: str, *, length: int = 12) -> str:
    """Return a short stable digest for deterministic IDs."""
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def _stable_memory_id(prefix: str, stable_basis: str, *, fallback: str) -> str:
    """Build a collision-resistant memory ID from stable input text."""
    # AUDIT-FIX(#5): Append a stable digest so slug collisions do not overwrite unrelated memories.
    slug = slugify_identifier(stable_basis, fallback=fallback)
    return f"{prefix}:{slug}:{_stable_digest(stable_basis)}"


def _normalize_confidence(value: object, *, default: float = 0.5) -> float:
    """Clamp a payload confidence score into the supported range."""
    # AUDIT-FIX(#4): Reject bool/NaN/inf and clamp out-of-range scores to keep ranking/persistence stable.
    if isinstance(value, bool):
        return default
    parsed: float
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        text = _normalize_text(value, limit=32)
        if not text:
            return default
        try:
            parsed = float(text)
        except ValueError:
            return default
    else:
        return default
    if not math.isfinite(parsed):
        return default
    return min(max(parsed, 0.0), 1.0)


def _normalize_bool(value: object, *, default: bool = False) -> bool:
    """Parse common boolean-like payload values."""
    # AUDIT-FIX(#1): Parse boolean-like payload values explicitly so "false" does not become True.
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric != 0.0
        return default
    if isinstance(value, str):
        normalized = _normalize_text(value, limit=32).lower()
        if normalized in _TRUE_TOKENS:
            return True
        if normalized in _FALSE_TOKENS:
            return False
    return default


def _resolve_timezone(timezone_name: str | None) -> tuple[str, ZoneInfo]:
    """Resolve the configured timezone or fall back deterministically."""
    normalized_name = _normalize_text(timezone_name, limit=128) or _DEFAULT_TIMEZONE_NAME
    try:
        return normalized_name, ZoneInfo(normalized_name)
    except ZoneInfoNotFoundError:
        # AUDIT-FIX(#2): Fall back deterministically when config carries an invalid timezone instead of crashing at runtime.
        _LOGGER.warning(
            "Invalid timezone_name=%r for LongTermTurnExtractor. Falling back to %s.",
            timezone_name,
            _DEFAULT_TIMEZONE_NAME,
        )
        return _DEFAULT_TIMEZONE_NAME, ZoneInfo(_DEFAULT_TIMEZONE_NAME)


@dataclass(frozen=True, slots=True)
class LongTermTurnExtractor:
    """Turn one conversation exchange into long-term memory inputs.

    Attributes:
        timezone_name: IANA timezone used for localizing timestamps.
        program: Optional structured extractor that enriches the raw turn.
        proposition_compiler: Compiler for proposition payloads and graph edges.
    """

    timezone_name: str = "Europe/Berlin"
    program: LongTermStructuredTurnProgram | None = None
    proposition_compiler: LongTermTurnPropositionCompiler = field(default_factory=LongTermTurnPropositionCompiler)
    _timezone: ZoneInfo = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate and cache the configured timezone once."""
        normalized_timezone_name, timezone = _resolve_timezone(self.timezone_name)
        object.__setattr__(self, "timezone_name", normalized_timezone_name)  # AUDIT-FIX(#2): Cache a validated timezone once.
        object.__setattr__(self, "_timezone", timezone)  # AUDIT-FIX(#2): Avoid repeated runtime ZoneInfo lookups and failures.

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        program: LongTermStructuredTurnProgram | None = None,
    ) -> "LongTermTurnExtractor":
        """Build an extractor from Twinr configuration.

        Args:
            config: Runtime configuration carrying timezone and provider settings.
            program: Optional prebuilt structured turn program override.

        Returns:
            A configured turn extractor ready for runtime use.
        """
        if program is not None:
            return cls(
                timezone_name=config.local_timezone_name,
                program=program,
            )
        extractor_program = structured_turn_program_from_config(config)
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
        source: str = "conversation",
        modality: str = "voice",
        episode_attributes: Mapping[str, object] | None = None,
    ) -> LongTermTurnExtractionV1:
        """Extract the episode and candidate memories for one turn.

        Args:
            transcript: User transcript text for the turn.
            response: Assistant response paired with the transcript.
            occurred_at: Optional event time. Naive datetimes are interpreted in
                the configured local timezone.
            turn_id: Optional stable turn identifier override.
            source: Provenance label for the turn origin such as
                ``conversation`` or ``whatsapp``.
            modality: Input modality such as ``voice`` or ``text``.
            episode_attributes: Optional extra episode-level attributes merged
                into the generated episode object before persistence.

        Returns:
            A normalized turn extraction containing the episode object plus any
            candidate objects and graph edges.

        Raises:
            ValueError: If ``transcript`` is empty after normalization.
        """
        clean_transcript = _normalize_text(transcript, limit=500)  # AUDIT-FIX(#7): Reject whitespace-only transcript input.
        clean_response = _normalize_text(response, limit=500)
        if not clean_transcript:
            raise ValueError("transcript is required.")
        occurred = self._resolve_occurred_at(occurred_at)  # AUDIT-FIX(#2): Return timezone-aware datetimes in the configured zone.
        clean_source = _normalize_identifier(source, limit=_MAX_SOURCE_LEN, fallback="conversation")  # AUDIT-FIX(#8)
        clean_modality = _normalize_identifier(modality, limit=_MAX_SOURCE_LEN, fallback="voice")  # AUDIT-FIX(#8)
        source_ref_type = clean_source if clean_modality != "voice" else "conversation_turn"

        generated_turn_id = self._build_turn_id(
            transcript=clean_transcript,
            response=clean_response,
            source=clean_source,
            occurred_at=occurred,
        )
        resolved_turn_id = (
            _normalize_identifier(turn_id, limit=_MAX_TURN_ID_LEN, fallback=generated_turn_id)
            if turn_id is not None
            else generated_turn_id
        )

        source_ref = LongTermSourceRefV1(
            source_type=source_ref_type,
            event_ids=(resolved_turn_id,),
            speaker="user",
            modality=clean_modality,
        )
        episode_metadata: dict[str, object] = {
            "raw_transcript": clean_transcript,
            "raw_response": clean_response,
            "request_source": clean_source,
            "input_modality": clean_modality,
            "structured_extraction_status": "disabled" if self.program is None else "ready",
        }
        if episode_attributes is not None:
            for raw_key, raw_value in episode_attributes.items():
                clean_key = _normalize_text(raw_key, limit=_MAX_ATTRIBUTE_KEY_LEN)
                if clean_key:
                    episode_metadata[clean_key] = raw_value
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
                if not isinstance(payload, Mapping):
                    raise TypeError("structured extraction payload must be a mapping.")
                if isinstance(payload.get("propositions"), list):
                    proposition_bundle = LongTermTurnPropositionBundleV1.from_payload(payload)
                    candidate_objects, graph_edges = self.proposition_compiler.compile(
                        bundle=proposition_bundle,
                        source_ref=source_ref,
                    )
                else:
                    candidate_objects = self._candidate_objects_from_payload(
                        payload=payload,
                        source_ref=source_ref,
                    )
                    graph_edges = self._graph_edges_from_payload(payload)
            except Exception as exc:
                # AUDIT-FIX(#9): Keep the turn usable but emit an operational signal for production debugging.
                _LOGGER.warning(
                    "Long-term turn extraction failed for turn_id=%s: %s",
                    resolved_turn_id,
                    exc,
                    exc_info=True,
                )
                episode_metadata["structured_extraction_status"] = "failed"
                episode_metadata["structured_extraction_error"] = _normalize_text(str(exc), limit=220)
        episode = LongTermMemoryObjectV1(
            memory_id=_stable_memory_id("episode", resolved_turn_id, fallback="turn"),  # AUDIT-FIX(#5)
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
            created_at=occurred,
            updated_at=occurred,
            attributes=episode_metadata,
        )
        return LongTermTurnExtractionV1(
            turn_id=resolved_turn_id,
            occurred_at=occurred,
            episode=episode,
            candidate_objects=candidate_objects,
            graph_edges=graph_edges,
        )

    def _resolve_occurred_at(self, occurred_at: datetime | None) -> datetime:
        """Resolve an optional timestamp into the configured timezone."""
        if occurred_at is None:
            return datetime.now(self._timezone)
        if occurred_at.tzinfo is None or occurred_at.tzinfo.utcoffset(occurred_at) is None:
            # AUDIT-FIX(#2): Interpret naive datetimes in the configured local timezone instead of the host timezone.
            return occurred_at.replace(tzinfo=self._timezone)
        return occurred_at.astimezone(self._timezone)

    def _build_turn_id(
        self,
        *,
        transcript: str,
        response: str,
        source: str,
        occurred_at: datetime,
    ) -> str:
        """Build a deterministic turn ID from the normalized turn payload."""
        stable_basis = "\n".join((transcript, response, source, occurred_at.isoformat()))
        stable_suffix = _stable_digest(stable_basis, length=10)
        return "turn:" + occurred_at.strftime("%Y%m%dT%H%M%S%f%z") + f":{stable_suffix}"

    def _candidate_objects_from_payload(
        self,
        *,
        payload: Mapping[str, object],
        source_ref: LongTermSourceRefV1,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Convert object payload items into deduplicated memory candidates."""
        objects_payload = payload.get("objects")
        if not isinstance(objects_payload, list):
            return ()
        candidates_by_id: dict[str, LongTermMemoryObjectV1] = {}
        for index, raw_item in enumerate(islice(objects_payload, _MAX_OBJECTS), start=1):
            # AUDIT-FIX(#3): Bound processed objects to protect the single-process RPi worker from oversized payloads.
            if not isinstance(raw_item, Mapping):
                continue
            kind = _normalize_text(raw_item.get("kind", ""), limit=64)
            summary = _normalize_text(raw_item.get("summary", ""), limit=220)
            slot_key = _normalize_text(raw_item.get("slot_key", ""), limit=220)
            value_key = _normalize_text(raw_item.get("value_key", ""), limit=220)
            attributes = self._normalize_attributes(raw_item.get("attributes"))
            normalized_kind, normalized_attributes = normalize_memory_kind(kind, attributes)
            if normalized_kind not in _EXTRACTION_OBJECT_KINDS or not summary or not slot_key or not value_key:
                continue
            confidence = _normalize_confidence(raw_item.get("confidence"), default=0.5)  # AUDIT-FIX(#4)
            sensitivity = _normalize_text(raw_item.get("sensitivity", "normal"), limit=32) or "normal"
            prefix = memory_kind_prefix(normalized_kind)
            stable_basis = f"{slot_key}:{value_key}"
            memory_id = _stable_memory_id(prefix, stable_basis, fallback=f"item_{index}")  # AUDIT-FIX(#5)
            candidate = LongTermMemoryObjectV1(
                memory_id=memory_id,
                kind=normalized_kind,
                summary=summary,
                details=_normalize_text(raw_item.get("details"), limit=320) or None,
                source=source_ref,
                status="candidate",
                confidence=confidence,
                sensitivity=sensitivity,
                slot_key=slot_key,
                value_key=value_key,
                valid_from=self._optional_text(raw_item.get("valid_from")),
                valid_to=self._optional_text(raw_item.get("valid_to")),
                attributes=normalized_attributes or None,
            )
            existing = candidates_by_id.get(memory_id)
            if existing is None or candidate.confidence > existing.confidence:
                # AUDIT-FIX(#10): De-duplicate semantically identical objects and keep the strongest candidate.
                candidates_by_id[memory_id] = candidate
        return tuple(candidates_by_id.values())

    def _graph_edges_from_payload(
        self,
        payload: Mapping[str, object],
    ) -> tuple[LongTermGraphEdgeCandidateV1, ...]:
        """Convert edge payload items into deduplicated graph candidates."""
        edges_payload = payload.get("graph_edges")
        if not isinstance(edges_payload, list):
            return ()
        edges_by_key: dict[tuple[object, ...], LongTermGraphEdgeCandidateV1] = {}
        for raw_edge in islice(edges_payload, _MAX_GRAPH_EDGES):
            # AUDIT-FIX(#3): Bound processed graph edges so malformed model output cannot monopolize the device.
            if not isinstance(raw_edge, Mapping):
                continue
            edge_type = _normalize_text(raw_edge.get("edge_type", ""), limit=80)
            attributes = self._normalize_attributes(raw_edge.get("attributes"))
            normalized_edge_type, normalized_attributes = normalize_graph_edge_type(edge_type, attributes)
            if normalized_edge_type not in TWINR_GRAPH_ALLOWED_EDGE_TYPES:
                continue
            edge_source_ref = _normalize_identifier(
                raw_edge.get("source_ref", ""),
                limit=_MAX_REFERENCE_LEN,
                fallback="",
            )  # AUDIT-FIX(#8)
            target_ref = _normalize_identifier(
                raw_edge.get("target_ref", ""),
                limit=_MAX_REFERENCE_LEN,
                fallback="",
            )  # AUDIT-FIX(#8)
            if not edge_source_ref or not target_ref:
                continue
            valid_from = self._optional_text(raw_edge.get("valid_from"))
            valid_to = self._optional_text(raw_edge.get("valid_to"))
            edge_attributes = normalized_attributes or None
            candidate = LongTermGraphEdgeCandidateV1(
                source_ref=edge_source_ref,
                edge_type=normalized_edge_type,
                target_ref=target_ref,
                confidence=_normalize_confidence(raw_edge.get("confidence"), default=0.5),  # AUDIT-FIX(#4)
                confirmed_by_user=_normalize_bool(raw_edge.get("confirmed_by_user"), default=False),  # AUDIT-FIX(#1)
                valid_from=valid_from,
                valid_to=valid_to,
                attributes=edge_attributes,
            )
            dedupe_key = (
                candidate.source_ref,
                candidate.edge_type,
                candidate.target_ref,
                candidate.confirmed_by_user,
                candidate.valid_from,
                candidate.valid_to,
                tuple(sorted((edge_attributes or {}).items())),
            )
            existing = edges_by_key.get(dedupe_key)
            if existing is None or candidate.confidence > existing.confidence:
                # AUDIT-FIX(#10): Collapse duplicate edges to reduce noisy recalls and downstream graph churn.
                edges_by_key[dedupe_key] = candidate
        return tuple(edges_by_key.values())

    def _normalize_attributes(self, value: object) -> dict[str, str]:
        """Normalize bounded attribute mappings from model payloads."""
        if isinstance(value, list):
            normalized: dict[str, str] = {}
            for item in islice(value, _MAX_ATTRIBUTES):
                # AUDIT-FIX(#3): Cap attribute fan-out to bound payload size on constrained hardware.
                if not isinstance(item, Mapping):
                    continue
                clean_key = _normalize_text(item.get("key", ""), limit=_MAX_ATTRIBUTE_KEY_LEN)
                clean_value = self._optional_text(item.get("value"))
                if clean_key and clean_value:
                    normalized[clean_key] = clean_value
            return normalized
        if not isinstance(value, Mapping):
            return {}
        normalized_mapping: dict[str, str] = {}
        for key, raw in islice(value.items(), _MAX_ATTRIBUTES):
            # AUDIT-FIX(#3): Cap attribute fan-out to bound payload size on constrained hardware.
            clean_key = _normalize_text(key, limit=_MAX_ATTRIBUTE_KEY_LEN)
            clean_value = self._optional_text(raw)
            if clean_key and clean_value:
                normalized_mapping[clean_key] = clean_value
        return normalized_mapping

    def _optional_text(self, value: object) -> str | None:
        """Return normalized text or ``None`` when empty."""
        if value is None:
            return None
        clean_value = _normalize_text(value, limit=160)
        return clean_value or None


def _turn_extraction_schema() -> dict[str, object]:
    """Expose the structured schema used for turn extraction payloads."""
    return _turn_proposition_schema()


__all__ = [
    "LongTermStructuredTurnProgram",
    "LongTermTurnExtractor",
    "OpenAIStructuredTurnProgram",
]
