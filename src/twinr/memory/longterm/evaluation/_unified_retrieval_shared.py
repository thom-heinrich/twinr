"""Shared fixture, case, and evaluation helpers for unified retrieval quality checks.

This module defines one fixed long-term memory fixture that exercises the
unified retrieval planner across durable, conflict, midterm, episodic,
adaptive, and graph-backed sources. Both the deterministic goldset runner and
the live remote-memory acceptance reuse these same cases so local quality
signals and Pi/live signals stay directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Iterable, Mapping

from twinr.memory.longterm.core.models import (
    LongTermMemoryConflictV1,
    LongTermMemoryContext,
    LongTermMemoryObjectV1,
    LongTermMidtermPacketV1,
    LongTermSourceRefV1,
)
from twinr.memory.longterm.runtime.service import LongTermMemoryService
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError
from twinr.memory.query_normalization import LongTermQueryProfile
from twinr.text_utils import folded_lookup_text


_FIXTURE_OCCURRED_AT = datetime(2026, 3, 30, 10, 0, tzinfo=timezone.utc)
_REMOTE_GRAPH_QUERY_FIRST_ACCESS_PATH = (
    "catalog_current_head",
    "topk_scope_query",
    "retrieve_batch",
    "graph_path_query",
    "graph_neighbors_query",
)
_FORBIDDEN_REMOTE_ACCESS_PATH = (
    "documents_full",
    "legacy_snapshot_compat",
    "graph_document_load",
)
_CONFLICT_SLOT_KEY = "contact:person:corinna_maier:phone"
_CORINNA_PHONE_OLD = "+15555551234"
_CORINNA_PHONE_CURRENT = "+15555558877"
_CORINNA_GRAPH_PHONE = "5551234"
_ANNA_EMAIL = "anna.becker@example.com"
_SECTION_NAMES = (
    "subtext_context",
    "midterm_context",
    "durable_context",
    "episodic_context",
    "graph_context",
    "conflict_context",
)


def _normalize_text(value: object | None) -> str:
    """Return one normalized single-line string."""

    return " ".join(str(value or "").split()).strip()


def _coerce_text_tuple(values: Iterable[object] | object | None) -> tuple[str, ...]:
    """Normalize text-like values into a compact unique tuple."""

    if values is None:
        return ()
    if isinstance(values, str):
        normalized = _normalize_text(values)
        return (normalized,) if normalized else ()
    result: list[str] = []
    seen: set[str] = set()
    for item in values if isinstance(values, Iterable) else (values,):
        normalized = _normalize_text(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def _coerce_pair_tuple(
    values: Mapping[str, Iterable[object] | object | None]
    | Iterable[tuple[str, Iterable[object] | object | None]]
    | None,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Normalize mapping-like expectations into stable tuple pairs."""

    if values is None:
        return ()
    items = values.items() if isinstance(values, Mapping) else values
    normalized_pairs: list[tuple[str, tuple[str, ...]]] = []
    for key, raw_values in items:
        normalized_key = _normalize_text(key)
        normalized_values = _coerce_text_tuple(raw_values)
        if normalized_key and normalized_values:
            normalized_pairs.append((normalized_key, normalized_values))
    return tuple(sorted(normalized_pairs, key=lambda item: item[0]))


def _pairs_to_dict(
    values: tuple[tuple[str, tuple[str, ...]], ...] | Mapping[str, Iterable[object] | object | None],
) -> dict[str, tuple[str, ...]]:
    """Convert stable tuple-pair payloads into a dictionary view."""

    if isinstance(values, Mapping):
        return {
            _normalize_text(key): _coerce_text_tuple(raw_values)
            for key, raw_values in values.items()
            if _normalize_text(key)
        }
    return {key: tuple(raw_values) for key, raw_values in values}


def _normalized_lookup_text(text: object | None) -> str:
    """Fold and normalize text for deterministic contains checks."""

    return " ".join(folded_lookup_text(str(text or "")).split())


def _matched_terms(text: str | None, expected_terms: Iterable[str]) -> tuple[str, ...]:
    """Return the expected terms that are present inside one text block."""

    normalized = _normalized_lookup_text(text)
    matches: list[str] = []
    for item in _coerce_text_tuple(expected_terms):
        if _normalized_lookup_text(item) in normalized:
            matches.append(item)
    return tuple(matches)


def _missing_terms(text: str | None, expected_terms: Iterable[str]) -> tuple[str, ...]:
    """Return the expected terms that are missing from one text block."""

    matched = set(_matched_terms(text, expected_terms))
    return tuple(item for item in _coerce_text_tuple(expected_terms) if item not in matched)


def _context_sections(context: LongTermMemoryContext) -> dict[str, str]:
    """Return a normalized mapping of context section name to rendered content."""

    sections: dict[str, str] = {}
    for field_name in _SECTION_NAMES:
        value = getattr(context, field_name, None)
        if isinstance(value, str) and value.strip():
            sections[field_name] = value
    return sections


def _source(event_id: str) -> LongTermSourceRefV1:
    """Build one canonical source reference for seeded retrieval fixtures."""

    return LongTermSourceRefV1(
        source_type="unified_retrieval_goldset",
        event_ids=(event_id,),
        speaker="user",
        modality="text",
    )


@dataclass(frozen=True, slots=True)
class UnifiedRetrievalGoldsetCase:
    """Define one fixed unified-retrieval quality case."""

    case_id: str
    query_text: str
    canonical_query_text: str
    required_candidate_sources: tuple[str, ...] = ()
    required_selected_ids: tuple[tuple[str, tuple[str, ...]], ...] = ()
    required_join_anchors: tuple[tuple[str, tuple[str, ...]], ...] = ()
    required_access_path: tuple[str, ...] = ()
    forbidden_access_path: tuple[str, ...] = ()
    required_sections: tuple[str, ...] = ()
    forbidden_sections: tuple[str, ...] = ()
    required_context_terms: tuple[tuple[str, tuple[str, ...]], ...] = ()
    forbidden_context_terms: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _normalize_text(self.case_id))
        object.__setattr__(self, "query_text", _normalize_text(self.query_text))
        object.__setattr__(self, "canonical_query_text", _normalize_text(self.canonical_query_text))
        object.__setattr__(self, "required_candidate_sources", _coerce_text_tuple(self.required_candidate_sources))
        object.__setattr__(self, "required_selected_ids", _coerce_pair_tuple(self.required_selected_ids))
        object.__setattr__(self, "required_join_anchors", _coerce_pair_tuple(self.required_join_anchors))
        object.__setattr__(self, "required_access_path", _coerce_text_tuple(self.required_access_path))
        object.__setattr__(self, "forbidden_access_path", _coerce_text_tuple(self.forbidden_access_path))
        object.__setattr__(self, "required_sections", _coerce_text_tuple(self.required_sections))
        object.__setattr__(self, "forbidden_sections", _coerce_text_tuple(self.forbidden_sections))
        object.__setattr__(self, "required_context_terms", _coerce_pair_tuple(self.required_context_terms))
        object.__setattr__(self, "forbidden_context_terms", _coerce_pair_tuple(self.forbidden_context_terms))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "case_id": self.case_id,
            "query_text": self.query_text,
            "canonical_query_text": self.canonical_query_text,
            "required_candidate_sources": list(self.required_candidate_sources),
            "required_selected_ids": {
                key: list(values) for key, values in self.required_selected_ids
            },
            "required_join_anchors": {
                key: list(values) for key, values in self.required_join_anchors
            },
            "required_access_path": list(self.required_access_path),
            "forbidden_access_path": list(self.forbidden_access_path),
            "required_sections": list(self.required_sections),
            "forbidden_sections": list(self.forbidden_sections),
            "required_context_terms": {
                key: list(values) for key, values in self.required_context_terms
            },
            "forbidden_context_terms": {
                key: list(values) for key, values in self.forbidden_context_terms
            },
        }


@dataclass(frozen=True, slots=True)
class UnifiedRetrievalFixtureSeedStats:
    """Describe the seeded fixed fixture used by unified retrieval quality checks."""

    episodic_objects: int
    durable_objects: int
    conflict_count: int
    midterm_packets: int
    graph_nodes: int
    graph_edges: int

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serializable representation."""

        return {
            "episodic_objects": int(self.episodic_objects),
            "durable_objects": int(self.durable_objects),
            "conflict_count": int(self.conflict_count),
            "midterm_packets": int(self.midterm_packets),
            "graph_nodes": int(self.graph_nodes),
            "graph_edges": int(self.graph_edges),
        }


@dataclass(frozen=True, slots=True)
class UnifiedRetrievalGoldsetCaseResult:
    """Capture the outcome of one unified retrieval quality case."""

    case_id: str
    phase: str
    query_text: str
    candidate_sources: tuple[str, ...] = ()
    access_path: tuple[str, ...] = ()
    graph_mode: str | None = None
    selected_ids: tuple[tuple[str, tuple[str, ...]], ...] = ()
    join_anchors: tuple[tuple[str, tuple[str, ...]], ...] = ()
    present_sections: tuple[str, ...] = ()
    missing_candidate_sources: tuple[str, ...] = ()
    missing_access_path: tuple[str, ...] = ()
    forbidden_access_path: tuple[str, ...] = ()
    missing_selected_ids: tuple[tuple[str, tuple[str, ...]], ...] = ()
    missing_join_anchors: tuple[tuple[str, tuple[str, ...]], ...] = ()
    missing_sections: tuple[str, ...] = ()
    unexpected_sections: tuple[str, ...] = ()
    missing_context_terms: tuple[tuple[str, tuple[str, ...]], ...] = ()
    leaked_context_terms: tuple[tuple[str, tuple[str, ...]], ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _normalize_text(self.case_id))
        object.__setattr__(self, "phase", _normalize_text(self.phase))
        object.__setattr__(self, "query_text", _normalize_text(self.query_text))
        object.__setattr__(self, "candidate_sources", _coerce_text_tuple(self.candidate_sources))
        object.__setattr__(self, "access_path", _coerce_text_tuple(self.access_path))
        object.__setattr__(self, "graph_mode", _normalize_text(self.graph_mode) or None)
        object.__setattr__(self, "selected_ids", _coerce_pair_tuple(self.selected_ids))
        object.__setattr__(self, "join_anchors", _coerce_pair_tuple(self.join_anchors))
        object.__setattr__(self, "present_sections", _coerce_text_tuple(self.present_sections))
        object.__setattr__(self, "missing_candidate_sources", _coerce_text_tuple(self.missing_candidate_sources))
        object.__setattr__(self, "missing_access_path", _coerce_text_tuple(self.missing_access_path))
        object.__setattr__(self, "forbidden_access_path", _coerce_text_tuple(self.forbidden_access_path))
        object.__setattr__(self, "missing_selected_ids", _coerce_pair_tuple(self.missing_selected_ids))
        object.__setattr__(self, "missing_join_anchors", _coerce_pair_tuple(self.missing_join_anchors))
        object.__setattr__(self, "missing_sections", _coerce_text_tuple(self.missing_sections))
        object.__setattr__(self, "unexpected_sections", _coerce_text_tuple(self.unexpected_sections))
        object.__setattr__(self, "missing_context_terms", _coerce_pair_tuple(self.missing_context_terms))
        object.__setattr__(self, "leaked_context_terms", _coerce_pair_tuple(self.leaked_context_terms))
        object.__setattr__(self, "error", _normalize_text(self.error) or None)

    @property
    def passed(self) -> bool:
        """Return whether the case satisfied all required assertions."""

        return not any(
            (
                self.missing_candidate_sources,
                self.missing_access_path,
                self.forbidden_access_path,
                self.missing_selected_ids,
                self.missing_join_anchors,
                self.missing_sections,
                self.unexpected_sections,
                self.missing_context_terms,
                self.leaked_context_terms,
                self.error,
            )
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "case_id": self.case_id,
            "phase": self.phase,
            "query_text": self.query_text,
            "passed": self.passed,
            "candidate_sources": list(self.candidate_sources),
            "access_path": list(self.access_path),
            "graph_mode": self.graph_mode,
            "selected_ids": {key: list(values) for key, values in self.selected_ids},
            "join_anchors": {key: list(values) for key, values in self.join_anchors},
            "present_sections": list(self.present_sections),
            "missing_candidate_sources": list(self.missing_candidate_sources),
            "missing_access_path": list(self.missing_access_path),
            "forbidden_access_path": list(self.forbidden_access_path),
            "missing_selected_ids": {key: list(values) for key, values in self.missing_selected_ids},
            "missing_join_anchors": {key: list(values) for key, values in self.missing_join_anchors},
            "missing_sections": list(self.missing_sections),
            "unexpected_sections": list(self.unexpected_sections),
            "missing_context_terms": {key: list(values) for key, values in self.missing_context_terms},
            "leaked_context_terms": {key: list(values) for key, values in self.leaked_context_terms},
            "error": self.error,
        }


def default_unified_retrieval_goldset_cases() -> tuple[UnifiedRetrievalGoldsetCase, ...]:
    """Return the fixed unified-retrieval quality cases shared by both runners."""

    full_stack_access = ("structured_query_first", *_REMOTE_GRAPH_QUERY_FIRST_ACCESS_PATH)
    graph_only_access = _REMOTE_GRAPH_QUERY_FIRST_ACCESS_PATH
    return (
        UnifiedRetrievalGoldsetCase(
            case_id="corinna_phone_full_stack",
            query_text="What is Corinna Maier's phone number?",
            canonical_query_text="What is Corinna Maier's phone number?",
            required_candidate_sources=("adaptive", "conflict", "durable", "episodic", "graph", "midterm"),
            required_selected_ids={
                "episodic_entry_ids": ("episode:corinna_called",),
                "durable_memory_ids": ("fact:corinna_phone_current", "fact:corinna_phone_old"),
                "conflict_slot_keys": (_CONFLICT_SLOT_KEY,),
                "midterm_packet_ids": ("midterm:corinna_today",),
                "adaptive_packet_ids": ("adaptive:confirmed:fact_corinna_phone_current",),
                "graph_node_ids": ("person:corinna_maier",),
            },
            required_join_anchors={
                "person_ref:person:corinna_maier": ("adaptive", "conflict", "durable", "episodic", "graph", "midterm"),
            },
            required_access_path=full_stack_access,
            forbidden_access_path=_FORBIDDEN_REMOTE_ACCESS_PATH,
            required_sections=(
                "subtext_context",
                "midterm_context",
                "durable_context",
                "episodic_context",
                "graph_context",
                "conflict_context",
            ),
            required_context_terms={
                "midterm_context": ("recent_contact_bundle",),
                "durable_context": (_CORINNA_PHONE_CURRENT,),
                "episodic_context": ("Corinna called earlier today",),
                "graph_context": ("Corinna Maier",),
                "conflict_context": (_CONFLICT_SLOT_KEY,),
            },
        ),
        UnifiedRetrievalGoldsetCase(
            case_id="corinna_recent_call_continuity",
            query_text="Did Corinna call earlier today?",
            canonical_query_text="Did Corinna call earlier today?",
            required_candidate_sources=("episodic", "graph", "midterm"),
            required_selected_ids={
                "episodic_entry_ids": ("episode:corinna_called",),
                "midterm_packet_ids": ("midterm:corinna_today",),
                "graph_node_ids": ("person:corinna_maier",),
            },
            required_join_anchors={
                "person_ref:person:corinna_maier": ("episodic", "graph", "midterm"),
            },
            required_access_path=full_stack_access,
            forbidden_access_path=_FORBIDDEN_REMOTE_ACCESS_PATH,
            required_sections=(
                "subtext_context",
                "midterm_context",
                "episodic_context",
                "graph_context",
            ),
            required_context_terms={
                "midterm_context": ("recent_contact_bundle",),
                "episodic_context": ("Corinna called earlier today",),
                "graph_context": ("Corinna Maier",),
            },
        ),
        UnifiedRetrievalGoldsetCase(
            case_id="anna_email_graph_only",
            query_text="What is Anna Becker's email address?",
            canonical_query_text="What is Anna Becker's email address?",
            required_candidate_sources=("graph",),
            required_selected_ids={
                "graph_node_ids": ("person:anna_becker",),
            },
            required_access_path=graph_only_access,
            forbidden_access_path=_FORBIDDEN_REMOTE_ACCESS_PATH,
            required_sections=("graph_context",),
            required_context_terms={
                "graph_context": ("Anna Becker", _ANNA_EMAIL),
            },
            forbidden_sections=(
                "midterm_context",
                "durable_context",
                "episodic_context",
                "conflict_context",
            ),
        ),
    )


def seed_unified_retrieval_fixture(service: LongTermMemoryService) -> UnifiedRetrievalFixtureSeedStats:
    """Seed the fixed unified-retrieval fixture into the supplied service."""

    episodic_corinna = LongTermMemoryObjectV1(
        memory_id="episode:corinna_called",
        kind="episode",
        summary='Conversation about "Corinna called earlier today."',
        details='User said: "Corinna called earlier today." Twinr answered: "I can keep that in mind."',
        source=_source("episode:corinna_called"),
        status="active",
        confidence=1.0,
        attributes={
            "person_ref": "person:corinna_maier",
            "raw_transcript": "Corinna called earlier today.",
            "raw_response": "I can keep that in mind.",
        },
        created_at=_FIXTURE_OCCURRED_AT,
        updated_at=_FIXTURE_OCCURRED_AT,
    )
    episodic_janina = LongTermMemoryObjectV1(
        memory_id="episode:janina_called",
        kind="episode",
        summary='Conversation about "Janina called earlier today."',
        details='User said: "Janina called earlier today."',
        source=_source("episode:janina_called"),
        status="active",
        confidence=1.0,
        attributes={"person_ref": "person:janina"},
        created_at=_FIXTURE_OCCURRED_AT,
        updated_at=_FIXTURE_OCCURRED_AT,
    )
    durable_old = LongTermMemoryObjectV1(
        memory_id="fact:corinna_phone_old",
        kind="contact_method_fact",
        summary=f"Corinna Maier used to use the number {_CORINNA_PHONE_OLD}.",
        details="Older phone number for Corinna Maier.",
        source=_source("fact:corinna_phone_old"),
        status="active",
        confidence=0.82,
        confirmed_by_user=False,
        slot_key=_CONFLICT_SLOT_KEY,
        value_key=_CORINNA_PHONE_OLD,
        attributes={
            "person_ref": "person:corinna_maier",
            "support_count": 1,
        },
        created_at=_FIXTURE_OCCURRED_AT,
        updated_at=_FIXTURE_OCCURRED_AT,
    )
    durable_current = LongTermMemoryObjectV1(
        memory_id="fact:corinna_phone_current",
        kind="contact_method_fact",
        summary=f"Corinna Maier currently uses the number {_CORINNA_PHONE_CURRENT}.",
        details="Current confirmed phone number for Corinna Maier.",
        source=_source("fact:corinna_phone_current"),
        status="active",
        confidence=0.98,
        confirmed_by_user=True,
        slot_key=_CONFLICT_SLOT_KEY,
        value_key=_CORINNA_PHONE_CURRENT,
        attributes={
            "person_ref": "person:corinna_maier",
            "support_count": 3,
        },
        created_at=_FIXTURE_OCCURRED_AT,
        updated_at=_FIXTURE_OCCURRED_AT,
    )
    durable_unrelated = LongTermMemoryObjectV1(
        memory_id="fact:janina_phone",
        kind="contact_method_fact",
        summary="Janina can be reached at +15555550001.",
        details="Unrelated distractor phone number.",
        source=_source("fact:janina_phone"),
        status="active",
        confidence=0.9,
        slot_key="contact:person:janina:phone",
        value_key="+15555550001",
        attributes={"person_ref": "person:janina"},
        created_at=_FIXTURE_OCCURRED_AT,
        updated_at=_FIXTURE_OCCURRED_AT,
    )
    conflict = LongTermMemoryConflictV1(
        slot_key=_CONFLICT_SLOT_KEY,
        candidate_memory_id=durable_current.memory_id,
        existing_memory_ids=(durable_old.memory_id,),
        question="Which phone number should I use for Corinna Maier?",
        reason="Conflicting phone numbers exist for Corinna Maier.",
    )
    midterm_corinna = LongTermMidtermPacketV1(
        packet_id="midterm:corinna_today",
        kind="recent_contact_bundle",
        summary="Corinna Maier is a recent practical contact and phone questions may require disambiguation.",
        details="Useful for recent-call continuity and current phone-number questions.",
        source_memory_ids=(durable_current.memory_id, durable_old.memory_id),
        query_hints=("corinna", "phone", "called", "today"),
        sensitivity="normal",
        attributes={"person_ref": "person:corinna_maier"},
    )
    midterm_janina = LongTermMidtermPacketV1(
        packet_id="midterm:janina_today",
        kind="recent_contact_bundle",
        summary="Janina is another recent contact.",
        details="Unrelated distractor packet.",
        source_memory_ids=(durable_unrelated.memory_id,),
        query_hints=("janina", "phone"),
        sensitivity="normal",
        attributes={"person_ref": "person:janina"},
    )

    service.object_store.write_snapshot(
        objects=(
            episodic_corinna,
            episodic_janina,
            durable_old,
            durable_current,
            durable_unrelated,
        ),
        conflicts=(conflict,),
        archived_objects=(),
    )
    service.midterm_store.save_packets(packets=(midterm_corinna, midterm_janina))
    service.graph_store.remember_contact(
        given_name="Corinna",
        family_name="Maier",
        phone=_CORINNA_GRAPH_PHONE,
        role="Physiotherapist",
    )
    service.graph_store.remember_contact(
        given_name="Anna",
        family_name="Becker",
        email=_ANNA_EMAIL,
        role="Daughter",
    )
    graph_document = service.graph_store.load_document()
    return UnifiedRetrievalFixtureSeedStats(
        episodic_objects=2,
        durable_objects=3,
        conflict_count=1,
        midterm_packets=2,
        graph_nodes=len(graph_document.nodes),
        graph_edges=len(graph_document.edges),
    )


def ensure_unified_retrieval_remote_ready(service: LongTermMemoryService) -> object:
    """Fail closed when the remote backend itself is not currently reachable.

    Goldset and live unified-retrieval runners seed their own fresh remote
    namespace immediately afterwards. They therefore must not require existing
    current heads or archive heads before the fixture has even been written.
    What they *do* need is a fail-closed proof that the remote backend, auth,
    and routing path are currently alive.
    """

    status = service.remote_status()
    if getattr(status, "ready", False):
        return status
    if service.remote_required():
        raise LongTermRemoteUnavailableError(
            getattr(status, "detail", None) or "Remote-primary long-term memory is not ready."
        )
    return status


def run_unified_retrieval_cases(
    *,
    service: LongTermMemoryService,
    cases: Iterable[UnifiedRetrievalGoldsetCase],
    phase: str,
) -> tuple[UnifiedRetrievalGoldsetCaseResult, ...]:
    """Evaluate the supplied unified-retrieval cases against one live service."""

    return tuple(_run_unified_retrieval_case(service=service, case=case, phase=phase) for case in cases)


def wait_for_unified_retrieval_cases(
    *,
    service: LongTermMemoryService,
    cases: Iterable[UnifiedRetrievalGoldsetCase],
    phase: str,
    timeout_s: float,
    poll_interval_s: float = 2.0,
) -> tuple[UnifiedRetrievalGoldsetCaseResult, ...]:
    """Poll unified-retrieval cases until all pass or the deadline expires."""

    case_list = tuple(cases)
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    last_results = run_unified_retrieval_cases(service=service, cases=case_list, phase=phase)
    while time.monotonic() < deadline and any(not item.passed for item in last_results):
        time.sleep(max(0.1, float(poll_interval_s)))
        last_results = run_unified_retrieval_cases(service=service, cases=case_list, phase=phase)
    return last_results


def _run_unified_retrieval_case(
    *,
    service: LongTermMemoryService,
    case: UnifiedRetrievalGoldsetCase,
    phase: str,
) -> UnifiedRetrievalGoldsetCaseResult:
    """Execute one unified-retrieval case against one service instance."""

    try:
        query = LongTermQueryProfile.from_text(
            case.query_text,
            canonical_english_text=case.canonical_query_text,
        )
        retriever = service.retriever
        retrieval_text = retriever._normalize_query_text(query, fallback_text=case.query_text)  # pylint: disable=protected-access
        query_texts = retriever._query_text_variants(query, fallback_text=case.query_text)  # pylint: disable=protected-access
        context_inputs = retriever._load_context_inputs(  # pylint: disable=protected-access
            query_texts=query_texts,
            retrieval_text=retrieval_text,
        )
        context = retriever.build_context(query=query, original_query_text=case.query_text)
    except Exception as exc:
        return UnifiedRetrievalGoldsetCaseResult(
            case_id=case.case_id,
            phase=phase,
            query_text=case.query_text,
            error=f"{type(exc).__name__}: {exc}",
        )

    query_plan = context_inputs.unified_query_plan or {}
    candidate_sources = tuple(
        sorted(
            {
                normalized
                for normalized in (
                    _normalize_text(item.get("source"))
                    for item in query_plan.get("candidates", ())
                    if isinstance(item, Mapping)
                )
                if normalized
            }
        )
    )
    access_path = _coerce_text_tuple(query_plan.get("access_path"))
    selected_ids = _extract_selected_ids(query_plan)
    join_anchors = _extract_join_anchors(query_plan)
    present_sections = tuple(_context_sections(context).keys())

    missing_candidate_sources = tuple(
        source for source in case.required_candidate_sources if source not in set(candidate_sources)
    )
    missing_access_path = tuple(
        item for item in case.required_access_path if item not in set(access_path)
    )
    forbidden_access_path = tuple(
        item for item in case.forbidden_access_path if item in set(access_path)
    )
    missing_selected_ids = _missing_required_pairs(case.required_selected_ids, selected_ids)
    missing_join_anchors = _missing_required_pairs(case.required_join_anchors, join_anchors)
    missing_sections = tuple(
        name for name in case.required_sections if name not in set(present_sections)
    )
    unexpected_sections = tuple(
        name for name in case.forbidden_sections if name in set(present_sections)
    )
    missing_context_terms, leaked_context_terms = _context_term_failures(
        case=case,
        context=context,
    )

    graph_mode = None
    graph_query_plan = query_plan.get("graph_query_plan")
    if isinstance(graph_query_plan, Mapping):
        graph_mode = _normalize_text(graph_query_plan.get("mode")) or None

    return UnifiedRetrievalGoldsetCaseResult(
        case_id=case.case_id,
        phase=phase,
        query_text=case.query_text,
        candidate_sources=candidate_sources,
        access_path=access_path,
        graph_mode=graph_mode,
        selected_ids=selected_ids,
        join_anchors=join_anchors,
        present_sections=present_sections,
        missing_candidate_sources=missing_candidate_sources,
        missing_access_path=missing_access_path,
        forbidden_access_path=forbidden_access_path,
        missing_selected_ids=missing_selected_ids,
        missing_join_anchors=missing_join_anchors,
        missing_sections=missing_sections,
        unexpected_sections=unexpected_sections,
        missing_context_terms=missing_context_terms,
        leaked_context_terms=leaked_context_terms,
    )


def _extract_selected_ids(query_plan: Mapping[str, object]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Extract the selected-id map from one unified query plan."""

    selected = query_plan.get("selected")
    if not isinstance(selected, Mapping):
        return ()
    payload: dict[str, tuple[str, ...]] = {}
    for key, raw_values in selected.items():
        normalized_key = _normalize_text(key)
        normalized_values = _coerce_text_tuple(raw_values)
        if normalized_key:
            payload[normalized_key] = normalized_values
    return tuple(sorted(payload.items(), key=lambda item: item[0]))


def _extract_join_anchors(query_plan: Mapping[str, object]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Extract the join-anchor map from one unified query plan."""

    anchors = query_plan.get("join_anchors")
    if not isinstance(anchors, list):
        return ()
    payload: dict[str, tuple[str, ...]] = {}
    for item in anchors:
        if not isinstance(item, Mapping):
            continue
        anchor = _normalize_text(item.get("anchor"))
        sources = _coerce_text_tuple(item.get("sources"))
        if anchor and sources:
            payload[anchor] = sources
    return tuple(sorted(payload.items(), key=lambda item: item[0]))


def _missing_required_pairs(
    required: tuple[tuple[str, tuple[str, ...]], ...],
    actual: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return missing required values from one pair-mapping assertion."""

    actual_map = _pairs_to_dict(actual)
    missing: list[tuple[str, tuple[str, ...]]] = []
    for key, required_values in required:
        actual_values = set(actual_map.get(key, ()))
        missing_values = tuple(value for value in required_values if value not in actual_values)
        if missing_values:
            missing.append((key, missing_values))
    return tuple(missing)


def _context_term_failures(
    *,
    case: UnifiedRetrievalGoldsetCase,
    context: LongTermMemoryContext,
) -> tuple[tuple[tuple[str, tuple[str, ...]], ...], tuple[tuple[str, tuple[str, ...]], ...]]:
    """Return missing and leaked context-term assertions for one case."""

    sections = _context_sections(context)
    missing: list[tuple[str, tuple[str, ...]]] = []
    leaked: list[tuple[str, tuple[str, ...]]] = []
    for key, required_terms in case.required_context_terms:
        missing_terms = _missing_terms(sections.get(key), required_terms)
        if missing_terms:
            missing.append((key, missing_terms))
    for key, forbidden_terms in case.forbidden_context_terms:
        leaked_terms = _matched_terms(sections.get(key), forbidden_terms)
        if leaked_terms:
            leaked.append((key, leaked_terms))
    return tuple(missing), tuple(leaked)


def unified_retrieval_case_summary(
    case_results: Iterable[UnifiedRetrievalGoldsetCaseResult],
) -> tuple[int, int, tuple[str, ...]]:
    """Return total, passed, and failed-case ids for one result sequence."""

    results = tuple(case_results)
    failed = tuple(item.case_id for item in results if not item.passed)
    return len(results), len(results) - len(failed), failed


__all__ = [
    "ensure_unified_retrieval_remote_ready",
    "UnifiedRetrievalFixtureSeedStats",
    "UnifiedRetrievalGoldsetCase",
    "UnifiedRetrievalGoldsetCaseResult",
    "default_unified_retrieval_goldset_cases",
    "run_unified_retrieval_cases",
    "seed_unified_retrieval_fixture",
    "unified_retrieval_case_summary",
    "wait_for_unified_retrieval_cases",
]
