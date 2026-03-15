from __future__ import annotations

from dataclasses import dataclass
import json

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb.personal_graph import TwinrPersonalGraphStore
from twinr.memory.context_store import PersistentMemoryEntry, PromptContextStore
from twinr.memory.fulltext import FullTextDocument, FullTextSelector
from twinr.memory.longterm.conflicts import LongTermConflictResolver
from twinr.memory.longterm.midterm_store import LongTermMidtermStore
from twinr.memory.longterm.models import LongTermConflictQueueItemV1, LongTermMemoryContext
from twinr.memory.longterm.store import LongTermStructuredStore
from twinr.memory.longterm.subtext import LongTermSubtextBuilder
from twinr.memory.query_normalization import LongTermQueryProfile
from twinr.text_utils import collapse_whitespace


@dataclass(frozen=True, slots=True)
class LongTermRetriever:
    config: TwinrConfig
    prompt_context_store: PromptContextStore
    graph_store: TwinrPersonalGraphStore
    object_store: LongTermStructuredStore
    midterm_store: LongTermMidtermStore
    conflict_resolver: LongTermConflictResolver
    subtext_builder: LongTermSubtextBuilder

    def build_context(
        self,
        *,
        query: LongTermQueryProfile,
        original_query_text: str | None = None,
    ) -> LongTermMemoryContext:
        episodic_entries = self._select_episodic_entries(query, fallback_limit=0)
        midterm_packets = self.midterm_store.select_relevant_packets(
            query.retrieval_text,
            limit=max(1, self.config.long_term_memory_midterm_limit),
        )
        durable_objects = self.object_store.select_relevant_objects(
            query_text=query.retrieval_text,
            limit=max(1, self.config.long_term_memory_recall_limit),
        )
        conflict_queue = self.select_conflict_queue(query=query)
        graph_context = self.graph_store.build_prompt_context(query.retrieval_text)
        durable_context = self._render_durable_context(durable_objects)
        episodic_context = self._render_episodic_context(episodic_entries)
        conflict_context = self._render_conflict_context(conflict_queue)
        subtext_context = self.subtext_builder.build(
            query_text=original_query_text,
            retrieval_query_text=query.retrieval_text,
            episodic_entries=self._select_episodic_entries(
                query,
                fallback_limit=max(1, self.config.long_term_memory_recall_limit),
                require_query_match=False,
            ),
        )
        return LongTermMemoryContext(
            subtext_context=subtext_context,
            midterm_context=self._render_midterm_context(midterm_packets),
            durable_context=durable_context,
            episodic_context=episodic_context,
            graph_context=graph_context,
            conflict_context=conflict_context,
        )

    def select_conflict_queue(
        self,
        *,
        query: LongTermQueryProfile,
        limit: int | None = None,
    ) -> tuple[LongTermConflictQueueItemV1, ...]:
        conflicts = self.object_store.select_open_conflicts(
            query_text=query.retrieval_text,
            limit=max(1, limit or self.config.long_term_memory_recall_limit),
        )
        return self.conflict_resolver.build_queue_items(
            conflicts=conflicts,
            objects=self.object_store.load_objects(),
        )

    def _select_episodic_entries(
        self,
        query: LongTermQueryProfile | str | None,
        *,
        fallback_limit: int = 2,
        require_query_match: bool = False,
    ) -> list[PersistentMemoryEntry]:
        entries = [
            entry
            for entry in self.prompt_context_store.memory_store.load_entries()
            if entry.kind == "episodic_turn"
        ]
        if not entries:
            return []
        limit = max(1, self.config.long_term_memory_recall_limit)
        query_profile = query if isinstance(query, LongTermQueryProfile) else LongTermQueryProfile.from_text(query)
        clean_query = collapse_whitespace(query_profile.retrieval_text)
        if clean_query:
            selector = FullTextSelector(
                tuple(
                    FullTextDocument(
                        doc_id=entry.entry_id,
                        category="episodic",
                        content=collapse_whitespace(" ".join(part for part in (entry.summary, entry.details or "") if part)),
                    )
                    for entry in entries
                )
            )
            selected_ids = selector.search(
                clean_query,
                limit=limit,
                category="episodic",
                allow_fallback=not require_query_match and fallback_limit > 0,
            )
            by_id = {entry.entry_id: entry for entry in entries}
            selected = [by_id[entry_id] for entry_id in selected_ids if entry_id in by_id]
            if not selected and not require_query_match and fallback_limit > 0:
                selected = list(entries[: min(limit, fallback_limit)])
        else:
            if require_query_match:
                return []
            selected = list(entries[:limit])
        return selected

    def _render_episodic_context(self, entries: list[PersistentMemoryEntry]) -> str | None:
        if not entries:
            return None
        payload = {
            "schema": "twinr_long_term_episodic_context_v1",
            "recent_episodes": [
                {
                    "summary": entry.summary,
                    "details": entry.details,
                    "created_at": entry.created_at.isoformat(),
                }
                for entry in entries
            ],
        }
        return (
            "Structured long-term episodic memory for this turn. Internal memory is canonical English. "
            "Use only when clearly relevant, and weave it in gently instead of reciting it. "
            "Do not announce prior memory with phrases like earlier, before, last time, or neulich unless the user directly asks to recall prior conversation. "
            "Do not invent personal details that are not grounded in the user's request or this memory context.\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

    def _render_durable_context(self, objects: tuple[object, ...]) -> str | None:
        if not objects:
            return None
        payload = {
            "schema": "twinr_long_term_durable_context_v1",
            "facts": [
                {
                    "kind": item.kind,
                    "summary": item.summary,
                    "details": item.details,
                    "status": item.status,
                    "confidence": item.confidence,
                    "valid_from": item.valid_from,
                    "valid_to": item.valid_to,
                }
                for item in objects
            ],
        }
        return (
            "Structured durable long-term memory for this turn. Internal memory is canonical English. "
            "Use these facts carefully, prefer grounded continuity over explicit memory announcements, and do not overstate uncertain details.\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

    def _render_midterm_context(self, packets: tuple[object, ...]) -> str | None:
        if not packets:
            return None
        payload = {
            "schema": "twinr_long_term_midterm_context_v1",
            "packets": [
                {
                    "packet_id": item.packet_id,
                    "kind": item.kind,
                    "summary": item.summary,
                    "details": item.details,
                    "query_hints": list(item.query_hints),
                    "source_memory_ids": list(item.source_memory_ids),
                    "valid_from": item.valid_from,
                    "valid_to": item.valid_to,
                    "sensitivity": item.sensitivity,
                }
                for item in packets
            ],
        }
        return (
            "Structured mid-term memory for this turn. This is the near-term continuity layer between immediate dialogue and durable facts. "
            "Use it to keep answers situationally aware, current, and personally grounded without explicitly narrating that memory exists.\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

    def _render_conflict_context(self, conflicts: tuple[LongTermConflictQueueItemV1, ...]) -> str | None:
        if not conflicts:
            return None
        payload = {
            "schema": "twinr_long_term_conflict_context_v1",
            "open_conflicts": [
                {
                    "slot_key": item.slot_key,
                    "question": item.question,
                    "reason": item.reason,
                    "options": [option.to_payload() for option in item.options],
                }
                for item in conflicts
            ],
        }
        return (
            "Structured unresolved long-term memory conflicts. If the current conversation naturally touches one of these, ask a short clarification instead of guessing. Use the option summaries to ask a concrete disambiguation question.\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )


__all__ = ["LongTermRetriever"]
