from __future__ import annotations

from twinr.memory import ManagedContextEntry, MemoryLedgerItem, PersistentMemoryEntry, SearchMemoryEntry
from twinr.ops.events import compact_text


class TwinrRuntimeMemoryMixin:
    def remember_search_result(
        self,
        *,
        question: str,
        answer: str,
        sources: tuple[str, ...] = (),
        location_hint: str | None = None,
        date_context: str | None = None,
    ) -> SearchMemoryEntry:
        entry = self.memory.remember_search(
            question=question,
            answer=answer,
            sources=sources,
            location_hint=location_hint,
            date_context=date_context,
        )
        self._persist_snapshot()
        self.ops_events.append(
            event="search_result_stored",
            message="Search result stored in structured on-device memory.",
            data={
                "question_preview": compact_text(question),
                "answer_preview": compact_text(answer),
                "sources": len(sources),
            },
        )
        return entry

    def store_durable_memory(
        self,
        *,
        kind: str,
        summary: str,
        details: str | None = None,
    ) -> PersistentMemoryEntry:
        return self.long_term_memory.store_explicit_memory(
            kind=kind,
            summary=summary,
            details=details,
        )

    def update_user_profile_context(
        self,
        *,
        category: str,
        instruction: str,
    ) -> ManagedContextEntry:
        return self.long_term_memory.update_user_profile(
            category=category,
            instruction=instruction,
        )

    def update_personality_context(
        self,
        *,
        category: str,
        instruction: str,
    ) -> ManagedContextEntry:
        return self.long_term_memory.update_personality(
            category=category,
            instruction=instruction,
        )

    def flush_long_term_memory(self, *, timeout_s: float = 2.0) -> bool:
        return self.long_term_memory.flush(timeout_s=timeout_s)

    def remember_note(
        self,
        *,
        kind: str,
        content: str,
        source: str = "tool",
        metadata: dict[str, str] | None = None,
    ) -> MemoryLedgerItem:
        item = self.memory.remember_note(
            kind=kind,
            content=content,
            source=source,
            metadata=metadata,
        )
        self._persist_snapshot()
        self.ops_events.append(
            event="memory_note_stored",
            message="Structured memory note stored in on-device memory.",
            data={
                "kind": kind,
                "content_preview": compact_text(content),
            },
        )
        return item

    def remember_contact(
        self,
        *,
        given_name: str,
        family_name: str | None = None,
        phone: str | None = None,
        email: str | None = None,
        role: str | None = None,
        relation: str | None = None,
        notes: str | None = None,
        source: str = "tool",
    ):
        result = self.graph_memory.remember_contact(
            given_name=given_name,
            family_name=family_name,
            phone=phone,
            email=email,
            role=role,
            relation=relation,
            notes=notes,
        )
        if result.status != "needs_clarification":
            self.remember_note(
                kind="contact",
                content=f"Stored contact: {result.label}",
                source=source,
                metadata={"graph_node_id": result.node_id, "graph_status": result.status},
            )
            self.ops_events.append(
                event="graph_contact_saved",
                message="Structured contact memory was stored in the personal graph.",
                data={"label": compact_text(result.label), "status": result.status},
            )
        return result

    def lookup_contact(
        self,
        *,
        name: str,
        family_name: str | None = None,
        role: str | None = None,
    ):
        return self.graph_memory.lookup_contact(name=name, family_name=family_name, role=role)

    def remember_preference(
        self,
        *,
        category: str,
        value: str,
        for_product: str | None = None,
        sentiment: str = "prefer",
        details: str | None = None,
        source: str = "tool",
    ):
        result = self.graph_memory.remember_preference(
            category=category,
            value=value,
            for_product=for_product,
            sentiment=sentiment,
            details=details,
        )
        self.remember_note(
            kind="preference",
            content=f"Stored preference: {result.label}",
            source=source,
            metadata={"graph_node_id": result.node_id, "graph_edge_type": result.edge_type or ""},
        )
        self.ops_events.append(
            event="graph_preference_saved",
            message="Structured preference memory was stored in the personal graph.",
            data={"label": compact_text(result.label), "edge_type": result.edge_type or ""},
        )
        return result

    def remember_plan(
        self,
        *,
        summary: str,
        when_text: str | None = None,
        details: str | None = None,
        source: str = "tool",
    ):
        result = self.graph_memory.remember_plan(summary=summary, when_text=when_text, details=details)
        self.remember_note(
            kind="plan",
            content=f"Stored plan: {result.label}",
            source=source,
            metadata={"graph_node_id": result.node_id, "graph_edge_type": result.edge_type or ""},
        )
        self.ops_events.append(
            event="graph_plan_saved",
            message="Structured plan memory was stored in the personal graph.",
            data={"label": compact_text(result.label), "edge_type": result.edge_type or ""},
        )
        return result

    def select_long_term_memory_conflicts(
        self,
        *,
        query_text: str | None = None,
        limit: int | None = None,
    ):
        return self.long_term_memory.select_conflict_queue(
            query_text=query_text,
            limit=limit,
        )

    def resolve_long_term_memory_conflict(
        self,
        *,
        slot_key: str,
        selected_memory_id: str,
    ):
        return self.long_term_memory.resolve_conflict(
            slot_key=slot_key,
            selected_memory_id=selected_memory_id,
        )

    def reserve_long_term_proactive_candidate(self, *, now=None):
        return self.long_term_memory.reserve_proactive_candidate(now=now)

    def preview_long_term_proactive_candidate(self, *, now=None):
        return self.long_term_memory.preview_proactive_candidate(now=now)

    def reserve_specific_long_term_proactive_candidate(self, candidate, *, now=None):
        return self.long_term_memory.reserve_specific_proactive_candidate(candidate, now=now)

    def mark_long_term_proactive_candidate_delivered(
        self,
        reservation,
        *,
        delivered_at=None,
        prompt_text: str | None = None,
    ):
        return self.long_term_memory.mark_proactive_candidate_delivered(
            reservation,
            delivered_at=delivered_at,
            prompt_text=prompt_text,
        )

    def mark_long_term_proactive_candidate_skipped(
        self,
        reservation,
        *,
        reason: str,
        skipped_at=None,
    ):
        return self.long_term_memory.mark_proactive_candidate_skipped(
            reservation,
            reason=reason,
            skipped_at=skipped_at,
        )
