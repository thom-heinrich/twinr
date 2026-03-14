from __future__ import annotations

from datetime import datetime

from twinr.memory import ConversationTurn, MemoryLedgerItem, MemoryState, SearchMemoryEntry


class TwinrRuntimeSnapshotMixin:
    def _persist_snapshot(self, *, error_message: str | None = None) -> None:
        self.snapshot_store.save(
            status=self.status.value,
            memory_turns=self.memory.turns,
            memory_raw_tail=self.memory.raw_tail,
            memory_ledger=self.memory.ledger,
            memory_search_results=self.memory.search_results,
            memory_state=self.memory.state,
            last_transcript=self.last_transcript,
            last_response=self.last_response,
            error_message=error_message,
            user_voice_status=self.user_voice_status,
            user_voice_confidence=self.user_voice_confidence,
            user_voice_checked_at=self.user_voice_checked_at,
        )

    def _restore_snapshot_context(self) -> None:
        snapshot = self.snapshot_store.load()
        self.last_transcript = snapshot.last_transcript
        self.last_response = snapshot.last_response or None
        self.user_voice_status = snapshot.user_voice_status or None
        self.user_voice_confidence = snapshot.user_voice_confidence
        self.user_voice_checked_at = snapshot.user_voice_checked_at or None
        if snapshot.memory_raw_tail or snapshot.memory_ledger or snapshot.memory_search_results:
            self.memory.restore_structured(
                raw_tail=tuple(
                    ConversationTurn(
                        role=turn.role,
                        content=turn.content,
                        created_at=self._parse_snapshot_timestamp(turn.created_at),
                    )
                    for turn in snapshot.memory_raw_tail
                ),
                ledger=tuple(
                    MemoryLedgerItem(
                        kind=item.kind,
                        content=item.content,
                        created_at=self._parse_snapshot_timestamp(item.created_at),
                        source=item.source,
                        metadata=dict(item.metadata),
                    )
                    for item in snapshot.memory_ledger
                ),
                search_results=tuple(
                    SearchMemoryEntry(
                        question=item.question,
                        answer=item.answer,
                        sources=tuple(item.sources),
                        created_at=self._parse_snapshot_timestamp(item.created_at),
                        location_hint=item.location_hint,
                        date_context=item.date_context,
                    )
                    for item in snapshot.memory_search_results
                ),
                state=MemoryState(
                    active_topic=snapshot.memory_state.active_topic,
                    last_user_goal=snapshot.memory_state.last_user_goal,
                    pending_printable=snapshot.memory_state.pending_printable,
                    last_search_summary=snapshot.memory_state.last_search_summary,
                    open_loops=tuple(snapshot.memory_state.open_loops),
                ),
            )
        else:
            restored_turns: list[ConversationTurn] = []
            for turn in snapshot.memory_turns:
                created_at = self._parse_snapshot_timestamp(turn.created_at)
                restored_turns.append(
                    ConversationTurn(
                        role=turn.role,
                        content=turn.content,
                        created_at=created_at,
                    )
                )
            self.memory.restore(tuple(restored_turns))
        if not self.last_response:
            self.last_response = self.memory.last_assistant_message()

    @staticmethod
    def _parse_snapshot_timestamp(value: str) -> datetime:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.now().astimezone()
