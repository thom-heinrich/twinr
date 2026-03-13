from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class ConversationTurn:
    role: str
    content: str
    created_at: datetime = field(default_factory=_utcnow)


@dataclass(slots=True)
class MemoryLedgerItem:
    kind: str
    content: str
    created_at: datetime = field(default_factory=_utcnow)
    source: str = "conversation"
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class SearchMemoryEntry:
    question: str
    answer: str
    sources: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=_utcnow)
    location_hint: str | None = None
    date_context: str | None = None


@dataclass(slots=True)
class MemoryState:
    active_topic: str | None = None
    last_user_goal: str | None = None
    pending_printable: str | None = None
    last_search_summary: str | None = None
    open_loops: tuple[str, ...] = ()


class OnDeviceMemory:
    def __init__(self, max_turns: int = 12, keep_recent: int = 6) -> None:
        if max_turns < 3:
            raise ValueError("max_turns must be at least 3")
        if keep_recent < 1 or keep_recent >= max_turns:
            raise ValueError("keep_recent must be between 1 and max_turns - 1")
        self.max_turns = max_turns
        self.keep_recent = keep_recent
        self._raw_tail: list[ConversationTurn] = []
        self._ledger: list[MemoryLedgerItem] = []
        self._search_results: list[SearchMemoryEntry] = []
        self._state = MemoryState()

    @property
    def turns(self) -> tuple[ConversationTurn, ...]:
        summary_turn = self._build_summary_turn()
        if summary_turn is None:
            return tuple(self._raw_tail)
        return (summary_turn, *self._raw_tail)

    @property
    def raw_tail(self) -> tuple[ConversationTurn, ...]:
        return tuple(self._raw_tail)

    @property
    def ledger(self) -> tuple[MemoryLedgerItem, ...]:
        return tuple(self._ledger)

    @property
    def search_results(self) -> tuple[SearchMemoryEntry, ...]:
        return tuple(self._search_results)

    @property
    def state(self) -> MemoryState:
        return MemoryState(
            active_topic=self._state.active_topic,
            last_user_goal=self._state.last_user_goal,
            pending_printable=self._state.pending_printable,
            last_search_summary=self._state.last_search_summary,
            open_loops=tuple(self._state.open_loops),
        )

    def restore(self, turns: tuple[ConversationTurn, ...]) -> None:
        self._raw_tail = []
        self._ledger = []
        self._search_results = []
        self._state = MemoryState()
        for turn in turns:
            if turn.role == "system" and turn.content.strip():
                self._remember_ledger_item(
                    MemoryLedgerItem(
                        kind="conversation_summary",
                        content=turn.content.strip(),
                        created_at=turn.created_at,
                        source="snapshot",
                    )
                )
                continue
            self._raw_tail.append(
                ConversationTurn(
                    role=turn.role,
                    content=turn.content.strip(),
                    created_at=turn.created_at,
                )
            )
        self._trim_raw_tail()
        self._rebuild_state()

    def restore_structured(
        self,
        *,
        raw_tail: tuple[ConversationTurn, ...],
        ledger: tuple[MemoryLedgerItem, ...],
        search_results: tuple[SearchMemoryEntry, ...],
        state: MemoryState | None,
    ) -> None:
        self._raw_tail = [
            ConversationTurn(role=turn.role, content=turn.content.strip(), created_at=turn.created_at)
            for turn in raw_tail
            if turn.content.strip()
        ]
        self._ledger = [
            MemoryLedgerItem(
                kind=item.kind,
                content=item.content.strip(),
                created_at=item.created_at,
                source=item.source,
                metadata=dict(item.metadata),
            )
            for item in ledger
            if item.content.strip()
        ]
        self._search_results = [
            SearchMemoryEntry(
                question=item.question.strip(),
                answer=item.answer.strip(),
                sources=tuple(item.sources),
                created_at=item.created_at,
                location_hint=item.location_hint.strip() if item.location_hint else None,
                date_context=item.date_context.strip() if item.date_context else None,
            )
            for item in search_results
            if item.question.strip() and item.answer.strip()
        ]
        if state is None:
            self._state = MemoryState()
        else:
            self._state = MemoryState(
                active_topic=state.active_topic,
                last_user_goal=state.last_user_goal,
                pending_printable=state.pending_printable,
                last_search_summary=state.last_search_summary,
                open_loops=tuple(state.open_loops),
            )
        self._trim_raw_tail()
        self._trim_ledger()
        self._trim_search_results()
        self._rebuild_state(preserve_search_summary=state is not None)

    def remember(self, role: str, content: str) -> ConversationTurn:
        clean_content = content.strip()
        turn = ConversationTurn(role=role, content=clean_content)
        self._raw_tail.append(turn)
        self._update_state_from_turn(turn)
        if len(self._raw_tail) > self.max_turns:
            self._compact()
        return turn

    def remember_note(
        self,
        *,
        kind: str,
        content: str,
        source: str = "tool",
        metadata: dict[str, str] | None = None,
    ) -> MemoryLedgerItem:
        clean_kind = self._short_text(kind.strip().lower(), limit=32) or "memory"
        clean_content = self._short_text(content, limit=320)
        if not clean_content:
            raise ValueError("memory notes require content")
        item = MemoryLedgerItem(
            kind=clean_kind,
            content=clean_content,
            created_at=_utcnow(),
            source=source,
            metadata=dict(metadata or {}),
        )
        self._remember_ledger_item(item)
        if clean_kind in {"fact", "preference"}:
            self._state.last_user_goal = clean_content
        return item

    def remember_search(
        self,
        *,
        question: str,
        answer: str,
        sources: tuple[str, ...] = (),
        location_hint: str | None = None,
        date_context: str | None = None,
    ) -> SearchMemoryEntry:
        entry = SearchMemoryEntry(
            question=question.strip(),
            answer=answer.strip(),
            sources=tuple(source.strip() for source in sources if source.strip()),
            location_hint=location_hint.strip() if location_hint else None,
            date_context=date_context.strip() if date_context else None,
        )
        if not entry.question or not entry.answer:
            raise ValueError("search memory entries require both question and answer")
        self._search_results.append(entry)
        self._trim_search_results()
        self._remember_ledger_item(
            MemoryLedgerItem(
                kind="search_result",
                content=self._format_search_ledger_text(entry),
                created_at=entry.created_at,
                source="search",
                metadata={
                    "sources_count": str(len(entry.sources)),
                    "location_hint": entry.location_hint or "",
                    "date_context": entry.date_context or "",
                },
            )
        )
        self._state.active_topic = self._short_text(entry.question, limit=120)
        self._state.last_user_goal = self._short_text(entry.question, limit=160)
        self._state.last_search_summary = self._short_text(entry.answer, limit=220)
        if not self._state.pending_printable:
            self._state.pending_printable = self._short_text(entry.answer, limit=220)
        return entry

    def last_assistant_message(self) -> str | None:
        for turn in reversed(self._raw_tail):
            if turn.role == "assistant":
                return turn.content
        return None

    def _compact(self) -> None:
        older_turns = self._raw_tail[:-self.keep_recent]
        recent_turns = self._raw_tail[-self.keep_recent:]
        for entry in self._summarize_older_turns(tuple(older_turns)):
            self._remember_ledger_item(entry)
        self._raw_tail = list(recent_turns)
        self._rebuild_state()

    def _trim_raw_tail(self) -> None:
        if len(self._raw_tail) > self.max_turns:
            self._raw_tail = self._raw_tail[-self.max_turns :]

    def _trim_ledger(self) -> None:
        max_entries = max(self.max_turns * 3, 12)
        if len(self._ledger) > max_entries:
            self._ledger = self._ledger[-max_entries:]

    def _trim_search_results(self) -> None:
        max_entries = max(2, min(6, self.keep_recent))
        if len(self._search_results) > max_entries:
            self._search_results = self._search_results[-max_entries:]

    def _summarize_older_turns(self, turns: tuple[ConversationTurn, ...]) -> tuple[MemoryLedgerItem, ...]:
        if not turns:
            return ()
        grouped: list[list[ConversationTurn]] = []
        current: list[ConversationTurn] = []
        for turn in turns:
            if turn.role == "user" and current:
                grouped.append(current)
                current = [turn]
                continue
            current.append(turn)
            if turn.role == "assistant" and current and current[0].role == "user":
                grouped.append(current)
                current = []
        if current:
            grouped.append(current)

        summaries: list[MemoryLedgerItem] = []
        for group in grouped:
            user_parts = [turn.content for turn in group if turn.role == "user" and turn.content]
            assistant_parts = [turn.content for turn in group if turn.role == "assistant" and turn.content]
            if user_parts and assistant_parts:
                content = (
                    f"User asked: {self._short_text(' '.join(user_parts), limit=180)} "
                    f"Twinr answered: {self._short_text(' '.join(assistant_parts), limit=220)}"
                )
            elif user_parts:
                content = f"User said: {self._short_text(' '.join(user_parts), limit=220)}"
            else:
                content = f"Twinr said: {self._short_text(' '.join(assistant_parts), limit=220)}"
            summaries.append(
                MemoryLedgerItem(
                    kind="conversation_summary",
                    content=content.strip(),
                    created_at=group[-1].created_at,
                    source="compactor",
                )
            )
        return tuple(summaries)

    def _remember_ledger_item(self, item: MemoryLedgerItem) -> None:
        normalized = self._ledger_key(item.kind, item.content)
        merged_metadata = {
            key: value
            for key, value in dict(item.metadata).items()
            if str(value or "").strip()
        }
        for index, existing in enumerate(self._ledger):
            if self._ledger_key(existing.kind, existing.content) != normalized:
                continue
            self._ledger[index] = MemoryLedgerItem(
                kind=item.kind,
                content=item.content.strip(),
                created_at=item.created_at,
                source=item.source,
                metadata={**existing.metadata, **merged_metadata},
            )
            self._trim_ledger()
            return
        self._ledger.append(
            MemoryLedgerItem(
                kind=item.kind,
                content=item.content.strip(),
                created_at=item.created_at,
                source=item.source,
                metadata=merged_metadata,
            )
        )
        self._trim_ledger()

    def _build_summary_turn(self) -> ConversationTurn | None:
        summary_lines: list[str] = []
        if self._state.active_topic:
            summary_lines.append(f"Active topic: {self._state.active_topic}")
        if self._state.last_user_goal:
            summary_lines.append(f"Last user goal: {self._state.last_user_goal}")
        if self._state.open_loops:
            summary_lines.append("Open loops: " + " | ".join(self._state.open_loops[:2]))
        for entry in self._search_results[-2:]:
            summary_lines.append(
                "Verified web lookup: "
                f"{self._short_text(entry.question, limit=120)} -> {self._short_text(entry.answer, limit=220)}"
            )
        ledger_lines = [
            self._format_ledger_summary_line(item)
            for item in self._ledger[-3:]
            if item.kind != "search_result"
        ]
        summary_lines.extend(line for line in ledger_lines if line)
        if not summary_lines:
            return None
        return ConversationTurn(
            role="system",
            content="Twinr memory summary:\n" + "\n".join(f"- {line}" for line in summary_lines),
            created_at=self._summary_timestamp(),
        )

    def _rebuild_state(self, *, preserve_search_summary: bool = True) -> None:
        latest_user = next((turn.content for turn in reversed(self._raw_tail) if turn.role == "user"), None)
        latest_assistant = next((turn.content for turn in reversed(self._raw_tail) if turn.role == "assistant"), None)
        latest_search = self._search_results[-1] if self._search_results else None
        self._state.active_topic = self._short_text(
            latest_user or (latest_search.question if latest_search is not None else ""),
            limit=120,
        ) or self._state.active_topic
        self._state.last_user_goal = self._short_text(
            latest_user or (latest_search.question if latest_search is not None else ""),
            limit=160,
        ) or self._state.last_user_goal
        self._state.pending_printable = self._short_text(latest_assistant or "", limit=220) or self._state.pending_printable
        if latest_search is not None:
            self._state.last_search_summary = self._short_text(latest_search.answer, limit=220)
        elif not preserve_search_summary:
            self._state.last_search_summary = None

        if self._raw_tail and self._raw_tail[-1].role == "user":
            self._state.open_loops = (self._short_text(self._raw_tail[-1].content, limit=120),)
        else:
            self._state.open_loops = ()

    def _update_state_from_turn(self, turn: ConversationTurn) -> None:
        if turn.role == "user":
            self._state.active_topic = self._short_text(turn.content, limit=120)
            self._state.last_user_goal = self._short_text(turn.content, limit=160)
            self._state.open_loops = (self._short_text(turn.content, limit=120),)
            return
        if turn.role == "assistant":
            self._state.pending_printable = self._short_text(turn.content, limit=220)
            self._state.open_loops = ()

    def _summary_timestamp(self) -> datetime:
        candidates: list[datetime] = []
        if self._raw_tail:
            candidates.append(self._raw_tail[-1].created_at)
        if self._ledger:
            candidates.append(self._ledger[-1].created_at)
        if self._search_results:
            candidates.append(self._search_results[-1].created_at)
        return max(candidates) if candidates else _utcnow()

    def _format_search_ledger_text(self, entry: SearchMemoryEntry) -> str:
        return (
            f"Question: {self._short_text(entry.question, limit=140)} "
            f"Answer: {self._short_text(entry.answer, limit=220)}"
        )

    def _format_ledger_summary_line(self, item: MemoryLedgerItem) -> str:
        kind_prefix = {
            "conversation_summary": "Earlier context",
            "search_result": "Verified web lookup",
            "fact": "Fact",
            "preference": "Preference",
        }.get(item.kind, "Memory")
        return f"{kind_prefix}: {self._short_text(item.content, limit=240)}"

    def _ledger_key(self, kind: str, content: str) -> tuple[str, str]:
        return kind.strip().lower(), " ".join(content.strip().lower().split())

    def _short_text(self, text: str, *, limit: int) -> str:
        normalized = " ".join(str(text or "").split()).strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: max(limit - 1, 0)].rstrip() + "…"
