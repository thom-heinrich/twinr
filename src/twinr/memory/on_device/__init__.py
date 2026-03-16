from __future__ import annotations

from collections.abc import Iterable, Mapping  # AUDIT-FIX(#5): Sanitize restore inputs without assuming perfect runtime types.
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Final  # AUDIT-FIX(#7): Use explicit bounded constants for in-memory payload sizes.


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
    _SUMMARY_HEADER: Final[str] = "Twinr memory summary:"  # AUDIT-FIX(#1): Detect only explicit memory-summary snapshots.
    _SUMMARY_ROLE: Final[str] = "system"  # AUDIT-FIX(#1): The synthetic summary is Twinr-authored framing for follow-up context.
    _SUMMARY_RESTORE_ROLES: Final[frozenset[str]] = frozenset({"system", "assistant"})  # AUDIT-FIX(#1): Accept older assistant-role snapshots written before the role regression was fixed.
    _ALLOWED_TURN_ROLES: Final[frozenset[str]] = frozenset({"user", "assistant"})  # AUDIT-FIX(#4): Keep conversation state limited to supported roles.
    _ROLE_ALIASES: Final[dict[str, str]] = {  # AUDIT-FIX(#4): Accept common role aliases without weakening invariants.
        "human": "user",
        "person": "user",
        "bot": "assistant",
        "ai": "assistant",
        "model": "assistant",
        "twinr": "assistant",
    }
    _MAX_TURN_CONTENT_CHARS: Final[int] = 4_000  # AUDIT-FIX(#7): Bound raw turn size for RPi-class devices.
    _MAX_NOTE_KIND_CHARS: Final[int] = 32
    _MAX_NOTE_CONTENT_CHARS: Final[int] = 320
    _MAX_LEDGER_CONTENT_CHARS: Final[int] = 1_200
    _MAX_SEARCH_QUESTION_CHARS: Final[int] = 320
    _MAX_SEARCH_ANSWER_CHARS: Final[int] = 2_000
    _MAX_SOURCE_CHARS: Final[int] = 256
    _MAX_SOURCE_COUNT: Final[int] = 8
    _MAX_METADATA_KEY_CHARS: Final[int] = 64
    _MAX_METADATA_VALUE_CHARS: Final[int] = 256
    _MAX_SUMMARY_CONTENT_CHARS: Final[int] = 2_000

    def __init__(self, max_turns: int = 20, keep_recent: int = 10) -> None:
        self._validate_limits(max_turns=max_turns, keep_recent=keep_recent)
        self.max_turns = max_turns
        self.keep_recent = keep_recent
        self._raw_tail: list[ConversationTurn] = []
        self._ledger: list[MemoryLedgerItem] = []
        self._search_results: list[SearchMemoryEntry] = []
        self._state = MemoryState()

    @property
    def turns(self) -> tuple[ConversationTurn, ...]:
        summary_turn = self._build_summary_turn()
        raw_tail = tuple(self._clone_turn(turn) for turn in self._raw_tail)  # AUDIT-FIX(#3): Do not expose mutable internal turn objects.
        if summary_turn is None:
            return raw_tail
        return (summary_turn, *raw_tail)

    @property
    def raw_tail(self) -> tuple[ConversationTurn, ...]:
        return tuple(self._clone_turn(turn) for turn in self._raw_tail)  # AUDIT-FIX(#3): Return defensive copies.

    @property
    def ledger(self) -> tuple[MemoryLedgerItem, ...]:
        return tuple(self._clone_ledger_item(item) for item in self._ledger)  # AUDIT-FIX(#3): Return defensive copies.

    @property
    def search_results(self) -> tuple[SearchMemoryEntry, ...]:
        return tuple(self._clone_search_entry(item) for item in self._search_results)  # AUDIT-FIX(#3): Return defensive copies.

    @property
    def state(self) -> MemoryState:
        return self._clone_state(self._state)  # AUDIT-FIX(#3): Keep callers from mutating internal state through returned objects.

    def reconfigure(self, *, max_turns: int, keep_recent: int) -> None:
        self._validate_limits(max_turns=max_turns, keep_recent=keep_recent)
        self.max_turns = max_turns
        self.keep_recent = keep_recent
        if len(self._raw_tail) > self.max_turns:
            self._compact_until_within_limits()  # AUDIT-FIX(#6): Preserve overflow turns via compaction instead of silent truncation.
        self._trim_raw_tail()
        self._trim_ledger()
        self._trim_search_results()
        self._rebuild_state()

    def restore(self, turns: tuple[ConversationTurn, ...]) -> None:
        self._raw_tail = []
        self._ledger = []
        self._search_results = []
        self._state = MemoryState()
        for turn in turns:
            summary_content = self._extract_summary_content(turn)  # AUDIT-FIX(#1): Only ingest explicitly marked summaries.
            if summary_content:
                self._remember_ledger_item(
                    MemoryLedgerItem(
                        kind="conversation_summary",
                        content=summary_content,
                        created_at=self._normalize_datetime(getattr(turn, "created_at", None)),  # AUDIT-FIX(#2): Normalize restored timestamps to UTC-aware datetimes.
                        source="snapshot",
                    )
                )
                continue
            normalized_turn = self._coerce_conversation_turn(turn)  # AUDIT-FIX(#5): Skip malformed restored turns instead of crashing on .strip().
            if normalized_turn is None:
                continue
            self._raw_tail.append(normalized_turn)
        self._compact_until_within_limits()  # AUDIT-FIX(#6): Summarize restored overflow rather than dropping it.
        self._trim_ledger()
        self._rebuild_state(preserve_search_summary=False)

    def restore_structured(
        self,
        *,
        raw_tail: tuple[ConversationTurn, ...],
        ledger: tuple[MemoryLedgerItem, ...],
        search_results: tuple[SearchMemoryEntry, ...],
        state: MemoryState | None,
    ) -> None:
        self._raw_tail = []
        for turn in raw_tail:
            normalized_turn = self._coerce_conversation_turn(turn)  # AUDIT-FIX(#5): Defensive restore against malformed turn payloads.
            if normalized_turn is not None:
                self._raw_tail.append(normalized_turn)

        self._ledger = []
        for item in ledger:
            normalized_item = self._coerce_ledger_item(item)  # AUDIT-FIX(#5): Defensive restore against malformed ledger payloads.
            if normalized_item is not None:
                self._ledger.append(normalized_item)

        self._search_results = []
        for item in search_results:
            normalized_entry = self._coerce_search_entry(item)  # AUDIT-FIX(#5): Defensive restore against malformed search payloads.
            if normalized_entry is not None:
                self._search_results.append(normalized_entry)

        self._state = self._coerce_state(state)  # AUDIT-FIX(#5): Sanitize persisted state before reuse.
        self._compact_until_within_limits()  # AUDIT-FIX(#6): Preserve overflow context on structured restore.
        self._trim_ledger()
        self._trim_search_results()
        self._rebuild_state(preserve_search_summary=state is not None)

    def remember(self, role: str, content: str) -> ConversationTurn:
        normalized_role = self._normalize_role(role)  # AUDIT-FIX(#4): Enforce supported conversation roles.
        if normalized_role is None:
            raise ValueError("role must be 'user' or 'assistant'")
        clean_content = self._normalize_text(content, limit=self._MAX_TURN_CONTENT_CHARS)
        if not clean_content:
            return ConversationTurn(role=normalized_role, content="", created_at=_utcnow())  # AUDIT-FIX(#4): Ignore empty turns instead of poisoning memory state.
        stored_turn = ConversationTurn(
            role=normalized_role,
            content=clean_content,
            created_at=_utcnow(),
        )
        self._raw_tail.append(stored_turn)
        self._update_state_from_turn(stored_turn)
        if len(self._raw_tail) > self.max_turns:
            self._compact()
        return self._clone_turn(stored_turn)  # AUDIT-FIX(#3): Return a detached copy, not the stored object.

    def remember_note(
        self,
        *,
        kind: str,
        content: str,
        source: str = "tool",
        metadata: dict[str, str] | None = None,
    ) -> MemoryLedgerItem:
        clean_kind = self._normalize_text(kind, limit=self._MAX_NOTE_KIND_CHARS, collapse_whitespace=True).lower() or "memory"  # AUDIT-FIX(#5): Sanitize note kind inputs.
        clean_content = self._normalize_text(content, limit=self._MAX_NOTE_CONTENT_CHARS, collapse_whitespace=True)
        if not clean_content:
            raise ValueError("memory notes require content")
        item = MemoryLedgerItem(
            kind=clean_kind,
            content=clean_content,
            created_at=_utcnow(),
            source=self._normalize_text(source, limit=self._MAX_NOTE_KIND_CHARS, collapse_whitespace=True) or "tool",  # AUDIT-FIX(#9): Sanitize provenance labels.
            metadata=self._normalize_metadata(metadata),  # AUDIT-FIX(#9): Sanitize metadata keys and values.
        )
        self._remember_ledger_item(item)
        if clean_kind in {"fact", "preference"}:
            self._state.last_user_goal = clean_content
        return self._clone_ledger_item(item)  # AUDIT-FIX(#3): Return a detached copy.

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
            question=self._normalize_text(question, limit=self._MAX_SEARCH_QUESTION_CHARS),
            answer=self._normalize_text(answer, limit=self._MAX_SEARCH_ANSWER_CHARS),
            sources=self._normalize_sources(sources),  # AUDIT-FIX(#5): Sanitize source collections from heterogeneous callers.
            created_at=_utcnow(),
            location_hint=self._normalize_optional_text(location_hint, limit=120),
            date_context=self._normalize_optional_text(date_context, limit=120),
        )
        if not entry.question or not entry.answer:
            raise ValueError("search memory entries require both question and answer")
        stored_entry = self._clone_search_entry(entry)
        self._search_results.append(stored_entry)
        self._trim_search_results()
        self._remember_ledger_item(
            MemoryLedgerItem(
                kind="search_result",
                content=self._format_search_ledger_text(stored_entry),
                created_at=stored_entry.created_at,
                source="search",
                metadata={
                    "sources_count": str(len(stored_entry.sources)),
                    "location_hint": stored_entry.location_hint or "",
                    "date_context": stored_entry.date_context or "",
                },
            )
        )
        self._state.active_topic = self._short_text(stored_entry.question, limit=120)
        self._state.last_user_goal = self._short_text(stored_entry.question, limit=160)
        self._state.last_search_summary = self._short_text(stored_entry.answer, limit=220)
        if not self._state.pending_printable:
            self._state.pending_printable = self._short_text(stored_entry.answer, limit=220)
        return self._clone_search_entry(stored_entry)  # AUDIT-FIX(#3): Return a detached copy.

    def last_assistant_message(self) -> str | None:
        for turn in reversed(self._raw_tail):
            if turn.role == "assistant":
                return turn.content
        return None

    def _compact(self) -> None:
        self._compact_until_within_limits()  # AUDIT-FIX(#6): Reuse the same safe compaction path for restore and runtime growth.
        self._rebuild_state()

    def _compact_until_within_limits(self) -> None:  # AUDIT-FIX(#6): Summarize overflow instead of slicing it away.
        while len(self._raw_tail) > self.max_turns:
            older_turns = self._raw_tail[:-self.keep_recent]
            recent_turns = self._raw_tail[-self.keep_recent:]
            if not older_turns:
                break
            for entry in self._summarize_older_turns(tuple(older_turns)):
                self._remember_ledger_item(entry)
            self._raw_tail = list(recent_turns)
        self._trim_raw_tail()

    def _trim_raw_tail(self) -> None:
        if len(self._raw_tail) > self.max_turns:
            self._raw_tail = self._raw_tail[-self.max_turns :]

    def _trim_ledger(self) -> None:
        max_entries = max(self.max_turns * 3, 12)
        if len(self._ledger) > max_entries:
            self._ledger = self._ledger[-max_entries:]

    def _trim_search_results(self) -> None:
        max_entries = max(4, min(10, self.keep_recent))
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
                    created_at=self._normalize_datetime(group[-1].created_at),  # AUDIT-FIX(#2): Guarantee UTC-aware timestamps even after restore.
                    source="compactor",
                )
            )
        return tuple(summaries)

    def _remember_ledger_item(self, item: MemoryLedgerItem) -> None:
        normalized_item = self._coerce_ledger_item(item)
        if normalized_item is None:
            return  # AUDIT-FIX(#5): Ignore malformed ledger writes instead of raising mid-compaction.
        normalized = self._ledger_key(normalized_item.kind, normalized_item.content)
        merged_metadata = {
            key: value
            for key, value in dict(normalized_item.metadata).items()
            if str(value or "").strip()
        }
        for index, existing in enumerate(self._ledger):
            if self._ledger_key(existing.kind, existing.content) != normalized:
                continue
            self._ledger[index] = MemoryLedgerItem(
                kind=normalized_item.kind,
                content=normalized_item.content,
                created_at=max(existing.created_at, normalized_item.created_at),  # AUDIT-FIX(#2): Keep deduplicated timestamps comparable and monotonic.
                source=normalized_item.source,
                metadata={**dict(existing.metadata), **merged_metadata},
            )
            self._trim_ledger()
            return
        self._ledger.append(
            MemoryLedgerItem(
                kind=normalized_item.kind,
                content=normalized_item.content,
                created_at=normalized_item.created_at,
                source=normalized_item.source,
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
            summary_lines.append("Open loops: " + " | ".join(loop for loop in self._state.open_loops[:2] if loop))
        for entry in self._search_results[-self._summary_search_limit() :]:
            summary_lines.append(
                "Verified web lookup: "
                f"{self._short_text(entry.question, limit=120)} -> {self._short_text(entry.answer, limit=220)}"
            )
        ledger_lines = [
            self._format_ledger_summary_line(item)
            for item in self._ledger[-self._summary_ledger_limit() :]
            if item.kind != "search_result"
        ]
        summary_lines.extend(line for line in ledger_lines if line)
        if not summary_lines:
            return None
        return ConversationTurn(
            role=self._SUMMARY_ROLE,  # AUDIT-FIX(#1): Keep memory summaries out of the system prompt channel.
            content=self._SUMMARY_HEADER + "\n" + "\n".join(f"- {line}" for line in summary_lines),
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
            latest_open_loop = self._short_text(self._raw_tail[-1].content, limit=120)
            self._state.open_loops = (latest_open_loop,) if latest_open_loop else ()  # AUDIT-FIX(#4): Never emit blank open loops.
        else:
            self._state.open_loops = ()

    def _update_state_from_turn(self, turn: ConversationTurn) -> None:
        if turn.role == "user":
            self._state.active_topic = self._short_text(turn.content, limit=120)
            self._state.last_user_goal = self._short_text(turn.content, limit=160)
            open_loop = self._short_text(turn.content, limit=120)
            self._state.open_loops = (open_loop,) if open_loop else ()  # AUDIT-FIX(#4): Never emit blank open loops.
            return
        if turn.role == "assistant":
            self._state.pending_printable = self._short_text(turn.content, limit=220)
            self._state.open_loops = ()

    def _summary_timestamp(self) -> datetime:
        candidates: list[datetime] = []
        if self._raw_tail:
            candidates.append(self._normalize_datetime(self._raw_tail[-1].created_at))  # AUDIT-FIX(#2): Normalize before timestamp comparison.
        if self._ledger:
            candidates.append(self._normalize_datetime(self._ledger[-1].created_at))
        if self._search_results:
            candidates.append(self._normalize_datetime(self._search_results[-1].created_at))
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
        return (
            self._normalize_text(kind, limit=self._MAX_NOTE_KIND_CHARS, collapse_whitespace=True).lower(),  # AUDIT-FIX(#9): Normalize dedupe keys consistently.
            self._normalize_text(content, limit=self._MAX_SUMMARY_CONTENT_CHARS, collapse_whitespace=True).lower(),
        )

    def _summary_search_limit(self) -> int:
        return max(2, min(4, (self.keep_recent // 3) + 1))

    def _summary_ledger_limit(self) -> int:
        return max(3, min(6, (self.keep_recent // 2) + 1))

    def _validate_limits(self, *, max_turns: int, keep_recent: int) -> None:
        if not isinstance(max_turns, int) or isinstance(max_turns, bool):  # AUDIT-FIX(#5): Reject non-integer limit types early.
            raise ValueError("max_turns must be an integer")
        if not isinstance(keep_recent, int) or isinstance(keep_recent, bool):
            raise ValueError("keep_recent must be an integer")
        if max_turns < 3:
            raise ValueError("max_turns must be at least 3")
        if keep_recent < 1 or keep_recent >= max_turns:
            raise ValueError("keep_recent must be between 1 and max_turns - 1")

    def _short_text(self, text: str, *, limit: int) -> str:
        if limit <= 0:
            return ""  # AUDIT-FIX(#5): Avoid producing stray ellipses for invalid limits.
        normalized = " ".join(self._coerce_text(text).split()).strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: max(limit - 1, 0)].rstrip() + "…"

    def _clone_turn(self, turn: ConversationTurn) -> ConversationTurn:  # AUDIT-FIX(#3): Defensive copy for external callers.
        return ConversationTurn(
            role=turn.role,
            content=turn.content,
            created_at=self._normalize_datetime(turn.created_at),
        )

    def _clone_ledger_item(self, item: MemoryLedgerItem) -> MemoryLedgerItem:  # AUDIT-FIX(#3): Defensive copy for external callers.
        return MemoryLedgerItem(
            kind=item.kind,
            content=item.content,
            created_at=self._normalize_datetime(item.created_at),
            source=item.source,
            metadata=dict(item.metadata),
        )

    def _clone_search_entry(self, item: SearchMemoryEntry) -> SearchMemoryEntry:  # AUDIT-FIX(#3): Defensive copy for external callers.
        return SearchMemoryEntry(
            question=item.question,
            answer=item.answer,
            sources=tuple(item.sources),
            created_at=self._normalize_datetime(item.created_at),
            location_hint=item.location_hint,
            date_context=item.date_context,
        )

    def _clone_state(self, state: MemoryState) -> MemoryState:  # AUDIT-FIX(#3): Defensive copy for external callers.
        return MemoryState(
            active_topic=state.active_topic,
            last_user_goal=state.last_user_goal,
            pending_printable=state.pending_printable,
            last_search_summary=state.last_search_summary,
            open_loops=tuple(state.open_loops),
        )

    def _coerce_text(self, value: object) -> str:  # AUDIT-FIX(#5): Accept bytes/non-str restore payloads without crashing.
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def _normalize_text(self, value: object, *, limit: int, collapse_whitespace: bool = False) -> str:  # AUDIT-FIX(#5): Centralize safe string normalization.
        text = self._coerce_text(value).strip()
        if not text or limit <= 0:
            return ""
        if collapse_whitespace:
            text = " ".join(text.split()).strip()
        if len(text) <= limit:
            return text
        return text[: max(limit - 1, 0)].rstrip() + "…"

    def _normalize_optional_text(self, value: object, *, limit: int, collapse_whitespace: bool = False) -> str | None:  # AUDIT-FIX(#5): Preserve Optional[str] semantics after normalization.
        clean = self._normalize_text(value, limit=limit, collapse_whitespace=collapse_whitespace)
        return clean or None

    def _normalize_role(self, role: object) -> str | None:  # AUDIT-FIX(#4): Keep state logic restricted to supported roles.
        clean_role = self._normalize_text(role, limit=32, collapse_whitespace=True).lower()
        clean_role = self._ROLE_ALIASES.get(clean_role, clean_role)
        if clean_role in self._ALLOWED_TURN_ROLES:
            return clean_role
        return None

    def _normalize_datetime(self, value: object) -> datetime:  # AUDIT-FIX(#2): Repair naive/invalid timestamps during restore and copy-out.
        if not isinstance(value, datetime):
            return _utcnow()
        try:
            if value.tzinfo is None or value.utcoffset() is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        except Exception:
            return _utcnow()

    def _normalize_metadata(self, metadata: object) -> dict[str, str]:  # AUDIT-FIX(#9): Sanitize metadata shape and size.
        if metadata is None:
            return {}
        items: Iterable[tuple[object, object]]
        if isinstance(metadata, Mapping):
            items = metadata.items()
        else:
            try:
                items = dict(metadata).items()
            except (TypeError, ValueError):
                return {}
        normalized: dict[str, str] = {}
        for key, value in items:
            clean_key = self._normalize_text(key, limit=self._MAX_METADATA_KEY_CHARS, collapse_whitespace=True)
            clean_value = self._normalize_text(value, limit=self._MAX_METADATA_VALUE_CHARS, collapse_whitespace=True)
            if clean_key and clean_value:
                normalized[clean_key] = clean_value
        return normalized

    def _normalize_sources(self, sources: object) -> tuple[str, ...]:  # AUDIT-FIX(#5): Sanitize source iterables from restore/tool paths.
        if sources is None:
            return ()
        iterable: Iterable[object]
        if isinstance(sources, (str, bytes)):
            iterable = (sources,)
        else:
            try:
                iterable = tuple(sources)
            except TypeError:
                return ()
        normalized: list[str] = []
        for source in iterable:
            clean_source = self._normalize_text(source, limit=self._MAX_SOURCE_CHARS, collapse_whitespace=True)
            if clean_source:
                normalized.append(clean_source)
            if len(normalized) >= self._MAX_SOURCE_COUNT:
                break
        return tuple(normalized)

    def _extract_summary_content(self, turn: object) -> str | None:  # AUDIT-FIX(#1): Only trust explicitly marked memory-summary turns.
        role = self._normalize_text(getattr(turn, "role", None), limit=32, collapse_whitespace=True).lower()
        if role not in self._SUMMARY_RESTORE_ROLES:
            return None
        content = self._normalize_text(getattr(turn, "content", None), limit=self._MAX_SUMMARY_CONTENT_CHARS)
        if not content.startswith(self._SUMMARY_HEADER):
            return None
        body = content[len(self._SUMMARY_HEADER) :].strip()
        if not body:
            return None
        lines = []
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                stripped = stripped[2:].strip()
            if stripped:
                lines.append(stripped)
        flattened = " | ".join(lines) if lines else body
        return self._short_text(flattened, limit=self._MAX_SUMMARY_CONTENT_CHARS)

    def _coerce_conversation_turn(self, turn: object) -> ConversationTurn | None:  # AUDIT-FIX(#5): Make restore resilient to malformed turn payloads.
        role = self._normalize_role(getattr(turn, "role", None))
        content = self._normalize_text(getattr(turn, "content", None), limit=self._MAX_TURN_CONTENT_CHARS)
        if role is None or not content:
            return None
        return ConversationTurn(
            role=role,
            content=content,
            created_at=self._normalize_datetime(getattr(turn, "created_at", None)),
        )

    def _coerce_ledger_item(self, item: object) -> MemoryLedgerItem | None:  # AUDIT-FIX(#5): Make restore resilient to malformed ledger payloads.
        kind = self._normalize_text(getattr(item, "kind", None), limit=self._MAX_NOTE_KIND_CHARS, collapse_whitespace=True).lower() or "memory"
        content = self._normalize_text(getattr(item, "content", None), limit=self._MAX_LEDGER_CONTENT_CHARS, collapse_whitespace=True)
        if not content:
            return None
        source = self._normalize_text(getattr(item, "source", None), limit=self._MAX_NOTE_KIND_CHARS, collapse_whitespace=True) or "conversation"
        metadata = self._normalize_metadata(getattr(item, "metadata", None))
        return MemoryLedgerItem(
            kind=kind,
            content=content,
            created_at=self._normalize_datetime(getattr(item, "created_at", None)),
            source=source,
            metadata=metadata,
        )

    def _coerce_search_entry(self, item: object) -> SearchMemoryEntry | None:  # AUDIT-FIX(#5): Make restore resilient to malformed search payloads.
        question = self._normalize_text(getattr(item, "question", None), limit=self._MAX_SEARCH_QUESTION_CHARS)
        answer = self._normalize_text(getattr(item, "answer", None), limit=self._MAX_SEARCH_ANSWER_CHARS)
        if not question or not answer:
            return None
        return SearchMemoryEntry(
            question=question,
            answer=answer,
            sources=self._normalize_sources(getattr(item, "sources", None)),
            created_at=self._normalize_datetime(getattr(item, "created_at", None)),
            location_hint=self._normalize_optional_text(getattr(item, "location_hint", None), limit=120),
            date_context=self._normalize_optional_text(getattr(item, "date_context", None), limit=120),
        )

    def _coerce_state(self, state: MemoryState | None) -> MemoryState:  # AUDIT-FIX(#5): Sanitize restored state objects instead of trusting them blindly.
        if state is None:
            return MemoryState()
        open_loops = self._normalize_open_loops(getattr(state, "open_loops", ()))
        return MemoryState(
            active_topic=self._normalize_optional_text(getattr(state, "active_topic", None), limit=120, collapse_whitespace=True),
            last_user_goal=self._normalize_optional_text(getattr(state, "last_user_goal", None), limit=160, collapse_whitespace=True),
            pending_printable=self._normalize_optional_text(getattr(state, "pending_printable", None), limit=220),
            last_search_summary=self._normalize_optional_text(getattr(state, "last_search_summary", None), limit=220),
            open_loops=open_loops,
        )

    def _normalize_open_loops(self, open_loops: object) -> tuple[str, ...]:  # AUDIT-FIX(#5): Handle malformed state.open_loops values safely.
        if open_loops is None:
            return ()
        iterable: Iterable[object]
        if isinstance(open_loops, (str, bytes)):
            iterable = (open_loops,)
        else:
            try:
                iterable = tuple(open_loops)
            except TypeError:
                return ()
        normalized: list[str] = []
        for raw_loop in iterable:
            clean_loop = self._normalize_text(raw_loop, limit=120, collapse_whitespace=True)
            if clean_loop:
                normalized.append(clean_loop)
            if len(normalized) >= 4:
                break
        return tuple(normalized)
