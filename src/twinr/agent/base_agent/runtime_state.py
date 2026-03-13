from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import tempfile

from twinr.memory import ConversationTurn, MemoryLedgerItem, MemoryState, SearchMemoryEntry


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class RuntimeSnapshotTurn:
    role: str
    content: str
    created_at: str


@dataclass(frozen=True, slots=True)
class RuntimeSnapshotLedgerItem:
    kind: str
    content: str
    created_at: str
    source: str
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class RuntimeSnapshotSearchEntry:
    question: str
    answer: str
    sources: tuple[str, ...]
    created_at: str
    location_hint: str | None = None
    date_context: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeSnapshotMemoryState:
    active_topic: str | None = None
    last_user_goal: str | None = None
    pending_printable: str | None = None
    last_search_summary: str | None = None
    open_loops: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    status: str = "waiting"
    last_transcript: str | None = None
    last_response: str | None = None
    updated_at: str | None = None
    error_message: str | None = None
    memory_turns: tuple[RuntimeSnapshotTurn, ...] = ()
    memory_raw_tail: tuple[RuntimeSnapshotTurn, ...] = ()
    memory_ledger: tuple[RuntimeSnapshotLedgerItem, ...] = ()
    memory_search_results: tuple[RuntimeSnapshotSearchEntry, ...] = ()
    memory_state: RuntimeSnapshotMemoryState = RuntimeSnapshotMemoryState()

    @property
    def memory_count(self) -> int:
        return len(self.memory_turns)


class RuntimeSnapshotStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> RuntimeSnapshot:
        if not self.path.exists():
            return RuntimeSnapshot()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return RuntimeSnapshot()

        return RuntimeSnapshot(
            status=str(data.get("status", "waiting")),
            last_transcript=data.get("last_transcript"),
            last_response=data.get("last_response"),
            updated_at=data.get("updated_at"),
            error_message=data.get("error_message"),
            memory_turns=tuple(
                RuntimeSnapshotTurn(
                    role=str(item.get("role", "")),
                    content=str(item.get("content", "")),
                    created_at=str(item.get("created_at", "")),
                )
                for item in data.get("memory_turns", [])
            ),
            memory_raw_tail=tuple(
                RuntimeSnapshotTurn(
                    role=str(item.get("role", "")),
                    content=str(item.get("content", "")),
                    created_at=str(item.get("created_at", "")),
                )
                for item in data.get("memory_raw_tail", [])
            ),
            memory_ledger=tuple(
                RuntimeSnapshotLedgerItem(
                    kind=str(item.get("kind", "")),
                    content=str(item.get("content", "")),
                    created_at=str(item.get("created_at", "")),
                    source=str(item.get("source", "conversation")),
                    metadata={
                        str(key): str(value)
                        for key, value in dict(item.get("metadata", {})).items()
                        if str(value or "").strip()
                    },
                )
                for item in data.get("memory_ledger", [])
            ),
            memory_search_results=tuple(
                RuntimeSnapshotSearchEntry(
                    question=str(item.get("question", "")),
                    answer=str(item.get("answer", "")),
                    sources=tuple(
                        str(source)
                        for source in item.get("sources", [])
                        if str(source or "").strip()
                    ),
                    created_at=str(item.get("created_at", "")),
                    location_hint=(
                        str(item.get("location_hint")).strip()
                        if item.get("location_hint") is not None
                        else None
                    ),
                    date_context=(
                        str(item.get("date_context")).strip()
                        if item.get("date_context") is not None
                        else None
                    ),
                )
                for item in data.get("memory_search_results", [])
            ),
            memory_state=RuntimeSnapshotMemoryState(
                active_topic=_optional_str(data.get("memory_state", {}), "active_topic"),
                last_user_goal=_optional_str(data.get("memory_state", {}), "last_user_goal"),
                pending_printable=_optional_str(data.get("memory_state", {}), "pending_printable"),
                last_search_summary=_optional_str(data.get("memory_state", {}), "last_search_summary"),
                open_loops=tuple(
                    str(item)
                    for item in (data.get("memory_state", {}) or {}).get("open_loops", [])
                    if str(item or "").strip()
                ),
            ),
        )

    def save(
        self,
        *,
        status: str,
        memory_turns: tuple[ConversationTurn, ...],
        memory_raw_tail: tuple[ConversationTurn, ...] | None = None,
        memory_ledger: tuple[MemoryLedgerItem, ...] | None = None,
        memory_search_results: tuple[SearchMemoryEntry, ...] | None = None,
        memory_state: MemoryState | None = None,
        last_transcript: str | None,
        last_response: str | None,
        error_message: str | None = None,
    ) -> RuntimeSnapshot:
        snapshot = RuntimeSnapshot(
            status=status,
            last_transcript=last_transcript,
            last_response=last_response,
            updated_at=_utcnow().isoformat(),
            error_message=error_message,
            memory_turns=tuple(
                RuntimeSnapshotTurn(
                    role=turn.role,
                    content=turn.content,
                    created_at=turn.created_at.isoformat(),
                )
                for turn in memory_turns
            ),
            memory_raw_tail=tuple(
                RuntimeSnapshotTurn(
                    role=turn.role,
                    content=turn.content,
                    created_at=turn.created_at.isoformat(),
                )
                for turn in (memory_raw_tail or ())
            ),
            memory_ledger=tuple(
                RuntimeSnapshotLedgerItem(
                    kind=item.kind,
                    content=item.content,
                    created_at=item.created_at.isoformat(),
                    source=item.source,
                    metadata={str(key): str(value) for key, value in item.metadata.items() if str(value or "").strip()},
                )
                for item in (memory_ledger or ())
            ),
            memory_search_results=tuple(
                RuntimeSnapshotSearchEntry(
                    question=item.question,
                    answer=item.answer,
                    sources=tuple(item.sources),
                    created_at=item.created_at.isoformat(),
                    location_hint=item.location_hint,
                    date_context=item.date_context,
                )
                for item in (memory_search_results or ())
            ),
            memory_state=RuntimeSnapshotMemoryState(
                active_topic=memory_state.active_topic if memory_state is not None else None,
                last_user_goal=memory_state.last_user_goal if memory_state is not None else None,
                pending_printable=memory_state.pending_printable if memory_state is not None else None,
                last_search_summary=memory_state.last_search_summary if memory_state is not None else None,
                open_loops=tuple(memory_state.open_loops) if memory_state is not None else (),
            ),
        )
        payload = {
            "status": snapshot.status,
            "last_transcript": snapshot.last_transcript,
            "last_response": snapshot.last_response,
            "updated_at": snapshot.updated_at,
            "error_message": snapshot.error_message,
            "memory_turns": [
                {
                    "role": turn.role,
                    "content": turn.content,
                    "created_at": turn.created_at,
                }
                for turn in snapshot.memory_turns
            ],
            "memory_raw_tail": [
                {
                    "role": turn.role,
                    "content": turn.content,
                    "created_at": turn.created_at,
                }
                for turn in snapshot.memory_raw_tail
            ],
            "memory_ledger": [
                {
                    "kind": item.kind,
                    "content": item.content,
                    "created_at": item.created_at,
                    "source": item.source,
                    "metadata": item.metadata,
                }
                for item in snapshot.memory_ledger
            ],
            "memory_search_results": [
                {
                    "question": item.question,
                    "answer": item.answer,
                    "sources": list(item.sources),
                    "created_at": item.created_at,
                    "location_hint": item.location_hint,
                    "date_context": item.date_context,
                }
                for item in snapshot.memory_search_results
            ],
            "memory_state": {
                "active_topic": snapshot.memory_state.active_topic,
                "last_user_goal": snapshot.memory_state.last_user_goal,
                "pending_printable": snapshot.memory_state.pending_printable,
                "last_search_summary": snapshot.memory_state.last_search_summary,
                "open_loops": list(snapshot.memory_state.open_loops),
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
        temp_file_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(payload_text)
                temp_file_path = Path(handle.name)
            temp_file_path.replace(self.path)
        finally:
            if temp_file_path is not None and temp_file_path.exists():
                temp_file_path.unlink(missing_ok=True)
        return snapshot


def _optional_str(data: dict[str, object] | None, key: str) -> str | None:
    if not isinstance(data, dict):
        return None
    value = data.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None
