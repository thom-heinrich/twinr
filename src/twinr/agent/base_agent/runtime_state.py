from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json

from twinr.memory import ConversationTurn


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class RuntimeSnapshotTurn:
    role: str
    content: str
    created_at: str


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    status: str = "waiting"
    last_transcript: str | None = None
    last_response: str | None = None
    updated_at: str | None = None
    error_message: str | None = None
    memory_turns: tuple[RuntimeSnapshotTurn, ...] = ()

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
        )

    def save(
        self,
        *,
        status: str,
        memory_turns: tuple[ConversationTurn, ...],
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
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.path)
        return snapshot
