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


class OnDeviceMemory:
    def __init__(self, max_turns: int = 12, keep_recent: int = 6) -> None:
        if max_turns < 3:
            raise ValueError("max_turns must be at least 3")
        if keep_recent < 1 or keep_recent >= max_turns:
            raise ValueError("keep_recent must be between 1 and max_turns - 1")
        self.max_turns = max_turns
        self.keep_recent = keep_recent
        self._turns: list[ConversationTurn] = []

    @property
    def turns(self) -> tuple[ConversationTurn, ...]:
        return tuple(self._turns)

    def remember(self, role: str, content: str) -> ConversationTurn:
        turn = ConversationTurn(role=role, content=content.strip())
        self._turns.append(turn)
        if len(self._turns) > self.max_turns:
            self._compact()
        return turn

    def last_assistant_message(self) -> str | None:
        for turn in reversed(self._turns):
            if turn.role == "assistant":
                return turn.content
        return None

    def _compact(self) -> None:
        older_turns = self._turns[:-self.keep_recent]
        recent_turns = self._turns[-self.keep_recent:]
        summary_lines = [f"{turn.role}: {turn.content}" for turn in older_turns]
        summary_text = "Compact conversation summary:\n" + "\n".join(summary_lines)
        self._turns = [ConversationTurn(role="system", content=summary_text), *recent_turns]
