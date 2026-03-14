from __future__ import annotations

from typing import Any


def handle_schedule_reminder(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    due_at = str(arguments.get("due_at", "")).strip()
    summary = str(arguments.get("summary", "")).strip()
    details = str(arguments.get("details", "")).strip()
    kind = str(arguments.get("kind", "")).strip() or "reminder"
    original_request = str(arguments.get("original_request", "")).strip()
    if not due_at or not summary:
        raise RuntimeError("schedule_reminder requires `due_at` and `summary`")

    entry = owner.runtime.schedule_reminder(
        due_at=due_at,
        summary=summary,
        details=details or None,
        kind=kind,
        source="schedule_reminder",
        original_request=original_request or None,
    )
    owner.emit("reminder_tool_call=true")
    owner.emit(f"reminder_scheduled={entry.summary}")
    owner.emit(f"reminder_due_at={entry.due_at.isoformat()}")
    owner._record_event(
        "reminder_tool_scheduled",
        "Realtime tool scheduled a reminder or timer.",
        reminder_id=entry.reminder_id,
        kind=entry.kind,
        due_at=entry.due_at.isoformat(),
    )
    return {
        "status": "scheduled",
        "reminder_id": entry.reminder_id,
        "kind": entry.kind,
        "summary": entry.summary,
        "due_at": entry.due_at.isoformat(),
    }
