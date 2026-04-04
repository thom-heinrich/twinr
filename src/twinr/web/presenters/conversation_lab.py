"""Shape portal conversation-lab sessions for the debug page."""

from __future__ import annotations

def build_conversation_lab_panel_context(
    *,
    state: dict[str, object] | None,
) -> dict[str, object]:
    """Return template-ready context for the interactive conversation-lab tab."""

    normalized_state = dict(state or {})
    sessions = _mapping_tuple(normalized_state.get("sessions"))
    active_session = normalized_state.get("active_session")
    missing_session = bool(normalized_state.get("missing_session", False))

    if missing_session:
        status = {
            "label": "Session missing",
            "detail": "Twinr could not find that stored portal conversation. The newest session is shown instead.",
            "status": "warn",
        }
    elif active_session is None:
        status = {
            "label": "Ready",
            "detail": "Start a text turn against the real Twinr tool and memory path. This lab can mutate real reminders, settings, automations, and long-term memory.",
            "status": "muted",
        }
    else:
        active_turns = _mapping_tuple(active_session.get("turns")) if isinstance(active_session, dict) else ()
        latest_turn = active_turns[-1] if active_turns else {}
        latest_status = str(latest_turn.get("status", "ok") if isinstance(latest_turn, dict) else "ok")
        status = {
            "label": "Live session",
            "detail": f"{len(active_turns)} stored turn(s) in this portal conversation.",
            "status": "ok" if latest_status == "ok" else "warn",
        }

    active_session_id = ""
    active_session_rows: tuple[dict[str, object], ...] = ()
    turns: tuple[dict[str, object], ...] = ()
    if isinstance(active_session, dict):
        active_session_id = str(active_session.get("session_id", "") or "")
        turns = _mapping_tuple(active_session.get("turns"))
        active_session_rows = (
            _detail_item("Title", active_session.get("title") or "Portal Conversation", wide=True),
            _detail_item("Created", active_session.get("created_at") or "—", copy=True),
            _detail_item("Updated", active_session.get("updated_at") or "—", copy=True),
            _detail_item("Turns", len(turns)),
        )

    session_rows = tuple(
        {
            "session_id": str(item.get("session_id", "") or ""),
            "title": str(item.get("title", "") or "Portal Conversation"),
            "updated_at": str(item.get("updated_at", "") or ""),
            "turn_count": _coerce_non_negative_int(item.get("turn_count")),
            "href": f"/ops/debug?tab=conversation_lab&lab_session={item.get('session_id', '')}",
            "active": str(item.get("session_id", "") or "") == active_session_id,
            "status": str(item.get("status", "muted") or "muted"),
        }
        for item in sessions
        if isinstance(item, dict)
    )

    return {
        "status": status,
        "sessions": session_rows,
        "active_session": active_session,
        "active_session_rows": active_session_rows,
        "active_session_id": active_session_id,
        "turns": turns,
        "new_action": "/ops/debug/conversation-lab/new",
        "send_action": "/ops/debug/conversation-lab/send",
    }


def _detail_item(
    label: str,
    value: object,
    *,
    detail: str | None = None,
    status: str = "muted",
    copy: bool = False,
    wide: bool = False,
) -> dict[str, object]:
    return {
        "label": label,
        "value": str(value),
        "detail": detail,
        "status": status,
        "copy": copy,
        "wide": wide,
    }


def _mapping_tuple(value: object) -> tuple[dict[str, object], ...]:
    """Return only mapping-shaped items from one optional sequence payload."""

    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def _coerce_non_negative_int(value: object) -> int:
    """Return one non-negative integer-like display count."""

    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, (str, bytes, bytearray)):
        try:
            return max(0, int(value))
        except (TypeError, ValueError, OverflowError):
            return 0
    return 0
