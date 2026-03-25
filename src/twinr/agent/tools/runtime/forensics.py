"""Build redacted dual-lane trace summaries without storing raw user text."""

from __future__ import annotations

from hashlib import sha1
from typing import Any

from twinr.agent.base_agent.contracts import ConversationLike

from .handoff import normalize_decision_action, normalize_handoff_kind
from .loop_support import coerce_text, strip_text


def text_summary(value: Any) -> dict[str, Any]:
    """Describe text safely for forensics without storing raw content."""

    normalized = strip_text(value)
    if not normalized:
        return {"present": False, "chars": 0, "words": 0, "sha12": None}
    return {
        "present": True,
        "chars": len(normalized),
        "words": len(normalized.split()),
        "sha12": sha1(normalized.encode("utf-8")).hexdigest()[:12],
    }


def conversation_summary(conversation: ConversationLike | None) -> dict[str, Any]:
    """Summarize conversation context shape without leaking raw text."""

    if not conversation:
        return {"present": False, "messages": 0, "tail": []}
    tail: list[dict[str, Any]] = []
    total_chars = 0
    for item in conversation:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            role = strip_text(item[0]) or "unknown"
            content = coerce_text(item[1])
        else:
            role = "unknown"
            content = coerce_text(item)
        total_chars += len(content.strip())
        tail.append({"role": role, "content": text_summary(content)})
    return {
        "present": True,
        "messages": len(tail),
        "total_chars": total_chars,
        "tail": tail[-3:],
    }


def decision_summary(decision: Any | None) -> dict[str, Any]:
    """Summarize a supervisor decision for branch forensics."""

    if decision is None:
        return {"present": False}
    return {
        "present": True,
        "action": normalize_decision_action(getattr(decision, "action", None)),
        "kind": normalize_handoff_kind(getattr(decision, "kind", None)),
        "context_scope": strip_text(getattr(decision, "context_scope", None)) or None,
        "allow_web_search": getattr(decision, "allow_web_search", None),
        "spoken_ack": text_summary(getattr(decision, "spoken_ack", None)),
        "spoken_reply": text_summary(getattr(decision, "spoken_reply", None)),
        "goal": text_summary(getattr(decision, "goal", None)),
        "prompt": text_summary(getattr(decision, "prompt", None)),
        "location_hint": text_summary(getattr(decision, "location_hint", None)),
        "date_context": text_summary(getattr(decision, "date_context", None)),
    }


def handoff_summary(arguments: dict[str, Any]) -> dict[str, Any]:
    """Summarize a normalized handoff payload without raw user text."""

    return {
        "kind": normalize_handoff_kind(arguments.get("kind")),
        "goal": text_summary(arguments.get("goal")),
        "spoken_ack": text_summary(arguments.get("spoken_ack")),
        "prompt": text_summary(arguments.get("prompt")),
        "allow_web_search": arguments.get("allow_web_search"),
        "location_hint": text_summary(arguments.get("location_hint")),
        "date_context": text_summary(arguments.get("date_context")),
    }


def loop_result_summary(result: Any | None) -> dict[str, Any]:
    """Summarize a loop result for branch-level forensic trace output."""

    if result is None:
        return {"present": False}
    return {
        "present": True,
        "text": text_summary(getattr(result, "text", None)),
        "rounds": getattr(result, "rounds", 0),
        "tool_calls": len(getattr(result, "tool_calls", ())),
        "tool_results": len(getattr(result, "tool_results", ())),
        "used_web_search": bool(getattr(result, "used_web_search", False)),
        "response_id_present": bool(strip_text(getattr(result, "response_id", None))),
        "request_id_present": bool(strip_text(getattr(result, "request_id", None))),
        "model_present": bool(strip_text(getattr(result, "model", None))),
    }
