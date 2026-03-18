"""Shape operator memory-search results for the Twinr debug portal."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from twinr.memory.longterm.retrieval.operator_search import LongTermOperatorSearchResult


def build_memory_search_panel_context(
    *,
    query_text: str,
    result: LongTermOperatorSearchResult | None,
    error_message: str | None,
) -> dict[str, object]:
    """Return template-ready context for the debug portal memory-search tab."""

    normalized_query = " ".join(str(query_text or "").split()).strip()
    status = _memory_search_status(
        query_text=normalized_query,
        result=result,
        error_message=error_message,
    )
    return {
        "query": normalized_query,
        "status": status,
        "summary_rows": _memory_search_summary_rows(result),
        "sections": _memory_search_sections(result),
        "context_blocks": _memory_search_context_blocks(result),
    }


def _memory_search_status(
    *,
    query_text: str,
    result: LongTermOperatorSearchResult | None,
    error_message: str | None,
) -> dict[str, str]:
    if error_message:
        return {"label": "Search failed", "detail": error_message, "status": "fail"}
    if not query_text:
        return {
            "label": "Ready",
            "detail": "Searches the same long-term retrieval path Twinr uses for live recall.",
            "status": "muted",
        }
    if result is None:
        return {"label": "No result", "detail": "Twinr did not produce a search result.", "status": "warn"}
    if result.total_hits == 0:
        return {
            "label": "No hits",
            "detail": "No durable, episodic, midterm, or conflict hits matched this query.",
            "status": "warn",
        }
    return {
        "label": f"{result.total_hits} hits",
        "detail": (
            f"durable {len(result.durable_objects)} · episodic {len(result.episodic_entries)} · "
            f"midterm {len(result.midterm_packets)} · conflicts {len(result.conflict_queue)}"
        ),
        "status": "ok",
    }


def _memory_search_summary_rows(
    result: LongTermOperatorSearchResult | None,
) -> tuple[dict[str, object], ...]:
    if result is None or not result.query_profile.retrieval_text:
        return ()
    rows = [
        _detail_item("Query", result.query_profile.original_text or "—", copy=True, wide=True),
        _detail_item("Retrieval query", result.query_profile.retrieval_text or "—", copy=True, wide=True),
        _detail_item(
            "Canonical English",
            result.query_profile.canonical_english_text or "—",
            copy=True,
            wide=True,
        ),
        _detail_item("Durable hits", str(len(result.durable_objects))),
        _detail_item("Episodic hits", str(len(result.episodic_entries))),
        _detail_item("Midterm hits", str(len(result.midterm_packets))),
        _detail_item("Conflict hits", str(len(result.conflict_queue))),
    ]
    return tuple(rows)


def _memory_search_sections(
    result: LongTermOperatorSearchResult | None,
) -> tuple[dict[str, object], ...]:
    if result is None:
        return ()
    return (
        {
            "title": "Durable memory",
            "description": "Stable facts and structured long-term objects that matched the query.",
            "empty_message": "No durable facts matched this query.",
            "items": tuple(_durable_item(item) for item in result.durable_objects),
        },
        {
            "title": "Midterm memory",
            "description": "Near-term continuity packets that would shape recent recall.",
            "empty_message": "No midterm packets matched this query.",
            "items": tuple(_midterm_item(item) for item in result.midterm_packets),
        },
        {
            "title": "Episodic memory",
            "description": "Conversation excerpts and episodic traces relevant to the query.",
            "empty_message": "No episodic memories matched this query.",
            "items": tuple(_episodic_item(item) for item in result.episodic_entries),
        },
        {
            "title": "Open conflicts",
            "description": "Unresolved memory conflicts that still need clarification.",
            "empty_message": "No open conflicts matched this query.",
            "items": tuple(_conflict_item(item) for item in result.conflict_queue),
        },
    )


def _memory_search_context_blocks(
    result: LongTermOperatorSearchResult | None,
) -> tuple[dict[str, str], ...]:
    if result is None or not result.graph_context:
        return ()
    return (
        {
            "title": "Graph context preview",
            "body": result.graph_context,
        },
    )


def _durable_item(item: Any) -> dict[str, object]:
    meta = [
        f"memory_id={getattr(item, 'memory_id', '—')}",
        f"kind={getattr(item, 'kind', '—')}",
        f"status={getattr(item, 'status', '—')}",
    ]
    slot_key = getattr(item, "slot_key", None)
    if slot_key:
        meta.append(f"slot={slot_key}")
    value_key = getattr(item, "value_key", None)
    if value_key:
        meta.append(f"value={value_key}")
    return {
        "badge": getattr(item, "kind", "object"),
        "level": _status_class(getattr(item, "status", None)),
        "title": getattr(item, "summary", "Stored durable object"),
        "time": _format_datetime(getattr(item, "updated_at", None)),
        "body": getattr(item, "details", None) or "No extra details stored.",
        "meta_lines": tuple(meta),
    }


def _midterm_item(item: Any) -> dict[str, object]:
    meta = [
        f"packet_id={getattr(item, 'packet_id', '—')}",
        f"kind={getattr(item, 'kind', '—')}",
    ]
    query_hints = tuple(getattr(item, "query_hints", ()) or ())
    if query_hints:
        meta.append("query_hints=" + " | ".join(str(hint) for hint in query_hints[:4]))
    return {
        "badge": getattr(item, "kind", "midterm"),
        "level": "muted",
        "title": getattr(item, "summary", "Stored midterm packet"),
        "time": _format_datetime(getattr(item, "updated_at", None)),
        "body": getattr(item, "details", None) or "No extra details stored.",
        "meta_lines": tuple(meta),
    }


def _episodic_item(item: Any) -> dict[str, object]:
    return {
        "badge": getattr(item, "kind", "episode"),
        "level": "muted",
        "title": getattr(item, "summary", "Stored episode"),
        "time": _format_datetime(getattr(item, "updated_at", None) or getattr(item, "created_at", None)),
        "body": getattr(item, "details", None) or "No extra details stored.",
        "meta_lines": (f"entry_id={getattr(item, 'entry_id', '—')}",),
    }


def _conflict_item(item: Any) -> dict[str, object]:
    option_summaries = []
    for option in tuple(getattr(item, "options", ()) or ())[:4]:
        option_summaries.append(f"{getattr(option, 'summary', 'option')} ({getattr(option, 'memory_id', '—')})")
    meta = [
        f"slot={getattr(item, 'slot_key', '—')}",
        f"candidate={getattr(item, 'candidate_memory_id', '—')}",
    ]
    if option_summaries:
        meta.append("options=" + " | ".join(option_summaries))
    return {
        "badge": "conflict",
        "level": "warn",
        "title": getattr(item, "question", "Open memory conflict"),
        "time": None,
        "body": getattr(item, "reason", None) or "No conflict reason stored.",
        "meta_lines": tuple(meta),
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


def _status_class(value: object | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"active", "ok", "ready", "confirmed"}:
        return "ok"
    if normalized in {"warn", "uncertain", "candidate"}:
        return "warn"
    if normalized in {"invalid", "fail", "error", "expired", "superseded"}:
        return "fail"
    return "muted"


def _format_datetime(value: object | None) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value or "").strip()
    return text or None
