from __future__ import annotations

import json


def _format_log_rows(entries: list[dict[str, object]]) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for entry in reversed(entries):
        data = entry.get("data")
        rows.append(
            {
                "created_at": entry.get("created_at", "—"),
                "level": entry.get("level", "info"),
                "event": entry.get("event", "unknown"),
                "message": entry.get("message", ""),
                "data_pretty": (
                    json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)
                    if isinstance(data, dict) and data
                    else ""
                ),
            }
        )
    return tuple(rows)


def _format_usage_rows(records) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for record in reversed(records):
        token_usage = record.token_usage.to_dict() if record.token_usage is not None else {}
        rows.append(
            {
                "created_at": record.created_at,
                "source": record.source,
                "request_kind": record.request_kind,
                "model": record.model or "unknown",
                "response_id": record.response_id or "—",
                "request_id": record.request_id or "—",
                "used_web_search": (
                    "yes"
                    if record.used_web_search is True
                    else ("no" if record.used_web_search is False else "—")
                ),
                "total_tokens": record.total_tokens if record.total_tokens is not None else "—",
                "input_tokens": token_usage.get("input_tokens", "—"),
                "output_tokens": token_usage.get("output_tokens", "—"),
                "cached_input_tokens": token_usage.get("cached_input_tokens", "—"),
                "reasoning_tokens": token_usage.get("reasoning_tokens", "—"),
                "metadata_pretty": (
                    json.dumps(record.metadata, indent=2, ensure_ascii=False, sort_keys=True)
                    if record.metadata
                    else ""
                ),
            }
        )
    return tuple(rows)


def _health_card_detail(health) -> str:
    parts: list[str] = []
    if health.cpu_temperature_c is not None:
        parts.append(f"{health.cpu_temperature_c:.1f}C")
    if health.memory_used_percent is not None:
        parts.append(f"mem {health.memory_used_percent:.0f}%")
    if health.disk_used_percent is not None:
        parts.append(f"disk {health.disk_used_percent:.0f}%")
    return " · ".join(parts) if parts else "Live Pi snapshot"
