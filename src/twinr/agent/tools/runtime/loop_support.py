"""Share small runtime-loop helpers across Twinr tool orchestration modules."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Sequence
from uuid import uuid4

if TYPE_CHECKING:
    from twinr.agent.base_agent.contracts import AgentToolCall, AgentToolResult


logger = logging.getLogger(__name__)


def raise_if_should_stop(
    should_stop: Callable[[], bool] | None,
    *,
    context: str,
    actor: str = "tool loop",
) -> None:
    """Abort cooperative runtime work once the owning turn was stopped."""

    if should_stop is None:
        return
    if should_stop():
        raise InterruptedError(f"{actor} stopped during {context}")


def coerce_text(value: Any) -> str:
    """Coerce arbitrary callback payloads into text."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def strip_text(value: Any) -> str:
    """Coerce text and strip surrounding whitespace."""

    return coerce_text(value).strip()


def safe_json_dumps(value: Any) -> str:
    """Serialize arbitrary diagnostic payloads into JSON for tool envelopes."""

    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        logger.exception("Failed to serialize tool output to JSON.")
        return json.dumps(
            {"status": "serialization_error", "type": type(value).__name__},
            ensure_ascii=False,
        )


def first_non_none(*values: Any) -> Any:
    """Return the first non-None value from the provided candidates."""

    for value in values:
        if value is not None:
            return value
    return None


def make_call_id(prefix: str) -> str:
    """Create a deterministic-looking unique tool call identifier."""

    safe_prefix = strip_text(prefix) or "tool"
    return f"{safe_prefix}_{uuid4().hex}"


def merge_token_usage(base: Any, extra: Any) -> Any:
    """Merge provider token-usage payloads across supervisor and specialist work."""

    if base is None:
        return extra
    if extra is None:
        return base
    if isinstance(base, (int, float)) and isinstance(extra, (int, float)):
        return base + extra
    if isinstance(base, dict) and isinstance(extra, dict):
        merged = dict(base)
        for key, value in extra.items():
            if key in merged:
                merged[key] = merge_token_usage(merged[key], value)
            else:
                merged[key] = value
        return merged
    return base


def make_loop_result(
    *,
    text: str,
    rounds: int,
    tool_calls: Sequence[AgentToolCall],
    tool_results: Sequence[AgentToolResult],
    response_id: str | None,
    request_id: str | None,
    model: str | None,
    token_usage: Any,
    used_web_search: bool,
):
    """Build an immutable loop result from the collected turn state."""

    from .streaming_loop import StreamingToolLoopResult

    return StreamingToolLoopResult(
        text=text,
        rounds=rounds,
        tool_calls=tuple(tool_calls),
        tool_results=tuple(tool_results),
        response_id=response_id,
        request_id=request_id,
        model=model,
        token_usage=token_usage,
        used_web_search=used_web_search,
    )
