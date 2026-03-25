"""Provide shared decision helpers for conversation evaluators.

This module owns the bounded coercion, JSON extraction, conversation
compaction, timeout detection, and transcript normalization helpers shared by
the turn-boundary and closure evaluators. Keep the helpers provider-agnostic,
deterministic, and free of workflow orchestration so they remain safe to reuse
across runtime-facing conversation modules.
"""

from __future__ import annotations

from collections.abc import Callable
import inspect
import json
import logging
import math
from typing import Any, cast

from twinr.agent.base_agent.contracts import ConversationLike

_LOGGER = logging.getLogger(__name__)


def safe_getattr(obj: object, name: str, default: object = None) -> object:
    """Return one attribute or a fallback without propagating access errors.

    Args:
        obj: Object to inspect.
        name: Attribute name to read from ``obj``.
        default: Fallback value returned when the attribute is missing or
            raises during access.

    Returns:
        The resolved attribute value or ``default`` when access fails.
    """

    try:
        return getattr(obj, name)
    except Exception:
        return default


def coerce_text(
    value: object,
    *,
    default: str = "",
    max_chars: int | None = None,
) -> str:
    """Normalize arbitrary runtime values into bounded stripped text.

    Args:
        value: Raw value to normalize.
        default: Fallback text used when ``value`` is missing or not
            stringifiable.
        max_chars: Optional hard upper bound for the returned text length.

    Returns:
        A stripped string truncated to ``max_chars`` when configured.
    """

    if value is None:
        text = default
    else:
        try:
            text = str(value)
        except Exception:
            text = default
    text = text.strip()
    if max_chars is not None and max_chars >= 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def coerce_bool(value: object, *, default: bool) -> bool:
    """Parse one runtime value into a conservative boolean.

    Args:
        value: Raw boolean-like value from config or provider payloads.
        default: Fallback boolean used when parsing fails.

    Returns:
        A normalized boolean value.
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    normalized = coerce_text(value).lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def coerce_int(
    value: object,
    *,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Parse one integer-like value and clamp it into a safe range.

    Args:
        value: Raw integer-like value.
        default: Fallback integer used when parsing fails.
        minimum: Optional lower bound for the returned number.
        maximum: Optional upper bound for the returned number.

    Returns:
        A bounded integer value.
    """

    try:
        number = int(cast(Any, value))
    except (TypeError, ValueError):
        number = default
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def coerce_float(
    value: object,
    *,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Parse one float-like value and reject non-finite numbers.

    Args:
        value: Raw float-like value.
        default: Fallback float used when parsing fails.
        minimum: Optional lower bound for the returned number.
        maximum: Optional upper bound for the returned number.

    Returns:
        A bounded finite float value.
    """

    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        number = default
    if not math.isfinite(number):
        number = default
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def coerce_probability(value: object, *, default: float) -> float:
    """Clamp one confidence-like value into the ``0.0`` to ``1.0`` range."""

    return coerce_float(value, default=default, minimum=0.0, maximum=1.0)


def config_bool(config: object, name: str, default: bool) -> bool:
    """Read one boolean config attribute with tolerant coercion."""

    return coerce_bool(safe_getattr(config, name, default), default=default)


def config_int(
    config: object,
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Read one integer config attribute with bounds applied."""

    return coerce_int(
        safe_getattr(config, name, default),
        default=default,
        minimum=minimum,
        maximum=maximum,
    )


def config_float(
    config: object,
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Read one float config attribute with bounds applied."""

    return coerce_float(
        safe_getattr(config, name, default),
        default=default,
        minimum=minimum,
        maximum=maximum,
    )


def sanitize_emit_value(value: object, *, max_chars: int) -> str:
    """Strip control characters from one emitted controller status value."""

    sanitized = coerce_text(value, max_chars=max_chars).replace("\r", " ").replace("\n", " ")
    return " ".join(sanitized.split())


def extract_json_object(text: object) -> dict[str, object] | None:
    """Parse one JSON object from raw, fenced, or prose-wrapped text.

    Args:
        text: Provider text that may contain a JSON object directly or wrapped
            in Markdown fences or leading prose.

    Returns:
        The first parsed JSON object, or ``None`` when parsing fails.
    """

    raw = coerce_text(text)
    if not raw:
        return None

    candidates: list[str] = [raw]
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines:
            inner_lines = lines[1:]
            if inner_lines and inner_lines[-1].strip().startswith("```"):
                inner_lines = inner_lines[:-1]
            fenced_body = "\n".join(inner_lines).strip()
            if fenced_body:
                candidates.append(fenced_body)

    start_index = raw.find("{")
    end_index = raw.rfind("}")
    if start_index != -1 and end_index > start_index:
        candidates.append(raw[start_index : end_index + 1])

    seen: set[str] = set()
    for candidate in candidates:
        stripped = candidate.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        try:
            payload = json.loads(stripped)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def compact_conversation(
    conversation: ConversationLike | None,
    *,
    max_turns: int,
    max_item_chars: int,
    max_total_chars: int,
) -> tuple[tuple[str, str], ...]:
    """Snapshot and bound recent conversation turns for evaluator prompts.

    Args:
        conversation: Optional conversation history to compact.
        max_turns: Maximum number of most-recent turns to retain.
        max_item_chars: Maximum characters kept for one turn payload.
        max_total_chars: Maximum total characters kept across the compacted
            conversation.

    Returns:
        A tuple of ``(role, content)`` pairs safe to reuse in evaluator calls.
    """

    if not conversation:
        return ()
    try:
        turns = list(conversation)
    except Exception:
        _LOGGER.warning("Conversation decision core failed to snapshot conversation for compaction.", exc_info=True)
        return ()
    if max_turns > 0 and len(turns) > max_turns:
        turns = turns[-max_turns:]
    compacted: list[tuple[str, str]] = []
    total_chars = 0
    for item in turns:
        try:
            if isinstance(item, tuple) and len(item) == 2:
                role, content = item
            else:
                role = safe_getattr(item, "role", "")
                content = safe_getattr(item, "content", "")
        except Exception:
            _LOGGER.warning(
                "Conversation decision core skipped a malformed conversation item during compaction.",
                exc_info=True,
            )
            continue
        role_text = coerce_text(role, default="user", max_chars=64) or "user"
        content_text = coerce_text(content, max_chars=max_item_chars)
        if not content_text:
            continue
        projected_chars = total_chars + len(role_text) + len(content_text)
        if max_total_chars > 0 and projected_chars > max_total_chars:
            remaining_chars = max_total_chars - total_chars - len(role_text)
            if remaining_chars <= 0:
                break
            content_text = content_text[:remaining_chars].rstrip()
            if not content_text:
                break
            compacted.append((role_text, content_text))
            break
        compacted.append((role_text, content_text))
        total_chars = projected_chars
    return tuple(compacted)


def normalize_turn_text(text: str) -> str:
    """Normalize one transcript for stable equality checks across STT updates."""

    return " ".join(coerce_text(text).lower().split())


def detect_provider_timeout_kwarg(call: Callable[..., object]) -> str | None:
    """Detect the timeout keyword supported by one provider call surface.

    Args:
        call: Callable provider entrypoint to inspect.

    Returns:
        The preferred timeout keyword name, or ``None`` when no timeout keyword
        is visible.
    """

    try:
        signature = inspect.signature(call)
    except (TypeError, ValueError):
        return None
    parameter_names = set(signature.parameters)
    if "timeout_seconds" in parameter_names:
        return "timeout_seconds"
    if "timeout" in parameter_names:
        return "timeout"
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return "timeout_seconds"
    return None
