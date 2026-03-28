# CHANGELOG: 2026-03-27
# BUG-1: compact_conversation no longer drops the newest turn when max_total_chars is hit;
#        it now budgets from the tail, matching modern sliding-window compaction behavior.
# BUG-2: extract_json_object now finds the first decodable JSON object inside prose/fences
#        instead of failing on earlier stray braces or trailing junk.
# BUG-3: normalize_turn_text now performs Unicode-aware, punctuation-tolerant normalization
#        so STT punctuation updates do not churn equality checks.
# BUG-4: detect_provider_timeout_kwarg now prefers the de-facto 2026 SDK keyword "timeout"
#        for **kwargs call surfaces instead of the often-ignored "timeout_seconds".
# BUG-5: coerce_int now treats overflow/non-finite numeric inputs as parse failures instead
#        of raising unexpectedly.
# SEC-1: extract_json_object now enforces a hard parse-size cap and rejects NaN/Infinity-like
#        pseudo-JSON constants to reduce CPU/memory abuse from untrusted payloads.
# SEC-2: sanitize_emit_value now strips control/format characters including ANSI escape and
#        bidi controls to prevent practical log/terminal injection.
# IMP-1: compact_conversation now supports modern provider content arrays / block objects
#        (OpenAI Responses/Conversations, Anthropic Messages, Gemini parts) via deterministic
#        text flattening instead of unstable repr() output.
# IMP-2: optional partial JSON recovery via pydantic-core is supported when installed,
#        enabling safer frontier-style streaming / truncated structured-output parsing.
# IMP-3: recent-turn snapshotting uses deque(maxlen=...) to avoid copying full histories on
#        long-running Raspberry Pi deployments.

"""Provide shared decision helpers for conversation evaluators.

This module owns the bounded coercion, JSON extraction, conversation
compaction, timeout detection, and transcript normalization helpers shared by
the turn-boundary and closure evaluators. Keep the helpers provider-agnostic,
deterministic, and free of workflow orchestration so they remain safe to reuse
across runtime-facing conversation modules.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping
import inspect
import json
import logging
import math
from typing import Any, cast
import unicodedata

from twinr.agent.base_agent.contracts import ConversationLike

_LOGGER = logging.getLogger(__name__)

_JSON_PARSE_MAX_CHARS = 262_144
_JSON_SCAN_MAX_STARTS = 512
_ROLE_MAX_CHARS = 64
_TRUNCATION_MARKER = "…"

_JSON_TEXT_FIELDS = (
    "text",
    "parts",
    "turns",
    "transcript",
    "refusal",
    "summary",
    "title",
    "message",
    "reasoning",
    "content",
    "input",
    "arguments",
    "output",
    "result",
    "value",
)
_JSON_SKIP_FIELDS = frozenset(
    {
        "annotations",
        "audio",
        "b64_json",
        "bytes",
        "citations",
        "data",
        "file",
        "file_data",
        "file_id",
        "file_ids",
        "file_url",
        "id",
        "image",
        "image_url",
        "images",
        "index",
        "inline_data",
        "logprobs",
        "metadata",
        "mime_type",
        "mimetype",
        "raw",
        "role",
        "status",
        "token_count",
        "tokens",
        "type",
        "url",
        "urls",
        "video",
    }
)
_PLAIN_TEXT_BLOCK_TYPES = frozenset({"text", "input_text", "output_text"})
_TIMEOUT_KWARG_PREFERENCE = (
    "timeout",
    "timeout_seconds",
    "request_timeout",
    "deadline",
    "request_deadline",
)

_PydanticFromJson = Callable[..., Any]
_pydantic_from_json: _PydanticFromJson | None

try:
    from pydantic_core import from_json as _pydantic_from_json_impl
except Exception:  # pragma: no cover - optional dependency
    _pydantic_from_json = None
else:
    _pydantic_from_json = _pydantic_from_json_impl


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


def _coerce_bytes_text(value: object) -> str | None:
    """Best-effort decode bytes-like values into text."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, bytearray):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, memoryview):
        return value.tobytes().decode("utf-8", errors="replace")
    return None


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
        decoded = _coerce_bytes_text(value)
        if decoded is not None:
            text = decoded
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
        if isinstance(value, float) and not math.isfinite(value):
            return default
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
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite float")
        number = int(cast(Any, value))
    except (TypeError, ValueError, OverflowError):
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


def _neutralize_control_text(text: str, *, max_chars: int | None = None) -> str:
    """Collapse whitespace and strip control / format characters."""

    pieces: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if char.isspace() or category.startswith("Z"):
            pieces.append(" ")
            continue
        if category.startswith("C"):
            continue
        pieces.append(char)
        if max_chars is not None and max_chars >= 0 and len(pieces) >= max_chars:
            break
    normalized = "".join(pieces)
    if max_chars is not None and max_chars >= 0 and len(normalized) > max_chars:
        normalized = normalized[:max_chars]
    return " ".join(normalized.split())


def sanitize_emit_value(value: object, *, max_chars: int) -> str:
    """Strip control characters from one emitted controller status value."""

    sanitized = _neutralize_control_text(coerce_text(value), max_chars=max_chars)
    if len(sanitized) > max_chars:
        sanitized = sanitized[:max_chars].rstrip()
    return sanitized


def _reject_nonstandard_json_constant(token: str) -> object:
    """Reject NaN / Infinity-style constants that are invalid JSON."""

    raise ValueError(f"Invalid JSON constant: {token!r}")


def _iter_markdown_code_blocks(text: str) -> Iterable[str]:
    """Yield fenced code block bodies found anywhere in ``text``."""

    start = 0
    while True:
        fence_open = text.find("```", start)
        if fence_open == -1:
            return
        line_end = text.find("\n", fence_open)
        if line_end == -1:
            return
        fence_close = text.find("```", line_end + 1)
        if fence_close == -1:
            return
        block = text[line_end + 1 : fence_close].strip()
        if block:
            yield block
        start = fence_close + 3


def _looks_like_json_object_start(text: str, start: int) -> bool:
    """Return ``True`` when ``text[start:]`` plausibly begins a JSON object."""

    index = start + 1
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text):
        return False
    return text[index] in {'"', "}"}


def _iter_json_object_starts(text: str, *, max_starts: int) -> Iterable[int]:
    """Yield candidate object start offsets in ``text``."""

    found = 0
    index = text.find("{")
    while index != -1 and found < max_starts:
        if _looks_like_json_object_start(text, index):
            yield index
            found += 1
        index = text.find("{", index + 1)


def _iter_json_candidates(raw: str) -> Iterable[str]:
    """Yield likely JSON-containing slices in priority order."""

    stripped = raw.strip()
    if stripped:
        yield stripped
    for block in _iter_markdown_code_blocks(raw):
        yield block
    start_index = stripped.find("{")
    end_index = stripped.rfind("}")
    if start_index != -1 and end_index > start_index:
        yield stripped[start_index : end_index + 1]


def _decode_json_object_from_text(text: str) -> dict[str, object] | None:
    """Return the first strict JSON object found in ``text``."""

    decoder = json.JSONDecoder(parse_constant=_reject_nonstandard_json_constant)
    stripped = text.strip()
    if not stripped:
        return None

    try:
        payload = decoder.decode(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = None
    if isinstance(payload, dict):
        return payload

    first_start = next(iter(_iter_json_object_starts(text, max_starts=1)), None)
    if first_start is not None:
        try:
            payload, _ = decoder.raw_decode(text, first_start)
        except (TypeError, ValueError, json.JSONDecodeError):
            if first_start == 0:
                return None
        else:
            if isinstance(payload, dict):
                return payload

    seen_starts: set[int] = set()
    for start in _iter_json_object_starts(text, max_starts=_JSON_SCAN_MAX_STARTS):
        if start in seen_starts or start == first_start:
            continue
        seen_starts.add(start)
        try:
            payload, _ = decoder.raw_decode(text, start)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _decode_partial_json_object(text: str) -> dict[str, object] | None:
    """Best-effort partial JSON parsing when pydantic-core is available."""

    if _pydantic_from_json is None:
        return None
    for start in _iter_json_object_starts(text, max_starts=_JSON_SCAN_MAX_STARTS):
        try:
            payload = _pydantic_from_json(text[start:], allow_partial=True)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def extract_json_object(
    text: object,
    *,
    max_chars: int = _JSON_PARSE_MAX_CHARS,
    allow_partial: bool = False,
) -> dict[str, object] | None:
    """Parse one JSON object from raw, fenced, or prose-wrapped text.

    Args:
        text: Provider text that may contain a JSON object directly or wrapped
            in Markdown fences or leading prose.
        max_chars: Hard upper bound for the parsed source text.
        allow_partial: When ``True``, attempt best-effort partial JSON recovery
            via ``pydantic_core.from_json`` when that optional dependency is
            installed.

    Returns:
        The first parsed JSON object, or ``None`` when parsing fails.
    """

    raw = coerce_text(text, max_chars=max_chars)
    if not raw:
        return None

    seen_candidates: set[str] = set()
    for candidate in _iter_json_candidates(raw):
        normalized_candidate = candidate.strip()
        if not normalized_candidate or normalized_candidate in seen_candidates:
            continue
        seen_candidates.add(normalized_candidate)
        payload = _decode_json_object_from_text(normalized_candidate)
        if payload is not None:
            return payload

    if allow_partial:
        for candidate in tuple(seen_candidates):
            payload = _decode_partial_json_object(candidate)
            if payload is not None:
                return payload
    return None


def _truncate_middle(text: str, *, max_chars: int) -> str:
    """Truncate text with a middle ellipsis while preserving both ends."""

    if max_chars < 0 or len(text) <= max_chars:
        return text
    if max_chars <= len(_TRUNCATION_MARKER):
        return text[:max_chars]
    if max_chars <= 8:
        return text[: max_chars - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER
    head_chars = (max_chars - len(_TRUNCATION_MARKER)) // 2
    tail_chars = max_chars - len(_TRUNCATION_MARKER) - head_chars
    head = text[:head_chars].rstrip()
    tail = text[-tail_chars:].lstrip() if tail_chars > 0 else ""
    truncated = f"{head}{_TRUNCATION_MARKER}{tail}"
    if len(truncated) > max_chars:
        truncated = truncated[:max_chars]
    return truncated


def _append_with_budget(parts: list[str], piece: str, remaining: int | None) -> int | None:
    """Append a piece of text while respecting an optional character budget."""

    normalized = coerce_text(piece)
    if not normalized:
        return remaining
    prefix = " " if parts else ""
    allowance = None if remaining is None else remaining - len(prefix)
    if allowance is not None and allowance <= 0:
        return 0
    if allowance is not None and len(normalized) > allowance:
        normalized = _truncate_middle(normalized, max_chars=allowance).rstrip()
        if not normalized:
            return 0
        parts.append(prefix + normalized)
        return 0
    parts.append(prefix + normalized)
    if remaining is None:
        return None
    return max(remaining - len(prefix) - len(normalized), 0)


def _mapping_block_marker(mapping: Mapping[object, object]) -> str:
    """Create a compact marker for non-plain provider block types."""

    type_text = coerce_text(mapping.get("type"), max_chars=64).lower()
    if not type_text or type_text in _PLAIN_TEXT_BLOCK_TYPES:
        return ""
    name_text = coerce_text(mapping.get("name"), max_chars=64)
    if name_text:
        return f"[{type_text}:{name_text}]"
    return f"[{type_text}]"


def _render_content_text(
    value: object,
    *,
    max_chars: int | None,
    depth: int = 0,
    seen_ids: set[int] | None = None,
) -> str:
    """Render common provider content blocks into stable bounded text."""

    if max_chars is not None and max_chars < 0:
        max_chars = None
    if max_chars is not None and max_chars == 0:
        return ""
    if value is None:
        return ""
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        return coerce_text(value, max_chars=max_chars)

    if depth >= 6:
        return coerce_text(value, max_chars=max_chars)

    if seen_ids is None:
        seen_ids = set()

    if isinstance(value, Mapping):
        object_id = id(value)
        if object_id in seen_ids:
            return ""
        seen_ids.add(object_id)
        try:
            parts: list[str] = []
            remaining = max_chars
            marker = _mapping_block_marker(value)
            if marker:
                remaining = _append_with_budget(parts, marker, remaining)
            for field in _JSON_TEXT_FIELDS:
                if field == "content" and field in value and value.get("content") is value:
                    continue
                if field not in value:
                    continue
                piece = _render_content_text(
                    value[field],
                    max_chars=remaining,
                    depth=depth + 1,
                    seen_ids=seen_ids,
                )
                remaining = _append_with_budget(parts, piece, remaining)
                if remaining == 0:
                    break
            if remaining != 0:
                known_keys = set(_JSON_TEXT_FIELDS) | {"name"} | _JSON_SKIP_FIELDS
                for key in sorted(value, key=lambda item: coerce_text(item)):
                    if key in known_keys:
                        continue
                    piece = _render_content_text(
                        value[key],
                        max_chars=remaining,
                        depth=depth + 1,
                        seen_ids=seen_ids,
                    )
                    if not piece:
                        continue
                    labeled_piece = f"{coerce_text(key, max_chars=32)}={piece}"
                    remaining = _append_with_budget(parts, labeled_piece, remaining)
                    if remaining == 0:
                        break
            return "".join(parts)
        finally:
            seen_ids.discard(object_id)

    if isinstance(value, (list, tuple, deque)):
        object_id = id(value)
        if object_id in seen_ids:
            return ""
        seen_ids.add(object_id)
        try:
            sequence_parts: list[str] = []
            remaining = max_chars
            for item in value:
                piece = _render_content_text(
                    item,
                    max_chars=remaining,
                    depth=depth + 1,
                    seen_ids=seen_ids,
                )
                remaining = _append_with_budget(sequence_parts, piece, remaining)
                if remaining == 0:
                    break
            return "".join(sequence_parts)
        finally:
            seen_ids.discard(object_id)

    if isinstance(value, (set, frozenset)):
        rendered_items = sorted(
            (_render_content_text(item, max_chars=max_chars, depth=depth + 1, seen_ids=seen_ids) for item in value),
            key=str,
        )
        return _render_content_text(rendered_items, max_chars=max_chars, depth=depth + 1, seen_ids=seen_ids)

    for attribute_name in ("text", "transcript", "refusal", "content", "input", "output", "result"):
        attribute_value = safe_getattr(value, attribute_name, None)
        if attribute_value is not None and attribute_value is not value:
            return _render_content_text(
                attribute_value,
                max_chars=max_chars,
                depth=depth + 1,
                seen_ids=seen_ids,
            )

    return coerce_text(value, max_chars=max_chars)


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
        if max_turns > 0:
            turns = list(deque(cast(Iterable[object], conversation), maxlen=max_turns))
        else:
            turns = list(conversation)
    except Exception:
        _LOGGER.warning(
            "Conversation decision core failed to snapshot conversation for compaction.",
            exc_info=True,
        )
        return ()

    compacted_reversed: list[tuple[str, str]] = []
    total_chars = 0

    for item in reversed(turns):
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

        role_text = coerce_text(role, default="user", max_chars=_ROLE_MAX_CHARS) or "user"
        content_text = _render_content_text(content, max_chars=max_item_chars)
        if not content_text:
            continue

        projected_chars = total_chars + len(role_text) + len(content_text)
        if max_total_chars > 0 and projected_chars > max_total_chars:
            remaining_chars = max_total_chars - total_chars - len(role_text)
            if remaining_chars <= 0:
                break
            truncated_content = _truncate_middle(content_text, max_chars=remaining_chars).rstrip()
            if not truncated_content:
                break
            compacted_reversed.append((role_text, truncated_content))
            break

        compacted_reversed.append((role_text, content_text))
        total_chars = projected_chars

    compacted_reversed.reverse()
    return tuple(compacted_reversed)


def normalize_turn_text(text: str) -> str:
    """Normalize one transcript for stable equality checks across STT updates."""

    raw = unicodedata.normalize("NFKC", coerce_text(text)).casefold()
    normalized_chars: list[str] = []
    length = len(raw)

    for index, char in enumerate(raw):
        category = unicodedata.category(char)
        if category.startswith("C"):
            continue
        if char.isspace() or category.startswith("Z"):
            normalized_chars.append(" ")
            continue
        if category.startswith("P"):
            previous_char = raw[index - 1] if index > 0 else ""
            next_char = raw[index + 1] if index + 1 < length else ""
            if char in {".", ",", "/", "-"} and previous_char.isdigit() and next_char.isdigit():
                normalized_chars.append(char)
            else:
                normalized_chars.append(" ")
            continue
        normalized_chars.append(char)

    return " ".join("".join(normalized_chars).split())


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

    parameter_names = tuple(signature.parameters)
    for name in _TIMEOUT_KWARG_PREFERENCE:
        if name in parameter_names:
            return name

    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return "timeout"
    return None
