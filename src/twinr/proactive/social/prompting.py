"""Route social-trigger prompts and render bounded evidence facts.

This module keeps safety-routing decisions and LLM-facing evidence summaries
small, deterministic, and resilient to malformed trigger payloads.
"""

from __future__ import annotations

# AUDIT-FIX(#1): Add lightweight diagnostics for conservative fallbacks.
# AUDIT-FIX(#2): Add safe numeric/text handling and generic evidence iteration support.
# AUDIT-FIX(#5): Use iterator-based limiting instead of brittle slicing.
# AUDIT-FIX(#6): Add immutable constants for stable mode strings and defaults.
import logging
import math
from collections.abc import Iterable, Mapping
from itertools import islice
from typing import Any, Final

from .engine import SocialTriggerDecision

# AUDIT-FIX(#1): Emit diagnostics for malformed safety-routing input without leaking payload data.
logger = logging.getLogger(__name__)

_SAFETY_TRIGGER_IDS: Final[frozenset[str]] = frozenset(
    {
        "possible_fall",
        "floor_stillness",
        "distress_possible",
    }
)
# AUDIT-FIX(#6): Eliminate duplicated mode literals and centralize stable defaults.
_DIRECT_SAFETY_MODE: Final[str] = "direct_safety"
_LLM_MODE: Final[str] = "llm"
_DEFAULT_MAX_ITEMS: Final[int] = 5

# AUDIT-FIX(#2): Preserve fact rendering when evidence is partial or malformed.
_UNKNOWN_KEY_PLACEHOLDER: Final[str] = "[unknown]"
_NONE_DETAIL_PLACEHOLDER: Final[str] = "[none]"
_INVALID_NUMBER_PLACEHOLDER: Final[str] = "nan"

# AUDIT-FIX(#4): Bound untrusted evidence text before it reaches prompts/logs.
_MAX_FACT_KEY_CHARS: Final[int] = 64
_MAX_FACT_DETAIL_CHARS: Final[int] = 160


# AUDIT-FIX(#1): Read fields defensively from objects or mappings and never let malformed models raise here.
def _get_field(source: object, field_name: str) -> Any:
    """Read one named field from an object or mapping."""

    if isinstance(source, Mapping):
        return source.get(field_name)
    try:
        return getattr(source, field_name)
    except Exception:
        return None


# AUDIT-FIX(#3): Invalid trigger IDs normalize to an empty sentinel instead of crashing .strip().lower().
def _normalize_trigger_id(trigger_id: object) -> str:
    """Normalize one trigger identifier to a lowercase key."""

    if not isinstance(trigger_id, str):
        return ""
    return trigger_id.strip().lower()


# AUDIT-FIX(#2): Invalid, missing, or non-finite numeric evidence must not break prompt construction.
def _format_number(value: object) -> str:
    """Format one numeric evidence value for prompt text."""

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return _INVALID_NUMBER_PLACEHOLDER

    if not math.isfinite(number):
        return _INVALID_NUMBER_PLACEHOLDER
    return f"{number:.2f}"


# AUDIT-FIX(#2): Handle malformed or missing text fields without raising.
# AUDIT-FIX(#4): Collapse whitespace, drop control characters, and cap lengths for prompt safety.
def _sanitize_fact_text(value: object, *, placeholder: str, max_length: int) -> str:
    """Collapse and bound one evidence text fragment."""

    try:
        raw_text = value if isinstance(value, str) else "" if value is None else str(value)
    except Exception:
        return placeholder

    printable_text = "".join(char if char.isprintable() else " " for char in raw_text)
    normalized_text = " ".join(printable_text.split())
    if not normalized_text:
        return placeholder
    if len(normalized_text) <= max_length:
        return normalized_text

    truncated = normalized_text[: max_length - 3].rstrip()
    return f"{truncated}..."


# AUDIT-FIX(#5): Reject surprising negative/non-int limits that would otherwise slice incorrectly or raise.
def _validated_max_items(max_items: int) -> int:
    """Validate the public evidence-item limit."""

    if isinstance(max_items, bool) or not isinstance(max_items, int):
        logger.warning(
            "proactive_observation_facts received invalid max_items; using default %d",
            _DEFAULT_MAX_ITEMS,
        )
        return _DEFAULT_MAX_ITEMS
    if max_items < 0:
        logger.warning("proactive_observation_facts received negative max_items; clamping to 0")
        return 0
    return max_items


# AUDIT-FIX(#2): Support lists, tuples, generators, and mapping-style singleton evidence without slicing.
def _iter_evidence_items(trigger: object, *, limit: int) -> Iterable[object]:
    """Yield up to ``limit`` evidence items from one trigger."""

    evidence = _get_field(trigger, "evidence")
    if evidence is None:
        return ()
    if isinstance(evidence, Mapping):
        return (evidence,)
    if isinstance(evidence, (str, bytes)):
        logger.warning("proactive_observation_facts received textual evidence container; ignoring it")
        return ()
    try:
        iterator = iter(evidence)
    except TypeError:
        logger.warning("proactive_observation_facts received non-iterable evidence container; ignoring it")
        return ()
    return islice(iterator, limit)


def is_safety_trigger(trigger_id: str) -> bool:
    """Return whether one trigger id uses the direct safety path."""

    # AUDIT-FIX(#3): Normalize untrusted trigger IDs safely.
    return _normalize_trigger_id(trigger_id) in _SAFETY_TRIGGER_IDS


def proactive_prompt_mode(trigger: SocialTriggerDecision) -> str:
    """Return the prompt-generation mode for one trigger."""

    # AUDIT-FIX(#1): Avoid AttributeError on malformed triggers and fail safe for seniors.
    normalized_trigger_id = _normalize_trigger_id(_get_field(trigger, "trigger_id"))
    if not normalized_trigger_id:
        logger.warning(
            "proactive_prompt_mode received a trigger without a usable trigger_id; "
            "falling back to direct_safety"
        )
        return _DIRECT_SAFETY_MODE
    return _DIRECT_SAFETY_MODE if normalized_trigger_id in _SAFETY_TRIGGER_IDS else _LLM_MODE


# AUDIT-FIX(#5): Keep the public default aligned with validated fallback logic.
def proactive_observation_facts(
    trigger: SocialTriggerDecision,
    *,
    max_items: int = _DEFAULT_MAX_ITEMS,
) -> tuple[str, ...]:
    """Render bounded evidence facts for one trigger."""

    # AUDIT-FIX(#5): Clamp and validate limits before iteration.
    limit = _validated_max_items(max_items)
    if limit == 0:
        return ()

    facts: list[str] = []
    for item in _iter_evidence_items(trigger, limit=limit):
        key_value = _get_field(item, "key")
        detail_value = _get_field(item, "detail")
        value_value = _get_field(item, "value")
        weight_value = _get_field(item, "weight")
        if (
            key_value is None
            and detail_value is None
            and value_value is None
            and weight_value is None
        ):
            continue  # AUDIT-FIX(#2): Skip wholly malformed evidence items instead of emitting nonsense facts.

        # AUDIT-FIX(#2): Missing text fields must not break serialization.
        # AUDIT-FIX(#4): Sanitize untrusted evidence before it reaches prompt text.
        key = _sanitize_fact_text(
            key_value,
            placeholder=_UNKNOWN_KEY_PLACEHOLDER,
            max_length=_MAX_FACT_KEY_CHARS,
        )
        detail = _sanitize_fact_text(
            detail_value,
            placeholder=_NONE_DETAIL_PLACEHOLDER,
            max_length=_MAX_FACT_DETAIL_CHARS,
        )
        facts.append(
            f"{key}: value={_format_number(value_value)}, "
            f"weight={_format_number(weight_value)}, "
            f"detail={detail}"
        )  # AUDIT-FIX(#2): Never raise while serializing partial evidence.
    return tuple(facts)


__all__ = [
    "is_safety_trigger",
    "proactive_observation_facts",
    "proactive_prompt_mode",
]
