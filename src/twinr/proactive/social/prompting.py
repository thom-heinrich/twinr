# CHANGELOG: 2026-03-29
# BUG-1: Fixed evidence starvation where malformed leading entries consumed the max_items budget and hid later valid evidence.
# BUG-2: Fixed mid-iteration crashes from broken/generator-based evidence containers by consuming iterables defensively.
# SEC-1: Replaced ad-hoc fact strings with explicit UNTRUSTED_EVIDENCE JSON lines to prevent field spoofing and reduce indirect prompt-injection risk.
# SEC-2: Added hard caps for max_items / scan budget / raw text coercion to block prompt-amplification and memory/CPU DoS on Raspberry Pi 4.
# IMP-1: Added typed ObservationFact records, deduplication, and weight-based prioritization so the highest-signal evidence reaches the LLM first.
# IMP-2: Added cheap instruction-like / encoded-content heuristics and trust-boundary metadata for downstream planners and auditors.

"""Route social-trigger prompts and render bounded evidence facts.

This module keeps safety-routing decisions and LLM-facing evidence summaries
small, deterministic, and resilient to malformed trigger payloads.

Design note:
- Untrusted evidence is rendered as structured data with an explicit trust label.
- A new typed record API is exposed for downstream systems that want to keep
  prompt/data separation instead of concatenating ad-hoc strings.
"""

from __future__ import annotations

import json
import logging
import math
import re
import reprlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from .engine import SocialTriggerDecision

logger = logging.getLogger(__name__)

_SAFETY_TRIGGER_IDS: Final[frozenset[str]] = frozenset(
    {
        "possible_fall",
        "floor_stillness",
        "distress_possible",
    }
)

_DIRECT_SAFETY_MODE: Final[str] = "direct_safety"
_LLM_MODE: Final[str] = "llm"

_DEFAULT_MAX_ITEMS: Final[int] = 5
_HARD_MAX_ITEMS: Final[int] = 16
_SCAN_MULTIPLIER: Final[int] = 8
_MIN_SCAN_ITEMS: Final[int] = 16
_HARD_SCAN_LIMIT: Final[int] = 128

_UNKNOWN_KEY_PLACEHOLDER: Final[str] = "[unknown]"
_NONE_DETAIL_PLACEHOLDER: Final[str] = "[none]"

_MAX_FACT_KEY_CHARS: Final[int] = 64
_MAX_FACT_DETAIL_CHARS: Final[int] = 160
_MAX_RAW_KEY_SCAN_CHARS: Final[int] = 256
_MAX_RAW_DETAIL_SCAN_CHARS: Final[int] = 1024
_NUMERIC_DECIMALS: Final[int] = 4

_UNTRUSTED_SOURCE_TRUST: Final[str] = "untrusted_evidence"
_FACT_KIND: Final[str] = "observation_fact"
_FACT_SCHEMA_VERSION: Final[int] = 2

_INSTRUCTION_LIKE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:"
    r"\bignore(?: all)?(?: previous| prior| above)? instructions?\b|"
    r"\bsystem prompt\b|"
    r"\bdeveloper message\b|"
    r"\bassistant message\b|"
    r"\btool (?:call|result)\b|"
    r"\bfunction call\b|"
    r"\boverride\b|"
    r"\bjailbreak\b|"
    r"\bbypass\b|"
    r"\byou are (?:chatgpt|claude|an ai)\b|"
    r"\bdo not obey\b|"
    r"\bfollow these steps\b|"
    r"```|<system>|</system>|<assistant>|</assistant>"
    r")"
)
_OBFUSCATED_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:\b(?:[A-Za-z0-9+/]{24,}={0,2})\b|\b(?:[0-9A-F]{24,})\b)"
)
_ROLE_PREFIX_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)^(?:system|assistant|developer|tool|user)\s*:"
)


def _get_field(source: object, field_name: str) -> Any:
    """Read one named field from an object or mapping."""

    if isinstance(source, Mapping):
        return source.get(field_name)
    try:
        return getattr(source, field_name)
    except Exception:
        return None


def _normalize_trigger_id(trigger_id: object) -> str:
    """Normalize one trigger identifier to a lowercase key."""

    if not isinstance(trigger_id, str):
        return ""
    return trigger_id.strip().lower()


def _coerce_finite_number(value: object) -> float | None:
    """Convert one numeric value to a bounded finite float, or None."""

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(number):
        return None
    return round(number, _NUMERIC_DECIMALS)


def _bounded_repr(value: object, *, raw_limit: int) -> str:
    """Render one arbitrary object to a bounded repr for safe prompt use."""

    helper = reprlib.Repr()
    helper.maxstring = raw_limit
    helper.maxother = raw_limit
    helper.maxlist = 8
    helper.maxtuple = 8
    helper.maxset = 8
    helper.maxfrozenset = 8
    helper.maxdeque = 8
    helper.maxdict = 8
    helper.maxarray = 8
    return helper.repr(value)


def _bounded_text_prefix(value: object, *, raw_limit: int) -> tuple[str, tuple[str, ...]]:
    """Convert one value to a bounded text prefix without scanning unlimited input."""

    if value is None:
        return "", ()
    if isinstance(value, str):
        if len(value) > raw_limit:
            return value[:raw_limit], ("raw_scan_truncated",)
        return value, ()
    if isinstance(value, bytes):
        clipped = value[:raw_limit]
        flags = ["bytes_decoded"]
        if len(value) > raw_limit:
            flags.append("raw_scan_truncated")
        try:
            return clipped.decode("utf-8", errors="replace"), tuple(flags)
        except Exception:
            return "", ("bytes_decode_failed",)
    if isinstance(value, (int, float, bool)):
        return str(value), ()

    try:
        rendered = _bounded_repr(value, raw_limit=raw_limit)
    except Exception:
        return "", ("stringify_failed",)
    return rendered, ("repr_used",)


def _sanitize_fact_text(
    value: object,
    *,
    placeholder: str,
    max_length: int,
    raw_scan_limit: int,
) -> tuple[str, tuple[str, ...]]:
    """Collapse, bound, and annotate one evidence text fragment."""

    raw_text, raw_flags = _bounded_text_prefix(value, raw_limit=raw_scan_limit)
    if not raw_text:
        return placeholder, raw_flags

    printable_text = "".join(char if char.isprintable() else " " for char in raw_text)
    flags = list(raw_flags)
    if printable_text != raw_text:
        flags.append("control_chars_removed")

    normalized_text = " ".join(printable_text.split())
    if not normalized_text:
        return placeholder, tuple(sorted(set(flags)))

    if len(normalized_text) > max_length:
        normalized_text = f"{normalized_text[: max_length - 3].rstrip()}..."
        flags.append("text_truncated")

    if _ROLE_PREFIX_RE.search(normalized_text) or _INSTRUCTION_LIKE_RE.search(normalized_text):
        flags.append("instruction_like")
    if _OBFUSCATED_TOKEN_RE.search(normalized_text):
        flags.append("encoded_like")

    return normalized_text, tuple(sorted(set(flags)))


def _validated_max_items(max_items: int) -> int:
    """Validate and hard-cap the public evidence-item limit."""

    if isinstance(max_items, bool) or not isinstance(max_items, int):
        logger.warning(
            "proactive_observation_facts received invalid max_items; using default %d",
            _DEFAULT_MAX_ITEMS,
        )
        return _DEFAULT_MAX_ITEMS
    if max_items < 0:
        logger.warning(
            "proactive_observation_facts received negative max_items; clamping to 0"
        )
        return 0
    if max_items > _HARD_MAX_ITEMS:
        logger.warning(
            "proactive_observation_facts received oversized max_items=%d; clamping to %d",
            max_items,
            _HARD_MAX_ITEMS,
        )
        return _HARD_MAX_ITEMS
    return max_items


def _scan_budget(limit: int) -> int:
    """Compute how many raw items may be inspected to recover valid facts."""

    return min(max(limit * _SCAN_MULTIPLIER, _MIN_SCAN_ITEMS), _HARD_SCAN_LIMIT)


def _iter_evidence_items(trigger: object, *, scan_limit: int) -> Iterator[object]:
    """Yield up to scan_limit raw evidence items while containing iterable failures."""

    evidence = _get_field(trigger, "evidence")
    if evidence is None:
        return

    if isinstance(evidence, Mapping):
        yield evidence
        return
    if isinstance(evidence, (str, bytes)):
        logger.warning(
            "proactive_observation_facts received textual evidence container; ignoring it"
        )
        return

    try:
        iterator = iter(evidence)
    except TypeError:
        logger.warning(
            "proactive_observation_facts received non-iterable evidence container; ignoring it"
        )
        return

    for _ in range(scan_limit):
        try:
            item = next(iterator)
        except StopIteration:
            return
        except Exception:
            logger.warning(
                "proactive_observation_facts encountered an exception while iterating evidence; stopping early"
            )
            return
        yield item


@dataclass(frozen=True, slots=True)
class ObservationFact:
    """One bounded, typed observation fact ready for structured downstream use."""

    key: str
    detail: str
    value: float | None
    weight: float | None
    flags: tuple[str, ...] = field(default_factory=tuple)
    source_trust: str = _UNTRUSTED_SOURCE_TRUST
    kind: str = _FACT_KIND
    schema_version: int = _FACT_SCHEMA_VERSION

    def as_payload(self) -> dict[str, object]:
        """Return one structured payload for tool/LLM integrations."""

        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "source_trust": self.source_trust,
            "key": self.key,
            "value": self.value,
            "weight": self.weight,
            "detail": self.detail,
            "flags": list(self.flags),
        }

    def render(self) -> str:
        """Render one fact as a canonical JSON line with an explicit trust boundary."""

        return (
            "UNTRUSTED_EVIDENCE "
            + json.dumps(
                self.as_payload(),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )


def is_safety_trigger(trigger_id: object) -> bool:
    """Return whether one trigger id uses the direct safety path."""

    return _normalize_trigger_id(trigger_id) in _SAFETY_TRIGGER_IDS


def proactive_prompt_mode(trigger: SocialTriggerDecision) -> str:
    """Return the prompt-generation mode for one trigger."""

    normalized_trigger_id = _normalize_trigger_id(_get_field(trigger, "trigger_id"))
    if not normalized_trigger_id:
        logger.warning(
            "proactive_prompt_mode received a trigger without a usable trigger_id; "
            "falling back to direct_safety"
        )
        return _DIRECT_SAFETY_MODE
    return _DIRECT_SAFETY_MODE if normalized_trigger_id in _SAFETY_TRIGGER_IDS else _LLM_MODE


def _to_observation_fact(item: object) -> ObservationFact | None:
    """Convert one raw evidence item into a normalized typed fact."""

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
        return None

    key, key_flags = _sanitize_fact_text(
        key_value,
        placeholder=_UNKNOWN_KEY_PLACEHOLDER,
        max_length=_MAX_FACT_KEY_CHARS,
        raw_scan_limit=_MAX_RAW_KEY_SCAN_CHARS,
    )
    detail, detail_flags = _sanitize_fact_text(
        detail_value,
        placeholder=_NONE_DETAIL_PLACEHOLDER,
        max_length=_MAX_FACT_DETAIL_CHARS,
        raw_scan_limit=_MAX_RAW_DETAIL_SCAN_CHARS,
    )
    value = _coerce_finite_number(value_value)
    weight = _coerce_finite_number(weight_value)

    flags = list(key_flags + detail_flags)
    if key_value is None:
        flags.append("missing_key")
    if detail_value is None:
        flags.append("missing_detail")
    if value is None and value_value is not None:
        flags.append("invalid_value")
    if weight is None and weight_value is not None:
        flags.append("invalid_weight")

    return ObservationFact(
        key=key,
        detail=detail,
        value=value,
        weight=weight,
        flags=tuple(sorted(set(flags))),
    )


def proactive_observation_records(
    trigger: SocialTriggerDecision,
    *,
    max_items: int = _DEFAULT_MAX_ITEMS,
) -> tuple[ObservationFact, ...]:
    """Return bounded, ranked observation records for one trigger."""

    limit = _validated_max_items(max_items)
    if limit == 0:
        return ()

    candidates: list[tuple[int, ObservationFact]] = []
    for index, item in enumerate(_iter_evidence_items(trigger, scan_limit=_scan_budget(limit))):
        fact = _to_observation_fact(item)
        if fact is None:
            continue
        candidates.append((index, fact))

    if not candidates:
        return ()

    deduped: dict[tuple[str, str, float | None], tuple[int, ObservationFact]] = {}
    for index, fact in candidates:
        dedupe_key = (fact.key, fact.detail, fact.value)
        existing = deduped.get(dedupe_key)
        if existing is None:
            deduped[dedupe_key] = (index, fact)
            continue

        existing_index, existing_fact = existing
        existing_weight = float("-inf") if existing_fact.weight is None else existing_fact.weight
        current_weight = float("-inf") if fact.weight is None else fact.weight
        if (current_weight, -index) > (existing_weight, -existing_index):
            deduped[dedupe_key] = (index, fact)

    ranked = sorted(
        deduped.values(),
        key=lambda pair: (
            float("-inf") if pair[1].weight is None else pair[1].weight,
            float("-inf") if pair[1].value is None else pair[1].value,
            -pair[0],
        ),
        reverse=True,
    )
    return tuple(fact for _, fact in ranked[:limit])


def proactive_observation_facts(
    trigger: SocialTriggerDecision,
    *,
    max_items: int = _DEFAULT_MAX_ITEMS,
) -> tuple[str, ...]:
    """Render bounded evidence facts for one trigger."""

    # BREAKING: Facts now render as canonical UNTRUSTED_EVIDENCE JSON lines instead
    # of ad-hoc strings like "key: value=..., weight=..., detail=...".
    # BREAKING: Invalid numeric fields now serialize as JSON null, not the string "nan".
    return tuple(
        record.render() for record in proactive_observation_records(trigger, max_items=max_items)
    )


__all__ = [
    "ObservationFact",
    "is_safety_trigger",
    "proactive_observation_facts",
    "proactive_observation_records",
    "proactive_prompt_mode",
]