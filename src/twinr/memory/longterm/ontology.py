from __future__ import annotations

import re  # AUDIT-FIX(#2): Canonicalize externally supplied kinds/attribute tokens into stable, safe identifiers.
import unicodedata  # AUDIT-FIX(#2): Normalize Unicode confusables/spacing before taxonomy matching.
from collections.abc import Mapping


LONGTERM_GENERIC_KINDS = frozenset(
    {
        "episode",
        "fact",
        "event",
        "observation",
        "pattern",
        "plan",
        "summary",
    }
)

LONGTERM_MEMORY_SENSITIVITIES = frozenset(
    {
        "low",
        "normal",
        "private",
        "sensitive",
        "critical",
    }
)

_LEGACY_KIND_DEFAULTS: dict[str, tuple[str, dict[str, str]]] = {
    "relationship_fact": (
        "fact",
        {
            "memory_domain": "social",
            "fact_type": "relationship",
        },
    ),
    "contact_method_fact": (
        "fact",
        {
            "memory_domain": "contact",
            "fact_type": "contact_method",
        },
    ),
    "preference_fact": (
        "fact",
        {
            "memory_domain": "preference",
            "fact_type": "preference",
        },
    ),
    "plan_fact": (
        "plan",
        {
            "memory_domain": "planning",
            "plan_type": "stated_plan",
        },
    ),
    "situational_observation": (
        "observation",
        {
            "memory_domain": "situational",
            "observation_type": "situational",
        },
    ),
    "medical_event": (
        "event",
        {
            "memory_domain": "medical",
            "event_domain": "medical",
        },
    ),
    "event_fact": (
        "event",
        {
            "memory_domain": "general",
            "event_domain": "general",
        },
    ),
    "presence_pattern_fact": (
        "pattern",
        {
            "memory_domain": "presence",
            "pattern_type": "presence",
        },
    ),
    "interaction_pattern_fact": (
        "pattern",
        {
            "memory_domain": "interaction",
            "pattern_type": "interaction",
        },
    ),
    "thread_summary": (
        "summary",
        {
            "memory_domain": "thread",
            "summary_type": "thread",
        },
    ),
}

_GENERIC_KIND_DEFAULTS: dict[str, dict[str, str]] = {
    "event": {"event_domain": "general"},
    "fact": {"fact_type": "general"},
    "observation": {"observation_type": "general"},
    "pattern": {"pattern_type": "general"},
    "plan": {"plan_type": "general"},
    "summary": {"summary_type": "general"},
}

_DURABLE_KINDS = frozenset({"fact", "event", "pattern", "plan", "summary"})
_EPISODIC_KINDS = frozenset({"episode", "observation"})

_NORMALIZED_ATTRIBUTE_KEYS = frozenset(  # AUDIT-FIX(#3): Normalize only taxonomy-control attributes; preserve unrelated payload fields.
    {
        "memory_domain",
        "fact_type",
        "event_domain",
        "observation_type",
        "pattern_type",
        "plan_type",
        "summary_type",
    }
)
_DOMAIN_INFERENCE_KEYS = (
    "memory_domain",
    "fact_type",
    "event_domain",
    "observation_type",
    "pattern_type",
    "plan_type",
    "summary_type",
)
_SENSITIVITY_ALIASES: dict[str, str] = {  # AUDIT-FIX(#1): Centralize explicit alias handling before applying fail-closed defaults.
    "medical": "sensitive",
    "restricted": "sensitive",
    "confidential": "private",
    "high": "critical",
}
_TOKEN_SEPARATOR_RE = re.compile(r"[\s\-/]+", re.UNICODE)  # AUDIT-FIX(#2): Treat common separators as equivalent taxonomy delimiters.
_NON_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)
_MULTI_UNDERSCORE_RE = re.compile(r"_+", re.UNICODE)


def _normalize_text(value: object | None) -> str:
    # AUDIT-FIX(#2): Apply NFKC + whitespace collapsing so STT/copy-paste variants do not fragment canonical kinds.
    if value is None:
        return ""
    try:
        text = str(value)
    except Exception:
        return ""
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.split()).strip()


def _normalize_token(value: object | None) -> str:
    # AUDIT-FIX(#2): Canonicalize user/model-provided classifier tokens to lowercase underscore form.
    clean_value = _normalize_text(value).casefold()
    if not clean_value:
        return ""
    clean_value = _TOKEN_SEPARATOR_RE.sub("_", clean_value)
    clean_value = _NON_WORD_RE.sub("_", clean_value)
    return _MULTI_UNDERSCORE_RE.sub("_", clean_value).strip("_")


def _normalize_attributes(attributes: Mapping[str, object] | None) -> dict[str, object]:
    # AUDIT-FIX(#3): Ignore malformed/non-mapping attribute payloads instead of raising in hot paths over corrupted state.
    if attributes is None or not isinstance(attributes, Mapping):
        return {}

    try:
        items = attributes.items()
    except Exception:
        return {}

    normalized: dict[str, object] = {}
    for raw_key, raw_value in items:
        normalized_key = _normalize_token(raw_key)
        if not normalized_key:
            continue
        if normalized_key in _NORMALIZED_ATTRIBUTE_KEYS:
            normalized_value = _normalize_token(raw_value)
            if normalized_value:
                normalized[normalized_key] = normalized_value
            continue
        if isinstance(raw_key, str):
            normalized[raw_key] = raw_value
    return normalized


def normalize_memory_kind(
    kind: str,
    attributes: Mapping[str, object] | None = None,
) -> tuple[str, dict[str, object]]:
    clean_kind = _normalize_token(kind)  # AUDIT-FIX(#2): Canonicalize kind tokens before legacy/generic lookup.
    normalized_attributes = _normalize_attributes(attributes)  # AUDIT-FIX(#3): Sanitize classifier fields and tolerate malformed payloads.
    if clean_kind in _LEGACY_KIND_DEFAULTS:
        canonical_kind, seeded = _LEGACY_KIND_DEFAULTS[clean_kind]
    else:
        canonical_kind = clean_kind
        seeded = _GENERIC_KIND_DEFAULTS.get(canonical_kind, {})
    for key, value in seeded.items():
        normalized_attributes.setdefault(key, value)
    if "memory_domain" not in normalized_attributes:
        inferred = _infer_memory_domain(canonical_kind=canonical_kind, attributes=normalized_attributes)
        if inferred:
            normalized_attributes["memory_domain"] = inferred
    return canonical_kind, normalized_attributes


def normalize_memory_sensitivity(value: str | None) -> str:
    clean_value = _normalize_token(value)  # AUDIT-FIX(#1): Normalize case/separators before applying the sensitivity policy.
    if not clean_value:
        return "normal"
    if clean_value in _SENSITIVITY_ALIASES:
        return _SENSITIVITY_ALIASES[clean_value]
    if clean_value in LONGTERM_MEMORY_SENSITIVITIES:
        return clean_value
    return "sensitive"  # AUDIT-FIX(#1): Unknown non-empty sensitivities fail closed instead of silently downgrading to normal.


def memory_kind_prefix(kind: str) -> str:
    canonical_kind, _attributes = normalize_memory_kind(kind, None)
    return canonical_kind or "memory"  # AUDIT-FIX(#5): Prefix generation now uses sanitized canonical kinds, not raw caller-controlled text.


def is_durable_kind(kind: str) -> bool:
    canonical_kind, _attributes = normalize_memory_kind(kind, None)
    return canonical_kind in _DURABLE_KINDS


def is_episodic_kind(kind: str) -> bool:
    canonical_kind, _attributes = normalize_memory_kind(kind, None)
    return canonical_kind in _EPISODIC_KINDS


def kind_matches(
    kind: str,
    expected_kind: str,
    attributes: Mapping[str, object] | None = None,
    *,
    attr_key: str | None = None,
    attr_value: str | None = None,
) -> bool:
    canonical_kind, normalized_attributes = normalize_memory_kind(kind, attributes)
    expected_canonical_kind, _expected_attributes = normalize_memory_kind(
        expected_kind,
        None,
    )  # AUDIT-FIX(#4): Normalize expected_kind through the same canonicalization pipeline as incoming kinds.
    if canonical_kind != expected_canonical_kind:
        return False
    if attr_key is None:
        return True
    normalized_attr_key = _normalize_token(attr_key)  # AUDIT-FIX(#4): Normalize attribute key aliases/casing before lookup.
    if not normalized_attr_key:
        return False
    raw_value = normalized_attributes.get(normalized_attr_key)
    if raw_value is None:
        return False
    return _normalize_token(raw_value) == _normalize_token(
        attr_value
    )  # AUDIT-FIX(#4): Compare normalized attribute values to avoid false negatives on benign formatting drift.


def is_thread_summary(kind: str, attributes: Mapping[str, object] | None = None) -> bool:
    return kind_matches(kind, "summary", attributes, attr_key="summary_type", attr_value="thread")


def _infer_memory_domain(
    *,
    canonical_kind: str,
    attributes: Mapping[str, object],
) -> str | None:
    for key in _DOMAIN_INFERENCE_KEYS:
        raw_value = attributes.get(key)
        clean_value = _normalize_token(raw_value)  # AUDIT-FIX(#3): Infer only from sanitized non-blank classifier values.
        if clean_value:
            return clean_value
    if canonical_kind in LONGTERM_GENERIC_KINDS and canonical_kind != "episode":
        return canonical_kind
    return None


__all__ = [
    "LONGTERM_GENERIC_KINDS",
    "LONGTERM_MEMORY_SENSITIVITIES",
    "is_durable_kind",
    "is_episodic_kind",
    "is_thread_summary",
    "kind_matches",
    "memory_kind_prefix",
    "normalize_memory_kind",
    "normalize_memory_sensitivity",
]