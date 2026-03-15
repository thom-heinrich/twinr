from __future__ import annotations

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


def normalize_memory_kind(
    kind: str,
    attributes: Mapping[str, object] | None = None,
) -> tuple[str, dict[str, object]]:
    clean_kind = " ".join(str(kind or "").split()).strip()
    normalized_attributes = dict(attributes or {})
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
    clean_value = " ".join(str(value or "").split()).strip().lower()
    if clean_value == "medical":
        return "sensitive"
    if clean_value == "restricted":
        return "sensitive"
    if clean_value == "confidential":
        return "private"
    if clean_value == "high":
        return "critical"
    if clean_value in LONGTERM_MEMORY_SENSITIVITIES:
        return clean_value
    return "normal"


def memory_kind_prefix(kind: str) -> str:
    canonical_kind, _attributes = normalize_memory_kind(kind, None)
    return canonical_kind or "memory"


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
    if canonical_kind != expected_kind:
        return False
    if attr_key is None:
        return True
    raw_value = normalized_attributes.get(attr_key)
    if raw_value is None:
        return False
    return str(raw_value).strip().lower() == str(attr_value or "").strip().lower()


def is_thread_summary(kind: str, attributes: Mapping[str, object] | None = None) -> bool:
    return kind_matches(kind, "summary", attributes, attr_key="summary_type", attr_value="thread")


def _infer_memory_domain(
    *,
    canonical_kind: str,
    attributes: Mapping[str, object],
) -> str | None:
    for key in (
        "memory_domain",
        "fact_type",
        "event_domain",
        "observation_type",
        "pattern_type",
        "plan_type",
        "summary_type",
    ):
        raw_value = attributes.get(key)
        clean_value = " ".join(str(raw_value or "").split()).strip()
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
