"""Conversation-recap helpers shared by long-term memory retrieval paths.

Generic recap questions should match real conversation-turn episodes even when
their stored summary/details stay intentionally generic. The helpers here keep
that recap vocabulary bounded and scoped to genuine conversation episodes so
sensor or automation episodes do not get widened into the same lane.
"""

from __future__ import annotations

from collections.abc import Mapping

from twinr.text_utils import collapse_whitespace, retrieval_terms, truncate_text

_MAX_NORMALIZED_CHARS = 256
_CONVERSATION_EPISODE_RECALL_HINTS = (
    "recent conversation",
    "latest conversation",
    "conversation recap",
    "conversation history",
    "what we talk about",
    "what we talked about",
    "what we said",
    "recent chat",
    "letztes gespraech",
    "gespraech zusammenfassung",
    "unterhaltung zusammenfassung",
    "worueber gesprochen",
    "was wir gesagt haben",
)
_CONVERSATION_RECAP_QUERY_VARIANTS = (
    "worueber gesprochen",
    "conversation recap",
    "what we talked about",
)
_CONVERSATION_RECAP_SIGNAL_TERMS = frozenset(
    {
        "conversation",
        "recap",
        "history",
        "talk",
        "talked",
        "said",
        "chat",
        "gespraech",
        "zusammenfassung",
        "unterhaltung",
        "worueber",
        "gesprochen",
    }
)


def _normalize_text(value: object | None, *, limit: int = _MAX_NORMALIZED_CHARS) -> str:
    """Collapse arbitrary text into one bounded line for recall checks."""

    return truncate_text(collapse_whitespace(str(value or "")), limit=limit)


def is_conversation_episode_object(
    *,
    kind: object | None,
    attributes: Mapping[str, object] | None,
) -> bool:
    """Return whether one object/payload represents a real conversation turn."""

    if _normalize_text(kind, limit=64) != "episode":
        return False
    if not isinstance(attributes, Mapping):
        return False
    transcript = _normalize_text(attributes.get("raw_transcript"))
    response = _normalize_text(attributes.get("raw_response"))
    return bool(transcript or response)


def conversation_episode_recall_hints(
    *,
    kind: object | None,
    attributes: Mapping[str, object] | None,
) -> tuple[str, ...]:
    """Return generic recap hints for conversation-turn episode objects."""

    if not is_conversation_episode_object(kind=kind, attributes=attributes):
        return ()
    return _CONVERSATION_EPISODE_RECALL_HINTS


def query_has_conversation_recap_semantics(query_text: str | None) -> bool:
    """Return whether one query asks for a generic conversation recap."""

    query_terms = {
        term
        for term in retrieval_terms(_normalize_text(query_text))
        if isinstance(term, str) and term
    }
    if not query_terms:
        return False
    return bool(query_terms.intersection(_CONVERSATION_RECAP_SIGNAL_TERMS))


def conversation_recap_query_variants(query_text: str | None) -> tuple[str, ...]:
    """Return bounded search variants for generic conversation recap queries."""

    clean_query = _normalize_text(query_text)
    if not clean_query:
        return ()
    variants = [clean_query]
    if not query_has_conversation_recap_semantics(clean_query):
        return tuple(variants)
    for variant in _CONVERSATION_RECAP_QUERY_VARIANTS:
        normalized = _normalize_text(variant)
        if normalized and normalized not in variants:
            variants.append(normalized)
    return tuple(variants)


def conversation_recap_specific_terms(query_text: str | None) -> tuple[str, ...]:
    """Return informative non-recap terms that make one recap query specific.

    Generic recap questions like "What did we talk about?" should still widen
    to recent conversation episodes. Queries such as "What did we say about
    topic 045?" must keep their topic anchor instead of collapsing into an
    arbitrary generic recap lane.
    """

    clean_query = _normalize_text(query_text)
    if not clean_query:
        return ()
    query_terms = [
        term
        for term in retrieval_terms(clean_query)
        if isinstance(term, str) and term
    ]
    specific_terms = {
        term
        for term in query_terms
        if term not in _CONVERSATION_RECAP_SIGNAL_TERMS and (term.isdigit() or len(term) >= 4)
    }
    return tuple(sorted(specific_terms))


__all__ = [
    "conversation_episode_recall_hints",
    "conversation_recap_specific_terms",
    "conversation_recap_query_variants",
    "query_has_conversation_recap_semantics",
    "is_conversation_episode_object",
]
