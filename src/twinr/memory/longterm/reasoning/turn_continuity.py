"""Compile immediate midterm continuity packets from raw conversation turns.

This module gives runtime persistence a deterministic near-term continuity
layer that does not depend on the slower structured extraction/reflection
pipeline finishing first. The resulting packet is intentionally bounded and
source-language-friendly so fresh follow-up turns can recall the latest
conversation even while the background writer is still enriching durable state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256

from twinr.memory.longterm.core.models import LongTermConversationTurn, LongTermMidtermPacketV1
from twinr.text_utils import collapse_whitespace, retrieval_terms, truncate_text


_DEFAULT_SUMMARY = "Recent conversation continuity from the latest user-assistant turn."
_TURN_CONTINUITY_SCOPE = "turn_continuity"
_MAX_SUMMARY_CHARS = 160
_MAX_DETAILS_CHARS = 1100
_MAX_DETAIL_TEXT_CHARS = 480
_MAX_QUERY_HINTS = 24
_MAX_EXCERPT_CHARS = 120
_TURN_CONTINUITY_RECALL_HINTS = (
    "recent conversation",
    "latest conversation",
    "conversation recap",
    "what we talked about",
    "what we said",
    "letztes gespraech",
    "gespraech zusammenfassung",
    "worueber gesprochen",
    "was wir gesagt haben",
)


def _normalize_text(value: object | None, *, limit: int) -> str:
    """Collapse arbitrary text into one bounded single-line string."""

    return truncate_text(collapse_whitespace(str(value or "")), limit=limit)


def _stable_packet_id(turn: LongTermConversationTurn) -> str:
    """Return a stable midterm packet id for one queued conversation turn."""

    basis = "\x1f".join(
        (
            turn.created_at.isoformat(),
            turn.source,
            turn.modality,
            turn.transcript,
            turn.response,
        )
    )
    return "midterm:turn:" + sha256(basis.encode("utf-8")).hexdigest()[:20]


def _query_hints(*texts: object, limit: int) -> tuple[str, ...]:
    """Return bounded retrieval hints derived from the raw turn text."""

    seen: set[str] = set()
    hints: list[str] = []
    for text in texts:
        clean_text = _normalize_text(text, limit=_MAX_DETAIL_TEXT_CHARS)
        if not clean_text:
            continue
        try:
            terms = retrieval_terms(clean_text)
        except Exception:
            terms = ()
        for term in terms:
            normalized = _normalize_text(term, limit=96)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            hints.append(normalized)
            if len(hints) >= max(0, int(limit)):
                return tuple(hints)
    return tuple(hints)


def is_turn_continuity_packet(
    *,
    kind: object | None,
    attributes: Mapping[str, object] | None,
) -> bool:
    """Return whether one packet/payload belongs to the turn-continuity lane."""

    kind_text = _normalize_text(kind, limit=64)
    if kind_text == "recent_turn_continuity":
        return True
    if not isinstance(attributes, Mapping):
        return False
    return _normalize_text(attributes.get("persistence_scope"), limit=64) == _TURN_CONTINUITY_SCOPE


def turn_continuity_recall_hints(
    *,
    kind: object | None,
    attributes: Mapping[str, object] | None,
) -> tuple[str, ...]:
    """Return generic recap hints for recent turn-continuity packets."""

    if not is_turn_continuity_packet(kind=kind, attributes=attributes):
        return ()
    return _TURN_CONTINUITY_RECALL_HINTS


@dataclass(frozen=True, slots=True)
class LongTermTurnContinuityCompiler:
    """Compile one immediate midterm continuity packet from a conversation turn."""

    max_query_hints: int = _MAX_QUERY_HINTS

    def compile_packet(
        self,
        *,
        turn: LongTermConversationTurn,
    ) -> LongTermMidtermPacketV1 | None:
        """Return a bounded immediate-turn packet or ``None`` when empty."""

        transcript = _normalize_text(turn.transcript, limit=_MAX_DETAIL_TEXT_CHARS)
        response = _normalize_text(turn.response, limit=_MAX_DETAIL_TEXT_CHARS)
        if not transcript and not response:
            return None

        details_parts = [
            "This packet preserves immediate continuity until slower durable-memory enrichment finishes.",
        ]
        if transcript:
            details_parts.append(f"User said: {transcript}")
        if response:
            details_parts.append(f"Assistant answered: {response}")
        details = _normalize_text(" ".join(details_parts), limit=_MAX_DETAILS_CHARS) or None
        semantic_hints = turn_continuity_recall_hints(
            kind="recent_turn_continuity",
            attributes={"persistence_scope": _TURN_CONTINUITY_SCOPE},
        )
        content_hints = _query_hints(
            turn.transcript,
            turn.response,
            turn.source,
            turn.modality,
            limit=max(0, self.max_query_hints - len(semantic_hints)),
        )
        query_hints = tuple(dict.fromkeys((*semantic_hints, *content_hints)))[: max(0, self.max_query_hints)]

        return LongTermMidtermPacketV1(
            packet_id=_stable_packet_id(turn),
            kind="recent_turn_continuity",
            summary=_normalize_text(_DEFAULT_SUMMARY, limit=_MAX_SUMMARY_CHARS),
            details=details,
            query_hints=query_hints,
            sensitivity="normal",
            updated_at=turn.created_at,
            attributes={
                "persistence_scope": _TURN_CONTINUITY_SCOPE,
                "source_type": turn.source,
                "source_modality": turn.modality,
                "source_created_at": turn.created_at.isoformat(),
                # Preserve short raw-turn anchors so downstream display code can
                # ask natural follow-up questions without parsing English
                # internal summary/details text.
                "transcript_excerpt": _normalize_text(turn.transcript, limit=_MAX_EXCERPT_CHARS),
                "response_excerpt": _normalize_text(turn.response, limit=_MAX_EXCERPT_CHARS),
            },
        )


__all__ = [
    "LongTermTurnContinuityCompiler",
    "is_turn_continuity_packet",
    "turn_continuity_recall_hints",
]
