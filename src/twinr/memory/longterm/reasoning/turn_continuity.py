"""Compile immediate midterm continuity packets from raw conversation turns.

This module gives runtime persistence a deterministic near-term continuity
layer that does not depend on the slower structured extraction/reflection
pipeline finishing first. The resulting packet is intentionally bounded and
source-language-friendly so fresh follow-up turns can recall the latest
conversation even while the background writer is still enriching durable state.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from twinr.memory.longterm.core.models import LongTermConversationTurn, LongTermMidtermPacketV1
from twinr.text_utils import collapse_whitespace, retrieval_terms, truncate_text


_DEFAULT_SUMMARY = "Recent conversation continuity from the latest user-assistant turn."
_TURN_CONTINUITY_SCOPE = "turn_continuity"
_MAX_SUMMARY_CHARS = 160
_MAX_DETAILS_CHARS = 1100
_MAX_DETAIL_TEXT_CHARS = 480
_MAX_QUERY_HINTS = 16
_MAX_EXCERPT_CHARS = 120


def _normalize_text(value: object | None, *, limit: int) -> str:
    """Collapse arbitrary text into one bounded single-line string."""

    return truncate_text(collapse_whitespace(str(value or "")), limit=limit)


def _stable_packet_id(turn: LongTermConversationTurn) -> str:
    """Return a stable midterm packet id for one queued conversation turn."""

    basis = "\x1f".join(
        (
            turn.created_at.isoformat(),
            turn.source,
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

        return LongTermMidtermPacketV1(
            packet_id=_stable_packet_id(turn),
            kind="recent_turn_continuity",
            summary=_normalize_text(_DEFAULT_SUMMARY, limit=_MAX_SUMMARY_CHARS),
            details=details,
            query_hints=_query_hints(
                turn.transcript,
                turn.response,
                turn.source,
                limit=self.max_query_hints,
            ),
            sensitivity="normal",
            updated_at=turn.created_at,
            attributes={
                "persistence_scope": _TURN_CONTINUITY_SCOPE,
                "source_type": turn.source,
                "source_created_at": turn.created_at.isoformat(),
                # Preserve short raw-turn anchors so downstream display code can
                # ask natural follow-up questions without parsing English
                # internal summary/details text.
                "transcript_excerpt": _normalize_text(turn.transcript, limit=_MAX_EXCERPT_CHARS),
                "response_excerpt": _normalize_text(turn.response, limit=_MAX_EXCERPT_CHARS),
            },
        )


__all__ = ["LongTermTurnContinuityCompiler"]
