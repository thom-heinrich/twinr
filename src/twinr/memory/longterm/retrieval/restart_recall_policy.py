"""Compile persistent restart-recall packets from stable durable memories.

This module turns highly stable durable memories into explicit mid-term packets
that survive fresh runtime roots. The goal is not to replace durable retrieval;
it is to preserve a small provenance-rich continuity layer so confirmed or
otherwise strong durable facts can still steer prompt context immediately after
restart.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from twinr.memory.longterm.core.models import LongTermMemoryObjectV1, LongTermMidtermPacketV1
from twinr.text_utils import collapse_whitespace, retrieval_terms, slugify_identifier


_MAX_RESTART_PACKETS = 24
_MIN_CONFIDENCE = 0.97
_MIN_SUPPORT_COUNT = 2
_PERSISTENCE_SCOPE = "restart_recall"


def _normalize_text(value: object | None) -> str:
    """Collapse arbitrary values into one comparable text line."""

    if value is None:
        return ""
    return collapse_whitespace(str(value))


def _safe_terms(value: object | None) -> tuple[str, ...]:
    """Tokenize one text-like value into stable retrieval terms."""

    clean_value = _normalize_text(value)
    if not clean_value:
        return ()
    try:
        return tuple(
            term
            for term in retrieval_terms(clean_value)
            if isinstance(term, str) and term and not term.isdigit()
        )
    except Exception:
        return ()


def _support_count(item: LongTermMemoryObjectV1) -> int:
    """Read the normalized support count from one memory object."""

    attributes = item.attributes if isinstance(item.attributes, Mapping) else {}
    try:
        return max(0, int(attributes.get("support_count", 0)))
    except (TypeError, ValueError):
        return 0


def _query_hints(*texts: object) -> tuple[str, ...]:
    """Return a bounded unique hint list for one restart packet."""

    seen: set[str] = set()
    hints: list[str] = []
    for text in texts:
        for term in _safe_terms(text):
            if term in seen:
                continue
            seen.add(term)
            hints.append(term)
            if len(hints) >= 12:
                return tuple(hints)
    return tuple(hints)


def _source_event_ids(item: LongTermMemoryObjectV1) -> tuple[str, ...]:
    """Return stable source event ids from one durable memory object."""

    source = getattr(item, "source", None)
    event_ids = getattr(source, "event_ids", ())
    if not isinstance(event_ids, (list, tuple)):
        return ()
    return tuple(str(value) for value in event_ids if isinstance(value, str) and value)


@dataclass(frozen=True, slots=True)
class LongTermRestartRecallPolicyCompiler:
    """Compile persistent restart-recall packets from durable memories."""

    packet_limit: int = _MAX_RESTART_PACKETS
    min_confidence: float = _MIN_CONFIDENCE
    min_support_count: int = _MIN_SUPPORT_COUNT

    def build_packets(
        self,
        *,
        objects: Iterable[object],
    ) -> tuple[LongTermMidtermPacketV1, ...]:
        """Return persistent restart packets for stable durable memories."""

        stable_objects = tuple(
            item
            for item in objects
            if isinstance(item, LongTermMemoryObjectV1) and self._should_persist(item)
        )
        ranked = sorted(
            stable_objects,
            key=lambda item: (
                1 if item.confirmed_by_user else 0,
                _support_count(item),
                item.confidence,
                item.updated_at,
                item.memory_id,
            ),
            reverse=True,
        )
        return tuple(
            self._packet_from_object(item)
            for item in ranked[: max(0, int(self.packet_limit))]
        )

    def _should_persist(self, item: LongTermMemoryObjectV1) -> bool:
        """Return whether one durable memory is stable enough for restart recall."""

        if item.kind == "episode":
            return False
        if item.status != "active":
            return False
        return (
            item.confirmed_by_user
            or item.confidence >= float(self.min_confidence)
            or _support_count(item) >= int(self.min_support_count)
        )

    def _packet_from_object(self, item: LongTermMemoryObjectV1) -> LongTermMidtermPacketV1:
        """Build one provenance-rich restart packet from a durable memory."""

        source_event_ids = _source_event_ids(item)
        summary = (
            "Persistent restart recall for this stable durable memory: "
            f"{item.summary}"
        )
        details = (
            "Use this packet as direct grounding after fresh runtime restarts when the current turn overlaps the same topic. "
            "Prefer it over generic sibling memories when the overlap is clear."
        )
        return LongTermMidtermPacketV1(
            packet_id=self._packet_id(item.memory_id),
            kind="adaptive_restart_recall_policy",
            summary=summary,
            details=details,
            source_memory_ids=(item.memory_id,),
            query_hints=_query_hints(
                item.summary,
                item.details,
                item.slot_key,
                item.value_key,
                "confirmed_by_user" if item.confirmed_by_user else "",
                "bestaetigt" if item.confirmed_by_user else "",
                item.status,
            ),
            sensitivity=item.sensitivity,
            valid_from=item.valid_from,
            valid_to=item.valid_to,
            attributes={
                "policy_scope": _PERSISTENCE_SCOPE,
                "persistence_scope": _PERSISTENCE_SCOPE,
                "memory_kind": item.kind,
                "memory_status": item.status,
                "confirmed_by_user": item.confirmed_by_user,
                "slot_key": item.slot_key,
                "value_key": item.value_key,
                "support_count": _support_count(item),
                "confidence": round(float(item.confidence), 6),
                "source_type": getattr(getattr(item, "source", None), "source_type", None),
                "source_speaker": getattr(getattr(item, "source", None), "speaker", None),
                "source_modality": getattr(getattr(item, "source", None), "modality", None),
                "source_event_ids": list(source_event_ids),
            },
        )

    def _packet_id(self, memory_id: str) -> str:
        """Build one stable packet id from the source memory id."""

        return f"adaptive:restart:{slugify_identifier(memory_id, fallback='memory')}"


__all__ = ["LongTermRestartRecallPolicyCompiler"]
