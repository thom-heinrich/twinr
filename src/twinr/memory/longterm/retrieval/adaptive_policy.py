"""Compile prompt-facing adaptive policy hints from stored long-term signals.

The builder in this module turns repeated confirmations, recurring routines,
and proactive delivery/skip outcomes into explicit mid-term policy packets.
These packets do not replace factual recall; they teach the model how to use
that recall more effectively in future turns.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from twinr.memory.longterm.core.models import LongTermMemoryObjectV1, LongTermMidtermPacketV1
from twinr.memory.longterm.proactive.state import LongTermProactiveHistoryEntryV1, LongTermProactiveStateStore
from twinr.text_utils import collapse_whitespace, retrieval_terms, slugify_identifier


_MAX_HINTS = 6


def _normalize_text(value: object | None) -> str:
    """Collapse arbitrary values into one comparable text line."""

    if value is None:
        return ""
    return collapse_whitespace(str(value))


def _safe_terms(value: object | None) -> tuple[str, ...]:
    """Tokenize one text-like value into stable lowercase retrieval terms."""

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


def _attributes_mapping(item: LongTermMemoryObjectV1) -> Mapping[str, object]:
    """Return the object's attributes as a mapping or an empty mapping."""

    if isinstance(item.attributes, Mapping):
        return item.attributes
    return {}


def _object_search_text(item: LongTermMemoryObjectV1) -> str:
    """Build one query-matching text blob from a memory object."""

    attributes = _attributes_mapping(item)
    parts: list[str] = [
        item.kind,
        item.summary,
        item.details or "",
        item.slot_key or "",
        item.value_key or "",
    ]
    for key, value in attributes.items():
        parts.append(_normalize_text(key))
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (list, tuple, set, frozenset)):
            parts.extend(_normalize_text(entry) for entry in value)
        elif value is not None:
            parts.append(_normalize_text(value))
    return _normalize_text(" ".join(part for part in parts if part))


def _history_search_text(entry: LongTermProactiveHistoryEntryV1) -> str:
    """Build one query-matching text blob from proactive history."""

    return _normalize_text(
        " ".join(
            part
            for part in (
                entry.kind,
                entry.summary,
                entry.last_skip_reason or "",
                entry.last_prompt_text or "",
                *entry.source_memory_ids,
            )
            if part
        )
    )


def _query_hints(*texts: object) -> tuple[str, ...]:
    """Return a bounded unique hint list for one adaptive packet."""

    seen: set[str] = set()
    hints: list[str] = []
    for text in texts:
        for term in _safe_terms(text):
            if term in seen:
                continue
            seen.add(term)
            hints.append(term)
            if len(hints) >= _MAX_HINTS:
                return tuple(hints)
    return tuple(hints)


@dataclass(frozen=True, slots=True)
class LongTermAdaptivePolicyBuilder:
    """Compile explicit behavior policies from stored long-term learning signals."""

    proactive_state_store: LongTermProactiveStateStore | None = None
    packet_limit: int = 3
    min_pattern_support: int = 2
    min_proactive_delivery_count: int = 2
    min_proactive_skip_count: int = 2

    def build_packets(
        self,
        *,
        query_text: str | None,
        durable_objects: Iterable[object],
    ) -> tuple[LongTermMidtermPacketV1, ...]:
        """Return adaptive mid-term packets relevant to the current query."""

        clean_query = _normalize_text(query_text)
        query_terms = frozenset(_safe_terms(clean_query))
        if not clean_query or not query_terms:
            return ()

        relevant_objects = tuple(
            item
            for item in durable_objects
            if isinstance(item, LongTermMemoryObjectV1)
        )
        if not relevant_objects and self.proactive_state_store is None:
            return ()

        ranked_packets: list[tuple[int, LongTermMidtermPacketV1]] = []
        seen_packet_ids: set[str] = set()
        relevant_object_ids = {item.memory_id for item in relevant_objects}

        for item in relevant_objects:
            object_terms = frozenset(_safe_terms(_object_search_text(item)))
            if not query_terms.intersection(object_terms):
                continue

            attributes = _attributes_mapping(item)
            if item.status == "active" and (
                item.confirmed_by_user or attributes.get("review_confirmed_by_user") is True
            ):
                packet = self._confirmed_policy_packet(item)
                if packet.packet_id not in seen_packet_ids:
                    ranked_packets.append((100 + _support_count(item), packet))
                    seen_packet_ids.add(packet.packet_id)

            if item.status == "active" and item.kind in {"pattern", "plan"} and _support_count(item) >= self.min_pattern_support:
                packet = self._routine_policy_packet(item)
                if packet.packet_id not in seen_packet_ids:
                    ranked_packets.append((70 + _support_count(item), packet))
                    seen_packet_ids.add(packet.packet_id)

        for entry in self._load_proactive_history():
            history_terms = frozenset(_safe_terms(_history_search_text(entry)))
            source_overlap = bool(relevant_object_ids.intersection(entry.source_memory_ids))
            if not source_overlap and not query_terms.intersection(history_terms):
                continue
            if entry.delivery_count >= self.min_proactive_delivery_count and entry.delivery_count > entry.skip_count:
                packet = self._delivery_policy_packet(entry)
                if packet.packet_id not in seen_packet_ids:
                    ranked_packets.append((90 + entry.delivery_count, packet))
                    seen_packet_ids.add(packet.packet_id)
            if entry.skip_count >= self.min_proactive_skip_count and entry.skip_count > entry.delivery_count:
                packet = self._avoidance_policy_packet(entry)
                if packet.packet_id not in seen_packet_ids:
                    ranked_packets.append((80 + entry.skip_count, packet))
                    seen_packet_ids.add(packet.packet_id)

        ranked_packets.sort(key=lambda row: (row[0], row[1].packet_id), reverse=True)
        return tuple(packet for _, packet in ranked_packets[: max(0, self.packet_limit)])

    def _load_proactive_history(self) -> tuple[LongTermProactiveHistoryEntryV1, ...]:
        """Load proactive history best-effort for adaptive guidance."""

        if self.proactive_state_store is None:
            return ()
        try:
            return self.proactive_state_store.load_entries()
        except Exception:
            return ()

    def _confirmed_policy_packet(self, item: LongTermMemoryObjectV1) -> LongTermMidtermPacketV1:
        """Build one policy that prioritizes a user-confirmed memory."""

        summary = (
            "Treat this as user-confirmed ground truth when the current reply touches the same topic: "
            f"{item.summary}"
        )
        details = (
            "Prefer the confirmed version over generic alternatives or stale assumptions. "
            "If the request is still ambiguous, ask one focused clarification instead of guessing around it."
        )
        return LongTermMidtermPacketV1(
            packet_id=self._packet_id("confirmed", item.memory_id),
            kind="adaptive_confirmed_memory_policy",
            summary=summary,
            details=details,
            source_memory_ids=(item.memory_id,),
            query_hints=_query_hints(item.summary, item.details, item.slot_key, item.value_key),
            sensitivity=item.sensitivity,
            attributes={
                "policy_scope": "confirmed_memory",
                "memory_kind": item.kind,
                "support_count": _support_count(item),
            },
        )

    def _routine_policy_packet(self, item: LongTermMemoryObjectV1) -> LongTermMidtermPacketV1:
        """Build one policy from a repeated routine or pattern."""

        summary = (
            "Ground suggestions in this established routine before offering broader alternatives: "
            f"{item.summary}"
        )
        details = (
            "Use the recurring pattern as a practical default starting point, but keep the suggestion optional "
            "and easy to decline."
        )
        return LongTermMidtermPacketV1(
            packet_id=self._packet_id("routine", item.memory_id),
            kind="adaptive_routine_policy",
            summary=summary,
            details=details,
            source_memory_ids=(item.memory_id,),
            query_hints=_query_hints(item.summary, item.details, item.slot_key, item.value_key),
            sensitivity=item.sensitivity,
            attributes={
                "policy_scope": "routine_pattern",
                "memory_kind": item.kind,
                "support_count": _support_count(item),
            },
        )

    def _delivery_policy_packet(self, entry: LongTermProactiveHistoryEntryV1) -> LongTermMidtermPacketV1:
        """Build one policy from repeated successful proactive usage."""

        summary = (
            "Brief concrete follow-up suggestions around this topic have worked well before: "
            f"{entry.summary}"
        )
        details = (
            "If the current turn touches this area, prefer one practical next step instead of a long menu of vague options."
        )
        return LongTermMidtermPacketV1(
            packet_id=self._packet_id("delivery", entry.candidate_id),
            kind="adaptive_delivery_policy",
            summary=summary,
            details=details,
            source_memory_ids=entry.source_memory_ids,
            query_hints=_query_hints(entry.summary, entry.kind, entry.last_prompt_text),
            sensitivity=entry.sensitivity,
            attributes={
                "policy_scope": "proactive_success",
                "candidate_kind": entry.kind,
                "delivery_count": entry.delivery_count,
                "skip_count": entry.skip_count,
            },
        )

    def _avoidance_policy_packet(self, entry: LongTermProactiveHistoryEntryV1) -> LongTermMidtermPacketV1:
        """Build one policy from repeated skipped proactive attempts."""

        summary = (
            "Similar proactive nudges around this topic were often skipped: "
            f"{entry.summary}"
        )
        details = (
            "Keep related suggestions lightweight and optional. Do not push them unless the user explicitly asks or the need is very clear."
        )
        return LongTermMidtermPacketV1(
            packet_id=self._packet_id("avoid", entry.candidate_id),
            kind="adaptive_avoidance_policy",
            summary=summary,
            details=details,
            source_memory_ids=entry.source_memory_ids,
            query_hints=_query_hints(entry.summary, entry.kind, entry.last_skip_reason),
            sensitivity=entry.sensitivity,
            attributes={
                "policy_scope": "proactive_skip_pattern",
                "candidate_kind": entry.kind,
                "delivery_count": entry.delivery_count,
                "skip_count": entry.skip_count,
                "last_skip_reason": entry.last_skip_reason,
            },
        )

    def _packet_id(self, category: str, seed: str) -> str:
        """Build one stable packet identifier."""

        return f"adaptive:{category}:{slugify_identifier(seed, fallback='policy')}"
