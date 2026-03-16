"""Maintain slot-level truth state for long-term memory objects.

This module canonicalizes candidate and existing memories, detects live
slot conflicts, and produces bounded clarification prompts when more than
one plausible value remains active for the same slot.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging  # AUDIT-FIX(#2): allow guarded handling of malformed persisted memory rows.
import unicodedata  # AUDIT-FIX(#5): strip control characters from user-facing clarification text.

from twinr.memory.longterm.core.ontology import kind_matches
from twinr.memory.longterm.core.models import LongTermMemoryConflictV1, LongTermMemoryObjectV1


_LOGGER = logging.getLogger(__name__)  # AUDIT-FIX(#2): surface corrupted-memory skips to operational logs.
_MAX_CLARIFICATION_LABEL_CHARS = 160  # AUDIT-FIX(#5): keep spoken/displayed clarification prompts short and readable.


def _normalize_text(value: str | None) -> str:
    if not isinstance(value, str):  # AUDIT-FIX(#5): avoid silently turning arbitrary objects into spoken Python repr strings.
        return ""
    sanitized = "".join(
        " " if unicodedata.category(char).startswith("C") else char
        for char in value
    )
    normalized = " ".join(sanitized.split()).strip()
    if len(normalized) > _MAX_CLARIFICATION_LABEL_CHARS:  # AUDIT-FIX(#5): bound prompt length for TTS/eInk usability.
        return f"{normalized[: _MAX_CLARIFICATION_LABEL_CHARS - 1].rstrip()}…"
    return normalized


def _dedupe_memory_ids(memory_ids: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered_unique: list[str] = []
    for memory_id in memory_ids:
        if not memory_id or memory_id in seen:  # AUDIT-FIX(#4): drop empty and duplicate IDs from degraded file-backed state.
            continue
        seen.add(memory_id)
        ordered_unique.append(memory_id)
    return tuple(ordered_unique)


@dataclass(frozen=True, slots=True)
class LongTermTruthMaintainer:
    """Coordinate slot-level conflict detection and clean activation.

    Attributes:
        active_statuses: Status values that still participate in conflict
            detection for the same slot.
    """

    active_statuses: frozenset[str] = frozenset({"active", "uncertain", "candidate"})

    def detect_conflicts(
        self,
        *,
        existing_objects: tuple[LongTermMemoryObjectV1, ...],
        candidate: LongTermMemoryObjectV1,
    ) -> tuple[LongTermMemoryConflictV1, ...]:
        """Detect slot conflicts for one candidate against existing objects.

        Args:
            existing_objects: Existing long-term memory objects to compare
                against after canonicalization.
            candidate: Candidate object that may be activated.

        Returns:
            Zero or more conflict records describing competing live values
            for the candidate's slot.
        """

        candidate = candidate.canonicalized()
        return self._detect_conflicts_for_canonical_candidate(
            existing_objects=existing_objects,
            candidate=candidate,
        )  # AUDIT-FIX(#1): compare against canonicalized existing memories so normalization differences cannot hide conflicts.

    def activate_candidate(
        self,
        *,
        candidate: LongTermMemoryObjectV1,
        existing_objects: tuple[LongTermMemoryObjectV1, ...],
    ) -> LongTermMemoryObjectV1:
        """Activate a clean candidate or preserve uncertainty when blocked.

        Args:
            candidate: Candidate object to normalize and evaluate.
            existing_objects: Existing long-term memory objects that can
                block activation through slot conflicts.

        Returns:
            The candidate with status and conflict metadata updated to match
            the current slot truth state.
        """

        candidate = candidate.canonicalized()
        conflicts = self._detect_conflicts_for_canonical_candidate(
            existing_objects=existing_objects,
            candidate=candidate,
        )  # AUDIT-FIX(#1): reuse the canonical-candidate path and avoid mixed normalized/raw conflict evaluation.
        if conflicts:
            return candidate.with_updates(
                status="uncertain",
                conflicts_with=_dedupe_memory_ids(
                    tuple(
                        memory_id
                        for conflict in conflicts
                        for memory_id in conflict.existing_memory_ids
                    )
                ),  # AUDIT-FIX(#4): persist only valid unique conflict references.
            )

        had_conflicts = bool(getattr(candidate, "conflicts_with", ()))
        if candidate.status == "candidate":
            return candidate.with_updates(
                status="active",
                conflicts_with=(),
            )  # AUDIT-FIX(#3): clean activation must clear stale conflict references.
        if candidate.status == "uncertain" and had_conflicts:
            return candidate.with_updates(
                status="active",
                conflicts_with=(),
            )  # AUDIT-FIX(#3): conflict-driven uncertainty should auto-resolve once blockers disappear.
        if had_conflicts:
            return candidate.with_updates(conflicts_with=())  # AUDIT-FIX(#3): keep non-conflicting objects from carrying stale conflict edges.
        return candidate

    def _detect_conflicts_for_canonical_candidate(
        self,
        *,
        existing_objects: tuple[LongTermMemoryObjectV1, ...],
        candidate: LongTermMemoryObjectV1,
    ) -> tuple[LongTermMemoryConflictV1, ...]:
        if not candidate.slot_key or not candidate.value_key:
            return ()

        conflicting = tuple(
            item
            for item in self._canonical_existing_objects(existing_objects)
            if item.memory_id != candidate.memory_id
            and item.status in self.active_statuses
            and item.slot_key == candidate.slot_key
            and item.value_key
            and item.value_key != candidate.value_key
        )  # AUDIT-FIX(#1): normalize all existing rows before status/slot/value comparisons.
        if not conflicting:
            return ()

        return (
            LongTermMemoryConflictV1(
                slot_key=candidate.slot_key,
                candidate_memory_id=candidate.memory_id,
                existing_memory_ids=_dedupe_memory_ids(
                    tuple(item.memory_id for item in conflicting)
                ),  # AUDIT-FIX(#4): conflict objects must not contain duplicate or empty memory IDs.
                question=self._clarification_question(candidate, conflicting),
                reason=f"Conflicting active memories exist for slot {candidate.slot_key}.",
            ),
        )

    def _canonical_existing_objects(
        self,
        existing_objects: tuple[LongTermMemoryObjectV1, ...],
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        canonical_existing: list[LongTermMemoryObjectV1] = []
        for item in existing_objects:
            try:
                canonical_existing.append(item.canonicalized())
            except Exception:
                _LOGGER.warning(
                    "Skipping invalid long-term memory object during conflict detection.",
                    exc_info=True,
                )  # AUDIT-FIX(#2): degrade gracefully on partially corrupted persisted state instead of crashing the whole pass.
        return tuple(canonical_existing)

    def _clarification_question(
        self,
        candidate: LongTermMemoryObjectV1,
        existing_objects: tuple[LongTermMemoryObjectV1, ...],
    ) -> str:
        label = _normalize_text(candidate.summary)
        if kind_matches(candidate.kind, "fact", candidate.attributes, attr_key="fact_type", attr_value="contact_method"):
            return "I have more than one contact detail for this person. Which one should I use?"
        if kind_matches(candidate.kind, "fact", candidate.attributes, attr_key="fact_type", attr_value="relationship"):
            return "I have conflicting relationship information. Which one is correct now?"
        if candidate.kind == "event":
            return "I have more than one memory about this event. Which one is correct now?"  # AUDIT-FIX(#5): keep the spoken noun accurate for the memory kind.
        if candidate.kind == "plan":
            return "I have more than one memory about this plan. Which one is correct now?"  # AUDIT-FIX(#5): avoid calling plans “events” in user-facing speech.
        if label:
            return f"I have conflicting memories about this detail: {label.rstrip('.!?')}. Which one is correct now?"  # AUDIT-FIX(#5): add a clear sentence boundary for TTS and avoid doubled punctuation.
        return "I have conflicting memories here. Which one is correct now?"


__all__ = ["LongTermTruthMaintainer"]
