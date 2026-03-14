from __future__ import annotations

from dataclasses import dataclass

from twinr.memory.longterm.models import (
    LongTermConsolidationResultV1,
    LongTermGraphEdgeCandidateV1,
    LongTermMemoryConflictV1,
    LongTermMemoryObjectV1,
    LongTermTurnExtractionV1,
)
from twinr.memory.longterm.truth import LongTermTruthMaintainer


_DURABLE_KINDS = frozenset(
    {
        "relationship_fact",
        "contact_method_fact",
        "preference_fact",
        "plan_fact",
        "medical_event",
        "event_fact",
    }
)
_EPISODIC_ONLY_KINDS = frozenset({"episode", "situational_observation"})


@dataclass(frozen=True, slots=True)
class LongTermMemoryConsolidator:
    truth_maintainer: LongTermTruthMaintainer
    promotion_confidence_threshold: float = 0.75

    def consolidate(
        self,
        *,
        extraction: LongTermTurnExtractionV1,
        existing_objects: tuple[LongTermMemoryObjectV1, ...] = (),
    ) -> LongTermConsolidationResultV1:
        episodic_objects: list[LongTermMemoryObjectV1] = [extraction.episode.with_updates(status="active")]
        durable_objects: list[LongTermMemoryObjectV1] = []
        deferred_objects: list[LongTermMemoryObjectV1] = []
        conflicts: list[LongTermMemoryConflictV1] = []
        accepted_memory_ids: set[str] = set()

        for candidate in extraction.candidate_objects:
            if candidate.kind in _EPISODIC_ONLY_KINDS:
                episodic_objects.append(candidate.with_updates(status="active"))
                accepted_memory_ids.add(candidate.memory_id)
                continue
            if candidate.kind not in _DURABLE_KINDS:
                deferred_objects.append(candidate)
                continue
            if candidate.confidence < self.promotion_confidence_threshold and not candidate.confirmed_by_user:
                deferred_objects.append(candidate.with_updates(status="candidate"))
                continue
            candidate_conflicts = self.truth_maintainer.detect_conflicts(
                existing_objects=existing_objects,
                candidate=candidate,
            )
            if candidate_conflicts:
                conflicts.extend(candidate_conflicts)
                deferred_objects.append(
                    candidate.with_updates(
                        status="uncertain",
                        conflicts_with=tuple(
                            memory_id
                            for conflict in candidate_conflicts
                            for memory_id in conflict.existing_memory_ids
                        ),
                    )
                )
                continue
            activated = self.truth_maintainer.activate_candidate(
                candidate=candidate,
                existing_objects=existing_objects,
            )
            durable_objects.append(activated)
            accepted_memory_ids.add(candidate.memory_id)

        graph_edges = tuple(
            edge
            for edge in extraction.graph_edges
            if self._edge_is_allowed(edge=edge, accepted_memory_ids=accepted_memory_ids)
        )
        return LongTermConsolidationResultV1(
            turn_id=extraction.turn_id,
            occurred_at=extraction.occurred_at,
            episodic_objects=tuple(episodic_objects),
            durable_objects=tuple(durable_objects),
            deferred_objects=tuple(deferred_objects),
            conflicts=tuple(conflicts),
            graph_edges=graph_edges,
        )

    def _edge_is_allowed(
        self,
        *,
        edge: LongTermGraphEdgeCandidateV1,
        accepted_memory_ids: set[str],
    ) -> bool:
        origin_memory_id = str((edge.attributes or {}).get("origin_memory_id", "")).strip()
        if not origin_memory_id:
            return True
        return origin_memory_id in accepted_memory_ids


__all__ = ["LongTermMemoryConsolidator"]
