from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass

from twinr.memory.longterm.core.ontology import is_durable_kind, is_episodic_kind
from twinr.memory.longterm.core.models import (
    LongTermConsolidationResultV1,
    LongTermGraphEdgeCandidateV1,
    LongTermMemoryConflictV1,
    LongTermMemoryObjectV1,
    LongTermTurnExtractionV1,
)
from twinr.memory.longterm.reasoning.truth import LongTermTruthMaintainer

# AUDIT-FIX(#2): Structured logging makes degraded candidate handling observable instead of silently losing signal.
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LongTermMemoryConsolidator:
    truth_maintainer: LongTermTruthMaintainer
    promotion_confidence_threshold: float = 0.75

    # AUDIT-FIX(#5): Validate and normalize configuration once so misconfigured thresholds cannot silently over-promote or suppress memories.
    def __post_init__(self) -> None:
        if self.truth_maintainer is None:
            raise ValueError("truth_maintainer must not be None")

        threshold = self.promotion_confidence_threshold
        if isinstance(threshold, bool):
            raise ValueError("promotion_confidence_threshold must be a float in the range [0.0, 1.0]")

        try:
            normalized_threshold = float(threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError("promotion_confidence_threshold must be a float in the range [0.0, 1.0]") from exc

        if not math.isfinite(normalized_threshold) or not 0.0 <= normalized_threshold <= 1.0:
            raise ValueError("promotion_confidence_threshold must be a finite float in the range [0.0, 1.0]")

        object.__setattr__(self, "promotion_confidence_threshold", normalized_threshold)

    def consolidate(
        self,
        *,
        extraction: LongTermTurnExtractionV1,
        existing_objects: tuple[LongTermMemoryObjectV1, ...] = (),
    ) -> LongTermConsolidationResultV1:
        if extraction is None:
            raise ValueError("extraction must not be None")
        if extraction.episode is None:
            raise ValueError("extraction.episode must not be None")

        # AUDIT-FIX(#3): Use a rolling effective view so same-turn activations participate in later conflict checks.
        effective_existing_objects: tuple[LongTermMemoryObjectV1, ...] = tuple(existing_objects or ())

        # AUDIT-FIX(#1): Guard episode activation so a single malformed episode does not crash the entire turn consolidation.
        active_episode = self._safe_with_status(
            extraction.episode,
            status="active",
            context="episode",
        )
        episodic_objects: list[LongTermMemoryObjectV1] = [active_episode]
        durable_objects: list[LongTermMemoryObjectV1] = []
        deferred_objects: list[LongTermMemoryObjectV1] = []
        conflicts: list[LongTermMemoryConflictV1] = []

        # AUDIT-FIX(#4,#6): Seed accepted IDs with the always-accepted episode and keep IDs normalized for edge filtering.
        accepted_memory_ids: set[str] = self._memory_ids(extraction.episode, active_episode)

        for index, candidate in enumerate(getattr(extraction, "candidate_objects", ()) or ()):
            # AUDIT-FIX(#2): Skip null candidates defensively instead of crashing while trying to canonicalize them.
            if candidate is None:
                logger.warning(
                    "Skipping null long-term memory candidate",
                    extra={"candidate_index": index, "turn_id": getattr(extraction, "turn_id", None)},
                )
                continue

            try:
                normalized_candidate = candidate.canonicalized()
                candidate_memory_ids = self._memory_ids(candidate, normalized_candidate)

                if is_episodic_kind(normalized_candidate.kind):
                    active_candidate = self._safe_with_status(
                        normalized_candidate,
                        status="active",
                        context=f"episodic candidate #{index + 1}",
                    )
                    episodic_objects.append(active_candidate)
                    accepted_memory_ids.update(self._memory_ids(candidate, normalized_candidate, active_candidate))
                    continue

                if not is_durable_kind(normalized_candidate.kind):
                    deferred_objects.append(normalized_candidate)
                    continue

                if self._requires_confirmation(normalized_candidate):
                    deferred_objects.append(
                        self._safe_with_status(
                            normalized_candidate,
                            status="candidate",
                            context=f"durable candidate #{index + 1}",
                        )
                    )
                    continue

                # AUDIT-FIX(#2,#3): Materialize conflicts and evaluate them against the rolling effective object set.
                candidate_conflicts = tuple(
                    self.truth_maintainer.detect_conflicts(
                        existing_objects=effective_existing_objects,
                        candidate=normalized_candidate,
                    )
                )
                if candidate_conflicts:
                    conflicts.extend(candidate_conflicts)
                    deferred_objects.append(
                        self._safe_with_status(
                            normalized_candidate,
                            status="uncertain",
                            conflicts_with=self._conflict_memory_ids(candidate_conflicts),
                            context=f"conflicted candidate #{index + 1}",
                        )
                    )
                    continue

                activated = self.truth_maintainer.activate_candidate(
                    candidate=normalized_candidate,
                    existing_objects=effective_existing_objects,
                )
                if activated is None:
                    raise ValueError("truth_maintainer.activate_candidate returned None")

                durable_objects.append(activated)
                # AUDIT-FIX(#3): Feed accepted same-turn activations back into later truth maintenance within the same batch.
                effective_existing_objects = (*effective_existing_objects, activated)
                accepted_memory_ids.update(self._memory_ids(candidate, normalized_candidate, activated))
            except Exception:
                # AUDIT-FIX(#2): Isolate candidate failures so one bad object degrades to deferred instead of aborting the full turn.
                logger.exception(
                    "Failed to consolidate long-term memory candidate",
                    extra={
                        "candidate_index": index,
                        "candidate_memory_ids": sorted(self._memory_ids(candidate)),
                        "turn_id": getattr(extraction, "turn_id", None),
                    },
                )
                deferred_objects.append(
                    self._safe_with_status(
                        candidate,
                        status="uncertain",
                        context=f"failed candidate #{index + 1}",
                    )
                )

        graph_edges = tuple(
            edge
            for edge in (getattr(extraction, "graph_edges", ()) or ())
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
        # AUDIT-FIX(#7): Drop null edges safely instead of returning them as implicitly allowed.
        if edge is None:
            logger.warning("Dropping null long-term graph edge")
            return False

        attributes = getattr(edge, "attributes", None)
        if attributes is None:
            return True

        # AUDIT-FIX(#7): Reject malformed edge metadata safely instead of assuming a dict-like payload and crashing on `.get`.
        if not isinstance(attributes, Mapping):
            logger.warning(
                "Dropping long-term graph edge with non-mapping attributes",
                extra={"edge_type": type(edge).__name__},
            )
            return False

        origin_memory_id = self._normalize_memory_id(attributes.get("origin_memory_id", ""))
        if not origin_memory_id:
            return True
        return origin_memory_id in accepted_memory_ids

    def _requires_confirmation(self, candidate: LongTermMemoryObjectV1) -> bool:
        confidence = self._safe_confidence(candidate)
        confirmed_by_user = getattr(candidate, "confirmed_by_user", False) is True
        return confidence < self.promotion_confidence_threshold and not confirmed_by_user

    def _safe_confidence(self, candidate: LongTermMemoryObjectV1) -> float:
        raw_confidence = getattr(candidate, "confidence", 0.0)
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid long-term memory confidence encountered; treating as 0.0",
                extra={"memory_ids": sorted(self._memory_ids(candidate))},
            )
            return 0.0

        if not math.isfinite(confidence):
            logger.warning(
                "Non-finite long-term memory confidence encountered; treating as 0.0",
                extra={"memory_ids": sorted(self._memory_ids(candidate))},
            )
            return 0.0

        # AUDIT-FIX(#5): Confidence values outside [0.0, 1.0] are treated as invalid so malformed models cannot auto-promote durable memories.
        if not 0.0 <= confidence <= 1.0:
            logger.warning(
                "Out-of-range long-term memory confidence encountered; treating as 0.0",
                extra={"memory_ids": sorted(self._memory_ids(candidate)), "confidence": confidence},
            )
            return 0.0

        return confidence

    def _safe_with_status(
        self,
        memory_object: LongTermMemoryObjectV1,
        *,
        status: str,
        context: str,
        conflicts_with: tuple[str, ...] = (),
    ) -> LongTermMemoryObjectV1:
        update_kwargs: dict[str, object] = {"status": status}
        if conflicts_with:
            update_kwargs["conflicts_with"] = conflicts_with

        try:
            return memory_object.with_updates(**update_kwargs)
        except Exception:
            logger.exception(
                "Failed to update long-term memory object state",
                extra={
                    "context": context,
                    "memory_ids": sorted(self._memory_ids(memory_object)),
                    "requested_status": status,
                },
            )
            return memory_object

    def _conflict_memory_ids(
        self,
        candidate_conflicts: tuple[LongTermMemoryConflictV1, ...],
    ) -> tuple[str, ...]:
        seen: set[str] = set()
        ordered_ids: list[str] = []
        for conflict in candidate_conflicts:
            for memory_id in getattr(conflict, "existing_memory_ids", ()):
                normalized_memory_id = self._normalize_memory_id(memory_id)
                if not normalized_memory_id or normalized_memory_id in seen:
                    continue
                seen.add(normalized_memory_id)
                ordered_ids.append(normalized_memory_id)
        return tuple(ordered_ids)

    def _memory_ids(self, *memory_objects: LongTermMemoryObjectV1) -> set[str]:
        normalized_ids: set[str] = set()
        for memory_object in memory_objects:
            if memory_object is None:
                continue
            memory_id = self._normalize_memory_id(getattr(memory_object, "memory_id", ""))
            if memory_id:
                normalized_ids.add(memory_id)
        return normalized_ids

    def _normalize_memory_id(self, memory_id: object) -> str:
        # AUDIT-FIX(#6): Treat `None` as missing instead of converting it into the literal string `"None"`.
        if memory_id is None:
            return ""
        normalized = str(memory_id).strip()
        return normalized


__all__ = ["LongTermMemoryConsolidator"]