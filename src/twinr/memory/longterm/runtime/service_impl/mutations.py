# mypy: disable-error-code=attr-defined
"""Mutation and review entry points for the long-term runtime service."""

from __future__ import annotations

from twinr.memory.context_store import ManagedContextEntry, PersistentMemoryEntry
from twinr.memory.longterm.core.models import (
    LongTermConflictQueueItemV1,
    LongTermConflictResolutionV1,
    LongTermMemoryMutationResultV1,
    LongTermMemoryReviewResultV1,
)
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError

from ._typing import ServiceMixinBase
from .compat import (
    _DEFAULT_REVIEW_LIMIT,
    _MAX_REVIEW_LIMIT,
    _coerce_positive_int,
    logger,
)


class LongTermMemoryServiceMutationMixin(ServiceMixinBase):
    """Conflict resolution, review, and prompt-context mutation helpers."""

    def select_conflict_queue(
        self,
        query_text: str | None,
        *,
        limit: int | None = None,
    ) -> tuple[LongTermConflictQueueItemV1, ...]:
        """Select open memory conflicts relevant to one query."""

        try:
            query = self.query_rewriter.profile(query_text)
            normalized_limit = None if limit is None else _coerce_positive_int(limit, default=1, maximum=_MAX_REVIEW_LIMIT)
            with self._store_lock:
                with self._temporary_remote_probe_cache():
                    return self.retriever.select_conflict_queue(
                        query=query,
                        limit=normalized_limit,
                    )
        except LongTermRemoteUnavailableError:
            raise
        except Exception:
            logger.exception("Failed to select long-term conflict queue.")
            return ()

    def resolve_conflict(
        self,
        *,
        slot_key: str,
        selected_memory_id: str,
    ) -> LongTermConflictResolutionV1:
        """Resolve one open conflict by selecting the surviving memory."""

        with self._store_lock:
            conflicts = self.object_store.load_conflicts()
            conflict = next((item for item in conflicts if item.slot_key == slot_key), None)
            if conflict is None:
                raise ValueError(f"No open long-term memory conflict found for slot {slot_key!r}.")
            result = self.conflict_resolver.resolve(
                conflict=conflict,
                objects=self.object_store.load_objects(),
                remaining_conflicts=conflicts,
                selected_memory_id=selected_memory_id,
            )
            self.object_store.apply_conflict_resolution(result)
            self._refresh_restart_recall_packets_locked()
            return result

    def review_memory(
        self,
        *,
        query_text: str | None = None,
        status: str | None = None,
        kind: str | None = None,
        include_episodes: bool = False,
        limit: int = _DEFAULT_REVIEW_LIMIT,
    ) -> LongTermMemoryReviewResultV1:
        """Review stored memories with optional query and status filters."""

        normalized_limit = _coerce_positive_int(limit, default=_DEFAULT_REVIEW_LIMIT, maximum=_MAX_REVIEW_LIMIT)
        with self._store_lock:
            return self.object_store.review_objects(
                query_text=query_text,
                status=status,
                kind=kind,
                include_episodes=include_episodes,
                limit=normalized_limit,
            )

    def confirm_memory(self, *, memory_id: str) -> LongTermMemoryMutationResultV1 | LongTermConflictResolutionV1:
        """Confirm one memory, or resolve its conflict if it is disputed."""

        with self._store_lock:
            conflicts = self.object_store.load_conflicts()
            conflict = next(
                (
                    item
                    for item in conflicts
                    if memory_id == item.candidate_memory_id or memory_id in item.existing_memory_ids
                ),
                None,
            )
            if conflict is not None:
                result = self.conflict_resolver.resolve(
                    conflict=conflict,
                    objects=self.object_store.load_objects(),
                    remaining_conflicts=conflicts,
                    selected_memory_id=memory_id,
                )
                self.object_store.apply_conflict_resolution(result)
                self._refresh_restart_recall_packets_locked()
                return result
            result = self.object_store.confirm_object(memory_id)
            self.object_store.apply_memory_mutation(result)
            self._refresh_restart_recall_packets_locked()
            return result

    def invalidate_memory(
        self,
        *,
        memory_id: str,
        reason: str | None = None,
    ) -> LongTermMemoryMutationResultV1:
        """Mark one memory invalid and persist the resulting mutation."""

        with self._store_lock:
            result = self.object_store.invalidate_object(memory_id, reason=reason)
            self.object_store.apply_memory_mutation(result)
            self._refresh_restart_recall_packets_locked()
            return result

    def delete_memory(self, *, memory_id: str) -> LongTermMemoryMutationResultV1:
        """Delete one memory and persist the resulting mutation."""

        with self._store_lock:
            result = self.object_store.delete_object(memory_id)
            self.object_store.apply_memory_mutation(result)
            self._refresh_restart_recall_packets_locked()
            return result

    def store_explicit_memory(
        self,
        *,
        kind: str,
        summary: str,
        details: str | None = None,
    ) -> PersistentMemoryEntry:
        """Store an operator-requested explicit memory in prompt memory."""

        return self.prompt_context_store.memory_store.remember(
            kind=kind,
            summary=summary,
            details=details,
        )

    def delete_explicit_memory(self, *, entry_id: str) -> PersistentMemoryEntry | None:
        """Delete one explicit durable-memory entry by id."""

        return self.prompt_context_store.memory_store.delete(entry_id=entry_id)

    def update_user_profile(
        self,
        *,
        category: str,
        instruction: str,
    ) -> ManagedContextEntry:
        """Upsert one managed user-profile instruction entry."""

        return self.prompt_context_store.user_store.upsert(
            category=category,
            instruction=instruction,
        )

    def remove_user_profile(
        self,
        *,
        category: str,
    ) -> ManagedContextEntry | None:
        """Delete one managed user-profile instruction entry by category."""

        return self.prompt_context_store.user_store.delete(category=category)

    def update_personality(
        self,
        *,
        category: str,
        instruction: str,
    ) -> ManagedContextEntry:
        """Upsert one managed personality instruction entry."""

        return self.prompt_context_store.personality_store.upsert(
            category=category,
            instruction=instruction,
        )

    def remove_personality(
        self,
        *,
        category: str,
    ) -> ManagedContextEntry | None:
        """Delete one managed personality instruction entry by category."""

        return self.prompt_context_store.personality_store.delete(category=category)
