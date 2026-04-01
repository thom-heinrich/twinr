"""Shared payload parsing, merge helpers, and query text/ranking logic."""

# mypy: disable-error-code=attr-defined

from __future__ import annotations

from collections.abc import Iterable, Mapping

from twinr.memory.fulltext import FullTextDocument, FullTextSelector
from twinr.memory.longterm.core.models import LongTermMemoryConflictV1, LongTermMemoryObjectV1, LongTermMemoryReviewItemV1
from twinr.memory.longterm.reasoning.conversation_recall import (
    conversation_episode_recall_hints,
    is_conversation_episode_object,
    query_has_conversation_recap_semantics,
)
from twinr.text_utils import retrieval_terms

from .shared import (
    _LOG,
    _NON_SEMANTIC_ATTRIBUTE_KEYS,
    _OBJECT_STATE_QUERY_TERMS,
    _coerce_aware_utc,
    _coerce_positive_int,
    _normalize_text,
    _utcnow,
)


class StructuredStoreQueryMixin:
    """Own shared parsing, merge, ranking, and query-overlap helpers."""

    def _load_memory_objects_from_payload(
        self,
        payload: dict[str, object] | None,
        *,
        snapshot_kind: str,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        if payload is None:
            return ()
        items = payload.get("objects", [])
        if not isinstance(items, list):
            return ()
        objects: list[LongTermMemoryObjectV1] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                objects.append(LongTermMemoryObjectV1.from_payload(item))
            except Exception:
                _LOG.warning("Skipping invalid long-term memory object in %s snapshot.", snapshot_kind, exc_info=True)
        return tuple(objects)

    def _merge_object(
        self,
        *,
        existing: LongTermMemoryObjectV1 | None,
        incoming: LongTermMemoryObjectV1,
        increment_support: bool,
    ) -> LongTermMemoryObjectV1:
        if existing is None:
            attrs = dict(incoming.attributes or {})
            attrs["support_count"] = _coerce_positive_int(attrs.get("support_count", 1), default=1)
            return incoming.with_updates(attributes=attrs)
        attrs = dict(existing.attributes or {})
        incoming_attrs = dict(incoming.attributes or {})
        existing_support = _coerce_positive_int(attrs.get("support_count", 1), default=1)
        incoming_support = _coerce_positive_int(incoming_attrs.get("support_count", 1), default=1)
        support_count = existing_support + 1 if increment_support else max(existing_support, incoming_support)
        attrs.update(incoming_attrs)
        attrs["support_count"] = support_count

        existing_source_payload = (
            existing.source.to_payload()
            if getattr(existing, "source", None) is not None and hasattr(existing.source, "to_payload")
            else {}
        )
        incoming_source_payload = (
            incoming.source.to_payload()
            if getattr(incoming, "source", None) is not None and hasattr(incoming.source, "to_payload")
            else {}
        )
        raw_existing_event_ids = existing_source_payload.get("event_ids")
        existing_event_ids = tuple(
            value
            for value in raw_existing_event_ids
            if isinstance(value, str) and value
        ) if isinstance(raw_existing_event_ids, (list, tuple)) else ()
        raw_incoming_event_ids = incoming_source_payload.get("event_ids")
        incoming_event_ids = tuple(
            value
            for value in raw_incoming_event_ids
            if isinstance(value, str) and value
        ) if isinstance(raw_incoming_event_ids, (list, tuple)) else ()
        merged_event_ids = tuple(dict.fromkeys((*existing_event_ids, *incoming_event_ids)))
        merged_source = dict(existing_source_payload)
        merged_source.update(incoming_source_payload)
        merged_source["event_ids"] = list(merged_event_ids)

        return incoming.with_updates(
            source=merged_source,
            attributes=attrs,
            created_at=min(_coerce_aware_utc(existing.created_at), _coerce_aware_utc(incoming.created_at)),
            updated_at=max(_coerce_aware_utc(existing.updated_at), _coerce_aware_utc(incoming.updated_at)),
            confidence=max(existing.confidence, incoming.confidence),
            status=self._preferred_status(existing.status, incoming.status),
        )

    def _preferred_status(self, existing: str, incoming: str) -> str:
        rank = {
            "active": 5,
            "candidate": 4,
            "uncertain": 3,
            "superseded": 2,
            "expired": 1,
            "invalid": 0,
        }
        return existing if rank.get(existing, -1) >= rank.get(incoming, -1) else incoming

    def _rewrite_conflicts_without_memory(
        self,
        memory_id: str,
    ) -> tuple[LongTermMemoryConflictV1, ...]:
        rewritten: list[LongTermMemoryConflictV1] = []
        for conflict in self.load_conflicts_for_memory_ids((memory_id,)):
            if conflict.candidate_memory_id == memory_id:
                continue
            existing_ids = tuple(value for value in conflict.existing_memory_ids if value != memory_id)
            if not existing_ids:
                continue
            rewritten.append(
                LongTermMemoryConflictV1(
                    slot_key=conflict.slot_key,
                    candidate_memory_id=conflict.candidate_memory_id,
                    existing_memory_ids=existing_ids,
                    question=conflict.question,
                    reason=conflict.reason,
                )
            )
        return tuple(rewritten)

    def _conflict_delta_without_memory(
        self,
        memory_id: str,
    ) -> tuple[tuple[LongTermMemoryConflictV1, ...], tuple[str, ...]]:
        """Return rewritten conflicts plus the touched slot keys for one memory.

        Mutation writers need a delta contract, not a full conflict snapshot.
        This helper keeps the mutation builder selective by returning only the
        rewritten conflict rows plus the slot keys that must be deleted or
        replaced in storage.
        """

        conflicts = self.load_conflicts_for_memory_ids((memory_id,))
        rewritten: list[LongTermMemoryConflictV1] = []
        deleted_slot_keys: list[str] = []
        for conflict in conflicts:
            deleted_slot_keys.append(conflict.slot_key)
            if conflict.candidate_memory_id == memory_id:
                continue
            existing_ids = tuple(value for value in conflict.existing_memory_ids if value != memory_id)
            if not existing_ids:
                continue
            rewritten.append(
                LongTermMemoryConflictV1(
                    slot_key=conflict.slot_key,
                    candidate_memory_id=conflict.candidate_memory_id,
                    existing_memory_ids=existing_ids,
                    question=conflict.question,
                    reason=conflict.reason,
                )
            )
        return tuple(rewritten), tuple(dict.fromkeys(deleted_slot_keys))

    def _cleanup_references_after_mutation(
        self,
        *,
        target_memory_id: str,
        drop_supersedes: bool,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        updated_objects: list[LongTermMemoryObjectV1] = []
        update_time = _utcnow()
        for item in self.load_objects_referencing_memory_ids((target_memory_id,)):
            if item.memory_id == target_memory_id:
                continue
            conflicts_with = tuple(value for value in item.conflicts_with if value != target_memory_id)
            supersedes = (
                tuple(value for value in item.supersedes if value != target_memory_id)
                if drop_supersedes
                else item.supersedes
            )
            if conflicts_with == item.conflicts_with and supersedes == item.supersedes:
                continue
            updated_objects.append(
                item.with_updates(
                    conflicts_with=conflicts_with,
                    supersedes=supersedes,
                    updated_at=update_time,
                )
            )
        return tuple(updated_objects)

    def _to_review_item(self, item: LongTermMemoryObjectV1) -> LongTermMemoryReviewItemV1:
        return LongTermMemoryReviewItemV1(
            memory_id=item.memory_id,
            kind=item.kind,
            summary=item.summary,
            details=item.details,
            status=item.status,
            confidence=item.confidence,
            updated_at=item.updated_at,
            confirmed_by_user=item.confirmed_by_user,
            sensitivity=item.sensitivity,
            slot_key=item.slot_key,
            value_key=item.value_key,
        )

    def _object_selector(
        self,
        objects: tuple[LongTermMemoryObjectV1, ...] | list[LongTermMemoryObjectV1],
    ) -> FullTextSelector:
        return FullTextSelector(
            tuple(
                FullTextDocument(
                    doc_id=item.memory_id,
                    category="object",
                    content=self._object_search_text(item),
                )
                for item in objects
            )
        )

    def rank_selected_objects(
        self,
        *,
        query_texts: Iterable[str],
        objects: Iterable[LongTermMemoryObjectV1],
        limit: int | None = None,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Rank selected objects by query overlap, confirmation state, and recency."""

        unique_objects: list[LongTermMemoryObjectV1] = []
        seen_memory_ids: set[str] = set()
        for item in objects:
            if item.memory_id in seen_memory_ids:
                continue
            seen_memory_ids.add(item.memory_id)
            unique_objects.append(item)
        if not unique_objects:
            return ()
        bounded_limit = max(1, limit) if isinstance(limit, int) else len(unique_objects)
        query_terms = self._combined_query_terms(query_texts)
        ranked = sorted(
            enumerate(unique_objects),
            key=lambda pair: self._object_query_sort_key(
                item=pair[1],
                query_terms=query_terms,
                original_index=pair[0],
            ),
            reverse=True,
        )
        return tuple(item for _index, item in ranked[:bounded_limit])

    def _object_search_text(self, item: LongTermMemoryObjectV1) -> str:
        parts = [
            item.kind,
            item.summary,
            item.details or "",
            f"status {item.status}",
            self._object_state_search_text(item),
        ]
        for key, value in (item.attributes or {}).items():
            if key in _NON_SEMANTIC_ATTRIBUTE_KEYS:
                continue
            parts.append(key)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, bool):
                parts.append("true" if value else "false")
            elif isinstance(value, (list, tuple)):
                parts.extend(str(entry) for entry in value if isinstance(entry, str))
        parts.extend(
            conversation_episode_recall_hints(
                kind=item.kind,
                attributes=item.attributes if isinstance(item.attributes, Mapping) else None,
            )
        )
        return _normalize_text(" ".join(part for part in parts if part))

    def _object_state_search_text(self, item: LongTermMemoryObjectV1) -> str:
        parts = [item.status]
        if item.status == "active":
            parts.extend(("current", "stored", "available", "aktuell", "gespeichert"))
        elif item.status == "superseded":
            parts.extend(("previous", "former", "superseded", "frueher", "vorher"))
        elif item.status in {"candidate", "uncertain"}:
            parts.extend(("pending", "unconfirmed", "candidate", "unbestaetigt", "unklar"))
        elif item.status == "invalid":
            parts.extend(("invalid", "discarded"))
        elif item.status == "expired":
            parts.extend(("expired", "outdated"))
        if item.confirmed_by_user:
            parts.extend(("confirmed_by_user", "confirmed", "user_confirmed", "bestaetigt"))
        return _normalize_text(" ".join(parts))

    def _combined_query_terms(self, query_texts: Iterable[str]) -> set[str]:
        query_terms: set[str] = set()
        for query_text in query_texts:
            if not isinstance(query_text, str):
                continue
            query_terms.update(retrieval_terms(query_text))
        return query_terms

    def _query_match_terms(self, query_terms: Iterable[str]) -> set[str]:
        """Prefer content-bearing query terms over auxiliary-word-only overlap."""

        normalized = {
            str(term).strip()
            for term in query_terms
            if isinstance(term, str) and str(term).strip()
        }
        if not normalized:
            return set()
        informative = {
            term
            for term in normalized
            if term.isdigit() or len(term) >= 4
        }
        return informative or normalized

    def _semantic_query_terms(self, query_terms: Iterable[str]) -> set[str]:
        """Return topic-bearing terms after removing memory-state-only vocabulary."""

        return {
            term
            for term in self._query_match_terms(query_terms)
            if term not in _OBJECT_STATE_QUERY_TERMS
        }

    def _has_query_overlap(
        self,
        *,
        query_terms: Iterable[str],
        document_terms: Iterable[str],
    ) -> bool:
        """Return whether document terms overlap one query through exact or compound matches."""

        informative_query_terms = self._query_match_terms(query_terms)
        informative_document_terms = self._query_match_terms(document_terms)
        if not informative_query_terms or not informative_document_terms:
            return False
        if informative_query_terms.intersection(informative_document_terms):
            return True
        for query_term in informative_query_terms:
            for document_term in informative_document_terms:
                if query_term in document_term or document_term in query_term:
                    return True
        return False

    def _object_query_overlap_score(
        self,
        *,
        item: LongTermMemoryObjectV1,
        query_terms: set[str],
    ) -> int:
        match_terms = self._query_match_terms(query_terms)
        if not match_terms:
            return 0
        object_terms = set(retrieval_terms(self._object_search_text(item)))
        return len(match_terms.intersection(object_terms))

    def _object_status_priority(self, status: str) -> int:
        if status == "active":
            return 4
        if status == "candidate":
            return 3
        if status == "uncertain":
            return 2
        if status == "superseded":
            return 1
        return 0

    def _object_query_sort_key(
        self,
        *,
        item: LongTermMemoryObjectV1,
        query_terms: set[str],
        original_index: int,
    ) -> tuple[object, ...]:
        return (
            self._object_query_overlap_score(item=item, query_terms=query_terms),
            1 if item.confirmed_by_user else 0,
            self._object_status_priority(item.status),
            _coerce_aware_utc(item.updated_at),
            _coerce_aware_utc(item.created_at),
            item.confidence,
            -original_index,
        )

    def _filter_query_relevant_objects(
        self,
        query_text: str,
        *,
        selected: list[LongTermMemoryObjectV1],
        limit: int,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        if query_has_conversation_recap_semantics(query_text):
            conversation_matches = [
                item
                for item in selected
                if is_conversation_episode_object(
                    kind=item.kind,
                    attributes=item.attributes if isinstance(item.attributes, Mapping) else None,
                )
            ]
            if conversation_matches:
                return tuple(conversation_matches[: max(1, limit)])
        query_terms = self._query_match_terms(retrieval_terms(query_text))
        if not query_terms:
            return tuple(selected[: max(1, limit)])
        semantic_query_terms = self._semantic_query_terms(query_terms)
        topic_terms_by_id = {
            item.memory_id: tuple(retrieval_terms(self._object_semantic_search_text(item)))
            for item in selected
        }
        if semantic_query_terms:
            semantic_matches = [
                item
                for item in selected
                if self._has_query_overlap(
                    query_terms=semantic_query_terms,
                    document_terms=topic_terms_by_id.get(item.memory_id, ()),
                )
            ]
            if semantic_matches:
                anchor_terms: list[str] = []
                for item in semantic_matches:
                    anchor_terms.extend(topic_terms_by_id.get(item.memory_id, ()))
                expanded_matches = [
                    item
                    for item in selected
                    if self._has_query_overlap(
                        query_terms=anchor_terms,
                        document_terms=topic_terms_by_id.get(item.memory_id, ()),
                    )
                ]
                if expanded_matches:
                    return tuple(expanded_matches[: max(1, limit)])
                return tuple(semantic_matches[: max(1, limit)])
        filtered = [
            item
            for item in selected
            if self._has_query_overlap(
                query_terms=query_terms,
                document_terms=retrieval_terms(self._object_search_text(item)),
            )
        ]
        return tuple(filtered[: max(1, limit)])

    def _object_semantic_search_text(self, item: LongTermMemoryObjectV1) -> str:
        """Return one retrieval text stripped of synthetic state-only terms."""

        return _normalize_text(" ".join(part for part in (item.summary, item.details or "") if part))

    def _filter_query_relevant_conflicts(
        self,
        query_text: str,
        *,
        selected: Iterable[LongTermMemoryConflictV1],
        limit: int,
        objects_by_id: Mapping[str, LongTermMemoryObjectV1] | None = None,
    ) -> tuple[LongTermMemoryConflictV1, ...]:
        query_terms = self._query_match_terms(retrieval_terms(query_text))
        selected_conflicts = list(selected)
        if not query_terms:
            return tuple(selected_conflicts[: max(1, limit)])
        related_objects = objects_by_id or {}
        filtered = [
            conflict
            for conflict in selected_conflicts
            if query_terms.intersection(
                retrieval_terms(
                    self._conflict_search_text(
                        conflict,
                        objects_by_id=related_objects,
                    )
                )
            )
        ]
        return tuple(filtered[: max(1, limit)])

    def _conflict_search_text(
        self,
        conflict: LongTermMemoryConflictV1,
        *,
        objects_by_id: Mapping[str, LongTermMemoryObjectV1],
    ) -> str:
        related_parts = [
            conflict.slot_key,
            conflict.question,
            conflict.reason,
        ]
        candidate = objects_by_id.get(conflict.candidate_memory_id)
        if candidate is not None:
            related_parts.append(self._object_search_text(candidate))
        for memory_id in conflict.existing_memory_ids:
            existing = objects_by_id.get(memory_id)
            if existing is not None:
                related_parts.append(self._object_search_text(existing))
        return _normalize_text(" ".join(part for part in related_parts if part))
