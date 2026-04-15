"""Shared payload parsing, merge helpers, and query text/ranking logic."""


from __future__ import annotations

from collections.abc import Iterable, Mapping

from twinr.memory.fulltext import FullTextDocument, FullTextSelector
from twinr.memory.longterm.core.models import LongTermMemoryConflictV1, LongTermMemoryObjectV1, LongTermMemoryReviewItemV1
from twinr.memory.longterm.reasoning.conversation_recall import (
    conversation_episode_recall_hints,
    conversation_recap_query_variants,
    conversation_recap_specific_terms,
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

    def _candidate_window_limit(self, *, limit: int, minimum: int = 12) -> int:
        """Return a bounded pre-ranking candidate window for noisy mixed corpora."""

        bounded_limit = max(1, int(limit))
        return max(int(minimum), bounded_limit * 4)

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
        episode_text = self._episode_search_text(item)
        pattern_aliases = self._pattern_search_text(item)
        parts = [
            item.kind,
            item.summary,
            episode_text or (item.details or ""),
            pattern_aliases,
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

    def _episode_search_text(self, item: LongTermMemoryObjectV1) -> str:
        """Return one content-bearing retrieval text for true conversation episodes."""

        attributes = item.attributes if isinstance(item.attributes, Mapping) else None
        if not is_conversation_episode_object(kind=item.kind, attributes=attributes):
            return ""
        parts: list[str] = []
        if isinstance(item.summary, str):
            parts.append(item.summary)
        if isinstance(attributes, Mapping):
            for key in ("raw_transcript", "raw_response", "topic", "subject"):
                value = attributes.get(key)
                if isinstance(value, str):
                    parts.append(value)
        if not parts and isinstance(item.details, str):
            parts.append(item.details)
        return _normalize_text(" ".join(part for part in parts if part))

    def _pattern_search_text(self, item: LongTermMemoryObjectV1) -> str:
        """Return bounded implicit aliases for device-local multimodal patterns."""

        attributes = item.attributes if isinstance(item.attributes, Mapping) else None
        if not isinstance(attributes, Mapping):
            return ""
        kind = _normalize_text(item.kind).lower()
        pattern_type = _normalize_text(attributes.get("pattern_type")).lower()
        memory_domain = _normalize_text(attributes.get("memory_domain")).lower()
        routine_type = _normalize_text(attributes.get("routine_type")).lower()
        interaction_type = _normalize_text(attributes.get("interaction_type")).lower()
        button = _normalize_text(attributes.get("button_label") or attributes.get("button")).lower()
        action = _normalize_text(attributes.get("action")).lower()
        request_source = _normalize_text(
            attributes.get("request_source_label") or attributes.get("request_source")
        ).lower()
        purpose = _normalize_text(attributes.get("purpose_label") or attributes.get("purpose")).lower()
        daypart = _normalize_text(attributes.get("daypart")).lower()
        if kind not in {"pattern", "summary"} and not pattern_type and not interaction_type and not routine_type:
            return ""

        aliases: list[str] = []
        if pattern_type or interaction_type or memory_domain in {"interaction", "sensor_routine"}:
            aliases.extend(("twinr", "device"))
        if pattern_type == "presence":
            aliases.extend(("presence near twinr", "near twinr", "near the device"))
        if button:
            aliases.extend((button, f"{button} button"))
        if action == "start_listening" or interaction_type in {"conversation_start", "conversation_start_audio"}:
            aliases.extend(("start conversation", "start a conversation", "conversation with twinr"))
        if action == "print_request" or interaction_type == "print":
            aliases.extend(("print request", "printed answer", "printed twinr output"))
        if interaction_type in {"camera_use", "camera_showing"}:
            aliases.extend(("camera", "camera use", "camera interaction"))
        if request_source:
            aliases.append(request_source)
        if purpose:
            aliases.extend((purpose, purpose.replace("_", " ")))
        if memory_domain == "sensor_routine":
            aliases.extend(("routine", "usual routine", "typical routine"))
            if routine_type == "presence":
                aliases.extend(("usually near the device", "usual presence near twinr"))
            if interaction_type == "conversation_start":
                aliases.extend(("usually start a conversation", "habitual conversation start"))
            if interaction_type == "print":
                aliases.extend(("usually print answers", "habitual printing"))
            if interaction_type in {"camera_use", "camera_showing"}:
                aliases.extend(("usually use the camera", "habitual camera interaction"))
        if daypart:
            aliases.append(daypart)
        deduped = tuple(dict.fromkeys(alias for alias in aliases if alias))
        return _normalize_text(" ".join(deduped))

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
        """Return topic-bearing terms after removing state-only and noisy numeric cues.

        Queries that mix words and numbers should not recall episodic objects on
        number overlap alone. Otherwise distractor memories such as "topic 14"
        or "topic 27" leak into unrelated math/time-conversion questions that
        merely happen to contain those digits.
        """

        normalized = {
            str(term).strip()
            for term in query_terms
            if isinstance(term, str) and str(term).strip()
        }
        semantic_terms = {
            term
            for term in normalized
            if term not in _OBJECT_STATE_QUERY_TERMS
        }
        if not semantic_terms:
            return set()
        informative_alpha_terms = {
            term
            for term in semantic_terms
            if not term.isdigit() and len(term) >= 4
        }
        if informative_alpha_terms:
            return informative_alpha_terms
        alpha_terms = {
            term
            for term in semantic_terms
            if not term.isdigit()
        }
        if alpha_terms:
            return alpha_terms
        return self._query_match_terms(semantic_terms)

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
                shorter_term = query_term if len(query_term) <= len(document_term) else document_term
                if len(shorter_term) < 4:
                    continue
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

    def _object_focus_attribute_search_text(self, item: LongTermMemoryObjectV1) -> str:
        """Return query-relevant structured labels that signal object specificity."""

        attributes = item.attributes if isinstance(item.attributes, Mapping) else None
        parts: list[str] = []
        if isinstance(attributes, Mapping):
            for key in (
                "memory_domain",
                "routine_type",
                "interaction_type",
                "pattern_type",
                "button",
                "button_label",
                "action",
                "action_label",
                "request_source",
                "request_source_label",
                "purpose",
                "purpose_label",
                "daypart",
            ):
                value = attributes.get(key)
                if isinstance(value, str):
                    parts.append(value)
                    parts.append(value.replace("_", " "))
        if item.kind in {"pattern", "summary"}:
            parts.append(item.memory_id.replace(":", " ").replace("_", " "))
        return _normalize_text(" ".join(part for part in parts if part))

    def _object_focus_overlap_score(
        self,
        *,
        item: LongTermMemoryObjectV1,
        query_terms: set[str],
    ) -> int:
        """Return overlap against structured multimodal attributes."""

        match_terms = self._query_match_terms(query_terms)
        if not match_terms:
            return 0
        focus_terms = set(retrieval_terms(self._object_focus_attribute_search_text(item)))
        if not focus_terms:
            return 0
        return len(match_terms.intersection(focus_terms))

    def _object_specificity_priority(self, item: LongTermMemoryObjectV1) -> int:
        """Prefer concrete multimodal patterns over generic aggregated routines."""

        attributes = item.attributes if isinstance(item.attributes, Mapping) else None
        if not isinstance(attributes, Mapping):
            return 0
        memory_domain = _normalize_text(attributes.get("memory_domain")).lower()
        if memory_domain == "sensor_routine":
            return 0
        if item.kind not in {"pattern", "summary"}:
            return 0
        for key in (
            "pattern_type",
            "interaction_type",
            "button",
            "button_label",
            "action",
            "request_source",
            "purpose",
        ):
            if _normalize_text(attributes.get(key)).strip():
                return 2
        return 1

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
        focus_overlap = self._object_focus_overlap_score(item=item, query_terms=query_terms)
        specificity_priority = self._object_specificity_priority(item) if focus_overlap > 0 else 0
        return (
            focus_overlap,
            specificity_priority,
            self._object_query_overlap_score(item=item, query_terms=query_terms),
            1 if item.confirmed_by_user else 0,
            self._object_status_priority(item.status),
            -original_index,
            _coerce_aware_utc(item.updated_at),
            _coerce_aware_utc(item.created_at),
            item.confidence,
        )

    def _query_variants_for_loaded_objects(
        self,
        *,
        query_text: str,
        include_episodes: bool,
    ) -> tuple[str, ...]:
        """Return bounded query variants for one in-memory selector pass."""

        clean_query = _normalize_text(query_text)
        if not clean_query:
            return ()
        if include_episodes:
            return conversation_recap_query_variants(clean_query)
        return (clean_query,)

    def _filter_query_relevant_objects(
        self,
        query_text: str,
        *,
        selected: list[LongTermMemoryObjectV1],
        limit: int,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        if query_has_conversation_recap_semantics(query_text) and not conversation_recap_specific_terms(query_text):
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
        raw_query_terms = retrieval_terms(query_text)
        query_terms = self._query_match_terms(raw_query_terms)
        if not query_terms:
            return tuple(selected[: max(1, limit)])
        semantic_query_terms = self._semantic_query_terms(raw_query_terms)
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
        fallback_query_terms = semantic_query_terms or query_terms
        filtered = [
            item
            for item in selected
            if self._has_query_overlap(
                query_terms=fallback_query_terms,
                document_terms=retrieval_terms(self._object_search_text(item)),
            )
        ]
        return tuple(filtered[: max(1, limit)])

    def _object_semantic_search_text(self, item: LongTermMemoryObjectV1) -> str:
        """Return one retrieval text stripped of synthetic state-only terms."""

        episode_text = self._episode_search_text(item)
        if episode_text:
            return episode_text
        return _normalize_text(
            " ".join(
                part
                for part in (item.summary, item.details or "", self._pattern_search_text(item))
                if part
            )
        )

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
