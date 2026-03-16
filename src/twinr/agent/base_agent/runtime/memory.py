"""Mediate structured memory, graph memory, and durable memory mutations."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterable, Mapping
from datetime import datetime
from threading import Lock, RLock
from typing import Any

from twinr.memory import ManagedContextEntry, MemoryLedgerItem, PersistentMemoryEntry, SearchMemoryEntry
from twinr.ops.events import compact_text


LOGGER = logging.getLogger(__name__)
_LOCK_INIT_GUARD = Lock()
_PERSIST_RETRIES = 2
_PERSIST_RETRY_DELAY_S = 0.05
_NON_STORING_GRAPH_STATUSES = frozenset({"needs_clarification", "validation_error", "error", "rejected"})


class TwinrRuntimeMemoryMixin:
    """Provide the runtime-facing memory mutation and flush API."""

    # AUDIT-FIX(#1/#9): Serialize runtime-memory operations to avoid interleaved writes/reads across
    # worker threads while still allowing re-entrant calls such as remember_contact -> remember_note.
    def _memory_runtime_lock(self) -> RLock:
        lock = getattr(self, "_twinr_runtime_memory_lock", None)
        if lock is None:
            with _LOCK_INIT_GUARD:
                lock = getattr(self, "_twinr_runtime_memory_lock", None)
                if lock is None:
                    lock = RLock()
                    setattr(self, "_twinr_runtime_memory_lock", lock)
        return lock

    # AUDIT-FIX(#7): Normalize and reject blank required strings so ASR glitches and empty payloads do
    # not poison structured memory.
    def _normalize_required_text(self, field_name: str, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string")
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{field_name} must not be empty")
        return normalized

    # AUDIT-FIX(#7): Convert blank optional strings to None and keep only valid text values.
    def _normalize_optional_text(self, field_name: str, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string when provided")
        normalized = value.strip()
        return normalized or None

    # AUDIT-FIX(#7): Coerce metadata to a clean str->str mapping for stable downstream serialization.
    def _normalize_metadata(self, metadata: Mapping[str, object] | None) -> dict[str, str] | None:
        if metadata is None:
            return None
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping when provided")

        normalized: dict[str, str] = {}
        for raw_key, raw_value in metadata.items():
            key = self._normalize_required_text("metadata key", str(raw_key))
            if raw_value is None:
                continue
            normalized[key] = str(raw_value)
        return normalized or None

    # AUDIT-FIX(#7): Accept iterable sources defensively and drop blank entries instead of leaking junk.
    def _normalize_sources(self, sources: Iterable[str] | str | None) -> tuple[str, ...]:
        if sources is None:
            return ()
        if isinstance(sources, str):
            source = self._normalize_optional_text("sources", sources)
            return (source,) if source is not None else ()

        try:
            iterator = iter(sources)
        except TypeError as exc:
            raise TypeError("sources must be an iterable of strings") from exc

        normalized: list[str] = []
        for index, item in enumerate(iterator):
            source = self._normalize_optional_text(f"sources[{index}]", item)
            if source is not None:
                normalized.append(source)
        return tuple(normalized)

    # AUDIT-FIX(#8): Reject invalid timeouts early so long-term flush behavior is deterministic.
    def _normalize_timeout(self, timeout_s: float) -> float:
        try:
            timeout = float(timeout_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("timeout_s must be a positive finite float") from exc
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_s must be a positive finite float")
        if self.config.long_term_memory_mode == "remote_primary":
            timeout = max(timeout, float(self.config.long_term_memory_remote_flush_timeout_s))
        return timeout

    # AUDIT-FIX(#6): Prevent timezone-naive datetimes from entering reminder/proactive scheduling paths.
    def _validate_aware_datetime(self, field_name: str, value: Any) -> Any:
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{field_name} must be timezone-aware")
        return value

    # AUDIT-FIX(#5): Ops-event logging is best-effort only and must never turn a successful memory
    # mutation into a user-visible failure.
    def _append_ops_event(
        self,
        *,
        event: str,
        message: str,
        data: Mapping[str, object] | None = None,
    ) -> None:
        ops_events = getattr(self, "ops_events", None)
        if ops_events is None:
            return

        payload = {
            "event": self._normalize_required_text("event", event),
            "message": self._normalize_required_text("message", message),
            "data": dict(data or {}),
        }

        try:
            append = ops_events.append
        except AttributeError:
            LOGGER.warning("ops_events has no append() method; dropping event '%s'", payload["event"])
            return

        try:
            append(
                event=payload["event"],
                message=payload["message"],
                data=payload["data"],
            )
            return
        except TypeError:
            try:
                append(payload)
                return
            except Exception:
                LOGGER.exception("Failed to append ops event '%s'", payload["event"])
                return
        except Exception:
            LOGGER.exception("Failed to append ops event '%s'", payload["event"])
            return

    # AUDIT-FIX(#1): Snapshot persistence is the durability boundary for runtime memory, so retry
    # transient failures briefly and then raise a clear durability error.
    def _persist_snapshot_or_raise(self, *, operation: str) -> None:
        for attempt in range(_PERSIST_RETRIES + 1):
            try:
                self._persist_snapshot()
                return
            except Exception as exc:
                if attempt >= _PERSIST_RETRIES:
                    LOGGER.exception("Failed to persist runtime snapshot after %s", operation)
                    raise RuntimeError(
                        f"Runtime memory changed during {operation}, but the updated snapshot could not be persisted."
                    ) from exc
                time.sleep(_PERSIST_RETRY_DELAY_S * (attempt + 1))

    # AUDIT-FIX(#2): Long-term-memory writes are not durable until flush() confirms persistence.
    def _flush_long_term_memory_strict(self, *, operation: str, timeout_s: float = 2.0) -> None:
        timeout = self._normalize_timeout(timeout_s)
        try:
            flushed = self.long_term_memory.flush(timeout_s=timeout)
        except Exception as exc:
            LOGGER.exception("Failed to flush long-term memory after %s", operation)
            raise RuntimeError(
                f"Long-term memory changed during {operation}, but the update could not be flushed to durable storage."
            ) from exc
        if not flushed:
            raise TimeoutError(
                f"Timed out after {timeout:.2f}s while flushing long-term memory after {operation}."
            )

    # AUDIT-FIX(#4): Downstream graph-memory result shapes vary by status; handle them defensively.
    def _graph_result_status(self, result: Any) -> str | None:
        status = getattr(result, "status", None)
        if status is None:
            return None
        return str(status)

    # AUDIT-FIX(#4): Prefer a stable label when available, but allow note generation to fall back later.
    def _graph_result_label(self, result: Any) -> str | None:
        label = getattr(result, "label", None)
        if label is None:
            return None
        return str(label)

    # AUDIT-FIX(#4): Graph IDs may be UUID/int-like; stringify them before storing in note metadata.
    def _graph_result_node_id(self, result: Any) -> str | None:
        node_id = getattr(result, "node_id", None)
        if node_id is None:
            return None
        return str(node_id)

    # AUDIT-FIX(#4): Edge type is optional on graph results; normalize it before logging/storing.
    def _graph_result_edge_type(self, result: Any) -> str:
        edge_type = getattr(result, "edge_type", None)
        return "" if edge_type is None else str(edge_type)

    # AUDIT-FIX(#4): Only emit "stored" side effects for graph results that are not explicitly
    # non-storing statuses such as clarification requests or validation failures.
    def _should_record_graph_storage(self, result: Any) -> bool:
        status = self._graph_result_status(result)
        return status not in _NON_STORING_GRAPH_STATUSES

    # AUDIT-FIX(#7): Reject nonsensical conflict limits up front.
    def _normalize_limit(self, limit: int | None) -> int | None:
        if limit is None:
            return None
        if not isinstance(limit, int):
            raise TypeError("limit must be an integer when provided")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        return limit

    def remember_search_result(
        self,
        *,
        question: str,
        answer: str,
        sources: tuple[str, ...] = (),
        location_hint: str | None = None,
        date_context: str | None = None,
    ) -> SearchMemoryEntry:
        """Store a verified search result in structured on-device memory."""

        question = self._normalize_required_text("question", question)  # AUDIT-FIX(#7)
        answer = self._normalize_required_text("answer", answer)  # AUDIT-FIX(#7)
        normalized_sources = self._normalize_sources(sources)  # AUDIT-FIX(#7)
        location_hint = self._normalize_optional_text("location_hint", location_hint)  # AUDIT-FIX(#7)
        date_context = self._normalize_optional_text("date_context", date_context)  # AUDIT-FIX(#7)

        with self._memory_runtime_lock():  # AUDIT-FIX(#1/#9)
            entry = self.memory.remember_search(
                question=question,
                answer=answer,
                sources=normalized_sources,
                location_hint=location_hint,
                date_context=date_context,
            )
            self._persist_snapshot_or_raise(operation="remember_search_result")  # AUDIT-FIX(#1)
            self._append_ops_event(  # AUDIT-FIX(#3/#5)
                event="search_result_stored",
                message="Search result stored in structured on-device memory.",
                data={
                    "question_chars": len(question),
                    "answer_chars": len(answer),
                    "sources": len(normalized_sources),
                },
            )
        return entry

    def store_durable_memory(
        self,
        *,
        kind: str,
        summary: str,
        details: str | None = None,
    ) -> PersistentMemoryEntry:
        """Write an explicit durable memory entry and flush it."""

        kind = self._normalize_required_text("kind", kind)  # AUDIT-FIX(#7)
        summary = self._normalize_required_text("summary", summary)  # AUDIT-FIX(#7)
        details = self._normalize_optional_text("details", details)  # AUDIT-FIX(#7)

        with self._memory_runtime_lock():  # AUDIT-FIX(#2/#9)
            entry = self.long_term_memory.store_explicit_memory(
                kind=kind,
                summary=summary,
                details=details,
            )
            self._flush_long_term_memory_strict(operation="store_durable_memory")  # AUDIT-FIX(#2)
        return entry

    def update_user_profile_context(
        self,
        *,
        category: str,
        instruction: str,
    ) -> ManagedContextEntry:
        """Update managed user-profile context and flush it durably."""

        category = self._normalize_required_text("category", category)  # AUDIT-FIX(#7)
        instruction = self._normalize_required_text("instruction", instruction)  # AUDIT-FIX(#7)

        with self._memory_runtime_lock():  # AUDIT-FIX(#2/#9)
            entry = self.long_term_memory.update_user_profile(
                category=category,
                instruction=instruction,
            )
            self._flush_long_term_memory_strict(operation="update_user_profile_context")  # AUDIT-FIX(#2)
        return entry

    def update_personality_context(
        self,
        *,
        category: str,
        instruction: str,
    ) -> ManagedContextEntry:
        """Update managed personality context and flush it durably."""

        category = self._normalize_required_text("category", category)  # AUDIT-FIX(#7)
        instruction = self._normalize_required_text("instruction", instruction)  # AUDIT-FIX(#7)

        with self._memory_runtime_lock():  # AUDIT-FIX(#2/#9)
            entry = self.long_term_memory.update_personality(
                category=category,
                instruction=instruction,
            )
            self._flush_long_term_memory_strict(operation="update_personality_context")  # AUDIT-FIX(#2)
        return entry

    def flush_long_term_memory(self, *, timeout_s: float = 2.0) -> bool:
        """Flush queued long-term memory work within the given timeout."""

        timeout = self._normalize_timeout(timeout_s)  # AUDIT-FIX(#8)
        with self._memory_runtime_lock():  # AUDIT-FIX(#9)
            return self.long_term_memory.flush(timeout_s=timeout)

    def remember_note(
        self,
        *,
        kind: str,
        content: str,
        source: str = "tool",
        metadata: Mapping[str, object] | None = None,
    ) -> MemoryLedgerItem:
        """Store a structured note in on-device runtime memory."""

        kind = self._normalize_required_text("kind", kind)  # AUDIT-FIX(#7)
        content = self._normalize_required_text("content", content)  # AUDIT-FIX(#7)
        source = self._normalize_required_text("source", source)  # AUDIT-FIX(#7)
        normalized_metadata = self._normalize_metadata(metadata)  # AUDIT-FIX(#7)

        with self._memory_runtime_lock():  # AUDIT-FIX(#1/#9)
            item = self.memory.remember_note(
                kind=kind,
                content=content,
                source=source,
                metadata=normalized_metadata,
            )
            self._persist_snapshot_or_raise(operation="remember_note")  # AUDIT-FIX(#1)
            self._append_ops_event(  # AUDIT-FIX(#3/#5)
                event="memory_note_stored",
                message="Structured memory note stored in on-device memory.",
                data={
                    "kind": compact_text(kind),
                    "content_chars": len(content),
                },
            )
        return item

    def remember_contact(
        self,
        *,
        given_name: str,
        family_name: str | None = None,
        phone: str | None = None,
        email: str | None = None,
        role: str | None = None,
        relation: str | None = None,
        notes: str | None = None,
        source: str = "tool",
    ):
        """Store or update a contact in personal graph memory."""

        given_name = self._normalize_required_text("given_name", given_name)  # AUDIT-FIX(#7)
        family_name = self._normalize_optional_text("family_name", family_name)  # AUDIT-FIX(#7)
        phone = self._normalize_optional_text("phone", phone)  # AUDIT-FIX(#7)
        email = self._normalize_optional_text("email", email)  # AUDIT-FIX(#7)
        role = self._normalize_optional_text("role", role)  # AUDIT-FIX(#7)
        relation = self._normalize_optional_text("relation", relation)  # AUDIT-FIX(#7)
        notes = self._normalize_optional_text("notes", notes)  # AUDIT-FIX(#7)
        source = self._normalize_required_text("source", source)  # AUDIT-FIX(#7)

        with self._memory_runtime_lock():  # AUDIT-FIX(#1/#4/#9)
            result = self.graph_memory.remember_contact(
                given_name=given_name,
                family_name=family_name,
                phone=phone,
                email=email,
                role=role,
                relation=relation,
                notes=notes,
            )

            status = self._graph_result_status(result)
            label = self._graph_result_label(result)
            note_stored = False

            if self.config.long_term_memory_mode != "remote_primary" and self._should_record_graph_storage(result):
                note_content = f"Stored contact: {label}" if label else "Stored contact."
                note_metadata = {
                    "graph_status": status or "",
                }
                node_id = self._graph_result_node_id(result)
                if node_id is not None:
                    note_metadata["graph_node_id"] = node_id

                self.memory.remember_note(
                    kind="contact",
                    content=note_content,
                    source=source,
                    metadata=note_metadata,
                )
                note_stored = True

            self._persist_snapshot_or_raise(operation="remember_contact")  # AUDIT-FIX(#1)

            if note_stored:
                self._append_ops_event(  # AUDIT-FIX(#3/#5)
                    event="memory_note_stored",
                    message="Structured memory note stored in on-device memory.",
                    data={
                        "kind": "contact",
                        "content_chars": len(note_content),
                    },
                )
                self._append_ops_event(  # AUDIT-FIX(#3/#5)
                    event="graph_contact_saved",
                    message="Structured contact memory was stored in the personal graph.",
                    data={
                        "status": compact_text(status or "unknown"),
                        "has_label": label is not None,
                    },
                )
        return result

    def lookup_contact(
        self,
        *,
        name: str,
        family_name: str | None = None,
        role: str | None = None,
    ):
        """Look up a stored contact in graph memory."""

        name = self._normalize_required_text("name", name)  # AUDIT-FIX(#7)
        family_name = self._normalize_optional_text("family_name", family_name)  # AUDIT-FIX(#7)
        role = self._normalize_optional_text("role", role)  # AUDIT-FIX(#7)

        with self._memory_runtime_lock():  # AUDIT-FIX(#9)
            return self.graph_memory.lookup_contact(name=name, family_name=family_name, role=role)

    def remember_preference(
        self,
        *,
        category: str,
        value: str,
        for_product: str | None = None,
        sentiment: str = "prefer",
        details: str | None = None,
        source: str = "tool",
    ):
        """Store or update a preference in personal graph memory."""

        category = self._normalize_required_text("category", category)  # AUDIT-FIX(#7)
        value = self._normalize_required_text("value", value)  # AUDIT-FIX(#7)
        for_product = self._normalize_optional_text("for_product", for_product)  # AUDIT-FIX(#7)
        sentiment = self._normalize_required_text("sentiment", sentiment)  # AUDIT-FIX(#7)
        details = self._normalize_optional_text("details", details)  # AUDIT-FIX(#7)
        source = self._normalize_required_text("source", source)  # AUDIT-FIX(#7)

        with self._memory_runtime_lock():  # AUDIT-FIX(#1/#4/#9)
            result = self.graph_memory.remember_preference(
                category=category,
                value=value,
                for_product=for_product,
                sentiment=sentiment,
                details=details,
            )

            status = self._graph_result_status(result)
            label = self._graph_result_label(result)
            edge_type = self._graph_result_edge_type(result)
            note_stored = False

            if self.config.long_term_memory_mode != "remote_primary" and self._should_record_graph_storage(result):
                note_content = f"Stored preference: {label}" if label else "Stored preference."
                note_metadata = {
                    "graph_edge_type": edge_type,
                    "graph_status": status or "",
                }
                node_id = self._graph_result_node_id(result)
                if node_id is not None:
                    note_metadata["graph_node_id"] = node_id

                self.memory.remember_note(
                    kind="preference",
                    content=note_content,
                    source=source,
                    metadata=note_metadata,
                )
                note_stored = True

            self._persist_snapshot_or_raise(operation="remember_preference")  # AUDIT-FIX(#1)

            if note_stored:
                self._append_ops_event(  # AUDIT-FIX(#3/#5)
                    event="memory_note_stored",
                    message="Structured memory note stored in on-device memory.",
                    data={
                        "kind": "preference",
                        "content_chars": len(note_content),
                    },
                )
                self._append_ops_event(  # AUDIT-FIX(#3/#5)
                    event="graph_preference_saved",
                    message="Structured preference memory was stored in the personal graph.",
                    data={
                        "status": compact_text(status or "unknown"),
                        "edge_type": compact_text(edge_type),
                    },
                )
        return result

    def remember_plan(
        self,
        *,
        summary: str,
        when_text: str | None = None,
        details: str | None = None,
        source: str = "tool",
    ):
        """Store or update a future plan in personal graph memory."""

        summary = self._normalize_required_text("summary", summary)  # AUDIT-FIX(#7)
        when_text = self._normalize_optional_text("when_text", when_text)  # AUDIT-FIX(#7)
        details = self._normalize_optional_text("details", details)  # AUDIT-FIX(#7)
        source = self._normalize_required_text("source", source)  # AUDIT-FIX(#7)

        with self._memory_runtime_lock():  # AUDIT-FIX(#1/#4/#9)
            result = self.graph_memory.remember_plan(summary=summary, when_text=when_text, details=details)

            status = self._graph_result_status(result)
            label = self._graph_result_label(result)
            edge_type = self._graph_result_edge_type(result)
            note_stored = False

            if self.config.long_term_memory_mode != "remote_primary" and self._should_record_graph_storage(result):
                note_content = f"Stored plan: {label}" if label else "Stored plan."
                note_metadata = {
                    "graph_edge_type": edge_type,
                    "graph_status": status or "",
                }
                node_id = self._graph_result_node_id(result)
                if node_id is not None:
                    note_metadata["graph_node_id"] = node_id

                self.memory.remember_note(
                    kind="plan",
                    content=note_content,
                    source=source,
                    metadata=note_metadata,
                )
                note_stored = True

            self._persist_snapshot_or_raise(operation="remember_plan")  # AUDIT-FIX(#1)

            if note_stored:
                self._append_ops_event(  # AUDIT-FIX(#3/#5)
                    event="memory_note_stored",
                    message="Structured memory note stored in on-device memory.",
                    data={
                        "kind": "plan",
                        "content_chars": len(note_content),
                    },
                )
                self._append_ops_event(  # AUDIT-FIX(#3/#5)
                    event="graph_plan_saved",
                    message="Structured plan memory was stored in the personal graph.",
                    data={
                        "status": compact_text(status or "unknown"),
                        "edge_type": compact_text(edge_type),
                    },
                )
        return result

    def select_long_term_memory_conflicts(
        self,
        *,
        query_text: str | None = None,
        limit: int | None = None,
    ):
        """Return pending long-term memory conflicts for review."""

        query_text = self._normalize_optional_text("query_text", query_text)  # AUDIT-FIX(#7)
        limit = self._normalize_limit(limit)  # AUDIT-FIX(#7)

        with self._memory_runtime_lock():  # AUDIT-FIX(#9)
            return self.long_term_memory.select_conflict_queue(
                query_text=query_text,
                limit=limit,
            )

    def resolve_long_term_memory_conflict(
        self,
        *,
        slot_key: str,
        selected_memory_id: str,
    ):
        """Resolve a queued long-term memory conflict and flush it."""

        slot_key = self._normalize_required_text("slot_key", slot_key)  # AUDIT-FIX(#7)
        selected_memory_id = self._normalize_required_text("selected_memory_id", selected_memory_id)  # AUDIT-FIX(#7)

        with self._memory_runtime_lock():  # AUDIT-FIX(#2/#9)
            result = self.long_term_memory.resolve_conflict(
                slot_key=slot_key,
                selected_memory_id=selected_memory_id,
            )
            self._flush_long_term_memory_strict(operation="resolve_long_term_memory_conflict")  # AUDIT-FIX(#2)
        return result

    def reserve_long_term_proactive_candidate(self, *, now=None, live_facts=None):
        """Reserve the next proactive long-term memory candidate, if any."""

        now = self._validate_aware_datetime("now", now)  # AUDIT-FIX(#6)
        with self._memory_runtime_lock():  # AUDIT-FIX(#2/#9)
            result = self.long_term_memory.reserve_proactive_candidate(now=now, live_facts=live_facts)
            if result is not None:
                self._flush_long_term_memory_strict(operation="reserve_long_term_proactive_candidate")  # AUDIT-FIX(#2)
        return result

    def preview_long_term_proactive_candidate(self, *, now=None, live_facts=None):
        """Preview the next proactive long-term memory candidate without reserving it."""

        now = self._validate_aware_datetime("now", now)  # AUDIT-FIX(#6)
        with self._memory_runtime_lock():  # AUDIT-FIX(#9)
            return self.long_term_memory.preview_proactive_candidate(now=now, live_facts=live_facts)

    def reserve_specific_long_term_proactive_candidate(self, candidate, *, now=None):
        """Reserve a specific proactive long-term memory candidate."""

        now = self._validate_aware_datetime("now", now)  # AUDIT-FIX(#6)
        with self._memory_runtime_lock():  # AUDIT-FIX(#2/#9)
            result = self.long_term_memory.reserve_specific_proactive_candidate(candidate, now=now)
            if result is not None:
                self._flush_long_term_memory_strict(
                    operation="reserve_specific_long_term_proactive_candidate"
                )  # AUDIT-FIX(#2)
        return result

    def mark_long_term_proactive_candidate_delivered(
        self,
        reservation,
        *,
        delivered_at=None,
        prompt_text: str | None = None,
    ):
        """Mark a reserved proactive candidate as delivered and flush it."""

        delivered_at = self._validate_aware_datetime("delivered_at", delivered_at)  # AUDIT-FIX(#6)
        prompt_text = self._normalize_optional_text("prompt_text", prompt_text)  # AUDIT-FIX(#7)

        with self._memory_runtime_lock():  # AUDIT-FIX(#2/#9)
            result = self.long_term_memory.mark_proactive_candidate_delivered(
                reservation,
                delivered_at=delivered_at,
                prompt_text=prompt_text,
            )
            self._flush_long_term_memory_strict(
                operation="mark_long_term_proactive_candidate_delivered"
            )  # AUDIT-FIX(#2)
        return result

    def mark_long_term_proactive_candidate_skipped(
        self,
        reservation,
        *,
        reason: str,
        skipped_at=None,
    ):
        """Mark a reserved proactive candidate as skipped and flush it."""

        reason = self._normalize_required_text("reason", reason)  # AUDIT-FIX(#7)
        skipped_at = self._validate_aware_datetime("skipped_at", skipped_at)  # AUDIT-FIX(#6)

        with self._memory_runtime_lock():  # AUDIT-FIX(#2/#9)
            result = self.long_term_memory.mark_proactive_candidate_skipped(
                reservation,
                reason=reason,
                skipped_at=skipped_at,
            )
            self._flush_long_term_memory_strict(
                operation="mark_long_term_proactive_candidate_skipped"
            )  # AUDIT-FIX(#2)
        return result
