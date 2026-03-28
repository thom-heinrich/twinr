"""Commit and mutation helpers for the discovery service."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime

from .common import _compact_text, _managed_context_category, _sentence_list, _utc_now
from .models import (
    UserDiscoveryCommitCallbacks,
    UserDiscoveryMemoryRoute,
    UserDiscoveryState,
    UserDiscoveryStoredFact,
    UserDiscoveryTopicState,
    _active_managed_fact_texts,
)


class UserDiscoveryCommitMixin:
    """Own durable fact commit and delete helpers."""

    def _apply_memory_routes(
        self,
        topic_id: str,
        topic_state: UserDiscoveryTopicState,
        *,
        memory_routes: Sequence[UserDiscoveryMemoryRoute],
        callbacks: UserDiscoveryCommitCallbacks | None,
        now: datetime,
    ) -> tuple[UserDiscoveryTopicState, int, tuple[str, ...]]:
        if not memory_routes:
            return topic_state, 0, ()

        updated_facts = list(topic_state.stored_facts)
        existing_keys = {fact.dedupe_key() for fact in topic_state.active_facts}
        added_fact_ids: list[str] = []
        saved_targets: list[str] = []

        for route in memory_routes:
            if route.is_empty():
                continue
            dedupe_key = route.dedupe_key()
            if dedupe_key in existing_keys:
                continue
            stored_fact = UserDiscoveryStoredFact.from_route(route, topic_id=topic_id, now=now)
            updated_facts.append(stored_fact)
            added_fact_ids.append(stored_fact.fact_id)
            existing_keys.add(dedupe_key)
            saved_targets.append(route.route_kind)

        if not added_fact_ids:
            return topic_state, 0, ()

        updated_topic = self._sync_topic_legacy_lists(replace(topic_state, stored_facts=tuple(updated_facts)))
        updated_topic = self._commit_added_facts(
            topic_id,
            updated_topic,
            added_fact_ids=tuple(added_fact_ids),
            callbacks=callbacks,
            now=now,
        )
        return updated_topic, len(added_fact_ids), tuple(dict.fromkeys(saved_targets))

    def _commit_added_facts(
        self,
        topic_id: str,
        topic_state: UserDiscoveryTopicState,
        *,
        added_fact_ids: Sequence[str],
        callbacks: UserDiscoveryCommitCallbacks | None,
        now: datetime,
    ) -> UserDiscoveryTopicState:
        callback_bundle = callbacks or UserDiscoveryCommitCallbacks()
        updated_facts = list(topic_state.stored_facts)
        added_fact_id_set = set(added_fact_ids)
        for index, fact in enumerate(updated_facts):
            if fact.fact_id not in added_fact_id_set or fact.route_kind in {"user_profile", "personality"}:
                continue
            commit_ref = self._commit_route(fact.to_route(), callbacks=callback_bundle)
            updated_facts[index] = replace(fact, commit_ref=commit_ref, updated_at=now.isoformat())
        updated_topic = replace(topic_state, stored_facts=tuple(updated_facts))
        updated_topic = self._rebuild_managed_context_targets(
            topic_id,
            updated_topic,
            callbacks=callback_bundle,
            now=now,
        )
        return self._sync_topic_legacy_lists(updated_topic)

    def _commit_route(
        self,
        route: UserDiscoveryMemoryRoute,
        *,
        callbacks: UserDiscoveryCommitCallbacks,
    ) -> dict[str, object]:
        if route.route_kind == "contact":
            if callbacks.remember_contact is None:
                raise RuntimeError("User-discovery contact commit callback is missing.")
            result = callbacks.remember_contact(
                given_name=route.given_name,
                family_name=route.family_name or None,
                phone=route.phone or None,
                email=route.email or None,
                role=route.role or None,
                relation=route.relation or None,
                notes=route.notes or None,
            )
            status = _compact_text(getattr(result, "status", None), max_len=32).lower()
            if status in {"needs_clarification", "error", "validation_error"}:
                raise RuntimeError("User-discovery contact commit did not converge cleanly.")
            return {
                "node_id": _compact_text(getattr(result, "node_id", None), max_len=120),
                "edge_type": _compact_text(getattr(result, "edge_type", None), max_len=80),
            }
        if route.route_kind == "preference":
            if callbacks.remember_preference is None:
                raise RuntimeError("User-discovery preference commit callback is missing.")
            result = callbacks.remember_preference(
                category=route.category,
                value=route.value,
                for_product=route.for_product or None,
                sentiment=route.sentiment or "prefer",
                details=route.details or None,
            )
            status = _compact_text(getattr(result, "status", None), max_len=32).lower()
            if status in {"needs_clarification", "error", "validation_error"}:
                raise RuntimeError("User-discovery preference commit did not converge cleanly.")
            return {
                "node_id": _compact_text(getattr(result, "node_id", None), max_len=120),
                "edge_type": _compact_text(getattr(result, "edge_type", None), max_len=80),
            }
        if route.route_kind == "plan":
            if callbacks.remember_plan is None:
                raise RuntimeError("User-discovery plan commit callback is missing.")
            result = callbacks.remember_plan(
                summary=route.summary,
                when_text=route.when_text or None,
                details=route.details or None,
            )
            status = _compact_text(getattr(result, "status", None), max_len=32).lower()
            if status in {"needs_clarification", "error", "validation_error"}:
                raise RuntimeError("User-discovery plan commit did not converge cleanly.")
            return {
                "node_id": _compact_text(getattr(result, "node_id", None), max_len=120),
                "edge_type": _compact_text(getattr(result, "edge_type", None), max_len=80),
            }
        if route.route_kind == "durable_memory":
            if callbacks.store_durable_memory is None:
                raise RuntimeError("User-discovery durable-memory commit callback is missing.")
            result = callbacks.store_durable_memory(
                kind=route.kind,
                summary=route.summary,
                details=route.details or None,
            )
            return {"entry_id": _compact_text(getattr(result, "entry_id", None), max_len=80)}
        raise RuntimeError(f"Unsupported structured discovery route kind: {route.route_kind!r}")

    def _rebuild_managed_context_targets(
        self,
        topic_id: str,
        topic_state: UserDiscoveryTopicState,
        *,
        callbacks: UserDiscoveryCommitCallbacks,
        now: datetime,
    ) -> UserDiscoveryTopicState:
        category = _managed_context_category(topic_id)
        updated_facts = list(topic_state.stored_facts)
        for route_kind, updater, deleter in (
            ("user_profile", callbacks.update_user_profile, callbacks.delete_user_profile),
            ("personality", callbacks.update_personality, callbacks.delete_personality),
        ):
            texts = _active_managed_fact_texts(updated_facts, route_kind=route_kind)
            if texts:
                if updater is None:
                    raise RuntimeError(f"User-discovery {route_kind} commit callback is missing.")
                updater(category, _sentence_list(texts))
            elif deleter is not None:
                deleter(category)
            for index, fact in enumerate(updated_facts):
                if fact.route_kind != route_kind:
                    continue
                updated_facts[index] = replace(fact, commit_ref={"category": category}, updated_at=now.isoformat())
        return replace(topic_state, stored_facts=tuple(updated_facts))

    def _commit_fact_removal(
        self,
        topic_id: str,
        topic_state: UserDiscoveryTopicState,
        removed_fact: UserDiscoveryStoredFact,
        *,
        callbacks: UserDiscoveryCommitCallbacks | None,
    ) -> UserDiscoveryTopicState:
        callback_bundle = callbacks or UserDiscoveryCommitCallbacks()
        if removed_fact.route_kind not in {"user_profile", "personality"}:
            self._delete_structured_fact(removed_fact, callbacks=callback_bundle)
        updated_topic = self._rebuild_managed_context_targets(
            topic_id,
            topic_state,
            callbacks=callback_bundle,
            now=_utc_now(),
        )
        return self._sync_topic_legacy_lists(updated_topic)

    def _delete_structured_fact(
        self,
        fact: UserDiscoveryStoredFact,
        *,
        callbacks: UserDiscoveryCommitCallbacks | None,
    ) -> None:
        callback_bundle = callbacks or UserDiscoveryCommitCallbacks()
        commit_ref = dict(fact.commit_ref or {})
        if fact.route_kind == "contact":
            node_id = _compact_text(commit_ref.get("node_id"), max_len=120)
            if not node_id:
                raise RuntimeError("User-discovery contact deletion is missing a node_id commit reference.")
            if callback_bundle.delete_contact is None:
                raise RuntimeError("User-discovery contact delete callback is missing.")
            callback_bundle.delete_contact(node_id)
            return
        if fact.route_kind == "preference":
            node_id = _compact_text(commit_ref.get("node_id"), max_len=120)
            edge_type = _compact_text(commit_ref.get("edge_type"), max_len=80) or None
            if not node_id:
                raise RuntimeError("User-discovery preference deletion is missing a node_id commit reference.")
            if callback_bundle.delete_preference is None:
                raise RuntimeError("User-discovery preference delete callback is missing.")
            callback_bundle.delete_preference(node_id, edge_type)
            return
        if fact.route_kind == "plan":
            node_id = _compact_text(commit_ref.get("node_id"), max_len=120)
            if not node_id:
                raise RuntimeError("User-discovery plan deletion is missing a node_id commit reference.")
            if callback_bundle.delete_plan is None:
                raise RuntimeError("User-discovery plan delete callback is missing.")
            callback_bundle.delete_plan(node_id)
            return
        if fact.route_kind == "durable_memory":
            entry_id = _compact_text(commit_ref.get("entry_id"), max_len=80)
            if not entry_id:
                raise RuntimeError("User-discovery durable-memory deletion is missing an entry_id commit reference.")
            if callback_bundle.delete_durable_memory is None:
                raise RuntimeError("User-discovery durable-memory delete callback is missing.")
            callback_bundle.delete_durable_memory(entry_id)

    def _locate_fact(
        self,
        state: UserDiscoveryState,
        fact_id: str,
    ) -> tuple[UserDiscoveryTopicState, int, UserDiscoveryStoredFact]:
        for topic in state.topics:
            for index, fact in enumerate(topic.stored_facts):
                if fact.fact_id == fact_id and fact.is_active:
                    return topic, index, fact
        raise ValueError(f"Unknown active discovery fact id: {fact_id!r}")

    def _set_fact_status(
        self,
        topic_state: UserDiscoveryTopicState,
        *,
        fact_id: str,
        status: str,
        now: datetime,
    ) -> UserDiscoveryTopicState:
        updated_facts = [
            replace(fact, status=status, updated_at=now.isoformat()) if fact.fact_id == fact_id else fact
            for fact in topic_state.stored_facts
        ]
        return self._sync_topic_legacy_lists(replace(topic_state, stored_facts=tuple(updated_facts)))

    def _sync_topic_legacy_lists(self, topic_state: UserDiscoveryTopicState) -> UserDiscoveryTopicState:
        return replace(
            topic_state,
            profile_facts=_active_managed_fact_texts(topic_state.stored_facts, route_kind="user_profile"),
            personality_facts=_active_managed_fact_texts(topic_state.stored_facts, route_kind="personality"),
        )
