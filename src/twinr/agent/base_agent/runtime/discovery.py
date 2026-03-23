"""Expose runtime helpers for Twinr's guided user-discovery flow."""

from __future__ import annotations

from datetime import datetime
from threading import Lock

from twinr.memory.user_discovery import (
    UserDiscoveryCommitCallbacks,
    UserDiscoveryFact,
    UserDiscoveryMemoryRoute,
    UserDiscoveryResult,
    UserDiscoveryService,
)

_DISCOVERY_LOCK_GUARD = Lock()


class TwinrRuntimeDiscoveryMixin:
    """Provide runtime-owned access to the user-discovery service."""

    def _user_discovery_service(self) -> UserDiscoveryService:
        service = getattr(self, "_twinr_user_discovery_service", None)
        if service is None:
            with _DISCOVERY_LOCK_GUARD:
                service = getattr(self, "_twinr_user_discovery_service", None)
                if service is None:
                    service = UserDiscoveryService.from_config(self.config)
                    setattr(self, "_twinr_user_discovery_service", service)
        return service

    def manage_user_discovery(
        self,
        *,
        action: str,
        topic_id: str | None = None,
        learned_facts: tuple[UserDiscoveryFact, ...] = (),
        memory_routes: tuple[UserDiscoveryMemoryRoute, ...] = (),
        fact_id: str | None = None,
        topic_complete: bool | None = None,
        permission_granted: bool | None = None,
        snooze_days: int | None = None,
        now: datetime | None = None,
    ) -> UserDiscoveryResult:
        """Advance or inspect the bounded guided user-discovery flow."""

        with self._memory_runtime_lock():
            callbacks = UserDiscoveryCommitCallbacks(
                update_user_profile=lambda category, instruction: self.update_user_profile_context(
                    category=category,
                    instruction=instruction,
                ),
                delete_user_profile=lambda category: self.remove_user_profile_context(category=category),
                update_personality=lambda category, instruction: self.update_personality_context(
                    category=category,
                    instruction=instruction,
                ),
                delete_personality=lambda category: self.remove_personality_context(category=category),
                remember_contact=lambda **kwargs: self.remember_contact(
                    **kwargs,
                    source="user_discovery",
                ),
                delete_contact=lambda node_id: self.delete_contact(node_id=node_id),
                remember_preference=lambda **kwargs: self.remember_preference(
                    **kwargs,
                    source="user_discovery",
                ),
                delete_preference=lambda node_id, edge_type: self.delete_preference(node_id=node_id, edge_type=edge_type),
                remember_plan=lambda **kwargs: self.remember_plan(
                    **kwargs,
                    source="user_discovery",
                ),
                delete_plan=lambda node_id: self.delete_plan(node_id=node_id),
                store_durable_memory=lambda **kwargs: self.store_durable_memory(**kwargs),
                delete_durable_memory=lambda entry_id: self.delete_durable_memory_entry(entry_id=entry_id),
            )
            return self._user_discovery_service().manage(
                action=action,
                topic_id=topic_id,
                learned_facts=learned_facts,
                memory_routes=memory_routes,
                fact_id=fact_id,
                topic_complete=topic_complete,
                permission_granted=permission_granted,
                snooze_days=snooze_days,
                callbacks=callbacks,
                now=now,
            )


__all__ = ["TwinrRuntimeDiscoveryMixin"]
