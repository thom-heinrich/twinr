"""Expose runtime helpers for Twinr's guided user-discovery flow."""

# mypy: disable-error-code=attr-defined

from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock

from twinr.memory.user_discovery import (
    UserDiscoveryCommitCallbacks,
    UserDiscoveryFact,
    UserDiscoveryMemoryRoute,
    UserDiscoveryResult,
    UserDiscoveryService,
)
from twinr.proactive.runtime.display_reserve_user_discovery_feedback import (
    record_user_discovery_invite_feedback,
)

_DISCOVERY_LOCK_GUARD = Lock()
_DIRECT_DISCOVERY_FEEDBACK_STATUS = {
    "start_or_resume": "engaged",
    "answer": "engaged",
    "review_profile": "engaged",
    "replace_fact": "engaged",
    "delete_fact": "engaged",
    "pause_session": "cooled",
    "snooze": "cooled",
    "skip_topic": "avoided",
}


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

        effective_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        normalized_action = str(action or "").strip().lower().replace("-", "_").replace(" ", "_")
        service = self._user_discovery_service()
        pending_invite = None
        if normalized_action in _DIRECT_DISCOVERY_FEEDBACK_STATUS:
            pending_invite = service.build_invitation(now=effective_now)
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
            result = service.manage(
                action=action,
                topic_id=topic_id,
                learned_facts=learned_facts,
                memory_routes=memory_routes,
                fact_id=fact_id,
                topic_complete=topic_complete,
                permission_granted=permission_granted,
                snooze_days=snooze_days,
                callbacks=callbacks,
                now=effective_now,
            )
        feedback_status = _DIRECT_DISCOVERY_FEEDBACK_STATUS.get(normalized_action)
        if (
            pending_invite is not None
            and feedback_status is not None
            and (topic_id is None or str(topic_id).strip().lower().replace("-", "_").replace(" ", "_") == pending_invite.topic_id)
        ):
            record_user_discovery_invite_feedback(
                self.config,
                invite=pending_invite,
                status=feedback_status,
                occurred_at=effective_now,
                summary=getattr(result, "assistant_brief", None) or f"user_discovery:{normalized_action}",
            )
        return result


__all__ = ["TwinrRuntimeDiscoveryMixin"]
