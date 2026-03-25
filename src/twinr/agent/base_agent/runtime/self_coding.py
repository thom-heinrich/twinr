"""Expose bounded runtime guidance for active self-coding flows.

This module keeps self-coding state lookup out of the generic context builder.
The tool lane only needs a compact snapshot that tells it whether there is an
active requirements dialogue or a compile job it can continue without asking
the user for internal identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from twinr.agent.self_coding import SelfCodingStore


_SELF_CODING_GUIDANCE_LOCK_GUARD = Lock()
_ACTIVE_DIALOGUE_STATUSES = frozenset({"questioning", "confirming", "ready_for_compile"})
_ACTIVE_SKILL_STATUSES = frozenset({"active", "soft_launch_ready"})


@dataclass(frozen=True, slots=True)
class SelfCodingGuidanceState:
    """Compact self-coding state needed by the live tool lane."""

    session_id: str | None = None
    session_state: str | None = None
    skill_id: str | None = None
    skill_name: str | None = None
    current_question_id: str | None = None
    compile_job_id: str | None = None
    compile_job_status: str | None = None
    active_versions: tuple[int, ...] = ()
    paused_versions: tuple[int, ...] = ()


class TwinrRuntimeSelfCodingMixin:
    """Provide runtime-owned read access to active self-coding state."""

    def _self_coding_store(self) -> SelfCodingStore:
        store = getattr(self, "_twinr_self_coding_guidance_store", None)
        if isinstance(store, SelfCodingStore):
            return store
        with _SELF_CODING_GUIDANCE_LOCK_GUARD:
            store = getattr(self, "_twinr_self_coding_guidance_store", None)
            if not isinstance(store, SelfCodingStore):
                store = SelfCodingStore.from_config(self.config)
                setattr(self, "_twinr_self_coding_guidance_store", store)
        return store

    def self_coding_guidance_state(self) -> SelfCodingGuidanceState | None:
        """Return the latest active self-coding dialogue or activation state."""

        try:
            store = self._self_coding_store()
            session = next(
                (
                    item
                    for item in store.list_dialogue_sessions()
                    if str(getattr(item, "status", "") or "").strip().lower() in _ACTIVE_DIALOGUE_STATUSES
                ),
                None,
            )
            if session is None:
                return None
            session_id = str(getattr(session, "session_id", "") or "").strip() or None
            skill_id = str(getattr(session, "skill_id", "") or "").strip() or None
            compile_job = store.find_job_for_session(session.session_id) if session_id is not None else None
            activations = store.list_activations(skill_id=skill_id) if skill_id else ()
            active_versions = tuple(
                int(getattr(item, "version", 0))
                for item in activations
                if str(getattr(item, "status", "") or "").strip().lower() in _ACTIVE_SKILL_STATUSES
            )
            paused_versions = tuple(
                int(getattr(item, "version", 0))
                for item in activations
                if str(getattr(item, "status", "") or "").strip().lower() == "paused"
            )
            return SelfCodingGuidanceState(
                session_id=session_id,
                session_state=str(getattr(session, "status", "") or "").strip().lower() or None,
                skill_id=skill_id,
                skill_name=str(getattr(session, "skill_name", "") or "").strip() or None,
                current_question_id=str(getattr(session, "current_question_id", "") or "").strip() or None,
                compile_job_id=str(getattr(compile_job, "job_id", "") or "").strip() or None,
                compile_job_status=str(getattr(compile_job, "status", "") or "").strip().lower() or None,
                active_versions=active_versions,
                paused_versions=paused_versions,
            )
        except Exception:
            return None


__all__ = ["SelfCodingGuidanceState", "TwinrRuntimeSelfCodingMixin"]
