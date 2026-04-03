"""Public guided user-discovery service over the decomposed helper modules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.context_store import ManagedContextFileStore
from twinr.memory.user_discovery_authoritative_profile import (
    UserDiscoveryAuthoritativeCoverage,
    UserDiscoveryAuthoritativeProfileReader,
)
from twinr.memory.user_discovery_policy import UserDiscoveryPolicyEngine

from .common import (
    _DEFAULT_INITIAL_SNOOZE_DAYS,
    _DEFAULT_LIFELONG_SNOOZE_DAYS,
    _DEFAULT_USER_PROFILE_PATH,
    _DEFAULT_USER_PROFILE_SECTION_TITLE,
    _INITIAL_PAUSE_DELAY,
    _INITIAL_SESSION_MINUTES,
    _LIFELONG_INVITE_DELAY,
    _LIFELONG_SESSION_MINUTES,
    _TOPIC_ORDER,
    _TOPICS_BY_ID,
    _compact_text,
    _normalize_bool,
    _normalize_int,
    _utc_now,
)
from .models import (
    UserDiscoveryCommitCallbacks,
    UserDiscoveryFact,
    UserDiscoveryMemoryRoute,
    UserDiscoveryResult,
    UserDiscoveryState,
    UserDiscoveryTopicState,
)
from .service_commits import UserDiscoveryCommitMixin
from .service_queries import UserDiscoveryQueryMixin
from .store import UserDiscoveryStateStore


@dataclass(slots=True)
class UserDiscoveryService(UserDiscoveryCommitMixin, UserDiscoveryQueryMixin):
    """Own the lifecycle of Twinr's bounded discovery sessions."""

    store: UserDiscoveryStateStore
    user_profile_store: ManagedContextFileStore | None = None
    policy_engine: UserDiscoveryPolicyEngine | None = None
    authoritative_profile_reader: UserDiscoveryAuthoritativeProfileReader | None = None
    initial_session_minutes: int = _INITIAL_SESSION_MINUTES
    lifelong_session_minutes: int = _LIFELONG_SESSION_MINUTES
    _authoritative_coverage_cache: UserDiscoveryAuthoritativeCoverage | None = field(
        init=False,
        default=None,
        repr=False,
    )

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "UserDiscoveryService":
        project_root = Path(config.project_root).expanduser().resolve()
        configured_personality_dir = Path(config.personality_dir or "personality")
        personality_dir = (
            configured_personality_dir
            if configured_personality_dir.is_absolute()
            else project_root / configured_personality_dir
        )
        return cls(
            store=UserDiscoveryStateStore.from_config(config),
            user_profile_store=ManagedContextFileStore(
                personality_dir / Path(_DEFAULT_USER_PROFILE_PATH).name,
                section_title=_DEFAULT_USER_PROFILE_SECTION_TITLE,
                root_dir=config.project_root,
            ),
            policy_engine=UserDiscoveryPolicyEngine.from_config(config),
            authoritative_profile_reader=UserDiscoveryAuthoritativeProfileReader.from_config(config),
        )

    def _invalidate_authoritative_coverage_cache(self) -> None:
        """Drop cached authoritative coverage before discovery mutations."""

        self._authoritative_coverage_cache = None

    def load_state(self) -> UserDiscoveryState:
        self._invalidate_authoritative_coverage_cache()
        try:
            return self._refresh_phase(self.store.load(), now=_utc_now())
        finally:
            self._invalidate_authoritative_coverage_cache()

    def manage(
        self,
        *,
        action: str,
        topic_id: str | None = None,
        learned_facts: Sequence[UserDiscoveryFact] = (),
        memory_routes: Sequence[UserDiscoveryMemoryRoute] = (),
        fact_id: str | None = None,
        topic_complete: bool | None = None,
        permission_granted: bool | None = None,
        snooze_days: int | None = None,
        callbacks: UserDiscoveryCommitCallbacks | None = None,
        now: datetime | None = None,
    ) -> UserDiscoveryResult:
        effective_now = (now or _utc_now()).astimezone(timezone.utc)
        normalized_action = _compact_text(action, max_len=32).lower().replace("-", "_").replace(" ", "_")
        normalized_topic_id = self._normalize_topic_id(topic_id)
        normalized_fact_id = _compact_text(fact_id, max_len=40) or None
        normalized_routes = tuple(route for route in memory_routes if not route.is_empty()) + tuple(
            UserDiscoveryMemoryRoute.from_legacy_fact(fact)
            for fact in learned_facts
            if not fact.is_empty()
        )
        state = self._refresh_phase(self.store.load(), now=effective_now)
        if normalized_action == "status":
            self.store.save(state)
            return self._build_status_result(state, topic_id=normalized_topic_id, now=effective_now)
        if normalized_action == "start_or_resume":
            return self._start_or_resume(state, now=effective_now, topic_id=normalized_topic_id)
        if normalized_action == "pause_session":
            return self._pause_session(state, now=effective_now)
        if normalized_action == "snooze":
            return self._snooze(state, now=effective_now, snooze_days=snooze_days)
        if normalized_action == "skip_topic":
            return self._skip_topic(state, now=effective_now, topic_id=normalized_topic_id)
        if normalized_action == "answer":
            self._invalidate_authoritative_coverage_cache()
            return self._record_answer(
                state,
                now=effective_now,
                topic_id=normalized_topic_id,
                memory_routes=normalized_routes,
                topic_complete=_normalize_bool(topic_complete),
                permission_granted=permission_granted,
                callbacks=callbacks,
            )
        if normalized_action == "review_profile":
            return self._review_profile(state, now=effective_now, topic_id=normalized_topic_id)
        if normalized_action == "delete_fact":
            if normalized_fact_id is None:
                raise ValueError("delete_fact requires a fact_id.")
            self._invalidate_authoritative_coverage_cache()
            return self._delete_fact(state, now=effective_now, fact_id=normalized_fact_id, callbacks=callbacks)
        if normalized_action == "replace_fact":
            if normalized_fact_id is None:
                raise ValueError("replace_fact requires a fact_id.")
            self._invalidate_authoritative_coverage_cache()
            return self._replace_fact(
                state,
                now=effective_now,
                fact_id=normalized_fact_id,
                replacement_routes=normalized_routes,
                callbacks=callbacks,
            )
        raise ValueError(f"Unsupported user discovery action: {action!r}")

    def _start_or_resume(
        self,
        state: UserDiscoveryState,
        *,
        now: datetime,
        topic_id: str | None,
    ) -> UserDiscoveryResult:
        active_topic_id = topic_id or state.active_topic_id or self._select_topic_for_phase(state, now=now)
        if active_topic_id is None:
            refreshed = self._refresh_phase(state, now=now)
            self.store.save(refreshed)
            return self._wrap_up_result(
                refreshed,
                assistant_brief="Thank the user warmly and explain that Twinr has already covered the current discovery setup for now.",
            )

        updated_state = replace(
            state,
            session_state="active",
            active_topic_id=active_topic_id,
            session_started_at=now.isoformat(),
            session_answer_count=0,
            session_answer_limit=self._session_answer_limit_for_phase(state.phase),
            last_interaction_at=now.isoformat(),
            accepted_session_count=state.accepted_session_count + 1,
        )
        updated_state = self._touch_topic(updated_state, active_topic_id, now=now, touch_question=True, touch_visit=True)
        refreshed = self._refresh_phase(updated_state, now=now)
        self.store.save(refreshed)
        return self._build_question_result(
            refreshed,
            topic_id=active_topic_id,
            start_of_session=True,
            facts_saved=0,
            saved_targets=(),
            now=now,
        )

    def _pause_session(
        self,
        state: UserDiscoveryState,
        *,
        now: datetime,
    ) -> UserDiscoveryResult:
        delay = _INITIAL_PAUSE_DELAY if state.phase == "initial_setup" else _LIFELONG_INVITE_DELAY
        updated_state = replace(
            state,
            session_state="paused",
            session_started_at=None,
            session_answer_count=0,
            session_answer_limit=self._session_answer_limit_for_phase(state.phase),
            last_interaction_at=now.isoformat(),
            last_session_closed_at=now.isoformat(),
            next_invite_after=(now + delay).isoformat(),
        )
        refreshed = self._refresh_phase(updated_state, now=now)
        self.store.save(refreshed)
        return self._wrap_up_result(
            refreshed,
            assistant_brief="Acknowledge the pause, say Twinr can continue later, and stop asking discovery questions now.",
        )

    def _snooze(
        self,
        state: UserDiscoveryState,
        *,
        now: datetime,
        snooze_days: int | None,
    ) -> UserDiscoveryResult:
        default_days = _DEFAULT_INITIAL_SNOOZE_DAYS if state.phase == "initial_setup" else _DEFAULT_LIFELONG_SNOOZE_DAYS
        days = _normalize_int(snooze_days, default=default_days, minimum=1, maximum=14)
        updated_state = replace(
            state,
            session_state="snoozed",
            session_started_at=None,
            session_answer_count=0,
            session_answer_limit=self._session_answer_limit_for_phase(state.phase),
            last_interaction_at=now.isoformat(),
            last_session_closed_at=now.isoformat(),
            next_invite_after=(now + timedelta(days=days)).isoformat(),
            snooze_count=state.snooze_count + 1,
        )
        refreshed = self._refresh_phase(updated_state, now=now)
        self.store.save(refreshed)
        return self._wrap_up_result(
            refreshed,
            assistant_brief="Acknowledge that Twinr will ask again later and do not continue the discovery flow now.",
        )

    def _skip_topic(
        self,
        state: UserDiscoveryState,
        *,
        now: datetime,
        topic_id: str | None,
    ) -> UserDiscoveryResult:
        active_topic_id = topic_id or state.active_topic_id or self._select_topic_for_phase(state, now=now)
        if active_topic_id is None:
            refreshed = self._refresh_phase(state, now=now)
            self.store.save(refreshed)
            return self._build_status_result(refreshed, topic_id=None, now=now)
        topic_state = self._topic_state(state, active_topic_id)
        updated_topic = replace(
            topic_state,
            completed_once=True,
            skip_count=topic_state.skip_count + 1,
            last_answer_at=now.isoformat(),
            last_completed_at=now.isoformat(),
        )
        updated_state = self._replace_topic(
            state,
            updated_topic,
            now=now,
            session_answer_increment=1,
        )
        return self._advance_after_topic_progress(
            updated_state,
            now=now,
            current_topic_id=active_topic_id,
            topic_completed=True,
            facts_saved=0,
            facts_deleted=0,
            facts_replaced=0,
            saved_targets=(),
        )

    def _record_answer(
        self,
        state: UserDiscoveryState,
        *,
        now: datetime,
        topic_id: str | None,
        memory_routes: Sequence[UserDiscoveryMemoryRoute],
        topic_complete: bool,
        permission_granted: bool | None,
        callbacks: UserDiscoveryCommitCallbacks | None,
    ) -> UserDiscoveryResult:
        active_topic_id = topic_id or state.active_topic_id or self._select_topic_for_phase(state, now=now)
        if active_topic_id is None:
            return self._start_or_resume(state, now=now, topic_id=topic_id)

        if state.session_state != "active":
            state = replace(
                state,
                session_state="active",
                active_topic_id=active_topic_id,
                session_started_at=now.isoformat(),
                session_answer_count=0,
                session_answer_limit=self._session_answer_limit_for_phase(state.phase),
            )

        topic_state = self._topic_state(state, active_topic_id)
        definition = _TOPICS_BY_ID[active_topic_id]
        updated_topic = replace(
            topic_state,
            question_count=topic_state.question_count + 1,
            last_answer_at=now.isoformat(),
        )
        effective_topic_complete = topic_complete

        if definition.sensitive and updated_topic.permission_state != "granted":
            if permission_granted is None:
                updated_state = replace(
                    state,
                    active_topic_id=active_topic_id,
                    last_interaction_at=now.isoformat(),
                )
                refreshed = self._refresh_phase(updated_state, now=now)
                self.store.save(refreshed)
                return self._build_permission_result(refreshed, topic_id=active_topic_id, facts_saved=0, saved_targets=())
            if not permission_granted:
                updated_topic = replace(
                    updated_topic,
                    permission_state="declined",
                    completed_once=True,
                    skip_count=updated_topic.skip_count + 1,
                    last_completed_at=now.isoformat(),
                )
                updated_state = self._replace_topic(
                    state,
                    updated_topic,
                    now=now,
                    session_answer_increment=1,
                )
                return self._advance_after_topic_progress(
                    updated_state,
                    now=now,
                    current_topic_id=active_topic_id,
                    topic_completed=True,
                    facts_saved=0,
                    facts_deleted=0,
                    facts_replaced=0,
                    saved_targets=(),
                )
            updated_topic = replace(updated_topic, permission_state="granted")

        updated_topic, facts_saved, saved_targets = self._apply_memory_routes(
            active_topic_id,
            updated_topic,
            memory_routes=memory_routes,
            callbacks=callbacks,
            now=now,
        )

        if effective_topic_complete:
            updated_topic = replace(updated_topic, completed_once=True, last_completed_at=now.isoformat())

        updated_state = self._replace_topic(
            state,
            updated_topic,
            now=now,
            session_answer_increment=1,
        )
        return self._advance_after_topic_progress(
            updated_state,
            now=now,
            current_topic_id=active_topic_id,
            topic_completed=effective_topic_complete,
            facts_saved=facts_saved,
            facts_deleted=0,
            facts_replaced=0,
            saved_targets=saved_targets,
        )

    def _review_profile(
        self,
        state: UserDiscoveryState,
        *,
        now: datetime,
        topic_id: str | None,
    ) -> UserDiscoveryResult:
        updated_state = self._close_review_window(
            replace(
                state,
                review_count=state.review_count + 1,
                last_reviewed_at=now.isoformat(),
                last_interaction_at=now.isoformat(),
            ),
            now=now,
        )
        self.store.save(updated_state)
        return self._build_review_result(
            updated_state,
            now=now,
            topic_id=topic_id,
            assistant_brief=(
                "Summarize what Twinr has learned so far, keep it short, and invite the user to confirm, correct, or delete anything that no longer fits."
            ),
        )

    def _delete_fact(
        self,
        state: UserDiscoveryState,
        *,
        now: datetime,
        fact_id: str,
        callbacks: UserDiscoveryCommitCallbacks | None,
    ) -> UserDiscoveryResult:
        topic_state, _index, fact = self._locate_fact(state, fact_id)
        updated_topic = self._set_fact_status(topic_state, fact_id=fact_id, status="deleted", now=now)
        updated_topic = replace(
            updated_topic,
            deletion_count=topic_state.deletion_count + 1,
            last_answer_at=now.isoformat(),
        )
        updated_topic = self._commit_fact_removal(topic_state.topic_id, updated_topic, fact, callbacks=callbacks)
        updated_state = self._replace_topic(state, updated_topic, now=now, session_answer_increment=0)
        updated_state = self._close_review_window(
            replace(
                updated_state,
                review_count=updated_state.review_count + 1,
                last_reviewed_at=now.isoformat(),
            ),
            now=now,
        )
        self.store.save(updated_state)
        return self._build_review_result(
            updated_state,
            now=now,
            topic_id=topic_state.topic_id,
            facts_deleted=1,
            saved_targets=(fact.route_kind,),
            assistant_brief="Acknowledge the change and briefly review the remaining learned details.",
        )

    def _replace_fact(
        self,
        state: UserDiscoveryState,
        *,
        now: datetime,
        fact_id: str,
        replacement_routes: Sequence[UserDiscoveryMemoryRoute],
        callbacks: UserDiscoveryCommitCallbacks | None,
    ) -> UserDiscoveryResult:
        if not replacement_routes:
            raise ValueError("replace_fact requires at least one replacement route.")
        topic_state, _index, fact = self._locate_fact(state, fact_id)
        updated_topic = self._set_fact_status(topic_state, fact_id=fact_id, status="superseded", now=now)
        updated_topic = replace(
            updated_topic,
            correction_count=topic_state.correction_count + 1,
            last_answer_at=now.isoformat(),
        )
        if fact.route_kind not in {"user_profile", "personality"}:
            self._delete_structured_fact(fact, callbacks=callbacks)
        updated_topic, facts_saved, saved_targets = self._apply_memory_routes(
            topic_state.topic_id,
            updated_topic,
            memory_routes=replacement_routes,
            callbacks=callbacks,
            now=now,
        )
        updated_state = self._replace_topic(state, updated_topic, now=now, session_answer_increment=0)
        updated_state = self._close_review_window(
            replace(
                updated_state,
                review_count=updated_state.review_count + 1,
                last_reviewed_at=now.isoformat(),
            ),
            now=now,
        )
        self.store.save(updated_state)
        return self._build_review_result(
            updated_state,
            now=now,
            topic_id=topic_state.topic_id,
            facts_saved=facts_saved,
            facts_replaced=1,
            saved_targets=saved_targets,
            assistant_brief="Acknowledge the correction and briefly review the updated learned details.",
        )

    def _close_review_window(self, state: UserDiscoveryState, *, now: datetime) -> UserDiscoveryState:
        delay = _INITIAL_PAUSE_DELAY if state.phase == "initial_setup" else _LIFELONG_INVITE_DELAY
        return replace(
            state,
            session_state="idle",
            session_started_at=None,
            session_answer_count=0,
            session_answer_limit=self._session_answer_limit_for_phase(state.phase),
            last_interaction_at=now.isoformat(),
            last_session_closed_at=now.isoformat(),
            next_invite_after=(now + delay).isoformat(),
        )

    def _advance_after_topic_progress(
        self,
        state: UserDiscoveryState,
        *,
        now: datetime,
        current_topic_id: str,
        topic_completed: bool,
        facts_saved: int,
        facts_deleted: int,
        facts_replaced: int,
        saved_targets: tuple[str, ...],
    ) -> UserDiscoveryResult:
        refreshed = self._refresh_phase(state, now=now)
        if refreshed.phase == "lifelong_learning" and refreshed.setup_completed_at is not None and state.phase == "initial_setup":
            finalized = replace(
                refreshed,
                session_state="idle",
                session_started_at=None,
                session_answer_count=0,
                session_answer_limit=self._session_answer_limit_for_phase("lifelong_learning"),
                last_session_closed_at=now.isoformat(),
                next_invite_after=(now + _LIFELONG_INVITE_DELAY).isoformat(),
            )
            finalized = replace(
                finalized,
                active_topic_id=self._select_lifelong_topic(finalized, now=now),
            )
            self.store.save(finalized)
            return self._wrap_up_result(
                finalized,
                facts_saved=facts_saved,
                facts_deleted=facts_deleted,
                facts_replaced=facts_replaced,
                saved_targets=saved_targets,
                assistant_brief="Thank the user warmly, say the first discovery setup is complete for now, and mention that Twinr can ask small follow-up questions again in the future.",
            )

        answer_limit_reached = refreshed.session_answer_count >= refreshed.session_answer_limit
        if refreshed.phase == "lifelong_learning":
            if topic_completed or answer_limit_reached:
                finalized = replace(
                    refreshed,
                    session_state="idle",
                    session_started_at=None,
                    session_answer_count=0,
                    session_answer_limit=self._session_answer_limit_for_phase(refreshed.phase),
                    last_session_closed_at=now.isoformat(),
                    next_invite_after=(now + _LIFELONG_INVITE_DELAY).isoformat(),
                    active_topic_id=self._select_lifelong_topic(refreshed, now=now),
                )
                self.store.save(finalized)
                return self._wrap_up_result(
                    finalized,
                    facts_saved=facts_saved,
                    facts_deleted=facts_deleted,
                    facts_replaced=facts_replaced,
                    saved_targets=saved_targets,
                    assistant_brief="Thank the user, say this was enough for today, and stop the discovery run for now.",
                )
            continued = self._touch_topic(refreshed, current_topic_id, now=now, touch_question=True, touch_visit=False)
            self.store.save(continued)
            return self._build_question_result(
                continued,
                topic_id=current_topic_id,
                start_of_session=False,
                facts_saved=facts_saved,
                saved_targets=saved_targets,
                now=now,
            )

        if topic_completed:
            next_topic_id = self._select_initial_topic(refreshed)
            if next_topic_id is not None and not answer_limit_reached:
                continued = replace(refreshed, active_topic_id=next_topic_id)
                continued = self._touch_topic(continued, next_topic_id, now=now, touch_question=True, touch_visit=True)
                self.store.save(continued)
                return self._build_question_result(
                    continued,
                    topic_id=next_topic_id,
                    start_of_session=False,
                    facts_saved=facts_saved,
                    saved_targets=saved_targets,
                    now=now,
                )

            paused = replace(
                refreshed,
                session_state="paused",
                session_started_at=None,
                session_answer_count=0,
                session_answer_limit=self._session_answer_limit_for_phase(refreshed.phase),
                last_session_closed_at=now.isoformat(),
                next_invite_after=(now + _INITIAL_PAUSE_DELAY).isoformat(),
                active_topic_id=next_topic_id or current_topic_id,
            )
            self.store.save(paused)
            return self._wrap_up_result(
                paused,
                facts_saved=facts_saved,
                facts_deleted=facts_deleted,
                facts_replaced=facts_replaced,
                saved_targets=saved_targets,
                assistant_brief="Thank the user, say this is enough for now, and mention that Twinr can continue the setup later.",
            )

        if answer_limit_reached:
            paused = replace(
                refreshed,
                session_state="paused",
                session_started_at=None,
                session_answer_count=0,
                session_answer_limit=self._session_answer_limit_for_phase(refreshed.phase),
                last_session_closed_at=now.isoformat(),
                next_invite_after=(now + _INITIAL_PAUSE_DELAY).isoformat(),
                active_topic_id=current_topic_id,
            )
            self.store.save(paused)
            return self._wrap_up_result(
                paused,
                facts_saved=facts_saved,
                facts_deleted=facts_deleted,
                facts_replaced=facts_replaced,
                saved_targets=saved_targets,
                assistant_brief="Thank the user, say this is enough for now, and pause the setup so it can continue later.",
            )

        continued = self._touch_topic(refreshed, current_topic_id, now=now, touch_question=True, touch_visit=False)
        self.store.save(continued)
        return self._build_question_result(
            continued,
            topic_id=current_topic_id,
            start_of_session=False,
            facts_saved=facts_saved,
            saved_targets=saved_targets,
            now=now,
        )

    def _replace_topic(
        self,
        state: UserDiscoveryState,
        topic_state: UserDiscoveryTopicState,
        *,
        now: datetime,
        session_answer_increment: int,
    ) -> UserDiscoveryState:
        updated_topics = {topic.topic_id: topic for topic in state.topics}
        updated_topics[topic_state.topic_id] = topic_state
        ordered_topics = tuple(
            updated_topics.get(topic_id, UserDiscoveryTopicState.empty(topic_id))
            for topic_id in _TOPIC_ORDER
            if topic_id in updated_topics or topic_id == topic_state.topic_id
        )
        return replace(
            state,
            topics=ordered_topics,
            active_topic_id=topic_state.topic_id,
            last_interaction_at=now.isoformat(),
            session_answer_count=state.session_answer_count + max(0, session_answer_increment),
        )

    def _touch_topic(
        self,
        state: UserDiscoveryState,
        topic_id: str,
        *,
        now: datetime,
        touch_question: bool,
        touch_visit: bool,
    ) -> UserDiscoveryState:
        topic_state = self._topic_state(state, topic_id)
        updated_topic = replace(
            topic_state,
            visit_count=topic_state.visit_count + (1 if touch_visit else 0),
            last_question_at=now.isoformat() if touch_question else topic_state.last_question_at,
        )
        updated_topics = {topic.topic_id: topic for topic in state.topics}
        updated_topics[topic_id] = updated_topic
        ordered = tuple(updated_topics[topic_id] for topic_id in _TOPIC_ORDER if topic_id in updated_topics)
        return replace(state, topics=ordered)
