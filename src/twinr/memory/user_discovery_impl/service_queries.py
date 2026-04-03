"""Selection and presentation helpers for the discovery service."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone

from twinr.memory.user_discovery_authoritative_profile import UserDiscoveryAuthoritativeCoverage
from twinr.memory.user_discovery_policy import UserDiscoveryEngagementSignals, UserDiscoveryTopicPolicy

from .common import (
    LOGGER,
    _INITIAL_SESSION_ANSWER_LIMIT,
    _LIFELONG_SESSION_ANSWER_LIMIT,
    _MAX_BRIEF_LENGTH,
    _TOPIC_DEFINITIONS,
    _TOPIC_ORDER,
    _TOPICS_BY_ID,
    _compact_text,
    _parse_timestamp,
    _sentence_list,
    _utc_now,
)
from .models import (
    UserDiscoveryInvite,
    UserDiscoveryResult,
    UserDiscoveryReviewItem,
    UserDiscoveryState,
    UserDiscoveryTopicState,
)
from .store import UserDiscoveryStateStore
from twinr.memory.context_store import ManagedContextFileStore
from twinr.memory.user_discovery_authoritative_profile import UserDiscoveryAuthoritativeProfileReader
from twinr.memory.user_discovery_policy import UserDiscoveryPolicyEngine


class UserDiscoveryQueryMixin:
    """Own selection, review, and result rendering helpers."""

    store: UserDiscoveryStateStore
    user_profile_store: ManagedContextFileStore | None
    policy_engine: UserDiscoveryPolicyEngine | None
    authoritative_profile_reader: UserDiscoveryAuthoritativeProfileReader | None
    initial_session_minutes: int
    lifelong_session_minutes: int
    _authoritative_coverage_cache: UserDiscoveryAuthoritativeCoverage | None

    def build_invitation(self, *, now: datetime | None = None) -> UserDiscoveryInvite | None:
        effective_now = (now or _utc_now()).astimezone(timezone.utc)
        state = self._refresh_phase(self.store.load(), now=effective_now)
        if state.session_state == "active":
            return None
        next_invite_after = _parse_timestamp(state.next_invite_after)
        if next_invite_after is not None and effective_now < next_invite_after:
            return None

        if state.phase != "initial_setup" and self._review_due(state, now=effective_now):
            review_topic_id = self._select_review_topic(state, now=effective_now) or "basics"
            return UserDiscoveryInvite(
                invite_kind="review_profile",
                phase=state.phase,
                topic_id=review_topic_id,
                topic_label="Profile Review",
                display_topic_label="Gelerntes",
                session_minutes=self.lifelong_session_minutes,
                headline="Ich kann dir zeigen, was ich schon ueber dich gelernt habe.",
                body="Dann kannst du bestaetigen, aendern oder loeschen, was nicht mehr passt.",
                salience=0.9,
                reason="user_discovery_review_due",
                display_prompt_stage="review",
            )

        if state.phase == "initial_setup":
            topic_id = state.active_topic_id or self._select_initial_topic(state)
            if topic_id is None:
                return None
            definition = _TOPICS_BY_ID[topic_id]
            display_prompt_stage = self._display_prompt_stage(state, topic_id)
            if state.accepted_session_count <= 0:
                return UserDiscoveryInvite(
                    invite_kind="start_setup",
                    phase=state.phase,
                    topic_id=topic_id,
                    topic_label=definition.label,
                    display_topic_label=definition.display_label,
                    session_minutes=self.initial_session_minutes,
                    headline="Hast du 15 Minuten? Lass uns die Einrichtung starten.",
                    body=_compact_text(
                        f"Wir koennen jederzeit pausieren. Ich beginne mit {definition.display_label}.",
                        max_len=112,
                    ),
                    salience=0.98,
                    reason="user_discovery_initial_setup_due",
                    display_prompt_stage=display_prompt_stage,
                )
            return UserDiscoveryInvite(
                invite_kind="resume_setup",
                phase=state.phase,
                topic_id=topic_id,
                topic_label=definition.label,
                display_topic_label=definition.display_label,
                session_minutes=self.initial_session_minutes,
                headline="Wollen wir bei deiner Einrichtung weitermachen?",
                body=_compact_text(
                    f"Als Naechstes waere {definition.display_label} dran. Das geht in kleinen Schritten.",
                    max_len=112,
                ),
                salience=0.94,
                reason="user_discovery_resume_due",
                display_prompt_stage=display_prompt_stage,
            )

        topic_id = state.active_topic_id or self._select_lifelong_topic(state, now=effective_now)
        if topic_id is None:
            return None
        definition = _TOPICS_BY_ID[topic_id]
        policy = self._topic_policy(state, topic_id, now=effective_now)
        display_prompt_stage = self._display_prompt_stage(state, topic_id)
        if policy.question_style == "gentle_optional":
            body = (
                f"Nur wenn es fuer dich gerade passt. Ich haette heute ein paar leichte Fragen zu "
                f"{definition.display_label}."
            )
        elif policy.question_style == "deeper_follow_up":
            body = f"Ich haette heute ein paar kurze Folgefragen zu {definition.display_label}."
        else:
            body = f"Ich haette heute ein paar kurze Fragen zu {definition.display_label}."
        return UserDiscoveryInvite(
            invite_kind="lifelong_learning",
            phase=state.phase,
            topic_id=topic_id,
            topic_label=definition.label,
            display_topic_label=definition.display_label,
            session_minutes=self.lifelong_session_minutes,
            headline="Ich wuerde dich gern noch besser kennenlernen. Hast du 5 Minuten?",
            body=_compact_text(body, max_len=112),
            salience=max(0.62, min(0.94, 0.84 + policy.invite_salience_adjustment)),
            reason="user_discovery_lifelong_due",
            display_prompt_stage=display_prompt_stage,
        )

    def _build_status_result(
        self,
        state: UserDiscoveryState,
        *,
        topic_id: str | None,
        now: datetime,
    ) -> UserDiscoveryResult:
        resolved_topic_id = topic_id or state.active_topic_id
        definition = _TOPICS_BY_ID.get(resolved_topic_id or "")
        topic_state = self._topic_state(state, resolved_topic_id) if resolved_topic_id else None
        policy = self._topic_policy(state, resolved_topic_id, now=now) if resolved_topic_id else None
        return UserDiscoveryResult(
            phase=state.phase,
            session_state=state.session_state,
            response_mode="status",
            topic_id=resolved_topic_id,
            topic_label=definition.label if definition is not None else None,
            display_topic_label=definition.display_label if definition is not None else None,
            topic_goal=definition.goal if definition is not None else None,
            assistant_brief="Use this status to decide whether to resume discovery, ask one bounded question, leave the flow inactive, or offer a short review.",
            question_brief=self._question_brief(definition, topic_state, state.phase, policy=policy) if definition and topic_state else None,
            current_topic_summary=self._topic_summary(topic_state) if topic_state is not None else None,
            session_minutes=self._session_minutes_for_phase(state.phase),
            session_answers_used=state.session_answer_count,
            session_answers_remaining=max(0, state.session_answer_limit - state.session_answer_count),
            facts_saved=0,
            saved_targets=(),
            review_items=(),
            question_style=policy.question_style if policy is not None else None,
            engagement_state=policy.engagement_state if policy is not None else None,
            can_pause=True,
            can_skip_topic=True,
            sensitive_permission_required=bool(definition.sensitive and topic_state.permission_state != "granted") if definition and topic_state else False,
            setup_topics_completed=self._setup_topics_completed(state),
            setup_topics_total=len(_TOPIC_ORDER),
            setup_complete=bool(state.setup_completed_at),
            next_invite_after=state.next_invite_after,
        )

    def _build_permission_result(
        self,
        state: UserDiscoveryState,
        *,
        topic_id: str,
        facts_saved: int,
        saved_targets: tuple[str, ...],
    ) -> UserDiscoveryResult:
        definition = _TOPICS_BY_ID[topic_id]
        topic_state = self._topic_state(state, topic_id)
        return UserDiscoveryResult(
            phase=state.phase,
            session_state=state.session_state,
            response_mode="ask_permission",
            topic_id=topic_id,
            topic_label=definition.label,
            display_topic_label=definition.display_label,
            topic_goal=definition.goal,
            assistant_brief="Ask for explicit permission before continuing with this sensitive discovery topic.",
            question_brief=definition.permission_brief,
            current_topic_summary=self._topic_summary(topic_state),
            session_minutes=self._session_minutes_for_phase(state.phase),
            session_answers_used=state.session_answer_count,
            session_answers_remaining=max(0, state.session_answer_limit - state.session_answer_count),
            facts_saved=facts_saved,
            saved_targets=saved_targets,
            can_pause=True,
            can_skip_topic=True,
            sensitive_permission_required=True,
            setup_topics_completed=self._setup_topics_completed(state),
            setup_topics_total=len(_TOPIC_ORDER),
            setup_complete=bool(state.setup_completed_at),
            next_invite_after=state.next_invite_after,
        )

    def _build_question_result(
        self,
        state: UserDiscoveryState,
        *,
        topic_id: str,
        start_of_session: bool,
        facts_saved: int,
        saved_targets: tuple[str, ...],
        now: datetime,
    ) -> UserDiscoveryResult:
        definition = _TOPICS_BY_ID[topic_id]
        topic_state = self._topic_state(state, topic_id)
        if definition.sensitive and topic_state.permission_state != "granted":
            return self._build_permission_result(
                state,
                topic_id=topic_id,
                facts_saved=facts_saved,
                saved_targets=saved_targets,
            )
        policy = self._topic_policy(state, topic_id, now=now)
        assistant_brief = (
            "Open the bounded discovery session warmly, remind the user that it can pause at any time, and ask one short question in the returned topic."
            if start_of_session
            else "Stay in the returned topic and ask one short follow-up question."
        )
        if policy.question_style == "gentle_optional":
            assistant_brief = (
                f"{assistant_brief} Keep the tone especially light, explicitly easy to skip, and avoid stacking multiple follow-ups."
            )
        elif policy.question_style == "deeper_follow_up":
            assistant_brief = f"{assistant_brief} One slightly deeper follow-up is welcome if the user sounds engaged."
        return UserDiscoveryResult(
            phase=state.phase,
            session_state=state.session_state,
            response_mode="ask_question",
            topic_id=topic_id,
            topic_label=definition.label,
            display_topic_label=definition.display_label,
            topic_goal=definition.goal,
            assistant_brief=assistant_brief,
            question_brief=self._question_brief(definition, topic_state, state.phase, policy=policy),
            current_topic_summary=self._topic_summary(topic_state),
            session_minutes=self._session_minutes_for_phase(state.phase),
            session_answers_used=state.session_answer_count,
            session_answers_remaining=max(0, state.session_answer_limit - state.session_answer_count),
            facts_saved=facts_saved,
            saved_targets=saved_targets,
            question_style=policy.question_style,
            engagement_state=policy.engagement_state,
            can_pause=True,
            can_skip_topic=True,
            sensitive_permission_required=False,
            setup_topics_completed=self._setup_topics_completed(state),
            setup_topics_total=len(_TOPIC_ORDER),
            setup_complete=bool(state.setup_completed_at),
            next_invite_after=state.next_invite_after,
        )

    def _wrap_up_result(
        self,
        state: UserDiscoveryState,
        *,
        assistant_brief: str,
        facts_saved: int = 0,
        facts_deleted: int = 0,
        facts_replaced: int = 0,
        saved_targets: tuple[str, ...] = (),
    ) -> UserDiscoveryResult:
        return UserDiscoveryResult(
            phase=state.phase,
            session_state=state.session_state,
            response_mode="wrap_up",
            topic_id=state.active_topic_id,
            topic_label=_TOPICS_BY_ID[state.active_topic_id].label if state.active_topic_id in _TOPICS_BY_ID else None,
            display_topic_label=_TOPICS_BY_ID[state.active_topic_id].display_label if state.active_topic_id in _TOPICS_BY_ID else None,
            topic_goal=_TOPICS_BY_ID[state.active_topic_id].goal if state.active_topic_id in _TOPICS_BY_ID else None,
            assistant_brief=_compact_text(assistant_brief, max_len=_MAX_BRIEF_LENGTH),
            question_brief=None,
            current_topic_summary=self._topic_summary(self._topic_state(state, state.active_topic_id)) if state.active_topic_id else None,
            session_minutes=self._session_minutes_for_phase(state.phase),
            session_answers_used=0,
            session_answers_remaining=0,
            facts_saved=facts_saved,
            facts_deleted=facts_deleted,
            facts_replaced=facts_replaced,
            saved_targets=saved_targets,
            can_pause=False,
            can_skip_topic=False,
            sensitive_permission_required=False,
            setup_topics_completed=self._setup_topics_completed(state),
            setup_topics_total=len(_TOPIC_ORDER),
            setup_complete=bool(state.setup_completed_at),
            next_invite_after=state.next_invite_after,
        )

    def _build_review_result(
        self,
        state: UserDiscoveryState,
        *,
        now: datetime,
        topic_id: str | None,
        assistant_brief: str,
        facts_saved: int = 0,
        facts_deleted: int = 0,
        facts_replaced: int = 0,
        saved_targets: tuple[str, ...] = (),
    ) -> UserDiscoveryResult:
        resolved_topic_id = topic_id or self._select_review_topic(state, now=now)
        definition = _TOPICS_BY_ID.get(resolved_topic_id or "")
        topic_state = self._topic_state(state, resolved_topic_id) if resolved_topic_id else None
        return UserDiscoveryResult(
            phase=state.phase,
            session_state=state.session_state,
            response_mode="review_profile",
            topic_id=resolved_topic_id,
            topic_label=definition.label if definition is not None else None,
            display_topic_label="Gelerntes",
            topic_goal=definition.goal if definition is not None else None,
            assistant_brief=_compact_text(assistant_brief, max_len=_MAX_BRIEF_LENGTH),
            question_brief="Keep the spoken review short, mention only a few learned items, and offer correction or deletion when needed.",
            current_topic_summary=self._topic_summary(topic_state) if topic_state is not None else self._profile_summary(state),
            session_minutes=self.lifelong_session_minutes,
            session_answers_used=0,
            session_answers_remaining=0,
            facts_saved=facts_saved,
            facts_deleted=facts_deleted,
            facts_replaced=facts_replaced,
            saved_targets=saved_targets,
            review_items=self._review_items(state, topic_id=topic_id),
            can_pause=False,
            can_skip_topic=False,
            sensitive_permission_required=False,
            setup_topics_completed=self._setup_topics_completed(state),
            setup_topics_total=len(_TOPIC_ORDER),
            setup_complete=bool(state.setup_completed_at),
            next_invite_after=state.next_invite_after,
        )

    def _refresh_phase(self, state: UserDiscoveryState, *, now: datetime) -> UserDiscoveryState:
        if state.setup_completed_at is not None and state.phase != "lifelong_learning":
            return replace(state, phase="lifelong_learning")
        if state.phase == "initial_setup" and self._setup_topics_completed(state) >= len(_TOPIC_ORDER):
            return replace(
                state,
                phase="lifelong_learning",
                setup_completed_at=state.setup_completed_at or now.isoformat(),
            )
        return state

    def _setup_topics_completed(self, state: UserDiscoveryState) -> int:
        completed = 0
        for topic_id in _TOPIC_ORDER:
            topic_state = self._effective_topic_state(state, topic_id)
            if topic_state.completed_once or topic_state.skip_count > 0:
                completed += 1
        return completed

    def _session_answer_limit_for_phase(self, phase: str) -> int:
        return _INITIAL_SESSION_ANSWER_LIMIT if phase == "initial_setup" else _LIFELONG_SESSION_ANSWER_LIMIT

    def _session_minutes_for_phase(self, phase: str) -> int:
        return self.initial_session_minutes if phase == "initial_setup" else self.lifelong_session_minutes

    def _topic_state(self, state: UserDiscoveryState, topic_id: str | None) -> UserDiscoveryTopicState:
        normalized_topic_id = self._normalize_topic_id(topic_id)
        if normalized_topic_id is None:
            raise ValueError("A valid discovery topic is required.")
        for topic in state.topics:
            if topic.topic_id == normalized_topic_id:
                return topic
        return UserDiscoveryTopicState.empty(normalized_topic_id)

    def _normalize_topic_id(self, topic_id: str | None) -> str | None:
        if topic_id is None:
            return None
        normalized = _compact_text(topic_id, max_len=48).lower().replace("-", "_").replace(" ", "_")
        if normalized not in _TOPICS_BY_ID:
            return None
        return normalized

    def _select_topic_for_phase(self, state: UserDiscoveryState, *, now: datetime) -> str | None:
        if state.phase == "initial_setup":
            return self._select_initial_topic(state)
        return self._select_lifelong_topic(state, now=now)

    def _select_initial_topic(self, state: UserDiscoveryState) -> str | None:
        for topic_id in _TOPIC_ORDER:
            topic_state = self._effective_topic_state(state, topic_id)
            if topic_state.completed_once or topic_state.skip_count > 0:
                continue
            if _TOPICS_BY_ID[topic_id].sensitive and topic_state.permission_state == "declined":
                continue
            return topic_id
        return None

    def _select_lifelong_topic(self, state: UserDiscoveryState, *, now: datetime) -> str | None:
        signals = self.policy_engine.load_signals(now=now) if self.policy_engine is not None else {}
        candidates: list[tuple[tuple[object, ...], str]] = []
        for definition in _TOPIC_DEFINITIONS:
            topic_state = self._effective_topic_state(state, definition.topic_id)
            if definition.sensitive and topic_state.permission_state == "declined":
                continue
            policy = self._topic_policy(state, definition.topic_id, now=now, signals=signals)
            key = (-round(policy.score, 6), definition.initial_order)
            candidates.append((key, definition.topic_id))
        if not candidates:
            return None
        candidates.sort(key=lambda entry: entry[0])
        return candidates[0][1]

    def _display_prompt_stage(self, state: UserDiscoveryState, topic_id: str) -> str:
        topic_state = self._effective_topic_state(state, topic_id)
        if topic_state.fact_count > 0 or topic_state.last_answer_at is not None:
            return "follow_up"
        return "opener"

    def _effective_topic_state(self, state: UserDiscoveryState, topic_id: str) -> UserDiscoveryTopicState:
        topic_state = self._topic_state(state, topic_id)
        if topic_state.completed_once or topic_state.fact_count > 0 or topic_state.last_answer_at is not None:
            return topic_state
        curated_coverage = topic_id == "basics" and self._curated_user_profile_has_name()
        authoritative_coverage = self._authoritative_profile_coverage().covers(topic_id)
        if not curated_coverage and not authoritative_coverage:
            return topic_state
        coverage_timestamp = _utc_now().isoformat()
        return replace(
            topic_state,
            completed_once=True,
            last_answer_at=topic_state.last_answer_at or coverage_timestamp,
            last_completed_at=topic_state.last_completed_at or coverage_timestamp,
        )

    def _authoritative_profile_coverage(self) -> UserDiscoveryAuthoritativeCoverage:
        if self._authoritative_coverage_cache is None:
            if self.authoritative_profile_reader is None:
                self._authoritative_coverage_cache = UserDiscoveryAuthoritativeCoverage()
            else:
                self._authoritative_coverage_cache = self.authoritative_profile_reader.load()
        return self._authoritative_coverage_cache

    def _curated_user_profile_has_name(self) -> bool:
        return self._curated_user_profile_name() is not None

    def _curated_user_profile_name(self) -> str | None:
        base_text = self._curated_user_profile_base_text()
        if not base_text:
            return None
        for raw_line in base_text.splitlines():
            field, separator, value = raw_line.partition(":")
            if separator != ":":
                continue
            if field.strip().casefold() != "user":
                continue
            name = _compact_text(value, max_len=80).strip().strip(".")
            if name:
                return name
        return None

    def _curated_user_profile_base_text(self) -> str:
        if self.user_profile_store is None:
            return ""
        try:
            return self.user_profile_store.load_base_text()
        except Exception:
            LOGGER.warning("Failed to read curated USER.md base text for discovery coverage.", exc_info=True)
            return ""

    def _question_brief(
        self,
        definition,
        topic_state: UserDiscoveryTopicState | None,
        phase: str,
        *,
        policy: UserDiscoveryTopicPolicy | None = None,
    ) -> str:
        if topic_state is None:
            topic_state = UserDiscoveryTopicState.empty(definition.topic_id)
        if phase == "lifelong_learning" and topic_state.fact_count > 0:
            variants = definition.follow_up_briefs or definition.opener_briefs
            index = min(len(variants) - 1, topic_state.visit_count % max(1, len(variants)))
            brief = _compact_text(variants[index], max_len=_MAX_BRIEF_LENGTH)
        elif topic_state.question_count <= 0:
            variants = definition.opener_briefs or definition.follow_up_briefs
            brief = _compact_text(variants[0], max_len=_MAX_BRIEF_LENGTH)
        else:
            variants = definition.follow_up_briefs or definition.opener_briefs
            index = min(len(variants) - 1, topic_state.question_count % max(1, len(variants)))
            brief = _compact_text(variants[index], max_len=_MAX_BRIEF_LENGTH)
        if policy is None:
            return brief
        if policy.question_style == "gentle_optional":
            return _compact_text(f"{brief} Keep it especially easy to skip.", max_len=_MAX_BRIEF_LENGTH)
        if policy.question_style == "deeper_follow_up":
            return _compact_text(
                f"{brief} One slightly deeper but still short follow-up is welcome.",
                max_len=_MAX_BRIEF_LENGTH,
            )
        return brief

    def _topic_summary(self, topic_state: UserDiscoveryTopicState | None) -> str | None:
        if topic_state is None:
            return None
        grouped: dict[str, list[str]] = {}
        for fact in topic_state.active_facts:
            grouped.setdefault(fact.route_kind, []).append(fact.review_text)
        parts: list[str] = []
        for route_kind, values in grouped.items():
            summary = _sentence_list(values)
            if summary:
                parts.append(f"{route_kind}: {summary}")
        summary = _compact_text(" ".join(parts), max_len=1200)
        return summary or None

    def _topic_policy(
        self,
        state: UserDiscoveryState,
        topic_id: str | None,
        *,
        now: datetime,
        signals: Mapping[str, UserDiscoveryEngagementSignals] | None = None,
    ) -> UserDiscoveryTopicPolicy:
        normalized_topic_id = self._normalize_topic_id(topic_id)
        if normalized_topic_id is None or self.policy_engine is None:
            return UserDiscoveryTopicPolicy(
                topic_id=normalized_topic_id or "",
                score=0.0,
                engagement_state="neutral",
                question_style="standard",
                invite_style="warm",
                invite_salience_adjustment=0.0,
            )
        definition = _TOPICS_BY_ID[normalized_topic_id]
        topic_state = self._effective_topic_state(state, normalized_topic_id)
        return self.policy_engine.topic_policy(
            topic_id=normalized_topic_id,
            completed_once=topic_state.completed_once,
            fact_count=topic_state.fact_count,
            skip_count=topic_state.skip_count,
            correction_count=topic_state.correction_count + topic_state.deletion_count,
            last_completed_at=topic_state.last_completed_at,
            initial_order=definition.initial_order,
            now=now,
            signals=signals,
        )

    def _review_items(
        self,
        state: UserDiscoveryState,
        *,
        topic_id: str | None,
    ) -> tuple[UserDiscoveryReviewItem, ...]:
        selected_topic_id = self._normalize_topic_id(topic_id)
        items: list[UserDiscoveryReviewItem] = []
        for topic in state.topics:
            if selected_topic_id is not None and topic.topic_id != selected_topic_id:
                continue
            for fact in topic.active_facts:
                items.append(
                    UserDiscoveryReviewItem(
                        fact_id=fact.fact_id,
                        topic_id=topic.topic_id,
                        topic_label=_TOPICS_BY_ID[topic.topic_id].label,
                        route_kind=fact.route_kind,
                        summary=fact.review_text,
                        created_at=fact.created_at,
                        updated_at=fact.updated_at,
                    )
                )
        items.sort(key=lambda item: item.updated_at or "", reverse=True)
        return tuple(items[:24])

    def _profile_summary(self, state: UserDiscoveryState) -> str | None:
        facts: list[str] = []
        for topic in state.topics:
            for fact in topic.active_facts:
                facts.append(fact.review_text)
        summary = _sentence_list(facts[:10])
        return summary or None

    def _active_fact_count(self, state: UserDiscoveryState) -> int:
        return sum(topic.fact_count for topic in state.topics)

    def _correction_total(self, state: UserDiscoveryState) -> int:
        return sum(topic.correction_count + topic.deletion_count for topic in state.topics)

    def _review_due(self, state: UserDiscoveryState, *, now: datetime) -> bool:
        if self.policy_engine is None:
            return self._active_fact_count(state) >= 4 and state.last_reviewed_at is None
        return self.policy_engine.review_due(
            active_fact_count=self._active_fact_count(state),
            correction_total=self._correction_total(state),
            review_count=state.review_count,
            last_reviewed_at=state.last_reviewed_at,
            now=now,
        )

    def _select_review_topic(self, state: UserDiscoveryState, *, now: datetime) -> str | None:
        pressured: list[tuple[tuple[object, ...], str]] = []
        for definition in _TOPIC_DEFINITIONS:
            topic_state = self._topic_state(state, definition.topic_id)
            if topic_state.fact_count <= 0:
                continue
            last_answer = _parse_timestamp(topic_state.last_answer_at) or datetime(1970, 1, 1, tzinfo=timezone.utc)
            key = (
                -(topic_state.correction_count + topic_state.deletion_count),
                -topic_state.fact_count,
                -last_answer.timestamp(),
                definition.initial_order,
            )
            pressured.append((key, definition.topic_id))
        if pressured:
            pressured.sort(key=lambda entry: entry[0])
            return pressured[0][1]
        return self._select_lifelong_topic(state, now=now)
