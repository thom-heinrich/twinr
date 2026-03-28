"""Data models for guided user discovery."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime

from .common import (
    _ALLOWED_FACT_STORAGES,
    _INITIAL_SESSION_ANSWER_LIMIT,
    _LIFELONG_SESSION_ANSWER_LIMIT,
    _MAX_FACT_TEXT_LENGTH,
    _SCHEMA_VERSION,
    _TOPICS_BY_ID,
    _compact_text,
    _fact_id,
    _managed_context_category,
    _mapping_or_none,
    _normalize_bool,
    _normalize_fact_mapping,
    _normalize_fact_status,
    _normalize_int,
    _normalize_permission_state,
    _normalize_phase,
    _normalize_route_kind,
    _normalize_session_state,
    _normalize_timestamp,
    _text_tuple,
    _unique_fact_texts,
)


@dataclass(frozen=True, slots=True)
class UserDiscoveryFact:
    """Describe one compact learned fact extracted by the LLM."""

    storage: str
    text: str

    def __post_init__(self) -> None:
        storage = _compact_text(self.storage, max_len=24).lower().replace("-", "_").replace(" ", "_")
        if storage not in _ALLOWED_FACT_STORAGES:
            storage = "user_profile"
        object.__setattr__(self, "storage", storage)
        object.__setattr__(self, "text", _compact_text(self.text, max_len=_MAX_FACT_TEXT_LENGTH))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "UserDiscoveryFact":
        return cls(
            storage=_compact_text(payload.get("storage"), max_len=24) or "user_profile",
            text=_compact_text(payload.get("text"), max_len=_MAX_FACT_TEXT_LENGTH),
        )

    def is_empty(self) -> bool:
        return not bool(self.text)


@dataclass(frozen=True, slots=True)
class UserDiscoveryMemoryRoute:
    """Describe one structured discovery memory write."""

    route_kind: str
    text: str = ""
    category: str = ""
    given_name: str = ""
    family_name: str = ""
    phone: str = ""
    email: str = ""
    role: str = ""
    relation: str = ""
    notes: str = ""
    value: str = ""
    sentiment: str = ""
    for_product: str = ""
    summary: str = ""
    when_text: str = ""
    details: str = ""
    kind: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "route_kind", _normalize_route_kind(self.route_kind))
        object.__setattr__(self, "text", _compact_text(self.text, max_len=_MAX_FACT_TEXT_LENGTH))
        object.__setattr__(self, "category", _compact_text(self.category, max_len=48))
        object.__setattr__(self, "given_name", _compact_text(self.given_name, max_len=80))
        object.__setattr__(self, "family_name", _compact_text(self.family_name, max_len=80))
        object.__setattr__(self, "phone", _compact_text(self.phone, max_len=48))
        object.__setattr__(self, "email", _compact_text(self.email, max_len=120))
        object.__setattr__(self, "role", _compact_text(self.role, max_len=80))
        object.__setattr__(self, "relation", _compact_text(self.relation, max_len=80))
        object.__setattr__(self, "notes", _compact_text(self.notes, max_len=160))
        object.__setattr__(self, "value", _compact_text(self.value, max_len=120))
        object.__setattr__(self, "sentiment", _compact_text(self.sentiment, max_len=24).lower())
        object.__setattr__(self, "for_product", _compact_text(self.for_product, max_len=80))
        object.__setattr__(self, "summary", _compact_text(self.summary, max_len=160))
        object.__setattr__(self, "when_text", _compact_text(self.when_text, max_len=80))
        object.__setattr__(self, "details", _compact_text(self.details, max_len=220))
        object.__setattr__(
            self,
            "kind",
            _compact_text(self.kind, max_len=48).lower().replace("-", "_").replace(" ", "_"),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "UserDiscoveryMemoryRoute":
        return cls(
            route_kind=_compact_text(payload.get("route_kind"), max_len=32),
            text=_compact_text(payload.get("text"), max_len=_MAX_FACT_TEXT_LENGTH),
            category=_compact_text(payload.get("category"), max_len=48),
            given_name=_compact_text(payload.get("given_name"), max_len=80),
            family_name=_compact_text(payload.get("family_name"), max_len=80),
            phone=_compact_text(payload.get("phone"), max_len=48),
            email=_compact_text(payload.get("email"), max_len=120),
            role=_compact_text(payload.get("role"), max_len=80),
            relation=_compact_text(payload.get("relation"), max_len=80),
            notes=_compact_text(payload.get("notes"), max_len=160),
            value=_compact_text(payload.get("value"), max_len=120),
            sentiment=_compact_text(payload.get("sentiment"), max_len=24),
            for_product=_compact_text(payload.get("for_product"), max_len=80),
            summary=_compact_text(payload.get("summary"), max_len=160),
            when_text=_compact_text(payload.get("when_text"), max_len=80),
            details=_compact_text(payload.get("details"), max_len=220),
            kind=_compact_text(payload.get("kind"), max_len=48),
        )

    @classmethod
    def from_legacy_fact(cls, fact: UserDiscoveryFact) -> "UserDiscoveryMemoryRoute":
        return cls(route_kind=fact.storage, text=fact.text)

    def is_empty(self) -> bool:
        if self.route_kind in {"user_profile", "personality"}:
            return not bool(self.text)
        if self.route_kind == "contact":
            return not bool(self.given_name)
        if self.route_kind == "preference":
            return not bool(self.category and self.value)
        if self.route_kind == "plan":
            return not bool(self.summary)
        if self.route_kind == "durable_memory":
            return not bool(self.kind and self.summary)
        return True

    def dedupe_key(self) -> tuple[str, ...]:
        if self.route_kind in {"user_profile", "personality"}:
            return (self.route_kind, self.text.casefold())
        if self.route_kind == "contact":
            return (
                self.route_kind,
                self.given_name.casefold(),
                self.family_name.casefold(),
                self.relation.casefold(),
                self.role.casefold(),
                self.phone.casefold(),
                self.email.casefold(),
            )
        if self.route_kind == "preference":
            return (
                self.route_kind,
                self.category.casefold(),
                self.value.casefold(),
                self.sentiment.casefold(),
                self.for_product.casefold(),
            )
        if self.route_kind == "plan":
            return (self.route_kind, self.summary.casefold(), self.when_text.casefold())
        return (self.route_kind, self.kind.casefold(), self.summary.casefold())

    def review_summary(self) -> str:
        if self.route_kind in {"user_profile", "personality"}:
            return self.text
        if self.route_kind == "contact":
            name = " ".join(part for part in (self.given_name, self.family_name) if part).strip()
            if self.relation:
                return _compact_text(f"Important person: {name} ({self.relation}).", max_len=_MAX_FACT_TEXT_LENGTH)
            if self.role:
                return _compact_text(f"Important person: {name} ({self.role}).", max_len=_MAX_FACT_TEXT_LENGTH)
            return _compact_text(f"Important person: {name}.", max_len=_MAX_FACT_TEXT_LENGTH)
        if self.route_kind == "preference":
            sentiment = self.sentiment or "prefer"
            product = f" for {self.for_product}" if self.for_product else ""
            if sentiment in {"avoid", "dislike"}:
                return _compact_text(
                    f"User prefers to avoid {self.value}{product} in category {self.category}.",
                    max_len=_MAX_FACT_TEXT_LENGTH,
                )
            return _compact_text(
                f"User prefers {self.value}{product} in category {self.category}.",
                max_len=_MAX_FACT_TEXT_LENGTH,
            )
        if self.route_kind == "plan":
            when_text = f" ({self.when_text})" if self.when_text else ""
            return _compact_text(f"Future plan: {self.summary}{when_text}.", max_len=_MAX_FACT_TEXT_LENGTH)
        return _compact_text(f"Durable memory: {self.summary}.", max_len=_MAX_FACT_TEXT_LENGTH)

    def payload(self) -> dict[str, object]:
        payload = {
            "text": self.text,
            "category": self.category,
            "given_name": self.given_name,
            "family_name": self.family_name,
            "phone": self.phone,
            "email": self.email,
            "role": self.role,
            "relation": self.relation,
            "notes": self.notes,
            "value": self.value,
            "sentiment": self.sentiment,
            "for_product": self.for_product,
            "summary": self.summary,
            "when_text": self.when_text,
            "details": self.details,
            "kind": self.kind,
        }
        return {key: value for key, value in payload.items() if value}


@dataclass(frozen=True, slots=True)
class UserDiscoveryStoredFact:
    """Persist one reviewable discovery fact plus commit references."""

    fact_id: str
    route_kind: str
    review_text: str
    payload: Mapping[str, object] | None = None
    commit_ref: Mapping[str, object] | None = None
    status: str = "active"
    created_at: str | None = None
    updated_at: str | None = None

    def __post_init__(self) -> None:
        fact_id = _compact_text(self.fact_id, max_len=40)
        if not fact_id:
            fact_id = _fact_id(self.review_text or self.route_kind or "fact")
        object.__setattr__(self, "fact_id", fact_id)
        object.__setattr__(self, "route_kind", _normalize_route_kind(self.route_kind))
        object.__setattr__(self, "review_text", _compact_text(self.review_text, max_len=_MAX_FACT_TEXT_LENGTH))
        object.__setattr__(self, "payload", _normalize_fact_mapping(self.payload))
        object.__setattr__(self, "commit_ref", _normalize_fact_mapping(self.commit_ref))
        object.__setattr__(self, "status", _normalize_fact_status(self.status))
        object.__setattr__(self, "created_at", _normalize_timestamp(self.created_at))
        object.__setattr__(self, "updated_at", _normalize_timestamp(self.updated_at))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "UserDiscoveryStoredFact":
        return cls(
            fact_id=_compact_text(payload.get("fact_id"), max_len=40),
            route_kind=_compact_text(payload.get("route_kind"), max_len=32),
            review_text=_compact_text(payload.get("review_text"), max_len=_MAX_FACT_TEXT_LENGTH),
            payload=_mapping_or_none(payload.get("payload")),
            commit_ref=_mapping_or_none(payload.get("commit_ref")),
            status=_compact_text(payload.get("status"), max_len=24),
            created_at=_normalize_timestamp(payload.get("created_at")),
            updated_at=_normalize_timestamp(payload.get("updated_at")),
        )

    @classmethod
    def from_route(
        cls,
        route: UserDiscoveryMemoryRoute,
        *,
        topic_id: str,
        now: datetime,
    ) -> "UserDiscoveryStoredFact":
        seed = f"{topic_id}:{route.route_kind}:{route.review_summary()}:{now.isoformat()}"
        return cls(
            fact_id=_fact_id(seed),
            route_kind=route.route_kind,
            review_text=route.review_summary(),
            payload=route.payload(),
            commit_ref={},
            status="active",
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
        )

    def to_route(self) -> UserDiscoveryMemoryRoute:
        payload = dict(self.payload or {})
        payload["route_kind"] = self.route_kind
        return UserDiscoveryMemoryRoute.from_dict(payload)

    def dedupe_key(self) -> tuple[str, ...]:
        return self.to_route().dedupe_key()

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    def to_dict(self) -> dict[str, object]:
        return {
            "fact_id": self.fact_id,
            "route_kind": self.route_kind,
            "review_text": self.review_text,
            "payload": dict(self.payload or {}),
            "commit_ref": dict(self.commit_ref or {}),
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def _stored_fact_payloads(
    value: object | None,
) -> tuple[UserDiscoveryStoredFact | Mapping[str, object], ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for item in value if isinstance(item, (UserDiscoveryStoredFact, Mapping)))
    return ()


def _migrate_legacy_stored_facts(
    topic_id: str,
    *,
    profile_facts: Sequence[str],
    personality_facts: Sequence[str],
) -> tuple[UserDiscoveryStoredFact, ...]:
    category = _managed_context_category(topic_id)
    migrated: list[UserDiscoveryStoredFact] = []
    for route_kind, values in (("user_profile", profile_facts), ("personality", personality_facts)):
        for value in values:
            review_text = _compact_text(value, max_len=_MAX_FACT_TEXT_LENGTH)
            if not review_text:
                continue
            seed = f"{topic_id}:{route_kind}:{review_text}"
            migrated.append(
                UserDiscoveryStoredFact(
                    fact_id=_fact_id(seed),
                    route_kind=route_kind,
                    review_text=review_text,
                    payload={"text": review_text},
                    commit_ref={"category": category},
                    status="active",
                )
            )
    return tuple(migrated)


def _active_managed_fact_texts(
    stored_facts: Sequence[UserDiscoveryStoredFact],
    *,
    route_kind: str,
) -> tuple[str, ...]:
    values = [
        _compact_text((fact.payload or {}).get("text"), max_len=_MAX_FACT_TEXT_LENGTH)
        for fact in stored_facts
        if fact.is_active and fact.route_kind == route_kind
    ]
    return _unique_fact_texts(values)


@dataclass(frozen=True, slots=True)
class UserDiscoveryReviewItem:
    """Describe one reviewable discovery fact returned to the model."""

    fact_id: str
    topic_id: str
    topic_label: str
    route_kind: str
    summary: str
    created_at: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class UserDiscoveryTopicState:
    """Persist one topic's accumulated coverage and reviewable fact state."""

    topic_id: str
    question_count: int = 0
    visit_count: int = 0
    completed_once: bool = False
    skip_count: int = 0
    correction_count: int = 0
    deletion_count: int = 0
    permission_state: str = "unknown"
    profile_facts: tuple[str, ...] = ()
    personality_facts: tuple[str, ...] = ()
    stored_facts: tuple[UserDiscoveryStoredFact, ...] = ()
    last_question_at: str | None = None
    last_answer_at: str | None = None
    last_completed_at: str | None = None

    def __post_init__(self) -> None:
        topic_id = _compact_text(self.topic_id, max_len=48).lower().replace("-", "_").replace(" ", "_")
        if topic_id not in _TOPICS_BY_ID:
            raise ValueError(f"Unsupported discovery topic: {self.topic_id!r}")
        object.__setattr__(self, "topic_id", topic_id)
        object.__setattr__(self, "question_count", _normalize_int(self.question_count, default=0, minimum=0))
        object.__setattr__(self, "visit_count", _normalize_int(self.visit_count, default=0, minimum=0))
        object.__setattr__(self, "completed_once", _normalize_bool(self.completed_once))
        object.__setattr__(self, "skip_count", _normalize_int(self.skip_count, default=0, minimum=0))
        object.__setattr__(self, "correction_count", _normalize_int(self.correction_count, default=0, minimum=0))
        object.__setattr__(self, "deletion_count", _normalize_int(self.deletion_count, default=0, minimum=0))
        object.__setattr__(self, "permission_state", _normalize_permission_state(self.permission_state))
        legacy_profile_facts = _unique_fact_texts(self.profile_facts)
        legacy_personality_facts = _unique_fact_texts(self.personality_facts)
        normalized_stored_facts: list[UserDiscoveryStoredFact] = []
        seen_fact_ids: set[str] = set()
        for fact in self.stored_facts:
            normalized = fact if isinstance(fact, UserDiscoveryStoredFact) else UserDiscoveryStoredFact.from_dict(fact)
            if normalized.fact_id in seen_fact_ids:
                continue
            seen_fact_ids.add(normalized.fact_id)
            normalized_stored_facts.append(normalized)
        if not normalized_stored_facts and (legacy_profile_facts or legacy_personality_facts):
            normalized_stored_facts.extend(
                _migrate_legacy_stored_facts(
                    topic_id,
                    profile_facts=legacy_profile_facts,
                    personality_facts=legacy_personality_facts,
                )
            )
        object.__setattr__(self, "stored_facts", tuple(normalized_stored_facts))
        object.__setattr__(
            self,
            "profile_facts",
            _active_managed_fact_texts(normalized_stored_facts, route_kind="user_profile") or legacy_profile_facts,
        )
        object.__setattr__(
            self,
            "personality_facts",
            _active_managed_fact_texts(normalized_stored_facts, route_kind="personality")
            or legacy_personality_facts,
        )
        object.__setattr__(self, "last_question_at", _normalize_timestamp(self.last_question_at))
        object.__setattr__(self, "last_answer_at", _normalize_timestamp(self.last_answer_at))
        object.__setattr__(self, "last_completed_at", _normalize_timestamp(self.last_completed_at))

    @classmethod
    def empty(cls, topic_id: str) -> "UserDiscoveryTopicState":
        return cls(topic_id=topic_id)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "UserDiscoveryTopicState":
        return cls(
            topic_id=_compact_text(payload.get("topic_id"), max_len=48),
            question_count=_normalize_int(payload.get("question_count"), default=0, minimum=0),
            visit_count=_normalize_int(payload.get("visit_count"), default=0, minimum=0),
            completed_once=_normalize_bool(payload.get("completed_once", False)),
            skip_count=_normalize_int(payload.get("skip_count"), default=0, minimum=0),
            correction_count=_normalize_int(payload.get("correction_count"), default=0, minimum=0),
            deletion_count=_normalize_int(payload.get("deletion_count"), default=0, minimum=0),
            permission_state=_normalize_permission_state(payload.get("permission_state", "unknown")),
            profile_facts=_text_tuple(payload.get("profile_facts")),
            personality_facts=_text_tuple(payload.get("personality_facts")),
            stored_facts=tuple(
                UserDiscoveryStoredFact.from_dict(item) if isinstance(item, Mapping) else item
                for item in _stored_fact_payloads(payload.get("stored_facts"))
            ),
            last_question_at=_normalize_timestamp(payload.get("last_question_at")),
            last_answer_at=_normalize_timestamp(payload.get("last_answer_at")),
            last_completed_at=_normalize_timestamp(payload.get("last_completed_at")),
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["stored_facts"] = [fact.to_dict() for fact in self.stored_facts]
        return payload

    @property
    def active_facts(self) -> tuple[UserDiscoveryStoredFact, ...]:
        return tuple(fact for fact in self.stored_facts if fact.is_active)

    @property
    def fact_count(self) -> int:
        return len(self.active_facts)


@dataclass(frozen=True, slots=True)
class UserDiscoveryState:
    """Persist the resumable onboarding plus lifelong-learning session state."""

    schema_version: int = _SCHEMA_VERSION
    phase: str = "initial_setup"
    session_state: str = "idle"
    active_topic_id: str | None = None
    session_started_at: str | None = None
    session_answer_count: int = 0
    session_answer_limit: int = _INITIAL_SESSION_ANSWER_LIMIT
    last_interaction_at: str | None = None
    last_session_closed_at: str | None = None
    next_invite_after: str | None = None
    setup_completed_at: str | None = None
    accepted_session_count: int = 0
    review_count: int = 0
    last_reviewed_at: str | None = None
    snooze_count: int = 0
    topics: tuple[UserDiscoveryTopicState, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _normalize_int(self.schema_version, default=_SCHEMA_VERSION, minimum=1))
        object.__setattr__(self, "phase", _normalize_phase(self.phase))
        object.__setattr__(self, "session_state", _normalize_session_state(self.session_state))
        active_topic_id = self.active_topic_id
        if active_topic_id is not None:
            normalized_topic_id = _compact_text(active_topic_id, max_len=48).lower().replace("-", "_").replace(" ", "_")
            active_topic_id = normalized_topic_id if normalized_topic_id in _TOPICS_BY_ID else None
        object.__setattr__(self, "active_topic_id", active_topic_id)
        object.__setattr__(self, "session_started_at", _normalize_timestamp(self.session_started_at))
        object.__setattr__(self, "session_answer_count", _normalize_int(self.session_answer_count, default=0, minimum=0))
        default_limit = (
            _INITIAL_SESSION_ANSWER_LIMIT
            if self.phase == "initial_setup"
            else _LIFELONG_SESSION_ANSWER_LIMIT
        )
        object.__setattr__(
            self,
            "session_answer_limit",
            _normalize_int(self.session_answer_limit, default=default_limit, minimum=1, maximum=8),
        )
        object.__setattr__(self, "last_interaction_at", _normalize_timestamp(self.last_interaction_at))
        object.__setattr__(self, "last_session_closed_at", _normalize_timestamp(self.last_session_closed_at))
        object.__setattr__(self, "next_invite_after", _normalize_timestamp(self.next_invite_after))
        object.__setattr__(self, "setup_completed_at", _normalize_timestamp(self.setup_completed_at))
        object.__setattr__(self, "accepted_session_count", _normalize_int(self.accepted_session_count, default=0, minimum=0))
        object.__setattr__(self, "review_count", _normalize_int(self.review_count, default=0, minimum=0))
        object.__setattr__(self, "last_reviewed_at", _normalize_timestamp(self.last_reviewed_at))
        object.__setattr__(self, "snooze_count", _normalize_int(self.snooze_count, default=0, minimum=0))

        normalized_topics: list[UserDiscoveryTopicState] = []
        seen: set[str] = set()
        for topic in self.topics:
            normalized = topic if isinstance(topic, UserDiscoveryTopicState) else UserDiscoveryTopicState.from_dict(topic)
            if normalized.topic_id in seen:
                continue
            seen.add(normalized.topic_id)
            normalized_topics.append(normalized)
        object.__setattr__(self, "topics", tuple(normalized_topics))

    @classmethod
    def empty(cls) -> "UserDiscoveryState":
        return cls()

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "UserDiscoveryState":
        raw_topics = payload.get("topics")
        topics: list[UserDiscoveryTopicState] = []
        if isinstance(raw_topics, Sequence) and not isinstance(raw_topics, (str, bytes, bytearray)):
            for item in raw_topics:
                if isinstance(item, Mapping):
                    try:
                        topics.append(UserDiscoveryTopicState.from_dict(item))
                    except Exception:
                        continue
        return cls(
            schema_version=_normalize_int(payload.get("schema_version"), default=_SCHEMA_VERSION, minimum=1),
            phase=_normalize_phase(payload.get("phase", "initial_setup")),
            session_state=_normalize_session_state(payload.get("session_state", "idle")),
            active_topic_id=_compact_text(payload.get("active_topic_id"), max_len=48) or None,
            session_started_at=_normalize_timestamp(payload.get("session_started_at")),
            session_answer_count=_normalize_int(payload.get("session_answer_count"), default=0, minimum=0),
            session_answer_limit=_normalize_int(
                payload.get("session_answer_limit"),
                default=_INITIAL_SESSION_ANSWER_LIMIT,
                minimum=1,
                maximum=8,
            ),
            last_interaction_at=_normalize_timestamp(payload.get("last_interaction_at")),
            last_session_closed_at=_normalize_timestamp(payload.get("last_session_closed_at")),
            next_invite_after=_normalize_timestamp(payload.get("next_invite_after")),
            setup_completed_at=_normalize_timestamp(payload.get("setup_completed_at")),
            accepted_session_count=_normalize_int(payload.get("accepted_session_count"), default=0, minimum=0),
            review_count=_normalize_int(payload.get("review_count"), default=0, minimum=0),
            last_reviewed_at=_normalize_timestamp(payload.get("last_reviewed_at")),
            snooze_count=_normalize_int(payload.get("snooze_count"), default=0, minimum=0),
            topics=tuple(topics),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "phase": self.phase,
            "session_state": self.session_state,
            "active_topic_id": self.active_topic_id,
            "session_started_at": self.session_started_at,
            "session_answer_count": self.session_answer_count,
            "session_answer_limit": self.session_answer_limit,
            "last_interaction_at": self.last_interaction_at,
            "last_session_closed_at": self.last_session_closed_at,
            "next_invite_after": self.next_invite_after,
            "setup_completed_at": self.setup_completed_at,
            "accepted_session_count": self.accepted_session_count,
            "review_count": self.review_count,
            "last_reviewed_at": self.last_reviewed_at,
            "snooze_count": self.snooze_count,
            "topics": [topic.to_dict() for topic in self.topics],
        }


@dataclass(frozen=True, slots=True)
class UserDiscoveryCommitCallbacks:
    """Expose the runtime commit hooks used for durable discovery writes."""

    update_user_profile: Callable[[str, str], object] | None = None
    delete_user_profile: Callable[[str], object] | None = None
    update_personality: Callable[[str, str], object] | None = None
    delete_personality: Callable[[str], object] | None = None
    remember_contact: Callable[..., object] | None = None
    delete_contact: Callable[[str], object] | None = None
    remember_preference: Callable[..., object] | None = None
    delete_preference: Callable[[str, str | None], object] | None = None
    remember_plan: Callable[..., object] | None = None
    delete_plan: Callable[[str], object] | None = None
    store_durable_memory: Callable[..., object] | None = None
    delete_durable_memory: Callable[[str], object] | None = None


@dataclass(frozen=True, slots=True)
class UserDiscoveryInvite:
    """Describe one display-side invitation into the discovery flow."""

    invite_kind: str
    phase: str
    topic_id: str
    topic_label: str
    display_topic_label: str
    session_minutes: int
    headline: str
    body: str
    salience: float
    reason: str
    display_prompt_stage: str = "opener"


@dataclass(frozen=True, slots=True)
class UserDiscoveryResult:
    """Describe the next structured step in the guided discovery flow."""

    phase: str
    session_state: str
    response_mode: str
    topic_id: str | None = None
    topic_label: str | None = None
    display_topic_label: str | None = None
    topic_goal: str | None = None
    assistant_brief: str | None = None
    question_brief: str | None = None
    current_topic_summary: str | None = None
    session_minutes: int = 0
    session_answers_used: int = 0
    session_answers_remaining: int = 0
    facts_saved: int = 0
    facts_deleted: int = 0
    facts_replaced: int = 0
    saved_targets: tuple[str, ...] = ()
    review_items: tuple[UserDiscoveryReviewItem, ...] = ()
    question_style: str | None = None
    engagement_state: str | None = None
    can_pause: bool = True
    can_skip_topic: bool = True
    sensitive_permission_required: bool = False
    setup_topics_completed: int = 0
    setup_topics_total: int = 0
    setup_complete: bool = False
    next_invite_after: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.saved_targets:
            payload["saved_targets"] = list(self.saved_targets)
        if self.review_items:
            payload["review_items"] = [item.to_dict() for item in self.review_items]
        return payload
