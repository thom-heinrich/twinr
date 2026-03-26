"""Persist and steer Twinr's guided user-discovery flow.

This module owns the bounded state machine behind Twinr's initial get-to-know-
you setup and the shorter lifelong follow-up runs that continue learning over
time. The LLM still decides what it learned from the user's answer, but this
service keeps the progression, cooldowns, and durable memory commits
structured:

- one resumable initial setup that can span multiple short sessions
- later five-minute follow-up runs that revisit profile gaps and old topics
- predefined topic families instead of ad-hoc question sprawl
- explicit permission gating before sensitive health questions
- high-value facts committed into managed user/personality context
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.context_store import ManagedContextFileStore
from twinr.memory.user_discovery_authoritative_profile import (
    UserDiscoveryAuthoritativeCoverage,
    UserDiscoveryAuthoritativeProfileReader,
)
from twinr.memory.user_discovery_policy import (
    UserDiscoveryEngagementSignals,
    UserDiscoveryPolicyEngine,
    UserDiscoveryTopicPolicy,
)

LOGGER = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_DEFAULT_STATE_PATH = "state/user_discovery.json"
_DEFAULT_USER_PROFILE_PATH = "personality/USER.md"
_DEFAULT_USER_PROFILE_SECTION_TITLE = "Twinr managed user updates"
_INITIAL_SESSION_MINUTES = 15
_LIFELONG_SESSION_MINUTES = 5
_INITIAL_SESSION_ANSWER_LIMIT = 5
_LIFELONG_SESSION_ANSWER_LIMIT = 3
_INITIAL_PAUSE_DELAY = timedelta(hours=18)
_LIFELONG_INVITE_DELAY = timedelta(days=3)
_DEFAULT_INITIAL_SNOOZE_DAYS = 1
_DEFAULT_LIFELONG_SNOOZE_DAYS = 3
_MAX_FACTS_PER_TOPIC = 12
_MAX_FACT_TEXT_LENGTH = 220
_MAX_BRIEF_LENGTH = 240
_ALLOWED_SESSION_STATES = frozenset({"idle", "active", "paused", "snoozed"})
_ALLOWED_PHASES = frozenset({"initial_setup", "lifelong_learning"})
_ALLOWED_FACT_STORAGES = frozenset({"user_profile", "personality"})
_ALLOWED_PERMISSION_STATES = frozenset({"unknown", "granted", "declined"})
_ALLOWED_MEMORY_ROUTE_KINDS = frozenset(
    {"user_profile", "personality", "contact", "preference", "plan", "durable_memory"}
)
_ALLOWED_STORED_FACT_STATUSES = frozenset({"active", "superseded", "deleted"})


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _compact_text(value: object | None, *, max_len: int) -> str:
    compact = " ".join(str(value or "").split()).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "..."


def _normalize_bool(value: object | None) -> bool:
    return bool(value)


def _normalize_int(value: object | None, *, default: int, minimum: int = 0, maximum: int | None = None) -> int:
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def _mapping_or_none(value: object | None) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return value
    return None


def _text_tuple(value: object | None) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(item) for item in value)
    return ()


def _stored_fact_payloads(value: object | None) -> tuple[UserDiscoveryStoredFact | Mapping[str, object], ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            item
            for item in value
            if isinstance(item, (UserDiscoveryStoredFact, Mapping))
        )
    return ()


def _normalize_timestamp(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat()


def _parse_timestamp(value: object | None) -> datetime | None:
    normalized = _normalize_timestamp(value)
    if normalized is None:
        return None
    return datetime.fromisoformat(normalized)


def _normalize_session_state(value: object | None) -> str:
    compact = _compact_text(value, max_len=24).lower().replace("-", "_").replace(" ", "_")
    if compact in _ALLOWED_SESSION_STATES:
        return compact
    return "idle"


def _normalize_phase(value: object | None) -> str:
    compact = _compact_text(value, max_len=32).lower().replace("-", "_").replace(" ", "_")
    if compact in _ALLOWED_PHASES:
        return compact
    return "initial_setup"


def _normalize_permission_state(value: object | None) -> str:
    compact = _compact_text(value, max_len=24).lower().replace("-", "_").replace(" ", "_")
    if compact in _ALLOWED_PERMISSION_STATES:
        return compact
    return "unknown"


def _normalize_route_kind(value: object | None) -> str:
    compact = _compact_text(value, max_len=32).lower().replace("-", "_").replace(" ", "_")
    if compact in _ALLOWED_MEMORY_ROUTE_KINDS:
        return compact
    return "user_profile"


def _normalize_fact_status(value: object | None) -> str:
    compact = _compact_text(value, max_len=24).lower().replace("-", "_").replace(" ", "_")
    if compact in _ALLOWED_STORED_FACT_STATUSES:
        return compact
    return "active"


def _normalize_fact_mapping(value: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, object] = {}
    for raw_key, raw_value in value.items():
        key = _compact_text(raw_key, max_len=40).lower().replace("-", "_").replace(" ", "_")
        if not key or raw_value is None:
            continue
        if isinstance(raw_value, bool):
            normalized[key] = raw_value
            continue
        compact = _compact_text(raw_value, max_len=_MAX_FACT_TEXT_LENGTH)
        if compact:
            normalized[key] = compact
    return normalized


def _fact_id(seed: str) -> str:
    return f"udf_{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:12]}"


def _managed_context_category(topic_id: str) -> str:
    return f"user_discovery_{topic_id}"


def _unique_fact_texts(values: Sequence[object] | None) -> tuple[str, ...]:
    if not values:
        return ()
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        compact = _compact_text(value, max_len=_MAX_FACT_TEXT_LENGTH)
        if not compact:
            continue
        key = compact.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(compact)
        if len(ordered) >= _MAX_FACTS_PER_TOPIC:
            break
    return tuple(ordered)


def _merge_fact_texts(existing: Sequence[str], additions: Sequence[str]) -> tuple[tuple[str, ...], int]:
    ordered = list(_unique_fact_texts(existing))
    seen = {item.casefold() for item in ordered}
    added = 0
    for value in additions:
        compact = _compact_text(value, max_len=_MAX_FACT_TEXT_LENGTH)
        if not compact:
            continue
        key = compact.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(compact)
        added += 1
        if len(ordered) >= _MAX_FACTS_PER_TOPIC:
            break
    return tuple(ordered[:_MAX_FACTS_PER_TOPIC]), added


def _sentence_list(facts: Sequence[str]) -> str:
    parts: list[str] = []
    for fact in facts:
        compact = _compact_text(fact, max_len=_MAX_FACT_TEXT_LENGTH)
        if not compact:
            continue
        if compact[-1] not in ".!?":
            compact += "."
        parts.append(compact)
    return _compact_text(" ".join(parts), max_len=1200)


def _atomic_write_text(path: Path, text: str) -> None:
    parent = path.parent.resolve(strict=False)
    parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(parent),
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temp_path), str(path))
    except Exception:
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise


@dataclass(frozen=True, slots=True)
class UserDiscoveryTopicDefinition:
    """Describe one bounded predefined discovery topic family."""

    topic_id: str
    label: str
    display_label: str
    goal: str
    opener_briefs: tuple[str, ...]
    follow_up_briefs: tuple[str, ...]
    sensitive: bool = False
    permission_brief: str | None = None
    initial_order: int = 0


_TOPIC_DEFINITIONS: tuple[UserDiscoveryTopicDefinition, ...] = (
    UserDiscoveryTopicDefinition(
        topic_id="basics",
        label="Basic Info",
        display_label="Basisinfos",
        goal="Learn the core facts that help Twinr address the user naturally and stay oriented in daily life.",
        opener_briefs=(
            "Ask one short question about the user's preferred name, form of address, or core home situation.",
            "Ask one short question that helps Twinr address the user correctly and understand the basic living situation.",
        ),
        follow_up_briefs=(
            "Ask one short follow-up about a basic orientation detail that would make future small talk or reminders feel more personal.",
            "Ask one short follow-up that clarifies how Twinr should refer to the user or to daily home context.",
        ),
        initial_order=0,
    ),
    UserDiscoveryTopicDefinition(
        topic_id="companion_style",
        label="Companion Style",
        display_label="Ansprache",
        goal="Learn how Twinr should address the user and how proactive, warm, or playful it should sound.",
        opener_briefs=(
            "Ask one short question about how Twinr should address the user and how proactive or reserved it should be.",
            "Ask one short question about preferred tone, form of address, or how often Twinr should actively start conversations.",
        ),
        follow_up_briefs=(
            "Ask one short follow-up about pacing, humor, or whether Twinr should sound more calm, direct, or chatty.",
            "Ask one short follow-up that clarifies what makes Twinr's communication style feel comfortable or uncomfortable.",
        ),
        initial_order=1,
    ),
    UserDiscoveryTopicDefinition(
        topic_id="social",
        label="Social Circle",
        display_label="Soziales Umfeld",
        goal="Learn who matters in the user's social world and which relationships Twinr should keep in mind.",
        opener_briefs=(
            "Ask one short question about important people, family, neighbors, or helpers in the user's life.",
            "Ask one short question about the people the user talks to, meets often, or wants Twinr to remember.",
        ),
        follow_up_briefs=(
            "Ask one short follow-up about who is especially important or how Twinr should talk about those people.",
            "Ask one short follow-up that clarifies which relationships feel supportive or central in daily life.",
        ),
        initial_order=2,
    ),
    UserDiscoveryTopicDefinition(
        topic_id="interests",
        label="Interests",
        display_label="Interessen",
        goal="Learn which broader topics, themes, or kinds of news and conversation spark interest.",
        opener_briefs=(
            "Ask one short question about the topics, themes, or kinds of conversations the user usually enjoys.",
            "Ask one short question about what the user likes hearing, reading, or talking about.",
        ),
        follow_up_briefs=(
            "Ask one short follow-up about why a topic matters or which angle feels especially engaging.",
            "Ask one short follow-up that distinguishes lasting interests from things the user only tolerates occasionally.",
        ),
        initial_order=3,
    ),
    UserDiscoveryTopicDefinition(
        topic_id="hobbies",
        label="Hobbies",
        display_label="Hobbys",
        goal="Learn which activities give the user joy, calm, energy, or a sense of routine.",
        opener_briefs=(
            "Ask one short question about hobbies, regular activities, or small things the user likes doing.",
            "Ask one short question about activities that feel fun, calming, or meaningful to the user.",
        ),
        follow_up_briefs=(
            "Ask one short follow-up about when the user enjoys that activity or what makes it special.",
            "Ask one short follow-up about what helps the user get into that hobby or keep it going.",
        ),
        initial_order=4,
    ),
    UserDiscoveryTopicDefinition(
        topic_id="routines",
        label="Routines",
        display_label="Routinen",
        goal="Learn the user's daily rhythm so Twinr can choose better timing, pacing, and gentle follow-ups.",
        opener_briefs=(
            "Ask one short question about the user's daily rhythm, good times for conversation, or important recurring habits.",
            "Ask one short question about what a normal day looks like and when Twinr should be more or less proactive.",
        ),
        follow_up_briefs=(
            "Ask one short follow-up about mornings, evenings, or times when the user especially likes peace or company.",
            "Ask one short follow-up that clarifies which parts of the day feel busiest, calmest, or most predictable.",
        ),
        initial_order=5,
    ),
    UserDiscoveryTopicDefinition(
        topic_id="pets",
        label="Pets",
        display_label="Haustiere",
        goal="Learn whether pets are part of the user's life and whether they are an important source of joy or daily structure.",
        opener_briefs=(
            "Ask one short question about pets or animals that are part of the user's life or memories.",
            "Ask one short question about whether animals or pets play a role in daily life.",
        ),
        follow_up_briefs=(
            "Ask one short follow-up about the pet's name, personality, or what makes that bond meaningful.",
            "Ask one short follow-up about daily routines, memories, or comfort connected to animals.",
        ),
        initial_order=6,
    ),
    UserDiscoveryTopicDefinition(
        topic_id="no_goes",
        label="No Go Topics",
        display_label="No-Gos",
        goal="Learn which topics, styles, reminders, or behaviors Twinr should avoid or handle carefully.",
        opener_briefs=(
            "Ask one short question about topics, styles, or situations Twinr should avoid or handle more gently.",
            "Ask one short question about what the user does not want Twinr to push, repeat, or bring up casually.",
        ),
        follow_up_briefs=(
            "Ask one short follow-up about how Twinr should notice discomfort and back off gracefully.",
            "Ask one short follow-up that clarifies whether a topic is always off-limits or only needs gentler timing.",
        ),
        initial_order=7,
    ),
    UserDiscoveryTopicDefinition(
        topic_id="health",
        label="Health",
        display_label="Gesundheit",
        goal="Learn only the health-related preferences, limitations, or support needs that the user explicitly wants Twinr to know.",
        opener_briefs=(
            "Ask one short question about health-related preferences, limits, or practical needs only after explicit permission.",
            "Ask one short question about helpful health context only after the user clearly agreed to that topic.",
        ),
        follow_up_briefs=(
            "Ask one short follow-up about practical support needs, comfort boundaries, or useful health-related preferences.",
            "Ask one short follow-up about what Twinr should keep in mind without turning the conversation clinical.",
        ),
        sensitive=True,
        permission_brief=(
            "Before asking health-related questions, explicitly ask whether the user is comfortable sharing a few health-related preferences or limitations."
        ),
        initial_order=8,
    ),
)
_TOPICS_BY_ID = {definition.topic_id: definition for definition in _TOPIC_DEFINITIONS}
_TOPIC_ORDER = tuple(definition.topic_id for definition in sorted(_TOPIC_DEFINITIONS, key=lambda item: item.initial_order))


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
        object.__setattr__(self, "kind", _compact_text(self.kind, max_len=48).lower().replace("-", "_").replace(" ", "_"))

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
            _active_managed_fact_texts(normalized_stored_facts, route_kind="personality") or legacy_personality_facts,
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
        default_limit = _INITIAL_SESSION_ANSWER_LIMIT if self.phase == "initial_setup" else _LIFELONG_SESSION_ANSWER_LIMIT
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


@dataclass(slots=True)
class UserDiscoveryStateStore:
    """Read and write the file-backed user-discovery state."""

    path: Path

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "UserDiscoveryStateStore":
        project_root = Path(config.project_root).expanduser().resolve()
        configured = Path(_DEFAULT_STATE_PATH)
        resolved = configured if configured.is_absolute() else project_root / configured
        return cls(path=resolved)

    def load(self) -> UserDiscoveryState:
        if not self.path.exists():
            return UserDiscoveryState.empty()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            LOGGER.warning("Failed to read user-discovery state from %s.", self.path, exc_info=True)
            return UserDiscoveryState.empty()
        if not isinstance(payload, Mapping):
            return UserDiscoveryState.empty()
        try:
            return UserDiscoveryState.from_dict(payload)
        except Exception:
            LOGGER.warning("Failed to normalize user-discovery state from %s.", self.path, exc_info=True)
            return UserDiscoveryState.empty()

    def save(self, state: UserDiscoveryState) -> UserDiscoveryState:
        payload = json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n"
        _atomic_write_text(self.path, payload)
        return state


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


@dataclass(slots=True)
class UserDiscoveryService:
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

    def load_state(self) -> UserDiscoveryState:
        self._authoritative_coverage_cache = None
        try:
            return self._refresh_phase(self.store.load(), now=_utc_now())
        finally:
            self._authoritative_coverage_cache = None

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
        self._authoritative_coverage_cache = None
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
        try:
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
                return self._delete_fact(state, now=effective_now, fact_id=normalized_fact_id, callbacks=callbacks)
            if normalized_action == "replace_fact":
                if normalized_fact_id is None:
                    raise ValueError("replace_fact requires a fact_id.")
                return self._replace_fact(
                    state,
                    now=effective_now,
                    fact_id=normalized_fact_id,
                    replacement_routes=normalized_routes,
                    callbacks=callbacks,
                )
            raise ValueError(f"Unsupported user discovery action: {action!r}")
        finally:
            self._authoritative_coverage_cache = None

    def build_invitation(self, *, now: datetime | None = None) -> UserDiscoveryInvite | None:
        self._authoritative_coverage_cache = None
        effective_now = (now or _utc_now()).astimezone(timezone.utc)
        state = self._refresh_phase(self.store.load(), now=effective_now)
        try:
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
        finally:
            self._authoritative_coverage_cache = None

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
        updated_topic = self._rebuild_managed_context_targets(topic_id, updated_topic, callbacks=callback_bundle, now=now)
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
        """Return whether the next visible discovery card is an opener or follow-up."""

        topic_state = self._effective_topic_state(state, topic_id)
        if topic_state.fact_count > 0 or topic_state.last_answer_at is not None:
            return "follow_up"
        return "opener"

    def _effective_topic_state(self, state: UserDiscoveryState, topic_id: str) -> UserDiscoveryTopicState:
        """Return topic state plus authoritative profile-coverage overlays."""

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
        """Load authoritative discovery coverage once per public service call."""

        if self._authoritative_coverage_cache is None:
            if self.authoritative_profile_reader is None:
                self._authoritative_coverage_cache = UserDiscoveryAuthoritativeCoverage()
            else:
                self._authoritative_coverage_cache = self.authoritative_profile_reader.load()
        return self._authoritative_coverage_cache

    def _curated_user_profile_has_name(self) -> bool:
        """Return whether the curated USER.md base text already names the user."""

        return self._curated_user_profile_name() is not None

    def _curated_user_profile_name(self) -> str | None:
        """Return the curated USER.md display name when one explicit `User:` line exists."""

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
        """Load the authored USER.md base text outside Twinr-managed updates."""

        if self.user_profile_store is None:
            return ""
        try:
            return self.user_profile_store.load_base_text()
        except Exception:
            LOGGER.warning("Failed to read curated USER.md base text for discovery coverage.", exc_info=True)
            return ""

    def _question_brief(
        self,
        definition: UserDiscoveryTopicDefinition,
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


__all__ = [
    "UserDiscoveryCommitCallbacks",
    "UserDiscoveryFact",
    "UserDiscoveryInvite",
    "UserDiscoveryMemoryRoute",
    "UserDiscoveryResult",
    "UserDiscoveryReviewItem",
    "UserDiscoveryService",
    "UserDiscoveryState",
    "UserDiscoveryStateStore",
    "UserDiscoveryStoredFact",
    "UserDiscoveryTopicState",
]
