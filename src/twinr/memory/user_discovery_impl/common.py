"""Shared catalog, constants, and helpers for guided user discovery."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import logging
import os
from pathlib import Path
import tempfile

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


def _normalize_int(
    value: object | None,
    *,
    default: int,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
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


def _fsync_directory(path: Path) -> None:
    """Flush a directory entry after replacing a user-discovery state file."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(path, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_write_text(path: Path, text: str) -> None:
    cross_service_read_mode = 0o644
    parent = path.parent.resolve(strict=False)
    parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_dir():
        raise IsADirectoryError(f"User-discovery state path points to a directory: {path}")
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
            os.fchmod(handle.fileno(), cross_service_read_mode)
            os.fsync(handle.fileno())
        os.replace(str(temp_path), str(path))
        os.chmod(path, cross_service_read_mode)
        _fsync_directory(parent)
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
_TOPIC_ORDER = tuple(
    definition.topic_id for definition in sorted(_TOPIC_DEFINITIONS, key=lambda item: item.initial_order)
)
