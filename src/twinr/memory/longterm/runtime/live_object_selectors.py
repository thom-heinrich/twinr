"""Shared query-first object selectors for live-near runtime flows.

This module centralizes the bounded ``select_fast_topic_objects`` views used by
runtime callers that must not hydrate full remote object snapshots just to
derive a small live-facing object set.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import inspect
from typing import TypeVar, cast

from twinr.memory.longterm.core.models import LongTermMemoryObjectV1

T = TypeVar("T")

_DISCOVERY_OBJECT_QUERY_LIMIT = 8
_DISCOVERY_OBJECT_QUERY_TIMEOUT_S = 2.0
_PROACTIVE_OBJECT_QUERY_LIMIT = 16
_REFLECTION_SUMMARY_QUERY_LIMIT = 24
_REFLECTION_SOURCE_QUERY_LIMIT = 32
_SENSOR_MEMORY_QUERY_LIMIT = 64
_RESTART_RECALL_QUERY_LIMIT = 48
_LIVE_OBJECT_STATUSES = frozenset({"active", "candidate", "uncertain"})
_SENSOR_NEIGHBORHOOD_DOMAINS = frozenset(
    {
        "sensor_routine",
        "smart_home_environment",
        "respeaker_audio_preference",
        "respeaker_audio_routine",
    }
)


@dataclass(frozen=True, slots=True)
class _FastTopicSelection:
    """Describe one bounded fast-topic query."""

    query_text: str
    limit: int


def _bounded_limit(value: object, *, default: int) -> int:
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, limit)


def _select_fast_topic_union(
    object_store: object | None,
    *,
    selections: tuple[_FastTopicSelection, ...],
    timeout_s: float | None = None,
) -> tuple[T, ...]:
    """Run a bounded union of fast-topic queries without snapshot hydration."""

    if object_store is None:
        return ()
    fast_selector = getattr(object_store, "select_fast_topic_objects", None)
    if not callable(fast_selector):
        return ()
    try:
        selector_parameters = inspect.signature(fast_selector).parameters
    except (TypeError, ValueError):
        selector_parameters = {}
    supports_timeout = "timeout_s" in selector_parameters
    selected_by_id: dict[str, T] = {}
    for selection in selections:
        selector_kwargs = {
            "query_text": selection.query_text,
            "limit": _bounded_limit(selection.limit, default=1),
        }
        if supports_timeout and timeout_s is not None:
            selector_kwargs["timeout_s"] = timeout_s
        for item in fast_selector(**selector_kwargs):
            memory_id = getattr(item, "memory_id", None)
            if not isinstance(memory_id, str) or not memory_id or memory_id in selected_by_id:
                continue
            selected_by_id[memory_id] = cast(T, item)
    return tuple(selected_by_id.values())


def _normalize_text(value: object | None) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalize_seed_objects(
    seed_objects: Iterable[LongTermMemoryObjectV1],
) -> tuple[LongTermMemoryObjectV1, ...]:
    selected_by_id: dict[str, LongTermMemoryObjectV1] = {}
    for item in seed_objects:
        if not isinstance(item, LongTermMemoryObjectV1):
            continue
        memory_id = _normalize_text(item.memory_id)
        if memory_id and memory_id not in selected_by_id:
            selected_by_id[memory_id] = item
    return tuple(selected_by_id.values())


def _object_attributes(item: LongTermMemoryObjectV1) -> Mapping[str, object]:
    attributes = getattr(item, "attributes", None)
    if isinstance(attributes, Mapping):
        return attributes
    return {}


def _projection_attributes(projection: Mapping[str, object]) -> Mapping[str, object]:
    attributes = projection.get("attributes")
    if isinstance(attributes, Mapping):
        return attributes
    return {}


def _mapping_text(mapping: Mapping[str, object], field_name: str) -> str:
    value = mapping.get(field_name)
    return _normalize_text(value) if isinstance(value, str) else ""


def _mapping_text_list(mapping: Mapping[str, object], field_name: str) -> tuple[str, ...]:
    values = mapping.get(field_name)
    if not isinstance(values, list):
        return ()
    return tuple(
        normalized
        for normalized in (_normalize_text(value) for value in values if isinstance(value, str))
        if normalized
    )


def _deduped_texts(values: Iterable[object]) -> tuple[str, ...]:
    ordered: dict[str, None] = {}
    for value in values:
        normalized = _normalize_text(value)
        if normalized:
            ordered.setdefault(normalized, None)
    return tuple(ordered)


def _seed_event_ids(seed_objects: Iterable[LongTermMemoryObjectV1]) -> tuple[str, ...]:
    event_ids: list[str] = []
    for item in seed_objects:
        source = getattr(item, "source", None)
        source_event_ids = getattr(source, "event_ids", ())
        if isinstance(source_event_ids, tuple):
            event_ids.extend(source_event_ids)
    return _deduped_texts(event_ids)


def _seed_person_refs(seed_objects: Iterable[LongTermMemoryObjectV1]) -> tuple[str, ...]:
    refs: list[str] = []
    for item in seed_objects:
        attributes = _object_attributes(item)
        for field_name in ("person_ref", "subject_ref", "object_ref"):
            value = _normalize_text(attributes.get(field_name))
            if value.startswith("person:"):
                refs.append(value)
    return _deduped_texts(refs)


def _seed_attribute_values(
    seed_objects: Iterable[LongTermMemoryObjectV1],
    field_name: str,
) -> tuple[str, ...]:
    values = (
        _normalize_text(_object_attributes(item).get(field_name))
        for item in seed_objects
    )
    return _deduped_texts(value for value in values if value)


def _projection_has_live_status(projection: Mapping[str, object]) -> bool:
    return _mapping_text(projection, "status") in _LIVE_OBJECT_STATUSES


def _projection_matches_person_neighborhood(
    projection: Mapping[str, object],
    *,
    person_refs: frozenset[str],
) -> bool:
    if not person_refs:
        return False
    attributes = _projection_attributes(projection)
    for field_name in ("person_ref", "subject_ref", "object_ref"):
        value = _mapping_text(attributes, field_name)
        if value in person_refs:
            return True
    if _mapping_text(attributes, "summary_type") == "thread":
        thread_value_key = _mapping_text(projection, "value_key")
        if thread_value_key in person_refs:
            return True
    return False


def _projection_matches_environment_neighborhood(
    projection: Mapping[str, object],
    *,
    environment_ids: frozenset[str],
    include_environment_domain: bool,
) -> bool:
    if not include_environment_domain and not environment_ids:
        return False
    attributes = _projection_attributes(projection)
    if _mapping_text(attributes, "environment_id") in environment_ids:
        return True
    if not include_environment_domain:
        return False
    return (
        _mapping_text(attributes, "memory_domain") == "smart_home_environment"
        or _mapping_text(attributes, "summary_type").startswith("environment")
        or _mapping_text(attributes, "pattern_type").startswith("environment")
    )


def _projection_matches_sensor_neighborhood(
    projection: Mapping[str, object],
    *,
    sensor_domains: frozenset[str],
    summary_types: frozenset[str],
    pattern_types: frozenset[str],
    routine_types: frozenset[str],
    environment_ids: frozenset[str],
) -> bool:
    if not sensor_domains and not summary_types and not pattern_types and not routine_types and not environment_ids:
        return False
    attributes = _projection_attributes(projection)
    if _mapping_text(attributes, "memory_domain") in sensor_domains:
        return True
    if _mapping_text(attributes, "summary_type") in summary_types:
        return True
    if _mapping_text(attributes, "pattern_type") in pattern_types:
        return True
    if _mapping_text(attributes, "pattern_type") in routine_types:
        return True
    if _mapping_text(attributes, "routine_type") in routine_types:
        return True
    if _mapping_text(attributes, "routine_type") in pattern_types:
        return True
    if _mapping_text(attributes, "environment_id") in environment_ids:
        return True
    return False


def _load_projection_filtered_objects(
    object_store: object | None,
    *,
    predicate,
) -> tuple[LongTermMemoryObjectV1, ...]:
    if object_store is None:
        return ()
    loader = getattr(object_store, "load_objects_by_projection_filter", None)
    if not callable(loader):
        return ()
    return tuple(loader(predicate=predicate))


def _load_event_objects(
    object_store: object | None,
    *,
    event_ids: tuple[str, ...],
) -> tuple[LongTermMemoryObjectV1, ...]:
    if object_store is None or not event_ids:
        return ()
    loader = getattr(object_store, "load_objects_by_event_ids", None)
    if not callable(loader):
        return ()
    return tuple(loader(event_ids))


def _add_selected_objects(
    selected_by_id: dict[str, LongTermMemoryObjectV1],
    items: Iterable[LongTermMemoryObjectV1],
) -> None:
    for item in items:
        if not isinstance(item, LongTermMemoryObjectV1):
            continue
        memory_id = _normalize_text(item.memory_id)
        if memory_id and memory_id not in selected_by_id:
            selected_by_id[memory_id] = item


def select_reflection_neighborhood_objects(
    object_store: object | None,
    *,
    seed_objects: Iterable[LongTermMemoryObjectV1] = (),
) -> tuple[LongTermMemoryObjectV1, ...]:
    """Return reflection inputs scoped to the touched memory neighborhood.

    Active persistence and backfill already know the fresh/touched object set
    they are extending. For those hot paths, reflection should stay bounded to
    the exact touched objects plus their nearby person/event/environment
    neighbors instead of widening back out to a generic compile-source union.
    """

    seed_tuple = _normalize_seed_objects(seed_objects)
    if not seed_tuple:
        return select_reflection_compile_source_objects(object_store)

    selected_by_id: dict[str, LongTermMemoryObjectV1] = {}
    _add_selected_objects(selected_by_id, seed_tuple)

    event_ids = _seed_event_ids(seed_tuple)
    person_refs = frozenset(_seed_person_refs(seed_tuple))
    environment_ids = frozenset(_seed_attribute_values(seed_tuple, "environment_id"))
    include_environment_domain = (
        "smart_home_environment" in _seed_attribute_values(seed_tuple, "memory_domain")
        or any(
            summary_type.startswith("environment")
            for summary_type in _seed_attribute_values(seed_tuple, "summary_type")
        )
    )

    _add_selected_objects(
        selected_by_id,
        _load_event_objects(object_store, event_ids=event_ids),
    )
    _add_selected_objects(
        selected_by_id,
        _load_projection_filtered_objects(
            object_store,
            predicate=lambda projection: _projection_has_live_status(projection)
            and (
                bool(frozenset(_mapping_text_list(projection, "source_event_ids")).intersection(event_ids))
                or _projection_matches_person_neighborhood(
                    projection,
                    person_refs=person_refs,
                )
                or _projection_matches_environment_neighborhood(
                    projection,
                    environment_ids=environment_ids,
                    include_environment_domain=include_environment_domain,
                )
            ),
        ),
    )

    return tuple(selected_by_id.values())


def select_sensor_memory_neighborhood_objects(
    object_store: object | None,
    *,
    seed_objects: Iterable[LongTermMemoryObjectV1] = (),
) -> tuple[LongTermMemoryObjectV1, ...]:
    """Return sensor-memory inputs scoped to the touched sensor neighborhood.

    Seeded runtime paths must not widen to the broader generic sensor compile
    union when the touched neighborhood is small; that would reintroduce the
    broad current-state reasoning shape this selector exists to avoid.
    """

    seed_tuple = _normalize_seed_objects(seed_objects)
    if not seed_tuple:
        return select_sensor_memory_source_objects(object_store)

    selected_by_id: dict[str, LongTermMemoryObjectV1] = {}
    _add_selected_objects(selected_by_id, seed_tuple)

    event_ids = _seed_event_ids(seed_tuple)
    sensor_domains = frozenset(
        value
        for value in _seed_attribute_values(seed_tuple, "memory_domain")
        if value in _SENSOR_NEIGHBORHOOD_DOMAINS
    )
    summary_types = frozenset(_seed_attribute_values(seed_tuple, "summary_type"))
    pattern_types = frozenset(_seed_attribute_values(seed_tuple, "pattern_type"))
    routine_types = frozenset(_seed_attribute_values(seed_tuple, "routine_type"))
    environment_ids = frozenset(_seed_attribute_values(seed_tuple, "environment_id"))

    _add_selected_objects(
        selected_by_id,
        _load_event_objects(object_store, event_ids=event_ids),
    )
    _add_selected_objects(
        selected_by_id,
        _load_projection_filtered_objects(
            object_store,
            predicate=lambda projection: _projection_has_live_status(projection)
            and (
                bool(frozenset(_mapping_text_list(projection, "source_event_ids")).intersection(event_ids))
                or _projection_matches_sensor_neighborhood(
                    projection,
                    sensor_domains=sensor_domains,
                    summary_types=summary_types,
                    pattern_types=pattern_types,
                    routine_types=routine_types,
                    environment_ids=environment_ids,
                )
            ),
        ),
    )

    return tuple(selected_by_id.values())


_DISCOVERY_BASICS_SELECTIONS = (
    _FastTopicSelection(
        query_text="subject_ref user:main preference_type name predicate prefers_name preferred_name",
        limit=_DISCOVERY_OBJECT_QUERY_LIMIT,
    ),
)

_DISCOVERY_COMPANION_STYLE_SELECTIONS = (
    _FastTopicSelection(
        query_text=(
            "subject_ref user:main preference_type initiative verbosity humor "
            "style_dimension initiative verbosity humor "
            "feedback_target humor "
            "predicate user_prefers_answer_style user_prefers_small_follow_up_when_helpful "
            "address_preference address_style answer_style communication_style tone"
        ),
        limit=_DISCOVERY_OBJECT_QUERY_LIMIT,
    ),
)

_PROACTIVE_PLANNER_SELECTIONS = (
    _FastTopicSelection(
        query_text="event plan appointment reminder calendar visit meeting",
        limit=_PROACTIVE_OBJECT_QUERY_LIMIT,
    ),
    _FastTopicSelection(
        query_text="thread_summary summary summary_type thread support_count ongoing",
        limit=_PROACTIVE_OBJECT_QUERY_LIMIT,
    ),
    _FastTopicSelection(
        query_text="sensor_routine routine interaction camera print check_in",
        limit=_PROACTIVE_OBJECT_QUERY_LIMIT,
    ),
)

_REFLECTION_SUMMARY_SELECTIONS = (
    _FastTopicSelection(
        query_text=(
            "summary summary_type thread environment_reflection smart_home_environment "
            "person_name display_anchor"
        ),
        limit=_REFLECTION_SUMMARY_QUERY_LIMIT,
    ),
)

_REFLECTION_SOURCE_SELECTIONS = (
    _FastTopicSelection(
        query_text=(
            "person_ref person_name relation relationship family spouse partner wife husband "
            "daughter son mother father brother sister friend caregiver "
            "confirmed_by_user support_count active"
        ),
        limit=_REFLECTION_SOURCE_QUERY_LIMIT,
    ),
    _FastTopicSelection(
        query_text=(
            "event appointment reminder calendar visit meeting trip follow_up "
            "thread_summary summary_type thread memory_domain thread person_ref person_name support_count"
        ),
        limit=_REFLECTION_SOURCE_QUERY_LIMIT,
    ),
    _FastTopicSelection(
        query_text=(
            "memory_domain smart_home_environment summary_type environment_day_profile "
            "environment_deviation environment_deviation_event environment_quality_state "
            "environment_change_point environment_reflection environment_node "
            "pattern_type environment_baseline environment_regime environment_id"
        ),
        limit=_REFLECTION_SOURCE_QUERY_LIMIT,
    ),
)

_SENSOR_MEMORY_SOURCE_SELECTIONS = (
    _FastTopicSelection(
        query_text=(
            "pattern_type presence interaction daypart event_names "
            "memory_domain smart_home_environment"
        ),
        limit=_SENSOR_MEMORY_QUERY_LIMIT,
    ),
    _FastTopicSelection(
        query_text=(
            "memory_domain sensor_routine routine_type presence interaction "
            "summary_type sensor_deviation "
            "memory_domain respeaker_audio_preference memory_domain respeaker_audio_routine"
        ),
        limit=_SENSOR_MEMORY_QUERY_LIMIT,
    ),
    _FastTopicSelection(
        query_text=(
            "memory_domain smart_home_environment summary_type environment_day_profile "
            "environment_deviation environment_deviation_event environment_quality_state "
            "environment_change_point environment_node "
            "pattern_type environment_baseline environment_regime environment_id"
        ),
        limit=_SENSOR_MEMORY_QUERY_LIMIT,
    ),
)

_RESTART_RECALL_SOURCE_SELECTIONS = (
    _FastTopicSelection(
        query_text="confirmed_by_user confirmed active fact preference relationship support_count",
        limit=_RESTART_RECALL_QUERY_LIMIT,
    ),
    _FastTopicSelection(
        query_text=(
            "location place room drawer shelf stored kept found where "
            "wo ort ortsangabe stand steht liegt aufbewahrt object item possession"
        ),
        limit=_RESTART_RECALL_QUERY_LIMIT,
    ),
    _FastTopicSelection(
        query_text="event appointment reminder calendar visit meeting active support_count confirmed_by_user",
        limit=_RESTART_RECALL_QUERY_LIMIT,
    ),
    _FastTopicSelection(
        query_text="summary_type thread memory_domain thread thread_summary active support_count confirmed_by_user",
        limit=_RESTART_RECALL_QUERY_LIMIT,
    ),
    _FastTopicSelection(
        query_text=(
            "memory_domain sensor_routine routine_type interaction routine_type presence "
            "summary_type sensor_deviation active support_count confirmed_by_user"
        ),
        limit=_RESTART_RECALL_QUERY_LIMIT,
    ),
)


def select_discovery_basics_objects(
    object_store: object | None,
    *,
    timeout_s: float | None = None,
) -> tuple[LongTermMemoryObjectV1, ...]:
    """Return the bounded object slice that can satisfy the discovery basics topic."""

    return _select_fast_topic_union(
        object_store,
        selections=_DISCOVERY_BASICS_SELECTIONS,
        timeout_s=_DISCOVERY_OBJECT_QUERY_TIMEOUT_S if timeout_s is None else timeout_s,
    )


def select_discovery_companion_style_objects(
    object_store: object | None,
    *,
    timeout_s: float | None = None,
) -> tuple[LongTermMemoryObjectV1, ...]:
    """Return the bounded object slice that can satisfy companion-style discovery."""

    return _select_fast_topic_union(
        object_store,
        selections=_DISCOVERY_COMPANION_STYLE_SELECTIONS,
        timeout_s=_DISCOVERY_OBJECT_QUERY_TIMEOUT_S if timeout_s is None else timeout_s,
    )


def select_proactive_planner_objects(object_store: object | None) -> tuple[LongTermMemoryObjectV1, ...]:
    """Return the bounded object slice consumed by proactive planning."""

    return _select_fast_topic_union(object_store, selections=_PROACTIVE_PLANNER_SELECTIONS)


def select_reflection_summary_objects(object_store: object | None) -> tuple[LongTermMemoryObjectV1, ...]:
    """Return the visible summary slice for display-reserve reflection cards."""

    return _select_fast_topic_union(object_store, selections=_REFLECTION_SUMMARY_SELECTIONS)


def select_reflection_compile_source_objects(object_store: object | None) -> tuple[LongTermMemoryObjectV1, ...]:
    """Return the bounded reflection source slice for live runtime reflection runs."""

    return _select_fast_topic_union(object_store, selections=_REFLECTION_SOURCE_SELECTIONS)


def select_sensor_memory_source_objects(object_store: object | None) -> tuple[LongTermMemoryObjectV1, ...]:
    """Return the bounded source slice for live sensor-memory compilation."""

    return _select_fast_topic_union(object_store, selections=_SENSOR_MEMORY_SOURCE_SELECTIONS)


def select_restart_recall_source_objects(object_store: object | None) -> tuple[LongTermMemoryObjectV1, ...]:
    """Return the bounded stable-memory slice for restart-recall packet refresh."""

    return _select_fast_topic_union(object_store, selections=_RESTART_RECALL_SOURCE_SELECTIONS)


__all__ = [
    "select_discovery_basics_objects",
    "select_discovery_companion_style_objects",
    "select_reflection_neighborhood_objects",
    "select_proactive_planner_objects",
    "select_reflection_compile_source_objects",
    "select_reflection_summary_objects",
    "select_restart_recall_source_objects",
    "select_sensor_memory_neighborhood_objects",
    "select_sensor_memory_source_objects",
]
