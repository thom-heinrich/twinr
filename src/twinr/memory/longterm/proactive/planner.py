"""Rank proactive memory candidates from stored objects and live facts.

This module turns active long-term memory objects plus optional live sensor
facts into a bounded ``LongTermProactivePlanV1`` for the runtime. Import
``LongTermProactivePlanner`` from ``twinr.memory.longterm.proactive`` or
``twinr.memory.longterm``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING
import logging
from math import isfinite
from zoneinfo import ZoneInfo

from twinr.memory.longterm.core.ontology import is_thread_summary
from twinr.memory.longterm.core.models import (
    LongTermMemoryObjectV1,
    LongTermProactiveCandidateV1,
    LongTermProactivePlanV1,
)
from twinr.text_utils import collapse_whitespace, slugify_identifier

if TYPE_CHECKING:
    from twinr.proactive.runtime.ambiguous_room_guard import AmbiguousRoomGuardSnapshot

_DEFAULT_TIMEZONE_NAME = "Europe/Berlin"
_FALLBACK_TIMEZONE_NAMES = (_DEFAULT_TIMEZONE_NAME, "UTC")
_DEFAULT_MAX_CANDIDATES = 4
_CONCERNING_BODY_POSES = frozenset({"floor", "slumped"})
_TRUE_VALUES = frozenset({"1", "true", "yes", "y", "on"})
_FALSE_VALUES = frozenset({"", "0", "false", "no", "n", "off", "none", "null"})

LOGGER = logging.getLogger(__name__)


def _normalize_text(value: object | None) -> str:
    # AUDIT-FIX(#7): Normalize nullable and non-string text before using it in candidate IDs or user-facing text.
    if value is None:
        return ""
    try:
        text = str(value)
    except Exception:
        return ""
    return collapse_whitespace(text).strip()


def _slugify(value: object | None, *, fallback: str) -> str:
    # AUDIT-FIX(#7): Ensure slug generation is stable even when upstream identifiers are blank or malformed.
    normalized = _normalize_text(value) or fallback
    return slugify_identifier(normalized, fallback=fallback)


def _attrs(item: LongTermMemoryObjectV1) -> dict[str, object]:
    # AUDIT-FIX(#4): Treat persisted attributes as untrusted so one malformed memory object cannot crash planning.
    raw_attributes = getattr(item, "attributes", None)
    if not isinstance(raw_attributes, Mapping):
        return {}
    try:
        return dict(raw_attributes)
    except Exception:
        return {}


def _daypart_for_datetime(value: datetime) -> str:
    hour = value.hour
    if 5 <= hour < 11:
        return "morning"
    if 11 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"


def _weekday_class_for_datetime(value: datetime) -> str:
    return "weekend" if value.weekday() >= 5 else "weekday"


def _weekday_class_rank(value: str) -> int:
    return 2 if value in {"weekday", "weekend"} else 1


def _root_value(live_facts: Mapping[str, object] | None, key: str) -> object | None:
    # AUDIT-FIX(#5): Centralize root-level live-fact access so non-mapping inputs degrade safely instead of raising.
    if not isinstance(live_facts, Mapping):
        return None
    return live_facts.get(key)


def _section_value(live_facts: Mapping[str, object] | None, section: str, key: str) -> object | None:
    if not isinstance(live_facts, Mapping):
        return None
    payload = live_facts.get(section)
    if not isinstance(payload, Mapping):
        return None
    return payload.get(key)


def _bool_from_value(value: object | None) -> bool:
    # AUDIT-FIX(#2): Parse boolean-like payloads strictly; bool("false") must not become True.
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, float):
        return isfinite(value) and value != 0.0
    normalized = _normalize_text(value).lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return False


def _live_bool(live_facts: Mapping[str, object] | None, section: str, key: str) -> bool:
    return _bool_from_value(_section_value(live_facts, section, key))


def _root_bool(live_facts: Mapping[str, object] | None, key: str) -> bool:
    return _bool_from_value(_root_value(live_facts, key))


def _coerce_int(
    value: object,
    *,
    default: int,
    minimum: int | None = None,
) -> int:
    # AUDIT-FIX(#4,#6): Coerce numeric config/state defensively so malformed persisted values do not crash the planner.
    numeric = default
    if isinstance(value, bool):
        numeric = int(value)
    elif isinstance(value, int):
        numeric = value
    elif isinstance(value, float):
        if isfinite(value):
            numeric = int(value)
    else:
        normalized = _normalize_text(value)
        if normalized:
            try:
                numeric = int(normalized)
            except ValueError:
                try:
                    fallback_numeric = float(normalized)
                except ValueError:
                    numeric = default
                else:
                    if isfinite(fallback_numeric):
                        numeric = int(fallback_numeric)
    if minimum is not None and numeric < minimum:
        return minimum
    return numeric


def _coerce_confidence(
    value: object,
    *,
    default: float = 0.0,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    # AUDIT-FIX(#4): Clamp confidence values to a finite float range so corrupt state cannot break ranking or validation.
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    if not isfinite(numeric):
        numeric = default
    return min(maximum, max(minimum, numeric))


def _resolve_timezone(timezone_name: object | None) -> ZoneInfo:
    # AUDIT-FIX(#3): Fall back to safe built-in zones when configuration is missing or invalid.
    requested = _normalize_text(timezone_name)
    for candidate in (requested, *_FALLBACK_TIMEZONE_NAMES):
        if not candidate:
            continue
        try:
            return ZoneInfo(candidate)
        except Exception:
            LOGGER.warning("Failed to resolve proactive planner timezone %r; trying fallback.", candidate, exc_info=True)
            continue
    return ZoneInfo("UTC")


def _coerce_reference_datetime(value: datetime | None, *, timezone_name: object | None) -> datetime:
    # AUDIT-FIX(#3): Normalize all reference datetimes into the configured local timezone before deriving day/date buckets.
    tz = _resolve_timezone(timezone_name)
    if value is None:
        return datetime.now(tz)
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=tz)
    try:
        return value.astimezone(tz)
    except Exception:
        return value.replace(tzinfo=tz)


def _parse_memory_date(value: object | None, *, timezone_name: object | None) -> date | None:
    """Return one calendar date from a persisted memory date-like field."""

    if isinstance(value, datetime):
        return _coerce_reference_datetime(value, timezone_name=timezone_name).date()
    if isinstance(value, date):
        return value
    normalized = _normalize_text(value)
    if not normalized:
        return None
    try:
        return date.fromisoformat(normalized[:10])
    except ValueError:
        return None


def _parse_memory_datetime(value: object | None, *, timezone_name: object | None) -> datetime | None:
    """Return one local reference datetime from a persisted memory date-time field."""

    if isinstance(value, datetime):
        return _coerce_reference_datetime(value, timezone_name=timezone_name)
    normalized = _normalize_text(value)
    if not normalized or len(normalized) <= 10:
        return None
    candidate = normalized.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    return _coerce_reference_datetime(parsed, timezone_name=timezone_name)


def _has_concerning_body_pose(live_facts: Mapping[str, object] | None) -> bool:
    # AUDIT-FIX(#1): Suppress routine proactive offers when the visible body pose suggests possible distress.
    body_pose = _normalize_text(_section_value(live_facts, "camera", "body_pose")).lower()
    return body_pose in _CONCERNING_BODY_POSES


def _recent_follow_up_due_date(
    *,
    valid_from: object | None,
    valid_to: object | None,
    reference: datetime,
    timezone_name: object | None,
) -> str | None:
    """Return one due date when an event or plan has just passed recently enough."""

    recent_threshold = reference - timedelta(hours=2)
    today = reference.date()
    yesterday = today - timedelta(days=1)
    end_datetime = _parse_memory_datetime(valid_to, timezone_name=timezone_name)
    if end_datetime is not None and end_datetime <= recent_threshold:
        return end_datetime.date().isoformat()
    start_datetime = _parse_memory_datetime(valid_from, timezone_name=timezone_name)
    if start_datetime is not None and start_datetime <= recent_threshold:
        return start_datetime.date().isoformat()

    for candidate_date in (
        _parse_memory_date(valid_to, timezone_name=timezone_name),
        _parse_memory_date(valid_from, timezone_name=timezone_name),
    ):
        if candidate_date == yesterday:
            return candidate_date.isoformat()
    return None


def _ambiguous_room_guard(live_facts: Mapping[str, object] | None) -> AmbiguousRoomGuardSnapshot | None:
    """Return the current room-ambiguity guard when live facts are available."""

    if not isinstance(live_facts, Mapping):
        return None
    from twinr.proactive.runtime.ambiguous_room_guard import (
        AmbiguousRoomGuardSnapshot,
        derive_ambiguous_room_guard,
    )

    return AmbiguousRoomGuardSnapshot.from_fact_map(live_facts.get("ambiguous_room_guard")) or derive_ambiguous_room_guard(
        observed_at=_section_value(live_facts, "sensor", "observed_at"),
        live_facts=live_facts,
    )


@dataclass(frozen=True, slots=True)
class LongTermProactivePlanner:
    """Rank bounded proactive candidates for the long-term memory runtime.

    The planner combines durable same-day events, next-day reminders, thread
    summaries, and sensor-routine memories into a deduplicated candidate list.
    Live sensor facts are optional, but routine offers only surface when the
    current context confirms that the suggestion fits.

    Attributes:
        timezone_name: IANA timezone used to derive today, tomorrow, and
            current dayparts.
        max_candidates: Maximum number of ranked candidates returned per plan.
    """

    timezone_name: str = _DEFAULT_TIMEZONE_NAME
    max_candidates: int = _DEFAULT_MAX_CANDIDATES

    def plan(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
        now: datetime | None = None,
        live_facts: Mapping[str, object] | None = None,
    ) -> LongTermProactivePlanV1:
        """Build a bounded proactive plan for the current runtime context.

        Args:
            objects: Canonical or canonicalizable long-term memory objects to
                inspect.
            now: Reference time for day-sensitive planning. Defaults to the
                current time in ``timezone_name``.
            live_facts: Optional live sensor snapshot used for routine
                check-ins, camera offers, and print offers.

        Returns:
            A ``LongTermProactivePlanV1`` ordered by descending confidence and
            candidate stability. Malformed objects are skipped instead of
            aborting the full plan.
        """

        # AUDIT-FIX(#3): Derive all calendar buckets from a timezone-normalized reference instant.
        reference = _coerce_reference_datetime(now, timezone_name=self.timezone_name)
        today = reference.date().isoformat()
        tomorrow = (reference.date() + timedelta(days=1)).isoformat()
        candidates: list[LongTermProactiveCandidateV1] = []
        canonicalized_items: list[LongTermMemoryObjectV1] = []
        # AUDIT-FIX(#4): Accept empty or missing object collections without relying on fragile truthiness.
        input_objects = () if objects is None else tuple(objects)
        for raw_item in input_objects:
            # AUDIT-FIX(#4): Isolate canonicalization failures to the bad object instead of dropping the entire proactive plan.
            canonicalize = getattr(raw_item, "canonicalized", None)
            try:
                item = canonicalize() if callable(canonicalize) else raw_item
            except Exception:
                LOGGER.warning("Skipping malformed long-term proactive object during canonicalization.", exc_info=True)
                continue
            canonicalized_items.append(item)
        canonical_objects = tuple(canonicalized_items)

        for item in canonical_objects:
            # AUDIT-FIX(#4): Consume persisted memory fields defensively so a single malformed record cannot abort planning.
            try:
                memory_id = _normalize_text(getattr(item, "memory_id", None))
                if not memory_id:
                    continue
                status = _normalize_text(getattr(item, "status", None)).lower()
                if status != "active":
                    continue
                kind = _normalize_text(getattr(item, "kind", None)).lower()
                summary = _normalize_text(getattr(item, "summary", None))
                item_summary = summary or "an important item"
                valid_from = _normalize_text(getattr(item, "valid_from", None))
                confidence = _coerce_confidence(getattr(item, "confidence", None), default=0.5)
                sensitivity = item.sensitivity
                attributes = _attrs(item)
            except Exception:
                LOGGER.warning("Skipping malformed long-term proactive object during planning.", exc_info=True)
                continue

            if kind in {"event", "plan"} and valid_from == today:
                candidates.append(
                    LongTermProactiveCandidateV1(
                        candidate_id=f"candidate:{_slugify(memory_id, fallback='today')}:today",
                        kind="same_day_reminder",
                        # AUDIT-FIX(#7): Use normalized summaries so spoken/rendered text never contains raw None or broken whitespace.
                        summary=f"Gently remind the user about: {item_summary}",
                        rationale="A same-day event or plan is active in long-term memory.",
                        due_date=today,
                        confidence=_coerce_confidence(confidence, default=0.5, maximum=0.99),
                        source_memory_ids=(memory_id,),
                        sensitivity=sensitivity,
                    )
                )
            elif kind in {"event", "plan"} and valid_from == tomorrow:
                candidates.append(
                    LongTermProactiveCandidateV1(
                        candidate_id=f"candidate:{_slugify(memory_id, fallback='tomorrow')}:tomorrow",
                        kind="next_day_reminder",
                        summary=f"Consider reminding the user tomorrow about: {item_summary}",
                        rationale="A next-day event or plan is active in long-term memory.",
                        due_date=tomorrow,
                        confidence=_coerce_confidence(confidence - 0.05, default=0.55, minimum=0.55, maximum=0.95),
                        source_memory_ids=(memory_id,),
                        sensitivity=sensitivity,
                    )
                )
            elif kind in {"event", "plan"}:
                recent_follow_up_due_date = _recent_follow_up_due_date(
                    valid_from=getattr(item, "valid_from", None),
                    valid_to=getattr(item, "valid_to", None),
                    reference=reference,
                    timezone_name=self.timezone_name,
                )
                if recent_follow_up_due_date is not None:
                    candidates.append(
                        LongTermProactiveCandidateV1(
                            candidate_id=f"candidate:{_slugify(memory_id, fallback='recent')}:recent_follow_up",
                            kind="gentle_follow_up",
                            summary=f"If relevant, gently follow up on: {item_summary}",
                            rationale=(
                                "A recent event or plan appears to have just happened, so a calm continuity "
                                "check-in may help the user add the outcome or next step."
                            ),
                            due_date=recent_follow_up_due_date,
                            confidence=_coerce_confidence(
                                confidence + 0.04,
                                default=0.62,
                                minimum=0.62,
                                maximum=0.9,
                            ),
                            source_memory_ids=(memory_id,),
                            sensitivity=sensitivity,
                        )
                    )
            elif is_thread_summary(kind, attributes):
                # AUDIT-FIX(#4): Coerce support_count defensively because persisted summaries may store it as text or junk.
                support_count = _coerce_int(attributes.get("support_count", 1), default=1, minimum=1)
                if support_count >= 2:
                    candidates.append(
                        LongTermProactiveCandidateV1(
                            candidate_id=f"candidate:{_slugify(memory_id, fallback='thread')}:followup",
                            kind="gentle_follow_up",
                            summary=f"If relevant, gently follow up on: {item_summary}",
                            rationale="Multiple long-term signals point to an ongoing life thread worth soft continuity.",
                            due_date=valid_from or today,
                            confidence=_coerce_confidence(0.45 + 0.08 * support_count, default=0.45, maximum=0.9),
                            source_memory_ids=(memory_id,),
                            sensitivity=sensitivity,
                        )
                    )

        candidates.extend(
            self._plan_sensor_memory_candidates(
                objects=canonical_objects,
                reference=reference,
                live_facts=live_facts,
            )
        )

        unique: dict[str, LongTermProactiveCandidateV1] = {}
        for candidate in candidates:
            existing = unique.get(candidate.candidate_id)
            if existing is None or candidate.confidence >= existing.confidence:
                unique[candidate.candidate_id] = candidate
        ranked = sorted(
            unique.values(),
            key=lambda item: (-item.confidence, item.kind, item.candidate_id),
        )
        limit = _coerce_int(self.max_candidates, default=_DEFAULT_MAX_CANDIDATES, minimum=0)
        # AUDIT-FIX(#6): Guard against negative or non-numeric configuration values before slicing.
        return LongTermProactivePlanV1(candidates=tuple(ranked[:limit]))

    def _plan_sensor_memory_candidates(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
        reference: datetime,
        live_facts: Mapping[str, object] | None,
    ) -> tuple[LongTermProactiveCandidateV1, ...]:
        # AUDIT-FIX(#5): Only process sensor inputs when the live-facts payload has the expected mapping shape.
        if not isinstance(live_facts, Mapping):
            return ()
        # AUDIT-FIX(#3): Normalize internal calls too, so daypart/date logic stays timezone-stable even in tests or direct invocation.
        reference = _coerce_reference_datetime(reference, timezone_name=self.timezone_name)
        # AUDIT-FIX(#1): Distress-like body poses must suppress routine proactive offers in this planner.
        if _has_concerning_body_pose(live_facts):
            return ()
        room_guard = _ambiguous_room_guard(live_facts)
        if room_guard is not None and room_guard.guard_active:
            return ()
        candidates: list[LongTermProactiveCandidateV1] = []
        current_daypart = _daypart_for_datetime(reference)
        current_weekday_class = _weekday_class_for_datetime(reference)
        today = reference.date().isoformat()

        if candidate := self._build_routine_check_in_candidate(
            objects=objects,
            current_daypart=current_daypart,
            current_weekday_class=current_weekday_class,
            today=today,
            live_facts=live_facts,
        ):
            candidates.append(candidate)
        if candidate := self._build_routine_camera_offer_candidate(
            objects=objects,
            current_daypart=current_daypart,
            current_weekday_class=current_weekday_class,
            today=today,
            live_facts=live_facts,
        ):
            candidates.append(candidate)
        if candidate := self._build_routine_print_offer_candidate(
            objects=objects,
            current_daypart=current_daypart,
            current_weekday_class=current_weekday_class,
            today=today,
            live_facts=live_facts,
        ):
            candidates.append(candidate)
        return tuple(candidates)

    def _build_routine_check_in_candidate(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
        current_daypart: str,
        current_weekday_class: str,
        today: str,
        live_facts: Mapping[str, object],
    ) -> LongTermProactiveCandidateV1 | None:
        person_visible = _live_bool(live_facts, "camera", "person_visible")
        quiet = _live_bool(live_facts, "vad", "quiet")
        speech_detected = _live_bool(live_facts, "vad", "speech_detected")
        # AUDIT-FIX(#1): Keep a second safety gate here so direct unit calls cannot bypass the distress-pose suppression.
        if _has_concerning_body_pose(live_facts):
            return None
        if not person_visible or speech_detected or not quiet:
            return None
        source = self._best_sensor_object(
            objects=objects,
            predicate=lambda item, attrs: (
                _normalize_text(getattr(item, "kind", None)).lower() == "summary"
                and _normalize_text(getattr(item, "status", None)).lower() in {"active", "candidate", "uncertain"}
                and attrs.get("memory_domain") == "sensor_routine"
                and attrs.get("summary_type") == "sensor_deviation"
                and attrs.get("deviation_type") == "missing_presence"
                and attrs.get("daypart") == current_daypart
                and attrs.get("date") == today
                and attrs.get("weekday_class") in {current_weekday_class, "all_days"}
            ),
        )
        if source is None:
            return None
        # AUDIT-FIX(#4): Validate source object fields before building a candidate from matched sensor memory.
        source_memory_id = _normalize_text(getattr(source, "memory_id", None))
        if not source_memory_id:
            return None
        try:
            source_sensitivity = source.sensitivity
        except Exception:
            return None
        weekday_class = _normalize_text(_attrs(source).get("weekday_class")) or current_weekday_class
        confidence = _coerce_confidence(
            _coerce_confidence(getattr(source, "confidence", None), default=0.5) + 0.08 + (0.03 if quiet else 0.0),
            default=0.76,
            minimum=0.76,
            maximum=0.92,
        )
        return LongTermProactiveCandidateV1(
            candidate_id=f"candidate:sensor_check_in:{today}:{current_daypart}",
            kind="routine_check_in",
            summary="If it feels right, gently check in and ask whether everything is okay.",
            rationale=(
                f"Presence has been lower than usual in the {current_daypart} for "
                f"{weekday_class.replace('_', ' ')}s, and the user is visible and quiet now."
            ),
            due_date=today,
            confidence=confidence,
            source_memory_ids=(source_memory_id,),
            sensitivity=source_sensitivity,
        )

    def _build_routine_camera_offer_candidate(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
        current_daypart: str,
        current_weekday_class: str,
        today: str,
        live_facts: Mapping[str, object],
    ) -> LongTermProactiveCandidateV1 | None:
        person_visible = _live_bool(live_facts, "camera", "person_visible")
        hand_near = _live_bool(live_facts, "camera", "hand_or_object_near_camera")
        looking = _live_bool(live_facts, "camera", "looking_toward_device")
        speech_detected = _live_bool(live_facts, "vad", "speech_detected")
        # AUDIT-FIX(#1): Suppress routine offers when pose signals potential distress.
        if _has_concerning_body_pose(live_facts):
            return None
        if not person_visible or speech_detected or not (hand_near or looking):
            return None
        source = self._best_sensor_object(
            objects=objects,
            predicate=lambda item, attrs: (
                _normalize_text(getattr(item, "kind", None)).lower() == "pattern"
                and _normalize_text(getattr(item, "status", None)).lower() == "active"
                and attrs.get("memory_domain") == "sensor_routine"
                and attrs.get("routine_type") == "interaction"
                and attrs.get("interaction_type") in {"camera_showing", "camera_use"}
                and attrs.get("daypart") == current_daypart
                and attrs.get("weekday_class") in {current_weekday_class, "all_days"}
            ),
        )
        if source is None:
            return None
        # AUDIT-FIX(#4): Validate source object fields before building a candidate from matched sensor memory.
        source_memory_id = _normalize_text(getattr(source, "memory_id", None))
        if not source_memory_id:
            return None
        try:
            source_sensitivity = source.sensitivity
        except Exception:
            return None
        interaction_type = _normalize_text(_attrs(source).get("interaction_type")) or "camera_showing"
        confidence = _coerce_confidence(
            _coerce_confidence(getattr(source, "confidence", None), default=0.5) + (0.08 if hand_near else 0.04),
            default=0.78,
            minimum=0.78,
            maximum=0.91,
        )
        return LongTermProactiveCandidateV1(
            candidate_id=f"candidate:sensor_camera_offer:{today}:{current_daypart}",
            kind="routine_camera_offer",
            summary="If it fits, offer to look at something in the camera.",
            rationale=(
                f"{interaction_type.replace('_', ' ')} is typical in the {current_daypart}, "
                "and the current live camera view suggests the user may want to show something."
            ),
            due_date=today,
            confidence=confidence,
            source_memory_ids=(source_memory_id,),
            sensitivity=source_sensitivity,
        )

    def _build_routine_print_offer_candidate(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
        current_daypart: str,
        current_weekday_class: str,
        today: str,
        live_facts: Mapping[str, object],
    ) -> LongTermProactiveCandidateV1 | None:
        person_visible = _live_bool(live_facts, "camera", "person_visible")
        looking = _live_bool(live_facts, "camera", "looking_toward_device")
        quiet = _live_bool(live_facts, "vad", "quiet")
        speech_detected = _live_bool(live_facts, "vad", "speech_detected")
        # AUDIT-FIX(#2): Parse root-level booleans strictly so strings like "false" do not flip print logic.
        last_response_available = _root_bool(live_facts, "last_response_available")
        recent_print_completed = _root_bool(live_facts, "recent_print_completed")
        # AUDIT-FIX(#1): Never propose printing during active speech or when body pose looks concerning.
        if _has_concerning_body_pose(live_facts):
            return None
        if not person_visible or speech_detected or not last_response_available or recent_print_completed:
            return None
        if not (looking or quiet):
            return None
        source = self._best_sensor_object(
            objects=objects,
            predicate=lambda item, attrs: (
                _normalize_text(getattr(item, "kind", None)).lower() == "pattern"
                and _normalize_text(getattr(item, "status", None)).lower() == "active"
                and attrs.get("memory_domain") == "sensor_routine"
                and attrs.get("routine_type") == "interaction"
                and attrs.get("interaction_type") == "print"
                and attrs.get("daypart") == current_daypart
                and attrs.get("weekday_class") in {current_weekday_class, "all_days"}
            ),
        )
        if source is None:
            return None
        # AUDIT-FIX(#4): Validate source object fields before building a candidate from matched sensor memory.
        source_memory_id = _normalize_text(getattr(source, "memory_id", None))
        if not source_memory_id:
            return None
        try:
            source_sensitivity = source.sensitivity
        except Exception:
            return None
        confidence = _coerce_confidence(
            _coerce_confidence(getattr(source, "confidence", None), default=0.5) + 0.04 + (0.03 if looking else 0.0),
            default=0.76,
            minimum=0.76,
            maximum=0.89,
        )
        return LongTermProactiveCandidateV1(
            candidate_id=f"candidate:sensor_print_offer:{today}:{current_daypart}",
            kind="routine_print_offer",
            summary="If it helps, offer to print the latest answer.",
            rationale=(
                f"Printing is typical in the {current_daypart}, and there is a recent answer available while the user is still present."
            ),
            due_date=today,
            confidence=confidence,
            source_memory_ids=(source_memory_id,),
            sensitivity=source_sensitivity,
        )

    def _best_sensor_object(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
        predicate: Callable[[LongTermMemoryObjectV1, Mapping[str, object]], bool],
    ) -> LongTermMemoryObjectV1 | None:
        # AUDIT-FIX(#8): Type the predicate to make misuse visible in static analysis and review.
        matches: list[tuple[float, int, LongTermMemoryObjectV1]] = []
        for item in objects:
            attrs = _attrs(item)
            # AUDIT-FIX(#4): Ignore malformed records that make predicate evaluation fail instead of crashing the full planner.
            try:
                if not predicate(item, attrs):
                    continue
            except Exception:
                LOGGER.warning("Skipping proactive planner candidate because its predicate raised.", exc_info=True)
                continue
            matches.append(
                (
                    _coerce_confidence(getattr(item, "confidence", None), default=0.0),
                    _weekday_class_rank(_normalize_text(attrs.get("weekday_class")).lower()),
                    item,
                )
            )
        if not matches:
            return None
        matches.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return matches[0][2]


__all__ = ["LongTermProactivePlanner"]
