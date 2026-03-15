from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from twinr.memory.longterm.ontology import is_thread_summary
from twinr.memory.longterm.models import (
    LongTermMemoryObjectV1,
    LongTermProactiveCandidateV1,
    LongTermProactivePlanV1,
)
from twinr.text_utils import collapse_whitespace, slugify_identifier


def _normalize_text(value: str | None) -> str:
    return collapse_whitespace(value)


def _slugify(value: str, *, fallback: str) -> str:
    return slugify_identifier(value, fallback=fallback)


def _attrs(item: LongTermMemoryObjectV1) -> dict[str, object]:
    return dict(item.attributes or {})


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


def _section_value(live_facts: Mapping[str, object] | None, section: str, key: str) -> object | None:
    if not isinstance(live_facts, Mapping):
        return None
    payload = live_facts.get(section)
    if not isinstance(payload, Mapping):
        return None
    return payload.get(key)


def _live_bool(live_facts: Mapping[str, object] | None, section: str, key: str) -> bool:
    return bool(_section_value(live_facts, section, key))


@dataclass(frozen=True, slots=True)
class LongTermProactivePlanner:
    timezone_name: str = "Europe/Berlin"
    max_candidates: int = 4

    def plan(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
        now: datetime | None = None,
        live_facts: Mapping[str, object] | None = None,
    ) -> LongTermProactivePlanV1:
        reference = now or datetime.now(ZoneInfo(self.timezone_name))
        today = reference.date().isoformat()
        tomorrow = (reference.date() + timedelta(days=1)).isoformat()
        candidates: list[LongTermProactiveCandidateV1] = []
        canonical_objects = tuple(item.canonicalized() for item in objects)

        for item in canonical_objects:
            if item.status != "active":
                continue
            if item.kind in {"event", "plan"} and item.valid_from == today:
                candidates.append(
                    LongTermProactiveCandidateV1(
                        candidate_id=f"candidate:{_slugify(item.memory_id, fallback='today')}:today",
                        kind="same_day_reminder",
                        summary=f"Gently remind the user about: {item.summary}",
                        rationale="A same-day event or plan is active in long-term memory.",
                        due_date=today,
                        confidence=min(0.99, item.confidence),
                        source_memory_ids=(item.memory_id,),
                        sensitivity=item.sensitivity,
                    )
                )
            elif item.kind in {"event", "plan"} and item.valid_from == tomorrow:
                candidates.append(
                    LongTermProactiveCandidateV1(
                        candidate_id=f"candidate:{_slugify(item.memory_id, fallback='tomorrow')}:tomorrow",
                        kind="next_day_reminder",
                        summary=f"Consider reminding the user tomorrow about: {item.summary}",
                        rationale="A next-day event or plan is active in long-term memory.",
                        due_date=tomorrow,
                        confidence=max(0.55, min(0.95, item.confidence - 0.05)),
                        source_memory_ids=(item.memory_id,),
                        sensitivity=item.sensitivity,
                    )
                )
            elif is_thread_summary(item.kind, item.attributes):
                support_count = int((item.attributes or {}).get("support_count", 1))
                if support_count >= 2:
                    candidates.append(
                        LongTermProactiveCandidateV1(
                            candidate_id=f"candidate:{_slugify(item.memory_id, fallback='thread')}:followup",
                            kind="gentle_follow_up",
                            summary=f"If relevant, gently follow up on: {item.summary}",
                            rationale="Multiple long-term signals point to an ongoing life thread worth soft continuity.",
                            due_date=item.valid_from or today,
                            confidence=min(0.9, 0.45 + 0.08 * support_count),
                            source_memory_ids=(item.memory_id,),
                            sensitivity=item.sensitivity,
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
        return LongTermProactivePlanV1(candidates=tuple(ranked[: self.max_candidates]))

    def _plan_sensor_memory_candidates(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
        reference: datetime,
        live_facts: Mapping[str, object] | None,
    ) -> tuple[LongTermProactiveCandidateV1, ...]:
        if live_facts is None:
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
        body_pose = str(_section_value(live_facts, "camera", "body_pose") or "").strip().lower()
        if not person_visible or speech_detected or not quiet:
            return None
        if body_pose in {"floor", "slumped"}:
            return None
        source = self._best_sensor_object(
            objects=objects,
            predicate=lambda item, attrs: (
                item.kind == "summary"
                and item.status in {"active", "candidate", "uncertain"}
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
        weekday_class = str(_attrs(source).get("weekday_class", current_weekday_class)).strip() or current_weekday_class
        confidence = min(0.92, max(0.76, source.confidence + 0.08 + (0.03 if quiet else 0.0)))
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
            source_memory_ids=(source.memory_id,),
            sensitivity=source.sensitivity,
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
        if not person_visible or speech_detected or not (hand_near or looking):
            return None
        source = self._best_sensor_object(
            objects=objects,
            predicate=lambda item, attrs: (
                item.kind == "pattern"
                and item.status == "active"
                and attrs.get("memory_domain") == "sensor_routine"
                and attrs.get("routine_type") == "interaction"
                and attrs.get("interaction_type") in {"camera_showing", "camera_use"}
                and attrs.get("daypart") == current_daypart
                and attrs.get("weekday_class") in {current_weekday_class, "all_days"}
            ),
        )
        if source is None:
            return None
        interaction_type = str(_attrs(source).get("interaction_type", "camera_showing")).strip() or "camera_showing"
        confidence = min(0.91, max(0.78, source.confidence + (0.08 if hand_near else 0.04)))
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
            source_memory_ids=(source.memory_id,),
            sensitivity=source.sensitivity,
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
        last_response_available = bool(live_facts.get("last_response_available"))
        recent_print_completed = bool(live_facts.get("recent_print_completed"))
        if not person_visible or not last_response_available or recent_print_completed:
            return None
        if not (looking or quiet):
            return None
        source = self._best_sensor_object(
            objects=objects,
            predicate=lambda item, attrs: (
                item.kind == "pattern"
                and item.status == "active"
                and attrs.get("memory_domain") == "sensor_routine"
                and attrs.get("routine_type") == "interaction"
                and attrs.get("interaction_type") == "print"
                and attrs.get("daypart") == current_daypart
                and attrs.get("weekday_class") in {current_weekday_class, "all_days"}
            ),
        )
        if source is None:
            return None
        confidence = min(0.89, max(0.76, source.confidence + 0.04 + (0.03 if looking else 0.0)))
        return LongTermProactiveCandidateV1(
            candidate_id=f"candidate:sensor_print_offer:{today}:{current_daypart}",
            kind="routine_print_offer",
            summary="If it helps, offer to print the latest answer.",
            rationale=(
                f"Printing is typical in the {current_daypart}, and there is a recent answer available while the user is still present."
            ),
            due_date=today,
            confidence=confidence,
            source_memory_ids=(source.memory_id,),
            sensitivity=source.sensitivity,
        )

    def _best_sensor_object(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
        predicate,
    ) -> LongTermMemoryObjectV1 | None:
        matches = []
        for item in objects:
            attrs = _attrs(item)
            if not predicate(item, attrs):
                continue
            matches.append(
                (
                    item.confidence,
                    _weekday_class_rank(str(attrs.get("weekday_class", ""))),
                    item,
                )
            )
        if not matches:
            return None
        matches.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return matches[0][2]


__all__ = ["LongTermProactivePlanner"]
