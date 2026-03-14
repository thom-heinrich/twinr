from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

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


@dataclass(frozen=True, slots=True)
class LongTermProactivePlanner:
    timezone_name: str = "Europe/Berlin"
    max_candidates: int = 4

    def plan(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
        now: datetime | None = None,
    ) -> LongTermProactivePlanV1:
        reference = now or datetime.now(ZoneInfo(self.timezone_name))
        today = reference.date().isoformat()
        tomorrow = (reference.date() + timedelta(days=1)).isoformat()
        candidates: list[LongTermProactiveCandidateV1] = []

        for item in objects:
            if item.status != "active":
                continue
            if item.kind in {"medical_event", "event_fact"} and item.valid_from == today:
                candidates.append(
                    LongTermProactiveCandidateV1(
                        candidate_id=f"candidate:{_slugify(item.memory_id, fallback='today')}:today",
                        kind="same_day_reminder",
                        summary=f"Gently remind the user about: {item.summary}",
                        rationale="A same-day event is active in long-term memory.",
                        due_date=today,
                        confidence=min(0.99, item.confidence),
                        source_memory_ids=(item.memory_id,),
                        sensitivity=item.sensitivity,
                    )
                )
            elif item.kind in {"medical_event", "event_fact"} and item.valid_from == tomorrow:
                candidates.append(
                    LongTermProactiveCandidateV1(
                        candidate_id=f"candidate:{_slugify(item.memory_id, fallback='tomorrow')}:tomorrow",
                        kind="next_day_reminder",
                        summary=f"Consider reminding the user tomorrow about: {item.summary}",
                        rationale="A next-day event is active in long-term memory.",
                        due_date=tomorrow,
                        confidence=max(0.55, min(0.95, item.confidence - 0.05)),
                        source_memory_ids=(item.memory_id,),
                        sensitivity=item.sensitivity,
                    )
                )
            elif item.kind == "thread_summary":
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

        unique: dict[str, LongTermProactiveCandidateV1] = {}
        for candidate in candidates:
            unique[candidate.candidate_id] = candidate
        ranked = sorted(
            unique.values(),
            key=lambda item: (-item.confidence, item.kind, item.candidate_id),
        )
        return LongTermProactivePlanV1(candidates=tuple(ranked[: self.max_candidates]))


__all__ = ["LongTermProactivePlanner"]
