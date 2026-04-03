"""Extract bounded nightly insight lines from reflection and world refresh.

This helper keeps nightly insight shaping out of the main orchestrator loop.
It converts already-computed reflection, world-intelligence, and personality
results into short, operator-auditable lines that the nightly artifacts and
morning digest prompts can reuse directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from twinr.agent.personality.evolution import PersonalityEvolutionResult
from twinr.agent.personality.intelligence.models import (
    SituationalAwarenessThread,
    WorldIntelligenceRefreshResult,
)
from twinr.agent.personality.models import ContinuityThread
from twinr.memory.longterm.core.models import LongTermReflectionResultV1
from twinr.proactive.runtime.display_reserve_support import compact_text

_MAX_LINE = 220
_MAX_TIME = 64
_DEFAULT_INSIGHT_LIMIT = 5
_DEFAULT_SHIFT_LIMIT = 5


def _bounded_text(value: object | None, *, max_len: int) -> str:
    """Collapse one arbitrary value into bounded single-line text."""

    return compact_text(value, max_len=max_len) if value is not None else ""


def _bounded_lines(
    candidates: Iterable[str],
    *,
    limit: int,
) -> tuple[str, ...]:
    """Normalize, dedupe, and cap one ordered line sequence."""

    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        line = _bounded_text(candidate, max_len=_MAX_LINE)
        if not line:
            continue
        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(line)
        if len(normalized) >= max(1, int(limit)):
            break
    return tuple(normalized)


def _continuity_line(thread: ContinuityThread) -> str:
    """Render one continuity thread into a bounded calm line."""

    title = _bounded_text(getattr(thread, "title", None), max_len=120)
    summary = _bounded_text(getattr(thread, "summary", None), max_len=180)
    if title and summary and summary.casefold() != title.casefold():
        return _bounded_text(f"{title}: {summary}", max_len=_MAX_LINE)
    return title or summary


def _awareness_line(thread: SituationalAwarenessThread) -> str:
    """Render one awareness thread into a bounded calm line."""

    title = _bounded_text(getattr(thread, "title", None), max_len=120)
    summary = _bounded_text(getattr(thread, "summary", None), max_len=180)
    if title and summary and summary.casefold() != title.casefold():
        return _bounded_text(f"{title}: {summary}", max_len=_MAX_LINE)
    return title or summary


def _sort_by_salience_and_time(items: Sequence[object]) -> tuple[object, ...]:
    """Rank persistent items by salience first and recency second."""

    return tuple(
        sorted(
            items,
            key=lambda item: (
                float(getattr(item, "salience", 0.0) or 0.0),
                _bounded_text(getattr(item, "updated_at", None), max_len=_MAX_TIME),
            ),
            reverse=True,
        )
    )


def _reflection_lines(reflection_result: LongTermReflectionResultV1 | None) -> tuple[str, ...]:
    """Extract stable insight candidates from reflection output."""

    if reflection_result is None:
        return ()
    candidates: list[str] = []
    for item in tuple(getattr(reflection_result, "created_summaries", ())):
        summary = _bounded_text(getattr(item, "summary", None), max_len=180)
        if summary:
            candidates.append(summary)
    for item in tuple(getattr(reflection_result, "reflected_objects", ())):
        summary = _bounded_text(getattr(item, "summary", None), max_len=180)
        if summary:
            candidates.append(summary)
    return tuple(candidates)


def _personality_delta_lines(
    personality_results: Sequence[PersonalityEvolutionResult],
) -> tuple[str, ...]:
    """Extract accepted personality-delta summaries for operator review."""

    candidates: list[str] = []
    for result in personality_results:
        for delta in tuple(getattr(result, "accepted_deltas", ())):
            summary = _bounded_text(getattr(delta, "summary", None), max_len=180)
            if summary:
                candidates.append(summary)
    return tuple(candidates)


def _continuity_shift_lines(
    world_refresh_result: WorldIntelligenceRefreshResult | None,
) -> tuple[str, ...]:
    """Extract continuity-shift lines from refreshed world intelligence."""

    if world_refresh_result is None:
        return ()
    continuity_threads = _sort_by_salience_and_time(
        tuple(getattr(world_refresh_result, "continuity_threads", ()))
    )
    candidates = [_continuity_line(thread) for thread in continuity_threads]
    if any(candidates):
        return tuple(candidates)
    awareness_threads = _sort_by_salience_and_time(
        tuple(getattr(world_refresh_result, "awareness_threads", ()))
    )
    return tuple(_awareness_line(thread) for thread in awareness_threads)


@dataclass(frozen=True, slots=True)
class NightlyInsightBundle:
    """Carry bounded nightly insight lines between orchestration stages."""

    new_insights: tuple[str, ...] = ()
    continuity_shifts: tuple[str, ...] = ()


def build_nightly_insight_bundle(
    *,
    reflection_result: LongTermReflectionResultV1 | None,
    world_refresh_result: WorldIntelligenceRefreshResult | None,
    personality_results: Sequence[PersonalityEvolutionResult],
    max_new_insights: int = _DEFAULT_INSIGHT_LIMIT,
    max_continuity_shifts: int = _DEFAULT_SHIFT_LIMIT,
) -> NightlyInsightBundle:
    """Build the bounded nightly insight/continuity payload for artifacts."""

    new_insight_candidates = (
        *_personality_delta_lines(personality_results),
        *_reflection_lines(reflection_result),
    )
    continuity_candidates = _continuity_shift_lines(world_refresh_result)
    return NightlyInsightBundle(
        new_insights=_bounded_lines(new_insight_candidates, limit=max_new_insights),
        continuity_shifts=_bounded_lines(continuity_candidates, limit=max_continuity_shifts),
    )


__all__ = [
    "NightlyInsightBundle",
    "build_nightly_insight_bundle",
]
