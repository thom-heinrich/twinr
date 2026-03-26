"""Normalize reserve-card seed families and select a broader daily mix.

The reserve lane already receives candidates from multiple modules, but those
modules emit source- or implementation-shaped family names such as
``world_awareness`` or ``memory_follow_up``. That is useful for tracing, but
it is too fine-grained for diversity decisions. This module owns one explicit,
generic policy that:

- maps raw candidates onto broader conversational seed families
- exposes one coarse axis for mix balancing (``public``, ``personal``,
  ``setup``)
- greedily selects a bounded diverse subset before copy rewrite
- gives the day planner one shared family token for spacing decisions

The policy intentionally stays topic-agnostic. It relies only on existing
structured fields such as ``source``, ``candidate_family``, ``topic_id``,
``memory_goal``, and ``attention_state``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate

_LOCAL_WORLD_SOURCES = frozenset({"local_news", "regional_news", "place"})
_LOCAL_WORLD_SCOPES = frozenset({"local", "regional"})
_IDENTITY_DISCOVERY_TOPICS = frozenset({"basics", "companion_style"})
_RELATIONSHIP_DISCOVERY_TOPICS = frozenset({"social"})
_PREFERENCE_DISCOVERY_TOPICS = frozenset(
    {"interests", "hobbies", "routines", "pets", "no_goes", "health"}
)
_CONTINUITY_REFLECTION_KINDS = frozenset({"thread", "recent_turn_continuity", "conversation_context"})


def _compact_text(value: object | None, *, max_len: int = 64) -> str:
    """Collapse one arbitrary value into bounded single-line text."""

    compact = " ".join(str(value or "").split()).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _mapping(value: object | None) -> Mapping[str, object]:
    """Return one mapping or an empty mapping."""

    return value if isinstance(value, Mapping) else {}


def _text_tokens(value: object | None, *, max_len: int = 48) -> tuple[str, ...]:
    """Normalize one scalar or sequence into ordered lowercase tokens."""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        tokens: list[str] = []
        seen: set[str] = set()
        for entry in value:
            token = _compact_text(entry, max_len=max_len).casefold()
            if not token or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
        return tuple(tokens)
    token = _compact_text(value, max_len=max_len).casefold()
    return (token,) if token else ()


@dataclass(frozen=True, slots=True)
class DisplayReserveSeedProfile:
    """Describe one broad conversation family for diversity policy."""

    family: str
    axis: str
    source_group: str


@dataclass(frozen=True, slots=True)
class DisplayReserveDiversityPolicy:
    """Explicit generic policy weights for reserve candidate mixing."""

    family_novelty_bonus: float = 0.58
    axis_novelty_bonus: float = 0.22
    source_group_novelty_bonus: float = 0.10
    family_repeat_penalty: float = 0.26
    axis_repeat_penalty: float = 0.11
    source_group_repeat_penalty: float = 0.06
    consecutive_family_penalty: float = 0.18
    public_axis_soft_cap_ratio: float = 0.5
    public_axis_soft_penalty: float = 0.18
    max_setup_candidates: int = 1
    extra_setup_penalty: float = 0.36


DEFAULT_DISPLAY_RESERVE_DIVERSITY_POLICY = DisplayReserveDiversityPolicy()


def display_reserve_seed_profile(
    candidate: AmbientDisplayImpulseCandidate,
) -> DisplayReserveSeedProfile:
    """Return the normalized diversity profile for one reserve candidate."""

    context = _mapping(getattr(candidate, "generation_context", None))
    raw_family = (
        _compact_text(context.get("candidate_family"), max_len=48).casefold()
        or _compact_text(getattr(candidate, "candidate_family", None), max_len=48).casefold()
    )
    source = _compact_text(getattr(candidate, "source", None), max_len=48).casefold()
    attention_state = _compact_text(getattr(candidate, "attention_state", None), max_len=32).casefold()
    topic_id = _compact_text(context.get("topic_id"), max_len=48).casefold()
    invite_kind = _compact_text(context.get("invite_kind"), max_len=48).casefold()
    memory_goal = _compact_text(context.get("memory_goal"), max_len=48).casefold()
    reflection_kind = _compact_text(context.get("reflection_kind"), max_len=48).casefold()
    memory_domain = _compact_text(context.get("memory_domain"), max_len=48).casefold()
    scopes = {
        *_text_tokens(context.get("scope")),
        *_text_tokens(context.get("scopes")),
    }

    if source == "user_discovery" or raw_family == "user_discovery":
        if invite_kind == "review_profile":
            return DisplayReserveSeedProfile(
                family="profile_review",
                axis="personal",
                source_group="discovery",
            )
        if topic_id in _IDENTITY_DISCOVERY_TOPICS:
            return DisplayReserveSeedProfile(
                family="identity_setup",
                axis="setup",
                source_group="discovery",
            )
        if topic_id in _RELATIONSHIP_DISCOVERY_TOPICS:
            return DisplayReserveSeedProfile(
                family="relationship_discovery",
                axis="personal",
                source_group="discovery",
            )
        if topic_id in _PREFERENCE_DISCOVERY_TOPICS:
            return DisplayReserveSeedProfile(
                family="preference_discovery",
                axis="personal",
                source_group="discovery",
            )
        return DisplayReserveSeedProfile(
            family="discovery",
            axis="setup",
            source_group="discovery",
        )

    if raw_family == "memory_conflict" or memory_goal == "clarify_conflict":
        return DisplayReserveSeedProfile(
            family="memory_clarify",
            axis="personal",
            source_group="memory",
        )
    if raw_family == "memory_follow_up" or memory_goal == "gentle_follow_up":
        return DisplayReserveSeedProfile(
            family="memory_follow_up",
            axis="personal",
            source_group="memory",
        )

    if (
        raw_family == "memory_thread"
        or source in {"continuity", "relationship"}
        or attention_state == "shared_thread"
    ):
        return DisplayReserveSeedProfile(
            family="shared_thread",
            axis="personal",
            source_group="continuity",
        )

    if raw_family in {"reflection_summary", "reflection_packet"} or source.startswith("reflection"):
        if attention_state == "shared_thread" or reflection_kind in _CONTINUITY_REFLECTION_KINDS or memory_domain == "thread":
            return DisplayReserveSeedProfile(
                family="shared_thread",
                axis="personal",
                source_group="continuity",
            )
        return DisplayReserveSeedProfile(
            family="reflection",
            axis="personal",
            source_group="reflection",
        )

    if raw_family == "place" or source in _LOCAL_WORLD_SOURCES or scopes.intersection(_LOCAL_WORLD_SCOPES):
        return DisplayReserveSeedProfile(
            family="local_world",
            axis="public",
            source_group="world",
        )

    if (
        raw_family in {"world_awareness", "world_subscription", "world"}
        or source in {"world", "situational_awareness", "regional_news", "local_news"}
    ):
        return DisplayReserveSeedProfile(
            family="public_world",
            axis="public",
            source_group="world",
        )

    return DisplayReserveSeedProfile(
        family="general_interest",
        axis="personal",
        source_group="general",
    )


def reserve_seed_family(candidate: AmbientDisplayImpulseCandidate) -> str:
    """Return the normalized seed family for one reserve candidate."""

    return display_reserve_seed_profile(candidate).family


def reserve_seed_axis(candidate: AmbientDisplayImpulseCandidate) -> str:
    """Return the coarse diversity axis for one reserve candidate."""

    return display_reserve_seed_profile(candidate).axis


def _diversity_score(
    candidate: AmbientDisplayImpulseCandidate,
    *,
    ranked_index: int,
    ranked_count: int,
    selected: Sequence[AmbientDisplayImpulseCandidate],
    policy: DisplayReserveDiversityPolicy,
) -> float:
    """Return the greedy diversity-selection score for one candidate."""

    profile = display_reserve_seed_profile(candidate)
    selected_profiles = tuple(display_reserve_seed_profile(item) for item in selected)
    family_count = sum(1 for item in selected_profiles if item.family == profile.family)
    axis_count = sum(1 for item in selected_profiles if item.axis == profile.axis)
    source_group_count = sum(1 for item in selected_profiles if item.source_group == profile.source_group)
    setup_count = sum(1 for item in selected_profiles if item.axis == "setup")
    public_count = sum(1 for item in selected_profiles if item.axis == "public")
    base_rank_bonus = max(0.0, 0.34 - ((ranked_index / float(max(1, ranked_count))) * 0.34))
    score = max(0.0, float(candidate.salience)) + base_rank_bonus

    if family_count == 0:
        score += policy.family_novelty_bonus
    else:
        score -= policy.family_repeat_penalty * float(family_count)

    if axis_count == 0:
        score += policy.axis_novelty_bonus
    else:
        score -= policy.axis_repeat_penalty * float(axis_count)

    if source_group_count == 0:
        score += policy.source_group_novelty_bonus
    else:
        score -= policy.source_group_repeat_penalty * float(source_group_count)

    if selected and display_reserve_seed_profile(selected[-1]).family == profile.family:
        score -= policy.consecutive_family_penalty

    if profile.axis == "setup" and setup_count >= policy.max_setup_candidates:
        score -= policy.extra_setup_penalty * float(setup_count - policy.max_setup_candidates + 1)

    if profile.axis == "public":
        soft_cap = max(
            1,
            int(
                math.ceil(
                    policy.public_axis_soft_cap_ratio
                    * float(max(1, len(selected) + 1))
                )
            ),
        )
        if public_count >= soft_cap:
            score -= policy.public_axis_soft_penalty * float(public_count - soft_cap + 1)

    return score


def select_diverse_candidates(
    candidates: Sequence[AmbientDisplayImpulseCandidate],
    *,
    max_items: int,
    policy: DisplayReserveDiversityPolicy = DEFAULT_DISPLAY_RESERVE_DIVERSITY_POLICY,
) -> tuple[AmbientDisplayImpulseCandidate, ...]:
    """Return one bounded candidate subset with broader topic-family coverage."""

    ranked = tuple(candidate for candidate in candidates if _compact_text(candidate.topic_key, max_len=96))
    if not ranked:
        return ()
    limited_max = min(max(1, int(max_items)), len(ranked))
    remaining = list(enumerate(ranked))
    selected: list[AmbientDisplayImpulseCandidate] = []
    while remaining and len(selected) < limited_max:
        best_offset, (_ranked_index, best_candidate) = max(
            enumerate(remaining),
            key=lambda item: (
                _diversity_score(
                    item[1][1],
                    ranked_index=item[1][0],
                    ranked_count=len(ranked),
                    selected=selected,
                    policy=policy,
                ),
                -item[1][0],
            ),
        )
        selected.append(best_candidate)
        remaining.pop(best_offset)
    return tuple(selected)


__all__ = [
    "DEFAULT_DISPLAY_RESERVE_DIVERSITY_POLICY",
    "DisplayReserveDiversityPolicy",
    "DisplayReserveSeedProfile",
    "display_reserve_seed_profile",
    "reserve_seed_axis",
    "reserve_seed_family",
    "select_diverse_candidates",
]
