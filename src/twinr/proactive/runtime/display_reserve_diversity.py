# CHANGELOG: 2026-03-29
# BUG-1: select_diverse_candidates(max_items=0) previously still returned one
#   item because max_items was clamped to >= 1.
# BUG-2: candidates without topic_key were silently dropped, although the
#   module contract already allowed selection from structured metadata such as
#   topic_id/source/memory_goal.
# BUG-3: already-normalized family names (for example profile_review,
#   shared_thread, local_world, public_world) were not handled idempotently and
#   could regress to general_interest unless extra source hints were present.
# SEC-1: normalization of untrusted metadata was unbounded; oversized strings
#   and iterables could waste CPU/RAM on Raspberry Pi-class hardware. Scalar
#   normalization, iterable fan-in, and candidate fan-in are now bounded.
# IMP-1: replaced purely hand-tuned repeat penalties with an explicit
#   saturating coverage objective over structured facets (family, axis,
#   source_group, topic, goal, scope, continuity, extra tags).
# IMP-2: added stable fallback identities, duplicate-cluster suppression, and
#   precomputed candidate signals so the selector remains edge-friendly while
#   supporting richer structured metadata when upstream provides it.

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

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
import math

from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate


_LOCAL_WORLD_SOURCES = frozenset({"local_news", "regional_news", "place"})
_LOCAL_WORLD_SCOPES = frozenset({"local", "regional"})
_IDENTITY_DISCOVERY_TOPICS = frozenset({"basics", "companion_style"})
_RELATIONSHIP_DISCOVERY_TOPICS = frozenset({"social"})
_PREFERENCE_DISCOVERY_TOPICS = frozenset(
    {"interests", "hobbies", "routines", "pets", "no_goes", "health"}
)
_CONTINUITY_REFLECTION_KINDS = frozenset(
    {"thread", "recent_turn_continuity", "conversation_context"}
)
_EXTRA_DIVERSITY_TAG_KEYS = (
    "coverage_tags",
    "diversity_tags",
    "facet_tags",
    "novelty_tags",
    "semantic_tags",
)
_OPTIONAL_TEXT_FINGERPRINT_ATTRS = (
    "title",
    "topic_key",
    "seed_text",
    "text",
    "prompt",
    "label",
)
_MAX_NORMALIZATION_SCAN_CHARS = 4096
_MAX_NORMALIZED_SEQUENCE_ITEMS = 16


def _compact_text(
    value: object | None,
    *,
    max_len: int = 64,
    max_scan_chars: int = _MAX_NORMALIZATION_SCAN_CHARS,
) -> str:
    """Collapse one arbitrary value into bounded single-line text."""

    if value is None:
        return ""
    if isinstance(value, str):
        raw = value
    elif isinstance(value, (bytes, bytearray)):
        raw = bytes(value).decode("utf-8", errors="ignore")
    else:
        raw = str(value)
    if len(raw) > max_scan_chars:
        raw = raw[:max_scan_chars]
    compact = " ".join(raw.split()).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _mapping(value: object | None) -> Mapping[str, object]:
    """Return one mapping or an empty mapping."""

    return value if isinstance(value, Mapping) else {}


def _text_tokens(
    value: object | None,
    *,
    max_len: int = 48,
    max_items: int = _MAX_NORMALIZED_SEQUENCE_ITEMS,
) -> tuple[str, ...]:
    """Normalize one scalar or iterable into bounded ordered lowercase tokens."""

    if value is None:
        return ()
    if isinstance(value, Mapping):
        return ()
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        tokens: list[str] = []
        seen: set[str] = set()
        sequence_like = isinstance(value, Sequence)
        for index, entry in enumerate(value):
            if index >= max_items:
                break
            token = _compact_text(entry, max_len=max_len).casefold()
            if not token or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
        if not sequence_like:
            tokens.sort()
        return tuple(tokens)
    token = _compact_text(value, max_len=max_len).casefold()
    return (token,) if token else ()


def _unique_tokens(*token_groups: Iterable[str]) -> tuple[str, ...]:
    """Return deduplicated tokens while preserving first occurrence order."""

    ordered: list[str] = []
    seen: set[str] = set()
    for group in token_groups:
        for token in group:
            if not token or token in seen:
                continue
            seen.add(token)
            ordered.append(token)
    return tuple(ordered)


def _safe_float(value: object | None) -> float:
    """Return one finite float, falling back to zero for invalid values."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return number


@dataclass(frozen=True, slots=True)
class DisplayReserveSeedProfile:
    """Describe one broad conversation family for diversity policy."""

    family: str
    axis: str
    source_group: str


_KNOWN_NORMALIZED_FAMILY_PROFILES: Mapping[str, DisplayReserveSeedProfile] = {
    "profile_review": DisplayReserveSeedProfile(
        family="profile_review",
        axis="personal",
        source_group="discovery",
    ),
    "identity_setup": DisplayReserveSeedProfile(
        family="identity_setup",
        axis="setup",
        source_group="discovery",
    ),
    "relationship_discovery": DisplayReserveSeedProfile(
        family="relationship_discovery",
        axis="personal",
        source_group="discovery",
    ),
    "preference_discovery": DisplayReserveSeedProfile(
        family="preference_discovery",
        axis="personal",
        source_group="discovery",
    ),
    "discovery": DisplayReserveSeedProfile(
        family="discovery",
        axis="setup",
        source_group="discovery",
    ),
    "memory_clarify": DisplayReserveSeedProfile(
        family="memory_clarify",
        axis="personal",
        source_group="memory",
    ),
    "memory_follow_up": DisplayReserveSeedProfile(
        family="memory_follow_up",
        axis="personal",
        source_group="memory",
    ),
    "shared_thread": DisplayReserveSeedProfile(
        family="shared_thread",
        axis="personal",
        source_group="continuity",
    ),
    "reflection": DisplayReserveSeedProfile(
        family="reflection",
        axis="personal",
        source_group="reflection",
    ),
    "local_world": DisplayReserveSeedProfile(
        family="local_world",
        axis="public",
        source_group="world",
    ),
    "public_world": DisplayReserveSeedProfile(
        family="public_world",
        axis="public",
        source_group="world",
    ),
    "general_interest": DisplayReserveSeedProfile(
        family="general_interest",
        axis="personal",
        source_group="general",
    ),
}


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
    topic_novelty_bonus: float = 0.46
    topic_repeat_penalty: float = 0.24
    goal_novelty_bonus: float = 0.16
    goal_repeat_penalty: float = 0.07
    scope_novelty_bonus: float = 0.12
    scope_repeat_penalty: float = 0.05
    continuity_novelty_bonus: float = 0.16
    continuity_repeat_penalty: float = 0.07
    extra_tag_novelty_bonus: float = 0.10
    extra_tag_repeat_penalty: float = 0.04
    duplicate_cluster_penalty: float = 1.10
    exact_duplicate_penalty: float = 2.40
    coverage_decay: float = 0.55
    max_candidates_considered: int = 128
    rank_bonus_max: float = 0.34


DEFAULT_DISPLAY_RESERVE_DIVERSITY_POLICY = DisplayReserveDiversityPolicy()


@dataclass(frozen=True, slots=True)
class _ReserveCandidateSignals:
    """Normalized selection-time metadata for one candidate."""

    candidate: AmbientDisplayImpulseCandidate
    profile: DisplayReserveSeedProfile
    rank_index: int
    salience: float
    topic_token: str
    goal_tokens: tuple[str, ...]
    scope_tokens: tuple[str, ...]
    continuity_tokens: tuple[str, ...]
    extra_tag_tokens: tuple[str, ...]
    selection_key: str
    duplicate_key: str


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
    attention_state = _compact_text(
        getattr(candidate, "attention_state", None),
        max_len=32,
    ).casefold()
    topic_id = _compact_text(context.get("topic_id"), max_len=48).casefold()
    invite_kind = _compact_text(context.get("invite_kind"), max_len=48).casefold()
    memory_goal = _compact_text(context.get("memory_goal"), max_len=48).casefold()
    reflection_kind = _compact_text(context.get("reflection_kind"), max_len=48).casefold()
    memory_domain = _compact_text(context.get("memory_domain"), max_len=48).casefold()
    scopes = {
        *_text_tokens(context.get("scope")),
        *_text_tokens(context.get("scopes")),
    }
    worldish = (
        raw_family in {"world", "world_awareness", "world_subscription", "place", "local_world", "public_world"}
        or source in _LOCAL_WORLD_SOURCES
        or source in {"world", "situational_awareness"}
        or bool(scopes.intersection(_LOCAL_WORLD_SCOPES))
    )

    if source == "user_discovery" or raw_family == "user_discovery":
        if invite_kind == "review_profile":
            return _KNOWN_NORMALIZED_FAMILY_PROFILES["profile_review"]
        if topic_id in _IDENTITY_DISCOVERY_TOPICS:
            return _KNOWN_NORMALIZED_FAMILY_PROFILES["identity_setup"]
        if topic_id in _RELATIONSHIP_DISCOVERY_TOPICS:
            return _KNOWN_NORMALIZED_FAMILY_PROFILES["relationship_discovery"]
        if topic_id in _PREFERENCE_DISCOVERY_TOPICS:
            return _KNOWN_NORMALIZED_FAMILY_PROFILES["preference_discovery"]
        return _KNOWN_NORMALIZED_FAMILY_PROFILES["discovery"]

    if raw_family == "memory_conflict" or memory_goal == "clarify_conflict":
        return _KNOWN_NORMALIZED_FAMILY_PROFILES["memory_clarify"]
    if raw_family == "memory_follow_up" or memory_goal == "gentle_follow_up":
        return _KNOWN_NORMALIZED_FAMILY_PROFILES["memory_follow_up"]

    if (
        raw_family == "memory_thread"
        or source in {"continuity", "relationship"}
        or (attention_state == "shared_thread" and not worldish)
    ):
        return _KNOWN_NORMALIZED_FAMILY_PROFILES["shared_thread"]

    if raw_family in {"reflection", "reflection_summary", "reflection_packet"} or source.startswith("reflection"):
        if (
            attention_state == "shared_thread"
            or reflection_kind in _CONTINUITY_REFLECTION_KINDS
            or memory_domain == "thread"
        ):
            return _KNOWN_NORMALIZED_FAMILY_PROFILES["shared_thread"]
        return _KNOWN_NORMALIZED_FAMILY_PROFILES["reflection"]

    if raw_family == "place" or source in _LOCAL_WORLD_SOURCES or scopes.intersection(_LOCAL_WORLD_SCOPES):
        return _KNOWN_NORMALIZED_FAMILY_PROFILES["local_world"]

    if raw_family in {"world_awareness", "world_subscription", "world"} or source in {
        "world",
        "situational_awareness",
    }:
        return _KNOWN_NORMALIZED_FAMILY_PROFILES["public_world"]

    normalized = _KNOWN_NORMALIZED_FAMILY_PROFILES.get(raw_family)
    if normalized is not None:
        return normalized

    return _KNOWN_NORMALIZED_FAMILY_PROFILES["general_interest"]


def reserve_seed_family(candidate: AmbientDisplayImpulseCandidate) -> str:
    """Return the normalized seed family for one reserve candidate."""

    return display_reserve_seed_profile(candidate).family


def reserve_seed_axis(candidate: AmbientDisplayImpulseCandidate) -> str:
    """Return the coarse diversity axis for one reserve candidate."""

    return display_reserve_seed_profile(candidate).axis


def _rank_bonus(*, ranked_index: int, ranked_count: int, max_bonus: float) -> float:
    """Return a bounded bonus that preserves preference for earlier-ranked items."""

    if ranked_count <= 1:
        return max(0.0, max_bonus)
    return max(0.0, max_bonus - ((ranked_index / float(ranked_count)) * max_bonus))


def _saturating_novelty_gain(*, count: int, weight: float, decay: float) -> float:
    """Return the marginal gain for one facet under a saturating coverage objective."""

    clipped_decay = min(max(decay, 0.0), 0.999999)
    return max(0.0, weight) * (clipped_decay ** float(max(0, count)))


def _candidate_topic_token(
    candidate: AmbientDisplayImpulseCandidate,
    *,
    profile: DisplayReserveSeedProfile,
    context: Mapping[str, object],
) -> str:
    """Return one stable topic-level token for structured coverage and dedupe."""

    identity_tokens = _unique_tokens(
        _text_tokens(getattr(candidate, "topic_key", None), max_len=96, max_items=1),
        _text_tokens(context.get("topic_key"), max_len=96, max_items=1),
        _text_tokens(context.get("topic_id"), max_len=96, max_items=1),
        _text_tokens(context.get("semantic_fingerprint"), max_len=96, max_items=1),
        _text_tokens(context.get("cluster_id"), max_len=96, max_items=1),
        _text_tokens(context.get("entity_id"), max_len=96, max_items=1),
    )
    if identity_tokens:
        return identity_tokens[0]
    fallback = _unique_tokens(
        (profile.family,),
        _text_tokens(getattr(candidate, "source", None), max_len=32, max_items=1),
        _text_tokens(context.get("memory_goal"), max_len=32, max_items=1),
        _text_tokens(context.get("invite_kind"), max_len=32, max_items=1),
        _text_tokens(context.get("reflection_kind"), max_len=32, max_items=1),
        _text_tokens(getattr(candidate, "attention_state", None), max_len=32, max_items=1),
        _text_tokens(context.get("scope"), max_len=24, max_items=2),
        _text_tokens(context.get("scopes"), max_len=24, max_items=2),
    )
    return "|".join(fallback[:6])


def _candidate_duplicate_key(
    candidate: AmbientDisplayImpulseCandidate,
    *,
    topic_token: str,
    context: Mapping[str, object],
) -> str:
    """Return one duplicate-cluster key with topic-aware fallback behavior."""

    explicit = _unique_tokens(
        _text_tokens(context.get("duplicate_key"), max_len=96, max_items=1),
        _text_tokens(context.get("dedupe_key"), max_len=96, max_items=1),
    )
    if explicit:
        return explicit[0]
    text_fingerprint = _unique_tokens(
        *(
            _text_tokens(getattr(candidate, attr_name, None), max_len=96, max_items=1)
            for attr_name in _OPTIONAL_TEXT_FINGERPRINT_ATTRS
        )
    )
    if text_fingerprint:
        return "|".join((topic_token, text_fingerprint[0])) if topic_token else text_fingerprint[0]
    return topic_token


def _candidate_extra_tag_tokens(context: Mapping[str, object]) -> tuple[str, ...]:
    """Return optional structured diversity tags when upstream provides them."""

    return _unique_tokens(
        *(_text_tokens(context.get(key), max_len=32) for key in _EXTRA_DIVERSITY_TAG_KEYS)
    )


def _candidate_goal_tokens(context: Mapping[str, object]) -> tuple[str, ...]:
    """Return optional goal-like structured tokens used for coverage balancing."""

    return _unique_tokens(
        _text_tokens(context.get("memory_goal"), max_len=48),
        _text_tokens(context.get("invite_kind"), max_len=48),
    )


def _candidate_scope_tokens(context: Mapping[str, object]) -> tuple[str, ...]:
    """Return scope/location-like tokens for coverage balancing."""

    return _unique_tokens(
        _text_tokens(context.get("scope"), max_len=32),
        _text_tokens(context.get("scopes"), max_len=32),
    )


def _candidate_continuity_tokens(
    candidate: AmbientDisplayImpulseCandidate,
    *,
    context: Mapping[str, object],
) -> tuple[str, ...]:
    """Return continuity and memory-route tokens for structured balancing."""

    return _unique_tokens(
        _text_tokens(getattr(candidate, "attention_state", None), max_len=32),
        _text_tokens(context.get("reflection_kind"), max_len=32),
        _text_tokens(context.get("memory_domain"), max_len=32),
        _text_tokens(context.get("candidate_family"), max_len=32, max_items=1),
    )


def _build_candidate_signals(
    candidates: Sequence[AmbientDisplayImpulseCandidate],
    *,
    policy: DisplayReserveDiversityPolicy,
) -> tuple[_ReserveCandidateSignals, ...]:
    """Return bounded, normalized selection-time signals for the input candidates."""

    if not candidates:
        return ()

    try:
        candidate_limit = int(policy.max_candidates_considered)
    except (TypeError, ValueError):
        candidate_limit = 0
    if candidate_limit <= 0:
        return ()

    limited = tuple(candidates[:candidate_limit])
    signals: list[_ReserveCandidateSignals] = []
    for ranked_index, candidate in enumerate(limited):
        profile = display_reserve_seed_profile(candidate)
        context = _mapping(getattr(candidate, "generation_context", None))
        topic_token = _candidate_topic_token(candidate, profile=profile, context=context)
        if not topic_token:
            # The module can operate without candidate.topic_key, but it still
            # needs some structured identity to avoid duplicate collapse.
            fallback_identity = _unique_tokens(
                (profile.family,),
                _text_tokens(getattr(candidate, "source", None), max_len=32, max_items=1),
                _candidate_goal_tokens(context),
                _candidate_scope_tokens(context),
                _candidate_continuity_tokens(candidate, context=context),
            )
            topic_token = "|".join(fallback_identity[:6])
        if not topic_token:
            continue
        selection_key = "|".join(
            _unique_tokens(
                (topic_token,),
                _candidate_goal_tokens(context),
                _candidate_scope_tokens(context),
                _candidate_continuity_tokens(candidate, context=context),
                _candidate_extra_tag_tokens(context),
            )[:8]
        )
        signals.append(
            _ReserveCandidateSignals(
                candidate=candidate,
                profile=profile,
                rank_index=ranked_index,
                salience=_safe_float(getattr(candidate, "salience", 0.0)),
                topic_token=topic_token,
                goal_tokens=_candidate_goal_tokens(context),
                scope_tokens=_candidate_scope_tokens(context),
                continuity_tokens=_candidate_continuity_tokens(candidate, context=context),
                extra_tag_tokens=_candidate_extra_tag_tokens(context),
                selection_key=selection_key or topic_token,
                duplicate_key=_candidate_duplicate_key(candidate, topic_token=topic_token, context=context),
            )
        )
    return tuple(signals)


def _dedupe_exact_duplicates(
    signals: Sequence[_ReserveCandidateSignals],
    *,
    ranked_count: int,
    policy: DisplayReserveDiversityPolicy,
) -> tuple[_ReserveCandidateSignals, ...]:
    """Keep the highest-utility representative for exact duplicate selection keys."""

    if not signals:
        return ()
    best_by_key: dict[str, tuple[float, _ReserveCandidateSignals]] = {}
    for signal in signals:
        key = signal.selection_key or signal.topic_token
        utility = signal.salience + _rank_bonus(
            ranked_index=signal.rank_index,
            ranked_count=ranked_count,
            max_bonus=policy.rank_bonus_max,
        )
        existing = best_by_key.get(key)
        if existing is None or utility > existing[0] or (
            utility == existing[0] and signal.rank_index < existing[1].rank_index
        ):
            best_by_key[key] = (utility, signal)
    ordered = sorted(best_by_key.values(), key=lambda item: item[1].rank_index)
    return tuple(signal for _utility, signal in ordered)


def _coverage_bonus(
    tokens: Iterable[str],
    *,
    counts: Counter[str],
    novelty_bonus: float,
    repeat_penalty: float,
    policy: DisplayReserveDiversityPolicy,
) -> float:
    """Return one marginal coverage contribution for a token group."""

    score = 0.0
    for token in tokens:
        if not token:
            continue
        seen_count = counts[token]
        score += _saturating_novelty_gain(
            count=seen_count,
            weight=novelty_bonus,
            decay=policy.coverage_decay,
        )
        if seen_count > 0:
            score -= repeat_penalty * float(seen_count)
    return score


def _marginal_gain(
    signal: _ReserveCandidateSignals,
    *,
    ranked_count: int,
    selected: Sequence[_ReserveCandidateSignals],
    family_counts: Counter[str],
    axis_counts: Counter[str],
    source_group_counts: Counter[str],
    topic_counts: Counter[str],
    goal_counts: Counter[str],
    scope_counts: Counter[str],
    continuity_counts: Counter[str],
    extra_tag_counts: Counter[str],
    duplicate_counts: Counter[str],
    selection_key_counts: Counter[str],
    policy: DisplayReserveDiversityPolicy,
) -> float:
    """Return the greedy marginal gain for one candidate under the coverage objective."""

    score = max(0.0, signal.salience)
    score += _rank_bonus(
        ranked_index=signal.rank_index,
        ranked_count=ranked_count,
        max_bonus=policy.rank_bonus_max,
    )

    score += _coverage_bonus(
        (signal.profile.family,),
        counts=family_counts,
        novelty_bonus=policy.family_novelty_bonus,
        repeat_penalty=policy.family_repeat_penalty,
        policy=policy,
    )
    score += _coverage_bonus(
        (signal.profile.axis,),
        counts=axis_counts,
        novelty_bonus=policy.axis_novelty_bonus,
        repeat_penalty=policy.axis_repeat_penalty,
        policy=policy,
    )
    score += _coverage_bonus(
        (signal.profile.source_group,),
        counts=source_group_counts,
        novelty_bonus=policy.source_group_novelty_bonus,
        repeat_penalty=policy.source_group_repeat_penalty,
        policy=policy,
    )
    score += _coverage_bonus(
        (signal.topic_token,),
        counts=topic_counts,
        novelty_bonus=policy.topic_novelty_bonus,
        repeat_penalty=policy.topic_repeat_penalty,
        policy=policy,
    )
    score += _coverage_bonus(
        signal.goal_tokens,
        counts=goal_counts,
        novelty_bonus=policy.goal_novelty_bonus,
        repeat_penalty=policy.goal_repeat_penalty,
        policy=policy,
    )
    score += _coverage_bonus(
        signal.scope_tokens,
        counts=scope_counts,
        novelty_bonus=policy.scope_novelty_bonus,
        repeat_penalty=policy.scope_repeat_penalty,
        policy=policy,
    )
    score += _coverage_bonus(
        signal.continuity_tokens,
        counts=continuity_counts,
        novelty_bonus=policy.continuity_novelty_bonus,
        repeat_penalty=policy.continuity_repeat_penalty,
        policy=policy,
    )
    score += _coverage_bonus(
        signal.extra_tag_tokens,
        counts=extra_tag_counts,
        novelty_bonus=policy.extra_tag_novelty_bonus,
        repeat_penalty=policy.extra_tag_repeat_penalty,
        policy=policy,
    )

    if selected and selected[-1].profile.family == signal.profile.family:
        score -= policy.consecutive_family_penalty

    if signal.duplicate_key and duplicate_counts[signal.duplicate_key] > 0:
        score -= policy.duplicate_cluster_penalty * float(duplicate_counts[signal.duplicate_key])

    if signal.selection_key and selection_key_counts[signal.selection_key] > 0:
        score -= policy.exact_duplicate_penalty * float(selection_key_counts[signal.selection_key])

    setup_count = axis_counts["setup"]
    if signal.profile.axis == "setup" and setup_count >= policy.max_setup_candidates:
        score -= policy.extra_setup_penalty * float(setup_count - policy.max_setup_candidates + 1)

    if signal.profile.axis == "public":
        public_count = axis_counts["public"]
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


def _apply_signal_to_counts(
    signal: _ReserveCandidateSignals,
    *,
    family_counts: Counter[str],
    axis_counts: Counter[str],
    source_group_counts: Counter[str],
    topic_counts: Counter[str],
    goal_counts: Counter[str],
    scope_counts: Counter[str],
    continuity_counts: Counter[str],
    extra_tag_counts: Counter[str],
    duplicate_counts: Counter[str],
    selection_key_counts: Counter[str],
) -> None:
    """Update coverage counts after one signal has been selected."""

    family_counts[signal.profile.family] += 1
    axis_counts[signal.profile.axis] += 1
    source_group_counts[signal.profile.source_group] += 1
    if signal.topic_token:
        topic_counts[signal.topic_token] += 1
    for token in signal.goal_tokens:
        goal_counts[token] += 1
    for token in signal.scope_tokens:
        scope_counts[token] += 1
    for token in signal.continuity_tokens:
        continuity_counts[token] += 1
    for token in signal.extra_tag_tokens:
        extra_tag_counts[token] += 1
    if signal.duplicate_key:
        duplicate_counts[signal.duplicate_key] += 1
    if signal.selection_key:
        selection_key_counts[signal.selection_key] += 1


def _diversity_score(
    candidate: AmbientDisplayImpulseCandidate,
    *,
    ranked_index: int,
    ranked_count: int,
    selected: Sequence[AmbientDisplayImpulseCandidate],
    policy: DisplayReserveDiversityPolicy,
) -> float:
    """Compatibility wrapper for the previous private scoring helper."""

    target_signals = _build_candidate_signals((candidate,), policy=policy)
    if not target_signals:
        return float("-inf")
    target_signal = replace(target_signals[0], rank_index=ranked_index)

    selected_signals = _build_candidate_signals(tuple(selected), policy=policy)
    family_counts: Counter[str] = Counter()
    axis_counts: Counter[str] = Counter()
    source_group_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    goal_counts: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    continuity_counts: Counter[str] = Counter()
    extra_tag_counts: Counter[str] = Counter()
    duplicate_counts: Counter[str] = Counter()
    selection_key_counts: Counter[str] = Counter()
    for signal in selected_signals:
        _apply_signal_to_counts(
            signal,
            family_counts=family_counts,
            axis_counts=axis_counts,
            source_group_counts=source_group_counts,
            topic_counts=topic_counts,
            goal_counts=goal_counts,
            scope_counts=scope_counts,
            continuity_counts=continuity_counts,
            extra_tag_counts=extra_tag_counts,
            duplicate_counts=duplicate_counts,
            selection_key_counts=selection_key_counts,
        )
    return _marginal_gain(
        target_signal,
        ranked_count=ranked_count,
        selected=selected_signals,
        family_counts=family_counts,
        axis_counts=axis_counts,
        source_group_counts=source_group_counts,
        topic_counts=topic_counts,
        goal_counts=goal_counts,
        scope_counts=scope_counts,
        continuity_counts=continuity_counts,
        extra_tag_counts=extra_tag_counts,
        duplicate_counts=duplicate_counts,
        selection_key_counts=selection_key_counts,
        policy=policy,
    )


def select_diverse_candidates(
    candidates: Sequence[AmbientDisplayImpulseCandidate],
    *,
    max_items: int,
    policy: DisplayReserveDiversityPolicy = DEFAULT_DISPLAY_RESERVE_DIVERSITY_POLICY,
) -> tuple[AmbientDisplayImpulseCandidate, ...]:
    """Return one bounded candidate subset with broader topic-family coverage."""

    if not candidates:
        return ()
    try:
        requested_max = int(max_items)
    except (TypeError, ValueError):
        requested_max = 0
    if requested_max <= 0:
        return ()

    signals = _build_candidate_signals(candidates, policy=policy)
    if not signals:
        return ()

    ranked_count = len(signals)
    deduped = _dedupe_exact_duplicates(signals, ranked_count=ranked_count, policy=policy)
    if not deduped:
        return ()

    limited_max = min(requested_max, len(deduped))
    remaining = list(enumerate(deduped))
    selected: list[_ReserveCandidateSignals] = []

    family_counts: Counter[str] = Counter()
    axis_counts: Counter[str] = Counter()
    source_group_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    goal_counts: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    continuity_counts: Counter[str] = Counter()
    extra_tag_counts: Counter[str] = Counter()
    duplicate_counts: Counter[str] = Counter()
    selection_key_counts: Counter[str] = Counter()

    while remaining and len(selected) < limited_max:
        best_offset, (_remaining_index, best_signal) = max(
            enumerate(remaining),
            key=lambda item: (
                _marginal_gain(
                    item[1][1],
                    ranked_count=ranked_count,
                    selected=selected,
                    family_counts=family_counts,
                    axis_counts=axis_counts,
                    source_group_counts=source_group_counts,
                    topic_counts=topic_counts,
                    goal_counts=goal_counts,
                    scope_counts=scope_counts,
                    continuity_counts=continuity_counts,
                    extra_tag_counts=extra_tag_counts,
                    duplicate_counts=duplicate_counts,
                    selection_key_counts=selection_key_counts,
                    policy=policy,
                ),
                -item[1][1].rank_index,
            ),
        )
        selected.append(best_signal)
        _apply_signal_to_counts(
            best_signal,
            family_counts=family_counts,
            axis_counts=axis_counts,
            source_group_counts=source_group_counts,
            topic_counts=topic_counts,
            goal_counts=goal_counts,
            scope_counts=scope_counts,
            continuity_counts=continuity_counts,
            extra_tag_counts=extra_tag_counts,
            duplicate_counts=duplicate_counts,
            selection_key_counts=selection_key_counts,
        )
        remaining.pop(best_offset)

    return tuple(signal.candidate for signal in selected)


__all__ = [
    "DEFAULT_DISPLAY_RESERVE_DIVERSITY_POLICY",
    "DisplayReserveDiversityPolicy",
    "DisplayReserveSeedProfile",
    "display_reserve_seed_profile",
    "reserve_seed_axis",
    "reserve_seed_family",
    "select_diverse_candidates",
]
