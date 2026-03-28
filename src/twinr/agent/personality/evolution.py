# CHANGELOG: 2026-03-27
# BUG-1: Enabled relationship-topic learning by default; the old allow-list rejected all relationship deltas unless callers overrode the policy.
# BUG-2: Fixed relationship-salience decay so one stale signal decays once per decay window instead of re-decaying on every maintenance pass.
# BUG-3: Fixed background queue races that could drop or double-process signals when enqueue_* and process_pending() ran concurrently.
# BUG-4: Added bounded step sizes for style and relationship updates so one malformed signal cannot snap the personality to an extreme.
# SEC-1: Added prompt-memory hardening for persistent summaries/topics to reduce practical memory-poisoning / prompt-injection persistence.
# SEC-2: Added queue/store compaction and hard pending-queue limits to prevent practical RAM/disk exhaustion on Raspberry Pi deployments.
# IMP-1: Upgraded consolidation to support- and confidence-weighted aggregation with bounded prompt-facing evidence/history.
# IMP-2: Kept rejected deltas in the background store only; the foreground PersonalitySnapshot now carries accepted deltas only, matching the module contract.

"""Evolve structured personality state from persistent learning signals.

This module owns the background-learning path for the agent personality
package. It deliberately separates three concerns:

- policy: deterministic guardrails that decide when evidence may shape the
  promptable personality state
- evolution: pure state transforms from signals into deltas and updated
  snapshots
- background orchestration: queue pending signals, persist them via the store
  seam, and commit the latest snapshot

The foreground conversation runtime should read only the resulting
``PersonalitySnapshot``. Raw signals and rejected deltas stay in the background
path so Twinr can learn slowly without letting one conversation rewrite its
core stance.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import re
from threading import Lock, RLock
from typing import TypeVar

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality.models import (
    ContinuityThread,
    ConversationStyleProfile,
    HumorProfile,
    InteractionSignal,
    PersonalityDelta,
    PersonalitySnapshot,
    PlaceFocus,
    PlaceSignal,
    RelationshipSignal,
    WorldSignal,
)
from twinr.agent.personality.profile_defaults import (
    default_humor_profile,
    default_style_profile,
)
from twinr.agent.personality.signals import (
    RELATIONSHIP_TOPIC_AVERSION_DELTA_PREFIX,
    RELATIONSHIP_TOPIC_DELTA_PREFIX,
    STYLE_INITIATIVE_DELTA_TARGET,
    STYLE_VERBOSITY_DELTA_TARGET,
)
from twinr.agent.personality.store import (
    PersonalityEvolutionStore,
    PersonalitySnapshotStore,
    RemoteStatePersonalityEvolutionStore,
    RemoteStatePersonalitySnapshotStore,
)
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore

_ItemT = TypeVar("_ItemT")

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_ROLE_HEADER_RE = re.compile(r"(?im)^\s*(system|developer|assistant|tool)\s*:")
_UNSAFE_PERSISTENT_TEXT_MARKERS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard previous instructions",
    "forget previous instructions",
    "reveal the system prompt",
    "reveal your system prompt",
    "reveal hidden prompt",
    "developer message",
    "system prompt",
    "<system>",
    "</system>",
    "<assistant>",
    "</assistant>",
    "<developer>",
    "</developer>",
    "jailbreak",
    "exfiltrate",
    "bypass safety",
)


def _utcnow() -> datetime:
    """Return the current aware UTC timestamp."""

    return datetime.now(timezone.utc)


def _isoformat(value: datetime) -> str:
    """Render one aware timestamp in ISO-8601 UTC form."""

    return value.astimezone(timezone.utc).isoformat()


def _parse_iso_datetime(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp or return ``None`` for blank values."""

    if not value:
        return None

    candidate = str(value).strip()
    if not candidate:
        return None
    if candidate.endswith("Z"):
        candidate = f"{candidate[:-1]}+00:00"

    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _clamp(value: float, *, minimum: float, maximum: float) -> float:
    """Clamp a numeric value into an inclusive range."""

    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def _mean(values: Sequence[float], *, default: float = 0.0) -> float:
    """Return the arithmetic mean of a sequence or a default when empty."""

    if not values:
        return default
    return sum(values) / len(values)


def _weighted_mean(
    values: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
    default: float = 0.0,
) -> float:
    """Return the weighted mean of a sequence or a default when empty."""

    if not values:
        return default
    if weights is None:
        return _mean(values, default=default)

    total_weight = 0.0
    weighted_sum = 0.0
    for value, weight in zip(values, weights, strict=False):
        clamped_weight = max(0.0, float(weight))
        if clamped_weight <= 0.0:
            continue
        total_weight += clamped_weight
        weighted_sum += float(value) * clamped_weight
    if total_weight <= 0.0:
        return default
    return weighted_sum / total_weight


def _tail(items: Sequence[_ItemT], *, limit: int) -> tuple[_ItemT, ...]:
    """Return the last ``limit`` items of a sequence."""

    if limit <= 0:
        return ()
    if len(items) <= limit:
        return tuple(items)
    return tuple(items[-limit:])


def _merge_by_key(
    existing_items: Sequence[_ItemT],
    new_items: Sequence[_ItemT],
    *,
    key_fn: Callable[[_ItemT], object],
) -> tuple[_ItemT, ...]:
    """Merge ordered items by key while letting newer items replace older ones."""

    merged: dict[object, _ItemT] = {}
    order: list[object] = []
    for item in tuple(existing_items) + tuple(new_items):
        key = key_fn(item)
        if key not in merged:
            order.append(key)
        merged[key] = item
    return tuple(merged[key] for key in order)


def _bounded_evidence(
    existing_items: Sequence[str],
    new_items: Sequence[str],
    *,
    limit: int,
) -> tuple[str, ...]:
    """Append evidence ids while keeping only the bounded tail."""

    return _tail(tuple(existing_items) + tuple(new_items), limit=limit)


def _place_key(signal: PlaceSignal | PlaceFocus) -> str:
    """Return the stable merge key for place-aware context."""

    name = getattr(signal, "place_name", None) or getattr(signal, "name", "")
    geography = getattr(signal, "geography", None) or ""
    return f"{str(name).strip().casefold()}::{str(geography).strip().casefold()}"


def _world_key(signal: WorldSignal) -> str:
    """Return the stable merge key for world-aware context."""

    return "::".join(
        (
            signal.topic.strip().casefold(),
            (signal.region or "").strip().casefold(),
            signal.source.strip().casefold(),
        )
    )


def _delta_key(delta: PersonalityDelta) -> str:
    """Return the stable merge key for persistent personality deltas."""

    return delta.delta_id


def _relationship_topic_key(signal: RelationshipSignal) -> str:
    """Return the stable merge key for prompt-facing relationship signals."""

    return signal.topic.strip().casefold()


def _continuity_key(thread: ContinuityThread) -> str:
    """Return the stable merge key for continuity threads."""

    return thread.title.strip().casefold()


def _topic_target_parts(target: str) -> tuple[str | None, str | None]:
    """Return the relationship target family and topic for one delta target."""

    if target.startswith(RELATIONSHIP_TOPIC_DELTA_PREFIX):
        return "affinity", target[len(RELATIONSHIP_TOPIC_DELTA_PREFIX) :].strip().casefold()
    if target.startswith(RELATIONSHIP_TOPIC_AVERSION_DELTA_PREFIX):
        return "aversion", target[len(RELATIONSHIP_TOPIC_AVERSION_DELTA_PREFIX) :].strip().casefold()
    return None, None


def _display_topic_from_target(target: str, *, max_chars: int) -> str:
    """Return a safe display topic extracted from a relationship delta target."""

    _, topic = _topic_target_parts(target)
    normalized = _normalize_text(topic or "", max_chars=max_chars)
    return normalized


def _sensitive_context_present(signals: Sequence[InteractionSignal]) -> bool:
    """Return whether any signal in one group came from a sensitive context."""

    for signal in signals:
        metadata = signal.metadata or {}
        if metadata.get("sensitive_context") is True:
            return True
    return False


def _fresh_world_signals(
    world_signals: Sequence[WorldSignal],
    *,
    now: datetime,
) -> tuple[WorldSignal, ...]:
    """Return only world signals that are still fresh at ``now``."""

    kept: list[WorldSignal] = []
    for signal in world_signals:
        fresh_until = _parse_iso_datetime(signal.fresh_until)
        if fresh_until is not None and fresh_until <= now:
            continue
        kept.append(signal)
    return tuple(kept)


def _active_continuity_threads(
    continuity_threads: Sequence[ContinuityThread],
    *,
    now: datetime,
) -> tuple[ContinuityThread, ...]:
    """Return only continuity threads that have not expired at ``now``."""

    kept: list[ContinuityThread] = []
    for thread in continuity_threads:
        expires_at = _parse_iso_datetime(thread.expires_at)
        if expires_at is not None and expires_at <= now:
            continue
        kept.append(thread)
    return tuple(kept)


def _normalize_text(
    value: str | None,
    *,
    max_chars: int,
) -> str:
    """Normalize one user-controlled text field for safe persistence."""

    if not value:
        return ""
    normalized = _CONTROL_CHARS_RE.sub(" ", str(value)).replace("\r", "\n")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return ""
    if max_chars > 0 and len(normalized) > max_chars:
        return normalized[: max_chars - 1].rstrip() + "…"
    return normalized


def _looks_like_unsafe_persistent_text(value: str | None) -> bool:
    """Return whether text looks like prompt-injection content for core memory."""

    if not value:
        return False
    lowered = str(value).casefold()
    if any(marker in lowered for marker in _UNSAFE_PERSISTENT_TEXT_MARKERS):
        return True
    if _ROLE_HEADER_RE.search(str(value)) is not None:
        return True
    if "```" in str(value) and ("prompt" in lowered or "system" in lowered):
        return True
    return False


def _default_delta_summary(
    target: str,
    *,
    max_topic_chars: int,
    max_summary_chars: int,
) -> str:
    """Return a safe fallback summary for one persistent delta."""

    if target == STYLE_VERBOSITY_DELTA_TARGET:
        return _normalize_text(
            "Repeated feedback about preferred response length.",
            max_chars=max_summary_chars,
        )
    if target == STYLE_INITIATIVE_DELTA_TARGET:
        return _normalize_text(
            "Repeated feedback about preferred conversational proactivity.",
            max_chars=max_summary_chars,
        )
    if target == "humor.intensity":
        return _normalize_text(
            "Repeated feedback about preferred humor intensity.",
            max_chars=max_summary_chars,
        )

    family, _topic = _topic_target_parts(target)
    display_topic = _display_topic_from_target(target, max_chars=max_topic_chars)
    if family == "affinity" and display_topic:
        return _normalize_text(
            f"Stable positive preference for {display_topic}.",
            max_chars=max_summary_chars,
        )
    if family == "aversion" and display_topic:
        return _normalize_text(
            f"Stable aversion to {display_topic}.",
            max_chars=max_summary_chars,
        )
    return _normalize_text(
        "Repeated supported interaction feedback.",
        max_chars=max_summary_chars,
    )


def _safe_delta_summary(
    target: str,
    signals: Sequence[InteractionSignal],
    *,
    max_summary_chars: int,
    max_topic_chars: int,
) -> str:
    """Return a bounded, non-instructional summary for persistent memory."""

    fallback = _default_delta_summary(
        target,
        max_topic_chars=max_topic_chars,
        max_summary_chars=max_summary_chars,
    )
    best_summary = ""
    best_weight = float("-inf")
    for signal in signals:
        candidate = _normalize_text(
            signal.delta_summary or signal.summary,
            max_chars=max_summary_chars,
        )
        if not candidate:
            continue
        if _looks_like_unsafe_persistent_text(candidate):
            continue
        weight = _signal_weight(signal)
        if weight > best_weight:
            best_summary = candidate
            best_weight = weight
    return best_summary or fallback


def _signal_weight(signal: InteractionSignal) -> float:
    """Return the consolidation weight for one interaction signal."""

    evidence_count = max(1, int(signal.evidence_count))
    confidence = _clamp(float(signal.confidence), minimum=0.0, maximum=1.0)
    explicit_multiplier = 1.15 if signal.explicit_user_requested else 1.0
    return evidence_count * max(confidence, 0.01) * explicit_multiplier


def _stable_delta_id(
    *,
    target: str,
    signal_ids: Sequence[str],
    status: str,
    delta_value: float,
) -> str:
    """Return a deterministic compact delta id suitable for persistent storage."""

    # BREAKING: Newly generated delta ids use a compact v2 hash format instead
    # of concatenating every source signal id. Existing stored ids remain valid.
    safe_target = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", target).strip("_.:-") or "delta"
    safe_target = safe_target[:80]
    canonical_ids = "|".join(sorted({str(signal_id) for signal_id in signal_ids}))
    digest = sha256(
        f"v2|{target}|{status}|{delta_value:.6f}|{canonical_ids}".encode("utf-8")
    ).hexdigest()[:20]
    return f"v2:{safe_target}:{digest}"


def _accepted_signal_ids(deltas: Iterable[PersonalityDelta]) -> set[str]:
    """Return the interaction signal ids that already drove an accepted delta."""

    consumed: set[str] = set()
    for delta in deltas:
        if delta.status != "accepted":
            continue
        consumed.update(delta.source_signal_ids)
    return consumed


@dataclass(frozen=True, slots=True)
class PersonalityEvolutionPolicy:
    """Define deterministic gates for personality learning.

    Attributes:
        min_support_count: Minimum repeated support required before an implicit
            interaction signal may mutate promptable personality state.
        min_confidence: Minimum mean confidence for a group of signals to be
            accepted implicitly.
        min_explicit_confidence: Minimum confidence required even for one-shot
            explicit user requests before they may persist into core memory.
        max_humor_step: Largest single evolution step allowed for humor
            intensity in one processing run.
        max_style_step: Largest single evolution step allowed for verbosity or
            initiative in one processing run.
        max_relationship_step: Largest single salience adjustment allowed for a
            relationship topic in one processing run.
        max_humor_intensity: Upper cap for the learned humor intensity.
        relationship_decay_days: Age after which dormant relationship signals
            begin to decay when no fresh support refreshes them.
        relationship_decay_step: Salience step removed for every elapsed decay
            window.
        min_relationship_salience: Floor below which decayed relationship
            signals fall out of the prompt-facing snapshot.
        max_profile_evidence: Maximum evidence ids retained in prompt-facing
            style/humor profiles.
        max_prompt_deltas: Maximum accepted deltas retained in the foreground
            snapshot.
        max_persisted_deltas: Maximum delta history items retained in the
            background store.
        max_interaction_signals: Maximum raw interaction signals retained in the
            background store after compaction.
        max_place_signals: Maximum place signals retained in the background
            store.
        max_world_signals: Maximum world signals retained in the background
            store.
        max_continuity_threads: Maximum active continuity threads retained in
            the prompt-facing snapshot.
        max_relationship_signals: Maximum relationship signals retained in the
            prompt-facing snapshot.
        max_summary_chars: Maximum characters retained for persistent summaries.
        max_topic_chars: Maximum characters retained for relationship topics.
        supported_delta_targets: Structured delta targets that this version of
            the policy is allowed to apply. Entries ending in ``:`` act as
            target prefixes for families such as topic-affinity deltas.
    """

    min_support_count: int = 2
    min_confidence: float = 0.6
    min_explicit_confidence: float = 0.45
    max_humor_step: float = 0.08
    max_style_step: float = 0.12
    max_relationship_step: float = 0.18
    max_humor_intensity: float = 0.6
    relationship_decay_days: int = 30
    relationship_decay_step: float = 0.08
    min_relationship_salience: float = 0.08
    max_profile_evidence: int = 64
    max_prompt_deltas: int = 128
    max_persisted_deltas: int = 1024
    max_interaction_signals: int = 4096
    max_place_signals: int = 256
    max_world_signals: int = 256
    max_continuity_threads: int = 128
    max_relationship_signals: int = 64
    max_summary_chars: int = 240
    max_topic_chars: int = 80
    supported_delta_targets: tuple[str, ...] = (
        STYLE_VERBOSITY_DELTA_TARGET,
        STYLE_INITIATIVE_DELTA_TARGET,
        "humor.intensity",
        RELATIONSHIP_TOPIC_DELTA_PREFIX,
        RELATIONSHIP_TOPIC_AVERSION_DELTA_PREFIX,
    )

    def __post_init__(self) -> None:
        """Normalize policy fields into safe bounded values."""

        object.__setattr__(self, "min_support_count", max(1, int(self.min_support_count)))
        object.__setattr__(
            self,
            "min_confidence",
            _clamp(float(self.min_confidence), minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "min_explicit_confidence",
            _clamp(float(self.min_explicit_confidence), minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "max_humor_step",
            _clamp(abs(float(self.max_humor_step)), minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "max_style_step",
            _clamp(abs(float(self.max_style_step)), minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "max_relationship_step",
            _clamp(abs(float(self.max_relationship_step)), minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "max_humor_intensity",
            _clamp(float(self.max_humor_intensity), minimum=0.0, maximum=1.0),
        )
        object.__setattr__(self, "relationship_decay_days", max(1, int(self.relationship_decay_days)))
        object.__setattr__(
            self,
            "relationship_decay_step",
            _clamp(abs(float(self.relationship_decay_step)), minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "min_relationship_salience",
            _clamp(float(self.min_relationship_salience), minimum=0.0, maximum=1.0),
        )
        object.__setattr__(self, "max_profile_evidence", max(1, int(self.max_profile_evidence)))
        object.__setattr__(self, "max_prompt_deltas", max(1, int(self.max_prompt_deltas)))
        object.__setattr__(self, "max_persisted_deltas", max(1, int(self.max_persisted_deltas)))
        object.__setattr__(self, "max_interaction_signals", max(1, int(self.max_interaction_signals)))
        object.__setattr__(self, "max_place_signals", max(1, int(self.max_place_signals)))
        object.__setattr__(self, "max_world_signals", max(1, int(self.max_world_signals)))
        object.__setattr__(self, "max_continuity_threads", max(1, int(self.max_continuity_threads)))
        object.__setattr__(self, "max_relationship_signals", max(1, int(self.max_relationship_signals)))
        object.__setattr__(self, "max_summary_chars", max(32, int(self.max_summary_chars)))
        object.__setattr__(self, "max_topic_chars", max(8, int(self.max_topic_chars)))
        object.__setattr__(
            self,
            "supported_delta_targets",
            tuple(str(item).strip() for item in self.supported_delta_targets if str(item).strip()),
        )

    def supports_target(self, target: str) -> bool:
        """Return whether one delta target is allowed by this policy version."""

        normalized_target = str(target).strip()
        for allowed_target in self.supported_delta_targets:
            if allowed_target.endswith(":"):
                if normalized_target.startswith(allowed_target):
                    return True
                continue
            if normalized_target == allowed_target:
                return True
        return False


@dataclass(frozen=True, slots=True)
class PersonalityEvolutionResult:
    """Return the result of one policy-gated evolution pass."""

    snapshot: PersonalitySnapshot
    accepted_deltas: tuple[PersonalityDelta, ...] = ()
    rejected_deltas: tuple[PersonalityDelta, ...] = ()


@dataclass(frozen=True, slots=True)
class _PendingBatch:
    """One drained batch of pending background-learning items."""

    interaction_signals: tuple[InteractionSignal, ...] = ()
    place_signals: tuple[PlaceSignal, ...] = ()
    world_signals: tuple[WorldSignal, ...] = ()
    continuity_threads: tuple[ContinuityThread, ...] = ()

    def has_items(self) -> bool:
        """Return whether the batch contains any pending items."""

        return bool(
            self.interaction_signals
            or self.place_signals
            or self.world_signals
            or self.continuity_threads
        )


@dataclass(slots=True)
class PersonalityEvolutionLoop:
    """Convert learning signals into a new promptable personality snapshot."""

    policy: PersonalityEvolutionPolicy = field(default_factory=PersonalityEvolutionPolicy)
    now_provider: Callable[[], datetime] = _utcnow

    def maintain_snapshot(
        self,
        *,
        snapshot: PersonalitySnapshot,
    ) -> PersonalitySnapshot:
        """Apply decay and freshness maintenance without consuming new signals."""

        now = self.now_provider().astimezone(timezone.utc)
        return self._maintain_snapshot(snapshot=snapshot, now=now)

    def evolve(
        self,
        *,
        snapshot: PersonalitySnapshot,
        interaction_signals: Sequence[InteractionSignal],
        place_signals: Sequence[PlaceSignal],
        world_signals: Sequence[WorldSignal],
        continuity_threads: Sequence[ContinuityThread] = (),
    ) -> PersonalityEvolutionResult:
        """Apply one policy-gated learning pass to the current snapshot."""

        now = self.now_provider().astimezone(timezone.utc)
        maintained_snapshot = self._maintain_snapshot(snapshot=snapshot, now=now)
        accepted_deltas, rejected_deltas = self._propose_deltas(
            interaction_signals=interaction_signals,
            now=now,
        )
        evolved_snapshot = self._apply(
            snapshot=maintained_snapshot,
            accepted_deltas=accepted_deltas,
            place_signals=place_signals,
            world_signals=world_signals,
            continuity_threads=continuity_threads,
            now=now,
        )
        return PersonalityEvolutionResult(
            snapshot=evolved_snapshot,
            accepted_deltas=accepted_deltas,
            rejected_deltas=rejected_deltas,
        )

    def _maintain_snapshot(
        self,
        *,
        snapshot: PersonalitySnapshot,
        now: datetime,
    ) -> PersonalitySnapshot:
        """Decay and prune time-sensitive prompt state before applying deltas."""

        relationship_signals = self._prune_relationship_signals(
            self._decay_relationship_signals(
                relationship_signals=snapshot.relationship_signals,
                now=now,
            )
        )
        continuity_threads = self._prune_continuity_threads(
            _active_continuity_threads(snapshot.continuity_threads, now=now)
        )
        world_signals = self._prune_world_signals(
            _fresh_world_signals(snapshot.world_signals, now=now)
        )
        personality_deltas = self._prompt_personality_deltas(snapshot.personality_deltas)
        return PersonalitySnapshot(
            schema_version=snapshot.schema_version,
            generated_at=snapshot.generated_at,
            core_traits=snapshot.core_traits,
            style_profile=self._prune_style_profile(snapshot.style_profile),
            humor_profile=self._prune_humor_profile(snapshot.humor_profile),
            relationship_signals=relationship_signals,
            continuity_threads=continuity_threads,
            place_focuses=snapshot.place_focuses,
            world_signals=world_signals,
            reflection_deltas=tuple(delta.to_reflection_delta() for delta in personality_deltas),
            personality_deltas=personality_deltas,
        )

    def _prune_style_profile(
        self,
        style_profile: ConversationStyleProfile | None,
    ) -> ConversationStyleProfile | None:
        """Return a bounded style profile suitable for prompt-facing state."""

        if style_profile is None:
            return None
        bounded_evidence = _bounded_evidence(
            style_profile.evidence,
            (),
            limit=self.policy.max_profile_evidence,
        )
        if bounded_evidence == tuple(style_profile.evidence):
            return style_profile
        return ConversationStyleProfile(
            verbosity=style_profile.verbosity,
            initiative=style_profile.initiative,
            evidence=bounded_evidence,
        )

    def _prune_humor_profile(
        self,
        humor_profile: HumorProfile | None,
    ) -> HumorProfile | None:
        """Return a bounded humor profile suitable for prompt-facing state."""

        if humor_profile is None:
            return None
        bounded_evidence = _bounded_evidence(
            humor_profile.evidence,
            (),
            limit=self.policy.max_profile_evidence,
        )
        if bounded_evidence == tuple(humor_profile.evidence):
            return humor_profile
        return HumorProfile(
            style=humor_profile.style,
            summary=humor_profile.summary,
            intensity=humor_profile.intensity,
            boundaries=humor_profile.boundaries,
            evidence=bounded_evidence,
        )

    def _prune_relationship_signals(
        self,
        relationship_signals: Sequence[RelationshipSignal],
    ) -> tuple[RelationshipSignal, ...]:
        """Keep only the strongest prompt-facing relationship signals."""

        ranked = sorted(
            relationship_signals,
            key=lambda signal: (
                float(signal.salience),
                _parse_iso_datetime(signal.updated_at) or datetime.min.replace(tzinfo=timezone.utc),
                signal.topic.casefold(),
            ),
            reverse=True,
        )
        return tuple(ranked[: self.policy.max_relationship_signals])

    def _prune_world_signals(
        self,
        world_signals: Sequence[WorldSignal],
    ) -> tuple[WorldSignal, ...]:
        """Keep only the freshest world signals for the prompt-facing snapshot."""

        ranked = sorted(
            world_signals,
            key=lambda signal: (
                _parse_iso_datetime(signal.fresh_until) or datetime.max.replace(tzinfo=timezone.utc),
                signal.topic.casefold(),
                (signal.region or "").casefold(),
            ),
            reverse=True,
        )
        return tuple(ranked[: self.policy.max_world_signals])

    def _prune_continuity_threads(
        self,
        continuity_threads: Sequence[ContinuityThread],
    ) -> tuple[ContinuityThread, ...]:
        """Keep only the most relevant active continuity threads."""

        ranked = sorted(
            continuity_threads,
            key=lambda thread: (
                _parse_iso_datetime(thread.expires_at) or datetime.max.replace(tzinfo=timezone.utc),
                thread.title.casefold(),
            ),
            reverse=True,
        )
        return tuple(ranked[: self.policy.max_continuity_threads])

    def _prompt_personality_deltas(
        self,
        delta_history: Sequence[PersonalityDelta],
    ) -> tuple[PersonalityDelta, ...]:
        """Return the accepted delta subset that may appear in the snapshot."""

        accepted = tuple(delta for delta in delta_history if delta.status == "accepted")
        return _tail(accepted, limit=self.policy.max_prompt_deltas)

    def _decay_relationship_signals(
        self,
        *,
        relationship_signals: Sequence[RelationshipSignal],
        now: datetime,
    ) -> tuple[RelationshipSignal, ...]:
        """Decay dormant relationship signals based on their last update time."""

        decayed: list[RelationshipSignal] = []
        decay_window = timedelta(days=self.policy.relationship_decay_days)
        for signal in relationship_signals:
            updated_at = _parse_iso_datetime(signal.updated_at)
            if updated_at is None:
                decayed.append(signal)
                continue

            age = now - updated_at
            if age < decay_window:
                decayed.append(signal)
                continue

            elapsed_windows = int(age // decay_window)
            if elapsed_windows <= 0:
                decayed.append(signal)
                continue

            reduced_salience = _clamp(
                signal.salience - (elapsed_windows * self.policy.relationship_decay_step),
                minimum=0.0,
                maximum=1.0,
            )
            if reduced_salience < self.policy.min_relationship_salience:
                continue

            decay_anchor = updated_at + (decay_window * elapsed_windows)
            decayed.append(
                RelationshipSignal(
                    topic=signal.topic,
                    summary=signal.summary,
                    salience=reduced_salience,
                    source=signal.source,
                    stance=signal.stance,
                    updated_at=_isoformat(decay_anchor),
                )
            )
        return tuple(decayed)

    def _propose_deltas(
        self,
        *,
        interaction_signals: Sequence[InteractionSignal],
        now: datetime,
    ) -> tuple[tuple[PersonalityDelta, ...], tuple[PersonalityDelta, ...]]:
        """Group interaction signals into accepted and rejected deltas."""

        grouped_signals: dict[str, list[InteractionSignal]] = {}
        for signal in interaction_signals:
            if signal.delta_target is None or signal.delta_value is None:
                continue
            target = str(signal.delta_target).strip()
            if not target:
                continue
            grouped_signals.setdefault(target, []).append(signal)

        prioritized_groups = {
            target: self._prioritize_explicit_signals(grouped)
            for target, grouped in grouped_signals.items()
        }
        contradictions = self._contradictions(prioritized_groups)
        accepted: list[PersonalityDelta] = []
        rejected: list[PersonalityDelta] = []

        for target, grouped in prioritized_groups.items():
            family, topic = _topic_target_parts(target)
            if family is not None:
                safe_topic = _display_topic_from_target(
                    target,
                    max_chars=self.policy.max_topic_chars,
                )
                if not safe_topic or _looks_like_unsafe_persistent_text(topic):
                    rejected.append(
                        self._build_delta(
                            target=target,
                            signals=grouped,
                            status="rejected",
                            rationale="Unsafe or malformed relationship topic for persistent memory.",
                            delta_value=0.0,
                            now=now,
                        )
                    )
                    continue

            if target in contradictions:
                rejected.append(
                    self._build_delta(
                        target=target,
                        signals=grouped,
                        status="rejected",
                        rationale=contradictions[target],
                        delta_value=0.0,
                        now=now,
                    )
                )
                continue
            if not self.policy.supports_target(target):
                rejected.append(
                    self._build_delta(
                        target=target,
                        signals=grouped,
                        status="rejected",
                        rationale="Unsupported delta target for this policy version.",
                        delta_value=0.0,
                        now=now,
                    )
                )
                continue

            support_weights = [max(1, int(signal.evidence_count)) for signal in grouped]
            support_count = sum(support_weights)
            mean_confidence = _weighted_mean(
                [float(signal.confidence) for signal in grouped],
                weights=support_weights,
                default=0.0,
            )
            explicit_request = any(signal.explicit_user_requested for signal in grouped)

            if (
                not explicit_request
                and target in {"humor.intensity", STYLE_INITIATIVE_DELTA_TARGET}
                and _sensitive_context_present(grouped)
                and any((signal.delta_value or 0.0) > 0.0 for signal in grouped)
            ):
                rejected.append(
                    self._build_delta(
                        target=target,
                        signals=grouped,
                        status="rejected",
                        rationale="Sensitive-context turns must not increase humor or initiative implicitly.",
                        delta_value=0.0,
                        now=now,
                    )
                )
                continue
            if explicit_request and mean_confidence < self.policy.min_explicit_confidence:
                rejected.append(
                    self._build_delta(
                        target=target,
                        signals=grouped,
                        status="rejected",
                        rationale="Explicit preference signal is too uncertain for persistent memory.",
                        delta_value=0.0,
                        now=now,
                    )
                )
                continue
            if not explicit_request and support_count < self.policy.min_support_count:
                rejected.append(
                    self._build_delta(
                        target=target,
                        signals=grouped,
                        status="rejected",
                        rationale="Insufficient repeated support for implicit personality drift.",
                        delta_value=0.0,
                        now=now,
                    )
                )
                continue
            if not explicit_request and mean_confidence < self.policy.min_confidence:
                rejected.append(
                    self._build_delta(
                        target=target,
                        signals=grouped,
                        status="rejected",
                        rationale="Signal confidence is too weak for a stable personality update.",
                        delta_value=0.0,
                        now=now,
                    )
                )
                continue

            proposed_delta_value = _weighted_mean(
                [float(signal.delta_value) for signal in grouped if signal.delta_value is not None],
                weights=[_signal_weight(signal) for signal in grouped if signal.delta_value is not None],
                default=0.0,
            )
            bounded_delta_value = self._bound_delta_value(target=target, delta_value=proposed_delta_value)

            accepted.append(
                self._build_delta(
                    target=target,
                    signals=grouped,
                    status="accepted",
                    rationale=(
                        "Explicit user-requested preference with sufficient confidence."
                        if explicit_request
                        else "Repeated supported interaction feedback."
                    ),
                    delta_value=bounded_delta_value,
                    now=now,
                )
            )
        return tuple(accepted), tuple(rejected)

    def _bound_delta_value(self, *, target: str, delta_value: float) -> float:
        """Return the bounded delta step for one target."""

        if target == "humor.intensity":
            return _clamp(
                delta_value,
                minimum=-self.policy.max_humor_step,
                maximum=self.policy.max_humor_step,
            )
        if target in {STYLE_VERBOSITY_DELTA_TARGET, STYLE_INITIATIVE_DELTA_TARGET}:
            return _clamp(
                delta_value,
                minimum=-self.policy.max_style_step,
                maximum=self.policy.max_style_step,
            )
        family, _topic = _topic_target_parts(target)
        if family is not None:
            return _clamp(
                delta_value,
                minimum=-self.policy.max_relationship_step,
                maximum=self.policy.max_relationship_step,
            )
        return delta_value

    def _prioritize_explicit_signals(
        self,
        signals: Sequence[InteractionSignal],
    ) -> tuple[InteractionSignal, ...]:
        """Prefer explicit user requests when a target has mixed evidence."""

        explicit = tuple(signal for signal in signals if signal.explicit_user_requested)
        if explicit:
            return explicit
        return tuple(signals)

    def _contradictions(
        self,
        grouped_signals: dict[str, tuple[InteractionSignal, ...]],
    ) -> dict[str, str]:
        """Return contradictory targets and their rejection rationales."""

        contradictions: dict[str, str] = {}
        for target, grouped in grouped_signals.items():
            values = [signal.delta_value for signal in grouped if signal.delta_value is not None]
            if any(value > 0.0 for value in values) and any(value < 0.0 for value in values):
                contradictions[target] = "Contradictory feedback points the same target in opposite directions."

        topic_groups: dict[str, list[tuple[str, tuple[InteractionSignal, ...]]]] = defaultdict(list)
        for target, grouped in grouped_signals.items():
            family, topic_key = _topic_target_parts(target)
            if family is None or topic_key is None:
                continue
            topic_groups[topic_key].append((target, grouped))

        for _topic_key, entries in topic_groups.items():
            families = {(_topic_target_parts(target)[0] or "") for target, _grouped in entries}
            if families != {"affinity", "aversion"}:
                continue
            explicit_targets = [
                target
                for target, grouped in entries
                if any(signal.explicit_user_requested for signal in grouped)
            ]
            if len(explicit_targets) == 1:
                for target, _grouped in entries:
                    if target == explicit_targets[0]:
                        continue
                    contradictions[target] = "Explicit preference overrides contradictory implicit topic feedback."
                continue
            for target, _grouped in entries:
                contradictions[target] = "Contradictory affinity and aversion evidence for the same topic."
        return contradictions

    def _build_delta(
        self,
        *,
        target: str,
        signals: Sequence[InteractionSignal],
        status: str,
        rationale: str,
        delta_value: float,
        now: datetime,
    ) -> PersonalityDelta:
        """Build one persistent delta from a homogeneous signal group."""

        support_count = sum(max(1, int(signal.evidence_count)) for signal in signals)
        summary = _safe_delta_summary(
            target,
            signals,
            max_summary_chars=self.policy.max_summary_chars,
            max_topic_chars=self.policy.max_topic_chars,
        )
        signal_ids = tuple(signal.signal_id for signal in signals)
        confidence = _weighted_mean(
            [float(signal.confidence) for signal in signals],
            weights=[max(1, int(signal.evidence_count)) for signal in signals],
            default=0.0,
        )
        return PersonalityDelta(
            delta_id=_stable_delta_id(
                target=target,
                signal_ids=signal_ids,
                status=status,
                delta_value=delta_value,
            ),
            target=target,
            summary=summary,
            rationale=_normalize_text(rationale, max_chars=self.policy.max_summary_chars),
            delta_value=delta_value,
            confidence=confidence,
            support_count=support_count,
            source_signal_ids=signal_ids,
            status=status,
            explicit_user_requested=any(signal.explicit_user_requested for signal in signals),
            review_at=_isoformat(now),
        )

    def _apply(
        self,
        *,
        snapshot: PersonalitySnapshot,
        accepted_deltas: Sequence[PersonalityDelta],
        place_signals: Sequence[PlaceSignal],
        world_signals: Sequence[WorldSignal],
        continuity_threads: Sequence[ContinuityThread],
        now: datetime,
    ) -> PersonalitySnapshot:
        """Apply accepted deltas and context signals to the snapshot."""

        style_profile = self._prune_style_profile(snapshot.style_profile)
        humor_profile = self._prune_humor_profile(snapshot.humor_profile)
        relationship_signals = snapshot.relationship_signals

        for delta in accepted_deltas:
            if delta.target == STYLE_VERBOSITY_DELTA_TARGET:
                style_baseline = style_profile or default_style_profile()
                style_profile = ConversationStyleProfile(
                    verbosity=_clamp(
                        style_baseline.verbosity + delta.delta_value,
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    initiative=style_baseline.initiative,
                    evidence=_bounded_evidence(
                        style_baseline.evidence,
                        delta.source_signal_ids,
                        limit=self.policy.max_profile_evidence,
                    ),
                )
                continue

            if delta.target == STYLE_INITIATIVE_DELTA_TARGET:
                style_baseline = style_profile or default_style_profile()
                style_profile = ConversationStyleProfile(
                    verbosity=style_baseline.verbosity,
                    initiative=_clamp(
                        style_baseline.initiative + delta.delta_value,
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    evidence=_bounded_evidence(
                        style_baseline.evidence,
                        delta.source_signal_ids,
                        limit=self.policy.max_profile_evidence,
                    ),
                )
                continue

            if delta.target == "humor.intensity":
                humor_baseline = humor_profile or default_humor_profile()
                humor_profile = HumorProfile(
                    style=humor_baseline.style,
                    summary=humor_baseline.summary,
                    intensity=_clamp(
                        humor_baseline.intensity + delta.delta_value,
                        minimum=0.0,
                        maximum=self.policy.max_humor_intensity,
                    ),
                    boundaries=humor_baseline.boundaries,
                    evidence=_bounded_evidence(
                        humor_baseline.evidence,
                        delta.source_signal_ids,
                        limit=self.policy.max_profile_evidence,
                    ),
                )
                continue

            family, topic_key = _topic_target_parts(delta.target)
            display_topic = _display_topic_from_target(
                delta.target,
                max_chars=self.policy.max_topic_chars,
            )
            if family is None or topic_key is None or not display_topic:
                continue

            current_signal = next(
                (item for item in relationship_signals if _relationship_topic_key(item) == topic_key),
                None,
            )
            current_salience = (
                current_signal.salience
                if current_signal is not None and current_signal.stance == family
                else 0.0
            )
            updated_salience = _clamp(
                current_salience + delta.delta_value,
                minimum=0.0,
                maximum=1.0,
            )

            relationship_signals = tuple(
                item for item in relationship_signals if _relationship_topic_key(item) != topic_key
            )
            if updated_salience < self.policy.min_relationship_salience:
                continue

            relationship_signals = _merge_by_key(
                relationship_signals,
                (
                    RelationshipSignal(
                        topic=display_topic,
                        summary=_normalize_text(
                            delta.summary,
                            max_chars=self.policy.max_summary_chars,
                        ),
                        salience=updated_salience,
                        source="personality_learning",
                        stance=family,
                        updated_at=_isoformat(now),
                    ),
                ),
                key_fn=_relationship_topic_key,
            )

        place_focuses = _merge_by_key(
            snapshot.place_focuses,
            tuple(signal.to_place_focus() for signal in place_signals),
            key_fn=_place_key,
        )
        merged_world_signals = self._prune_world_signals(
            _fresh_world_signals(
                _merge_by_key(snapshot.world_signals, tuple(world_signals), key_fn=_world_key),
                now=now,
            )
        )
        merged_continuity_threads = self._prune_continuity_threads(
            _active_continuity_threads(
                _merge_by_key(snapshot.continuity_threads, tuple(continuity_threads), key_fn=_continuity_key),
                now=now,
            )
        )
        merged_personality_deltas = self._prompt_personality_deltas(
            _merge_by_key(
                snapshot.personality_deltas,
                tuple(accepted_deltas),
                key_fn=_delta_key,
            )
        )

        return PersonalitySnapshot(
            schema_version=snapshot.schema_version,
            generated_at=_isoformat(now),
            core_traits=snapshot.core_traits,
            style_profile=style_profile,
            humor_profile=humor_profile,
            relationship_signals=self._prune_relationship_signals(relationship_signals),
            continuity_threads=merged_continuity_threads,
            place_focuses=place_focuses,
            world_signals=merged_world_signals,
            reflection_deltas=tuple(delta.to_reflection_delta() for delta in merged_personality_deltas),
            personality_deltas=merged_personality_deltas,
        )


@dataclass(slots=True)
class BackgroundPersonalityEvolutionLoop:
    """Persist signals, apply policy-gated learning, and commit snapshots."""

    config: TwinrConfig
    remote_state: LongTermRemoteStateStore | None = None
    evolution_store: PersonalityEvolutionStore = field(
        default_factory=RemoteStatePersonalityEvolutionStore
    )
    snapshot_store: PersonalitySnapshotStore = field(
        default_factory=RemoteStatePersonalitySnapshotStore
    )
    evolution_loop: PersonalityEvolutionLoop = field(default_factory=PersonalityEvolutionLoop)
    max_pending_items_per_queue: int = 2048
    _pending_interaction_signals: deque[InteractionSignal] = field(default_factory=deque, init=False, repr=False)
    _pending_place_signals: deque[PlaceSignal] = field(default_factory=deque, init=False, repr=False)
    _pending_world_signals: deque[WorldSignal] = field(default_factory=deque, init=False, repr=False)
    _pending_continuity_threads: deque[ContinuityThread] = field(default_factory=deque, init=False, repr=False)
    _queue_lock: RLock = field(default_factory=RLock, init=False, repr=False)
    _process_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize queue limits."""

        self.max_pending_items_per_queue = max(1, int(self.max_pending_items_per_queue))

    def enqueue_interaction_signal(self, signal: InteractionSignal) -> None:
        """Queue one interaction signal for the next background processing pass."""

        self._enqueue_pending(
            queue=self._pending_interaction_signals,
            item=signal,
            label="interaction signal",
        )

    def enqueue_place_signal(self, signal: PlaceSignal) -> None:
        """Queue one place signal for the next background processing pass."""

        self._enqueue_pending(
            queue=self._pending_place_signals,
            item=signal,
            label="place signal",
        )

    def enqueue_world_signal(self, signal: WorldSignal) -> None:
        """Queue one world signal for the next background processing pass."""

        self._enqueue_pending(
            queue=self._pending_world_signals,
            item=signal,
            label="world signal",
        )

    def enqueue_continuity_thread(self, thread: ContinuityThread) -> None:
        """Queue one continuity thread refresh for the next processing pass."""

        self._enqueue_pending(
            queue=self._pending_continuity_threads,
            item=thread,
            label="continuity thread",
        )

    def _enqueue_pending(
        self,
        *,
        queue: deque[_ItemT],
        item: _ItemT,
        label: str,
    ) -> None:
        """Append one pending item with a hard safety bound."""

        with self._queue_lock:
            if len(queue) >= self.max_pending_items_per_queue:
                # BREAKING: Overload now fails fast instead of allowing unbounded
                # in-memory growth that can exhaust a Raspberry Pi 4.
                raise OverflowError(
                    f"Pending {label} queue reached the hard limit of "
                    f"{self.max_pending_items_per_queue} items."
                )
            queue.append(item)

    def has_pending_items(self) -> bool:
        """Return whether any learning signals are waiting for commit."""

        with self._queue_lock:
            return bool(
                self._pending_interaction_signals
                or self._pending_place_signals
                or self._pending_world_signals
                or self._pending_continuity_threads
            )

    def _drain_pending_batch(self) -> _PendingBatch:
        """Atomically move all currently pending items into one process batch."""

        with self._queue_lock:
            batch = _PendingBatch(
                interaction_signals=tuple(self._pending_interaction_signals),
                place_signals=tuple(self._pending_place_signals),
                world_signals=tuple(self._pending_world_signals),
                continuity_threads=tuple(self._pending_continuity_threads),
            )
            self._pending_interaction_signals = deque()
            self._pending_place_signals = deque()
            self._pending_world_signals = deque()
            self._pending_continuity_threads = deque()
            return batch

    def _restore_pending_batch(self, batch: _PendingBatch) -> None:
        """Restore one failed process batch ahead of newly enqueued items."""

        if not batch.has_items():
            return

        with self._queue_lock:
            for interaction_signal in reversed(batch.interaction_signals):
                self._pending_interaction_signals.appendleft(interaction_signal)
            for place_signal in reversed(batch.place_signals):
                self._pending_place_signals.appendleft(place_signal)
            for world_signal in reversed(batch.world_signals):
                self._pending_world_signals.appendleft(world_signal)
            for thread in reversed(batch.continuity_threads):
                self._pending_continuity_threads.appendleft(thread)

    def _compact_interaction_signals(
        self,
        signals: Sequence[InteractionSignal],
        deltas: Sequence[PersonalityDelta],
    ) -> tuple[InteractionSignal, ...]:
        """Drop already-consumed interaction signals and bound retained history."""

        consumed_ids = _accepted_signal_ids(deltas)
        pending = tuple(signal for signal in signals if signal.signal_id not in consumed_ids)
        return _tail(pending, limit=self.evolution_loop.policy.max_interaction_signals)

    def _compact_place_signals(
        self,
        signals: Sequence[PlaceSignal],
    ) -> tuple[PlaceSignal, ...]:
        """Bound retained place-signal history."""

        return _tail(tuple(signals), limit=self.evolution_loop.policy.max_place_signals)

    def _compact_world_signals(
        self,
        signals: Sequence[WorldSignal],
        *,
        now: datetime,
    ) -> tuple[WorldSignal, ...]:
        """Prune stale world signals and bound retained history."""

        return self.evolution_loop._prune_world_signals(
            _fresh_world_signals(tuple(signals), now=now)
        )

    def _compact_delta_history(
        self,
        deltas: Sequence[PersonalityDelta],
    ) -> tuple[PersonalityDelta, ...]:
        """Bound retained background delta history."""

        return _tail(tuple(deltas), limit=self.evolution_loop.policy.max_persisted_deltas)

    def _snapshot_prompt_deltas(
        self,
        delta_history: Sequence[PersonalityDelta],
    ) -> tuple[PersonalityDelta, ...]:
        """Return the accepted delta subset that belongs in the foreground snapshot."""

        return self.evolution_loop._prompt_personality_deltas(delta_history)

    def _build_committed_snapshot(
        self,
        *,
        base_snapshot: PersonalitySnapshot,
        delta_history: Sequence[PersonalityDelta],
        now: datetime,
    ) -> PersonalitySnapshot:
        """Build the foreground snapshot from maintained state and background history."""

        # BREAKING: PersonalitySnapshot.personality_deltas is now foreground-safe
        # accepted history only. Rejected deltas stay in the background store.
        prompt_deltas = self._snapshot_prompt_deltas(delta_history)
        return PersonalitySnapshot(
            schema_version=base_snapshot.schema_version,
            generated_at=base_snapshot.generated_at or _isoformat(now),
            core_traits=base_snapshot.core_traits,
            style_profile=base_snapshot.style_profile,
            humor_profile=base_snapshot.humor_profile,
            relationship_signals=base_snapshot.relationship_signals,
            continuity_threads=base_snapshot.continuity_threads,
            place_focuses=base_snapshot.place_focuses,
            world_signals=base_snapshot.world_signals,
            reflection_deltas=tuple(delta.to_reflection_delta() for delta in prompt_deltas),
            personality_deltas=prompt_deltas,
        )

    def process_pending(self) -> PersonalityEvolutionResult:
        """Persist pending signals, evolve the snapshot, and clear the queue."""

        with self._process_lock:
            batch = self._drain_pending_batch()
            try:
                persisted_snapshot = self.snapshot_store.load_snapshot(
                    config=self.config,
                    remote_state=self.remote_state,
                ) or PersonalitySnapshot()
                persisted_interaction_signals = self.evolution_store.load_interaction_signals(
                    config=self.config,
                    remote_state=self.remote_state,
                )
                persisted_place_signals = self.evolution_store.load_place_signals(
                    config=self.config,
                    remote_state=self.remote_state,
                )
                persisted_world_signals = self.evolution_store.load_world_signals(
                    config=self.config,
                    remote_state=self.remote_state,
                )
                persisted_deltas = self.evolution_store.load_personality_deltas(
                    config=self.config,
                    remote_state=self.remote_state,
                )

                now = self.evolution_loop.now_provider().astimezone(timezone.utc)
                compacted_delta_history = self._compact_delta_history(persisted_deltas)
                compacted_interaction_signals = self._compact_interaction_signals(
                    persisted_interaction_signals,
                    compacted_delta_history,
                )
                compacted_place_signals = self._compact_place_signals(persisted_place_signals)
                compacted_world_signals = self._compact_world_signals(
                    persisted_world_signals,
                    now=now,
                )

                if not batch.has_items():
                    maintained_snapshot = self.evolution_loop.maintain_snapshot(snapshot=persisted_snapshot)
                    committed_snapshot = self._build_committed_snapshot(
                        base_snapshot=maintained_snapshot,
                        delta_history=compacted_delta_history,
                        now=now,
                    )

                    if (
                        maintained_snapshot == persisted_snapshot
                        and compacted_interaction_signals == persisted_interaction_signals
                        and compacted_place_signals == persisted_place_signals
                        and compacted_world_signals == persisted_world_signals
                        and compacted_delta_history == persisted_deltas
                        and committed_snapshot == persisted_snapshot
                    ):
                        return PersonalityEvolutionResult(snapshot=persisted_snapshot)

                    self.evolution_store.save_interaction_signals(
                        config=self.config,
                        signals=compacted_interaction_signals,
                        remote_state=self.remote_state,
                    )
                    self.evolution_store.save_place_signals(
                        config=self.config,
                        signals=compacted_place_signals,
                        remote_state=self.remote_state,
                    )
                    self.evolution_store.save_world_signals(
                        config=self.config,
                        signals=compacted_world_signals,
                        remote_state=self.remote_state,
                    )
                    self.evolution_store.save_personality_deltas(
                        config=self.config,
                        deltas=compacted_delta_history,
                        remote_state=self.remote_state,
                    )
                    self.snapshot_store.save_snapshot(
                        config=self.config,
                        snapshot=committed_snapshot,
                        remote_state=self.remote_state,
                    )
                    return PersonalityEvolutionResult(snapshot=committed_snapshot)

                all_interaction_signals = _merge_by_key(
                    compacted_interaction_signals,
                    batch.interaction_signals,
                    key_fn=lambda signal: signal.signal_id,
                )
                all_place_signals = _merge_by_key(
                    compacted_place_signals,
                    batch.place_signals,
                    key_fn=lambda signal: signal.signal_id,
                )
                all_world_signals = self._compact_world_signals(
                    _merge_by_key(
                        compacted_world_signals,
                        batch.world_signals,
                        key_fn=_world_key,
                    ),
                    now=now,
                )

                candidate_interaction_signals = tuple(
                    signal
                    for signal in all_interaction_signals
                    if signal.signal_id not in _accepted_signal_ids(compacted_delta_history)
                )

                result = self.evolution_loop.evolve(
                    snapshot=persisted_snapshot,
                    interaction_signals=candidate_interaction_signals,
                    place_signals=all_place_signals,
                    world_signals=all_world_signals,
                    continuity_threads=batch.continuity_threads,
                )

                merged_delta_history = self._compact_delta_history(
                    _merge_by_key(
                        compacted_delta_history,
                        result.accepted_deltas + result.rejected_deltas,
                        key_fn=_delta_key,
                    )
                )
                committed_snapshot = self._build_committed_snapshot(
                    base_snapshot=result.snapshot,
                    delta_history=merged_delta_history,
                    now=now,
                )

                saved_interaction_signals = self._compact_interaction_signals(
                    all_interaction_signals,
                    merged_delta_history,
                )
                saved_place_signals = self._compact_place_signals(all_place_signals)

                self.evolution_store.save_interaction_signals(
                    config=self.config,
                    signals=saved_interaction_signals,
                    remote_state=self.remote_state,
                )
                self.evolution_store.save_place_signals(
                    config=self.config,
                    signals=saved_place_signals,
                    remote_state=self.remote_state,
                )
                self.evolution_store.save_world_signals(
                    config=self.config,
                    signals=all_world_signals,
                    remote_state=self.remote_state,
                )
                self.evolution_store.save_personality_deltas(
                    config=self.config,
                    deltas=merged_delta_history,
                    remote_state=self.remote_state,
                )
                self.snapshot_store.save_snapshot(
                    config=self.config,
                    snapshot=committed_snapshot,
                    remote_state=self.remote_state,
                )

                return PersonalityEvolutionResult(
                    snapshot=committed_snapshot,
                    accepted_deltas=result.accepted_deltas,
                    rejected_deltas=result.rejected_deltas,
                )
            except Exception:
                self._restore_pending_batch(batch)
                raise
