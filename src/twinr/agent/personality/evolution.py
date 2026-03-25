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

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    try:
        parsed = datetime.fromisoformat(str(value))
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


def _place_key(signal: PlaceSignal | PlaceFocus) -> str:
    """Return the stable merge key for place-aware context."""

    name = getattr(signal, "place_name", None) or getattr(signal, "name", "")
    geography = getattr(signal, "geography", None) or ""
    return f"{str(name).strip().lower()}::{str(geography).strip().lower()}"


def _world_key(signal: WorldSignal) -> str:
    """Return the stable merge key for world-aware context."""

    return "::".join(
        (
            signal.topic.strip().lower(),
            (signal.region or "").strip().lower(),
            signal.source.strip().lower(),
        )
    )


def _delta_key(delta: PersonalityDelta) -> str:
    """Return the stable merge key for persistent personality deltas."""

    return delta.delta_id


def _relationship_topic_key(signal: RelationshipSignal) -> str:
    """Return the stable merge key for prompt-facing relationship signals."""

    return signal.topic.strip().lower()


def _continuity_key(thread: ContinuityThread) -> str:
    """Return the stable merge key for continuity threads."""

    return thread.title.strip().lower()


def _topic_target_parts(target: str) -> tuple[str | None, str | None]:
    """Return the relationship target family and topic for one delta target."""

    if target.startswith(RELATIONSHIP_TOPIC_DELTA_PREFIX):
        return "affinity", target[len(RELATIONSHIP_TOPIC_DELTA_PREFIX) :].strip().lower()
    if target.startswith(RELATIONSHIP_TOPIC_AVERSION_DELTA_PREFIX):
        return "aversion", target[len(RELATIONSHIP_TOPIC_AVERSION_DELTA_PREFIX) :].strip().lower()
    return None, None


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


@dataclass(frozen=True, slots=True)
class PersonalityEvolutionPolicy:
    """Define deterministic gates for personality learning.

    Attributes:
        min_support_count: Minimum repeated support required before an implicit
            interaction signal may mutate promptable personality state.
        min_confidence: Minimum mean confidence for a group of signals to be
            accepted implicitly.
        max_humor_step: Largest single evolution step allowed for humor
            intensity in one processing run.
        max_humor_intensity: Upper cap for the learned humor intensity.
        relationship_decay_days: Age after which dormant relationship signals
            begin to decay when no fresh support refreshes them.
        relationship_decay_step: Salience step removed for every elapsed decay
            window.
        min_relationship_salience: Floor below which decayed relationship
            signals fall out of the prompt-facing snapshot.
        supported_delta_targets: Structured delta targets that this version of
            the policy is allowed to apply. Entries ending in ``:`` act as
            target prefixes for families such as topic-affinity deltas.
    """

    min_support_count: int = 2
    min_confidence: float = 0.6
    max_humor_step: float = 0.08
    max_humor_intensity: float = 0.6
    relationship_decay_days: int = 30
    relationship_decay_step: float = 0.08
    min_relationship_salience: float = 0.08
    supported_delta_targets: tuple[str, ...] = (
        STYLE_VERBOSITY_DELTA_TARGET,
        STYLE_INITIATIVE_DELTA_TARGET,
        "humor.intensity",
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
            "max_humor_step",
            _clamp(abs(float(self.max_humor_step)), minimum=0.0, maximum=1.0),
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

        relationship_signals = self._decay_relationship_signals(
            relationship_signals=snapshot.relationship_signals,
            now=now,
        )
        continuity_threads = _active_continuity_threads(
            snapshot.continuity_threads,
            now=now,
        )
        world_signals = _fresh_world_signals(snapshot.world_signals, now=now)
        return PersonalitySnapshot(
            schema_version=snapshot.schema_version,
            generated_at=snapshot.generated_at,
            core_traits=snapshot.core_traits,
            style_profile=snapshot.style_profile,
            humor_profile=snapshot.humor_profile,
            relationship_signals=relationship_signals,
            continuity_threads=continuity_threads,
            place_focuses=snapshot.place_focuses,
            world_signals=world_signals,
            reflection_deltas=snapshot.reflection_deltas,
            personality_deltas=snapshot.personality_deltas,
        )

    def _decay_relationship_signals(
        self,
        *,
        relationship_signals: Sequence[RelationshipSignal],
        now: datetime,
    ) -> tuple[RelationshipSignal, ...]:
        """Decay dormant relationship signals based on their last update time."""

        decayed: list[RelationshipSignal] = []
        decay_window_seconds = self.policy.relationship_decay_days * 24 * 60 * 60
        for signal in relationship_signals:
            updated_at = _parse_iso_datetime(signal.updated_at)
            if updated_at is None:
                decayed.append(signal)
                continue
            age_seconds = (now - updated_at).total_seconds()
            if age_seconds < decay_window_seconds:
                decayed.append(signal)
                continue
            elapsed_windows = int(age_seconds // decay_window_seconds)
            reduced_salience = _clamp(
                signal.salience - (elapsed_windows * self.policy.relationship_decay_step),
                minimum=0.0,
                maximum=1.0,
            )
            if reduced_salience < self.policy.min_relationship_salience:
                continue
            decayed.append(
                RelationshipSignal(
                    topic=signal.topic,
                    summary=signal.summary,
                    salience=reduced_salience,
                    source=signal.source,
                    stance=signal.stance,
                    updated_at=signal.updated_at,
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

        grouped_signals: dict[str, tuple[InteractionSignal, ...]] = {}
        for signal in interaction_signals:
            if signal.delta_target is None or signal.delta_value is None:
                continue
            grouped = grouped_signals.setdefault(signal.delta_target, ())
            grouped_signals[signal.delta_target] = grouped + (signal,)

        prioritized_groups = {
            target: self._prioritize_explicit_signals(grouped)
            for target, grouped in grouped_signals.items()
        }
        contradictions = self._contradictions(prioritized_groups)
        accepted: list[PersonalityDelta] = []
        rejected: list[PersonalityDelta] = []

        for target, grouped in prioritized_groups.items():
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

            support_count = sum(signal.evidence_count for signal in grouped)
            mean_confidence = _mean([signal.confidence for signal in grouped], default=0.0)
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

            proposed_delta_value = _mean(
                [signal.delta_value for signal in grouped if signal.delta_value is not None],
                default=0.0,
            )
            if target == "humor.intensity":
                bounded_delta_value = _clamp(
                    proposed_delta_value,
                    minimum=-self.policy.max_humor_step,
                    maximum=self.policy.max_humor_step,
                )
            else:
                bounded_delta_value = proposed_delta_value
            accepted.append(
                self._build_delta(
                    target=target,
                    signals=grouped,
                    status="accepted",
                    rationale="Repeated supported interaction feedback.",
                    delta_value=bounded_delta_value,
                    now=now,
                )
            )
        return tuple(accepted), tuple(rejected)

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

        support_count = sum(signal.evidence_count for signal in signals)
        summary = next(
            (
                signal.delta_summary
                for signal in signals
                if signal.delta_summary is not None and signal.delta_summary.strip()
            ),
            signals[0].summary,
        )
        signal_ids = tuple(signal.signal_id for signal in signals)
        return PersonalityDelta(
            delta_id=f"{target}:{'|'.join(signal_ids)}",
            target=target,
            summary=summary,
            rationale=rationale,
            delta_value=delta_value,
            confidence=_mean([signal.confidence for signal in signals], default=0.0),
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

        style_profile = snapshot.style_profile
        humor_profile = snapshot.humor_profile
        relationship_signals = snapshot.relationship_signals
        for delta in accepted_deltas:
            if delta.target == STYLE_VERBOSITY_DELTA_TARGET:
                style_baseline = style_profile or default_style_profile()
                style_profile = ConversationStyleProfile(
                    verbosity=_clamp(style_baseline.verbosity + delta.delta_value, minimum=0.0, maximum=1.0),
                    initiative=style_baseline.initiative,
                    evidence=style_baseline.evidence + delta.source_signal_ids,
                )
                continue
            if delta.target == STYLE_INITIATIVE_DELTA_TARGET:
                style_baseline = style_profile or default_style_profile()
                style_profile = ConversationStyleProfile(
                    verbosity=style_baseline.verbosity,
                    initiative=_clamp(style_baseline.initiative + delta.delta_value, minimum=0.0, maximum=1.0),
                    evidence=style_baseline.evidence + delta.source_signal_ids,
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
                    evidence=humor_baseline.evidence + delta.source_signal_ids,
                )
                continue

            family, topic = _topic_target_parts(delta.target)
            if family is None or topic is None or not topic:
                continue
            current_signal = next(
                (item for item in relationship_signals if _relationship_topic_key(item) == topic),
                None,
            )
            updated_signal = RelationshipSignal(
                topic=delta.target.split(":", 1)[1].strip(),
                summary=delta.summary,
                salience=_clamp(
                    (
                        current_signal.salience
                        if current_signal is not None and current_signal.stance == family
                        else 0.0
                    )
                    + max(0.0, delta.delta_value),
                    minimum=0.0,
                    maximum=1.0,
                ),
                source="personality_learning",
                stance=family,
                updated_at=_isoformat(now),
            )
            relationship_signals = _merge_by_key(
                relationship_signals,
                (updated_signal,),
                key_fn=_relationship_topic_key,
            )

        place_focuses = _merge_by_key(
            snapshot.place_focuses,
            tuple(signal.to_place_focus() for signal in place_signals),
            key_fn=_place_key,
        )
        merged_world_signals = _fresh_world_signals(
            _merge_by_key(snapshot.world_signals, tuple(world_signals), key_fn=_world_key),
            now=now,
        )
        merged_continuity_threads = _active_continuity_threads(
            _merge_by_key(snapshot.continuity_threads, tuple(continuity_threads), key_fn=_continuity_key),
            now=now,
        )
        merged_personality_deltas = _merge_by_key(
            snapshot.personality_deltas,
            tuple(accepted_deltas),
            key_fn=_delta_key,
        )

        return PersonalitySnapshot(
            schema_version=snapshot.schema_version,
            generated_at=_isoformat(now),
            core_traits=snapshot.core_traits,
            style_profile=style_profile,
            humor_profile=humor_profile,
            relationship_signals=relationship_signals,
            continuity_threads=merged_continuity_threads,
            place_focuses=place_focuses,
            world_signals=merged_world_signals,
            reflection_deltas=tuple(
                delta.to_reflection_delta()
                for delta in merged_personality_deltas
                if delta.status == "accepted"
            ),
            personality_deltas=merged_personality_deltas,
        )


def _accepted_signal_ids(deltas: Iterable[PersonalityDelta]) -> set[str]:
    """Return the interaction signal ids that already drove an accepted delta."""

    consumed: set[str] = set()
    for delta in deltas:
        if delta.status != "accepted":
            continue
        consumed.update(delta.source_signal_ids)
    return consumed


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
    _pending_interaction_signals: list[InteractionSignal] = field(default_factory=list, init=False, repr=False)
    _pending_place_signals: list[PlaceSignal] = field(default_factory=list, init=False, repr=False)
    _pending_world_signals: list[WorldSignal] = field(default_factory=list, init=False, repr=False)
    _pending_continuity_threads: list[ContinuityThread] = field(default_factory=list, init=False, repr=False)

    def enqueue_interaction_signal(self, signal: InteractionSignal) -> None:
        """Queue one interaction signal for the next background processing pass."""

        self._pending_interaction_signals.append(signal)

    def enqueue_place_signal(self, signal: PlaceSignal) -> None:
        """Queue one place signal for the next background processing pass."""

        self._pending_place_signals.append(signal)

    def enqueue_world_signal(self, signal: WorldSignal) -> None:
        """Queue one world signal for the next background processing pass."""

        self._pending_world_signals.append(signal)

    def enqueue_continuity_thread(self, thread: ContinuityThread) -> None:
        """Queue one continuity thread refresh for the next processing pass."""

        self._pending_continuity_threads.append(thread)

    def has_pending_items(self) -> bool:
        """Return whether any learning signals are waiting for commit."""

        return bool(
            self._pending_interaction_signals
            or self._pending_place_signals
            or self._pending_world_signals
            or self._pending_continuity_threads
        )

    def process_pending(self) -> PersonalityEvolutionResult:
        """Persist pending signals, evolve the snapshot, and clear the queue."""

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
        fresh_world_signals = _fresh_world_signals(persisted_world_signals, now=now)

        if not (
            self._pending_interaction_signals
            or self._pending_place_signals
            or self._pending_world_signals
            or self._pending_continuity_threads
        ):
            maintained_snapshot = self.evolution_loop.maintain_snapshot(snapshot=persisted_snapshot)
            if maintained_snapshot == persisted_snapshot and fresh_world_signals == persisted_world_signals:
                return PersonalityEvolutionResult(snapshot=persisted_snapshot)
            committed_snapshot = PersonalitySnapshot(
                schema_version=maintained_snapshot.schema_version,
                generated_at=_isoformat(now),
                core_traits=maintained_snapshot.core_traits,
                style_profile=maintained_snapshot.style_profile,
                humor_profile=maintained_snapshot.humor_profile,
                relationship_signals=maintained_snapshot.relationship_signals,
                continuity_threads=maintained_snapshot.continuity_threads,
                place_focuses=maintained_snapshot.place_focuses,
                world_signals=maintained_snapshot.world_signals,
                reflection_deltas=tuple(
                    delta.to_reflection_delta()
                    for delta in persisted_deltas
                    if delta.status == "accepted"
                ),
                personality_deltas=persisted_deltas,
            )
            self.evolution_store.save_world_signals(
                config=self.config,
                signals=fresh_world_signals,
                remote_state=self.remote_state,
            )
            self.snapshot_store.save_snapshot(
                config=self.config,
                snapshot=committed_snapshot,
                remote_state=self.remote_state,
            )
            return PersonalityEvolutionResult(snapshot=committed_snapshot)

        all_interaction_signals = _merge_by_key(
            persisted_interaction_signals,
            tuple(self._pending_interaction_signals),
            key_fn=lambda signal: signal.signal_id,
        )
        all_place_signals = _merge_by_key(
            persisted_place_signals,
            tuple(self._pending_place_signals),
            key_fn=lambda signal: signal.signal_id,
        )
        all_world_signals = _fresh_world_signals(
            _merge_by_key(
                fresh_world_signals,
                tuple(self._pending_world_signals),
                key_fn=_world_key,
            ),
            now=now,
        )

        candidate_interaction_signals: tuple[InteractionSignal, ...]
        if self._pending_interaction_signals:
            consumed_signal_ids = _accepted_signal_ids(persisted_deltas)
            candidate_interaction_signals = tuple(
                signal
                for signal in all_interaction_signals
                if signal.signal_id not in consumed_signal_ids
            )
        else:
            candidate_interaction_signals = ()

        result = self.evolution_loop.evolve(
            snapshot=persisted_snapshot,
            interaction_signals=candidate_interaction_signals,
            place_signals=all_place_signals,
            world_signals=all_world_signals,
            continuity_threads=tuple(self._pending_continuity_threads),
        )

        merged_deltas = _merge_by_key(
            persisted_deltas,
            result.accepted_deltas + result.rejected_deltas,
            key_fn=_delta_key,
        )
        committed_snapshot = PersonalitySnapshot(
            schema_version=result.snapshot.schema_version,
            generated_at=result.snapshot.generated_at or _isoformat(now),
            core_traits=result.snapshot.core_traits,
            style_profile=result.snapshot.style_profile,
            humor_profile=result.snapshot.humor_profile,
            relationship_signals=result.snapshot.relationship_signals,
            continuity_threads=result.snapshot.continuity_threads,
            place_focuses=result.snapshot.place_focuses,
            world_signals=result.snapshot.world_signals,
            reflection_deltas=tuple(
                delta.to_reflection_delta()
                for delta in merged_deltas
                if delta.status == "accepted"
            ),
            personality_deltas=merged_deltas,
        )

        self.evolution_store.save_interaction_signals(
            config=self.config,
            signals=all_interaction_signals,
            remote_state=self.remote_state,
        )
        self.evolution_store.save_place_signals(
            config=self.config,
            signals=all_place_signals,
            remote_state=self.remote_state,
        )
        self.evolution_store.save_world_signals(
            config=self.config,
            signals=all_world_signals,
            remote_state=self.remote_state,
        )
        self.evolution_store.save_personality_deltas(
            config=self.config,
            deltas=merged_deltas,
            remote_state=self.remote_state,
        )
        self.snapshot_store.save_snapshot(
            config=self.config,
            snapshot=committed_snapshot,
            remote_state=self.remote_state,
        )

        self._pending_interaction_signals.clear()
        self._pending_place_signals.clear()
        self._pending_world_signals.clear()
        self._pending_continuity_threads.clear()

        return PersonalityEvolutionResult(
            snapshot=committed_snapshot,
            accepted_deltas=result.accepted_deltas,
            rejected_deltas=result.rejected_deltas,
        )
