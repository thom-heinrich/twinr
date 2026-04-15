# CHANGELOG: 2026-03-27
# BUG-1: Zero/negative max_items no longer leaks through to downstream builders that forced at least one item.
# BUG-2: Schema drift, partial corruption, and unexpected remote-state shapes now
#        surface as hard failures instead of silently substituting stale/default
#        prompt state.
# BUG-3: Repeated same-turn loads now reuse short-lived cached state instead of re-hitting remote storage on every helper call.
# SEC-1: Remote-derived sections and engagement signals are now bounded and sanitized to reduce prompt-stuffing and Pi-side resource exhaustion.
# SEC-2: Added stale-if-error caching, failure cooldowns, and request coalescing to avoid outage amplification against remote-primary state.
# IMP-1: Added one runtime-state boundary with validation, cache trimming, and thread-safe hot-key collapse.
# IMP-2: Added prompt-budget enforcement for legacy/structured sections and conservative snapshot list capping for Raspberry-Pi-friendly operation.

"""Bridge prompt callers onto the structured personality package."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, is_dataclass, replace
from datetime import datetime
from itertools import islice
from threading import Event, RLock
from typing import Generic, TypeVar

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality.context_builder import PersonalityContextBuilder
from twinr.agent.personality.display_impulses import (
    AmbientDisplayImpulseCandidate,
    build_ambient_display_impulse_candidates,
)
from twinr.agent.personality.intelligence.models import WorldInterestSignal
from twinr.agent.personality.intelligence.store import RemoteStateWorldIntelligenceStore
from twinr.agent.personality.models import PersonalitySnapshot
from twinr.agent.personality.positive_engagement import (
    PositiveEngagementTopicPolicy,
    build_positive_engagement_policies,
)
from twinr.agent.personality.steering import ConversationTurnSteeringCue, build_turn_steering_cues
from twinr.agent.personality.store import (
    PersonalitySnapshotStore,
    RemoteStatePersonalitySnapshotStore,
)
from twinr.memory.longterm.storage.remote_state import (
    LongTermRemoteStateStore,
    LongTermRemoteUnavailableError,
)

_LOGGER = logging.getLogger(__name__)

_T = TypeVar("_T")
_MISSING = object()
_CacheKey = tuple[int, str]

_SNAPSHOT_LIST_FIELD_CAPS: dict[str, int] = {
    "core_traits": 8,
    "relationship_signals": 12,
    "continuity_threads": 12,
    "place_focuses": 8,
    "world_signals": 12,
    "reflection_deltas": 8,
}


@dataclass(frozen=True, slots=True)
class _TimedValue(Generic[_T]):
    value: _T
    loaded_at_monotonic: float


@dataclass(frozen=True, slots=True)
class _RuntimeState:
    snapshot: PersonalitySnapshot | None
    engagement_signals: tuple[WorldInterestSignal, ...]


@dataclass(slots=True)
class PersonalityContextService:
    """Load structured personality state and convert it into legacy sections."""

    builder: PersonalityContextBuilder = field(default_factory=PersonalityContextBuilder)
    store: PersonalitySnapshotStore = field(default_factory=RemoteStatePersonalitySnapshotStore)
    intelligence_store: RemoteStateWorldIntelligenceStore = field(default_factory=RemoteStateWorldIntelligenceStore)

    runtime_cache_ttl_seconds: float = 0.75
    stale_if_error_ttl_seconds: float = 30.0
    failure_cooldown_seconds: float = 2.0
    request_collapse_wait_seconds: float = 1.0
    max_cache_entries: int = 64
    max_interest_signals: int = 64
    max_public_items: int = 16
    max_prompt_sections: int = 32
    max_section_chars: int = 12_000
    max_total_section_chars: int = 48_000

    _lock: RLock = field(default_factory=RLock, init=False, repr=False)
    _snapshot_cache: dict[_CacheKey, _TimedValue[PersonalitySnapshot | None]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _engagement_cache: dict[_CacheKey, _TimedValue[tuple[WorldInterestSignal, ...]]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _snapshot_failures_until: dict[_CacheKey, float] = field(default_factory=dict, init=False, repr=False)
    _engagement_failures_until: dict[_CacheKey, float] = field(default_factory=dict, init=False, repr=False)
    _snapshot_inflight: dict[_CacheKey, Event] = field(default_factory=dict, init=False, repr=False)
    _engagement_inflight: dict[_CacheKey, Event] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.runtime_cache_ttl_seconds = self._bounded_float(
            self.runtime_cache_ttl_seconds,
            default=0.75,
            minimum=0.0,
            maximum=300.0,
        )
        self.stale_if_error_ttl_seconds = self._bounded_float(
            self.stale_if_error_ttl_seconds,
            default=30.0,
            minimum=0.0,
            maximum=3_600.0,
        )
        if self.stale_if_error_ttl_seconds < self.runtime_cache_ttl_seconds:
            self.stale_if_error_ttl_seconds = self.runtime_cache_ttl_seconds
        self.failure_cooldown_seconds = self._bounded_float(
            self.failure_cooldown_seconds,
            default=2.0,
            minimum=0.0,
            maximum=300.0,
        )
        self.request_collapse_wait_seconds = self._bounded_float(
            self.request_collapse_wait_seconds,
            default=1.0,
            minimum=0.0,
            maximum=30.0,
        )
        self.max_cache_entries = self._bounded_int(self.max_cache_entries, default=64, minimum=4, maximum=512)
        self.max_interest_signals = self._bounded_int(
            self.max_interest_signals,
            default=64,
            minimum=1,
            maximum=512,
        )
        self.max_public_items = self._bounded_int(self.max_public_items, default=16, minimum=1, maximum=64)
        self.max_prompt_sections = self._bounded_int(
            self.max_prompt_sections,
            default=32,
            minimum=4,
            maximum=128,
        )
        self.max_section_chars = self._bounded_int(
            self.max_section_chars,
            default=12_000,
            minimum=256,
            maximum=65_536,
        )
        self.max_total_section_chars = self._bounded_int(
            self.max_total_section_chars,
            default=48_000,
            minimum=self.max_section_chars,
            maximum=262_144,
        )

    def load_snapshot(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> PersonalitySnapshot | None:
        """Load the optional structured personality snapshot for this turn."""

        cache_key = self._cache_key(config=config, remote_state=remote_state)
        return self._load_cached_value(
            cache_key=cache_key,
            cache=self._snapshot_cache,
            failure_deadlines=self._snapshot_failures_until,
            inflight=self._snapshot_inflight,
            default=None,
            loader=lambda: self._sanitize_snapshot(
                self._validate_snapshot(
                    self.store.load_snapshot(config=config, remote_state=remote_state),
                ),
            ),
            unavailable_log_message=(
                "Unable to load structured personality snapshot from remote state: %s"
            ),
            malformed_log_message=(
                "Ignoring malformed structured personality snapshot."
            ),
            stale_debug_label="structured personality snapshot",
        )

    def build_static_sections(
        self,
        *,
        legacy_sections: tuple[tuple[str, str], ...],
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[tuple[str, str], ...]:
        """Merge legacy sections with any available structured personality state."""

        runtime_state = self._load_runtime_state(config=config, remote_state=remote_state)
        plan = self.builder.build_prompt_plan(
            legacy_sections=legacy_sections,
            snapshot=runtime_state.snapshot,
            engagement_signals=runtime_state.engagement_signals,
        )
        return self._sanitize_sections(plan.as_sections())

    def build_supervisor_sections(
        self,
        *,
        legacy_sections: tuple[tuple[str, str], ...],
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[tuple[str, str], ...]:
        """Build a lean supervisor bundle without dynamic topic-contamination.

        The fast supervisor should keep stable character/style context, but it
        must not inherit volatile prompt layers such as `MINDSHARE`, `PLACE`,
        `WORLD`, or `REFLECTION`, because those layers can distort routing for
        noisy freshness-sensitive turns.
        """

        snapshot = self.load_snapshot(config=config, remote_state=remote_state)
        plan = self.builder.build_supervisor_prompt_plan(
            legacy_sections=legacy_sections,
            snapshot=snapshot,
        )
        return self._sanitize_sections(plan.as_sections())

    def load_turn_steering_cues(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
        max_items: int = 3,
    ) -> tuple[ConversationTurnSteeringCue, ...]:
        """Load the bounded steering cues that may influence one turn.

        Args:
            config: Runtime configuration that points to remote personality and
                world-intelligence state.
            remote_state: Optional shared remote-state instance to reuse.
            max_items: Maximum number of cues to surface for the current turn.

        Returns:
            The current bounded steering cues derived from structured
            personality and world-intelligence state.
        """

        limited_max_items = self._normalize_requested_items(max_items=max_items, default=3)
        if limited_max_items == 0:
            return ()

        runtime_state = self._load_runtime_state(config=config, remote_state=remote_state)
        return build_turn_steering_cues(
            runtime_state.snapshot,
            engagement_signals=runtime_state.engagement_signals,
            max_items=limited_max_items,
        )

    def load_engagement_signals(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[WorldInterestSignal, ...]:
        """Load durable world-interest signals for display and turn shaping."""

        return self._load_engagement_signals(
            config=config,
            remote_state=remote_state,
        )

    def load_positive_engagement_policies(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
        max_items: int = 3,
    ) -> tuple[PositiveEngagementTopicPolicy, ...]:
        """Load the current bounded positive-engagement topic actions."""

        limited_max_items = self._normalize_requested_items(max_items=max_items, default=3)
        if limited_max_items == 0:
            return ()

        runtime_state = self._load_runtime_state(config=config, remote_state=remote_state)
        return build_positive_engagement_policies(
            runtime_state.snapshot,
            engagement_signals=runtime_state.engagement_signals,
            max_items=limited_max_items,
        )

    def load_display_impulse_candidates(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
        local_now: datetime | None = None,
        max_items: int = 4,
    ) -> tuple[AmbientDisplayImpulseCandidate, ...]:
        """Load the current bounded silent display-impulse candidates."""

        limited_max_items = self._normalize_requested_items(max_items=max_items, default=4)
        if limited_max_items == 0:
            return ()

        runtime_state = self._load_runtime_state(config=config, remote_state=remote_state)
        return build_ambient_display_impulse_candidates(
            runtime_state.snapshot,
            engagement_signals=runtime_state.engagement_signals,
            local_now=local_now,
            max_items=limited_max_items,
        )

    def _load_runtime_state(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> _RuntimeState:
        return _RuntimeState(
            snapshot=self.load_snapshot(config=config, remote_state=remote_state),
            engagement_signals=self._load_engagement_signals(
                config=config,
                remote_state=remote_state,
            ),
        )

    def _load_engagement_signals(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[WorldInterestSignal, ...]:
        """Load durable interest/engagement signals for mindshare surfacing."""

        cache_key = self._cache_key(config=config, remote_state=remote_state)
        return self._load_cached_value(
            cache_key=cache_key,
            cache=self._engagement_cache,
            failure_deadlines=self._engagement_failures_until,
            inflight=self._engagement_inflight,
            default=(),
            loader=lambda: self._extract_interest_signals(
                self.intelligence_store.load_state(
                    config=config,
                    remote_state=remote_state,
                ),
            ),
            unavailable_log_message=(
                "Unable to load world-intelligence engagement state from remote state: %s"
            ),
            malformed_log_message=(
                "Ignoring malformed world-intelligence state snapshot."
            ),
            stale_debug_label="world-intelligence engagement state",
        )

    def _load_cached_value(
        self,
        *,
        cache_key: _CacheKey,
        cache: dict[_CacheKey, _TimedValue[_T]],
        failure_deadlines: dict[_CacheKey, float],
        inflight: dict[_CacheKey, Event],
        default: _T,
        loader: Callable[[], _T],
        unavailable_log_message: str,
        malformed_log_message: str,
        stale_debug_label: str,
    ) -> _T:
        del default, stale_debug_label
        now = time.monotonic()

        fresh_value = self._read_cache_value(
            cache=cache,
            cache_key=cache_key,
            now=now,
            max_age_seconds=self.runtime_cache_ttl_seconds,
        )
        if fresh_value is not _MISSING:
            return fresh_value

        event, owns_inflight = self._claim_inflight_slot(
            inflight=inflight,
            cache_key=cache_key,
        )
        if not owns_inflight:
            event.wait(timeout=self.request_collapse_wait_seconds)
            refreshed_value = self._read_cache_value(
                cache=cache,
                cache_key=cache_key,
                now=time.monotonic(),
                max_age_seconds=self.runtime_cache_ttl_seconds,
            )
            if refreshed_value is not _MISSING:
                return refreshed_value

            event, owns_inflight = self._claim_inflight_slot(
                inflight=inflight,
                cache_key=cache_key,
            )
            if not owns_inflight:
                raise RuntimeError("Personality context load did not produce a fresh value.")

        try:
            loaded_value = loader()
        except LongTermRemoteUnavailableError as exc:
            _LOGGER.warning(unavailable_log_message, exc)
            self._record_failure(
                failure_deadlines=failure_deadlines,
                cache_key=cache_key,
                now=time.monotonic(),
            )
            raise
        except Exception:
            _LOGGER.exception(malformed_log_message)
            self._record_failure(
                failure_deadlines=failure_deadlines,
                cache_key=cache_key,
                now=time.monotonic(),
            )
            raise
        else:
            self._store_cache_value(
                cache=cache,
                cache_key=cache_key,
                value=loaded_value,
                now=time.monotonic(),
            )
            self._clear_failure(
                failure_deadlines=failure_deadlines,
                cache_key=cache_key,
            )
            return loaded_value
        finally:
            if owns_inflight:
                self._release_inflight_slot(
                    inflight=inflight,
                    cache_key=cache_key,
                    event=event,
                )

    def _validate_snapshot(self, snapshot: object) -> PersonalitySnapshot | None:
        if snapshot is None:
            return None
        if not isinstance(snapshot, PersonalitySnapshot):
            raise TypeError(
                f"Expected PersonalitySnapshot | None, got {type(snapshot).__name__}.",
            )
        return snapshot

    def _sanitize_snapshot(self, snapshot: PersonalitySnapshot | None) -> PersonalitySnapshot | None:
        if snapshot is None or not is_dataclass(snapshot):
            return snapshot

        updates: dict[str, object] = {}
        for field_name, limit in _SNAPSHOT_LIST_FIELD_CAPS.items():
            try:
                current_value = getattr(snapshot, field_name)
            except AttributeError:
                continue

            bounded_value = self._bounded_sequence_like(current_value, limit=limit)
            if bounded_value is current_value:
                continue
            updates[field_name] = bounded_value

        if not updates:
            return snapshot

        try:
            _LOGGER.warning(
                "Truncated oversized structured personality snapshot lists to bounded prompt-time limits.",
            )
            return replace(snapshot, **updates)
        except Exception:
            _LOGGER.exception(
                "Unable to create bounded snapshot copy; continuing with original snapshot.",
            )
            return snapshot

    def _extract_interest_signals(
        self,
        state: object,
    ) -> tuple[WorldInterestSignal, ...]:
        raw_signals = getattr(state, "interest_signals", ())
        if raw_signals is None:
            return ()
        if isinstance(raw_signals, WorldInterestSignal):
            return (raw_signals,)
        if isinstance(raw_signals, (str, bytes, bytearray)):
            raise TypeError("interest_signals must be an iterable of WorldInterestSignal values.")

        iterator = iter(raw_signals)
        loaded: list[WorldInterestSignal] = []
        dropped_invalid = 0

        for item in islice(iterator, self.max_interest_signals):
            if isinstance(item, WorldInterestSignal):
                loaded.append(item)
            else:
                dropped_invalid += 1

        overflow_detected = False
        if isinstance(raw_signals, Sequence):
            overflow_detected = len(raw_signals) > self.max_interest_signals
        else:
            overflow_detected = next(iterator, _MISSING) is not _MISSING

        if dropped_invalid:
            _LOGGER.warning(
                "Dropped %d malformed world-interest signals while building turn context.",
                dropped_invalid,
            )
        if overflow_detected:
            _LOGGER.warning(
                "Truncated world-interest signals to %d items for bounded prompt-time use.",
                self.max_interest_signals,
            )

        return tuple(loaded)

    def _sanitize_sections(
        self,
        sections: Iterable[tuple[str, str]],
    ) -> tuple[tuple[str, str], ...]:
        remaining_budget = self.max_total_section_chars
        normalized: list[tuple[str, str]] = []

        for title, content in sections:
            if len(normalized) >= self.max_prompt_sections or remaining_budget <= 0:
                break

            clean_title = self._clean_prompt_text(title)
            clean_content = self._clean_prompt_text(content)

            if not clean_title or not clean_content:
                continue

            if len(clean_content) > self.max_section_chars:
                clean_content = clean_content[: self.max_section_chars].rstrip()

            if len(clean_content) > remaining_budget:
                clean_content = clean_content[:remaining_budget].rstrip()

            if not clean_content:
                continue

            normalized.append((clean_title, clean_content))
            remaining_budget -= len(clean_content)

        return tuple(normalized)

    def _clean_prompt_text(self, value: object) -> str:
        text = str(value or "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = "".join(
            character
            for character in text
            if character.isprintable() or character in "\n\t"
        )
        return "\n".join(line.rstrip() for line in text.split("\n")).strip()

    def _normalize_requested_items(self, *, max_items: object, default: int) -> int:
        try:
            parsed = int(max_items)
        except (TypeError, ValueError):
            parsed = default
        if parsed <= 0:
            return 0
        return min(parsed, self.max_public_items)

    def _bounded_sequence_like(self, value: object, *, limit: int) -> object:
        if isinstance(value, tuple):
            return value if len(value) <= limit else value[:limit]
        if isinstance(value, list):
            return value if len(value) <= limit else value[:limit]
        return value

    def _cache_key(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None,
    ) -> _CacheKey:
        return (
            id(config),
            self._remote_state_cache_marker(remote_state),
        )

    @staticmethod
    def _remote_state_cache_marker(
        remote_state: LongTermRemoteStateStore | None,
    ) -> str:
        """Return a stable cache marker for equivalent remote-state adapters.

        Prompt assembly frequently recreates ``LongTermRemoteStateStore``
        wrappers and timeout-capped clones around the same namespace. Caching on
        the object id turns those semantically identical handles into cache
        misses and replays the same remote reads unnecessarily.
        """

        if remote_state is None:
            return "none"
        namespace = getattr(remote_state, "namespace", None)
        if isinstance(namespace, str) and namespace.strip():
            return f"namespace:{namespace.strip()}"
        explicit_marker = getattr(remote_state, "cache_key", None) or getattr(
            remote_state,
            "instance_id",
            None,
        )
        if isinstance(explicit_marker, str) and explicit_marker.strip():
            return f"identity:{explicit_marker.strip()}"
        return f"type:{remote_state.__class__.__module__}.{remote_state.__class__.__qualname__}"

    def _read_cache_value(
        self,
        *,
        cache: dict[_CacheKey, _TimedValue[_T]],
        cache_key: _CacheKey,
        now: float,
        max_age_seconds: float,
    ) -> _T | object:
        with self._lock:
            entry = cache.get(cache_key)
        if entry is None:
            return _MISSING
        if now - entry.loaded_at_monotonic > max_age_seconds:
            return _MISSING
        return entry.value

    def _store_cache_value(
        self,
        *,
        cache: dict[_CacheKey, _TimedValue[_T]],
        cache_key: _CacheKey,
        value: _T,
        now: float,
    ) -> None:
        with self._lock:
            cache[cache_key] = _TimedValue(value=value, loaded_at_monotonic=now)
            self._trim_cache_locked(cache)

    def _trim_cache_locked(
        self,
        cache: dict[_CacheKey, _TimedValue[object]],
    ) -> None:
        expiry_before = time.monotonic() - self.stale_if_error_ttl_seconds
        expired_keys = [
            cache_key
            for cache_key, entry in cache.items()
            if entry.loaded_at_monotonic < expiry_before
        ]
        for cache_key in expired_keys:
            cache.pop(cache_key, None)

        if len(cache) <= self.max_cache_entries:
            return

        overflow = len(cache) - self.max_cache_entries
        oldest_keys = sorted(
            cache,
            key=lambda cache_key: cache[cache_key].loaded_at_monotonic,
        )[:overflow]
        for cache_key in oldest_keys:
            cache.pop(cache_key, None)

    def _record_failure(
        self,
        *,
        failure_deadlines: dict[_CacheKey, float],
        cache_key: _CacheKey,
        now: float,
    ) -> None:
        with self._lock:
            failure_deadlines[cache_key] = now + self.failure_cooldown_seconds
            self._trim_failure_deadlines_locked(failure_deadlines)

    def _clear_failure(
        self,
        *,
        failure_deadlines: dict[_CacheKey, float],
        cache_key: _CacheKey,
    ) -> None:
        with self._lock:
            failure_deadlines.pop(cache_key, None)

    def _is_failure_open(
        self,
        *,
        failure_deadlines: dict[_CacheKey, float],
        cache_key: _CacheKey,
        now: float,
    ) -> bool:
        with self._lock:
            deadline = failure_deadlines.get(cache_key)
        return deadline is not None and deadline > now

    def _trim_failure_deadlines_locked(
        self,
        failure_deadlines: dict[_CacheKey, float],
    ) -> None:
        now = time.monotonic()
        expired_keys = [
            cache_key
            for cache_key, deadline in failure_deadlines.items()
            if deadline <= now
        ]
        for cache_key in expired_keys:
            failure_deadlines.pop(cache_key, None)

        if len(failure_deadlines) <= self.max_cache_entries:
            return

        overflow = len(failure_deadlines) - self.max_cache_entries
        oldest_keys = sorted(
            failure_deadlines,
            key=failure_deadlines.__getitem__,
        )[:overflow]
        for cache_key in oldest_keys:
            failure_deadlines.pop(cache_key, None)

    def _claim_inflight_slot(
        self,
        *,
        inflight: dict[_CacheKey, Event],
        cache_key: _CacheKey,
    ) -> tuple[Event, bool]:
        with self._lock:
            existing = inflight.get(cache_key)
            if existing is not None:
                return existing, False
            created = Event()
            inflight[cache_key] = created
            return created, True

    def _release_inflight_slot(
        self,
        *,
        inflight: dict[_CacheKey, Event],
        cache_key: _CacheKey,
        event: Event,
    ) -> None:
        with self._lock:
            current = inflight.get(cache_key)
            if current is event:
                inflight.pop(cache_key, None)
                event.set()

    @staticmethod
    def _bounded_float(
        value: object,
        *,
        default: float,
        minimum: float,
        maximum: float,
    ) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        if numeric != numeric:
            return default
        if numeric < minimum:
            return minimum
        if numeric > maximum:
            return maximum
        return numeric

    @staticmethod
    def _bounded_int(
        value: object,
        *,
        default: int,
        minimum: int,
        maximum: int,
    ) -> int:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return default
        if numeric < minimum:
            return minimum
        if numeric > maximum:
            return maximum
        return numeric
