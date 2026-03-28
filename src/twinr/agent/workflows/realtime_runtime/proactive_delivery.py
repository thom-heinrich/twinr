# CHANGELOG: 2026-03-27
# BUG-1: Blank/whitespace-only source_id values no longer collapse into one shared "unknown" bucket.
#        Missing IDs now use only the global cooldown, preventing unrelated proactive sources from
#        suppressing one another.
# BUG-2: Fixed race-prone shared mutable state. All runtime state transitions are now protected by
#        an RLock, and cooldown updates no longer shrink on stale or out-of-order event timestamps.
# BUG-3: Config booleans now parse conservatively, so strings like "false" or "off" no longer
#        become truthy by accident through Python's default bool() coercion.
# SEC-1: Hardened availability on Raspberry Pi by bounding source-id normalization work and storage.
#        Oversized source IDs are digested into stable short keys instead of being stored verbatim.
# IMP-1: Upgraded from fixed one-size-fits-all cooldowns to bounded adaptive backoff driven by
#        repeated ignored/interrupted prompts, matching 2026 just-in-time / adaptive proactivity.
# IMP-2: Quiet-hours parsing is now cached, ISO-8601 tolerant, and hot-path safe. Display cue hold
#        can scale with the remaining cooldown instead of always returning one fixed 45-second hint.
# IMP-3: Runtime config is sanitized lazily, so direct construction and live field mutation stay
#        bounded instead of depending on callers to always use from_config().

"""Choose speech versus visual-first delivery for proactive background work.

This module keeps delivery-mode policy out of the realtime background mixin.
It owns bounded state for visual-first cooldowns after ignored or interrupted
proactive speech and resolves quiet-hours plus audio/media suppression into one
delivery decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time as LocalTime
import hashlib
import math
from threading import RLock
from typing import Final, Literal

from twinr.agent.base_agent.config import TwinrConfig
from twinr.proactive.runtime.audio_policy import ReSpeakerAudioPolicySnapshot


_DEFAULT_VISUAL_FIRST_GLOBAL_COOLDOWN_S: Final[float] = 5.0 * 60.0
_DEFAULT_VISUAL_FIRST_SOURCE_REPEAT_COOLDOWN_S: Final[float] = 15.0 * 60.0
_DEFAULT_VISUAL_FIRST_CUE_HOLD_S: Final[float] = 45.0
_DEFAULT_VISUAL_FIRST_MAX_CUE_HOLD_S: Final[float] = 5.0 * 60.0
_DEFAULT_VISUAL_FIRST_ADAPTIVE_ENABLED: Final[bool] = True
_DEFAULT_VISUAL_FIRST_ADAPTIVE_BACKOFF_MULTIPLIER: Final[float] = 1.6
_DEFAULT_VISUAL_FIRST_ADAPTIVE_BACKOFF_MAX_FACTOR: Final[float] = 4.0
_DEFAULT_VISUAL_FIRST_ADAPTIVE_RESET_AFTER_S: Final[float] = 30.0 * 60.0
_DEFAULT_VISUAL_FIRST_INTERRUPTED_EXTRA_FACTOR: Final[float] = 1.25
_DEFAULT_QUIET_HOURS_START: Final[str] = "21:00"
_DEFAULT_QUIET_HOURS_END: Final[str] = "07:00"
_MAX_SOURCE_REPEAT_TRACKING: Final[int] = 64
_MAX_SOURCE_ID_INPUT_CHARS: Final[int] = 512
_MAX_NORMALIZED_SOURCE_ID_CHARS: Final[int] = 96
_SOURCE_ID_DIGEST_HEX_CHARS: Final[int] = 16

_REASON_QUIET_HOURS: Final[str] = "quiet_hours_visual_only"
_REASON_GLOBAL_COOLDOWN: Final[str] = "recent_audio_display_first_cooldown"
_REASON_SOURCE_COOLDOWN: Final[str] = "recent_source_audio_display_first_cooldown"

DeliveryChannel = Literal["speech", "display"]


@dataclass(frozen=True, slots=True)
class ProactiveDeliveryDecision:
    """Describe how one proactive prompt should be delivered."""

    channel: DeliveryChannel
    reason: str | None = None
    cue_hold_seconds: float | None = None


@dataclass(slots=True)
class ProactiveDeliveryPolicy:
    """Track bounded visual-first state for proactive background delivery."""

    visual_first_global_cooldown_s: float = _DEFAULT_VISUAL_FIRST_GLOBAL_COOLDOWN_S
    visual_first_source_repeat_cooldown_s: float = _DEFAULT_VISUAL_FIRST_SOURCE_REPEAT_COOLDOWN_S
    visual_first_cue_hold_s: float = _DEFAULT_VISUAL_FIRST_CUE_HOLD_S
    visual_first_max_cue_hold_s: float = _DEFAULT_VISUAL_FIRST_MAX_CUE_HOLD_S
    adaptive_visual_first_enabled: bool = _DEFAULT_VISUAL_FIRST_ADAPTIVE_ENABLED
    adaptive_visual_first_backoff_multiplier: float = (
        _DEFAULT_VISUAL_FIRST_ADAPTIVE_BACKOFF_MULTIPLIER
    )
    adaptive_visual_first_backoff_max_factor: float = (
        _DEFAULT_VISUAL_FIRST_ADAPTIVE_BACKOFF_MAX_FACTOR
    )
    adaptive_visual_first_backoff_reset_after_s: float = (
        _DEFAULT_VISUAL_FIRST_ADAPTIVE_RESET_AFTER_S
    )
    adaptive_visual_first_interrupted_extra_factor: float = (
        _DEFAULT_VISUAL_FIRST_INTERRUPTED_EXTRA_FACTOR
    )
    quiet_hours_visual_only_enabled: bool = True
    quiet_hours_start_local: str = _DEFAULT_QUIET_HOURS_START
    quiet_hours_end_local: str = _DEFAULT_QUIET_HOURS_END
    quiet_hours_visual_cue_hold_s: float | None = None
    _global_visual_only_until: float | None = field(default=None, repr=False)
    _source_visual_only_until: dict[str, float] = field(default_factory=dict, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    _runtime_config_signature: tuple[object, ...] | None = field(default=None, init=False, repr=False)
    _runtime_visual_first_global_cooldown_s: float = field(
        default=_DEFAULT_VISUAL_FIRST_GLOBAL_COOLDOWN_S,
        init=False,
        repr=False,
    )
    _runtime_visual_first_source_repeat_cooldown_s: float = field(
        default=_DEFAULT_VISUAL_FIRST_SOURCE_REPEAT_COOLDOWN_S,
        init=False,
        repr=False,
    )
    _runtime_visual_first_cue_hold_s: float = field(
        default=_DEFAULT_VISUAL_FIRST_CUE_HOLD_S,
        init=False,
        repr=False,
    )
    _runtime_visual_first_max_cue_hold_s: float = field(
        default=_DEFAULT_VISUAL_FIRST_MAX_CUE_HOLD_S,
        init=False,
        repr=False,
    )
    _runtime_adaptive_visual_first_enabled: bool = field(
        default=_DEFAULT_VISUAL_FIRST_ADAPTIVE_ENABLED,
        init=False,
        repr=False,
    )
    _runtime_adaptive_visual_first_backoff_multiplier: float = field(
        default=_DEFAULT_VISUAL_FIRST_ADAPTIVE_BACKOFF_MULTIPLIER,
        init=False,
        repr=False,
    )
    _runtime_adaptive_visual_first_backoff_max_factor: float = field(
        default=_DEFAULT_VISUAL_FIRST_ADAPTIVE_BACKOFF_MAX_FACTOR,
        init=False,
        repr=False,
    )
    _runtime_adaptive_visual_first_backoff_reset_after_s: float = field(
        default=_DEFAULT_VISUAL_FIRST_ADAPTIVE_RESET_AFTER_S,
        init=False,
        repr=False,
    )
    _runtime_adaptive_visual_first_interrupted_extra_factor: float = field(
        default=_DEFAULT_VISUAL_FIRST_INTERRUPTED_EXTRA_FACTOR,
        init=False,
        repr=False,
    )
    _runtime_quiet_hours_visual_only_enabled: bool = field(default=True, init=False, repr=False)
    _runtime_quiet_hours_start_local: str = field(
        default=_DEFAULT_QUIET_HOURS_START,
        init=False,
        repr=False,
    )
    _runtime_quiet_hours_end_local: str = field(
        default=_DEFAULT_QUIET_HOURS_END,
        init=False,
        repr=False,
    )
    _runtime_quiet_hours_start_time: LocalTime = field(
        default_factory=lambda: _parse_local_time(
            _DEFAULT_QUIET_HOURS_START,
            fallback=_DEFAULT_QUIET_HOURS_START,
        ),
        init=False,
        repr=False,
    )
    _runtime_quiet_hours_end_time: LocalTime = field(
        default_factory=lambda: _parse_local_time(
            _DEFAULT_QUIET_HOURS_END,
            fallback=_DEFAULT_QUIET_HOURS_END,
        ),
        init=False,
        repr=False,
    )
    _runtime_quiet_hours_visual_cue_hold_s: float = field(
        default=_DEFAULT_VISUAL_FIRST_CUE_HOLD_S,
        init=False,
        repr=False,
    )

    _global_backoff_level: int = field(default=0, init=False, repr=False)
    _global_last_feedback_monotonic: float | None = field(default=None, init=False, repr=False)
    _source_backoff_levels: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _source_last_feedback_monotonic: dict[str, float] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        with self._lock:
            self._refresh_runtime_config_locked()
            self._trim_source_repeat_locked(monotonic_now=0.0)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "ProactiveDeliveryPolicy":
        """Build one delivery policy from Twinr config."""

        return cls(
            visual_first_global_cooldown_s=getattr(
                config,
                "proactive_visual_first_audio_global_cooldown_s",
                _DEFAULT_VISUAL_FIRST_GLOBAL_COOLDOWN_S,
            ),
            visual_first_source_repeat_cooldown_s=getattr(
                config,
                "proactive_visual_first_audio_source_repeat_cooldown_s",
                _DEFAULT_VISUAL_FIRST_SOURCE_REPEAT_COOLDOWN_S,
            ),
            visual_first_cue_hold_s=getattr(
                config,
                "proactive_visual_first_cue_hold_s",
                _DEFAULT_VISUAL_FIRST_CUE_HOLD_S,
            ),
            visual_first_max_cue_hold_s=getattr(
                config,
                "proactive_visual_first_max_cue_hold_s",
                _DEFAULT_VISUAL_FIRST_MAX_CUE_HOLD_S,
            ),
            adaptive_visual_first_enabled=getattr(
                config,
                "proactive_visual_first_adaptive_enabled",
                _DEFAULT_VISUAL_FIRST_ADAPTIVE_ENABLED,
            ),
            adaptive_visual_first_backoff_multiplier=getattr(
                config,
                "proactive_visual_first_backoff_multiplier",
                _DEFAULT_VISUAL_FIRST_ADAPTIVE_BACKOFF_MULTIPLIER,
            ),
            adaptive_visual_first_backoff_max_factor=getattr(
                config,
                "proactive_visual_first_backoff_max_factor",
                _DEFAULT_VISUAL_FIRST_ADAPTIVE_BACKOFF_MAX_FACTOR,
            ),
            adaptive_visual_first_backoff_reset_after_s=getattr(
                config,
                "proactive_visual_first_backoff_reset_after_s",
                _DEFAULT_VISUAL_FIRST_ADAPTIVE_RESET_AFTER_S,
            ),
            adaptive_visual_first_interrupted_extra_factor=getattr(
                config,
                "proactive_visual_first_interrupted_extra_factor",
                _DEFAULT_VISUAL_FIRST_INTERRUPTED_EXTRA_FACTOR,
            ),
            quiet_hours_visual_only_enabled=getattr(
                config,
                "proactive_quiet_hours_visual_only_enabled",
                True,
            ),
            quiet_hours_start_local=getattr(
                config,
                "proactive_quiet_hours_start_local",
                _DEFAULT_QUIET_HOURS_START,
            ),
            quiet_hours_end_local=getattr(
                config,
                "proactive_quiet_hours_end_local",
                _DEFAULT_QUIET_HOURS_END,
            ),
            quiet_hours_visual_cue_hold_s=getattr(
                config,
                "proactive_quiet_hours_visual_cue_hold_s",
                None,
            ),
        )

    def decide(
        self,
        *,
        monotonic_now: float,
        local_now: datetime,
        source_id: str,
        safety_exempt: bool,
        audio_policy_snapshot: ReSpeakerAudioPolicySnapshot | None,
    ) -> ProactiveDeliveryDecision:
        """Choose whether one proactive candidate should speak or stay visual."""

        normalized_now = _coerce_monotonic_seconds(monotonic_now)
        defer_reason = _normalize_optional_text(
            getattr(audio_policy_snapshot, "speech_delivery_defer_reason", None)
            if audio_policy_snapshot is not None
            else None
        )
        normalized_source_id = _normalize_source_id(source_id)

        with self._lock:
            self._refresh_runtime_config_locked()
            self._trim_source_repeat_locked(normalized_now)

            if safety_exempt:
                return ProactiveDeliveryDecision(channel="speech")

            if self._quiet_hours_visual_only_locked(local_now):
                return ProactiveDeliveryDecision(
                    channel="display",
                    reason=_REASON_QUIET_HOURS,
                    cue_hold_seconds=self._runtime_quiet_hours_visual_cue_hold_s,
                )

            if defer_reason:
                return ProactiveDeliveryDecision(
                    channel="display",
                    reason=defer_reason,
                    cue_hold_seconds=self._runtime_visual_first_cue_hold_s,
                )

            global_remaining = self._remaining_global_visual_only_locked(normalized_now)
            if global_remaining > 0.0:
                return ProactiveDeliveryDecision(
                    channel="display",
                    reason=_REASON_GLOBAL_COOLDOWN,
                    cue_hold_seconds=self._cooldown_cue_hold_locked(global_remaining),
                )

            source_remaining = self._remaining_source_visual_only_locked(
                normalized_source_id,
                normalized_now,
            )
            if source_remaining > 0.0:
                return ProactiveDeliveryDecision(
                    channel="display",
                    reason=_REASON_SOURCE_COOLDOWN,
                    cue_hold_seconds=self._cooldown_cue_hold_locked(source_remaining),
                )

            return ProactiveDeliveryDecision(channel="speech")

    def note_ignored(self, *, source_id: str, monotonic_now: float) -> None:
        """Start a bounded visual-first window after a prompt was ignored."""

        self._register_visual_first(
            source_id=source_id,
            monotonic_now=monotonic_now,
            interrupted=False,
        )

    def note_interrupted(self, *, source_id: str, monotonic_now: float) -> None:
        """Start a bounded visual-first window after a prompt was interrupted."""

        self._register_visual_first(
            source_id=source_id,
            monotonic_now=monotonic_now,
            interrupted=True,
        )

    def _register_visual_first(
        self,
        *,
        source_id: str,
        monotonic_now: float,
        interrupted: bool,
    ) -> None:
        """Record global and per-source visual-first cooldown windows."""

        normalized_now = _coerce_monotonic_seconds(monotonic_now)
        normalized_source_id = _normalize_source_id(source_id)

        with self._lock:
            self._refresh_runtime_config_locked()
            self._trim_source_repeat_locked(normalized_now)

            event_factor = (
                self._runtime_adaptive_visual_first_interrupted_extra_factor
                if interrupted
                else 1.0
            )

            global_factor = self._advance_global_backoff_locked(normalized_now)
            global_duration = (
                self._runtime_visual_first_global_cooldown_s * global_factor * event_factor
            )
            global_duration = _coerce_seconds(
                global_duration,
                default=self._runtime_visual_first_global_cooldown_s,
                minimum=0.0,
                maximum=4.0 * 3600.0,
            )
            if global_duration > 0.0:
                self._global_visual_only_until = _max_until(
                    self._global_visual_only_until,
                    normalized_now + global_duration,
                )

            if (
                normalized_source_id is not None
                and self._runtime_visual_first_source_repeat_cooldown_s > 0.0
            ):
                source_factor = self._advance_source_backoff_locked(
                    normalized_source_id,
                    normalized_now,
                )
                source_duration = (
                    self._runtime_visual_first_source_repeat_cooldown_s
                    * source_factor
                    * event_factor
                )
                source_duration = _coerce_seconds(
                    source_duration,
                    default=self._runtime_visual_first_source_repeat_cooldown_s,
                    minimum=0.0,
                    maximum=24.0 * 3600.0,
                )
                if source_duration > 0.0:
                    self._source_visual_only_until[normalized_source_id] = _max_until(
                        self._source_visual_only_until.get(normalized_source_id),
                        normalized_now + source_duration,
                    )

            self._trim_source_repeat_locked(normalized_now)

    def _refresh_runtime_config_locked(self) -> None:
        """Sanitize mutable public config fields into hot-path runtime state."""

        signature = (
            self.visual_first_global_cooldown_s,
            self.visual_first_source_repeat_cooldown_s,
            self.visual_first_cue_hold_s,
            self.visual_first_max_cue_hold_s,
            self.adaptive_visual_first_enabled,
            self.adaptive_visual_first_backoff_multiplier,
            self.adaptive_visual_first_backoff_max_factor,
            self.adaptive_visual_first_backoff_reset_after_s,
            self.adaptive_visual_first_interrupted_extra_factor,
            self.quiet_hours_visual_only_enabled,
            self.quiet_hours_start_local,
            self.quiet_hours_end_local,
            self.quiet_hours_visual_cue_hold_s,
        )
        if signature == self._runtime_config_signature:
            return

        self._runtime_visual_first_global_cooldown_s = _coerce_seconds(
            self.visual_first_global_cooldown_s,
            default=_DEFAULT_VISUAL_FIRST_GLOBAL_COOLDOWN_S,
            minimum=0.0,
            maximum=4.0 * 3600.0,
        )
        self._runtime_visual_first_source_repeat_cooldown_s = _coerce_seconds(
            self.visual_first_source_repeat_cooldown_s,
            default=_DEFAULT_VISUAL_FIRST_SOURCE_REPEAT_COOLDOWN_S,
            minimum=0.0,
            maximum=24.0 * 3600.0,
        )
        self._runtime_visual_first_cue_hold_s = _coerce_seconds(
            self.visual_first_cue_hold_s,
            default=_DEFAULT_VISUAL_FIRST_CUE_HOLD_S,
            minimum=1.0,
            maximum=10.0 * 60.0,
        )
        self._runtime_visual_first_max_cue_hold_s = _coerce_seconds(
            self.visual_first_max_cue_hold_s,
            default=_DEFAULT_VISUAL_FIRST_MAX_CUE_HOLD_S,
            minimum=self._runtime_visual_first_cue_hold_s,
            maximum=30.0 * 60.0,
        )

        self._runtime_adaptive_visual_first_enabled = _coerce_bool(
            self.adaptive_visual_first_enabled,
            default=_DEFAULT_VISUAL_FIRST_ADAPTIVE_ENABLED,
        )
        self._runtime_adaptive_visual_first_backoff_multiplier = _coerce_seconds(
            self.adaptive_visual_first_backoff_multiplier,
            default=_DEFAULT_VISUAL_FIRST_ADAPTIVE_BACKOFF_MULTIPLIER,
            minimum=1.0,
            maximum=4.0,
        )
        self._runtime_adaptive_visual_first_backoff_max_factor = _coerce_seconds(
            self.adaptive_visual_first_backoff_max_factor,
            default=_DEFAULT_VISUAL_FIRST_ADAPTIVE_BACKOFF_MAX_FACTOR,
            minimum=1.0,
            maximum=16.0,
        )
        self._runtime_adaptive_visual_first_backoff_reset_after_s = _coerce_seconds(
            self.adaptive_visual_first_backoff_reset_after_s,
            default=_DEFAULT_VISUAL_FIRST_ADAPTIVE_RESET_AFTER_S,
            minimum=1.0,
            maximum=24.0 * 3600.0,
        )
        self._runtime_adaptive_visual_first_interrupted_extra_factor = _coerce_seconds(
            self.adaptive_visual_first_interrupted_extra_factor,
            default=_DEFAULT_VISUAL_FIRST_INTERRUPTED_EXTRA_FACTOR,
            minimum=1.0,
            maximum=4.0,
        )

        self._runtime_quiet_hours_visual_only_enabled = _coerce_bool(
            self.quiet_hours_visual_only_enabled,
            default=True,
        )
        self._runtime_quiet_hours_start_local = _normalize_time_text(
            self.quiet_hours_start_local,
            fallback=_DEFAULT_QUIET_HOURS_START,
        )
        self._runtime_quiet_hours_end_local = _normalize_time_text(
            self.quiet_hours_end_local,
            fallback=_DEFAULT_QUIET_HOURS_END,
        )
        self._runtime_quiet_hours_start_time = _parse_local_time(
            self._runtime_quiet_hours_start_local,
            fallback=_DEFAULT_QUIET_HOURS_START,
        )
        self._runtime_quiet_hours_end_time = _parse_local_time(
            self._runtime_quiet_hours_end_local,
            fallback=_DEFAULT_QUIET_HOURS_END,
        )

        quiet_hours_cue_hold_default = self._runtime_visual_first_cue_hold_s
        self._runtime_quiet_hours_visual_cue_hold_s = _coerce_seconds(
            self.quiet_hours_visual_cue_hold_s,
            default=quiet_hours_cue_hold_default,
            minimum=self._runtime_visual_first_cue_hold_s,
            maximum=60.0 * 60.0,
        )

        self._runtime_config_signature = signature

    def _quiet_hours_visual_only_locked(self, local_now: datetime) -> bool:
        """Return whether the current local time falls into quiet hours."""

        if not self._runtime_quiet_hours_visual_only_enabled:
            return False

        start = self._runtime_quiet_hours_start_time
        end = self._runtime_quiet_hours_end_time
        if start == end:
            return False

        current = local_now.timetz().replace(tzinfo=None)
        if start < end:
            return start <= current < end
        return current >= start or current < end

    def _remaining_global_visual_only_locked(self, monotonic_now: float) -> float:
        """Return the remaining global visual-first cooldown in seconds."""

        until = self._global_visual_only_until
        if until is None:
            return 0.0
        remaining = until - monotonic_now
        if remaining <= 0.0:
            self._global_visual_only_until = None
            return 0.0
        return remaining

    def _remaining_source_visual_only_locked(
        self,
        source_id: str | None,
        monotonic_now: float,
    ) -> float:
        """Return the remaining per-source visual-first cooldown in seconds."""

        if source_id is None:
            return 0.0

        until = self._source_visual_only_until.get(source_id)
        if until is None:
            return 0.0

        remaining = until - monotonic_now
        if remaining <= 0.0:
            self._source_visual_only_until.pop(source_id, None)
            return 0.0
        return remaining

    def _cooldown_cue_hold_locked(self, remaining_cooldown_s: float) -> float:
        """Choose a bounded cue hold based on the remaining suppression window."""

        return min(
            self._runtime_visual_first_max_cue_hold_s,
            max(self._runtime_visual_first_cue_hold_s, remaining_cooldown_s),
        )

    def _advance_global_backoff_locked(self, monotonic_now: float) -> float:
        """Advance adaptive backoff for global suppression and return its factor."""

        self._global_backoff_level = self._next_backoff_level(
            current_level=self._global_backoff_level,
            previous_monotonic=self._global_last_feedback_monotonic,
            monotonic_now=monotonic_now,
        )
        self._global_last_feedback_monotonic = monotonic_now
        return self._backoff_factor_locked(self._global_backoff_level)

    def _advance_source_backoff_locked(self, source_id: str, monotonic_now: float) -> float:
        """Advance adaptive backoff for one source and return its factor."""

        current_level = self._source_backoff_levels.get(source_id, 0)
        previous_monotonic = self._source_last_feedback_monotonic.get(source_id)
        next_level = self._next_backoff_level(
            current_level=current_level,
            previous_monotonic=previous_monotonic,
            monotonic_now=monotonic_now,
        )
        self._source_backoff_levels[source_id] = next_level
        self._source_last_feedback_monotonic[source_id] = monotonic_now
        return self._backoff_factor_locked(next_level)

    def _next_backoff_level(
        self,
        *,
        current_level: int,
        previous_monotonic: float | None,
        monotonic_now: float,
    ) -> int:
        """Return the next bounded adaptive-backoff level."""

        if not self._runtime_adaptive_visual_first_enabled:
            return 1

        if previous_monotonic is None:
            return 1

        if monotonic_now < previous_monotonic:
            return 1

        if monotonic_now - previous_monotonic > self._runtime_adaptive_visual_first_backoff_reset_after_s:
            return 1

        return min(current_level + 1, 32)

    def _backoff_factor_locked(self, level: int) -> float:
        """Map one adaptive backoff level to a bounded duration multiplier."""

        if not self._runtime_adaptive_visual_first_enabled or level <= 1:
            return 1.0

        factor = self._runtime_adaptive_visual_first_backoff_multiplier ** (level - 1)
        if not math.isfinite(factor):
            factor = self._runtime_adaptive_visual_first_backoff_max_factor
        return min(self._runtime_adaptive_visual_first_backoff_max_factor, factor)

    def _trim_source_repeat_locked(self, monotonic_now: float) -> None:
        """Keep bounded source-repeat and adaptive-backoff state."""

        active_until: dict[str, float] = {
            source_id: until
            for source_id, until in self._source_visual_only_until.items()
            if until > monotonic_now
        }

        recent_feedback: dict[str, float] = {}
        for source_id, last_feedback in self._source_last_feedback_monotonic.items():
            if monotonic_now < last_feedback:
                recent_feedback[source_id] = last_feedback
                continue
            if (
                monotonic_now - last_feedback
                <= self._runtime_adaptive_visual_first_backoff_reset_after_s
            ):
                recent_feedback[source_id] = last_feedback

        keep_ids = set(active_until) | set(recent_feedback)
        if len(keep_ids) > _MAX_SOURCE_REPEAT_TRACKING:
            keep_ids = {
                source_id
                for source_id, _ in sorted(
                    (
                        (
                            source_id,
                            max(
                                active_until.get(source_id, float("-inf")),
                                recent_feedback.get(source_id, float("-inf")),
                            ),
                        )
                        for source_id in keep_ids
                    ),
                    key=lambda item: item[1],
                    reverse=True,
                )[:_MAX_SOURCE_REPEAT_TRACKING]
            }

        self._source_visual_only_until = {
            source_id: until
            for source_id, until in active_until.items()
            if source_id in keep_ids
        }
        self._source_last_feedback_monotonic = {
            source_id: last_feedback
            for source_id, last_feedback in recent_feedback.items()
            if source_id in keep_ids
        }
        self._source_backoff_levels = {
            source_id: level
            for source_id, level in self._source_backoff_levels.items()
            if source_id in keep_ids
        }


def _coerce_seconds(
    value: object,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    """Return one finite bounded duration."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default

    if not math.isfinite(number):
        number = default

    return max(minimum, min(maximum, number))


def _coerce_bool(value: object, *, default: bool) -> bool:
    """Return one conservative boolean from typed or text config values."""

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value) != 0.0

    text = _normalize_optional_text(value)
    if text is None:
        return default

    lowered = text.casefold()
    if lowered in {"1", "true", "t", "yes", "y", "on", "enable", "enabled"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off", "disable", "disabled"}:
        return False
    return default


def _coerce_monotonic_seconds(value: object) -> float:
    """Return one finite monotonic-style timestamp, falling back to 0.0."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not math.isfinite(number):
        return 0.0

    return number


def _normalize_optional_text(value: object) -> str | None:
    """Return one collapsed text value or ``None`` when empty."""

    text = " ".join(str(value or "").split()).strip()
    if not text:
        return None
    return text


def _normalize_time_text(value: object, *, fallback: str) -> str:
    """Return one normalized local-time text value."""

    return _normalize_optional_text(value) or fallback


def _parse_local_time(value: str, *, fallback: str) -> LocalTime:
    """Parse one local time using a tolerant ISO-8601-first strategy."""

    for candidate in (value, fallback, _DEFAULT_QUIET_HOURS_START):
        text = _normalize_optional_text(candidate)
        if not text:
            continue

        for parse_candidate in _time_parse_candidates(text):
            try:
                parsed = LocalTime.fromisoformat(parse_candidate)
            except ValueError:
                continue
            return parsed.replace(tzinfo=None)

        try:
            parsed_legacy = datetime.strptime(text, "%H:%M")
        except ValueError:
            continue
        return parsed_legacy.time()

    return datetime.strptime(_DEFAULT_QUIET_HOURS_START, "%H:%M").time()


def _time_parse_candidates(text: str) -> tuple[str, ...]:
    """Return local-time parse candidates for one admin-provided time string."""

    if text.isdigit():
        numeric = int(text)
        if 0 <= numeric <= 23:
            return (f"{numeric:02d}:00", text)

    if ":" not in text and len(text) in {3, 4} and text.isdigit():
        padded = text.zfill(4)
        return (f"{padded[:2]}:{padded[2:]}", text)

    return (text,)


def _normalize_source_id(value: object) -> str | None:
    """Return a bounded stable source id, or ``None`` when no source exists."""

    raw_text = str(value or "")
    # BREAKING: blank/whitespace-only source ids no longer map to "unknown".
    # Missing ids now skip per-source cooldown and only participate in the
    # global cooldown. This avoids false suppression across unrelated sources.
    if not raw_text or raw_text.isspace():
        return None

    oversize_input = len(raw_text) > _MAX_SOURCE_ID_INPUT_CHARS
    digest: str | None = None
    if oversize_input:
        digest = _stable_text_digest(raw_text)
        raw_text = raw_text[:_MAX_SOURCE_ID_INPUT_CHARS]

    normalized = " ".join(raw_text.split()).strip()
    if not normalized:
        return None

    if not oversize_input and len(normalized) <= _MAX_NORMALIZED_SOURCE_ID_CHARS:
        return normalized

    digest = digest or _stable_text_digest(normalized)
    head = normalized[:_MAX_NORMALIZED_SOURCE_ID_CHARS].rstrip()
    return f"{head}#{digest[:_SOURCE_ID_DIGEST_HEX_CHARS]}"


def _stable_text_digest(text: str) -> str:
    """Return one stable short digest for bounded identity keys."""

    digest = hashlib.blake2s(digest_size=16)
    for index in range(0, len(text), 4096):
        digest.update(text[index : index + 4096].encode("utf-8", "surrogatepass"))
    return digest.hexdigest()


def _max_until(existing_until: float | None, candidate_until: float) -> float:
    """Return the later of two cooldown end timestamps."""

    if existing_until is None:
        return candidate_until
    return max(existing_until, candidate_until)