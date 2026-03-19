"""Choose speech versus visual-first delivery for proactive background work.

This module keeps delivery-mode policy out of the realtime background mixin.
It owns bounded state for visual-first cooldowns after ignored or interrupted
proactive speech and resolves quiet-hours plus audio/media suppression into one
delivery decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time as LocalTime
import math

from twinr.agent.base_agent.config import TwinrConfig
from twinr.proactive.runtime.audio_policy import ReSpeakerAudioPolicySnapshot


_DEFAULT_VISUAL_FIRST_GLOBAL_COOLDOWN_S = 5.0 * 60.0
_DEFAULT_VISUAL_FIRST_SOURCE_REPEAT_COOLDOWN_S = 15.0 * 60.0
_DEFAULT_VISUAL_FIRST_CUE_HOLD_S = 45.0
_DEFAULT_QUIET_HOURS_START = "21:00"
_DEFAULT_QUIET_HOURS_END = "07:00"
_MAX_SOURCE_REPEAT_TRACKING = 64


@dataclass(frozen=True, slots=True)
class ProactiveDeliveryDecision:
    """Describe how one proactive prompt should be delivered."""

    channel: str
    reason: str | None = None
    cue_hold_seconds: float | None = None


@dataclass(slots=True)
class ProactiveDeliveryPolicy:
    """Track bounded visual-first state for proactive background delivery."""

    visual_first_global_cooldown_s: float = _DEFAULT_VISUAL_FIRST_GLOBAL_COOLDOWN_S
    visual_first_source_repeat_cooldown_s: float = _DEFAULT_VISUAL_FIRST_SOURCE_REPEAT_COOLDOWN_S
    visual_first_cue_hold_s: float = _DEFAULT_VISUAL_FIRST_CUE_HOLD_S
    quiet_hours_visual_only_enabled: bool = True
    quiet_hours_start_local: str = _DEFAULT_QUIET_HOURS_START
    quiet_hours_end_local: str = _DEFAULT_QUIET_HOURS_END
    _global_visual_only_until: float | None = field(default=None, repr=False)
    _source_visual_only_until: dict[str, float] = field(default_factory=dict, repr=False)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "ProactiveDeliveryPolicy":
        """Build one delivery policy from Twinr config."""

        return cls(
            visual_first_global_cooldown_s=_coerce_seconds(
                getattr(
                    config,
                    "proactive_visual_first_audio_global_cooldown_s",
                    _DEFAULT_VISUAL_FIRST_GLOBAL_COOLDOWN_S,
                ),
                default=_DEFAULT_VISUAL_FIRST_GLOBAL_COOLDOWN_S,
                minimum=0.0,
                maximum=4.0 * 3600.0,
            ),
            visual_first_source_repeat_cooldown_s=_coerce_seconds(
                getattr(
                    config,
                    "proactive_visual_first_audio_source_repeat_cooldown_s",
                    _DEFAULT_VISUAL_FIRST_SOURCE_REPEAT_COOLDOWN_S,
                ),
                default=_DEFAULT_VISUAL_FIRST_SOURCE_REPEAT_COOLDOWN_S,
                minimum=0.0,
                maximum=24.0 * 3600.0,
            ),
            visual_first_cue_hold_s=_coerce_seconds(
                getattr(
                    config,
                    "proactive_visual_first_cue_hold_s",
                    _DEFAULT_VISUAL_FIRST_CUE_HOLD_S,
                ),
                default=_DEFAULT_VISUAL_FIRST_CUE_HOLD_S,
                minimum=1.0,
                maximum=10.0 * 60.0,
            ),
            quiet_hours_visual_only_enabled=bool(
                getattr(config, "proactive_quiet_hours_visual_only_enabled", True)
            ),
            quiet_hours_start_local=_normalize_time_text(
                getattr(config, "proactive_quiet_hours_start_local", _DEFAULT_QUIET_HOURS_START),
                fallback=_DEFAULT_QUIET_HOURS_START,
            ),
            quiet_hours_end_local=_normalize_time_text(
                getattr(config, "proactive_quiet_hours_end_local", _DEFAULT_QUIET_HOURS_END),
                fallback=_DEFAULT_QUIET_HOURS_END,
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

        self._trim_source_repeat(monotonic_now)
        normalized_source_id = " ".join(str(source_id or "").split()).strip() or "unknown"
        if safety_exempt:
            return ProactiveDeliveryDecision(channel="speech")
        if self._quiet_hours_visual_only(local_now):
            return ProactiveDeliveryDecision(
                channel="display",
                reason="quiet_hours_visual_only",
                cue_hold_seconds=self.visual_first_cue_hold_s,
            )
        if audio_policy_snapshot is not None and audio_policy_snapshot.speech_delivery_defer_reason:
            return ProactiveDeliveryDecision(
                channel="display",
                reason=audio_policy_snapshot.speech_delivery_defer_reason,
                cue_hold_seconds=self.visual_first_cue_hold_s,
            )
        if self._global_visual_only_active(monotonic_now):
            return ProactiveDeliveryDecision(
                channel="display",
                reason="recent_audio_display_first_cooldown",
                cue_hold_seconds=self.visual_first_cue_hold_s,
            )
        if self._source_visual_only_active(normalized_source_id, monotonic_now):
            return ProactiveDeliveryDecision(
                channel="display",
                reason="recent_source_audio_display_first_cooldown",
                cue_hold_seconds=self.visual_first_cue_hold_s,
            )
        return ProactiveDeliveryDecision(channel="speech")

    def note_ignored(self, *, source_id: str, monotonic_now: float) -> None:
        """Start a bounded visual-first window after a prompt was ignored."""

        self._register_visual_first(source_id=source_id, monotonic_now=monotonic_now)

    def note_interrupted(self, *, source_id: str, monotonic_now: float) -> None:
        """Start a bounded visual-first window after a prompt was interrupted."""

        self._register_visual_first(source_id=source_id, monotonic_now=monotonic_now)

    def _register_visual_first(self, *, source_id: str, monotonic_now: float) -> None:
        """Record global and per-source visual-first cooldown windows."""

        normalized_source_id = " ".join(str(source_id or "").split()).strip() or "unknown"
        if self.visual_first_global_cooldown_s > 0.0:
            self._global_visual_only_until = monotonic_now + self.visual_first_global_cooldown_s
        if self.visual_first_source_repeat_cooldown_s > 0.0:
            self._source_visual_only_until[normalized_source_id] = (
                monotonic_now + self.visual_first_source_repeat_cooldown_s
            )
        self._trim_source_repeat(monotonic_now)

    def _quiet_hours_visual_only(self, local_now: datetime) -> bool:
        """Return whether the current local time falls into quiet hours."""

        if not self.quiet_hours_visual_only_enabled:
            return False
        start = _parse_local_time(self.quiet_hours_start_local, fallback=_DEFAULT_QUIET_HOURS_START)
        end = _parse_local_time(self.quiet_hours_end_local, fallback=_DEFAULT_QUIET_HOURS_END)
        if start == end:
            return False
        current = local_now.timetz().replace(tzinfo=None)
        if start < end:
            return start <= current < end
        return current >= start or current < end

    def _global_visual_only_active(self, monotonic_now: float) -> bool:
        """Return whether the global visual-first cooldown is active."""

        if self._global_visual_only_until is None:
            return False
        if monotonic_now >= self._global_visual_only_until:
            self._global_visual_only_until = None
            return False
        return True

    def _source_visual_only_active(self, source_id: str, monotonic_now: float) -> bool:
        """Return whether one source stays in visual-first repeat cooldown."""

        until = self._source_visual_only_until.get(source_id)
        if until is None:
            return False
        if monotonic_now >= until:
            self._source_visual_only_until.pop(source_id, None)
            return False
        return True

    def _trim_source_repeat(self, monotonic_now: float) -> None:
        """Keep bounded source-repeat state only for active cooldowns."""

        active_entries = {
            source_id: until
            for source_id, until in self._source_visual_only_until.items()
            if until > monotonic_now
        }
        if len(active_entries) <= _MAX_SOURCE_REPEAT_TRACKING:
            self._source_visual_only_until = active_entries
            return
        newest = sorted(active_entries.items(), key=lambda item: item[1], reverse=True)
        self._source_visual_only_until = dict(newest[:_MAX_SOURCE_REPEAT_TRACKING])


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


def _normalize_time_text(value: object, *, fallback: str) -> str:
    """Return one normalized ``HH:MM``-style text value."""

    text = " ".join(str(value or "").split()).strip()
    if not text:
        return fallback
    return text


def _parse_local_time(value: str, *, fallback: str) -> LocalTime:
    """Parse one local ``HH:MM`` time with a deterministic fallback."""

    for candidate in (value, fallback):
        try:
            parsed = datetime.strptime(candidate, "%H:%M")
        except ValueError:
            continue
        return parsed.time()
    return datetime.strptime(_DEFAULT_QUIET_HOURS_START, "%H:%M").time()
