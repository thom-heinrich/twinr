"""Publish planned calm reserve-card impulses for the HDMI waiting surface.

This module keeps the live publication path very small. Daily sequencing,
candidate weighting, and persistence live in ``display_reserve_day_plan.py``.
The publisher here only decides whether the current runtime context may expose
the next planned reserve-card item right now.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, time as LocalTime
import inspect

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.display.ambient_impulse_cues import (
    DisplayAmbientImpulseController,
    DisplayAmbientImpulseCue,
    DisplayAmbientImpulseCueStore,
)
from twinr.display.emoji_cues import DisplayEmojiCueStore
from twinr.display.presentation_cues import DisplayPresentationStore

from .display_reserve_day_plan import DisplayReserveDayPlanner

_DEFAULT_ENABLED = True
_DEFAULT_QUIET_HOURS_START = "21:00"
_DEFAULT_QUIET_HOURS_END = "07:00"
_SOURCE = "proactive_ambient_impulse"


def _default_local_now() -> datetime:
    """Return the current local wall clock as an aware datetime."""

    return datetime.now().astimezone()


def _parse_local_time(text: object, *, fallback: str) -> LocalTime:
    """Parse one normalized local-time string."""

    compact = str(text or "").strip() or fallback
    hour_text, separator, minute_text = compact.partition(":")
    if separator != ":":
        hour_text, minute_text = fallback.split(":", 1)
    try:
        hour = int(hour_text)
        minute = int(minute_text)
    except ValueError:
        fallback_hour, fallback_minute = fallback.split(":", 1)
        return LocalTime(hour=int(fallback_hour), minute=int(fallback_minute))
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        fallback_hour, fallback_minute = fallback.split(":", 1)
        return LocalTime(hour=int(fallback_hour), minute=int(fallback_minute))
    return LocalTime(hour=hour, minute=minute)


def _supports_ambient_impulses(config: TwinrConfig) -> bool:
    """Return whether the current display/runtime setup should allow impulses."""

    if not bool(getattr(config, "display_ambient_impulses_enabled", _DEFAULT_ENABLED)):
        return False
    driver = str(getattr(config, "display_driver", "") or "").strip().lower()
    return driver.startswith("hdmi")


@dataclass(frozen=True, slots=True)
class DisplayAmbientImpulsePublishResult:
    """Summarize one ambient-impulse publish attempt."""

    action: str
    reason: str
    topic_key: str | None = None
    cue: DisplayAmbientImpulseCue | None = None


@dataclass(slots=True)
class DisplayAmbientImpulsePublisher:
    """Publish the next planned reserve impulse when the runtime is ready."""

    controller: DisplayAmbientImpulseController
    active_store: DisplayAmbientImpulseCueStore
    emoji_store: DisplayEmojiCueStore
    presentation_store: DisplayPresentationStore
    planner: DisplayReserveDayPlanner
    source: str = _SOURCE
    local_now: Callable[[], datetime] = _default_local_now

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayAmbientImpulsePublisher":
        """Build one publisher from the configured display cue stores."""

        controller = DisplayAmbientImpulseController.from_config(
            config,
            default_source=_SOURCE,
        )
        return cls(
            controller=controller,
            active_store=controller.store,
            emoji_store=DisplayEmojiCueStore.from_config(config),
            presentation_store=DisplayPresentationStore.from_config(config),
            planner=DisplayReserveDayPlanner.from_config(config),
        )

    @property
    def candidate_loader(self) -> Callable[..., tuple[AmbientDisplayImpulseCandidate, ...]]:
        """Expose the planner candidate loader for tests and dependency injection."""

        return self.planner.candidate_loader

    @candidate_loader.setter
    def candidate_loader(self, value: Callable[..., tuple[AmbientDisplayImpulseCandidate, ...]]) -> None:
        """Replace the planner candidate loader for tests or alternate wiring."""

        try:
            signature = inspect.signature(value)
        except (TypeError, ValueError):
            self.planner.candidate_loader = value
            return
        accepts_max_items = "max_items" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_max_items:
            self.planner.candidate_loader = value
            return

        def _wrapped_candidate_loader(
            config: TwinrConfig,
            *,
            local_now: datetime,
            max_items: int,
        ) -> tuple[AmbientDisplayImpulseCandidate, ...]:
            del max_items
            return tuple(value(config, local_now=local_now))

        self.planner.candidate_loader = _wrapped_candidate_loader

    def publish_if_due(
        self,
        *,
        config: TwinrConfig,
        monotonic_now: float,
        runtime_status: str,
        presence_active: bool,
        local_now: datetime | None = None,
    ) -> DisplayAmbientImpulsePublishResult:
        """Publish the next planned reserve cue when the live context allows it."""

        del monotonic_now
        if not _supports_ambient_impulses(config):
            return DisplayAmbientImpulsePublishResult(action="inactive", reason="unsupported")
        if runtime_status != "waiting":
            return DisplayAmbientImpulsePublishResult(action="blocked", reason="runtime_not_waiting")
        if not presence_active:
            return DisplayAmbientImpulsePublishResult(action="blocked", reason="no_active_presence")
        effective_local_now = (local_now or self.local_now()).astimezone()
        if self._quiet_hours_active(config=config, local_now=effective_local_now):
            return DisplayAmbientImpulsePublishResult(action="blocked", reason="quiet_hours")
        if self.emoji_store.load_active(now=effective_local_now) is not None:
            return DisplayAmbientImpulsePublishResult(action="blocked", reason="emoji_surface_owned")
        if self.presentation_store.load_active(now=effective_local_now) is not None:
            return DisplayAmbientImpulsePublishResult(action="blocked", reason="presentation_surface_owned")
        if self.active_store.load_active(now=effective_local_now) is not None:
            return DisplayAmbientImpulsePublishResult(action="blocked", reason="ambient_impulse_active")

        item = self.planner.peek_next_item(
            config=config,
            local_now=effective_local_now,
        )
        if item is None:
            return DisplayAmbientImpulsePublishResult(action="inactive", reason="no_planned_item")

        cue = self.controller.show_impulse(
            topic_key=item.topic_key,
            eyebrow=item.eyebrow,
            headline=item.headline,
            body=item.body,
            symbol=item.symbol,
            accent=item.accent,
            action=item.action,
            attention_state=item.attention_state,
            hold_seconds=item.hold_seconds,
            source=self.source,
            now=effective_local_now,
        )
        self.planner.mark_published(
            config=config,
            local_now=effective_local_now,
        )
        return DisplayAmbientImpulsePublishResult(
            action="published",
            reason=item.reason,
            topic_key=item.topic_key,
            cue=cue,
        )

    def _quiet_hours_active(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime,
    ) -> bool:
        """Return whether the current local time falls into quiet hours."""

        start = _parse_local_time(
            getattr(config, "proactive_quiet_hours_start_local", _DEFAULT_QUIET_HOURS_START),
            fallback=_DEFAULT_QUIET_HOURS_START,
        )
        end = _parse_local_time(
            getattr(config, "proactive_quiet_hours_end_local", _DEFAULT_QUIET_HOURS_END),
            fallback=_DEFAULT_QUIET_HOURS_END,
        )
        if start == end:
            return False
        current = local_now.timetz().replace(tzinfo=None)
        if start < end:
            return start <= current < end
        return current >= start or current < end
