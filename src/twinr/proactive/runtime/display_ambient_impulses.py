"""Publish planned calm reserve-card impulses for the HDMI waiting surface.

This module keeps the live publication path very small. Daily sequencing,
nightly preparation, candidate weighting, and persistence live in the reserve
planner modules. The publisher here only decides whether the current runtime
context may expose the next planned reserve-card item right now.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, time as LocalTime, timedelta
import inspect

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.display.ambient_impulse_cues import (
    DisplayAmbientImpulseCue,
    DisplayAmbientImpulseCueStore,
)
from twinr.display.ambient_impulse_history import DisplayAmbientImpulseHistoryStore
from twinr.display.emoji_cues import DisplayEmojiCueStore
from twinr.display.presentation_cues import DisplayPresentationStore

from .display_reserve_companion_planner import DisplayReserveCompanionPlanner
from .display_reserve_day_plan import _DEFAULT_REFRESH_AFTER_LOCAL
from .display_reserve_runtime import DisplayReserveRuntimePublisher, DisplayReserveRuntimeRequest
from .display_reserve_support import default_local_now, parse_local_time as _parse_local_time

_DEFAULT_ENABLED = True
_DEFAULT_QUIET_HOURS_START = "21:00"
_DEFAULT_QUIET_HOURS_END = "07:00"
_SOURCE = "proactive_ambient_impulse"

def _supports_ambient_impulses(config: TwinrConfig) -> bool:
    """Return whether the current display/runtime setup should allow impulses."""

    if not bool(getattr(config, "display_ambient_impulses_enabled", _DEFAULT_ENABLED)):
        return False
    driver = str(getattr(config, "display_driver", "") or "").strip().lower()
    return driver.startswith("hdmi")


def _next_local_boundary(*, local_now: datetime, at_time: LocalTime) -> datetime:
    """Return the next local datetime for one wall-clock boundary."""

    candidate = local_now.replace(
        hour=at_time.hour,
        minute=at_time.minute,
        second=0,
        microsecond=0,
    )
    if candidate <= local_now:
        candidate = candidate + timedelta(days=1)
    return candidate


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

    runtime_publisher: DisplayReserveRuntimePublisher
    active_store: DisplayAmbientImpulseCueStore
    emoji_store: DisplayEmojiCueStore
    presentation_store: DisplayPresentationStore
    planner: DisplayReserveCompanionPlanner
    source: str = _SOURCE
    local_now: Callable[[], datetime] = default_local_now

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayAmbientImpulsePublisher":
        """Build one publisher from the configured display cue stores."""

        runtime_publisher = DisplayReserveRuntimePublisher.from_config(
            config,
            default_source=_SOURCE,
        )
        return cls(
            runtime_publisher=runtime_publisher,
            active_store=runtime_publisher.active_store,
            emoji_store=DisplayEmojiCueStore.from_config(config),
            presentation_store=DisplayPresentationStore.from_config(config),
            planner=DisplayReserveCompanionPlanner.from_config(config),
        )

    @property
    def candidate_loader(self) -> Callable[..., tuple[AmbientDisplayImpulseCandidate, ...]]:
        """Expose the planner candidate loader for tests and dependency injection."""

        return self.planner.candidate_loader

    @property
    def history_store(self) -> DisplayAmbientImpulseHistoryStore:
        """Expose the shared reserve history store for tests and observability."""

        return self.runtime_publisher.history_store

    def set_candidate_loader(
        self,
        value: Callable[..., tuple[AmbientDisplayImpulseCandidate, ...]],
    ) -> None:
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
        effective_local_now = (local_now or self.local_now()).astimezone()
        if self.emoji_store.load_active(now=effective_local_now) is not None:
            return DisplayAmbientImpulsePublishResult(action="blocked", reason="emoji_surface_owned")
        if self.presentation_store.load_active(now=effective_local_now) is not None:
            return DisplayAmbientImpulsePublishResult(action="blocked", reason="presentation_surface_owned")
        if self.active_store.load_active(now=effective_local_now) is not None:
            return DisplayAmbientImpulsePublishResult(action="blocked", reason="ambient_impulse_active")
        if self._quiet_hours_active(config=config, local_now=effective_local_now):
            restored = self._restore_passive_fill(
                config=config,
                local_now=effective_local_now,
                reason="quiet_hours_passive_fill",
            )
            if restored is not None:
                return restored
            return DisplayAmbientImpulsePublishResult(action="blocked", reason="quiet_hours")
        if not presence_active:
            restored = self._restore_passive_fill(
                config=config,
                local_now=effective_local_now,
                reason="no_active_presence_passive_fill",
            )
            if restored is not None:
                return restored
            return DisplayAmbientImpulsePublishResult(action="blocked", reason="no_active_presence")

        item = self.planner.peek_next_item(
            config=config,
            local_now=effective_local_now,
        )
        if item is None:
            fallback_item = self.planner.peek_idle_fill_item(
                config=config,
                local_now=effective_local_now,
            )
            if fallback_item is None:
                return DisplayAmbientImpulsePublishResult(action="inactive", reason="no_planned_item")
            visible_only_result = self.runtime_publisher.show_visible_only(
                DisplayReserveRuntimeRequest(
                    topic_key=fallback_item.topic_key,
                    title=fallback_item.title,
                    cue_source=self.source,
                    history_source=fallback_item.source,
                    action=fallback_item.action,
                    attention_state=fallback_item.attention_state,
                    eyebrow=fallback_item.eyebrow,
                    headline=fallback_item.headline,
                    body=fallback_item.body,
                    symbol=fallback_item.symbol,
                    accent=fallback_item.accent,
                    hold_seconds=self._idle_fill_hold_seconds(
                        config=config,
                        local_now=effective_local_now,
                        fallback_item=fallback_item,
                    ),
                    reason=f"{fallback_item.reason}; idle_fill",
                    semantic_topic_key=fallback_item.semantic_key(),
                    candidate_family=fallback_item.candidate_family,
                    match_anchors=(fallback_item.title, fallback_item.headline, fallback_item.body),
                    metadata={
                        "eyebrow": fallback_item.eyebrow,
                        "accent": fallback_item.accent,
                        "symbol": fallback_item.symbol,
                        "idle_fill": True,
                    },
                ),
                now=effective_local_now,
            )
            return DisplayAmbientImpulsePublishResult(
                action="restored_fill",
                reason="plan_exhausted_idle_fill",
                topic_key=fallback_item.topic_key,
                cue=visible_only_result.cue,
            )

        published = self.runtime_publisher.publish(
            DisplayReserveRuntimeRequest(
                topic_key=item.topic_key,
                title=item.title,
                cue_source=self.source,
                history_source=item.source,
                action=item.action,
                attention_state=item.attention_state,
                eyebrow=item.eyebrow,
                headline=item.headline,
                body=item.body,
                symbol=item.symbol,
                accent=item.accent,
                hold_seconds=item.hold_seconds,
                reason=item.reason,
                semantic_topic_key=item.semantic_key(),
                candidate_family=item.candidate_family,
                match_anchors=(item.title, item.headline, item.body),
                metadata={
                    "eyebrow": item.eyebrow,
                    "accent": item.accent,
                    "symbol": item.symbol,
                },
            ),
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
            cue=published.cue,
        )

    def _restore_passive_fill(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime,
        reason: str,
    ) -> DisplayAmbientImpulsePublishResult | None:
        """Restore one passive reserve fill without recording a new exposure.

        Temporary right-lane overrides such as social prompts may expire while
        normal ambient publishing is intentionally blocked. In that state Twinr
        should only restore a card that was already truly shown earlier the
        same day instead of surfacing the next unpublished plan item and
        pinning the rotation on that first topic.
        """

        plan = self.planner.ensure_plan(
            config=config,
            local_now=local_now,
        )
        fill_item = plan.last_shown_item()
        if fill_item is None:
            fill_item = self.planner.peek_idle_fill_item(
                config=config,
                local_now=local_now,
            )
        if fill_item is None:
            return None
        restored = self.runtime_publisher.show_visible_only(
            DisplayReserveRuntimeRequest(
                topic_key=fill_item.topic_key,
                title=fill_item.title,
                cue_source=self.source,
                history_source=fill_item.source,
                action=fill_item.action,
                attention_state=fill_item.attention_state,
                eyebrow=fill_item.eyebrow,
                headline=fill_item.headline,
                body=fill_item.body,
                symbol=fill_item.symbol,
                accent=fill_item.accent,
                hold_seconds=self._idle_fill_hold_seconds(
                    config=config,
                    local_now=local_now,
                    fallback_item=fill_item,
                ),
                reason=f"{fill_item.reason}; {reason}",
                semantic_topic_key=fill_item.semantic_key(),
                candidate_family=fill_item.candidate_family,
                match_anchors=(fill_item.title, fill_item.headline, fill_item.body),
                metadata={
                    "eyebrow": fill_item.eyebrow,
                    "accent": fill_item.accent,
                    "symbol": fill_item.symbol,
                    "idle_fill": True,
                },
            ),
            now=local_now,
        )
        return DisplayAmbientImpulsePublishResult(
            action="restored_fill",
            reason=reason,
            topic_key=fill_item.topic_key,
            cue=restored.cue,
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

    def _idle_fill_hold_seconds(
        self,
        *,
        config: TwinrConfig,
        local_now: datetime,
        fallback_item: object,
    ) -> float:
        """Return how long one passive exhausted-plan fill may stay visible."""

        quiet_start = _parse_local_time(
            getattr(config, "proactive_quiet_hours_start_local", _DEFAULT_QUIET_HOURS_START),
            fallback=_DEFAULT_QUIET_HOURS_START,
        )
        refresh_after = _parse_local_time(
            getattr(config, "display_reserve_bus_refresh_after_local", _DEFAULT_REFRESH_AFTER_LOCAL),
            fallback=_DEFAULT_REFRESH_AFTER_LOCAL,
        )
        next_quiet = _next_local_boundary(local_now=local_now, at_time=quiet_start)
        next_refresh = _next_local_boundary(local_now=local_now, at_time=refresh_after)
        seconds_until_boundary = min(
            (next_quiet - local_now).total_seconds(),
            (next_refresh - local_now).total_seconds(),
        )
        base_hold_seconds = max(
            60.0,
            float(getattr(fallback_item, "hold_seconds", 0.0) or 0.0),
        )
        return max(60.0, min(max(base_hold_seconds, 15.0 * 60.0), seconds_until_boundary))
