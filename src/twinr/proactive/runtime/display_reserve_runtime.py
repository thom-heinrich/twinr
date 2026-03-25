"""Publish all right-lane reserve items through one shared runtime path.

The HDMI reserve area is a single surface even when the underlying items come
from different producers such as the daily ambient planner or a real-time
social/focus opener. This module centralizes the actual cue-store + exposure-
history write path so those producers do not keep reimplementing slightly
different reserve-side effects.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone

from twinr.agent.base_agent.config import TwinrConfig
from twinr.display.ambient_impulse_cues import (
    DisplayAmbientImpulseController,
    DisplayAmbientImpulseCue,
    DisplayAmbientImpulseCueStore,
)
from twinr.display.ambient_impulse_history import DisplayAmbientImpulseHistoryStore
from .display_reserve_support import compact_text, utc_now


@dataclass(frozen=True, slots=True)
class DisplayReserveRuntimeRequest:
    """Describe one concrete item to render on the right reserve lane."""

    topic_key: str
    title: str
    cue_source: str
    history_source: str
    action: str
    attention_state: str
    eyebrow: str
    headline: str
    body: str
    symbol: str
    accent: str
    hold_seconds: float
    reason: str
    candidate_family: str = "general"
    match_anchors: tuple[str, ...] = ()
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class DisplayReserveRuntimePublishResult:
    """Summarize one shared reserve-lane publish."""

    cue: DisplayAmbientImpulseCue
    exposure_id: str | None = None


@dataclass(slots=True)
class DisplayReserveRuntimePublisher:
    """Persist one reserve-lane item through the shared cue/history contract."""

    controller: DisplayAmbientImpulseController
    active_store: DisplayAmbientImpulseCueStore
    history_store: DisplayAmbientImpulseHistoryStore

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        default_source: str,
    ) -> "DisplayReserveRuntimePublisher":
        """Build one shared reserve publisher from Twinr configuration."""

        controller = DisplayAmbientImpulseController.from_config(
            config,
            default_source=default_source,
        )
        return cls(
            controller=controller,
            active_store=controller.store,
            history_store=DisplayAmbientImpulseHistoryStore.from_config(config),
        )

    def publish(
        self,
        request: DisplayReserveRuntimeRequest,
        *,
        now: datetime | None = None,
    ) -> DisplayReserveRuntimePublishResult:
        """Show one reserve-lane item and append one exposure history record."""

        effective_now = (now or utc_now()).astimezone(timezone.utc)
        cue = self.controller.show_impulse(
            topic_key=request.topic_key,
            eyebrow=request.eyebrow,
            headline=request.headline,
            body=request.body,
            symbol=request.symbol,
            accent=request.accent,
            action=request.action,
            attention_state=request.attention_state,
            hold_seconds=request.hold_seconds,
            source=request.cue_source,
            now=effective_now,
        )
        expires_at = datetime.fromisoformat(str(cue.expires_at).replace("Z", "+00:00"))
        metadata: dict[str, object] = {"reason": request.reason, "candidate_family": request.candidate_family}
        if isinstance(request.metadata, Mapping):
            for raw_key, raw_value in request.metadata.items():
                key = compact_text(raw_key, max_len=80)
                if not key:
                    continue
                metadata[key] = raw_value
        exposure = self.history_store.append_exposure(
            source=request.history_source,
            topic_key=request.topic_key,
            title=request.title,
            headline=cue.headline,
            body=cue.body,
            action=request.action,
            attention_state=request.attention_state,
            shown_at=effective_now,
            expires_at=expires_at,
            match_anchors=self._match_anchors(request),
            metadata=metadata,
        )
        return DisplayReserveRuntimePublishResult(
            cue=cue,
            exposure_id=exposure.exposure_id,
        )

    def show_visible_only(
        self,
        request: DisplayReserveRuntimeRequest,
        *,
        now: datetime | None = None,
    ) -> DisplayReserveRuntimePublishResult:
        """Show one reserve-lane item without appending a new exposure entry.

        This is used for passive idle-fill behavior after a daily plan is
        exhausted. The surface stays populated, but the same item is not
        counted as a second shown-card exposure.
        """

        effective_now = (now or utc_now()).astimezone(timezone.utc)
        cue = self.controller.show_impulse(
            topic_key=request.topic_key,
            eyebrow=request.eyebrow,
            headline=request.headline,
            body=request.body,
            symbol=request.symbol,
            accent=request.accent,
            action=request.action,
            attention_state=request.attention_state,
            hold_seconds=request.hold_seconds,
            source=request.cue_source,
            now=effective_now,
        )
        return DisplayReserveRuntimePublishResult(cue=cue, exposure_id=None)

    def _match_anchors(self, request: DisplayReserveRuntimeRequest) -> tuple[str, ...]:
        """Return bounded extra anchors for later feedback correlation."""

        anchors: list[str] = []
        for value in request.match_anchors:
            compact = compact_text(value, max_len=160)
            if compact:
                anchors.append(compact)
        return tuple(anchors[:8])


__all__ = [
    "DisplayReserveRuntimePublishResult",
    "DisplayReserveRuntimePublisher",
    "DisplayReserveRuntimeRequest",
]
