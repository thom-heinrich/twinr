"""Route visual-first proactive social prompts into the HDMI reserve lane.

Display-first proactive social prompts should not take over the fullscreen
presentation surface when the product intent is one small right-lane opener
beside the face. This module keeps that routing separate from the background
worker and reuses the existing reserve-lane cue/history contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from twinr.agent.base_agent.config import TwinrConfig
from twinr.display.ambient_impulse_cues import DisplayAmbientImpulseCue

from .display_reserve_runtime import (
    DisplayReserveRuntimePublisher,
    DisplayReserveRuntimeRequest,
)
from .display_reserve_support import compact_text, utc_now


@dataclass(frozen=True, slots=True)
class DisplaySocialReservePublishResult:
    """Summarize one visual-first social prompt routed into the reserve lane."""

    cue: DisplayAmbientImpulseCue
    exposure_id: str | None = None


@dataclass(slots=True)
class DisplaySocialReservePublisher:
    """Publish one proactive social prompt as a reserve-lane ambient cue."""

    runtime_publisher: DisplayReserveRuntimePublisher
    default_source: str = "proactive_social"

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplaySocialReservePublisher":
        """Build one social-reserve publisher from Twinr configuration."""

        runtime_publisher = DisplayReserveRuntimePublisher.from_config(
            config,
            default_source="proactive_social",
        )
        return cls(
            runtime_publisher=runtime_publisher,
        )

    def publish(
        self,
        *,
        trigger_id: str,
        prompt_text: str,
        display_reason: str | None,
        hold_seconds: float | None,
        now: datetime | None = None,
    ) -> DisplaySocialReservePublishResult:
        """Persist one visual-first social prompt in the HDMI reserve lane.

        The visible copy stays simple on purpose: one large prompt line block
        on the right-hand reserve lane, no eyebrow, and no second rendering
        surface. The later learning path still gets one exposure history entry.
        """

        effective_now = (now or utc_now()).astimezone(timezone.utc)
        normalized_trigger = " ".join(str(trigger_id or "").strip().split()).lower().replace(" ", "_") or "social"
        published = self.runtime_publisher.publish(
            DisplayReserveRuntimeRequest(
                topic_key=normalized_trigger,
                title=prompt_text,
                cue_source=self.default_source,
                history_source="social_trigger",
                action="ask_one",
                attention_state="foreground",
                eyebrow="",
                headline=prompt_text,
                body="",
                symbol="question",
                accent="warm",
                hold_seconds=float(hold_seconds or 0.0),
                reason=compact_text(display_reason, max_len=120) or "social_trigger_display_first",
                semantic_topic_key=normalized_trigger,
                candidate_family="social",
                match_anchors=(prompt_text, normalized_trigger),
                metadata={
                    "trigger_id": normalized_trigger,
                    "display_reason": (display_reason or "").strip(),
                    "signal_source": "social_trigger_display_first",
                },
            ),
            now=effective_now,
        )
        return DisplaySocialReservePublishResult(
            cue=published.cue,
            exposure_id=published.exposure_id,
        )
