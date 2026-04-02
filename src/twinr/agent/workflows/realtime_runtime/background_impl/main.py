"""Compose the realtime background mixin from focused helper mixins."""

# mypy: ignore-errors

from __future__ import annotations

from twinr.agent.workflows.realtime_runtime.background_impl.automation import BackgroundAutomationMixin
from twinr.agent.workflows.realtime_runtime.background_impl.longterm import BackgroundLongTermMixin
from twinr.agent.workflows.realtime_runtime.background_impl.nightly import BackgroundNightlyMixin
from twinr.agent.workflows.realtime_runtime.background_impl.observation import BackgroundObservationMixin
from twinr.agent.workflows.realtime_runtime.background_impl.social import BackgroundSocialMixin
from twinr.agent.workflows.realtime_runtime.background_impl.support import BackgroundSupportMixin


class TwinrRealtimeBackgroundMixinImpl(
    BackgroundAutomationMixin,
    BackgroundNightlyMixin,
    BackgroundLongTermMixin,
    BackgroundSocialMixin,
    BackgroundObservationMixin,
    BackgroundSupportMixin,
):
    """Handle background delivery paths without destabilizing the live loop."""
