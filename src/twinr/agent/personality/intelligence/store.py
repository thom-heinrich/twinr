"""Persist RSS/world-intelligence state through remote-primary snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality.intelligence.models import (
    DEFAULT_WORLD_INTELLIGENCE_STATE_KIND,
    DEFAULT_WORLD_INTELLIGENCE_SUBSCRIPTIONS_KIND,
    WorldFeedSubscription,
    WorldIntelligenceState,
)
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore


def _resolve_remote_state(
    *,
    config: TwinrConfig,
    remote_state: LongTermRemoteStateStore | None,
) -> LongTermRemoteStateStore | None:
    """Resolve the remote snapshot adapter for one intelligence call."""

    if remote_state is not None:
        return remote_state
    resolved = LongTermRemoteStateStore.from_config(config)
    if not getattr(resolved, "enabled", False):
        return None
    return resolved


class WorldIntelligenceStore(Protocol):
    """Describe remote-backed persistence for subscriptions and timing state."""

    def load_subscriptions(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[WorldFeedSubscription, ...]:
        """Load persisted world-intelligence subscriptions."""

    def save_subscriptions(
        self,
        *,
        config: TwinrConfig,
        subscriptions: Sequence[WorldFeedSubscription],
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist world-intelligence subscriptions."""

    def load_state(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> WorldIntelligenceState:
        """Load persisted global refresh/discovery timing state."""

    def save_state(
        self,
        *,
        config: TwinrConfig,
        state: WorldIntelligenceState,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist global refresh/discovery timing state."""


@dataclass(slots=True)
class RemoteStateWorldIntelligenceStore:
    """Load and save world-intelligence snapshots via remote-primary state."""

    subscriptions_snapshot_kind: str = DEFAULT_WORLD_INTELLIGENCE_SUBSCRIPTIONS_KIND
    state_snapshot_kind: str = DEFAULT_WORLD_INTELLIGENCE_STATE_KIND

    def load_subscriptions(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[WorldFeedSubscription, ...]:
        """Load persisted feed subscriptions when present."""

        resolved = _resolve_remote_state(config=config, remote_state=remote_state)
        if resolved is None:
            return ()
        payload = resolved.load_snapshot(snapshot_kind=self.subscriptions_snapshot_kind)
        if payload is None:
            return ()
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.subscriptions_snapshot_kind} must decode to a mapping payload.")
        items = payload.get("items")
        if items is None:
            return ()
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
            raise ValueError(f"{self.subscriptions_snapshot_kind}.items must be a sequence.")
        loaded: list[WorldFeedSubscription] = []
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise ValueError(f"{self.subscriptions_snapshot_kind}.items[{index}] must be a mapping.")
            loaded.append(WorldFeedSubscription.from_payload(item))
        return tuple(loaded)

    def save_subscriptions(
        self,
        *,
        config: TwinrConfig,
        subscriptions: Sequence[WorldFeedSubscription],
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist feed subscriptions through remote-primary state."""

        resolved = _resolve_remote_state(config=config, remote_state=remote_state)
        if resolved is None:
            return
        resolved.save_snapshot(
            snapshot_kind=self.subscriptions_snapshot_kind,
            payload={
                "schema_version": 1,
                "items": [item.to_payload() for item in subscriptions],
            },
        )

    def load_state(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> WorldIntelligenceState:
        """Load the global refresh/discovery timing snapshot."""

        resolved = _resolve_remote_state(config=config, remote_state=remote_state)
        if resolved is None:
            return WorldIntelligenceState()
        payload = resolved.load_snapshot(snapshot_kind=self.state_snapshot_kind)
        if payload is None:
            return WorldIntelligenceState()
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.state_snapshot_kind} must decode to a mapping payload.")
        return WorldIntelligenceState.from_payload(payload)

    def save_state(
        self,
        *,
        config: TwinrConfig,
        state: WorldIntelligenceState,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist the global refresh/discovery timing snapshot."""

        resolved = _resolve_remote_state(config=config, remote_state=remote_state)
        if resolved is None:
            return
        resolved.save_snapshot(
            snapshot_kind=self.state_snapshot_kind,
            payload=state.to_payload(),
        )

