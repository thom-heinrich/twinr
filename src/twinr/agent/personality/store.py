"""Load and save structured personality state via remote-backed snapshots."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality.models import (
    DEFAULT_PERSONALITY_SNAPSHOT_KIND,
    INTERACTION_SIGNAL_SNAPSHOT_KIND,
    PERSONALITY_DELTA_SNAPSHOT_KIND,
    PLACE_SIGNAL_SNAPSHOT_KIND,
    WORLD_SIGNAL_SNAPSHOT_KIND,
    InteractionSignal,
    PersonalityDelta,
    PersonalitySnapshot,
    PlaceSignal,
    WorldSignal,
)
from twinr.agent.personality._remote_state_utils import (
    resolve_remote_state as _resolve_remote_state,
)
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore

_ItemT = TypeVar("_ItemT")


def _load_list_snapshot(
    *,
    config: TwinrConfig,
    remote_state: LongTermRemoteStateStore | None,
    snapshot_kind: str,
    item_factory: Callable[[Mapping[str, object]], _ItemT],
) -> tuple[_ItemT, ...]:
    """Load a typed list snapshot from remote state when present."""

    resolved_remote_state = _resolve_remote_state(config=config, remote_state=remote_state)
    if resolved_remote_state is None:
        return ()
    payload = resolved_remote_state.load_snapshot(snapshot_kind=snapshot_kind)
    if payload is None:
        return ()
    if not isinstance(payload, Mapping):
        raise ValueError(f"{snapshot_kind} must decode to a mapping payload.")
    items = payload.get("items")
    if items is None:
        return ()
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        raise ValueError(f"{snapshot_kind}.items must be a sequence.")
    loaded: list[_ItemT] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"{snapshot_kind}.items[{index}] must be a mapping.")
        loaded.append(item_factory(item))
    return tuple(loaded)


def _save_list_snapshot(
    *,
    config: TwinrConfig,
    remote_state: LongTermRemoteStateStore | None,
    snapshot_kind: str,
    items: Sequence[object],
    item_serializer: Callable[[object], Mapping[str, object]],
) -> None:
    """Save a typed list snapshot through the remote snapshot adapter."""

    resolved_remote_state = _resolve_remote_state(config=config, remote_state=remote_state)
    if resolved_remote_state is None:
        return
    payload = {
        "schema_version": 1,
        "items": [dict(item_serializer(item)) for item in items],
    }
    resolved_remote_state.save_snapshot(snapshot_kind=snapshot_kind, payload=payload)


class PersonalitySnapshotStore(Protocol):
    """Describe a loader/saver for the promptable personality snapshot."""

    def load_snapshot(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> PersonalitySnapshot | None:
        """Load the latest personality snapshot for prompt assembly."""

    def save_snapshot(
        self,
        *,
        config: TwinrConfig,
        snapshot: PersonalitySnapshot,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist the latest promptable personality snapshot."""


@dataclass(slots=True)
class RemoteStatePersonalitySnapshotStore:
    """Load and save structured personality state via remote snapshots."""

    snapshot_kind: str = DEFAULT_PERSONALITY_SNAPSHOT_KIND

    def load_snapshot(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> PersonalitySnapshot | None:
        """Load a typed snapshot from remote state when one exists."""

        resolved_remote_state = _resolve_remote_state(config=config, remote_state=remote_state)
        if resolved_remote_state is None:
            return None
        payload = resolved_remote_state.load_snapshot(snapshot_kind=self.snapshot_kind)
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.snapshot_kind} must decode to a mapping payload.")
        return PersonalitySnapshot.from_payload(payload)

    def save_snapshot(
        self,
        *,
        config: TwinrConfig,
        snapshot: PersonalitySnapshot,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist a typed snapshot through remote-primary state."""

        resolved_remote_state = _resolve_remote_state(config=config, remote_state=remote_state)
        if resolved_remote_state is None:
            return
        resolved_remote_state.save_snapshot(
            snapshot_kind=self.snapshot_kind,
            payload=snapshot.to_payload(),
        )


class PersonalityEvolutionStore(Protocol):
    """Describe persistence for learning signals and accepted deltas."""

    def load_interaction_signals(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[InteractionSignal, ...]:
        """Load persisted interaction signals."""

    def save_interaction_signals(
        self,
        *,
        config: TwinrConfig,
        signals: Sequence[InteractionSignal],
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist interaction signals."""

    def load_place_signals(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[PlaceSignal, ...]:
        """Load persisted place signals."""

    def save_place_signals(
        self,
        *,
        config: TwinrConfig,
        signals: Sequence[PlaceSignal],
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist place signals."""

    def load_world_signals(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[WorldSignal, ...]:
        """Load persisted world signals."""

    def save_world_signals(
        self,
        *,
        config: TwinrConfig,
        signals: Sequence[WorldSignal],
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist world signals."""

    def load_personality_deltas(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[PersonalityDelta, ...]:
        """Load accepted or rejected personality deltas."""

    def save_personality_deltas(
        self,
        *,
        config: TwinrConfig,
        deltas: Sequence[PersonalityDelta],
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist personality deltas."""


@dataclass(slots=True)
class RemoteStatePersonalityEvolutionStore:
    """Persist learning signals and deltas through remote-primary snapshots."""

    interaction_snapshot_kind: str = INTERACTION_SIGNAL_SNAPSHOT_KIND
    place_snapshot_kind: str = PLACE_SIGNAL_SNAPSHOT_KIND
    world_snapshot_kind: str = WORLD_SIGNAL_SNAPSHOT_KIND
    delta_snapshot_kind: str = PERSONALITY_DELTA_SNAPSHOT_KIND

    def load_interaction_signals(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[InteractionSignal, ...]:
        """Load persisted interaction signals."""

        return _load_list_snapshot(
            config=config,
            remote_state=remote_state,
            snapshot_kind=self.interaction_snapshot_kind,
            item_factory=InteractionSignal.from_payload,
        )

    def save_interaction_signals(
        self,
        *,
        config: TwinrConfig,
        signals: Sequence[InteractionSignal],
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist interaction signals."""

        _save_list_snapshot(
            config=config,
            remote_state=remote_state,
            snapshot_kind=self.interaction_snapshot_kind,
            items=signals,
            item_serializer=lambda item: item.to_payload(),
        )

    def load_place_signals(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[PlaceSignal, ...]:
        """Load persisted place signals."""

        return _load_list_snapshot(
            config=config,
            remote_state=remote_state,
            snapshot_kind=self.place_snapshot_kind,
            item_factory=PlaceSignal.from_payload,
        )

    def save_place_signals(
        self,
        *,
        config: TwinrConfig,
        signals: Sequence[PlaceSignal],
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist place signals."""

        _save_list_snapshot(
            config=config,
            remote_state=remote_state,
            snapshot_kind=self.place_snapshot_kind,
            items=signals,
            item_serializer=lambda item: item.to_payload(),
        )

    def load_world_signals(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[WorldSignal, ...]:
        """Load persisted world signals."""

        return _load_list_snapshot(
            config=config,
            remote_state=remote_state,
            snapshot_kind=self.world_snapshot_kind,
            item_factory=WorldSignal.from_payload,
        )

    def save_world_signals(
        self,
        *,
        config: TwinrConfig,
        signals: Sequence[WorldSignal],
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist world signals."""

        _save_list_snapshot(
            config=config,
            remote_state=remote_state,
            snapshot_kind=self.world_snapshot_kind,
            items=signals,
            item_serializer=lambda item: item.to_payload(),
        )

    def load_personality_deltas(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[PersonalityDelta, ...]:
        """Load persisted personality deltas."""

        return _load_list_snapshot(
            config=config,
            remote_state=remote_state,
            snapshot_kind=self.delta_snapshot_kind,
            item_factory=PersonalityDelta.from_payload,
        )

    def save_personality_deltas(
        self,
        *,
        config: TwinrConfig,
        deltas: Sequence[PersonalityDelta],
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> None:
        """Persist personality deltas."""

        _save_list_snapshot(
            config=config,
            remote_state=remote_state,
            snapshot_kind=self.delta_snapshot_kind,
            items=deltas,
            item_serializer=lambda item: item.to_payload(),
        )
