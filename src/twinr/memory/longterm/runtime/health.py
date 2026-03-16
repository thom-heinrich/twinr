"""Probe remote long-term snapshots before runtime uses them.

This module verifies that every required remote-primary snapshot and shard can
be loaded before Twinr enters runtime flows that depend on long-term memory.
It does not mutate runtime state; it only proves readiness or raises a hard
failure for required remote backends.
"""

from __future__ import annotations

from dataclasses import dataclass

from twinr.memory.context_store import PromptContextStore
from twinr.memory.chonkydb.personal_graph import TwinrPersonalGraphStore
from twinr.memory.longterm.storage.midterm_store import LongTermMidtermStore
from twinr.memory.longterm.storage.remote_state import (
    LongTermRemoteStateStore,
    LongTermRemoteUnavailableError,
)
from twinr.memory.longterm.storage.store import LongTermStructuredStore


@dataclass(frozen=True, slots=True)
class LongTermRemoteWarmResult:
    """Capture which remote snapshots were checked during a warm probe."""

    checked_snapshots: tuple[str, ...]


@dataclass(slots=True)
class LongTermRemoteHealthProbe:
    """Verify that required remote-primary long-term snapshots are readable.

    The runtime service uses this probe after store-level readiness checks to
    ensure every required snapshot kind and shard can actually be loaded.
    """

    prompt_context_store: PromptContextStore
    object_store: LongTermStructuredStore
    graph_store: TwinrPersonalGraphStore
    midterm_store: LongTermMidtermStore

    def ensure_operational(self) -> LongTermRemoteWarmResult:
        """Load every required remote snapshot and shard once.

        Returns:
            A record of the snapshot kinds that were checked successfully.

        Raises:
            LongTermRemoteUnavailableError: If any required remote state,
                snapshot, or shard is missing or unreadable.
        """

        checked: list[str] = []

        prompt_remote_state = self._require_remote_state(self.prompt_context_store.memory_store.remote_state)
        self._ensure_state_ready(prompt_remote_state)
        self._ensure_snapshot(prompt_remote_state, self.prompt_context_store.memory_store.remote_snapshot_kind, checked)
        self._ensure_snapshot(prompt_remote_state, self.prompt_context_store.user_store.remote_snapshot_kind, checked)
        self._ensure_snapshot(
            prompt_remote_state,
            self.prompt_context_store.personality_store.remote_snapshot_kind,
            checked,
        )

        object_remote_state = self._require_remote_state(self.object_store.remote_state)
        self._ensure_state_ready(object_remote_state)
        self._ensure_sharded_snapshot_tree(object_remote_state, "objects", checked)
        self._ensure_snapshot(object_remote_state, "conflicts", checked)
        self._ensure_sharded_snapshot_tree(object_remote_state, "archive", checked)

        graph_remote_state = self._require_remote_state(self.graph_store.remote_state)
        self._ensure_state_ready(graph_remote_state)
        self._ensure_snapshot(graph_remote_state, "graph", checked)

        midterm_remote_state = self._require_remote_state(self.midterm_store.remote_state)
        self._ensure_state_ready(midterm_remote_state)
        self._ensure_snapshot(midterm_remote_state, "midterm", checked)

        return LongTermRemoteWarmResult(checked_snapshots=tuple(checked))

    def _ensure_sharded_snapshot_tree(
        self,
        remote_state: LongTermRemoteStateStore,
        snapshot_kind: str,
        checked: list[str],
    ) -> None:
        """Load a manifest snapshot and then every shard it references."""

        payload = self._load_required_snapshot(remote_state, snapshot_kind, checked)
        shards = payload.get("shards")
        if not isinstance(shards, list):
            return
        for shard_kind in shards:
            if isinstance(shard_kind, str) and shard_kind:
                self._load_required_snapshot(remote_state, shard_kind, checked)

    def _ensure_snapshot(
        self,
        remote_state: LongTermRemoteStateStore,
        snapshot_kind: str | None,
        checked: list[str],
    ) -> None:
        """Require one named snapshot kind and record the successful load."""

        normalized_kind = str(snapshot_kind or "").strip()
        if not normalized_kind:
            raise LongTermRemoteUnavailableError("Required remote long-term snapshot kind is missing.")
        self._load_required_snapshot(remote_state, normalized_kind, checked)

    def _load_required_snapshot(
        self,
        remote_state: LongTermRemoteStateStore,
        snapshot_kind: str,
        checked: list[str],
    ) -> dict[str, object]:
        """Load one snapshot and fail closed if it is missing or malformed."""

        payload = remote_state.load_snapshot(snapshot_kind=snapshot_kind)
        if not isinstance(payload, dict):
            raise LongTermRemoteUnavailableError(
                f"Required remote long-term snapshot {snapshot_kind!r} is unavailable."
            )
        checked.append(snapshot_kind)
        return payload

    def _ensure_state_ready(self, remote_state: LongTermRemoteStateStore) -> None:
        """Require the remote state backend to report itself ready."""

        status = remote_state.status()
        if status.ready:
            return
        raise LongTermRemoteUnavailableError(
            str(status.detail or "Required remote long-term memory is unavailable.")
        )

    def _require_remote_state(self, remote_state: LongTermRemoteStateStore | None) -> LongTermRemoteStateStore:
        """Require an enabled remote state adapter before probing snapshots."""

        if remote_state is None or not remote_state.enabled:
            raise LongTermRemoteUnavailableError("Required remote long-term memory state is not configured.")
        return remote_state
