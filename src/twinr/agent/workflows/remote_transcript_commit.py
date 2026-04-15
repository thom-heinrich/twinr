# CHANGELOG: 2026-03-28
# BUG-1: Positive-timeout waits now auto-abandon and remove themselves from the coordinator,
#        so a timeout/exception path can no longer wedge future transcript waits.
# BUG-2: Wait resolution is no longer keyed only by `source`; waits can now be correlated by
#        generated `wait_id` and/or server `item_id` to prevent stale same-stream commits from
#        resolving the wrong listen window.
# SEC-1: Unsafe source-only matching is disabled by default because late/replayed same-stream
#        events can spoof the current wait. Legacy behavior is opt-in via
#        `allow_legacy_source_match=True`.
# IMP-1: Upgraded from one global active wait to a bounded multi-wait coordinator so pipelined
#        / asynchronous transcript windows can coexist safely.
# IMP-2: Added per-handle state locking, explicit abandonment state, wake-on-cancel, monotonic
#        timing, and richer result metadata (`wait_id`, `item_id`) aligned with 2026 realtime
#        voice APIs.

"""Coordinate remote transcript commits for same-stream realtime audio windows.

Why this exists:
- Modern realtime voice APIs expose a stable per-item identifier (`item_id`) after an
  audio buffer is committed, and later transcription events reference that same item.
- Server-side transcription is asynchronous and can arrive after the live loop has already
  moved on to a newer listen window.
- On edge deployments, a missed timeout/cleanup path must not wedge the audio loop.

Recommended 2026 usage pattern:
    handle = coordinator.begin_wait(source="remote_input")

    # When the server confirms the input audio commit / speech window:
    coordinator.bind_item(handle, item_id=server_item_id)

    # When the final transcription arrives:
    coordinator.commit(item_id=server_item_id, transcript=final_text)

Compatibility:
- Safe matching is by `wait_id` or `item_id`.
- Source-only matching is intentionally disabled by default because it is unsafe against
  stale/replayed same-stream events.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from threading import Event, Lock
from time import monotonic
from typing import Final
from uuid import uuid4

_PENDING: Final = "pending"
_COMMITTED: Final = "committed"
_CLOSED: Final = "closed"
_ABANDONED: Final = "abandoned"


@dataclass(frozen=True, slots=True)
class RemoteTranscriptCommit:
    """Represent one server-side transcript commit for a live stream window."""

    source: str
    transcript: str
    wait_id: str
    item_id: str | None = None


@dataclass(frozen=True, slots=True)
class RemoteTranscriptClose:
    """Represent one server-side close/cancel signal for a live stream window."""

    source: str
    reason: str
    wait_id: str
    item_id: str | None = None


@dataclass(slots=True)
class RemoteTranscriptWaitHandle:
    """Track one active wait for a server-side transcript commit."""

    source: str
    wait_id: str
    item_id: str | None = None
    opened_monotonic: float = field(default_factory=monotonic)
    _event: Event = field(default_factory=Event, repr=False)
    _state_lock: Lock = field(default_factory=Lock, repr=False)
    _coordinator: "RemoteTranscriptCommitCoordinator | None" = field(default=None, repr=False)
    _state: str = field(default=_PENDING, init=False, repr=False)
    _resolved_monotonic: float | None = field(default=None, init=False, repr=False)
    _abandon_reason: str | None = field(default=None, init=False, repr=False)
    commit: RemoteTranscriptCommit | None = None
    close: RemoteTranscriptClose | None = None

    def wait(self, timeout_s: float | None = None) -> bool:
        """Block until commit/close arrives or the timeout expires.

        Return True only when the handle resolved to a commit or close.
        Return False on timeout or explicit abandonment/cancellation.

        # BREAKING:
        A *positive* timeout now auto-abandons the handle and removes it from the
        coordinator. This prevents a timed-out wait from blocking future listen
        windows forever. Use `wait(0)` if you need a non-destructive poll.
        """

        signaled = self._event.wait(timeout_s)
        if signaled:
            return self._is_resolution_state()

        if timeout_s is not None and timeout_s > 0:
            coordinator = self._coordinator
            if coordinator is not None:
                coordinator._abandon_wait(self, reason="timeout")
            return self._is_resolution_state()

        return False

    @property
    def state(self) -> str:
        """Return one of: pending, committed, closed, abandoned."""

        with self._state_lock:
            return self._state

    @property
    def abandon_reason(self) -> str | None:
        """Return the abandonment reason for an abandoned handle."""

        with self._state_lock:
            return self._abandon_reason

    @property
    def resolved_monotonic(self) -> float | None:
        """Return the monotonic timestamp when the handle left the pending state."""

        with self._state_lock:
            return self._resolved_monotonic

    def done(self) -> bool:
        """Return whether the handle is no longer pending."""

        with self._state_lock:
            return self._state != _PENDING

    def result(self) -> RemoteTranscriptCommit | RemoteTranscriptClose | None:
        """Return the final commit/close object if present."""

        with self._state_lock:
            return self.commit if self.commit is not None else self.close

    def set_commit(self, commit: RemoteTranscriptCommit) -> None:
        """Resolve this wait with a committed transcript.

        Prefer resolving through the coordinator so internal indexes are updated.
        """

        if not self._try_set_commit(commit):
            raise RuntimeError("Remote transcript wait already resolved or abandoned")

    def set_close(self, close: RemoteTranscriptClose) -> None:
        """Resolve this wait with a close signal.

        Prefer resolving through the coordinator so internal indexes are updated.
        """

        if not self._try_set_close(close):
            raise RuntimeError("Remote transcript wait already resolved or abandoned")

    def _is_resolution_state(self) -> bool:
        with self._state_lock:
            return self._state in (_COMMITTED, _CLOSED)

    def _try_bind_item(self, item_id: str) -> bool:
        if not item_id:
            raise ValueError("item_id must be non-empty")

        with self._state_lock:
            if self._state != _PENDING:
                return False
            if self.item_id is None:
                self.item_id = item_id
                return True
            return self.item_id == item_id

    def _try_set_commit(self, commit: RemoteTranscriptCommit) -> bool:
        with self._state_lock:
            if self._state != _PENDING:
                return False
            self.commit = commit
            self.close = None
            self._state = _COMMITTED
            self._resolved_monotonic = monotonic()

        self._event.set()
        return True

    def _try_set_close(self, close: RemoteTranscriptClose) -> bool:
        with self._state_lock:
            if self._state != _PENDING:
                return False
            self.close = close
            self.commit = None
            self._state = _CLOSED
            self._resolved_monotonic = monotonic()

        self._event.set()
        return True

    def _try_abandon(self, *, reason: str) -> bool:
        with self._state_lock:
            if self._state != _PENDING:
                return False
            self._state = _ABANDONED
            self._abandon_reason = reason
            self._resolved_monotonic = monotonic()

        self._event.set()
        return True


class RemoteTranscriptCommitCoordinator:
    """Coordinate transcript waits against asynchronous websocket callbacks.

    Frontier behavior:
    - Multiple concurrent waits are supported.
    - Strong correlation is by `wait_id` and/or `item_id`.
    - Source-only matching is an explicit unsafe compatibility mode.
    """

    def __init__(
        self,
        *,
        max_active_waits: int = 16,
        allow_legacy_source_match: bool = False,
    ) -> None:
        if max_active_waits < 1:
            raise ValueError("max_active_waits must be >= 1")

        self._lock = Lock()
        self._waits: dict[str, RemoteTranscriptWaitHandle] = {}
        self._source_index: dict[str, set[str]] = defaultdict(set)
        self._item_index: dict[str, str] = {}
        self._max_active_waits = max_active_waits

        # BREAKING:
        # Source-only commit/close matching is disabled by default because it is
        # unsafe against stale/replayed same-stream events. Re-enable only if you
        # knowingly accept that integrity risk.
        self._allow_legacy_source_match = allow_legacy_source_match

    def begin_wait(
        self,
        *,
        source: str,
        wait_id: str | None = None,
        item_id: str | None = None,
    ) -> RemoteTranscriptWaitHandle:
        """Open a new transcript wait.

        `wait_id` is optional. If omitted, a unique ID is generated locally.
        `item_id` can be pre-bound if the caller already knows the server item.
        """

        if wait_id is None:
            wait_id = uuid4().hex
        if not wait_id:
            raise ValueError("wait_id must be non-empty")
        if item_id == "":
            raise ValueError("item_id must be non-empty when provided")

        with self._lock:
            if wait_id in self._waits:
                raise RuntimeError(f"Remote transcript wait_id already active: {wait_id}")
            if item_id is not None and item_id in self._item_index:
                raise RuntimeError(f"Remote transcript item_id already bound: {item_id}")
            if len(self._waits) >= self._max_active_waits:
                raise RuntimeError("Too many active remote transcript waits")

            handle = RemoteTranscriptWaitHandle(
                source=source,
                wait_id=wait_id,
                item_id=item_id,
                _coordinator=self,
            )
            self._waits[wait_id] = handle
            self._source_index[source].add(wait_id)
            if item_id is not None:
                self._item_index[item_id] = wait_id
            return handle

    def clear_wait(self, handle: RemoteTranscriptWaitHandle) -> None:
        """Drop one completed/abandoned wait if it is still active."""

        self._abandon_wait(handle, reason="cleared")

    def cancel_wait(
        self,
        handle: RemoteTranscriptWaitHandle,
        *,
        reason: str = "cancelled",
    ) -> bool:
        """Abandon one wait explicitly."""

        return self._abandon_wait(handle, reason=reason)

    def bind_item(self, handle: RemoteTranscriptWaitHandle, *, item_id: str) -> bool:
        """Bind a server `item_id` to an existing wait handle."""

        if not item_id:
            raise ValueError("item_id must be non-empty")

        with self._lock:
            current = self._waits.get(handle.wait_id)
            if current is not handle:
                return False
            return self._bind_item_unlocked(handle, item_id=item_id)

    def bind_item_id(self, *, wait_id: str, item_id: str) -> bool:
        """Bind a server `item_id` to an active wait by wait_id."""

        if not item_id:
            raise ValueError("item_id must be non-empty")

        with self._lock:
            handle = self._waits.get(wait_id)
            if handle is None:
                return False
            return self._bind_item_unlocked(handle, item_id=item_id)

    def has_wait(self, *, wait_id: str) -> bool:
        """Return whether one wait_id is still active inside the coordinator."""

        if not wait_id:
            raise ValueError("wait_id must be non-empty")

        with self._lock:
            return wait_id in self._waits

    def commit(
        self,
        *,
        transcript: str,
        source: str | None = None,
        wait_id: str | None = None,
        item_id: str | None = None,
    ) -> bool:
        """Resolve a wait with a committed transcript.

        Safe matching is by `wait_id` or `item_id`.
        Source-only matching works only when:
        - `allow_legacy_source_match=True`, and
        - exactly one pending wait exists for that source.
        """

        with self._lock:
            handle = self._lookup_unlocked(source=source, wait_id=wait_id, item_id=item_id)
            if handle is None:
                return False
            if source is not None and handle.source != source:
                return False
            if item_id is not None and handle.item_id != item_id:
                return False

            commit = RemoteTranscriptCommit(
                source=handle.source,
                transcript=transcript,
                wait_id=handle.wait_id,
                item_id=handle.item_id,
            )
            if not handle._try_set_commit(commit):
                return False

            self._remove_unlocked(handle)
            return True

    def close(
        self,
        *,
        reason: str,
        source: str | None = None,
        wait_id: str | None = None,
        item_id: str | None = None,
    ) -> bool:
        """Resolve a wait with a close signal instead of a transcript."""

        with self._lock:
            handle = self._lookup_unlocked(source=source, wait_id=wait_id, item_id=item_id)
            if handle is None:
                return False
            if source is not None and handle.source != source:
                return False
            if item_id is not None and handle.item_id != item_id:
                return False

            close = RemoteTranscriptClose(
                source=handle.source,
                reason=reason,
                wait_id=handle.wait_id,
                item_id=handle.item_id,
            )
            if not handle._try_set_close(close):
                return False

            self._remove_unlocked(handle)
            return True

    def pending_count(self) -> int:
        """Return the number of currently pending waits."""

        with self._lock:
            return len(self._waits)

    def _lookup_unlocked(
        self,
        *,
        source: str | None,
        wait_id: str | None,
        item_id: str | None,
    ) -> RemoteTranscriptWaitHandle | None:
        if wait_id is not None:
            return self._waits.get(wait_id)

        if item_id is not None:
            bound_wait_id = self._item_index.get(item_id)
            if bound_wait_id is None:
                return None
            return self._waits.get(bound_wait_id)

        if source is not None and self._allow_legacy_source_match:
            wait_ids = self._source_index.get(source)
            if not wait_ids or len(wait_ids) != 1:
                return None
            return self._waits.get(next(iter(wait_ids)))

        return None

    def _bind_item_unlocked(self, handle: RemoteTranscriptWaitHandle, *, item_id: str) -> bool:
        existing_wait_id = self._item_index.get(item_id)
        if existing_wait_id is not None and existing_wait_id != handle.wait_id:
            return False
        if not handle._try_bind_item(item_id):
            return False

        self._item_index[item_id] = handle.wait_id
        return True

    def _abandon_wait(self, handle: RemoteTranscriptWaitHandle, *, reason: str) -> bool:
        with self._lock:
            current = self._waits.get(handle.wait_id)
            if current is not handle:
                return False
            if not handle._try_abandon(reason=reason):
                return False

            self._remove_unlocked(handle)
            return True

    def _remove_unlocked(self, handle: RemoteTranscriptWaitHandle) -> None:
        wait_id = handle.wait_id
        self._waits.pop(wait_id, None)

        wait_ids = self._source_index.get(handle.source)
        if wait_ids is not None:
            wait_ids.discard(wait_id)
            if not wait_ids:
                self._source_index.pop(handle.source, None)

        item_id = handle.item_id
        if item_id is not None and self._item_index.get(item_id) == wait_id:
            self._item_index.pop(item_id, None)

        handle._coordinator = None
