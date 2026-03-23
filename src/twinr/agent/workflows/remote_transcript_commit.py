"""Coordinate same-stream server transcript commits for remote listen windows.

The realtime loop can keep one conversation session open while the Pi streams
audio continuously to the remote voice gateway. This helper tracks the bounded
wait for a server-side transcript commit so the live loop does not have to
embed thread coordination details into `realtime_runner.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event, Lock


@dataclass(frozen=True, slots=True)
class RemoteTranscriptCommit:
    """Represent one server-side transcript commit for the live stream."""

    source: str
    transcript: str


@dataclass(frozen=True, slots=True)
class RemoteTranscriptClose:
    """Represent one server-side close signal for a waiting transcript window."""

    source: str
    reason: str


@dataclass(slots=True)
class RemoteTranscriptWaitHandle:
    """Track one active wait for a server-side transcript commit."""

    source: str
    _event: Event = field(default_factory=Event, repr=False)
    commit: RemoteTranscriptCommit | None = None
    close: RemoteTranscriptClose | None = None

    def wait(self, timeout_s: float | None = None) -> bool:
        """Block until a commit/close arrives or the timeout expires."""

        return self._event.wait(timeout_s)

    def set_commit(self, commit: RemoteTranscriptCommit) -> None:
        """Resolve this wait with a committed transcript."""

        self.commit = commit
        self._event.set()

    def set_close(self, close: RemoteTranscriptClose) -> None:
        """Resolve this wait with a close signal instead of a transcript."""

        self.close = close
        self._event.set()


class RemoteTranscriptCommitCoordinator:
    """Serialize one active remote transcript wait against websocket callbacks."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._active_wait: RemoteTranscriptWaitHandle | None = None

    def begin_wait(self, *, source: str) -> RemoteTranscriptWaitHandle:
        """Open one active wait for a transcript commit from the given source."""

        with self._lock:
            active_wait = self._active_wait
            if active_wait is not None and not active_wait._event.is_set():
                raise RuntimeError("Remote transcript wait already active")
            handle = RemoteTranscriptWaitHandle(source=source)
            self._active_wait = handle
            return handle

    def clear_wait(self, handle: RemoteTranscriptWaitHandle) -> None:
        """Drop one completed/abandoned wait if it is still the active one."""

        with self._lock:
            if self._active_wait is handle:
                self._active_wait = None

    def commit(self, *, source: str, transcript: str) -> bool:
        """Resolve the active wait if it expects the given transcript source."""

        with self._lock:
            handle = self._active_wait
            if handle is None or handle._event.is_set() or handle.source != source:
                return False
            handle.set_commit(RemoteTranscriptCommit(source=source, transcript=transcript))
            return True

    def close(self, *, source: str, reason: str) -> bool:
        """Resolve the active wait if the server closed the matching source."""

        with self._lock:
            handle = self._active_wait
            if handle is None or handle._event.is_set() or handle.source != source:
                return False
            handle.set_close(RemoteTranscriptClose(source=source, reason=reason))
            return True

