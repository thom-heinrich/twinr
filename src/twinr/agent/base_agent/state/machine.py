# CHANGELOG: 2026-03-27
# BUG-1: Replaced the flat exclusive PRINTING state model with an orthogonal
#        printing activity so valid event orders like
#        PRINT_REQUESTED -> RESPONSE_READY -> PRINT_FINISHED no longer break.
# BUG-2: FAILURE routed through transition() no longer destroys the root cause;
#        transition(..., error=...) now preserves diagnostic context.
# BUG-3: Added epoch-gated stale-event suppression for async callbacks so late
#        worker completions after RESET / state changes do not derail runtime.
# BUG-4: Accept delayed follow-up reopen events from `waiting` as well as
#        `answering`, so the runtime can stop claiming `speaking` immediately
#        after audio drains while still allowing a later gated reopen to
#        `listening`.
# SEC-1: Normalized and size-capped error/source payloads to reduce practical
#        log-injection / terminal-control / resource-exhaustion risk on edge
#        deployments.
# IMP-1: Replaced the module-global lock with per-instance locking.
# IMP-2: Added structured audit records, schema-versioned snapshots, and
#        orthogonal-status introspection for 2026-grade observability.
#
# BREAKING: PRINT_REQUESTED / YELLOW_BUTTON_PRESSED during PROCESSING or
# ANSWERING no longer steal the primary status; they only activate
# printing_active. Read active_statuses for the full configuration.
# BREAKING: Async producers should pass expected_epoch=machine.epoch when
# completing work. Without that token, stale completions from old workers
# cannot be distinguished from current work.

"""Define deterministic runtime status transitions for the base Twinr agent.

This module exports the canonical status and event enums plus a concurrency-safe
runtime state machine for Twinr orchestration.

Frontier upgrades in this revision:
- orthogonal printing activity instead of a flat mutually-exclusive PRINTING
  state only;
- epoch-gated stale-event rejection for async worker callbacks;
- structured audit records with timestamps and sequence numbers;
- schema-versioned snapshot import/export for durable persistence.

Import these definitions from the package root when coordinating runtime
behavior or consuming persisted status values elsewhere in Twinr.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from threading import RLock
from time import monotonic_ns
from typing import cast


class TwinrStatus(StrEnum):
    """Enumerate the canonical top-level Twinr runtime states."""

    WAITING = "waiting"
    LISTENING = "listening"
    PROCESSING = "processing"
    ANSWERING = "answering"
    PRINTING = "printing"
    ERROR = "error"


class TwinrEvent(StrEnum):
    """Enumerate the canonical events that drive runtime transitions."""

    GREEN_BUTTON_PRESSED = "green_button_pressed"
    FOLLOW_UP_ARMED = "follow_up_armed"
    SPEECH_PAUSE_DETECTED = "speech_pause_detected"
    PROCESSING_RESUMED = "processing_resumed"
    RESPONSE_READY = "response_ready"
    PROACTIVE_PROMPT_READY = "proactive_prompt_ready"
    TTS_FINISHED = "tts_finished"
    YELLOW_BUTTON_PRESSED = "yellow_button_pressed"
    PRINT_REQUESTED = "print_requested"
    PRINT_FINISHED = "print_finished"
    FAILURE = "failure"
    RESET = "reset"


class InvalidTransitionError(RuntimeError):
    """Raise when a requested state transition is not allowed."""


class _IgnoredEvent(RuntimeError):
    """Internal control-flow exception for safe no-op async events."""

    def __init__(self, note: str) -> None:
        super().__init__(note)
        self.note = note


_HISTORY_LIMIT = 256
_AUDIT_LIMIT = 512
_DEFAULT_FAILURE_MESSAGE = "Unknown failure"
_MAX_ERROR_LENGTH = 2048
_MAX_SOURCE_LENGTH = 128
_SNAPSHOT_SCHEMA_VERSION = 2

_PRINT_START_EVENTS = frozenset({TwinrEvent.YELLOW_BUTTON_PRESSED, TwinrEvent.PRINT_REQUESTED})

# Async events that are produced by workers watching a specific source state.
# If they arrive after the machine has already moved on, they are safely
# ignored instead of raising and destabilizing runtime recovery.
_ASYNC_SOURCE_STATES: dict[TwinrEvent, frozenset[TwinrStatus]] = {
    TwinrEvent.SPEECH_PAUSE_DETECTED: frozenset({TwinrStatus.LISTENING}),
    TwinrEvent.RESPONSE_READY: frozenset({TwinrStatus.PROCESSING}),
    TwinrEvent.PROCESSING_RESUMED: frozenset({TwinrStatus.ANSWERING}),
    TwinrEvent.FOLLOW_UP_ARMED: frozenset({TwinrStatus.ANSWERING, TwinrStatus.WAITING}),
    TwinrEvent.TTS_FINISHED: frozenset({TwinrStatus.ANSWERING}),
}

# Base transitions for the primary status axis. Printing is handled as an
# orthogonal activity in the state machine methods below.
_BASE_TRANSITIONS: dict[TwinrStatus, dict[TwinrEvent, TwinrStatus]] = {
    TwinrStatus.WAITING: {
        TwinrEvent.GREEN_BUTTON_PRESSED: TwinrStatus.LISTENING,
        TwinrEvent.FOLLOW_UP_ARMED: TwinrStatus.LISTENING,
        TwinrEvent.PROACTIVE_PROMPT_READY: TwinrStatus.ANSWERING,
    },
    TwinrStatus.LISTENING: {
        TwinrEvent.SPEECH_PAUSE_DETECTED: TwinrStatus.PROCESSING,
    },
    TwinrStatus.PROCESSING: {
        TwinrEvent.RESPONSE_READY: TwinrStatus.ANSWERING,
    },
    TwinrStatus.ANSWERING: {
        TwinrEvent.PROCESSING_RESUMED: TwinrStatus.PROCESSING,
        TwinrEvent.FOLLOW_UP_ARMED: TwinrStatus.LISTENING,
    },
    # BREAKING: PRINTING now means "print is the only active foreground mode".
    # While printing is active concurrently with ANSWERING / PROCESSING,
    # `status` stays ANSWERING / PROCESSING and `printing_active` is True.
    TwinrStatus.PRINTING: {
        TwinrEvent.GREEN_BUTTON_PRESSED: TwinrStatus.LISTENING,
        TwinrEvent.PROACTIVE_PROMPT_READY: TwinrStatus.ANSWERING,
    },
    TwinrStatus.ERROR: {},
}


def _utc_now_iso() -> str:
    """Return a UTC timestamp suitable for audit records."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _coerce_status(value: TwinrStatus | str) -> TwinrStatus:
    """Coerce a raw status value to ``TwinrStatus``."""

    if isinstance(value, TwinrStatus):
        return value
    if isinstance(value, str):
        try:
            return TwinrStatus(value)
        except ValueError as exc:
            raise ValueError(f"Unknown status: {value!r}") from exc
    raise TypeError(f"status must be TwinrStatus or str, got {type(value).__name__}")


def _coerce_event(value: TwinrEvent | str) -> TwinrEvent:
    """Coerce a raw event value to ``TwinrEvent``."""

    if isinstance(value, TwinrEvent):
        return value
    if isinstance(value, str):
        try:
            return TwinrEvent(value)
        except ValueError as exc:
            raise InvalidTransitionError(f"Unknown event: {value!r}") from exc
    raise InvalidTransitionError(f"event must be TwinrEvent or str, got {type(value).__name__}")


def _safe_status_for_history(value: TwinrStatus | str | object) -> TwinrStatus:
    """Return a history-safe status value even for corrupted input."""

    try:
        return _coerce_status(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return TwinrStatus.ERROR


def _coerce_non_negative_int(value: object) -> int:
    """Best-effort coerce arbitrary restored values to a non-negative int."""

    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, str):
        try:
            return max(0, int(value.strip()))
        except ValueError:
            return 0
    return 0


def _neutralize_control_chars(text: str) -> str:
    """Escape control characters so diagnostics are log/terminal-safe."""

    parts: list[str] = []
    for char in text:
        code = ord(char)
        if char in ("\n", "\r", "\t"):
            parts.append(char.encode("unicode_escape").decode("ascii"))
        elif code < 32 or 127 <= code <= 159:
            parts.append(f"\\x{code:02x}")
        else:
            parts.append(char)
    return "".join(parts)


def _normalize_error_message(message: object) -> str:
    """Normalize arbitrary failure payloads to a bounded error string."""

    if message is None:
        text = _DEFAULT_FAILURE_MESSAGE
    else:
        text = str(message).strip() or _DEFAULT_FAILURE_MESSAGE
    text = _neutralize_control_chars(text)
    if len(text) > _MAX_ERROR_LENGTH:
        text = f"{text[:_MAX_ERROR_LENGTH - 1]}…"
    return text or _DEFAULT_FAILURE_MESSAGE


def _normalize_source(source: object) -> str | None:
    """Normalize an optional event source label for audit records."""

    if source is None:
        return None
    text = _neutralize_control_chars(str(source).strip())
    if not text:
        return None
    if len(text) > _MAX_SOURCE_LENGTH:
        return f"{text[:_MAX_SOURCE_LENGTH - 1]}…"
    return text


@dataclass(slots=True, frozen=True)
class TwinrTransitionRecord:
    """Structured transition/audit record for observability and persistence."""

    sequence: int
    epoch_before: int
    epoch_after: int
    occurred_at: str
    monotonic_time_ns: int
    previous: TwinrStatus
    event: TwinrEvent
    next: TwinrStatus
    printing_active: bool
    source: str | None = None
    error: str | None = None
    ignored: bool = False
    note: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the audit record."""

        return {
            "sequence": self.sequence,
            "epoch_before": self.epoch_before,
            "epoch_after": self.epoch_after,
            "occurred_at": self.occurred_at,
            "monotonic_time_ns": self.monotonic_time_ns,
            "previous": self.previous.value,
            "event": self.event.value,
            "next": self.next.value,
            "printing_active": self.printing_active,
            "source": self.source,
            "error": self.error,
            "ignored": self.ignored,
            "note": self.note,
        }


class TwinrStateMachine:
    """Track Twinr runtime status and enforce legal transitions.

    The machine supports persisted string state values, bounded legacy history,
    structured audit logs, and epoch-based async callback suppression.

    Attributes:
        status: Canonical top-level status. When printing overlaps with another
            foreground activity, this remains the foreground status and
            ``printing_active`` becomes the secondary orthogonal activity flag.
        printing_active: Whether a print job is currently active.
        last_error: Last normalized failure message while in ``ERROR``.
        epoch: Monotonic token that changes on every accepted transition.
            Async workers should capture it when work starts and provide it back
            via ``expected_epoch=...`` when signaling completion.
        sequence: Monotonic audit record counter.
        history: Bounded legacy list of ``(previous, event, next)`` tuples.
        audit: Bounded structured audit records.
    """

    __slots__ = (
        "_status",
        "last_error",
        "printing_active",
        "epoch",
        "sequence",
        "_history",
        "_audit",
        "_lock",
    )

    def __init__(
        self,
        status: TwinrStatus | str = TwinrStatus.WAITING,
        last_error: str | None = None,
        history: Iterable[object] = (),
        audit: Iterable[object] = (),
        printing_active: bool = False,
        epoch: int = 0,
        sequence: int = 0,
    ) -> None:
        self._lock = RLock()
        self._history: deque[tuple[TwinrStatus, TwinrEvent, TwinrStatus]] = deque(
            maxlen=_HISTORY_LIMIT
        )
        self._audit: deque[TwinrTransitionRecord] = deque(maxlen=_AUDIT_LIMIT)
        self._status = TwinrStatus.WAITING
        self.last_error = None
        self.printing_active = bool(printing_active)
        self.epoch = _coerce_non_negative_int(epoch)
        self.sequence = _coerce_non_negative_int(sequence)

        restored_status_error: str | None = None
        try:
            self._status = _coerce_status(status)
        except (TypeError, ValueError):
            self._status = TwinrStatus.ERROR
            restored_status_error = f"Invalid restored status: {status!r}"

        self._restore_history(history)
        self._restore_audit(audit)
        self._canonicalize_locked()

        if self._status is TwinrStatus.ERROR:
            self.last_error = _normalize_error_message(restored_status_error or last_error)
        else:
            self.last_error = None

        if self._audit:
            self.sequence = max(self.sequence, self._audit[-1].sequence)

    def __repr__(self) -> str:
        return (
            "TwinrStateMachine("
            f"status={self.status.value!r}, "
            f"printing_active={self.printing_active!r}, "
            f"last_error={self.last_error!r}, "
            f"epoch={self.epoch!r}, "
            f"sequence={self.sequence!r})"
        )

    @property
    def status(self) -> TwinrStatus:
        """Return the current canonical top-level status."""

        with self._lock:
            return _safe_status_for_history(self._status)

    @status.setter
    def status(self, value: TwinrStatus | str) -> None:
        """Set the current canonical status and re-apply invariants."""

        with self._lock:
            self._status = _coerce_status(value)
            self._canonicalize_locked()

    def restore_snapshot_state(
        self,
        *,
        status: TwinrStatus | str,
        printing_active: bool = False,
    ) -> None:
        """Restore canonical status fields from persisted runtime snapshot data."""

        with self._lock:
            self._status = _coerce_status(status)
            self.printing_active = bool(printing_active)
            self._canonicalize_locked()

    @property
    def history(self) -> list[tuple[TwinrStatus, TwinrEvent, TwinrStatus]]:
        """Return a snapshot of the bounded legacy transition history."""

        with self._lock:
            return list(self._history)

    @property
    def audit(self) -> list[TwinrTransitionRecord]:
        """Return a snapshot of the bounded structured audit log."""

        with self._lock:
            return list(self._audit)

    @property
    def primary_status(self) -> TwinrStatus:
        """Return the non-printing foreground status.

        When ``status`` is ``PRINTING``, printing is the only active foreground
        activity, so the primary status is conceptually ``WAITING``.
        """

        with self._lock:
            current = _safe_status_for_history(self._status)
            if current is TwinrStatus.PRINTING:
                return TwinrStatus.WAITING
            return current

    @property
    def active_statuses(self) -> tuple[TwinrStatus, ...]:
        """Return the full active status configuration.

        This exposes the orthogonal printing activity without forcing callers to
        infer it from legacy top-level state alone.
        """

        with self._lock:
            current = _safe_status_for_history(self._status)
            if current is TwinrStatus.ERROR:
                return (TwinrStatus.ERROR,)
            if current is TwinrStatus.PRINTING:
                return (TwinrStatus.PRINTING,)
            if self.printing_active:
                return (current, TwinrStatus.PRINTING)
            return (current,)

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, object]) -> "TwinrStateMachine":
        """Restore a state machine from a schema-versioned snapshot mapping."""

        if not isinstance(snapshot, Mapping):
            raise TypeError("snapshot must be a mapping")

        restored_status = cast(TwinrStatus | str, snapshot.get("status", TwinrStatus.WAITING))
        return cls(
            status=restored_status,
            last_error=snapshot.get("last_error"),  # type: ignore[arg-type]
            history=snapshot.get("history", ()),  # type: ignore[arg-type]
            audit=snapshot.get("audit", ()),  # type: ignore[arg-type]
            printing_active=bool(snapshot.get("printing_active", False)),
            epoch=_coerce_non_negative_int(snapshot.get("epoch", 0)),
            sequence=_coerce_non_negative_int(snapshot.get("sequence", 0)),
        )

    def snapshot(self, *, include_audit: bool = False) -> dict[str, object]:
        """Export a schema-versioned, JSON-serializable snapshot."""

        with self._lock:
            payload: dict[str, object] = {
                "schema_version": _SNAPSHOT_SCHEMA_VERSION,
                "status": _safe_status_for_history(self._status).value,
                "printing_active": self.printing_active,
                "last_error": self.last_error,
                "epoch": self.epoch,
                "sequence": self.sequence,
                "history": [
                    (previous.value, event.value, next_status.value)
                    for previous, event, next_status in self._history
                ],
            }
            if include_audit:
                payload["audit"] = [record.to_dict() for record in self._audit]
            return payload

    def can(self, event: TwinrEvent | str) -> bool:
        """Return whether an event is valid or safely ignorable right now."""

        normalized_event = _coerce_event(event)
        with self._lock:
            if normalized_event in {TwinrEvent.FAILURE, TwinrEvent.RESET}:
                return True

            current_status = _safe_status_for_history(self._status)
            try:
                self._precheck_async_completion_locked(current_status, normalized_event)
                self._resolve_next_status_locked(current_status, normalized_event)
            except _IgnoredEvent:
                return True
            except InvalidTransitionError:
                return False
            return True

    def transition(
        self,
        event: TwinrEvent | str,
        *,
        expected_epoch: int | None = None,
        error: object | None = None,
        source: str | None = None,
    ) -> TwinrStatus:
        """Apply one event-driven transition and return the new status.

        Args:
            event: Canonical event enum or its persisted string value.
            expected_epoch: Optional epoch token captured when async work began.
                If supplied and stale, the event is ignored instead of applied.
            error: Optional failure payload used only when ``event`` is
                ``TwinrEvent.FAILURE``.
            source: Optional source label for audit records.

        Returns:
            The current canonical runtime status after applying or safely
            ignoring the event.

        Raises:
            InvalidTransitionError: If the event is unknown or truly illegal
                for the current state.
        """

        normalized_event = _coerce_event(event)
        normalized_source = _normalize_source(source)

        if normalized_event is TwinrEvent.FAILURE:
            return self.fail(
                error if error is not None else _DEFAULT_FAILURE_MESSAGE,
                expected_epoch=expected_epoch,
                source=normalized_source,
            )

        with self._lock:
            if normalized_event is TwinrEvent.RESET:
                previous = _safe_status_for_history(self._status)
                epoch_before = self.epoch
                self._status = TwinrStatus.WAITING
                self.printing_active = False
                self.last_error = None
                self.epoch += 1
                self._canonicalize_locked()
                self._append_history(previous, normalized_event, self._status)
                self._append_audit(
                    previous=previous,
                    event=normalized_event,
                    next_status=self._status,
                    epoch_before=epoch_before,
                    source=normalized_source,
                )
                return self._status

            if expected_epoch is not None and expected_epoch != self.epoch:
                current = _safe_status_for_history(self._status)
                self._append_audit(
                    previous=current,
                    event=normalized_event,
                    next_status=current,
                    epoch_before=self.epoch,
                    source=normalized_source,
                    ignored=True,
                    note=f"ignored stale event for epoch {expected_epoch}; current epoch is {self.epoch}",
                )
                return current

            try:
                current_status = _coerce_status(self._status)
            except (TypeError, ValueError):
                return self.fail(
                    f"Invalid internal status: {self._status!r}",
                    expected_epoch=expected_epoch,
                    source=normalized_source,
                )

            try:
                self._precheck_async_completion_locked(current_status, normalized_event)
                previous = current_status
                epoch_before = self.epoch
                next_status = self._resolve_next_status_locked(current_status, normalized_event)
                if next_status is not TwinrStatus.ERROR:
                    self.last_error = None
                self._status = next_status
                self.epoch += 1
                self._canonicalize_locked()
                self._append_history(previous, normalized_event, self._status)
                self._append_audit(
                    previous=previous,
                    event=normalized_event,
                    next_status=self._status,
                    epoch_before=epoch_before,
                    source=normalized_source,
                )
                return self._status
            except _IgnoredEvent as ignored:
                current = _safe_status_for_history(self._status)
                self._append_audit(
                    previous=current,
                    event=normalized_event,
                    next_status=current,
                    epoch_before=self.epoch,
                    source=normalized_source,
                    ignored=True,
                    note=ignored.note,
                )
                return current

    def fail(
        self,
        message: object,
        *,
        expected_epoch: int | None = None,
        source: str | None = None,
    ) -> TwinrStatus:
        """Move the state machine into ``ERROR`` with a normalized message."""

        normalized_message = _normalize_error_message(message)
        normalized_source = _normalize_source(source)

        with self._lock:
            if expected_epoch is not None and expected_epoch != self.epoch:
                current = _safe_status_for_history(self._status)
                self._append_audit(
                    previous=current,
                    event=TwinrEvent.FAILURE,
                    next_status=current,
                    epoch_before=self.epoch,
                    source=normalized_source,
                    error=normalized_message,
                    ignored=True,
                    note=f"ignored stale failure for epoch {expected_epoch}; current epoch is {self.epoch}",
                )
                return current

            previous = _safe_status_for_history(self._status)
            epoch_before = self.epoch
            self._status = TwinrStatus.ERROR
            self.printing_active = False
            self.last_error = normalized_message
            self.epoch += 1
            self._canonicalize_locked()
            self._append_history(previous, TwinrEvent.FAILURE, self._status)
            self._append_audit(
                previous=previous,
                event=TwinrEvent.FAILURE,
                next_status=self._status,
                epoch_before=epoch_before,
                source=normalized_source,
                error=normalized_message,
            )
            return self._status

    def reset(
        self,
        *,
        expected_epoch: int | None = None,
        source: str | None = None,
    ) -> TwinrStatus:
        """Reset the state machine through the canonical ``RESET`` path.

        ``RESET`` is supervisory and intentionally ignores ``expected_epoch`` so
        operators can always recover the machine.
        """

        del expected_epoch
        return self.transition(TwinrEvent.RESET, source=source)

    def _restore_history(self, raw_history: Iterable[object]) -> None:
        """Best-effort restore legacy history tuples from persistence."""

        try:
            iterable = raw_history or ()
        except Exception:
            iterable = ()

        normalized: deque[tuple[TwinrStatus, TwinrEvent, TwinrStatus]] = deque(
            maxlen=_HISTORY_LIMIT
        )
        for entry in iterable:
            if not isinstance(entry, (tuple, list)) or len(entry) != 3:
                continue

            previous_raw, event_raw, next_raw = entry
            try:
                normalized.append(
                    (
                        _coerce_status(previous_raw),
                        _coerce_event(event_raw),
                        _coerce_status(next_raw),
                    )
                )
            except (InvalidTransitionError, TypeError, ValueError):
                continue

        self._history = normalized

    def _restore_audit(self, raw_audit: Iterable[object]) -> None:
        """Best-effort restore structured audit records from persistence."""

        try:
            iterable = raw_audit or ()
        except Exception:
            iterable = ()

        normalized: deque[TwinrTransitionRecord] = deque(maxlen=_AUDIT_LIMIT)
        for entry in iterable:
            record = self._coerce_audit_record(entry)
            if record is not None:
                normalized.append(record)

        self._audit = normalized

    def _coerce_audit_record(self, entry: object) -> TwinrTransitionRecord | None:
        """Coerce one persisted audit record mapping into a dataclass record."""

        if isinstance(entry, TwinrTransitionRecord):
            return entry
        if not isinstance(entry, Mapping):
            return None

        try:
            previous = _coerce_status(entry["previous"])  # type: ignore[index]
            event = _coerce_event(entry["event"])  # type: ignore[index]
            next_status = _coerce_status(entry["next"])  # type: ignore[index]
        except (KeyError, TypeError, ValueError, InvalidTransitionError):
            return None

        error = entry.get("error")
        note = entry.get("note")

        return TwinrTransitionRecord(
            sequence=_coerce_non_negative_int(entry.get("sequence", 0)),
            epoch_before=_coerce_non_negative_int(entry.get("epoch_before", 0)),
            epoch_after=_coerce_non_negative_int(entry.get("epoch_after", 0)),
            occurred_at=str(entry.get("occurred_at", _utc_now_iso())),
            monotonic_time_ns=_coerce_non_negative_int(entry.get("monotonic_time_ns", 0)),
            previous=previous,
            event=event,
            next=next_status,
            printing_active=bool(entry.get("printing_active", False)),
            source=_normalize_source(entry.get("source")),
            error=_normalize_error_message(error) if error is not None else None,
            ignored=bool(entry.get("ignored", False)),
            note=_normalize_source(note),
        )

    def _append_history(
        self,
        previous: TwinrStatus,
        event: TwinrEvent,
        next_status: TwinrStatus,
    ) -> None:
        """Append one accepted transition to the bounded legacy history."""

        self._history.append((previous, event, next_status))

    def _append_audit(
        self,
        *,
        previous: TwinrStatus,
        event: TwinrEvent,
        next_status: TwinrStatus,
        epoch_before: int,
        source: str | None,
        error: str | None = None,
        ignored: bool = False,
        note: str | None = None,
    ) -> None:
        """Append one structured audit record."""

        self.sequence += 1
        epoch_after = self.epoch if not ignored else epoch_before
        self._audit.append(
            TwinrTransitionRecord(
                sequence=self.sequence,
                epoch_before=epoch_before,
                epoch_after=epoch_after,
                occurred_at=_utc_now_iso(),
                monotonic_time_ns=monotonic_ns(),
                previous=previous,
                event=event,
                next=next_status,
                printing_active=self.printing_active,
                source=source,
                error=error,
                ignored=ignored,
                note=note,
            )
        )

    def _canonicalize_locked(self) -> None:
        """Restore internal invariants after direct status / flag updates."""

        if self._status is TwinrStatus.ERROR:
            self.printing_active = False

        if self._status is TwinrStatus.PRINTING:
            self.printing_active = True
        elif self.printing_active and self._status is TwinrStatus.WAITING:
            # print is the only active operation, so legacy callers still see
            # the canonical top-level PRINTING status.
            self._status = TwinrStatus.PRINTING
        elif not self.printing_active and self._status is TwinrStatus.PRINTING:
            self._status = TwinrStatus.WAITING

    def _precheck_async_completion_locked(
        self,
        current_status: TwinrStatus,
        event: TwinrEvent,
    ) -> None:
        """Ignore safe late async completions instead of raising."""

        if event is TwinrEvent.PRINT_FINISHED:
            if not self.printing_active and current_status is not TwinrStatus.PRINTING:
                raise _IgnoredEvent("ignored print completion without an active print job")
            return

        expected_sources = _ASYNC_SOURCE_STATES.get(event)
        if expected_sources is not None and current_status not in expected_sources:
            allowed_states = ", ".join(status.value for status in expected_sources)
            raise _IgnoredEvent(
                f"ignored late {event.value} while in {current_status.value}; expected one of {allowed_states}"
            )

    def _resolve_next_status_locked(
        self,
        current_status: TwinrStatus,
        event: TwinrEvent,
    ) -> TwinrStatus:
        """Resolve the next canonical status for one accepted event."""

        if event in _PRINT_START_EVENTS:
            return self._start_print_locked(current_status, event)

        if event is TwinrEvent.PRINT_FINISHED:
            return self._finish_print_locked(current_status)

        if event is TwinrEvent.TTS_FINISHED:
            return TwinrStatus.PRINTING if self.printing_active else TwinrStatus.WAITING

        next_status = _BASE_TRANSITIONS.get(current_status, {}).get(event)
        if next_status is None:
            raise InvalidTransitionError(
                f"Cannot apply {event.value} while in {current_status.value}"
            )
        return next_status

    def _start_print_locked(
        self,
        current_status: TwinrStatus,
        event: TwinrEvent,
    ) -> TwinrStatus:
        """Start printing as either an orthogonal or standalone activity."""

        if self.printing_active or current_status is TwinrStatus.PRINTING:
            raise _IgnoredEvent(f"ignored duplicate {event.value} while printing is already active")

        if current_status is TwinrStatus.WAITING:
            self.printing_active = True
            return TwinrStatus.PRINTING

        if current_status in {TwinrStatus.PROCESSING, TwinrStatus.ANSWERING}:
            self.printing_active = True
            return current_status

        raise InvalidTransitionError(f"Cannot apply {event.value} while in {current_status.value}")

    def _finish_print_locked(self, current_status: TwinrStatus) -> TwinrStatus:
        """Finish printing and preserve any concurrent foreground activity."""

        if not self.printing_active and current_status is not TwinrStatus.PRINTING:
            raise _IgnoredEvent("ignored print completion without an active print job")

        self.printing_active = False
        if current_status is TwinrStatus.PRINTING:
            return TwinrStatus.WAITING
        return current_status
