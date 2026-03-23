"""Define deterministic runtime status transitions for the base Twinr agent.

This module exports the canonical status and event enums plus the state
machine used by runtime flow orchestration. Import these definitions from the
package root when coordinating runtime behavior or consuming persisted status
values elsewhere in Twinr.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from threading import RLock


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

    pass


# AUDIT-FIX(#4): Serialize mutations so shared state cannot be torn by concurrent threadpool
# or hardware-callback access.
_STATE_MACHINE_LOCK = RLock()

# AUDIT-FIX(#6): Keep diagnostic history bounded on a long-running Raspberry Pi process.
_HISTORY_LIMIT = 256

# AUDIT-FIX(#7): Guarantee a deterministic, non-empty fallback error message.
_DEFAULT_FAILURE_MESSAGE = "Unknown failure"


def _coerce_status(value: TwinrStatus | str) -> TwinrStatus:
    """Coerce a raw status value to ``TwinrStatus``."""

    # AUDIT-FIX(#3): Accept restored string state values and reject corrupted values explicitly.
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

    # AUDIT-FIX(#5): Normalize raw/deserialized event values before transition lookup.
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

    # AUDIT-FIX(#3): Preserve recoverability even if external code injected a corrupted status.
    try:
        return _coerce_status(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return TwinrStatus.ERROR


def _normalize_error_message(message: object) -> str:
    """Normalize arbitrary failure payloads to a non-empty error string."""

    # AUDIT-FIX(#7): Prevent blank/None payloads from erasing actionable diagnostics.
    if message is None:
        return _DEFAULT_FAILURE_MESSAGE
    if isinstance(message, str):
        normalized = message.strip()
    else:
        normalized = str(message).strip()
    return normalized or _DEFAULT_FAILURE_MESSAGE


_TRANSITIONS: dict[TwinrStatus, dict[TwinrEvent, TwinrStatus]] = {
    TwinrStatus.WAITING: {
        TwinrEvent.GREEN_BUTTON_PRESSED: TwinrStatus.LISTENING,
        TwinrEvent.YELLOW_BUTTON_PRESSED: TwinrStatus.PRINTING,
        TwinrEvent.PROACTIVE_PROMPT_READY: TwinrStatus.ANSWERING,
        TwinrEvent.RESET: TwinrStatus.WAITING,  # AUDIT-FIX(#2): Make reset idempotent from idle.
    },
    TwinrStatus.LISTENING: {
        TwinrEvent.SPEECH_PAUSE_DETECTED: TwinrStatus.PROCESSING,
        TwinrEvent.RESET: TwinrStatus.WAITING,
    },
    TwinrStatus.PROCESSING: {
        TwinrEvent.RESPONSE_READY: TwinrStatus.ANSWERING,
        TwinrEvent.PRINT_REQUESTED: TwinrStatus.PRINTING,
        TwinrEvent.RESET: TwinrStatus.WAITING,  # AUDIT-FIX(#2): Allow operator recovery from stuck processing.
    },
    TwinrStatus.ANSWERING: {
        TwinrEvent.PROCESSING_RESUMED: TwinrStatus.PROCESSING,
        TwinrEvent.FOLLOW_UP_ARMED: TwinrStatus.LISTENING,
        TwinrEvent.TTS_FINISHED: TwinrStatus.WAITING,
        TwinrEvent.PRINT_REQUESTED: TwinrStatus.PRINTING,
        TwinrEvent.RESET: TwinrStatus.WAITING,  # AUDIT-FIX(#2): Allow recovery if audio/TTS stalls.
    },
    TwinrStatus.PRINTING: {
        TwinrEvent.PRINT_FINISHED: TwinrStatus.WAITING,
        TwinrEvent.RESPONSE_READY: TwinrStatus.ANSWERING,
        TwinrEvent.RESET: TwinrStatus.WAITING,  # AUDIT-FIX(#2): Allow recovery from printer stalls.
    },
    TwinrStatus.ERROR: {
        TwinrEvent.RESET: TwinrStatus.WAITING,
    },
}


@dataclass(slots=True)
class TwinrStateMachine:
    """Track Twinr runtime status and enforce legal transitions.

    The state machine accepts persisted string values during restoration,
    normalizes them to the canonical enums, and keeps a bounded transition
    history for diagnostics.

    Attributes:
        status: Current runtime status.
        last_error: Last normalized failure message while in ``ERROR``.
        history: Bounded list of ``(previous, event, next)`` transitions.
    """

    status: TwinrStatus = TwinrStatus.WAITING
    last_error: str | None = None
    history: list[tuple[TwinrStatus, TwinrEvent, TwinrStatus]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalize restored status and history after dataclass construction."""

        raw_status = self.status
        restored_status_error: str | None = None
        try:
            # AUDIT-FIX(#3): Normalize file-backed/deserialized status immediately at construction time.
            self.status = _coerce_status(raw_status)
        except (TypeError, ValueError):
            # AUDIT-FIX(#3): Recover from corrupted persisted state by entering ERROR instead of crashing startup.
            self.status = TwinrStatus.ERROR
            restored_status_error = f"Invalid restored status: {raw_status!r}"

        normalized_history: list[tuple[TwinrStatus, TwinrEvent, TwinrStatus]] = []
        try:
            raw_history = list(self.history or [])
        except TypeError:
            # AUDIT-FIX(#3): Treat non-iterable restored history as corrupted and continue with an empty buffer.
            raw_history = []

        # AUDIT-FIX(#3): Best-effort sanitize restored history instead of crashing on malformed entries.
        for entry in raw_history:
            if not isinstance(entry, (tuple, list)) or len(entry) != 3:
                continue

            previous_raw, event_raw, next_raw = entry
            try:
                normalized_history.append(
                    (
                        _coerce_status(previous_raw),
                        _coerce_event(event_raw),
                        _coerce_status(next_raw),
                    )
                )
            except (InvalidTransitionError, TypeError, ValueError):
                continue

        # AUDIT-FIX(#6): Trim restored history eagerly to maintain a bounded memory footprint.
        self.history = normalized_history[-_HISTORY_LIMIT:]

        if self.status == TwinrStatus.ERROR:
            # AUDIT-FIX(#7): Ensure ERROR state always carries a readable diagnostic.
            self.last_error = _normalize_error_message(restored_status_error or self.last_error)
        else:
            # AUDIT-FIX(#3): Drop stale error text when restoring a healthy state.
            self.last_error = None

    def _append_history(
        self,
        previous: TwinrStatus,
        event: TwinrEvent,
        next_status: TwinrStatus,
    ) -> None:
        """Append one transition to the bounded history buffer."""

        # AUDIT-FIX(#6): Preserve list API compatibility while implementing a bounded ring buffer.
        self.history.append((previous, event, next_status))
        if len(self.history) > _HISTORY_LIMIT:
            del self.history[:-_HISTORY_LIMIT]

    def transition(self, event: TwinrEvent | str) -> TwinrStatus:
        """Apply one event-driven transition and return the next status.

        Args:
            event: Canonical event enum or its persisted string value.

        Returns:
            The new normalized runtime status after the transition.

        Raises:
            InvalidTransitionError: If the event is unknown or not allowed
                from the current status.
        """

        normalized_event = _coerce_event(event)
        with _STATE_MACHINE_LOCK:
            # AUDIT-FIX(#1): Make FAILURE usable through the same dispatcher path as every other event.
            if normalized_event is TwinrEvent.FAILURE:
                return self.fail(_DEFAULT_FAILURE_MESSAGE)

            try:
                current_status = _coerce_status(self.status)
            except (TypeError, ValueError):
                if normalized_event is TwinrEvent.RESET:
                    # AUDIT-FIX(#2): RESET must remain a hard recovery path even after state corruption.
                    previous = _safe_status_for_history(self.status)
                    self.status = TwinrStatus.WAITING
                    self.last_error = None
                    self._append_history(previous, normalized_event, TwinrStatus.WAITING)
                    return self.status

                # AUDIT-FIX(#3): Fail closed with a readable diagnostic instead of crashing on corrupted state.
                return self.fail(f"Invalid internal status: {self.status!r}")

            self.status = current_status
            next_status = _TRANSITIONS.get(current_status, {}).get(normalized_event)
            if next_status is None:
                raise InvalidTransitionError(
                    f"Cannot apply {normalized_event.value} while in {current_status.value}"
                )

            previous = current_status
            self.status = next_status
            if next_status != TwinrStatus.ERROR:
                self.last_error = None
            self._append_history(previous, normalized_event, next_status)
            return self.status

    def fail(self, message: object) -> TwinrStatus:
        """Move the state machine into ``ERROR`` with a normalized message.

        Args:
            message: Arbitrary failure payload to normalize for diagnostics.

        Returns:
            ``TwinrStatus.ERROR`` after the failure has been recorded.
        """

        # AUDIT-FIX(#7): Normalize arbitrary caller payloads into a stable user/log-visible error string.
        normalized_message = _normalize_error_message(message)
        with _STATE_MACHINE_LOCK:
            previous = _safe_status_for_history(self.status)
            self.status = TwinrStatus.ERROR
            self.last_error = normalized_message
            self._append_history(previous, TwinrEvent.FAILURE, TwinrStatus.ERROR)
            return self.status

    def reset(self) -> TwinrStatus:
        """Reset the state machine through the canonical ``RESET`` path."""

        return self.transition(TwinrEvent.RESET)
