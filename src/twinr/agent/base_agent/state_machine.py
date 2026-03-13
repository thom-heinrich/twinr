from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class TwinrStatus(StrEnum):
    WAITING = "waiting"
    LISTENING = "listening"
    PROCESSING = "processing"
    ANSWERING = "answering"
    PRINTING = "printing"
    ERROR = "error"


class TwinrEvent(StrEnum):
    GREEN_BUTTON_PRESSED = "green_button_pressed"
    SPEECH_PAUSE_DETECTED = "speech_pause_detected"
    RESPONSE_READY = "response_ready"
    PROACTIVE_PROMPT_READY = "proactive_prompt_ready"
    TTS_FINISHED = "tts_finished"
    YELLOW_BUTTON_PRESSED = "yellow_button_pressed"
    PRINT_REQUESTED = "print_requested"
    PRINT_FINISHED = "print_finished"
    FAILURE = "failure"
    RESET = "reset"


class InvalidTransitionError(RuntimeError):
    pass


_TRANSITIONS: dict[TwinrStatus, dict[TwinrEvent, TwinrStatus]] = {
    TwinrStatus.WAITING: {
        TwinrEvent.GREEN_BUTTON_PRESSED: TwinrStatus.LISTENING,
        TwinrEvent.YELLOW_BUTTON_PRESSED: TwinrStatus.PRINTING,
        TwinrEvent.PROACTIVE_PROMPT_READY: TwinrStatus.ANSWERING,
    },
    TwinrStatus.LISTENING: {
        TwinrEvent.SPEECH_PAUSE_DETECTED: TwinrStatus.PROCESSING,
        TwinrEvent.RESET: TwinrStatus.WAITING,
    },
    TwinrStatus.PROCESSING: {
        TwinrEvent.RESPONSE_READY: TwinrStatus.ANSWERING,
        TwinrEvent.PRINT_REQUESTED: TwinrStatus.PRINTING,
    },
    TwinrStatus.ANSWERING: {
        TwinrEvent.TTS_FINISHED: TwinrStatus.WAITING,
        TwinrEvent.PRINT_REQUESTED: TwinrStatus.PRINTING,
    },
    TwinrStatus.PRINTING: {
        TwinrEvent.PRINT_FINISHED: TwinrStatus.WAITING,
        TwinrEvent.RESPONSE_READY: TwinrStatus.ANSWERING,
    },
    TwinrStatus.ERROR: {
        TwinrEvent.RESET: TwinrStatus.WAITING,
    },
}


@dataclass(slots=True)
class TwinrStateMachine:
    status: TwinrStatus = TwinrStatus.WAITING
    last_error: str | None = None
    history: list[tuple[TwinrStatus, TwinrEvent, TwinrStatus]] = field(default_factory=list)

    def transition(self, event: TwinrEvent) -> TwinrStatus:
        next_status = _TRANSITIONS.get(self.status, {}).get(event)
        if next_status is None:
            raise InvalidTransitionError(f"Cannot apply {event.value} while in {self.status.value}")
        previous = self.status
        self.status = next_status
        if next_status != TwinrStatus.ERROR:
            self.last_error = None
        self.history.append((previous, event, next_status))
        return self.status

    def fail(self, message: str) -> TwinrStatus:
        previous = self.status
        self.status = TwinrStatus.ERROR
        self.last_error = message
        self.history.append((previous, TwinrEvent.FAILURE, TwinrStatus.ERROR))
        return self.status

    def reset(self) -> TwinrStatus:
        return self.transition(TwinrEvent.RESET)
