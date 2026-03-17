"""Drive state-machine transitions for turns, speech, and printing."""

from __future__ import annotations

import logging
from threading import RLock

from twinr.agent.base_agent.state.machine import InvalidTransitionError, TwinrEvent, TwinrStatus
from twinr.ops.events import compact_text

_LOGGER = logging.getLogger(__name__)
_RUNTIME_FLOW_LOCK = RLock()


class TwinrRuntimeFlowMixin:
    """Provide serialized runtime flow mutations and user-turn helpers."""

    def _runtime_flow_lock(self) -> RLock:
        return _RUNTIME_FLOW_LOCK  # AUDIT-FIX(#5): Serialize shared runtime mutations across concurrent callers.

    @staticmethod
    def _require_text(value: str, *, field_name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{field_name} must not be empty")
        return cleaned

    @staticmethod
    def _optional_text(value: str | None, *, field_name: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string when provided")
        cleaned = value.strip()
        return cleaned or None

    @staticmethod
    def _sanitize_error_message(message: str) -> str:
        if not isinstance(message, str):
            return "Runtime error"
        printable = "".join(character if character.isprintable() else " " for character in message)
        cleaned = " ".join(printable.split())
        if not cleaned:
            return "Runtime error"
        if len(cleaned) > 200:
            cleaned = f"{cleaned[:197].rstrip()}..."
        return cleaned  # AUDIT-FIX(#4): Avoid persisting/logging raw multiline or oversized internal errors.

    def _ops_content_preview_enabled(self) -> bool:
        raw_value = getattr(self, "ops_event_include_content_preview", False)
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, str):
            return raw_value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(raw_value)

    def _content_preview(self, text: str) -> str:
        if self._ops_content_preview_enabled():
            return compact_text(text)
        return "[redacted]"  # AUDIT-FIX(#10): Do not copy speech content into ops logs unless explicitly enabled.

    def _persist_snapshot_safe(self, *, error_message: str | None = None) -> None:
        try:
            if error_message is None:
                self._persist_snapshot()
            else:
                self._persist_snapshot(error_message=error_message)
        except Exception as exc:
            _LOGGER.exception("Failed to persist Twinr runtime snapshot.")
            raise RuntimeError("Failed to persist Twinr runtime state.") from exc  # AUDIT-FIX(#9): Normalize snapshot failures.

    def _append_ops_event(self, **kwargs: object) -> None:
        try:
            self.ops_events.append(**kwargs)
        except Exception:
            _LOGGER.exception("Failed to append Twinr ops event %r.", kwargs.get("event"))  # AUDIT-FIX(#3): Telemetry must stay best-effort.

    def _enqueue_multimodal_evidence_safe(self, **kwargs: object) -> None:
        try:
            self.long_term_memory.enqueue_multimodal_evidence(**kwargs)
        except Exception:
            _LOGGER.exception("Failed to enqueue Twinr multimodal evidence.")  # AUDIT-FIX(#3): Auxiliary memory failures must not break UX.

    def _enqueue_conversation_turn_safe(self, *, transcript: str, response: str) -> None:
        try:
            self.long_term_memory.enqueue_conversation_turn(
                transcript=transcript,
                response=response,
            )
        except Exception:
            _LOGGER.exception("Failed to enqueue Twinr conversation turn.")  # AUDIT-FIX(#3): Auxiliary memory failures must not break UX.

    def _validated_last_transcript(self) -> str:
        transcript = getattr(self, "last_transcript", None)
        if not isinstance(transcript, str) or not transcript.strip():
            raise RuntimeError("No transcript is available for response generation")
        return transcript.strip()

    def _store_agent_turn(self, cleaned_answer: str, *, transition_to_answering: bool) -> str:
        with self._runtime_flow_lock():  # AUDIT-FIX(#7): Keep transition-to-answering and response storage atomic.
            transcript = self._validated_last_transcript()
            if transition_to_answering:
                status_value = self.state_machine.transition(TwinrEvent.RESPONSE_READY).value
            else:
                status_value = self.state_machine.status.value

            self.memory.remember("user", transcript)
            self.memory.remember("assistant", cleaned_answer)
            self.last_response = cleaned_answer
            self._persist_snapshot_safe()  # AUDIT-FIX(#2): Persist runtime state before durable background indexing.

            event_data = {
                "status": status_value,
                "transcript_preview": self._content_preview(transcript),
                "transcript_chars": len(transcript),
                "response_preview": self._content_preview(cleaned_answer),
                "response_chars": len(cleaned_answer),
            }

        self._enqueue_conversation_turn_safe(
            transcript=transcript,
            response=cleaned_answer,
        )
        self._append_ops_event(
            event="turn_completed",
            message="Assistant response stored in runtime memory.",
            data=event_data,
        )
        return cleaned_answer

    def begin_listening(
        self,
        *,
        request_source: str,
        button: str | None = None,
        proactive_trigger: str | None = None,
    ) -> TwinrStatus:
        """Open a listening turn and record its trigger metadata."""

        normalized_request_source = self._require_text(request_source, field_name="request_source")
        normalized_button = self._optional_text(button, field_name="button")
        normalized_trigger = self._optional_text(proactive_trigger, field_name="proactive_trigger")

        with self._runtime_flow_lock():  # AUDIT-FIX(#5): Serialize listen-start mutations with other runtime events.
            status = self.state_machine.transition(TwinrEvent.GREEN_BUTTON_PRESSED)
            self.last_transcript = ""  # AUDIT-FIX(#6): Clear stale transcript state before a new turn starts.
            self._persist_snapshot_safe()

            data = {
                "status": status.value,
                "request_source": normalized_request_source,
            }
            if normalized_button is not None:
                data["button"] = normalized_button
            if normalized_trigger is not None:
                data["proactive_trigger"] = normalized_trigger

        self._append_ops_event(
            event="turn_started",
            message="Conversation listening window started.",
            data=data,
        )
        return status

    def press_green_button(self) -> TwinrStatus:
        """Start a listening turn from the physical green button."""

        status = self.begin_listening(request_source="button", button="green")
        self._enqueue_multimodal_evidence_safe(
            event_name="button_interaction",
            modality="button",
            source="runtime_button",
            message="Green button started a conversation turn.",
            data={"button": "green", "action": "start_listening"},
        )
        return status

    def submit_transcript(self, transcript: str) -> TwinrStatus:
        """Store the captured transcript for the active turn."""

        cleaned_transcript = self._require_text(transcript, field_name="transcript")  # AUDIT-FIX(#1): Validate before mutating runtime state.

        with self._runtime_flow_lock():
            status = self.state_machine.transition(TwinrEvent.SPEECH_PAUSE_DETECTED)
            self.last_transcript = cleaned_transcript
            self._persist_snapshot_safe()

            event_data = {
                "status": status.value,
                "transcript_preview": self._content_preview(cleaned_transcript),
                "transcript_chars": len(cleaned_transcript),
            }

        self._append_ops_event(
            event="transcript_submitted",
            message="Transcript captured for the active turn.",
            data=event_data,
        )
        return status

    def begin_answering(self) -> TwinrStatus:
        """Move the runtime into the answering state without storing a reply."""

        with self._runtime_flow_lock():  # AUDIT-FIX(#5): Serialize state transitions with transcript/print operations.
            status = self.state_machine.transition(TwinrEvent.RESPONSE_READY)
            self._persist_snapshot_safe()
        return status

    def begin_proactive_prompt(self, prompt: str) -> str:
        """Start a proactive spoken prompt outside a user-initiated turn."""

        return self._begin_background_prompt(
            prompt,
            event="proactive_prompt_started",
            message="Twinr started a proactive spoken prompt.",
        )

    def begin_reminder_prompt(self, prompt: str) -> str:
        """Start speaking a due reminder prompt."""

        return self._begin_background_prompt(
            prompt,
            event="reminder_prompt_started",
            message="Twinr started speaking a due reminder.",
        )

    def begin_automation_prompt(self, prompt: str) -> str:
        """Start speaking a due automation prompt."""

        return self._begin_background_prompt(
            prompt,
            event="automation_prompt_started",
            message="Twinr started speaking a due automation prompt.",
        )

    def begin_wakeword_prompt(self, prompt: str) -> str:
        """Start a wakeword acknowledgement prompt before hands-free listening."""

        return self._begin_background_prompt(
            prompt,
            event="wakeword_prompt_started",
            message="Twinr acknowledged a wakeword before opening hands-free listening.",
        )

    def _begin_background_prompt(self, prompt: str, *, event: str, message: str) -> str:
        spoken_prompt = self._require_text(prompt, field_name="prompt")

        with self._runtime_flow_lock():  # AUDIT-FIX(#5): Keep prompt state, memory, and snapshot aligned.
            status = self.state_machine.transition(TwinrEvent.PROACTIVE_PROMPT_READY)
            self.memory.remember("assistant", spoken_prompt)
            self._persist_snapshot_safe()

            event_data = {
                "status": status.value,
                "response_preview": self._content_preview(spoken_prompt),
                "response_chars": len(spoken_prompt),
            }

        self._append_ops_event(
            event=event,
            message=message,
            data=event_data,
        )
        return spoken_prompt

    def cancel_listening(self) -> TwinrStatus:
        """Abort the active listening turn and clear stale transcript state."""

        with self._runtime_flow_lock():  # AUDIT-FIX(#6): Reset stale transcript state when a listening turn is cancelled.
            status = self.state_machine.reset()
            self.last_transcript = ""
            self._persist_snapshot_safe()
        return status

    def finalize_agent_turn(self, answer: str) -> str:
        """Store an assistant answer without changing the current status."""

        cleaned_answer = self._require_text(answer, field_name="answer")  # AUDIT-FIX(#2): Reject blank assistant answers early.
        return self._store_agent_turn(cleaned_answer, transition_to_answering=False)

    def complete_agent_turn(self, answer: str) -> str:
        """Store an assistant answer and transition into answering."""

        cleaned_answer = self._require_text(answer, field_name="answer")
        return self._store_agent_turn(cleaned_answer, transition_to_answering=True)

    def finish_speaking(self) -> TwinrStatus:
        """Mark the spoken response as finished and return toward idle."""

        with self._runtime_flow_lock():  # AUDIT-FIX(#6): Drop stale transcript after the spoken turn is complete.
            status = self.state_machine.transition(TwinrEvent.TTS_FINISHED)
            self.last_transcript = ""
            self._persist_snapshot_safe()

        self._append_ops_event(
            event="tts_finished",
            message="Twinr finished speaking the response.",
            data={"status": status.value},
        )
        return status

    def rearm_follow_up(
        self,
        *,
        request_source: str = "follow_up",
    ) -> TwinrStatus:
        """Re-open listening directly after speaking without a transient idle state.

        This is used for conversation follow-up turns so the runtime can move
        straight from ``answering`` back to ``listening`` and the operator
        surfaces do not briefly show ``waiting`` between the spoken reply and
        the reopened microphone window.
        """

        normalized_request_source = self._require_text(request_source, field_name="request_source")

        with self._runtime_flow_lock():
            status = self.state_machine.transition(TwinrEvent.FOLLOW_UP_ARMED)
            self.last_transcript = ""
            self._persist_snapshot_safe()

        self._append_ops_event(
            event="follow_up_rearmed",
            message="Conversation follow-up listening window opened immediately after speech.",
            data={"status": status.value, "request_source": normalized_request_source},
        )
        return status

    def press_yellow_button(self) -> str:
        """Request printing of the last assistant response."""

        with self._runtime_flow_lock():  # AUDIT-FIX(#8): Use the transition result directly and keep print state changes atomic.
            last_response = getattr(self, "last_response", None)
            if not isinstance(last_response, str) or not last_response.strip():
                raise RuntimeError("No assistant response is available for printing")

            status = self.state_machine.transition(TwinrEvent.YELLOW_BUTTON_PRESSED)
            self._persist_snapshot_safe()

            event_data = {
                "button": "yellow",
                "status": status.value,
                "response_preview": self._content_preview(last_response),
                "response_chars": len(last_response),
            }

        self._append_ops_event(
            event="print_started",
            message="Yellow button requested a print.",
            data=event_data,
        )
        self._enqueue_multimodal_evidence_safe(
            event_name="button_interaction",
            modality="button",
            source="runtime_button",
            message="Yellow button requested a printed answer.",
            data={"button": "yellow", "action": "print_request"},
        )
        return last_response

    def prepare_background_button_print_request(self) -> str:
        """Queue a background print request without changing runtime state."""

        with self._runtime_flow_lock():
            last_response = getattr(self, "last_response", None)
            if not isinstance(last_response, str) or not last_response.strip():
                raise RuntimeError("No assistant response is available for printing")

            event_data = {
                "button": "yellow",
                "request_source": "button",
                "background": True,
                "status": self.state_machine.status.value,
                "response_preview": self._content_preview(last_response),
                "response_chars": len(last_response),
            }

        self._append_ops_event(
            event="print_started",
            message="Yellow button queued a background print.",
            data=event_data,
        )
        self._enqueue_multimodal_evidence_safe(
            event_name="button_interaction",
            modality="button",
            source="runtime_button",
            message="Yellow button queued a printed answer in the background.",
            data={"button": "yellow", "action": "background_print_request"},
        )
        return last_response

    def begin_tool_print(self) -> TwinrStatus:
        """Transition into printing for a tool-requested print job."""

        with self._runtime_flow_lock():  # AUDIT-FIX(#5): Serialize print state with other runtime actions.
            status = self.state_machine.transition(TwinrEvent.PRINT_REQUESTED)
            self._persist_snapshot_safe()

        self._append_ops_event(
            event="print_started",
            message="Assistant tool requested a print.",
            data={"request_source": "tool", "status": status.value},
        )
        return status

    def resume_answering_after_print(self) -> TwinrStatus:
        """Return from printing to the answering state."""

        with self._runtime_flow_lock():  # AUDIT-FIX(#5): Serialize print/resume state transitions.
            status = self.state_machine.transition(TwinrEvent.RESPONSE_READY)
            self._persist_snapshot_safe()
        return status

    def maybe_begin_tool_print(self) -> TwinrStatus | None:
        """Attempt a tool print transition and return None if disallowed."""

        try:
            return self.begin_tool_print()
        except InvalidTransitionError:
            return None

    def begin_automation_print(self) -> TwinrStatus:
        """Transition into printing for a scheduled automation output."""

        with self._runtime_flow_lock():  # AUDIT-FIX(#5): Keep automation-print fallback transitions consistent.
            try:
                status = self.state_machine.transition(TwinrEvent.PRINT_REQUESTED)
            except InvalidTransitionError:
                if self.state_machine.status != TwinrStatus.WAITING:
                    raise
                status = self.state_machine.transition(TwinrEvent.YELLOW_BUTTON_PRESSED)
            self._persist_snapshot_safe()

        self._append_ops_event(
            event="print_started",
            message="Scheduled automation requested a print.",
            data={"request_source": "automation", "status": status.value},
        )
        return status

    def maybe_begin_automation_print(self) -> TwinrStatus | None:
        """Attempt an automation print transition and return None if disallowed."""

        try:
            return self.begin_automation_print()
        except InvalidTransitionError:
            return None

    def finish_printing(self) -> TwinrStatus:
        """Mark printing as complete and return to the next idle state."""

        with self._runtime_flow_lock():  # AUDIT-FIX(#5): Serialize print completion with other runtime transitions.
            status = self.state_machine.transition(TwinrEvent.PRINT_FINISHED)
            self._persist_snapshot_safe()

        self._append_ops_event(
            event="print_finished",
            message="Twinr returned to idle after printing.",
            data={"status": status.value},
        )
        return status

    def fail(self, message: str) -> TwinrStatus:
        """Enter the error state with a sanitized operator-safe message."""

        safe_message = self._sanitize_error_message(message)

        with self._runtime_flow_lock():  # AUDIT-FIX(#4): Store a bounded, sanitized error and clear transient turn state.
            status = self.state_machine.fail(safe_message)
            self.last_transcript = ""
            self._persist_snapshot_safe(error_message=safe_message)

        self._append_ops_event(
            event="error",
            level="error",
            message="Twinr runtime entered the error state.",
            data={"status": status.value, "error": safe_message},
        )
        return status

    def reset_error(self) -> TwinrStatus:
        """Leave the error state and clear transient turn state."""

        with self._runtime_flow_lock():  # AUDIT-FIX(#6): Clear stale transcript state when leaving the error path.
            status = self.state_machine.reset()
            self.last_transcript = ""
            self._persist_snapshot_safe()

        self._append_ops_event(
            event="error_reset",
            message="Twinr runtime left the error state.",
            data={"status": status.value},
        )
        return status
