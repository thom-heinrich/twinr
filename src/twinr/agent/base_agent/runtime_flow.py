from __future__ import annotations

from twinr.agent.base_agent.state_machine import InvalidTransitionError, TwinrEvent, TwinrStatus
from twinr.ops.events import compact_text


class TwinrRuntimeFlowMixin:
    def begin_listening(
        self,
        *,
        request_source: str,
        button: str | None = None,
        proactive_trigger: str | None = None,
    ) -> TwinrStatus:
        status = self.state_machine.transition(TwinrEvent.GREEN_BUTTON_PRESSED)
        self._persist_snapshot()
        data = {
            "status": status.value,
            "request_source": request_source,
        }
        if button:
            data["button"] = button
        if proactive_trigger:
            data["proactive_trigger"] = proactive_trigger
        self.ops_events.append(
            event="turn_started",
            message="Conversation listening window started.",
            data=data,
        )
        return status

    def press_green_button(self) -> TwinrStatus:
        status = self.begin_listening(request_source="button", button="green")
        self.long_term_memory.enqueue_multimodal_evidence(
            event_name="button_interaction",
            modality="button",
            source="runtime_button",
            message="Green button started a conversation turn.",
            data={"button": "green", "action": "start_listening"},
        )
        return status

    def submit_transcript(self, transcript: str) -> TwinrStatus:
        self.last_transcript = transcript.strip()
        status = self.state_machine.transition(TwinrEvent.SPEECH_PAUSE_DETECTED)
        self._persist_snapshot()
        self.ops_events.append(
            event="transcript_submitted",
            message="Transcript captured for the active turn.",
            data={"status": status.value, "transcript_preview": compact_text(self.last_transcript)},
        )
        return status

    def begin_answering(self) -> TwinrStatus:
        status = self.state_machine.transition(TwinrEvent.RESPONSE_READY)
        self._persist_snapshot()
        return status

    def begin_proactive_prompt(self, prompt: str) -> str:
        return self._begin_background_prompt(
            prompt,
            event="proactive_prompt_started",
            message="Twinr started a proactive spoken prompt.",
        )

    def begin_reminder_prompt(self, prompt: str) -> str:
        return self._begin_background_prompt(
            prompt,
            event="reminder_prompt_started",
            message="Twinr started speaking a due reminder.",
        )

    def begin_automation_prompt(self, prompt: str) -> str:
        return self._begin_background_prompt(
            prompt,
            event="automation_prompt_started",
            message="Twinr started speaking a due automation prompt.",
        )

    def begin_wakeword_prompt(self, prompt: str) -> str:
        return self._begin_background_prompt(
            prompt,
            event="wakeword_prompt_started",
            message="Twinr acknowledged a wakeword before opening hands-free listening.",
        )

    def _begin_background_prompt(self, prompt: str, *, event: str, message: str) -> str:
        spoken_prompt = prompt.strip()
        if not spoken_prompt:
            raise RuntimeError("Background prompt text must not be empty")
        status = self.state_machine.transition(TwinrEvent.PROACTIVE_PROMPT_READY)
        self.memory.remember("assistant", spoken_prompt)
        self._persist_snapshot()
        self.ops_events.append(
            event=event,
            message=message,
            data={
                "status": status.value,
                "response_preview": compact_text(spoken_prompt),
            },
        )
        return spoken_prompt

    def cancel_listening(self) -> TwinrStatus:
        status = self.state_machine.reset()
        self._persist_snapshot()
        return status

    def finalize_agent_turn(self, answer: str) -> str:
        if not self.last_transcript:
            raise RuntimeError("No transcript is available for response generation")
        response = answer.strip()
        self.memory.remember("user", self.last_transcript)
        self.memory.remember("assistant", response)
        self.last_response = response
        self.long_term_memory.enqueue_conversation_turn(
            transcript=self.last_transcript,
            response=response,
        )
        self._persist_snapshot()
        self.ops_events.append(
            event="turn_completed",
            message="Assistant response stored in runtime memory.",
            data={
                "status": self.status.value,
                "transcript_preview": compact_text(self.last_transcript),
                "response_preview": compact_text(response),
            },
        )
        return response

    def complete_agent_turn(self, answer: str) -> str:
        self.begin_answering()
        return self.finalize_agent_turn(answer)

    def finish_speaking(self) -> TwinrStatus:
        status = self.state_machine.transition(TwinrEvent.TTS_FINISHED)
        self._persist_snapshot()
        self.ops_events.append(
            event="tts_finished",
            message="Twinr finished speaking the response.",
            data={"status": status.value},
        )
        return status

    def press_yellow_button(self) -> str:
        if not self.last_response:
            raise RuntimeError("No assistant response is available for printing")
        self.state_machine.transition(TwinrEvent.YELLOW_BUTTON_PRESSED)
        self._persist_snapshot()
        self.ops_events.append(
            event="print_started",
            message="Yellow button requested a print.",
            data={
                "button": "yellow",
                "status": self.status.value,
                "response_preview": compact_text(self.last_response),
            },
        )
        self.long_term_memory.enqueue_multimodal_evidence(
            event_name="button_interaction",
            modality="button",
            source="runtime_button",
            message="Yellow button requested a printed answer.",
            data={"button": "yellow", "action": "print_request"},
        )
        return self.last_response

    def begin_tool_print(self) -> TwinrStatus:
        status = self.state_machine.transition(TwinrEvent.PRINT_REQUESTED)
        self._persist_snapshot()
        self.ops_events.append(
            event="print_started",
            message="Assistant tool requested a print.",
            data={"request_source": "tool", "status": status.value},
        )
        return status

    def resume_answering_after_print(self) -> TwinrStatus:
        status = self.state_machine.transition(TwinrEvent.RESPONSE_READY)
        self._persist_snapshot()
        return status

    def maybe_begin_tool_print(self) -> TwinrStatus | None:
        try:
            return self.begin_tool_print()
        except InvalidTransitionError:
            return None

    def begin_automation_print(self) -> TwinrStatus:
        try:
            status = self.state_machine.transition(TwinrEvent.PRINT_REQUESTED)
        except InvalidTransitionError:
            if self.state_machine.status != TwinrStatus.WAITING:
                raise
            status = self.state_machine.transition(TwinrEvent.YELLOW_BUTTON_PRESSED)
        self._persist_snapshot()
        self.ops_events.append(
            event="print_started",
            message="Scheduled automation requested a print.",
            data={"request_source": "automation", "status": status.value},
        )
        return status

    def maybe_begin_automation_print(self) -> TwinrStatus | None:
        try:
            return self.begin_automation_print()
        except InvalidTransitionError:
            return None

    def finish_printing(self) -> TwinrStatus:
        status = self.state_machine.transition(TwinrEvent.PRINT_FINISHED)
        self._persist_snapshot()
        self.ops_events.append(
            event="print_finished",
            message="Twinr returned to idle after printing.",
            data={"status": status.value},
        )
        return status

    def fail(self, message: str) -> TwinrStatus:
        status = self.state_machine.fail(message)
        self._persist_snapshot(error_message=message)
        self.ops_events.append(
            event="error",
            level="error",
            message="Twinr runtime entered the error state.",
            data={"status": status.value, "error": message},
        )
        return status

    def reset_error(self) -> TwinrStatus:
        status = self.state_machine.reset()
        self._persist_snapshot()
        self.ops_events.append(
            event="error_reset",
            message="Twinr runtime left the error state.",
            data={"status": status.value},
        )
        return status
