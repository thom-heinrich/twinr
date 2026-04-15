# CHANGELOG: 2026-03-27
# BUG-1: Made workflow_event emission best-effort so telemetry failures no longer abort user turns or snapshot refreshes.
# BUG-2: Added a dedicated printable-output path so proactive/reminder/automation prompts do not leave the yellow-button print flow pointing at a stale older answer.
# BUG-3: Coalesced duplicate green/yellow button activations inside a short debounce window to survive real Raspberry Pi GPIO bounce / duplicated callbacks without invalid transitions or duplicate prints.
# BUG-4: Allow follow-up reopen after `tts_finished` has already moved the
#        runtime to `waiting`, so display-facing speech completion can clear
#        immediately without blocking later legitimate re-entry to `listening`.
# BUG-5: Add one atomic transcript-first follow-up reopen path so provisional
#        remote follow-up turns do not persist an intermediate `waiting`
#        snapshot before immediately returning to `listening`.
# SEC-1: Bounded and sanitized all externally supplied strings (transcripts, answers, prompts, request metadata) to prevent control-character injection into operator surfaces and resource-exhaustion via oversized payloads.
# IMP-1: Replaced the process-global runtime lock with a per-instance lock/state container, with a shared fallback only when instance attribute injection is unavailable.
# IMP-2: Added per-turn correlation metadata (turn_id, parent_turn_id, event_id, event_seq, timestamps, status_before/status_after) to ops events for 2026-grade forensics and observability.
# IMP-3: Switched internal timing to nanosecond monotonic / wall clocks and moved telemetry emission out of critical sections where possible to reduce lock hold time on slow Raspberry Pi storage.

"""Drive state-machine transitions for turns, speech, and printing."""

from __future__ import annotations

import logging
import time
import uuid
from threading import RLock
from typing import Any, cast

from twinr.agent.base_agent.state.machine import InvalidTransitionError, TwinrEvent, TwinrStatus
from twinr.agent.workflows.forensics import workflow_event
from twinr.ops.events import compact_text

_LOGGER = logging.getLogger(__name__)

class TwinrRuntimeFlowMixin:
    """Provide serialized runtime flow mutations and user-turn helpers."""

    _DEFAULT_BUTTON_DEBOUNCE_MS = 250
    _DEFAULT_CONTENT_PREVIEW_MAX_CHARS = 240
    _DEFAULT_FIELD_LIMITS = {
        "request_source": 64,
        "button": 32,
        "proactive_trigger": 128,
        "source": 32,
        "modality": 32,
        "transcript": 8_192,
        "answer": 16_384,
        "prompt": 8_192,
        "error_message": 200,
    }

    def _runtime_flow_state(self) -> dict[str, Any]:
        state = getattr(self, "_twinr_runtime_flow_state", None)
        if isinstance(state, dict):
            return state

        state = {
            "lock": RLock(),
            "event_sequence": 0,
            "active_turn_id": None,
            "active_parent_turn_id": None,
            "last_printable_output": getattr(self, "last_response", None),
            "last_printable_turn_id": None,
            "recent_actions_ns": {},
        }
        try:
            setattr(self, "_twinr_runtime_flow_state", state)
            return state
        except Exception as exc:
            raise RuntimeError("Twinr runtime flow state could not be attached to the runtime instance.") from exc

    def _runtime_flow_lock(self) -> RLock:
        return cast(RLock, self._runtime_flow_state()["lock"])

    @staticmethod
    def _monotonic_ns() -> int:
        return time.monotonic_ns()

    @staticmethod
    def _unix_ms() -> int:
        return time.time_ns() // 1_000_000

    @staticmethod
    def _elapsed_ms(start_ns: int, *, end_ns: int | None = None) -> float:
        finished_ns = TwinrRuntimeFlowMixin._monotonic_ns() if end_ns is None else end_ns
        return round((finished_ns - start_ns) / 1_000_000.0, 3)

    def _field_limit(self, field_name: str) -> int:
        raw_limit = getattr(self, f"runtime_max_{field_name}_chars", None)
        if raw_limit is None:
            raw_limit = self._DEFAULT_FIELD_LIMITS.get(field_name, 4_096)
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            limit = self._DEFAULT_FIELD_LIMITS.get(field_name, 4_096)
        return max(8, limit)

    def _button_debounce_ms(self) -> int:
        raw_value = getattr(self, "runtime_button_debounce_ms", self._DEFAULT_BUTTON_DEBOUNCE_MS)
        try:
            debounce_ms = int(raw_value)
        except (TypeError, ValueError):
            debounce_ms = self._DEFAULT_BUTTON_DEBOUNCE_MS
        return max(0, min(debounce_ms, 5_000))

    def _content_preview_limit(self) -> int:
        raw_value = getattr(
            self,
            "runtime_content_preview_max_chars",
            self._DEFAULT_CONTENT_PREVIEW_MAX_CHARS,
        )
        try:
            preview_limit = int(raw_value)
        except (TypeError, ValueError):
            preview_limit = self._DEFAULT_CONTENT_PREVIEW_MAX_CHARS
        return max(16, preview_limit)

    @staticmethod
    def _strip_nonprintable(value: str, *, preserve_layout: bool) -> str:
        safe_characters: list[str] = []
        for character in value:
            if character.isprintable():
                safe_characters.append(character)
            elif preserve_layout and character in {"\n", "\r", "\t"}:
                safe_characters.append(character)
            else:
                safe_characters.append(" ")
        return "".join(safe_characters)

    @staticmethod
    def _truncate_text(value: str, *, max_chars: int) -> str:
        if len(value) <= max_chars:
            return value
        if max_chars <= 1:
            return value[:max_chars]
        return f"{value[: max_chars - 1].rstrip()}…"

    def _normalize_text(
        self,
        value: str,
        *,
        field_name: str,
        collapse_whitespace: bool,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string")
        cleaned = self._strip_nonprintable(value, preserve_layout=not collapse_whitespace)
        cleaned = " ".join(cleaned.split()) if collapse_whitespace else cleaned.strip()
        if not cleaned:
            raise ValueError(f"{field_name} must not be empty")
        return self._truncate_text(cleaned, max_chars=self._field_limit(field_name))

    def _require_text(self, value: str, *, field_name: str) -> str:
        collapse_whitespace = field_name in {
            "request_source",
            "button",
            "proactive_trigger",
            "source",
            "modality",
            "error_message",
        }
        return self._normalize_text(
            value,
            field_name=field_name,
            collapse_whitespace=collapse_whitespace,
        )

    def _optional_text(self, value: str | None, *, field_name: str) -> str | None:
        if value is None:
            return None
        return self._require_text(value, field_name=field_name)

    def _sanitize_error_message(self, message: str) -> str:
        try:
            return self._require_text(message, field_name="error_message")
        except (TypeError, ValueError):
            return "Runtime error"

    def _ops_content_preview_enabled(self) -> bool:
        raw_value = getattr(self, "ops_event_include_content_preview", False)
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, str):
            return raw_value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(raw_value)

    def _content_preview(self, text: str) -> str:
        if not self._ops_content_preview_enabled():
            return "[redacted]"
        preview = compact_text(self._truncate_text(text, max_chars=self._field_limit("prompt")))
        return self._truncate_text(preview, max_chars=self._content_preview_limit())

    def _persist_snapshot_safe(self, *, error_message: str | None = None) -> None:
        try:
            if error_message is None:
                self._persist_snapshot()
            else:
                self._persist_snapshot(error_message=error_message)
        except Exception as exc:
            _LOGGER.exception("Failed to persist Twinr runtime snapshot.")
            raise RuntimeError("Failed to persist Twinr runtime state.") from exc

    def _append_ops_event(self, **kwargs: object) -> None:
        try:
            self.ops_events.append(**kwargs)
        except Exception:
            _LOGGER.exception("Failed to append Twinr ops event %r.", kwargs.get("event"))

    def _workflow_event_safe(self, **kwargs: object) -> None:
        try:
            workflow_event(**kwargs)
        except Exception:
            _LOGGER.exception("Failed to emit Twinr workflow event %r.", kwargs.get("msg"))

    def _enqueue_multimodal_evidence_safe(self, **kwargs: object) -> None:
        try:
            self.long_term_memory.enqueue_multimodal_evidence(**kwargs)
        except Exception:
            _LOGGER.exception("Failed to enqueue Twinr multimodal evidence.")

    def _enqueue_conversation_turn_safe(
        self,
        *,
        transcript: str,
        response: str,
        source: str = "conversation",
        modality: str = "voice",
    ) -> None:
        try:
            self.long_term_memory.enqueue_conversation_turn(
                transcript=transcript,
                response=response,
                source=source,
                modality=modality,
            )
        except Exception:
            _LOGGER.exception("Failed to enqueue Twinr conversation turn.")

    def _validated_last_transcript(self) -> str:
        transcript = getattr(self, "last_transcript", None)
        if not isinstance(transcript, str) or not transcript.strip():
            raise RuntimeError("No transcript is available for response generation")
        return transcript.strip()

    def _new_runtime_id(self) -> str:
        uuid7_fn = getattr(uuid, "uuid7", None)
        if callable(uuid7_fn):
            try:
                # pylint: disable-next=not-callable
                return str(cast(Any, uuid7_fn)())
            except Exception:
                _LOGGER.exception("Failed to generate uuid7 correlation id; falling back to uuid4.")
        return str(uuid.uuid4())

    def _current_turn_id_locked(self) -> str | None:
        state = self._runtime_flow_state()
        turn_id = state.get("active_turn_id")
        return turn_id if isinstance(turn_id, str) and turn_id else None

    def _set_active_turn_locked(self, turn_id: str, *, parent_turn_id: str | None = None) -> str:
        state = self._runtime_flow_state()
        state["active_turn_id"] = turn_id
        state["active_parent_turn_id"] = parent_turn_id
        return turn_id

    def _ensure_active_turn_id_locked(self) -> str:
        turn_id = self._current_turn_id_locked()
        if turn_id is not None:
            return turn_id
        return self._set_active_turn_locked(self._new_runtime_id(), parent_turn_id=None)

    def _start_new_turn_locked(self, *, parent_turn_id: str | None = None) -> str:
        return self._set_active_turn_locked(self._new_runtime_id(), parent_turn_id=parent_turn_id)

    def _mark_last_printable_output_locked(self, text: str) -> None:
        state = self._runtime_flow_state()
        state["last_printable_output"] = text
        state["last_printable_turn_id"] = state.get("active_turn_id")

    def _clear_last_printable_output_locked(self) -> None:
        state = self._runtime_flow_state()
        state["last_printable_output"] = None
        state["last_printable_turn_id"] = None

    def _validated_last_printable_output(self) -> tuple[str, str | None]:
        state = self._runtime_flow_state()
        printable = state.get("last_printable_output")
        if isinstance(printable, str) and printable.strip():
            printable_turn_id = state.get("last_printable_turn_id")
            return printable, printable_turn_id if isinstance(printable_turn_id, str) else None

        last_response = getattr(self, "last_response", None)
        if isinstance(last_response, str) and last_response.strip():
            active_turn_id = state.get("active_turn_id")
            return last_response, active_turn_id if isinstance(active_turn_id, str) else None

        raise RuntimeError("No assistant response is available for printing")

    def _set_visible_status_override_locked(self, status: TwinrStatus | None) -> None:
        state = self._runtime_flow_state()
        if status is None:
            state.pop("visible_status_override", None)
            return
        state["visible_status_override"] = status.value

    def _runtime_visible_status_override(self) -> TwinrStatus | None:
        state_getter = getattr(self, "_runtime_flow_state", None)
        if not callable(state_getter):
            return None
        try:
            state = state_getter()
        except Exception:
            return None
        if not isinstance(state, dict):
            return None
        override = state.get("visible_status_override")
        if not isinstance(override, str):
            return None
        if override != TwinrStatus.PRINTING.value:
            return None
        if not getattr(self.state_machine, "printing_active", False):
            return None
        return TwinrStatus.PRINTING

    def _should_coalesce_action_locked(self, action_key: str) -> bool:
        debounce_ms = self._button_debounce_ms()
        if debounce_ms <= 0:
            return False

        state = self._runtime_flow_state()
        recent_actions = state.setdefault("recent_actions_ns", {})
        if not isinstance(recent_actions, dict):
            recent_actions = {}
            state["recent_actions_ns"] = recent_actions

        now_ns = self._monotonic_ns()
        last_seen_ns = recent_actions.get(action_key)
        recent_actions[action_key] = now_ns

        current_status = self.state_machine.status
        if current_status == TwinrStatus.WAITING:
            return False

        return isinstance(last_seen_ns, int) and (now_ns - last_seen_ns) < (debounce_ms * 1_000_000)

    def _ops_event_payload_locked(
        self,
        *,
        status_before: str,
        status_after: str,
        turn_id: str | None = None,
        parent_turn_id: str | None = None,
        **extra: object,
    ) -> dict[str, object]:
        state = self._runtime_flow_state()
        state["event_sequence"] = int(state.get("event_sequence", 0)) + 1

        active_turn_id = turn_id if turn_id is not None else state.get("active_turn_id")
        active_parent_turn_id = (
            parent_turn_id
            if parent_turn_id is not None
            else state.get("active_parent_turn_id")
        )

        payload: dict[str, object] = {
            "event_id": self._new_runtime_id(),
            "event_seq": state["event_sequence"],
            "ts_unix_ms": self._unix_ms(),
            "ts_monotonic_ns": self._monotonic_ns(),
            "status_before": status_before,
            "status": status_after,
        }
        if isinstance(active_turn_id, str) and active_turn_id:
            payload["turn_id"] = active_turn_id
        if isinstance(active_parent_turn_id, str) and active_parent_turn_id:
            payload["parent_turn_id"] = active_parent_turn_id

        for key, value in extra.items():
            if value is not None:
                payload[key] = value
        return payload

    def refresh_snapshot_activity(self) -> None:
        """Refresh the runtime snapshot timestamp without changing state.

        Long-running playback can legitimately outlive the supervisor snapshot
        age budget. This keeps the current runtime state visible without
        pretending that a new state transition happened.
        """

        lock = self._runtime_flow_lock()
        lock_wait_started_ns = self._monotonic_ns()
        with lock:
            lock_wait_ms = self._elapsed_ms(lock_wait_started_ns)
            refresh_started_ns = self._monotonic_ns()
            status_value = self.state_machine.status.value
            turn_id = self._current_turn_id_locked()
            self._persist_snapshot_safe()
            duration_ms = self._elapsed_ms(refresh_started_ns)

        self._workflow_event_safe(
            kind="metric",
            msg="runtime_snapshot_activity_refreshed",
            details={"status": status_value, "turn_id": turn_id},
            kpi={"lock_wait_ms": lock_wait_ms, "duration_ms": duration_ms},
        )

    def _store_agent_turn(
        self,
        cleaned_answer: str,
        *,
        transition_to_answering: bool,
        source: str = "conversation",
        modality: str = "voice",
    ) -> str:
        normalized_source = self._require_text(source, field_name="source")
        normalized_modality = self._require_text(modality, field_name="modality")

        lock = self._runtime_flow_lock()
        lock_wait_started_ns = self._monotonic_ns()
        with lock:
            lock_wait_ms = self._elapsed_ms(lock_wait_started_ns)
            transcript = self._validated_last_transcript()
            turn_id = self._ensure_active_turn_id_locked()
            status_before = self.state_machine.status.value
            if transition_to_answering:
                status_after = self.state_machine.transition(TwinrEvent.RESPONSE_READY).value
            else:
                status_after = status_before

            memory_started_ns = self._monotonic_ns()
            self.memory.remember("user", transcript)
            self.memory.remember("assistant", cleaned_answer)
            self.last_response = cleaned_answer
            self._mark_last_printable_output_locked(cleaned_answer)
            memory_duration_ms = self._elapsed_ms(memory_started_ns)

            snapshot_started_ns = self._monotonic_ns()
            self._persist_snapshot_safe()
            snapshot_duration_ms = self._elapsed_ms(snapshot_started_ns)

            event_data = self._ops_event_payload_locked(
                status_before=status_before,
                status_after=status_after,
                turn_id=turn_id,
                transcript_preview=self._content_preview(transcript),
                transcript_chars=len(transcript),
                response_preview=self._content_preview(cleaned_answer),
                response_chars=len(cleaned_answer),
                source=normalized_source,
                modality=normalized_modality,
            )

        self._workflow_event_safe(
            kind="metric",
            msg="runtime_store_agent_turn_lock_acquired",
            details={
                "transition_to_answering": transition_to_answering,
                "turn_id": turn_id,
            },
            kpi={"lock_wait_ms": lock_wait_ms},
        )
        self._workflow_event_safe(
            kind="metric",
            msg="runtime_store_agent_turn_memory_updated",
            details={
                "transition_to_answering": transition_to_answering,
                "turn_id": turn_id,
                "transcript_chars": len(transcript),
                "response_chars": len(cleaned_answer),
            },
            kpi={"duration_ms": memory_duration_ms},
        )
        self._workflow_event_safe(
            kind="metric",
            msg="runtime_store_agent_turn_snapshot_persisted",
            details={
                "transition_to_answering": transition_to_answering,
                "turn_id": turn_id,
            },
            kpi={"duration_ms": snapshot_duration_ms},
        )

        enqueue_started_ns = self._monotonic_ns()
        self._enqueue_conversation_turn_safe(
            transcript=transcript,
            response=cleaned_answer,
            source=normalized_source,
            modality=normalized_modality,
        )
        enqueue_duration_ms = self._elapsed_ms(enqueue_started_ns)
        self._workflow_event_safe(
            kind="metric",
            msg="runtime_store_agent_turn_longterm_enqueued",
            details={
                "transition_to_answering": transition_to_answering,
                "turn_id": turn_id,
            },
            kpi={"duration_ms": enqueue_duration_ms},
        )

        ops_started_ns = self._monotonic_ns()
        self._append_ops_event(
            event="turn_completed",
            message="Assistant response stored in runtime memory.",
            data=event_data,
        )
        ops_duration_ms = self._elapsed_ms(ops_started_ns)
        self._workflow_event_safe(
            kind="metric",
            msg="runtime_store_agent_turn_ops_appended",
            details={
                "transition_to_answering": transition_to_answering,
                "turn_id": turn_id,
            },
            kpi={"duration_ms": ops_duration_ms},
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

        with self._runtime_flow_lock():
            status_before = self.state_machine.status.value
            turn_id = self._new_runtime_id()
            status = self.state_machine.transition(TwinrEvent.GREEN_BUTTON_PRESSED)
            self._set_active_turn_locked(turn_id, parent_turn_id=None)
            self.last_transcript = ""
            self._persist_snapshot_safe()

            data = self._ops_event_payload_locked(
                status_before=status_before,
                status_after=status.value,
                turn_id=turn_id,
                request_source=normalized_request_source,
                button=normalized_button,
                proactive_trigger=normalized_trigger,
            )

        self._append_ops_event(
            event="turn_started",
            message="Conversation listening window started.",
            data=data,
        )
        return status

    def press_green_button(self) -> TwinrStatus:
        """Start a listening turn from the physical green button."""

        coalesced = False
        event_data: dict[str, object] | None = None

        with self._runtime_flow_lock():
            if self._should_coalesce_action_locked("button:green:start_listening"):
                coalesced = True
                current_status = self.state_machine.status
                current_turn_id = self._ensure_active_turn_id_locked()
                event_data = self._ops_event_payload_locked(
                    status_before=current_status.value,
                    status_after=current_status.value,
                    turn_id=current_turn_id,
                    button="green",
                    request_source="button",
                    coalesced=True,
                )
                status = current_status
            else:
                status_before = self.state_machine.status.value
                turn_id = self._new_runtime_id()
                status = self.state_machine.transition(TwinrEvent.GREEN_BUTTON_PRESSED)
                self._set_active_turn_locked(turn_id, parent_turn_id=None)
                self.last_transcript = ""
                self._persist_snapshot_safe()
                event_data = self._ops_event_payload_locked(
                    status_before=status_before,
                    status_after=status.value,
                    turn_id=turn_id,
                    button="green",
                    request_source="button",
                )

        if coalesced:
            self._append_ops_event(
                event="turn_started",
                message="Duplicate green button press was coalesced.",
                data=event_data,
            )
            return status

        self._append_ops_event(
            event="turn_started",
            message="Conversation listening window started.",
            data=event_data,
        )
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

        cleaned_transcript = self._require_text(transcript, field_name="transcript")

        with self._runtime_flow_lock():
            turn_id = self._ensure_active_turn_id_locked()
            status_before = self.state_machine.status.value
            status = self.state_machine.transition(TwinrEvent.SPEECH_PAUSE_DETECTED)
            self.last_transcript = cleaned_transcript
            self._persist_snapshot_safe()

            event_data = self._ops_event_payload_locked(
                status_before=status_before,
                status_after=status.value,
                turn_id=turn_id,
                transcript_preview=self._content_preview(cleaned_transcript),
                transcript_chars=len(cleaned_transcript),
            )

        self._append_ops_event(
            event="transcript_submitted",
            message="Transcript captured for the active turn.",
            data=event_data,
        )
        return status

    def begin_answering(self) -> TwinrStatus:
        """Move the runtime into the answering state without storing a reply."""

        with self._runtime_flow_lock():
            turn_id = self._ensure_active_turn_id_locked()
            status_before = self.state_machine.status.value
            status = self.state_machine.transition(TwinrEvent.RESPONSE_READY)
            self._persist_snapshot_safe()
            event_data = self._ops_event_payload_locked(
                status_before=status_before,
                status_after=status.value,
                turn_id=turn_id,
            )

        self._append_ops_event(
            event="answering_started",
            message="Twinr entered the answering state.",
            data=event_data,
        )
        return status

    def resume_processing(self) -> TwinrStatus:
        """Return from a spoken bridge acknowledgement back to processing."""

        with self._runtime_flow_lock():
            turn_id = self._ensure_active_turn_id_locked()
            status_before = self.state_machine.status.value
            status = self.state_machine.transition(TwinrEvent.PROCESSING_RESUMED)
            self._persist_snapshot_safe()
            event_data = self._ops_event_payload_locked(
                status_before=status_before,
                status_after=status.value,
                turn_id=turn_id,
            )

        self._append_ops_event(
            event="processing_resumed",
            message="Twinr resumed processing after the spoken acknowledgement.",
            data=event_data,
        )
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

    def begin_voice_activation_prompt(self, prompt: str) -> str:
        """Start the remote voice-activation acknowledgement prompt."""

        return self._begin_background_prompt(
            prompt,
            event="voice_activation_prompt_started",
            message="Twinr acknowledged a remote voice activation before hands-free listening.",
        )

    def _begin_background_prompt(self, prompt: str, *, event: str, message: str) -> str:
        spoken_prompt = self._require_text(prompt, field_name="prompt")

        with self._runtime_flow_lock():
            turn_id = self._new_runtime_id()
            status_before = self.state_machine.status.value
            status = self.state_machine.transition(TwinrEvent.PROACTIVE_PROMPT_READY)
            self._set_active_turn_locked(turn_id, parent_turn_id=None)
            self.memory.remember("assistant", spoken_prompt)
            self._mark_last_printable_output_locked(spoken_prompt)
            self._persist_snapshot_safe()

            event_data = self._ops_event_payload_locked(
                status_before=status_before,
                status_after=status.value,
                turn_id=turn_id,
                response_preview=self._content_preview(spoken_prompt),
                response_chars=len(spoken_prompt),
            )

        self._append_ops_event(
            event=event,
            message=message,
            data=event_data,
        )
        return spoken_prompt

    def cancel_listening(self) -> TwinrStatus:
        """Abort the active listening turn and clear stale transcript state."""

        with self._runtime_flow_lock():
            turn_id = self._ensure_active_turn_id_locked()
            status_before = self.state_machine.status.value
            status = self.state_machine.reset()
            self.last_transcript = ""
            self._persist_snapshot_safe()
            event_data = self._ops_event_payload_locked(
                status_before=status_before,
                status_after=status.value,
                turn_id=turn_id,
            )

        self._append_ops_event(
            event="turn_cancelled",
            message="Twinr cancelled the active listening turn.",
            data=event_data,
        )
        return status

    def finalize_agent_turn(
        self,
        answer: str,
        *,
        source: str = "conversation",
        modality: str = "voice",
    ) -> str:
        """Store an assistant answer without changing the current status."""

        cleaned_answer = self._require_text(answer, field_name="answer")
        return self._store_agent_turn(
            cleaned_answer,
            transition_to_answering=False,
            source=source,
            modality=modality,
        )

    def finalize_interrupted_turn(
        self,
        *,
        source: str = "conversation",
        modality: str = "voice",
    ) -> None:
        """Persist the user turn when an unheard assistant reply is discarded."""

        normalized_source = self._require_text(source, field_name="source")
        normalized_modality = self._require_text(modality, field_name="modality")

        with self._runtime_flow_lock():
            transcript = self._validated_last_transcript()
            turn_id = self._ensure_active_turn_id_locked()
            status_before = self.state_machine.status.value
            self.memory.remember("user", transcript)
            self.last_response = None
            self._clear_last_printable_output_locked()
            self._persist_snapshot_safe()
            event_data = self._ops_event_payload_locked(
                status_before=status_before,
                status_after=status_before,
                turn_id=turn_id,
                transcript_preview=self._content_preview(transcript),
                transcript_chars=len(transcript),
                source=normalized_source,
                modality=normalized_modality,
                interrupted=True,
            )

        self._append_ops_event(
            event="interrupted_turn_recorded",
            message="Twinr recorded the user turn after discarding an interrupted assistant reply.",
            data=event_data,
        )

    def complete_agent_turn(
        self,
        answer: str,
        *,
        source: str = "conversation",
        modality: str = "voice",
    ) -> str:
        """Store an assistant answer and transition into answering."""

        cleaned_answer = self._require_text(answer, field_name="answer")
        return self._store_agent_turn(
            cleaned_answer,
            transition_to_answering=True,
            source=source,
            modality=modality,
        )

    def finish_speaking(self) -> TwinrStatus:
        """Mark the spoken response as finished and return toward idle."""

        with self._runtime_flow_lock():
            turn_id = self._ensure_active_turn_id_locked()
            status_before = self.state_machine.status.value
            status = self.state_machine.transition(TwinrEvent.TTS_FINISHED)
            self.last_transcript = ""
            self._persist_snapshot_safe()
            event_data = self._ops_event_payload_locked(
                status_before=status_before,
                status_after=status.value,
                turn_id=turn_id,
            )

        self._append_ops_event(
            event="tts_finished",
            message="Twinr finished speaking the response.",
            data=event_data,
        )
        return status

    def finish_speaking_and_rearm_follow_up(
        self,
        *,
        request_source: str = "follow_up",
    ) -> TwinrStatus:
        """Atomically clear speaking and reopen follow-up listening.

        Transcript-first remote follow-up already keeps the authoritative
        server-side window open while playback ends. Persisting `waiting`
        immediately before reopening `listening` only adds Pi-side snapshot I/O
        and visible lag, so this helper preserves the state-machine transitions
        and ops events while persisting only the final reopened state.
        """

        normalized_request_source = self._require_text(request_source, field_name="request_source")

        with self._runtime_flow_lock():
            parent_turn_id = self._ensure_active_turn_id_locked()
            status_before = self.state_machine.status.value
            waiting_status = self.state_machine.transition(TwinrEvent.TTS_FINISHED)
            waiting_event_data = self._ops_event_payload_locked(
                status_before=status_before,
                status_after=waiting_status.value,
                turn_id=parent_turn_id,
            )

            turn_id = self._new_runtime_id()
            listening_status = self.state_machine.transition(TwinrEvent.FOLLOW_UP_ARMED)
            self._set_active_turn_locked(turn_id, parent_turn_id=parent_turn_id)
            self.last_transcript = ""
            self._persist_snapshot_safe()
            follow_up_event_data = self._ops_event_payload_locked(
                status_before=waiting_status.value,
                status_after=listening_status.value,
                turn_id=turn_id,
                parent_turn_id=parent_turn_id,
                request_source=normalized_request_source,
            )

        self._append_ops_event(
            event="tts_finished",
            message="Twinr finished speaking the response.",
            data=waiting_event_data,
        )
        self._append_ops_event(
            event="follow_up_rearmed",
            message="Conversation follow-up listening window opened immediately after speech.",
            data=follow_up_event_data,
        )
        return listening_status

    def rearm_follow_up(
        self,
        *,
        request_source: str = "follow_up",
    ) -> TwinrStatus:
        """Re-open listening for a conversation follow-up turn.

        When the closure decision already finished during playback, the runtime
        can move straight from ``answering`` back to ``listening``. If closure
        evaluation is still running after audio drained, the coordinator may
        first clear ``speaking`` via ``waiting`` and then re-open listening
        once the follow-up gate resolves.
        """

        normalized_request_source = self._require_text(request_source, field_name="request_source")

        with self._runtime_flow_lock():
            parent_turn_id = self._ensure_active_turn_id_locked()
            status_before = self.state_machine.status.value
            turn_id = self._new_runtime_id()
            status = self.state_machine.transition(TwinrEvent.FOLLOW_UP_ARMED)
            self._set_active_turn_locked(turn_id, parent_turn_id=parent_turn_id)
            self.last_transcript = ""
            self._persist_snapshot_safe()
            event_data = self._ops_event_payload_locked(
                status_before=status_before,
                status_after=status.value,
                turn_id=turn_id,
                parent_turn_id=parent_turn_id,
                request_source=normalized_request_source,
            )

        self._append_ops_event(
            event="follow_up_rearmed",
            message="Conversation follow-up listening window opened immediately after speech.",
            data=event_data,
        )
        return status

    def press_yellow_button(self) -> str:
        """Request printing of the last assistant response."""

        coalesced = False
        with self._runtime_flow_lock():
            printable_output, printable_turn_id = self._validated_last_printable_output()
            if self._should_coalesce_action_locked("button:yellow:print_request"):
                coalesced = True
                status_before = self.state_machine.status.value
                event_data = self._ops_event_payload_locked(
                    status_before=status_before,
                    status_after=status_before,
                    turn_id=printable_turn_id,
                    button="yellow",
                    request_source="button",
                    response_preview=self._content_preview(printable_output),
                    response_chars=len(printable_output),
                    coalesced=True,
                )
            else:
                status_before = self.state_machine.status.value
                status = self.state_machine.transition(TwinrEvent.YELLOW_BUTTON_PRESSED)
                self._persist_snapshot_safe()
                event_data = self._ops_event_payload_locked(
                    status_before=status_before,
                    status_after=status.value,
                    turn_id=printable_turn_id,
                    button="yellow",
                    request_source="button",
                    response_preview=self._content_preview(printable_output),
                    response_chars=len(printable_output),
                )

        if coalesced:
            self._append_ops_event(
                event="print_started",
                message="Duplicate yellow button press was coalesced.",
                data=event_data,
            )
            return printable_output

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
        return printable_output

    def prepare_background_button_print_request(self) -> str:
        """Queue a background print request without changing runtime state."""

        coalesced = False
        with self._runtime_flow_lock():
            printable_output, printable_turn_id = self._validated_last_printable_output()
            status_value = self.state_machine.status.value
            if self._should_coalesce_action_locked("button:yellow:background_print_request"):
                coalesced = True
            event_data = self._ops_event_payload_locked(
                status_before=status_value,
                status_after=status_value,
                turn_id=printable_turn_id,
                button="yellow",
                request_source="button",
                background=True,
                response_preview=self._content_preview(printable_output),
                response_chars=len(printable_output),
                coalesced=coalesced or None,
            )

        self._append_ops_event(
            event="print_started",
            message=(
                "Duplicate yellow button background print request was coalesced."
                if coalesced
                else "Yellow button queued a background print."
            ),
            data=event_data,
        )
        if not coalesced:
            self._enqueue_multimodal_evidence_safe(
                event_name="button_interaction",
                modality="button",
                source="runtime_button",
                message="Yellow button queued a printed answer in the background.",
                data={"button": "yellow", "action": "background_print_request"},
            )
        return printable_output

    def begin_tool_print(self) -> TwinrStatus:
        """Transition into printing for a tool-requested print job."""

        with self._runtime_flow_lock():
            turn_id = self._ensure_active_turn_id_locked()
            status_before = self.state_machine.status.value
            status = self.state_machine.transition(TwinrEvent.PRINT_REQUESTED)
            self._set_visible_status_override_locked(TwinrStatus.PRINTING)
            self._persist_snapshot_safe()
            event_data = self._ops_event_payload_locked(
                status_before=status_before,
                status_after=self.status.value,
                turn_id=turn_id,
                request_source="tool",
            )

        self._append_ops_event(
            event="print_started",
            message="Assistant tool requested a print.",
            data=event_data,
        )
        return status

    def resume_answering_after_print(self) -> TwinrStatus:
        """Return from printing to the answering state."""

        with self._runtime_flow_lock():
            turn_id = self._ensure_active_turn_id_locked()
            status_before = self.status.value
            status = self.state_machine.transition(TwinrEvent.RESPONSE_READY)
            self._set_visible_status_override_locked(None)
            self._persist_snapshot_safe()
            event_data = self._ops_event_payload_locked(
                status_before=status_before,
                status_after=self.status.value,
                turn_id=turn_id,
            )

        self._append_ops_event(
            event="answering_resumed",
            message="Twinr returned from printing to answering.",
            data=event_data,
        )
        return status

    def maybe_begin_tool_print(self) -> TwinrStatus | None:
        """Attempt a tool print transition and return None if disallowed."""

        try:
            return self.begin_tool_print()
        except InvalidTransitionError:
            return None

    def begin_automation_print(self) -> TwinrStatus:
        """Transition into printing for a scheduled automation output."""

        with self._runtime_flow_lock():
            turn_id = self._ensure_active_turn_id_locked()
            status_before = self.state_machine.status.value
            try:
                status = self.state_machine.transition(TwinrEvent.PRINT_REQUESTED)
            except InvalidTransitionError:
                if self.state_machine.status != TwinrStatus.WAITING:
                    raise
                status = self.state_machine.transition(TwinrEvent.YELLOW_BUTTON_PRESSED)
            self._set_visible_status_override_locked(TwinrStatus.PRINTING)
            self._persist_snapshot_safe()
            event_data = self._ops_event_payload_locked(
                status_before=status_before,
                status_after=self.status.value,
                turn_id=turn_id,
                request_source="automation",
            )

        self._append_ops_event(
            event="print_started",
            message="Scheduled automation requested a print.",
            data=event_data,
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

        with self._runtime_flow_lock():
            turn_id = self._ensure_active_turn_id_locked()
            status_before = self.status.value
            status = self.state_machine.transition(TwinrEvent.PRINT_FINISHED)
            self._set_visible_status_override_locked(None)
            self._persist_snapshot_safe()
            event_data = self._ops_event_payload_locked(
                status_before=status_before,
                status_after=self.status.value,
                turn_id=turn_id,
            )

        self._append_ops_event(
            event="print_finished",
            message="Twinr returned to idle after printing.",
            data=event_data,
        )
        return status

    def fail(self, message: str) -> TwinrStatus:
        """Enter the error state with a sanitized operator-safe message."""

        safe_message = self._sanitize_error_message(message)

        with self._runtime_flow_lock():
            turn_id = self._ensure_active_turn_id_locked()
            status_before = self.state_machine.status.value
            status = self.state_machine.fail(safe_message)
            self.last_transcript = ""
            self._persist_snapshot_safe(error_message=safe_message)
            event_data = self._ops_event_payload_locked(
                status_before=status_before,
                status_after=status.value,
                turn_id=turn_id,
                error=safe_message,
            )

        self._append_ops_event(
            event="error",
            level="error",
            message="Twinr runtime entered the error state.",
            data=event_data,
        )
        return status

    def reset_error(self) -> TwinrStatus:
        """Leave the error state and clear transient turn state."""

        with self._runtime_flow_lock():
            turn_id = self._ensure_active_turn_id_locked()
            status_before = self.state_machine.status.value
            status = self.state_machine.reset()
            self.last_transcript = ""
            self._persist_snapshot_safe()
            event_data = self._ops_event_payload_locked(
                status_before=status_before,
                status_after=status.value,
                turn_id=turn_id,
            )

        self._append_ops_event(
            event="error_reset",
            message="Twinr runtime left the error state.",
            data=event_data,
        )
        return status
