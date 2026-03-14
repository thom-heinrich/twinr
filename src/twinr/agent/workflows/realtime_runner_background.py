from __future__ import annotations

from datetime import datetime
from queue import Empty
import time
from zoneinfo import ZoneInfo

from twinr.agent.tools.automations import normalize_delivery
from twinr.automations import AutomationAction, AutomationDefinition
from twinr.memory.reminders import format_due_label
from twinr.proactive import (
    ProactiveGovernorCandidate,
    ProactiveGovernorReservation,
    SocialTriggerDecision,
    is_safety_trigger,
    proactive_observation_facts,
    proactive_prompt_mode,
)
from twinr.providers.openai.backend import REMINDER_DELIVERY_INSTRUCTIONS


class TwinrRealtimeBackgroundMixin:
    def handle_social_trigger(self, trigger: SocialTriggerDecision) -> bool:
        if not self._background_work_allowed():
            skip_reason = "busy" if self.runtime.status.value != "waiting" else "conversation_active"
            self.emit(f"social_trigger_skipped={skip_reason}")
            self._record_event(
                "social_trigger_skipped",
                "Social trigger prompt was skipped because Twinr was not idle for background prompts.",
                trigger=trigger.trigger_id,
                reason=trigger.reason,
                prompt=trigger.prompt,
                priority=int(trigger.priority),
                skip_reason=skip_reason,
            )
            return False

        governor_reservation = self._reserve_governed_prompt(
            ProactiveGovernorCandidate(
                source_kind="social",
                source_id=trigger.trigger_id,
                summary=trigger.prompt,
                priority=int(trigger.priority),
                presence_session_id=self._current_presence_session_id(),
                safety_exempt=is_safety_trigger(trigger.trigger_id),
                counts_toward_presence_budget=not is_safety_trigger(trigger.trigger_id),
            )
        )
        if governor_reservation is None:
            return False

        phrase_response = None
        prompt_mode = proactive_prompt_mode(trigger)
        prompt_text = trigger.prompt
        try:
            if prompt_mode == "llm":
                stop_processing_feedback = self._start_working_feedback_loop("processing")
                try:
                    try:
                        phrase_response = self.agent_provider.phrase_proactive_prompt_with_metadata(
                            trigger_id=trigger.trigger_id,
                            reason=trigger.reason,
                            default_prompt=trigger.prompt,
                            priority=int(trigger.priority),
                            conversation=self.runtime.conversation_context(),
                            recent_prompts=self._recent_proactive_prompts(trigger_id=trigger.trigger_id),
                            observation_facts=proactive_observation_facts(trigger),
                        )
                    finally:
                        stop_processing_feedback()
                    candidate_prompt = phrase_response.text.strip()
                    if candidate_prompt:
                        prompt_text = candidate_prompt
                    else:
                        prompt_mode = "default_fallback"
                        self.emit("social_prompt_fallback=empty_phrase")
                except Exception as exc:
                    prompt_mode = "default_fallback"
                    self.emit("social_prompt_fallback=default")
                    self.emit(f"social_prompt_phrase_error={exc}")
                    self._record_event(
                        "social_trigger_phrase_fallback",
                        "Twinr fell back to the default proactive prompt after proactive phrasing failed.",
                        level="warning",
                        trigger=trigger.trigger_id,
                        error=str(exc),
                    )

            prompt = self.runtime.begin_proactive_prompt(prompt_text)
            self._emit_status(force=True)
            tts_started = time.monotonic()
            tts_ms, first_audio_ms = self._play_streaming_tts_with_feedback(prompt, turn_started=tts_started)
            self.runtime.finish_speaking()
            self._emit_status(force=True)
            self.runtime.proactive_governor.mark_delivered(
                governor_reservation,
            )
            self.emit(f"social_trigger={trigger.trigger_id}")
            self.emit(f"social_trigger_priority={int(trigger.priority)}")
            self.emit(f"social_prompt_mode={prompt_mode}")
            self.emit(f"social_prompt={prompt}")
            if phrase_response is not None:
                if phrase_response.response_id:
                    self.emit(f"social_response_id={phrase_response.response_id}")
                if phrase_response.request_id:
                    self.emit(f"social_request_id={phrase_response.request_id}")
                self._record_usage(
                    request_kind="proactive_prompt",
                    source="realtime_loop",
                    model=phrase_response.model,
                    response_id=phrase_response.response_id,
                    request_id=phrase_response.request_id,
                    used_web_search=False,
                    token_usage=phrase_response.token_usage,
                    proactive_trigger=trigger.trigger_id,
                )
            self.emit(f"timing_social_tts_ms={tts_ms}")
            if first_audio_ms is not None:
                self.emit(f"timing_social_first_audio_ms={first_audio_ms}")
            self._record_event(
                "social_trigger_prompted",
                "Twinr spoke a proactive social prompt.",
                trigger=trigger.trigger_id,
                reason=trigger.reason,
                priority=int(trigger.priority),
                prompt=prompt,
                default_prompt=trigger.prompt,
                prompt_mode=prompt_mode,
            )
            self._run_proactive_follow_up(trigger)
            return True
        except Exception as exc:
            if self.runtime.status.value == "answering":
                self.runtime.finish_speaking()
                self._emit_status(force=True)
            self.runtime.proactive_governor.mark_skipped(
                governor_reservation,
                reason=f"delivery_failed: {exc}",
            )
            raise

    def handle_sensor_observation(self, facts: dict[str, object], event_names: tuple[str, ...]) -> None:
        self._sensor_observation_queue.put((dict(facts), tuple(event_names)))
        self.runtime.long_term_memory.enqueue_multimodal_evidence(
            event_name="sensor_observation",
            modality="sensor",
            source="proactive_monitor",
            message="Changed multimodal sensor observation recorded.",
            data={
                "facts": dict(facts),
                "event_names": list(event_names),
            },
        )

    def _maybe_deliver_due_reminder(self) -> bool:
        now_monotonic = time.monotonic()
        if now_monotonic < self._next_reminder_check_at:
            return False
        self._next_reminder_check_at = now_monotonic + self.config.reminder_poll_interval_s
        if not self._background_work_allowed():
            return False
        preview_entries = self.runtime.peek_due_reminders(limit=1)
        if not preview_entries:
            return False
        governor_reservation = self._reserve_governed_prompt(
            ProactiveGovernorCandidate(
                source_kind="reminder",
                source_id=preview_entries[0].reminder_id,
                summary=preview_entries[0].summary,
                priority=80,
                presence_session_id=self._current_presence_session_id(),
                safety_exempt=False,
                counts_toward_presence_budget=False,
            )
        )
        if governor_reservation is None:
            return False
        due_entries = self.runtime.reserve_due_reminders(limit=1)
        if not due_entries:
            self.runtime.proactive_governor.cancel(governor_reservation)
            return False
        if due_entries[0].reminder_id != preview_entries[0].reminder_id:
            self.runtime.proactive_governor.cancel(governor_reservation)
            return False
        return self._deliver_due_reminder(due_entries[0], governor_reservation=governor_reservation)

    def _maybe_run_due_automation(self) -> bool:
        now_monotonic = time.monotonic()
        if now_monotonic < self._next_automation_check_at:
            return False
        self._next_automation_check_at = now_monotonic + self.config.automation_poll_interval_s
        if not self._background_work_allowed():
            return False
        due_entries = self.runtime.due_time_automations()
        if not due_entries:
            return False
        return self._run_automation_entry(due_entries[0], trigger_source="time_schedule")

    def _maybe_run_sensor_automation(self) -> bool:
        if not self._background_work_allowed():
            return False
        while True:
            try:
                facts, event_names = self._sensor_observation_queue.get_nowait()
            except Empty:
                return False
            if self._run_matching_sensor_automations(facts=facts, event_names=event_names):
                return True

    def _background_work_allowed(self) -> bool:
        return self.runtime.status.value == "waiting" and not self._conversation_session_active

    def _current_presence_session_id(self) -> int | None:
        monitor = getattr(self, "proactive_monitor", None)
        coordinator = None if monitor is None else getattr(monitor, "coordinator", None)
        snapshot = None if coordinator is None else getattr(coordinator, "latest_presence_snapshot", None)
        if snapshot is None or not getattr(snapshot, "armed", False):
            return None
        return getattr(snapshot, "session_id", None)

    def _reserve_governed_prompt(
        self,
        candidate: ProactiveGovernorCandidate,
    ) -> ProactiveGovernorReservation | None:
        decision = self.runtime.proactive_governor.try_reserve(candidate)
        if decision.allowed:
            return decision.reservation
        emit_prefix = {
            "social": "social_trigger",
            "longterm": "longterm_proactive",
            "reminder": "reminder",
            "automation": "automation",
        }.get(candidate.source_kind, candidate.source_kind)
        self.emit(f"{emit_prefix}_skipped={decision.reason}")
        self._record_event(
            "proactive_governor_blocked",
            "Proactive delivery was blocked by the shared governor policy.",
            source_kind=candidate.source_kind,
            source_id=candidate.source_id,
            summary=candidate.summary,
            reason=decision.reason,
            channel=candidate.channel,
            priority=int(candidate.priority),
            presence_session_id=candidate.presence_session_id,
        )
        return None

    def _recent_proactive_prompts(
        self,
        *,
        trigger_id: str | None = None,
        limit: int = 3,
    ) -> tuple[str, ...]:
        prompts: list[str] = []
        for entry in reversed(self.runtime.ops_events.tail(limit=100)):
            if entry.get("event") not in {"social_trigger_prompted", "longterm_proactive_prompted"}:
                continue
            data = entry.get("data") or {}
            if trigger_id is not None and data.get("trigger") != trigger_id:
                continue
            prompt = str(data.get("prompt", "")).strip()
            if not prompt:
                continue
            prompts.append(prompt)
            if len(prompts) >= limit:
                break
        prompts.reverse()
        return tuple(prompts)

    def _run_matching_sensor_automations(
        self,
        *,
        facts: dict[str, object],
        event_names: tuple[str, ...],
    ) -> bool:
        matched: dict[str, AutomationDefinition] = {}
        for entry in self.runtime.matching_if_then_automations(facts=facts, event_name=None):
            matched[entry.automation_id] = entry
        for event_name in event_names:
            for entry in self.runtime.matching_if_then_automations(facts=facts, event_name=event_name):
                matched[entry.automation_id] = entry
        if not matched:
            return False
        executed_any = False
        for entry in sorted(matched.values(), key=lambda item: item.name.lower()):
            event_label = ",".join(event_names) or "sensor.state"
            executed = self._run_automation_entry(
                entry,
                trigger_source="sensor",
                event_name=event_label,
                facts=facts,
            )
            executed_any = executed or executed_any
        return executed_any

    def _maybe_run_long_term_memory_proactive(self) -> bool:
        now_monotonic = time.monotonic()
        if now_monotonic < self._next_long_term_memory_proactive_check_at:
            return False
        self._next_long_term_memory_proactive_check_at = (
            now_monotonic + self.config.long_term_memory_proactive_poll_interval_s
        )
        if not self._background_work_allowed():
            return False
        preview = self.runtime.preview_long_term_proactive_candidate()
        if preview is None:
            return False
        governor_reservation = self._reserve_governed_prompt(
            ProactiveGovernorCandidate(
                source_kind="longterm",
                source_id=preview.candidate_id,
                summary=preview.summary,
                priority=max(1, min(99, int(preview.confidence * 100))),
                presence_session_id=self._current_presence_session_id(),
                safety_exempt=False,
                counts_toward_presence_budget=True,
            )
        )
        if governor_reservation is None:
            return False
        reservation = self.runtime.reserve_specific_long_term_proactive_candidate(
            preview,
            now=datetime.now(ZoneInfo(self.config.local_timezone_name)),
        )
        if reservation is None or reservation.candidate.candidate_id != preview.candidate_id:
            self.runtime.proactive_governor.cancel(governor_reservation)
            return False
        candidate = reservation.candidate
        response = None
        prompt_mode = "default"
        prompt_text = candidate.summary
        trigger_id = f"longterm:{candidate.candidate_id}"
        try:
            try:
                stop_processing_feedback = self._start_working_feedback_loop("processing")
                try:
                    response = self.agent_provider.phrase_proactive_prompt_with_metadata(
                        trigger_id=trigger_id,
                        reason=candidate.rationale,
                        default_prompt=candidate.summary,
                        priority=max(1, min(99, int(candidate.confidence * 100))),
                        conversation=self.runtime.conversation_context(),
                        recent_prompts=self._recent_proactive_prompts(trigger_id=trigger_id),
                        observation_facts=(
                            f"candidate_kind={candidate.kind}",
                            f"sensitivity={candidate.sensitivity}",
                        ),
                    )
                finally:
                    stop_processing_feedback()
                candidate_prompt = response.text.strip()
                if candidate_prompt:
                    prompt_text = candidate_prompt
                    prompt_mode = "llm"
                else:
                    prompt_mode = "default_fallback"
            except Exception as exc:
                prompt_mode = "default_fallback"
                self.emit("longterm_proactive_phrase_fallback=default")
                self.emit(f"longterm_proactive_phrase_error={exc}")
                self._record_event(
                    "longterm_proactive_phrase_fallback",
                    "Twinr fell back to the default long-term proactive prompt after phrasing failed.",
                    level="warning",
                    candidate_id=candidate.candidate_id,
                    error=str(exc),
                )

            spoken_prompt = self.runtime.begin_proactive_prompt(prompt_text)
            self._emit_status(force=True)
            tts_started = time.monotonic()
            tts_ms, first_audio_ms = self._play_streaming_tts_with_feedback(
                spoken_prompt,
                turn_started=tts_started,
            )
            self.runtime.finish_speaking()
            self._emit_status(force=True)
            self.runtime.mark_long_term_proactive_candidate_delivered(
                reservation,
                prompt_text=spoken_prompt,
            )
            self.runtime.proactive_governor.mark_delivered(governor_reservation)
            self.emit(f"longterm_proactive_candidate={candidate.candidate_id}")
            self.emit(f"longterm_proactive_kind={candidate.kind}")
            self.emit(f"longterm_proactive_prompt_mode={prompt_mode}")
            self.emit(f"longterm_proactive_prompt={spoken_prompt}")
            self.emit(f"timing_longterm_proactive_tts_ms={tts_ms}")
            if first_audio_ms is not None:
                self.emit(f"timing_longterm_proactive_first_audio_ms={first_audio_ms}")
            if response is not None:
                self._record_usage(
                    request_kind="longterm_proactive_prompt",
                    source="realtime_loop",
                    model=response.model,
                    response_id=response.response_id,
                    request_id=response.request_id,
                    used_web_search=False,
                    token_usage=response.token_usage,
                    proactive_trigger=trigger_id,
                )
            self._record_event(
                "longterm_proactive_prompted",
                "Twinr spoke a proactive prompt derived from long-term memory.",
                trigger=trigger_id,
                candidate_id=candidate.candidate_id,
                candidate_kind=candidate.kind,
                prompt=spoken_prompt,
                prompt_mode=prompt_mode,
                rationale=candidate.rationale,
            )
            return True
        except Exception as exc:
            if self.runtime.status.value == "answering":
                self.runtime.finish_speaking()
                self._emit_status(force=True)
            self.runtime.mark_long_term_proactive_candidate_skipped(
                reservation,
                reason=f"delivery_failed: {exc}",
            )
            self.runtime.proactive_governor.mark_skipped(
                governor_reservation,
                reason=f"delivery_failed: {exc}",
            )
            self.emit(f"longterm_proactive_error={exc}")
            return False

    def _deliver_due_reminder(
        self,
        reminder,
        *,
        governor_reservation: ProactiveGovernorReservation,
    ) -> bool:
        response = None
        spoken_prompt = ""
        try:
            stop_processing_feedback = self._start_working_feedback_loop("processing")
            try:
                response = self._phrase_due_reminder_with_fallback(reminder)
            finally:
                stop_processing_feedback()
            spoken_prompt = self.runtime.begin_reminder_prompt(response.text)
            self._emit_status(force=True)
            tts_started = time.monotonic()
            tts_ms, first_audio_ms = self._play_streaming_tts_with_feedback(
                spoken_prompt,
                turn_started=tts_started,
            )
            self.runtime.finish_speaking()
            self._emit_status(force=True)
            delivered = self.runtime.mark_reminder_delivered(reminder.reminder_id)
            self.runtime.proactive_governor.mark_delivered(governor_reservation)
            self.emit("reminder_delivered=true")
            self.emit(f"reminder_due_at={delivered.due_at.isoformat()}")
            self.emit(f"reminder_text={spoken_prompt}")
            if response.response_id:
                self.emit(f"reminder_response_id={response.response_id}")
            if response.request_id:
                self.emit(f"reminder_request_id={response.request_id}")
            self.emit(f"timing_reminder_tts_ms={tts_ms}")
            if first_audio_ms is not None:
                self.emit(f"timing_reminder_first_audio_ms={first_audio_ms}")
            self._record_usage(
                request_kind="reminder_delivery",
                source="realtime_loop",
                model=response.model,
                response_id=response.response_id,
                request_id=response.request_id,
                used_web_search=False,
                token_usage=response.token_usage,
                reminder_id=delivered.reminder_id,
                reminder_kind=delivered.kind,
            )
            return True
        except Exception as exc:
            if self.runtime.status.value == "answering":
                self.runtime.finish_speaking()
                self._emit_status(force=True)
            self.runtime.mark_reminder_failed(reminder.reminder_id, error=str(exc))
            self.runtime.proactive_governor.mark_skipped(
                governor_reservation,
                reason=f"delivery_failed: {exc}",
            )
            self.emit(f"reminder_error={exc}")
            return False

    def _phrase_due_reminder_with_fallback(self, reminder):
        helper = getattr(self.agent_provider, "phrase_due_reminder_with_metadata", None)
        if callable(helper):
            return helper(reminder)

        generic = getattr(self.agent_provider, "respond_with_metadata", None)
        if not callable(generic):
            raise RuntimeError("The configured backend cannot phrase due reminders")

        current_time = datetime.now(ZoneInfo(self.config.local_timezone_name))
        prompt_parts = [
            "A stored Twinr reminder is due now.",
            f"Current local time: {format_due_label(current_time, timezone_name=self.config.local_timezone_name)}",
            f"Scheduled reminder time: {format_due_label(reminder.due_at, timezone_name=self.config.local_timezone_name)}",
            f"Reminder kind: {reminder.kind}",
            f"Reminder summary: {reminder.summary}",
        ]
        if reminder.details:
            prompt_parts.append(f"Reminder details: {reminder.details}")
        if reminder.original_request:
            prompt_parts.append(f"Original user request: {reminder.original_request}")
        prompt_parts.append("Speak the reminder now.")
        self.emit("reminder_backend_fallback=generic")
        return generic(
            "\n".join(prompt_parts),
            instructions=REMINDER_DELIVERY_INSTRUCTIONS,
            allow_web_search=False,
        )

    def _run_automation_entry(
        self,
        entry: AutomationDefinition,
        *,
        trigger_source: str,
        event_name: str | None = None,
        facts: dict[str, object] | None = None,
    ) -> bool:
        governor_reservation: ProactiveGovernorReservation | None = None
        if self._automation_uses_speech(entry):
            governor_reservation = self._reserve_governed_prompt(
                ProactiveGovernorCandidate(
                    source_kind="automation",
                    source_id=entry.automation_id,
                    summary=entry.name,
                    priority=70,
                    presence_session_id=self._current_presence_session_id(),
                    safety_exempt=False,
                    counts_toward_presence_budget=True,
                )
            )
            if governor_reservation is None:
                return False
        executed = False
        try:
            for action in entry.actions:
                if not action.enabled:
                    continue
                self._execute_automation_action(entry, action)
                executed = True
            if not executed:
                raise RuntimeError("Automation has no enabled actions")
            self.runtime.mark_automation_triggered(entry.automation_id)
            if governor_reservation is not None:
                self.runtime.proactive_governor.mark_delivered(governor_reservation)
            self.emit("automation_executed=true")
            self.emit(f"automation_name={entry.name}")
            self.emit(f"automation_id={entry.automation_id}")
            self.emit(f"automation_trigger_source={trigger_source}")
            if event_name:
                self.emit(f"automation_event_name={event_name}")
            self._record_event(
                "automation_executed",
                "An automation was executed.",
                automation_id=entry.automation_id,
                name=entry.name,
                trigger_source=trigger_source,
                event_name=event_name,
                facts=facts,
            )
            return True
        except Exception as exc:
            self._recover_automation_output_state()
            self.runtime.mark_automation_triggered(entry.automation_id, source="automation_failed")
            if governor_reservation is not None:
                self.runtime.proactive_governor.mark_skipped(
                    governor_reservation,
                    reason=f"delivery_failed: {exc}",
                )
            self.emit(f"automation_error={exc}")
            self._record_event(
                "automation_execution_failed",
                "An automation failed during execution.",
                level="error",
                automation_id=entry.automation_id,
                name=entry.name,
                trigger_source=trigger_source,
                event_name=event_name,
                error=str(exc),
            )
            return False

    def _automation_uses_speech(self, entry: AutomationDefinition) -> bool:
        for action in entry.actions:
            if not action.enabled:
                continue
            if action.kind == "say":
                return True
            if action.kind == "llm_prompt":
                delivery = normalize_delivery(action.payload.get("delivery"))
                if delivery != "printed":
                    return True
        return False

    def _execute_automation_action(self, entry: AutomationDefinition, action: AutomationAction) -> None:
        if action.kind == "say":
            self._speak_automation_text(entry, action.text or "")
            return
        if action.kind == "print":
            self._print_automation_text(entry, action.text or "", request_source="automation_static")
            return
        if action.kind == "llm_prompt":
            delivery = normalize_delivery(action.payload.get("delivery"))
            allow_web_search = bool(action.payload.get("allow_web_search", False))
            stop_processing_feedback = self._start_working_feedback_loop("processing")
            try:
                response = self.agent_provider.fulfill_automation_prompt_with_metadata(
                    action.text or "",
                    allow_web_search=allow_web_search,
                    delivery=delivery,
                )
            finally:
                stop_processing_feedback()
            self._record_usage(
                request_kind="automation_generation",
                source="automation",
                model=response.model,
                response_id=response.response_id,
                request_id=response.request_id,
                used_web_search=response.used_web_search,
                token_usage=response.token_usage,
                automation_id=entry.automation_id,
                automation_name=entry.name,
                delivery=delivery,
            )
            if delivery == "printed":
                composed = self.agent_provider.compose_print_job_with_metadata(
                    focus_hint=f"{entry.name}: {action.text or ''}".strip(": "),
                    direct_text=response.text,
                    request_source="automation",
                )
                self._record_usage(
                    request_kind="automation_print_compose",
                    source="automation",
                    model=composed.model,
                    response_id=composed.response_id,
                    request_id=composed.request_id,
                    used_web_search=False,
                    token_usage=composed.token_usage,
                    automation_id=entry.automation_id,
                    automation_name=entry.name,
                )
                self._print_automation_text(entry, composed.text, request_source="automation")
                return
            self._speak_automation_text(entry, response.text)
            return
        raise RuntimeError(f"Unsupported automation action kind during execution: {action.kind}")

    def _speak_automation_text(self, entry: AutomationDefinition, text: str) -> None:
        spoken_prompt = self.runtime.begin_automation_prompt(text)
        self._emit_status(force=True)
        tts_started = time.monotonic()
        tts_ms, first_audio_ms = self._play_streaming_tts_with_feedback(
            spoken_prompt,
            turn_started=tts_started,
        )
        self.runtime.finish_speaking()
        self._emit_status(force=True)
        self.emit(f"automation_spoken={spoken_prompt}")
        self.emit(f"automation_name={entry.name}")
        self.emit(f"automation_id={entry.automation_id}")
        self.emit(f"timing_automation_tts_ms={tts_ms}")
        if first_audio_ms is not None:
            self.emit(f"timing_automation_first_audio_ms={first_audio_ms}")

    def _print_automation_text(self, entry: AutomationDefinition, text: str, *, request_source: str) -> None:
        self.runtime.maybe_begin_automation_print()
        self._emit_status(force=True)
        stop_printing_feedback = self._start_working_feedback_loop("printing")
        try:
            print_job = self.printer.print_text(text)
        finally:
            stop_printing_feedback()
        self.emit(f"automation_print_text={text}")
        self.emit(f"automation_name={entry.name}")
        self.emit(f"automation_id={entry.automation_id}")
        if print_job:
            self.emit(f"automation_print_job={print_job}")
        self._record_event(
            "automation_print_sent",
            "Scheduled automation sent content to the printer.",
            automation_id=entry.automation_id,
            name=entry.name,
            request_source=request_source,
            queue=self.config.printer_queue,
            job=print_job,
        )
        self.runtime.long_term_memory.enqueue_multimodal_evidence(
            event_name="print_completed",
            modality="printer",
            source="automation_print",
            message="Scheduled automation finished a printer delivery.",
            data={
                "request_source": request_source,
                "queue": self.config.printer_queue,
                "job": print_job or "",
                "automation_id": entry.automation_id,
            },
        )
        self.runtime.finish_printing()
        self._emit_status(force=True)

    def _recover_automation_output_state(self) -> None:
        if self.runtime.status.value == "answering":
            self.runtime.finish_speaking()
            self._emit_status(force=True)
            return
        if self.runtime.status.value == "printing":
            self.runtime.finish_printing()
            self._emit_status(force=True)
