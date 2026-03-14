from __future__ import annotations

from datetime import datetime
from queue import Empty
import time
from zoneinfo import ZoneInfo

from twinr.agent.tools.automations import normalize_delivery
from twinr.automations import AutomationAction, AutomationDefinition
from twinr.memory.reminders import format_due_label
from twinr.proactive import proactive_observation_facts, proactive_prompt_mode, SocialTriggerDecision
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

        phrase_response = None
        prompt_mode = proactive_prompt_mode(trigger)
        prompt_text = trigger.prompt
        if prompt_mode == "llm":
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
        first_audio_at: list[float | None] = [None]

        def mark_first_chunk():
            for chunk in self.tts_provider.synthesize_stream(prompt):
                if first_audio_at[0] is None:
                    first_audio_at[0] = time.monotonic()
                yield chunk

        with self._audio_lock:
            self.player.play_wav_chunks(mark_first_chunk())
        tts_ms = int((time.monotonic() - tts_started) * 1000)
        self.runtime.finish_speaking()
        self._emit_status(force=True)
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
        if first_audio_at[0] is not None:
            self.emit(f"timing_social_first_audio_ms={int((first_audio_at[0] - tts_started) * 1000)}")
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

    def handle_sensor_observation(self, facts: dict[str, object], event_names: tuple[str, ...]) -> None:
        self._sensor_observation_queue.put((dict(facts), tuple(event_names)))

    def _maybe_deliver_due_reminder(self) -> bool:
        now_monotonic = time.monotonic()
        if now_monotonic < self._next_reminder_check_at:
            return False
        self._next_reminder_check_at = now_monotonic + self.config.reminder_poll_interval_s
        if not self._background_work_allowed():
            return False
        due_entries = self.runtime.reserve_due_reminders(limit=1)
        if not due_entries:
            return False
        return self._deliver_due_reminder(due_entries[0])

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

    def _recent_proactive_prompts(
        self,
        *,
        trigger_id: str | None = None,
        limit: int = 3,
    ) -> tuple[str, ...]:
        prompts: list[str] = []
        for entry in reversed(self.runtime.ops_events.tail(limit=100)):
            if entry.get("event") != "social_trigger_prompted":
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

    def _deliver_due_reminder(self, reminder) -> bool:
        response = None
        spoken_prompt = ""
        try:
            response = self._phrase_due_reminder_with_fallback(reminder)
            spoken_prompt = self.runtime.begin_reminder_prompt(response.text)
            self._emit_status(force=True)
            tts_started = time.monotonic()
            first_audio_at: list[float | None] = [None]

            def mark_first_chunk():
                for chunk in self.tts_provider.synthesize_stream(spoken_prompt):
                    if first_audio_at[0] is None:
                        first_audio_at[0] = time.monotonic()
                    yield chunk

            with self._audio_lock:
                self.player.play_wav_chunks(mark_first_chunk())
            tts_ms = int((time.monotonic() - tts_started) * 1000)
            self.runtime.finish_speaking()
            self._emit_status(force=True)
            delivered = self.runtime.mark_reminder_delivered(reminder.reminder_id)
            self.emit("reminder_delivered=true")
            self.emit(f"reminder_due_at={delivered.due_at.isoformat()}")
            self.emit(f"reminder_text={spoken_prompt}")
            if response.response_id:
                self.emit(f"reminder_response_id={response.response_id}")
            if response.request_id:
                self.emit(f"reminder_request_id={response.request_id}")
            self.emit(f"timing_reminder_tts_ms={tts_ms}")
            if first_audio_at[0] is not None:
                self.emit(
                    f"timing_reminder_first_audio_ms={int((first_audio_at[0] - tts_started) * 1000)}"
                )
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
            response = self.agent_provider.fulfill_automation_prompt_with_metadata(
                action.text or "",
                allow_web_search=allow_web_search,
                delivery=delivery,
            )
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
        first_audio_at: list[float | None] = [None]

        def mark_first_chunk():
            for chunk in self.tts_provider.synthesize_stream(spoken_prompt):
                if first_audio_at[0] is None:
                    first_audio_at[0] = time.monotonic()
                yield chunk

        with self._audio_lock:
            self.player.play_wav_chunks(mark_first_chunk())
        tts_ms = int((time.monotonic() - tts_started) * 1000)
        self.runtime.finish_speaking()
        self._emit_status(force=True)
        self.emit(f"automation_spoken={spoken_prompt}")
        self.emit(f"automation_name={entry.name}")
        self.emit(f"automation_id={entry.automation_id}")
        self.emit(f"timing_automation_tts_ms={tts_ms}")
        if first_audio_at[0] is not None:
            self.emit(
                f"timing_automation_first_audio_ms={int((first_audio_at[0] - tts_started) * 1000)}"
            )

    def _print_automation_text(self, entry: AutomationDefinition, text: str, *, request_source: str) -> None:
        self.runtime.maybe_begin_automation_print()
        self._emit_status(force=True)
        print_job = self.printer.print_text(text)
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
