# CHANGELOG: 2026-03-27
# BUG-1: Fix reminder reservation leaks on preview/reserve mismatches by best-effort releasing the mismatched reservation and retrying.
# BUG-2: Fix duplicate side effects after partial automation failures by checkpointing completed actions per execution key and skipping already-completed actions on retry.
# BUG-3: Fix malformed llm_prompt delivery metadata causing automation failure by centralizing delivery normalization with a safe fallback.
# BUG-4: Fix joinable sensor queue stalls by calling task_done() after each consumed observation and handling Queue.ShutDown gracefully.
# BUG-5: Treat non-success tool_call return statuses as failures instead of silently marking the automation successful.
# SEC-1: Restrict automation tool_call execution to an explicit allowlist / unsafe opt-in and sanitize tool names, payload sizes, and error text.
# SEC-2: Require explicit opt-in for autonomous background web search and cap emitted / spoken / printed / evidence payload sizes to reduce exfiltration and resource exhaustion risk.
# IMP-1: Use monotonic_ns deadlines and bounded sensor batch / time budgets for better fairness and lower scheduler drift on Raspberry Pi 4.
# IMP-2: Add in-flight guards, execution keys, modern queue shutdown handling, sanitized evidence snapshots, and best-effort reservation release hooks.

"""Reminder and automation helpers for the realtime background loop."""

# mypy: ignore-errors

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import hashlib
import inspect
import json
import queue
from queue import Empty
import re
import threading
import time

from twinr.agent.tools.handlers.automation_support import normalize_delivery
from twinr.automations import AutomationAction, AutomationDefinition
from twinr.agent.workflows.realtime_runtime.background_delivery import (
    BackgroundDeliveryBlocked as _BackgroundDeliveryBlocked,
)
from twinr.agent.workflows.realtime_runtime.reminder_delivery import (
    deliver_due_reminder,
    phrase_due_reminder_with_fallback,
)
from twinr.proactive.governance.governor import ProactiveGovernorCandidate, ProactiveGovernorReservation
from twinr.providers.openai.core.instructions import REMINDER_DELIVERY_INSTRUCTIONS


_QueueShutDown = getattr(queue, "ShutDown", None)
_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")


class _AutomationDeferredDueToBackground(RuntimeError):
    def __init__(self, reason: str, *, executed_partial: bool = False) -> None:
        self.reason = reason
        self.executed_partial = executed_partial
        super().__init__(reason)


class BackgroundAutomationMixin:
    """Handle reminder polling, sensor automations, and automation execution."""

    def _maybe_deliver_due_reminder(self) -> bool:
        if not self._deadline_elapsed(
            attr_name="_next_reminder_check_at",
            interval_name="reminder_poll_interval_s",
            default_seconds=5.0,
        ):
            return False
        if not self._background_work_allowed():
            return False

        max_attempts = self._config_int(
            "reminder_claim_attempts",
            default=2,
            minimum=1,
            maximum=4,
        )
        last_mismatch_id = None
        try:
            for _ in range(max_attempts):
                preview_entries = self.runtime.peek_due_reminders(limit=1)
                if not preview_entries:
                    return False

                preview_entry = preview_entries[0]
                governor_inputs = self._current_governor_inputs(requested_channel="speech")
                governor_reservation = self._reserve_governed_prompt(
                    ProactiveGovernorCandidate(
                        source_kind="reminder",
                        source_id=preview_entry.reminder_id,
                        summary=preview_entry.summary,
                        priority=80,
                        presence_session_id=governor_inputs.presence_session_id,
                        safety_exempt=False,
                        counts_toward_presence_budget=False,
                    ),
                    governor_inputs=governor_inputs,
                )
                if governor_reservation is None:
                    return False

                try:
                    blocked_reason = self._background_block_reason()
                    if blocked_reason is not None:
                        self._safe_cancel_governor_reservation(governor_reservation)
                        self._emit_kv("reminder_skipped", blocked_reason)
                        self._safe_record_event(
                            "reminder_skipped",
                            "A due reminder was skipped because Twinr stopped being idle before reminder delivery started.",
                            reminder_id=preview_entry.reminder_id,
                            skip_reason=blocked_reason,
                        )
                        return False

                    due_entries = self.runtime.reserve_due_reminders(limit=1)
                    if not due_entries:
                        self._safe_cancel_governor_reservation(governor_reservation)
                        return False

                    due_entry = due_entries[0]
                    preview_id = self._coerce_text(getattr(preview_entry, "reminder_id", None))
                    due_id = self._coerce_text(getattr(due_entry, "reminder_id", None))
                    if due_id != preview_id:
                        self._safe_cancel_governor_reservation(governor_reservation)
                        released = self._safe_release_due_reminder(due_entry)
                        last_mismatch_id = due_id or last_mismatch_id
                        self._safe_record_event(
                            "reminder_claim_mismatch",
                            "Twinr observed a due reminder preview/reservation mismatch and retried the claim.",
                            level="warning",
                            preview_reminder_id=preview_id,
                            reserved_reminder_id=due_id,
                            released_reserved_reminder=released,
                        )
                        continue

                    return self._deliver_due_reminder(
                        due_entry,
                        governor_reservation=governor_reservation,
                    )
                except Exception:
                    self._safe_cancel_governor_reservation(governor_reservation)
                    raise

            if last_mismatch_id:
                self._emit_kv("reminder_claim_mismatch", last_mismatch_id)
            return False
        except Exception as exc:
            self._emit_kv("reminder_due_check_error", self._safe_exception_text(exc))
            self._safe_record_event(
                "reminder_due_check_failed",
                "Twinr failed while checking for due reminders.",
                level="error",
                error=self._safe_exception_text(exc),
            )
            return False

    def _maybe_run_due_automation(self) -> bool:
        if not self._deadline_elapsed(
            attr_name="_next_automation_check_at",
            interval_name="automation_poll_interval_s",
            default_seconds=5.0,
        ):
            return False
        if not self._background_work_allowed():
            return False

        try:
            due_matches = tuple(self.runtime.due_time_matches() or ())
            if not due_matches:
                return False
            for match in due_matches:
                entry = match.entry
                automation_id = self._coerce_text(getattr(entry, "automation_id", None))
                if automation_id and self._automation_retry_blocked(automation_id):
                    continue
                if self._run_automation_entry(
                    entry,
                    trigger_source="time_schedule",
                    scheduled_for_at=match.scheduled_for_at,
                ):
                    return True
            return False
        except Exception as exc:
            self._emit_kv("automation_due_check_error", self._safe_exception_text(exc))
            self._safe_record_event(
                "automation_due_check_failed",
                "Twinr failed while checking for due automations.",
                level="error",
                error=self._safe_exception_text(exc),
            )
            return False

    def _maybe_run_sensor_automation(self) -> bool:
        if not self._background_work_allowed():
            return False

        max_batch = self._config_int(
            "sensor_automation_max_batch",
            default=16,
            minimum=1,
            maximum=512,
        )
        budget_ms = self._config_float(
            "sensor_automation_batch_budget_ms",
            default=75.0,
            minimum=1.0,
            maximum=5_000.0,
        )
        started_ns = time.monotonic_ns()
        budget_ns = int(budget_ms * 1_000_000)
        processed = 0

        while processed < max_batch and (time.monotonic_ns() - started_ns) <= budget_ns:
            try:
                observation = self._sensor_observation_queue.get_nowait()
            except Empty:
                return False
            except Exception as exc:
                if self._is_queue_shutdown_exception(exc):
                    return False
                self._emit_kv("sensor_automation_queue_error", self._safe_exception_text(exc))
                self._safe_record_event(
                    "sensor_automation_queue_failed",
                    "Twinr failed while reading the sensor automation queue.",
                    level="error",
                    error=self._safe_exception_text(exc),
                )
                return False

            processed += 1
            try:
                facts, event_names = observation
                normalized_event_names = self._normalize_and_limit_event_names(event_names)
                self._safe_enqueue_multimodal_evidence(
                    event_name="sensor_observation",
                    modality="sensor",
                    source="proactive_monitor",
                    message="Changed multimodal sensor observation recorded.",
                    data={
                        "facts": self._sanitize_jsonish(
                            facts,
                            max_depth=3,
                            max_items=16,
                            max_string_chars=self._config_int(
                                "sensor_automation_fact_value_max_chars",
                                default=256,
                                minimum=32,
                                maximum=4_096,
                            ),
                        ),
                        "event_names": list(normalized_event_names),
                    },
                )
                if self._run_matching_sensor_automations(
                    facts=facts,
                    event_names=normalized_event_names,
                ):
                    return True
            except Exception as exc:
                safe_event_names = []
                try:
                    safe_event_names = list(self._normalize_and_limit_event_names(event_names))
                except Exception:
                    safe_event_names = []
                self._emit_kv("sensor_automation_error", self._safe_exception_text(exc))
                self._safe_record_event(
                    "sensor_automation_failed",
                    "Twinr failed while executing sensor-triggered automations.",
                    level="error",
                    event_names=safe_event_names,
                    error=self._safe_exception_text(exc),
                )
            finally:
                self._safe_queue_task_done(self._sensor_observation_queue)

        return False

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
        event_label = self._join_event_names(event_names)
        for entry in sorted(
            matched.values(),
            key=lambda item: self._coerce_text(getattr(item, "name", None)).casefold(),
        ):
            executed = self._run_automation_entry(
                entry,
                trigger_source="sensor",
                event_name=event_label,
                facts=facts,
            )
            executed_any = executed or executed_any
        return executed_any

    def _deliver_due_reminder(
        self,
        reminder,
        *,
        governor_reservation: ProactiveGovernorReservation,
    ) -> bool:
        return deliver_due_reminder(
            self,
            reminder,
            governor_reservation=governor_reservation,
            instructions=REMINDER_DELIVERY_INSTRUCTIONS,
        )

    def _phrase_due_reminder_with_fallback(self, reminder):
        return phrase_due_reminder_with_fallback(
            self,
            reminder,
            instructions=REMINDER_DELIVERY_INSTRUCTIONS,
        )

    def _run_automation_entry(
        self,
        entry: AutomationDefinition,
        *,
        trigger_source: str,
        event_name: str | None = None,
        facts: dict[str, object] | None = None,
        scheduled_for_at: datetime | None = None,
    ) -> bool:
        automation_id = self._coerce_text(getattr(entry, "automation_id", None)) or "unknown"
        automation_name = self._coerce_text(getattr(entry, "name", None)) or automation_id
        if self._automation_retry_blocked(automation_id):
            self._emit_kv("automation_retry_backoff", automation_id)
            return False

        if not self._try_begin_automation_execution(automation_id):
            self._emit_kv("automation_inflight", automation_id)
            return False

        governor_reservation: ProactiveGovernorReservation | None = None
        executed = False
        executed_speech = False
        enabled_action_count = 0
        resumed_action_count = 0
        execution_key = self._automation_execution_key(
            entry,
            trigger_source=trigger_source,
            event_name=event_name,
            facts=facts,
            scheduled_for_at=scheduled_for_at,
        )
        try:
            self._cleanup_stale_automation_execution_state()

            if self._automation_uses_speech(entry):
                governor_inputs = self._current_governor_inputs(requested_channel="speech")
                governor_reservation = self._reserve_governed_prompt(
                    ProactiveGovernorCandidate(
                        source_kind="automation",
                        source_id=automation_id,
                        summary=automation_name,
                        priority=70,
                        presence_session_id=governor_inputs.presence_session_id,
                        safety_exempt=False,
                        counts_toward_presence_budget=True,
                    ),
                    governor_inputs=governor_inputs,
                )
                if governor_reservation is None:
                    return False

            blocked_reason = self._background_block_reason()
            if blocked_reason is not None:
                if governor_reservation is not None:
                    self._safe_cancel_governor_reservation(governor_reservation)
                self._emit_kv("automation_skipped", blocked_reason)
                self._safe_record_event(
                    "automation_skipped",
                    "A due automation was skipped because Twinr stopped being idle before automation delivery started.",
                    automation_id=automation_id,
                    name=automation_name,
                    trigger_source=trigger_source,
                    event_name=event_name,
                    skip_reason=blocked_reason,
                )
                return False

            for action_index, action in enumerate(tuple(getattr(entry, "actions", ()) or ())):
                if not getattr(action, "enabled", False):
                    continue
                enabled_action_count += 1

                blocked_reason = self._background_block_reason()
                if blocked_reason is not None:
                    raise _AutomationDeferredDueToBackground(
                        blocked_reason,
                        executed_partial=executed,
                    )

                checkpoint_token = self._automation_action_checkpoint_token(
                    entry,
                    action,
                    action_index=action_index,
                )
                if self._automation_action_already_completed(execution_key, checkpoint_token):
                    resumed_action_count += 1
                    continue

                effect_channel = self._execute_automation_action(
                    entry,
                    action,
                    execution_key=execution_key,
                )
                executed = True
                executed_speech = executed_speech or effect_channel == "speech"
                self._mark_automation_action_completed(execution_key, checkpoint_token)

            if enabled_action_count == 0:
                raise RuntimeError("Automation has no enabled actions")
            if not executed and resumed_action_count == enabled_action_count:
                self._safe_record_event(
                    "automation_execution_resumed",
                    "Twinr resumed a partially completed automation and only finalized its completion state.",
                    level="info",
                    automation_id=automation_id,
                    execution_key=execution_key,
                )

            self.runtime.mark_automation_triggered(
                entry.automation_id,
                scheduled_for_at=scheduled_for_at,
            )
            self._clear_automation_failure_backoff(automation_id)
            self._clear_automation_execution_state(execution_key)
            if governor_reservation is not None:
                self._safe_mark_governor_delivered(governor_reservation)
            self._emit_kv("automation_executed", "true")
            self._emit_kv("automation_name", automation_name)
            self._emit_kv("automation_id", automation_id)
            self._emit_kv("automation_trigger_source", trigger_source)
            self._emit_kv("automation_execution_key", execution_key)
            if event_name:
                self._emit_kv("automation_event_name", event_name)
            if scheduled_for_at is not None:
                self._emit_kv("automation_scheduled_for_at", scheduled_for_at.isoformat())
            self._safe_record_event(
                "automation_executed",
                "An automation was executed.",
                automation_id=automation_id,
                name=automation_name,
                trigger_source=trigger_source,
                event_name=event_name,
                scheduled_for_at=scheduled_for_at.isoformat() if scheduled_for_at is not None else None,
                execution_key=execution_key,
                facts=self._sanitize_jsonish(facts, max_depth=2, max_items=12, max_string_chars=192),
            )
            return True
        except (_BackgroundDeliveryBlocked, _AutomationDeferredDueToBackground) as blocked:
            if governor_reservation is not None:
                if executed_speech:
                    self._safe_mark_governor_delivered(governor_reservation)
                else:
                    self._safe_cancel_governor_reservation(governor_reservation)
            self._emit_kv("automation_skipped", blocked.reason)
            self._safe_record_event(
                "automation_skipped",
                "A due automation was skipped because Twinr stopped being idle before automation output started.",
                automation_id=automation_id,
                name=automation_name,
                trigger_source=trigger_source,
                event_name=event_name,
                skip_reason=blocked.reason,
                executed_partial=executed or getattr(blocked, "executed_partial", False),
                execution_key=execution_key,
            )
            return False
        except Exception as exc:
            self._recover_automation_output_state()
            self._register_automation_failure_backoff(automation_id)
            if governor_reservation is not None:
                if executed_speech:
                    self._safe_mark_governor_delivered(governor_reservation)
                else:
                    self._safe_mark_governor_skipped(
                        governor_reservation,
                        reason=f"delivery_failed: {self._safe_exception_text(exc)}",
                    )
            self._emit_kv("automation_error", self._safe_exception_text(exc))
            self._safe_record_event(
                "automation_execution_failed",
                "An automation failed during execution.",
                level="error",
                automation_id=automation_id,
                name=automation_name,
                trigger_source=trigger_source,
                event_name=event_name,
                execution_key=execution_key,
                error=self._safe_exception_text(exc),
            )
            return False
        finally:
            self._finish_automation_execution(automation_id)

    def _automation_uses_speech(self, entry: AutomationDefinition) -> bool:
        for action in tuple(getattr(entry, "actions", ()) or ()):
            if not getattr(action, "enabled", False):
                continue
            if getattr(action, "kind", None) == "say":
                return True
            if getattr(action, "kind", None) == "llm_prompt":
                payload = self._validated_automation_payload(action, context="automation llm prompt payload")
                delivery = self._normalize_automation_delivery(payload)
                if delivery != "printed":
                    return True
        return False

    def _execute_automation_action(
        self,
        entry: AutomationDefinition,
        action: AutomationAction,
        *,
        execution_key: str,
    ) -> str:
        action_kind = getattr(action, "kind", None)
        action_text = getattr(action, "text", None)

        if action_kind == "say":
            self._speak_automation_text(
                entry,
                self._prepare_automation_text(
                    action_text,
                    context="automation say action",
                    channel="speech",
                ),
            )
            return "speech"

        if action_kind == "print":
            self._print_automation_text(
                entry,
                self._prepare_automation_text(
                    action_text,
                    context="automation print action",
                    channel="print",
                ),
                request_source="automation_static",
            )
            return "print"

        if action_kind == "llm_prompt":
            payload = self._validated_automation_payload(action, context="automation llm prompt payload")
            delivery = self._normalize_automation_delivery(payload)
            allow_web_search = self._background_web_search_allowed(payload)
            prompt_text = self._prepare_automation_text(
                action_text,
                context="automation llm prompt action",
                channel="prompt",
            )
            stop_processing_feedback = self._start_working_feedback_loop("processing")
            try:
                response = self._call_with_supported_kwargs(
                    self.agent_provider.fulfill_automation_prompt_with_metadata,
                    prompt_text,
                    allow_web_search=allow_web_search,
                    delivery=delivery,
                    request_source="automation",
                    background=True,
                    timeout_s=self._config_float(
                        "automation_llm_timeout_s",
                        default=45.0,
                        minimum=1.0,
                        maximum=300.0,
                    ),
                    max_output_tokens=self._config_int(
                        "automation_llm_max_output_tokens",
                        default=512,
                        minimum=64,
                        maximum=8_192,
                    ),
                )
            finally:
                stop_processing_feedback()

            response_text = self._prepare_generated_automation_text(
                getattr(response, "text", None),
                context=(
                    "automation "
                    f"{self._coerce_text(getattr(entry, 'automation_id', None)) or 'unknown'} llm output"
                ),
                delivery=delivery,
            )
            self._safe_record_usage(
                request_kind="automation_generation",
                source="automation",
                model=getattr(response, "model", "unknown"),
                response_id=getattr(response, "response_id", None),
                request_id=getattr(response, "request_id", None),
                used_web_search=getattr(response, "used_web_search", False),
                token_usage=getattr(response, "token_usage", None),
                automation_id=getattr(entry, "automation_id", None),
                automation_name=getattr(entry, "name", None),
                delivery=delivery,
                execution_key=execution_key,
            )
            if delivery == "printed":
                composed = self._call_with_supported_kwargs(
                    self.agent_provider.compose_print_job_with_metadata,
                    focus_hint=f"{self._coerce_text(getattr(entry, 'name', None))}: {prompt_text}".strip(": "),
                    direct_text=response_text,
                    request_source="automation",
                    background=True,
                    max_output_tokens=self._config_int(
                        "automation_print_compose_max_output_tokens",
                        default=768,
                        minimum=64,
                        maximum=8_192,
                    ),
                )
                composed_text = self._prepare_automation_text(
                    getattr(composed, "text", None),
                    context=(
                        "automation "
                        f"{self._coerce_text(getattr(entry, 'automation_id', None)) or 'unknown'} print composition"
                    ),
                    channel="print",
                )
                self._safe_record_usage(
                    request_kind="automation_print_compose",
                    source="automation",
                    model=getattr(composed, "model", "unknown"),
                    response_id=getattr(composed, "response_id", None),
                    request_id=getattr(composed, "request_id", None),
                    used_web_search=False,
                    token_usage=getattr(composed, "token_usage", None),
                    automation_id=getattr(entry, "automation_id", None),
                    automation_name=getattr(entry, "name", None),
                    execution_key=execution_key,
                )
                self._print_automation_text(entry, composed_text, request_source="automation")
                return "print"

            self._speak_automation_text(entry, response_text)
            return "speech"

        if action_kind == "tool_call":
            tool_name = self._normalize_tool_name(getattr(action, "tool_name", None))
            if not self._automation_tool_allowed(tool_name):
                raise PermissionError(f"Automation tool_call denied for tool '{tool_name}'")
            arguments = self._validated_automation_payload(action, context=f"automation tool payload for {tool_name}")
            self._ensure_background_delivery_allowed()
            handler = self._automation_tool_broker_policy().resolve_handler(self.tool_executor, tool_name)
            result = handler(arguments)
            self._emit_kv("automation_tool_call", tool_name)
            if isinstance(result, Mapping):
                if result.get("status"):
                    self._emit_kv("automation_tool_status", result.get("status"))
                status_text = self._coerce_text(result.get("status", "")).strip().casefold()
                if status_text in {
                    "error",
                    "failed",
                    "failure",
                    "denied",
                    "blocked",
                    "rejected",
                    "invalid",
                    "timeout",
                    "timed_out",
                    "unavailable",
                }:
                    raise RuntimeError(f"Automation tool '{tool_name}' failed with status '{status_text}'")
                if "ok" in result and not self._coerce_bool(result.get("ok")):
                    raise RuntimeError(f"Automation tool '{tool_name}' reported ok=false")
            return "tool"

        raise RuntimeError(f"Unsupported automation action kind during execution: {action_kind}")

    def _speak_automation_text(self, entry: AutomationDefinition, text: str) -> None:
        spoken_prompt = self._begin_background_delivery(
            lambda lease: lease.run_locked(
                lambda: self.runtime.begin_automation_prompt(
                    self._prepare_automation_text(
                        text,
                        context="automation speech output",
                        channel="speech",
                    )
                )
            )
        )
        self._safe_emit_status(force=True)
        tts_started = time.monotonic()
        tts_ms, first_audio_ms = self._play_streaming_tts_with_feedback(
            spoken_prompt,
            turn_started=tts_started,
        )
        self._finalize_speaking_output()
        self._emit_kv("automation_spoken", spoken_prompt, max_chars=self._config_int("automation_emit_text_max_chars", 512, 64, 8_192))
        self._emit_kv("automation_name", self._coerce_text(getattr(entry, "name", None)))
        self._emit_kv("automation_id", self._coerce_text(getattr(entry, "automation_id", None)))
        self._emit_kv("timing_automation_tts_ms", tts_ms)
        if first_audio_ms is not None:
            self._emit_kv("timing_automation_first_audio_ms", first_audio_ms)

    def _print_automation_text(self, entry: AutomationDefinition, text: str, *, request_source: str) -> None:
        printable_text = self._prepare_automation_text(text, context="automation print output", channel="print")
        self._begin_background_delivery(
            lambda lease: lease.run_locked(self.runtime.maybe_begin_automation_print)
        )
        self._safe_emit_status(force=True)
        stop_printing_feedback = self._start_working_feedback_loop("printing")
        try:
            print_job = self.printer.print_text(printable_text)
        finally:
            stop_printing_feedback()
        self._emit_kv("automation_print_text", printable_text, max_chars=self._config_int("automation_emit_text_max_chars", 512, 64, 8_192))
        self._emit_kv("automation_name", self._coerce_text(getattr(entry, "name", None)))
        self._emit_kv("automation_id", self._coerce_text(getattr(entry, "automation_id", None)))
        if print_job:
            self._emit_kv("automation_print_job", print_job)
        self._safe_record_event(
            "automation_print_sent",
            "Scheduled automation sent content to the printer.",
            automation_id=getattr(entry, "automation_id", None),
            name=getattr(entry, "name", None),
            request_source=request_source,
            queue=getattr(self.config, "printer_queue", None),
            job=print_job,
        )
        self._safe_enqueue_multimodal_evidence(
            event_name="print_completed",
            modality="printer",
            source="automation_print",
            message="Scheduled automation finished a printer delivery.",
            data={
                "request_source": request_source,
                "queue": getattr(self.config, "printer_queue", None),
                "job": print_job or "",
                "automation_id": getattr(entry, "automation_id", None),
            },
        )
        self._finalize_printing_output()

    def _recover_automation_output_state(self) -> None:
        self._recover_speaking_output_state()
        self._recover_printing_output_state()

    # -------------------------------------------------------------------------
    # Modernized helper methods for 2026-style reliability, fairness, and
    # least-privilege execution.
    # -------------------------------------------------------------------------

    def _deadline_elapsed(self, *, attr_name: str, interval_name: str, default_seconds: float) -> bool:
        now_ns = time.monotonic_ns()
        attr_name_ns = f"{attr_name}_ns"
        deadline_ns = int(getattr(self, attr_name_ns, 0) or 0)
        if deadline_ns and now_ns < deadline_ns:
            return False

        interval_s = self._config_float(
            interval_name,
            default=default_seconds,
            minimum=0.0,
            maximum=86_400.0,
        )
        next_deadline_ns = now_ns + int(interval_s * 1_000_000_000)
        setattr(self, attr_name_ns, next_deadline_ns)
        setattr(self, attr_name, next_deadline_ns / 1_000_000_000)
        return True

    def _config_value(self, name: str, default):
        config = getattr(self, "config", None)
        if config is not None and hasattr(config, name):
            value = getattr(config, name)
            if value is not None:
                return value
        value = getattr(self, name, None)
        return default if value is None else value

    def _config_float(self, name: str, default: float, minimum: float, maximum: float) -> float:
        value = self._config_value(name, default)
        try:
            numeric = float(value)
        except Exception:
            numeric = float(default)
        return max(minimum, min(maximum, numeric))

    def _config_int(self, name: str, default: int, minimum: int, maximum: int) -> int:
        value = self._config_value(name, default)
        try:
            numeric = int(value)
        except Exception:
            numeric = int(default)
        return max(minimum, min(maximum, numeric))

    def _config_name_set(self, name: str) -> frozenset[str]:
        raw = self._config_value(name, ())
        items: list[str]
        if raw is None:
            items = []
        elif isinstance(raw, str):
            items = [part.strip() for part in raw.split(",")]
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            items = [self._coerce_text(part).strip() for part in raw]
        else:
            items = [self._coerce_text(raw).strip()]
        return frozenset(item for item in items if item)

    def _emit_kv(self, key: str, value: object, *, max_chars: int | None = None) -> None:
        safe_key = self._coerce_text(key).strip() or "automation_event"
        safe_key = re.sub(r"[^A-Za-z0-9_.:-]", "_", safe_key)
        safe_value = self._sanitize_text(value, max_chars=max_chars)
        self._safe_emit(f"{safe_key}={safe_value}")

    def _sanitize_text(self, value: object, *, max_chars: int | None = None) -> str:
        text = self._coerce_text(value)
        if not text and value is not None and not isinstance(value, str):
            text = repr(value)
        text = "".join(ch if ch.isprintable() else " " for ch in text).strip()
        limit = max_chars or self._config_int("automation_emit_value_max_chars", 256, 32, 16_384)
        if len(text) > limit:
            return f"{text[: max(0, limit - 1)]}…"
        return text

    def _safe_exception_text(self, exc: Exception) -> str:
        return self._sanitize_text(exc, max_chars=self._config_int("automation_error_max_chars", 240, 64, 4_096))

    def _sanitize_jsonish(
        self,
        value,
        *,
        max_depth: int = 3,
        max_items: int = 12,
        max_string_chars: int = 256,
        _depth: int = 0,
    ):
        if _depth >= max_depth:
            return self._sanitize_text(value, max_chars=max_string_chars)

        if value is None or isinstance(value, (bool, int, float)):
            return value

        if isinstance(value, str):
            return self._sanitize_text(value, max_chars=max_string_chars)

        if isinstance(value, Mapping):
            sanitized: dict[str, object] = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= max_items:
                    sanitized["_truncated_items"] = max(0, len(value) - max_items)
                    break
                sanitized[self._sanitize_text(key, max_chars=64)] = self._sanitize_jsonish(
                    item,
                    max_depth=max_depth,
                    max_items=max_items,
                    max_string_chars=max_string_chars,
                    _depth=_depth + 1,
                )
            return sanitized

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            items = list(value[:max_items])
            sanitized_items = [
                self._sanitize_jsonish(
                    item,
                    max_depth=max_depth,
                    max_items=max_items,
                    max_string_chars=max_string_chars,
                    _depth=_depth + 1,
                )
                for item in items
            ]
            if len(value) > max_items:
                sanitized_items.append({"_truncated_items": len(value) - max_items})
            return sanitized_items

        return self._sanitize_text(value, max_chars=max_string_chars)

    def _stable_digest(self, value) -> str:
        try:
            payload = json.dumps(
                self._sanitize_jsonish(value, max_depth=4, max_items=24, max_string_chars=512),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        except Exception:
            payload = self._sanitize_text(value, max_chars=2_048)
        return hashlib.blake2s(payload.encode("utf-8", errors="replace"), digest_size=12).hexdigest()

    def _normalize_and_limit_event_names(self, event_names) -> tuple[str, ...]:
        normalized = tuple(self._normalize_event_names(event_names))
        max_events = self._config_int("sensor_automation_max_event_names", 8, 1, 128)
        max_chars = self._config_int("sensor_automation_event_name_max_chars", 64, 8, 512)
        limited = tuple(
            self._sanitize_text(name, max_chars=max_chars)
            for name in normalized[:max_events]
            if self._sanitize_text(name, max_chars=max_chars)
        )
        if limited:
            return limited
        return ("sensor.state",)

    def _join_event_names(self, event_names: Sequence[str]) -> str:
        joined = ",".join(event_names) or "sensor.state"
        return self._sanitize_text(
            joined,
            max_chars=self._config_int("sensor_automation_event_label_max_chars", 256, 32, 2_048),
        )

    def _safe_queue_task_done(self, queue_obj) -> None:
        task_done = getattr(queue_obj, "task_done", None)
        if not callable(task_done):
            return
        try:
            task_done()
        except ValueError:
            return
        except Exception:
            return

    def _is_queue_shutdown_exception(self, exc: Exception) -> bool:
        return _QueueShutDown is not None and isinstance(exc, _QueueShutDown)

    def _safe_release_due_reminder(self, reminder) -> bool:
        runtime = getattr(self, "runtime", None)
        if runtime is None:
            return False

        reminder_id = self._coerce_text(getattr(reminder, "reminder_id", None))
        method_names = (
            "release_due_reminder",
            "release_reserved_due_reminder",
            "cancel_reserved_reminder",
            "cancel_due_reminder_reservation",
            "unreserve_due_reminder",
            "return_due_reminder",
        )
        for method_name in method_names:
            method = getattr(runtime, method_name, None)
            if not callable(method):
                continue
            for candidate in (reminder_id, reminder):
                if candidate in ("", None):
                    continue
                try:
                    method(candidate)  # pylint: disable=not-callable
                    return True
                except TypeError:
                    continue
                except Exception:
                    break
        return False

    def _automation_execution_state_lock(self) -> threading.RLock:
        lock = getattr(self, "_automation_execution_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._automation_execution_lock = lock
        return lock

    def _try_begin_automation_execution(self, automation_id: str) -> bool:
        with self._automation_execution_state_lock():
            inflight = getattr(self, "_automation_inflight_ids", None)
            if inflight is None:
                inflight = set()
                self._automation_inflight_ids = inflight
            if automation_id in inflight:
                return False
            inflight.add(automation_id)
            return True

    def _finish_automation_execution(self, automation_id: str) -> None:
        with self._automation_execution_state_lock():
            inflight = getattr(self, "_automation_inflight_ids", None)
            if inflight is not None:
                inflight.discard(automation_id)

    def _automation_execution_key(
        self,
        entry: AutomationDefinition,
        *,
        trigger_source: str,
        event_name: str | None,
        facts: dict[str, object] | None,
        scheduled_for_at: datetime | None,
    ) -> str:
        automation_id = self._coerce_text(getattr(entry, "automation_id", None)) or "unknown"
        marker = None
        for attr_name in (
            "run_id",
            "schedule_run_id",
            "scheduled_run_id",
            "trigger_id",
            "scheduled_for",
            "scheduled_at",
            "due_at",
            "run_at",
            "last_fire_time",
            "next_fire_time",
            "triggered_at",
            "acquired_until",
        ):
            marker_value = self._coerce_text(getattr(entry, attr_name, None))
            if marker_value:
                marker = f"{attr_name}:{marker_value}"
                break

        if marker is None and scheduled_for_at is not None:
            marker = f"scheduled_for:{scheduled_for_at.isoformat()}"

        if trigger_source == "sensor":
            marker = marker or f"sensor:{self._stable_digest({'event_name': event_name or '', 'facts': facts or {}})}"
        else:
            marker = marker or f"fallback:{self._fallback_execution_slot_token(automation_id, trigger_source)}"

        event_marker = self._sanitize_text(event_name or "", max_chars=128)
        key = f"{automation_id}:{trigger_source}:{marker}"
        if event_marker:
            key = f"{key}:{event_marker}"
        return self._sanitize_text(key, max_chars=256)

    def _fallback_execution_slot_token(self, automation_id: str, trigger_source: str) -> str:
        slot_key = f"{automation_id}:{trigger_source}"
        now_ns = time.monotonic_ns()
        ttl_s = self._config_float("automation_checkpoint_ttl_s", 300.0, 30.0, 86_400.0)
        ttl_ns = int(ttl_s * 1_000_000_000)
        with self._automation_execution_state_lock():
            slots = getattr(self, "_automation_fallback_slots", None)
            if slots is None:
                slots = {}
                self._automation_fallback_slots = slots
            slot = slots.get(slot_key)
            if slot is None or (now_ns - int(slot.get("updated_ns", 0) or 0)) > ttl_ns:
                slot = {
                    "token": self._stable_digest({"automation_id": automation_id, "trigger_source": trigger_source, "opened_ns": now_ns}),
                    "updated_ns": now_ns,
                }
                slots[slot_key] = slot
            else:
                slot["updated_ns"] = now_ns
            return slot["token"]

    def _cleanup_stale_automation_execution_state(self) -> None:
        ttl_s = self._config_float("automation_checkpoint_ttl_s", 300.0, 30.0, 86_400.0)
        ttl_ns = int(ttl_s * 1_000_000_000)
        cutoff_ns = time.monotonic_ns() - ttl_ns
        with self._automation_execution_state_lock():
            state = getattr(self, "_automation_action_state", None)
            if state:
                for key in list(state.keys()):
                    if int(state[key].get("updated_ns", 0) or 0) < cutoff_ns:
                        state.pop(key, None)

            slots = getattr(self, "_automation_fallback_slots", None)
            if slots:
                for key in list(slots.keys()):
                    if int(slots[key].get("updated_ns", 0) or 0) < cutoff_ns:
                        slots.pop(key, None)

    def _automation_action_checkpoint_token(
        self,
        entry: AutomationDefinition,
        action: AutomationAction,
        *,
        action_index: int,
    ) -> str:
        return self._stable_digest(
            {
                "automation_id": self._coerce_text(getattr(entry, "automation_id", None)),
                "action_index": action_index,
                "kind": self._coerce_text(getattr(action, "kind", None)),
                "text": self._coerce_text(getattr(action, "text", None)),
                "tool_name": self._coerce_text(getattr(action, "tool_name", None)),
                "payload": self._validated_automation_payload(
                    action,
                    context=(
                        "automation checkpoint payload "
                        f"{self._coerce_text(getattr(entry, 'automation_id', None)) or 'unknown'}"
                    ),
                    allow_large_payload=True,
                ),
            }
        )

    def _automation_action_already_completed(self, execution_key: str, checkpoint_token: str) -> bool:
        with self._automation_execution_state_lock():
            state = getattr(self, "_automation_action_state", None)
            if not state:
                return False
            completed = state.get(execution_key, {}).get("completed", set())
            return checkpoint_token in completed

    def _mark_automation_action_completed(self, execution_key: str, checkpoint_token: str) -> None:
        with self._automation_execution_state_lock():
            state = getattr(self, "_automation_action_state", None)
            if state is None:
                state = {}
                self._automation_action_state = state
            execution_state = state.setdefault(execution_key, {"completed": set(), "updated_ns": 0})
            execution_state["completed"].add(checkpoint_token)
            execution_state["updated_ns"] = time.monotonic_ns()

    def _clear_automation_execution_state(self, execution_key: str) -> None:
        with self._automation_execution_state_lock():
            state = getattr(self, "_automation_action_state", None)
            if state is not None:
                state.pop(execution_key, None)

    def _validated_automation_payload(
        self,
        action: AutomationAction,
        *,
        context: str,
        allow_large_payload: bool = False,
    ) -> dict[str, object]:
        payload = self._automation_payload(action)
        if payload is None:
            return {}
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"{context} must be a mapping")
        payload_dict = dict(payload)

        if allow_large_payload:
            return payload_dict

        max_payload_bytes = self._config_int(
            "automation_payload_max_bytes",
            default=32_768,
            minimum=1_024,
            maximum=1_048_576,
        )
        try:
            encoded = json.dumps(
                self._sanitize_jsonish(payload_dict, max_depth=5, max_items=64, max_string_chars=2_048),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8", errors="replace")
        except Exception:
            encoded = self._sanitize_text(payload_dict, max_chars=max_payload_bytes).encode("utf-8", errors="replace")
        if len(encoded) > max_payload_bytes:
            raise RuntimeError(f"{context} exceeds {max_payload_bytes} bytes")
        return payload_dict

    def _normalize_automation_delivery(self, payload: Mapping[str, object]) -> str:
        requested = payload.get("delivery")
        try:
            return normalize_delivery(requested)
        except Exception:
            fallback = self._sanitize_text(
                self._config_value("automation_default_delivery", "spoken"),
                max_chars=16,
            ).casefold()
            if fallback not in {"spoken", "printed"}:
                fallback = "spoken"
            self._safe_record_event(
                "automation_invalid_delivery",
                "Twinr fell back to the default automation delivery because the configured delivery was invalid.",
                level="warning",
                requested_delivery=self._sanitize_text(requested, max_chars=64),
                fallback_delivery=fallback,
            )
            self._emit_kv("automation_invalid_delivery", requested)
            return fallback

    def _prepare_automation_text(self, text: object, *, context: str, channel: str) -> str:
        required = self._require_non_empty_text(text, context=context)
        if channel == "speech":
            max_chars = self._config_int("automation_spoken_text_max_chars", 1_500, 64, 32_000)
        elif channel == "print":
            max_chars = self._config_int("automation_print_text_max_chars", 6_000, 64, 128_000)
        elif channel == "prompt":
            max_chars = self._config_int("automation_prompt_text_max_chars", 4_000, 64, 64_000)
        else:
            max_chars = self._config_int("automation_text_max_chars", 4_000, 64, 64_000)

        if len(required) <= max_chars:
            return required

        truncated = f"{required[: max_chars - 1]}…"
        self._safe_record_event(
            "automation_text_truncated",
            "Twinr truncated oversized automation text before execution.",
            level="warning",
            channel=channel,
            context=context,
            original_chars=len(required),
            max_chars=max_chars,
        )
        return truncated

    def _prepare_generated_automation_text(self, text: object, *, context: str, delivery: str) -> str:
        channel = "print" if delivery == "printed" else "speech"
        return self._prepare_automation_text(text, context=context, channel=channel)

    def _background_web_search_allowed(self, payload: Mapping[str, object]) -> bool:
        requested = self._coerce_bool(payload.get("allow_web_search", False))
        if not requested:
            return False
        # BREAKING: autonomous background web search is now opt-in at the device/config level.
        globally_allowed = self._coerce_bool(
            self._config_value("background_automation_allow_web_search", False)
        )
        if globally_allowed:
            return True
        self._safe_record_event(
            "automation_web_search_blocked",
            "Twinr blocked autonomous background web search because the device configuration did not explicitly opt in.",
            level="warning",
        )
        return False

    def _normalize_tool_name(self, tool_name: object) -> str:
        normalized = self._require_non_empty_text(tool_name, context="automation tool name").strip()
        if not _TOOL_NAME_PATTERN.fullmatch(normalized):
            raise RuntimeError("Unsupported automation tool name format")
        return normalized

    def _automation_tool_allowed(self, tool_name: str) -> bool:
        # BREAKING: background tool_call actions are deny-by-default unless explicitly allowed.
        if self._coerce_bool(self._config_value("automation_allow_unsafe_tool_calls", False)):
            return True
        allowlist = self._config_name_set("automation_tool_allowlist")
        return "*" in allowlist or tool_name in allowlist

    def _ensure_background_delivery_allowed(self) -> None:
        blocked_reason = self._background_block_reason()
        if blocked_reason is not None:
            raise _AutomationDeferredDueToBackground(blocked_reason, executed_partial=False)

    def _call_with_supported_kwargs(self, func, /, *args, **kwargs):
        try:
            signature = inspect.signature(func)
        except Exception:
            return func(*args, **kwargs)

        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_var_kwargs:
            return func(*args, **kwargs)

        filtered_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters
        }
        return func(*args, **filtered_kwargs)
