from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from queue import Empty, Full
import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from twinr.agent.tools.handlers.automations import normalize_delivery
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


@dataclass(slots=True)
class _LocalMetadataResponse:
    text: str
    model: str = "local_fallback"
    response_id: str | None = None
    request_id: str | None = None
    token_usage: dict[str, int] | None = None
    used_web_search: bool = False


class TwinrRealtimeBackgroundMixin:
    # AUDIT-FIX(#8): Telemetry/status side-effects must be best-effort so completed speech/print actions do not get reclassified as failures.
    def _remember_background_fault(self, source: str, error: Exception | str) -> None:
        faults = getattr(self, "_background_faults", None)
        if not isinstance(faults, list):
            faults = []
            setattr(self, "_background_faults", faults)
        faults.append(
            {
                "source": source,
                "error": str(error),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        del faults[:-20]

    # AUDIT-FIX(#8): `emit()` is observability, not business logic; it must never crash delivery paths.
    def _safe_emit(self, message: str) -> None:
        try:
            self.emit(message)
        except Exception as exc:
            self._remember_background_fault("emit", exc)

    # AUDIT-FIX(#8): File-backed ops/event logging must not turn a successful action into a failed one.
    def _safe_record_event(self, event: str, message: str, **kwargs: object) -> None:
        try:
            self._record_event(event, message, **kwargs)
        except Exception as exc:
            self._remember_background_fault(f"record_event:{event}", exc)

    # AUDIT-FIX(#8): Usage accounting is best-effort telemetry and must not break user-visible work.
    def _safe_record_usage(self, **kwargs: object) -> None:
        try:
            self._record_usage(**kwargs)
        except Exception as exc:
            self._remember_background_fault("record_usage", exc)

    # AUDIT-FIX(#8): Status fan-out failures should not wedge the background loop after state transitions already happened.
    def _safe_emit_status(self, *, force: bool = False) -> None:
        try:
            self._emit_status(force=force)
        except Exception as exc:
            self._remember_background_fault("emit_status", exc)

    # AUDIT-FIX(#4): Multimodal evidence persistence is secondary; a memory-store hiccup must not drop the live sensor pipeline.
    def _safe_enqueue_multimodal_evidence(self, **kwargs: object) -> None:
        try:
            self.runtime.long_term_memory.enqueue_multimodal_evidence(**kwargs)
        except Exception as exc:
            self._remember_background_fault("enqueue_multimodal_evidence", exc)

    # AUDIT-FIX(#3): Post-delivery follow-up logic must not bubble up and tear down the background loop.
    def _safe_run_proactive_follow_up(self, trigger: SocialTriggerDecision) -> None:
        try:
            self._run_proactive_follow_up(trigger)
        except Exception as exc:
            self._remember_background_fault("run_proactive_follow_up", exc)
            self._safe_emit(f"social_follow_up_error={exc}")
            self._safe_record_event(
                "social_trigger_follow_up_failed",
                "A proactive social follow-up failed after the prompt was already delivered.",
                level="error",
                trigger=getattr(trigger, "trigger_id", None),
                error=str(exc),
            )

    # AUDIT-FIX(#8): Cleanup paths must survive secondary exceptions, otherwise the device can get stuck in answering state.
    def _finalize_speaking_output(self) -> None:
        self.runtime.finish_speaking()
        self._safe_emit_status(force=True)

    # AUDIT-FIX(#8): Failure recovery must be best-effort; cleanup errors should not mask the original problem.
    def _recover_speaking_output_state(self) -> None:
        if getattr(getattr(self.runtime, "status", None), "value", None) != "answering":
            return
        try:
            self.runtime.finish_speaking()
        except Exception as exc:
            self._remember_background_fault("finish_speaking", exc)
        self._safe_emit_status(force=True)

    # AUDIT-FIX(#8): Printing state finalization needs the same protection as TTS finalization.
    def _finalize_printing_output(self) -> None:
        self.runtime.finish_printing()
        self._safe_emit_status(force=True)

    # AUDIT-FIX(#8): Printing recovery must not cascade exceptions during already-failing automations.
    def _recover_printing_output_state(self) -> None:
        if getattr(getattr(self.runtime, "status", None), "value", None) != "printing":
            return
        try:
            self.runtime.finish_printing()
        except Exception as exc:
            self._remember_background_fault("finish_printing", exc)
        self._safe_emit_status(force=True)

    # AUDIT-FIX(#12): Governor bookkeeping should fail closed for delivery decisions but must not crash the loop during cleanup.
    def _safe_cancel_governor_reservation(self, reservation: ProactiveGovernorReservation) -> None:
        try:
            self.runtime.proactive_governor.cancel(reservation)
        except Exception as exc:
            self._remember_background_fault("governor_cancel", exc)

    # AUDIT-FIX(#8): Governor delivery bookkeeping is policy state, not a reason to retroactively fail spoken output.
    def _safe_mark_governor_delivered(self, reservation: ProactiveGovernorReservation) -> None:
        try:
            self.runtime.proactive_governor.mark_delivered(reservation)
        except Exception as exc:
            self._remember_background_fault("governor_mark_delivered", exc)

    # AUDIT-FIX(#8): Governor skip bookkeeping during error handling must never raise a second exception.
    def _safe_mark_governor_skipped(
        self,
        reservation: ProactiveGovernorReservation,
        *,
        reason: str,
    ) -> None:
        try:
            self.runtime.proactive_governor.mark_skipped(reservation, reason=reason)
        except Exception as exc:
            self._remember_background_fault("governor_mark_skipped", exc)

    # AUDIT-FIX(#12): Failure-path state updates must not raise and hide the original delivery error.
    def _safe_mark_long_term_proactive_candidate_skipped(self, reservation, *, reason: str) -> None:
        try:
            self.runtime.mark_long_term_proactive_candidate_skipped(reservation, reason=reason)
        except Exception as exc:
            self._remember_background_fault("mark_long_term_proactive_candidate_skipped", exc)

    # AUDIT-FIX(#12): Reminder failure bookkeeping is important, but secondary write failures must not crash the worker.
    def _safe_mark_reminder_failed(self, reminder_id: str, *, error: str) -> None:
        try:
            self.runtime.mark_reminder_failed(reminder_id, error=error)
        except Exception as exc:
            self._remember_background_fault("mark_reminder_failed", exc)

    # AUDIT-FIX(#9): Poll intervals come from config/env and need strict validation to prevent busy loops or type crashes on the RPi.
    def _config_interval_seconds(self, attr_name: str, default: float) -> float:
        raw_value = getattr(self.config, attr_name, default)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            self._remember_background_fault(f"config_interval:{attr_name}", f"invalid interval {raw_value!r}")
            return max(0.25, default)
        if value != value or value <= 0.0:
            self._remember_background_fault(f"config_interval:{attr_name}", f"invalid interval {raw_value!r}")
            return max(0.25, default)
        return max(0.25, value)

    # AUDIT-FIX(#9): Monotonic deadlines are persisted on the instance and can be missing or malformed during startup/reload.
    def _monotonic_deadline(self, attr_name: str) -> float:
        raw_value = getattr(self, attr_name, 0.0)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return 0.0
        if value != value:
            return 0.0
        return value

    # AUDIT-FIX(#5): Invalid timezone names from `.env` must degrade to UTC instead of crashing reminders and proactive delivery.
    def _local_timezone_name(self) -> str:
        configured = str(getattr(self.config, "local_timezone_name", "") or "").strip()
        if not configured:
            return "UTC"
        try:
            ZoneInfo(configured)
        except ZoneInfoNotFoundError as exc:
            self._remember_background_fault("local_timezone_name", exc)
            if not bool(getattr(self, "_invalid_local_timezone_notified", False)):
                setattr(self, "_invalid_local_timezone_notified", True)
                self._safe_record_event(
                    "invalid_local_timezone_fallback",
                    "Twinr fell back to UTC because the configured local timezone name was invalid.",
                    level="warning",
                    configured_timezone=configured,
                    fallback_timezone="UTC",
                    error=str(exc),
                )
            return "UTC"
        return configured

    # AUDIT-FIX(#5): All local-time calculations must use the validated timezone helper to avoid config-driven crashes.
    def _local_timezone(self) -> ZoneInfo:
        return ZoneInfo(self._local_timezone_name())

    # AUDIT-FIX(#11): Dynamic state and config values need explicit coercion instead of relying on brittle implicit conversions.
    def _coerce_text(self, value: object) -> str:
        if value is None:
            return ""
        return str(value).strip()

    # AUDIT-FIX(#7): Empty spoken/printed payloads are user-visible failures and must be rejected before output starts.
    def _require_non_empty_text(self, value: object, *, context: str) -> str:
        text = self._coerce_text(value)
        if text:
            return text
        raise RuntimeError(f"{context} is empty")

    # AUDIT-FIX(#6): String booleans like `"false"` must not silently become True and enable unintended web access.
    def _coerce_bool(self, value: object, *, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off", ""}:
                return False
        return default

    # AUDIT-FIX(#11): Priorities/confidences can arrive as None/NaN/strings and must be clamped defensively.
    def _coerce_priority(
        self,
        value: object,
        *,
        default: int,
        minimum: int = 1,
        maximum: int = 99,
    ) -> int:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        if number != number:
            return default
        return max(minimum, min(maximum, int(number)))

    # AUDIT-FIX(#11): Confidence values should share one hardened conversion path instead of repeated inline math.
    def _confidence_to_priority(self, confidence: object, *, default: int = 50) -> int:
        try:
            scaled = float(confidence) * 100.0
        except (TypeError, ValueError):
            return default
        if scaled != scaled:
            return default
        return self._coerce_priority(scaled, default=default)

    # AUDIT-FIX(#6): Automation payloads are stored state and may be missing or malformed; callers need a guaranteed dict.
    def _automation_payload(self, action: AutomationAction) -> dict[str, object]:
        payload = getattr(action, "payload", None)
        return payload if isinstance(payload, dict) else {}

    # AUDIT-FIX(#2): Sensor facts cross queue/state boundaries; they must be recursively cloned to avoid shared-mutable races.
    def _clone_background_value(self, value: object) -> object:
        if isinstance(value, dict):
            return {key: self._clone_background_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._clone_background_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._clone_background_value(item) for item in value)
        if isinstance(value, set):
            return {self._clone_background_value(item) for item in value}
        return value

    # AUDIT-FIX(#11): Event names can come from persisted or external state and must be normalized before joins/comparisons.
    def _normalize_event_names(self, event_names: object) -> tuple[str, ...]:
        try:
            raw_items = tuple(event_names or ())
        except TypeError:
            return ()
        normalized: list[str] = []
        for item in raw_items:
            text = self._coerce_text(item)
            if text:
                normalized.append(text)
        return tuple(normalized)

    # AUDIT-FIX(#10): ISO timestamps may be naive; normalize them to UTC before any recency computation.
    def _parse_event_timestamp(self, value: object) -> float | None:
        if isinstance(value, datetime):
            when = value
        else:
            text = self._coerce_text(value)
            if not text:
                return None
            try:
                when = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return when.astimezone(timezone.utc).timestamp()

    # AUDIT-FIX(#2): Sensor freshness must support monotonic timestamps, epoch timestamps, and ISO datetimes without mixing clocks.
    def _observation_age_seconds(self, observed_at: object) -> float | None:
        if isinstance(observed_at, datetime):
            timestamp = self._parse_event_timestamp(observed_at)
            if timestamp is None:
                return None
            return max(0.0, datetime.now(timezone.utc).timestamp() - timestamp)

        if isinstance(observed_at, str):
            stripped = observed_at.strip()
            if not stripped:
                return None
            try:
                return self._observation_age_seconds(datetime.fromisoformat(stripped.replace("Z", "+00:00")))
            except ValueError:
                observed_at = stripped

        try:
            value = float(observed_at)
        except (TypeError, ValueError):
            return None
        if value != value:
            return None

        monotonic_now = time.monotonic()
        if 0.0 <= value <= monotonic_now + 60.0:
            return max(0.0, monotonic_now - value)

        epoch_now = datetime.now(timezone.utc).timestamp()
        if 946684800.0 <= value <= epoch_now + 3600.0:
            return max(0.0, epoch_now - value)

        return None

    # AUDIT-FIX(#7): Due reminders need a deterministic local fallback so a backend hiccup never produces silence.
    def _default_due_reminder_text(self, reminder) -> str:
        summary = self._coerce_text(getattr(reminder, "summary", None))
        details = self._coerce_text(getattr(reminder, "details", None))
        original_request = self._coerce_text(getattr(reminder, "original_request", None))
        reminder_text = summary or details or original_request
        if reminder_text:
            return f"Reminder. {reminder_text}"
        return "This is your reminder."

    # AUDIT-FIX(#5): Formatting helper must tolerate invalid timezone config and naive datetimes without exploding.
    def _safe_format_due_label(self, value: object) -> str:
        try:
            return format_due_label(value, timezone_name=self._local_timezone_name())
        except Exception as exc:
            self._remember_background_fault("format_due_label", exc)
            if isinstance(value, datetime):
                when = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
                try:
                    return when.astimezone(self._local_timezone()).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    return when.isoformat()
            return self._coerce_text(value)

    # AUDIT-FIX(#1): Failed automations need bounded retry state instead of being permanently marked as successfully triggered.
    def _automation_failure_backoff_seconds(self) -> float:
        default = max(30.0, self._config_interval_seconds("automation_poll_interval_s", 5.0) * 2.0)
        raw_value = getattr(self.config, "automation_failure_retry_backoff_s", default)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return default
        if value != value or value < 1.0:
            return default
        return min(value, 3600.0)

    # AUDIT-FIX(#1): Retry state must be tracked per automation so one bad definition does not starve the whole scheduler.
    def _automation_failure_backoff_map(self) -> dict[str, float]:
        state = getattr(self, "_automation_failure_backoff_until", None)
        if not isinstance(state, dict):
            state = {}
            setattr(self, "_automation_failure_backoff_until", state)
        return state

    # AUDIT-FIX(#1): Time-based automations should be skipped only while their own retry backoff window is active.
    def _automation_retry_blocked(self, automation_id: str) -> bool:
        backoff_map = self._automation_failure_backoff_map()
        until = backoff_map.get(automation_id)
        if until is None:
            return False
        if time.monotonic() >= until:
            backoff_map.pop(automation_id, None)
            return False
        return True

    # AUDIT-FIX(#1): Failures should schedule a retry instead of recording a false success in persisted automation state.
    def _register_automation_failure_backoff(self, automation_id: str) -> None:
        self._automation_failure_backoff_map()[automation_id] = (
            time.monotonic() + self._automation_failure_backoff_seconds()
        )

    # AUDIT-FIX(#1): Successful execution must clear any stale retry backoff.
    def _clear_automation_failure_backoff(self, automation_id: str) -> None:
        self._automation_failure_backoff_map().pop(automation_id, None)

    def handle_social_trigger(self, trigger: SocialTriggerDecision) -> bool:
        # AUDIT-FIX(#11): Trigger metadata is dynamic input; coerce it once and reuse the validated values.
        trigger_id = self._coerce_text(getattr(trigger, "trigger_id", None)) or "unknown"
        priority = self._coerce_priority(getattr(trigger, "priority", None), default=50)
        default_prompt = self._coerce_text(getattr(trigger, "prompt", None))
        trigger_reason = self._coerce_text(getattr(trigger, "reason", None))
        safety_trigger = is_safety_trigger(trigger_id)

        if not self._background_work_allowed():
            skip_reason = "busy" if getattr(getattr(self.runtime, "status", None), "value", None) != "waiting" else "conversation_active"
            self._safe_emit(f"social_trigger_skipped={skip_reason}")
            self._safe_record_event(
                "social_trigger_skipped",
                "Social trigger prompt was skipped because Twinr was not idle for background prompts.",
                trigger=trigger_id,
                reason=trigger_reason,
                prompt=default_prompt,
                priority=priority,
                skip_reason=skip_reason,
            )
            return False

        governor_reservation = self._reserve_governed_prompt(
            ProactiveGovernorCandidate(
                source_kind="social",
                source_id=trigger_id,
                summary=default_prompt,
                priority=priority,
                presence_session_id=self._current_presence_session_id(),
                safety_exempt=safety_trigger,
                counts_toward_presence_budget=not safety_trigger,
            )
        )
        if governor_reservation is None:
            return False

        phrase_response = None
        prompt_mode = proactive_prompt_mode(trigger)
        prompt_text = default_prompt
        try:
            if prompt_mode == "llm":
                stop_processing_feedback = self._start_working_feedback_loop("processing")
                try:
                    try:
                        phrase_response = self.agent_provider.phrase_proactive_prompt_with_metadata(
                            trigger_id=trigger_id,
                            reason=trigger_reason,
                            default_prompt=default_prompt,
                            priority=priority,
                            conversation=self.runtime.conversation_context(),
                            recent_prompts=self._recent_proactive_prompts(trigger_id=trigger_id),
                            observation_facts=proactive_observation_facts(trigger),
                        )
                    finally:
                        stop_processing_feedback()
                    candidate_prompt = self._coerce_text(getattr(phrase_response, "text", None))
                    if candidate_prompt:
                        prompt_text = candidate_prompt
                    else:
                        prompt_mode = "default_fallback"
                        self._safe_emit("social_prompt_fallback=empty_phrase")
                except Exception as exc:
                    prompt_mode = "default_fallback"
                    self._safe_emit("social_prompt_fallback=default")
                    self._safe_emit(f"social_prompt_phrase_error={exc}")
                    self._safe_record_event(
                        "social_trigger_phrase_fallback",
                        "Twinr fell back to the default proactive prompt after proactive phrasing failed.",
                        level="warning",
                        trigger=trigger_id,
                        error=str(exc),
                    )

            # AUDIT-FIX(#7): Never begin a spoken prompt with empty text; blank output is a real user-visible failure.
            prompt = self.runtime.begin_proactive_prompt(
                self._require_non_empty_text(prompt_text, context=f"social trigger {trigger_id} prompt")
            )
            self._safe_emit_status(force=True)
            tts_started = time.monotonic()
            tts_ms, first_audio_ms = self._play_streaming_tts_with_feedback(prompt, turn_started=tts_started)
            self._finalize_speaking_output()
            self._safe_mark_governor_delivered(governor_reservation)
            self._safe_emit(f"social_trigger={trigger_id}")
            self._safe_emit(f"social_trigger_priority={priority}")
            self._safe_emit(f"social_prompt_mode={prompt_mode}")
            self._safe_emit(f"social_prompt={prompt}")
            if phrase_response is not None:
                if getattr(phrase_response, "response_id", None):
                    self._safe_emit(f"social_response_id={phrase_response.response_id}")
                if getattr(phrase_response, "request_id", None):
                    self._safe_emit(f"social_request_id={phrase_response.request_id}")
                self._safe_record_usage(
                    request_kind="proactive_prompt",
                    source="realtime_loop",
                    model=getattr(phrase_response, "model", "unknown"),
                    response_id=getattr(phrase_response, "response_id", None),
                    request_id=getattr(phrase_response, "request_id", None),
                    used_web_search=False,
                    token_usage=getattr(phrase_response, "token_usage", None),
                    proactive_trigger=trigger_id,
                )
            self._safe_emit(f"timing_social_tts_ms={tts_ms}")
            if first_audio_ms is not None:
                self._safe_emit(f"timing_social_first_audio_ms={first_audio_ms}")
            self._safe_record_event(
                "social_trigger_prompted",
                "Twinr spoke a proactive social prompt.",
                trigger=trigger_id,
                reason=trigger_reason,
                priority=priority,
                prompt=prompt,
                default_prompt=default_prompt,
                prompt_mode=prompt_mode,
            )
            self._safe_run_proactive_follow_up(trigger)
            return True
        except Exception as exc:
            # AUDIT-FIX(#3): Social-trigger failures must be contained to this call path instead of crashing the background worker.
            self._recover_speaking_output_state()
            self._safe_mark_governor_skipped(
                governor_reservation,
                reason=f"delivery_failed: {exc}",
            )
            self._safe_emit(f"social_trigger_error={exc}")
            self._safe_record_event(
                "social_trigger_failed",
                "A proactive social trigger failed during delivery.",
                level="error",
                trigger=trigger_id,
                reason=trigger_reason,
                priority=priority,
                error=str(exc),
            )
            return False

    def handle_sensor_observation(self, facts: dict[str, object], event_names: tuple[str, ...]) -> None:
        # AUDIT-FIX(#2): Clone nested sensor facts so producers cannot mutate queued/shared observations after handoff.
        cloned_facts = self._clone_background_value(facts)
        copied_facts = cloned_facts if isinstance(cloned_facts, dict) else dict(facts)
        normalized_event_names = self._normalize_event_names(event_names)
        self._latest_sensor_observation_facts = copied_facts

        # AUDIT-FIX(#4): Never block on a full observation queue; drop the oldest entry and keep the newest live state.
        try:
            self._sensor_observation_queue.put_nowait((copied_facts, normalized_event_names))
        except Full:
            try:
                self._sensor_observation_queue.get_nowait()
            except Empty:
                pass
            try:
                self._sensor_observation_queue.put_nowait((copied_facts, normalized_event_names))
            except Full as exc:
                self._remember_background_fault("sensor_observation_queue", exc)
                self._safe_emit("sensor_observation_queue_overflow=true")
                self._safe_record_event(
                    "sensor_observation_queue_overflow",
                    "Twinr dropped a sensor observation because the queue stayed full.",
                    level="warning",
                    event_names=list(normalized_event_names),
                )

        self._safe_enqueue_multimodal_evidence(
            event_name="sensor_observation",
            modality="sensor",
            source="proactive_monitor",
            message="Changed multimodal sensor observation recorded.",
            data={
                "facts": copied_facts,
                "event_names": list(normalized_event_names),
            },
        )

    def _maybe_deliver_due_reminder(self) -> bool:
        # AUDIT-FIX(#9): Validate timer state and poll interval before touching reminder delivery logic.
        now_monotonic = time.monotonic()
        if now_monotonic < self._monotonic_deadline("_next_reminder_check_at"):
            return False
        self._next_reminder_check_at = now_monotonic + self._config_interval_seconds("reminder_poll_interval_s", 5.0)
        if not self._background_work_allowed():
            return False

        governor_reservation: ProactiveGovernorReservation | None = None
        try:
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
                self._safe_cancel_governor_reservation(governor_reservation)
                return False
            if due_entries[0].reminder_id != preview_entries[0].reminder_id:
                self._safe_cancel_governor_reservation(governor_reservation)
                return False
            return self._deliver_due_reminder(due_entries[0], governor_reservation=governor_reservation)
        except Exception as exc:
            # AUDIT-FIX(#12): Runtime/state-store failures in polling paths must be contained so later polls can recover.
            if governor_reservation is not None:
                self._safe_cancel_governor_reservation(governor_reservation)
            self._safe_emit(f"reminder_due_check_error={exc}")
            self._safe_record_event(
                "reminder_due_check_failed",
                "Twinr failed while checking for due reminders.",
                level="error",
                error=str(exc),
            )
            return False

    def _maybe_run_due_automation(self) -> bool:
        # AUDIT-FIX(#1): Iterate across due automations so one blocked/failing entry cannot starve the rest of the queue.
        now_monotonic = time.monotonic()
        if now_monotonic < self._monotonic_deadline("_next_automation_check_at"):
            return False
        self._next_automation_check_at = now_monotonic + self._config_interval_seconds("automation_poll_interval_s", 5.0)
        if not self._background_work_allowed():
            return False

        try:
            due_entries = tuple(self.runtime.due_time_automations() or ())
            if not due_entries:
                return False
            for entry in due_entries:
                automation_id = self._coerce_text(getattr(entry, "automation_id", None))
                if automation_id and self._automation_retry_blocked(automation_id):
                    continue
                if self._run_automation_entry(entry, trigger_source="time_schedule"):
                    return True
            return False
        except Exception as exc:
            # AUDIT-FIX(#12): Scheduler errors must not stop future automation polls.
            self._safe_emit(f"automation_due_check_error={exc}")
            self._safe_record_event(
                "automation_due_check_failed",
                "Twinr failed while checking for due automations.",
                level="error",
                error=str(exc),
            )
            return False

    def _maybe_run_sensor_automation(self) -> bool:
        if not self._background_work_allowed():
            return False
        while True:
            try:
                facts, event_names = self._sensor_observation_queue.get_nowait()
            except Empty:
                return False
            except Exception as exc:
                # AUDIT-FIX(#12): Queue/runtime faults in the polling loop must not stop later sensor automation delivery.
                self._safe_emit(f"sensor_automation_queue_error={exc}")
                self._safe_record_event(
                    "sensor_automation_queue_failed",
                    "Twinr failed while reading the sensor automation queue.",
                    level="error",
                    error=str(exc),
                )
                return False
            try:
                normalized_event_names = self._normalize_event_names(event_names)
                if self._run_matching_sensor_automations(facts=facts, event_names=normalized_event_names):
                    return True
            except Exception as exc:
                # AUDIT-FIX(#12): A single bad sensor event must be dropped, logged, and not kill the background engine.
                self._safe_emit(f"sensor_automation_error={exc}")
                self._safe_record_event(
                    "sensor_automation_failed",
                    "Twinr failed while executing sensor-triggered automations.",
                    level="error",
                    event_names=list(self._normalize_event_names(event_names)),
                    error=str(exc),
                )

    def _background_work_allowed(self) -> bool:
        # AUDIT-FIX(#11): Background gating should tolerate partially initialized runtime state during startup/recovery.
        return (
            getattr(getattr(self.runtime, "status", None), "value", None) == "waiting"
            and not bool(getattr(self, "_conversation_session_active", False))
        )

    def _current_presence_session_id(self) -> int | None:
        monitor = getattr(self, "proactive_monitor", None)
        coordinator = None if monitor is None else getattr(monitor, "coordinator", None)
        snapshot = None if coordinator is None else getattr(coordinator, "latest_presence_snapshot", None)
        if snapshot is None or not getattr(snapshot, "armed", False):
            return None
        session_id = getattr(snapshot, "session_id", None)
        try:
            return int(session_id) if session_id is not None else None
        except (TypeError, ValueError):
            return None

    def _current_longterm_live_facts(self) -> dict[str, object] | None:
        facts = getattr(self, "_latest_sensor_observation_facts", None)
        if not isinstance(facts, dict):
            return None

        # AUDIT-FIX(#2): Long-term proactive logic must read an immutable snapshot and compute freshness against the correct clock source.
        cloned_facts = self._clone_background_value(facts)
        copied = cloned_facts if isinstance(cloned_facts, dict) else dict(facts)

        sensor = copied.get("sensor")
        if isinstance(sensor, dict):
            age_s = self._observation_age_seconds(sensor.get("observed_at"))
            if age_s is not None and age_s > 45.0:
                return None
        copied["last_response_available"] = bool(getattr(self.runtime, "last_response", None))
        copied["recent_print_completed"] = self._recent_print_completed()
        return copied

    def _recent_print_completed(self, *, within_s: float = 900.0) -> bool:
        cutoff = datetime.now(timezone.utc).timestamp() - max(0.0, within_s)
        for entry in reversed(tuple(self.runtime.ops_events.tail(limit=40) or ())):
            # AUDIT-FIX(#10): Persisted ops-event rows can be malformed; ignore bad entries instead of crashing recency checks.
            if not isinstance(entry, dict):
                continue
            if entry.get("event") not in {"print_finished", "print_completed"}:
                continue
            when = self._parse_event_timestamp(entry.get("created_at", ""))
            if when is None:
                continue
            return when >= cutoff
        return False

    def _longterm_proactive_observation_facts(
        self,
        *,
        candidate,
        live_facts: dict[str, object] | None,
    ) -> tuple[str, ...]:
        # AUDIT-FIX(#11): Candidate/persisted-memory fields can be missing or malformed; sanitize them before prompt construction.
        candidate_kind = self._coerce_text(getattr(candidate, "kind", None)) or "unknown"
        facts = [
            f"candidate_kind={candidate_kind}",
            f"sensitivity={self._coerce_text(getattr(candidate, 'sensitivity', None)) or 'unknown'}",
        ]
        if candidate_kind.startswith("routine_"):
            source = None
            for memory_id in tuple(getattr(candidate, "source_memory_ids", ()) or ()):
                source = self.runtime.long_term_memory.object_store.get_object(memory_id)
                if source is not None:
                    break
            raw_attrs = getattr(source, "attributes", None) if source is not None else None
            attrs = raw_attrs if isinstance(raw_attrs, dict) else {}
            if attrs.get("routine_type"):
                facts.append(f"sensor_routine_type={attrs.get('routine_type')}")
            if attrs.get("interaction_type"):
                facts.append(f"sensor_interaction_type={attrs.get('interaction_type')}")
            if attrs.get("deviation_type"):
                facts.append(f"sensor_deviation_type={attrs.get('deviation_type')}")
            if attrs.get("daypart"):
                facts.append(f"sensor_daypart={attrs.get('daypart')}")
            if attrs.get("weekday_class"):
                facts.append(f"sensor_weekday_class={attrs.get('weekday_class')}")
            if live_facts is not None:
                camera = live_facts.get("camera")
                vad = live_facts.get("vad")
                if isinstance(camera, dict):
                    facts.append(f"live_person_visible={bool(camera.get('person_visible'))}")
                    facts.append(f"live_looking_toward_device={bool(camera.get('looking_toward_device'))}")
                    facts.append(f"live_hand_or_object_near_camera={bool(camera.get('hand_or_object_near_camera'))}")
                    facts.append(f"live_body_pose={camera.get('body_pose') or 'unknown'}")
                if isinstance(vad, dict):
                    facts.append(f"live_quiet={bool(vad.get('quiet'))}")
                    facts.append(f"live_speech_detected={bool(vad.get('speech_detected'))}")
                facts.append(f"last_response_available={bool(live_facts.get('last_response_available'))}")
                facts.append(f"recent_print_completed={bool(live_facts.get('recent_print_completed'))}")
        return tuple(facts)

    def _reserve_governed_prompt(
        self,
        candidate: ProactiveGovernorCandidate,
    ) -> ProactiveGovernorReservation | None:
        # AUDIT-FIX(#12): Governor errors should fail closed for this prompt, not crash the whole background worker.
        try:
            decision = self.runtime.proactive_governor.try_reserve(candidate)
        except Exception as exc:
            emit_prefix = {
                "social": "social_trigger",
                "longterm": "longterm_proactive",
                "reminder": "reminder",
                "automation": "automation",
            }.get(candidate.source_kind, candidate.source_kind)
            self._safe_emit(f"{emit_prefix}_reservation_error={exc}")
            self._safe_record_event(
                "proactive_governor_error",
                "The shared proactive governor raised an error while evaluating a delivery candidate.",
                level="error",
                source_kind=candidate.source_kind,
                source_id=candidate.source_id,
                summary=candidate.summary,
                error=str(exc),
            )
            return None

        if decision.allowed:
            return decision.reservation

        emit_prefix = {
            "social": "social_trigger",
            "longterm": "longterm_proactive",
            "reminder": "reminder",
            "automation": "automation",
        }.get(candidate.source_kind, candidate.source_kind)
        self._safe_emit(f"{emit_prefix}_skipped={decision.reason}")
        self._safe_record_event(
            "proactive_governor_blocked",
            "Proactive delivery was blocked by the shared governor policy.",
            source_kind=candidate.source_kind,
            source_id=candidate.source_id,
            summary=candidate.summary,
            reason=decision.reason,
            channel=getattr(candidate, "channel", None),
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
        bounded_limit = max(1, int(limit))
        for entry in reversed(tuple(self.runtime.ops_events.tail(limit=100) or ())):
            # AUDIT-FIX(#11): Persisted ops events may have bad shapes; sanitize before reading prompt history.
            if not isinstance(entry, dict):
                continue
            if entry.get("event") not in {"social_trigger_prompted", "longterm_proactive_prompted"}:
                continue
            data = entry.get("data")
            if not isinstance(data, dict):
                data = {}
            if trigger_id is not None and data.get("trigger") != trigger_id:
                continue
            prompt = self._coerce_text(data.get("prompt", ""))
            if not prompt:
                continue
            prompts.append(prompt)
            if len(prompts) >= bounded_limit:
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
        event_label = ",".join(self._normalize_event_names(event_names)) or "sensor.state"
        # AUDIT-FIX(#11): Automation names come from persisted state; sort on a sanitized string to avoid `None.lower()` crashes.
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

    def _maybe_run_long_term_memory_proactive(self) -> bool:
        # AUDIT-FIX(#9): Poll timing for long-term proactive work needs the same interval hardening as reminders/automations.
        now_monotonic = time.monotonic()
        if now_monotonic < self._monotonic_deadline("_next_long_term_memory_proactive_check_at"):
            return False
        self._next_long_term_memory_proactive_check_at = (
            now_monotonic + self._config_interval_seconds("long_term_memory_proactive_poll_interval_s", 30.0)
        )
        if not self._background_work_allowed():
            return False

        governor_reservation: ProactiveGovernorReservation | None = None
        reservation = None
        candidate_id: str | None = None
        try:
            live_facts = self._current_longterm_live_facts()
            preview = self.runtime.preview_long_term_proactive_candidate(live_facts=live_facts)
            if preview is None:
                return False

            candidate_id = self._coerce_text(getattr(preview, "candidate_id", None)) or "unknown"
            governor_reservation = self._reserve_governed_prompt(
                ProactiveGovernorCandidate(
                    source_kind="longterm",
                    source_id=candidate_id,
                    summary=self._coerce_text(getattr(preview, "summary", None)),
                    priority=self._confidence_to_priority(getattr(preview, "confidence", None), default=50),
                    presence_session_id=self._current_presence_session_id(),
                    safety_exempt=False,
                    counts_toward_presence_budget=True,
                )
            )
            if governor_reservation is None:
                return False

            reservation = self.runtime.reserve_specific_long_term_proactive_candidate(
                preview,
                now=datetime.now(self._local_timezone()),
            )
            if reservation is None or reservation.candidate.candidate_id != getattr(preview, "candidate_id", None):
                self._safe_cancel_governor_reservation(governor_reservation)
                return False

            candidate = reservation.candidate
            response = None
            prompt_mode = "default"
            prompt_text = self._coerce_text(getattr(candidate, "summary", None))
            trigger_id = f"longterm:{self._coerce_text(getattr(candidate, 'candidate_id', None)) or 'unknown'}"
            try:
                stop_processing_feedback = self._start_working_feedback_loop("processing")
                try:
                    response = self.agent_provider.phrase_proactive_prompt_with_metadata(
                        trigger_id=trigger_id,
                        reason=self._coerce_text(getattr(candidate, "rationale", None)),
                        default_prompt=prompt_text,
                        priority=self._confidence_to_priority(getattr(candidate, "confidence", None), default=50),
                        conversation=self.runtime.conversation_context(),
                        recent_prompts=self._recent_proactive_prompts(trigger_id=trigger_id),
                        observation_facts=self._longterm_proactive_observation_facts(
                            candidate=candidate,
                            live_facts=live_facts,
                        ),
                    )
                finally:
                    stop_processing_feedback()
                candidate_prompt = self._coerce_text(getattr(response, "text", None))
                if candidate_prompt:
                    prompt_text = candidate_prompt
                    prompt_mode = "llm"
                else:
                    prompt_mode = "default_fallback"
            except Exception as exc:
                # AUDIT-FIX(#8): Prompt-phrasing fallbacks should log best-effort and then continue with the default summary text.
                prompt_mode = "default_fallback"
                self._safe_emit("longterm_proactive_phrase_fallback=default")
                self._safe_emit(f"longterm_proactive_phrase_error={exc}")
                self._safe_record_event(
                    "longterm_proactive_phrase_fallback",
                    "Twinr fell back to the default long-term proactive prompt after phrasing failed.",
                    level="warning",
                    candidate_id=self._coerce_text(getattr(candidate, "candidate_id", None)),
                    error=str(exc),
                )

            spoken_prompt = self.runtime.begin_proactive_prompt(
                self._require_non_empty_text(
                    prompt_text,
                    context=f"long-term proactive candidate {self._coerce_text(getattr(candidate, 'candidate_id', None)) or 'unknown'} prompt",
                )
            )
            self._safe_emit_status(force=True)
            tts_started = time.monotonic()
            tts_ms, first_audio_ms = self._play_streaming_tts_with_feedback(
                spoken_prompt,
                turn_started=tts_started,
            )
            self._finalize_speaking_output()
            self.runtime.mark_long_term_proactive_candidate_delivered(
                reservation,
                prompt_text=spoken_prompt,
            )
            self._safe_mark_governor_delivered(governor_reservation)
            self._safe_emit(f"longterm_proactive_candidate={self._coerce_text(getattr(candidate, 'candidate_id', None))}")
            self._safe_emit(f"longterm_proactive_kind={self._coerce_text(getattr(candidate, 'kind', None))}")
            self._safe_emit(f"longterm_proactive_prompt_mode={prompt_mode}")
            self._safe_emit(f"longterm_proactive_prompt={spoken_prompt}")
            self._safe_emit(f"timing_longterm_proactive_tts_ms={tts_ms}")
            if first_audio_ms is not None:
                self._safe_emit(f"timing_longterm_proactive_first_audio_ms={first_audio_ms}")
            if response is not None:
                self._safe_record_usage(
                    request_kind="longterm_proactive_prompt",
                    source="realtime_loop",
                    model=getattr(response, "model", "unknown"),
                    response_id=getattr(response, "response_id", None),
                    request_id=getattr(response, "request_id", None),
                    used_web_search=False,
                    token_usage=getattr(response, "token_usage", None),
                    proactive_trigger=trigger_id,
                )
            self._safe_record_event(
                "longterm_proactive_prompted",
                "Twinr spoke a proactive prompt derived from long-term memory.",
                trigger=trigger_id,
                candidate_id=self._coerce_text(getattr(candidate, "candidate_id", None)),
                candidate_kind=self._coerce_text(getattr(candidate, "kind", None)),
                prompt=spoken_prompt,
                prompt_mode=prompt_mode,
                rationale=self._coerce_text(getattr(candidate, "rationale", None)),
            )
            return True
        except Exception as exc:
            # AUDIT-FIX(#12): Reservation/state failures in long-term proactive delivery must be contained so the loop can recover.
            self._recover_speaking_output_state()
            if reservation is not None:
                self._safe_mark_long_term_proactive_candidate_skipped(
                    reservation,
                    reason=f"delivery_failed: {exc}",
                )
                if governor_reservation is not None:
                    self._safe_mark_governor_skipped(
                        governor_reservation,
                        reason=f"delivery_failed: {exc}",
                    )
            elif governor_reservation is not None:
                self._safe_cancel_governor_reservation(governor_reservation)
            self._safe_emit(f"longterm_proactive_error={exc}")
            self._safe_record_event(
                "longterm_proactive_failed",
                "A long-term proactive prompt failed during delivery.",
                level="error",
                candidate_id=candidate_id,
                error=str(exc),
            )
            return False

    def _deliver_due_reminder(
        self,
        reminder,
        *,
        governor_reservation: ProactiveGovernorReservation,
    ) -> bool:
        response = None
        spoken_prompt = ""
        reminder_id = self._coerce_text(getattr(reminder, "reminder_id", None)) or "unknown"
        try:
            stop_processing_feedback = self._start_working_feedback_loop("processing")
            try:
                response = self._phrase_due_reminder_with_fallback(reminder)
            finally:
                stop_processing_feedback()

            # AUDIT-FIX(#7): Reminder delivery must reject blank backend output and use deterministic fallback text instead.
            spoken_prompt = self.runtime.begin_reminder_prompt(
                self._require_non_empty_text(
                    getattr(response, "text", None),
                    context=f"reminder {reminder_id} prompt",
                )
            )
            self._safe_emit_status(force=True)
            tts_started = time.monotonic()
            tts_ms, first_audio_ms = self._play_streaming_tts_with_feedback(
                spoken_prompt,
                turn_started=tts_started,
            )
            self._finalize_speaking_output()
            delivered = self.runtime.mark_reminder_delivered(reminder.reminder_id)
            self._safe_mark_governor_delivered(governor_reservation)
            self._safe_emit("reminder_delivered=true")
            self._safe_emit(f"reminder_due_at={delivered.due_at.isoformat()}")
            self._safe_emit(f"reminder_text={spoken_prompt}")
            if getattr(response, "response_id", None):
                self._safe_emit(f"reminder_response_id={response.response_id}")
            if getattr(response, "request_id", None):
                self._safe_emit(f"reminder_request_id={response.request_id}")
            self._safe_emit(f"timing_reminder_tts_ms={tts_ms}")
            if first_audio_ms is not None:
                self._safe_emit(f"timing_reminder_first_audio_ms={first_audio_ms}")
            self._safe_record_usage(
                request_kind="reminder_delivery",
                source="realtime_loop",
                model=getattr(response, "model", "unknown"),
                response_id=getattr(response, "response_id", None),
                request_id=getattr(response, "request_id", None),
                used_web_search=False,
                token_usage=getattr(response, "token_usage", None),
                reminder_id=delivered.reminder_id,
                reminder_kind=delivered.kind,
            )
            return True
        except Exception as exc:
            # AUDIT-FIX(#8): Reminder failures must clean up state and log without leaving the device stuck in answering mode.
            self._recover_speaking_output_state()
            self._safe_mark_reminder_failed(reminder.reminder_id, error=str(exc))
            self._safe_mark_governor_skipped(
                governor_reservation,
                reason=f"delivery_failed: {exc}",
            )
            self._safe_emit(f"reminder_error={exc}")
            self._safe_record_event(
                "reminder_delivery_failed",
                "A due reminder failed during delivery.",
                level="error",
                reminder_id=reminder_id,
                spoken_prompt=spoken_prompt,
                error=str(exc),
            )
            return False

    def _phrase_due_reminder_with_fallback(self, reminder):
        # AUDIT-FIX(#7): Due reminders are safety-relevant and need a local deterministic fallback when all provider paths fail.
        fallback_text = self._default_due_reminder_text(reminder)

        helper = getattr(self.agent_provider, "phrase_due_reminder_with_metadata", None)
        if callable(helper):
            try:
                response = helper(reminder)
            except Exception as exc:
                self._safe_emit(f"reminder_backend_primary_error={exc}")
                self._safe_record_event(
                    "reminder_backend_primary_failed",
                    "The dedicated reminder phrasing backend failed.",
                    level="warning",
                    reminder_id=self._coerce_text(getattr(reminder, "reminder_id", None)),
                    error=str(exc),
                )
            else:
                if self._coerce_text(getattr(response, "text", None)):
                    return response
                self._safe_emit("reminder_backend_fallback=empty_primary_phrase")

        generic = getattr(self.agent_provider, "respond_with_metadata", None)
        if callable(generic):
            current_time = datetime.now(self._local_timezone())
            timezone_name = self._local_timezone_name()
            prompt_parts = [
                "A stored Twinr reminder is due now.",
                f"Current local time: {self._safe_format_due_label(current_time)}",
                f"Scheduled reminder time: {self._safe_format_due_label(getattr(reminder, 'due_at', None))}",
                f"Reminder kind: {self._coerce_text(getattr(reminder, 'kind', None)) or 'reminder'}",
                f"Reminder summary: {self._coerce_text(getattr(reminder, 'summary', None))}",
            ]
            if getattr(reminder, "details", None):
                prompt_parts.append(f"Reminder details: {self._coerce_text(getattr(reminder, 'details', None))}")
            if getattr(reminder, "original_request", None):
                prompt_parts.append(
                    f"Original user request: {self._coerce_text(getattr(reminder, 'original_request', None))}"
                )
            prompt_parts.append(f"Use timezone context: {timezone_name}")
            prompt_parts.append("Speak the reminder now.")
            self._safe_emit("reminder_backend_fallback=generic")
            try:
                response = generic(
                    "\n".join(prompt_parts),
                    instructions=REMINDER_DELIVERY_INSTRUCTIONS,
                    allow_web_search=False,
                )
            except Exception as exc:
                self._safe_emit(f"reminder_backend_generic_error={exc}")
                self._safe_record_event(
                    "reminder_backend_generic_failed",
                    "The generic reminder phrasing backend failed.",
                    level="warning",
                    reminder_id=self._coerce_text(getattr(reminder, "reminder_id", None)),
                    error=str(exc),
                )
            else:
                if self._coerce_text(getattr(response, "text", None)):
                    return response
                self._safe_emit("reminder_backend_fallback=empty_generic_phrase")

        self._safe_emit("reminder_backend_fallback=local")
        return _LocalMetadataResponse(text=fallback_text)

    def _run_automation_entry(
        self,
        entry: AutomationDefinition,
        *,
        trigger_source: str,
        event_name: str | None = None,
        facts: dict[str, object] | None = None,
    ) -> bool:
        # AUDIT-FIX(#1): A failed automation must be retried later, not permanently suppressed by marking it as triggered.
        automation_id = self._coerce_text(getattr(entry, "automation_id", None)) or "unknown"
        automation_name = self._coerce_text(getattr(entry, "name", None)) or automation_id
        if self._automation_retry_blocked(automation_id):
            self._safe_emit(f"automation_retry_backoff={automation_id}")
            return False

        governor_reservation: ProactiveGovernorReservation | None = None
        executed = False
        try:
            if self._automation_uses_speech(entry):
                governor_reservation = self._reserve_governed_prompt(
                    ProactiveGovernorCandidate(
                        source_kind="automation",
                        source_id=automation_id,
                        summary=automation_name,
                        priority=70,
                        presence_session_id=self._current_presence_session_id(),
                        safety_exempt=False,
                        counts_toward_presence_budget=True,
                    )
                )
                if governor_reservation is None:
                    return False

            for action in tuple(getattr(entry, "actions", ()) or ()):
                if not getattr(action, "enabled", False):
                    continue
                self._execute_automation_action(entry, action)
                executed = True
            if not executed:
                raise RuntimeError("Automation has no enabled actions")

            self.runtime.mark_automation_triggered(entry.automation_id)
            self._clear_automation_failure_backoff(automation_id)
            if governor_reservation is not None:
                self._safe_mark_governor_delivered(governor_reservation)
            self._safe_emit("automation_executed=true")
            self._safe_emit(f"automation_name={automation_name}")
            self._safe_emit(f"automation_id={automation_id}")
            self._safe_emit(f"automation_trigger_source={trigger_source}")
            if event_name:
                self._safe_emit(f"automation_event_name={event_name}")
            self._safe_record_event(
                "automation_executed",
                "An automation was executed.",
                automation_id=automation_id,
                name=automation_name,
                trigger_source=trigger_source,
                event_name=event_name,
                facts=facts,
            )
            return True
        except Exception as exc:
            self._recover_automation_output_state()
            self._register_automation_failure_backoff(automation_id)
            if governor_reservation is not None:
                self._safe_mark_governor_skipped(
                    governor_reservation,
                    reason=f"delivery_failed: {exc}",
                )
            self._safe_emit(f"automation_error={exc}")
            self._safe_record_event(
                "automation_execution_failed",
                "An automation failed during execution.",
                level="error",
                automation_id=automation_id,
                name=automation_name,
                trigger_source=trigger_source,
                event_name=event_name,
                error=str(exc),
            )
            return False

    def _automation_uses_speech(self, entry: AutomationDefinition) -> bool:
        for action in tuple(getattr(entry, "actions", ()) or ()):
            if not getattr(action, "enabled", False):
                continue
            if getattr(action, "kind", None) == "say":
                return True
            if getattr(action, "kind", None) == "llm_prompt":
                # AUDIT-FIX(#6): Delivery decisions must tolerate malformed payloads and invalid stored boolean/string values.
                payload = self._automation_payload(action)
                try:
                    delivery = normalize_delivery(payload.get("delivery"))
                except Exception:
                    return True
                if delivery != "printed":
                    return True
        return False

    def _execute_automation_action(self, entry: AutomationDefinition, action: AutomationAction) -> None:
        action_kind = getattr(action, "kind", None)
        action_text = getattr(action, "text", None)

        if action_kind == "say":
            self._speak_automation_text(
                entry,
                self._require_non_empty_text(action_text, context="automation say action"),
            )
            return
        if action_kind == "print":
            self._print_automation_text(
                entry,
                self._require_non_empty_text(action_text, context="automation print action"),
                request_source="automation_static",
            )
            return
        if action_kind == "llm_prompt":
            payload = self._automation_payload(action)
            delivery = normalize_delivery(payload.get("delivery"))
            # AUDIT-FIX(#6): Parse boolean flags strictly so `"false"` never becomes truthy and leaks data to web search.
            allow_web_search = self._coerce_bool(payload.get("allow_web_search", False))
            prompt_text = self._require_non_empty_text(action_text, context="automation llm prompt action")
            stop_processing_feedback = self._start_working_feedback_loop("processing")
            try:
                response = self.agent_provider.fulfill_automation_prompt_with_metadata(
                    prompt_text,
                    allow_web_search=allow_web_search,
                    delivery=delivery,
                )
            finally:
                stop_processing_feedback()

            # AUDIT-FIX(#7): Generated automation output must be non-empty before speaking/printing or composing print jobs.
            response_text = self._require_non_empty_text(
                getattr(response, "text", None),
                context=f"automation {self._coerce_text(getattr(entry, 'automation_id', None)) or 'unknown'} llm output",
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
            )
            if delivery == "printed":
                composed = self.agent_provider.compose_print_job_with_metadata(
                    focus_hint=f"{self._coerce_text(getattr(entry, 'name', None))}: {prompt_text}".strip(": "),
                    direct_text=response_text,
                    request_source="automation",
                )
                composed_text = self._require_non_empty_text(
                    getattr(composed, "text", None),
                    context=f"automation {self._coerce_text(getattr(entry, 'automation_id', None)) or 'unknown'} print composition",
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
                )
                self._print_automation_text(entry, composed_text, request_source="automation")
                return
            self._speak_automation_text(entry, response_text)
            return
        raise RuntimeError(f"Unsupported automation action kind during execution: {action_kind}")

    def _speak_automation_text(self, entry: AutomationDefinition, text: str) -> None:
        # AUDIT-FIX(#7): Static and generated automation speech must be non-empty before TTS starts.
        spoken_prompt = self.runtime.begin_automation_prompt(
            self._require_non_empty_text(text, context="automation speech output")
        )
        self._safe_emit_status(force=True)
        tts_started = time.monotonic()
        tts_ms, first_audio_ms = self._play_streaming_tts_with_feedback(
            spoken_prompt,
            turn_started=tts_started,
        )
        self._finalize_speaking_output()
        self._safe_emit(f"automation_spoken={spoken_prompt}")
        self._safe_emit(f"automation_name={self._coerce_text(getattr(entry, 'name', None))}")
        self._safe_emit(f"automation_id={self._coerce_text(getattr(entry, 'automation_id', None))}")
        self._safe_emit(f"timing_automation_tts_ms={tts_ms}")
        if first_audio_ms is not None:
            self._safe_emit(f"timing_automation_first_audio_ms={first_audio_ms}")

    def _print_automation_text(self, entry: AutomationDefinition, text: str, *, request_source: str) -> None:
        # AUDIT-FIX(#7): Empty print jobs are silent failures for seniors and should be rejected before hitting the printer.
        printable_text = self._require_non_empty_text(text, context="automation print output")
        self.runtime.maybe_begin_automation_print()
        self._safe_emit_status(force=True)
        stop_printing_feedback = self._start_working_feedback_loop("printing")
        try:
            print_job = self.printer.print_text(printable_text)
        finally:
            stop_printing_feedback()
        self._safe_emit(f"automation_print_text={printable_text}")
        self._safe_emit(f"automation_name={self._coerce_text(getattr(entry, 'name', None))}")
        self._safe_emit(f"automation_id={self._coerce_text(getattr(entry, 'automation_id', None))}")
        if print_job:
            self._safe_emit(f"automation_print_job={print_job}")
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
        # AUDIT-FIX(#8): Automation recovery must clear whichever output channel was active without raising secondary errors.
        self._recover_speaking_output_state()
        self._recover_printing_output_state()
