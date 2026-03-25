"""Reminder phrasing and delivery helpers for the realtime background loop."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Any

from twinr.memory.reminders import format_due_label

from twinr.agent.workflows.realtime_runtime.background_delivery import BackgroundDeliveryBlocked


@dataclass(slots=True)
class LocalMetadataResponse:
    """Represent a locally generated fallback response for reminder delivery."""

    text: str
    model: str = "local_fallback"
    response_id: str | None = None
    request_id: str | None = None
    token_usage: dict[str, int] | None = None
    used_web_search: bool = False


def default_due_reminder_text(loop: Any, reminder: Any) -> str:
    """Build a deterministic reminder sentence when all provider paths fail."""

    summary = loop._coerce_text(getattr(reminder, "summary", None))
    details = loop._coerce_text(getattr(reminder, "details", None))
    original_request = loop._coerce_text(getattr(reminder, "original_request", None))
    reminder_text = summary or details or original_request
    if reminder_text:
        return f"Reminder. {reminder_text}"
    return "This is your reminder."


def safe_format_due_label(loop: Any, value: object) -> str:
    """Format one due label without letting timezone/config issues bubble up."""

    if not isinstance(value, datetime):
        return loop._coerce_text(value)
    try:
        return format_due_label(value, timezone_name=loop._local_timezone_name())
    except Exception as exc:
        loop._remember_background_fault("format_due_label", exc)
        if isinstance(value, datetime):
            when = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
            try:
                return when.astimezone(loop._local_timezone()).strftime("%Y-%m-%d %H:%M")
            except Exception:
                return when.isoformat()
        return loop._coerce_text(value)


def phrase_due_reminder_with_fallback(
    loop: Any,
    reminder: Any,
    *,
    instructions: str,
) -> Any:
    """Phrase a due reminder via provider helpers with a deterministic local fallback."""

    fallback_text = default_due_reminder_text(loop, reminder)

    helper = getattr(loop.agent_provider, "phrase_due_reminder_with_metadata", None)
    if callable(helper):
        try:
            response = helper(reminder)
        except Exception as exc:
            loop._safe_emit(f"reminder_backend_primary_error={exc}")
            loop._safe_record_event(
                "reminder_backend_primary_failed",
                "The dedicated reminder phrasing backend failed.",
                level="warning",
                reminder_id=loop._coerce_text(getattr(reminder, "reminder_id", None)),
                error=str(exc),
            )
        else:
            if loop._coerce_text(getattr(response, "text", None)):
                return response
            loop._safe_emit("reminder_backend_fallback=empty_primary_phrase")

    generic = getattr(loop.agent_provider, "respond_with_metadata", None)
    if callable(generic):
        current_time = datetime.now(loop._local_timezone())
        timezone_name = loop._local_timezone_name()
        prompt_parts = [
            "A stored Twinr reminder is due now.",
            f"Current local time: {safe_format_due_label(loop, current_time)}",
            f"Scheduled reminder time: {safe_format_due_label(loop, getattr(reminder, 'due_at', None))}",
            f"Reminder kind: {loop._coerce_text(getattr(reminder, 'kind', None)) or 'reminder'}",
            f"Reminder summary: {loop._coerce_text(getattr(reminder, 'summary', None))}",
        ]
        if getattr(reminder, "details", None):
            prompt_parts.append(f"Reminder details: {loop._coerce_text(getattr(reminder, 'details', None))}")
        if getattr(reminder, "original_request", None):
            prompt_parts.append(
                f"Original user request: {loop._coerce_text(getattr(reminder, 'original_request', None))}"
            )
        prompt_parts.append(f"Use timezone context: {timezone_name}")
        prompt_parts.append("Speak the reminder now.")
        loop._safe_emit("reminder_backend_fallback=generic")
        try:
            response = generic(
                "\n".join(prompt_parts),
                instructions=instructions,
                allow_web_search=False,
            )
        except Exception as exc:
            loop._safe_emit(f"reminder_backend_generic_error={exc}")
            loop._safe_record_event(
                "reminder_backend_generic_failed",
                "The generic reminder phrasing backend failed.",
                level="warning",
                reminder_id=loop._coerce_text(getattr(reminder, "reminder_id", None)),
                error=str(exc),
            )
        else:
            if loop._coerce_text(getattr(response, "text", None)):
                return response
            loop._safe_emit("reminder_backend_fallback=empty_generic_phrase")

    loop._safe_emit("reminder_backend_fallback=local")
    return LocalMetadataResponse(text=fallback_text)


def deliver_due_reminder(loop: Any, reminder: Any, *, governor_reservation: Any, instructions: str) -> bool:
    """Deliver one due reminder while preserving idle-window and cleanup invariants."""

    response = None
    spoken_prompt = ""
    reminder_id = loop._coerce_text(getattr(reminder, "reminder_id", None)) or "unknown"
    try:
        stop_processing_feedback = loop._start_working_feedback_loop("processing")
        try:
            response = phrase_due_reminder_with_fallback(loop, reminder, instructions=instructions)
        finally:
            stop_processing_feedback()

        try:
            spoken_prompt = loop._begin_background_delivery(
                lambda: loop.runtime.begin_reminder_prompt(
                    loop._require_non_empty_text(
                        getattr(response, "text", None),
                        context=f"reminder {reminder_id} prompt",
                    )
                )
            )
        except BackgroundDeliveryBlocked as blocked:
            loop._safe_release_reminder_reservation(reminder.reminder_id)
            loop._safe_cancel_governor_reservation(governor_reservation)
            loop._safe_emit(f"reminder_skipped={blocked.reason}")
            loop._safe_record_event(
                "reminder_skipped",
                "A due reminder was released because Twinr stopped being idle before reminder speech started.",
                reminder_id=reminder_id,
                skip_reason=blocked.reason,
            )
            return False
        loop._safe_emit_status(force=True)
        tts_started = time.monotonic()
        tts_ms, first_audio_ms = loop._play_streaming_tts_with_feedback(
            spoken_prompt,
            turn_started=tts_started,
        )
        loop._finalize_speaking_output()
        delivered = loop.runtime.mark_reminder_delivered(reminder.reminder_id)
        loop._safe_mark_governor_delivered(governor_reservation)
        loop._safe_emit("reminder_delivered=true")
        loop._safe_emit(f"reminder_due_at={delivered.due_at.isoformat()}")
        loop._safe_emit(f"reminder_text={spoken_prompt}")
        if getattr(response, "response_id", None):
            loop._safe_emit(f"reminder_response_id={response.response_id}")
        if getattr(response, "request_id", None):
            loop._safe_emit(f"reminder_request_id={response.request_id}")
        loop._safe_emit(f"timing_reminder_tts_ms={tts_ms}")
        if first_audio_ms is not None:
            loop._safe_emit(f"timing_reminder_first_audio_ms={first_audio_ms}")
        loop._safe_record_usage(
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
        loop._recover_speaking_output_state()
        loop._safe_mark_reminder_failed(reminder.reminder_id, error=str(exc))
        loop._safe_mark_governor_skipped(
            governor_reservation,
            reason=f"delivery_failed: {exc}",
        )
        loop._safe_emit(f"reminder_error={exc}")
        loop._safe_record_event(
            "reminder_delivery_failed",
            "A due reminder failed during delivery.",
            level="error",
            reminder_id=reminder_id,
            spoken_prompt=spoken_prompt,
            error=str(exc),
        )
        return False
