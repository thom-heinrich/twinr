"""Handle reminder scheduling tool calls for realtime Twinr sessions.

Normalizes reminder payloads, calls the runtime scheduler, and returns stable
success or failure payloads for spoken tool flows.
"""

from __future__ import annotations

import asyncio  # AUDIT-FIX(#2): Support best-effort async instrumentation from a synchronous handler.
import inspect  # AUDIT-FIX(#3): Detect unexpected awaitables returned by runtime hooks.
import re  # AUDIT-FIX(#5): Restrict free-form `kind` values to a stable token format.
from collections.abc import Mapping  # AUDIT-FIX(#1): Validate that tool arguments are a mapping before reading them.
from typing import Any

_MAX_DUE_AT_LENGTH = 128  # AUDIT-FIX(#1): Bound inputs to stop bogus or abusive payloads on an RPi-sized system.
_MAX_SUMMARY_LENGTH = 200
_MAX_DETAILS_LENGTH = 4000
_MAX_KIND_LENGTH = 32
_MAX_ORIGINAL_REQUEST_LENGTH = 4000
_MAX_REMINDER_ID_LENGTH = 128
_MAX_EMIT_VALUE_LENGTH = 80

_KIND_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


class _ReminderToolError(RuntimeError):
    pass


class _ReminderToolInputError(_ReminderToolError):
    pass


class _ReminderToolRuntimeError(_ReminderToolError):
    pass


def _safe_getattr(object_: Any, attribute: str, default: Any = None) -> Any:
    # AUDIT-FIX(#3): Treat owner/runtime contract drift as a missing hook instead of exploding on attribute access.
    try:
        return getattr(object_, attribute)
    except Exception:
        return default


def _read_text_argument(
    arguments: Mapping[str, object],
    key: str,
    *,
    max_length: int,
    required: bool = False,
    allow_multiline: bool = False,
) -> str:
    # AUDIT-FIX(#1): Reject non-string payloads instead of silently coercing None, lists, dicts, or booleans into bogus reminder content.
    raw_value = arguments.get(key, "")
    if raw_value is None:
        raw_value = ""
    if not isinstance(raw_value, str):
        raise _ReminderToolInputError(f"`{key}` must be a string")
    value = raw_value.strip()
    if "\x00" in value:
        raise _ReminderToolInputError(f"`{key}` contains invalid characters")
    if not allow_multiline and any(character in value for character in ("\r", "\n", "\t")):
        raise _ReminderToolInputError(f"`{key}` must be a single line")
    if len(value) > max_length:
        raise _ReminderToolInputError(f"`{key}` is too long")
    if required and not value:
        raise _ReminderToolInputError(f"`{key}` is required")
    return value


def _normalize_kind(kind: str, *, error_type: type[_ReminderToolError]) -> str:
    # AUDIT-FIX(#5): Canonicalize `kind` to a small, stable token space so downstream state and analytics do not drift on casing or punctuation.
    normalized = kind.strip().casefold() or "reminder"
    if len(normalized) > _MAX_KIND_LENGTH or not _KIND_RE.fullmatch(normalized):
        raise error_type("`kind` must use lowercase letters, numbers, `_` or `-`")
    return normalized


def _sanitize_emit_value(value: object, *, max_length: int = _MAX_EMIT_VALUE_LENGTH) -> str:
    # AUDIT-FIX(#4): Remove control characters and separators before interpolating user-derived data into event strings.
    try:
        text = str(value)
    except Exception:
        return "unknown"
    sanitized = "".join(
        character if character.isprintable() and character not in "\r\n\t=" else " "
        for character in text
    )
    collapsed = " ".join(sanitized.split())
    if not collapsed:
        return "unknown"
    return collapsed[:max_length]


def _consume_task_exception(task: asyncio.Task[Any]) -> None:
    try:
        task.result()
    except Exception:
        return


def _run_best_effort_awaitable(awaitable: Any) -> None:
    # AUDIT-FIX(#2): Avoid unhandled coroutine warnings while keeping instrumentation failures non-fatal.
    if inspect.iscoroutine(awaitable):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                asyncio.run(awaitable)
            except Exception:
                return
            return
        task = loop.create_task(awaitable)
        task.add_done_callback(_consume_task_exception)
        return

    close = getattr(awaitable, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            return


def _best_effort_call(function: Any, *args: object, **kwargs: object) -> None:
    # AUDIT-FIX(#2): Instrumentation must never abort a successful reminder schedule.
    if not callable(function):
        return
    try:
        result = function(*args, **kwargs)
    except Exception:
        return
    if inspect.isawaitable(result):
        _run_best_effort_awaitable(result)


def _read_entry_text(
    entry: Any,
    field_name: str,
    *,
    fallback: str | None = None,
    max_length: int,
) -> str:
    # AUDIT-FIX(#3): Validate the scheduler return contract explicitly instead of assuming attributes exist and are well-formed.
    try:
        raw_value = getattr(entry, field_name)
    except AttributeError:
        raw_value = fallback
    except Exception as exc:
        raise _ReminderToolRuntimeError(f"schedule_reminder returned invalid `{field_name}`") from exc

    if raw_value is None:
        raw_value = fallback
    if raw_value is None:
        raise _ReminderToolRuntimeError(f"schedule_reminder returned no `{field_name}`")

    try:
        value = str(raw_value).strip()
    except Exception as exc:
        raise _ReminderToolRuntimeError(f"schedule_reminder returned invalid `{field_name}`") from exc

    if not value:
        raise _ReminderToolRuntimeError(f"schedule_reminder returned empty `{field_name}`")
    if "\x00" in value:
        raise _ReminderToolRuntimeError(f"schedule_reminder returned invalid `{field_name}`")
    if len(value) > max_length:
        raise _ReminderToolRuntimeError(f"schedule_reminder returned overlong `{field_name}`")
    return value


def _read_due_at_iso(entry: Any) -> str:
    # AUDIT-FIX(#7): Serialize `due_at` once, defensively, so malformed return objects cannot trigger multiple secondary failures.
    try:
        due_at_value = getattr(entry, "due_at")
    except AttributeError as exc:
        raise _ReminderToolRuntimeError("schedule_reminder returned no `due_at`") from exc
    except Exception as exc:
        raise _ReminderToolRuntimeError("schedule_reminder returned invalid `due_at`") from exc

    isoformat = getattr(due_at_value, "isoformat", None)
    if callable(isoformat):
        try:
            due_at_text = str(isoformat())
        except Exception as exc:
            raise _ReminderToolRuntimeError("schedule_reminder returned invalid `due_at`") from exc
    else:
        try:
            due_at_text = str(due_at_value).strip()
        except Exception as exc:
            raise _ReminderToolRuntimeError("schedule_reminder returned invalid `due_at`") from exc

    if not due_at_text:
        raise _ReminderToolRuntimeError("schedule_reminder returned empty `due_at`")
    if len(due_at_text) > _MAX_DUE_AT_LENGTH:
        raise _ReminderToolRuntimeError("schedule_reminder returned overlong `due_at`")
    return due_at_text


def _resolve_schedule_reminder(owner: Any) -> Any:
    runtime = _safe_getattr(owner, "runtime")
    schedule_reminder = _safe_getattr(runtime, "schedule_reminder")
    if not callable(schedule_reminder):
        raise _ReminderToolRuntimeError("reminder scheduler is unavailable")
    return schedule_reminder


def _call_schedule_reminder(schedule_reminder: Any, **kwargs: object) -> Any:
    # AUDIT-FIX(#3): Fail fast with a controlled error if the scheduler contract changes to async or returns an unsupported shape.
    try:
        result = schedule_reminder(**kwargs)
    except Exception as exc:
        raise _ReminderToolRuntimeError("reminder scheduler rejected the request") from exc

    if inspect.iscoroutine(result):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                return asyncio.run(result)
            except Exception as exc:
                raise _ReminderToolRuntimeError("reminder scheduler rejected the request") from exc
        result.close()
        raise _ReminderToolRuntimeError(
            "reminder scheduler returned an async result to a synchronous handler"
        )

    if inspect.isawaitable(result):
        close = getattr(result, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        raise _ReminderToolRuntimeError(
            "reminder scheduler returned an unsupported async result"
        )

    return result


def _failure_response(error: str, message: str) -> dict[str, object]:
    # AUDIT-FIX(#6): Return a structured, plain-language failure payload instead of throwing raw internal exceptions at the caller.
    return {
        "status": "error",
        "error": error,
        "message": message,
    }


def handle_schedule_reminder(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Schedule a reminder or timer from a tool payload.

    Args:
        owner: Tool executor owner exposing runtime, telemetry, and audit
            hooks.
        arguments: Tool payload with required ``due_at`` and ``summary`` plus
            optional ``details``, ``kind``, and ``original_request`` fields.

    Returns:
        JSON-safe payload with ``status="scheduled"`` on success or a stable
        failure payload with an error code and user-facing message.
    """
    due_at_for_failure = ""
    kind_for_failure = "reminder"

    emit = _safe_getattr(owner, "emit")
    record_event = _safe_getattr(owner, "_record_event")

    try:
        _best_effort_call(emit, "reminder_tool_call=true")  # AUDIT-FIX(#2): Track tool invocation even when scheduling later fails.

        if not isinstance(arguments, Mapping):
            raise _ReminderToolInputError("tool arguments must be an object")  # AUDIT-FIX(#1): Reject non-mapping payloads before field access.

        due_at = _read_text_argument(arguments, "due_at", max_length=_MAX_DUE_AT_LENGTH, required=True)
        summary = _read_text_argument(arguments, "summary", max_length=_MAX_SUMMARY_LENGTH, required=True)
        details = _read_text_argument(arguments, "details", max_length=_MAX_DETAILS_LENGTH, allow_multiline=True)
        kind = _normalize_kind(
            _read_text_argument(arguments, "kind", max_length=_MAX_KIND_LENGTH),
            error_type=_ReminderToolInputError,
        )
        original_request = _read_text_argument(
            arguments,
            "original_request",
            max_length=_MAX_ORIGINAL_REQUEST_LENGTH,
            allow_multiline=True,
        )

        due_at_for_failure = due_at
        kind_for_failure = kind

        schedule_reminder = _resolve_schedule_reminder(owner)
        entry = _call_schedule_reminder(
            schedule_reminder,
            due_at=due_at,
            summary=summary,
            details=details or None,
            kind=kind,
            source="schedule_reminder",
            original_request=original_request or None,
        )

        reminder_id = _read_entry_text(entry, "reminder_id", max_length=_MAX_REMINDER_ID_LENGTH)
        entry_kind = _normalize_kind(
            _read_entry_text(entry, "kind", fallback=kind, max_length=_MAX_KIND_LENGTH),
            error_type=_ReminderToolRuntimeError,
        )
        entry_summary = _read_entry_text(
            entry,
            "summary",
            fallback=summary,
            max_length=_MAX_SUMMARY_LENGTH,
        )
        due_at_iso = _read_due_at_iso(entry)

        _best_effort_call(
            emit,
            f"reminder_scheduled={_sanitize_emit_value(entry_summary)}",
        )  # AUDIT-FIX(#4): Sanitize user-derived summary before interpolating it into event strings.
        _best_effort_call(
            emit,
            f"reminder_due_at={_sanitize_emit_value(due_at_iso, max_length=64)}",
        )  # AUDIT-FIX(#4): Sanitize due_at before emitting it to avoid control-character injection.
        _best_effort_call(
            record_event,
            "reminder_tool_scheduled",
            "Realtime tool scheduled a reminder or timer.",
            reminder_id=reminder_id,
            kind=entry_kind,
            due_at=due_at_iso,
        )

        return {
            "status": "scheduled",
            "reminder_id": reminder_id,
            "kind": entry_kind,
            "summary": entry_summary,
            "due_at": due_at_iso,
        }
    except _ReminderToolInputError:
        safe_due_at = (
            _sanitize_emit_value(due_at_for_failure, max_length=_MAX_DUE_AT_LENGTH)
            if due_at_for_failure
            else None
        )
        safe_kind = _sanitize_emit_value(kind_for_failure, max_length=_MAX_KIND_LENGTH)
        _best_effort_call(emit, "reminder_schedule_failed=true")  # AUDIT-FIX(#6): Emit structured failure state for graceful recovery.
        _best_effort_call(emit, "reminder_error=invalid_arguments")
        _best_effort_call(
            record_event,
            "reminder_tool_schedule_failed",
            "Realtime tool could not schedule a reminder because the request was invalid.",
            kind=safe_kind,
            due_at=safe_due_at,
            reason="invalid_arguments",
        )
        return _failure_response(
            "invalid_arguments",
            "I couldn't set the reminder because the time or note was missing or invalid.",
        )
    except _ReminderToolRuntimeError:
        safe_due_at = (
            _sanitize_emit_value(due_at_for_failure, max_length=_MAX_DUE_AT_LENGTH)
            if due_at_for_failure
            else None
        )
        safe_kind = _sanitize_emit_value(kind_for_failure, max_length=_MAX_KIND_LENGTH)
        _best_effort_call(emit, "reminder_schedule_failed=true")  # AUDIT-FIX(#2): Keep scheduler and owner-hook failures inside the tool boundary.
        _best_effort_call(emit, "reminder_error=scheduler_failure")
        _best_effort_call(
            record_event,
            "reminder_tool_schedule_failed",
            "Realtime tool failed to schedule a reminder or timer.",
            kind=safe_kind,
            due_at=safe_due_at,
            reason="scheduler_failure",
        )
        return _failure_response(
            "scheduler_failure",
            "I couldn't set the reminder right now. Please try again.",
        )
    except Exception:
        safe_due_at = (
            _sanitize_emit_value(due_at_for_failure, max_length=_MAX_DUE_AT_LENGTH)
            if due_at_for_failure
            else None
        )
        safe_kind = _sanitize_emit_value(kind_for_failure, max_length=_MAX_KIND_LENGTH)
        _best_effort_call(emit, "reminder_schedule_failed=true")  # AUDIT-FIX(#2): Contain unknown failures instead of crashing the request path.
        _best_effort_call(emit, "reminder_error=unexpected_failure")
        _best_effort_call(
            record_event,
            "reminder_tool_schedule_failed",
            "Realtime tool hit an unexpected error while scheduling a reminder or timer.",
            kind=safe_kind,
            due_at=safe_due_at,
            reason="unexpected_failure",
        )
        return _failure_response(
            "unexpected_failure",
            "I couldn't set the reminder right now. Please try again.",
        )
