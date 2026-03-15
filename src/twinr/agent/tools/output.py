from __future__ import annotations

import re
import threading
import time
import weakref
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from typing import Any, Callable, TypeVar
from zoneinfo import ZoneInfo


_T = TypeVar("_T")

# AUDIT-FIX(#1): Serialize printer access per owner so concurrent tool calls cannot interleave jobs on the same device.
_FALLBACK_PRINT_LOCK = threading.RLock()
_PRINT_LOCKS: weakref.WeakKeyDictionary[object, threading.RLock] = weakref.WeakKeyDictionary()
_PRINT_LOCKS_GUARD = threading.Lock()


# AUDIT-FIX(#7): Validate tool arguments up front so malformed payloads fail predictably instead of crashing on `.get()`.
def _ensure_arguments_mapping(arguments: dict[str, object]) -> Mapping[str, object]:
    if not isinstance(arguments, Mapping):
        raise RuntimeError("tool arguments must be a JSON object")
    return arguments


# AUDIT-FIX(#8): Separate plain stringification from stripped normalization so backend outputs can retain formatting where needed.
def _stringify_text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


# AUDIT-FIX(#8): Normalize optional text inputs so `None` does not become the literal string `'None'`.
def _normalize_optional_text(value: object) -> str:
    return _stringify_text(value).strip()


# AUDIT-FIX(#7): Read text arguments through a single coercion path after payload validation.
def _coerce_argument_text(arguments: Mapping[str, object], key: str) -> str:
    return _normalize_optional_text(arguments.get(key, ""))


# AUDIT-FIX(#9): Reject blank backend outputs so the system does not report success for empty receipts or silent answers.
def _require_non_empty_text(value: object, error_message: str) -> str:
    text = _stringify_text(value)
    if not text.strip():
        raise RuntimeError(error_message)
    return text


# AUDIT-FIX(#8): Normalize backend source collections defensively so `None`, generators, and mixed values do not break telemetry or memory writes.
def _normalize_sources(raw_sources: object) -> list[str]:
    if raw_sources is None:
        return []
    if isinstance(raw_sources, Mapping):
        text = _normalize_optional_text(raw_sources)
        return [text] if text else []
    if isinstance(raw_sources, (str, bytes, bytearray)):
        text = _normalize_optional_text(raw_sources)
        return [text] if text else []

    normalized: list[str] = []
    if isinstance(raw_sources, Iterable):
        for item in raw_sources:
            text = _normalize_optional_text(item)
            if text:
                normalized.append(text)
        return normalized

    text = _normalize_optional_text(raw_sources)
    return [text] if text else []


_TODAY_PATTERNS = (
    re.compile(r"\bheute\b", re.IGNORECASE),
    re.compile(r"\btoday\b", re.IGNORECASE),
)
_TOMORROW_PATTERNS = (
    re.compile(r"\bmorgen\b", re.IGNORECASE),
    re.compile(r"\btomorrow\b", re.IGNORECASE),
)
_DAY_AFTER_TOMORROW_PATTERNS = (
    re.compile(r"\bübermorgen\b", re.IGNORECASE),
    re.compile(r"\buebermorgen\b", re.IGNORECASE),
    re.compile(r"\bday after tomorrow\b", re.IGNORECASE),
)
_YESTERDAY_PATTERNS = (
    re.compile(r"\bgestern\b", re.IGNORECASE),
    re.compile(r"\byesterday\b", re.IGNORECASE),
)


def _resolve_search_date_context(
    question: str,
    provided_date_context: str,
    *,
    timezone_name: str,
    reference: datetime | None = None,
) -> str:
    normalized_question = _normalize_optional_text(question)
    normalized_context = _normalize_optional_text(provided_date_context)
    now = reference
    if now is None:
        try:
            now = datetime.now(ZoneInfo(timezone_name))
        except Exception:
            now = datetime.now()

    offset_days: int | None = None
    if any(pattern.search(normalized_question) for pattern in _DAY_AFTER_TOMORROW_PATTERNS):
        offset_days = 2
    elif any(pattern.search(normalized_question) for pattern in _TOMORROW_PATTERNS):
        offset_days = 1
    elif any(pattern.search(normalized_question) for pattern in _TODAY_PATTERNS):
        offset_days = 0
    elif any(pattern.search(normalized_question) for pattern in _YESTERDAY_PATTERNS):
        offset_days = -1

    if offset_days is None:
        return normalized_context

    resolved = (now + timedelta(days=offset_days)).date().isoformat()
    weekday = (now + timedelta(days=offset_days)).strftime("%A")
    if normalized_context:
        return f"{weekday}, {resolved} ({timezone_name})"
    return f"{weekday}, {resolved} ({timezone_name})"


# AUDIT-FIX(#6): Normalize vision image collections defensively and drop null entries before invoking the vision backend.
def _normalize_images(raw_images: object) -> list[Any]:
    if raw_images is None:
        return []

    if isinstance(raw_images, Iterable) and not isinstance(raw_images, (str, bytes, bytearray)):
        return [item for item in raw_images if item is not None]

    return [raw_images] if raw_images is not None else []


# AUDIT-FIX(#4): Strip control characters from emitted values so user/model text cannot inject fake events into logs or SSE-like streams.
def _sanitize_emit_value(value: object) -> str:
    text = _stringify_text(value)
    return text.replace("\x00", "").replace("\r", "\\r").replace("\n", "\\n")


# AUDIT-FIX(#5): Make non-critical `emit()` calls best-effort so telemetry outages do not turn successful tool executions into user-visible failures.
def _emit_safe(owner: Any, payload: str) -> None:
    try:
        owner.emit(payload)
    except Exception:
        return None


# AUDIT-FIX(#4): Emit structured key/value telemetry through a sanitized path.
def _emit_kv_safe(owner: Any, key: str, value: object) -> None:
    _emit_safe(owner, f"{key}={_sanitize_emit_value(value)}")


# AUDIT-FIX(#5): Make event recording best-effort for the same reason as telemetry emission.
def _record_event_safe(owner: Any, event_name: str, message: str, **data: object) -> None:
    try:
        owner._record_event(event_name, message, **data)
    except Exception:
        return None


# AUDIT-FIX(#5): Make usage recording best-effort so analytics persistence failures do not flip successful tool calls into errors.
def _record_usage_safe(owner: Any, **data: object) -> None:
    try:
        owner._record_usage(**data)
    except Exception:
        return None


# AUDIT-FIX(#5): Generalize best-effort side effects that should never be allowed to fail the primary user action.
def _call_best_effort(
    owner: Any,
    callback: Callable[[], _T],
    *,
    event_name: str,
    message: str,
    default: _T,
    **data: object,
) -> _T:
    try:
        return callback()
    except Exception as exc:
        _record_event_safe(owner, event_name, message, error_type=type(exc).__name__, **data)
        return default


# AUDIT-FIX(#2): Emit bounded, generic failure metadata without leaking raw exception text or secrets into telemetry.
def _emit_tool_failure(owner: Any, tool_name: str, message: str, exc: Exception, **data: object) -> None:
    _emit_kv_safe(owner, f"{tool_name}_status", "failed")
    _emit_kv_safe(owner, f"{tool_name}_error", message)
    _record_event_safe(owner, f"{tool_name}_failed", message, error_type=type(exc).__name__, **data)


# AUDIT-FIX(#3): Stop the feedback loop without masking the original print failure if cleanup itself breaks.
def _stop_feedback_loop_safely(owner: Any, stop_feedback: Callable[[], Any] | None) -> None:
    if stop_feedback is None:
        return
    if not callable(stop_feedback):
        _record_event_safe(
            owner,
            "print_feedback_stop_invalid",
            "Printing feedback stop handler was not callable.",
        )
        return
    try:
        stop_feedback()
    except Exception as exc:
        _record_event_safe(
            owner,
            "print_feedback_stop_failed",
            "Printing feedback loop could not be stopped cleanly.",
            error_type=type(exc).__name__,
        )


# AUDIT-FIX(#1): Use a stable per-owner lock with a process-wide fallback for non-weakrefable owners.
def _get_print_lock(owner: Any) -> threading.RLock:
    try:
        with _PRINT_LOCKS_GUARD:
            lock = _PRINT_LOCKS.get(owner)
            if lock is None:
                lock = threading.RLock()
                _PRINT_LOCKS[owner] = lock
            return lock
    except TypeError:
        return _FALLBACK_PRINT_LOCK


def handle_print_receipt(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    # AUDIT-FIX(#7): Validate the payload shape before reading fields from it.
    arguments = _ensure_arguments_mapping(arguments)
    # AUDIT-FIX(#8): Normalize optional tool arguments without turning missing values into junk strings.
    focus_hint = _coerce_argument_text(arguments, "focus_hint")
    direct_text = _coerce_argument_text(arguments, "text")
    if not focus_hint and not direct_text:
        raise RuntimeError("print_receipt requires `focus_hint` or `text`")

    composed: Any = None
    composed_text = ""
    print_job: Any = None
    stop_printing_feedback: Callable[[], Any] | None = None

    # AUDIT-FIX(#1): Serialize the full print path, including shared printer state and timestamp mutation.
    with _get_print_lock(owner):
        # AUDIT-FIX(#4): Route telemetry through sanitized emission so printed/model text cannot forge extra events.
        _emit_kv_safe(owner, "print_tool_call", "true")
        try:
            owner.runtime.maybe_begin_tool_print()
            # AUDIT-FIX(#5): Status emission is observability-only and must not prevent an otherwise valid print.
            _call_best_effort(
                owner,
                lambda: owner._emit_status(force=True),
                event_name="print_status_emit_failed",
                message="Printing status update failed.",
                default=None,
            )
            # AUDIT-FIX(#5): Working feedback is auxiliary; if it cannot start we continue with the actual print.
            stop_printing_feedback = _call_best_effort(
                owner,
                lambda: owner._start_working_feedback_loop("printing"),
                event_name="print_feedback_start_failed",
                message="Printing feedback loop could not be started.",
                default=None,
            )
            composed = owner.print_backend.compose_print_job_with_metadata(
                conversation=owner.runtime.provider_conversation_context(),
                focus_hint=focus_hint or None,
                direct_text=direct_text or None,
                request_source="tool",
            )
            # AUDIT-FIX(#9): Empty composed text is a failed print, not a successful blank receipt.
            composed_text = _require_non_empty_text(
                getattr(composed, "text", None),
                "print receipt generation returned empty text",
            )
            print_job = owner.printer.print_text(composed_text)
            # AUDIT-FIX(#5): Update the print timestamp immediately after the physical print succeeds, before best-effort telemetry.
            owner._last_print_request_at = time.monotonic()
        except Exception as exc:
            # AUDIT-FIX(#2): Convert backend/printer exceptions into a consistent tool failure signal with preserved traceback chaining.
            _emit_tool_failure(
                owner,
                "print",
                "Print receipt failed.",
                exc,
                focus_hint=focus_hint,
            )
            raise RuntimeError("print receipt failed") from exc
        finally:
            # AUDIT-FIX(#3): Cleanup failures must never overwrite the original print error.
            _stop_feedback_loop_safely(owner, stop_printing_feedback)

    # AUDIT-FIX(#4): Emit printed text through the sanitized event channel.
    _emit_kv_safe(owner, "print_text", composed_text)
    # AUDIT-FIX(#5): Analytics persistence is non-critical and must remain best-effort.
    _record_usage_safe(
        owner,
        request_kind="print",
        source="realtime_tool",
        model=getattr(composed, "model", None),
        response_id=getattr(composed, "response_id", None),
        request_id=getattr(composed, "request_id", None),
        used_web_search=False,
        token_usage=getattr(composed, "token_usage", None),
        request_source="tool",
    )
    if print_job is not None and _normalize_optional_text(print_job):
        # AUDIT-FIX(#4): Emit job identifiers through the sanitized event channel.
        _emit_kv_safe(owner, "print_job", print_job)
    # AUDIT-FIX(#5): Long-term evidence capture is non-critical and must not retroactively fail a completed print.
    _call_best_effort(
        owner,
        lambda: owner.runtime.long_term_memory.enqueue_multimodal_evidence(
            event_name="print_completed",
            modality="printer",
            source="tool_print",
            message="Printed Twinr output was delivered from a tool call.",
            data={
                "request_source": "tool",
                "job": _normalize_optional_text(print_job),
                "focus_hint": focus_hint,
            },
        ),
        event_name="print_memory_store_failed",
        message="Printed output evidence could not be persisted.",
        default=None,
    )
    return {
        "status": "printed",
        "text": composed_text,
        "job": print_job,
    }


def handle_end_conversation(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    # AUDIT-FIX(#7): Validate the payload shape before reading fields from it.
    arguments = _ensure_arguments_mapping(arguments)
    # AUDIT-FIX(#8): Normalize the optional reason field without converting missing values to `'None'`.
    reason = _coerce_argument_text(arguments, "reason")
    if reason:
        # AUDIT-FIX(#4): Emit the stop reason through the sanitized channel to avoid event injection.
        _emit_kv_safe(owner, "end_conversation_reason", reason)
    return {
        "status": "ending",
        "reason": reason or "user_requested_stop",
    }


def handle_search_live_info(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    # AUDIT-FIX(#7): Validate the payload shape before reading fields from it.
    arguments = _ensure_arguments_mapping(arguments)
    # AUDIT-FIX(#8): Normalize text arguments so missing optional fields stay empty.
    question = _coerce_argument_text(arguments, "question")
    location_hint = _coerce_argument_text(arguments, "location_hint")
    date_context = _coerce_argument_text(arguments, "date_context")
    if not question:
        raise RuntimeError("search_live_info requires `question`")
    date_context = _resolve_search_date_context(
        question,
        date_context,
        timezone_name=getattr(owner.config, "local_timezone_name", "Europe/Berlin"),
    )

    # AUDIT-FIX(#4): Emit sanitized telemetry for user-controlled search inputs.
    _emit_kv_safe(owner, "search_tool_call", "true")
    _emit_kv_safe(owner, "search_question", question)
    # AUDIT-FIX(#5): Search start auditing is non-critical and must not block the actual lookup.
    _record_event_safe(owner, "search_started", "Live web search tool was invoked.", question=question)
    if location_hint:
        # AUDIT-FIX(#4): Sanitize optional location hints before emitting them.
        _emit_kv_safe(owner, "search_location_hint", location_hint)
    if date_context:
        # AUDIT-FIX(#4): Sanitize optional date-context hints before emitting them.
        _emit_kv_safe(owner, "search_date_context", date_context)

    try:
        result = owner.print_backend.search_live_info_with_metadata(
            question,
            conversation=owner.runtime.provider_conversation_context(),
            location_hint=location_hint or None,
            date_context=date_context or None,
        )
        # AUDIT-FIX(#9): A blank answer is a failed search result, not a successful response.
        answer = _require_non_empty_text(
            getattr(result, "answer", None),
            "live search returned empty answer",
        )
        # AUDIT-FIX(#8): Normalize backend source collections before iterating, measuring, or storing them.
        sources = _normalize_sources(getattr(result, "sources", None))
        used_web_search = bool(getattr(result, "used_web_search", False))
    except Exception as exc:
        # AUDIT-FIX(#2): Surface a consistent search failure without leaking raw provider errors into the event stream.
        _emit_tool_failure(
            owner,
            "search",
            "Live search failed.",
            exc,
            question=question,
        )
        raise RuntimeError("live search failed") from exc

    # AUDIT-FIX(#4): Emit sanitized backend metadata returned from the search path.
    _emit_kv_safe(owner, "search_used_web_search", str(used_web_search).lower())
    response_id = _normalize_optional_text(getattr(result, "response_id", None))
    if response_id:
        # AUDIT-FIX(#4): Response identifiers may come from external providers and still need sanitized emission.
        _emit_kv_safe(owner, "search_response_id", response_id)
    request_id = _normalize_optional_text(getattr(result, "request_id", None))
    if request_id:
        # AUDIT-FIX(#4): Request identifiers may come from external providers and still need sanitized emission.
        _emit_kv_safe(owner, "search_request_id", request_id)
    # AUDIT-FIX(#5): Analytics persistence is non-critical and must remain best-effort.
    _record_usage_safe(
        owner,
        request_kind="search",
        source="realtime_tool",
        model=getattr(result, "model", None),
        response_id=response_id or None,
        request_id=request_id or None,
        used_web_search=used_web_search,
        token_usage=getattr(result, "token_usage", None),
        question=question,
    )
    for index, source in enumerate(sources, start=1):
        # AUDIT-FIX(#4): Sanitize each emitted source string before writing it to the event stream.
        _emit_kv_safe(owner, f"search_source_{index}", source)
    # AUDIT-FIX(#5): Completion auditing is best-effort; the answer is already available at this point.
    _record_event_safe(
        owner,
        "search_finished",
        "Live web search completed.",
        sources=len(sources),
        used_web_search=used_web_search,
    )
    # AUDIT-FIX(#5): Remembering search results should not fail the live answer path if persistence is temporarily down.
    _call_best_effort(
        owner,
        lambda: owner.runtime.remember_search_result(
            question=question,
            answer=answer,
            sources=sources,
            location_hint=location_hint or None,
            date_context=date_context or None,
        ),
        event_name="search_memory_store_failed",
        message="Search result could not be persisted.",
        default=None,
        question=question,
    )
    return {
        "status": "ok",
        "answer": answer,
        "sources": sources,
    }


def handle_inspect_camera(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    # AUDIT-FIX(#7): Validate the payload shape before reading fields from it.
    arguments = _ensure_arguments_mapping(arguments)
    # AUDIT-FIX(#8): Normalize the required question field without converting null to `'None'`.
    question = _coerce_argument_text(arguments, "question")
    if not question:
        raise RuntimeError("inspect_camera requires `question`")

    # AUDIT-FIX(#4): Emit sanitized camera-tool telemetry for user-controlled questions.
    _emit_kv_safe(owner, "camera_tool_call", "true")
    _emit_kv_safe(owner, "camera_question", question)
    try:
        # AUDIT-FIX(#6): Normalize captured images and fail fast if the camera path produced nothing usable.
        images = _normalize_images(owner._build_vision_images())
        if not images:
            raise RuntimeError("camera did not provide any images")
        response = owner.print_backend.respond_to_images_with_metadata(
            owner._build_vision_prompt(question, include_reference=len(images) > 1),
            images=images,
            conversation=owner.runtime.provider_conversation_context(),
            allow_web_search=False,
        )
        # AUDIT-FIX(#9): Empty vision text is a failed inspection, not a valid answer.
        answer = _require_non_empty_text(
            getattr(response, "text", None),
            "camera inspection returned empty answer",
        )
    except Exception as exc:
        # AUDIT-FIX(#2): Surface a consistent camera failure without leaking raw provider errors into telemetry.
        _emit_tool_failure(
            owner,
            "camera",
            "Camera inspection failed.",
            exc,
            question=question,
        )
        raise RuntimeError("camera inspection failed") from exc

    # AUDIT-FIX(#4): Emit sanitized response metadata for downstream telemetry consumers.
    _emit_kv_safe(owner, "vision_image_count", len(images))
    response_id = _normalize_optional_text(getattr(response, "response_id", None))
    if response_id:
        # AUDIT-FIX(#4): Response identifiers may come from external providers and still need sanitized emission.
        _emit_kv_safe(owner, "camera_response_id", response_id)
    request_id = _normalize_optional_text(getattr(response, "request_id", None))
    if request_id:
        # AUDIT-FIX(#4): Request identifiers may come from external providers and still need sanitized emission.
        _emit_kv_safe(owner, "camera_request_id", request_id)
    # AUDIT-FIX(#5): Usage persistence is non-critical and must remain best-effort.
    _record_usage_safe(
        owner,
        request_kind="vision",
        source="realtime_tool",
        model=getattr(response, "model", None),
        response_id=response_id or None,
        request_id=request_id or None,
        used_web_search=False,
        token_usage=getattr(response, "token_usage", None),
        question=question,
        vision_image_count=len(images),
    )
    return {
        "status": "ok",
        "answer": answer,
    }
