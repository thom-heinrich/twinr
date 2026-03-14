from __future__ import annotations

import time
from typing import Any


def handle_print_receipt(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    focus_hint = str(arguments.get("focus_hint", "")).strip()
    direct_text = str(arguments.get("text", "")).strip()
    if not focus_hint and not direct_text:
        raise RuntimeError("print_receipt requires `focus_hint` or `text`")

    owner.runtime.maybe_begin_tool_print()
    owner._emit_status(force=True)
    stop_printing_feedback = owner._start_working_feedback_loop("printing")
    try:
        composed = owner.print_backend.compose_print_job_with_metadata(
            conversation=owner.runtime.provider_conversation_context(),
            focus_hint=focus_hint or None,
            direct_text=direct_text or None,
            request_source="tool",
        )
        print_job = owner.printer.print_text(composed.text)
    finally:
        stop_printing_feedback()
    owner.emit("print_tool_call=true")
    owner.emit(f"print_text={composed.text}")
    owner._record_usage(
        request_kind="print",
        source="realtime_tool",
        model=composed.model,
        response_id=composed.response_id,
        request_id=composed.request_id,
        used_web_search=False,
        token_usage=composed.token_usage,
        request_source="tool",
    )
    if print_job:
        owner.emit(f"print_job={print_job}")
    owner.runtime.long_term_memory.enqueue_multimodal_evidence(
        event_name="print_completed",
        modality="printer",
        source="tool_print",
        message="Printed Twinr output was delivered from a tool call.",
        data={
            "request_source": "tool",
            "job": print_job or "",
            "focus_hint": focus_hint or "",
        },
    )
    owner._last_print_request_at = time.monotonic()
    return {
        "status": "printed",
        "text": composed.text,
        "job": print_job,
    }


def handle_end_conversation(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    reason = str(arguments.get("reason", "")).strip()
    if reason:
        owner.emit(f"end_conversation_reason={reason}")
    return {
        "status": "ending",
        "reason": reason or "user_requested_stop",
    }


def handle_search_live_info(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    question = str(arguments.get("question", "")).strip()
    location_hint = str(arguments.get("location_hint", "")).strip()
    date_context = str(arguments.get("date_context", "")).strip()
    if not question:
        raise RuntimeError("search_live_info requires `question`")

    owner.emit("search_tool_call=true")
    owner.emit(f"search_question={question}")
    owner._record_event("search_started", "Live web search tool was invoked.", question=question)
    if location_hint:
        owner.emit(f"search_location_hint={location_hint}")
    if date_context:
        owner.emit(f"search_date_context={date_context}")

    result = owner.print_backend.search_live_info_with_metadata(
        question,
        conversation=owner.runtime.provider_conversation_context(),
        location_hint=location_hint or None,
        date_context=date_context or None,
    )

    owner.emit(f"search_used_web_search={str(result.used_web_search).lower()}")
    if result.response_id:
        owner.emit(f"search_response_id={result.response_id}")
    if result.request_id:
        owner.emit(f"search_request_id={result.request_id}")
    owner._record_usage(
        request_kind="search",
        source="realtime_tool",
        model=result.model,
        response_id=result.response_id,
        request_id=result.request_id,
        used_web_search=result.used_web_search,
        token_usage=result.token_usage,
        question=question,
    )
    for index, source in enumerate(result.sources, start=1):
        owner.emit(f"search_source_{index}={source}")
    owner._record_event(
        "search_finished",
        "Live web search completed.",
        sources=len(result.sources),
        used_web_search=result.used_web_search,
    )
    owner.runtime.remember_search_result(
        question=question,
        answer=result.answer,
        sources=result.sources,
        location_hint=location_hint or None,
        date_context=date_context or None,
    )
    return {
        "status": "ok",
        "answer": result.answer,
        "sources": list(result.sources),
    }


def handle_inspect_camera(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    question = str(arguments.get("question", "")).strip()
    if not question:
        raise RuntimeError("inspect_camera requires `question`")

    owner.emit("camera_tool_call=true")
    owner.emit(f"camera_question={question}")
    images = owner._build_vision_images()
    response = owner.print_backend.respond_to_images_with_metadata(
        owner._build_vision_prompt(question, include_reference=len(images) > 1),
        images=images,
        conversation=owner.runtime.provider_conversation_context(),
        allow_web_search=False,
    )
    owner.emit(f"vision_image_count={len(images)}")
    if response.response_id:
        owner.emit(f"camera_response_id={response.response_id}")
    if response.request_id:
        owner.emit(f"camera_request_id={response.request_id}")
    owner._record_usage(
        request_kind="vision",
        source="realtime_tool",
        model=response.model,
        response_id=response.response_id,
        request_id=response.request_id,
        used_web_search=False,
        token_usage=response.token_usage,
        question=question,
        vision_image_count=len(images),
    )
    return {
        "status": "ok",
        "answer": response.text,
    }
