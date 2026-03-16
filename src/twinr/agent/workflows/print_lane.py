"""Run the bounded background print lane used by realtime workflow loops."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock, Thread, current_thread
from typing import Callable

from twinr.agent.base_agent.contracts import CombinedSpeechAgentProvider, ConversationLike
from twinr.hardware.printer import RawReceiptPrinter


@dataclass(frozen=True)
class PrintLaneRequest:
    """Describe one queued background print request."""

    conversation: ConversationLike | None
    focus_hint: str | None
    direct_text: str | None
    request_source: str
    usage_source: str
    printer_queue: str
    multimodal_source: str


class TwinrPrintLane:
    """Serialize print composition, printer delivery, and print telemetry."""

    def __init__(
        self,
        *,
        backend: CombinedSpeechAgentProvider,
        printer: RawReceiptPrinter,
        emit: Callable[[str], None],
        record_event: Callable[..., None],
        record_usage: Callable[..., None],
        start_feedback_loop: Callable[[str], Callable[[], None]],
        format_exception: Callable[[BaseException], str],
        on_print_submitted: Callable[[], None],
        enqueue_multimodal_evidence: Callable[..., None],
    ) -> None:
        self._backend = backend
        self._printer = printer
        self._emit = emit
        self._record_event = record_event
        self._record_usage = record_usage
        self._start_feedback_loop = start_feedback_loop
        self._format_exception = format_exception
        self._on_print_submitted = on_print_submitted
        self._enqueue_multimodal_evidence = enqueue_multimodal_evidence
        self._busy_lock = Lock()
        self._thread_lock = Lock()
        self._thread: Thread | None = None

    def is_busy(self) -> bool:
        return self._busy_lock.locked()

    def submit(self, request: PrintLaneRequest) -> bool:
        if not self._busy_lock.acquire(blocking=False):
            return False
        worker = Thread(
            target=self._run_request,
            args=(request,),
            daemon=True,
            name="twinr-print-lane",
        )
        with self._thread_lock:
            self._thread = worker
        worker.start()
        return True

    def wait_for_idle(self, timeout_s: float = 1.0) -> bool:
        with self._thread_lock:
            worker = self._thread
        if worker is None:
            return True
        worker.join(timeout=max(0.0, float(timeout_s)))
        return not worker.is_alive()

    def _run_request(self, request: PrintLaneRequest) -> None:
        stop_feedback = self._start_feedback_loop("printing")
        self._emit("print_lane=started")
        try:
            composed = self._backend.compose_print_job_with_metadata(
                conversation=request.conversation,
                focus_hint=request.focus_hint,
                direct_text=request.direct_text,
                request_source=request.request_source,
            )
            print_job = self._printer.print_text(composed.text)
            self._on_print_submitted()
            self._emit(f"print_text={composed.text}")
            if composed.response_id:
                self._emit(f"print_response_id={composed.response_id}")
            self._record_usage(
                request_kind="print",
                source=request.usage_source,
                model=composed.model,
                response_id=composed.response_id,
                request_id=composed.request_id,
                used_web_search=False,
                token_usage=composed.token_usage,
                request_source=request.request_source,
            )
            if print_job:
                self._emit(f"print_job={print_job}")
            self._record_event(
                "print_job_sent",
                "Print job was sent to the configured printer.",
                queue=request.printer_queue,
                job=print_job,
            )
            self._record_event(
                "print_completed",
                "Background print lane finished the queued print request.",
                queue=request.printer_queue,
                job=print_job,
            )
            self._enqueue_multimodal_evidence(
                event_name="print_completed",
                modality="printer",
                source=request.multimodal_source,
                message="Printed Twinr output was delivered from the realtime loop.",
                data={
                    "request_source": request.request_source,
                    "queue": request.printer_queue,
                    "job": print_job or "",
                },
            )
            self._emit("print_lane=completed")
        except Exception as exc:
            error_text = self._format_exception(exc)
            self._record_event(
                "print_failed",
                "Print composition or delivery failed.",
                level="error",
                error=error_text,
            )
            self._emit(f"print_error={error_text}")
            self._emit("print_lane=failed")
        finally:
            stop_feedback()
            self._release()

    def _release(self) -> None:
        with self._thread_lock:
            if self._thread is not None and self._thread is current_thread():
                self._thread = None
        if self._busy_lock.locked():
            self._busy_lock.release()
