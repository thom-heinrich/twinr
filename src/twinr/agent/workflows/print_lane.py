"""Run the bounded background print lane used by realtime workflow loops."""

from __future__ import annotations

# CHANGELOG: 2026-03-28
# BUG-1: Fixed permanent lane wedging if worker startup fails after submit().
# BUG-2: Fixed permanent lane wedging if start_feedback_loop() or stop_feedback() raises.
# BUG-3: Fixed false print_failed outcomes caused by non-critical telemetry callback failures.
# SEC-1: Removed raw print/error payload emission by default; logs now emit bounded metadata and optional previews only.
# SEC-2: Sanitized ESC/POS control characters out of printer payloads to prevent printer-command injection.
# SEC-3: Bounded print payload size/line count to prevent paper/buffer denial-of-service on Pi deployments.
# IMP-1: Replaced per-request daemon thread spawning with a persistent bounded worker lane.
# IMP-2: Added explicit backpressure, idle waiting across queued work, timing telemetry, and graceful shutdown.
# BREAKING: submit() now accepts one queued request by default (configurable via max_pending_requests).
# BREAKING: raw print content is no longer emitted by default; set emit_print_preview/content to restore visibility.

from dataclasses import dataclass
import hashlib
from queue import Full, Queue
from threading import Condition, Lock, Thread, current_thread
from time import monotonic
from typing import Callable, Final

from twinr.agent.base_agent.contracts import CombinedSpeechAgentProvider, ConversationLike
from twinr.hardware.printer import RawReceiptPrinter


_STOP_WORKER: Final[object] = object()
_DEFAULT_MAX_PENDING_REQUESTS: Final[int] = 1
_DEFAULT_MAX_PRINT_CHARS: Final[int] = 4000
_DEFAULT_MAX_PRINT_LINES: Final[int] = 200
_DEFAULT_MAX_LOG_PREVIEW_CHARS: Final[int] = 160


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


@dataclass(frozen=True, slots=True)
class _SanitizedPrintPayload:
    text: str
    preview: str
    sha256: str
    char_count: int
    line_count: int
    removed_control_chars: int
    truncated: bool


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
        max_pending_requests: int = _DEFAULT_MAX_PENDING_REQUESTS,
        max_print_chars: int = _DEFAULT_MAX_PRINT_CHARS,
        max_print_lines: int = _DEFAULT_MAX_PRINT_LINES,
        max_log_preview_chars: int = _DEFAULT_MAX_LOG_PREVIEW_CHARS,
        emit_print_preview: bool = False,
        emit_print_content: bool = False,
    ) -> None:
        if max_pending_requests < 0:
            raise ValueError("max_pending_requests must be >= 0")
        if max_print_chars <= 0:
            raise ValueError("max_print_chars must be > 0")
        if max_print_lines <= 0:
            raise ValueError("max_print_lines must be > 0")
        if max_log_preview_chars <= 0:
            raise ValueError("max_log_preview_chars must be > 0")
        if emit_print_content and not emit_print_preview:
            emit_print_preview = True

        self._backend = backend
        self._printer = printer
        self._emit = emit
        self._record_event = record_event
        self._record_usage = record_usage
        self._start_feedback_loop = start_feedback_loop
        self._format_exception = format_exception
        self._on_print_submitted = on_print_submitted
        self._enqueue_multimodal_evidence = enqueue_multimodal_evidence

        self._max_pending_requests = int(max_pending_requests)
        self._max_print_chars = int(max_print_chars)
        self._max_print_lines = int(max_print_lines)
        self._max_log_preview_chars = int(max_log_preview_chars)
        self._emit_print_preview = bool(emit_print_preview)
        self._emit_print_content = bool(emit_print_content)

        self._capacity = 1 + self._max_pending_requests
        # Reserve one extra queue slot for the stop sentinel so graceful shutdown
        # does not deadlock behind a full request queue.
        self._request_queue: Queue[PrintLaneRequest | object] = Queue(maxsize=self._capacity + 1)

        self._state_lock = Lock()
        self._idle_condition = Condition(self._state_lock)
        self._worker_lock = Lock()

        self._queued_count = 0
        self._active_count = 0
        self._shutdown = False
        self._stop_enqueued = False
        self._worker: Thread | None = None

    def is_busy(self) -> bool:
        with self._idle_condition:
            return (self._queued_count + self._active_count) > 0

    def submit(self, request: PrintLaneRequest) -> bool:
        if not self._ensure_worker_started():
            self._safe_emit("print_lane=worker_unavailable")
            self._safe_call(
                self._record_event,
                "print_lane_start_failed",
                "Print worker could not be started.",
                level="error",
                queue=request.printer_queue,
                request_source=request.request_source,
            )
            return False

        reason: str | None = None
        with self._idle_condition:
            if self._shutdown:
                reason = "shutdown"
            else:
                outstanding = self._queued_count + self._active_count
                if outstanding >= self._capacity:
                    reason = "busy"
                else:
                    self._queued_count += 1
                    self._idle_condition.notify_all()

        if reason is not None:
            self._safe_emit(f"print_lane={reason}")
            return False

        try:
            self._request_queue.put_nowait(request)
        except Full:
            with self._idle_condition:
                self._queued_count = max(0, self._queued_count - 1)
                self._idle_condition.notify_all()
            self._safe_emit("print_lane=busy")
            return False

        self._safe_emit("print_lane=queued")
        return True

    def wait_for_idle(self, timeout_s: float = 1.0) -> bool:
        timeout_s = max(0.0, float(timeout_s))
        deadline = monotonic() + timeout_s
        with self._idle_condition:
            while (self._queued_count + self._active_count) > 0:
                remaining = deadline - monotonic()
                if remaining <= 0.0:
                    return False
                self._idle_condition.wait(timeout=remaining)
            return True

    def shutdown(self, *, wait: bool = True, timeout_s: float | None = None) -> bool:
        with self._idle_condition:
            self._shutdown = True
            enqueue_stop = not self._stop_enqueued
            if enqueue_stop:
                self._stop_enqueued = True

        if enqueue_stop:
            try:
                self._request_queue.put_nowait(_STOP_WORKER)
            except Full:
                # This should not happen because the queue reserves one extra slot for the sentinel,
                # but stay defensive to avoid a shutdown-time crash.
                self._safe_call(
                    self._record_event,
                    "print_lane_shutdown_queue_full",
                    "Print lane shutdown could not enqueue stop sentinel immediately.",
                    level="warning",
                )

        if not wait:
            return True

        deadline = None if timeout_s is None else monotonic() + max(0.0, float(timeout_s))
        if deadline is None:
            with self._idle_condition:
                while (self._queued_count + self._active_count) > 0:
                    self._idle_condition.wait()
        else:
            remaining = deadline - monotonic()
            if remaining <= 0.0:
                return False
            if not self.wait_for_idle(timeout_s=remaining):
                return False

        worker = self._get_worker()
        if worker is None:
            return True

        join_timeout = None if deadline is None else max(0.0, deadline - monotonic())
        worker.join(timeout=join_timeout)
        return not worker.is_alive()

    def _ensure_worker_started(self) -> bool:
        error_text: str | None = None
        with self._worker_lock:
            worker = self._worker
            if worker is not None and worker.is_alive():
                return True
            if self._shutdown:
                return False
            worker = Thread(
                target=self._worker_loop,
                daemon=True,
                name="twinr-print-lane",
            )
            try:
                worker.start()
            except Exception as exc:
                error_text = self._safe_format_exception(exc)
            else:
                self._worker = worker
                return True

        if error_text is not None:
            self._safe_emit(f"print_worker_error={self._summarize_for_log(error_text)}")
        return False

    def _get_worker(self) -> Thread | None:
        with self._worker_lock:
            return self._worker

    def _worker_loop(self) -> None:
        try:
            while True:
                item = self._request_queue.get()
                if item is _STOP_WORKER:
                    self._request_queue.task_done()
                    break

                with self._idle_condition:
                    self._queued_count = max(0, self._queued_count - 1)
                    self._active_count += 1
                    self._idle_condition.notify_all()

                try:
                    self._run_request(item)  # type: ignore[arg-type]
                finally:
                    with self._idle_condition:
                        self._active_count = max(0, self._active_count - 1)
                        self._idle_condition.notify_all()
                    self._request_queue.task_done()
        finally:
            with self._worker_lock:
                if self._worker is not None and self._worker is current_thread():
                    self._worker = None

    def _run_request(self, request: PrintLaneRequest) -> None:
        stop_feedback = self._safe_start_feedback_loop("printing")
        started_at = monotonic()
        self._safe_emit("print_lane=started")

        try:
            compose_started_at = monotonic()
            composed = self._backend.compose_print_job_with_metadata(
                conversation=request.conversation,
                focus_hint=request.focus_hint,
                direct_text=request.direct_text,
                request_source=request.request_source,
            )
            compose_elapsed_ms = int((monotonic() - compose_started_at) * 1000)

            payload = self._sanitize_print_payload(str(composed.text))
            request_id = getattr(composed, "request_id", None)
            response_id = getattr(composed, "response_id", None)
            model = getattr(composed, "model", None)
            token_usage = getattr(composed, "token_usage", None)

            if self._emit_print_content:
                self._safe_emit(f"print_text={self._escape_for_log(payload.text)}")
            elif self._emit_print_preview:
                self._safe_emit(f"print_preview={payload.preview}")

            self._safe_emit(f"print_text_sha256={payload.sha256}")
            self._safe_emit(f"print_text_chars={payload.char_count}")
            self._safe_emit(f"print_text_lines={payload.line_count}")
            self._safe_emit(f"print_text_truncated={str(payload.truncated).lower()}")
            self._safe_emit(f"print_text_control_chars_removed={payload.removed_control_chars}")

            print_started_at = monotonic()
            print_job = self._printer.print_text(payload.text)
            print_elapsed_ms = int((monotonic() - print_started_at) * 1000)
            total_elapsed_ms = int((monotonic() - started_at) * 1000)

            self._safe_call(self._on_print_submitted)

            if response_id:
                self._safe_emit(f"print_response_id={self._summarize_for_log(str(response_id))}")
            if request_id:
                self._safe_emit(f"print_request_id={self._summarize_for_log(str(request_id))}")
            if print_job:
                self._safe_emit(f"print_job={self._summarize_for_log(str(print_job))}")

            self._safe_emit(f"print_compose_ms={compose_elapsed_ms}")
            self._safe_emit(f"print_io_ms={print_elapsed_ms}")
            self._safe_emit(f"print_total_ms={total_elapsed_ms}")

            self._safe_call(
                self._record_usage,
                request_kind="print",
                source=request.usage_source,
                model=model,
                response_id=response_id,
                request_id=request_id,
                used_web_search=False,
                token_usage=token_usage,
                request_source=request.request_source,
            )
            self._safe_call(
                self._record_event,
                "print_job_sent",
                "Print job was sent to the configured printer.",
                queue=request.printer_queue,
                job=print_job,
                request_source=request.request_source,
                request_id=request_id,
                response_id=response_id,
                compose_ms=compose_elapsed_ms,
                print_ms=print_elapsed_ms,
                total_ms=total_elapsed_ms,
                chars=payload.char_count,
                lines=payload.line_count,
                truncated=payload.truncated,
                removed_control_chars=payload.removed_control_chars,
            )
            self._safe_call(
                self._record_event,
                "print_completed",
                "Background print lane finished the queued print request.",
                queue=request.printer_queue,
                job=print_job,
                request_source=request.request_source,
                request_id=request_id,
                response_id=response_id,
                compose_ms=compose_elapsed_ms,
                print_ms=print_elapsed_ms,
                total_ms=total_elapsed_ms,
                chars=payload.char_count,
                lines=payload.line_count,
                truncated=payload.truncated,
                removed_control_chars=payload.removed_control_chars,
            )
            self._safe_call(
                self._enqueue_multimodal_evidence,
                event_name="print_completed",
                modality="printer",
                source=request.multimodal_source,
                message="Printed Twinr output was delivered from the realtime loop.",
                data={
                    "request_source": request.request_source,
                    "queue": request.printer_queue,
                    "job": print_job or "",
                    "request_id": request_id or "",
                    "response_id": response_id or "",
                    "text_sha256": payload.sha256,
                    "chars": payload.char_count,
                    "lines": payload.line_count,
                    "truncated": payload.truncated,
                },
            )
            self._safe_emit("print_lane=completed")
        except Exception as exc:
            error_text = self._safe_format_exception(exc)
            self._safe_call(
                self._record_event,
                "print_failed",
                "Print composition or delivery failed.",
                level="error",
                queue=request.printer_queue,
                request_source=request.request_source,
                error=error_text,
            )
            self._safe_emit(f"print_error={self._summarize_for_log(error_text)}")
            self._safe_emit("print_lane=failed")
        finally:
            self._safe_call(stop_feedback)

    def _safe_start_feedback_loop(self, state: str) -> Callable[[], None]:
        try:
            stop_feedback = self._start_feedback_loop(state)
        except Exception as exc:
            self._safe_emit(f"print_feedback_error={self._summarize_for_log(self._safe_format_exception(exc))}")
            return self._noop
        return stop_feedback if callable(stop_feedback) else self._noop

    def _safe_format_exception(self, exc: BaseException) -> str:
        try:
            formatted = self._format_exception(exc)
        except Exception:
            formatted = f"{exc.__class__.__name__}: {exc}"
        return str(formatted)

    def _safe_call(self, func: Callable[..., None], /, *args, **kwargs) -> None:
        try:
            func(*args, **kwargs)
        except Exception:
            pass

    def _safe_emit(self, message: str) -> None:
        self._safe_call(self._emit, message)

    def _sanitize_print_payload(self, text: str) -> _SanitizedPrintPayload:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned_parts: list[str] = []
        removed_control_chars = 0
        line_count = 1
        truncated = False

        for ch in normalized:
            if ch == "\n":
                if line_count >= self._max_print_lines:
                    truncated = True
                    break
                cleaned_parts.append("\n")
                line_count += 1
                continue
            if ch == "\t":
                cleaned_parts.append("    ")
                continue
            if ch.isprintable():
                cleaned_parts.append(ch)
                continue
            removed_control_chars += 1

        cleaned = "".join(cleaned_parts)
        if len(cleaned) > self._max_print_chars:
            cleaned = cleaned[: self._max_print_chars]
            truncated = True

        if truncated:
            suffix = "\n[TRUNCATED]"
            budget = max(0, self._max_print_chars - len(suffix))
            cleaned = f"{cleaned[:budget].rstrip()}{suffix}"

        cleaned = cleaned.rstrip("\n")
        if not cleaned:
            raise ValueError("Print payload is empty after sanitization")

        line_count = cleaned.count("\n") + 1
        digest = hashlib.sha256(cleaned.encode("utf-8", "replace")).hexdigest()
        preview = self._summarize_for_log(cleaned)

        return _SanitizedPrintPayload(
            text=cleaned,
            preview=preview,
            sha256=digest,
            char_count=len(cleaned),
            line_count=line_count,
            removed_control_chars=removed_control_chars,
            truncated=truncated,
        )

    def _summarize_for_log(self, text: str) -> str:
        escaped = self._escape_for_log(text)
        if len(escaped) > self._max_log_preview_chars:
            return f"{escaped[: self._max_log_preview_chars - 3]}..."
        return escaped

    def _escape_for_log(self, text: str) -> str:
        escaped = text.replace("\\", "\\\\")
        escaped = escaped.replace("\n", "\\n")
        escaped = escaped.replace("\t", "\\t")
        return escaped

    @staticmethod
    def _noop() -> None:
        return None
