from __future__ import annotations

from typing import Callable, Generic, TypeVar
from queue import Empty, Full, Queue
from threading import Condition, Event, Lock, Thread, current_thread
import logging
import math
import time

from twinr.memory.longterm.core.models import LongTermConversationTurn, LongTermEnqueueResult, LongTermMultimodalEvidence


LOGGER = logging.getLogger(__name__)
TLongTermItem = TypeVar("TLongTermItem")


class _AsyncLongTermWriter(Generic[TLongTermItem]):
    def __init__(
        self,
        *,
        write_callback: Callable[[TLongTermItem], None],
        max_queue_size: int = 32,
        poll_interval_s: float = 0.1,
        worker_name: str = "twinr-longterm-memory",
    ) -> None:
        # AUDIT-FIX(#3): Validate config eagerly so bad values cannot create an unbounded
        # queue or non-finite timing behavior on a memory-constrained RPi.
        if not callable(write_callback):
            raise TypeError("write_callback must be callable")
        if isinstance(max_queue_size, bool) or not isinstance(max_queue_size, int):
            raise TypeError("max_queue_size must be an int")
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be > 0")
        if isinstance(poll_interval_s, bool) or not isinstance(poll_interval_s, (int, float)):
            raise TypeError("poll_interval_s must be a real number")

        normalized_poll_interval_s = float(poll_interval_s)
        if not math.isfinite(normalized_poll_interval_s) or normalized_poll_interval_s <= 0.0:
            raise ValueError("poll_interval_s must be finite and > 0")

        self._write_callback = write_callback
        self._queue: Queue[TLongTermItem] = Queue(maxsize=max_queue_size)
        self._poll_interval_s = max(normalized_poll_interval_s, 0.01)
        self._stop_event = Event()
        self._drain_lock = Lock()
        # AUDIT-FIX(#4,#6): Use a condition with exact counters instead of qsize()/empty()
        # plus sleep-based polling for correctness and low idle CPU load.
        self._drain_condition = Condition(self._drain_lock)
        self._pending = 0
        self._inflight = 0
        self._dropped_count = 0
        self._last_error_message: str | None = None
        # AUDIT-FIX(#2,#9): Stop accepting new items once shutdown starts or the worker exits,
        # so late enqueues cannot be accepted after the consumer is gone.
        self._accepting = True
        self._worker_exited = False
        self._worker = Thread(
            target=self._run,
            name=worker_name,
            daemon=True,
        )
        self._worker.start()

    @staticmethod
    def _normalize_timeout(timeout_s: float) -> float:
        # AUDIT-FIX(#3): Reject NaN/inf timeouts instead of letting them corrupt flush/shutdown behavior.
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise TypeError("timeout_s must be a real number")
        normalized_timeout_s = float(timeout_s)
        if not math.isfinite(normalized_timeout_s):
            raise ValueError("timeout_s must be finite")
        return max(normalized_timeout_s, 0.0)

    @property
    def dropped_count(self) -> int:
        with self._drain_lock:
            return self._dropped_count

    def pending_count(self) -> int:
        with self._drain_lock:
            # AUDIT-FIX(#4): Return an exact pending count instead of Queue.qsize().
            return self._pending

    @property
    def last_error_message(self) -> str | None:
        with self._drain_lock:
            return self._last_error_message

    def enqueue(self, item: TLongTermItem) -> LongTermEnqueueResult:
        accepted = False
        with self._drain_condition:
            if not self._accepting:
                # AUDIT-FIX(#2): Reject writes once shutdown/drain has started; accepting them
                # here would silently lose data after the worker exits.
                self._dropped_count += 1
            elif self._worker_exited or not self._worker.is_alive():
                # AUDIT-FIX(#9): Refuse enqueues if the worker is already dead and surface degraded state.
                self._dropped_count += 1
                if self._last_error_message is None:
                    self._last_error_message = "background writer is not running"
            else:
                try:
                    self._queue.put_nowait(item)
                except Full:
                    self._dropped_count += 1
                else:
                    # AUDIT-FIX(#1,#4): Queue admission must not clear prior permanent errors; it
                    # only increments the exact pending counter for later drain tracking.
                    accepted = True
                    self._pending += 1
                    self._drain_condition.notify_all()

            pending_count = self._pending
            dropped_count = self._dropped_count

        return LongTermEnqueueResult(
            accepted=accepted,
            pending_count=pending_count,
            dropped_count=dropped_count,
        )

    def flush(self, *, timeout_s: float = 2.0) -> bool:
        timeout_s = self._normalize_timeout(timeout_s)
        if current_thread() is self._worker:
            # AUDIT-FIX(#7): A worker thread cannot wait for its own in-flight item to finish.
            LOGGER.error("%s flush() called from its own worker thread; refusing to self-wait", self._worker.name)
            return False

        with self._drain_condition:
            # AUDIT-FIX(#4,#6): Wait on exact state changes instead of qsize()/empty() plus busy sleep.
            drained = self._drain_condition.wait_for(
                lambda: self._pending == 0 or self._worker_exited or not self._worker.is_alive(),
                timeout=timeout_s,
            )
            if (
                self._pending > 0
                and (self._worker_exited or not self._worker.is_alive())
                and self._last_error_message is None
            ):
                self._last_error_message = "background writer exited before draining all pending items"
            return drained and self._pending == 0 and self._last_error_message is None

    def shutdown(self, *, timeout_s: float = 2.0) -> None:
        timeout_s = self._normalize_timeout(timeout_s)
        shutdown_started_at = time.monotonic()
        with self._drain_condition:
            # AUDIT-FIX(#2): Make shutdown idempotent and close admission before the drain starts.
            self._accepting = False
            self._stop_event.set()
            self._drain_condition.notify_all()

        if current_thread() is self._worker:
            # AUDIT-FIX(#7): Joining the current thread would raise and deadlock the in-flight write.
            LOGGER.error("%s shutdown() called from its own worker thread; stop requested without join", self._worker.name)
            return

        self.flush(timeout_s=timeout_s)
        join_timeout_s = max(timeout_s, 0.1)
        self._worker.join(timeout=join_timeout_s)
        if self._worker.is_alive():
            # AUDIT-FIX(#8): Expose stalled shutdown so the service can fail closed instead of
            # pretending persistence drained cleanly.
            with self._drain_condition:
                if self._last_error_message is None:
                    self._last_error_message = "background writer did not stop within shutdown timeout"
                self._drain_condition.notify_all()
            elapsed_s = time.monotonic() - shutdown_started_at
            LOGGER.error(
                "%s did not stop within %.2fs (elapsed %.2fs)",
                self._worker.name,
                join_timeout_s,
                elapsed_s,
            )

    def _run(self) -> None:
        try:
            while True:
                with self._drain_condition:
                    # AUDIT-FIX(#4): Exit only when shutdown is requested and the exact pending count is zero.
                    if self._stop_event.is_set() and self._pending == 0:
                        return
                try:
                    item = self._queue.get(timeout=self._poll_interval_s)
                except Empty:
                    continue
                with self._drain_condition:
                    self._inflight += 1
                try:
                    self._write_callback(item)
                    with self._drain_condition:
                        self._last_error_message = None
                except Exception as exc:
                    # AUDIT-FIX(#5): Count permanently failed writes as dropped data, keep the
                    # error latched, and emit a structured log without leaking the item payload.
                    with self._drain_condition:
                        self._dropped_count += 1
                        self._last_error_message = f"{type(exc).__name__}: {exc}"
                    LOGGER.exception("%s failed to persist a long-term item", self._worker.name)
                except BaseException as exc:
                    # AUDIT-FIX(#5): A BaseException from the callback also means the accepted item
                    # was not safely persisted, so count it as dropped before the worker unwinds.
                    with self._drain_condition:
                        self._dropped_count += 1
                        self._last_error_message = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    with self._drain_condition:
                        self._inflight = max(0, self._inflight - 1)
                        self._pending = max(0, self._pending - 1)
                        self._drain_condition.notify_all()
                    self._queue.task_done()
        except BaseException as exc:
            # AUDIT-FIX(#9): Unexpected worker crashes must latch a terminal error and wake waiters immediately.
            with self._drain_condition:
                if self._last_error_message is None:
                    self._last_error_message = f"Worker crashed with {type(exc).__name__}: {exc}"
                self._drain_condition.notify_all()
            LOGGER.exception("%s crashed unexpectedly", self._worker.name)
            raise
        finally:
            with self._drain_condition:
                self._accepting = False
                self._worker_exited = True
                self._drain_condition.notify_all()


class AsyncLongTermMemoryWriter(_AsyncLongTermWriter[LongTermConversationTurn]):
    def __init__(
        self,
        *,
        write_callback: Callable[[LongTermConversationTurn], None],
        max_queue_size: int = 32,
        poll_interval_s: float = 0.1,
    ) -> None:
        super().__init__(
            write_callback=write_callback,
            max_queue_size=max_queue_size,
            poll_interval_s=poll_interval_s,
            worker_name="twinr-longterm-memory",
        )


class AsyncLongTermMultimodalWriter(_AsyncLongTermWriter[LongTermMultimodalEvidence]):
    def __init__(
        self,
        *,
        write_callback: Callable[[LongTermMultimodalEvidence], None],
        max_queue_size: int = 32,
        poll_interval_s: float = 0.1,
    ) -> None:
        super().__init__(
            write_callback=write_callback,
            max_queue_size=max_queue_size,
            poll_interval_s=poll_interval_s,
            worker_name="twinr-longterm-multimodal",
        )
