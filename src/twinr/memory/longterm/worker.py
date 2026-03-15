from __future__ import annotations

from typing import Callable, Generic, TypeVar
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
import time

from twinr.memory.longterm.models import LongTermConversationTurn, LongTermEnqueueResult, LongTermMultimodalEvidence


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
        self._write_callback = write_callback
        self._queue: Queue[TLongTermItem] = Queue(maxsize=max_queue_size)
        self._poll_interval_s = max(poll_interval_s, 0.01)
        self._stop_event = Event()
        self._drain_lock = Lock()
        self._inflight = 0
        self._dropped_count = 0
        self._last_error_message: str | None = None
        self._worker = Thread(
            target=self._run,
            name=worker_name,
            daemon=True,
        )
        self._worker.start()

    @property
    def dropped_count(self) -> int:
        with self._drain_lock:
            return self._dropped_count

    def pending_count(self) -> int:
        with self._drain_lock:
            return self._queue.qsize() + self._inflight

    @property
    def last_error_message(self) -> str | None:
        with self._drain_lock:
            return self._last_error_message

    def enqueue(self, item: TLongTermItem) -> LongTermEnqueueResult:
        try:
            self._queue.put_nowait(item)
            with self._drain_lock:
                self._last_error_message = None
            return LongTermEnqueueResult(
                accepted=True,
                pending_count=self.pending_count(),
                dropped_count=self.dropped_count,
            )
        except Full:
            with self._drain_lock:
                self._dropped_count += 1
                dropped_count = self._dropped_count
            return LongTermEnqueueResult(
                accepted=False,
                pending_count=self.pending_count(),
                dropped_count=dropped_count,
            )

    def flush(self, *, timeout_s: float = 2.0) -> bool:
        deadline = time.monotonic() + max(timeout_s, 0.0)
        while time.monotonic() <= deadline:
            if self.pending_count() <= 0:
                return self.last_error_message is None
            time.sleep(self._poll_interval_s)
        return self.pending_count() <= 0 and self.last_error_message is None

    def shutdown(self, *, timeout_s: float = 2.0) -> None:
        self._stop_event.set()
        self.flush(timeout_s=timeout_s)
        self._worker.join(timeout=max(timeout_s, 0.1))

    def _run(self) -> None:
        while True:
            if self._stop_event.is_set() and self._queue.empty():
                return
            try:
                item = self._queue.get(timeout=self._poll_interval_s)
            except Empty:
                continue
            with self._drain_lock:
                self._inflight += 1
            try:
                self._write_callback(item)
            except Exception as exc:
                with self._drain_lock:
                    self._last_error_message = f"{type(exc).__name__}: {exc}"
            finally:
                with self._drain_lock:
                    self._inflight = max(0, self._inflight - 1)
                self._queue.task_done()


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
