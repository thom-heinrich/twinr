from __future__ import annotations

from pathlib import Path
import sys
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.longterm.core.models import LongTermConversationTurn
from twinr.memory.longterm.runtime.worker import AsyncLongTermMemoryWriter


class LongTermWorkerTests(unittest.TestCase):
    def test_flush_returns_false_when_callback_fails(self) -> None:
        started = threading.Event()

        def fail_write(item: LongTermConversationTurn) -> None:
            del item
            started.set()
            raise RuntimeError("boom")

        writer = AsyncLongTermMemoryWriter(write_callback=fail_write, poll_interval_s=0.01)
        try:
            result = writer.enqueue(
                LongTermConversationTurn(
                    transcript="Hello",
                    response="Hi",
                )
            )
            self.assertTrue(result.accepted)
            self.assertTrue(started.wait(timeout=1.0))
            self.assertFalse(writer.flush(timeout_s=1.0))
            self.assertEqual(writer.last_error_message, "RuntimeError: boom")
        finally:
            writer.shutdown(timeout_s=1.0)

    def test_new_batch_clears_previous_error_after_successful_write(self) -> None:
        call_count = {"count": 0}

        def flaky_write(item: LongTermConversationTurn) -> None:
            del item
            call_count["count"] += 1
            if call_count["count"] == 1:
                raise RuntimeError("boom")

        writer = AsyncLongTermMemoryWriter(write_callback=flaky_write, poll_interval_s=0.01)
        try:
            writer.enqueue(
                LongTermConversationTurn(
                    transcript="First",
                    response="First reply",
                )
            )
            self.assertFalse(writer.flush(timeout_s=1.0))
            self.assertEqual(writer.last_error_message, "RuntimeError: boom")

            writer.enqueue(
                LongTermConversationTurn(
                    transcript="Second",
                    response="Second reply",
                )
            )
            deadline = time.monotonic() + 1.0
            while time.monotonic() <= deadline and writer.pending_count() > 0:
                time.sleep(0.01)

            self.assertTrue(writer.flush(timeout_s=1.0))
            self.assertIsNone(writer.last_error_message)
        finally:
            writer.shutdown(timeout_s=1.0)


if __name__ == "__main__":
    unittest.main()
