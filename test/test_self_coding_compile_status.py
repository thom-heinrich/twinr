from datetime import UTC, datetime
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.self_coding import CompileRunStatusRecord, SelfCodingStore


class SelfCodingCompileStatusStoreTests(unittest.TestCase):
    def test_save_and_load_compile_status(self) -> None:
        status = CompileRunStatusRecord(
            job_id="job_status123",
            phase="streaming",
            driver_name="CodexSdkDriver",
            driver_attempts=("CodexSdkDriver",),
            event_count=3,
            last_event_kind="assistant_delta",
            last_event_message="drafting",
            thread_id="thread-123",
            turn_id="turn-123",
            final_message_seen=False,
            turn_completed=False,
            started_at=datetime(2026, 3, 16, 16, 10, tzinfo=UTC),
            updated_at=datetime(2026, 3, 16, 16, 11, tzinfo=UTC),
            diagnostics={"timeout_seconds": 90},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            store = SelfCodingStore.from_project_root(temp_dir)
            store.save_compile_status(status)

            loaded = store.load_compile_status("job_status123")

        self.assertEqual(loaded.driver_name, "CodexSdkDriver")
        self.assertEqual(loaded.driver_attempts, ("CodexSdkDriver",))
        self.assertEqual(loaded.event_count, 3)
        self.assertEqual(loaded.thread_id, "thread-123")
        self.assertEqual(loaded.diagnostics["timeout_seconds"], 90)


if __name__ == "__main__":
    unittest.main()
