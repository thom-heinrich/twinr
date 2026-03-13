from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.display.service import TwinrStatusDisplayLoop
from twinr.agent.base_agent.runtime_state import RuntimeSnapshot, RuntimeSnapshotStore


class FakeDisplay:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, tuple[str, ...], int]] = []

    def show_status(
        self,
        status: str,
        *,
        headline: str | None = None,
        details: tuple[str, ...] = (),
        animation_frame: int = 0,
    ) -> None:
        self.calls.append((status, headline, details, animation_frame))


class DisplayServiceTests(unittest.TestCase):
    def test_display_loop_renders_when_snapshot_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "runtime-state.json"
            store = RuntimeSnapshotStore(snapshot_path)
            store.save(
                status="waiting",
                memory_turns=(),
                last_transcript=None,
                last_response="Hallo Thom",
            )
            display = FakeDisplay()
            loop = TwinrStatusDisplayLoop(
                config=TwinrConfig(runtime_state_path=str(snapshot_path), display_poll_interval_s=0.0),
                display=display,
                snapshot_store=store,
                emit=lambda _line: None,
                sleep=lambda _seconds: None,
            )

            loop.run(max_cycles=1)
            store.save(
                status="printing",
                memory_turns=(),
                last_transcript=None,
                last_response="Wichtige Info",
            )
            loop.run(max_cycles=1)

            self.assertEqual(
                [(status, headline, details) for status, headline, details, _frame in display.calls],
                [
                    ("waiting", "Waiting", ()),
                    ("printing", "Printing", ()),
                ],
            )

    def test_build_status_content_uses_error_message(self) -> None:
        loop = TwinrStatusDisplayLoop(
            config=TwinrConfig(display_poll_interval_s=0.0),
            display=FakeDisplay(),
            snapshot_store=RuntimeSnapshotStore("/tmp/nonexistent"),
            emit=lambda _line: None,
            sleep=lambda _seconds: None,
        )

        headline, details = loop._build_status_content(
            RuntimeSnapshot(status="error", error_message="Printer offline")
        )

        self.assertEqual(headline, "Error")
        self.assertEqual(details, ())

    def test_waiting_animation_frame_changes_over_time(self) -> None:
        loop = TwinrStatusDisplayLoop(
            config=TwinrConfig(display_poll_interval_s=0.0),
            display=FakeDisplay(),
            snapshot_store=RuntimeSnapshotStore("/tmp/nonexistent"),
            emit=lambda _line: None,
            sleep=lambda _seconds: None,
        )

        original_monotonic = time.monotonic
        try:
            time.monotonic = lambda: 0.0
            first = loop._animation_frame("waiting")
            time.monotonic = lambda: 24.5
            second = loop._animation_frame("waiting")
        finally:
            time.monotonic = original_monotonic

        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
