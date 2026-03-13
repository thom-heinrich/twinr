from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.memory import OnDeviceMemory
from twinr.runtime import TwinrRuntime
from twinr.state_machine import TwinrStatus


class TwinrRuntimeTests(unittest.TestCase):
    def test_happy_path_returns_to_waiting(self) -> None:
        runtime = TwinrRuntime(config=TwinrConfig())

        runtime.press_green_button()
        self.assertEqual(runtime.status, TwinrStatus.LISTENING)

        runtime.submit_transcript("Hello Twinr")
        self.assertEqual(runtime.status, TwinrStatus.PROCESSING)

        response = runtime.complete_agent_turn("Hello back")
        self.assertEqual(response, "Hello back")
        self.assertEqual(runtime.status, TwinrStatus.ANSWERING)

        runtime.finish_speaking()
        self.assertEqual(runtime.status, TwinrStatus.WAITING)
        self.assertEqual(runtime.memory.last_assistant_message(), "Hello back")

    def test_print_requires_previous_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "runtime-state.json"
            runtime = TwinrRuntime(config=TwinrConfig(runtime_state_path=str(snapshot_path)))
            with self.assertRaises(RuntimeError):
                runtime.press_yellow_button()

    def test_memory_compacts_when_limit_is_exceeded(self) -> None:
        memory = OnDeviceMemory(max_turns=4, keep_recent=2)
        memory.remember("user", "one")
        memory.remember("assistant", "two")
        memory.remember("user", "three")
        memory.remember("assistant", "four")
        memory.remember("user", "five")

        self.assertEqual(memory.turns[0].role, "system")
        self.assertIn("Compact conversation summary", memory.turns[0].content)
        self.assertLessEqual(len(memory.turns), 3)

    def test_tool_print_can_resume_answering(self) -> None:
        runtime = TwinrRuntime(config=TwinrConfig())

        runtime.press_green_button()
        runtime.submit_transcript("Print this")
        runtime.begin_tool_print()

        self.assertEqual(runtime.status, TwinrStatus.PRINTING)

        runtime.resume_answering_after_print()
        self.assertEqual(runtime.status, TwinrStatus.ANSWERING)

    def test_runtime_persists_snapshot_for_web_ui(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "runtime-state.json"
            runtime = TwinrRuntime(config=TwinrConfig(runtime_state_path=str(snapshot_path)))

            runtime.press_green_button()
            runtime.submit_transcript("Was steht heute an?")
            runtime.complete_agent_turn("Heute steht der Arzttermin um 14 Uhr an.")
            runtime.finish_speaking()

            self.assertTrue(snapshot_path.exists())
            snapshot_text = snapshot_path.read_text(encoding="utf-8")
            self.assertIn('"status": "waiting"', snapshot_text)
            self.assertIn("Was steht heute an?", snapshot_text)
            self.assertIn("Arzttermin um 14 Uhr", snapshot_text)

    def test_runtime_restores_last_response_from_snapshot_on_startup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "runtime-state.json"

            runtime = TwinrRuntime(config=TwinrConfig(runtime_state_path=str(snapshot_path)))
            runtime.press_green_button()
            runtime.submit_transcript("Bitte merke dir den Zahnarzttermin.")
            runtime.complete_agent_turn("Zahnarzttermin ist am Montag um 14 Uhr.")
            runtime.finish_speaking()

            restarted = TwinrRuntime(
                config=TwinrConfig(
                    runtime_state_path=str(snapshot_path),
                    restore_runtime_state_on_startup=True,
                )
            )

            self.assertEqual(
                restarted.press_yellow_button(),
                "Zahnarzttermin ist am Montag um 14 Uhr.",
            )


if __name__ == "__main__":
    unittest.main()
