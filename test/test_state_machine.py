from pathlib import Path
import sys
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
        runtime = TwinrRuntime(config=TwinrConfig())
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


if __name__ == "__main__":
    unittest.main()
