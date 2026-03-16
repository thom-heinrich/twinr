from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.memory import ConversationTurn, OnDeviceMemory
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

    def test_background_print_request_does_not_change_runtime_state(self) -> None:
        runtime = TwinrRuntime(config=TwinrConfig())
        runtime.last_response = "Hallo"

        printed = runtime.prepare_background_button_print_request()

        self.assertEqual(printed, "Hallo")
        self.assertEqual(runtime.status, TwinrStatus.WAITING)

    def test_memory_compacts_when_limit_is_exceeded(self) -> None:
        memory = OnDeviceMemory(max_turns=4, keep_recent=2)
        memory.remember("user", "one")
        memory.remember("assistant", "two")
        memory.remember("user", "three")
        memory.remember("assistant", "four")
        memory.remember("user", "five")

        self.assertEqual(tuple(turn.content for turn in memory.raw_tail), ("four", "five"))
        self.assertEqual(memory.ledger[0].kind, "conversation_summary")
        self.assertIn("User asked: one", memory.ledger[0].content)
        self.assertEqual(memory.turns[0].role, "system")
        self.assertIn("Twinr memory summary", memory.turns[0].content)
        self.assertLessEqual(len(memory.turns), 3)

    def test_memory_restores_legacy_assistant_summary_snapshot(self) -> None:
        memory = OnDeviceMemory(max_turns=4, keep_recent=2)
        memory.restore(
            (
                ConversationTurn(
                    role="assistant",
                    content="Twinr memory summary:\n- Earlier context: User asked: eins | Twinr answered: zwei",
                ),
                ConversationTurn(role="user", content="drei"),
                ConversationTurn(role="assistant", content="vier"),
            )
        )

        self.assertEqual(memory.ledger[0].kind, "conversation_summary")
        self.assertIn("Earlier context", memory.ledger[0].content)
        self.assertEqual(tuple(turn.role for turn in memory.raw_tail), ("user", "assistant"))
        self.assertEqual(memory.turns[0].role, "system")
        self.assertIn("Twinr memory summary", memory.turns[0].content)

    def test_memory_stores_search_results_as_typed_entries(self) -> None:
        memory = OnDeviceMemory(max_turns=6, keep_recent=3)
        memory.remember("user", "Wie wird das Wetter morgen?")
        memory.remember_search(
            question="Wie wird das Wetter morgen in Schwarzenbek?",
            answer="Morgen wird es 11 Grad und windig.",
            sources=("https://example.com/weather",),
            location_hint="Schwarzenbek",
            date_context="2026-03-14",
        )

        self.assertEqual(len(memory.search_results), 1)
        self.assertEqual(memory.search_results[0].question, "Wie wird das Wetter morgen in Schwarzenbek?")
        self.assertEqual(memory.search_results[0].sources, ("https://example.com/weather",))
        self.assertEqual(memory.ledger[-1].kind, "search_result")
        self.assertIn("Verified web lookup", memory.turns[0].content)
        self.assertIn("Morgen wird es 11 Grad und windig.", memory.turns[0].content)

    def test_memory_reconfigure_expands_summary_and_preserves_recent_turns(self) -> None:
        memory = OnDeviceMemory(max_turns=6, keep_recent=3)
        for index in range(5):
            memory.remember("user", f"Frage {index}")
            memory.remember("assistant", f"Antwort {index}")
        for index in range(5):
            memory.remember_search(
                question=f"Suche {index}",
                answer=f"Antwort Suche {index}",
                sources=(f"https://example.com/{index}",),
            )

        memory.reconfigure(max_turns=20, keep_recent=10)

        self.assertEqual(memory.max_turns, 20)
        self.assertEqual(memory.keep_recent, 10)
        self.assertEqual(len(memory.raw_tail), 6)
        self.assertEqual(len(memory.search_results), 4)
        self.assertGreaterEqual(memory.turns[0].content.count("Verified web lookup"), 3)

    def test_tool_print_can_resume_answering(self) -> None:
        runtime = TwinrRuntime(config=TwinrConfig())

        runtime.press_green_button()
        runtime.submit_transcript("Print this")
        runtime.begin_tool_print()

        self.assertEqual(runtime.status, TwinrStatus.PRINTING)

        runtime.resume_answering_after_print()
        self.assertEqual(runtime.status, TwinrStatus.ANSWERING)

    def test_proactive_prompt_can_speak_without_user_turn(self) -> None:
        runtime = TwinrRuntime(config=TwinrConfig())

        spoken = runtime.begin_proactive_prompt("Ist alles in Ordnung?")

        self.assertEqual(spoken, "Ist alles in Ordnung?")
        self.assertEqual(runtime.status, TwinrStatus.ANSWERING)
        self.assertEqual(runtime.last_response, None)
        self.assertEqual(runtime.memory.last_assistant_message(), "Ist alles in Ordnung?")

        runtime.finish_speaking()
        self.assertEqual(runtime.status, TwinrStatus.WAITING)

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
            runtime.remember_search_result(
                question="Wann ist mein Zahnarzttermin?",
                answer="Am Montag um 14 Uhr.",
                sources=("https://example.com/termin",),
            )
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
            self.assertEqual(restarted.memory.search_results[0].answer, "Am Montag um 14 Uhr.")
            self.assertIn("Verified web lookup", restarted.memory.turns[0].content)

    def test_runtime_persists_generic_memory_notes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "runtime-state.json"
            runtime = TwinrRuntime(config=TwinrConfig(runtime_state_path=str(snapshot_path)))

            runtime.remember_note(
                kind="preference",
                content="Behavior update (response_style): Keep answers very short and calm.",
                metadata={"category": "response_style"},
            )

            restarted = TwinrRuntime(
                config=TwinrConfig(
                    runtime_state_path=str(snapshot_path),
                    restore_runtime_state_on_startup=True,
                )
            )

        self.assertEqual(restarted.memory.ledger[-1].kind, "preference")
        self.assertIn("Keep answers very short and calm.", restarted.memory.turns[0].content)


if __name__ == "__main__":
    unittest.main()
