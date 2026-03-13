from pathlib import Path
from types import SimpleNamespace
import base64
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.providers.openai_realtime import OpenAIRealtimeSession


def _fake_usage(
    *,
    input_tokens: int = 80,
    output_tokens: int = 24,
    total_tokens: int = 104,
):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_tokens_details=SimpleNamespace(cached_tokens=8),
        output_tokens_details=SimpleNamespace(reasoning_tokens=5),
    )


class FakeSessionResource:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def update(self, *, session) -> None:
        self.calls.append(session)


class FakeInputAudioBufferResource:
    def __init__(self) -> None:
        self.append_calls: list[str] = []
        self.commit_calls = 0

    async def append(self, *, audio: str) -> None:
        self.append_calls.append(audio)

    async def commit(self) -> None:
        self.commit_calls += 1


class FakeResponseResource:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, *, response) -> None:
        self.calls.append(response)


class FakeConversationItemResource:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, *, item) -> None:
        self.calls.append(item)


class FakeConversationResource:
    def __init__(self) -> None:
        self.item = FakeConversationItemResource()


class FakeRealtimeConnection:
    def __init__(self, events: list[object]) -> None:
        self.session = FakeSessionResource()
        self.input_audio_buffer = FakeInputAudioBufferResource()
        self.response = FakeResponseResource()
        self.conversation = FakeConversationResource()
        self._events = list(events)

    async def recv(self):
        if not self._events:
            raise RuntimeError("No fake realtime events remaining")
        return self._events.pop(0)


class FakeConnectionManager:
    def __init__(self, connection: FakeRealtimeConnection) -> None:
        self.connection = connection
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self.connection

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.exited = True


class FakeRealtimeClient:
    def __init__(self, manager: FakeConnectionManager) -> None:
        self.manager = manager
        self.realtime = SimpleNamespace(connect=self.connect)

    def connect(self, *, model: str):
        self.model = model
        return self.manager


class OpenAIRealtimeSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TwinrConfig(
            openai_api_key="test-key",
            project_root="/tmp/test-project",
            personality_dir="personality",
            openai_realtime_model="gpt-4o-realtime-preview",
            openai_realtime_voice="sage",
            openai_realtime_instructions="Speak concise German.",
            openai_realtime_transcription_model="whisper-1",
            openai_realtime_language="de",
            openai_realtime_input_sample_rate=24000,
        )

    def make_session(self, *events: object) -> tuple[OpenAIRealtimeSession, FakeRealtimeConnection, FakeConnectionManager]:
        connection = FakeRealtimeConnection(list(events))
        manager = FakeConnectionManager(connection)
        client = FakeRealtimeClient(manager)
        session = OpenAIRealtimeSession(self.config, client=client)
        return session, connection, manager

    def test_open_configures_realtime_session(self) -> None:
        session, connection, manager = self.make_session()

        with session:
            pass

        self.assertTrue(manager.entered)
        self.assertTrue(manager.exited)
        self.assertEqual(connection.session.calls[0]["type"], "realtime")
        self.assertEqual(connection.session.calls[0]["output_modalities"], ["audio"])
        self.assertEqual(connection.session.calls[0]["audio"]["output"]["voice"], "sage")
        self.assertEqual(connection.session.calls[0]["audio"]["input"]["format"]["type"], "audio/pcm")
        self.assertEqual(connection.session.calls[0]["audio"]["input"]["format"]["rate"], 24000)
        self.assertEqual(connection.session.calls[0]["audio"]["input"]["noise_reduction"]["type"], "far_field")
        self.assertEqual(connection.session.calls[0]["audio"]["output"]["format"]["rate"], 24000)
        self.assertEqual(connection.session.calls[0]["audio"]["input"]["transcription"]["language"], "de")
        self.assertIsNone(connection.session.calls[0]["audio"]["input"]["turn_detection"])

    def test_open_merges_base_and_realtime_instructions(self) -> None:
        session, connection, _manager = self.make_session()
        session = OpenAIRealtimeSession(
            self.config,
            client=FakeRealtimeClient(FakeConnectionManager(connection)),
            base_instructions="Base context",
        )

        with session:
            pass

        self.assertEqual(
            connection.session.calls[0]["instructions"],
            "Base context\n\n"
            "Speak in clear, warm, natural standard German. "
            "Keep responses concise, practical, and easy for a senior user to understand. "
            "Do not use an English accent. "
            "If the user explicitly asks for a printout, use the print_receipt tool with a short focus hint and optional exact text. "
            "If the user asks for any current, external, or otherwise freshness-sensitive information that benefits from web research, first say one short German sentence that you are checking the web and that this may take a moment, then call the search_live_info tool. "
            "If the user explicitly asks you to remember an important fact for future turns, use the remember_memory tool. "
            "If the user explicitly asks you to change your future speaking style or behavior, use the update_personality tool. "
            "If the user explicitly asks you to remember a stable user-profile fact or preference, use the update_user_profile tool. "
            "If the user asks you to look at them, an object, a document, or something they are showing to the camera, call the inspect_camera tool. "
            "If the user clearly wants to stop or pause the conversation for now, call the end_conversation tool and then say a short goodbye.\n\n"
            "Speak concise German.",
        )

    def test_open_loads_latest_hidden_context_from_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            personality_dir = Path(temp_dir) / "personality"
            personality_dir.mkdir()
            state_dir = Path(temp_dir) / "state"
            state_dir.mkdir()
            (personality_dir / "SYSTEM.md").write_text("System context", encoding="utf-8")
            (personality_dir / "PERSONALITY.md").write_text("Old style context", encoding="utf-8")
            (personality_dir / "USER.md").write_text("User profile", encoding="utf-8")
            (state_dir / "MEMORY.md").write_text(
                "\n".join(
                    [
                        "# Twinr Memory",
                        "",
                        "## Entries",
                        "",
                        "### MEM-20260313T120000Z",
                        "- kind: appointment",
                        "- created_at: 2026-03-13T12:00:00+00:00",
                        "- updated_at: 2026-03-13T12:00:00+00:00",
                        "- summary: Arzttermin am Montag um 14 Uhr.",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(state_dir / "MEMORY.md"),
                openai_realtime_model="gpt-4o-realtime-preview",
                openai_realtime_voice="sage",
            )
            connection = FakeRealtimeConnection([])
            session = OpenAIRealtimeSession(
                config,
                client=FakeRealtimeClient(FakeConnectionManager(connection)),
            )
            (personality_dir / "PERSONALITY.md").write_text("Updated style context", encoding="utf-8")

            with session:
                pass

        instructions = connection.session.calls[0]["instructions"]
        self.assertIn("SYSTEM:\nSystem context", instructions)
        self.assertIn("PERSONALITY:\nUpdated style context", instructions)
        self.assertIn("MEMORY:\nDurable remembered items explicitly saved for future turns:", instructions)
        self.assertIn("Arzttermin am Montag um 14 Uhr.", instructions)

    def test_open_includes_expected_tools_when_handlers_exist(self) -> None:
        session, connection, _manager = self.make_session()
        session = OpenAIRealtimeSession(
            self.config,
            client=FakeRealtimeClient(FakeConnectionManager(connection)),
            tool_handlers={
                "print_receipt": lambda _arguments: {"status": "printed"},
                "search_live_info": lambda _arguments: {"status": "ok", "answer": "Antwort"},
                "remember_memory": lambda _arguments: {"status": "saved"},
                "update_user_profile": lambda _arguments: {"status": "updated"},
                "update_personality": lambda _arguments: {"status": "updated"},
                "inspect_camera": lambda _arguments: {"status": "ok", "answer": "Kamerabild"},
                "end_conversation": lambda _arguments: {"status": "ending"},
            },
        )

        with session:
            pass

        self.assertEqual(connection.session.calls[0]["tool_choice"], "auto")
        self.assertEqual(
            [tool["name"] for tool in connection.session.calls[0]["tools"]],
            [
                "print_receipt",
                "search_live_info",
                "remember_memory",
                "update_user_profile",
                "update_personality",
                "end_conversation",
                "inspect_camera",
            ],
        )
        self.assertIn("focus_hint", connection.session.calls[0]["tools"][0]["parameters"]["properties"])
        self.assertIn("question", connection.session.calls[0]["tools"][1]["parameters"]["properties"])
        self.assertIn("summary", connection.session.calls[0]["tools"][2]["parameters"]["properties"])
        self.assertIn("instruction", connection.session.calls[0]["tools"][3]["parameters"]["properties"])
        self.assertIn("instruction", connection.session.calls[0]["tools"][4]["parameters"]["properties"])
        self.assertIn("reason", connection.session.calls[0]["tools"][5]["parameters"]["properties"])
        self.assertIn("question", connection.session.calls[0]["tools"][6]["parameters"]["properties"])

    def test_run_audio_turn_streams_audio_and_collects_text(self) -> None:
        audio_chunks: list[bytes] = []
        text_deltas: list[str] = []
        session, connection, _manager = self.make_session(
            SimpleNamespace(type="conversation.item.input_audio_transcription.completed", transcript="Hallo Twinr"),
            SimpleNamespace(type="response.output_audio_transcript.delta", delta="Guten "),
            SimpleNamespace(type="response.output_audio.delta", delta=base64.b64encode(b"PCM").decode("ascii")),
            SimpleNamespace(type="response.output_audio_transcript.delta", delta="Tag"),
            SimpleNamespace(
                type="response.done",
                response=SimpleNamespace(
                    id="resp_rt_123",
                    model="gpt-4o-realtime-preview",
                    usage=_fake_usage(),
                ),
            ),
        )

        with session:
            turn = session.run_audio_turn(
                b"\x01\x02" * 40_000,
                conversation=(("system", "Be concise"), ("assistant", "Earlier answer")),
                on_audio_chunk=audio_chunks.append,
                on_output_text_delta=text_deltas.append,
            )

        self.assertEqual(turn.transcript, "Hallo Twinr")
        self.assertEqual(turn.response_text, "Guten Tag")
        self.assertEqual(turn.response_id, "resp_rt_123")
        self.assertEqual(turn.model, "gpt-4o-realtime-preview")
        assert turn.token_usage is not None
        self.assertEqual(turn.token_usage.total_tokens, 104)
        self.assertEqual(audio_chunks, [b"PCM"])
        self.assertEqual(text_deltas, ["Guten ", "Tag"])
        self.assertEqual(connection.conversation.item.calls[0]["role"], "system")
        self.assertEqual(connection.conversation.item.calls[0]["content"][0]["type"], "input_text")
        self.assertEqual(connection.conversation.item.calls[1]["role"], "assistant")
        self.assertEqual(connection.conversation.item.calls[1]["content"][0]["type"], "output_text")
        self.assertEqual(connection.input_audio_buffer.commit_calls, 1)
        self.assertGreater(len(connection.input_audio_buffer.append_calls), 1)
        self.assertEqual(connection.response.calls[0], {})

    def test_run_audio_turn_handles_function_call_and_continues_response(self) -> None:
        tool_calls: list[dict] = []
        session, connection, _manager = self.make_session(
            SimpleNamespace(
                type="response.done",
                response=SimpleNamespace(
                    id="resp_tool_1",
                    output=[
                        SimpleNamespace(
                            type="function_call",
                            name="print_receipt",
                            call_id="call_print_1",
                            arguments='{"text":"Wichtige Info"}',
                        )
                    ],
                ),
            ),
            SimpleNamespace(type="response.output_audio_transcript.delta", delta="Ist gedruckt."),
            SimpleNamespace(
                type="response.done",
                response=SimpleNamespace(
                    id="resp_tool_2",
                    output=[
                        SimpleNamespace(
                            type="message",
                            content=[SimpleNamespace(transcript="Ist gedruckt.", text=None)],
                        )
                    ],
                ),
            ),
        )
        session = OpenAIRealtimeSession(
            self.config,
            client=FakeRealtimeClient(FakeConnectionManager(connection)),
            tool_handlers={
                "print_receipt": lambda arguments: tool_calls.append(arguments) or {"status": "printed"}
            },
        )

        with session:
            turn = session.run_audio_turn(b"\x01\x02" * 100)

        self.assertEqual(tool_calls, [{"text": "Wichtige Info"}])
        self.assertEqual(turn.response_text, "Ist gedruckt.")
        self.assertEqual(connection.conversation.item.calls[0]["type"], "function_call_output")
        self.assertEqual(connection.conversation.item.calls[0]["call_id"], "call_print_1")
        self.assertEqual(connection.response.calls, [{}, {}])

    def test_run_audio_turn_marks_end_conversation_when_tool_called(self) -> None:
        end_calls: list[dict] = []
        session, connection, _manager = self.make_session(
            SimpleNamespace(
                type="response.done",
                response=SimpleNamespace(
                    id="resp_end_1",
                    output=[
                        SimpleNamespace(
                            type="function_call",
                            name="end_conversation",
                            call_id="call_end_1",
                            arguments='{"reason":"user said stop"}',
                        )
                    ],
                ),
            ),
            SimpleNamespace(type="response.output_audio_transcript.delta", delta="Bis bald."),
            SimpleNamespace(
                type="response.done",
                response=SimpleNamespace(
                    id="resp_end_2",
                    output=[
                        SimpleNamespace(
                            type="message",
                            content=[SimpleNamespace(transcript="Bis bald.", text=None)],
                        )
                    ],
                ),
            ),
        )
        session = OpenAIRealtimeSession(
            self.config,
            client=FakeRealtimeClient(FakeConnectionManager(connection)),
            tool_handlers={
                "end_conversation": lambda arguments: end_calls.append(arguments) or {"status": "ending"}
            },
        )

        with session:
            turn = session.run_audio_turn(b"\x01\x02" * 100)

        self.assertEqual(end_calls, [{"reason": "user said stop"}])
        self.assertTrue(turn.end_conversation)
        self.assertEqual(turn.response_text, "Bis bald.")

    def test_run_audio_turn_handles_search_function_call_and_continues_response(self) -> None:
        search_calls: list[dict] = []
        session, connection, _manager = self.make_session(
            SimpleNamespace(
                type="response.done",
                response=SimpleNamespace(
                    id="resp_search_1",
                    output=[
                        SimpleNamespace(
                            type="function_call",
                            name="search_live_info",
                            call_id="call_search_1",
                            arguments='{"question":"Wie wird das Wetter morgen?","location_hint":"Schwarzenbek"}',
                        )
                    ],
                ),
            ),
            SimpleNamespace(type="response.output_audio_transcript.delta", delta="Morgen wird es kuehl und nass."),
            SimpleNamespace(
                type="response.done",
                response=SimpleNamespace(
                    id="resp_search_2",
                    output=[
                        SimpleNamespace(
                            type="message",
                            content=[SimpleNamespace(transcript="Morgen wird es kuehl und nass.", text=None)],
                        )
                    ],
                ),
            ),
        )
        session = OpenAIRealtimeSession(
            self.config,
            client=FakeRealtimeClient(FakeConnectionManager(connection)),
            tool_handlers={
                "search_live_info": lambda arguments: search_calls.append(arguments) or {"status": "ok", "answer": "11 Grad"}
            },
        )

        with session:
            turn = session.run_audio_turn(b"\x01\x02" * 100)

        self.assertEqual(
            search_calls,
            [{"question": "Wie wird das Wetter morgen?", "location_hint": "Schwarzenbek"}],
        )
        self.assertEqual(turn.response_text, "Morgen wird es kuehl und nass.")
        self.assertEqual(connection.conversation.item.calls[0]["type"], "function_call_output")
        self.assertEqual(connection.response.calls, [{}, {}])

    def test_run_text_turn_creates_user_message(self) -> None:
        session, connection, _manager = self.make_session(
            SimpleNamespace(type="response.output_audio_transcript.delta", delta="Hallo"),
            SimpleNamespace(type="response.done", response=SimpleNamespace(id="resp_rt_456")),
        )

        with session:
            turn = session.run_text_turn(
                "Sag hallo",
                conversation=(("user", "Früher"), ("assistant", "Antwort")),
            )

        self.assertEqual(turn.transcript, "Sag hallo")
        self.assertEqual(turn.response_text, "Hallo")
        self.assertEqual(connection.conversation.item.calls[0]["role"], "user")
        self.assertEqual(connection.conversation.item.calls[0]["content"][0]["text"], "Früher")
        self.assertEqual(connection.conversation.item.calls[1]["role"], "assistant")
        self.assertEqual(connection.conversation.item.calls[1]["content"][0]["type"], "output_text")
        self.assertEqual(connection.conversation.item.calls[2]["role"], "user")
        self.assertEqual(connection.conversation.item.calls[2]["content"][0]["type"], "input_text")
        self.assertEqual(connection.conversation.item.calls[2]["content"][0]["text"], "Sag hallo")

    def test_run_audio_turn_uses_placeholder_when_transcription_missing(self) -> None:
        session, _connection, _manager = self.make_session(
            SimpleNamespace(type="response.output_audio_transcript.delta", delta="Antwort"),
            SimpleNamespace(type="response.done", response=SimpleNamespace(id="resp_rt_789")),
        )

        with session:
            turn = session.run_audio_turn(b"\x01\x02" * 100)

        self.assertEqual(turn.transcript, "[voice input]")
        self.assertEqual(turn.response_text, "Antwort")


if __name__ == "__main__":
    unittest.main()
