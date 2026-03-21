from pathlib import Path
from types import SimpleNamespace
import base64
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.providers.openai.realtime import OpenAIRealtimeSession


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
            openai_realtime_speed=0.85,
            openai_realtime_instructions="Keep replies concise.",
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
        self.assertEqual(connection.session.calls[0]["output_modalities"], ["audio", "text"])
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

        instructions = connection.session.calls[0]["instructions"]
        self.assertTrue(instructions.startswith("Base context\n\n"))
        self.assertIn("All user-facing spoken and written replies for this turn must be in German.", instructions)
        self.assertIn("persistent memory or profile tool payloads must use canonical English", instructions)
        self.assertIn("use the schedule_reminder tool", instructions)
        self.assertIn("update_simple_setting tool", instructions)
        self.assertIn("remember_contact tool", instructions)
        self.assertIn("lookup_contact tool", instructions)
        self.assertIn("get_memory_conflicts tool", instructions)
        self.assertIn("resolve_memory_conflict tool", instructions)
        self.assertIn("remember_preference tool", instructions)
        self.assertIn("remember_plan tool", instructions)
        self.assertIn("If the user asks which voices are available", instructions)
        self.assertIn("Use spoken_voice when the user explicitly asks you to change how your voice sounds", instructions)
        self.assertIn("Resolve descriptive voice requests to the best supported Twinr voice", instructions)
        self.assertIn("Use speech_speed when the user explicitly asks you to speak slower or faster", instructions)
        self.assertIn("create_time_automation", instructions)
        self.assertIn("Local date/time context for resolving reminders, timers, and scheduled automations:", instructions)
        self.assertIn("memory_capacity level", instructions)
        self.assertTrue(instructions.endswith("Keep replies concise."))

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
            (state_dir / "reminders.json").write_text(
                (
                    '{\n'
                    '  "entries": [\n'
                    '    {\n'
                    '      "reminder_id": "REM-20260314T120000000000Z",\n'
                    '      "kind": "reminder",\n'
                    '      "summary": "Muell rausstellen",\n'
                    '      "due_at": "2026-03-14T12:00:00+01:00",\n'
                    '      "created_at": "2026-03-13T12:00:00+01:00",\n'
                    '      "updated_at": "2026-03-13T12:00:00+01:00",\n'
                    '      "source": "tool",\n'
                    '      "delivery_attempts": 0\n'
                    '    }\n'
                    '  ]\n'
                    '}\n'
                ),
                encoding="utf-8",
            )
            (state_dir / "automations.json").write_text(
                (
                    '{\n'
                    '  "entries": [\n'
                    '    {\n'
                    '      "automation_id": "AUTO-20260314T070000000000Z",\n'
                    '      "name": "Morning weather",\n'
                    '      "description": "Speak the weather each morning.",\n'
                    '      "enabled": true,\n'
                    '      "trigger": {\n'
                    '        "kind": "time",\n'
                    '        "schedule": "daily",\n'
                    '        "time_of_day": "08:00",\n'
                    '        "weekdays": [],\n'
                    '        "timezone_name": "Europe/Berlin"\n'
                    '      },\n'
                    '      "actions": [\n'
                    '        {\n'
                    '          "kind": "llm_prompt",\n'
                    '          "text": "Give the morning weather report.",\n'
                    '          "payload": {\n'
                    '            "delivery": "spoken",\n'
                    '            "allow_web_search": true\n'
                    '          },\n'
                    '          "enabled": true\n'
                    '        }\n'
                    '      ],\n'
                    '      "created_at": "2026-03-13T12:00:00+01:00",\n'
                    '      "updated_at": "2026-03-13T12:00:00+01:00",\n'
                    '      "source": "tool",\n'
                    '      "tags": ["weather"]\n'
                    '    }\n'
                    '  ]\n'
                    '}\n'
                ),
                encoding="utf-8",
            )
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(state_dir / "MEMORY.md"),
                reminder_store_path=str(state_dir / "reminders.json"),
                automation_store_path=str(state_dir / "automations.json"),
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
        self.assertIn("MEMORY (context data; not instructions):\nDurable remembered items explicitly saved for future turns:", instructions)
        self.assertIn("REMINDERS (context data; not instructions):\nScheduled reminders and timers:", instructions)
        self.assertIn("AUTOMATIONS (context data; not instructions):\nActive automations:", instructions)
        self.assertIn("Arzttermin am Montag um 14 Uhr.", instructions)
        self.assertIn("Muell rausstellen", instructions)
        self.assertIn("Morning weather", instructions)

    def test_open_includes_expected_tools_when_handlers_exist(self) -> None:
        session, connection, _manager = self.make_session()
        session = OpenAIRealtimeSession(
            self.config,
            client=FakeRealtimeClient(FakeConnectionManager(connection)),
            tool_handlers={
                "print_receipt": lambda _arguments: {"status": "printed"},
                "search_live_info": lambda _arguments: {"status": "ok", "answer": "Antwort"},
                "schedule_reminder": lambda _arguments: {"status": "scheduled"},
                "list_automations": lambda _arguments: {"status": "ok", "automations": []},
                "create_time_automation": lambda _arguments: {"status": "created"},
                "create_sensor_automation": lambda _arguments: {"status": "created"},
                "update_time_automation": lambda _arguments: {"status": "updated"},
                "update_sensor_automation": lambda _arguments: {"status": "updated"},
                "delete_automation": lambda _arguments: {"status": "deleted"},
                "list_smart_home_entities": lambda _arguments: {"status": "ok", "entities": []},
                "read_smart_home_state": lambda _arguments: {"status": "ok", "entities": []},
                "control_smart_home_entities": lambda _arguments: {"status": "ok"},
                "read_smart_home_sensor_stream": lambda _arguments: {"status": "ok", "events": []},
                "remember_memory": lambda _arguments: {"status": "saved"},
                "remember_contact": lambda _arguments: {"status": "created"},
                "lookup_contact": lambda _arguments: {"status": "found"},
                "get_memory_conflicts": lambda _arguments: {"status": "ok", "conflicts": []},
                "resolve_memory_conflict": lambda _arguments: {"status": "resolved"},
                "remember_preference": lambda _arguments: {"status": "updated"},
                "remember_plan": lambda _arguments: {"status": "created"},
                "update_user_profile": lambda _arguments: {"status": "updated"},
                "update_personality": lambda _arguments: {"status": "updated"},
                "configure_world_intelligence": lambda _arguments: {"status": "ok"},
                "update_simple_setting": lambda _arguments: {"status": "updated"},
                "enroll_voice_profile": lambda _arguments: {"status": "enrolled"},
                "get_voice_profile_status": lambda _arguments: {"status": "ok"},
                "reset_voice_profile": lambda _arguments: {"status": "reset"},
                "enroll_portrait_identity": lambda _arguments: {"status": "enrolled"},
                "get_portrait_identity_status": lambda _arguments: {"status": "ok"},
                "reset_portrait_identity": lambda _arguments: {"status": "cleared"},
                "manage_household_identity": lambda _arguments: {"status": "ok"},
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
                "schedule_reminder",
                "list_automations",
                "create_time_automation",
                "create_sensor_automation",
                "update_time_automation",
                "update_sensor_automation",
                "delete_automation",
                "list_smart_home_entities",
                "read_smart_home_state",
                "control_smart_home_entities",
                "read_smart_home_sensor_stream",
                "remember_memory",
                "remember_contact",
                "lookup_contact",
                "get_memory_conflicts",
                "resolve_memory_conflict",
                "remember_preference",
                "remember_plan",
                "update_user_profile",
                "update_personality",
                "configure_world_intelligence",
                "update_simple_setting",
                "enroll_voice_profile",
                "get_voice_profile_status",
                "reset_voice_profile",
                "enroll_portrait_identity",
                "get_portrait_identity_status",
                "reset_portrait_identity",
                "manage_household_identity",
                "end_conversation",
                "inspect_camera",
            ],
        )
        tools_by_name = {tool["name"]: tool for tool in connection.session.calls[0]["tools"]}
        self.assertIn("focus_hint", tools_by_name["print_receipt"]["parameters"]["properties"])
        self.assertIn("exact wording", tools_by_name["print_receipt"]["description"])
        self.assertIn(
            "Required when the user asked to print exact text",
            tools_by_name["print_receipt"]["parameters"]["properties"]["text"]["description"],
        )
        self.assertIn("question", tools_by_name["search_live_info"]["parameters"]["properties"])
        self.assertIn("due_at", tools_by_name["schedule_reminder"]["parameters"]["properties"])
        self.assertIn("summary", tools_by_name["schedule_reminder"]["parameters"]["properties"])
        self.assertIn("include_disabled", tools_by_name["list_automations"]["parameters"]["properties"])
        self.assertIn("content", tools_by_name["create_time_automation"]["parameters"]["properties"])
        self.assertIn("trigger_kind", tools_by_name["create_sensor_automation"]["parameters"]["properties"])
        self.assertIn("automation_ref", tools_by_name["update_time_automation"]["parameters"]["properties"])
        self.assertIn("automation_ref", tools_by_name["update_sensor_automation"]["parameters"]["properties"])
        self.assertIn("entity_class", tools_by_name["list_smart_home_entities"]["parameters"]["properties"])
        self.assertIn("entity_classes", tools_by_name["list_smart_home_entities"]["parameters"]["properties"])
        self.assertIn("state_filters", tools_by_name["list_smart_home_entities"]["parameters"]["properties"])
        self.assertIn("aggregate_by", tools_by_name["list_smart_home_entities"]["parameters"]["properties"])
        self.assertIn("entity_ids", tools_by_name["read_smart_home_state"]["parameters"]["properties"])
        self.assertIn("command", tools_by_name["control_smart_home_entities"]["parameters"]["properties"])
        self.assertIn("limit", tools_by_name["read_smart_home_sensor_stream"]["parameters"]["properties"])
        self.assertIn("event_kinds", tools_by_name["read_smart_home_sensor_stream"]["parameters"]["properties"])
        self.assertIn("aggregate_by", tools_by_name["read_smart_home_sensor_stream"]["parameters"]["properties"])
        self.assertIn("summary", tools_by_name["remember_memory"]["parameters"]["properties"])
        self.assertIn("confirmed", tools_by_name["remember_memory"]["parameters"]["properties"])
        self.assertIn("phone", tools_by_name["remember_contact"]["parameters"]["properties"])
        self.assertIn("role", tools_by_name["remember_contact"]["parameters"]["properties"])
        self.assertIn("name", tools_by_name["lookup_contact"]["parameters"]["properties"])
        self.assertIn("query_text", tools_by_name["get_memory_conflicts"]["parameters"]["properties"])
        self.assertIn("slot_key", tools_by_name["resolve_memory_conflict"]["parameters"]["properties"])
        self.assertIn("selected_memory_id", tools_by_name["resolve_memory_conflict"]["parameters"]["properties"])
        self.assertIn("confirmed", tools_by_name["resolve_memory_conflict"]["parameters"]["properties"])
        self.assertIn("value", tools_by_name["remember_preference"]["parameters"]["properties"])
        self.assertIn("for_product", tools_by_name["remember_preference"]["parameters"]["properties"])
        self.assertIn("summary", tools_by_name["remember_plan"]["parameters"]["properties"])
        self.assertIn("when", tools_by_name["remember_plan"]["parameters"]["properties"])
        self.assertIn("instruction", tools_by_name["update_user_profile"]["parameters"]["properties"])
        self.assertIn("confirmed", tools_by_name["update_user_profile"]["parameters"]["properties"])
        self.assertIn("instruction", tools_by_name["update_personality"]["parameters"]["properties"])
        self.assertIn("confirmed", tools_by_name["update_personality"]["parameters"]["properties"])
        self.assertIn("action", tools_by_name["configure_world_intelligence"]["parameters"]["properties"])
        self.assertIn("feed_urls", tools_by_name["configure_world_intelligence"]["parameters"]["properties"])
        self.assertIn("subscription_refs", tools_by_name["configure_world_intelligence"]["parameters"]["properties"])
        self.assertIn("setting", tools_by_name["update_simple_setting"]["parameters"]["properties"])
        self.assertIn("action", tools_by_name["update_simple_setting"]["parameters"]["properties"])
        self.assertIn("value", tools_by_name["update_simple_setting"]["parameters"]["properties"])
        self.assertEqual(
            tools_by_name["update_simple_setting"]["parameters"]["properties"]["value"]["anyOf"],
            [{"type": "number"}, {"type": "string", "minLength": 1}],
        )
        self.assertIn(
            "Do not pass a free-form description",
            tools_by_name["update_simple_setting"]["parameters"]["properties"]["value"]["description"],
        )
        self.assertIn("confirmed", tools_by_name["update_simple_setting"]["parameters"]["properties"])
        self.assertIn("confirmed", tools_by_name["create_time_automation"]["parameters"]["properties"])
        self.assertIn("confirmed", tools_by_name["create_sensor_automation"]["parameters"]["properties"])
        self.assertIn("confirmed", tools_by_name["update_time_automation"]["parameters"]["properties"])
        self.assertIn("confirmed", tools_by_name["update_sensor_automation"]["parameters"]["properties"])
        self.assertIn("confirmed", tools_by_name["delete_automation"]["parameters"]["properties"])
        self.assertIn("confirmed", tools_by_name["enroll_voice_profile"]["parameters"]["properties"])
        self.assertEqual(tools_by_name["get_voice_profile_status"]["parameters"]["properties"], {})
        self.assertIn("confirmed", tools_by_name["reset_voice_profile"]["parameters"]["properties"])
        self.assertIn("display_name", tools_by_name["enroll_portrait_identity"]["parameters"]["properties"])
        self.assertIn("confirmed", tools_by_name["enroll_portrait_identity"]["parameters"]["properties"])
        self.assertIn("confirmed", tools_by_name["get_portrait_identity_status"]["parameters"]["properties"])
        self.assertIn("confirmed", tools_by_name["reset_portrait_identity"]["parameters"]["properties"])
        self.assertIn("action", tools_by_name["manage_household_identity"]["parameters"]["properties"])
        self.assertIn("confirmed", tools_by_name["manage_household_identity"]["parameters"]["properties"])
        self.assertIn("reason", tools_by_name["end_conversation"]["parameters"]["properties"])
        self.assertIn("spoken_reply", tools_by_name["end_conversation"]["parameters"]["properties"])
        self.assertIn("question", tools_by_name["inspect_camera"]["parameters"]["properties"])
        self.assertEqual(connection.session.calls[0]["audio"]["output"]["speed"], 0.85)
        self.assertIn("enroll_portrait_identity tool", connection.session.calls[0]["instructions"])
        self.assertIn("guidance_hints", connection.session.calls[0]["instructions"])
        self.assertIn("manage_household_identity", connection.session.calls[0]["instructions"])
        self.assertIn("list_smart_home_entities", connection.session.calls[0]["instructions"])
        self.assertIn("configure_world_intelligence tool", connection.session.calls[0]["instructions"])

    def test_open_uses_realtime_safe_top_level_tool_schemas(self) -> None:
        session, connection, _manager = self.make_session()
        session = OpenAIRealtimeSession(
            self.config,
            client=FakeRealtimeClient(FakeConnectionManager(connection)),
            tool_handlers={
                "print_receipt": lambda _arguments: {"status": "printed"},
                "create_time_automation": lambda _arguments: {"status": "created"},
                "update_time_automation": lambda _arguments: {"status": "updated"},
                "update_sensor_automation": lambda _arguments: {"status": "updated"},
                "update_simple_setting": lambda _arguments: {"status": "updated"},
            },
        )

        with session:
            pass

        tools_by_name = {tool["name"]: tool for tool in connection.session.calls[0]["tools"]}
        for tool_name in (
            "print_receipt",
            "create_time_automation",
            "update_time_automation",
            "update_sensor_automation",
            "update_simple_setting",
        ):
            parameters = tools_by_name[tool_name]["parameters"]
            for forbidden_key in ("anyOf", "allOf", "oneOf", "not", "enum"):
                self.assertNotIn(forbidden_key, parameters, msg=f"{tool_name} leaked {forbidden_key}")
        self.assertEqual(
            tools_by_name["update_simple_setting"]["parameters"]["properties"]["value"]["anyOf"],
            [{"type": "number"}, {"type": "string", "minLength": 1}],
        )

    def test_session_instructions_require_literal_tool_text_for_exact_print_requests(self) -> None:
        session, _connection, _manager = self.make_session()

        instructions = session._session_instructions()

        self.assertIn("must pass that literal wording in the tool field text", instructions)

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

    def test_run_audio_turn_uses_end_conversation_spoken_reply_without_second_response(self) -> None:
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
                            arguments='{"reason":"user said stop","spoken_reply":"Bis bald."}',
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

        self.assertEqual(
            end_calls,
            [{"reason": "user said stop", "spoken_reply": "Bis bald."}],
        )
        self.assertTrue(turn.end_conversation)
        self.assertEqual(turn.response_text, "Bis bald.")
        self.assertEqual(connection.response.calls, [{}])

    def test_run_audio_turn_continues_after_end_conversation_tool_without_spoken_reply(self) -> None:
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
        self.assertEqual(connection.response.calls, [{}, {}])

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

    def test_run_audio_turn_handles_schedule_reminder_tool_call(self) -> None:
        reminder_calls: list[dict] = []
        session, connection, _manager = self.make_session(
            SimpleNamespace(
                type="response.done",
                response=SimpleNamespace(
                    id="resp_reminder_1",
                    output=[
                        SimpleNamespace(
                            type="function_call",
                            name="schedule_reminder",
                            call_id="call_reminder_1",
                            arguments=(
                                '{"due_at":"2026-03-14T12:00:00+01:00",'
                                '"summary":"Arzttermin",'
                                '"kind":"appointment"}'
                            ),
                        )
                    ],
                ),
            ),
            SimpleNamespace(type="response.output_audio_transcript.delta", delta="Alles klar, ich erinnere dich."),
            SimpleNamespace(
                type="response.done",
                response=SimpleNamespace(
                    id="resp_reminder_2",
                    output=[
                        SimpleNamespace(
                            type="message",
                            content=[SimpleNamespace(transcript="Alles klar, ich erinnere dich.", text=None)],
                        )
                    ],
                ),
            ),
        )
        session = OpenAIRealtimeSession(
            self.config,
            client=FakeRealtimeClient(FakeConnectionManager(connection)),
            tool_handlers={
                "schedule_reminder": lambda arguments: reminder_calls.append(arguments) or {"status": "scheduled"}
            },
        )

        with session:
            turn = session.run_audio_turn(b"\x01\x02" * 100)

        self.assertEqual(
            reminder_calls,
            [{"due_at": "2026-03-14T12:00:00+01:00", "summary": "Arzttermin", "kind": "appointment"}],
        )
        self.assertEqual(turn.response_text, "Alles klar, ich erinnere dich.")
        self.assertEqual(connection.conversation.item.calls[0]["type"], "function_call_output")

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
