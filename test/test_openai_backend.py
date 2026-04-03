from dataclasses import replace
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.tools.schemas.contracts import build_agent_tool_schemas
from twinr.providers.openai import OpenAIBackend, OpenAIImageInput
from twinr.providers.openai.api.adapters import _SUPERVISOR_DECISION_SCHEMA, _normalize_openai_tool_schema
from twinr.providers.openai.core.client import _default_client_factory, close_cached_openai_clients


def _search_voice_payload(
    spoken_answer: str,
    *,
    verification_status: str = "verified",
    question_resolved: bool = True,
    site_follow_up_recommended: bool = False,
    site_follow_up_reason: str | None = None,
    site_follow_up_url: str | None = None,
    site_follow_up_domain: str | None = None,
) -> str:
    return json.dumps(
        {
            "spoken_answer": spoken_answer,
            "verification_status": verification_status,
            "question_resolved": question_resolved,
            "site_follow_up_recommended": site_follow_up_recommended,
            "site_follow_up_reason": site_follow_up_reason,
            "site_follow_up_url": site_follow_up_url,
            "site_follow_up_domain": site_follow_up_domain,
        }
    )


def _fake_usage(
    *,
    input_tokens: int = 120,
    output_tokens: int = 48,
    total_tokens: int = 168,
    cached_tokens: int = 12,
    reasoning_tokens: int = 7,
):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
    )


class FakeBinaryResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.closed = False

    def read(self) -> bytes:
        return self._payload

    def iter_bytes(self, _chunk_size: int | None = None):
        midpoint = max(1, len(self._payload) // 2)
        yield self._payload[:midpoint]
        if self.closed:
            return
        yield self._payload[midpoint:]

    def close(self) -> None:
        self.closed = True


class FakeResponsesAPI:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.stream_calls: list[dict] = []
        self.output_text = "Backend answer"
        self.output = [SimpleNamespace(type="web_search_call")]
        self.queued_output_texts: list[str] = []
        self.queued_exceptions: list[Exception] = []
        self.queued_payloads: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.queued_exceptions:
            raise self.queued_exceptions.pop(0)
        payload = self.queued_payloads.pop(0) if self.queued_payloads else {}
        output_text = self.queued_output_texts.pop(0) if self.queued_output_texts else self.output_text
        response_format = ((kwargs.get("text") or {}).get("format") or {})
        if response_format.get("name") == "twinr_live_search_spoken_answer":
            output_text = _search_voice_payload(output_text)
        elif response_format.get("name") == "twinr_print_receipt":
            output_text = json.dumps({"status": "ready", "text": output_text})
        return SimpleNamespace(
            id=payload.get("id", "resp_123"),
            _request_id=payload.get("_request_id", "req_123"),
            model=payload.get("model", kwargs["model"]),
            usage=payload.get("usage", _fake_usage()),
            output_text=payload.get("output_text", output_text),
            output=payload.get("output", self.output),
            status=payload.get("status", "completed"),
            incomplete_details=payload.get("incomplete_details"),
        )

    def stream(self, **kwargs):
        self.stream_calls.append(kwargs)

        class _StreamManager:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def __iter__(self):
                yield SimpleNamespace(type="response.output_text.delta", delta="Hello")
                yield SimpleNamespace(type="response.output_text.delta", delta=" there")

            def get_final_response(self):
                return SimpleNamespace(
                    id="resp_stream",
                    _request_id="req_stream",
                    model=kwargs["model"],
                    usage=_fake_usage(input_tokens=96, output_tokens=32, total_tokens=128),
                    output_text="Hello there",
                    output=[],
                )

        return _StreamManager()


class FakeTranscriptionsAPI:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return "Transcribed speech"


class InternalServerError(RuntimeError):
    pass


class FakeModelAccessError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.status_code = 403
        self.body = {"error": {"code": "model_not_found"}}


class FakeSpeechAPI:
    def __init__(self, fail_first: bool = False) -> None:
        self.calls: list[dict] = []
        self.fail_first = fail_first
        self.streaming_responses: list[FakeBinaryResponse] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_first and len(self.calls) == 1:
            raise FakeModelAccessError("project does not have access to model")
        return FakeBinaryResponse(b"AUDIO")

    @property
    def with_streaming_response(self):
        parent = self

        class _StreamingWrapper:
            def create(self, **kwargs):
                parent.calls.append(kwargs)
                if parent.fail_first and len(parent.calls) == 1:
                    raise FakeModelAccessError("project does not have access to model")

                class _Manager:
                    def __enter__(self):
                        response = FakeBinaryResponse(b"AUDIO")
                        parent.streaming_responses.append(response)
                        return response

                    def __exit__(self, exc_type, exc, tb):
                        return None

                return _Manager()

        return _StreamingWrapper()


class FakeChatCompletionsAPI:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.content = "Morgen in Schwarzenbek 8 Grad, leichter Regen."
        self.annotations = [
            SimpleNamespace(
                url_citation=SimpleNamespace(url="https://weather.example/forecast")
            )
        ]

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.content
        response_format = kwargs.get("response_format") or {}
        schema_name = (response_format.get("json_schema") or {}).get("name")
        if schema_name == "twinr_live_search_spoken_answer":
            content = _search_voice_payload(content)
        return SimpleNamespace(
            id="chatcmpl_123",
            _request_id="req_chat_123",
            model=kwargs["model"],
            usage=SimpleNamespace(
                prompt_tokens=32,
                completion_tokens=44,
                total_tokens=76,
                prompt_tokens_details=SimpleNamespace(cached_tokens=4),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
            ),
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=content,
                        annotations=self.annotations,
                    ),
                )
            ],
        )


class OpenAIBackendTests(unittest.TestCase):
    def test_self_coding_scope_schema_survives_openai_normalization_without_freeform_objects(self) -> None:
        schemas = {
            schema["name"]: _normalize_openai_tool_schema(schema)
            for schema in build_agent_tool_schemas(("propose_skill_learning", "answer_skill_question"))
        }

        self.assertEqual(
            schemas["propose_skill_learning"]["parameters"]["properties"]["scope"]["type"],
            ["string", "null"],
        )
        self.assertEqual(
            schemas["answer_skill_question"]["parameters"]["properties"]["scope"]["type"],
            ["string", "null"],
        )

    def test_openai_normalization_makes_optional_tool_fields_nullable_in_strict_mode(self) -> None:
        schemas = {
            schema["name"]: _normalize_openai_tool_schema(schema)
            for schema in build_agent_tool_schemas(("print_receipt", "send_whatsapp_message"))
        }

        print_schema = schemas["print_receipt"]["parameters"]["properties"]
        self.assertEqual(print_schema["focus_hint"]["type"], ["string", "null"])
        self.assertEqual(print_schema["text"]["type"], ["string", "null"])

        whatsapp_schema = schemas["send_whatsapp_message"]["parameters"]["properties"]
        self.assertEqual(whatsapp_schema["family_name"]["type"], ["string", "null"])
        self.assertEqual(whatsapp_schema["role"]["type"], ["string", "null"])
        self.assertEqual(whatsapp_schema["contact_label"]["type"], ["string", "null"])
        self.assertEqual(whatsapp_schema["phone_last4"]["type"], ["string", "null"])
        self.assertEqual(whatsapp_schema["message"]["type"], ["string", "null"])
        self.assertEqual(whatsapp_schema["confirmed"]["type"], ["boolean", "null"])
        self.assertEqual(whatsapp_schema["name"]["type"], "string")

    def setUp(self) -> None:
        self.responses = FakeResponsesAPI()
        self.transcriptions = FakeTranscriptionsAPI()
        self.speech = FakeSpeechAPI()
        self.client = SimpleNamespace(
            responses=self.responses,
            chat=SimpleNamespace(completions=FakeChatCompletionsAPI()),
            audio=SimpleNamespace(
                transcriptions=self.transcriptions,
                speech=self.speech,
            ),
        )
        self.config = TwinrConfig(
            openai_api_key="test-key",
            default_model="gpt-5.2",
            openai_reasoning_effort="medium",
            openai_enable_web_search=False,
            openai_stt_model="whisper-1",
            openai_tts_model="gpt-4o-mini-tts",
            openai_tts_voice="marin",
            openai_tts_speed=0.9,
            openai_tts_format="wav",
            openai_tts_instructions="Speak in natural German.",
            openai_web_search_context_size="medium",
            openai_web_search_country="DE",
            openai_web_search_city="Berlin",
            openai_web_search_timezone="Europe/Berlin",
            openai_vision_detail="high",
        )
        self.backend = OpenAIBackend(config=self.config, client=self.client)

    def test_respond_builds_reasoning_and_web_search_request(self) -> None:
        response = self.backend.respond_with_metadata(
            "What is new today?",
            conversation=[("system", "Be concise"), ("assistant", "Earlier answer")],
            allow_web_search=True,
        )

        self.assertEqual(response.text, "Backend answer")
        self.assertTrue(response.used_web_search)
        self.assertEqual(response.response_id, "resp_123")
        self.assertEqual(response.request_id, "req_123")
        self.assertEqual(response.model, "gpt-5.2")
        assert response.token_usage is not None
        self.assertEqual(response.token_usage.input_tokens, 120)
        self.assertEqual(response.token_usage.output_tokens, 48)
        self.assertEqual(response.token_usage.cached_input_tokens, 12)
        self.assertEqual(response.token_usage.reasoning_tokens, 7)

        request = self.responses.calls[0]
        self.assertEqual(request["model"], "gpt-5.2")
        self.assertEqual(request["reasoning"], {"effort": "medium"})
        self.assertFalse(request["store"])
        self.assertTrue(request["prompt_cache_key"].startswith("twinr:response:"))
        self.assertLessEqual(len(request["prompt_cache_key"]), 64)
        self.assertNotIn("gpt-5.2:de", request["prompt_cache_key"])
        self.assertIn(
            "All user-facing spoken and written replies for this turn must be in German.",
            request["instructions"],
        )
        assistant_messages = [item for item in request["input"] if item.get("role") == "assistant"]
        self.assertEqual(len(assistant_messages), 1)
        self.assertEqual(assistant_messages[0]["content"][0]["type"], "output_text")
        self.assertEqual(request["tools"][0]["type"], "web_search")
        self.assertEqual(request["tools"][0]["search_context_size"], "medium")
        self.assertEqual(request["tools"][0]["user_location"]["country"], "DE")
        self.assertEqual(request["input"][0]["role"], "system")
        self.assertEqual(request["input"][0]["content"][0]["type"], "input_text")
        self.assertEqual(request["input"][1]["role"], "assistant")
        self.assertEqual(request["input"][1]["content"][0]["type"], "output_text")
        self.assertEqual(request["input"][2]["role"], "user")
        self.assertEqual(request["input"][2]["content"][0]["type"], "input_text")

    def test_respond_merges_base_and_request_instructions(self) -> None:
        backend = OpenAIBackend(
            config=self.config,
            client=self.client,
            base_instructions="Base context",
        )

        backend.respond_with_metadata("Hello", instructions="Task context")

        request = self.responses.calls[0]
        self.assertEqual(
            request["instructions"],
            (
                "Base context\n\n"
                "Task context\n\n"
                "All user-facing spoken and written replies for this turn must be in German."
            ),
        )

    def test_respond_loads_latest_hidden_context_from_files(self) -> None:
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
                        "- kind: contact",
                        "- created_at: 2026-03-13T12:00:00+00:00",
                        "- updated_at: 2026-03-13T12:00:00+00:00",
                        "- summary: Telefonnummer Rathaus Schwarzenbek 04151 8810.",
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
            config = replace(
                self.config,
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(state_dir / "MEMORY.md"),
                reminder_store_path=str(state_dir / "reminders.json"),
            )
            backend = OpenAIBackend(config=config, client=self.client)
            (personality_dir / "PERSONALITY.md").write_text("Updated style context", encoding="utf-8")

            backend.respond_with_metadata("Hallo")

        request = self.responses.calls[-1]
        self.assertIn(
            '<section title="SYSTEM" authority="configuration" encoding="verbatim_text">',
            request["instructions"],
        )
        self.assertIn(
            '<section title="PERSONALITY" authority="configuration" encoding="verbatim_text">',
            request["instructions"],
        )
        self.assertIn("System context", request["instructions"])
        self.assertIn("Updated style context", request["instructions"])
        self.assertIn(
            '<section title="MEMORY" authority="context_data" encoding="verbatim_text">',
            request["instructions"],
        )
        self.assertIn(
            '<section title="REMINDERS" authority="context_data" encoding="verbatim_text">',
            request["instructions"],
        )
        self.assertIn("Durable remembered items explicitly saved for future turns:", request["instructions"])
        self.assertIn("Scheduled reminders and timers:", request["instructions"])
        self.assertIn("04151 8810", request["instructions"])
        self.assertIn("Muell rausstellen", request["instructions"])
        self.assertIn("All user-facing spoken and written replies for this turn must be in German.", request["instructions"])

    def test_respond_to_images_sends_multiple_input_images(self) -> None:
        response = self.backend.respond_to_images_with_metadata(
            "Compare both images.",
            images=[
                OpenAIImageInput(
                    data=b"\x89PNG\r\n\x1a\ncamera",
                    content_type="image/png",
                    filename="camera.png",
                    label="Image 1: live camera frame.",
                ),
                OpenAIImageInput(
                    data=b"\xff\xd8\xffreference",
                    content_type="image/jpeg",
                    filename="reference.jpg",
                    label="Image 2: reference user portrait.",
                    detail="low",
                ),
            ],
        )

        self.assertEqual(response.text, "Backend answer")
        request = self.responses.calls[0]
        user_message = request["input"][0]
        self.assertEqual(user_message["role"], "user")
        self.assertEqual(
            [item["type"] for item in user_message["content"]],
            ["input_text", "input_text", "input_image", "input_text", "input_image"],
        )
        self.assertEqual(user_message["content"][2]["detail"], "high")
        self.assertEqual(user_message["content"][4]["detail"], "low")
        self.assertTrue(user_message["content"][2]["image_url"].startswith("data:image/png;base64,"))
        self.assertTrue(user_message["content"][4]["image_url"].startswith("data:image/jpeg;base64,"))

    def test_transcribe_passes_audio_tuple(self) -> None:
        transcript = self.backend.transcribe(b"RIFF", filename="sample.wav", content_type="audio/wav")

        self.assertEqual(transcript, "Transcribed speech")
        request = self.transcriptions.calls[0]
        self.assertEqual(request["model"], "whisper-1")
        self.assertEqual(request["file"], ("sample.wav", b"RIFF", "audio/wav"))
        self.assertEqual(request["response_format"], "text")

    def test_synthesize_returns_audio_bytes(self) -> None:
        audio_bytes = self.backend.synthesize("Hello from Twinr")

        self.assertEqual(audio_bytes, b"AUDIO")
        request = self.speech.calls[0]
        self.assertEqual(request["model"], "gpt-4o-mini-tts")
        self.assertEqual(request["voice"], "marin")
        self.assertEqual(request["speed"], 0.9)
        self.assertEqual(request["response_format"], "wav")
        self.assertEqual(request["instructions"], "Speak in natural German.")

    def test_synthesize_falls_back_to_legacy_tts_model_when_project_lacks_model_access(self) -> None:
        fallback_config = replace(self.config, openai_tts_model="gpt-4o-mini-tts")
        backend = OpenAIBackend(
            config=fallback_config,
            client=SimpleNamespace(
                responses=self.responses,
                audio=SimpleNamespace(
                    transcriptions=self.transcriptions,
                    speech=FakeSpeechAPI(fail_first=True),
                ),
            ),
        )

        audio_bytes = backend.synthesize("Hello from Twinr")

        self.assertEqual(audio_bytes, b"AUDIO")
        self.assertEqual(backend._client.audio.speech.calls[0]["model"], "gpt-4o-mini-tts")
        self.assertEqual(backend._client.audio.speech.calls[1]["model"], "tts-1-hd")
        self.assertEqual(backend._client.audio.speech.calls[1]["voice"], "sage")
        self.assertEqual(backend._client.audio.speech.calls[1]["speed"], 0.9)

    def test_synthesize_stream_yields_audio_chunks(self) -> None:
        chunks = list(self.backend.synthesize_stream("Hello from Twinr"))

        self.assertEqual(chunks, [b"AU", b"DIO"])
        self.assertEqual(self.speech.calls[0]["model"], "gpt-4o-mini-tts")
        self.assertEqual(self.speech.calls[0]["speed"], 0.9)
        self.assertEqual(self.speech.calls[0]["instructions"], "Speak in natural German.")

    def test_synthesize_stream_close_stops_remaining_chunks_and_closes_response(self) -> None:
        stream = self.backend.synthesize_stream("Hello from Twinr")

        first_chunk = next(stream)
        stream.close()
        remaining_chunks = list(stream)

        self.assertEqual(first_chunk, b"AU")
        self.assertEqual(remaining_chunks, [])
        self.assertTrue(self.speech.streaming_responses)
        self.assertTrue(self.speech.streaming_responses[0].closed)

    def test_phrase_due_reminder_builds_short_reminder_request(self) -> None:
        from twinr.memory.reminders import ReminderEntry, now_in_timezone

        due_at = now_in_timezone("Europe/Berlin")
        reminder = ReminderEntry(
            reminder_id="REM-1",
            kind="appointment",
            summary="Arzttermin",
            details="Bei Dr. Meyer",
            due_at=due_at,
            created_at=due_at,
            updated_at=due_at,
        )

        response = self.backend.phrase_due_reminder_with_metadata(reminder)

        self.assertEqual(response.text, "Backend answer")
        request = self.responses.calls[-1]
        self.assertEqual(request["model"], "gpt-5.2")
        self.assertEqual(request["reasoning"], {"effort": "low"})
        self.assertIn("A stored Twinr reminder is due now.", request["input"][0]["content"][0]["text"])
        self.assertIn('Reminder summary: "Arzttermin"', request["input"][0]["content"][0]["text"])
        self.assertIn('Reminder details: "Bei Dr. Meyer"', request["input"][0]["content"][0]["text"])
        self.assertIn("speaking a due reminder", request["instructions"])
        self.assertIn("All user-facing spoken and written replies for this turn must be in German.", request["instructions"])

    def test_phrase_proactive_prompt_uses_recent_context_and_repeat_guard(self) -> None:
        response = self.backend.phrase_proactive_prompt_with_metadata(
            trigger_id="showing_intent",
            reason="Person looked toward the device while holding an object near the camera.",
            default_prompt="Möchtest du mir etwas zeigen?",
            priority=30,
            conversation=[("user", "Ich suche meinen Ausweis."), ("assistant", "Ich helfe dir gern.")],
            recent_prompts=("Möchtest du mir etwas zeigen?", "Willst du mir das kurz zeigen?"),
            observation_facts=(
                "showing_hold: value=1.00, weight=0.55, detail=active_for=2.4s target=1.5s",
                "looking_toward_device: value=1.00, weight=0.20, detail=looking_toward_device=True",
            ),
        )

        self.assertEqual(response.text, "Backend answer")
        request = self.responses.calls[-1]
        self.assertEqual(request["model"], "gpt-5.2")
        self.assertIn("short proactive sentence", request["instructions"])
        self.assertIn('Trigger id: "showing_intent"', request["input"][-1]["content"][0]["text"])
        self.assertIn(
            "Recent proactive wording to avoid repeating too closely",
            request["input"][-1]["content"][0]["text"],
        )
        self.assertIn("Observed evidence:", request["input"][-1]["content"][0]["text"])
        self.assertIn("showing_hold: value=1.00", request["input"][-1]["content"][0]["text"])
        self.assertEqual(request["input"][0]["role"], "user")
        self.assertEqual(request["input"][1]["role"], "assistant")

    def test_respond_streaming_emits_text_deltas_and_returns_metadata(self) -> None:
        deltas: list[str] = []

        response = self.backend.respond_streaming(
            "Say hello",
            allow_web_search=False,
            on_text_delta=deltas.append,
        )

        self.assertEqual(deltas, ["Hello", " there"])
        self.assertEqual(response.text, "Hello there")
        self.assertEqual(response.response_id, "resp_stream")
        self.assertEqual(response.request_id, "req_stream")
        self.assertFalse(response.used_web_search)
        self.assertEqual(self.responses.stream_calls[0]["model"], "gpt-5.2")
        self.assertTrue(self.responses.stream_calls[0]["prompt_cache_key"].startswith("twinr:response_stream:"))
        self.assertLessEqual(len(self.responses.stream_calls[0]["prompt_cache_key"]), 64)

    def test_search_live_info_uses_web_search_and_extracts_sources(self) -> None:
        self.responses.output_text = "Morgen in Schwarzenbek 11 Grad, leichter Regen."
        self.responses.output = [
            SimpleNamespace(
                type="web_search_call",
                action=SimpleNamespace(
                    sources=[
                        SimpleNamespace(url="https://weather.example/forecast"),
                        SimpleNamespace(url="https://weather.example/forecast"),
                        SimpleNamespace(url="https://stadt.example/wetter"),
                    ]
                ),
            )
        ]

        backend = OpenAIBackend(
            config=replace(self.config, openai_search_model="gpt-5.2-chat-latest"),
            client=self.client,
        )

        result = backend.search_live_info_with_metadata(
            "Wie wird das Wetter morgen?",
            location_hint="Schwarzenbek",
            date_context="Friday, 2026-03-13 10:00 (Europe/Berlin)",
        )

        self.assertEqual(result.answer, "Morgen in Schwarzenbek 11 Grad, leichter Regen.")
        self.assertEqual(result.verification_status, "verified")
        self.assertTrue(result.question_resolved)
        self.assertFalse(result.site_follow_up_recommended)
        self.assertEqual(
            result.sources,
            (
                "https://weather.example/forecast",
                "https://stadt.example/wetter",
            ),
        )
        request = self.responses.calls[0]
        self.assertEqual(request["model"], "gpt-5.2-chat-latest")
        self.assertIn("web_search_call.action.sources", request["include"])
        self.assertEqual(request["tools"][0]["type"], "web_search")
        self.assertEqual(request["tools"][0]["search_context_size"], "medium")
        self.assertTrue(request["prompt_cache_key"].startswith("twinr:search:"))
        self.assertLessEqual(len(request["prompt_cache_key"]), 64)
        self.assertEqual(request["reasoning"], {"effort": "low"})
        self.assertEqual(request["max_output_tokens"], 1024)
        prompt = request["input"][-1]["content"][0]["text"]
        self.assertIn("User question: Wie wird das Wetter morgen?", prompt)
        self.assertIn("Resolved explicit-date variant: Wie wird das Wetter am 2026-03-13?", prompt)
        self.assertIn("Explicit place context: Schwarzenbek", prompt)
        self.assertIn("Explicit local date/time context: Friday, 2026-03-13 10:00 (Europe/Berlin)", prompt)
        self.assertEqual(request["tools"][0]["user_location"]["city"], "Schwarzenbek")
        self.assertIn(
            "authoritative disambiguation for partial wording",
            request["instructions"],
        )
        self.assertIn(
            "Avoid stiff, bureaucratic, or institutional wording.",
            self.responses.calls[-1]["instructions"],
        )
        self.assertEqual(self.responses.calls[-1]["text"]["format"]["name"], "twinr_live_search_spoken_answer")
        self.assertNotIn("tools", self.responses.calls[-1])

    def test_search_live_info_uses_bounded_search_conversation_and_explicit_context_prompt(self) -> None:
        self.responses.output_text = "Morgen in Schwarzenbek 11 Grad, leichter Regen."

        backend = OpenAIBackend(
            config=replace(self.config, openai_search_model="gpt-5.2-chat-latest"),
            client=SimpleNamespace(
                responses=self.responses,
                audio=SimpleNamespace(
                    transcriptions=self.transcriptions,
                    speech=self.speech,
                ),
            ),
            base_instructions="THIS SHOULD NOT REACH THE SEARCH HELPER",
        )

        backend.search_live_info_with_metadata(
            "Und morgen?",
            conversation=(
                ("system", "Twinr memory summary:\n- very long system context " * 20),
                ("user", "Wie ist das Wetter heute in Schwarzenbek?"),
                ("assistant", "Heute 8 Grad und bewölkt."),
                ("system", "Long-term system block " * 20),
                ("user", "Und morgen?"),
                ("assistant", "Ich prüfe das live."),
                ("user", "Bitte mit Temperatur und Regen."),
            ),
            location_hint="Schwarzenbek",
        )

        request = self.responses.calls[0]
        self.assertNotIn("THIS SHOULD NOT REACH THE SEARCH HELPER", request["instructions"])
        self.assertEqual([item["role"] for item in request["input"]], ["user", "assistant", "user", "user"])
        self.assertEqual(request["input"][0]["content"][0]["text"], "Und morgen?")
        self.assertEqual(request["input"][1]["content"][0]["text"], "Ich prüfe das live.")
        self.assertEqual(request["input"][2]["content"][0]["text"], "Bitte mit Temperatur und Regen.")
        prompt = request["input"][-1]["content"][0]["text"]
        self.assertIn("User question: Und morgen?", prompt)
        self.assertIn("Explicit place context: Schwarzenbek", prompt)
        self.assertNotIn("very long system context", str(request["input"]))
        self.assertNotIn("Long-term system block", str(request["input"]))

    def test_search_live_info_does_not_inject_default_city_when_location_hint_is_missing(self) -> None:
        self.responses.output_text = "In Schwarzenbek 9 Grad, trocken."

        backend = OpenAIBackend(
            config=replace(self.config, openai_search_model="gpt-5.2-chat-latest"),
            client=self.client,
        )

        backend.search_live_info_with_metadata(
            "Wie ist das Wetter in Schwarzenbek?",
        )

        request = self.responses.calls[0]
        prompt = request["input"][-1]["content"][0]["text"]
        self.assertEqual(prompt, "Wie ist das Wetter in Schwarzenbek?")
        self.assertNotIn("user_location", request["tools"][0])

    def test_search_live_info_passes_news_query_literally(self) -> None:
        self.responses.output_text = "Heute dominieren zwei Schlagzeilen die Nachrichtenlage."

        backend = OpenAIBackend(
            config=replace(self.config, openai_search_model="gpt-5.2-chat-latest"),
            client=self.client,
        )

        backend.search_live_info_with_metadata("Was sind die neuesten Nachrichten?")

        request = self.responses.calls[0]
        prompt = request["input"][-1]["content"][0]["text"]
        self.assertEqual(prompt, "Was sind die neuesten Nachrichten?")
        self.assertNotIn("user_location", request["tools"][0])
        self.assertIn("Interpret the user's request semantically", request["instructions"])
        self.assertIn("prefer major national or international headlines", request["instructions"])

    def test_supervisor_decision_schema_requires_every_property_key(self) -> None:
        self.assertEqual(
            set(_SUPERVISOR_DECISION_SCHEMA["required"]),
            set(_SUPERVISOR_DECISION_SCHEMA["properties"]),
        )

    def test_search_live_info_falls_back_to_main_model_when_primary_search_model_returns_blank(self) -> None:
        class BlankThenAnswerResponses:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                model = kwargs["model"]
                response_format = ((kwargs.get("text") or {}).get("format") or {})
                wrap_structured = response_format.get("name") == "twinr_live_search_spoken_answer"
                if model == "gpt-5.2":
                    return SimpleNamespace(
                        id="resp_blank",
                        _request_id="req_blank",
                        output_text="",
                        output=[SimpleNamespace(type="web_search_call")],
                    )
                output_text = "Morgen in Schwarzenbek bis 11 Grad und zeitweise Regen."
                if wrap_structured:
                    output_text = _search_voice_payload(output_text)
                return SimpleNamespace(
                    id="resp_chat",
                    _request_id="req_chat",
                    output_text=output_text,
                    output=[
                        SimpleNamespace(type="web_search_call"),
                        SimpleNamespace(
                            type="message",
                            status="completed",
                            content=[SimpleNamespace(type="output_text", text="Morgen in Schwarzenbek bis 11 Grad und zeitweise Regen.")],
                        ),
                    ],
                )

        backend = OpenAIBackend(
            config=replace(self.config, default_model="gpt-5.4-mini", openai_search_model="gpt-5.2"),
            client=SimpleNamespace(
                responses=BlankThenAnswerResponses(),
                audio=SimpleNamespace(
                    transcriptions=self.transcriptions,
                    speech=self.speech,
                ),
            ),
        )

        result = backend.search_live_info_with_metadata("Wie wird das Wetter morgen?")

        self.assertEqual(result.answer, "Morgen in Schwarzenbek bis 11 Grad und zeitweise Regen.")
        self.assertEqual(
            [call["model"] for call in backend._client.responses.calls],
            ["gpt-5.2", "gpt-5.2", "gpt-5.4-mini", "gpt-5.4-mini"],
        )
        self.assertEqual([call["max_output_tokens"] for call in backend._client.responses.calls[:3]], [1024, 1536, 1024])
        self.assertEqual(result.requested_model, "gpt-5.2")
        self.assertEqual(result.model, "gpt-5.4-mini")
        self.assertIn("gpt-5.2->gpt-5.4-mini", result.fallback_reason or "")
        self.assertEqual(tuple(attempt.outcome for attempt in result.attempt_log), ("unusable", "unusable", "success"))
        self.assertEqual(result.verification_status, "verified")
        self.assertTrue(result.question_resolved)

    def test_search_live_info_retries_blank_completed_same_model_before_fallback(self) -> None:
        class BlankCompletedThenAnswerResponses:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                response_format = ((kwargs.get("text") or {}).get("format") or {})
                if response_format.get("name") == "twinr_live_search_spoken_answer":
                    return SimpleNamespace(
                        id="resp_voice",
                        _request_id="req_voice",
                        model="gpt-5.4-mini-2026-03-17",
                        usage=_fake_usage(),
                        output_text=_search_voice_payload(
                            "Agentische KI wird gerade stärker in echte Arbeitsabläufe eingebaut."
                        ),
                        output=[
                            SimpleNamespace(
                                type="message",
                                status="completed",
                                content=[
                                    SimpleNamespace(
                                        type="output_text",
                                        text="Agentische KI wird gerade stärker in echte Arbeitsabläufe eingebaut.",
                                    )
                                ],
                            )
                        ],
                    )

                budget = kwargs["max_output_tokens"]
                if budget < 1536:
                    return SimpleNamespace(
                        id=f"resp_blank_{budget}",
                        _request_id=f"req_blank_{budget}",
                        model="gpt-5.4-mini-2026-03-17",
                        usage=_fake_usage(),
                        status="completed",
                        output_text="",
                        output=[SimpleNamespace(type="web_search_call")],
                    )

                return SimpleNamespace(
                    id="resp_complete",
                    _request_id="req_complete",
                    model="gpt-5.4-mini-2026-03-17",
                    usage=_fake_usage(),
                    status="completed",
                    output_text="Agentische KI wird gerade stärker in echte Arbeitsabläufe eingebaut.",
                    output=[
                        SimpleNamespace(type="web_search_call"),
                        SimpleNamespace(
                            type="message",
                            status="completed",
                            content=[
                                SimpleNamespace(
                                    type="output_text",
                                    text="Agentische KI wird gerade stärker in echte Arbeitsabläufe eingebaut.",
                                )
                            ],
                        ),
                    ],
                )

        backend = OpenAIBackend(
            config=replace(
                self.config,
                openai_search_model="gpt-5.4-mini",
                openai_search_max_output_tokens=160,
                openai_search_retry_max_output_tokens=240,
            ),
            client=SimpleNamespace(
                responses=BlankCompletedThenAnswerResponses(),
                audio=SimpleNamespace(
                    transcriptions=self.transcriptions,
                    speech=self.speech,
                ),
            ),
        )

        result = backend.search_live_info_with_metadata("Was gibt es Neues zu KI-Begleitern und agentischer KI?")

        self.assertEqual(
            result.answer,
            "Agentische KI wird gerade stärker in echte Arbeitsabläufe eingebaut.",
        )
        self.assertEqual(result.requested_model, "gpt-5.4-mini")
        self.assertEqual(result.model, "gpt-5.4-mini-2026-03-17")
        self.assertIsNone(result.fallback_reason)
        self.assertEqual(
            [call["max_output_tokens"] for call in backend._client.responses.calls[:-1]],
            [160, 240, 512, 768, 1024, 1536],
        )
        self.assertEqual(backend._client.responses.calls[-1]["max_output_tokens"], 160)
        self.assertEqual(
            tuple(attempt.outcome for attempt in result.attempt_log),
            ("retry", "retry", "retry", "retry", "retry", "success"),
        )
        self.assertTrue(all(attempt.model == "gpt-5.4-mini" for attempt in result.attempt_log))
        self.assertEqual(result.verification_status, "verified")
        self.assertTrue(result.question_resolved)

    def test_search_live_info_retries_incomplete_max_output_tokens_until_completed(self) -> None:
        class IncompleteThenCompletedResponses:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                response_format = ((kwargs.get("text") or {}).get("format") or {})
                if response_format.get("name") == "twinr_live_search_spoken_answer":
                    return SimpleNamespace(
                        id="resp_voice",
                        _request_id="req_voice",
                        model=kwargs["model"],
                        usage=_fake_usage(),
                        output_text=_search_voice_payload(
                            "In Hamburg sind gerade Schulbau und Verkehr besonders umkämpft."
                        ),
                        output=[
                            SimpleNamespace(
                                type="message",
                                status="completed",
                                content=[
                                    SimpleNamespace(
                                        type="output_text",
                                        text="In Hamburg sind gerade Schulbau und Verkehr besonders umkämpft.",
                                    )
                                ],
                            )
                        ],
                    )

                budget = kwargs["max_output_tokens"]
                base_output = [
                    SimpleNamespace(
                        type="web_search_call",
                        action=SimpleNamespace(
                            sources=[SimpleNamespace(url="https://hamburg.example/politik")]
                        ),
                    )
                ]
                if budget < 2048:
                    partial_output = "" if budget < 1024 else "Hamburg ringt gerade um Schulbau."
                    incomplete_message = (
                        []
                        if budget < 1024
                        else [
                            SimpleNamespace(
                                type="message",
                                status="incomplete",
                                content=[
                                    SimpleNamespace(
                                        type="output_text",
                                        text="Hamburg ringt gerade um Schulbau.",
                                    )
                                ],
                            )
                        ]
                    )
                    return SimpleNamespace(
                        id=f"resp_{budget}",
                        _request_id=f"req_{budget}",
                        model=kwargs["model"],
                        usage=_fake_usage(),
                        status="incomplete",
                        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                        output_text=partial_output,
                        output=base_output + incomplete_message,
                    )

                return SimpleNamespace(
                    id="resp_complete",
                    _request_id="req_complete",
                    model=kwargs["model"],
                    usage=_fake_usage(),
                    status="completed",
                    output_text="In Hamburg sind gerade Schulbau und Verkehr besonders umkämpft.",
                    output=base_output
                    + [
                        SimpleNamespace(
                            type="message",
                            status="completed",
                            content=[
                                SimpleNamespace(
                                    type="output_text",
                                    text="In Hamburg sind gerade Schulbau und Verkehr besonders umkämpft.",
                                )
                            ],
                        )
                    ],
                )

        backend = OpenAIBackend(
            config=replace(self.config, openai_search_model="gpt-5.4-mini"),
            client=SimpleNamespace(
                responses=IncompleteThenCompletedResponses(),
                audio=SimpleNamespace(
                    transcriptions=self.transcriptions,
                    speech=self.speech,
                ),
            ),
        )

        result = backend.search_live_info_with_metadata("Was ist in der Hamburger Lokalpolitik spannend?")

        self.assertEqual(result.answer, "In Hamburg sind gerade Schulbau und Verkehr besonders umkämpft.")
        self.assertEqual(result.sources, ("https://hamburg.example/politik",))
        self.assertEqual(result.verification_status, "verified")
        self.assertTrue(result.question_resolved)
        self.assertEqual(
            [call["max_output_tokens"] for call in backend._client.responses.calls[:-1]],
            [1024, 1536, 2048],
        )
        self.assertEqual(backend._client.responses.calls[-1]["max_output_tokens"], 160)

    def test_search_live_info_returns_structured_follow_up_metadata_from_voice_rewrite(self) -> None:
        class SearchThenStructuredRewriteResponses:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                response_format = ((kwargs.get("text") or {}).get("format") or {})
                if response_format.get("name") == "twinr_live_search_spoken_answer":
                    return SimpleNamespace(
                        id="resp_voice",
                        _request_id="req_voice",
                        model=kwargs["model"],
                        usage=_fake_usage(),
                        output_text=_search_voice_payload(
                            "Ich konnte online keinen aktuellen Mittagsplan eindeutig bestätigen.",
                            verification_status="partial",
                            question_resolved=False,
                            site_follow_up_recommended=True,
                            site_follow_up_reason=(
                                "Die offizielle Café-Seite könnte den aktuellen Tagesplan nur auf der Website selbst zeigen."
                            ),
                            site_follow_up_url="https://cafeluise.example/mittag",
                            site_follow_up_domain="cafeluise.example",
                        ),
                        output=[
                            SimpleNamespace(
                                type="message",
                                status="completed",
                                content=[
                                    SimpleNamespace(
                                        type="output_text",
                                        text="Ich konnte online keinen aktuellen Mittagsplan eindeutig bestätigen.",
                                    )
                                ],
                            )
                        ],
                    )
                return SimpleNamespace(
                    id="resp_search",
                    _request_id="req_search",
                    model=kwargs["model"],
                    usage=_fake_usage(),
                    status="completed",
                    output_text="Auf der offiziellen Seite ist kein aktuelles Mittagsmenü im Suchergebnis sichtbar.",
                    output=[
                        SimpleNamespace(
                            type="web_search_call",
                            action=SimpleNamespace(
                                sources=[SimpleNamespace(url="https://cafeluise.example/mittag")]
                            ),
                        ),
                        SimpleNamespace(
                            type="message",
                            status="completed",
                            content=[
                                SimpleNamespace(
                                    type="output_text",
                                    text="Auf der offiziellen Seite ist kein aktuelles Mittagsmenü im Suchergebnis sichtbar.",
                                )
                            ],
                        ),
                    ],
                )

        backend = OpenAIBackend(
            config=replace(self.config, openai_search_model="gpt-5.4-mini"),
            client=SimpleNamespace(
                responses=SearchThenStructuredRewriteResponses(),
                audio=SimpleNamespace(
                    transcriptions=self.transcriptions,
                    speech=self.speech,
                ),
            ),
        )

        result = backend.search_live_info_with_metadata("Hat das Café Luise heute online ein Mittagsmenü veröffentlicht?")

        self.assertEqual(result.answer, "Ich konnte online keinen aktuellen Mittagsplan eindeutig bestätigen.")
        self.assertEqual(result.verification_status, "partial")
        self.assertFalse(result.question_resolved)
        self.assertTrue(result.site_follow_up_recommended)
        self.assertEqual(
            result.site_follow_up_reason,
            "Die offizielle Café-Seite könnte den aktuellen Tagesplan nur auf der Website selbst zeigen.",
        )
        self.assertEqual(result.site_follow_up_url, "https://cafeluise.example/mittag")
        self.assertEqual(result.site_follow_up_domain, "cafeluise.example")

    def test_search_live_info_raises_budget_error_when_incomplete_retry_ladder_is_exhausted(self) -> None:
        class AlwaysIncompleteResponses:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    id=f"resp_{kwargs['max_output_tokens']}",
                    _request_id=f"req_{kwargs['max_output_tokens']}",
                    model=kwargs["model"],
                    usage=_fake_usage(),
                    status="incomplete",
                    incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                    output_text="",
                    output=[SimpleNamespace(type="web_search_call")],
                )

        backend = OpenAIBackend(
            config=replace(self.config, openai_search_model="gpt-5.4-mini"),
            client=SimpleNamespace(
                responses=AlwaysIncompleteResponses(),
                audio=SimpleNamespace(
                    transcriptions=self.transcriptions,
                    speech=self.speech,
                ),
            ),
        )

        with self.assertRaises(RuntimeError) as ctx:
            backend.search_live_info_with_metadata("Was gibt es Neues?")

        self.assertIn("max_output_tokens=3072", str(ctx.exception))
        self.assertIn("incomplete=max_output_tokens", str(ctx.exception))
        attempts_by_model: dict[str, list[int]] = {}
        for call in backend._client.responses.calls:
            attempts_by_model.setdefault(call["model"], []).append(call["max_output_tokens"])
        self.assertEqual(
            tuple(attempts_by_model),
            backend._candidate_search_models(),
        )
        for budgets in attempts_by_model.values():
            self.assertEqual(budgets, [1024, 1536, 2048, 3072])

    def test_search_live_info_uses_search_preview_chat_path(self) -> None:
        backend = OpenAIBackend(
            config=replace(self.config, openai_search_model="gpt-4o-mini-search-preview"),
            client=self.client,
        )
        self.responses.output_text = self.client.chat.completions.content

        result = backend.search_live_info_with_metadata(
            "Wie wird das Wetter morgen?",
            location_hint="Schwarzenbek",
            date_context="Monday, 2026-03-16 08:00 (Europe/Berlin)",
        )

        self.assertEqual(result.answer, "Morgen in Schwarzenbek 8 Grad, leichter Regen.")
        self.assertEqual(result.sources, ("https://weather.example/forecast",))
        self.assertTrue(result.used_web_search)
        self.assertEqual(result.model, "gpt-4o-mini-search-preview")
        self.assertEqual(result.token_usage.total_tokens, 76)
        self.assertEqual(self.client.chat.completions.calls[0]["model"], "gpt-4o-mini-search-preview")
        self.assertEqual(
            self.client.chat.completions.calls[0]["web_search_options"],
            {
                "search_context_size": "medium",
                "user_location": {
                    "type": "approximate",
                    "approximate": {
                        "city": "Schwarzenbek",
                        "country": "DE",
                        "timezone": "Europe/Berlin",
                    },
                },
            },
        )
        self.assertEqual(self.client.chat.completions.calls[0]["messages"][0]["role"], "system")
        prompt = self.client.chat.completions.calls[0]["messages"][-1]["content"]
        self.assertIn("User question: Wie wird das Wetter morgen?", prompt)
        self.assertIn("Resolved explicit-date variant: Wie wird das Wetter am 2026-03-16?", prompt)
        self.assertIn("Explicit place context: Schwarzenbek", prompt)
        self.assertEqual(len(self.responses.calls), 1)
        self.assertEqual(self.responses.calls[0]["text"]["format"]["name"], "twinr_live_search_spoken_answer")

    def test_search_live_info_preview_path_passes_literal_explicit_date_question(self) -> None:
        backend = OpenAIBackend(
            config=replace(self.config, openai_search_model="gpt-4o-mini-search-preview"),
            client=self.client,
        )

        backend.search_live_info_with_metadata(
            "Wettervorhersage für Schwarzenbek am 16. März 2026",
            location_hint="Schwarzenbek",
            date_context="Monday, 2026-03-16 (Europe/Berlin)",
        )

        prompt = self.client.chat.completions.calls[-1]["messages"][-1]["content"]
        self.assertIn("User question: Wettervorhersage für Schwarzenbek am 16. März 2026", prompt)
        self.assertIn("Explicit place context: Schwarzenbek", prompt)
        self.assertIn("Explicit local date/time context: Monday, 2026-03-16 (Europe/Berlin)", prompt)

    def test_compose_print_job_uses_context_and_request_source(self) -> None:
        self.responses.output_text = "TERMINE\nMontag 14 Uhr\nAdresse Praxis"
        response = self.backend.compose_print_job_with_metadata(
            conversation=(("user", "Wann ist der Termin?"), ("assistant", "Montag 14 Uhr.")),
            focus_hint="appointment details",
            direct_text="Montag 14 Uhr bei Dr. Meyer",
            request_source="button",
        )

        request = self.responses.calls[0]
        self.assertEqual(response.text, "TERMINE\nMontag 14 Uhr\nAdresse Praxis")
        self.assertEqual(request["reasoning"], {"effort": "low"})
        self.assertEqual(request["max_output_tokens"], 320)
        self.assertNotIn("tools", request)
        self.assertIn("Request source: button", request["input"][-1]["content"][0]["text"])
        self.assertIn("Focus hint: appointment details", request["input"][-1]["content"][0]["text"])
        self.assertIn("Latest user message: Wann ist der Termin?", request["input"][-1]["content"][0]["text"])
        self.assertIn("Latest assistant message: Montag 14 Uhr.", request["input"][-1]["content"][0]["text"])

    def test_compose_print_job_falls_back_to_plain_request_after_structured_internal_server_error(self) -> None:
        self.responses.queued_exceptions = [InternalServerError("provider 500")]
        self.responses.output_text = "TERMINE\nMontag 14 Uhr\nAdresse Praxis"

        response = self.backend.compose_print_job_with_metadata(
            direct_text="Montag 14 Uhr bei Dr. Meyer",
            request_source="button",
        )

        self.assertEqual(response.text, "TERMINE\nMontag 14 Uhr\nAdresse Praxis")
        self.assertEqual(len(self.responses.calls), 2)
        self.assertIn("format", self.responses.calls[0]["text"])
        self.assertNotIn("format", self.responses.calls[1]["text"])

    def test_compose_print_job_rejects_incomplete_truncated_structured_output_and_uses_fallback(self) -> None:
        self.responses.queued_payloads = [
            {
                "status": "incomplete",
                "incomplete_details": SimpleNamespace(reason="max_output_tokens"),
                "output_text": '{"status":"ready","text":"Morgen-Digest | 03.04.2026\\nWetter Berlin',
            }
        ]

        response = self.backend.compose_print_job_with_metadata(
            direct_text="Morgen-Digest | 03.04.2026\nWetter Berlin: ruhig und kuehl",
            request_source="button",
        )

        self.assertEqual(
            response.text,
            "Morgen-Digest | 03.04.2026\nWetter Berlin: ruhig und kuehl",
        )

    def test_compose_print_job_uses_literal_tool_text_without_llm_rewrite(self) -> None:
        response = self.backend.compose_print_job_with_metadata(
            conversation=(
                ("user", "Wie ist das Wetter morgen in Schwarzenbek?"),
                ("assistant", "Morgen Regen und kühl."),
            ),
            focus_hint="appointment details",
            direct_text="Zahnarzt Montag 14 Uhr.",
            request_source="tool",
        )

        self.assertEqual(response.text, "Zahnarzt Montag 14 Uhr.")
        self.assertEqual(self.responses.calls, [])

    def test_compose_print_job_enforces_output_limits(self) -> None:
        limited_backend = OpenAIBackend(
            config=replace(self.config, print_max_lines=2, print_max_chars=18),
            client=self.client,
        )
        self.responses.output_text = "Zeile eins\nZeile zwei\nZeile drei"

        response = limited_backend.compose_print_job_with_metadata(
            direct_text="Too much text",
            request_source="button",
        )

        self.assertEqual(response.text, "Zeile eins\nZeile…")

    def test_compose_print_job_falls_back_when_summary_is_too_short(self) -> None:
        verbose_source = (
            "Der Termin ist am Montag um 14 Uhr bei Dr. Meyer in Hamburg. "
            "Bitte bring deine Versicherungskarte und den Medikationsplan mit."
        )
        self.responses.queued_output_texts = ["Montag 14 Uhr", "Montag 14 Uhr\nDr. Meyer, Hamburg"]

        response = self.backend.compose_print_job_with_metadata(
            direct_text=verbose_source,
            request_source="button",
        )

        self.assertEqual(response.text, "Montag 14 Uhr")
        self.assertEqual(len(self.responses.calls), 1)

    def test_compose_print_job_falls_back_when_composer_returns_empty_output(self) -> None:
        self.responses.queued_output_texts = ["", "Zahnarzt Montag 14 Uhr"]

        response = self.backend.compose_print_job_with_metadata(
            direct_text="Zahnarzt Montag 14 Uhr.",
            request_source="tool",
        )

        self.assertEqual(response.text, "Zahnarzt Montag 14 Uhr.")
        self.assertEqual(len(self.responses.calls), 0)

    def test_compose_print_job_falls_back_when_composer_returns_no_print_content(self) -> None:
        self.responses.queued_output_texts = ["NO_PRINT_CONTENT", "Zahnarzt Montag 14 Uhr"]

        response = self.backend.compose_print_job_with_metadata(
            direct_text="Zahnarzt Montag 14 Uhr.",
            request_source="tool",
        )

        self.assertEqual(response.text, "Zahnarzt Montag 14 Uhr.")
        self.assertEqual(len(self.responses.calls), 0)

    def test_default_client_factory_skips_project_header_for_project_scoped_key(self) -> None:
        captured_kwargs: dict[str, object] = {}

        class FakeOpenAI:
            def __init__(self, **kwargs) -> None:
                captured_kwargs.update(kwargs)

        original_module = sys.modules.get("openai")
        fake_openai_module = ModuleType("openai")
        setattr(fake_openai_module, "OpenAI", FakeOpenAI)
        close_cached_openai_clients()
        sys.modules["openai"] = fake_openai_module
        try:
            _default_client_factory(
                TwinrConfig(
                    openai_api_key="sk-proj-example",
                    openai_project_id="proj_123",
                )
            )
        finally:
            close_cached_openai_clients()
            if original_module is None:
                sys.modules.pop("openai", None)
            else:
                sys.modules["openai"] = original_module

        self.assertEqual(captured_kwargs["api_key"], "sk-proj-example")
        self.assertEqual(captured_kwargs["base_url"], "https://api.openai.com/v1")
        self.assertEqual(captured_kwargs["max_retries"], 1)
        self.assertNotIn("project", captured_kwargs)
        self.assertIn("timeout", captured_kwargs)
        self.assertIn("http_client", captured_kwargs)

    def test_format_for_print_uses_low_reasoning_without_web_search(self) -> None:
        self.backend.format_for_print("This is a longer answer that should be compressed.")

        request = self.responses.calls[0]
        self.assertEqual(request["reasoning"], {"effort": "low"})
        self.assertNotIn("tools", request)
        self.assertEqual(request["max_output_tokens"], 140)
        self.assertIn("All user-facing spoken and written replies for this turn must be in German.", request["instructions"])
