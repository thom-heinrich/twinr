from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.providers.openai.backend import OpenAIBackend, OpenAIImageInput, _default_client_factory


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

    def read(self) -> bytes:
        return self._payload

    def iter_bytes(self, _chunk_size: int | None = None):
        midpoint = max(1, len(self._payload) // 2)
        yield self._payload[:midpoint]
        yield self._payload[midpoint:]


class FakeResponsesAPI:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.stream_calls: list[dict] = []
        self.output_text = "Backend answer"
        self.output = [SimpleNamespace(type="web_search_call")]
        self.queued_output_texts: list[str] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        output_text = self.queued_output_texts.pop(0) if self.queued_output_texts else self.output_text
        return SimpleNamespace(
            id="resp_123",
            _request_id="req_123",
            model=kwargs["model"],
            usage=_fake_usage(),
            output_text=output_text,
            output=self.output,
        )

    def stream(self, **kwargs):
        self.stream_calls.append(kwargs)

        class _StreamManager:
            def __enter__(self_nonlocal):
                return self_nonlocal

            def __exit__(self_nonlocal, exc_type, exc, tb):
                return None

            def __iter__(self_nonlocal):
                yield SimpleNamespace(type="response.output_text.delta", delta="Hello")
                yield SimpleNamespace(type="response.output_text.delta", delta=" there")

            def get_final_response(self_nonlocal):
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


class FakeModelAccessError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.status_code = 403
        self.body = {"error": {"code": "model_not_found"}}


class FakeSpeechAPI:
    def __init__(self, fail_first: bool = False) -> None:
        self.calls: list[dict] = []
        self.fail_first = fail_first

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
                    def __enter__(self_nonlocal):
                        return FakeBinaryResponse(b"AUDIO")

                    def __exit__(self_nonlocal, exc_type, exc, tb):
                        return None

                return _Manager()

        return _StreamingWrapper()


class OpenAIBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.responses = FakeResponsesAPI()
        self.transcriptions = FakeTranscriptionsAPI()
        self.speech = FakeSpeechAPI()
        self.client = SimpleNamespace(
            responses=self.responses,
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
        self.assertEqual(request["instructions"], "Base context\n\nTask context")

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
        self.assertIn("SYSTEM:\nSystem context", request["instructions"])
        self.assertIn("PERSONALITY:\nUpdated style context", request["instructions"])
        self.assertIn("MEMORY:\nDurable remembered items explicitly saved for future turns:", request["instructions"])
        self.assertIn("REMINDERS:\nScheduled reminders and timers:", request["instructions"])
        self.assertIn("04151 8810", request["instructions"])
        self.assertIn("Muell rausstellen", request["instructions"])

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
        self.assertEqual(request["response_format"], "wav")
        self.assertEqual(request["instructions"], "Speak in natural German.")

    def test_synthesize_falls_back_to_tts_1_when_project_lacks_model_access(self) -> None:
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
        self.assertEqual(backend._client.audio.speech.calls[1]["model"], "tts-1")
        self.assertEqual(backend._client.audio.speech.calls[1]["voice"], "sage")

    def test_synthesize_stream_yields_audio_chunks(self) -> None:
        chunks = list(self.backend.synthesize_stream("Hello from Twinr"))

        self.assertEqual(chunks, [b"AU", b"DIO"])
        self.assertEqual(self.speech.calls[0]["model"], "gpt-4o-mini-tts")
        self.assertEqual(self.speech.calls[0]["instructions"], "Speak in natural German.")

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
        self.assertIn("Reminder summary: Arzttermin", request["input"][0]["content"][0]["text"])
        self.assertIn("Reminder details: Bei Dr. Meyer", request["input"][0]["content"][0]["text"])
        self.assertIn("speaking a due reminder", request["instructions"])

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

        result = self.backend.search_live_info_with_metadata(
            "Wie wird das Wetter morgen?",
            location_hint="Schwarzenbek",
            date_context="Friday, 2026-03-13 10:00 (Europe/Berlin)",
        )

        self.assertEqual(result.answer, "Morgen in Schwarzenbek 11 Grad, leichter Regen.")
        self.assertEqual(
            result.sources,
            (
                "https://weather.example/forecast",
                "https://stadt.example/wetter",
            ),
        )
        request = self.responses.calls[0]
        self.assertEqual(request["model"], "gpt-5.2-chat-latest")
        self.assertEqual(request["include"], ["web_search_call.action.sources"])
        self.assertEqual(request["tools"][0]["type"], "web_search")
        self.assertIn("Location hint: Schwarzenbek", request["input"][-1]["content"][0]["text"])
        self.assertIn("Local date/time context: Friday, 2026-03-13 10:00", request["input"][-1]["content"][0]["text"])

    def test_search_live_info_falls_back_to_chat_latest_when_primary_search_model_returns_blank(self) -> None:
        class BlankThenAnswerResponses:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                model = kwargs["model"]
                if model == "gpt-5.2":
                    return SimpleNamespace(
                        id="resp_blank",
                        _request_id="req_blank",
                        output_text="",
                        output=[SimpleNamespace(type="web_search_call")],
                    )
                return SimpleNamespace(
                    id="resp_chat",
                    _request_id="req_chat",
                    output_text="Morgen in Schwarzenbek bis 11 Grad und zeitweise Regen.",
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
            config=replace(self.config, openai_search_model="gpt-5.2"),
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
            ["gpt-5.2", "gpt-5.2", "gpt-5.2-chat-latest"],
        )

    def test_compose_print_job_uses_context_and_request_source(self) -> None:
        self.responses.output_text = "TERMINE\nMontag 14 Uhr\nAdresse Praxis"
        response = self.backend.compose_print_job_with_metadata(
            conversation=(("user", "Wann ist der Termin?"), ("assistant", "Montag 14 Uhr.")),
            focus_hint="appointment details",
            direct_text="Montag 14 Uhr bei Dr. Meyer",
            request_source="tool",
        )

        request = self.responses.calls[0]
        self.assertEqual(response.text, "TERMINE\nMontag 14 Uhr\nAdresse Praxis")
        self.assertEqual(request["reasoning"], {"effort": "medium"})
        self.assertEqual(request["max_output_tokens"], 180)
        self.assertNotIn("tools", request)
        self.assertIn("Request source: tool", request["input"][-1]["content"][0]["text"])
        self.assertIn("Focus hint: appointment details", request["input"][-1]["content"][0]["text"])
        self.assertIn("Latest user message: Wann ist der Termin?", request["input"][-1]["content"][0]["text"])
        self.assertIn("Latest assistant message: Montag 14 Uhr.", request["input"][-1]["content"][0]["text"])

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

        self.assertEqual(response.text, "Montag 14 Uhr\nDr. Meyer, Hamburg")
        self.assertEqual(len(self.responses.calls), 2)

    def test_default_client_factory_skips_project_header_for_project_scoped_key(self) -> None:
        captured_kwargs: dict[str, str] = {}

        class FakeOpenAI:
            def __init__(self, **kwargs) -> None:
                captured_kwargs.update(kwargs)

        original_module = sys.modules.get("openai")
        sys.modules["openai"] = SimpleNamespace(OpenAI=FakeOpenAI)
        try:
            _default_client_factory(
                TwinrConfig(
                    openai_api_key="sk-proj-example",
                    openai_project_id="proj_123",
                )
            )
        finally:
            if original_module is None:
                sys.modules.pop("openai", None)
            else:
                sys.modules["openai"] = original_module

        self.assertEqual(captured_kwargs, {"api_key": "sk-proj-example"})

    def test_format_for_print_uses_low_reasoning_without_web_search(self) -> None:
        self.backend.format_for_print("This is a longer answer that should be compressed.")

        request = self.responses.calls[0]
        self.assertEqual(request["reasoning"], {"effort": "low"})
        self.assertNotIn("tools", request)
        self.assertEqual(request["max_output_tokens"], 140)
