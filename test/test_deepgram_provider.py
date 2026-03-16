from pathlib import Path
import sys
import json
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import StreamingSpeechEndpointEvent
from twinr.providers.deepgram import DeepgramSpeechToTextProvider


class FakeDeepgramResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return dict(self._payload)


class FakeDeepgramClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return FakeDeepgramResponse(
            {
                "results": {
                    "channels": [
                        {
                            "alternatives": [
                                {"transcript": "Guten Morgen aus Schwarzenbek."}
                            ]
                        }
                    ]
                }
            }
        )


class FakeWebSocketConnection:
    def __init__(self, messages: list[str]) -> None:
        self.messages = list(messages)
        self.sent: list[object] = []
        self.closed = False
        self.response = type(
            "Response",
            (),
            {"headers": {"dg-request-id": "dg-req-123"}},
        )()

    def send(self, payload) -> None:
        self.sent.append(payload)

    def close(self) -> None:
        self.closed = True

    def __iter__(self):
        return iter(self.messages)


class DeepgramSpeechToTextProviderTests(unittest.TestCase):
    def test_transcribe_posts_audio_bytes_and_extracts_transcript(self) -> None:
        client = FakeDeepgramClient()
        provider = DeepgramSpeechToTextProvider(
            config=TwinrConfig(
                deepgram_api_key="deepgram-key",
                deepgram_base_url="https://api.deepgram.example/v1",
                deepgram_stt_model="nova-3",
                deepgram_stt_language="de",
                deepgram_stt_smart_format=True,
            ),
            client=client,
        )

        transcript = provider.transcribe(
            b"WAVDATA",
            filename="turn.wav",
            content_type="audio/wav",
        )

        self.assertEqual(transcript, "Guten Morgen aus Schwarzenbek.")
        call = client.calls[0]
        self.assertEqual(call["url"], "https://api.deepgram.example/v1/listen")
        self.assertEqual(call["params"]["model"], "nova-3")
        self.assertEqual(call["params"]["language"], "de")
        self.assertEqual(call["params"]["smart_format"], "true")
        self.assertEqual(call["headers"]["Authorization"], "Token deepgram-key")
        self.assertEqual(call["content"], b"WAVDATA")

    def test_start_streaming_session_uses_websocket_and_returns_final_transcript(self) -> None:
        captured = {}

        def fake_connector(url: str, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return FakeWebSocketConnection(
                [
                    json.dumps(
                        {
                            "type": "Results",
                            "is_final": False,
                            "speech_final": False,
                            "channel": {"alternatives": [{"transcript": "Guten Mor"}]},
                        }
                    ),
                    json.dumps(
                        {
                            "type": "Results",
                            "is_final": True,
                            "speech_final": False,
                            "from_finalize": True,
                            "channel": {"alternatives": [{"transcript": "Guten Morgen"}]},
                            "metadata": {"request_id": "dg-req-123"},
                        }
                    ),
                ]
            )

        provider = DeepgramSpeechToTextProvider(
            config=TwinrConfig(
                deepgram_api_key="deepgram-key",
                deepgram_base_url="https://api.deepgram.example/v1",
                deepgram_stt_model="nova-3",
                deepgram_stt_language="de",
                deepgram_stt_smart_format=True,
                deepgram_streaming_interim_results=True,
                deepgram_streaming_endpointing_ms=450,
                deepgram_streaming_utterance_end_ms=900,
                deepgram_streaming_finalize_timeout_s=1.0,
            ),
            websocket_connector=fake_connector,
        )
        partials: list[str] = []
        endpoints: list[StreamingSpeechEndpointEvent] = []
        session = provider.start_streaming_session(
            sample_rate=16000,
            channels=1,
            on_interim=partials.append,
            on_endpoint=endpoints.append,
        )

        session.send_pcm(b"PCM")
        result = session.finalize()
        session.close()

        self.assertIn("wss://api.deepgram.example/v1/listen", captured["url"])
        self.assertIn("model=nova-3", captured["url"])
        self.assertIn("language=de", captured["url"])
        self.assertIn("sample_rate=16000", captured["url"])
        self.assertIn("endpointing=450", captured["url"])
        self.assertEqual(
            captured["kwargs"]["additional_headers"]["Authorization"],
            "Token deepgram-key",
        )
        self.assertEqual(partials, ["Guten Mor"])
        self.assertEqual(len(endpoints), 0)
        self.assertEqual(result.transcript, "Guten Morgen")
        self.assertEqual(result.request_id, "dg-req-123")
        self.assertTrue(result.saw_interim)
        self.assertFalse(result.saw_speech_final)

    def test_streaming_session_emits_endpoint_events_before_finalize(self) -> None:
        def fake_connector(url: str, **kwargs):
            del url, kwargs
            return FakeWebSocketConnection(
                [
                    json.dumps(
                        {
                            "type": "Results",
                            "is_final": False,
                            "speech_final": False,
                            "channel": {"alternatives": [{"transcript": "ich bin noch"}]},
                        }
                    ),
                    json.dumps(
                        {
                            "type": "Results",
                            "is_final": True,
                            "speech_final": True,
                            "channel": {"alternatives": [{"transcript": "ich bin noch am programmieren"}]},
                        }
                    ),
                ]
            )

        provider = DeepgramSpeechToTextProvider(
            config=TwinrConfig(
                deepgram_api_key="deepgram-key",
                deepgram_base_url="https://api.deepgram.example/v1",
                deepgram_stt_model="nova-3",
                deepgram_stt_language="de",
                deepgram_stt_smart_format=True,
            ),
            websocket_connector=fake_connector,
        )
        endpoints: list[StreamingSpeechEndpointEvent] = []
        session = provider.start_streaming_session(
            sample_rate=16000,
            channels=1,
            on_endpoint=endpoints.append,
        )

        session.send_pcm(b"PCM")
        result = session.finalize()

        self.assertEqual(len(endpoints), 1)
        self.assertEqual(endpoints[0].event_type, "speech_final")
        self.assertEqual(endpoints[0].transcript, "ich bin noch am programmieren")
        self.assertEqual(result.transcript, "ich bin noch am programmieren")


if __name__ == "__main__":
    unittest.main()
