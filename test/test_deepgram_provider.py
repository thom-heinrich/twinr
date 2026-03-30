from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import sys
import json
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import StreamingSpeechEndpointEvent
from twinr.providers.deepgram import DeepgramSpeechToTextProvider


class FakeDeepgramResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self.text = ""
        self.closed = False

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return dict(self._payload)

    def close(self) -> None:
        self.closed = True


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


class DelayedFinalizeResultWebSocketConnection(FakeWebSocketConnection):
    def __init__(
        self,
        *,
        speech_final_message: str,
        final_message: str,
        finalize_delay_s: float = 0.15,
    ) -> None:
        super().__init__([])
        self._speech_final_message = speech_final_message
        self._final_message = final_message
        self._finalize_delay_s = max(0.0, float(finalize_delay_s))
        self._finalize_sent = False

    def send(self, payload) -> None:
        super().send(payload)
        if payload == json.dumps({"type": "Finalize"}):
            self._finalize_sent = True

    def __iter__(self):
        yield self._speech_final_message
        deadline = time.monotonic() + 1.0
        while not self._finalize_sent and time.monotonic() < deadline:
            time.sleep(0.01)
        if not self._finalize_sent:
            raise AssertionError("Finalize was not sent before the delayed final result.")
        time.sleep(self._finalize_delay_s)
        yield self._final_message


class DelayedFluxCloseStreamWebSocketConnection(FakeWebSocketConnection):
    def __init__(
        self,
        *,
        update_message: str,
        end_of_turn_message: str,
        close_delay_s: float = 0.15,
    ) -> None:
        super().__init__([])
        self._update_message = update_message
        self._end_of_turn_message = end_of_turn_message
        self._close_delay_s = max(0.0, float(close_delay_s))
        self._close_stream_sent = False
        self.response = type(
            "Response",
            (),
            {"headers": {"dg-request-id": "dg-flux-123"}},
        )()

    def send(self, payload) -> None:
        super().send(payload)
        if payload == json.dumps({"type": "CloseStream"}):
            self._close_stream_sent = True

    def __iter__(self):
        yield self._update_message
        deadline = time.monotonic() + 1.0
        while not self._close_stream_sent and time.monotonic() < deadline:
            time.sleep(0.01)
        if not self._close_stream_sent:
            raise AssertionError("CloseStream was not sent before the delayed Flux final result.")
        time.sleep(self._close_delay_s)
        yield self._end_of_turn_message


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
        split = urlsplit(call["url"])
        params = parse_qs(split.query)
        self.assertEqual(
            f"{split.scheme}://{split.netloc}{split.path}",
            "https://api.deepgram.example/v1/listen",
        )
        self.assertEqual(params["model"], ["nova-3"])
        self.assertEqual(params["language"], ["de"])
        self.assertEqual(params["smart_format"], ["true"])
        self.assertEqual(params["mip_opt_out"], ["true"])
        self.assertEqual(call["headers"]["Authorization"], "Token deepgram-key")
        self.assertEqual(call["content"], b"WAVDATA")

    def test_transcribe_maps_prompt_terms_to_nova3_keyterms(self) -> None:
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

        provider.transcribe(
            b"WAVDATA",
            filename="turn.wav",
            content_type="audio/wav",
            prompt="Twinr, Twinna, Twinner.",
        )

        params = parse_qs(urlsplit(client.calls[0]["url"]).query)
        self.assertEqual(params["keyterm"], ["Twinr", "Twinna", "Twinner"])

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

    def test_streaming_session_waits_for_real_final_after_bare_speech_final(self) -> None:
        def fake_connector(url: str, **kwargs):
            del url, kwargs
            return DelayedFinalizeResultWebSocketConnection(
                speech_final_message=json.dumps(
                    {
                        "type": "Results",
                        "is_final": False,
                        "speech_final": True,
                        "channel": {"alternatives": [{"transcript": "Heute schon geredet."}]},
                    }
                ),
                final_message=json.dumps(
                    {
                        "type": "Results",
                        "is_final": True,
                        "speech_final": True,
                        "from_finalize": True,
                        "channel": {
                            "alternatives": [{"transcript": "Worüber haben wir heute schon geredet?"}]
                        },
                        "metadata": {"request_id": "dg-req-456"},
                    }
                ),
            )

        provider = DeepgramSpeechToTextProvider(
            config=TwinrConfig(
                deepgram_api_key="deepgram-key",
                deepgram_base_url="https://api.deepgram.example/v1",
                deepgram_stt_model="nova-3",
                deepgram_stt_language="de",
                deepgram_stt_smart_format=True,
                deepgram_streaming_finalize_timeout_s=1.0,
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
        self.assertEqual(endpoints[0].transcript, "Heute schon geredet.")
        self.assertEqual(result.transcript, "Worüber haben wir heute schon geredet?")
        self.assertEqual(result.request_id, "dg-req-123")
        self.assertTrue(result.saw_speech_final)

    def test_flux_streaming_session_waits_for_end_of_turn_after_close_stream(self) -> None:
        captured = {}

        def fake_connector(url: str, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return DelayedFluxCloseStreamWebSocketConnection(
                update_message=json.dumps(
                    {
                        "type": "TurnInfo",
                        "event": "Update",
                        "transcript": "Good mor",
                    }
                ),
                end_of_turn_message=json.dumps(
                    {
                        "type": "TurnInfo",
                        "event": "EndOfTurn",
                        "transcript": "Good morning",
                        "request_id": "dg-flux-123",
                        "end_of_turn_confidence": 0.91,
                    }
                ),
            )

        provider = DeepgramSpeechToTextProvider(
            config=TwinrConfig(
                deepgram_api_key="deepgram-key",
                deepgram_base_url="https://api.deepgram.example/v1",
                deepgram_stt_model="flux-general-en",
                deepgram_stt_language="de",
                deepgram_stt_smart_format=True,
                deepgram_streaming_finalize_timeout_s=1.0,
            ),
            websocket_connector=fake_connector,
        )
        partials: list[str] = []
        endpoints: list[StreamingSpeechEndpointEvent] = []
        session = provider.start_streaming_session(
            sample_rate=16000,
            channels=1,
            prompt="Twinr, Twinna",
            on_interim=partials.append,
            on_endpoint=endpoints.append,
        )

        session.send_pcm(b"PCM" * 2000)
        result = session.finalize()

        self.assertIn("wss://api.deepgram.example/v2/listen", captured["url"])
        self.assertIn("model=flux-general-en", captured["url"])
        self.assertNotIn("language=", captured["url"])
        self.assertIn("keyterm=Twinr", captured["url"])
        self.assertIn("keyterm=Twinna", captured["url"])
        self.assertEqual(
            captured["kwargs"]["additional_headers"]["Authorization"],
            "Token deepgram-key",
        )
        self.assertEqual(partials, ["Good mor"])
        self.assertEqual(len(endpoints), 1)
        self.assertEqual(endpoints[0].event_type, "end_of_turn")
        self.assertEqual(endpoints[0].transcript, "Good morning")
        self.assertEqual(result.transcript, "Good morning")
        self.assertEqual(result.request_id, "dg-flux-123")
        self.assertTrue(result.saw_interim)
        self.assertTrue(result.saw_speech_final)


if __name__ == "__main__":
    unittest.main()
