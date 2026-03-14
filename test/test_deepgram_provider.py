from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
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


if __name__ == "__main__":
    unittest.main()
