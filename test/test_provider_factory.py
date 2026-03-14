from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.providers import build_streaming_provider_bundle
from twinr.providers.deepgram import DeepgramSpeechToTextProvider
from twinr.providers.groq import GroqAgentTextProvider, GroqToolCallingAgentProvider


class StreamingProviderFactoryTests(unittest.TestCase):
    def test_builds_deepgram_groq_openai_bundle(self) -> None:
        bundle = build_streaming_provider_bundle(
            TwinrConfig(
                openai_api_key="openai-key",
                groq_api_key="groq-key",
                deepgram_api_key="deepgram-key",
                stt_provider="deepgram",
                llm_provider="groq",
                tts_provider="openai",
            )
        )

        self.assertIsInstance(bundle.stt, DeepgramSpeechToTextProvider)
        self.assertIsInstance(bundle.agent, GroqAgentTextProvider)
        self.assertIsInstance(bundle.tool_agent, GroqToolCallingAgentProvider)
        self.assertEqual(bundle.tts.__class__.__name__, "OpenAITextToSpeechProvider")
        self.assertEqual(bundle.print_backend.__class__.__name__, "CompositeSpeechAgentProvider")


if __name__ == "__main__":
    unittest.main()
