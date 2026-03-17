from pathlib import Path
import sys
import unittest
from unittest.mock import patch

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
        self.assertEqual(bundle.verification_stt.__class__.__name__, "OpenAISpeechToTextProvider")

    @patch("twinr.providers.factory._build_openai_streaming_verification_provider")
    def test_builds_verification_stt_when_enabled(
        self,
        build_verifier,
    ) -> None:
        verifier_stt = object()
        build_verifier.return_value = verifier_stt
        bundle = build_streaming_provider_bundle(
            TwinrConfig(
                openai_api_key="openai-key",
                stt_provider="openai",
                llm_provider="openai",
                tts_provider="openai",
                streaming_transcript_verifier_enabled=True,
                streaming_transcript_verifier_model="gpt-4o-mini-transcribe",
            )
        )

        self.assertIs(bundle.verification_stt, verifier_stt)
        build_verifier.assert_called_once()


if __name__ == "__main__":
    unittest.main()
