from __future__ import annotations

from dataclasses import dataclass

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import CombinedSpeechAgentProvider, ProviderBundle
from twinr.providers.deepgram import DeepgramSpeechToTextProvider
from twinr.providers.groq import GroqAgentTextProvider, GroqToolCallingAgentProvider
from twinr.providers.openai import OpenAIBackend, OpenAIProviderBundle


@dataclass
class StreamingProviderBundle(ProviderBundle):
    print_backend: CombinedSpeechAgentProvider
    support_backend: OpenAIBackend


def build_streaming_provider_bundle(
    config: TwinrConfig,
    *,
    support_backend: OpenAIBackend | None = None,
) -> StreamingProviderBundle:
    support = support_backend or OpenAIBackend(config=config)
    openai_bundle = OpenAIProviderBundle.from_backend(support)

    stt_name = (config.stt_provider or "openai").strip().lower()
    llm_name = (config.llm_provider or "openai").strip().lower()
    tts_name = (config.tts_provider or "openai").strip().lower()

    if stt_name == "openai":
        stt = openai_bundle.stt
    elif stt_name == "deepgram":
        stt = DeepgramSpeechToTextProvider(config=config)
    else:
        raise RuntimeError(f"Unsupported TWINR_STT_PROVIDER: {config.stt_provider}")

    if llm_name == "openai":
        agent = openai_bundle.agent
        tool_agent = openai_bundle.tool_agent
    elif llm_name == "groq":
        agent = GroqAgentTextProvider(
            config=config,
            support_provider=openai_bundle.agent,
        )
        tool_agent = GroqToolCallingAgentProvider(config=config)
    else:
        raise RuntimeError(f"Unsupported TWINR_LLM_PROVIDER: {config.llm_provider}")

    if tts_name == "openai":
        tts = openai_bundle.tts
    else:
        raise RuntimeError(f"Unsupported TWINR_TTS_PROVIDER: {config.tts_provider}")

    if tool_agent is None:
        raise RuntimeError("The selected LLM provider does not expose tool-calling support")

    return StreamingProviderBundle(
        stt=stt,
        agent=agent,
        tts=tts,
        tool_agent=tool_agent,
        print_backend=openai_bundle.combined,
        support_backend=support,
    )
