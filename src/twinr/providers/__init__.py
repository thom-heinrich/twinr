"""Expose Twinr's provider package root and compatibility imports.

Import ``build_streaming_provider_bundle`` from this package to assemble the
configured runtime provider stack. The root package also preserves the most
common OpenAI-facing imports so existing callers do not need to know the
internal split across ``deepgram``, ``groq``, and ``openai``.
"""

from twinr.providers.factory import StreamingProviderBundle, build_streaming_provider_bundle
from twinr.providers.openai import OpenAIBackend, OpenAIImageInput, OpenAIRealtimeSession, OpenAIRealtimeTurn, OpenAITextResponse

__all__ = [
    "OpenAIBackend",
    "OpenAIImageInput",
    "OpenAIRealtimeSession",
    "OpenAIRealtimeTurn",
    "OpenAITextResponse",
    "StreamingProviderBundle",
    "build_streaming_provider_bundle",
]
