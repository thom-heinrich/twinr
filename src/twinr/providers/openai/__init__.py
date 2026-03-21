"""Expose the stable package-root OpenAI provider surface for Twinr.

Import from ``twinr.providers.openai`` for the backend, provider-bundle,
realtime, and shared value-object types that callers use across the codebase.
Implementation stays split across the focused ``api``, ``core``, and
``realtime`` subpackages; ``capabilities`` remains an internal mixin layer.
"""

from twinr.providers.openai.api import (
    OpenAIAgentTextProvider,
    OpenAIBackend,
    OpenAIConversationClosureDecisionProvider,
    OpenAIFirstWordProvider,
    OpenAIProviderBundle,
    OpenAISpeechToTextProvider,
    OpenAISupervisorDecisionProvider,
    OpenAIToolCallingAgentProvider,
    OpenAITextToSpeechProvider,
)
from twinr.providers.openai.core import OpenAIImageInput, OpenAITextResponse
from twinr.providers.openai.realtime import OpenAIRealtimeSession, OpenAIRealtimeTurn

__all__ = [
    "OpenAIAgentTextProvider",
    "OpenAIBackend",
    "OpenAIConversationClosureDecisionProvider",
    "OpenAIFirstWordProvider",
    "OpenAIImageInput",
    "OpenAIProviderBundle",
    "OpenAIRealtimeSession",
    "OpenAIRealtimeTurn",
    "OpenAISpeechToTextProvider",
    "OpenAISupervisorDecisionProvider",
    "OpenAITextResponse",
    "OpenAITextToSpeechProvider",
    "OpenAIToolCallingAgentProvider",
]
