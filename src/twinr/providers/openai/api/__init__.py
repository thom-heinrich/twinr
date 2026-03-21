"""Export the public OpenAI backend and adapter surfaces.

This module is the stable API-layer import surface for Twinr code that needs
the shared OpenAI backend or one of its contract-specific adapters. Import
from ``twinr.providers.openai`` for cross-package use and from this module
only when working inside the OpenAI provider package itself.
"""

from .adapters import (
    OpenAIAgentTextProvider,
    OpenAIConversationClosureDecisionProvider,
    OpenAIFirstWordProvider,
    OpenAIProviderBundle,
    OpenAISpeechToTextProvider,
    OpenAISupervisorDecisionProvider,
    OpenAIToolCallingAgentProvider,
    OpenAITextToSpeechProvider,
)
from .backend import OpenAIBackend

__all__ = [
    "OpenAIAgentTextProvider",
    "OpenAIBackend",
    "OpenAIConversationClosureDecisionProvider",
    "OpenAIFirstWordProvider",
    "OpenAIProviderBundle",
    "OpenAISpeechToTextProvider",
    "OpenAISupervisorDecisionProvider",
    "OpenAITextToSpeechProvider",
    "OpenAIToolCallingAgentProvider",
]
