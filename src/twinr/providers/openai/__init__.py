"""Expose the stable OpenAI provider surface without eager subpackage imports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

    _TYPECHECK_EXPORTS = (
        OpenAIAgentTextProvider,
        OpenAIBackend,
        OpenAIConversationClosureDecisionProvider,
        OpenAIFirstWordProvider,
        OpenAIProviderBundle,
        OpenAISpeechToTextProvider,
        OpenAISupervisorDecisionProvider,
        OpenAIToolCallingAgentProvider,
        OpenAITextToSpeechProvider,
        OpenAIImageInput,
        OpenAITextResponse,
        OpenAIRealtimeSession,
        OpenAIRealtimeTurn,
    )


_EXPORTS: dict[str, tuple[str, str]] = {
    "OpenAIAgentTextProvider": ("twinr.providers.openai.api", "OpenAIAgentTextProvider"),
    "OpenAIBackend": ("twinr.providers.openai.api", "OpenAIBackend"),
    "OpenAIConversationClosureDecisionProvider": (
        "twinr.providers.openai.api",
        "OpenAIConversationClosureDecisionProvider",
    ),
    "OpenAIFirstWordProvider": ("twinr.providers.openai.api", "OpenAIFirstWordProvider"),
    "OpenAIProviderBundle": ("twinr.providers.openai.api", "OpenAIProviderBundle"),
    "OpenAISpeechToTextProvider": ("twinr.providers.openai.api", "OpenAISpeechToTextProvider"),
    "OpenAISupervisorDecisionProvider": (
        "twinr.providers.openai.api",
        "OpenAISupervisorDecisionProvider",
    ),
    "OpenAIToolCallingAgentProvider": (
        "twinr.providers.openai.api",
        "OpenAIToolCallingAgentProvider",
    ),
    "OpenAITextToSpeechProvider": ("twinr.providers.openai.api", "OpenAITextToSpeechProvider"),
    "OpenAIImageInput": ("twinr.providers.openai.core", "OpenAIImageInput"),
    "OpenAITextResponse": ("twinr.providers.openai.core", "OpenAITextResponse"),
    "OpenAIRealtimeSession": ("twinr.providers.openai.realtime", "OpenAIRealtimeSession"),
    "OpenAIRealtimeTurn": ("twinr.providers.openai.realtime", "OpenAIRealtimeTurn"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
