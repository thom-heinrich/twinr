"""Export the public OpenAI API surface without eager adapter imports.

This module is the stable API-layer import surface for Twinr code that needs
the shared OpenAI backend or one of its contract-specific adapters. Import
from ``twinr.providers.openai`` for cross-package use and from this module
only when working inside the OpenAI provider package itself.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    )


_EXPORTS: dict[str, tuple[str, str]] = {
    "OpenAIAgentTextProvider": ("twinr.providers.openai.api.adapters", "OpenAIAgentTextProvider"),
    "OpenAIBackend": ("twinr.providers.openai.api.backend", "OpenAIBackend"),
    "OpenAIConversationClosureDecisionProvider": (
        "twinr.providers.openai.api.adapters",
        "OpenAIConversationClosureDecisionProvider",
    ),
    "OpenAIFirstWordProvider": ("twinr.providers.openai.api.adapters", "OpenAIFirstWordProvider"),
    "OpenAIProviderBundle": ("twinr.providers.openai.api.adapters", "OpenAIProviderBundle"),
    "OpenAISpeechToTextProvider": (
        "twinr.providers.openai.api.adapters",
        "OpenAISpeechToTextProvider",
    ),
    "OpenAISupervisorDecisionProvider": (
        "twinr.providers.openai.api.adapters",
        "OpenAISupervisorDecisionProvider",
    ),
    "OpenAIToolCallingAgentProvider": (
        "twinr.providers.openai.api.adapters",
        "OpenAIToolCallingAgentProvider",
    ),
    "OpenAITextToSpeechProvider": (
        "twinr.providers.openai.api.adapters",
        "OpenAITextToSpeechProvider",
    ),
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
