"""Compose the shared OpenAI backend from capability mixins.

``OpenAIBackend`` is the canonical API-backed implementation used by Twinr's
OpenAI adapters. It gathers request construction from ``core`` and behavioral
mixins from ``capabilities`` into one importable backend surface.
"""

from __future__ import annotations

from ..capabilities.phrasing import OpenAIMessagePhrasingMixin
from ..capabilities.printing import OpenAIPrintMixin
from ..capabilities.responses import OpenAIResponseMixin
from ..capabilities.search import OpenAISearchMixin
from ..capabilities.speech import OpenAISpeechMixin
from ..core.base import OpenAIBackendBase
from ..core.client import _default_client_factory, _should_send_project_header
from ..core.instructions import (
    AUTOMATION_EXECUTION_INSTRUCTIONS,
    PRINT_COMPOSER_INSTRUCTIONS,
    PROACTIVE_PROMPT_INSTRUCTIONS,
    REMINDER_DELIVERY_INSTRUCTIONS,
)
from ..core.types import ConversationLike, OpenAIImageInput, OpenAISearchResult, OpenAITextResponse


class OpenAIBackend(
    OpenAIResponseMixin,
    OpenAISearchMixin,
    OpenAISpeechMixin,
    OpenAIMessagePhrasingMixin,
    OpenAIPrintMixin,
    OpenAIBackendBase,
):
    """Combine capability mixins into Twinr's canonical OpenAI backend class."""

    pass


__all__ = [
    "AUTOMATION_EXECUTION_INSTRUCTIONS",
    "ConversationLike",
    "OpenAIBackend",
    "OpenAIImageInput",
    "OpenAISearchResult",
    "OpenAITextResponse",
    "PRINT_COMPOSER_INSTRUCTIONS",
    "PROACTIVE_PROMPT_INSTRUCTIONS",
    "REMINDER_DELIVERY_INSTRUCTIONS",
    "_default_client_factory",
    "_should_send_project_header",
]
