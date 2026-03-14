from __future__ import annotations

from .base import OpenAIBackendBase
from .client import _default_client_factory, _should_send_project_header
from .instructions import (
    AUTOMATION_EXECUTION_INSTRUCTIONS,
    PRINT_COMPOSER_INSTRUCTIONS,
    PROACTIVE_PROMPT_INSTRUCTIONS,
    REMINDER_DELIVERY_INSTRUCTIONS,
)
from .phrasing import OpenAIMessagePhrasingMixin
from .printing import OpenAIPrintMixin
from .responses import OpenAIResponseMixin
from .search import OpenAISearchMixin
from .speech import OpenAISpeechMixin
from .types import ConversationLike, OpenAIImageInput, OpenAISearchResult, OpenAITextResponse


class OpenAIBackend(
    OpenAIResponseMixin,
    OpenAISearchMixin,
    OpenAISpeechMixin,
    OpenAIMessagePhrasingMixin,
    OpenAIPrintMixin,
    OpenAIBackendBase,
):
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
