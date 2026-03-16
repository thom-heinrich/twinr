"""Expose the shared OpenAI provider primitives used across Twinr.

Import from this package for the canonical response and image dataclasses plus
the small helper surface that higher OpenAI layers re-export.
"""

from .client import _default_client_factory, _should_send_project_header
from .types import ConversationLike, OpenAIImageInput, OpenAISearchResult, OpenAITextResponse

__all__ = [
    "ConversationLike",
    "OpenAIImageInput",
    "OpenAISearchResult",
    "OpenAITextResponse",
    "_default_client_factory",
    "_should_send_project_header",
]
