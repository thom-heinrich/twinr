"""Expose shared OpenAI primitives without eager client or type imports.

Import from this package for the canonical response and image dataclasses plus
the small helper surface that higher OpenAI layers re-export.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import _default_client_factory, _should_send_project_header
    from .types import (
        ConversationLike,
        OpenAIImageInput,
        OpenAISearchResult,
        OpenAITextResponse,
    )

    _TYPECHECK_EXPORTS = (
        ConversationLike,
        OpenAIImageInput,
        OpenAISearchResult,
        OpenAITextResponse,
        _default_client_factory,
        _should_send_project_header,
    )


_EXPORTS: dict[str, tuple[str, str]] = {
    "ConversationLike": ("twinr.providers.openai.core.types", "ConversationLike"),
    "OpenAIImageInput": ("twinr.providers.openai.core.types", "OpenAIImageInput"),
    "OpenAISearchResult": ("twinr.providers.openai.core.types", "OpenAISearchResult"),
    "OpenAITextResponse": ("twinr.providers.openai.core.types", "OpenAITextResponse"),
    "_default_client_factory": (
        "twinr.providers.openai.core.client",
        "_default_client_factory",
    ),
    "_should_send_project_header": (
        "twinr.providers.openai.core.client",
        "_should_send_project_header",
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
