"""Export OpenAI capability mixins without eager sibling-module imports.

Import mixins from this package when composing backend surfaces such as
``OpenAIBackend``. The individual modules keep capability-specific behavior
split by concern: responses, search, speech, phrasing, and printing.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .phrasing import OpenAIMessagePhrasingMixin
    from .printing import OpenAIPrintMixin
    from .responses import OpenAIResponseMixin
    from .search import OpenAISearchMixin
    from .speech import OpenAISpeechMixin

    _TYPECHECK_EXPORTS = (
        OpenAIMessagePhrasingMixin,
        OpenAIPrintMixin,
        OpenAIResponseMixin,
        OpenAISearchMixin,
        OpenAISpeechMixin,
    )


_EXPORTS: dict[str, tuple[str, str]] = {
    "OpenAIMessagePhrasingMixin": (
        "twinr.providers.openai.capabilities.phrasing",
        "OpenAIMessagePhrasingMixin",
    ),
    "OpenAIPrintMixin": ("twinr.providers.openai.capabilities.printing", "OpenAIPrintMixin"),
    "OpenAIResponseMixin": (
        "twinr.providers.openai.capabilities.responses",
        "OpenAIResponseMixin",
    ),
    "OpenAISearchMixin": ("twinr.providers.openai.capabilities.search", "OpenAISearchMixin"),
    "OpenAISpeechMixin": ("twinr.providers.openai.capabilities.speech", "OpenAISpeechMixin"),
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
