"""Export Twinr's OpenAI realtime surface without eager session imports.

Re-export ``OpenAIRealtimeSession`` and ``OpenAIRealtimeTurn`` so callers can
depend on ``twinr.providers.openai.realtime`` instead of the implementation
module.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .session import OpenAIRealtimeSession, OpenAIRealtimeTurn

    _TYPECHECK_EXPORTS = (
        OpenAIRealtimeSession,
        OpenAIRealtimeTurn,
    )


_EXPORTS: dict[str, tuple[str, str]] = {
    "OpenAIRealtimeSession": ("twinr.providers.openai.realtime.session", "OpenAIRealtimeSession"),
    "OpenAIRealtimeTurn": ("twinr.providers.openai.realtime.session", "OpenAIRealtimeTurn"),
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
