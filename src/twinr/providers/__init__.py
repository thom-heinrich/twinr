"""Expose Twinr provider compatibility imports without eager backend imports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from twinr.providers.factory import StreamingProviderBundle, build_streaming_provider_bundle
    from twinr.providers.openai.api import OpenAIBackend
    from twinr.providers.openai.core import OpenAIImageInput, OpenAITextResponse
    from twinr.providers.openai.realtime import OpenAIRealtimeSession, OpenAIRealtimeTurn

    _TYPECHECK_EXPORTS = (
        StreamingProviderBundle,
        build_streaming_provider_bundle,
        OpenAIBackend,
        OpenAIImageInput,
        OpenAITextResponse,
        OpenAIRealtimeSession,
        OpenAIRealtimeTurn,
    )


_EXPORTS: dict[str, tuple[str, str]] = {
    "StreamingProviderBundle": ("twinr.providers.factory", "StreamingProviderBundle"),
    "build_streaming_provider_bundle": (
        "twinr.providers.factory",
        "build_streaming_provider_bundle",
    ),
    "OpenAIBackend": ("twinr.providers.openai", "OpenAIBackend"),
    "OpenAIImageInput": ("twinr.providers.openai", "OpenAIImageInput"),
    "OpenAIRealtimeSession": ("twinr.providers.openai", "OpenAIRealtimeSession"),
    "OpenAIRealtimeTurn": ("twinr.providers.openai", "OpenAIRealtimeTurn"),
    "OpenAITextResponse": ("twinr.providers.openai", "OpenAITextResponse"),
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
