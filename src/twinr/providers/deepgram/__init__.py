"""Expose the Deepgram provider surface without eager transport imports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from twinr.providers.deepgram.speech import DeepgramSpeechToTextProvider

    _TYPECHECK_EXPORTS = (DeepgramSpeechToTextProvider,)


_EXPORTS: dict[str, tuple[str, str]] = {
    "DeepgramSpeechToTextProvider": (
        "twinr.providers.deepgram.speech",
        "DeepgramSpeechToTextProvider",
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
