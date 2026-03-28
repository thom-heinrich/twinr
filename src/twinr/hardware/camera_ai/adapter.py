"""Expose the stable public AI-camera adapter surface over the decomposed stack.

This module intentionally stays thin. The production implementation now lives
under ``twinr.hardware.camera_ai.adapter_impl`` so the long-lived import path
remains stable while the adapter is split by responsibility.
"""

from __future__ import annotations

##REFACTOR: 2026-03-27##

from . import adapter_impl as _impl

LocalAICameraAdapter = _impl.LocalAICameraAdapter
PoseResult = _impl.PoseResult

_PUBLIC_EXPORTS = (
    "LocalAICameraAdapter",
    "PoseResult",
)
_COMPAT_EXPORTS = tuple(name for name in _impl.__all__ if name not in _PUBLIC_EXPORTS)

globals().update({name: getattr(_impl, name) for name in _COMPAT_EXPORTS})

__all__ = [
    "LocalAICameraAdapter",
    "PoseResult",
]

for _name in _PUBLIC_EXPORTS + _COMPAT_EXPORTS:
    _exported = globals()[_name]
    try:
        _exported.__module__ = __name__
    except (AttributeError, TypeError):
        continue
