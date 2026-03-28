"""Local typing helpers for the refactored long-term runtime service mixins."""

from __future__ import annotations

from typing import Any


class ServiceMixinBase:
    """Expose runtime-owned attributes through `__getattr__` for mixin typing.

    The real attributes live on `LongTermMemoryService`. The refactor splits
    methods into focused mixins, and this base keeps mypy from treating every
    mixin as attribute-less while still raising normal `AttributeError` at
    runtime if a field is genuinely missing.
    """

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)
