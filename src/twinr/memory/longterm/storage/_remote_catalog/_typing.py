"""Local typing helpers for the refactored remote-catalog mixins."""

from __future__ import annotations

from typing import Any

from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore

from .shared import (
    _CachedCatalogEntries,
    _CachedItemPayload,
    _CachedLocalSearchSelector,
    _CachedSearchResult,
)


class RemoteCatalogMixinBase:
    """Expose store-owned attributes through ``__getattr__`` for mixin typing.

    The real implementations live on ``LongTermRemoteCatalogStoreBase``. The
    remote-catalog refactor split behavior into focused mixins, and this base
    keeps mypy from treating every mixin as attribute-less while still raising
    normal ``AttributeError`` at runtime if a field is genuinely missing.
    """

    remote_state: LongTermRemoteStateStore | None
    _catalog_entries_cache: dict[str, _CachedCatalogEntries]
    _item_payload_cache: dict[tuple[str, str], _CachedItemPayload]
    _search_result_cache: dict[tuple[str, str, int, tuple[str, ...]], _CachedSearchResult]
    _local_search_selector_cache: dict[str, _CachedLocalSearchSelector]
    _recent_catalog_head_cache: dict[str, _CachedItemPayload]
    _recent_catalog_entries_cache: dict[str, _CachedCatalogEntries]
    _unsupported_scope_search_cache: dict[str, float]
    _invalid_catalog_head_cache: dict[str, float]

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)
