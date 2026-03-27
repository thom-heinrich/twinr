"""Persist fine-grained remote long-term memory documents plus compact catalogs.

This module remains the stable public import path for Twinr callers. The
implementation now lives in focused `_remote_catalog` modules so caching,
search, document hydration, and remote-write attestation can evolve without
growing another monolith.
"""

##REFACTOR: 2026-03-27##
from __future__ import annotations

from twinr.memory.longterm.storage._remote_catalog import (
    LongTermRemoteCatalogAssemblyResult,
    LongTermRemoteCatalogEntry,
    LongTermRemoteCatalogStoreBase,
)


class LongTermRemoteCatalogStore(LongTermRemoteCatalogStoreBase):
    """Compatibility wrapper preserving the historic remote-catalog import path."""


__all__ = [
    "LongTermRemoteCatalogAssemblyResult",
    "LongTermRemoteCatalogEntry",
    "LongTermRemoteCatalogStore",
]
