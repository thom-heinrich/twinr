"""Persist durable long-term memory objects, conflicts, and archives.

This module remains the stable public import path for Twinr callers. The
implementation now lives in focused `_structured_store` modules so snapshot
I/O, mutation flows, query ranking, and retrieval selection can evolve
without growing another monolith.
"""

##REFACTOR: 2026-03-27##
from __future__ import annotations

from twinr.memory.longterm.storage._structured_store import (
    LongTermStructuredStoreBase,
    _write_json_atomic,
)


class LongTermStructuredStore(LongTermStructuredStoreBase):
    """Compatibility wrapper preserving the historic structured-store import path."""


__all__ = [
    "LongTermStructuredStore",
    "_write_json_atomic",
]
