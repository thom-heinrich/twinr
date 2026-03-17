"""Shared test helpers for self_coding contracts and fixtures."""

from __future__ import annotations

import hashlib


def stable_sha256(text: str) -> str:
    """Return one deterministic SHA-256 digest for test fixtures."""

    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()
