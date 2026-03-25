"""Shared text and hash helpers for ambient display impulse generation."""

from __future__ import annotations

import hashlib


def normalized_text(value: object | None) -> str:
    """Collapse arbitrary text into one compact single line."""

    return " ".join(str(value or "").split()).strip()


def truncate_text(value: object | None, *, max_len: int) -> str:
    """Return one bounded display-safe text field."""

    compact = normalized_text(value)
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def stable_fraction(*parts: object) -> float:
    """Return one deterministic 0..1 fraction for bounded copy variation."""

    digest = hashlib.sha1(
        "::".join(normalized_text(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") / 4_294_967_295.0
