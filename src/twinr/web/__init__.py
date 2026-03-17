"""Expose the stable FastAPI app factory for Twinr's local web UI."""

from __future__ import annotations

from typing import Any

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:
    """Lazily resolve package exports so support submodules stay import-safe."""

    if name == "create_app":
        from twinr.web.app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
