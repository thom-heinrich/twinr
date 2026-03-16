"""Expose the stable FastAPI app factory for Twinr's local web UI."""

from twinr.web.app import create_app

__all__ = ["create_app"]
