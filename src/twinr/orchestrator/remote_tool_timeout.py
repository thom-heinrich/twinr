"""Shared timeout policy for remote orchestrator tool calls.

The websocket client and the orchestrator-side remote-tool bridge both wait on
the same logical tool execution. They must therefore share the same default and
environment-controlled timeout policy; otherwise one side can fail a still-
running tool while the other side continues waiting for it.
"""

from __future__ import annotations

import logging
import os


DEFAULT_REMOTE_TOOL_TIMEOUT_SECONDS = 90.0
REMOTE_TOOL_TIMEOUT_ENV = "TWINR_REMOTE_TOOL_TIMEOUT_SECONDS"


def read_remote_tool_timeout_seconds(*, logger: logging.Logger | None = None) -> float:
    """Return the shared remote-tool timeout budget from env or default."""

    raw_value = os.getenv(REMOTE_TOOL_TIMEOUT_ENV)
    if raw_value is None:
        return DEFAULT_REMOTE_TOOL_TIMEOUT_SECONDS
    try:
        parsed = float(raw_value)
    except ValueError:
        if logger is not None:
            logger.warning(
                "Invalid %s=%r; using default %.1f",
                REMOTE_TOOL_TIMEOUT_ENV,
                raw_value,
                DEFAULT_REMOTE_TOOL_TIMEOUT_SECONDS,
            )
        return DEFAULT_REMOTE_TOOL_TIMEOUT_SECONDS
    if parsed <= 0:
        if logger is not None:
            logger.warning(
                "Non-positive %s=%r; using default %.1f",
                REMOTE_TOOL_TIMEOUT_ENV,
                raw_value,
                DEFAULT_REMOTE_TOOL_TIMEOUT_SECONDS,
            )
        return DEFAULT_REMOTE_TOOL_TIMEOUT_SECONDS
    return parsed


__all__ = [
    "DEFAULT_REMOTE_TOOL_TIMEOUT_SECONDS",
    "REMOTE_TOOL_TIMEOUT_ENV",
    "read_remote_tool_timeout_seconds",
]
