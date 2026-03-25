"""Provide shared best-effort telemetry helpers for realtime tool handlers.

These helpers centralize the owner callback patterns that multiple handler
modules reuse for telemetry, audit events, and usage accounting. Callers stay
responsible for any domain-specific value sanitization before formatting the
payloads passed here.
"""

from __future__ import annotations

import logging
from typing import Any


def emit_best_effort(
    owner: Any,
    payload: str,
    *,
    logger: logging.Logger | None = None,
    failure_message: str | None = None,
    failure_log_level: str = "warning",
) -> None:
    """Emit one non-critical telemetry payload when the owner supports it."""

    emit = getattr(owner, "emit", None)
    if not callable(emit):
        return
    try:
        emit(payload)
    except Exception:
        if logger is not None and failure_message:
            getattr(logger, failure_log_level, logger.warning)(failure_message, exc_info=True)


def record_event_best_effort(
    owner: Any,
    event_name: str,
    message: str,
    fields: dict[str, object] | None = None,
    /,
    *,
    logger: logging.Logger | None = None,
    failure_message: str | None = None,
    failure_log_level: str = "warning",
) -> None:
    """Record one non-critical tool event when the owner supports it."""

    record_event = getattr(owner, "_record_event", None)
    if not callable(record_event):
        return
    try:
        record_event(event_name, message, **(fields or {}))
    except Exception:
        if logger is not None and failure_message:
            getattr(logger, failure_log_level, logger.warning)(failure_message, exc_info=True)


def record_usage_best_effort(
    owner: Any,
    fields: dict[str, object] | None = None,
    /,
    *,
    logger: logging.Logger | None = None,
    failure_message: str | None = None,
    failure_log_level: str = "warning",
) -> None:
    """Record one non-critical usage payload when the owner supports it."""

    record_usage = getattr(owner, "_record_usage", None)
    if not callable(record_usage):
        return
    try:
        record_usage(**(fields or {}))
    except Exception:
        if logger is not None and failure_message:
            getattr(logger, failure_log_level, logger.warning)(failure_message, exc_info=True)
