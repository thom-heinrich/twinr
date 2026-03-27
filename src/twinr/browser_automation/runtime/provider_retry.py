"""Retry transient OpenAI browser-automation provider failures.

These helpers keep bounded retry policy out of the ignored repo-root
``browser_automation/`` workspace while still giving local browser runtimes a
stable, tracked place for provider-transient handling. The policy is purposely
small: retry only transport/time-out/server failures that are likely transient,
and fail closed for everything else.
"""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import TypeVar

from openai import APIConnectionError, APITimeoutError, InternalServerError

_ResultT = TypeVar("_ResultT")
_RETRIABLE_ERRORS = (APIConnectionError, APITimeoutError, InternalServerError)


def is_retryable_openai_error(error: BaseException) -> bool:
    """Return whether one OpenAI exception is worth a bounded retry."""

    return isinstance(error, _RETRIABLE_ERRORS)


def call_openai_with_retry(
    *,
    label: str,
    func: Callable[[], _ResultT],
    max_attempts: int = 5,
    base_delay_s: float = 0.75,
) -> _ResultT:
    """Run one OpenAI call with small bounded retries for transient failures."""

    attempts = max(1, int(max_attempts))
    delay_s = max(0.0, float(base_delay_s))
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:
            if not is_retryable_openai_error(exc) or attempt >= attempts:
                raise
            if delay_s > 0.0:
                time.sleep(min(6.0, delay_s * (2 ** (attempt - 1))))
    raise RuntimeError(f"{label} exhausted retry loop without returning or raising cleanly")
