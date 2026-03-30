"""Bounded retry helpers for transient remote long-term failures."""

from __future__ import annotations

from twinr.memory.chonkydb.client import ChonkyDBError

_MAX_TRANSIENT_WRITE_ATTEMPTS = 6
_MAX_TRANSIENT_WRITE_BACKOFF_S = 8.0
_MAX_TRANSIENT_READ_BACKOFF_S = 2.0
_RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_RETRYABLE_DETAIL_MARKERS = frozenset(
    {
        "queue_saturated",
        "payload_sync_bulk_busy",
        "serverbusy",
    }
)


def exception_chain(exc: BaseException) -> tuple[BaseException, ...]:
    """Return one de-duplicated causal chain for classification."""

    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return tuple(chain)


def retryable_remote_write_attempts(configured_attempts: int, *, exc: BaseException) -> int:
    """Return the bounded retry budget for one transient remote write error."""

    attempts = max(1, int(configured_attempts))
    if is_rate_limited_remote_write_error(exc):
        return max(attempts, _MAX_TRANSIENT_WRITE_ATTEMPTS)
    return attempts


def should_retry_remote_write_error(exc: BaseException) -> bool:
    """Return whether one remote write failure is worth retrying."""

    saw_chonky_error = False
    for item in exception_chain(exc):
        if isinstance(item, ChonkyDBError):
            saw_chonky_error = True
            if item.status_code is None:
                continue
            try:
                status_code = int(item.status_code)
            except (TypeError, ValueError):
                continue
            if status_code in _RETRYABLE_HTTP_STATUS_CODES:
                return True
            if 400 <= status_code < 500:
                return False
        if isinstance(item, (TimeoutError, OSError)):
            return True
    return not saw_chonky_error


def should_retry_remote_read_error(exc: BaseException) -> bool:
    """Return whether one remote read failure is worth retrying."""

    saw_chonky_error = False
    for item in exception_chain(exc):
        if isinstance(item, ChonkyDBError):
            saw_chonky_error = True
            if item.status_code is None:
                continue
            try:
                status_code = int(item.status_code)
            except (TypeError, ValueError):
                continue
            if status_code in _RETRYABLE_HTTP_STATUS_CODES:
                return True
            if 400 <= status_code < 500:
                return False
        if isinstance(item, (TimeoutError, OSError)):
            return True
    return not saw_chonky_error


def should_fallback_async_job_resolution_error(exc: BaseException) -> bool:
    """Return whether exact async-job resolution may defer to readback attestation.

    Exact document ids are an optimization. Once the write itself was accepted,
    later jobs-endpoint transport failures may fall back to same-URI readback
    attestation, which remains the authoritative visibility proof.
    """

    for item in exception_chain(exc):
        if isinstance(item, (ChonkyDBError, TimeoutError, OSError)):
            return True
    return False


def clone_client_with_capped_timeout(client: object, *, timeout_s: float) -> object:
    """Clone one client with a smaller timeout when the object supports it."""

    clone_with_timeout = getattr(client, "clone_with_timeout", None)
    if not callable(clone_with_timeout):
        return client
    config = getattr(client, "config", None)
    try:
        current_timeout_s = float(getattr(config, "timeout_s"))
        target_timeout_s = max(0.1, float(timeout_s))
    except (AttributeError, TypeError, ValueError):
        return client
    if target_timeout_s >= current_timeout_s:
        return client
    return clone_with_timeout(target_timeout_s)


def remote_write_retry_delay_s(
    exc: BaseException,
    *,
    default_backoff_s: float,
    attempt_index: int,
) -> float:
    """Return the bounded sleep before retrying one transient remote write."""

    configured_backoff_s = max(0.0, float(default_backoff_s))
    for item in exception_chain(exc):
        if not isinstance(item, ChonkyDBError):
            continue
        retry_after_s = item.retry_after_seconds()
        if retry_after_s is not None:
            return min(_MAX_TRANSIENT_WRITE_BACKOFF_S, max(configured_backoff_s, retry_after_s))
    if is_rate_limited_remote_write_error(exc):
        base_delay_s = max(configured_backoff_s, 1.0)
        return min(_MAX_TRANSIENT_WRITE_BACKOFF_S, base_delay_s * (2 ** max(0, attempt_index)))
    return configured_backoff_s


def remote_read_retry_delay_s(
    exc: BaseException,
    *,
    default_backoff_s: float,
    attempt_index: int,
) -> float:
    """Return the bounded sleep before retrying one transient remote read."""

    configured_backoff_s = max(0.0, float(default_backoff_s))
    for item in exception_chain(exc):
        if not isinstance(item, ChonkyDBError):
            continue
        retry_after_s = item.retry_after_seconds()
        if retry_after_s is not None:
            return min(_MAX_TRANSIENT_READ_BACKOFF_S, max(configured_backoff_s, retry_after_s))
    base_delay_s = max(configured_backoff_s, 0.1)
    return min(_MAX_TRANSIENT_READ_BACKOFF_S, base_delay_s * (2 ** max(0, attempt_index)))


def is_rate_limited_remote_write_error(exc: BaseException) -> bool:
    """Return whether one write failed because the remote backend is saturated."""

    for item in exception_chain(exc):
        if not isinstance(item, ChonkyDBError):
            continue
        try:
            status_code = int(item.status_code) if item.status_code is not None else None
        except (TypeError, ValueError):
            status_code = None
        if status_code == 429:
            return True
        response_json = item.response_json if isinstance(item.response_json, dict) else {}
        detail_bits = " ".join(
            str(response_json.get(field) or "").strip().lower()
            for field in ("detail", "error", "error_type", "title")
        )
        compact_bits = detail_bits.replace(" ", "")
        if any(marker in compact_bits for marker in _RETRYABLE_DETAIL_MARKERS):
            return True
    return False


__all__ = [
    "clone_client_with_capped_timeout",
    "exception_chain",
    "remote_read_retry_delay_s",
    "remote_write_retry_delay_s",
    "should_retry_remote_read_error",
    "retryable_remote_write_attempts",
    "should_fallback_async_job_resolution_error",
    "should_retry_remote_write_error",
]
