"""Bounded PIR startup gate for transient GPIO busy overlaps.

Twinr's productive Pi runtime can briefly overlap with a previous Twinr
process during restarts. When that happens, GPIO17 may still be requested by
the previous process while the new proactive monitor starts, and legacy gpiod
raises ``OSError(errno.EBUSY)``. This helper retries only that proven transient
case for a short bounded window so startup stays fail-closed for real faults
without flapping into error on short handover races.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import errno
import subprocess
import time


_DEFAULT_BUSY_RETRY_TIMEOUT_S = 2.0
_DEFAULT_BUSY_RETRY_INTERVAL_S = 0.1


class PirOpenable:
    """Describe the tiny PIR startup surface needed by the runtime gate."""

    def open(self) -> object:
        """Open the PIR monitor and return the monitor-like object."""


@dataclass(frozen=True, slots=True)
class PirOpenGateResult:
    """Summarize one bounded PIR startup-open attempt."""

    attempt_count: int
    busy_retry_count: int


def open_pir_monitor_with_busy_retry(
    monitor: PirOpenable,
    *,
    timeout_s: float = _DEFAULT_BUSY_RETRY_TIMEOUT_S,
    retry_interval_s: float = _DEFAULT_BUSY_RETRY_INTERVAL_S,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> PirOpenGateResult:
    """Open one PIR monitor, retrying only proven transient GPIO busy races.

    Args:
        monitor: PIR monitor-like object exposing ``open()``.
        timeout_s: Maximum total retry window for exact ``EBUSY`` failures.
        retry_interval_s: Delay between retries inside the bounded window.
        monotonic: Injected monotonic clock for tests.
        sleep: Injected sleeper for tests.

    Returns:
        One summary describing how many attempts were needed.

    Raises:
        The original open exception when the error is not a proven transient
        GPIO busy condition or when the bounded retry window expires.
    """

    normalized_timeout_s = max(0.0, float(timeout_s))
    normalized_retry_interval_s = max(0.0, float(retry_interval_s))
    deadline = monotonic() + normalized_timeout_s
    attempt_count = 0
    busy_retry_count = 0

    while True:
        attempt_count += 1
        try:
            monitor.open()
            return PirOpenGateResult(
                attempt_count=attempt_count,
                busy_retry_count=busy_retry_count,
            )
        except Exception:
            error = _current_exception()
            if not _is_gpio_busy_error(error):
                raise
            remaining_s = deadline - monotonic()
            if remaining_s <= 0.0:
                raise
            busy_retry_count += 1
            if normalized_retry_interval_s > 0.0:
                sleep(min(normalized_retry_interval_s, remaining_s))


def _current_exception() -> BaseException:
    """Return the currently handled exception with a concrete type."""

    import sys

    error = sys.exc_info()[1]
    if error is None:
        raise RuntimeError("No active exception was available for PIR open gate handling.")
    return error


def _is_gpio_busy_error(error: BaseException | None) -> bool:
    """Return whether the failure chain contains one proven transient GPIO busy."""

    current = error
    seen: set[int] = set()
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            break
        seen.add(current_id)
        if isinstance(current, OSError) and getattr(current, "errno", None) == errno.EBUSY:
            return True
        if _looks_like_gpio_cli_busy(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def _looks_like_gpio_cli_busy(error: BaseException) -> bool:
    """Return whether one exception matches the libgpiod CLI busy signature."""

    text = _exception_text(error).lower()
    if "device or resource busy" not in text:
        return False
    return any(token in text for token in ("gpioget", "gpiomon", "gpiochip", "gpio"))


def _exception_text(error: BaseException) -> str:
    """Render exception text plus relevant subprocess context for matching."""

    parts = [str(error)]
    if isinstance(error, subprocess.CalledProcessError):
        parts.append(_render_subprocess_cmd(error.cmd))
        if isinstance(error.stderr, bytes):
            parts.append(error.stderr.decode(errors="replace"))
        elif isinstance(error.stderr, str):
            parts.append(error.stderr)
    return " ".join(part.strip() for part in parts if part).strip()


def _render_subprocess_cmd(command: object) -> str:
    """Render a subprocess command into one stable string."""

    if isinstance(command, (list, tuple)):
        return " ".join(str(part) for part in command)
    return str(command).strip()


__all__ = [
    "PirOpenGateResult",
    "PirOpenable",
    "open_pir_monitor_with_busy_retry",
]
