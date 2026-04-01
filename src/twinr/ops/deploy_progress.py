"""Structured progress events for Pi runtime deploy workflows.

The operator-facing deploy command keeps stdout reserved for the final JSON
payload, so any live progress must travel over an out-of-band callback
channel, typically stderr. This helper keeps those progress events uniform
across orchestration phases and nested remote-install substeps.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
import time

ProgressCallback = Callable[[dict[str, object]], None]


def emit_deploy_progress(
    progress_callback: ProgressCallback | None,
    *,
    phase: str,
    event: str,
    step: str | None = None,
    detail: str | None = None,
    elapsed_s: float | None = None,
    extra: Mapping[str, object] | None = None,
) -> None:
    """Emit one deploy progress event if a callback is configured.

    Progress reporting must never break the deploy itself, so callback errors
    are swallowed deliberately.
    """

    if progress_callback is None:
        return
    payload: dict[str, object] = {
        "kind": "pi_runtime_deploy_progress",
        "phase": phase,
        "event": event,
    }
    if step:
        payload["step"] = step
    if detail:
        payload["detail"] = detail
    if elapsed_s is not None:
        payload["elapsed_s"] = round(max(0.0, float(elapsed_s)), 3)
    if extra:
        payload.update(dict(extra))
    try:
        progress_callback(payload)
    except Exception:
        return


@contextmanager
def progress_span(
    progress_callback: ProgressCallback | None,
    *,
    phase: str,
    step: str | None = None,
    detail: str | None = None,
) -> Iterator[None]:
    """Emit one start/end/error span for a deploy phase or substep."""

    started = time.monotonic()
    emit_deploy_progress(
        progress_callback,
        phase=phase,
        event="start",
        step=step,
        detail=detail,
    )
    try:
        yield
    except Exception as exc:
        emit_deploy_progress(
            progress_callback,
            phase=phase,
            event="error",
            step=step,
            detail=detail,
            elapsed_s=time.monotonic() - started,
            extra={
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    emit_deploy_progress(
        progress_callback,
        phase=phase,
        event="end",
        step=step,
        detail=detail,
        elapsed_s=time.monotonic() - started,
    )
