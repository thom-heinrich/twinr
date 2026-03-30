# CHANGELOG: 2026-03-29
# BUG-1: `probe_count` was normalized via `int()`, silently truncating non-integral values
#        and weakening startup gating enough to flip capture to `ok` too early.
# BUG-2: failure contextualization assumed `exc.probe` was always present/dataclass-like;
#        missing probe metadata could mask the real readiness failure.
# SEC-1: unbounded gate parameters could be abused via config/OTA to keep the assistant
#        offline in startup for excessive time on Raspberry Pi deployments.
# IMP-1: add a monotonic end-to-end startup deadline and richer probe context while
#        preserving the last successful probe as the return value.
# IMP-2: add cancellable settle waits, configurable `chunk_count`, and an optional
#        `probe_validator` hook for stronger 2026-grade health checks.

"""Bounded startup gating for ReSpeaker capture stability.

Twinr should not treat a briefly enumerated XVF3800 path as healthy startup.
This helper now enforces:

* strict, bounded configuration validation;
* a monotonic end-to-end startup deadline;
* several consecutive readable-frame probes separated by bounded settle waits; and
* an optional semantic `probe_validator` hook for deeper health checks.

These changes keep the helper drop-in for existing callers while making startup
behavior more predictable and resilient on Raspberry Pi deployments.
"""

from __future__ import annotations

from copy import copy
from dataclasses import replace
from typing import Protocol
import math
import operator
import time

from twinr.hardware.audio import AudioCaptureReadinessError, AudioCaptureReadinessProbe

_DEFAULT_STABILITY_PROBE_COUNT = 3
_DEFAULT_STABILITY_SETTLE_S = 0.25
_DEFAULT_CHUNK_COUNT = 1
_DEFAULT_DEADLINE_SLACK_S = 0.5

_MIN_DURATION_MS = 1
_MAX_DURATION_MS = 10_000
_MAX_PROBE_COUNT = 32
_MAX_SETTLE_S = 5.0
_MAX_CHUNK_COUNT = 32
_MAX_PLANNED_GATE_S = 30.0
_MAX_TOTAL_TIMEOUT_S = _MAX_PLANNED_GATE_S + _DEFAULT_DEADLINE_SLACK_S


class ReadableFrameProbeSampler(Protocol):
    """Small protocol for startup capture gating."""

    def require_readable_frames(
        self,
        *,
        duration_ms: int | None = None,
        chunk_count: int = 1,
    ) -> AudioCaptureReadinessProbe:
        """Return one bounded readable-frame probe or raise."""


class CancellationSignal(Protocol):
    """Protocol accepted by `cancel_event` for cooperative early abort."""

    def is_set(self) -> bool:
        """Return whether the gate should abort."""

    def wait(self, timeout: float | None = None) -> bool:
        """Block until set or timeout; return True when set."""


class ProbeValidator(Protocol):
    """Optional semantic validator for a successful readable-frame probe."""

    def __call__(self, probe: AudioCaptureReadinessProbe) -> AudioCaptureReadinessProbe:
        """Return the validated probe or raise if the probe is not healthy enough."""


def _normalize_int(name: str, value: object, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{name} must not be empty")
        try:
            candidate = int(stripped, 10)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer value, got {value!r}") from exc
    elif isinstance(value, float):
        # BREAKING: non-integral floats are now rejected instead of silently truncating.
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
        if not value.is_integer():
            raise ValueError(f"{name} must be an integer value, got {value!r}")
        candidate = int(value)
    else:
        try:
            candidate = operator.index(value)
        except TypeError as exc:
            raise TypeError(f"{name} must be an integer-like value, got {value!r}") from exc

    if candidate < minimum or candidate > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got {candidate}")
    return candidate


def _normalize_float(name: str, value: object, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        candidate = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(candidate):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if candidate < minimum or candidate > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got {candidate}")
    return candidate


def _elapsed_ms(started_ns: int) -> int:
    return max(0, (time.monotonic_ns() - started_ns) // 1_000_000)


def _remaining_s(deadline_ns: int | None) -> float | None:
    if deadline_ns is None:
        return None
    remaining_ns = deadline_ns - time.monotonic_ns()
    if remaining_ns <= 0:
        return 0.0
    return remaining_ns / 1_000_000_000.0


def _append_context(detail: str | None, context: str) -> str:
    return f"{detail} | {context}" if detail else context


def _safe_probe_detail(probe: AudioCaptureReadinessProbe | None) -> str | None:
    if probe is None:
        return None
    detail = getattr(probe, "detail", None)
    return detail if isinstance(detail, str) and detail else None


def _safe_replace_probe_detail(
    probe: AudioCaptureReadinessProbe | None,
    detail: str,
) -> AudioCaptureReadinessProbe | None:
    if probe is None:
        return None
    try:
        return replace(probe, detail=detail)
    except Exception:
        try:
            probe_copy = copy(probe)
            setattr(probe_copy, "detail", detail)
            return probe_copy
        except Exception:
            return probe


def _build_readiness_error(
    detail: str,
    *,
    probe: AudioCaptureReadinessProbe | None = None,
) -> AudioCaptureReadinessError:
    if probe is None:
        probe = AudioCaptureReadinessProbe(
            device="unknown",
            sample_rate=0,
            channels=0,
            chunk_ms=0,
            duration_ms=0,
            target_chunk_count=0,
            captured_chunk_count=0,
            captured_bytes=0,
            failure_reason="startup_stability_gate",
            detail=detail,
        )
    return AudioCaptureReadinessError(detail, probe=probe)


def _raise_if_cancelled(
    *,
    cancel_event: CancellationSignal | None,
    started_ns: int,
    probe_count: int,
    last_probe: AudioCaptureReadinessProbe | None,
) -> None:
    if cancel_event is None or not cancel_event.is_set():
        return
    detail = _append_context(
        _safe_probe_detail(last_probe),
        (
            "startup stability gate cancelled before completion "
            f"after {_elapsed_ms(started_ns)}ms while requiring {probe_count} probe(s)"
        ),
    )
    raise _build_readiness_error(detail, probe=_safe_replace_probe_detail(last_probe, detail))


def _wait_for_settle(
    *,
    delay_s: float,
    cancel_event: CancellationSignal | None,
    started_ns: int,
    probe_count: int,
    last_probe: AudioCaptureReadinessProbe | None,
) -> None:
    if delay_s <= 0.0:
        return
    if cancel_event is None:
        time.sleep(delay_s)
        return
    if cancel_event.wait(delay_s):
        detail = _append_context(
            _safe_probe_detail(last_probe),
            (
                "startup stability gate cancelled during settle wait "
                f"after {_elapsed_ms(started_ns)}ms while requiring {probe_count} probe(s)"
            ),
        )
        raise _build_readiness_error(detail, probe=_safe_replace_probe_detail(last_probe, detail))


def require_stable_respeaker_capture(
    *,
    sampler: ReadableFrameProbeSampler,
    duration_ms: int,
    probe_count: int = _DEFAULT_STABILITY_PROBE_COUNT,
    settle_s: float = _DEFAULT_STABILITY_SETTLE_S,
    chunk_count: int = _DEFAULT_CHUNK_COUNT,
    total_timeout_s: float | None = None,
    cancel_event: CancellationSignal | None = None,
    probe_validator: ProbeValidator | None = None,
) -> AudioCaptureReadinessProbe:
    """Require several consecutive readable-frame probes before startup succeeds.

    Args:
        sampler: Object that can perform bounded readable-frame probes.
        duration_ms: Duration for each individual probe.
        probe_count: Number of consecutive successful probes required.
        settle_s: Delay between successful probes.
        chunk_count: Number of chunks required per probe.
        total_timeout_s: Optional end-to-end deadline for the whole gate.
        cancel_event: Optional cooperative cancellation primitive.
        probe_validator: Optional hook for semantic validation of a successful
            readable-frame probe (for example: non-silence, timestamp monotonicity,
            channel count, or expected metadata).
    """

    normalized_duration_ms = _normalize_int(
        "duration_ms",
        duration_ms,
        minimum=_MIN_DURATION_MS,
        maximum=_MAX_DURATION_MS,
    )
    normalized_probe_count = _normalize_int(
        "probe_count",
        probe_count,
        minimum=1,
        maximum=_MAX_PROBE_COUNT,
    )
    normalized_chunk_count = _normalize_int(
        "chunk_count",
        chunk_count,
        minimum=1,
        maximum=_MAX_CHUNK_COUNT,
    )
    normalized_settle_s = _normalize_float(
        "settle_s",
        settle_s,
        minimum=0.0,
        maximum=_MAX_SETTLE_S,
    )

    planned_window_s = (
        normalized_probe_count * (normalized_duration_ms / 1000.0)
        + max(0, normalized_probe_count - 1) * normalized_settle_s
    )
    if planned_window_s > _MAX_PLANNED_GATE_S:
        # BREAKING: startup gating parameters are now bounded to prevent
        # configuration-driven denial of service on real devices.
        raise ValueError(
            "planned startup stability gate is too long: "
            f"{planned_window_s:.3f}s exceeds {_MAX_PLANNED_GATE_S:.3f}s"
        )

    if total_timeout_s is None:
        normalized_total_timeout_s = min(
            _MAX_TOTAL_TIMEOUT_S,
            planned_window_s + _DEFAULT_DEADLINE_SLACK_S,
        )
    else:
        normalized_total_timeout_s = _normalize_float(
            "total_timeout_s",
            total_timeout_s,
            minimum=0.0,
            maximum=_MAX_TOTAL_TIMEOUT_S,
        )

    started_ns = time.monotonic_ns()
    deadline_ns = started_ns + int(normalized_total_timeout_s * 1_000_000_000)
    last_probe: AudioCaptureReadinessProbe | None = None

    for index in range(normalized_probe_count):
        attempt = index + 1
        _raise_if_cancelled(
            cancel_event=cancel_event,
            started_ns=started_ns,
            probe_count=normalized_probe_count,
            last_probe=last_probe,
        )

        remaining_before_attempt_s = _remaining_s(deadline_ns)
        if remaining_before_attempt_s is not None and remaining_before_attempt_s <= 0.0:
            detail = _append_context(
                _safe_probe_detail(last_probe),
                (
                    "startup stability gate timed out before probe "
                    f"{attempt}/{normalized_probe_count} after {_elapsed_ms(started_ns)}ms "
                    f"(limit {normalized_total_timeout_s:.3f}s)"
                ),
            )
            raise _build_readiness_error(
                detail,
                probe=_safe_replace_probe_detail(last_probe, detail),
            )

        try:
            candidate_probe = sampler.require_readable_frames(
                duration_ms=normalized_duration_ms,
                chunk_count=normalized_chunk_count,
            )
            if candidate_probe is None:
                detail = (
                    "sampler.require_readable_frames() returned None during startup "
                    f"probe {attempt}/{normalized_probe_count}"
                )
                raise _build_readiness_error(detail)
            if probe_validator is not None:
                candidate_probe = probe_validator(candidate_probe)
                if candidate_probe is None:
                    raise TypeError("probe_validator returned None")
            last_probe = candidate_probe
        except AudioCaptureReadinessError as exc:
            prior_successes = attempt - 1
            detail = _safe_probe_detail(getattr(exc, "probe", None)) or str(exc)
            contextual_detail = _append_context(
                detail,
                (
                    "startup stability gate failed on probe "
                    f"{attempt}/{normalized_probe_count} after {prior_successes} "
                    f"consecutive readable probe(s); elapsed={_elapsed_ms(started_ns)}ms"
                ),
            )
            raise _build_readiness_error(
                contextual_detail,
                probe=_safe_replace_probe_detail(getattr(exc, "probe", None), contextual_detail),
            ) from exc
        except Exception as exc:
            prior_successes = attempt - 1
            contextual_detail = (
                "unexpected startup capture probe failure: "
                f"{exc!r} on probe {attempt}/{normalized_probe_count} after "
                f"{prior_successes} consecutive readable probe(s); "
                f"elapsed={_elapsed_ms(started_ns)}ms"
            )
            raise _build_readiness_error(
                contextual_detail,
                probe=_safe_replace_probe_detail(last_probe, contextual_detail),
            ) from exc

        if attempt < normalized_probe_count:
            remaining_before_wait_s = _remaining_s(deadline_ns)
            if remaining_before_wait_s is not None and remaining_before_wait_s <= 0.0:
                detail = _append_context(
                    _safe_probe_detail(last_probe),
                    (
                        "startup stability gate timed out after probe "
                        f"{attempt}/{normalized_probe_count} while settling; "
                        f"elapsed={_elapsed_ms(started_ns)}ms "
                        f"(limit {normalized_total_timeout_s:.3f}s)"
                    ),
                )
                raise _build_readiness_error(
                    detail,
                    probe=_safe_replace_probe_detail(last_probe, detail),
                )
            wait_s = normalized_settle_s
            if remaining_before_wait_s is not None:
                wait_s = min(wait_s, remaining_before_wait_s)
            _wait_for_settle(
                delay_s=wait_s,
                cancel_event=cancel_event,
                started_ns=started_ns,
                probe_count=normalized_probe_count,
                last_probe=last_probe,
            )

    if last_probe is None:
        detail = "startup stability gate completed without a probe result"
        raise _build_readiness_error(detail)

    success_detail = _append_context(
        _safe_probe_detail(last_probe),
        (
            "startup stability gate passed with "
            f"{normalized_probe_count}/{normalized_probe_count} consecutive readable probe(s); "
            f"chunk_count={normalized_chunk_count}; elapsed={_elapsed_ms(started_ns)}ms"
        ),
    )
    return _safe_replace_probe_detail(last_probe, success_detail) or last_probe
