"""Bounded startup gating for ReSpeaker capture stability.

Twinr should not treat a briefly enumerated XVF3800 path as healthy startup.
This helper requires several consecutive readable-frame probes before the
runtime leaves startup error state, so transient USB/ALSA enumeration cannot
flip the device back to `ok` too early.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol
import time

from twinr.hardware.audio import AudioCaptureReadinessError, AudioCaptureReadinessProbe

_DEFAULT_STABILITY_PROBE_COUNT = 3
_DEFAULT_STABILITY_SETTLE_S = 0.25


class ReadableFrameProbeSampler(Protocol):
    """Small protocol for startup capture gating."""

    def require_readable_frames(
        self,
        *,
        duration_ms: int | None = None,
        chunk_count: int = 1,
    ) -> AudioCaptureReadinessProbe:
        """Return one bounded readable-frame probe or raise."""


def require_stable_respeaker_capture(
    *,
    sampler: ReadableFrameProbeSampler,
    duration_ms: int,
    probe_count: int = _DEFAULT_STABILITY_PROBE_COUNT,
    settle_s: float = _DEFAULT_STABILITY_SETTLE_S,
) -> AudioCaptureReadinessProbe:
    """Require several consecutive readable-frame probes before startup succeeds."""

    normalized_probe_count = max(1, int(probe_count))
    normalized_settle_s = max(0.0, float(settle_s))
    last_probe: AudioCaptureReadinessProbe | None = None

    for index in range(normalized_probe_count):
        attempt = index + 1
        try:
            last_probe = sampler.require_readable_frames(duration_ms=duration_ms)
        except AudioCaptureReadinessError as exc:
            if attempt <= 1:
                raise
            prior_successes = attempt - 1
            detail = exc.probe.detail or str(exc)
            contextual_detail = (
                f"{detail} (startup stability probe {attempt}/{normalized_probe_count} "
                f"failed after {prior_successes} consecutive readable probe(s))"
            )
            raise AudioCaptureReadinessError(
                contextual_detail,
                probe=replace(exc.probe, detail=contextual_detail),
            ) from exc
        if attempt < normalized_probe_count and normalized_settle_s > 0.0:
            time.sleep(normalized_settle_s)

    if last_probe is None:
        raise RuntimeError("ReSpeaker startup capture gate completed without a probe result")
    return last_probe
