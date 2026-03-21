"""Build backend-agnostic local wakeword verifier gates from config.

Twinr's second-stage sequence verifier operates on localized audio captures and
is therefore independent from the first-stage detector implementation. The
runtime, eval, and promotion paths should all instantiate the same gate from
config so a new stage-1 backend such as WeKws does not silently bypass the
existing local verifier capability.
"""

from __future__ import annotations

from twinr.agent.base_agent.config import TwinrConfig

from .cascade import WakewordSequenceCaptureVerifier
from .policy import normalize_wakeword_backend

_LOCAL_SEQUENCE_VERIFIER_BACKENDS = frozenset({"openwakeword", "kws", "wekws"})


def build_configured_sequence_capture_verifier(
    config: TwinrConfig,
) -> WakewordSequenceCaptureVerifier | None:
    """Build the configured capture-level sequence verifier when applicable.

    The current verifier asset format remains the same regardless of the
    first-stage detector. Only local streaming detector backends should attach
    this gate; STT-only paths do not produce the detector-localized captures
    that the verifier expects.
    """

    primary_backend = normalize_wakeword_backend(
        getattr(config, "wakeword_primary_backend", config.wakeword_backend),
        default="openwakeword",
    )
    if primary_backend not in _LOCAL_SEQUENCE_VERIFIER_BACKENDS:
        return None
    if not config.wakeword_openwakeword_sequence_verifier_models:
        return None
    return WakewordSequenceCaptureVerifier(
        verifier_models=dict(config.wakeword_openwakeword_sequence_verifier_models),
        threshold=config.wakeword_openwakeword_sequence_verifier_threshold,
    )


__all__ = ["build_configured_sequence_capture_verifier"]
