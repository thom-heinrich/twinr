"""Resolve calm ReSpeaker indicator semantics from runtime facts.

Twinr does not want the XVF3800 ring to become a second noisy display. This
module defines one conservative contract that other runtime surfaces can share:
the ring may mirror explicit listening or mute state, but it must not twitch on
weak direction or speech evidence.
"""

from __future__ import annotations

from dataclasses import dataclass


_UNAVAILABLE_ALERT_CODES = frozenset(
    {
        "dfu_mode",
        "disconnected",
        "probe_unavailable",
        "capture_unknown",
        "host_control_unavailable",
        "transport_blocked",
        "signal_provider_error",
        "provider_lock_timeout",
    }
)


@dataclass(frozen=True, slots=True)
class ReSpeakerIndicatorState:
    """Describe the calm ring semantics Twinr wants for the XVF3800."""

    semantics: str
    mode: str
    reason: str | None = None
    direction_hint_mirrored: bool = False

    def event_data(self) -> dict[str, object]:
        """Render one JSON-safe indicator payload for ops or automation facts."""

        return {
            "indicator_semantics": self.semantics,
            "indicator_mode": self.mode,
            "indicator_reason": self.reason,
            "indicator_direction_hint_mirrored": self.direction_hint_mirrored,
        }


def resolve_respeaker_indicator_state(
    *,
    runtime_status: str | None,
    runtime_alert_code: str | None,
    mute_active: bool | None,
) -> ReSpeakerIndicatorState:
    """Return the shared calm indicator policy for one runtime observation.

    The current Twinr product decision is conservative:
    - mirror explicit listening
    - mirror explicit mute
    - never mirror direction hints
    - stay idle/off for everything else
    """

    semantics = "listening_mute_only"
    alert_code = _normalize_optional_text(runtime_alert_code)
    status = _normalize_optional_text(runtime_status)
    if alert_code in _UNAVAILABLE_ALERT_CODES:
        return ReSpeakerIndicatorState(
            semantics=semantics,
            mode="off",
            reason="respeaker_unavailable",
        )
    if alert_code == "mic_muted" or mute_active is True:
        return ReSpeakerIndicatorState(
            semantics=semantics,
            mode="muted",
            reason="mic_muted",
        )
    if status == "listening":
        return ReSpeakerIndicatorState(
            semantics=semantics,
            mode="listening",
            reason="runtime_listening",
        )
    return ReSpeakerIndicatorState(
        semantics=semantics,
        mode="idle",
        reason="semantics_idle",
    )


def _normalize_optional_text(value: object) -> str | None:
    text = " ".join(str(value or "").split()).strip()
    return text or None


__all__ = [
    "ReSpeakerIndicatorState",
    "resolve_respeaker_indicator_state",
]
