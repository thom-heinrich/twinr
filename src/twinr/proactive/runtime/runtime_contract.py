"""Enforce the ReSpeaker runtime contract before proactive monitoring starts.

This module keeps XVF3800 startup-blocker policy out of the runtime assembly
path. It translates one initial signal snapshot into a small contract decision
so the monitor can fail clearly on unsupported startup states such as DFU mode
without growing more inline conditionals in the orchestration layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from twinr.hardware.respeaker.models import ReSpeakerSignalSnapshot


_HARD_BLOCK_ALERT_CODES = frozenset({"dfu_mode"})


class ReSpeakerRuntimeContractError(RuntimeError):
    """Raised when the XVF3800 startup state violates Twinr's runtime contract."""


@dataclass(frozen=True, slots=True)
class ReSpeakerStartupContractAssessment:
    """Describe whether ReSpeaker startup may proceed for proactive runtime."""

    blocking: bool
    blocker_code: str | None = None
    ops_reason: str | None = None
    detail: str | None = None


def assess_respeaker_monitor_startup_contract(
    signal: ReSpeakerSignalSnapshot,
) -> ReSpeakerStartupContractAssessment:
    """Return whether the initial XVF3800 state blocks monitor startup.

    Twinr currently hard-blocks only the clear DFU/safe-mode state where the
    board enumerates on USB but exposes no ALSA capture device. Other degraded
    transport states remain operator-visible warnings so the runtime can still
    use the fallback observation path when available.
    """

    mode = _normalize_text(signal.device_runtime_mode)
    if mode == "usb_visible_no_capture":
        return ReSpeakerStartupContractAssessment(
            blocking=True,
            blocker_code="dfu_mode",
            ops_reason="respeaker_dfu_mode_blocked",
            detail=(
                "ReSpeaker XVF3800 is visible on USB but has no ALSA capture device. "
                "Twinr refuses to start proactive or wakeword audio against DFU/safe mode. "
                "Reboot or reflash the board into its normal USB-audio runtime first."
            ),
        )
    return ReSpeakerStartupContractAssessment(blocking=False)


def is_respeaker_runtime_hard_block(alert_code: str | None) -> bool:
    """Return whether one operator alert represents a hard runtime blocker."""

    return _normalize_text(alert_code) in _HARD_BLOCK_ALERT_CODES


def _normalize_text(value: object) -> str:
    """Return one compact single-line text value."""

    return " ".join(str(value or "").split()).strip()


__all__ = [
    "ReSpeakerRuntimeContractError",
    "ReSpeakerStartupContractAssessment",
    "assess_respeaker_monitor_startup_contract",
    "is_respeaker_runtime_hard_block",
]
