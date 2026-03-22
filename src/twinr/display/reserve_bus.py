"""Resolve the current owner of Twinr's HDMI reserve area.

The right-hand HDMI area is intentionally separate from the face and header.
Multiple bounded producers may claim it, for example gesture emoji cues or
calmer personality-driven ambient cards. This module keeps that arbitration
out of both the display loop and the scene renderer.

Only one reserve owner may render at a time. The current priority is:

1. gesture / emoji acknowledgements
2. calm ambient reserve cards
3. empty reserve area
"""

from __future__ import annotations

from dataclasses import dataclass

from twinr.display.ambient_impulse_cues import DisplayAmbientImpulseCue
from twinr.display.emoji_cues import DisplayEmojiCue


@dataclass(frozen=True, slots=True)
class DisplayReserveBusState:
    """Describe the currently active owner of the HDMI reserve area.

    Attributes:
        owner: Stable owner token such as ``emoji``, ``ambient_impulse``, or
            ``empty``.
        emoji_cue: Active emoji cue when the reserve is owned by emoji.
        ambient_impulse_cue: Active ambient reserve-card cue when the reserve
            is owned by the ambient companion layer.
        reason: Short auditable explanation for the selected owner.
    """

    owner: str = "empty"
    emoji_cue: DisplayEmojiCue | None = None
    ambient_impulse_cue: DisplayAmbientImpulseCue | None = None
    reason: str = "empty"

    @classmethod
    def empty(cls, *, reason: str = "empty") -> "DisplayReserveBusState":
        """Return one explicit empty reserve state."""

        return cls(owner="empty", reason=reason)

    def signature(self) -> tuple[object, ...]:
        """Return one stable signature fragment for display rerender checks."""

        emoji_signature = self.emoji_cue.signature() if self.emoji_cue is not None else None
        ambient_signature = (
            self.ambient_impulse_cue.signature() if self.ambient_impulse_cue is not None else None
        )
        return (
            self.owner,
            self.reason,
            emoji_signature,
            ambient_signature,
        )


def resolve_display_reserve_bus(
    *,
    emoji_cue: DisplayEmojiCue | None,
    ambient_impulse_cue: DisplayAmbientImpulseCue | None,
) -> DisplayReserveBusState:
    """Resolve the currently visible reserve owner.

    Args:
        emoji_cue: The active emoji acknowledgement cue, if any.
        ambient_impulse_cue: The active ambient reserve-card cue, if any.

    Returns:
        One stable reserve-bus state describing which bounded cue family owns
        the reserve area right now.
    """

    if emoji_cue is not None:
        return DisplayReserveBusState(
            owner="emoji",
            emoji_cue=emoji_cue,
            reason="emoji_surface_owned",
        )
    if ambient_impulse_cue is not None:
        return DisplayReserveBusState(
            owner="ambient_impulse",
            ambient_impulse_cue=ambient_impulse_cue,
            reason="ambient_impulse_active",
        )
    return DisplayReserveBusState.empty()
