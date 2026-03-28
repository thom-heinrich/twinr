# CHANGELOG: 2026-03-28
# BUG-1: Fail fast on impossible owner/cue combinations instead of silently accepting corrupted reserve states.
# BUG-2: Snapshot and harden cue signatures so rerender checks stay stable and producer signature failures do not crash arbitration.
# SEC-1: Sanitize and bound audit reasons and reject malformed in-process states; no practical script-local remote exploit primitive was found here.
# IMP-1: Replace free-form owner tokens with StrEnum-backed finite variants while preserving string compatibility for existing callers.
# IMP-2: Exclude non-visual audit metadata from the render signature and add explicit owner constructors plus active-cue accessors.
# mypy: enable-error-code=exhaustive-match

"""Resolve the current owner of Twinr's HDMI reserve area.

The right-hand HDMI area is intentionally separate from the face and header.
Multiple bounded producers may claim it, for example gesture emoji cues or
calmer personality-driven ambient cards. This module keeps that arbitration
out of both the display loop and the scene renderer.

Only one reserve owner may render at a time. The current priority is:

1. service-connect flows that need a QR or pairing state
2. gesture / emoji acknowledgements
3. calm ambient reserve cards
4. empty reserve area
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum, unique
import logging
from threading import Lock
from typing import Final, assert_never

from twinr.display.ambient_impulse_cues import DisplayAmbientImpulseCue
from twinr.display.emoji_cues import DisplayEmojiCue
from twinr.display.service_connect_cues import DisplayServiceConnectCue

LOGGER = logging.getLogger(__name__)

_REASON_MAX_CHARS: Final[int] = 96
_SIGNATURE_ERROR: Final[str] = "signature_error"
_REPORTED_SIGNATURE_ISSUES: set[tuple[str, str, str]] = set()
_REPORTED_SIGNATURE_ISSUES_LOCK = Lock()


@unique
class DisplayReserveOwner(StrEnum):
    """Stable owner token for the HDMI reserve area."""

    SERVICE_CONNECT = "service_connect"
    EMOJI = "emoji"
    AMBIENT_IMPULSE = "ambient_impulse"
    EMPTY = "empty"


@unique
class DisplayReserveReason(StrEnum):
    """Canonical built-in reasons emitted by this resolver."""

    SERVICE_CONNECT_ACTIVE = "service_connect_active"
    EMOJI_SURFACE_OWNED = "emoji_surface_owned"
    AMBIENT_IMPULSE_ACTIVE = "ambient_impulse_active"
    EMPTY = "empty"


def _sanitize_reason(reason: str | None) -> str:
    """Return one bounded printable reason token."""

    if reason is None:
        return DisplayReserveReason.EMPTY.value
    if not isinstance(reason, str):
        reason = str(reason)

    normalized = "".join(
        character if character.isprintable() and character not in "\r\n\t" else " "
        for character in reason
    ).strip()

    if not normalized:
        return DisplayReserveReason.EMPTY.value
    if len(normalized) > _REASON_MAX_CHARS:
        normalized = normalized[:_REASON_MAX_CHARS]
    return normalized


def _freeze_signature_fragment(value: object) -> object:
    """Return one recursively immutable signature fragment where possible."""

    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return tuple(_freeze_signature_fragment(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_signature_fragment(item) for item in value)
    if isinstance(value, dict):
        frozen_items = tuple(
            (_freeze_signature_fragment(key), _freeze_signature_fragment(item_value))
            for key, item_value in value.items()
        )
        return tuple(sorted(frozen_items, key=repr))
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_signature_fragment(item) for item in value), key=repr))
    return value


def _log_signature_issue_once(
    *, cue_label: str, cue_type: str, problem: str, exc: Exception | None = None
) -> None:
    """Emit at most one log line per distinct signature failure mode."""

    issue_key = (cue_label, cue_type, problem)

    with _REPORTED_SIGNATURE_ISSUES_LOCK:
        if issue_key in _REPORTED_SIGNATURE_ISSUES:
            return
        _REPORTED_SIGNATURE_ISSUES.add(issue_key)

    if exc is None:
        LOGGER.error(
            "Display reserve %s cue %s cannot provide a usable signature: %s",
            cue_label,
            cue_type,
            problem,
        )
        return

    LOGGER.error(
        "Display reserve %s cue %s signature() failed with %s",
        cue_label,
        cue_type,
        problem,
        exc_info=(type(exc), exc, exc.__traceback__),
    )


def _snapshot_cue_signature(cue: object | None, *, cue_label: str) -> object | None:
    """Return one stable cue-signature snapshot that never raises."""

    if cue is None:
        return None

    signature = getattr(cue, "signature", None)
    cue_type = type(cue).__name__

    if not callable(signature):
        _log_signature_issue_once(
            cue_label=cue_label,
            cue_type=cue_type,
            problem="missing_signature",
        )
        return (_SIGNATURE_ERROR, cue_label, cue_type, id(cue), "missing_signature")

    try:
        return _freeze_signature_fragment(signature())
    except Exception as exc:  # noqa: BLE001 - producer bugs must not take down the reserve bus
        problem = exc.__class__.__name__
        _log_signature_issue_once(
            cue_label=cue_label,
            cue_type=cue_type,
            problem=problem,
            exc=exc,
        )
        return (_SIGNATURE_ERROR, cue_label, cue_type, id(cue), problem)


@dataclass(frozen=True, slots=True)
class DisplayReserveBusState:
    """Describe the currently active owner of the HDMI reserve area.

    Attributes:
        owner: Stable owner token such as ``service_connect``, ``emoji``,
            ``ambient_impulse``, or ``empty``.
        service_connect_cue: Active service-connect cue when the reserve is
            owned by a pairing flow.
        emoji_cue: Active emoji cue when the reserve is owned by emoji.
        ambient_impulse_cue: Active ambient reserve-card cue when the reserve
            is owned by the ambient companion layer.
        reason: Short auditable explanation for the selected owner.
    """

    owner: DisplayReserveOwner | str = DisplayReserveOwner.EMPTY
    service_connect_cue: DisplayServiceConnectCue | None = None
    emoji_cue: DisplayEmojiCue | None = None
    ambient_impulse_cue: DisplayAmbientImpulseCue | None = None
    reason: str = DisplayReserveReason.EMPTY
    _signature_fragment: tuple[object, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            owner = DisplayReserveOwner(self.owner)
        except ValueError as exc:
            # BREAKING: invalid owner tokens now fail fast instead of flowing
            # downstream as undefined reserve-bus states.
            raise ValueError(f"unknown display reserve owner: {self.owner!r}") from exc

        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "reason", _sanitize_reason(self.reason))

        self._validate_owner_invariants(owner)

        active_signature = self._build_active_signature(owner)
        object.__setattr__(self, "_signature_fragment", (owner.value, active_signature))

    def _validate_owner_invariants(self, owner: DisplayReserveOwner) -> None:
        match owner:
            case DisplayReserveOwner.SERVICE_CONNECT:
                if (
                    self.service_connect_cue is None
                    or self.emoji_cue is not None
                    or self.ambient_impulse_cue is not None
                ):
                    # BREAKING: impossible reserve states now raise immediately
                    # instead of being silently accepted.
                    raise ValueError(
                        "service_connect owner requires exactly one service_connect cue"
                    )
            case DisplayReserveOwner.EMOJI:
                if (
                    self.emoji_cue is None
                    or self.service_connect_cue is not None
                    or self.ambient_impulse_cue is not None
                ):
                    # BREAKING: impossible reserve states now raise immediately
                    # instead of being silently accepted.
                    raise ValueError("emoji owner requires exactly one emoji cue")
            case DisplayReserveOwner.AMBIENT_IMPULSE:
                if (
                    self.ambient_impulse_cue is None
                    or self.service_connect_cue is not None
                    or self.emoji_cue is not None
                ):
                    # BREAKING: impossible reserve states now raise immediately
                    # instead of being silently accepted.
                    raise ValueError(
                        "ambient_impulse owner requires exactly one ambient_impulse cue"
                    )
            case DisplayReserveOwner.EMPTY:
                if (
                    self.service_connect_cue is not None
                    or self.emoji_cue is not None
                    or self.ambient_impulse_cue is not None
                ):
                    # BREAKING: impossible reserve states now raise immediately
                    # instead of being silently accepted.
                    raise ValueError("empty owner may not carry active reserve cues")
            case _ as unreachable:
                assert_never(unreachable)

    def _build_active_signature(self, owner: DisplayReserveOwner) -> object | None:
        match owner:
            case DisplayReserveOwner.SERVICE_CONNECT:
                return _snapshot_cue_signature(
                    self.service_connect_cue,
                    cue_label=DisplayReserveOwner.SERVICE_CONNECT.value,
                )
            case DisplayReserveOwner.EMOJI:
                return _snapshot_cue_signature(
                    self.emoji_cue,
                    cue_label=DisplayReserveOwner.EMOJI.value,
                )
            case DisplayReserveOwner.AMBIENT_IMPULSE:
                return _snapshot_cue_signature(
                    self.ambient_impulse_cue,
                    cue_label=DisplayReserveOwner.AMBIENT_IMPULSE.value,
                )
            case DisplayReserveOwner.EMPTY:
                return None
            case _ as unreachable:
                assert_never(unreachable)

    @classmethod
    def empty(cls, *, reason: str = DisplayReserveReason.EMPTY) -> "DisplayReserveBusState":
        """Return one explicit empty reserve state."""

        return cls(owner=DisplayReserveOwner.EMPTY, reason=reason)

    @classmethod
    def from_service_connect(
        cls,
        cue: DisplayServiceConnectCue,
        *,
        reason: str = DisplayReserveReason.SERVICE_CONNECT_ACTIVE,
    ) -> "DisplayReserveBusState":
        """Return one service-connect-owned reserve state."""

        return cls(
            owner=DisplayReserveOwner.SERVICE_CONNECT,
            service_connect_cue=cue,
            reason=reason,
        )

    @classmethod
    def from_emoji(
        cls,
        cue: DisplayEmojiCue,
        *,
        reason: str = DisplayReserveReason.EMOJI_SURFACE_OWNED,
    ) -> "DisplayReserveBusState":
        """Return one emoji-owned reserve state."""

        return cls(
            owner=DisplayReserveOwner.EMOJI,
            emoji_cue=cue,
            reason=reason,
        )

    @classmethod
    def from_ambient_impulse(
        cls,
        cue: DisplayAmbientImpulseCue,
        *,
        reason: str = DisplayReserveReason.AMBIENT_IMPULSE_ACTIVE,
    ) -> "DisplayReserveBusState":
        """Return one ambient-owned reserve state."""

        return cls(
            owner=DisplayReserveOwner.AMBIENT_IMPULSE,
            ambient_impulse_cue=cue,
            reason=reason,
        )

    @property
    def active_cue(
        self,
    ) -> DisplayServiceConnectCue | DisplayEmojiCue | DisplayAmbientImpulseCue | None:
        """Return the sole active cue for the selected owner."""

        match self.owner:
            case DisplayReserveOwner.SERVICE_CONNECT:
                return self.service_connect_cue
            case DisplayReserveOwner.EMOJI:
                return self.emoji_cue
            case DisplayReserveOwner.AMBIENT_IMPULSE:
                return self.ambient_impulse_cue
            case DisplayReserveOwner.EMPTY:
                return None
            case _ as unreachable:
                assert_never(unreachable)

    def signature(self) -> tuple[object, ...]:
        """Return one stable render signature for display rerender checks.

        The signature intentionally excludes ``reason`` because the reason is
        audit metadata and should not trigger visual rerenders on its own.
        """

        return self._signature_fragment


def resolve_display_reserve_bus(
    *,
    service_connect_cue: DisplayServiceConnectCue | None,
    emoji_cue: DisplayEmojiCue | None,
    ambient_impulse_cue: DisplayAmbientImpulseCue | None,
) -> DisplayReserveBusState:
    """Resolve the currently visible reserve owner.

    Args:
        service_connect_cue: The active service-connect cue, if any.
        emoji_cue: The active emoji acknowledgement cue, if any.
        ambient_impulse_cue: The active ambient reserve-card cue, if any.

    Returns:
        One stable reserve-bus state describing which bounded cue family owns
        the reserve area right now.
    """

    if service_connect_cue is not None:
        return DisplayReserveBusState.from_service_connect(service_connect_cue)
    if emoji_cue is not None:
        return DisplayReserveBusState.from_emoji(emoji_cue)
    if ambient_impulse_cue is not None:
        return DisplayReserveBusState.from_ambient_impulse(ambient_impulse_cue)
    return DisplayReserveBusState.empty()