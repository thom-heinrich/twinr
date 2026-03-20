"""Drive calm HDMI gaze-follow cues from conservative proactive sensor facts.

This module translates the current primary visible-person anchor and, when
available, the current speaker association into small HDMI face cues. It keeps
that producer path separate from the generic runtime snapshot schema and will
not overwrite active cues owned by other display producers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import math

from twinr.agent.base_agent.config import TwinrConfig
from twinr.display.face_cues import DisplayFaceCue, DisplayFaceCueStore
from twinr.display.face_expressions import (
    DisplayFaceBrowStyle,
    DisplayFaceExpression,
    DisplayFaceExpressionController,
    DisplayFaceGazeDirection,
    DisplayFaceMouthStyle,
)

from .speaker_association import (
    ReSpeakerSpeakerAssociationSnapshot,
    derive_respeaker_speaker_association,
)


_SOURCE = "proactive_attention_follow"
_LEFT_THRESHOLD = 0.36
_RIGHT_THRESHOLD = 0.64
_UP_THRESHOLD = 0.38
_DOWN_THRESHOLD = 0.72
_MIN_REFRESH_INTERVAL_S = 0.2


@dataclass(frozen=True, slots=True)
class DisplayAttentionCueDecision:
    """Describe one conservative HDMI attention-follow decision."""

    active: bool = False
    reason: str = "inactive"
    source: str = _SOURCE
    gaze: DisplayFaceGazeDirection = DisplayFaceGazeDirection.CENTER
    mouth: DisplayFaceMouthStyle | None = None
    brows: DisplayFaceBrowStyle | None = None
    head_dx: int = 0
    head_dy: int = 0
    hold_seconds: float = 0.0
    speaker_locked: bool = False

    def expression(self) -> DisplayFaceExpression:
        """Translate the decision into one producer-facing display expression."""

        return DisplayFaceExpression(
            gaze=self.gaze,
            mouth=self.mouth,
            brows=self.brows,
            head_dx=self.head_dx,
            head_dy=self.head_dy,
        )


@dataclass(frozen=True, slots=True)
class DisplayAttentionCuePublishResult:
    """Summarize one publish attempt for tests and bounded telemetry."""

    action: str
    decision: DisplayAttentionCueDecision
    owner: str | None = None


def derive_display_attention_cue(
    *,
    config: TwinrConfig,
    live_facts: Mapping[str, object],
) -> DisplayAttentionCueDecision:
    """Derive one calm HDMI gaze-follow cue from live proactive facts."""

    camera = _coerce_mapping(live_facts.get("camera"))
    if _coerce_optional_bool(camera.get("person_visible")) is not True:
        return DisplayAttentionCueDecision(reason="no_visible_person")

    gaze = _gaze_from_camera(camera)
    if gaze is None:
        return DisplayAttentionCueDecision(reason="no_visual_anchor")

    speaker_association = _speaker_association_from_facts(live_facts)
    speaking_to_visible_person = (
        _coerce_optional_bool(_coerce_mapping(live_facts.get("vad")).get("speech_detected")) is True
        and speaker_association.associated
    )
    engaged = _coerce_optional_bool(camera.get("engaged_with_device")) is True
    looking_toward_device = _coerce_optional_bool(camera.get("looking_toward_device")) is True
    showing_intent = _coerce_optional_bool(camera.get("showing_intent_likely")) is True

    brows = DisplayFaceBrowStyle.STRAIGHT
    if engaged or looking_toward_device:
        brows = DisplayFaceBrowStyle.SOFT
    if showing_intent and not speaking_to_visible_person:
        brows = DisplayFaceBrowStyle.RAISED

    mouth = DisplayFaceMouthStyle.SPEAK if speaking_to_visible_person else None
    head_dx, head_dy = _head_offsets_for_gaze(gaze)
    refresh_interval_s = resolve_display_attention_refresh_interval(config)
    hold_seconds = max(
        float(getattr(config, "display_face_cue_ttl_s", 4.0) or 4.0),
        (refresh_interval_s if refresh_interval_s is not None else float(getattr(config, "proactive_capture_interval_s", 6.0) or 6.0))
        + 0.75,
    )
    return DisplayAttentionCueDecision(
        active=True,
        reason="speaker_visible_person" if speaking_to_visible_person else "visible_person",
        gaze=gaze,
        mouth=mouth,
        brows=brows,
        head_dx=head_dx,
        head_dy=head_dy,
        hold_seconds=hold_seconds,
        speaker_locked=speaking_to_visible_person,
    )


@dataclass(slots=True)
class DisplayAttentionCuePublisher:
    """Persist conservative HDMI attention-follow cues without stomping others."""

    controller: DisplayFaceExpressionController
    source: str = _SOURCE

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayAttentionCuePublisher":
        """Build one publisher from the configured HDMI face cue store."""

        return cls(
            controller=DisplayFaceExpressionController.from_config(
                config,
                default_source=_SOURCE,
            )
        )

    @property
    def store(self) -> DisplayFaceCueStore:
        """Expose the underlying cue store for tests and ownership checks."""

        return self.controller.store

    def publish_from_facts(
        self,
        *,
        config: TwinrConfig,
        live_facts: Mapping[str, object],
        now: datetime | None = None,
    ) -> DisplayAttentionCuePublishResult:
        """Derive and publish one attention-follow cue from live facts."""

        decision = derive_display_attention_cue(config=config, live_facts=live_facts)
        return self.publish(decision, now=now)

    def publish(
        self,
        decision: DisplayAttentionCueDecision,
        *,
        now: datetime | None = None,
    ) -> DisplayAttentionCuePublishResult:
        """Persist or clear one derived attention-follow cue."""

        effective_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        active_cue = self.store.load_active(now=effective_now)
        active_owner = None if active_cue is None else str(active_cue.source or "").strip() or None
        if active_owner is not None and active_owner != self.source:
            return DisplayAttentionCuePublishResult(
                action="blocked_foreign_cue",
                decision=decision,
                owner=active_owner,
            )
        if not decision.active:
            if active_owner == self.source:
                self.controller.clear()
                return DisplayAttentionCuePublishResult(
                    action="cleared",
                    decision=decision,
                    owner=active_owner,
                )
            return DisplayAttentionCuePublishResult(
                action="inactive",
                decision=decision,
                owner=active_owner,
            )

        if active_cue is not None and _cue_matches_decision(active_cue, decision):
            return DisplayAttentionCuePublishResult(
                action="unchanged",
                decision=decision,
                owner=active_owner,
            )

        self.controller.show_expression(
            decision.expression(),
            source=self.source,
            hold_seconds=decision.hold_seconds,
            now=effective_now,
        )
        return DisplayAttentionCuePublishResult(
            action="updated",
            decision=decision,
            owner=self.source,
        )


def resolve_display_attention_refresh_interval(config: TwinrConfig) -> float | None:
    """Return the bounded local refresh cadence for HDMI attention-follow."""

    try:
        interval_s = float(getattr(config, "display_attention_refresh_interval_s", 1.25) or 0.0)
    except (TypeError, ValueError):
        return 1.25
    if not math.isfinite(interval_s) or interval_s <= 0.0:
        return None
    return max(_MIN_REFRESH_INTERVAL_S, interval_s)


def display_attention_refresh_supported(
    *,
    config: TwinrConfig,
    vision_observer: object | None,
) -> bool:
    """Return whether a bounded fast attention-refresh path is safe to run."""

    if resolve_display_attention_refresh_interval(config) is None:
        return False
    display_driver = str(getattr(config, "display_driver", "") or "").strip().lower()
    if not display_driver.startswith("hdmi"):
        return False
    return getattr(vision_observer, "supports_attention_refresh", False) is True


def _speaker_association_from_facts(
    live_facts: Mapping[str, object],
) -> ReSpeakerSpeakerAssociationSnapshot:
    """Return the current speaker-association snapshot, deriving it when needed."""

    snapshot = ReSpeakerSpeakerAssociationSnapshot.from_fact_map(live_facts.get("speaker_association"))
    if snapshot is not None:
        return snapshot
    return derive_respeaker_speaker_association(
        observed_at=None,
        live_facts=live_facts,
    )


def _gaze_from_camera(camera: Mapping[str, object]) -> DisplayFaceGazeDirection | None:
    """Map the stabilized primary visible-person anchor to one discrete gaze."""

    center_x = None if _coerce_optional_bool(camera.get("primary_person_center_x_unknown")) is True else _coerce_optional_ratio(camera.get("primary_person_center_x"))
    center_y = None if _coerce_optional_bool(camera.get("primary_person_center_y_unknown")) is True else _coerce_optional_ratio(camera.get("primary_person_center_y"))
    zone = None if _coerce_optional_bool(camera.get("primary_person_zone_unknown")) is True else _normalize_text(camera.get("primary_person_zone")).lower()

    horizontal = _horizontal_direction(center_x=center_x, zone=zone)
    vertical = _vertical_direction(center_y)
    if horizontal is None and vertical is None:
        return None
    if horizontal is None:
        horizontal = "center"
    if vertical is None:
        vertical = "center"
    return _GAZE_DIRECTION_MAP[(vertical, horizontal)]


def _horizontal_direction(*, center_x: float | None, zone: str | None) -> str | None:
    """Return one coarse left/center/right anchor from center-x or fallback zone."""

    if center_x is not None:
        if center_x <= _LEFT_THRESHOLD:
            return "left"
        if center_x >= _RIGHT_THRESHOLD:
            return "right"
        return "center"
    if zone in {"left", "center", "right"}:
        return zone
    return None


def _vertical_direction(center_y: float | None) -> str | None:
    """Return one coarse up/center/down anchor from center-y when available."""

    if center_y is None:
        return None
    if center_y <= _UP_THRESHOLD:
        return "up"
    if center_y >= _DOWN_THRESHOLD:
        return "down"
    return "center"


def _head_offsets_for_gaze(gaze: DisplayFaceGazeDirection) -> tuple[int, int]:
    """Add a tiny head drift so the face appears to turn with the eyes."""

    gaze_x, gaze_y = gaze.axes()
    return _sign(gaze_x), _sign(gaze_y)


def _cue_matches_decision(cue: DisplayFaceCue, decision: DisplayAttentionCueDecision) -> bool:
    """Return whether one active cue already represents the desired decision."""

    expression = decision.expression().to_cue(source=decision.source)
    return (
        cue.source == decision.source
        and cue.gaze_x == expression.gaze_x
        and cue.gaze_y == expression.gaze_y
        and cue.mouth == expression.mouth
        and cue.brows == expression.brows
        and cue.head_dx == expression.head_dx
        and cue.head_dy == expression.head_dy
    )


def _sign(value: int) -> int:
    """Collapse one signed gaze axis into a tiny head-offset step."""

    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _normalize_text(value: object | None) -> str:
    """Normalize one optional token-like value into compact text."""

    return " ".join(str(value or "").split()).strip()


def _coerce_mapping(value: object | None) -> dict[str, object]:
    """Normalize one optional mapping-like object into a plain dictionary."""

    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    try:
        return dict(value or {})
    except (TypeError, ValueError):
        return {}


def _coerce_optional_bool(value: object | None) -> bool | None:
    """Parse one optional conservative boolean token."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = _normalize_text(value).lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    return None


def _coerce_optional_ratio(value: object | None) -> float | None:
    """Parse one optional ratio in ``[0.0, 1.0]``."""

    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return max(0.0, min(1.0, numeric))


_GAZE_DIRECTION_MAP: dict[tuple[str, str], DisplayFaceGazeDirection] = {
    ("up", "left"): DisplayFaceGazeDirection.UP_LEFT,
    ("up", "center"): DisplayFaceGazeDirection.UP,
    ("up", "right"): DisplayFaceGazeDirection.UP_RIGHT,
    ("center", "left"): DisplayFaceGazeDirection.LEFT,
    ("center", "center"): DisplayFaceGazeDirection.CENTER,
    ("center", "right"): DisplayFaceGazeDirection.RIGHT,
    ("down", "left"): DisplayFaceGazeDirection.DOWN_LEFT,
    ("down", "center"): DisplayFaceGazeDirection.DOWN,
    ("down", "right"): DisplayFaceGazeDirection.DOWN_RIGHT,
}


__all__ = [
    "DisplayAttentionCueDecision",
    "DisplayAttentionCuePublishResult",
    "DisplayAttentionCuePublisher",
    "display_attention_refresh_supported",
    "derive_display_attention_cue",
    "resolve_display_attention_refresh_interval",
]
