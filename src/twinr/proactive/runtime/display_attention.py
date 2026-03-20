"""Drive calm HDMI gaze-follow cues from conservative proactive sensor facts.

This module translates the current primary visible-person anchor and, when
available, the current speaker association into small HDMI face cues. Normal
attention-follow stays horizontal-only so stronger "thoughtful" or other
semantic up/down poses remain reserved for explicit face-expression producers.
It mirrors camera-space left/right anchors into user-facing screen gaze, keeps
that producer path separate from the generic runtime snapshot schema, and will
not overwrite active cues owned by other display producers. Matching attention
cues refresh their TTL shortly before expiry, and recent local cues can
briefly resist center jitter or short camera dropouts, so a stationary person
does not make the face drift back to center or blink out between refresh ticks.
Near-center people can still produce a small head turn before the eyes commit
to a full side gaze, which makes the face feel more responsive on the Pi
without reintroducing jittery eye snaps.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
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

from .attention_targeting import MultimodalAttentionTargetSnapshot
from .speaker_association import (
    ReSpeakerSpeakerAssociationSnapshot,
    derive_respeaker_speaker_association,
)


_SOURCE = "proactive_attention_follow"
_LEFT_ENTER_THRESHOLD = 0.44
_RIGHT_ENTER_THRESHOLD = 0.56
_LEFT_EXIT_THRESHOLD = 0.48
_RIGHT_EXIT_THRESHOLD = 0.52
_MIN_REFRESH_INTERVAL_S = 0.2
_MATCHING_CUE_REFRESH_LOOKAHEAD_S = 1.5
_MIN_DIRECTION_HOLD_S = 0.35
_MAX_DIRECTION_HOLD_S = 1.25
_HEAD_SOFT_OFFSET_THRESHOLD = 0.04
_HEAD_STRONG_OFFSET_THRESHOLD = 0.18


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
    camera_center_x: float | None = None
    camera_zone: str | None = None

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
    center_x = None if _coerce_optional_bool(camera.get("primary_person_center_x_unknown")) is True else _coerce_optional_ratio(camera.get("primary_person_center_x"))
    zone = None if _coerce_optional_bool(camera.get("primary_person_zone_unknown")) is True else _normalize_text(camera.get("primary_person_zone")).lower()
    attention_target = MultimodalAttentionTargetSnapshot.from_fact_map(live_facts.get("attention_target"))
    if attention_target is not None and attention_target.active:
        gaze = _gaze_from_attention_target(attention_target)
        if gaze is None:
            return DisplayAttentionCueDecision(reason="no_attention_target")
        brows = _brows_from_attention_target(attention_target, camera)
        mouth = DisplayFaceMouthStyle.SPEAK if attention_target.speaker_locked else None
        head_dx, head_dy = _head_offsets_for_gaze(
            gaze,
            camera_center_x=center_x,
            camera_zone=zone,
        )
        return DisplayAttentionCueDecision(
            active=True,
            reason=attention_target.state,
            gaze=gaze,
            mouth=mouth,
            brows=brows,
            head_dx=head_dx,
            head_dy=head_dy,
            hold_seconds=_hold_seconds_from_config(config),
            speaker_locked=attention_target.speaker_locked,
            camera_center_x=center_x,
            camera_zone=zone,
        )

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
    head_dx, head_dy = _head_offsets_for_gaze(
        gaze,
        camera_center_x=center_x,
        camera_zone=zone,
    )
    return DisplayAttentionCueDecision(
        active=True,
        reason="speaker_visible_person" if speaking_to_visible_person else "visible_person",
        gaze=gaze,
        mouth=mouth,
        brows=brows,
        head_dx=head_dx,
        head_dy=head_dy,
        hold_seconds=_hold_seconds_from_config(config),
        speaker_locked=speaking_to_visible_person,
        camera_center_x=center_x,
        camera_zone=zone,
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
        return self.publish(decision, now=now, config=config)

    def publish(
        self,
        decision: DisplayAttentionCueDecision,
        *,
        now: datetime | None = None,
        config: TwinrConfig | None = None,
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
        decision = _stabilize_inactive_decision(
            config=config,
            decision=decision,
            active_cue=active_cue,
            now=effective_now,
        )
        decision = _stabilize_center_decision(
            config=config,
            decision=decision,
            active_cue=active_cue,
            now=effective_now,
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
            if _matching_cue_needs_refresh(
                cue=active_cue,
                decision=decision,
                now=effective_now,
            ):
                self.controller.show_expression(
                    decision.expression(),
                    source=self.source,
                    hold_seconds=decision.hold_seconds,
                    now=effective_now,
                )
                return DisplayAttentionCuePublishResult(
                    action="refreshed",
                    decision=decision,
                    owner=self.source,
                )
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
        interval_s = float(getattr(config, "display_attention_refresh_interval_s", 0.6) or 0.0)
    except (TypeError, ValueError):
        return 0.6
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


def _gaze_from_attention_target(
    attention_target: MultimodalAttentionTargetSnapshot,
) -> DisplayFaceGazeDirection | None:
    """Map one camera-space attention target to a user-facing display gaze."""

    horizontal = _normalize_text(attention_target.target_horizontal).lower()
    if horizontal not in {"left", "center", "right"}:
        return None
    horizontal = _mirror_camera_horizontal(horizontal)
    vertical = "center"
    return _GAZE_DIRECTION_MAP[(vertical, horizontal)]


def _brows_from_attention_target(
    attention_target: MultimodalAttentionTargetSnapshot,
    camera: Mapping[str, object],
) -> DisplayFaceBrowStyle:
    """Translate one attention-target state into a small brow cue."""

    if attention_target.showing_intent_active:
        return DisplayFaceBrowStyle.RAISED
    if attention_target.speaker_locked or attention_target.session_focus_active:
        return DisplayFaceBrowStyle.SOFT
    engaged = _coerce_optional_bool(camera.get("engaged_with_device")) is True
    looking_toward_device = _coerce_optional_bool(camera.get("looking_toward_device")) is True
    if engaged or looking_toward_device:
        return DisplayFaceBrowStyle.SOFT
    return DisplayFaceBrowStyle.STRAIGHT


def _hold_seconds_from_config(config: TwinrConfig) -> float:
    """Return one bounded hold time for HDMI attention cues."""

    refresh_interval_s = resolve_display_attention_refresh_interval(config)
    return max(
        float(getattr(config, "display_face_cue_ttl_s", 4.0) or 4.0),
        (refresh_interval_s if refresh_interval_s is not None else float(getattr(config, "proactive_capture_interval_s", 6.0) or 6.0))
        + 0.75,
    )


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
    """Map the stabilized camera anchor to one user-facing display gaze."""

    center_x = None if _coerce_optional_bool(camera.get("primary_person_center_x_unknown")) is True else _coerce_optional_ratio(camera.get("primary_person_center_x"))
    zone = None if _coerce_optional_bool(camera.get("primary_person_zone_unknown")) is True else _normalize_text(camera.get("primary_person_zone")).lower()

    horizontal = _horizontal_direction(center_x=center_x, zone=zone)
    if horizontal is None:
        return None
    horizontal = _mirror_camera_horizontal(horizontal)
    vertical = "center"
    return _GAZE_DIRECTION_MAP[(vertical, horizontal)]


def _horizontal_direction(*, center_x: float | None, zone: str | None) -> str | None:
    """Return one coarse left/center/right anchor from center-x or fallback zone."""

    if center_x is not None:
        if center_x <= _LEFT_ENTER_THRESHOLD:
            return "left"
        if center_x >= _RIGHT_ENTER_THRESHOLD:
            return "right"
        return "center"
    if zone in {"left", "center", "right"}:
        return zone
    return None


def _head_offsets_for_gaze(
    gaze: DisplayFaceGazeDirection,
    *,
    camera_center_x: float | None,
    camera_zone: str | None,
) -> tuple[int, int]:
    """Add a bounded head drift so the face turns before full eye commits."""

    gaze_x, gaze_y = gaze.axes()
    horizontal = _head_dx_from_camera(
        camera_center_x=camera_center_x,
        camera_zone=camera_zone,
    )
    if horizontal == 0 and gaze_x:
        horizontal = _sign(gaze_x)
    vertical = 0 if horizontal else _sign(gaze_y)
    return horizontal, vertical


def _mirror_camera_horizontal(horizontal: str) -> str:
    """Translate one camera-space horizontal anchor into screen-facing gaze."""

    if horizontal == "left":
        return "right"
    if horizontal == "right":
        return "left"
    return horizontal


def _direction_hold_seconds(config: TwinrConfig) -> float:
    """Return how long a recent non-center cue may resist center jitter."""

    refresh_interval_s = resolve_display_attention_refresh_interval(config)
    if refresh_interval_s is None:
        try:
            refresh_interval_s = float(getattr(config, "proactive_capture_interval_s", 6.0) or 6.0)
        except (TypeError, ValueError):
            refresh_interval_s = 0.6
    if not math.isfinite(refresh_interval_s) or refresh_interval_s <= 0.0:
        refresh_interval_s = 0.6
    return max(_MIN_DIRECTION_HOLD_S, min(_MAX_DIRECTION_HOLD_S, refresh_interval_s * 1.8))


def _stabilize_center_decision(
    *,
    config: TwinrConfig | None,
    decision: DisplayAttentionCueDecision,
    active_cue: DisplayFaceCue | None,
    now: datetime,
) -> DisplayAttentionCueDecision:
    """Briefly hold a recent non-center cue so camera jitter does not recenter."""

    if config is None or active_cue is None:
        return decision
    if not decision.active or decision.gaze != DisplayFaceGazeDirection.CENTER:
        return decision
    if str(active_cue.source or "").strip() != decision.source:
        return decision
    if active_cue.gaze_x == 0 and active_cue.gaze_y == 0:
        return decision
    updated_at = _parse_timestamp(active_cue.updated_at)
    if updated_at is None:
        return decision
    if not _should_hold_center_direction(
        config=config,
        decision=decision,
        active_cue=active_cue,
        now=now,
        updated_at=updated_at,
    ):
        return decision
    held_gaze = _gaze_from_axes(active_cue.gaze_x, active_cue.gaze_y)
    if held_gaze is None or held_gaze == DisplayFaceGazeDirection.CENTER:
        return decision
    return replace(
        decision,
        reason=f"{decision.reason}_held_direction",
        gaze=held_gaze,
        head_dx=active_cue.head_dx,
        head_dy=active_cue.head_dy,
    )


def _stabilize_inactive_decision(
    *,
    config: TwinrConfig | None,
    decision: DisplayAttentionCueDecision,
    active_cue: DisplayFaceCue | None,
    now: datetime,
) -> DisplayAttentionCueDecision:
    """Briefly hold the latest cue across transient camera dropouts."""

    if config is None or active_cue is None:
        return decision
    if decision.active:
        return decision
    if str(active_cue.source or "").strip() != decision.source:
        return decision
    if decision.reason not in {"no_visible_person", "no_visual_anchor", "no_attention_target"}:
        return decision
    updated_at = _parse_timestamp(active_cue.updated_at)
    if updated_at is None:
        return decision
    if (now - updated_at).total_seconds() > _direction_hold_seconds(config):
        return decision
    return _decision_from_cue(
        active_cue,
        reason=f"{decision.reason}_held_cue",
        hold_seconds=_hold_seconds_from_config(config),
    )


def _gaze_from_axes(gaze_x: int, gaze_y: int) -> DisplayFaceGazeDirection | None:
    """Translate persisted cue axes back into one display gaze direction."""

    return _GAZE_FROM_AXES.get((gaze_x, gaze_y))


def _decision_from_cue(
    cue: DisplayFaceCue,
    *,
    reason: str,
    hold_seconds: float,
) -> DisplayAttentionCueDecision:
    """Reconstruct one decision from an already persisted local cue."""

    gaze = _gaze_from_axes(cue.gaze_x, cue.gaze_y) or DisplayFaceGazeDirection.CENTER
    mouth = None
    if cue.mouth:
        try:
            mouth = DisplayFaceMouthStyle(cue.mouth)
        except ValueError:
            mouth = None
    brows = None
    if cue.brows:
        try:
            brows = DisplayFaceBrowStyle(cue.brows)
        except ValueError:
            brows = None
    return DisplayAttentionCueDecision(
        active=True,
        reason=reason,
        source=str(cue.source or _SOURCE).strip() or _SOURCE,
        gaze=gaze,
        mouth=mouth,
        brows=brows,
        head_dx=cue.head_dx,
        head_dy=cue.head_dy,
        hold_seconds=hold_seconds,
        speaker_locked=cue.mouth == DisplayFaceMouthStyle.SPEAK.value,
    )


def _head_dx_from_camera(
    *,
    camera_center_x: float | None,
    camera_zone: str | None,
) -> int:
    """Return one small user-facing head turn from camera-space position."""

    display_offset = _display_offset_from_camera(
        camera_center_x=camera_center_x,
        camera_zone=camera_zone,
    )
    if display_offset is None:
        return 0
    magnitude = abs(display_offset)
    if magnitude < _HEAD_SOFT_OFFSET_THRESHOLD:
        return 0
    strength = 2 if magnitude >= _HEAD_STRONG_OFFSET_THRESHOLD else 1
    return strength if display_offset > 0.0 else -strength


def _display_offset_from_camera(
    *,
    camera_center_x: float | None,
    camera_zone: str | None,
) -> float | None:
    """Return one user-facing horizontal offset in ``[-0.5, 0.5]``."""

    if camera_center_x is not None:
        return max(-0.5, min(0.5, 0.5 - camera_center_x))
    if camera_zone == "left":
        return 0.25
    if camera_zone == "right":
        return -0.25
    if camera_zone == "center":
        return 0.0
    return None


def _should_hold_center_direction(
    *,
    config: TwinrConfig,
    decision: DisplayAttentionCueDecision,
    active_cue: DisplayFaceCue,
    now: datetime,
    updated_at: datetime,
) -> bool:
    """Return whether one center decision should preserve the recent side cue."""

    if _spatially_matches_active_side(decision=decision, active_cue=active_cue):
        return True
    return (now - updated_at).total_seconds() <= _direction_hold_seconds(config)


def _spatially_matches_active_side(
    *,
    decision: DisplayAttentionCueDecision,
    active_cue: DisplayFaceCue,
) -> bool:
    """Return whether the current camera anchor still sits on the same side."""

    center_x = decision.camera_center_x
    if center_x is not None:
        if active_cue.gaze_x > 0:
            return center_x < _LEFT_EXIT_THRESHOLD
        if active_cue.gaze_x < 0:
            return center_x > _RIGHT_EXIT_THRESHOLD
        return False
    zone = decision.camera_zone
    if active_cue.gaze_x > 0:
        return zone == "left"
    if active_cue.gaze_x < 0:
        return zone == "right"
    return False


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


def _matching_cue_needs_refresh(
    *,
    cue: DisplayFaceCue,
    decision: DisplayAttentionCueDecision,
    now: datetime,
) -> bool:
    """Return whether one unchanged cue should be renewed before expiry."""

    expires_at = _parse_timestamp(cue.expires_at)
    if expires_at is None:
        return False
    seconds_left = (expires_at - now).total_seconds()
    threshold_s = min(float(decision.hold_seconds), _MATCHING_CUE_REFRESH_LOOKAHEAD_S)
    return seconds_left <= max(_MIN_REFRESH_INTERVAL_S, threshold_s)


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


def _parse_timestamp(value: object | None) -> datetime | None:
    """Parse one optional ISO timestamp into an aware UTC datetime."""

    text = _normalize_text(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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

_GAZE_FROM_AXES: dict[tuple[int, int], DisplayFaceGazeDirection] = {
    direction.axes(): direction
    for direction in DisplayFaceGazeDirection
}


__all__ = [
    "DisplayAttentionCueDecision",
    "DisplayAttentionCuePublishResult",
    "DisplayAttentionCuePublisher",
    "display_attention_refresh_supported",
    "derive_display_attention_cue",
    "resolve_display_attention_refresh_interval",
]
