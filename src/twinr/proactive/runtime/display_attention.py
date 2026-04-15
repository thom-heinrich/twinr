# CHANGELOG: 2026-03-29
# BUG-1: Fixed cue-vs-gaze divergence when attention_target only exposes coarse horizontal/vertical labels.
# BUG-2: Fixed center/dropout stabilization losing soft local head-turn cues (e.g. gaze_x=±1) during rehydration.
# BUG-3: Fixed runtime crashes from invalid numeric config values in hold/refresh timing paths.
# BUG-4: Reduced producer-stomp race by re-checking cue ownership immediately before mutating the shared store.
# BUG-5: Disabled continuous hdmi_wayland face-cue publishing because sustained live Wayland rerenders still
#        drive Pi runtime RSS into memory-pressure territory even after TTL-only cue refresh churn was removed.
# BUG-6: Disabled the entire hdmi_wayland attention-refresh lane because Pi evidence showed the hidden
#        camera/servo path still pushes the streaming runtime into memory-pressure territory there.
# BUG-7: Restored the hdmi_wayland attention-refresh fail-close after a regression drifted the hidden
#        camera/servo lane back into the transcript-first voice runtime and reintroduced Pi backpressure.
# SEC-1: Hardened shared-store reads by bounding persisted cue axes/head offsets before reuse, limiting malformed or tampered cue payload impact.
# IMP-1: Added low-latency adaptive 1€ smoothing for camera / attention-target coordinates in the publisher path (Pi 4 friendly, no extra dependency).
# IMP-2: Added optional freshness/confidence gating for sensor facts to ignore stale or weak anchors instead of rendering outdated attention.
# IMP-3: Added direction-consistent fallback cue synthesis for coarse target labels and stronger explicit directional cues for older-adult readability.

"""Drive calm HDMI gaze-follow cues from conservative proactive sensor facts.

This module translates the current primary visible-person anchor and, when
available, the current speaker association into small HDMI face cues. Normal
attention-follow uses bounded vertical drift only for clearly above/below
camera people, so stronger "thoughtful" or other semantic up/down poses remain
reserved for explicit face-expression producers.
It mirrors camera-space left/right anchors into user-facing screen gaze, keeps
that producer path separate from the generic runtime snapshot schema, and will
not overwrite active cues owned by other display producers. Matching attention
cues refresh their TTL shortly before expiry, and recent local cues can
briefly resist center jitter or short camera dropouts, so a stationary person
does not make the face drift back to center or blink out between refresh ticks.
Near-center people can still produce a small head turn before the eyes commit
to a full side gaze, which makes the face feel more responsive on the Pi
without reintroducing jittery eye snaps.

2026 upgrade notes:
- Directional fallbacks from multimodal attention targets now produce
  directionally consistent persisted cues, even when the upstream module only
  exposes coarse left/right/up/down labels.
- The publisher path can optionally apply an adaptive 1€ filter to numeric
  anchors. This reduces idle jitter without adding the lag penalty of a simple
  fixed low-pass filter.
- Freshness and confidence gates can suppress stale or weak anchors when the
  upstream fact maps include timestamps and confidence values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
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
_UP_ENTER_THRESHOLD = 0.38
_DOWN_ENTER_THRESHOLD = 0.62

_MIN_REFRESH_INTERVAL_S = 0.2
_MATCHING_CUE_REFRESH_LOOKAHEAD_S = 1.5
_MIN_DIRECTION_HOLD_S = 0.35
_MAX_DIRECTION_HOLD_S = 1.25

_MAX_DYNAMIC_GAZE_AXIS = 3
_MAX_DYNAMIC_HEAD_AXIS = 2
_MIN_EXPLICIT_DIRECTION_AXIS = 2

_GAZE_SOFT_OFFSET_THRESHOLD = 0.025
_GAZE_STRONG_OFFSET_THRESHOLD = 0.13
_GAZE_VERTICAL_SOFT_OFFSET_THRESHOLD = 0.08
_GAZE_VERTICAL_STRONG_OFFSET_THRESHOLD = 0.16
_HEAD_SOFT_OFFSET_THRESHOLD = 0.04
_HEAD_STRONG_OFFSET_THRESHOLD = 0.18
_HEAD_VERTICAL_SOFT_OFFSET_THRESHOLD = 0.09
_HEAD_VERTICAL_STRONG_OFFSET_THRESHOLD = 0.2

_DEFAULT_FACT_MAX_AGE_S = 1.25
_DEFAULT_CAMERA_MAX_AGE_S = 1.25
_DEFAULT_TARGET_MAX_AGE_S = 1.25
_DEFAULT_CAMERA_MIN_CONFIDENCE = 0.35
_DEFAULT_TARGET_MIN_CONFIDENCE = 0.35

_DEFAULT_ONE_EURO_MIN_CUTOFF_HZ = 1.15
_DEFAULT_ONE_EURO_BETA = 0.18
_DEFAULT_ONE_EURO_DERIVATIVE_CUTOFF_HZ = 1.0
_MAX_FILTER_GAP_S = 1.5

_MAX_SOURCE_LENGTH = 96


@dataclass(frozen=True, slots=True)
class DisplayAttentionCueDecision:
    """Describe one conservative HDMI attention-follow decision."""

    active: bool = False
    reason: str = "inactive"
    source: str = _SOURCE
    gaze: DisplayFaceGazeDirection = DisplayFaceGazeDirection.CENTER
    cue_gaze_x: int = 0
    cue_gaze_y: int = 0
    mouth: DisplayFaceMouthStyle | None = None
    brows: DisplayFaceBrowStyle | None = None
    head_dx: int = 0
    head_dy: int = 0
    hold_seconds: float = 0.0
    speaker_locked: bool = False
    camera_center_x: float | None = None
    camera_zone: str | None = None
    observed_at: datetime | None = None
    confidence: float | None = None
    smoothed: bool = False

    def expression(self) -> DisplayFaceExpression:
        """Translate the decision into one producer-facing display expression."""

        return DisplayFaceExpression(
            gaze=self.gaze,
            mouth=self.mouth,
            brows=self.brows,
            head_dx=_clamp_int(self.head_dx, -_MAX_DYNAMIC_HEAD_AXIS, _MAX_DYNAMIC_HEAD_AXIS),
            head_dy=_clamp_int(self.head_dy, -_MAX_DYNAMIC_HEAD_AXIS, _MAX_DYNAMIC_HEAD_AXIS),
        )

    def cue(self) -> DisplayFaceCue:
        """Translate the decision into one persisted HDMI face cue."""

        return DisplayFaceCue(
            source=_normalize_source(self.source),
            gaze_x=_clamp_int(self.cue_gaze_x, -_MAX_DYNAMIC_GAZE_AXIS, _MAX_DYNAMIC_GAZE_AXIS),
            gaze_y=_clamp_int(self.cue_gaze_y, -_MAX_DYNAMIC_GAZE_AXIS, _MAX_DYNAMIC_GAZE_AXIS),
            head_dx=_clamp_int(self.head_dx, -_MAX_DYNAMIC_HEAD_AXIS, _MAX_DYNAMIC_HEAD_AXIS),
            head_dy=_clamp_int(self.head_dy, -_MAX_DYNAMIC_HEAD_AXIS, _MAX_DYNAMIC_HEAD_AXIS),
            mouth=(None if self.mouth is None else self.mouth.value),
            brows=(None if self.brows is None else self.brows.value),
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
    now: datetime | None = None,
) -> DisplayAttentionCueDecision:
    """Derive one calm HDMI gaze-follow cue from live proactive facts."""

    effective_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    camera = _coerce_mapping(live_facts.get("camera"))
    attention_target_map = _coerce_mapping(live_facts.get("attention_target"))
    person_visible = _coerce_optional_bool(camera.get("person_visible")) is True

    camera_observed_at = _fact_timestamp(camera)
    camera_confidence = _camera_confidence(camera)
    camera_fresh = _fact_is_fresh(
        config=config,
        observed_at=camera_observed_at,
        now=effective_now,
        config_attr="display_attention_camera_max_age_s",
        default=_DEFAULT_CAMERA_MAX_AGE_S,
    )
    camera_confident = _confidence_meets_minimum(
        value=camera_confidence,
        minimum=_config_float(
            config,
            "display_attention_camera_min_confidence",
            _DEFAULT_CAMERA_MIN_CONFIDENCE,
            minimum=0.0,
            maximum=1.0,
        ),
    )
    if _camera_attention_unavailable(camera) and not person_visible:
        return DisplayAttentionCueDecision(
            reason="no_visible_person",
            observed_at=camera_observed_at,
            confidence=camera_confidence,
        )

    center_x = None
    if (
        camera_fresh
        and camera_confident
        and _coerce_optional_bool(camera.get("primary_person_center_x_unknown")) is not True
    ):
        center_x = _coerce_optional_ratio(camera.get("primary_person_center_x"))
    zone = None
    if camera_fresh and camera_confident and _coerce_optional_bool(camera.get("primary_person_zone_unknown")) is not True:
        zone = _normalize_text(camera.get("primary_person_zone")).lower() or None
    center_y = None
    if (
        camera_fresh
        and camera_confident
        and _coerce_optional_bool(camera.get("primary_person_center_y_unknown")) is not True
    ):
        center_y = _coerce_optional_ratio(camera.get("primary_person_center_y"))

    attention_target = MultimodalAttentionTargetSnapshot.from_fact_map(live_facts.get("attention_target"))
    attention_observed_at = _fact_timestamp(attention_target_map)
    attention_confidence = _attention_target_confidence(attention_target_map)
    attention_fresh = _fact_is_fresh(
        config=config,
        observed_at=attention_observed_at,
        now=effective_now,
        config_attr="display_attention_target_max_age_s",
        default=_DEFAULT_TARGET_MAX_AGE_S,
    )
    attention_confident = _confidence_meets_minimum(
        value=attention_confidence,
        minimum=_config_float(
            config,
            "display_attention_target_min_confidence",
            _DEFAULT_TARGET_MIN_CONFIDENCE,
            minimum=0.0,
            maximum=1.0,
        ),
    )

    attention_target_invalid_reason: str | None = None
    if attention_target is not None and attention_target.active:
        if not attention_fresh:
            attention_target_invalid_reason = "stale_attention_target"
        elif not attention_confident:
            attention_target_invalid_reason = "low_attention_target_confidence"
        else:
            target_center_x = attention_target.target_center_x
            target_center_y = attention_target.target_center_y
            horizontal_fallback = _normalize_text(attention_target.target_horizontal).lower()
            vertical_fallback = _normalize_text(attention_target.target_vertical).lower()
            target_zone = (
                _normalize_text(attention_target.target_zone).lower()
                or zone
                or _horizontal_zone_from_direction(horizontal_fallback)
            )
            target_has_geometry = any(
                value is not None
                for value in (target_center_x, target_center_y)
            ) or horizontal_fallback in {"left", "center", "right"} or vertical_fallback in {"up", "center", "down"} or bool(target_zone)
            if target_has_geometry and (
                not _camera_attention_unavailable(camera)
                or target_center_x is not None
                or target_center_y is not None
                or horizontal_fallback in {"left", "right"}
                or vertical_fallback in {"up", "down"}
            ):
                cue_gaze_x, cue_gaze_y = _cue_axes_for_target(
                    camera_center_x=(center_x if target_center_x is None else target_center_x),
                    camera_center_y=(center_y if target_center_y is None else target_center_y),
                    camera_zone=target_zone,
                    horizontal_fallback=horizontal_fallback,
                    vertical_fallback=vertical_fallback,
                )
                gaze = _gaze_from_attention_target(attention_target)
                if gaze is None:
                    gaze = _gaze_from_axes(cue_gaze_x, cue_gaze_y)
                if gaze is None:
                    attention_target_invalid_reason = "no_attention_target"
                else:
                    brows = _brows_from_attention_target(attention_target, camera)
                    mouth = DisplayFaceMouthStyle.SPEAK if attention_target.speaker_locked else None
                    head_dx, head_dy = _head_offsets_for_target(
                        cue_gaze_x=cue_gaze_x,
                        cue_gaze_y=cue_gaze_y,
                        camera_center_x=(center_x if target_center_x is None else target_center_x),
                        camera_center_y=(center_y if target_center_y is None else target_center_y),
                        camera_zone=target_zone,
                        horizontal_fallback=horizontal_fallback,
                        vertical_fallback=vertical_fallback,
                    )
                    return DisplayAttentionCueDecision(
                        active=True,
                        reason=attention_target.state,
                        gaze=gaze,
                        cue_gaze_x=cue_gaze_x,
                        cue_gaze_y=cue_gaze_y,
                        mouth=mouth,
                        brows=brows,
                        head_dx=head_dx,
                        head_dy=head_dy,
                        hold_seconds=_hold_seconds_from_config(config),
                        speaker_locked=attention_target.speaker_locked,
                        camera_center_x=(center_x if target_center_x is None else target_center_x),
                        camera_zone=target_zone,
                        observed_at=attention_observed_at,
                        confidence=attention_confidence,
                    )
            else:
                attention_target_invalid_reason = "no_attention_target"

    if not camera_fresh:
        return DisplayAttentionCueDecision(
            reason="stale_camera_anchor",
            observed_at=camera_observed_at,
            confidence=camera_confidence,
        )
    if not camera_confident:
        return DisplayAttentionCueDecision(
            reason="low_camera_confidence",
            observed_at=camera_observed_at,
            confidence=camera_confidence,
        )
    if not person_visible:
        if attention_target_invalid_reason is not None:
            return DisplayAttentionCueDecision(
                reason=attention_target_invalid_reason,
                observed_at=attention_observed_at,
                confidence=attention_confidence,
            )
        return DisplayAttentionCueDecision(
            reason="no_visible_person",
            observed_at=camera_observed_at,
            confidence=camera_confidence,
        )

    gaze = _gaze_from_camera(camera)
    if gaze is None:
        if attention_target_invalid_reason is not None:
            return DisplayAttentionCueDecision(
                reason=attention_target_invalid_reason,
                observed_at=attention_observed_at,
                confidence=attention_confidence,
            )
        return DisplayAttentionCueDecision(
            reason="no_visual_anchor",
            observed_at=camera_observed_at,
            confidence=camera_confidence,
        )
    cue_gaze_x, cue_gaze_y = _cue_axes_for_target(
        camera_center_x=center_x,
        camera_center_y=center_y,
        camera_zone=zone,
        horizontal_fallback=zone,
        vertical_fallback=None,
    )

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
    head_dx, head_dy = _head_offsets_for_target(
        cue_gaze_x=cue_gaze_x,
        cue_gaze_y=cue_gaze_y,
        camera_center_x=center_x,
        camera_center_y=center_y,
        camera_zone=zone,
        horizontal_fallback=zone,
        vertical_fallback=None,
    )
    return DisplayAttentionCueDecision(
        active=True,
        reason="speaker_visible_person" if speaking_to_visible_person else "visible_person",
        gaze=gaze,
        cue_gaze_x=cue_gaze_x,
        cue_gaze_y=cue_gaze_y,
        mouth=mouth,
        brows=brows,
        head_dx=head_dx,
        head_dy=head_dy,
        hold_seconds=_hold_seconds_from_config(config),
        speaker_locked=speaking_to_visible_person,
        camera_center_x=center_x,
        camera_zone=zone,
        observed_at=camera_observed_at,
        confidence=camera_confidence,
    )


def _camera_attention_unavailable(camera: Mapping[str, object]) -> bool:
    return (
        _coerce_optional_bool(camera.get("camera_online")) is False
        or _coerce_optional_bool(camera.get("camera_ready")) is False
    )


@dataclass(slots=True)
class _OneEuroFilter1D:
    min_cutoff_hz: float
    beta: float
    derivative_cutoff_hz: float
    value: float | None = None
    derivative: float = 0.0
    last_t: float | None = None

    def reset(self) -> None:
        self.value = None
        self.derivative = 0.0
        self.last_t = None

    def step(self, value: float, t_s: float) -> float:
        if self.value is None or self.last_t is None:
            self.value = value
            self.derivative = 0.0
            self.last_t = t_s
            return value
        dt = t_s - self.last_t
        if not math.isfinite(dt) or dt <= 0.0 or dt > _MAX_FILTER_GAP_S:
            self.value = value
            self.derivative = 0.0
            self.last_t = t_s
            return value
        raw_derivative = (value - self.value) / dt
        derivative_alpha = _one_euro_alpha(dt, self.derivative_cutoff_hz)
        self.derivative = derivative_alpha * raw_derivative + (1.0 - derivative_alpha) * self.derivative
        cutoff = self.min_cutoff_hz + self.beta * abs(self.derivative)
        alpha = _one_euro_alpha(dt, cutoff)
        self.value = alpha * value + (1.0 - alpha) * self.value
        self.last_t = t_s
        return self.value


def _one_euro_alpha(dt_s: float, cutoff_hz: float) -> float:
    if dt_s <= 0.0:
        return 1.0
    cutoff_hz = max(1e-6, cutoff_hz)
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + (tau / dt_s))


@dataclass(slots=True)
class _AttentionCoordinateFilterState:
    x: _OneEuroFilter1D | None = None
    y: _OneEuroFilter1D | None = None

    def reset(self) -> None:
        if self.x is not None:
            self.x.reset()
        if self.y is not None:
            self.y.reset()

    def ensure(self, *, min_cutoff_hz: float, beta: float, derivative_cutoff_hz: float) -> None:
        if self.x is None:
            self.x = _OneEuroFilter1D(min_cutoff_hz=min_cutoff_hz, beta=beta, derivative_cutoff_hz=derivative_cutoff_hz)
        else:
            self.x.min_cutoff_hz = min_cutoff_hz
            self.x.beta = beta
            self.x.derivative_cutoff_hz = derivative_cutoff_hz
        if self.y is None:
            self.y = _OneEuroFilter1D(min_cutoff_hz=min_cutoff_hz, beta=beta, derivative_cutoff_hz=derivative_cutoff_hz)
        else:
            self.y.min_cutoff_hz = min_cutoff_hz
            self.y.beta = beta
            self.y.derivative_cutoff_hz = derivative_cutoff_hz

    def step(
        self,
        *,
        x: float | None,
        y: float | None,
        t_s: float,
        min_cutoff_hz: float,
        beta: float,
        derivative_cutoff_hz: float,
    ) -> tuple[float | None, float | None, bool]:
        if x is None and y is None:
            self.reset()
            return None, None, False
        self.ensure(
            min_cutoff_hz=min_cutoff_hz,
            beta=beta,
            derivative_cutoff_hz=derivative_cutoff_hz,
        )
        smoothed = False
        out_x = x
        out_y = y
        if x is not None and self.x is not None:
            out_x = _coerce_optional_ratio(self.x.step(x, t_s))
            smoothed = True
        elif self.x is not None:
            self.x.reset()
        if y is not None and self.y is not None:
            out_y = _coerce_optional_ratio(self.y.step(y, t_s))
            smoothed = True
        elif self.y is not None:
            self.y.reset()
        return out_x, out_y, smoothed


@dataclass(slots=True)
class DisplayAttentionCuePublisher:
    """Persist conservative HDMI attention-follow cues without stomping others."""

    controller: DisplayFaceExpressionController
    source: str = _SOURCE
    _camera_filter: _AttentionCoordinateFilterState = field(default_factory=_AttentionCoordinateFilterState, init=False, repr=False)
    _target_filter: _AttentionCoordinateFilterState = field(default_factory=_AttentionCoordinateFilterState, init=False, repr=False)

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

        effective_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        prepared_facts, smoothed = self._prepare_live_facts(
            config=config,
            live_facts=live_facts,
            now=effective_now,
        )
        decision = derive_display_attention_cue(
            config=config,
            live_facts=prepared_facts,
            now=effective_now,
        )
        if smoothed and decision.active:
            decision = replace(decision, smoothed=True)
        return self.publish(decision, now=effective_now, config=config)

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
        active_owner = _cue_source(active_cue)
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
            latest_owner = _cue_source(self.store.load_active(now=effective_now))
            if latest_owner is not None and latest_owner != self.source:
                return DisplayAttentionCuePublishResult(
                    action="blocked_foreign_cue",
                    decision=decision,
                    owner=latest_owner,
                )
            if latest_owner == self.source:
                self.controller.clear(now=effective_now)
                return DisplayAttentionCuePublishResult(
                    action="cleared",
                    decision=decision,
                    owner=latest_owner,
                )
            return DisplayAttentionCuePublishResult(
                action="inactive",
                decision=decision,
                owner=latest_owner,
            )

        if active_cue is not None and _cue_matches_decision(active_cue, decision):
            if _matching_cue_needs_refresh(
                cue=active_cue,
                decision=decision,
                now=effective_now,
            ):
                latest_owner = _cue_source(self.store.load_active(now=effective_now))
                if latest_owner is not None and latest_owner != self.source:
                    return DisplayAttentionCuePublishResult(
                        action="blocked_foreign_cue",
                        decision=decision,
                        owner=latest_owner,
                    )
                self.store.save(
                    decision.cue(),
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

        latest_owner = _cue_source(self.store.load_active(now=effective_now))
        if latest_owner is not None and latest_owner != self.source:
            return DisplayAttentionCuePublishResult(
                action="blocked_foreign_cue",
                decision=decision,
                owner=latest_owner,
            )
        self.store.save(
            decision.cue(),
            hold_seconds=decision.hold_seconds,
            now=effective_now,
        )
        return DisplayAttentionCuePublishResult(
            action="updated",
            decision=decision,
            owner=self.source,
        )

    def _prepare_live_facts(
        self,
        *,
        config: TwinrConfig,
        live_facts: Mapping[str, object],
        now: datetime,
    ) -> tuple[dict[str, object], bool]:
        """Apply optional publisher-local smoothing to numeric target coordinates."""

        prepared = {str(key): value for key, value in live_facts.items()}
        if not _config_bool(config, "display_attention_smoothing_enabled", True):
            return prepared, False

        min_cutoff_hz = _config_float(
            config,
            "display_attention_filter_min_cutoff_hz",
            _DEFAULT_ONE_EURO_MIN_CUTOFF_HZ,
            minimum=1e-6,
        )
        beta = _config_float(
            config,
            "display_attention_filter_beta",
            _DEFAULT_ONE_EURO_BETA,
            minimum=0.0,
        )
        derivative_cutoff_hz = _config_float(
            config,
            "display_attention_filter_derivative_cutoff_hz",
            _DEFAULT_ONE_EURO_DERIVATIVE_CUTOFF_HZ,
            minimum=1e-6,
        )
        t_s = now.timestamp()

        smoothed = False

        raw_camera = _coerce_mapping(live_facts.get("camera"))
        camera = dict(raw_camera)
        camera_center_x = None if _coerce_optional_bool(camera.get("primary_person_center_x_unknown")) is True else _coerce_optional_ratio(camera.get("primary_person_center_x"))
        camera_center_y = None if _coerce_optional_bool(camera.get("primary_person_center_y_unknown")) is True else _coerce_optional_ratio(camera.get("primary_person_center_y"))
        if (
            _coerce_optional_bool(camera.get("person_visible")) is True
            and _fact_is_fresh(
                config=config,
                observed_at=_fact_timestamp(camera),
                now=now,
                config_attr="display_attention_camera_max_age_s",
                default=_DEFAULT_CAMERA_MAX_AGE_S,
            )
            and _confidence_meets_minimum(
                value=_camera_confidence(camera),
                minimum=_config_float(
                    config,
                    "display_attention_camera_min_confidence",
                    _DEFAULT_CAMERA_MIN_CONFIDENCE,
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        ):
            filtered_x, filtered_y, applied = self._camera_filter.step(
                x=camera_center_x,
                y=camera_center_y,
                t_s=t_s,
                min_cutoff_hz=min_cutoff_hz,
                beta=beta,
                derivative_cutoff_hz=derivative_cutoff_hz,
            )
            if filtered_x is not None:
                camera["primary_person_center_x"] = filtered_x
                camera["primary_person_center_x_unknown"] = False
            if filtered_y is not None:
                camera["primary_person_center_y"] = filtered_y
                camera["primary_person_center_y_unknown"] = False
            smoothed = smoothed or applied
        else:
            self._camera_filter.reset()
        prepared["camera"] = camera

        raw_target = _coerce_mapping(live_facts.get("attention_target"))
        target = dict(raw_target)
        target_snapshot = MultimodalAttentionTargetSnapshot.from_fact_map(raw_target)
        target_center_x = _coerce_optional_ratio(target.get("target_center_x"))
        target_center_y = _coerce_optional_ratio(target.get("target_center_y"))
        if (
            target_snapshot is not None
            and target_snapshot.active
            and _fact_is_fresh(
                config=config,
                observed_at=_fact_timestamp(raw_target),
                now=now,
                config_attr="display_attention_target_max_age_s",
                default=_DEFAULT_TARGET_MAX_AGE_S,
            )
            and _confidence_meets_minimum(
                value=_attention_target_confidence(raw_target),
                minimum=_config_float(
                    config,
                    "display_attention_target_min_confidence",
                    _DEFAULT_TARGET_MIN_CONFIDENCE,
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        ):
            filtered_x, filtered_y, applied = self._target_filter.step(
                x=target_center_x,
                y=target_center_y,
                t_s=t_s,
                min_cutoff_hz=min_cutoff_hz,
                beta=beta,
                derivative_cutoff_hz=derivative_cutoff_hz,
            )
            if filtered_x is not None:
                target["target_center_x"] = filtered_x
            if filtered_y is not None:
                target["target_center_y"] = filtered_y
            smoothed = smoothed or applied
        else:
            self._target_filter.reset()
        prepared["attention_target"] = target

        return prepared, smoothed


def resolve_display_attention_refresh_interval(config: TwinrConfig) -> float | None:
    """Return the bounded local refresh cadence for HDMI attention-follow."""

    interval_s = _config_float(
        config,
        "display_attention_refresh_interval_s",
        0.35,
        minimum=None,
    )
    if not math.isfinite(interval_s) or interval_s <= 0.0:
        return None
    return max(_MIN_REFRESH_INTERVAL_S, interval_s)


def display_attention_refresh_backend_supported(
    *,
    config: TwinrConfig,
) -> bool:
    """Return whether the current display backend may run the attention-refresh lane.

    `hdmi_wayland` stays fail-closed here. The transcript-first voice runtime
    shares one streaming PID with live voice transport, and fresh Pi evidence
    on 2026-04-13 showed that re-opening the hidden IMX500/servo lane there
    still reproduces the old memory-pressure and backpressure signature.
    Continuous face publication stays gated separately below, but the cheaper
    hidden lane is not treated as safe on Wayland until new isolated Pi proof
    says otherwise.
    """

    if resolve_display_attention_refresh_interval(config) is None:
        return False
    display_driver = str(getattr(config, "display_driver", "") or "").strip().lower()
    if not display_driver.startswith("hdmi"):
        return False
    return display_driver != "hdmi_wayland"


def display_attention_refresh_supported(
    *,
    config: TwinrConfig,
    vision_observer: object | None,
) -> bool:
    """Return whether a bounded fast attention-refresh path is safe to run."""

    if not display_attention_refresh_backend_supported(config=config):
        return False
    return getattr(vision_observer, "supports_attention_refresh", False) is True


def display_attention_face_publish_supported(
    *,
    config: TwinrConfig,
) -> bool:
    """Return whether the current HDMI backend may accept live face-follow cue writes.

    `hdmi_wayland` still stays fail-closed for continuous face publication.
    The hidden camera/servo fast path is now separately cleared there, but
    fullscreen Wayland face rerenders remain an independent risk surface that
    still requires dedicated display-side proof before re-enabling.
    """

    display_driver = str(getattr(config, "display_driver", "") or "").strip().lower()
    return display_driver != "hdmi_wayland"


def _gaze_from_attention_target(
    attention_target: MultimodalAttentionTargetSnapshot,
) -> DisplayFaceGazeDirection | None:
    """Map one camera-space attention target to a user-facing display gaze."""

    horizontal = _horizontal_direction(
        center_x=attention_target.target_center_x,
        zone=_normalize_text(attention_target.target_horizontal).lower(),
    )
    if horizontal not in {"left", "center", "right"}:
        return None
    horizontal = _mirror_camera_horizontal(horizontal)
    vertical = _vertical_direction(
        center_y=attention_target.target_center_y,
        fallback=_normalize_text(attention_target.target_vertical).lower(),
    )
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
    ttl_s = _config_float(
        config,
        "display_face_cue_ttl_s",
        4.0,
        minimum=0.05,
    )
    capture_interval_s = _config_float(
        config,
        "proactive_capture_interval_s",
        6.0,
        minimum=0.05,
    )
    return max(
        ttl_s,
        (refresh_interval_s if refresh_interval_s is not None else capture_interval_s) + 0.75,
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
    vertical = _vertical_direction(
        center_y=(
            None
            if _coerce_optional_bool(camera.get("primary_person_center_y_unknown")) is True
            else _coerce_optional_ratio(camera.get("primary_person_center_y"))
        ),
        fallback=None,
    )
    return _GAZE_DIRECTION_MAP[(vertical, horizontal)]


def _cue_axes_for_target(
    *,
    camera_center_x: float | None,
    camera_center_y: float | None,
    camera_zone: str | None,
    horizontal_fallback: str | None,
    vertical_fallback: str | None,
) -> tuple[int, int]:
    """Return one bounded direct cue axis from the current target position."""

    gaze_x = _cue_axis_from_horizontal(
        camera_center_x=camera_center_x,
        camera_zone=camera_zone,
        horizontal_fallback=horizontal_fallback,
    )
    gaze_y = _cue_axis_from_vertical(
        camera_center_y=camera_center_y,
        vertical_fallback=vertical_fallback,
    )
    return gaze_x, gaze_y


def _cue_axis_from_horizontal(
    *,
    camera_center_x: float | None,
    camera_zone: str | None,
    horizontal_fallback: str | None,
) -> int:
    display_offset = _display_offset_from_camera(
        camera_center_x=camera_center_x,
        camera_zone=camera_zone,
    )
    if display_offset is not None:
        magnitude = abs(display_offset)
        if magnitude < _GAZE_SOFT_OFFSET_THRESHOLD:
            return 0
        if magnitude < _GAZE_STRONG_OFFSET_THRESHOLD:
            return 1 if display_offset > 0.0 else -1
        scaled = (magnitude / 0.5) * _MAX_DYNAMIC_GAZE_AXIS
        strength = max(1, min(_MAX_DYNAMIC_GAZE_AXIS, int(round(scaled))))
        return strength if display_offset > 0.0 else -strength

    fallback = _normalize_text(horizontal_fallback).lower()
    if fallback == "left":
        return _MIN_EXPLICIT_DIRECTION_AXIS
    if fallback == "right":
        return -_MIN_EXPLICIT_DIRECTION_AXIS
    return 0


def _cue_axis_from_vertical(
    *,
    camera_center_y: float | None,
    vertical_fallback: str | None,
) -> int:
    vertical_offset = _vertical_offset_from_camera(camera_center_y=camera_center_y)
    if vertical_offset is not None:
        vertical_magnitude = abs(vertical_offset)
        if vertical_magnitude < _GAZE_VERTICAL_SOFT_OFFSET_THRESHOLD:
            return 0
        if vertical_magnitude < _GAZE_VERTICAL_STRONG_OFFSET_THRESHOLD:
            vertical_strength = 2
        else:
            scaled_vertical = (vertical_magnitude / 0.5) * _MAX_DYNAMIC_GAZE_AXIS
            vertical_strength = max(2, min(_MAX_DYNAMIC_GAZE_AXIS, int(round(scaled_vertical))))
        return vertical_strength if vertical_offset > 0.0 else -vertical_strength

    fallback = _normalize_text(vertical_fallback).lower()
    if fallback == "down":
        return _MIN_EXPLICIT_DIRECTION_AXIS
    if fallback == "up":
        return -_MIN_EXPLICIT_DIRECTION_AXIS
    return 0


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


def _head_offsets_for_target(
    *,
    cue_gaze_x: int,
    cue_gaze_y: int,
    camera_center_x: float | None,
    camera_center_y: float | None,
    camera_zone: str | None,
    horizontal_fallback: str | None,
    vertical_fallback: str | None,
) -> tuple[int, int]:
    """Add a bounded head drift aligned with the direct eye cue."""

    if cue_gaze_x:
        horizontal = _sign(cue_gaze_x) * min(_MAX_DYNAMIC_HEAD_AXIS, max(1, abs(cue_gaze_x)))
    else:
        horizontal = _head_dx_from_camera(
            camera_center_x=camera_center_x,
            camera_zone=camera_zone,
            horizontal_fallback=horizontal_fallback,
        )
    vertical = _head_dy_from_camera(
        cue_gaze_y=cue_gaze_y,
        camera_center_y=camera_center_y,
        vertical_fallback=vertical_fallback,
    )
    return horizontal, vertical


def _head_offsets_for_gaze(
    gaze: DisplayFaceGazeDirection,
    *,
    camera_center_x: float | None,
    camera_center_y: float | None,
    camera_zone: str | None,
) -> tuple[int, int]:
    """Add a bounded head drift so the face turns before full eye commits."""

    gaze_x, gaze_y = gaze.axes()
    if gaze_x:
        horizontal = _MAX_DYNAMIC_HEAD_AXIS * _sign(gaze_x)
    else:
        horizontal = _head_dx_from_camera(
            camera_center_x=camera_center_x,
            camera_zone=camera_zone,
            horizontal_fallback=None,
        )
    vertical = _head_dy_from_camera(
        cue_gaze_y=gaze_y,
        camera_center_y=camera_center_y,
        vertical_fallback=None,
    )
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
        refresh_interval_s = _config_float(
            config,
            "proactive_capture_interval_s",
            6.0,
            minimum=0.05,
        )
    if not math.isfinite(refresh_interval_s) or refresh_interval_s <= 0.0:
        refresh_interval_s = 0.35
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
    if not decision.active:
        return decision
    if decision.gaze != DisplayFaceGazeDirection.CENTER:
        return decision
    if decision.cue_gaze_y != 0 or abs(decision.cue_gaze_x) > 1:
        return decision
    if _cue_source(active_cue) != decision.source:
        return decision
    if _cue_gaze_x(active_cue) == 0 and _cue_gaze_y(active_cue) == 0:
        return decision
    updated_at = _parse_timestamp(getattr(active_cue, "updated_at", None))
    if updated_at is None:
        return decision
    if not _should_hold_center_direction(
        config=config,
        now=now,
        updated_at=updated_at,
    ):
        return decision

    held_gaze = _gaze_from_axes(_cue_gaze_x(active_cue), _cue_gaze_y(active_cue))
    return replace(
        decision,
        reason=f"{decision.reason}_held_direction",
        gaze=(held_gaze or decision.gaze),
        cue_gaze_x=_cue_gaze_x(active_cue),
        cue_gaze_y=_cue_gaze_y(active_cue),
        head_dx=_cue_head_dx(active_cue),
        head_dy=_cue_head_dy(active_cue),
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
    if _cue_source(active_cue) != decision.source:
        return decision
    if decision.reason not in {
        "no_visible_person",
        "no_visual_anchor",
        "no_attention_target",
        "stale_attention_target",
        "stale_camera_anchor",
        "low_attention_target_confidence",
        "low_camera_confidence",
    }:
        return decision
    updated_at = _parse_timestamp(getattr(active_cue, "updated_at", None))
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

    horizontal = "center"
    vertical = "center"
    if abs(int(gaze_x)) >= 2:
        horizontal = "right" if gaze_x > 0 else "left"
    if abs(int(gaze_y)) >= 2:
        vertical = "down" if gaze_y > 0 else "up"
    return _GAZE_DIRECTION_MAP.get((vertical, horizontal))


def _decision_from_cue(
    cue: DisplayFaceCue,
    *,
    reason: str,
    hold_seconds: float,
) -> DisplayAttentionCueDecision:
    """Reconstruct one decision from an already persisted local cue."""

    gaze_x = _cue_gaze_x(cue)
    gaze_y = _cue_gaze_y(cue)
    gaze = _gaze_from_axes(gaze_x, gaze_y) or DisplayFaceGazeDirection.CENTER
    mouth = None
    if getattr(cue, "mouth", None):
        try:
            mouth = DisplayFaceMouthStyle(cue.mouth)
        except ValueError:
            mouth = None
    brows = None
    if getattr(cue, "brows", None):
        try:
            brows = DisplayFaceBrowStyle(cue.brows)
        except ValueError:
            brows = None
    return DisplayAttentionCueDecision(
        active=True,
        reason=reason,
        source=_cue_source(cue) or _SOURCE,
        gaze=gaze,
        cue_gaze_x=gaze_x,
        cue_gaze_y=gaze_y,
        mouth=mouth,
        brows=brows,
        head_dx=_cue_head_dx(cue),
        head_dy=_cue_head_dy(cue),
        hold_seconds=hold_seconds,
        speaker_locked=getattr(cue, "mouth", None) == DisplayFaceMouthStyle.SPEAK.value,
    )


def _head_dx_from_camera(
    *,
    camera_center_x: float | None,
    camera_zone: str | None,
    horizontal_fallback: str | None,
) -> int:
    """Return one small user-facing head turn from camera-space position."""

    display_offset = _display_offset_from_camera(
        camera_center_x=camera_center_x,
        camera_zone=camera_zone,
    )
    if display_offset is not None:
        magnitude = abs(display_offset)
        if magnitude < _HEAD_SOFT_OFFSET_THRESHOLD:
            return 0
        strength = 2 if magnitude >= _HEAD_STRONG_OFFSET_THRESHOLD else 1
        return strength if display_offset > 0.0 else -strength

    fallback = _normalize_text(horizontal_fallback).lower()
    if fallback == "left":
        return 1
    if fallback == "right":
        return -1
    return 0


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


def _vertical_offset_from_camera(*, camera_center_y: float | None) -> float | None:
    """Return one vertical screen offset in ``[-0.5, 0.5]``."""

    if camera_center_y is None:
        return None
    return max(-0.5, min(0.5, camera_center_y - 0.5))


def _vertical_direction(*, center_y: float | None, fallback: str | None) -> str:
    """Return one coarse up/center/down anchor from center-y."""

    if center_y is not None:
        if center_y <= _UP_ENTER_THRESHOLD:
            return "up"
        if center_y >= _DOWN_ENTER_THRESHOLD:
            return "down"
        return "center"
    if fallback in {"up", "center", "down"}:
        return fallback
    return "center"


def _head_dy_from_camera(
    *,
    cue_gaze_y: int,
    camera_center_y: float | None,
    vertical_fallback: str | None,
) -> int:
    """Return one small vertical head drift aligned with the eye cue."""

    if cue_gaze_y:
        return _sign(cue_gaze_y) * (2 if abs(cue_gaze_y) >= 3 else 1)
    vertical_offset = _vertical_offset_from_camera(camera_center_y=camera_center_y)
    if vertical_offset is not None:
        magnitude = abs(vertical_offset)
        if magnitude < _HEAD_VERTICAL_SOFT_OFFSET_THRESHOLD:
            return 0
        strength = 2 if magnitude >= _HEAD_VERTICAL_STRONG_OFFSET_THRESHOLD else 1
        return strength if vertical_offset > 0.0 else -strength
    fallback = _normalize_text(vertical_fallback).lower()
    if fallback == "down":
        return 1
    if fallback == "up":
        return -1
    return 0


def _should_hold_center_direction(
    *,
    config: TwinrConfig,
    now: datetime,
    updated_at: datetime,
) -> bool:
    """Return whether one center decision should preserve the recent side cue."""

    return (now - updated_at).total_seconds() <= _direction_hold_seconds(config)


def _cue_matches_decision(cue: DisplayFaceCue, decision: DisplayAttentionCueDecision) -> bool:
    """Return whether one active cue already represents the desired decision."""

    expression = decision.cue()
    return (
        _cue_source(cue) == decision.source
        and _cue_gaze_x(cue) == expression.gaze_x
        and _cue_gaze_y(cue) == expression.gaze_y
        and getattr(cue, "mouth", None) == expression.mouth
        and getattr(cue, "brows", None) == expression.brows
        and _cue_head_dx(cue) == expression.head_dx
        and _cue_head_dy(cue) == expression.head_dy
    )


def _matching_cue_needs_refresh(
    *,
    cue: DisplayFaceCue,
    decision: DisplayAttentionCueDecision,
    now: datetime,
) -> bool:
    """Return whether one unchanged cue should be renewed before expiry."""

    expires_at = _parse_timestamp(getattr(cue, "expires_at", None))
    if expires_at is None:
        return False
    seconds_left = (expires_at - now).total_seconds()
    threshold_s = min(float(decision.hold_seconds), _MATCHING_CUE_REFRESH_LOOKAHEAD_S)
    return seconds_left <= max(_MIN_REFRESH_INTERVAL_S, threshold_s)


def _cue_source(cue: DisplayFaceCue | None) -> str | None:
    if cue is None:
        return None
    text = _normalize_source(getattr(cue, "source", None))
    return text or None


def _cue_gaze_x(cue: DisplayFaceCue) -> int:
    return _clamp_int(getattr(cue, "gaze_x", 0), -_MAX_DYNAMIC_GAZE_AXIS, _MAX_DYNAMIC_GAZE_AXIS)


def _cue_gaze_y(cue: DisplayFaceCue) -> int:
    return _clamp_int(getattr(cue, "gaze_y", 0), -_MAX_DYNAMIC_GAZE_AXIS, _MAX_DYNAMIC_GAZE_AXIS)


def _cue_head_dx(cue: DisplayFaceCue) -> int:
    return _clamp_int(getattr(cue, "head_dx", 0), -_MAX_DYNAMIC_HEAD_AXIS, _MAX_DYNAMIC_HEAD_AXIS)


def _cue_head_dy(cue: DisplayFaceCue) -> int:
    return _clamp_int(getattr(cue, "head_dy", 0), -_MAX_DYNAMIC_HEAD_AXIS, _MAX_DYNAMIC_HEAD_AXIS)


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


def _normalize_source(value: object | None) -> str:
    return _normalize_text(value)[:_MAX_SOURCE_LENGTH]


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
    if numeric != numeric or not math.isfinite(numeric):
        return None
    return max(0.0, min(1.0, numeric))


def _coerce_optional_float(value: object | None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _clamp_int(value: object | None, minimum: int, maximum: int) -> int:
    try:
        numeric = int(round(float(value or 0)))
    except (TypeError, ValueError):
        return 0
    return max(minimum, min(maximum, numeric))


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


def _fact_timestamp(fact_map: Mapping[str, object]) -> datetime | None:
    for key in ("observed_at", "updated_at", "timestamp", "captured_at", "detected_at", "seen_at"):
        parsed = _parse_timestamp(fact_map.get(key))
        if parsed is not None:
            return parsed
    return None


def _fact_is_fresh(
    *,
    config: TwinrConfig,
    observed_at: datetime | None,
    now: datetime,
    config_attr: str,
    default: float,
) -> bool:
    if observed_at is None:
        return True
    max_age_s = _config_float(config, config_attr, default, minimum=0.0)
    return (now - observed_at).total_seconds() <= max_age_s


def _camera_confidence(fact_map: Mapping[str, object]) -> float | None:
    for key in (
        "primary_person_confidence",
        "person_confidence",
        "visual_anchor_confidence",
        "confidence",
    ):
        value = _coerce_optional_ratio(fact_map.get(key))
        if value is not None:
            return value
    return None


def _attention_target_confidence(fact_map: Mapping[str, object]) -> float | None:
    for key in (
        "target_confidence",
        "attention_confidence",
        "confidence",
    ):
        value = _coerce_optional_ratio(fact_map.get(key))
        if value is not None:
            return value
    return None


def _confidence_meets_minimum(*, value: float | None, minimum: float) -> bool:
    if value is None:
        return True
    return value >= minimum


def _config_bool(config: TwinrConfig, name: str, default: bool) -> bool:
    value = getattr(config, name, default)
    parsed = _coerce_optional_bool(value)
    return default if parsed is None else parsed


def _config_float(
    config: TwinrConfig,
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = _coerce_optional_float(getattr(config, name, default))
    if value is None:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _horizontal_zone_from_direction(direction: str | None) -> str | None:
    direction = _normalize_text(direction).lower()
    if direction in {"left", "center", "right"}:
        return direction
    return None


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
    "display_attention_face_publish_supported",
    "display_attention_refresh_backend_supported",
    "display_attention_refresh_supported",
    "derive_display_attention_cue",
    "resolve_display_attention_refresh_interval",
]
