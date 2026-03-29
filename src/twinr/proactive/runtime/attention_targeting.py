# CHANGELOG: 2026-03-29
# BUG-1: Focus matching is now track/geometry-aware instead of horizontal-zone only, so two
#        visible people in the same zone are no longer conflated.
# BUG-2: Fresh stable identity-fusion focus now overrides stale remembered focus instead of
#        being masked for the full hold window.
# SEC-1: Replayed/stale upstream sensor claims are now rejected with bounded timestamp freshness
#        checks and clock-skew guards before they can pin HDMI attention.
# IMP-1: Source timestamps, optional track IDs, center coordinates, and tracking confidence now
#        propagate through the policy for better edge-runtime fusion.
# IMP-2: Focus memory now survives brief dropouts more safely and expires correctly even under
#        out-of-order telemetry common in Raspberry Pi real-time pipelines.

"""Prioritize conservative HDMI attention targets from multimodal runtime signals.

This module keeps short-lived attention-target policy out of the proactive
orchestrator and the display cue renderer. It combines the current camera
anchor with speaker association, showing-intent signals, and bounded
session-focus memory so Twinr can look at the most relevant visible person
without pretending to identify secondary people in the room. Normal
attention-follow stays horizontal-only so stronger up/down body-language poses
remain reserved for explicit semantic face states.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from twinr.agent.base_agent.config import TwinrConfig
from twinr.proactive.runtime.claim_metadata import (
    coerce_mapping,
    coerce_optional_bool,
    coerce_optional_int,
    coerce_optional_ratio,
    mean_confidence,
    normalize_text,
)
from twinr.proactive.runtime.continuous_attention import (
    ContinuousAttentionTargetSnapshot,
    ContinuousAttentionTracker,
)
from twinr.proactive.runtime.identity_fusion import MultimodalIdentityFusionSnapshot
from twinr.proactive.runtime.speaker_association import (
    ReSpeakerSpeakerAssociationSnapshot,
    derive_respeaker_speaker_association,
)


_DEFAULT_SESSION_FOCUS_HOLD_S = 4.5
_MIN_SESSION_FOCUS_HOLD_S = 0.5
_DEFAULT_SOURCE_MAX_STALENESS_S = 1.25
_MIN_SOURCE_MAX_STALENESS_S = 0.25
_DEFAULT_IDENTITY_MAX_STALENESS_S = 2.5
_MIN_IDENTITY_MAX_STALENESS_S = 0.5
_DEFAULT_CLOCK_SKEW_TOLERANCE_S = 0.35
_MIN_CLOCK_SKEW_TOLERANCE_S = 0.0
_DEFAULT_FOCUS_MATCH_CENTER_X_DELTA = 0.14
_MIN_FOCUS_MATCH_CENTER_X_DELTA = 0.02
_DEFAULT_FOCUS_MATCH_CENTER_Y_DELTA = 0.2
_MIN_FOCUS_MATCH_CENTER_Y_DELTA = 0.02

_INTERACTIVE_RUNTIME_STATES = frozenset({"listening", "processing", "answering"})
_ACTIVE_RUNTIME_STATES = _INTERACTIVE_RUNTIME_STATES | {"waiting"}
_VALID_HORIZONTAL = frozenset({"left", "center", "right"})
_VALID_VERTICAL = frozenset({"up", "center", "down"})
_FOLLOW_VERTICAL = "center"
_LEFT_THRESHOLD = 0.36
_RIGHT_THRESHOLD = 0.64


@dataclass(frozen=True, slots=True)
class MultimodalAttentionTargetConfig:
    """Store bounded tuning values for multimodal HDMI attention targeting."""

    session_focus_hold_s: float = _DEFAULT_SESSION_FOCUS_HOLD_S
    source_max_staleness_s: float = _DEFAULT_SOURCE_MAX_STALENESS_S
    identity_max_staleness_s: float = _DEFAULT_IDENTITY_MAX_STALENESS_S
    clock_skew_tolerance_s: float = _DEFAULT_CLOCK_SKEW_TOLERANCE_S
    focus_match_center_x_delta: float = _DEFAULT_FOCUS_MATCH_CENTER_X_DELTA
    focus_match_center_y_delta: float = _DEFAULT_FOCUS_MATCH_CENTER_Y_DELTA

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "MultimodalAttentionTargetConfig":
        """Build one bounded targeting config from the global Twinr config."""

        return cls(
            session_focus_hold_s=_bounded_config_float(
                config=config,
                attr_name="display_attention_session_focus_hold_s",
                default=_DEFAULT_SESSION_FOCUS_HOLD_S,
                minimum=_MIN_SESSION_FOCUS_HOLD_S,
            ),
            source_max_staleness_s=_bounded_config_float(
                config=config,
                attr_name="display_attention_source_max_staleness_s",
                default=_DEFAULT_SOURCE_MAX_STALENESS_S,
                minimum=_MIN_SOURCE_MAX_STALENESS_S,
            ),
            identity_max_staleness_s=_bounded_config_float(
                config=config,
                attr_name="display_attention_identity_max_staleness_s",
                default=_DEFAULT_IDENTITY_MAX_STALENESS_S,
                minimum=_MIN_IDENTITY_MAX_STALENESS_S,
            ),
            clock_skew_tolerance_s=_bounded_config_float(
                config=config,
                attr_name="display_attention_clock_skew_tolerance_s",
                default=_DEFAULT_CLOCK_SKEW_TOLERANCE_S,
                minimum=_MIN_CLOCK_SKEW_TOLERANCE_S,
            ),
            focus_match_center_x_delta=_bounded_config_float(
                config=config,
                attr_name="display_attention_focus_match_center_x_delta",
                default=_DEFAULT_FOCUS_MATCH_CENTER_X_DELTA,
                minimum=_MIN_FOCUS_MATCH_CENTER_X_DELTA,
            ),
            focus_match_center_y_delta=_bounded_config_float(
                config=config,
                attr_name="display_attention_focus_match_center_y_delta",
                default=_DEFAULT_FOCUS_MATCH_CENTER_Y_DELTA,
                minimum=_MIN_FOCUS_MATCH_CENTER_Y_DELTA,
            ),
        )


@dataclass(frozen=True, slots=True)
class MultimodalAttentionTargetSnapshot:
    """Describe one conservative HDMI attention target."""

    observed_at: float | None = None
    state: str = "inactive"
    active: bool = False
    target_horizontal: str | None = None
    target_vertical: str | None = None
    target_zone: str | None = None
    target_track_id: str | None = None
    target_center_x: float | None = None
    target_center_y: float | None = None
    target_velocity_x: float | None = None
    focus_source: str = "none"
    runtime_status: str | None = None
    presence_session_id: int | None = None
    speaker_locked: bool = False
    session_focus_active: bool = False
    showing_intent_active: bool = False
    confidence: float = 0.0

    def to_automation_facts(self) -> dict[str, object]:
        """Serialize the snapshot into automation-friendly facts."""

        return {
            "observed_at": self.observed_at,
            "state": self.state,
            "active": self.active,
            "target_horizontal": self.target_horizontal,
            "target_vertical": self.target_vertical,
            "target_zone": self.target_zone,
            "target_track_id": self.target_track_id,
            "target_center_x": self.target_center_x,
            "target_center_y": self.target_center_y,
            "target_velocity_x": self.target_velocity_x,
            "focus_source": self.focus_source,
            "runtime_status": self.runtime_status,
            "presence_session_id": self.presence_session_id,
            "speaker_locked": self.speaker_locked,
            "session_focus_active": self.session_focus_active,
            "showing_intent_active": self.showing_intent_active,
            "confidence": self.confidence,
        }

    def event_data(self) -> dict[str, object]:
        """Serialize the snapshot into flat ops-event fields."""

        return {
            "attention_target_state": self.state,
            "attention_target_active": self.active,
            "attention_target_horizontal": self.target_horizontal,
            "attention_target_vertical": self.target_vertical,
            "attention_target_track_id": self.target_track_id,
            "attention_target_center_x": self.target_center_x,
            "attention_target_source": self.focus_source,
            "attention_target_speaker_locked": self.speaker_locked,
            "attention_target_session_focus_active": self.session_focus_active,
        }

    @classmethod
    def from_fact_map(
        cls,
        value: object | None,
    ) -> "MultimodalAttentionTargetSnapshot | None":
        """Parse one serialized attention-target payload conservatively."""

        payload = coerce_mapping(value)
        if not payload:
            return None
        confidence = coerce_optional_ratio(payload.get("confidence"))
        return cls(
            observed_at=_coerce_optional_float(payload.get("observed_at")),
            state=_normalize_optional_text(payload.get("state")) or "inactive",
            active=coerce_optional_bool(payload.get("active")) is True,
            target_horizontal=_normalize_direction(payload.get("target_horizontal"), allowed=_VALID_HORIZONTAL),
            target_vertical=_normalize_direction(payload.get("target_vertical"), allowed=_VALID_VERTICAL),
            target_zone=_normalize_direction(payload.get("target_zone"), allowed=_VALID_HORIZONTAL),
            target_track_id=_normalize_optional_text(payload.get("target_track_id")) or None,
            target_center_x=coerce_optional_ratio(payload.get("target_center_x")),
            target_center_y=coerce_optional_ratio(payload.get("target_center_y")),
            target_velocity_x=_coerce_optional_float(payload.get("target_velocity_x")),
            focus_source=_normalize_optional_text(payload.get("focus_source")) or "none",
            runtime_status=_normalize_optional_text(payload.get("runtime_status")) or None,
            presence_session_id=coerce_optional_int(payload.get("presence_session_id")),
            speaker_locked=coerce_optional_bool(payload.get("speaker_locked")) is True,
            session_focus_active=coerce_optional_bool(payload.get("session_focus_active")) is True,
            showing_intent_active=coerce_optional_bool(payload.get("showing_intent_active")) is True,
            confidence=(0.0 if confidence is None else confidence),
        )


@dataclass(frozen=True, slots=True)
class _AttentionAnchor:
    horizontal: str
    vertical: str
    source: str
    track_id: str | None = None
    center_x: float | None = None
    center_y: float | None = None
    velocity_x: float | None = None
    observed_at: float | None = None


@dataclass(slots=True)
class _SessionFocusMemory:
    presence_session_id: int | None
    horizontal: str
    vertical: str
    source: str
    track_id: str | None
    center_x: float | None
    center_y: float | None
    velocity_x: float | None
    updated_at: float


class MultimodalAttentionTargetTracker:
    """Keep bounded session-focus memory for HDMI attention targeting."""

    def __init__(self, *, config: MultimodalAttentionTargetConfig) -> None:
        self.config = config
        self._focus_memory: _SessionFocusMemory | None = None
        self._continuous_tracker: ContinuousAttentionTracker | None = None

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "MultimodalAttentionTargetTracker":
        """Build one tracker from the global Twinr config."""

        tracker = cls(config=MultimodalAttentionTargetConfig.from_config(config))
        tracker._continuous_tracker = ContinuousAttentionTracker.from_config(config)
        return tracker

    def observe(
        self,
        *,
        observed_at: float | None,
        live_facts: Mapping[str, object] | None,
        runtime_status: object | None,
        presence_session_id: int | None,
        speaker_association: ReSpeakerSpeakerAssociationSnapshot | None = None,
        identity_fusion: MultimodalIdentityFusionSnapshot | None = None,
    ) -> MultimodalAttentionTargetSnapshot:
        """Return one prioritized HDMI attention target snapshot."""

        camera = coerce_mapping(None if live_facts is None else live_facts.get("camera"))
        vad = coerce_mapping(None if live_facts is None else live_facts.get("vad"))
        continuous_target = (
            None
            if self._continuous_tracker is None
            else self._continuous_tracker.observe(
                observed_at=observed_at,
                live_facts=live_facts,
            )
        )
        checked_at = (
            _latest_timestamp(
                observed_at,
                _mapping_timestamp(camera),
                _mapping_timestamp(live_facts),
                _object_timestamp(continuous_target),
                _object_timestamp(speaker_association),
                _object_timestamp(identity_fusion),
            )
            or 0.0
        )
        normalized_runtime_status = (_normalize_optional_text(runtime_status) or "").lower() or None

        camera_is_fresh = _source_is_fresh(
            checked_at=checked_at,
            source_observed_at=_mapping_timestamp(camera) or _mapping_timestamp(live_facts),
            max_age_s=self.config.source_max_staleness_s,
            clock_skew_tolerance_s=self.config.clock_skew_tolerance_s,
        )
        camera_policy: Mapping[str, object] = (
            camera
            if camera_is_fresh
            else {
                "camera_online": False,
                "camera_ready": False,
            }
        )

        current_speaker_association = speaker_association
        if current_speaker_association is None:
            current_speaker_association = derive_respeaker_speaker_association(
                observed_at=observed_at,
                live_facts=(live_facts or {}),
            )
        if not _source_is_fresh(
            checked_at=checked_at,
            source_observed_at=_object_timestamp(current_speaker_association),
            max_age_s=self.config.source_max_staleness_s,
            clock_skew_tolerance_s=self.config.clock_skew_tolerance_s,
        ):
            current_speaker_association = None

        if not _source_is_fresh(
            checked_at=checked_at,
            source_observed_at=_object_timestamp(identity_fusion),
            max_age_s=self.config.identity_max_staleness_s,
            clock_skew_tolerance_s=self.config.clock_skew_tolerance_s,
        ):
            identity_fusion = None

        current_anchor = _anchor_from_continuous_target(
            continuous_target,
            checked_at=checked_at,
            config=self.config,
        )
        current_camera_anchor = _camera_anchor(
            camera_policy,
            default_observed_at=_mapping_timestamp(camera) or _mapping_timestamp(live_facts),
        )
        if current_anchor is None:
            current_anchor = current_camera_anchor

        speech_detected = coerce_optional_bool(vad.get("speech_detected")) is True
        speaker_associated = (
            current_speaker_association is not None
            and coerce_optional_bool(getattr(current_speaker_association, "associated", None)) is True
        )
        speaking_to_visible_person = (
            current_anchor is not None
            and speech_detected
            and (
                (continuous_target is not None and continuous_target.speaker_locked)
                or speaker_associated
            )
        )
        showing_intent_active = current_camera_anchor is not None and _showing_intent_active(camera_policy)

        if speaking_to_visible_person:
            assert current_anchor is not None
            speaker_anchor = _AttentionAnchor(
                horizontal=current_anchor.horizontal,
                vertical=current_anchor.vertical,
                source="speaker_association",
                track_id=current_anchor.track_id,
                center_x=current_anchor.center_x,
                center_y=current_anchor.center_y,
                velocity_x=current_anchor.velocity_x,
                observed_at=current_anchor.observed_at,
            )
            self._remember_focus(
                observed_at=checked_at,
                presence_session_id=presence_session_id,
                anchor=speaker_anchor,
            )
            return _snapshot_from_anchor(
                observed_at=observed_at,
                runtime_status=normalized_runtime_status,
                presence_session_id=presence_session_id,
                anchor=speaker_anchor,
                state=(
                    "active_visible_speaker_track"
                    if continuous_target is not None and continuous_target.speaker_locked
                    else "active_visible_speaker"
                ),
                speaker_locked=True,
                confidence=mean_confidence(
                    (
                        None if continuous_target is None else continuous_target.confidence,
                        None if current_speaker_association is None else _coerce_optional_ratio_attr(
                            current_speaker_association,
                            "confidence",
                        ),
                        _camera_anchor_confidence(camera_policy),
                    )
                )
                or 0.82,
            )

        if current_anchor is not None and showing_intent_active and not _showing_intent_must_yield(
            camera=camera_policy,
            speech_detected=speech_detected,
            continuous_target=continuous_target,
        ):
            showing_anchor = _AttentionAnchor(
                horizontal=current_anchor.horizontal,
                vertical=current_anchor.vertical,
                source="showing_intent",
                track_id=current_anchor.track_id,
                center_x=current_anchor.center_x,
                center_y=current_anchor.center_y,
                velocity_x=current_anchor.velocity_x,
                observed_at=current_anchor.observed_at,
            )
            self._remember_focus(
                observed_at=checked_at,
                presence_session_id=presence_session_id,
                anchor=showing_anchor,
            )
            return _snapshot_from_anchor(
                observed_at=observed_at,
                runtime_status=normalized_runtime_status,
                presence_session_id=presence_session_id,
                anchor=showing_anchor,
                state="showing_intent_visible_person",
                showing_intent_active=True,
                confidence=mean_confidence(
                    (
                        None if continuous_target is None else continuous_target.confidence,
                        _camera_anchor_confidence(camera_policy),
                        coerce_optional_ratio(camera_policy.get("visual_attention_score")),
                        0.82,
                    )
                )
                or 0.82,
            )

        focus_anchor = self._resolve_focus_anchor(
            observed_at=checked_at,
            runtime_status=normalized_runtime_status,
            presence_session_id=presence_session_id,
            current_anchor=current_anchor,
            camera=camera_policy,
            identity_fusion=identity_fusion,
        )
        if current_anchor is not None and focus_anchor is not None and _prefer_focus_anchor(
            camera=camera_policy,
            runtime_status=normalized_runtime_status,
            focus_anchor=focus_anchor,
            current_anchor=current_anchor,
            continuous_target=continuous_target,
            config=self.config,
        ):
            return _snapshot_from_anchor(
                observed_at=observed_at,
                runtime_status=normalized_runtime_status,
                presence_session_id=presence_session_id,
                anchor=focus_anchor,
                state="session_focus_locked",
                session_focus_active=True,
                confidence=mean_confidence(
                    (
                        None if continuous_target is None else continuous_target.confidence,
                        _focus_anchor_confidence(identity_fusion, camera_policy),
                        _camera_anchor_confidence(camera_policy),
                    )
                )
                or 0.78,
            )

        if current_anchor is not None:
            if (
                continuous_target is not None
                and continuous_target.active
                and continuous_target.state != "active_visible_person"
            ):
                self._remember_focus(
                    observed_at=checked_at,
                    presence_session_id=presence_session_id,
                    anchor=current_anchor,
                )
                return _snapshot_from_anchor(
                    observed_at=observed_at,
                    runtime_status=normalized_runtime_status,
                    presence_session_id=presence_session_id,
                    anchor=current_anchor,
                    state=continuous_target.state,
                    confidence=mean_confidence(
                        (
                            continuous_target.confidence,
                            _camera_anchor_confidence(camera_policy),
                        )
                    )
                    or 0.74,
                    speaker_locked=continuous_target.speaker_locked,
                    showing_intent_active=showing_intent_active,
                )
            if focus_anchor is not None and _matches_focus(
                current_anchor,
                focus_anchor,
                config=self.config,
            ):
                matched_anchor = _merge_preferred_focus_with_current(focus_anchor, current_anchor)
                self._remember_focus(
                    observed_at=checked_at,
                    presence_session_id=presence_session_id,
                    anchor=matched_anchor,
                )
                return _snapshot_from_anchor(
                    observed_at=observed_at,
                    runtime_status=normalized_runtime_status,
                    presence_session_id=presence_session_id,
                    anchor=matched_anchor,
                    state="session_focus_visible_person",
                    session_focus_active=True,
                    confidence=mean_confidence(
                        (
                            None if continuous_target is None else continuous_target.confidence,
                            _camera_anchor_confidence(camera_policy),
                            _focus_anchor_confidence(identity_fusion, camera_policy),
                        )
                    )
                    or 0.76,
                )
            visible_track_count = None if continuous_target is None else continuous_target.visible_track_count
            person_count = coerce_optional_int(camera_policy.get("person_count"))
            if (
                normalized_runtime_status in _ACTIVE_RUNTIME_STATES
                and (
                    (visible_track_count is not None and visible_track_count <= 1)
                    or person_count is None
                    or person_count <= 1
                )
            ):
                self._remember_focus(
                    observed_at=checked_at,
                    presence_session_id=presence_session_id,
                    anchor=current_anchor,
                )
            return _snapshot_from_anchor(
                observed_at=observed_at,
                runtime_status=normalized_runtime_status,
                presence_session_id=presence_session_id,
                anchor=current_anchor,
                state="visible_primary_person",
                confidence=mean_confidence(
                    (
                        None if continuous_target is None else continuous_target.confidence,
                        _camera_anchor_confidence(camera_policy),
                    )
                )
                or 0.7,
            )

        if focus_anchor is not None and _allow_focus_hold(camera=camera_policy, runtime_status=normalized_runtime_status):
            return _snapshot_from_anchor(
                observed_at=observed_at,
                runtime_status=normalized_runtime_status,
                presence_session_id=presence_session_id,
                anchor=focus_anchor,
                state="holding_session_focus",
                session_focus_active=True,
                confidence=_focus_anchor_confidence(identity_fusion, camera_policy) or 0.72,
            )

        return MultimodalAttentionTargetSnapshot(
            observed_at=observed_at,
            state="inactive",
            active=False,
            runtime_status=normalized_runtime_status,
            presence_session_id=presence_session_id,
            confidence=0.0,
        )

    def debug_snapshot(self, *, observed_at: float | None) -> dict[str, object]:
        """Return one bounded debug snapshot of the targeting internals."""

        checked_at = _latest_timestamp(observed_at) or 0.0
        focus_memory = self._focus_memory
        focus_summary = None
        if focus_memory is not None:
            focus_age = checked_at - focus_memory.updated_at
            if focus_age < 0.0:
                focus_age = 0.0
            focus_summary = {
                "presence_session_id": focus_memory.presence_session_id,
                "source": focus_memory.source,
                "track_id": focus_memory.track_id,
                "center_x": focus_memory.center_x,
                "center_y": focus_memory.center_y,
                "velocity_x": focus_memory.velocity_x,
                "age_s": round(focus_age, 3),
            }
        continuous_summary = None
        if self._continuous_tracker is not None:
            continuous_summary = self._continuous_tracker.debug_snapshot(observed_at=observed_at)
        return {
            "session_focus_memory": focus_summary,
            "continuous_tracker": continuous_summary,
        }

    def _remember_focus(
        self,
        *,
        observed_at: float,
        presence_session_id: int | None,
        anchor: _AttentionAnchor,
    ) -> None:
        anchor_observed_at = _coerce_optional_float(anchor.observed_at)
        if anchor_observed_at is not None and not _source_is_fresh(
            checked_at=observed_at,
            source_observed_at=anchor_observed_at,
            max_age_s=self.config.identity_max_staleness_s,
            clock_skew_tolerance_s=self.config.clock_skew_tolerance_s,
        ):
            return
        updated_at = observed_at
        if anchor_observed_at is not None and anchor_observed_at <= (observed_at + self.config.clock_skew_tolerance_s):
            updated_at = min(observed_at, anchor_observed_at)
        self._focus_memory = _SessionFocusMemory(
            presence_session_id=presence_session_id,
            horizontal=anchor.horizontal,
            vertical=anchor.vertical,
            source=anchor.source,
            track_id=anchor.track_id,
            center_x=anchor.center_x,
            center_y=anchor.center_y,
            velocity_x=anchor.velocity_x,
            updated_at=updated_at,
        )

    def _resolve_focus_anchor(
        self,
        *,
        observed_at: float,
        runtime_status: str | None,
        presence_session_id: int | None,
        current_anchor: _AttentionAnchor | None,
        camera: Mapping[str, object],
        identity_fusion: MultimodalIdentityFusionSnapshot | None,
    ) -> _AttentionAnchor | None:
        identity_anchor = _identity_focus_anchor(
            identity_fusion=identity_fusion,
            presence_session_id=presence_session_id,
            current_anchor=current_anchor,
        )
        remembered = self._remembered_focus_anchor(
            observed_at=observed_at,
            presence_session_id=presence_session_id,
        )
        if identity_anchor is not None:
            if current_anchor is not None and _matches_focus(
                current_anchor,
                identity_anchor,
                config=self.config,
            ):
                merged_anchor = _merge_preferred_focus_with_current(identity_anchor, current_anchor)
                self._remember_focus(
                    observed_at=observed_at,
                    presence_session_id=presence_session_id,
                    anchor=merged_anchor,
                )
                return merged_anchor
            self._remember_focus(
                observed_at=observed_at,
                presence_session_id=presence_session_id,
                anchor=identity_anchor,
            )
            return identity_anchor
        if remembered is None:
            return None
        if runtime_status not in _ACTIVE_RUNTIME_STATES and not _allow_focus_hold(
            camera=camera,
            runtime_status=runtime_status,
        ):
            return None
        return remembered

    def _remembered_focus_anchor(
        self,
        *,
        observed_at: float,
        presence_session_id: int | None,
    ) -> _AttentionAnchor | None:
        memory = self._focus_memory
        if memory is None:
            return None
        if memory.presence_session_id != presence_session_id:
            return None
        age_s = observed_at - memory.updated_at
        if age_s < -self.config.clock_skew_tolerance_s:
            return None
        if age_s < 0.0:
            age_s = 0.0
        if age_s > self.config.session_focus_hold_s:
            return None
        return _AttentionAnchor(
            horizontal=memory.horizontal,
            vertical=memory.vertical,
            source=memory.source,
            track_id=memory.track_id,
            center_x=memory.center_x,
            center_y=memory.center_y,
            velocity_x=memory.velocity_x,
            observed_at=memory.updated_at,
        )


def _snapshot_from_anchor(
    *,
    observed_at: float | None,
    runtime_status: str | None,
    presence_session_id: int | None,
    anchor: _AttentionAnchor,
    state: str,
    confidence: float,
    speaker_locked: bool = False,
    session_focus_active: bool = False,
    showing_intent_active: bool = False,
) -> MultimodalAttentionTargetSnapshot:
    return MultimodalAttentionTargetSnapshot(
        observed_at=observed_at,
        state=state,
        active=True,
        target_horizontal=anchor.horizontal,
        target_vertical=anchor.vertical,
        target_zone=anchor.horizontal,
        target_track_id=anchor.track_id,
        target_center_x=anchor.center_x,
        target_center_y=anchor.center_y,
        target_velocity_x=anchor.velocity_x,
        focus_source=anchor.source,
        runtime_status=runtime_status,
        presence_session_id=presence_session_id,
        speaker_locked=speaker_locked,
        session_focus_active=session_focus_active,
        showing_intent_active=showing_intent_active,
        confidence=round(max(0.0, min(1.0, confidence)), 4),
    )


def _camera_anchor(
    camera: Mapping[str, object],
    *,
    default_observed_at: float | None = None,
) -> _AttentionAnchor | None:
    if coerce_optional_bool(camera.get("person_visible")) is not True:
        return None
    horizontal = _camera_horizontal(camera)
    if horizontal is None:
        return None
    return _AttentionAnchor(
        horizontal=horizontal,
        vertical=_FOLLOW_VERTICAL,
        source="camera_primary_person",
        track_id=_normalize_optional_text(camera.get("primary_person_track_id")) or None,
        center_x=coerce_optional_ratio(camera.get("primary_person_center_x")),
        center_y=coerce_optional_ratio(camera.get("primary_person_center_y")),
        velocity_x=_coerce_optional_float(camera.get("primary_person_velocity_x")),
        observed_at=_mapping_timestamp(camera) or default_observed_at,
    )


def _anchor_from_continuous_target(
    snapshot: ContinuousAttentionTargetSnapshot | None,
    *,
    checked_at: float,
    config: MultimodalAttentionTargetConfig,
) -> _AttentionAnchor | None:
    """Translate one continuous visible-person target into the generic anchor."""

    if snapshot is None or snapshot.active is not True:
        return None
    observed_at = _object_timestamp(snapshot)
    if not _source_is_fresh(
        checked_at=checked_at,
        source_observed_at=observed_at,
        max_age_s=config.source_max_staleness_s,
        clock_skew_tolerance_s=config.clock_skew_tolerance_s,
    ):
        return None
    horizontal = _normalize_direction(snapshot.target_horizontal, allowed=_VALID_HORIZONTAL)
    if horizontal is None:
        horizontal = _camera_horizontal(
            {
                "primary_person_center_x": snapshot.target_center_x,
                "primary_person_zone": snapshot.target_zone,
            }
        )
    if horizontal is None:
        return None
    return _AttentionAnchor(
        horizontal=horizontal,
        vertical=_FOLLOW_VERTICAL,
        source=snapshot.focus_source,
        track_id=snapshot.target_track_id,
        center_x=snapshot.target_center_x,
        center_y=snapshot.target_center_y,
        velocity_x=snapshot.target_velocity_x,
        observed_at=observed_at,
    )


def _camera_horizontal(camera: Mapping[str, object]) -> str | None:
    if coerce_optional_bool(camera.get("primary_person_center_x_unknown")) is not True:
        center_x = coerce_optional_ratio(camera.get("primary_person_center_x"))
        if center_x is not None:
            if center_x <= _LEFT_THRESHOLD:
                return "left"
            if center_x >= _RIGHT_THRESHOLD:
                return "right"
            return "center"
    if coerce_optional_bool(camera.get("primary_person_zone_unknown")) is True:
        return None
    return _normalize_direction(camera.get("primary_person_zone"), allowed=_VALID_HORIZONTAL)


def _showing_intent_active(camera: Mapping[str, object]) -> bool:
    if coerce_optional_bool(camera.get("showing_intent_likely")) is True:
        return True
    return (
        coerce_optional_bool(camera.get("person_near_device")) is True
        and coerce_optional_bool(camera.get("looking_toward_device")) is True
    )


def _showing_intent_must_yield(
    *,
    camera: Mapping[str, object],
    speech_detected: bool,
    continuous_target: ContinuousAttentionTargetSnapshot | None,
) -> bool:
    """Return whether showing-intent must not override multi-person targeting.

    In actual household scenes Twinr should prefer the speaking person, and if
    nobody is speaking then the most recently moving visible person. A generic
    showing-intent flag must not erase those higher-value targets just because
    one participant happens to be near the device.
    """

    person_count = coerce_optional_int(camera.get("person_count")) or 0
    if continuous_target is not None and continuous_target.visible_track_count > 1:
        return True
    if person_count > 1 and speech_detected:
        return True
    return False


def _identity_focus_anchor(
    *,
    identity_fusion: MultimodalIdentityFusionSnapshot | None,
    presence_session_id: int | None,
    current_anchor: _AttentionAnchor | None,
) -> _AttentionAnchor | None:
    if identity_fusion is None:
        return None
    if identity_fusion.presence_session_id != presence_session_id:
        return None
    if identity_fusion.temporal_state != "stable_multimodal_match":
        return None
    if identity_fusion.track_consistency_state not in {"stable_anchor", "speaker_locked"}:
        return None
    center_x = _coerce_optional_ratio_attr(
        identity_fusion,
        "track_anchor_center_x",
        "track_center_x",
        "center_x",
    )
    horizontal = _normalize_direction(getattr(identity_fusion, "track_anchor_zone", None), allowed=_VALID_HORIZONTAL)
    if horizontal is None and center_x is not None:
        horizontal = _camera_horizontal(
            {
                "primary_person_center_x": center_x,
            }
        )
    if horizontal is None:
        return None
    vertical = _FOLLOW_VERTICAL if current_anchor is None else current_anchor.vertical
    return _AttentionAnchor(
        horizontal=horizontal,
        vertical=vertical,
        source="identity_fusion_track",
        track_id=_normalize_optional_text_attr(
            identity_fusion,
            "track_anchor_track_id",
            "anchor_track_id",
            "track_id",
        ),
        center_x=center_x,
        center_y=_coerce_optional_ratio_attr(
            identity_fusion,
            "track_anchor_center_y",
            "track_center_y",
            "center_y",
        ),
        velocity_x=_coerce_optional_float_attr(
            identity_fusion,
            "track_anchor_velocity_x",
            "track_velocity_x",
            "velocity_x",
        ),
        observed_at=_object_timestamp(identity_fusion),
    )


def _prefer_focus_anchor(
    *,
    camera: Mapping[str, object],
    runtime_status: str | None,
    focus_anchor: _AttentionAnchor,
    current_anchor: _AttentionAnchor,
    continuous_target: ContinuousAttentionTargetSnapshot | None,
    config: MultimodalAttentionTargetConfig,
) -> bool:
    if runtime_status not in _INTERACTIVE_RUNTIME_STATES:
        return False
    person_count = coerce_optional_int(camera.get("person_count"))
    visible_track_count = None if continuous_target is None else continuous_target.visible_track_count
    multi_person_visible = (
        (visible_track_count is not None and visible_track_count > 1)
        or (person_count is not None and person_count > 1)
    )
    if not multi_person_visible:
        return False
    return not _matches_focus(focus_anchor, current_anchor, config=config)


def _allow_focus_hold(*, camera: Mapping[str, object], runtime_status: str | None) -> bool:
    if _camera_attention_unavailable(camera):
        return False
    if runtime_status in _INTERACTIVE_RUNTIME_STATES:
        return True
    return coerce_optional_bool(camera.get("person_recently_visible")) is True


def _matches_focus(
    left: _AttentionAnchor,
    right: _AttentionAnchor,
    *,
    config: MultimodalAttentionTargetConfig,
) -> bool:
    if left.track_id and right.track_id:
        return left.track_id == right.track_id
    if left.center_x is not None and right.center_x is not None:
        if abs(left.center_x - right.center_x) <= config.focus_match_center_x_delta:
            if left.center_y is None or right.center_y is None:
                return True
            return abs(left.center_y - right.center_y) <= config.focus_match_center_y_delta
    return left.horizontal == right.horizontal


def _merge_preferred_focus_with_current(
    preferred_focus: _AttentionAnchor,
    current_anchor: _AttentionAnchor,
) -> _AttentionAnchor:
    return _AttentionAnchor(
        horizontal=current_anchor.horizontal,
        vertical=current_anchor.vertical,
        source=preferred_focus.source,
        track_id=current_anchor.track_id or preferred_focus.track_id,
        center_x=current_anchor.center_x if current_anchor.center_x is not None else preferred_focus.center_x,
        center_y=current_anchor.center_y if current_anchor.center_y is not None else preferred_focus.center_y,
        velocity_x=current_anchor.velocity_x if current_anchor.velocity_x is not None else preferred_focus.velocity_x,
        observed_at=current_anchor.observed_at if current_anchor.observed_at is not None else preferred_focus.observed_at,
    )


def _camera_attention_unavailable(camera: Mapping[str, object]) -> bool:
    return (
        coerce_optional_bool(camera.get("camera_online")) is False
        or coerce_optional_bool(camera.get("camera_ready")) is False
    )


def _camera_anchor_confidence(camera: Mapping[str, object]) -> float | None:
    base_confidence = mean_confidence(
        (
            coerce_optional_ratio(camera.get("visual_attention_score")),
            coerce_optional_ratio(camera.get("primary_person_confidence")),
            coerce_optional_ratio(camera.get("primary_person_tracking_confidence")),
            0.84 if coerce_optional_bool(camera.get("engaged_with_device")) is True else None,
            0.8 if coerce_optional_bool(camera.get("looking_toward_device")) is True else None,
            0.76 if coerce_optional_bool(camera.get("person_near_device")) is True else None,
        )
    )
    if base_confidence is not None:
        return base_confidence
    if coerce_optional_bool(camera.get("person_visible")) is True:
        return 0.7
    return None


def _focus_anchor_confidence(
    identity_fusion: MultimodalIdentityFusionSnapshot | None,
    camera: Mapping[str, object],
) -> float | None:
    if identity_fusion is not None:
        fusion_confidence = coerce_optional_ratio(getattr(getattr(identity_fusion, "claim", None), "confidence", None))
        if fusion_confidence is not None:
            return fusion_confidence
    return _camera_anchor_confidence(camera)


def _normalize_direction(value: object | None, *, allowed: frozenset[str]) -> str | None:
    normalized = (_normalize_optional_text(value) or "").lower()
    if normalized in allowed:
        return normalized
    return None


def _normalize_optional_text(value: object | None) -> str | None:
    normalized = normalize_text(value)
    if not normalized:
        return None
    return normalized


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


def _bounded_config_float(
    *,
    config: TwinrConfig,
    attr_name: str,
    default: float,
    minimum: float,
) -> float:
    try:
        candidate = float(getattr(config, attr_name, default))
    except (TypeError, ValueError):
        candidate = default
    if not math.isfinite(candidate):
        candidate = default
    if candidate < minimum:
        candidate = default
    return max(minimum, candidate)


def _latest_timestamp(*values: object | None) -> float | None:
    latest = None
    for value in values:
        candidate = _coerce_optional_float(value)
        if candidate is None:
            continue
        if latest is None or candidate > latest:
            latest = candidate
    return latest


def _mapping_timestamp(mapping: Mapping[str, object] | None) -> float | None:
    if not mapping:
        return None
    for key in ("observed_at", "captured_at", "frame_observed_at", "updated_at", "timestamp"):
        value = _coerce_optional_float(mapping.get(key))
        if value is not None:
            return value
    return None


def _object_timestamp(value: object | None) -> float | None:
    if value is None:
        return None
    for attr_name in ("observed_at", "captured_at", "frame_observed_at", "updated_at", "timestamp"):
        candidate = _coerce_optional_float(getattr(value, attr_name, None))
        if candidate is not None:
            return candidate
    claim = getattr(value, "claim", None)
    if claim is not None:
        for attr_name in ("observed_at", "captured_at", "updated_at", "timestamp"):
            candidate = _coerce_optional_float(getattr(claim, attr_name, None))
            if candidate is not None:
                return candidate
    return None


def _source_is_fresh(
    *,
    checked_at: float,
    source_observed_at: float | None,
    max_age_s: float,
    clock_skew_tolerance_s: float,
) -> bool:
    if source_observed_at is None:
        return True
    age_s = checked_at - source_observed_at
    if age_s < -clock_skew_tolerance_s:
        return False
    if age_s < 0.0:
        age_s = 0.0
    return age_s <= max_age_s


def _coerce_optional_ratio_attr(value: object | None, *attr_names: str) -> float | None:
    for attr_name in attr_names:
        candidate = coerce_optional_ratio(getattr(value, attr_name, None))
        if candidate is not None:
            return candidate
    return None


def _coerce_optional_float_attr(value: object | None, *attr_names: str) -> float | None:
    for attr_name in attr_names:
        candidate = _coerce_optional_float(getattr(value, attr_name, None))
        if candidate is not None:
            return candidate
    return None


def _normalize_optional_text_attr(value: object | None, *attr_names: str) -> str | None:
    for attr_name in attr_names:
        candidate = _normalize_optional_text(getattr(value, attr_name, None))
        if candidate:
            return candidate
    return None


__all__ = [
    "MultimodalAttentionTargetConfig",
    "MultimodalAttentionTargetSnapshot",
    "MultimodalAttentionTargetTracker",
]