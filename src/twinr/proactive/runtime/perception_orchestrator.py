"""Centralize authoritative runtime perception truth for proactive consumers.

This module turns one camera `perception_stream` observation plus the small
amount of runtime context that genuinely matters for consumption into one
authoritative runtime snapshot. HDMI eye-follow, servo follow, gesture
acknowledgement, and visual wake should all consume this orchestrated truth
instead of independently re-deriving temporal meaning from the same frame.
"""

from __future__ import annotations

from dataclasses import dataclass

from twinr.agent.base_agent.config import TwinrConfig
from twinr.proactive.social.camera_surface import ProactiveCameraSnapshot
from twinr.proactive.social.engine import SocialAudioObservation, SocialVisionObservation

from .attention_targeting import MultimodalAttentionTargetSnapshot, MultimodalAttentionTargetTracker
from .audio_policy import ReSpeakerAudioPolicySnapshot
from .display_gesture_emoji import DisplayGestureEmojiDecision
from .gesture_ack_lane import GestureAckLane
from .gesture_wakeup_lane import GestureWakeupDecision, GestureWakeupLane
from .identity_fusion import MultimodalIdentityFusionSnapshot
from .speaker_association import (
    ReSpeakerSpeakerAssociationSnapshot,
    derive_respeaker_speaker_association,
)


@dataclass(frozen=True, slots=True)
class PerceptionAttentionRuntimeSnapshot:
    """Describe one authoritative runtime attention result."""

    live_facts: dict[str, object]
    speaker_association: ReSpeakerSpeakerAssociationSnapshot
    attention_target: MultimodalAttentionTargetSnapshot
    attention_target_debug: dict[str, object]


@dataclass(frozen=True, slots=True)
class PerceptionGestureRuntimeSnapshot:
    """Describe one authoritative runtime gesture result."""

    ack_decision: DisplayGestureEmojiDecision
    wakeup_decision: GestureWakeupDecision


@dataclass(frozen=True, slots=True)
class PerceptionRuntimeSnapshot:
    """Bundle the current authoritative runtime perception state."""

    observed_at: float
    source: str | None = None
    captured_at: float | None = None
    attention: PerceptionAttentionRuntimeSnapshot | None = None
    gesture: PerceptionGestureRuntimeSnapshot | None = None


class PerceptionStreamOrchestrator:
    """Own the single runtime-facing interpretation of local perception state."""

    def __init__(
        self,
        *,
        attention_target_tracker: MultimodalAttentionTargetTracker,
        gesture_ack_lane: GestureAckLane,
        gesture_wakeup_lane: GestureWakeupLane,
    ) -> None:
        self._attention_target_tracker = attention_target_tracker
        self._gesture_ack_lane = gesture_ack_lane
        self._gesture_wakeup_lane = gesture_wakeup_lane

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "PerceptionStreamOrchestrator":
        """Build one orchestrator from the global Twinr config."""

        return cls(
            attention_target_tracker=MultimodalAttentionTargetTracker.from_config(config),
            gesture_ack_lane=GestureAckLane.from_config(config),
            gesture_wakeup_lane=GestureWakeupLane.from_config(config),
        )

    def observe(
        self,
        *,
        observed_at: float,
        source: str | None = None,
        captured_at: float | None = None,
        include_attention: bool = False,
        include_gesture: bool = False,
        camera_snapshot: ProactiveCameraSnapshot | None = None,
        vision_observation: SocialVisionObservation | None = None,
        audio_observation: SocialAudioObservation | None = None,
        audio_policy_snapshot: ReSpeakerAudioPolicySnapshot | None = None,
        runtime_status: object | None = None,
        presence_session_id: int | None = None,
        identity_fusion: MultimodalIdentityFusionSnapshot | None = None,
        speaker_association: ReSpeakerSpeakerAssociationSnapshot | None = None,
    ) -> PerceptionRuntimeSnapshot:
        """Return one authoritative runtime snapshot for the requested consumers.

        Args:
            observed_at: Monotonic runtime timestamp for the current refresh.
            source: Human-readable lane/source label for bounded forensics.
            captured_at: Optional underlying camera capture timestamp.
            include_attention: Whether to derive attention/servo runtime truth.
            include_gesture: Whether to derive gesture ack/wakeup runtime truth.
            camera_snapshot: Stabilized camera snapshot used for attention truth.
            vision_observation: Raw social vision observation used for gesture truth.
            audio_observation: Current audio observation used for speaker targeting.
            audio_policy_snapshot: Current audio-policy overlay for attention facts.
            runtime_status: Current runtime status used by attention targeting.
            presence_session_id: Current presence-session id for focus memory.
            identity_fusion: Current multimodal identity-fusion snapshot.
            speaker_association: Optional precomputed speaker association to reuse.

        Returns:
            One snapshot that carries only the requested attention and/or
            gesture runtime results.

        Raises:
            ValueError: If the required inputs for a requested lane are absent.
        """

        attention = None
        if include_attention:
            attention = self._observe_attention(
                observed_at=observed_at,
                camera_snapshot=_require_camera_snapshot(camera_snapshot),
                audio_observation=audio_observation or SocialAudioObservation(),
                audio_policy_snapshot=audio_policy_snapshot,
                runtime_status=runtime_status,
                presence_session_id=presence_session_id,
                identity_fusion=identity_fusion,
                speaker_association=speaker_association,
            )

        gesture = None
        if include_gesture:
            gesture = self._observe_gesture(
                observed_at=observed_at,
                vision_observation=_require_vision_observation(vision_observation),
            )

        return PerceptionRuntimeSnapshot(
            observed_at=float(observed_at),
            source=_normalize_optional_text(source),
            captured_at=_coerce_optional_timestamp(captured_at),
            attention=attention,
            gesture=gesture,
        )

    def observe_attention(
        self,
        *,
        observed_at: float,
        camera_snapshot: ProactiveCameraSnapshot,
        audio_observation: SocialAudioObservation,
        audio_policy_snapshot: ReSpeakerAudioPolicySnapshot | None,
        runtime_status: object | None,
        presence_session_id: int | None,
        identity_fusion: MultimodalIdentityFusionSnapshot | None = None,
        speaker_association: ReSpeakerSpeakerAssociationSnapshot | None = None,
        source: str | None = None,
        captured_at: float | None = None,
    ) -> PerceptionRuntimeSnapshot:
        """Return one attention-only runtime snapshot."""

        return self.observe(
            observed_at=observed_at,
            source=source,
            captured_at=captured_at,
            include_attention=True,
            camera_snapshot=camera_snapshot,
            audio_observation=audio_observation,
            audio_policy_snapshot=audio_policy_snapshot,
            runtime_status=runtime_status,
            presence_session_id=presence_session_id,
            identity_fusion=identity_fusion,
            speaker_association=speaker_association,
        )

    def observe_gesture(
        self,
        *,
        observed_at: float,
        vision_observation: SocialVisionObservation,
        source: str | None = None,
        captured_at: float | None = None,
    ) -> PerceptionRuntimeSnapshot:
        """Return one gesture-only runtime snapshot."""

        return self.observe(
            observed_at=observed_at,
            source=source,
            captured_at=captured_at,
            include_gesture=True,
            vision_observation=vision_observation,
        )

    def _observe_attention(
        self,
        *,
        observed_at: float,
        camera_snapshot: ProactiveCameraSnapshot,
        audio_observation: SocialAudioObservation,
        audio_policy_snapshot: ReSpeakerAudioPolicySnapshot | None,
        runtime_status: object | None,
        presence_session_id: int | None,
        identity_fusion: MultimodalIdentityFusionSnapshot | None,
        speaker_association: ReSpeakerSpeakerAssociationSnapshot | None,
    ) -> PerceptionAttentionRuntimeSnapshot:
        """Derive one authoritative attention-target result."""

        live_facts = {
            "camera": camera_snapshot.to_automation_facts(),
            "vad": {
                "speech_detected": audio_observation.speech_detected,
            },
            "respeaker": {
                "azimuth_deg": audio_observation.azimuth_deg,
                "direction_confidence": audio_observation.direction_confidence,
            },
            "audio_policy": {
                "speaker_direction_stable": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.speaker_direction_stable
                ),
            },
        }
        current_speaker_association = speaker_association
        if current_speaker_association is None:
            current_speaker_association = derive_respeaker_speaker_association(
                observed_at=observed_at,
                live_facts=live_facts,
            )
        attention_target = self._attention_target_tracker.observe(
            observed_at=observed_at,
            live_facts=live_facts,
            runtime_status=runtime_status,
            presence_session_id=presence_session_id,
            speaker_association=current_speaker_association,
            identity_fusion=identity_fusion,
        )
        attention_target_debug = self._attention_target_tracker.debug_snapshot(observed_at=observed_at)
        live_facts["speaker_association"] = current_speaker_association.to_automation_facts()
        live_facts["attention_target"] = attention_target.to_automation_facts()
        return PerceptionAttentionRuntimeSnapshot(
            live_facts=live_facts,
            speaker_association=current_speaker_association,
            attention_target=attention_target,
            attention_target_debug=attention_target_debug,
        )

    def _observe_gesture(
        self,
        *,
        observed_at: float,
        vision_observation: SocialVisionObservation,
    ) -> PerceptionGestureRuntimeSnapshot:
        """Derive one authoritative gesture ack/wakeup result."""

        return PerceptionGestureRuntimeSnapshot(
            ack_decision=self._gesture_ack_lane.observe(
                observed_at=observed_at,
                observation=vision_observation,
            ),
            wakeup_decision=self._gesture_wakeup_lane.observe(
                observed_at=observed_at,
                observation=vision_observation,
            ),
        )


def _require_camera_snapshot(value: ProactiveCameraSnapshot | None) -> ProactiveCameraSnapshot:
    """Return one required camera snapshot or raise a clear contract error."""

    if value is None:
        raise ValueError("camera_snapshot is required when include_attention=True")
    return value


def _require_vision_observation(value: SocialVisionObservation | None) -> SocialVisionObservation:
    """Return one required vision observation or raise a clear contract error."""

    if value is None:
        raise ValueError("vision_observation is required when include_gesture=True")
    return value


def _normalize_optional_text(value: object) -> str | None:
    """Normalize one optional telemetry string conservatively."""

    text = str(value or "").strip()
    return text or None


def _coerce_optional_timestamp(value: object) -> float | None:
    """Return one optional finite timestamp-like value."""

    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


__all__ = [
    "PerceptionAttentionRuntimeSnapshot",
    "PerceptionGestureRuntimeSnapshot",
    "PerceptionRuntimeSnapshot",
    "PerceptionStreamOrchestrator",
]
