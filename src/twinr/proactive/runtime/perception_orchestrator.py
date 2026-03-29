"""Centralize authoritative runtime perception truth for proactive consumers.

This module turns one camera `perception_stream` observation plus the small
amount of runtime context that genuinely matters for consumption into one
authoritative runtime snapshot. HDMI eye-follow, servo follow, gesture
acknowledgement, and visual wake should all consume this orchestrated truth
instead of independently re-deriving temporal meaning from the same frame.
"""

from __future__ import annotations

# CHANGELOG: 2026-03-29
# BUG-1: Serialized access to the stateful tracker/gesture lanes to remove
#        race conditions and debug/state incoherence under concurrent callers.
# BUG-2: Frozen snapshots previously leaked mutable nested dict/list payloads;
#        published runtime truth is now recursively frozen before exposure.
# BUG-3: `observed_at`/`captured_at` accepted non-finite values and stale or
#        misaligned audio could silently steer speaker targeting.
# SEC-1: Mutable nested snapshot payloads allowed in-process consumers to
#        tamper with authoritative runtime truth after publication.
# IMP-1: Added sequence/provenance/freshness metadata for camera/audio/vision,
#        including audio↔camera skew classification.
# IMP-2: Added optional suppression of attention debug capture on hot paths.

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from threading import RLock

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

_DEFAULT_CAMERA_STALE_AFTER_S = 0.75
_DEFAULT_AUDIO_STALE_AFTER_S = 0.75
_DEFAULT_VISION_STALE_AFTER_S = 0.75
_DEFAULT_AUDIO_CAMERA_SKEW_TOLERANCE_S = 0.35


class FrozenDict(dict[str, object]):
    """Small read-only dict used to publish authoritative snapshot payloads."""

    __slots__ = ()

    def _immutable(self, *args: object, **kwargs: object) -> None:
        raise TypeError("FrozenDict is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


@dataclass(frozen=True, slots=True)
class PerceptionModalityTimingSnapshot:
    """Describe freshness for one modality contributing to runtime truth."""

    timestamp: float | None = None
    age_s: float | None = None
    fresh: bool | None = None
    status: str = "unknown"

    def to_automation_facts(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "age_s": self.age_s,
            "fresh": self.fresh,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class PerceptionRuntimeTimingSnapshot:
    """Describe provenance and freshness for the current runtime snapshot."""

    camera: PerceptionModalityTimingSnapshot = field(default_factory=PerceptionModalityTimingSnapshot)
    audio: PerceptionModalityTimingSnapshot = field(default_factory=PerceptionModalityTimingSnapshot)
    vision: PerceptionModalityTimingSnapshot = field(default_factory=PerceptionModalityTimingSnapshot)
    audio_camera_skew_s: float | None = None
    audio_camera_aligned: bool | None = None
    warnings: tuple[str, ...] = ()

    def to_automation_facts(self) -> dict[str, object]:
        return {
            "camera": self.camera.to_automation_facts(),
            "audio": self.audio.to_automation_facts(),
            "vision": self.vision.to_automation_facts(),
            "audio_camera_skew_s": self.audio_camera_skew_s,
            "audio_camera_aligned": self.audio_camera_aligned,
            "warnings": self.warnings,
        }


@dataclass(frozen=True, slots=True)
class PerceptionAttentionRuntimeSnapshot:
    """Describe one authoritative runtime attention result."""

    # BREAKING: nested payloads are now read-only recursive snapshots to prevent
    # post-publication mutation by downstream consumers.
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
    sequence_id: int = 0
    timing: PerceptionRuntimeTimingSnapshot = field(default_factory=PerceptionRuntimeTimingSnapshot)


class PerceptionStreamOrchestrator:
    """Own the single runtime-facing interpretation of local perception state."""

    def __init__(
        self,
        *,
        attention_target_tracker: MultimodalAttentionTargetTracker,
        gesture_ack_lane: GestureAckLane,
        gesture_wakeup_lane: GestureWakeupLane,
        camera_stale_after_s: float = _DEFAULT_CAMERA_STALE_AFTER_S,
        audio_stale_after_s: float = _DEFAULT_AUDIO_STALE_AFTER_S,
        vision_stale_after_s: float = _DEFAULT_VISION_STALE_AFTER_S,
        audio_camera_skew_tolerance_s: float = _DEFAULT_AUDIO_CAMERA_SKEW_TOLERANCE_S,
        gate_stale_audio_for_attention: bool = True,
    ) -> None:
        self._attention_target_tracker = attention_target_tracker
        self._gesture_ack_lane = gesture_ack_lane
        self._gesture_wakeup_lane = gesture_wakeup_lane
        self._camera_stale_after_s = _require_non_negative_finite(
            camera_stale_after_s,
            name="camera_stale_after_s",
        )
        self._audio_stale_after_s = _require_non_negative_finite(
            audio_stale_after_s,
            name="audio_stale_after_s",
        )
        self._vision_stale_after_s = _require_non_negative_finite(
            vision_stale_after_s,
            name="vision_stale_after_s",
        )
        self._audio_camera_skew_tolerance_s = _require_non_negative_finite(
            audio_camera_skew_tolerance_s,
            name="audio_camera_skew_tolerance_s",
        )
        self._gate_stale_audio_for_attention = bool(gate_stale_audio_for_attention)
        self._lock = RLock()
        self._sequence_id = 0

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
        include_attention_debug: bool = True,
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
            include_attention_debug: Whether to capture tracker debug state.
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

        resolved_observed_at = _require_finite_timestamp(observed_at, name="observed_at")
        resolved_source = _normalize_optional_text(source)
        resolved_captured_at = _coerce_optional_timestamp(captured_at)

        resolved_camera_snapshot = _require_camera_snapshot(camera_snapshot) if include_attention else camera_snapshot
        resolved_vision_observation = _require_vision_observation(vision_observation) if include_gesture else vision_observation
        resolved_audio_observation = audio_observation
        if include_attention and resolved_audio_observation is None:
            resolved_audio_observation = SocialAudioObservation()

        timing = self._build_runtime_timing(
            observed_at=resolved_observed_at,
            captured_at=resolved_captured_at,
            camera_snapshot=resolved_camera_snapshot,
            vision_observation=resolved_vision_observation,
            audio_observation=resolved_audio_observation,
        )
        published_captured_at = resolved_captured_at
        if published_captured_at is None:
            published_captured_at = timing.camera.timestamp
        if published_captured_at is None:
            published_captured_at = timing.vision.timestamp

        with self._lock:
            self._sequence_id += 1
            sequence_id = self._sequence_id

            attention = None
            if include_attention:
                attention = self._observe_attention(
                    observed_at=resolved_observed_at,
                    sequence_id=sequence_id,
                    timing=timing,
                    camera_snapshot=_require_camera_snapshot(resolved_camera_snapshot),
                    audio_observation=resolved_audio_observation or SocialAudioObservation(),
                    audio_policy_snapshot=audio_policy_snapshot,
                    runtime_status=runtime_status,
                    presence_session_id=presence_session_id,
                    identity_fusion=identity_fusion,
                    speaker_association=speaker_association,
                    include_attention_debug=include_attention_debug,
                )

            gesture = None
            if include_gesture:
                gesture = self._observe_gesture(
                    observed_at=resolved_observed_at,
                    vision_observation=_require_vision_observation(resolved_vision_observation),
                )

        return PerceptionRuntimeSnapshot(
            observed_at=resolved_observed_at,
            source=resolved_source,
            captured_at=published_captured_at,
            attention=attention,
            gesture=gesture,
            sequence_id=sequence_id,
            timing=timing,
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
        include_attention_debug: bool = True,
    ) -> PerceptionRuntimeSnapshot:
        """Return one attention-only runtime snapshot."""

        return self.observe(
            observed_at=observed_at,
            source=source,
            captured_at=captured_at,
            include_attention=True,
            include_attention_debug=include_attention_debug,
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

    def _build_runtime_timing(
        self,
        *,
        observed_at: float,
        captured_at: float | None,
        camera_snapshot: ProactiveCameraSnapshot | None,
        vision_observation: SocialVisionObservation | None,
        audio_observation: SocialAudioObservation | None,
    ) -> PerceptionRuntimeTimingSnapshot:
        """Build one freshness/provenance snapshot for the current observation."""

        warnings: list[str] = []
        camera_timestamp = captured_at
        if camera_timestamp is None:
            camera_timestamp = _extract_monotonic_timestamp(camera_snapshot)
        vision_timestamp = captured_at
        if vision_timestamp is None:
            vision_timestamp = _extract_monotonic_timestamp(vision_observation)
        audio_timestamp = _extract_monotonic_timestamp(audio_observation)

        camera_timing = _build_modality_timing(
            observed_at=observed_at,
            source_timestamp=camera_timestamp,
            stale_after_s=self._camera_stale_after_s,
            label="camera",
            warnings=warnings,
        )
        audio_timing = _build_modality_timing(
            observed_at=observed_at,
            source_timestamp=audio_timestamp,
            stale_after_s=self._audio_stale_after_s,
            label="audio",
            warnings=warnings,
        )
        vision_timing = _build_modality_timing(
            observed_at=observed_at,
            source_timestamp=vision_timestamp,
            stale_after_s=self._vision_stale_after_s,
            label="vision",
            warnings=warnings,
        )

        audio_camera_skew_s = None
        audio_camera_aligned = None
        if camera_timing.timestamp is not None and audio_timing.timestamp is not None:
            audio_camera_skew_s = abs(camera_timing.timestamp - audio_timing.timestamp)
            audio_camera_aligned = audio_camera_skew_s <= self._audio_camera_skew_tolerance_s
            if not audio_camera_aligned:
                warnings.append("audio_camera_skew")

        return PerceptionRuntimeTimingSnapshot(
            camera=camera_timing,
            audio=audio_timing,
            vision=vision_timing,
            audio_camera_skew_s=audio_camera_skew_s,
            audio_camera_aligned=audio_camera_aligned,
            warnings=tuple(warnings),
        )

    def _observe_attention(
        self,
        *,
        observed_at: float,
        sequence_id: int,
        timing: PerceptionRuntimeTimingSnapshot,
        camera_snapshot: ProactiveCameraSnapshot,
        audio_observation: SocialAudioObservation,
        audio_policy_snapshot: ReSpeakerAudioPolicySnapshot | None,
        runtime_status: object | None,
        presence_session_id: int | None,
        identity_fusion: MultimodalIdentityFusionSnapshot | None,
        speaker_association: ReSpeakerSpeakerAssociationSnapshot | None,
        include_attention_debug: bool,
    ) -> PerceptionAttentionRuntimeSnapshot:
        """Derive one authoritative attention-target result."""

        audio_targeting_enabled = True
        if self._gate_stale_audio_for_attention:
            audio_targeting_enabled = _timing_is_usable_for_targeting(timing.audio)
            if timing.audio_camera_aligned is False:
                audio_targeting_enabled = False

        camera_facts = _copy_mapping_like(
            camera_snapshot.to_automation_facts(),
            name="camera_snapshot.to_automation_facts()",
        )
        working_live_facts = {
            "camera": camera_facts,
            "vad": {
                # BREAKING: if timestamps are available and audio is stale or
                # misaligned, audio is now masked out of attention targeting.
                "speech_detected": (
                    audio_observation.speech_detected if audio_targeting_enabled else False
                ),
            },
            "respeaker": {
                "azimuth_deg": (
                    audio_observation.azimuth_deg if audio_targeting_enabled else None
                ),
                "direction_confidence": (
                    audio_observation.direction_confidence if audio_targeting_enabled else None
                ),
            },
            "audio_policy": {
                "speaker_direction_stable": (
                    None
                    if audio_policy_snapshot is None or not audio_targeting_enabled
                    else audio_policy_snapshot.speaker_direction_stable
                ),
            },
            "runtime": {
                "sequence_id": sequence_id,
                "observed_at": observed_at,
                "audio_targeting_enabled": audio_targeting_enabled,
                "timing": timing.to_automation_facts(),
                "warnings": timing.warnings,
            },
        }

        current_speaker_association = speaker_association
        if current_speaker_association is None or not audio_targeting_enabled:
            current_speaker_association = derive_respeaker_speaker_association(
                observed_at=observed_at,
                live_facts=working_live_facts,
            )

        attention_target = self._attention_target_tracker.observe(
            observed_at=observed_at,
            live_facts=working_live_facts,
            runtime_status=runtime_status,
            presence_session_id=presence_session_id,
            speaker_association=current_speaker_association,
            identity_fusion=identity_fusion,
        )

        attention_target_debug = {}
        if include_attention_debug:
            attention_target_debug = _copy_mapping_like(
                self._attention_target_tracker.debug_snapshot(observed_at=observed_at),
                name="attention_target_tracker.debug_snapshot()",
            )

        published_live_facts = deepcopy(working_live_facts)
        published_live_facts["speaker_association"] = _copy_mapping_like(
            current_speaker_association.to_automation_facts(),
            name="speaker_association.to_automation_facts()",
        )
        published_live_facts["attention_target"] = _copy_mapping_like(
            attention_target.to_automation_facts(),
            name="attention_target.to_automation_facts()",
        )

        return PerceptionAttentionRuntimeSnapshot(
            live_facts=_freeze_mapping(published_live_facts),
            speaker_association=current_speaker_association,
            attention_target=attention_target,
            attention_target_debug=_freeze_mapping(attention_target_debug),
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


def _require_finite_timestamp(value: object, *, name: str) -> float:
    """Return one required finite timestamp-like value or raise."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite float-like timestamp")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float-like timestamp") from exc
    if not isfinite(numeric):
        raise ValueError(f"{name} must be a finite float-like timestamp")
    return numeric


def _coerce_optional_timestamp(value: object) -> float | None:
    """Return one optional finite timestamp-like value."""

    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(numeric):
        return None
    return numeric


def _require_non_negative_finite(value: object, *, name: str) -> float:
    """Return one required non-negative finite scalar or raise."""

    numeric = _require_finite_timestamp(value, name=name)
    if numeric < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return numeric


def _extract_monotonic_timestamp(value: object | None) -> float | None:
    """Extract one optional monotonic-like timestamp from a known observation."""

    if value is None:
        return None
    for attr_name in (
        "observed_at",
        "captured_at",
        "monotonic_at",
        "monotonic_ts",
        "received_at",
        "received_monotonic_at",
    ):
        try:
            candidate = getattr(value, attr_name)
        except AttributeError:
            continue
        coerced = _coerce_optional_timestamp(candidate)
        if coerced is not None:
            return coerced
    return None


def _build_modality_timing(
    *,
    observed_at: float,
    source_timestamp: float | None,
    stale_after_s: float,
    label: str,
    warnings: list[str],
) -> PerceptionModalityTimingSnapshot:
    """Build freshness classification for one modality."""

    if source_timestamp is None:
        return PerceptionModalityTimingSnapshot()

    age_s = observed_at - source_timestamp
    if age_s < 0.0:
        warnings.append(f"{label}_timestamp_in_future")
        return PerceptionModalityTimingSnapshot(
            timestamp=source_timestamp,
            age_s=age_s,
            fresh=False,
            status="future",
        )
    if age_s <= stale_after_s:
        return PerceptionModalityTimingSnapshot(
            timestamp=source_timestamp,
            age_s=age_s,
            fresh=True,
            status="fresh",
        )
    warnings.append(f"{label}_stale")
    return PerceptionModalityTimingSnapshot(
        timestamp=source_timestamp,
        age_s=age_s,
        fresh=False,
        status="stale",
    )


def _timing_is_usable_for_targeting(value: PerceptionModalityTimingSnapshot) -> bool:
    """Return whether one modality timing snapshot is safe to use for targeting."""

    return value.fresh is not False


def _copy_mapping_like(value: object, *, name: str) -> dict[str, object]:
    """Return one detached dict copy from a mapping-like value."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return deepcopy(dict(value))


def _freeze_mapping(value: Mapping[str, object]) -> dict[str, object]:
    """Return one recursively frozen mapping published to consumers."""

    frozen = _freeze_value(dict(value))
    if not isinstance(frozen, dict):
        raise TypeError("frozen snapshot payload must be a dict-like mapping")
    return frozen


def _freeze_value(value: object) -> object:
    """Recursively freeze runtime payloads without losing JSON-like shape."""

    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, Mapping):
        return FrozenDict({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        ordered_items = sorted(value, key=repr)
        return tuple(_freeze_value(item) for item in ordered_items)
    return value


__all__ = [
    "PerceptionAttentionRuntimeSnapshot",
    "PerceptionGestureRuntimeSnapshot",
    "PerceptionModalityTimingSnapshot",
    "PerceptionRuntimeSnapshot",
    "PerceptionRuntimeTimingSnapshot",
    "PerceptionStreamOrchestrator",
]