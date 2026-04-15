# CHANGELOG: 2026-03-29
# BUG-1: Prevented infinite dropout-hold ghost persons by never re-caching held-only visibility as a fresh visible frame.
# BUG-2: Prevented stale pose/hand semantics from self-refreshing forever by tracking semantic-origin timestamps separately from fused-frame timestamps.
# BUG-3: Rejected out-of-order older full/gesture observations and replaced wall-clock-only aging with monotonic fallback to stop time-warp regressions.
# BUG-4: Exposed one bounded recent-hand-semantics query so transcript-first voice runtimes can arm the heavier gesture lane only when fresh hand/intent evidence exists.
# SEC-1: Sanitized non-finite timestamps, anchors, and confidences to block data-plane poisoning that could previously pin stale semantics indefinitely.
# IMP-1: Added thread-safe multi-lane state access plus confidence-decayed, uncertainty-aware semantic transfer.
# IMP-2: Added frontier-grade identity gating hooks (track-id and optional bbox IoU) while preserving center/zone fallback.

from __future__ import annotations
from dataclasses import dataclass, replace
import math
import threading
import time
from twinr.agent.base_agent.config import TwinrConfig
from ..social.engine import SocialBodyPose, SocialFineHandGesture, SocialGestureEvent, SocialMotionState, SocialPersonZone, SocialVisionObservation
from .display_attention import resolve_display_attention_refresh_interval

def _finite_optional_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed

def _clamp(value: float, *, minimum: float | None=None, maximum: float | None=None) -> float:
    if minimum is not None and value < minimum:
        value = minimum
    if maximum is not None and value > maximum:
        value = maximum
    return value

def _config_float(config: TwinrConfig, name: str, *, default: float, minimum: float | None=None, maximum: float | None=None) -> float:
    parsed = _finite_optional_float(getattr(config, name, default))
    if parsed is None:
        parsed = float(default)
    return _clamp(parsed, minimum=minimum, maximum=maximum)

def _config_bool(config: TwinrConfig, name: str, *, default: bool) -> bool:
    value = getattr(config, name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'1', 'true', 'yes', 'on'}:
            return True
        if normalized in {'0', 'false', 'no', 'off'}:
            return False
    return bool(default)

@dataclass(frozen=True, slots=True)
class DisplayAttentionCameraFusionConfig:
    dropout_hold_s: float
    gesture_semantic_hold_s: float
    pose_semantic_hold_s: float
    anchor_match_delta_x: float = 0.2
    anchor_match_delta_y: float = 0.24
    anchor_match_iou: float = 0.18
    blind_single_person_match_max_age_s: float = 0.35
    pose_decay_tau_s: float = 3.0
    gesture_decay_tau_s: float = 1.2
    min_pose_transfer_score: float = 0.14
    min_hand_transfer_score: float = 0.18
    clock_skew_tolerance_s: float = 0.35
    prefer_track_id_match: bool = True

    @classmethod
    def from_config(cls, config: TwinrConfig) -> 'DisplayAttentionCameraFusionConfig':
        refresh_interval_s = _finite_optional_float(resolve_display_attention_refresh_interval(config))
        if refresh_interval_s is None:
            refresh_interval_s = _config_float(config, 'display_attention_refresh_interval_s', default=0.35, minimum=0.2)
        refresh_interval_s = max(0.2, refresh_interval_s)
        proactive_capture_interval_s = _config_float(config, 'proactive_capture_interval_s', default=6.0, minimum=refresh_interval_s)
        dropout_hold_s = _config_float(config, 'display_attention_dropout_hold_s', default=max(0.75, min(1.5, refresh_interval_s * 3.0)), minimum=0.2, maximum=3.0)
        gesture_semantic_hold_s = _config_float(config, 'display_attention_gesture_semantic_hold_s', default=max(0.9, min(2.5, refresh_interval_s * 4.0)), minimum=0.2, maximum=4.0)
        pose_semantic_hold_s = _config_float(config, 'display_attention_pose_semantic_hold_s', default=max(2.0, min(8.0, proactive_capture_interval_s + refresh_interval_s)), minimum=0.5, maximum=12.0)
        return cls(dropout_hold_s=dropout_hold_s, gesture_semantic_hold_s=gesture_semantic_hold_s, pose_semantic_hold_s=pose_semantic_hold_s, anchor_match_delta_x=_config_float(config, 'display_attention_anchor_match_delta_x', default=0.2, minimum=0.02, maximum=1.0), anchor_match_delta_y=_config_float(config, 'display_attention_anchor_match_delta_y', default=0.24, minimum=0.02, maximum=1.0), anchor_match_iou=_config_float(config, 'display_attention_anchor_match_iou', default=0.18, minimum=0.01, maximum=0.95), blind_single_person_match_max_age_s=_config_float(config, 'display_attention_blind_single_person_match_max_age_s', default=min(0.35, dropout_hold_s), minimum=0.05, maximum=dropout_hold_s), pose_decay_tau_s=_config_float(config, 'display_attention_pose_decay_tau_s', default=max(0.75, min(pose_semantic_hold_s, pose_semantic_hold_s * 0.65)), minimum=0.1, maximum=max(0.1, pose_semantic_hold_s)), gesture_decay_tau_s=_config_float(config, 'display_attention_gesture_decay_tau_s', default=max(0.35, min(gesture_semantic_hold_s, gesture_semantic_hold_s * 0.55)), minimum=0.1, maximum=max(0.1, gesture_semantic_hold_s)), min_pose_transfer_score=_config_float(config, 'display_attention_min_pose_transfer_score', default=0.14, minimum=0.0, maximum=1.0), min_hand_transfer_score=_config_float(config, 'display_attention_min_hand_transfer_score', default=0.18, minimum=0.0, maximum=1.0), clock_skew_tolerance_s=_config_float(config, 'display_attention_clock_skew_tolerance_s', default=0.35, minimum=0.0, maximum=2.0), prefer_track_id_match=_config_bool(config, 'display_attention_prefer_track_id_match', default=True))

@dataclass(frozen=True, slots=True)
class DisplayAttentionCameraFusionResult:
    observation: SocialVisionObservation
    debug_details: dict[str, object]

@dataclass(frozen=True, slots=True)
class _StoredObservation:
    observed_at: float | None
    received_monotonic_ns: int
    observation: SocialVisionObservation
    source: str
    pose_origin_observed_at: float | None = None
    pose_origin_received_monotonic_ns: int | None = None
    hand_origin_observed_at: float | None = None
    hand_origin_received_monotonic_ns: int | None = None

@dataclass(frozen=True, slots=True)
class _PreparedSemanticSource:
    stored: _StoredObservation
    semantic_age_s: float
    transfer_score: float

class DisplayAttentionCameraFusion:

    def __init__(self, *, config: DisplayAttentionCameraFusionConfig) -> None:
        self.config = config
        self._last_full_observation: _StoredObservation | None = None
        self._last_gesture_observation: _StoredObservation | None = None
        self._last_fused_visible_observation: _StoredObservation | None = None
        self._lock = threading.RLock()

    @classmethod
    def from_config(cls, config: TwinrConfig) -> 'DisplayAttentionCameraFusion':
        return cls(config=DisplayAttentionCameraFusionConfig.from_config(config))

    def remember_full(self, *, observed_at: float, observation: SocialVisionObservation) -> None:
        received_monotonic_ns = time.monotonic_ns()
        candidate = _build_source_record(observed_at=observed_at, observation=observation, source='full_observe', received_monotonic_ns=received_monotonic_ns, capture_pose_origin=True, capture_hand_origin=True)
        with self._lock:
            self._last_full_observation = _select_newer_store(existing=self._last_full_observation, candidate=candidate, clock_skew_tolerance_s=self.config.clock_skew_tolerance_s)

    def remember_gesture(self, *, observed_at: float, observation: SocialVisionObservation) -> None:
        received_monotonic_ns = time.monotonic_ns()
        candidate = _build_source_record(observed_at=observed_at, observation=observation, source='gesture_refresh', received_monotonic_ns=received_monotonic_ns, capture_pose_origin=False, capture_hand_origin=True)
        with self._lock:
            self._last_gesture_observation = _select_newer_store(existing=self._last_gesture_observation, candidate=candidate, clock_skew_tolerance_s=self.config.clock_skew_tolerance_s)

    def fuse_attention(self, *, observed_at: float, observation: SocialVisionObservation) -> DisplayAttentionCameraFusionResult:
        checked_at = _finite_optional_float(observed_at)
        checked_monotonic_ns = time.monotonic_ns()
        with self._lock:
            full_age_s = _observation_age_s(reference_observed_at=checked_at, reference_monotonic_ns=checked_monotonic_ns, source_observed_at=None if self._last_full_observation is None else self._last_full_observation.observed_at, source_received_monotonic_ns=None if self._last_full_observation is None else self._last_full_observation.received_monotonic_ns, clock_skew_tolerance_s=self.config.clock_skew_tolerance_s)
            gesture_age_s = _observation_age_s(reference_observed_at=checked_at, reference_monotonic_ns=checked_monotonic_ns, source_observed_at=None if self._last_gesture_observation is None else self._last_gesture_observation.observed_at, source_received_monotonic_ns=None if self._last_gesture_observation is None else self._last_gesture_observation.received_monotonic_ns, clock_skew_tolerance_s=self.config.clock_skew_tolerance_s)
            fused_visible_age_s = _observation_age_s(reference_observed_at=checked_at, reference_monotonic_ns=checked_monotonic_ns, source_observed_at=None if self._last_fused_visible_observation is None else self._last_fused_visible_observation.observed_at, source_received_monotonic_ns=None if self._last_fused_visible_observation is None else self._last_fused_visible_observation.received_monotonic_ns, clock_skew_tolerance_s=self.config.clock_skew_tolerance_s)
            debug_details: dict[str, object] = {'fast_person_visible': observation.person_visible, 'fast_person_count': observation.person_count, 'full_age_s': _round_seconds(full_age_s), 'gesture_age_s': _round_seconds(gesture_age_s), 'last_fused_visible_age_s': _round_seconds(fused_visible_age_s), 'used_pose_source': None, 'used_pose_age_s': None, 'used_pose_transfer_score': None, 'used_hand_source': None, 'used_hand_age_s': None, 'used_hand_transfer_score': None, 'dropout_hold_source': None, 'dropout_hold_age_s': None}
            fused = observation
            pose_source: _PreparedSemanticSource | None = None
            hand_source: _PreparedSemanticSource | None = None
            if observation.person_visible:
                pose_source = self._best_pose_source(observed_at=checked_at, observed_monotonic_ns=checked_monotonic_ns, current=fused)
                if pose_source is not None:
                    fused = _apply_pose_semantics(fused, pose_source.stored.observation, semantic_age_s=pose_source.semantic_age_s, config=self.config)
                    debug_details['used_pose_source'] = pose_source.stored.source
                    debug_details['used_pose_age_s'] = _round_seconds(pose_source.semantic_age_s)
                    debug_details['used_pose_transfer_score'] = round(pose_source.transfer_score, 3)
                hand_source = self._best_hand_source(observed_at=checked_at, observed_monotonic_ns=checked_monotonic_ns, current=fused)
                if hand_source is not None:
                    fused = _apply_hand_semantics(fused, hand_source.stored.observation, semantic_age_s=hand_source.semantic_age_s, config=self.config)
                    debug_details['used_hand_source'] = hand_source.stored.source
                    debug_details['used_hand_age_s'] = _round_seconds(hand_source.semantic_age_s)
                    debug_details['used_hand_transfer_score'] = round(hand_source.transfer_score, 3)
            else:
                held_visible = self._fresh_visible_source(self._last_fused_visible_observation, observed_at=checked_at, observed_monotonic_ns=checked_monotonic_ns, max_age_s=self.config.dropout_hold_s)
                if held_visible is not None:
                    dropout_hold_age_s = _observation_age_s(reference_observed_at=checked_at, reference_monotonic_ns=checked_monotonic_ns, source_observed_at=held_visible.observed_at, source_received_monotonic_ns=held_visible.received_monotonic_ns, clock_skew_tolerance_s=self.config.clock_skew_tolerance_s)
                    fused = _merge_health_fields(current=observation, held=held_visible.observation)
                    debug_details['dropout_hold_source'] = held_visible.source
                    debug_details['dropout_hold_age_s'] = _round_seconds(dropout_hold_age_s)
            if observation.person_visible:
                self._last_fused_visible_observation = _build_fused_record(observed_at=checked_at, authoritative_current=observation, fused_observation=fused, received_monotonic_ns=checked_monotonic_ns, pose_source=pose_source, hand_source=hand_source)
            debug_details.update({'result_person_visible': fused.person_visible, 'result_looking_toward_device': fused.looking_toward_device, 'result_looking_signal_state': fused.looking_signal_state, 'result_looking_signal_source': fused.looking_signal_source, 'result_hand_or_object_near_camera': fused.hand_or_object_near_camera, 'result_showing_intent_likely': fused.showing_intent_likely, 'result_body_pose': fused.body_pose.value, 'result_motion_state': fused.motion_state.value})
            return DisplayAttentionCameraFusionResult(observation=fused, debug_details=debug_details)

    def has_recent_hand_semantics(self, *, observed_at: float) -> bool:
        """Return whether a fresh local observation still carries hand intent."""

        checked_at = _finite_optional_float(observed_at)
        checked_monotonic_ns = time.monotonic_ns()
        with self._lock:
            candidates = (
                self._fresh_visible_source(
                    self._last_gesture_observation,
                    observed_at=checked_at,
                    observed_monotonic_ns=checked_monotonic_ns,
                    max_age_s=self.config.gesture_semantic_hold_s,
                ),
                self._fresh_visible_source(
                    self._last_fused_visible_observation,
                    observed_at=checked_at,
                    observed_monotonic_ns=checked_monotonic_ns,
                    max_age_s=self.config.gesture_semantic_hold_s,
                ),
                self._fresh_visible_source(
                    self._last_full_observation,
                    observed_at=checked_at,
                    observed_monotonic_ns=checked_monotonic_ns,
                    max_age_s=self.config.gesture_semantic_hold_s,
                ),
            )
            return any(
                candidate is not None and _has_hand_semantics(candidate.observation)
                for candidate in candidates
            )

    def _best_pose_source(self, *, observed_at: float | None, observed_monotonic_ns: int, current: SocialVisionObservation) -> _PreparedSemanticSource | None:
        candidates = (self._compatible_visible_source(self._last_full_observation, observed_at=observed_at, observed_monotonic_ns=observed_monotonic_ns, current=current, semantic_kind='pose', max_age_s=self.config.pose_semantic_hold_s, require_semantics=_has_pose_semantics), self._compatible_visible_source(self._last_fused_visible_observation, observed_at=observed_at, observed_monotonic_ns=observed_monotonic_ns, current=current, semantic_kind='pose', max_age_s=self.config.pose_semantic_hold_s, require_semantics=_has_pose_semantics))
        return _best_semantic_source(candidates)

    def _best_hand_source(self, *, observed_at: float | None, observed_monotonic_ns: int, current: SocialVisionObservation) -> _PreparedSemanticSource | None:
        candidates = (self._compatible_visible_source(self._last_gesture_observation, observed_at=observed_at, observed_monotonic_ns=observed_monotonic_ns, current=current, semantic_kind='hand', max_age_s=self.config.gesture_semantic_hold_s, require_semantics=_has_hand_semantics), self._compatible_visible_source(self._last_full_observation, observed_at=observed_at, observed_monotonic_ns=observed_monotonic_ns, current=current, semantic_kind='hand', max_age_s=self.config.pose_semantic_hold_s, require_semantics=_has_hand_semantics), self._compatible_visible_source(self._last_fused_visible_observation, observed_at=observed_at, observed_monotonic_ns=observed_monotonic_ns, current=current, semantic_kind='hand', max_age_s=self.config.gesture_semantic_hold_s, require_semantics=_has_hand_semantics))
        return _best_semantic_source(candidates)

    def _compatible_visible_source(self, stored: _StoredObservation | None, *, observed_at: float | None, observed_monotonic_ns: int, current: SocialVisionObservation, semantic_kind: str, max_age_s: float, require_semantics) -> _PreparedSemanticSource | None:
        candidate = self._fresh_visible_source(stored, observed_at=observed_at, observed_monotonic_ns=observed_monotonic_ns, max_age_s=max_age_s)
        if candidate is None:
            return None
        if not require_semantics(candidate.observation):
            return None
        semantic_age_s = _semantic_age_s(semantic_kind=semantic_kind, reference_observed_at=observed_at, reference_monotonic_ns=observed_monotonic_ns, stored=candidate, clock_skew_tolerance_s=self.config.clock_skew_tolerance_s)
        if semantic_age_s is None or semantic_age_s > max_age_s:
            return None
        if not _anchors_compatible(current=current, candidate=candidate.observation, config=self.config, candidate_age_s=semantic_age_s):
            return None
        transfer_score = _semantic_transfer_score(semantic_kind=semantic_kind, observation=candidate.observation, semantic_age_s=semantic_age_s, config=self.config)
        minimum_score = self.config.min_pose_transfer_score if semantic_kind == 'pose' else self.config.min_hand_transfer_score
        if transfer_score < minimum_score:
            return None
        return _PreparedSemanticSource(stored=candidate, semantic_age_s=semantic_age_s, transfer_score=transfer_score)

    def _fresh_visible_source(self, stored: _StoredObservation | None, *, observed_at: float | None, observed_monotonic_ns: int, max_age_s: float) -> _StoredObservation | None:
        if stored is None or not stored.observation.person_visible:
            return None
        age_s = _observation_age_s(reference_observed_at=observed_at, reference_monotonic_ns=observed_monotonic_ns, source_observed_at=stored.observed_at, source_received_monotonic_ns=stored.received_monotonic_ns, clock_skew_tolerance_s=self.config.clock_skew_tolerance_s)
        if age_s is None or age_s > max_age_s:
            return None
        return stored

def _build_source_record(*, observed_at: float, observation: SocialVisionObservation, source: str, received_monotonic_ns: int, capture_pose_origin: bool, capture_hand_origin: bool) -> _StoredObservation:
    normalized_observed_at = _finite_optional_float(observed_at)
    pose_origin_observed_at = normalized_observed_at if capture_pose_origin and _has_pose_semantics(observation) else None
    hand_origin_observed_at = normalized_observed_at if capture_hand_origin and _has_hand_semantics(observation) else None
    return _StoredObservation(observed_at=normalized_observed_at, received_monotonic_ns=int(received_monotonic_ns), observation=observation, source=source, pose_origin_observed_at=pose_origin_observed_at, pose_origin_received_monotonic_ns=int(received_monotonic_ns) if pose_origin_observed_at is not None else None, hand_origin_observed_at=hand_origin_observed_at, hand_origin_received_monotonic_ns=int(received_monotonic_ns) if hand_origin_observed_at is not None else None)

def _build_fused_record(*, observed_at: float | None, authoritative_current: SocialVisionObservation, fused_observation: SocialVisionObservation, received_monotonic_ns: int, pose_source: _PreparedSemanticSource | None, hand_source: _PreparedSemanticSource | None) -> _StoredObservation:
    pose_origin_observed_at = observed_at if _has_authoritative_pose_source_data(authoritative_current) else None
    pose_origin_received_monotonic_ns = int(received_monotonic_ns) if pose_origin_observed_at is not None else None
    if pose_origin_observed_at is None and pose_source is not None:
        pose_origin_observed_at = pose_source.stored.pose_origin_observed_at
        pose_origin_received_monotonic_ns = pose_source.stored.pose_origin_received_monotonic_ns
    hand_origin_observed_at = observed_at if _has_hand_semantics(authoritative_current) else None
    hand_origin_received_monotonic_ns = int(received_monotonic_ns) if hand_origin_observed_at is not None else None
    if hand_origin_observed_at is None and hand_source is not None:
        hand_origin_observed_at = hand_source.stored.hand_origin_observed_at
        hand_origin_received_monotonic_ns = hand_source.stored.hand_origin_received_monotonic_ns
    return _StoredObservation(observed_at=observed_at, received_monotonic_ns=int(received_monotonic_ns), observation=fused_observation, source='attention_fused', pose_origin_observed_at=pose_origin_observed_at, pose_origin_received_monotonic_ns=pose_origin_received_monotonic_ns, hand_origin_observed_at=hand_origin_observed_at, hand_origin_received_monotonic_ns=hand_origin_received_monotonic_ns)

def _select_newer_store(*, existing: _StoredObservation | None, candidate: _StoredObservation, clock_skew_tolerance_s: float) -> _StoredObservation:
    if existing is None:
        return candidate
    if candidate.observed_at is not None and existing.observed_at is not None:
        if candidate.observed_at + clock_skew_tolerance_s < existing.observed_at:
            return existing
        if candidate.observed_at > existing.observed_at + clock_skew_tolerance_s:
            return candidate
    if candidate.received_monotonic_ns >= existing.received_monotonic_ns:
        return candidate
    return existing

def _best_semantic_source(candidates: tuple[_PreparedSemanticSource | None, ...]) -> _PreparedSemanticSource | None:
    available = [candidate for candidate in candidates if candidate is not None]
    if not available:
        return None
    return max(available, key=lambda item: (item.transfer_score, -item.semantic_age_s, item.stored.received_monotonic_ns))

def _observation_age_s(*, reference_observed_at: float | None, reference_monotonic_ns: int, source_observed_at: float | None, source_received_monotonic_ns: int | None, clock_skew_tolerance_s: float) -> float | None:
    if source_received_monotonic_ns is None:
        return None
    if reference_observed_at is not None and source_observed_at is not None:
        event_age_s = reference_observed_at - source_observed_at
        if event_age_s >= -clock_skew_tolerance_s:
            return max(0.0, event_age_s)
    receipt_age_ns = int(reference_monotonic_ns) - int(source_received_monotonic_ns)
    if receipt_age_ns < 0:
        return 0.0
    return receipt_age_ns / 1000000000.0

def _semantic_age_s(*, semantic_kind: str, reference_observed_at: float | None, reference_monotonic_ns: int, stored: _StoredObservation, clock_skew_tolerance_s: float) -> float | None:
    if semantic_kind == 'pose':
        return _observation_age_s(reference_observed_at=reference_observed_at, reference_monotonic_ns=reference_monotonic_ns, source_observed_at=stored.pose_origin_observed_at, source_received_monotonic_ns=stored.pose_origin_received_monotonic_ns, clock_skew_tolerance_s=clock_skew_tolerance_s)
    if semantic_kind == 'hand':
        return _observation_age_s(reference_observed_at=reference_observed_at, reference_monotonic_ns=reference_monotonic_ns, source_observed_at=stored.hand_origin_observed_at, source_received_monotonic_ns=stored.hand_origin_received_monotonic_ns, clock_skew_tolerance_s=clock_skew_tolerance_s)
    raise ValueError(f'Unsupported semantic kind: {semantic_kind!r}')

def _round_seconds(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)

def _normalized_confidence(value: object) -> float | None:
    parsed = _finite_optional_float(value)
    if parsed is None:
        return None
    return _clamp(parsed, minimum=0.0, maximum=1.0)

def _decayed_confidence(*, base_confidence: object, semantic_age_s: float, decay_tau_s: float, fallback_confidence: float | None) -> float:
    base = _normalized_confidence(base_confidence)
    if base is None:
        base = _normalized_confidence(fallback_confidence)
    if base is None:
        return 0.0
    tau = max(0.001, float(decay_tau_s))
    return _clamp(base * math.exp(-max(0.0, float(semantic_age_s)) / tau), minimum=0.0, maximum=1.0)

def _semantic_transfer_score(*, semantic_kind: str, observation: SocialVisionObservation, semantic_age_s: float, config: DisplayAttentionCameraFusionConfig) -> float:
    if semantic_kind == 'pose':
        pose_score = _decayed_confidence(base_confidence=observation.pose_confidence, semantic_age_s=semantic_age_s, decay_tau_s=config.pose_decay_tau_s, fallback_confidence=0.72 if observation.body_pose != SocialBodyPose.UNKNOWN else None)
        motion_score = _decayed_confidence(base_confidence=observation.motion_confidence, semantic_age_s=semantic_age_s, decay_tau_s=config.pose_decay_tau_s, fallback_confidence=0.68 if observation.motion_state != SocialMotionState.UNKNOWN else None)
        timeline_score = _decayed_confidence(base_confidence=None, semantic_age_s=semantic_age_s, decay_tau_s=config.pose_decay_tau_s, fallback_confidence=0.42 if _pose_timeline_semantics_present(observation) else None)
        return max(pose_score, motion_score, timeline_score)
    if semantic_kind == 'hand':
        fine_score = _decayed_confidence(base_confidence=observation.fine_hand_gesture_confidence, semantic_age_s=semantic_age_s, decay_tau_s=config.gesture_decay_tau_s, fallback_confidence=0.82 if observation.fine_hand_gesture not in {SocialFineHandGesture.NONE, SocialFineHandGesture.UNKNOWN} else None)
        gesture_score = _decayed_confidence(base_confidence=observation.gesture_confidence, semantic_age_s=semantic_age_s, decay_tau_s=config.gesture_decay_tau_s, fallback_confidence=0.78 if observation.gesture_event not in {SocialGestureEvent.NONE, SocialGestureEvent.UNKNOWN} else None)
        intent_score = _decayed_confidence(base_confidence=None, semantic_age_s=semantic_age_s, decay_tau_s=config.gesture_decay_tau_s, fallback_confidence=0.64 if observation.showing_intent_likely is True else None)
        proximity_score = _decayed_confidence(base_confidence=None, semantic_age_s=semantic_age_s, decay_tau_s=config.gesture_decay_tau_s, fallback_confidence=0.58 if observation.hand_or_object_near_camera else None)
        return max(fine_score, gesture_score, intent_score, proximity_score)
    raise ValueError(f'Unsupported semantic kind: {semantic_kind!r}')

def _pose_timeline_semantics_present(observation: SocialVisionObservation) -> bool:
    return any((observation.person_recently_visible is not None, observation.person_appeared_at is not None, observation.person_disappeared_at is not None))

def _has_pose_semantics(observation: SocialVisionObservation) -> bool:
    return any((observation.body_pose != SocialBodyPose.UNKNOWN, observation.motion_state != SocialMotionState.UNKNOWN, _pose_timeline_semantics_present(observation)))

def _has_authoritative_pose_source_data(observation: SocialVisionObservation) -> bool:
    return any((observation.body_pose != SocialBodyPose.UNKNOWN, observation.motion_state != SocialMotionState.UNKNOWN))

def _has_hand_semantics(observation: SocialVisionObservation) -> bool:
    return any((observation.hand_or_object_near_camera, observation.showing_intent_likely is True, observation.fine_hand_gesture not in {SocialFineHandGesture.NONE, SocialFineHandGesture.UNKNOWN}, observation.gesture_event not in {SocialGestureEvent.NONE, SocialGestureEvent.UNKNOWN}))

def _primary_person_track_id(observation: SocialVisionObservation) -> object | None:
    for attribute_name in ('primary_person_track_id', 'primary_track_id', 'person_track_id', 'track_id'):
        value = getattr(observation, attribute_name, None)
        if value not in {None, '', -1}:
            return value
    return None

def _primary_person_bbox(observation: SocialVisionObservation) -> tuple[float, float, float, float] | None:
    for field_names in (('primary_person_bbox_left', 'primary_person_bbox_top', 'primary_person_bbox_right', 'primary_person_bbox_bottom'), ('primary_person_box_left', 'primary_person_box_top', 'primary_person_box_right', 'primary_person_box_bottom'), ('primary_person_x1', 'primary_person_y1', 'primary_person_x2', 'primary_person_y2'), ('primary_person_bbox_x1', 'primary_person_bbox_y1', 'primary_person_bbox_x2', 'primary_person_bbox_y2')):
        coordinates = [_finite_optional_float(getattr(observation, field_name, None)) for field_name in field_names]
        if any((coordinate is None for coordinate in coordinates)):
            continue
        x1, y1, x2, y2 = coordinates
        if x2 > x1 and y2 > y1:
            return (x1, y1, x2, y2)
    return None

def _bbox_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right
    intersection_x1 = max(left_x1, right_x1)
    intersection_y1 = max(left_y1, right_y1)
    intersection_x2 = min(left_x2, right_x2)
    intersection_y2 = min(left_y2, right_y2)
    intersection_w = max(0.0, intersection_x2 - intersection_x1)
    intersection_h = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_w * intersection_h
    if intersection_area <= 0.0:
        return 0.0
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union_area = left_area + right_area - intersection_area
    if union_area <= 0.0:
        return 0.0
    return intersection_area / union_area

def _anchors_compatible(*, current: SocialVisionObservation, candidate: SocialVisionObservation, config: DisplayAttentionCameraFusionConfig, candidate_age_s: float) -> bool:
    if not current.person_visible or not candidate.person_visible:
        return False
    if config.prefer_track_id_match:
        current_track_id = _primary_person_track_id(current)
        candidate_track_id = _primary_person_track_id(candidate)
        if current_track_id is not None and candidate_track_id is not None:
            return current_track_id == candidate_track_id
    current_bbox = _primary_person_bbox(current)
    candidate_bbox = _primary_person_bbox(candidate)
    if current_bbox is not None and candidate_bbox is not None:
        if _bbox_iou(current_bbox, candidate_bbox) >= config.anchor_match_iou:
            return True
    current_center_x = _finite_optional_float(current.primary_person_center_x)
    candidate_center_x = _finite_optional_float(candidate.primary_person_center_x)
    current_center_y = _finite_optional_float(current.primary_person_center_y)
    candidate_center_y = _finite_optional_float(candidate.primary_person_center_y)
    if current_center_x is not None and candidate_center_x is not None:
        if abs(current_center_x - candidate_center_x) <= config.anchor_match_delta_x:
            if current_center_y is None or candidate_center_y is None:
                return True
            if abs(current_center_y - candidate_center_y) <= config.anchor_match_delta_y:
                return True
    if current.primary_person_zone != SocialPersonZone.UNKNOWN and current.primary_person_zone == candidate.primary_person_zone:
        return True
    if candidate_age_s <= config.blind_single_person_match_max_age_s and current.person_count <= 1 and (candidate.person_count <= 1) and (current.primary_person_zone == SocialPersonZone.UNKNOWN) and (candidate.primary_person_zone == SocialPersonZone.UNKNOWN) and (current_center_x is None) and (candidate_center_x is None):
        return True
    return False

def _apply_pose_semantics(current: SocialVisionObservation, candidate: SocialVisionObservation, *, semantic_age_s: float, config: DisplayAttentionCameraFusionConfig) -> SocialVisionObservation:
    updates: dict[str, object] = {}
    pose_confidence = _decayed_confidence(base_confidence=candidate.pose_confidence, semantic_age_s=semantic_age_s, decay_tau_s=config.pose_decay_tau_s, fallback_confidence=0.72 if candidate.body_pose != SocialBodyPose.UNKNOWN else None)
    if candidate.body_pose != SocialBodyPose.UNKNOWN and current.body_pose == SocialBodyPose.UNKNOWN and (pose_confidence >= config.min_pose_transfer_score):
        updates['body_pose'] = candidate.body_pose
        updates['pose_confidence'] = pose_confidence
    motion_confidence = _decayed_confidence(base_confidence=candidate.motion_confidence, semantic_age_s=semantic_age_s, decay_tau_s=config.pose_decay_tau_s, fallback_confidence=0.68 if candidate.motion_state != SocialMotionState.UNKNOWN else None)
    if candidate.motion_state != SocialMotionState.UNKNOWN and current.motion_state == SocialMotionState.UNKNOWN and (motion_confidence >= config.min_pose_transfer_score):
        updates['motion_state'] = candidate.motion_state
        updates['motion_confidence'] = motion_confidence
    if current.person_recently_visible is None and candidate.person_recently_visible is not None:
        updates['person_recently_visible'] = candidate.person_recently_visible
    if current.person_appeared_at is None and candidate.person_appeared_at is not None:
        updates['person_appeared_at'] = candidate.person_appeared_at
    if current.person_disappeared_at is None and candidate.person_disappeared_at is not None:
        updates['person_disappeared_at'] = candidate.person_disappeared_at
    if not updates:
        return current
    return replace(current, **updates)

def _apply_hand_semantics(current: SocialVisionObservation, candidate: SocialVisionObservation, *, semantic_age_s: float, config: DisplayAttentionCameraFusionConfig) -> SocialVisionObservation:
    updates: dict[str, object] = {}
    proximity_confidence = _decayed_confidence(base_confidence=None, semantic_age_s=semantic_age_s, decay_tau_s=config.gesture_decay_tau_s, fallback_confidence=0.58 if candidate.hand_or_object_near_camera else None)
    if candidate.hand_or_object_near_camera and (not current.hand_or_object_near_camera) and (proximity_confidence >= config.min_hand_transfer_score):
        updates['hand_or_object_near_camera'] = True
    intent_confidence = _decayed_confidence(base_confidence=None, semantic_age_s=semantic_age_s, decay_tau_s=config.gesture_decay_tau_s, fallback_confidence=0.64 if candidate.showing_intent_likely is True else None)
    if candidate.showing_intent_likely is True and current.showing_intent_likely is not True and (max(intent_confidence, proximity_confidence) >= config.min_hand_transfer_score):
        updates['showing_intent_likely'] = True
        if current.showing_intent_started_at is None and candidate.showing_intent_started_at is not None:
            updates['showing_intent_started_at'] = candidate.showing_intent_started_at
    fine_gesture_confidence = _decayed_confidence(base_confidence=candidate.fine_hand_gesture_confidence, semantic_age_s=semantic_age_s, decay_tau_s=config.gesture_decay_tau_s, fallback_confidence=0.82 if candidate.fine_hand_gesture not in {SocialFineHandGesture.NONE, SocialFineHandGesture.UNKNOWN} else None)
    if candidate.fine_hand_gesture not in {SocialFineHandGesture.NONE, SocialFineHandGesture.UNKNOWN} and current.fine_hand_gesture in {SocialFineHandGesture.NONE, SocialFineHandGesture.UNKNOWN} and (fine_gesture_confidence >= config.min_hand_transfer_score):
        updates['fine_hand_gesture'] = candidate.fine_hand_gesture
        updates['fine_hand_gesture_confidence'] = fine_gesture_confidence
    gesture_confidence = _decayed_confidence(base_confidence=candidate.gesture_confidence, semantic_age_s=semantic_age_s, decay_tau_s=config.gesture_decay_tau_s, fallback_confidence=0.78 if candidate.gesture_event not in {SocialGestureEvent.NONE, SocialGestureEvent.UNKNOWN} else None)
    if candidate.gesture_event not in {SocialGestureEvent.NONE, SocialGestureEvent.UNKNOWN} and current.gesture_event in {SocialGestureEvent.NONE, SocialGestureEvent.UNKNOWN} and (gesture_confidence >= config.min_hand_transfer_score):
        updates['gesture_event'] = candidate.gesture_event
        candidate_coarse_arm_gesture = getattr(candidate, 'coarse_arm_gesture', None)
        updates['coarse_arm_gesture'] = candidate_coarse_arm_gesture if candidate_coarse_arm_gesture is not None else candidate.gesture_event
        updates['gesture_confidence'] = gesture_confidence
    if not updates:
        return current
    return replace(current, **updates)

def _merge_health_fields(*, current: SocialVisionObservation, held: SocialVisionObservation) -> SocialVisionObservation:
    return replace(held, camera_online=current.camera_online, camera_ready=current.camera_ready, camera_ai_ready=current.camera_ai_ready, camera_error=current.camera_error, last_camera_frame_at=current.last_camera_frame_at if current.last_camera_frame_at is not None else held.last_camera_frame_at, last_camera_health_change_at=current.last_camera_health_change_at if current.last_camera_health_change_at is not None else held.last_camera_health_change_at)
__all__ = ['DisplayAttentionCameraFusion', 'DisplayAttentionCameraFusionConfig', 'DisplayAttentionCameraFusionResult']
