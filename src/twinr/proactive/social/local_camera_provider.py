# CHANGELOG: 2026-03-29
# BUG-1: Serialize adapter observe/debug capture so attention, gesture, and full-observe lanes cannot mix.
# BUG-2: Canonicalize person evidence from boxes/lists/counts and degrade stale frames instead of emitting false "no person" snapshots.
# SEC-1: Sanitize and bound all outward text fields (errors, labels, signal/debug tokens) to reduce log/prompt-injection risk from tampered model bundles or driver strings.
# SEC-2: Bound visible-person/object fan-out to prevent Pi 4 stalls from bad max_detections/model configs or malformed adapter payloads.
# IMP-1: Add freshness gating and conservative degradation for stale observations/streams.
# IMP-2: Capture and correlate stream debug payloads atomically before marking them authoritative.
# IMP-3: Add lightweight temporal stability tracking for detected objects instead of always reporting stable=False.

"""Map the local IMX500 hardware adapter onto Twinr's social vision contract."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import threading
import time
from typing import Iterable, Mapping

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.ai_camera import (
    AICameraBodyPose,
    AICameraGestureEvent,
    AICameraFineHandGesture,
    AICameraMotionState,
    AICameraObservation,
    AICameraZone,
    LocalAICameraAdapter,
)

from .engine import (
    SocialBodyPose,
    SocialDetectedObject,
    SocialFineHandGesture,
    SocialGestureEvent,
    SocialMotionState,
    SocialPersonZone,
    SocialSpatialBox,
    SocialVisiblePerson,
    SocialVisionObservation,
)
from .observers import ProactiveVisionSnapshot
from .perception_stream import (
    PerceptionAttentionStreamObservation,
    PerceptionGestureStreamObservation,
    PerceptionStreamObservation,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LocalAICameraProviderConfig:
    """Store provider-level wiring settings for the local camera path."""

    source_device: str = "imx500"
    input_format: str | None = None
    max_visible_persons: int = 4
    max_objects: int = 12
    max_text_length: int = 160
    frame_stale_after_seconds: float = 1.5
    stream_debug_max_skew_seconds: float = 0.35
    stable_object_hits: int = 2
    stable_object_window_seconds: float = 1.2

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "LocalAICameraProviderConfig":
        return cls(
            source_device=_sanitize_text(
                getattr(config, "proactive_local_camera_source_device", "imx500"),
                max_length=64,
            )
            or "imx500",
            input_format=_sanitize_text(
                getattr(config, "proactive_local_camera_input_format", None),
                max_length=64,
            ),
            max_visible_persons=_coerce_positive_int(
                getattr(config, "proactive_local_camera_max_visible_persons", 4),
                default=4,
            ),
            max_objects=_coerce_positive_int(
                getattr(config, "proactive_local_camera_max_objects", 12),
                default=12,
            ),
            max_text_length=_coerce_positive_int(
                getattr(config, "proactive_local_camera_max_text_length", 160),
                default=160,
            ),
            frame_stale_after_seconds=_coerce_positive_float(
                getattr(config, "proactive_local_camera_frame_stale_after_seconds", 1.5),
                default=1.5,
            ),
            stream_debug_max_skew_seconds=_coerce_positive_float(
                getattr(config, "proactive_local_camera_stream_debug_max_skew_seconds", 0.35),
                default=0.35,
            ),
            stable_object_hits=_coerce_positive_int(
                getattr(config, "proactive_local_camera_stable_object_hits", 2),
                default=2,
            ),
            stable_object_window_seconds=_coerce_positive_float(
                getattr(config, "proactive_local_camera_stable_object_window_seconds", 1.2),
                default=1.2,
            ),
        )


@dataclass(slots=True)
class _ObjectStabilityState:
    hits: int
    last_seen_at: float


class LocalAICameraObservationProvider:
    """Return bounded local-first camera observations without cloud dependency."""

    supports_attention_refresh = True
    supports_gesture_refresh = True
    supports_perception_refresh = True

    def __init__(
        self,
        *,
        adapter: LocalAICameraAdapter,
        config: LocalAICameraProviderConfig | None = None,
    ) -> None:
        self.adapter = adapter
        self.config = config or LocalAICameraProviderConfig()
        self._adapter_lock = threading.RLock()
        self._object_stability: dict[tuple[str, str], _ObjectStabilityState] = {}

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "LocalAICameraObservationProvider":
        return cls(
            adapter=LocalAICameraAdapter.from_config(config),
            config=LocalAICameraProviderConfig.from_config(config),
        )

    def observe(self) -> ProactiveVisionSnapshot:
        observation, attention_details, gesture_details = self._capture_with_debug(
            observe_fn=self.adapter.observe,
            camera_error="local_ai_camera_provider_failed",
            capture_attention_debug=True,
            capture_gesture_debug=True,
        )
        # BREAKING: full observe() now surfaces a correlated perception_stream whenever
        # enough bounded signal exists to build it safely.
        social = self._to_social_observation(
            observation,
            perception_stream=self._combined_perception_stream(
                observation,
                attention_details=attention_details,
                gesture_details=gesture_details,
            ),
        )
        return self._snapshot_from_observation(observation, social)

    def observe_attention(self) -> ProactiveVisionSnapshot:
        used_stream_lane = False
        observe_attention_stream = getattr(self.adapter, "observe_attention_stream", None)

        if callable(observe_attention_stream):
            observation, attention_details, _ = self._capture_with_debug(
                observe_fn=observe_attention_stream,
                camera_error="local_ai_camera_attention_provider_failed",
                capture_attention_debug=True,
            )
            used_stream_lane = True
        else:
            observation, attention_details, _ = self._capture_with_debug(
                observe_fn=self.adapter.observe_attention,
                camera_error="local_ai_camera_attention_provider_failed",
                capture_attention_debug=True,
            )

        social = self._to_social_observation(
            observation,
            perception_stream=self._attention_perception_stream(
                observation,
                details=attention_details,
                source="attention_stream" if used_stream_lane else "attention_snapshot",
                authoritative_hint=used_stream_lane,
            ),
        )
        return self._snapshot_from_observation(observation, social)

    def observe_gesture(self) -> ProactiveVisionSnapshot:
        used_stream_lane = False
        observe_gesture_stream = getattr(self.adapter, "observe_gesture_stream", None)

        if callable(observe_gesture_stream):
            observation, _, gesture_details = self._capture_with_debug(
                observe_fn=observe_gesture_stream,
                camera_error="local_ai_camera_gesture_provider_failed",
                capture_gesture_debug=True,
            )
            used_stream_lane = True
        else:
            observation, _, gesture_details = self._capture_with_debug(
                observe_fn=lambda: self.adapter.observe_gesture(gesture_fast_path=True),
                camera_error="local_ai_camera_gesture_provider_failed",
                capture_gesture_debug=True,
            )

        social = self._to_social_observation(
            observation,
            perception_stream=self._gesture_perception_stream(
                observation,
                details=gesture_details,
                source="gesture_stream" if used_stream_lane else "gesture_snapshot",
                authoritative_hint=used_stream_lane,
            ),
        )
        return self._snapshot_from_observation(observation, social)

    def observe_perception_stream(self) -> ProactiveVisionSnapshot:
        observe_perception_stream = getattr(self.adapter, "observe_perception_stream", None)

        if callable(observe_perception_stream):
            observation, attention_details, gesture_details = self._capture_with_debug(
                observe_fn=observe_perception_stream,
                camera_error="local_ai_camera_perception_provider_failed",
                capture_attention_debug=True,
                capture_gesture_debug=True,
            )
            perception_stream = self._combined_perception_stream(
                observation,
                attention_details=attention_details,
                gesture_details=gesture_details,
                attention_authoritative_hint=True,
                gesture_authoritative_hint=True,
            )
        else:
            observation, attention_details, gesture_details = self._capture_with_debug(
                observe_fn=self.adapter.observe,
                camera_error="local_ai_camera_perception_provider_failed",
                capture_attention_debug=True,
                capture_gesture_debug=True,
            )
            perception_stream = self._combined_perception_stream(
                observation,
                attention_details=attention_details,
                gesture_details=gesture_details,
            )

        social = self._to_social_observation(
            observation,
            perception_stream=perception_stream,
        )
        return self._snapshot_from_observation(observation, social)

    def gesture_debug_details(self) -> dict[str, object] | None:
        with self._adapter_lock:
            details = self._copy_debug_details_unlocked("get_last_gesture_debug_details")
        return _sanitize_debug_details(details, max_text_length=self.config.max_text_length)

    def attention_debug_details(self) -> dict[str, object] | None:
        with self._adapter_lock:
            details = self._copy_debug_details_unlocked("get_last_attention_debug_details")
        return _sanitize_debug_details(details, max_text_length=self.config.max_text_length)

    def _capture_with_debug(
        self,
        *,
        observe_fn,
        camera_error: str,
        capture_attention_debug: bool = False,
        capture_gesture_debug: bool = False,
    ) -> tuple[AICameraObservation, dict[str, object] | None, dict[str, object] | None]:
        try:
            with self._adapter_lock:
                observation = observe_fn()
                if not isinstance(observation, AICameraObservation):
                    raise TypeError(f"adapter returned unexpected observation type: {type(observation)!r}")
                attention_details = (
                    self._copy_debug_details_unlocked("get_last_attention_debug_details")
                    if capture_attention_debug
                    else None
                )
                gesture_details = (
                    self._copy_debug_details_unlocked("get_last_gesture_debug_details")
                    if capture_gesture_debug
                    else None
                )
        except Exception:  # pragma: no cover
            logger.exception(
                "Local AI camera provider failed unexpectedly; returning a conservative health snapshot."
            )
            observation = AICameraObservation(
                observed_at=0.0,
                camera_online=False,
                camera_ready=False,
                camera_ai_ready=False,
                camera_error=camera_error,
            )
            attention_details = None
            gesture_details = None
        return observation, attention_details, gesture_details

    def _copy_debug_details_unlocked(self, getter_name: str) -> dict[str, object] | None:
        getter = getattr(self.adapter, getter_name, None)
        if not callable(getter):
            return None
        details = getter()
        if details is None:
            return None
        if isinstance(details, Mapping):
            return dict(details.items())
        try:
            return dict(details)
        except Exception:
            return None

    def _snapshot_from_observation(
        self,
        observation: AICameraObservation,
        social: SocialVisionObservation,
    ) -> ProactiveVisionSnapshot:
        captured_at = self._effective_captured_at(observation)
        return ProactiveVisionSnapshot(
            observation=social,
            response_text=self._response_text(observation, social),
            captured_at=captured_at if captured_at is not None else 0.0,
            image=None,
            source_device=self.config.source_device,
            input_format=self.config.input_format,
            response_id=None,
            request_id=None,
            model=_sanitize_text(getattr(observation, "model", None), max_length=128),
        )

    def _to_social_observation(
        self,
        observation: AICameraObservation,
        *,
        perception_stream: PerceptionStreamObservation | None = None,
    ) -> SocialVisionObservation:
        stale = self._is_stale(observation)
        usable = self._is_camera_usable(observation) and not stale

        visible_persons: list[SocialVisiblePerson] = []
        for item in _take_iterable(
            getattr(observation, "visible_persons", None),
            self.config.max_visible_persons,
        ):
            visible_persons.append(
                SocialVisiblePerson(
                    box=self._coerce_social_box(getattr(item, "box", None)),
                    zone=_map_zone(getattr(item, "zone", AICameraZone.UNKNOWN)),
                    confidence=_coerce_ratio(getattr(item, "confidence", None)),
                )
            )

        primary_person_box = self._coerce_social_box(getattr(observation, "primary_person_box", None))
        if primary_person_box is None:
            for item in visible_persons:
                if item.box is not None:
                    primary_person_box = item.box
                    break

        primary_person_zone = _map_zone(getattr(observation, "primary_person_zone", AICameraZone.UNKNOWN))
        if primary_person_zone is SocialPersonZone.UNKNOWN and visible_persons:
            primary_person_zone = visible_persons[0].zone

        primary_person_center_x = _coerce_optional_float(getattr(observation, "primary_person_center_x", None))
        primary_person_center_y = _coerce_optional_float(getattr(observation, "primary_person_center_y", None))
        if primary_person_box is not None:
            if primary_person_center_x is None:
                primary_person_center_x = (primary_person_box.left + primary_person_box.right) / 2.0
            if primary_person_center_y is None:
                primary_person_center_y = (primary_person_box.top + primary_person_box.bottom) / 2.0

        raw_person_count = _coerce_optional_non_negative_int(getattr(observation, "person_count", None))
        inferred_person_count = len(visible_persons)
        if primary_person_box is not None or primary_person_center_x is not None or primary_person_center_y is not None:
            inferred_person_count = max(inferred_person_count, 1)
        person_count = max(raw_person_count or 0, inferred_person_count)

        if not usable:
            # BREAKING: stale or unhealthy frames are now downgraded to a conservative empty
            # observation instead of leaking last-known detections downstream.
            self._prune_object_stability(time.time(), keep=())
            person_visible = False
            person_count = 0
            primary_person_zone = SocialPersonZone.UNKNOWN
            primary_person_box = None
            visible_persons = []
            primary_person_center_x = None
            primary_person_center_y = None
            looking_toward_device = False
            looking_signal_state = None
            looking_signal_source = None
            person_near_device = False
            engaged_with_device = False
            visual_attention_score = 0.0
            body_pose = SocialBodyPose.UNKNOWN
            pose_confidence = 0.0
            motion_state = SocialMotionState.UNKNOWN
            motion_confidence = 0.0
            hand_or_object_near_camera = False
            showing_intent_likely = False
            coarse_arm_gesture = SocialGestureEvent.NONE
            gesture_event = SocialGestureEvent.NONE
            gesture_confidence = 0.0
            fine_hand_gesture = SocialFineHandGesture.NONE
            fine_hand_gesture_confidence = 0.0
            objects = ()
        else:
            person_visible = person_count > 0
            looking_toward_device = _coerce_bool(getattr(observation, "looking_toward_device", False))
            looking_signal_state = _sanitize_text(
                getattr(observation, "looking_signal_state", None),
                max_length=self.config.max_text_length,
            )
            looking_signal_source = _sanitize_text(
                getattr(observation, "looking_signal_source", None),
                max_length=self.config.max_text_length,
            )
            person_near_device = _coerce_bool(getattr(observation, "person_near_device", False))
            engaged_with_device = _coerce_bool(getattr(observation, "engaged_with_device", False))
            visual_attention_score = _coerce_ratio(getattr(observation, "visual_attention_score", None))
            body_pose = _map_body_pose(getattr(observation, "body_pose", AICameraBodyPose.UNKNOWN))
            pose_confidence = _coerce_ratio(getattr(observation, "pose_confidence", None))
            motion_state = _map_motion_state(getattr(observation, "motion_state", AICameraMotionState.UNKNOWN))
            motion_confidence = _coerce_ratio(getattr(observation, "motion_confidence", None))
            hand_or_object_near_camera = _coerce_bool(getattr(observation, "hand_or_object_near_camera", False))
            showing_intent_likely = _coerce_bool(getattr(observation, "showing_intent_likely", False))
            gesture_event = _map_gesture(getattr(observation, "gesture_event", AICameraGestureEvent.NONE))
            coarse_arm_gesture = gesture_event
            gesture_confidence = _coerce_ratio(getattr(observation, "gesture_confidence", None))
            fine_hand_gesture = _map_fine_hand_gesture(
                getattr(observation, "fine_hand_gesture", AICameraFineHandGesture.NONE)
            )
            fine_hand_gesture_confidence = _coerce_ratio(
                getattr(observation, "fine_hand_gesture_confidence", None)
            )
            objects = self._map_objects(observation)

        return SocialVisionObservation(
            person_visible=person_visible,
            person_count=person_count,
            primary_person_zone=primary_person_zone,
            primary_person_box=primary_person_box,
            visible_persons=tuple(visible_persons),
            primary_person_center_x=primary_person_center_x,
            primary_person_center_y=primary_person_center_y,
            looking_toward_device=looking_toward_device,
            looking_signal_state=looking_signal_state,
            looking_signal_source=looking_signal_source,
            person_near_device=person_near_device,
            engaged_with_device=engaged_with_device,
            visual_attention_score=visual_attention_score,
            body_pose=body_pose,
            pose_confidence=pose_confidence,
            motion_state=motion_state,
            motion_confidence=motion_confidence,
            smiling=False,
            hand_or_object_near_camera=hand_or_object_near_camera,
            showing_intent_likely=showing_intent_likely,
            coarse_arm_gesture=coarse_arm_gesture,
            gesture_event=gesture_event,
            gesture_confidence=gesture_confidence,
            fine_hand_gesture=fine_hand_gesture,
            fine_hand_gesture_confidence=fine_hand_gesture_confidence,
            objects=objects,
            camera_online=_coerce_bool(getattr(observation, "camera_online", False)),
            camera_ready=_coerce_bool(getattr(observation, "camera_ready", False)) and not stale,
            camera_ai_ready=_coerce_bool(getattr(observation, "camera_ai_ready", False)) and not stale,
            camera_error=self._compose_camera_error(
                _sanitize_text(getattr(observation, "camera_error", None), max_length=self.config.max_text_length),
                stale=stale,
            ),
            last_camera_frame_at=self._effective_captured_at(observation),
            last_camera_health_change_at=_coerce_optional_timestamp(
                getattr(observation, "last_camera_health_change_at", None)
            ),
            perception_stream=perception_stream,
        )

    def _map_objects(self, observation: AICameraObservation) -> tuple[SocialDetectedObject, ...]:
        captured_at = self._effective_captured_at(observation) or time.time()
        mapped_objects: list[SocialDetectedObject] = []
        seen_keys: list[tuple[str, str]] = []

        for item in _take_iterable(getattr(observation, "objects", None), self.config.max_objects):
            label = _sanitize_text(getattr(item, "label", None), max_length=96)
            zone = _map_zone(getattr(item, "zone", AICameraZone.UNKNOWN))
            box = self._coerce_social_box(getattr(item, "box", None))
            confidence = _coerce_ratio(getattr(item, "confidence", None))
            stability_key = (label or "unknown", zone.value)
            seen_keys.append(stability_key)
            stable = self._mark_object_seen(stability_key, captured_at)
            mapped_objects.append(
                SocialDetectedObject(
                    label=label or "unknown",
                    confidence=confidence,
                    zone=zone,
                    stable=stable,
                    box=box,
                )
            )

        self._prune_object_stability(captured_at, keep=seen_keys)
        return tuple(mapped_objects)

    def _mark_object_seen(self, key: tuple[str, str], captured_at: float) -> bool:
        state = self._object_stability.get(key)
        if state is None or (captured_at - state.last_seen_at) > self.config.stable_object_window_seconds:
            state = _ObjectStabilityState(hits=1, last_seen_at=captured_at)
        else:
            state.hits += 1
            state.last_seen_at = captured_at
        self._object_stability[key] = state
        return state.hits >= self.config.stable_object_hits

    def _prune_object_stability(self, captured_at: float, *, keep: Iterable[tuple[str, str]]) -> None:
        keep_set = set(keep)
        cutoff = captured_at - max(self.config.stable_object_window_seconds * 2.0, 0.5)

        for key, state in list(self._object_stability.items()):
            if state.last_seen_at < cutoff and key not in keep_set:
                self._object_stability.pop(key, None)

        max_states = max(self.config.max_objects * 4, 16)
        if len(self._object_stability) <= max_states:
            return

        for key, _ in sorted(
            self._object_stability.items(),
            key=lambda item: item[1].last_seen_at,
        )[:-max_states]:
            self._object_stability.pop(key, None)

    def _combined_perception_stream(
        self,
        observation: AICameraObservation,
        *,
        attention_details: dict[str, object] | None = None,
        gesture_details: dict[str, object] | None = None,
        attention_authoritative_hint: bool = False,
        gesture_authoritative_hint: bool = False,
    ) -> PerceptionStreamObservation | None:
        attention = self._build_attention_stream_observation(
            observation,
            details=attention_details,
            authoritative_hint=attention_authoritative_hint,
        )
        gesture = self._build_gesture_stream_observation(
            observation,
            details=gesture_details,
            authoritative_hint=gesture_authoritative_hint,
        )
        if attention is None and gesture is None:
            return None
        return PerceptionStreamObservation(
            source="local_camera",
            captured_at=self._effective_captured_at(observation) or 0.0,
            attention=attention,
            gesture=gesture,
        )

    def _attention_perception_stream(
        self,
        observation: AICameraObservation,
        *,
        details: dict[str, object] | None,
        source: str,
        authoritative_hint: bool = False,
    ) -> PerceptionStreamObservation | None:
        attention = self._build_attention_stream_observation(
            observation,
            details=details,
            authoritative_hint=authoritative_hint,
        )
        if attention is None:
            return None
        return PerceptionStreamObservation(
            source=source,
            captured_at=self._effective_captured_at(observation) or 0.0,
            attention=attention,
        )

    def _gesture_perception_stream(
        self,
        observation: AICameraObservation,
        *,
        details: dict[str, object] | None,
        source: str,
        authoritative_hint: bool = False,
    ) -> PerceptionStreamObservation | None:
        gesture = self._build_gesture_stream_observation(
            observation,
            details=details,
            authoritative_hint=authoritative_hint,
        )
        if gesture is None:
            return None
        return PerceptionStreamObservation(
            source=source,
            captured_at=self._effective_captured_at(observation) or 0.0,
            gesture=gesture,
        )

    def _build_attention_stream_observation(
        self,
        observation: AICameraObservation,
        *,
        details: dict[str, object] | None = None,
        authoritative_hint: bool = False,
    ) -> PerceptionAttentionStreamObservation | None:
        if not self._is_camera_usable(observation) or self._is_stale(observation):
            return None

        details = details or {}
        if not self._attention_details_match_observation(details, observation, authoritative_hint=authoritative_hint):
            details = {}

        authoritative = authoritative_hint or str(details.get("stream_mode") or "").strip().lower() == "attention_stream"
        if not authoritative and not details:
            return None

        return PerceptionAttentionStreamObservation(
            authoritative=authoritative,
            stable_looking_toward_device=_coerce_bool(getattr(observation, "looking_toward_device", False)),
            stable_visual_attention_score=_coerce_ratio(getattr(observation, "visual_attention_score", None)),
            stable_signal_state=_sanitize_text(
                getattr(observation, "looking_signal_state", None),
                max_length=self.config.max_text_length,
            ),
            stable_signal_source=_sanitize_text(
                getattr(observation, "looking_signal_source", None),
                max_length=self.config.max_text_length,
            ),
            instant_looking_toward_device=_coerce_optional_bool(
                details.get("attention_instant_looking_toward_device")
            ),
            instant_visual_attention_score=_coerce_optional_ratio(
                details.get("attention_instant_visual_attention_score")
            ),
            instant_signal_state=_sanitize_text(
                details.get("attention_instant_looking_signal_state"),
                max_length=self.config.max_text_length,
            ),
            instant_signal_source=_sanitize_text(
                details.get("attention_instant_looking_signal_source"),
                max_length=self.config.max_text_length,
            ),
            candidate_signal_state=_sanitize_text(
                details.get("attention_stream_candidate_state"),
                max_length=self.config.max_text_length,
            ),
            candidate_signal_source=_sanitize_text(
                details.get("attention_stream_candidate_source"),
                max_length=self.config.max_text_length,
            ),
            changed=_coerce_bool(details.get("attention_stream_changed")),
        )

    def _build_gesture_stream_observation(
        self,
        observation: AICameraObservation,
        *,
        details: dict[str, object] | None = None,
        authoritative_hint: bool = False,
    ) -> PerceptionGestureStreamObservation | None:
        if not self._is_camera_usable(observation) or self._is_stale(observation):
            return None

        details = details or {}
        if not self._gesture_details_match_observation(details, observation):
            details = {}

        authoritative = _coerce_bool(getattr(observation, "gesture_temporal_authoritative", False)) or authoritative_hint
        if not authoritative:
            return None

        stream_changed = bool(
            _coerce_bool(getattr(observation, "gesture_activation_rising", False))
            or _coerce_bool(details.get("authoritative_gesture_rising"))
            or _coerce_bool(details.get("gesture_stream_output_changed"))
            or _coerce_bool(details.get("temporal_output_changed"))
        )
        return PerceptionGestureStreamObservation(
            authoritative=True,
            activation_key=_sanitize_text(
                getattr(observation, "gesture_activation_key", None),
                max_length=self.config.max_text_length,
            ),
            activation_token=_coerce_optional_non_negative_int(
                getattr(observation, "gesture_activation_token", None)
            ),
            activation_started_at=_coerce_optional_timestamp(
                getattr(observation, "gesture_activation_started_at", None)
            ),
            activation_changed_at=_coerce_optional_timestamp(
                getattr(observation, "gesture_activation_changed_at", None)
            ),
            activation_source=_sanitize_text(
                getattr(observation, "gesture_activation_source", None),
                max_length=self.config.max_text_length,
            ),
            activation_rising=_coerce_bool(getattr(observation, "gesture_activation_rising", False)),
            stable_gesture_event=_map_gesture(getattr(observation, "gesture_event", AICameraGestureEvent.NONE)),
            stable_gesture_confidence=_coerce_ratio(getattr(observation, "gesture_confidence", None)),
            stable_fine_hand_gesture=_map_fine_hand_gesture(
                getattr(observation, "fine_hand_gesture", AICameraFineHandGesture.NONE)
            ),
            stable_fine_hand_gesture_confidence=_coerce_ratio(
                getattr(observation, "fine_hand_gesture_confidence", None)
            ),
            instant_gesture_event=_coerce_gesture_value(details.get("live_gesture_event")),
            instant_gesture_confidence=_coerce_optional_ratio(details.get("live_gesture_confidence")),
            instant_fine_hand_gesture=_coerce_fine_hand_gesture_value(details.get("live_fine_hand_gesture")),
            instant_fine_hand_gesture_confidence=_coerce_optional_ratio(
                details.get("live_fine_hand_gesture_confidence")
            ),
            hand_or_object_near_camera=_coerce_bool(
                getattr(observation, "hand_or_object_near_camera", False)
            ),
            temporal_reason=_sanitize_text(
                details.get("gesture_stream_temporal_reason") or details.get("temporal_reason"),
                max_length=self.config.max_text_length,
            ),
            resolved_source=_sanitize_text(
                details.get("gesture_stream_resolved_source") or details.get("resolved_source"),
                max_length=self.config.max_text_length,
            ),
            changed=stream_changed,
        )

    def _attention_details_match_observation(
        self,
        details: dict[str, object],
        observation: AICameraObservation,
        *,
        authoritative_hint: bool,
    ) -> bool:
        if not details:
            return authoritative_hint
        stream_mode = _sanitize_text(details.get("stream_mode"), max_length=64)
        if not authoritative_hint and stream_mode not in {None, "attention_stream"}:
            return False
        return self._detail_timestamp_matches_observation(details, observation)

    def _gesture_details_match_observation(
        self,
        details: dict[str, object],
        observation: AICameraObservation,
    ) -> bool:
        if not details:
            return True
        detail_token = _coerce_optional_non_negative_int(
            details.get("gesture_activation_token") or details.get("activation_token")
        )
        observation_token = _coerce_optional_non_negative_int(
            getattr(observation, "gesture_activation_token", None)
        )
        if detail_token is not None and observation_token is not None and detail_token != observation_token:
            return False
        return self._detail_timestamp_matches_observation(details, observation)

    def _detail_timestamp_matches_observation(
        self,
        details: dict[str, object],
        observation: AICameraObservation,
    ) -> bool:
        detail_timestamp = _coerce_optional_timestamp(
            details.get("captured_at")
            or details.get("frame_at")
            or details.get("last_camera_frame_at")
            or details.get("observed_at")
            or details.get("attention_captured_at")
            or details.get("gesture_captured_at")
        )
        observation_timestamp = self._effective_captured_at(observation)
        if detail_timestamp is None or observation_timestamp is None:
            return True
        return abs(detail_timestamp - observation_timestamp) <= self.config.stream_debug_max_skew_seconds

    def _is_camera_usable(self, observation: AICameraObservation) -> bool:
        return (
            _coerce_bool(getattr(observation, "camera_online", False))
            and _coerce_bool(getattr(observation, "camera_ready", False))
            and _coerce_bool(getattr(observation, "camera_ai_ready", False))
        )

    def _effective_captured_at(self, observation: AICameraObservation) -> float | None:
        frame_at = _coerce_optional_positive_timestamp(getattr(observation, "last_camera_frame_at", None))
        observed_at = _coerce_optional_positive_timestamp(getattr(observation, "observed_at", None))
        return frame_at if frame_at is not None else observed_at

    def _frame_age_seconds(self, observation: AICameraObservation) -> float | None:
        observed_at = _coerce_optional_positive_timestamp(getattr(observation, "observed_at", None))
        frame_at = _coerce_optional_positive_timestamp(getattr(observation, "last_camera_frame_at", None))

        if observed_at is not None and frame_at is not None and observed_at >= frame_at:
            return observed_at - frame_at

        captured_at = frame_at if frame_at is not None else observed_at
        if captured_at is not None and captured_at > 946684800.0:
            age = time.time() - captured_at
            if age >= 0.0 and math.isfinite(age):
                return age
        return None

    def _is_stale(self, observation: AICameraObservation) -> bool:
        age = self._frame_age_seconds(observation)
        return age is not None and age > self.config.frame_stale_after_seconds

    def _response_text(self, observation: AICameraObservation, social: SocialVisionObservation) -> str:
        frame_age = self._frame_age_seconds(observation)
        frame_age_ms = "na" if frame_age is None else str(int(round(frame_age * 1000.0)))
        return (
            "provider=local_ai_camera "
            f"ready={'yes' if social.camera_ready else 'no'} "
            f"ai_ready={'yes' if social.camera_ai_ready else 'no'} "
            f"stale={'yes' if self._is_stale(observation) else 'no'} "
            f"frame_age_ms={frame_age_ms} "
            f"person_count={social.person_count} "
            f"body_pose={social.body_pose.value} "
            f"motion={social.motion_state.value} "
            f"coarse_arm_gesture={social.gesture_event.value} "
            f"fine_hand_gesture={social.fine_hand_gesture.value} "
            f"error={social.camera_error or 'none'}"
        )

    def _compose_camera_error(self, raw_error: str | None, *, stale: bool) -> str | None:
        reasons: list[str] = []
        if raw_error:
            reasons.append(raw_error)
        if stale:
            reasons.append("stale_camera_frame")
        if not reasons:
            return None
        return " | ".join(reasons[:2])

    def _coerce_social_box(self, value: object) -> SocialSpatialBox | None:
        if value is None:
            return None
        top = _coerce_optional_float(getattr(value, "top", None))
        left = _coerce_optional_float(getattr(value, "left", None))
        bottom = _coerce_optional_float(getattr(value, "bottom", None))
        right = _coerce_optional_float(getattr(value, "right", None))
        if top is None or left is None or bottom is None or right is None:
            return None
        top, bottom = sorted((top, bottom))
        left, right = sorted((left, right))
        return SocialSpatialBox(top=top, left=left, bottom=bottom, right=right)


def _map_zone(value: AICameraZone) -> SocialPersonZone:
    mapping = {
        AICameraZone.LEFT: SocialPersonZone.LEFT,
        AICameraZone.CENTER: SocialPersonZone.CENTER,
        AICameraZone.RIGHT: SocialPersonZone.RIGHT,
        AICameraZone.UNKNOWN: SocialPersonZone.UNKNOWN,
    }
    return mapping.get(value, SocialPersonZone.UNKNOWN)


def _map_body_pose(value: AICameraBodyPose) -> SocialBodyPose:
    mapping = {
        AICameraBodyPose.UPRIGHT: SocialBodyPose.UPRIGHT,
        AICameraBodyPose.SEATED: SocialBodyPose.SEATED,
        AICameraBodyPose.SLUMPED: SocialBodyPose.SLUMPED,
        AICameraBodyPose.LYING_LOW: SocialBodyPose.LYING_LOW,
        AICameraBodyPose.FLOOR: SocialBodyPose.FLOOR,
        AICameraBodyPose.UNKNOWN: SocialBodyPose.UNKNOWN,
    }
    return mapping.get(value, SocialBodyPose.UNKNOWN)


def _map_motion_state(value: AICameraMotionState) -> SocialMotionState:
    mapping = {
        AICameraMotionState.STILL: SocialMotionState.STILL,
        AICameraMotionState.WALKING: SocialMotionState.WALKING,
        AICameraMotionState.APPROACHING: SocialMotionState.APPROACHING,
        AICameraMotionState.LEAVING: SocialMotionState.LEAVING,
        AICameraMotionState.UNKNOWN: SocialMotionState.UNKNOWN,
    }
    return mapping.get(value, SocialMotionState.UNKNOWN)


def _map_gesture(value: AICameraGestureEvent) -> SocialGestureEvent:
    mapping = {
        AICameraGestureEvent.NONE: SocialGestureEvent.NONE,
        AICameraGestureEvent.WAVE: SocialGestureEvent.WAVE,
        AICameraGestureEvent.STOP: SocialGestureEvent.STOP,
        AICameraGestureEvent.DISMISS: SocialGestureEvent.DISMISS,
        AICameraGestureEvent.CONFIRM: SocialGestureEvent.CONFIRM,
        AICameraGestureEvent.ARMS_CROSSED: SocialGestureEvent.ARMS_CROSSED,
        AICameraGestureEvent.TWO_HAND_DISMISS: SocialGestureEvent.TWO_HAND_DISMISS,
        AICameraGestureEvent.TIMEOUT_T: SocialGestureEvent.TIMEOUT_T,
        AICameraGestureEvent.UNKNOWN: SocialGestureEvent.UNKNOWN,
    }
    return mapping.get(value, SocialGestureEvent.UNKNOWN)


def _map_fine_hand_gesture(value: AICameraFineHandGesture) -> SocialFineHandGesture:
    mapping = {
        AICameraFineHandGesture.NONE: SocialFineHandGesture.NONE,
        AICameraFineHandGesture.THUMBS_UP: SocialFineHandGesture.THUMBS_UP,
        AICameraFineHandGesture.THUMBS_DOWN: SocialFineHandGesture.THUMBS_DOWN,
        AICameraFineHandGesture.POINTING: SocialFineHandGesture.POINTING,
        AICameraFineHandGesture.PEACE_SIGN: SocialFineHandGesture.PEACE_SIGN,
        AICameraFineHandGesture.OPEN_PALM: SocialFineHandGesture.OPEN_PALM,
        AICameraFineHandGesture.OK_SIGN: SocialFineHandGesture.OK_SIGN,
        AICameraFineHandGesture.MIDDLE_FINGER: SocialFineHandGesture.MIDDLE_FINGER,
        AICameraFineHandGesture.UNKNOWN: SocialFineHandGesture.UNKNOWN,
    }
    return mapping.get(value, SocialFineHandGesture.UNKNOWN)


def _sanitize_text(
    value: object,
    *,
    max_length: int = 160,
) -> str | None:
    if value is None:
        return None
    text = str(value)
    text = "".join(character for character in text if character.isprintable())
    text = " ".join(text.split())
    if not text:
        return None
    return text[: max(1, max_length)]


def _coerce_optional_ratio(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if number <= 0.0:
        return 0.0
    if number >= 1.0:
        return 1.0
    return number


def _coerce_ratio(value: object) -> float:
    return _coerce_optional_ratio(value) or 0.0


def _coerce_optional_timestamp(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0.0:
        return None
    return number


def _coerce_optional_positive_timestamp(value: object) -> float | None:
    number = _coerce_optional_timestamp(value)
    if number is None or number <= 0.0:
        return None
    return number


def _coerce_optional_non_negative_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if not math.isfinite(value) or value < 0.0 or not value.is_integer():
            return None
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            return int(text)
        try:
            number = float(text)
        except ValueError:
            return None
        if not math.isfinite(number) or number < 0.0 or not number.is_integer():
            return None
        return int(number)
    return None


def _coerce_positive_int(value: object, *, default: int) -> int:
    number = _coerce_optional_non_negative_int(value)
    if number is None or number <= 0:
        return default
    return number


def _coerce_positive_float(value: object, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number) or number <= 0.0:
        return default
    return number


def _coerce_optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _coerce_bool(value: object) -> bool:
    optional = _coerce_optional_bool(value)
    return bool(optional)


def _coerce_optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return None


def _normalize_enum_token(value: str) -> str:
    return "".join(character for character in value.strip().lower() if character.isalnum())


def _coerce_gesture_value(value: object) -> SocialGestureEvent:
    if isinstance(value, SocialGestureEvent):
        return value
    if isinstance(value, AICameraGestureEvent):
        return _map_gesture(value)
    if isinstance(value, str):
        normalized = _normalize_enum_token(value)
        for member in SocialGestureEvent:
            if normalized in {_normalize_enum_token(member.value), _normalize_enum_token(member.name)}:
                return member
    return SocialGestureEvent.NONE


def _coerce_fine_hand_gesture_value(value: object) -> SocialFineHandGesture:
    if isinstance(value, SocialFineHandGesture):
        return value
    if isinstance(value, AICameraFineHandGesture):
        return _map_fine_hand_gesture(value)
    if isinstance(value, str):
        normalized = _normalize_enum_token(value)
        for member in SocialFineHandGesture:
            if normalized in {_normalize_enum_token(member.value), _normalize_enum_token(member.name)}:
                return member
    return SocialFineHandGesture.NONE


def _take_iterable(value: object, limit: int) -> list[object]:
    if limit <= 0 or value is None:
        return []
    try:
        iterator = iter(value)
    except TypeError:
        return []
    items: list[object] = []
    for item in iterator:
        items.append(item)
        if len(items) >= limit:
            break
    return items


def _sanitize_debug_details(
    details: dict[str, object] | None,
    *,
    max_text_length: int,
) -> dict[str, object] | None:
    if not details:
        return None
    sanitized: dict[str, object] = {}
    for index, (key, value) in enumerate(details.items()):
        if index >= 64:
            break
        safe_key = _sanitize_text(key, max_length=64)
        if not safe_key:
            continue
        if isinstance(value, (bool, int)):
            sanitized[safe_key] = value
        elif isinstance(value, float):
            if math.isfinite(value):
                sanitized[safe_key] = value
        elif isinstance(value, str):
            safe_value = _sanitize_text(value, max_length=max_text_length)
            if safe_value is not None:
                sanitized[safe_key] = safe_value
        elif value is None:
            sanitized[safe_key] = None
        else:
            safe_value = _sanitize_text(repr(value), max_length=max_text_length)
            if safe_value is not None:
                sanitized[safe_key] = safe_value
    return sanitized or None


__all__ = [
    "LocalAICameraObservationProvider",
    "LocalAICameraProviderConfig",
]