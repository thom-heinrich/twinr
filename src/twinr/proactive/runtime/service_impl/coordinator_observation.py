"""Observation recording and composition layer for the coordinator.

Purpose: keep changed-only ops recording at the historic mixin path while
delegating display/camera-surface bridges and automation-fact assembly into
focused sibling mixins.

Invariants: fact payloads, rising-edge event names, ops-event schemas, and
display helper calls must remain compatible with the legacy implementation.
"""

# CHANGELOG: 2026-03-29
# BUG-1: Changed-only observation dedupe now tracks discrete camera/audio health
#        and snapshot event-data fields that were emitted but not part of the
#        dedupe key, fixing silent misses for camera outages, person-count
#        changes, gesture-state changes, and similar operator-visible state.
# BUG-2: ReSpeaker runtime blocker lifecycle now clears correctly when the
#        runtime alert code returns to a neutral/None state.
# BUG-3: Ops-state caches are only advanced after successful event append so a
#        transient logging/backend failure does not permanently suppress the
#        next retry for the same state.
# SEC-1: All externally influenced event strings and nested payload values are
#        now bounded and control-character neutralized before emit/logging to
#        prevent log injection and log-flood amplification on edge deployments.
# IMP-1: Observation keys now use stable numeric normalization to avoid event
#        churn from float jitter in multimodal confidence fields.
# IMP-2: Snapshot-key derivation now reuses snapshot event_data() payloads so
#        new snapshot fields participate in changed-only recording without
#        manual tuple maintenance in this mixin.


from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Callable, cast

from .coordinator_observation_display import ProactiveCoordinatorObservationDisplayMixin
from .coordinator_observation_facts import ProactiveCoordinatorObservationFactsMixin
from twinr.hardware.respeaker import resolve_respeaker_indicator_state
from twinr.proactive.runtime.service_impl.compat import (
    _format_firmware_version,
    _record_respeaker_dead_capture_blocker,
    _round_optional_ratio,
    _round_optional_seconds,
)

from ...social.engine import SocialAudioObservation, SocialObservation, SocialTriggerDecision
from ..audio_policy import ReSpeakerAudioPolicySnapshot
from ..runtime_contract import is_respeaker_runtime_hard_block


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_EVENT_TEXT_MAX_LEN = 512
_EVENT_MESSAGE_MAX_LEN = 256
_EVENT_EMIT_TOKEN_MAX_LEN = 96
_EVENT_LIST_MAX_ITEMS = 16
_EVENT_DICT_MAX_ITEMS = 32
_EVENT_MAX_DEPTH = 4
_AUDIO_POLICY_UNBLOCKED_TRIGGERS = frozenset(
    {"possible_fall", "floor_stillness", "distress_possible"}
)


def _neutralize_log_text(value: str | None, *, max_len: int = _EVENT_TEXT_MAX_LEN) -> str | None:
    """Return one bounded single-line string safe to emit into operator logs."""

    if value is None:
        return None
    normalized = _CONTROL_CHAR_RE.sub(" ", str(value))
    normalized = " ".join(normalized.split())
    if len(normalized) <= max_len:
        return normalized
    if max_len <= 1:
        return normalized[:max_len]
    return normalized[: max_len - 1] + "…"


def _stable_number_for_key(value: Any) -> Any:
    """Return a hash-stable numeric representation for changed-only keys."""

    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 3)
    return value


def _freeze_for_key(value: Any, *, depth: int = 0) -> Any:
    """Convert nested payloads to a bounded hashable form for dedupe keys."""

    if depth >= _EVENT_MAX_DEPTH:
        return "max_depth"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return _stable_number_for_key(value)
    if isinstance(value, str):
        return _neutralize_log_text(value, max_len=128)
    if isinstance(value, Mapping):
        items = list(value.items())[:_EVENT_DICT_MAX_ITEMS]
        return tuple(
            sorted(
                (
                    _neutralize_log_text(str(key), max_len=96),
                    _freeze_for_key(item, depth=depth + 1),
                )
                for key, item in items
            )
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_for_key(item, depth=depth + 1) for item in list(value)[:_EVENT_LIST_MAX_ITEMS])
    return _neutralize_log_text(repr(value), max_len=128)


def _sanitize_event_value(value: Any, *, depth: int = 0) -> Any:
    """Bound nested event values before shipping them to logging backends."""

    if depth >= _EVENT_MAX_DEPTH:
        return _neutralize_log_text("max_depth", max_len=32)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, str):
        return _neutralize_log_text(value)
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in list(value.items())[:_EVENT_DICT_MAX_ITEMS]:
            safe_key = _neutralize_log_text(str(key), max_len=96)
            if safe_key is None:
                continue
            sanitized[safe_key] = _sanitize_event_value(item, depth=depth + 1)
        return sanitized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitize_event_value(item, depth=depth + 1) for item in list(value)[:_EVENT_LIST_MAX_ITEMS]]
    return _neutralize_log_text(repr(value))


class ProactiveCoordinatorObservationMixin(
    ProactiveCoordinatorObservationDisplayMixin,
    ProactiveCoordinatorObservationFactsMixin,
):
    """Provide changed-only ops recording plus composed observation helpers."""

    def _emit_safe(self, key: str, value: str | None) -> None:
        """Emit one bounded operator token without propagating logging junk."""

        safe_value = _neutralize_log_text(value, max_len=_EVENT_EMIT_TOKEN_MAX_LEN) or "unknown"
        emit = getattr(self, "_emit", None)
        if not callable(emit):
            return
        emit_fn = cast(Callable[[str], None] | None, emit)
        try:
            if emit_fn is None:
                return
            emit_fn(f"{key}={safe_value}")  # pylint: disable=not-callable
        except Exception:
            return

    def _append_ops_event_safe(
        self,
        *,
        event: str,
        message: str,
        data: Mapping[str, Any],
        level: str | None = None,
    ) -> bool:
        """Append one ops event with bounded payloads and failure isolation."""

        payload = cast(dict[str, Any], _sanitize_event_value(dict(data)))
        kwargs: dict[str, Any] = {
            "event": event,
            "message": _neutralize_log_text(message, max_len=_EVENT_MESSAGE_MAX_LEN) or message,
            "data": payload,
        }
        if level is not None:
            kwargs["level"] = level
        try:
            self._append_ops_event(**kwargs)
            return True
        except Exception as exc:
            self._emit_safe("ops_event_recording_failed", type(exc).__name__)
            return False

    def _snapshot_event_key(self, snapshot: Any) -> Any:
        """Return one hashable representation of snapshot.event_data()."""

        if snapshot is None:
            return None
        callback = getattr(snapshot, "event_data", None)
        if not callable(callback):
            return None
        try:
            return _freeze_for_key(callback())
        except Exception:
            return None

    def _normalised_azimuth_key(self, value: Any) -> int | None:
        """Bucket azimuth readings to suppress jitter-driven event churn."""

        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        return int(round(numeric / 5.0) * 5)

    def _observation_key(
        self,
        observation: SocialObservation,
        *,
        inspected: bool,
        audio_policy_snapshot: ReSpeakerAudioPolicySnapshot | None,
        presence_snapshot: Any,
        runtime_status_value: str | None,
    ) -> tuple[Any, ...]:
        """Return the stable changed-only key for one observation event."""

        presence_session_id = None if presence_snapshot is None else getattr(presence_snapshot, "session_id", None)
        top_evaluation = self._safety_trigger_fusion.best_evaluation

        return (
            inspected,
            runtime_status_value,
            observation.pir_motion_detected,
            observation.low_motion,
            observation.vision.person_visible,
            observation.vision.person_count,
            observation.vision.primary_person_zone.value,
            observation.vision.looking_toward_device,
            observation.vision.person_near_device,
            observation.vision.engaged_with_device,
            observation.vision.body_pose.value,
            observation.vision.motion_state.value,
            observation.vision.smiling,
            observation.vision.hand_or_object_near_camera,
            observation.vision.gesture_event.value,
            observation.vision.fine_hand_gesture.value,
            observation.vision.camera_online,
            observation.vision.camera_ready,
            observation.vision.camera_ai_ready,
            _neutralize_log_text(observation.vision.camera_error, max_len=128),
            observation.audio.speech_detected,
            observation.audio.distress_detected,
            observation.audio.room_quiet,
            observation.audio.assistant_output_active,
            observation.audio.device_runtime_mode,
            observation.audio.host_control_ready,
            observation.audio.transport_reason,
            observation.audio.non_speech_audio_likely,
            observation.audio.background_media_likely,
            observation.audio.signal_source,
            _round_optional_ratio(observation.audio.direction_confidence),
            observation.audio.speech_overlap_likely,
            observation.audio.barge_in_detected,
            observation.audio.mute_active,
            None if audio_policy_snapshot is None else audio_policy_snapshot.presence_audio_active,
            None if audio_policy_snapshot is None else audio_policy_snapshot.recent_follow_up_speech,
            None if audio_policy_snapshot is None else audio_policy_snapshot.room_busy_or_overlapping,
            None if audio_policy_snapshot is None else audio_policy_snapshot.quiet_window_open,
            None if audio_policy_snapshot is None else audio_policy_snapshot.non_speech_audio_likely,
            None if audio_policy_snapshot is None else audio_policy_snapshot.background_media_likely,
            None if audio_policy_snapshot is None else audio_policy_snapshot.barge_in_recent,
            None if audio_policy_snapshot is None else audio_policy_snapshot.speaker_direction_stable,
            None if audio_policy_snapshot is None else audio_policy_snapshot.mute_blocks_voice_capture,
            None if audio_policy_snapshot is None else audio_policy_snapshot.resume_window_open,
            None if audio_policy_snapshot is None else audio_policy_snapshot.initiative_block_reason,
            None if audio_policy_snapshot is None else audio_policy_snapshot.speech_delivery_defer_reason,
            None if audio_policy_snapshot is None else audio_policy_snapshot.runtime_alert_code,
            None if presence_snapshot is None else presence_snapshot.armed,
            None if presence_snapshot is None else presence_snapshot.reason,
            presence_session_id,
            None if presence_snapshot is None else presence_snapshot.presence_audio_active,
            None if presence_snapshot is None else presence_snapshot.recent_follow_up_speech,
            None if presence_snapshot is None else presence_snapshot.room_busy_or_overlapping,
            None if presence_snapshot is None else presence_snapshot.quiet_window_open,
            None if presence_snapshot is None else presence_snapshot.barge_in_recent,
            None if presence_snapshot is None else presence_snapshot.speaker_direction_stable,
            None if presence_snapshot is None else presence_snapshot.mute_blocks_voice_capture,
            None if presence_snapshot is None else presence_snapshot.resume_window_open,
            self._snapshot_event_key(self.latest_speaker_association_snapshot),
            self._snapshot_event_key(self.latest_multimodal_initiative_snapshot),
            self._snapshot_event_key(self.latest_ambiguous_room_guard_snapshot),
            self._snapshot_event_key(self.latest_identity_fusion_snapshot),
            self._snapshot_event_key(self.latest_portrait_match_snapshot),
            self._snapshot_event_key(self.latest_known_user_hint_snapshot),
            self._snapshot_event_key(self.latest_affect_proxy_snapshot),
            self._snapshot_event_key(self.latest_attention_target_snapshot),
            self._snapshot_event_key(self.latest_person_state_snapshot),
            None if top_evaluation is None else top_evaluation.trigger_id,
            None if top_evaluation is None else top_evaluation.passed,
            None if top_evaluation is None else top_evaluation.blocked_reason,
        )

    def _is_low_motion(self, now: float, *, motion_active: bool) -> bool:
        """Return whether recent PIR history qualifies as low motion."""

        if motion_active:
            return False
        if self._last_motion_at is None:
            return False
        return (now - self._last_motion_at) >= self.config.proactive_low_motion_after_s

    def _note_audio_observer_runtime_context(
        self,
        *,
        now: float,
        motion_active: bool,
        inspect_requested: bool,
        runtime_status_value: str,
    ) -> None:
        """Forward current runtime context to schedulable audio observers."""

        callback = getattr(self.audio_observer, "note_runtime_context", None)
        if not callable(callback):
            return
        note_runtime_context = cast(Callable[..., None], callback)
        presence_snapshot = self.latest_presence_snapshot
        note_runtime_context(
            observed_at=now,
            motion_active=motion_active,
            inspect_requested=inspect_requested,
            presence_session_armed=bool(
                presence_snapshot is not None and getattr(presence_snapshot, "armed", False)
            ),
            assistant_output_active=runtime_status_value == "answering",
        )

    def _observe_audio_policy(
        self,
        *,
        now: float,
        audio_observation: SocialAudioObservation,
    ) -> ReSpeakerAudioPolicySnapshot:
        """Derive one conservative ReSpeaker policy snapshot for this tick."""

        snapshot = self.audio_policy_tracker.observe(now=now, audio=audio_observation)
        self.latest_audio_policy_snapshot = snapshot
        return snapshot

    def _record_observation_if_changed(
        self,
        observation: SocialObservation,
        *,
        inspected: bool,
        vision_snapshot=None,
        audio_snapshot=None,
        audio_policy_snapshot: ReSpeakerAudioPolicySnapshot | None = None,
        presence_snapshot=None,
        runtime_status_value: str | None = None,
    ) -> None:
        """Append one observation event only when the visible state changed."""

        presence_session_id = None if presence_snapshot is None else getattr(presence_snapshot, "session_id", None)
        observation_key = self._observation_key(
            observation,
            inspected=inspected,
            audio_policy_snapshot=audio_policy_snapshot,
            presence_snapshot=presence_snapshot,
            runtime_status_value=runtime_status_value,
        )
        if observation_key == self._last_observation_key:
            return
        if not inspected and self._last_observation_key is None:
            self._last_observation_key = observation_key
            return

        if audio_policy_snapshot is not None:
            self._record_respeaker_runtime_alert_if_changed(
                observation.audio,
                audio_policy_snapshot=audio_policy_snapshot,
            )

        indicator_state = self._indicator_state_for_observation(
            observation=observation,
            audio_policy_snapshot=audio_policy_snapshot,
            runtime_status_value=runtime_status_value,
        )
        data: dict[str, Any] = {
            "inspected": inspected,
            "runtime_status": runtime_status_value,
            "pir_motion_detected": observation.pir_motion_detected,
            "low_motion": observation.low_motion,
            "person_visible": observation.vision.person_visible,
            "camera_person_count": observation.vision.person_count,
            "camera_primary_person_zone": observation.vision.primary_person_zone.value,
            "camera_primary_person_center_x": _round_optional_ratio(observation.vision.primary_person_center_x),
            "camera_primary_person_center_y": _round_optional_ratio(observation.vision.primary_person_center_y),
            "looking_toward_device": observation.vision.looking_toward_device,
            "camera_person_near_device": observation.vision.person_near_device,
            "camera_engaged_with_device": observation.vision.engaged_with_device,
            "camera_visual_attention_score": _round_optional_ratio(observation.vision.visual_attention_score),
            "body_pose": observation.vision.body_pose.value,
            "camera_pose_confidence": _round_optional_ratio(observation.vision.pose_confidence),
            "camera_motion_state": observation.vision.motion_state.value,
            "camera_motion_confidence": _round_optional_ratio(observation.vision.motion_confidence),
            "smiling": observation.vision.smiling,
            "hand_or_object_near_camera": observation.vision.hand_or_object_near_camera,
            "camera_gesture_event": observation.vision.gesture_event.value,
            "camera_gesture_confidence": _round_optional_ratio(observation.vision.gesture_confidence),
            "camera_fine_hand_gesture": observation.vision.fine_hand_gesture.value,
            "camera_fine_hand_gesture_confidence": _round_optional_ratio(
                observation.vision.fine_hand_gesture_confidence
            ),
            "camera_online": observation.vision.camera_online,
            "camera_ready": observation.vision.camera_ready,
            "camera_ai_ready": observation.vision.camera_ai_ready,
            "camera_error": observation.vision.camera_error,
            "speech_detected": observation.audio.speech_detected,
            "distress_detected": observation.audio.distress_detected,
            "audio_room_quiet": observation.audio.room_quiet,
            "audio_recent_speech_age_s": _round_optional_seconds(observation.audio.recent_speech_age_s),
            "audio_assistant_output_active": observation.audio.assistant_output_active,
            "audio_azimuth_deg": observation.audio.azimuth_deg,
            "audio_direction_confidence": observation.audio.direction_confidence,
            "audio_signal_source": observation.audio.signal_source,
            "audio_device_runtime_mode": observation.audio.device_runtime_mode,
            "audio_host_control_ready": observation.audio.host_control_ready,
            "audio_transport_reason": observation.audio.transport_reason,
            "audio_non_speech_audio_likely": observation.audio.non_speech_audio_likely,
            "audio_background_media_likely": observation.audio.background_media_likely,
            "audio_speech_overlap_likely": observation.audio.speech_overlap_likely,
            "audio_barge_in_detected": observation.audio.barge_in_detected,
            "audio_mute_active": observation.audio.mute_active,
            "audio_indicator_mode": indicator_state.mode,
            "audio_indicator_semantics": indicator_state.semantics,
        }
        if self.latest_speaker_association_snapshot is not None:
            data.update(self.latest_speaker_association_snapshot.event_data())
        if self.latest_multimodal_initiative_snapshot is not None:
            data.update(self.latest_multimodal_initiative_snapshot.event_data())
        if self.latest_ambiguous_room_guard_snapshot is not None:
            data.update(self.latest_ambiguous_room_guard_snapshot.event_data())
        if self.latest_identity_fusion_snapshot is not None:
            data.update(self.latest_identity_fusion_snapshot.event_data())
        if self.latest_portrait_match_snapshot is not None:
            data.update(self.latest_portrait_match_snapshot.event_data())
        if self.latest_known_user_hint_snapshot is not None:
            data.update(self.latest_known_user_hint_snapshot.event_data())
        if self.latest_affect_proxy_snapshot is not None:
            data.update(self.latest_affect_proxy_snapshot.event_data())
        if self.latest_attention_target_snapshot is not None:
            data.update(self.latest_attention_target_snapshot.event_data())
        if self.latest_person_state_snapshot is not None:
            data.update(self.latest_person_state_snapshot.event_data())
        if audio_policy_snapshot is not None:
            data.update(
                {
                    "presence_audio_active": audio_policy_snapshot.presence_audio_active,
                    "recent_follow_up_speech": audio_policy_snapshot.recent_follow_up_speech,
                    "room_busy_or_overlapping": audio_policy_snapshot.room_busy_or_overlapping,
                    "quiet_window_open": audio_policy_snapshot.quiet_window_open,
                    "non_speech_audio_likely": audio_policy_snapshot.non_speech_audio_likely,
                    "background_media_likely": audio_policy_snapshot.background_media_likely,
                    "barge_in_recent": audio_policy_snapshot.barge_in_recent,
                    "speaker_direction_stable": audio_policy_snapshot.speaker_direction_stable,
                    "mute_blocks_voice_capture": audio_policy_snapshot.mute_blocks_voice_capture,
                    "resume_window_open": audio_policy_snapshot.resume_window_open,
                    "audio_initiative_block_reason": audio_policy_snapshot.initiative_block_reason,
                    "audio_speech_delivery_defer_reason": audio_policy_snapshot.speech_delivery_defer_reason,
                    "respeaker_runtime_alert_code": audio_policy_snapshot.runtime_alert_code,
                }
            )
        if presence_snapshot is not None:
            data.update(
                {
                    "voice_activation_armed": presence_snapshot.armed,
                    "voice_activation_presence_reason": presence_snapshot.reason,
                    "voice_activation_presence_session_id": presence_session_id,
                    "voice_activation_presence_audio_active": presence_snapshot.presence_audio_active,
                    "voice_activation_recent_follow_up_speech": presence_snapshot.recent_follow_up_speech,
                    "voice_activation_room_busy_or_overlapping": presence_snapshot.room_busy_or_overlapping,
                    "voice_activation_quiet_window_open": presence_snapshot.quiet_window_open,
                    "voice_activation_barge_in_recent": presence_snapshot.barge_in_recent,
                    "voice_activation_speaker_direction_stable": presence_snapshot.speaker_direction_stable,
                    "voice_activation_mute_blocks_voice_capture": presence_snapshot.mute_blocks_voice_capture,
                    "voice_activation_resume_window_open": presence_snapshot.resume_window_open,
                }
            )
        if vision_snapshot is not None:
            data.update(
                {
                    "vision_model": vision_snapshot.model,
                    "vision_request_id": vision_snapshot.request_id,
                    "vision_response_id": vision_snapshot.response_id,
                }
            )
        if audio_snapshot is not None and audio_snapshot.sample is not None:
            data.update(
                {
                    "audio_average_rms": audio_snapshot.sample.average_rms,
                    "audio_peak_rms": audio_snapshot.sample.peak_rms,
                    "audio_active_ratio": audio_snapshot.sample.active_ratio,
                    "audio_active_chunks": audio_snapshot.sample.active_chunk_count,
                    "audio_chunk_count": audio_snapshot.sample.chunk_count,
                }
            )
        if audio_snapshot is not None and audio_snapshot.signal_snapshot is not None:
            data.update(
                {
                    "audio_requires_elevated_permissions": audio_snapshot.signal_snapshot.requires_elevated_permissions,
                    "audio_firmware_version": _format_firmware_version(audio_snapshot.signal_snapshot.firmware_version),
                    "audio_gpo_logic_levels": audio_snapshot.signal_snapshot.gpo_logic_levels,
                }
            )
        top_evaluation = self._safety_trigger_fusion.best_evaluation
        if top_evaluation is not None and top_evaluation.score > 0.0:
            data.update(
                {
                    "top_trigger": top_evaluation.trigger_id,
                    "top_score": top_evaluation.score,
                    "top_threshold": top_evaluation.threshold,
                    "top_trigger_passed": top_evaluation.passed,
                }
            )
            if top_evaluation.blocked_reason is not None:
                data["top_blocked_reason"] = top_evaluation.blocked_reason
            if top_evaluation.passed and audio_policy_snapshot is not None:
                data["top_audio_policy_block_reason"] = audio_policy_snapshot.initiative_block_reason

        if self._append_ops_event_safe(
            event="proactive_observation",
            message="Proactive monitor recorded a changed observation.",
            data=data,
        ):
            self._last_observation_key = observation_key

    def _indicator_state_for_observation(
        self,
        *,
        observation: SocialObservation,
        audio_policy_snapshot: ReSpeakerAudioPolicySnapshot | None,
        runtime_status_value: str | None,
    ):
        """Return the bounded indicator-state projection for one observation event."""

        return resolve_respeaker_indicator_state(
            runtime_status=runtime_status_value,
            runtime_alert_code=(
                None if audio_policy_snapshot is None else audio_policy_snapshot.runtime_alert_code
            ),
            mute_active=observation.audio.mute_active,
        )

    def _record_presence_if_changed(self, snapshot) -> None:
        """Append one ops event when the presence-session state changes."""

        session_id = getattr(snapshot, "session_id", None)
        key = (
            snapshot.armed,
            snapshot.reason,
            session_id,
            snapshot.presence_audio_active,
            snapshot.recent_follow_up_speech,
            snapshot.room_busy_or_overlapping,
            snapshot.quiet_window_open,
            snapshot.barge_in_recent,
            snapshot.speaker_direction_stable,
            snapshot.mute_blocks_voice_capture,
            snapshot.resume_window_open,
            snapshot.device_runtime_mode,
            snapshot.transport_reason,
        )
        if key == self._last_presence_key:
            return
        if self._append_ops_event_safe(
            event="voice_activation_presence_changed",
            message="Voice-activation presence session changed.",
            data={
                "armed": snapshot.armed,
                "reason": snapshot.reason,
                "session_id": session_id,
                "person_visible": snapshot.person_visible,
                "last_person_seen_age_s": snapshot.last_person_seen_age_s,
                "last_motion_age_s": snapshot.last_motion_age_s,
                "last_speech_age_s": snapshot.last_speech_age_s,
                "presence_audio_active": snapshot.presence_audio_active,
                "recent_follow_up_speech": snapshot.recent_follow_up_speech,
                "room_busy_or_overlapping": snapshot.room_busy_or_overlapping,
                "quiet_window_open": snapshot.quiet_window_open,
                "barge_in_recent": snapshot.barge_in_recent,
                "speaker_direction_stable": snapshot.speaker_direction_stable,
                "mute_blocks_voice_capture": snapshot.mute_blocks_voice_capture,
                "resume_window_open": snapshot.resume_window_open,
                "device_runtime_mode": snapshot.device_runtime_mode,
                "transport_reason": snapshot.transport_reason,
            },
        ):
            self._last_presence_key = key

    def _record_respeaker_runtime_alert_if_changed(
        self,
        audio: SocialAudioObservation,
        *,
        audio_policy_snapshot: ReSpeakerAudioPolicySnapshot,
    ) -> None:
        """Append one explicit operator-readable ReSpeaker runtime alert on change."""

        alert_code = audio_policy_snapshot.runtime_alert_code
        if alert_code is None:
            self._last_respeaker_runtime_alert_code = None
            self._record_respeaker_runtime_blocker_if_changed(
                alert_code=None,
                message="ReSpeaker runtime state returned to neutral monitoring.",
                audio=audio,
            )
            return

        if alert_code == self._last_respeaker_runtime_alert_code:
            self._record_respeaker_runtime_blocker_if_changed(
                alert_code=alert_code,
                message=audio_policy_snapshot.runtime_alert_message or "ReSpeaker runtime state changed.",
                audio=audio,
            )
            return

        level = "warning" if alert_code != "ready" else "info"
        message = audio_policy_snapshot.runtime_alert_message or "ReSpeaker runtime state changed."
        self._emit_safe("respeaker_runtime_alert", alert_code)
        if self._append_ops_event_safe(
            event="respeaker_runtime_alert",
            level=level,
            message=message,
            data={
                "alert_code": alert_code,
                "device_runtime_mode": audio.device_runtime_mode,
                "host_control_ready": audio.host_control_ready,
                "transport_reason": audio.transport_reason,
                "mute_active": audio.mute_active,
            },
        ):
            self._last_respeaker_runtime_alert_code = alert_code
        self._record_respeaker_runtime_blocker_if_changed(
            alert_code=alert_code,
            message=message,
            audio=audio,
        )

    def _record_respeaker_runtime_blocker_if_changed(
        self,
        *,
        alert_code: str | None,
        message: str,
        audio: SocialAudioObservation,
    ) -> None:
        """Emit explicit hard-block lifecycle events for DFU runtime states."""

        if alert_code is not None and is_respeaker_runtime_hard_block(alert_code):
            if alert_code == self._last_respeaker_runtime_blocker_code:
                return
            self._emit_safe("respeaker_runtime_blocker", alert_code)
            if self._append_ops_event_safe(
                event="respeaker_runtime_blocker",
                level="error",
                message=message,
                data={
                    "alert_code": alert_code,
                    "device_runtime_mode": audio.device_runtime_mode,
                    "host_control_ready": audio.host_control_ready,
                    "transport_reason": audio.transport_reason,
                    "mute_active": audio.mute_active,
                },
            ):
                self._last_respeaker_runtime_blocker_code = alert_code
            return

        if self._last_respeaker_runtime_blocker_code is None:
            return
        cleared_code = self._last_respeaker_runtime_blocker_code
        self._emit_safe("respeaker_runtime_blocker_cleared", cleared_code)
        if self._append_ops_event_safe(
            event="respeaker_runtime_blocker_cleared",
            message="ReSpeaker hard runtime blocker cleared and capture is usable again.",
            data={
                "cleared_alert_code": cleared_code,
                "current_alert_code": alert_code,
                "device_runtime_mode": audio.device_runtime_mode,
                "host_control_ready": audio.host_control_ready,
            },
        ):
            self._last_respeaker_runtime_blocker_code = None

    def _block_respeaker_dead_capture(self, error) -> None:
        """Fail closed on unreadable ReSpeaker capture during live monitoring."""

        self.audio_observer = self._null_audio_observer
        self._audio_observer_fallback_factory = None
        if self._last_respeaker_runtime_blocker_code == "dead_capture":
            self._last_respeaker_runtime_alert_code = "capture_unknown"
            return
        _record_respeaker_dead_capture_blocker(
            runtime=self.runtime,
            emit=self.emit,
            probe=error.probe,
            stage="runtime",
            signal=None,
        )
        self._last_respeaker_runtime_alert_code = "capture_unknown"
        self._last_respeaker_runtime_blocker_code = "dead_capture"

    def _record_trigger_detected(
        self,
        decision: SocialTriggerDecision,
        *,
        observation: SocialObservation,
        review=None,
    ) -> None:
        """Record one proactive trigger that reached dispatch evaluation."""

        data: dict[str, Any] = {
            "trigger": decision.trigger_id,
            "reason": decision.reason,
            "priority": int(decision.priority),
            "prompt": decision.prompt,
            "score": decision.score,
            "threshold": decision.threshold,
            "evidence": [item.to_dict() for item in decision.evidence],
            "person_visible": observation.vision.person_visible,
            "body_pose": observation.vision.body_pose.value,
            "speech_detected": observation.audio.speech_detected,
            "distress_detected": observation.audio.distress_detected,
            "low_motion": observation.low_motion,
            "trigger_source": self._safety_trigger_fusion.last_selected_source,
        }
        fused_claim = self._safety_trigger_fusion.last_selected_claim
        if fused_claim is not None:
            data.update(
                {
                    "fused_claim_state": fused_claim.state,
                    "fused_claim_confidence": fused_claim.confidence,
                    "fused_claim_action_level": fused_claim.action_level.value,
                    "fused_claim_supporting_audio_events": list(fused_claim.supporting_audio_events),
                    "fused_claim_supporting_vision_events": list(fused_claim.supporting_vision_events),
                    "fused_claim_blocked_by": list(fused_claim.blocked_by),
                }
            )
        if review is not None:
            data.update(
                {
                    "vision_review_decision": review.decision,
                    "vision_review_confidence": review.confidence,
                    "vision_review_reason": review.reason,
                    "vision_review_scene": review.scene,
                    "vision_review_frame_count": review.frame_count,
                    "vision_review_response_id": review.response_id,
                    "vision_review_request_id": review.request_id,
                    "vision_review_model": review.model,
                }
            )
        self._append_ops_event_safe(
            event="proactive_trigger_detected",
            message="Proactive trigger conditions were met.",
            data=data,
        )

    def _record_vision_review(
        self,
        decision: SocialTriggerDecision,
        *,
        review,
    ) -> None:
        """Record one buffered vision review result."""

        self._append_ops_event_safe(
            event="proactive_vision_reviewed",
            message="Buffered proactive camera frames were reviewed before speaking.",
            data={
                "trigger": decision.trigger_id,
                "approved": review.approved,
                "decision": review.decision,
                "confidence": review.confidence,
                "reason": review.reason,
                "scene": review.scene,
                "frame_count": review.frame_count,
                "response_id": review.response_id,
                "request_id": review.request_id,
                "model": review.model,
            },
        )

    def _record_trigger_skipped_vision_review(
        self,
        decision: SocialTriggerDecision,
        *,
        review,
    ) -> None:
        """Record that buffered vision review rejected one trigger."""

        self._emit_safe("social_trigger_skipped", "vision_review_rejected")
        self._append_ops_event_safe(
            event="social_trigger_skipped",
            message="Social trigger prompt was skipped because buffered frame review rejected it.",
            data={
                "trigger": decision.trigger_id,
                "reason": "vision_review_rejected",
                "priority": int(decision.priority),
                "score": decision.score,
                "threshold": decision.threshold,
                "vision_review_decision": review.decision,
                "vision_review_confidence": review.confidence,
                "vision_review_reason": review.reason,
                "vision_review_scene": review.scene,
                "vision_review_frame_count": review.frame_count,
                "vision_review_response_id": review.response_id,
                "vision_review_request_id": review.request_id,
                "vision_review_model": review.model,
            },
        )

    def _record_trigger_skipped_vision_review_unavailable(
        self,
        decision: SocialTriggerDecision,
    ) -> None:
        """Record that buffered vision review was unavailable for one trigger."""

        self._emit_safe("social_trigger_skipped", "vision_review_unavailable")
        self._append_ops_event_safe(
            event="social_trigger_skipped",
            message="Social trigger prompt was skipped because buffered frame review was unavailable.",
            data={
                "trigger": decision.trigger_id,
                "reason": "vision_review_unavailable",
                "priority": int(decision.priority),
                "score": decision.score,
                "threshold": decision.threshold,
            },
        )

    def _presence_session_block_reason(
        self,
        decision: SocialTriggerDecision,
        *,
        presence_snapshot,
    ) -> str | None:
        """Return a presence-session block reason for one trigger if any."""

        if presence_snapshot is None:
            return None
        session_id = getattr(presence_snapshot, "session_id", None)
        if decision.trigger_id != "possible_fall":
            return None
        if not self.config.proactive_possible_fall_once_per_presence_session:
            return None
        if not presence_snapshot.armed or session_id is None:
            return None
        if self._last_possible_fall_prompted_session_id != session_id:
            return None
        return "already_prompted_this_presence_session"

    def _record_trigger_skipped_presence_session(
        self,
        decision: SocialTriggerDecision,
        *,
        presence_snapshot,
        reason: str,
    ) -> None:
        """Record that a trigger was skipped by per-session suppression."""

        session_id = None if presence_snapshot is None else getattr(presence_snapshot, "session_id", None)
        self._emit_safe("social_trigger_skipped", reason)
        self._append_ops_event_safe(
            event="social_trigger_skipped",
            message="Social trigger prompt was skipped because it already fired in the current presence session.",
            data={
                "trigger": decision.trigger_id,
                "reason": reason,
                "presence_session_id": session_id,
                "presence_reason": None if presence_snapshot is None else presence_snapshot.reason,
                "priority": int(decision.priority),
                "score": decision.score,
                "threshold": decision.threshold,
            },
        )

    def _audio_policy_block_reason(
        self,
        decision: SocialTriggerDecision,
        *,
        presence_snapshot,
        audio_policy_snapshot: ReSpeakerAudioPolicySnapshot | None,
    ) -> str | None:
        """Return a conservative ReSpeaker-driven suppression reason for one trigger."""

        del presence_snapshot
        if decision.trigger_id in _AUDIO_POLICY_UNBLOCKED_TRIGGERS:
            return None
        if audio_policy_snapshot is None:
            return None
        return audio_policy_snapshot.initiative_block_reason

    def _record_trigger_skipped_audio_policy(
        self,
        decision: SocialTriggerDecision,
        *,
        presence_snapshot,
        audio_policy_snapshot: ReSpeakerAudioPolicySnapshot | None,
        reason: str,
    ) -> None:
        """Record that a trigger was skipped by conservative ReSpeaker policy hooks."""

        session_id = None if presence_snapshot is None else getattr(presence_snapshot, "session_id", None)
        self._emit_safe("social_trigger_skipped", reason)
        self._append_ops_event_safe(
            event="social_trigger_skipped",
            message="Social trigger prompt was skipped by conservative ReSpeaker audio policy.",
            data={
                "trigger": decision.trigger_id,
                "reason": reason,
                "presence_session_id": session_id,
                "presence_reason": None if presence_snapshot is None else presence_snapshot.reason,
                "priority": int(decision.priority),
                "score": decision.score,
                "threshold": decision.threshold,
                "presence_audio_active": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.presence_audio_active
                ),
                "recent_follow_up_speech": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.recent_follow_up_speech
                ),
                "room_busy_or_overlapping": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.room_busy_or_overlapping
                ),
                "quiet_window_open": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.quiet_window_open
                ),
                "non_speech_audio_likely": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.non_speech_audio_likely
                ),
                "background_media_likely": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.background_media_likely
                ),
                "barge_in_recent": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.barge_in_recent
                ),
                "resume_window_open": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.resume_window_open
                ),
                "mute_blocks_voice_capture": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.mute_blocks_voice_capture
                ),
                "speech_delivery_defer_reason": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.speech_delivery_defer_reason
                ),
                "runtime_alert_code": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.runtime_alert_code
                ),
            },
        )
