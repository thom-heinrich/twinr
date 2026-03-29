"""Automation-fact assembly helpers for proactive observation dispatch.

Purpose: keep fact payload construction and rising-edge event derivation
separate from ops recording and display bridge helpers.

Invariants: automation fact schemas, snapshot side effects, and derived sensor
event names must remain compatible with the legacy observation mixin, while
attention-target derivation now reuses the same runtime perception
orchestrator that drives HDMI and servo follow.
"""

# mypy: ignore-errors

from __future__ import annotations

from typing import Any

from twinr.hardware.respeaker import build_respeaker_claim_payloads, resolve_respeaker_indicator_state

from ...social.camera_surface import ProactiveCameraSnapshot
from ...social.engine import SocialObservation
from ..affect_proxy import derive_affect_proxy
from ..ambiguous_room_guard import derive_ambiguous_room_guard
from ..audio_policy import ReSpeakerAudioPolicySnapshot
from ..known_user_hint import derive_known_user_hint
from ..multimodal_initiative import derive_respeaker_multimodal_initiative
from ..person_state import derive_person_state
from ..portrait_match import derive_portrait_match
from ..speaker_association import derive_respeaker_speaker_association
from .compat import _round_optional_seconds


class ProactiveCoordinatorObservationFactsMixin:
    """Build automation-facing facts and rising-edge events from observations."""

    def _build_automation_facts(
        self,
        observation: SocialObservation,
        *,
        inspected: bool,
        audio_snapshot=None,
        camera_snapshot: ProactiveCameraSnapshot,
        audio_policy_snapshot: ReSpeakerAudioPolicySnapshot | None,
        presence_snapshot,
    ) -> dict[str, Any]:
        """Build the automation-facing fact payload for one observation."""

        now = observation.observed_at
        audio = observation.audio

        speech_detected = audio.speech_detected is True
        quiet = audio.speech_detected is False
        self._speech_detected_since = self._next_since(speech_detected, self._speech_detected_since, now)
        self._quiet_since = self._next_since(quiet, self._quiet_since, now)

        no_motion_for_s = 0.0
        if not observation.pir_motion_detected and self._last_motion_at is not None:
            no_motion_for_s = max(0.0, now - self._last_motion_at)

        presence_session_id = None if presence_snapshot is None else getattr(presence_snapshot, "session_id", None)
        signal_snapshot = None if audio_snapshot is None else getattr(audio_snapshot, "signal_snapshot", None)
        respeaker_claim_contract = build_respeaker_claim_payloads(
            signal_snapshot=signal_snapshot,
            session_id=presence_session_id,
            non_speech_audio_likely=audio.non_speech_audio_likely,
            background_media_likely=audio.background_media_likely,
        )

        facts = {
            "sensor": {
                "inspected": inspected,
                "observed_at": now,
                "captured_at": now,
                "presence_session_id": presence_session_id,
                "voice_activation_armed": None if presence_snapshot is None else presence_snapshot.armed,
                "voice_activation_presence_reason": None if presence_snapshot is None else presence_snapshot.reason,
            },
            "pir": {
                "motion_detected": observation.pir_motion_detected,
                "low_motion": observation.low_motion,
                "no_motion_for_s": round(no_motion_for_s, 3),
            },
            "camera": camera_snapshot.to_automation_facts(),
            "vad": {
                "speech_detected": speech_detected,
                "speech_detected_for_s": round(self._duration_since(self._speech_detected_since, now), 3),
                "quiet": quiet,
                "quiet_for_s": round(self._duration_since(self._quiet_since, now), 3),
                "distress_detected": audio.distress_detected is True,
                "room_quiet": audio.room_quiet,
                "recent_speech_age_s": _round_optional_seconds(audio.recent_speech_age_s),
                "assistant_output_active": audio.assistant_output_active,
                "signal_source": audio.signal_source,
            },
            "respeaker": {
                "runtime_mode": audio.device_runtime_mode,
                "host_control_ready": audio.host_control_ready,
                "transport_reason": audio.transport_reason,
                "azimuth_deg": audio.azimuth_deg,
                "direction_confidence": audio.direction_confidence,
                "non_speech_audio_likely": audio.non_speech_audio_likely,
                "background_media_likely": audio.background_media_likely,
                "speech_overlap_likely": audio.speech_overlap_likely,
                "barge_in_detected": audio.barge_in_detected,
                "mute_active": audio.mute_active,
                **resolve_respeaker_indicator_state(
                    runtime_status=getattr(getattr(self.runtime, "status", None), "value", None),
                    runtime_alert_code=(
                        None if audio_policy_snapshot is None else audio_policy_snapshot.runtime_alert_code
                    ),
                    mute_active=audio.mute_active,
                ).event_data(),
                "claim_contract": respeaker_claim_contract,
            },
            "audio_policy": {
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
                "speaker_direction_stable": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.speaker_direction_stable
                ),
                "mute_blocks_voice_capture": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.mute_blocks_voice_capture
                ),
                "resume_window_open": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.resume_window_open
                ),
                "initiative_block_reason": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.initiative_block_reason
                ),
                "speech_delivery_defer_reason": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.speech_delivery_defer_reason
                ),
                "runtime_alert_code": (
                    None if audio_policy_snapshot is None else audio_policy_snapshot.runtime_alert_code
                ),
            },
        }
        speaker_association = derive_respeaker_speaker_association(
            observed_at=now,
            live_facts=facts,
        )
        multimodal_initiative = derive_respeaker_multimodal_initiative(
            observed_at=now,
            live_facts=facts,
            speaker_association=speaker_association,
        )
        self.latest_speaker_association_snapshot = speaker_association
        self.latest_multimodal_initiative_snapshot = multimodal_initiative
        ambiguous_room_guard = derive_ambiguous_room_guard(
            observed_at=now,
            live_facts=facts,
        )
        portrait_match = derive_portrait_match(
            observed_at=now,
            live_facts=facts,
            provider=self.portrait_match_provider,
            ambiguous_room_guard=ambiguous_room_guard,
            now_monotonic=now,
        )
        presence_session_id = None if presence_snapshot is None else getattr(presence_snapshot, "session_id", None)
        identity_fusion = self.identity_fusion_tracker.observe(
            observed_at=now,
            live_facts=facts,
            voice_status=getattr(self.runtime, "user_voice_status", None),
            voice_confidence=getattr(self.runtime, "user_voice_confidence", None),
            voice_checked_at=getattr(self.runtime, "user_voice_checked_at", None),
            voice_matched_user_id=getattr(self.runtime, "user_voice_user_id", None),
            voice_matched_user_display_name=getattr(self.runtime, "user_voice_user_display_name", None),
            voice_match_source=getattr(self.runtime, "user_voice_match_source", None),
            max_voice_age_s=int(getattr(self.config, "voice_assessment_max_age_s", 120) or 120),
            presence_session_id=presence_session_id,
            ambiguous_room_guard=ambiguous_room_guard,
            speaker_association=speaker_association,
            portrait_match=portrait_match,
        )
        known_user_hint = derive_known_user_hint(
            observed_at=now,
            live_facts=facts,
            voice_status=getattr(self.runtime, "user_voice_status", None),
            voice_confidence=getattr(self.runtime, "user_voice_confidence", None),
            voice_checked_at=getattr(self.runtime, "user_voice_checked_at", None),
            max_voice_age_s=int(getattr(self.config, "voice_assessment_max_age_s", 120) or 120),
            ambiguous_room_guard=ambiguous_room_guard,
            speaker_association=speaker_association,
            portrait_match=portrait_match,
            identity_fusion=identity_fusion,
        )
        affect_proxy = derive_affect_proxy(
            observed_at=now,
            live_facts=facts,
            ambiguous_room_guard=ambiguous_room_guard,
        )
        perception_runtime = self.perception_orchestrator.observe_attention(
            observed_at=now,
            source="automation_observation",
            captured_at=camera_snapshot.last_camera_frame_at,
            camera_snapshot=camera_snapshot,
            audio_observation=audio_snapshot.observation,
            audio_policy_snapshot=audio_policy_snapshot,
            runtime_status=getattr(getattr(self.runtime, "status", None), "value", None),
            presence_session_id=presence_session_id,
            speaker_association=speaker_association,
            identity_fusion=identity_fusion,
        )
        assert perception_runtime.attention is not None
        attention_target = perception_runtime.attention.attention_target
        person_state = derive_person_state(
            observed_at=now,
            live_facts={
                **facts,
                "speaker_association": speaker_association.to_automation_facts(),
                "multimodal_initiative": multimodal_initiative.to_automation_facts(),
                "ambiguous_room_guard": ambiguous_room_guard.to_automation_facts(),
                "identity_fusion": identity_fusion.to_automation_facts(),
                "portrait_match": portrait_match.to_automation_facts(),
                "known_user_hint": known_user_hint.to_automation_facts(),
                "affect_proxy": affect_proxy.to_automation_facts(),
                "attention_target": attention_target.to_automation_facts(),
            },
        )
        self.latest_ambiguous_room_guard_snapshot = ambiguous_room_guard
        self.latest_identity_fusion_snapshot = identity_fusion
        self.latest_portrait_match_snapshot = portrait_match
        self.latest_known_user_hint_snapshot = known_user_hint
        self.latest_affect_proxy_snapshot = affect_proxy
        self.latest_perception_runtime_snapshot = perception_runtime
        self.latest_attention_target_snapshot = attention_target
        self.latest_person_state_snapshot = person_state
        facts["speaker_association"] = speaker_association.to_automation_facts()
        facts["multimodal_initiative"] = multimodal_initiative.to_automation_facts()
        facts["ambiguous_room_guard"] = ambiguous_room_guard.to_automation_facts()
        facts["identity_fusion"] = identity_fusion.to_automation_facts()
        facts["portrait_match"] = portrait_match.to_automation_facts()
        facts["known_user_hint"] = known_user_hint.to_automation_facts()
        facts["affect_proxy"] = affect_proxy.to_automation_facts()
        facts["attention_target"] = attention_target.to_automation_facts()
        facts["person_state"] = person_state.to_automation_facts()
        return facts

    def _derive_sensor_events(
        self,
        facts: dict[str, Any],
        *,
        camera_event_names: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        """Return rising-edge event names derived from the latest fact payload."""

        current_flags = {
            "pir.motion_detected": bool(facts["pir"]["motion_detected"]),
            "vad.speech_detected": bool(facts["vad"]["speech_detected"]),
            "audio_policy.presence_audio_active": bool(facts["audio_policy"]["presence_audio_active"]),
            "audio_policy.quiet_window_open": bool(facts["audio_policy"]["quiet_window_open"]),
            "audio_policy.resume_window_open": bool(facts["audio_policy"]["resume_window_open"]),
            "audio_policy.room_busy_or_overlapping": bool(facts["audio_policy"]["room_busy_or_overlapping"]),
            "audio_policy.barge_in_recent": bool(facts["audio_policy"]["barge_in_recent"]),
            "speaker_association.associated": bool(facts["speaker_association"]["associated"]),
            "multimodal_initiative.ready": bool(facts["multimodal_initiative"]["ready"]),
            "ambiguous_room_guard.guard_active": bool(facts["ambiguous_room_guard"]["guard_active"]),
            "identity_fusion.matches_main_user": bool(facts["identity_fusion"]["matches_main_user"]),
            "portrait_match.matches_reference_user": bool(facts["portrait_match"]["matches_reference_user"]),
            "known_user_hint.matches_main_user": bool(facts["known_user_hint"]["matches_main_user"]),
            "affect_proxy.concern_cue": facts["affect_proxy"]["state"] == "concern_cue",
            "attention_target.session_focus_active": bool(facts["attention_target"]["session_focus_active"]),
            "person_state.interaction_ready": bool(facts["person_state"]["interaction_ready"]),
            "person_state.safety_concern_active": bool(facts["person_state"]["safety_concern_active"]),
            "person_state.calm_personalization_allowed": bool(facts["person_state"]["calm_personalization_allowed"]),
        }
        event_names: list[str] = list(camera_event_names)
        for key, value in current_flags.items():
            previous = self._last_sensor_flags.get(key)
            if value and previous is not True:
                event_names.append(key)
        self._last_sensor_flags = current_flags
        return tuple(event_names)

    def _next_since(self, active: bool, since: float | None, now: float) -> float | None:
        """Advance or clear one duration anchor depending on activity."""

        if active:
            return now if since is None else since
        return None

    def _duration_since(self, since: float | None, now: float) -> float:
        """Return the elapsed duration for one optional activity anchor."""

        if since is None:
            return 0.0
        return max(0.0, now - since)
