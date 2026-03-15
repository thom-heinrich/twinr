from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any, Callable
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware import GpioPirMonitor, V4L2StillCamera, configured_pir_monitor
from twinr.hardware.audio import AmbientAudioCaptureWindow, AmbientAudioSampler, pcm16_to_wav_bytes
from twinr.ops.paths import resolve_ops_paths_for_config
from twinr.proactive.engine import SocialAudioObservation, SocialObservation, SocialTriggerDecision, SocialTriggerEngine, SocialVisionObservation
from twinr.proactive.openwakeword_stream import OpenWakeWordStreamingMonitor, WakewordStreamDetection
from twinr.proactive.openwakeword_spotter import WakewordOpenWakeWordSpotter
from twinr.proactive.observers import AmbientAudioObservationProvider, NullAudioObservationProvider, OpenAIVisionObservationProvider
from twinr.proactive.presence import PresenceSessionController, PresenceSessionSnapshot
from twinr.proactive.vision_review import (
    OpenAIProactiveVisionReviewer,
    ProactiveVisionFrameBuffer,
    ProactiveVisionReview,
    is_reviewable_image_trigger,
)
from twinr.proactive.wakeword import WakewordMatch, WakewordPhraseSpotter
from twinr.providers.openai.backend import OpenAIBackend

if TYPE_CHECKING:
    from twinr.agent.base_agent.runtime import TwinrRuntime

_WAKEWORD_SUPPRESSION_EXEMPT_TRIGGERS = frozenset(
    {"possible_fall", "floor_stillness", "distress_possible"}
)


@dataclass(frozen=True, slots=True)
class ProactiveTickResult:
    decision: SocialTriggerDecision | None = None
    wakeword_match: WakewordMatch | None = None
    inspected: bool = False
    person_visible: bool = False


class ProactiveCoordinator:
    def __init__(
        self,
        *,
        config: TwinrConfig,
        runtime: TwinrRuntime,
        engine: SocialTriggerEngine,
        trigger_handler: Callable[[SocialTriggerDecision], bool],
        vision_observer,
        pir_monitor: GpioPirMonitor | None = None,
        audio_observer=None,
        presence_session: PresenceSessionController | None = None,
        wakeword_spotter: WakewordPhraseSpotter | None = None,
        wakeword_stream: OpenWakeWordStreamingMonitor | None = None,
        vision_reviewer: OpenAIProactiveVisionReviewer | None = None,
        wakeword_handler: Callable[[WakewordMatch], bool] | None = None,
        idle_predicate: Callable[[], bool] | None = None,
        observation_handler: Callable[[dict[str, Any], tuple[str, ...]], None] | None = None,
        emit: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.engine = engine
        self.trigger_handler = trigger_handler
        self.vision_observer = vision_observer
        self.pir_monitor = pir_monitor
        self.audio_observer = audio_observer or NullAudioObservationProvider()
        self.presence_session = presence_session
        self.wakeword_spotter = wakeword_spotter
        self.wakeword_stream = wakeword_stream
        self.vision_reviewer = vision_reviewer
        self.wakeword_handler = wakeword_handler
        self.idle_predicate = idle_predicate
        self.observation_handler = observation_handler
        self.emit = emit or (lambda _line: None)
        self.clock = clock
        self._last_motion_at: float | None = None
        self._last_capture_at: float | None = None
        self._last_observation_key: tuple[object, ...] | None = None
        self._last_sensor_flags: dict[str, bool] = {}
        self._person_visible_since: float | None = None
        self._hand_near_since: float | None = None
        self._speech_detected_since: float | None = None
        self._quiet_since: float | None = None
        self._last_presence_key: tuple[bool, str | None] | None = None
        self._last_wakeword_attempt_at: float | None = None
        self._last_possible_fall_prompted_session_id: int | None = None
        self.latest_presence_snapshot: PresenceSessionSnapshot | None = None

    def tick(self) -> ProactiveTickResult:
        now = self.clock()
        if self.idle_predicate is not None and not self.idle_predicate():
            return ProactiveTickResult()
        motion_active = self._update_motion(now)
        if self.runtime.status.value != "waiting":
            return ProactiveTickResult()
        audio_snapshot = self.audio_observer.observe()
        if not self._should_inspect(now, motion_active=motion_active):
            observation, decision = self._feed_absence(
                now=now,
                motion_active=motion_active,
                audio_observation=audio_snapshot.observation,
            )
            presence_snapshot = self._observe_presence(
                now=now,
                person_visible=None,
                motion_active=motion_active,
                speech_detected=audio_snapshot.observation.speech_detected is True,
            )
            self._dispatch_automation_observation(observation, inspected=False)
            self._record_observation_if_changed(
                observation,
                inspected=False,
                audio_snapshot=audio_snapshot,
                presence_snapshot=presence_snapshot,
            )
            wakeword_match = self._maybe_detect_wakeword(
                now=now,
                audio_snapshot=audio_snapshot,
                presence_snapshot=presence_snapshot,
            )
            if wakeword_match is not None:
                return ProactiveTickResult(
                    wakeword_match=wakeword_match,
                    inspected=False,
                    person_visible=False,
                )
            if decision is not None:
                suppressed_age_s = self._recent_wakeword_attempt_age_s(now, decision)
                if suppressed_age_s is not None:
                    self._record_trigger_skipped_recent_wakeword_attempt(
                        decision,
                        wakeword_attempt_age_s=suppressed_age_s,
                    )
                    return ProactiveTickResult(inspected=False, person_visible=False)
                blocked_reason = self._presence_session_block_reason(
                    decision,
                    presence_snapshot=presence_snapshot,
                )
                if blocked_reason is not None:
                    self._record_trigger_skipped_presence_session(
                        decision,
                        presence_snapshot=presence_snapshot,
                        reason=blocked_reason,
                    )
                    return ProactiveTickResult(inspected=False, person_visible=False)
                reviewed_decision, _review = self._review_trigger(
                    decision,
                    observation=observation,
                )
                if reviewed_decision is None:
                    return ProactiveTickResult(inspected=False, person_visible=False)
                return self._handle_decision(
                    reviewed_decision,
                    observation=observation,
                    inspected=False,
                    presence_snapshot=presence_snapshot,
                    review=_review,
                )
            return ProactiveTickResult(inspected=False, person_visible=False)

        snapshot = self.vision_observer.observe()
        self._last_capture_at = now
        if self.vision_reviewer is not None and getattr(snapshot, "image", None) is not None:
            self.vision_reviewer.record_snapshot(snapshot)
        low_motion = self._is_low_motion(now, motion_active=motion_active)
        observation = SocialObservation(
            observed_at=now,
            inspected=True,
            pir_motion_detected=motion_active,
            low_motion=low_motion,
            vision=snapshot.observation,
            audio=audio_snapshot.observation,
        )
        decision = self.engine.observe(observation)
        presence_snapshot = self._observe_presence(
            now=now,
            person_visible=snapshot.observation.person_visible,
            motion_active=motion_active,
            speech_detected=audio_snapshot.observation.speech_detected is True,
        )
        self._dispatch_automation_observation(observation, inspected=True)
        self._record_observation_if_changed(
            observation,
            inspected=True,
            vision_snapshot=snapshot,
            audio_snapshot=audio_snapshot,
            presence_snapshot=presence_snapshot,
        )
        self.emit(f"proactive_person_visible={str(snapshot.observation.person_visible).lower()}")
        self.emit(
            "proactive_speech_detected="
            f"{str(audio_snapshot.observation.speech_detected).lower()}"
        )
        if self.presence_session is not None:
            self.emit(f"wakeword_armed={str(presence_snapshot.armed).lower()}")
        if audio_snapshot.observation.distress_detected is not None:
            self.emit(
                "proactive_distress_detected="
                f"{str(audio_snapshot.observation.distress_detected).lower()}"
            )
        if audio_snapshot.sample is not None:
            self.emit(f"proactive_audio_peak_rms={audio_snapshot.sample.peak_rms}")
        wakeword_match = self._maybe_detect_wakeword(
            now=now,
            audio_snapshot=audio_snapshot,
            presence_snapshot=presence_snapshot,
        )
        if wakeword_match is not None:
            return ProactiveTickResult(
                wakeword_match=wakeword_match,
                inspected=True,
                person_visible=snapshot.observation.person_visible,
            )
        if decision is not None:
            suppressed_age_s = self._recent_wakeword_attempt_age_s(now, decision)
            if suppressed_age_s is not None:
                self._record_trigger_skipped_recent_wakeword_attempt(
                    decision,
                    wakeword_attempt_age_s=suppressed_age_s,
                )
                return ProactiveTickResult(
                    inspected=True,
                    person_visible=snapshot.observation.person_visible,
                )
            blocked_reason = self._presence_session_block_reason(
                decision,
                presence_snapshot=presence_snapshot,
            )
            if blocked_reason is not None:
                self._record_trigger_skipped_presence_session(
                    decision,
                    presence_snapshot=presence_snapshot,
                    reason=blocked_reason,
                )
                return ProactiveTickResult(
                    inspected=True,
                    person_visible=snapshot.observation.person_visible,
                )
            reviewed_decision, _review = self._review_trigger(
                decision,
                observation=observation,
            )
            if reviewed_decision is None:
                return ProactiveTickResult(
                    inspected=True,
                    person_visible=snapshot.observation.person_visible,
                )
            return self._handle_decision(
                reviewed_decision,
                observation=observation,
                inspected=True,
                presence_snapshot=presence_snapshot,
                review=_review,
            )
        return ProactiveTickResult(
            inspected=True,
            person_visible=snapshot.observation.person_visible,
        )

    def _feed_absence(
        self,
        *,
        now: float,
        motion_active: bool,
        audio_observation: SocialAudioObservation,
    ) -> tuple[SocialObservation, SocialTriggerDecision | None]:
        observation = SocialObservation(
            observed_at=now,
            inspected=False,
            pir_motion_detected=motion_active,
            low_motion=self._is_low_motion(now, motion_active=motion_active),
            vision=SocialVisionObservation(person_visible=False),
            audio=audio_observation,
        )
        return observation, self.engine.observe(observation)

    def _observe_presence(
        self,
        *,
        now: float,
        person_visible: bool | None,
        motion_active: bool,
        speech_detected: bool,
    ) -> PresenceSessionSnapshot:
        if self.presence_session is None:
            snapshot = PresenceSessionSnapshot(
                armed=False,
                reason="disabled",
                person_visible=bool(person_visible),
            )
            self.latest_presence_snapshot = snapshot
            return snapshot
        snapshot = self.presence_session.observe(
            now=now,
            person_visible=person_visible,
            motion_active=motion_active,
            speech_detected=speech_detected,
        )
        self.latest_presence_snapshot = snapshot
        self._record_presence_if_changed(snapshot)
        if self.wakeword_stream is not None:
            self.wakeword_stream.update_presence(snapshot)
        return snapshot

    def poll_wakeword_stream(self) -> WakewordMatch | None:
        if self.wakeword_stream is None or self.wakeword_handler is None:
            return None
        error = self.wakeword_stream.poll_error()
        if error:
            raise RuntimeError(f"openWakeWord stream failed: {error}")
        detection = self.wakeword_stream.poll_detection()
        if detection is None:
            return None
        return self._handle_stream_detection(detection)

    def _maybe_detect_wakeword(
        self,
        *,
        now: float,
        audio_snapshot,
        presence_snapshot: PresenceSessionSnapshot,
    ) -> WakewordMatch | None:
        if self.wakeword_stream is not None:
            return None
        if self.wakeword_spotter is None or self.wakeword_handler is None:
            return None
        if not presence_snapshot.armed:
            return None
        if not self._wakeword_audio_candidate(audio_snapshot):
            return None
        if self._last_wakeword_attempt_at is not None:
            if (now - self._last_wakeword_attempt_at) < self.config.wakeword_attempt_cooldown_s:
                return None
        self._last_wakeword_attempt_at = now
        match = self.wakeword_spotter.detect(
            AmbientAudioCaptureWindow(
                sample=audio_snapshot.sample,
                pcm_bytes=audio_snapshot.pcm_bytes,
                sample_rate=audio_snapshot.sample_rate,
                channels=audio_snapshot.channels,
            )
        )
        self._record_wakeword_attempt(match, presence_snapshot=presence_snapshot)
        if not match.detected:
            return None
        handled = self.wakeword_handler(match)
        if handled:
            self.emit("wakeword_detected=true")
            return match
        return None

    def _handle_stream_detection(self, detection: WakewordStreamDetection) -> WakewordMatch | None:
        self._last_wakeword_attempt_at = self.clock()
        self._record_wakeword_attempt(detection.match, presence_snapshot=detection.presence_snapshot)
        if detection.match.detected and detection.capture_window is not None:
            self._persist_wakeword_capture(detection.match, capture_window=detection.capture_window)
        if not detection.match.detected:
            return None
        handled = self.wakeword_handler(detection.match)
        if handled:
            self.emit("wakeword_detected=true")
            return detection.match
        return None

    def _persist_wakeword_capture(
        self,
        match: WakewordMatch,
        *,
        capture_window: AmbientAudioCaptureWindow,
    ) -> None:
        if not capture_window.pcm_bytes:
            return
        captures_dir = resolve_ops_paths_for_config(self.config).artifacts_root / "ops" / "wakeword_captures"
        captures_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        score_value = 0.0 if match.score is None else max(0.0, float(match.score))
        score_token = f"{score_value:.4f}".replace(".", "_")
        phrase_token = (match.matched_phrase or match.detector_label or "unknown").strip().replace(" ", "_")
        safe_phrase = "".join(char for char in phrase_token if char.isalnum() or char in {"_", "-"}) or "unknown"
        path = captures_dir / f"{timestamp}__score-{score_token}__{safe_phrase}.wav"
        wav_bytes = pcm16_to_wav_bytes(
            capture_window.pcm_bytes,
            sample_rate=capture_window.sample_rate,
            channels=capture_window.channels,
        )
        path.write_bytes(wav_bytes)
        self.runtime.ops_events.append(
            event="wakeword_capture_saved",
            message="Saved the local audio window that triggered a wakeword detection.",
            data={
                "path": str(path),
                "matched_phrase": match.matched_phrase,
                "detector_label": match.detector_label,
                "score": match.score,
                "duration_ms": capture_window.sample.duration_ms,
                "peak_rms": capture_window.sample.peak_rms,
            },
        )

    def _wakeword_audio_candidate(self, audio_snapshot) -> bool:
        sample = getattr(audio_snapshot, "sample", None)
        if sample is None:
            return False
        if not getattr(audio_snapshot, "pcm_bytes", None):
            return False
        if getattr(audio_snapshot, "sample_rate", None) is None:
            return False
        if getattr(audio_snapshot, "channels", None) is None:
            return False
        if self.config.wakeword_backend == "openwakeword":
            if sample.active_chunk_count >= max(1, self.config.wakeword_min_active_chunks):
                return True
            if audio_snapshot.observation.speech_detected is True:
                return True
            peak_threshold = max(1400, int(self.config.audio_speech_threshold * 2))
            return sample.peak_rms >= peak_threshold
        if sample.active_chunk_count < max(1, self.config.wakeword_min_active_chunks):
            return False
        if sample.active_ratio >= self.config.wakeword_min_active_ratio:
            return True
        peak_threshold = max(1400, int(self.config.audio_speech_threshold * 2))
        return sample.peak_rms >= peak_threshold

    def _handle_decision(
        self,
        decision: SocialTriggerDecision,
        *,
        observation: SocialObservation,
        inspected: bool,
        presence_snapshot: PresenceSessionSnapshot,
        review: ProactiveVisionReview | None = None,
    ) -> ProactiveTickResult:
        self._record_trigger_detected(decision, observation=observation, review=review)
        handled = self.trigger_handler(decision)
        presence_session_id = getattr(presence_snapshot, "session_id", None)
        if handled:
            self.emit(f"proactive_trigger={decision.trigger_id}")
            if (
                decision.trigger_id == "possible_fall"
                and self.config.proactive_possible_fall_once_per_presence_session
                and presence_snapshot.armed
                and presence_session_id is not None
            ):
                self._last_possible_fall_prompted_session_id = presence_session_id
        return ProactiveTickResult(
            decision=decision if handled else None,
            inspected=inspected,
            person_visible=observation.vision.person_visible,
        )

    def _review_trigger(
        self,
        decision: SocialTriggerDecision,
        *,
        observation: SocialObservation,
    ) -> tuple[SocialTriggerDecision | None, ProactiveVisionReview | None]:
        if self.vision_reviewer is None:
            return decision, None
        if not is_reviewable_image_trigger(decision.trigger_id):
            return decision, None
        try:
            review = self.vision_reviewer.review(decision, observation=observation)
        except Exception as exc:
            self.emit(f"proactive_vision_review_failed={exc}")
            self.runtime.ops_events.append(
                event="proactive_vision_review_failed",
                level="error",
                message="Buffered proactive vision review failed.",
                data={
                    "trigger": decision.trigger_id,
                    "error": str(exc),
                },
            )
            self._record_trigger_skipped_vision_review_unavailable(decision)
            return None, None
        if review is None:
            self.runtime.ops_events.append(
                event="proactive_vision_review_unavailable",
                message="Buffered proactive vision review was enabled but no usable result was available.",
                data={"trigger": decision.trigger_id},
            )
            self._record_trigger_skipped_vision_review_unavailable(decision)
            return None, None
        self._record_vision_review(decision, review=review)
        if not review.approved:
            self._record_trigger_skipped_vision_review(decision, review=review)
            return None, review
        return decision, review

    def _recent_wakeword_attempt_age_s(
        self,
        now: float,
        decision: SocialTriggerDecision,
    ) -> float | None:
        if decision.trigger_id in _WAKEWORD_SUPPRESSION_EXEMPT_TRIGGERS:
            return None
        if self._last_wakeword_attempt_at is None:
            return None
        age_s = now - self._last_wakeword_attempt_at
        if age_s < 0:
            return None
        if age_s > self.config.wakeword_block_proactive_after_attempt_s:
            return None
        return age_s

    def _update_motion(self, now: float) -> bool:
        if self.pir_monitor is None:
            return False
        motion_active = False
        while True:
            event = self.pir_monitor.poll(timeout=0.0)
            if event is None:
                break
            if event.motion_detected:
                self._last_motion_at = now
                motion_active = True
        try:
            if self.pir_monitor.motion_detected():
                self._last_motion_at = now
                motion_active = True
        except Exception:
            return motion_active
        return motion_active

    def _should_inspect(self, now: float, *, motion_active: bool) -> bool:
        if self.vision_observer is None:
            return False
        if self._last_motion_at is None:
            return False
        if not motion_active and (now - self._last_motion_at) > self.config.proactive_motion_window_s:
            return False
        if self._last_capture_at is not None and (now - self._last_capture_at) < self.config.proactive_capture_interval_s:
            return False
        return True

    def _is_low_motion(self, now: float, *, motion_active: bool) -> bool:
        if motion_active:
            return False
        if self._last_motion_at is None:
            return True
        return (now - self._last_motion_at) >= self.config.proactive_low_motion_after_s

    def _record_observation_if_changed(
        self,
        observation: SocialObservation,
        *,
        inspected: bool,
        vision_snapshot=None,
        audio_snapshot=None,
        presence_snapshot: PresenceSessionSnapshot | None = None,
    ) -> None:
        presence_session_id = None if presence_snapshot is None else getattr(presence_snapshot, "session_id", None)
        observation_key = (
            inspected,
            observation.pir_motion_detected,
            observation.low_motion,
            observation.vision.person_visible,
            observation.vision.looking_toward_device,
            observation.vision.body_pose.value,
            observation.vision.smiling,
            observation.vision.hand_or_object_near_camera,
            observation.audio.speech_detected,
            observation.audio.distress_detected,
            None if presence_snapshot is None else presence_snapshot.armed,
            None if presence_snapshot is None else presence_snapshot.reason,
            presence_session_id,
        )
        if observation_key == self._last_observation_key:
            return
        if not inspected and self._last_observation_key is None:
            self._last_observation_key = observation_key
            return
        self._last_observation_key = observation_key
        data = {
            "inspected": inspected,
            "pir_motion_detected": observation.pir_motion_detected,
            "low_motion": observation.low_motion,
            "person_visible": observation.vision.person_visible,
            "looking_toward_device": observation.vision.looking_toward_device,
            "body_pose": observation.vision.body_pose.value,
            "smiling": observation.vision.smiling,
            "hand_or_object_near_camera": observation.vision.hand_or_object_near_camera,
            "speech_detected": observation.audio.speech_detected,
            "distress_detected": observation.audio.distress_detected,
        }
        if presence_snapshot is not None:
            data.update(
                {
                    "wakeword_armed": presence_snapshot.armed,
                    "wakeword_presence_reason": presence_snapshot.reason,
                    "wakeword_presence_session_id": presence_session_id,
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
        top_evaluation = self.engine.best_evaluation
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
        self.runtime.ops_events.append(
            event="proactive_observation",
            message="Proactive monitor recorded a changed observation.",
            data=data,
        )

    def _record_presence_if_changed(self, snapshot: PresenceSessionSnapshot) -> None:
        session_id = getattr(snapshot, "session_id", None)
        key = (snapshot.armed, snapshot.reason, session_id)
        if key == self._last_presence_key:
            return
        self._last_presence_key = key
        self.runtime.ops_events.append(
            event="wakeword_presence_changed",
            message="Wakeword presence session changed.",
            data={
                "armed": snapshot.armed,
                "reason": snapshot.reason,
                "session_id": session_id,
                "person_visible": snapshot.person_visible,
                "last_person_seen_age_s": snapshot.last_person_seen_age_s,
                "last_motion_age_s": snapshot.last_motion_age_s,
                "last_speech_age_s": snapshot.last_speech_age_s,
            },
        )

    def _record_wakeword_attempt(
        self,
        match: WakewordMatch,
        *,
        presence_snapshot: PresenceSessionSnapshot,
    ) -> None:
        data = {
            "detected": match.detected,
            "matched_phrase": match.matched_phrase,
            "remaining_text": match.remaining_text,
            "presence_reason": presence_snapshot.reason,
            "wakeword_armed": presence_snapshot.armed,
            "backend": match.backend,
            "detector_label": match.detector_label,
            "score": match.score,
        }
        if match.transcript:
            data["transcript_preview"] = match.transcript[:160]
        if match.normalized_transcript:
            data["normalized_transcript_preview"] = match.normalized_transcript[:160]
        self.runtime.ops_events.append(
            event="wakeword_attempted",
            message="Wakeword phrase spotting inspected recent ambient speech.",
            data=data,
        )

    def _record_trigger_detected(
        self,
        decision: SocialTriggerDecision,
        *,
        observation: SocialObservation,
        review: ProactiveVisionReview | None = None,
    ) -> None:
        data = {
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
        }
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
        self.runtime.ops_events.append(
            event="proactive_trigger_detected",
            message="Proactive trigger conditions were met.",
            data=data,
        )

    def _record_vision_review(
        self,
        decision: SocialTriggerDecision,
        *,
        review: ProactiveVisionReview,
    ) -> None:
        self.runtime.ops_events.append(
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
        review: ProactiveVisionReview,
    ) -> None:
        self.emit("social_trigger_skipped=vision_review_rejected")
        self.runtime.ops_events.append(
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
        self.emit("social_trigger_skipped=vision_review_unavailable")
        self.runtime.ops_events.append(
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

    def _record_trigger_skipped_recent_wakeword_attempt(
        self,
        decision: SocialTriggerDecision,
        *,
        wakeword_attempt_age_s: float,
    ) -> None:
        self.emit("social_trigger_skipped=recent_wakeword_attempt")
        self.runtime.ops_events.append(
            event="social_trigger_skipped",
            message="Social trigger prompt was skipped because a wakeword attempt was seen recently.",
            data={
                "trigger": decision.trigger_id,
                "reason": "recent_wakeword_attempt",
                "wakeword_attempt_age_s": round(wakeword_attempt_age_s, 3),
                "priority": int(decision.priority),
                "score": decision.score,
                "threshold": decision.threshold,
            },
        )

    def _presence_session_block_reason(
        self,
        decision: SocialTriggerDecision,
        *,
        presence_snapshot: PresenceSessionSnapshot,
    ) -> str | None:
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
        presence_snapshot: PresenceSessionSnapshot,
        reason: str,
    ) -> None:
        session_id = getattr(presence_snapshot, "session_id", None)
        self.emit(f"social_trigger_skipped={reason}")
        self.runtime.ops_events.append(
            event="social_trigger_skipped",
            message="Social trigger prompt was skipped because it already fired in the current presence session.",
            data={
                "trigger": decision.trigger_id,
                "reason": reason,
                "presence_session_id": session_id,
                "presence_reason": presence_snapshot.reason,
                "priority": int(decision.priority),
                "score": decision.score,
                "threshold": decision.threshold,
            },
        )

    def _dispatch_automation_observation(self, observation: SocialObservation, *, inspected: bool) -> None:
        if self.observation_handler is None:
            return
        facts = self._build_automation_facts(observation, inspected=inspected)
        event_names = self._derive_sensor_events(facts)
        self.observation_handler(facts, event_names)

    def _build_automation_facts(
        self,
        observation: SocialObservation,
        *,
        inspected: bool,
    ) -> dict[str, Any]:
        now = observation.observed_at
        vision = observation.vision
        audio = observation.audio

        self._person_visible_since = self._next_since(vision.person_visible, self._person_visible_since, now)
        self._hand_near_since = self._next_since(
            vision.hand_or_object_near_camera,
            self._hand_near_since,
            now,
        )
        speech_detected = audio.speech_detected is True
        quiet = audio.speech_detected is False
        self._speech_detected_since = self._next_since(speech_detected, self._speech_detected_since, now)
        self._quiet_since = self._next_since(quiet, self._quiet_since, now)

        no_motion_for_s = 0.0
        if not observation.pir_motion_detected and self._last_motion_at is not None:
            no_motion_for_s = max(0.0, now - self._last_motion_at)

        return {
            "sensor": {
                "inspected": inspected,
                "observed_at": now,
            },
            "pir": {
                "motion_detected": observation.pir_motion_detected,
                "low_motion": observation.low_motion,
                "no_motion_for_s": round(no_motion_for_s, 3),
            },
            "camera": {
                "person_visible": vision.person_visible,
                "person_visible_for_s": round(self._duration_since(self._person_visible_since, now), 3),
                "looking_toward_device": vision.looking_toward_device,
                "body_pose": vision.body_pose.value,
                "smiling": vision.smiling,
                "hand_or_object_near_camera": vision.hand_or_object_near_camera,
                "hand_or_object_near_camera_for_s": round(self._duration_since(self._hand_near_since, now), 3),
            },
            "vad": {
                "speech_detected": speech_detected,
                "speech_detected_for_s": round(self._duration_since(self._speech_detected_since, now), 3),
                "quiet": quiet,
                "quiet_for_s": round(self._duration_since(self._quiet_since, now), 3),
                "distress_detected": audio.distress_detected is True,
            },
        }

    def _derive_sensor_events(self, facts: dict[str, Any]) -> tuple[str, ...]:
        current_flags = {
            "pir.motion_detected": bool(facts["pir"]["motion_detected"]),
            "camera.person_visible": bool(facts["camera"]["person_visible"]),
            "camera.hand_or_object_near_camera": bool(facts["camera"]["hand_or_object_near_camera"]),
            "vad.speech_detected": bool(facts["vad"]["speech_detected"]),
        }
        event_names: list[str] = []
        for key, value in current_flags.items():
            previous = self._last_sensor_flags.get(key)
            if value and previous is not True:
                event_names.append(key)
        self._last_sensor_flags = current_flags
        return tuple(event_names)

    def _next_since(self, active: bool, since: float | None, now: float) -> float | None:
        if active:
            return now if since is None else since
        return None

    def _duration_since(self, since: float | None, now: float) -> float:
        if since is None:
            return 0.0
        return max(0.0, now - since)


class ProactiveMonitorService:
    def __init__(
        self,
        coordinator: ProactiveCoordinator,
        *,
        poll_interval_s: float,
        wakeword_stream: OpenWakeWordStreamingMonitor | None = None,
        emit: Callable[[str], None] | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.poll_interval_s = max(0.2, poll_interval_s)
        self.wakeword_stream = wakeword_stream
        self.emit = emit or (lambda _line: None)
        self._stop_event = Event()
        self._thread: Thread | None = None

    def open(self) -> "ProactiveMonitorService":
        if self._thread is not None:
            return self
        self._stop_event.clear()
        self._thread = Thread(target=self._run, daemon=True, name="twinr-proactive")
        self._thread.start()
        self.coordinator.runtime.ops_events.append(
            event="proactive_monitor_started",
            message="Proactive monitor started.",
            data={
                "poll_interval_s": self.poll_interval_s,
                "wakeword_streaming": self.wakeword_stream is not None,
            },
        )
        self.emit("proactive_monitor=started")
        return self

    def close(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop_event.set()
        thread.join()
        self._thread = None
        if self.wakeword_stream is not None:
            self.wakeword_stream.close()
        if self.coordinator.pir_monitor is not None:
            self.coordinator.pir_monitor.close()
        self.coordinator.runtime.ops_events.append(
            event="proactive_monitor_stopped",
            message="Proactive monitor stopped.",
            data={},
        )
        self.emit("proactive_monitor=stopped")

    def __enter__(self) -> "ProactiveMonitorService":
        if self.coordinator.pir_monitor is not None:
            self.coordinator.pir_monitor.open()
        if self.wakeword_stream is not None:
            self.wakeword_stream.open()
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _run(self) -> None:
        next_tick_at = 0.0
        while not self._stop_event.is_set():
            did_work = False
            try:
                while self.coordinator.poll_wakeword_stream() is not None:
                    did_work = True
            except Exception as exc:
                self.emit(f"proactive_error={exc}")
                self.coordinator.runtime.ops_events.append(
                    event="proactive_error",
                    level="error",
                    message="Proactive wakeword stream handling failed.",
                    data={"error": str(exc)},
                )
            now = time.monotonic()
            if now >= next_tick_at:
                try:
                    self.coordinator.tick()
                    did_work = True
                except Exception as exc:
                    self.emit(f"proactive_error={exc}")
                    self.coordinator.runtime.ops_events.append(
                        event="proactive_error",
                        level="error",
                        message="Proactive monitor tick failed.",
                        data={"error": str(exc)},
                    )
                next_tick_at = time.monotonic() + self.poll_interval_s
            wait_s = max(0.02, min(0.05, max(0.0, next_tick_at - time.monotonic())))
            if did_work:
                wait_s = 0.02
            if self._stop_event.wait(wait_s):
                return


def build_default_proactive_monitor(
    *,
    config: TwinrConfig,
    runtime: TwinrRuntime,
    backend: OpenAIBackend,
    camera: V4L2StillCamera,
    camera_lock: Lock | None,
    audio_lock: Lock | None,
    trigger_handler: Callable[[SocialTriggerDecision], bool],
    wakeword_handler: Callable[[WakewordMatch], bool] | None = None,
    idle_predicate: Callable[[], bool] | None = None,
    observation_handler: Callable[[dict[str, Any], tuple[str, ...]], None] | None = None,
    emit: Callable[[str], None] | None = None,
) -> ProactiveMonitorService | None:
    if not config.proactive_enabled and not config.wakeword_enabled:
        return None
    if not config.pir_enabled:
        if emit is not None:
            emit("proactive_disabled=pir_unconfigured")
        return None

    engine = SocialTriggerEngine.from_config(config)
    use_openwakeword_stream = (
        config.wakeword_enabled
        and (config.wakeword_backend or "stt").strip().lower() == "openwakeword"
    )
    presence_session = None
    wakeword_spotter = None
    wakeword_stream = None
    if config.wakeword_enabled:
        presence_session = PresenceSessionController(
            presence_grace_s=config.wakeword_presence_grace_s,
            motion_grace_s=config.wakeword_motion_grace_s,
            speech_grace_s=config.wakeword_speech_grace_s,
        )
        if use_openwakeword_stream:
            wakeword_stream = _build_openwakeword_stream(
                config=config,
                runtime=runtime,
                emit=emit,
            )
            if wakeword_stream is None:
                wakeword_spotter = _build_wakeword_spotter(
                    config=config,
                    runtime=runtime,
                    backend=backend,
                    emit=emit,
                    selected_backend="stt",
                )
        else:
            wakeword_spotter = _build_wakeword_spotter(
                config=config,
                runtime=runtime,
                backend=backend,
                emit=emit,
            )
    audio_observer = NullAudioObservationProvider()
    if config.proactive_audio_enabled or config.wakeword_enabled:
        sampler = wakeword_stream if wakeword_stream is not None else AmbientAudioSampler.from_config(config)
        audio_observer = AmbientAudioObservationProvider(
            sampler=sampler,
            audio_lock=None if wakeword_stream is not None else audio_lock,
            sample_ms=config.proactive_audio_sample_ms,
            distress_enabled=config.proactive_audio_distress_enabled,
        )
    vision_reviewer = None
    if config.proactive_vision_review_enabled:
        vision_reviewer = OpenAIProactiveVisionReviewer(
            backend=backend,
            frame_buffer=ProactiveVisionFrameBuffer(
                max_items=config.proactive_vision_review_buffer_frames,
            ),
            max_frames=config.proactive_vision_review_max_frames,
            max_age_s=config.proactive_vision_review_max_age_s,
            min_spacing_s=config.proactive_vision_review_min_spacing_s,
        )
    coordinator = ProactiveCoordinator(
        config=config,
        runtime=runtime,
        engine=engine,
        trigger_handler=trigger_handler,
        vision_observer=OpenAIVisionObservationProvider(
            backend=backend,
            camera=camera,
            camera_lock=camera_lock,
        ),
        audio_observer=audio_observer,
        presence_session=presence_session,
        wakeword_spotter=wakeword_spotter,
        wakeword_stream=wakeword_stream,
        vision_reviewer=vision_reviewer,
        wakeword_handler=wakeword_handler,
        pir_monitor=configured_pir_monitor(config),
        idle_predicate=idle_predicate,
        observation_handler=observation_handler,
        emit=emit,
    )
    return ProactiveMonitorService(
        coordinator,
        poll_interval_s=config.proactive_poll_interval_s,
        wakeword_stream=wakeword_stream,
        emit=emit,
    )


def _build_openwakeword_stream(
    *,
    config: TwinrConfig,
    runtime: TwinrRuntime,
    emit: Callable[[str], None] | None,
) -> OpenWakeWordStreamingMonitor | None:
    models = tuple(model for model in config.wakeword_openwakeword_models if str(model).strip())
    if config.audio_sample_rate != 16000 or config.audio_channels != 1:
        _record_wakeword_backend_warning(
            runtime=runtime,
            emit=emit,
            reason="openwakeword_requires_16khz_mono",
            detail=(
                "openWakeWord needs 16 kHz mono audio. Configure "
                "TWINR_AUDIO_SAMPLE_RATE=16000 and TWINR_AUDIO_CHANNELS=1."
            ),
        )
        return None
    if not models:
        _record_wakeword_backend_warning(
            runtime=runtime,
            emit=emit,
            reason="openwakeword_models_missing",
            detail=(
                "openWakeWord backend selected, but no models were configured. "
                "Set TWINR_WAKEWORD_OPENWAKEWORD_MODELS to model names or file paths."
            ),
        )
        return None
    try:
        from twinr.proactive.openwakeword_spotter import WakewordOpenWakeWordFrameSpotter

        return OpenWakeWordStreamingMonitor(
            device=(config.proactive_audio_input_device or config.audio_input_device).strip(),
            sample_rate=config.audio_sample_rate,
            channels=config.audio_channels,
            spotter=WakewordOpenWakeWordFrameSpotter(
                wakeword_models=models,
                phrases=config.wakeword_phrases,
                threshold=config.wakeword_openwakeword_threshold,
                vad_threshold=config.wakeword_openwakeword_vad_threshold,
                patience_frames=config.wakeword_openwakeword_patience_frames,
                activation_samples=config.wakeword_openwakeword_activation_samples,
                deactivation_threshold=config.wakeword_openwakeword_deactivation_threshold,
                enable_speex_noise_suppression=config.wakeword_openwakeword_enable_speex,
                inference_framework=config.wakeword_openwakeword_inference_framework,
            ),
            attempt_cooldown_s=config.wakeword_attempt_cooldown_s,
            speech_threshold=config.audio_speech_threshold,
            emit=emit,
        )
    except Exception as exc:
        _record_wakeword_backend_warning(
            runtime=runtime,
            emit=emit,
            reason="openwakeword_init_failed",
            detail=f"openWakeWord streaming initialization failed: {exc}",
        )
        return None


def _build_wakeword_spotter(
    *,
    config: TwinrConfig,
    runtime: TwinrRuntime,
    backend: OpenAIBackend,
    emit: Callable[[str], None] | None,
    selected_backend: str | None = None,
):
    selected_backend = (
        selected_backend
        if selected_backend is not None
        else (config.wakeword_backend or "stt").strip().lower() or "stt"
    )
    if selected_backend == "openwakeword":
        models = tuple(model for model in config.wakeword_openwakeword_models if str(model).strip())
        if config.audio_sample_rate != 16000 or config.audio_channels != 1:
            _record_wakeword_backend_warning(
                runtime=runtime,
                emit=emit,
                reason="openwakeword_requires_16khz_mono",
                detail=(
                    "openWakeWord needs 16 kHz mono audio. Falling back to STT wakeword spotting "
                    "until TWINR_AUDIO_SAMPLE_RATE=16000 and TWINR_AUDIO_CHANNELS=1 are set."
                ),
            )
        elif not models:
            _record_wakeword_backend_warning(
                runtime=runtime,
                emit=emit,
                reason="openwakeword_models_missing",
                detail=(
                    "openWakeWord backend selected, but no models were configured. "
                    "Set TWINR_WAKEWORD_OPENWAKEWORD_MODELS to model names or file paths."
                ),
            )
        else:
            try:
                return WakewordOpenWakeWordSpotter(
                    wakeword_models=models,
                    phrases=config.wakeword_phrases,
                    threshold=config.wakeword_openwakeword_threshold,
                    vad_threshold=config.wakeword_openwakeword_vad_threshold,
                    patience_frames=config.wakeword_openwakeword_patience_frames,
                    activation_samples=config.wakeword_openwakeword_activation_samples,
                    deactivation_threshold=config.wakeword_openwakeword_deactivation_threshold,
                    enable_speex_noise_suppression=config.wakeword_openwakeword_enable_speex,
                    inference_framework=config.wakeword_openwakeword_inference_framework,
                    backend=backend,
                    language=config.openai_realtime_language,
                    transcribe_on_detect=config.wakeword_openwakeword_transcribe_on_detect,
                )
            except Exception as exc:
                _record_wakeword_backend_warning(
                    runtime=runtime,
                    emit=emit,
                    reason="openwakeword_init_failed",
                    detail=f"openWakeWord initialization failed: {exc}",
                )

    return WakewordPhraseSpotter(
        backend=backend,
        phrases=config.wakeword_phrases,
        language=config.openai_realtime_language,
    )


def _record_wakeword_backend_warning(
    *,
    runtime: TwinrRuntime,
    emit: Callable[[str], None] | None,
    reason: str,
    detail: str,
) -> None:
    if emit is not None:
        emit(f"wakeword_backend_warning={reason}")
    runtime.ops_events.append(
        event="wakeword_backend_warning",
        level="warning",
        message="Wakeword backend configuration fell back to the STT matcher.",
        data={
            "reason": reason,
            "detail": detail,
            "fallback_backend": "stt",
        },
    )


__all__ = [
    "ProactiveCoordinator",
    "ProactiveMonitorService",
    "ProactiveTickResult",
    "build_default_proactive_monitor",
]
