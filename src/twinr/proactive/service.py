from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Any, Callable
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.runtime import TwinrRuntime
from twinr.hardware import GpioPirMonitor, V4L2StillCamera, configured_pir_monitor
from twinr.hardware.audio import AmbientAudioSampler
from twinr.proactive.engine import SocialAudioObservation, SocialObservation, SocialTriggerDecision, SocialTriggerEngine, SocialVisionObservation
from twinr.proactive.observers import AmbientAudioObservationProvider, NullAudioObservationProvider, OpenAIVisionObservationProvider
from twinr.providers.openai.backend import OpenAIBackend


@dataclass(frozen=True, slots=True)
class ProactiveTickResult:
    decision: SocialTriggerDecision | None = None
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
            self._dispatch_automation_observation(observation, inspected=False)
            self._record_observation_if_changed(
                observation,
                inspected=False,
                audio_snapshot=audio_snapshot,
            )
            if decision is not None:
                return self._handle_decision(
                    decision,
                    observation=observation,
                    inspected=False,
                )
            return ProactiveTickResult(inspected=False, person_visible=False)

        snapshot = self.vision_observer.observe()
        self._last_capture_at = now
        low_motion = self._is_low_motion(now, motion_active=motion_active)
        observation = SocialObservation(
            observed_at=now,
            pir_motion_detected=motion_active,
            low_motion=low_motion,
            vision=snapshot.observation,
            audio=audio_snapshot.observation,
        )
        decision = self.engine.observe(observation)
        self._dispatch_automation_observation(observation, inspected=True)
        self._record_observation_if_changed(
            observation,
            inspected=True,
            vision_snapshot=snapshot,
            audio_snapshot=audio_snapshot,
        )
        self.emit(f"proactive_person_visible={str(snapshot.observation.person_visible).lower()}")
        self.emit(
            "proactive_speech_detected="
            f"{str(audio_snapshot.observation.speech_detected).lower()}"
        )
        if audio_snapshot.observation.distress_detected is not None:
            self.emit(
                "proactive_distress_detected="
                f"{str(audio_snapshot.observation.distress_detected).lower()}"
            )
        if audio_snapshot.sample is not None:
            self.emit(f"proactive_audio_peak_rms={audio_snapshot.sample.peak_rms}")
        if decision is not None:
            return self._handle_decision(
                decision,
                observation=observation,
                inspected=True,
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
            pir_motion_detected=motion_active,
            low_motion=self._is_low_motion(now, motion_active=motion_active),
            vision=SocialVisionObservation(person_visible=False),
            audio=audio_observation,
        )
        return observation, self.engine.observe(observation)

    def _handle_decision(
        self,
        decision: SocialTriggerDecision,
        *,
        observation: SocialObservation,
        inspected: bool,
    ) -> ProactiveTickResult:
        self._record_trigger_detected(decision, observation=observation)
        handled = self.trigger_handler(decision)
        if handled:
            self.emit(f"proactive_trigger={decision.trigger_id}")
        return ProactiveTickResult(
            decision=decision if handled else None,
            inspected=inspected,
            person_visible=observation.vision.person_visible,
        )

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
    ) -> None:
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

    def _record_trigger_detected(
        self,
        decision: SocialTriggerDecision,
        *,
        observation: SocialObservation,
    ) -> None:
        self.runtime.ops_events.append(
            event="proactive_trigger_detected",
            message="Proactive trigger conditions were met.",
            data={
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
        emit: Callable[[str], None] | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.poll_interval_s = max(0.2, poll_interval_s)
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
            data={"poll_interval_s": self.poll_interval_s},
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
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.coordinator.tick()
            except Exception as exc:
                self.emit(f"proactive_error={exc}")
                self.coordinator.runtime.ops_events.append(
                    event="proactive_error",
                    level="error",
                    message="Proactive monitor tick failed.",
                    data={"error": str(exc)},
                )
            if self._stop_event.wait(self.poll_interval_s):
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
    idle_predicate: Callable[[], bool] | None = None,
    observation_handler: Callable[[dict[str, Any], tuple[str, ...]], None] | None = None,
    emit: Callable[[str], None] | None = None,
) -> ProactiveMonitorService | None:
    if not config.proactive_enabled:
        return None
    if not config.pir_enabled:
        if emit is not None:
            emit("proactive_disabled=pir_unconfigured")
        return None

    engine = SocialTriggerEngine.from_config(config)
    audio_observer = NullAudioObservationProvider()
    if config.proactive_audio_enabled:
        audio_observer = AmbientAudioObservationProvider(
            sampler=AmbientAudioSampler.from_config(config),
            audio_lock=audio_lock,
            sample_ms=config.proactive_audio_sample_ms,
            distress_enabled=config.proactive_audio_distress_enabled,
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
        pir_monitor=configured_pir_monitor(config),
        idle_predicate=idle_predicate,
        observation_handler=observation_handler,
        emit=emit,
    )
    return ProactiveMonitorService(
        coordinator,
        poll_interval_s=config.proactive_poll_interval_s,
        emit=emit,
    )


__all__ = [
    "ProactiveCoordinator",
    "ProactiveMonitorService",
    "ProactiveTickResult",
    "build_default_proactive_monitor",
]
