from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.hardware.respeaker.models import ReSpeakerSignalSnapshot
from twinr.hardware.audio import AmbientAudioLevelSample
from twinr.proactive import (
    AmbientAudioObservationProvider,
    OpenAIVisionObservationProvider,
    PresenceSessionController,
    ProactiveAudioSnapshot,
    ProactiveCoordinator,
    PresenceSessionSnapshot,
    SocialAudioObservation,
    SocialBodyPose,
    SocialObservation,
    SocialPersonZone,
    SocialTriggerDecision,
    SocialTriggerEngine,
    SocialTriggerPriority,
    SocialVisionObservation,
    WakewordMatch,
    WakewordStreamDetection,
    WakewordPhraseSpotter,
    build_default_proactive_monitor,
    parse_vision_observation_text,
)
from twinr.proactive.social.observers import ReSpeakerAudioObservationProvider
from twinr.proactive.social.vision_review import ProactiveVisionReview
from twinr.proactive.runtime.audio_policy import ReSpeakerAudioPolicySnapshot
from twinr.runtime import TwinrRuntime


class FakeVisionObserver:
    def __init__(self, observations):
        self.observations = list(observations)
        self.calls = 0

    def observe(self):
        self.calls += 1
        observation = self.observations.pop(0)
        return SimpleNamespace(
            observation=observation,
            response_text="ok",
            captured_at=None,
            image=None,
            source_device=None,
            input_format=None,
            response_id="resp_1",
            request_id="req_1",
            model="gpt-5.2",
        )


class FakePirMonitor:
    def __init__(self, *, events=None, level=False) -> None:
        self.events = list(events or [])
        self.level = level
        self.opened = False
        self.closed = False

    def open(self):
        self.opened = True
        return self

    def close(self):
        self.closed = True

    def poll(self, timeout=None):
        if not self.events:
            return None
        motion = self.events.pop(0)
        return SimpleNamespace(motion_detected=motion)

    def motion_detected(self):
        return self.level


class FakeBackend:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = []

    def respond_to_images_with_metadata(
        self,
        prompt,
        *,
        images,
        conversation=None,
        instructions=None,
        allow_web_search=None,
    ):
        self.calls.append((prompt, list(images), conversation, instructions, allow_web_search))
        return SimpleNamespace(
            text=self.text,
            response_id="resp_vision",
            request_id="req_vision",
            model="gpt-5.2",
        )


class FakeCamera:
    def capture_photo(self, *, filename: str = "x.png", output_path=None):
        return SimpleNamespace(
            data=b"\x89PNG\r\n\x1a\ncamera",
            content_type="image/png",
            filename=filename,
            source_device="/dev/video0",
            input_format="bayer_grbg8",
        )


class FakeAudioSampler:
    def __init__(self, sample: AmbientAudioLevelSample) -> None:
        self.sample = sample
        self.calls = 0
        self.durations: list[int] = []

    def sample_levels(self, *, duration_ms=None):
        self.calls += 1
        self.durations.append(duration_ms)
        return self.sample


class FakeAudioObserver:
    def __init__(self, observation: SocialAudioObservation, *, sample: AmbientAudioLevelSample | None = None) -> None:
        self.snapshot = ProactiveAudioSnapshot(observation=observation, sample=sample)
        self.calls = 0

    def observe(self):
        self.calls += 1
        return self.snapshot


class FakeReSpeakerSignalProvider:
    def __init__(self, snapshot: ReSpeakerSignalSnapshot) -> None:
        self.snapshot = snapshot
        self.calls = 0
        self.closed = False

    def observe(self) -> ReSpeakerSignalSnapshot:
        self.calls += 1
        return self.snapshot

    def close(self) -> None:
        self.closed = True


class FakeWakewordStream:
    def __init__(self, detections=None) -> None:
        self.detections = list(detections or [])
        self.errors: list[str] = []
        self.presence_snapshots: list[PresenceSessionSnapshot] = []
        self.opened = False
        self.closed = False

    def open(self):
        self.opened = True
        return self

    def close(self):
        self.closed = True

    def update_presence(self, snapshot: PresenceSessionSnapshot) -> None:
        self.presence_snapshots.append(snapshot)

    def poll_detection(self):
        if not self.detections:
            return None
        return self.detections.pop(0)

    def poll_error(self):
        if not self.errors:
            return None
        return self.errors.pop(0)


class FakeVisionReviewer:
    def __init__(self, review: ProactiveVisionReview | None) -> None:
        self.review_result = review
        self.recorded_snapshots = []
        self.calls = []

    def record_snapshot(self, snapshot) -> None:
        self.recorded_snapshots.append(snapshot)

    def review(self, decision, *, observation):
        self.calls.append((decision, observation))
        return self.review_result


class MutableClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class ProactiveMonitorTests(unittest.TestCase):
    def test_parse_vision_observation_text(self) -> None:
        observation = parse_vision_observation_text(
            "\n".join(
                [
                    "person_visible=yes",
                    "person_count=2",
                    "primary_person_zone=right",
                    "looking_toward_device=yes",
                    "body_pose=slumped",
                    "smiling=no",
                    "hand_or_object_near_camera=yes",
                ]
            )
        )

        self.assertTrue(observation.person_visible)
        self.assertEqual(observation.person_count, 2)
        self.assertEqual(observation.primary_person_zone, SocialPersonZone.RIGHT)
        self.assertTrue(observation.looking_toward_device)
        self.assertEqual(observation.body_pose, SocialBodyPose.SLUMPED)
        self.assertFalse(observation.smiling)
        self.assertTrue(observation.hand_or_object_near_camera)

    def test_parse_vision_observation_text_defaults_visible_person_to_one_person_unknown_zone(self) -> None:
        observation = parse_vision_observation_text(
            "\n".join(
                [
                    "person_visible=yes",
                    "looking_toward_device=no",
                    "body_pose=upright",
                    "smiling=no",
                    "hand_or_object_near_camera=no",
                ]
            )
        )

        self.assertTrue(observation.person_visible)
        self.assertEqual(observation.person_count, 1)
        self.assertEqual(observation.primary_person_zone, SocialPersonZone.UNKNOWN)

    def test_parse_vision_observation_text_forces_hidden_person_to_zero_count_and_unknown_zone(self) -> None:
        observation = parse_vision_observation_text(
            "\n".join(
                [
                    "person_visible=no",
                    "person_count=2",
                    "primary_person_zone=left",
                    "looking_toward_device=no",
                    "body_pose=unknown",
                    "smiling=no",
                    "hand_or_object_near_camera=no",
                ]
            )
        )

        self.assertFalse(observation.person_visible)
        self.assertEqual(observation.person_count, 0)
        self.assertEqual(observation.primary_person_zone, SocialPersonZone.UNKNOWN)

    def test_openai_vision_observer_parses_backend_response(self) -> None:
        backend = FakeBackend(
            "\n".join(
                [
                    "person_visible=yes",
                    "person_count=2",
                    "primary_person_zone=center",
                    "looking_toward_device=no",
                    "body_pose=upright",
                    "smiling=yes",
                    "hand_or_object_near_camera=no",
                ]
            )
        )
        observer = OpenAIVisionObservationProvider(
            backend=backend,
            camera=FakeCamera(),
        )

        snapshot = observer.observe()

        self.assertEqual(len(backend.calls), 1)
        self.assertIn("person_count=0|1|2|...", backend.calls[0][0])
        self.assertIn("primary_person_zone=left|center|right|unknown", backend.calls[0][0])
        self.assertTrue(snapshot.observation.person_visible)
        self.assertEqual(snapshot.observation.person_count, 2)
        self.assertEqual(snapshot.observation.primary_person_zone, SocialPersonZone.CENTER)
        self.assertEqual(snapshot.observation.body_pose, SocialBodyPose.UPRIGHT)
        self.assertTrue(snapshot.observation.smiling)

    def test_ambient_audio_observer_maps_levels_to_audio_observation(self) -> None:
        observer = AmbientAudioObservationProvider(
            sampler=FakeAudioSampler(
                AmbientAudioLevelSample(
                    duration_ms=1000,
                    chunk_count=5,
                    active_chunk_count=3,
                    average_rms=980,
                    peak_rms=2200,
                    active_ratio=0.6,
                )
            ),
            sample_ms=900,
            distress_enabled=True,
        )

        snapshot = observer.observe()

        self.assertTrue(snapshot.observation.speech_detected)
        self.assertTrue(snapshot.observation.distress_detected)
        self.assertIsNotNone(snapshot.sample)
        self.assertEqual(snapshot.sample.peak_rms, 2200)

    def test_ambient_audio_observer_treats_short_loud_speech_like_window_as_speech(self) -> None:
        observer = AmbientAudioObservationProvider(
            sampler=FakeAudioSampler(
                AmbientAudioLevelSample(
                    duration_ms=2600,
                    chunk_count=26,
                    active_chunk_count=3,
                    average_rms=462,
                    peak_rms=5336,
                    active_ratio=3 / 26,
                )
            ),
            sample_ms=2600,
            distress_enabled=False,
        )

        snapshot = observer.observe()

        self.assertTrue(snapshot.observation.speech_detected)
        self.assertFalse(snapshot.observation.distress_detected)

    def test_respeaker_audio_observer_overlays_signal_fields_on_fallback_audio(self) -> None:
        signal_provider = FakeReSpeakerSignalProvider(
            ReSpeakerSignalSnapshot(
                captured_at=10.0,
                source="respeaker_xvf3800",
                source_type="observed",
                sensor_window_ms=1000,
                device_runtime_mode="audio_ready",
                host_control_ready=True,
                speech_detected=True,
                room_quiet=False,
                recent_speech_age_s=0.0,
                assistant_output_active=False,
                azimuth_deg=277,
                direction_confidence=0.88,
                speech_overlap_likely=False,
                barge_in_detected=False,
            )
        )
        fallback = FakeAudioObserver(
            SocialAudioObservation(
                speech_detected=False,
                distress_detected=True,
            ),
            sample=AmbientAudioLevelSample(
                duration_ms=1000,
                chunk_count=5,
                active_chunk_count=3,
                average_rms=980,
                peak_rms=2200,
                active_ratio=0.6,
            ),
        )
        observer = ReSpeakerAudioObservationProvider(
            signal_provider=signal_provider,
            fallback_observer=fallback,
        )

        snapshot = observer.observe()

        self.assertTrue(snapshot.observation.speech_detected)
        self.assertTrue(snapshot.observation.distress_detected)
        self.assertEqual(snapshot.observation.azimuth_deg, 277)
        self.assertEqual(snapshot.observation.direction_confidence, 0.88)
        self.assertEqual(snapshot.observation.device_runtime_mode, "audio_ready")
        self.assertEqual(snapshot.observation.signal_source, "respeaker_xvf3800")
        self.assertTrue(snapshot.observation.host_control_ready)
        self.assertIsNotNone(snapshot.sample)
        self.assertIs(snapshot.signal_snapshot, signal_provider.snapshot)

    def test_respeaker_audio_observer_skips_fallback_sampling_while_assistant_output_is_active(self) -> None:
        signal_provider = FakeReSpeakerSignalProvider(
            ReSpeakerSignalSnapshot(
                captured_at=10.0,
                source="respeaker_xvf3800",
                source_type="observed",
                sensor_window_ms=1000,
                device_runtime_mode="audio_ready",
                host_control_ready=True,
                speech_detected=True,
                room_quiet=False,
                recent_speech_age_s=0.0,
                assistant_output_active=True,
                azimuth_deg=92,
                direction_confidence=0.97,
                speech_overlap_likely=False,
                barge_in_detected=True,
            )
        )
        fallback = FakeAudioObserver(
            SocialAudioObservation(
                speech_detected=False,
                distress_detected=True,
            )
        )
        observer = ReSpeakerAudioObservationProvider(
            signal_provider=signal_provider,
            fallback_observer=fallback,
        )

        snapshot = observer.observe()

        self.assertEqual(fallback.calls, 0)
        self.assertTrue(snapshot.observation.assistant_output_active)
        self.assertTrue(snapshot.observation.barge_in_detected)
        self.assertIsNone(snapshot.sample)

    def test_coordinator_triggers_person_returned_after_absence(self) -> None:
        config = TwinrConfig(
            proactive_enabled=True,
            proactive_capture_interval_s=1.0,
            proactive_motion_window_s=20.0,
        )
        runtime = TwinrRuntime(config=config)
        clock = MutableClock(0.0)
        pir_monitor = FakePirMonitor(events=[True], level=True)
        handled: list[str] = []
        coordinator = ProactiveCoordinator(
            config=config,
            runtime=runtime,
            engine=SocialTriggerEngine(user_name="Thom"),
            trigger_handler=lambda decision: handled.append(decision.trigger_id) or True,
            vision_observer=FakeVisionObserver(
                [
                    SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.UPRIGHT),
                    SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.UPRIGHT),
                ]
            ),
            pir_monitor=pir_monitor,
            emit=lambda _line: None,
            clock=clock,
        )

        coordinator.tick()
        clock.now = 30.0
        pir_monitor.level = False
        absence_result = coordinator.tick()
        self.assertFalse(absence_result.inspected)

        clock.now = 21.0 * 60.0
        pir_monitor.events = [True]
        pir_monitor.level = True
        result = coordinator.tick()

        self.assertEqual(handled, ["person_returned"])
        self.assertIsNotNone(result.decision)
        self.assertEqual(result.decision.trigger_id, "person_returned")

    def test_coordinator_updates_wakeword_stream_presence(self) -> None:
        config = TwinrConfig(
            wakeword_enabled=True,
            proactive_capture_interval_s=1.0,
            proactive_motion_window_s=20.0,
        )
        runtime = TwinrRuntime(config=config)
        wakeword_stream = FakeWakewordStream()
        coordinator = ProactiveCoordinator(
            config=config,
            runtime=runtime,
            engine=SocialTriggerEngine(),
            trigger_handler=lambda _decision: True,
            vision_observer=FakeVisionObserver(
                [SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.UPRIGHT)]
            ),
            pir_monitor=FakePirMonitor(events=[True], level=True),
            audio_observer=FakeAudioObserver(SocialAudioObservation(speech_detected=False)),
            presence_session=PresenceSessionController(
                presence_grace_s=600.0,
                motion_grace_s=120.0,
                speech_grace_s=45.0,
            ),
            wakeword_stream=wakeword_stream,
            emit=lambda _line: None,
            clock=MutableClock(0.0),
        )

        coordinator.tick()

        self.assertTrue(wakeword_stream.presence_snapshots)
        self.assertTrue(wakeword_stream.presence_snapshots[-1].armed)
        self.assertEqual(wakeword_stream.presence_snapshots[-1].reason, "pir_motion")

    def test_coordinator_handles_streaming_wakeword_detection(self) -> None:
        config = TwinrConfig(
            wakeword_enabled=True,
            proactive_capture_interval_s=1.0,
            proactive_motion_window_s=20.0,
        )
        runtime = TwinrRuntime(config=config)
        handled: list[str | None] = []
        detection = WakewordStreamDetection(
            match=WakewordMatch(
                detected=True,
                transcript="",
                matched_phrase="twinna",
                backend="openwakeword",
                detector_label="twinr_multivoice_v2",
                score=0.82,
            ),
            presence_snapshot=PresenceSessionSnapshot(
                armed=True,
                reason="person_visible",
                person_visible=True,
                session_id=1,
            ),
        )
        coordinator = ProactiveCoordinator(
            config=config,
            runtime=runtime,
            engine=SocialTriggerEngine(),
            trigger_handler=lambda _decision: True,
            vision_observer=FakeVisionObserver(
                [SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.UPRIGHT)]
            ),
            pir_monitor=FakePirMonitor(level=False),
            audio_observer=FakeAudioObserver(SocialAudioObservation(speech_detected=False)),
            wakeword_stream=FakeWakewordStream([detection]),
            wakeword_handler=lambda match: handled.append(match.matched_phrase) or True,
            emit=lambda _line: None,
            clock=MutableClock(0.0),
        )

        match = coordinator.poll_wakeword_stream()

        self.assertIsNotNone(match)
        self.assertEqual(match.matched_phrase, "twinna")
        self.assertEqual(handled, ["twinna"])
        attempted = [entry for entry in runtime.ops_events.tail(limit=10) if entry.get("event") == "wakeword_attempted"]
        self.assertTrue(attempted)
        self.assertEqual(attempted[-1]["data"]["matched_phrase"], "twinna")

    def test_openwakeword_backend_requires_speech_like_audio_window(self) -> None:
        config = TwinrConfig(
            wakeword_enabled=True,
            wakeword_backend="openwakeword",
            wakeword_openwakeword_models=("hey_twinna.onnx",),
        )
        runtime = TwinrRuntime(config=config)
        coordinator = ProactiveCoordinator(
            config=config,
            runtime=runtime,
            engine=SocialTriggerEngine(user_name="Thom"),
            trigger_handler=lambda _decision: False,
            vision_observer=FakeVisionObserver([]),
            emit=lambda _line: None,
        )
        quiet_snapshot = ProactiveAudioSnapshot(
            observation=SocialAudioObservation(speech_detected=False),
            sample=AmbientAudioLevelSample(
                duration_ms=2600,
                chunk_count=26,
                active_chunk_count=0,
                average_rms=133,
                peak_rms=241,
                active_ratio=0.0,
            ),
            pcm_bytes=b"\x00\x01" * 128,
            sample_rate=16000,
            channels=1,
        )
        speechy_snapshot = ProactiveAudioSnapshot(
            observation=SocialAudioObservation(speech_detected=True),
            sample=AmbientAudioLevelSample(
                duration_ms=2600,
                chunk_count=26,
                active_chunk_count=3,
                average_rms=478,
                peak_rms=3556,
                active_ratio=3 / 26,
            ),
            pcm_bytes=b"\x00\x01" * 128,
            sample_rate=16000,
            channels=1,
        )

        self.assertFalse(coordinator._wakeword_audio_candidate(quiet_snapshot))
        self.assertTrue(coordinator._wakeword_audio_candidate(speechy_snapshot))

    def test_coordinator_pauses_completely_when_idle_predicate_is_false(self) -> None:
        config = TwinrConfig(
            proactive_enabled=True,
            proactive_capture_interval_s=1.0,
            proactive_motion_window_s=20.0,
        )
        runtime = TwinrRuntime(config=config)
        vision_observer = FakeVisionObserver(
            [SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.UPRIGHT)]
        )
        audio_observer = FakeAudioObserver(SocialAudioObservation(speech_detected=True))
        handled: list[str] = []
        coordinator = ProactiveCoordinator(
            config=config,
            runtime=runtime,
            engine=SocialTriggerEngine(user_name="Thom"),
            trigger_handler=lambda decision: handled.append(decision.trigger_id) or True,
            vision_observer=vision_observer,
            pir_monitor=FakePirMonitor(events=[True], level=True),
            audio_observer=audio_observer,
            idle_predicate=lambda: False,
            emit=lambda _line: None,
            clock=MutableClock(5.0),
        )

        result = coordinator.tick()

        self.assertFalse(result.inspected)

    def test_coordinator_observes_audio_while_runtime_is_answering(self) -> None:
        config = TwinrConfig(
            proactive_enabled=True,
            proactive_capture_interval_s=1.0,
            proactive_motion_window_s=20.0,
        )
        runtime = TwinrRuntime(config=config)
        runtime.begin_proactive_prompt("Hallo")
        audio_observer = FakeAudioObserver(
            SocialAudioObservation(
                speech_detected=True,
                assistant_output_active=True,
                barge_in_detected=True,
            )
        )
        coordinator = ProactiveCoordinator(
            config=config,
            runtime=runtime,
            engine=SocialTriggerEngine(),
            trigger_handler=lambda _decision: True,
            vision_observer=None,
            audio_observer=audio_observer,
            pir_monitor=FakePirMonitor(events=[False], level=False),
            emit=lambda _line: None,
            clock=MutableClock(0.0),
        )

        result = coordinator.tick()

        self.assertEqual(audio_observer.calls, 1)
        self.assertIsNone(result.decision)
        self.assertIsNone(result.wakeword_match)

    def test_coordinator_suppresses_trigger_when_resume_window_is_open(self) -> None:
        config = TwinrConfig(
            proactive_enabled=True,
            proactive_capture_interval_s=1.0,
            proactive_motion_window_s=20.0,
        )
        runtime = TwinrRuntime(config=config)
        emitted: list[str] = []
        handled: list[str] = []
        coordinator = ProactiveCoordinator(
            config=config,
            runtime=runtime,
            engine=SocialTriggerEngine(),
            trigger_handler=lambda decision: handled.append(decision.trigger_id) or True,
            vision_observer=None,
            pir_monitor=FakePirMonitor(level=False),
            emit=emitted.append,
            clock=MutableClock(10.0),
        )

        result = coordinator._process_decision(
            now=10.0,
            decision=SocialTriggerDecision(
                trigger_id="attention_window",
                prompt="Kann ich dir helfen?",
                reason="short quiet attention window",
                observed_at=10.0,
                priority=SocialTriggerPriority.ATTENTION_WINDOW,
                score=0.92,
                threshold=0.86,
            ),
            observation=SocialObservation(
                observed_at=10.0,
                inspected=False,
                pir_motion_detected=False,
                low_motion=False,
                vision=SocialVisionObservation(person_visible=False),
                audio=SocialAudioObservation(speech_detected=False),
            ),
            inspected=False,
            presence_snapshot=PresenceSessionSnapshot(
                armed=True,
                reason="recent_person_visible",
                person_visible=False,
                session_id=7,
            ),
            audio_policy_snapshot=ReSpeakerAudioPolicySnapshot(
                observed_at=10.0,
                recent_follow_up_speech=True,
                barge_in_recent=True,
                resume_window_open=True,
                initiative_block_reason="resume_window_open",
            ),
        )

        self.assertIsNone(result.decision)
        self.assertEqual(handled, [])
        self.assertIn("social_trigger_skipped=resume_window_open", emitted)
        skip_events = [entry for entry in runtime.ops_events.tail(limit=10) if entry.get("event") == "social_trigger_skipped"]
        self.assertEqual(skip_events[-1]["data"]["reason"], "resume_window_open")

    def test_coordinator_records_explicit_respeaker_runtime_alerts(self) -> None:
        config = TwinrConfig(
            proactive_enabled=True,
            proactive_capture_interval_s=1.0,
            proactive_motion_window_s=20.0,
        )
        runtime = TwinrRuntime(config=config)
        emitted: list[str] = []
        coordinator = ProactiveCoordinator(
            config=config,
            runtime=runtime,
            engine=SocialTriggerEngine(),
            trigger_handler=lambda _decision: True,
            vision_observer=None,
            pir_monitor=FakePirMonitor(level=False),
            emit=emitted.append,
            clock=MutableClock(5.0),
        )

        coordinator._record_observation_if_changed(
            SocialObservation(
                observed_at=5.0,
                inspected=False,
                pir_motion_detected=False,
                low_motion=False,
                vision=SocialVisionObservation(person_visible=False),
                audio=SocialAudioObservation(
                    speech_detected=False,
                    device_runtime_mode="usb_visible_no_capture",
                    host_control_ready=False,
                ),
            ),
            inspected=True,
            audio_policy_snapshot=ReSpeakerAudioPolicySnapshot(
                observed_at=5.0,
                mute_blocks_voice_capture=True,
                runtime_alert_code="dfu_mode",
                runtime_alert_message="ReSpeaker is visible on USB but has no ALSA capture device.",
            ),
            runtime_status_value="waiting",
        )
        coordinator._record_observation_if_changed(
            SocialObservation(
                observed_at=7.0,
                inspected=False,
                pir_motion_detected=False,
                low_motion=False,
                vision=SocialVisionObservation(person_visible=False),
                audio=SocialAudioObservation(
                    speech_detected=False,
                    device_runtime_mode="audio_ready",
                    host_control_ready=True,
                ),
            ),
            inspected=True,
            audio_policy_snapshot=ReSpeakerAudioPolicySnapshot(
                observed_at=7.0,
                mute_blocks_voice_capture=False,
                runtime_alert_code="ready",
                runtime_alert_message="ReSpeaker capture and host-control are ready.",
            ),
            runtime_status_value="waiting",
        )

        alerts = [entry for entry in runtime.ops_events.tail(limit=10) if entry.get("event") == "respeaker_runtime_alert"]
        self.assertEqual([entry["data"]["alert_code"] for entry in alerts[-2:]], ["dfu_mode", "ready"])
        self.assertIn("respeaker_runtime_alert=dfu_mode", emitted)
        self.assertIn("respeaker_runtime_alert=ready", emitted)

    def test_coordinator_does_not_inspect_without_recent_motion(self) -> None:
        config = TwinrConfig(proactive_enabled=True)
        runtime = TwinrRuntime(config=config)
        vision_observer = FakeVisionObserver(
            [SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.UPRIGHT)]
        )
        audio_observer = AmbientAudioObservationProvider(
            sampler=FakeAudioSampler(
                AmbientAudioLevelSample(
                    duration_ms=1000,
                    chunk_count=5,
                    active_chunk_count=0,
                    average_rms=120,
                    peak_rms=140,
                    active_ratio=0.0,
                )
            )
        )
        coordinator = ProactiveCoordinator(
            config=config,
            runtime=runtime,
            engine=SocialTriggerEngine(),
            trigger_handler=lambda _decision: True,
            vision_observer=vision_observer,
            pir_monitor=FakePirMonitor(level=False),
            audio_observer=audio_observer,
            emit=lambda _line: None,
            clock=MutableClock(5.0),
        )

        result = coordinator.tick()

        self.assertFalse(result.inspected)
        self.assertEqual(vision_observer.calls, 0)
        self.assertEqual(audio_observer.sampler.calls, 1)

    def test_coordinator_persists_changed_observations_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                proactive_enabled=True,
                proactive_capture_interval_s=1.0,
                proactive_motion_window_s=20.0,
            )
            runtime = TwinrRuntime(config=config)
            clock = MutableClock(1.0)
            sample = AmbientAudioLevelSample(
                duration_ms=900,
                chunk_count=5,
                active_chunk_count=0,
                average_rms=140,
                peak_rms=220,
                active_ratio=0.0,
            )
            coordinator = ProactiveCoordinator(
                config=config,
                runtime=runtime,
                engine=SocialTriggerEngine(),
                trigger_handler=lambda _decision: True,
                vision_observer=FakeVisionObserver(
                    [
                        SocialVisionObservation(
                            person_visible=True,
                            looking_toward_device=True,
                            body_pose=SocialBodyPose.UPRIGHT,
                        ),
                        SocialVisionObservation(
                            person_visible=True,
                            looking_toward_device=True,
                            body_pose=SocialBodyPose.UPRIGHT,
                        ),
                    ]
                ),
                pir_monitor=FakePirMonitor(events=[True], level=True),
                audio_observer=FakeAudioObserver(
                    SocialAudioObservation(speech_detected=False),
                    sample=sample,
                ),
                emit=lambda _line: None,
                clock=clock,
            )

            coordinator.tick()
            clock.now = 2.5
            coordinator.tick()

            events = runtime.ops_events.tail(limit=10)

        observation_events = [entry for entry in events if entry.get("event") == "proactive_observation"]
        self.assertEqual(len(observation_events), 1)
        self.assertEqual(observation_events[0]["data"]["person_visible"], True)
        self.assertEqual(observation_events[0]["data"]["body_pose"], "upright")
        self.assertEqual(observation_events[0]["data"]["speech_detected"], False)
        self.assertIn("top_trigger", observation_events[0]["data"])
        self.assertIn("top_score", observation_events[0]["data"])
        self.assertIn("top_threshold", observation_events[0]["data"])

    def test_coordinator_logs_trigger_detection_and_absence_transition(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                proactive_enabled=True,
                proactive_capture_interval_s=1.0,
                proactive_motion_window_s=5.0,
            )
            runtime = TwinrRuntime(config=config)
            clock = MutableClock(0.0)
            pir_monitor = FakePirMonitor(events=[True], level=True)
            handled: list[str] = []
            coordinator = ProactiveCoordinator(
                config=config,
                runtime=runtime,
                engine=SocialTriggerEngine.from_config(config),
                trigger_handler=lambda decision: handled.append(decision.trigger_id) or True,
                vision_observer=FakeVisionObserver(
                    [
                        SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.UPRIGHT),
                        SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.UPRIGHT),
                        SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.UPRIGHT),
                    ]
                ),
                pir_monitor=pir_monitor,
                audio_observer=FakeAudioObserver(SocialAudioObservation()),
                emit=lambda _line: None,
                clock=clock,
            )

            coordinator.tick()
            clock.now = 30.0
            pir_monitor.level = False
            coordinator.tick()
            clock.now = 21.0 * 60.0
            pir_monitor.events = [True]
            pir_monitor.level = True
            coordinator.tick()
            clock.now = (21.0 * 60.0) + 10.0
            pir_monitor.level = False
            coordinator.tick()

            events = runtime.ops_events.tail(limit=20)

        self.assertEqual(handled, ["person_returned"])
        trigger_events = [entry for entry in events if entry.get("event") == "proactive_trigger_detected"]
        self.assertEqual(len(trigger_events), 1)
        self.assertEqual(trigger_events[0]["data"]["trigger"], "person_returned")
        self.assertIn("Wie geht's dir?", trigger_events[0]["data"]["prompt"])
        self.assertGreaterEqual(trigger_events[0]["data"]["score"], trigger_events[0]["data"]["threshold"])
        self.assertTrue(trigger_events[0]["data"]["evidence"])
        observation_events = [entry for entry in events if entry.get("event") == "proactive_observation"]
        self.assertEqual(observation_events[-1]["data"]["person_visible"], False)
        self.assertEqual(observation_events[-1]["data"]["inspected"], False)

    def test_coordinator_suppresses_attention_window_after_recent_wakeword_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                proactive_enabled=True,
                proactive_capture_interval_s=1.0,
                proactive_motion_window_s=20.0,
                proactive_attention_window_s=1.0,
                proactive_attention_window_score_threshold=0.86,
                wakeword_block_proactive_after_attempt_s=15.0,
            )
            runtime = TwinrRuntime(config=config)
            clock = MutableClock(0.0)
            emitted: list[str] = []
            handled: list[str] = []
            coordinator = ProactiveCoordinator(
                config=config,
                runtime=runtime,
                engine=SocialTriggerEngine.from_config(config),
                trigger_handler=lambda decision: handled.append(decision.trigger_id) or True,
                vision_observer=FakeVisionObserver(
                    [
                        SocialVisionObservation(
                            person_visible=True,
                            looking_toward_device=True,
                            body_pose=SocialBodyPose.UPRIGHT,
                        ),
                        SocialVisionObservation(
                            person_visible=True,
                            looking_toward_device=True,
                            body_pose=SocialBodyPose.UPRIGHT,
                        ),
                    ]
                ),
                pir_monitor=FakePirMonitor(events=[True], level=True),
                audio_observer=FakeAudioObserver(SocialAudioObservation(speech_detected=False)),
                emit=emitted.append,
                clock=clock,
            )

            coordinator.tick()
            coordinator._last_wakeword_attempt_at = 1.4
            clock.now = 2.5
            result = coordinator.tick()

            events = runtime.ops_events.tail(limit=20)

        self.assertIsNone(result.decision)
        self.assertEqual(handled, [])
        self.assertIn("social_trigger_skipped=recent_wakeword_attempt", emitted)
        skip_events = [entry for entry in events if entry.get("event") == "social_trigger_skipped"]
        self.assertEqual(skip_events[-1]["data"]["trigger"], "attention_window")
        self.assertEqual(skip_events[-1]["data"]["reason"], "recent_wakeword_attempt")

    def test_coordinator_skips_attention_window_when_buffered_review_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                proactive_enabled=True,
                proactive_capture_interval_s=1.0,
                proactive_motion_window_s=20.0,
                proactive_attention_window_s=1.0,
                proactive_attention_window_score_threshold=0.86,
            )
            runtime = TwinrRuntime(config=config)
            clock = MutableClock(0.0)
            emitted: list[str] = []
            handled: list[str] = []
            reviewer = FakeVisionReviewer(
                ProactiveVisionReview(
                    approved=False,
                    decision="skip",
                    confidence="high",
                    reason="room looks empty",
                    scene="empty room",
                    frame_count=3,
                    response_id="resp_review",
                    request_id="req_review",
                    model="gpt-5.2",
                )
            )
            coordinator = ProactiveCoordinator(
                config=config,
                runtime=runtime,
                engine=SocialTriggerEngine.from_config(config),
                trigger_handler=lambda decision: handled.append(decision.trigger_id) or True,
                vision_observer=FakeVisionObserver(
                    [
                        SocialVisionObservation(
                            person_visible=True,
                            looking_toward_device=True,
                            body_pose=SocialBodyPose.UPRIGHT,
                        ),
                        SocialVisionObservation(
                            person_visible=True,
                            looking_toward_device=True,
                            body_pose=SocialBodyPose.UPRIGHT,
                        ),
                    ]
                ),
                pir_monitor=FakePirMonitor(events=[True], level=True),
                audio_observer=FakeAudioObserver(SocialAudioObservation(speech_detected=False)),
                vision_reviewer=reviewer,
                emit=emitted.append,
                clock=clock,
            )

            coordinator.tick()
            clock.now = 2.5
            result = coordinator.tick()

            events = runtime.ops_events.tail(limit=20)

        self.assertIsNone(result.decision)
        self.assertEqual(handled, [])
        self.assertEqual(len(reviewer.calls), 1)
        self.assertIn("social_trigger_skipped=vision_review_rejected", emitted)
        review_events = [entry for entry in events if entry.get("event") == "proactive_vision_reviewed"]
        self.assertEqual(review_events[-1]["data"]["trigger"], "attention_window")
        self.assertFalse(review_events[-1]["data"]["approved"])
        skip_events = [entry for entry in events if entry.get("event") == "social_trigger_skipped"]
        self.assertEqual(skip_events[-1]["data"]["trigger"], "attention_window")
        self.assertEqual(skip_events[-1]["data"]["reason"], "vision_review_rejected")

    def test_coordinator_skips_attention_window_when_buffered_review_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                proactive_enabled=True,
                proactive_capture_interval_s=1.0,
                proactive_motion_window_s=20.0,
                proactive_attention_window_s=1.0,
                proactive_attention_window_score_threshold=0.86,
            )
            runtime = TwinrRuntime(config=config)
            clock = MutableClock(0.0)
            emitted: list[str] = []
            handled: list[str] = []
            reviewer = FakeVisionReviewer(None)
            coordinator = ProactiveCoordinator(
                config=config,
                runtime=runtime,
                engine=SocialTriggerEngine.from_config(config),
                trigger_handler=lambda decision: handled.append(decision.trigger_id) or True,
                vision_observer=FakeVisionObserver(
                    [
                        SocialVisionObservation(
                            person_visible=True,
                            looking_toward_device=True,
                            body_pose=SocialBodyPose.UPRIGHT,
                        ),
                        SocialVisionObservation(
                            person_visible=True,
                            looking_toward_device=True,
                            body_pose=SocialBodyPose.UPRIGHT,
                        ),
                    ]
                ),
                pir_monitor=FakePirMonitor(events=[True], level=True),
                audio_observer=FakeAudioObserver(SocialAudioObservation(speech_detected=False)),
                vision_reviewer=reviewer,
                emit=emitted.append,
                clock=clock,
            )

            coordinator.tick()
            clock.now = 2.5
            result = coordinator.tick()

            events = runtime.ops_events.tail(limit=20)

        self.assertIsNone(result.decision)
        self.assertEqual(handled, [])
        self.assertEqual(len(reviewer.calls), 1)
        self.assertIn("social_trigger_skipped=vision_review_unavailable", emitted)
        unavailable_events = [entry for entry in events if entry.get("event") == "proactive_vision_review_unavailable"]
        self.assertEqual(unavailable_events[-1]["data"]["trigger"], "attention_window")
        skip_events = [entry for entry in events if entry.get("event") == "social_trigger_skipped"]
        self.assertEqual(skip_events[-1]["data"]["trigger"], "attention_window")
        self.assertEqual(skip_events[-1]["data"]["reason"], "vision_review_unavailable")

    def test_coordinator_reviews_absence_path_trigger_before_prompting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                proactive_enabled=True,
                proactive_capture_interval_s=0.1,
                proactive_motion_window_s=5.0,
                proactive_low_motion_after_s=1.0,
                proactive_possible_fall_stillness_s=4.0,
                proactive_possible_fall_visibility_loss_hold_s=4.0,
                proactive_possible_fall_visibility_loss_arming_s=2.0,
                proactive_possible_fall_slumped_visibility_loss_arming_s=2.0,
                proactive_possible_fall_score_threshold=0.65,
            )
            runtime = TwinrRuntime(config=config)
            clock = MutableClock(0.0)
            pir_monitor = FakePirMonitor(events=[True], level=True)
            handled: list[str] = []
            reviewer = FakeVisionReviewer(
                ProactiveVisionReview(
                    approved=True,
                    decision="speak",
                    confidence="medium",
                    reason="recent frames support concern",
                    scene="person drops out of view after slumped posture",
                    frame_count=2,
                    response_id="resp_review",
                    request_id="req_review",
                    model="gpt-5.2",
                )
            )
            coordinator = ProactiveCoordinator(
                config=config,
                runtime=runtime,
                engine=SocialTriggerEngine.from_config(config),
                trigger_handler=lambda decision: handled.append(decision.trigger_id) or True,
                vision_observer=FakeVisionObserver(
                    [
                        SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.SLUMPED),
                        SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.SLUMPED),
                    ]
                ),
                pir_monitor=pir_monitor,
                audio_observer=FakeAudioObserver(SocialAudioObservation(speech_detected=False)),
                vision_reviewer=reviewer,
                emit=lambda _line: None,
                clock=clock,
            )

            coordinator.tick()
            clock.now = 2.5
            coordinator.tick()
            pir_monitor.level = False
            clock.now = 2.55
            coordinator.tick()
            clock.now = 7.6
            result = coordinator.tick()

            events = runtime.ops_events.tail(limit=30)

        self.assertEqual(handled, ["possible_fall"])
        self.assertIsNotNone(result.decision)
        self.assertEqual(result.decision.trigger_id, "possible_fall")
        self.assertEqual(len(reviewer.calls), 1)
        trigger_events = [entry for entry in events if entry.get("event") == "proactive_trigger_detected"]
        self.assertEqual(trigger_events[-1]["data"]["trigger"], "possible_fall")
        self.assertEqual(trigger_events[-1]["data"]["vision_review_decision"], "speak")
        self.assertEqual(trigger_events[-1]["data"]["vision_review_reason"], "recent frames support concern")

    def test_coordinator_dispatches_absence_path_trigger_when_inspection_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                proactive_enabled=True,
                proactive_capture_interval_s=0.1,
                proactive_motion_window_s=5.0,
                proactive_low_motion_after_s=1.0,
                proactive_possible_fall_stillness_s=4.0,
                proactive_possible_fall_visibility_loss_hold_s=4.0,
                proactive_possible_fall_visibility_loss_arming_s=2.0,
                proactive_possible_fall_slumped_visibility_loss_arming_s=2.0,
                proactive_possible_fall_score_threshold=0.65,
            )
            runtime = TwinrRuntime(config=config)
            clock = MutableClock(0.0)
            pir_monitor = FakePirMonitor(events=[True], level=True)
            handled: list[str] = []
            coordinator = ProactiveCoordinator(
                config=config,
                runtime=runtime,
                engine=SocialTriggerEngine.from_config(config),
                trigger_handler=lambda decision: handled.append(decision.trigger_id) or True,
                vision_observer=FakeVisionObserver(
                    [
                        SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.SLUMPED),
                        SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.SLUMPED),
                    ]
                ),
                pir_monitor=pir_monitor,
                audio_observer=FakeAudioObserver(SocialAudioObservation(speech_detected=False)),
                emit=lambda _line: None,
                clock=clock,
            )

            coordinator.tick()
            clock.now = 2.5
            coordinator.tick()
            pir_monitor.level = False
            clock.now = 2.55
            result = coordinator.tick()
            self.assertIsNone(result.decision)
            self.assertFalse(result.inspected)

            clock.now = 7.6
            result = coordinator.tick()

            events = runtime.ops_events.tail(limit=20)

        self.assertEqual(handled, ["possible_fall"])
        self.assertIsNotNone(result.decision)
        self.assertEqual(result.decision.trigger_id, "possible_fall")
        self.assertFalse(result.inspected)
        trigger_events = [entry for entry in events if entry.get("event") == "proactive_trigger_detected"]
        self.assertEqual(trigger_events[-1]["data"]["trigger"], "possible_fall")
        observation_events = [entry for entry in events if entry.get("event") == "proactive_observation"]
        self.assertEqual(observation_events[-1]["data"]["inspected"], False)

    def test_coordinator_does_not_dispatch_absence_path_trigger_from_single_visible_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                proactive_enabled=True,
                proactive_capture_interval_s=10.0,
                proactive_motion_window_s=5.0,
                proactive_low_motion_after_s=1.0,
                proactive_possible_fall_stillness_s=4.0,
                proactive_possible_fall_visibility_loss_hold_s=4.0,
                proactive_possible_fall_visibility_loss_arming_s=2.0,
                proactive_possible_fall_slumped_visibility_loss_arming_s=2.0,
                proactive_possible_fall_score_threshold=0.65,
            )
            runtime = TwinrRuntime(config=config)
            clock = MutableClock(0.0)
            pir_monitor = FakePirMonitor(events=[True], level=True)
            handled: list[str] = []
            coordinator = ProactiveCoordinator(
                config=config,
                runtime=runtime,
                engine=SocialTriggerEngine.from_config(config),
                trigger_handler=lambda decision: handled.append(decision.trigger_id) or True,
                vision_observer=FakeVisionObserver(
                    [
                        SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.SLUMPED),
                    ]
                ),
                pir_monitor=pir_monitor,
                audio_observer=FakeAudioObserver(SocialAudioObservation(speech_detected=False)),
                emit=lambda _line: None,
                clock=clock,
            )

            coordinator.tick()
            clock.now = 2.5
            pir_monitor.level = False
            result = coordinator.tick()
            self.assertIsNone(result.decision)
            self.assertFalse(result.inspected)

            clock.now = 12.6
            result = coordinator.tick()

            events = runtime.ops_events.tail(limit=20)

        self.assertEqual(handled, [])
        self.assertIsNone(result.decision)
        trigger_events = [entry for entry in events if entry.get("event") == "proactive_trigger_detected"]
        self.assertEqual(trigger_events, [])

    def test_coordinator_suppresses_second_possible_fall_prompt_within_same_presence_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                proactive_enabled=True,
                proactive_capture_interval_s=0.1,
                proactive_motion_window_s=20.0,
                proactive_low_motion_after_s=1.0,
                proactive_possible_fall_stillness_s=4.0,
                proactive_possible_fall_visibility_loss_hold_s=4.0,
                proactive_possible_fall_visibility_loss_arming_s=2.0,
                proactive_possible_fall_slumped_visibility_loss_arming_s=2.0,
                proactive_possible_fall_score_threshold=0.65,
                wakeword_presence_grace_s=600.0,
                wakeword_motion_grace_s=120.0,
                wakeword_speech_grace_s=45.0,
                proactive_possible_fall_once_per_presence_session=True,
            )
            runtime = TwinrRuntime(config=config)
            clock = MutableClock(0.0)
            pir_monitor = FakePirMonitor(events=[True], level=False)
            handled: list[str] = []
            coordinator = ProactiveCoordinator(
                config=config,
                runtime=runtime,
                engine=SocialTriggerEngine.from_config(config),
                trigger_handler=lambda decision: handled.append(decision.trigger_id) or True,
                vision_observer=FakeVisionObserver(
                    [
                        SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.SLUMPED),
                        SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.SLUMPED),
                        SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.SLUMPED),
                        SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.SLUMPED),
                    ]
                ),
                presence_session=PresenceSessionController(
                    presence_grace_s=config.wakeword_presence_grace_s,
                    motion_grace_s=config.wakeword_motion_grace_s,
                    speech_grace_s=config.wakeword_speech_grace_s,
                ),
                pir_monitor=pir_monitor,
                audio_observer=FakeAudioObserver(SocialAudioObservation(speech_detected=False)),
                emit=lambda _line: None,
                clock=clock,
            )

            coordinator.tick()
            clock.now = 2.5
            coordinator.tick()
            clock.now = 2.55
            coordinator.tick()
            clock.now = 22.6
            first = coordinator.tick()
            self.assertIsNotNone(first.decision)
            self.assertEqual(first.decision.trigger_id, "possible_fall")

            clock.now = 70.0
            pir_monitor.events = [True]
            coordinator.tick()
            clock.now = 72.5
            coordinator.tick()
            clock.now = 72.55
            coordinator.tick()
            clock.now = 92.6
            second = coordinator.tick()

            events = runtime.ops_events.tail(limit=40)

        self.assertIsNone(second.decision)
        self.assertEqual(handled, ["possible_fall"])
        skip_events = [entry for entry in events if entry.get("event") == "social_trigger_skipped"]
        self.assertEqual(skip_events[-1]["data"]["trigger"], "possible_fall")
        self.assertEqual(
            skip_events[-1]["data"]["reason"],
            "already_prompted_this_presence_session",
        )

    def test_build_default_monitor_requires_pir(self) -> None:
        config = TwinrConfig(proactive_enabled=True)
        runtime = TwinrRuntime(config=config)

        monitor = build_default_proactive_monitor(
            config=config,
            runtime=runtime,
            backend=FakeBackend("person_visible=no"),
            camera=FakeCamera(),
            camera_lock=None,
            audio_lock=None,
            trigger_handler=lambda _decision: True,
            emit=lambda _line: None,
        )

        self.assertIsNone(monitor)

    def test_build_default_monitor_falls_back_to_stt_when_openwakeword_models_are_missing(self) -> None:
        config = TwinrConfig(
            wakeword_enabled=True,
            wakeword_backend="openwakeword",
            pir_motion_gpio=26,
        )
        runtime = TwinrRuntime(config=config)

        with patch("twinr.proactive.runtime.service.configured_pir_monitor", return_value=FakePirMonitor()):
            monitor = build_default_proactive_monitor(
                config=config,
                runtime=runtime,
                backend=FakeBackend("person_visible=no"),
                camera=FakeCamera(),
                camera_lock=None,
                audio_lock=None,
                trigger_handler=lambda _decision: True,
                emit=lambda _line: None,
            )

        self.assertIsNotNone(monitor)
        self.assertIsInstance(monitor.coordinator.wakeword_spotter, WakewordPhraseSpotter)
        events = runtime.ops_events.tail(limit=10)
        warning = next(entry for entry in events if entry.get("event") == "wakeword_backend_warning")
        self.assertEqual(warning["data"]["reason"], "openwakeword_models_missing")
        self.assertEqual(warning["data"]["fallback_backend"], "stt")

    def test_build_default_monitor_prefers_local_camera_provider_by_default(self) -> None:
        config = TwinrConfig(
            proactive_enabled=True,
            pir_motion_gpio=26,
        )
        runtime = TwinrRuntime(config=config)
        local_provider = FakeVisionObserver([SocialVisionObservation(person_visible=False)])

        with (
            patch("twinr.proactive.runtime.service.configured_pir_monitor", return_value=FakePirMonitor()),
            patch(
                "twinr.proactive.runtime.service.LocalAICameraObservationProvider.from_config",
                return_value=local_provider,
            ) as build_local_provider,
        ):
            monitor = build_default_proactive_monitor(
                config=config,
                runtime=runtime,
                backend=FakeBackend("person_visible=no"),
                camera=FakeCamera(),
                camera_lock=None,
                audio_lock=None,
                trigger_handler=lambda _decision: True,
                emit=lambda _line: None,
            )

        self.assertIsNotNone(monitor)
        self.assertIs(monitor.coordinator.vision_observer, local_provider)
        build_local_provider.assert_called_once_with(config)

    def test_build_default_monitor_wraps_audio_with_respeaker_signals_and_warns_on_permission_issue(self) -> None:
        config = TwinrConfig(
            wakeword_enabled=True,
            wakeword_backend="stt",
            pir_motion_gpio=26,
            audio_input_device="plughw:CARD=Array,DEV=0",
            proactive_audio_enabled=True,
            proactive_audio_input_device="plughw:CARD=Array,DEV=0",
        )
        runtime = TwinrRuntime(config=config)
        fake_signal_provider = FakeReSpeakerSignalProvider(
            ReSpeakerSignalSnapshot(
                captured_at=10.0,
                source="respeaker_xvf3800",
                source_type="observed",
                sensor_window_ms=config.proactive_audio_sample_ms,
                device_runtime_mode="audio_ready",
                host_control_ready=False,
                transport_reason="permission_denied_or_transport_blocked",
                requires_elevated_permissions=True,
            )
        )
        fake_sampler = FakeAudioSampler(
            AmbientAudioLevelSample(
                duration_ms=1000,
                chunk_count=5,
                active_chunk_count=1,
                average_rms=200,
                peak_rms=500,
                active_ratio=0.2,
            )
        )

        with (
            patch("twinr.proactive.runtime.service.configured_pir_monitor", return_value=FakePirMonitor()),
            patch("twinr.proactive.runtime.service.AmbientAudioSampler.from_config", return_value=fake_sampler),
            patch("twinr.proactive.runtime.service.ReSpeakerSignalProvider", return_value=fake_signal_provider),
        ):
            monitor = build_default_proactive_monitor(
                config=config,
                runtime=runtime,
                backend=FakeBackend("person_visible=no"),
                camera=FakeCamera(),
                camera_lock=None,
                audio_lock=None,
                trigger_handler=lambda _decision: True,
                emit=lambda _line: None,
            )

        self.assertIsNotNone(monitor)
        self.assertIsInstance(monitor.coordinator.audio_observer, ReSpeakerAudioObservationProvider)
        self.assertIs(monitor.coordinator.audio_observer.signal_provider, fake_signal_provider)
        events = runtime.ops_events.tail(limit=10)
        warning = next(
            entry
            for entry in events
            if entry.get("event") == "proactive_component_warning"
            and entry.get("data", {}).get("reason") == "respeaker_signal_provider_degraded"
        )
        self.assertEqual(warning["data"]["reason"], "respeaker_signal_provider_degraded")
        self.assertIn("USB permissions are likely missing", warning["data"]["detail"])

    def test_coordinator_emits_sensor_facts_and_events_for_automations(self) -> None:
        config = TwinrConfig(
            proactive_enabled=True,
            proactive_capture_interval_s=1.0,
            proactive_motion_window_s=20.0,
        )
        runtime = TwinrRuntime(config=config)
        clock = MutableClock(2.0)
        observations: list[tuple[dict[str, object], tuple[str, ...]]] = []
        coordinator = ProactiveCoordinator(
            config=config,
            runtime=runtime,
            engine=SocialTriggerEngine(),
            trigger_handler=lambda _decision: True,
            vision_observer=FakeVisionObserver(
                [
                    SocialVisionObservation(
                        person_visible=True,
                        looking_toward_device=True,
                        hand_or_object_near_camera=True,
                        body_pose=SocialBodyPose.UPRIGHT,
                    ),
                    SocialVisionObservation(
                        person_visible=True,
                        looking_toward_device=True,
                        hand_or_object_near_camera=True,
                        body_pose=SocialBodyPose.UPRIGHT,
                    ),
                ]
            ),
            pir_monitor=FakePirMonitor(events=[True], level=True),
            audio_observer=FakeAudioObserver(
                SocialAudioObservation(
                    speech_detected=True,
                    room_quiet=False,
                    recent_speech_age_s=0.0,
                    azimuth_deg=277,
                    device_runtime_mode="audio_ready",
                    signal_source="respeaker_xvf3800",
                    host_control_ready=True,
                )
            ),
            observation_handler=lambda facts, event_names: observations.append((facts, event_names)),
            emit=lambda _line: None,
            clock=clock,
        )

        coordinator.tick()
        clock.now = 8.0
        coordinator.tick()

        self.assertEqual(len(observations), 2)
        first_facts, first_events = observations[0]
        second_facts, second_events = observations[1]
        self.assertIn("pir.motion_detected", first_events)
        self.assertIn("camera.person_visible", first_events)
        self.assertIn("camera.hand_or_object_near_camera", first_events)
        self.assertIn("vad.speech_detected", first_events)
        self.assertEqual(first_facts["camera"]["person_visible"], True)
        self.assertEqual(first_facts["vad"]["speech_detected"], True)
        self.assertEqual(first_facts["vad"]["signal_source"], "respeaker_xvf3800")
        self.assertEqual(first_facts["vad"]["assistant_output_active"], None)
        self.assertEqual(first_facts["respeaker"]["runtime_mode"], "audio_ready")
        self.assertEqual(first_facts["respeaker"]["azimuth_deg"], 277)
        self.assertEqual(first_facts["respeaker"]["direction_confidence"], None)
        self.assertEqual(first_facts["respeaker"]["speech_overlap_likely"], None)
        self.assertEqual(first_facts["respeaker"]["barge_in_detected"], None)
        self.assertEqual(first_facts["respeaker"]["host_control_ready"], True)
        self.assertEqual(first_facts["audio_policy"]["presence_audio_active"], True)
        self.assertEqual(first_facts["audio_policy"]["recent_follow_up_speech"], False)
        self.assertEqual(first_facts["audio_policy"]["room_busy_or_overlapping"], None)
        self.assertEqual(first_facts["audio_policy"]["quiet_window_open"], False)
        self.assertEqual(first_facts["audio_policy"]["resume_window_open"], False)
        self.assertEqual(first_facts["audio_policy"]["initiative_block_reason"], "presence_audio_active")
        self.assertEqual(first_facts["audio_policy"]["runtime_alert_code"], "ready")
        self.assertEqual(second_events, ())
        self.assertGreaterEqual(second_facts["camera"]["person_visible_for_s"], 6.0)


if __name__ == "__main__":
    unittest.main()
