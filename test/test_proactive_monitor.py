from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.display.emoji_cues import DisplayEmojiCueStore
from twinr.display.face_cues import DisplayFaceCueStore
from twinr.hardware.portrait_match import PortraitMatchObservation
from twinr.hardware.respeaker.models import ReSpeakerSignalSnapshot
from twinr.hardware.audio import (
    AmbientAudioLevelSample,
    AudioCaptureReadinessError,
    AudioCaptureReadinessProbe,
)
from twinr.proactive import (
    AmbientAudioObservationProvider,
    OpenAIVisionObservationProvider,
    PresenceSessionController,
    ProactiveAudioSnapshot,
    ProactiveCoordinator,
    PresenceSessionSnapshot,
    SocialAudioObservation,
    SocialBodyPose,
    SocialFineHandGesture,
    SocialGestureEvent,
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
    def __init__(self, observations, *, supports_attention_refresh: bool = False):
        self.observations = list(observations)
        self.calls = 0
        self.supports_attention_refresh = supports_attention_refresh

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
        self.probe_calls = 0

    def sample_levels(self, *, duration_ms=None):
        self.calls += 1
        self.durations.append(duration_ms)
        return self.sample

    def require_readable_frames(self, *, duration_ms=None, chunk_count=1):
        self.probe_calls += 1
        return AudioCaptureReadinessProbe(
            device="plughw:CARD=Array,DEV=0",
            sample_rate=16000,
            channels=1,
            chunk_ms=100,
            duration_ms=duration_ms or 100,
            target_chunk_count=chunk_count,
            captured_chunk_count=chunk_count,
            captured_bytes=3200,
            detail="Captured readable frames.",
        )


class FakeAudioObserver:
    def __init__(self, observation: SocialAudioObservation, *, sample: AmbientAudioLevelSample | None = None) -> None:
        self.snapshot = ProactiveAudioSnapshot(observation=observation, sample=sample)
        self.calls = 0

    def observe(self):
        self.calls += 1
        return self.snapshot


class FakeSequencedAudioObserver:
    def __init__(self, observations: list[SocialAudioObservation]) -> None:
        self.snapshots = [ProactiveAudioSnapshot(observation=item, sample=None) for item in observations]
        self.calls = 0

    def observe(self):
        self.calls += 1
        if len(self.snapshots) > 1:
            return self.snapshots.pop(0)
        return self.snapshots[0]


class FastAttentionAudioObserver:
    def __init__(self, observation: SocialAudioObservation) -> None:
        self.snapshot = ProactiveAudioSnapshot(observation=observation, sample=None)
        self.observe_calls = 0
        self.signal_only_calls = 0

    def observe(self):
        self.observe_calls += 1
        raise AssertionError("refresh_display_attention should not call the slow audio path")

    def observe_signal_only(self):
        self.signal_only_calls += 1
        return self.snapshot


class FakeClock:
    def __init__(self, start: float) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, delta: float) -> None:
        self.now += delta


class FailingAudioObserver:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def observe(self):
        self.calls += 1
        raise self.error


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


class FakePortraitMatchProvider:
    def __init__(self, observation: PortraitMatchObservation) -> None:
        self.observation = observation
        self.backend = SimpleNamespace(name="fake_portrait_backend")
        self.calls = 0

    def observe(self) -> PortraitMatchObservation:
        self.calls += 1
        return self.observation


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
        blocker_events = [
            entry
            for entry in runtime.ops_events.tail(limit=12)
            if entry.get("event") in {"respeaker_runtime_blocker", "respeaker_runtime_blocker_cleared"}
        ]
        self.assertEqual(
            [entry["event"] for entry in blocker_events[-2:]],
            ["respeaker_runtime_blocker", "respeaker_runtime_blocker_cleared"],
        )
        self.assertIn("respeaker_runtime_alert=dfu_mode", emitted)
        self.assertIn("respeaker_runtime_alert=ready", emitted)
        self.assertIn("respeaker_runtime_blocker=dfu_mode", emitted)
        self.assertIn("respeaker_runtime_blocker_cleared=dfu_mode", emitted)

    def test_build_default_monitor_fails_closed_when_respeaker_starts_in_dfu_mode(self) -> None:
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
                device_runtime_mode="usb_visible_no_capture",
                host_control_ready=False,
                transport_reason="dfu_mode",
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
            with self.assertRaises(RuntimeError) as ctx:
                build_default_proactive_monitor(
                    config=config,
                    runtime=runtime,
                    backend=FakeBackend("person_visible=no"),
                    camera=FakeCamera(),
                    camera_lock=None,
                    audio_lock=None,
                    trigger_handler=lambda _decision: True,
                    emit=lambda _line: None,
                )

        self.assertIn("DFU/safe mode", str(ctx.exception))
        events = runtime.ops_events.tail(limit=10)
        blocker = next(
            entry
            for entry in events
            if entry.get("event") == "proactive_component_blocked"
            and entry.get("data", {}).get("reason") == "respeaker_dfu_mode_blocked"
        )
        self.assertEqual(blocker["data"]["blocker_code"], "dfu_mode")
        self.assertEqual(blocker["data"]["device_runtime_mode"], "usb_visible_no_capture")

    def test_build_default_monitor_fails_closed_when_respeaker_capture_has_no_readable_frames(self) -> None:
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
                host_control_ready=True,
                transport_reason=None,
                firmware_version=(2, 0, 7),
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
        probe = AudioCaptureReadinessProbe(
            device="plughw:CARD=Array,DEV=0",
            sample_rate=16000,
            channels=1,
            chunk_ms=100,
            duration_ms=300,
            target_chunk_count=1,
            captured_chunk_count=0,
            captured_bytes=0,
            failure_reason="stalled_waiting",
            detail="Ambient audio capture stalled while waiting for microphone data",
        )

        with (
            patch("twinr.proactive.runtime.service.configured_pir_monitor", return_value=FakePirMonitor()),
            patch("twinr.proactive.runtime.service.AmbientAudioSampler.from_config", return_value=fake_sampler),
            patch("twinr.proactive.runtime.service.ReSpeakerSignalProvider", return_value=fake_signal_provider),
            patch.object(fake_sampler, "require_readable_frames", side_effect=AudioCaptureReadinessError(str(probe.detail), probe=probe)),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                build_default_proactive_monitor(
                    config=config,
                    runtime=runtime,
                    backend=FakeBackend("person_visible=no"),
                    camera=FakeCamera(),
                    camera_lock=None,
                    audio_lock=None,
                    trigger_handler=lambda _decision: True,
                    emit=lambda _line: None,
                )

        self.assertIn("yielded no readable audio frames", str(ctx.exception))
        events = runtime.ops_events.tail(limit=12)
        alert = next(
            entry
            for entry in events
            if entry.get("event") == "respeaker_runtime_alert"
            and entry.get("data", {}).get("alert_code") == "capture_unknown"
        )
        blocker = next(
            entry
            for entry in events
            if entry.get("event") == "proactive_component_blocked"
            and entry.get("data", {}).get("reason") == "respeaker_dead_capture_blocked"
        )
        self.assertEqual(alert["data"]["capture_probe_failure_reason"], "stalled_waiting")
        self.assertEqual(blocker["data"]["blocker_code"], "dead_capture")
        self.assertEqual(blocker["data"]["device_runtime_mode"], "audio_ready")
        self.assertEqual(blocker["data"]["host_control_ready"], True)

    def test_coordinator_blocks_respeaker_runtime_when_capture_dies_midstream(self) -> None:
        config = TwinrConfig(
            wakeword_enabled=True,
            wakeword_backend="stt",
            audio_input_device="plughw:CARD=Array,DEV=0",
            proactive_audio_enabled=True,
            proactive_audio_input_device="plughw:CARD=Array,DEV=0",
        )
        runtime = TwinrRuntime(config=config)
        probe = AudioCaptureReadinessProbe(
            device="plughw:CARD=Array,DEV=0",
            sample_rate=16000,
            channels=1,
            chunk_ms=100,
            duration_ms=300,
            target_chunk_count=1,
            captured_chunk_count=0,
            captured_bytes=0,
            failure_reason="stalled_waiting",
            detail="Ambient audio capture stalled while waiting for microphone data",
        )
        coordinator = ProactiveCoordinator(
            config=config,
            runtime=runtime,
            engine=SocialTriggerEngine(),
            trigger_handler=lambda _decision: True,
            vision_observer=None,
            audio_observer=FailingAudioObserver(
                AudioCaptureReadinessError(str(probe.detail), probe=probe)
            ),
            emit=lambda _line: None,
        )

        snapshot = coordinator._observe_audio_safe()

        self.assertIs(coordinator.audio_observer, coordinator._null_audio_observer)
        self.assertIsNotNone(snapshot)
        events = runtime.ops_events.tail(limit=12)
        alert = next(
            entry
            for entry in events
            if entry.get("event") == "respeaker_runtime_alert"
            and entry.get("data", {}).get("alert_code") == "capture_unknown"
        )
        blocker = next(
            entry
            for entry in events
            if entry.get("event") == "respeaker_runtime_blocker"
            and entry.get("data", {}).get("blocker_code") == "dead_capture"
        )
        self.assertEqual(alert["data"]["stage"], "runtime")
        self.assertEqual(blocker["data"]["capture_probe_failure_reason"], "stalled_waiting")

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
                            person_count=1,
                            primary_person_zone=SocialPersonZone.CENTER,
                            primary_person_center_x=0.52,
                            primary_person_center_y=0.47,
                            looking_toward_device=True,
                            person_near_device=True,
                            engaged_with_device=True,
                            visual_attention_score=0.8123,
                            body_pose=SocialBodyPose.UPRIGHT,
                            pose_confidence=0.9132,
                            motion_confidence=0.6421,
                            hand_or_object_near_camera=True,
                            gesture_event=SocialGestureEvent.WAVE,
                            gesture_confidence=0.7321,
                            fine_hand_gesture=SocialFineHandGesture.THUMBS_UP,
                            fine_hand_gesture_confidence=0.6543,
                            camera_online=True,
                            camera_ready=True,
                            camera_ai_ready=True,
                        ),
                        SocialVisionObservation(
                            person_visible=True,
                            person_count=1,
                            primary_person_zone=SocialPersonZone.CENTER,
                            primary_person_center_x=0.52,
                            primary_person_center_y=0.47,
                            looking_toward_device=True,
                            person_near_device=True,
                            engaged_with_device=True,
                            visual_attention_score=0.8123,
                            body_pose=SocialBodyPose.UPRIGHT,
                            pose_confidence=0.9132,
                            motion_confidence=0.6421,
                            hand_or_object_near_camera=True,
                            gesture_event=SocialGestureEvent.WAVE,
                            gesture_confidence=0.7321,
                            fine_hand_gesture=SocialFineHandGesture.THUMBS_UP,
                            fine_hand_gesture_confidence=0.6543,
                            camera_online=True,
                            camera_ready=True,
                            camera_ai_ready=True,
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
        self.assertEqual(observation_events[0]["data"]["camera_person_count"], 1)
        self.assertEqual(observation_events[0]["data"]["camera_primary_person_zone"], "center")
        self.assertEqual(observation_events[0]["data"]["camera_primary_person_center_x"], 0.52)
        self.assertEqual(observation_events[0]["data"]["camera_primary_person_center_y"], 0.47)
        self.assertEqual(observation_events[0]["data"]["body_pose"], "upright")
        self.assertEqual(observation_events[0]["data"]["camera_person_near_device"], True)
        self.assertEqual(observation_events[0]["data"]["camera_engaged_with_device"], True)
        self.assertEqual(observation_events[0]["data"]["camera_visual_attention_score"], 0.8123)
        self.assertEqual(observation_events[0]["data"]["camera_pose_confidence"], 0.9132)
        self.assertEqual(observation_events[0]["data"]["camera_motion_state"], "unknown")
        self.assertEqual(observation_events[0]["data"]["camera_motion_confidence"], 0.6421)
        self.assertEqual(observation_events[0]["data"]["camera_gesture_event"], "wave")
        self.assertEqual(observation_events[0]["data"]["camera_gesture_confidence"], 0.7321)
        self.assertEqual(observation_events[0]["data"]["camera_fine_hand_gesture"], "thumbs_up")
        self.assertEqual(observation_events[0]["data"]["camera_fine_hand_gesture_confidence"], 0.6543)
        self.assertEqual(observation_events[0]["data"]["camera_online"], True)
        self.assertEqual(observation_events[0]["data"]["camera_ready"], True)
        self.assertEqual(observation_events[0]["data"]["camera_ai_ready"], True)
        self.assertIsNone(observation_events[0]["data"]["camera_error"])
        self.assertEqual(observation_events[0]["data"]["speech_detected"], False)
        self.assertIn("top_trigger", observation_events[0]["data"])
        self.assertIn("top_score", observation_events[0]["data"])
        self.assertIn("top_threshold", observation_events[0]["data"])

    def test_coordinator_updates_display_attention_cue_for_visible_person(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                proactive_enabled=True,
                proactive_capture_interval_s=1.0,
                proactive_motion_window_s=20.0,
            )
            runtime = TwinrRuntime(config=config)
            coordinator = ProactiveCoordinator(
                config=config,
                runtime=runtime,
                engine=SocialTriggerEngine(),
                trigger_handler=lambda _decision: True,
                vision_observer=FakeVisionObserver(
                    [
                        SocialVisionObservation(
                            person_visible=True,
                            primary_person_center_x=0.14,
                            primary_person_center_y=0.48,
                            body_pose=SocialBodyPose.UPRIGHT,
                        )
                    ]
                ),
                pir_monitor=FakePirMonitor(events=[True], level=True),
                audio_observer=FakeAudioObserver(SocialAudioObservation(speech_detected=False)),
                emit=lambda _line: None,
                clock=MutableClock(1.0),
            )

            result = coordinator.tick()
            cue = DisplayFaceCueStore.from_config(config).load_active()

        self.assertTrue(result.inspected)
        self.assertIsNotNone(cue)
        assert cue is not None
        self.assertEqual(cue.source, "proactive_attention_follow")
        self.assertEqual(cue.gaze_x, 2)
        self.assertEqual(cue.gaze_y, 0)

    def test_display_attention_refresh_updates_cue_without_pir_motion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                proactive_enabled=True,
                display_driver="hdmi_wayland",
                display_attention_refresh_interval_s=1.0,
            )
            runtime = TwinrRuntime(config=config)
            clock = MutableClock(10.0)
            coordinator = ProactiveCoordinator(
                config=config,
                runtime=runtime,
                engine=SocialTriggerEngine(),
                trigger_handler=lambda _decision: True,
                vision_observer=FakeVisionObserver(
                    [
                        SocialVisionObservation(
                            person_visible=True,
                            primary_person_center_x=0.81,
                            primary_person_center_y=0.44,
                            engaged_with_device=True,
                        )
                    ],
                    supports_attention_refresh=True,
                ),
                pir_monitor=FakePirMonitor(),
                audio_observer=FakeAudioObserver(SocialAudioObservation(speech_detected=False)),
                emit=lambda _line: None,
                clock=clock,
            )

            refreshed = coordinator.refresh_display_attention()
            cue = DisplayFaceCueStore.from_config(config).load_active()

        self.assertTrue(refreshed)
        self.assertIsNotNone(cue)
        assert cue is not None
        self.assertEqual(cue.source, "proactive_attention_follow")
        self.assertEqual(cue.gaze_x, -2)
        self.assertEqual(cue.gaze_y, 0)

    def test_display_attention_refresh_records_changed_ops_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                proactive_enabled=True,
                display_driver="hdmi_wayland",
                display_attention_refresh_interval_s=1.0,
            )
            runtime = TwinrRuntime(config=config)
            clock = MutableClock(10.0)
            coordinator = ProactiveCoordinator(
                config=config,
                runtime=runtime,
                engine=SocialTriggerEngine(),
                trigger_handler=lambda _decision: True,
                vision_observer=FakeVisionObserver(
                    [
                        SocialVisionObservation(
                            person_visible=True,
                            person_count=1,
                            primary_person_zone=SocialPersonZone.CENTER,
                            primary_person_center_x=0.81,
                            primary_person_center_y=0.44,
                            engaged_with_device=True,
                            camera_online=True,
                            camera_ready=True,
                            camera_ai_ready=True,
                            last_camera_frame_at=10.0,
                        )
                    ],
                    supports_attention_refresh=True,
                ),
                pir_monitor=FakePirMonitor(),
                audio_observer=FakeAudioObserver(SocialAudioObservation(speech_detected=False)),
                emit=lambda _line: None,
                clock=clock,
            )

            refreshed = coordinator.refresh_display_attention()
            events = runtime.ops_events.tail(limit=10)

        self.assertTrue(refreshed)
        follow_events = [entry for entry in events if entry.get("event") == "proactive_display_attention_follow"]
        self.assertTrue(follow_events)
        follow = follow_events[-1]
        self.assertEqual(follow["data"]["publish_action"], "updated")
        self.assertEqual(follow["data"]["camera_primary_person_zone"], "center")
        self.assertEqual(follow["data"]["attention_target_horizontal"], "right")

    def test_display_attention_refresh_acknowledges_fine_hand_gesture_with_emoji(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                proactive_enabled=True,
                display_driver="hdmi_wayland",
                display_attention_refresh_interval_s=1.0,
            )
            runtime = TwinrRuntime(config=config)
            clock = MutableClock(10.0)
            coordinator = ProactiveCoordinator(
                config=config,
                runtime=runtime,
                engine=SocialTriggerEngine(),
                trigger_handler=lambda _decision: True,
                vision_observer=FakeVisionObserver(
                    [
                        SocialVisionObservation(
                            person_visible=True,
                            hand_or_object_near_camera=True,
                            showing_intent_likely=True,
                            fine_hand_gesture=SocialFineHandGesture.THUMBS_UP,
                            fine_hand_gesture_confidence=0.91,
                        )
                    ],
                    supports_attention_refresh=True,
                ),
                pir_monitor=FakePirMonitor(),
                audio_observer=FakeAudioObserver(SocialAudioObservation(speech_detected=False)),
                emit=lambda _line: None,
                clock=clock,
            )

            refreshed = coordinator.refresh_display_attention()
            cue = DisplayEmojiCueStore.from_config(config).load_active()

        self.assertTrue(refreshed)
        self.assertIsNotNone(cue)
        assert cue is not None
        self.assertEqual(cue.source, "proactive_gesture_ack")
        self.assertEqual(cue.symbol, "thumbs_up")
        self.assertEqual(cue.accent, "success")

    def test_display_attention_refresh_uses_fast_signal_only_audio_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                proactive_enabled=True,
                display_driver="hdmi_wayland",
                display_attention_refresh_interval_s=1.0,
            )
            runtime = TwinrRuntime(config=config)
            clock = MutableClock(10.0)
            audio_observer = FastAttentionAudioObserver(
                SocialAudioObservation(
                    speech_detected=False,
                    azimuth_deg=0,
                    direction_confidence=0.92,
                )
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
                            primary_person_center_x=0.81,
                            primary_person_center_y=0.44,
                            engaged_with_device=True,
                        )
                    ],
                    supports_attention_refresh=True,
                ),
                pir_monitor=FakePirMonitor(),
                audio_observer=audio_observer,
                emit=lambda _line: None,
                clock=clock,
            )

            refreshed = coordinator.refresh_display_attention()
            cue = DisplayFaceCueStore.from_config(config).load_active()

        self.assertTrue(refreshed)
        self.assertIsNotNone(cue)
        assert cue is not None
        self.assertEqual(audio_observer.signal_only_calls, 1)
        self.assertEqual(audio_observer.observe_calls, 0)
        self.assertEqual(cue.gaze_x, -2)

    def test_tick_acknowledges_wave_gesture_with_emoji(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                proactive_enabled=True,
                proactive_capture_interval_s=1.0,
                proactive_motion_window_s=5.0,
                display_driver="hdmi_wayland",
            )
            runtime = TwinrRuntime(config=config)
            coordinator = ProactiveCoordinator(
                config=config,
                runtime=runtime,
                engine=SocialTriggerEngine(),
                trigger_handler=lambda _decision: True,
                vision_observer=FakeVisionObserver(
                    [
                        SocialVisionObservation(
                            person_visible=True,
                            gesture_event=SocialGestureEvent.WAVE,
                            gesture_confidence=0.84,
                        )
                    ]
                ),
                pir_monitor=FakePirMonitor(events=[True], level=True),
                audio_observer=FakeAudioObserver(SocialAudioObservation(speech_detected=False)),
                emit=lambda _line: None,
                clock=MutableClock(1.0),
            )

            result = coordinator.tick()
            cue = DisplayEmojiCueStore.from_config(config).load_active()

        self.assertTrue(result.inspected)
        self.assertIsNotNone(cue)
        assert cue is not None
        self.assertEqual(cue.source, "proactive_gesture_ack")
        self.assertEqual(cue.symbol, "waving_hand")
        self.assertEqual(cue.accent, "warm")

    def test_display_attention_refresh_updates_cue_during_error_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                proactive_enabled=True,
                display_driver="hdmi_wayland",
                display_attention_refresh_interval_s=1.0,
            )
            runtime = TwinrRuntime(config=config)
            runtime.fail("Remote long-term memory is unavailable.")
            clock = MutableClock(10.0)
            coordinator = ProactiveCoordinator(
                config=config,
                runtime=runtime,
                engine=SocialTriggerEngine(),
                trigger_handler=lambda _decision: True,
                vision_observer=FakeVisionObserver(
                    [
                        SocialVisionObservation(
                            person_visible=True,
                            primary_person_center_x=0.18,
                            primary_person_center_y=0.44,
                            engaged_with_device=True,
                        )
                    ],
                    supports_attention_refresh=True,
                ),
                pir_monitor=FakePirMonitor(),
                audio_observer=FakeAudioObserver(SocialAudioObservation(speech_detected=False)),
                emit=lambda _line: None,
                clock=clock,
            )

            refreshed = coordinator.refresh_display_attention()
            cue = DisplayFaceCueStore.from_config(config).load_active()

        self.assertEqual(runtime.status.value, "error")
        self.assertTrue(refreshed)
        self.assertIsNotNone(cue)
        assert cue is not None
        self.assertEqual(cue.source, "proactive_attention_follow")
        self.assertEqual(cue.gaze_x, 2)
        self.assertEqual(cue.gaze_y, 0)

    def test_display_attention_refresh_keeps_recent_session_focus_in_multi_person_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                proactive_enabled=True,
                display_driver="hdmi_wayland",
                display_attention_refresh_interval_s=1.0,
                display_attention_session_focus_hold_s=5.0,
            )
            runtime = TwinrRuntime(config=config)
            runtime.begin_listening(request_source="test")
            clock = MutableClock(10.0)
            coordinator = ProactiveCoordinator(
                config=config,
                runtime=runtime,
                engine=SocialTriggerEngine(),
                trigger_handler=lambda _decision: True,
                vision_observer=FakeVisionObserver(
                    [
                        SocialVisionObservation(
                            person_visible=True,
                            person_count=1,
                            primary_person_center_x=0.16,
                            primary_person_center_y=0.48,
                            looking_toward_device=True,
                        ),
                        SocialVisionObservation(
                            person_visible=True,
                            person_count=2,
                            person_recently_visible=True,
                            primary_person_center_x=0.84,
                            primary_person_center_y=0.5,
                        ),
                    ],
                    supports_attention_refresh=True,
                ),
                pir_monitor=FakePirMonitor(),
                audio_observer=FakeSequencedAudioObserver(
                    [
                        SocialAudioObservation(
                            speech_detected=True,
                            azimuth_deg=0,
                            direction_confidence=0.9,
                        ),
                        SocialAudioObservation(
                            speech_detected=False,
                            azimuth_deg=0,
                            direction_confidence=0.9,
                        ),
                    ]
                ),
                emit=lambda _line: None,
                clock=clock,
            )

            first_refresh = coordinator.refresh_display_attention()
            clock.now = 11.5
            second_refresh = coordinator.refresh_display_attention()
            cue = DisplayFaceCueStore.from_config(config).load_active()

        self.assertTrue(first_refresh)
        self.assertTrue(second_refresh)
        assert cue is not None
        self.assertEqual(cue.source, "proactive_attention_follow")
        self.assertEqual(cue.gaze_x, 2)
        self.assertEqual(cue.gaze_y, 0)

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

    def test_build_default_monitor_does_not_inspect_without_pir_motion_history(self) -> None:
        config = TwinrConfig(
            proactive_enabled=True,
            display_driver="waveshare_4in2_v2",
            display_attention_refresh_interval_s=0.0,
        )
        runtime = TwinrRuntime(config=config)
        local_provider = FakeVisionObserver([SocialVisionObservation(person_visible=True)])

        with patch(
            "twinr.proactive.runtime.service.LocalAICameraObservationProvider.from_config",
            return_value=local_provider,
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
        result = monitor.coordinator.tick()
        self.assertFalse(result.inspected)
        self.assertEqual(local_provider.calls, 0)

    def test_build_default_monitor_bootstraps_inspection_from_speech_when_wakeword_enabled(self) -> None:
        config = TwinrConfig(
            proactive_enabled=True,
            wakeword_enabled=True,
            wakeword_backend="stt",
            wakeword_primary_backend="stt",
            display_driver="waveshare_4in2_v2",
            display_attention_refresh_interval_s=0.0,
            proactive_capture_interval_s=1.0,
        )
        runtime = TwinrRuntime(config=config)
        local_provider = FakeVisionObserver([SocialVisionObservation(person_visible=True)])

        with patch(
            "twinr.proactive.runtime.service.LocalAICameraObservationProvider.from_config",
            return_value=local_provider,
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
        monitor.coordinator.audio_observer = FakeAudioObserver(
            SocialAudioObservation(speech_detected=True)
        )
        result = monitor.coordinator.tick()
        events = runtime.ops_events.tail(limit=20)
        presence_event = next(
            entry for entry in reversed(events) if entry.get("event") == "wakeword_presence_changed"
        )

        self.assertTrue(result.inspected)
        self.assertEqual(local_provider.calls, 1)
        self.assertTrue(presence_event["data"]["armed"])
        self.assertEqual(presence_event["data"]["reason"], "person_visible")

    def test_build_default_monitor_falls_back_to_stt_when_openwakeword_models_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
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

    def test_build_default_monitor_falls_back_to_stt_when_kws_assets_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                wakeword_enabled=True,
                wakeword_backend="kws",
                wakeword_primary_backend="kws",
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
            self.assertEqual(warning["data"]["reason"], "kws_assets_missing")
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

    def test_build_default_monitor_keeps_hdmi_attention_camera_when_proactive_triggers_are_disabled(self) -> None:
        config = TwinrConfig(
            proactive_enabled=False,
            wakeword_enabled=False,
            display_driver="hdmi_wayland",
            display_attention_refresh_interval_s=1.25,
        )
        runtime = TwinrRuntime(config=config)
        local_provider = FakeVisionObserver(
            [SocialVisionObservation(person_visible=False)],
            supports_attention_refresh=True,
        )

        with (
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
        self.assertEqual(type(monitor.coordinator.engine).__name__, "_NullSocialTriggerEngine")
        self.assertIsNotNone(monitor.coordinator.audio_observer)
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
        scheduled_provider = monitor.coordinator.audio_observer.signal_provider
        self.assertIsNotNone(scheduled_provider)
        self.assertIs(getattr(scheduled_provider, "provider", None), fake_signal_provider)
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
        self.assertIn("audio_policy.presence_audio_active", first_events)
        self.assertEqual(first_facts["camera"]["person_visible"], True)
        self.assertEqual(first_facts["sensor"]["captured_at"], 2.0)
        self.assertIsNone(first_facts["sensor"]["presence_session_id"])
        self.assertEqual(first_facts["vad"]["speech_detected"], True)
        self.assertEqual(first_facts["vad"]["signal_source"], "respeaker_xvf3800")
        self.assertEqual(first_facts["vad"]["assistant_output_active"], None)
        self.assertEqual(first_facts["respeaker"]["runtime_mode"], "audio_ready")
        self.assertEqual(first_facts["respeaker"]["azimuth_deg"], 277)
        self.assertEqual(first_facts["respeaker"]["direction_confidence"], None)
        self.assertEqual(first_facts["respeaker"]["speech_overlap_likely"], None)
        self.assertEqual(first_facts["respeaker"]["barge_in_detected"], None)
        self.assertEqual(first_facts["respeaker"]["host_control_ready"], True)
        self.assertEqual(
            first_facts["speaker_association"]["state"],
            "audio_direction_unavailable",
        )
        self.assertEqual(first_facts["speaker_association"]["associated"], False)
        self.assertEqual(
            first_facts["multimodal_initiative"]["block_reason"],
            "low_confidence_speaker_association",
        )
        self.assertEqual(first_facts["multimodal_initiative"]["recommended_channel"], "display")
        self.assertTrue(first_facts["ambiguous_room_guard"]["guard_active"])
        self.assertEqual(
            first_facts["ambiguous_room_guard"]["reason"],
            "low_confidence_audio_direction",
        )
        self.assertEqual(
            first_facts["known_user_hint"]["state"],
            "blocked_ambiguous_room",
        )
        self.assertEqual(first_facts["known_user_hint"]["policy_recommendation"], "blocked")
        self.assertEqual(first_facts["affect_proxy"]["state"], "unknown")
        self.assertEqual(first_facts["audio_policy"]["presence_audio_active"], True)
        self.assertEqual(first_facts["audio_policy"]["recent_follow_up_speech"], False)
        self.assertEqual(first_facts["audio_policy"]["room_busy_or_overlapping"], None)
        self.assertEqual(first_facts["audio_policy"]["quiet_window_open"], False)
        self.assertEqual(first_facts["audio_policy"]["resume_window_open"], False)
        self.assertEqual(first_facts["audio_policy"]["initiative_block_reason"], "presence_audio_active")
        self.assertEqual(first_facts["audio_policy"]["runtime_alert_code"], "ready")
        self.assertEqual(second_events, ())
        self.assertGreaterEqual(second_facts["camera"]["person_visible_for_s"], 6.0)

    def test_coordinator_exports_portrait_match_and_multimodal_known_user_hint(self) -> None:
        config = TwinrConfig(
            proactive_enabled=True,
            proactive_capture_interval_s=1.0,
            proactive_motion_window_s=20.0,
        )
        runtime = TwinrRuntime(config=config)
        runtime.user_voice_status = "likely_user"
        runtime.user_voice_confidence = 0.84
        runtime.user_voice_checked_at = (
            datetime.now(timezone.utc) - timedelta(seconds=8)
        ).isoformat().replace("+00:00", "Z")
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
                        person_count=1,
                        looking_toward_device=True,
                        hand_or_object_near_camera=False,
                        body_pose=SocialBodyPose.UPRIGHT,
                    )
                ]
            ),
            pir_monitor=FakePirMonitor(events=[True], level=True),
            audio_observer=FakeAudioObserver(
                SocialAudioObservation(
                    speech_detected=False,
                    room_quiet=True,
                    recent_speech_age_s=12.0,
                )
            ),
            portrait_match_provider=FakePortraitMatchProvider(
                PortraitMatchObservation(
                    checked_at=1.9,
                    state="likely_reference_user",
                    matches_reference_user=True,
                    confidence=0.89,
                    similarity_score=0.64,
                    live_face_count=1,
                    reference_face_count=1,
                    backend_name="fake_portrait_backend",
                )
            ),
            observation_handler=lambda facts, event_names: observations.append((facts, event_names)),
            emit=lambda _line: None,
            clock=lambda: 2.0,
        )

        coordinator.tick()

        self.assertEqual(len(observations), 1)
        facts, events = observations[0]
        self.assertIn("portrait_match.matches_reference_user", events)
        self.assertEqual(facts["portrait_match"]["state"], "likely_reference_user")
        self.assertEqual(facts["portrait_match"]["policy_recommendation"], "calm_personalization_only")
        self.assertEqual(facts["known_user_hint"]["state"], "likely_main_user_multimodal")
        self.assertEqual(
            facts["known_user_hint"]["source"],
            "voice_profile_plus_portrait_match_plus_single_visible_person_context",
        )
        self.assertEqual(facts["known_user_hint"]["portrait_match_state"], "likely_reference_user")
        self.assertTrue(facts["known_user_hint"]["matches_main_user"])

    def test_coordinator_promotes_identity_fusion_after_repeated_stable_ticks(self) -> None:
        config = TwinrConfig(
            proactive_enabled=True,
            proactive_capture_interval_s=1.0,
            proactive_motion_window_s=20.0,
        )
        runtime = TwinrRuntime(config=config)
        runtime.user_voice_status = "likely_user"
        runtime.user_voice_confidence = 0.84
        runtime.user_voice_checked_at = (
            datetime.now(timezone.utc) - timedelta(seconds=6)
        ).isoformat().replace("+00:00", "Z")
        observations: list[tuple[dict[str, object], tuple[str, ...]]] = []
        clock = FakeClock(2.0)
        coordinator = ProactiveCoordinator(
            config=config,
            runtime=runtime,
            engine=SocialTriggerEngine(),
            trigger_handler=lambda _decision: True,
            vision_observer=FakeVisionObserver(
                [
                    SocialVisionObservation(
                        person_visible=True,
                        person_count=1,
                        primary_person_zone=SocialPersonZone.CENTER,
                        looking_toward_device=True,
                        engaged_with_device=True,
                        visual_attention_score=0.84,
                        body_pose=SocialBodyPose.UPRIGHT,
                    ),
                    SocialVisionObservation(
                        person_visible=True,
                        person_count=1,
                        primary_person_zone=SocialPersonZone.CENTER,
                        looking_toward_device=True,
                        engaged_with_device=True,
                        visual_attention_score=0.84,
                        body_pose=SocialBodyPose.UPRIGHT,
                    ),
                    SocialVisionObservation(
                        person_visible=True,
                        person_count=1,
                        primary_person_zone=SocialPersonZone.CENTER,
                        looking_toward_device=True,
                        engaged_with_device=True,
                        visual_attention_score=0.84,
                        body_pose=SocialBodyPose.UPRIGHT,
                    ),
                ]
            ),
            pir_monitor=FakePirMonitor(events=[True], level=True),
            audio_observer=FakeAudioObserver(
                SocialAudioObservation(
                    speech_detected=False,
                    room_quiet=True,
                    recent_speech_age_s=12.0,
                )
            ),
            portrait_match_provider=FakePortraitMatchProvider(
                PortraitMatchObservation(
                    checked_at=1.9,
                    state="likely_reference_user",
                    matches_reference_user=True,
                    confidence=0.89,
                    fused_confidence=0.93,
                    temporal_state="stable_match",
                    temporal_observation_count=3,
                    similarity_score=0.64,
                    live_face_count=1,
                    reference_face_count=1,
                    reference_image_count=3,
                    matched_user_id="main_user",
                    backend_name="fake_portrait_backend",
                )
            ),
            observation_handler=lambda facts, event_names: observations.append((facts, event_names)),
            emit=lambda _line: None,
            clock=clock,
        )

        coordinator.tick()
        clock.advance(2.0)
        coordinator.tick()
        clock.advance(2.0)
        coordinator.tick()

        self.assertGreaterEqual(len(observations), 2)
        facts, _events = observations[-1]
        self.assertEqual(facts["identity_fusion"]["state"], "stable_main_user_multimodal")
        self.assertEqual(facts["identity_fusion"]["temporal_state"], "stable_multimodal_match")
        self.assertEqual(facts["known_user_hint"]["state"], "likely_main_user_temporal_multimodal")
        self.assertEqual(
            facts["known_user_hint"]["source"],
            "voice_profile_plus_temporal_portrait_match_plus_track_history_plus_presence_session_memory",
        )
        self.assertEqual(facts["known_user_hint"]["identity_fusion_state"], "stable_main_user_multimodal")


if __name__ == "__main__":
    unittest.main()
