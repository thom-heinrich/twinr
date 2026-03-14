from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
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
    SocialTriggerEngine,
    SocialVisionObservation,
    WakewordMatch,
    WakewordStreamDetection,
    WakewordPhraseSpotter,
    build_default_proactive_monitor,
    parse_vision_observation_text,
)
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

    def respond_to_images_with_metadata(self, prompt, *, images, allow_web_search=None):
        self.calls.append((prompt, list(images), allow_web_search))
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
                    "looking_toward_device=yes",
                    "body_pose=slumped",
                    "smiling=no",
                    "hand_or_object_near_camera=yes",
                ]
            )
        )

        self.assertTrue(observation.person_visible)
        self.assertTrue(observation.looking_toward_device)
        self.assertEqual(observation.body_pose, SocialBodyPose.SLUMPED)
        self.assertFalse(observation.smiling)
        self.assertTrue(observation.hand_or_object_near_camera)

    def test_openai_vision_observer_parses_backend_response(self) -> None:
        backend = FakeBackend(
            "\n".join(
                [
                    "person_visible=yes",
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
        self.assertTrue(snapshot.observation.person_visible)
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

    def test_coordinator_triggers_person_returned_after_absence(self) -> None:
        config = TwinrConfig(
            proactive_enabled=True,
            proactive_capture_interval_s=1.0,
            proactive_motion_window_s=20.0,
        )
        runtime = TwinrRuntime(config=config)
        clock = MutableClock(0.0)
        pir_monitor = FakePirMonitor(level=False)
        handled: list[str] = []
        coordinator = ProactiveCoordinator(
            config=config,
            runtime=runtime,
            engine=SocialTriggerEngine(user_name="Thom"),
            trigger_handler=lambda decision: handled.append(decision.trigger_id) or True,
            vision_observer=FakeVisionObserver(
                [
                    SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.UPRIGHT),
                ]
            ),
            pir_monitor=pir_monitor,
            emit=lambda _line: None,
            clock=clock,
        )

        coordinator.tick()
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
        self.assertEqual(wakeword_stream.presence_snapshots[-1].reason, "person_visible")

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
        self.assertIsNone(result.decision)
        self.assertEqual(vision_observer.calls, 0)
        self.assertEqual(audio_observer.calls, 0)
        self.assertEqual(handled, [])

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
            pir_monitor = FakePirMonitor(level=False)
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
                    ]
                ),
                pir_monitor=pir_monitor,
                audio_observer=FakeAudioObserver(SocialAudioObservation()),
                emit=lambda _line: None,
                clock=clock,
            )

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

    def test_coordinator_dispatches_absence_path_trigger_when_inspection_is_skipped(self) -> None:
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

        self.assertEqual(handled, ["possible_fall"])
        self.assertIsNotNone(result.decision)
        self.assertEqual(result.decision.trigger_id, "possible_fall")
        self.assertFalse(result.inspected)
        trigger_events = [entry for entry in events if entry.get("event") == "proactive_trigger_detected"]
        self.assertEqual(trigger_events[-1]["data"]["trigger"], "possible_fall")
        observation_events = [entry for entry in events if entry.get("event") == "proactive_observation"]
        self.assertEqual(observation_events[-1]["data"]["inspected"], False)

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
                        SocialVisionObservation(person_visible=False, body_pose=SocialBodyPose.UNKNOWN),
                        SocialVisionObservation(person_visible=False, body_pose=SocialBodyPose.UNKNOWN),
                        SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.SLUMPED),
                        SocialVisionObservation(person_visible=False, body_pose=SocialBodyPose.UNKNOWN),
                        SocialVisionObservation(person_visible=False, body_pose=SocialBodyPose.UNKNOWN),
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
            clock.now = 3.0
            coordinator.tick()
            clock.now = 7.3
            first = coordinator.tick()
            self.assertIsNotNone(first.decision)
            self.assertEqual(first.decision.trigger_id, "possible_fall")

            clock.now = 70.0
            pir_monitor.events = [True]
            coordinator.tick()
            clock.now = 72.5
            coordinator.tick()
            clock.now = 76.9
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

        with patch("twinr.proactive.service.configured_pir_monitor", return_value=FakePirMonitor()):
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
            audio_observer=FakeAudioObserver(SocialAudioObservation(speech_detected=True)),
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
        self.assertEqual(second_events, ())
        self.assertGreaterEqual(second_facts["camera"]["person_visible_for_s"], 6.0)


if __name__ == "__main__":
    unittest.main()
