from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.hardware.audio import AmbientAudioLevelSample
from twinr.proactive import (
    AmbientAudioObservationProvider,
    OpenAIVisionObservationProvider,
    ProactiveCoordinator,
    SocialBodyPose,
    SocialTriggerEngine,
    SocialVisionObservation,
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
        self.assertEqual(audio_observer.sampler.calls, 0)

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


if __name__ == "__main__":
    unittest.main()
