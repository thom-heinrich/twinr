from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.hardware.audio import AmbientAudioCaptureWindow, AmbientAudioLevelSample
from twinr.proactive import (
    PresenceSessionController,
    ProactiveAudioSnapshot,
    ProactiveCoordinator,
    SocialAudioObservation,
    SocialBodyPose,
    SocialTriggerEngine,
    SocialVisionObservation,
    WakewordMatch,
    WakewordOpenWakeWordFrameSpotter,
    WakewordOpenWakeWordSpotter,
    WakewordPhraseSpotter,
    normalize_detector_label,
    wakeword_primary_prompt,
)
from twinr.runtime import TwinrRuntime


class FakeBackend:
    def __init__(self, transcript: str | list[str] = "hey twinr") -> None:
        self.transcript = transcript
        self.calls: list[tuple[bytes, str | None, str | None]] = []

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        self.calls.append((audio_bytes, language, prompt))
        if isinstance(self.transcript, list):
            if not self.transcript:
                return ""
            return self.transcript.pop(0)
        return self.transcript


class FakeVisionObserver:
    def __init__(self, observations):
        self.observations = list(observations)

    def observe(self):
        observation = self.observations.pop(0)
        return SimpleNamespace(
            observation=observation,
            response_text="ok",
            response_id="resp_vision",
            request_id="req_vision",
            model="gpt-5.2",
        )


class FakePirMonitor:
    def __init__(self, *, events=None, level=False) -> None:
        self.events = list(events or [])
        self.level = level

    def poll(self, timeout=None):
        if not self.events:
            return None
        motion = self.events.pop(0)
        return SimpleNamespace(motion_detected=motion)

    def motion_detected(self):
        return self.level


class FakeAudioObserver:
    def __init__(self, snapshot: ProactiveAudioSnapshot) -> None:
        self.snapshot = snapshot

    def observe(self):
        return self.snapshot


class FakeWakewordSpotter:
    def __init__(self, match: WakewordMatch) -> None:
        self.match = match
        self.calls = 0

    def detect(self, capture) -> WakewordMatch:
        self.calls += 1
        return self.match


class FakeOpenWakeWordModel:
    def __init__(self, predictions: dict[str, float] | list[dict[str, float]], *, model_names: tuple[str, ...] = ("twinr",)) -> None:
        self.predictions = predictions
        self.models = {name: object() for name in model_names}
        self.predict_calls: list[tuple[object, dict[str, int], dict[str, float]]] = []
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def predict(self, samples, patience=None, threshold=None):
        self.predict_calls.append((samples, patience or {}, threshold or {}))
        if isinstance(self.predictions, list):
            if not self.predictions:
                return {}
            return self.predictions.pop(0)
        return self.predictions


class FakeClipOpenWakeWordModel:
    def __init__(self, clip_predictions: list[dict[str, float]], *, model_names: tuple[str, ...] = ("twinr",)) -> None:
        self.clip_predictions = list(clip_predictions)
        self.models = {name: object() for name in model_names}
        self.predict_clip_calls: list[tuple[object, int, int, dict[str, int], dict[str, float]]] = []
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def predict_clip(self, samples, *, padding=1, chunk_size=1280, patience=None, threshold=None):
        self.predict_clip_calls.append((samples, padding, chunk_size, patience or {}, threshold or {}))
        return list(self.clip_predictions)


class MutableClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class WakewordTests(unittest.TestCase):
    def test_presence_session_stays_armed_after_recent_visible_presence(self) -> None:
        controller = PresenceSessionController(
            presence_grace_s=600.0,
            motion_grace_s=120.0,
            speech_grace_s=45.0,
        )

        first = controller.observe(now=0.0, person_visible=True, motion_active=True, speech_detected=False)
        second = controller.observe(now=300.0, person_visible=None, motion_active=False, speech_detected=False)
        third = controller.observe(now=901.0, person_visible=None, motion_active=False, speech_detected=False)

        self.assertTrue(first.armed)
        self.assertEqual(first.reason, "person_visible")
        self.assertTrue(second.armed)
        self.assertEqual(second.reason, "recent_person_visible")
        self.assertFalse(third.armed)
        self.assertEqual(third.reason, "idle")

    def test_presence_session_id_stays_stable_until_idle_and_then_rotates(self) -> None:
        controller = PresenceSessionController(
            presence_grace_s=600.0,
            motion_grace_s=120.0,
            speech_grace_s=45.0,
        )

        first = controller.observe(now=0.0, person_visible=True, motion_active=True, speech_detected=False)
        second = controller.observe(now=300.0, person_visible=None, motion_active=False, speech_detected=False)
        third = controller.observe(now=901.0, person_visible=None, motion_active=False, speech_detected=False)
        fourth = controller.observe(now=902.0, person_visible=True, motion_active=True, speech_detected=False)

        self.assertTrue(first.armed)
        self.assertEqual(first.session_id, 1)
        self.assertTrue(second.armed)
        self.assertEqual(second.session_id, 1)
        self.assertFalse(third.armed)
        self.assertIsNone(third.session_id)
        self.assertTrue(fourth.armed)
        self.assertEqual(fourth.session_id, 2)

    def test_presence_session_uses_recent_follow_up_speech_for_resume_window(self) -> None:
        controller = PresenceSessionController(
            presence_grace_s=600.0,
            motion_grace_s=120.0,
            speech_grace_s=45.0,
        )

        controller.observe(
            now=0.0,
            person_visible=True,
            motion_active=False,
            speech_detected=False,
        )
        resumed = controller.observe(
            now=5.0,
            person_visible=None,
            motion_active=False,
            speech_detected=False,
            recent_speech_age_s=0.4,
            presence_audio_active=False,
            recent_follow_up_speech=True,
            room_busy_or_overlapping=False,
            quiet_window_open=False,
            barge_in_recent=True,
            speaker_direction_stable=True,
            mute_blocks_voice_capture=False,
            resume_window_open=True,
            device_runtime_mode="audio_ready",
            transport_reason=None,
        )

        self.assertTrue(resumed.armed)
        self.assertEqual(resumed.reason, "follow_up_speech_while_recently_present")
        self.assertAlmostEqual(resumed.last_speech_age_s or 0.0, 0.4, delta=0.05)
        self.assertTrue(resumed.recent_follow_up_speech)
        self.assertTrue(resumed.barge_in_recent)
        self.assertTrue(resumed.resume_window_open)
        self.assertEqual(resumed.device_runtime_mode, "audio_ready")

    def test_wakeword_phrase_spotter_matches_prefix_and_remaining_text(self) -> None:
        backend = FakeBackend("Hallo Twinner bist du da")
        spotter = WakewordPhraseSpotter(
            backend=backend,
            phrases=("hey twinr", "hey twinna", "hallo twinner", "twinr"),
            language="de",
        )

        match = spotter.detect(
            AmbientAudioCaptureWindow(
                sample=AmbientAudioLevelSample(
                    duration_ms=1600,
                    chunk_count=8,
                    active_chunk_count=5,
                    average_rms=900,
                    peak_rms=1600,
                    active_ratio=0.62,
                ),
                pcm_bytes=b"\x01\x02" * 800,
                sample_rate=16000,
                channels=1,
            )
        )

        self.assertTrue(match.detected)
        self.assertEqual(match.matched_phrase, "hallo twinner")
        self.assertEqual(match.remaining_text, "bist du da")
        self.assertEqual(len(backend.calls), 1)

    def test_wakeword_phrase_spotter_matches_trailing_greeting_variant(self) -> None:
        backend = FakeBackend("Twina hallo")
        spotter = WakewordPhraseSpotter(
            backend=backend,
            phrases=("twinr hallo", "twina hallo", "twinner hey"),
            language="de",
        )

        match = spotter.detect(
            AmbientAudioCaptureWindow(
                sample=AmbientAudioLevelSample(
                    duration_ms=1600,
                    chunk_count=8,
                    active_chunk_count=4,
                    average_rms=780,
                    peak_rms=1500,
                    active_ratio=0.5,
                ),
                pcm_bytes=b"\x01\x02" * 800,
                sample_rate=16000,
                channels=1,
            )
        )

        self.assertTrue(match.detected)
        self.assertEqual(match.matched_phrase, "twina hallo")
        self.assertEqual(match.remaining_text, "")
        self.assertEqual(len(backend.calls), 1)

    def test_wakeword_phrase_spotter_ignores_short_leading_filler_words(self) -> None:
        backend = FakeBackend("Ja hallo Twina bist du da")
        spotter = WakewordPhraseSpotter(
            backend=backend,
            phrases=("hallo twina", "hey twinr", "twina"),
            language="de",
        )

        match = spotter.detect(
            AmbientAudioCaptureWindow(
                sample=AmbientAudioLevelSample(
                    duration_ms=1600,
                    chunk_count=8,
                    active_chunk_count=4,
                    average_rms=820,
                    peak_rms=1550,
                    active_ratio=0.5,
                ),
                pcm_bytes=b"\x01\x02" * 800,
                sample_rate=16000,
                channels=1,
            )
        )

        self.assertTrue(match.detected)
        self.assertEqual(match.matched_phrase, "hallo twina")
        self.assertEqual(match.remaining_text, "bist du da")
        self.assertEqual(len(backend.calls), 1)

    def test_wakeword_phrase_spotter_retries_after_prompt_contamination(self) -> None:
        backend = FakeBackend(
            [
                "If a name sounds close to one of these, use that spelling exactly.",
                "Hallo Twina",
            ]
        )
        spotter = WakewordPhraseSpotter(
            backend=backend,
            phrases=("hallo twina", "hey twinr", "twina"),
            language="de",
        )

        match = spotter.detect(
            AmbientAudioCaptureWindow(
                sample=AmbientAudioLevelSample(
                    duration_ms=1600,
                    chunk_count=8,
                    active_chunk_count=4,
                    average_rms=820,
                    peak_rms=1550,
                    active_ratio=0.5,
                ),
                pcm_bytes=b"\x01\x02" * 800,
                sample_rate=16000,
                channels=1,
            )
        )

        self.assertTrue(match.detected)
        self.assertEqual(match.matched_phrase, "hallo twina")
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(backend.calls[0][2], wakeword_primary_prompt(("hallo twina", "hey twinr", "twina")))
        self.assertIsNone(backend.calls[1][2])

    def test_openwakeword_spotter_detects_local_model_hit(self) -> None:
        model = FakeOpenWakeWordModel({"twinr": 0.91})
        spotter = WakewordOpenWakeWordSpotter(
            wakeword_models=("twinr",),
            phrases=("hey twinr", "twinr"),
            threshold=0.5,
            transcribe_on_detect=False,
            model_factory=lambda **_kwargs: model,
        )

        match = spotter.detect(
            AmbientAudioCaptureWindow(
                sample=AmbientAudioLevelSample(
                    duration_ms=640,
                    chunk_count=8,
                    active_chunk_count=5,
                    average_rms=900,
                    peak_rms=1800,
                    active_ratio=0.62,
                ),
                pcm_bytes=b"\x01\x02" * 1600,
                sample_rate=16000,
                channels=1,
            )
        )

        self.assertTrue(match.detected)
        self.assertEqual(match.backend, "openwakeword")
        self.assertEqual(match.matched_phrase, "twinr")
        self.assertEqual(match.detector_label, "twinr")
        self.assertAlmostEqual(match.score or 0.0, 0.91)
        self.assertEqual(model.reset_calls, 1)
        self.assertEqual(len(model.predict_calls), 1)
        self.assertEqual(model.predict_calls[0][1], {})
        self.assertEqual(model.predict_calls[0][2], {})

    def test_openwakeword_spotter_passes_custom_verifier_config_to_model_factory(self) -> None:
        recorded: dict[str, object] = {}
        model = FakeOpenWakeWordModel({"twinr_v1": 0.91}, model_names=("twinr_v1",))

        def factory(**kwargs):
            recorded.update(kwargs)
            return model

        WakewordOpenWakeWordSpotter(
            wakeword_models=("twinr_v1.onnx",),
            phrases=("twinr",),
            custom_verifier_models={"twinr_v1": "/tmp/twinr_v1.verifier.pkl"},
            custom_verifier_threshold=0.23,
            threshold=0.5,
            transcribe_on_detect=False,
            model_factory=factory,
        )

        self.assertEqual(
            recorded["custom_verifier_models"],
            {"twinr_v1": "/tmp/twinr_v1.verifier.pkl"},
        )
        self.assertEqual(recorded["custom_verifier_threshold"], 0.23)

    def test_openwakeword_frame_spotter_detects_streaming_hit_without_transcription(self) -> None:
        model = FakeOpenWakeWordModel(
            [
                {"twinr_multivoice_v2": 0.0},
                {"twinr_multivoice_v2": 0.86},
            ],
            model_names=("twinr_multivoice_v2",),
        )
        spotter = WakewordOpenWakeWordFrameSpotter(
            wakeword_models=("twinr_multivoice_v2",),
            phrases=("twinr", "twinna"),
            threshold=0.5,
            patience_frames=1,
            model_factory=lambda **_kwargs: model,
        )

        first = spotter.process_pcm_frame(b"\x01\x02" * 1280)
        second = spotter.process_pcm_frame(b"\x01\x02" * 1280)

        self.assertIsNone(first)
        self.assertIsNotNone(second)
        self.assertTrue(second.detected)
        self.assertEqual(second.matched_phrase, "twinr")
        self.assertEqual(second.detector_label, "twinr_multivoice_v2")
        self.assertAlmostEqual(second.score or 0.0, 0.86)
        self.assertEqual(len(model.predict_calls), 2)
        self.assertEqual(model.predict_calls[0][1], {})
        self.assertEqual(model.predict_calls[0][2], {})

    def test_openwakeword_frame_spotter_passes_custom_verifier_config_to_model_factory(self) -> None:
        recorded: dict[str, object] = {}
        model = FakeOpenWakeWordModel({"twinr_v1": 0.91}, model_names=("twinr_v1",))

        def factory(**kwargs):
            recorded.update(kwargs)
            return model

        WakewordOpenWakeWordFrameSpotter(
            wakeword_models=("twinr_v1.onnx",),
            phrases=("twinr",),
            custom_verifier_models={"twinr_v1": "/tmp/twinr_v1.verifier.pkl"},
            custom_verifier_threshold=0.19,
            threshold=0.5,
            patience_frames=1,
            model_factory=factory,
        )

        self.assertEqual(
            recorded["custom_verifier_models"],
            {"twinr_v1": "/tmp/twinr_v1.verifier.pkl"},
        )
        self.assertEqual(recorded["custom_verifier_threshold"], 0.19)

    def test_openwakeword_frame_spotter_requires_smoothed_activation(self) -> None:
        model = FakeOpenWakeWordModel(
            [
                {"hey_twinna": 0.92},
                {"hey_twinna": 0.0},
                {"hey_twinna": 0.88},
            ],
            model_names=("hey_twinna",),
        )
        spotter = WakewordOpenWakeWordFrameSpotter(
            wakeword_models=("hey_twinna",),
            phrases=("hey twinna", "twinna"),
            threshold=0.7,
            activation_samples=3,
            patience_frames=1,
            model_factory=lambda **_kwargs: model,
        )

        first = spotter.process_pcm_frame(b"\x01\x02" * 1280)
        second = spotter.process_pcm_frame(b"\x01\x02" * 1280)
        third = spotter.process_pcm_frame(b"\x01\x02" * 1280)

        self.assertIsNone(first)
        self.assertIsNone(second)
        self.assertIsNone(third)

    def test_openwakeword_frame_spotter_latches_until_average_deactivates(self) -> None:
        model = FakeOpenWakeWordModel(
            [
                {"hey_twinna": 0.8},
                {"hey_twinna": 0.82},
                {"hey_twinna": 0.84},
                {"hey_twinna": 0.83},
                {"hey_twinna": 0.02},
                {"hey_twinna": 0.01},
                {"hey_twinna": 0.0},
                {"hey_twinna": 0.9},
                {"hey_twinna": 0.92},
                {"hey_twinna": 0.94},
            ],
            model_names=("hey_twinna",),
        )
        spotter = WakewordOpenWakeWordFrameSpotter(
            wakeword_models=("hey_twinna",),
            phrases=("hey twinna", "twinna"),
            threshold=0.5,
            activation_samples=3,
            deactivation_threshold=0.2,
            patience_frames=1,
            model_factory=lambda **_kwargs: model,
        )

        detections = [
            spotter.process_pcm_frame(b"\x01\x02" * 1280)
            for _ in range(10)
        ]

        fired = [item for item in detections if item is not None]
        self.assertEqual(len(fired), 2)
        self.assertEqual(fired[0].matched_phrase, "hey twinna")
        self.assertEqual(fired[1].matched_phrase, "hey twinna")

    def test_openwakeword_spotter_uses_peak_clip_score(self) -> None:
        model = FakeClipOpenWakeWordModel(
            [
                {"hey_twinna": 0.0},
                {"hey_twinna": 0.87},
                {"hey_twinna": 0.12},
            ],
            model_names=("hey_twinna",),
        )
        spotter = WakewordOpenWakeWordSpotter(
            wakeword_models=("hey_twinna",),
            phrases=("hey twinna", "twinna"),
            threshold=0.5,
            transcribe_on_detect=False,
            model_factory=lambda **_kwargs: model,
        )

        match = spotter.detect(
            AmbientAudioCaptureWindow(
                sample=AmbientAudioLevelSample(
                    duration_ms=900,
                    chunk_count=9,
                    active_chunk_count=6,
                    average_rms=960,
                    peak_rms=1800,
                    active_ratio=0.66,
                ),
                pcm_bytes=b"\x01\x02" * 2000,
                sample_rate=16000,
                channels=1,
            )
        )

        self.assertTrue(match.detected)
        self.assertEqual(match.matched_phrase, "hey twinna")
        self.assertEqual(match.detector_label, "hey_twinna")
        self.assertAlmostEqual(match.score or 0.0, 0.87)
        self.assertEqual(model.reset_calls, 1)
        self.assertEqual(len(model.predict_clip_calls), 1)
        self.assertEqual(model.predict_clip_calls[0][1], 1)
        self.assertEqual(model.predict_clip_calls[0][2], 1280)
        self.assertEqual(model.predict_clip_calls[0][3], {})
        self.assertEqual(model.predict_clip_calls[0][4], {})

    def test_openwakeword_spotter_transcribes_after_local_hit(self) -> None:
        backend = FakeBackend("Twinr bist du da")
        model = FakeOpenWakeWordModel({"twinr": 0.88})
        spotter = WakewordOpenWakeWordSpotter(
            wakeword_models=("twinr",),
            phrases=("hey twinr", "twinr"),
            threshold=0.5,
            backend=backend,
            transcribe_on_detect=True,
            language="de",
            model_factory=lambda **_kwargs: model,
        )

        match = spotter.detect(
            AmbientAudioCaptureWindow(
                sample=AmbientAudioLevelSample(
                    duration_ms=900,
                    chunk_count=9,
                    active_chunk_count=6,
                    average_rms=1040,
                    peak_rms=1900,
                    active_ratio=0.66,
                ),
                pcm_bytes=b"\x01\x02" * 2000,
                sample_rate=16000,
                channels=1,
            )
        )

        self.assertTrue(match.detected)
        self.assertEqual(match.backend, "openwakeword")
        self.assertEqual(match.matched_phrase, "twinr")
        self.assertEqual(match.remaining_text, "bist du da")
        self.assertEqual(match.transcript, "Twinr bist du da")
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0][1], "de")
        self.assertEqual(backend.calls[0][2], wakeword_primary_prompt(("hey twinr", "twinr")))

    def test_openwakeword_spotter_requires_transcript_confirmation_when_enabled(self) -> None:
        backend = FakeBackend("irgendwas anderes")
        model = FakeOpenWakeWordModel({"hey_twinna": 0.88}, model_names=("hey_twinna",))
        spotter = WakewordOpenWakeWordSpotter(
            wakeword_models=("hey_twinna",),
            phrases=("hey twinna", "twinna"),
            threshold=0.5,
            backend=backend,
            transcribe_on_detect=True,
            language="de",
            model_factory=lambda **_kwargs: model,
        )

        match = spotter.detect(
            AmbientAudioCaptureWindow(
                sample=AmbientAudioLevelSample(
                    duration_ms=900,
                    chunk_count=9,
                    active_chunk_count=6,
                    average_rms=1040,
                    peak_rms=1900,
                    active_ratio=0.66,
                ),
                pcm_bytes=b"\x01\x02" * 2000,
                sample_rate=16000,
                channels=1,
            )
        )

        self.assertFalse(match.detected)
        self.assertEqual(match.backend, "openwakeword")
        self.assertEqual(match.matched_phrase, "hey twinna")
        self.assertEqual(match.transcript, "irgendwas anderes")
        self.assertEqual(match.normalized_transcript, "irgendwas anderes")
        self.assertEqual(len(backend.calls), 1)

    def test_normalize_detector_label_strips_decorative_words(self) -> None:
        self.assertEqual(normalize_detector_label("twinr_multivoice_v2"), "twinr")

    def test_coordinator_routes_detected_wakeword_to_handler(self) -> None:
        config = TwinrConfig(
            wakeword_enabled=True,
            proactive_capture_interval_s=1.0,
            proactive_motion_window_s=20.0,
            wakeword_attempt_cooldown_s=1.0,
            wakeword_min_active_ratio=0.08,
            wakeword_min_active_chunks=2,
        )
        runtime = TwinrRuntime(config=config)
        clock = MutableClock(0.0)
        handled: list[str] = []
        coordinator = ProactiveCoordinator(
            config=config,
            runtime=runtime,
            engine=SocialTriggerEngine(),
            trigger_handler=lambda _decision: True,
            vision_observer=FakeVisionObserver(
                [SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.UPRIGHT)]
            ),
            pir_monitor=FakePirMonitor(events=[True], level=True),
            audio_observer=FakeAudioObserver(
                ProactiveAudioSnapshot(
                    observation=SocialAudioObservation(speech_detected=False),
                    sample=AmbientAudioLevelSample(
                        duration_ms=1800,
                        chunk_count=17,
                        active_chunk_count=3,
                        average_rms=527,
                        peak_rms=2662,
                        active_ratio=3 / 17,
                    ),
                    pcm_bytes=b"\x01\x02" * 900,
                    sample_rate=16000,
                    channels=1,
                )
            ),
            presence_session=PresenceSessionController(
                presence_grace_s=600.0,
                motion_grace_s=120.0,
                speech_grace_s=45.0,
            ),
            wakeword_spotter=FakeWakewordSpotter(
                WakewordMatch(
                    detected=True,
                    transcript="hey twinr wie spaet ist es",
                    matched_phrase="hey twinr",
                    remaining_text="wie spaet ist es",
                    normalized_transcript="hey twinr wie spaet ist es",
                )
            ),
            wakeword_handler=lambda match: handled.append(match.remaining_text) or True,
            emit=lambda _line: None,
            clock=clock,
        )

        result = coordinator.tick()

        self.assertIsNotNone(result.wakeword_match)
        self.assertEqual(result.wakeword_match.matched_phrase, "hey twinr")
        self.assertEqual(handled, ["wie spaet ist es"])
        events = runtime.ops_events.tail(limit=10)
        self.assertTrue(any(entry.get("event") == "wakeword_attempted" for entry in events))

    def test_coordinator_logs_transcript_preview_for_failed_wakeword_attempt(self) -> None:
        config = TwinrConfig(
            wakeword_enabled=True,
            proactive_capture_interval_s=1.0,
            proactive_motion_window_s=20.0,
            wakeword_attempt_cooldown_s=1.0,
            wakeword_min_active_ratio=0.08,
            wakeword_min_active_chunks=2,
        )
        runtime = TwinrRuntime(config=config)
        coordinator = ProactiveCoordinator(
            config=config,
            runtime=runtime,
            engine=SocialTriggerEngine(),
            trigger_handler=lambda _decision: True,
            vision_observer=FakeVisionObserver(
                [SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.UPRIGHT)]
            ),
            pir_monitor=FakePirMonitor(events=[True], level=True),
            audio_observer=FakeAudioObserver(
                ProactiveAudioSnapshot(
                    observation=SocialAudioObservation(speech_detected=False),
                    sample=AmbientAudioLevelSample(
                        duration_ms=1800,
                        chunk_count=17,
                        active_chunk_count=3,
                        average_rms=527,
                        peak_rms=2662,
                        active_ratio=3 / 17,
                    ),
                    pcm_bytes=b"\x01\x02" * 900,
                    sample_rate=16000,
                    channels=1,
                )
            ),
            presence_session=PresenceSessionController(
                presence_grace_s=600.0,
                motion_grace_s=120.0,
                speech_grace_s=45.0,
            ),
            wakeword_spotter=FakeWakewordSpotter(
                WakewordMatch(
                    detected=False,
                    transcript="ja hallo twina",
                    matched_phrase=None,
                    remaining_text="",
                    normalized_transcript="ja hallo twina",
                )
            ),
            wakeword_handler=lambda _match: True,
            emit=lambda _line: None,
            clock=MutableClock(0.0),
        )

        result = coordinator.tick()

        self.assertIsNone(result.wakeword_match)
        events = runtime.ops_events.tail(limit=10)
        attempted = [entry for entry in events if entry.get("event") == "wakeword_attempted"][-1]
        self.assertEqual(attempted["data"]["transcript_preview"], "ja hallo twina")
        self.assertEqual(attempted["data"]["normalized_transcript_preview"], "ja hallo twina")


if __name__ == "__main__":
    unittest.main()
