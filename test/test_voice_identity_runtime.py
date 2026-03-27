from array import array
import math
from pathlib import Path
import sys
import tempfile
from typing import cast
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.workflows.voice_identity_runtime import (
    build_voice_identity_profiles_event,
    update_household_voice_assessment_from_pcm,
)
from twinr.hardware.household_identity import (
    HouseholdIdentityFeedbackStore,
    HouseholdIdentityManager,
)
from twinr.hardware.household_voice_identity import (
    HouseholdVoiceIdentityMonitor,
    HouseholdVoiceIdentityStore,
)
from twinr.hardware.portrait_match import PortraitMatchObservation, PortraitMatchProvider
from twinr.orchestrator.voice_contracts import OrchestratorVoiceIdentityProfilesEvent


def _voice_sample_pcm_bytes(
    *,
    frequency_hz: float = 175.0,
    amplitude: float = 0.35,
    duration_s: float = 2.8,
) -> bytes:
    sample_rate = 16000
    total_frames = int(sample_rate * duration_s)
    frames = array("h")
    for index in range(total_frames):
        t = index / sample_rate
        envelope = min(
            1.0,
            index / (sample_rate * 0.2),
            (total_frames - index) / (sample_rate * 0.2),
        )
        sample = amplitude * envelope * (
            (0.70 * math.sin(2.0 * math.pi * frequency_hz * t))
            + (0.20 * math.sin(2.0 * math.pi * frequency_hz * 2.0 * t))
            + (0.10 * math.sin(2.0 * math.pi * (frequency_hz + 35.0) * t))
        )
        frames.append(max(-32767, min(32767, int(sample * 32767))))
    return frames.tobytes()


class _DummyPortraitProvider:
    def list_profiles(self) -> tuple[object, ...]:
        return ()

    def observe(self) -> PortraitMatchObservation:
        raise AssertionError("portrait observe should not be called in this test")


class _DummyRuntime:
    def __init__(self) -> None:
        self.last_voice_assessment: dict[str, object] | None = None

    def update_user_voice_assessment(self, **kwargs) -> None:
        self.last_voice_assessment = kwargs


class _DummyVoiceOrchestrator:
    def __init__(self) -> None:
        self.events: list[OrchestratorVoiceIdentityProfilesEvent] = []

    def notify_identity_profiles(self, event: OrchestratorVoiceIdentityProfilesEvent) -> None:
        self.events.append(event)


class _FakeLoop:
    def __init__(self, config: TwinrConfig, manager: HouseholdIdentityManager) -> None:
        self.config = config
        self.household_identity_manager = manager
        self.runtime = _DummyRuntime()
        self.voice_orchestrator = _DummyVoiceOrchestrator()
        self._latest_sensor_observation_facts: dict[str, dict[str, object]] = {
            "camera": {
                "person_visible": True,
                "person_count": 1,
            },
            "speaker_association": {
                "associated": True,
                "confidence": 0.9,
            },
            "audio_policy": {
                "background_media_likely": False,
                "speech_overlap_likely": False,
            },
        }
        self.emitted: list[str] = []
        self.traces: list[tuple[str, str, dict[str, object]]] = []

    def _try_emit(self, text: str) -> None:
        self.emitted.append(text)

    def emit(self, text: str) -> None:
        self.emitted.append(text)

    def _trace_event(self, name: str, *, kind: str, details: dict[str, object]) -> None:
        self.traces.append((name, kind, details))

    def _safe_error_text(self, exc: Exception) -> str:
        return str(exc)


class VoiceIdentityRuntimeTests(unittest.TestCase):
    def _make_manager(
        self,
        temp_dir: str,
        *,
        passive_update_min_confidence: float = 0.86,
    ) -> HouseholdIdentityManager:
        config = TwinrConfig(
            project_root=temp_dir,
            personality_dir="personality",
            openai_realtime_input_sample_rate=16000,
            audio_channels=1,
            voice_profile_passive_update_min_confidence=passive_update_min_confidence,
        )
        voice_monitor = HouseholdVoiceIdentityMonitor(
            store=HouseholdVoiceIdentityStore(Path(temp_dir) / "household_voice_identities.json"),
            primary_user_id="main_user",
            likely_threshold=0.72,
            uncertain_threshold=0.55,
            identity_margin=0.06,
            min_sample_ms=1200,
            max_enrollment_samples=6,
        )
        feedback_store = HouseholdIdentityFeedbackStore(Path(temp_dir) / "household_identity_feedback.json")
        return HouseholdIdentityManager(
            config=config,
            portrait_provider=cast(PortraitMatchProvider, _DummyPortraitProvider()),
            voice_monitor=voice_monitor,
            feedback_store=feedback_store,
        )

    def test_passive_update_merges_enrolled_profile_and_syncs_gateway(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self._make_manager(temp_dir, passive_update_min_confidence=0.82)
            pcm_bytes = _voice_sample_pcm_bytes()
            manager.enroll_voice(
                pcm_bytes,
                sample_rate=16000,
                channels=1,
                user_id="main_user",
                display_name="Theo",
            )
            loop = _FakeLoop(manager.config, manager)
            before_event = build_voice_identity_profiles_event(loop)
            assert before_event is not None
            before_revision = before_event.revision

            assessment = update_household_voice_assessment_from_pcm(loop, pcm_bytes, source="wake")
            after_event = build_voice_identity_profiles_event(loop)
            assert assessment is not None
            assert after_event is not None
            sample_count = manager.voice_monitor.summary("main_user").sample_count

        self.assertEqual(assessment.status, "likely_user")
        self.assertEqual(sample_count, 2)
        self.assertNotEqual(before_revision, after_event.revision)
        self.assertEqual(len(loop.voice_orchestrator.events), 1)
        self.assertIn("voice_profile_passive_update=applied", loop.emitted)

    def test_passive_update_is_blocked_when_background_media_is_likely(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self._make_manager(temp_dir, passive_update_min_confidence=0.82)
            pcm_bytes = _voice_sample_pcm_bytes()
            manager.enroll_voice(
                pcm_bytes,
                sample_rate=16000,
                channels=1,
                user_id="main_user",
                display_name="Theo",
            )
            loop = _FakeLoop(manager.config, manager)
            loop._latest_sensor_observation_facts["audio_policy"]["background_media_likely"] = True

            assessment = update_household_voice_assessment_from_pcm(loop, pcm_bytes, source="wake")
            assert assessment is not None
            sample_count = manager.voice_monitor.summary("main_user").sample_count

        self.assertEqual(assessment.status, "likely_user")
        self.assertEqual(sample_count, 1)
        self.assertEqual(loop.voice_orchestrator.events, [])


if __name__ == "__main__":
    unittest.main()
