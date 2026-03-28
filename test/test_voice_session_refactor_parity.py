from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.orchestrator.voice_contracts import (
    OrchestratorVoiceAudioFrame,
    OrchestratorVoiceHelloRequest,
    OrchestratorVoiceRuntimeStateEvent,
)
from twinr.orchestrator.voice_session import EdgeOrchestratorVoiceSession
from twinr.orchestrator.voice_session_impl import EdgeOrchestratorVoiceSessionImpl
from test.test_orchestrator import (
    _MinDurationWakePhraseSpotter,
    _TranscriptOnlyNonWakePhraseSpotter,
    _pcm_frame,
)

_EXPECTED_GOLDEN_DIGESTS = {
    "waiting_wake_confirmed": "d228aabe13c6978118935edec1113f280396e027788b40e9326915e96063ac20",
    "follow_up_transcript_committed": "f0916410adf74722ae1063635b202a64227fd51b47ea536c212c6600cf8b1860",
    "stage1_capture_window": "64d6328afd2e751081e4d329e25bd65241602a8d94e2ad762e14a95bdf24a2be",
}


def _normalize_payload(value):
    if isinstance(value, dict):
        return {str(key): _normalize_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_payload(item) for item in value]
    if isinstance(value, bytes):
        return {"len": len(value), "prefix": list(value[:8])}
    return value


def _payload_digest(payload: object) -> str:
    rendered = json.dumps(
        _normalize_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return sha256(rendered.encode("utf-8")).hexdigest()


class VoiceSessionRefactorParityTests(unittest.TestCase):
    def _waiting_wake_confirmed_payload(
        self,
        *,
        session_cls=EdgeOrchestratorVoiceSession,
    ) -> dict[str, object]:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            spotter = _MinDurationWakePhraseSpotter(
                min_duration_ms=100,
                remaining_text="schau mal im web",
            )
            session = session_cls(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "Twinna schau mal im web"),
                wake_phrase_spotter=spotter,
            )

            hello_events = session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    trace_id="trace-fixed",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                    state_attested=True,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(0))
            )

        return {
            "hello_events": hello_events,
            "frame_events": [first, second],
            "state": session._state,
            "follow_up_allowed": session._follow_up_allowed,
            "pending": session._pending_transcript_utterance is not None,
            "spotter_capture_durations_ms": list(spotter.capture_durations_ms),
        }

    def _follow_up_transcript_committed_payload(
        self,
        *,
        session_cls=EdgeOrchestratorVoiceSession,
    ) -> dict[str, object]:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_follow_up_window_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            backend_calls: list[dict[str, object]] = []

            def _transcribe(*args, **kwargs):
                del args
                backend_calls.append({"prompt": kwargs.get("prompt")})
                return "wie geht es dir"

            session = session_cls(
                config,
                backend=SimpleNamespace(transcribe=_transcribe),
                wake_phrase_spotter=_TranscriptOnlyNonWakePhraseSpotter("wie geht es dir"),
            )

            hello_events = session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            runtime_events = session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="follow_up_open",
                    detail="voice_activation",
                    follow_up_allowed=True,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(0))
            )

        return {
            "hello_events": hello_events,
            "runtime_events": runtime_events,
            "frame_events": [first, second],
            "state": session._state,
            "follow_up_allowed": session._follow_up_allowed,
            "backend_calls": backend_calls,
        }

    def _stage1_capture_window_payload(
        self,
        *,
        session_cls=EdgeOrchestratorVoiceSession,
    ) -> dict[str, object]:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=5,
                voice_orchestrator_history_ms=2400,
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
            )
            session = session_cls(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=_MinDurationWakePhraseSpotter(min_duration_ms=100),
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                    state_attested=True,
                )
            )
            for value in (1, 2, 7, 7, 0, 0, 0):
                session._remember_frame(_pcm_frame(value))
            capture = session._recent_remote_asr_stage1_capture()

        return {
            "duration_ms": capture.sample.duration_ms,
            "chunk_count": capture.sample.chunk_count,
            "active_chunk_count": capture.sample.active_chunk_count,
            "average_rms": capture.sample.average_rms,
            "peak_rms": capture.sample.peak_rms,
            "active_ratio": capture.sample.active_ratio,
            "pcm_prefix": bytes(capture.pcm_bytes[:12]),
            "pcm_len": len(capture.pcm_bytes),
        }

    def test_public_wrapper_preserves_class_module(self) -> None:
        self.assertEqual(
            EdgeOrchestratorVoiceSession.__module__,
            "twinr.orchestrator.voice_session",
        )

    def test_golden_master_hashes_remain_stable(self) -> None:
        cases = {
            "waiting_wake_confirmed": self._waiting_wake_confirmed_payload(),
            "follow_up_transcript_committed": self._follow_up_transcript_committed_payload(),
            "stage1_capture_window": self._stage1_capture_window_payload(),
        }
        for name, payload in cases.items():
            with self.subTest(case=name):
                self.assertEqual(_payload_digest(payload), _EXPECTED_GOLDEN_DIGESTS[name])

    def test_public_wrapper_matches_internal_implementation_payloads(self) -> None:
        cases = (
            ("waiting_wake_confirmed", self._waiting_wake_confirmed_payload),
            ("follow_up_transcript_committed", self._follow_up_transcript_committed_payload),
            ("stage1_capture_window", self._stage1_capture_window_payload),
        )
        for name, builder in cases:
            with self.subTest(case=name):
                wrapped = _normalize_payload(builder(session_cls=EdgeOrchestratorVoiceSession))
                internal = _normalize_payload(builder(session_cls=EdgeOrchestratorVoiceSessionImpl))
                self.assertEqual(wrapped, internal)
