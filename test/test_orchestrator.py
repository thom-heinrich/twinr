from array import array
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
from threading import Event, Thread
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.contracts import AgentToolCall, ToolCallingTurnResponse
from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.voice_profile import extract_voice_embedding_from_pcm16
from twinr.orchestrator.acks import ack_text_for_id
from twinr.orchestrator.client import OrchestratorWebSocketClient
from twinr.orchestrator.contracts import OrchestratorToolResponse, OrchestratorTurnCompleteEvent
from twinr.orchestrator.contracts import OrchestratorTurnRequest
from twinr.orchestrator.remote_tool_timeout import (
    DEFAULT_REMOTE_TOOL_TIMEOUT_SECONDS,
    REMOTE_TOOL_TIMEOUT_ENV,
    read_remote_tool_timeout_seconds,
)
from twinr.orchestrator.remote_asr import RemoteAsrBackendAdapter, _encode_multipart_form
from twinr.orchestrator.remote_asr_service import (
    RemoteAsrHttpService,
    remote_asr_url_targets_local_orchestrator,
)
from twinr.orchestrator.server import EdgeOrchestratorServer, create_app as create_orchestrator_app
from twinr.orchestrator.session import EdgeOrchestratorSession, RemoteToolBridge
from twinr.orchestrator.voice_activation import (
    DEFAULT_VOICE_ACTIVATION_PHRASES,
    VoiceActivationMatch,
    match_voice_activation_transcript,
)
from twinr.orchestrator.voice_client import OrchestratorVoiceWebSocketClient
from twinr.orchestrator.voice_contracts import (
    OrchestratorVoiceAudioFrame,
    OrchestratorVoiceHelloRequest,
    OrchestratorVoiceIdentityProfile,
    OrchestratorVoiceIdentityProfilesEvent,
    OrchestratorVoiceRuntimeStateEvent,
)
from twinr.orchestrator.voice_runtime_intent import VoiceRuntimeIntentContext
from twinr.orchestrator.voice_session import EdgeOrchestratorVoiceSession
from twinr.ops.usage import TokenUsage


class _FakeDecisionProvider:
    def decide(self, prompt: str, *, conversation=None, instructions=None):
        del prompt, conversation, instructions
        return type(
            "Decision",
            (),
            {
                "action": "handoff",
                "spoken_ack": "Ich schaue kurz nach.",
                "spoken_reply": None,
                "kind": "search",
                "goal": "Check the weather.",
                "allow_web_search": True,
                "response_id": "decision-1",
                "request_id": "req-1",
                "model": "gpt-4o-mini",
                "token_usage": None,
            },
        )()


class _FakeSpecialistProvider:
    def __init__(self) -> None:
        self.start_calls: list[dict[str, object]] = []
        self.continue_calls: list[dict[str, object]] = []

    def start_turn_streaming(
        self,
        prompt: str,
        *,
        conversation=None,
        instructions=None,
        tool_schemas=(),
        allow_web_search=None,
        on_text_delta=None,
    ) -> ToolCallingTurnResponse:
        del prompt, conversation, instructions, tool_schemas, allow_web_search, on_text_delta
        self.start_calls.append({})
        return ToolCallingTurnResponse(
            text="",
            tool_calls=(
                AgentToolCall(
                    name="search_live_info",
                    call_id="call-search-1",
                    arguments={"question": "Wie wird das Wetter morgen?"},
                    raw_arguments='{"question":"Wie wird das Wetter morgen?"}',
                ),
            ),
            response_id="worker-start",
            continuation_token="worker-start",
        )

    def continue_turn_streaming(
        self,
        *,
        continuation_token: str,
        tool_results,
        instructions=None,
        tool_schemas=(),
        allow_web_search=None,
        on_text_delta=None,
    ) -> ToolCallingTurnResponse:
        del continuation_token, tool_results, instructions, tool_schemas, allow_web_search, on_text_delta
        self.continue_calls.append({})
        return ToolCallingTurnResponse(
            text="Morgen wird es sonnig.",
            response_id="worker-done",
            used_web_search=True,
        )


class _FakeServerSession:
    def run_turn(self, prompt, *, conversation, supervisor_conversation, emit_event, tool_bridge):
        del prompt, conversation, supervisor_conversation
        emit_event({"type": "ack", "ack_id": "checking_now", "text": "Ich schaue kurz nach."})
        handler = tool_bridge.build_handlers(("search_live_info",))["search_live_info"]
        tool_call = AgentToolCall(
            name="search_live_info",
            call_id="server-call-1",
            arguments={"question": "Wie wird das Wetter morgen?"},
            raw_arguments='{"question":"Wie wird das Wetter morgen?"}',
        )
        output = handler(tool_call)
        self.last_output = output
        from twinr.orchestrator.contracts import OrchestratorTurnCompleteEvent

        return OrchestratorTurnCompleteEvent(
            text="Morgen wird es sonnig.",
            rounds=2,
            used_web_search=True,
            response_id="resp-1",
            request_id="req-1",
            model="gpt-4o-mini",
        )


class _FakeVoiceServerSession:
    def handle_hello(self, request):
        del request
        return [{"type": "voice_ready", "session_id": "voice-1", "backend": "remote_asr"}]

    def handle_identity_profiles(self, event):
        self.last_identity_profiles = event
        return []

    def handle_runtime_state(self, event):
        del event
        return []

    def handle_audio_frame(self, frame):
        del frame
        return [
            {
                "type": "wake_confirmed",
                "matched_phrase": "twinna",
                "remaining_text": "schau mal im web",
                "backend": "remote_asr",
                "detector_label": "twinna",
                "score": 0.81,
            }
        ]


class _CloseTracker:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _FakeWakePhraseSpotter:
    def detect(self, capture):
        del capture

        return VoiceActivationMatch(
            detected=True,
            transcript="Twinna schau mal im Web",
            matched_phrase="twinna",
            remaining_text="schau mal im web",
            backend="remote_asr",
            detector_label="twinna",
            score=0.92,
        )

    def match_transcript(self, transcript: str):
        return VoiceActivationMatch(
            detected=True,
            transcript=transcript,
            matched_phrase="twinna",
            remaining_text="wie geht es dir" if "wie geht es dir" in transcript.lower() else "schau mal im web",
            normalized_transcript=str(transcript or "").strip().lower(),
            backend="remote_asr",
            detector_label="twinna",
            score=0.92,
        )


class _CountingTranscriptBackend:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript
        self.calls = 0

    def transcribe(self, *args, **kwargs):
        del args, kwargs
        self.calls += 1
        return self.transcript


class _PromptSensitiveTranscriptBackend:
    def __init__(self, *, prompted_transcript: str, unprompted_transcript: str) -> None:
        self.prompted_transcript = prompted_transcript
        self.unprompted_transcript = unprompted_transcript
        self.calls: list[dict[str, object]] = []

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        del audio_bytes, filename, content_type, language
        normalized_prompt = str(prompt or "").strip() or None
        self.calls.append({"prompt": normalized_prompt})
        if normalized_prompt is not None:
            return self.prompted_transcript
        return self.unprompted_transcript


class _MinDurationWakePhraseSpotter:
    def __init__(self, *, min_duration_ms: int, remaining_text: str = "wie geht es dir") -> None:
        self.min_duration_ms = min_duration_ms
        self.remaining_text = remaining_text
        self.capture_durations_ms: list[int] = []

    def detect(self, capture):
        self.capture_durations_ms.append(int(capture.sample.duration_ms))

        if int(capture.sample.duration_ms) < self.min_duration_ms:
            return VoiceActivationMatch(
                detected=False,
                transcript="",
                backend="remote_asr",
            )
        return VoiceActivationMatch(
            detected=True,
            transcript=f"Twinna {self.remaining_text}",
            matched_phrase="twinna",
            remaining_text=self.remaining_text,
            backend="remote_asr",
            detector_label="twinna",
            score=0.93,
        )


class _ExactDurationTranscriptSpotter:
    def __init__(self, *, accepted_duration_ms: int, transcript: str) -> None:
        self.accepted_duration_ms = int(accepted_duration_ms)
        self.transcript = transcript
        self.capture_durations_ms: list[int] = []

    def detect(self, capture):
        duration_ms = int(capture.sample.duration_ms)
        self.capture_durations_ms.append(duration_ms)
        transcript = self.transcript if duration_ms == self.accepted_duration_ms else ""
        return VoiceActivationMatch(
            detected=False,
            transcript=transcript,
            normalized_transcript=str(transcript or "").strip().lower(),
            backend="remote_asr",
        )


class _RejectingWakePhraseSpotter:
    def detect(self, capture):
        del capture

        return VoiceActivationMatch(
            detected=False,
            transcript="Maurizio Puppe",
            normalized_transcript="maurizio puppe",
            backend="remote_asr",
        )

    def match_transcript(self, transcript: str):
        return VoiceActivationMatch(
            detected=False,
            transcript=transcript,
            normalized_transcript=str(transcript or "").strip().lower(),
            backend="remote_asr",
        )


class _TranscriptOnlyNonWakePhraseSpotter:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript

    def detect(self, capture):
        del capture

        return VoiceActivationMatch(
            detected=False,
            transcript=self.transcript,
            normalized_transcript=str(self.transcript or "").strip().lower(),
            backend="remote_asr",
        )


class _BusyWakePhraseSpotter:
    def __init__(self, message: str = 'Remote ASR service returned HTTP 429: {"detail":"stt busy"}') -> None:
        self.message = message

    def detect(self, capture):
        del capture
        raise RuntimeError(self.message)


class _LengthSensitiveWakePhraseSpotter:
    def __init__(self, *, min_duration_ms: int) -> None:
        self.min_duration_ms = int(min_duration_ms)
        self.capture_durations_ms: list[int] = []

    def detect(self, capture):
        self.capture_durations_ms.append(int(capture.sample.duration_ms))

        if int(capture.sample.duration_ms) < self.min_duration_ms:
            return VoiceActivationMatch(
                detected=False,
                transcript="",
                backend="remote_asr",
            )
        return VoiceActivationMatch(
            detected=True,
            transcript="Twinna",
            matched_phrase="twinna",
            remaining_text="",
            normalized_transcript="twinna",
            backend="remote_asr",
            detector_label="twinna",
            score=0.92,
        )


class _ShortPrefixWakePhraseSpotter:
    def __init__(self, *, min_duration_ms: int, max_duration_ms: int) -> None:
        self.min_duration_ms = int(min_duration_ms)
        self.max_duration_ms = int(max_duration_ms)
        self.capture_durations_ms: list[int] = []

    def detect(self, capture):
        self.capture_durations_ms.append(int(capture.sample.duration_ms))

        duration_ms = int(capture.sample.duration_ms)
        if duration_ms < self.min_duration_ms or duration_ms > self.max_duration_ms:
            return VoiceActivationMatch(
                detected=False,
                transcript="schau mal im web nach dem wetter in berlin",
                normalized_transcript="schau mal im web nach dem wetter in berlin",
                backend="remote_asr",
            )
        return VoiceActivationMatch(
            detected=True,
            transcript="Twinner",
            matched_phrase="twinna",
            remaining_text="",
            normalized_transcript="twinner",
            backend="remote_asr",
            detector_label="twinna",
            score=0.92,
        )


class _GrowingWakePhraseSpotter:
    def __init__(self) -> None:
        self.capture_durations_ms: list[int] = []

    def detect(self, capture):
        self.capture_durations_ms.append(int(capture.sample.duration_ms))

        duration_ms = int(capture.sample.duration_ms)
        if duration_ms < 100:
            return VoiceActivationMatch(
                detected=False,
                transcript="",
                backend="remote_asr",
            )
        if duration_ms < 300:
            return VoiceActivationMatch(
                detected=True,
                transcript="Twinner",
                matched_phrase="twinna",
                remaining_text="",
                normalized_transcript="twinner",
                backend="remote_asr",
                detector_label="twinna",
                score=0.92,
            )
        return VoiceActivationMatch(
            detected=True,
            transcript="Twinner schau mal im web",
            matched_phrase="twinna",
            remaining_text="schau mal im web",
            normalized_transcript="twinner schau mal im web",
            backend="remote_asr",
            detector_label="twinna",
            score=0.92,
        )


class _BurstAnchoredWakePhraseSpotter:
    def __init__(self, *, wake_prefix: bytes, min_duration_ms: int, max_duration_ms: int) -> None:
        self.wake_prefix = wake_prefix
        self.min_duration_ms = int(min_duration_ms)
        self.max_duration_ms = int(max_duration_ms)
        self.capture_prefixes: list[bytes] = []
        self.capture_durations_ms: list[int] = []

    def detect(self, capture):
        self.capture_prefixes.append(bytes(capture.pcm_bytes[: len(self.wake_prefix)]))
        self.capture_durations_ms.append(int(capture.sample.duration_ms))

        duration_ms = int(capture.sample.duration_ms)
        if (
            duration_ms < self.min_duration_ms
            or duration_ms > self.max_duration_ms
            or not bytes(capture.pcm_bytes).startswith(self.wake_prefix)
        ):
            return VoiceActivationMatch(
                detected=False,
                transcript="schau mal im web nach dem wetter in berlin",
                normalized_transcript="schau mal im web nach dem wetter in berlin",
                backend="remote_asr",
            )
        return VoiceActivationMatch(
            detected=True,
            transcript="Twinner",
            matched_phrase="twinna",
            remaining_text="",
            normalized_transcript="twinner",
            backend="remote_asr",
            detector_label="twinna",
            score=0.92,
        )


def _pcm_frame(value: int) -> bytes:
    """Build one mono PCM16 frame with a stable amplitude for orchestrator tests."""

    sample = int(value).to_bytes(2, byteorder="little", signed=True)
    return sample * 1600


def _voice_sample_pcm_bytes(
    *,
    frequency_hz: float = 175.0,
    amplitude: float = 0.35,
    duration_s: float = 1.8,
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


def _voice_identity_profiles_event(
    *,
    user_id: str,
    display_name: str,
    pcm_bytes: bytes,
    primary_user: bool = True,
    revision: str = "rev-test",
) -> OrchestratorVoiceIdentityProfilesEvent:
    embedding = extract_voice_embedding_from_pcm16(
        pcm_bytes,
        sample_rate=16000,
        channels=1,
        min_sample_ms=1200,
    )
    return OrchestratorVoiceIdentityProfilesEvent(
        revision=revision,
        profiles=(
            OrchestratorVoiceIdentityProfile(
                user_id=user_id,
                display_name=display_name,
                primary_user=primary_user,
                embedding=embedding.vector,
                sample_count=3,
                average_duration_ms=embedding.duration_ms,
                updated_at="2026-03-27T11:00:00+00:00",
            ),
        ),
    )


def _pcm_frames_from_audio(pcm_bytes: bytes) -> tuple[bytes, ...]:
    frame_bytes = 1600 * 2
    frames: list[bytes] = []
    for start in range(0, len(pcm_bytes), frame_bytes):
        chunk = pcm_bytes[start : start + frame_bytes]
        if not chunk:
            continue
        if len(chunk) < frame_bytes:
            chunk = chunk + (b"\x00" * (frame_bytes - len(chunk)))
        frames.append(chunk)
    return tuple(frames)


class VoiceActivationMatcherTests(unittest.TestCase):
    _EXPLICIT_ALIAS_FAMILY = (
        "hey twinr",
        "he twinr",
        "hey twinna",
        "hey tynna",
        "hey twina",
        "hey twinner",
        "hallo twinr",
        "hallo twinna",
        "hallo tynna",
        "hallo twina",
        "hallo twinner",
        "twinr hallo",
        "twinr hey",
        "twinna hallo",
        "twinna hey",
        "tynna hallo",
        "tynna hey",
        "twina hallo",
        "twina hey",
        "twinner hallo",
        "twinner hey",
        "twinr",
        "twinna",
        "tynna",
        "twina",
        "twinner",
    )

    def test_default_voice_activation_phrases_keep_safe_exact_alias_family(self) -> None:
        self.assertEqual(
            DEFAULT_VOICE_ACTIVATION_PHRASES,
            (
                "hey twinr",
                "he twinr",
                "hey twinna",
                "hey twina",
                "hey twinner",
                "hallo twinr",
                "hallo twinna",
                "hallo twina",
                "hallo twinner",
                "twinr hallo",
                "twinr hey",
                "twinna hallo",
                "twinna hey",
                "twina hallo",
                "twina hey",
                "twinner hallo",
                "twinner hey",
                "twinr",
                "twinna",
                "twina",
                "twinner",
            ),
        )

    def test_match_voice_activation_transcript_default_rejects_alias_like_ambient_heads(self) -> None:
        for transcript in (
            "Tynna, wie ist das Wetter heute?",
            "Twitter wie Google, Meta oder Snapshell.",
            "Ich bin gerade Twinril unterwegs.",
            "Was über Twinrang dazu liefern, würde er mich in seine Einheit holen.",
            "Gewinner des Tages ist Max.",
        ):
            with self.subTest(transcript=transcript):
                match = match_voice_activation_transcript(
                    transcript,
                    phrases=DEFAULT_VOICE_ACTIVATION_PHRASES,
                )

                self.assertFalse(match.detected)
                self.assertIsNone(match.matched_phrase)

    def test_match_voice_activation_transcript_default_accepts_exact_safe_alias_family(self) -> None:
        scenarios = (
            ("Twinna, wie ist das Wetter heute?", "twinna", "wie ist das Wetter heute"),
            ("Twina, wie spät ist es?", "twina", "wie spät ist es"),
            ("Hey Twinner, schau mal im Web", "hey twinner", "schau mal im Web"),
        )

        for transcript, phrase, remaining_text in scenarios:
            with self.subTest(transcript=transcript):
                match = match_voice_activation_transcript(
                    transcript,
                    phrases=DEFAULT_VOICE_ACTIVATION_PHRASES,
                )

                self.assertTrue(match.detected)
                self.assertEqual(match.matched_phrase, phrase)
                self.assertEqual(match.remaining_text, remaining_text)

    def test_match_voice_activation_transcript_accepts_tynna_alias_when_explicitly_configured(self) -> None:
        match = match_voice_activation_transcript(
            "Tynna, wie ist das Wetter heute?",
            phrases=self._EXPLICIT_ALIAS_FAMILY,
        )

        self.assertTrue(match.detected)
        self.assertEqual(match.matched_phrase, "tynna")
        self.assertEqual(match.remaining_text, "wie ist das Wetter heute")

    def test_match_voice_activation_transcript_rejects_similar_non_alias_words(self) -> None:
        for transcript in ("Tina, wie spät ist es?", "Timer, wie spät ist es?", "Winter, wie spät ist es?"):
            with self.subTest(transcript=transcript):
                match = match_voice_activation_transcript(
                    transcript,
                    phrases=DEFAULT_VOICE_ACTIVATION_PHRASES,
                )

                self.assertFalse(match.detected)
                self.assertIsNone(match.matched_phrase)

    def test_match_voice_activation_transcript_accepts_head_variant_winner_family(self) -> None:
        scenarios = (
            ("Gewinner, wie ist das Wetter heute in Schwarzenbeek?", "wie ist das Wetter heute in Schwarzenbeek"),
            ("Zwinner?", ""),
            ("Hatewinner?", ""),
            ("Geldwinner.", ""),
            ("Hey Gewinner, wie spät ist es?", "wie spät ist es"),
        )

        for transcript, remaining_text in scenarios:
            with self.subTest(transcript=transcript):
                match = match_voice_activation_transcript(
                    transcript,
                    phrases=self._EXPLICIT_ALIAS_FAMILY,
                )

                self.assertTrue(match.detected)
                self.assertEqual(match.matched_phrase, "twinner")
                self.assertEqual(match.remaining_text, remaining_text)

    def test_match_voice_activation_transcript_rejects_non_head_winner_family(self) -> None:
        scenarios = (
            "Wie ist das Wetter heute in Schwarzenbeek? Gewinner, wie ist",
            "Heute ist der Gewinner Max",
            "Spinner, wie spät ist es?",
        )

        for transcript in scenarios:
            with self.subTest(transcript=transcript):
                match = match_voice_activation_transcript(
                    transcript,
                    phrases=self._EXPLICIT_ALIAS_FAMILY,
                )

                self.assertFalse(match.detected)
                self.assertIsNone(match.matched_phrase)

    def test_match_voice_activation_transcript_rejects_non_head_exact_wake_phrase(self) -> None:
        scenarios = (
            "Wie ist das Wetter heute in Schwarzenbeek? Twinna, wie ist",
            "Weil ich aus Sie hast vergessen, was sie tun wollen, Twinna.",
            "Heute hat Twinna Geburtstag.",
        )

        for transcript in scenarios:
            with self.subTest(transcript=transcript):
                match = match_voice_activation_transcript(
                    transcript,
                    phrases=self._EXPLICIT_ALIAS_FAMILY,
                )

                self.assertFalse(match.detected)
                self.assertIsNone(match.matched_phrase)

    def test_match_voice_activation_transcript_accepts_generic_head_leadin_before_wake(self) -> None:
        match = match_voice_activation_transcript(
            "Okay hey Twinr, wie spät ist es?",
            phrases=DEFAULT_VOICE_ACTIVATION_PHRASES,
        )

        self.assertTrue(match.detected)
        self.assertEqual(match.matched_phrase, "hey twinr")
        self.assertEqual(match.remaining_text, "wie spät ist es")


class VoiceRuntimeIntentContextTests(unittest.TestCase):
    def test_from_sensor_facts_surfaces_speaker_association_bias(self) -> None:
        facts = {
            "camera": {
                "person_visible": True,
            },
            "person_state": {
                "presence_active": True,
                "interaction_ready": True,
                "targeted_inference_blocked": False,
                "recommended_channel": "speech",
                "attention_state": {"state": "attending_to_device"},
                "interaction_intent_state": {"state": "showing_intent"},
            },
            "speaker_association": {
                "associated": True,
                "confidence": 0.86,
            },
            "audio_policy": {
                "background_media_likely": False,
                "speech_overlap_likely": False,
            },
        }

        context = VoiceRuntimeIntentContext.from_sensor_facts(facts)

        self.assertTrue(context.audio_bias_allowed())
        self.assertTrue(context.speaker_associated)
        self.assertEqual(context.speaker_association_confidence, 0.86)
        self.assertTrue(context.strong_speaker_bias_allowed())
        self.assertTrue(context.familiar_speaker_bias_allowed())

    def test_strong_speaker_bias_requires_association_confidence(self) -> None:
        context = VoiceRuntimeIntentContext(
            person_visible=True,
            presence_active=True,
            interaction_ready=True,
            targeted_inference_blocked=False,
            recommended_channel="speech",
            speaker_associated=True,
            speaker_association_confidence=0.79,
        )

        self.assertTrue(context.audio_bias_allowed())
        self.assertFalse(context.strong_speaker_bias_allowed())

    def test_familiar_speaker_bias_is_blocked_by_background_media(self) -> None:
        context = VoiceRuntimeIntentContext(
            person_visible=True,
            presence_active=True,
            interaction_ready=True,
            targeted_inference_blocked=False,
            recommended_channel="speech",
            speaker_associated=True,
            speaker_association_confidence=0.86,
            background_media_likely=True,
            speech_overlap_likely=False,
        )

        self.assertTrue(context.strong_speaker_bias_allowed())
        self.assertFalse(context.familiar_speaker_bias_allowed())

    def test_from_sensor_facts_keeps_waiting_activation_allowed_during_focus_hold(self) -> None:
        facts = {
            "camera": {
                "person_visible": False,
                "person_recently_visible": True,
            },
            "attention_target": {
                "active": True,
                "state": "holding_session_focus",
                "session_focus_active": True,
            },
            "person_state": {
                "presence_active": True,
                "interaction_ready": False,
                "targeted_inference_blocked": True,
                "recommended_channel": "display",
                "attention_state": {"state": "attending_to_device"},
                "interaction_intent_state": {"state": "possible_intent"},
            },
        }

        context = VoiceRuntimeIntentContext.from_sensor_facts(facts)

        self.assertFalse(context.person_visible)
        self.assertTrue(context.waiting_activation_allowed())
        self.assertTrue(context.presence_active)
        self.assertTrue(context.waiting_activation_allowed())
        self.assertFalse(context.audio_bias_allowed())

    def test_from_sensor_facts_keeps_waiting_activation_allowed_when_camera_is_down_but_near_presence_is_attested(
        self,
    ) -> None:
        facts = {
            "camera": {
                "person_visible": False,
                "person_recently_visible": True,
                "camera_online": False,
                "camera_ready": False,
                "camera_ai_ready": False,
            },
            "near_device_presence": {
                "occupied_likely": True,
                "person_visible": False,
                "person_recently_visible": True,
                "speech_recent": True,
                "voice_activation_armed": True,
            },
            "attention_target": {
                "active": False,
                "state": "inactive",
                "session_focus_active": False,
            },
            "person_state": {
                "presence_active": True,
                "interaction_ready": False,
                "targeted_inference_blocked": True,
                "recommended_channel": "display",
                "attention_state": {"state": "inactive"},
                "interaction_intent_state": {"state": "passive"},
            },
        }

        context = VoiceRuntimeIntentContext.from_sensor_facts(facts)

        self.assertTrue(context.person_visible)
        self.assertTrue(context.presence_active)
        self.assertTrue(context.waiting_activation_allowed())
        self.assertFalse(context.audio_bias_allowed())

    def test_from_sensor_facts_keeps_waiting_activation_allowed_for_camera_down_without_other_presence_evidence(
        self,
    ) -> None:
        facts = {
            "camera": {
                "person_visible": False,
                "person_recently_visible": False,
                "camera_online": False,
                "camera_ready": False,
                "camera_ai_ready": False,
            },
            "near_device_presence": {
                "occupied_likely": True,
                "person_visible": False,
                "person_recently_visible": False,
                "room_motion_recent": True,
                "speech_recent": False,
                "voice_activation_armed": False,
            },
            "attention_target": {
                "active": False,
                "state": "inactive",
                "session_focus_active": False,
            },
            "person_state": {
                "presence_active": True,
                "interaction_ready": False,
                "targeted_inference_blocked": True,
                "recommended_channel": "display",
                "attention_state": {"state": "inactive"},
                "interaction_intent_state": {"state": "passive"},
            },
        }

        context = VoiceRuntimeIntentContext.from_sensor_facts(facts)

        self.assertIsNone(context.person_visible)
        self.assertTrue(context.presence_active)
        self.assertTrue(context.waiting_activation_allowed())
        self.assertFalse(context.audio_bias_allowed())

    def test_from_sensor_facts_keeps_waiting_activation_allowed_with_attested_presence_without_visible_person(
        self,
    ) -> None:
        facts = {
            "camera": {
                "person_visible": False,
                "person_recently_visible": True,
            },
            "attention_target": {
                "active": False,
                "state": "inactive",
                "session_focus_active": False,
            },
            "person_state": {
                "presence_active": True,
                "interaction_ready": False,
                "targeted_inference_blocked": True,
                "recommended_channel": "display",
                "attention_state": {"state": "inactive"},
                "interaction_intent_state": {"state": "passive"},
            },
        }

        context = VoiceRuntimeIntentContext.from_sensor_facts(facts)

        self.assertFalse(context.person_visible)
        self.assertTrue(context.presence_active)
        self.assertTrue(context.waiting_activation_allowed())

    def test_from_sensor_facts_keeps_waiting_activation_blocked_without_local_presence(self) -> None:
        facts = {
            "camera": {
                "person_visible": False,
                "person_recently_visible": False,
            },
            "attention_target": {
                "active": False,
                "state": "inactive",
                "session_focus_active": False,
            },
            "person_state": {
                "presence_active": False,
                "interaction_ready": False,
                "targeted_inference_blocked": True,
                "recommended_channel": "display",
                "attention_state": {"state": "inactive"},
                "interaction_intent_state": {"state": "passive"},
            },
        }

        context = VoiceRuntimeIntentContext.from_sensor_facts(facts)

        self.assertFalse(context.person_visible)
        self.assertFalse(context.presence_active)
        self.assertFalse(context.waiting_activation_allowed())


class OrchestratorSessionTests(unittest.TestCase):
    def test_edge_orchestrator_session_emits_ack_and_remote_tool_request(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
            )
            session = EdgeOrchestratorSession(
                config,
                supervisor_decision_provider=_FakeDecisionProvider(),
                specialist_provider=_FakeSpecialistProvider(),
                tool_names=("search_live_info",),
            )
            events: list[dict[str, object]] = []
            tool_bridge = None

            def emit(payload: dict[str, object]) -> None:
                events.append(payload)
                if payload.get("type") == "tool_request":
                    tool_bridge.submit_result(
                        str(payload["call_id"]),
                        output={"answer": "Morgen wird es sonnig."},
                        error=None,
                    )

            tool_bridge = RemoteToolBridge(emit)

            result = session.run_turn(
                "Wie wird das Wetter morgen?",
                conversation=(),
                supervisor_conversation=(),
                emit_event=emit,
                tool_bridge=tool_bridge,
            )

        self.assertEqual(events[0]["type"], "ack")
        self.assertEqual(events[0]["ack_id"], "checking_now")
        self.assertEqual(events[1]["type"], "tool_request")
        self.assertEqual(events[1]["name"], "search_live_info")
        self.assertEqual(result.text, "Morgen wird es sonnig.")
        self.assertTrue(result.used_web_search)

    def test_remote_tool_bridge_uses_shared_default_timeout_budget(self) -> None:
        bridge = RemoteToolBridge(lambda payload: None)

        self.assertEqual(
            bridge._tool_result_timeout_seconds,
            DEFAULT_REMOTE_TOOL_TIMEOUT_SECONDS,
        )

    def test_remote_tool_timeout_env_applies_to_bridge_budget(self) -> None:
        with patch.dict("os.environ", {REMOTE_TOOL_TIMEOUT_ENV: "123"}, clear=False):
            self.assertEqual(read_remote_tool_timeout_seconds(), 123.0)
            bridge = RemoteToolBridge(lambda payload: None)

        self.assertEqual(bridge._tool_result_timeout_seconds, 123.0)


class OrchestratorServerTests(unittest.TestCase):
    def test_server_websocket_bridges_tool_request_and_result(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                orchestrator_shared_secret="secret-token",
            )
            app = EdgeOrchestratorServer(config, session_factory=lambda _config: _FakeServerSession()).create_app()
            with TestClient(app) as client:
                with client.websocket_connect(
                    "/ws/orchestrator",
                    headers={"x-twinr-secret": "secret-token"},
                ) as websocket:
                    websocket.send_json(OrchestratorTurnRequest(prompt="Wie wird das Wetter morgen?").to_payload())
                    ack = websocket.receive_json()
                    tool_request = websocket.receive_json()
                    websocket.send_json(
                        {
                            "type": "tool_result",
                            "call_id": tool_request["call_id"],
                            "ok": True,
                            "output": {"answer": "Morgen wird es sonnig."},
                        }
                    )
                    completed = websocket.receive_json()

        self.assertEqual(ack["type"], "ack")
        self.assertEqual(ack_text_for_id(ack["ack_id"]), "Ich schaue kurz nach.")
        self.assertEqual(tool_request["type"], "tool_request")
        self.assertEqual(completed["type"], "turn_complete")
        self.assertEqual(completed["text"], "Morgen wird es sonnig.")

    def test_server_voice_websocket_emits_ready_and_wake_confirmed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                orchestrator_shared_secret="secret-token",
            )
            app = EdgeOrchestratorServer(
                config,
                voice_session_factory=lambda _config: _FakeVoiceServerSession(),
            ).create_app()
            with TestClient(app) as client:
                with client.websocket_connect(
                    "/ws/orchestrator/voice",
                    headers={"x-twinr-secret": "secret-token"},
                ) as websocket:
                    websocket.send_json(
                        OrchestratorVoiceHelloRequest(
                            session_id="voice-1",
                            sample_rate=16000,
                            channels=1,
                            chunk_ms=100,
                        ).to_payload()
                    )
                    ready = websocket.receive_json()
                    websocket.send_json(
                        OrchestratorVoiceAudioFrame(
                            sequence=0,
                            pcm_bytes=b"\x00\x00" * 1600,
                        ).to_payload()
                    )
                    wake = websocket.receive_json()

        self.assertEqual(ready["type"], "voice_ready")
        self.assertEqual(wake["type"], "wake_confirmed")
        self.assertEqual(wake["remaining_text"], "schau mal im web")

    def test_server_lifespan_closes_forensics_and_remote_asr_service(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                orchestrator_shared_secret="secret-token",
            )
            fake_forensics = _CloseTracker()
            fake_remote_asr_service = _CloseTracker()
            with patch(
                "twinr.orchestrator.server.WorkflowForensics.from_env",
                return_value=fake_forensics,
            ):
                server = EdgeOrchestratorServer(config)
            app = server.create_app()
            server._remote_asr_service = fake_remote_asr_service

            with TestClient(app):
                pass

        self.assertEqual(fake_forensics.close_calls, 1)
        self.assertEqual(fake_remote_asr_service.close_calls, 1)

    def test_create_app_requires_remote_asr_service_url(self) -> None:
        with TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY=test-key",
                        "TWINR_ORCHESTRATOR_SHARED_SECRET=secret-token",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "TWINR_VOICE_ORCHESTRATOR_REMOTE_ASR_URL",
            ):
                create_orchestrator_app(env_path)

class OrchestratorClientTests(unittest.TestCase):
    def test_client_uses_shared_default_tool_timeout_budget(self) -> None:
        client = OrchestratorWebSocketClient(
            "ws://127.0.0.1/ws",
            connector=lambda *args, **kwargs: None,
            require_tls=False,
        )

        self.assertEqual(
            client.tool_timeout_seconds,
            DEFAULT_REMOTE_TOOL_TIMEOUT_SECONDS,
        )

    def test_voice_audio_frame_round_trips_embedded_runtime_state(self) -> None:
        frame = OrchestratorVoiceAudioFrame(
            sequence=7,
            pcm_bytes=b"\x01\x00" * 160,
            runtime_state=OrchestratorVoiceRuntimeStateEvent(
                state="waiting",
                detail="idle",
                follow_up_allowed=False,
                attention_state="attending_to_device",
                interaction_intent_state="showing_intent",
                person_visible=True,
                presence_active=True,
                interaction_ready=True,
                targeted_inference_blocked=False,
                recommended_channel="speech",
                speaker_associated=True,
                speaker_association_confidence=0.88,
                background_media_likely=False,
                speech_overlap_likely=False,
                voice_quiet_until_utc="2026-03-25T12:15:00Z",
            ),
        )

        decoded = OrchestratorVoiceAudioFrame.from_payload(frame.to_payload())

        self.assertEqual(decoded.sequence, 7)
        self.assertEqual(decoded.pcm_bytes, b"\x01\x00" * 160)
        self.assertIsNotNone(decoded.runtime_state)
        self.assertEqual(decoded.runtime_state.state, "waiting")
        self.assertEqual(decoded.runtime_state.detail, "idle")
        self.assertEqual(decoded.runtime_state.attention_state, "attending_to_device")
        self.assertEqual(decoded.runtime_state.interaction_intent_state, "showing_intent")
        self.assertTrue(decoded.runtime_state.person_visible)
        self.assertTrue(decoded.runtime_state.presence_active)
        self.assertTrue(decoded.runtime_state.interaction_ready)
        self.assertFalse(decoded.runtime_state.targeted_inference_blocked)
        self.assertEqual(decoded.runtime_state.recommended_channel, "speech")
        self.assertTrue(decoded.runtime_state.speaker_associated)
        self.assertEqual(decoded.runtime_state.speaker_association_confidence, 0.88)
        self.assertFalse(decoded.runtime_state.background_media_likely)
        self.assertFalse(decoded.runtime_state.speech_overlap_likely)
        self.assertEqual(decoded.runtime_state.voice_quiet_until_utc, "2026-03-25T12:15:00Z")

    def test_voice_identity_profiles_event_round_trips(self) -> None:
        event = OrchestratorVoiceIdentityProfilesEvent(
            revision="rev-123",
            profiles=(
                OrchestratorVoiceIdentityProfile(
                    user_id="main_user",
                    display_name="Theo",
                    primary_user=True,
                    embedding=(0.1, 0.2, 0.3),
                    sample_count=4,
                    average_duration_ms=2600,
                    updated_at="2026-03-27T11:00:00+00:00",
                ),
            ),
        )

        decoded = OrchestratorVoiceIdentityProfilesEvent.from_payload(event.to_payload())

        self.assertEqual(decoded.revision, "rev-123")
        self.assertEqual(len(decoded.profiles), 1)
        self.assertEqual(decoded.profiles[0].user_id, "main_user")
        self.assertEqual(decoded.profiles[0].display_name, "Theo")
        self.assertTrue(decoded.profiles[0].primary_user)
        self.assertEqual(decoded.profiles[0].embedding, (0.1, 0.2, 0.3))

    def test_client_handles_ack_tool_request_and_completion(self) -> None:
        class _FakeSocket:
            def __init__(self):
                self.sent: list[str] = []
                self.messages = iter(
                    [
                        '{"type":"ack","ack_id":"checking_now","text":"Ich schaue kurz nach."}',
                        '{"type":"tool_request","call_id":"call-1","name":"search_live_info","arguments":{"question":"Wie wird das Wetter morgen?"}}',
                        '{"type":"turn_complete","text":"Morgen wird es sonnig.","rounds":2,"used_web_search":true,"response_id":"resp-1","request_id":"req-1","model":"gpt-4o-mini","tool_calls":[],"tool_results":[]}',
                    ]
                )

            def send(self, payload: str) -> None:
                self.sent.append(payload)

            def recv(self, timeout=None) -> str:
                del timeout
                try:
                    return next(self.messages)
                except StopIteration as exc:
                    raise TimeoutError() from exc

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        ack_events = []
        sockets: list[_FakeSocket] = []

        def connector(*args, **kwargs):
            del args, kwargs
            socket = _FakeSocket()
            sockets.append(socket)
            return socket

        client = OrchestratorWebSocketClient("ws://127.0.0.1/ws", connector=connector, require_tls=False)

        result = client.run_turn(
            OrchestratorTurnRequest(prompt="Wie wird das Wetter morgen?"),
            tool_handlers={
                "search_live_info": lambda arguments: {
                    "answer": arguments["question"],
                    "token_usage": TokenUsage(input_tokens=21, output_tokens=21, total_tokens=42),
                }
            },
            on_ack=ack_events.append,
        )

        self.assertEqual(len(ack_events), 1)
        self.assertEqual(ack_events[0].ack_id, "checking_now")
        self.assertEqual(result.text, "Morgen wird es sonnig.")
        self.assertEqual(len(sockets), 1)
        tool_response = json.loads(sockets[0].sent[1])
        self.assertEqual(tool_response["type"], "tool_result")
        self.assertTrue(tool_response["ok"])
        self.assertEqual(tool_response["output"]["token_usage"]["input_tokens"], 21)
        self.assertEqual(tool_response["output"]["token_usage"]["output_tokens"], 21)
        self.assertEqual(tool_response["output"]["token_usage"]["total_tokens"], 42)


    def test_voice_client_decodes_ready_and_wake_events(self) -> None:
        class _FakeSocket:
            def __init__(self):
                self.sent: list[str] = []
                self.messages = iter(
                    [
                        '{"type":"voice_ready","session_id":"voice-1","backend":"remote_asr"}',
                        '{"type":"wake_confirmed","matched_phrase":"twinna","remaining_text":"schau mal im web","backend":"remote_asr","detector_label":"twinna","score":0.9}',
                    ]
                )

            def send(self, payload: str) -> None:
                self.sent.append(payload)

            def recv(self, timeout=None) -> str:
                del timeout
                try:
                    return next(self.messages)
                except StopIteration as exc:
                    raise TimeoutError() from exc

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        events = []
        client = OrchestratorVoiceWebSocketClient(
            "ws://127.0.0.1/ws",
            connector=lambda *args, **kwargs: _FakeSocket(),
            on_event=events.append,
            require_tls=False,
        )

        client.open()
        client.send_hello(
            OrchestratorVoiceHelloRequest(
                session_id="voice-1",
                sample_rate=16000,
                channels=1,
                chunk_ms=100,
            )
        )
        client.send_runtime_state(
            OrchestratorVoiceRuntimeStateEvent(
                state="waiting",
                follow_up_allowed=False,
            )
        )
        client.close()

        self.assertTrue(any(getattr(event, "backend", "") == "remote_asr" for event in events))
        self.assertTrue(any(getattr(event, "remaining_text", "") == "schau mal im web" for event in events))

    def test_voice_client_close_tolerates_receiver_thread_starting_late(self) -> None:
        class _FakeSocket:
            def __init__(self):
                self.sent: list[str] = []

            def send(self, payload: str) -> None:
                self.sent.append(payload)

            def recv(self, timeout=None) -> str:
                raise TimeoutError()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class _ImmediateWaitEvent:
            def clear(self) -> None:
                return None

            def set(self) -> None:
                return None

            def wait(self, timeout=None) -> bool:
                del timeout
                return True

            def is_set(self) -> bool:
                return False

        class _DeferredThread:
            latest = None

            def __init__(self, *, target, args=(), daemon=None, name=None):
                del daemon, name
                self._target = target
                self._args = args
                _DeferredThread.latest = self

            def start(self) -> None:
                return None

            def join(self, timeout=None) -> None:
                del timeout

            def is_alive(self) -> bool:
                return True

            def run_now(self) -> None:
                self._target(*self._args)

        client = OrchestratorVoiceWebSocketClient(
            "ws://127.0.0.1/ws",
            connector=lambda *args, **kwargs: _FakeSocket(),
            on_event=lambda event: None,
            require_tls=False,
        )
        client._receiver_started = _ImmediateWaitEvent()

        with patch("twinr.orchestrator.voice_client.Thread", _DeferredThread):
            client.open()
            client.send_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            client.close()
            self.assertIsNotNone(_DeferredThread.latest)
            _DeferredThread.latest.run_now()

    def test_voice_client_serializes_concurrent_runtime_state_and_audio_sends(self) -> None:
        class _BlockingSocket:
            def __init__(self):
                self.sent: list[str] = []
                self._first_send_entered = Event()
                self._release_first_send = Event()
                self._active_sends = 0
                self.max_active_sends = 0
                self._send_calls = 0

            def send(self, payload: str) -> None:
                self._send_calls += 1
                self._active_sends += 1
                self.max_active_sends = max(self.max_active_sends, self._active_sends)
                try:
                    if self._send_calls == 1:
                        self._first_send_entered.set()
                        self._release_first_send.wait(timeout=1.0)
                    time.sleep(0.01)
                    self.sent.append(payload)
                finally:
                    self._active_sends -= 1

            def recv(self, timeout=None) -> str:
                raise TimeoutError()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        socket = _BlockingSocket()
        client = OrchestratorVoiceWebSocketClient(
            "ws://127.0.0.1/ws",
            connector=lambda *args, **kwargs: socket,
            on_event=lambda event: None,
            require_tls=False,
        )

        client.open()
        audio_thread = Thread(
            target=client.send_audio_frame,
            args=(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=b"\x00\x00" * 160),
            ),
        )
        audio_thread.start()
        self.assertTrue(socket._first_send_entered.wait(timeout=1.0))

        state_thread = Thread(
            target=client.send_runtime_state,
            args=(
                OrchestratorVoiceRuntimeStateEvent(state="listening", follow_up_allowed=False),
            ),
        )
        state_thread.start()
        time.sleep(0.05)
        self.assertEqual(socket.max_active_sends, 1)

        socket._release_first_send.set()
        audio_thread.join(timeout=1.0)
        state_thread.join(timeout=1.0)
        client.close()

        self.assertEqual(len(socket.sent), 2)


class OrchestratorVoiceSessionTests(unittest.TestCase):
    def test_voice_session_hello_uses_client_trace_id(self) -> None:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
            voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
        )
        session = EdgeOrchestratorVoiceSession(config)

        session.handle_hello(
            OrchestratorVoiceHelloRequest(
                session_id="voice-1",
                trace_id="trace-under-test",
                sample_rate=16_000,
                channels=1,
                chunk_ms=100,
            )
        )

        self.assertEqual(session._trace_id, "trace-under-test")

    def test_voice_session_waiting_activation_blocks_while_voice_quiet_active(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "Twinna wie geht es dir"),
                wake_phrase_spotter=_MinDurationWakePhraseSpotter(
                    min_duration_ms=100,
                    remaining_text="wie geht es dir",
                ),
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                    voice_quiet_until_utc=(
                        datetime.now(timezone.utc) + timedelta(minutes=15)
                    ).replace(microsecond=0).isoformat(),
                )
            )
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )

        self.assertFalse(session._waiting_activation_allowed())
        self.assertEqual(events, [])
        self.assertIsNone(session._pending_transcript_utterance)

    def test_voice_session_closes_follow_up_when_voice_quiet_runtime_state_arrives(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "wie geht es dir"),
                wake_phrase_spotter=_TranscriptOnlyNonWakePhraseSpotter("wie geht es dir"),
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="follow_up_open",
                    detail="voice_activation",
                    follow_up_allowed=True,
                )
            )
            closed = session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="follow_up_open",
                    detail="quiet_mode",
                    follow_up_allowed=True,
                    voice_quiet_until_utc=(
                        datetime.now(timezone.utc) + timedelta(minutes=10)
                    ).replace(microsecond=0).isoformat(),
                )
            )

        self.assertEqual(closed[0]["type"], "follow_up_closed")
        self.assertEqual(closed[0]["reason"], "voice_quiet_active")
        self.assertEqual(session._state, "waiting")

    def test_remote_asr_backend_adapter_posts_audio_and_returns_text(self) -> None:
        captured = {}

        def _handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["headers"] = dict(request.headers.items())
            captured["body"] = request.read()
            return httpx.Response(
                200,
                json={
                    "text": "Hallo, Twinner!",
                    "language": "de",
                    "segments": [{"start": 0.0, "end": 1.0, "text": "Hallo, Twinner!"}],
                    "duration_sec": 1.0,
                },
            )

        adapter = RemoteAsrBackendAdapter(
            base_url="http://127.0.0.1:18090",
            bearer_token="secret",
            language="de",
            mode="active_listening",
            timeout_s=2.5,
            transport=httpx.MockTransport(_handler),
        )

        text = adapter.transcribe(
            b"RIFFdata",
            filename="voice-activation.wav",
            content_type="audio/wav",
            prompt="Twinr, Twinna.",
        )

        self.assertEqual(text, "Hallo, Twinner!")
        self.assertEqual(captured["url"], "http://127.0.0.1:18090/v1/transcribe")
        headers = {key.lower(): value for key, value in captured["headers"].items()}
        self.assertEqual(headers["authorization"], "Bearer secret")
        self.assertIn(b'name="language"', captured["body"])
        self.assertIn(b"active_listening", captured["body"])
        self.assertIn(b'name="prompt"', captured["body"])
        self.assertIn(b"Twinr, Twinna.", captured["body"])
        self.assertIn(b'filename="voice-activation.wav"', captured["body"])
        self.assertIn(b"RIFFdata", captured["body"])

    def test_remote_asr_backend_adapter_retries_busy_service_once(self) -> None:
        attempts: list[int] = []
        sleep_calls: list[float] = []

        def _handler(request: httpx.Request) -> httpx.Response:
            del request
            attempts.append(1)
            if len(attempts) == 1:
                return httpx.Response(
                    429,
                    json={"detail": "stt busy"},
                )
            return httpx.Response(
                200,
                json={
                    "text": "Hey Twinna",
                    "language": "de",
                    "segments": [],
                    "duration_sec": 1.0,
                },
            )

        adapter = RemoteAsrBackendAdapter(
            base_url="http://127.0.0.1:18090",
            language="de",
            mode="active_listening",
            timeout_s=2.5,
            retry_attempts=1,
            retry_backoff_s=0.4,
            retry_jitter_s=0.0,
            transport=httpx.MockTransport(_handler),
        )

        with patch("time.sleep", lambda seconds: sleep_calls.append(seconds)):
            text = adapter.transcribe(b"RIFFdata", filename="voice-activation.wav", content_type="audio/wav")

        self.assertEqual(text, "Hey Twinna")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(sleep_calls, [0.4])

    def test_remote_asr_backend_adapter_projects_request_context_into_headers(self) -> None:
        captured = {}

        def _handler(request: httpx.Request) -> httpx.Response:
            captured["headers"] = dict(request.headers.items())
            return httpx.Response(
                200,
                json={"text": "", "language": "de", "segments": [], "duration_sec": 0.5},
            )

        adapter = RemoteAsrBackendAdapter(
            base_url="http://127.0.0.1:18090",
            language="de",
            mode="active_listening",
            timeout_s=1.5,
            transport=httpx.MockTransport(_handler),
        )

        with adapter.bind_request_context(
            {
                "session_id": "voice-test-1",
                "trace_id": "trace-voice-test-1",
                "stage": "activation_utterance",
                "state": "waiting",
                "capture_duration_ms": 2200,
                "capture_signal_sha256": "abc123def456",
            }
        ):
            adapter.transcribe(b"RIFFdata", filename="voice-activation.wav", content_type="audio/wav")

        headers = {key.lower(): value for key, value in captured["headers"].items()}
        self.assertEqual(headers["x-twinr-voice-session-id"], "voice-test-1")
        self.assertEqual(headers["x-twinr-voice-trace-id"], "trace-voice-test-1")
        self.assertEqual(headers["x-twinr-voice-stage"], "activation_utterance")
        self.assertEqual(headers["x-twinr-voice-state"], "waiting")
        self.assertEqual(headers["x-twinr-voice-capture-duration-ms"], "2200")
        self.assertEqual(headers["x-twinr-voice-capture-sha256"], "abc123def456")

    def test_remote_asr_http_service_accepts_adapter_multipart_upload(self) -> None:
        class _FakeProvider:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def transcribe(
                self,
                audio_bytes: bytes,
                *,
                filename: str = "audio.wav",
                content_type: str = "audio/wav",
                language: str | None = None,
                prompt: str | None = None,
            ) -> str:
                self.calls.append(
                    {
                        "audio_bytes": audio_bytes,
                        "filename": filename,
                        "content_type": content_type,
                        "language": language,
                        "prompt": prompt,
                    }
                )
                return "Twitter, wie geht es dir"

        provider = _FakeProvider()
        app = FastAPI()
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
            voice_orchestrator_remote_asr_bearer_token="voice-secret",
        )
        app.include_router(RemoteAsrHttpService(config, provider=provider).build_router())
        client = TestClient(app)
        body, content_type = _encode_multipart_form(
            file_field="audio",
            filename="voice-window.wav",
            file_content_type="audio/wav",
            file_bytes=b"RIFFdata",
            text_fields={"language": "de", "mode": "active_listening", "prompt": "Twinr, Twinna."},
        )

        response = client.post(
            "/v1/transcribe",
            content=body,
            headers={
                "Content-Type": content_type,
                "Authorization": "Bearer voice-secret",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], "Twitter, wie geht es dir")
        self.assertEqual(response.json()["language"], "de")
        self.assertEqual(provider.calls[0]["audio_bytes"], b"RIFFdata")
        self.assertEqual(provider.calls[0]["filename"], "voice-window.wav")
        self.assertEqual(provider.calls[0]["content_type"], "audio/wav")
        self.assertEqual(provider.calls[0]["language"], "de")
        self.assertEqual(provider.calls[0]["prompt"], "Twinr, Twinna.")

    def test_remote_asr_http_service_rejects_missing_bearer_token(self) -> None:
        app = FastAPI()
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
            voice_orchestrator_remote_asr_bearer_token="voice-secret",
        )
        app.include_router(
            RemoteAsrHttpService(
                config,
                provider=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
            ).build_router()
        )
        client = TestClient(app)
        body, content_type = _encode_multipart_form(
            file_field="audio",
            filename="voice-window.wav",
            file_content_type="audio/wav",
            file_bytes=b"RIFFdata",
            text_fields={},
        )

        response = client.post(
            "/v1/transcribe",
            content=body,
            headers={"Content-Type": content_type},
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "missing_bearer_token")

    def test_remote_asr_url_targets_local_orchestrator_only_when_pointing_to_same_server(self) -> None:
        matching = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
            orchestrator_host="127.0.0.1",
            orchestrator_port=8798,
            voice_orchestrator_remote_asr_url="http://127.0.0.1:8798",
        )
        non_matching = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
            orchestrator_host="127.0.0.1",
            orchestrator_port=8798,
            voice_orchestrator_remote_asr_url="http://127.0.0.1:18091",
        )

        self.assertTrue(remote_asr_url_targets_local_orchestrator(matching))
        self.assertFalse(remote_asr_url_targets_local_orchestrator(non_matching))

    def test_voice_session_follow_up_window_routes_repeated_wake_phrase_as_fresh_wake(self) -> None:
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
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "Twinna wie geht es dir"),
                wake_phrase_spotter=_MinDurationWakePhraseSpotter(
                    min_duration_ms=100,
                    remaining_text="wie geht es dir",
                ),
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                    state_attested=False,
                )
            )
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="follow_up_open",
                    detail="voice_activation",
                    follow_up_allowed=True,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(first, [])
        self.assertEqual(events[0]["type"], "wake_confirmed")
        self.assertEqual(events[0]["matched_phrase"], "twinna")
        self.assertEqual(events[0]["remaining_text"], "wie geht es dir")

    def test_voice_session_follow_up_window_rewake_uses_generic_transcript_before_match(self) -> None:
        with TemporaryDirectory() as temp_dir:
            backend = _PromptSensitiveTranscriptBackend(
                prompted_transcript="Skyler",
                unprompted_transcript="Twinr wie geht es dir",
            )
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                openai_realtime_language="de",
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_follow_up_window_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=backend,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="follow_up_open",
                    detail="voice_activation",
                    follow_up_allowed=True,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(first, [])
        self.assertEqual(events[0]["type"], "wake_confirmed")
        self.assertEqual(events[0]["matched_phrase"], "twinr")
        self.assertEqual(events[0]["remaining_text"], "wie geht es dir")
        self.assertEqual(backend.calls, [{"prompt": None}])

    def test_voice_session_follow_up_window_commits_transcript_without_wake_phrase(self) -> None:
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
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "wie geht es dir"),
                wake_phrase_spotter=_TranscriptOnlyNonWakePhraseSpotter("wie geht es dir"),
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="follow_up_open",
                    detail="voice_activation",
                    follow_up_allowed=True,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(first, [])
        self.assertEqual(events[0]["type"], "transcript_committed")
        self.assertEqual(events[0]["source"], "follow_up")
        self.assertEqual(events[0]["transcript"], "wie geht es dir")

    def test_voice_session_listening_window_commits_same_stream_transcript(self) -> None:
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
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "wie geht es dir"),
                wake_phrase_spotter=_TranscriptOnlyNonWakePhraseSpotter("wie geht es dir"),
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="listening",
                    detail="voice_activation",
                    follow_up_allowed=False,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(first, [])
        self.assertEqual(events[0]["type"], "transcript_committed")
        self.assertEqual(events[0]["source"], "listening")
        self.assertEqual(events[0]["transcript"], "wie geht es dir")

    def test_voice_session_listening_window_uses_generic_transcript_before_wake_match(self) -> None:
        with TemporaryDirectory() as temp_dir:
            backend = _PromptSensitiveTranscriptBackend(
                prompted_transcript="Skyler",
                unprompted_transcript="Sei bitte 20 Minuten ruhig",
            )
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                openai_realtime_language="de",
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_follow_up_window_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=backend,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="listening",
                    detail="voice_activation",
                    follow_up_allowed=False,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(first, [])
        self.assertEqual(events[0]["type"], "transcript_committed")
        self.assertEqual(events[0]["source"], "listening")
        self.assertEqual(events[0]["transcript"], "Sei bitte 20 Minuten ruhig")
        self.assertEqual(backend.calls, [{"prompt": None}])

    def test_voice_session_listening_window_resets_stale_history_before_same_stream_commit(self) -> None:
        with TemporaryDirectory() as temp_dir:
            spotter = _ExactDurationTranscriptSpotter(
                accepted_duration_ms=200,
                transcript="wie geht es dir",
            )
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
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "wie geht es dir"),
                wake_phrase_spotter=spotter,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            session._remember_frame(_pcm_frame(2))
            session._remember_frame(_pcm_frame(2))
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="listening",
                    detail="voice_activation",
                    follow_up_allowed=False,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(first, [])
        self.assertEqual(events[0]["type"], "transcript_committed")
        self.assertEqual(events[0]["transcript"], "wie geht es dir")
        self.assertEqual(spotter.capture_durations_ms, [200])

    def test_voice_session_listening_window_preserves_inflight_waiting_utterance_across_handoff(self) -> None:
        with TemporaryDirectory() as temp_dir:
            spotter = _ExactDurationTranscriptSpotter(
                accepted_duration_ms=200,
                transcript="am sensor gewesen",
            )
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                audio_beep_duration_ms=180,
                audio_beep_settle_ms=120,
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_follow_up_window_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "am sensor gewesen"),
                wake_phrase_spotter=spotter,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="waiting",
                    detail="idle",
                    follow_up_allowed=False,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="listening",
                    detail="gesture",
                    follow_up_allowed=False,
                )
            )
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(first, [])
        self.assertEqual(events[0]["type"], "transcript_committed")
        self.assertEqual(events[0]["source"], "listening")
        self.assertEqual(events[0]["transcript"], "am sensor gewesen")
        self.assertEqual(spotter.capture_durations_ms, [200])

    def test_voice_session_follow_up_window_resets_stale_history_before_same_stream_commit(self) -> None:
        with TemporaryDirectory() as temp_dir:
            spotter = _ExactDurationTranscriptSpotter(
                accepted_duration_ms=200,
                transcript="wie geht es dir",
            )
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
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "wie geht es dir"),
                wake_phrase_spotter=spotter,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            session._remember_frame(_pcm_frame(2))
            session._remember_frame(_pcm_frame(2))
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="follow_up_open",
                    detail="voice_activation",
                    follow_up_allowed=True,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(first, [])
        self.assertEqual(events[0]["type"], "transcript_committed")
        self.assertEqual(events[0]["transcript"], "wie geht es dir")
        self.assertEqual(spotter.capture_durations_ms, [200])

    def test_voice_session_follow_up_window_waits_for_full_window_before_generic_capture(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            backend = _CountingTranscriptBackend("Untertitel der Amara.org-Community")
            wake_phrase_spotter = _MinDurationWakePhraseSpotter(min_duration_ms=300, remaining_text="wie geht es dir")
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_follow_up_window_ms=300,
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=backend,
                wake_phrase_spotter=wake_phrase_spotter,
                monotonic_fn=clock,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="follow_up_open",
                    detail="voice_activation",
                    follow_up_allowed=True,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            clock.step(0.1)
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(2))
            )
            clock.step(0.1)
            third = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=2, pcm_bytes=_pcm_frame(2))
            )
            clock.step(0.1)
            fourth = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=3, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(third, [])
        self.assertEqual(fourth[0]["type"], "wake_confirmed")
        self.assertEqual(fourth[0]["matched_phrase"], "twinna")
        self.assertEqual(fourth[0]["remaining_text"], "wie geht es dir")
        self.assertEqual(backend.calls, 0)

    def test_voice_session_remote_asr_context_bias_accepts_shorter_wake_utterance(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_remote_asr_min_wake_duration_ms=300,
                voice_orchestrator_intent_min_wake_duration_relief_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            wake_phrase_spotter = _MinDurationWakePhraseSpotter(min_duration_ms=200)
            biased_session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=wake_phrase_spotter,
            )
            strict_session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=_MinDurationWakePhraseSpotter(min_duration_ms=200),
            )

            for session in (biased_session, strict_session):
                session.handle_hello(
                    OrchestratorVoiceHelloRequest(
                        session_id="voice-1",
                        sample_rate=16000,
                        channels=1,
                        chunk_ms=100,
                    )
                )

            strict_session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="waiting",
                    detail="idle",
                    follow_up_allowed=False,
                )
            )
            biased_session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="waiting",
                    detail="idle",
                    follow_up_allowed=False,
                    attention_state="attending_to_device",
                    interaction_intent_state="showing_intent",
                    person_visible=True,
                    presence_active=True,
                    interaction_ready=True,
                    targeted_inference_blocked=False,
                    recommended_channel="speech",
                )
            )

            strict_session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            strict_session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(2))
            )
            strict_events = strict_session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=2, pcm_bytes=_pcm_frame(0))
            )

            biased_session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            biased_session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(2))
            )
            biased_events = biased_session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=2, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(strict_events, [])
        self.assertEqual(biased_events[0]["type"], "wake_confirmed")
        self.assertEqual(biased_events[0]["matched_phrase"], "twinna")

    def test_voice_session_familiar_speaker_bias_accepts_tynna_only_with_known_voice_profile(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_remote_asr_min_wake_duration_ms=300,
                voice_orchestrator_intent_min_wake_duration_relief_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            familiar_pcm = _voice_sample_pcm_bytes(frequency_hz=175.0)
            familiar_frames = _pcm_frames_from_audio(familiar_pcm)
            strict_session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(
                    transcribe=lambda *args, **kwargs: "Tynna, wie ist das Wetter heute?"
                ),
            )
            biased_session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(
                    transcribe=lambda *args, **kwargs: "Tynna, wie ist das Wetter heute?"
                ),
            )
            biased_session.handle_identity_profiles(
                _voice_identity_profiles_event(
                    user_id="main_user",
                    display_name="Theo",
                    pcm_bytes=familiar_pcm,
                )
            )

            for session in (strict_session, biased_session):
                session.handle_hello(
                    OrchestratorVoiceHelloRequest(
                        session_id="voice-1",
                        sample_rate=16000,
                        channels=1,
                        chunk_ms=100,
                    )
                )

            strict_session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="waiting",
                    detail="idle",
                    follow_up_allowed=False,
                    attention_state="attending_to_device",
                    interaction_intent_state="showing_intent",
                    person_visible=True,
                    presence_active=True,
                    interaction_ready=True,
                    targeted_inference_blocked=False,
                    recommended_channel="speech",
                )
            )
            biased_session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="waiting",
                    detail="idle",
                    follow_up_allowed=False,
                    attention_state="attending_to_device",
                    interaction_intent_state="showing_intent",
                    person_visible=True,
                    presence_active=True,
                    interaction_ready=True,
                    targeted_inference_blocked=False,
                    recommended_channel="speech",
                    speaker_associated=True,
                    speaker_association_confidence=0.85,
                    background_media_likely=False,
                    speech_overlap_likely=False,
                )
            )

            strict_events: list[dict[str, object]] = []
            biased_events: list[dict[str, object]] = []
            for sequence, frame in enumerate(familiar_frames):
                strict_events = strict_session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=sequence, pcm_bytes=frame)
                )
                biased_events = biased_session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=sequence, pcm_bytes=frame)
                )

            strict_events = strict_session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=len(familiar_frames), pcm_bytes=_pcm_frame(0))
            )
            biased_events = biased_session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=len(familiar_frames), pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(strict_events, [])
        self.assertEqual(biased_events[0]["type"], "wake_confirmed")
        self.assertEqual(biased_events[0]["matched_phrase"], "tynna")

    def test_voice_session_familiar_speaker_bias_accepts_winner_family_only_with_known_voice_profile(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_remote_asr_min_wake_duration_ms=300,
                voice_orchestrator_intent_min_wake_duration_relief_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            familiar_pcm = _voice_sample_pcm_bytes(frequency_hz=175.0)
            familiar_frames = _pcm_frames_from_audio(familiar_pcm)
            strict_session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(
                    transcribe=lambda *args, **kwargs: "Gewinner, wie spät ist es?"
                ),
            )
            biased_session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(
                    transcribe=lambda *args, **kwargs: "Gewinner, wie spät ist es?"
                ),
            )
            biased_session.handle_identity_profiles(
                _voice_identity_profiles_event(
                    user_id="guest_user",
                    display_name="Guest",
                    pcm_bytes=familiar_pcm,
                    primary_user=False,
                )
            )

            for session in (strict_session, biased_session):
                session.handle_hello(
                    OrchestratorVoiceHelloRequest(
                        session_id="voice-1",
                        sample_rate=16000,
                        channels=1,
                        chunk_ms=100,
                    )
                )

            strict_session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="waiting",
                    detail="idle",
                    follow_up_allowed=False,
                    attention_state="attending_to_device",
                    interaction_intent_state="showing_intent",
                    person_visible=True,
                    presence_active=True,
                    interaction_ready=True,
                    targeted_inference_blocked=False,
                    recommended_channel="speech",
                )
            )
            biased_session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="waiting",
                    detail="idle",
                    follow_up_allowed=False,
                    attention_state="attending_to_device",
                    interaction_intent_state="showing_intent",
                    person_visible=True,
                    presence_active=True,
                    interaction_ready=True,
                    targeted_inference_blocked=False,
                    recommended_channel="speech",
                    speaker_associated=True,
                    speaker_association_confidence=0.85,
                    background_media_likely=False,
                    speech_overlap_likely=False,
                )
            )

            strict_events: list[dict[str, object]] = []
            biased_events: list[dict[str, object]] = []
            for sequence, frame in enumerate(familiar_frames):
                strict_events = strict_session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=sequence, pcm_bytes=frame)
                )
                biased_events = biased_session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=sequence, pcm_bytes=frame)
                )

            strict_events = strict_session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=len(familiar_frames), pcm_bytes=_pcm_frame(0))
            )
            biased_events = biased_session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=len(familiar_frames), pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(strict_events, [])
        self.assertEqual(biased_events[0]["type"], "wake_confirmed")
        self.assertEqual(biased_events[0]["matched_phrase"], "twinner")

    def test_voice_session_familiar_speaker_bias_is_blocked_by_background_media(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_remote_asr_min_wake_duration_ms=300,
                voice_orchestrator_intent_min_wake_duration_relief_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            familiar_pcm = _voice_sample_pcm_bytes(frequency_hz=175.0)
            familiar_frames = _pcm_frames_from_audio(familiar_pcm)
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(
                    transcribe=lambda *args, **kwargs: "Tynna, wie ist das Wetter heute?"
                ),
            )
            session.handle_identity_profiles(
                _voice_identity_profiles_event(
                    user_id="main_user",
                    display_name="Theo",
                    pcm_bytes=familiar_pcm,
                )
            )
            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="waiting",
                    detail="idle",
                    follow_up_allowed=False,
                    attention_state="attending_to_device",
                    interaction_intent_state="showing_intent",
                    person_visible=True,
                    presence_active=True,
                    interaction_ready=True,
                    targeted_inference_blocked=False,
                    recommended_channel="speech",
                    speaker_associated=True,
                    speaker_association_confidence=0.85,
                    background_media_likely=True,
                    speech_overlap_likely=False,
                )
            )

            events: list[dict[str, object]] = []
            for sequence, frame in enumerate(familiar_frames):
                events = session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=sequence, pcm_bytes=frame)
                )
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=len(familiar_frames), pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(events, [])

    def test_voice_session_waiting_activation_is_blocked_when_person_state_disallows_speech(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
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
                )
            )
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="waiting",
                    detail="idle",
                    follow_up_allowed=False,
                    attention_state="inactive",
                    interaction_intent_state="passive",
                    person_visible=False,
                    presence_active=False,
                    interaction_ready=False,
                    targeted_inference_blocked=True,
                    recommended_channel="display",
                )
            )

            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(2))
            )
            third = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=2, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(third, [])

    def test_voice_session_waiting_activation_allows_visible_explicit_wake_even_when_room_guard_is_blocked(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
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
                )
            )
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="waiting",
                    detail="idle",
                    follow_up_allowed=False,
                    attention_state="engaged_with_device",
                    interaction_intent_state="passive",
                    person_visible=True,
                    presence_active=True,
                    interaction_ready=False,
                    targeted_inference_blocked=True,
                    recommended_channel="display",
                )
            )

            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(2))
            )
            third = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=2, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(third[0]["type"], "wake_confirmed")
        self.assertEqual(third[0]["matched_phrase"], "twinna")

    def test_voice_session_waiting_activation_allows_attested_presence_without_visible_person(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
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
                )
            )
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="waiting",
                    detail="idle",
                    follow_up_allowed=False,
                    attention_state="inactive",
                    interaction_intent_state="passive",
                    person_visible=False,
                    presence_active=True,
                    interaction_ready=False,
                    targeted_inference_blocked=True,
                    recommended_channel="display",
                )
            )

            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(2))
            )
            third = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=2, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(third[0]["type"], "wake_confirmed")
        self.assertEqual(third[0]["matched_phrase"], "twinna")

    def test_voice_session_waiting_activation_requires_attested_runtime_state(self) -> None:
        with TemporaryDirectory() as temp_dir:
            transcribe_calls: list[str] = []

            def _transcribe(*args, **kwargs) -> str:
                transcribe_calls.append("called")
                return "Twinna wie geht es dir"

            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=300,
                voice_orchestrator_wake_tail_endpoint_silence_ms=300,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=_transcribe),
                wake_phrase_spotter=_MinDurationWakePhraseSpotter(min_duration_ms=300),
            )
            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            for sequence in range(8):
                amplitude = 2 if sequence < 4 else 0
                session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=sequence, pcm_bytes=_pcm_frame(amplitude))
                )

        self.assertEqual(transcribe_calls, [])

    def test_voice_session_audio_frame_runtime_state_refresh_allows_wake_despite_stale_blocked_state(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
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
                )
            )
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="waiting",
                    detail="idle",
                    follow_up_allowed=False,
                    attention_state="inactive",
                    interaction_intent_state="passive",
                    person_visible=False,
                    presence_active=False,
                    interaction_ready=False,
                    targeted_inference_blocked=True,
                    recommended_channel="display",
                )
            )

            refreshed_state = OrchestratorVoiceRuntimeStateEvent(
                state="waiting",
                detail="idle",
                follow_up_allowed=False,
                attention_state="attending_to_device",
                interaction_intent_state="showing_intent",
                person_visible=True,
                presence_active=True,
                interaction_ready=True,
                targeted_inference_blocked=False,
                recommended_channel="speech",
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(
                    sequence=0,
                    pcm_bytes=_pcm_frame(2),
                    runtime_state=refreshed_state,
                )
            )
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(
                    sequence=1,
                    pcm_bytes=_pcm_frame(2),
                    runtime_state=refreshed_state,
                )
            )
            third = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(
                    sequence=2,
                    pcm_bytes=_pcm_frame(0),
                    runtime_state=refreshed_state,
                )
            )

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(third[0]["type"], "wake_confirmed")
        self.assertEqual(third[0]["matched_phrase"], "twinna")

    def test_voice_session_attested_hello_blocks_waiting_activation_before_any_runtime_state_refresh(self) -> None:
        with TemporaryDirectory() as temp_dir:
            transcribe_calls: list[str] = []

            def _transcribe(*args, **kwargs) -> str:
                transcribe_calls.append("called")
                return "untertitel der amara.org-community"

            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=300,
                voice_orchestrator_wake_tail_endpoint_silence_ms=300,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=_transcribe),
                wake_phrase_spotter=_RejectingWakePhraseSpotter(),
            )
            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                    initial_state="waiting",
                    detail="idle",
                    follow_up_allowed=False,
                    person_visible=False,
                    interaction_ready=False,
                    targeted_inference_blocked=True,
                    recommended_channel="display",
                    state_attested=True,
                )
            )
            for sequence in range(8):
                amplitude = 2 if sequence < 4 else 0
                session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=sequence, pcm_bytes=_pcm_frame(amplitude))
                )

        self.assertEqual(transcribe_calls, [])

    def test_voice_session_cancels_pending_waiting_utterance_once_person_state_blocks_speech(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            transcribe_calls: list[str] = []

            def _transcribe(*args, **kwargs) -> str:
                transcribe_calls.append("called")
                return "untertitel der amara.org-community"

            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=300,
                voice_orchestrator_wake_tail_endpoint_silence_ms=300,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=_transcribe),
                wake_phrase_spotter=_RejectingWakePhraseSpotter(),
                monotonic_fn=clock,
            )
            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="waiting",
                    detail="idle",
                    follow_up_allowed=False,
                    attention_state="engaged_with_device",
                    interaction_intent_state="passive",
                    person_visible=True,
                    interaction_ready=False,
                    targeted_inference_blocked=False,
                    recommended_channel="display",
                )
            )
            for sequence in range(4):
                session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=sequence, pcm_bytes=_pcm_frame(2))
                )
            clock.step(EdgeOrchestratorVoiceSession._WAITING_VISIBILITY_GRACE_S + 0.1)
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="waiting",
                    detail="idle",
                    follow_up_allowed=False,
                    attention_state="inactive",
                    interaction_intent_state="passive",
                    person_visible=False,
                    interaction_ready=False,
                    targeted_inference_blocked=True,
                    recommended_channel="display",
                )
            )
            for sequence in range(4, 8):
                session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=sequence, pcm_bytes=_pcm_frame(0))
                )

            transcript_log_path = (
                Path(temp_dir) / "artifacts" / "stores" / "ops" / "voice_gateway_transcripts.jsonl"
            )
            entries = [
                json.loads(line)
                for line in transcript_log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(transcribe_calls, [])
        self.assertEqual(entries[-1]["stage"], "activation_utterance")
        self.assertEqual(entries[-1]["outcome"], "cancelled_context_blocked")
        self.assertFalse(entries[-1]["details"]["intent_person_visible"])
        self.assertFalse(entries[-1]["details"]["intent_interaction_ready"])
        self.assertEqual(entries[-1]["details"]["intent_recommended_channel"], "display")
        self.assertTrue(entries[-1]["details"]["intent_targeted_inference_blocked"])

    def test_voice_session_waiting_visibility_grace_keeps_short_wake_alive_through_brief_false_dip(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_remote_asr_min_wake_duration_ms=300,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=_MinDurationWakePhraseSpotter(min_duration_ms=200),
                monotonic_fn=clock,
            )
            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="waiting",
                    detail="idle",
                    follow_up_allowed=False,
                    attention_state="engaged_with_device",
                    interaction_intent_state="passive",
                    person_visible=True,
                    interaction_ready=False,
                    targeted_inference_blocked=False,
                    recommended_channel="display",
                )
            )

            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(2))
            )
            clock.step(0.2)
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="waiting",
                    detail="idle",
                    follow_up_allowed=False,
                    attention_state="inactive",
                    interaction_intent_state="passive",
                    person_visible=False,
                    interaction_ready=False,
                    targeted_inference_blocked=True,
                    recommended_channel="display",
                )
            )
            third = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=2, pcm_bytes=_pcm_frame(0))
            )

            transcript_log_path = (
                Path(temp_dir) / "artifacts" / "stores" / "ops" / "voice_gateway_transcripts.jsonl"
            )
            entries = [
                json.loads(line)
                for line in transcript_log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(third[0]["type"], "wake_confirmed")
        self.assertEqual(third[0]["matched_phrase"], "twinna")
        self.assertNotIn("cancelled_context_blocked", [entry["outcome"] for entry in entries])

    def test_voice_session_follow_up_context_refresh_uses_open_time_for_bonus_deadline(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_follow_up_timeout_s=6.0,
                voice_orchestrator_intent_follow_up_timeout_bonus_s=2.0,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=_RejectingWakePhraseSpotter(),
                monotonic_fn=clock,
            )
            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="follow_up_open",
                    detail="voice_activation",
                    follow_up_allowed=True,
                )
            )

            clock.step(5.0)
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="follow_up_open",
                    detail="voice_activation",
                    follow_up_allowed=True,
                    attention_state="attending_to_device",
                    interaction_intent_state="showing_intent",
                    person_visible=True,
                    interaction_ready=True,
                    targeted_inference_blocked=False,
                    recommended_channel="speech",
                )
            )
            clock.step(2.0)
            before_deadline = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=b"")
            )
            clock.step(1.1)
            after_deadline = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=b"")
            )

        self.assertEqual(before_deadline, [])
        self.assertEqual(after_deadline[0]["type"], "follow_up_closed")
        self.assertEqual(after_deadline[0]["reason"], "timeout")

    def test_voice_session_normalizes_legacy_wake_armed_to_waiting(self) -> None:
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
            session = EdgeOrchestratorVoiceSession(
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
                )
            )

            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="wake_armed",
                    detail="legacy_state",
                    follow_up_allowed=False,
                )
            )
            self.assertEqual(session._state, "waiting")
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(first, [])
        self.assertEqual(events[0]["type"], "wake_confirmed")

    def test_voice_session_uses_remote_asr_stage1_for_wake_candidates(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
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
                voice_orchestrator_wake_candidate_window_ms=100,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=_FakeWakePhraseSpotter(),
                monotonic_fn=clock,
            )
            self.assertEqual(session.wake_candidate_min_active_ratio, 0.0)

            ready = session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=b"\x01\x00" * 1600)
            )
            clock.step(1.0)
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=b"\x01\x00" * 1600)
            )
            third = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=2, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(ready[0]["backend"], "remote_asr")
        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(third[0]["type"], "wake_confirmed")
        self.assertEqual(third[0]["remaining_text"], "schau mal im web")

    def test_voice_session_remote_asr_stage1_still_requires_transcript_match(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
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
                voice_orchestrator_wake_candidate_window_ms=100,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=_RejectingWakePhraseSpotter(),
                monotonic_fn=clock,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=b"\x01\x00" * 1600)
            )
            clock.step(1.0)
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=b"\x01\x00" * 1600)
            )

        self.assertEqual(first, [])
        self.assertEqual(second, [])

    def test_voice_session_remote_asr_stage1_ignores_sub_threshold_nonzero_noise_bursts(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        class _CountingRejectingWakePhraseSpotter:
            def __init__(self) -> None:
                self.calls = 0

            def detect(self, capture):
                del capture
                self.calls += 1
                return VoiceActivationMatch(
                    detected=False,
                    transcript="noise",
                    normalized_transcript="noise",
                    backend="remote_asr",
                )

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            wake_phrase_spotter = _CountingRejectingWakePhraseSpotter()
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=10,
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_candidate_window_ms=100,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=wake_phrase_spotter,
                monotonic_fn=clock,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            clock.step(1.0)
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(2))
            )
            third = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=2, pcm_bytes=_pcm_frame(0))
            )
            transcript_log_path = (
                Path(temp_dir) / "artifacts" / "stores" / "ops" / "voice_gateway_transcripts.jsonl"
            )
            entries = (
                [
                    json.loads(line)
                    for line in transcript_log_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if transcript_log_path.exists()
                else []
            )

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(third, [])
        self.assertEqual(wake_phrase_spotter.calls, 0)
        self.assertEqual(entries, [])

    def test_voice_session_remote_asr_stage1_counts_quiet_follow_through_after_strong_onset(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            wake_phrase_spotter = _MinDurationWakePhraseSpotter(min_duration_ms=500)
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=10,
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=300,
                voice_orchestrator_wake_candidate_window_ms=300,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=300,
                voice_orchestrator_wake_tail_endpoint_silence_ms=300,
                voice_orchestrator_candidate_cooldown_s=0.0,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=wake_phrase_spotter,
                monotonic_fn=clock,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            events: list[dict[str, object]] = []
            for sequence, frame in enumerate(
                (
                    _pcm_frame(12),
                    _pcm_frame(4),
                    _pcm_frame(4),
                    _pcm_frame(0),
                    _pcm_frame(0),
                    _pcm_frame(0),
                )
            ):
                events = session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=sequence, pcm_bytes=frame)
                )
                clock.step(0.1)

        self.assertEqual(events[0]["type"], "wake_confirmed")
        self.assertEqual(events[0]["matched_phrase"], "twinna")
        self.assertEqual(events[0]["remaining_text"], "wie geht es dir")
        self.assertEqual(wake_phrase_spotter.capture_durations_ms, [600])

    def test_voice_session_remote_asr_stage1_capture_keeps_wake_and_tail_in_one_utterance(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=10,
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_wake_candidate_window_ms=2200,
                voice_orchestrator_follow_up_window_ms=900,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=_FakeWakePhraseSpotter(),
            )

            for frame in (
                _pcm_frame(20),
                _pcm_frame(20),
                _pcm_frame(0),
                _pcm_frame(0),
                _pcm_frame(0),
                _pcm_frame(0),
                _pcm_frame(20),
                _pcm_frame(20),
                _pcm_frame(20),
            ):
                session._remember_frame(frame)

            capture = session._recent_remote_asr_stage1_capture()

        self.assertEqual(capture.sample.duration_ms, 900)
        self.assertEqual(capture.sample.chunk_count, 9)
        self.assertEqual(capture.pcm_bytes[: len(_pcm_frame(20))], _pcm_frame(20))

    def test_voice_session_remote_asr_stage1_waits_for_minimum_wake_burst(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            wake_phrase_spotter = _MinDurationWakePhraseSpotter(min_duration_ms=800)
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=800,
                voice_orchestrator_wake_candidate_window_ms=2200,
                voice_orchestrator_candidate_cooldown_s=0.0,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=wake_phrase_spotter,
                monotonic_fn=clock,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            events: list[dict[str, object]] = []
            for sequence in range(7):
                events = session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=sequence, pcm_bytes=_pcm_frame(2))
                )
                self.assertEqual(events, [])
                clock.step(0.1)
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=7, pcm_bytes=_pcm_frame(2))
            )
            self.assertEqual(events, [])
            clock.step(0.1)
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=8, pcm_bytes=_pcm_frame(2))
            )
            self.assertEqual(events, [])
            clock.step(0.1)
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=9, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(events[0]["type"], "wake_confirmed")
        self.assertEqual(events[0]["matched_phrase"], "twinna")
        self.assertEqual(events[0]["remaining_text"], "wie geht es dir")
        self.assertEqual(wake_phrase_spotter.capture_durations_ms, [1000])

    def test_voice_session_remote_asr_stage1_persists_buffering_debug_entry(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

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
                voice_orchestrator_remote_asr_min_wake_duration_ms=300,
                voice_orchestrator_wake_candidate_window_ms=2200,
                voice_orchestrator_candidate_cooldown_s=0.0,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=_FakeWakePhraseSpotter(),
                monotonic_fn=_FakeClock(),
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )

            transcript_log_path = (
                Path(temp_dir) / "artifacts" / "stores" / "ops" / "voice_gateway_transcripts.jsonl"
            )
            entries = [
                json.loads(line)
                for line in transcript_log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(events, [])
        self.assertEqual(entries[-1]["stage"], "activation_utterance")
        self.assertEqual(entries[-1]["outcome"], "buffering_short_utterance")
        self.assertEqual(entries[-1]["sample"]["duration_ms"], 100)
        self.assertEqual(entries[-1]["details"]["required_active_ms"], 300)

    def test_voice_session_remote_asr_stage1_persists_raw_match_debug_entry(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

        with TemporaryDirectory() as temp_dir:
            wake_phrase_spotter = _ShortPrefixWakePhraseSpotter(min_duration_ms=100, max_duration_ms=200)
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
                voice_orchestrator_wake_candidate_window_ms=100,
                voice_orchestrator_candidate_cooldown_s=0.0,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=wake_phrase_spotter,
                monotonic_fn=_FakeClock(),
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            self.assertEqual(events, [])
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(0))
            )

            transcript_log_path = (
                Path(temp_dir) / "artifacts" / "stores" / "ops" / "voice_gateway_transcripts.jsonl"
            )
            entries = [
                json.loads(line)
                for line in transcript_log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(events[0]["type"], "wake_confirmed")
        self.assertEqual(entries[-1]["stage"], "activation_utterance")
        self.assertEqual(entries[-1]["outcome"], "matched")
        self.assertEqual(entries[-1]["transcript"], "Twinner")
        self.assertEqual(entries[-1]["matched_phrase"], "twinna")
        self.assertEqual(entries[-1]["remaining_text"], "")

    def test_voice_session_remote_asr_stage1_persists_raw_no_match_debug_entry(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

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
                voice_orchestrator_wake_candidate_window_ms=100,
                voice_orchestrator_candidate_cooldown_s=0.0,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=_RejectingWakePhraseSpotter(),
                monotonic_fn=_FakeClock(),
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            self.assertEqual(events, [])
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(0))
            )

            transcript_log_path = (
                Path(temp_dir) / "artifacts" / "stores" / "ops" / "voice_gateway_transcripts.jsonl"
            )
            entries = [
                json.loads(line)
                for line in transcript_log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(events, [])
        self.assertEqual(entries[-1]["stage"], "activation_utterance")
        self.assertEqual(entries[-1]["outcome"], "no_match")
        self.assertEqual(entries[-1]["transcript"], "Maurizio Puppe")

    def test_voice_session_remote_asr_stage1_persists_opt_in_audio_artifact(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

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
                voice_orchestrator_wake_candidate_window_ms=100,
                voice_orchestrator_candidate_cooldown_s=0.0,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
                voice_orchestrator_audio_debug_enabled=True,
                voice_orchestrator_audio_debug_max_files=8,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=_RejectingWakePhraseSpotter(),
                monotonic_fn=_FakeClock(),
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            self.assertEqual(events, [])
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(0))
            )

            transcript_log_path = (
                Path(temp_dir) / "artifacts" / "stores" / "ops" / "voice_gateway_transcripts.jsonl"
            )
            entries = [
                json.loads(line)
                for line in transcript_log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            artifact_path = Path(entries[-1]["details"]["audio_artifact_path"])
            artifact_exists = artifact_path.is_file()
            artifact_suffix = artifact_path.suffix
            artifact_prefix = artifact_path.read_bytes()[:4] if artifact_exists else b""

        self.assertEqual(events, [])
        self.assertEqual(entries[-1]["stage"], "activation_utterance")
        self.assertTrue(artifact_exists)
        self.assertEqual(artifact_suffix, ".wav")
        self.assertEqual(artifact_prefix, b"RIFF")
        self.assertEqual(entries[-1]["details"]["audio_artifact_duration_ms"], 200)

    def test_voice_session_remote_asr_stage1_persists_backend_error_debug_entry(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

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
                voice_orchestrator_wake_candidate_window_ms=100,
                voice_orchestrator_candidate_cooldown_s=0.0,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=_BusyWakePhraseSpotter(),
                monotonic_fn=_FakeClock(),
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            self.assertEqual(events, [])
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(0))
            )

            transcript_log_path = (
                Path(temp_dir) / "artifacts" / "stores" / "ops" / "voice_gateway_transcripts.jsonl"
            )
            entries = [
                json.loads(line)
                for line in transcript_log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(events, [])
        self.assertEqual(entries[-1]["stage"], "activation_utterance")
        self.assertEqual(entries[-1]["outcome"], "backend_error")
        self.assertEqual(entries[-1]["details"]["error_type"], "RuntimeError")
        self.assertIn("stt busy", entries[-1]["details"]["error_message"])

    def test_voice_session_remote_asr_stage1_confirms_wake_only_turn_after_endpoint_silence(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            wake_phrase_spotter = _MinDurationWakePhraseSpotter(min_duration_ms=800, remaining_text="")
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=800,
                voice_orchestrator_wake_candidate_window_ms=2200,
                voice_orchestrator_candidate_cooldown_s=0.0,
                voice_orchestrator_wake_postroll_ms=250,
                voice_orchestrator_wake_tail_endpoint_silence_ms=300,
                voice_orchestrator_wake_tail_max_ms=2200,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=wake_phrase_spotter,
                monotonic_fn=clock,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            for sequence in range(8):
                events = session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=sequence, pcm_bytes=_pcm_frame(2))
                )
                self.assertEqual(events, [])
                clock.step(0.1)
            for silence_index in range(3):
                events = session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=8 + silence_index, pcm_bytes=_pcm_frame(0))
                )
                clock.step(0.1)

        self.assertEqual(events[0]["type"], "wake_confirmed")
        self.assertEqual(events[0]["matched_phrase"], "twinna")
        self.assertEqual(events[0]["remaining_text"], "")
        self.assertEqual(wake_phrase_spotter.capture_durations_ms, [1100])

    def test_voice_session_remote_asr_stage1_expands_with_available_history_up_to_short_scan_window(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            wake_phrase_spotter = _LengthSensitiveWakePhraseSpotter(min_duration_ms=200)
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_history_ms=300,
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_candidate_window_ms=100,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=wake_phrase_spotter,
                monotonic_fn=clock,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=b"\x01\x00" * 1600)
            )
            clock.step(1.0)
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=b"\x01\x00" * 1600)
            )
            third = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=2, pcm_bytes=b"\x01\x00" * 1600)
            )
            fourth = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=3, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(third, [])
        self.assertEqual(fourth[0]["type"], "wake_confirmed")
        self.assertEqual(wake_phrase_spotter.capture_durations_ms, [400])

    def test_voice_session_remote_asr_stage1_uses_short_prefix_before_full_sentence_capture(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            wake_phrase_spotter = _GrowingWakePhraseSpotter()
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_history_ms=2400,
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_candidate_window_ms=2200,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=wake_phrase_spotter,
                monotonic_fn=clock,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            for sequence in range(10):
                session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=sequence, pcm_bytes=b"\x01\x00" * 1600)
                )
            wake_candidate = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=10, pcm_bytes=b"\x01\x00" * 1600)
            )
            confirmed = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=11, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(wake_candidate, [])
        self.assertEqual(confirmed[0]["type"], "wake_confirmed")
        self.assertEqual(confirmed[0]["matched_phrase"], "twinna")
        self.assertEqual(confirmed[0]["remaining_text"], "schau mal im web")
        self.assertEqual(wake_phrase_spotter.capture_durations_ms, [1200])

    def test_voice_session_remote_asr_stage1_caps_scans_to_short_window(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            wake_phrase_spotter = _GrowingWakePhraseSpotter()
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_history_ms=3200,
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_candidate_window_ms=2200,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
                voice_orchestrator_candidate_cooldown_s=0.9,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=wake_phrase_spotter,
                monotonic_fn=clock,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            final_events: list[dict[str, object]] = []
            for sequence in range(27):
                events = session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=sequence, pcm_bytes=b"\x01\x00" * 1600)
                )
                if events:
                    final_events = events
                clock.step(0.1)
            final_events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=27, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(final_events[0]["type"], "wake_confirmed")
        self.assertEqual(final_events[0]["matched_phrase"], "twinna")
        self.assertEqual(final_events[0]["remaining_text"], "schau mal im web")
        self.assertTrue(wake_phrase_spotter.capture_durations_ms)
        self.assertEqual(wake_phrase_spotter.capture_durations_ms, [2800])

    def test_voice_session_remote_asr_same_stream_wake_waits_for_endpoint_silence(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
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
                voice_orchestrator_wake_candidate_window_ms=100,
                voice_orchestrator_wake_postroll_ms=250,
                voice_orchestrator_wake_tail_max_ms=250,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=_FakeWakePhraseSpotter(),
                monotonic_fn=clock,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            events: list[dict[str, object]] = []
            for sequence in range(4):
                events = session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=sequence, pcm_bytes=b"\x01\x00" * 1600)
                )
                clock.step(0.1)
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=4, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(events[0]["type"], "wake_confirmed")
        self.assertEqual(events[0]["remaining_text"], "schau mal im web")

    def test_voice_session_remote_asr_stage1_delays_next_scan_until_slow_scan_finishes(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        class _SlowRejectingWakePhraseSpotter:
            def __init__(self, clock: _FakeClock) -> None:
                self.clock = clock
                self.calls = 0

            def detect(self, capture):
                del capture
                self.calls += 1
                self.clock.step(0.5)
                return VoiceActivationMatch(
                    detected=False,
                    transcript="",
                    backend="remote_asr",
                )

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            wake_phrase_spotter = _SlowRejectingWakePhraseSpotter(clock)
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
                voice_orchestrator_wake_candidate_window_ms=2200,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=wake_phrase_spotter,
                monotonic_fn=clock,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=b"\x01\x00" * 1600)
            )
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=b"\x01\x00" * 1600)
            )
            third = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=2, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(third, [])
        self.assertEqual(wake_phrase_spotter.calls, 1)

    def test_voice_session_remote_asr_same_stream_wake_returns_remaining_text_without_tail_stage(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            wake_phrase_spotter = _GrowingWakePhraseSpotter()
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
                voice_orchestrator_wake_candidate_window_ms=100,
                voice_orchestrator_wake_postroll_ms=250,
                voice_orchestrator_wake_tail_max_ms=250,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=wake_phrase_spotter,
                monotonic_fn=clock,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            events: list[dict[str, object]] = []
            for sequence in range(4):
                events = session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=sequence, pcm_bytes=b"\x01\x00" * 1600)
                )
                clock.step(0.1)
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=4, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(events[0]["type"], "wake_confirmed")
        self.assertEqual(events[0]["remaining_text"], "schau mal im web")
        self.assertEqual(wake_phrase_spotter.capture_durations_ms, [500])

    def test_voice_session_remote_asr_stage1_anchors_short_scan_at_latest_speech_burst_start(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            wake_frame = (1000).to_bytes(2, "little", signed=True) * 1600
            tail_frame = (2000).to_bytes(2, "little", signed=True) * 1600
            silence_frame = b"\x00\x00" * 1600
            wake_phrase_spotter = _BurstAnchoredWakePhraseSpotter(
                wake_prefix=wake_frame[:16],
                min_duration_ms=900,
                max_duration_ms=2500,
            )
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=800,
                voice_orchestrator_history_ms=3200,
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_candidate_window_ms=2200,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=wake_phrase_spotter,
                monotonic_fn=clock,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            frames = [silence_frame] * 4 + [wake_frame] * 3 + [tail_frame] * 15
            for sequence, pcm_bytes in enumerate(frames):
                session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=sequence, pcm_bytes=pcm_bytes)
                )
            wake_candidate = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=len(frames), pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(wake_candidate[0]["type"], "wake_confirmed")
        self.assertEqual(wake_candidate[0]["matched_phrase"], "twinna")
        self.assertEqual(wake_candidate[0]["remaining_text"], "")
        self.assertTrue(wake_phrase_spotter.capture_prefixes)
        self.assertEqual(wake_phrase_spotter.capture_prefixes[-1], wake_frame[:16])
        self.assertEqual(wake_phrase_spotter.capture_durations_ms, [1900])

    def test_voice_session_remote_asr_stage1_preserves_quiet_onset_before_active_burst(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            quiet_onset_frame = (600).to_bytes(2, "little", signed=True) * 1600
            active_wake_frame = (1200).to_bytes(2, "little", signed=True) * 1600
            tail_frame = (2000).to_bytes(2, "little", signed=True) * 1600
            silence_frame = b"\x00\x00" * 1600
            wake_phrase_spotter = _BurstAnchoredWakePhraseSpotter(
                wake_prefix=quiet_onset_frame[:16],
                min_duration_ms=900,
                max_duration_ms=2500,
            )
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=800,
                voice_orchestrator_history_ms=3200,
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_candidate_window_ms=2200,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=wake_phrase_spotter,
                monotonic_fn=clock,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            frames = [silence_frame] * 4 + [quiet_onset_frame] + [active_wake_frame] * 2 + [tail_frame] * 15
            for sequence, pcm_bytes in enumerate(frames):
                session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=sequence, pcm_bytes=pcm_bytes)
                )
            wake_candidate = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=len(frames), pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(wake_candidate[0]["type"], "wake_confirmed")
        self.assertEqual(wake_candidate[0]["matched_phrase"], "twinna")
        self.assertEqual(wake_candidate[0]["remaining_text"], "")
        self.assertTrue(wake_phrase_spotter.capture_prefixes)
        self.assertEqual(wake_phrase_spotter.capture_prefixes[-1], quiet_onset_frame[:16])
        self.assertEqual(wake_phrase_spotter.capture_durations_ms, [1900])

    def test_voice_session_remote_asr_stage1_preserves_nonzero_onset_below_half_threshold(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            quiet_onset_frame = (624).to_bytes(2, "little", signed=True) * 1600
            active_wake_frame = (2783).to_bytes(2, "little", signed=True) * 1600
            tail_frame = (3853).to_bytes(2, "little", signed=True) * 1600
            silence_frame = b"\x00\x00" * 1600
            wake_phrase_spotter = _BurstAnchoredWakePhraseSpotter(
                wake_prefix=quiet_onset_frame[:16],
                min_duration_ms=900,
                max_duration_ms=2500,
            )
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1500,
                voice_orchestrator_history_ms=3200,
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_candidate_window_ms=2200,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=wake_phrase_spotter,
                monotonic_fn=clock,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            frames = [silence_frame] * 4 + [quiet_onset_frame] + [active_wake_frame] * 2 + [tail_frame] * 15
            for sequence, pcm_bytes in enumerate(frames):
                session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=sequence, pcm_bytes=pcm_bytes)
                )
            wake_candidate = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=len(frames), pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(wake_candidate[0]["type"], "wake_confirmed")
        self.assertEqual(wake_candidate[0]["matched_phrase"], "twinna")
        self.assertEqual(wake_candidate[0]["remaining_text"], "")
        self.assertTrue(wake_phrase_spotter.capture_prefixes)
        self.assertEqual(wake_phrase_spotter.capture_prefixes[-1], quiet_onset_frame[:16])
        self.assertEqual(wake_phrase_spotter.capture_durations_ms, [1900])

    def test_voice_session_remote_asr_mode_requires_service_url(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
            )

            with self.assertRaises(ValueError):
                EdgeOrchestratorVoiceSession(config)

    def test_voice_session_remote_asr_mode_does_not_require_openai_backend(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
            )

            session = EdgeOrchestratorVoiceSession(config)
            ready = session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )

        self.assertEqual(ready[0]["type"], "voice_ready")
        self.assertEqual(ready[0]["backend"], "remote_asr")


class OrchestratorContractTests(unittest.TestCase):
    def test_turn_complete_payload_serializes_model_like_token_usage(self) -> None:
        @dataclass
        class TokenUsage:
            total_tokens: int = 42

        token_usage = TokenUsage()
        payload = OrchestratorTurnCompleteEvent(
            text="Hallo!",
            rounds=1,
            used_web_search=False,
            token_usage=token_usage,
        ).to_payload()
        self.assertEqual(payload["token_usage"], {"total_tokens": 42})

    def test_tool_response_payload_sanitizes_nested_non_json_output(self) -> None:
        payload = OrchestratorToolResponse(
            call_id="call-1",
            ok=True,
            output={"meta": {"path": Path("/tmp/test.txt")}},
        ).to_payload()
        self.assertEqual(payload["output"], {"meta": {"path": "/tmp/test.txt"}})


if __name__ == "__main__":
    unittest.main()
