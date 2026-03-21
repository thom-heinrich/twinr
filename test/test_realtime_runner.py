from array import array
from dataclasses import replace
from datetime import datetime, timedelta
import io
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import math
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock
import wave
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.automations import AutomationAction, AutomationStore
from twinr.agent.base_agent.contracts import (
    AgentToolCall,
    StreamingSpeechEndpointEvent,
    StreamingTranscriptionResult,
    ToolCallingTurnResponse,
)
from twinr.config import TwinrConfig
from twinr.hardware import VoiceAssessment
from twinr.hardware.buttons import ButtonAction, ButtonEvent
from twinr.memory.longterm.core.models import (
    LongTermConsolidationResultV1,
    LongTermMemoryConflictV1,
    LongTermMemoryObjectV1,
    LongTermReflectionResultV1,
    LongTermSourceRefV1,
)
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStatus, LongTermRemoteUnavailableError
from twinr.memory.reminders import now_in_timezone
from twinr.proactive import SocialTriggerDecision, SocialTriggerPriority, WakewordMatch
from twinr.proactive.runtime.audio_policy import ReSpeakerAudioPolicySnapshot
from twinr.proactive.runtime.runtime_contract import ReSpeakerRuntimeContractError
from twinr.providers.openai import OpenAITextResponse
from twinr.providers.openai.realtime import OpenAIRealtimeTurn
from twinr.realtime_runner import TwinrRealtimeHardwareLoop
from twinr.runtime import TwinrRuntime
from twinr.state_machine import TwinrStatus
from twinr.agent.base_agent.conversation.closure import ConversationClosureDecision
from twinr.agent.personality.intelligence import WorldFeedItem
from twinr.agent.personality.steering import ConversationTurnSteeringCue
from twinr.agent.self_coding import (
    ArtifactKind,
    CompileTarget,
    FeasibilityOutcome,
    FeasibilityResult,
    RequirementsDialogueSession,
    RequirementsDialogueStatus,
    SelfCodingActivationService,
    SelfCodingCompileWorker,
    SelfCodingStore,
)
from twinr.agent.self_coding.codex_driver import CodexCompileArtifact, CodexCompileResult
from twinr.hardware.audio import (
    AmbientAudioCaptureWindow,
    AmbientAudioLevelSample,
    ListenTimeoutCaptureDiagnostics,
    SpeechCaptureResult,
    SpeechStartTimeoutError,
    normalize_wav_playback_level,
)
from twinr.agent.workflows.button_dispatch import ButtonPressDispatcher


def _voice_sample_pcm_bytes(*, frequency_hz: float = 175.0, amplitude: float = 0.35, duration_s: float = 1.8) -> bytes:
    sample_rate = 24000
    total_frames = int(sample_rate * duration_s)
    frames = array("h")
    for index in range(total_frames):
        t = index / sample_rate
        envelope = min(1.0, index / (sample_rate * 0.2), (total_frames - index) / (sample_rate * 0.2))
        sample = amplitude * envelope * (
            (0.70 * math.sin(2.0 * math.pi * frequency_hz * t))
            + (0.20 * math.sin(2.0 * math.pi * frequency_hz * 2.0 * t))
            + (0.10 * math.sin(2.0 * math.pi * (frequency_hz + 35.0) * t))
        )
        frames.append(max(-32767, min(32767, int(sample * 32767))))
    return frames.tobytes()


def _pcm_to_test_wav_bytes(pcm_bytes: bytes, *, sample_rate: int = 24000, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm_bytes)
    return buffer.getvalue()


def _longterm_source(event_id: str) -> LongTermSourceRefV1:
    return LongTermSourceRefV1(
        source_type="conversation_turn",
        event_ids=(event_id,),
        speaker="user",
        modality="voice",
    )


def _fresh_checked_at() -> str:
    return datetime.now(ZoneInfo("UTC")).replace(microsecond=0).isoformat()


class FakeRealtimeSession:
    def __init__(self) -> None:
        self.calls: list[bytes] = []
        self.text_calls: list[str] = []
        self.conversations: list[tuple[tuple[str, str], ...] | None] = []
        self.entered = False
        self.exited = False
        self.turns = [
            OpenAIRealtimeTurn(
                transcript="Hallo Twinr",
                response_text="Guten Tag",
                response_id="resp_rt_123",
                end_conversation=False,
            )
        ]

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.exited = True

    def run_audio_turn(
        self,
        audio_pcm: bytes,
        *,
        conversation=None,
        on_audio_chunk=None,
        on_output_text_delta=None,
    ) -> OpenAIRealtimeTurn:
        self.calls.append(audio_pcm)
        self.conversations.append(conversation)
        if on_output_text_delta is not None:
            on_output_text_delta("Guten ")
            on_output_text_delta("Tag")
        if on_audio_chunk is not None:
            on_audio_chunk(b"PCM")
        if self.turns:
            return self.turns.pop(0)
        return OpenAIRealtimeTurn(
            transcript="Hallo Twinr",
            response_text="Guten Tag",
            response_id="resp_rt_123",
            end_conversation=False,
        )

    def run_text_turn(
        self,
        prompt: str,
        *,
        conversation=None,
        on_audio_chunk=None,
        on_output_text_delta=None,
    ) -> OpenAIRealtimeTurn:
        self.text_calls.append(prompt)
        self.conversations.append(conversation)
        if on_output_text_delta is not None:
            on_output_text_delta("Guten ")
            on_output_text_delta("Tag")
        if on_audio_chunk is not None:
            on_audio_chunk(b"PCM")
        if self.turns:
            return self.turns.pop(0)
        return OpenAIRealtimeTurn(
            transcript=prompt,
            response_text="Guten Tag",
            response_id="resp_rt_123",
            end_conversation=False,
        )


class FakePrintBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[tuple[str, str], ...] | None, str | None, str | None, str]] = []
        self.search_calls: list[tuple[str, tuple[tuple[str, str], ...] | None, str | None, str | None]] = []
        self.vision_calls: list[tuple[str, list[object], tuple[tuple[str, str], ...] | None, bool | None]] = []
        self.summary_calls: list[str] = []
        self.search_sleep_s = 0.0
        self.synthesize_sleep_s = 0.0
        self.synthesize_calls: list[str] = []
        self.synthesize_result = b"WAVPCM"
        self.reminder_calls: list[object] = []
        self.generic_reminder_calls: list[tuple[str, str | None, bool | None]] = []
        self.automation_calls: list[tuple[str, bool, str]] = []
        self.proactive_calls: list[tuple[str, str, str, int, tuple[tuple[str, str], ...] | None, tuple[str, ...]]] = []
        self.transcribe_calls: list[tuple[bytes, str | None, str | None]] = []
        self.transcribe_result = "hey twinr"

    def compose_print_job_with_metadata(
        self,
        *,
        conversation=None,
        focus_hint: str | None = None,
        direct_text: str | None = None,
        request_source: str = "button",
    ) -> OpenAITextResponse:
        self.calls.append((conversation, focus_hint, direct_text, request_source))
        return OpenAITextResponse(text="GUTEN TAG")

    def search_live_info_with_metadata(
        self,
        question: str,
        *,
        conversation=None,
        location_hint: str | None = None,
        date_context: str | None = None,
    ):
        self.search_calls.append((question, conversation, location_hint, date_context))
        if self.search_sleep_s > 0:
            time.sleep(self.search_sleep_s)
        return SimpleNamespace(
            answer="Bus 24 faehrt um 07:30 Uhr.",
            sources=("https://example.com/fahrplan",),
            response_id="resp_search_1",
            request_id="req_search_1",
            used_web_search=True,
            model="gpt-5.2-chat-latest",
            token_usage=None,
        )

    def respond_to_images_with_metadata(
        self,
        prompt: str,
        *,
        images,
        conversation=None,
        allow_web_search=None,
        instructions=None,
    ) -> OpenAITextResponse:
        self.vision_calls.append((prompt, list(images), conversation, allow_web_search))
        return OpenAITextResponse(text="Ich sehe die Kameraansicht.")

    def respond_with_metadata(self, prompt: str, *, instructions=None, allow_web_search=None) -> OpenAITextResponse:
        del instructions, allow_web_search
        self.summary_calls.append(prompt)
        return OpenAITextResponse(text="Guten Morgen. Hier ist dein Morgenabstract.")

    def synthesize_stream(self, text: str):
        self.synthesize_calls.append(text)
        if self.synthesize_sleep_s > 0:
            time.sleep(self.synthesize_sleep_s)
        yield b"PCM"

    def synthesize(self, text: str, *, voice=None, response_format=None, instructions=None) -> bytes:
        del voice, response_format, instructions
        self.synthesize_calls.append(text)
        if self.synthesize_sleep_s > 0:
            time.sleep(self.synthesize_sleep_s)
        return self.synthesize_result

    def phrase_due_reminder_with_metadata(self, reminder):
        self.reminder_calls.append(reminder)
        return OpenAITextResponse(
            text=f"Erinnerung: {reminder.summary}",
            response_id="resp_reminder_1",
            request_id="req_reminder_1",
            used_web_search=False,
        )

    def phrase_proactive_prompt_with_metadata(
        self,
        *,
        trigger_id: str,
        reason: str,
        default_prompt: str,
        priority: int,
        conversation=None,
        recent_prompts=(),
        observation_facts=(),
    ) -> OpenAITextResponse:
        self.proactive_calls.append(
            (
                trigger_id,
                reason,
                default_prompt,
                priority,
                conversation,
                tuple(recent_prompts),
                tuple(observation_facts),
            )
        )
        return OpenAITextResponse(
            text=f"Proaktiv: {trigger_id}",
            response_id="resp_social_1",
            request_id="req_social_1",
            model="gpt-5.2",
            token_usage=None,
            used_web_search=False,
        )

    def fulfill_automation_prompt_with_metadata(
        self,
        prompt: str,
        *,
        allow_web_search: bool,
        delivery: str = "spoken",
    ) -> OpenAITextResponse:
        self.automation_calls.append((prompt, allow_web_search, delivery))
        text = "Die Wettervorhersage ist trocken und mild."
        if delivery == "printed":
            text = "Schlagzeilen: Markt ruhig. Wetter mild."
        return OpenAITextResponse(
            text=text,
            response_id="resp_auto_1",
            request_id="req_auto_1",
            used_web_search=allow_web_search,
            model="gpt-5.2",
            token_usage=None,
        )

    def respond_with_metadata(
        self,
        prompt: str,
        *,
        conversation=None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
    ) -> OpenAITextResponse:
        self.generic_reminder_calls.append((prompt, instructions, allow_web_search))
        return OpenAITextResponse(
            text="Erinnerung: Medikament nehmen",
            response_id="resp_generic_1",
            request_id="req_generic_1",
            used_web_search=False,
            model="gpt-5.2",
            token_usage=None,
        )

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        self.transcribe_calls.append((audio_bytes, language, prompt))
        return self.transcribe_result


def _self_coding_briefing_skill_code() -> str:
    return """
from __future__ import annotations


SEARCH_TERMS = (
    "seniorenpolitik deutschland",
    "pflege gesundheit deutschland",
    "lokale nachrichten schleswig-holstein",
)


def refresh_briefing(ctx):
    research_lines = []
    for search_term in SEARCH_TERMS:
        result = ctx.search_web(search_term)
        research_lines.append(f"{search_term}: {result.answer}")
    abstract = ctx.summarize_text("\\n".join(research_lines), instructions="Schreibe einen kurzen deutschen Morgenabstract.")
    ctx.store_json(
        f"briefing:{ctx.today_local_date()}",
        {
            "abstract": abstract,
            "delivered_date": None,
        },
    )


def deliver_briefing(ctx, *, event_name=None):
    payload = ctx.load_json(f"briefing:{ctx.today_local_date()}", {}) or {}
    abstract = str(payload.get("abstract") or "").strip()
    if not abstract:
        return
    if payload.get("delivered_date") == ctx.today_local_date():
        return
    if ctx.is_night_mode() or not ctx.is_private_for_speech():
        return
    ctx.say(abstract)
    payload["delivered_date"] = ctx.today_local_date()
    payload["delivered_event_name"] = event_name
    ctx.store_json(f"briefing:{ctx.today_local_date()}", payload)
""".strip()


def _self_coding_skill_package_payload() -> str:
    return json.dumps(
        {
            "skill_package": {
                "name": "Morning Briefing",
                "description": "Research three topics at 08:00 and read an abstract aloud when I enter the room.",
                "entry_module": "skill_main.py",
                "scheduled_triggers": [
                    {
                        "trigger_id": "refresh_briefing",
                        "schedule": "daily",
                        "time_of_day": "08:00",
                        "timezone_name": "Europe/Berlin",
                        "handler": "refresh_briefing",
                    }
                ],
                "sensor_triggers": [
                    {
                        "trigger_id": "deliver_briefing",
                        "sensor_trigger_kind": "camera_person_visible",
                        "cooldown_seconds": 30,
                        "handler": "deliver_briefing",
                    }
                ],
                "files": [
                    {
                        "path": "skill_main.py",
                        "content": _self_coding_briefing_skill_code(),
                    }
                ],
            }
        }
    )


def _ready_self_coding_skill_session() -> RequirementsDialogueSession:
    return RequirementsDialogueSession(
        session_id="dialogue_loop_briefing",
        request_summary="Research three morning topics at 08:00 and read the abstract aloud when I enter the room.",
        skill_name="Morning Briefing",
        action="Research three morning topics, prepare an abstract, and read it aloud when I enter the room.",
        capabilities=("web_search", "llm_call", "memory", "speaker", "camera", "scheduler", "safety"),
        feasibility=FeasibilityResult(
            outcome=FeasibilityOutcome.YELLOW,
            summary="This request needs the later skill-package path.",
            suggested_target=CompileTarget.SKILL_PACKAGE,
        ),
        status=RequirementsDialogueStatus.READY_FOR_COMPILE,
        trigger_mode="push",
        trigger_conditions=("camera_person_visible", "daily_0800"),
        scope={"channel": "voice", "time_of_day": "08:00"},
        constraints=("read_once_per_morning",),
    )


class FakeRecorder:
    def __init__(self, recordings: list[bytes | Exception] | None = None) -> None:
        self.pause_values: list[int] = []
        self.start_timeouts: list[float | None] = []
        self.speech_start_chunks: list[int | None] = []
        self.ignore_initial_ms: list[int] = []
        self.pause_grace_values: list[int] = []
        self.recordings = list(recordings or [b"PCMINPUT"])

    def capture_pcm_until_pause_with_options(
        self,
        *,
        pause_ms: int,
        start_timeout_s: float | None = None,
        max_record_seconds: float | None = None,
        speech_start_chunks: int | None = None,
        ignore_initial_ms: int = 0,
        pause_grace_ms: int = 0,
        on_chunk=None,
        should_stop=None,
    ):
        del max_record_seconds
        self.pause_values.append(pause_ms)
        self.start_timeouts.append(start_timeout_s)
        self.speech_start_chunks.append(speech_start_chunks)
        self.ignore_initial_ms.append(ignore_initial_ms)
        self.pause_grace_values.append(pause_grace_ms)
        if on_chunk is not None:
            on_chunk(b"PCM-A")
            on_chunk(b"PCM-B")
        if should_stop is not None:
            for _ in range(50):
                if should_stop():
                    break
                time.sleep(0.001)
        if not self.recordings:
            value: bytes | Exception = b"PCMINPUT"
        else:
            value = self.recordings.pop(0)
        if isinstance(value, Exception):
            raise value
        if hasattr(value, "pcm_bytes"):
            return value
        return SimpleNamespace(
            pcm_bytes=value,
            speech_started_after_ms=2100,
            resumed_after_pause_count=0,
        )

    def record_pcm_until_pause_with_options(
        self,
        *,
        pause_ms: int,
        start_timeout_s: float | None = None,
        max_record_seconds: float | None = None,
        speech_start_chunks: int | None = None,
        ignore_initial_ms: int = 0,
        pause_grace_ms: int = 0,
    ) -> bytes:
        result = self.capture_pcm_until_pause_with_options(
            pause_ms=pause_ms,
            start_timeout_s=start_timeout_s,
            max_record_seconds=max_record_seconds,
            speech_start_chunks=speech_start_chunks,
            ignore_initial_ms=ignore_initial_ms,
            pause_grace_ms=pause_grace_ms,
        )
        return result.pcm_bytes


class FakeTurnStreamingSpeechSession:
    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self.closed = False
        self._on_endpoint = None
        self.finalize_calls = 0

    def send_pcm(self, pcm_bytes: bytes) -> None:
        self.sent.append(pcm_bytes)
        if len(self.sent) == 2 and self._on_endpoint is not None:
            self._on_endpoint(
                StreamingSpeechEndpointEvent(
                    transcript="ich bin immernoch am programmieren, nur damit du es weisst",
                    event_type="speech_final",
                    speech_final=True,
                )
            )

    def snapshot(self) -> StreamingTranscriptionResult:
        return StreamingTranscriptionResult(
            transcript="ich bin immernoch am programmieren, nur damit du es weisst",
            request_id="turn-stt-1",
            saw_interim=True,
            saw_speech_final=True,
            saw_utterance_end=False,
        )

    def finalize(self) -> StreamingTranscriptionResult:
        self.finalize_calls += 1
        return StreamingTranscriptionResult(
            transcript="ich bin immernoch am programmieren, nur damit du es weisst",
            request_id="turn-stt-1",
            saw_interim=True,
            saw_speech_final=True,
            saw_utterance_end=False,
        )

    def close(self) -> None:
        self.closed = True


class FakeTurnStreamingSpeechToTextProvider:
    def __init__(self, config: TwinrConfig) -> None:
        self.config = config
        self.session = FakeTurnStreamingSpeechSession()
        self.start_calls: list[dict[str, object]] = []

    def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
        del audio_bytes, kwargs
        return "ich bin immernoch am programmieren, nur damit du es weisst"

    def transcribe_path(self, path, **kwargs) -> str:
        del path, kwargs
        return "ich bin immernoch am programmieren, nur damit du es weisst"

    def start_streaming_session(
        self,
        *,
        sample_rate: int,
        channels: int,
        language: str | None = None,
        prompt: str | None = None,
        on_interim=None,
        on_endpoint=None,
    ):
        del prompt
        self.start_calls.append(
            {
                "sample_rate": sample_rate,
                "channels": channels,
                "language": language,
            }
        )
        self.session._on_endpoint = on_endpoint
        if on_interim is not None:
            on_interim("ich bin immernoch")
        return self.session


class FakeTurnToolAgentProvider:
    def __init__(self, config: TwinrConfig) -> None:
        self.config = config
        self.start_calls: list[dict[str, object]] = []

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
        del on_text_delta
        self.start_calls.append(
            {
                "prompt": prompt,
                "conversation": conversation,
                "instructions": instructions,
                "tool_schemas": list(tool_schemas),
                "allow_web_search": allow_web_search,
            }
        )
        arguments = {
            "decision": "end_turn",
            "label": "complete",
            "confidence": 0.94,
            "reason": "complete_request",
            "transcript": "ich bin immernoch am programmieren, nur damit du es weisst",
        }
        return ToolCallingTurnResponse(
            text="",
            tool_calls=(
                AgentToolCall(
                    name="submit_turn_decision",
                    call_id="turn-decision-1",
                    arguments=arguments,
                    raw_arguments='{"decision":"end_turn","label":"complete","confidence":0.94,"reason":"complete_request","transcript":"ich bin immernoch am programmieren, nur damit du es weisst"}',
                ),
            ),
            response_id="resp_turn_1",
        )


class FakePlayer:
    def __init__(self) -> None:
        self.played: list[bytes] = []
        self.tones: list[tuple[int, int, float, int]] = []
        self.stop_calls = 0

    def play_tone(
        self,
        *,
        frequency_hz: int = 1046,
        duration_ms: int = 180,
        volume: float = 0.8,
        sample_rate: int = 24000,
    ) -> None:
        self.tones.append((frequency_hz, duration_ms, volume, sample_rate))

    def play_pcm16_chunks(self, chunks, *, sample_rate: int, channels: int = 1, should_stop=None) -> None:
        rendered = bytearray()
        for chunk in chunks:
            if should_stop is not None and should_stop():
                break
            rendered.extend(chunk)
        self.played.append(bytes(rendered))
        self.sample_rate = sample_rate
        self.channels = channels

    def play_wav_chunks(self, chunks, *, should_stop=None) -> None:
        rendered = bytearray()
        for chunk in chunks:
            if should_stop is not None and should_stop():
                break
            rendered.extend(chunk)
        self.played.append(bytes(rendered))

    def play_wav_bytes(self, audio_bytes: bytes) -> None:
        self.played.append(audio_bytes)

    def stop_playback(self) -> None:
        self.stop_calls += 1


class FakeAmbientAudioSampler:
    def __init__(self, windows: list[AmbientAudioCaptureWindow]) -> None:
        self.windows = list(windows)
        self.calls = 0

    def sample_window(self, *, duration_ms: int | None = None) -> AmbientAudioCaptureWindow:
        del duration_ms
        self.calls += 1
        if self.windows:
            return self.windows.pop(0)
        return AmbientAudioCaptureWindow(
            sample=AmbientAudioLevelSample(
                duration_ms=420,
                chunk_count=4,
                active_chunk_count=0,
                average_rms=120,
                peak_rms=180,
                active_ratio=0.0,
            ),
            pcm_bytes=b"\x00\x00" * 3200,
            sample_rate=16000,
            channels=1,
        )


class FakeConversationClosureEvaluator:
    def __init__(self, decision: ConversationClosureDecision) -> None:
        self.decision = decision
        self.calls: list[dict[str, object]] = []

    def evaluate(
        self,
        *,
        user_transcript: str,
        assistant_response: str,
        request_source: str,
        proactive_trigger: str | None = None,
        conversation=None,
        turn_steering_cues=(),
    ) -> ConversationClosureDecision:
        self.calls.append(
            {
                "user_transcript": user_transcript,
                "assistant_response": assistant_response,
                "request_source": request_source,
                "proactive_trigger": proactive_trigger,
                "conversation": conversation,
                "turn_steering_cues": tuple(turn_steering_cues),
            }
        )
        return self.decision


class FakePrinter:
    def __init__(self) -> None:
        self.printed: list[str] = []

    def print_text(self, text: str) -> str:
        self.printed.append(text)
        return "request id is Test-2 (1 file(s))"


class FakeCamera:
    def __init__(self) -> None:
        self.capture_calls = 0

    def capture_photo(self, *, output_path=None, filename: str = "camera-capture.png"):
        self.capture_calls += 1
        return SimpleNamespace(
            data=b"\x89PNG\r\n\x1a\ncamera",
            content_type="image/png",
            filename=filename,
            source_device="/dev/video0",
            input_format="bayer_grbg8",
        )


class FakeIdleButtonMonitor:
    def __init__(self) -> None:
        self.entered = False
        self.exited = False
        self.poll_calls = 0

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.exited = True

    def poll(self, timeout=None):
        self.poll_calls += 1
        if timeout:
            time.sleep(min(timeout, 0.001))
        return None


class FakeScheduledButtonMonitor(FakeIdleButtonMonitor):
    def __init__(self, *, event_after_s: float, name: str = "yellow") -> None:
        super().__init__()
        self._event_after_s = event_after_s
        self._name = name
        self._started_at: float | None = None
        self._emitted = False

    def __enter__(self):
        self._started_at = time.monotonic()
        return super().__enter__()

    def poll(self, timeout=None):
        self.poll_calls += 1
        if timeout:
            time.sleep(min(timeout, 0.001))
        if self._emitted or self._started_at is None:
            return None
        if time.monotonic() - self._started_at < self._event_after_s:
            return None
        self._emitted = True
        return ButtonEvent(
            name=self._name,
            line_offset=22 if self._name == "yellow" else 23,
            action=ButtonAction.PRESSED,
            raw_edge="falling",
            timestamp_ns=time.monotonic_ns(),
        )


class FakeBurstButtonMonitor(FakeIdleButtonMonitor):
    def __init__(self, events: list[tuple[float, str]]) -> None:
        super().__init__()
        self._events = list(events)
        self._started_at: float | None = None

    def __enter__(self):
        self._started_at = time.monotonic()
        return super().__enter__()

    def poll(self, timeout=None):
        self.poll_calls += 1
        if timeout:
            time.sleep(min(timeout, 0.001))
        if self._started_at is None or not self._events:
            return None
        delay_s, name = self._events[0]
        if time.monotonic() - self._started_at < delay_s:
            return None
        self._events.pop(0)
        return ButtonEvent(
            name=name,
            line_offset=22 if name == "yellow" else 23,
            action=ButtonAction.PRESSED,
            raw_edge="falling",
            timestamp_ns=time.monotonic_ns(),
        )


class FakeProactiveMonitor:
    def __init__(self) -> None:
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.exited = True


class RealtimeHardwareLoopTests(unittest.TestCase):
    _LANGUAGE_CONTRACT = "All user-facing spoken and written replies for this turn must be in German."

    def make_loop(
        self,
        *,
        config: TwinrConfig | None = None,
        recorder: FakeRecorder | None = None,
        camera: FakeCamera | None = None,
        button_monitor=None,
        print_backend: FakePrintBackend | None = None,
        stt_provider=None,
        agent_provider=None,
        tts_provider=None,
        turn_stt_provider=None,
        turn_tool_agent_provider=None,
        ambient_audio_sampler=None,
        conversation_closure_evaluator=None,
        voice_profile_monitor=None,
        proactive_monitor=None,
    ) -> tuple[TwinrRealtimeHardwareLoop, list[str], FakeRealtimeSession, FakePrintBackend, FakeRecorder, FakePlayer, FakePrinter]:
        temp_dir_handle = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir_handle.cleanup)
        temp_root = Path(temp_dir_handle.name)
        config = config or TwinrConfig()
        original_project_root = Path(config.project_root).resolve()
        sandbox_project_default = config.project_root == "."
        config = replace(
            config,
            project_root=str(temp_root if config.project_root == "." else Path(config.project_root)),
            runtime_state_path=self._sandbox_path(
                config.runtime_state_path,
                temp_root / "runtime-state.json",
                default="/tmp/twinr-runtime-state.json",
                project_root=original_project_root,
                sandbox_project_default=sandbox_project_default,
            ),
            reminder_store_path=self._sandbox_path(
                config.reminder_store_path,
                temp_root / "state" / "reminders.json",
                default="state/reminders.json",
                project_root=original_project_root,
                sandbox_project_default=sandbox_project_default,
            ),
            memory_markdown_path=self._sandbox_path(
                config.memory_markdown_path,
                temp_root / "state" / "MEMORY.md",
                default="state/MEMORY.md",
                project_root=original_project_root,
                sandbox_project_default=sandbox_project_default,
            ),
            automation_store_path=self._sandbox_path(
                config.automation_store_path,
                temp_root / "state" / "automations.json",
                default="state/automations.json",
                project_root=original_project_root,
                sandbox_project_default=sandbox_project_default,
            ),
            adaptive_timing_store_path=self._sandbox_path(
                config.adaptive_timing_store_path,
                temp_root / "state" / "adaptive-timing.json",
                default="state/adaptive_timing.json",
                project_root=original_project_root,
                sandbox_project_default=sandbox_project_default,
            ),
            long_term_memory_path=self._sandbox_path(
                config.long_term_memory_path,
                temp_root / "state" / "chonkydb",
                default="state/chonkydb",
                project_root=original_project_root,
                sandbox_project_default=sandbox_project_default,
            ),
            display_face_cue_path=self._sandbox_path(
                config.display_face_cue_path,
                temp_root / "artifacts" / "stores" / "ops" / "display_face_cue.json",
                default="artifacts/stores/ops/display_face_cue.json",
                project_root=original_project_root,
                sandbox_project_default=sandbox_project_default,
            ),
            display_presentation_path=self._sandbox_path(
                config.display_presentation_path,
                temp_root / "artifacts" / "stores" / "ops" / "display_presentation.json",
                default="artifacts/stores/ops/display_presentation.json",
                project_root=original_project_root,
                sandbox_project_default=sandbox_project_default,
            ),
            display_news_ticker_store_path=self._sandbox_path(
                config.display_news_ticker_store_path,
                temp_root / "artifacts" / "stores" / "ops" / "display_news_ticker.json",
                default="artifacts/stores/ops/display_news_ticker.json",
                project_root=original_project_root,
                sandbox_project_default=sandbox_project_default,
            ),
        )
        lines: list[str] = []
        realtime_session = FakeRealtimeSession()
        resolved_print_backend = print_backend
        if resolved_print_backend is None:
            for provider in (agent_provider, stt_provider, tts_provider):
                if provider is not None:
                    resolved_print_backend = provider
                    break
        if resolved_print_backend is None:
            resolved_print_backend = FakePrintBackend()
        recorder = recorder or FakeRecorder()
        player = FakePlayer()
        printer = FakePrinter()
        loop = TwinrRealtimeHardwareLoop(
            config=config,
            runtime=TwinrRuntime(config=config),
            realtime_session=realtime_session,
            print_backend=resolved_print_backend,
            stt_provider=stt_provider,
            agent_provider=agent_provider,
            tts_provider=tts_provider,
            turn_stt_provider=turn_stt_provider,
            turn_tool_agent_provider=turn_tool_agent_provider,
            button_monitor=button_monitor or SimpleNamespace(__enter__=lambda self: self, __exit__=lambda self, exc_type, exc, tb: None),
            recorder=recorder,
            player=player,
            printer=printer,
            camera=camera or FakeCamera(),
            voice_profile_monitor=voice_profile_monitor,
            ambient_audio_sampler=ambient_audio_sampler,
            conversation_closure_evaluator=conversation_closure_evaluator,
            proactive_monitor=proactive_monitor,
            emit=lines.append,
            sleep=lambda _seconds: None,
            error_reset_seconds=0.0,
        )
        loop._test_temp_dir = temp_dir_handle
        return loop, lines, realtime_session, resolved_print_backend, recorder, player, printer

    @staticmethod
    def _sandbox_path(
        current: str,
        isolated: Path,
        *,
        default: str,
        project_root: Path,
        sandbox_project_default: bool,
    ) -> str:
        path = Path(current)
        default_path = Path(default)
        project_default = project_root / default_path if not default_path.is_absolute() else default_path
        if current == default or not path.is_absolute() or (sandbox_project_default and path == project_default):
            return str(isolated)
        return current

    def _assert_language_contract_only(self, conversation: tuple[tuple[str, str], ...] | None) -> None:
        self.assertIsNotNone(conversation)
        assert conversation is not None
        self.assertEqual(len(conversation), 1)
        self.assertEqual(conversation[0][0], "system")
        self.assertIn(self._LANGUAGE_CONTRACT, conversation[0][1])

    def test_make_loop_registers_tempdir_cleanup(self) -> None:
        class _FakeTempDir:
            def __init__(self, path: str) -> None:
                self.name = path
                self.cleaned = False

            def cleanup(self) -> None:
                self.cleaned = True
                shutil.rmtree(self.name, ignore_errors=True)

        fake_temp_dir = _FakeTempDir(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, fake_temp_dir.name, True)

        with mock.patch("test.test_realtime_runner.tempfile.TemporaryDirectory", return_value=fake_temp_dir):
            self.make_loop()

        cleanup_functions = [cleanup[0] for cleanup in self._cleanups]
        self.assertIn(fake_temp_dir.cleanup, cleanup_functions)

    def test_green_button_runs_realtime_audio_turn(self) -> None:
        loop, lines, realtime_session, _print_backend, recorder, player, _printer = self.make_loop()

        loop.handle_button_press("green")

        self.assertEqual(recorder.pause_values, [1200])
        self.assertEqual(recorder.start_timeouts, [8.0])
        self.assertEqual(recorder.speech_start_chunks, [None])
        self.assertEqual(recorder.ignore_initial_ms, [0])
        self.assertEqual(recorder.pause_grace_values, [450])
        self.assertEqual(realtime_session.calls, [b"PCMINPUT"])
        self.assertEqual(len(realtime_session.conversations), 1)
        self._assert_language_contract_only(realtime_session.conversations[0])
        self.assertTrue(realtime_session.entered)
        self.assertTrue(realtime_session.exited)
        self.assertGreaterEqual(len(player.tones), 1)
        self.assertEqual(player.played, [b"PCM"])
        self.assertEqual(loop.runtime.last_transcript, "")
        self.assertEqual(loop.runtime.last_response, "Guten Tag")
        self.assertIn("transcript=Hallo Twinr", lines)
        self.assertIn("status=listening", lines)
        self.assertIn("status=processing", lines)
        self.assertIn("status=answering", lines)
        self.assertIn("status=waiting", lines)
        self.assertTrue(any(line.startswith("timing_realtime_ms=") for line in lines))

    def test_green_button_can_end_turn_from_streaming_turn_controller(self) -> None:
        config = TwinrConfig(
            turn_controller_enabled=True,
            turn_controller_fast_endpoint_enabled=False,
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
        )
        turn_stt_provider = FakeTurnStreamingSpeechToTextProvider(config)
        turn_tool_agent_provider = FakeTurnToolAgentProvider(config)
        loop, lines, realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
            config=config,
            turn_stt_provider=turn_stt_provider,
            turn_tool_agent_provider=turn_tool_agent_provider,
        )

        loop.handle_button_press("green")

        self.assertEqual(turn_stt_provider.start_calls[0]["sample_rate"], config.openai_realtime_input_sample_rate)
        self.assertEqual(turn_stt_provider.session.sent, [b"PCM-A", b"PCM-B"])
        self.assertTrue(turn_stt_provider.session.closed)
        self.assertEqual(len(turn_tool_agent_provider.start_calls), 1)
        self.assertIn("turn_controller_candidate=speech_final", lines)
        self.assertIn("turn_controller_decision=end_turn", lines)
        self.assertIn("turn_controller_label=complete", lines)
        self.assertIn("turn_controller_selected_label=complete", lines)
        self.assertIn("stt_partial=ich bin immernoch", lines)
        self.assertIn("timing_stt_ms=", "\n".join(lines))
        self.assertEqual(realtime_session.calls, [b"PCMINPUT"])

    def test_turn_guidance_runtime_caps_controller_context_and_traces_gate(self) -> None:
        config = TwinrConfig(
            turn_controller_context_turns=2,
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
        )
        loop, _lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)
        conversation = (
            ("user", "eins"),
            ("assistant", "zwei"),
            ("user", "drei"),
            ("assistant", "vier"),
        )

        with (
            mock.patch.object(loop.runtime, "conversation_context", return_value=conversation),
            mock.patch.object(loop, "_trace_event") as trace_event,
        ):
            selected = loop._turn_controller_conversation()

        self.assertEqual(selected, conversation[-2:])
        self.assertEqual(trace_event.call_count, 1)
        self.assertEqual(trace_event.call_args.args[0], "turn_guidance_controller_context_gate")
        self.assertEqual(trace_event.call_args.kwargs["details"]["available_turns"], 4)
        self.assertEqual(trace_event.call_args.kwargs["details"]["selected_turns"], 2)
        self.assertEqual(trace_event.call_args.kwargs["kpi"]["selected_turns"], 2)

    def test_turn_guidance_runtime_traces_backchannel_context_gate(self) -> None:
        loop, _lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop()
        conversation = (("user", "Wie spät ist es?"), ("assistant", "Es ist kurz nach acht."))

        with (
            mock.patch.object(loop.runtime, "provider_conversation_context", return_value=conversation),
            mock.patch.object(loop, "_trace_event") as trace_event,
        ):
            context = loop.turn_guidance_runtime.context_for_turn_label("backchannel")

        self.assertEqual(context.turn_label, "backchannel")
        self.assertEqual(context.available_context_turns, 2)
        self.assertEqual(context.guidance_message_count, 1)
        self.assertEqual(context.conversation[:-1], conversation)
        self.assertEqual(context.conversation[-1][0], "system")
        self.assertIn("short backchannel", context.conversation[-1][1])
        self.assertEqual(trace_event.call_count, 1)
        self.assertEqual(trace_event.call_args.args[0], "turn_guidance_context_gate")
        self.assertEqual(trace_event.call_args.kwargs["details"]["guidance_message_count"], 1)
        self.assertEqual(trace_event.call_args.kwargs["kpi"]["conversation_turns"], 3)

    def test_green_button_recovers_when_streaming_stt_hears_speech_before_local_threshold(self) -> None:
        config = TwinrConfig(
            turn_controller_enabled=True,
            turn_controller_fast_endpoint_enabled=False,
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
        )
        turn_stt_provider = FakeTurnStreamingSpeechToTextProvider(config)
        turn_tool_agent_provider = FakeTurnToolAgentProvider(config)
        recorder = FakeRecorder(recordings=[RuntimeError("No speech detected before timeout")])
        loop, lines, realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
            config=config,
            recorder=recorder,
            turn_stt_provider=turn_stt_provider,
            turn_tool_agent_provider=turn_tool_agent_provider,
        )

        loop.handle_button_press("green")

        self.assertIn("turn_controller_capture_recovered=true", lines)
        self.assertNotIn("turn_controller_fallback=RuntimeError", lines)
        self.assertNotIn("follow_up_timeout=true", lines)
        self.assertEqual(realtime_session.calls, [b"PCM-APCM-B"])
        self.assertEqual(loop.runtime.last_transcript, "")
        self.assertIn("transcript=Hallo Twinr", lines)

    def test_backchannel_turn_adds_short_reply_guidance(self) -> None:
        config = TwinrConfig(
            turn_controller_enabled=True,
            turn_controller_fast_endpoint_enabled=False,
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
        )

        class BackchannelStreamingSession(FakeTurnStreamingSpeechSession):
            def send_pcm(self, pcm_bytes: bytes) -> None:
                self.sent.append(pcm_bytes)
                if len(self.sent) == 2 and self._on_endpoint is not None:
                    self._on_endpoint(
                        StreamingSpeechEndpointEvent(
                            transcript="ja",
                            event_type="speech_final",
                            speech_final=True,
                        )
                    )

            def snapshot(self) -> StreamingTranscriptionResult:
                return StreamingTranscriptionResult(
                    transcript="ja",
                    request_id="turn-stt-1",
                    saw_interim=True,
                    saw_speech_final=True,
                    saw_utterance_end=False,
                )

            def finalize(self) -> StreamingTranscriptionResult:
                self.finalize_calls += 1
                return self.snapshot()

        class BackchannelStreamingProvider(FakeTurnStreamingSpeechToTextProvider):
            def __init__(self, config: TwinrConfig) -> None:
                super().__init__(config)
                self.session = BackchannelStreamingSession()

            def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
                del audio_bytes, kwargs
                return "ja"

            def transcribe_path(self, path, **kwargs) -> str:
                del path, kwargs
                return "ja"

        class BackchannelTurnToolProvider(FakeTurnToolAgentProvider):
            def start_turn_streaming(self, *args, **kwargs) -> ToolCallingTurnResponse:
                kwargs.pop("timeout_seconds", None)
                kwargs.pop("timeout", None)
                response = super().start_turn_streaming(*args, **kwargs)
                arguments = {
                    "decision": "end_turn",
                    "label": "backchannel",
                    "confidence": 0.88,
                    "reason": "short_answer",
                    "transcript": "ja",
                }
                return ToolCallingTurnResponse(
                    text="",
                    tool_calls=(
                        AgentToolCall(
                            name="submit_turn_decision",
                            call_id="turn-decision-1",
                            arguments=arguments,
                            raw_arguments='{"decision":"end_turn","label":"backchannel","confidence":0.88,"reason":"short_answer","transcript":"ja"}',
                        ),
                    ),
                    response_id=response.response_id,
                )

        turn_stt_provider = BackchannelStreamingProvider(config)
        turn_tool_agent_provider = BackchannelTurnToolProvider(config)
        loop, lines, realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
            config=config,
            turn_stt_provider=turn_stt_provider,
            turn_tool_agent_provider=turn_tool_agent_provider,
        )

        loop.handle_button_press("green")

        self.assertIn("turn_controller_selected_label=backchannel", lines)
        conversation = realtime_session.conversations[0]
        assert conversation is not None
        self.assertTrue(any("short backchannel" in message for role, message in conversation if role == "system"))

    def test_streaming_transcript_verifier_gate_opens_for_empty_short_audio(self) -> None:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
            streaming_transcript_verifier_enabled=True,
            streaming_transcript_verifier_max_capture_ms=6500,
        )
        loop, _lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
            config=config,
        )
        loop.transcript_verifier_provider = FakePrintBackend()

        with mock.patch.object(loop, "_trace_event") as trace_event:
            decision = loop.streaming_transcript_verifier_runtime.verification_gate(
                capture_result=SpeechCaptureResult(
                    pcm_bytes=_voice_sample_pcm_bytes(duration_s=2.1),
                    speech_started_after_ms=10128,
                    resumed_after_pause_count=0,
                ),
                transcript="",
                capture_ms=12027,
                saw_speech_final=True,
                saw_utterance_end=False,
                confidence=None,
            )

        self.assertTrue(decision.should_verify)
        self.assertEqual(decision.reason, "empty_transcript")
        self.assertEqual(trace_event.call_count, 1)
        self.assertEqual(trace_event.call_args.args[0], "streaming_transcript_verifier_gate")
        self.assertEqual(trace_event.call_args.kwargs["details"]["reason"], "empty_transcript")
        self.assertEqual(trace_event.call_args.kwargs["kpi"]["effective_audio_ms"], decision.effective_audio_ms)

    def test_streaming_transcript_verifier_gate_closes_for_long_audio(self) -> None:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
            streaming_transcript_verifier_enabled=True,
            streaming_transcript_verifier_max_capture_ms=6500,
        )
        loop, _lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
            config=config,
        )
        loop.transcript_verifier_provider = FakePrintBackend()

        with mock.patch.object(loop, "_trace_event") as trace_event:
            decision = loop.streaming_transcript_verifier_runtime.verification_gate(
                capture_result=SpeechCaptureResult(
                    pcm_bytes=_voice_sample_pcm_bytes(duration_s=7.0),
                    speech_started_after_ms=1500,
                    resumed_after_pause_count=0,
                ),
                transcript="",
                capture_ms=12027,
                saw_speech_final=True,
                saw_utterance_end=False,
                confidence=None,
            )

        self.assertFalse(decision.should_verify)
        self.assertEqual(decision.reason, "capture_too_long")
        self.assertGreater(decision.effective_audio_ms, 6500)
        self.assertEqual(trace_event.call_count, 1)
        self.assertEqual(trace_event.call_args.args[0], "streaming_transcript_verifier_gate")
        self.assertEqual(trace_event.call_args.kwargs["details"]["reason"], "capture_too_long")
        self.assertEqual(trace_event.call_args.kwargs["kpi"]["effective_audio_ms"], decision.effective_audio_ms)

    def test_streaming_transcript_verifier_recovers_empty_short_audio_after_late_start(self) -> None:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
            streaming_transcript_verifier_enabled=True,
            streaming_transcript_verifier_max_capture_ms=6500,
        )
        loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
            config=config,
        )
        verifier_provider = FakePrintBackend()
        verifier_provider.transcribe_result = "Wie geht es dir heute?"
        loop.transcript_verifier_provider = verifier_provider

        transcript = loop._maybe_verify_streaming_transcript(
            capture_result=SpeechCaptureResult(
                pcm_bytes=_voice_sample_pcm_bytes(duration_s=2.1),
                speech_started_after_ms=10128,
                resumed_after_pause_count=0,
            ),
            transcript="",
            capture_ms=12027,
            saw_speech_final=True,
            saw_utterance_end=False,
            confidence=None,
        )

        self.assertEqual(transcript, "Wie geht es dir heute?")
        self.assertEqual(len(verifier_provider.transcribe_calls), 1)
        self.assertIn("stt_streaming_verified_via_openai=true", lines)

    def test_streaming_transcript_verifier_traces_selected_source_when_replacing_transcript(self) -> None:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
            streaming_transcript_verifier_enabled=True,
            streaming_transcript_verifier_max_capture_ms=6500,
        )
        loop, _lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
            config=config,
        )
        verifier_provider = FakePrintBackend()
        verifier_provider.transcribe_result = "Worüber haben wir heute gesprochen?"
        loop.transcript_verifier_provider = verifier_provider

        with mock.patch.object(loop, "_trace_event") as trace_event:
            transcript = loop._maybe_verify_streaming_transcript(
                capture_result=SpeechCaptureResult(
                    pcm_bytes=_voice_sample_pcm_bytes(duration_s=2.1),
                    speech_started_after_ms=1400,
                    resumed_after_pause_count=0,
                ),
                transcript="wir heute",
                capture_ms=2100,
                saw_speech_final=True,
                saw_utterance_end=False,
                confidence=0.41,
            )

        self.assertEqual(transcript, "Worüber haben wir heute gesprochen?")
        selected_events = [
            call.kwargs["details"]
            for call in trace_event.call_args_list
            if call.args and call.args[0] == "streaming_transcript_verifier_selected"
        ]
        self.assertTrue(selected_events)
        self.assertEqual(selected_events[-1]["selected_source"], "verifier")
        self.assertEqual(selected_events[-1]["reason"], "verifier_more_informative")
        self.assertTrue(selected_events[-1]["verified"]["present"])

    def test_streaming_transcript_verifier_skips_empty_long_audio(self) -> None:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
            streaming_transcript_verifier_enabled=True,
            streaming_transcript_verifier_max_capture_ms=6500,
        )
        loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
            config=config,
        )
        verifier_provider = FakePrintBackend()
        verifier_provider.transcribe_result = "Das darf hier nicht auftauchen."
        loop.transcript_verifier_provider = verifier_provider

        transcript = loop._maybe_verify_streaming_transcript(
            capture_result=SpeechCaptureResult(
                pcm_bytes=_voice_sample_pcm_bytes(duration_s=7.0),
                speech_started_after_ms=1500,
                resumed_after_pause_count=0,
            ),
            transcript="",
            capture_ms=12027,
            saw_speech_final=True,
            saw_utterance_end=False,
            confidence=None,
        )

        self.assertEqual(transcript, "")
        self.assertEqual(verifier_provider.transcribe_calls, [])
        self.assertNotIn("stt_streaming_verified_via_openai=true", lines)

    def test_user_interrupt_stops_answer_and_opens_follow_up_turn(self) -> None:
        config = TwinrConfig(
            turn_controller_interrupt_enabled=True,
            turn_controller_interrupt_window_ms=120,
            turn_controller_interrupt_poll_ms=10,
            turn_controller_interrupt_min_active_ratio=0.1,
            turn_controller_interrupt_min_transcript_chars=4,
            turn_controller_interrupt_consecutive_windows=2,
            conversation_follow_up_timeout_s=3.5,
        )

        class InterruptibleRealtimeSession(FakeRealtimeSession):
            def run_audio_turn(
                self,
                audio_pcm: bytes,
                *,
                conversation=None,
                on_audio_chunk=None,
                on_output_text_delta=None,
            ) -> OpenAIRealtimeTurn:
                self.calls.append(audio_pcm)
                self.conversations.append(conversation)
                if on_output_text_delta is not None:
                    on_output_text_delta("Ich ")
                if on_audio_chunk is not None:
                    on_audio_chunk(b"PCM1")
                    time.sleep(0.03)
                    on_audio_chunk(b"PCM2")
                time.sleep(0.05)
                return OpenAIRealtimeTurn(
                    transcript="Hallo Twinr",
                    response_text="Ich bin noch nicht fertig",
                    response_id="resp_rt_interrupt",
                    end_conversation=False,
                )

        backend = FakePrintBackend()
        backend.transcribe_result = "warte mal"
        speech_window = AmbientAudioCaptureWindow(
            sample=AmbientAudioLevelSample(
                duration_ms=120,
                chunk_count=2,
                active_chunk_count=2,
                average_rms=1800,
                peak_rms=2200,
                active_ratio=1.0,
            ),
            pcm_bytes=_voice_sample_pcm_bytes(duration_s=0.15),
            sample_rate=24000,
            channels=1,
        )
        sampler = FakeAmbientAudioSampler([speech_window, speech_window])
        loop, lines, realtime_session, _print_backend, recorder, _player, _printer = self.make_loop(
            config=config,
            print_backend=backend,
            ambient_audio_sampler=sampler,
        )
        loop.realtime_session = InterruptibleRealtimeSession()

        loop.handle_button_press("green")

        self.assertIn("user_interrupt_detected=true", lines)
        self.assertIn("assistant_interrupted=true", lines)
        self.assertIn("interrupt_transcript=warte mal", lines)
        self.assertEqual(len(loop.realtime_session.calls), 2)
        self.assertEqual(recorder.start_timeouts, [8.0, 3.5])

    def test_green_button_uses_processing_and_answering_feedback(self) -> None:
        loop, _lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop()
        feedback_kinds: list[str] = []

        def fake_start(kind: str):
            feedback_kinds.append(kind)
            return lambda: None

        loop._start_working_feedback_loop = fake_start  # type: ignore[method-assign]

        loop.handle_button_press("green")

        self.assertEqual(feedback_kinds[:2], ["processing", "answering"])

    def test_green_button_falls_back_to_tts_when_realtime_returns_text_only(self) -> None:
        class TextOnlyRealtimeSession(FakeRealtimeSession):
            def run_audio_turn(
                self,
                audio_pcm: bytes,
                *,
                conversation=None,
                on_audio_chunk=None,
                on_output_text_delta=None,
            ) -> OpenAIRealtimeTurn:
                self.calls.append(audio_pcm)
                self.conversations.append(conversation)
                if on_output_text_delta is not None:
                    on_output_text_delta("Guten ")
                    on_output_text_delta("Tag")
                if self.turns:
                    return self.turns.pop(0)
                return OpenAIRealtimeTurn(
                    transcript="Hallo Twinr",
                    response_text="Guten Tag",
                    response_id="resp_rt_text_only",
                    end_conversation=False,
                )

        backend = FakePrintBackend()
        backend.synthesize_sleep_s = 0.12
        loop, lines, realtime_session, _print_backend, _recorder, player, _printer = self.make_loop(
            print_backend=backend,
        )
        loop.realtime_session = TextOnlyRealtimeSession()

        loop.handle_button_press("green")

        self.assertEqual(realtime_session.calls, [])
        self.assertEqual(loop.realtime_session.calls, [b"PCMINPUT"])
        self.assertEqual(backend.synthesize_calls, ["Guten Tag"])
        self.assertEqual(player.played, [b"PCM"])
        self.assertIn("realtime_audio_fallback=true", lines)
        self.assertTrue(any(line.startswith("timing_tts_fallback_ms=") for line in lines))

    def test_yellow_button_accepts_split_support_providers(self) -> None:
        backend = FakePrintBackend()
        loop, _lines, _realtime_session, _print_backend, _recorder, _player, printer = self.make_loop(
            print_backend=None,
            stt_provider=backend,
            agent_provider=backend,
            tts_provider=backend,
        )
        loop.runtime.last_response = "Guten Tag"

        loop.handle_button_press("yellow")
        loop.wait_for_print_lane_idle()

        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(printer.printed, ["GUTEN TAG"])

    def test_wakeword_with_remaining_text_runs_direct_text_turn(self) -> None:
        loop, lines, realtime_session, _print_backend, recorder, player, _printer = self.make_loop()

        handled = loop.handle_wakeword_match(
            WakewordMatch(
                detected=True,
                transcript="Hey Twinr wie spaet ist es",
                matched_phrase="hey twinr",
                remaining_text="wie spaet ist es",
                normalized_transcript="hey twinr wie spaet ist es",
            )
        )

        self.assertTrue(handled)
        self.assertEqual(realtime_session.text_calls, ["wie spaet ist es"])
        self.assertEqual(realtime_session.calls, [])
        self.assertEqual(recorder.pause_values, [])
        self.assertGreaterEqual(len(player.tones), 1)
        self.assertIn("wakeword_mode=direct_text", lines)
        self.assertEqual(loop.runtime.last_transcript, "")
        self.assertIn("transcript=Hallo Twinr", lines)

    def test_wakeword_without_remaining_text_opens_listening_window(self) -> None:
        loop, lines, realtime_session, _print_backend, recorder, player, _printer = self.make_loop()

        handled = loop.handle_wakeword_match(
            WakewordMatch(
                detected=True,
                transcript="Hey Twinr",
                matched_phrase="hey twinr",
                remaining_text="",
                normalized_transcript="hey twinr",
            )
        )

        self.assertTrue(handled)
        self.assertEqual(realtime_session.calls, [b"PCMINPUT"])
        self.assertEqual(realtime_session.text_calls, [])
        self.assertEqual(recorder.pause_values, [1200])
        self.assertEqual(recorder.start_timeouts, [loop.config.conversation_follow_up_timeout_s])
        self.assertEqual(recorder.speech_start_chunks, [loop.config.audio_follow_up_speech_start_chunks])
        self.assertEqual(recorder.ignore_initial_ms, [loop.config.audio_follow_up_ignore_ms])
        self.assertEqual(recorder.pause_grace_values, [loop.config.adaptive_timing_pause_grace_ms])
        self.assertGreaterEqual(len(player.tones), 1)
        self.assertEqual(player.played, [b"WAVPCM", b"PCM"])
        self.assertIn("wakeword_mode=listen", lines)
        self.assertIn("wakeword_ack=Ja?", lines)

    def test_wakeword_ack_uses_cached_audio_when_prefetched(self) -> None:
        loop, lines, _realtime_session, print_backend, _recorder, player, _printer = self.make_loop(
            config=TwinrConfig(wakeword_enabled=True)
        )

        loop._prime_wakeword_ack_cache()
        loop._acknowledge_wakeword()

        self.assertEqual(print_backend.synthesize_calls, ["Ja?"])
        self.assertEqual(player.played, [b"WAVPCM"])
        self.assertIn("wakeword_ack=Ja?", lines)
        self.assertIn("wakeword_ack_cached=true", lines)

    def test_wakeword_ack_normalizes_quiet_valid_wav(self) -> None:
        loop, _lines, _realtime_session, print_backend, _recorder, player, _printer = self.make_loop(
            config=TwinrConfig(wakeword_enabled=True)
        )
        quiet_wav = _pcm_to_test_wav_bytes((b"\x10\x00" + b"\xf0\xff") * 800)
        print_backend.synthesize_result = quiet_wav

        loop._acknowledge_wakeword()

        self.assertEqual(player.played, [normalize_wav_playback_level(quiet_wav)])

    def test_yellow_button_uses_print_backend(self) -> None:
        loop, lines, _realtime_session, print_backend, _recorder, _player, printer = self.make_loop()
        loop.runtime.last_response = "Guten Tag"

        loop.handle_button_press("yellow")
        loop.wait_for_print_lane_idle()

        self.assertEqual(len(print_backend.calls), 1)
        self.assertEqual(print_backend.calls[0][0], ())
        self.assertEqual(print_backend.calls[0][1:], (None, "Guten Tag", "button"))
        self.assertEqual(printer.printed, ["GUTEN TAG"])
        self.assertEqual(loop.runtime.status, TwinrStatus.WAITING)
        self.assertIn("print_lane=queued", lines)
        self.assertIn("print_lane=completed", lines)

    def test_yellow_button_uses_local_conversation_context_without_provider_lookup(self) -> None:
        loop, lines, _realtime_session, print_backend, _recorder, _player, printer = self.make_loop()
        loop.runtime.last_response = "Guten Tag"
        local_conversation = (
            ("user", "Kannst du das drucken?"),
            ("assistant", "Guten Tag"),
        )
        loop.runtime.conversation_context = mock.Mock(return_value=local_conversation)  # type: ignore[method-assign]
        loop.runtime.provider_conversation_context = mock.Mock(  # type: ignore[method-assign]
            side_effect=AssertionError("yellow print must not build provider context")
        )

        loop.handle_button_press("yellow")
        self.assertTrue(loop.wait_for_print_lane_idle(timeout_s=1.0))

        loop.runtime.conversation_context.assert_called_once_with()
        loop.runtime.provider_conversation_context.assert_not_called()
        self.assertEqual(len(print_backend.calls), 1)
        self.assertEqual(print_backend.calls[0][0], local_conversation)
        self.assertEqual(print_backend.calls[0][1:], (None, "Guten Tag", "button"))
        self.assertEqual(printer.printed, ["GUTEN TAG"])
        self.assertIn("print_lane=queued", lines)
        self.assertIn("print_lane=completed", lines)

    def test_yellow_button_uses_printing_feedback(self) -> None:
        loop, _lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop()
        loop.runtime.last_response = "Guten Tag"
        feedback_kinds: list[str] = []

        def fake_start(kind: str):
            feedback_kinds.append(kind)
            return lambda: None

        loop._start_working_feedback_loop = fake_start  # type: ignore[method-assign]

        loop.handle_button_press("yellow")
        loop.wait_for_print_lane_idle()

        self.assertEqual(feedback_kinds, ["printing"])

    def test_yellow_button_print_lane_returns_before_slow_print_finishes(self) -> None:
        from threading import Event, Thread

        class SlowPrintBackend(FakePrintBackend):
            def __init__(self) -> None:
                super().__init__()
                self.entered = Event()
                self.release = Event()

            def compose_print_job_with_metadata(self, *, conversation=None, focus_hint=None, direct_text=None, request_source="button"):
                self.calls.append((conversation, focus_hint, direct_text, request_source))
                self.entered.set()
                self.release.wait(1.0)
                return OpenAITextResponse(text="GUTEN TAG")

        backend = SlowPrintBackend()
        loop, _lines, _realtime_session, _print_backend, _recorder, _player, printer = self.make_loop(
            print_backend=backend,
            stt_provider=backend,
            agent_provider=backend,
            tts_provider=backend,
        )
        loop.runtime.last_response = "Guten Tag"
        completed = Event()

        def press_button() -> None:
            loop.handle_button_press("yellow")
            completed.set()

        caller = Thread(target=press_button, daemon=True)
        caller.start()

        self.assertTrue(backend.entered.wait(0.3))
        self.assertTrue(completed.wait(0.3))
        self.assertEqual(printer.printed, [])

        backend.release.set()
        self.assertTrue(loop.wait_for_print_lane_idle(timeout_s=1.0))
        caller.join(timeout=1.0)
        self.assertEqual(printer.printed, ["GUTEN TAG"])

    def test_green_button_can_run_while_print_lane_is_busy(self) -> None:
        from threading import Event, Thread

        class SlowPrintBackend(FakePrintBackend):
            def __init__(self) -> None:
                super().__init__()
                self.entered = Event()
                self.release = Event()

            def compose_print_job_with_metadata(self, *, conversation=None, focus_hint=None, direct_text=None, request_source="button"):
                self.calls.append((conversation, focus_hint, direct_text, request_source))
                self.entered.set()
                self.release.wait(1.0)
                return OpenAITextResponse(text="GUTEN TAG")

        backend = SlowPrintBackend()
        loop, _lines, realtime_session, _print_backend, _recorder, _player, printer = self.make_loop(
            print_backend=backend,
        )
        loop.runtime.last_response = "Guten Tag"

        caller = Thread(target=lambda: loop.handle_button_press("yellow"), daemon=True)
        caller.start()
        self.assertTrue(backend.entered.wait(0.3))
        self.assertTrue(loop.print_lane.is_busy())
        self.assertEqual(loop.runtime.status, TwinrStatus.WAITING)

        loop.handle_button_press("green")

        self.assertEqual(realtime_session.calls, [b"PCMINPUT"])
        self.assertEqual(loop.runtime.status, TwinrStatus.WAITING)
        backend.release.set()
        self.assertTrue(loop.wait_for_print_lane_idle(timeout_s=1.0))
        caller.join(timeout=1.0)
        self.assertEqual(printer.printed, ["GUTEN TAG"])

    def test_print_lane_failure_is_nonfatal(self) -> None:
        class FailingPrintBackend(FakePrintBackend):
            def compose_print_job_with_metadata(self, *, conversation=None, focus_hint=None, direct_text=None, request_source="button"):
                self.calls.append((conversation, focus_hint, direct_text, request_source))
                raise RuntimeError("printer compose failed")

        backend = FailingPrintBackend()
        loop, lines, realtime_session, _print_backend, _recorder, _player, printer = self.make_loop(
            print_backend=backend,
        )
        loop.runtime.last_response = "Guten Tag"

        loop.handle_button_press("yellow")
        self.assertTrue(loop.wait_for_print_lane_idle(timeout_s=1.0))

        self.assertEqual(printer.printed, [])
        self.assertEqual(loop.runtime.status, TwinrStatus.WAITING)
        self.assertIn("print_lane=failed", lines)
        self.assertTrue(any(line.endswith("printer compose failed") for line in lines if line.startswith("print_error=")))
        self.assertNotIn("status=error", lines)

        loop.handle_button_press("green")

        self.assertEqual(realtime_session.calls, [b"PCMINPUT"])
        self.assertEqual(loop.runtime.status, TwinrStatus.WAITING)

    def test_follow_up_turn_beeps_again_and_stops_on_timeout(self) -> None:
        config = TwinrConfig(
            conversation_follow_up_enabled=True,
            conversation_follow_up_timeout_s=3.5,
            audio_follow_up_speech_start_chunks=5,
            audio_follow_up_ignore_ms=420,
        )
        recorder = FakeRecorder(
            recordings=[
                b"PCMINPUT",
                RuntimeError("No speech detected before timeout"),
            ]
        )
        loop, lines, realtime_session, _print_backend, recorder, player, _printer = self.make_loop(
            config=config,
            recorder=recorder,
        )

        loop.handle_button_press("green")

        self.assertEqual(recorder.pause_values, [1200, 1200])
        self.assertEqual(recorder.start_timeouts, [8.0, 3.5])
        self.assertEqual(recorder.speech_start_chunks, [None, 5])
        self.assertEqual(recorder.ignore_initial_ms, [0, 420])
        self.assertEqual(recorder.pause_grace_values, [450, 450])
        self.assertEqual(realtime_session.calls, [b"PCMINPUT"])
        self._assert_language_contract_only(realtime_session.conversations[0])
        self.assertGreaterEqual(len(player.tones), 2)
        self.assertIn("follow_up_timeout=true", lines)
        self.assertEqual(loop.runtime.status.value, "waiting")

    def test_button_listening_timeout_adapts_next_start_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                adaptive_timing_store_path=str(Path(temp_dir) / "adaptive-timing.json"),
            )
            recorder = FakeRecorder(
                recordings=[
                    RuntimeError("No speech detected before timeout"),
                    b"PCMINPUT",
                ]
            )
            loop, lines, realtime_session, _print_backend, recorder, player, _printer = self.make_loop(
                config=config,
                recorder=recorder,
            )

            loop.handle_button_press("green")
            loop.handle_button_press("green")

        self.assertEqual(recorder.start_timeouts, [8.0, 8.75])
        self.assertEqual(realtime_session.calls, [b"PCMINPUT"])
        self.assertGreaterEqual(len(player.tones), 2)
        self.assertIn("listen_timeout=true", lines)

    def test_button_listening_timeout_emits_capture_diagnostics(self) -> None:
        diagnostics = ListenTimeoutCaptureDiagnostics(
            device="default",
            sample_rate=16000,
            channels=1,
            chunk_ms=100,
            speech_threshold=700,
            chunk_count=12,
            active_chunk_count=1,
            average_rms=188,
            peak_rms=412,
            listened_ms=8040,
        )
        recorder = FakeRecorder(
            recordings=[
                SpeechStartTimeoutError(
                    "No speech detected before timeout",
                    diagnostics=diagnostics,
                )
            ]
        )
        loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
            recorder=recorder,
        )

        loop.handle_button_press("green")

        self.assertIn("listen_timeout=true", lines)
        self.assertIn("listen_timeout_capture_device=default", lines)
        self.assertIn("listen_timeout_speech_threshold=700", lines)
        self.assertIn("listen_timeout_chunk_count=12", lines)
        self.assertIn("listen_timeout_peak_rms=412", lines)
        self.assertIn("listen_timeout_active_ratio=0.08", lines)

    def test_resumed_pause_adapts_next_pause_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                adaptive_timing_store_path=str(Path(temp_dir) / "adaptive-timing.json"),
            )
            recorder = FakeRecorder(
                recordings=[
                    SimpleNamespace(
                        pcm_bytes=b"FIRST",
                        speech_started_after_ms=2100,
                        resumed_after_pause_count=1,
                    ),
                    b"SECOND",
                ]
            )
            loop, _lines, realtime_session, _print_backend, recorder, _player, _printer = self.make_loop(
                config=config,
                recorder=recorder,
            )

            loop.handle_button_press("green")
            loop.handle_button_press("green")

        self.assertEqual(recorder.pause_values, [1200, 1230])
        self.assertEqual(recorder.pause_grace_values, [450, 470])
        self.assertEqual(realtime_session.calls, [b"FIRST", b"SECOND"])

    def test_follow_up_turn_receives_previous_conversation_history(self) -> None:
        config = TwinrConfig(
            conversation_follow_up_enabled=True,
            conversation_follow_up_timeout_s=3.5,
            audio_follow_up_speech_start_chunks=5,
            audio_follow_up_ignore_ms=420,
        )
        recorder = FakeRecorder(
            recordings=[
                b"FIRST",
                b"SECOND",
                RuntimeError("No speech detected before timeout"),
            ]
        )
        loop, _lines, realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
            config=config,
            recorder=recorder,
        )
        realtime_session.turns = [
            OpenAIRealtimeTurn(
                transcript="Erste Frage",
                response_text="Erste Antwort",
                response_id="resp_one",
                end_conversation=False,
            ),
            OpenAIRealtimeTurn(
                transcript="Zweite Frage",
                response_text="Zweite Antwort",
                response_id="resp_two",
                end_conversation=False,
            ),
        ]

        loop.handle_button_press("green")

        self._assert_language_contract_only(realtime_session.conversations[0])
        self.assertIsNotNone(realtime_session.conversations[1])
        self.assertEqual(realtime_session.conversations[1][0][0], "system")
        self.assertIn(self._LANGUAGE_CONTRACT, realtime_session.conversations[1][0][1])
        self.assertEqual(realtime_session.conversations[1][1][0], "system")
        self.assertIn("Twinr memory summary", realtime_session.conversations[1][1][1])
        self.assertEqual(
            realtime_session.conversations[1][2:],
            (
                ("user", "Erste Frage"),
                ("assistant", "Erste Antwort"),
            ),
        )

    def test_end_conversation_tool_stops_follow_up_loop(self) -> None:
        config = TwinrConfig(
            conversation_follow_up_enabled=True,
            conversation_follow_up_timeout_s=3.5,
        )
        loop, lines, realtime_session, _print_backend, recorder, player, _printer = self.make_loop(
            config=config,
            recorder=FakeRecorder(recordings=[b"PCMINPUT"]),
        )
        realtime_session.turns = [
            OpenAIRealtimeTurn(
                transcript="Danke, das war's",
                response_text="Gern. Bis spaeter.",
                response_id="resp_end",
                end_conversation=True,
            )
        ]

        loop.handle_button_press("green")

        self.assertEqual(recorder.pause_values, [1200])
        self.assertEqual(recorder.pause_grace_values, [450])
        self.assertGreaterEqual(len(player.tones), 1)
        self.assertIn("conversation_ended=true", lines)
        self.assertEqual(loop.runtime.status.value, "waiting")

    def test_proactive_turn_does_not_chain_second_follow_up_by_default(self) -> None:
        config = TwinrConfig(
            conversation_follow_up_enabled=True,
            conversation_follow_up_after_proactive_enabled=False,
            conversation_follow_up_timeout_s=3.5,
        )
        loop, lines, realtime_session, _print_backend, recorder, player, _printer = self.make_loop(
            config=config,
            recorder=FakeRecorder(recordings=[b"PROACTIVE"]),
        )
        realtime_session.turns = [
            OpenAIRealtimeTurn(
                transcript="Alles ist okay, bis bald.",
                response_text="Gut, danke fuer die Rueckmeldung. Bis spaeter.",
                response_id="resp_proactive",
                end_conversation=False,
            )
        ]

        loop._run_proactive_follow_up(
            SocialTriggerDecision(
                trigger_id="slumped_quiet",
                prompt="Ist alles in Ordnung?",
                reason="concern check",
                observed_at=0.0,
                priority=SocialTriggerPriority.SLUMPED_QUIET,
            )
        )

        self.assertEqual(len(recorder.pause_values), 1)
        self.assertEqual(realtime_session.calls, [b"PROACTIVE"])
        self.assertNotIn("follow_up_timeout=true", lines)
        self.assertEqual(loop.runtime.status.value, "waiting")

    def test_explicit_goodbye_vetoes_follow_up_listening(self) -> None:
        config = TwinrConfig(
            conversation_follow_up_enabled=True,
            conversation_closure_guard_enabled=True,
            conversation_closure_min_confidence=0.6,
            conversation_follow_up_timeout_s=3.5,
        )
        closure_evaluator = FakeConversationClosureEvaluator(
            ConversationClosureDecision(
                close_now=True,
                confidence=0.92,
                reason="explicit_goodbye",
            )
        )
        loop, lines, realtime_session, _print_backend, recorder, _player, _printer = self.make_loop(
            config=config,
            recorder=FakeRecorder(recordings=[b"TURN1", b"TURN2"]),
            conversation_closure_evaluator=closure_evaluator,
        )
        realtime_session.turns = [
            OpenAIRealtimeTurn(
                transcript="Danke, bis bald.",
                response_text="Gern. Bis bald.",
                response_id="resp_goodbye",
                end_conversation=False,
            ),
        ]

        loop.handle_button_press("green")

        self.assertEqual(len(recorder.pause_values), 1)
        self.assertEqual(realtime_session.calls, [b"TURN1"])
        self.assertIn("conversation_closure_close_now=true", lines)
        self.assertIn("conversation_follow_up_vetoed=closure", lines)
        self.assertEqual(closure_evaluator.calls[0]["request_source"], "button")

    def test_cooling_topic_steering_releases_follow_up_after_answer(self) -> None:
        config = TwinrConfig(
            conversation_follow_up_enabled=True,
            conversation_closure_guard_enabled=True,
            conversation_closure_min_confidence=0.6,
            conversation_follow_up_timeout_s=3.5,
        )
        closure_evaluator = FakeConversationClosureEvaluator(
            ConversationClosureDecision(
                close_now=False,
                confidence=0.24,
                reason="still_engaged",
                matched_topics=("local politics",),
            )
        )
        loop, lines, realtime_session, _print_backend, recorder, _player, _printer = self.make_loop(
            config=config,
            recorder=FakeRecorder(recordings=[b"TURN1", b"TURN2"]),
            conversation_closure_evaluator=closure_evaluator,
        )
        loop.follow_up_steering_runtime.load_turn_steering_cues = lambda: (  # type: ignore[method-assign]
            ConversationTurnSteeringCue(
                title="local politics",
                salience=0.74,
                attention_state="cooling",
                open_offer="do_not_steer",
                user_pull="answer_briefly_then_release",
                observe_mode="keep_observing_without_steering",
            ),
        )
        realtime_session.turns = [
            OpenAIRealtimeTurn(
                transcript="Und wie sieht es lokalpolitisch gerade aus?",
                response_text="Kurz gesagt gibt es da gerade Bewegung, aber ich lasse es für jetzt knapp.",
                response_id="resp_local_politics",
                end_conversation=False,
            ),
        ]

        loop.handle_button_press("green")

        self.assertEqual(len(recorder.pause_values), 1)
        self.assertEqual(realtime_session.calls, [b"TURN1"])
        self.assertIn("conversation_follow_up_steering=release_after_answer", lines)
        self.assertIn("conversation_follow_up_vetoed=closure", lines)
        self.assertEqual(
            tuple(cue.title for cue in closure_evaluator.calls[0]["turn_steering_cues"]),
            ("local politics",),
        )

    def test_green_button_updates_runtime_voice_assessment(self) -> None:
        class FakeVoiceProfileMonitor:
            def assess_pcm16(self, audio_pcm: bytes, *, sample_rate: int, channels: int) -> VoiceAssessment:
                self.audio_pcm = audio_pcm
                self.sample_rate = sample_rate
                self.channels = channels
                checked_at = _fresh_checked_at()
                return VoiceAssessment(
                    status="uncertain",
                    label="Uncertain",
                    detail="Partial match.",
                    confidence=0.63,
                    checked_at=checked_at,
                )

        monitor = FakeVoiceProfileMonitor()
        loop, lines, realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
            voice_profile_monitor=monitor,
        )

        loop.handle_button_press("green")

        self.assertEqual(monitor.audio_pcm, b"PCMINPUT")
        self.assertEqual(monitor.sample_rate, loop.config.openai_realtime_input_sample_rate)
        self.assertEqual(monitor.channels, loop.config.audio_channels)
        self.assertEqual(loop.runtime.user_voice_status, "uncertain")
        self.assertEqual(loop.runtime.user_voice_confidence, 0.63)
        self.assertIsNotNone(loop.runtime.user_voice_checked_at)
        assert loop.runtime.user_voice_checked_at is not None
        self.assertTrue(loop.runtime.user_voice_checked_at.endswith("Z"))
        self.assertIsNotNone(realtime_session.conversations[0])
        self.assertTrue(
            any(
                role == "system" and "Speaker signal: partial match" in content
                for role, content in realtime_session.conversations[0]
            )
        )
        self.assertIn("voice_profile_status=uncertain", lines)
        self.assertIn("voice_profile_confidence=0.63", lines)

    def test_print_tool_call_prints_without_formatter(self) -> None:
        loop, lines, _realtime_session, print_backend, _recorder, _player, printer = self.make_loop()
        loop.runtime.press_green_button()
        loop.runtime.submit_transcript("Bitte drucke das")

        result = loop._handle_print_tool_call({"text": "Wichtige Info"})

        self.assertEqual(len(print_backend.calls), 1)
        self._assert_language_contract_only(print_backend.calls[0][0])
        self.assertEqual(print_backend.calls[0][1:], (None, "Wichtige Info", "tool"))
        self.assertEqual(printer.printed, ["GUTEN TAG"])
        self.assertEqual(result["status"], "printed")
        self.assertEqual(result["text"], "GUTEN TAG")
        self.assertIn("status=printing", lines)
        self.assertIn("print_tool_call=true", lines)

    def test_self_coding_tool_calls_create_and_advance_dialogue(self) -> None:
        loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop()

        proposed = loop._handle_propose_skill_learning_tool_call(
            {
                "name": "Announce Family Updates",
                "action": "Read new family updates aloud",
                "request_summary": "Read out new family updates.",
                "capabilities": ["speaker", "safety", "rules"],
                "trigger_mode": "push",
                "trigger_conditions": ["new_message"],
                "scope": {"channel": "messages"},
                "constraints": ["not_during_night_mode"],
            }
        )

        self.assertEqual(proposed["status"], "questioning")
        self.assertEqual(proposed["outcome"], "green")
        self.assertEqual(proposed["dialogue_status"], "questioning")
        self.assertIn("session_id", proposed)
        self.assertEqual(
            proposed["prompt"],
            "Should I do that automatically, or only when you ask me?",
        )

        when_answer = loop._handle_answer_skill_question_tool_call(
            {
                "session_id": proposed["session_id"],
                "trigger_conditions": ["user_visible"],
            }
        )
        what_answer = loop._handle_answer_skill_question_tool_call(
            {
                "session_id": proposed["session_id"],
                "scope": {"contacts": ["family"]},
            }
        )
        how_answer = loop._handle_answer_skill_question_tool_call(
            {
                "session_id": proposed["session_id"],
                "action": "Ask first, then read the update aloud",
                "constraints": ["ask_first"],
            }
        )
        final = loop._handle_answer_skill_question_tool_call(
            {
                "session_id": proposed["session_id"],
                "confirmed": True,
            }
        )

        self.assertEqual(when_answer["status"], "questioning")
        self.assertEqual(what_answer["status"], "questioning")
        self.assertEqual(how_answer["status"], "confirming")
        self.assertEqual(final["status"], "ready_for_compile")
        self.assertEqual(final["compile_job_status"], "queued")
        self.assertIn("compile_job_id", final)
        self.assertEqual(final["skill_spec"]["scope"]["contacts"], ["family"])
        self.assertIn("ask_first", final["skill_spec"]["constraints"])
        self.assertIn("user_visible", final["skill_spec"]["trigger"]["conditions"])
        self.assertEqual(loop._self_coding_store.load_job(final["compile_job_id"]).status.value, "queued")
        self.assertIn("self_coding_tool_call=true", lines)

    def test_self_coding_confirm_activation_tool_call_enables_soft_launch(self) -> None:
        from twinr.agent.self_coding.codex_driver import CodexCompileArtifact, CodexCompileEvent, CodexCompileProgress, CodexCompileResult
        from twinr.agent.self_coding.status import ArtifactKind
        from twinr.agent.self_coding.worker import SelfCodingCompileWorker

        class FakeCompileDriver:
            def run_compile(self, request, *, event_sink=None):
                if event_sink is not None:
                    event_sink(
                        CodexCompileEvent(kind="turn_started"),
                        CodexCompileProgress(driver_name="FakeCompileDriver", event_count=1, last_event_kind="turn_started"),
                    )
                return CodexCompileResult(
                    status="ok",
                    summary="Compiled",
                    artifacts=(
                        CodexCompileArtifact(
                            kind=ArtifactKind.AUTOMATION_MANIFEST,
                            artifact_name="automation_manifest.json",
                            media_type="application/json",
                            content='{"automation":{"name":"Announce Family Updates","trigger":{"kind":"if_then","event_name":"new_message","all_conditions":[],"any_conditions":[],"cooldown_seconds":45},"actions":[{"kind":"say","text":"Family update arrived."}]}}',
                            summary="Manifest",
                        ),
                    ),
                )

        loop, _lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop()
        loop._self_coding_compile_worker = SelfCodingCompileWorker(
            store=loop._self_coding_store,
            driver=FakeCompileDriver(),
        )

        proposed = loop._handle_propose_skill_learning_tool_call(
            {
                "name": "Announce Family Updates",
                "action": "Read new family updates aloud",
                "request_summary": "Read out new family updates.",
                "capabilities": ["speaker", "safety", "rules"],
                "trigger_mode": "push",
                "trigger_conditions": ["new_message"],
            }
        )
        loop._handle_answer_skill_question_tool_call({"session_id": proposed["session_id"], "trigger_conditions": ["user_visible"]})
        loop._handle_answer_skill_question_tool_call({"session_id": proposed["session_id"], "scope": {"contacts": ["family"]}})
        loop._handle_answer_skill_question_tool_call({"session_id": proposed["session_id"], "constraints": ["ask_first"]})
        final = loop._handle_answer_skill_question_tool_call({"session_id": proposed["session_id"], "confirmed": True})

        loop._self_coding_compile_worker.run_job(final["compile_job_id"])
        activated = loop._handle_confirm_skill_activation_tool_call(
            {
                "job_id": final["compile_job_id"],
                "confirmed": True,
            }
        )

        self.assertEqual(activated["status"], "active")
        self.assertEqual(activated["skill_id"], "announce_family_updates")
        self.assertEqual(activated["version"], 1)
        self.assertTrue(activated["automation_enabled"])

    def test_remember_memory_tool_call_writes_memory_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)

            result = loop._handle_remember_memory_tool_call(
                {
                    "kind": "appointment",
                    "summary": "Arzttermin am Montag um 14 Uhr.",
                    "details": "Bei Dr. Meyer in Hamburg.",
                    "confirmed": True,
                }
            )

            memory_text = Path(config.memory_markdown_path).read_text(encoding="utf-8")

        self.assertEqual(result["status"], "saved")
        self.assertIn("Arzttermin am Montag um 14 Uhr.", memory_text)
        self.assertEqual(loop.runtime.memory.ledger[-1].kind, "fact")
        self.assertIn("memory_tool_call=true", lines)

    def test_schedule_reminder_tool_call_writes_reminder_store(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reminder_path = Path(temp_dir) / "state" / "reminders.json"
            future_due_at = now_in_timezone("Europe/Berlin").replace(second=0, microsecond=0) + timedelta(minutes=10)
            config = TwinrConfig(
                project_root=temp_dir,
                reminder_store_path=str(reminder_path),
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)

            result = loop._handle_schedule_reminder_tool_call(
                {
                    "due_at": future_due_at.isoformat(),
                    "summary": "Arzttermin",
                    "details": "Bei Dr. Meyer",
                    "kind": "appointment",
                }
            )

            reminder_text = reminder_path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "scheduled")
        self.assertIn("Arzttermin", reminder_text)
        self.assertIn("appointment", reminder_text)
        self.assertEqual(loop.runtime.memory.ledger[-1].kind, "reminder")
        self.assertIn("reminder_tool_call=true", lines)

    def test_list_create_update_and_delete_time_automation_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                automation_store_path=str(Path(temp_dir) / "state" / "automations.json"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)

            created = loop._handle_create_time_automation_tool_call(
                {
                    "name": "Daily weather",
                    "schedule": "daily",
                    "time_of_day": "08:00",
                    "delivery": "spoken",
                    "content_mode": "llm_prompt",
                    "content": "Give the morning weather report.",
                    "allow_web_search": True,
                    "confirmed": True,
                }
            )
            listed = loop._handle_list_automations_tool_call({})
            updated = loop._handle_update_time_automation_tool_call(
                {
                    "automation_ref": created["automation"]["automation_id"],
                    "time_of_day": "09:15",
                    "delivery": "printed",
                    "content": "Print the morning weather report.",
                    "confirmed": True,
                }
            )
            deleted = loop._handle_delete_automation_tool_call(
                {
                    "automation_ref": created["automation"]["automation_id"],
                    "confirmed": True,
                }
            )

        self.assertEqual(created["status"], "created")
        self.assertEqual(created["automation"]["schedule"], "daily")
        self.assertEqual(created["automation"]["delivery"], "spoken")
        self.assertEqual(listed["count"], 1)
        self.assertEqual(updated["status"], "updated")
        self.assertEqual(updated["automation"]["time_of_day"], "09:15")
        self.assertEqual(updated["automation"]["delivery"], "printed")
        self.assertEqual(deleted["status"], "deleted")
        self.assertIn("automation_tool_call=true", lines)

    def test_list_create_update_and_delete_sensor_automation_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                automation_store_path=str(Path(temp_dir) / "state" / "automations.json"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)

            created = loop._handle_create_sensor_automation_tool_call(
                {
                    "name": "Visitor hello",
                    "trigger_kind": "camera_person_visible",
                    "delivery": "spoken",
                    "content_mode": "llm_prompt",
                    "content": "Say hello to the visitor.",
                    "allow_web_search": False,
                    "cooldown_seconds": 90,
                    "confirmed": True,
                }
            )
            listed = loop._handle_list_automations_tool_call({})
            updated = loop._handle_update_sensor_automation_tool_call(
                {
                    "automation_ref": created["automation"]["automation_id"],
                    "trigger_kind": "vad_quiet",
                    "hold_seconds": 45,
                    "delivery": "printed",
                    "content": "Print a quiet-room note.",
                    "confirmed": True,
                }
            )
            deleted = loop._handle_delete_automation_tool_call(
                {
                    "automation_ref": created["automation"]["automation_id"],
                    "confirmed": True,
                }
            )

        self.assertEqual(created["status"], "created")
        self.assertEqual(created["automation"]["trigger_kind"], "if_then")
        self.assertEqual(created["automation"]["sensor_trigger_kind"], "camera_person_visible")
        self.assertEqual(listed["count"], 1)
        self.assertEqual(updated["status"], "updated")
        self.assertEqual(updated["automation"]["sensor_trigger_kind"], "vad_quiet")
        self.assertEqual(updated["automation"]["sensor_hold_seconds"], 45.0)
        self.assertEqual(updated["automation"]["delivery"], "printed")
        self.assertEqual(deleted["status"], "deleted")
        self.assertIn("automation_tool_call=true", lines)

    def test_idle_loop_executes_due_spoken_automation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                automation_store_path=str(Path(temp_dir) / "state" / "automations.json"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                automation_poll_interval_s=0.0,
            )
            loop, lines, _realtime_session, print_backend, _recorder, player, _printer = self.make_loop(config=config)
            entry = loop.runtime.create_time_automation(
                name="Daily weather",
                schedule="daily",
                time_of_day=now_in_timezone(config.local_timezone_name).strftime("%H:%M"),
                actions=(
                    AutomationAction(
                        kind="llm_prompt",
                        text="Give the morning weather report.",
                        payload={"delivery": "spoken", "allow_web_search": True},
                        enabled=True,
                    ),
                ),
                source="test",
            )

            executed = loop._maybe_run_due_automation()
            stored = loop.runtime.automation_store.get(entry.automation_id)

        self.assertTrue(executed)
        self.assertEqual(print_backend.automation_calls, [("Give the morning weather report.", True, "spoken")])
        self.assertEqual(print_backend.synthesize_calls, ["Die Wettervorhersage ist trocken und mild."])
        self.assertEqual(player.played, [b"PCM"])
        assert stored is not None
        self.assertIsNotNone(stored.last_triggered_at)
        self.assertIn("automation_executed=true", lines)

    def test_due_spoken_automation_is_governor_blocked_after_recent_social_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                automation_store_path=str(Path(temp_dir) / "state" / "automations.json"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                automation_poll_interval_s=0.0,
                proactive_governor_global_prompt_cooldown_s=300.0,
                proactive_governor_source_repeat_cooldown_s=0.0,
                proactive_quiet_hours_visual_only_enabled=False,
            )
            loop, lines, _realtime_session, print_backend, _recorder, player, _printer = self.make_loop(config=config)
            entry = loop.runtime.create_time_automation(
                name="Daily weather",
                schedule="daily",
                time_of_day=now_in_timezone(config.local_timezone_name).strftime("%H:%M"),
                actions=(
                    AutomationAction(
                        kind="llm_prompt",
                        text="Give the morning weather report.",
                        payload={"delivery": "spoken", "allow_web_search": True},
                        enabled=True,
                    ),
                ),
                source="test",
            )

            spoke = loop.handle_social_trigger(
                SocialTriggerDecision(
                    trigger_id="attention_window",
                    prompt="Soll ich dir helfen?",
                    reason="User seems attentive and quiet.",
                    observed_at=12.0,
                    priority=SocialTriggerPriority.ATTENTION_WINDOW,
                    score=0.92,
                    threshold=0.86,
                    evidence=(),
                )
            )
            executed = loop._maybe_run_due_automation()
            stored = loop.runtime.automation_store.get(entry.automation_id)

        self.assertTrue(spoke)
        self.assertFalse(executed)
        self.assertEqual(len(print_backend.automation_calls), 0)
        self.assertIn("automation_skipped=governor_global_prompt_cooldown_active", lines)
        assert stored is not None
        self.assertIsNone(stored.last_triggered_at)

    def test_idle_loop_spoken_automation_uses_processing_and_answering_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                automation_store_path=str(Path(temp_dir) / "state" / "automations.json"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                automation_poll_interval_s=0.0,
            )
            loop, _lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
                config=config
            )
            loop.runtime.create_time_automation(
                name="Daily weather",
                schedule="daily",
                time_of_day=now_in_timezone(config.local_timezone_name).strftime("%H:%M"),
                actions=(
                    AutomationAction(
                        kind="llm_prompt",
                        text="Give the morning weather report.",
                        payload={"delivery": "spoken", "allow_web_search": True},
                        enabled=True,
                    ),
                ),
                source="test",
            )
            loop.print_backend.synthesize_sleep_s = 0.12
            feedback_kinds: list[str] = []

            def fake_start(kind: str):
                feedback_kinds.append(kind)
                return lambda: None

            loop._start_working_feedback_loop = fake_start  # type: ignore[method-assign]

            executed = loop._maybe_run_due_automation()

        self.assertTrue(executed)
        self.assertEqual(feedback_kinds[:2], ["processing", "answering"])

    def test_idle_loop_executes_due_printed_automation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                automation_store_path=str(Path(temp_dir) / "state" / "automations.json"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                automation_poll_interval_s=0.0,
            )
            loop, lines, _realtime_session, print_backend, _recorder, _player, printer = self.make_loop(config=config)
            entry = loop.runtime.create_time_automation(
                name="Daily headlines",
                schedule="daily",
                time_of_day=now_in_timezone(config.local_timezone_name).strftime("%H:%M"),
                actions=(
                    AutomationAction(
                        kind="llm_prompt",
                        text="Print the main headlines of the day.",
                        payload={"delivery": "printed", "allow_web_search": True},
                        enabled=True,
                    ),
                ),
                source="test",
            )

            executed = loop._maybe_run_due_automation()
            stored = loop.runtime.automation_store.get(entry.automation_id)

        self.assertTrue(executed)
        self.assertEqual(print_backend.automation_calls, [("Print the main headlines of the day.", True, "printed")])
        self.assertEqual(printer.printed, ["GUTEN TAG"])

    def test_idle_loop_executes_allowed_tool_call_automation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                automation_store_path=str(Path(temp_dir) / "state" / "automations.json"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                automation_poll_interval_s=0.0,
            )
            loop, lines, _realtime_session, print_backend, _recorder, _player, printer = self.make_loop(config=config)
            entry = loop.runtime.create_time_automation(
                name="Tool print",
                schedule="daily",
                time_of_day=now_in_timezone(config.local_timezone_name).strftime("%H:%M"),
                actions=(
                    AutomationAction(
                        kind="tool_call",
                        tool_name="print_receipt",
                        payload={"text": "Bitte ausdrucken"},
                        enabled=True,
                    ),
                ),
                source="test",
            )

            executed = loop._maybe_run_due_automation()
            stored = loop.runtime.automation_store.get(entry.automation_id)

        self.assertTrue(executed)
        self.assertEqual(print_backend.calls[-1][1:], (None, "Bitte ausdrucken", "tool"))
        self.assertEqual(printer.printed, ["GUTEN TAG"])
        assert stored is not None
        self.assertIsNotNone(stored.last_triggered_at)
        self.assertIn("automation_tool_call=print_receipt", lines)

    def test_idle_loop_rejects_unsafe_tool_call_automation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                automation_store_path=str(Path(temp_dir) / "state" / "automations.json"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                automation_poll_interval_s=0.0,
            )
            loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)
            entry = loop.runtime.create_time_automation(
                name="Unsafe tool",
                schedule="daily",
                time_of_day=now_in_timezone(config.local_timezone_name).strftime("%H:%M"),
                actions=(
                    AutomationAction(
                        kind="tool_call",
                        tool_name="update_simple_setting",
                        payload={"setting": "memory_capacity", "value": 4, "action": "set"},
                        enabled=True,
                    ),
                ),
                source="test",
            )

            executed = loop._maybe_run_due_automation()
            stored = loop.runtime.automation_store.get(entry.automation_id)

        self.assertFalse(executed)
        assert stored is not None
        self.assertIsNone(stored.last_triggered_at)
        self.assertTrue(any("Automation tool_call is not allowed" in line for line in lines))

    def test_idle_loop_printed_automation_uses_processing_and_printing_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                automation_store_path=str(Path(temp_dir) / "state" / "automations.json"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                automation_poll_interval_s=0.0,
            )
            loop, _lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
                config=config
            )
            loop.runtime.create_time_automation(
                name="Daily headlines",
                schedule="daily",
                time_of_day=now_in_timezone(config.local_timezone_name).strftime("%H:%M"),
                actions=(
                    AutomationAction(
                        kind="llm_prompt",
                        text="Print the main headlines of the day.",
                        payload={"delivery": "printed", "allow_web_search": True},
                        enabled=True,
                    ),
                ),
                source="test",
            )
            feedback_kinds: list[str] = []

            def fake_start(kind: str):
                feedback_kinds.append(kind)
                return lambda: None

            loop._start_working_feedback_loop = fake_start  # type: ignore[method-assign]

            executed = loop._maybe_run_due_automation()

        self.assertTrue(executed)
        self.assertEqual(feedback_kinds[:2], ["processing", "printing"])

    def test_idle_loop_executes_sensor_automation_from_camera_event(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                automation_store_path=str(Path(temp_dir) / "state" / "automations.json"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, lines, _realtime_session, print_backend, _recorder, player, _printer = self.make_loop(config=config)
            loop._handle_create_sensor_automation_tool_call(
                {
                    "name": "Visitor hello",
                    "trigger_kind": "camera_person_visible",
                    "delivery": "spoken",
                    "content_mode": "llm_prompt",
                    "content": "Say hello to the visitor.",
                    "allow_web_search": False,
                    "confirmed": True,
                }
            )

            loop.handle_sensor_observation(
                {
                    "sensor": {"inspected": True, "observed_at": 5.0},
                    "pir": {"motion_detected": True, "low_motion": False, "no_motion_for_s": 0.0},
                    "camera": {
                        "person_visible": True,
                        "person_visible_for_s": 0.0,
                        "looking_toward_device": True,
                        "body_pose": "upright",
                        "smiling": False,
                        "hand_or_object_near_camera": False,
                        "hand_or_object_near_camera_for_s": 0.0,
                    },
                    "vad": {
                        "speech_detected": False,
                        "speech_detected_for_s": 0.0,
                        "quiet": True,
                        "quiet_for_s": 12.0,
                        "distress_detected": False,
                    },
                },
                ("camera.person_visible",),
            )
            executed = loop._maybe_run_sensor_automation()

        self.assertTrue(executed)
        self.assertEqual(print_backend.automation_calls, [("Say hello to the visitor.", False, "spoken")])
        self.assertEqual(player.played, [b"PCM"])
        self.assertIn("automation_trigger_source=sensor", lines)
        self.assertIn("automation_event_name=camera.person_visible", lines)

    def test_idle_loop_executes_sensor_automation_from_duration_facts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                automation_store_path=str(Path(temp_dir) / "state" / "automations.json"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, lines, _realtime_session, _print_backend, _recorder, _player, printer = self.make_loop(config=config)
            loop._handle_create_sensor_automation_tool_call(
                {
                    "name": "Quiet room print",
                    "trigger_kind": "vad_quiet",
                    "hold_seconds": 30,
                    "delivery": "printed",
                    "content_mode": "static_text",
                    "content": "Bitte leise bleiben.",
                    "cooldown_seconds": 120,
                    "confirmed": True,
                }
            )

            loop.handle_sensor_observation(
                {
                    "sensor": {"inspected": False, "observed_at": 60.0},
                    "pir": {"motion_detected": False, "low_motion": True, "no_motion_for_s": 45.0},
                    "camera": {
                        "person_visible": False,
                        "person_visible_for_s": 0.0,
                        "looking_toward_device": False,
                        "body_pose": "unknown",
                        "smiling": False,
                        "hand_or_object_near_camera": False,
                        "hand_or_object_near_camera_for_s": 0.0,
                    },
                    "vad": {
                        "speech_detected": False,
                        "speech_detected_for_s": 0.0,
                        "quiet": True,
                        "quiet_for_s": 45.0,
                        "distress_detected": False,
                    },
                },
                (),
            )
            executed = loop._maybe_run_sensor_automation()

        self.assertTrue(executed)
        self.assertEqual(printer.printed, ["Bitte leise bleiben."])
        self.assertIn("automation_trigger_source=sensor", lines)

    def test_idle_loop_executes_self_coding_skill_package_schedule_and_sensor_delivery(self) -> None:
        class _LoopBriefingBackend(FakePrintBackend):
            def respond_with_metadata(
                self,
                prompt: str,
                *,
                conversation=None,
                instructions: str | None = None,
                allow_web_search: bool | None = None,
            ) -> OpenAITextResponse:
                del conversation, instructions, allow_web_search
                self.summary_calls.append(prompt)
                return OpenAITextResponse(
                    text="Guten Morgen. Hier ist dein Morgenabstract.",
                    response_id="resp_skill_summary_1",
                    request_id="req_skill_summary_1",
                    used_web_search=False,
                    model="gpt-5.2",
                    token_usage=None,
                )

        class _SkillPackageCompileDriver:
            def run_compile(self, request, *, event_sink=None) -> CodexCompileResult:
                del request, event_sink
                return CodexCompileResult(
                    status="ok",
                    summary="Compiled a self-coding morning briefing package.",
                    artifacts=(
                        CodexCompileArtifact(
                            kind=ArtifactKind.SKILL_PACKAGE,
                            artifact_name="skill_package.json",
                            media_type="application/json",
                            content=_self_coding_skill_package_payload(),
                            summary="Morning briefing package.",
                        ),
                    ),
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                automation_store_path=str(Path(temp_dir) / "state" / "automations.json"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            backend = _LoopBriefingBackend()
            loop, lines, _realtime_session, _print_backend, _recorder, player, _printer = self.make_loop(
                config=config,
                print_backend=backend,
                agent_provider=backend,
            )
            store = SelfCodingStore.from_project_root(config.project_root)
            worker = SelfCodingCompileWorker(store=store, driver=_SkillPackageCompileDriver())
            activation = SelfCodingActivationService(
                store=store,
                automation_store=AutomationStore(config.automation_store_path, timezone_name=config.local_timezone_name),
            )

            job = worker.ensure_job_for_session(_ready_self_coding_skill_session())
            completed = worker.run_job(job.job_id)
            active = activation.confirm_activation(job_id=completed.job_id, confirmed=True)

            automation_ids = tuple(active.metadata["automation_ids"])
            schedule_entry = next(
                loop.runtime.automation_store.get(item)
                for item in automation_ids
                if loop.runtime.automation_store.get(item) is not None
                and getattr(loop.runtime.automation_store.get(item).trigger, "kind", None) == "time"
            )
            sensor_entry = next(
                loop.runtime.automation_store.get(item)
                for item in automation_ids
                if loop.runtime.automation_store.get(item) is not None
                and getattr(loop.runtime.automation_store.get(item).trigger, "kind", None) == "if_then"
            )
            sensor_facts = {
                "sensor": {"inspected": True, "observed_at": 5.0},
                "pir": {"motion_detected": True, "low_motion": False, "no_motion_for_s": 0.0},
                "camera": {
                    "person_visible": True,
                    "count_persons": 1,
                    "person_visible_for_s": 0.0,
                    "looking_toward_device": True,
                    "body_pose": "upright",
                    "smiling": False,
                    "hand_or_object_near_camera": False,
                    "hand_or_object_near_camera_for_s": 0.0,
                },
                "vad": {
                    "speech_detected": False,
                    "speech_detected_for_s": 0.0,
                    "quiet": True,
                    "quiet_for_s": 12.0,
                    "distress_detected": False,
                },
            }

            scheduled_executed = loop._run_automation_entry(schedule_entry, trigger_source="time_schedule")
            loop._latest_sensor_observation_facts = sensor_facts
            first_delivery = loop._run_automation_entry(
                sensor_entry,
                trigger_source="sensor",
                event_name="camera.person_visible",
                facts=sensor_facts,
            )
            second_delivery_same_day = loop._run_automation_entry(
                sensor_entry,
                trigger_source="sensor",
                event_name="camera.person_visible",
                facts=sensor_facts,
            )

        self.assertTrue(scheduled_executed)
        self.assertTrue(first_delivery)
        self.assertTrue(second_delivery_same_day)
        self.assertEqual(len(backend.search_calls), 3)
        self.assertEqual(len(backend.summary_calls), 1)
        self.assertEqual(player.played, [b"PCM"])
        self.assertIn("automation_tool_call=run_self_coding_skill_scheduled", lines)
        self.assertIn("automation_tool_call=run_self_coding_skill_sensor", lines)

    def test_idle_loop_delivers_due_reminder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                reminder_store_path=str(Path(temp_dir) / "state" / "reminders.json"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                reminder_poll_interval_s=0.0,
            )
            loop, lines, _realtime_session, print_backend, _recorder, player, _printer = self.make_loop(config=config)
            loop.runtime.schedule_reminder(
                due_at=now_in_timezone(config.local_timezone_name).isoformat(),
                summary="Medikament nehmen",
                kind="medication",
                source="test",
            )

            delivered = loop._maybe_deliver_due_reminder()
            stored_entries = loop.runtime.reminder_store.load_entries()

        self.assertTrue(delivered)
        self.assertEqual(len(print_backend.reminder_calls), 1)
        self.assertEqual(print_backend.synthesize_calls, ["Erinnerung: Medikament nehmen"])
        self.assertEqual(player.played, [b"PCM"])
        self.assertIn("reminder_delivered=true", lines)
        self.assertTrue(stored_entries[0].delivered)

    def test_social_trigger_is_governor_blocked_after_recent_reminder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                reminder_store_path=str(Path(temp_dir) / "state" / "reminders.json"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                reminder_poll_interval_s=0.0,
                proactive_governor_global_prompt_cooldown_s=300.0,
                proactive_governor_source_repeat_cooldown_s=0.0,
                proactive_quiet_hours_visual_only_enabled=False,
            )
            loop, lines, _realtime_session, print_backend, _recorder, player, _printer = self.make_loop(config=config)
            loop.runtime.schedule_reminder(
                due_at=now_in_timezone(config.local_timezone_name).isoformat(),
                summary="Medikament nehmen",
                kind="medication",
                source="test",
            )

            delivered = loop._maybe_deliver_due_reminder()
            spoke = loop.handle_social_trigger(
                SocialTriggerDecision(
                    trigger_id="attention_window",
                    prompt="Soll ich dir helfen?",
                    reason="User seems attentive and quiet.",
                    observed_at=12.0,
                    priority=SocialTriggerPriority.ATTENTION_WINDOW,
                    score=0.92,
                    threshold=0.86,
                    evidence=(),
                )
            )

        self.assertTrue(delivered)
        self.assertFalse(spoke)
        self.assertEqual(len(print_backend.reminder_calls), 1)
        self.assertEqual(player.played, [b"PCM"])
        self.assertIn("social_trigger_skipped=governor_global_prompt_cooldown_active", lines)

    def test_idle_loop_reminder_uses_processing_and_answering_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                reminder_store_path=str(Path(temp_dir) / "state" / "reminders.json"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                reminder_poll_interval_s=0.0,
            )
            loop, _lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
                config=config
            )
            loop.runtime.schedule_reminder(
                due_at=now_in_timezone(config.local_timezone_name).isoformat(),
                summary="Medikament nehmen",
                kind="medication",
                source="test",
            )
            loop.print_backend.synthesize_sleep_s = 0.12
            feedback_kinds: list[str] = []

            def fake_start(kind: str):
                feedback_kinds.append(kind)
                return lambda: None

            loop._start_working_feedback_loop = fake_start  # type: ignore[method-assign]

            delivered = loop._maybe_deliver_due_reminder()

        self.assertTrue(delivered)
        self.assertEqual(feedback_kinds[:2], ["processing", "answering"])

    def test_idle_loop_delivers_longterm_proactive_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_enabled=True,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                long_term_memory_proactive_enabled=True,
                long_term_memory_proactive_poll_interval_s=0.0,
                long_term_memory_proactive_min_confidence=0.7,
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, lines, _realtime_session, print_backend, _recorder, player, _printer = self.make_loop(config=config)
            loop.runtime.long_term_memory.object_store.apply_consolidation(
                LongTermConsolidationResultV1(
                    turn_id="turn:thread",
                    occurred_at=datetime(2026, 3, 14, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                    episodic_objects=(),
                    durable_objects=(
                        LongTermMemoryObjectV1(
                            memory_id="thread:walk_weather",
                            kind="thread_summary",
                            summary="Ongoing thread about the user's plan to walk if the weather is nice.",
                            details="Reflected from multiple related turns.",
                            source=_longterm_source("turn:thread"),
                            status="active",
                            confidence=0.82,
                            sensitivity="normal",
                            slot_key="thread:user:main:walk_weather",
                            value_key="walk_weather",
                            attributes={"support_count": 4},
                        ),
                    ),
                    deferred_objects=(),
                    conflicts=(),
                    graph_edges=(),
                )
            )

            delivered = loop._maybe_run_long_term_memory_proactive()
            history = loop.runtime.long_term_memory.proactive_policy.state_store.load_entries()

        self.assertTrue(delivered)
        self.assertEqual(len(print_backend.proactive_calls), 1)
        self.assertEqual(print_backend.proactive_calls[0][0], "longterm:candidate:thread_walk_weather:followup")
        self.assertEqual(player.played, [b"PCM"])
        self.assertIn("longterm_proactive_candidate=candidate:thread_walk_weather:followup", lines)
        self.assertIn("longterm_proactive_prompt_mode=llm", lines)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].delivery_count, 1)

    def test_idle_loop_delivers_sensor_memory_camera_offer_with_live_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_enabled=True,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                long_term_memory_proactive_enabled=True,
                long_term_memory_proactive_poll_interval_s=0.0,
                long_term_memory_proactive_min_confidence=0.7,
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, lines, _realtime_session, print_backend, _recorder, player, _printer = self.make_loop(config=config)
            reference = datetime.now(ZoneInfo(config.local_timezone_name))
            daypart = (
                "morning"
                if 5 <= reference.hour < 11
                else "afternoon"
                if 11 <= reference.hour < 17
                else "evening"
                if 17 <= reference.hour < 22
                else "night"
            )
            weekday_class = "weekend" if reference.weekday() >= 5 else "weekday"
            today = reference.date().isoformat()
            loop._latest_sensor_observation_facts = {
                "sensor": {"observed_at": time.monotonic()},
                "camera": {
                    "person_visible": True,
                    "looking_toward_device": False,
                    "hand_or_object_near_camera": True,
                    "body_pose": "upright",
                },
                "vad": {
                    "speech_detected": False,
                    "quiet": True,
                },
            }
            loop.runtime.long_term_memory.object_store.apply_reflection(
                LongTermReflectionResultV1(
                    reflected_objects=(),
                    created_summaries=(
                        LongTermMemoryObjectV1(
                            memory_id=f"routine:interaction:camera_showing:{weekday_class}:{daypart}",
                            kind="pattern",
                            summary=f"Camera showing is typical in the {daypart} on {weekday_class}s.",
                            source=_longterm_source("turn:sensor"),
                            status="active",
                            confidence=0.82,
                            sensitivity="low",
                            valid_from="2026-03-03",
                            valid_to="2026-03-17",
                            attributes={
                                "memory_domain": "sensor_routine",
                                "routine_type": "interaction",
                                "interaction_type": "camera_showing",
                                "weekday_class": weekday_class,
                                "daypart": daypart,
                            },
                        ),
                    ),
                )
            )

            delivered = loop._maybe_run_long_term_memory_proactive()
            history = loop.runtime.long_term_memory.proactive_policy.state_store.load_entries()

        self.assertTrue(delivered)
        self.assertEqual(len(print_backend.proactive_calls), 1)
        expected_trigger = f"longterm:candidate:sensor_camera_offer:{today}:{daypart}"
        self.assertEqual(print_backend.proactive_calls[0][0], expected_trigger)
        self.assertEqual(player.played, [b"PCM"])
        self.assertIn(f"longterm_proactive_candidate=candidate:sensor_camera_offer:{today}:{daypart}", lines)
        self.assertIn("longterm_proactive_prompt_mode=llm", lines)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].delivery_count, 1)

    def test_idle_loop_blocks_sensitive_longterm_proactive_candidate_in_multi_person_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_enabled=True,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                long_term_memory_proactive_enabled=True,
                long_term_memory_proactive_poll_interval_s=0.0,
                long_term_memory_proactive_min_confidence=0.7,
                long_term_memory_proactive_allow_sensitive=True,
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, lines, _realtime_session, print_backend, _recorder, player, _printer = self.make_loop(config=config)
            loop._latest_sensor_observation_facts = {
                "sensor": {
                    "observed_at": time.monotonic(),
                    "presence_session_id": 17,
                },
                "camera": {
                    "person_visible": True,
                    "person_count": 2,
                    "person_count_unknown": False,
                },
                "vad": {
                    "speech_detected": True,
                    "quiet": False,
                },
                "respeaker": {
                    "direction_confidence": 0.92,
                },
                "audio_policy": {
                    "presence_audio_active": True,
                    "speaker_direction_stable": True,
                    "room_busy_or_overlapping": False,
                },
            }
            loop.runtime.long_term_memory.object_store.apply_consolidation(
                LongTermConsolidationResultV1(
                    turn_id="turn:thread",
                    occurred_at=datetime(2026, 3, 14, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                    episodic_objects=(),
                    durable_objects=(
                        LongTermMemoryObjectV1(
                            memory_id="thread:family_doctor",
                            kind="thread_summary",
                            summary="Ongoing thread about Janina's doctor appointment.",
                            details="Reflected from multiple related turns.",
                            source=_longterm_source("turn:thread"),
                            status="active",
                            confidence=0.84,
                            sensitivity="private",
                            slot_key="thread:user:main:family_doctor",
                            value_key="family_doctor",
                            attributes={"support_count": 4},
                        ),
                    ),
                    deferred_objects=(),
                    conflicts=(),
                    graph_edges=(),
                )
            )

            delivered = loop._maybe_run_long_term_memory_proactive()
            history = loop.runtime.long_term_memory.proactive_policy.state_store.load_entries()

        self.assertFalse(delivered)
        self.assertEqual(print_backend.proactive_calls, [])
        self.assertEqual(player.played, [])
        self.assertIn("longterm_proactive_skipped=multi_person_context", lines)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].skip_count, 1)
        self.assertEqual(history[0].last_skip_reason, "multi_person_context")

    def test_idle_loop_blocks_longterm_proactive_candidate_when_multimodal_initiative_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_enabled=True,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                long_term_memory_proactive_enabled=True,
                long_term_memory_proactive_poll_interval_s=0.0,
                long_term_memory_proactive_min_confidence=0.7,
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, lines, _realtime_session, print_backend, _recorder, player, _printer = self.make_loop(config=config)
            reference = datetime.now(ZoneInfo(config.local_timezone_name))
            daypart = (
                "morning"
                if 5 <= reference.hour < 11
                else "afternoon"
                if 11 <= reference.hour < 17
                else "evening"
                if 17 <= reference.hour < 22
                else "night"
            )
            weekday_class = "weekend" if reference.weekday() >= 5 else "weekday"
            loop._latest_sensor_observation_facts = {
                "sensor": {"observed_at": time.monotonic()},
                "camera": {
                    "person_visible": True,
                    "looking_toward_device": False,
                    "hand_or_object_near_camera": True,
                    "body_pose": "upright",
                },
                "vad": {
                    "speech_detected": False,
                    "quiet": True,
                },
                "multimodal_initiative": {
                    "ready": False,
                    "confidence": 0.58,
                    "block_reason": "low_confidence_speaker_association",
                    "recommended_channel": "display",
                },
            }
            loop.runtime.long_term_memory.object_store.apply_reflection(
                LongTermReflectionResultV1(
                    reflected_objects=(),
                    created_summaries=(
                        LongTermMemoryObjectV1(
                            memory_id=f"routine:interaction:camera_showing:{weekday_class}:{daypart}",
                            kind="pattern",
                            summary=f"Camera showing is typical in the {daypart} on {weekday_class}s.",
                            source=_longterm_source("turn:sensor"),
                            status="active",
                            confidence=0.82,
                            sensitivity="low",
                            valid_from="2026-03-03",
                            valid_to="2026-03-17",
                            attributes={
                                "memory_domain": "sensor_routine",
                                "routine_type": "interaction",
                                "interaction_type": "camera_showing",
                                "weekday_class": weekday_class,
                                "daypart": daypart,
                            },
                        ),
                    ),
                )
            )

            delivered = loop._maybe_run_long_term_memory_proactive()
            history = loop.runtime.long_term_memory.proactive_policy.state_store.load_entries()

        self.assertFalse(delivered)
        self.assertEqual(len(print_backend.proactive_calls), 0)
        self.assertEqual(player.played, [])
        self.assertIn("longterm_proactive_skipped=low_confidence_speaker_association", lines)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].last_skip_reason, "low_confidence_speaker_association")

    def test_idle_loop_delivers_due_reminder_with_generic_backend_fallback(self) -> None:
        class LegacyReminderBackend(FakePrintBackend):
            phrase_due_reminder_with_metadata = None

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                reminder_store_path=str(Path(temp_dir) / "state" / "reminders.json"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                reminder_poll_interval_s=0.0,
            )
            print_backend = LegacyReminderBackend()
            loop, lines, _realtime_session, _print_backend, _recorder, player, _printer = self.make_loop(
                config=config,
                print_backend=print_backend,
            )
            loop.runtime.schedule_reminder(
                due_at=now_in_timezone(config.local_timezone_name).isoformat(),
                summary="Medikament nehmen",
                kind="medication",
                source="test",
            )

            delivered = loop._maybe_deliver_due_reminder()
            stored_entries = loop.runtime.reminder_store.load_entries()

        self.assertTrue(delivered)
        self.assertEqual(len(print_backend.reminder_calls), 0)
        self.assertEqual(len(print_backend.generic_reminder_calls), 1)
        self.assertIn("Speak the reminder now.", print_backend.generic_reminder_calls[0][0])
        self.assertEqual(player.played, [b"PCM"])
        self.assertIn("reminder_backend_fallback=generic", lines)
        self.assertIn("reminder_delivered=true", lines)
        self.assertTrue(stored_entries[0].delivered)

    def test_update_user_profile_tool_call_updates_user_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            personality_dir = Path(temp_dir) / "personality"
            personality_dir.mkdir(parents=True, exist_ok=True)
            user_path = personality_dir / "USER.md"
            user_path.write_text("User: Thom.\n", encoding="utf-8")
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)

            result = loop._handle_update_user_profile_tool_call(
                {
                    "category": "preferred_name",
                    "instruction": "Call the user Thom in future turns.",
                    "confirmed": True,
                }
            )

            user_text = user_path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "updated")
        self.assertIn("User: Thom.", user_text)
        self.assertIn("preferred_name: Call the user Thom in future turns.", user_text)
        self.assertEqual(loop.runtime.memory.ledger[-1].kind, "preference")
        self.assertIn("user_profile_tool_call=true", lines)

    def test_remember_contact_and_lookup_contact_tool_calls_handle_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)

            first = loop._handle_remember_contact_tool_call(
                {
                    "given_name": "Corinna",
                    "family_name": "Maier",
                    "phone": "01761234",
                    "role": "Physiotherapeutin",
                    "confirmed": True,
                }
            )
            second = loop._handle_remember_contact_tool_call(
                {
                    "given_name": "Corinna",
                    "family_name": "Schmidt",
                    "phone": "0309988",
                    "role": "Nachbarin",
                    "confirmed": True,
                }
            )
            lookup = loop._handle_lookup_contact_tool_call(
                {
                    "name": "Corinna",
                    "confirmed": True,
                }
            )
            resolved = loop._handle_lookup_contact_tool_call(
                {
                    "name": "Corinna",
                    "role": "Physiotherapeutin",
                    "confirmed": True,
                }
            )

        self.assertEqual(first["status"], "created")
        self.assertEqual(second["status"], "created")
        self.assertEqual(lookup["status"], "needs_clarification")
        self.assertEqual(len(lookup["options"]), 2)
        self.assertEqual(resolved["status"], "found")
        self.assertEqual(resolved["label"], "Corinna Maier")
        self.assertEqual(resolved["phones"], ["01761234"])
        self.assertIn("graph_contact_tool_call=true", lines)
        self.assertIn("graph_contact_lookup=true", lines)

    def test_memory_conflict_tool_calls_list_and_resolve_open_conflicts(self) -> None:
        existing = LongTermMemoryObjectV1(
            memory_id="fact:corinna_phone_old",
            kind="contact_method_fact",
            summary="Corinna Maier can be reached at +491761234.",
            details="Use the mobile number ending in 1234.",
            source=_longterm_source("turn:1"),
            status="active",
            confidence=0.95,
            slot_key="contact:person:corinna_maier:phone",
            value_key="+491761234",
            attributes={"person_ref": "person:corinna_maier"},
        )
        candidate = LongTermMemoryObjectV1(
            memory_id="fact:corinna_phone_new",
            kind="contact_method_fact",
            summary="Corinna Maier can be reached at +4940998877.",
            details="Use the office number ending in 8877.",
            source=_longterm_source("turn:2"),
            status="uncertain",
            confidence=0.92,
            slot_key="contact:person:corinna_maier:phone",
            value_key="+4940998877",
            attributes={"person_ref": "person:corinna_maier"},
        )
        conflict = LongTermMemoryConflictV1(
            slot_key="contact:person:corinna_maier:phone",
            candidate_memory_id="fact:corinna_phone_new",
            existing_memory_ids=("fact:corinna_phone_old",),
            question="Which phone number should I use for Corinna Maier?",
            reason="Conflicting phone numbers exist.",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)
            loop.runtime.long_term_memory.object_store.apply_consolidation(
                LongTermConsolidationResultV1(
                    turn_id="turn:2",
                    occurred_at=datetime(2026, 3, 14, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                    episodic_objects=(),
                    durable_objects=(existing,),
                    deferred_objects=(candidate,),
                    conflicts=(conflict,),
                    graph_edges=(),
                )
            )

            listed = loop._handle_get_memory_conflicts_tool_call(
                {
                    "query_text": "Corinna phone number",
                    "confirmed": True,
                }
            )
            resolved = loop._handle_resolve_memory_conflict_tool_call(
                {
                    "slot_key": "contact:person:corinna_maier:phone",
                    "selected_memory_id": "fact:corinna_phone_new",
                    "confirmed": True,
                }
            )
            objects = {
                item.memory_id: item
                for item in loop.runtime.long_term_memory.object_store.load_objects()
            }
            conflicts = loop.runtime.long_term_memory.object_store.load_conflicts()

        self.assertEqual(listed["status"], "ok")
        self.assertEqual(listed["conflict_count"], 1)
        self.assertEqual(listed["conflicts"][0]["slot_key"], "contact:person:corinna_maier:phone")
        self.assertEqual(len(listed["conflicts"][0]["options"]), 2)
        self.assertEqual(resolved["status"], "resolved")
        self.assertEqual(resolved["selected_memory_id"], "fact:corinna_phone_new")
        self.assertEqual(resolved["superseded_memory_ids"], ["fact:corinna_phone_old"])
        self.assertEqual(objects["fact:corinna_phone_new"].status, "active")
        self.assertEqual(objects["fact:corinna_phone_old"].status, "superseded")
        self.assertEqual(conflicts, ())
        self.assertIn("memory_conflict_tool_call=true", lines)
        self.assertIn("memory_conflict_resolved=contact:person:corinna_maier:phone", lines)

    def test_resolve_memory_conflict_requires_confirmation_when_speaker_signal_is_uncertain(self) -> None:
        existing = LongTermMemoryObjectV1(
            memory_id="fact:corinna_phone_old",
            kind="contact_method_fact",
            summary="Corinna Maier can be reached at +491761234.",
            source=_longterm_source("turn:1"),
            status="active",
            confidence=0.95,
            slot_key="contact:person:corinna_maier:phone",
            value_key="+491761234",
        )
        candidate = LongTermMemoryObjectV1(
            memory_id="fact:corinna_phone_new",
            kind="contact_method_fact",
            summary="Corinna Maier can be reached at +4940998877.",
            source=_longterm_source("turn:2"),
            status="uncertain",
            confidence=0.92,
            slot_key="contact:person:corinna_maier:phone",
            value_key="+4940998877",
        )
        conflict = LongTermMemoryConflictV1(
            slot_key="contact:person:corinna_maier:phone",
            candidate_memory_id="fact:corinna_phone_new",
            existing_memory_ids=("fact:corinna_phone_old",),
            question="Which phone number should I use for Corinna Maier?",
            reason="Conflicting phone numbers exist.",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, _lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)
            loop.runtime.long_term_memory.object_store.apply_consolidation(
                LongTermConsolidationResultV1(
                    turn_id="turn:2",
                    occurred_at=datetime(2026, 3, 14, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                    episodic_objects=(),
                    durable_objects=(existing,),
                    deferred_objects=(candidate,),
                    conflicts=(conflict,),
                    graph_edges=(),
                )
            )
            loop.runtime.user_voice_status = "unknown_voice"

            with self.assertRaisesRegex(RuntimeError, "Please ask for clear confirmation"):
                loop._handle_resolve_memory_conflict_tool_call(
                    {
                        "slot_key": "contact:person:corinna_maier:phone",
                        "selected_memory_id": "fact:corinna_phone_new",
                    }
                )

            resolved = loop._handle_resolve_memory_conflict_tool_call(
                {
                    "slot_key": "contact:person:corinna_maier:phone",
                    "selected_memory_id": "fact:corinna_phone_new",
                    "confirmed": True,
                }
            )

        self.assertEqual(resolved["status"], "resolved")

    def test_remember_preference_and_plan_tool_calls_feed_provider_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                openai_web_search_timezone="Europe/Berlin",
            )
            loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)

            preference = loop._handle_remember_preference_tool_call(
                {
                    "category": "brand",
                    "value": "Melitta",
                    "for_product": "coffee",
                    "confirmed": True,
                }
            )
            plan = loop._handle_remember_plan_tool_call(
                {
                    "summary": "go for a walk",
                    "when": "today",
                    "confirmed": True,
                }
            )
            loop.runtime.last_transcript = "Wie wird das Wetter heute?"
            provider_context = loop.runtime.provider_conversation_context()
            system_contexts = [content for role, content in provider_context if role == "system"]

        self.assertEqual(preference["edge_type"], "user_prefers")
        self.assertEqual(plan["edge_type"], "user_plans")
        self.assertTrue(any("All user-facing spoken and written replies for this turn must be in German." in content for content in system_contexts))
        self.assertTrue(any("Melitta" in content for content in system_contexts))
        self.assertTrue(any("go for a walk" in content for content in system_contexts))
        self.assertIn("graph_preference_tool_call=true", lines)
        self.assertIn("graph_plan_tool_call=true", lines)

    def test_update_user_profile_requires_confirmation_for_unknown_voice(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            personality_dir = Path(temp_dir) / "personality"
            personality_dir.mkdir(parents=True, exist_ok=True)
            (personality_dir / "USER.md").write_text("User: Thom.\n", encoding="utf-8")
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, _lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)
            loop.runtime.update_user_voice_assessment(
                status="unknown_voice",
                confidence=0.22,
                checked_at="2026-03-13T12:30:00+00:00",
            )

            with self.assertRaisesRegex(RuntimeError, "confirmed=true"):
                loop._handle_update_user_profile_tool_call(
                    {
                        "category": "preferred_name",
                        "instruction": "Call the user Thom in future turns.",
                    }
                )

            result = loop._handle_update_user_profile_tool_call(
                {
                    "category": "preferred_name",
                    "instruction": "Call the user Thom in future turns.",
                    "confirmed": True,
                }
            )

        self.assertEqual(result["status"], "updated")

    def test_update_personality_tool_call_updates_personality_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            personality_dir = Path(temp_dir) / "personality"
            personality_dir.mkdir(parents=True, exist_ok=True)
            personality_path = personality_dir / "PERSONALITY.md"
            personality_path.write_text("Be warm and practical.\n", encoding="utf-8")
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)

            result = loop._handle_update_personality_tool_call(
                {
                    "category": "response_style",
                    "instruction": "Keep answers very short and calm.",
                    "confirmed": True,
                }
            )

            personality_text = personality_path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "updated")
        self.assertIn("Be warm and practical.", personality_text)
        self.assertIn("response_style: Keep answers very short and calm.", personality_text)
        self.assertEqual(loop.runtime.memory.ledger[-1].kind, "preference")
        self.assertIn("personality_tool_call=true", lines)

    def test_configure_world_intelligence_tool_call_updates_rss_subscriptions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, lines, _realtime_session, print_backend, _recorder, _player, _printer = self.make_loop(config=config)
            class _FakeRemoteState:
                def __init__(self) -> None:
                    self.enabled = True
                    self.snapshots: dict[str, dict[str, object]] = {}

                def load_snapshot(self, *, snapshot_kind: str, local_path=None):
                    del local_path
                    payload = self.snapshots.get(snapshot_kind)
                    if payload is None:
                        return None
                    return dict(payload)

                def save_snapshot(self, *, snapshot_kind: str, payload):
                    self.snapshots[snapshot_kind] = dict(payload)

            remote_state = _FakeRemoteState()
            service = loop.runtime.long_term_memory.personality_learning.world_intelligence
            service.remote_state = remote_state
            loop.runtime.long_term_memory.personality_learning.background_loop.remote_state = remote_state
            service.page_loader = lambda url: SimpleNamespace(
                url=url,
                content_type="text/html; charset=utf-8",
                text=(
                    '<html><head><link rel="alternate" '
                    'type="application/rss+xml" href="/feeds/local.xml"></head></html>'
                ),
            )
            service.feed_reader = lambda feed_url, *, max_items, timeout_s: (
                WorldFeedItem(
                    feed_url=feed_url,
                    source="Hamburg News",
                    title="District council approves quieter night buses",
                    link="https://example.com/night-bus",
                    published_at="2026-03-20T07:00:00+00:00",
                ),
            )
            print_backend.search_live_info_with_metadata = lambda *args, **kwargs: SimpleNamespace(
                answer="Discovered source pages.",
                sources=("https://example.com/hamburg",),
                used_web_search=True,
                response_id="resp_world_discovery_1",
                request_id="req_world_discovery_1",
                model="gpt-5.2-chat-latest",
                token_usage=None,
            )

            result = loop._handle_configure_world_intelligence_tool_call(
                {
                    "action": "discover",
                    "query": "Find RSS feeds for Hamburg local politics.",
                    "label": "Hamburg local politics",
                    "location_hint": "Hamburg",
                    "region": "Hamburg",
                    "topics": ["local politics"],
                    "auto_subscribe": True,
                    "refresh_after_change": True,
                    "confirmed": True,
                }
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["subscription_count"], 1)
        self.assertEqual(result["discovered_feed_urls"], ["https://example.com/feeds/local.xml"])
        self.assertEqual(result["refresh"]["status"], "refreshed")
        self.assertEqual(result["refresh"]["world_signal_count"], 2)
        self.assertIn("world_intelligence_tool_call=discover", lines)

    def test_update_simple_setting_tool_call_persists_and_applies_memory_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "TWINR_MEMORY_MAX_TURNS=20\nTWINR_MEMORY_KEEP_RECENT=10\n",
                encoding="utf-8",
            )
            config = TwinrConfig.from_env(env_path)
            loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)

            result = loop._handle_update_simple_setting_tool_call(
                {
                    "setting": "memory_capacity",
                    "action": "increase",
                    "confirmed": True,
                }
            )

            env_text = env_path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "updated")
        self.assertEqual(result["memory_max_turns"], 28)
        self.assertEqual(result["memory_keep_recent"], 12)
        self.assertEqual(loop.config.memory_max_turns, 28)
        self.assertEqual(loop.runtime.memory.max_turns, 28)
        self.assertIn("TWINR_MEMORY_MAX_TURNS=28", env_text)
        self.assertIn("TWINR_MEMORY_KEEP_RECENT=12", env_text)
        self.assertEqual(loop.runtime.memory.ledger[-1].kind, "preference")
        self.assertIn("simple_setting_tool_call=true", lines)

    def test_update_simple_setting_requires_confirmation_for_unknown_voice(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "TWINR_MEMORY_MAX_TURNS=20\nTWINR_MEMORY_KEEP_RECENT=10\n",
                encoding="utf-8",
            )
            config = TwinrConfig.from_env(env_path)
            loop, _lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)
            loop.runtime.update_user_voice_assessment(
                status="unknown_voice",
                confidence=0.18,
                checked_at="2026-03-13T21:00:00+00:00",
            )

            with self.assertRaisesRegex(RuntimeError, "confirmed=true"):
                loop._handle_update_simple_setting_tool_call(
                    {
                        "setting": "memory_capacity",
                        "action": "increase",
                    }
                )

            result = loop._handle_update_simple_setting_tool_call(
                {
                    "setting": "memory_capacity",
                    "action": "increase",
                    "confirmed": True,
                }
            )

        self.assertEqual(result["status"], "updated")

    def test_update_simple_setting_can_change_voice_and_speed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_TTS_VOICE=marin",
                        "OPENAI_REALTIME_VOICE=sage",
                        "OPENAI_TTS_SPEED=1.00",
                        "OPENAI_REALTIME_SPEED=1.00",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = TwinrConfig.from_env(env_path)
            loop, _lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)

            voice_result = loop._handle_update_simple_setting_tool_call(
                {
                    "setting": "spoken_voice",
                    "action": "set",
                    "value": "cedar",
                    "confirmed": True,
                }
            )
            speed_result = loop._handle_update_simple_setting_tool_call(
                {
                    "setting": "speech_speed",
                    "action": "decrease",
                    "confirmed": True,
                }
            )

            env_text = env_path.read_text(encoding="utf-8")

        self.assertEqual(voice_result["status"], "updated")
        self.assertEqual(voice_result["voice"], "cedar")
        self.assertEqual(loop.config.openai_tts_voice, "cedar")
        self.assertEqual(loop.config.openai_realtime_voice, "cedar")
        self.assertEqual(speed_result["status"], "updated")
        self.assertEqual(speed_result["speech_speed"], 0.9)
        self.assertEqual(loop.config.openai_tts_speed, 0.9)
        self.assertEqual(loop.config.openai_realtime_speed, 0.9)
        self.assertIn("OPENAI_TTS_VOICE=cedar", env_text)
        self.assertIn("OPENAI_REALTIME_VOICE=cedar", env_text)
        self.assertIn("OPENAI_TTS_SPEED=0.90", env_text)
        self.assertIn("OPENAI_REALTIME_SPEED=0.90", env_text)

    def test_voice_profile_tools_can_enroll_status_and_reset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                voice_profile_store_path=str(Path(temp_dir) / "state" / "voice_profile.json"),
            )
            loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)
            loop._current_turn_audio_pcm = _voice_sample_pcm_bytes()

            enroll = loop._handle_enroll_voice_profile_tool_call({"confirmed": True})
            status = loop._handle_get_voice_profile_status_tool_call({})
            reset = loop._handle_reset_voice_profile_tool_call({})

        self.assertEqual(enroll["status"], "enrolled")
        self.assertGreaterEqual(enroll["sample_count"], 1)
        self.assertEqual(status["status"], "ok")
        self.assertTrue(status["enrolled"])
        self.assertEqual(status["current_signal"], "likely_user")
        self.assertEqual(reset["status"], "reset")
        self.assertFalse(reset["enrolled"])
        self.assertIn("voice_profile_tool_call=true", lines)

    def test_portrait_identity_tools_can_enroll_status_and_reset(self) -> None:
        fake_provider = mock.Mock()
        fake_provider.capture_and_enroll_reference.return_value = SimpleNamespace(
            status="enrolled",
            user_id="main_user",
            display_name="Thom",
            reference_id="ref_1",
            reference_image_count=1,
        )
        fake_provider.summary.return_value = SimpleNamespace(
            user_id="main_user",
            display_name="Thom",
            primary_user=True,
            enrolled=True,
            reference_image_count=1,
            updated_at="2026-03-19T12:00:00Z",
            store_path="/tmp/portrait_identities.json",
        )
        fake_provider.observe.return_value = SimpleNamespace(
            state="likely_reference_user",
            confidence=0.82,
            fused_confidence=0.88,
            temporal_state="stable_match",
            temporal_observation_count=2,
            matched_user_id="main_user",
            matched_user_display_name="Thom",
            candidate_user_count=1,
            capture_source_device="/dev/video0",
        )
        fake_provider.clear_identity_profile.return_value = SimpleNamespace(
            status="cleared",
            user_id="main_user",
            reference_image_count=0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                portrait_match_store_path=str(Path(temp_dir) / "state" / "portrait_identities.json"),
            )
            loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)
            with mock.patch(
                "twinr.agent.tools.handlers.portrait_identity._get_portrait_provider",
                return_value=fake_provider,
            ):
                enroll = loop._handle_enroll_portrait_identity_tool_call({"display_name": "Thom", "confirmed": True})
                status = loop._handle_get_portrait_identity_status_tool_call({"confirmed": True})
                reset = loop._handle_reset_portrait_identity_tool_call({"confirmed": True})

        self.assertEqual(enroll["status"], "enrolled")
        self.assertTrue(enroll["saved"])
        self.assertEqual(enroll["reference_id"], "ref_1")
        self.assertEqual(enroll["coverage_state"], "single_reference")
        self.assertEqual(enroll["recommended_next_step"], "capture_more_variation")
        self.assertIn("capture_more_angle_variation", enroll["guidance_hints"])
        self.assertEqual(status["status"], "ok")
        self.assertTrue(status["enrolled"])
        self.assertEqual(status["current_signal"], "likely_reference_user")
        self.assertEqual(status["current_confidence"], 0.88)
        self.assertEqual(status["matched_user_id"], "main_user")
        self.assertEqual(reset["status"], "cleared")
        self.assertEqual(reset["deleted_reference_count"], 1)
        self.assertIn("portrait_identity_tool_call=true", lines)

    def test_portrait_identity_tool_returns_retry_guidance_for_quality_failures(self) -> None:
        fake_provider = mock.Mock()
        fake_provider.capture_and_enroll_reference.return_value = SimpleNamespace(
            status="no_face_detected",
            user_id="main_user",
            display_name=None,
            reference_id=None,
            reference_image_count=0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                portrait_match_store_path=str(Path(temp_dir) / "state" / "portrait_identities.json"),
            )
            loop, _lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)
            with mock.patch(
                "twinr.agent.tools.handlers.portrait_identity._get_portrait_provider",
                return_value=fake_provider,
            ):
                result = loop._handle_enroll_portrait_identity_tool_call({"confirmed": True})

        self.assertEqual(result["status"], "no_face_detected")
        self.assertEqual(result["quality_state"], "needs_retry")
        self.assertEqual(result["recommended_next_step"], "retry_capture")
        self.assertEqual(result["suggested_follow_up_tool"], "inspect_camera")
        self.assertIn("single_face_in_frame", result["guidance_hints"])
        self.assertIn("centered", result["suggested_inspection_question"])

    def test_inspect_camera_tool_uses_backend_and_reference_image(self) -> None:
        camera = FakeCamera()
        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "user-reference.jpg"
            reference_path.write_bytes(b"\xff\xd8\xffreference")
            config = TwinrConfig(vision_reference_image_path=str(reference_path))
            loop, lines, _realtime_session, print_backend, _recorder, _player, _printer = self.make_loop(
                config=config,
                camera=camera,
            )

            result = loop._handle_inspect_camera_tool_call({"question": "Schau mich mal an"})

        self.assertEqual(camera.capture_calls, 1)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["answer"], "Ich sehe die Kameraansicht.")
        self.assertEqual(len(print_backend.vision_calls), 1)
        prompt, images, conversation, allow_web_search = print_backend.vision_calls[0]
        self.assertIn("Image 2 is a stored reference image of the main user.", prompt)
        self.assertEqual(len(images), 2)
        self._assert_language_contract_only(conversation)
        self.assertFalse(allow_web_search)
        self.assertIn("camera_tool_call=true", lines)
        self.assertIn("vision_image_count=2", lines)

    def test_search_tool_call_runs_search_backend_and_emits_sources(self) -> None:
        config = TwinrConfig(
            search_feedback_delay_ms=0,
            search_feedback_pause_ms=0,
        )
        loop, lines, _realtime_session, print_backend, _recorder, _player, _printer = self.make_loop(config=config)
        print_backend.search_sleep_s = 0.02

        result = loop._handle_search_tool_call(
            {
                "question": "Wann faehrt der Bus nach Hamburg?",
                "location_hint": "Schwarzenbek",
                "date_context": "Friday, 2026-03-13 10:00 (Europe/Berlin)",
            }
        )

        self.assertEqual(len(print_backend.search_calls), 1)
        question, conversation, location_hint, date_context = print_backend.search_calls[0]
        self.assertEqual(question, "Wann faehrt der Bus nach Hamburg?")
        self._assert_language_contract_only(conversation)
        self.assertEqual(location_hint, "Schwarzenbek")
        self.assertEqual(date_context, "Friday, 2026-03-13 10:00 (Europe/Berlin)")
        self.assertEqual(result["answer"], "Bus 24 faehrt um 07:30 Uhr.")
        self.assertEqual(result["sources"], ["https://example.com/fahrplan"])
        self.assertEqual(len(loop.runtime.memory.search_results), 1)
        self.assertEqual(loop.runtime.memory.search_results[0].question, "Wann faehrt der Bus nach Hamburg?")
        self.assertEqual(loop.runtime.memory.search_results[0].sources, ("https://example.com/fahrplan",))
        self.assertIn("Verified web lookup", loop.runtime.memory.turns[0].content)
        self.assertIn("search_tool_call=true", lines)
        self.assertTrue(any(line.startswith("search_source_1=") for line in lines))

    def test_search_tool_call_re_resolves_relative_tomorrow_context(self) -> None:
        config = TwinrConfig(
            search_feedback_delay_ms=0,
            search_feedback_pause_ms=0,
        )
        loop, _lines, _realtime_session, print_backend, _recorder, _player, _printer = self.make_loop(config=config)

        loop._handle_search_tool_call(
            {
                "question": "Wie wird das Wetter morgen in Schwarzenbek?",
                "location_hint": "Schwarzenbek",
                "date_context": "2026-03-15",
            }
        )

        self.assertEqual(len(print_backend.search_calls), 1)
        _question, _conversation, _location_hint, date_context = print_backend.search_calls[0]
        expected_date = (datetime.now(ZoneInfo(config.local_timezone_name)).date() + timedelta(days=1)).isoformat()
        self.assertIn(expected_date, date_context)

    def test_social_trigger_speaks_proactive_prompt(self) -> None:
        loop, lines, _realtime_session, print_backend, _recorder, player, _printer = self.make_loop(
            config=TwinrConfig(proactive_quiet_hours_visual_only_enabled=False)
        )

        spoke = loop.handle_social_trigger(
            SocialTriggerDecision(
                trigger_id="person_returned",
                prompt="Hey Thom, schön dich zu sehen. Wie geht's dir?",
                reason="Person returned after a long absence.",
                observed_at=42.0,
                priority=SocialTriggerPriority.PERSON_RETURNED,
            )
        )

        self.assertTrue(spoke)
        self.assertEqual(loop.runtime.status.value, "waiting")
        self.assertEqual(len(print_backend.proactive_calls), 1)
        self.assertEqual(loop.runtime.last_transcript, "")
        self.assertEqual(loop.runtime.last_response, "Guten Tag")
        self.assertIn("transcript=Hallo Twinr", lines)
        proactive_call = print_backend.proactive_calls[0]
        self.assertEqual(
            proactive_call[:5],
            (
                "person_returned",
                "Person returned after a long absence.",
                "Hey Thom, schön dich zu sehen. Wie geht's dir?",
                int(SocialTriggerPriority.PERSON_RETURNED),
                (),
            ),
        )
        self.assertIsInstance(proactive_call[5], tuple)
        self.assertIsInstance(proactive_call[6], tuple)
        self.assertEqual(print_backend.synthesize_calls, ["Proaktiv: person_returned"])
        self.assertGreaterEqual(len(player.tones), 1)
        self.assertEqual(player.played, [b"PCM", b"PCM"])
        self.assertIn("social_trigger=person_returned", lines)
        self.assertIn("social_prompt_mode=llm", lines)
        self.assertIn("proactive_listen=true", lines)
        self.assertIn("status=listening", lines)
        self.assertIn("social_response_id=resp_social_1", lines)
        self.assertTrue(any(line.startswith("timing_social_tts_ms=") for line in lines))
        social_events = [
            entry
            for entry in loop.runtime.ops_events.tail(limit=40)
            if entry["event"] == "social_trigger_prompted"
            and entry.get("data", {}).get("trigger") == "person_returned"
        ]
        self.assertTrue(social_events)
        self.assertEqual(social_events[-1]["data"]["prompt"], "Proaktiv: person_returned")
        self.assertEqual(
            social_events[-1]["data"]["default_prompt"],
            "Hey Thom, schön dich zu sehen. Wie geht's dir?",
        )
        self.assertEqual(social_events[-1]["data"]["prompt_mode"], "llm")

    def test_social_trigger_uses_processing_and_answering_feedback(self) -> None:
        recorder = FakeRecorder(
            recordings=[
                RuntimeError("No speech detected before timeout"),
            ]
        )
        loop, _lines, _realtime_session, print_backend, _recorder, _player, _printer = self.make_loop(
            config=TwinrConfig(proactive_quiet_hours_visual_only_enabled=False),
            recorder=recorder,
        )
        print_backend.synthesize_sleep_s = 0.12
        feedback_kinds: list[str] = []

        def fake_start(kind: str):
            feedback_kinds.append(kind)
            return lambda: None

        loop._start_working_feedback_loop = fake_start  # type: ignore[method-assign]

        spoke = loop.handle_social_trigger(
            SocialTriggerDecision(
                trigger_id="person_returned",
                prompt="Hey Thom, schön dich zu sehen. Wie geht's dir?",
                reason="Person returned after a long absence.",
                observed_at=42.0,
                priority=SocialTriggerPriority.PERSON_RETURNED,
            )
        )

        self.assertTrue(spoke)
        self.assertEqual(feedback_kinds[:2], ["processing", "answering"])

    def test_social_trigger_opens_hands_free_listening_window_and_times_out(self) -> None:
        recorder = FakeRecorder(
            recordings=[
                RuntimeError("No speech detected before timeout"),
            ]
        )
        loop, lines, realtime_session, print_backend, recorder, player, _printer = self.make_loop(
            config=TwinrConfig(
                proactive_quiet_hours_visual_only_enabled=False,
                conversation_follow_up_timeout_s=3.5,
                audio_follow_up_speech_start_chunks=5,
                audio_follow_up_ignore_ms=420,
            ),
            recorder=recorder,
        )

        spoke = loop.handle_social_trigger(
            SocialTriggerDecision(
                trigger_id="attention_window",
                prompt="Kann ich dir bei etwas helfen?",
                reason="Person was visible and quiet.",
                observed_at=42.0,
                priority=SocialTriggerPriority.ATTENTION_WINDOW,
            )
        )

        self.assertTrue(spoke)
        self.assertEqual(realtime_session.calls, [])
        self.assertEqual(recorder.pause_values, [1200])
        self.assertEqual(recorder.start_timeouts, [3.5])
        self.assertEqual(recorder.speech_start_chunks, [5])
        self.assertEqual(recorder.ignore_initial_ms, [420])
        self.assertEqual(len(print_backend.proactive_calls), 1)
        self.assertGreaterEqual(len(player.tones), 1)
        self.assertEqual(player.played, [b"PCM"])
        self.assertIn("proactive_listen_timeout=true", lines)
        self.assertEqual(loop.runtime.status.value, "waiting")

    def test_social_trigger_uses_direct_prompt_for_safety_events(self) -> None:
        loop, lines, _realtime_session, print_backend, _recorder, player, _printer = self.make_loop()

        spoke = loop.handle_social_trigger(
            SocialTriggerDecision(
                trigger_id="possible_fall",
                prompt="Brauchst du Hilfe?",
                reason="Person disappeared after a fall-like transition.",
                observed_at=42.0,
                priority=SocialTriggerPriority.POSSIBLE_FALL,
            )
        )

        self.assertTrue(spoke)
        self.assertEqual(print_backend.proactive_calls, [])
        self.assertEqual(print_backend.synthesize_calls, ["Brauchst du Hilfe?"])
        self.assertIn("social_prompt_mode=direct_safety", lines)
        social_events = [
            entry
            for entry in loop.runtime.ops_events.tail(limit=40)
            if entry["event"] == "social_trigger_prompted"
            and entry.get("data", {}).get("trigger") == "possible_fall"
        ]
        self.assertTrue(social_events)
        self.assertEqual(social_events[-1]["data"]["prompt"], "Brauchst du Hilfe?")
        self.assertEqual(social_events[-1]["data"]["prompt_mode"], "direct_safety")

    def test_social_trigger_uses_visual_first_delivery_when_background_media_is_active(self) -> None:
        config = TwinrConfig(proactive_quiet_hours_visual_only_enabled=False)
        proactive_monitor = SimpleNamespace(
            coordinator=SimpleNamespace(
                latest_audio_policy_snapshot=ReSpeakerAudioPolicySnapshot(
                    observed_at=42.0,
                    speech_delivery_defer_reason="background_media_active",
                    background_media_likely=True,
                ),
            )
        )
        loop, lines, _realtime_session, print_backend, _recorder, player, _printer = self.make_loop(
            config=config,
            proactive_monitor=proactive_monitor,
        )

        spoke = loop.handle_social_trigger(
            SocialTriggerDecision(
                trigger_id="attention_window",
                prompt="Soll ich dir helfen?",
                reason="User seems attentive and quiet.",
                observed_at=42.0,
                priority=SocialTriggerPriority.ATTENTION_WINDOW,
            )
        )

        presentation_path = Path(loop.config.display_presentation_path)
        payload = json.loads(presentation_path.read_text(encoding="utf-8"))

        self.assertTrue(spoke)
        self.assertEqual(print_backend.proactive_calls, [])
        self.assertEqual(player.played, [])
        self.assertIn("social_prompt_mode=display_first", lines)
        self.assertIn("social_display_reason=background_media_active", lines)
        self.assertEqual(payload["title"], "Soll ich dir helfen?")
        social_events = [
            entry
            for entry in loop.runtime.ops_events.tail(limit=20)
            if entry["event"] == "social_trigger_displayed"
            and entry.get("data", {}).get("trigger") == "attention_window"
        ]
        self.assertEqual(len(social_events), 1)
        self.assertEqual(social_events[0]["data"]["display_reason"], "background_media_active")

    def test_social_trigger_uses_visual_first_delivery_when_multimodal_initiative_is_not_ready(self) -> None:
        config = TwinrConfig(proactive_quiet_hours_visual_only_enabled=False)
        loop, lines, _realtime_session, print_backend, _recorder, player, _printer = self.make_loop(
            config=config,
        )
        loop._latest_sensor_observation_facts = {
            "multimodal_initiative": {
                "ready": False,
                "confidence": 0.59,
                "block_reason": "low_confidence_speaker_association",
                "recommended_channel": "display",
            },
        }

        spoke = loop.handle_social_trigger(
            SocialTriggerDecision(
                trigger_id="attention_window",
                prompt="Soll ich dir helfen?",
                reason="User seems attentive and quiet.",
                observed_at=42.0,
                priority=SocialTriggerPriority.ATTENTION_WINDOW,
            )
        )

        presentation_path = Path(loop.config.display_presentation_path)
        payload = json.loads(presentation_path.read_text(encoding="utf-8"))

        self.assertTrue(spoke)
        self.assertEqual(print_backend.proactive_calls, [])
        self.assertEqual(player.played, [])
        self.assertIn("social_prompt_mode=display_first", lines)
        self.assertIn("social_display_reason=low_confidence_speaker_association", lines)
        self.assertEqual(payload["title"], "Soll ich dir helfen?")
        social_events = [
            entry
            for entry in loop.runtime.ops_events.tail(limit=20)
            if entry["event"] == "social_trigger_displayed"
            and entry.get("data", {}).get("trigger") == "attention_window"
        ]
        self.assertEqual(len(social_events), 1)
        self.assertEqual(
            social_events[0]["data"]["display_reason"],
            "low_confidence_speaker_association",
        )

    def test_social_trigger_is_skipped_when_runtime_is_busy(self) -> None:
        loop, lines, _realtime_session, _print_backend, _recorder, player, _printer = self.make_loop()
        loop.runtime.press_green_button()

        spoke = loop.handle_social_trigger(
            SocialTriggerDecision(
                trigger_id="positive_contact",
                prompt="Schön, dich zu sehen. Was möchtest du machen?",
                reason="Visible smile.",
                observed_at=42.0,
                priority=SocialTriggerPriority.POSITIVE_CONTACT,
            )
        )

        self.assertFalse(spoke)
        self.assertEqual(player.played, [])
        self.assertIn("social_trigger_skipped=busy", lines)
        social_events = [entry for entry in loop.runtime.ops_events.tail(limit=20) if entry["event"] == "social_trigger_skipped"]
        self.assertEqual(len(social_events), 1)
        self.assertEqual(social_events[0]["data"]["prompt"], "Schön, dich zu sehen. Was möchtest du machen?")

    def test_social_trigger_is_skipped_while_follow_up_session_is_active(self) -> None:
        loop, lines, _realtime_session, _print_backend, _recorder, player, _printer = self.make_loop()
        loop._conversation_session_active = True

        spoke = loop.handle_social_trigger(
            SocialTriggerDecision(
                trigger_id="showing_intent",
                prompt="Möchtest du mir etwas zeigen?",
                reason="Object near the camera.",
                observed_at=42.0,
                priority=SocialTriggerPriority.SHOWING_INTENT,
            )
        )

        self.assertFalse(spoke)
        self.assertEqual(player.played, [])
        self.assertIn("social_trigger_skipped=conversation_active", lines)
        social_events = [entry for entry in loop.runtime.ops_events.tail(limit=20) if entry["event"] == "social_trigger_skipped"]
        self.assertGreaterEqual(len(social_events), 1)
        self.assertEqual(social_events[-1]["data"]["skip_reason"], "conversation_active")

    def test_run_opens_and_closes_proactive_monitor(self) -> None:
        button_monitor = FakeIdleButtonMonitor()
        proactive_monitor = FakeProactiveMonitor()
        loop, _lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
            button_monitor=button_monitor,
            proactive_monitor=proactive_monitor,
        )

        result = loop.run(duration_s=0.01, poll_timeout=0.001)

        self.assertEqual(result, 0)
        self.assertTrue(button_monitor.entered)
        self.assertTrue(button_monitor.exited)
        self.assertTrue(proactive_monitor.entered)
        self.assertTrue(proactive_monitor.exited)

    def test_run_holds_error_when_required_remote_dependency_is_unavailable(self) -> None:
        button_monitor = FakeIdleButtonMonitor()
        loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
            config=TwinrConfig(
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=True,
                long_term_memory_remote_keepalive_interval_s=0.01,
            ),
            button_monitor=button_monitor,
        )

        class _FailingRequiredRemote:
            def remote_required(self):
                return True

            def ensure_remote_ready(self):
                raise LongTermRemoteUnavailableError("remote unavailable")

            def remote_status(self):
                return LongTermRemoteStatus(
                    mode="remote_primary",
                    ready=False,
                    detail="remote unavailable",
                )

        loop.runtime.long_term_memory = _FailingRequiredRemote()

        result = loop.run(duration_s=0.03, poll_timeout=0.001)

        self.assertEqual(result, 0)
        self.assertEqual(loop.runtime.status.value, "error")
        self.assertIn("status=error", lines)
        self.assertNotIn("status=listening", lines)
        self.assertEqual(button_monitor.poll_calls, 0)

    def test_run_emits_required_remote_correlation_id_for_bulk_write_failures(self) -> None:
        button_monitor = FakeIdleButtonMonitor()
        loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
            config=TwinrConfig(
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=True,
                long_term_memory_remote_keepalive_interval_s=0.01,
            ),
            button_monitor=button_monitor,
        )
        loop.runtime.reset_error()
        exc = LongTermRemoteUnavailableError(
            "Failed to persist fine-grained remote long-term memory items "
            "(request_id=ltw-test123, batch=1/1, items=51, bytes=518642)."
        )
        setattr(
            exc,
            "remote_write_context",
            {
                "snapshot_kind": "objects",
                "operation": "store_records_bulk",
                "request_correlation_id": "ltw-test123",
                "batch_index": 1,
                "batch_count": 1,
                "request_item_count": 51,
                "request_bytes": 518642,
            },
        )

        entered = loop._enter_required_remote_error(exc)

        self.assertTrue(entered)
        self.assertEqual(loop.runtime.status.value, "error")
        self.assertIn("status=error", lines)
        self.assertIn("required_remote_correlation_id=ltw-test123", lines)

    def test_run_recovers_after_required_remote_dependency_returns(self) -> None:
        loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
            config=TwinrConfig(
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=True,
                long_term_memory_remote_keepalive_interval_s=0.01,
            ),
        )

        ready_after = time.monotonic() + 0.02

        class _RecoveringRequiredRemote:
            def remote_required(self):
                return True

            def ensure_remote_ready(self):
                if time.monotonic() < ready_after:
                    raise LongTermRemoteUnavailableError("remote unavailable")

            def remote_status(self):
                if time.monotonic() < ready_after:
                    return LongTermRemoteStatus(
                        mode="remote_primary",
                        ready=False,
                        detail="remote unavailable",
                    )
                return LongTermRemoteStatus(mode="remote_primary", ready=True)

        loop.runtime.long_term_memory = _RecoveringRequiredRemote()
        loop.runtime.reset_error()
        loop._enter_required_remote_error(LongTermRemoteUnavailableError("remote unavailable"))

        while time.monotonic() < ready_after:
            self.assertFalse(loop._refresh_required_remote_dependency(force=True, force_sync=True))
            time.sleep(0.005)
        refreshed = loop._refresh_required_remote_dependency(force=True, force_sync=True)

        self.assertTrue(refreshed)
        self.assertIn("status=error", lines)
        self.assertIn("status=waiting", lines)
        self.assertIn("required_remote_dependency_restored=true", lines)
        self.assertEqual(loop.runtime.status.value, "waiting")

    def test_run_does_not_restore_when_deep_remote_check_still_fails(self) -> None:
        button_monitor = FakeIdleButtonMonitor()
        loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
            config=TwinrConfig(
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=True,
                long_term_memory_remote_keepalive_interval_s=0.01,
            ),
            button_monitor=button_monitor,
        )

        class _FalsePositiveRecoveringRemote:
            def __init__(self) -> None:
                self.calls = 0

            def remote_required(self):
                return True

            def ensure_remote_ready(self):
                self.calls += 1
                raise LongTermRemoteUnavailableError("archive shard unavailable")

            def remote_status(self):
                return LongTermRemoteStatus(mode="remote_primary", ready=True)

        loop.runtime.long_term_memory = _FalsePositiveRecoveringRemote()

        result = loop.run(duration_s=0.05, poll_timeout=0.001)

        self.assertEqual(result, 0)
        self.assertEqual(loop.runtime.status.value, "error")
        self.assertIn("status=error", lines)
        self.assertNotIn("required_remote_dependency_restored=true", lines)

    def test_handle_error_keeps_runtime_in_error_for_respeaker_runtime_contract_blockers(self) -> None:
        loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
            config=TwinrConfig(
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=True,
                long_term_memory_remote_keepalive_interval_s=0.01,
            ),
        )
        loop.error_reset_seconds = 0.01
        loop._required_remote_dependency_error_active = True

        class _ReadyRequiredRemote:
            def remote_required(self):
                return True

            def ensure_remote_ready(self):
                return None

            def remote_status(self):
                return LongTermRemoteStatus(mode="remote_primary", ready=True)

        loop.runtime.long_term_memory = _ReadyRequiredRemote()

        loop._handle_error(
            ReSpeakerRuntimeContractError(
                "ReSpeaker XVF3800 is not visible to the Pi, so Twinr refuses to start proactive audio."
            )
        )
        refreshed = loop._refresh_required_remote_dependency(force=True, force_sync=True)

        self.assertTrue(refreshed)
        self.assertEqual(loop.runtime.status.value, "error")
        self.assertIn("status=error", lines)
        self.assertNotIn("status=waiting", lines)
        self.assertNotIn("required_remote_dependency_restored=true", lines)
        self.assertTrue(any(line.startswith("error=") for line in lines))

    def test_required_remote_restore_does_not_clear_foreign_runtime_error(self) -> None:
        loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
            config=TwinrConfig(
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=True,
                long_term_memory_remote_keepalive_interval_s=0.01,
            ),
        )

        class _ReadyRequiredRemote:
            def remote_required(self):
                return True

            def ensure_remote_ready(self):
                return None

            def remote_status(self):
                return LongTermRemoteStatus(mode="remote_primary", ready=True)

        loop.runtime.long_term_memory = _ReadyRequiredRemote()
        loop.runtime.fail("ReSpeaker XVF3800 is not visible to the Pi.")
        loop._required_remote_dependency_error_active = True
        loop._required_remote_dependency_error_message = "Required remote long-term memory is unavailable."

        refreshed = loop._refresh_required_remote_dependency(force=True, force_sync=True)

        self.assertTrue(refreshed)
        self.assertEqual(loop.runtime.status.value, "error")
        self.assertNotIn("required_remote_dependency_restored=true", lines)
        self.assertNotIn("status=waiting", lines)

    def test_run_polls_buttons_while_required_remote_watch_blocks(self) -> None:
        button_monitor = FakeScheduledButtonMonitor(event_after_s=0.02, name="yellow")
        loop, lines, _realtime_session, _print_backend, _recorder, _player, printer = self.make_loop(
            config=TwinrConfig(
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=True,
                long_term_memory_remote_keepalive_interval_s=0.01,
            ),
            button_monitor=button_monitor,
        )
        loop.runtime.last_response = "Bitte druck das."
        loop.runtime.reset_error()
        remote_probe_started = Event()

        class _SlowReadyRemote:
            def remote_required(self):
                return True

            def ensure_remote_ready(self):
                remote_probe_started.set()
                time.sleep(0.2)

            def remote_status(self):
                return LongTermRemoteStatus(mode="remote_primary", ready=True)

        loop.runtime.long_term_memory = _SlowReadyRemote()

        result = loop.run(duration_s=0.12, poll_timeout=0.001)

        self.assertEqual(result, 0)
        self.assertTrue(remote_probe_started.is_set())
        self.assertEqual(len(printer.printed), 1)
        self.assertIn("button=yellow", lines)

    def test_run_uses_external_watchdog_artifact_instead_of_deep_remote_check(self) -> None:
        button_monitor = FakeScheduledButtonMonitor(event_after_s=0.02, name="yellow")
        startup_ready_assessment = SimpleNamespace(
            artifact_path="/tmp/remote_memory_watchdog.json",
            sample_age_s=0.0,
            max_sample_age_s=45.0,
            pid_alive=True,
        )
        with mock.patch(
            "twinr.agent.workflows.required_remote_snapshot.ensure_required_remote_watchdog_snapshot_ready",
            return_value=startup_ready_assessment,
        ):
            loop, lines, _realtime_session, _print_backend, _recorder, _player, printer = self.make_loop(
                config=TwinrConfig(
                    long_term_memory_enabled=True,
                    long_term_memory_mode="remote_primary",
                    long_term_memory_remote_required=True,
                    long_term_memory_remote_runtime_check_mode="watchdog_artifact",
                    long_term_memory_remote_watchdog_interval_s=0.01,
                ),
                button_monitor=button_monitor,
            )
        loop.runtime.last_response = "Bitte druck das."

        class _ShouldNotBeCalledRemote:
            def __init__(self) -> None:
                self.calls = 0

            def remote_required(self):
                return True

            def ensure_remote_ready(self):
                self.calls += 1
                raise AssertionError("deep remote check must not run in watchdog_artifact mode")

            def remote_status(self):
                return LongTermRemoteStatus(mode="remote_primary", ready=True)

        remote = _ShouldNotBeCalledRemote()
        loop.runtime.long_term_memory = remote

        with mock.patch(
            "twinr.agent.workflows.realtime_runtime.support.ensure_required_remote_watchdog_snapshot_ready",
            return_value=SimpleNamespace(
                artifact_path="/tmp/remote_memory_watchdog.json",
                sample_age_s=1.0,
                max_sample_age_s=45.0,
                pid_alive=True,
            ),
        ):
            result = loop.run(duration_s=0.12, poll_timeout=0.001)

        self.assertEqual(result, 0)
        self.assertEqual(remote.calls, 0)
        self.assertEqual(len(printer.printed), 1)
        self.assertIn("button=yellow", lines)

    def test_run_enters_error_when_external_watchdog_artifact_reports_failure(self) -> None:
        button_monitor = FakeIdleButtonMonitor()
        startup_ready_assessment = SimpleNamespace(
            artifact_path="/tmp/remote_memory_watchdog.json",
            sample_age_s=0.0,
            max_sample_age_s=45.0,
            pid_alive=True,
        )
        with mock.patch(
            "twinr.agent.workflows.required_remote_snapshot.ensure_required_remote_watchdog_snapshot_ready",
            return_value=startup_ready_assessment,
        ):
            loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
                config=TwinrConfig(
                    long_term_memory_enabled=True,
                    long_term_memory_mode="remote_primary",
                    long_term_memory_remote_required=True,
                    long_term_memory_remote_runtime_check_mode="watchdog_artifact",
                    long_term_memory_remote_watchdog_interval_s=0.01,
                ),
                button_monitor=button_monitor,
            )

        class _RemoteStillRequired:
            def remote_required(self):
                return True

            def ensure_remote_ready(self):
                raise AssertionError("deep remote check must not run in watchdog_artifact mode")

            def remote_status(self):
                return LongTermRemoteStatus(mode="remote_primary", ready=True)

        loop.runtime.long_term_memory = _RemoteStillRequired()

        failing_assessment = SimpleNamespace(
            artifact_path="/tmp/remote_memory_watchdog.json",
            detail="Remote memory watchdog snapshot is stale.",
            sample_age_s=91.0,
            max_sample_age_s=45.0,
            pid_alive=False,
            sample_status="fail",
            sample_ready=False,
        )

        with mock.patch(
            "twinr.agent.workflows.realtime_runtime.support.ensure_required_remote_watchdog_snapshot_ready",
            side_effect=LongTermRemoteUnavailableError(failing_assessment.detail),
        ):
            with mock.patch(
                "twinr.agent.workflows.realtime_runtime.support.assess_required_remote_watchdog_snapshot",
                return_value=failing_assessment,
            ):
                result = loop.run(duration_s=0.05, poll_timeout=0.001)

        self.assertEqual(result, 0)
        self.assertEqual(loop.runtime.status.value, "error")
        self.assertIn("status=error", lines)
        self.assertEqual(button_monitor.poll_calls, 0)

    def test_run_watchdog_artifact_does_not_restore_on_transient_ready_blip(self) -> None:
        button_monitor = FakeIdleButtonMonitor()
        startup_ready_assessment = SimpleNamespace(
            artifact_path="/tmp/remote_memory_watchdog.json",
            sample_age_s=0.0,
            max_sample_age_s=45.0,
            pid_alive=True,
        )
        with mock.patch(
            "twinr.agent.workflows.required_remote_snapshot.ensure_required_remote_watchdog_snapshot_ready",
            return_value=startup_ready_assessment,
        ):
            loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
                config=TwinrConfig(
                    long_term_memory_enabled=True,
                    long_term_memory_mode="remote_primary",
                    long_term_memory_remote_required=True,
                    long_term_memory_remote_runtime_check_mode="watchdog_artifact",
                    long_term_memory_remote_watchdog_interval_s=0.01,
                    long_term_memory_remote_keepalive_interval_s=0.05,
                ),
                button_monitor=button_monitor,
            )

        class _RemoteStillRequired:
            def remote_required(self):
                return True

            def ensure_remote_ready(self):
                raise AssertionError("deep remote check must not run in watchdog_artifact mode")

            def remote_status(self):
                return LongTermRemoteStatus(mode="remote_primary", ready=True)

        loop.runtime.long_term_memory = _RemoteStillRequired()
        started = time.monotonic()
        ready_assessment = SimpleNamespace(
            artifact_path="/tmp/remote_memory_watchdog.json",
            sample_age_s=1.0,
            max_sample_age_s=45.0,
            pid_alive=True,
        )
        failing_assessment = SimpleNamespace(
            artifact_path="/tmp/remote_memory_watchdog.json",
            detail="Remote memory watchdog snapshot is stale.",
            sample_age_s=91.0,
            max_sample_age_s=45.0,
            pid_alive=False,
            sample_status="fail",
            sample_ready=False,
        )

        def _ensure_watchdog_ready(*_args, **_kwargs):
            elapsed = time.monotonic() - started
            if elapsed < 0.02:
                raise LongTermRemoteUnavailableError(failing_assessment.detail)
            if elapsed < 0.045:
                return ready_assessment
            raise LongTermRemoteUnavailableError(failing_assessment.detail)

        with mock.patch(
            "twinr.agent.workflows.realtime_runtime.support.ensure_required_remote_watchdog_snapshot_ready",
            side_effect=_ensure_watchdog_ready,
        ):
            with mock.patch(
                "twinr.agent.workflows.realtime_runtime.support.assess_required_remote_watchdog_snapshot",
                return_value=failing_assessment,
            ):
                result = loop.run(duration_s=0.12, poll_timeout=0.001)

        self.assertEqual(result, 0)
        self.assertEqual(loop.runtime.status.value, "error")
        self.assertIn("status=error", lines)
        self.assertNotIn("required_remote_dependency_restored=true", lines)

    def test_run_watchdog_artifact_restores_after_stable_ready_window(self) -> None:
        button_monitor = FakeIdleButtonMonitor()
        startup_ready_assessment = SimpleNamespace(
            artifact_path="/tmp/remote_memory_watchdog.json",
            sample_age_s=0.0,
            max_sample_age_s=45.0,
            pid_alive=True,
        )
        with mock.patch(
            "twinr.agent.workflows.required_remote_snapshot.ensure_required_remote_watchdog_snapshot_ready",
            return_value=startup_ready_assessment,
        ):
            loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
                config=TwinrConfig(
                    long_term_memory_enabled=True,
                    long_term_memory_mode="remote_primary",
                    long_term_memory_remote_required=True,
                    long_term_memory_remote_runtime_check_mode="watchdog_artifact",
                    long_term_memory_remote_watchdog_interval_s=0.01,
                    long_term_memory_remote_keepalive_interval_s=0.04,
                ),
                button_monitor=button_monitor,
            )

        class _RemoteStillRequired:
            def remote_required(self):
                return True

            def ensure_remote_ready(self):
                raise AssertionError("deep remote check must not run in watchdog_artifact mode")

            def remote_status(self):
                return LongTermRemoteStatus(mode="remote_primary", ready=True)

        loop.runtime.long_term_memory = _RemoteStillRequired()
        ready_assessment = SimpleNamespace(
            artifact_path="/tmp/remote_memory_watchdog.json",
            sample_age_s=1.0,
            max_sample_age_s=45.0,
            pid_alive=True,
        )
        failing_assessment = SimpleNamespace(
            artifact_path="/tmp/remote_memory_watchdog.json",
            detail="Remote memory watchdog snapshot is stale.",
            sample_age_s=91.0,
            max_sample_age_s=45.0,
            pid_alive=False,
            sample_status="fail",
            sample_ready=False,
        )

        loop._enter_required_remote_error(LongTermRemoteUnavailableError(failing_assessment.detail))

        with mock.patch(
            "twinr.agent.workflows.realtime_runtime.support.ensure_required_remote_watchdog_snapshot_ready",
            return_value=ready_assessment,
        ):
            first_refresh = loop._refresh_required_remote_dependency(force=True)
            time.sleep(0.32)
            second_refresh = loop._refresh_required_remote_dependency(force=True)

        self.assertFalse(first_refresh)
        self.assertTrue(second_refresh)
        self.assertIn("status=error", lines)
        self.assertIn("status=waiting", lines)
        self.assertIn("required_remote_dependency_restored=true", lines)
        self.assertEqual(loop.runtime.status.value, "waiting")

    def test_run_handles_button_press_while_housekeeping_blocks(self) -> None:
        button_monitor = FakeScheduledButtonMonitor(event_after_s=0.02, name="yellow")
        loop, lines, _realtime_session, _print_backend, _recorder, _player, printer = self.make_loop(
            button_monitor=button_monitor,
        )
        loop.runtime.last_response = "Bitte druck das."

        def slow_housekeeping() -> bool:
            time.sleep(0.2)
            return False

        loop._maybe_deliver_due_reminder = slow_housekeeping
        loop._maybe_run_due_automation = lambda: False
        loop._maybe_run_sensor_automation = lambda: False
        loop._maybe_run_long_term_memory_proactive = lambda: False

        result = loop.run(duration_s=0.12, poll_timeout=0.001)

        self.assertEqual(result, 0)
        self.assertEqual(len(printer.printed), 1)
        self.assertIn("button=yellow", lines)

    def test_button_dispatcher_interrupts_busy_green_turn(self) -> None:
        handled: list[str] = []
        interrupted: list[str] = []
        allow_first_turn_to_finish = Event()

        def handle_press(button_name: str) -> None:
            handled.append(button_name)
            if len(handled) == 1:
                allow_first_turn_to_finish.wait(timeout=1.0)

        dispatcher = ButtonPressDispatcher(
            handle_press=handle_press,
            interrupt_current=lambda source: interrupted.append(source) or allow_first_turn_to_finish.set() or True,
        )
        dispatcher.submit("green")
        time.sleep(0.02)
        dispatcher.submit("green")
        time.sleep(0.05)
        dispatcher.close(timeout_s=1.0)

        self.assertEqual(interrupted, ["green"])
        self.assertEqual(handled, ["green", "green"])

    def test_run_interrupts_busy_turn_on_second_green_press(self) -> None:
        button_monitor = FakeBurstButtonMonitor(events=[(0.0, "green"), (0.02, "green")])
        loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
            button_monitor=button_monitor,
        )
        started = Event()
        released = Event()

        def fake_green_turn() -> None:
            stop_event = Event()
            loop._set_active_turn_stop_event(stop_event)
            started.set()
            try:
                while not loop._active_turn_stop_requested():
                    time.sleep(0.005)
                loop.runtime.cancel_listening()
                loop.emit("fake_turn_interrupted=true")
                released.set()
            finally:
                loop._clear_active_turn_stop_event(stop_event)

        loop._handle_green_turn = fake_green_turn  # type: ignore[method-assign]

        result = loop.run(duration_s=0.12, poll_timeout=0.001)

        self.assertEqual(result, 0)
        self.assertTrue(started.is_set())
        self.assertTrue(released.is_set())
        self.assertIn("button=green", lines)
        self.assertIn("turn_interrupt_requested=green", lines)

    def test_interrupt_requests_stop_active_player(self) -> None:
        loop, _lines, _realtime_session, _print_backend, _recorder, player, _printer = self.make_loop()
        stop_event = Event()
        loop._set_active_turn_stop_event(stop_event)

        try:
            self.assertTrue(loop._request_active_turn_interrupt())
        finally:
            loop._clear_active_turn_stop_event(stop_event)

        self.assertEqual(player.stop_calls, 1)


if __name__ == "__main__":
    unittest.main()
