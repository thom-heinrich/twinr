import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from types import SimpleNamespace
from typing import Callable
import sys
import time
import unittest
import gc
from datetime import datetime
from unittest import mock
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.contracts import (
    AgentToolCall,
    FirstWordReply,
    StreamingTranscriptionResult,
    ToolCallingTurnResponse,
)
from twinr.agent.base_agent.conversation.closure import ConversationClosureDecision
from twinr.agent.tools.runtime.dual_lane_loop import DualLaneToolLoop
from twinr.agent.workflows.realtime_runner import TwinrRealtimeHardwareLoop
from twinr.agent.workflows.streaming_runner import TwinrStreamingHardwareLoop
from twinr.agent.workflows.streaming_semantic_router import _synthesize_supervisor_decision
from twinr.agent.workflows.streaming_turn_coordinator import (
    StreamingTurnCoordinator,
    StreamingTurnCoordinatorHooks,
    StreamingTurnLanePlan,
    StreamingTurnRequest,
    StreamingTurnSpeechServices,
    _SpeechLifecycle,
)
from twinr.agent.workflows.streaming_turn_orchestrator import FinalLaneTimeoutError
from twinr.agent.base_agent import TwinrConfig
from twinr.hardware.audio import ListenTimeoutCaptureDiagnostics, SpeechStartTimeoutError
from twinr.memory.longterm.core.models import (
    LongTermConsolidationResultV1,
    LongTermMemoryConflictV1,
    LongTermMemoryObjectV1,
    LongTermSourceRefV1,
)
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError
from twinr.providers.openai import (
    OpenAIBackend,
    OpenAIFirstWordProvider,
    OpenAIToolCallingAgentProvider,
    OpenAITextResponse,
)
from twinr.proactive.runtime.gesture_wakeup_lane import GestureWakeupDecision
from twinr.agent.base_agent import TwinrRuntime
from twinr.agent.base_agent import TwinrEvent, TwinrStatus

_TEST_CONTACT_PHONE = "555-0100"
_TEST_CORINNA_PHONE_OLD = "+15555551234"
_TEST_CORINNA_PHONE_NEW = "+15555558877"


class FakeToolAgentProvider:
    def __init__(self, config: TwinrConfig) -> None:
        self.config = config
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
        self.start_calls.append(
            {
                "prompt": prompt,
                "conversation": conversation,
                "instructions": instructions,
                "tool_schemas": list(tool_schemas),
                "allow_web_search": allow_web_search,
            }
        )
        if on_text_delta is not None:
            on_text_delta("Ich drucke das.")
        return ToolCallingTurnResponse(
            text="Ich drucke das.",
            tool_calls=(
                AgentToolCall(
                    name="print_receipt",
                    call_id="call_print_1",
                    arguments={"text": "Termine"},
                    raw_arguments='{"text":"Termine"}',
                ),
            ),
            response_id="resp_start_1",
            continuation_token="resp_start_1",
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
        self.continue_calls.append(
            {
                "continuation_token": continuation_token,
                "tool_results": list(tool_results),
                "instructions": instructions,
                "tool_schemas": list(tool_schemas),
                "allow_web_search": allow_web_search,
            }
        )
        if on_text_delta is not None:
            on_text_delta(" Ist erledigt.")
        return ToolCallingTurnResponse(
            text="Ist erledigt.",
            response_id="resp_done_1",
        )


class FakePrintBackend:
    def __init__(self, config: TwinrConfig) -> None:
        self.config = config
        self.print_calls: list[tuple[str | None, str | None, str]] = []

    def compose_print_job_with_metadata(
        self,
        *,
        conversation=None,
        focus_hint: str | None = None,
        direct_text: str | None = None,
        request_source: str = "button",
    ) -> OpenAITextResponse:
        del conversation
        self.print_calls.append((focus_hint, direct_text, request_source))
        return OpenAITextResponse(text="AUSDRUCK")

    def phrase_due_reminder_with_metadata(self, reminder, *, now=None) -> OpenAITextResponse:
        del reminder, now
        return OpenAITextResponse(text="Erinnerung")

    def phrase_proactive_prompt_with_metadata(self, **kwargs) -> OpenAITextResponse:
        del kwargs
        return OpenAITextResponse(text="Proaktiv")

    def search_live_info_with_metadata(self, question: str, **kwargs):
        del question, kwargs
        return SimpleNamespace(
            answer="Antwort",
            sources=(),
            response_id="resp_search",
            request_id="req_search",
            model="gpt-5.2",
            token_usage=None,
            used_web_search=True,
        )

    def respond_to_images_with_metadata(self, prompt: str, **kwargs) -> OpenAITextResponse:
        del prompt, kwargs
        return OpenAITextResponse(text="Kamera")

    def fulfill_automation_prompt_with_metadata(self, prompt: str, **kwargs) -> OpenAITextResponse:
        del prompt, kwargs
        return OpenAITextResponse(text="Automation")


class FakeSpeechToTextProvider:
    def __init__(self, config: TwinrConfig) -> None:
        self.config = config

    def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
        del audio_bytes, kwargs
        return "Hallo Twinr"

    def transcribe_path(self, path, **kwargs) -> str:
        del path, kwargs
        return "Hallo Twinr"


class FakeStreamingSpeechSession:
    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self.closed = False
        self.finalize_calls = 0

    def send_pcm(self, pcm_bytes: bytes) -> None:
        self.sent.append(pcm_bytes)

    def snapshot(self) -> StreamingTranscriptionResult:
        return StreamingTranscriptionResult(
            transcript="Streaming Hallo Twinr",
            request_id="dg-stream-1",
            saw_interim=True,
            saw_speech_final=True,
            saw_utterance_end=False,
        )

    def finalize(self) -> StreamingTranscriptionResult:
        self.finalize_calls += 1
        return StreamingTranscriptionResult(
            transcript="Streaming Hallo Twinr",
            request_id="dg-stream-1",
            saw_interim=True,
            saw_speech_final=True,
            saw_utterance_end=False,
        )

    def close(self) -> None:
        self.closed = True


class _FakeSnapshotRefreshRuntime:
    def __init__(self) -> None:
        self.status = SimpleNamespace(value="processing")
        self.begin_answering_calls = 0
        self.resume_processing_calls = 0
        self.refresh_snapshot_activity_calls = 0

    def begin_answering(self) -> None:
        self.begin_answering_calls += 1
        self.status.value = "answering"

    def resume_processing(self) -> None:
        self.resume_processing_calls += 1
        self.status.value = "processing"

    def refresh_snapshot_activity(self) -> None:
        self.refresh_snapshot_activity_calls += 1


class DivergentSpeechFinalStreamingSpeechSession(FakeStreamingSpeechSession):
    def snapshot(self) -> StreamingTranscriptionResult:
        return StreamingTranscriptionResult(
            transcript="Sind die neuesten Nachrichten.",
            request_id="dg-stream-divergent",
            saw_interim=True,
            saw_speech_final=True,
            saw_utterance_end=False,
        )

    def finalize(self) -> StreamingTranscriptionResult:
        self.finalize_calls += 1
        return StreamingTranscriptionResult(
            transcript="Was sind die neuesten Nachrichten?",
            request_id="dg-stream-divergent",
            saw_interim=True,
            saw_speech_final=True,
            saw_utterance_end=False,
        )


class BareSpeechFinalStreamingSpeechSession(FakeStreamingSpeechSession):
    def snapshot(self) -> StreamingTranscriptionResult:
        return StreamingTranscriptionResult(
            transcript="Bitte gut.",
            request_id="dg-stream-2",
            saw_interim=False,
            saw_speech_final=True,
            saw_utterance_end=False,
        )

    def finalize(self) -> StreamingTranscriptionResult:
        self.finalize_calls += 1
        return StreamingTranscriptionResult(
            transcript="Gut.",
            request_id="dg-stream-2",
            saw_interim=False,
            saw_speech_final=True,
            saw_utterance_end=False,
        )


class FakeStreamingSpeechToTextProvider(FakeSpeechToTextProvider):
    def __init__(self, config: TwinrConfig) -> None:
        super().__init__(config)
        self.session = FakeStreamingSpeechSession()
        self.start_calls: list[dict[str, object]] = []
        self.interim_callback = None

    def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
        raise AssertionError("streaming STT path should not fall back to file transcription")

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
        del prompt, on_endpoint
        self.start_calls.append(
            {
                "sample_rate": sample_rate,
                "channels": channels,
                "language": language,
            }
        )
        self.interim_callback = on_interim
        if on_interim is not None:
            on_interim("Stream partiell")
        return self.session


class BareSpeechFinalStreamingSpeechToTextProvider(FakeStreamingSpeechToTextProvider):
    def __init__(self, config: TwinrConfig) -> None:
        super().__init__(config)
        self.session = BareSpeechFinalStreamingSpeechSession()
        self.transcribe_calls: list[dict[str, object]] = []

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
        del prompt, on_interim, on_endpoint
        self.start_calls.append(
            {
                "sample_rate": sample_rate,
                "channels": channels,
                "language": language,
            }
        )
        return self.session

    def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
        self.transcribe_calls.append(
            {
                "audio_len": len(audio_bytes),
                **kwargs,
            }
        )
        return "Geht's dir heute gut?"


class DivergentSpeechFinalStreamingSpeechToTextProvider(FakeStreamingSpeechToTextProvider):
    def __init__(self, config: TwinrConfig) -> None:
        super().__init__(config)
        self.session = DivergentSpeechFinalStreamingSpeechSession()

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
        del prompt, on_endpoint
        self.start_calls.append(
            {
                "sample_rate": sample_rate,
                "channels": channels,
                "language": language,
            }
        )
        if on_interim is not None:
            on_interim("Was sind die neuesten Nachrichten?")
        return self.session


class EmptySpeechFinalStreamingSpeechSession(FakeStreamingSpeechSession):
    def snapshot(self) -> StreamingTranscriptionResult:
        return StreamingTranscriptionResult(
            transcript="",
            request_id="dg-stream-5",
            saw_interim=False,
            saw_speech_final=True,
            saw_utterance_end=False,
        )

    def finalize(self) -> StreamingTranscriptionResult:
        self.finalize_calls += 1
        return StreamingTranscriptionResult(
            transcript="",
            request_id="dg-stream-5",
            saw_interim=False,
            saw_speech_final=True,
            saw_utterance_end=False,
        )


class EmptySpeechFinalStreamingSpeechToTextProvider(FakeStreamingSpeechToTextProvider):
    def __init__(self, config: TwinrConfig) -> None:
        super().__init__(config)
        self.session = EmptySpeechFinalStreamingSpeechSession()
        self.transcribe_calls: list[dict[str, object]] = []

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
        del prompt, on_interim, on_endpoint
        self.start_calls.append(
            {
                "sample_rate": sample_rate,
                "channels": channels,
                "language": language,
            }
        )
        return self.session

    def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
        self.transcribe_calls.append(
            {
                "audio_len": len(audio_bytes),
                **kwargs,
            }
        )
        return "Wie geht es dir heute?"


class ShortInterimStreamingSpeechSession(FakeStreamingSpeechSession):
    def snapshot(self) -> StreamingTranscriptionResult:
        return StreamingTranscriptionResult(
            transcript="Gut?",
            request_id="dg-stream-3",
            saw_interim=True,
            saw_speech_final=True,
            saw_utterance_end=False,
        )

    def finalize(self) -> StreamingTranscriptionResult:
        self.finalize_calls += 1
        return StreamingTranscriptionResult(
            transcript="Gut?",
            request_id="dg-stream-3",
            saw_interim=True,
            saw_speech_final=True,
            saw_utterance_end=False,
        )


class ShortInterimStreamingSpeechToTextProvider(FakeStreamingSpeechToTextProvider):
    def __init__(self, config: TwinrConfig) -> None:
        super().__init__(config)
        self.session = ShortInterimStreamingSpeechSession()
        self.transcribe_calls: list[dict[str, object]] = []

    def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
        self.transcribe_calls.append(
            {
                "audio_len": len(audio_bytes),
                **kwargs,
            }
        )
        return "Geht's dir heute gut?"


class UtteranceEndOnlyStreamingSpeechSession(FakeStreamingSpeechSession):
    def snapshot(self) -> StreamingTranscriptionResult:
        return StreamingTranscriptionResult(
            transcript="Heute gut, alles okay?",
            request_id="dg-stream-4",
            saw_interim=True,
            saw_speech_final=False,
            saw_utterance_end=True,
            confidence=0.74,
        )

    def finalize(self) -> StreamingTranscriptionResult:
        self.finalize_calls += 1
        return StreamingTranscriptionResult(
            transcript="Heute gut, alles okay?",
            request_id="dg-stream-4",
            saw_interim=True,
            saw_speech_final=False,
            saw_utterance_end=True,
            confidence=0.74,
        )


class UtteranceEndOnlyStreamingSpeechToTextProvider(FakeStreamingSpeechToTextProvider):
    def __init__(self, config: TwinrConfig) -> None:
        super().__init__(config)
        self.session = UtteranceEndOnlyStreamingSpeechSession()
        self.transcribe_calls: list[dict[str, object]] = []

    def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
        self.transcribe_calls.append({"audio_len": len(audio_bytes), **kwargs})
        return "Heute gut, alles okay?"


class FakeVerifierSpeechToTextProvider(FakeSpeechToTextProvider):
    def __init__(self, config: TwinrConfig, *, transcript: str) -> None:
        super().__init__(config)
        self.transcript_value = transcript
        self.calls: list[dict[str, object]] = []

    def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
        self.calls.append({"audio_len": len(audio_bytes), **kwargs})
        return self.transcript_value


class FakeFirstWordProvider:
    def __init__(self, config: TwinrConfig, *, reply: FirstWordReply | None = None) -> None:
        self.config = config
        self.reply_value = reply or FirstWordReply(mode="filler", spoken_text="Ich schaue kurz nach.")
        self.calls: list[dict[str, object]] = []

    def reply(self, prompt: str, *, conversation=None, instructions=None) -> FirstWordReply:
        self.calls.append(
            {
                "prompt": prompt,
                "conversation": conversation,
                "instructions": instructions,
            }
        )
        return self.reply_value


class DelayedFirstWordProvider(FakeFirstWordProvider):
    def __init__(
        self,
        config: TwinrConfig,
        *,
        delay_s: float,
        reply: FirstWordReply | None = None,
        trace: list[str] | None = None,
    ) -> None:
        super().__init__(config, reply=reply)
        self.delay_s = delay_s
        self.trace = trace

    def reply(self, prompt: str, *, conversation=None, instructions=None) -> FirstWordReply:
        if self.trace is not None:
            self.trace.append("first_word_start")
        time.sleep(self.delay_s)
        result = super().reply(
            prompt,
            conversation=conversation,
            instructions=instructions,
        )
        if self.trace is not None:
            self.trace.append("first_word_end")
        return result


class ExplodingFirstWordProvider(FakeFirstWordProvider):
    def reply(self, prompt: str, *, conversation=None, instructions=None) -> FirstWordReply:
        raise AssertionError(
            f"first-word provider must not be called in supervisor-bridge mode: {prompt!r}"
        )


class FakeSupervisorDecisionProvider:
    def __init__(self, decision) -> None:
        self.decision = decision
        self.calls: list[dict[str, object]] = []

    def decide(self, prompt: str, *, conversation=None, instructions=None):
        self.calls.append(
            {
                "prompt": prompt,
                "conversation": conversation,
                "instructions": instructions,
            }
        )
        return self.decision


class FakeTextToSpeechProvider:
    def __init__(self, config: TwinrConfig) -> None:
        self.config = config
        self.calls: list[str] = []
        self.stream_calls: list[str] = []
        self.stream_chunk_sizes: list[int] = []

    def synthesize(self, text: str, **kwargs) -> bytes:
        del kwargs
        self.calls.append(text)
        return b"RIFF"

    def synthesize_stream(self, text: str, **kwargs):
        self.stream_chunk_sizes.append(int(kwargs.get("chunk_size", 4096)))
        self.calls.append(text)
        self.stream_calls.append(text)
        yield b"RI"
        yield b"FF"


class TraceTextToSpeechProvider(FakeTextToSpeechProvider):
    def __init__(self, config: TwinrConfig, trace: list[str]) -> None:
        super().__init__(config)
        self.trace = trace

    def synthesize_stream(self, text: str, **kwargs):
        self.trace.append(f"tts_start:{text}")
        yield from super().synthesize_stream(text, **kwargs)


class BlockingTextToSpeechProvider(FakeTextToSpeechProvider):
    def __init__(self, config: TwinrConfig) -> None:
        super().__init__(config)
        self.started = Event()
        self.release = Event()

    def synthesize_stream(self, text: str, **kwargs):
        self.stream_chunk_sizes.append(int(kwargs.get("chunk_size", 4096)))
        self.calls.append(text)
        self.stream_calls.append(text)
        self.started.set()
        self.release.wait(timeout=5.0)
        if not self.release.is_set():
            return
        yield b"RI"
        yield b"FF"


class FakePlayer:
    def __init__(self) -> None:
        self.played: list[bytes] = []
        self.played_wav_bytes: list[bytes] = []

    def play_wav_chunks(self, chunks, *, should_stop=None) -> None:
        payload = bytearray()
        for chunk in chunks:
            payload.extend(chunk)
            if should_stop is not None and should_stop():
                break
        self.played.append(bytes(payload))

    def play_wav_bytes(self, audio_bytes: bytes) -> None:
        self.played_wav_bytes.append(audio_bytes)

    def play_tone(self, **kwargs) -> None:
        del kwargs


class TimedPlayer(FakePlayer):
    def __init__(self, *, playback_delay_s: float) -> None:
        super().__init__()
        self.playback_delay_s = playback_delay_s
        self.playback_started_at: float | None = None
        self.playback_finished_at: float | None = None
        self.playback_started = Event()
        self.playback_finished = Event()

    def play_wav_chunks(self, chunks, *, should_stop=None) -> None:
        self.playback_finished.clear()
        self.playback_started_at = time.monotonic()
        self.playback_started.set()
        super().play_wav_chunks(chunks, should_stop=should_stop)
        time.sleep(self.playback_delay_s)
        self.playback_finished_at = time.monotonic()
        self.playback_finished.set()


class BlockingFirstPlaybackPlayer(FakePlayer):
    def __init__(self, *, release_first_playback: Event) -> None:
        super().__init__()
        self.release_first_playback = release_first_playback
        self.first_playback_started = Event()
        self.play_calls = 0

    def play_wav_chunks(self, chunks, *, should_stop=None) -> None:
        self.play_calls += 1
        super().play_wav_chunks(chunks, should_stop=should_stop)
        if self.play_calls != 1:
            return
        self.first_playback_started.set()
        self.release_first_playback.wait(timeout=1.0)


class FakePrinter:
    def __init__(self) -> None:
        self.printed: list[str] = []

    def print_text(self, text: str) -> str:
        self.printed.append(text)
        return "job-1"


class FakeVoiceOrchestrator:
    def __init__(self) -> None:
        self.states: list[tuple[str, str | None, bool]] = []
        self.paused: list[str] = []
        self.resumed: list[str] = []

    def notify_runtime_state(self, *, state: str, detail: str | None = None, follow_up_allowed: bool = False, **_kwargs) -> None:
        self.states.append((state, detail, follow_up_allowed))

    def pause_capture(self, *, reason: str) -> None:
        self.paused.append(reason)

    def resume_capture(self, *, reason: str) -> None:
        self.resumed.append(reason)


class FakeVoiceProfileMonitor:
    def summary(self):
        return SimpleNamespace(enrolled=False, sample_count=0, updated_at=None, average_duration_ms=None)

    def assess_pcm16(self, *args, **kwargs):
        del args, kwargs
        return SimpleNamespace(should_persist=False, status=None, confidence=None, checked_at=None)


class FakeUsageStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def append(self, **kwargs) -> None:
        self.calls.append(kwargs)


class FakeRecorder:
    def __init__(self) -> None:
        self.capture_calls: list[dict[str, object]] = []

    def capture_pcm_until_pause_with_options(self, **kwargs):
        self.capture_calls.append(dict(kwargs))
        on_chunk = kwargs.get("on_chunk")
        if on_chunk is not None:
            on_chunk(b"PCM-A")
            on_chunk(b"PCM-B")
        return SimpleNamespace(
            pcm_bytes=b"PCM-AB",
            speech_started_after_ms=120,
            resumed_after_pause_count=1,
        )


class BlockingFallbackRecorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.started = Event()

    def capture_pcm_until_pause_with_options(self, **kwargs):
        self.calls.append(dict(kwargs))
        self.started.set()
        should_stop = kwargs.get("should_stop")
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if callable(should_stop) and should_stop():
                raise RuntimeError("Audio capture stopped before speech started")
            time.sleep(0.01)
        raise AssertionError("Fallback capture did not observe an interrupt request")


class FailIfCalledRecorder:
    def capture_pcm_until_pause_with_options(self, **kwargs):
        raise AssertionError(f"Fallback recorder should not have been used: {kwargs}")


class CapturingDualLaneLoop(DualLaneToolLoop):
    def __init__(self) -> None:
        self.run_calls: list[dict[str, object]] = []
        self.run_handoff_calls: list[dict[str, object]] = []
        self.run_runtime_local_tool_only_calls: list[dict[str, object]] = []
        self.recovery_calls: list[dict[str, object]] = []
        self.recovery_text = "Ich fasse das gerade passend fuer dich zusammen."
        self.supervisor_instructions = ""
        self._trace_event_callback = None
        self._trace_decision_callback = None

    def run(self, prompt: str, **kwargs):
        self.run_calls.append({"prompt": prompt, **kwargs})
        prefetched_decision = kwargs.get("prefetched_decision")
        if prefetched_decision is not None:
            action = str(getattr(prefetched_decision, "action", "") or "").strip().lower()
            if action == "direct" and str(getattr(prefetched_decision, "context_scope", "") or "").strip().lower() != "full_context":
                return SimpleNamespace(
                    text=getattr(prefetched_decision, "spoken_reply", ""),
                    response_id=getattr(prefetched_decision, "response_id", None),
                    request_id=getattr(prefetched_decision, "request_id", None),
                    rounds=1,
                    tool_calls=(),
                    used_web_search=False,
                    model=getattr(prefetched_decision, "model", None),
                    token_usage=getattr(prefetched_decision, "token_usage", None),
                )
            if action == "end_conversation":
                return SimpleNamespace(
                    text=getattr(prefetched_decision, "spoken_reply", ""),
                    response_id=getattr(prefetched_decision, "response_id", None),
                    request_id=getattr(prefetched_decision, "request_id", None),
                    rounds=1,
                    tool_calls=(),
                    used_web_search=False,
                    model=getattr(prefetched_decision, "model", None),
                    token_usage=getattr(prefetched_decision, "token_usage", None),
                )
        return SimpleNamespace(
            text="Ich schaue kurz nach.\nMorgen wird es sonnig.",
            response_id="resp_dual_lane",
            request_id="req_dual_lane",
            rounds=1,
            tool_calls=(),
            used_web_search=False,
            model="gpt-4o-mini",
            token_usage=None,
        )

    def run_handoff_only(self, prompt: str, **kwargs):
        self.run_handoff_calls.append({"prompt": prompt, **kwargs})
        on_lane_text_delta = kwargs.get("on_lane_text_delta")
        if on_lane_text_delta is not None:
            on_lane_text_delta(
                SimpleNamespace(
                    text="Heute wird es sonnig.",
                    lane="final",
                    replace_current=True,
                    atomic=True,
                )
            )
        return SimpleNamespace(
            text="Heute wird es sonnig.",
            response_id="resp_handoff",
            request_id="req_handoff",
            rounds=2,
            tool_calls=(),
            used_web_search=True,
            model="gpt-4o-mini",
            token_usage=None,
        )

    def run_runtime_local_tool_only(self, prompt: str, **kwargs):
        self.run_runtime_local_tool_only_calls.append({"prompt": prompt, **kwargs})
        return SimpleNamespace(
            text="Okay. Ich bin jetzt für 2 Minuten ruhig.",
            response_id="resp_runtime_local_tool",
            request_id="req_runtime_local_tool",
            rounds=1,
            tool_calls=(),
            used_web_search=False,
            model="gpt-4o-mini",
            token_usage=None,
        )

    def recover_with_llm(
        self,
        prompt: str,
        *,
        conversation=None,
        instructions=None,
        failure_reason: str,
        rounds: int = 1,
        tool_calls=(),
        tool_results=(),
        used_web_search: bool = False,
    ):
        self.recovery_calls.append(
            {
                "prompt": prompt,
                "conversation": conversation,
                "instructions": instructions,
                "failure_reason": failure_reason,
                "rounds": rounds,
                "tool_calls": tuple(tool_calls),
                "tool_results": tuple(tool_results),
                "used_web_search": used_web_search,
            }
        )
        return SimpleNamespace(
            text=self.recovery_text,
            response_id="resp_recovery",
            request_id="req_recovery",
            rounds=max(1, rounds),
            tool_calls=tuple(tool_calls),
            tool_results=tuple(tool_results),
            used_web_search=used_web_search,
            model="gpt-4o-mini",
            token_usage=None,
        )


class StubSupervisorDecision:
    def __init__(self, *, action: str, spoken_reply: str | None = None) -> None:
        self.action = action
        self.spoken_reply = spoken_reply
        self.spoken_ack = None


class StubConversationClosureEvaluator:
    def __init__(self, *, close_now: bool, confidence: float = 0.93, reason: str = "explicit_goodbye") -> None:
        self.decision = ConversationClosureDecision(
            close_now=close_now,
            confidence=confidence,
            reason=reason,
        )
        self.calls: list[dict[str, object]] = []

    def evaluate(self, **kwargs) -> ConversationClosureDecision:
        self.calls.append(kwargs)
        return self.decision
        self.kind = None
        self.goal = None
        self.allow_web_search = None
        self.response_id = "prefetch_resp"
        self.request_id = "prefetch_req"
        self.model = "gpt-4o-mini"
        self.token_usage = None


class TimedConversationClosureEvaluator:
    def __init__(self, *, delay_s: float, close_now: bool = False) -> None:
        self.delay_s = delay_s
        self.close_now = close_now
        self.started_at: float | None = None
        self.finished_at: float | None = None

    def evaluate(self, **kwargs) -> ConversationClosureDecision:
        del kwargs
        self.started_at = time.monotonic()
        time.sleep(self.delay_s)
        self.finished_at = time.monotonic()
        return ConversationClosureDecision(
            close_now=self.close_now,
            confidence=0.91 if self.close_now else 0.18,
            reason="explicit_goodbye" if self.close_now else "still_engaged",
        )


class BlockingConversationClosureEvaluator:
    def __init__(self, *, delay_s: float) -> None:
        self.delay_s = delay_s

    def evaluate(self, **kwargs) -> ConversationClosureDecision:
        del kwargs
        time.sleep(self.delay_s)
        return ConversationClosureDecision(
            close_now=False,
            confidence=0.12,
            reason="slow_closure_eval",
        )


class InterruptingConversationClosureEvaluator:
    def __init__(self, *, delay_s: float, trigger_interrupt: Callable[[], None]) -> None:
        self.delay_s = delay_s
        self.trigger_interrupt = trigger_interrupt
        self._triggered = False

    def evaluate(self, **kwargs) -> ConversationClosureDecision:
        del kwargs
        if not self._triggered:
            self._triggered = True
            Thread(target=self.trigger_interrupt, daemon=True).start()
        time.sleep(self.delay_s)
        return ConversationClosureDecision(
            close_now=False,
            confidence=0.12,
            reason="slow_closure_eval",
        )


class StreamingRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._workflow_trace_env = {
            key: os.environ.get(key)
            for key in (
                "TWINR_WORKFLOW_TRACE_ENABLED",
                "TWINR_WORKFLOW_TRACE_MODE",
                "TWINR_WORKFLOW_TRACE_DIR",
            )
        }
        os.environ["TWINR_WORKFLOW_TRACE_ENABLED"] = "0"
        os.environ.pop("TWINR_WORKFLOW_TRACE_MODE", None)
        os.environ.pop("TWINR_WORKFLOW_TRACE_DIR", None)
        self._created_streaming_loops: list[TwinrStreamingHardwareLoop] = []
        original_init = TwinrStreamingHardwareLoop.__init__

        def tracking_init(loop_self, *args, **kwargs):
            original_init(loop_self, *args, **kwargs)
            self._created_streaming_loops.append(loop_self)

        self._streaming_loop_init_patcher = mock.patch.object(
            TwinrStreamingHardwareLoop,
            "__init__",
            new=tracking_init,
        )
        self._streaming_loop_init_patcher.start()

    def tearDown(self) -> None:
        self._streaming_loop_init_patcher.stop()
        for loop in reversed(self._created_streaming_loops):
            try:
                loop.close(timeout_s=0.2)
            except Exception:
                pass
        gc.collect()
        for key, value in self._workflow_trace_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        super().tearDown()

    def test_streaming_tool_surface_omits_smart_home_tools_when_integration_is_unavailable(self) -> None:
        with TemporaryDirectory() as temp_dir, mock.patch(
            "twinr.agent.tools.runtime.availability.build_smart_home_hub_adapter",
            return_value=None,
        ):
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            tool_agent = FakeToolAgentProvider(config)
            support_provider = FakePrintBackend(config)

            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=TwinrRuntime(config=config),
                tool_agent_provider=tool_agent,
                print_backend=support_provider,
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=support_provider,
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )

        self.assertNotIn("list_smart_home_entities", loop._tool_handlers)
        self.assertNotIn("read_smart_home_sensor_stream", loop._tool_handlers)
        self.assertNotIn("inspect_camera", loop._tool_handlers)
        tool_schema_names = {schema["name"] for schema in loop.streaming_turn_loop.tool_schemas}
        self.assertNotIn("list_smart_home_entities", tool_schema_names)
        self.assertNotIn("read_smart_home_sensor_stream", tool_schema_names)
        self.assertNotIn("inspect_camera", tool_schema_names)

    def test_speech_lifecycle_refreshes_snapshot_while_answering(self) -> None:
        runtime = _FakeSnapshotRefreshRuntime()
        lifecycle = _SpeechLifecycle(
            runtime=runtime,
            emit_status=lambda: None,
            trace_event=lambda *args, **kwargs: None,
            processing_feedback=SimpleNamespace(stop=lambda: None),
            state_machine=SimpleNamespace(transition=lambda *args, **kwargs: None),
            turn_started=time.monotonic(),
            snapshot_refresh_interval_s=0.01,
        )

        lifecycle.on_speaking_started()
        time.sleep(0.04)
        lifecycle.resume_processing()
        refresh_calls_after_resume = runtime.refresh_snapshot_activity_calls
        time.sleep(0.03)
        lifecycle.stop_snapshot_heartbeat()

        self.assertEqual(runtime.begin_answering_calls, 1)
        self.assertEqual(runtime.resume_processing_calls, 1)
        self.assertGreaterEqual(refresh_calls_after_resume, 1)
        self.assertEqual(runtime.refresh_snapshot_activity_calls, refresh_calls_after_resume)

    def test_close_speech_output_stops_feedback_before_waiting_for_close(self) -> None:
        events: list[str] = []
        runtime = SimpleNamespace(
            status=SimpleNamespace(value="processing"),
            submit_transcript=lambda transcript: None,
            begin_answering=lambda: None,
            resume_processing=lambda: None,
            resume_answering_after_print=lambda: None,
            finalize_agent_turn=lambda response_text: response_text,
            finish_speaking=lambda: None,
            rearm_follow_up=lambda request_source="follow_up": None,
            refresh_snapshot_activity=lambda: None,
        )
        coordinator = StreamingTurnCoordinator(
            config=TwinrConfig(),
            runtime=runtime,
            request=StreamingTurnRequest(
                transcript="Hallo Twinr",
                listen_source="button",
                proactive_trigger=None,
                turn_started=time.monotonic(),
                capture_ms=0,
                stt_ms=0,
            ),
            lane_plan_factory=lambda: StreamingTurnLanePlan(turn_instructions=None),
            speech_services=StreamingTurnSpeechServices(
                tts_provider=SimpleNamespace(),
                player=SimpleNamespace(),
                playback_coordinator=None,
                segment_boundary=lambda _text: None,
            ),
            hooks=StreamingTurnCoordinatorHooks(
                emit=lambda _line: None,
                emit_status=lambda: None,
                trace_event=lambda *args, **kwargs: None,
                trace_decision=lambda *args, **kwargs: None,
                start_processing_feedback_loop=lambda _kind: (lambda: None),
                is_search_feedback_active=lambda: False,
                stop_search_feedback=lambda: events.append("search_stop"),
                should_stop=lambda: False,
                request_turn_stop=lambda _reason: None,
                cancel_interrupted_turn=lambda: None,
                record_usage=lambda **kwargs: None,
                evaluate_follow_up_closure=lambda **kwargs: SimpleNamespace(),
                apply_follow_up_closure_evaluation=lambda **kwargs: False,
                follow_up_rearm_allowed_now=lambda _source: False,
            ),
        )

        class _FakeSpeechOutput:
            def close(self, *, timeout_s: float | None = None) -> None:
                del timeout_s
                events.append("speech_close")

        coordinator._speech_output = _FakeSpeechOutput()
        coordinator.processing_feedback = SimpleNamespace(stop=lambda: events.append("processing_stop"))

        coordinator._close_speech_output()

        self.assertEqual(
            events,
            [
                "search_stop",
                "processing_stop",
                "speech_close",
                "search_stop",
                "processing_stop",
            ],
        )

    def test_streaming_loop_close_shuts_down_warmup_executor_and_chains_base_close(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=TwinrRuntime(config=config),
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=CapturingDualLaneLoop(),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )

        fake_future = mock.Mock()
        fake_executor = mock.Mock()
        loop._warmup_executor = fake_executor
        loop._warmup_futures = {"first_word": fake_future}

        with mock.patch.object(TwinrRealtimeHardwareLoop, "close", autospec=True) as parent_close:
            loop.close(timeout_s=0.7)

        fake_future.cancel.assert_called_once_with()
        fake_executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
        self.assertEqual(loop._warmup_futures, {})
        self.assertIsNone(loop._warmup_executor)
        parent_close.assert_called_once_with(loop, timeout_s=0.7)

    def test_streaming_loop_del_delegates_to_close(self) -> None:
        loop = object.__new__(TwinrStreamingHardwareLoop)
        with mock.patch.object(TwinrStreamingHardwareLoop, "close", autospec=True) as close:
            TwinrStreamingHardwareLoop.__del__(loop)

        close.assert_called_once_with(loop, timeout_s=0.2)

    def test_audio_turn_uses_same_stream_remote_commit_when_voice_orchestrator_owns_listening(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            recorder = FailIfCalledRecorder()
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=TwinrRuntime(config=config),
                tool_agent_provider=FakeToolAgentProvider(config),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                recorder=recorder,
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            class _AutoCommitVoiceOrchestrator(FakeVoiceOrchestrator):
                def notify_runtime_state(
                    self,
                    *,
                    state: str,
                    detail: str | None = None,
                    follow_up_allowed: bool = False,
                    **kwargs,
                ) -> None:
                    super().notify_runtime_state(
                        state=state,
                        detail=detail,
                        follow_up_allowed=follow_up_allowed,
                        **kwargs,
                    )
                    if state == "listening":
                        loop.handle_remote_transcript_committed("wie geht es dir", "listening")

            fake_voice = _AutoCommitVoiceOrchestrator()
            loop.voice_orchestrator = fake_voice
            loop._latest_sensor_observation_facts = {
                "camera": {"person_visible": True},
                "person_state": {
                    "interaction_ready": True,
                    "targeted_inference_blocked": False,
                    "recommended_channel": "speech",
                    "attention_state": {"state": "attending_to_device"},
                    "interaction_intent_state": {"state": "showing_intent"},
                },
            }
            completed_turns: list[dict[str, object]] = []

            def complete_streaming_turn(**kwargs):
                completed_turns.append(dict(kwargs))
                return True

            loop._complete_streaming_turn = complete_streaming_turn  # type: ignore[method-assign]

            handled = loop._run_single_audio_turn(
                initial_source="voice_activation",
                follow_up=False,
                listening_window=SimpleNamespace(
                    speech_pause_ms=600,
                    start_timeout_s=8.0,
                    pause_grace_ms=450,
                ),
                listen_source="voice_activation",
                proactive_trigger=None,
                speech_start_chunks=None,
                ignore_initial_ms=0,
                timeout_emit_key="listen_timeout",
                timeout_message="Listening timed out.",
                play_initial_beep=False,
            )

        self.assertTrue(handled)
        self.assertEqual(fake_voice.states, [("listening", "voice_activation", False)])
        self.assertEqual(fake_voice.paused, [])
        self.assertEqual(fake_voice.resumed, [])
        self.assertEqual(completed_turns[0]["transcript"], "wie geht es dir")
        self.assertEqual(completed_turns[0]["listen_source"], "voice_activation")

    def test_gesture_wakeup_streaming_loop_uses_same_stream_remote_commit(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            recorder = FailIfCalledRecorder()
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=TwinrRuntime(config=config),
                tool_agent_provider=FakeToolAgentProvider(config),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                recorder=recorder,
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            class _AutoCommitVoiceOrchestrator(FakeVoiceOrchestrator):
                def notify_runtime_state(
                    self,
                    *,
                    state: str,
                    detail: str | None = None,
                    follow_up_allowed: bool = False,
                    **kwargs,
                ) -> None:
                    super().notify_runtime_state(
                        state=state,
                        detail=detail,
                        follow_up_allowed=follow_up_allowed,
                        **kwargs,
                    )
                    if state == "listening":
                        loop.handle_remote_transcript_committed("erzaehl mir die news", "listening")

            fake_voice = _AutoCommitVoiceOrchestrator()
            loop.voice_orchestrator = fake_voice
            loop._latest_sensor_observation_facts = {
                "camera": {"person_visible": True},
                "person_state": {
                    "interaction_ready": True,
                    "targeted_inference_blocked": False,
                    "recommended_channel": "speech",
                    "attention_state": {"state": "attending_to_device"},
                    "interaction_intent_state": {"state": "showing_intent"},
                },
            }
            completed_turns: list[dict[str, object]] = []

            def complete_streaming_turn(**kwargs):
                completed_turns.append(dict(kwargs))
                return True

            loop._complete_streaming_turn = complete_streaming_turn  # type: ignore[method-assign]

            handled = loop.handle_gesture_wakeup(
                GestureWakeupDecision(
                    active=True,
                    reason="gesture_wakeup:peace_sign",
                    confidence=0.92,
                )
            )

        self.assertTrue(handled)
        self.assertEqual(fake_voice.states, [("listening", "gesture", False)])
        self.assertEqual(fake_voice.paused, [])
        self.assertEqual(fake_voice.resumed, [])
        self.assertEqual(completed_turns[0]["transcript"], "erzaehl mir die news")
        self.assertEqual(completed_turns[0]["listen_source"], "gesture")

    def test_openai_dual_lane_uses_separate_backends_per_lane(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                llm_provider="openai",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            loop = object.__new__(TwinrStreamingHardwareLoop)
            loop.config = config
            loop.tool_agent_provider = OpenAIToolCallingAgentProvider(
                OpenAIBackend(config=config, client=SimpleNamespace()),
            )
            loop._tool_handlers = {}
            loop.first_word_provider = None
            tool_schemas = ()
            streaming_turn_loop = TwinrStreamingHardwareLoop._build_streaming_turn_loop(
                loop,
                tool_schemas=tool_schemas,
            )

        self.assertIsInstance(streaming_turn_loop, DualLaneToolLoop)
        self.assertIsInstance(loop.first_word_provider, OpenAIFirstWordProvider)
        self.assertIsInstance(streaming_turn_loop.supervisor_provider, OpenAIToolCallingAgentProvider)
        self.assertIsInstance(streaming_turn_loop.specialist_provider, OpenAIToolCallingAgentProvider)
        self.assertIsNot(
            loop.first_word_provider.backend,
            streaming_turn_loop.specialist_provider.backend,
        )
        self.assertIsNot(
            loop.first_word_provider.backend,
            streaming_turn_loop.supervisor_provider.backend,
        )
        self.assertIsNot(
            streaming_turn_loop.supervisor_provider.backend,
            streaming_turn_loop.specialist_provider.backend,
        )
        self.assertIsNotNone(streaming_turn_loop.supervisor_decision_provider)
        self.assertIsNot(
            streaming_turn_loop.supervisor_decision_provider.backend,
            streaming_turn_loop.specialist_provider.backend,
        )

    def test_text_turn_executes_tool_calls_and_streams_tts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
                realtime_sensitive_tools_require_identity=False,
            )
            runtime = TwinrRuntime(config=config)
            tool_agent = FakeToolAgentProvider(config)
            support_provider = FakePrintBackend(config)
            tts_provider = FakeTextToSpeechProvider(config)
            player = FakePlayer()
            printer = FakePrinter()
            usage_store = FakeUsageStore()

            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=tool_agent,
                print_backend=support_provider,
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=support_provider,
                tts_provider=tts_provider,
                player=player,
                printer=printer,
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=usage_store,
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )

            keep_listening = loop._run_single_text_turn(
                transcript="Bitte druck das aus",
                listen_source="button",
                proactive_trigger=None,
            )

        self.assertTrue(keep_listening)
        self.assertEqual(printer.printed, ["AUSDRUCK"])
        self.assertIn("Ich drucke das.", runtime.last_response or "")
        self.assertIn("Ist erledigt.", runtime.last_response or "")
        self.assertEqual(tool_agent.continue_calls[0]["continuation_token"], "resp_start_1")
        self.assertEqual(tool_agent.continue_calls[0]["tool_results"][0].name, "print_receipt")
        self.assertEqual(tool_agent.start_calls[0]["allow_web_search"], False)
        self.assertTrue(any(call["request_kind"] == "print" for call in usage_store.calls))
        self.assertTrue(any(call["request_kind"] == "conversation" for call in usage_store.calls))

    def test_audio_turn_uses_streaming_stt_session_when_available(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            tool_agent = FakeToolAgentProvider(config)
            support_provider = FakePrintBackend(config)
            tts_provider = FakeTextToSpeechProvider(config)
            player = FakePlayer()
            printer = FakePrinter()
            usage_store = FakeUsageStore()
            stt_provider = FakeStreamingSpeechToTextProvider(config)
            lines: list[str] = []

            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=tool_agent,
                print_backend=support_provider,
                stt_provider=stt_provider,
                agent_provider=support_provider,
                tts_provider=tts_provider,
                recorder=FakeRecorder(),
                player=player,
                printer=printer,
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=usage_store,
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
                emit=lines.append,
            )

            keep_listening = loop._run_single_audio_turn(
                initial_source="button",
                follow_up=False,
                listening_window=runtime.listening_window(initial_source="button", follow_up=False),
                listen_source="button",
                proactive_trigger=None,
                speech_start_chunks=None,
                ignore_initial_ms=0,
                timeout_emit_key="listen_timeout",
                timeout_message="Listening timed out before speech started.",
                play_initial_beep=False,
            )

        self.assertTrue(keep_listening)
        self.assertEqual(stt_provider.start_calls[0]["sample_rate"], config.audio_sample_rate)
        self.assertEqual(stt_provider.start_calls[0]["channels"], config.audio_channels)
        self.assertEqual(stt_provider.start_calls[0]["language"], config.deepgram_stt_language)
        self.assertEqual(stt_provider.session.sent, [b"PCM-A", b"PCM-B"])
        self.assertEqual(stt_provider.session.finalize_calls, 1)
        self.assertTrue(stt_provider.session.closed)
        self.assertIn("transcript=Streaming Hallo Twinr", lines)
        self.assertIn("stt_streaming_early=true", lines)
        self.assertIn("stt_streaming_deferred_until_finalize=true", lines)

    def test_audio_turn_prefers_finalize_result_after_early_speech_final_snapshot(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            tool_agent = FakeToolAgentProvider(config)
            support_provider = FakePrintBackend(config)
            tts_provider = FakeTextToSpeechProvider(config)
            player = FakePlayer()
            printer = FakePrinter()
            usage_store = FakeUsageStore()
            stt_provider = DivergentSpeechFinalStreamingSpeechToTextProvider(config)
            lines: list[str] = []

            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=tool_agent,
                print_backend=support_provider,
                stt_provider=stt_provider,
                agent_provider=support_provider,
                tts_provider=tts_provider,
                recorder=FakeRecorder(),
                player=player,
                printer=printer,
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=usage_store,
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
                emit=lines.append,
            )

            keep_listening = loop._run_single_audio_turn(
                initial_source="button",
                follow_up=False,
                listening_window=runtime.listening_window(initial_source="button", follow_up=False),
                listen_source="button",
                proactive_trigger=None,
                speech_start_chunks=None,
                ignore_initial_ms=0,
                timeout_emit_key="listen_timeout",
                timeout_message="Listening timed out before speech started.",
                play_initial_beep=False,
            )

        self.assertTrue(keep_listening)
        self.assertEqual(stt_provider.session.finalize_calls, 1)
        self.assertIn("transcript=Was sind die neuesten Nachrichten?", lines)
        self.assertNotIn("transcript=Sind die neuesten Nachrichten.", lines)

    def test_audio_turn_sets_listening_before_initial_beep(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            tool_agent = FakeToolAgentProvider(config)
            support_provider = FakePrintBackend(config)
            tts_provider = FakeTextToSpeechProvider(config)
            player = FakePlayer()
            printer = FakePrinter()
            usage_store = FakeUsageStore()
            stt_provider = FakeStreamingSpeechToTextProvider(config)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=tool_agent,
                print_backend=support_provider,
                stt_provider=stt_provider,
                agent_provider=support_provider,
                tts_provider=tts_provider,
                recorder=FakeRecorder(),
                player=player,
                printer=printer,
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=usage_store,
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            beep_statuses: list[str] = []
            loop._play_listen_beep = lambda: beep_statuses.append(loop.runtime.status.value)

            keep_listening = loop._run_single_audio_turn(
                initial_source="button",
                follow_up=False,
                listening_window=runtime.listening_window(initial_source="button", follow_up=False),
                listen_source="button",
                proactive_trigger=None,
                speech_start_chunks=None,
                ignore_initial_ms=0,
                timeout_emit_key="listen_timeout",
                timeout_message="Listening timed out before speech started.",
                play_initial_beep=True,
            )

        self.assertTrue(keep_listening)
        self.assertEqual(beep_statuses, ["listening"])

    def test_audio_turn_defers_bare_speech_final_until_finalize(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            tool_agent = FakeToolAgentProvider(config)
            support_provider = FakePrintBackend(config)
            tts_provider = FakeTextToSpeechProvider(config)
            player = FakePlayer()
            printer = FakePrinter()
            usage_store = FakeUsageStore()
            stt_provider = BareSpeechFinalStreamingSpeechToTextProvider(config)
            lines: list[str] = []

            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=tool_agent,
                print_backend=support_provider,
                stt_provider=stt_provider,
                agent_provider=support_provider,
                tts_provider=tts_provider,
                recorder=FakeRecorder(),
                player=player,
                printer=printer,
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=usage_store,
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
                emit=lines.append,
            )

            keep_listening = loop._run_single_audio_turn(
                initial_source="button",
                follow_up=False,
                listening_window=runtime.listening_window(initial_source="button", follow_up=False),
                listen_source="button",
                proactive_trigger=None,
                speech_start_chunks=None,
                ignore_initial_ms=0,
                timeout_emit_key="listen_timeout",
                timeout_message="Listening timed out before speech started.",
                play_initial_beep=False,
            )

        self.assertTrue(keep_listening)
        self.assertEqual(stt_provider.session.finalize_calls, 1)
        self.assertEqual(len(stt_provider.transcribe_calls), 1)
        self.assertIn("transcript=Geht's dir heute gut?", lines)
        self.assertIn("stt_streaming_recovered_via_batch=true", lines)
        self.assertNotIn("stt_streaming_early=true", lines)

    def test_streaming_fallback_capture_is_interruptible(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            tool_agent = FakeToolAgentProvider(config)
            support_provider = FakePrintBackend(config)
            tts_provider = FakeTextToSpeechProvider(config)
            player = FakePlayer()
            printer = FakePrinter()
            usage_store = FakeUsageStore()
            recorder = BlockingFallbackRecorder()
            lines: list[str] = []
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=tool_agent,
                print_backend=support_provider,
                stt_provider=FakeStreamingSpeechToTextProvider(config),
                agent_provider=support_provider,
                tts_provider=tts_provider,
                recorder=recorder,
                player=player,
                printer=printer,
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=usage_store,
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
                emit=lines.append,
            )
            loop._capture_and_transcribe_streaming = lambda **kwargs: (_ for _ in ()).throw(
                ValueError("streaming setup failed")
            )
            stop_event = Event()
            loop._set_active_turn_stop_event(stop_event)
            results: list[bool] = []
            worker = Thread(
                target=lambda: results.append(
                    loop._run_single_audio_turn(
                        initial_source="button",
                        follow_up=False,
                        listening_window=runtime.listening_window(initial_source="button", follow_up=False),
                        listen_source="button",
                        proactive_trigger=None,
                        speech_start_chunks=None,
                        ignore_initial_ms=0,
                        timeout_emit_key="listen_timeout",
                        timeout_message="Listening timed out before speech started.",
                        play_initial_beep=False,
                    )
                ),
                daemon=True,
            )

            worker.start()
            self.assertTrue(recorder.started.wait(timeout=0.5))
            self.assertTrue(loop._request_active_turn_interrupt())
            worker.join(timeout=1.0)
            loop._clear_active_turn_stop_event(stop_event)

        self.assertFalse(worker.is_alive())
        self.assertEqual(results, [False])
        self.assertTrue(callable(recorder.calls[0].get("should_stop")))
        self.assertIn("turn_interrupted=true", lines)
        self.assertNotIn("listen_timeout=true", lines)

    def test_streaming_no_speech_timeout_does_not_start_second_capture_window(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            lines: list[str] = []
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeStreamingSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                recorder=FailIfCalledRecorder(),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
                emit=lines.append,
            )
            loop._capture_and_transcribe_streaming = lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("No speech detected before timeout")
            )

            keep_listening = loop._run_single_audio_turn(
                initial_source="button",
                follow_up=False,
                listening_window=runtime.listening_window(initial_source="button", follow_up=False),
                listen_source="button",
                proactive_trigger=None,
                speech_start_chunks=None,
                ignore_initial_ms=0,
                timeout_emit_key="listen_timeout",
                timeout_message="Listening timed out before speech started.",
                play_initial_beep=False,
            )

        self.assertFalse(keep_listening)
        self.assertIn("listen_timeout=true", lines)
        self.assertNotIn("turn_controller_fallback=RuntimeError", lines)

    def test_streaming_no_speech_timeout_emits_capture_diagnostics(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            lines: list[str] = []
            diagnostics = ListenTimeoutCaptureDiagnostics(
                device="default",
                sample_rate=16000,
                channels=1,
                chunk_ms=100,
                speech_threshold=700,
                chunk_count=11,
                active_chunk_count=2,
                average_rms=205,
                peak_rms=477,
                listened_ms=8012,
            )
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeStreamingSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                recorder=FailIfCalledRecorder(),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
                emit=lines.append,
            )
            loop._capture_and_transcribe_streaming = lambda **kwargs: (_ for _ in ()).throw(
                SpeechStartTimeoutError(
                    "No speech detected before timeout",
                    diagnostics=diagnostics,
                )
            )

            keep_listening = loop._run_single_audio_turn(
                initial_source="button",
                follow_up=False,
                listening_window=runtime.listening_window(initial_source="button", follow_up=False),
                listen_source="button",
                proactive_trigger=None,
                speech_start_chunks=None,
                ignore_initial_ms=0,
                timeout_emit_key="listen_timeout",
                timeout_message="Listening timed out before speech started.",
                play_initial_beep=False,
            )

        self.assertFalse(keep_listening)
        self.assertIn("listen_timeout=true", lines)
        self.assertIn("listen_timeout_capture_device=default", lines)
        self.assertIn("listen_timeout_chunk_count=11", lines)
        self.assertIn("listen_timeout_peak_rms=477.0", lines)
        self.assertIn("listen_timeout_active_ratio=0.18", lines)

    def test_audio_turn_recovers_empty_speech_final_with_batch_stt(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            tool_agent = FakeToolAgentProvider(config)
            support_provider = FakePrintBackend(config)
            tts_provider = FakeTextToSpeechProvider(config)
            player = FakePlayer()
            printer = FakePrinter()
            usage_store = FakeUsageStore()
            stt_provider = EmptySpeechFinalStreamingSpeechToTextProvider(config)
            lines: list[str] = []

            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=tool_agent,
                print_backend=support_provider,
                stt_provider=stt_provider,
                agent_provider=support_provider,
                tts_provider=tts_provider,
                recorder=FakeRecorder(),
                player=player,
                printer=printer,
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=usage_store,
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
                emit=lines.append,
            )

            keep_listening = loop._run_single_audio_turn(
                initial_source="button",
                follow_up=False,
                listening_window=runtime.listening_window(initial_source="button", follow_up=False),
                listen_source="button",
                proactive_trigger=None,
                speech_start_chunks=None,
                ignore_initial_ms=0,
                timeout_emit_key="listen_timeout",
                timeout_message="Listening timed out before speech started.",
                play_initial_beep=False,
            )

        self.assertTrue(keep_listening)
        self.assertEqual(stt_provider.session.finalize_calls, 1)
        self.assertEqual(len(stt_provider.transcribe_calls), 1)
        self.assertIn("transcript=Wie geht es dir heute?", lines)
        self.assertIn("stt_streaming_recovered_via_batch=true", lines)

    def test_audio_turn_recovers_short_interim_finalize_with_batch_stt(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            tool_agent = FakeToolAgentProvider(config)
            support_provider = FakePrintBackend(config)
            tts_provider = FakeTextToSpeechProvider(config)
            player = FakePlayer()
            printer = FakePrinter()
            usage_store = FakeUsageStore()
            stt_provider = ShortInterimStreamingSpeechToTextProvider(config)
            lines: list[str] = []

            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=tool_agent,
                print_backend=support_provider,
                stt_provider=stt_provider,
                agent_provider=support_provider,
                tts_provider=tts_provider,
                recorder=FakeRecorder(),
                player=player,
                printer=printer,
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=usage_store,
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
                emit=lines.append,
            )

            keep_listening = loop._run_single_audio_turn(
                initial_source="button",
                follow_up=False,
                listening_window=runtime.listening_window(initial_source="button", follow_up=False),
                listen_source="button",
                proactive_trigger=None,
                speech_start_chunks=None,
                ignore_initial_ms=0,
                timeout_emit_key="listen_timeout",
                timeout_message="Listening timed out before speech started.",
                play_initial_beep=False,
            )

        self.assertTrue(keep_listening)
        self.assertEqual(stt_provider.session.finalize_calls, 1)
        self.assertEqual(len(stt_provider.transcribe_calls), 1)
        self.assertIn("transcript=Geht's dir heute gut?", lines)
        self.assertIn("stt_streaming_recovered_via_batch=true", lines)

    def test_audio_turn_verifies_utterance_end_fast_path_with_openai_verifier(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            tool_agent = FakeToolAgentProvider(config)
            support_provider = FakePrintBackend(config)
            tts_provider = FakeTextToSpeechProvider(config)
            player = FakePlayer()
            printer = FakePrinter()
            usage_store = FakeUsageStore()
            stt_provider = UtteranceEndOnlyStreamingSpeechToTextProvider(config)
            verifier_provider = FakeVerifierSpeechToTextProvider(
                config,
                transcript="Geht's dir heute gut?",
            )
            lines: list[str] = []

            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=tool_agent,
                print_backend=support_provider,
                stt_provider=stt_provider,
                verification_stt_provider=verifier_provider,
                agent_provider=support_provider,
                tts_provider=tts_provider,
                recorder=FakeRecorder(),
                player=player,
                printer=printer,
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=usage_store,
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
                emit=lines.append,
            )

            keep_listening = loop._run_single_audio_turn(
                initial_source="button",
                follow_up=False,
                listening_window=runtime.listening_window(initial_source="button", follow_up=False),
                listen_source="button",
                proactive_trigger=None,
                speech_start_chunks=None,
                ignore_initial_ms=0,
                timeout_emit_key="listen_timeout",
                timeout_message="Listening timed out before speech started.",
                play_initial_beep=False,
            )

        self.assertTrue(keep_listening)
        self.assertEqual(stt_provider.session.finalize_calls, 1)
        self.assertEqual(len(verifier_provider.calls), 1)
        self.assertIn("stt_streaming_verified_via_openai=true", lines)
        self.assertIn("transcript=Geht's dir heute gut?", lines)

    def test_segment_boundary_prefers_clause_and_soft_wrap(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                streaming_tts_clause_min_chars=20,
                streaming_tts_soft_segment_chars=40,
                streaming_tts_hard_segment_chars=60,
                long_term_memory_query_rewrite_enabled=False,
            )
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=TwinrRuntime(config=config),
                tool_agent_provider=FakeToolAgentProvider(config),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                recorder=FakeRecorder(),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )

            clause_boundary = loop._segment_boundary("Das Wetter morgen ist wechselhaft, aber am Nachmittag trockener")
            soft_wrap_boundary = loop._segment_boundary(
                "Das Wetter morgen bleibt insgesamt wechselhaft und am Nachmittag wieder trockener"
            )

        self.assertEqual(clause_boundary, len("Das Wetter morgen ist wechselhaft,"))
        self.assertIsNotNone(soft_wrap_boundary)
        self.assertLess(soft_wrap_boundary, len("Das Wetter morgen bleibt insgesamt wechselhaft und am Nachmittag wieder trockener"))

    def test_streaming_tts_uses_configured_chunk_size(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                openai_tts_stream_chunk_size=1536,
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            tool_agent = FakeToolAgentProvider(config)
            tts_provider = FakeTextToSpeechProvider(config)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=tool_agent,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=tts_provider,
                recorder=FakeRecorder(),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )

            loop._run_single_text_turn(
                transcript="Bitte druck das aus",
                listen_source="button",
                proactive_trigger=None,
            )

        self.assertIn(1536, tts_provider.stream_chunk_sizes)

    def test_dual_lane_filler_uses_streamed_tts_not_prefetched_audio(self) -> None:
        class AckOnlyLoop(DualLaneToolLoop):
            def __init__(self) -> None:
                pass

            def run(self, *args, **kwargs):
                on_lane_text_delta = kwargs.get("on_lane_text_delta")
                if on_lane_text_delta is not None:
                    on_lane_text_delta(
                        SimpleNamespace(
                            text="Ich schaue kurz nach.",
                            lane="filler",
                            replace_current=False,
                        )
                    )
                return SimpleNamespace(
                    text="Ich schaue kurz nach.",
                    response_id="resp_ack",
                    request_id="req_ack",
                    rounds=1,
                    tool_calls=(),
                    used_web_search=False,
                    model="gpt-4o-mini",
                    token_usage=None,
                )

        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            tts_provider = FakeTextToSpeechProvider(config)
            player = FakePlayer()
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=AckOnlyLoop(),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=tts_provider,
                player=player,
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )

            loop._run_single_text_turn(
                transcript="Wie wird das Wetter morgen?",
                listen_source="button",
                proactive_trigger=None,
            )

        self.assertEqual(player.played_wav_bytes, [])
        self.assertIn("Ich schaue kurz nach.", tts_provider.stream_calls)

    def test_prefetched_search_handoff_runs_handoff_only_lane_once(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            tts_provider = FakeTextToSpeechProvider(config)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=tts_provider,
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            decision = SimpleNamespace(
                action="handoff",
                spoken_ack="Ich schaue kurz nach.",
                spoken_reply=None,
                kind="search",
                goal="Check the weather.",
                allow_web_search=True,
                response_id="prefetch_resp",
                request_id="prefetch_req",
                model="gpt-4o-mini",
                token_usage=None,
            )

            loop._consume_speculative_supervisor_decision = lambda transcript: decision  # type: ignore[method-assign]

            keep_listening = loop._run_single_text_turn(
                transcript="Wie ist das Wetter heute?",
                listen_source="button",
                proactive_trigger=None,
            )

        self.assertTrue(keep_listening)
        self.assertEqual(dual_lane.run_calls, [])
        self.assertEqual(len(dual_lane.run_handoff_calls), 1)
        self.assertFalse(dual_lane.run_handoff_calls[0]["emit_filler"])
        self.assertEqual(tts_provider.stream_calls, ["Ich schaue kurz nach.", "Heute wird es sonnig."])
        self.assertEqual(runtime.last_response, "Heute wird es sonnig.")

    def test_prefetched_search_handoff_handoff_only_lane_skips_internal_filler(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            tts_provider = FakeTextToSpeechProvider(config)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=tts_provider,
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            loop.first_word_provider = FakeFirstWordProvider(
                config,
                reply=FirstWordReply(mode="filler", spoken_text="Ich schaue kurz nach."),
            )
            decision = SimpleNamespace(
                action="handoff",
                spoken_ack="Ich schaue kurz nach.",
                spoken_reply=None,
                kind="search",
                goal="Check the weather.",
                allow_web_search=True,
                response_id="prefetch_resp",
                request_id="prefetch_req",
                model="gpt-4o-mini",
                token_usage=None,
            )

            loop._consume_speculative_supervisor_decision = lambda transcript: decision  # type: ignore[method-assign]

            keep_listening = loop._run_single_text_turn(
                transcript="Wie ist das Wetter heute?",
                listen_source="button",
                proactive_trigger=None,
            )

        self.assertTrue(keep_listening)
        self.assertEqual(dual_lane.run_calls, [])
        self.assertEqual(len(dual_lane.run_handoff_calls), 1)
        self.assertFalse(dual_lane.run_handoff_calls[0]["emit_filler"])
        self.assertEqual(tts_provider.stream_calls[-1], "Heute wird es sonnig.")
        self.assertEqual(runtime.last_response, "Heute wird es sonnig.")

    def test_prefetched_supervisor_ack_is_used_before_final_lane(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            tts_provider = FakeTextToSpeechProvider(config)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=tts_provider,
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            loop._consume_speculative_supervisor_decision = lambda transcript: SimpleNamespace(  # type: ignore[method-assign]
                action="handoff",
                spoken_ack="Ich schaue kurz nach.",
                spoken_reply=None,
                kind="search",
                goal="Check the weather.",
                allow_web_search=True,
                response_id="decision_resp",
                request_id="decision_req",
                model="gpt-4o-mini",
                token_usage=None,
            )
            loop._run_dual_lane_final_response = lambda transcript, turn_instructions, prefetched_decision=None: SimpleNamespace(  # type: ignore[method-assign]
                text="Heute wird es sonnig.",
                response_id="resp_final",
                request_id="req_final",
                rounds=2,
                tool_calls=(),
                used_web_search=True,
                model="gpt-4o-mini",
                token_usage=None,
            )

            keep_listening = loop._run_single_text_turn(
                transcript="Wie ist das Wetter heute?",
                listen_source="button",
                proactive_trigger=None,
            )

        self.assertTrue(keep_listening)
        self.assertEqual(tts_provider.stream_calls, ["Ich schaue kurz nach.", "Heute wird es sonnig."])
        self.assertEqual(runtime.last_response, "Heute wird es sonnig.")

    def test_prefetched_search_handoff_starts_search_feedback_during_final_lane(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            feedback_events: list[str] = []
            loop._start_search_feedback_loop = lambda: (  # type: ignore[method-assign]
                feedback_events.append("start"),
                (lambda: feedback_events.append("stop")),
            )[1]
            loop._consume_speculative_supervisor_decision = lambda transcript: SimpleNamespace(  # type: ignore[method-assign]
                action="handoff",
                spoken_ack="Ich schaue kurz nach.",
                spoken_reply=None,
                kind="search",
                goal="Check the weather.",
                allow_web_search=True,
                response_id="decision_resp",
                request_id="decision_req",
                model="gpt-4o-mini",
                token_usage=None,
            )

            result = loop._run_dual_lane_final_response(
                "Wie ist das Wetter heute?",
                turn_instructions=None,
            )

        self.assertEqual(result.text, "Heute wird es sonnig.")
        self.assertEqual(feedback_events, ["start", "stop"])
        self.assertEqual(len(dual_lane.run_handoff_calls), 1)

    def test_resolved_search_handoff_starts_search_feedback_during_final_lane(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            feedback_events: list[str] = []
            loop._start_search_feedback_loop = lambda: (  # type: ignore[method-assign]
                feedback_events.append("start"),
                (lambda: feedback_events.append("stop")),
            )[1]
            loop._consume_speculative_supervisor_decision = lambda transcript: None  # type: ignore[method-assign]
            dual_lane.supervisor_decision_provider = object()  # type: ignore[attr-defined]
            dual_lane.resolve_supervisor_decision = lambda *args, **kwargs: SimpleNamespace(  # type: ignore[method-assign]
                action="handoff",
                spoken_ack="Ich schaue kurz nach.",
                spoken_reply=None,
                kind="search",
                goal="Check the weather.",
                allow_web_search=True,
                response_id="decision_resp",
                request_id="decision_req",
                model="gpt-4o-mini",
                token_usage=None,
            )

            result = loop._run_dual_lane_final_response(
                "Wie ist das Wetter heute?",
                turn_instructions=None,
            )

        self.assertEqual(result.text, "Heute wird es sonnig.")
        self.assertEqual(feedback_events, ["start", "stop"])
        self.assertEqual(len(dual_lane.run_handoff_calls), 1)

    def test_active_processing_feedback_suppresses_search_feedback_swap(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            feedback_events: list[str] = []
            loop._working_feedback_stop = lambda: feedback_events.append("working-stop")  # type: ignore[attr-defined]
            loop._start_search_feedback_loop = lambda: (  # type: ignore[method-assign]
                feedback_events.append("search-start"),
                (lambda: feedback_events.append("search-stop")),
            )[1]
            loop._consume_speculative_supervisor_decision = lambda transcript: SimpleNamespace(  # type: ignore[method-assign]
                action="handoff",
                spoken_ack="Ich schaue kurz nach.",
                spoken_reply=None,
                kind="search",
                goal="Check the weather.",
                allow_web_search=True,
                response_id="decision_resp",
                request_id="decision_req",
                model="gpt-4o-mini",
                token_usage=None,
            )

            result = loop._run_dual_lane_final_response(
                "Was ist bei agentischer KI momentan aktuell?",
                turn_instructions=None,
            )

        self.assertEqual(result.text, "Heute wird es sonnig.")
        self.assertEqual(feedback_events, [])
        self.assertEqual(len(dual_lane.run_handoff_calls), 1)

    def test_prefetched_direct_supervisor_reply_returns_to_waiting_without_processing_feedback(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            tts_provider = FakeTextToSpeechProvider(config)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=tts_provider,
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            loop._consume_speculative_supervisor_decision = lambda transcript: SimpleNamespace(  # type: ignore[method-assign]
                action="direct",
                spoken_ack=None,
                spoken_reply="Ja, alles gut.",
                kind=None,
                goal=None,
                allow_web_search=None,
                response_id="decision_resp",
                request_id="decision_req",
                model="gpt-4o-mini",
                token_usage=None,
            )
            processing_feedback_calls: list[str] = []
            loop._start_working_feedback_loop = lambda kind: (  # type: ignore[method-assign]
                processing_feedback_calls.append(kind),
                (lambda: None),
            )[1]
            def _unexpected_final_lane(*args, **kwargs):
                del args, kwargs
                return SimpleNamespace(
                    text="Ja, alles gut.",
                    response_id="resp_direct",
                    request_id="req_direct",
                    rounds=1,
                    tool_calls=(),
                    used_web_search=False,
                    model="gpt-4o-mini",
                    token_usage=None,
                )

            loop._run_dual_lane_final_response = _unexpected_final_lane  # type: ignore[method-assign]

            keep_listening = loop._run_single_text_turn(
                transcript="Alles ok bei dir?",
                listen_source="button",
                proactive_trigger=None,
            )

        self.assertTrue(keep_listening)
        self.assertEqual(tts_provider.stream_calls, ["Ja, alles gut."])
        self.assertEqual(runtime.last_response, "Ja, alles gut.")
        self.assertEqual(processing_feedback_calls, [])
        self.assertEqual(runtime.status.value, "waiting")
        self.assertEqual(runtime.snapshot_store.load().status, "waiting")

    def test_dual_lane_direct_reply_uses_supervisor_instead_of_first_word_provider(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
                streaming_first_word_final_lane_wait_ms=0,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            decision = SimpleNamespace(
                action="direct",
                spoken_ack=None,
                spoken_reply="Mir geht's gut, danke! Und dir?",
                kind=None,
                goal=None,
                allow_web_search=None,
                response_id="decision_resp",
                request_id="decision_req",
                model="gpt-4o-mini",
                token_usage=None,
            )
            decision_provider = FakeSupervisorDecisionProvider(decision)
            dual_lane.supervisor_decision_provider = decision_provider
            dual_lane.supervisor_instructions = "Kurz und warm antworten."
            tts_provider = FakeTextToSpeechProvider(config)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=tts_provider,
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            loop.first_word_provider = ExplodingFirstWordProvider(config)
            loop._run_dual_lane_final_response = lambda *args, **kwargs: SimpleNamespace(  # type: ignore[method-assign]
                text="Mir geht's gut, danke! Und dir?",
                response_id="resp_direct",
                request_id="req_direct",
                rounds=1,
                tool_calls=(),
                used_web_search=False,
                model="gpt-4o-mini",
                token_usage=None,
            )

            keep_listening = loop._run_single_text_turn(
                transcript="Wie gehts dir heute so?",
                listen_source="button",
                proactive_trigger=None,
            )

        self.assertTrue(keep_listening)
        self.assertEqual(decision_provider.calls[-1]["prompt"], "Wie gehts dir heute so?")
        self.assertEqual(tts_provider.stream_calls, ["Mir geht's gut, danke! Und dir?"])
        self.assertEqual(runtime.last_response, "Mir geht's gut, danke! Und dir?")
        self.assertEqual(runtime.status.value, "waiting")

    def test_dual_lane_skips_speculative_first_word_when_supervisor_bridge_is_available(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            dual_lane.supervisor_decision_provider = FakeSupervisorDecisionProvider(
                SimpleNamespace(
                    action="handoff",
                    spoken_ack="Ich schaue kurz nach.",
                    spoken_reply=None,
                    kind="search",
                    goal="search",
                    allow_web_search=True,
                    response_id="resp_prefetch",
                    request_id="req_prefetch",
                    model="gpt-4o-mini",
                    token_usage=None,
                )
            )
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            exploding = ExplodingFirstWordProvider(config)
            loop.first_word_provider = exploding

            loop._maybe_start_speculative_first_word("Wie gehts")

        self.assertEqual(exploding.calls, [])

    def test_supervisor_bridge_reply_reuses_shared_speculative_decision(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            decision = SimpleNamespace(
                action="handoff",
                spoken_ack="Ich schaue kurz nach.",
                spoken_reply=None,
                kind="search",
                goal="search",
                allow_web_search=True,
                response_id="resp_prefetch",
                request_id="req_prefetch",
                model="gpt-4o-mini",
                token_usage=None,
            )
            decision_provider = FakeSupervisorDecisionProvider(decision)
            dual_lane.supervisor_decision_provider = decision_provider
            dual_lane.supervisor_instructions = "Kurz und klar antworten."
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            prewarm_calls = len(decision_provider.calls)

            loop._maybe_start_speculative_supervisor_decision("Was gibt es heute fuer Nachrichten?")
            reply = loop._generate_supervisor_bridge_reply(
                "Was gibt es heute fuer Nachrichten?",
                instructions=None,
            )

        self.assertIsNotNone(reply)
        self.assertEqual(reply.spoken_text, "Ich schaue kurz nach.")
        self.assertEqual(len(decision_provider.calls), prewarm_calls + 1)

    def test_search_prefetch_uses_extended_final_lane_timeout_budget(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                streaming_final_lane_watchdog_timeout_ms=4000,
                streaming_final_lane_hard_timeout_ms=15000,
                streaming_search_final_lane_watchdog_timeout_ms=6500,
                streaming_search_final_lane_hard_timeout_ms=28000,
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            decision = SimpleNamespace(
                action="handoff",
                spoken_ack=None,
                spoken_reply=None,
                kind="search",
                goal="search",
                allow_web_search=True,
                response_id="resp_prefetch",
                request_id="req_prefetch",
                model="gpt-4o-mini",
                token_usage=None,
            )
            loop._resolve_local_semantic_route = lambda transcript: None  # type: ignore[method-assign]
            loop._consume_speculative_supervisor_decision = lambda transcript: decision  # type: ignore[method-assign]

            lane_plan = loop._build_streaming_turn_lane_plan("Was ist in Hamburg los?")

        self.assertTrue(lane_plan.is_dual_lane)
        self.assertIsNotNone(lane_plan.timeout_policy)
        assert lane_plan.timeout_policy is not None
        self.assertEqual(lane_plan.timeout_policy.final_lane_watchdog_timeout_ms, 6500)
        self.assertEqual(lane_plan.timeout_policy.final_lane_hard_timeout_ms, 28000)

    def test_automation_handoff_uses_extended_final_lane_timeout_budget(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                streaming_final_lane_watchdog_timeout_ms=4000,
                streaming_final_lane_hard_timeout_ms=15000,
                streaming_search_final_lane_watchdog_timeout_ms=6500,
                streaming_search_final_lane_hard_timeout_ms=28000,
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            decision = SimpleNamespace(
                action="handoff",
                spoken_ack="Ich prüfe das kurz.",
                spoken_reply=None,
                kind="automation",
                goal="smart_home_status",
                allow_web_search=False,
                response_id="resp_prefetch",
                request_id="req_prefetch",
                model="gpt-4o-mini",
                token_usage=None,
            )
            loop._resolve_local_semantic_route = lambda transcript: None  # type: ignore[method-assign]
            loop._consume_speculative_supervisor_decision = lambda transcript: decision  # type: ignore[method-assign]

            lane_plan = loop._build_streaming_turn_lane_plan("Ist das Licht im Flur an?")

        self.assertTrue(lane_plan.is_dual_lane)
        self.assertIsNotNone(lane_plan.timeout_policy)
        assert lane_plan.timeout_policy is not None
        self.assertEqual(lane_plan.timeout_policy.final_lane_watchdog_timeout_ms, 6500)
        self.assertEqual(lane_plan.timeout_policy.final_lane_hard_timeout_ms, 28000)

    def test_direct_dual_lane_turn_keeps_generic_final_lane_timeout_budget(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                streaming_final_lane_watchdog_timeout_ms=4000,
                streaming_final_lane_hard_timeout_ms=15000,
                streaming_search_final_lane_watchdog_timeout_ms=6500,
                streaming_search_final_lane_hard_timeout_ms=28000,
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            decision = SimpleNamespace(
                action="direct",
                spoken_ack=None,
                spoken_reply="Kurz gesagt: Es geht um Haushalt und Verkehr.",
                kind=None,
                goal=None,
                allow_web_search=False,
                response_id="resp_prefetch",
                request_id="req_prefetch",
                model="gpt-4o-mini",
                token_usage=None,
            )
            loop._resolve_local_semantic_route = lambda transcript: None  # type: ignore[method-assign]
            loop._consume_speculative_supervisor_decision = lambda transcript: decision  # type: ignore[method-assign]

            lane_plan = loop._build_streaming_turn_lane_plan("Was ist in Hamburg los?")

        self.assertTrue(lane_plan.is_dual_lane)
        self.assertIsNotNone(lane_plan.timeout_policy)
        assert lane_plan.timeout_policy is not None
        self.assertEqual(lane_plan.timeout_policy.final_lane_watchdog_timeout_ms, 4000)
        self.assertEqual(lane_plan.timeout_policy.final_lane_hard_timeout_ms, 15000)

    def test_unresolved_supervisor_dual_lane_turn_uses_extended_final_lane_timeout_budget(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                streaming_final_lane_watchdog_timeout_ms=4000,
                streaming_final_lane_hard_timeout_ms=15000,
                streaming_search_final_lane_watchdog_timeout_ms=6500,
                streaming_search_final_lane_hard_timeout_ms=28000,
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            dual_lane.supervisor_decision_provider = object()  # type: ignore[attr-defined]
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            loop._resolve_local_semantic_route = lambda transcript: None  # type: ignore[method-assign]
            loop._consume_speculative_supervisor_decision = lambda transcript: None  # type: ignore[method-assign]

            lane_plan = loop._build_streaming_turn_lane_plan("Was ist bei agentischer KI momentan aktuell?")

        self.assertTrue(lane_plan.is_dual_lane)
        self.assertIsNotNone(lane_plan.timeout_policy)
        assert lane_plan.timeout_policy is not None
        self.assertEqual(lane_plan.timeout_policy.final_lane_watchdog_timeout_ms, 6500)
        self.assertEqual(lane_plan.timeout_policy.final_lane_hard_timeout_ms, 28000)

    def test_direct_goodbye_turn_uses_closure_guard_to_suppress_follow_up(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                conversation_follow_up_enabled=True,
                conversation_closure_guard_enabled=True,
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            tts_provider = FakeTextToSpeechProvider(config)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=tts_provider,
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            loop._consume_speculative_supervisor_decision = lambda transcript: SimpleNamespace(  # type: ignore[method-assign]
                action="direct",
                spoken_ack=None,
                spoken_reply="Bis dann!",
                kind=None,
                goal=None,
                allow_web_search=None,
                response_id="decision_resp",
                request_id="decision_req",
                model="gpt-4o-mini",
                token_usage=None,
            )
            closure = StubConversationClosureEvaluator(close_now=True)
            loop.conversation_closure_evaluator = closure

            keep_listening = loop._run_single_text_turn(
                transcript="Bis später.",
                listen_source="button",
                proactive_trigger=None,
            )

        self.assertFalse(keep_listening)
        self.assertEqual(tts_provider.stream_calls, ["Bis dann!"])
        self.assertEqual(runtime.last_response, "Bis dann!")
        self.assertEqual(len(closure.calls), 1)
        self.assertEqual(closure.calls[0]["user_transcript"], "Bis später.")

    def test_direct_remote_follow_up_rearms_runtime_and_notifies_gateway(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                conversation_follow_up_enabled=True,
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            tts_provider = FakeTextToSpeechProvider(config)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=CapturingDualLaneLoop(),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=tts_provider,
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            loop.voice_orchestrator = FakeVoiceOrchestrator()
            loop._latest_sensor_observation_facts = {
                "camera": {"person_visible": True},
                "person_state": {
                    "interaction_ready": True,
                    "targeted_inference_blocked": False,
                    "recommended_channel": "speech",
                    "attention_state": {"state": "attending_to_device"},
                    "interaction_intent_state": {"state": "showing_intent"},
                },
            }
            loop._consume_speculative_supervisor_decision = lambda transcript: SimpleNamespace(  # type: ignore[method-assign]
                action="direct",
                spoken_ack=None,
                spoken_reply="Mir geht's gut, danke! Und dir?",
                kind=None,
                goal=None,
                allow_web_search=None,
                response_id="decision_resp",
                request_id="decision_req",
                model="gpt-4o-mini",
                token_usage=None,
            )

            keep_listening = loop._run_single_text_turn(
                transcript="Wie geht es dir?",
                listen_source="voice_activation",
                proactive_trigger=None,
            )

        self.assertTrue(keep_listening)
        self.assertEqual(runtime.status.value, "listening")
        self.assertIn(
            (TwinrStatus.ANSWERING, TwinrEvent.FOLLOW_UP_ARMED, TwinrStatus.LISTENING),
            runtime.state_machine.history,
        )
        assert loop.voice_orchestrator is not None
        self.assertEqual(
            loop.voice_orchestrator.states,
            [
                ("thinking", "voice_activation", False),
                ("follow_up_open", "voice_activation", True),
            ],
        )

    def test_voice_activation_quiet_turn_finishes_speaking_instead_of_rearming_follow_up(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                conversation_follow_up_enabled=True,
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            tts_provider = FakeTextToSpeechProvider(config)
            lines: list[str] = []
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=CapturingDualLaneLoop(),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=tts_provider,
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
                emit=lines.append,
            )
            loop.voice_orchestrator = FakeVoiceOrchestrator()
            loop._latest_sensor_observation_facts = {
                "camera": {"person_visible": True},
                "person_state": {
                    "interaction_ready": True,
                    "targeted_inference_blocked": False,
                    "recommended_channel": "speech",
                    "attention_state": {"state": "attending_to_device"},
                    "interaction_intent_state": {"state": "showing_intent"},
                },
            }
            loop._consume_speculative_supervisor_decision = lambda transcript: SimpleNamespace(  # type: ignore[method-assign]
                action="direct",
                spoken_ack=None,
                spoken_reply="Gern. Ich bin jetzt 20 Minuten ruhig.",
                kind=None,
                goal=None,
                allow_web_search=None,
                response_id="decision_resp",
                request_id="decision_req",
                model="gpt-4o-mini",
                token_usage=None,
            )
            original_finalize = runtime.finalize_agent_turn

            def finalize_and_enable_quiet(response_text: str) -> str:
                answer = original_finalize(response_text)
                runtime.set_voice_quiet_minutes(minutes=20, reason="tv news")
                return answer

            runtime.finalize_agent_turn = finalize_and_enable_quiet  # type: ignore[method-assign]

            keep_listening = loop._run_single_text_turn(
                transcript="Sei bitte 20 Minuten ruhig.",
                listen_source="voice_activation",
                proactive_trigger=None,
            )

        self.assertTrue(keep_listening)
        self.assertTrue(runtime.voice_quiet_active())
        self.assertEqual(runtime.status.value, "waiting")
        self.assertNotIn(
            (TwinrStatus.ANSWERING, TwinrEvent.FOLLOW_UP_ARMED, TwinrStatus.LISTENING),
            runtime.state_machine.history,
        )
        self.assertIn("streaming_follow_up_rearm_snapshot=true", lines)
        self.assertIn("streaming_follow_up_rearm_allowed_now=false", lines)
        self.assertIn("streaming_turn_finish_path=finish_speaking", lines)
        assert loop.voice_orchestrator is not None
        self.assertEqual(
            loop.voice_orchestrator.states,
            [
                ("thinking", "voice_activation", False),
                ("waiting", "voice_activation", False),
            ],
        )

    def test_remote_follow_up_commit_reopens_streaming_turn_while_runtime_stays_listening(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                conversation_follow_up_enabled=True,
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=CapturingDualLaneLoop(),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            loop.voice_orchestrator = FakeVoiceOrchestrator()
            loop._latest_sensor_observation_facts = {
                "camera": {"person_visible": True},
                "person_state": {
                    "interaction_ready": True,
                    "targeted_inference_blocked": False,
                    "recommended_channel": "speech",
                    "attention_state": {"state": "attending_to_device"},
                    "interaction_intent_state": {"state": "showing_intent"},
                },
            }
            loop._consume_speculative_supervisor_decision = lambda transcript: SimpleNamespace(  # type: ignore[method-assign]
                action="direct",
                spoken_ack=None,
                spoken_reply="Mir geht's gut, danke! Und dir?",
                kind=None,
                goal=None,
                allow_web_search=None,
                response_id="decision_resp",
                request_id="decision_req",
                model="gpt-4o-mini",
                token_usage=None,
            )
            self.assertTrue(
                loop._run_single_text_turn(
                    transcript="Wie geht es dir?",
                    listen_source="voice_activation",
                    proactive_trigger=None,
                )
            )
            self.assertEqual(runtime.status.value, "listening")
            captured_kwargs: dict[str, object] = {}

            def fake_run_conversation_session(**kwargs):
                captured_kwargs.update(kwargs)
                return True

            loop._run_conversation_session = fake_run_conversation_session  # type: ignore[method-assign]

            handled = loop.handle_remote_transcript_committed(
                "Ich meinte, ich wollte mein WhatsApp bei dir als App einrichten.",
                "follow_up",
            )

        self.assertTrue(handled)
        self.assertEqual(captured_kwargs["initial_source"], "follow_up")
        self.assertEqual(
            captured_kwargs["seed_transcript"],
            "Ich meinte, ich wollte mein WhatsApp bei dir als App einrichten.",
        )
        self.assertFalse(captured_kwargs["play_initial_beep"])

    def test_streaming_text_turn_reuses_existing_listening_state(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                conversation_follow_up_enabled=True,
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=CapturingDualLaneLoop(),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            loop.runtime.begin_listening(request_source="follow_up")
            active_turn_id = loop.runtime._runtime_flow_state()["active_turn_id"]
            completed_turns: list[dict[str, object]] = []

            def fake_complete_streaming_turn(**kwargs):
                completed_turns.append(dict(kwargs))
                return True

            loop._complete_streaming_turn = fake_complete_streaming_turn  # type: ignore[method-assign]

            handled = loop._run_single_text_turn(
                transcript="wie geht es dir",
                listen_source="follow_up",
                proactive_trigger=None,
            )

        self.assertTrue(handled)
        self.assertEqual(loop.runtime.status.value, "listening")
        self.assertEqual(loop.runtime._runtime_flow_state()["active_turn_id"], active_turn_id)
        self.assertEqual(completed_turns[0]["transcript"], "wie geht es dir")

    def test_direct_follow_up_rearms_to_listening_without_waiting_gap(self) -> None:
        class SequencedRecorder:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def capture_pcm_until_pause_with_options(self, **kwargs):
                self.calls.append(dict(kwargs))
                if len(self.calls) == 1:
                    return SimpleNamespace(
                        pcm_bytes=b"PCM1",
                        speech_started_after_ms=80,
                        resumed_after_pause_count=0,
                    )
                raise RuntimeError("No speech detected before timeout")

        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                conversation_follow_up_enabled=True,
                conversation_follow_up_timeout_s=3.0,
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            recorder = SequencedRecorder()
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=CapturingDualLaneLoop(),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                recorder=recorder,
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            loop._consume_speculative_supervisor_decision = lambda transcript: SimpleNamespace(  # type: ignore[method-assign]
                action="direct",
                spoken_ack=None,
                spoken_reply="Mir geht's gut, danke! Und dir?",
                kind=None,
                goal=None,
                allow_web_search=None,
                response_id="decision_resp",
                request_id="decision_req",
                model="gpt-4o-mini",
                token_usage=None,
            )
            beep_statuses: list[str] = []
            loop._play_listen_beep = lambda: beep_statuses.append(loop.runtime.status.value)  # type: ignore[method-assign]

            result = loop._run_conversation_session(initial_source="button")

        self.assertTrue(result)
        self.assertEqual(beep_statuses, ["listening", "listening"])
        self.assertEqual(runtime.last_response, "Mir geht's gut, danke! Und dir?")
        self.assertEqual(recorder.calls[0]["speech_start_chunks"], None)
        self.assertEqual(recorder.calls[0]["ignore_initial_ms"], 0)
        self.assertEqual(recorder.calls[1]["speech_start_chunks"], 1)
        self.assertEqual(recorder.calls[1]["ignore_initial_ms"], 0)
        self.assertIn(
            (TwinrStatus.ANSWERING, TwinrEvent.FOLLOW_UP_ARMED, TwinrStatus.LISTENING),
            runtime.state_machine.history,
        )
        self.assertNotIn(
            (TwinrStatus.ANSWERING, TwinrEvent.TTS_FINISHED, TwinrStatus.WAITING),
            runtime.state_machine.history,
        )

    def test_follow_up_closure_eval_runs_during_playback(self) -> None:
        class SequencedRecorder:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def capture_pcm_until_pause_with_options(self, **kwargs):
                self.calls.append(dict(kwargs))
                if len(self.calls) == 1:
                    return SimpleNamespace(
                        pcm_bytes=b"PCM1",
                        speech_started_after_ms=80,
                        resumed_after_pause_count=0,
                    )
                raise RuntimeError("No speech detected before timeout")

        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                conversation_follow_up_enabled=True,
                conversation_follow_up_timeout_s=3.0,
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            recorder = SequencedRecorder()
            player = TimedPlayer(playback_delay_s=0.25)
            evaluator = TimedConversationClosureEvaluator(delay_s=0.12, close_now=False)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=CapturingDualLaneLoop(),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                recorder=recorder,
                player=player,
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
                conversation_closure_evaluator=evaluator,
            )
            loop._consume_speculative_supervisor_decision = lambda transcript: SimpleNamespace(  # type: ignore[method-assign]
                action="direct",
                spoken_ack=None,
                spoken_reply="Mir geht's gut, danke! Und dir?",
                kind=None,
                goal=None,
                allow_web_search=None,
                response_id="decision_resp",
                request_id="decision_req",
                model="gpt-4o-mini",
                token_usage=None,
            )
            beep_times: list[float] = []
            loop._play_listen_beep = lambda: beep_times.append(time.monotonic())  # type: ignore[method-assign]

            result = loop._run_conversation_session(initial_source="button")

        self.assertTrue(result)
        self.assertEqual(len(beep_times), 2)
        self.assertIsNotNone(evaluator.started_at)
        self.assertIsNotNone(evaluator.finished_at)
        self.assertIsNotNone(player.playback_finished_at)
        assert evaluator.started_at is not None
        assert evaluator.finished_at is not None
        assert player.playback_finished_at is not None
        self.assertLess(evaluator.started_at, player.playback_finished_at)
        self.assertLessEqual(evaluator.finished_at, player.playback_finished_at + 0.05)
        self.assertLess(beep_times[1] - player.playback_finished_at, 0.12)

    def test_closure_eval_timeout_does_not_strand_text_turn_in_speaking(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                conversation_follow_up_enabled=True,
                conversation_closure_guard_enabled=True,
                conversation_closure_provider_timeout_seconds=0.25,
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            tts_provider = FakeTextToSpeechProvider(config)
            lines: list[str] = []
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=CapturingDualLaneLoop(),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=tts_provider,
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
                conversation_closure_evaluator=BlockingConversationClosureEvaluator(delay_s=1.5),
                emit=lines.append,
            )
            loop._consume_speculative_supervisor_decision = lambda transcript: SimpleNamespace(  # type: ignore[method-assign]
                action="direct",
                spoken_ack=None,
                spoken_reply="Heute wird es sonnig.",
                kind=None,
                goal=None,
                allow_web_search=None,
                response_id="decision_resp",
                request_id="decision_req",
                model="gpt-4o-mini",
                token_usage=None,
            )

            started = time.monotonic()
            keep_listening = loop._run_single_text_turn(
                transcript="Wie ist das Wetter heute?",
                listen_source="button",
                proactive_trigger=None,
            )
            elapsed_s = time.monotonic() - started

        self.assertTrue(keep_listening)
        self.assertLess(elapsed_s, 1.0)
        self.assertEqual(runtime.status.value, "waiting")
        self.assertEqual(tts_provider.stream_calls, ["Heute wird es sonnig."])
        self.assertIn("conversation_closure_fallback=closure_eval_timeout", lines)

    def test_tool_history_recording_does_not_delay_turn_completion_after_playback(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            player = TimedPlayer(playback_delay_s=0.02)
            tool_history_started = Event()
            tool_history_finished = Event()
            tool_history_errors: list[str] = []
            runtime_status = SimpleNamespace(value="processing")

            def delayed_record_personality_tool_history(*, tool_calls, tool_results) -> None:
                del tool_calls, tool_results
                tool_history_started.set()
                try:
                    if not player.playback_finished.wait(timeout=1.0):
                        tool_history_errors.append("playback did not finish before tool-history recording")
                        return
                    time.sleep(0.25)
                finally:
                    tool_history_finished.set()

            runtime = SimpleNamespace(
                status=runtime_status,
                submit_transcript=lambda transcript: None,
                begin_answering=lambda: setattr(runtime_status, "value", "answering"),
                resume_processing=lambda: setattr(runtime_status, "value", "processing"),
                resume_answering_after_print=lambda: setattr(runtime_status, "value", "answering"),
                finalize_agent_turn=lambda response_text: response_text,
                finish_speaking=lambda: setattr(runtime_status, "value", "waiting"),
                rearm_follow_up=lambda request_source="follow_up": setattr(runtime_status, "value", "listening"),
                refresh_snapshot_activity=lambda: None,
                record_personality_tool_history=delayed_record_personality_tool_history,
            )
            coordinator = StreamingTurnCoordinator(
                config=config,
                runtime=runtime,
                request=StreamingTurnRequest(
                    transcript="Wie wird das Wetter?",
                    listen_source="button",
                    proactive_trigger=None,
                    turn_started=time.monotonic(),
                    capture_ms=0,
                    stt_ms=0,
                ),
                lane_plan_factory=lambda: StreamingTurnLanePlan(
                    turn_instructions=None,
                    run_single_lane=lambda on_text_delta: (
                        on_text_delta("Es wird kuehler auf zwei bis sechs Grad.")
                        or SimpleNamespace(
                            text="Es wird kuehler auf zwei bis sechs Grad.",
                            tool_calls=(SimpleNamespace(name="search_weather"),),
                            tool_results=(SimpleNamespace(name="search_weather"),),
                            response_id="resp_weather",
                            request_id="req_weather",
                            rounds=1,
                            used_web_search=True,
                            model="gpt-4o-mini",
                            token_usage=None,
                        )
                    ),
                ),
                speech_services=StreamingTurnSpeechServices(
                    tts_provider=FakeTextToSpeechProvider(config),
                    player=player,
                    playback_coordinator=None,
                    segment_boundary=lambda text: len(text) if text.strip() else None,
                ),
                hooks=StreamingTurnCoordinatorHooks(
                    emit=lambda _line: None,
                    emit_status=lambda: None,
                    trace_event=lambda *args, **kwargs: None,
                    trace_decision=lambda *args, **kwargs: None,
                    start_processing_feedback_loop=lambda _kind: (lambda: None),
                    is_search_feedback_active=lambda: False,
                    stop_search_feedback=lambda: None,
                    should_stop=lambda: False,
                    request_turn_stop=lambda _reason: None,
                    cancel_interrupted_turn=lambda: None,
                    record_usage=lambda **kwargs: None,
                    evaluate_follow_up_closure=lambda **kwargs: SimpleNamespace(
                        error_type=None,
                        decision=None,
                    ),
                    apply_follow_up_closure_evaluation=lambda **kwargs: False,
                    follow_up_rearm_allowed_now=lambda _source: False,
                ),
            )

            started = time.monotonic()
            outcome = coordinator.execute()
            elapsed_s = time.monotonic() - started

        self.assertTrue(outcome.keep_listening)
        self.assertLess(elapsed_s, 0.2)
        self.assertTrue(tool_history_started.wait(timeout=1.0))
        self.assertTrue(tool_history_finished.wait(timeout=1.0))
        self.assertEqual(tool_history_errors, [])
        self.assertEqual(runtime.status.value, "waiting")

    def test_required_remote_interrupt_during_closure_eval_preserves_error_state(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                conversation_follow_up_enabled=True,
                conversation_closure_guard_enabled=True,
                conversation_closure_provider_timeout_seconds=2.0,
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=True,
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            tts_provider = FakeTextToSpeechProvider(config)
            player = TimedPlayer(playback_delay_s=0.02)
            interrupt_errors: list[str] = []
            interrupt_completed = Event()
            lines: list[str] = []
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=CapturingDualLaneLoop(),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=tts_provider,
                player=player,
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
                emit=lines.append,
            )
            runtime.reset_error()
            runtime.search_provider_conversation_context = lambda: ()  # type: ignore[method-assign]
            runtime.supervisor_provider_conversation_context = lambda: ()  # type: ignore[method-assign]
            runtime.supervisor_direct_provider_conversation_context = lambda transcript: ()  # type: ignore[method-assign]
            runtime.tool_provider_conversation_context = lambda: ()  # type: ignore[method-assign]
            loop._consume_speculative_supervisor_decision = lambda transcript: SimpleNamespace(  # type: ignore[method-assign]
                action="direct",
                spoken_ack=None,
                spoken_reply="Welchen Ort meinst du?",
                kind=None,
                goal=None,
                allow_web_search=None,
                response_id="decision_resp",
                request_id="decision_req",
                model="gpt-4o-mini",
                token_usage=None,
            )

            def _trigger_required_remote() -> None:
                try:
                    if not player.playback_finished.wait(timeout=1.0):
                        interrupt_errors.append("playback did not finish before required-remote interrupt")
                        return
                    loop._handle_error(LongTermRemoteUnavailableError("remote unavailable during closure"))
                finally:
                    interrupt_completed.set()

            loop.conversation_closure_evaluator = InterruptingConversationClosureEvaluator(
                delay_s=1.5,
                trigger_interrupt=_trigger_required_remote,
            )

            started = time.monotonic()
            keep_listening = loop._run_single_text_turn(
                transcript="Wie ist das Wetter in",
                listen_source="button",
                proactive_trigger=None,
            )
            elapsed_s = time.monotonic() - started

        self.assertFalse(keep_listening)
        self.assertLess(elapsed_s, 1.0)
        self.assertTrue(interrupt_completed.wait(timeout=1.0))
        self.assertEqual(interrupt_errors, [])
        self.assertEqual(runtime.status.value, "error")
        self.assertIn("status=error", lines)
        self.assertIn("required_remote_dependency=false", lines)
        self.assertNotIn("conversation_closure_fallback=closure_eval_timeout", lines)
        self.assertNotIn("turn_interrupted=true", lines)
        self.assertFalse(any("Cannot apply tts_finished while in error" in line for line in lines))

    def test_required_remote_repeat_error_emits_dependency_false_when_already_error(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=True,
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            lines: list[str] = []
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=CapturingDualLaneLoop(),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
                emit=lines.append,
            )
            runtime.reset_error()

            loop._handle_error(LongTermRemoteUnavailableError("initial remote unavailable"))
            lines.clear()
            loop._handle_error(LongTermRemoteUnavailableError("remote unavailable again"))

        self.assertEqual(runtime.status.value, "error")
        self.assertIn("required_remote_dependency=false", lines)
        self.assertNotIn("runtime_reset_error=", lines)

    def test_final_lane_waits_for_first_audio_gate(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            trace: list[str] = []
            tts_provider = TraceTextToSpeechProvider(config, trace)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=tts_provider,
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            loop._consume_speculative_supervisor_decision = lambda transcript: SimpleNamespace(  # type: ignore[method-assign]
                action="handoff",
                spoken_ack="Ich schaue kurz nach.",
                spoken_reply=None,
                kind="search",
                goal="Check the weather.",
                allow_web_search=True,
                response_id="decision_resp",
                request_id="decision_req",
                model="gpt-4o-mini",
                token_usage=None,
            )

            def fake_final_response(transcript: str, *, turn_instructions: str | None, prefetched_decision=None):
                del transcript, turn_instructions, prefetched_decision
                trace.append("final_start")
                return SimpleNamespace(
                    text="Heute wird es sonnig.",
                    response_id="resp_final",
                    request_id="req_final",
                    rounds=1,
                    tool_calls=(),
                    used_web_search=True,
                    model="gpt-4o-mini",
                    token_usage=None,
                )

            loop._run_dual_lane_final_response = fake_final_response  # type: ignore[method-assign]

            keep_listening = loop._run_single_text_turn(
                transcript="Wie ist das Wetter heute?",
                listen_source="button",
                proactive_trigger=None,
            )

        self.assertTrue(keep_listening)
        self.assertLess(
            trace.index("tts_start:Ich schaue kurz nach."),
            trace.index("tts_start:Heute wird es sonnig."),
        )

    def test_full_context_direct_supervisor_decision_uses_filler_and_final_lane(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            trace: list[str] = []
            tts_provider = TraceTextToSpeechProvider(config, trace)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=tts_provider,
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            loop._consume_speculative_supervisor_decision = lambda transcript: SimpleNamespace(  # type: ignore[method-assign]
                action="direct",
                spoken_ack="Ich hole kurz unser heutiges Gespräch zusammen.",
                spoken_reply="Ich kann mich nicht erinnern.",
                kind="memory",
                goal="Recall what Twinr and the user discussed earlier today.",
                allow_web_search=None,
                context_scope="full_context",
                response_id="decision_resp",
                request_id="decision_req",
                model="gpt-4o-mini",
                token_usage=None,
            )

            def fake_final_response(transcript: str, *, turn_instructions: str | None, prefetched_decision=None):
                del transcript, turn_instructions, prefetched_decision
                return SimpleNamespace(
                    text="Vorhin haben wir über das Wetter gesprochen.",
                    response_id="resp_final",
                    request_id="req_final",
                    rounds=1,
                    tool_calls=(),
                    used_web_search=False,
                    model="gpt-4o-mini",
                    token_usage=None,
                )

            loop._run_dual_lane_final_response = fake_final_response  # type: ignore[method-assign]

            keep_listening = loop._run_single_text_turn(
                transcript="Worüber haben wir heute geredet?",
                listen_source="button",
                proactive_trigger=None,
            )

        self.assertTrue(keep_listening)
        self.assertEqual(runtime.last_response, "Vorhin haben wir über das Wetter gesprochen.")
        self.assertLess(
            trace.index("tts_start:Ich hole kurz unser heutiges Gespräch zusammen."),
            trace.index("tts_start:Vorhin haben wir über das Wetter gesprochen."),
        )

    def test_interrupt_does_not_wait_for_stalled_tts_first_chunk(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            tts_provider = BlockingTextToSpeechProvider(config)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=tts_provider,
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            loop.first_word_provider = FakeFirstWordProvider(
                config,
                reply=FirstWordReply(mode="direct", spoken_text="Ja, alles gut."),
            )
            stop_event = Event()
            loop._set_active_turn_stop_event(stop_event)
            results: list[bool] = []
            worker = Thread(
                target=lambda: results.append(
                    loop._run_single_text_turn(
                        transcript="Geht's dir heute gut?",
                        listen_source="button",
                        proactive_trigger=None,
                    )
                ),
                daemon=True,
            )

            worker.start()
            self.assertTrue(tts_provider.started.wait(timeout=1.0))

            started_at = time.monotonic()
            self.assertTrue(loop._request_active_turn_interrupt())
            worker.join(timeout=0.5)
            elapsed = time.monotonic() - started_at
            tts_provider.release.set()
            loop._clear_active_turn_stop_event(stop_event)

        self.assertFalse(worker.is_alive())
        self.assertEqual(results, [False])
        self.assertLess(elapsed, 0.5)

    def test_dual_lane_uses_model_first_word_provider_when_supervisor_prefetch_misses(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                streaming_bridge_reply_timeout_ms=250,
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            trace: list[str] = []
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            loop.first_word_provider = DelayedFirstWordProvider(
                config,
                delay_s=0.18,
                trace=trace,
            )

            def fake_final_response(transcript: str, *, turn_instructions: str | None, prefetched_decision=None):
                del transcript, turn_instructions, prefetched_decision
                trace.append("final_start")
                time.sleep(0.02)
                return SimpleNamespace(
                    text="Heute wird es sonnig.",
                    response_id="resp_final",
                    request_id="req_final",
                    rounds=1,
                    tool_calls=(),
                    used_web_search=True,
                    model="gpt-4o-mini",
                    token_usage=None,
                )

            loop._run_dual_lane_final_response = fake_final_response  # type: ignore[method-assign]

            keep_listening = loop._run_single_text_turn(
                transcript="Wie ist das Wetter heute?",
                listen_source="button",
                proactive_trigger=None,
            )
            time.sleep(0.22)

        self.assertTrue(keep_listening)
        self.assertIn("first_word_start", trace)
        self.assertIn("first_word_end", trace)
        self.assertEqual(runtime.last_response, "Heute wird es sonnig.")

    def test_speculative_first_word_requires_multiple_words(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                streaming_first_word_prefetch_min_chars=4,
                streaming_first_word_prefetch_min_words=2,
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=CapturingDualLaneLoop(),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            provider = FakeFirstWordProvider(
                config,
                reply=FirstWordReply(mode="filler", spoken_text="Ich schaue kurz nach."),
            )
            loop.first_word_provider = provider

            loop._maybe_start_speculative_first_word("heute")
            self.assertFalse(getattr(loop, "_speculative_first_word_started"))
            self.assertEqual(provider.calls, [])

            loop._maybe_start_speculative_first_word("wie heute")
            self.assertTrue(getattr(loop, "_speculative_first_word_done").wait(timeout=0.5))
            self.assertEqual(len(provider.calls), 1)

    def test_bridge_watchdog_does_not_emit_canned_fallback_when_model_first_word_misses(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                streaming_bridge_reply_timeout_ms=20,
                streaming_final_lane_watchdog_timeout_ms=250,
                streaming_final_lane_hard_timeout_ms=1000,
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            tts_provider = FakeTextToSpeechProvider(config)
            lines: list[str] = []
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=tts_provider,
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
                emit=lines.append,
            )
            loop.first_word_provider = DelayedFirstWordProvider(
                config,
                delay_s=0.2,
                reply=FirstWordReply(mode="filler", spoken_text="Ich schaue kurz nach."),
            )

            def fake_final_response(transcript: str, *, turn_instructions: str | None, prefetched_decision=None):
                del transcript, turn_instructions, prefetched_decision
                time.sleep(0.05)
                return SimpleNamespace(
                    text="Heute wird es sonnig.",
                    response_id="resp_final",
                    request_id="req_final",
                    rounds=1,
                    tool_calls=(),
                    used_web_search=True,
                    model="gpt-4o-mini",
                    token_usage=None,
                )

            loop._run_dual_lane_final_response = fake_final_response  # type: ignore[method-assign]

            keep_listening = loop._run_single_text_turn(
                transcript="Wie ist das Wetter heute?",
                listen_source="button",
                proactive_trigger=None,
            )
            time.sleep(0.24)

        self.assertTrue(keep_listening)
        self.assertEqual(tts_provider.stream_calls[:1], ["Heute wird es sonnig."])
        self.assertNotIn("Einen Moment bitte.", tts_provider.stream_calls)
        self.assertIn("first_word_timeout=true", lines)
        self.assertEqual(runtime.last_response, "Heute wird es sonnig.")

    def test_final_lane_hard_timeout_raises_error_and_stops_search_feedback(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                streaming_bridge_reply_timeout_ms=10,
                streaming_final_lane_watchdog_timeout_ms=30,
                streaming_final_lane_hard_timeout_ms=80,
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            tts_provider = FakeTextToSpeechProvider(config)
            lines: list[str] = []
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=tts_provider,
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
                emit=lines.append,
            )
            feedback_events: list[str] = []
            active_feedback_stop: dict[str, object | None] = {"callback": None}

            def start_search_feedback():
                feedback_events.append("start")

                def stop() -> None:
                    feedback_events.append("stop")

                active_feedback_stop["callback"] = stop
                return stop

            def stop_search_feedback() -> None:
                callback = active_feedback_stop.get("callback")
                active_feedback_stop["callback"] = None
                if callable(callback):
                    callback()

            loop._start_search_feedback_loop = start_search_feedback  # type: ignore[method-assign]
            loop._stop_search_feedback = stop_search_feedback  # type: ignore[method-assign]

            def blocking_final_response(transcript: str, *, turn_instructions: str | None, prefetched_decision=None):
                del transcript, turn_instructions, prefetched_decision
                loop._start_search_feedback_loop()
                time.sleep(0.25)
                return SimpleNamespace(
                    text="Zu spät.",
                    response_id="resp_final",
                    request_id="req_final",
                    rounds=1,
                    tool_calls=(),
                    used_web_search=True,
                    model="gpt-4o-mini",
                    token_usage=None,
                )

            loop._run_dual_lane_final_response = blocking_final_response  # type: ignore[method-assign]

            with self.assertRaises(FinalLaneTimeoutError) as raised:
                loop._run_single_text_turn(
                    transcript="Wie ist das Wetter heute?",
                    listen_source="button",
                    proactive_trigger=None,
                )
            loop._handle_error(raised.exception)

        self.assertEqual(lines.count("first_word_timeout=true"), 1)
        self.assertIn("final_lane_watchdog=true", lines)
        self.assertIn("final_lane_timeout=true", lines)
        self.assertEqual(tts_provider.stream_calls, [])
        self.assertEqual(dual_lane.recovery_calls, [])
        self.assertEqual(runtime.status.value, "waiting")
        self.assertEqual(feedback_events, ["start", "stop"])

    def test_final_lane_timeout_does_not_open_recovery_fast_topic_context(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                streaming_bridge_reply_timeout_ms=10,
                streaming_final_lane_watchdog_timeout_ms=30,
                streaming_final_lane_hard_timeout_ms=80,
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            tts_provider = FakeTextToSpeechProvider(config)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=tts_provider,
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            supervisor_context = (("system", "fast lane"),)
            runtime.supervisor_provider_conversation_context = lambda: supervisor_context  # type: ignore[method-assign]
            runtime.supervisor_direct_provider_conversation_context = lambda transcript: (_ for _ in ()).throw(  # type: ignore[method-assign]
                LongTermRemoteUnavailableError("Required remote long-term fast-topic retrieval failed.")
            )

            def blocking_final_response(transcript: str, *, turn_instructions: str | None, prefetched_decision=None):
                del transcript, turn_instructions, prefetched_decision
                time.sleep(0.25)
                return SimpleNamespace(
                    text="Zu spät.",
                    response_id="resp_final",
                    request_id="req_final",
                    rounds=1,
                    tool_calls=(),
                    used_web_search=True,
                    model="gpt-4o-mini",
                    token_usage=None,
                )

            loop._run_dual_lane_final_response = blocking_final_response  # type: ignore[method-assign]

            with self.assertRaises(FinalLaneTimeoutError):
                loop._run_single_text_turn(
                    transcript="Was sind die heutigen Nachrichten?",
                    listen_source="button",
                    proactive_trigger=None,
                )

        self.assertEqual(runtime.status.value, "processing")
        self.assertEqual(dual_lane.recovery_calls, [])

    def test_bridge_filler_completion_reenters_processing_before_final_answer(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                streaming_bridge_reply_timeout_ms=20,
                streaming_final_lane_watchdog_timeout_ms=400,
                streaming_final_lane_hard_timeout_ms=1000,
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            tts_provider = FakeTextToSpeechProvider(config)
            release_first_playback = Event()
            player = BlockingFirstPlaybackPlayer(release_first_playback=release_first_playback)
            lines: list[str] = []
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=tts_provider,
                player=player,
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
                emit=lines.append,
            )
            loop.first_word_provider = DelayedFirstWordProvider(
                config,
                delay_s=0.01,
                reply=FirstWordReply(mode="filler", spoken_text="Ich schaue kurz nach."),
            )

            def fake_final_response(transcript: str, *, turn_instructions: str | None, prefetched_decision=None):
                del transcript, turn_instructions, prefetched_decision
                if not player.first_playback_started.wait(timeout=1.0):
                    raise AssertionError("bridge playback never started")
                release_first_playback.set()
                return SimpleNamespace(
                    text="Hier sind die heutigen Nachrichten.",
                    response_id="resp_final",
                    request_id="req_final",
                    rounds=1,
                    tool_calls=(),
                    used_web_search=True,
                    model="gpt-4o-mini",
                    token_usage=None,
                )

            loop._run_dual_lane_final_response = fake_final_response  # type: ignore[method-assign]

            keep_listening = loop._run_single_text_turn(
                transcript="Was gibt es heute fuer Nachrichten?",
                listen_source="button",
                proactive_trigger=None,
            )

        self.assertTrue(keep_listening)
        self.assertIn("bridge_ack_completed_while_final_lane_running=true", lines)
        processing_indexes = [index for index, line in enumerate(lines) if line == "status=processing"]
        answering_indexes = [index for index, line in enumerate(lines) if line == "status=answering"]
        self.assertGreaterEqual(len(processing_indexes), 2)
        self.assertTrue(answering_indexes)
        self.assertGreater(processing_indexes[-1], answering_indexes[0])
        self.assertEqual(runtime.status.value, "waiting")

    def test_required_remote_final_lane_error_enters_runtime_error_without_recovery(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=True,
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            runtime.reset_error()

            def _fatal_final_response(transcript: str, *, turn_instructions: str | None, prefetched_decision=None):
                del transcript, turn_instructions, prefetched_decision
                raise LongTermRemoteUnavailableError("Required remote long-term fast-topic retrieval failed.")

            loop._run_dual_lane_final_response = _fatal_final_response  # type: ignore[method-assign]

            with self.assertRaises(LongTermRemoteUnavailableError) as raised:
                loop._run_single_text_turn(
                    transcript="Was sind die heutigen Nachrichten?",
                    listen_source="button",
                    proactive_trigger=None,
                )
            loop._handle_error(raised.exception)

        self.assertEqual(runtime.status.value, "error")
        self.assertEqual(dual_lane.recovery_calls, [])

    def test_streaming_turn_builds_lane_plan_after_transcript_submission(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=CapturingDualLaneLoop(),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            observed: dict[str, str] = {}

            def fake_build_lane_plan(transcript: str) -> StreamingTurnLanePlan:
                observed["transcript"] = transcript
                observed["last_transcript"] = runtime.last_transcript

                def _run_single_lane(_on_text_delta):
                    return SimpleNamespace(
                        text="Heute wird es sonnig.",
                        response_id="resp_final",
                        request_id="req_final",
                        rounds=1,
                        tool_calls=(),
                        used_web_search=True,
                        model="gpt-4o-mini",
                        token_usage=None,
                    )

                return StreamingTurnLanePlan(
                    turn_instructions="test",
                    run_single_lane=_run_single_lane,
                )

            loop._build_streaming_turn_lane_plan = fake_build_lane_plan  # type: ignore[method-assign]

            keep_listening = loop._run_single_text_turn(
                transcript="Wie ist das Wetter heute?",
                listen_source="button",
                proactive_trigger=None,
            )

        self.assertTrue(keep_listening)
        self.assertEqual(observed["transcript"], "Wie ist das Wetter heute?")
        self.assertEqual(observed["last_transcript"], "Wie ist das Wetter heute?")

    def test_dual_lane_streaming_runner_passes_slim_supervisor_context(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                streaming_supervisor_context_turns=2,
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            runtime.memory.remember("user", "Alter Turn eins")
            runtime.memory.remember("assistant", "Alter Turn zwei")
            runtime.memory.remember("user", "Letzte Frage")
            runtime.memory.remember("assistant", "Letzte Antwort")
            dual_lane = CapturingDualLaneLoop()

            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )

            loop._run_single_text_turn(
                transcript="Wie wird das Wetter morgen?",
                listen_source="button",
                proactive_trigger=None,
            )

        call = dual_lane.run_calls[0]
        supervisor_context = call["supervisor_conversation"]
        specialist_context = call["conversation"]
        self.assertEqual(
            [(role, content) for role, content in supervisor_context if role != "system"],
            [("user", "Letzte Frage"), ("assistant", "Letzte Antwort")],
        )
        self.assertGreater(len(specialist_context), len(supervisor_context))

    def test_search_handoff_avoids_heavy_tool_context(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            runtime.tool_provider_conversation_context = lambda: (_ for _ in ()).throw(AssertionError("heavy tool context must not be requested"))  # type: ignore[method-assign]
            decision = SimpleNamespace(
                action="handoff",
                spoken_ack="Ich schaue kurz nach.",
                spoken_reply=None,
                kind="search",
                goal="Check the weather.",
                allow_web_search=True,
                response_id="decision_resp",
                request_id="decision_req",
                model="gpt-4o-mini",
                token_usage=None,
            )
            loop._consume_speculative_supervisor_decision = lambda transcript: None  # type: ignore[method-assign]
            dual_lane.supervisor_decision_provider = object()  # type: ignore[attr-defined]
            dual_lane.resolve_supervisor_decision = lambda *args, **kwargs: decision  # type: ignore[method-assign]

            result = loop._run_dual_lane_final_response(
                "Wie wird das Wetter heute?",
                turn_instructions=None,
            )

        self.assertEqual(result.text, "Heute wird es sonnig.")
        self.assertEqual(len(dual_lane.run_handoff_calls), 1)
        self.assertEqual(len(dual_lane.run_calls), 0)

    def test_sync_automation_handoff_short_circuits_to_tool_handoff_only(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            tool_context = (("system", "tool memory"), ("user", "Licht und Geraetestatus."),)
            runtime.search_provider_conversation_context = lambda: (_ for _ in ()).throw(AssertionError("search context must not be requested for automation handoff"))  # type: ignore[method-assign]
            runtime.tool_provider_conversation_context = lambda: tool_context  # type: ignore[method-assign]
            decision = SimpleNamespace(
                action="handoff",
                spoken_ack="Ich kuemmere mich darum.",
                spoken_reply=None,
                kind="automation",
                goal="Check smart home device state and reply clearly.",
                allow_web_search=False,
                response_id="decision_resp",
                request_id="decision_req",
                model="gpt-4o-mini",
                token_usage=None,
            )
            loop._consume_speculative_supervisor_decision = lambda transcript: None  # type: ignore[method-assign]
            dual_lane.supervisor_decision_provider = object()  # type: ignore[attr-defined]
            dual_lane.resolve_supervisor_decision = lambda *args, **kwargs: decision  # type: ignore[method-assign]

            result = loop._run_dual_lane_final_response(
                "Ist das Licht im Flur an?",
                turn_instructions=None,
            )

        self.assertEqual(result.text, "Heute wird es sonnig.")
        self.assertEqual(len(dual_lane.run_handoff_calls), 1)
        self.assertEqual(len(dual_lane.run_calls), 0)
        self.assertEqual(dual_lane.run_handoff_calls[0]["conversation"], tool_context)
        self.assertEqual(dual_lane.run_handoff_calls[0]["specialist_conversation"], tool_context)
        self.assertFalse(dual_lane.run_handoff_calls[0]["emit_filler"])

    def test_tiny_recent_tool_handoff_avoids_heavy_tool_context(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            tiny_recent_context = (("system", "compact tool context"), ("user", "Bitte sei ruhig."),)
            runtime.search_provider_conversation_context = lambda: (_ for _ in ()).throw(AssertionError("search context must not be requested for local tool handoff"))  # type: ignore[method-assign]
            runtime.tool_provider_conversation_context = lambda: (_ for _ in ()).throw(AssertionError("heavy tool context must not be requested"))  # type: ignore[method-assign]
            runtime.tool_provider_tiny_recent_conversation_context = lambda: tiny_recent_context  # type: ignore[method-assign]
            decision = SimpleNamespace(
                action="handoff",
                spoken_ack="Ich kuemmere mich darum.",
                spoken_reply=None,
                kind="general",
                goal="Inspect local runtime state and handle the request.",
                allow_web_search=False,
                context_scope="tiny_recent",
                response_id="decision_resp",
                request_id="decision_req",
                model="gpt-4o-mini",
                token_usage=None,
            )
            loop._consume_speculative_supervisor_decision = lambda transcript: None  # type: ignore[method-assign]
            dual_lane.supervisor_decision_provider = object()  # type: ignore[attr-defined]
            dual_lane.resolve_supervisor_decision = lambda *args, **kwargs: decision  # type: ignore[method-assign]

            result = loop._run_dual_lane_final_response(
                "Bist du ruhig?",
                turn_instructions=None,
            )

        self.assertEqual(result.text, "Heute wird es sonnig.")
        self.assertEqual(len(dual_lane.run_handoff_calls), 1)
        self.assertEqual(dual_lane.run_handoff_calls[0]["conversation"], tiny_recent_context)
        self.assertEqual(dual_lane.run_handoff_calls[0]["specialist_conversation"], tiny_recent_context)

    def test_full_context_direct_final_lane_uses_tool_context(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            tool_context = (("system", "tool memory"), ("user", "Vorhin sprachen wir ueber Arzt und Wetter."))
            supervisor_context = (("system", "fast lane"),)
            direct_context = (("system", "full memory"), ("user", "Wir haben heute ueber Arzt und Wetter gesprochen."))
            runtime.search_provider_conversation_context = lambda: (_ for _ in ()).throw(AssertionError("search context must not be requested for full-context direct turns"))  # type: ignore[method-assign]
            runtime.tool_provider_conversation_context = lambda: tool_context  # type: ignore[method-assign]
            runtime.supervisor_provider_conversation_context = lambda: supervisor_context  # type: ignore[method-assign]
            runtime.supervisor_direct_provider_conversation_context = lambda transcript: direct_context  # type: ignore[method-assign]
            decision = SimpleNamespace(
                action="direct",
                spoken_ack="Ich fasse unser heutiges Gespräch kurz zusammen.",
                spoken_reply="Ich erinnere mich.",
                kind="memory",
                goal="Recall what Twinr and the user discussed earlier today.",
                allow_web_search=None,
                context_scope="full_context",
                response_id="decision_resp",
                request_id="decision_req",
                model="gpt-4o-mini",
                token_usage=None,
            )
            loop._consume_speculative_supervisor_decision = lambda transcript: None  # type: ignore[method-assign]
            dual_lane.supervisor_decision_provider = object()  # type: ignore[attr-defined]
            dual_lane.resolve_supervisor_decision = lambda *args, **kwargs: decision  # type: ignore[method-assign]

            result = loop._run_dual_lane_final_response(
                "Worueber haben wir heute geredet?",
                turn_instructions=None,
            )

        self.assertEqual(result.text, "Ich schaue kurz nach.\nMorgen wird es sonnig.")
        self.assertEqual(len(dual_lane.run_calls), 1)
        self.assertEqual(dual_lane.run_calls[0]["conversation"], tool_context)
        self.assertEqual(dual_lane.run_calls[0]["supervisor_conversation"], direct_context)
        self.assertIs(dual_lane.run_calls[0]["prefetched_decision"], decision)
        self.assertEqual(len(dual_lane.run_handoff_calls), 0)

    def test_direct_final_lane_reresolves_with_memory_aware_supervisor_context(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            supervisor_context = (("system", "fast lane"),)
            direct_context = (("system", "full memory"), ("user", "Heute ging es um Medikamente und Wetter."))
            runtime.supervisor_provider_conversation_context = lambda: supervisor_context  # type: ignore[method-assign]
            runtime.supervisor_direct_provider_conversation_context = lambda transcript: direct_context  # type: ignore[method-assign]
            runtime.search_provider_conversation_context = lambda: (("system", "search"),)  # type: ignore[method-assign]
            runtime.tool_provider_conversation_context = lambda: (("system", "tool"),)  # type: ignore[method-assign]
            loop._consume_speculative_supervisor_decision = lambda transcript: None  # type: ignore[method-assign]
            dual_lane.supervisor_decision_provider = object()  # type: ignore[attr-defined]
            resolve_calls: list[dict[str, object]] = []
            decisions = [
                SimpleNamespace(
                    action="direct",
                    spoken_ack=None,
                    spoken_reply="Ja, wir haben heute gesprochen.",
                    kind="memory",
                    goal="Recall what Twinr and the user discussed earlier today.",
                    allow_web_search=None,
                    context_scope="tiny_recent",
                    response_id="route_resp",
                    request_id="route_req",
                    model="gpt-4o-mini",
                    token_usage=None,
                ),
                SimpleNamespace(
                    action="direct",
                    spoken_ack=None,
                    spoken_reply="Heute haben wir über Medikamente und das Wetter gesprochen.",
                    kind="memory",
                    goal="Recall what Twinr and the user discussed earlier today.",
                    allow_web_search=None,
                    context_scope="tiny_recent",
                    response_id="final_resp",
                    request_id="final_req",
                    model="gpt-4o-mini",
                    token_usage=None,
                ),
            ]

            def fake_resolve(
                prompt: str,
                *,
                conversation=None,
                instructions=None,
                prefetched_decision=None,
                should_stop=None,
            ):
                del instructions, prefetched_decision, should_stop
                resolve_calls.append({"prompt": prompt, "conversation": conversation})
                return decisions[len(resolve_calls) - 1]

            def fake_run(prompt: str, **kwargs):
                dual_lane.run_calls.append({"prompt": prompt, **kwargs})
                decision = kwargs.get("prefetched_decision")
                return SimpleNamespace(
                    text=getattr(decision, "spoken_reply", ""),
                    response_id=getattr(decision, "response_id", None),
                    request_id=getattr(decision, "request_id", None),
                    rounds=1,
                    tool_calls=(),
                    used_web_search=False,
                    model=getattr(decision, "model", None),
                    token_usage=getattr(decision, "token_usage", None),
                )

            dual_lane.resolve_supervisor_decision = fake_resolve  # type: ignore[method-assign]
            dual_lane.run = fake_run  # type: ignore[method-assign]

            result = loop._run_dual_lane_final_response(
                "Worueber haben wir heute geredet?",
                turn_instructions=None,
            )

        self.assertEqual(
            [call["conversation"] for call in resolve_calls],
            [supervisor_context, direct_context],
        )
        self.assertEqual(
            result.text,
            "Heute haben wir über Medikamente und das Wetter gesprochen.",
        )
        self.assertEqual(len(dual_lane.run_calls), 1)
        self.assertIs(dual_lane.run_calls[0]["prefetched_decision"], decisions[1])

    def test_run_dual_lane_final_response_falls_back_to_structured_supervisor_after_shared_wait_miss(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
                streaming_bridge_reply_timeout_ms=25,
                streaming_supervisor_prefetch_wait_ms=5,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            runtime.supervisor_provider_conversation_context = lambda: (("system", "fast lane"),)  # type: ignore[method-assign]
            runtime.tool_provider_conversation_context = lambda: (("system", "tool"),)  # type: ignore[method-assign]
            runtime.search_provider_conversation_context = lambda: (("system", "search"),)  # type: ignore[method-assign]
            dual_lane.supervisor_decision_provider = object()  # type: ignore[attr-defined]
            resolve_calls: list[dict[str, object]] = []

            def fake_resolve(
                prompt: str,
                *,
                conversation=None,
                instructions=None,
                prefetched_decision=None,
                should_stop=None,
            ):
                del instructions, prefetched_decision, should_stop
                resolve_calls.append({"prompt": prompt, "conversation": conversation})
                return SimpleNamespace(
                    action="handoff",
                    spoken_ack="Ich schaue kurz nach.",
                    spoken_reply=None,
                    kind="search",
                    goal="Use fresh web information for the user.",
                    allow_web_search=True,
                    location_hint=None,
                    date_context=None,
                    context_scope=None,
                    response_id="decision_resp",
                    request_id="decision_req",
                    model="gpt-4o-mini",
                    token_usage=None,
                )

            def fake_run(prompt: str, **kwargs):
                dual_lane.run_calls.append({"prompt": prompt, **kwargs})
                return SimpleNamespace(
                    text="Hier ist die Antwort.",
                    response_id="resp_generic",
                    request_id="req_generic",
                    rounds=1,
                    tool_calls=(),
                    used_web_search=False,
                    model="gpt-4o-mini",
                    token_usage=None,
                )

            loop._has_shared_speculative_supervisor_decision = lambda transcript: True  # type: ignore[method-assign]
            loop._wait_for_speculative_supervisor_decision = lambda transcript, wait_ms=None: None  # type: ignore[method-assign]
            dual_lane.resolve_supervisor_decision = fake_resolve  # type: ignore[method-assign]
            dual_lane.run = fake_run  # type: ignore[method-assign]

            result = loop._run_dual_lane_final_response(
                "Was gibt es heute Neues?",
                turn_instructions=None,
            )

        self.assertEqual(result.text, "Heute wird es sonnig.")
        self.assertEqual(
            resolve_calls,
            [{"prompt": "Was gibt es heute Neues?", "conversation": (("system", "fast lane"),)}],
        )
        self.assertEqual(len(dual_lane.run_calls), 0)
        self.assertEqual(len(dual_lane.run_handoff_calls), 1)
        self.assertEqual(dual_lane.run_handoff_calls[0]["conversation"], (("system", "search"),))

    def test_prefetched_memory_handoff_uses_tool_context(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            tool_context = (("system", "tool memory"), ("user", "Vorhin sprachen wir ueber Medikamente."))
            runtime.search_provider_conversation_context = lambda: (_ for _ in ()).throw(AssertionError("search context must not be requested for memory handoffs"))  # type: ignore[method-assign]
            runtime.tool_provider_conversation_context = lambda: tool_context  # type: ignore[method-assign]
            loop._consume_speculative_supervisor_decision = lambda transcript: SimpleNamespace(  # type: ignore[method-assign]
                action="handoff",
                spoken_ack="Einen Moment bitte.",
                spoken_reply=None,
                kind="memory",
                goal="Recall what Twinr and the user discussed earlier today.",
                allow_web_search=None,
                context_scope="full_context",
                response_id="decision_resp",
                request_id="decision_req",
                model="gpt-4o-mini",
                token_usage=None,
            )

            result = loop._run_dual_lane_final_response(
                "Worueber haben wir heute geredet?",
                turn_instructions=None,
            )

        self.assertEqual(result.text, "Heute wird es sonnig.")
        self.assertEqual(len(dual_lane.run_handoff_calls), 1)
        self.assertEqual(dual_lane.run_handoff_calls[0]["conversation"], tool_context)
        self.assertEqual(dual_lane.run_handoff_calls[0]["specialist_conversation"], tool_context)
        self.assertEqual(len(dual_lane.run_calls), 0)

    def test_local_semantic_router_web_route_short_circuits_to_search_handoff(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            search_context = (("system", "search memory"),)
            runtime.search_provider_conversation_context = lambda: search_context  # type: ignore[method-assign]
            runtime.tool_provider_conversation_context = lambda: (_ for _ in ()).throw(AssertionError("tool context must not be requested for web handoff"))  # type: ignore[method-assign]
            loop._resolve_local_semantic_route = lambda transcript: SimpleNamespace(  # type: ignore[method-assign]
                route_decision=SimpleNamespace(label="web"),
                supervisor_decision=SimpleNamespace(
                    action="handoff",
                    spoken_ack="Ich schaue kurz nach.",
                    spoken_reply=None,
                    kind="search",
                    goal="Use fresh external information and reply clearly.",
                    allow_web_search=True,
                    context_scope=None,
                    response_id="route_resp",
                    request_id="route_req",
                    model="local-router",
                    token_usage=None,
                ),
                bridge_reply=FirstWordReply(mode="filler", spoken_text="Ich schaue kurz nach."),
            )
            loop._consume_speculative_supervisor_decision = lambda transcript: (_ for _ in ()).throw(AssertionError("supervisor prefetch should not be consumed when the local router is authoritative"))  # type: ignore[method-assign]

            lane_plan = loop._build_streaming_turn_lane_plan("Wie ist die Lage heute?")
            result = lane_plan.run_final_lane()

        self.assertEqual(lane_plan.prefetched_first_word_source, "local_semantic_router")
        self.assertEqual(lane_plan.prefetched_first_word.spoken_text, "Ich schaue kurz nach.")
        self.assertEqual(result.text, "Heute wird es sonnig.")
        self.assertEqual(len(dual_lane.run_handoff_calls), 1)
        self.assertEqual(dual_lane.run_handoff_calls[0]["conversation"], search_context)
        self.assertEqual(dual_lane.run_handoff_calls[0]["specialist_conversation"], search_context)

    def test_run_dual_lane_final_response_uses_local_tool_route_when_available(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            tool_context = (("system", "compact tool context"),)
            runtime.search_provider_conversation_context = lambda: (_ for _ in ()).throw(AssertionError("search context must not be requested for generic tool handoff"))  # type: ignore[method-assign]
            runtime.tool_provider_conversation_context = lambda: (_ for _ in ()).throw(AssertionError("heavy tool context must not be requested for tiny_recent local tool handoff"))  # type: ignore[method-assign]
            runtime.tool_provider_tiny_recent_conversation_context = lambda: tool_context  # type: ignore[method-assign]
            loop._resolve_local_semantic_route = lambda transcript: SimpleNamespace(  # type: ignore[method-assign]
                route_decision=SimpleNamespace(label="tool"),
                supervisor_decision=SimpleNamespace(
                    action="handoff",
                    spoken_ack="Ich kuemmere mich darum.",
                    spoken_reply=None,
                    kind="general",
                    goal="Use the appropriate Twinr tools or device actions to handle the request.",
                    allow_web_search=False,
                    context_scope="tiny_recent",
                    response_id="route_resp",
                    request_id="route_req",
                    model="local-router",
                    token_usage=None,
                ),
                bridge_reply=FirstWordReply(mode="filler", spoken_text="Ich kuemmere mich darum."),
            )
            loop._consume_speculative_supervisor_decision = lambda transcript: (_ for _ in ()).throw(AssertionError("supervisor prefetch should not be consumed when the local router is authoritative"))  # type: ignore[method-assign]

            result = loop._run_dual_lane_final_response(
                "Mach bitte das Licht an.",
                turn_instructions=None,
            )

        self.assertEqual(result.text, "Heute wird es sonnig.")
        self.assertEqual(len(dual_lane.run_handoff_calls), 1)
        self.assertEqual(dual_lane.run_handoff_calls[0]["conversation"], tool_context)
        self.assertEqual(dual_lane.run_handoff_calls[0]["specialist_conversation"], tool_context)

    def test_runtime_local_tool_handoff_skips_specialist_loop(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            runtime.search_provider_conversation_context = lambda: (_ for _ in ()).throw(AssertionError("search context must not be requested for runtime-local direct tool handoff"))  # type: ignore[method-assign]
            runtime.tool_provider_conversation_context = lambda: (_ for _ in ()).throw(AssertionError("heavy tool context must not be requested for runtime-local direct tool handoff"))  # type: ignore[method-assign]
            runtime.tool_provider_tiny_recent_conversation_context = lambda: (_ for _ in ()).throw(AssertionError("tiny_recent tool context must not be requested once the supervisor already resolved the direct runtime-local tool"))  # type: ignore[method-assign]
            loop._resolve_local_semantic_route = lambda transcript: None  # type: ignore[method-assign]
            loop._consume_speculative_supervisor_decision = lambda transcript: SimpleNamespace(  # type: ignore[method-assign]
                action="handoff",
                spoken_ack="Ich schalte mich kurz stumm.",
                spoken_reply=None,
                kind="automation",
                goal="Set temporary quiet mode.",
                allow_web_search=False,
                context_scope="tiny_recent",
                runtime_tool_name="manage_voice_quiet_mode",
                runtime_tool_arguments={"action": "set", "duration_minutes": 2},
                response_id="decision_resp",
                request_id="decision_req",
                model="gpt-4o-mini",
                token_usage=None,
            )

            result = loop._run_dual_lane_final_response(
                "Sei bitte 2 Minuten ruhig.",
                turn_instructions=None,
            )

        self.assertEqual(result.text, "Okay. Ich bin jetzt für 2 Minuten ruhig.")
        self.assertEqual(len(dual_lane.run_runtime_local_tool_only_calls), 1)
        self.assertEqual(len(dual_lane.run_handoff_calls), 0)
        self.assertEqual(len(dual_lane.run_calls), 0)

    def test_runtime_local_tool_handoff_omits_supervisor_bridge_ack_in_full_turn(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            tts_provider = FakeTextToSpeechProvider(config)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=tts_provider,
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            runtime.search_provider_conversation_context = lambda: (_ for _ in ()).throw(AssertionError("search context must not be requested for runtime-local direct tool handoff"))  # type: ignore[method-assign]
            runtime.tool_provider_conversation_context = lambda: (_ for _ in ()).throw(AssertionError("heavy tool context must not be requested for runtime-local direct tool handoff"))  # type: ignore[method-assign]
            runtime.tool_provider_tiny_recent_conversation_context = lambda: (_ for _ in ()).throw(AssertionError("tiny_recent tool context must not be requested once the supervisor already resolved the direct runtime-local tool"))  # type: ignore[method-assign]
            loop._resolve_local_semantic_route = lambda transcript: None  # type: ignore[method-assign]
            loop._consume_speculative_supervisor_decision = lambda transcript: SimpleNamespace(  # type: ignore[method-assign]
                action="handoff",
                spoken_ack="Ich schalte mich kurz stumm.",
                spoken_reply=None,
                kind="automation",
                goal="Set temporary quiet mode.",
                allow_web_search=False,
                context_scope="tiny_recent",
                runtime_tool_name="manage_voice_quiet_mode",
                runtime_tool_arguments={"action": "set", "duration_minutes": 2},
                response_id="decision_resp",
                request_id="decision_req",
                model="gpt-4o-mini",
                token_usage=None,
            )

            keep_listening = loop._run_single_text_turn(
                transcript="Sei bitte 2 Minuten ruhig.",
                listen_source="button",
                proactive_trigger=None,
            )

        self.assertTrue(keep_listening)
        self.assertEqual(tts_provider.stream_calls, ["Okay. Ich bin jetzt für 2 Minuten ruhig."])
        self.assertEqual(runtime.last_response, "Okay. Ich bin jetzt für 2 Minuten ruhig.")
        self.assertEqual(len(dual_lane.run_runtime_local_tool_only_calls), 1)
        self.assertFalse(dual_lane.run_runtime_local_tool_only_calls[0]["emit_filler"])
        self.assertEqual(len(dual_lane.run_handoff_calls), 0)
        self.assertEqual(len(dual_lane.run_calls), 0)

    def test_local_semantic_router_tool_route_synthesizes_tiny_recent_handoff(self) -> None:
        decision = _synthesize_supervisor_decision(
            SimpleNamespace(
                label="tool",
                confidence=0.93,
                margin=0.41,
                model_id="router-v1",
            ),
            FirstWordReply(mode="filler", spoken_text="Ich kuemmere mich darum."),
        )

        self.assertEqual(decision.action, "handoff")
        self.assertEqual(decision.kind, "general")
        self.assertEqual(decision.context_scope, "tiny_recent")

    def test_local_semantic_router_bridge_reply_uses_route_aware_first_word_overlay(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                deepgram_stt_language="de",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=CapturingDualLaneLoop(),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            observed: dict[str, str | None] = {"transcript": None, "instructions": None}

            def _fake_generate_first_word_reply(transcript: str, *, instructions: str | None = None):
                observed["transcript"] = transcript
                observed["instructions"] = instructions
                return FirstWordReply(mode="direct", spoken_text="Ich schaue kurz in meinen Erinnerungen nach.")

            loop._generate_first_word_reply = _fake_generate_first_word_reply  # type: ignore[method-assign]

            reply = loop._streaming_semantic_router._build_bridge_reply(  # type: ignore[attr-defined]
                "Was habe ich heute fuer Termine?",
                SimpleNamespace(label="memory"),
            )

        self.assertEqual(observed["transcript"], "Was habe ich heute fuer Termine?")
        self.assertIsNotNone(observed["instructions"])
        assert observed["instructions"] is not None
        self.assertIn("Return mode filler only", observed["instructions"])
        self.assertIn("recalling or checking remembered details", observed["instructions"])
        self.assertIn("natural German", observed["instructions"])
        self.assertEqual(reply.mode, "filler")
        self.assertEqual(reply.spoken_text, "Ich schaue kurz in meinen Erinnerungen nach.")

    def test_local_semantic_router_bridge_reply_stays_empty_without_fast_first_word_text(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                streaming_first_word_enabled=True,
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=CapturingDualLaneLoop(),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            loop._generate_first_word_reply = lambda transcript, *, instructions=None: None  # type: ignore[method-assign]

            reply = loop._streaming_semantic_router._build_bridge_reply(  # type: ignore[attr-defined]
                "Wie ist die Lage heute?",
                SimpleNamespace(label="web"),
            )

        self.assertIsNone(reply)

    def test_local_semantic_router_web_route_handoff_allows_missing_bridge_reply(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            search_context = (("system", "search memory"),)
            runtime.search_provider_conversation_context = lambda: search_context  # type: ignore[method-assign]
            runtime.tool_provider_conversation_context = lambda: (_ for _ in ()).throw(AssertionError("tool context must not be requested for web handoff"))  # type: ignore[method-assign]
            loop._resolve_local_semantic_route = lambda transcript: SimpleNamespace(  # type: ignore[method-assign]
                route_decision=SimpleNamespace(label="web"),
                supervisor_decision=SimpleNamespace(
                    action="handoff",
                    spoken_ack=None,
                    spoken_reply=None,
                    kind="search",
                    goal="Use fresh external information and reply clearly.",
                    allow_web_search=True,
                    context_scope=None,
                    response_id="route_resp",
                    request_id="route_req",
                    model="local-router",
                    token_usage=None,
                ),
                bridge_reply=None,
            )
            loop._consume_speculative_supervisor_decision = lambda transcript: (_ for _ in ()).throw(AssertionError("supervisor prefetch should not be consumed when the local router is authoritative"))  # type: ignore[method-assign]

            lane_plan = loop._build_streaming_turn_lane_plan("Wie ist die Lage heute?")
            result = lane_plan.run_final_lane()

        self.assertEqual(lane_plan.prefetched_first_word_source, "none")
        self.assertIsNone(lane_plan.prefetched_first_word)
        self.assertEqual(result.text, "Heute wird es sonnig.")
        self.assertEqual(len(dual_lane.run_handoff_calls), 1)
        self.assertEqual(dual_lane.run_handoff_calls[0]["conversation"], search_context)
        self.assertEqual(dual_lane.run_handoff_calls[0]["specialist_conversation"], search_context)

    def test_local_semantic_router_shadow_mode_keeps_supervisor_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            tool_context = (("system", "tool memory"),)
            runtime.search_provider_conversation_context = lambda: (_ for _ in ()).throw(AssertionError("search context must not be requested for memory handoff"))  # type: ignore[method-assign]
            runtime.tool_provider_conversation_context = lambda: tool_context  # type: ignore[method-assign]
            loop._resolve_local_semantic_route = lambda transcript: SimpleNamespace(  # type: ignore[method-assign]
                route_decision=SimpleNamespace(label="web"),
                supervisor_decision=None,
                bridge_reply=None,
            )
            loop._consume_speculative_supervisor_decision = lambda transcript: SimpleNamespace(  # type: ignore[method-assign]
                action="handoff",
                spoken_ack="Einen Moment bitte.",
                spoken_reply=None,
                kind="memory",
                goal="Recall what Twinr and the user discussed earlier today.",
                allow_web_search=None,
                context_scope="full_context",
                response_id="decision_resp",
                request_id="decision_req",
                model="gpt-4o-mini",
                token_usage=None,
            )

            result = loop._run_dual_lane_final_response(
                "Worueber haben wir heute geredet?",
                turn_instructions=None,
            )

        self.assertEqual(result.text, "Heute wird es sonnig.")
        self.assertEqual(len(dual_lane.run_handoff_calls), 1)
        self.assertEqual(dual_lane.run_handoff_calls[0]["conversation"], tool_context)
        self.assertEqual(dual_lane.run_handoff_calls[0]["specialist_conversation"], tool_context)

    def test_consume_speculative_supervisor_decision_accepts_direct_reply(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=TwinrRuntime(config=config),
                tool_agent_provider=FakeToolAgentProvider(config),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            loop._speculative_supervisor_started = True
            loop._speculative_supervisor_transcript = "Alles ok"
            loop._speculative_supervisor_decision = StubSupervisorDecision(
                action="direct",
                spoken_reply="Mir geht's gut.",
            )
            getattr(loop, "_speculative_supervisor_done").set()

            decision = loop._consume_speculative_supervisor_decision("Alles ok bei dir?")

        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, "direct")

    def test_direct_supervisor_prefetch_does_not_speak_memory_blind_bridge_reply(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=TwinrRuntime(config=config),
                tool_agent_provider=FakeToolAgentProvider(config),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )

            reply = loop._dual_lane_bridge_reply_from_decision(
                SimpleNamespace(
                    action="direct",
                    spoken_ack=None,
                    spoken_reply="Ja, wir haben heute gesprochen.",
                    context_scope="tiny_recent",
                    response_id="decision_resp",
                    request_id="decision_req",
                    model="gpt-4o-mini",
                    token_usage=None,
                )
            )

        self.assertIsNone(reply)

    def test_streaming_endpoint_can_prime_speculative_supervisor(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=TwinrRuntime(config=config),
                tool_agent_provider=FakeToolAgentProvider(config),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            seen: list[str] = []
            loop._maybe_start_speculative_supervisor_decision = seen.append  # type: ignore[method-assign]

            loop._on_streaming_stt_endpoint(SimpleNamespace(transcript="Alles okay bei dir", event_type="speech_final"))

        self.assertEqual(seen, ["Alles okay bei dir"])

    def test_capture_and_transcribe_streaming_resets_speculative_supervisor_before_capture(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=TwinrRuntime(config=config),
                tool_agent_provider=FakeToolAgentProvider(config),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            sequence: list[str] = []
            expected = (SimpleNamespace(pcm_bytes=b""), "Hallo", 11, 22, None)
            loop._reset_speculative_supervisor_decision = lambda: sequence.append("reset")  # type: ignore[method-assign]
            loop._capture_and_transcribe_with_turn_controller = lambda **kwargs: (  # type: ignore[method-assign]
                sequence.append("capture") or expected
            )

            result = loop._capture_and_transcribe_streaming(
                listening_window=SimpleNamespace(speech_pause_ms=450),
                speech_start_chunks=2,
                ignore_initial_ms=75,
            )

        self.assertEqual(sequence, ["reset", "capture"])
        self.assertIs(result, expected)

    def test_streaming_endpoint_can_prime_speculative_first_word(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=TwinrRuntime(config=config),
                tool_agent_provider=FakeToolAgentProvider(config),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            seen: list[str] = []
            loop._maybe_start_speculative_first_word = seen.append  # type: ignore[method-assign]

            loop._on_streaming_stt_endpoint(SimpleNamespace(transcript="Alles okay bei dir", event_type="speech_final"))

        self.assertEqual(seen, ["Alles okay bei dir"])

    def test_groq_config_uses_compact_tool_schemas_and_instructions(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                groq_api_key="groq-key",
                llm_provider="groq",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
                realtime_sensitive_tools_require_identity=False,
            )
            runtime = TwinrRuntime(config=config)
            tool_agent = FakeToolAgentProvider(config)
            support_provider = FakePrintBackend(config)

            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=tool_agent,
                print_backend=support_provider,
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=support_provider,
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )

            loop._run_single_text_turn(
                transcript="Bitte druck das aus",
                listen_source="button",
                proactive_trigger=None,
            )

        schema_properties = tool_agent.start_calls[0]["tool_schemas"][0]["parameters"]["properties"]
        schemas_by_name = {
            schema["name"]: schema
            for schema in tool_agent.start_calls[0]["tool_schemas"]
        }
        self.assertLess(len(tool_agent.start_calls[0]["tool_schemas"][0]["description"]), 80)
        self.assertTrue(all("description" not in value for value in schema_properties.values()))
        self.assertEqual(set(schema_properties), {"focus_hint", "text"})
        self.assertIn("question", schemas_by_name["search_live_info"]["parameters"]["properties"])
        self.assertIn("Available Twinr spoken voices", tool_agent.start_calls[0]["instructions"])
        self.assertNotIn("Current bounded simple settings", tool_agent.start_calls[0]["instructions"])

    def test_groq_config_streams_final_text_only(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                groq_api_key="groq-key",
                llm_provider="groq",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            tool_agent = FakeToolAgentProvider(config)
            support_provider = FakePrintBackend(config)
            tts_provider = FakeTextToSpeechProvider(config)

            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=tool_agent,
                print_backend=support_provider,
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=support_provider,
                tts_provider=tts_provider,
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )

            loop._run_single_text_turn(
                transcript="Bitte druck das aus",
                listen_source="button",
                proactive_trigger=None,
            )

        self.assertEqual(tts_provider.calls, ["Ist erledigt."])
        self.assertEqual(runtime.last_response, "Ist erledigt.")

    def test_text_turn_uses_processing_feedback_before_answering(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            tool_agent = FakeToolAgentProvider(config)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=tool_agent,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            feedback_kinds: list[str] = []

            def fake_start(kind: str):
                feedback_kinds.append(kind)
                return lambda: None

            loop._start_working_feedback_loop = fake_start  # type: ignore[method-assign]

            loop._run_single_text_turn(
                transcript="Bitte druck das aus",
                listen_source="button",
                proactive_trigger=None,
            )

        self.assertEqual(feedback_kinds[0], "processing")

    def test_tool_provider_context_hides_exact_contact_methods_and_conflicts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            runtime.remember_contact(given_name="Anna", family_name="Schulz", phone=_TEST_CONTACT_PHONE)
            runtime.long_term_memory.object_store.apply_consolidation(
                LongTermConsolidationResultV1(
                    turn_id="turn:2",
                    occurred_at=datetime(2026, 3, 14, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                    episodic_objects=(),
                    durable_objects=(
                        LongTermMemoryObjectV1(
                            memory_id="fact:corinna_phone_old",
                            kind="contact_method_fact",
                            summary=f"Corinna Maier can be reached at {_TEST_CORINNA_PHONE_OLD}.",
                            source=LongTermSourceRefV1(
                                source_type="conversation_turn",
                                event_ids=("turn:1",),
                                speaker="user",
                                modality="voice",
                            ),
                            status="active",
                            confidence=0.95,
                            slot_key="contact:person:corinna_maier:phone",
                            value_key=_TEST_CORINNA_PHONE_OLD,
                        ),
                    ),
                    deferred_objects=(
                        LongTermMemoryObjectV1(
                            memory_id="fact:corinna_phone_new",
                            kind="contact_method_fact",
                            summary=f"Corinna Maier can be reached at {_TEST_CORINNA_PHONE_NEW}.",
                            source=LongTermSourceRefV1(
                                source_type="conversation_turn",
                                event_ids=("turn:2",),
                                speaker="user",
                                modality="voice",
                            ),
                            status="uncertain",
                            confidence=0.92,
                            slot_key="contact:person:corinna_maier:phone",
                            value_key=_TEST_CORINNA_PHONE_NEW,
                        ),
                    ),
                    conflicts=(
                        LongTermMemoryConflictV1(
                            slot_key="contact:person:corinna_maier:phone",
                            candidate_memory_id="fact:corinna_phone_new",
                            existing_memory_ids=("fact:corinna_phone_old",),
                            question="Which phone number should I use for Corinna Maier?",
                            reason="Conflicting phone numbers exist.",
                        ),
                    ),
                    graph_edges=(),
                )
            )
            runtime.last_transcript = "Wie ist die Telefonnummer von Anna Schulz?"
            provider_contact_context = runtime.provider_conversation_context()
            tool_contact_context = runtime.tool_provider_conversation_context()
            supervisor_contact_context = runtime.supervisor_provider_conversation_context()
            runtime.last_transcript = "Gibt es bei Corinna Maier offene Erinnerungskonflikte?"
            provider_conflict_context = runtime.provider_conversation_context()
            tool_conflict_context = runtime.tool_provider_conversation_context()
            supervisor_conflict_context = runtime.supervisor_provider_conversation_context()

        provider_contact_system = "\n".join(content for role, content in provider_contact_context if role == "system")
        tool_contact_system = "\n".join(content for role, content in tool_contact_context if role == "system")
        supervisor_contact_system = "\n".join(content for role, content in supervisor_contact_context if role == "system")
        provider_conflict_system = "\n".join(content for role, content in provider_conflict_context if role == "system")
        tool_conflict_system = "\n".join(content for role, content in tool_conflict_context if role == "system")
        supervisor_conflict_system = "\n".join(content for role, content in supervisor_conflict_context if role == "system")
        self.assertIn(_TEST_CONTACT_PHONE, provider_contact_system)
        self.assertIn("Structured unresolved long-term memory conflicts", provider_conflict_system)
        self.assertIn(_TEST_CORINNA_PHONE_NEW, provider_conflict_system)
        self.assertNotIn(_TEST_CONTACT_PHONE, tool_contact_system)
        self.assertNotIn("Structured unresolved long-term memory conflicts", tool_conflict_system)
        self.assertNotIn(_TEST_CORINNA_PHONE_NEW, tool_conflict_system)
        self.assertNotIn(_TEST_CONTACT_PHONE, supervisor_contact_system)
        self.assertNotIn("Structured unresolved long-term memory conflicts", supervisor_conflict_system)
        self.assertNotIn(_TEST_CORINNA_PHONE_NEW, supervisor_conflict_system)

    def test_supervisor_context_uses_recent_raw_tail_only(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runtime = TwinrRuntime(
                TwinrConfig(
                    openai_api_key="test-key",
                    project_root=temp_dir,
                    personality_dir="personality",
                    memory_max_turns=8,
                    memory_keep_recent=4,
                    streaming_supervisor_context_turns=2,
                )
            )
            runtime.memory.remember("user", "Turn one")
            runtime.memory.remember("assistant", "Turn two")
            runtime.memory.remember("user", "Turn three")
            runtime.memory.remember("assistant", "Turn four")
            context = runtime.supervisor_provider_conversation_context()

        non_system_messages = [(role, content) for role, content in context if role != "system"]
        self.assertEqual(
            non_system_messages,
            [("user", "Turn three"), ("assistant", "Turn four")],
        )

    def test_streaming_conversation_writes_forensic_runpack_when_enabled(self) -> None:
        with TemporaryDirectory() as temp_dir:
            trace_dir = Path(temp_dir) / "state" / "forensics" / "workflow"
            previous_env = {
                key: os.environ.get(key)
                for key in (
                    "TWINR_WORKFLOW_TRACE_ENABLED",
                    "TWINR_WORKFLOW_TRACE_MODE",
                    "TWINR_WORKFLOW_TRACE_DIR",
                )
            }
            os.environ["TWINR_WORKFLOW_TRACE_ENABLED"] = "1"
            os.environ["TWINR_WORKFLOW_TRACE_MODE"] = "forensic"
            os.environ["TWINR_WORKFLOW_TRACE_DIR"] = str(trace_dir)
            try:
                config = TwinrConfig(
                    openai_api_key="test-key",
                    project_root=temp_dir,
                    personality_dir="personality",
                    conversation_follow_up_enabled=False,
                    long_term_memory_query_rewrite_enabled=False,
                )
                runtime = TwinrRuntime(config=config)
                dual_lane = CapturingDualLaneLoop()
                dual_lane.supervisor_decision_provider = SimpleNamespace(
                    decide=lambda *args, **kwargs: None
                )
                dual_lane.supervisor_instructions = "Kurz und klar antworten."
                loop = TwinrStreamingHardwareLoop(
                    config=config,
                    runtime=runtime,
                    tool_agent_provider=FakeToolAgentProvider(config),
                    streaming_turn_loop=dual_lane,
                    print_backend=FakePrintBackend(config),
                    stt_provider=FakeSpeechToTextProvider(config),
                    agent_provider=FakePrintBackend(config),
                    tts_provider=FakeTextToSpeechProvider(config),
                    recorder=FakeRecorder(),
                    player=FakePlayer(),
                    printer=FakePrinter(),
                    voice_profile_monitor=FakeVoiceProfileMonitor(),
                    usage_store=FakeUsageStore(),
                    button_monitor=SimpleNamespace(),
                    proactive_monitor=SimpleNamespace(),
                )
                loop.first_word_provider = FakeFirstWordProvider(config)

                result = loop._run_conversation_session(initial_source="button")
                loop.workflow_forensics.close()

                self.assertTrue(result)
                run_id = (trace_dir / "LATEST").read_text(encoding="utf-8").strip()
                run_dir = trace_dir / run_id
                self.assertTrue((run_dir / "run.jsonl").exists())
                self.assertTrue((run_dir / "run.trace").exists())
                self.assertTrue((run_dir / "run.metrics.json").exists())
                self.assertTrue((run_dir / "run.summary.json").exists())
                self.assertTrue((run_dir / "run.repro" / "runtime.json").exists())
                self.assertTrue((run_dir / "run.repro" / "env.json").exists())

                records = [
                    json.loads(line)
                    for line in (run_dir / "run.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                msgs = {record["msg"] for record in records}
                self.assertGreaterEqual(len(msgs), 20)
                self.assertIn("conversation_session_started", msgs)
                self.assertIn("streaming_audio_turn_started", msgs)
                self.assertIn("streaming_audio_capture_started", msgs)
                self.assertIn("streaming_batch_stt_completed", msgs)
                self.assertIn("streaming_transcript_ready", msgs)
                self.assertIn("streaming_lane_plan_build", msgs)
                self.assertTrue({"supervisor_cache_prewarmed", "dual_lane_context_materialized"} & msgs)
                self.assertIn("streaming_audio_turn_finished", msgs)
                self.assertIn("streaming_first_audio_observed", msgs)
                self.assertIn("runtime_status_emitted", msgs)
            finally:
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
