"""Run a minimal interrupt/follow-up repro harness for the realtime loop.

Purpose
-------
Prove that Twinr safely discards an unheard interrupted assistant reply,
persists the user transcript, and opens the immediate follow-up turn with the
correct conversation context. This harness is intentionally lighter than the
full realtime pytest module so it can be executed directly on the Raspberry Pi.

Usage
-----
Command-line invocation examples::

    PYTHONPATH=src ./.venv/bin/python test/test_realtime_interrupt_follow_up_repro.py
    PYTHONPATH=src ./.venv/bin/python test/test_realtime_interrupt_follow_up_repro.py --json
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig, TwinrRuntime
from twinr.agent.workflows.realtime_runner import TwinrRealtimeHardwareLoop
from twinr.hardware.audio import AmbientAudioCaptureWindow, AmbientAudioLevelSample


_FIRST_TRANSCRIPT = "Hallo Twinr"
_INTERRUPT_TRANSCRIPT = "warte mal"
_FOLLOW_UP_TRANSCRIPT = "Wie spaet ist es?"
_DISCARDED_ASSISTANT_RESPONSE = "Ich bin noch nicht fertig"
_FOLLOW_UP_RESPONSE = "Es ist 12 Uhr."


@dataclass(frozen=True)
class _FakeRealtimeTurn:
    """Carry the minimal realtime-turn attributes the loop consumes."""

    transcript: str
    response_text: str
    response_id: str
    end_conversation: bool
    model: str | None = None
    token_usage: object | None = None


@dataclass(frozen=True)
class ReproResult:
    """Capture the operator-relevant proof emitted by the harness."""

    session_call_count: int
    recorder_start_timeouts: list[float | None]
    interrupt_transcribe_calls: int
    assistant_response_omitted_after_interrupt: bool
    user_interrupt_detected: bool
    assistant_interrupted: bool
    second_conversation: tuple[tuple[str, str], ...] | None
    runtime_conversation: tuple[tuple[str, str], ...] | None
    tail_lines: list[str]


class _PassiveMonitor:
    """Provide a no-op context manager for loop collaborators."""

    def __enter__(self) -> _PassiveMonitor:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb


class _FakeUsageStore:
    """Accept usage appends without touching the operator stores."""

    def __init__(self) -> None:
        self.entries: list[dict[str, object]] = []

    def append(self, **payload: object) -> None:
        """Record the payload so the harness can fail on unexpected writes."""

        self.entries.append(dict(payload))


class _FakeVoiceProfileMonitor:
    """Report that no enrolled speaker matched the synthetic audio."""

    def assess_pcm16(self, audio_pcm: bytes, *, sample_rate: int, channels: int) -> SimpleNamespace:
        """Return a stable voice-assessment stub for the captured audio."""

        del audio_pcm, sample_rate, channels
        return SimpleNamespace(status="not_enrolled", should_persist=False)


class _FakeCombinedBackend:
    """Provide the minimal STT/TTS surface the repro path touches."""

    def __init__(self) -> None:
        self.transcribe_calls: list[tuple[bytes, str | None, str | None]] = []

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        """Return the interrupt transcript regardless of the synthetic audio."""

        del filename, content_type
        self.transcribe_calls.append((audio_bytes, language, prompt))
        return _INTERRUPT_TRANSCRIPT

    def synthesize_stream(self, text: str):
        """Yield a single inert PCM chunk if the loop ever falls back to TTS."""

        del text
        yield b"PCM"

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        response_format: str | None = None,
        instructions: str | None = None,
    ) -> bytes:
        """Return inert WAV-like bytes for unexpected fallback paths."""

        del text, voice, response_format, instructions
        return b"WAVPCM"


class _FakeRecorder:
    """Return two synthetic captures and record the follow-up timeouts."""

    def __init__(self) -> None:
        self.start_timeouts: list[float | None] = []
        self._captures = [b"TURN1", b"TURN2"]

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
    ) -> SimpleNamespace:
        """Return a deterministic capture payload for each listen window."""

        del pause_ms, max_record_seconds, speech_start_chunks, ignore_initial_ms, pause_grace_ms
        self.start_timeouts.append(start_timeout_s)
        if on_chunk is not None:
            on_chunk(b"PCM-A")
            on_chunk(b"PCM-B")
        if should_stop is not None and should_stop():
            return SimpleNamespace(
                pcm_bytes=b"",
                speech_started_after_ms=0,
                resumed_after_pause_count=0,
            )
        pcm_bytes = self._captures.pop(0) if self._captures else b"TURNX"
        return SimpleNamespace(
            pcm_bytes=pcm_bytes,
            speech_started_after_ms=250,
            resumed_after_pause_count=0,
        )


class _FakePlayer:
    """Consume streamed chunks so playback timing stays bounded."""

    def __init__(self) -> None:
        self.played: list[bytes] = []
        self.stop_calls = 0

    def play_pcm16_chunks(
        self,
        chunks,
        *,
        sample_rate: int,
        channels: int = 1,
        should_stop=None,
    ) -> None:
        """Collect the PCM chunks until playback is interrupted or complete."""

        del sample_rate, channels
        rendered = bytearray()
        for chunk in chunks:
            if should_stop is not None and should_stop():
                break
            rendered.extend(chunk)
        self.played.append(bytes(rendered))

    def play_wav_chunks(self, chunks, *, should_stop=None) -> None:
        """Collect streamed WAV chunks for completeness."""

        rendered = bytearray()
        for chunk in chunks:
            if should_stop is not None and should_stop():
                break
            rendered.extend(chunk)
        self.played.append(bytes(rendered))

    def play_wav_bytes(self, audio_bytes: bytes) -> None:
        """Record direct WAV playback requests."""

        self.played.append(audio_bytes)

    def play_tone(
        self,
        *,
        frequency_hz: int = 1046,
        duration_ms: int = 180,
        volume: float = 0.8,
        sample_rate: int = 24000,
    ) -> None:
        """Ignore tone playback while keeping the public surface intact."""

        del frequency_hz, duration_ms, volume, sample_rate

    def stop_playback(self) -> None:
        """Record explicit stop requests triggered by the interrupt path."""

        self.stop_calls += 1


class _FakeAmbientAudioSampler:
    """Return two consecutive active windows so the watcher confirms interrupt."""

    def __init__(self) -> None:
        self.calls = 0
        self._windows = [
            _active_interrupt_window(),
            _active_interrupt_window(),
        ]

    def sample_window(self, *, duration_ms: int | None = None) -> AmbientAudioCaptureWindow:
        """Return the next prepared active window, then silence forever."""

        del duration_ms
        self.calls += 1
        if self._windows:
            return self._windows.pop(0)
        return AmbientAudioCaptureWindow(
            sample=AmbientAudioLevelSample(
                duration_ms=120,
                chunk_count=2,
                active_chunk_count=0,
                average_rms=120,
                peak_rms=180,
                active_ratio=0.0,
            ),
            pcm_bytes=b"\x00\x00" * 1600,
            sample_rate=24_000,
            channels=1,
        )


class _FakePrinter:
    """Accept printed text without touching the real receipt printer."""

    def __init__(self) -> None:
        self.printed: list[str] = []

    def print_text(self, text: str) -> str:
        """Record the text and return a stable fake request identifier."""

        self.printed.append(text)
        return "pi-repro-print-id"


class _FakeCamera:
    """Return a stable fake image payload when camera tooling is touched."""

    def capture_photo(self, *, output_path=None, filename: str = "camera-capture.png") -> SimpleNamespace:
        """Return a minimal fake camera response."""

        del output_path
        return SimpleNamespace(
            data=b"PNG",
            content_type="image/png",
            filename=filename,
            source_device="/dev/video0",
            input_format="bayer_grbg8",
        )


class _InterruptingRealtimeSession:
    """Stream one interrupted answer and one follow-up answer."""

    def __init__(self) -> None:
        self.calls: list[bytes] = []
        self.conversations: list[tuple[tuple[str, str], ...] | None] = []

    def __enter__(self) -> _InterruptingRealtimeSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb

    def run_audio_turn(
        self,
        audio_pcm: bytes,
        *,
        conversation=None,
        on_audio_chunk=None,
        on_output_text_delta=None,
    ) -> _FakeRealtimeTurn:
        """Return the interrupted turn first, then the follow-up turn."""

        call_index = len(self.calls)
        self.calls.append(audio_pcm)
        self.conversations.append(conversation)
        if call_index == 0:
            if on_output_text_delta is not None:
                on_output_text_delta("Ich ")
            if on_audio_chunk is not None:
                on_audio_chunk(b"PCM1")
                time.sleep(0.03)
                on_audio_chunk(b"PCM2")
            time.sleep(0.05)
            return _FakeRealtimeTurn(
                transcript=_FIRST_TRANSCRIPT,
                response_text=_DISCARDED_ASSISTANT_RESPONSE,
                response_id="resp_rt_interrupt",
                end_conversation=False,
            )
        if on_output_text_delta is not None:
            on_output_text_delta("Alles ")
        if on_audio_chunk is not None:
            on_audio_chunk(b"PCM3")
        return _FakeRealtimeTurn(
            transcript=_FOLLOW_UP_TRANSCRIPT,
            response_text=_FOLLOW_UP_RESPONSE,
            response_id="resp_rt_follow_up",
            end_conversation=True,
        )


def _active_interrupt_window() -> AmbientAudioCaptureWindow:
    """Build an active ambient-audio sample that should confirm interruption."""

    return AmbientAudioCaptureWindow(
        sample=AmbientAudioLevelSample(
            duration_ms=120,
            chunk_count=2,
            active_chunk_count=2,
            average_rms=1800,
            peak_rms=2200,
            active_ratio=1.0,
        ),
        pcm_bytes=b"\x01\x02" * 1800,
        sample_rate=24_000,
        channels=1,
    )


def _build_config(temp_root: Path) -> TwinrConfig:
    """Build an isolated config so the harness never touches live Pi state."""

    repo_root = Path(__file__).resolve().parents[1]
    return TwinrConfig(
        openai_api_key="test-key",
        project_root=str(repo_root),
        personality_dir="personality",
        runtime_state_path=str(temp_root / "runtime-state.json"),
        reminder_store_path=str(temp_root / "state" / "reminders.json"),
        memory_markdown_path=str(temp_root / "state" / "MEMORY.md"),
        automation_store_path=str(temp_root / "state" / "automations.json"),
        adaptive_timing_store_path=str(temp_root / "state" / "adaptive-timing.json"),
        long_term_memory_path=str(temp_root / "state" / "chonkydb"),
        display_face_cue_path=str(temp_root / "artifacts" / "stores" / "ops" / "display_face_cue.json"),
        display_presentation_path=str(temp_root / "artifacts" / "stores" / "ops" / "display_presentation.json"),
        display_news_ticker_store_path=str(
            temp_root / "artifacts" / "stores" / "ops" / "display_news_ticker.json"
        ),
        turn_controller_interrupt_enabled=True,
        turn_controller_interrupt_window_ms=120,
        turn_controller_interrupt_poll_ms=10,
        turn_controller_interrupt_min_active_ratio=0.1,
        turn_controller_interrupt_min_transcript_chars=4,
        turn_controller_interrupt_consecutive_windows=2,
        turn_controller_interrupt_min_transcribe_interval_ms=0,
        conversation_follow_up_timeout_s=3.5,
        conversation_closure_guard_enabled=False,
    )


def _conversation_has_message(
    conversation: tuple[tuple[str, str], ...] | None,
    *,
    role: str,
    content: str,
) -> bool:
    """Return whether a conversation contains the exact role/content pair."""

    if conversation is None:
        return False
    return any(message_role == role and message_content == content for message_role, message_content in conversation)


def run_repro() -> ReproResult:
    """Execute the minimal interrupt/follow-up repro and return its evidence."""

    artifacts_root = Path(__file__).resolve().parents[1] / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="twinr-interrupt-follow-up-repro-",
        dir=artifacts_root,
    ) as temp_dir:
        config = _build_config(Path(temp_dir))
        lines: list[str] = []
        backend = _FakeCombinedBackend()
        recorder = _FakeRecorder()
        player = _FakePlayer()
        session = _InterruptingRealtimeSession()
        loop = TwinrRealtimeHardwareLoop(
            config=config,
            runtime=TwinrRuntime(config=config),
            realtime_session=session,
            print_backend=backend,
            button_monitor=_PassiveMonitor(),
            recorder=recorder,
            player=player,
            printer=_FakePrinter(),
            camera=_FakeCamera(),
            usage_store=_FakeUsageStore(),
            voice_profile_monitor=_FakeVoiceProfileMonitor(),
            ambient_audio_sampler=_FakeAmbientAudioSampler(),
            proactive_monitor=_PassiveMonitor(),
            emit=lines.append,
            sleep=lambda _seconds: None,
            error_reset_seconds=0.0,
        )
        try:
            loop.handle_button_press("green")
            return ReproResult(
                session_call_count=len(session.calls),
                recorder_start_timeouts=list(recorder.start_timeouts),
                interrupt_transcribe_calls=len(backend.transcribe_calls),
                assistant_response_omitted_after_interrupt=(
                    "assistant_response_omitted_after_interrupt=true" in lines
                ),
                user_interrupt_detected=("user_interrupt_detected=true" in lines),
                assistant_interrupted=("assistant_interrupted=true" in lines),
                second_conversation=session.conversations[1] if len(session.conversations) > 1 else None,
                runtime_conversation=loop.runtime.conversation_context(),
                tail_lines=list(lines[-20:]),
            )
        finally:
            loop.close(timeout_s=0.2)


def assert_repro_result(result: ReproResult) -> None:
    """Verify that the harness proved the intended interrupt/follow-up contract."""

    if result.session_call_count != 2:
        raise AssertionError(f"expected 2 realtime calls, got {result.session_call_count}")
    if result.recorder_start_timeouts != [8.0, 3.5]:
        raise AssertionError(
            f"expected capture timeouts [8.0, 3.5], got {result.recorder_start_timeouts!r}"
        )
    if result.interrupt_transcribe_calls < 1:
        raise AssertionError("expected at least one interrupt STT transcription call")
    if not result.user_interrupt_detected:
        raise AssertionError("interrupt watcher did not emit user_interrupt_detected=true")
    if not result.assistant_interrupted:
        raise AssertionError("loop did not emit assistant_interrupted=true")
    if not result.assistant_response_omitted_after_interrupt:
        raise AssertionError("interrupted assistant response was not omitted before follow-up")
    if not _conversation_has_message(
        result.second_conversation,
        role="user",
        content=_FIRST_TRANSCRIPT,
    ):
        raise AssertionError("follow-up conversation is missing the persisted first user turn")
    if _conversation_has_message(
        result.second_conversation,
        role="assistant",
        content=_DISCARDED_ASSISTANT_RESPONSE,
    ):
        raise AssertionError("follow-up conversation still contains the discarded assistant reply")
    if not _conversation_has_message(
        result.runtime_conversation,
        role="user",
        content=_FIRST_TRANSCRIPT,
    ):
        raise AssertionError("runtime conversation is missing the interrupted first user turn")
    if not _conversation_has_message(
        result.runtime_conversation,
        role="user",
        content=_FOLLOW_UP_TRANSCRIPT,
    ):
        raise AssertionError("runtime conversation is missing the follow-up user turn")
    if not _conversation_has_message(
        result.runtime_conversation,
        role="assistant",
        content=_FOLLOW_UP_RESPONSE,
    ):
        raise AssertionError("runtime conversation is missing the follow-up assistant reply")
    if _conversation_has_message(
        result.runtime_conversation,
        role="assistant",
        content=_DISCARDED_ASSISTANT_RESPONSE,
    ):
        raise AssertionError("runtime conversation still contains the discarded assistant reply")


def _format_text_report(result: ReproResult) -> str:
    """Render a concise operator-facing report for direct Pi execution."""

    return "\n".join(
        (
            "interrupt_follow_up_repro=pass",
            f"session_call_count={result.session_call_count}",
            f"recorder_start_timeouts={result.recorder_start_timeouts!r}",
            f"interrupt_transcribe_calls={result.interrupt_transcribe_calls}",
            f"user_interrupt_detected={str(result.user_interrupt_detected).lower()}",
            f"assistant_interrupted={str(result.assistant_interrupted).lower()}",
            "second_conversation="
            + json.dumps(result.second_conversation, ensure_ascii=True),
            "runtime_conversation="
            + json.dumps(result.runtime_conversation, ensure_ascii=True),
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Run the harness directly and return a shell-friendly exit code."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit the full repro result as JSON.")
    args = parser.parse_args(argv)

    result = run_repro()
    assert_repro_result(result)
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=True, indent=2, sort_keys=True))
    else:
        print(_format_text_report(result))
    return 0


class InterruptFollowUpReproTests(unittest.TestCase):
    """Keep the tiny Pi harness executable under pytest as well."""

    def test_repro_proves_interrupt_user_turn_persists_without_discarded_reply(self) -> None:
        """Run the standalone harness and verify its proof payload."""

        result = run_repro()
        assert_repro_result(result)


if __name__ == "__main__":
    raise SystemExit(main())
