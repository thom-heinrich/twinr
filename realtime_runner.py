from __future__ import annotations

from queue import Queue
from threading import Thread
from typing import Callable
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.runtime import TwinrRuntime
from twinr.hardware.audio import SilenceDetectedRecorder, WaveAudioPlayer
from twinr.hardware.buttons import ButtonAction, configured_button_monitor
from twinr.hardware.printer import RawReceiptPrinter
from twinr.provider.openai.backend import OpenAIBackend
from twinr.provider.openai.realtime import OpenAIRealtimeSession


def _default_emit(line: str) -> None:
    print(line, flush=True)


class TwinrRealtimeHardwareLoop:
    def __init__(
        self,
        config: TwinrConfig,
        *,
        runtime: TwinrRuntime | None = None,
        realtime_session: OpenAIRealtimeSession | None = None,
        print_backend: OpenAIBackend | None = None,
        button_monitor=None,
        recorder: SilenceDetectedRecorder | None = None,
        player: WaveAudioPlayer | None = None,
        printer: RawReceiptPrinter | None = None,
        emit: Callable[[str], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        error_reset_seconds: float = 1.0,
    ) -> None:
        self.config = config
        self.runtime = runtime or TwinrRuntime(config=config)
        self.print_backend = print_backend or OpenAIBackend(config=config)
        self.button_monitor = button_monitor or configured_button_monitor(config)
        self.recorder = recorder or SilenceDetectedRecorder(
            device=config.audio_input_device,
            sample_rate=config.openai_realtime_input_sample_rate,
            channels=config.audio_channels,
            chunk_ms=config.audio_chunk_ms,
            preroll_ms=config.audio_preroll_ms,
            speech_threshold=config.audio_speech_threshold,
            speech_start_chunks=config.audio_speech_start_chunks,
            start_timeout_s=config.audio_start_timeout_s,
            max_record_seconds=config.audio_max_record_seconds,
        )
        self.player = player or WaveAudioPlayer.from_config(config)
        self.printer = printer or RawReceiptPrinter.from_config(config)
        self.realtime_session = realtime_session or OpenAIRealtimeSession(
            config=config,
            tool_handlers={"print_receipt": self._handle_print_tool_call},
        )
        self.emit = emit or _default_emit
        self.sleep = sleep
        self.error_reset_seconds = error_reset_seconds
        self._last_status: str | None = None
        self._last_print_request_at: float | None = None

    def run(self, *, duration_s: float | None = None, poll_timeout: float = 0.25) -> int:
        started_at = time.monotonic()
        self._emit_status(force=True)
        with self.button_monitor as monitor:
            while True:
                if duration_s is not None and time.monotonic() - started_at >= duration_s:
                    return 0
                event = monitor.poll(timeout=poll_timeout)
                if event is None or event.action != ButtonAction.PRESSED:
                    continue
                self.emit(f"button={event.name}")
                self.handle_button_press(event.name)

    def handle_button_press(self, button_name: str) -> None:
        try:
            if button_name == "green":
                self._handle_green_turn()
                return
            if button_name == "yellow":
                self._handle_print_turn()
                return
            raise ValueError(f"Unsupported button: {button_name}")
        except Exception as exc:
            self._handle_error(exc)

    def _handle_green_turn(self) -> None:
        follow_up = False
        while True:
            if self._run_single_green_turn(follow_up=follow_up):
                if self.config.conversation_follow_up_enabled:
                    follow_up = True
                    continue
            return

    def _run_single_green_turn(self, *, follow_up: bool) -> bool:
        turn_started = time.monotonic()
        self._play_listen_beep()
        self.runtime.press_green_button()
        self._emit_status(force=True)

        capture_started = time.monotonic()
        try:
            audio_pcm = self.recorder.record_pcm_until_pause_with_options(
                pause_ms=self.config.speech_pause_ms,
                start_timeout_s=(
                    self.config.conversation_follow_up_timeout_s if follow_up else None
                ),
                speech_start_chunks=(
                    self.config.audio_follow_up_speech_start_chunks if follow_up else None
                ),
                ignore_initial_ms=(
                    self.config.audio_follow_up_ignore_ms if follow_up else 0
                ),
            )
        except RuntimeError as exc:
            if not self._is_no_speech_timeout(exc):
                raise
            self.runtime.cancel_listening()
            self._emit_status(force=True)
            if follow_up:
                self.emit("follow_up_timeout=true")
            else:
                self.emit("listen_timeout=true")
            return False
        capture_ms = int((time.monotonic() - capture_started) * 1000)

        self.runtime.submit_transcript("[voice input]")
        self._emit_status(force=True)

        audio_chunks: Queue[bytes | None] = Queue()
        playback_error: list[Exception] = []
        first_audio_at: list[float | None] = [None]
        answer_started = False

        def begin_answering() -> None:
            nonlocal answer_started
            if answer_started:
                return
            self.runtime.begin_answering()
            self._emit_status(force=True)
            answer_started = True

        def playback_generator():
            while True:
                chunk = audio_chunks.get()
                if chunk is None:
                    return
                yield chunk

        def playback_worker() -> None:
            try:
                self.player.play_pcm16_chunks(
                    playback_generator(),
                    sample_rate=self.config.openai_realtime_input_sample_rate,
                    channels=self.config.audio_channels,
                )
            except Exception as exc:
                playback_error.append(exc)

        worker = Thread(target=playback_worker, daemon=True)
        worker.start()

        def on_audio_chunk(chunk: bytes) -> None:
            begin_answering()
            if first_audio_at[0] is None:
                first_audio_at[0] = time.monotonic()
            audio_chunks.put(chunk)

        def on_output_text_delta(_delta: str) -> None:
            begin_answering()

        realtime_started = time.monotonic()
        try:
            with self.realtime_session:
                turn = self.realtime_session.run_audio_turn(
                    audio_pcm,
                    conversation=self.runtime.conversation_context(),
                    on_audio_chunk=on_audio_chunk,
                    on_output_text_delta=on_output_text_delta,
                )
        finally:
            audio_chunks.put(None)
        worker.join()
        realtime_ms = int((time.monotonic() - realtime_started) * 1000)
        if playback_error:
            raise playback_error[0]

        self.runtime.last_transcript = turn.transcript
        self.emit(f"transcript={turn.transcript}")
        if not answer_started:
            begin_answering()
        answer = self.runtime.finalize_agent_turn(turn.response_text)
        self.emit(f"response={answer}")
        if turn.response_id:
            self.emit(f"openai_response_id={turn.response_id}")
        self.runtime.finish_speaking()
        self._emit_status(force=True)
        self.emit(f"timing_capture_ms={capture_ms}")
        self.emit(f"timing_realtime_ms={realtime_ms}")
        self.emit("timing_playback_ms=streamed")
        if first_audio_at[0] is not None:
            self.emit(f"timing_first_audio_ms={int((first_audio_at[0] - turn_started) * 1000)}")
        self.emit(f"timing_total_ms={int((time.monotonic() - turn_started) * 1000)}")
        return True

    def _handle_print_turn(self) -> None:
        if self._is_print_cooldown_active():
            self.emit("print_skipped=cooldown")
            return
        response_to_print = self.runtime.press_yellow_button()
        self._emit_status(force=True)

        composed = self.print_backend.compose_print_job_with_metadata(
            conversation=self.runtime.conversation_context(),
            focus_hint=self.runtime.last_transcript,
            direct_text=response_to_print,
            request_source="button",
        )
        print_job = self.printer.print_text(composed.text)
        self.emit(f"print_text={composed.text}")
        if composed.response_id:
            self.emit(f"print_response_id={composed.response_id}")
        if print_job:
            self.emit(f"print_job={print_job}")

        self.runtime.finish_printing()
        self._emit_status(force=True)
        self._last_print_request_at = time.monotonic()

    def _handle_print_tool_call(self, arguments: dict[str, object]) -> dict[str, object]:
        focus_hint = str(arguments.get("focus_hint", "")).strip()
        direct_text = str(arguments.get("text", "")).strip()
        if not focus_hint and not direct_text:
            raise RuntimeError("print_receipt requires `focus_hint` or `text`")

        self.runtime.maybe_begin_tool_print()
        self._emit_status(force=True)

        composed = self.print_backend.compose_print_job_with_metadata(
            conversation=self.runtime.conversation_context(),
            focus_hint=focus_hint or None,
            direct_text=direct_text or None,
            request_source="tool",
        )
        print_job = self.printer.print_text(composed.text)
        self.emit("print_tool_call=true")
        self.emit(f"print_text={composed.text}")
        if print_job:
            self.emit(f"print_job={print_job}")
        self._last_print_request_at = time.monotonic()
        return {
            "status": "printed",
            "text": composed.text,
            "job": print_job,
        }

    def _handle_error(self, exc: Exception) -> None:
        self.runtime.fail(str(exc))
        self._emit_status(force=True)
        self.emit(f"error={exc}")
        if self.error_reset_seconds > 0:
            self.sleep(self.error_reset_seconds)
        self.runtime.reset_error()
        self._emit_status(force=True)

    def _emit_status(self, *, force: bool = False) -> None:
        status = self.runtime.status.value
        if force or status != self._last_status:
            self.emit(f"status={status}")
            self._last_status = status

    def _play_listen_beep(self) -> None:
        try:
            self.player.play_tone(
                frequency_hz=self.config.audio_beep_frequency_hz,
                duration_ms=self.config.audio_beep_duration_ms,
                volume=self.config.audio_beep_volume,
                sample_rate=self.config.openai_realtime_input_sample_rate,
            )
        except Exception as exc:
            self.emit(f"beep_error={exc}")
            return
        if self.config.audio_beep_settle_ms > 0:
            self.sleep(self.config.audio_beep_settle_ms / 1000.0)

    def _is_no_speech_timeout(self, exc: Exception) -> bool:
        return "No speech detected before timeout" in str(exc)

    def _is_print_cooldown_active(self) -> bool:
        if self._last_print_request_at is None:
            return False
        return (time.monotonic() - self._last_print_request_at) < self.config.print_button_cooldown_s
