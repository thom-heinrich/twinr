from __future__ import annotations

from queue import Queue
from threading import Thread
from typing import Callable
import time

from twinr.config import TwinrConfig
from twinr.hardware.audio import SilenceDetectedRecorder, WaveAudioPlayer
from twinr.hardware.buttons import ButtonAction, configured_button_monitor
from twinr.hardware.printer import RawReceiptPrinter
from twinr.providers.openai_backend import OpenAIBackend
from twinr.runtime import TwinrRuntime

_WEB_SEARCH_KEYWORDS = (
    "latest",
    "current",
    "today",
    "tonight",
    "tomorrow",
    "yesterday",
    "news",
    "weather",
    "forecast",
    "temperature",
    "price",
    "prices",
    "stock",
    "stocks",
    "score",
    "scores",
    "schedule",
    "schedules",
    "standings",
    "traffic",
    "version",
    "release",
    "update",
    "updates",
    "recent",
    "recently",
    "breaking",
    "ceo",
    "president",
    "election",
    "exchange rate",
    "market",
)


def _default_emit(line: str) -> None:
    print(line, flush=True)


class TwinrHardwareLoop:
    def __init__(
        self,
        config: TwinrConfig,
        *,
        runtime: TwinrRuntime | None = None,
        backend: OpenAIBackend | None = None,
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
        self.backend = backend or OpenAIBackend(config=config)
        self.button_monitor = button_monitor or configured_button_monitor(config)
        self.recorder = recorder or SilenceDetectedRecorder.from_config(config)
        self.player = player or WaveAudioPlayer.from_config(config)
        self.printer = printer or RawReceiptPrinter.from_config(config)
        self.emit = emit or _default_emit
        self.sleep = sleep
        self.error_reset_seconds = error_reset_seconds
        self._last_status: str | None = None

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
        turn_started = time.monotonic()
        self.runtime.press_green_button()
        self._emit_status(force=True)

        capture_started = time.monotonic()
        audio_bytes = self.recorder.record_until_pause(pause_ms=self.config.speech_pause_ms)
        capture_ms = int((time.monotonic() - capture_started) * 1000)

        stt_started = time.monotonic()
        transcript = self.backend.transcribe(
            audio_bytes,
            filename="twinr-listen.wav",
            content_type="audio/wav",
        ).strip()
        stt_ms = int((time.monotonic() - stt_started) * 1000)
        if not transcript:
            raise RuntimeError("Speech-to-text returned an empty transcript")

        self.emit(f"transcript={transcript}")
        self.runtime.submit_transcript(transcript)
        self._emit_status(force=True)

        allow_web_search = self._should_use_web_search(transcript)
        llm_started = time.monotonic()
        spoken_segments: Queue[str | None] = Queue()
        tts_error: list[Exception] = []
        first_audio_at: list[float | None] = [None]
        answer_started = False
        pending_segment = ""
        worker_started = False

        def tts_worker() -> None:
            while True:
                segment = spoken_segments.get()
                if segment is None:
                    return
                try:
                    def mark_first_chunk() -> object:
                        for chunk in self.backend.synthesize_stream(segment):
                            if first_audio_at[0] is None:
                                first_audio_at[0] = time.monotonic()
                            yield chunk

                    self.player.play_wav_chunks(mark_first_chunk())
                except Exception as exc:
                    tts_error.append(exc)
                    return

        worker = Thread(target=tts_worker, daemon=True)
        worker.start()
        worker_started = True

        def queue_ready_segments(delta: str) -> None:
            nonlocal answer_started, pending_segment
            pending_segment += delta
            while True:
                boundary = self._segment_boundary(pending_segment)
                if boundary is None:
                    return
                segment = pending_segment[:boundary].strip()
                pending_segment = pending_segment[boundary:].lstrip()
                if not segment:
                    continue
                if not answer_started:
                    self.runtime.begin_answering()
                    self._emit_status(force=True)
                    answer_started = True
                spoken_segments.put(segment)

        try:
            response = self.backend.respond_streaming(
                transcript,
                conversation=self.runtime.conversation_context(),
                allow_web_search=allow_web_search,
                on_text_delta=queue_ready_segments,
            )
            llm_ms = int((time.monotonic() - llm_started) * 1000)
            answer = self.runtime.finalize_agent_turn(response.text)
            if pending_segment.strip():
                if not answer_started:
                    self.runtime.begin_answering()
                    self._emit_status(force=True)
                    answer_started = True
                spoken_segments.put(pending_segment.strip())
        finally:
            if worker_started:
                spoken_segments.put(None)
        tts_started = time.monotonic()
        if worker_started:
            worker.join()
        tts_ms = int((time.monotonic() - tts_started) * 1000)
        if tts_error:
            raise tts_error[0]
        if not answer_started:
            self.runtime.begin_answering()
            self._emit_status(force=True)
        self.emit(f"response={answer}")
        if response.response_id:
            self.emit(f"openai_response_id={response.response_id}")
        if response.request_id:
            self.emit(f"openai_request_id={response.request_id}")
        self.emit(f"openai_allow_web_search={str(allow_web_search).lower()}")
        self.emit(f"openai_used_web_search={str(response.used_web_search).lower()}")
        self.runtime.finish_speaking()
        self._emit_status(force=True)
        self.emit(f"timing_capture_ms={capture_ms}")
        self.emit(f"timing_stt_ms={stt_ms}")
        self.emit(f"timing_llm_ms={llm_ms}")
        self.emit(f"timing_tts_ms={tts_ms}")
        self.emit("timing_playback_ms=streamed")
        if first_audio_at[0] is not None:
            self.emit(f"timing_first_audio_ms={int((first_audio_at[0] - turn_started) * 1000)}")
        self.emit(f"timing_total_ms={int((time.monotonic() - turn_started) * 1000)}")

    def _handle_print_turn(self) -> None:
        response_to_print = self.runtime.press_yellow_button()
        self._emit_status(force=True)

        composed = self.backend.compose_print_job_with_metadata(
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

    def _should_use_web_search(self, transcript: str) -> bool:
        mode = self.config.conversation_web_search.strip().lower()
        if mode == "always":
            return True
        if mode == "never":
            return False
        normalized = transcript.lower()
        return any(keyword in normalized for keyword in _WEB_SEARCH_KEYWORDS)

    def _segment_boundary(self, text: str) -> int | None:
        for index, character in enumerate(text):
            if character in ".?!":
                return index + 1
        if len(text) >= 140:
            return len(text)
        return None
