from __future__ import annotations

from contextlib import ExitStack
from queue import Queue
from threading import Lock, Thread
from typing import Callable
import time

from twinr.agent.base_agent.adaptive_timing import AdaptiveListeningWindow
from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import (
    AgentTextProvider,
    CombinedSpeechAgentProvider,
    CompositeSpeechAgentProvider,
    SpeechToTextProvider,
    TextToSpeechProvider,
)
from twinr.agent.base_agent.runtime import TwinrRuntime
from twinr.agent.tools import RealtimeToolExecutor, bind_realtime_tool_handlers
from twinr.hardware.audio import SilenceDetectedRecorder, WaveAudioPlayer
from twinr.hardware.buttons import ButtonAction, configured_button_monitor
from twinr.hardware.camera import V4L2StillCamera
from twinr.hardware.printer import RawReceiptPrinter
from twinr.hardware.voice_profile import VoiceProfileMonitor
from twinr.ops import TwinrUsageStore
from twinr.proactive import SocialTriggerDecision, WakewordMatch, build_default_proactive_monitor
from twinr.providers.openai import OpenAIProviderBundle
from twinr.providers.openai.realtime import OpenAIRealtimeSession
from twinr.agent.workflows.realtime_runner_background import TwinrRealtimeBackgroundMixin
from twinr.agent.workflows.realtime_runner_support import TwinrRealtimeSupportMixin, _default_emit
from twinr.agent.workflows.realtime_runner_tools import TwinrRealtimeToolDelegatesMixin


class TwinrRealtimeHardwareLoop(
    TwinrRealtimeBackgroundMixin,
    TwinrRealtimeToolDelegatesMixin,
    TwinrRealtimeSupportMixin,
):
    def __init__(
        self,
        config: TwinrConfig,
        *,
        runtime: TwinrRuntime | None = None,
        realtime_session: OpenAIRealtimeSession | None = None,
        print_backend: CombinedSpeechAgentProvider | None = None,
        stt_provider: SpeechToTextProvider | None = None,
        agent_provider: AgentTextProvider | None = None,
        tts_provider: TextToSpeechProvider | None = None,
        button_monitor=None,
        recorder: SilenceDetectedRecorder | None = None,
        player: WaveAudioPlayer | None = None,
        printer: RawReceiptPrinter | None = None,
        camera: V4L2StillCamera | None = None,
        usage_store: TwinrUsageStore | None = None,
        voice_profile_monitor: VoiceProfileMonitor | None = None,
        proactive_monitor=None,
        emit: Callable[[str], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        error_reset_seconds: float = 1.0,
    ) -> None:
        self.config = config
        self.runtime = runtime or TwinrRuntime(config=config)
        openai_bundle: OpenAIProviderBundle | None = None
        if print_backend is None and (stt_provider is None or agent_provider is None or tts_provider is None):
            openai_bundle = OpenAIProviderBundle.from_config(config)
        self.stt_provider = stt_provider or print_backend or (openai_bundle.stt if openai_bundle is not None else None)
        self.agent_provider = agent_provider or print_backend or (openai_bundle.agent if openai_bundle is not None else None)
        self.tts_provider = tts_provider or print_backend or (openai_bundle.tts if openai_bundle is not None else None)
        if self.stt_provider is None or self.agent_provider is None or self.tts_provider is None:
            raise ValueError("TwinrRealtimeHardwareLoop requires STT, agent, and TTS providers")
        self.print_backend = print_backend or (
            openai_bundle.combined
            if openai_bundle is not None
            else CompositeSpeechAgentProvider(
                stt=self.stt_provider,
                agent=self.agent_provider,
                tts=self.tts_provider,
            )
        )
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
        self.camera = camera or V4L2StillCamera.from_config(config)
        self.usage_store = usage_store or TwinrUsageStore.from_config(config)
        self.voice_profile_monitor = voice_profile_monitor or VoiceProfileMonitor.from_config(config)
        self._camera_lock = Lock()
        self._audio_lock = Lock()
        self._current_turn_audio_pcm: bytes | None = None
        self.tool_executor = RealtimeToolExecutor(self)
        self.realtime_session = realtime_session or OpenAIRealtimeSession(
            config=config,
            tool_handlers=bind_realtime_tool_handlers(self.tool_executor),
        )
        self.emit = emit or _default_emit
        self.sleep = sleep
        self.error_reset_seconds = error_reset_seconds
        self._last_status: str | None = None
        self._last_print_request_at: float | None = None
        self._next_reminder_check_at: float = 0.0
        self._next_automation_check_at: float = 0.0
        self._conversation_session_active = False
        self._sensor_observation_queue: Queue[tuple[dict[str, object], tuple[str, ...]]] = Queue()
        self.proactive_monitor = proactive_monitor or build_default_proactive_monitor(
            config=config,
            runtime=self.runtime,
            backend=self.print_backend,
            camera=self.camera,
            camera_lock=self._camera_lock,
            audio_lock=self._audio_lock,
            trigger_handler=self.handle_social_trigger,
            wakeword_handler=self.handle_wakeword_match,
            idle_predicate=self._background_work_allowed,
            observation_handler=self.handle_sensor_observation,
            emit=self.emit,
        )

    def run(self, *, duration_s: float | None = None, poll_timeout: float = 0.25) -> int:
        started_at = time.monotonic()
        self._emit_status(force=True)
        with ExitStack() as stack:
            monitor = stack.enter_context(self.button_monitor)
            if self.proactive_monitor is not None:
                stack.enter_context(self.proactive_monitor)
            while True:
                if duration_s is not None and time.monotonic() - started_at >= duration_s:
                    return 0
                event = monitor.poll(timeout=poll_timeout)
                if event is None:
                    if self._maybe_deliver_due_reminder():
                        continue
                    if self._maybe_run_due_automation():
                        continue
                    if self._maybe_run_sensor_automation():
                        continue
                    continue
                if event.action != ButtonAction.PRESSED:
                    continue
                self.emit(f"button={event.name}")
                self._record_event(
                    "button_pressed",
                    f"Physical button `{event.name}` was pressed.",
                    button=event.name,
                    line_offset=event.line_offset,
                )
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

    def handle_wakeword_match(self, match: WakewordMatch) -> bool:
        if not self._background_work_allowed():
            skip_reason = "busy" if self.runtime.status.value != "waiting" else "conversation_active"
            self.emit(f"wakeword_skipped={skip_reason}")
            self._record_event(
                "wakeword_skipped",
                "Wakeword was detected but Twinr was not idle enough to open a turn.",
                skip_reason=skip_reason,
                matched_phrase=match.matched_phrase,
            )
            return False
        self.emit(f"wakeword_phrase={match.matched_phrase or 'unknown'}")
        if match.remaining_text:
            self.emit("wakeword_mode=direct_text")
        else:
            self.emit("wakeword_mode=listen")
        self._record_event(
            "wakeword_detected",
            "Wakeword matched while Twinr was attentive.",
            matched_phrase=match.matched_phrase,
            transcript_preview=match.transcript[:160],
            remaining_text=match.remaining_text,
        )
        play_initial_beep = True
        if not match.remaining_text:
            self._acknowledge_wakeword()
            play_initial_beep = False
        self._run_conversation_session(
            initial_source="wakeword",
            seed_transcript=match.remaining_text or None,
            play_initial_beep=play_initial_beep,
        )
        return True

    def _handle_green_turn(self) -> None:
        self._run_conversation_session(initial_source="button")

    def _run_proactive_follow_up(self, trigger: SocialTriggerDecision) -> None:
        self.emit("proactive_listen=true")
        self._record_event(
            "proactive_listen_started",
            "Twinr opened a hands-free listening window after a proactive prompt.",
            trigger=trigger.trigger_id,
            timeout_s=self.config.conversation_follow_up_timeout_s,
        )
        self._run_conversation_session(
            initial_source="proactive",
            proactive_trigger=trigger.trigger_id,
        )

    def _run_conversation_session(
        self,
        *,
        initial_source: str,
        proactive_trigger: str | None = None,
        seed_transcript: str | None = None,
        play_initial_beep: bool = True,
    ) -> None:
        previous_session_state = self._conversation_session_active
        self._conversation_session_active = True
        try:
            follow_up = False
            if seed_transcript:
                if self._run_single_text_turn(
                    transcript=seed_transcript,
                    listen_source=initial_source,
                    proactive_trigger=proactive_trigger,
                ):
                    if self.config.conversation_follow_up_enabled:
                        follow_up = True
                    else:
                        return
                else:
                    return
            while True:
                listening_window = self.runtime.listening_window(
                    initial_source=initial_source,
                    follow_up=follow_up,
                )
                if self._run_single_audio_turn(
                    initial_source=initial_source,
                    follow_up=follow_up,
                    listening_window=listening_window,
                    listen_source="follow_up" if follow_up else initial_source,
                    proactive_trigger=None if follow_up else proactive_trigger,
                    speech_start_chunks=self._listening_window_speech_start_chunks(
                        initial_source=initial_source,
                        follow_up=follow_up,
                    ),
                    ignore_initial_ms=self._listening_window_ignore_initial_ms(
                        initial_source=initial_source,
                        follow_up=follow_up,
                    ),
                    timeout_emit_key=self._listening_timeout_emit_key(initial_source=initial_source, follow_up=follow_up),
                    timeout_message=self._listening_timeout_message(initial_source=initial_source, follow_up=follow_up),
                    play_initial_beep=True if follow_up else play_initial_beep,
                ):
                    if self.config.conversation_follow_up_enabled:
                        follow_up = True
                        continue
                return
        finally:
            self._conversation_session_active = previous_session_state

    def _run_single_audio_turn(
        self,
        *,
        initial_source: str,
        follow_up: bool,
        listening_window: AdaptiveListeningWindow,
        listen_source: str,
        proactive_trigger: str | None,
        speech_start_chunks: int | None,
        ignore_initial_ms: int,
        timeout_emit_key: str,
        timeout_message: str,
        play_initial_beep: bool,
    ) -> bool:
        turn_started = time.monotonic()
        if play_initial_beep:
            self._play_listen_beep()
        if listen_source == "button":
            self.runtime.press_green_button()
        else:
            self.runtime.begin_listening(
                request_source=listen_source,
                proactive_trigger=proactive_trigger,
            )
        self._emit_status(force=True)

        capture_started = time.monotonic()
        try:
            with self._audio_lock:
                capture_result = self.recorder.capture_pcm_until_pause_with_options(
                    pause_ms=listening_window.speech_pause_ms,
                    start_timeout_s=listening_window.start_timeout_s,
                    speech_start_chunks=speech_start_chunks,
                    ignore_initial_ms=ignore_initial_ms,
                    pause_grace_ms=listening_window.pause_grace_ms,
                )
        except RuntimeError as exc:
            if not self._is_no_speech_timeout(exc):
                raise
            self.runtime.remember_listen_timeout(
                initial_source=initial_source,
                follow_up=follow_up,
            )
            self.runtime.cancel_listening()
            self._emit_status(force=True)
            self.emit(f"{timeout_emit_key}=true")
            self._record_event("listen_timeout", timeout_message, request_source=listen_source)
            return False
        audio_pcm = capture_result.pcm_bytes
        self.runtime.remember_listen_capture(
            initial_source=initial_source,
            follow_up=follow_up,
            speech_started_after_ms=capture_result.speech_started_after_ms,
            resumed_after_pause_count=capture_result.resumed_after_pause_count,
        )
        capture_ms = int((time.monotonic() - capture_started) * 1000)
        self._update_voice_assessment_from_pcm(audio_pcm)
        realtime_started = time.monotonic()
        self._current_turn_audio_pcm = audio_pcm
        try:
            return self._complete_realtime_turn(
                transcript_seed="[voice input]",
                listen_source=listen_source,
                proactive_trigger=proactive_trigger,
                turn_started=turn_started,
                capture_ms=capture_ms,
                turn_runner=lambda on_audio_chunk, on_output_text_delta: self.realtime_session.run_audio_turn(
                    audio_pcm,
                    conversation=self.runtime.provider_conversation_context(),
                    on_audio_chunk=on_audio_chunk,
                    on_output_text_delta=on_output_text_delta,
                ),
                realtime_started=realtime_started,
            )
        finally:
            self._current_turn_audio_pcm = None

    def _run_single_text_turn(
        self,
        *,
        transcript: str,
        listen_source: str,
        proactive_trigger: str | None,
    ) -> bool:
        turn_started = time.monotonic()
        self.runtime.begin_listening(
            request_source=listen_source,
            proactive_trigger=proactive_trigger,
        )
        self._emit_status(force=True)
        realtime_started = time.monotonic()
        return self._complete_realtime_turn(
            transcript_seed=transcript,
            listen_source=listen_source,
            proactive_trigger=proactive_trigger,
            turn_started=turn_started,
            capture_ms=0,
            turn_runner=lambda on_audio_chunk, on_output_text_delta: self.realtime_session.run_text_turn(
                transcript,
                conversation=self.runtime.provider_conversation_context(),
                on_audio_chunk=on_audio_chunk,
                on_output_text_delta=on_output_text_delta,
            ),
            realtime_started=realtime_started,
        )

    def _complete_realtime_turn(
        self,
        *,
        transcript_seed: str,
        listen_source: str,
        proactive_trigger: str | None,
        turn_started: float,
        capture_ms: int,
        turn_runner,
        realtime_started: float,
    ) -> bool:
        self.runtime.submit_transcript(transcript_seed)
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

        try:
            with self.realtime_session:
                turn = turn_runner(on_audio_chunk, on_output_text_delta)
        finally:
            audio_chunks.put(None)
        worker.join()
        realtime_ms = int((time.monotonic() - realtime_started) * 1000)
        if playback_error:
            raise playback_error[0]

        final_transcript = turn.transcript or transcript_seed
        self.runtime.last_transcript = final_transcript
        self.emit(f"transcript={final_transcript}")
        if not answer_started:
            begin_answering()
        answer = self.runtime.finalize_agent_turn(turn.response_text)
        self.emit(f"response={answer}")
        if turn.response_id:
            self.emit(f"openai_response_id={turn.response_id}")
        self._record_usage(
            request_kind="realtime_conversation",
            source="realtime_loop",
            model=turn.model or self.config.openai_realtime_model,
            response_id=turn.response_id,
            request_id=None,
            used_web_search=None,
            token_usage=turn.token_usage,
            transcript=final_transcript,
            request_source=listen_source,
            proactive_trigger=proactive_trigger,
        )
        self.runtime.finish_speaking()
        self._emit_status(force=True)
        if turn.end_conversation:
            self.emit("conversation_ended=true")
        self.emit(f"timing_capture_ms={capture_ms}")
        self.emit(f"timing_realtime_ms={realtime_ms}")
        self.emit("timing_playback_ms=streamed")
        if first_audio_at[0] is not None:
            self.emit(f"timing_first_audio_ms={int((first_audio_at[0] - turn_started) * 1000)}")
        self.emit(f"timing_total_ms={int((time.monotonic() - turn_started) * 1000)}")
        return not turn.end_conversation

    def _acknowledge_wakeword(self) -> None:
        prompt = self.runtime.begin_wakeword_prompt("Ja?")
        self._emit_status(force=True)
        self._play_listen_beep()
        tts_started = time.monotonic()
        with self._audio_lock:
            self.player.play_wav_chunks(self.tts_provider.synthesize_stream(prompt))
        self.runtime.finish_speaking()
        self._emit_status(force=True)
        self.emit(f"wakeword_ack={prompt}")
        self.emit(f"timing_wakeword_ack_tts_ms={int((time.monotonic() - tts_started) * 1000)}")
        self._record_event(
            "wakeword_acknowledged",
            "Twinr confirmed a wakeword before opening hands-free listening.",
            prompt=prompt,
        )

    def _listening_window_speech_start_chunks(self, *, initial_source: str, follow_up: bool) -> int | None:
        if initial_source == "button" and not follow_up:
            return None
        return self.config.audio_follow_up_speech_start_chunks

    def _listening_window_ignore_initial_ms(self, *, initial_source: str, follow_up: bool) -> int:
        if initial_source == "button" and not follow_up:
            return 0
        return self.config.audio_follow_up_ignore_ms

    def _listening_timeout_emit_key(self, *, initial_source: str, follow_up: bool) -> str:
        if follow_up:
            return "follow_up_timeout"
        if initial_source == "proactive":
            return "proactive_listen_timeout"
        if initial_source == "wakeword":
            return "wakeword_listen_timeout"
        return "listen_timeout"

    def _listening_timeout_message(self, *, initial_source: str, follow_up: bool) -> str:
        if follow_up:
            return "Follow-up listening window expired."
        if initial_source == "proactive":
            return "Hands-free listening window after a proactive prompt expired."
        if initial_source == "wakeword":
            return "Wakeword listening window expired."
        return "Listening timed out before speech started."

    def _handle_print_turn(self) -> None:
        if self._is_print_cooldown_active():
            self.emit("print_skipped=cooldown")
            self._record_event("print_skipped", "Print request ignored because cooldown is active.")
            return
        response_to_print = self.runtime.press_yellow_button()
        self._emit_status(force=True)

        composed = self.agent_provider.compose_print_job_with_metadata(
            conversation=self.runtime.provider_conversation_context(),
            focus_hint=self.runtime.last_transcript,
            direct_text=response_to_print,
            request_source="button",
        )
        print_job = self.printer.print_text(composed.text)
        self.emit(f"print_text={composed.text}")
        if composed.response_id:
            self.emit(f"print_response_id={composed.response_id}")
        self._record_usage(
            request_kind="print",
            source="realtime_loop",
            model=composed.model,
            response_id=composed.response_id,
            request_id=composed.request_id,
            used_web_search=False,
            token_usage=composed.token_usage,
            request_source="button",
        )
        if print_job:
            self.emit(f"print_job={print_job}")
        self._record_event(
            "print_job_sent",
            "Print job was sent to the configured printer.",
            queue=self.config.printer_queue,
            job=print_job,
        )

        self.runtime.finish_printing()
        self._emit_status(force=True)
        self._last_print_request_at = time.monotonic()
