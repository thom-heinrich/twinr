from __future__ import annotations

from pathlib import Path
from queue import Queue
from threading import Thread
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import (
    ToolCallingAgentProvider,
)
from twinr.agent.tools import (
    ToolCallingStreamingLoop,
    build_agent_tool_schemas,
    build_compact_agent_tool_schemas,
    build_compact_tool_agent_instructions,
    build_tool_agent_instructions,
    bind_realtime_tool_handlers,
    realtime_tool_names,
)
from twinr.agent.workflows.realtime_runner import TwinrRealtimeHardwareLoop
from twinr.hardware.audio import pcm16_to_wav_bytes
from twinr.providers.factory import build_streaming_provider_bundle


class _StreamingSessionPlaceholder:
    def __init__(self, config: TwinrConfig) -> None:
        self.config = config


class TwinrStreamingHardwareLoop(TwinrRealtimeHardwareLoop):
    def __init__(
        self,
        config: TwinrConfig,
        *,
        tool_agent_provider: ToolCallingAgentProvider | None = None,
        streaming_turn_loop: ToolCallingStreamingLoop | None = None,
        **kwargs,
    ) -> None:
        if (
            tool_agent_provider is None
            and kwargs.get("print_backend") is None
            and (
                kwargs.get("stt_provider") is None
                or kwargs.get("agent_provider") is None
                or kwargs.get("tts_provider") is None
            )
        ):
            provider_bundle = build_streaming_provider_bundle(config)
            kwargs.setdefault("print_backend", provider_bundle.print_backend)
            kwargs.setdefault("stt_provider", provider_bundle.stt)
            kwargs.setdefault("agent_provider", provider_bundle.agent)
            kwargs.setdefault("tts_provider", provider_bundle.tts)
            tool_agent_provider = provider_bundle.tool_agent
        resolved_tool_agent = tool_agent_provider
        if resolved_tool_agent is None:
            raise ValueError("TwinrStreamingHardwareLoop requires a tool-capable agent provider")

        super().__init__(
            config,
            realtime_session=_StreamingSessionPlaceholder(config),
            **kwargs,
        )
        self.tool_agent_provider = resolved_tool_agent
        self._tool_handlers = bind_realtime_tool_handlers(self.tool_executor)
        tool_schemas = (
            build_compact_agent_tool_schemas(realtime_tool_names())
            if (self.config.llm_provider or "").strip().lower() == "groq"
            else build_agent_tool_schemas(realtime_tool_names())
        )
        self.streaming_turn_loop = streaming_turn_loop or ToolCallingStreamingLoop(
            provider=self.tool_agent_provider,
            tool_handlers=self._tool_handlers,
            tool_schemas=tool_schemas,
            stream_final_only=((self.config.llm_provider or "").strip().lower() == "groq"),
        )

    def _reload_live_config_from_env(self, env_path: Path) -> None:
        updated_config = TwinrConfig.from_env(env_path)
        self.config = updated_config
        self.runtime.apply_live_config(updated_config)
        seen: set[int] = set()
        for provider in (
            self.stt_provider,
            self.agent_provider,
            self.tts_provider,
            self.print_backend,
            self.tool_agent_provider,
        ):
            provider_id = id(provider)
            if provider_id in seen:
                continue
            seen.add(provider_id)
            provider.config = updated_config
        self.realtime_session.config = updated_config

    def _run_single_audio_turn(
        self,
        *,
        initial_source: str,
        follow_up: bool,
        listening_window,
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
        audio_bytes = pcm16_to_wav_bytes(
            audio_pcm,
            sample_rate=self.config.openai_realtime_input_sample_rate,
            channels=self.config.audio_channels,
        )

        self._current_turn_audio_pcm = audio_pcm
        try:
            stt_started = time.monotonic()
            transcript = self.stt_provider.transcribe(
                audio_bytes,
                filename="twinr-streaming-listen.wav",
                content_type="audio/wav",
            ).strip()
            stt_ms = int((time.monotonic() - stt_started) * 1000)
            if not transcript:
                raise RuntimeError("Speech-to-text returned an empty transcript")
            self.emit(f"transcript={transcript}")
            return self._complete_streaming_turn(
                transcript=transcript,
                listen_source=listen_source,
                proactive_trigger=proactive_trigger,
                turn_started=turn_started,
                capture_ms=capture_ms,
                stt_ms=stt_ms,
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
        return self._complete_streaming_turn(
            transcript=transcript,
            listen_source=listen_source,
            proactive_trigger=proactive_trigger,
            turn_started=turn_started,
            capture_ms=0,
            stt_ms=0,
        )

    def _complete_streaming_turn(
        self,
        *,
        transcript: str,
        listen_source: str,
        proactive_trigger: str | None,
        turn_started: float,
        capture_ms: int,
        stt_ms: int,
    ) -> bool:
        self.runtime.submit_transcript(transcript)
        self._emit_status(force=True)

        spoken_segments: Queue[str | None] = Queue()
        tts_error: list[Exception] = []
        first_audio_at: list[float | None] = [None]
        answer_started = False
        pending_segment = ""

        def tts_worker() -> None:
            while True:
                segment = spoken_segments.get()
                if segment is None:
                    return
                try:
                    def mark_first_chunk():
                        for chunk in self.tts_provider.synthesize_stream(segment):
                            if first_audio_at[0] is None:
                                first_audio_at[0] = time.monotonic()
                            yield chunk

                    self.player.play_wav_chunks(mark_first_chunk())
                except Exception as exc:
                    tts_error.append(exc)
                    return

        worker = Thread(target=tts_worker, daemon=True)
        worker.start()

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

        llm_started = time.monotonic()
        try:
            response = self.streaming_turn_loop.run(
                transcript,
                conversation=self.runtime.provider_conversation_context(),
                instructions=(
                    build_compact_tool_agent_instructions(
                        self.config,
                        extra_instructions=self.config.openai_realtime_instructions,
                    )
                    if (self.config.llm_provider or "").strip().lower() == "groq"
                    else build_tool_agent_instructions(
                        self.config,
                        extra_instructions=self.config.openai_realtime_instructions,
                    )
                ),
                allow_web_search=False,
                on_text_delta=queue_ready_segments,
            )
            llm_ms = int((time.monotonic() - llm_started) * 1000)
            if not response.text.strip():
                raise RuntimeError("Streaming tool loop completed without text output")
            if self.runtime.status.value == "printing":
                self.runtime.resume_answering_after_print()
                self._emit_status(force=True)
                answer_started = True
            answer = self.runtime.finalize_agent_turn(response.text)
            if pending_segment.strip():
                if not answer_started:
                    self.runtime.begin_answering()
                    self._emit_status(force=True)
                    answer_started = True
                spoken_segments.put(pending_segment.strip())
        finally:
            spoken_segments.put(None)

        worker.join()
        if tts_error:
            raise tts_error[0]
        if not answer_started:
            self.runtime.begin_answering()
            self._emit_status(force=True)

        self.emit(f"response={answer}")
        if response.response_id:
            self.emit(f"agent_response_id={response.response_id}")
        if response.request_id:
            self.emit(f"agent_request_id={response.request_id}")
        self.emit(f"agent_tool_rounds={response.rounds}")
        self.emit(f"agent_tool_calls={len(response.tool_calls)}")
        self.emit(f"agent_used_web_search={str(response.used_web_search).lower()}")
        self._record_usage(
            request_kind="conversation",
            source="streaming_loop",
            model=response.model,
            response_id=response.response_id,
            request_id=response.request_id,
            used_web_search=response.used_web_search,
            token_usage=response.token_usage,
            transcript=transcript,
            request_source=listen_source,
            proactive_trigger=proactive_trigger,
            tool_rounds=response.rounds,
            tool_calls=len(response.tool_calls),
        )
        self.runtime.finish_speaking()
        self._emit_status(force=True)
        self.emit(f"timing_capture_ms={capture_ms}")
        self.emit(f"timing_stt_ms={stt_ms}")
        self.emit(f"timing_llm_ms={llm_ms}")
        self.emit("timing_playback_ms=streamed")
        if first_audio_at[0] is not None:
            self.emit(f"timing_first_audio_ms={int((first_audio_at[0] - turn_started) * 1000)}")
        self.emit(f"timing_total_ms={int((time.monotonic() - turn_started) * 1000)}")
        return not any(call.name == "end_conversation" for call in response.tool_calls)

    def _segment_boundary(self, text: str) -> int | None:
        for index, character in enumerate(text):
            if character in ".?!":
                return index + 1
        if len(text) >= 140:
            return len(text)
        return None
