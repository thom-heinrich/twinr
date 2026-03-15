from __future__ import annotations

import mimetypes
import os
import stat
import time
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, RLock, Thread, current_thread
from typing import Callable

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.conversation_closure import ToolCallingConversationClosureEvaluator
from twinr.agent.base_agent.turn_controller import ToolCallingTurnDecisionEvaluator
from twinr.agent.workflows.working_feedback import WorkingFeedbackKind, start_working_feedback_loop
from twinr.providers.openai import OpenAIImageInput

_SEARCH_FEEDBACK_TONE_PATTERN: tuple[tuple[int, int], ...] = (
    (784, 90),
    (1175, 70),
    (988, 80),
    (1318, 60),
)
_ALLOWED_REFERENCE_IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
)
_DEFAULT_REFERENCE_IMAGE_MAX_BYTES = 10 * 1024 * 1024
_DEFAULT_TTS_QUEUE_MAX_CHUNKS = 16
_DEFAULT_TTS_FIRST_CHUNK_TIMEOUT_SECONDS = 20.0
_DEFAULT_TTS_STREAM_CHUNK_TIMEOUT_SECONDS = 15.0
_DEFAULT_STOP_JOIN_TIMEOUT_SECONDS = 2.0
_NO_SPEECH_TIMEOUT_MARKERS: tuple[str, ...] = (
    "no speech detected before timeout",
    "no speech detected",
    "speech timeout",
    "timeout waiting for speech",
    "timeout waiting for user speech",
    "no input audio received",
)


def _default_emit(line: str) -> None:
    print(line, flush=True)


class TwinrRealtimeSupportMixin:
    # AUDIT-FIX(#4,#5,#10): Lazily create missing locks so mixin methods stay safe under concurrent use.
    def _get_lock(self, name: str) -> RLock:
        lock = getattr(self, name, None)
        if lock is None:
            lock = RLock()
            setattr(self, name, lock)
        return lock

    # AUDIT-FIX(#4): Sanitize exception text before emitting it outside the process boundary.
    def _safe_error_text(self, exc: BaseException) -> str:
        message = " ".join(str(exc).split()).strip()
        if not message:
            message = exc.__class__.__name__
        lower_message = message.casefold()
        if any(
            marker in lower_message
            for marker in ("api_key", "authorization", "bearer ", "token=", "password=", "secret=")
        ):
            message = "internal error"
        if len(message) > 240:
            message = f"{message[:237]}..."
        if message == exc.__class__.__name__:
            return message
        return f"{exc.__class__.__name__}: {message}"

    # AUDIT-FIX(#4): Error reporting must not throw while handling another failure.
    def _try_emit(self, line: str) -> None:
        try:
            self.emit(line)
        except Exception:
            _default_emit(line)

    # AUDIT-FIX(#2): Resolve the parent directory strictly while deferring the final component to O_NOFOLLOW open().
    def _normalize_reference_image_path(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.parent.resolve(strict=True) / path.name

    # AUDIT-FIX(#2): Optional base-dir enforcement keeps reference media inside an explicit safe area when configured.
    def _validate_reference_image_base_dir(self, path: Path) -> bool:
        base_dir_raw = (getattr(self.config, "vision_reference_image_base_dir", "") or "").strip()
        if not base_dir_raw:
            return True
        try:
            base_dir = Path(base_dir_raw).expanduser().resolve(strict=True)
        except OSError:
            self._try_emit("vision_reference_rejected=invalid_base_dir")
            return False
        try:
            path.relative_to(base_dir)
        except ValueError:
            self._try_emit(f"vision_reference_rejected=outside_base_dir:{path.name}")
            return False
        return True

    # AUDIT-FIX(#2): Open reference images without following symlinks and reject oversized/non-regular files.
    def _safe_read_reference_image_bytes(self, path: Path, *, max_bytes: int) -> bytes:
        flags = os.O_RDONLY
        nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
        if nofollow_flag:
            flags |= nofollow_flag
        fd = os.open(path, flags)
        try:
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise OSError("Reference image must be a regular file")
            if file_stat.st_size > max_bytes:
                raise OSError(f"Reference image exceeds {max_bytes} bytes")
            with os.fdopen(fd, "rb", closefd=True) as handle:
                data = handle.read(max_bytes + 1)
            fd = -1
        finally:
            if fd >= 0:
                os.close(fd)
        if len(data) > max_bytes:
            raise OSError(f"Reference image exceeds {max_bytes} bytes")
        return data

    # AUDIT-FIX(#2,#7): Guess a safe content type and keep only a basename when passing files downstream.
    def _build_image_input(self, data: bytes, *, path: Path, label: str) -> OpenAIImageInput:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return OpenAIImageInput(
            data=data,
            content_type=content_type,
            filename=path.name,
            label=label,
        )

    # AUDIT-FIX(#1): Support both legacy and renamed cooldown config fields with a safe default.
    def _print_button_cooldown_seconds(self) -> float:
        raw_value = getattr(
            self.config,
            "print_button_cooldown_seconds",
            getattr(self.config, "print_button_cooldown_s", 0.0),
        )
        try:
            return max(0.0, float(raw_value))
        except (TypeError, ValueError):
            return 0.0

    # AUDIT-FIX(#6): Bound stream buffering on the Raspberry Pi and validate provider output before playback.
    def _coerce_audio_chunk(self, chunk: object) -> bytes:
        if isinstance(chunk, bytes):
            return chunk
        if isinstance(chunk, bytearray):
            return bytes(chunk)
        if isinstance(chunk, memoryview):
            return chunk.tobytes()
        raise TypeError(f"TTS stream yielded unsupported chunk type: {type(chunk).__name__}")

    # AUDIT-FIX(#3): Centralize the provider list so config updates and rollbacks stay consistent.
    def _iter_config_targets(self) -> tuple[object, ...]:
        seen: set[int] = set()
        targets: list[object] = []
        for name in (
            "stt_provider",
            "agent_provider",
            "tts_provider",
            "print_backend",
            "tool_agent_provider",
            "turn_stt_provider",
            "turn_tool_agent_provider",
        ):
            provider = getattr(self, name, None)
            if provider is None:
                continue
            provider_id = id(provider)
            if provider_id in seen:
                continue
            seen.add(provider_id)
            targets.append(provider)
        return tuple(targets)

    # AUDIT-FIX(#3): Apply config only to targets that actually accept a config attribute.
    def _apply_config_to_targets(self, config: TwinrConfig) -> None:
        for provider in self._iter_config_targets():
            if hasattr(provider, "config"):
                provider.config = config
        session = getattr(self, "realtime_session", None)
        if session is not None and hasattr(session, "config"):
            session.config = config

    # AUDIT-FIX(#3): Build the turn evaluator from the new config without leaving stale state behind.
    def _build_turn_decision_evaluator(self, config: TwinrConfig) -> ToolCallingTurnDecisionEvaluator | None:
        turn_tool_agent_provider = getattr(self, "turn_tool_agent_provider", None)
        if turn_tool_agent_provider is None or not config.turn_controller_enabled:
            return None
        return ToolCallingTurnDecisionEvaluator(
            config=config,
            provider=turn_tool_agent_provider,
        )

    def _build_conversation_closure_evaluator(self, config: TwinrConfig) -> ToolCallingConversationClosureEvaluator | None:
        turn_tool_agent_provider = getattr(self, "turn_tool_agent_provider", None)
        if turn_tool_agent_provider is None or not config.conversation_closure_guard_enabled:
            return None
        return ToolCallingConversationClosureEvaluator(
            config=config,
            provider=turn_tool_agent_provider,
        )

    def _handle_error(self, exc: Exception) -> None:
        safe_error = self._safe_error_text(exc)  # AUDIT-FIX(#4): Never emit unsanitized exception text.
        try:
            self.runtime.fail(safe_error)
        except Exception as runtime_exc:  # AUDIT-FIX(#4): Preserve the original failure even if error-state persistence fails.
            _default_emit(f"runtime_fail_error={self._safe_error_text(runtime_exc)}")
        self._emit_status(force=True)
        self._try_emit(f"error={safe_error}")  # AUDIT-FIX(#4): Emission is best-effort only.
        sleep_seconds = max(0.0, float(getattr(self, "error_reset_seconds", 0.0) or 0.0))
        if sleep_seconds > 0:
            try:
                self.sleep(sleep_seconds)
            except Exception as sleep_exc:  # AUDIT-FIX(#4): A broken sleep implementation must not trap the device in error handling.
                _default_emit(f"error_reset_sleep_error={self._safe_error_text(sleep_exc)}")
        try:
            self.runtime.reset_error()
        except Exception as runtime_exc:  # AUDIT-FIX(#4): Failing to clear the error state should not raise a second exception here.
            _default_emit(f"runtime_reset_error={self._safe_error_text(runtime_exc)}")
        self._emit_status(force=True)

    def _emit_status(self, *, force: bool = False) -> None:
        status = getattr(getattr(self.runtime, "status", None), "value", "unknown")  # AUDIT-FIX(#9): Guard the first emit when _last_status is unset or runtime is partially initialised.
        if force or status != getattr(self, "_last_status", None):
            self._try_emit(f"status={status}")  # AUDIT-FIX(#4): Status reporting must not crash the main flow.
            self._last_status = status

    def _reload_live_config_from_env(self, env_path: Path) -> None:
        config_lock = self._get_lock("_config_lock")  # AUDIT-FIX(#3): Serialise live config reloads and rollbacks.
        with config_lock:
            previous_config = getattr(self, "config", None)
            previous_evaluator = getattr(self, "turn_decision_evaluator", None)
            previous_closure_evaluator = getattr(self, "conversation_closure_evaluator", None)
            try:
                updated_config = TwinrConfig.from_env(env_path)
                self.runtime.apply_live_config(updated_config)
                self.config = updated_config
                if hasattr(self, "_current_turn_audio_sample_rate"):
                    self._current_turn_audio_sample_rate = updated_config.openai_realtime_input_sample_rate
                self._apply_config_to_targets(updated_config)
                if hasattr(self, "turn_decision_evaluator"):
                    self.turn_decision_evaluator = self._build_turn_decision_evaluator(updated_config)
                if hasattr(self, "conversation_closure_evaluator"):
                    self.conversation_closure_evaluator = self._build_conversation_closure_evaluator(updated_config)
            except Exception as exc:
                if previous_config is not None:
                    try:
                        self.runtime.apply_live_config(previous_config)
                        self.config = previous_config
                        if hasattr(self, "_current_turn_audio_sample_rate"):
                            self._current_turn_audio_sample_rate = previous_config.openai_realtime_input_sample_rate
                        self._apply_config_to_targets(previous_config)
                        if hasattr(self, "turn_decision_evaluator"):
                            self.turn_decision_evaluator = previous_evaluator
                        if hasattr(self, "conversation_closure_evaluator"):
                            self.conversation_closure_evaluator = previous_closure_evaluator
                    except Exception as rollback_exc:
                        _default_emit(f"config_reload_rollback_error={self._safe_error_text(rollback_exc)}")
                self._try_emit(f"config_reload_error={self._safe_error_text(exc)}")
                return

    def _record_event(self, event: str, message: str, *, level: str = "info", **data: object) -> None:
        try:
            self.runtime.ops_events.append(event=event, message=message, level=level, data=data)
        except Exception as exc:  # AUDIT-FIX(#8): Ops-event persistence failures must not break the active interaction.
            self._try_emit(f"ops_event_error={self._safe_error_text(exc)}")

    def _record_usage(
        self,
        *,
        request_kind: str,
        source: str,
        model: str | None,
        response_id: str | None,
        request_id: str | None,
        used_web_search: bool | None,
        token_usage,
        **metadata: object,
    ) -> None:
        try:
            self.usage_store.append(
                source=source,
                request_kind=request_kind,
                model=model,
                response_id=response_id,
                request_id=request_id,
                used_web_search=used_web_search,
                token_usage=token_usage,
                metadata=metadata,
            )
        except Exception as exc:  # AUDIT-FIX(#8): Usage-accounting write failures must degrade gracefully on flaky storage.
            self._try_emit(f"usage_store_error={self._safe_error_text(exc)}")

    def _update_voice_assessment_from_pcm(self, audio_pcm: bytes) -> None:
        config = self.config  # AUDIT-FIX(#3): Snapshot config during assessment to avoid mixed live-reload reads.
        try:
            assessment = self.voice_profile_monitor.assess_pcm16(
                audio_pcm,
                sample_rate=config.openai_realtime_input_sample_rate,
                channels=config.audio_channels,
            )
        except Exception as exc:
            self._try_emit(f"voice_profile_error={self._safe_error_text(exc)}")  # AUDIT-FIX(#4): Sanitise provider errors.
            return
        if not assessment.should_persist:
            return
        try:
            self.runtime.update_user_voice_assessment(
                status=assessment.status,
                confidence=assessment.confidence,
                checked_at=assessment.checked_at,
            )
        except Exception as exc:  # AUDIT-FIX(#8): State persistence is best-effort here.
            self._try_emit(f"voice_profile_persist_error={self._safe_error_text(exc)}")
            return
        self._try_emit(f"voice_profile_status={assessment.status}")
        if assessment.confidence is not None:
            self._try_emit(f"voice_profile_confidence={assessment.confidence:.2f}")

    def _play_listen_beep(self) -> None:
        config = self.config  # AUDIT-FIX(#3): Use one config snapshot per tone.
        try:
            with self._get_lock("_audio_lock"):  # AUDIT-FIX(#5): Serialize tone playback with all other speaker output.
                self.player.play_tone(
                    frequency_hz=config.audio_beep_frequency_hz,
                    duration_ms=config.audio_beep_duration_ms,
                    volume=config.audio_beep_volume,
                    sample_rate=config.openai_realtime_input_sample_rate,
                )
        except Exception as exc:
            self._try_emit(f"beep_error={self._safe_error_text(exc)}")
            return
        if config.audio_beep_settle_ms > 0:
            try:
                self.sleep(config.audio_beep_settle_ms / 1000.0)
            except Exception as exc:  # AUDIT-FIX(#4): Post-tone settling must not crash the interaction loop.
                self._try_emit(f"beep_settle_error={self._safe_error_text(exc)}")

    def _start_working_feedback_loop(self, kind: WorkingFeedbackKind) -> Callable[[], None]:
        feedback_lock = self._get_lock("_feedback_lock")  # AUDIT-FIX(#10): Coordinate stop/start mutations across threads.
        with feedback_lock:
            previous_stop = getattr(self, "_working_feedback_stop", None)
            previous_generation = int(getattr(self, "_working_feedback_generation", 0) or 0)
        if callable(previous_stop):
            try:
                previous_stop()
            except Exception as exc:  # AUDIT-FIX(#10): A broken previous stop callback must not block the new loop.
                self._try_emit(f"working_feedback_stop_error={self._safe_error_text(exc)}")
        config = self.config  # AUDIT-FIX(#3): Snapshot delay-related config for the new loop.
        try:
            stop = start_working_feedback_loop(
                self.player,
                kind=kind,
                sample_rate=config.openai_realtime_input_sample_rate,
                emit=self._try_emit,
                delay_override_ms=(
                    config.processing_feedback_delay_ms
                    if kind == "processing"
                    else None
                ),
            )
        except Exception as exc:  # AUDIT-FIX(#10): Feedback audio is optional and must not take down the turn.
            self._try_emit(f"working_feedback_error={self._safe_error_text(exc)}")
            return lambda: None
        generation = previous_generation + 1
        with feedback_lock:
            self._working_feedback_generation = generation
            self._working_feedback_stop = stop

        def stop_current() -> None:
            with feedback_lock:
                if getattr(self, "_working_feedback_generation", None) != generation:
                    return
                active_stop = getattr(self, "_working_feedback_stop", None)
                self._working_feedback_stop = None
            if callable(active_stop):
                try:
                    active_stop()
                except Exception as exc:  # AUDIT-FIX(#10): Feedback stop callbacks are best-effort.
                    self._try_emit(f"working_feedback_stop_error={self._safe_error_text(exc)}")

        return stop_current

    def _stop_working_feedback(self) -> None:
        feedback_lock = self._get_lock("_feedback_lock")  # AUDIT-FIX(#10): Prevent concurrent stop/start races.
        with feedback_lock:
            active_stop = getattr(self, "_working_feedback_stop", None)
            self._working_feedback_stop = None
        if callable(active_stop):
            try:
                active_stop()
            except Exception as exc:  # AUDIT-FIX(#10): Feedback stop callbacks are best-effort.
                self._try_emit(f"working_feedback_stop_error={self._safe_error_text(exc)}")

    def _start_search_feedback_loop(self) -> Callable[[], None]:
        config = self.config  # AUDIT-FIX(#3): Freeze timing/volume values for the lifetime of this loop.
        if not config.search_feedback_tones_enabled:
            return lambda: None

        feedback_lock = self._get_lock("_feedback_lock")  # AUDIT-FIX(#10): Ensure only one search-tone loop is active at once.
        with feedback_lock:
            previous_stop = getattr(self, "_search_feedback_stop", None)
            previous_generation = int(getattr(self, "_search_feedback_generation", 0) or 0)
        if callable(previous_stop):
            try:
                previous_stop()
            except Exception as exc:
                self._try_emit(f"search_feedback_stop_error={self._safe_error_text(exc)}")

        generation = previous_generation + 1
        stop_event = Event()
        delay_seconds = max(0.0, config.search_feedback_delay_ms / 1000.0)
        pause_seconds = max(0.12, config.search_feedback_pause_ms / 1000.0)
        join_timeout_seconds = max(
            0.1,
            float(getattr(config, "search_feedback_stop_join_timeout_seconds", _DEFAULT_STOP_JOIN_TIMEOUT_SECONDS)),
        )

        def worker() -> None:
            if stop_event.wait(delay_seconds):
                return
            while not stop_event.is_set():
                try:
                    for frequency_hz, duration_ms in _SEARCH_FEEDBACK_TONE_PATTERN:
                        if stop_event.is_set():
                            return
                        with self._get_lock("_audio_lock"):  # AUDIT-FIX(#5): Prevent speaker races with TTS/beeps.
                            self.player.play_tone(
                                frequency_hz=frequency_hz,
                                duration_ms=duration_ms,
                                volume=config.search_feedback_volume,
                                sample_rate=config.openai_realtime_input_sample_rate,
                            )
                        if stop_event.wait(0.03):
                            return
                except Exception as exc:
                    self._try_emit(f"search_feedback_error={self._safe_error_text(exc)}")
                    return
                if stop_event.wait(pause_seconds):
                    return

        thread = Thread(target=worker, daemon=True)
        thread.start()

        def stop() -> None:
            stop_event.set()
            if thread is current_thread():
                return
            thread.join(timeout=join_timeout_seconds)  # AUDIT-FIX(#5): Avoid indefinite hangs during stop/shutdown.
            if thread.is_alive():
                self._try_emit("search_feedback_warning=stop_timeout")
            with feedback_lock:
                if getattr(self, "_search_feedback_generation", None) == generation:
                    self._search_feedback_stop = None

        with feedback_lock:
            self._search_feedback_generation = generation
            self._search_feedback_stop = stop

        return stop

    def _play_streaming_tts_with_feedback(
        self,
        text: str,
        *,
        turn_started: float,
        should_stop: Callable[[], bool] | None = None,
    ) -> tuple[int, int | None]:
        config = self.config  # AUDIT-FIX(#3): Keep timing and queue limits consistent for this TTS turn.
        tts_started = time.monotonic()
        first_audio_at: list[float | None] = [None]
        queue_max_chunks = max(
            4,
            int(getattr(config, "tts_stream_queue_max_chunks", _DEFAULT_TTS_QUEUE_MAX_CHUNKS)),
        )
        first_chunk_timeout_seconds = max(
            1.0,
            float(
                getattr(
                    config,
                    "tts_first_chunk_timeout_seconds",
                    _DEFAULT_TTS_FIRST_CHUNK_TIMEOUT_SECONDS,
                )
            ),
        )
        chunk_timeout_seconds = max(
            1.0,
            float(
                getattr(
                    config,
                    "tts_stream_chunk_timeout_seconds",
                    _DEFAULT_TTS_STREAM_CHUNK_TIMEOUT_SECONDS,
                )
            ),
        )
        chunk_queue: Queue[bytes | Exception | object] = Queue(maxsize=queue_max_chunks)  # AUDIT-FIX(#6): Bound memory growth under slow playback.
        sentinel = object()
        producer_stop = Event()
        feedback_started = False
        stop_answering_feedback: Callable[[], None] = lambda: None

        def queue_put(item: bytes | Exception | object) -> bool:
            while not producer_stop.is_set():
                try:
                    chunk_queue.put(item, timeout=0.1)
                    return True
                except Full:
                    continue
            return False

        def synth_worker() -> None:
            try:
                for chunk in self.tts_provider.synthesize_stream(text):
                    if producer_stop.is_set():
                        return
                    chunk_bytes = self._coerce_audio_chunk(chunk)
                    if not chunk_bytes:
                        continue
                    if not queue_put(chunk_bytes):
                        return
            except Exception as exc:
                if not producer_stop.is_set():
                    queue_put(exc)
            finally:
                if not producer_stop.is_set():
                    queue_put(sentinel)

        synth_thread = Thread(target=synth_worker, daemon=True)
        synth_thread.start()
        first_chunk: bytes | None = None
        first_chunk_deadline = tts_started + first_chunk_timeout_seconds
        try:
            while first_chunk is None:
                if should_stop is not None and should_stop():
                    return int((time.monotonic() - tts_started) * 1000), None
                timeout_remaining = first_chunk_deadline - time.monotonic()
                if timeout_remaining <= 0:
                    raise TimeoutError("TTS stream timed out before first audio chunk")
                try:
                    item = chunk_queue.get(timeout=min(0.05, timeout_remaining))
                except Empty:
                    if not feedback_started:
                        feedback_started = True
                        stop_answering_feedback = self._start_working_feedback_loop("answering")
                    continue
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                first_chunk = item
            stop_answering_feedback()
            if first_chunk is None:
                raise RuntimeError("TTS stream ended without audio")

            def playback_chunks():
                if first_audio_at[0] is None:
                    first_audio_at[0] = time.monotonic()
                yield first_chunk
                while True:
                    if should_stop is not None and should_stop():
                        return
                    try:
                        item = chunk_queue.get(timeout=chunk_timeout_seconds)
                    except Empty:
                        raise TimeoutError("TTS stream stalled while waiting for audio chunk")
                    if item is sentinel:
                        return
                    if isinstance(item, Exception):
                        raise item
                    yield item

            with self._get_lock("_audio_lock"):
                self.player.play_wav_chunks(playback_chunks(), should_stop=should_stop)
        finally:
            producer_stop.set()  # AUDIT-FIX(#6): Let the worker thread exit if the consumer aborts early.
            stop_answering_feedback()
            synth_thread.join(timeout=_DEFAULT_STOP_JOIN_TIMEOUT_SECONDS)
            if synth_thread.is_alive():
                self._try_emit("tts_warning=synth_thread_still_running")

        tts_ms = int((time.monotonic() - tts_started) * 1000)
        if first_audio_at[0] is None:
            return tts_ms, None
        return tts_ms, int((first_audio_at[0] - turn_started) * 1000)

    def _build_vision_images(self) -> list[OpenAIImageInput]:
        capture_filename = f"camera-capture-{time.time_ns()}.png"  # AUDIT-FIX(#7): Avoid filename collisions and stale capture reuse.
        try:
            with self._get_lock("_camera_lock"):
                capture = self.camera.capture_photo(filename=capture_filename)
        except Exception as exc:
            self._try_emit(f"camera_error={self._safe_error_text(exc)}")  # AUDIT-FIX(#7): Camera failure should surface as a controlled turn error.
            raise RuntimeError("Camera capture failed") from exc
        self._try_emit(f"camera_device={capture.source_device}")
        self._try_emit(f"camera_input_format={capture.input_format or 'default'}")
        self._try_emit(f"camera_capture_bytes={len(capture.data)}")
        try:
            self.runtime.long_term_memory.enqueue_multimodal_evidence(
                event_name="camera_capture",
                modality="camera",
                source="camera_tool",
                message="Live camera frame captured for device interaction.",
                data={
                    "purpose": "vision_inspection",
                    "source_device": capture.source_device,
                    "input_format": capture.input_format or "default",
                },
            )
        except Exception as exc:  # AUDIT-FIX(#7,#8): Memory persistence is optional; vision should still continue without it.
            self._try_emit(f"camera_memory_error={self._safe_error_text(exc)}")
        capture_path = Path(getattr(capture, "filename", capture_filename)).name
        capture_content_type = capture.content_type or mimetypes.guess_type(capture_path)[0] or "application/octet-stream"
        images = [
            OpenAIImageInput(
                data=capture.data,
                content_type=capture_content_type,
                filename=capture_path,
                label="Image 1: live camera frame from the device.",
            )
        ]
        reference_image = self._load_reference_image()
        if reference_image is not None:
            images.append(reference_image)
        return images

    def _load_reference_image(self) -> OpenAIImageInput | None:
        raw_path = (getattr(self.config, "vision_reference_image_path", "") or "").strip()
        if not raw_path:
            return None
        try:
            path = self._normalize_reference_image_path(raw_path)
        except FileNotFoundError:
            self._try_emit(f"vision_reference_missing={Path(raw_path).name}")
            return None
        except OSError as exc:
            self._try_emit(f"vision_reference_error={self._safe_error_text(exc)}")  # AUDIT-FIX(#2): Reject invalid paths without leaking the full filesystem layout.
            return None
        if not self._validate_reference_image_base_dir(path):
            return None
        if path.suffix.casefold() not in _ALLOWED_REFERENCE_IMAGE_SUFFIXES:
            self._try_emit(f"vision_reference_rejected=unsupported_file_type:{path.name}")  # AUDIT-FIX(#2): Only permit known image formats.
            return None
        max_bytes = max(
            1024,
            int(getattr(self.config, "vision_reference_image_max_bytes", _DEFAULT_REFERENCE_IMAGE_MAX_BYTES)),
        )
        try:
            data = self._safe_read_reference_image_bytes(path, max_bytes=max_bytes)
        except FileNotFoundError:
            self._try_emit(f"vision_reference_missing={path.name}")
            return None
        except OSError as exc:
            self._try_emit(f"vision_reference_error={self._safe_error_text(exc)}")
            return None
        self._try_emit(f"vision_reference_image={path.name}")
        return self._build_image_input(
            data,
            path=path,
            label="Image 2: stored reference image of the main user. Use it only for person or identity comparison.",
        )

    def _build_vision_prompt(self, question: str, *, include_reference: bool) -> str:
        clean_question = question.strip()  # AUDIT-FIX(#7): Avoid sending accidental leading/trailing control whitespace to the model.
        if include_reference:
            return (
                "This request includes camera input. "
                "Image 1 is the current live camera frame from the device. "
                "Image 2 is a stored reference image of the main user. "
                "Use the reference image only when the user's question depends on whether the live image shows that user. "
                "If identity is uncertain, say that clearly. "
                "If the camera view is too unclear, tell the user how to position themselves or the object.\n\n"
                f"User request: {clean_question}"
            )
        return (
            "This request includes camera input. "
            "Image 1 is the current live camera frame from the device. "
            "Answer from what is actually visible. "
            "If the view is too unclear, tell the user how to position themselves or the object in front of the camera.\n\n"
            f"User request: {clean_question}"
        )

    def _is_no_speech_timeout(self, exc: Exception) -> bool:
        message = str(exc).casefold()  # AUDIT-FIX(#11): Provider errors drift over time; avoid brittle exact-match classification.
        return any(marker in message for marker in _NO_SPEECH_TIMEOUT_MARKERS)

    def _is_print_cooldown_active(self) -> bool:
        last_print_request_at = getattr(self, "_last_print_request_at", None)
        if last_print_request_at is None:
            return False
        try:
            elapsed_seconds = time.monotonic() - float(last_print_request_at)
        except (TypeError, ValueError):  # AUDIT-FIX(#1): Corrupt in-memory state should fail open, not crash the print path.
            return False
        return elapsed_seconds < self._print_button_cooldown_seconds()
