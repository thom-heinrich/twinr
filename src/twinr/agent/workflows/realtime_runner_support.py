from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
from typing import Callable
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.providers.openai import OpenAIImageInput

_SEARCH_FEEDBACK_TONE_PATTERN: tuple[tuple[int, int], ...] = (
    (784, 90),
    (1175, 70),
    (988, 80),
    (1318, 60),
)


def _default_emit(line: str) -> None:
    print(line, flush=True)


class TwinrRealtimeSupportMixin:
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

    def _reload_live_config_from_env(self, env_path: Path) -> None:
        updated_config = TwinrConfig.from_env(env_path)
        self.config = updated_config
        self.runtime.apply_live_config(updated_config)
        seen: set[int] = set()
        for provider in (self.stt_provider, self.agent_provider, self.tts_provider, self.print_backend):
            provider_id = id(provider)
            if provider_id in seen:
                continue
            seen.add(provider_id)
            provider.config = updated_config
        self.realtime_session.config = updated_config

    def _record_event(self, event: str, message: str, *, level: str = "info", **data: object) -> None:
        self.runtime.ops_events.append(event=event, message=message, level=level, data=data)

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

    def _update_voice_assessment_from_pcm(self, audio_pcm: bytes) -> None:
        try:
            assessment = self.voice_profile_monitor.assess_pcm16(
                audio_pcm,
                sample_rate=self.config.openai_realtime_input_sample_rate,
                channels=self.config.audio_channels,
            )
        except Exception as exc:
            self.emit(f"voice_profile_error={exc}")
            return
        if not assessment.should_persist:
            return
        self.runtime.update_user_voice_assessment(
            status=assessment.status,
            confidence=assessment.confidence,
            checked_at=assessment.checked_at,
        )
        self.emit(f"voice_profile_status={assessment.status}")
        if assessment.confidence is not None:
            self.emit(f"voice_profile_confidence={assessment.confidence:.2f}")

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

    def _start_search_feedback_loop(self) -> Callable[[], None]:
        if not self.config.search_feedback_tones_enabled:
            return lambda: None

        stop_event = Event()
        delay_seconds = max(0.0, self.config.search_feedback_delay_ms / 1000.0)
        pause_seconds = max(0.12, self.config.search_feedback_pause_ms / 1000.0)

        def worker() -> None:
            if stop_event.wait(delay_seconds):
                return
            while not stop_event.is_set():
                try:
                    for frequency_hz, duration_ms in _SEARCH_FEEDBACK_TONE_PATTERN:
                        if stop_event.is_set():
                            return
                        self.player.play_tone(
                            frequency_hz=frequency_hz,
                            duration_ms=duration_ms,
                            volume=self.config.search_feedback_volume,
                            sample_rate=self.config.openai_realtime_input_sample_rate,
                        )
                        if stop_event.wait(0.03):
                            return
                except Exception as exc:
                    self.emit(f"search_feedback_error={exc}")
                    return
                if stop_event.wait(pause_seconds):
                    return

        thread = Thread(target=worker, daemon=True)
        thread.start()

        def stop() -> None:
            stop_event.set()
            thread.join()

        return stop

    def _build_vision_images(self) -> list[OpenAIImageInput]:
        with self._camera_lock:
            capture = self.camera.capture_photo(filename="camera-capture.png")
        self.emit(f"camera_device={capture.source_device}")
        self.emit(f"camera_input_format={capture.input_format or 'default'}")
        self.emit(f"camera_capture_bytes={len(capture.data)}")
        images = [
            OpenAIImageInput(
                data=capture.data,
                content_type=capture.content_type,
                filename=capture.filename,
                label="Image 1: live camera frame from the device.",
            )
        ]
        reference_image = self._load_reference_image()
        if reference_image is not None:
            images.append(reference_image)
        return images

    def _load_reference_image(self) -> OpenAIImageInput | None:
        raw_path = (self.config.vision_reference_image_path or "").strip()
        if not raw_path:
            return None
        path = Path(raw_path)
        if not path.exists():
            self.emit(f"vision_reference_missing={path}")
            return None
        self.emit(f"vision_reference_image={path}")
        return OpenAIImageInput.from_path(
            path,
            label="Image 2: stored reference image of the main user. Use it only for person or identity comparison.",
        )

    def _build_vision_prompt(self, question: str, *, include_reference: bool) -> str:
        if include_reference:
            return (
                "This request includes camera input. "
                "Image 1 is the current live camera frame from the device. "
                "Image 2 is a stored reference image of the main user. "
                "Use the reference image only when the user's question depends on whether the live image shows that user. "
                "If identity is uncertain, say that clearly. "
                "If the camera view is too unclear, tell the user how to position themselves or the object.\n\n"
                f"User request: {question}"
            )
        return (
            "This request includes camera input. "
            "Image 1 is the current live camera frame from the device. "
            "Answer from what is actually visible. "
            "If the view is too unclear, tell the user how to position themselves or the object in front of the camera.\n\n"
            f"User request: {question}"
        )

    def _is_no_speech_timeout(self, exc: Exception) -> bool:
        return "No speech detected before timeout" in str(exc)

    def _is_print_cooldown_active(self) -> bool:
        if self._last_print_request_at is None:
            return False
        return (time.monotonic() - self._last_print_request_at) < self.config.print_button_cooldown_s
